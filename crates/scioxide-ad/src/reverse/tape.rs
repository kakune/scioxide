//! The caller-owned tape and its capacity protocol.

use alloc::vec::Vec;

use scioxide_core::{
    Capacity, DomainReason, ExecutionContext, InvalidInputReason, NonFiniteOrigin, SciOxideError,
};

use super::call::Call;
use super::linearization::Linearization;
use super::recorder::{Counts, Frozen, MIXED, Recorder, WORD_LIMIT};
use super::variables::Variables;
use super::word::{self, MAX_WORDS};
use super::{TapeValue, Var, sweep};
use crate::{JacobianMode, KinkPolicy, Mask, NonFinitePolicy};

/// Options of a [`Tape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TapeOptions {
    non_finite: NonFinitePolicy,
    kink: KinkPolicy,
    jacobian: JacobianMode,
    byte_limit: Option<usize>,
    pub(crate) word_limit: usize,
}

impl Default for TapeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl TapeOptions {
    /// Returns the defaults: [`NonFinitePolicy::Propagate`], [`KinkPolicy::Midpoint`],
    /// [`JacobianMode::Auto`] and no byte limit.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            non_finite: NonFinitePolicy::Propagate,
            kink: KinkPolicy::Midpoint,
            jacobian: JacobianMode::Auto,
            byte_limit: None,
            word_limit: MAX_WORDS,
        }
    }
    /// Sets the treatment of non-finite points and results.
    #[must_use]
    pub const fn with_non_finite_policy(mut self, policy: NonFinitePolicy) -> Self {
        self.non_finite = policy;
        self
    }
    /// Sets what a kink hit does.
    #[must_use]
    pub const fn with_kink_policy(mut self, policy: KinkPolicy) -> Self {
        self.kink = policy;
        self
    }
    /// Sets the sweep direction of dense Jacobians.
    #[must_use]
    pub const fn with_jacobian_mode(mut self, mode: JacobianMode) -> Self {
        self.jacobian = mode;
        self
    }
    /// Limits the bytes the tape reserves, [`Tape::reserved_bytes`]; `None` removes
    /// the limit.
    ///
    /// Every growth (columns, variables, seeds and staging buffers) is admitted only
    /// if the bytes after it stay within the limit, otherwise the transform fails
    /// with `ResourceLimit` and the tape keeps its storage. The one exception is an
    /// output vector that the function grows by pushing more outputs than the tape
    /// reserved.
    #[must_use]
    pub const fn with_byte_limit(mut self, limit: Option<usize>) -> Self {
        self.byte_limit = limit;
        self
    }
    /// Returns the non-finite policy.
    #[must_use]
    pub const fn non_finite_policy(&self) -> NonFinitePolicy {
        self.non_finite
    }
    /// Returns the kink policy.
    #[must_use]
    pub const fn kink_policy(&self) -> KinkPolicy {
        self.kink
    }
    /// Returns the Jacobian mode.
    #[must_use]
    pub const fn jacobian_mode(&self) -> JacobianMode {
        self.jacobian
    }
    /// Returns the byte limit.
    #[must_use]
    pub const fn byte_limit(&self) -> Option<usize> {
        self.byte_limit
    }
    #[cfg(test)]
    pub(crate) const fn with_word_limit(mut self, limit: usize) -> Self {
        self.word_limit = limit;
        self
    }
}

/// The storage a recording needs: statement words, full and scalar partials, call
/// table entries, call scratch and variables (inputs plus outputs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TapeRequirements {
    words: usize,
    partials: usize,
    scalars: usize,
    calls: usize,
    scratch: usize,
    variables: usize,
}

impl TapeRequirements {
    /// Builds a requirement from its counts.
    #[must_use]
    pub const fn new(
        words: usize,
        partials: usize,
        scalars: usize,
        calls: usize,
        scratch: usize,
        variables: usize,
    ) -> Self {
        Self {
            words,
            partials,
            scalars,
            calls,
            scratch,
            variables,
        }
    }
    /// Returns the number of statement words.
    #[must_use]
    pub const fn words(&self) -> usize {
        self.words
    }
    /// Returns the number of full partials.
    #[must_use]
    pub const fn partials(&self) -> usize {
        self.partials
    }
    /// Returns the number of scalar partials.
    #[must_use]
    pub const fn scalars(&self) -> usize {
        self.scalars
    }
    /// Returns the number of call table entries.
    #[must_use]
    pub const fn calls(&self) -> usize {
        self.calls
    }
    /// Returns the values of call scratch: the largest evaluation of an external
    /// rule, solve or loop call, taken twice more for its sweeps.
    #[must_use]
    pub const fn scratch(&self) -> usize {
        self.scratch
    }
    /// Returns the number of variables.
    #[must_use]
    pub const fn variables(&self) -> usize {
        self.variables
    }
}

/// Facts about the last transform of a [`Tape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TapeReport {
    pub(crate) words: usize,
    pub(crate) reserved_bytes: usize,
    pub(crate) kinks: u64,
    pub(crate) recorded_twice: bool,
}

impl TapeReport {
    /// Returns the statement words of the last recording.
    #[must_use]
    pub const fn words(&self) -> usize {
        self.words
    }
    /// Returns the bytes the tape had reserved after the transform.
    #[must_use]
    pub const fn reserved_bytes(&self) -> usize {
        self.reserved_bytes
    }
    /// Returns the kink hits of the last recording: one per operation and lane in
    /// which an active argument sits on a kink.
    #[must_use]
    pub const fn kinks(&self) -> u64 {
        self.kinks
    }
    /// Returns whether the last transform recorded the function twice because the
    /// tape had to grow.
    #[must_use]
    pub const fn recorded_twice(&self) -> bool {
        self.recorded_twice
    }
}

/// A reverse-mode tape: plain vectors that hold one recording at a time.
///
/// A transform borrows the tape mutably, builds a recorder on its stack whose
/// columns are cell views of these vectors, hands [`Var`] inputs to the function,
/// and sweeps the statements it recorded. A recording that does not fit is not
/// stored, but its statements are still counted; the transform then grows the
/// vectors to the counted requirement and runs the function again, so a function
/// runs at most twice per transform call and must be deterministic. Recording
/// itself never allocates: warm transforms on a tape that has recorded the same
/// function allocate nothing. The tape is `Send + Sync`; give each thread its own.
///
/// `V` is a [`TapeValue`]: scalars for `Tape<f64>`, lane packs for
/// `Tape<Lanes<f64, 8>>`, and seeded duals for second-order products on
/// `Tape<Dual<f64>>`.
///
/// # Accuracy
///
/// A sweep applies the rules of the [crate documentation](crate#accuracy) in
/// reverse statement order and accumulates each adjoint from `+0`. The gradient of
/// one operation equals its [`Dual`](crate::Dual) derivative bit for bit, with two
/// exceptions: an exactly zero derivative is `+0`, and in the rules with IEEE
/// products an infinite or NaN partial times an exactly zero tangent is NaN in
/// `Dual` but zero in the sweep (`x·y` at `(∞, 2)` with tangent `(1, 0)`). Longer
/// recordings add the contributions to an adjoint in another order and agree to
/// rounding.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, Tape, TapeOptions};
/// use scioxide_core::ExecutionContext;
///
/// let mut tape = Tape::<f64>::new(TapeOptions::new());
/// let mut gradient = [0.0; 2];
/// for point in [[1.0, 2.0], [0.5, -1.0]] {
///     let value = tape.gradient_into(|x| Ok(x[0] * x[1].sin()), &point, &mut gradient, ExecutionContext::sequential())?;
///     assert_eq!(value, point[0] * point[1].sin());
///     assert!((gradient[0] - point[1].sin()).abs() < 1e-15);
///     assert!((gradient[1] - point[0] * point[1].cos()).abs() < 1e-15);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct Tape<V: TapeValue> {
    pub(crate) options: TapeOptions,
    pub(crate) words: Vec<u64>,
    pub(crate) full: Vec<V>,
    pub(crate) scalars: Vec<V::Scalar>,
    pub(crate) calls: Vec<Option<Call<V>>>,
    pub(crate) scratch: Vec<V>,
    pub(crate) work: Vec<V>,
    pub(crate) call_work: Vec<V>,
    pub(crate) clean: bool,
    pub(crate) inputs: Variables<V>,
    pub(crate) outputs: Variables<V>,
    pub(crate) output_slots: Vec<u32>,
    pub(crate) output_values: Vec<V>,
    pub(crate) stage: Vec<V>,
    pub(crate) seeds: Vec<V>,
    /// Values held by the seed and staging buffers while a transform has taken them
    /// out of the tape, counted by `reserved_bytes` and the byte limit meanwhile.
    pub(crate) held: [usize; 2],
    pub(crate) counts: Counts,
    pub(crate) faults: u8,
    pub(crate) point_len: usize,
    pub(crate) report: TapeReport,
}

impl<V: TapeValue> core::fmt::Debug for Tape<V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tape")
            .field("options", &self.options)
            .field("report", &self.report)
            .finish_non_exhaustive()
    }
}

/// The tape of [`TapeOptions::new`]: no policy but the defaults and no storage until
/// its first recording.
impl<V: TapeValue> Default for Tape<V> {
    fn default() -> Self {
        Self::new(TapeOptions::new())
    }
}

/// A buffer a transform takes out of the tape.
#[derive(Clone, Copy)]
pub(crate) enum Taken {
    Seeds = 0,
    Stage = 1,
}

/// The result of one recording attempt.
enum Attempt {
    Done,
    /// The recording did not fit: the counts it needs and the outputs it pushed.
    Grow(Counts, usize),
}

fn grow<T: Clone>(
    vector: &mut Vec<T>,
    len: usize,
    fill: T,
    quantity: scioxide_core::Quantity,
) -> Result<(), SciOxideError> {
    if vector.len() < len {
        vector
            .try_reserve_exact(len - vector.len())
            .map_err(|_| SciOxideError::allocation_failure(quantity, len))?;
        vector.resize(len, fill);
    }
    Ok(())
}

impl<V: TapeValue> Tape<V> {
    /// Builds an empty tape.
    #[must_use]
    pub fn new(options: TapeOptions) -> Self {
        Self {
            options,
            words: Vec::new(),
            full: Vec::new(),
            scalars: Vec::new(),
            calls: Vec::new(),
            scratch: Vec::new(),
            work: Vec::new(),
            call_work: Vec::new(),
            clean: true,
            inputs: Variables::default(),
            outputs: Variables::default(),
            output_slots: Vec::new(),
            output_values: Vec::new(),
            stage: Vec::new(),
            seeds: Vec::new(),
            held: [0; 2],
            counts: Counts::default(),
            faults: 0,
            point_len: 0,
            report: TapeReport::default(),
        }
    }

    /// Returns the options.
    #[must_use]
    pub const fn options(&self) -> &TapeOptions {
        &self.options
    }

    /// Sets the kink policy and the byte limit of later transforms; a batch applies
    /// those of its plan to the tapes of its workers before every evaluation.
    pub(crate) fn set_batch_options(&mut self, kink: KinkPolicy, byte_limit: Option<usize>) {
        self.options.kink = kink;
        self.options.byte_limit = byte_limit;
    }

    /// Returns the report of the last transform.
    #[must_use]
    pub const fn report(&self) -> TapeReport {
        self.report
    }

    /// Returns the requirement of the last completed recording.
    #[must_use]
    pub fn requirements(&self) -> TapeRequirements {
        TapeRequirements::new(
            self.counts.words,
            self.counts.full,
            self.counts.scalars,
            self.counts.calls,
            self.counts.scratch,
            self.point_len + self.output_slots.len(),
        )
    }

    /// Returns the bytes the tape has reserved.
    #[must_use]
    pub fn reserved_bytes(&self) -> usize {
        let value = size_of::<V>();
        self.words.capacity() * size_of::<u64>()
            + (self.full.capacity()
                + self.scratch.capacity()
                + self.work.capacity()
                + self.call_work.capacity()
                + self.output_values.capacity()
                + self.stage.capacity()
                + self.seeds.capacity()
                + self.held[0]
                + self.held[1])
                * value
            + self.scalars.capacity() * size_of::<V::Scalar>()
            + self.calls.capacity() * size_of::<Option<Call<V>>>()
            + (self.inputs.capacity() + self.outputs.capacity()) * size_of::<Var<'static, V>>()
            + self.output_slots.capacity() * size_of::<u32>()
    }

    /// Bytes the tape holds after growing to `counts` and `variables`: growth is
    /// exact, so every vector keeps the larger of its capacity and its target.
    fn bytes_after(&self, counts: Counts, variables: usize) -> Option<usize> {
        let at = |capacity: usize, len: usize| capacity.max(len);
        let values = at(self.full.capacity(), counts.full)
            .checked_add(at(self.scratch.capacity(), counts.scratch))?
            .checked_add(at(self.work.capacity(), counts.words))?
            .checked_add(at(
                self.call_work.capacity(),
                counts.scratch.checked_mul(2)?,
            ))?
            .checked_add(at(self.output_values.capacity(), variables))?
            .checked_add(self.stage.capacity())?
            .checked_add(self.seeds.capacity())?
            .checked_add(self.held[0])?
            .checked_add(self.held[1])?;
        let vars = at(self.inputs.capacity(), variables)
            .checked_add(at(self.outputs.capacity(), variables))?;
        at(self.words.capacity(), counts.words)
            .checked_mul(size_of::<u64>())?
            .checked_add(values.checked_mul(size_of::<V>())?)?
            .checked_add(
                at(self.scalars.capacity(), counts.scalars).checked_mul(size_of::<V::Scalar>())?,
            )?
            .checked_add(
                at(self.calls.capacity(), counts.calls).checked_mul(size_of::<Option<Call<V>>>())?,
            )?
            .checked_add(vars.checked_mul(size_of::<Var<'static, V>>())?)?
            .checked_add(at(self.output_slots.capacity(), variables).checked_mul(size_of::<u32>())?)
    }

    /// The lengths the columns hold now.
    fn retained(&self) -> Counts {
        Counts {
            words: self.words.len(),
            full: self.full.len(),
            scalars: self.scalars.len(),
            calls: self.calls.len(),
            scratch: self.scratch.len(),
        }
    }

    /// Reserves `variables` input and output variables within the byte limit.
    pub(crate) fn reserve_variables(&mut self, variables: usize) -> Result<(), SciOxideError> {
        if self.inputs.capacity() < variables || self.outputs.capacity() < variables {
            self.grow_to(self.retained(), variables)?;
        }
        Ok(())
    }

    /// Grows `buffer`, taken out of the tape, to `len` values within the byte limit.
    pub(crate) fn grow_taken(
        &mut self,
        buffer: &mut Vec<V>,
        taken: Taken,
        len: usize,
        quantity: scioxide_core::Quantity,
    ) -> Result<(), SciOxideError> {
        let slot = taken as usize;
        if buffer.len() < len {
            if let Some(limit) = self.options.byte_limit {
                // `held` already counts the buffer at its current capacity.
                let others = self.reserved_bytes() - self.held[slot] * size_of::<V>();
                let bytes = len
                    .max(buffer.capacity())
                    .checked_mul(size_of::<V>())
                    .and_then(|taken| taken.checked_add(others))
                    .ok_or_else(|| {
                        SciOxideError::size_overflow(scioxide_core::quantity!("tape_bytes"))
                    })?;
                if bytes > limit {
                    return Err(SciOxideError::resource_limit(
                        scioxide_core::quantity!("tape_bytes"),
                        bytes,
                        limit,
                    ));
                }
            }
            grow(buffer, len, V::zero(), quantity)?;
        }
        self.held[slot] = buffer.capacity();
        Ok(())
    }

    /// Takes the seed or staging buffer out of the tape for a transform; `held`
    /// counts it until [`Tape::restore`] puts it back.
    pub(crate) fn take_buffer(&mut self, taken: Taken) -> Vec<V> {
        let buffer = core::mem::take(match taken {
            Taken::Seeds => &mut self.seeds,
            Taken::Stage => &mut self.stage,
        });
        self.held[taken as usize] = buffer.capacity();
        buffer
    }

    /// Puts back a buffer of [`Tape::take_buffer`].
    pub(crate) fn restore(&mut self, buffer: Vec<V>, taken: Taken) {
        self.held[taken as usize] = 0;
        *(match taken {
            Taken::Seeds => &mut self.seeds,
            Taken::Stage => &mut self.stage,
        }) = buffer;
    }

    fn admit(&self, counts: Counts, variables: usize) -> Result<(), SciOxideError> {
        if counts.words > self.options.word_limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("tape_words"),
                counts.words,
                self.options.word_limit,
            ));
        }
        let bytes = self
            .bytes_after(counts, variables)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("tape_bytes")))?;
        if let Some(limit) = self.options.byte_limit
            && bytes > limit
        {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("tape_bytes"),
                bytes,
                limit,
            ));
        }
        Ok(())
    }

    /// Grows every vector to the given lengths.
    pub(crate) fn grow_to(
        &mut self,
        counts: Counts,
        variables: usize,
    ) -> Result<(), SciOxideError> {
        self.admit(counts, variables)?;
        let zero = V::zero();
        // `work` and `call_work` grow before the columns the recorder compares, so a
        // failure leaves a tape whose next recording overflows and grows again.
        grow(
            &mut self.work,
            counts.words,
            zero,
            scioxide_core::quantity!("tape_adjoints"),
        )?;
        grow(
            &mut self.call_work,
            2 * counts.scratch,
            zero,
            scioxide_core::quantity!("tape_call_work"),
        )?;
        grow(
            &mut self.words,
            counts.words,
            word::INPUT,
            scioxide_core::quantity!("tape_words"),
        )?;
        grow(
            &mut self.full,
            counts.full,
            zero,
            scioxide_core::quantity!("tape_partials"),
        )?;
        grow(
            &mut self.scalars,
            counts.scalars,
            <V::Scalar as scioxide_core::AlgebraicScalar>::ZERO,
            scioxide_core::quantity!("tape_scalars"),
        )?;
        if self.calls.len() < counts.calls {
            self.calls
                .try_reserve_exact(counts.calls - self.calls.len())
                .map_err(|_| {
                    SciOxideError::allocation_failure(
                        scioxide_core::quantity!("tape_calls"),
                        counts.calls,
                    )
                })?;
            self.calls.resize_with(counts.calls, || None);
        }
        grow(
            &mut self.scratch,
            counts.scratch,
            zero,
            scioxide_core::quantity!("tape_scratch"),
        )?;
        self.inputs.try_reserve(variables)?;
        self.outputs.try_reserve(variables)?;
        // Exact reservations up to `variables` slots: the amortized policy would
        // double the retained capacity on every regrowth.
        let slots = variables.saturating_sub(self.output_slots.len());
        let values = variables.saturating_sub(self.output_values.len());
        self.output_slots
            .try_reserve_exact(slots)
            .and_then(|()| self.output_values.try_reserve_exact(values))
            .map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("tape_outputs"),
                    variables,
                )
            })?;
        Ok(())
    }

    /// Grows the tape to hold `requirements`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the requirement exceeds the byte limit or 2²⁸ words.
    /// - `SizeOverflow` if the byte count overflows `usize`.
    /// - `AllocationFailure` if a vector cannot grow; the tape stays usable.
    pub fn try_reserve(&mut self, requirements: &TapeRequirements) -> Result<(), SciOxideError> {
        let counts = Counts {
            words: requirements.words,
            full: requirements.partials,
            scalars: requirements.scalars,
            calls: requirements.calls,
            scratch: requirements.scratch,
        };
        self.grow_to(counts, requirements.variables)
    }

    /// Classifies the retained storage against `requirements`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the requirement exceeds the byte limit or 2²⁸ words.
    /// - `SizeOverflow` if the byte count overflows `usize`.
    pub fn check(&self, requirements: &TapeRequirements) -> Result<Capacity, SciOxideError> {
        let counts = Counts {
            words: requirements.words,
            full: requirements.partials,
            scalars: requirements.scalars,
            calls: requirements.calls,
            scratch: requirements.scratch,
        };
        self.admit(counts, requirements.variables)?;
        let enough = self.words.len() >= counts.words
            && self.work.len() >= counts.words
            && self.full.len() >= counts.full
            && self.scalars.len() >= counts.scalars
            && self.calls.len() >= counts.calls
            && self.scratch.len() >= counts.scratch
            && self.call_work.len() >= 2 * counts.scratch
            && self.inputs.capacity() >= requirements.variables
            && self.outputs.capacity() >= requirements.variables;
        Ok(if enough {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }

    fn check_point(&self, point: &[V]) -> Result<(), SciOxideError> {
        if self.options.non_finite == NonFinitePolicy::Error
            && point.iter().any(|value| !value.is_finite().all())
        {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "the point is not finite",
            ));
        }
        Ok(())
    }

    /// Checks results under [`NonFinitePolicy::Error`].
    pub(crate) fn check_results(&self, values: &[V]) -> Result<(), SciOxideError> {
        if self.options.non_finite == NonFinitePolicy::Error
            && values.iter().any(|value| !value.is_finite().all())
        {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Output,
                "a value or derivative is not finite",
            ));
        }
        Ok(())
    }

    /// Records `function` at `point`, leaving its outputs in `output_slots` and
    /// `output_values`.
    pub(crate) fn record<F>(&mut self, point: &[V], function: &F) -> Result<(), SciOxideError>
    where
        F: for<'t> Fn(&[Var<'t, V>], &mut Vec<Var<'t, V>>) -> Result<(), SciOxideError>,
    {
        self.check_point(point)?;
        // The function pushes its outputs; as many as the last recording had (at
        // least one) are reserved fallibly beforehand. More outputs grow the vector
        // in the function's own `push`.
        self.reserve_variables(point.len() + self.output_slots.len().max(1))?;
        let mut twice = false;
        let outcome = loop {
            let (attempt, outcome) = self.record_once(point, function);
            match attempt {
                Attempt::Done => break outcome,
                // An error of the function is its own, even from a run that did not
                // fit: return it without running the function again.
                Attempt::Grow(..) if outcome.is_err() => break outcome,
                Attempt::Grow(counts, outputs) if !twice => {
                    twice = true;
                    // Size the variables from what this recording pushed, never from a
                    // retained capacity, so regrowth does not feed on earlier growth.
                    let variables = point.len() + outputs.max(1);
                    self.grow_to(counts, variables)?;
                }
                Attempt::Grow(..) => {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("function"),
                        InvalidInputReason::Inconsistent,
                        "the function recorded more on its second run than on its first",
                    ));
                }
            }
        };
        self.report.recorded_twice = twice;
        self.report.words = self.counts.words;
        self.report.reserved_bytes = self.reserved_bytes();
        outcome?;
        // A warm tape satisfies its own requirements: both variable vectors hold the
        // inputs and outputs of the recording.
        self.reserve_variables(self.point_len + self.output_slots.len())?;
        self.report.reserved_bytes = self.reserved_bytes();
        let faults = self.faults;
        if faults & WORD_LIMIT != 0 {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("tape_words"),
                self.counts.words,
                self.options.word_limit,
            ));
        }
        if faults & MIXED != 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("function"),
                InvalidInputReason::Inconsistent,
                "an operation mixed values of two recordings",
            ));
        }
        if self.options.kink == KinkPolicy::Error && self.report.kinks != 0 {
            return Err(SciOxideError::domain_error(
                scioxide_core::argument!("point"),
                DomainReason::Undefined,
                "an active argument sits on a kink",
            ));
        }
        self.check_results(&self.output_values)
    }

    fn record_once<F>(&mut self, point: &[V], function: &F) -> (Attempt, Result<(), SciOxideError>)
    where
        F: for<'t> Fn(&[Var<'t, V>], &mut Vec<Var<'t, V>>) -> Result<(), SciOxideError>,
    {
        let Self {
            words,
            full,
            scalars,
            calls,
            scratch,
            inputs,
            outputs,
            output_slots,
            output_values,
            options,
            ..
        } = self;
        let recorder = Recorder::new(words, full, scalars, calls, scratch, options.word_limit);
        let mut variables = inputs.take();
        let mut results = outputs.take();
        recorder.push_word(word::INPUT);
        variables.extend(point.iter().map(|&value| recorder.input(value)));
        let outcome = function(&variables, &mut results);
        let mut published = Ok(());
        if !recorder.overflowed() && outcome.is_ok() {
            output_slots.clear();
            output_values.clear();
            // Both vectors were just cleared, so this asks for room for every output.
            published = output_slots
                .try_reserve_exact(results.len())
                .and_then(|()| output_values.try_reserve_exact(results.len()))
                .map_err(|_| {
                    SciOxideError::allocation_failure(
                        scioxide_core::quantity!("tape_outputs"),
                        results.len(),
                    )
                });
        }
        if published.is_ok() && !recorder.overflowed() && outcome.is_ok() {
            for result in &results {
                let slot = match result.recorder() {
                    Some(r) if core::ptr::eq(r, &recorder) => result.slot(),
                    Some(_) => {
                        recorder.fault(MIXED);
                        0
                    }
                    None => 0,
                };
                output_slots.push(slot);
                output_values.push(result.value());
            }
        }
        let counts = recorder.counts();
        let overflowed = recorder.overflowed();
        let faults = recorder.faults();
        let kinks = recorder.kinks();
        let pushed = results.len();
        inputs.keep(variables);
        outputs.keep(results);
        if overflowed {
            return (Attempt::Grow(counts, pushed), outcome);
        }
        self.report.kinks = kinks;
        let outcome = outcome.and(published);
        if outcome.is_ok() {
            // `requirements` describes the last completed recording.
            self.counts = counts;
            self.faults = faults;
            self.point_len = point.len();
        }
        (Attempt::Done, outcome)
    }

    /// The columns of the last recording.
    pub(crate) fn frozen(&self) -> Frozen<'_, V> {
        Frozen {
            words: &self.words[..self.counts.words],
            full: &self.full[..self.counts.full],
            scalars: &self.scalars[..self.counts.scalars],
            calls: &self.calls[..self.counts.calls],
        }
    }

    /// Zeroes the work buffer if a tangent sweep or a failure left it dirty.
    pub(crate) fn clean_work(&mut self) {
        if !self.clean {
            // The dirt may come from an earlier, longer recording.
            self.work.fill(V::zero());
            self.clean = true;
        }
    }

    /// Zeroes the input adjoints that a completed reverse sweep leaves; the sweep
    /// cleared every other entry.
    pub(crate) fn clear_inputs(&mut self, inputs: usize) {
        self.work[..=inputs].fill(V::zero());
        self.clean = true;
    }

    /// Sweeps the whole recording in reverse after the caller seeded `work`.
    pub(crate) fn sweep_reverse(
        &mut self,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let Self {
            words,
            full,
            scalars,
            calls,
            work,
            call_work,
            counts,
            ..
        } = self;
        let frozen = Frozen {
            words: &words[..counts.words],
            full: &full[..counts.full],
            scalars: &scalars[..counts.scalars],
            calls: &calls[..counts.calls],
        };
        // Dirty until the caller clears the input adjoints, also if a rule panics.
        self.clean = false;
        sweep::reverse(&frozen, *counts, 0, work, call_work, context)
    }

    /// Sweeps the whole recording forward after the caller seeded `work`.
    pub(crate) fn sweep_tangent(
        &mut self,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.clean = false;
        let Self {
            words,
            full,
            scalars,
            calls,
            work,
            call_work,
            counts,
            ..
        } = self;
        let frozen = Frozen {
            words: &words[..counts.words],
            full: &full[..counts.full],
            scalars: &scalars[..counts.scalars],
            calls: &calls[..counts.calls],
        };
        let start = Counts::default();
        sweep::tangent(&frozen, start, counts.words, work, call_work, context)
    }

    /// Reverse product `Jᵀ·cotangent` of the last recording into `out`.
    pub(crate) fn vjp_into(
        &mut self,
        cotangent: &[V],
        out: &mut [V],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        crate::support::check_len(cotangent.len(), self.output_slots.len())?;
        crate::support::check_len(out.len(), self.point_len)?;
        context.check()?;
        self.clean_work();
        for (&slot, &seed) in self.output_slots.iter().zip(cotangent) {
            if let Some(target) = self.work.get_mut(slot as usize) {
                *target += seed;
            }
        }
        self.sweep_reverse(context)?;
        let inputs = &self.work[1..=self.point_len];
        let checked = self.check_results(inputs);
        if checked.is_ok() {
            out.copy_from_slice(inputs);
        }
        self.clear_inputs(self.point_len);
        checked
    }

    /// Tangent product `J·direction` of the last recording into `out`.
    pub(crate) fn jvp_into(
        &mut self,
        direction: &[V],
        out: &mut [V],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        crate::support::check_len(direction.len(), self.point_len)?;
        crate::support::check_len(out.len(), self.output_slots.len())?;
        context.check()?;
        self.clean = false;
        if let Some(constant) = self.work.first_mut() {
            *constant = V::zero();
        }
        self.work[1..=self.point_len].copy_from_slice(direction);
        self.sweep_tangent(context)?;
        let work = &self.work;
        let slots = &self.output_slots;
        let value = |k: usize| work.get(slots[k] as usize).copied().unwrap_or_else(V::zero);
        if self.options.non_finite == NonFinitePolicy::Error
            && (0..slots.len()).any(|k| !value(k).is_finite().all())
        {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Output,
                "a derivative is not finite",
            ));
        }
        for (k, target) in out.iter_mut().enumerate() {
            *target = value(k);
        }
        Ok(())
    }

    /// Computes the value and gradient of `function` at `point`.
    ///
    /// `function` maps the inputs to one output. The gradient is written to
    /// `gradient` (one entry per input) and the value is returned. Cost: one
    /// recording and one reverse sweep, a small constant times the evaluation.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `gradient.len() != point.len()`.
    /// - The errors of `function`, unchanged.
    /// - `NonFinite` under [`NonFinitePolicy::Error`]: origin `Input` for the point,
    ///   `Output` for the value or the gradient.
    /// - `DomainError` (argument `point`, reason `Undefined`) under
    ///   [`KinkPolicy::Error`] when an active argument sits on a kink.
    /// - `InvalidInput` (`Inconsistent`) if an operation mixed two recordings or the
    ///   second run of the function did not fit the tape grown to the counts of
    ///   the first, that is, recorded more (a smaller second run is accepted).
    /// - `ResourceLimit`, `AllocationFailure` and `SizeOverflow` when the tape cannot
    ///   grow; `ExecutionStopped` if `context` is cancelled.
    ///
    /// `gradient` is unchanged on error.
    pub fn gradient_into<F>(
        &mut self,
        function: F,
        point: &[V],
        gradient: &mut [V],
        context: ExecutionContext<'_>,
    ) -> Result<V, SciOxideError>
    where
        F: for<'t> Fn(&[Var<'t, V>]) -> Result<Var<'t, V>, SciOxideError>,
    {
        crate::support::check_len(gradient.len(), point.len())?;
        context.check()?;
        self.record(
            point,
            &|inputs: &[Var<'_, V>], outputs: &mut Vec<Var<'_, V>>| {
                outputs.push(function(inputs)?);
                Ok(())
            },
        )?;
        let value = self.output_values.first().copied().unwrap_or_else(V::zero);
        self.vjp_into(&[V::one()], gradient, context)?;
        Ok(value)
    }

    /// Records a vector function at `point` and returns its linearization.
    ///
    /// `function` pushes its outputs onto the vector it receives. The
    /// [`Linearization`] reuses the recording for any number of products at this
    /// point.
    ///
    /// # Errors
    ///
    /// The recording errors of [`Tape::gradient_into`].
    pub fn linearize<F>(
        &mut self,
        function: F,
        point: &[V],
        context: ExecutionContext<'_>,
    ) -> Result<Linearization<'_, V>, SciOxideError>
    where
        F: for<'t> Fn(&[Var<'t, V>], &mut Vec<Var<'t, V>>) -> Result<(), SciOxideError>,
    {
        context.check()?;
        self.record(point, &function)?;
        Ok(Linearization::new(self))
    }
}
