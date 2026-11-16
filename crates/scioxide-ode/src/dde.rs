//! Retarded delay differential equations with constant delays, integrated by the
//! method of steps.
//!
//! The history is continuous and piecewise smooth. The right-hand side sees only past
//! state values, so neutral, advanced and state-dependent delay problems are not
//! supported.

use crate::dense_output::DenseOutputStorage;
use crate::ivp::{Drive, Finish, IvpStorage, drive};
use crate::safety::invalid;
use crate::shell::Shell;
use crate::stepper::{MethodInfo, Stepper};
use crate::termination::Terminal;
use crate::termination::admit;
use crate::{
    DormandPrince5, EventFunction, EventSearchEvidence, OdeOptions, OdeOutputSink, OdeProgress,
    OdeState, OdeSystem, OdeTrajectory,
};
use alloc::vec::Vec;
use core::cell::RefCell;
use scioxide_core::backend::{checked_add, checked_mul, try_reserve, try_with_capacity};
use scioxide_core::checked_difference;
use scioxide_core::{
    AttemptRecord, Budget, Capacity, EvaluationCounts, EvaluationRole, ExecutionContext,
    NonFiniteOrigin, NumericScalar, PartialFailure, PartialResultTrust, ProblemToken, RealScalar,
    SciOxideError, SolveCounts, SolveIdentity, SolveScope,
};
use scioxide_linalg::{MatrixView, Vector};

/// Defines the initial history `φ(t)` of a delay equation on `[t₀ − τ_max, t₀]`.
///
/// `t₀` is the start of the integration span and `τ_max` the largest delay. `φ` must be
/// continuous; declare every point where one of its derivatives jumps as a
/// [`DdeHistoryBreak`]. It must return the same value for the same time, regardless of
/// the order of calls. A closure `Fn(t, out) -> Result<(), SciOxideError>` implements this
/// trait.
pub trait DdeHistory<S: RealScalar> {
    /// Writes `φ(time)` to `output`.
    ///
    /// Each call counts as one `Function` evaluation. A NaN or infinite value is a
    /// `NonFinite` error with origin `Callback`.
    ///
    /// # Errors
    ///
    /// An error stops the integration and is returned unchanged.
    fn evaluate(&self, time: S, output: &mut [S]) -> Result<(), SciOxideError>;
}
impl<S: RealScalar, F: Fn(S, &mut [S]) -> Result<(), SciOxideError>> DdeHistory<S> for F {
    fn evaluate(&self, time: S, output: &mut [S]) -> Result<(), SciOxideError> {
        self(time, output)
    }
}

/// Defines the right-hand side of a retarded delay equation with constant delays.
///
/// ```text
/// y′(t) = f(t, y(t), y(t − τ₁), …, y(t − τₖ)),   t₀ ≤ t ≤ t_end,
/// y(t) = φ(t),                                  t₀ − τ_max ≤ t ≤ t₀,
/// ```
///
/// where `τ₁, …, τₖ > 0` are the delays of the [`DdePlan`] and `φ` the [`DdeHistory`].
/// Row `j` of `delayed` holds `y(time − τⱼ)` and its columns are the state components;
/// rows follow the order of the delays given to [`DdePlan::new`], including duplicates.
/// A closure `Fn(t, y, delayed, out) -> Result<(), SciOxideError>` implements this trait.
pub trait DdeRhs<S: RealScalar + NumericScalar<Real = S>> {
    /// Writes `f(time, state, delayed)` to `output`.
    ///
    /// Write every component; the slices do not overlap. Each call counts as one
    /// `RightHandSide` evaluation. A NaN or infinite value is a `NonFinite` error with
    /// origin `Callback`.
    ///
    /// # Errors
    ///
    /// An error stops the integration and is returned unchanged.
    fn evaluate(
        &self,
        time: S,
        state: &[S],
        delayed: MatrixView<'_, S>,
        output: &mut [S],
    ) -> Result<(), SciOxideError>;
}
impl<S, F> DdeRhs<S> for F
where
    S: RealScalar + NumericScalar<Real = S>,
    F: Fn(S, &[S], MatrixView<'_, S>, &mut [S]) -> Result<(), SciOxideError>,
{
    fn evaluate(
        &self,
        time: S,
        state: &[S],
        delayed: MatrixView<'_, S>,
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        self(time, state, delayed, output)
    }
}

/// Pairs the right-hand side and history of a delay equation with optional events.
///
/// The dimension, span and delays belong to the [`DdePlan`].
#[derive(Clone, Copy)]
pub struct DdeProblem<'a, S: RealScalar + NumericScalar<Real = S>> {
    rhs: &'a dyn DdeRhs<S>,
    history: &'a dyn DdeHistory<S>,
    events: &'a [&'a dyn EventFunction<S>],
}
impl<'a, S: RealScalar + NumericScalar<Real = S>> DdeProblem<'a, S> {
    /// Creates a problem from the right-hand side and the initial history, without
    /// events.
    pub const fn new(rhs: &'a dyn DdeRhs<S>, history: &'a dyn DdeHistory<S>) -> Self {
        Self {
            rhs,
            history,
            events: &[],
        }
    }
    /// Sets event functions of `(t, y(t))` to locate during the integration.
    ///
    /// Events cannot depend on delayed values, and the state cannot be reset at an event.
    /// See [events and output](crate#events-and-output).
    pub const fn with_events(mut self, events: &'a [&'a dyn EventFunction<S>]) -> Self {
        self.events = events;
        self
    }
    /// Returns the event functions.
    pub const fn events(&self) -> &'a [&'a dyn EventFunction<S>] {
        self.events
    }
}

/// Configures [`DdePlan`]: the initial-value options and two storage limits.
///
/// Error control, step bounds, output, events and budgets come from the
/// [`OdeOptions`].
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct DdeOptions<S: RealScalar> {
    ode: OdeOptions<S>,
    history_capacity: Budget,
    schedule_capacity: Budget,
}
impl<S: RealScalar> DdeOptions<S> {
    /// Creates default [`OdeOptions`] with no history or schedule capacity limit.
    /// The limits are compared, not reserved.
    ///
    /// The schedule limit counts the breaks before the start time too. Both are upper
    /// bounds: the plan stores only the schedule it computes, and history storage grows
    /// as needed. One more step of history storage is used for the step in progress.
    pub fn new() -> Self {
        let history_capacity = Budget::ONE.saturating_add(usize::MAX);
        let schedule_capacity = history_capacity;
        Self {
            ode: OdeOptions::new(),
            history_capacity,
            schedule_capacity,
        }
    }
    /// Limits retained history segments; the default has no limit.
    pub fn with_history_capacity(mut self, segments: Budget) -> Self {
        self.history_capacity = segments;
        self
    }
    /// Limits discontinuity entries; the default has no limit.
    pub fn with_schedule_capacity(mut self, entries: Budget) -> Self {
        self.schedule_capacity = entries;
        self
    }
    /// Sets the error control, step bounds, output, events and budgets.
    ///
    /// The evaluation budget counts history calls as well as right-hand-side calls.
    pub fn with_ode_options(mut self, options: OdeOptions<S>) -> Self {
        self.ode = options;
        self
    }
    /// Returns the initial-value options.
    pub const fn ode_options(&self) -> &OdeOptions<S> {
        &self.ode
    }
    /// Returns the largest number of accepted steps kept as history.
    pub const fn history_capacity(&self) -> Budget {
        self.history_capacity
    }
    /// Returns the largest number of discontinuity times.
    pub const fn schedule_capacity(&self) -> Budget {
        self.schedule_capacity
    }
}

/// Marks a time where a derivative of the initial history `φ` jumps.
///
/// `derivative_order` is the lowest derivative that jumps: one for a jump in `φ′`. Order
/// zero is invalid, because `φ` must be continuous. A break of order below five
/// propagates into the solution at the break time plus each delay, one derivative order
/// higher; jumps in higher derivatives need no step to end on them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DdeHistoryBreak<S: RealScalar> {
    time: S,
    derivative_order: usize,
}

impl<S: RealScalar> DdeHistoryBreak<S> {
    /// Creates a break at `time` in derivative `derivative_order`.
    ///
    /// [`DdePlan::new`] checks both values.
    pub const fn new(time: S, derivative_order: usize) -> Self {
        Self {
            time,
            derivative_order,
        }
    }
    /// Creates a break at `time` in the first derivative.
    pub const fn first_derivative(time: S) -> Self {
        Self::new(time, 1)
    }
    /// Returns the time of the break.
    pub const fn time(&self) -> S {
        self.time
    }
    /// Returns the lowest derivative that jumps.
    pub const fn derivative_order(&self) -> usize {
        self.derivative_order
    }
}

/// Integrates a retarded delay equation with constant delays by the method of steps.
///
/// The problem is the one stated on [`DdeRhs`]. The plan steps with the adaptive
/// Dormand–Prince 5(4) pair ([`DormandPrince5`](crate::DormandPrince5)) under the error
/// control of the [`OdeOptions`]. No step is longer than the smallest delay, so every
/// delayed value comes from the initial history or from the continuous extension of an
/// accepted step, and no step needs an iteration. History older than `t − τ_max` is
/// discarded as the integration advances.
///
/// The solution of a delay equation has derivative jumps that propagate from the start
/// time and from the declared [`DdeHistoryBreak`]s: a jump in derivative `m` at time `s`
/// causes a jump in derivative `m + 1` at `s + τⱼ` for every delay `τⱼ`. The plan
/// precomputes these times with `O(b d)` schedule insertions for `b` propagated
/// breaks and `d` delays, up to derivative order five, and steps end exactly on
/// them. Times within `8 ε max(|a|, |b|)` of each other are merged, where `ε` is the
/// machine epsilon. The start time is always a break in the first derivative, which
/// allows `φ′(t₀)` to differ from `f` at `t₀`.
///
/// [`Self::advance`] writes into an [`OdeState`] you own and sends output to an
/// [`OdeOutputSink`]; [`Self::solve`] returns the final state and an [`OdeTrajectory`].
/// The local error estimate does not include the error of the interpolated history.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, SciOxideError};
/// use scioxide_linalg::MatrixView;
/// use scioxide_ode::{DdeOptions, DdePlan, DdeProblem};
///
/// // y′(t) = y(t − 1), y(t) = 1 for t ≤ 0; the method of steps gives y(2) = 7/2.
/// let history = |_: f64, output: &mut [f64]| {
///     output.fill(1.0);
///     Ok(())
/// };
/// let rhs = |_: f64, _: &[f64], delayed: MatrixView<'_, f64>, output: &mut [f64]| {
///     output[0] = delayed.get(0, 0)?;
///     Ok::<(), SciOxideError>(())
/// };
/// let options = DdeOptions::new().with_history_capacity(Budget::try_from(128)?).with_schedule_capacity(Budget::try_from(64)?);
/// let plan = DdePlan::new(1, (0.0, 2.0), &[1.0], &[], options)?;
/// let solution = plan.solve(&DdeProblem::new(&rhs, &history))?;
/// assert!((solution.state().values()[0] - 3.5).abs() < 1e-6);
/// # Ok::<(), SciOxideError>(())
/// ```
///
/// # References
///
/// A. Bellen and M. Zennaro, *Numerical Methods for Delay Differential Equations*,
/// Oxford University Press, 2003.
#[derive(Debug, Clone)]
#[must_use]
pub struct DdePlan<S: RealScalar + NumericScalar<Real = S>> {
    dimension: usize,
    span: (S, S),
    delays: Vec<S>,
    schedule: Vec<S>,
    minimum: S,
    maximum: S,
    options: DdeOptions<S>,
    method: DormandPrince5<S>,
    info: MethodInfo,
}
impl<S: RealScalar + NumericScalar<Real = S>> DdePlan<S> {
    /// Creates a plan for a `dimension`-component equation on `span = (t₀, t_end)` with
    /// the given `delays`, computing the schedule of discontinuities.
    ///
    /// `history_breaks` lists the derivative jumps of the history in
    /// `[t₀ − τ_max, t₀]`. Computing the schedule costs time and memory proportional to
    /// its length, not to the `schedule_capacity` bound. No callback runs.
    ///
    /// ```
    /// use scioxide_core::Budget;
    /// use scioxide_ode::{DdeHistoryBreak, DdeOptions, DdePlan};
    ///
    /// let options = DdeOptions::new().with_history_capacity(Budget::try_from(1024)?).with_schedule_capacity(Budget::try_from(64)?);
    /// let breaks = [DdeHistoryBreak::new(-0.5, 4)];
    /// let plan = DdePlan::new(1, (0.0_f64, 10.0), &[1.0], &breaks, options)?;
    /// // The fourth-derivative jump at −0.5 propagates to 0.5 with the delay 1.
    /// assert!(plan.discontinuities().contains(&0.5));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `NonFinite` (origin `Input`) if a span end, a delay or a break time is not
    ///   finite.
    /// - `InvalidInput` if `t₀ ≥ t_end`, there is no delay, a delay is not positive, the
    ///   smallest delay is too small to change `t₀` or `t_end` in floating point, or a
    ///   break has order zero or lies outside `[t₀ − τ_max, t₀]`.
    /// - `ResourceLimit` if the merged schedule, including the breaks before `t₀`, needs
    ///   more than `schedule_capacity` entries.
    /// - The errors of invalid [`OdeOptions`] for `dimension`.
    /// - `SizeOverflow` or `AllocationFailure` if the schedule cannot be stored.
    pub fn new(
        dimension: usize,
        span: (S, S),
        delays: &[S],
        history_breaks: &[DdeHistoryBreak<S>],
        options: DdeOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.ode.validate(dimension)?;
        Self::prepare(
            dimension,
            span,
            scioxide_core::backend::try_copy_slice(delays, scioxide_core::quantity!("delays"))?,
            history_breaks.iter().copied(),
            options,
        )
    }
    fn prepare(
        dimension: usize,
        span: (S, S),
        delays: Vec<S>,
        history_breaks: impl Iterator<Item = DdeHistoryBreak<S>> + Clone,
        options: DdeOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let schedule_capacity = options.schedule_capacity;
        if !span.0.is_finite()
            || !span.1.is_finite()
            || delays.iter().any(|&value| !value.is_finite())
            || history_breaks.clone().any(|b| !b.time.is_finite())
        {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "time span, delays and history breaks must be finite",
            ));
        }
        if span.0 >= span.1 {
            return Err(invalid(
                scioxide_core::argument!("span"),
                "a finite strictly forward span is required",
            ));
        }
        if delays.is_empty()
            || delays
                .iter()
                .any(|&d| !d.is_finite() || d <= <S as scioxide_core::AlgebraicScalar>::ZERO)
        {
            return Err(invalid(
                scioxide_core::argument!("delays"),
                "at least one finite strictly positive constant delay is required",
            ));
        }
        checked_mul(
            dimension,
            delays.len(),
            scioxide_core::quantity!("delayed_values"),
        )?;
        let minimum = delays.iter().copied().fold(S::INFINITY, RealScalar::min);
        let maximum = delays
            .iter()
            .copied()
            .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, RealScalar::max);
        let history_start = checked_difference(span.0, maximum)?;
        if history_breaks
            .clone()
            .any(|b| b.time < history_start || b.time > span.0 || b.derivative_order == 0)
        {
            return Err(invalid(
                scioxide_core::argument!("history_breaks"),
                "history breaks need a positive derivative order and a time in the required history interval",
            ));
        }
        if span.0 - minimum >= span.0 || span.1 - minimum >= span.1 {
            return Err(invalid(
                scioxide_core::argument!("delays"),
                "the smallest delay must make representable backward progress at both span endpoints",
            ));
        }
        let mut breaks = Vec::new();
        let mut seeds = try_with_capacity(
            history_breaks.clone().count().saturating_add(1),
            scioxide_core::quantity!("history_breaks"),
        )?;
        seeds.push(DdeHistoryBreak {
            time: span.0,
            derivative_order: 1,
        });
        seeds.extend(history_breaks);
        seeds.sort_unstable_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| a.derivative_order.cmp(&b.derivative_order))
        });
        for b in seeds {
            insert_break(
                &mut breaks,
                b.time,
                b.derivative_order,
                schedule_capacity.get(),
            )?;
        }
        // Delay order must not affect representatives; callback rows retain their order.
        let mut propagation = scioxide_core::backend::try_copy_slice(
            &delays,
            scioxide_core::quantity!("propagation_delays"),
        )?;
        propagation.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        propagation.dedup();
        while let Some(index) = breaks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.order < 5 && !b.propagated && b.time < span.1)
            .min_by(|(_, a), (_, b)| {
                a.order.cmp(&b.order).then_with(|| {
                    a.time
                        .partial_cmp(&b.time)
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
            })
            .map(|(i, _)| i)
        {
            let time = breaks[index].time;
            let order = breaks[index].order;
            breaks[index].propagated = true;
            for &delay in &propagation {
                let next = time + delay;
                // Positive overflow lies beyond the finite horizon.
                if !next.is_finite() || (next > span.1 && !coincident(next, span.1)) {
                    continue;
                }
                let next = next.min(span.1);
                if next <= time {
                    return Err(invalid(
                        scioxide_core::argument!("delays"),
                        "discontinuity propagation cannot make representable progress",
                    ));
                }
                insert_break(&mut breaks, next, order + 1, schedule_capacity.get())?;
            }
        }
        let mut schedule = try_with_capacity(
            breaks.len(),
            scioxide_core::quantity!("discontinuity_schedule"),
        )?;
        schedule.extend(
            breaks
                .iter()
                .filter(|b| b.time > span.0 && !coincident(b.time, span.0))
                .map(|b| {
                    if coincident(b.time, span.1) {
                        span.1
                    } else {
                        b.time
                    }
                }),
        );
        Ok(Self {
            dimension,
            span,
            delays,
            schedule,
            minimum,
            maximum,
            options,
            method: DormandPrince5::new(),
            info: Stepper::<S, S>::info(&DormandPrince5::new()),
        })
    }
    /// Returns the number of state components, which may be zero.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    /// Returns the integration span `(t₀, t_end)`.
    pub const fn time_span(&self) -> (S, S) {
        self.span
    }
    /// Returns the delays, in the row order of [`DdeRhs::evaluate`].
    pub fn delays(&self) -> &[S] {
        &self.delays
    }
    /// Returns the discontinuity times after `t₀` that steps end on, in increasing order.
    pub fn discontinuities(&self) -> &[S] {
        &self.schedule
    }
    /// Returns the options.
    pub const fn options(&self) -> &DdeOptions<S> {
        &self.options
    }
}

struct ScheduledBreak<S> {
    time: S,
    order: usize,
    propagated: bool,
}
fn coincident<S: RealScalar>(a: S, b: S) -> bool {
    let eight = S::TWO * S::TWO * S::TWO;
    (b - a).abs() <= eight * S::EPSILON * a.abs().max(b.abs())
}
fn insert_break<S: RealScalar>(
    times: &mut Vec<ScheduledBreak<S>>,
    time: S,
    order: usize,
    capacity: usize,
) -> Result<(), SciOxideError> {
    let index = times.partition_point(|b| b.time < time);
    let match_index = if index > 0 && coincident(times[index - 1].time, time) {
        Some(index - 1)
    } else if times.get(index).is_some_and(|b| coincident(time, b.time)) {
        Some(index)
    } else {
        None
    };
    if let Some(i) = match_index {
        let b = &mut times[i];
        if time < b.time || order < b.order {
            b.propagated = false;
            b.time = b.time.min(time);
            b.order = b.order.min(order);
        }
        // A shifted representative can also coincide with its following neighbor.
        while i + 1 < times.len() && coincident(times[i].time, times[i + 1].time) {
            let next = times.remove(i + 1);
            if next.order < times[i].order {
                times[i].order = next.order;
                times[i].propagated = false;
            }
        }
        return Ok(());
    }
    if times.len() == capacity {
        return Err(SciOxideError::resource_limit(
            scioxide_core::quantity!("discontinuity_schedule"),
            capacity.saturating_add(1),
            capacity,
        ));
    }
    if times.len() == times.capacity() {
        // Geometric growth capped at the limit keeps preparation proportional to the
        // computed schedule rather than to its bound.
        let additional = times.len().max(4).min(capacity - times.len());
        try_reserve(
            times,
            additional,
            scioxide_core::quantity!("discontinuity_schedule"),
        )?;
    }
    times.insert(
        index,
        ScheduledBreak {
            time,
            order,
            propagated: false,
        },
    );
    Ok(())
}

pub(crate) struct DdeStorage<S: RealScalar + NumericScalar<Real = S>> {
    segments: Vec<DenseOutputStorage<S>>,
    pending: DenseOutputStorage<S>,
    initial: Vec<S>,
    head: usize,
    len: usize,
    peak: usize,
    /// The initial history of the current epoch was evaluated.
    initialized: bool,
    /// Set only after the fixed history storage is fully prepared.
    dimension: Option<usize>,
}
impl<S: RealScalar + NumericScalar<Real = S>> DdeStorage<S> {
    const fn empty() -> Self {
        Self {
            segments: Vec::new(),
            pending: DenseOutputStorage::new(),
            initial: Vec::new(),
            head: 0,
            len: 0,
            peak: 0,
            initialized: false,
            dimension: None,
        }
    }
    fn fits(&self, requirements: &DdeRequirements) -> bool {
        self.dimension == Some(requirements.dimension)
            && self.segments.len() >= requirements.history_segments
    }
    fn new(plan: &DdeRequirements) -> Result<Self, SciOxideError> {
        let mut pending = DenseOutputStorage::new();
        pending.try_resize(plan.dimension, 4)?;
        let mut initial =
            try_with_capacity(plan.dimension, scioxide_core::quantity!("initial_history"))?;
        initial.resize(plan.dimension, <S as scioxide_core::AlgebraicScalar>::ZERO);
        let mut storage = Self {
            segments: Vec::new(),
            pending,
            initial,
            head: 0,
            len: 0,
            peak: 0,
            initialized: false,
            dimension: Some(plan.dimension),
        };
        storage.reserve_segments(plan.history_segments)?;
        Ok(storage)
    }
    fn reserve_segments(&mut self, count: usize) -> Result<(), SciOxideError> {
        if count <= self.segments.len() {
            return Ok(());
        }
        Self::validate_size(self.initial.len(), count)?;
        // Normalize before changing the ring's modulus. Partial growth keeps all
        // accepted segments in order, even if a later slot allocation fails.
        self.segments.rotate_left(self.head);
        self.head = 0;
        let additional = count - self.segments.len();
        try_reserve(
            &mut self.segments,
            additional,
            scioxide_core::quantity!("history_segments"),
        )?;
        while self.segments.len() < count {
            let mut segment = DenseOutputStorage::new();
            segment.try_resize(self.initial.len(), 4)?;
            self.segments.push(segment);
        }
        Ok(())
    }
    /// The history scalars of `count` accepted slots and the staging segment.
    fn validate_size(dimension: usize, count: usize) -> Result<usize, SciOxideError> {
        let slots = checked_add(count, 1, scioxide_core::quantity!("history_segments"))?;
        checked_mul(
            slots,
            core::mem::size_of::<DenseOutputStorage<S>>(),
            scioxide_core::quantity!("history_segments"),
        )?;
        checked_mul(
            checked_mul(slots, dimension, scioxide_core::quantity!("history_states"))?,
            6,
            scioxide_core::quantity!("history_coefficients"),
        )
    }
    fn segment(&self, index: usize) -> &DenseOutputStorage<S> {
        &self.segments[(self.head + index) % self.segments.len()]
    }
    fn evaluate(&self, time: S, output: &mut [S]) -> Result<(), SciOxideError> {
        let mut lo = 0;
        let mut hi = self.len;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.segment(mid).interval().1 < time {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == self.len {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "delay query has no accepted history segment",
            ));
        }
        self.segment(lo).eval_into(time, output)
    }
}

/// Delay runtime seen by the driver. The runtime's [`OdeSystem`] counts its own work:
/// each history query as one `Function` evaluation, then the RHS as `RightHandSide`.
pub(crate) trait DelayRuntime<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> {
    /// Reset history for a new epoch and write the initial state, counting the initial
    /// history query in `cx`.
    fn initialize(
        &self,
        output: &mut Vector<T>,
        cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
    fn limit(&self, time: S) -> Result<S, SciOxideError>;
    fn reserve_step(&self, time: S) -> Result<bool, SciOxideError>;
    fn stage(&self, segment: &DenseOutputStorage<S, T>) -> Result<(), SciOxideError>;
    fn commit(&self);
    fn is_break(&self, time: S) -> bool;
    /// The attempt trace the driver reads; budget denials of the runtime's own work pass
    /// through it without building an error.
    fn trace(&self) -> &crate::stepper::StepTrace;
}

/// Delay runtime of one solve.
struct DdeRuntime<'a, S: RealScalar + NumericScalar<Real = S>> {
    plan: &'a DdePlan<S>,
    problem: &'a DdeProblem<'a, S>,
    storage: &'a RefCell<DdeStorage<S>>,
    delayed: &'a RefCell<Vec<S>>,
    trace: crate::stepper::StepTrace,
}
impl<S: RealScalar + NumericScalar<Real = S>> DelayRuntime<S, S> for DdeRuntime<'_, S> {
    fn initialize(
        &self,
        output: &mut Vector<S>,
        cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let mut storage = self.storage.borrow_mut();
        storage.head = 0;
        storage.len = 0;
        storage.peak = 0;
        storage.initial.fill(S::NAN);
        storage.initialized = false;
        // A run stopped before the initial history exists has no accepted state, so the
        // stop is an error here rather than an unsuccessful report.
        admit(cx, EvaluationRole::Function)?;
        self.problem
            .history
            .evaluate(self.plan.span.0, &mut storage.initial)?;
        crate::safety::finite(&storage.initial, scioxide_core::NonFiniteOrigin::Callback)?;
        output.as_mut_slice().copy_from_slice(&storage.initial);
        storage.initialized = true;
        Ok(())
    }
    fn limit(&self, time: S) -> Result<S, SciOxideError> {
        let next_break = self.plan.schedule.partition_point(|&t| t <= time);
        let boundary = self
            .plan
            .schedule
            .get(next_break)
            .copied()
            .unwrap_or(self.plan.span.1);
        let mut endpoint = (time + self.plan.minimum)
            .min(boundary)
            .min(self.plan.span.1);
        // Rounding of addition/subtraction must not admit a future history query.
        if endpoint - self.plan.minimum > time {
            endpoint = endpoint.next_after(time);
        }
        if endpoint <= time || endpoint - self.plan.minimum > time {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "method-of-steps boundary cannot make causal representable progress",
            ));
        }
        Ok(endpoint)
    }
    fn reserve_step(&self, time: S) -> Result<bool, SciOxideError> {
        let cutoff = checked_difference(time, self.plan.maximum)?;
        let mut storage = self.storage.borrow_mut();
        while storage.len > 0 && storage.segment(0).interval().1 <= cutoff {
            storage.head = (storage.head + 1) % storage.segments.len();
            storage.len -= 1;
        }
        let capacity = self.plan.options.history_capacity.get();
        if storage.len == capacity {
            return Ok(false);
        }
        if storage.len == storage.segments.len() {
            let count = storage
                .segments
                .len()
                .saturating_mul(2)
                .max(1)
                .min(capacity);
            storage.reserve_segments(count)?;
        }
        Ok(true)
    }
    fn stage(&self, segment: &DenseOutputStorage<S>) -> Result<(), SciOxideError> {
        self.storage.borrow_mut().pending.copy_from(segment)
    }
    fn commit(&self) {
        let mut storage = self.storage.borrow_mut();
        let index = (storage.head + storage.len) % storage.segments.len();
        let DdeStorage {
            segments, pending, ..
        } = &mut *storage;
        core::mem::swap(&mut segments[index], pending);
        storage.len += 1;
        storage.peak = storage.peak.max(storage.len);
    }
    fn is_break(&self, time: S) -> bool {
        let index = self.plan.schedule.partition_point(|&t| t < time);
        self.plan.schedule.get(index) == Some(&time)
    }
    fn trace(&self) -> &crate::stepper::StepTrace {
        &self.trace
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for DdeRuntime<'_, S> {
    fn dimension(&self) -> usize {
        self.plan.dimension
    }
    /// Self-metered: each history query counts as one `Function` evaluation and the RHS
    /// last as one `RightHandSide` evaluation, all in `cx`. A denial crosses the stepper
    /// through the driver's trace.
    fn rhs(
        &self,
        time: S,
        state: &[S],
        output: &mut [S],
        cx: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let storage = self.storage.borrow();
        let mut delayed = self.delayed.borrow_mut();
        for (index, &delay) in self.plan.delays.iter().enumerate() {
            let query = checked_difference(time, delay)?;
            if query >= time {
                return Err(invalid(
                    scioxide_core::argument!("self"),
                    "delay query must be strictly in the past",
                ));
            }
            let begin = index * self.plan.dimension;
            let values = &mut delayed[begin..begin + self.plan.dimension];
            if query < self.plan.span.0 {
                values.fill(S::NAN);
                admit(cx, EvaluationRole::Function)
                    .map_err(|fault| self.trace.denial.carry(fault))?;
                self.problem.history.evaluate(query, values)?;
                crate::safety::finite(values, scioxide_core::NonFiniteOrigin::Callback)?;
            } else if query == self.plan.span.0 {
                values.copy_from_slice(&storage.initial);
            } else {
                storage.evaluate(query, values)?;
            }
        }
        let view = MatrixView::from_slice(self.plan.delays.len(), self.plan.dimension, &delayed)?;
        admit(cx, EvaluationRole::RightHandSide).map_err(|fault| self.trace.denial.carry(fault))?;
        output.fill(S::NAN);
        self.problem.rhs.evaluate(time, state, view, output)
    }
}

/// Says why a delay integration stopped.
///
/// The first five variants have the same meaning in every time-evolution family (see
/// [termination](crate#termination)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DdeTermination {
    /// The integration reached `t_end`.
    ReachedEnd,
    /// A terminal event stopped the integration.
    TerminalEvent,
    /// A budget of the options or of an enclosing solve ran out. History calls count as
    /// `Function` evaluations.
    BudgetLimit(scioxide_core::BudgetKind),
    /// The step size fell below what can advance the time or below the minimum step.
    StepSizeLimit,
    /// Cancellation or a deadline stopped the integration.
    ExecutionStopped(scioxide_core::ExecutionStopReason),
    /// The next step would need more history than
    /// [`DdeOptions::history_capacity`] allows.
    HistoryCapacity,
}
crate::termination::shared_termination!(DdeTermination);

/// Reports the status, work and error diagnostics of one [`DdePlan::advance`] call.
///
/// States and events went to the output sink.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct DdeReport<S: RealScalar> {
    termination: DdeTermination,
    final_time: S,
    counts: SolveCounts,
    steps: EvaluationCounts,
    last_step_error_estimate: Option<S>,
    trust: PartialResultTrust,
    events: usize,
    terminal_event: Option<usize>,
    event_search: Option<EventSearchEvidence>,
    peak_history_segments: usize,
}
impl<S: RealScalar> DdeReport<S> {
    /// Returns why the integration stopped; always `Some`.
    pub const fn termination(&self) -> Option<DdeTermination> {
        Some(self.termination)
    }
    /// Returns `true` if the integration reached `t_end` or a terminal event.
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }
    /// Returns the time of the last accepted state.
    pub const fn final_time(&self) -> S {
        self.final_time
    }
    /// Returns the evaluations of this call, excluding nested solves: `Function` counts
    /// history calls and `RightHandSide` counts right-hand-side calls.
    pub const fn counts(&self) -> EvaluationCounts {
        self.counts.own()
    }
    /// Returns the work of this call and of nested solves.
    pub const fn accounting(&self) -> SolveCounts {
        self.counts
    }
    /// Returns the number of attempted steps, accepted or rejected.
    pub const fn attempted_steps(&self) -> usize {
        self.steps.proposals()
    }
    /// Returns the number of accepted steps.
    pub const fn accepted_steps(&self) -> usize {
        self.steps.accepted()
    }
    /// Returns the number of rejected steps.
    pub const fn rejected_steps(&self) -> usize {
        self.steps.rejected()
    }
    /// Returns the normalized local error estimate of the last attempted step.
    ///
    /// It does not include the error of the interpolated history.
    pub const fn last_step_error_estimate(&self) -> Option<S> {
        self.last_step_error_estimate
    }
    /// Returns the trust level of the last accepted state; it is not an accuracy
    /// certificate.
    pub const fn trust(&self) -> PartialResultTrust {
        self.trust
    }
    /// Returns the number of events sent to the sink.
    pub const fn events(&self) -> usize {
        self.events
    }
    /// Returns the index of the event function that stopped the integration, if any.
    pub const fn terminal_event(&self) -> Option<usize> {
        self.terminal_event
    }
    /// Returns the diagnostics of the event search, or `None` without events.
    pub const fn event_search(&self) -> Option<EventSearchEvidence> {
        self.event_search
    }
    /// Returns the largest number of accepted steps held as history at one time.
    pub const fn peak_history_segments(&self) -> usize {
        self.peak_history_segments
    }
}

/// Records the progress of a delay integration call, as kept in
/// [`DdeWorkspace::last_attempt`].
#[derive(Debug, Clone, PartialEq)]
pub struct DdeProgress<S: RealScalar> {
    ivp: OdeProgress<S>,
    peak_history_segments: usize,
}
impl<S: RealScalar + NumericScalar<Real = S>> DdeProgress<S> {
    /// Returns the time of the last accepted state.
    pub const fn time(&self) -> S {
        self.ivp.time()
    }
    /// Returns the last accepted state.
    pub fn state(&self) -> &[S] {
        self.ivp.state()
    }
    /// Returns the number of attempted steps, accepted or rejected.
    pub const fn attempted_steps(&self) -> usize {
        self.ivp.attempted_steps()
    }
    /// Returns the number of accepted steps.
    pub const fn accepted_steps(&self) -> usize {
        self.ivp.accepted_steps()
    }
    /// Returns the number of rejected steps.
    pub const fn rejected_steps(&self) -> usize {
        self.ivp.rejected_steps()
    }
    /// Returns the normalized local error estimate of the last attempted step.
    pub const fn last_step_error_estimate(&self) -> Option<S> {
        self.ivp.last_step_error_estimate()
    }
    /// Returns the trust level of the last accepted state.
    pub const fn trust(&self) -> PartialResultTrust {
        self.ivp.trust()
    }
    /// Returns the largest number of accepted steps held as history at one time.
    pub const fn peak_history_segments(&self) -> usize {
        self.peak_history_segments
    }
}

/// Holds the final state, report and trajectory of [`DdePlan::solve`].
#[derive(Debug, Clone)]
#[must_use]
pub struct DdeSolution<S: RealScalar + NumericScalar<Real = S>> {
    state: OdeState<S>,
    report: DdeReport<S>,
    trajectory: OdeTrajectory<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> DdeSolution<S> {
    /// Returns the last accepted state.
    pub const fn state(&self) -> &OdeState<S> {
        &self.state
    }
    /// Returns the report of the integration.
    pub const fn report(&self) -> &DdeReport<S> {
        &self.report
    }
    /// Returns the samples and events of the output schedule.
    pub const fn trajectory(&self) -> &OdeTrajectory<S> {
        &self.trajectory
    }
    /// Returns the final state, the report and the trajectory.
    pub fn into_parts(self) -> (OdeState<S>, DdeReport<S>, OdeTrajectory<S>) {
        (self.state, self.report, self.trajectory)
    }
}

/// Says how many event functions and history steps a [`DdeWorkspace`] should hold.
///
/// History storage doubles as needed during an integration, up to
/// [`DdeOptions::history_capacity`]. Request enough history steps to avoid that growth
/// in the first integration; later integrations reuse the storage. One more step of
/// storage is always added for the step in progress.
///
/// ```
/// use scioxide_core::{Budget, Capacity};
/// use scioxide_ode::{DdeOptions, DdePlan, DdeWorkspaceRequest};
///
/// let options = DdeOptions::new().with_history_capacity(Budget::try_from(1024)?).with_schedule_capacity(Budget::try_from(64)?);
/// let plan = DdePlan::new(1, (0.0_f64, 2.0), &[1.0], &[], options)?;
/// let request = DdeWorkspaceRequest::new(0).with_history_segments(32);
/// let workspace = plan.workspace(request)?;
/// let requirements = plan.workspace_requirements(request)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct DdeWorkspaceRequest {
    events: usize,
    history_segments: usize,
}
impl DdeWorkspaceRequest {
    /// Creates a request for `events` event functions and no history steps.
    pub const fn new(events: usize) -> Self {
        Self {
            events,
            history_segments: 0,
        }
    }
    /// Sets the number of history steps to allocate in advance; the plan checks it
    /// against its limit.
    pub const fn with_history_segments(mut self, count: usize) -> Self {
        self.history_segments = count;
        self
    }
    /// Returns the number of event functions.
    pub const fn events(&self) -> usize {
        self.events
    }
    /// Returns the number of history steps to allocate in advance.
    pub const fn history_segments(&self) -> usize {
        self.history_segments
    }
}

/// Describes the storage of a [`DdeWorkspaceRequest`], as returned by
/// [`DdePlan::workspace_requirements`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DdeRequirements {
    dimension: usize,
    delayed: usize,
    history_capacity: usize,
    history_segments: usize,
    events: usize,
    event_records: usize,
    bytes: usize,
}
impl DdeRequirements {
    /// Returns the state dimension.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    /// Returns the bytes of delay storage: six scalars per state component for each
    /// requested history step and the step in progress (four interpolation coefficients
    /// and both endpoints), plus the delayed values.
    ///
    /// Integrator stages, event storage and allocator overhead are not included.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Returns the largest number of accepted steps kept as history.
    pub const fn history_capacity(&self) -> usize {
        self.history_capacity
    }
    /// Returns the number of history steps to allocate in advance, excluding the step
    /// in progress.
    pub const fn history_segments(&self) -> usize {
        self.history_segments
    }
    /// Returns the number of event functions.
    pub const fn events(&self) -> usize {
        self.events
    }
}

/// Holds the Runge–Kutta, history and delayed-value storage of a [`DdePlan`], and the
/// record of the last call.
///
/// [`DdeWorkspace::new`] creates an empty workspace; [`DdePlan::workspace`] and
/// [`Self::try_reserve`] allocate its storage. History storage is kept between calls
/// of the same dimension, also for plans with a smaller history limit, and replaced
/// when the dimension changes. See [`DdeWorkspaceRequest`] and
/// [plans and workspaces](scioxide_core#plans-and-workspaces).
pub struct DdeWorkspace<S: RealScalar + NumericScalar<Real = S>> {
    ivp: IvpStorage<S, <DormandPrince5<S> as Stepper<S, S>>::Workspace, S>,
    history: RefCell<DdeStorage<S>>,
    delayed: RefCell<Vec<S>>,
    shell: Shell<DdeProgress<S>>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for DdeWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> core::fmt::Debug for DdeWorkspace<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DdeWorkspace")
            .field("dimension", &self.ivp.dimension)
            .field("last_attempt", &self.shell.last_attempt())
            .finish_non_exhaustive()
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> DdeWorkspace<S> {
    /// Creates an empty workspace without allocating.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ivp: IvpStorage::default(),
            history: RefCell::new(DdeStorage::empty()),
            delayed: RefCell::new(Vec::new()),
            shell: Shell::new(),
        }
    }
    fn fits(&self, requirements: &DdeRequirements) -> bool {
        self.ivp.fits(
            &DormandPrince5::new(),
            requirements.dimension,
            requirements.events,
            requirements.event_records,
            false,
        ) && self.history.borrow().fits(requirements)
            && self.delayed.borrow().len() == requirements.delayed
            && self
                .shell
                .has_spare(|progress| progress.ivp.state.len() == requirements.dimension)
    }
    /// Reports whether this workspace holds the storage of `requirements`.
    ///
    /// An integration still grows the history if it needs more steps than requested.
    ///
    /// # Errors
    ///
    /// Never returns an error.
    pub fn check(&self, requirements: &DdeRequirements) -> Result<Capacity, SciOxideError> {
        Ok(if self.fits(requirements) {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }
    /// Allocates the storage of `requirements`.
    ///
    /// # Errors
    ///
    /// `AllocationFailure` or `SizeOverflow` if the storage cannot be allocated.
    pub fn try_reserve(&mut self, requirements: &DdeRequirements) -> Result<(), SciOxideError> {
        self.ivp.reserve(
            &DormandPrince5::new(),
            requirements.dimension,
            requirements.events,
            requirements.event_records,
            false,
        )?;
        if self.history.get_mut().dimension != Some(requirements.dimension) {
            *self.history.get_mut() = DdeStorage::new(requirements)?;
        } else {
            self.history
                .get_mut()
                .reserve_segments(requirements.history_segments)?;
        }
        let delayed = self.delayed.get_mut();
        if delayed.len() != requirements.delayed {
            *delayed = try_with_capacity(
                requirements.delayed,
                scioxide_core::quantity!("delayed_values"),
            )?;
            delayed.resize(requirements.delayed, S::ZERO);
        }
        let dimension = requirements.dimension;
        self.shell.reserve_spare(
            |progress| progress.ivp.state.len() == dimension,
            || {
                Ok(DdeProgress {
                    ivp: OdeProgress::new(dimension)?,
                    peak_history_segments: 0,
                })
            },
        )
    }
    /// Returns the record of the last call: its outcome, work counts and progress.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<DdeProgress<S>>> {
        self.shell.last_attempt()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> DdePlan<S> {
    /// Returns the storage for the event functions and history steps of `request`, in
    /// O(1) time.
    ///
    /// The plan's history limit applies independently of this request. With zero
    /// history steps, all history storage is allocated as the integration needs it.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if `request` asks for more history steps than
    ///   [`DdeOptions::history_capacity`].
    /// - `SizeOverflow` if the storage size overflows `usize`.
    pub fn workspace_requirements(
        &self,
        request: DdeWorkspaceRequest,
    ) -> Result<DdeRequirements, SciOxideError> {
        let history_capacity = self.options.history_capacity.get();
        if request.history_segments > history_capacity {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("history_segments"),
                request.history_segments,
                history_capacity,
            ));
        }
        let history = DdeStorage::<S>::validate_size(self.dimension, request.history_segments)?;
        let delayed = checked_mul(
            self.dimension,
            self.delays.len(),
            scioxide_core::quantity!("delayed_values"),
        )?;
        let bytes = checked_mul(
            checked_add(history, delayed, scioxide_core::quantity!("delay_storage"))?,
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("delay_storage"),
        )?;
        Ok(DdeRequirements {
            dimension: self.dimension,
            delayed,
            history_capacity,
            history_segments: request.history_segments,
            events: request.events,
            event_records: self.options.ode.event_records(request.events)?,
            bytes,
        })
    }

    /// Allocates a workspace for `request`.
    ///
    /// Time and memory are proportional to the requested storage, not to the plan's
    /// history limit.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure` if the
    /// storage cannot be allocated.
    pub fn workspace(
        &self,
        request: DdeWorkspaceRequest,
    ) -> Result<DdeWorkspace<S>, SciOxideError> {
        let mut workspace = DdeWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    fn preflight(
        &self,
        problem: &DdeProblem<'_, S>,
        state: &OdeState<S>,
        workspace: &mut DdeWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        if state.values().len() != self.dimension {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.dimension),
                scioxide_core::Dimensions::one(state.values().len()),
            ));
        }
        if let Some(times) = self.options.ode.requested_times() {
            crate::ivp::validate_requested_times(
                times,
                self.span.0,
                self.span.1,
                scioxide_core::argument!("options"),
            )?;
        }
        let events = problem.events();
        if matches!(
            self.options.ode.event_options().search(),
            crate::EventSearchPolicy::DerivativeAssisted(_)
        ) && events.iter().any(|event| !event.has_total_derivative())
        {
            return Err(crate::ivp::unsupported(
                self.info.name,
                "derivative-assisted search needs every event's total derivative",
            ));
        }
        let requirements = self.workspace_requirements(DdeWorkspaceRequest::new(events.len()))?;
        if !workspace.fits(&requirements) {
            workspace.try_reserve(&requirements)?;
        }
        Ok(())
    }

    /// Integrates `problem` over the plan's span, writing the solution to `state` and
    /// the output to `sink`.
    ///
    /// Every call starts again from the initial history: after the input checks, `state`
    /// is set to `φ(t₀)`, and when the call returns it holds the last accepted state. A
    /// stopped integration cannot be resumed. If an input check fails, `state` is
    /// unchanged. A budget, the step size, the history limit, cancellation or a deadline
    /// ends the call with `Ok` and an unsuccessful [`DdeReport`]; see
    /// [termination](crate#termination). [`DdeWorkspace::last_attempt`] records the
    /// progress of every call. `scope` is an
    /// [`ExecutionContext`](scioxide_core#execution-contexts) or the solve context of an
    /// enclosing solve.
    ///
    /// History storage grows as needed up to the plan's limit and is kept for reuse. The
    /// call does not allocate if the workspace already holds the largest history the
    /// integration needs and the callbacks and sink do not allocate.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `state` does not have the plan's dimension.
    /// - `InvalidInput` if the requested output times are invalid for the span, or the
    ///   event search uses derivatives and an event function has no total derivative.
    /// - `NonFinite` (origin `Callback`) if the history or the right-hand side returns
    ///   NaN or infinity.
    /// - `ConvergenceFailure` or `ExecutionStopped` if a budget, cancellation or a
    ///   deadline stops the call before `φ(t₀)` is evaluated.
    /// - Errors returned by the history, the right-hand side, the events or the sink,
    ///   unchanged.
    pub fn advance<'scope, 'capability, K>(
        &self,
        problem: &DdeProblem<'_, S>,
        state: &mut OdeState<S>,
        sink: &mut K,
        workspace: &mut DdeWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<DdeReport<S>, SciOxideError>
    where
        K: OdeOutputSink<S> + ?Sized,
    {
        let scope = scope.into();
        if let Err(error) = self.preflight(problem, state, workspace) {
            return workspace.shell.reject(error);
        }
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("dde", "method-of-steps-dp5", 1),
            self.dimension as u128,
            0,
        );
        let DdeWorkspace {
            ivp,
            history,
            delayed,
            shell,
        } = workspace;
        let runtime = DdeRuntime {
            plan: self,
            problem,
            storage: history,
            delayed,
            trace: crate::stepper::StepTrace::default(),
        };
        let run = Drive {
            method: &self.method,
            info: self.info,
            options: &self.options.ode,
            tolerance: self.options.ode.error_tolerance(),
            system: &runtime,
            events: problem.events(),
            delay: Some(&runtime),
            provider: crate::stiff::stage::Provider::Native,
            end: self.span.1,
        };
        let (time, values) = state.parts_mut();
        let search =
            (!problem.events().is_empty()).then_some(self.options.ode.event_options().search());
        let executed = shell.execute(scope, identity, self.options.ode.budgets(), |cx| {
            *time = self.span.0;
            let outcome = drive(&run, ivp, time, values, sink, &mut (), cx)?;
            let termination = match outcome.finish {
                Finish::ReachedEnd => DdeTermination::ReachedEnd,
                Finish::TerminalEvent => DdeTermination::TerminalEvent,
                Finish::Stop(stop) => DdeTermination::from_stop(stop),
                Finish::HistoryCapacity => DdeTermination::HistoryCapacity,
            };
            let ode = crate::ivp::report(
                outcome,
                crate::OdeTermination::ReachedEnd,
                *time,
                ivp,
                cx,
                search,
            );
            Ok(DdeReport {
                termination,
                final_time: ode.final_time(),
                counts: ode.accounting(),
                steps: ode.steps,
                last_step_error_estimate: ode.last_step_error_estimate(),
                trust: ode.trust(),
                events: ode.events(),
                terminal_event: ode.terminal_event(),
                event_search: ode.event_search(),
                peak_history_segments: history.borrow().peak,
            })
        });
        let (peak, initialized) = {
            let history = history.borrow();
            (history.peak, history.initialized)
        };
        shell.conclude(executed, DdeReport::is_success, |recycled| {
            // Without initial history there is no accepted state to report.
            if !initialized {
                return None;
            }
            let ivp_progress = crate::ivp::progress(
                recycled.map(|progress| progress.ivp),
                *time,
                values.as_slice(),
                ivp,
            )?;
            Some(DdeProgress {
                ivp: ivp_progress,
                peak_history_segments: peak,
            })
        })
    }

    /// Integrates `problem` over the plan's span with a new workspace on the calling
    /// thread and returns the final state and an [`OdeTrajectory`] of the output.
    ///
    /// It runs [`Self::advance`] with [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`] and [`Self::advance`], wrapped in a
    /// [`PartialFailure`] together with the progress of the failed call.
    #[allow(
        clippy::result_large_err,
        reason = "owned conveniences return progress inside PartialFailure"
    )]
    pub fn solve(
        &self,
        problem: &DdeProblem<'_, S>,
    ) -> Result<DdeSolution<S>, PartialFailure<DdeProgress<S>>> {
        let mut workspace = self
            .workspace(DdeWorkspaceRequest::new(problem.events().len()))
            .map_err(PartialFailure::new)?;
        let mut state = OdeState::new(
            self.span.0,
            Vector::zeros(self.dimension).map_err(PartialFailure::new)?,
        );
        let mut trajectory = OdeTrajectory::new();
        match self.advance(
            problem,
            &mut state,
            &mut trajectory,
            &mut workspace,
            ExecutionContext::sequential_native(),
        ) {
            Ok(report) => Ok(DdeSolution {
                state,
                report,
                trajectory,
            }),
            Err(error) => Err(workspace.shell.into_failure(error)),
        }
    }
}

impl<S: RealScalar> Default for DdeOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Integrates `y′(t) = rhs(t, y(t), delayed)` with constant positive `delays`.
///
/// Row `j` of `delayed` is `y(t − delays[j])`; `history` supplies the state for
/// `t ≤ t0`. Uses the Dormand–Prince method of steps with default tolerances.
/// Schedule preparation uses `O(b d)` insertions for `b` propagated breaks and
/// `d` delays; retained history follows the largest delay.
///
/// # Errors
/// Returns `InvalidInput` for `t1 ≤ t0` or invalid delays, the errors of
/// [`DdePlan::new`] and [`DdePlan::advance`], and an error for every unsuccessful
/// termination. Failures retain the last accepted progress.
///
/// # Examples
/// ```
/// let y = scioxide_ode::solve_dde(
///     |_t: f64, _y: &[f64], d: scioxide_linalg::MatrixView<'_, f64>, dy: &mut [f64]| {
///         dy[0] = d.get(0, 0)?; Ok(())
///     }, |_t: f64, y: &mut [f64]| { y.fill(1.0); Ok(()) },
///     &[1.0], 0.0, 3.0, 1,
/// ).map_err(scioxide_core::SciOxideError::from)?;
/// assert!((y.as_slice()[0] - 37.0 / 6.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
/// # References
/// Bellen and Zennaro, *Numerical Methods for Delay Differential Equations*, §3.
#[allow(
    clippy::result_large_err,
    reason = "owned evolution retains accepted progress"
)]
pub fn solve_dde<S, F, H>(
    rhs: F,
    history: H,
    delays: &[S],
    t0: S,
    t1: S,
    dimension: usize,
) -> Result<Vector<S>, PartialFailure<DdeProgress<S>>>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: Fn(S, &[S], MatrixView<'_, S>, &mut [S]) -> Result<(), SciOxideError>,
    H: Fn(S, &mut [S]) -> Result<(), SciOxideError>,
{
    if RealScalar::is_finite(t0) && RealScalar::is_finite(t1) && t1 <= t0 {
        return Err(PartialFailure::new(SciOxideError::invalid_input(
            scioxide_core::argument!("t1"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "end time must exceed start time",
        )));
    }
    let plan = DdePlan::new(dimension, (t0, t1), delays, &[], DdeOptions::default())
        .map_err(PartialFailure::new)?;
    let problem = DdeProblem::new(&rhs, &history);
    let mut ws = plan
        .workspace(DdeWorkspaceRequest::new(0))
        .map_err(PartialFailure::new)?;
    let mut state = OdeState::new(t0, Vector::zeros(dimension).map_err(PartialFailure::new)?);
    match plan.advance(
        &problem,
        &mut state,
        &mut (),
        &mut ws,
        ExecutionContext::sequential_native(),
    ) {
        Ok(report) => {
            if let Some(stop) = report.termination().and_then(Terminal::stop) {
                return Err(ws
                    .shell
                    .into_failure(crate::termination::stop_error(stop, report.accounting())));
            }
            if !report.is_success() {
                return Err(ws.shell.into_failure(SciOxideError::resource_limit(
                    scioxide_core::quantity!("history_segments"),
                    report.peak_history_segments().saturating_add(1),
                    usize::MAX,
                )));
            }
            Ok(state.into_parts().1)
        }
        Err(error) => Err(ws.shell.into_failure(error)),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
}
