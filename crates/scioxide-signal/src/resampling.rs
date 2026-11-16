//! Polyphase FIR filtering and rational sample-rate conversion.
use crate::util::{
    capacity, check_axis_shape, empty, inconsistent, invalid, scalar_from_usize_rounded,
};
use crate::util::{copy, inputs as finite, zeros};
use crate::{SignalLayout, SignalRequirements, SignalWorkspace, StreamChunk, StreamProgress};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_add, checked_mul, gcd, try_filled};
use scioxide_core::{ExecutionContext, NdView, NdViewMut, NumericScalar, SciOxideError};

/// Time origin and output length of a [`PolyphasePlan`] on a finite record.
///
/// With `n` input samples, `L` taps and rates `up` and `down`:
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResampleAlignment {
    /// The first output is high-rate sample `phase` of the filtered signal, and
    /// the outputs run to the end of the FIR tail at high-rate sample
    /// `(n − 1) up + L − 1`.
    Causal,
    /// The first output is high-rate sample `(L − 1)/2`, the center of an odd
    /// symmetric filter, so the filter adds no delay. There are
    /// `⌈n up / down⌉` outputs.
    Centered,
}
/// Plan for FIR filtering with rational sample-rate conversion by the
/// polyphase method.
///
/// The plan upsamples the input `x` by `up`, inserting `up − 1` zeros after
/// each sample, filters with the taps `h` at the high rate, and keeps every
/// `down`-th sample:
///
/// ```text
/// v[t] = Σᵢ h[i] x[(t − i)/up],   y[j] = v[t₀ + j down],
/// ```
///
/// where the sum runs over the `i` for which `(t − i)/up` is an integer index
/// of the record, and the origin `t₀` is set by [`ResampleAlignment`]. The
/// polyphase form evaluates only the kept outputs, at a cost of about
/// `L / up` products per output for `L` taps. With `up = down = 1` this is an
/// ordinary causal FIR filter. The record methods
/// [`resample_into`](Self::resample_into) and
/// [`resample_axis_into`](Self::resample_axis_into) and the streams from
/// [`stream`](Self::stream) give the same results. Accuracy is described in
/// [Accuracy](crate#accuracy).
///
/// # References
///
/// R. E. Crochiere and L. R. Rabiner, *Multirate Digital Signal Processing*,
/// Prentice-Hall, 1983.
#[derive(Clone, Debug)]
pub struct PolyphasePlan<T: NumericScalar> {
    taps: Vec<T>,
    packed: Vec<T>,
    up: usize,
    down: usize,
    phase: usize,
    origin: usize,
    alignment: ResampleAlignment,
    history_len: usize,
}
impl<T: NumericScalar> PolyphasePlan<T> {
    /// Prepares a plan with the taps `taps` at the high rate, the rates `up` and
    /// `down` as given, and the decimation phase `phase`.
    ///
    /// The rates are not reduced by their common divisor; see
    /// [`rational`](Self::rational) for that. With [`ResampleAlignment::Causal`],
    /// the outputs are the high-rate samples `phase + j down`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `taps` is empty, `up` or `down` is zero, or `phase` is
    ///   not below `down`.
    /// - `InvalidInput` if the alignment is `Centered` and `phase` is not zero,
    ///   or `taps` does not have odd length and exact symmetry
    ///   `h[i] = h[L − 1 − i]`.
    /// - `NonFinite` if a tap is NaN or infinite.
    pub fn new(
        taps: &[T],
        up: usize,
        down: usize,
        phase: usize,
        alignment: ResampleAlignment,
    ) -> Result<Self, SciOxideError> {
        if taps.is_empty() {
            return Err(empty(
                scioxide_core::argument!("taps"),
                "at least one tap is required",
            ));
        }
        check_rates(up, down)?;
        if phase >= down {
            return Err(invalid(
                scioxide_core::argument!("phase"),
                "phase must be below the downsampling factor",
            ));
        }
        finite(taps)?;
        if alignment == ResampleAlignment::Centered {
            if phase != 0 {
                return Err(inconsistent(
                    scioxide_core::argument!("phase"),
                    "centered alignment requires phase zero",
                ));
            }
            if taps.len().is_multiple_of(2)
                || !taps.iter().zip(taps.iter().rev()).all(|(a, b)| a == b)
            {
                return Err(inconsistent(
                    scioxide_core::argument!("taps"),
                    "centered alignment requires exactly symmetric odd taps",
                ));
            }
        }
        let history_len = checked_add(
            (taps.len() - 1) / up,
            1,
            scioxide_core::quantity!("polyphase_history"),
        )?;
        let mut packed = Vec::new();
        if up != 1 {
            packed = scioxide_core::backend::try_with_capacity(
                taps.len(),
                scioxide_core::quantity!("polyphase_coefficients"),
            )?;
            for phase in 0..up.min(taps.len()) {
                packed.extend(taps[phase..].iter().step_by(up).copied());
            }
        }
        Ok(Self {
            taps: copy(taps)?,
            packed,
            up,
            down,
            phase,
            origin: if alignment == ResampleAlignment::Centered {
                (taps.len() - 1) / 2
            } else {
                phase
            },
            alignment,
            history_len,
        })
    }

    /// Borrow a contiguous polyphase row in the original tap order.
    fn phase_coefficients(&self, phase: usize) -> &[T] {
        if self.up == 1 {
            return &self.taps;
        }
        let length = (self.taps.len() - 1 - phase) / self.up + 1;
        let start = phase * (self.taps.len() / self.up) + phase.min(self.taps.len() % self.up);
        &self.packed[start..start + length]
    }

    /// Prepares a causal FIR filter `y[n] = Σᵢ h[i] x[n − i]`, without rate
    /// change.
    ///
    /// On a record of `n` samples it returns the full convolution of
    /// `n + L − 1` samples.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `taps` is empty.
    /// - `NonFinite` if a tap is NaN or infinite.
    pub fn fir(taps: &[T]) -> Result<Self, SciOxideError> {
        Self::new(taps, 1, 1, 0, ResampleAlignment::Causal)
    }
    /// Prepares a conversion by the rational factor `up / down` with your own
    /// filter taps.
    ///
    /// The rates are first divided by their greatest common divisor, and `taps`
    /// is applied at the reduced high rate. The decimation phase is zero.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`new`](Self::new).
    pub fn rational(
        taps: &[T],
        up: usize,
        down: usize,
        alignment: ResampleAlignment,
    ) -> Result<Self, SciOxideError> {
        check_rates(up, down)?;
        let divisor = gcd(up, down);
        Self::new(taps, up / divisor, down / divisor, 0, alignment)
    }
    /// Prepares a conversion by the rational factor `up / down` with a designed
    /// anti-aliasing low-pass filter.
    ///
    /// After the rates are divided by their greatest common divisor, the filter
    /// is a Kaiser-window design with `β = 5` and `2 · 10 · max(up, down) + 1`
    /// taps. Its cutoff is `1 / max(up, down)` in multiples of the high-rate
    /// Nyquist frequency, and its DC gain is `up`, which restores the amplitude
    /// lost to zero insertion. If the reduced rates are both one, the filter is
    /// the identity.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `up` or `down` is zero.
    /// - `SizeOverflow` if the filter length does not fit in `usize`.
    /// - `NumericalInstability` if a filter tap overflows.
    ///
    /// # References
    ///
    /// J. F. Kaiser, "Nonrecursive digital filter design using the I₀-sinh
    /// window function", *Proc. IEEE Int. Symp. Circuits and Systems*, 1974,
    /// pp. 20–23.
    pub fn rational_default(
        up: usize,
        down: usize,
        alignment: ResampleAlignment,
    ) -> Result<Self, SciOxideError> {
        check_rates(up, down)?;
        let divisor = gcd(up, down);
        let up = up / divisor;
        let down = down / divisor;
        if up == 1 && down == 1 {
            return Self::new(&[T::ONE], 1, 1, 0, alignment);
        }
        let half = checked_mul(
            10,
            up.max(down),
            scioxide_core::quantity!("resampling_filter"),
        )?;
        let length = checked_add(
            checked_mul(2, half, scioxide_core::quantity!("resampling_filter"))?,
            1,
            scioxide_core::quantity!("resampling_filter"),
        )?;
        let cutoff = T::Real::ONE / scalar_from_usize_rounded::<T::Real>(up.max(down))?;
        let design = crate::design_fir_window(
            length,
            crate::FirSymmetry::Even,
            &[crate::FirBand::new(
                T::Real::ZERO,
                cutoff,
                T::Real::ONE,
                T::Real::ONE,
            )],
            crate::WindowKind::Kaiser { beta: 5.0 },
            crate::FirNormalization::At {
                frequency: T::Real::ZERO,
                gain: scalar_from_usize_rounded::<T::Real>(up)?,
            },
        )?;
        let mut taps = try_filled(length, T::ZERO, scioxide_core::quantity!("resampling_taps"))?;
        taps.iter_mut()
            .zip(design.coefficients())
            .for_each(|(t, &h)| *t = T::from_real(h));
        Self::new(&taps, up, down, 0, alignment)
    }
    /// Returns the upsampling factor.
    pub fn up(&self) -> usize {
        self.up
    }
    /// Returns the downsampling factor.
    pub fn down(&self) -> usize {
        self.down
    }
    /// Returns the decimation phase.
    pub fn phase(&self) -> usize {
        self.phase
    }
    /// Returns the filter taps at the high rate, including the DC gain of a
    /// [`rational_default`](Self::rational_default) design.
    pub fn coefficients(&self) -> &[T] {
        &self.taps
    }
    /// Returns the number of output samples for a record of `input_len`
    /// samples.
    ///
    /// It is zero for an empty record. See [`ResampleAlignment`] for the other
    /// lengths.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the count does not fit in `usize`.
    pub fn output_len(&self, input_len: usize) -> Result<usize, SciOxideError> {
        if input_len == 0 {
            return Ok(0);
        }
        if self.alignment == ResampleAlignment::Centered {
            let ticks = checked_mul(
                input_len,
                self.up,
                scioxide_core::quantity!("resampled_length"),
            )?;
            Ok(ticks / self.down + usize::from(ticks % self.down != 0))
        } else {
            let last = checked_add(
                checked_mul(
                    input_len - 1,
                    self.up,
                    scioxide_core::quantity!("fir_duration"),
                )?,
                self.taps.len() - 1,
                scioxide_core::quantity!("fir_tail"),
            )?;
            if last < self.origin {
                Ok(0)
            } else {
                checked_add(
                    (last - self.origin) / self.down,
                    1,
                    scioxide_core::quantity!("polyphase_output_count"),
                )
            }
        }
    }

    fn available(&self, seen: usize) -> Result<usize, SciOxideError> {
        if seen == 0 {
            return Ok(0);
        }
        let last = checked_mul(seen, self.up, scioxide_core::quantity!("polyphase_clock"))? - 1;
        Ok((if last < self.origin {
            0
        } else {
            (last - self.origin) / self.down + 1
        })
        .min(self.output_len(seen)?))
    }
    /// Starts a stream at the beginning of a record, with zero history.
    ///
    /// The stream allocates `(L − 1) / up + 1` samples of history for `L`
    /// taps.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the allocator fails.
    pub fn stream(&self) -> Result<PolyphaseStream<'_, T>, SciOxideError> {
        Ok(PolyphaseStream {
            plan: self,
            ring: zeros(self.history_len)?,
            seen: 0,
            emitted: 0,
            finished: false,
        })
    }
    /// Returns the workspace that one call to
    /// [`resample_into`](Self::resample_into) (for a contiguous layout) or
    /// [`resample_axis_into`](Self::resample_axis_into) (for an axis layout)
    /// needs.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the axis is not below the rank of the shape, or the
    ///   shape is zero along it.
    /// - `SizeOverflow` if the size does not fit in `usize`.
    pub fn workspace_requirements(
        &self,
        layout: SignalLayout<'_>,
    ) -> Result<SignalRequirements, SciOxideError> {
        match layout {
            SignalLayout::Contiguous(n) => Ok(SignalRequirements::values(self.output_len(n)?)),
            SignalLayout::Axis { shape, axis } => {
                self.axis_requirements(shape, axis, scioxide_core::argument!("layout"))
            }
        }
    }
    /// Allocates a workspace for
    /// [`workspace_requirements(layout)`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// Returns the errors of
    /// [`workspace_requirements`](Self::workspace_requirements), and
    /// `AllocationFailure` if the allocator fails.
    pub fn workspace(&self, layout: SignalLayout<'_>) -> Result<SignalWorkspace<T>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(layout)?)
    }
    /// Filters and resamples a whole record, writing
    /// [`output_len(input.len())`](Self::output_len) samples to the start of
    /// `output`.
    ///
    /// The result equals the concatenated outputs of a stream fed the same
    /// record.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_signal::{PolyphasePlan, SignalLayout};
    /// let plan = PolyphasePlan::fir(&[0.5_f64, 0.5])?;
    /// let mut workspace = plan.workspace(SignalLayout::Contiguous(3))?;
    /// let mut output = [0.0; 4];
    /// plan.resample_into(
    ///     &[2.0, 4.0, 6.0],
    ///     &mut output,
    ///     &mut workspace,
    ///     ExecutionContext::sequential_native(),
    /// )?;
    /// // Two-point moving average, including the one-sample tail.
    /// assert_eq!(output, [1.0, 3.0, 5.0, 3.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is shorter than the output count, or
    ///   `workspace` is too small.
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` if an output overflows.
    /// - `ExecutionStopped` if `context` is cancelled or past its deadline.
    pub fn resample_into(
        &self,
        input: &[T],
        output: &mut [T],
        workspace: &mut SignalWorkspace<T>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = self.output_len(input.len())?;
        capacity(n, output.len())?;
        let staged = workspace.buffers(&SignalRequirements::values(n))?.values;
        finite(input)?;
        context.check()?;
        for (i, slot) in staged.iter_mut().enumerate() {
            let tick = checked_add(
                self.origin,
                checked_mul(i, self.down, scioxide_core::quantity!("polyphase_tick"))?,
                scioxide_core::quantity!("polyphase_origin"),
            )?;
            let phase = tick % self.up;
            if phase >= self.taps.len() {
                *slot = T::ZERO;
                continue;
            }
            let latest = tick / self.up;
            let tap_count = (self.taps.len() - 1 - phase) / self.up + 1;
            let first = latest.saturating_sub(tap_count - 1);
            let last = latest.min(input.len() - 1);
            if first > last {
                *slot = T::ZERO;
                continue;
            }
            let taps = self.phase_coefficients(phase);
            *slot = crate::util::contiguous_products::<T, T, true>(
                &input[first..=last],
                &taps[latest - last..latest - first + 1],
                core::convert::identity,
            )?;
        }
        context.check()?;
        output[..n].copy_from_slice(staged);
        Ok(())
    }

    /// Returns the filtered and resampled record, like
    /// [`resample_into`](Self::resample_into).
    ///
    /// This allocates the output and a workspace, and runs with
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` if an output overflows.
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn resample(&self, input: &[T]) -> Result<Vec<T>, SciOxideError> {
        let mut output = try_filled(
            self.output_len(input.len())?,
            T::ZERO,
            scioxide_core::quantity!("resampling_output"),
        )?;
        let mut workspace = self.workspace(SignalLayout::Contiguous(input.len()))?;
        self.resample_into(
            input,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}
/// Converts the sample rate of a record by the rational factor `up / down`.
///
/// The record is upsampled by `up`, filtered with the anti-aliasing low-pass
/// design of [`PolyphasePlan::rational_default`] (a Kaiser window with `β = 5`
/// and `20·max(up, down) + 1` taps after the rates are divided by their greatest
/// common divisor, DC gain `up`), and downsampled by `down`. The alignment is
/// fixed to [`ResampleAlignment::Centered`]: the alignment has no neutral
/// default, and the centered one makes the filter add no delay, so output `j`
/// lies at input time `j·down/up`. A record of `n` samples gives
/// `⌈n·up/down⌉` outputs, an empty record gives none, and `up == down` returns
/// the record. The polyphase form costs about `L/up` products per output for
/// `L` taps, and each output meets the product-sum bound of
/// [Accuracy](crate#accuracy). [`PolyphasePlan`] takes other taps or alignments
/// and reuses the filter across records. This function runs with
/// [`ExecutionContext::sequential_native`].
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) if `up` or `down` is zero.
/// - `NonFinite` if `input` contains NaN or infinity.
/// - `NumericalInstability` if a filter tap or an output overflows.
/// - `SizeOverflow` or `AllocationFailure` if the filter or the output cannot
///   be sized or allocated.
///
/// # Examples
///
/// ```
/// use core::f64::consts::TAU;
/// // Resampling a slow sine by 3/2 samples the same sine at times 2j/3.
/// let x: Vec<f64> = (0..120).map(|k| (TAU * 0.02 * k as f64).sin()).collect();
/// let y = scioxide_signal::resample(&x, 3, 2)?;
/// assert_eq!(y.len(), 180);
/// for (j, value) in y.iter().enumerate().take(150).skip(30) {
///     assert!((value - (TAU * 0.02 * j as f64 * 2.0 / 3.0).sin()).abs() < 2e-3);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. E. Crochiere and L. R. Rabiner, *Multirate Digital Signal Processing*,
/// Prentice-Hall, 1983.
pub fn resample<T: NumericScalar>(
    input: &[T],
    up: usize,
    down: usize,
) -> Result<Vec<T>, SciOxideError> {
    PolyphasePlan::rational_default(up, down, ResampleAlignment::Centered)?.resample(input)
}
/// Require positive rates `up` and `down`.
fn check_rates(up: usize, down: usize) -> Result<(), SciOxideError> {
    if up == 0 {
        return Err(invalid(
            scioxide_core::argument!("up"),
            "upsampling factor must be positive",
        ));
    }
    if down == 0 {
        return Err(invalid(
            scioxide_core::argument!("down"),
            "downsampling factor must be positive",
        ));
    }
    Ok(())
}
/// Stream of a [`PolyphasePlan`], holding the most recent input samples that
/// later outputs need.
///
/// Each call writes every output that the input so far determines. The
/// [`StreamChunk::Final`] call also writes the outputs that depend on the zero
/// extension past the end of the record and closes the stream. The outputs of
/// all calls together equal [`PolyphasePlan::resample`] of the whole record.
/// Cloning a stream takes a snapshot, and `clone_from` restores one. See
/// [Streams](crate#streams).
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{PolyphasePlan, ResampleAlignment, StreamChunk};
/// let plan = PolyphasePlan::rational(&[1.0_f64], 1, 2, ResampleAlignment::Causal)?;
/// let mut stream = plan.stream()?;
/// let mut workspace = stream.workspace(4)?;
/// let mut output = [0.0; 4];
/// let progress = stream.process_into(
///     &[1.0, 2.0, 3.0, 4.0],
///     StreamChunk::Final,
///     &mut output,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// // Downsampling by two without filtering keeps every other sample.
/// assert_eq!(&output[..progress.produced()], &[1.0, 3.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct PolyphaseStream<'p, T: NumericScalar> {
    plan: &'p PolyphasePlan<T>,
    ring: Vec<T>,
    seen: usize,
    emitted: usize,
    finished: bool,
}
impl<T: NumericScalar> Clone for PolyphaseStream<'_, T> {
    fn clone(&self) -> Self {
        Self {
            plan: self.plan,
            ring: self.ring.clone(),
            seen: self.seen,
            emitted: self.emitted,
            finished: self.finished,
        }
    }
    fn clone_from(&mut self, source: &Self) {
        self.plan = source.plan;
        self.ring.clone_from(&source.ring);
        self.seen = source.seen;
        self.emitted = source.emitted;
        self.finished = source.finished;
    }
}
impl<T: NumericScalar> PolyphaseStream<'_, T> {
    /// Returns the stream to the beginning of a new record, with zero history,
    /// and reopens it.
    pub fn reset(&mut self) {
        self.ring.fill(T::ZERO);
        self.seen = 0;
        self.emitted = 0;
        self.finished = false;
    }
    /// Returns the number of samples that a call with `input_len` input samples
    /// writes.
    ///
    /// For [`StreamChunk::Partial`] this counts the outputs that the input so
    /// far determines. For [`StreamChunk::Final`] it counts every remaining
    /// output of the record.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the stream is closed.
    /// - `SizeOverflow` if the count does not fit in `usize`.
    pub fn output_len(&self, input_len: usize, chunk: StreamChunk) -> Result<usize, SciOxideError> {
        if self.finished {
            return Err(inconsistent(
                scioxide_core::argument!("self"),
                "stream is closed until reset",
            ));
        }
        let seen = checked_add(
            self.seen,
            input_len,
            scioxide_core::quantity!("stream_length"),
        )?;
        let total = match chunk {
            StreamChunk::Final => self.plan.output_len(seen)?,
            StreamChunk::Partial => self.plan.available(seen)?,
        };
        total.checked_sub(self.emitted).ok_or_else(|| {
            inconsistent(
                scioxide_core::argument!("self"),
                "stream output clock is inconsistent",
            )
        })
    }
    /// Returns a workspace size that serves every later call with at most
    /// `max_input_len` input samples, whatever the chunk kind and the state of
    /// the stream.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    pub fn workspace_requirements(
        &self,
        max_input_len: usize,
    ) -> Result<SignalRequirements, SciOxideError> {
        let quantity = scioxide_core::quantity!("stream_scratch");
        let ticks = checked_add(
            checked_mul(max_input_len, self.plan.up, quantity)?,
            self.plan.taps.len(),
            quantity,
        )?;
        self.requirements(checked_add(ticks / self.plan.down, 2, quantity)?)
    }
    /// Allocates a workspace for
    /// [`workspace_requirements(max_input_len)`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    /// - `AllocationFailure` if the allocator fails.
    pub fn workspace(&self, max_input_len: usize) -> Result<SignalWorkspace<T>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(max_input_len)?)
    }
    fn requirements(&self, output: usize) -> Result<SignalRequirements, SciOxideError> {
        Ok(SignalRequirements::values(checked_add(
            self.ring.len(),
            output,
            scioxide_core::quantity!("stream_scratch"),
        )?))
    }

    /// Filters the next chunk of the record and writes the completed output
    /// samples to the start of `output`.
    ///
    /// The call writes [`output_len(input.len(), chunk)`](Self::output_len)
    /// samples. An empty chunk is valid. [`StreamChunk::Final`] closes the
    /// stream until [`reset`](Self::reset).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the stream is closed.
    /// - `DimensionMismatch` if `output` is shorter than the output count, or
    ///   `workspace` is too small.
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` if an output overflows.
    /// - `ExecutionStopped` if `context` is cancelled or past its deadline.
    pub fn process_into(
        &mut self,
        input: &[T],
        chunk: StreamChunk,
        output: &mut [T],
        workspace: &mut SignalWorkspace<T>,
        context: ExecutionContext<'_>,
    ) -> Result<StreamProgress, SciOxideError> {
        let count = self.output_len(input.len(), chunk)?;
        capacity(count, output.len())?;
        let buffers = workspace.buffers(&self.requirements(count)?)?;
        finite(input)?;
        context.check()?;
        let (ring, staged) = buffers.values.split_at_mut(self.ring.len());
        ring.copy_from_slice(&self.ring);
        let (seen, emitted) = advance(
            self.plan,
            ring,
            (self.seen, self.emitted),
            input,
            chunk,
            staged,
            context,
        )?;
        context.check()?;
        output[..count].copy_from_slice(staged);
        self.ring.copy_from_slice(ring);
        let progress = StreamProgress {
            replacements: 0,
            consumed: input.len(),
            produced: count,
            first_output_index: self.emitted,
        };
        self.seen = seen;
        self.emitted = emitted;
        self.finished = chunk == StreamChunk::Final;
        Ok(progress)
    }
}
/// Push `input` through `ring` from clock `(seen, emitted)` and stage exactly
/// `staged.len()` outputs; returns the advanced clock. Callers validate input
/// and size `staged` with the stream output count.
fn advance<T: NumericScalar>(
    plan: &PolyphasePlan<T>,
    ring: &mut [T],
    (mut seen, mut emitted): (usize, usize),
    input: &[T],
    chunk: StreamChunk,
    staged: &mut [T],
    context: ExecutionContext<'_>,
) -> Result<(usize, usize), SciOxideError> {
    let first = emitted;
    let end = checked_add(
        emitted,
        staged.len(),
        scioxide_core::quantity!("output_clock"),
    )?;
    for &sample in input {
        ring[seen % ring.len()] = sample;
        seen += 1;
        let available = plan.available(seen)?.min(end);
        while emitted < available {
            context.check()?;
            staged[emitted - first] = sample_at(plan, ring, seen, emitted)?;
            emitted += 1;
        }
    }
    if chunk == StreamChunk::Final {
        while emitted < end {
            context.check()?;
            staged[emitted - first] = sample_at(plan, ring, seen, emitted)?;
            emitted += 1;
        }
    }
    Ok((seen, emitted))
}
fn sample_at<T: NumericScalar>(
    plan: &PolyphasePlan<T>,
    ring: &[T],
    seen: usize,
    output: usize,
) -> Result<T, SciOxideError> {
    let tick = checked_add(
        plan.origin,
        checked_mul(
            output,
            plan.down,
            scioxide_core::quantity!("polyphase_tick"),
        )?,
        scioxide_core::quantity!("polyphase_origin"),
    )?;
    let phase = tick % plan.up;
    if seen == 0 || phase >= plan.taps.len() {
        return Ok(T::ZERO);
    }
    let latest = tick / plan.up;
    let tap_count = (plan.taps.len() - 1 - phase) / plan.up + 1;
    let first = latest
        .saturating_sub(tap_count - 1)
        .max(seen.saturating_sub(ring.len()));
    let last = latest.min(seen - 1);
    if first > last {
        return Ok(T::ZERO);
    }
    let position = last % ring.len();
    let samples = ring[..=position]
        .iter()
        .rev()
        .chain(ring[position + 1..].iter().rev())
        .take(last - first + 1);
    let taps = plan.phase_coefficients(phase)[latest - last..].iter();
    crate::util::working_sum_products(samples.zip(taps).map(|(&x, &h)| (x, h)))
}

impl<T: NumericScalar> PolyphasePlan<T> {
    /// Axis workspace: dense staged output, one lane, its history and its
    /// result. `shape_argument` names the parameter that supplied `shape`.
    fn axis_requirements(
        &self,
        shape: &[usize],
        axis: usize,
        shape_argument: scioxide_core::Argument,
    ) -> Result<SignalRequirements, SciOxideError> {
        if axis >= shape.len() {
            return Err(invalid(
                scioxide_core::argument!("axis"),
                "axis must be below the shape rank",
            ));
        }
        if shape[axis] == 0 {
            return Err(empty(shape_argument, "selected axis must be nonempty"));
        }
        let input = scioxide_core::Shape::new(shape)?.len();
        let n = shape[axis];
        let m = self.output_len(n)?;
        let output = checked_mul(
            input / n,
            m,
            scioxide_core::quantity!("resampling_channel_output"),
        )?;
        let values = [output, n, self.history_len, m]
            .into_iter()
            .try_fold(0, |total, v| {
                checked_add(
                    total,
                    v,
                    scioxide_core::quantity!("resampling_channel_workspace"),
                )
            })?;
        Ok(SignalRequirements::values(values))
    }
    /// Filters and resamples every lane of `input` along `axis`, writing the
    /// results to `output`.
    ///
    /// `output` has the shape of `input` with the extent `n` along `axis`
    /// replaced by [`output_len(n)`](Self::output_len). Each lane is an
    /// independent record. `workspace` must satisfy
    /// [`workspace_requirements`](Self::workspace_requirements) for
    /// [`SignalLayout::Axis`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `axis` is not below the rank of `input`, or `input`
    ///   is empty along it.
    /// - `DimensionMismatch` if `output` does not have the expected shape, or
    ///   `workspace` is too small.
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` if an output overflows.
    /// - `ExecutionStopped` if `context` is cancelled or past its deadline.
    pub fn resample_axis_into(
        &self,
        input: NdView<'_, '_, T>,
        axis: usize,
        mut output: NdViewMut<'_, '_, T>,
        workspace: &mut SignalWorkspace<T>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let shape = input.layout().shape().extents();
        let requirements =
            self.axis_requirements(shape, axis, scioxide_core::argument!("input"))?;
        let n = shape[axis];
        let m = self.output_len(n)?;
        check_axis_shape(shape, axis, m, output.layout().shape().extents())?;
        let buffers = workspace.buffers(&requirements)?;
        context.check()?;
        let no = output.layout().shape().len();
        let (staged, rest) = buffers.values.split_at_mut(no);
        let (line, rest) = rest.split_at_mut(n);
        let (ring, result) = rest.split_at_mut(self.history_len);
        let geometry = crate::axis::Geometry::new(shape, axis, n, m)?;
        for batch in 0..geometry.batches {
            geometry.gather(&input, line, batch)?;
            finite(line)?;
            ring.fill(T::ZERO);
            advance(
                self,
                ring,
                (0, 0),
                line,
                StreamChunk::Final,
                result,
                context,
            )?;
            geometry.scatter(result, batch, staged)?;
        }
        context.check()?;
        crate::axis::publish(staged, &mut output);
        Ok(())
    }
}
