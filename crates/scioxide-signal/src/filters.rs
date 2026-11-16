//! Sliding-window median, rank, Wiener and Hampel filters, in batch and streaming form.
use crate::util::{
    capacity, check_real_slice_finite, clamped_index, empty, inconsistent, invalid,
    scalar_from_usize_rounded as scalar, zeros,
};
use crate::{SignalRequirements, SignalWorkspace, StreamChunk, StreamProgress};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, try_filled, try_with_capacity};
use scioxide_core::{Argument, ExecutionContext, RealScalar, SciOxideError};
use scioxide_core::{PromotingCompensatedSum, Scaled, ScaledAccumulator};

mod moments;
mod order;
/// Operation a [`LocalFilterPlan`] applies to each window.
///
/// For output sample `i`, the window holds the `w` samples `x[i + o]` for the
/// plan's offsets `o`. Samples beyond the ends of the record repeat the nearest
/// end sample.
#[derive(Clone, Copy, Debug)]
pub enum LocalFilter<S: RealScalar> {
    /// Median of the window.
    Median,
    /// Order statistic of the window: the value at zero-based position `rank`
    /// after sorting in increasing order.
    Rank {
        /// Zero-based rank, less than the window size `w`.
        rank: usize,
    },
    /// Adaptive Wiener filter with local mean `μ` and population variance `σ²`:
    ///
    /// ```text
    /// y = x + (μ − x) · ν/σ²   if σ² > ν,      y = μ   otherwise,
    /// ```
    ///
    /// where `x` is the center sample and `ν` the noise variance. With `ν = 0`
    /// the output equals the input. [`estimate_wiener_noise`] estimates `ν`.
    Wiener {
        /// Noise variance `ν ≥ 0`, in squared input units.
        noise_variance: S,
    },
    /// Hampel identifier: replaces the center sample `x` by the window median `m`
    /// when `|x − m| > t · 1.4826 · MAD`, where `MAD` is the median of `|xⱼ − m|`
    /// over the window. The factor 1.4826 makes `1.4826 · MAD` a consistent
    /// estimate of the standard deviation of Gaussian data.
    Hampel {
        /// Threshold multiplier `t ≥ 0`.
        threshold: S,
    },
}
/// Applies a median filter over the centered window of `2·radius + 1` samples.
///
/// Output `i` is the median of `x[i − radius], …, x[i + radius]`, where samples
/// beyond the ends of the record repeat the nearest end sample. The outputs are
/// input samples, so the filter is exact. Radius zero returns the record, and an
/// empty record gives an empty result. This is [`LocalFilterPlan::new`] with
/// [`LocalFilter::Median`] followed by [`LocalFilterPlan::filter`]. For the
/// window size `w = 2·radius + 1` it costs `O(w log w)` for the first window and
/// `O(w)` per later sample. Rank, Wiener and Hampel filters and streams use the
/// plan. This function runs with [`ExecutionContext::sequential_native`].
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) if `radius` exceeds `isize::MAX`.
/// - `NonFinite` if a sample is NaN or infinite.
/// - `SizeOverflow` or `AllocationFailure` if the window, the workspace or the
///   result cannot be sized or allocated.
///
/// # Examples
///
/// ```
/// // The isolated spike is replaced by the median of its neighborhood.
/// let filtered = scioxide_signal::median_filter(&[1.0_f64, 2.0, 9.0, 3.0, 4.0], 1)?;
/// assert_eq!(filtered, [1.0, 2.0, 3.0, 4.0, 4.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn median_filter<S: RealScalar>(input: &[S], radius: usize) -> Result<Vec<S>, SciOxideError> {
    LocalFilterPlan::new(radius, LocalFilter::Median)?.filter(input)
}
/// Sliding-window filter over a fixed set of sample offsets.
///
/// [`LocalFilterPlan::new`] uses the centered window of `2r + 1` samples for
/// every [`LocalFilter`]. [`LocalFilterPlan::order`] takes any offsets for a
/// rank filter. Records are filtered with [`LocalFilterPlan::filter`] or
/// [`LocalFilterPlan::filter_into`], and unbounded sequences with a
/// [`LocalFilterStream`].
///
/// Median, rank and Hampel filters keep the window sorted and update it as it
/// slides: `O(w log w)` for the first window, then `O(w)` per sample for a
/// window of `w` contiguous offsets. Their outputs are input samples, so they are
/// exact. The Wiener filter updates compensated running sums of the window and
/// recomputes them at least every `w` samples, which costs amortized `O(n + w)`
/// for `n` samples. Its rounding can differ from moments computed separately for each
/// window, and a constant window returns its center sample exactly.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{LocalFilter, LocalFilterPlan};
/// let plan = LocalFilterPlan::new(1, LocalFilter::Median)?;
/// let mut output = [0.; 3];
/// let mut workspace = plan.workspace(3)?;
/// let cx = ExecutionContext::sequential_native();
/// plan.filter_into(&[1., 9., 1.], &mut output, &mut [], &mut workspace, cx)?;
/// // Each three-sample window, with repeated ends, holds two ones.
/// assert_eq!(output, [1.; 3]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. S. Lim, *Two-Dimensional Signal and Image Processing*, Prentice Hall, 1990
/// (adaptive Wiener filter).
///
/// F. R. Hampel, "The influence curve and its role in robust estimation",
/// *J. Amer. Statist. Assoc.* 69(346), 383–393, 1974.
#[derive(Clone, Debug)]
pub struct LocalFilterPlan<S: RealScalar> {
    offsets: Vec<isize>,
    transitions: Vec<(isize, isize)>,
    // Footprints with more offset runs than about log2(w) re-sort each window:
    // run-by-run replacement would cost O(runs * w) per advance.
    resort: bool,
    factor: S,
    kind: LocalFilter<S>,
    left: usize,
    right: usize,
}
impl<S: RealScalar> LocalFilterPlan<S> {
    /// Prepares a filter over the centered window of `2·radius + 1` samples.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `radius` exceeds `isize::MAX`, a rank is not less than
    ///   the window size, or a noise variance or threshold is negative.
    /// - `NonFinite` if a noise variance or threshold is NaN or infinite.
    /// - `SizeOverflow` if the window size overflows `usize`.
    /// - `AllocationFailure` if the offsets cannot be stored.
    pub fn new(radius: usize, kind: LocalFilter<S>) -> Result<Self, SciOxideError> {
        let r = isize::try_from(radius).map_err(|_| {
            invalid(
                scioxide_core::argument!("radius"),
                "radius exceeds the signed offset range",
            )
        })?;
        let len = scioxide_core::backend::checked_add(
            scioxide_core::backend::checked_add(
                radius,
                radius,
                scioxide_core::quantity!("filter_workspace"),
            )?,
            1,
            scioxide_core::quantity!("filter_workspace"),
        )?;
        let mut offsets = try_with_capacity(len, scioxide_core::quantity!("filter_offsets"))?;
        offsets.extend(-r..=r);
        Self::build(offsets, kind, scioxide_core::argument!("kind"))
    }
    /// Prepares a rank filter over the given window offsets.
    ///
    /// Output sample `i` is the zero-based `rank`-th smallest of `x[i + o]` over
    /// `o` in `offsets`. The offsets may be asymmetric and need not include zero.
    /// When the offsets form more than `⌊log₂ w⌋ + 1` runs of consecutive
    /// integers, each window is sorted afresh in `O(w log w)`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `offsets` is empty or not strictly increasing, or
    ///   `rank` is not less than `offsets.len()`.
    /// - `SizeOverflow` if the window span overflows `usize`.
    /// - `AllocationFailure` if the offsets cannot be stored.
    pub fn order(offsets: &[isize], rank: usize) -> Result<Self, SciOxideError> {
        if offsets.is_empty() {
            return Err(empty(
                scioxide_core::argument!("offsets"),
                "offsets must be nonempty",
            ));
        }
        if offsets.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid(
                scioxide_core::argument!("offsets"),
                "offsets must be strictly increasing",
            ));
        }
        let mut owned =
            try_with_capacity(offsets.len(), scioxide_core::quantity!("order_offsets"))?;
        owned.extend_from_slice(offsets);
        Self::build(
            owned,
            LocalFilter::Rank { rank },
            scioxide_core::argument!("rank"),
        )
    }
    /// Validate `kind`; its parameters are
    /// attributed to `kind_argument`, the caller parameter that supplied them.
    fn build(
        offsets: Vec<isize>,
        kind: LocalFilter<S>,
        kind_argument: Argument,
    ) -> Result<Self, SciOxideError> {
        match kind {
            LocalFilter::Wiener { noise_variance } => check_real_slice_finite(&[noise_variance])?,
            LocalFilter::Hampel { threshold } => check_real_slice_finite(&[threshold])?,
            _ => (),
        }
        match kind {
            LocalFilter::Rank { rank } if rank >= offsets.len() => {
                return Err(invalid(kind_argument, "rank must be below the window size"));
            }
            LocalFilter::Wiener { noise_variance }
                if !noise_variance.is_finite() || noise_variance < S::ZERO =>
            {
                return Err(invalid(
                    kind_argument,
                    "Wiener noise variance must be finite and nonnegative",
                ));
            }
            LocalFilter::Hampel { threshold } if !threshold.is_finite() || threshold < S::ZERO => {
                return Err(invalid(
                    kind_argument,
                    "Hampel threshold must be finite and nonnegative",
                ));
            }
            _ => (),
        }
        let left = offsets
            .iter()
            .filter(|&&i| i < 0)
            .map(|i| i.unsigned_abs())
            .max()
            .unwrap_or(0);
        let right = offsets
            .iter()
            .filter(|&&i| i > 0)
            .map(|&i| i as usize)
            .max()
            .unwrap_or(0);
        scioxide_core::backend::checked_add(
            scioxide_core::backend::checked_add(
                left,
                right,
                scioxide_core::quantity!("filter_workspace"),
            )?,
            1,
            scioxide_core::quantity!("filter_workspace"),
        )?;
        let run_count = 1 + offsets
            .windows(2)
            .filter(|pair| pair[0].checked_add(1) != Some(pair[1]))
            .count();
        let mut transitions =
            try_with_capacity(run_count, scioxide_core::quantity!("filter_transitions"))?;
        let mut start = offsets[0];
        for pair in offsets.windows(2) {
            if pair[0].checked_add(1) != Some(pair[1]) {
                transitions.push((start, pair[0]));
                start = pair[1];
            }
        }
        transitions.push((start, offsets[offsets.len() - 1]));
        let state_size = scioxide_core::backend::checked_add(
            left + right + 1,
            1,
            scioxide_core::quantity!("filter_workspace"),
        )?;
        scioxide_core::backend::checked_add(
            state_size,
            offsets.len(),
            scioxide_core::quantity!("filter_workspace"),
        )?;
        let resort = run_count > (usize::BITS - offsets.len().leading_zeros()) as usize;
        Ok(Self {
            transitions,
            resort,
            factor: S::try_from_f64_lossy(1.482602218505602)?,
            offsets,
            kind,
            left,
            right,
        })
    }
    /// Returns the delay of a stream: the largest positive offset, the number of
    /// later samples an output needs.
    pub fn delay(&self) -> usize {
        self.right
    }
    /// Stream history: the window span plus one outgoing sample during an advance.
    fn history_len(&self) -> usize {
        self.left + self.right + 2
    }
    /// Stream state: history and the sorted window.
    fn state_size(&self) -> usize {
        self.history_len() + self.offsets.len()
    }

    /// Returns the workspace size of [`LocalFilterPlan::filter_into`] for a record
    /// of `input_len` samples.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn workspace_requirements(
        &self,
        input_len: usize,
    ) -> Result<SignalRequirements, SciOxideError> {
        Ok(SignalRequirements::values(checked_add(
            checked_add(
                input_len,
                self.offsets.len(),
                scioxide_core::quantity!("filter_workspace"),
            )?,
            self.offsets.len(),
            scioxide_core::quantity!("filter_workspace"),
        )?))
    }
    /// Allocates a workspace for a record of `input_len` samples.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, input_len: usize) -> Result<SignalWorkspace<S>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(input_len)?)
    }
    // One driver serves batch and streams; `read` exposes the current window
    // and its outgoing observations without constructing a new sample buffer.
    fn evaluate(
        &self,
        index: usize,
        initialized: bool,
        sorted: &mut [S],
        work: &mut [S],
        moments: &mut moments::State<S>,
        read: impl Fn(usize, isize) -> S,
    ) -> Result<S, SciOxideError> {
        let center = read(index, 0);
        if let LocalFilter::Wiener { noise_variance } = self.kind {
            if noise_variance == S::ZERO {
                return Ok(center);
            }
            let change = initialized.then(|| {
                (
                    read(index - 1, self.offsets[0]),
                    read(index, self.offsets[self.offsets.len() - 1]),
                )
            });
            let value = moments.evaluate(
                self.offsets.iter().map(|&o| read(index, o)),
                self.offsets.len(),
                change,
            )?;
            return value.wiener(center, noise_variance);
        }
        if initialized && !self.resort {
            for &(start, end) in &self.transitions {
                order::replace(sorted, read(index - 1, start), read(index, end))?;
            }
        } else {
            for (slot, &offset) in sorted.iter_mut().zip(&self.offsets) {
                *slot = read(index, offset);
            }
            sorted.sort_unstable_by(order::compare);
        }
        let rank = if let LocalFilter::Rank { rank } = self.kind {
            rank
        } else {
            sorted.len() / 2
        };
        let mut median = sorted[rank];
        if sorted[0] == S::ZERO && sorted[sorted.len() - 1] == S::ZERO && center == S::ZERO {
            return Ok(center);
        }
        // The old unstable algorithms compared zero signs as equal. Reconstruct
        // offset order only when its selected representation can be observable.
        if median == S::ZERO
            && sorted.iter().any(|x| *x == S::ZERO && x.is_sign_negative())
            && sorted
                .iter()
                .any(|x| *x == S::ZERO && !x.is_sign_negative())
        {
            for (slot, &offset) in work.iter_mut().zip(&self.offsets) {
                *slot = read(index, offset);
            }
            if matches!(self.kind, LocalFilter::Hampel { .. }) {
                work.sort_unstable_by(order::compare);
                median = work[rank];
            } else {
                median = *work.select_nth_unstable_by(rank, order::compare).1;
            }
        }
        let LocalFilter::Hampel { threshold } = self.kind else {
            return Ok(median);
        };
        if let Some(value) = order::hampel(center, sorted, self.factor, threshold) {
            return Ok(if value != center { median } else { center });
        }
        let mad = median_deviation(sorted)?;
        let distance = Scaled::from_value(center)?
            .sub(Scaled::from_value(median)?)?
            .abs();
        let limit = mad
            .mul(Scaled::from_value(self.factor)?)?
            .mul(Scaled::from_value(threshold)?)?;
        Ok(if distance.abs_le(limit) {
            center
        } else {
            median
        })
    }
    /// Filters a record and returns the filtered samples.
    ///
    /// The result is the output of [`filter_into`](Self::filter_into), which
    /// also reports the samples that a Hampel filter replaces. This allocates
    /// the result, a workspace for `input.len()` samples and, for a Hampel
    /// filter, `input.len()` replacement indices, and runs with
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` if a Wiener output overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the workspace or the result
    ///   cannot be sized or allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{LocalFilter, LocalFilterPlan};
    /// let plan = LocalFilterPlan::new(1, LocalFilter::Hampel { threshold: 3.0_f64 })?;
    /// // Sample 2 lies six median deviations from its window median 3 and is replaced.
    /// let filtered = plan.filter(&[1.0, 2.0, 9.0, 3.0, 4.0])?;
    /// assert_eq!(filtered, [1.0, 2.0, 3.0, 3.0, 4.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn filter(&self, input: &[S]) -> Result<Vec<S>, SciOxideError> {
        let mut output = try_filled(
            input.len(),
            S::ZERO,
            scioxide_core::quantity!("filter_output"),
        )?;
        let mut workspace = self.workspace(input.len())?;
        let replaced = if matches!(self.kind, LocalFilter::Hampel { .. }) {
            input.len()
        } else {
            0
        };
        let mut replacements =
            try_filled(replaced, 0, scioxide_core::quantity!("filter_replacements"))?;
        self.filter_into(
            input,
            &mut output,
            &mut replacements,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Filters a record and returns the number of samples a Hampel filter replaced.
    ///
    /// Writes `input.len()` samples to the front of `output`. A Hampel filter also
    /// writes the indices of the replaced samples, in increasing order, to the
    /// front of `replacements`. Other filters write no indices and return zero.
    /// Entries past the written prefixes are not changed. The workspace must meet
    /// [`LocalFilterPlan::workspace_requirements`] for `input.len()`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is shorter than `input`, `replacements`
    ///   is shorter than the number of replaced samples, or the workspace is too
    ///   small.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` if a Wiener output overflows.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    pub fn filter_into(
        &self,
        input: &[S],
        output: &mut [S],
        replacements: &mut [usize],
        workspace: &mut SignalWorkspace<S>,
        context: ExecutionContext<'_>,
    ) -> Result<usize, SciOxideError> {
        check_real_slice_finite(input)?;
        capacity(input.len(), output.len())?;
        let scratch = workspace
            .buffers(&self.workspace_requirements(input.len())?)?
            .values;
        context.check()?;
        let (staged, work) = scratch.split_at_mut(input.len());
        let (sorted, work) = work.split_at_mut(self.offsets.len());
        let mut moments = moments::State::EMPTY;
        let mut replaced = 0;
        for (i, y) in staged.iter_mut().enumerate() {
            *y = self.evaluate(i, i > 0, sorted, work, &mut moments, |index, offset| {
                input[clamped_index(index, offset, input.len())]
            })?;
            if matches!(self.kind, LocalFilter::Hampel { .. }) && *y != input[i] {
                replaced += 1;
            }
        }
        capacity(replaced, replacements.len())?;
        context.check()?;
        let mut next = 0;
        if matches!(self.kind, LocalFilter::Hampel { .. }) {
            for (i, (&x, &y)) in input.iter().zip(staged.iter()).enumerate() {
                if x != y {
                    replacements[next] = i;
                    next += 1;
                }
            }
        }
        output[..input.len()].copy_from_slice(staged);
        Ok(replaced)
    }
    /// Starts a stream that filters a sequence delivered in chunks.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the stream state cannot be allocated.
    pub fn stream(&self) -> Result<LocalFilterStream<'_, S>, SciOxideError> {
        Ok(LocalFilterStream {
            plan: self,
            ring: zeros(self.history_len())?,
            sorted: zeros(self.offsets.len())?,
            moments: moments::State::EMPTY,
            seen: 0,
            emitted: 0,
            first: S::ZERO,
            last: S::ZERO,
            finished: false,
        })
    }
}
/// Stream that applies a [`LocalFilterPlan`] to a sequence delivered in chunks.
///
/// The stream produces the same samples as [`LocalFilterPlan::filter_into`] on
/// the whole sequence, independent of the chunk boundaries. Output `i` is
/// available once [`LocalFilterPlan::delay`] later samples have arrived, and the
/// final chunk flushes the rest. See [Streams](crate#streams).
///
/// ```
/// use scioxide_signal::{LocalFilter, LocalFilterPlan, StreamChunk};
/// let plan = LocalFilterPlan::new(1, LocalFilter::Median)?;
/// let mut stream = plan.stream()?;
/// let mut workspace = stream.workspace(3)?;
/// let mut output = [0.0; 4];
/// let progress = stream.process_into(
///     &[1.0, 9.0, 1.0],
///     StreamChunk::Final,
///     &mut output,
///     &mut [],
///     &mut workspace,
///     scioxide_core::ExecutionContext::sequential_native(),
/// )?;
/// // A single final chunk gives the batch result.
/// assert_eq!(&output[..progress.produced()], &[1.0; 3]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct LocalFilterStream<'a, S: RealScalar> {
    plan: &'a LocalFilterPlan<S>,
    ring: Vec<S>,
    sorted: Vec<S>,
    moments: moments::State<S>,
    seen: usize,
    emitted: usize,
    first: S,
    last: S,
    finished: bool,
}
impl<S: RealScalar> Clone for LocalFilterStream<'_, S> {
    fn clone(&self) -> Self {
        Self {
            plan: self.plan,
            ring: self.ring.clone(),
            sorted: self.sorted.clone(),
            moments: self.moments,
            seen: self.seen,
            emitted: self.emitted,
            first: self.first,
            last: self.last,
            finished: self.finished,
        }
    }
    fn clone_from(&mut self, source: &Self) {
        self.plan = source.plan;
        self.ring.clone_from(&source.ring);
        self.sorted.clone_from(&source.sorted);
        self.moments = source.moments;
        self.seen = source.seen;
        self.emitted = source.emitted;
        self.first = source.first;
        self.last = source.last;
        self.finished = source.finished;
    }
}
impl<S: RealScalar> LocalFilterStream<'_, S> {
    /// Resets the stream to the start of a new sequence.
    pub fn reset(&mut self) {
        self.seen = 0;
        self.emitted = 0;
        self.finished = false;
        self.first = S::ZERO;
        self.last = S::ZERO;
        self.ring.fill(S::ZERO);
        self.sorted.fill(S::ZERO);
        self.moments = moments::State::EMPTY;
    }
    /// Returns the number of samples a chunk of `input_len` samples produces.
    ///
    /// A [`StreamChunk::Partial`] chunk produces every output whose window has
    /// arrived. A [`StreamChunk::Final`] chunk produces every remaining output.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the stream is closed by a final chunk.
    /// - `SizeOverflow` if the total sample count overflows `usize`.
    pub fn output_len(&self, input_len: usize, chunk: StreamChunk) -> Result<usize, SciOxideError> {
        if self.finished {
            return Err(inconsistent(
                scioxide_core::argument!("self"),
                "stream is finished until reset",
            ));
        }
        let total = checked_add(
            self.seen,
            input_len,
            scioxide_core::quantity!("filter_workspace"),
        )?;
        Ok((match chunk {
            StreamChunk::Final => total,
            StreamChunk::Partial => total.saturating_sub(self.plan.right),
        }) - self.emitted)
    }
    /// Returns the workspace size for every later chunk of at most
    /// `max_input_len` samples, whatever the stream state.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn workspace_requirements(
        &self,
        max_input_len: usize,
    ) -> Result<SignalRequirements, SciOxideError> {
        self.requirements(checked_add(
            max_input_len,
            self.plan.right,
            scioxide_core::quantity!("filter_workspace"),
        )?)
    }
    /// Allocates a workspace for chunks of at most `max_input_len` samples.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, max_input_len: usize) -> Result<SignalWorkspace<S>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(max_input_len)?)
    }
    fn requirements(&self, produced: usize) -> Result<SignalRequirements, SciOxideError> {
        let quantity = scioxide_core::quantity!("filter_workspace");
        Ok(SignalRequirements::values(checked_add(
            self.plan.state_size(),
            checked_add(
                checked_add(produced, produced, quantity)?,
                self.plan.offsets.len(),
                quantity,
            )?,
            quantity,
        )?))
    }

    /// Filters one chunk and writes the outputs that become available.
    ///
    /// Writes [`LocalFilterStream::output_len`] samples to the front of `output`.
    /// A Hampel filter also writes the record indices of replaced samples to
    /// `replacements`. The returned [`StreamProgress`] reports the counts and
    /// the record index of the first output. Empty chunks are valid.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the stream is closed by a final chunk.
    /// - `DimensionMismatch` if `output` or `replacements` is too short, or the
    ///   workspace is too small.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` if a Wiener output overflows.
    /// - `SizeOverflow` if the total sample count overflows `usize`.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    pub fn process_into(
        &mut self,
        input: &[S],
        chunk: StreamChunk,
        output: &mut [S],
        replacements: &mut [usize],
        workspace: &mut SignalWorkspace<S>,
        context: ExecutionContext<'_>,
    ) -> Result<StreamProgress, SciOxideError> {
        let produced = self.output_len(input.len(), chunk)?;
        capacity(produced, output.len())?;
        let scratch = workspace.buffers(&self.requirements(produced)?)?.values;
        check_real_slice_finite(input)?;
        context.check()?;
        let finish = chunk == StreamChunk::Final;
        let (ring, rest) = scratch.split_at_mut(self.ring.len());
        ring.copy_from_slice(&self.ring);
        let (sorted, rest) = rest.split_at_mut(self.sorted.len());
        sorted.copy_from_slice(&self.sorted);
        let mut moments = self.moments;
        let (staged, rest) = rest.split_at_mut(produced);
        let (centers, window) = rest.split_at_mut(produced);
        let mut seen = self.seen;
        let mut emitted = self.emitted;
        let mut first = self.first;
        let mut last = self.last;
        let mut replaced = 0;
        for next in input
            .iter()
            .copied()
            .map(Some)
            .chain(core::iter::once(None))
        {
            if let Some(x) = next {
                if seen == 0 {
                    first = x;
                }
                last = x;
                ring[seen % ring.len()] = x;
                seen += 1;
            }
            let available = if next.is_none() && finish {
                seen
            } else {
                seen.saturating_sub(self.plan.right)
            };
            while emitted < available {
                let slot = emitted - self.emitted;
                let sample = |index: usize, offset: isize| {
                    if offset < 0 && index < offset.unsigned_abs() {
                        first
                    } else {
                        let j = if offset < 0 {
                            index - offset.unsigned_abs()
                        } else {
                            index.saturating_add(offset as usize)
                        };
                        if j >= seen {
                            last
                        } else {
                            ring[j % ring.len()]
                        }
                    }
                };
                let center = sample(emitted, 0);
                centers[slot] = center;
                staged[slot] = self.plan.evaluate(
                    emitted,
                    emitted > 0,
                    sorted,
                    window,
                    &mut moments,
                    sample,
                )?;
                if matches!(self.plan.kind, LocalFilter::Hampel { .. }) && staged[slot] != center {
                    replaced += 1;
                }
                emitted += 1;
            }
        }
        capacity(replaced, replacements.len())?;
        context.check()?;
        let mut next = 0;
        if matches!(self.plan.kind, LocalFilter::Hampel { .. }) {
            for (i, (&x, &y)) in centers.iter().zip(staged.iter()).enumerate() {
                if x != y {
                    replacements[next] = self.emitted + i;
                    next += 1;
                }
            }
        }
        output[..produced].copy_from_slice(staged);
        self.ring.copy_from_slice(ring);
        self.sorted.copy_from_slice(sorted);
        self.moments = moments;
        self.seen = seen;
        let first_output_index = self.emitted;
        self.emitted = emitted;
        self.first = first;
        self.last = last;
        self.finished = finish;
        Ok(StreamProgress {
            consumed: input.len(),
            produced,
            first_output_index,
            replacements: replaced,
        })
    }
}
/// Estimates the noise variance of a record for [`LocalFilter::Wiener`].
///
/// Returns the mean of the local population variances `σ²ᵢ` over all samples:
///
/// ```text
/// ν = (1/n) Σᵢ σ²ᵢ
/// ```
///
/// where `σ²ᵢ` is the variance of the centered window of `2·radius + 1` samples
/// around sample `i`, with the ends repeated as in [`LocalFilterPlan::new`].
/// The local variances use the running sums of the Wiener filter and cost
/// `O(n + radius)`.
///
/// ```
/// use scioxide_signal::estimate_wiener_noise;
/// // Every clamped radius-one window of an alternating sign holds two samples
/// // of one sign and one of the other, with population variance 8/9.
/// let noise = estimate_wiener_noise(&[1.0_f64, -1.0, 1.0, -1.0, 1.0, -1.0], 1)?;
/// assert!((noise - 8.0 / 9.0).abs() < 1e-14);
/// assert_eq!(estimate_wiener_noise(&[2.0_f64; 5], 2)?, 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `InvalidInput` if `input` is empty or `radius` exceeds `isize::MAX`.
/// - `NonFinite` if a sample is NaN or infinite.
/// - `NumericalInstability` if the estimate overflows.
/// - `SizeOverflow` if the window size overflows `usize`.
pub fn estimate_wiener_noise<S: RealScalar>(
    input: &[S],
    radius: usize,
) -> Result<S, SciOxideError> {
    if input.is_empty() {
        return Err(empty(
            scioxide_core::argument!("input"),
            "input must be nonempty",
        ));
    }
    check_real_slice_finite(input)?;
    let r = isize::try_from(radius).map_err(|_| {
        invalid(
            scioxide_core::argument!("radius"),
            "radius exceeds the signed offset range",
        )
    })?;
    let count = scioxide_core::backend::checked_add(
        scioxide_core::backend::checked_add(
            radius,
            radius,
            scioxide_core::quantity!("filter_workspace"),
        )?,
        1,
        scioxide_core::quantity!("filter_workspace"),
    )?;
    let mut sum = PromotingCompensatedSum::ZERO;
    let mut moments = moments::State::EMPTY;
    for i in 0..input.len() {
        let samples = (-r..=r).map(|offset| input[clamped_index(i, offset, input.len())]);
        let change = (i > 0).then(|| {
            (
                input[clamped_index(i - 1, -r, input.len())],
                input[clamped_index(i, r, input.len())],
            )
        });
        match moments.evaluate(samples, count, change)? {
            moments::Value::Native { variance, .. } => sum.add_value(variance)?,
            moments::Value::Extended { variance, .. } => sum.add_scaled(variance)?,
        }
    }
    sum.total()?
        .div(Scaled::from_value(scalar::<S>(input.len())?)?)?
        .round()
}
// A sorted window gives two sorted lists of distances from its median. Merge
// only to the median rank, keeping distances in extended range without buffers.
fn median_deviation<S: RealScalar>(window: &[S]) -> Result<Scaled<S>, SciOxideError> {
    let middle = window.len() / 2;
    let median = Scaled::from_value(window[middle])?;
    let mut left = middle;
    let mut right = middle + 1;
    let mut distance = Scaled::ZERO;
    for _ in 0..middle {
        let a = if left > 0 {
            Some(median.sub(Scaled::from_value(window[left - 1])?)?)
        } else {
            None
        };
        let b = if right < window.len() {
            Some(Scaled::from_value(window[right])?.sub(median)?)
        } else {
            None
        };
        match (a, b) {
            (Some(a), Some(b)) if a.abs_le(b) => {
                distance = a;
                left -= 1;
            }
            (_, Some(b)) => {
                distance = b;
                right += 1;
            }
            (Some(a), None) => {
                distance = a;
                left -= 1;
            }
            (None, None) => {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "window has no median-deviation rank",
                ));
            }
        }
    }
    Ok(distance)
}
fn local_moments<S: RealScalar>(
    samples: impl Iterator<Item = S> + Clone,
    count: usize,
) -> Result<(Scaled<S>, Scaled<S>), SciOxideError> {
    if let Some(first) = samples.clone().next()
        && samples.clone().all(|x| x == first)
    {
        return Ok((Scaled::from_value(first)?, Scaled::ZERO));
    }
    let count = Scaled::from_value(scalar::<S>(count)?)?;
    let mut sum = ScaledAccumulator::ZERO;
    for x in samples.clone() {
        sum.add_value(x)?;
    }
    let mean = sum.total()?.div(count)?;
    let mut squared = ScaledAccumulator::ZERO;
    for x in samples {
        let d = Scaled::from_value(x)?.sub(mean)?;
        squared.add_product(d, d)?;
    }
    Ok((mean, squared.total()?.div(count)?))
}

#[cfg(test)]
mod reference;

#[cfg(test)]
mod tests;
