//! Chunked STFT analysis and synthesis.
use super::*;
use crate::{StreamChunk, StreamProgress};
use scioxide_core::{AlgebraicScalar, Argument};

/// STFT analysis of a record that arrives in chunks.
///
/// The stream writes the coefficients of each frame as soon as the samples
/// it covers have arrived. The coefficients equal those of
/// [`StftPlan::analyze_into`] with zero extension, whatever the chunk sizes.
/// The stream keeps the last window length of samples. [Streams](crate#streams)
/// describes chunks, snapshots and workspaces. In the [`StreamProgress`] of a
/// call, `consumed` counts samples and `produced` counts coefficients.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext};
/// use scioxide_signal::{SignalWorkspace, StftOptions, StftPlan, StreamChunk};
///
/// let plan = StftPlan::new(&[1.0_f64; 4], 2, StftOptions::new())?;
/// let mut stream = plan.analysis_stream()?;
/// let mut ws = SignalWorkspace::new();
/// ws.try_reserve(&stream.workspace_requirements(8)?)?;
/// let input = [1.0; 8];
/// let mut output = vec![Complex::new(0.0, 0.0); stream.output_len(8, StreamChunk::Final)?];
/// let cx = ExecutionContext::sequential_native();
/// let progress = stream.process_into(&input, StreamChunk::Final, &mut output, &mut ws, cx)?;
/// // One final chunk gives the coefficients of offline analysis.
/// assert_eq!(output, plan.analyze(&input)?);
/// assert_eq!(progress.produced(), output.len());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct StftAnalysisStream<'p, T: NumericScalar> {
    plan: &'p StftPlan<T>,
    state: Vec<T>,
    seen: usize,
    emitted: usize,
    finished: bool,
}
/// Checked counters of one analysis call.
#[derive(Clone, Copy, Debug)]
struct AnalysisCounts {
    /// Frames emitted by the call.
    frames: usize,
    /// Absolute index of the call's first coefficient.
    first_output_index: usize,
    /// Coefficients emitted by the call.
    output_len: usize,
}
/// STFT synthesis of a record of known length from coefficients that arrive
/// in chunks of whole frames.
///
/// The stream writes each sample as soon as every frame that overlaps it has
/// arrived. The samples, and the real-reconstruction check of
/// [`StftPlan::synthesize_into`], do not depend on the chunk sizes. The stream
/// keeps one window length of partial sums. [Streams](crate#streams) describes
/// chunks, snapshots and workspaces. In the [`StreamProgress`] of a call,
/// `consumed` counts coefficients and `produced` counts samples.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{StftOptions, StftPlan, StreamChunk};
///
/// let plan = StftPlan::new(&[1.0_f64; 4], 2, StftOptions::new())?;
/// let signal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let coefficients = plan.analyze(&signal)?;
/// let mut stream = plan.synthesis_stream(signal.len())?;
/// let mut ws = stream.workspace(coefficients.len())?;
/// let cx = ExecutionContext::sequential_native();
/// // Chunks hold whole frames; the final chunk flushes the rest of the record.
/// let (head, tail) = coefficients.split_at(2 * plan.bins());
/// let mut restored = [0.0; 6];
/// let first = stream.process_into(head, StreamChunk::Partial, &mut restored, &mut ws, cx)?;
/// let rest = &mut restored[first.produced()..];
/// let last = stream.process_into(tail, StreamChunk::Final, rest, &mut ws, cx)?;
/// assert_eq!(last.first_output_index() + last.produced(), signal.len());
/// assert!(restored.iter().zip(signal).all(|(a, b)| (a - b).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct StftSynthesisStream<'p, T: NumericScalar> {
    plan: &'p StftPlan<T>,
    state: Vec<Complex<T::Real>>,
    scales: Vec<T::Real>,
    frames: StftFrames,
    received: usize,
    emitted: usize,
    finished: bool,
}
impl<T: NumericScalar> Clone for StftAnalysisStream<'_, T> {
    fn clone(&self) -> Self {
        Self {
            plan: self.plan,
            state: self.state.clone(),
            seen: self.seen,
            emitted: self.emitted,
            finished: self.finished,
        }
    }
    /// Copies the state of `source` into this stream, reusing its storage.
    fn clone_from(&mut self, source: &Self) {
        self.plan = source.plan;
        self.state.clone_from(&source.state);
        self.seen = source.seen;
        self.emitted = source.emitted;
        self.finished = source.finished;
    }
}
impl<T: NumericScalar> Clone for StftSynthesisStream<'_, T> {
    fn clone(&self) -> Self {
        Self {
            plan: self.plan,
            state: self.state.clone(),
            scales: self.scales.clone(),
            frames: self.frames,
            received: self.received,
            emitted: self.emitted,
            finished: self.finished,
        }
    }
    /// Copies the state of `source` into this stream, reusing its storage.
    fn clone_from(&mut self, source: &Self) {
        self.plan = source.plan;
        self.state.clone_from(&source.state);
        self.scales.clone_from(&source.scales);
        self.frames = source.frames;
        self.received = source.received;
        self.emitted = source.emitted;
        self.finished = source.finished;
    }
}
impl<T: NumericScalar> StftPlan<T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Starts a stream that analyzes a record in chunks.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the boundary option is
    ///   not [`SignalBoundary::Zero`].
    /// - `AllocationFailure` if the stream state cannot be allocated.
    pub fn analysis_stream(&self) -> Result<StftAnalysisStream<'_, T>, SciOxideError> {
        if self.frame.options.boundary != SignalBoundary::Zero {
            return Err(unsupported(
                scioxide_core::argument!("self"),
                "streaming analysis requires a zero boundary",
            ));
        }
        Ok(StftAnalysisStream {
            plan: self,
            state: try_filled(
                self.frame.window.len(),
                T::ZERO,
                scioxide_core::quantity!("stft_stream_state"),
            )?,
            seen: 0,
            emitted: 0,
            finished: false,
        })
    }
    /// Starts a stream that reconstructs a record of `signal_len` samples from
    /// coefficients that arrive in chunks.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the plan has no dual
    ///   window or the boundary option is not [`SignalBoundary::Zero`].
    /// - The errors of [`StftPlan::frames`] for `signal_len`.
    /// - `AllocationFailure` if the stream state cannot be allocated.
    pub fn synthesis_stream(
        &self,
        signal_len: usize,
    ) -> Result<StftSynthesisStream<'_, T>, SciOxideError> {
        if self.frame.options.boundary != SignalBoundary::Zero || self.dual.is_none() {
            return Err(unsupported(
                scioxide_core::argument!("self"),
                "streaming synthesis requires a zero boundary and a dual window",
            ));
        }
        let frames = self.frames(signal_len)?;
        Ok(StftSynthesisStream {
            plan: self,
            state: try_filled(
                self.frame.window.len(),
                zero(),
                scioxide_core::quantity!("istft_stream_state"),
            )?,
            scales: try_filled(
                self.frame.window.len(),
                T::Real::ZERO,
                scioxide_core::quantity!("istft_stream_state"),
            )?,
            frames,
            received: 0,
            emitted: 0,
            finished: false,
        })
    }
}
impl<T: NumericScalar> StftAnalysisStream<'_, T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Resets the stream to the start of a new record.
    pub fn reset(&mut self) {
        self.state.fill(T::ZERO);
        self.seen = 0;
        self.emitted = 0;
        self.finished = false;
    }
    fn frame_count(&self, input_len: usize, chunk: StreamChunk) -> Result<usize, SciOxideError> {
        if self.finished {
            return Err(inconsistent(
                scioxide_core::argument!("self"),
                "stream is finished until reset",
            ));
        }
        let seen = checked_add(
            self.seen,
            input_len,
            scioxide_core::quantity!("stft_stream_counter"),
        )?;
        let frames = self.plan.frames(seen)?;
        let ready = if chunk == StreamChunk::Final || seen == 0 {
            frames.count
        } else {
            let last = (seen as i128 + (self.state.len() / 2) as i128 - self.state.len() as i128)
                .div_euclid(self.plan.frame.hop as i128);
            (last + 1 - frames.first as i128).max(0) as usize
        };
        ready.checked_sub(self.emitted).ok_or_else(|| {
            inconsistent(
                scioxide_core::argument!("self"),
                "stream frame counter is inconsistent",
            )
        })
    }
    /// Checked counters of the next call. Its end coefficient index
    /// `(emitted + frames) * bins` must be representable, which also bounds
    /// its first coefficient index and its coefficient count.
    fn counts(
        &self,
        input_len: usize,
        chunk: StreamChunk,
    ) -> Result<AnalysisCounts, SciOxideError> {
        let frames = self.frame_count(input_len, chunk)?;
        let bins = self.plan.bins();
        let quantity = scioxide_core::quantity!("stft_coefficient_counter");
        let end = checked_mul(
            checked_add(
                self.emitted,
                frames,
                scioxide_core::quantity!("stft_frame_counter"),
            )?,
            bins,
            quantity,
        )?;
        let first_output_index = checked_mul(self.emitted, bins, quantity)?;
        Ok(AnalysisCounts {
            frames,
            first_output_index,
            output_len: end - first_output_index,
        })
    }
    /// Returns the number of coefficients that the next call writes for
    /// `input_len` samples and chunk kind `chunk`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if the stream is closed.
    /// - `SizeOverflow` if a sample or coefficient count does not fit in
    ///   `usize`.
    pub fn output_len(&self, input_len: usize, chunk: StreamChunk) -> Result<usize, SciOxideError> {
        Ok(self.counts(input_len, chunk)?.output_len)
    }

    /// Returns the workspace size for every later call with at most
    /// `max_input_len` samples.
    ///
    /// The size holds for either chunk kind, whatever the state of the stream.
    /// One call writes at most `⌊(max_input_len + m − 1)/h⌋ + 1` frames, for
    /// window length `m` and hop `h`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    pub fn workspace_requirements(
        &self,
        max_input_len: usize,
    ) -> Result<SignalRequirements, SciOxideError> {
        let quantity = scioxide_core::quantity!("stft_stream_workspace");
        let m = self.state.len();
        let frames = checked_add(
            checked_add(max_input_len, m - 1, quantity)? / self.plan.frame.hop,
            1,
            quantity,
        )?;
        Ok(SignalRequirements::complex(checked_add(
            checked_add(
                m,
                checked_mul(frames, self.plan.bins(), quantity)?,
                quantity,
            )?,
            self.plan.fft_work,
            quantity,
        )?))
    }
    /// Allocates a workspace of the size from
    /// [`Self::workspace_requirements`].
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, max_input_len: usize) -> Result<SignalWorkspace<T>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(max_input_len)?)
    }

    /// Consumes the samples `input` and writes the coefficients of the frames
    /// they complete to `output`, frame by frame.
    ///
    /// With [`StreamChunk::Final`], the call also writes the remaining frames,
    /// which extend past the end of the record, and closes the stream.
    /// `output` needs at least [`StftAnalysisStream::output_len`] elements,
    /// and `ws` the size of [`StftAnalysisStream::workspace_requirements`] for
    /// `input.len()`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if the stream is closed.
    /// - `DimensionMismatch` if `output` or `ws` is too short.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a coefficient overflows.
    /// - `ExecutionStopped` if `cx` cancels the call.
    pub fn process_into(
        &mut self,
        input: &[T],
        chunk: StreamChunk,
        output: &mut [Complex<T::Real>],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<StreamProgress, SciOxideError> {
        let counts = self.counts(input.len(), chunk)?;
        let n = counts.output_len;
        capacity(n, output.len())?;
        let buffers = ws.buffers(&self.workspace_requirements(input.len())?)?;
        finite(input)?;
        cx.check()?;
        let m = self.state.len();
        let (ring, rest) = buffers.complex.split_at_mut(m);
        for (z, &v) in ring.iter_mut().zip(self.state.iter()) {
            *z = complex(v);
        }
        let (staged, scratch) = rest.split_at_mut(n);
        let first = self.plan.frames(1)?.first;
        let mut seen = self.seen;
        let mut consumed = 0;
        for (i, out) in staged.chunks_exact_mut(self.plan.bins()).enumerate() {
            let start = self.plan.start(first + (self.emitted + i) as i64);
            let needed = (start + m as i128).max(0) as usize;
            while seen < needed && consumed < input.len() {
                ring[seen % m] = complex(input[consumed]);
                seen += 1;
                consumed += 1;
            }
            self.plan.analyze_frame(
                |j| {
                    let k = start + j as i128;
                    if k < 0 || k >= seen as i128 {
                        T::ZERO
                    } else {
                        let z = ring[k as usize % m];
                        T::from_components(z.re, z.im)
                    }
                },
                out,
                scratch,
                cx,
            )?;
        }
        for &v in &input[consumed..] {
            ring[seen % m] = complex(v);
            seen += 1;
        }
        cx.check()?;
        output[..n].copy_from_slice(staged);
        for (v, z) in self.state.iter_mut().zip(ring.iter()) {
            *v = T::from_components(z.re, z.im);
        }
        let progress = StreamProgress {
            replacements: 0,
            consumed: input.len(),
            produced: n,
            first_output_index: counts.first_output_index,
        };
        self.seen = seen;
        self.emitted += counts.frames;
        self.finished = chunk == StreamChunk::Final;
        Ok(progress)
    }
}
impl<T: NumericScalar> StftSynthesisStream<'_, T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Resets the stream to the first frame. The record length stays the same.
    pub fn reset(&mut self) {
        self.state.fill(zero());
        self.scales.fill(T::Real::ZERO);
        self.received = 0;
        self.emitted = 0;
        self.finished = false;
    }
    // `coefficients_argument` names the caller parameter carrying the chunk.
    fn end(
        &self,
        coefficients: usize,
        chunk: StreamChunk,
        coefficients_argument: Argument,
    ) -> Result<usize, SciOxideError> {
        if self.finished {
            return Err(inconsistent(
                scioxide_core::argument!("self"),
                "stream is finished until reset",
            ));
        }
        if !coefficients.is_multiple_of(self.plan.bins()) {
            return Err(inconsistent(
                coefficients_argument,
                "coefficients must hold whole frames",
            ));
        }
        let received = checked_add(
            self.received,
            coefficients / self.plan.bins(),
            scioxide_core::quantity!("istft_frame_counter"),
        )?;
        let finish = chunk == StreamChunk::Final;
        if received > self.frames.count || (finish && received != self.frames.count) {
            return Err(inconsistent(
                coefficients_argument,
                "coefficient frames exceed or fall short of the planned frames",
            ));
        }
        if finish {
            Ok(self.frames.signal_len)
        } else {
            Ok(
                (self.plan.start(self.frames.first + received as i64).max(0) as usize)
                    .min(self.frames.signal_len),
            )
        }
    }
    /// Returns the number of samples that the next call writes for
    /// `coefficients` coefficients and chunk kind `chunk`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if the stream is closed, `coefficients`
    ///   is not a whole number of frames, the frames exceed the frames of the
    ///   record, or a final chunk does not complete them.
    /// - `SizeOverflow` if the frame count does not fit in `usize`.
    pub fn output_len(
        &self,
        coefficients: usize,
        chunk: StreamChunk,
    ) -> Result<usize, SciOxideError> {
        self.output_count(
            coefficients,
            chunk,
            scioxide_core::argument!("coefficients"),
        )
    }
    fn output_count(
        &self,
        coefficients: usize,
        chunk: StreamChunk,
        coefficients_argument: Argument,
    ) -> Result<usize, SciOxideError> {
        self.end(coefficients, chunk, coefficients_argument)?
            .checked_sub(self.emitted)
            .ok_or_else(|| {
                inconsistent(
                    scioxide_core::argument!("self"),
                    "stream output counter is inconsistent",
                )
            })
    }
    /// Returns the workspace size for every later call with at most
    /// `max_input_len` coefficients.
    ///
    /// The size holds for either chunk kind, whatever the state of the stream.
    /// A call with `f` frames writes at most `min(f·h, signal_len)` samples,
    /// for hop `h`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    pub fn workspace_requirements(
        &self,
        max_input_len: usize,
    ) -> Result<SignalRequirements, SciOxideError> {
        let quantity = scioxide_core::quantity!("istft_stream_work");
        let samples = (max_input_len / self.plan.bins())
            .saturating_mul(self.plan.frame.hop)
            .min(self.frames.signal_len);
        let staged = checked_add(self.state.len(), samples, quantity)?;
        Ok(SignalRequirements::new(
            0,
            staged,
            checked_add(staged, self.plan.fft_work, quantity)?,
        ))
    }
    /// Allocates a workspace of the size from
    /// [`Self::workspace_requirements`].
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `AllocationFailure` if the memory cannot be reserved.
    pub fn workspace(&self, max_input_len: usize) -> Result<SignalWorkspace<T>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(max_input_len)?)
    }
    /// Consumes the coefficients of the next whole frames in `input` and
    /// writes the samples they complete to `output`.
    ///
    /// With [`StreamChunk::Final`], `input` must hold every remaining frame of
    /// the record. The call then writes the remaining samples and closes the
    /// stream. `output` needs at least [`StftSynthesisStream::output_len`]
    /// elements, and `ws` the size of
    /// [`StftSynthesisStream::workspace_requirements`] for `input.len()`.
    ///
    /// # Errors
    ///
    /// - The errors of [`StftSynthesisStream::output_len`].
    /// - `DimensionMismatch` if `output` or `ws` is too short.
    /// - `DomainError` (`OutsideDomain`) if, for a real sample type, a one-sided
    ///   DC or Nyquist coefficient or a written sample is not real within the
    ///   tolerance of [`StftOptions::with_tolerance`].
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a sample overflows.
    /// - `ExecutionStopped` if `cx` cancels the call.
    pub fn process_into(
        &mut self,
        input: &[Complex<T::Real>],
        chunk: StreamChunk,
        output: &mut [T],
        ws: &mut SignalWorkspace<T>,
        cx: ExecutionContext<'_>,
    ) -> Result<StreamProgress, SciOxideError> {
        let n = self.output_count(input.len(), chunk, scioxide_core::argument!("input"))?;
        capacity(n, output.len())?;
        let buffers = ws.buffers(&self.workspace_requirements(input.len())?)?;
        finite(input)?;
        cx.check()?;
        let m = self.state.len();
        let (ring, rest) = buffers.complex.split_at_mut(m);
        ring.copy_from_slice(&self.state);
        let (staged, scratch) = rest.split_at_mut(n);
        let (ring_scales, rest) = buffers.real.split_at_mut(m);
        ring_scales.copy_from_slice(&self.scales);
        let staged_scales = &mut rest[..n];
        let mut emitted = self.emitted;
        let mut received = self.received;
        // Move finalized ring samples and their scales into staging.
        let emit = |ring: &mut [Complex<T::Real>],
                    ring_scales: &mut [T::Real],
                    staged: &mut [Complex<T::Real>],
                    staged_scales: &mut [T::Real],
                    emitted: &mut usize,
                    end: usize| {
            while *emitted < end {
                let (i, j) = (*emitted % m, *emitted - self.emitted);
                staged[j] = core::mem::replace(&mut ring[i], zero());
                staged_scales[j] = core::mem::replace(&mut ring_scales[i], T::Real::ZERO);
                *emitted += 1;
            }
        };
        for frame in input.chunks_exact(self.plan.bins()) {
            let start = self.plan.start(self.frames.first + received as i64);
            let (values, frame_scale) = self.plan.inverse_frame(frame, scratch, cx)?;
            for (j, &v) in values.iter().enumerate() {
                let k = start + j as i128;
                if k >= 0 && k < self.frames.signal_len as i128 {
                    let i = k as usize % m;
                    super::overlap_add(&mut ring[i], &mut ring_scales[i], v, frame_scale)?;
                }
            }
            received += 1;
            let end = (self.plan.start(self.frames.first + received as i64).max(0) as usize)
                .min(self.frames.signal_len);
            emit(ring, ring_scales, staged, staged_scales, &mut emitted, end);
        }
        emit(
            ring,
            ring_scales,
            staged,
            staged_scales,
            &mut emitted,
            self.emitted + n,
        );
        self.plan.validate_real(staged, staged_scales)?;
        cx.check()?;
        for (slot, z) in output.iter_mut().zip(staged.iter()) {
            *slot = T::from_components(z.re, z.im);
        }
        self.state.copy_from_slice(ring);
        self.scales.copy_from_slice(ring_scales);
        let progress = StreamProgress {
            replacements: 0,
            consumed: input.len(),
            produced: n,
            first_output_index: self.emitted,
        };
        self.received = received;
        self.emitted = emitted;
        self.finished = chunk == StreamChunk::Final;
        Ok(progress)
    }
}
