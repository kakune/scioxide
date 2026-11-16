//! Overlap-add FFT convolution of streamed records.
use super::nd::{ConvolutionMethod, ConvolutionNdPlan, ConvolutionOptions};
use crate::util::inputs as finite;
use crate::util::{empty, inconsistent, invalid, output_overflow, zeros};
use crate::{
    SignalExecutionOptions, SignalRequirements, SignalWorkspace, StreamChunk, StreamProgress,
};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{ExecutionContext, NdView, NumericScalar, SciOxideError, StridedLayout};
/// Plan for the FIR filtering of a streamed record by overlap-add FFT
/// convolution.
///
/// The plan splits the input into blocks of `block_size` samples, convolves
/// each block with the `taps` by FFT, and adds the overlapping `taps.len() − 1`
/// samples of consecutive block results. The outputs of a whole stream are the
/// full linear convolution of the record with the taps, as computed by
/// [`convolve`](crate::convolve). Each block costs `O(L log L)`, where `L` is
/// `block_size + taps.len() − 1` rounded up to a power of two. A larger block
/// lowers the cost per sample, and a smaller one lowers the delay and the
/// workspace. See [Streams](crate#streams).
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{OverlapPlan, SignalExecutionOptions, StreamChunk};
/// let plan = OverlapPlan::new(&[1.0_f64, 1.0], 4, SignalExecutionOptions::new())?;
/// let mut stream = plan.stream()?;
/// let mut workspace = stream.workspace(3)?;
/// let mut output = [0.0; 4];
/// let progress = stream.process_into(
///     &[1.0, 2.0, 3.0],
///     StreamChunk::Final,
///     &mut output,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// // [1, 2, 3] ∗ [1, 1] = [1, 3, 5, 3]
/// assert_eq!(progress.produced(), 4);
/// assert!((output[3] - 3.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
/// Pearson, 2010, §8.7.
#[derive(Clone, Debug)]
pub struct OverlapPlan<T: NumericScalar> {
    convolution: ConvolutionNdPlan<T>,
    block: usize,
    tail: usize,
    output_len: usize,
}
impl<T: NumericScalar> OverlapPlan<T> {
    /// Prepares the FFTs and the spectrum of `taps` for blocks of `block_size`
    /// samples, using [`ExecutionContext::sequential_native`].
    ///
    /// The execution policy of `options` applies to the FFTs.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `block_size` is zero or `taps` is empty.
    /// - `NonFinite` if a tap is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the plan storage cannot be sized
    ///   or allocated.
    pub fn new(
        taps: &[T],
        block_size: usize,
        options: SignalExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            taps,
            block_size,
            options,
            ExecutionContext::sequential_native(),
        )
    }
    /// Prepares a plan like [`new`](Self::new), planning the FFTs with `context`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`new`](Self::new), and `ExecutionStopped` if
    /// `context` is cancelled or past its deadline.
    pub fn build(
        taps: &[T],
        block_size: usize,
        options: SignalExecutionOptions,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        if block_size == 0 {
            return Err(invalid(
                scioxide_core::argument!("block_size"),
                "block size must be positive",
            ));
        }
        if taps.is_empty() {
            return Err(empty(
                scioxide_core::argument!("taps"),
                "at least one tap is required",
            ));
        }
        let tail = taps.len() - 1;
        let output_len = checked_add(block_size, tail, scioxide_core::quantity!("overlap_output"))?;
        let convolution = ConvolutionNdPlan::build(
            &[block_size],
            NdView::new(taps, StridedLayout::new(&[taps.len()], &[1])?)?,
            &[0],
            ConvolutionOptions::new()
                .with_method(ConvolutionMethod::Fft)
                .with_execution_policy(options.execution_policy()),
            context,
        )?;
        Ok(Self {
            convolution,
            block: block_size,
            tail,
            output_len,
        })
    }
    /// Returns the number of input samples per block.
    ///
    /// A partial chunk writes only complete blocks, so up to `block_size − 1`
    /// input samples wait for the next call.
    pub fn block_size(&self) -> usize {
        self.block
    }
    /// Starts a stream at the beginning of a record, with zero history.
    ///
    /// The stream allocates `block_size + taps.len() − 1` samples of history.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the history length does not fit in `usize`.
    /// - `AllocationFailure` if the allocator fails.
    pub fn stream(&self) -> Result<OverlapStream<'_, T>, SciOxideError> {
        Ok(OverlapStream {
            plan: self,
            state: zeros(self.output_len)?,
            pending: 0,
            seen: 0,
            emitted: 0,
            finished: false,
        })
    }
}
/// Overlap-add stream of an [`OverlapPlan`], holding the samples of the
/// current block and the overlap of the previous one.
///
/// A [`StreamChunk::Partial`] call writes only complete blocks. The
/// [`StreamChunk::Final`] call zero-pads the last block and writes it with the
/// `taps.len() − 1` tail samples, so the outputs of all calls together are the
/// full linear convolution of the record. Cloning a stream takes a snapshot,
/// and `clone_from` restores one. See [Streams](crate#streams).
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideErrorKind};
/// use scioxide_signal::{OverlapPlan, SignalExecutionOptions, StreamChunk, convolve};
/// let plan = OverlapPlan::new(&[1.0_f64, -1.0], 2, SignalExecutionOptions::new())?;
/// let mut stream = plan.stream()?;
/// let mut workspace = stream.workspace(3)?;
/// let mut output = [0.0; 5];
/// let first = stream.process_into(
///     &[1.0, 2.0, 3.0],
///     StreamChunk::Partial,
///     &mut output,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// // One complete block of two samples; the third sample waits.
/// assert_eq!(first.produced(), 2);
/// let snapshot = stream.clone();
/// let last = stream.process_into(
///     &[4.0],
///     StreamChunk::Final,
///     &mut output[2..],
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// assert_eq!((last.first_output_index(), last.produced()), (2, 3));
/// let full = convolve(&[1.0, 2.0, 3.0, 4.0], &[1.0, -1.0])?;
/// assert!(output.iter().zip(&full).all(|(x, e)| (x - e).abs() < 1e-12));
/// // A final chunk closes the stream; restoring the snapshot reopens it.
/// let closed = stream.output_len(1, StreamChunk::Partial);
/// assert_eq!(closed.unwrap_err().kind(), SciOxideErrorKind::InvalidInput);
/// stream.clone_from(&snapshot);
/// assert_eq!(stream.output_len(1, StreamChunk::Final)?, 3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct OverlapStream<'p, T: NumericScalar> {
    plan: &'p OverlapPlan<T>,
    state: Vec<T>,
    pending: usize,
    seen: usize,
    emitted: usize,
    finished: bool,
}
impl<T: NumericScalar> Clone for OverlapStream<'_, T> {
    fn clone(&self) -> Self {
        Self {
            plan: self.plan,
            state: self.state.clone(),
            pending: self.pending,
            seen: self.seen,
            emitted: self.emitted,
            finished: self.finished,
        }
    }
    fn clone_from(&mut self, source: &Self) {
        self.plan = source.plan;
        self.state.clone_from(&source.state);
        self.pending = source.pending;
        self.seen = source.seen;
        self.emitted = source.emitted;
        self.finished = source.finished;
    }
}
impl<T: NumericScalar> OverlapStream<'_, T> {
    /// Returns the stream to the beginning of a new record, with zero history,
    /// and reopens it.
    pub fn reset(&mut self) {
        self.state.fill(T::ZERO);
        self.pending = 0;
        self.seen = 0;
        self.emitted = 0;
        self.finished = false;
    }
    /// Returns the number of samples that a call with `input_len` input samples
    /// writes.
    ///
    /// For [`StreamChunk::Partial`] this is the number of samples in the
    /// complete blocks. For [`StreamChunk::Final`] it is every remaining output
    /// sample, including the tail, or zero if the record is empty.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the stream is closed.
    /// - `SizeOverflow` if the sample count does not fit in `usize`.
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
            scioxide_core::quantity!("overlap_input_clock"),
        )?;
        match chunk {
            StreamChunk::Final if seen == 0 => Ok(0),
            StreamChunk::Final => checked_add(
                seen,
                self.plan.tail,
                scioxide_core::quantity!("overlap_tail"),
            )?
            .checked_sub(self.emitted)
            .ok_or_else(|| {
                inconsistent(
                    scioxide_core::argument!("self"),
                    "stream output clock is inconsistent",
                )
            }),
            StreamChunk::Partial => checked_mul(
                checked_add(
                    self.pending,
                    input_len,
                    scioxide_core::quantity!("pending_overlap_block"),
                )? / self.plan.block,
                self.plan.block,
                scioxide_core::quantity!("overlap_emitted_blocks"),
            ),
        }
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
        self.requirements(checked_add(
            max_input_len,
            self.plan.output_len,
            scioxide_core::quantity!("overlap_output_bound"),
        )?)
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
        let values = [
            self.state.len(),
            output,
            self.plan.output_len,
            self.plan.convolution.real_time_len(),
        ]
        .into_iter()
        .try_fold(0, |n, v| {
            checked_add(n, v, scioxide_core::quantity!("overlap_workspace"))
        })?;
        Ok(SignalRequirements::new(
            values,
            0,
            self.plan.convolution.complex_len()?,
        ))
    }

    /// Filters the next chunk of the record and writes the completed output
    /// samples to the start of `output`.
    ///
    /// The call writes [`output_len(input.len(), chunk)`](Self::output_len)
    /// samples. [`StreamChunk::Final`] closes the stream until
    /// [`reset`](Self::reset).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the stream is closed.
    /// - `DimensionMismatch` if `output` is shorter than the output count, or
    ///   `workspace` is too small.
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` if an output sample overflows.
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
        crate::util::capacity(count, output.len())?;
        let buffers = workspace.buffers(&self.requirements(count)?)?;
        finite(input)?;
        context.check()?;
        let finish = chunk == StreamChunk::Final;
        let (state, rest) = buffers.values.split_at_mut(self.state.len());
        state.copy_from_slice(&self.state);
        let (staged, rest) = rest.split_at_mut(count);
        let (block_output, real_time) = rest.split_at_mut(self.plan.output_len);
        let mut pending = self.pending;
        let mut consumed = 0;
        let mut produced = 0;
        while consumed < input.len() || (finish && produced < count) {
            let take = (self.plan.block - pending).min(input.len() - consumed);
            state[pending..pending + take].copy_from_slice(&input[consumed..consumed + take]);
            pending += take;
            consumed += take;
            if pending < self.plan.block && !finish {
                break;
            }
            if pending < self.plan.block {
                state[pending..self.plan.block].fill(T::ZERO);
            }
            let (block, history) = state.split_at_mut(self.plan.block);
            self.plan.convolution.convolve_staged(
                block,
                block_output,
                real_time,
                buffers.complex,
                context,
            )?;
            for (v, &h) in block_output.iter_mut().zip(history.iter()) {
                *v = *v + h;
                if !v.has_finite_components() {
                    return Err(output_overflow());
                }
            }
            history
                .copy_from_slice(&block_output[self.plan.block..self.plan.block + self.plan.tail]);
            let take = self.plan.block.min(count - produced);
            staged[produced..produced + take].copy_from_slice(&block_output[..take]);
            produced += take;
            pending = 0;
        }
        context.check()?;
        output[..count].copy_from_slice(staged);
        self.state.copy_from_slice(state);
        let progress = StreamProgress {
            replacements: 0,
            consumed: input.len(),
            produced: count,
            first_output_index: self.emitted,
        };
        self.pending = pending;
        self.seen += input.len();
        self.emitted += count;
        self.finished = finish;
        Ok(progress)
    }
}
