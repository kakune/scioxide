use super::*;
use crate::util::{
    capacity, exact_len, execute_blocks, inconsistent, layout_lanes, size, unsupported,
};
use crate::{SignalExecutionOptions, SignalLayout, SignalRequirements, SignalWorkspace};
use scioxide_core::AlgebraicScalar;
use scioxide_core::{ExecutionContext, NdView, NdViewMut};

/// Delay state of an [`IirFilter`] for one or more channels.
///
/// The state holds [`IirFilter::delays`] values per channel, stored
/// channel after channel. They are the delays of the transposed direct form II
/// of each section, in section order. Passing the same state to successive
/// [`filter_into`](IirFilter::filter_into) calls filters a record in
/// chunks, with the same result as one call on the whole record. A call that
/// fails leaves the state unchanged.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{
///     IirDomain, IirFilter, IirRequest, IirState, SignalExecutionOptions, SignalLayout,
///     TransferFunction,
/// };
/// // y[n] = 0.5 x[n] + 0.5 y[n − 1], filtered in two chunks of two samples.
/// let transfer = TransferFunction::new(IirDomain::Digital, &[0.5_f64], &[1.0, -0.5])?;
/// let filter = IirFilter::from_transfer(&transfer, SignalExecutionOptions::new())?;
/// let mut state = IirState::new(&filter, 1)?;
/// let mut workspace = filter.workspace(IirRequest::Filter(SignalLayout::Contiguous(2)))?;
/// let (mut first, mut second) = ([0.0; 2], [0.0; 2]);
/// let cx = ExecutionContext::sequential_native();
/// filter.filter_into(&[1.0, 0.0], &mut first, &mut state, &mut workspace, cx)?;
/// assert_eq!(state.values(), [0.125]);
/// filter.filter_into(&[0.0, 0.0], &mut second, &mut state, &mut workspace, cx)?;
/// let whole = filter.filter(&[1.0, 0.0, 0.0, 0.0])?;
/// assert_eq!([first, second].concat(), whole);
/// state.reset();
/// assert_eq!((state.channels(), state.values()), (1, &[0.0][..]));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct IirState<T: NumericScalar> {
    channels: usize,
    per_channel: usize,
    data: Vec<T>,
}
impl<T: NumericScalar> IirState<T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Creates a zero state for `channels` channels of `filter`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `channels` times [`IirFilter::delays`]
    ///   overflows `usize`.
    /// - `AllocationFailure` if the state cannot be allocated.
    pub fn new<C: NumericScalar<Real = T::Real>>(
        filter: &IirFilter<C>,
        channels: usize,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            channels,
            per_channel: filter.delays,
            data: zeros(size(channels, filter.delays)?)?,
        })
    }
    /// Returns the delay values, stored channel after channel.
    pub fn values(&self) -> &[T] {
        &self.data
    }
    /// Returns the delay values for modification.
    ///
    /// The next filtering call checks that every value is finite.
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
    /// Returns the number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }
    /// Sets every delay value to zero.
    pub fn reset(&mut self) {
        self.data.fill(T::ZERO);
    }
}
/// Extension of a record at both ends before forward-backward filtering.
///
/// The value is the number `L` of samples added at each end. A positive `L`
/// must be smaller than the record length, and `L = 0` is the same as
/// [`IirPadding::None`]. For a record `x₀, …, x_{n−1}`, the samples added
/// before `x₀` are, for `k = 1, …, L`:
///
/// ```text
/// Odd:       x₋ₖ = 2x₀ − xₖ
/// Even:      x₋ₖ = xₖ
/// Constant:  x₋ₖ = x₀
/// ```
///
/// The end of the record is extended in the same way about `x_{n−1}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IirPadding {
    /// No extension.
    None,
    /// Odd reflection about the end sample.
    Odd(usize),
    /// Even reflection, without repeating the end sample.
    Even(usize),
    /// Repetition of the end sample.
    Constant(usize),
}
impl IirPadding {
    /// Returns the number of samples added at each end.
    pub const fn length(self) -> usize {
        match self {
            Self::None => 0,
            Self::Odd(length) | Self::Even(length) | Self::Constant(length) => length,
        }
    }
}
/// Initial delay state of one pass of forward-backward filtering.
#[derive(Clone, Copy, Debug)]
pub enum IirInitialization<'a, T: NumericScalar> {
    /// Zero delays.
    Zero,
    /// Steady state of a constant input equal to the first sample of the pass.
    ///
    /// The output then equals `H(1)·x₀` for as long as the input stays at its
    /// first sample `x₀`, where `H(1)` is the DC gain. A filter with a section
    /// whose denominator vanishes at `z = 1` has no steady state, and
    /// filtering returns `DomainError`.
    SteadyState,
    /// Explicit transposed direct form II delays, [`IirFilter::delays`]
    /// values in the layout of [`IirState`].
    State(&'a [T]),
}
/// End extension and initial states of forward-backward filtering.
///
/// [`ForwardBackwardOptions::new`] uses no extension and zero initial delays
/// in both passes, which accepts a record of any length. To suppress the
/// transients at the record ends, extend the record by odd reflection over
/// `3·max(len(a), len(b))` samples, where `a` and `b` are the denominator and
/// numerator coefficients, and start both passes in
/// [`IirInitialization::SteadyState`].
/// [`TransferFunction::forward_backward`] and
/// [`SecondOrderSections::forward_backward`] apply this end treatment.
///
/// # Examples
///
/// ```
/// use scioxide_signal::{ForwardBackwardOptions, IirInitialization, IirPadding};
/// // 3·max(len(a), len(b)) = 6 samples for a first-order filter.
/// let options = ForwardBackwardOptions::<f64>::new()
///     .with_padding(IirPadding::Odd(6))
///     .with_forward(IirInitialization::SteadyState)
///     .with_backward(IirInitialization::SteadyState);
/// assert_eq!(options.padding().length(), 6);
/// ```
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct ForwardBackwardOptions<'a, T: NumericScalar> {
    padding: IirPadding,
    forward: IirInitialization<'a, T>,
    backward: IirInitialization<'a, T>,
}
impl<T: NumericScalar> Default for ForwardBackwardOptions<'_, T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<'a, T: NumericScalar> ForwardBackwardOptions<'a, T> {
    /// Returns options with no extension and zero initial delays in both passes.
    pub const fn new() -> Self {
        Self {
            padding: IirPadding::None,
            forward: IirInitialization::Zero,
            backward: IirInitialization::Zero,
        }
    }
    /// Sets the end extension.
    pub const fn with_padding(mut self, padding: IirPadding) -> Self {
        self.padding = padding;
        self
    }
    /// Sets the initial delays of the forward pass.
    pub const fn with_forward(mut self, forward: IirInitialization<'a, T>) -> Self {
        self.forward = forward;
        self
    }
    /// Sets the initial delays of the backward pass, which filters the
    /// time-reversed output of the forward pass.
    pub const fn with_backward(mut self, backward: IirInitialization<'a, T>) -> Self {
        self.backward = backward;
        self
    }
    /// Returns the end extension.
    pub const fn padding(self) -> IirPadding {
        self.padding
    }
    /// Returns the initial delays of the forward pass.
    pub const fn forward(self) -> IirInitialization<'a, T> {
        self.forward
    }
    /// Returns the initial delays of the backward pass.
    pub const fn backward(self) -> IirInitialization<'a, T> {
        self.backward
    }
}
/// Operation for which [`IirFilter::workspace_requirements`] sizes a workspace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IirRequest<'a> {
    /// Causal filtering of a record or of every lane along an axis.
    Filter(SignalLayout<'a>),
    /// Forward-backward filtering with the given endpoint extension.
    ForwardBackward(SignalLayout<'a>, IirPadding),
}
/// Digital IIR filter, from a transfer function or a cascade of
/// second-order sections.
///
/// Each section runs the transposed direct form II recurrence
///
/// ```text
/// y[n]  = b₀ x[n] + s₁[n−1]
/// sⱼ[n] = sⱼ₊₁[n−1] + bⱼ x[n] − aⱼ y[n],   j = 1, …, N,   s_{N+1} = 0
/// ```
///
/// where `x` is the section input, `y` its output, `b` and `a` its numerator
/// and denominator coefficients with `a₀ = 1`, `N` its order and `sⱼ` its
/// delays. Filtering costs `O(d)` operations per sample and channel, where
/// `d` is [`delays`](IirFilter::delays). The filter holds the normalized
/// coefficients and computes nothing at construction; it is reused as described
/// in [reusing plans and workspaces](crate#reusing-plans-and-workspaces), and
/// [`filter`](IirFilter::filter) is its owned form.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_signal::{
///     IirDomain, IirFilter, IirRequest, IirState, SignalExecutionOptions, SignalLayout,
///     TransferFunction,
/// };
/// // Two-point moving average y[n] = (x[n] + x[n − 1]) / 2.
/// let transfer = TransferFunction::new(IirDomain::Digital, &[0.5_f64, 0.5], &[1.0])?;
/// let filter = IirFilter::from_transfer(&transfer, SignalExecutionOptions::new())?;
/// let mut state = IirState::new(&filter, 1)?;
/// let mut workspace = filter.workspace(IirRequest::Filter(SignalLayout::Contiguous(3)))?;
/// let (input, mut output) = ([2.0, 4.0, 6.0], [0.0; 3]);
/// let cx = ExecutionContext::sequential_native();
/// filter.filter_into(&input, &mut output, &mut state, &mut workspace, cx)?;
/// assert_eq!(output, [1.0, 3.0, 5.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct IirFilter<T: NumericScalar> {
    sections: Vec<TransferFunction<T>>,
    delays: usize,
    options: SignalExecutionOptions,
}
impl<T: NumericScalar> IirFilter<T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Prepares filtering with a digital transfer function.
    ///
    /// The filter runs as one section of order `max(len(a), len(b)) − 1`. The
    /// filter accepts an unstable recursion: check `transfer.to_zpk()?.is_stable()`
    /// when bounded-input bounded-output stability is required. A high-order
    /// direct form is sensitive to coefficient rounding, so prefer
    /// [`from_sos`](Self::from_sos) for such filters. The execution policy of
    /// `options` bounds the worker threads of the axis forms.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `transfer` is analog.
    pub fn from_transfer(
        transfer: &TransferFunction<T>,
        options: SignalExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        if transfer.domain != IirDomain::Digital {
            return Err(invalid(
                scioxide_core::argument!("transfer"),
                "digital filtering requires digital coefficients",
            ));
        }
        let mut sections = Vec::new();
        sections.try_reserve_exact(1).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("iir_plan"), 1)
        })?;
        let delays = transfer.a.len().max(transfer.b.len()) - 1;
        sections.push(TransferFunction::new(
            transfer.domain,
            &transfer.b,
            &transfer.a,
        )?);
        Ok(Self {
            sections,
            delays,
            options,
        })
    }
    /// Prepares filtering with a cascade of digital second-order sections.
    ///
    /// The sections run in the given order, each with two delays per channel,
    /// also when it is of first order. The execution policy of `options`
    /// bounds the worker threads of the axis forms.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `sos` is analog.
    pub fn from_sos(
        sos: &SecondOrderSections<T>,
        options: SignalExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        if sos.domain != IirDomain::Digital {
            return Err(invalid(
                scioxide_core::argument!("sos"),
                "digital filtering requires digital sections",
            ));
        }
        let mut sections = Vec::new();
        sections
            .try_reserve_exact(sos.sections.len())
            .map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("iir_plan"),
                    sos.sections.len(),
                )
            })?;
        let mut delays = 0;
        for row in &sos.sections {
            // Keep two delays per SOS, including first-order padded sections.
            sections.push(TransferFunction {
                domain: IirDomain::Digital,
                b: copy(&row[..3])?,
                a: copy(&row[3..])?,
            });
            delays = add(delays, 2)?;
        }
        Ok(Self {
            sections,
            delays,
            options,
        })
    }
    /// Returns the number of delay values per channel.
    pub fn delays(&self) -> usize {
        self.delays
    }
    /// Returns the execution options of the filter.
    pub fn options(&self) -> SignalExecutionOptions {
        self.options
    }
    /// Returns the workspace size, in sample values, of one call with `request`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the axis of an axis layout is not below the rank of
    ///   its shape.
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn workspace_requirements(
        &self,
        request: IirRequest<'_>,
    ) -> Result<SignalRequirements, SciOxideError> {
        Ok(SignalRequirements::values(match request {
            IirRequest::Filter(SignalLayout::Contiguous(n)) => add(n, self.delays)?,
            IirRequest::Filter(layout) => {
                let (n, lanes) = layout_lanes(layout)?;
                size(add(size(n, 2)?, self.delays)?, lanes)?
            }
            IirRequest::ForwardBackward(SignalLayout::Contiguous(n), padding) => {
                self.record_len(n, padding.length())?
            }
            IirRequest::ForwardBackward(layout, padding) => {
                let (n, lanes) = layout_lanes(layout)?;
                size(self.lane_len(n, padding.length())?, lanes)?
            }
        }))
    }
    /// Allocates a workspace of the size that
    /// [`workspace_requirements`](Self::workspace_requirements) returns for
    /// `request`.
    ///
    /// # Errors
    ///
    /// - The errors of [`workspace_requirements`](Self::workspace_requirements).
    /// - `AllocationFailure` if the workspace cannot be allocated.
    pub fn workspace<U: NumericScalar<Real = T::Real>>(
        &self,
        request: IirRequest<'_>,
    ) -> Result<SignalWorkspace<U>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(request)?)
    }
    /// Padded forward and backward staging plus delays of one record.
    fn record_len(&self, samples: usize, padding: usize) -> Result<usize, SciOxideError> {
        add(size(add(samples, size(padding, 2)?)?, 2)?, self.delays)
    }
    /// One axis lane: input, output and its padded record.
    fn lane_len(&self, samples: usize, padding: usize) -> Result<usize, SciOxideError> {
        add(size(samples, 2)?, self.record_len(samples, padding)?)
    }

    fn sample_domain<U: NumericScalar<Real = T::Real>>(&self) -> Result<(), SciOxideError> {
        let complex =
            U::from_components(T::Real::ZERO, T::Real::ONE).imaginary_component() != T::Real::ZERO;
        if !complex
            && self.sections.iter().any(|s| {
                s.a.iter()
                    .chain(&s.b)
                    .any(|x| x.imaginary_component() != T::Real::ZERO)
            })
        {
            return Err(unsupported(
                scioxide_core::argument!("self"),
                "complex coefficients require complex samples",
            ));
        }
        Ok(())
    }
    fn validate_state<U: NumericScalar<Real = T::Real>>(
        &self,
        state: &IirState<U>,
        channels: usize,
    ) -> Result<(), SciOxideError> {
        if state.channels != channels || state.per_channel != self.delays {
            return Err(SciOxideError::dimension_mismatch(
                [channels, self.delays],
                [state.channels, state.per_channel],
            ));
        }
        inputs(&state.data)?;
        self.sample_domain::<U>()
    }
    /// Fill `state` with the initial delays of one pass. Steady state is the
    /// constant-input response to `first`, including the gain of preceding
    /// sections.
    pub(super) fn initialize<U: NumericScalar<Real = T::Real>>(
        &self,
        state: &mut [U],
        mode: IirInitialization<'_, U>,
        mut first: U,
    ) -> Result<(), SciOxideError> {
        state.fill(U::ZERO);
        match mode {
            IirInitialization::Zero => {}
            IirInitialization::State(given) => {
                exact_len(state.len(), given.len())?;
                inputs(given)?;
                state.copy_from_slice(given);
            }
            IirInitialization::SteadyState => {
                let mut offset = 0;
                for s in &self.sections {
                    let n = s.a.len().max(s.b.len()) - 1;
                    let b = crate::util::sum_products(s.b.iter().map(|x| (*x, T::ONE)))?;
                    let a = crate::util::sum_products(s.a.iter().map(|x| (*x, T::ONE)))?;
                    if a == T::ZERO {
                        return Err(SciOxideError::domain_error(
                            scioxide_core::argument!("self"),
                            scioxide_core::DomainReason::Undefined,
                            "steady state is undefined for a zero DC denominator",
                        ));
                    }
                    let y = check(first * lift::<T, U>(b / a))?;
                    let mut next = U::ZERO;
                    for j in (0..n).rev() {
                        next = check(
                            next + first * lift::<T, U>(s.b.get(j + 1).copied().unwrap_or(T::ZERO))
                                - y * lift::<T, U>(s.a.get(j + 1).copied().unwrap_or(T::ZERO)),
                        )?;
                        state[offset + j] = next;
                    }
                    offset += n;
                    first = y;
                }
            }
        }
        Ok(())
    }
    fn run<U: NumericScalar<Real = T::Real>>(
        &self,
        input: &[U],
        output: &mut [U],
        state: &mut [U],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        for (x, y) in input.iter().zip(output) {
            execution.check()?;
            let mut value = *x;
            let mut offset = 0;
            for s in &self.sections {
                let n = s.a.len().max(s.b.len()) - 1;
                let out = check(
                    lift::<T, U>(s.b[0]) * value + if n == 0 { U::ZERO } else { state[offset] },
                )?;
                for j in 0..n {
                    let next = if j + 1 < n {
                        state[offset + j + 1]
                    } else {
                        U::ZERO
                    };
                    state[offset + j] = check(
                        next + lift::<T, U>(s.b.get(j + 1).copied().unwrap_or(T::ZERO)) * value
                            - lift::<T, U>(s.a.get(j + 1).copied().unwrap_or(T::ZERO)) * out,
                    )?;
                }
                offset += n;
                value = out;
            }
            *y = value;
        }
        Ok(())
    }
    /// Filters one chunk of a single-channel record and advances `state`.
    ///
    /// Writes `input.len()` samples to the start of `output` and replaces
    /// `state` with the delays after the last sample. Complex coefficients
    /// need complex samples. The workspace must hold
    /// `workspace_requirements(IirRequest::Filter(SignalLayout::Contiguous(input.len())))`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `state` is not one channel of
    ///   [`delays`](Self::delays) values, if `output` is shorter than
    ///   `input`, or if the workspace is too small.
    /// - `NonFinite` if a sample or a delay value is NaN or infinite.
    /// - `InvalidInput` if the coefficients are complex and the samples real.
    /// - `NumericalInstability` (`Overflow`) if an output or delay value
    ///   overflows.
    /// - `ExecutionStopped` on cancellation or a passed deadline.
    pub fn filter_into<U: NumericScalar<Real = T::Real>>(
        &self,
        input: &[U],
        output: &mut [U],
        state: &mut IirState<U>,
        workspace: &mut SignalWorkspace<U>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.validate_state(state, 1)?;
        inputs(input)?;
        capacity(input.len(), output.len())?;
        let buffers =
            workspace.buffers(&self.workspace_requirements(IirRequest::Filter(
                SignalLayout::Contiguous(input.len()),
            ))?)?;
        execution.check()?;
        let (y, s) = buffers.values.split_at_mut(input.len());
        s.copy_from_slice(&state.data);
        self.run(input, y, s, execution)?;
        execution.check()?;
        output[..input.len()].copy_from_slice(y);
        state.data.copy_from_slice(s);
        Ok(())
    }
    /// Filters `input` from zero initial delays and returns the output.
    ///
    /// This is the owned form of [`filter_into`](Self::filter_into).
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `InvalidInput` if the coefficients are complex and the samples real.
    /// - `NumericalInstability` (`Overflow`) if an output or delay value
    ///   overflows.
    pub fn filter<U: NumericScalar<Real = T::Real>>(
        &self,
        input: &[U],
    ) -> Result<Vec<U>, SciOxideError> {
        let mut out = zeros(input.len())?;
        let mut state = IirState::new(self, 1)?;
        let mut workspace =
            self.workspace(IirRequest::Filter(SignalLayout::Contiguous(input.len())))?;
        self.filter_into(
            input,
            &mut out,
            &mut state,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(out)
    }
    /// Filters every lane of `input` along `axis` as an independent channel.
    ///
    /// `state` holds one channel per lane, with the lanes in row-major order
    /// of the shape without `axis`. Lanes may run in parallel under the
    /// execution policy of the filter. Each lane runs the recurrence of
    /// [`filter_into`](Self::filter_into), so the result does not depend on
    /// the number of threads. The workspace must hold
    /// `workspace_requirements(IirRequest::Filter(SignalLayout::Axis { shape, axis }))`
    /// for the shape of `input`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the shapes of `input` and `output` differ, if
    ///   `state` is not one channel of [`delays`](Self::delays) values per
    ///   lane, or if the workspace is too small.
    /// - `InvalidInput` if `axis` is not below the rank of `input`, or if the
    ///   coefficients are complex and the samples real.
    /// - `NonFinite` if a delay value is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if a sample is NaN or infinite,
    ///   or if an output or delay value overflows.
    /// - `ExecutionStopped` on cancellation or a passed deadline.
    pub fn filter_axis_into<U: NumericScalar<Real = T::Real>>(
        &self,
        input: NdView<'_, '_, U>,
        axis: usize,
        mut output: NdViewMut<'_, '_, U>,
        state: &mut IirState<U>,
        workspace: &mut SignalWorkspace<U>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (n, total) = crate::axis::shape(&input, &output, axis)?;
        let channels = input.layout().axis(axis)?.len();
        self.validate_state(state, channels)?;
        let blocks = workspace
            .buffers(
                &self.workspace_requirements(IirRequest::Filter(SignalLayout::Axis {
                    shape: input.layout().shape().extents(),
                    axis,
                }))?,
            )?
            .values;
        execution.check()?;
        if total == 0 {
            return Ok(());
        }
        let stride = 2 * n + self.delays;
        crate::axis::gather_blocks(&input, axis, blocks, stride, |x| check(x).map(|_| ()))?;
        for (lane, block) in blocks.chunks_exact_mut(stride).enumerate() {
            block[2 * n..]
                .copy_from_slice(&state.data[lane * self.delays..(lane + 1) * self.delays]);
        }
        execute_blocks(
            blocks,
            stride,
            self.options.execution_policy(),
            execution,
            |chunk| {
                for block in chunk.chunks_exact_mut(stride) {
                    let (x, rest) = block.split_at_mut(n);
                    let (y, s) = rest.split_at_mut(n);
                    self.run(x, y, s, execution)?;
                }
                Ok(())
            },
        )?;
        execution.check()?;
        crate::axis::publish_blocks(&mut output, axis, blocks, stride, n)?;
        for (lane, block) in blocks.chunks_exact(stride).enumerate() {
            state.data[lane * self.delays..(lane + 1) * self.delays]
                .copy_from_slice(&block[2 * n..]);
        }
        Ok(())
    }
    /// Reject padding and initialization choices that do not fit a record of `n` samples.
    fn validate_fb<U: NumericScalar<Real = T::Real>>(
        &self,
        n: usize,
        options: ForwardBackwardOptions<'_, U>,
    ) -> Result<(), SciOxideError> {
        let pad = options.padding.length();
        if pad != 0 && pad >= n {
            return Err(inconsistent(
                scioxide_core::argument!("options.padding"),
                "padding must be shorter than the record",
            ));
        }
        for mode in [options.forward, options.backward] {
            if let IirInitialization::State(s) = mode {
                exact_len(self.delays, s.len())?;
                inputs(s)?;
            }
        }
        Ok(())
    }
    /// Forward-backward kernel on a validated record; `work` holds at least
    /// `record_len(input.len(), padding)` values. Output is written only after
    /// every fallible step and a final execution check succeed.
    fn fb<U: NumericScalar<Real = T::Real>>(
        &self,
        input: &[U],
        output: &mut [U],
        work: &mut [U],
        options: ForwardBackwardOptions<'_, U>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = input.len();
        let pad = options.padding.length();
        execution.check()?;
        if n == 0 {
            return Ok(());
        }
        let m = n + 2 * pad;
        let (x, rest) = work.split_at_mut(m);
        let (y, state) = rest.split_at_mut(m);
        x[pad..pad + n].copy_from_slice(input);
        for i in 0..pad {
            let left = input[pad - i];
            let right = input[n - 2 - i];
            x[i] = check(match options.padding {
                IirPadding::Odd(_) => input[0] + (input[0] - left),
                IirPadding::Even(_) => left,
                _ => input[0],
            })?;
            x[pad + n + i] = check(match options.padding {
                IirPadding::Odd(_) => input[n - 1] + (input[n - 1] - right),
                IirPadding::Even(_) => right,
                _ => input[n - 1],
            })?;
        }
        self.initialize(state, options.forward, x[0])?;
        self.run(x, y, state, execution)?;
        y.reverse();
        self.initialize(state, options.backward, y[0])?;
        self.run(y, x, state, execution)?;
        execution.check()?;
        for (i, v) in output[..n].iter_mut().enumerate() {
            *v = x[m - 1 - pad - i];
        }
        Ok(())
    }
    /// Filters `input` forward and then backward, which gives zero phase.
    ///
    /// The record is extended at both ends as `options` specifies, filtered
    /// forward, reversed, filtered again and reversed back, and the extension
    /// is removed. Each pass starts from its initial state in `options`. For
    /// real coefficients the result has the response `|H|²` with zero phase,
    /// apart from transients at the record ends, and twice the filter order.
    /// The workspace must hold
    /// `workspace_requirements(IirRequest::ForwardBackward(SignalLayout::Contiguous(n), padding))`
    /// with `n = input.len()` and the padding of `options`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output` is shorter than `input`, if the
    ///   workspace is too small, or if an explicit initial state does not
    ///   have [`delays`](Self::delays) values.
    /// - `InvalidInput` if the padding length is positive and not smaller than
    ///   `input.len()`, or if the coefficients are complex and the samples
    ///   real.
    /// - `NonFinite` if a sample or an explicit initial delay is NaN or
    ///   infinite.
    /// - `DomainError` (`Undefined`) if a pass starts in steady state and a
    ///   section denominator vanishes at `z = 1`.
    /// - `NumericalInstability` (`Overflow`) if an extended sample, an output
    ///   or a delay value overflows.
    /// - `ExecutionStopped` on cancellation or a passed deadline.
    pub fn forward_backward_into<U: NumericScalar<Real = T::Real>>(
        &self,
        input: &[U],
        options: ForwardBackwardOptions<'_, U>,
        output: &mut [U],
        workspace: &mut SignalWorkspace<U>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.sample_domain::<U>()?;
        inputs(input)?;
        capacity(input.len(), output.len())?;
        let buffers = workspace.buffers(&self.workspace_requirements(
            IirRequest::ForwardBackward(SignalLayout::Contiguous(input.len()), options.padding),
        )?)?;
        self.validate_fb(input.len(), options)?;
        self.fb(input, output, buffers.values, options, execution)
    }
    /// Filters `input` forward and then backward with the end treatment of
    /// `options`, and returns the output.
    ///
    /// This is the owned form of
    /// [`forward_backward_into`](Self::forward_backward_into), and it costs
    /// `O(d·(n + 2p))` operations for `d` [`delays`](Self::delays), `n`
    /// samples and a padding length `p`.
    /// [`SecondOrderSections::forward_backward`] and
    /// [`TransferFunction::forward_backward`] choose the end treatment from the
    /// filter and the record.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if the padding length is positive and
    ///   not smaller than `input.len()`.
    /// - `InvalidInput` if the coefficients are complex and the samples real.
    /// - `DimensionMismatch` if an explicit initial state does not have
    ///   [`delays`](Self::delays) values.
    /// - `NonFinite` if a sample or an explicit initial delay is NaN or
    ///   infinite.
    /// - `DomainError` (`Undefined`) if a pass starts in steady state and a
    ///   section denominator vanishes at `z = 1`.
    /// - `NumericalInstability` (`Overflow`) if an extended sample, an output
    ///   or a delay value overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{ForwardBackwardOptions, IirDomain, IirFilter, IirInitialization};
    /// use scioxide_signal::{SignalExecutionOptions, TransferFunction};
    /// // y[n] = 0.5 x[n] + 0.5 y[n − 1] has unit DC gain.
    /// let transfer = TransferFunction::new(IirDomain::Digital, &[0.5_f64], &[1.0, -0.5])?;
    /// let filter = IirFilter::from_transfer(&transfer, SignalExecutionOptions::new())?;
    /// let steady = IirInitialization::SteadyState;
    /// let options = ForwardBackwardOptions::new()
    ///     .with_forward(steady)
    ///     .with_backward(steady);
    /// let input = [1.0_f64; 8];
    /// let output = filter.forward_backward_with_options(&input, options)?;
    /// assert!(output.iter().all(|y| (y - 1.0).abs() < 1e-14));
    /// // Passes from rest show transients at the ends.
    /// let rest = filter.forward_backward_with_options(&input, ForwardBackwardOptions::new())?;
    /// assert!((rest[0] - 1.0).abs() > 0.1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn forward_backward_with_options<U: NumericScalar<Real = T::Real>>(
        &self,
        input: &[U],
        options: ForwardBackwardOptions<'_, U>,
    ) -> Result<Vec<U>, SciOxideError> {
        let mut output = zeros(input.len())?;
        let mut workspace = self.workspace(IirRequest::ForwardBackward(
            SignalLayout::Contiguous(input.len()),
            options.padding(),
        ))?;
        self.forward_backward_into(
            input,
            options,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Forward-backward filtering with odd extension over
    /// `min(3·(delays + 1), n − 1)` samples and steady-state delays in both
    /// passes.
    fn forward_backward_default<U: NumericScalar<Real = T::Real>>(
        &self,
        input: &[U],
    ) -> Result<Vec<U>, SciOxideError> {
        let length = size(add(self.delays, 1)?, 3)?.min(input.len().saturating_sub(1));
        self.forward_backward_with_options(
            input,
            ForwardBackwardOptions::new()
                .with_padding(IirPadding::Odd(length))
                .with_forward(IirInitialization::SteadyState)
                .with_backward(IirInitialization::SteadyState),
        )
    }
    /// Filters every lane of `input` along `axis` forward and backward.
    ///
    /// Each lane is filtered as by
    /// [`forward_backward_into`](Self::forward_backward_into), with the same
    /// `options`, including any explicit initial state. Lanes may run in
    /// parallel under the execution policy of the filter. An array without
    /// lanes returns at once, except that a zero-length axis cannot be padded.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the shapes of `input` and `output` differ, if
    ///   the workspace is too small, or if an explicit initial state does not
    ///   have [`delays`](Self::delays) values.
    /// - `InvalidInput` if `axis` is not below the rank of `input`, if the
    ///   padding length is positive and not smaller than the lane length, or
    ///   if the coefficients are complex and the samples real.
    /// - `NonFinite` if an explicit initial delay is NaN or infinite.
    /// - `DomainError` (`Undefined`) if a pass starts in steady state and a
    ///   section denominator vanishes at `z = 1`.
    /// - `NumericalInstability` (`Overflow`) if a sample is NaN or infinite,
    ///   or if an output or delay value overflows.
    /// - `ExecutionStopped` on cancellation or a passed deadline.
    pub fn forward_backward_axis_into<U: NumericScalar<Real = T::Real>>(
        &self,
        input: NdView<'_, '_, U>,
        axis: usize,
        options: ForwardBackwardOptions<'_, U>,
        mut output: NdViewMut<'_, '_, U>,
        workspace: &mut SignalWorkspace<U>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (n, total) = crate::axis::shape(&input, &output, axis)?;
        self.sample_domain::<U>()?;
        execution.check()?;
        if total == 0 {
            if n == 0 && options.padding.length() != 0 {
                return Err(inconsistent(
                    scioxide_core::argument!("options.padding"),
                    "an empty axis cannot be padded",
                ));
            }
            return Ok(());
        }
        self.validate_fb(n, options)?;
        let stride = self.lane_len(n, options.padding.length())?;
        let blocks = workspace
            .buffers(&self.workspace_requirements(IirRequest::ForwardBackward(
                SignalLayout::Axis {
                    shape: input.layout().shape().extents(),
                    axis,
                },
                options.padding,
            ))?)?
            .values;
        crate::axis::gather_blocks(&input, axis, blocks, stride, |x| check(x).map(|_| ()))?;
        execute_blocks(
            blocks,
            stride,
            self.options.execution_policy(),
            execution,
            |chunk| {
                for block in chunk.chunks_exact_mut(stride) {
                    let (x, rest) = block.split_at_mut(n);
                    let (y, w) = rest.split_at_mut(n);
                    self.fb(x, y, w, options, execution)?;
                }
                Ok(())
            },
        )?;
        execution.check()?;
        crate::axis::publish_blocks(&mut output, axis, blocks, stride, n)?;
        Ok(())
    }
}
impl<T: NumericScalar> SecondOrderSections<T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Filters `input` with the cascade, causally from zero initial delays, and
    /// returns the output.
    ///
    /// Each section runs the transposed direct form II recurrence of
    /// [`IirFilter`], in section order, so the cost is `O(L)` operations per
    /// sample for `L` sections. This is [`IirFilter::from_sos`] followed by
    /// [`IirFilter::filter`]; the [`IirFilter`] also filters in chunks, along
    /// an axis and from a given state. Complex coefficients need complex
    /// samples.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if the sections are analog.
    /// - `InvalidInput` if the coefficients are complex and the samples real.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if an output or delay value
    ///   overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{IirBand, IirDomain, IirFamily, design_iir};
    /// let band = IirBand::Lowpass(0.2_f64);
    /// let sos = design_iir(4, IirFamily::Butterworth, band, IirDomain::Digital)?.to_sos()?;
    /// // The step response settles at the unit DC gain.
    /// let step = sos.filter(&[1.0_f64; 200])?;
    /// assert!((step[199] - 1.0).abs() < 1e-9);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn filter<U: NumericScalar<Real = T::Real>>(
        &self,
        input: &[U],
    ) -> Result<Vec<U>, SciOxideError> {
        require_digital(self.domain, "digital filtering requires digital sections")?;
        IirFilter::from_sos(self, SignalExecutionOptions::new())?.filter(input)
    }
    /// Filters `input` forward and then backward with the cascade, which gives
    /// zero phase, and returns the output.
    ///
    /// For a record of `n` samples and a cascade of `L` sections with
    /// `d = 2L` delays, the record is extended at both ends by odd reflection
    /// over `min(3·(d + 1), n − 1)` samples, both passes start in
    /// [`IirInitialization::SteadyState`], and the extension is removed, as
    /// described at [`IirFilter::forward_backward_into`]. This form fixes that
    /// end treatment because its length depends on the filter and the record;
    /// [`IirFilter::forward_backward_with_options`] takes any other. For real
    /// coefficients the result has the response `|H|²` with zero phase, apart
    /// from transients at the record ends. The cost is `O(L)` operations per
    /// extended sample and pass.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if the sections are analog.
    /// - `InvalidInput` if the coefficients are complex and the samples real.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `DomainError` (`Undefined`) if a section denominator vanishes at
    ///   `z = 1`, where no steady state exists.
    /// - `NumericalInstability` (`Overflow`) if an extended sample, an output
    ///   or a delay value overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{IirBand, IirDomain, IirFamily, design_iir};
    /// let band = IirBand::Lowpass(0.2_f64);
    /// let sos = design_iir(4, IirFamily::Butterworth, band, IirDomain::Digital)?.to_sos()?;
    /// // Steady-state delays leave a constant record without end transients.
    /// let output = sos.forward_backward(&[2.0_f64; 50])?;
    /// assert!(output.iter().all(|y| (y - 2.0).abs() < 1e-12));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn forward_backward<U: NumericScalar<Real = T::Real>>(
        &self,
        input: &[U],
    ) -> Result<Vec<U>, SciOxideError> {
        require_digital(self.domain, "digital filtering requires digital sections")?;
        IirFilter::from_sos(self, SignalExecutionOptions::new())?.forward_backward_default(input)
    }
}
impl<T: NumericScalar> TransferFunction<T>
where
    Complex<T::Real>: NumericScalar<Real = T::Real>,
{
    /// Filters `input` causally from zero initial delays and returns the
    /// output.
    ///
    /// The transfer function runs as one section of order
    /// `d = max(len(a), len(b)) − 1` in the transposed direct form II of
    /// [`IirFilter`], at a cost of `O(d)` operations per sample. This is
    /// [`IirFilter::from_transfer`] followed by [`IirFilter::filter`]. A
    /// high-order direct form is sensitive to coefficient rounding, so store
    /// such a filter as [`SecondOrderSections`]. Complex coefficients need
    /// complex samples.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if the transfer function is analog.
    /// - `InvalidInput` if the coefficients are complex and the samples real.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `NumericalInstability` (`Overflow`) if an output or delay value
    ///   overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{IirDomain, TransferFunction};
    /// // Two-point moving average y[n] = (x[n] + x[n − 1]) / 2.
    /// let transfer = TransferFunction::new(IirDomain::Digital, &[0.5_f64, 0.5], &[1.0])?;
    /// let output = transfer.filter(&[2.0_f64, 4.0, 6.0])?;
    /// for (y, e) in output.iter().zip([1.0, 3.0, 5.0]) {
    ///     assert!((y - e).abs() < 1e-15);
    /// }
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn filter<U: NumericScalar<Real = T::Real>>(
        &self,
        input: &[U],
    ) -> Result<Vec<U>, SciOxideError> {
        require_digital(
            self.domain,
            "digital filtering requires digital coefficients",
        )?;
        IirFilter::from_transfer(self, SignalExecutionOptions::new())?.filter(input)
    }
    /// Filters `input` forward and then backward, which gives zero phase, and
    /// returns the output.
    ///
    /// For a record of `n` samples and `d = max(len(a), len(b)) − 1` delays,
    /// the record is extended at both ends by odd reflection over
    /// `min(3·(d + 1), n − 1)` samples, which is `3·max(len(a), len(b))` for a
    /// long record, both passes start in [`IirInitialization::SteadyState`],
    /// and the extension is removed, as described at
    /// [`IirFilter::forward_backward_into`]. This form fixes that end
    /// treatment because its length depends on the filter and the record;
    /// [`IirFilter::forward_backward_with_options`] takes any other. For real
    /// coefficients the result has the response `|H|²` with zero phase, apart
    /// from transients at the record ends. The cost is `O(d)` operations per
    /// extended sample and pass.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if the transfer function is analog.
    /// - `InvalidInput` if the coefficients are complex and the samples real.
    /// - `NonFinite` if a sample is NaN or infinite.
    /// - `DomainError` (`Undefined`) if the denominator vanishes at `z = 1`,
    ///   where no steady state exists.
    /// - `NumericalInstability` (`Overflow`) if an extended sample, an output
    ///   or a delay value overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_signal::{IirDomain, TransferFunction};
    /// // y[n] = 0.5 x[n] + 0.5 y[n − 1] has unit DC gain.
    /// let transfer = TransferFunction::new(IirDomain::Digital, &[0.5_f64], &[1.0, -0.5])?;
    /// // Steady-state delays leave a constant record without end transients.
    /// let output = transfer.forward_backward(&[3.0_f64; 20])?;
    /// assert!(output.iter().all(|y| (y - 3.0).abs() < 1e-14));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn forward_backward<U: NumericScalar<Real = T::Real>>(
        &self,
        input: &[U],
    ) -> Result<Vec<U>, SciOxideError> {
        require_digital(
            self.domain,
            "digital filtering requires digital coefficients",
        )?;
        IirFilter::from_transfer(self, SignalExecutionOptions::new())?
            .forward_backward_default(input)
    }
}
/// Rejects an analog representation, the receiver `self` of a filtering call,
/// with the detail of the matching [`IirFilter`] constructor.
fn require_digital(domain: IirDomain, detail: &'static str) -> Result<(), SciOxideError> {
    if domain == IirDomain::Digital {
        Ok(())
    } else {
        Err(invalid(scioxide_core::argument!("self"), detail))
    }
}
fn lift<T: NumericScalar, U: NumericScalar<Real = T::Real>>(x: T) -> U {
    U::from_components(x.real_component(), x.imaginary_component())
}
