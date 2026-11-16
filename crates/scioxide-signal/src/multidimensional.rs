//! Multidimensional analytic signals, Riesz transforms and monogenic signals.
use crate::fourier::Transform;
use crate::util::{copy, fft_exponent, input_scale, restore_fft, strides};
use crate::util::{empty, exact_len, exact_shape, invalid, scalar_from_usize_rounded};
use crate::{SignalExecutionOptions, SignalRequirements, SignalWorkspace};
use alloc::vec::Vec;
use scioxide_core::backend::standard::scale;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    Complex, ExecutionContext, ExecutionPolicy, NdView, NdViewMut, NumericScalar, RealScalar,
    SciOxideError, Shape, StridedLayout,
};
use scioxide_fft::{FftDirection, FftLayout, FftNdPlanSpec, Normalization};

/// Writes `values` to `output` in its logical order.
fn publish<T: Copy>(values: &[T], output: &mut NdViewMut<'_, '_, T>) {
    let mut values = values.iter();
    output.for_each_mut(|out| {
        if let Some(&value) = values.next() {
            *out = value;
        }
    });
}
#[derive(Clone, Debug)]
struct SpectralPlan<S: RealScalar> {
    forward: Transform<S>,
    inverse: Transform<S>,
    complex: Option<(Transform<S>, Transform<S>)>,
    shape: Vec<usize>,
    axes: Vec<usize>,
    strides: Vec<usize>,
    packed_shape: Vec<usize>,
    packed_strides: Vec<usize>,
    len: usize,
    packed_len: usize,
    fft_scratch: usize,
}
impl<S: RealScalar> SpectralPlan<S> {
    fn new(
        shape: &[usize],
        axes: &[usize],
        policy: ExecutionPolicy,
        context: ExecutionContext<'_>,
        complex: bool,
    ) -> Result<Self, SciOxideError> {
        if axes.is_empty() {
            return Err(empty(
                scioxide_core::argument!("axes"),
                "at least one axis must be selected",
            ));
        }
        FftNdPlanSpec::new(
            shape,
            axes,
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Real,
        )?;
        let forward = Transform::new(shape, axes, FftDirection::Forward, true, policy, context)?;
        let inverse = Transform::new(
            shape,
            axes,
            FftDirection::Inverse,
            complex || axes.len() == 1,
            policy,
            context,
        )?;
        let complex = if complex {
            Some((
                Transform::new(shape, axes, FftDirection::Forward, false, policy, context)?,
                Transform::new(shape, axes, FftDirection::Inverse, false, policy, context)?,
            ))
        } else {
            None
        };
        let mut fft_scratch = forward.complex_work()?.max(inverse.complex_work()?);
        if let Some((a, b)) = &complex {
            fft_scratch = fft_scratch.max(a.complex_work()?).max(b.complex_work()?);
        }
        let mut packed_shape = copy(shape)?;
        let axis = axes[axes.len() - 1];
        packed_shape[axis] = shape[axis] / 2 + 1;
        Ok(Self {
            forward,
            inverse,
            complex,
            shape: copy(shape)?,
            axes: copy(axes)?,
            strides: strides(shape)?,
            packed_strides: strides(&packed_shape)?,
            packed_len: Shape::new(&packed_shape)?.len(),
            packed_shape,
            len: Shape::new(shape)?.len(),
            fft_scratch,
        })
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn axes(&self) -> &[usize] {
        &self.axes
    }
    fn layout(&self) -> Result<StridedLayout<'_>, SciOxideError> {
        StridedLayout::new(self.shape(), &self.strides)
    }
    fn layouts(&self) -> Result<(StridedLayout<'_>, StridedLayout<'_>), SciOxideError> {
        Ok((
            self.layout()?,
            StridedLayout::new(&self.packed_shape, &self.packed_strides)?,
        ))
    }
    fn coordinate(&self, index: usize, axis: usize) -> usize {
        index / self.strides[axis] % self.shape[axis]
    }
    fn packed_index(&self, index: usize) -> usize {
        let axis = self.axes[self.axes.len() - 1];
        let inner = self.strides[axis];
        let packed = self.packed_shape[axis] * inner;
        index / packed * self.shape[axis] * inner + index % packed
    }
}
/// Plan for the single-orthant analytic signal of real arrays along selected
/// axes.
///
/// The plan treats the array as one period of a periodic signal on the selected
/// axes. It multiplies the DFT over those axes by the product of the
/// one-dimensional weights of [`hilbert`](fn@crate::hilbert) on each selected axis
/// (`1` at DC and an even-length Nyquist bin, `2` at positive and `0` at
/// negative frequencies) and transforms back. The spectrum of the result is
/// therefore confined to one orthant. With one selected axis the real part is
/// the input exactly and the imaginary part is the periodic Hilbert transform.
/// With several axes the real part generally differs from the input. Other axes
/// are independent batches. The cost is `O(N log N)` for `N` elements.
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext, NdView, NdViewMut, StridedLayout};
/// use scioxide_signal::{AnalyticSignalNdPlan, SignalExecutionOptions};
/// let plan = AnalyticSignalNdPlan::<f64>::new(&[4], &[0], SignalExecutionOptions::new())?;
/// let mut ws = plan.workspace(())?;
/// let mut out = [Complex::new(0.0, 0.0); 4];
/// plan.analytic_into(
///     NdView::new(&[1.0, 0.0, -1.0, 0.0], StridedLayout::new(&[4], &[1])?)?,
///     NdViewMut::new(&mut out, StridedLayout::new(&[4], &[1])?)?,
///     &mut ws,
///     ExecutionContext::sequential_native(),
/// )?;
/// // The quadrature of cos(πn/2) is sin(πn/2).
/// assert!((out[1].im - 1.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// S. L. Hahn, "Multidimensional complex signals with single-orthant spectra",
/// *Proc. IEEE* 80(8), 1287–1300, 1992.
#[derive(Clone, Debug)]
pub struct AnalyticSignalNdPlan<S: RealScalar> {
    spectral: SpectralPlan<S>,
    requirements: SignalRequirements,
}
impl<S: RealScalar> AnalyticSignalNdPlan<S> {
    /// Prepares a plan for arrays of shape `shape`, transformed along `axes`,
    /// using [`ExecutionContext::sequential_native`].
    ///
    /// The execution policy of `options` applies to the FFTs.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `axes` is empty, an axis is repeated, or a selected
    ///   extent of `shape` is zero.
    /// - `OutOfBounds` if an axis is not below the rank of `shape`.
    /// - `SizeOverflow` or `AllocationFailure` if the plan storage cannot be sized
    ///   or allocated.
    pub fn new(
        shape: &[usize],
        axes: &[usize],
        options: SignalExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        Self::build(shape, axes, options, ExecutionContext::sequential_native())
    }
    /// Prepares a plan like [`new`](Self::new), planning the FFTs with `cx`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`new`](Self::new), and `ExecutionStopped` if `cx`
    /// is cancelled or past its deadline.
    pub fn build(
        shape: &[usize],
        axes: &[usize],
        options: SignalExecutionOptions,
        cx: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        let spectral = SpectralPlan::new(shape, axes, options.execution_policy(), cx, false)?;
        let complex = checked_add(
            checked_add(
                spectral.len,
                spectral.packed_len,
                scioxide_core::quantity!("spectral_scratch"),
            )?,
            spectral.fft_scratch,
            scioxide_core::quantity!("spectral_scratch"),
        )?;
        Ok(Self {
            requirements: SignalRequirements::new(0, spectral.len, complex),
            spectral,
        })
    }
    /// Returns the shape of the input and the output.
    pub fn shape(&self) -> &[usize] {
        self.spectral.shape()
    }
    /// Returns the selected axes.
    pub fn axes(&self) -> &[usize] {
        self.spectral.axes()
    }
    /// Returns the workspace that one call to
    /// [`analytic_into`](Self::analytic_into) or
    /// [`envelope_into`](Self::envelope_into) needs.
    ///
    /// # Errors
    ///
    /// This function does not fail for a prepared plan.
    pub fn workspace_requirements(
        &self,
        _request: (),
    ) -> Result<SignalRequirements, SciOxideError> {
        Ok(self.requirements)
    }
    /// Allocates a workspace for [`workspace_requirements`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the allocator fails.
    pub fn workspace(&self, request: ()) -> Result<SignalWorkspace<S>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(request)?)
    }
    /// Writes the envelope `|z|` of the analytic signal `z` of `input` to
    /// `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` does not have the prepared
    ///   shape, or `ws` is too small.
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` if a result overflows.
    /// - `ExecutionStopped` if `cx` is cancelled or past its deadline.
    pub fn envelope_into(
        &self,
        input: NdView<'_, '_, S>,
        mut output: NdViewMut<'_, '_, S>,
        ws: &mut SignalWorkspace<S>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        exact_shape(self.shape(), output.layout().shape().extents())?;
        let buffers = ws.buffers(&self.requirements)?;
        let samples = buffers.real;
        let result = self.transform(input, samples, buffers.complex, cx)?;
        let magnitudes = &mut samples[..self.spectral.len];
        for (out, z) in magnitudes.iter_mut().zip(result) {
            *out = crate::util::finite(z.norm())?;
        }
        cx.check()?;
        publish(magnitudes, &mut output);
        Ok(())
    }
    fn transform<'a>(
        &self,
        input: NdView<'_, '_, S>,
        samples: &mut [S],
        scratch: &'a mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<&'a [Complex<S>], SciOxideError> {
        exact_shape(self.shape(), input.layout().shape().extents())?;
        if let Some(values) = crate::util::vector_storage(&input) {
            return self.transform_samples(|| values.iter(), samples, scratch, context);
        }
        self.transform_samples(|| input.iter(), samples, scratch, context)
    }
    // Borrowed tensors and slice conveniences share one transform kernel.
    fn transform_samples<'a, 'i, I: ExactSizeIterator<Item = &'i S>>(
        &self,
        values: impl Fn() -> I,
        samples: &mut [S],
        scratch: &'a mut [Complex<S>],
        context: ExecutionContext<'_>,
    ) -> Result<&'a [Complex<S>], SciOxideError> {
        context.check()?;
        exact_len(self.spectral.len, values().len())?;
        let maximum = input_scale(values().copied())?;
        let exponent = fft_exponent(maximum, self.spectral.len)?;
        let samples = &mut samples[..self.spectral.len];
        samples
            .iter_mut()
            .zip(values())
            .for_each(|(out, &value)| *out = scale(value, -exponent));
        let (buffer, rest) = scratch.split_at_mut(self.spectral.len);
        let (packed, fft) = rest.split_at_mut(self.spectral.packed_len);
        self.spectral.forward.forward_real(
            samples,
            packed,
            self.spectral.layouts()?,
            fft,
            context,
        )?;
        if self.axes().len() == 1 {
            let axis = self.axes()[0];
            let n = self.shape()[axis];
            let inner = self.spectral.strides[axis];
            if !packed.is_empty() {
                for block in packed.chunks_exact_mut((n / 2 + 1) * inner) {
                    block[..inner].fill(Complex::new(S::ZERO, S::ZERO));
                    let end = if n.is_multiple_of(2) {
                        n / 2
                    } else {
                        n / 2 + 1
                    } * inner;
                    block[inner..end]
                        .iter_mut()
                        .for_each(|z| *z = Complex::new(z.im, -z.re));
                    block[end..].fill(Complex::new(S::ZERO, S::ZERO));
                }
            }
            self.spectral.inverse.inverse_real(
                packed,
                samples,
                self.spectral.layouts()?,
                fft,
                context,
            )?;
            restore_fft(samples, exponent)?;
            buffer
                .iter_mut()
                .zip(values())
                .zip(samples.iter())
                .for_each(|((out, &re), &im)| *out = Complex::new(re, im));
        } else {
            buffer.fill(Complex::new(S::ZERO, S::ZERO));
            let axis = self.axes()[self.axes().len() - 1];
            let inner = self.spectral.strides[axis];
            let row = self.shape()[axis] * inner;
            let packed_row = self.spectral.packed_shape[axis] * inner;
            if !buffer.is_empty() {
                for (out, source) in buffer
                    .chunks_exact_mut(row)
                    .zip(packed.chunks_exact(packed_row))
                {
                    out[..packed_row].copy_from_slice(source);
                }
                for &axis in self.axes() {
                    let n = self.shape()[axis];
                    let stride = self.spectral.strides[axis];
                    for block in buffer.chunks_mut(n * stride) {
                        for (k, bins) in block.chunks_mut(stride).enumerate() {
                            if k > n / 2 {
                                bins.fill(Complex::new(S::ZERO, S::ZERO));
                            } else if k != 0 && !(n.is_multiple_of(2) && k == n / 2) {
                                bins.iter_mut().for_each(|z| *z *= S::TWO);
                            }
                        }
                    }
                }
            }
            self.spectral.inverse.complex_in_place(
                buffer,
                self.spectral.layout()?,
                fft,
                context,
            )?;
            restore_fft(buffer, exponent)?;
        }
        context.check()?;
        Ok(buffer)
    }

    /// Writes the analytic signal of `input` to `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` does not have the prepared
    ///   shape, or `ws` is too small.
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` if a result overflows.
    /// - `ExecutionStopped` if `cx` is cancelled or past its deadline.
    pub fn analytic_into(
        &self,
        input: NdView<'_, '_, S>,
        mut output: NdViewMut<'_, '_, Complex<S>>,
        ws: &mut SignalWorkspace<S>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        exact_shape(self.shape(), output.layout().shape().extents())?;
        let buffers = ws.buffers(&self.requirements)?;
        publish(
            self.transform(input, buffers.real, buffers.complex, cx)?,
            &mut output,
        );
        Ok(())
    }
    /// Returns the analytic signal of `input` as a row-major array of shape
    /// [`shape`](Self::shape).
    ///
    /// This allocates the output and a workspace, and runs with
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` does not have the prepared shape.
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` if a result overflows.
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn analytic(&self, input: NdView<'_, '_, S>) -> Result<Vec<Complex<S>>, SciOxideError> {
        let mut output = try_filled(
            self.spectral.len,
            Complex::new(S::ZERO, S::ZERO),
            scioxide_core::quantity!("analytic_n_d_output"),
        )?;
        self.analytic_into(
            input,
            NdViewMut::new(&mut output, self.spectral.layout()?)?,
            &mut self.workspace(())?,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}

/// Plan for the periodic Riesz transform of real or complex arrays along
/// selected axes.
///
/// The Riesz transform has one component for each selected axis `j`. With `X`
/// the DFT over the selected axes, component `j` is
///
/// ```text
/// Rⱼx = IDFT(−i (ωⱼ / |ω|) X),   ωⱼ = kⱼ / (nⱼ hⱼ),   |ω|² = Σᵢ ωᵢ²,
/// ```
///
/// where `kⱼ` is the signed frequency index on axis `j`, in
/// `(−nⱼ/2, nⱼ/2]`, `nⱼ` is the extent and `hⱼ` the sample spacing of that
/// axis. The multiplier of component `j` is zero where `kⱼ = 0` and, for even
/// `nⱼ`, where `kⱼ = nⱼ/2`. Along one axis the Riesz transform is the periodic
/// Hilbert transform. Other axes are independent batches. The plan prepares the
/// multipliers once. Each call costs one forward FFT and one inverse FFT per
/// component, `O((d + 1) N log N)` for `d` selected axes and `N` elements.
///
/// The output has shape [`output_shape`](Self::output_shape), with the
/// components along a new leading axis in the order of `axes`. The monogenic
/// signal puts the input first, followed by the components.
///
/// ```
/// use scioxide_core::{ExecutionContext, NdView, NdViewMut, StridedLayout};
/// use scioxide_signal::{RieszPlan, SignalExecutionOptions};
/// let plan = RieszPlan::<f64>::new(&[4], &[0], &[1.0], SignalExecutionOptions::new())?;
/// let input = NdView::new(&[1.0_f64; 4], StridedLayout::new(&[4], &[1])?)?;
/// let mut ws = plan.workspace(())?;
/// let mut out = [1.0; 4];
/// plan.riesz_into(
///     input,
///     NdViewMut::new(&mut out, StridedLayout::new(&[1, 4], &[4, 1])?)?,
///     &mut ws,
///     ExecutionContext::sequential_native(),
/// )?;
/// // A constant has only a DC component, which the transform removes.
/// assert!(out.iter().all(|x| x.abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. Felsberg and G. Sommer, "The monogenic signal", *IEEE Trans. Signal
/// Process.* 49(12), 3136–3144, 2001.
#[derive(Clone, Debug)]
pub struct RieszPlan<S: RealScalar> {
    spectral: SpectralPlan<S>,
    spacing: Vec<S>,
    multipliers: Vec<S>,
    packed_multipliers: Vec<S>,
    output_shape: Vec<usize>,
    monogenic_shape: Vec<usize>,
    requirements: SignalRequirements,
}
impl<S: RealScalar> RieszPlan<S> {
    /// Prepares a plan for arrays of shape `shape`, transformed along `axes`
    /// with sample spacing `spacing[j]` on axis `axes[j]`, using
    /// [`ExecutionContext::sequential_native`].
    ///
    /// The execution policy of `options` applies to the FFTs.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `spacing` and `axes` have different lengths.
    /// - `InvalidInput` if a spacing is not positive.
    /// - `NonFinite` if a spacing is NaN or infinite.
    /// - `InvalidInput` if `axes` is empty, an axis is repeated, or a selected
    ///   extent of `shape` is zero.
    /// - `OutOfBounds` if an axis is not below the rank of `shape`.
    /// - `SizeOverflow` or `AllocationFailure` if the plan storage cannot be sized
    ///   or allocated.
    pub fn new(
        shape: &[usize],
        axes: &[usize],
        spacing: &[S],
        options: SignalExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            shape,
            axes,
            spacing,
            options,
            ExecutionContext::sequential_native(),
        )
    }
    /// Prepares a plan like [`new`](Self::new), planning the FFTs with `cx`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`new`](Self::new), and `ExecutionStopped` if `cx`
    /// is cancelled or past its deadline.
    pub fn build(
        shape: &[usize],
        axes: &[usize],
        spacing: &[S],
        options: SignalExecutionOptions,
        cx: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        crate::util::check_real_slice_finite(spacing)?;
        exact_len(axes.len(), spacing.len())?;
        if spacing.iter().any(|h| *h <= S::ZERO) {
            return Err(invalid(
                scioxide_core::argument!("spacing"),
                "spacing values must be positive",
            ));
        }
        let spectral = SpectralPlan::new(shape, axes, options.execution_policy(), cx, true)?;
        // A common frequency scale cancels in the normalized multiplier. Form it
        // in logarithms to avoid overflow for extreme finite lengths/spacings.
        let mut frequency_scale =
            try_with_capacity(axes.len(), scioxide_core::quantity!("riesz_frequencies"))?;
        for (&axis, &h) in axes.iter().zip(spacing) {
            frequency_scale.push(-scalar_from_usize_rounded::<S>(shape[axis])?.ln() - h.ln());
        }
        let mut output_shape = try_with_capacity(
            checked_add(shape.len(), 1, scioxide_core::quantity!("spectral_scratch"))?,
            scioxide_core::quantity!("riesz_output_shape"),
        )?;
        output_shape.push(axes.len());
        output_shape.extend_from_slice(shape);
        let mut monogenic_shape = copy(&output_shape)?;
        monogenic_shape[0] =
            checked_add(axes.len(), 1, scioxide_core::quantity!("spectral_scratch"))?;
        let complex = checked_add(
            checked_mul(
                checked_add(axes.len(), 3, scioxide_core::quantity!("spectral_scratch"))?,
                spectral.len,
                scioxide_core::quantity!("spectral_scratch"),
            )?,
            spectral.fft_scratch,
            scioxide_core::quantity!("spectral_scratch"),
        )?;
        let mut multipliers = try_filled(
            checked_mul(
                axes.len(),
                spectral.len,
                scioxide_core::quantity!("riesz_multipliers"),
            )?,
            S::ZERO,
            scioxide_core::quantity!("riesz_multipliers"),
        )?;
        let mut packed_multipliers = try_filled(
            checked_mul(
                axes.len(),
                spectral.packed_len,
                scioxide_core::quantity!("riesz_multipliers"),
            )?,
            S::ZERO,
            scioxide_core::quantity!("riesz_multipliers"),
        )?;
        for component in 0..axes.len() {
            cx.check()?;
            for (index, factor) in multipliers
                [component * spectral.len..(component + 1) * spectral.len]
                .iter_mut()
                .enumerate()
            {
                *factor = Self::multiplier(&spectral, &frequency_scale, index, component)?;
            }
            for (index, factor) in packed_multipliers
                [component * spectral.packed_len..(component + 1) * spectral.packed_len]
                .iter_mut()
                .enumerate()
            {
                *factor = multipliers[component * spectral.len + spectral.packed_index(index)];
            }
        }
        Ok(Self {
            requirements: SignalRequirements::new(0, spectral.len, complex),
            spectral,
            spacing: copy(spacing)?,
            multipliers,
            packed_multipliers,
            output_shape,
            monogenic_shape,
        })
    }
    /// Returns the input shape.
    pub fn shape(&self) -> &[usize] {
        self.spectral.shape()
    }
    /// Returns the selected axes, in the order of the components.
    pub fn axes(&self) -> &[usize] {
        self.spectral.axes()
    }
    /// Returns the sample spacing of each selected axis, in the order of the
    /// components.
    pub fn spacing(&self) -> &[S] {
        &self.spacing
    }
    /// Returns the shape of the Riesz transform: the number of selected axes,
    /// followed by the input shape.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
    /// Returns the shape of the monogenic signal: the number of selected axes
    /// plus one, followed by the input shape.
    pub fn monogenic_shape(&self) -> &[usize] {
        &self.monogenic_shape
    }
    /// Returns the workspace that one Riesz or monogenic call needs, for real or
    /// complex input.
    ///
    /// # Errors
    ///
    /// This function does not fail for a prepared plan.
    pub fn workspace_requirements(
        &self,
        _request: (),
    ) -> Result<SignalRequirements, SciOxideError> {
        Ok(self.requirements)
    }
    /// Allocates a workspace for [`workspace_requirements`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the allocator fails.
    pub fn workspace(&self, request: ()) -> Result<SignalWorkspace<S>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(request)?)
    }
    fn multiplier(
        spectral: &SpectralPlan<S>,
        frequency_scale: &[S],
        index: usize,
        component: usize,
    ) -> Result<S, SciOxideError> {
        let axis = spectral.axes[component];
        let n = spectral.shape[axis];
        let k = spectral.coordinate(index, axis);
        if k == 0 || (n.is_multiple_of(2) && k == n / 2) {
            return Ok(S::ZERO);
        }
        let frequency_log = |component: usize| -> Result<S, SciOxideError> {
            let axis = spectral.axes[component];
            let n = spectral.shape[axis];
            let k = spectral.coordinate(index, axis);
            Ok(scalar_from_usize_rounded::<S>(k.min(n - k))?.ln() + frequency_scale[component])
        };
        let mut maximum = S::NEG_INFINITY;
        for j in 0..spectral.axes.len() {
            maximum = maximum.max(frequency_log(j)?);
        }
        let mut norm = S::ZERO;
        for j in 0..spectral.axes.len() {
            norm = norm.hypot((frequency_log(j)? - maximum).exp());
        }
        let magnitude = (frequency_log(component)? - maximum).exp() / norm;
        Ok(if k > n / 2 { -magnitude } else { magnitude })
    }
    fn execute<T: NumericScalar<Real = S>>(
        &self,
        input: NdView<'_, '_, T>,
        mut output: NdViewMut<'_, '_, T>,
        ws: &mut SignalWorkspace<S>,
        monogenic: bool,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let shape = if monogenic {
            self.monogenic_shape()
        } else {
            self.output_shape()
        };
        exact_shape(shape, output.layout().shape().extents())?;
        exact_shape(self.shape(), input.layout().shape().extents())?;
        let buffers = ws.buffers(&self.requirements)?;
        let (samples, scratch) = (buffers.real, buffers.complex);
        context.check()?;
        let n = self.spectral.len;
        let bins = if T::IS_REAL {
            self.spectral.packed_len
        } else {
            n
        };
        let samples = &mut samples[..n];
        let (spectrum, rest) = scratch.split_at_mut(bins);
        let (buffer, rest) = rest.split_at_mut(n);
        let components = self.axes().len() + usize::from(monogenic);
        let (staged, fft) = rest.split_at_mut(checked_mul(
            components,
            n,
            scioxide_core::quantity!("spectral_scratch"),
        )?);
        let maximum = input_scale(input.iter().copied())?;
        let exponent = fft_exponent(maximum, n)?;
        if T::IS_REAL {
            samples
                .iter_mut()
                .zip(input.iter())
                .for_each(|(out, &value)| *out = scale(value.real_component(), -exponent));
            self.spectral.forward.forward_real(
                samples,
                spectrum,
                self.spectral.layouts()?,
                fft,
                context,
            )?;
        } else if let Some((forward, _)) = &self.spectral.complex {
            buffer
                .iter_mut()
                .zip(input.iter())
                .for_each(|(out, &value)| *out = crate::util::complex(scale(value, -exponent)));
            forward.complex(buffer, spectrum, self.spectral.layout()?, fft, context)?;
        }
        let offset = usize::from(monogenic);
        if monogenic {
            for (out, &value) in staged[..n].iter_mut().zip(input.iter()) {
                *out = crate::util::complex(value);
            }
        }
        for component in 0..self.axes().len() {
            context.check()?;
            let factors = if T::IS_REAL {
                &self.packed_multipliers[component * bins..(component + 1) * bins]
            } else {
                &self.multipliers[component * bins..(component + 1) * bins]
            };
            for ((out, value), &factor) in
                buffer[..bins].iter_mut().zip(spectrum.iter()).zip(factors)
            {
                *out = Complex::new(value.im * factor, -value.re * factor);
            }
            let result = &mut staged[(component + offset) * n..(component + offset + 1) * n];
            if T::IS_REAL {
                self.spectral.inverse.inverse_real(
                    &buffer[..bins],
                    samples,
                    self.spectral.layouts()?,
                    fft,
                    context,
                )?;
                restore_fft(samples, exponent)?;
                result
                    .iter_mut()
                    .zip(samples.iter())
                    .for_each(|(out, &value)| *out = Complex::new(value, S::ZERO));
            } else if let Some((_, inverse)) = &self.spectral.complex {
                inverse.complex(buffer, result, self.spectral.layout()?, fft, context)?;
                restore_fft(result, exponent)?;
            }
        }
        context.check()?;
        let mut values = staged.iter();
        output.for_each_mut(|out| {
            if let Some(z) = values.next() {
                *out = T::from_components(z.re, z.im);
            }
        });
        Ok(())
    }
    /// Writes the Riesz transform of `input` to `output`, which has shape
    /// [`output_shape`](Self::output_shape).
    ///
    /// The components of a real input are real.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` does not have the prepared
    ///   shape, or `ws` is too small.
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` if a result overflows.
    /// - `ExecutionStopped` if `cx` is cancelled or past its deadline.
    pub fn riesz_into<T: NumericScalar<Real = S>>(
        &self,
        input: NdView<'_, '_, T>,
        output: NdViewMut<'_, '_, T>,
        ws: &mut SignalWorkspace<S>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.execute(input, output, ws, false, cx)
    }
    /// Writes the monogenic signal of a real `input`, the input followed by its
    /// Riesz components, to `output`, which has shape
    /// [`monogenic_shape`](Self::monogenic_shape).
    ///
    /// # Errors
    ///
    /// Returns the errors of [`riesz_into`](Self::riesz_into).
    pub fn monogenic_into(
        &self,
        input: NdView<'_, '_, S>,
        output: NdViewMut<'_, '_, S>,
        ws: &mut SignalWorkspace<S>,
        cx: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.execute(input, output, ws, true, cx)
    }
    /// Allocate output and workspace, then run sequentially.
    fn allocating<T: NumericScalar<Real = S>>(
        &self,
        input: NdView<'_, '_, T>,
        monogenic: bool,
    ) -> Result<Vec<T>, SciOxideError> {
        let shape = if monogenic {
            self.monogenic_shape()
        } else {
            self.output_shape()
        };
        let mut output = try_filled(
            Shape::new(shape)?.len(),
            T::ZERO,
            scioxide_core::quantity!("riesz_output"),
        )?;
        let strides = strides(shape)?;
        self.execute(
            input,
            NdViewMut::new(&mut output, StridedLayout::new(shape, &strides)?)?,
            &mut self.workspace(())?,
            monogenic,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Returns the Riesz transform of `input` as a row-major array of shape
    /// [`output_shape`](Self::output_shape).
    ///
    /// This allocates the output and a workspace, and runs with
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` does not have the prepared shape.
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` if a result overflows.
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn riesz<T: NumericScalar<Real = S>>(
        &self,
        input: NdView<'_, '_, T>,
    ) -> Result<Vec<T>, SciOxideError> {
        self.allocating(input, false)
    }
    /// Returns the monogenic signal of `input` as a row-major array of shape
    /// [`monogenic_shape`](Self::monogenic_shape).
    ///
    /// This allocates the output and a workspace, and runs with
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`riesz`](Self::riesz).
    pub fn monogenic(&self, input: NdView<'_, '_, S>) -> Result<Vec<S>, SciOxideError> {
        self.allocating(input, true)
    }
}
/// Computes the periodic Riesz transform of a real or complex array along
/// `axes`, with sample spacing `spacing[j]` on axis `axes[j]`.
///
/// The result is a row-major array of shape [`RieszPlan::output_shape`]: one
/// component per selected axis along a new leading axis, each with the input
/// shape. See [`RieszPlan`] for the definition. Along one axis the Riesz
/// transform is the periodic Hilbert transform. This function prepares a plan
/// and runs it once with [`ExecutionContext::sequential_native`].
///
/// # Errors
///
/// Returns the errors of [`RieszPlan::new`] and [`RieszPlan::riesz`].
///
/// # Examples
///
/// ```
/// use scioxide_core::{NdView, StridedLayout};
/// let cosine = [1.0_f64, 0.0, -1.0, 0.0];
/// let input = NdView::new(&cosine, StridedLayout::new(&[4], &[1])?)?;
/// let riesz = scioxide_signal::riesz_nd(input, &[0], &[1.0])?;
/// // In one dimension this is the Hilbert transform: cos(πn/2) → sin(πn/2).
/// assert!(riesz.iter().zip([0.0, 1.0, 0.0, -1.0]).all(|(x, e)| (x - e).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn riesz_nd<T: NumericScalar>(
    input: NdView<'_, '_, T>,
    axes: &[usize],
    spacing: &[T::Real],
) -> Result<Vec<T>, SciOxideError> {
    RieszPlan::new(
        input.layout().shape().extents(),
        axes,
        spacing,
        SignalExecutionOptions::new(),
    )?
    .riesz(input)
}
/// Computes the monogenic signal of a real array along `axes`, with sample
/// spacing `spacing[j]` on axis `axes[j]`.
///
/// The monogenic signal is the input followed by its Riesz components
/// (see [`RieszPlan`]), as a row-major array of shape
/// [`RieszPlan::monogenic_shape`]. In one dimension it holds the real and
/// imaginary parts of the analytic signal. This function prepares a plan and
/// runs it once with [`ExecutionContext::sequential_native`].
///
/// # Errors
///
/// Returns the errors of [`RieszPlan::new`] and [`RieszPlan::monogenic`].
///
/// # Examples
///
/// ```
/// use scioxide_core::{NdView, StridedLayout};
/// let cosine = [1.0_f64, 0.0, -1.0, 0.0];
/// let input = NdView::new(&cosine, StridedLayout::new(&[4], &[1])?)?;
/// let monogenic = scioxide_signal::monogenic_nd(input, &[0], &[1.0])?;
/// let expected = [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0];
/// assert!(monogenic.iter().zip(expected).all(|(x, e)| (x - e).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. Felsberg and G. Sommer, "The monogenic signal", *IEEE Trans. Signal
/// Process.* 49(12), 3136–3144, 2001.
pub fn monogenic_nd<S: RealScalar>(
    input: NdView<'_, '_, S>,
    axes: &[usize],
    spacing: &[S],
) -> Result<Vec<S>, SciOxideError> {
    RieszPlan::new(
        input.layout().shape().extents(),
        axes,
        spacing,
        SignalExecutionOptions::new(),
    )?
    .monogenic(input)
}
