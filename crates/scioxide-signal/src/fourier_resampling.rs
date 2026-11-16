//! Fourier resampling of periodic records along one axis.
use crate::fourier::Transform;
use crate::util::{complex, copy, fft_exponent, input_scale, restore_fft, strides};
use crate::util::{empty, exact_shape, invalid, scalar_from_usize_rounded};
use crate::{SignalExecutionOptions, SignalRequirements, SignalWorkspace};
use alloc::vec::Vec;
use core::marker::PhantomData;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::standard::scale;
use scioxide_core::backend::{checked_add, try_filled};
use scioxide_core::{
    Complex, ExecutionContext, NdView, NdViewMut, NumericScalar, RealScalar, SciOxideError, Shape,
    StridedLayout,
};
use scioxide_fft::FftDirection;
/// Plan for Fourier resampling of a periodic record from `n` to `m` samples
/// along one axis of an N-dimensional array.
///
/// The plan treats each lane along the axis as one period of a band-limited
/// periodic signal. It computes the DFT `X` of the `n` input samples, keeps the
/// frequencies `|k| < min(n, m)/2`, and evaluates the trigonometric polynomial
/// at the `m` points `t_j = j n / m`, in units of the input sample spacing:
///
/// ```text
/// y[j] = (1/n) Σₖ Y[k] exp(2πi j k / m),   j = 0, …, m − 1,
/// ```
///
/// where `Y` holds the kept bins of `X`. When `min(n, m)` is even, the Nyquist
/// bin of the shorter length gets special treatment. For `n < m`, `X[n/2]` is
/// split equally between `Y[n/2]` and `Y[m − n/2]`. For `m < n`,
/// `Y[m/2] = X[m/2] + X[n − m/2]`. A real input therefore gives a real output.
/// Upsampling is exact trigonometric interpolation, and downsampling is ideal
/// low-pass filtering of the periodic signal. The other axes are independent
/// batches. One call costs `O(n log n + m log m)` per lane.
///
/// ```
/// use scioxide_core::{ExecutionContext, NdView, NdViewMut, StridedLayout};
/// use scioxide_signal::{FourierResamplePlan, SignalExecutionOptions};
/// let plan = FourierResamplePlan::<f64>::new(&[4], 0, 8, SignalExecutionOptions::new())?;
/// let mut workspace = plan.workspace(())?;
/// let mut out = [0.0; 8];
/// plan.resample_into(
///     NdView::new(&[1.0; 4], StridedLayout::new(&[4], &[1])?)?,
///     NdViewMut::new(&mut out, StridedLayout::new(&[8], &[1])?)?,
///     &mut workspace,
///     ExecutionContext::sequential_native(),
/// )?;
/// // A constant record interpolates to the same constant.
/// assert!(out.iter().all(|x| (x - 1.0).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug)]
pub struct FourierResamplePlan<T: NumericScalar> {
    shape: Vec<usize>,
    output_shape: Vec<usize>,
    input_strides: Vec<usize>,
    output_strides: Vec<usize>,
    axis: usize,
    forward: Transform<T::Real>,
    inverse: Transform<T::Real>,
    input_spectrum_shape: Vec<usize>,
    output_spectrum_shape: Vec<usize>,
    input_spectrum_strides: Vec<usize>,
    output_spectrum_strides: Vec<usize>,
    input_spectrum_len: usize,
    output_spectrum_len: usize,
    real_len: usize,
    inner: usize,
    ratio: T::Real,
    input_len: usize,
    output_len: usize,
    complex_len: usize,
    scalar: PhantomData<T>,
}
impl<T: NumericScalar> FourierResamplePlan<T> {
    /// Prepares a plan that resamples arrays of shape `shape` to
    /// `output_axis_len` samples along `axis`, using
    /// [`ExecutionContext::sequential_native`].
    ///
    /// The execution policy of `options` applies to the FFTs.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `axis` is not below the rank of `shape`, or
    ///   `shape[axis]` or `output_axis_len` is zero.
    /// - `SizeOverflow` or `AllocationFailure` if the plan storage cannot be sized
    ///   or allocated.
    pub fn new(
        shape: &[usize],
        axis: usize,
        output_axis_len: usize,
        options: SignalExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            shape,
            axis,
            output_axis_len,
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
        shape: &[usize],
        axis: usize,
        output_axis_len: usize,
        options: SignalExecutionOptions,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        let policy = options.execution_policy();
        if axis >= shape.len() {
            return Err(invalid(
                scioxide_core::argument!("axis"),
                "axis must be below the shape rank",
            ));
        }
        if shape[axis] == 0 {
            return Err(empty(
                scioxide_core::argument!("shape"),
                "the selected axis must be nonempty",
            ));
        }
        if output_axis_len == 0 {
            return Err(empty(
                scioxide_core::argument!("output_axis_len"),
                "output axis length must be positive",
            ));
        }
        let input_len = Shape::new(shape)?.len();
        let mut output_shape = copy(shape)?;
        output_shape[axis] = output_axis_len;
        let output_len = Shape::new(&output_shape)?.len();
        let forward = Transform::new(
            shape,
            &[axis],
            FftDirection::Forward,
            T::IS_REAL,
            policy,
            context,
        )?;
        let inverse = Transform::new(
            &output_shape,
            &[axis],
            FftDirection::Inverse,
            T::IS_REAL,
            policy,
            context,
        )?;
        let mut input_spectrum_shape = copy(shape)?;
        let mut output_spectrum_shape = copy(&output_shape)?;
        if T::IS_REAL {
            input_spectrum_shape[axis] = shape[axis] / 2 + 1;
            output_spectrum_shape[axis] = output_axis_len / 2 + 1;
        }
        let input_spectrum_len = Shape::new(&input_spectrum_shape)?.len();
        let output_spectrum_len = Shape::new(&output_spectrum_shape)?.len();
        let buffers = checked_add(
            input_spectrum_len,
            output_spectrum_len,
            scioxide_core::quantity!("resampling_buffers"),
        )?;
        let time_buffers = checked_add(
            input_len,
            output_len,
            scioxide_core::quantity!("resampling_buffers"),
        )?;
        let real_len = if T::IS_REAL { time_buffers } else { 0 };
        let complex_len = checked_add(
            checked_add(
                buffers,
                if T::IS_REAL { 0 } else { time_buffers },
                scioxide_core::quantity!("resampling_buffers"),
            )?,
            forward.complex_work()?.max(inverse.complex_work()?),
            scioxide_core::quantity!("resampling_scratch"),
        )?;
        let inner = if input_len == 0 {
            0
        } else {
            Shape::new(&shape[axis + 1..])?.len()
        };
        let ratio = scalar_from_usize_rounded::<T::Real>(output_axis_len)?
            / scalar_from_usize_rounded::<T::Real>(shape[axis])?;
        Ok(Self {
            shape: copy(shape)?,
            input_strides: strides(shape)?,
            output_strides: strides(&output_shape)?,
            output_shape,
            axis,
            input_spectrum_strides: strides(&input_spectrum_shape)?,
            output_spectrum_strides: strides(&output_spectrum_shape)?,
            input_spectrum_shape,
            output_spectrum_shape,
            input_spectrum_len,
            output_spectrum_len,
            real_len,
            inner,
            ratio,
            forward,
            inverse,
            input_len,
            output_len,
            complex_len,
            scalar: PhantomData,
        })
    }
    /// Returns the output shape: the input shape with `output_axis_len` on the
    /// resampled axis.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
    /// Returns the workspace that one call to
    /// [`resample_into`](Self::resample_into) needs.
    ///
    /// # Errors
    ///
    /// This function does not fail for a prepared plan.
    pub fn workspace_requirements(&self, _: ()) -> Result<SignalRequirements, SciOxideError> {
        Ok(SignalRequirements::new(0, self.real_len, self.complex_len))
    }
    /// Allocates a workspace for [`workspace_requirements`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the allocator fails.
    pub fn workspace(&self, request: ()) -> Result<SignalWorkspace<T>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(request)?)
    }
    /// Writes the resampled `input` to `output`.
    ///
    /// `output` has shape [`output_shape`](Self::output_shape).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` does not have the prepared
    ///   shape, or `workspace` is too small.
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` if a result overflows.
    /// - `ExecutionStopped` if `context` is cancelled or past its deadline.
    pub fn resample_into(
        &self,
        input: NdView<'_, '_, T>,
        output: NdViewMut<'_, '_, T>,
        workspace: &mut SignalWorkspace<T>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        exact_shape(&self.shape, input.layout().shape().extents())?;
        exact_shape(&self.output_shape, output.layout().shape().extents())?;
        let buffers = workspace.buffers(&self.workspace_requirements(())?)?;
        let (samples, scratch) = (buffers.real, buffers.complex);
        if let Some(values) = crate::util::vector_storage(&input) {
            self.resample_values(|| values.iter(), output, samples, scratch, context)
        } else {
            self.resample_values(|| input.iter(), output, samples, scratch, context)
        }
    }
    fn resample_values<'i, I: ExactSizeIterator<Item = &'i T>>(
        &self,
        values: impl Fn() -> I,
        mut output: NdViewMut<'_, '_, T>,
        samples: &mut [T::Real],
        scratch: &mut [Complex<T::Real>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.check()?;
        let maximum = input_scale(values().copied())?;
        let exponent = fft_exponent(maximum, self.input_len.max(self.output_len))?;
        let il = StridedLayout::new(&self.shape, &self.input_strides)?;
        let ol = StridedLayout::new(&self.output_shape, &self.output_strides)?;
        let isl = StridedLayout::new(&self.input_spectrum_shape, &self.input_spectrum_strides)?;
        let osl = StridedLayout::new(&self.output_spectrum_shape, &self.output_spectrum_strides)?;
        let (spectrum, rest) = scratch.split_at_mut(self.input_spectrum_len);
        let (mapped, rest) = rest.split_at_mut(self.output_spectrum_len);
        if T::IS_REAL {
            let (time, result) = samples.split_at_mut(self.input_len);
            time.iter_mut()
                .zip(values())
                .for_each(|(out, &value)| *out = scale(value.real_component(), -exponent));
            self.forward
                .forward_real(time, spectrum, (il, isl), rest, context)?;
            self.map_bins(spectrum, mapped);
            self.inverse
                .inverse_real(mapped, result, (ol, osl), rest, context)?;
            restore_fft(result, exponent)?;
            context.check()?;
            let mut values = result.iter();
            output.for_each_mut(|out| {
                if let Some(&value) = values.next() {
                    *out = T::from_real(value);
                }
            });
        } else {
            let (time, rest) = rest.split_at_mut(self.input_len);
            let (result, fft) = rest.split_at_mut(self.output_len);
            time.iter_mut()
                .zip(values())
                .for_each(|(out, &value)| *out = complex(scale(value, -exponent)));
            self.forward.complex(time, spectrum, il, fft, context)?;
            self.map_bins(spectrum, mapped);
            self.inverse.complex(mapped, result, ol, fft, context)?;
            restore_fft(result, exponent)?;
            context.check()?;
            let mut values = result.iter();
            output.for_each_mut(|out| {
                if let Some(value) = values.next() {
                    *out = T::from_components(value.re, value.im);
                }
            });
        }
        Ok(())
    }
    fn map_bins(&self, input: &[Complex<T::Real>], output: &mut [Complex<T::Real>]) {
        output.fill(Complex::new(T::Real::ZERO, T::Real::ZERO));
        if self.input_len == 0 {
            return;
        }
        let n = self.shape[self.axis];
        let m = self.output_shape[self.axis];
        let inner = self.inner;
        let ni = self.input_spectrum_shape[self.axis] * inner;
        let mi = self.output_spectrum_shape[self.axis] * inner;
        let common = n.min(m);
        let positive = common.div_ceil(2) * inner;
        for (source, target) in input.chunks_exact(ni).zip(output.chunks_exact_mut(mi)) {
            for (out, &value) in target[..positive].iter_mut().zip(&source[..positive]) {
                *out = value * self.ratio;
            }
            if !T::IS_REAL {
                let negative = (common - 1) / 2 * inner;
                for (out, &value) in target[mi - negative..]
                    .iter_mut()
                    .zip(&source[ni - negative..])
                {
                    *out = value * self.ratio;
                }
            }
            if common.is_multiple_of(2) {
                let start = common / 2 * inner;
                for j in 0..inner {
                    let value = source[start + j] * self.ratio;
                    if n == m {
                        target[start + j] = value;
                    } else if n < m {
                        let value = value * T::Real::HALF;
                        target[start + j] = value;
                        if !T::IS_REAL {
                            target[(m - common / 2) * inner + j] = value;
                        }
                    } else {
                        target[start + j] = if T::IS_REAL {
                            Complex::new(value.re * T::Real::TWO, T::Real::ZERO)
                        } else {
                            value + source[(n - common / 2) * inner + j] * self.ratio
                        };
                    }
                }
            }
        }
    }
}
/// Resamples one period of a band-limited periodic signal from `n` to
/// `output_len` samples by zero padding or truncating its DFT.
///
/// The `n` samples of `input` are one period. The result is their
/// trigonometric interpolant evaluated at the `m = output_len` points
/// `t_j = j n / m`, in units of the input sample spacing, with the Nyquist-bin
/// rule of [`FourierResamplePlan`]. Upsampling is exact trigonometric
/// interpolation, downsampling is ideal low-pass filtering of the periodic
/// signal, and a real input gives a real output. The cost is
/// `O(n log n + m log m)`, and the error is normwise as described in
/// [Accuracy](crate#accuracy). [`FourierResamplePlan`] resamples along one axis
/// of an array and reuses its FFT plans across calls. This function runs with
/// [`ExecutionContext::sequential_native`].
///
/// # Errors
///
/// - `InvalidInput` (`Empty`) if `input` is empty or `output_len` is zero.
/// - `NonFinite` if `input` contains NaN or infinity.
/// - `NumericalInstability` if a result overflows.
/// - `SizeOverflow` or `AllocationFailure` if the plan, the workspace or the
///   output cannot be sized or allocated.
///
/// # Examples
///
/// ```
/// use core::f64::consts::TAU;
/// // One period of a cosine interpolates to the same cosine on a finer grid.
/// let x: Vec<f64> = (0..8).map(|k| (TAU * k as f64 / 8.0).cos()).collect();
/// let y = scioxide_signal::resample_periodic(&x, 12)?;
/// for (j, value) in y.iter().enumerate() {
///     assert!((value - (TAU * j as f64 / 12.0).cos()).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn resample_periodic<T: NumericScalar>(
    input: &[T],
    output_len: usize,
) -> Result<Vec<T>, SciOxideError> {
    if input.is_empty() {
        return Err(empty(
            scioxide_core::argument!("input"),
            "the selected axis must be nonempty",
        ));
    }
    if output_len == 0 {
        return Err(empty(
            scioxide_core::argument!("output_len"),
            "output axis length must be positive",
        ));
    }
    let plan = FourierResamplePlan::<T>::new(
        &[input.len()],
        0,
        output_len,
        SignalExecutionOptions::new(),
    )?;
    let mut output = try_filled(
        output_len,
        T::ZERO,
        scioxide_core::quantity!("resampling_output"),
    )?;
    plan.resample_into(
        NdView::new(input, StridedLayout::new(&[input.len()], &[1])?)?,
        NdViewMut::new(&mut output, StridedLayout::new(&[output_len], &[1])?)?,
        &mut plan.workspace(())?,
        ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}
