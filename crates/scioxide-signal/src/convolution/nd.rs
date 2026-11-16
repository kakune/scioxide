//! N-dimensional convolution and correlation along selected axes.
use super::runs;
use crate::fourier::Transform;
use crate::util::{
    axis_mismatch, boundary_index, copy, empty, exact_shape, fft_exponent, inconsistent,
    input_scale, invalid, restore_fft, strides, unsupported,
};
use crate::{SignalRequirements, SignalWorkspace};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::RealBackend;
use scioxide_core::backend::standard::scale;
use scioxide_core::backend::{
    checked_add, checked_mul, checked_next_power_of_two, try_filled, try_with_capacity,
};
use scioxide_core::{
    Complex, ExecutionContext, ExecutionPolicy, NdView, NdViewMut, NumericScalar, SciOxideError,
    Shape, StridedLayout,
};
use scioxide_fft::FftDirection;

/// Algorithm that a [`ConvolutionNdPlan`] uses.
///
/// Both methods meet the error bounds in [Accuracy](crate#accuracy). They do not
/// round identically, so their results can differ in the last bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvolutionMethod {
    /// Sums the products of overlapping samples directly, at a cost proportional
    /// to [`ConvolutionSelection::direct_products`].
    Direct,
    /// Multiplies zero-padded spectra, with FFT lengths rounded up to powers of
    /// two, at a cost proportional to [`ConvolutionSelection::fft_work`].
    Fft,
    /// Chooses `Direct` when its product count does not exceed the FFT work
    /// estimate, and `Fft` otherwise.
    Automatic,
}
/// Part of the full convolution that a plan returns, chosen independently on
/// each selected axis.
///
/// On an axis with signal length `n` and kernel length `k`:
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvolutionExtent {
    /// Every position where signal and kernel overlap: `n + k − 1` outputs.
    Full,
    /// `n` outputs, starting at index `⌊(k − 1)/2⌋` of the full result.
    Same,
    /// Only positions where the kernel lies entirely inside the signal:
    /// `n − k + 1` outputs, starting at index `k − 1` of the full result.
    Valid,
    /// The circular product of length `n`. Index `j` is the output at `j`,
    /// or the lag `j` modulo `n` for correlation. Requires
    /// [`SignalBoundary::Periodic`].
    Circular,
}
/// Extension of a finite signal beyond its ends on the selected axes.
///
/// For a record `x[0], …, x[n−1]`:
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalBoundary {
    /// Samples outside the record are zero.
    Zero,
    /// The record repeats with period `n`: `x[−1] = x[n−1]`.
    Periodic,
    /// The record is mirrored about its end samples without repeating them:
    /// `x[−1] = x[1]` and `x[n] = x[n−2]`.
    Reflect,
}
/// Product that a [`ConvolutionNdPlan`] computes.
///
/// See [Conventions](crate#conventions) for the definitions and the lag order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalProduct {
    /// Convolution, `y[n] = Σₖ a[n − k] b[k]`.
    Convolution,
    /// Cross-correlation, `y[lag] = Σₖ a[k] conj(b[k − lag])`, in ascending lag
    /// order.
    Correlation,
}
/// Options of an N-dimensional convolution or correlation.
///
/// The default is [`ConvolutionMethod::Automatic`], [`ConvolutionExtent::Full`],
/// [`SignalBoundary::Zero`], [`SignalProduct::Convolution`] and
/// [`ExecutionPolicy::Sequential`]. The setters accept any value.
/// [`ConvolutionNdPlan::new`] checks the combination.
///
/// ```
/// use scioxide_signal::{ConvolutionExtent, ConvolutionOptions, SignalBoundary};
/// let options = ConvolutionOptions::new()
///     .with_extent(ConvolutionExtent::Circular)
///     .with_boundary(SignalBoundary::Periodic);
/// assert_eq!(options.extent(), ConvolutionExtent::Circular);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub struct ConvolutionOptions {
    method: ConvolutionMethod,
    extent: ConvolutionExtent,
    boundary: SignalBoundary,
    product: SignalProduct,
    policy: ExecutionPolicy,
}
impl Default for ConvolutionOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl ConvolutionOptions {
    /// Creates the default options.
    pub const fn new() -> Self {
        Self {
            method: ConvolutionMethod::Automatic,
            extent: ConvolutionExtent::Full,
            boundary: SignalBoundary::Zero,
            product: SignalProduct::Convolution,
            policy: ExecutionPolicy::Sequential,
        }
    }
    /// Sets the method.
    pub const fn with_method(mut self, method: ConvolutionMethod) -> Self {
        self.method = method;
        self
    }
    /// Sets the output extent.
    pub const fn with_extent(mut self, extent: ConvolutionExtent) -> Self {
        self.extent = extent;
        self
    }
    /// Sets the boundary extension.
    pub const fn with_boundary(mut self, boundary: SignalBoundary) -> Self {
        self.boundary = boundary;
        self
    }
    /// Sets the product: convolution or correlation.
    pub const fn with_product(mut self, product: SignalProduct) -> Self {
        self.product = product;
        self
    }
    /// Sets the execution policy of the FFTs.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Returns the method.
    pub const fn method(self) -> ConvolutionMethod {
        self.method
    }
    /// Returns the output extent.
    pub const fn extent(self) -> ConvolutionExtent {
        self.extent
    }
    /// Returns the boundary extension.
    pub const fn boundary(self) -> SignalBoundary {
        self.boundary
    }
    /// Returns the product.
    pub const fn product(self) -> SignalProduct {
        self.product
    }
    /// Returns the execution policy of the FFTs.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.policy
    }
}
/// Method a [`ConvolutionNdPlan`] selected, with the cost estimates behind the
/// choice.
///
/// The estimates are operation counts, not measured run times.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvolutionSelection {
    method: ConvolutionMethod,
    direct_products: usize,
    fft_work: usize,
}
impl ConvolutionSelection {
    /// Returns the selected method, `Direct` or `Fft`.
    pub const fn method(self) -> ConvolutionMethod {
        self.method
    }
    /// Returns an upper bound on the number of products of the direct method,
    /// over all batches.
    ///
    /// It is exact for [`ConvolutionExtent::Full`] with [`SignalBoundary::Zero`].
    /// For other extents with a zero boundary it counts each output on each
    /// selected axis with `min(n, k)` products.
    pub const fn direct_products(self) -> usize {
        self.direct_products
    }
    /// Returns the FFT work estimate `P · Σ log₂ pᵢ`, where `pᵢ` is the padded
    /// length on selected axis `i` and `P` is the number of elements of the
    /// padded array.
    ///
    /// [`ConvolutionMethod::Automatic`] selects the direct method when
    /// [`direct_products`](Self::direct_products) does not exceed this value.
    pub const fn fft_work(self) -> usize {
        self.fft_work
    }
}
/// Kernel that a convolution call uses, which determines the workspace it needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvolutionKernel {
    /// The kernel stored in the plan, used by [`ConvolutionNdPlan::convolve_into`].
    Prepared,
    /// A kernel passed to each call of [`ConvolutionNdPlan::convolve_pair_into`],
    /// with the shape of the stored kernel.
    Supplied,
}

/// Plan for the convolution or correlation of N-dimensional arrays of a fixed
/// shape with a fixed kernel, along selected axes.
///
/// The plan convolves along the selected axes and treats every other axis as a
/// batch axis. On a batch axis the kernel extent is either one, so the same
/// kernel applies to every batch, or equal to the signal extent, so each batch
/// has its own kernel. [`ConvolutionOptions`] sets the product, the output
/// extent, the boundary extension and the method.
///
/// With the direct method and a full zero-extended product, each batch costs
/// `N · K` multiplications, where `N` and `K` are the numbers of signal and
/// kernel elements on the selected axes. The FFT method costs `O(P log P)` for a
/// padded array of `P` elements. The plan prepares the kernel spectrum once.
/// [`selection`](Self::selection) reports the choice. Accuracy is described
/// in [Accuracy](crate#accuracy), and workspace reuse in [Reusing plans and
/// workspaces](crate#reusing-plans-and-workspaces).
///
/// ```
/// use scioxide_core::{NdView, StridedLayout};
/// use scioxide_signal::{ConvolutionNdPlan, ConvolutionOptions};
/// // Smooth a 1-D record with a two-point average: full length 3 + 2 − 1.
/// let kernel = [0.5_f64, 0.5];
/// let plan = ConvolutionNdPlan::new(
///     &[3],
///     NdView::new(&kernel, StridedLayout::new(&[2], &[1])?)?,
///     &[0],
///     ConvolutionOptions::default(),
/// )?;
/// let input = [2.0, 4.0, 6.0];
/// let output = plan.convolve(NdView::new(&input, StridedLayout::new(&[3], &[1])?)?)?;
/// let expected = [1.0, 3.0, 5.0, 3.0];
/// assert!(output.iter().zip(expected).all(|(x, e)| (x - e).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct ConvolutionNdPlan<T: NumericScalar> {
    shape: Vec<usize>,
    kernel_shape: Vec<usize>,
    output_shape: Vec<usize>,
    padded: Vec<usize>,
    input_strides: Vec<usize>,
    kernel_strides: Vec<usize>,
    output_strides: Vec<usize>,
    padded_strides: Vec<usize>,
    spectrum_shape: Vec<usize>,
    spectrum_strides: Vec<usize>,
    spectrum_len: usize,
    selected: Vec<bool>,
    crop: Vec<usize>,
    options: ConvolutionOptions,
    kernel: Vec<T>,
    kernel_scale: i32,
    input_runs: Vec<runs::CopyRun>,
    kernel_runs: Vec<runs::CopyRun>,
    output_runs: Vec<runs::CopyRun>,
    spectrum: Vec<Complex<T::Real>>,
    forward: Option<Transform<T::Real>>,
    inverse: Option<Transform<T::Real>>,
    selection: ConvolutionSelection,
    input_len: usize,
    output_len: usize,
    padded_len: usize,
    fft_len: usize,
}
struct KernelRef<'a, T: NumericScalar> {
    coefficients: &'a [T],
    scale: i32,
    spectrum: &'a [Complex<T::Real>],
}
impl<T: NumericScalar> ConvolutionNdPlan<T> {
    /// Prepares a plan for signals of shape `shape`, convolved with `kernel`
    /// along `axes`, using [`ExecutionContext::sequential_native`].
    ///
    /// `kernel` has the same rank as `shape`. When the FFT method is selected,
    /// this computes the kernel spectrum.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `shape` or `axes` is empty, an axis is repeated or not
    ///   below the rank, or a selected signal or kernel extent is zero.
    /// - `DimensionMismatch` if the kernel rank differs from the signal rank, or a
    ///   kernel extent on a batch axis is neither one nor the signal extent.
    /// - `InvalidInput` if the extent is `Circular` and the boundary is not
    ///   `Periodic`, or the extent is `Valid` and a selected signal extent is
    ///   shorter than the kernel extent.
    /// - `NonFinite` if a kernel coefficient is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the plan storage cannot be sized
    ///   or allocated.
    pub fn new(
        shape: &[usize],
        kernel: NdView<'_, '_, T>,
        axes: &[usize],
        options: ConvolutionOptions,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            shape,
            kernel,
            axes,
            options,
            ExecutionContext::sequential_native(),
        )
    }
    /// Prepares a plan like [`new`](Self::new), planning the FFTs and
    /// transforming the kernel with `context`.
    ///
    /// The plan does not keep `context`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`new`](Self::new), and `ExecutionStopped` if
    /// `context` is cancelled or past its deadline.
    pub fn build(
        shape: &[usize],
        kernel: NdView<'_, '_, T>,
        axes: &[usize],
        options: ConvolutionOptions,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        context.check()?;
        let ks = kernel.layout().shape().extents();
        if shape.is_empty() {
            return Err(empty(
                scioxide_core::argument!("shape"),
                "signal rank must be positive",
            ));
        }
        if ks.len() != shape.len() {
            return Err(SciOxideError::dimension_mismatch(shape.len(), ks.len()));
        }
        if axes.is_empty() {
            return Err(empty(
                scioxide_core::argument!("axes"),
                "at least one axis must be selected",
            ));
        }
        let input_len = Shape::new(shape)?.len();
        let mut selected = try_filled(
            shape.len(),
            false,
            scioxide_core::quantity!("convolution_axes"),
        )?;
        for &axis in axes {
            if axis >= shape.len() || selected[axis] {
                return Err(invalid(
                    scioxide_core::argument!("axes"),
                    "axes must be distinct and below the signal rank",
                ));
            }
            if shape[axis] == 0 {
                return Err(empty(
                    scioxide_core::argument!("shape"),
                    "convolved signal extents must be nonzero",
                ));
            }
            if ks[axis] == 0 {
                return Err(empty(
                    scioxide_core::argument!("kernel"),
                    "convolved kernel extents must be nonzero",
                ));
            }
            selected[axis] = true;
        }
        let mut output_shape = copy(shape)?;
        let mut padded = copy(shape)?;
        let mut crop = try_filled(shape.len(), 0, scioxide_core::quantity!("convolution_crop"))?;
        let mut logs = 0;
        for axis in 0..shape.len() {
            if selected[axis] {
                let full = checked_add(
                    shape[axis],
                    ks[axis] - 1,
                    scioxide_core::quantity!("convolution_extent"),
                )?;
                match options.extent {
                    ConvolutionExtent::Full => output_shape[axis] = full,
                    ConvolutionExtent::Circular => {
                        if options.boundary != SignalBoundary::Periodic {
                            return Err(unsupported(
                                scioxide_core::argument!("options.boundary"),
                                "circular extent requires periodic boundary",
                            ));
                        }
                        if options.product == SignalProduct::Correlation {
                            crop[axis] = ks[axis] - 1;
                        }
                    }
                    ConvolutionExtent::Same => crop[axis] = (ks[axis] - 1) / 2,
                    ConvolutionExtent::Valid => {
                        if shape[axis] < ks[axis] {
                            return Err(inconsistent(
                                scioxide_core::argument!("options.extent"),
                                "valid extent needs signal extents at least the kernel extents",
                            ));
                        }
                        output_shape[axis] = shape[axis] - ks[axis] + 1;
                        crop[axis] = ks[axis] - 1;
                    }
                }
                let extended = if options.boundary == SignalBoundary::Zero {
                    full
                } else {
                    checked_add(
                        full,
                        checked_mul(2, ks[axis] - 1, scioxide_core::quantity!("boundary_halo"))?,
                        scioxide_core::quantity!("boundary_padding"),
                    )?
                };
                padded[axis] = checked_next_power_of_two(
                    extended,
                    scioxide_core::quantity!("convolution_fft_extent"),
                )?;
                logs += padded[axis].ilog2() as usize;
            } else if ks[axis] != 1 && ks[axis] != shape[axis] {
                // Batch extents broadcast from one; report the signal extent.
                return Err(axis_mismatch(ks, axis, shape[axis], ks));
            }
        }
        let output_len = Shape::new(&output_shape)?.len();
        let padded_len = Shape::new(&padded)?.len();
        let direct_products = (0..shape.len()).try_fold(1, |work, axis| {
            let products = if !selected[axis] {
                output_shape[axis]
            } else if options.boundary == SignalBoundary::Zero {
                if options.extent == ConvolutionExtent::Full {
                    checked_mul(
                        shape[axis],
                        ks[axis],
                        scioxide_core::quantity!("direct_work"),
                    )?
                } else {
                    checked_mul(
                        output_shape[axis],
                        shape[axis].min(ks[axis]),
                        scioxide_core::quantity!("direct_work"),
                    )?
                }
            } else {
                checked_mul(
                    output_shape[axis],
                    ks[axis],
                    scioxide_core::quantity!("direct_work"),
                )?
            };
            checked_mul(work, products, scioxide_core::quantity!("direct_work"))
        })?;
        let fft_work = checked_mul(padded_len, logs, scioxide_core::quantity!("fft_work"))?;
        let method = match options.method {
            ConvolutionMethod::Automatic => {
                if direct_products <= fft_work {
                    ConvolutionMethod::Direct
                } else {
                    ConvolutionMethod::Fft
                }
            }
            m => m,
        };
        let build = |direction| {
            Transform::new(
                &padded,
                axes,
                direction,
                T::IS_REAL,
                options.policy,
                context,
            )
        };
        let mut spectrum_shape = copy(&padded)?;
        if T::IS_REAL {
            let axis = axes[axes.len() - 1];
            spectrum_shape[axis] = padded[axis] / 2 + 1;
        }
        let spectrum_len = Shape::new(&spectrum_shape)?.len();
        let (forward, inverse) = if method == ConvolutionMethod::Fft {
            (
                Some(build(FftDirection::Forward)?),
                Some(build(FftDirection::Inverse)?),
            )
        } else {
            (None, None)
        };
        let fft_len = forward
            .as_ref()
            .map(|plan| plan.complex_work())
            .transpose()?
            .unwrap_or(0)
            .max(
                inverse
                    .as_ref()
                    .map(|plan| plan.complex_work())
                    .transpose()?
                    .unwrap_or(0),
            );
        let mut coefficients = try_with_capacity(
            kernel.layout().shape().len(),
            scioxide_core::quantity!("kernel"),
        )?;
        coefficients.extend(kernel.iter().copied());
        let kernel_scale = fft_exponent(input_scale(coefficients.iter().copied())?, padded_len)?;
        let mut plan = Self {
            spectrum_strides: strides(&spectrum_shape)?,
            spectrum_shape,
            spectrum_len,
            shape: copy(shape)?,
            kernel_shape: copy(ks)?,
            input_strides: strides(shape)?,
            kernel_strides: strides(ks)?,
            output_strides: strides(&output_shape)?,
            padded_strides: strides(&padded)?,
            output_shape,
            padded,
            selected,
            crop,
            options,
            kernel: coefficients,
            kernel_scale,
            input_runs: Vec::new(),
            kernel_runs: Vec::new(),
            output_runs: Vec::new(),
            spectrum: Vec::new(),
            forward,
            inverse,
            selection: ConvolutionSelection {
                method,
                direct_products,
                fft_work,
            },
            input_len,
            output_len,
            padded_len,
            fft_len,
        };
        if plan.forward.is_some() {
            plan.input_runs = runs::prepare(padded_len, |index| {
                let mut source = 0;
                for axis in 0..plan.shape.len() {
                    let c = index / plan.padded_strides[axis] % plan.padded[axis];
                    let c = if plan.selected[axis] {
                        let halo = if plan.options.boundary == SignalBoundary::Zero {
                            0
                        } else {
                            plan.kernel_shape[axis] - 1
                        };
                        if c >= plan.shape[axis] + 2 * halo {
                            return None;
                        }
                        boundary_index(
                            c as i128 - halo as i128,
                            plan.shape[axis],
                            plan.options.boundary,
                        )?
                    } else {
                        c
                    };
                    source += c * plan.input_strides[axis];
                }
                Some(source)
            })?;
            plan.kernel_runs = runs::prepare(padded_len, |index| {
                let mut source = 0;
                for (axis, &extent) in plan.kernel_shape.iter().enumerate() {
                    let c = index / plan.padded_strides[axis] % plan.padded[axis];
                    let c = if plan.selected[axis] {
                        if c >= extent {
                            return None;
                        }
                        if plan.options.product == SignalProduct::Correlation {
                            extent - 1 - c
                        } else {
                            c
                        }
                    } else if extent == 1 {
                        0
                    } else {
                        c
                    };
                    source += c * plan.kernel_strides[axis];
                }
                Some(source)
            })?;
            plan.output_runs = runs::prepare(output_len, |index| {
                Some(
                    (0..plan.shape.len())
                        .map(|axis| {
                            let halo = if plan.selected[axis]
                                && plan.options.boundary != SignalBoundary::Zero
                            {
                                plan.kernel_shape[axis] - 1
                            } else {
                                0
                            };
                            (index / plan.output_strides[axis] % plan.output_shape[axis]
                                + plan.crop[axis]
                                + halo)
                                * plan.padded_strides[axis]
                        })
                        .sum(),
                )
            })?;
            let mut time = try_filled(
                padded_len,
                T::ZERO,
                scioxide_core::quantity!("kernel_fft_input"),
            )?;
            let mut spectrum = try_filled(
                spectrum_len,
                Complex::new(T::Real::ZERO, T::Real::ZERO),
                scioxide_core::quantity!("kernel_spectrum"),
            )?;
            let mut scratch = try_filled(
                plan.complex_len()?,
                Complex::new(T::Real::ZERO, T::Real::ZERO),
                scioxide_core::quantity!("kernel_fft_workspace"),
            )?;
            plan.prepare_spectrum(
                &plan.kernel,
                kernel_scale,
                &mut time,
                &mut spectrum,
                &mut scratch,
                context,
            )?;
            plan.spectrum = spectrum;
        }
        Ok(plan)
    }
    /// Returns the output shape: the signal shape with each selected axis sized
    /// by the [`ConvolutionExtent`].
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
    /// Prepared signal extents.
    pub(super) fn input_shape(&self) -> &[usize] {
        &self.shape
    }
    /// Returns the selected method and its cost estimates.
    pub fn selection(&self) -> ConvolutionSelection {
        self.selection
    }
    /// Returns the convolution of `input` with the stored kernel as a row-major
    /// array of shape [`output_shape`](Self::output_shape).
    ///
    /// This allocates the output and a workspace, and runs with
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the shape of `input` differs from the prepared
    ///   shape.
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` if a result overflows.
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn convolve(&self, input: NdView<'_, '_, T>) -> Result<Vec<T>, SciOxideError> {
        let mut output = try_filled(
            self.output_len,
            T::ZERO,
            scioxide_core::quantity!("convolution_output"),
        )?;
        let mut workspace = self.workspace(ConvolutionKernel::Prepared)?;
        self.convolve_into(
            input,
            NdViewMut::new(
                &mut output,
                StridedLayout::new(&self.output_shape, &self.output_strides)?,
            )?,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Returns the workspace that one call with the stored or a supplied kernel
    /// needs.
    ///
    /// A supplied kernel needs at least as much as the stored one.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    pub fn workspace_requirements(
        &self,
        kernel: ConvolutionKernel,
    ) -> Result<SignalRequirements, SciOxideError> {
        let quantity = scioxide_core::quantity!("convolution_workspace");
        let values = checked_add(
            checked_add(self.input_len, self.output_len, quantity)?,
            self.real_time_len(),
            quantity,
        )?;
        let complex = self.complex_len()?;
        Ok(match kernel {
            ConvolutionKernel::Prepared => SignalRequirements::new(values, 0, complex),
            ConvolutionKernel::Supplied => SignalRequirements::new(
                checked_add(values, self.kernel.len(), quantity)?,
                0,
                checked_add(complex, self.spectrum.len(), quantity)?,
            ),
        })
    }
    /// Allocates a workspace for [`workspace_requirements(kernel)`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size does not fit in `usize`.
    /// - `AllocationFailure` if the allocator fails.
    pub fn workspace(
        &self,
        kernel: ConvolutionKernel,
    ) -> Result<SignalWorkspace<T>, SciOxideError> {
        SignalWorkspace::reserved(&self.workspace_requirements(kernel)?)
    }
    fn layouts(&self) -> Result<(StridedLayout<'_>, StridedLayout<'_>), SciOxideError> {
        Ok((
            StridedLayout::new(&self.padded, &self.padded_strides)?,
            StridedLayout::new(&self.spectrum_shape, &self.spectrum_strides)?,
        ))
    }
    fn prepare_spectrum(
        &self,
        kernel: &[T],
        scale_exponent: i32,
        time: &mut [T],
        spectrum: &mut [Complex<T::Real>],
        work: &mut [Complex<T::Real>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let Some(forward) = &self.forward else {
            return Ok(());
        };
        let load = |value: T| {
            scale(
                if self.options.product == SignalProduct::Correlation {
                    value.conj()
                } else {
                    value
                },
                -scale_exponent,
            )
        };
        if T::IS_REAL {
            time.fill(T::ZERO);
            runs::copy(&self.kernel_runs, kernel, time, load);
            forward.forward_real(
                T::Real::from_scalar_slice(T::slice(time))?,
                spectrum,
                self.layouts()?,
                work,
                context,
            )
        } else {
            let (buffer, scratch) = work.split_at_mut(self.padded_len);
            buffer.fill(Complex::new(T::Real::ZERO, T::Real::ZERO));
            runs::copy(&self.kernel_runs, kernel, buffer, |value| {
                crate::util::complex(load(value))
            });
            forward.complex(buffer, spectrum, self.layouts()?.0, scratch, context)
        }
    }

    /// Writes the convolution of `input` with the stored kernel to `output`.
    ///
    /// `output` has shape [`output_shape`](Self::output_shape), and `workspace`
    /// must satisfy
    /// [`workspace_requirements(ConvolutionKernel::Prepared)`](Self::workspace_requirements).
    ///
    /// ```
    /// use scioxide_core::{ExecutionContext, NdView, NdViewMut, StridedLayout};
    /// use scioxide_signal::{ConvolutionKernel, ConvolutionNdPlan, ConvolutionOptions};
    /// let kernel = [1.0_f64, -1.0];
    /// let plan = ConvolutionNdPlan::new(
    ///     &[3],
    ///     NdView::new(&kernel, StridedLayout::new(&[2], &[1])?)?,
    ///     &[0],
    ///     ConvolutionOptions::default(),
    /// )?;
    /// let mut workspace = plan.workspace(ConvolutionKernel::Prepared)?;
    /// let mut output = [0.0; 4];
    /// plan.convolve_into(
    ///     NdView::new(&[1.0, 2.0, 3.0], StridedLayout::new(&[3], &[1])?)?,
    ///     NdViewMut::new(&mut output, StridedLayout::new(&[4], &[1])?)?,
    ///     &mut workspace,
    ///     ExecutionContext::sequential_native(),
    /// )?;
    /// // First difference of [1, 2, 3] with zero extension.
    /// assert_eq!(output, [1.0, 1.0, 1.0, -3.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` or `output` does not have the prepared
    ///   shape, or `workspace` is too small.
    /// - `NonFinite` if `input` contains NaN or infinity.
    /// - `NumericalInstability` if a result overflows.
    /// - `ExecutionStopped` if `context` is cancelled or past its deadline.
    pub fn convolve_into(
        &self,
        input: NdView<'_, '_, T>,
        output: NdViewMut<'_, '_, T>,
        workspace: &mut SignalWorkspace<T>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.check_views(&input, &output)?;
        let buffers =
            workspace.buffers(&self.workspace_requirements(ConvolutionKernel::Prepared)?)?;
        self.run_views(
            input,
            output,
            buffers.values,
            buffers.complex,
            self.prepared_kernel(),
            context,
        )
    }
    /// Writes the convolution of `input` with `kernel` to `output`, in place of
    /// the stored kernel.
    ///
    /// `kernel` has the shape of the stored kernel. With the FFT method, this
    /// transforms both operands on each call. `workspace` must satisfy
    /// [`workspace_requirements(ConvolutionKernel::Supplied)`](Self::workspace_requirements).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input`, `kernel` or `output` does not have the
    ///   prepared shape, or `workspace` is too small.
    /// - `NonFinite` if `input` or `kernel` contains NaN or infinity.
    /// - `NumericalInstability` if a result overflows.
    /// - `ExecutionStopped` if `context` is cancelled or past its deadline.
    pub fn convolve_pair_into(
        &self,
        input: NdView<'_, '_, T>,
        kernel: NdView<'_, '_, T>,
        output: NdViewMut<'_, '_, T>,
        workspace: &mut SignalWorkspace<T>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.check_views(&input, &output)?;
        exact_shape(&self.kernel_shape, kernel.layout().shape().extents())?;
        let buffers =
            workspace.buffers(&self.workspace_requirements(ConvolutionKernel::Supplied)?)?;
        context.check()?;
        let (coefficients, work) = buffers.values.split_at_mut(self.kernel.len());
        let coefficients = if let Some(values) = crate::util::vector_storage(&kernel) {
            values
        } else {
            crate::util::copy_view(&kernel, coefficients);
            coefficients
        };
        let kernel_scale =
            fft_exponent(input_scale(coefficients.iter().copied())?, self.padded_len)?;
        let (spectrum, complex_work) = buffers.complex.split_at_mut(self.spectrum.len());
        if self.forward.is_some() {
            // Reuse the real time part of the execution values for preparation.
            let offset = self.input_len + self.output_len;
            self.prepare_spectrum(
                coefficients,
                kernel_scale,
                &mut work[offset..],
                spectrum,
                complex_work,
                context,
            )?;
        }
        self.run_views(
            input,
            output,
            work,
            complex_work,
            KernelRef {
                coefficients,
                scale: kernel_scale,
                spectrum,
            },
            context,
        )
    }
    /// Specialize contiguous 1-D/2-D zero-extended convolution. Geometry is
    /// computed once per output, leaving only contiguous tap loads in each row.
    fn direct_rows(
        &self,
        input: &[T],
        kernel: &[T],
        output: &mut [T],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let axis = self.shape.len() - 1;
        let width = self.shape[axis];
        let kernel_width = self.kernel_shape[axis];
        let output_width = self.output_shape[axis];
        let (height, kernel_height, row_crop) = if axis == 0 {
            (1, 1, 0)
        } else {
            (self.shape[0], self.kernel_shape[0], self.crop[0])
        };
        for (i, slot) in output.iter_mut().enumerate() {
            context.check()?;
            let row = i / output_width + row_crop;
            let column = i % output_width + self.crop[axis];
            let rows = row.saturating_sub(height - 1)..kernel_height.min(row + 1);
            let columns = column.saturating_sub(width - 1)..kernel_width.min(column + 1);
            *slot = crate::util::working_sum_products(rows.flat_map(|r| {
                let source_row = (row - r) * width;
                let kernel_row = r * kernel_width;
                columns
                    .clone()
                    .map(move |c| (input[source_row + column - c], kernel[kernel_row + c]))
            }))?;
        }
        Ok(())
    }
    fn prepared_kernel(&self) -> KernelRef<'_, T> {
        KernelRef {
            coefficients: &self.kernel,
            scale: self.kernel_scale,
            spectrum: &self.spectrum,
        }
    }
    /// Reject views other than the prepared input and output shapes.
    pub(super) fn check_views(
        &self,
        input: &NdView<'_, '_, T>,
        output: &NdViewMut<'_, '_, T>,
    ) -> Result<(), SciOxideError> {
        exact_shape(&self.shape, input.layout().shape().extents())?;
        exact_shape(&self.output_shape, output.layout().shape().extents())?;
        Ok(())
    }
    /// Execute validated views over exactly sized prepared-kernel buffers and
    /// write the output after the complete result succeeds.
    pub(super) fn run_prepared(
        &self,
        input: NdView<'_, '_, T>,
        output: NdViewMut<'_, '_, T>,
        values: &mut [T],
        complex: &mut [Complex<T::Real>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.run_views(
            input,
            output,
            values,
            complex,
            self.prepared_kernel(),
            context,
        )
    }
    fn run_views(
        &self,
        input: NdView<'_, '_, T>,
        mut output: NdViewMut<'_, '_, T>,
        values: &mut [T],
        complex: &mut [Complex<T::Real>],
        kernel: KernelRef<'_, T>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.check()?;
        let (dense, rest) = values.split_at_mut(self.input_len);
        let (staged, real_time) = rest.split_at_mut(self.output_len);
        let dense = if let Some(values) = crate::util::vector_storage(&input) {
            values
        } else {
            crate::util::copy_view(&input, dense);
            dense
        };
        self.execute_dense(dense, staged, real_time, complex, kernel, context)?;
        context.check()?;
        crate::axis::publish(staged, &mut output);
        Ok(())
    }
    /// Values of real FFT time staging; zero for direct and complex execution.
    pub(super) fn real_time_len(&self) -> usize {
        if T::IS_REAL && self.forward.is_some() {
            self.padded_len
        } else {
            0
        }
    }
    /// Complex time, frequency and transform scratch; zero for direct execution.
    pub(super) fn complex_len(&self) -> Result<usize, SciOxideError> {
        if self.forward.is_none() {
            return Ok(0);
        }
        checked_add(
            checked_add(
                if T::IS_REAL { 0 } else { self.padded_len },
                self.spectrum_len,
                scioxide_core::quantity!("fft_buffers"),
            )?,
            self.fft_len,
            scioxide_core::quantity!("fft_workspace"),
        )
    }
    /// Execute a dense prepared-kernel record into private staging. Callers
    /// supply exactly `input_len`, `output_len`, `real_time_len` and
    /// `complex_len` values.
    pub(super) fn convolve_staged(
        &self,
        input: &[T],
        output: &mut [T],
        real_time: &mut [T],
        complex: &mut [Complex<T::Real>],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.execute_dense(
            input,
            output,
            real_time,
            complex,
            self.prepared_kernel(),
            context,
        )
    }
    fn execute_dense(
        &self,
        dense: &[T],
        staged: &mut [T],
        real_time: &mut [T],
        complex_scratch: &mut [Complex<T::Real>],
        kernel: KernelRef<'_, T>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let KernelRef {
            coefficients,
            scale: kernel_scale,
            spectrum,
        } = kernel;
        context.check()?;
        let maximum = input_scale(dense.iter().copied())?;
        let input_exponent = fft_exponent(maximum, self.padded_len)?;
        if maximum == T::Real::ZERO {
            staged.fill(T::ZERO);
        } else if let (Some(forward), Some(inverse)) = (&self.forward, &self.inverse) {
            let (time, rest) =
                complex_scratch.split_at_mut(if T::IS_REAL { 0 } else { self.padded_len });
            let (frequency, fft_scratch) = rest.split_at_mut(self.spectrum_len);
            if T::IS_REAL {
                real_time.fill(T::ZERO);
                runs::copy(&self.input_runs, dense, real_time, |value| {
                    scale(value, -input_exponent)
                });
                forward.forward_real(
                    T::Real::from_scalar_slice(T::slice(real_time))?,
                    frequency,
                    self.layouts()?,
                    fft_scratch,
                    context,
                )?;
            } else {
                time.fill(Complex::new(T::Real::ZERO, T::Real::ZERO));
                runs::copy(&self.input_runs, dense, time, |value| {
                    crate::util::complex(scale(value, -input_exponent))
                });
                forward.complex(time, frequency, self.layouts()?.0, fft_scratch, context)?;
            }
            frequency
                .iter_mut()
                .zip(spectrum)
                .for_each(|(a, b)| *a *= *b);
            if T::IS_REAL {
                inverse.inverse_real(
                    frequency,
                    T::Real::from_scalar_slice_mut(T::slice_mut(real_time))?,
                    self.layouts()?,
                    fft_scratch,
                    context,
                )?;
            } else {
                inverse.complex(frequency, time, self.layouts()?.0, fft_scratch, context)?;
            }
            if T::IS_REAL {
                runs::copy(&self.output_runs, real_time, staged, |value| value);
            } else {
                runs::copy(&self.output_runs, time, staged, |value| {
                    T::from_components(value.re, value.im)
                });
            }
            restore_fft(staged, input_exponent + kernel_scale)?;
        } else if self.shape.len() <= 2
            && self.selected.iter().all(|&selected| selected)
            && self.options.boundary == SignalBoundary::Zero
            && self.options.product == SignalProduct::Convolution
        {
            self.direct_rows(dense, coefficients, staged, context)?;
        } else {
            for (i, slot) in staged.iter_mut().enumerate() {
                context.check()?;
                // Bound each selected axis by its actual overlap. In particular,
                // a long kernel against a short zero-extended signal must not
                // scan the entire kernel for every output or scan other channels.
                let tap_range = |axis: usize| {
                    if !self.selected[axis] {
                        return 0..1;
                    }
                    let len = self.kernel_shape[axis];
                    if self.options.boundary != SignalBoundary::Zero {
                        return 0..len;
                    }
                    let out =
                        i / self.output_strides[axis] % self.output_shape[axis] + self.crop[axis];
                    let start = out.saturating_sub(self.shape[axis] - 1);
                    let end = len.min(out + 1);
                    if self.options.product == SignalProduct::Correlation {
                        len - end..len - start
                    } else {
                        start..end
                    }
                };
                // This product is bounded by the validated kernel volume.
                let taps: usize = (0..self.shape.len())
                    .map(|axis| tap_range(axis).len())
                    .product();
                let pairs = (0..taps).filter_map(|mut tap| {
                    let mut input_index = 0;
                    let mut kernel_index = 0;
                    for axis in (0..self.shape.len()).rev() {
                        let out = i / self.output_strides[axis] % self.output_shape[axis];
                        let kc = if self.selected[axis] {
                            let range = tap_range(axis);
                            let c = range.start + tap % range.len();
                            tap /= range.len();
                            c
                        } else if self.kernel_shape[axis] == 1 {
                            0
                        } else {
                            out
                        };
                        kernel_index += kc * self.kernel_strides[axis];
                        let source = if self.selected[axis] {
                            let offset = if self.options.product == SignalProduct::Correlation {
                                self.kernel_shape[axis] - 1 - kc
                            } else {
                                kc
                            };
                            boundary_index(
                                (out + self.crop[axis]) as i128 - offset as i128,
                                self.shape[axis],
                                self.options.boundary,
                            )?
                        } else {
                            out
                        };
                        input_index += source * self.input_strides[axis];
                    }
                    let b = coefficients[kernel_index];
                    Some((
                        dense[input_index],
                        if self.options.product == SignalProduct::Correlation {
                            b.conj()
                        } else {
                            b
                        },
                    ))
                });
                *slot = crate::util::working_sum_products(pairs)?;
            }
        }
        context.check()?;

        Ok(())
    }
}
