//! Tensor-product Bernstein nets for multivariate polynomials, curves and patches.
//!
//! Nets apply the univariate de Casteljau kernels of the Bernstein basis along
//! each axis; the univariate basis does not depend on this module.
use crate::basis::casteljau::{
    Blend, de_casteljau_in_place, elevate_curve, normalized, split_parameter, triangle,
};
use crate::basis::control_bounds;
use crate::util::{invalid, minimum_len, try_collect};
use scioxide_core::backend::try_copy_slice;

use crate::BernsteinBounds;
use crate::execution::dispatch;
use crate::{PolynomialExecutionOptions, PolynomialScratchRequirements};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{ExecutionContext, Interval, NdView, NumericScalar, RealScalar, SciOxideError};

/// Tensor-product Bernstein polynomial in `d` parameters with `k` output
/// coordinates: a Bézier curve, surface patch or volume.
///
/// With degree `nₗ` and interval `[aₗ, bₗ]` on axis `l`, the polynomial is
///
/// ```text
/// p(x) = Σ b_(i₁…i_d) B_(i₁,n₁)(t₁) ⋯ B_(i_d,n_d)(t_d),   tₗ = (xₗ − aₗ)/(bₗ − aₗ),
/// ```
///
/// where `B_(i,n)(t) = C(n, i) tⁱ (1 − t)ⁿ⁻ⁱ` and each control point
/// `b_(i₁…i_d)` has `k` coordinates. Control points are stored in row-major
/// order (the last axis varies fastest), with the `k` coordinates of each
/// control point adjacent. One coordinate gives a scalar polynomial. With
/// `d = 0` the net is a single point.
///
/// Construct the owned net with [`new`](Self::new) and evaluate it with
/// [`evaluate`](Self::evaluate); [`as_view`](Self::as_view) gives the
/// allocation-free and batch forms. The net supports differentiation, degree
/// elevation and subdivision along an axis, and bounds from the control
/// points.
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_poly::TensorBernstein;
///
/// // Quadratic Bézier curve with control points (0, 0), (1, 2), (2, 0);
/// // at t = 1/2 it passes through (1, 1).
/// let controls = vec![0.0_f64, 0.0, 1.0, 2.0, 2.0, 0.0];
/// let curve = TensorBernstein::new(&[2], vec![Interval::new(0.0, 1.0)?], 2, controls)?;
/// let point = curve.evaluate(&[0.5])?;
/// assert!((point[0] - 1.0).abs() < 1e-14 && (point[1] - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// G. Farin, *Curves and Surfaces for CAGD: A Practical Guide*, 5th ed.,
/// Morgan Kaufmann, 2002.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorBernstein<C: NumericScalar> {
    extents: Vec<usize>,
    domains: Vec<Interval<C::Real>>,
    components: usize,
    coefficients: Vec<C>,
    options: PolynomialExecutionOptions,
}
impl<C: NumericScalar> TensorBernstein<C> {
    /// Creates a net from per-axis degrees, intervals and control coordinates.
    ///
    /// `degrees[l]` is the degree on axis `l`, and `domains[l]` its interval.
    /// `coefficients` holds `k · Π (degrees[l] + 1)` values in the order
    /// described on [`TensorBernstein`], where `k = components`. Batch
    /// evaluation of the net runs sequentially; use
    /// [`with_options`](Self::with_options) to choose the execution options.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `components` is zero, `degrees` and
    ///   `domains` differ in length, a domain has `lower ≥ upper`, or the
    ///   number of coefficients is wrong.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `SizeOverflow` if the number of controls overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Interval;
    /// use scioxide_poly::TensorBernstein;
    ///
    /// let domains = vec![Interval::new(0.0_f64, 1.0)?];
    /// let tensor = TensorBernstein::new(&[1], domains, 1, vec![1.0_f64, 3.0])?;
    /// assert_eq!(tensor.components(), 1);
    /// assert_eq!(tensor.extents(), &[2]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        degrees: &[usize],
        domains: Vec<Interval<C::Real>>,
        components: usize,
        coefficients: Vec<C>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(
            degrees,
            domains,
            components,
            coefficients,
            PolynomialExecutionOptions::new(),
        )
    }
    /// Creates a net whose batch evaluation uses `options`.
    ///
    /// This is the configured form of [`new`](Self::new). The views of the net
    /// split [`evaluate_batch_into`](TensorBernsteinView::evaluate_batch_into)
    /// among workers as `options` sets.
    ///
    /// # Errors
    ///
    /// The errors of [`new`](Self::new), and `InvalidInput` if `options` has a
    /// worker bound above 64.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Interval;
    /// use scioxide_poly::{PolynomialExecutionOptions, TensorBernstein};
    ///
    /// let domains = vec![Interval::new(0.0_f64, 1.0)?];
    /// let options = PolynomialExecutionOptions::new();
    /// let tensor = TensorBernstein::with_options(&[1], domains, 1, vec![1.0_f64, 3.0], options)?;
    /// assert_eq!(tensor.options(), options);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn with_options(
        degrees: &[usize],
        domains: Vec<Interval<C::Real>>,
        components: usize,
        coefficients: Vec<C>,
        options: PolynomialExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let extents = try_collect(
            degrees
                .iter()
                .map(|&d| checked_add(d, 1, scioxide_core::quantity!("extents"))),
            scioxide_core::quantity!("extents"),
        )?;
        validate_tensor(&extents, &domains, components, &coefficients, options)?;
        Ok(Self {
            extents,
            domains,
            components,
            coefficients,
            options,
        })
    }
    /// Returns the execution options for batch evaluation.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let tensor = TensorBernstein::new(
    ///     &[1],
    ///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
    ///     1,
    ///     vec![1.0_f64, 3.0],
    /// )?;
    /// assert_eq!(tensor.options(), PolynomialExecutionOptions::default());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn options(&self) -> PolynomialExecutionOptions {
        self.options
    }
    /// Returns the parameter intervals in axis order.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let tensor = TensorBernstein::new(
    ///     &[1],
    ///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
    ///     1,
    ///     vec![1.0_f64, 3.0],
    /// )?;
    /// assert_eq!(tensor.domains(), &[scioxide_core::Interval::new(0.0, 1.0)?]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn domains(&self) -> &[Interval<C::Real>] {
        &self.domains
    }
    /// Returns the number of control points per axis, the degree plus one.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let tensor = TensorBernstein::new(
    ///     &[1],
    ///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
    ///     1,
    ///     vec![1.0_f64, 3.0],
    /// )?;
    /// assert_eq!(tensor.extents(), &[2]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn extents(&self) -> &[usize] {
        &self.extents
    }
    /// Returns the control coordinates in the order described on [`TensorBernstein`].
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let tensor = TensorBernstein::new(
    ///     &[1],
    ///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
    ///     1,
    ///     vec![1.0_f64, 3.0],
    /// )?;
    /// assert_eq!(tensor.coefficients(), &[1.0, 3.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn coefficients(&self) -> &[C] {
        &self.coefficients
    }
    /// Returns the number of output coordinates `k`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let tensor = TensorBernstein::new(
    ///     &[1],
    ///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
    ///     1,
    ///     vec![1.0_f64, 3.0],
    /// )?;
    /// assert_eq!(tensor.components(), 1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn components(&self) -> usize {
        self.components
    }
    /// Evaluates the net at `point` and returns its `k` coordinates.
    ///
    /// `point` has one real or complex parameter per axis. Each coordinate is
    /// computed by de Casteljau's algorithm along one axis after another, in
    /// O(N) operations for `N` control points. The call allocates the result
    /// and the scratch of [`TensorBernsteinView::evaluate_into`], which it runs
    /// on the calling thread.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `point` does not have one entry per axis.
    /// - `NonFinite` if `point` has a NaN or infinite component.
    /// - `NumericalInstability` if an intermediate value overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the result or scratch cannot
    ///   be sized or allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Interval;
    /// use scioxide_poly::TensorBernstein;
    ///
    /// // p(x, y) = x + y on [0, 1]² has controls 0, 1, 1, 2.
    /// let domains = vec![Interval::new(0.0_f64, 1.0)?; 2];
    /// let patch = TensorBernstein::new(&[1, 1], domains, 1, vec![0.0_f64, 1.0, 1.0, 2.0])?;
    /// assert!((patch.evaluate(&[0.25, 0.5])?[0] - 0.75).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate(&self, point: &[C]) -> Result<Vec<C>, SciOxideError> {
        let view = self.as_view();
        let mut output = try_filled(
            self.components,
            C::ZERO,
            scioxide_core::quantity!("tensor_point"),
        )?;
        let mut scratch = try_filled(
            view.workspace_requirements(1)?.scratch_len(),
            C::ZERO,
            scioxide_core::quantity!("tensor_scratch"),
        )?;
        view.evaluate_into(
            point,
            &mut output,
            &mut scratch,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Returns a view for evaluating this net.
    ///
    /// The view borrows the controls and copies nothing.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let tensor = TensorBernstein::new(
    ///     &[1],
    ///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
    ///     1,
    ///     vec![1.0_f64, 3.0],
    /// )?;
    /// let view = tensor.as_view();
    /// let mut scratch = vec![0.0; view.workspace_requirements(1)?.scratch_len()];
    /// let mut out = [0.0];
    /// let context = scioxide_core::ExecutionContext::sequential();
    /// view.evaluate_into(&[0.5], &mut out, &mut scratch, context)?;
    /// assert!((out[0] - 2.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn as_view(&self) -> TensorBernsteinView<'_, C> {
        TensorBernsteinView {
            extents: &self.extents,
            domains: &self.domains,
            components: self.components,
            coefficients: &self.coefficients,
            options: self.options,
        }
    }
    /// Extent and stride of the caller's `axis`.
    fn axis_size(&self, axis: usize) -> Result<(usize, usize), SciOxideError> {
        if axis >= self.extents.len() {
            return Err(invalid(
                scioxide_core::argument!("axis"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Bernstein axis out of bounds",
            ));
        }
        // `stride` is the distance between consecutive entries along `axis`; all
        // axes before it form `len / (n * stride)` independent blocks.
        let stride = self.extents[axis + 1..]
            .iter()
            .try_fold(self.components, |n, &e| {
                checked_mul(n, e, scioxide_core::quantity!("axis_stride"))
            })?;
        Ok((self.extents[axis], stride))
    }
    fn empty_axis(&self, axis: usize, new_extent: usize) -> Result<Self, SciOxideError> {
        let (n, _) = self.axis_size(axis)?;
        let size = checked_mul(
            self.coefficients.len() / n,
            new_extent,
            scioxide_core::quantity!("control_coefficients"),
        )?;
        let mut extents = try_copy_slice(&self.extents, scioxide_core::quantity!("extents"))?;
        extents[axis] = new_extent;
        Ok(Self {
            extents,
            domains: try_copy_slice(&self.domains, scioxide_core::quantity!("domains"))?,
            components: self.components,
            coefficients: try_filled(
                size,
                C::ZERO,
                scioxide_core::quantity!("control_coefficients"),
            )?,
            options: self.options,
        })
    }
    fn map_axis(
        &self,
        axis: usize,
        new_extent: usize,
        transform: impl Fn(&[C], &mut [C]) -> Result<(), SciOxideError>,
    ) -> Result<Self, SciOxideError> {
        let (n, stride) = self.axis_size(axis)?;
        let mut result = self.empty_axis(axis, new_extent)?;
        let mut lane = try_filled(n, C::ZERO, scioxide_core::quantity!("axis_lane"))?;
        let mut transformed =
            try_filled(new_extent, C::ZERO, scioxide_core::quantity!("axis_output"))?;
        for block in 0..self.coefficients.len() / (n * stride) {
            for offset in 0..stride {
                for (i, c) in lane.iter_mut().enumerate() {
                    *c = self.coefficients[(block * n + i) * stride + offset];
                }
                transform(&lane, &mut transformed)?;
                for (i, &c) in transformed.iter().enumerate() {
                    result.coefficients[(block * new_extent + i) * stride + offset] = c;
                }
            }
        }
        Ok(result)
    }
    /// Returns the partial derivative `∂p/∂x_axis` as a net on the same box.
    ///
    /// The degree on `axis` drops by one, with controls
    /// `nₗ (b_(…,i+1,…) − b_(…,i,…)) / (bₗ − aₗ)`. A constant axis stays
    /// constant with zero controls.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `axis` is not below the number of axes.
    /// - `NumericalInstability` if a control coordinate overflows.
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// The derivative of `p(x) = 1 + 2x` is 2.
    /// ```
    /// use scioxide_poly::*;
    /// let tensor = TensorBernstein::new(
    ///     &[1],
    ///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
    ///     1,
    ///     vec![1.0_f64, 3.0],
    /// )?;
    /// let result = tensor.derivative(0)?;
    /// let view = result.as_view();
    /// let mut scratch = vec![0.0_f64; view.workspace_requirements(1)?.scratch_len()];
    /// let mut output = [0.0_f64];
    /// let context = scioxide_core::ExecutionContext::sequential();
    /// view.evaluate_into(&[0.5_f64], &mut output, &mut scratch, context)?;
    /// assert!((output[0] - 2.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn derivative(&self, axis: usize) -> Result<Self, SciOxideError> {
        let (n, _) = self.axis_size(axis)?;
        let d = self.domains[axis];
        self.map_axis(axis, n.saturating_sub(1).max(1), |lane, output| {
            crate::basis::bernstein_derivative_coefficients_into(lane, d.lower(), d.upper(), output)
        })
    }
    /// Returns the same polynomial with degree `degree` on `axis`.
    ///
    /// Degree elevation forms convex combinations of neighboring controls
    /// along `axis`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `axis` is not below the number of
    ///   axes, or `degree` is below the current degree on `axis`.
    /// - `SizeOverflow` if the number of controls overflows `usize`.
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let tensor = TensorBernstein::new(
    ///     &[1],
    ///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
    ///     1,
    ///     vec![1.0_f64, 3.0],
    /// )?;
    /// let result = tensor.elevate_degree(0, 3)?;
    /// let view = result.as_view();
    /// let mut scratch = vec![0.0_f64; view.workspace_requirements(1)?.scratch_len()];
    /// let mut output = [0.0_f64];
    /// let context = scioxide_core::ExecutionContext::sequential();
    /// view.evaluate_into(&[0.5_f64], &mut output, &mut scratch, context)?;
    /// assert!((output[0] - 2.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn elevate_degree(&self, axis: usize, degree: usize) -> Result<Self, SciOxideError> {
        let (n, _) = self.axis_size(axis)?;
        let target = checked_add(degree, 1, scioxide_core::quantity!("axis_controls"))?;
        if target < n {
            return Err(invalid(
                scioxide_core::argument!("degree"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Bernstein degree reduction is unsupported",
            ));
        }
        checked_mul(
            self.coefficients.len() / n,
            target,
            scioxide_core::quantity!("control_coefficients"),
        )?;
        self.map_axis(axis, target, |lane, output| {
            output[..n].copy_from_slice(lane);
            elevate_curve(output, n)
        })
    }
    /// Splits the net along `axis` at `point` into nets on the two sub-boxes.
    ///
    /// Both nets represent the same polynomial. Their controls come from de
    /// Casteljau's algorithm on each line of controls along `axis`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `axis` is not below the number of
    ///   axes, or `point` is not strictly inside the interval of `axis`.
    /// - `NonFinite` if `point` is NaN or infinite.
    /// - `NumericalInstability` if a control coordinate overflows.
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let tensor = TensorBernstein::new(
    ///     &[1],
    ///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
    ///     1,
    ///     vec![1.0_f64, 3.0],
    /// )?;
    /// let (left, right) = tensor.subdivide(0, 0.5)?;
    /// assert!((left.domains()[0].upper() - 0.5).abs() < 1e-13);
    /// assert!((right.domains()[0].lower() - 0.5).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn subdivide(&self, axis: usize, point: C::Real) -> Result<(Self, Self), SciOxideError> {
        let (n, stride) = self.axis_size(axis)?;
        let domain = self.domains[axis];
        let t = split_parameter::<C>(point, domain)?;
        let empty = || self.empty_axis(axis, n);
        let mut left = empty()?;
        let mut right = empty()?;
        let mut lane = try_filled(n, C::ZERO, scioxide_core::quantity!("axis_lane"))?;
        let weights = Blend::new(t);
        for block in 0..self.coefficients.len() / (n * stride) {
            for offset in 0..stride {
                for (i, c) in lane.iter_mut().enumerate() {
                    *c = self.coefficients[(block * n + i) * stride + offset];
                }
                triangle(&mut lane, &weights, |level| {
                    left.coefficients[(block * n + n - level.len()) * stride + offset] = level[0];
                    right.coefficients[(block * n + level.len() - 1) * stride + offset] =
                        level[level.len() - 1];
                })?;
            }
        }
        left.domains[axis] = Interval::new(domain.lower(), point)?;
        right.domains[axis] = Interval::new(point, domain.upper())?;
        Ok((left, right))
    }
}
impl<R: RealScalar> TensorBernstein<R> {
    /// Returns the range of each output coordinate over the control points.
    ///
    /// By the convex-hull property, the polynomial on its parameter box lies in
    /// these ranges; see [`BernsteinBounds`] for the limits of this estimate.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the result cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let tensor = TensorBernstein::new(
    ///     &[1],
    ///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
    ///     1,
    ///     vec![1.0_f64, 3.0],
    /// )?;
    /// assert_eq!(
    ///     tensor.control_bounds()?.coordinates()[0],
    ///     scioxide_core::Interval::new(1.0, 3.0)?
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn control_bounds(&self) -> Result<BernsteinBounds<R>, SciOxideError> {
        control_bounds(&self.coefficients, self.components)
    }
}
fn validate_tensor<C: NumericScalar>(
    extents: &[usize],
    domains: &[Interval<C::Real>],
    components: usize,
    coefficients: &[C],
    options: PolynomialExecutionOptions,
) -> Result<(), SciOxideError> {
    options.validate()?;
    if extents.len() != domains.len()
        || components == 0
        || extents.contains(&0)
        || domains.iter().any(|d| d.lower() >= d.upper())
    {
        let argument = if components == 0 {
            scioxide_core::argument!("components")
        } else if extents.len() == domains.len() && extents.contains(&0) {
            scioxide_core::argument!("extents")
        } else {
            scioxide_core::argument!("domains")
        };
        return Err(invalid(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "invalid Bernstein shape or domain",
        ));
    }
    let size = extents.iter().try_fold(components, |n, &e| {
        checked_mul(n, e, scioxide_core::quantity!("control_coefficients"))
    })?;
    checked_add(
        size / components,
        components,
        scioxide_core::quantity!("evaluation_scratch"),
    )?;
    if size != coefficients.len() {
        return Err(invalid(
            scioxide_core::argument!("coefficients"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "Bernstein coefficient count mismatch",
        ));
    }
    for &c in coefficients {
        crate::util::finite_input(c, "nonfinite value")?;
    }
    Ok(())
}

/// Borrowed view of a tensor-product Bernstein net, for evaluation.
///
/// Create one with [`TensorBernstein::as_view`], or with [`new`](Self::new)
/// from borrowed slices in the layout described on [`TensorBernstein`].
/// Evaluation contracts one axis at a time with de Casteljau's algorithm,
/// starting with the last axis. One point costs O(N m) operations per output
/// coordinate, where `N` is the number of control points and `m` the largest
/// number of controls on one axis. Points outside the parameter box are
/// accepted.
///
/// ```
/// use scioxide_poly::*;
/// let tensor = TensorBernstein::new(
///     &[1],
///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
///     1,
///     vec![1.0_f64, 3.0],
/// )?;
/// let view = tensor.as_view();
/// let mut scratch = vec![0.0_f64; view.workspace_requirements(1)?.scratch_len()];
/// let mut out = [0.0_f64];
/// let context = scioxide_core::ExecutionContext::sequential();
/// view.evaluate_into(&[0.5_f64], &mut out, &mut scratch, context)?;
/// assert!((out[0] - 2.0).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct TensorBernsteinView<'a, C: NumericScalar> {
    extents: &'a [usize],
    domains: &'a [Interval<C::Real>],
    components: usize,
    coefficients: &'a [C],
    options: PolynomialExecutionOptions,
}
impl<'a, C: NumericScalar> TensorBernsteinView<'a, C> {
    /// Creates a view from borrowed extents, intervals and control coordinates.
    ///
    /// `extents[l]` is the number of control points on axis `l`, the degree
    /// plus one.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `components` or an extent is zero,
    ///   `extents` and `domains` differ in length, a domain has
    ///   `lower ≥ upper`, or the number of coefficients is wrong.
    /// - `InvalidInput` if `options` has a worker bound above 64.
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `SizeOverflow` if the number of controls overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    ///
    /// let extents = [2];
    /// let domains = [scioxide_core::Interval::new(0.0_f64, 1.0)?];
    /// let controls = [1.0_f64, 3.0];
    /// let view = TensorBernsteinView::new(&extents, &domains, 1, &controls, Default::default())?;
    /// let mut scratch = vec![0.0_f64; view.workspace_requirements(1)?.scratch_len()];
    /// let mut out = [0.0_f64];
    /// let context = scioxide_core::ExecutionContext::sequential();
    /// view.evaluate_into(&[0.5_f64], &mut out, &mut scratch, context)?;
    /// assert!((out[0] - 2.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(
        extents: &'a [usize],
        domains: &'a [Interval<C::Real>],
        components: usize,
        coefficients: &'a [C],
        options: PolynomialExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        validate_tensor(extents, domains, components, coefficients, options)?;
        Ok(Self {
            extents,
            domains,
            components,
            coefficients,
            options,
        })
    }
    /// Returns the execution options for batch evaluation.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let tensor = TensorBernstein::new(
    ///     &[1],
    ///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
    ///     1,
    ///     vec![1.0_f64, 3.0],
    /// )?;
    /// let view = tensor.as_view();
    /// let mut scratch = vec![0.0_f64; view.workspace_requirements(1)?.scratch_len()];
    /// let mut out = [0.0_f64];
    /// let context = scioxide_core::ExecutionContext::sequential();
    /// view.evaluate_into(&[0.5_f64], &mut out, &mut scratch, context)?;
    /// assert_eq!(view.options(), PolynomialExecutionOptions::default());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn options(&self) -> PolynomialExecutionOptions {
        self.options
    }
    /// Returns the scratch length for evaluating `points` points.
    ///
    /// The length is `points · k + w · N`, where `k` is the number of output
    /// coordinates, `N` the number of control points and `w` the number of
    /// workers, at most `points · k`. [`evaluate_into`](Self::evaluate_into)
    /// evaluates one point and needs `workspace_requirements(1)`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the length overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Interval;
    /// use scioxide_poly::{PolynomialExecutionOptions, TensorBernsteinView};
    /// let domains = [Interval::new(0.0, 1.0)?];
    /// let options = PolynomialExecutionOptions::new();
    /// let view = TensorBernsteinView::new(&[2], &domains, 1, &[1.0_f64, 3.0], options)?;
    /// let mut scratch = vec![0.0_f64; view.workspace_requirements(1)?.scratch_len()];
    /// let mut value = [0.0_f64];
    /// let context = scioxide_core::ExecutionContext::sequential();
    /// view.evaluate_into(&[0.5_f64], &mut value, &mut scratch, context)?;
    /// assert!((value[0] - 2.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace_requirements(
        &self,
        points: usize,
    ) -> Result<PolynomialScratchRequirements, SciOxideError> {
        let outputs = checked_mul(
            points,
            self.components,
            scioxide_core::quantity!("batch_outputs"),
        )?;
        let lanes = self.options.workers().get().min(outputs);
        Ok(PolynomialScratchRequirements::new(checked_add(
            outputs,
            checked_mul(
                lanes,
                self.coefficients.len() / self.components,
                scioxide_core::quantity!("worker_scratch"),
            )?,
            scioxide_core::quantity!("batch_scratch"),
        )?))
    }
    /// Evaluates the net at one point and writes the `k` coordinates to `output`.
    ///
    /// `point` has one real or complex parameter per axis. The call runs on the
    /// calling thread and checks `context` for cancellation before each output
    /// coordinate.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `point` does not have one entry per axis,
    ///   `output` does not have `k` entries, or `scratch` is shorter than
    ///   `workspace_requirements(1)`.
    /// - `NonFinite` if `point` has a NaN or infinite component.
    /// - `NumericalInstability` if an intermediate value overflows.
    /// - `ExecutionStopped` if `context` is cancelled or past its deadline.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let tensor = TensorBernstein::new(
    ///     &[1],
    ///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
    ///     1,
    ///     vec![1.0_f64, 3.0],
    /// )?;
    /// let view = tensor.as_view();
    /// let mut scratch = vec![0.0_f64; view.workspace_requirements(1)?.scratch_len()];
    /// let mut out = [0.0_f64];
    /// let context = scioxide_core::ExecutionContext::sequential();
    /// view.evaluate_into(&[0.5_f64], &mut out, &mut scratch, context)?;
    /// assert!((out[0] - 2.0).abs() < 1e-13);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_into(
        &self,
        point: &[C],
        output: &mut [C],
        scratch: &mut [C],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        context.check()?;
        if point.len() != self.extents.len() || output.len() != self.components {
            return Err(SciOxideError::dimension_mismatch(
                [self.extents.len(), self.components],
                [point.len(), output.len()],
            ));
        }
        minimum_len(self.workspace_requirements(1)?.scratch_len(), scratch.len())?;
        for &x in point {
            crate::util::finite_input(x, "nonfinite value")?;
        }
        let (stage, work) = scratch.split_at_mut(self.components);
        for (component, value) in stage.iter_mut().enumerate() {
            context.check()?;
            *value = self.component(component, |axis| Ok(point[axis]), work)?;
        }
        context.check()?;
        output.copy_from_slice(stage);
        Ok(())
    }
    fn component(
        &self,
        component: usize,
        point: impl Fn(usize) -> Result<C, SciOxideError>,
        work: &mut [C],
    ) -> Result<C, SciOxideError> {
        let mut length = self.coefficients.len() / self.components;
        for (i, value) in work[..length].iter_mut().enumerate() {
            *value = self.coefficients[i * self.components + component];
        }
        // Contract the tensor one axis at a time, last axis first: with the trailing
        // axes already reduced, the current axis is contiguous, so each lane is a
        // slice of length `n`. Result `i` is written to `work[i]`, which never
        // overtakes the lanes still to be read (`i <= i * n`).
        for axis in (0..self.extents.len()).rev() {
            let n = self.extents[axis];
            if n == 1 {
                continue;
            }
            let t = normalized(point(axis)?, self.domains[axis])?;
            let reduced = length / n;
            for i in 0..reduced {
                work[i] = de_casteljau_in_place(&mut work[i * n..(i + 1) * n], t)?;
            }
            length = reduced;
        }
        Ok(work[0])
    }
    /// Evaluates the net at many points.
    ///
    /// `points` has shape `[m, d]`: one row of `d` parameters per point.
    /// `output` receives `m · k` values, the `k` coordinates of each point in
    /// turn. `scratch` needs `workspace_requirements(m)` entries. The points are
    /// split among workers as set by [`options`](Self::options). Each output is
    /// computed by the same operations for any number of workers, so the
    /// results do not depend on the execution policy.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `points` is not of shape `[m, d]` or
    ///   `output` does not have `m · k` entries.
    /// - `DimensionMismatch` if `scratch` is too short.
    /// - `NonFinite` if a parameter is NaN or infinite.
    /// - `NumericalInstability` if an intermediate value overflows.
    /// - `ExecutionStopped` if `context` is cancelled or past its deadline.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_poly::*;
    ///
    /// // p(x) = 1 + 2x at x = 0.25 and 0.75.
    /// let tensor = TensorBernstein::new(
    ///     &[1],
    ///     vec![scioxide_core::Interval::new(0.0_f64, 1.0)?],
    ///     1,
    ///     vec![1.0_f64, 3.0],
    /// )?;
    /// let view = tensor.as_view();
    /// let layout = scioxide_core::StridedLayout::new(&[2, 1], &[1, 1])?;
    /// let points = [0.25_f64, 0.75];
    /// let mut out = [0.0_f64; 2];
    /// let mut scratch = vec![0.0_f64; view.workspace_requirements(2)?.scratch_len()];
    /// let points = scioxide_core::NdView::new(&points, layout)?;
    /// view.evaluate_batch_into(
    ///     points,
    ///     &mut out,
    ///     &mut scratch,
    ///     ExecutionContext::sequential(),
    /// )?;
    /// assert!(
    ///     out.iter()
    ///         .zip([1.5_f64, 2.5])
    ///         .all(|(a, b)| (a - b).abs() < 1e-13)
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn evaluate_batch_into(
        &self,
        points: NdView<'_, '_, C>,
        output: &mut [C],
        scratch: &mut [C],
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let extents = points.layout().shape().extents();
        if extents.len() != 2
            || extents[1] != self.extents.len()
            || checked_mul(
                extents[0],
                self.components,
                scioxide_core::quantity!("batch_outputs"),
            )? != output.len()
        {
            return Err(invalid(
                if extents.len() != 2 || extents[1] != self.extents.len() {
                    scioxide_core::argument!("points")
                } else {
                    scioxide_core::argument!("output")
                },
                scioxide_core::InvalidInputReason::OutOfRange,
                "Bernstein batch shape mismatch",
            ));
        }
        for &x in points.iter() {
            crate::util::finite_input(x, "nonfinite value")?;
        }
        dispatch(
            output,
            scratch,
            self.coefficients.len() / self.components,
            self.options.resolved_policy(),
            self.options.workers(),
            context,
            |i, work| {
                self.component(
                    // Output entry `i` is component `i % components` of point `i / components`,
                    // matching the interleaved coefficient layout.
                    i % self.components,
                    |axis| Ok(*points.get([i / self.components, axis])?),
                    work,
                )
            },
        )
    }
}
