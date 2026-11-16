//! Tensor-product B-splines of any rank `D` with affine values, evaluated over
//! the `(p + 1)^D` block of nonzero basis functions.

#![allow(
    clippy::needless_range_loop,
    reason = "axis loops index several parallel per-axis arrays"
)]

use alloc::vec::Vec;
use core::ops::Range;

use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, Capacity, Interval, PromotingCompensatedSum, PromotingProduct,
    RealScalar, Scaled, SciOxideError,
};

use super::antiderivative::{PreparedSplineAntiderivative, primitive};
use super::basis::{BSplineBasis, IntegralAxis, KnotSide, axis_scratch};
use super::collocation::TensorSplineInterpolationPlan;
use crate::checked::{div, interval_coordinate, mul, ordinary, sub};
use crate::protocol::{
    Differentiable, ExtrapolationPolicy, Integrable, Interpolant, TwiceDifferentiable,
    resolve_coordinate, try_array,
};
use crate::values::{QueryWorkspace, flatten, zeroed};
use scioxide_poly::backend::polynomial_basis::{
    bernstein_to_power_into, linear_difference_ratio, linear_multiply_difference,
};

/// Represents a tensor-product B-spline with affine values; `D = 1` is a
/// B-spline curve.
///
/// With a [`BSplineBasis`] of degree `p_a` on each axis `a`, the spline is
///
/// ```text
/// f(x) = Σ_k c_k Π_a N^(a)_{k_a, p_a}(x_a),
/// ```
///
/// where `k = (k_1, …, k_D)` runs over all control multi-indices and `c_k` are
/// the control values. Coefficients are stored in row-major order of the
/// multi-index (last axis fastest), with the real components of each value
/// interleaved. [`Self::coefficients`] returns that slice. Each real
/// component evaluates exactly as the scalar spline of that component.
///
/// A query evaluates only the `p_a + 1` nonzero basis functions on each axis,
/// at a cost of `O(Σ_a (p_a² + log n_a) + Π_a (p_a + 1) · m)` operations for
/// `n_a` controls per axis and `m` real components. At a knot, values use the
/// piece to the right. [`Self::derivative_into`] takes explicit [`KnotSide`]
/// limits and otherwise rejects derivatives that do not exist at a knot.
/// Products and sums switch to a wider exponent range when native arithmetic
/// would overflow or underflow, so a component fails only if its final value
/// is outside the range of `S`. Owned value, gradient and derivative queries
/// on scalar curves of degree at most three do not allocate.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{
///     BSpline, BSplineBasis, Differentiable, ExtrapolationPolicy, Integrable, Interpolant,
/// };
/// // Quadratic Bezier controls for x^2 on [0, 1].
/// let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0], 2)?;
/// let spline = BSpline::new([basis], vec![0.0, 0.0, 1.0])?;
/// let policy = ExtrapolationPolicy::Reject;
/// assert!((spline.evaluate([0.3], policy)? - 0.09).abs() < 1e-15);
/// assert!((spline.gradient([0.3], policy)?[0] - 0.6).abs() < 1e-15);
/// assert!((spline.integral([0.0], [1.0], policy)? - 1.0 / 3.0).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// A bilinear, complex-valued surface:
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_interp::{BSpline, BSplineBasis, ExtrapolationPolicy, Interpolant};
/// let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 1.0, 1.0], 1)?;
/// let surface = BSpline::new(
///     [basis.clone(), basis],
///     vec![
///         Complex::new(1.0, 0.0),
///         Complex::new(3.0, 1.0),
///         Complex::new(4.0, 2.0),
///         Complex::new(6.0, 3.0),
///     ],
/// )?;
/// let value = surface.evaluate([0.25, 0.5], ExtrapolationPolicy::Reject)?;
/// assert_eq!(value, Complex::new(2.75, 1.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - C. de Boor, *A Practical Guide to Splines*, rev. ed., Springer, 2001,
///   chapters IX and XVII.
/// - L. Piegl and W. Tiller, *The NURBS Book*, 2nd ed., Springer, 1997,
///   chapters 2–3.
#[derive(Debug, Clone)]
#[must_use]
pub struct BSpline<S: RealScalar, V: AffineValue<Scale = S> = S, const D: usize = 1> {
    bases: [BSplineBasis<S>; D],
    coefficients: Vec<S>,
    dimension: usize,
    template: V,
    scratch: usize,
    /// Point queries do not need the larger integral-tail buffers.
    query_scratch: usize,
    /// Per component: smallest nonzero and largest coefficient magnitude,
    /// which allow the native contraction without per-term range checks.
    magnitudes: Vec<[S; 2]>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>, const D: usize> BSpline<S, V, D> {
    /// Builds a spline from one basis per axis and the control values in
    /// row-major multi-index order.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `D = 0`.
    /// - `DimensionMismatch` if the number of values differs from the product of
    ///   the axis control counts, or the values differ in shape.
    /// - `NonFinite` if a value component is NaN or infinite.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be reserved.
    pub fn new(bases: [BSplineBasis<S>; D], coefficients: Vec<V>) -> Result<Self, SciOxideError> {
        let controls = control_count(&bases)?;
        let flat = flatten(coefficients, controls)?;
        Self::from_parts(bases, flat.components, flat.dimension, flat.template)
    }

    /// Interpolates values on a tensor grid by a tensor-product spline of odd
    /// degree `p` on every axis.
    ///
    /// Axis `a` has the strictly increasing nodes `nodes[a]`, at least `p + 1`
    /// of them, and the basis [`BSplineBasis::not_a_knot`] of degree `p`. The
    /// result `f` satisfies `f(x_{1,i_1}, …, x_{D,i_D}) = y_{i_1…i_D}` at every
    /// grid point, where `values` holds `y` in row-major order with the last
    /// axis fastest. For `p = 1` this is multilinear interpolation, and for
    /// `p = 3` each axis is a cubic spline with not-a-knot ends, so `f`
    /// reproduces polynomials of degree at most `p` in each variable.
    ///
    /// This builds a [`TensorSplineInterpolationPlan`] and fits once: each axis
    /// collocation matrix is factored by banded LU in `O(n_a (p + 1)²)`, and
    /// the fit solves along every grid line. Keep the plan to fit several value
    /// sets on the same grid.
    ///
    /// # Errors
    ///
    /// - The errors of [`BSplineBasis::not_a_knot`] for each axis: an even
    ///   degree, too few nodes, or nodes that are not strictly increasing.
    /// - The errors of [`TensorSplineInterpolationPlan::new`] and
    ///   [`TensorSplineInterpolationPlan::fit`], among them `InvalidInput` for
    ///   `D = 0`, `DimensionMismatch` if `values` does not have one entry per
    ///   grid point, and `NonFinite` for a NaN or infinite value.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{BSpline, ExtrapolationPolicy, Interpolant};
    /// // f(x, y) = x³ + x y² on a 5 × 4 grid, reproduced by bicubic interpolation.
    /// let f = |x: f64, y: f64| x * x * x + x * y * y;
    /// let x = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    /// let y = vec![0.0, 1.0, 2.0, 3.0];
    /// let values: Vec<f64> = x.iter().flat_map(|&a| y.iter().map(move |&b| f(a, b))).collect();
    /// let spline = BSpline::interpolate([x, y], &values, 3)?;
    /// let value = spline.evaluate([1.25, 2.5], ExtrapolationPolicy::Reject)?;
    /// assert!((value - f(1.25, 2.5)).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// C. de Boor, *A Practical Guide to Splines*, rev. ed., Springer, 2001,
    /// chapters XIII and XVII.
    pub fn interpolate(
        nodes: [Vec<S>; D],
        values: &[V],
        degree: usize,
    ) -> Result<BSpline<S, V::Owned, D>, SciOxideError> {
        let mut axes = nodes.iter();
        let bases = try_array(|| {
            let axis = axes.next().map_or(&[][..], Vec::as_slice);
            BSplineBasis::not_a_knot(axis, degree)
        })?;
        TensorSplineInterpolationPlan::new(bases, nodes)?.fit(values)
    }

    /// Builds a spline from flattened, row-major control components.
    pub(crate) fn from_parts(
        bases: [BSplineBasis<S>; D],
        coefficients: Vec<S>,
        dimension: usize,
        template: V,
    ) -> Result<Self, SciOxideError> {
        let controls = control_count(&bases)?;
        let expected = checked_mul(
            controls,
            dimension,
            scioxide_core::quantity!("coefficients"),
        )?;
        if coefficients.len() != expected {
            return Err(crate::error::shape(expected, coefficients.len()));
        }
        crate::validate::validate_finite(&coefficients)?;
        let stage = checked_mul(
            checked_mul(D, D, scioxide_core::quantity!("hessian_entries"))?,
            dimension,
            scioxide_core::quantity!("b_spline_workspace"),
        )?;
        let scratch = scratch_len(&bases, stage, dimension)?;
        let query_scratch = bases.iter().try_fold(
            checked_add(
                stage,
                checked_mul(2, dimension, scioxide_core::quantity!("b_spline_workspace"))?,
                scioxide_core::quantity!("b_spline_workspace"),
            )?,
            |total, basis| {
                checked_add(
                    total,
                    basis.degree() + 1,
                    scioxide_core::quantity!("b_spline_workspace"),
                )
            },
        )?;
        let mut magnitudes = try_filled(
            dimension,
            [S::MAX, S::ZERO],
            scioxide_core::quantity!("b_spline_magnitudes"),
        )?;
        for control in coefficients.chunks_exact(dimension.max(1)) {
            for ([smallest, largest], &value) in magnitudes.iter_mut().zip(control) {
                if value != S::ZERO {
                    *smallest = smallest.min(value.abs());
                }
                *largest = largest.max(value.abs());
            }
        }
        Ok(Self {
            bases,
            coefficients,
            dimension,
            template,
            scratch,
            query_scratch,
            magnitudes,
        })
    }

    /// Returns the axis bases in coordinate order.
    pub const fn bases(&self) -> &[BSplineBasis<S>; D] {
        &self.bases
    }

    /// Returns the coefficients in row-major multi-index order, with the real
    /// components of each value interleaved.
    pub fn coefficients(&self) -> &[S] {
        &self.coefficients
    }

    fn tensor(&self) -> Tensor<'_, S, D> {
        Tensor {
            bases: &self.bases,
            coefficients: &self.coefficients,
            width: self.dimension,
            centered: None,
            magnitude: false,
            magnitudes: Some(&self.magnitudes),
        }
    }

    /// Writes the mixed partial derivative `∂^|k| f / ∂x_1^k_1 ⋯ ∂x_D^k_D` with
    /// `orders = [k_1, …, k_D]` to `out`.
    ///
    /// All-zero orders give the value. With `sides = None` the ordinary
    /// derivative is returned and must exist. At an interior knot of
    /// multiplicity `m`, every order `j > p − m` up to the requested one is
    /// compared exactly between the two pieces, component by component. With
    /// explicit sides, each axis evaluates the one-sided limit of the selected
    /// piece without that comparison. A side that points out of the domain is
    /// rejected under [`ExtrapolationPolicy::Reject`] and is constant, with zero
    /// derivative, under [`ExtrapolationPolicy::Clamp`]. Along an axis clamped
    /// outside the domain, derivatives are zero. Orders above an axis degree
    /// give zero.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `out` does not have `value_dimension()`
    ///   components or `workspace` is too small.
    /// - The coordinate errors listed in the [crate docs](crate#errors).
    /// - `DomainError` (`Undefined`) if `sides` is `None` and the one-sided
    ///   derivatives at a knot differ, or if a side points out of the domain
    ///   under [`ExtrapolationPolicy::Reject`].
    /// - `NonFinite` if the result overflows.
    pub fn derivative_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        orders: [usize; D],
        sides: Option<[KnotSide; D]>,
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        scioxide_core::check_value_shape(out, self.dimension)?;
        let scratch = workspace.scratch(self.scratch)?;
        self.partial_with_scratch(point, orders, sides, policy, out, scratch)
    }

    /// Evaluates one point query into `out` using `scratch`.
    fn partial_with_scratch<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        orders: [usize; D],
        sides: Option<[KnotSide; D]>,
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        let (stage, rest) = scratch.split_at_mut(self.dimension);
        let (limits, axes) = rest.split_at_mut(2 * self.dimension);
        let tensor = self.tensor();
        let located = tensor.locate(point, sides, policy)?;
        tensor.partial(&located, orders, limits, axes, stage)?;
        scioxide_core::write_value(stage, out)
    }

    /// Returns the mixed partial derivative of [`Self::derivative_into`] as an
    /// owned value.
    ///
    /// Scalar curves of degree at most three do not allocate. Other value types
    /// and degrees allocate the output and scratch.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::derivative_into`].
    /// - `AllocationFailure` if the output or scratch cannot be reserved.
    pub fn derivative(
        &self,
        point: [S; D],
        orders: [usize; D],
        sides: Option<[KnotSide; D]>,
        policy: ExtrapolationPolicy<S>,
    ) -> Result<V::Owned, SciOxideError> {
        let mut out = zeroed(&self.template)?;
        let mut workspace = QueryWorkspace::new(self.query_scratch)?;
        self.partial_with_scratch(
            point,
            orders,
            sides,
            policy,
            &mut out,
            workspace.scratch(self.query_scratch)?,
        )?;
        Ok(out)
    }

    /// Computes the ordinary partials of `orders` in scratch, then writes them to `out`.
    fn stage<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        orders: &[[usize; D]],
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        let width = orders.len() * self.dimension;
        scioxide_core::check_value_shape(out, width)?;
        let scratch = workspace.scratch(self.scratch)?;
        self.stage_with_scratch(point, policy, orders, out, scratch)
    }

    fn stage_with_scratch<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        orders: &[[usize; D]],
        out: &mut O,
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        let width = orders.len() * self.dimension;
        let stage_len = D * D * self.dimension;
        let (stage, rest) = scratch.split_at_mut(stage_len);
        let (limits, axes) = rest.split_at_mut(2 * self.dimension);
        let tensor = self.tensor();
        let located = tensor.locate(point, None, policy)?;
        for (index, &partial) in orders.iter().enumerate() {
            let chunk = &mut stage[index * self.dimension..(index + 1) * self.dimension];
            tensor.partial(&located, partial, limits, axes, chunk)?;
        }
        scioxide_core::write_value(&stage[..width], out)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S>> BSpline<S, V, 1> {
    /// Builds the piecewise-linear interpolant of the samples `(x_i, y_i)`.
    ///
    /// The result is the degree-one spline on the knots
    /// `[x_0, x_0, x_1, …, x_{n−1}, x_{n−1}]` whose coefficients are the samples,
    /// so `f(x_i) = y_i` and `f` is linear between consecutive nodes.
    /// Continuation extends the outer segments. The nodes must be strictly
    /// increasing.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if there are fewer than two nodes, or the nodes are not
    ///   strictly increasing.
    /// - `DimensionMismatch` if `x` and `y` differ in length or the values
    ///   differ in shape.
    /// - `NonFinite` if a node or value is NaN or infinite.
    /// - `AllocationFailure` if the knot or coefficient storage cannot be
    ///   reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{BSpline, ExtrapolationPolicy, Integrable, Interpolant};
    /// let line = BSpline::linear(vec![0.0_f64, 1.0, 3.0], vec![1.0, 3.0, -1.0])?;
    /// let policy = ExtrapolationPolicy::Reject;
    /// assert_eq!(line.evaluate([0.5], policy)?, 2.0);
    /// assert_eq!(line.evaluate([2.0], policy)?, 1.0);
    /// assert_eq!(line.integral([0.0], [3.0], policy)?, 4.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn linear(x: Vec<S>, y: Vec<V>) -> Result<Self, SciOxideError> {
        let basis = BSplineBasis::with_repeated_ends(&x, 2, 1, 1, scioxide_core::argument!("x"))?;
        Self::new([basis], y)
    }

    /// Inserts the knot `t` and returns the refined spline, which represents the
    /// same function with one more control point.
    ///
    /// Uses Boehm's algorithm. With `t` in the knot span `[t_k, t_{k+1})`, the
    /// new controls are `Q_i = (1 − α_i) P_{i−1} + α_i P_i` with
    /// `α_i = (t − t_i)/(t_{i+p} − t_i)` for `k − p < i ≤ k`, and the other
    /// controls are copied. For `m` real components this costs `O(p m)` plus
    /// `O(n m)` for the copy. `self` is unchanged.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `t` is NaN or infinite, and `DomainError` if it is
    ///   outside the base domain.
    /// - `InvalidInput` (`Inconsistent`) if `t` is already a knot of
    ///   multiplicity `p + 1`.
    /// - `NonFinite` if a new control overflows, and `AllocationFailure` if the
    ///   storage cannot be reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{BSpline, BSplineBasis, ExtrapolationPolicy, Interpolant};
    /// let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0], 2)?;
    /// let spline = BSpline::new([basis], vec![0.0, 1.0, 0.0])?;
    /// let refined = spline.insert_knot(0.5)?;
    /// assert_eq!(refined.bases()[0].control_count(), 4);
    /// let policy = ExtrapolationPolicy::Reject;
    /// assert!((refined.evaluate([0.3], policy)? - spline.evaluate([0.3], policy)?).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// W. Boehm, "Inserting new knots into B-spline curves", *Computer-Aided
    /// Design* 12(4), 1980, 199–201.
    pub fn insert_knot(&self, t: S) -> Result<BSpline<S, V::Owned, 1>, SciOxideError> {
        let basis = &self.bases[0];
        let t = resolve_coordinate(&basis.domain(), t, ExtrapolationPolicy::Reject)?;
        let p = basis.degree();
        let n = basis.control_count();
        if basis.multiplicity(t) > p {
            return Err(crate::error::invalid(
                scioxide_core::argument!("t"),
                "insertion would exceed degree plus one multiplicity",
            ));
        }
        let knots = basis.knots();
        let k = basis.span(t, KnotSide::Right);
        let dim = self.dimension;
        // Boehm: controls up to k - p are kept, those after k shift by one, and
        // in between Q_i = (1 - α_i) P_{i-1} + α_i P_i, α_i = (t - t_i)/(t_{i+p} - t_i).
        let mut coefficients = try_with_capacity(
            checked_mul(
                checked_add(n, 1, scioxide_core::quantity!("b_spline_controls"))?,
                dim,
                scioxide_core::quantity!("b_spline_coefficients"),
            )?,
            scioxide_core::quantity!("b_spline_coefficients"),
        )?;
        for i in 0..=n {
            if i + p <= k {
                coefficients.extend_from_slice(&self.coefficients[i * dim..(i + 1) * dim]);
            } else if i > k {
                coefficients.extend_from_slice(&self.coefficients[(i - 1) * dim..i * dim]);
            } else {
                let alpha = if knots[i + p] == knots[i] {
                    S::ZERO
                } else {
                    interval_coordinate(t, knots[i], knots[i + p])?
                };
                for c in 0..dim {
                    coefficients.push(blend(
                        self.coefficients[(i - 1) * dim + c],
                        self.coefficients[i * dim + c],
                        alpha,
                    )?);
                }
            }
        }
        let mut refined = try_with_capacity(
            checked_add(knots.len(), 1, scioxide_core::quantity!("b_spline_knots"))?,
            scioxide_core::quantity!("b_spline_knots"),
        )?;
        refined.extend_from_slice(&knots[..=k]);
        refined.push(t);
        refined.extend_from_slice(&knots[k + 1..]);
        BSpline::from_parts(
            [BSplineBasis::new(refined, p)?],
            coefficients,
            dim,
            self.template.try_to_owned()?,
        )
    }

    /// Returns the derivative `f′` as a spline of degree `p − 1`.
    ///
    /// The derivative has controls `p (P_{i+1} − P_i)/(t_{i+p+1} − t_{i+1})` on the
    /// knot vector without its first and last knot. At a knot it takes the
    /// right-piece value, so it does not mark points where the ordinary
    /// derivative of `f` does not exist; use [`Self::derivative_into`] to check
    /// that. A spline of degree zero gives the zero spline. Basis functions with
    /// empty support, which come from knots of full multiplicity, are removed.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a derivative control overflows.
    /// - `AllocationFailure` if the storage cannot be reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{BSpline, BSplineBasis, ExtrapolationPolicy, Interpolant};
    /// let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 0.0, 2.0, 2.0, 2.0], 2)?;
    /// let square = BSpline::new([basis], vec![0.0, 0.0, 4.0])?; // x^2 on [0, 2]
    /// let slope = square.derivative_spline()?;
    /// assert_eq!(slope.bases()[0].degree(), 1);
    /// assert_eq!(slope.evaluate([1.5], ExtrapolationPolicy::Reject)?, 3.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// C. de Boor, *A Practical Guide to Splines*, rev. ed., Springer, 2001,
    /// chapter X.
    pub fn derivative_spline(&self) -> Result<BSpline<S, V::Owned, 1>, SciOxideError> {
        let basis = &self.bases[0];
        let p = basis.degree();
        let n = basis.control_count();
        let knots = basis.knots();
        let dim = self.dimension;
        let template = self.template.try_to_owned()?;
        if p == 0 {
            let mut constant = try_with_capacity(2, scioxide_core::quantity!("derivative_knots"))?;
            constant.extend_from_slice(&[knots[0], knots[n]]);
            return BSpline::from_parts(
                [BSplineBasis::new(constant, 0)?],
                try_filled(
                    dim,
                    S::ZERO,
                    scioxide_core::quantity!("derivative_controls"),
                )?,
                dim,
                template,
            );
        }
        // q_i = p (c_{i+1} - c_i)/(t_{i+p+1} - t_{i+1}) on the knots without the
        // first and last; basis i has empty support when t_{i+1} = t_{i+p+1}.
        let degree = S::try_from_usize_exact(p)?;
        let mut coefficients = try_with_capacity(
            checked_mul(n - 1, dim, scioxide_core::quantity!("derivative_controls"))?,
            scioxide_core::quantity!("derivative_controls"),
        )?;
        let mut derived = try_with_capacity(
            knots.len() - 2,
            scioxide_core::quantity!("derivative_knots"),
        )?;
        for i in 0..n - 1 {
            if knots[i + 1] == knots[i + p + 1] {
                continue;
            }
            derived.push(knots[i + 1]);
            for c in 0..dim {
                coefficients.push(crate::checked::intermediate(mul(
                    degree,
                    linear_difference_ratio(
                        self.coefficients[(i + 1) * dim + c],
                        self.coefficients[i * dim + c],
                        knots[i + p + 1],
                        knots[i + 1],
                    )?,
                )?)?);
            }
        }
        derived.extend_from_slice(&knots[n..knots.len() - 1]);
        BSpline::from_parts(
            [BSplineBasis::new(derived, p - 1)?],
            coefficients,
            dim,
            template,
        )
    }

    /// Builds an antiderivative for repeated definite integrals of a curve.
    ///
    /// Converts each nonempty knot span to a local polynomial by blossoming
    /// (polar forms, equivalent to Bézier extraction), integrates it, and stores
    /// the prefix integrals at the breakpoints. For `m` spans and `c` real
    /// components, preparation costs `O(m c)` for degrees up to three and
    /// `O(m p³ c)` otherwise, and stores `O(m (p + 1) c)` scalars. Each query
    /// costs `O(log m + p)` per component. For high degrees the conversion to
    /// local powers can amplify rounding errors.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a local coefficient or prefix integral overflows.
    /// - `AllocationFailure` if the storage cannot be reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_interp::{BSpline, BSplineBasis, ExtrapolationPolicy};
    /// let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 0.0, 2.0, 2.0, 2.0], 2)?;
    /// let square = BSpline::new([basis], vec![0.0, 0.0, 4.0])?; // x^2 on [0, 2]
    /// let prepared = square.prepare_antiderivative()?;
    /// let area = prepared.integral(0.0, 2.0, ExtrapolationPolicy::Reject)?;
    /// assert!((area - 8.0 / 3.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// L. Ramshaw, "Blossoms are polar forms", *Computer Aided Geometric Design*
    /// 6(4), 1989, 323–358.
    pub fn prepare_antiderivative(
        &self,
    ) -> Result<PreparedSplineAntiderivative<S, V::Owned>, SciOxideError> {
        let basis = &self.bases[0];
        let p = basis.degree();
        let n = basis.control_count();
        let knots = basis.knots();
        let dim = self.dimension;
        let terms = p + 1;
        let mut nodes = try_with_capacity(knots.len(), scioxide_core::quantity!("nodes"))?;
        let mut primitives = try_with_capacity(
            checked_mul(
                checked_mul(n, terms, scioxide_core::quantity!("primitives"))?,
                dim,
                scioxide_core::quantity!("primitives"),
            )?,
            scioxide_core::quantity!("primitives"),
        )?;
        let mut scratch = try_filled(
            2 * terms,
            S::ZERO,
            scioxide_core::quantity!("primitive_scratch"),
        )?;
        let (blossom, local) = scratch.split_at_mut(terms);
        for k in p..n {
            if knots[k] == knots[k + 1] {
                continue;
            }
            nodes.push(knots[k]);
            let small = SmallSpan::new(knots, k, p);
            for c in 0..dim {
                self.piece_primitive(k, c, blossom, local, small.as_ref())?;
                primitives.extend_from_slice(local);
            }
        }
        nodes.push(knots[n]);
        let mut endpoints = try_filled(
            checked_mul(2, dim, scioxide_core::quantity!("endpoints"))?,
            S::ZERO,
            scioxide_core::quantity!("endpoints"),
        )?;
        let mut workspace = QueryWorkspace::new(self.scratch)?;
        let domain = basis.domain();
        for (end, x) in [domain.lower(), domain.upper()].into_iter().enumerate() {
            self.derivative_into(
                [x],
                [0],
                None,
                ExtrapolationPolicy::Reject,
                &mut endpoints[end * dim..(end + 1) * dim],
                &mut workspace,
            )?;
        }
        PreparedSplineAntiderivative::new(
            nodes,
            primitives,
            terms,
            endpoints,
            dim,
            zeroed(&self.template)?,
        )
    }

    /// Local primitive of span `k`, component `c`, into `local[..=p]`.
    fn piece_primitive(
        &self,
        k: usize,
        c: usize,
        blossom: &mut [S],
        local: &mut [S],
        small: Option<&SmallSpan<S>>,
    ) -> Result<(), SciOxideError> {
        let basis = &self.bases[0];
        let p = basis.degree();
        let knots = basis.knots();
        let dim = self.dimension;
        let native = small.and_then(|span| {
            let controls = core::array::from_fn(|i| {
                if i <= p {
                    self.coefficients[(k - p + i) * dim + c]
                } else {
                    S::ZERO
                }
            });
            span.bernstein(controls)
        });
        if let Some(controls) = native {
            local.copy_from_slice(&controls[..=p]);
        } else {
            // The jth Bernstein control is the blossom at p - j left endpoints and
            // j right endpoints: local knot refinement of span k.
            for j in 0..=p {
                for (i, slot) in blossom.iter_mut().enumerate() {
                    *slot = self.coefficients[(k - p + i) * dim + c];
                }
                for r in 1..=p {
                    let t = if r <= p - j { knots[k] } else { knots[k + 1] };
                    for i in (r..=p).rev() {
                        let left = k - p + i;
                        let right = left + p + 1 - r;
                        let alpha = if knots[left] == knots[right] {
                            S::ZERO
                        } else {
                            interval_coordinate(t, knots[left], knots[right])?
                        };
                        blossom[i] = blend(blossom[i - 1], blossom[i], alpha)?;
                    }
                }
                local[j] = blossom[p];
            }
        }
        // Bernstein to local powers on s ∈ [0, 1].
        blossom.copy_from_slice(local);
        bernstein_to_power_into(
            blossom,
            local,
            S::ONE,
            sub,
            |value, binomial| mul(binomial, value),
            |binomial, numerator, denominator| {
                mul(
                    binomial,
                    div(
                        S::try_from_usize_exact(numerator)?,
                        S::try_from_usize_exact(denominator)?,
                    )?,
                )
            },
        )?;
        primitive(local)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync, const D: usize> Interpolant<S, D>
    for BSpline<S, V, D>
{
    type Value = V::Owned;
    type Workspace = QueryWorkspace<S>;

    fn domains(&self) -> [Interval<S>; D] {
        core::array::from_fn(|axis| self.bases[axis].domain())
    }
    fn value_dimension(&self) -> usize {
        self.dimension
    }
    fn zero_value(&self) -> Result<V::Owned, SciOxideError> {
        zeroed(&self.template)
    }
    fn workspace(&self) -> Result<QueryWorkspace<S>, SciOxideError> {
        QueryWorkspace::new(self.scratch)
    }
    fn workspace_capacity(&self, workspace: &QueryWorkspace<S>) -> Capacity {
        workspace.capacity(self.scratch)
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.derivative_into(point, [0; D], None, policy, out, workspace)
    }

    fn evaluate(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
    ) -> Result<V::Owned, SciOxideError> {
        self.derivative(point, [0; D], None, policy)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync, const D: usize> Differentiable<S, D>
    for BSpline<S, V, D>
{
    fn gradient_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        let orders: [[usize; D]; D] = core::array::from_fn(unit);
        self.stage(point, policy, &orders, out, workspace)
    }

    fn gradient(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
    ) -> Result<[V::Owned; D], SciOxideError> {
        let mut out = crate::protocol::try_array(|| zeroed(&self.template))?;
        let mut workspace = QueryWorkspace::new(self.query_scratch)?;
        let orders: [[usize; D]; D] = core::array::from_fn(unit);
        self.stage_with_scratch(
            point,
            policy,
            &orders,
            out.as_mut_slice(),
            workspace.scratch(self.query_scratch)?,
        )?;
        Ok(out)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync, const D: usize> TwiceDifferentiable<S, D>
    for BSpline<S, V, D>
{
    fn hessian_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        let orders: [[[usize; D]; D]; D] =
            core::array::from_fn(|row| core::array::from_fn(|column| pair(row, column)));
        self.stage(point, policy, orders.as_flattened(), out, workspace)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync, const D: usize> Integrable<S, D>
    for BSpline<S, V, D>
{
    fn integral_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        lower: [S; D],
        upper: [S; D],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        scioxide_core::check_value_shape(out, self.dimension)?;
        let scratch = workspace.scratch(self.scratch)?;
        let (stage, rest) = scratch.split_at_mut(self.dimension);
        let axes = &mut rest[2 * self.dimension..];
        self.tensor()
            .integral(lower, upper, policy, axes, &mut stage[..])?;
        scioxide_core::write_value(stage, out)
    }
}

/// Order vector of a first partial along `axis`.
fn unit<const D: usize>(axis: usize) -> [usize; D] {
    let mut orders = [0; D];
    orders[axis] = 1;
    orders
}

/// Order vector of the second partial along `row` and `column`.
fn pair<const D: usize>(row: usize, column: usize) -> [usize; D] {
    let mut orders = unit(row);
    orders[column] += 1;
    orders
}

/// Product of the axis control counts.
pub(crate) fn control_count<S: RealScalar, const D: usize>(
    bases: &[BSplineBasis<S>; D],
) -> Result<usize, SciOxideError> {
    if D == 0 {
        return Err(crate::error::out_of_range(
            scioxide_core::argument!("bases"),
            "a B-spline needs at least one axis",
        ));
    }
    bases.iter().try_fold(1usize, |count, basis| {
        checked_mul(
            count,
            basis.control_count(),
            scioxide_core::quantity!("tensor_controls"),
        )
    })
}

/// Query scratch: `stage` output entries, two limit buffers of `width` and the
/// per-axis basis scratch.
pub(crate) fn scratch_len<S: RealScalar, const D: usize>(
    bases: &[BSplineBasis<S>; D],
    stage: usize,
    width: usize,
) -> Result<usize, SciOxideError> {
    let quantity = scioxide_core::quantity!("b_spline_workspace");
    bases.iter().try_fold(
        checked_add(stage, checked_mul(2, width, quantity)?, quantity)?,
        |total, basis| checked_add(total, axis_scratch(basis.degree())?, quantity),
    )
}

/// Ordinary low-degree span geometry, shared by all real value components.
/// Repeated knots have the same zero-denominator convention as the general blossom.
struct SmallSpan<S> {
    degree: usize,
    ratios: [[[S; 4]; 4]; 2],
}

impl<S: RealScalar> SmallSpan<S> {
    fn new(knots: &[S], span: usize, degree: usize) -> Option<Self> {
        if degree > 3 {
            return None;
        }
        let mut ratios = [[[S::ZERO; 4]; 4]; 2];
        for (endpoint, triangle) in ratios.iter_mut().enumerate() {
            let t = knots[span + endpoint];
            for (r, row) in triangle.iter_mut().enumerate().take(degree + 1).skip(1) {
                for (i, ratio) in row.iter_mut().enumerate().take(degree + 1).skip(r) {
                    let left = knots[span - degree + i];
                    let right = knots[span + i + 1 - r];
                    if left == right {
                        continue;
                    }
                    let gap = right - left;
                    let distance = t - left;
                    if !ordinary(gap) || !ordinary(distance) {
                        return None;
                    }
                    let alpha = distance / gap;
                    if !(S::ZERO..=S::ONE).contains(&alpha)
                        || !ordinary(alpha)
                        || (alpha == S::ZERO && distance != S::ZERO)
                    {
                        return None;
                    }
                    *ratio = alpha;
                }
            }
        }
        Some(Self { degree, ratios })
    }

    fn bernstein(&self, controls: [S; 4]) -> Option<[S; 4]> {
        if !controls.iter().all(|&value| ordinary(value)) {
            return None;
        }
        let p = self.degree;
        let mut left = [controls; 4];
        for r in 1..=p {
            left[r] = left[r - 1];
            for i in (r..=p).rev() {
                left[r][i] = native_blend(left[r][i - 1], left[r][i], self.ratios[0][r][i])?;
            }
        }
        let mut result = [S::ZERO; 4];
        result[0] = left[p][p];
        for (j, value) in result.iter_mut().enumerate().take(p + 1).skip(1) {
            let mut row = left[p - j];
            for r in p - j + 1..=p {
                for i in (r..=p).rev() {
                    row[i] = native_blend(row[i - 1], row[i], self.ratios[1][r][i])?;
                }
            }
            *value = row[p];
        }
        Some(result)
    }
}

/// The magnitude check keeps products away from overflow and underflow. Any
/// exceptional difference or cancellation restarts the rescaled blossom.
fn native_blend<S: RealScalar>(left: S, right: S, alpha: S) -> Option<S> {
    if alpha == S::ZERO {
        return Some(left);
    }
    if alpha == S::ONE {
        return Some(right);
    }
    let difference = right - left;
    if !ordinary(difference) {
        return None;
    }
    let value = difference.mul_add(alpha, left);
    ordinary(value).then_some(value)
}

/// Convex blend `(1 - α) left + α right`, rescaled outside `[0, 1]`.
fn blend<S: RealScalar>(left: S, right: S, alpha: S) -> Result<S, SciOxideError> {
    if alpha >= S::ZERO && alpha <= S::ONE {
        scioxide_core::checked_affine(left, right, alpha)
    } else {
        crate::checked::add(left, linear_multiply_difference(alpha, right, left)?)
    }
}

/// Undefined ordinary derivative or outward side.
fn undefined_derivative() -> SciOxideError {
    SciOxideError::domain_error(
        scioxide_core::argument!("point"),
        scioxide_core::DomainReason::Undefined,
        "ordinary derivative or requested domain side is undefined",
    )
}

/// Scalar outputs project after contraction; homogeneous outputs keep the wide
/// exponent range until the rational quotient has been formed.
pub(crate) trait TensorComponent<S: RealScalar>: Copy + PartialEq {
    const ZERO: Self;
    fn add_weighted(
        self,
        sum: &mut PromotingCompensatedSum<S>,
        weight: PromotingProduct<S>,
    ) -> Result<(), SciOxideError>;
    fn finish(sum: PromotingCompensatedSum<S>) -> Result<Self, SciOxideError>;
    fn centered(self, origin: Self, weight: Self) -> Result<Self, SciOxideError>;
    fn magnitude(self) -> Self;
    /// The native scalar, when this component type is the scalar itself.
    fn native(self) -> Option<S>;
    fn from_native(value: S) -> Self;
}

impl<S: RealScalar> TensorComponent<S> for S {
    const ZERO: Self = S::ZERO;
    fn add_weighted(
        self,
        sum: &mut PromotingCompensatedSum<S>,
        weight: PromotingProduct<S>,
    ) -> Result<(), SciOxideError> {
        sum.add_weighted(weight, self)
    }
    fn finish(sum: PromotingCompensatedSum<S>) -> Result<Self, SciOxideError> {
        if let Some(native) = sum.native() {
            native.total()
        } else {
            sum.total()?.try_value()
        }
    }
    fn centered(self, origin: Self, weight: Self) -> Result<Self, SciOxideError> {
        crate::checked::mul(crate::checked::sub(self, origin)?, weight)
    }
    fn magnitude(self) -> Self {
        self.abs()
    }
    fn native(self) -> Option<S> {
        Some(self)
    }
    fn from_native(value: S) -> Self {
        value
    }
}

impl<S: RealScalar> TensorComponent<S> for Scaled<S> {
    const ZERO: Self = Scaled::ZERO;
    fn add_weighted(
        self,
        sum: &mut PromotingCompensatedSum<S>,
        weight: PromotingProduct<S>,
    ) -> Result<(), SciOxideError> {
        sum.add_scaled(weight.scaled()?.mul(self)?)
    }
    fn finish(sum: PromotingCompensatedSum<S>) -> Result<Self, SciOxideError> {
        sum.total()
    }
    fn centered(self, origin: Self, weight: Self) -> Result<Self, SciOxideError> {
        self.sub(origin)?.mul(weight)
    }
    fn magnitude(self) -> Self {
        self.abs()
    }
    fn native(self) -> Option<S> {
        None
    }
    fn from_native(value: S) -> Self {
        Scaled::from_value(value).unwrap_or(Scaled::ZERO)
    }
}

/// Borrowed tensor coefficients evaluated by the local-support kernel.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Tensor<'a, S: RealScalar, const D: usize, C = S> {
    pub(crate) bases: &'a [BSplineBasis<S>; D],
    pub(crate) coefficients: &'a [C],
    /// Real components per control.
    pub(crate) width: usize,
    /// Optional origin for factored rational controls `[c_0, ..., c_n, w]`.
    /// Numerator values and derivatives contract `w_k (c_k - origin)` before
    /// summing. The final component remains W.
    pub(crate) centered: Option<&'a [C]>,
    /// Contract `|weight| * |coefficient|` instead: the absolute mass that
    /// bounds the contraction's first-order rounding error.
    pub(crate) magnitude: bool,
    /// Per-component coefficient magnitude bounds `[smallest nonzero,
    /// largest]` of scalar coefficients, enabling the native contraction.
    pub(crate) magnitudes: Option<&'a [[S; 2]]>,
}

/// Coordinates and sides resolved once per query.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Located<S, const D: usize> {
    effective: [S; D],
    sides: [KnotSide; D],
    /// Ordinary derivatives: verify knot-side limits.
    checked: bool,
    /// Derivatives along this axis vanish (constant Clamp extension).
    zero: [bool; D],
}

impl<S: Copy, const D: usize> Located<S, D> {
    /// Evaluate a selected piece without testing polynomial continuity.
    pub(crate) fn with_sides(self, sides: [KnotSide; D]) -> Self {
        Self {
            sides,
            checked: false,
            ..self
        }
    }
}

impl<S: RealScalar, const D: usize, C: TensorComponent<S>> Tensor<'_, S, D, C> {
    /// Resolve the coordinate policy and the side semantics of each axis.
    pub(crate) fn locate(
        &self,
        point: [S; D],
        sides: Option<[KnotSide; D]>,
        policy: ExtrapolationPolicy<S>,
    ) -> Result<Located<S, D>, SciOxideError> {
        let mut located = Located {
            effective: point,
            sides: sides.unwrap_or([KnotSide::Right; D]),
            checked: sides.is_none(),
            zero: [false; D],
        };
        for axis in 0..D {
            let domain = self.bases[axis].domain();
            let t = resolve_coordinate(&domain, point[axis], policy)?;
            located.effective[axis] = t;
            let outward = sides.is_some_and(|sides| {
                (t == domain.lower() && sides[axis] == KnotSide::Left)
                    || (t == domain.upper() && sides[axis] == KnotSide::Right)
            });
            if outward && policy == ExtrapolationPolicy::Reject {
                return Err(undefined_derivative());
            }
            located.zero[axis] =
                policy == ExtrapolationPolicy::Clamp && (outward || !domain.contains(point[axis]));
        }
        Ok(located)
    }

    /// One mixed partial into `out[..width]`, with knot-side checks for ordinary
    /// derivatives (`limits` holds `2 · width` scratch entries).
    pub(crate) fn partial(
        &self,
        located: &Located<S, D>,
        orders: [usize; D],
        limits: &mut [C],
        axes: &mut [S],
        out: &mut [C],
    ) -> Result<(), SciOxideError> {
        if (0..D).any(|axis| orders[axis] > 0 && located.zero[axis]) {
            out.fill(C::ZERO);
            return Ok(());
        }
        self.evaluate(located.effective, orders, located.sides, axes, out)?;
        if !located.checked {
            return Ok(());
        }
        let (left, right) = limits[..2 * self.width].split_at_mut(self.width);
        for axis in 0..D {
            let basis = &self.bases[axis];
            let t = located.effective[axis];
            let domain = basis.domain();
            if orders[axis] == 0 || t <= domain.lower() || t >= domain.upper() {
                continue;
            }
            let multiplicity = basis.multiplicity(t);
            if multiplicity == 0 {
                continue;
            }
            // A knot of multiplicity m leaves the spline C^{p-m}; higher orders
            // exist only where the represented pieces agree exactly.
            let degree = basis.degree();
            for order in (degree + 1).saturating_sub(multiplicity)..=orders[axis].min(degree) {
                let mut checked = orders;
                checked[axis] = order;
                let mut sides = [KnotSide::Right; D];
                self.evaluate(located.effective, checked, sides, axes, right)?;
                sides[axis] = KnotSide::Left;
                self.evaluate(located.effective, checked, sides, axes, left)?;
                if left != right {
                    return Err(undefined_derivative());
                }
            }
        }
        Ok(())
    }

    /// Contract the active block at resolved coordinates.
    fn evaluate(
        &self,
        effective: [S; D],
        orders: [usize; D],
        sides: [KnotSide; D],
        axes: &mut [S],
        out: &mut [C],
    ) -> Result<(), SciOxideError> {
        let mut first = [0; D];
        let mut offsets = [0; D];
        let mut offset = 0;
        for axis in 0..D {
            let basis = &self.bases[axis];
            let degree = basis.degree();
            if orders[axis] > degree {
                out.fill(C::ZERO);
                return Ok(());
            }
            offsets[axis] = offset;
            first[axis] = basis.local(
                effective[axis],
                orders[axis],
                sides[axis],
                &mut axes[offset..=offset + degree],
            )?;
            offset += degree + 1;
        }
        let rows = &*axes;
        let ranges =
            core::array::from_fn(|axis| first[axis]..first[axis] + self.bases[axis].degree() + 1);
        self.contract(
            ranges,
            |axis, index| Ok(rows[offsets[axis] + index - first[axis]]),
            out,
        )
    }

    /// Integrate over a box with locally computed basis integrals.
    pub(crate) fn integral(
        &self,
        lower: [S; D],
        upper: [S; D],
        policy: ExtrapolationPolicy<S>,
        axes: &mut [S],
        out: &mut [C],
    ) -> Result<(), SciOxideError> {
        let mut prepared = [IntegralAxis::<S>::EMPTY; D];
        let mut offsets = [0; D];
        let mut offset = 0;
        for axis in 0..D {
            let length = axis_scratch(self.bases[axis].degree())?;
            prepared[axis] = IntegralAxis::prepare(
                &self.bases[axis],
                lower[axis],
                upper[axis],
                policy,
                &mut axes[offset..offset + length],
            )?;
            offsets[axis] = offset;
            offset += length;
        }
        let axes = &*axes;
        let ranges = core::array::from_fn(|axis| prepared[axis].range(&self.bases[axis]));
        self.contract(
            ranges,
            |axis, index| prepared[axis].weight(&self.bases[axis], &axes[offsets[axis]..], index),
            out,
        )
    }

    /// Whether native arithmetic provably keeps range for one component, in
    /// the native order (axis weights first, then the coefficient). With
    /// `w_a` the smallest nonzero and `W_a` the absolute sum of axis `a`'s
    /// active weights and `[c, C]` the component's coefficient bounds, every
    /// nonzero partial weight product and term is at least
    /// `4 * MIN_POSITIVE / EPSILON`: none is subnormal, nonzero sums are normal
    /// and cancellation to zero is ordinary rounding. Partial upper bounds
    /// `Π W_a` and `2 * Π W_a * C` must stay finite, excluding overflow.
    fn native_range(bounds: &[[S; 2]; D], magnitude: [S; 2]) -> bool {
        let [smallest, largest] = magnitude;
        if largest == S::ZERO {
            return true;
        }
        let threshold = S::try_from_usize_exact(4).unwrap_or(S::ONE) * S::MIN_POSITIVE / S::EPSILON;
        let (mut lower, mut upper) = (S::ONE, S::ONE);
        for &[w, total] in bounds {
            lower *= w;
            upper *= total;
            if lower < threshold || !upper.is_finite() {
                return false;
            }
        }
        lower * smallest >= threshold && (S::TWO * upper * largest).is_finite()
    }

    /// Native `Σ_k c_k Π_a weight(a, k_a)` for one component, in the
    /// pre-extended-range summation order.
    fn native(
        &self,
        ranges: &[Range<usize>; D],
        strides: &[usize; D],
        weight: &impl Fn(usize, usize) -> Result<S, SciOxideError>,
        component: usize,
    ) -> Result<Option<S>, SciOxideError> {
        let mut sum = S::ZERO;
        let mut index: [usize; D] = core::array::from_fn(|axis| ranges[axis].start);
        let last = D - 1;
        loop {
            let mut prefix = S::ONE;
            let mut base = 0;
            for axis in 0..last {
                prefix *= weight(axis, index[axis])?;
                base += index[axis] * strides[axis];
            }
            for final_index in ranges[last].clone() {
                let product = prefix * weight(last, final_index)?;
                let Some(coefficient) =
                    self.coefficients[(base + final_index) * self.width + component].native()
                else {
                    return Ok(None);
                };
                sum += product * coefficient;
            }
            let mut axis = last;
            loop {
                if axis == 0 {
                    return Ok(Some(sum));
                }
                axis -= 1;
                index[axis] += 1;
                if index[axis] < ranges[axis].end {
                    break;
                }
                index[axis] = ranges[axis].start;
            }
        }
    }

    /// Basis weights enter magnitude contractions by absolute value.
    fn weight_term(&self, weight: S) -> S {
        if self.magnitude { weight.abs() } else { weight }
    }

    /// `out = Σ_k c_k Π_a weight(a, k_a)` over the index box `ranges`.
    fn contract(
        &self,
        ranges: [Range<usize>; D],
        weight: impl Fn(usize, usize) -> Result<S, SciOxideError>,
        out: &mut [C],
    ) -> Result<(), SciOxideError> {
        let out = &mut out[..self.width];
        out.fill(C::ZERO);
        if ranges.iter().any(Range::is_empty) {
            return Ok(());
        }
        let mut strides = [1usize; D];
        for axis in (0..D - 1).rev() {
            strides[axis] = strides[axis + 1] * self.bases[axis + 1].control_count();
        }
        let magnitudes = self
            .magnitudes
            .filter(|_| C::ZERO.native().is_some() && self.centered.is_none() && !self.magnitude);
        let mut bounds = [[S::ZERO; 2]; D];
        if magnitudes.is_some() {
            for (axis, bound) in bounds.iter_mut().enumerate() {
                let mut smallest = S::MAX;
                let mut total = S::ZERO;
                for index in ranges[axis].clone() {
                    let w = weight(axis, index)?.abs();
                    if w != S::ZERO {
                        smallest = smallest.min(w);
                    }
                    total += w;
                }
                *bound = [smallest, total];
            }
        }
        for (component, slot) in out.iter_mut().enumerate() {
            if let Some(magnitudes) = magnitudes
                && Self::native_range(&bounds, magnitudes[component])
                && let Some(value) = self.native(&ranges, &strides, &weight, component)?
            {
                *slot = C::from_native(value);
                continue;
            }
            let mut sum = PromotingCompensatedSum::ZERO;
            let mut index: [usize; D] = core::array::from_fn(|axis| ranges[axis].start);
            let last = D - 1;
            'rows: loop {
                let mut prefix = PromotingProduct::ONE;
                let mut base = 0;
                for axis in 0..last {
                    prefix = prefix.multiplied(self.weight_term(weight(axis, index[axis])?))?;
                    base += index[axis] * strides[axis];
                }
                for final_index in ranges[last].clone() {
                    let product =
                        prefix.multiplied(self.weight_term(weight(last, final_index)?))?;
                    let linear = base + final_index;
                    let coefficient = if let Some(origin) = self.centered
                        && component < origin.len()
                    {
                        self.coefficients[linear * self.width + component].centered(
                            origin[component],
                            self.coefficients[linear * self.width + self.width - 1],
                        )?
                    } else {
                        self.coefficients[linear * self.width + component]
                    };
                    let coefficient = if self.magnitude {
                        coefficient.magnitude()
                    } else {
                        coefficient
                    };
                    coefficient.add_weighted(&mut sum, product)?;
                }
                let mut axis = last;
                loop {
                    if axis == 0 {
                        *slot = C::finish(sum)?;
                        break 'rows;
                    }
                    axis -= 1;
                    index[axis] += 1;
                    if index[axis] < ranges[axis].end {
                        break;
                    }
                    index[axis] = ranges[axis].start;
                }
            }
        }
        Ok(())
    }
}

impl<S: RealScalar, const D: usize> Tensor<'_, S, D, Scaled<S>> {
    /// Largest absolute local homogeneous basis weight. Its control is a fixed
    /// origin for the entire quotient jet, so small endpoint corrections are
    /// never recovered by subtracting separately rounded numerator derivatives.
    pub(crate) fn reference(
        &self,
        located: &Located<S, D>,
        axes: &mut [S],
    ) -> Result<usize, SciOxideError> {
        let mut first = [0; D];
        let mut offsets = [0; D];
        let mut offset = 0;
        for axis in 0..D {
            let degree = self.bases[axis].degree();
            offsets[axis] = offset;
            first[axis] = self.bases[axis].local(
                located.effective[axis],
                0,
                located.sides[axis],
                &mut axes[offset..=offset + degree],
            )?;
            offset += degree + 1;
        }
        let mut index = first;
        let mut maximum = Scaled::ZERO;
        let mut reference = 0;
        loop {
            let mut product = Scaled::ONE;
            let mut linear = 0;
            for axis in 0..D {
                product = product.mul(Scaled::from_value(
                    axes[offsets[axis] + index[axis] - first[axis]],
                )?)?;
                linear = linear * self.bases[axis].control_count() + index[axis];
            }
            let magnitude = product
                .mul(self.coefficients[linear * self.width + self.width - 1])?
                .abs();
            if magnitude.abs_cmp(maximum).is_gt() {
                maximum = magnitude;
                reference = linear;
            }
            let mut axis = D;
            loop {
                if axis == 0 {
                    return Ok(reference);
                }
                axis -= 1;
                index[axis] += 1;
                if index[axis] <= first[axis] + self.bases[axis].degree() {
                    break;
                }
                index[axis] = first[axis];
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::Complex;

    fn basis(knots: &[f64], degree: usize) -> BSplineBasis<f64> {
        BSplineBasis::new(knots.to_vec(), degree).unwrap()
    }

    fn uniform_basis(controls: usize, degree: usize) -> BSplineBasis<f64> {
        let spans = controls - degree;
        let knots = core::iter::repeat_n(0.0, degree + 1)
            .chain((1..spans).map(|k| k as f64))
            .chain(core::iter::repeat_n(spans as f64, degree + 1))
            .collect();
        BSplineBasis::new(knots, degree).unwrap()
    }

    #[test]
    fn point_results_and_scratch_are_independent_of_control_count() {
        let policy = ExtrapolationPolicy::Reject;
        for degree in 0..=5 {
            let small = BSpline::new([uniform_basis(1024, degree)], vec![1.0; 1024]).unwrap();
            let large = BSpline::new([uniform_basis(16384, degree)], vec![1.0; 16384]).unwrap();
            assert_eq!(small.scratch, large.scratch);
            assert_eq!(small.query_scratch, large.query_scratch);
            for order in 0..=degree {
                for spline in [&small, &large] {
                    let value = spline.derivative([3.25], [order], None, policy).unwrap();
                    assert!((value - f64::from(order == 0)).abs() < 1e-12);
                }
            }
        }
        for controls in [32, 128] {
            let basis = uniform_basis(controls, 3);
            let surface =
                BSpline::new([basis.clone(), basis], vec![1.0; controls * controls]).unwrap();
            for orders in [[0, 0], [1, 0], [1, 1]] {
                let value = surface
                    .derivative([3.25, 4.75], orders, None, policy)
                    .unwrap();
                crate::test_support::assert_close(
                    value,
                    f64::from(orders == [0, 0]),
                    128. * f64::EPSILON,
                    0.,
                );
            }
            let basis = uniform_basis(controls, 3);
            let rational =
                super::super::nurbs::Nurbs::new([basis], vec![1.0; controls], vec![1.0; controls])
                    .unwrap();
            crate::test_support::assert_close(
                rational.evaluate([3.25], policy).unwrap(),
                1.,
                128. * f64::EPSILON,
                0.,
            );
        }
    }

    #[test]
    fn extreme_opposite_sign_knots_use_normalized_coordinates() {
        let spline = BSpline::new(
            [basis(&[-f64::MAX, -f64::MAX, f64::MAX, f64::MAX], 1)],
            vec![2.0, 2.0],
        )
        .unwrap();
        let policy = ExtrapolationPolicy::Reject;
        assert_eq!(spline.evaluate([0.0], policy).unwrap(), 2.0);
        assert_eq!(spline.derivative([0.0], [1], None, policy).unwrap(), 0.0);
        assert_eq!(spline.gradient([0.0], policy).unwrap(), [0.0]);
    }

    #[test]
    fn tensor_values_partials_and_box_integrals() {
        let linear = basis(&[0., 0., 1., 1.], 1);
        // f(x, y) = 1 + 3x + 2y.
        let spline = BSpline::new([linear.clone(), linear], vec![1.0, 3.0, 4.0, 6.0]).unwrap();
        let policy = ExtrapolationPolicy::Reject;
        assert!((spline.evaluate([0.25, 0.5], policy).unwrap() - 2.75).abs() < 1e-15);
        assert_eq!(spline.gradient([0.25, 0.5], policy).unwrap(), [3.0, 2.0]);
        assert_eq!(spline.hessian([0.25, 0.5], policy).unwrap(), [[0.0; 2]; 2]);
        let integral = spline.integral([0.0, 0.0], [1.0, 1.0], policy).unwrap();
        assert!((integral - 3.5).abs() < 1e-15);
        // Clamp integrates constant extensions on both axes.
        let clamped = spline
            .integral([-1.0, 0.0], [1.0, 1.0], ExtrapolationPolicy::Clamp)
            .unwrap();
        assert!((clamped - (3.5 + 2.0)).abs() < 1e-14);
    }

    #[test]
    fn values_are_componentwise() {
        let quadratic = basis(&[0., 0., 0., 0.5, 1., 1., 1.], 2);
        let complex = BSpline::new(
            [quadratic.clone()],
            vec![
                Complex::new(1.0, -1.0),
                Complex::new(3.0, 1.0),
                Complex::new(-2.0, 0.5),
                Complex::new(0.25, 4.0),
            ],
        )
        .unwrap();
        let real = BSpline::new([quadratic.clone()], vec![1.0, 3.0, -2.0, 0.25]).unwrap();
        let imaginary = BSpline::new([quadratic], vec![-1.0, 1.0, 0.5, 4.0]).unwrap();
        let policy = ExtrapolationPolicy::MethodDefined;
        for x in [-0.25, 0.1, 0.5, 0.77, 1.0, 1.3] {
            let value = complex.evaluate([x], policy).unwrap();
            assert_eq!(value.re, real.evaluate([x], policy).unwrap());
            assert_eq!(value.im, imaginary.evaluate([x], policy).unwrap());
            let slope = complex.gradient([x], policy).unwrap()[0];
            assert_eq!(slope.re, real.gradient([x], policy).unwrap()[0]);
            assert_eq!(slope.im, imaginary.gradient([x], policy).unwrap()[0]);
        }
    }

    #[test]
    fn knot_side_semantics_follow_multiplicity() {
        let reject = ExtrapolationPolicy::Reject;
        let kink = BSpline::new([basis(&[0., 0., 1., 2., 2.], 1)], vec![0., 1., 0.]).unwrap();
        assert!(kink.derivative([1.], [1], None, reject).is_err());
        assert_eq!(kink.derivative([1.], [0], None, reject).unwrap(), 1.);
        assert!(
            kink.derivative([0.], [1], Some([KnotSide::Left]), reject)
                .is_err()
        );
        assert_eq!(
            kink.derivative(
                [0.],
                [1],
                Some([KnotSide::Left]),
                ExtrapolationPolicy::Clamp
            )
            .unwrap(),
            0.
        );
        // A cubic with a simple knot is C2: only the third derivative jumps.
        let cubic = BSpline::new(
            [basis(&[0., 0., 0., 0., 1., 2., 2., 2., 2.], 3)],
            vec![0., 1., -1., 2., 0.5],
        )
        .unwrap();
        for order in 0..3 {
            cubic.derivative([1.], [order], None, reject).unwrap();
        }
        assert!(cubic.derivative([1.], [3], None, reject).is_err());
        assert_eq!(
            cubic
                .derivative([1.], [4], Some([KnotSide::Left]), reject)
                .unwrap(),
            0.
        );
    }

    #[test]
    fn linear_constructor_interpolates() {
        let line = BSpline::linear(
            vec![0.0_f64, 1.0, 3.0],
            vec![[1.0, 0.0], [3.0, 1.0], [-1.0, 2.0]],
        )
        .unwrap();
        let policy = ExtrapolationPolicy::Reject;
        assert_eq!(line.evaluate([2.0], policy).unwrap(), [1.0, 1.5]);
        assert_eq!(line.gradient([2.0], policy).unwrap(), [[-2.0, 0.5]]);
        assert!(BSpline::linear(vec![0.0, 0.0], vec![1.0, 2.0]).is_err());
        assert!(BSpline::linear(vec![0.0, 1.0], vec![1.0]).is_err());
    }

    #[test]
    fn undersized_workspace_is_a_dimension_mismatch() {
        let big = BSpline::new(
            [basis(&[0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.], 5)],
            vec![[1.0; 4]; 6],
        )
        .unwrap();
        let small = BSpline::new([basis(&[0., 1.], 0)], vec![1.0]).unwrap();
        let mut workspace = small.workspace().unwrap();
        assert_eq!(big.workspace_capacity(&workspace), Capacity::NeedsGrowth);
        let mut out = [7.0; 4];
        let error = big
            .evaluate_into([0.5], ExtrapolationPolicy::Reject, &mut out, &mut workspace)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(out, [7.0; 4]);
    }
    #[test]
    fn small_span_primitives_match_general_blossoms() {
        for degree in 0..=3 {
            for multiplicity in 1..=degree + 1 {
                let knots: Vec<f64> = core::iter::repeat_n(0.0, degree + 1)
                    .chain(core::iter::repeat_n(0.125, multiplicity))
                    .chain([0.75])
                    .chain(core::iter::repeat_n(1.0, degree + 1))
                    .collect();
                let count = knots.len() - degree - 1;
                let controls: Vec<_> = (0..count)
                    .map(|i| [(i as f64 * 1.7).sin(), (i as f64 * 0.9).cos()])
                    .collect();
                let spline =
                    BSpline::new([BSplineBasis::new(knots, degree).unwrap()], controls).unwrap();
                let knots = spline.bases()[0].knots();
                for span in degree..count {
                    if knots[span] == knots[span + 1] {
                        continue;
                    }
                    let small = SmallSpan::new(knots, span, degree).unwrap();
                    for component in 0..2 {
                        let mut scratch = [0.0; 4];
                        let mut native = [0.0; 4];
                        let mut general = [0.0; 4];
                        spline
                            .piece_primitive(
                                span,
                                component,
                                &mut scratch[..=degree],
                                &mut native[..=degree],
                                Some(&small),
                            )
                            .unwrap();
                        spline
                            .piece_primitive(
                                span,
                                component,
                                &mut scratch[..=degree],
                                &mut general[..=degree],
                                None,
                            )
                            .unwrap();
                        for (actual, expected) in native.iter().zip(general) {
                            assert!(
                                (actual - expected).abs()
                                    <= 128.0 * f64::EPSILON * expected.abs().max(1.0)
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn small_span_admission_retains_extreme_range_fallback() {
        assert!(SmallSpan::new(&[-f64::MAX, -f64::MAX, f64::MAX, f64::MAX], 1, 1).is_none());
        assert!(SmallSpan::new(&[0.0, 0.0, f64::from_bits(1), f64::from_bits(1)], 1, 1).is_none());
        let span = SmallSpan::new(&[0.0, 0.0, 1.0, 1.0], 1, 1).unwrap();
        assert!(span.bernstein([f64::MAX, -f64::MAX, 0.0, 0.0]).is_none());
        assert!(span.bernstein([f64::from_bits(1), 0.0, 0.0, 0.0]).is_none());
    }
}
