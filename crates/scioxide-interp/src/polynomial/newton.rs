//! Polynomial interpolation in Newton form with divided differences.

use crate::checked::ordinary_nodes;
use crate::polynomial::antiderivative::{PreparedPolynomialAntiderivative, prepare_from_newton};
use crate::protocol::{
    Differentiable, ExtrapolationPolicy, Interpolant, NodeSeparation, SampleOptions, clamped_axes,
    domain_from_nodes, resolve_coordinate,
};
use crate::validate::validate_nodes;
use crate::values::{QueryWorkspace, flatten, zeroed};
use alloc::vec::Vec;
use scioxide_core::Scaled;
use scioxide_core::backend::{try_copy_slice, try_reserve};
use scioxide_core::{
    AffineValue, AffineValueMut, Capacity, Interval, RealScalar, SciOxideError, check_value_shape,
    validate_value,
};
use scioxide_poly::backend::polynomial_basis::{
    linear_difference_ratio, linear_multiply_difference,
};

/// Interpolates `n` samples with the polynomial of degree at most `n − 1`, in Newton form.
///
/// For nodes `x₀ < x₁ < … < xₙ₋₁` the interpolant is
///
/// ```text
/// p(x) = Σₖ f[x₀, …, xₖ] Πⱼ₌₀^(k−1) (x − xⱼ),
/// ```
///
/// where `f[x₀, …, xₖ]` are the divided differences of the data. Each real
/// component of the values has its own coefficients; a complex value has a real
/// and an imaginary component. With [`ExtrapolationPolicy::MethodDefined`]
/// the polynomial is evaluated outside `[x₀, xₙ₋₁]`.
///
/// Construction costs O(n² d) operations for `d` real components, and a value
/// or derivative query costs O(n d) by Horner's rule on the nested form.
/// [`Self::insert_point`] adds a node in O(n d). Horner's rule switches to an
/// extended exponent range if an intermediate would overflow or underflow.
/// High-degree interpolation can be ill-conditioned; for many nodes prefer
/// [`crate::BarycentricInterpolator`] on Chebyshev nodes. Definite integrals
/// come from [`Self::prepare_antiderivative`].
///
/// # Errors
///
/// Queries fail as described in the [crate docs](crate#errors), and with
/// `NonFinite` or `NumericalInstability` if the result overflows.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_interp::{ExtrapolationPolicy, Interpolant, NewtonInterpolator};
/// let polynomial = NewtonInterpolator::new(
///     vec![0.0_f64, 1.0],
///     vec![Complex::new(1.0, -1.0), Complex::new(3.0, 1.0)],
/// )?;
/// let mut workspace = polynomial.workspace()?;
/// let mut output = [0.0; 2];
/// polynomial.evaluate_into(
///     [0.5],
///     ExtrapolationPolicy::Reject,
///     &mut output,
///     &mut workspace,
/// )?;
/// assert_eq!(output, [2.0, 0.0]);
/// let extended = polynomial.insert_point(2.0, Complex::new(9.0, 3.0))?;
/// assert_eq!(extended.nodes(), &[0.0, 1.0, 2.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. Stoer and R. Bulirsch, *Introduction to Numerical Analysis*, 3rd ed.,
/// Springer, 2002, §2.1.
#[derive(Debug, Clone)]
#[must_use]
pub struct NewtonInterpolator<S: RealScalar, V: AffineValue<Scale = S> = S> {
    x: Vec<S>,
    coeffs: Vec<S>,
    template: V,
    dimension: usize,
    domain: Interval<S>,
    separation: NodeSeparation<S>,
    native: Vec<bool>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> NewtonInterpolator<S, V> {
    /// Builds the interpolant through the samples `(x[j], y[j])`.
    ///
    /// This is [`Self::with_options`] with [`SampleOptions::new`], so the nodes
    /// `x` must be strictly increasing.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `x` is empty or not strictly increasing.
    /// - `DimensionMismatch` if `x.len() != y.len()` or the values differ in shape.
    /// - `NonFinite` if a node or value is NaN or infinite.
    /// - `NonFinite` or `NumericalInstability` if a divided difference overflows.
    /// - `AllocationFailure` if the coefficients cannot be stored.
    pub fn new(x: Vec<S>, y: Vec<V>) -> Result<Self, SciOxideError> {
        Self::with_options(x, y, SampleOptions::new())
    }

    /// Builds the interpolant through the samples `(x[j], y[j])`, validating the
    /// nodes under `options`.
    ///
    /// The nodes `x` must be strictly increasing and distinct under
    /// [`SampleOptions::node_separation`]. [`Self::insert_point`] checks the
    /// new node under the same separation.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `x` is empty, or not strictly increasing and distinct
    ///   under the separation of `options`.
    /// - `DimensionMismatch` if `x.len() != y.len()` or the values differ in shape.
    /// - `NonFinite` if a node or value is NaN or infinite.
    /// - `NonFinite` or `NumericalInstability` if a divided difference overflows.
    /// - `AllocationFailure` if the coefficients cannot be stored.
    pub fn with_options(
        x: Vec<S>,
        y: Vec<V>,
        options: SampleOptions<S>,
    ) -> Result<Self, SciOxideError> {
        let separation = options.node_separation();
        validate_nodes(&x, 1, separation, scioxide_core::argument!("x"))?;
        let flat = flatten(y, x.len())?;
        let dimension = flat.dimension;
        let mut coeffs = flat.components;
        scioxide_poly::backend::polynomial_basis::real_divided_differences_into(
            &x,
            dimension,
            &mut coeffs,
        )?;
        let native = native_components(&x, ordinary_nodes(&x), &coeffs, dimension)?;
        Ok(Self {
            native,
            domain: domain_from_nodes(&x)?,
            x,
            coeffs,
            template: flat.template,
            dimension,
            separation,
        })
    }

    /// Returns a new interpolant with the sample `(x, y)` added to the right of the last node.
    ///
    /// `x` must be greater than every existing node and distinct from the last
    /// one under the interpolant's [`NodeSeparation`]. The existing divided
    /// differences stay valid, so only `f[x₀, …, xₙ₋₁, x]` is computed, in
    /// O(n d) operations. `self` is unchanged.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `x` or `y` is NaN or infinite.
    /// - `InvalidInput` if `x` is not greater than the last node.
    /// - `DimensionMismatch` if `y` has a different shape from the existing values.
    /// - `NonFinite` or `NumericalInstability` if the new divided difference overflows.
    /// - `AllocationFailure` if the new coefficients cannot be stored.
    pub fn insert_point(&self, x: S, y: V) -> Result<Self, SciOxideError> {
        validate_value(&x)?;
        check_value_shape(&y, self.dimension)?;
        validate_value(&y)?;
        let last = self.domain.upper();
        validate_nodes(
            &[last, x],
            2,
            self.separation,
            scioxide_core::argument!("x"),
        )?;
        // The new coefficient follows the recurrence
        // v <- (v - f[x_0..x_k]) / (x - x_k) for k = 0..n, starting from y.
        let mut coeffs = try_copy_slice(
            &self.coeffs,
            scioxide_core::quantity!("newton_coefficients"),
        )?;
        try_reserve(
            &mut coeffs,
            self.dimension,
            scioxide_core::quantity!("newton_coefficients"),
        )?;
        for component in 0..self.dimension {
            let mut value = y.component(component)?;
            for (index, node) in self.x.iter().enumerate() {
                value = linear_difference_ratio(
                    value,
                    self.coeffs[index * self.dimension + component],
                    x,
                    *node,
                )?;
            }
            coeffs.push(value);
        }
        let mut nodes = try_copy_slice(&self.x, scioxide_core::quantity!("newton_nodes"))?;
        try_reserve(&mut nodes, 1, scioxide_core::quantity!("newton_nodes"))?;
        nodes.push(x);
        let native = native_components(&nodes, ordinary_nodes(&nodes), &coeffs, self.dimension)?;
        Ok(Self {
            native,
            domain: domain_from_nodes(&nodes)?,
            x: nodes,
            coeffs,
            template: y,
            dimension: self.dimension,
            separation: self.separation,
        })
    }

    /// Returns the divided differences `f[x₀, …, xₖ]`, real components node by node (`[k][component]`).
    pub fn coefficients(&self) -> &[S] {
        &self.coeffs
    }

    /// Returns the nodes in increasing order.
    pub fn nodes(&self) -> &[S] {
        &self.x
    }

    /// Computes the antiderivative of the interpolant, for definite integrals.
    ///
    /// See [`PreparedPolynomialAntiderivative`]. Preparation costs O(n² d)
    /// operations, and each integral O(n d).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the coefficients cannot be stored.
    /// - `NumericalInstability` if a coefficient is not representable.
    pub fn prepare_antiderivative(
        &self,
    ) -> Result<PreparedPolynomialAntiderivative<S, V::Owned>, SciOxideError> {
        prepare_from_newton(
            &self.x,
            &self.coeffs,
            self.dimension,
            &self.template,
            self.domain,
        )
    }

    /// Computes values or derivatives of every component and writes them to `out`.
    fn stage<const DERIVATIVE: bool, O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        check_value_shape(out, self.dimension)?;
        let staged = workspace.scratch(self.dimension)?;
        let [clamped] = clamped_axes(policy, [self.domain], point);
        let t = resolve_coordinate(&self.domain, point[0], policy)?;
        let calculate = |component| {
            if DERIVATIVE && clamped {
                Ok(S::ZERO)
            } else {
                evaluate_newton::<S, DERIVATIVE>(
                    &self.x,
                    &self.coeffs,
                    self.dimension,
                    component,
                    t,
                    self.native[component] && self.domain.contains(t),
                )
            }
        };
        if self.dimension == 1 {
            let value = calculate(0)?;
            out.for_each_component_mut(|slot| *slot = value);
            return Ok(());
        }
        for (component, slot) in staged.iter_mut().enumerate() {
            *slot = calculate(component)?;
        }
        crate::values::commit_finite(staged, out);
        Ok(())
    }
}

/// One value or derivative component of a node-major Newton form at `t`.
///
/// With `native` set (a component whose bounds fit, at an interior query), plain
/// Horner arithmetic is used. Otherwise checked Horner runs first, and any loss
/// of range retries with `Scaled` exponents, rounding only at the end.
pub(super) fn evaluate_newton<S: RealScalar, const DERIVATIVE: bool>(
    nodes: &[S],
    coefficients: &[S],
    dimension: usize,
    component: usize,
    t: S,
    native: bool,
) -> Result<S, SciOxideError> {
    use scioxide_poly::backend::polynomial_basis::{
        newton_value_and_derivative_from_iter, newton_value_from_iter,
    };
    let last = nodes.len() - 1;
    let terms = (0..last).map(|index| (coefficients[index * dimension + component], nodes[index]));
    let leading = coefficients[last * dimension + component];
    if native {
        let (value, slope) =
            terms
                .clone()
                .rev()
                .fold((leading, S::ZERO), |(value, slope), (c, node)| {
                    let difference = t - node;
                    (
                        value * difference + c,
                        if DERIVATIVE {
                            slope * difference + value
                        } else {
                            S::ZERO
                        },
                    )
                });
        return crate::checked::output(if DERIVATIVE { slope } else { value });
    }
    if !DERIVATIVE {
        if let Ok(value) = newton_value_from_iter(
            terms.clone().map(Ok),
            leading,
            t,
            linear_multiply_difference,
            crate::checked::add,
        ) {
            return crate::checked::output(value);
        }
        return newton_value_from_iter(
            terms.map(|(c, node)| Ok((Scaled::from_value(c)?, Scaled::from_value(node)?))),
            Scaled::from_value(leading)?,
            Scaled::from_value(t)?,
            |v, x, node| v.mul(x.sub(node)?),
            |a, b| a.add(b),
        )?
        .round();
    }
    // Fast path: plain checked Horner in the working precision. If any
    // intermediate overflows (large coefficients times large `t - x_k`
    // products can do so even when the result is representable), retry
    // with `Scaled` arithmetic, which carries an explicit exponent and only
    // rounds back to `S` at the end. Errors from the fast path are
    // deliberately discarded because the slow path re-derives them.
    if let Ok((value, slope)) = newton_value_and_derivative_from_iter(
        terms.clone().map(Ok),
        leading,
        t,
        S::ZERO,
        linear_multiply_difference,
        |left, right| crate::checked::add(left, right),
    ) {
        return Ok(if DERIVATIVE { slope } else { value });
    }
    let (value, slope) = newton_value_and_derivative_from_iter(
        terms.map(|(coefficient, node)| {
            Ok((Scaled::from_value(coefficient)?, Scaled::from_value(node)?))
        }),
        Scaled::from_value(leading)?,
        Scaled::from_value(t)?,
        Scaled::ZERO,
        |value, point, node| value.mul(point.sub(node)?),
        |left, right| left.add(right),
    )?;
    if DERIVATIVE {
        slope.round()
    } else {
        value.round()
    }
}

/// Absolute Horner bounds on every value and derivative intermediate on the
/// domain decide, per component, whether plain arithmetic is safe.
///
/// `coordinates` says whether the node geometry is safe. The bounds hold for
/// repeated (confluent) nodes too, since every factor is bounded by the node span.
pub(super) fn native_components<S: RealScalar>(
    nodes: &[S],
    coordinates: bool,
    coefficients: &[S],
    dimension: usize,
) -> Result<Vec<bool>, SciOxideError> {
    let mut result = scioxide_core::backend::try_with_capacity(
        dimension,
        scioxide_core::quantity!("newton_ranges"),
    )?;
    let span = nodes[nodes.len() - 1] - nodes[0];
    for c in 0..dimension {
        let mut value = S::ZERO;
        let mut slope = S::ZERO;
        let mut admitted = coordinates;
        for &coefficient in coefficients.iter().skip(c).step_by(dimension).rev() {
            slope = slope * span + value;
            value = value * span + coefficient.abs();
            admitted &= value.is_finite()
                && slope.is_finite()
                && value <= S::MAX / S::TWO
                && slope <= S::MAX / S::TWO;
        }
        result.push(admitted);
    }
    Ok(result)
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Interpolant<S, 1>
    for NewtonInterpolator<S, V>
{
    type Value = V::Owned;
    type Workspace = QueryWorkspace<S>;

    fn domains(&self) -> [Interval<S>; 1] {
        [self.domain]
    }
    fn value_dimension(&self) -> usize {
        self.dimension
    }
    fn zero_value(&self) -> Result<V::Owned, SciOxideError> {
        let zero = zeroed(&self.template)?;
        check_value_shape(&zero, self.dimension)?;
        Ok(zero)
    }
    fn workspace(&self) -> Result<QueryWorkspace<S>, SciOxideError> {
        QueryWorkspace::new(self.dimension)
    }
    fn workspace_capacity(&self, workspace: &QueryWorkspace<S>) -> Capacity {
        workspace.capacity(self.dimension)
    }
    fn evaluate_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.stage::<false, _>(point, policy, out, workspace)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Differentiable<S, 1>
    for NewtonInterpolator<S, V>
{
    fn gradient_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.stage::<true, _>(point, policy, out, workspace)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, assert_error};
    use scioxide_core::{Complex, NonFiniteOrigin};

    #[test]
    fn value_only_does_not_require_a_representable_derivative() {
        let model =
            NewtonInterpolator::new(vec![0.0_f64, 0.5, 1.0], vec![0.0, 0.0, f64::MAX / 2.0])
                .unwrap();
        assert_eq!(
            model.evaluate([1.0], ExtrapolationPolicy::Reject).unwrap(),
            f64::MAX / 2.0
        );
        let mut ws = model.workspace().unwrap();
        let mut out = [123.0];
        assert!(
            model
                .gradient_into([1.0], ExtrapolationPolicy::Reject, &mut out, &mut ws)
                .is_err()
        );
        assert_eq!(out, [123.0]);
    }

    #[test]
    fn admitted_and_fallback_newton_queries_agree() {
        let native =
            NewtonInterpolator::new(vec![-1.0_f64, 0.0, 0.5, 2.0], vec![1.0, -2.0, 0.25, 3.0])
                .unwrap();
        assert!(native.native[0]);
        let mut fallback = native.clone();
        fallback.native.fill(false);
        for t in [-1.0, -0.7, 0.25, 1.9, 2.0] {
            assert_close(
                native.evaluate([t], ExtrapolationPolicy::Reject).unwrap(),
                fallback.evaluate([t], ExtrapolationPolicy::Reject).unwrap(),
                1e-14,
                1e-14,
            );
            assert_close(
                native.gradient([t], ExtrapolationPolicy::Reject).unwrap()[0],
                fallback.gradient([t], ExtrapolationPolicy::Reject).unwrap()[0],
                1e-14,
                1e-14,
            );
        }
        let extended = native.insert_point(3.0, 7.0).unwrap();
        assert!(extended.native[0]);
        assert_close(
            extended
                .evaluate([3.0], ExtrapolationPolicy::Reject)
                .unwrap(),
            7.0,
            1e-13,
            1e-13,
        );
    }

    #[test]
    fn test_new_valid() {
        let interp = NewtonInterpolator::new(vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 5.0]);
        assert!(interp.is_ok());
    }

    #[test]
    fn test_interpolate_at_nodes() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 2.0, 5.0];
        let interp = NewtonInterpolator::new(x.clone(), y.clone()).unwrap();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let v = interp.evaluate([*xi], ExtrapolationPolicy::Reject).unwrap();
            assert_close(v, *yi, 5e-14, 5e-14);
        }
    }

    #[test]
    fn test_insert_point() {
        // Start with 2 points, add a third
        let interp = NewtonInterpolator::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        let extended = interp.insert_point(2.0, 4.0).unwrap();
        // y = x^2 through (0,0), (1,1), (2,4)
        let v = extended
            .evaluate([1.5], ExtrapolationPolicy::Reject)
            .unwrap();
        assert_close(v, 2.25, 5e-14, 5e-14);
        assert_eq!(interp.nodes(), &[0.0, 1.0]);
        let rebuilt = NewtonInterpolator::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 4.0]).unwrap();
        assert_eq!(extended.coefficients(), rebuilt.coefficients());
    }

    #[test]
    fn test_insert_point_non_monotone() {
        let interp = NewtonInterpolator::new(vec![0.0, 2.0], vec![0.0, 1.0]).unwrap();
        let result = interp.insert_point(1.0, 0.5);
        assert_error(&result, scioxide_core::SciOxideErrorKind::InvalidInput);
    }

    #[test]
    fn failed_insert_point_reports_intermediate_overflow() {
        let interp = NewtonInterpolator::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        let result = interp.insert_point(1.0 + f64::EPSILON, f64::MAX);
        assert!(matches!(
            assert_error(&result, scioxide_core::SciOxideErrorKind::NonFinite).view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Intermediate,
                ..
            }
        ));
        assert_eq!(interp.nodes(), &[0.0, 1.0]);
    }

    #[test]
    fn test_derivative_quadratic() {
        // y = x^2 => y' = 2x
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let interp = NewtonInterpolator::new(x, y).unwrap();
        let [d] = interp.gradient([1.5], ExtrapolationPolicy::Reject).unwrap();
        assert_close(d, 3.0, 5e-11, 5e-11);
    }

    #[test]
    fn test_integral_linear() {
        // y = x => integral from 0 to 2 = 2
        let interp = NewtonInterpolator::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]).unwrap();
        let i = interp
            .prepare_antiderivative()
            .unwrap()
            .integral(0.0, 2.0, ExtrapolationPolicy::Reject)
            .unwrap();
        assert_close(i, 2.0, 5e-13, 5e-13);
    }

    #[test]
    fn affine_components_match_scalar_interpolants_bit_for_bit() {
        let x = vec![-1.0_f32, 0.0, 0.5, 2.0];
        let re = vec![1.0_f32, -2.0, 0.25, 3.0];
        let im = vec![0.5_f32, 4.0, -1.0, 2.0];
        let values: Vec<Complex<f32>> = re
            .iter()
            .zip(&im)
            .map(|(&a, &b)| Complex::new(a, b))
            .collect();
        let complex = NewtonInterpolator::new(x.clone(), values).unwrap();
        let real = NewtonInterpolator::new(x.clone(), re).unwrap();
        let imaginary = NewtonInterpolator::new(x, im).unwrap();
        for query in [-1.0_f32, 0.3, 1.7, 3.0] {
            let policy = ExtrapolationPolicy::MethodDefined;
            let value = complex.evaluate([query], policy).unwrap();
            assert_eq!(value.re, real.evaluate([query], policy).unwrap());
            assert_eq!(value.im, imaginary.evaluate([query], policy).unwrap());
            let [slope] = complex.gradient([query], policy).unwrap();
            assert_eq!(slope.re, real.gradient([query], policy).unwrap()[0]);
            assert_eq!(slope.im, imaginary.gradient([query], policy).unwrap()[0]);
        }
    }
}
