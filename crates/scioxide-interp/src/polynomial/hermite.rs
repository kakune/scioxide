//! Polynomial Hermite interpolation of values and first derivatives.

#![allow(clippy::many_single_char_names)]

use alloc::vec::Vec;

use scioxide_core::backend::try_filled;
use scioxide_core::{
    AffineValue, AffineValueMut, Capacity, Interval, RealScalar, SciOxideError, check_value_shape,
    write_value,
};

use crate::checked::{ordinary_nodes, output};
use crate::polynomial::antiderivative::{PreparedPolynomialAntiderivative, prepare_from_newton};
use crate::polynomial::newton::{evaluate_newton, native_components};
use crate::protocol::{
    Differentiable, ExtrapolationPolicy, Interpolant, SampleOptions, clamped_axes,
    domain_from_nodes, resolve_coordinate,
};
use crate::validate::validate_nodes;
use crate::values::{QueryWorkspace, flatten, zeroed};
use scioxide_poly::backend::polynomial_basis::confluent_divided_differences_into;

/// Interpolates values and first derivatives with one polynomial of degree at most `2n − 1`.
///
/// For nodes `x₀ < … < xₙ₋₁`, values `yⱼ` and derivatives `y′ⱼ`, the
/// interpolant is the unique polynomial `p` of degree at most `2n − 1` with
/// `p(xⱼ) = yⱼ` and `p′(xⱼ) = y′ⱼ`. It is stored in Newton form on the
/// doubled nodes `x₀, x₀, x₁, x₁, …`, with confluent divided differences
/// `f[xⱼ, xⱼ] = y′ⱼ`. Each real component of the values is interpolated
/// separately. With [`ExtrapolationPolicy::MethodDefined`] the polynomial is
/// evaluated outside `[x₀, xₙ₋₁]`.
///
/// Construction costs O(n² d) operations and O(n d) memory for `d` real
/// components. A value or derivative query costs O(n d). Definite integrals
/// come from [`Self::prepare_antiderivative`]. For a piecewise cubic Hermite
/// interpolant of moderate degree, see [`crate::PchipInterpolator`] or
/// [`crate::CubicSpline`].
///
/// # Errors
///
/// Queries fail as described in the [crate docs](crate#errors), and with
/// `NonFinite` or `NumericalInstability` if the result overflows.
///
/// # Examples
///
/// ```
/// use scioxide_interp::{Differentiable, ExtrapolationPolicy, HermiteInterpolator, Interpolant};
///
/// // Interpolate y = x² with exact derivatives y′ = 2x.
/// let x = vec![0.0_f64, 1.0, 2.0];
/// let y = vec![0.0_f64, 1.0, 4.0];
/// let dy = vec![0.0_f64, 2.0, 4.0];
/// let interp = HermiteInterpolator::new(x, y, dy)?;
/// let value = interp.evaluate([1.5], ExtrapolationPolicy::Reject)?;
/// assert!((value - 2.25_f64).abs() < 1e-14);
/// let [slope] = interp.gradient([1.5], ExtrapolationPolicy::Reject)?;
/// assert!((slope - 3.0_f64).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. Stoer and R. Bulirsch, *Introduction to Numerical Analysis*, 3rd ed.,
/// Springer, 2002, §2.1.5.
#[derive(Debug, Clone)]
#[must_use]
pub struct HermiteInterpolator<S: RealScalar, V: AffineValue<Scale = S> = S> {
    /// Doubled nodes: `[x0, x0, x1, x1, ...]`.
    z: Vec<S>,
    /// Node-major divided-difference coefficients on the doubled nodes.
    coeffs: Vec<S>,
    /// Components for which plain Horner arithmetic cannot overflow on the domain.
    native: Vec<bool>,
    template: V,
    dimension: usize,
    domain: Interval<S>,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> HermiteInterpolator<S, V> {
    /// Builds the interpolant with `p(x[j]) = y[j]` and `p′(x[j]) = dy[j]`.
    ///
    /// This is [`Self::with_options`] with [`SampleOptions::new`], so the nodes
    /// `x` must be strictly increasing. `dy` must have the shape of `y`.
    /// Construction costs O(n² d) operations.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `x` is empty or not strictly increasing.
    /// - `DimensionMismatch` if `x`, `y` and `dy` differ in length or the values
    ///   differ in shape.
    /// - `NonFinite` if an input is NaN or infinite (origin `Input`), or a
    ///   divided difference overflows (origin `Intermediate`).
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn new(x: Vec<S>, y: Vec<V>, dy: Vec<V>) -> Result<Self, SciOxideError> {
        Self::with_options(x, y, dy, SampleOptions::new())
    }

    /// Builds the interpolant with `p(x[j]) = y[j]` and `p′(x[j]) = dy[j]`,
    /// validating the nodes under `options`.
    ///
    /// The nodes `x` must be strictly increasing and distinct under
    /// [`SampleOptions::node_separation`], and `dy` must have the shape of `y`.
    /// Construction costs O(n² d) operations.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `x` is empty, or not strictly increasing and distinct
    ///   under the separation of `options`.
    /// - `DimensionMismatch` if `x`, `y` and `dy` differ in length or the values
    ///   differ in shape.
    /// - `NonFinite` if an input is NaN or infinite (origin `Input`), or a
    ///   divided difference overflows (origin `Intermediate`).
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn with_options(
        x: Vec<S>,
        y: Vec<V>,
        dy: Vec<V>,
        options: SampleOptions<S>,
    ) -> Result<Self, SciOxideError> {
        validate_nodes(
            &x,
            1,
            options.node_separation(),
            scioxide_core::argument!("x"),
        )?;
        let n = x.len();
        let values = flatten(y, n)?;
        let slopes = flatten(dy, n)?;
        let dimension = values.dimension;
        if slopes.dimension != dimension {
            return Err(crate::error::shape(dimension, slopes.dimension));
        }
        let m = scioxide_core::backend::checked_mul(
            2,
            n,
            scioxide_core::quantity!("hermite_doubled_nodes"),
        )?;
        let mut z = try_filled(
            m,
            S::ZERO,
            scioxide_core::quantity!("hermite_doubled_nodes"),
        )?;
        for (pair, &node) in z.chunks_exact_mut(2).zip(&x) {
            pair.fill(node);
        }
        let mut coeffs = try_filled(
            scioxide_core::backend::checked_mul(
                m,
                dimension,
                scioxide_core::quantity!("hermite_coefficients"),
            )?,
            S::ZERO,
            scioxide_core::quantity!("hermite_coefficients"),
        )?;
        confluent_divided_differences_into(
            &x,
            &values.components,
            &slopes.components,
            dimension,
            &mut coeffs,
        )?;
        let native = native_components(&z, ordinary_nodes(&x), &coeffs, dimension)?;
        Ok(Self {
            z,
            coeffs,
            native,
            template: values.template,
            dimension,
            domain: domain_from_nodes(&x)?,
        })
    }

    /// Computes the antiderivative of the interpolant, for definite integrals.
    ///
    /// See [`PreparedPolynomialAntiderivative`]. Preparation costs O(n² d)
    /// operations and O(n d) memory, and each integral O(n d).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the coefficients cannot be stored.
    /// - `NumericalInstability` if a coefficient is not representable.
    pub fn prepare_antiderivative(
        &self,
    ) -> Result<PreparedPolynomialAntiderivative<S, V::Owned>, SciOxideError> {
        prepare_from_newton(
            &self.z,
            &self.coeffs,
            self.dimension,
            &self.template,
            self.domain,
        )
    }

    /// Computes values or derivatives of every component and writes them to `out`.
    fn stage<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
        derivative: bool,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        check_value_shape(out, self.dimension)?;
        let staged = workspace.scratch(self.dimension)?;
        let [clamped] = clamped_axes(policy, [self.domain], point);
        let t = resolve_coordinate(&self.domain, point[0], policy)?;
        let native = self.domain.contains(t);
        for (component, slot) in staged.iter_mut().enumerate() {
            let native = native && self.native[component];
            let (z, coeffs, d) = (&self.z, &self.coeffs, self.dimension);
            *slot = match (derivative, clamped) {
                (true, true) => S::ZERO,
                (true, false) => output(evaluate_newton::<S, true>(
                    z, coeffs, d, component, t, native,
                )?)?,
                (false, _) => output(evaluate_newton::<S, false>(
                    z, coeffs, d, component, t, native,
                )?)?,
            };
        }
        write_value(staged, out)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Interpolant<S, 1>
    for HermiteInterpolator<S, V>
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
        zeroed(&self.template)
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
        self.stage(point, policy, false, out, workspace)
    }
}

impl<S: RealScalar, V: AffineValue<Scale = S> + Sync> Differentiable<S, 1>
    for HermiteInterpolator<S, V>
{
    fn gradient_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        point: [S; 1],
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        workspace: &mut QueryWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        self.stage(point, policy, true, out, workspace)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::{assert_close, assert_error};
    use scioxide_core::Complex;
    use scioxide_core::NonFiniteOrigin;

    fn hermite(
        x: &[f64],
        y: &[f64],
        dy: &[f64],
    ) -> Result<HermiteInterpolator<f64>, SciOxideError> {
        HermiteInterpolator::new(x.to_vec(), y.to_vec(), dy.to_vec())
    }

    #[test]
    fn test_new_valid() {
        assert!(hermite(&[0.0, 1.0], &[0.0, 1.0], &[1.0, 1.0]).is_ok());
    }

    #[test]
    fn test_new_dimension_mismatch_y() {
        let result = hermite(&[0.0, 1.0], &[0.0], &[1.0, 1.0]);
        assert_error(&result, scioxide_core::SciOxideErrorKind::DimensionMismatch);
    }

    #[test]
    fn test_new_dimension_mismatch_dy() {
        let result = hermite(&[0.0, 1.0], &[0.0, 1.0], &[1.0]);
        assert_error(&result, scioxide_core::SciOxideErrorKind::DimensionMismatch);
        let shapes = HermiteInterpolator::new(
            vec![0.0_f64, 1.0],
            vec![vec![0.0, 1.0], vec![1.0, 2.0]],
            vec![vec![1.0], vec![1.0]],
        );
        assert_error(&shapes, scioxide_core::SciOxideErrorKind::DimensionMismatch);
    }

    #[test]
    fn extreme_derived_divided_difference_reports_intermediate_origin() {
        let result = hermite(
            &[0.0, f64::MIN_POSITIVE],
            &[-f64::MAX, f64::MAX],
            &[0.0, 0.0],
        );

        assert!(matches!(
            assert_error(&result, scioxide_core::SciOxideErrorKind::NonFinite).view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Intermediate,
                ..
            }
        ));
    }

    #[test]
    fn non_finite_samples_report_input_origin() {
        let result = hermite(&[0.0], &[f64::NAN], &[0.0]);

        assert!(matches!(
            assert_error(&result, scioxide_core::SciOxideErrorKind::NonFinite).view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_interpolate_at_nodes() {
        let x: [f64; 3] = [0.0, 1.0, 2.0];
        let y: [f64; 3] = [0.0, 1.0, 4.0];
        let dy: [f64; 3] = [0.0, 2.0, 4.0];
        let interp = hermite(&x, &y, &dy).unwrap();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let v = interp.evaluate([*xi], ExtrapolationPolicy::Reject).unwrap();
            assert_close(v, *yi, 5e-13, 5e-13);
        }
    }

    #[test]
    fn test_interpolate_quadratic() {
        // y = x^2, dy = 2x: Hermite should recover x^2 exactly
        let x: [f64; 3] = [0.0, 1.0, 2.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let dy: Vec<f64> = x.iter().map(|xi| 2.0 * xi).collect();
        let interp = hermite(&x, &y, &dy).unwrap();
        let v = interp.evaluate([0.5], ExtrapolationPolicy::Reject).unwrap();
        assert_close(v, 0.25, 5e-14, 5e-14);
        let v = interp.evaluate([1.5], ExtrapolationPolicy::Reject).unwrap();
        assert_close(v, 2.25, 5e-14, 5e-14);
    }

    #[test]
    fn test_derivative_at_nodes() {
        let x: [f64; 3] = [0.0, 1.0, 2.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let dy: Vec<f64> = x.iter().map(|xi| 2.0 * xi).collect();
        let interp = hermite(&x, &y, &dy).unwrap();
        for (xi, dyi) in x.iter().zip(dy.iter()) {
            let [d] = interp.gradient([*xi], ExtrapolationPolicy::Reject).unwrap();
            assert_close(d, *dyi, 5e-11, 5e-11);
        }
    }

    #[test]
    fn test_integral_quadratic() {
        let x: [f64; 3] = [0.0, 1.0, 2.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let dy: Vec<f64> = x.iter().map(|xi| 2.0 * xi).collect();
        let interp = hermite(&x, &y, &dy).unwrap();
        let integ = interp
            .prepare_antiderivative()
            .unwrap()
            .integral(0.0, 2.0, ExtrapolationPolicy::Reject)
            .unwrap();
        assert_close(integ, 8.0 / 3.0, 5e-11, 5e-11);
    }

    #[test]
    fn affine_components_match_scalar_interpolants_bit_for_bit() {
        let x = vec![-1.0_f64, 0.25, 2.0];
        let y = [1.0, -2.0, 0.5];
        let dy = [0.5, 3.0, -1.0];
        let values: Vec<Complex<f64>> = y
            .iter()
            .zip(&dy)
            .map(|(&a, &b)| Complex::new(a, b))
            .collect();
        let slopes: Vec<Complex<f64>> = dy
            .iter()
            .zip(&y)
            .map(|(&a, &b)| Complex::new(a, -b))
            .collect();
        let complex = HermiteInterpolator::new(x.clone(), values, slopes).unwrap();
        let real = hermite(&x, &y, &dy).unwrap();
        let imaginary = hermite(&x, &dy, &y.map(|v| -v)).unwrap();
        for query in [-1.0, 0.0, 1.3, 2.0, 2.5] {
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
