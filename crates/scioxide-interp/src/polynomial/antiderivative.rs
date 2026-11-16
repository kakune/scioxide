//! Antiderivatives of the global polynomial interpolants.
//!
//! The barycentric, Newton and Hermite interpolants integrate through
//! [`PreparedPolynomialAntiderivative`] instead of [`crate::Integrable`], so the
//! O(n² d) conversion to power coefficients runs once rather than per integral.

#![allow(clippy::many_single_char_names)]

use alloc::vec::Vec;

use scioxide_core::backend::{
    checked_add, checked_mul, try_copy_slice, try_filled, try_with_capacity,
};
use scioxide_core::{
    AffineValue, AffineValueMut, Interval, NonFiniteOrigin, NumericalIssue, RealScalar,
    SciOxideError, SciOxideErrorKind, check_value_shape, write_value,
};

use crate::checked::{add, interval_normalization, mul, output};
use crate::error::instability;
use crate::protocol::{ExtrapolationPolicy, clamped_integral, integral_bounds};
use crate::spline::primitive;
use crate::validate::validate_finite_origin;
use crate::values::{fill_owned, gather, zeroed};
use scioxide_poly::backend::polynomial_basis::{
    linear_multiply_difference, newton_to_power_into, normalized_linear_difference,
    real_divided_differences_into,
};

/// Evaluates definite integrals `∫ₐᵇ p(x) dx` of a global interpolating polynomial `p`.
///
/// Created by `prepare_antiderivative` on [`crate::BarycentricInterpolator`],
/// [`crate::NewtonInterpolator`] and [`crate::HermiteInterpolator`]. Each real
/// component stores the power coefficients of an antiderivative `F` in the
/// normalized coordinate `u = (x − c) / h`, where `c` is the midpoint and `h`
/// the half-width of the node interval. The integral is `F(b) − F(a)`, at O(n d)
/// cost for degree `n − 1` and `d` real components. Preparation costs
/// O(n² d) operations and O(n d) memory.
///
/// Bounds are checked in the order of [`crate::Integrable`]. Under
/// [`ExtrapolationPolicy::Clamp`] the constant extension beyond the domain is
/// integrated exactly. Under the other continuation policies the polynomial is
/// integrated.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_interp::{ExtrapolationPolicy, NewtonInterpolator};
/// // p(x) = (x, 2x) on [0, 1].
/// let polynomial = NewtonInterpolator::new(
///     vec![0.0_f64, 1.0],
///     vec![Complex::new(0.0, 0.0), Complex::new(1.0, 2.0)],
/// )?;
/// let prepared = polynomial.prepare_antiderivative()?;
/// let integral = prepared.integral(0.0, 1.0, ExtrapolationPolicy::Reject)?;
/// assert!((integral - Complex::new(0.5, 1.0)).norm() < 1e-15);
/// let mut out = Complex::new(0.0, 0.0);
/// let mut scratch = [0.0; 2];
/// prepared.integral_into(
///     0.0,
///     1.0,
///     ExtrapolationPolicy::Reject,
///     &mut out,
///     &mut scratch,
/// )?;
/// assert_eq!(out, integral);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// Global polynomial interpolants do not implement [`crate::Integrable`]:
///
/// ```compile_fail
/// use scioxide_interp::{Integrable, NewtonInterpolator};
/// fn integrable<I: Integrable<f64, 1>>(_: &I) {}
/// let polynomial = NewtonInterpolator::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
/// integrable(&polynomial);
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct PreparedPolynomialAntiderivative<S: RealScalar, V: AffineValue<Scale = S> = S> {
    template: V,
    dimension: usize,
    /// Component-major integrated coefficients, `degree + 2` per component.
    integrated_coefficients: Vec<S>,
    domain: Interval<S>,
    center: S,
    scale: S,
    degree: usize,
}

impl<S: RealScalar, V: AffineValue<Scale = S>> PreparedPolynomialAntiderivative<S, V> {
    /// Returns the definite integral over `[a, b]`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if either bound is NaN or infinite.
    /// - `InvalidInput` if `a > b`.
    /// - `DomainError` if a bound lies outside [`Self::domain`] and `policy`
    ///   does not allow it.
    /// - `NonFinite` or `NumericalInstability` if the result overflows.
    /// - `AllocationFailure` if a dynamically sized value cannot be allocated.
    pub fn integral(
        &self,
        a: S,
        b: S,
        policy: ExtrapolationPolicy<S>,
    ) -> Result<V::Owned, SciOxideError> {
        integral_bounds(self.domain, a, b, policy)?;
        let mut out = zeroed(&self.template)?;
        check_value_shape(&out, self.dimension)?;
        fill_owned(&mut out, |component| {
            self.integral_component(a, b, component, policy)
        })?;
        Ok(out)
    }

    /// Writes the definite integral over `[a, b]` to `out`.
    ///
    /// `scratch` needs at least [`Self::value_dimension`] entries.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::integral`].
    /// - `DimensionMismatch` if `out` has the wrong shape or `scratch` is too short.
    pub fn integral_into<O: AffineValueMut<Scale = S> + ?Sized>(
        &self,
        a: S,
        b: S,
        policy: ExtrapolationPolicy<S>,
        out: &mut O,
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        check_value_shape(out, self.dimension)?;
        let available = scratch.len();
        let staged = scratch
            .get_mut(..self.dimension)
            .ok_or_else(|| crate::error::shape(self.dimension, available))?;
        integral_bounds(self.domain, a, b, policy)?;
        for (component, slot) in staged.iter_mut().enumerate() {
            *slot = self.integral_component(a, b, component, policy)?;
        }
        write_value(staged, out)
    }

    /// Returns the number of real components per value, which is also the minimum scratch length.
    #[must_use]
    pub fn value_dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the interval spanned by the interpolation nodes.
    pub fn domain(&self) -> Interval<S> {
        self.domain
    }

    /// Returns the degree of the source interpolant's representation.
    ///
    /// This is the nominal degree, `n − 1` for `n` nodes. The interpolated data
    /// may happen to define a polynomial of lower degree.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Integrate one real component over validated bounds.
    fn integral_component(
        &self,
        a: S,
        b: S,
        component: usize,
        policy: ExtrapolationPolicy<S>,
    ) -> Result<S, SciOxideError> {
        if policy == ExtrapolationPolicy::Clamp {
            return clamped_integral(
                self.domain,
                a,
                b,
                |x| self.value_component(x, component),
                |a, b| self.polynomial_integral(a, b, component),
            );
        }
        self.polynomial_integral(a, b, component)
    }

    /// Integral of the polynomial itself, `F(b) - F(a)`.
    fn polynomial_integral(&self, a: S, b: S, component: usize) -> Result<S, SciOxideError> {
        if a == b {
            return Ok(S::ZERO);
        }
        // Both bounds share the same centre and scale, so accuracy is governed by
        // evaluating F in the well-conditioned normalized coordinate rather than
        // by this final subtraction.
        let lower = self.evaluate_bound(a, component)?;
        let upper = self.evaluate_bound(b, component)?;
        output(add(upper, -lower)?)
    }

    /// Coefficients of one component.
    fn coefficients(&self, component: usize) -> &[S] {
        let stride = self.degree + 2;
        &self.integrated_coefficients[component * stride..(component + 1) * stride]
    }

    /// Interpolant value recovered from the antiderivative.
    fn value_component(&self, bound: S, component: usize) -> Result<S, SciOxideError> {
        // Recovers the interpolant value p(x) = dF/dx from the stored
        // antiderivative: with F(x) = scale * sum_k a_k u^k and du/dx = 1/scale,
        // p = sum_{k>=1} k a_k u^(k-1). Evaluated by Horner in u; `skip(1)` drops
        // the zero constant term. Used only for the clamped exterior pieces.
        let normalized = normalized_linear_difference(bound, self.center, self.scale)?;
        self.coefficients(component)
            .iter()
            .enumerate()
            .skip(1)
            .rev()
            .try_fold(S::ZERO, |acc, (power, &coefficient)| {
                add(
                    mul(acc, normalized)?,
                    mul(S::try_from_usize_exact(power)?, coefficient)?,
                )
            })
    }

    /// Antiderivative at `bound`.
    fn evaluate_bound(&self, bound: S, component: usize) -> Result<S, SciOxideError> {
        // The antiderivative is one degree higher than the interpolant. The
        // Horner sum is in `u`, so the final factor `scale` converts `du` back
        // to `dx`.
        let normalized = normalized_linear_difference(bound, self.center, self.scale)?;
        let value = self
            .coefficients(component)
            .iter()
            .rev()
            .copied()
            .try_fold(S::ZERO, |accumulator, coefficient| {
                let product = mul(accumulator, normalized)?;
                add(product, coefficient)
            })?;
        mul(self.scale, value)
    }
}

/// Prepare from node-major affine samples (`[node][component]`).
pub(crate) fn prepare_from_samples<S: RealScalar, V: AffineValue<Scale = S>>(
    nodes: &[S],
    values: &[S],
    dimension: usize,
    template: &V,
    domain: Interval<S>,
) -> Result<PreparedPolynomialAntiderivative<S, V::Owned>, SciOxideError> {
    check_value_shape(template, dimension)?;
    // Divided differences are computed directly on normalized nodes in
    // [-1, 1], so the resulting Newton form is already expressed in `u` and the
    // power conversion below uses the identity map (center 0, scale 1). Working
    // in raw coordinates would make the monomial coefficients scale like
    // |x|^k and lose precision for domains far from the origin.
    let (center, scale) = normalization(domain)?;
    let mut normalized_nodes = try_with_capacity(
        nodes.len(),
        scioxide_core::quantity!("normalized_interpolation_nodes"),
    )?;
    for &node in nodes {
        normalized_nodes.push(normalized_linear_difference(node, center, scale)?);
    }
    let mut coefficients = try_copy_slice(
        values,
        scioxide_core::quantity!("antiderivative_newton_coefficients"),
    )?;
    // Construction validated finite, distinct nodes and finite samples, so the
    // only input the table can reject is a pair of distinct nodes whose
    // normalized coordinates rounded together.
    real_divided_differences_into(&normalized_nodes, dimension, &mut coefficients).map_err(
        |error| {
            if error.kind() == SciOxideErrorKind::InvalidInput {
                instability(
                    NumericalIssue::LossOfSignificance,
                    "distinct nodes coincide after normalization",
                )
            } else {
                error
            }
        },
    )?;
    Ok(PreparedPolynomialAntiderivative {
        template: template.try_to_owned()?,
        dimension,
        integrated_coefficients: integrate_components(
            &normalized_nodes,
            &coefficients,
            dimension,
            S::ZERO,
            S::ONE,
        )?,
        domain,
        center,
        scale,
        degree: nodes.len() - 1,
    })
}

/// Prepare from a node-major affine Newton form (`[index][component]`).
pub(crate) fn prepare_from_newton<S: RealScalar, V: AffineValue<Scale = S>>(
    nodes: &[S],
    coefficients: &[S],
    dimension: usize,
    template: &V,
    domain: Interval<S>,
) -> Result<PreparedPolynomialAntiderivative<S, V::Owned>, SciOxideError> {
    check_value_shape(template, dimension)?;
    let (center, scale) = normalization(domain)?;
    Ok(PreparedPolynomialAntiderivative {
        template: template.try_to_owned()?,
        dimension,
        integrated_coefficients: integrate_components(
            nodes,
            coefficients,
            dimension,
            center,
            scale,
        )?,
        domain,
        center,
        scale,
        degree: nodes.len() - 1,
    })
}

/// Integrated power series of a node-major Newton form, one per component.
///
/// Each component is gathered into a contiguous scalar series, converted
/// independently over `x = center + scale * u`, and appended so the result is
/// component-major with a stride of `degree + 2`, as `evaluate_bound` expects.
fn integrate_components<S: RealScalar>(
    nodes: &[S],
    coefficients: &[S],
    dimension: usize,
    center: S,
    scale: S,
) -> Result<Vec<S>, SciOxideError> {
    let mut column = try_filled(
        nodes.len(),
        S::ZERO,
        scioxide_core::quantity!("antiderivative_newton_column"),
    )?;
    let mut integrated_coefficients = integrated_storage(nodes.len(), dimension)?;
    for component in 0..dimension {
        gather(coefficients, dimension, component, &mut column);
        validate_prepared_state(&column)?;
        let integrated = newton_to_integrated_power(nodes, &column, center, scale)?;
        integrated_coefficients.extend_from_slice(&integrated);
    }
    Ok(integrated_coefficients)
}

/// Empty storage for `dimension` integrated series over `count` nodes.
fn integrated_storage<S>(count: usize, dimension: usize) -> Result<Vec<S>, SciOxideError> {
    let stride = checked_add(
        count,
        1,
        scioxide_core::quantity!("prepared_antiderivative_coefficient_length"),
    )?;
    try_with_capacity(
        checked_mul(
            stride,
            dimension,
            scioxide_core::quantity!("prepared_antiderivative_coefficients"),
        )?,
        scioxide_core::quantity!("prepared_antiderivative_coefficients"),
    )
}

/// Domain centre and half-width; a single-node domain uses `(lower, 1)`.
fn normalization<S: RealScalar>(domain: Interval<S>) -> Result<(S, S), SciOxideError> {
    if domain.lower() == domain.upper() {
        return Ok((domain.lower(), S::ONE));
    }
    interval_normalization(domain.lower(), domain.upper())
}

/// Integrated power coefficients in `u = (x - center) / scale`, constant zero.
fn newton_to_integrated_power<S: RealScalar>(
    nodes: &[S],
    coefficients: &[S],
    center: S,
    scale: S,
) -> Result<Vec<S>, SciOxideError> {
    let storage_len = checked_add(
        coefficients.len(),
        1,
        scioxide_core::quantity!("prepared_antiderivative_coefficient_length"),
    )?;
    let mut power = try_filled(
        storage_len,
        S::ZERO,
        scioxide_core::quantity!("prepared_antiderivative_coefficients"),
    )?;
    // The Newton form uses every node but the last. `linear_multiply_difference`
    // forms `(center - x_i) q` without materializing the difference.
    let terms = coefficients.len();
    newton_to_power_into(
        &nodes[..terms - 1],
        coefficients.iter().copied().map(Ok),
        &mut power[1..],
        |value| mul(scale, value),
        |addend, value, node| add(linear_multiply_difference(value, center, node)?, addend),
    )?;
    // Integrate term by term with a zero constant (F(center) = 0).
    primitive(&mut power[1..])?;
    Ok(power)
}

/// Reject empty or non-finite coefficients.
fn validate_prepared_state<S: RealScalar>(values: &[S]) -> Result<(), SciOxideError> {
    if values.is_empty() {
        return Err(instability(
            NumericalIssue::Breakdown,
            "polynomial representation is empty",
        ));
    }
    validate_finite_origin(values, NonFiniteOrigin::Intermediate)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_support::assert_error;
    use scioxide_core::Complex;

    #[test]
    fn checked_preparation_storage_reports_size_and_allocation_errors() {
        assert!(
            matches!(integrated_storage::<f64>(usize::MAX, 1), Err(error) if matches!(error.view(), scioxide_core::SciOxideErrorView::SizeOverflow { .. }))
        );
        assert!(
            matches!(try_filled::<f64>(usize::MAX, 0.0,  scioxide_core::quantity!("prepared_antiderivative_coefficients")), Err(error) if matches!(error.view(), scioxide_core::SciOxideErrorView::AllocationFailure { .. }))
        );
    }

    #[test]
    fn nonrepresentable_newton_conversion_is_rejected() {
        let domain = Interval::new(-1.0, 1.0).unwrap();
        let result = prepare_from_newton(&[-1.0, 1.0], &[f64::MAX, f64::MAX], 1, &0.0, domain);
        assert!(matches!(
            assert_error(&result, scioxide_core::SciOxideErrorKind::NonFinite).view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Intermediate,
                ..
            }
        ));
    }

    #[test]
    fn affine_samples_integrate_per_component_bit_for_bit() {
        let domain = Interval::new(-1.0, 2.0).unwrap();
        let nodes = [-1.0_f64, 0.0, 0.5, 2.0];
        let flat = [1.0, -2.0, 0.5, 3.0, 2.0, 0.25, -1.0, 4.0];
        let template = Complex::new(0.0, 0.0);
        let prepared = prepare_from_samples(&nodes, &flat, 2, &template, domain).unwrap();
        let real = prepare_from_samples(&nodes, &[1.0, 0.5, 2.0, -1.0], 1, &0.0, domain).unwrap();
        let imaginary =
            prepare_from_samples(&nodes, &[-2.0, 3.0, 0.25, 4.0], 1, &0.0, domain).unwrap();
        for policy in [ExtrapolationPolicy::Reject, ExtrapolationPolicy::Clamp] {
            let value = prepared.integral(-0.5, 1.5, policy).unwrap();
            assert_eq!(
                value.re.to_bits(),
                real.integral(-0.5, 1.5, policy).unwrap().to_bits()
            );
            assert_eq!(
                value.im.to_bits(),
                imaginary.integral(-0.5, 1.5, policy).unwrap().to_bits()
            );
        }
    }

    #[test]
    fn borrowed_integral_checks_shapes_before_work() {
        let domain = Interval::new(0.0, 1.0).unwrap();
        let prepared = prepare_from_samples(&[0.0, 1.0], &[0.0, 1.0], 1, &0.0, domain).unwrap();
        let mut out = [7.0_f64];
        let error = prepared
            .integral_into(0.0, 1.0, ExtrapolationPolicy::Reject, &mut out, &mut [])
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        let error = prepared
            .integral_into(1.0, 0.0, ExtrapolationPolicy::Reject, &mut out, &mut [0.0])
            .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        assert_eq!(out, [7.0]);
        prepared
            .integral_into(
                0.0,
                1.0,
                ExtrapolationPolicy::Reject,
                &mut out,
                &mut [0.0; 3],
            )
            .unwrap();
        assert!((out[0] - 0.5).abs() < 1e-15);
    }
}
