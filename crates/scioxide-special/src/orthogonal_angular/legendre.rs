//! Legendre polynomials and functions: `P_l(x)`, `P_l^m(x)`, `Q_l(x)`.
//!
//! # Algorithms
//!
//! - **`legendre_p`** — `f64` recurrence with an FMA residual correction: P₀ = 1, P₁ = x,
//!   (l+1) P\_{l+1} = (2l+1) x P\_l − l P\_{l−1}. Wide arithmetic
//!   is used near zeros and for the rescaled growing recurrence outside [-1,1].
//! - **`legendre_p_assoc`** — starts from `P_m^m` via the double-factorial
//!   formula with compensated arithmetic and exact binary rescaling, then
//!   recurs upward in l without a normalized-to-raw conversion. Negative m uses the standard
//!   reduction `P_l^{−m}` = (−1)^m (l−m)!/(l+m)! `P_l^m`.
//! - **`legendre_q`** — a compensated logarithmic Q₀(x) = atanh(x),
//!   Q₁ = x Q₀ − 1, then a twofold three-term recurrence.
//!
//! Primitive f32 arguments widen exactly; both widths round after recurrence.
//! Both widths keep the residual until the final rounding.

use crate::support::wide::Wide as D;
use crate::support::{Work, require_finite_param};
use crate::util::is_exact;
use scioxide_core::Scaled;
use scioxide_core::{RealScalar, SciOxideError};

pub(crate) struct AssociatedValue {
    pub value: Scaled<f64>,
    pub envelope: Scaled<f64>,
}

/// The coefficient 2*l-1 must remain an exact `f64` integer. This is a
/// representation boundary, independent of the caller's recurrence budget.
#[inline]
fn require_recurrence_degree(l: usize) -> Result<(), SciOxideError> {
    if l as u128 > 1_u128 << 52 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("degree"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "Legendre recurrence coefficients must be exact binary64 integers",
        ));
    }
    Ok(())
}
/// Rescale the complete recurrence state exactly, keeping its low components.
fn normalize(values: &mut [D], exponent: &mut i64) -> Result<(), SciOxideError> {
    let magnitude = values.iter().map(|v| v.hi.abs()).fold(0.0_f64, f64::max);
    if magnitude > 0.0 && magnitude.is_finite() && !(1e-100..=1e100).contains(&magnitude) {
        // Finite IEEE decoding bounds this exponent to [-1074, 1023].
        let shift = Scaled::from_value(magnitude)?.exponent() as i32;
        let next_exponent = exponent.checked_add(i64::from(shift)).ok_or_else(|| {
            SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::ExponentOutOfRange,
                "recurrence exponent exceeds i64",
            )
        })?;
        for value in values {
            value.hi = value.hi.scalbn(-shift);
            value.lo = value.lo.scalbn(-shift);
        }
        *exponent = next_exponent;
    }
    Ok(())
}

fn scaled(value: D, exponent: i64) -> Result<Scaled<f64>, SciOxideError> {
    // Combine only after the recurrence and all order factors have completed.
    Scaled::from_value(project::<f64>(value.value())?)?.shifted(exponent)
}

fn project<S: RealScalar>(value: f64) -> Result<S, SciOxideError> {
    // Finite recurrence operands can overflow both parts of the twofold
    // state, producing NaN while recovering the rounding residual.
    if !value.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "Legendre value exceeds the scalar range",
        ));
    }
    crate::support::project_real(value)
}

#[allow(clippy::cast_precision_loss)]
fn recurrence(
    l: usize,
    x: f64,
    previous: D,
    current: D,
    work: &mut Work,
) -> Result<f64, SciOxideError> {
    let mut state = [previous, current];
    let mut exponent = 0;
    let growing = x.abs() > 1.0;
    if growing {
        normalize(&mut state, &mut exponent)?;
    }
    for k in 1..l {
        work.try_step()?;
        // require_recurrence_degree proves that 2*k+1 is exact in binary64.
        let k = k as f64;
        let next =
            (state[1].mul_f64(x).mul_f64(2.0 * k + 1.0) - state[0].mul_f64(k)).div_f64(k + 1.0);
        state = [state[1], next];
        if growing {
            normalize(&mut state, &mut exponent)?;
        }
    }
    if exponent == 0 {
        project(state[1].value())
    } else {
        scaled(state[1], exponent)?.round()
    }
}

/// Propagate the rounding residual of the `f64` recurrence alongside its value. Unlike a twofold
/// recurrence, the correction needs no normalization after every product: it satisfies the same
/// linear recurrence. FMA recovers product and division residuals; the subtraction uses an
/// error-free sum.
fn polynomial_recurrence(l: usize, x: f64, work: &mut Work) -> Result<f64, SciOxideError> {
    if x.abs() > 1.0 {
        return recurrence(l, x, D::from(1.0), D::from(x), work);
    }
    let (mut previous, mut current) = (1.0, x);
    let (mut previous_error, mut current_error) = (0.0, 0.0);
    for index in 1..l {
        work.try_step()?;
        let k = index as f64;
        let coefficient = 2.0 * k + 1.0;
        let product = x * current;
        let left = coefficient * product;
        let right = k * previous;
        let numerator = left - right;
        let virtual_right = numerator - left;
        let subtraction_error = (left - (numerator - virtual_right)) - (right + virtual_right);
        let residual = coefficient.mul_add(product, -left)
            + coefficient * x.mul_add(current, -product)
            - k.mul_add(previous, -right)
            + subtraction_error;
        let reciprocal = 1.0 / (k + 1.0);
        let next = numerator * reciprocal;
        let local_error = (-next).mul_add(k + 1.0, numerator) + residual;
        let next_error =
            (coefficient * x * current_error - k * previous_error + local_error) * reciprocal;
        previous = current;
        current = next;
        previous_error = current_error;
        current_error = next_error;
    }
    let value = current + current_error;
    // Near a polynomial zero the correction can be the entire answer. Retain
    // twofold propagation there rather than magnifying its own roundoff.
    if value != 0.0 && value.abs() < 64.0 * l as f64 * f64::EPSILON {
        recurrence(l, x, D::from(1.0), D::from(x), work)
    } else {
        Ok(value)
    }
}

// Real implementation of the public crate::legendre_p contract.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn legendre_p<S: RealScalar>(
    l: usize,
    x: S,
    work: &mut Work,
) -> Result<S, SciOxideError> {
    require_finite_param(x, "Legendre parameter x must be finite")?;
    if l == 0 {
        return Ok(S::ONE);
    }
    if l == 1 {
        return Ok(x);
    }
    if is_exact(x, S::ONE) {
        return Ok(S::ONE);
    }
    if is_exact(x, -S::ONE) {
        return Ok(if l.is_multiple_of(2) { S::ONE } else { -S::ONE });
    }
    require_recurrence_degree(l)?;

    let input = x.into_f64();
    let value = polynomial_recurrence(l, input, work)?;
    project(value)
}

/// Compensated raw associated recurrence with an explicit binary exponent.
pub(crate) fn associated_scaled(
    l: usize,
    m: i32,
    x: f64,
    work: &mut crate::support::Work<'_, '_, '_>,
) -> Result<AssociatedValue, SciOxideError> {
    require_finite_param(x, "Legendre parameter x must be finite")?;
    if !(-1.0..=1.0).contains(&x) {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::OutsideDomain,
            "x must be in [-1,1]",
        ));
    }
    let m_abs = m.unsigned_abs() as usize;
    if m_abs > l {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("order"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "|m| must not exceed degree",
        ));
    }
    work.check()?;
    if l == 0 {
        return Ok(AssociatedValue {
            value: Scaled::from_value(1.0)?,
            envelope: Scaled::ZERO,
        });
    }
    if x.abs() == 1.0 {
        let value = Scaled::from_value(if m_abs == 0 {
            if x < 0.0 && !l.is_multiple_of(2) {
                -1.0
            } else {
                1.0
            }
        } else if m >= 0 && !m_abs.is_multiple_of(2) {
            -0.0
        } else {
            0.0
        })?;
        return Ok(AssociatedValue {
            value,
            envelope: Scaled::ZERO,
        });
    }
    let upper = l.checked_add(m_abs).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("legendre_factorial_range"))
    })?;
    if upper as u128 > 1_u128 << 53 {
        return Err(SciOxideError::size_overflow(scioxide_core::quantity!(
            "legendre_exact_integer_indices"
        )));
    }
    let square = (D::from(1.0) - D::from(x)) * (D::from(1.0) + D::from(x));
    let root = square.hi.sqrt();
    let sine = D::from(root) + (square - D::from(root) * D::from(root)).div_f64(2.0 * root);
    let mut state = [D::from(1.0), D::ZERO, D::from(1.0)];
    let mut exponent = 0;
    for k in 1..=m_abs {
        work.try_step()?;
        state[0] = (state[0] * sine).mul_f64(-(2.0 * f64::try_from_usize_exact(k)? - 1.0));
        state[2] = D::from(state[0].hi.abs());
        normalize(&mut state, &mut exponent)?;
    }
    if l > m_abs {
        work.try_step()?;
        state[1] = state[0]
            .mul_f64(x)
            .mul_f64(2.0 * f64::try_from_usize_exact(m_abs)? + 1.0);
        state[2] = D::from(state[0].hi.abs() + state[1].hi.abs());
        normalize(&mut state, &mut exponent)?;
        for k in (m_abs + 2)..=l {
            work.try_step()?;
            let k = f64::try_from_usize_exact(k)?;
            let m = f64::try_from_usize_exact(m_abs)?;
            let first = state[1].mul_f64(x).mul_f64(2.0 * k - 1.0);
            let second = state[0].mul_f64(k + m - 1.0);
            let next = (first - second).div_f64(k - m);
            state = [
                state[1],
                next,
                D::from((first.hi.abs() + second.hi.abs()) / (k - m)),
            ];
            normalize(&mut state, &mut exponent)?;
        }
        state[0] = state[1];
    }
    // Apply negative-order reduction before projecting into the scalar range.
    if m < 0 {
        for k in (l - m_abs + 1)..=upper {
            work.try_step()?;
            let k = f64::try_from_usize_exact(k)?;
            state[0] = state[0].div_f64(k);
            state[2] = state[2].div_f64(k);
            state[1] = D::ZERO;
            normalize(&mut state, &mut exponent)?;
        }
        if !m_abs.is_multiple_of(2) {
            state[0] = -state[0];
        }
    }
    Ok(AssociatedValue {
        value: scaled(state[0], exponent)?,
        envelope: scaled(state[2], exponent)?,
    })
}

// Real implementation of the public crate::legendre_q contract.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn legendre_q<S: RealScalar>(
    l: usize,
    x: S,
    work: &mut Work,
) -> Result<S, SciOxideError> {
    require_finite_param(x, "Legendre parameter x must be finite")?;
    if is_exact(x, S::ONE) {
        return Ok(S::INFINITY);
    }
    if is_exact(x, -S::ONE) {
        return Ok(if l.is_multiple_of(2) {
            S::NEG_INFINITY
        } else {
            S::INFINITY
        });
    }
    if x < -S::ONE || x > S::ONE {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::OutsideDomain,
            "Legendre Q argument must be in (-1, 1)",
        ));
    }
    if l > 1 {
        require_recurrence_degree(l)?;
    }

    let input = x.into_f64();
    if l == 0 && input.abs() < 1e-150 {
        return project(input);
    }
    // Tiny atanh inputs round to x; avoid losing subnormal seeds in the
    // compensated logarithms. This also preserves Q_0's signed zero.
    let q0 = if input.abs() < 1e-150 {
        D::from(input)
    } else {
        ((D::from(1.0) + D::from(input)) / (D::from(1.0) - D::from(input)))
            .ln()
            .mul_f64(0.5)
    };
    let value = if l == 0 {
        q0.value()
    } else {
        let q1 = q0.mul_f64(input) - D::from(1.0);
        if l == 1 {
            q1.value()
        } else {
            recurrence(l, input, q0, q1, work)?
        }
    };
    project(value)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::suboptimal_flops)]
mod tests {
    use super::*;
    use crate::legendre_p_assoc;

    fn legendre_p<S: RealScalar>(l: usize, x: S) -> Result<S, SciOxideError> {
        super::legendre_p(l, x, &mut Work::new(usize::MAX))
    }

    fn legendre_q<S: RealScalar>(l: usize, x: S) -> Result<S, SciOxideError> {
        super::legendre_q(l, x, &mut Work::new(usize::MAX))
    }

    #[test]
    fn compensated_range_and_signed_zero() {
        assert_eq!(
            legendre_q(0, -0.0_f64).unwrap().to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(
            legendre_q(0, -0.0_f32).unwrap().to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(legendre_q(0, f64::from_bits(1)).unwrap().to_bits(), 1);
        assert!(legendre_p(2, 1e154_f64).unwrap().is_finite());
        for x in [f64::MAX, 1e155] {
            {
                let outcome = &(legendre_p(2, x));
                let err = outcome.as_ref().expect_err("expected error");
                assert!(matches!(
                    err.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Overflow,
                        ..
                    }
                ));
            }
        }
    }

    // ---- legendre_p ----

    #[test]
    fn test_legendre_p0_f64() {
        for &x in &[-1.0_f64, -0.5, 0.0, 0.5, 1.0] {
            let val = legendre_p(0, x).unwrap();
            assert!((val - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn test_legendre_p1_f64() {
        for &x in &[-1.0_f64, -0.5, 0.0, 0.5, 1.0] {
            let val = legendre_p(1, x).unwrap();
            assert!((val - x).abs() < 1e-15);
        }
    }

    #[test]
    fn test_legendre_p2_at_zero_f64() {
        let val = legendre_p(2, 0.0_f64).unwrap();
        assert!((val - (-0.5)).abs() < 1e-14);
    }

    #[test]
    fn test_legendre_p2_known_f64() {
        // P_2(x) = (3x² − 1)/2
        let x = 0.5_f64;
        let val = legendre_p(2, x).unwrap();
        let expected = (3.0 * x * x - 1.0) / 2.0;
        assert!((val - expected).abs() < 1e-14);
    }

    #[test]
    fn test_legendre_p3_known_f64() {
        // P_3(x) = (5x³ − 3x)/2
        let x = 0.6_f64;
        let val = legendre_p(3, x).unwrap();
        let expected = (5.0 * x * x * x - 3.0 * x) / 2.0;
        assert!((val - expected).abs() < 1e-14);
    }

    #[test]
    fn test_legendre_p_at_one_f64() {
        // P_l(1) = 1 for all l
        for l in 0..=10 {
            let val = legendre_p(l, 1.0_f64).unwrap();
            assert!((val - 1.0).abs() < 1e-13, "P_{l}(1) = {val}");
        }
    }

    #[test]
    fn test_legendre_p_at_minus_one_f64() {
        // P_l(-1) = (-1)^l
        for l in 0..=10 {
            let val = legendre_p(l, -1.0_f64).unwrap();
            let expected = if l % 2 == 0 { 1.0 } else { -1.0 };
            assert!((val - expected).abs() < 1e-13, "P_{l}(-1) = {val}");
        }
    }

    #[test]
    fn test_legendre_p_continues_outside_unit_interval() {
        assert_eq!(legendre_p(2, 2.0_f64).unwrap(), 5.5);
        assert_eq!(legendre_p(3, -2.0_f64).unwrap(), -17.0);
    }

    #[test]
    fn test_legendre_p_f32() {
        let val = legendre_p(2, 0.5_f32).unwrap();
        let expected = (3.0_f32 * 0.25 - 1.0) / 2.0;
        assert!((val - expected).abs() < 1e-6);
    }

    #[test]
    fn test_legendre_p_high_degree_f64() {
        // P_5(0) = 0 (odd degree)
        let val = legendre_p(5, 0.0_f64).unwrap();
        assert!(val.abs() < 1e-14);
    }

    // ---- legendre_p_assoc ----

    #[test]
    fn test_legendre_p_assoc_m0_f64() {
        // P_l^0 = P_l
        for l in 0..=5_usize {
            let x = 0.3_f64;
            let assoc = legendre_p_assoc(l, 0, x).unwrap();
            let plain = legendre_p(l, x).unwrap();
            assert!(
                (assoc - plain).abs() < 1e-13,
                "P_{l}^0({x}) mismatch: {assoc} vs {plain}"
            );
        }
    }

    #[test]
    fn test_legendre_p_assoc_1_1_f64() {
        // P_1^1(x) = −√(1 − x²)
        let x = 0.5_f64;
        let val = legendre_p_assoc(1, 1, x).unwrap();
        let expected = -(1.0 - x * x).sqrt();
        assert!((val - expected).abs() < 1e-14);
    }

    #[test]
    fn test_legendre_p_assoc_2_1_f64() {
        // P_2^1(x) = −3x√(1 − x²)
        let x = 0.5_f64;
        let val = legendre_p_assoc(2, 1, x).unwrap();
        let expected = -3.0 * x * (1.0 - x * x).sqrt();
        assert!((val - expected).abs() < 1e-14);
    }

    #[test]
    fn test_legendre_p_assoc_2_2_f64() {
        // P_2^2(x) = 3(1 − x²)
        let x = 0.5_f64;
        let val = legendre_p_assoc(2, 2, x).unwrap();
        let expected = 3.0 * (1.0 - x * x);
        assert!((val - expected).abs() < 1e-14);
    }

    #[test]
    fn test_legendre_p_assoc_neg_m_f64() {
        // P_l^{-m} = (-1)^m (l-m)!/(l+m)! P_l^m
        let l = 3_usize;
        let m_pos = 2_i32;
        let x = 0.4_f64;

        let pos = legendre_p_assoc(l, m_pos, x).unwrap();
        let neg = legendre_p_assoc(l, -m_pos, x).unwrap();

        // Manual: (-1)^2 · (3-2)!/(3+2)! · P_3^2 = 1 · 1/120 · P_3^2
        let ratio = 1.0 / 120.0; // (l-m)!/(l+m)! = 1!/5! = 1/120
        let expected = ratio * pos;
        assert!(
            (neg - expected).abs() < 1e-12,
            "P_{l}^{{{}}}: neg={neg}, expected={expected}",
            -m_pos
        );
    }

    #[test]
    fn test_legendre_p_assoc_m_too_large() {
        {
            let outcome = &(legendre_p_assoc(2_usize, 3_i32, 0.5_f64));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::InvalidInput { .. }
            ));
        }
    }

    #[test]
    fn test_legendre_p_assoc_minimum_i32_order_is_rejected_safely() {
        {
            let outcome = &(legendre_p_assoc(2_usize, i32::MIN, 0.5_f64));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::InvalidInput { .. }
            ));
        }
    }

    #[test]
    fn test_legendre_linear_work_is_bounded() {
        for result in [
            super::legendre_p(20_000, 0.25_f64, &mut Work::new(1)),
            super::legendre_q(20_000, 0.25_f64, &mut Work::new(1)),
        ] {
            let error = result.unwrap_err();
            assert_eq!(
                crate::test_support::exhausted_counts(&error).iterations(),
                1
            );
        }
        {
            let outcome = &(legendre_p_assoc(
                crate::LegendreOptions::default().iteration_budget().get() + 1,
                1,
                0.25_f64,
            ));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
            ));
        }

        // Exact endpoints are constant-time and therefore remain available.
        assert_eq!(legendre_p(usize::MAX, -1.0_f64).unwrap(), -1.0);
    }

    #[test]
    fn test_legendre_p_assoc_domain_error() {
        {
            let outcome = &(legendre_p_assoc(1_usize, 0_i32, 1.5_f64));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
    }

    #[test]
    fn test_legendre_p_assoc_f32() {
        let val = legendre_p_assoc(2_usize, 1_i32, 0.5_f32).unwrap();
        let expected = -3.0_f32 * 0.5 * (1.0 - 0.25_f32).sqrt();
        assert!((val - expected).abs() < 1e-5);
    }

    // ---- legendre_q ----

    #[test]
    fn test_legendre_q0_f64() {
        let x = 0.5_f64;
        let val = legendre_q(0, x).unwrap();
        let expected = x.atanh();
        assert!((val - expected).abs() < 1e-14);
    }

    #[test]
    fn test_legendre_q1_f64() {
        let x = 0.5_f64;
        let val = legendre_q(1, x).unwrap();
        let expected = x * x.atanh() - 1.0;
        assert!((val - expected).abs() < 1e-14);
    }

    #[test]
    fn test_legendre_q0_at_zero_f64() {
        let val = legendre_q(0, 0.0_f64).unwrap();
        assert!(val.abs() < 1e-15);
    }

    #[test]
    fn test_legendre_q1_at_zero_f64() {
        let val = legendre_q(1, 0.0_f64).unwrap();
        assert!((val - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn test_legendre_q2_f64() {
        // Q_2(x) = (3x² − 1)/4 · ln((1+x)/(1−x)) − 3x/2
        let x = 0.4_f64;
        let val = legendre_q(2, x).unwrap();
        let log_term = ((1.0 + x) / (1.0 - x)).ln();
        let expected = (3.0 * x * x - 1.0) / 4.0 * log_term - 3.0 * x / 2.0;
        assert!((val - expected).abs() < 1e-13);
    }

    #[test]
    fn test_legendre_q_domain_error() {
        assert_eq!(legendre_q(0_usize, 1.0_f64).unwrap(), f64::INFINITY);
        assert_eq!(legendre_q(0_usize, -1.0_f64).unwrap(), f64::NEG_INFINITY);
        {
            let outcome = &(legendre_q(0_usize, 1.5_f64));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
    }

    #[test]
    fn test_legendre_q_negative_endpoint_parity_signs() {
        assert_eq!(legendre_q(0, -1.0_f64).unwrap(), f64::NEG_INFINITY);
        assert_eq!(legendre_q(1, -1.0_f64).unwrap(), f64::INFINITY);
        assert_eq!(legendre_q(2, -1.0_f64).unwrap(), f64::NEG_INFINITY);
        assert_eq!(legendre_q(3, -1.0_f64).unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_legendre_q_f32() {
        let x = 0.5_f32;
        let val = legendre_q(0_usize, x).unwrap();
        let expected = x.atanh();
        assert!((val - expected).abs() < 1e-6);
    }
}
