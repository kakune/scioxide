//! Real Bessel and Hankel functions with exact order semantics.
//!
//! Reflections, parity, signed zeros and domain checks live here; the `f64`
//! kernels in `bessel_kernel` select among the defining series, Temme seeds,
//! Steed continued fractions, directional recurrence and Hankel expansions, and
//! orders of at least 32 use Debye uniform expansions for I and K. Large J/Y
//! orders use uniform Airy expansions through the turning point and Debye
//! expansions away from it. Every kernel
//! shares one caller work budget. `f32` evaluates through `f64`, and
//! two-term arithmetic is confined to the cancellation regions listed in
//! `bessel_kernel`.
//! Orders zero and one use root-factored rational and local ODE polynomials,
//! with a shared Hankel phase and adjacent-order derivative at larger arguments.

use crate::support::Work;
use crate::support::{check_finite, check_positive, require_finite_param};
use crate::util::{cos_pi, is_exact, is_exact_zero, sin_pi};
use scioxide_core::{Complex, RealScalar, SciOxideError};

fn is_exact_integer_order<S: RealScalar>(nu: S) -> bool {
    is_exact(nu, nu.round())
}

/// Adjacent-family connection support with an unrounded square-root phase.
pub(crate) fn bessel_jy_sqrt(
    order: crate::support::wide::Wide,
    squared_half_argument: f64,
    work: &mut Work,
) -> Result<Option<[(f64, f64); 2]>, SciOxideError> {
    super::bessel_kernel::sqrt_hankel(order, squared_half_argument, work)
}

use crate::support::project_real as narrow;

// Real implementation of the public crate::bessel_j contract.
pub(crate) fn bessel_j<S: RealScalar>(nu: S, x: S, work: &mut Work) -> Result<S, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || bessel_j(nu.into_f64(), x.into_f64(), work),
        narrow::<S>,
    ) {
        return result;
    }
    work.try_step()?;
    require_finite_param(nu, "Bessel parameter order must be finite")?;
    require_finite_param(x, "Bessel parameter x must be finite")?;
    let integer_order = is_exact_integer_order(nu);

    if x < S::ZERO {
        if !integer_order {
            return Err(SciOxideError::domain_error(
                scioxide_core::argument!("x"),
                scioxide_core::DomainReason::OutsideDomain,
                "Bessel argument must be non-negative for noninteger order",
            ));
        }
        return check_finite(cos_pi(nu) * bessel_j(nu, -x, work)?);
    }

    // J_0(0) = 1, J_ν(0) = 0 for ν > 0
    if is_exact_zero(x) {
        return if is_exact_zero(nu) {
            Ok(S::ONE)
        } else if nu > S::ZERO || integer_order {
            if integer_order {
                let odd = nu % (S::TWO) != S::ZERO;
                let negative = odd && (x.is_sign_negative() ^ nu.is_sign_negative());
                Ok(if negative { -S::ZERO } else { S::ZERO })
            } else {
                Ok(S::ZERO)
            }
        } else {
            Err(SciOxideError::domain_error(
                scioxide_core::argument!("x"),
                scioxide_core::DomainReason::Pole,
                "Bessel J at zero diverges for negative noninteger order",
            ))
        };
    }

    if integer_order && nu < S::ZERO {
        return check_finite(cos_pi(nu) * bessel_j(-nu, x, work)?);
    }

    if nu.abs() == S::HALF {
        return Ok(
            ((S::TWO) / S::PI).sqrt() / x.sqrt() * (if nu > S::ZERO { x.sin() } else { x.cos() })
        );
    }
    if nu < -S::HALF {
        let positive = -nu;
        return check_finite(
            cos_pi(positive) * bessel_j(positive, x, work)?
                - sin_pi(positive) * bessel_y(positive, x, work)?,
        );
    }
    let value = crate::bessel_airy::bessel_kernel::j(nu.into_f64(), x.into_f64(), work)?;
    S::try_from_f64_lossy(check_finite(value)?)
}

// Real implementation of the public crate::bessel_y contract.
pub(crate) fn bessel_y<S: RealScalar>(nu: S, x: S, work: &mut Work) -> Result<S, SciOxideError> {
    if let Some(result) = crate::support::delegate_f32::<S, _, _>(
        || bessel_y(nu.into_f64(), x.into_f64(), work),
        narrow::<S>,
    ) {
        return result;
    }
    work.try_step()?;
    require_finite_param(nu, "Bessel parameter order must be finite")?;
    check_positive(
        x,
        scioxide_core::argument!("x"),
        "Bessel parameter x must be positive",
    )?;
    if nu < -S::HALF {
        let positive = -nu;
        return check_finite(
            cos_pi(positive) * bessel_y(positive, x, work)?
                + sin_pi(positive) * bessel_j(positive, x, work)?,
        );
    }
    if nu == S::HALF {
        return check_finite(-((S::TWO) / S::PI).sqrt() / x.sqrt() * x.cos());
    }
    let value = crate::bessel_airy::bessel_kernel::y(nu.into_f64(), x.into_f64(), work)?;
    S::try_from_f64_lossy(check_finite(value)?)
}

/// `f64` Bessel evaluation that keeps range and connection conditioning.
#[derive(Clone, Copy)]
pub(crate) struct RealEstimate {
    pub(crate) value: f64,
    pub(crate) error: f64,
    pub(crate) logarithm: Option<crate::support::wide::Wide>,
    pub(crate) sign: f64,
    pub(crate) relative: f64,
}

impl RealEstimate {
    pub(super) fn direct(value: f64) -> Self {
        Self {
            value,
            error: 64.0 * f64::EPSILON * value.abs(),
            logarithm: None,
            sign: value.signum(),
            relative: 64.0 * f64::EPSILON,
        }
    }
}

/// I and K share validation, destination conversion and ordinary/scaled dispatch.
pub(crate) fn modified(
    nu: f64,
    x: f64,
    second: bool,
    scaled: bool,
    work: &mut Work,
) -> Result<RealEstimate, SciOxideError> {
    require_finite_param(nu, "Bessel parameter order must be finite")?;
    require_finite_param(x, "Bessel parameter x must be finite")?;
    work.try_step()?;
    let integer = nu == nu.round();
    if x < 0.0 && (second || !integer) {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::OutsideDomain,
            if second {
                "Bessel argument must be non-negative"
            } else {
                "Bessel argument must be non-negative for noninteger order"
            },
        ));
    }
    if x == 0.0 {
        if second {
            return Ok(RealEstimate::direct(f64::INFINITY));
        }
        if nu < 0.0 && !integer {
            return Err(SciOxideError::domain_error(
                scioxide_core::argument!("x"),
                scioxide_core::DomainReason::Pole,
                "Bessel I at zero diverges for negative noninteger order",
            ));
        }
        return Ok(RealEstimate::direct(if nu == 0.0 {
            1.0
        } else {
            let negative = integer && nu % 2.0 != 0.0 && x.is_sign_negative();
            if negative { -0.0 } else { 0.0 }
        }));
    }
    let a = x.abs();
    let order = nu.abs();
    let mut result = if order >= 32.0 {
        modified_uniform(nu, a, second, scaled, work)?
    } else if second {
        RealEstimate::direct(super::bessel_kernel::k(order, a, scaled, work)?)
    } else {
        let value = super::bessel_kernel::i(order, a, scaled, work)?;
        if nu < 0.0 && !integer {
            let k = super::bessel_kernel::k(order, a, true, work)?;
            let correction = (2.0 / core::f64::consts::PI)
                * sin_pi(order)
                * k
                * (if scaled { -a - a } else { -a }).exp();
            let sum = crate::support::wide::Wide::from(value)
                + crate::support::wide::Wide::from(correction);
            let mut estimate = RealEstimate::direct(sum.value());
            estimate.error = 64.0 * f64::EPSILON * (value.abs() + correction.abs());
            estimate.relative = estimate.error / estimate.value.abs();
            estimate
        } else {
            RealEstimate::direct(value)
        }
    };
    if x < 0.0 && nu % 2.0 != 0.0 {
        result.value = -result.value;
        result.sign = -result.sign;
    }
    Ok(result)
}

fn modified_value<S: RealScalar>(
    nu: S,
    x: S,
    second: bool,
    scaled: bool,
    work: &mut Work,
) -> Result<S, SciOxideError> {
    let e = modified(nu.into_f64(), x.into_f64(), second, scaled, work)?;
    if second && x == S::ZERO {
        return Ok(S::INFINITY);
    }
    narrow(e.value)
}

pub(crate) fn bessel_i<S: RealScalar>(nu: S, x: S, work: &mut Work) -> Result<S, SciOxideError> {
    modified_value(nu, x, false, false, work)
}
pub(crate) fn bessel_i_scaled<S: RealScalar>(
    nu: S,
    x: S,
    work: &mut Work,
) -> Result<S, SciOxideError> {
    modified_value(nu, x, false, true, work)
}
pub(crate) fn bessel_k_scaled<S: RealScalar>(
    nu: S,
    x: S,
    work: &mut Work,
) -> Result<S, SciOxideError> {
    modified_value(nu, x, true, true, work)
}

/// First Hankel function H_ν^(1)(x) = J_ν(x) + i Y_ν(x).
///
/// Requires finite order and finite x > 0; exact order semantics and component accuracy are
/// inherited from J and Y. Both components share the caller work budget.
///
/// # Errors
///
/// Returns structured nonfinite/domain errors for invalid inputs, iteration-limit failures when
/// the work budget is exhausted, and numerical errors for finite kernel breakdown or an
/// ordinary value outside the scalar range.
pub(crate) fn hankel_1<S: RealScalar>(
    nu: S,
    x: S,
    work: &mut Work,
) -> Result<Complex<S>, SciOxideError> {
    work.try_step()?;
    require_finite_param(nu, "Bessel parameter order must be finite")?;
    check_positive(
        x,
        scioxide_core::argument!("x"),
        "Bessel parameter x must be positive",
    )?;
    let j = bessel_j(nu, x, work)?;
    let y = bessel_y(nu, x, work)?;
    Ok(Complex::new(j, y))
}

/// Second Hankel function H_ν^(2)(x) = J_ν(x) - i Y_ν(x).
///
/// Requires finite order and finite x > 0; exact order semantics and component accuracy are
/// inherited from J and Y. Both components share the caller work budget.
///
/// # Errors
///
/// Returns structured nonfinite/domain errors for invalid inputs, iteration-limit failures when
/// the work budget is exhausted, and numerical errors for finite kernel breakdown or an
/// ordinary value outside the scalar range.
pub(crate) fn hankel_2<S: RealScalar>(
    nu: S,
    x: S,
    work: &mut Work,
) -> Result<Complex<S>, SciOxideError> {
    work.try_step()?;
    require_finite_param(nu, "Bessel parameter order must be finite")?;
    check_positive(
        x,
        scioxide_core::argument!("x"),
        "Bessel parameter x must be positive",
    )?;
    let j = bessel_j(nu, x, work)?;
    let y = bessel_y(nu, x, work)?;
    Ok(Complex::new(j, -y))
}

fn modified_uniform(
    nu: f64,
    x: f64,
    second: bool,
    scaled: bool,
    work: &mut Work,
) -> Result<RealEstimate, SciOxideError> {
    use crate::support::wide::Wide as D;
    let order = nu.abs();
    let log = super::bessel_base::uniform_log(order, x, second, scaled, work)?;
    if second || nu >= 0.0 || nu == nu.round() {
        let mut result = RealEstimate::direct(super::bessel_base::project_log(log, work)?);
        result.logarithm = Some(log);
        return Ok(result);
    }
    let k = super::bessel_base::uniform_log(order, x, true, true, work)?;
    let pi = crate::support::constants::PI;
    let sine = sin_pi(order);
    let factor = D::from(2.0 * sine.abs()) / pi;
    let other = k + factor.ln() - D::from(x).mul_f64(if scaled { 2.0 } else { 1.0 });
    let common = if log.hi >= other.hi { log } else { other };
    let first = (log - common).exp_wide();
    let second = (other - common).exp_wide().mul_f64(sine.signum());
    let sum = first + second;
    let error = 64.0 * f64::EPSILON * (first.value().abs() + second.value().abs());
    let value = crate::support::wide::project(sum, common);
    let sign = sum.value().signum();
    Ok(RealEstimate {
        value,
        error: crate::support::wide::project(D::from(error), common).min(f64::MAX),
        logarithm: (sum.value() != 0.0).then(|| common + sum.mul_f64(sign).ln()),
        sign,
        relative: error / sum.value().abs(),
    })
}

/// Log magnitude and sign shared with complex-axis and logarithmic APIs.
pub(crate) fn modified_uniform_log(
    nu: f64,
    x: f64,
    second: bool,
    scaled: bool,
    work: &mut Work,
) -> Result<(f64, f64), SciOxideError> {
    let e = modified_uniform(nu, x, second, scaled, work)?;
    Ok((e.logarithm.map_or(f64::NEG_INFINITY, |l| l.value()), e.sign))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn exact_integer_classification_excludes_adjacent_values() {
        for order in [-1.0_f64, 0.0, 1.0] {
            assert!(is_exact_integer_order(order));
            assert!(!is_exact_integer_order(order.next_down()));
            assert!(!is_exact_integer_order(order.next_up()));
        }
        for order in [-1.0_f32, 0.0, 1.0] {
            assert!(is_exact_integer_order(order));
            assert!(!is_exact_integer_order(order.next_down()));
            assert!(!is_exact_integer_order(order.next_up()));
        }
    }

    #[test]
    fn test_bessel_series_reports_iteration_exhaustion() {
        {
            let outcome = &(crate::bessel_airy::bessel_kernel::j(0.5, 1.0, &mut Work::new(0)));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
            ));
        }
    }

    fn tol_f64() -> f64 {
        1e-10
    }

    fn tol_f32() -> f32 {
        1e-4
    }

    // ---- bessel_j ----

    #[test]
    fn test_bessel_j_zero_at_origin() {
        let j0 = bessel_j(0.0_f64, 0.0, &mut Work::new(65536)).unwrap();
        assert!((j0 - 1.0).abs() < tol_f64(), "J_0(0) = 1, got {j0}");
    }

    #[test]
    fn test_bessel_j_1_at_origin() {
        let j1 = bessel_j(1.0_f64, 0.0, &mut Work::new(65536)).unwrap();
        assert!(j1.abs() < tol_f64(), "J_1(0) = 0, got {j1}");
    }

    #[test]
    fn test_bessel_j_0_at_1() {
        // NIST: J_0(1) ≈ 0.7651976865579666
        let val = bessel_j(0.0_f64, 1.0, &mut Work::new(65536)).unwrap();
        assert!(
            (val - 0.765_197_686_557_966_6).abs() < tol_f64(),
            "J_0(1) = 0.7651976865579666, got {val}"
        );
    }

    #[test]
    fn test_bessel_j_1_at_1() {
        // NIST: J_1(1) ≈ 0.4400505857449335
        let val = bessel_j(1.0_f64, 1.0, &mut Work::new(65536)).unwrap();
        assert!(
            (val - 0.440_050_585_744_933_5).abs() < tol_f64(),
            "J_1(1) ≈ 0.4400505857449335, got {val}"
        );
    }

    #[test]
    fn test_bessel_j_0_at_5() {
        // NIST: J_0(5) ≈ -0.1775967713143383
        let val = bessel_j(0.0_f64, 5.0, &mut Work::new(65536)).unwrap();
        assert!(
            (val - (-0.177_596_771_314_338_3)).abs() < tol_f64(),
            "J_0(5) ≈ -0.1776, got {val}"
        );
    }

    #[test]
    fn test_bessel_j_half_order() {
        // J_{0.5}(x) = sqrt(2/(πx)) sin(x)
        let x = 2.0_f64;
        let val = bessel_j(0.5, x, &mut Work::new(65536)).unwrap();
        let expected = (2.0 / (std::f64::consts::PI * x)).sqrt() * x.sin();
        assert!(
            (val - expected).abs() < tol_f64(),
            "J_0.5(2) ≈ {expected}, got {val}"
        );
    }

    #[test]
    fn test_bessel_j_asymptotic_region() {
        let val = bessel_j(0.0_f64, 30.0, &mut Work::new(65536)).unwrap();
        assert!(val.abs() < 0.15, "J_0(30) should be small, got {val}");
        assert!(val.is_finite());
    }

    #[test]
    fn test_bessel_j_negative_noninteger_order_at_zero() {
        let result = bessel_j(-0.5_f64, 0.0, &mut Work::new(65536));
        {
            let outcome = &(result);
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
    }

    #[test]
    fn test_bessel_j_f32() {
        let val = bessel_j(0.0_f32, 1.0, &mut Work::new(65536)).unwrap();
        assert!(
            (val - 0.765_197_7_f32).abs() < tol_f32(),
            "J_0(1) f32, got {val}"
        );
    }

    #[test]
    fn test_bessel_j_negative_integer_order_reflection() {
        let x = 1.25_f64;
        let j2 = bessel_j(2.0_f64, x, &mut Work::new(65536)).unwrap();
        let j_neg2 = bessel_j(-2.0_f64, x, &mut Work::new(65536)).unwrap();
        assert!((j_neg2 - j2).abs() < 1e-12, "J_-2(x) should equal J_2(x)");

        let j1 = bessel_j(1.0_f64, x, &mut Work::new(65536)).unwrap();
        let j_neg1 = bessel_j(-1.0_f64, x, &mut Work::new(65536)).unwrap();
        assert!((j_neg1 + j1).abs() < 1e-12, "J_-1(x) should equal -J_1(x)");
    }

    #[test]
    fn test_bessel_j_negative_x_integer_parity() {
        let x = 1.0_f64;
        let j1_pos = bessel_j(1.0_f64, x, &mut Work::new(65536)).unwrap();
        let j1_neg = bessel_j(1.0_f64, -x, &mut Work::new(65536)).unwrap();
        assert!(
            (j1_neg + j1_pos).abs() < 1e-12,
            "J_1(-x) should equal -J_1(x)"
        );
    }

    #[test]
    fn test_bessel_j_negative_x_noninteger_domain_error() {
        let result = bessel_j(0.5_f64, -1.0, &mut Work::new(65536));
        {
            let outcome = &(result);
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
    }

    #[test]
    fn test_bessel_j_2_at_0() {
        let val = bessel_j(2.0_f64, 0.0, &mut Work::new(65536)).unwrap();
        assert!(val.abs() < tol_f64(), "J_2(0) = 0, got {val}");
    }

    // ---- bessel_y ----

    #[test]
    fn test_bessel_y_0_at_1() {
        // NIST: Y_0(1) ≈ 0.0882569642156769
        let val = bessel_y(0.0_f64, 1.0, &mut Work::new(65536)).unwrap();
        assert!(
            (val - 0.088_256_964_215_676_9).abs() < tol_f64(),
            "Y_0(1) ≈ 0.0882569642156769, got {val}"
        );
    }

    #[test]
    fn test_bessel_y_1_at_1() {
        // NIST: Y_1(1) ≈ -0.7812128213002887
        let val = bessel_y(1.0_f64, 1.0, &mut Work::new(65536)).unwrap();
        assert!(
            (val - (-0.781_212_821_300_288_7)).abs() < 1e-8,
            "Y_1(1) ≈ -0.7812128213002887, got {val}"
        );
    }

    #[test]
    fn test_bessel_y_domain_error_nonpositive() {
        let result = bessel_y(0.0_f64, 0.0, &mut Work::new(65536));
        {
            let outcome = &(result);
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }

        let result = bessel_y(0.0_f64, -1.0, &mut Work::new(65536));
        {
            let outcome = &(result);
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
    }

    #[test]
    fn test_bessel_y_asymptotic_region() {
        let val = bessel_y(0.0_f64, 30.0, &mut Work::new(65536)).unwrap();
        assert!(val.is_finite());
        assert!(val.abs() < 0.2);
    }

    #[test]
    fn test_bessel_y_f32() {
        let val = bessel_y(0.0_f32, 1.0, &mut Work::new(65536)).unwrap();
        assert!(
            (val - 0.088_256_97_f32).abs() < tol_f32(),
            "Y_0(1) f32, got {val}"
        );
    }

    #[test]
    fn test_bessel_y_order_2() {
        // NIST: Y_2(1) ≈ -1.6506826068162544
        let val = bessel_y(2.0_f64, 1.0, &mut Work::new(65536)).unwrap();
        assert!(
            (val - (-1.650_682_606_816_254_4)).abs() < 1e-7,
            "Y_2(1) ≈ -1.6506826068162544, got {val}"
        );
    }

    #[test]
    fn test_bessel_y_negative_integer_order_reflection() {
        let x = 1.0_f64;
        let y2 = bessel_y(2.0_f64, x, &mut Work::new(65536)).unwrap();
        let y_neg2 = bessel_y(-2.0_f64, x, &mut Work::new(65536)).unwrap();
        assert!((y_neg2 - y2).abs() < 1e-10, "Y_-2(x) should equal Y_2(x)");

        let y1 = bessel_y(1.0_f64, x, &mut Work::new(65536)).unwrap();
        let y_neg1 = bessel_y(-1.0_f64, x, &mut Work::new(65536)).unwrap();
        assert!((y_neg1 + y1).abs() < 1e-10, "Y_-1(x) should equal -Y_1(x)");
    }

    // ---- bessel_i ----

    #[test]
    fn test_bessel_i_0_at_0() {
        let val = bessel_i(0.0_f64, 0.0, &mut Work::new(65536)).unwrap();
        assert!((val - 1.0).abs() < tol_f64(), "I_0(0) = 1, got {val}");
    }

    #[test]
    fn test_bessel_i_0_at_1() {
        // NIST: I_0(1) ≈ 1.2660658777520082
        let val = bessel_i(0.0_f64, 1.0, &mut Work::new(65536)).unwrap();
        assert!(
            (val - 1.266_065_877_752_008_2).abs() < tol_f64(),
            "I_0(1) ≈ 1.2660658777520082, got {val}"
        );
    }

    #[test]
    fn test_bessel_i_1_at_0() {
        let val = bessel_i(1.0_f64, 0.0, &mut Work::new(65536)).unwrap();
        assert!(val.abs() < tol_f64(), "I_1(0) = 0, got {val}");
    }

    #[test]
    fn test_bessel_i_1_at_1() {
        // NIST: I_1(1) ≈ 0.5651591039924850
        let val = bessel_i(1.0_f64, 1.0, &mut Work::new(65536)).unwrap();
        assert!(
            (val - 0.565_159_103_992_485).abs() < tol_f64(),
            "I_1(1) ≈ 0.5651591039924850, got {val}"
        );
    }

    #[test]
    fn test_bessel_i_f32() {
        let val = bessel_i(0.0_f32, 1.0, &mut Work::new(65536)).unwrap();
        assert!(
            (val - 1.266_066_f32).abs() < tol_f32(),
            "I_0(1) f32, got {val}"
        );
    }

    #[test]
    fn test_bessel_i_negative_integer_order_reflection() {
        let x = 1.0_f64;
        let i2 = bessel_i(2.0_f64, x, &mut Work::new(65536)).unwrap();
        let i_neg2 = bessel_i(-2.0_f64, x, &mut Work::new(65536)).unwrap();
        assert!((i_neg2 - i2).abs() < 1e-12, "I_-2(x) should equal I_2(x)");
    }

    #[test]
    fn test_bessel_i_negative_x_integer_parity() {
        let x = 0.75_f64;
        let i1_pos = bessel_i(1.0_f64, x, &mut Work::new(65536)).unwrap();
        let i1_neg = bessel_i(1.0_f64, -x, &mut Work::new(65536)).unwrap();
        assert!(
            (i1_neg + i1_pos).abs() < 1e-12,
            "I_1(-x) should equal -I_1(x)"
        );
    }

    #[test]
    fn test_bessel_i_negative_x_noninteger_domain_error() {
        let result = bessel_i(0.5_f64, -1.0, &mut Work::new(65536));
        {
            let outcome = &(result);
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
    }

    // ---- bessel_k ----

    #[test]
    fn test_bessel_k_0_at_1() {
        // NIST: K_0(1) ≈ 0.4210244382407084
        let val = modified_value(0.0_f64, 1.0, true, false, &mut Work::new(65536)).unwrap();
        assert!(
            (val - 0.421_024_438_240_708_4).abs() < tol_f64(),
            "K_0(1) ≈ 0.4210244382407084, got {val}"
        );
    }

    #[test]
    fn test_bessel_k_1_at_1() {
        // NIST: K_1(1) ≈ 0.6019072301972346
        let val = modified_value(1.0_f64, 1.0, true, false, &mut Work::new(65536)).unwrap();
        assert!(
            (val - 0.601_907_230_197_234_6).abs() < 1e-8,
            "K_1(1) ≈ 0.6019072301972346, got {val}"
        );
    }

    #[test]
    fn test_bessel_k_endpoint_and_negative_domain() {
        assert_eq!(
            modified_value(0.0_f64, 0.0, true, false, &mut Work::new(65536)).unwrap(),
            f64::INFINITY
        );

        let result = modified_value(0.0_f64, -1.0, true, false, &mut Work::new(65536));
        {
            let outcome = &(result);
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
    }

    #[test]
    fn test_scaled_modified_bessel_large_arguments() {
        let i_scaled = bessel_i_scaled(0.0_f64, 1_000.0, &mut Work::new(65536)).unwrap();
        let k_scaled = bessel_k_scaled(0.0_f64, 1_000.0, &mut Work::new(65536)).unwrap();
        assert!(i_scaled.is_finite() && i_scaled > 0.0);
        assert!(k_scaled.is_finite() && k_scaled > 0.0);
        {
            let outcome = &(bessel_i(0.0_f64, 1_000.0, &mut Work::new(65536)));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::Overflow,
                    ..
                }
            ));
        }
        assert_eq!(
            modified_value(0.0_f64, 1_000.0, true, false, &mut Work::new(65536))
                .unwrap()
                .to_bits(),
            0.0_f64.to_bits()
        );
    }

    #[test]
    fn test_modified_bessel_selection_is_order_aware() {
        let i = bessel_i(100.0_f64, 10.0, &mut Work::new(65536)).unwrap();
        let k = modified_value(100.0_f64, 10.0, true, false, &mut Work::new(65536)).unwrap();
        let i_scaled = bessel_i_scaled(100.0_f64, 10.0, &mut Work::new(65536)).unwrap();
        let k_scaled = bessel_k_scaled(100.0_f64, 10.0, &mut Work::new(65536)).unwrap();

        let expected_i = 1.082_344_201_749_201_6e-88;
        let expected_k = 4.596_674_084_269_528e85;
        assert!(i > 0.0 && ((i - expected_i) / expected_i).abs() < 2e-13);
        assert!(k > 0.0 && ((k - expected_k) / expected_k).abs() < 2e-8);
        assert!(((i_scaled - i * (-10.0_f64).exp()) / i_scaled).abs() < 2e-13);
        assert!(((k_scaled - k * 10.0_f64.exp()) / k_scaled).abs() < 2e-13);
    }

    #[test]
    fn test_bessel_signed_zero_and_work_bound() {
        assert!(
            bessel_j(1.0_f64, -0.0, &mut Work::new(65536))
                .unwrap()
                .is_sign_negative()
        );
        assert!(
            bessel_i(1.0_f64, -0.0, &mut Work::new(65536))
                .unwrap()
                .is_sign_negative()
        );
        assert!(
            !bessel_j(2.0_f64, -0.0, &mut Work::new(65536))
                .unwrap()
                .is_sign_negative()
        );

        let excessive = 1000.0_f64;
        {
            let outcome = &(bessel_y(excessive, 1.0_f64, &mut Work::new(1)));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
            ));
        }
        {
            let outcome = &(modified_value(excessive, 1.0_f64, true, false, &mut Work::new(1)));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
            ));
        }
    }

    #[test]
    fn test_bessel_k_symmetry() {
        // K_ν = K_{-ν}
        let k_pos = modified_value(0.5_f64, 2.0, true, false, &mut Work::new(65536)).unwrap();
        let k_neg = modified_value(-0.5_f64, 2.0, true, false, &mut Work::new(65536)).unwrap();
        assert!(
            (k_pos - k_neg).abs() < tol_f64(),
            "K_0.5(2) = K_-0.5(2), got {k_pos} vs {k_neg}"
        );
    }

    #[test]
    fn test_bessel_k_f32() {
        let val = modified_value(0.0_f32, 1.0, true, false, &mut Work::new(65536)).unwrap();
        assert!(
            (val - 0.421_024_4_f32).abs() < tol_f32(),
            "K_0(1) f32, got {val}"
        );
    }

    #[test]
    fn test_bessel_k_scaled_f32_transition() {
        let order = f32::from_bits(0x3e61_4df8);
        let argument = f32::from_bits(0x4090_d636);
        let expected = f32::from_bits(0x3f13_ca6c);
        let actual = bessel_k_scaled(order, argument, &mut Work::new(65536)).unwrap();
        assert!((actual - expected).abs() <= 1e-3 * expected.abs());
    }

    #[test]
    fn test_bessel_k_order_2() {
        // NIST: K_2(1) ≈ 1.6248388986351774
        let val = modified_value(2.0_f64, 1.0, true, false, &mut Work::new(65536)).unwrap();
        assert!(
            (val - 1.624_838_898_635_177_4).abs() < 1e-6,
            "K_2(1) ≈ 1.6248388986351774, got {val}"
        );
    }

    // ---- Wronskian ----

    #[test]
    fn test_wronskian_j_y() {
        // J_ν(x) Y_{ν-1}(x) - J_{ν-1}(x) Y_ν(x) = 2/(πx)
        let x = 3.0_f64;
        let pi = std::f64::consts::PI;

        for n in 1..=3 {
            let order = f64::from(n);
            let j_n = bessel_j(order, x, &mut Work::new(65536)).unwrap();
            let j_nm1 = bessel_j(order - 1.0, x, &mut Work::new(65536)).unwrap();
            let y_n = bessel_y(order, x, &mut Work::new(65536)).unwrap();
            let y_nm1 = bessel_y(order - 1.0, x, &mut Work::new(65536)).unwrap();

            let wronskian = j_n.mul_add(y_nm1, -(j_nm1 * y_n));
            let expected = 2.0 / (pi * x);
            assert!(
                (wronskian - expected).abs() < 1e-8,
                "Wronskian at n={n}, x={x}: expected {expected}, got {wronskian}"
            );
        }
    }

    // ---- Hankel functions ----

    #[test]
    fn test_hankel_1_definition() {
        let x = 2.0_f64;
        let order = 1.0_f64;
        let h = hankel_1(order, x, &mut Work::new(65536)).unwrap();
        let j = bessel_j(order, x, &mut Work::new(65536)).unwrap();
        let y = bessel_y(order, x, &mut Work::new(65536)).unwrap();
        assert!((h.re - j).abs() < tol_f64());
        assert!((h.im - y).abs() < tol_f64());
    }

    #[test]
    fn test_hankel_2_definition() {
        let x = 2.0_f64;
        let order = 1.0_f64;
        let h = hankel_2(order, x, &mut Work::new(65536)).unwrap();
        let j = bessel_j(order, x, &mut Work::new(65536)).unwrap();
        let y = bessel_y(order, x, &mut Work::new(65536)).unwrap();
        assert!((h.re - j).abs() < tol_f64());
        assert!((h.im - (-y)).abs() < tol_f64());
    }

    #[test]
    fn test_hankel_1_at_order_0() {
        let h = hankel_1(0.0_f64, 1.0, &mut Work::new(65536)).unwrap();
        assert!((h.re - 0.765_197_686_557_966_6).abs() < tol_f64());
        assert!((h.im - 0.088_256_964_215_676_9).abs() < tol_f64());
    }

    #[test]
    fn test_hankel_2_at_order_0() {
        let h = hankel_2(0.0_f64, 1.0, &mut Work::new(65536)).unwrap();
        assert!((h.re - 0.765_197_686_557_966_6).abs() < tol_f64());
        assert!((h.im - (-0.088_256_964_215_676_9)).abs() < tol_f64());
    }

    #[test]
    fn test_hankel_domain_error() {
        let result = hankel_1(0.0_f64, 0.0, &mut Work::new(65536));
        assert!(result.is_err());

        let result = hankel_2(0.0_f64, -1.0, &mut Work::new(65536));
        assert!(result.is_err());
    }

    #[test]
    fn test_hankel_f32() {
        let h1 = hankel_1(0.0_f32, 1.0, &mut Work::new(65536)).unwrap();
        assert!((h1.re - 0.765_197_7_f32).abs() < tol_f32());
        assert!((h1.im - 0.088_257_f32).abs() < tol_f32());

        let h2 = hankel_2(0.0_f32, 1.0, &mut Work::new(65536)).unwrap();
        assert!((h2.re - 0.765_197_7_f32).abs() < tol_f32());
        assert!((h2.im - (-0.088_257_f32)).abs() < tol_f32());
    }
}
