//! Rational arithmetic contracts.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_poly::PolynomialBasis;
use scioxide_poly::{ApproxPolynomial, ApproxRationalFunction, PolynomialApproximationPolicy};

fn arithmetic<C: NumericScalar>() {
    let c = |x| C::from_real(C::Real::try_from_f64_lossy(x).unwrap());
    let policy = PolynomialApproximationPolicy::default();
    let p = |v: &[f64]| {
        ApproxPolynomial::monomial_with_options(v.iter().map(|&x| c(x)).collect(), policy).unwrap()
    };
    let a = ApproxRationalFunction::new(p(&[2.0, 2.0]), p(&[4.0, 2.0])).unwrap();
    let b = ApproxRationalFunction::new(p(&[3.0]), p(&[1.0, 1.0])).unwrap();
    assert_eq!(a.denominator().leading_coefficient(), c(1.0));
    let x = c(3.0);
    for (actual, expected) in [
        (a.try_add(&b).unwrap(), 1.55),
        (a.try_sub(&b).unwrap(), 0.05),
        (a.try_mul(&b).unwrap(), 0.6),
        (a.try_div(&b).unwrap(), 0.8 / 0.75),
        (a.reciprocal().unwrap(), 1.25),
    ] {
        assert!(
            (actual.evaluate(x).unwrap() - c(expected)).abs()
                < C::Real::try_from_usize_exact(32).unwrap() * C::Real::EPSILON
        );
    }
    let zero = a.try_sub(&a).unwrap();
    assert!(zero.numerator().is_zero());
    assert_eq!(zero.denominator().basis().coefficients(), &[c(1.0)]);
    assert!(matches!(
        (zero.reciprocal()).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    assert!(a.try_div(&zero).is_err());
    // Cancellation must remain a separately requested operation.
    let same = ApproxRationalFunction::new(p(&[-1.0, 1.0]), p(&[-1.0, 1.0])).unwrap();
    assert_eq!(same.denominator().degree(), 1);
    assert!(same.evaluate(c(1.0)).is_err());
    assert_eq!(
        same.simplify().unwrap().rational().denominator().degree(),
        0
    );
}

#[test]
fn all_numeric_domains() {
    arithmetic::<f32>();
    arithmetic::<f64>();
    arithmetic::<Complex<f32>>();
    arithmetic::<Complex<f64>>();
}

#[test]
fn normalization_does_not_form_an_overflowing_reciprocal() {
    let policy = PolynomialApproximationPolicy::default();
    let tiny = f64::from_bits(1);
    let p = |v| ApproxPolynomial::monomial_with_options(v, policy).unwrap();
    let r = ApproxRationalFunction::new(p(vec![tiny]), p(vec![tiny, tiny])).unwrap();
    assert_eq!(r.numerator().basis().coefficients(), &[1.0]);
    assert_eq!(r.denominator().basis().coefficients(), &[1.0, 1.0]);
    assert!(ApproxRationalFunction::new(p(vec![f64::MAX]), p(vec![tiny])).is_err());
}

#[test]
fn nearly_common_cancellation_checks_both_original_scales() {
    use scioxide_core::{RankThreshold, Tolerance};
    use scioxide_linalg::RankPolicy;
    let make = |relative| {
        let policy = PolynomialApproximationPolicy::new(
            Tolerance::new(0.0_f64, relative).unwrap(),
            RankPolicy::Threshold(RankThreshold::new(0.0, relative).unwrap()),
        );
        let a = ApproxPolynomial::from_roots_with_options(&[1.0, -2.0], policy).unwrap();
        let b = ApproxPolynomial::from_roots_with_options(&[1.0 + 1e-9, 3.0], policy).unwrap();
        ApproxRationalFunction::new(a, b).unwrap()
    };
    let loose = make(1e-6).simplify().unwrap();
    assert_eq!(loose.gcd_report().effective_degree(), 1);
    assert!(loose.numerator_error() <= 1e-6);
    assert!(loose.denominator_error() <= 1e-6);
    assert!(loose.numerator_error() > 0.0 || loose.denominator_error() > 0.0);
    let strict = make(1e-14).simplify().unwrap();
    assert_eq!(strict.gcd_report().effective_degree(), 0);
    assert_eq!(strict.rational().denominator().degree(), 2);
}

#[test]
fn poles_follow_pointwise_roundoff_instead_of_coefficient_tolerance() {
    use scioxide_core::{Complex, SciOxideErrorKind};
    let p = |c| ApproxPolynomial::monomial(c).unwrap();
    let reciprocal = ApproxRationalFunction::new(p(vec![1.0_f64]), p(vec![0., 1.])).unwrap();
    for x in [1e-9, 1e-300, 1e-308, f64::from_bits(0x000f_ffff_ffff_ffff)] {
        let value = reciprocal.evaluate(x).unwrap();
        assert!((value * x - 1.).abs() <= 4. * f64::EPSILON);
    }
    assert_eq!(
        reciprocal.evaluate(0.).unwrap_err().kind(),
        SciOxideErrorKind::DomainError
    );
    let overflow = reciprocal.evaluate(f64::from_bits(1)).unwrap_err();
    assert_ne!(overflow.kind(), SciOxideErrorKind::DomainError);

    let near_root = ApproxRationalFunction::new(p(vec![1.]), p(vec![-1., 1.])).unwrap();
    assert_eq!(
        near_root.evaluate(1. + f64::EPSILON).unwrap_err().kind(),
        SciOxideErrorKind::DomainError
    );
    assert!(near_root.evaluate(1. + 1e-9).is_ok());

    let p = |c| ApproxPolynomial::monomial(c).unwrap();
    let reciprocal_f32 = ApproxRationalFunction::new(p(vec![1.0_f32]), p(vec![0., 1.])).unwrap();
    for x in [1e-9_f32, 1e-38, f32::from_bits(0x007f_ffff)] {
        assert!((reciprocal_f32.evaluate(x).unwrap() * x - 1.).abs() <= 4. * f32::EPSILON);
    }
    assert_eq!(
        reciprocal_f32.evaluate(0.).unwrap_err().kind(),
        SciOxideErrorKind::DomainError
    );

    let c = |x| Complex::new(x, 0.);
    let p = |coefficients| ApproxPolynomial::monomial(coefficients).unwrap();
    let reciprocal =
        ApproxRationalFunction::new(p(vec![c(1.0_f64)]), p(vec![c(0.), c(1.)])).unwrap();
    let x = Complex::new(1e-308, 1e-308);
    let value = reciprocal.evaluate(x).unwrap();
    let residual = value * x - c(1.);
    assert!(residual.re.abs() + residual.im.abs() <= 8. * f64::EPSILON);
}

#[test]
fn value_and_parts_share_the_pole_test_and_range_fallback() {
    for (numerator, denominator, points) in [
        (
            vec![1.0_f64, 2.0, 3.0],
            vec![1.0, -0.5],
            vec![-1.0, 0.0, 0.5, 2.0],
        ),
        (
            vec![1e300, 1e300, -1e300],
            vec![1.0, 1.0],
            vec![0.0, 1.0, -1.0],
        ),
        (vec![1.0], vec![1e-14], vec![0.0, 1.0]),
        (vec![1.0], vec![1.0, -2.0, 1.0], vec![1.0, 1.0 + 1e-8, 0.5]),
    ] {
        let function = ApproxRationalFunction::new(
            ApproxPolynomial::monomial(numerator).unwrap(),
            ApproxPolynomial::monomial(denominator).unwrap(),
        )
        .unwrap();
        for point in points {
            let value = function.evaluate(point);
            let full = function.evaluate_parts(point);
            assert_eq!(
                value.as_ref().err().map(|error| error.kind()),
                full.as_ref().err().map(|error| error.kind())
            );
            if let (Ok(value), Ok(full)) = (value, full) {
                assert!((value - full.value()).abs() <= 1e-13 * full.value().abs().max(1.0));
            }
        }
    }
}
