//! Accuracy, construction, and allocation regressions for prepared classical rules.

#![allow(clippy::cast_precision_loss, clippy::unwrap_used)]

use scioxide_integrate::{ClenshawCurtis, FixedRule, GaussianRulePlan, GaussianSupport};
use scioxide_poly::OrthogonalFamily;
use std::sync::atomic::{AtomicUsize, Ordering};

use scioxide_core::{Interval, RealScalar, SciOxideError};
use scioxide_special::{beta, binomial};

fn assert_close_f64(actual: f64, expected: f64, relative_tolerance: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= relative_tolerance * scale,
        "{actual} != {expected} within relative tolerance {relative_tolerance}"
    );
}

fn assert_close_f32(actual: f32, expected: f32, relative_tolerance: f32) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= relative_tolerance * scale,
        "{actual} != {expected} within relative tolerance {relative_tolerance}"
    );
}

/// Weighted integral of `f` over `support`.
fn weighted<S: RealScalar + Default>(
    plan: &GaussianRulePlan<S>,
    f: impl Fn(S) -> S + Sync,
    support: GaussianSupport<S>,
) -> S {
    plan.integrate_weighted(|x: S| Ok(f(x)), support).unwrap().0
}

/// Unweighted integral of `f` over `interval` through the fixed-rule protocol.
fn unweighted<S: RealScalar + Default, R: FixedRule<S>>(
    rule: &R,
    f: impl Fn(S) -> S + Sync,
    interval: Interval<S>,
) -> S {
    rule.integrate(|x: S| Ok(f(x)), interval).unwrap().0
}

fn legendre_moment(degree: usize) -> f64 {
    if degree.is_multiple_of(2) {
        2.0 / (degree + 1) as f64
    } else {
        0.0
    }
}

fn chebyshev_moment(degree: usize) -> f64 {
    if !degree.is_multiple_of(2) {
        return 0.0;
    }
    let half = degree / 2;
    let central = binomial::<f64>(degree, half).unwrap();
    std::f64::consts::PI * central / 4.0_f64.powi(half as i32)
}

fn hermite_moment(degree: usize) -> f64 {
    if !degree.is_multiple_of(2) {
        return 0.0;
    }
    let double_factorial = (1..=degree)
        .filter(|factor| !factor.is_multiple_of(2))
        .map(|factor| factor as f64)
        .product::<f64>();
    (2.0 * std::f64::consts::PI).sqrt() * double_factorial
}

fn laguerre_moment(degree: usize) -> f64 {
    (1..=degree).map(|factor| factor as f64).product()
}

fn jacobi_moment(degree: usize, alpha: f64, beta_parameter: f64) -> f64 {
    if alpha == beta_parameter && !degree.is_multiple_of(2) {
        return 0.0;
    }
    let transformed = (0..=degree).try_fold(0.0, |sum, power| {
        let choose = binomial::<f64>(degree, power)?;
        let sign = if (degree - power).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        let beta_value = beta(beta_parameter + power as f64 + 1.0, alpha + 1.0)?;
        Ok::<f64, SciOxideError>(sum + sign * choose * 2.0_f64.powi(power as i32) * beta_value)
    });
    2.0_f64.powf(alpha + beta_parameter + 1.0) * transformed.unwrap()
}

#[test]
fn gaussian_families_match_every_documented_moment_for_both_scalars() {
    const ORDER: usize = 6;
    let canonical_f64 = GaussianSupport::Interval(Interval::new(-1.0_f64, 1.0).unwrap());
    let canonical_f32 = GaussianSupport::Interval(Interval::new(-1.0_f32, 1.0).unwrap());
    let legendre = GaussianRulePlan::new(OrthogonalFamily::<f64>::Legendre, ORDER).unwrap();
    let legendre_single = GaussianRulePlan::new(OrthogonalFamily::<f32>::Legendre, ORDER).unwrap();
    let chebyshev = GaussianRulePlan::new(OrthogonalFamily::<f64>::ChebyshevT, ORDER).unwrap();
    let chebyshev_single =
        GaussianRulePlan::new(OrthogonalFamily::<f32>::ChebyshevT, ORDER).unwrap();
    let hermite = GaussianRulePlan::new(OrthogonalFamily::<f64>::HermiteHe, ORDER).unwrap();
    let hermite_single = GaussianRulePlan::new(OrthogonalFamily::<f32>::HermiteHe, ORDER).unwrap();
    let laguerre = GaussianRulePlan::new(OrthogonalFamily::<f64>::Laguerre, ORDER).unwrap();
    let laguerre_single = GaussianRulePlan::new(OrthogonalFamily::<f32>::Laguerre, ORDER).unwrap();

    for degree in 0..=(2 * ORDER - 1) {
        let exponent = degree as i32;

        assert_close_f64(
            weighted(&legendre, |x: f64| x.powi(exponent), canonical_f64),
            legendre_moment(degree),
            2.0e-12,
        );
        assert_close_f32(
            weighted(&legendre_single, |x: f32| x.powi(exponent), canonical_f32),
            legendre_moment(degree) as f32,
            3.0e-5,
        );

        assert_close_f64(
            weighted(&chebyshev, |x: f64| x.powi(exponent), canonical_f64),
            chebyshev_moment(degree),
            2.0e-12,
        );
        assert_close_f32(
            weighted(&chebyshev_single, |x: f32| x.powi(exponent), canonical_f32),
            chebyshev_moment(degree) as f32,
            4.0e-5,
        );

        assert_close_f64(
            weighted(
                &hermite,
                |x: f64| x.powi(exponent),
                GaussianSupport::RealLine,
            ),
            hermite_moment(degree),
            2.0e-11,
        );
        assert_close_f32(
            weighted(
                &hermite_single,
                |x: f32| x.powi(exponent),
                GaussianSupport::RealLine,
            ),
            hermite_moment(degree) as f32,
            2.0e-4,
        );

        assert_close_f64(
            weighted(
                &laguerre,
                |x: f64| x.powi(exponent),
                GaussianSupport::HalfLine(0.0),
            ),
            laguerre_moment(degree),
            2.0e-10,
        );
        assert_close_f32(
            weighted(
                &laguerre_single,
                |x: f32| x.powi(exponent),
                GaussianSupport::HalfLine(0.0),
            ),
            laguerre_moment(degree) as f32,
            3.0e-4,
        );
    }
}

#[test]
fn jacobi_matches_beta_moments_for_both_scalars_and_difficult_parameters() {
    const ORDER: usize = 6;
    let canonical_f64 = GaussianSupport::Interval(Interval::new(-1.0_f64, 1.0).unwrap());
    let canonical_f32 = GaussianSupport::Interval(Interval::new(-1.0_f32, 1.0).unwrap());

    for &(alpha, beta_parameter) in &[(0.5_f64, 0.5_f64), (0.25, 1.5), (-0.999, 0.0)] {
        let rule_f64 = GaussianRulePlan::new(
            OrthogonalFamily::Jacobi {
                alpha,
                beta: beta_parameter,
            },
            ORDER,
        )
        .unwrap();
        let rule_f32 = GaussianRulePlan::new(
            OrthogonalFamily::Jacobi {
                alpha: alpha as f32,
                beta: beta_parameter as f32,
            },
            ORDER,
        )
        .unwrap();
        let tolerance_f64 = if alpha < -0.99 { 5.0e-10 } else { 1.0e-10 };
        for degree in 0..=(2 * ORDER - 1) {
            let exponent = degree as i32;
            let expected = jacobi_moment(degree, alpha, beta_parameter);
            assert_close_f64(
                weighted(&rule_f64, |x: f64| x.powi(exponent), canonical_f64),
                expected,
                tolerance_f64,
            );
            assert_close_f32(
                weighted(&rule_f32, |x: f32| x.powi(exponent), canonical_f32),
                expected as f32,
                8.0e-4,
            );
        }
    }

    let semicircle = GaussianRulePlan::new(
        OrthogonalFamily::Jacobi {
            alpha: 0.5_f64,
            beta: 0.5,
        },
        5,
    )
    .unwrap();
    assert_close_f64(
        weighted(&semicircle, |_: f64| 1.0, canonical_f64),
        std::f64::consts::PI / 2.0,
        2.0e-14,
    );
    // The unweighted protocol never silently returns this weighted integral.
    let rejected = semicircle
        .integrate(|_: f64| Ok(1.0), Interval::new(-1.0, 1.0).unwrap())
        .unwrap_err();
    assert!(matches!(
        rejected.view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
}

#[test]
fn representative_high_orders_remain_accurate() {
    let canonical_f64 = Interval::new(-1.0_f64, 1.0).unwrap();
    let canonical_f32 = Interval::new(-1.0_f32, 1.0).unwrap();

    let legendre = GaussianRulePlan::new(OrthogonalFamily::<f64>::Legendre, 50).unwrap();
    for degree in [0_usize, 2, 24, 50, 99] {
        assert_close_f64(
            unweighted(&legendre, |x: f64| x.powi(degree as i32), canonical_f64),
            legendre_moment(degree),
            5.0e-12,
        );
    }

    let jacobi = GaussianRulePlan::new(
        OrthogonalFamily::Jacobi {
            alpha: 0.5_f64,
            beta: 0.5,
        },
        20,
    )
    .unwrap();
    for degree in [0_usize, 2, 10, 39] {
        assert_close_f64(
            weighted(
                &jacobi,
                |x: f64| x.powi(degree as i32),
                GaussianSupport::Interval(canonical_f64),
            ),
            jacobi_moment(degree, 0.5, 0.5),
            2.0e-10,
        );
    }

    let quartic_f64 = unweighted(
        &ClenshawCurtis::<f64>::new(64).unwrap(),
        |x: f64| x.powi(64),
        canonical_f64,
    );
    let quartic_f32 = unweighted(
        &ClenshawCurtis::<f32>::new(64).unwrap(),
        |x: f32| x.powi(64),
        canonical_f32,
    );
    assert_close_f64(quartic_f64, 2.0 / 65.0, 5.0e-13);
    assert_close_f32(quartic_f32, 2.0_f32 / 65.0, 2.0e-5);

    for family in [
        OrthogonalFamily::<f64>::HermiteH,
        OrthogonalFamily::HermiteHe,
    ] {
        let hermite = GaussianRulePlan::new(family, 200).unwrap();
        assert!(
            hermite
                .weights()
                .iter()
                .all(|&weight| weight.is_finite() && weight > 0.0)
        );
        assert!(hermite.weights()[0] < 1.0e-150);
        for degree in [0, 2, 20, 100, 200] {
            let moment = weighted(
                &hermite,
                |x: f64| x.powi(degree as i32),
                GaussianSupport::RealLine,
            );
            let expected = hermite_moment(degree)
                / if family == OrthogonalFamily::HermiteH {
                    2.0_f64.powi((degree / 2) as i32) * 2.0_f64.sqrt()
                } else {
                    1.0
                };
            assert_close_f64(
                moment,
                expected,
                if degree == 0 { 5.0e-13 } else { 2.0e-11 },
            );
        }
    }
}

#[test]
fn mapped_degenerate_shifted_and_reversed_domains_follow_existing_policy() {
    let degenerate = Interval::new(3.0_f64, 3.0).unwrap();
    let calls = AtomicUsize::new(0);
    let callback = |_: f64| -> Result<f64, SciOxideError> {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(1.0)
    };
    let zero =
        |result: Result<(f64, scioxide_integrate::IntegrationReport<f64>), SciOxideError>| {
            let (value, report) = result.unwrap();
            assert_eq!(value, 0.0);
            assert_eq!(
                report.termination(),
                scioxide_integrate::IntegrationTermination::DegenerateDomain
            );
        };

    zero(
        GaussianRulePlan::gauss_legendre(4)
            .unwrap()
            .integrate(&callback, degenerate),
    );
    zero(
        GaussianRulePlan::new(OrthogonalFamily::ChebyshevT, 4)
            .unwrap()
            .integrate_weighted(&callback, GaussianSupport::Interval(degenerate)),
    );
    zero(
        GaussianRulePlan::new(
            OrthogonalFamily::Jacobi {
                alpha: 0.25,
                beta: 0.75,
            },
            4,
        )
        .unwrap()
        .integrate_weighted(&callback, GaussianSupport::Interval(degenerate)),
    );
    zero(
        ClenshawCurtis::new(4)
            .unwrap()
            .integrate(&callback, degenerate),
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(Interval::new(1.0_f64, -1.0).is_err());

    let shifted = weighted(
        &GaussianRulePlan::new(OrthogonalFamily::Laguerre, 6).unwrap(),
        |x: f64| (x - 3.0).powi(4),
        GaussianSupport::HalfLine(3.0),
    );
    assert_close_f64(shifted, 24.0, 2.0e-12);
}

#[test]
fn constructors_reject_invalid_parameters_and_unrepresentable_storage() {
    assert!(
        GaussianRulePlan::new(
            OrthogonalFamily::Jacobi {
                alpha: f64::NAN,
                beta: 0.0
            },
            4
        )
        .is_err()
    );
    assert!(
        GaussianRulePlan::new(
            OrthogonalFamily::Jacobi {
                alpha: 0.0,
                beta: f64::INFINITY
            },
            4
        )
        .is_err()
    );
    assert!(
        GaussianRulePlan::new(
            OrthogonalFamily::Jacobi {
                alpha: -1.0,
                beta: 0.0
            },
            4
        )
        .is_err()
    );
    assert!(GaussianRulePlan::<f64>::gauss_legendre(usize::MAX).is_err());
    assert!(GaussianRulePlan::<f64>::new(OrthogonalFamily::HermiteHe, usize::MAX).is_err());
    assert!(GaussianRulePlan::<f64>::new(OrthogonalFamily::Laguerre, usize::MAX).is_err());
    assert!(GaussianRulePlan::<f64>::new(OrthogonalFamily::ChebyshevT, usize::MAX).is_err());
    assert!(ClenshawCurtis::<f64>::new(usize::MAX).is_err());
}
