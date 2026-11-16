//! Gaussian family construction, moments and weighted supports.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Interval, RealScalar};
use scioxide_integrate::{FixedRule, GaussianRulePlan, GaussianSupport, IntegrationTermination};
use scioxide_poly::OrthogonalFamily;

/// Weighted integral of `f` over `support`.
fn weighted<S: RealScalar + Default>(
    plan: &GaussianRulePlan<S>,
    f: impl Fn(S) -> S + Sync,
    support: GaussianSupport<S>,
) -> S {
    plan.integrate_weighted(|x: S| Ok(f(x)), support).unwrap().0
}

fn rejects_zero_order(family: OrthogonalFamily<f64>) {
    assert!(matches!(
        GaussianRulePlan::new(family, 0).err().unwrap().view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
}

mod legendre {
    use super::*;

    #[test]
    fn test_new_zero_order() {
        rejects_zero_order(OrthogonalFamily::Legendre);
    }

    // integral_0^1 x dx = 0.5
    #[test]
    fn test_linear_f64() {
        let gl = GaussianRulePlan::gauss_legendre(2).unwrap();
        let (value, report) = gl
            .integrate(|x: f64| Ok(x), Interval::new(0.0, 1.0).unwrap())
            .unwrap();
        assert!((value - 0.5).abs() < 1e-14);
        assert!(report.is_success());
        assert_eq!(
            report.termination(),
            IntegrationTermination::FixedRuleCompleted
        );
        assert_eq!(report.evaluations(), 2);
        assert!(report.error_estimate().is_none());
    }

    // integral_0^1 x^3 dx = 0.25 — exact for n=2 (degree 2n-1=3)
    #[test]
    fn test_cubic_exact_f64() {
        let gl = GaussianRulePlan::gauss_legendre(2).unwrap();
        let (value, _) = gl
            .integrate(|x: f64| Ok(x * x * x), Interval::new(0.0, 1.0).unwrap())
            .unwrap();
        assert!((value - 0.25).abs() < 1e-13);
    }

    // integral_0^pi sin(x) dx = 2
    #[test]
    fn test_sin_f64() {
        let gl = GaussianRulePlan::gauss_legendre(10).unwrap();
        let (value, _) = gl
            .integrate(
                |x: f64| Ok(x.sin()),
                Interval::new(0.0, core::f64::consts::PI).unwrap(),
            )
            .unwrap();
        assert!((value - 2.0).abs() < 1e-12);
    }

    // f32 test
    #[test]
    fn test_linear_f32() {
        let gl = GaussianRulePlan::gauss_legendre(3).unwrap();
        let (value, _) = gl
            .integrate(|x: f32| Ok(x), Interval::new(0.0_f32, 1.0_f32).unwrap())
            .unwrap();
        assert!((value - 0.5_f32).abs() < 1e-5);
    }

    #[test]
    fn test_non_finite_integrand_rejected() {
        let gl = GaussianRulePlan::gauss_legendre(3).unwrap();
        let result = gl.integrate(|_x: f64| Ok(f64::NAN), Interval::new(0.0, 1.0).unwrap());
        assert!(matches!(
            result.err().unwrap().view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Callback,
                ..
            }
        ));
    }
}

mod chebyshev {
    use super::*;

    fn canonical<S: RealScalar>() -> GaussianSupport<S> {
        GaussianSupport::Interval(Interval::new(-S::ONE, S::ONE).unwrap())
    }

    #[test]
    fn test_new_zero_order() {
        rejects_zero_order(OrthogonalFamily::ChebyshevT);
    }

    // integral_{-1}^{1} 1 / sqrt(1-x^2) dx = pi
    #[test]
    fn test_constant_weighted_f64() {
        let gc = GaussianRulePlan::new(OrthogonalFamily::ChebyshevT, 4).unwrap();
        let value = weighted(&gc, |_: f64| 1.0, canonical());
        assert!((value - core::f64::consts::PI).abs() < 1e-13, "got {value}");
    }

    // integral_{-1}^{1} x^2 / sqrt(1-x^2) dx = pi/2
    #[test]
    fn test_x_sq_weighted_f64() {
        let gc = GaussianRulePlan::new(OrthogonalFamily::ChebyshevT, 5).unwrap();
        let value = weighted(&gc, |x: f64| x * x, canonical());
        assert!(
            (value - core::f64::consts::PI / 2.0).abs() < 1e-13,
            "got {value}"
        );
    }

    // integral_{-1}^{1} x / sqrt(1-x^2) dx = 0 (odd function)
    #[test]
    fn test_odd_weighted_f64() {
        let gc = GaussianRulePlan::new(OrthogonalFamily::ChebyshevT, 5).unwrap();
        let value = weighted(&gc, |x: f64| x, canonical());
        assert!(value.abs() < 1e-15, "got {value}");
    }

    // f32 test: integral_{-1}^{1} 1 / sqrt(1-x^2) dx = pi
    #[test]
    fn test_constant_weighted_f32() {
        let gc = GaussianRulePlan::new(OrthogonalFamily::ChebyshevT, 5).unwrap();
        let value = weighted(&gc, |_: f32| 1.0, canonical());
        assert!((value - core::f32::consts::PI).abs() < 1e-5, "got {value}");
    }

    #[test]
    fn test_unweighted_application_is_rejected() {
        let gc = GaussianRulePlan::new(OrthogonalFamily::ChebyshevT, 5).unwrap();
        let error = gc
            .integrate(|_: f64| Ok(1.0), Interval::new(-1.0, 1.0).unwrap())
            .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
    }
}

mod jacobi {
    use super::*;

    #[test]
    fn test_new_zero_order() {
        rejects_zero_order(OrthogonalFamily::Jacobi {
            alpha: 0.0,
            beta: 0.0,
        });
    }

    #[test]
    fn test_new_bad_alpha() {
        assert!(matches!(
            GaussianRulePlan::new(
                OrthogonalFamily::<f64>::Jacobi {
                    alpha: -1.0,
                    beta: 0.5
                },
                3
            )
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }

    #[test]
    fn test_new_bad_beta() {
        assert!(matches!(
            GaussianRulePlan::new(
                OrthogonalFamily::<f64>::Jacobi {
                    alpha: 0.5,
                    beta: -1.5
                },
                3
            )
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }

    // alpha=0, beta=0 reduces to Legendre.
    // integral_{-1}^{1} 1 · (1-x)^0 · (1+x)^0 dx = 2
    #[test]
    fn test_legendre_case_constant_f64() {
        let gj = GaussianRulePlan::new(
            OrthogonalFamily::Jacobi {
                alpha: 0.0_f64,
                beta: 0.0,
            },
            5,
        )
        .unwrap();
        let (value, _) = gj
            .integrate(|_: f64| Ok(1.0), Interval::new(-1.0, 1.0).unwrap())
            .unwrap();
        assert!((value - 2.0).abs() < 1e-10, "got {value}");
    }

    // integral_0^1 x dx with alpha=0, beta=0 should give 0.5
    #[test]
    fn test_legendre_case_linear_f64() {
        let gj = GaussianRulePlan::new(
            OrthogonalFamily::Jacobi {
                alpha: 0.0_f64,
                beta: 0.0,
            },
            5,
        )
        .unwrap();
        let (value, _) = gj
            .integrate(|x: f64| Ok(x), Interval::new(0.0, 1.0).unwrap())
            .unwrap();
        assert!((value - 0.5).abs() < 1e-10, "got {value}");
    }
}

mod hermite {
    use super::*;

    #[test]
    fn test_new_zero_order() {
        rejects_zero_order(OrthogonalFamily::HermiteHe);
    }

    #[test]
    fn test_large_n_has_no_factorial_limit() {
        let gh = GaussianRulePlan::new(OrthogonalFamily::HermiteHe, 200).unwrap();
        let value = weighted(&gh, |_: f64| 1.0, GaussianSupport::RealLine);
        let expected = (2.0 * core::f64::consts::PI).sqrt();
        assert!((value - expected).abs() < 1e-12);
    }

    // integral_{-inf}^{inf} 1 * exp(-x^2/2) dx = sqrt(2*pi)
    #[test]
    fn test_constant_f64() {
        let gh = GaussianRulePlan::new(OrthogonalFamily::HermiteHe, 5).unwrap();
        let value = weighted(&gh, |_: f64| 1.0, GaussianSupport::RealLine);
        let expected = (2.0 * core::f64::consts::PI).sqrt();
        assert!((value - expected).abs() < 1e-12);
    }

    // integral_{-inf}^{inf} x^2 * exp(-x^2/2) dx = sqrt(2*pi)
    // (second moment of standard normal = 1, so integral = sqrt(2pi))
    #[test]
    fn test_x_sq_f64() {
        let gh = GaussianRulePlan::new(OrthogonalFamily::HermiteHe, 5).unwrap();
        let value = weighted(&gh, |x: f64| x * x, GaussianSupport::RealLine);
        let expected = (2.0 * core::f64::consts::PI).sqrt();
        assert!((value - expected).abs() < 1e-10);
    }

    // integral_{-inf}^{inf} x * exp(-x^2/2) dx = 0 (odd function)
    #[test]
    fn test_odd_function_f64() {
        let gh = GaussianRulePlan::new(OrthogonalFamily::HermiteHe, 6).unwrap();
        let value = weighted(&gh, |x: f64| x, GaussianSupport::RealLine);
        assert!(value.abs() < 1e-14);
    }
}

mod laguerre {
    use super::*;

    #[test]
    fn test_new_zero_order() {
        rejects_zero_order(OrthogonalFamily::Laguerre);
    }

    // integral_0^inf 1 * exp(-x) dx = 1
    #[test]
    fn test_constant_f64() {
        let gl = GaussianRulePlan::new(OrthogonalFamily::Laguerre, 5).unwrap();
        let value = weighted(&gl, |_: f64| 1.0, GaussianSupport::HalfLine(0.0));
        assert!((value - 1.0).abs() < 1e-12);
    }

    // integral_0^inf x * exp(-x) dx = 1
    #[test]
    fn test_x_exp_neg_x_f64() {
        let gl = GaussianRulePlan::new(OrthogonalFamily::Laguerre, 5).unwrap();
        let value = weighted(&gl, |x: f64| x, GaussianSupport::HalfLine(0.0));
        assert!((value - 1.0).abs() < 1e-12);
    }

    // integral_0^inf x^2 * exp(-x) dx = 2
    #[test]
    fn test_x_sq_exp_neg_x_f64() {
        let gl = GaussianRulePlan::new(OrthogonalFamily::Laguerre, 5).unwrap();
        let value = weighted(&gl, |x: f64| x * x, GaussianSupport::HalfLine(0.0));
        assert!((value - 2.0).abs() < 1e-11);
    }

    // Shifted: integral_1^inf 1 * exp(-(x-1)) dx = 1
    #[test]
    fn test_shifted_f64() {
        let gl = GaussianRulePlan::new(OrthogonalFamily::Laguerre, 5).unwrap();
        let value = weighted(&gl, |_: f64| 1.0, GaussianSupport::HalfLine(1.0));
        assert!((value - 1.0).abs() < 1e-12);
    }
}
