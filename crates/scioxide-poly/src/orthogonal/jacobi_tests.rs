use crate::OrthogonalFamily;

use crate::test_assertions::assert_near;

#[test]
fn test_jacobi_reduces_to_legendre() {
    // P_n^(0,0)(x) = P_n(x) (Legendre)
    let j = OrthogonalFamily::Jacobi {
        alpha: 0.0,
        beta: 0.0,
    };
    let l = OrthogonalFamily::Legendre;
    for n in 0..5 {
        let x = 0.3_f64;
        let jv = j.evaluate(n, x).unwrap();
        let lv = l.evaluate(n, x).unwrap();
        assert_near(jv, lv, 1e-10);
    }
}

#[test]
fn test_jacobi_p0() {
    let j = OrthogonalFamily::Jacobi {
        alpha: 1.0,
        beta: 2.0,
    };
    assert_near(j.evaluate(0, 0.5).unwrap(), 1.0, 1e-14);
}

#[test]
fn test_jacobi_zeros() {
    let j = OrthogonalFamily::Jacobi {
        alpha: 0.5,
        beta: 0.5,
    };
    let z = j.zeros(4).unwrap();
    assert_eq!(z.len(), 4);
    for &zi in &z {
        assert!(zi > -1.0 && zi < 1.0);
        assert_near(j.evaluate(4, zi).unwrap(), 0.0, 1e-9);
    }
}

#[test]
fn test_jacobi_weight() {
    let j = OrthogonalFamily::Jacobi {
        alpha: 1.0,
        beta: 2.0,
    };
    let w = j.weight(0.5).unwrap();
    // (1-0.5)^1 * (1+0.5)^2 = 0.5 * 2.25 = 1.125
    assert_near(w, 1.125, 1e-14);
}

#[test]
fn jacobi_weight_balances_overflowing_and_underflowing_factors() {
    let family = OrthogonalFamily::Jacobi {
        alpha: 2000.0,
        beta: 2000.0,
    };
    let expected = (2000.0_f64 * 0.75_f64.ln()).exp();
    let actual = family.weight(0.5).unwrap();
    assert!(((actual - expected) / expected).abs() < 1e-12);
    assert!(actual > 0.0);
}

#[test]
fn extreme_parameters_have_finite_coefficients_or_typed_failure() {
    let family = OrthogonalFamily::Jacobi {
        alpha: f64::MAX,
        beta: f64::MAX,
    };
    assert_eq!(family.evaluate(1, 1.0).unwrap(), f64::MAX);
    if let Err(error) = family.recurrence_coefficients(1) {
        assert!(matches!(
            (error).view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow
                    | scioxide_core::NumericalIssue::Breakdown,
                ..
            }
        ));
    }
    assert!(matches!(
        (family.evaluate(0, f64::NAN)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Input,
            ..
        }
    ));
}
