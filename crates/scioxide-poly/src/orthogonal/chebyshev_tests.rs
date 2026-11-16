use crate::OrthogonalFamily;

use crate::test_assertions::assert_near;

// ---- OrthogonalFamily::ChebyshevT ----

#[test]
fn test_chebyshev_t_eval_t0() {
    let t = OrthogonalFamily::ChebyshevT;
    assert_near(t.evaluate(0, 0.5_f64).unwrap(), 1.0, 1e-14);
}

#[test]
fn test_chebyshev_t_eval_t1() {
    let t = OrthogonalFamily::ChebyshevT;
    assert_near(t.evaluate(1, 0.5_f64).unwrap(), 0.5, 1e-14);
}

#[test]
fn test_chebyshev_t_eval_t2() {
    // T_2(x) = 2x^2 - 1 => T_2(0.5) = -0.5
    let t = OrthogonalFamily::ChebyshevT;
    assert_near(t.evaluate(2, 0.5_f64).unwrap(), -0.5, 1e-14);
}

#[test]
fn test_chebyshev_t_eval_t3() {
    // T_3(x) = 4x^3 - 3x => T_3(0.5) = -1.0
    let t = OrthogonalFamily::ChebyshevT;
    assert_near(t.evaluate(3, 0.5_f64).unwrap(), -1.0, 1e-14);
}

#[test]
fn test_chebyshev_t_zeros() {
    let t = OrthogonalFamily::ChebyshevT;
    let z = t.zeros(3).unwrap();
    assert_eq!(z.len(), 3);
    // Each zero should satisfy T_3(z) ≈ 0
    for &zi in &z {
        assert_near(t.evaluate(3, zi).unwrap(), 0.0, 1e-12);
    }
}

#[test]
fn test_chebyshev_t_zeros_degree_zero() {
    let t = OrthogonalFamily::<f64>::ChebyshevT;
    assert!(matches!(
        (t.zeros(0)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
}

#[test]
fn test_chebyshev_t_weight() {
    let t = OrthogonalFamily::ChebyshevT;
    let w = t.weight(0.5_f64).unwrap();
    // 1/sqrt(1 - 0.25) = 1/sqrt(0.75)
    assert_near(w, 1.0 / 0.75_f64.sqrt(), 1e-14);
}

#[test]
fn test_chebyshev_t_weight_boundary() {
    let t = OrthogonalFamily::ChebyshevT;
    assert!(matches!(
        (t.weight(1.0_f64)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::DomainError { .. }
    ));
}

#[test]
fn test_chebyshev_t_derivative() {
    // T_2'(x) = 4x => T_2'(0.5) = 2
    let t = OrthogonalFamily::ChebyshevT;
    assert_near(t.evaluate_derivative(2, 0.5_f64).unwrap(), 2.0, 1e-12);
}

// ---- OrthogonalFamily::ChebyshevU ----

#[test]
fn test_chebyshev_u_eval_u0() {
    let u = OrthogonalFamily::ChebyshevU;
    assert_near(u.evaluate(0, 0.5_f64).unwrap(), 1.0, 1e-14);
}

#[test]
fn test_chebyshev_u_eval_u1() {
    // U_1(x) = 2x => U_1(0.5) = 1
    let u = OrthogonalFamily::ChebyshevU;
    assert_near(u.evaluate(1, 0.5_f64).unwrap(), 1.0, 1e-14);
}

#[test]
fn test_chebyshev_u_eval_u2() {
    // U_2(x) = 4x^2 - 1 => U_2(0.5) = 0
    let u = OrthogonalFamily::ChebyshevU;
    assert_near(u.evaluate(2, 0.5_f64).unwrap(), 0.0, 1e-14);
}

#[test]
fn test_chebyshev_u_zeros() {
    let u = OrthogonalFamily::ChebyshevU;
    let z = u.zeros(3).unwrap();
    assert_eq!(z.len(), 3);
    for &zi in &z {
        assert_near(u.evaluate(3, zi).unwrap(), 0.0, 1e-12);
    }
}

#[test]
fn test_chebyshev_u_weight() {
    let u = OrthogonalFamily::ChebyshevU;
    let w = u.weight(0.5_f64).unwrap();
    assert_near(w, 0.75_f64.sqrt(), 1e-14);
}

#[test]
fn degree_one_overflow_and_non_finite_weights_are_rejected() {
    let u = OrthogonalFamily::ChebyshevU;
    assert!(matches!(
        (u.evaluate(1, f64::MAX)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
    assert!(matches!(
        (u.weight(f64::NAN)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Input,
            ..
        }
    ));
}
