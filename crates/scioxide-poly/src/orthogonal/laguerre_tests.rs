use crate::OrthogonalFamily;

use crate::test_assertions::assert_near;

// ---- OrthogonalFamily::Laguerre ----

#[test]
fn test_laguerre_l0() {
    let l = OrthogonalFamily::Laguerre;
    assert_near(l.evaluate(0, 1.0_f64).unwrap(), 1.0, 1e-14);
}

#[test]
fn test_laguerre_l1() {
    // L_1(x) = 1 - x => L_1(2) = -1
    let l = OrthogonalFamily::Laguerre;
    assert_near(l.evaluate(1, 2.0_f64).unwrap(), -1.0, 1e-14);
}

#[test]
fn test_laguerre_l2() {
    // L_2(x) = (x^2 - 4x + 2)/2 => L_2(1) = -0.5
    let l = OrthogonalFamily::Laguerre;
    assert_near(l.evaluate(2, 1.0_f64).unwrap(), -0.5, 1e-14);
}

#[test]
fn test_laguerre_zeros() {
    let l = OrthogonalFamily::Laguerre;
    let z = l.zeros(3).unwrap();
    assert_eq!(z.len(), 3);
    for &zi in &z {
        assert!(zi > 0.0, "Laguerre zeros should be positive");
        assert_near(l.evaluate(3, zi).unwrap(), 0.0, 1e-9);
    }
}

#[test]
fn test_laguerre_weight_positive() {
    let l = OrthogonalFamily::Laguerre;
    let w = l.weight(1.0_f64).unwrap();
    assert_near(w, (-1.0_f64).exp(), 1e-14);
}

#[test]
fn test_laguerre_weight_negative_domain() {
    let l = OrthogonalFamily::Laguerre;
    assert!(matches!(
        (l.weight(-1.0_f64)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::DomainError { .. }
    ));
}

// ---- Generalized Laguerre ----

#[test]
fn test_generalized_laguerre_alpha_zero_matches_standard() {
    let g = OrthogonalFamily::GeneralizedLaguerre { alpha: 0.0 };
    let l = OrthogonalFamily::Laguerre;
    for n in 0..5 {
        let x = 1.5_f64;
        assert_near(g.evaluate(n, x).unwrap(), l.evaluate(n, x).unwrap(), 1e-12);
    }
}

#[test]
fn test_generalized_laguerre_zeros() {
    let g = OrthogonalFamily::GeneralizedLaguerre { alpha: 0.5 };
    let z = g.zeros(3).unwrap();
    assert_eq!(z.len(), 3);
    for &zi in &z {
        assert!(zi > 0.0);
        assert_near(g.evaluate(3, zi).unwrap(), 0.0, 1e-9);
    }
}

#[test]
fn generalized_weight_uses_log_domain_for_balanced_extremes() {
    let family = OrthogonalFamily::GeneralizedLaguerre { alpha: 100.0 };
    let expected = (100.0 * 1000.0_f64.ln() - 1000.0).exp();
    let actual = family.weight(1000.0).unwrap();
    assert!(((actual - expected) / expected).abs() < 1e-12);
    assert!(actual > 0.0);
}
