use crate::OrthogonalFamily;

use crate::test_assertions::assert_near;

// ---- OrthogonalFamily::HermiteH ----

#[test]
fn test_hermite_h_h0() {
    let h = OrthogonalFamily::HermiteH;
    assert_near(h.evaluate(0, 1.0_f64).unwrap(), 1.0, 1e-14);
}

#[test]
fn test_hermite_h_h1() {
    // H_1(x) = 2x => H_1(1.5) = 3
    let h = OrthogonalFamily::HermiteH;
    assert_near(h.evaluate(1, 1.5_f64).unwrap(), 3.0, 1e-14);
}

#[test]
fn test_hermite_h_h2() {
    // H_2(x) = 4x^2 - 2 => H_2(1) = 2
    let h = OrthogonalFamily::HermiteH;
    assert_near(h.evaluate(2, 1.0_f64).unwrap(), 2.0, 1e-14);
}

#[test]
fn test_hermite_h_zeros() {
    let h = OrthogonalFamily::HermiteH;
    let z = h.zeros(3).unwrap();
    assert_eq!(z.len(), 3);
    for &zi in &z {
        assert_near(h.evaluate(3, zi).unwrap(), 0.0, 1e-10);
    }
}

// ---- OrthogonalFamily::HermiteHe ----

#[test]
fn test_hermite_he_he0() {
    let h = OrthogonalFamily::HermiteHe;
    assert_near(h.evaluate(0, 1.0_f64).unwrap(), 1.0, 1e-14);
}

#[test]
fn test_hermite_he_he1() {
    let h = OrthogonalFamily::HermiteHe;
    assert_near(h.evaluate(1, 2.0_f64).unwrap(), 2.0, 1e-14);
}

#[test]
fn test_hermite_he_he2() {
    // He_2(x) = x^2 - 1 => He_2(2) = 3
    let h = OrthogonalFamily::HermiteHe;
    assert_near(h.evaluate(2, 2.0_f64).unwrap(), 3.0, 1e-14);
}

#[test]
fn test_hermite_he_zeros() {
    let h = OrthogonalFamily::HermiteHe;
    let z = h.zeros(4).unwrap();
    assert_eq!(z.len(), 4);
    for &zi in &z {
        assert_near(h.evaluate(4, zi).unwrap(), 0.0, 1e-10);
    }
}

#[test]
fn extreme_finite_arguments_never_return_non_finite_successes() {
    let physicists = OrthogonalFamily::HermiteH;
    assert!(matches!(
        (physicists.evaluate(1, f64::MAX)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        }
    ));
    assert_eq!(physicists.weight(f64::MAX).unwrap(), 0.0);
    assert!(matches!(
        (physicists.weight(f64::NAN)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Input,
            ..
        }
    ));
}
