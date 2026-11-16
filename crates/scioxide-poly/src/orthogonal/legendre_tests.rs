use crate::OrthogonalFamily;

use crate::test_assertions::assert_near;

#[test]
fn test_legendre_p0() {
    let l = OrthogonalFamily::Legendre;
    assert_near(l.evaluate(0, 0.5_f64).unwrap(), 1.0, 1e-14);
}

#[test]
fn test_legendre_p1() {
    let l = OrthogonalFamily::Legendre;
    assert_near(l.evaluate(1, 0.5_f64).unwrap(), 0.5, 1e-14);
}

#[test]
fn test_legendre_p2() {
    // P_2(x) = (3x^2 - 1)/2 => P_2(0.5) = -0.125
    let l = OrthogonalFamily::Legendre;
    assert_near(l.evaluate(2, 0.5_f64).unwrap(), -0.125, 1e-14);
}

#[test]
fn test_legendre_zeros() {
    let l = OrthogonalFamily::Legendre;
    let z = l.zeros(4).unwrap();
    assert_eq!(z.len(), 4);
    for &zi in &z {
        assert_near(l.evaluate(4, zi).unwrap(), 0.0, 1e-10);
    }
}

#[test]
fn test_legendre_zeros_in_range() {
    let l = OrthogonalFamily::<f64>::Legendre;
    let z = l.zeros(5).unwrap();
    for &zi in &z {
        assert!(zi > -1.0 && zi < 1.0);
    }
}

#[test]
fn test_legendre_derivative() {
    // P_2'(x) = 3x => P_2'(0.5) = 1.5
    let l = OrthogonalFamily::Legendre;
    assert_near(l.evaluate_derivative(2, 0.5_f64).unwrap(), 1.5, 1e-12);
}

#[test]
fn test_legendre_weight_in_domain() {
    let l = OrthogonalFamily::Legendre;
    assert_near(l.weight(-1.0_f64).unwrap(), 1.0, 1e-14);
    assert_near(l.weight(0.25_f64).unwrap(), 1.0, 1e-14);
    assert_near(l.weight(1.0_f64).unwrap(), 1.0, 1e-14);
}

#[test]
fn test_legendre_weight_out_of_domain() {
    let l = OrthogonalFamily::Legendre;
    assert!(matches!(
        (l.weight(-1.01_f64)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::DomainError { .. }
    ));
    assert!(matches!(
        (l.weight(1.01_f64)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::DomainError { .. }
    ));
}

#[test]
fn recurrence_validates_all_degrees_and_preserves_signed_zero() {
    let family = OrthogonalFamily::Legendre;
    for degree in [0, 1, 4] {
        assert!(matches!(
            (OrthogonalFamily::Legendre.evaluate(degree, f64::NAN))
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }
    assert!(
        OrthogonalFamily::Legendre
            .evaluate(1, -0.0_f64)
            .unwrap()
            .is_sign_negative()
    );
    assert!(
        OrthogonalFamily::Legendre
            .evaluate(1, -0.0_f32)
            .unwrap()
            .is_sign_negative()
    );
    for x in [
        0.0_f32,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MAX,
        -f32::MAX,
    ] {
        assert_eq!(
            OrthogonalFamily::Legendre.evaluate(1, x).unwrap().to_bits(),
            x.to_bits()
        );
    }
    for x in [
        0.0_f64,
        -0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MAX,
        -f64::MAX,
    ] {
        assert_eq!(
            OrthogonalFamily::Legendre.evaluate(1, x).unwrap().to_bits(),
            x.to_bits()
        );
    }
    assert_eq!(
        OrthogonalFamily::Legendre
            .evaluate(1, f64::from_bits(1))
            .unwrap()
            .to_bits(),
        1
    );
    assert_eq!(
        OrthogonalFamily::Legendre
            .evaluate(1, -0.0_f32)
            .unwrap()
            .to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(
        OrthogonalFamily::Legendre
            .evaluate(1, f32::from_bits(1))
            .unwrap()
            .to_bits(),
        1
    );
    assert!(matches!(
        (family.weight(f64::NAN)).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Input,
            ..
        }
    ));
}
