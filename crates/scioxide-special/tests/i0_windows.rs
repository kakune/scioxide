//! Independent integral coverage of the I0 transition used by Kaiser windows.
#![allow(clippy::unwrap_used)]
use scioxide_special::{bessel_i, bessel_i_scaled};
#[test]
fn i0_transition_agrees_with_periodic_integral() {
    // DLMF 10.32.1: exp(-x) I0(x) = mean exp(x(cos(theta)-1)).
    // A 2048-point periodic trapezoid has negligible aliasing for x <= 40;
    // this is independent of both production's series and asymptotic expansion.
    for x in [6.0, 12.0, 14.0, 20.0, 31.0, 32.0, 40.0] {
        let reference = (0..2048)
            .map(|i| {
                let theta = 2.0 * std::f64::consts::PI * f64::from(i) / 2048.0;
                (x * (theta.cos() - 1.0)).exp() / 2048.0
            })
            .sum::<f64>();
        let actual = bessel_i_scaled(0.0_f64, x).unwrap();
        assert!(
            (actual / reference - 1.0).abs() < 5e-14,
            "x={x}, {actual} vs {reference}"
        );
        assert!((bessel_i(0.0_f64, x).unwrap() * (-x).exp() / reference - 1.0).abs() < 5e-14);
        let xf: f32 = scioxide_core::RealScalar::try_from_f64_lossy(x).unwrap();
        let narrow = bessel_i_scaled(0.0_f32, xf).unwrap();
        assert!((f64::from(narrow) / reference - 1.0).abs() < 2e-6);
    }
}
