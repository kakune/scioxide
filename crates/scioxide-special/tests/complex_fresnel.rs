#![allow(clippy::unwrap_used)]
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, Complex};
use scioxide_special::*;
#[test]
fn independent_mpmath_values_and_symmetries() {
    // mpmath 1.3.0, 160/240 digits; exact IEEE inputs. DLMF normalized integrals.
    for (z, c, s) in [
        (
            Complex::new(0.1, 0.2),
            Complex::new(0.09989880276189232, 0.20009374099919589),
            Complex::new(-0.005759853676579815, -0.0010497631428340612),
        ),
        (
            Complex::new(2.0, 1.0),
            Complex::new(-36.22568799288165, 16.08787137412548),
            Complex::new(-15.587751104404587, -36.72546488399144),
        ),
        (
            Complex::new(20.0, 0.1),
            Complex::new(0.4509800688005908, 4.260_984_281_409_1),
            Complex::new(-3.761013995157015, -0.04901929214839304),
        ),
    ] {
        for (function, expected) in [
            (
                fresnel_c::<Complex<f64>>
                    as fn(Complex<f64>) -> Result<Complex<f64>, scioxide_core::SciOxideError>,
                c,
            ),
            (
                fresnel_s::<Complex<f64>>
                    as fn(Complex<f64>) -> Result<Complex<f64>, scioxide_core::SciOxideError>,
                s,
            ),
        ] {
            let actual: Complex<f64> = function(z).unwrap();
            assert!(
                (actual - expected).norm() < 2e-12 * expected.norm().max(1.0),
                "{z:?}: {actual:?}, {expected:?}"
            );
            assert!(
                (function(z.conj()).unwrap() - actual.conj()).norm()
                    < 2e-12 * actual.norm().max(1.0)
            );
            assert!((function(-z).unwrap() + actual).norm() < 2e-12 * actual.norm().max(1.0));
        }
        let z32 = Complex::new(z.re as f32, z.im as f32);
        assert!(
            (fresnel_c(z32).unwrap() - Complex::new(c.re as f32, c.im as f32)).norm()
                < 2e-5 * (c.norm() as f32).max(1.0)
        );
        assert!(
            (fresnel_s(z32).unwrap() - Complex::new(s.re as f32, s.im as f32)).norm()
                < 2e-5 * (s.norm() as f32).max(1.0)
        );
    }
}
#[test]
fn origin_derivatives_and_exhaustion() {
    let z = Complex::new(0.0_f64, 0.0);
    let c = fresnel_c_with_options(z, IntegralOptions::default()).unwrap();
    let s = fresnel_s_with_options(z, IntegralOptions::default()).unwrap();
    assert_eq!(c.try_value().unwrap(), z);
    assert_eq!(s.try_value().unwrap(), z);
    assert_eq!(*c.derivative().unwrap().value(), Complex::new(1.0, 0.0));
    assert_eq!(*s.derivative().unwrap().value(), z);
    let result = fresnel_c_with_options(
        Complex::new(4.0, 1.0),
        IntegralOptions::default().with_iteration_budget(Budget::ONE),
    )
    .unwrap_err();
    let result_counts = publication::exhausted_counts(&result);
    assert_eq!(result_counts.iterations(), 1);
    assert!(fresnel_s(Complex::new(f64::NAN, 0.0)).is_err());
}
