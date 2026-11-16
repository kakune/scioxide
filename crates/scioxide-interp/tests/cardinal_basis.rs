//! Independent truncated-power and partition-of-unity tests.
#![allow(clippy::unwrap_used)]
use scioxide_core::RealScalar;
use scioxide_interp::cardinal_bspline;
fn verify<S: RealScalar>(tol: f64) {
    for degree in 0..=3 {
        for k in -80..=80 {
            let x = k as f64 / 16.;
            let got = cardinal_bspline(degree, S::try_from_f64_lossy(x).unwrap())
                .unwrap()
                .into_f64();
            let expected = match degree {
                0 => {
                    if (-0.5..0.5).contains(&x) {
                        1.
                    } else {
                        0.
                    }
                }
                1 => (1. - x.abs()).max(0.),
                2 => {
                    if x.abs() < 0.5 {
                        0.75 - x * x
                    } else if x.abs() < 1.5 {
                        0.5 * (1.5 - x.abs()).powi(2)
                    } else {
                        0.
                    }
                }
                _ => {
                    if x.abs() < 1. {
                        2. / 3. - x * x + 0.5 * x.abs().powi(3)
                    } else if x.abs() < 2. {
                        (2. - x.abs()).powi(3) / 6.
                    } else {
                        0.
                    }
                }
            };
            assert!(
                (got - expected).abs() < tol,
                "degree {degree}, x {x}: {got} vs {expected}"
            );
            let partition: f64 = (-8..=8)
                .map(|i| {
                    cardinal_bspline(degree, S::try_from_f64_lossy(x - i as f64).unwrap())
                        .unwrap()
                        .into_f64()
                })
                .sum();
            assert!((partition - 1.).abs() < tol);
        }
    }
    assert!(cardinal_bspline::<S>(4, S::ZERO).is_err());
}
#[test]
fn f32_basis() {
    verify::<f32>(5e-7);
}
#[test]
fn f64_basis() {
    verify::<f64>(5e-15);
}
