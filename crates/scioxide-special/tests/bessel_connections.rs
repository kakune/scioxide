//! Exact-input real-axis derivative and reflected-I error coverage.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideError};
use scioxide_special::*;

fn check<S: RealScalar>(scalar: &str, decode: impl Fn(&str) -> S)
where
    Complex<S>: NumericScalar<Real = S>,
{
    for row in include_str!("data/bessel_connections.tsv")
        .lines()
        .filter(|r| !r.starts_with('#'))
    {
        let fields: Vec<_> = row.split_whitespace().collect();
        if fields[0] != scalar {
            continue;
        }
        let nu = decode(fields[2]);
        let x = decode(fields[3]);
        let degree = fields[4].parse().unwrap();
        let expected = decode(fields[5]);
        let conditioning = decode(fields[6]);
        let o = BesselOptions::default();
        let evaluate = || match fields[1] {
            "j" => bessel_j_derivative_with_options(nu, x, degree, o),
            "y" => bessel_y_derivative_with_options(nu, x, degree, o),
            "i" if degree == 0 => bessel_i_with_options(nu, x, o),
            "i" => bessel_i_derivative_with_options(nu, x, degree, o),
            "is" => bessel_i_scaled_with_options(nu, x, o),
            "k" => bessel_k_derivative_with_options(nu, x, degree, o),
            _ => unreachable!(),
        };
        let result = evaluate().unwrap_or_else(|e| panic!("{row}: {e}"));
        if !expected.is_finite() {
            assert_eq!(
                result.scalar_representation(),
                ScalarRepresentation::Overflow,
                "{row}"
            );
            assert!(result.logarithmic().is_some(), "{row}");
            continue;
        }
        let actual = result.try_value().unwrap_or_else(|e| panic!("{row}: {e}"));
        let rounding = S::EPSILON * expected.abs() + S::MIN_SUBNORMAL;
        assert!(
            (actual - expected).abs() <= conditioning + rounding,
            "accuracy {row}: actual={actual:?}"
        );
        assert!(
            (actual - expected).abs() <= result.value().unwrap().absolute_error() + rounding,
            "coverage {row}: actual={actual:?}, bound={:?}",
            result.value().unwrap().absolute_error()
        );
        if degree == 0 && fields[1] != "is" && expected.abs() >= S::MIN_POSITIVE {
            for zero in [S::ZERO, -S::ZERO] {
                let order = Complex::new(nu, zero);
                let z = Complex::new(x, zero);
                let r = match fields[1] {
                    "j" => bessel_j_with_options(order, z, o),
                    "y" => bessel_y_with_options(order, z, o),
                    "i" => bessel_i_with_options(order, z, o),
                    "k" => bessel_k_with_options(order, z, o),
                    _ => unreachable!(),
                }
                .unwrap_or_else(|e| panic!("axis {row}: {e}"));
                let value = r.try_value().unwrap();
                assert!(
                    (value.re - expected).abs() <= conditioning + rounding,
                    "axis {row}: {value:?}"
                );
                assert!(
                    (value.re - expected).abs() <= r.value().unwrap().absolute_error() + rounding,
                    "axis coverage {row}"
                );
            }
        }
    }
}

#[test]
fn binary64_connections() {
    check("f64", |s| {
        f64::from_bits(u64::from_str_radix(s, 16).unwrap())
    });
}
#[test]
fn binary32_connections() {
    check("f32", |s| {
        f32::from_bits(u32::from_str_radix(s, 16).unwrap())
    });
}

#[test]
fn reflected_values_are_finite_near_zeros() -> Result<(), SciOxideError> {
    for (nu, x) in [(-33.5_f64, 22.394198737143967), (-31.5, 21.068764240299835)] {
        let r = bessel_i_with_options(nu, x, BesselOptions::default())?;
        assert!(r.try_value()?.is_finite());
        assert!(r.value().unwrap().absolute_error() > 0.0);
    }
    Ok(())
}
