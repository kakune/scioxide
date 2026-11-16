//! Independent signed-axis references for both cylindrical order paths and spherical waves.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_special::*;

fn close<S: RealScalar>(row: &str, actual: Complex<S>, expected: Complex<S>) {
    let (absolute, relative) = if S::EPSILON.into_f64() > 1e-10 {
        (2e-5, 1e-3)
    } else {
        (5e-12, 5e-9)
    };
    let absolute = S::try_from_f64_lossy(absolute).unwrap();
    let relative = S::try_from_f64_lossy(relative).unwrap();
    // Use the existing complex-Bessel absolute floors, in norm coordinates so
    // exact component zeros beside exponentially large components are conditioned.
    assert!(
        (actual - expected).norm() <= (S::TWO.sqrt() * absolute).max(relative * expected.norm()),
        "{row}: {actual:?} expected {expected:?}"
    );
}

fn report<S: RealScalar>(
    row: &str,
    plain: Complex<S>,
    result: SpecialFunctionResult<Complex<S>>,
    expected: Complex<S>,
    derivative: Complex<S>,
) {
    assert!(result.termination().is_success(), "{row}: {result:?}");
    let actual = result.try_value().unwrap();
    assert_eq!(plain, actual, "{row}: plain/detailed");
    close(row, actual, expected);
    let rounding = |z: Complex<S>| S::TWO * S::EPSILON * z.norm();
    assert!(
        (actual - expected).norm() <= result.value().unwrap().absolute_error() + rounding(expected),
        "{row}: actual {actual:?}, expected {expected:?}, reported value error {:?}",
        result.value().unwrap().absolute_error()
    );
    let d = result.derivative().unwrap();
    close(row, *d.value(), derivative);
    assert!(
        (*d.value() - derivative).norm() <= d.absolute_error() + rounding(derivative),
        "{row}: reported derivative error"
    );
}

fn check<S: RealScalar>(width: &str, decode: impl Fn(&str) -> S)
where
    Complex<S>: NumericScalar<Real = S>,
{
    for row in include_str!("data/bessel_signed_axes.tsv")
        .lines()
        .filter(|row| !row.starts_with('#'))
    {
        let fields: Vec<_> = row.split_whitespace().collect();
        if fields[0] != width {
            continue;
        }
        let c = |index| Complex::new(decode(fields[index]), decode(fields[index + 1]));
        let nu = c(2);
        let z = c(4);
        let opposite_zero = Complex::new(-z.re, z.im);
        let expected = |scaled: bool, degree| c(6 + usize::from(scaled) * 8 + degree * 2);
        let options = BesselOptions::default();
        macro_rules! derivatives {
            ($plain:ident, $detailed:ident, $scaled:expr) => {
                for degree in 0..=3 {
                    let value = $plain(nu, z, degree).unwrap_or_else(|e| panic!("{row}: {e}"));
                    close(row, value, $plain(nu, opposite_zero, degree).unwrap());
                    let result =
                        $detailed(nu, z, degree, options).unwrap_or_else(|e| panic!("{row}: {e}"));
                    assert!(result.termination().is_success(), "{row}: {result:?}");
                    assert_eq!(value, result.try_value().unwrap(), "{row}: degree {degree}");
                    close(row, value, expected($scaled, degree as usize));
                }
            };
        }
        macro_rules! cylindrical {
            ($plain:ident, $detailed:ident, $derivative:ident, $derivative_detailed:ident,
             $scaled:ident, $scaled_detailed:ident $(; $scaled_derivative:ident, $scaled_derivative_detailed:ident)?) => {{
                close(row, $plain(nu, z).unwrap(), $plain(nu, opposite_zero).unwrap());
                close(row, $scaled(nu, z).unwrap(), $scaled(nu, opposite_zero).unwrap());
                report(row, $plain(nu, z).unwrap(), $detailed(nu, z, options).unwrap(),
                    expected(false, 0), expected(false, 1));
                report(row, $scaled(nu, z).unwrap(), $scaled_detailed(nu, z, options).unwrap(),
                    expected(true, 0), expected(true, 1));
                derivatives!($derivative, $derivative_detailed, false);
                $(derivatives!($scaled_derivative, $scaled_derivative_detailed, true);)?
            }};
        }
        macro_rules! spherical {
            ($plain:ident, $detailed:ident, $scaled:ident, $scaled_detailed:ident) => {{
                let order = nu.re.try_to_usize_exact().unwrap() as u32;
                let options = SphericalBesselOptions::default();
                close(
                    row,
                    $plain(order, z).unwrap(),
                    $plain(order, opposite_zero).unwrap(),
                );
                close(
                    row,
                    $scaled(order, z).unwrap(),
                    $scaled(order, opposite_zero).unwrap(),
                );
                report(
                    row,
                    $plain(order, z).unwrap(),
                    $detailed(order, z, options).unwrap(),
                    expected(false, 0),
                    expected(false, 1),
                );
                report(
                    row,
                    $scaled(order, z).unwrap(),
                    $scaled_detailed(order, z, options).unwrap(),
                    expected(true, 0),
                    expected(true, 1),
                );
            }};
        }
        match fields[1] {
            "j" => cylindrical!(
                bessel_j,
                bessel_j_with_options,
                bessel_j_derivative,
                bessel_j_derivative_with_options,
                bessel_j_scaled,
                bessel_j_scaled_with_options
            ),
            "y" => cylindrical!(
                bessel_y,
                bessel_y_with_options,
                bessel_y_derivative,
                bessel_y_derivative_with_options,
                bessel_y_scaled,
                bessel_y_scaled_with_options
            ),
            "i" => {
                cylindrical!(bessel_i, bessel_i_with_options, bessel_i_derivative, bessel_i_derivative_with_options,
                bessel_i_scaled, bessel_i_scaled_with_options; bessel_i_scaled_derivative, bessel_i_scaled_derivative_with_options)
            }
            "k" => {
                cylindrical!(bessel_k, bessel_k_with_options, bessel_k_derivative, bessel_k_derivative_with_options,
                bessel_k_scaled, bessel_k_scaled_with_options; bessel_k_scaled_derivative, bessel_k_scaled_derivative_with_options)
            }
            "h1" => cylindrical!(
                hankel_1,
                hankel_1_with_options,
                hankel_1_derivative,
                hankel_1_derivative_with_options,
                hankel_1_scaled,
                hankel_1_scaled_with_options
            ),
            "h2" => cylindrical!(
                hankel_2,
                hankel_2_with_options,
                hankel_2_derivative,
                hankel_2_derivative_with_options,
                hankel_2_scaled,
                hankel_2_scaled_with_options
            ),
            "sh1" => spherical!(
                spherical_hankel_1,
                spherical_hankel_1_with_options,
                spherical_hankel_1_scaled,
                spherical_hankel_1_scaled_with_options
            ),
            "sh2" => spherical!(
                spherical_hankel_2,
                spherical_hankel_2_with_options,
                spherical_hankel_2_scaled,
                spherical_hankel_2_scaled_with_options
            ),
            _ => panic!("{row}: unknown family"),
        }
    }
}

#[test]
fn f32_oracles() {
    check("32", |s| {
        f32::from_bits(u32::from_str_radix(s, 16).unwrap())
    });
}

#[test]
fn f64_oracles() {
    check("64", |s| {
        f64::from_bits(u64::from_str_radix(s, 16).unwrap())
    });
}
