#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/fixture.rs"]
mod fixture;
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_special::*;
fn check<S: RealScalar>(width: &str, decode: impl Fn(&str) -> S)
where
    Complex<S>: NumericScalar<Real = S>,
{
    let mut checker = fixture::Checker::new("bessel_turning", width);
    for row in include_str!("data/bessel_turning.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let p: Vec<_> = row.split_whitespace().collect();
        if p[1] != width {
            continue;
        }
        let n = decode(p[2]);
        let x = decode(p[3]);
        let z = Complex::new(x, decode(p[4]));
        let expected = Complex::new(decode(p[5]), decode(p[6]));
        let real = |result: Result<SpecialFunctionResult<S>, scioxide_core::SciOxideError>| {
            let r = result?;
            Ok::<_, scioxide_core::SciOxideError>((
                Complex::new(r.try_value()?, S::ZERO),
                r.value().unwrap().absolute_error(),
            ))
        };
        let complex =
            |result: Result<SpecialFunctionResult<Complex<S>>, scioxide_core::SciOxideError>| {
                let r = result?;
                Ok::<_, scioxide_core::SciOxideError>((
                    r.try_value()?,
                    r.value().unwrap().absolute_error(),
                ))
            };
        let (v, error) = match p[0] {
            "j" => real(bessel_j_with_options(n, x, BesselOptions::default())),
            "y" => real(bessel_y_with_options(n, x, BesselOptions::default())),
            "is" => real(bessel_i_scaled_with_options(n, x, BesselOptions::default())),
            "ks" => real(bessel_k_scaled_with_options(n, x, BesselOptions::default())),
            "kl" => real(bessel_k_log_with_options(n, x, BesselOptions::default())),
            "jc" => complex(bessel_j_with_options(
                Complex::new(n, S::ZERO),
                z,
                BesselOptions::default(),
            )),
            "yc" => complex(bessel_y_with_options(
                Complex::new(n, S::ZERO),
                z,
                BesselOptions::default(),
            )),
            "ic" => complex(bessel_i_with_options(
                Complex::new(n, S::ZERO),
                z,
                BesselOptions::default(),
            )),
            _ => complex(bessel_k_with_options(
                Complex::new(n, S::ZERO),
                z,
                BesselOptions::default(),
            )),
        }
        .unwrap_or_else(|e| panic!("{row}: {e}"));
        checker.complex("value", v, expected, Some(error), row);
    }
}
#[test]
fn f32_turning() {
    check("32", |s| {
        f32::from_bits(u32::from_str_radix(s, 16).unwrap())
    });
}
#[test]
fn f64_turning() {
    check("64", |s| {
        f64::from_bits(u64::from_str_radix(s, 16).unwrap())
    });
}
