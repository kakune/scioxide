//! Independent scaled enclosures across large arguments and the Gauss singularity.
#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/scaled.rs"]
mod scaled;
use scioxide_core::{Complex, RealScalar};
use scioxide_special::*;

fn coordinate(text: &str) -> Complex<f64> {
    let (re, im) = text.split_once(',').unwrap();
    Complex::new(re.parse().unwrap(), im.parse().unwrap())
}
fn fixtures<S: RealScalar>() {
    let mut failures = Vec::new();
    for line in include_str!("data/complex_hypergeometric.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        if f[0] != if S::MANTISSA_DIGITS == 24 { "32" } else { "64" } {
            continue;
        }
        let convert = |z: Complex<f64>| {
            Complex::new(
                S::try_from_f64_lossy(z.re).unwrap(),
                S::try_from_f64_lossy(z.im).unwrap(),
            )
        };
        let args: Vec<_> = f[4].split(';').map(|x| convert(coordinate(x))).collect();
        let z = convert(coordinate(f[5]));
        let plan = HypergeometricFunction::pfq(
            &args[..args.len() - 1],
            &args[args.len() - 1..],
            f[2] == "1",
            Default::default(),
        )
        .unwrap();
        let selector = match f[3] {
            "none" => HypergeometricDerivative::None,
            "z" => HypergeometricDerivative::Argument,
            "a0" => HypergeometricDerivative::Numerator(0),
            "b0" => HypergeometricDerivative::Numerator(1),
            "c0" => HypergeometricDerivative::Denominator(0),
            _ => panic!("unknown selector"),
        };
        let result = plan.evaluate(z, selector);
        let result = match result {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("{line}: {e:?}"));
                continue;
            }
        };
        let exponent: i64 = f[6].parse().unwrap();
        let (actual, error) = if selector == HypergeometricDerivative::None {
            let Some(form) = result.scaled() else {
                failures.push(format!("{line}: missing scaled value"));
                continue;
            };
            (
                scaled::at_exponent(form.value(), exponent),
                scaled::error_at_exponent(form.value(), form.absolute_error().into_f64(), exponent),
            )
        } else {
            let Some(form) = result.derivative() else {
                failures.push(format!("{line}: missing derivative {result:?}"));
                continue;
            };
            (
                Complex::new(
                    form.value().re.into_f64().scalbn(-exponent as i32),
                    form.value().im.into_f64().scalbn(-exponent as i32),
                ),
                form.absolute_error().into_f64().scalbn(-exponent as i32),
            )
        };
        let bounds: Vec<f64> = f[7..].iter().map(|s| s.parse().unwrap()).collect();
        let gap = |x: f64, i: usize| (bounds[2 * i] - x).max(x - bounds[2 * i + 1]).max(0.0);
        let observed = gap(actual.re, 0).hypot(gap(actual.im, 1));
        let eps = S::EPSILON.into_f64();
        if !result.termination().is_success()
            || observed > error + 2.0 * eps * actual.norm()
            || observed > (256.0 * eps).max(2e-12) * actual.norm().max(1e-300)
        {
            failures.push(format!(
                "{line}: {actual:?}; observed {observed:e}, reported {error:e}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
#[test]
fn certified_f64() {
    fixtures::<f64>();
}
#[test]
fn certified_f32() {
    fixtures::<f32>();
}
