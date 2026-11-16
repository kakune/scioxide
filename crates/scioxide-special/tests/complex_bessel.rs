#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/fixture.rs"]
mod fixture;
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, Complex, NumericScalar, RealScalar};
use scioxide_special::*;
fn evaluate<S: RealScalar>(
    kind: &str,
    n: S,
    z: Complex<S>,
    o: BesselOptions,
) -> Result<SpecialFunctionResult<Complex<S>>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let n = Complex::new(n, S::ZERO);
    match kind {
        "j" => bessel_j_with_options(n, z, o),
        "y" => bessel_y_with_options(n, z, o),
        "i" => bessel_i_with_options(n, z, o),
        "k" => bessel_k_with_options(n, z, o),
        "h1" => hankel_1_with_options(n, z, o),
        "h2" => hankel_2_with_options(n, z, o),
        "js" => bessel_j_scaled_with_options(n, z, o),
        "ys" => bessel_y_scaled_with_options(n, z, o),
        "is" => bessel_i_scaled_with_options(n, z, o),
        "ks" => bessel_k_scaled_with_options(n, z, o),
        "h1s" => hankel_1_scaled_with_options(n, z, o),
        _ => hankel_2_scaled_with_options(n, z, o),
    }
}
fn check<S: RealScalar>(width: &str, decode: impl Fn(&str) -> S)
where
    Complex<S>: NumericScalar<Real = S>,
{
    let mut checker = fixture::Checker::new("complex_bessel", width);
    for row in include_str!("data/complex_bessel.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let p: Vec<_> = row.split_whitespace().collect();
        if p[1] != width {
            continue;
        }
        let z = Complex::new(decode(p[3]), decode(p[4]));
        let reference = Complex::new(decode(p[5]), decode(p[6]));
        let derivative = Complex::new(decode(p[7]), decode(p[8]));
        let r = evaluate(p[0], decode(p[2]), z, BesselOptions::default())
            .unwrap_or_else(|e| panic!("{row}: {e}"));
        let v = r.try_value().unwrap_or_else(|e| panic!("{row}: {e}"));
        let d = *r.derivative().unwrap().value();
        checker.complex(
            "value",
            v,
            reference,
            Some(r.value().unwrap().absolute_error()),
            row,
        );
        checker.complex(
            "derivative",
            d,
            derivative,
            Some(r.derivative().unwrap().absolute_error()),
            row,
        );
        // The reported errors cover the references, allowing their rounding.
        let rounding = |c: Complex<S>| S::EPSILON * (c.re.abs() + c.im.abs());
        let value_error = r.value().unwrap().absolute_error();
        assert!(
            (v - reference).norm() <= value_error + rounding(reference),
            "{row}: value error {value_error:?}"
        );
        let derivative_error = r.derivative().unwrap().absolute_error();
        assert!(
            (d - derivative).norm() <= derivative_error + rounding(derivative),
            "{row}: derivative error {derivative_error:?}"
        );
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
#[test]
fn imaginary_axis_hankel_and_y_ignore_the_real_zero_sign() {
    for (kind, order, imaginary, reference) in [
        (
            "h1",
            2.5,
            -3.0,
            Complex::new(-2.180854176352964, 2.180854176352964),
        ),
        (
            "h1",
            0.5,
            -3.0,
            Complex::new(6.542562529058893, -6.542562529058893),
        ),
        (
            "y",
            2.5,
            -3.0,
            Complex::new(1.109347377804676, 1.109347377804676),
        ),
        (
            "h2",
            2.5,
            3.0,
            Complex::new(-2.180854176352964, -2.180854176352964),
        ),
    ] {
        let values = [0.0, -0.0].map(|real| {
            evaluate(
                kind,
                order,
                Complex::new(real, imaginary),
                BesselOptions::default(),
            )
            .unwrap()
            .try_value()
            .unwrap()
        });
        for value in values {
            assert!(
                (value - reference).norm() < 2e-12,
                "{kind}({order}, {imaginary}i): {value:?}"
            );
        }
        assert!((values[0] - values[1]).norm() < 2e-12);
    }
}
#[test]
fn budgets_and_representations() {
    let o = BesselOptions::default().with_iteration_budget(Budget::ONE);
    let r =
        bessel_j_with_options(Complex::new(2.5_f64, 0.0), Complex::new(2.0, 1.0), o).unwrap_err();
    let r_counts = publication::exhausted_counts(&r);
    assert_eq!(r_counts.iterations(), 1);
    let z = Complex::new(2.0, 1000.0);
    let scaled = bessel_j_scaled(Complex::new(0.0_f64, 0.0), z).unwrap();
    assert!(scaled.is_finite());
    let detailed =
        bessel_j_with_options(Complex::new(0.0_f64, 0.0), z, BesselOptions::default()).unwrap();
    assert!(detailed.value().is_none());
    assert!(detailed.logarithmic().is_some());
}
