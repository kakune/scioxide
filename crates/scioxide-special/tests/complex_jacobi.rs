//! Independent Jacobi theta continuation with complex parameters.
#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/scaled.rs"]
mod scaled_reference;
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_special::*;
fn references<S: RealScalar>(width: &str)
where
    Complex<S>: NumericScalar<Real = S>,
{
    for row in include_str!("data/complex_jacobi.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let f: Vec<_> = row.split_whitespace().collect();
        if f[0] != width {
            continue;
        }
        let n = |i: usize| f[i].parse::<f64>().unwrap();
        let c = |i| {
            Complex::new(
                S::try_from_f64_lossy(n(i)).unwrap(),
                S::try_from_f64_lossy(n(i + 1)).unwrap(),
            )
        };
        let a = c(2);
        let m = c(4);
        let options = EllipticOptions::default();
        let r = match f[1] {
            "sn" => jacobi_sn_with_options(a, m, options),
            "cn" => jacobi_cn_with_options(a, m, options),
            "dn" => jacobi_dn_with_options(a, m, options),
            _ => jacobi_amplitude_with_options(a, m, options),
        }
        .unwrap_or_else(|error| panic!("{row}: {error:?}"));
        assert!(r.termination().is_success(), "{row}: {r:?}");
        let s = r.scaled().unwrap().value();
        let actual = scaled_reference::at_exponent(s, f[8].parse().unwrap());
        let expected = Complex::new(n(6), n(7));
        // The reported scaled error covers the reference, allowing its rounding.
        let scaled = r.scaled().unwrap();
        let reported = scaled_reference::error_at_exponent(
            scaled.value(),
            scaled.absolute_error().into_f64(),
            f[8].parse().unwrap(),
        );
        let precision = if width == "32" {
            f64::from(f32::EPSILON)
        } else {
            f64::EPSILON
        };
        assert!(
            (actual - expected).norm() <= reported + 2.0 * precision * expected.norm(),
            "{row}: reported scaled error {reported:e}"
        );
        assert!(
            (actual - expected).norm() < if width == "32" { 3e-6 } else { 5e-12 },
            "{row}: {actual:?} expected={expected:?}"
        );
    }
}
#[test]
fn independent_f64() {
    references::<f64>("64");
}
#[test]
fn independent_f32() {
    references::<f32>("32");
}

#[test]
fn identities_conjugation_and_local_components() {
    for m in [
        Complex::new(0.5, 0.25),
        Complex::new(2.0, 1.0),
        Complex::new(-0.9, 0.1),
    ] {
        for u in [
            Complex::new(0.05, 0.1),
            Complex::new(0.5, 0.25),
            Complex::new(2.0, -0.5),
        ] {
            let s = jacobi_sn(u, m).unwrap();
            let c = jacobi_cn(u, m).unwrap();
            let d = jacobi_dn(u, m).unwrap();
            let am = jacobi_amplitude(u, m).unwrap();
            assert!((s * s + c * c - Complex::new(1.0, 0.0)).norm() < 2e-12);
            assert!((m * s * s + d * d - Complex::new(1.0, 0.0)).norm() < 2e-12);
            assert!((am.sin() - s).norm() < 2e-12);
            assert!((am.cos() - c).norm() < 2e-12);
            assert!((jacobi_sn(u.conj(), m.conj()).unwrap() - s.conj()).norm() < 2e-12);
            let k = elliptic_k(m).unwrap();
            assert!((jacobi_sn(u + k * 2.0, m).unwrap() + s).norm() < 2e-12);
        }
    }
    let h = 1e-100;
    let m = Complex::new(0.5, 0.25);
    let u = Complex::new(0.0, h);
    let sn = jacobi_sn(u, m).unwrap();
    let dn = jacobi_dn(u, m).unwrap();
    let am = jacobi_amplitude(u, m).unwrap();
    assert!((sn.re / (-m.im * h * h * h / 6.0) - 1.0).abs() < 1e-12);
    assert!((dn.im / (m.im * h * h / 2.0) - 1.0).abs() < 1e-12);
    assert!((am.im / h - 1.0).abs() < 1e-12);
}
