//! Negative-order rational and continued polylogarithms, including range forms.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_special::*;

fn check<S: RealScalar>(scalar: &str, decode: impl Fn(&str) -> S)
where
    Complex<S>: NumericScalar<Real = S>,
{
    for row in include_str!("data/negative_polylog.tsv")
        .lines()
        .filter(|r| !r.starts_with('#'))
    {
        let f: Vec<_> = row.split_whitespace().collect();
        if f[0] != scalar {
            continue;
        }
        let order = decode(f[1]);
        let z = Complex::new(decode(f[2]), decode(f[3]));
        let expected = Complex::new(decode(f[4]), decode(f[5]));
        let logarithm = decode(f[6]);
        let r = polylog_with_options(Complex::new(order, S::ZERO), z, PolylogOptions::default())
            .unwrap_or_else(|e| panic!("{row}: {e}"));
        assert!(r.termination().is_success(), "{row}");
        if expected.is_finite() {
            let actual = r.try_value().unwrap_or_else(|e| panic!("{row}: {e}"));
            let rounding = S::EPSILON * expected.norm() * S::TWO + S::MIN_SUBNORMAL;
            assert!(
                (actual - expected).norm() <= r.value().unwrap().absolute_error() + rounding,
                "{row}: actual={actual:?}, error={:?}",
                r.value().unwrap().absolute_error()
            );
            if z.im == S::ZERO {
                let real = polylog_with_options(order, z.re, PolylogOptions::default()).unwrap();
                assert_eq!(
                    real.try_value().unwrap(),
                    actual.re,
                    "real projection {row}"
                );
            }
        } else {
            assert_eq!(
                r.scalar_representation(),
                ScalarRepresentation::Overflow,
                "{row}"
            );
            let log = r
                .logarithmic()
                .unwrap_or_else(|| panic!("missing logarithm {row}"));
            let actual = log.value().log_magnitude().unwrap();
            assert!(
                (actual - logarithm).abs() <= log.absolute_error() + S::EPSILON * logarithm.abs(),
                "log {row}: {actual:?}"
            );
        }
    }
}

#[test]
fn negative_binary64() {
    check("f64", |s| {
        f64::from_bits(u64::from_str_radix(s, 16).unwrap())
    });
}
#[test]
fn negative_binary32() {
    check("f32", |s| {
        f32::from_bits(u32::from_str_radix(s, 16).unwrap())
    });
}

#[test]
fn dilogarithm_real_domain_and_shared_integer_identities() {
    for x in [-1e300_f64, -1., 0., 0.5, 1., 1.0_f64.next_up(), 2., 1e300] {
        assert_eq!(
            polylog(2.0, x).unwrap().to_bits(),
            dilog(x).unwrap().to_bits()
        );
    }
    for (n, x) in [(1, -1000.), (4, -13.45), (70, -3.), (100, -50.)] {
        let p = polylog_with_options(-(n as f64), x, PolylogOptions::default()).unwrap();
        let l = lerch_phi_with_options(x, -(n as f64), 1., LerchOptions::default()).unwrap();
        assert!(
            (p.try_value().unwrap() - x * l.try_value().unwrap()).abs()
                <= p.value().unwrap().absolute_error()
                    + x.abs() * l.value().unwrap().absolute_error()
        );
        let f =
            fermi_dirac_int_with_options(-n - 1, (-x).ln(), FermiDiracOptions::default()).unwrap();
        assert!(
            (p.try_value().unwrap() + f.try_value().unwrap()).abs()
                <= p.value().unwrap().absolute_error() + f.value().unwrap().absolute_error()
        );
    }
}

#[test]
fn real_overflow_preserves_range_forms() {
    fn check<S: RealScalar>(order: S)
    where
        Complex<S>: NumericScalar<Real = S>,
    {
        let options = PolylogOptions::default();
        let real = polylog_with_options(order, S::HALF, options).unwrap();
        let complex = polylog_with_options(order, Complex::new(S::HALF, S::ZERO), options).unwrap();
        assert_eq!(real.scalar_representation(), ScalarRepresentation::Overflow);
        assert_eq!(real.termination(), complex.termination());
        assert!(real.termination().is_success());
        assert_eq!(real.counts(), complex.counts());
        assert_eq!(real.algorithm(), complex.algorithm());
        assert!(real.value().is_none());
        assert_eq!(
            real.scaled().unwrap().value().real(),
            complex.scaled().unwrap().value().real()
        );
        let real_log = real.logarithmic().unwrap();
        let complex_log = complex.logarithmic().unwrap();
        assert!(
            (real_log.value().log_magnitude().unwrap()
                - complex_log.value().log_magnitude().unwrap())
            .abs()
                <= real_log.absolute_error() + complex_log.absolute_error()
        );
        for error in [
            polylog(order, S::HALF).unwrap_err(),
            real.try_value().unwrap_err(),
        ] {
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::Overflow,
                    ..
                }
            ));
        }
    }
    check(-200.0_f64);
    check(-40.0_f32);
    for order in [-100.0_f64, -150.0] {
        let real = polylog_with_options(order, 0.5, PolylogOptions::default()).unwrap();
        let complex = polylog(order, Complex::new(0.5, 0.0)).unwrap();
        assert_eq!(real.scalar_representation(), ScalarRepresentation::Finite);
        assert!(
            (real.try_value().unwrap() - complex.re).abs()
                <= real.value().unwrap().absolute_error()
        );
    }
}
