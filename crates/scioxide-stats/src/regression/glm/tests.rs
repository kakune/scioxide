use super::GlmFamily;
use super::family::nonnegative_deviance;
use crate::regression::number;
use alloc::vec::Vec;
use scioxide_core::{
    NumericalIssue, RealScalar, Scaled, SciOxideError, SciOxideErrorKind, SciOxideErrorView,
};

fn decode<S: RealScalar>(bits: &str, scalar: &str) -> S {
    let value = if scalar == "f32" {
        f64::from(f32::from_bits(u32::from_str_radix(bits, 16).unwrap()))
    } else {
        f64::from_bits(u64::from_str_radix(bits, 16).unwrap())
    };
    S::try_from_f64_lossy(value).unwrap()
}

fn check_reference<S: RealScalar>(
    actual: Result<S, SciOxideError>,
    expected: &str,
    scalar: &str,
    case: &str,
) {
    if expected == "overflow" {
        assert!(
            matches!(
                actual.unwrap_err().view(),
                SciOxideErrorView::NumericalInstability {
                    issue: NumericalIssue::Overflow,
                    ..
                }
            ),
            "{case}"
        );
    } else {
        let actual = actual.unwrap();
        let expected: S = decode(expected, scalar);
        let tiny: S = decode(
            if scalar == "f32" {
                "00000001"
            } else {
                "0000000000000001"
            },
            scalar,
        );
        let tolerance = number::<S>(512.0).unwrap() * S::EPSILON * expected.abs() + tiny + tiny;
        assert!(
            (actual - expected).abs() <= tolerance,
            "{case}: actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
        );
        if expected == S::ZERO {
            assert_eq!(actual, S::ZERO, "{case}");
        }
    }
}

fn reference_cases<S: RealScalar>(scalar: &str) {
    for line in include_str!("../../../tests/data/glm_deviance.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let row: Vec<_> = line.split('\t').collect();
        if row[0] != scalar {
            continue;
        }
        let family = match row[1] {
            "Gaussian" => GlmFamily::Gaussian,
            "Binomial" => GlmFamily::Binomial,
            "Poisson" => GlmFamily::Poisson,
            "Gamma" => GlmFamily::Gamma,
            "InverseGaussian" => GlmFamily::InverseGaussian,
            _ => unreachable!(),
        };
        let y = decode::<S>(row[3], scalar);
        let mu = decode::<S>(row[4], scalar);
        let case = alloc::format!("{scalar} {family:?} {}", row[2]);
        check_reference(
            family.deviance(y, mu).and_then(Scaled::round),
            row[5],
            scalar,
            &case,
        );
        check_reference(
            family.score(y, mu).and_then(Scaled::round),
            row[6],
            scalar,
            &case,
        );
    }
}

#[test]
fn independent_deviance_and_score_references_f64() {
    reference_cases::<f64>("f64");
}

#[test]
fn independent_deviance_and_score_references_f32() {
    reference_cases::<f32>("f32");
}

#[test]
fn rejects_nonfinite_and_invalid_family_domains_before_equality() {
    for family in [
        GlmFamily::Gaussian,
        GlmFamily::Binomial,
        GlmFamily::Poisson,
        GlmFamily::Gamma,
        GlmFamily::InverseGaussian,
    ] {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for (y, mu) in [(bad, 0.5), (0.5, bad), (bad, bad)] {
                assert_eq!(
                    family.deviance(y, mu).unwrap_err().kind(),
                    SciOxideErrorKind::NonFinite
                );
            }
        }
        if family != GlmFamily::Gaussian {
            assert!(family.deviance(0.5, 0.0).is_err());
            assert!(family.deviance(-1.0, -1.0).is_err());
        }
        if matches!(family, GlmFamily::Gamma | GlmFamily::InverseGaussian) {
            assert!(family.deviance(0.0, 1.0).is_err());
        }
    }
    assert!(GlmFamily::Binomial.deviance(1.0, 1.0).is_err());
    assert!(GlmFamily::Binomial.deviance(2.0, 0.5).is_err());
}

#[test]
fn rejects_negative_results_instead_of_clamping() {
    let error = nonnegative_deviance(Scaled::from_value(-f64::EPSILON).unwrap()).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Breakdown,
            ..
        }
    ));
}
