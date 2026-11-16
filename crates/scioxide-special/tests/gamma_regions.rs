//! Independent mpmath references at cancellation and range boundaries.
#![allow(clippy::unwrap_used)]
use scioxide_core::Budget;
use scioxide_special::{
    GammaOptions, SpecialFunctionTermination, gamma, gamma_with_options, ln_gamma,
    ln_gamma_with_options,
};

#[test]
fn gamma_and_positive_log_gamma_meet_component_ulp_targets() {
    for row in include_str!("data/gamma_regions.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = row.split('\t').collect();
        let bits = u64::from_str_radix(fields[3].strip_prefix("f:").unwrap(), 16).unwrap();
        let expected = u64::from_str_radix(fields[4], 16).unwrap();
        if fields[1] == "f64" {
            let x = f64::from_bits(bits);
            if f64::from_bits(expected).is_infinite() {
                {
                    let outcome = &(gamma(x));
                    let err = outcome.as_ref().expect_err("expected error");
                    assert!(matches!(
                        err.view(),
                        scioxide_core::SciOxideErrorView::NumericalInstability {
                            issue: scioxide_core::NumericalIssue::Overflow,
                            ..
                        }
                    ));
                }
                continue;
            }
            // Negative log-gamma reflection is measured separately: its error
            // depends on cancellation between logarithms near negative roots.
            if fields[2] == "ln_gamma" && x < 0.0 {
                continue;
            }
            let result = if fields[2] == "gamma" {
                gamma(x)
            } else {
                ln_gamma(x)
            }
            .unwrap();
            assert!(
                result.to_bits().abs_diff(expected) <= 4,
                "{}({x:?}): {} ulp",
                fields[2],
                result.to_bits().abs_diff(expected)
            );
        } else {
            let x = f32::from_bits(bits as u32);
            if f32::from_bits(expected as u32).is_infinite() {
                {
                    let outcome = &(gamma(x));
                    let err = outcome.as_ref().expect_err("expected error");
                    assert!(matches!(
                        err.view(),
                        scioxide_core::SciOxideErrorView::NumericalInstability {
                            issue: scioxide_core::NumericalIssue::Overflow,
                            ..
                        }
                    ));
                }
                continue;
            }
            let result = if fields[2] == "gamma" {
                gamma(x)
            } else {
                ln_gamma(x)
            }
            .unwrap();
            assert!(
                u64::from(result.to_bits()).abs_diff(expected) <= 1,
                "{}({x:?})",
                fields[2]
            );
        }
    }
}

#[test]
fn fixed_kernel_options_enforce_work_and_publish_estimates() {
    let tiny_budget = GammaOptions::default().with_iteration_budget(Budget::ONE);
    assert!(gamma_with_options(3.25_f64, tiny_budget).is_err());
    assert!(ln_gamma_with_options(1.00000001_f64, tiny_budget).is_err());
    let result = gamma_with_options(3.25_f64, GammaOptions::default()).unwrap();
    assert_eq!(result.termination(), SpecialFunctionTermination::Converged);
    let value = result.value().unwrap();
    assert_eq!(
        value.reliability(),
        scioxide_core::EstimateReliability::StatisticallyCalibrated
    );
    assert!(value.absolute_error() <= 8.0 * f64::EPSILON * value.value().abs());
}

#[test]
fn stirling_gamma_retains_accuracy_and_identical_projections() {
    // mpmath 1.3.0 at 160/240 digits, with identical rounded references
    // for exact IEEE inputs: the 16 handoff, large arguments and range edge.
    for (input, expected) in [
        (0x402fffffffffffff, 0x42730777757fffe6),
        (0x4030000000000000, 0x4273077775800000),
        (0x4030000000000001, 0x4273077775800034),
        (0x4030800000000000, 0x4292e1900e84c080),
        (0x4036400000000000, 0x4417e4112cd22499),
        (0x4040400000000000, 0x4721d8e421e2396c),
        (0x4050200000000000, 0x523fd48ab83e8085),
        (0x4060100000000000, 0x6c79496c7383f48c),
        (0x4065500000000000, 0x7f69589f849167a8),
        (0x406573fae561f646, 0x7feffffffffff92c),
    ] {
        let x = f64::from_bits(input);
        let expected = f64::from_bits(expected);
        let actual = gamma(x).unwrap();
        let detailed = gamma_with_options(x, GammaOptions::default()).unwrap();
        let unit = expected.next_up() - expected;
        assert!((actual - expected).abs() <= 4.0 * unit, "gamma({x})");
        assert_eq!(actual.to_bits(), detailed.try_value().unwrap().to_bits());
    }
    for (input, expected) in [
        (0x417fffff, 0x53983ba2),
        (0x41800000, 0x53983bbc),
        (0x41800001, 0x53983bf0),
        (0x41840000, 0x54970c80),
        (0x420c290f, 0x7f7fff81),
    ] {
        let x = f32::from_bits(input);
        let expected = f32::from_bits(expected);
        let actual = gamma(x).unwrap();
        let detailed = gamma_with_options(x, GammaOptions::default()).unwrap();
        let unit = expected.next_up() - expected;
        assert!((actual - expected).abs() <= unit, "gamma({x})");
        assert_eq!(actual.to_bits(), detailed.try_value().unwrap().to_bits());
    }
}

#[test]
fn stirling_gamma_preserves_overflow_errors_in_both_precisions() {
    let x64 = f64::from_bits(0x406573fae561f648);
    let x32 = f32::from_bits(0x420c2910);
    for outcome in [
        gamma(x64).map(|_| ()),
        gamma_with_options(x64, GammaOptions::default())
            .and_then(|evaluation| evaluation.try_value())
            .map(|_| ()),
        gamma(x32).map(|_| ()),
        gamma_with_options(x32, GammaOptions::default())
            .and_then(|evaluation| evaluation.try_value())
            .map(|_| ()),
    ] {
        assert!(matches!(
            outcome.unwrap_err().view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
    }
}
