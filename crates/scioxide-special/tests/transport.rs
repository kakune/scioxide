//! Independent fixtures, identities, resource bounds and allocation tests.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::{Budget, RealScalar, SciOxideError};
use scioxide_special::*;

const NAMES: [&str; 9] = [
    "synchrotron_1",
    "synchrotron_2",
    "synchrotron_1_scaled",
    "synchrotron_2_scaled",
    "transport_2",
    "transport_3",
    "transport_4",
    "transport_5",
    "clausen",
];
fn detailed<S: RealScalar>(
    name: &str,
    x: S,
    budget: Budget,
) -> Result<SpecialFunctionResult<S>, SciOxideError> {
    let s = SynchrotronOptions::new(budget);
    let t = TransportOptions::new(budget);
    match name {
        "synchrotron_1" => synchrotron_1_with_options(x, s),
        "synchrotron_2" => synchrotron_2_with_options(x, s),
        "synchrotron_1_scaled" => synchrotron_1_scaled_with_options(x, s),
        "synchrotron_2_scaled" => synchrotron_2_scaled_with_options(x, s),
        "transport_2" => transport_2_with_options(x, t),
        "transport_3" => transport_3_with_options(x, t),
        "transport_4" => transport_4_with_options(x, t),
        "transport_5" => transport_5_with_options(x, t),
        "clausen" => clausen_with_options(x, ClausenOptions::new(budget)),
        _ => panic!("unknown fixture {name}"),
    }
}
fn scalar<S: RealScalar>(name: &str, x: S) -> Result<S, SciOxideError> {
    match name {
        "synchrotron_1" => synchrotron_1(x),
        "synchrotron_2" => synchrotron_2(x),
        "synchrotron_1_scaled" => synchrotron_1_scaled(x),
        "synchrotron_2_scaled" => synchrotron_2_scaled(x),
        "transport_2" => transport_2(x),
        "transport_3" => transport_3(x),
        "transport_4" => transport_4(x),
        "transport_5" => transport_5(x),
        "clausen" => clausen(x),
        _ => panic!("unknown fixture"),
    }
}
fn default_result<S: RealScalar>(name: &str, x: S) -> SpecialFunctionResult<S> {
    detailed(name, x, Budget::try_from(128).unwrap()).unwrap()
}

macro_rules! representation_check {
    ($name:ident, $scalar:ty) => {
        fn $name(
            result: SpecialFunctionResult<$scalar>,
            reference: $scalar,
            expected_log: Option<$scalar>,
        ) {
            if let Some(scaled) = result.scaled() {
                let value = *scaled.value();
                let significand = value.try_projective().unwrap().significand();
                assert!(
                    significand == 0.0 || (significand.abs() >= 1.0 && significand.abs() < 2.0),
                    "scaled normalization: {value:?}"
                );
            }
            if reference != 0.0
                && reference.is_finite()
                && let Some(scaled) = result.scaled()
            {
                let exponent = i32::try_from(scaled.value().scale_exponent()).unwrap();
                let actual = f64::from(scaled.value().real().significand());
                let expected = f64::from(reference).scalbn(-exponent);
                let rounding = f64::from(reference.next_after(<$scalar>::INFINITY) - reference)
                    .scalbn(-exponent);
                assert!(
                    (actual - expected).abs() <= f64::from(scaled.absolute_error()) + rounding,
                    "scaled bound: {actual:e} vs {expected:e}, {scaled:?}"
                );
            }
            if let (Some(reference), Some(log)) = (expected_log, result.logarithmic())
                && let Some(actual) = log.value().log_magnitude()
            {
                let rounding = 8.0 * <$scalar>::EPSILON * reference.abs().max(1.0);
                assert!(
                    (actual - reference).abs() <= log.absolute_error() + rounding,
                    "log error {actual:?} {reference:?} {log:?}"
                );
            }
        }
    };
}
representation_check!(check_representations64, f64);
representation_check!(check_representations32, f32);

#[test]
fn independent_oracles_and_error_estimates() {
    let mut counts = [0usize; 2];
    for line in include_str!("data/transport_oracle.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = line.split('\t').collect();
        let name = fields[1];
        let input = u64::from_str_radix(fields[2], 16).unwrap();
        let expected = u64::from_str_radix(fields[3], 16).unwrap();
        let (actual, reference, error, budget, ulp_error) = if fields[0] == "f64" {
            counts[0] += 1;
            let x = f64::from_bits(input);
            let result = default_result(name, x);
            let value = result
                .try_value()
                .unwrap_or_else(|e| panic!("{line}: {e:?} {result:?}"));
            assert_eq!(value.to_bits(), scalar(name, x).unwrap().to_bits());
            let expected = f64::from_bits(expected);
            check_representations64(
                result,
                expected,
                (fields[4] != "-")
                    .then(|| f64::from_bits(u64::from_str_radix(fields[4], 16).unwrap())),
            );
            (
                value,
                expected,
                result.value().unwrap().absolute_error(),
                (5e-13_f64).max(5e-11 * expected.abs()),
                value.to_bits().abs_diff(expected.to_bits()),
            )
        } else {
            counts[1] += 1;
            let x = f32::from_bits(input as u32);
            let result = default_result(name, x);
            let value = result
                .try_value()
                .unwrap_or_else(|e| panic!("{line}: {e:?} {result:?}"));
            assert_eq!(value.to_bits(), scalar(name, x).unwrap().to_bits());
            let expected = f32::from_bits(expected as u32);
            check_representations32(
                result,
                expected,
                (fields[4] != "-")
                    .then(|| f32::from_bits(u32::from_str_radix(fields[4], 16).unwrap())),
            );
            (
                f64::from(value),
                f64::from(expected),
                f64::from(result.value().unwrap().absolute_error()),
                (5e-6_f64).max(2e-4 * f64::from(expected).abs()),
                u64::from(value.to_bits().abs_diff(expected.to_bits())),
            )
        };
        assert!(
            (actual - reference).abs() <= budget,
            "{line}: actual={actual:e} reference={reference:e}"
        );
        let min_normal = if fields[0] == "f64" {
            f64::MIN_POSITIVE
        } else {
            f64::from(f32::MIN_POSITIVE)
        };
        if reference.abs() < min_normal {
            assert_eq!(
                actual == 0.0,
                reference == 0.0,
                "zero classification {line}"
            );
            assert!(
                ulp_error <= if fields[0] == "f64" { 8 } else { 16 },
                "subnormal {line}: {actual:e} ULP={ulp_error}"
            );
            assert_eq!(
                actual.is_sign_negative(),
                reference.is_sign_negative(),
                "{line}"
            );
        }
        // The reference is itself rounded once; permit half a target ULP.
        let rounding = if fields[0] == "f64" {
            (reference.next_up() - reference).abs() * 0.5
        } else {
            f64::from((reference as f32).next_up() - reference as f32).abs() * 0.5
        };
        assert!(
            (actual - reference).abs() <= error + rounding,
            "error model {line}: actual={actual:e} ref={reference:e} error={error:e}"
        );
    }
    assert!(counts.iter().all(|&count| count > 200));
}

#[test]
fn domains_endpoints_and_resource_bounds() {
    for name in NAMES {
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            {
                let outcome = &(scalar(name, x));
                let err = outcome.as_ref().expect_err("expected error");
                assert!(matches!(
                    err.view(),
                    scioxide_core::SciOxideErrorView::NonFinite { .. }
                ));
            }
            {
                let outcome = &(scalar(name, x as f32));
                let err = outcome.as_ref().expect_err("expected error");
                assert!(matches!(
                    err.view(),
                    scioxide_core::SciOxideErrorView::NonFinite { .. }
                ));
            }
        }
        if name != "clausen" {
            {
                let outcome = &(scalar(name, -1.0_f64));
                let err = outcome.as_ref().expect_err("expected error");
                assert!(matches!(
                    err.view(),
                    scioxide_core::SciOxideErrorView::DomainError { .. }
                ));
            }
        }
        let origin = detailed(name, -0.0_f64, Budget::ONE).unwrap();
        assert_eq!(origin.counts().iterations(), 0);
        assert_eq!(origin.termination(), SpecialFunctionTermination::Exact);
        assert_eq!(
            origin.try_value().unwrap().to_bits(),
            if name == "clausen" {
                (-0.0_f64).to_bits()
            } else {
                0
            }
        );
        for x in [0.1, 2., 4., 5., 100., f64::MAX] {
            for units in [1, 2, 10, 16, 24, 32, 48, 128] {
                let result = detailed(name, x, Budget::try_from(units).unwrap()).unwrap();
                assert!(result.counts().iterations() <= units);
                if result.termination() == SpecialFunctionTermination::IterationLimit {
                    {
                        let outcome = &(result.try_value());
                        let err = outcome.as_ref().expect_err("expected error");
                        assert!(matches!(
                            err.view(),
                            scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
                        ));
                    }
                }
            }
        }
    }
}

#[test]
fn scaled_synchrotron_retains_underflowed_tail() {
    // A log can remain useful when its absolute uncertainty makes conversion
    // back to a linear significand ill-conditioned in the working precision.
    let large = synchrotron_1_with_options(1e10_f32, SynchrotronOptions::default()).unwrap();
    assert_eq!(large.try_value().unwrap(), 0.0);
    assert!(large.logarithmic().is_some());
    assert!(large.scaled().is_none());
    let large = synchrotron_2_with_options(1e18_f64, SynchrotronOptions::default()).unwrap();
    assert!(large.logarithmic().is_some());
    assert!(large.scaled().is_none());
    for name in ["synchrotron_1", "synchrotron_2"] {
        for x in [1.0_f64, 4., 10., 100.] {
            let ordinary = scalar(name, x).unwrap();
            let scaled = if name == "synchrotron_1" {
                synchrotron_1_scaled(x).unwrap()
            } else {
                synchrotron_2_scaled(x).unwrap()
            };
            assert!((ordinary - scaled * (-x).exp()).abs() <= 1e-13 * ordinary);
        }
        for x in [1000., 1e20, f64::MAX] {
            let result = default_result(name, x);
            assert_eq!(result.try_value().unwrap(), 0.);
            assert_eq!(result.scalar_representation(), ScalarRepresentation::Finite);
            assert!(result.logarithmic().is_some());
            assert_eq!(result.scaled().is_some(), x == 1000.);
        }
    }
}

#[test]
fn defining_identities_and_asymptotics() {
    let h = 1e-5;
    for x in [0.1_f64, 0.5, 1., 2., 4., 10.] {
        let derivative = (synchrotron_1(x + h).unwrap() - synchrotron_1(x - h).unwrap()) / (2. * h);
        let expected = synchrotron_1(x).unwrap() / x - x * bessel_k(5. / 3., x).unwrap();
        assert!((derivative - expected).abs() < 2e-7);
        assert!((synchrotron_2(x).unwrap() - x * bessel_k(2. / 3., x).unwrap()).abs() < 1e-8);
        for (n, name) in [
            (2, "transport_2"),
            (3, "transport_3"),
            (4, "transport_4"),
            (5, "transport_5"),
        ] {
            let derivative =
                (scalar(name, x + h).unwrap() - scalar(name, x - h).unwrap()) / (2. * h);
            let expected = x.powi(n) * (-x).exp() / (-(-x).exp_m1()).powi(2);
            assert!(
                (derivative - expected).abs() < 2e-7,
                "{name} {x}: {derivative} {expected}"
            );
        }
        let derivative = (clausen(x + h).unwrap() - clausen(x - h).unwrap()) / (2. * h);
        assert!((derivative + (2. * (x / 2.).sin().abs()).ln()).abs() < 2e-7);
        assert!(
            (clausen(x + 2. * core::f64::consts::PI).unwrap() - clausen(x).unwrap()).abs() < 2e-14
        );
        assert_eq!(clausen(-x).unwrap(), -clausen(x).unwrap());
    }
    assert!((clausen(core::f64::consts::FRAC_PI_2).unwrap() - 0.915_965_594_177_219).abs() < 2e-15);
    for x in [1e-12_f64, 1e-9] {
        assert!((synchrotron_1(x).unwrap() / x.cbrt() - 2.149_528_241_534_479).abs() < 2e-6);
        assert!((synchrotron_2(x).unwrap() / x.cbrt() - 1.074_764_120_767_239_4).abs() < 2e-6);
    }
    for n in 2..=5 {
        let name = NAMES[n + 2];
        let limit = factorial::<f64>(n).unwrap() * riemann_zeta(n as f64).unwrap();
        // The cross-family identity includes the existing zeta error envelope.
        assert!((scalar(name, 1000.).unwrap() - limit).abs() < 5e-9 * limit);
        let values: Vec<_> = (0..100)
            .map(|i| scalar(name, f64::from(i) / 10.).unwrap())
            .collect();
        assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
    }
}

#[test]
fn all_regimes_and_failures_execute() {
    for name in NAMES {
        for x in [0., f64::from_bits(1), 0.1, 1., 2., 4., 5., 1000., f64::MAX] {
            let _ = default_result(name, std::hint::black_box(x));
            let x32 = if x > f64::from(f32::MAX) {
                f32::MAX
            } else {
                x as f32
            };
            let _ = default_result(name, std::hint::black_box(x32));
        }
        let _ = scalar(name, f64::NAN).unwrap_err();
        let _ = detailed(name, 2., Budget::ONE).unwrap();
        if name != "clausen" {
            let _ = scalar(name, -1.0_f64).unwrap_err();
        }
    }
}
