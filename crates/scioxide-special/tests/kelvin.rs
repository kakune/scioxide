#![allow(clippy::unwrap_used, clippy::panic)]
#[path = "support/fixture.rs"]
mod fixture;
#[path = "support/publication.rs"]
mod publication;
use scioxide_core::{Budget, RealScalar};
use scioxide_special::*;

fn check<S: RealScalar>(width: &str, decode: impl Fn(&str) -> S) {
    let mut checker = fixture::Checker::new("kelvin", width);
    let o = KelvinOptions::default();
    let references: std::collections::BTreeMap<_, _> = include_str!("data/kelvin.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| {
            let p: Vec<_> = line.split_whitespace().collect();
            (p[0] == width).then(|| ((p[2], p[1]), (decode(p[3]), decode(p[4]))))
        })
        .collect();
    for line in include_str!("data/kelvin.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let p: Vec<_> = line.split_whitespace().collect();
        if p[0] != width {
            continue;
        }
        let x = decode(p[2]);
        let reference = decode(p[3]);
        let d = decode(p[4]);
        let result = match p[1] {
            "0" => ber_with_options(x, o),
            "1" => bei_with_options(x, o),
            "2" => ker_with_options(x, o),
            _ => kei_with_options(x, o),
        }
        .unwrap_or_else(|e| panic!("{line}: {e}"));
        let actual = result.try_value().unwrap_or_else(|e| panic!("{line}: {e}"));
        let partner = if matches!(p[1], "0" | "2") {
            if p[1] == "0" { "1" } else { "3" }
        } else if p[1] == "1" {
            "0"
        } else {
            "2"
        };
        let (other_value, other_derivative) = references
            .get(&(p[2], partner))
            .copied()
            .unwrap_or((S::ZERO, S::ZERO));
        let value_scale = reference.abs().max(other_value.abs());
        let derivative_scale = d.abs().max(other_derivative.abs());
        checker.sample(
            "value",
            (actual - reference).abs().into_f64(),
            value_scale.into_f64(),
            Some(result.value().unwrap().absolute_error().into_f64()),
            line,
        );
        let actual_d = match result.derivative() {
            Some(derivative) => *derivative.value(),
            None => {
                assert_eq!(d, S::ZERO, "{line}: missing representable derivative");
                S::ZERO
            }
        };
        checker.sample(
            "derivative",
            (actual_d - d).abs().into_f64(),
            derivative_scale.into_f64(),
            Some(
                result
                    .derivative()
                    .map_or(S::ZERO, |d| d.absolute_error())
                    .into_f64(),
            ),
            line,
        );
        let prime = match p[1] {
            "0" => berp_with_options(x, o),
            "1" => beip_with_options(x, o),
            "2" => kerp_with_options(x, o),
            _ => keip_with_options(x, o),
        }
        .unwrap_or_else(|e| panic!("{line}: {e}"));
        let prime_value = prime.try_value().unwrap_or_else(|e| panic!("{line}: {e}"));
        checker.sample(
            "prime",
            (prime_value - d).abs().into_f64(),
            derivative_scale.into_f64(),
            Some(prime.value().unwrap().absolute_error().into_f64()),
            line,
        );
        // The reported errors cover the rounded references.
        let rounding = |reference: S| S::EPSILON * reference.abs() + S::ZERO.next_after(S::ONE);
        assert!(
            (prime_value - d).abs() <= prime.value().unwrap().absolute_error() + rounding(d),
            "{line}: prime error"
        );
        if x > S::ZERO {
            let partner = match p[1] {
                "0" => "1",
                "1" => "0",
                "2" => "3",
                _ => "2",
            };
            let other = references[&(p[2], partner)].0;
            let signed = if matches!(p[1], "0" | "2") {
                -other
            } else {
                other
            };
            // The differential equation combines independent rounded references,
            // including their rounding amplification by 1/x.
            let expected = signed - d / x;
            let reference_rounding =
                S::try_from_f64_lossy(8.0).unwrap() * S::EPSILON * (other.abs() + (d / x).abs())
                    + S::ZERO.next_after(S::ONE);
            if let Some(second) = prime.derivative() {
                assert!(expected.is_finite(), "{line}: second derivative range");
                // Propagate the fixture's componentwise accuracy contract
                // through the ODE, including its 1/x conditioning near zero.
                checker.sample(
                    "second",
                    (*second.value() - expected).abs().into_f64(),
                    (other_value.abs().into_f64() + derivative_scale.into_f64() / x.into_f64())
                        .min(f64::MAX),
                    Some(second.absolute_error().into_f64()),
                    line,
                );
                assert!(
                    (*second.value() - expected).abs()
                        <= second.absolute_error() + reference_rounding,
                    "{line}: second derivative error"
                );
            } else {
                assert!(
                    !expected.is_finite() || expected == S::ZERO,
                    "{line}: missing second derivative"
                );
            }
        }
        let reported = result.value().unwrap().absolute_error();
        assert!(
            (actual - reference).abs() <= reported + rounding(reference),
            "{line}: value error {reported:?}"
        );
        let reported = result.derivative().map_or(S::ZERO, |d| d.absolute_error());
        assert!(
            (actual_d - d).abs() <= reported + rounding(d),
            "{line}: derivative error {reported:?}"
        );
        assert!(result.counts().iterations() <= o.iteration_budget().get());
        if x > S::ZERO && x.into_f64() <= 100.0 {
            let index = p[1].parse::<usize>().unwrap();
            let combined = kelvin_with_options(x, o).unwrap();
            assert_eq!(combined[index].value(), result.value(), "{line}");
            assert_eq!(combined[index].derivative(), result.derivative(), "{line}");
            let plain = kelvin(x).unwrap();
            assert_eq!(plain[index], actual, "{line}");
        }
    }
}
#[test]
fn f64_oracles() {
    check("64", |s| {
        f64::from_bits(u64::from_str_radix(s, 16).unwrap())
    });
}
#[test]
fn f32_oracles() {
    check("32", |s| {
        f32::from_bits(u32::from_str_radix(s, 16).unwrap())
    });
}
#[test]
fn domains_and_budget() {
    assert_eq!(ber(0.0_f64).unwrap(), 1.0);
    assert_eq!(bei(0.0_f32).unwrap(), 0.0);
    assert_eq!(keip(0.0_f64).unwrap(), 0.0);
    assert!((kei(0.0_f64).unwrap() + core::f64::consts::FRAC_PI_4).abs() < 1e-15);
    assert!(ker(0.0_f64).is_err());
    assert!(kei(-1.0_f64).is_err());
    assert_eq!(ber(-2.0_f64).unwrap(), ber(2.0_f64).unwrap());
    assert_eq!(berp(-2.0_f64).unwrap(), -berp(2.0_f64).unwrap());
    let r = ker_with_options(
        8.0_f64,
        KelvinOptions::default().with_iteration_budget(Budget::ONE),
    )
    .unwrap_err();
    let r_counts = publication::exhausted_counts(&r);
    assert!(r_counts.iterations() <= 1);
}

#[test]
fn combined_and_f32_exhaustion() {
    let all = kelvin(2.0_f64).unwrap();
    assert_eq!(all[0], ber(2.0).unwrap());
    assert_eq!(all[3], kei(2.0).unwrap());
    assert_eq!(all[1], bei(2.0).unwrap());
    assert_eq!(all[2], ker(2.0).unwrap());
    let result = ker_with_options(
        8.0_f32,
        KelvinOptions::default().with_iteration_budget(Budget::ONE),
    )
    .unwrap_err();
    let _ = publication::exhausted_counts(&result);
}

#[test]
fn scaled_derivatives_and_underflow_keep_their_own_domains() {
    assert!(
        keip_with_options(0.0_f64, KelvinOptions::default())
            .unwrap()
            .derivative()
            .is_none()
    );
    let result = bei_with_options(1e-30_f32, KelvinOptions::default()).unwrap();
    assert_eq!(result.try_value().unwrap(), 0.0);
    assert!((*result.derivative().unwrap().value() - 5e-31).abs() < 1e-37);
    {
        macro_rules! check_component {
            ($plain:path, $detailed:path, $scoped:path) => {{
                let result = $detailed(10000.0_f64, KelvinOptions::default()).unwrap();
                assert!(result.try_value().unwrap().is_finite());
                assert!(result.derivative().unwrap().value().is_finite());
            }};
        }
        check_component!(ber_scaled, ber_scaled_with_options, ber_scaled_scoped);
        check_component!(bei_scaled, bei_scaled_with_options, bei_scaled_scoped);
        check_component!(ker_scaled, ker_scaled_with_options, ker_scaled_scoped);
        check_component!(kei_scaled, kei_scaled_with_options, kei_scaled_scoped);
        check_component!(berp_scaled, berp_scaled_with_options, berp_scaled_scoped);
        check_component!(beip_scaled, beip_scaled_with_options, beip_scaled_scoped);
        check_component!(kerp_scaled, kerp_scaled_with_options, kerp_scaled_scoped);
        check_component!(keip_scaled, keip_scaled_with_options, keip_scaled_scoped);
    }
}

#[test]
fn scaled_plain_values_match_detailed_and_scoped_forms() {
    fn check<S: scioxide_core::RealScalar>() {
        {
            macro_rules! check_component {
                ($plain:path, $detailed:path, $scoped:path) => {{
                    for x in [-1.0, 0.0, 1e-4, 1.0, 8.0, 10000.0] {
                        let x = S::try_from_f64_lossy(x).unwrap();
                        let plain = $plain(x);
                        let detailed = $detailed(x, KelvinOptions::default())
                            .and_then(|report| report.try_value());
                        let scoped = $scoped(
                            x,
                            KelvinOptions::default(),
                            scioxide_core::ExecutionContext::sequential(),
                        )
                        .and_then(|report| report.try_value());
                        match (plain, detailed, scoped) {
                            (Ok(value), Ok(report), Ok(scoped)) => {
                                let tolerance = S::EPSILON * (S::ONE + value.abs());
                                assert!((value - report).abs() <= tolerance);
                                assert!((value - scoped).abs() <= tolerance);
                            }
                            (Err(plain), Err(detailed), Err(scoped)) => {
                                assert_eq!(plain.kind(), detailed.kind());
                                assert_eq!(plain.kind(), scoped.kind());
                            }
                            values => panic!("mismatched Kelvin projections: {values:?}"),
                        }
                    }
                    for x in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
                        assert_eq!(
                            $plain(x).unwrap_err().kind(),
                            scioxide_core::SciOxideErrorKind::NonFinite
                        );
                    }
                }};
            }
            check_component!(ber_scaled, ber_scaled_with_options, ber_scaled_scoped);
            check_component!(bei_scaled, bei_scaled_with_options, bei_scaled_scoped);
            check_component!(ker_scaled, ker_scaled_with_options, ker_scaled_scoped);
            check_component!(kei_scaled, kei_scaled_with_options, kei_scaled_scoped);
            check_component!(berp_scaled, berp_scaled_with_options, berp_scaled_scoped);
            check_component!(beip_scaled, beip_scaled_with_options, beip_scaled_scoped);
            check_component!(kerp_scaled, kerp_scaled_with_options, kerp_scaled_scoped);
            check_component!(keip_scaled, keip_scaled_with_options, keip_scaled_scoped);
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn recessive_sample_retains_independent_accuracy() {
    // Issue #1305: destination-rounded mpmath references at the exact input.
    let x = f64::from_bits(0x4023_f4ba_dd29_a6ae);
    let all = kelvin(x).unwrap();
    for (actual, expected) in [
        (all[2], 0.0001364956621160478),
        (all[3], -0.0003105978086861258),
        (kerp(x).unwrap(), -0.0003231037570700931),
        (keip(x).unwrap(), 0.00013829464574578463),
    ] {
        assert!((actual - expected).abs() <= 64.0 * f64::EPSILON * expected.abs());
    }
}

#[test]
fn shared_budget_charges_each_pair_once() {
    for x in [0.1_f64, 2.0, 8.0, 10.0, 10.0_f64.next_up(), 24.0, 100.0] {
        let options = KelvinOptions::default();
        let complete = kelvin_with_options(x, options).unwrap();
        let growing = ber_with_options(x, options).unwrap().counts().iterations();
        let recessive = ker_with_options(x, options).unwrap().counts().iterations();
        assert_eq!(
            complete.map(|r| r.counts().iterations()),
            [growing, 0, recessive, 0]
        );
        for limit in [
            1,
            growing - 1,
            growing,
            growing + 1,
            growing + recessive - 1,
            growing + recessive,
        ] {
            let result = kelvin_with_options(
                x,
                options.with_iteration_budget(Budget::ONE.saturating_add(limit - 1)),
            );
            if limit < growing + recessive {
                assert_eq!(
                    publication::exhausted_counts(&result.unwrap_err()).iterations(),
                    limit
                );
            } else {
                assert_eq!(result.unwrap(), complete);
            }
        }
    }
}

#[test]
fn checked_domains_range_and_tiny_derivatives() {
    use scioxide_core::{DomainReason, SciOxideErrorKind, SciOxideErrorView};
    for (x, reason) in [
        (0.0, DomainReason::Pole),
        (-0.0, DomainReason::Pole),
        (-1.0, DomainReason::OutsideDomain),
    ] {
        assert!(
            matches!(kelvin(x).unwrap_err().view(), SciOxideErrorView::DomainError { reason: actual, .. } if actual == reason)
        );
    }
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(kelvin(x).unwrap_err().kind(), SciOxideErrorKind::NonFinite);
    }
    let options = KelvinOptions::default();
    let large = kelvin_with_options(1100.0_f64, options).unwrap();
    assert_eq!(
        large[0].scalar_representation(),
        ScalarRepresentation::Overflow
    );
    assert!(large[0].scaled().is_some());
    assert!(large[0].derivative().is_none());
    assert_eq!(
        kelvin(1100.0_f64).unwrap_err().kind(),
        SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(large[2].try_value().unwrap(), 0.0);
    // The value rounds to zero; a derivative below the range is absent.
    assert!(large[2].derivative().is_none());
    let tiny = bei_with_options(1e-200_f64, options).unwrap();
    assert_eq!(tiny.try_value().unwrap(), 0.0);
    assert!((*tiny.derivative().unwrap().value() - 5e-201).abs() <= 1e-215);
    for x in [1e-80_f64, 0.1, 8.0, 10.0, 10.0_f64.next_up(), 24.0] {
        assert_eq!(ber(-x).unwrap(), ber(x).unwrap());
        assert_eq!(bei(-x).unwrap(), bei(x).unwrap());
        assert_eq!(berp(-x).unwrap(), -berp(x).unwrap());
        assert_eq!(beip(-x).unwrap(), -beip(x).unwrap());
        for (function, value) in [
            (
                ber_scaled::<f64> as fn(f64) -> Result<f64, scioxide_core::SciOxideError>,
                ber(x).unwrap(),
            ),
            (
                bei_scaled::<f64> as fn(f64) -> Result<f64, scioxide_core::SciOxideError>,
                bei(x).unwrap(),
            ),
            (
                berp_scaled::<f64> as fn(f64) -> Result<f64, scioxide_core::SciOxideError>,
                berp(x).unwrap(),
            ),
            (
                beip_scaled::<f64> as fn(f64) -> Result<f64, scioxide_core::SciOxideError>,
                beip(x).unwrap(),
            ),
        ] {
            let scaled = function(x).unwrap();
            let expected = value * (-x * core::f64::consts::FRAC_1_SQRT_2).exp();
            assert!((scaled - expected).abs() <= 1e-12 * expected.abs() + f64::MIN_POSITIVE);
        }
    }
}

#[test]
fn scoped_accounting_cancellation_and_allocations() {
    use scioxide_core::{ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState};
    let identity = SolveIdentity::new(
        ProblemToken::new(1305),
        scioxide_core::algorithm_id!("Kelvin", "test", 1),
        0,
        0,
    );
    for x in [0.0_f64, 1.0, 10.0, 24.0, 10000.0] {
        {
            macro_rules! check_component {
                ($plain:path, $detailed:path, $scoped:path) => {{
                    let mut state = SolveState::new();
                    let mut budgets = SolveBudgets::unbounded();
                    let mut context = state
                        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
                        .unwrap();
                    let report = $scoped(x, KelvinOptions::default(), &mut context).unwrap();
                    assert_eq!(context.counts().own().iterations(), 0);
                    assert_eq!(
                        context.counts().nested().iterations(),
                        report.counts().iterations()
                    );
                    assert_eq!(report, $detailed(x, KelvinOptions::default()).unwrap());
                    let cancelled = scioxide_core::CancellationToken::new();
                    cancelled.cancel();
                    let error = $scoped(
                        x,
                        KelvinOptions::default(),
                        ExecutionContext::sequential().with_cancellation(&cancelled),
                    )
                    .unwrap_err();
                    assert!(matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
                    ));
                }};
            }
            check_component!(ber_scaled, ber_scaled_with_options, ber_scaled_scoped);
            check_component!(beip_scaled, beip_scaled_with_options, beip_scaled_scoped);
            check_component!(kei_scaled, kei_scaled_with_options, kei_scaled_scoped);
        }
    }
    for x in [0.1_f64, 8.0, 24.0, 100.0] {
        kelvin(x).unwrap();
        let reports = kelvin_with_options(x, KelvinOptions::default()).unwrap();
        assert!(reports.iter().all(|r| r.termination().is_success()));
        let reports = kelvin_with_options(
            x,
            KelvinOptions::default().with_iteration_budget(Budget::ONE),
        )
        .unwrap_err();
        assert_eq!(publication::exhausted_counts(&reports).iterations(), 1);
    }
}
