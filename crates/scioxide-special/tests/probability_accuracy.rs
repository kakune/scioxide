//! Output-lattice and Gaussian-tail regressions against independent references.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_special::*;

fn invoke<S: scioxide_core::RealScalar>(
    name: &str,
    args: &[S],
) -> Result<S, scioxide_core::SciOxideError> {
    match name {
        "beta_inc" => beta_inc(args[0], args[1], args[2]),
        "gamma_p_inv" => gamma_p_inv(args[0], args[1]),
        "gamma_q_inv" => gamma_q_inv(args[0], args[1]),
        "gamma_p_inv_log" => gamma_p_inv_log(args[0], args[1]),
        "gamma_q_inv_log" => gamma_q_inv_log(args[0], args[1]),
        "beta_inc_inv" => beta_inc_inv(args[0], args[1], args[2]),
        "beta_inc_q_inv" => beta_inc_q_inv(args[0], args[1], args[2]),
        "beta_inc_inv_log" => beta_inc_inv_log(args[0], args[1], args[2]),
        "beta_inc_q_inv_log" => beta_inc_q_inv_log(args[0], args[1], args[2]),
        "erfinv" => erfinv(args[0]),
        "erfcinv" => erfcinv(args[0]),
        "normal_quantile" => normal_quantile(args[0]),
        "normal_isf" => normal_isf(args[0]),
        "normal_inv_log_cdf" => normal_inv_log_cdf(args[0]),
        "normal_inv_log_sf" => normal_inv_log_sf(args[0]),
        "normal_cdf" => normal_cdf(args[0]),
        "normal_sf" => normal_sf(args[0]),
        "normal_log_cdf" => normal_log_cdf(args[0]),
        "normal_log_sf" => normal_log_sf(args[0]),
        _ => panic!("unknown reference function {name}"),
    }
}

fn check_fixture(fixture: &str) -> std::collections::BTreeMap<(&str, &str), usize> {
    let mut counts = std::collections::BTreeMap::new();
    let mut misses = 0;
    for line in fixture.lines().filter(|line| !line.starts_with('#')) {
        let fields: Vec<_> = line.split('\t').collect();
        let (scalar, name) = (fields[0], fields[1]);
        *counts.entry((scalar, name)).or_default() += 1;
        let bits: Vec<_> = fields[2]
            .split(',')
            .map(|x| u64::from_str_radix(x, 16).unwrap())
            .collect();
        let expected_bits = u64::from_str_radix(fields[3], 16).unwrap();
        let (actual, expected, quantum) = if scalar == "f64" {
            let args: Vec<_> = bits.into_iter().map(f64::from_bits).collect();
            let expected = f64::from_bits(expected_bits);
            let actual = invoke(name, &args).unwrap_or_else(|error| panic!("{line}: {error:?}"));
            let quantum = (expected.abs().next_up() - expected.abs()).max(f64::from_bits(1));
            (actual, expected, quantum)
        } else {
            let args: Vec<_> = bits.into_iter().map(|x| f32::from_bits(x as u32)).collect();
            let expected = f32::from_bits(expected_bits as u32);
            let actual = invoke(name, &args).unwrap_or_else(|error| panic!("{line}: {error:?}"));
            let quantum = (expected.abs().next_up() - expected.abs()).max(f32::from_bits(1));
            (f64::from(actual), f64::from(expected), f64::from(quantum))
        };
        let target = if scalar == "f32" {
            1.0
        } else if name == "beta_inc" {
            12.0
        } else if name.starts_with("gamma") || name.starts_with("beta") {
            32.0
        } else if name.contains("inv") || name == "normal_quantile" || name == "normal_isf" {
            4.0
        } else {
            8.0
        };
        if expected.is_infinite() {
            assert_eq!(actual, expected, "{line}: endpoint limit");
            continue;
        }
        assert!(actual.is_finite(), "{line}: non-finite result");
        if actual != expected && (actual - expected).abs() > target * quantum {
            if misses < 15 {
                eprintln!(
                    "{line}: {actual:.17e} vs {expected:.17e}, {} ULP",
                    (actual - expected).abs() / quantum
                );
            }
            misses += 1;
        }
    }
    assert_eq!(misses, 0, "oracle target misses");
    counts
}

#[test]
fn conditioned_probability_oracle() {
    let counts = check_fixture(include_str!("data/probability_conditioned.tsv"));
    for scalar in ["f64", "f32"] {
        for name in ["beta_inc", "beta_inc_inv", "gamma_p_inv"] {
            assert!(
                counts[&(scalar, name)] >= 100,
                "missing {scalar} {name} coverage"
            );
        }
    }
}

#[test]
fn normal_tail_oracle() {
    let counts = check_fixture(include_str!("data/probability_normal.tsv"));
    for scalar in ["f64", "f32"] {
        for name in [
            "erfcinv",
            "normal_quantile",
            "normal_isf",
            "normal_cdf",
            "normal_sf",
            "normal_log_cdf",
            "normal_log_sf",
            "normal_inv_log_cdf",
            "normal_inv_log_sf",
        ] {
            assert!(
                counts[&(scalar, name)] >= 2000,
                "missing {scalar} {name} coverage"
            );
        }
    }
}

#[test]
fn half_shape_beta_series_and_inverse_oracle() {
    let counts = check_fixture(include_str!("data/probability_half_shapes.tsv"));
    for scalar in ["f32", "f64"] {
        for name in [
            "beta_inc",
            "beta_inc_inv",
            "beta_inc_q_inv",
            "beta_inc_inv_log",
            "beta_inc_q_inv_log",
        ] {
            assert!(counts[&(scalar, name)] >= 36);
        }
    }
}

#[test]
fn inverse_lattice_oracle() {
    let counts = check_fixture(include_str!("data/probability_inverse.tsv"));
    for name in [
        "gamma_p_inv",
        "gamma_q_inv",
        "gamma_p_inv_log",
        "gamma_q_inv_log",
        "beta_inc_inv",
        "beta_inc_q_inv",
        "beta_inc_inv_log",
        "beta_inc_q_inv_log",
    ] {
        assert!(
            counts[&("f64", name)] >= 10_000,
            "missing seeded {name} coverage"
        );
    }
}

#[test]
fn normal_kernel_boundary_oracle() {
    let counts = check_fixture(include_str!("data/probability_normal_boundaries.tsv"));
    for scalar in ["f64", "f32"] {
        for name in ["normal_cdf", "normal_sf", "normal_log_cdf", "normal_log_sf"] {
            assert!(
                counts[&(scalar, name)] >= 500,
                "missing {scalar} {name} boundaries"
            );
        }
        for name in ["normal_quantile", "normal_isf"] {
            assert!(
                counts[&(scalar, name)] >= 20,
                "missing {scalar} {name} boundaries"
            );
        }
    }
}

#[test]
fn output_endpoints_and_work_limits() {
    assert_eq!(
        gamma_p_inv(1.0_f64, f64::from_bits(1)).unwrap(),
        f64::from_bits(1)
    );
    assert_eq!(gamma_p_inv(0.0001_f64, 0.5).unwrap(), 0.0);
    assert_eq!(gamma_q_inv(f64::from_bits(1), 0.5).unwrap(), 0.0);
    assert_eq!(beta_inc_inv(0.0001_f64, 1.0, 0.5).unwrap(), 0.0);
    assert_eq!(beta_inc_inv(1.0_f64, 0.0001, 0.5).unwrap(), 1.0);
    assert_eq!(normal_sf(f64::MAX).unwrap(), 0.0);
    assert_eq!(normal_cdf(f64::MAX).unwrap(), 1.0);
    // mpmath 1.3.0 log quantiles at 400/480 digits, identical binary64 bits.
    for (lp, bits) in [
        (-1e100_f64, 0xca58_30ec_3c0b_1d60),
        (-f64::MAX, 0xdff6_a09e_667f_3bcc),
    ] {
        let expected = f64::from_bits(bits);
        assert!(
            (normal_inv_log_cdf(lp).unwrap() - expected).abs()
                <= 2.0 * (expected.abs().next_up() - expected.abs())
        );
        assert!(
            (normal_inv_log_sf(lp).unwrap() + expected).abs()
                <= 2.0 * (expected.abs().next_up() - expected.abs())
        );
    }
    let symmetric = beta_inc_inv_log(0.0001_f64, 0.0001, -core::f64::consts::LN_2).unwrap();
    // The supplied binary64 log probability is slightly above real -ln(2),
    // so this ill-conditioned symmetric quantile is not exactly the median.
    assert!(symmetric > 0.5);
    assert_eq!(
        normal_log_sf(-f64::MAX).unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
    let options = GammaOptions::default().with_iteration_budget(scioxide_core::Budget::ONE);
    let result = gamma_q_inv_with_options(2.5_f64, 0.2, options).unwrap();
    assert_eq!(
        result.termination(),
        SpecialFunctionTermination::IterationLimit
    );
    assert_eq!(result.counts().iterations(), 1);
    let result = normal_inv_log_cdf_with_options(
        -1000.0_f64,
        ProbabilityOptions::new(scioxide_core::Budget::ONE),
    )
    .unwrap();
    assert_eq!(
        result.termination(),
        SpecialFunctionTermination::IterationLimit
    );
    assert_eq!(result.counts().iterations(), 1);
}

#[test]
fn inverse_tail_exhaustion_retains_the_candidate_and_total_work() {
    for limit in [1, 2] {
        let budget = scioxide_core::Budget::try_from(limit).unwrap();
        for report in [
            gamma_q_inv_with_options(0.02_f64, 0.2, GammaOptions::new(budget)),
            gamma_q_inv_with_options(8.0_f64, 0.2, GammaOptions::new(budget)),
            beta_inc_q_inv_with_options(0.02_f64, 0.8, 0.2, BetaOptions::new(budget)),
            beta_inc_q_inv_with_options(8.0_f64, 12.0, 0.2, BetaOptions::new(budget)),
        ] {
            let report = report.unwrap();
            assert_eq!(
                report.termination(),
                SpecialFunctionTermination::IterationLimit
            );
            assert_eq!(report.counts().iterations(), limit);
            assert!(report.value().is_some());
        }
    }
}

#[test]
fn inverse_refinement_shares_the_exact_work_budget() {
    for (a, b, p) in [(2.5_f64, 0.5, 0.9), (0.02, 0.8, 0.2), (8.0, 12.0, 0.5)] {
        for beta in [false, true] {
            let evaluate = |budget| {
                if beta {
                    beta_inc_q_inv_with_options(a, b, p, BetaOptions::new(budget))
                } else {
                    gamma_q_inv_with_options(a, p, GammaOptions::new(budget))
                }
            };
            let full = evaluate(GammaOptions::default().iteration_budget()).unwrap();
            let count = full.counts().iterations();
            assert!(count > 2);
            let admitted = evaluate(scioxide_core::Budget::try_from(count).unwrap()).unwrap();
            assert_eq!(
                admitted.termination(),
                SpecialFunctionTermination::Converged
            );
            assert_eq!(
                admitted.value().unwrap().value(),
                full.value().unwrap().value()
            );
            assert_eq!(admitted.counts().iterations(), count);
            let short = evaluate(scioxide_core::Budget::try_from(count - 1).unwrap()).unwrap();
            assert_eq!(
                short.termination(),
                SpecialFunctionTermination::IterationLimit
            );
            assert_eq!(short.counts().iterations(), count - 1);
            assert!(short.value().is_some());
        }
    }
}

#[test]
fn tails_and_inverses_are_monotone_in_both_precisions() {
    for pair in [
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        1e-300,
        1e-200,
        1e-20,
        0.1,
        0.5,
        0.9,
    ]
    .windows(2)
    {
        assert!(normal_quantile(pair[0]).unwrap() < normal_quantile(pair[1]).unwrap());
        assert!(normal_isf(pair[0]).unwrap() > normal_isf(pair[1]).unwrap());
    }
    for pair in [
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        1e-30,
        1e-10,
        0.1,
        0.5,
        0.9,
    ]
    .windows(2)
    {
        assert!(normal_quantile(pair[0]).unwrap() < normal_quantile(pair[1]).unwrap());
        assert!(normal_isf(pair[0]).unwrap() > normal_isf(pair[1]).unwrap());
        assert!(gamma_p_inv(0.02_f32, pair[0]).unwrap() <= gamma_p_inv(0.02_f32, pair[1]).unwrap());
        assert!(
            beta_inc_inv(0.02_f32, 3.0, pair[0]).unwrap()
                <= beta_inc_inv(0.02_f32, 3.0, pair[1]).unwrap()
        );
    }
}

#[test]
fn normal_fixed_work_preserves_published_envelopes() {
    let options = ProbabilityOptions::default().with_iteration_budget(scioxide_core::Budget::ONE);
    for x in [-39.0_f64, -38.0, -1.0, -0.0, 0.0, 1.0, 38.0, 39.0] {
        for result in [
            normal_cdf_with_options(x, options),
            normal_sf_with_options(x, options),
            normal_log_cdf_with_options(x, options),
            normal_log_sf_with_options(x, options),
        ] {
            let result = result.unwrap();
            assert_eq!(result.termination(), SpecialFunctionTermination::Converged);
            assert_eq!(result.counts().iterations(), 1);
            let estimate = result.value().unwrap();
            let magnitude = estimate.value().abs();
            let expected = if magnitude < f64::MIN_POSITIVE {
                68.0 * f64::from_bits(1)
            } else {
                64.0 * (f64::EPSILON * magnitude).max(f64::from_bits(1))
            };
            assert_eq!(estimate.absolute_error(), expected);
            assert_eq!(
                estimate.reliability(),
                scioxide_core::EstimateReliability::StatisticallyCalibrated
            );
        }
    }
    for p in [
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        0.125,
        0.5,
        0.75,
        1.0_f64.next_down(),
    ] {
        for result in [
            normal_quantile_with_options(p, options),
            normal_isf_with_options(p, options),
        ] {
            let result = result.unwrap();
            assert_eq!(result.termination(), SpecialFunctionTermination::Converged);
            assert_eq!(result.counts().iterations(), 1);
        }
    }
    for p in [0.0, 1.0] {
        for result in [
            normal_quantile_with_options(p, options),
            normal_isf_with_options(p, options),
        ] {
            let result = result.unwrap();
            assert_eq!(result.termination(), SpecialFunctionTermination::Limit);
            assert_eq!(result.counts().iterations(), 0);
            assert_eq!(result.value().unwrap().absolute_error(), 0.0);
        }
    }
}

fn normal_extreme_contracts<S: scioxide_core::RealScalar>() {
    for name in [
        "normal_cdf",
        "normal_sf",
        "normal_log_cdf",
        "normal_log_sf",
        "normal_quantile",
        "normal_isf",
        "normal_inv_log_cdf",
        "normal_inv_log_sf",
    ] {
        for x in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
            assert!(
                matches!(
                    invoke(name, &[x]).unwrap_err().view(),
                    scioxide_core::SciOxideErrorView::NonFinite { .. }
                ),
                "{name}"
            );
        }
    }
    for name in ["normal_quantile", "normal_isf"] {
        for p in [-S::MIN_SUBNORMAL, S::ONE.next_after(S::INFINITY)] {
            assert!(
                matches!(
                    invoke(name, &[p]).unwrap_err().view(),
                    scioxide_core::SciOxideErrorView::DomainError { .. }
                ),
                "{name}"
            );
        }
    }
    for name in ["normal_inv_log_cdf", "normal_inv_log_sf"] {
        assert!(
            matches!(
                invoke(name, &[S::MIN_SUBNORMAL]).unwrap_err().view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ),
            "{name}"
        );
    }
    assert_eq!(normal_cdf(-S::MAX).unwrap(), S::ZERO);
    assert_eq!(normal_sf(-S::MAX).unwrap(), S::ONE);
    assert_eq!(normal_cdf(S::MAX).unwrap(), S::ONE);
    assert_eq!(normal_sf(S::MAX).unwrap(), S::ZERO);
    assert!(normal_log_cdf(-S::MAX).is_err());
    assert!(normal_log_sf(S::MAX).is_err());
    assert_eq!(
        normal_log_cdf(S::MAX).unwrap().into_f64().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        normal_log_sf(-S::MAX).unwrap().into_f64().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        normal_quantile(S::HALF).unwrap().into_f64().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        normal_isf(S::HALF).unwrap().into_f64().to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(normal_quantile(S::ZERO).unwrap(), S::NEG_INFINITY);
    assert_eq!(normal_quantile(S::ONE).unwrap(), S::INFINITY);
    assert_eq!(normal_isf(S::ZERO).unwrap(), S::INFINITY);
    assert_eq!(normal_isf(S::ONE).unwrap(), S::NEG_INFINITY);
}

#[test]
fn normal_extremes_and_rejected_inputs_in_both_precisions() {
    normal_extreme_contracts::<f64>();
    normal_extreme_contracts::<f32>();
}
