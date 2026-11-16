//! Finite-null identities, ties and association edge cases.
#![allow(clippy::unwrap_used)]
use scioxide_linalg::Matrix;
use scioxide_stats::*;
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-11, "{a} != {b}");
}

// Independent pairwise definitions deliberately retain quadratic work on small samples.
fn pairwise_placements(x: &[f64], y: &[f64]) -> Option<(f64, f64, f64)> {
    let compare = |a: f64, b: f64| {
        if a > b {
            1.0
        } else if a == b {
            0.5
        } else {
            0.0
        }
    };
    let n = x.len() as f64;
    let m = y.len() as f64;
    let px: Vec<_> = x
        .iter()
        .map(|&a| y.iter().map(|&b| compare(a, b)).sum::<f64>() / m)
        .collect();
    let py: Vec<_> = y
        .iter()
        .map(|&b| x.iter().map(|&a| compare(a, b)).sum::<f64>() / n)
        .collect();
    let p = px.iter().sum::<f64>() / n;
    let vx = px.iter().map(|v| (v - p).powi(2)).sum::<f64>() / (n * (n - 1.0));
    let vy = py.iter().map(|v| (v - p).powi(2)).sum::<f64>() / (m * (m - 1.0));
    let v = vx + vy;
    (v > 0.0).then(|| {
        (
            (p - 0.5) / v.sqrt(),
            p,
            v * v / (vx * vx / (n - 1.0) + vy * vy / (m - 1.0)),
        )
    })
}

fn rank_preparation<S: scioxide_core::RealScalar>(tolerance: f64) {
    use scioxide_random::{Cdf, Normal, StudentT, Survival};
    let convert = |v| S::try_from_f64_lossy(v).unwrap();
    let close =
        |a: f64, b: f64| assert!((a - b).abs() <= tolerance * b.abs().max(1.0), "{a} != {b}");
    for (n, m) in [(2, 3), (5, 2), (9, 9), (17, 31)] {
        for tied in [false, true] {
            let x: Vec<_> = (0..n)
                .map(|i| {
                    if tied {
                        ((i * 2) % 4) as f64
                    } else {
                        ((i * 7) % n) as f64 * 2.0
                    }
                })
                .collect();
            let y: Vec<_> = (0..m)
                .map(|i| {
                    if tied {
                        ((i + 1) % 3) as f64
                    } else {
                        ((i * 11) % m) as f64 * 2.0 + 1.0
                    }
                })
                .collect();
            let xs: Vec<_> = x.iter().copied().map(convert).collect();
            let ys: Vec<_> = y.iter().copied().map(convert).collect();
            let (statistic, effect, df) = pairwise_placements(&x, &y).unwrap();
            for alternative in [
                TestAlternative::Less,
                TestAlternative::Greater,
                TestAlternative::TwoSided,
            ] {
                let options = ClassicalOptions::default().with_alternative(alternative);
                for calibration in [
                    BrunnerMunzelCalibration::Normal,
                    BrunnerMunzelCalibration::Student,
                ] {
                    let result =
                        brunner_munzel_test_with_options(&xs, &ys, calibration, options).unwrap();
                    close(result.statistic().into_f64(), statistic);
                    close(result.effect().unwrap().1.into_f64(), effect);
                    assert_eq!(result.retained(), [n, m]);
                    let (lower, upper) = match calibration {
                        BrunnerMunzelCalibration::Normal => {
                            assert_eq!(result.method(), ClassicalMethod::Normal);
                            let law = Normal::new(0.0, 1.0).unwrap();
                            (law.cdf(statistic).unwrap(), law.sf(statistic).unwrap())
                        }
                        BrunnerMunzelCalibration::Student => {
                            let ClassicalMethod::StudentT { degrees_of_freedom } = result.method()
                            else {
                                unreachable!()
                            };
                            close(degrees_of_freedom, df);
                            let law = StudentT::new(df).unwrap();
                            (law.cdf(statistic).unwrap(), law.sf(statistic).unwrap())
                        }
                    };
                    let p = match alternative {
                        TestAlternative::Less => lower,
                        TestAlternative::Greater => upper,
                        TestAlternative::TwoSided => 2.0 * lower.min(upper),
                    };
                    close(result.p_value(), p);
                    let reversed = brunner_munzel_test_with_options(
                        &ys,
                        &xs,
                        calibration,
                        options.with_alternative(match alternative {
                            TestAlternative::Less => TestAlternative::Greater,
                            TestAlternative::Greater => TestAlternative::Less,
                            TestAlternative::TwoSided => TestAlternative::TwoSided,
                        }),
                    )
                    .unwrap();
                    close(reversed.statistic().into_f64(), -statistic);
                    close(reversed.effect().unwrap().1.into_f64(), 1.0 - effect);
                    close(reversed.p_value(), p);
                }
                if tied {
                    assert!(rank_sum_test_with_options(&xs, &ys, options).is_err());
                } else {
                    let wins = x
                        .iter()
                        .map(|a| y.iter().filter(|&b| a > b).count())
                        .sum::<usize>() as f64;
                    let nn = n as f64;
                    let mm = m as f64;
                    let expected =
                        (wins - nn * mm / 2.0) / (nn * mm * (nn + mm + 1.0) / 12.0).sqrt();
                    let result = rank_sum_test_with_options(&xs, &ys, options).unwrap();
                    close(result.statistic().into_f64(), expected);
                    close(result.effect().unwrap().1.into_f64(), wins / (nn * mm));
                    assert_eq!(result.method(), ClassicalMethod::Normal);
                    // Continuity is deliberately ignored by this entrypoint.
                    let uncorrected = rank_sum_test_with_options(
                        &xs,
                        &ys,
                        options.with_continuity(scioxide_stats::ContinuityCorrection::None),
                    )
                    .unwrap();
                    assert_eq!(result, uncorrected);
                }
            }
        }
    }
}

#[test]
fn independent_rank_preparation_f32() {
    rank_preparation::<f32>(2e-6);
}

#[test]
fn independent_rank_preparation_f64() {
    rank_preparation::<f64>(2e-12);
}

#[test]
fn rank_preparation_rejects_ties_and_undefined_inputs() {
    for (x, y) in [
        (&[1.0, 1.0][..], &[2.0, 3.0][..]),
        (&[1.0, 2.0], &[3.0, 3.0]),
        (&[1.0, 2.0], &[2.0, 3.0]),
        (&[-0.0, 1.0], &[0.0, 2.0]),
    ] {
        assert!(rank_sum_test_with_options(x, y, Default::default()).is_err());
    }
    for calibration in [
        BrunnerMunzelCalibration::Normal,
        BrunnerMunzelCalibration::Student,
    ] {
        for (x, y) in [
            (&[1.0, 1.0][..], &[1.0, 1.0][..]),
            (&[0.0, 0.0], &[1.0, 1.0]),
            (&[-0.0, 0.0], &[0.0, -0.0]),
        ] {
            assert!(
                brunner_munzel_test_with_options(x, y, calibration, Default::default()).is_err()
            );
        }
        for x in [
            &[][..],
            &[1.0][..],
            &[1.0, f64::NAN][..],
            &[1.0, f64::INFINITY][..],
        ] {
            assert!(
                brunner_munzel_test_with_options(x, &[1.0, 2.0], calibration, Default::default())
                    .is_err()
            );
        }
        assert!(
            brunner_munzel_test_with_options(
                &[1.0, 3.0],
                &[2.0, 4.0],
                calibration,
                ClassicalOptions::default().with_calibration(TestCalibration::Exact)
            )
            .is_err()
        );
    }
    for x in [&[][..], &[1.0, f64::NAN][..], &[1.0, f64::NEG_INFINITY][..]] {
        assert!(rank_sum_test_with_options(x, &[2.0, 3.0], Default::default()).is_err());
    }
    assert!(
        rank_sum_test_with_options(
            &[1.0, 3.0],
            &[2.0, 4.0],
            ClassicalOptions::default().with_calibration(TestCalibration::Exact)
        )
        .is_err()
    );
}

fn weighted_extremes<S: scioxide_core::RealScalar>() {
    let x = [S::ZERO, S::ONE];
    for combination in [
        KendallWeightCombination::Additive,
        KendallWeightCombination::Multiplicative,
    ] {
        for weights in [[1e300, 1e-300], [f64::MAX, f64::from_bits(1)]] {
            let concordant = weighted_kendall_tau(&x, &x, Some(&weights), combination).unwrap();
            let discordant =
                weighted_kendall_tau(&x, &[S::ONE, S::ZERO], Some(&weights), combination).unwrap();
            close(concordant.into_f64(), 1.0);
            close(discordant.into_f64(), -1.0);
        }
        assert!(
            weighted_kendall_tau(
                &x,
                &x,
                Some(&[1.0, 0.0]),
                KendallWeightCombination::Multiplicative
            )
            .is_err()
        );
        assert!(
            weighted_kendall_tau(&x, &[S::ONE; 2], Some(&[1e300, 1e-300]), combination).is_err()
        );
    }
    let x = [S::ZERO, S::ONE, S::try_from_f64_lossy(2.0).unwrap()];
    let y = [S::ZERO, S::try_from_f64_lossy(2.0).unwrap(), S::ONE];
    for (mode, expected) in [
        (KendallWeightCombination::Additive, 1.0 / 6.0),
        (KendallWeightCombination::Multiplicative, -1.0 / 11.0),
    ] {
        let fit = weighted_kendall_tau(&x, &y, Some(&[1.0, 2.0, 3.0]), mode).unwrap();
        assert!((fit.into_f64() - expected).abs() < 2e-7);
        let tied = weighted_kendall_tau(
            &[S::ZERO, S::ZERO, S::ONE],
            &[S::ZERO, S::ONE, S::ONE],
            Some(&[1e300, 1e-300, 1e-300]),
            mode,
        )
        .unwrap();
        let expected = core::f64::consts::FRAC_1_SQRT_2;
        assert!((tied.into_f64() - expected).abs() < 2e-7);
    }
    let y = [S::ONE, S::try_from_f64_lossy(2.0).unwrap(), S::ZERO];
    for scale in [1.0, 0.125, 128.0] {
        let weights = [1e300 * scale, 1e-300 * scale, 2e-300 * scale];
        let fit = weighted_kendall_tau(
            &x,
            &y,
            Some(&weights),
            KendallWeightCombination::Multiplicative,
        )
        .unwrap();
        let tolerance = if core::mem::size_of::<S>() == 4 {
            2e-7
        } else {
            2e-14
        };
        assert!((fit.into_f64() + 1.0 / 3.0).abs() < tolerance);
        assert!((-1.0..=1.0).contains(&fit.into_f64()));
    }
}

#[test]
fn weighted_pair_mass_survives_the_full_exponent_range() {
    weighted_extremes::<f32>();
    weighted_extremes::<f64>();
}

#[test]
fn bounded_weights_retry_after_numerator_cancellation() {
    let tiny = 2.0_f64.powi(-64);
    let result = weighted_kendall_tau(
        &[3.0, 0.0, 1.0, 2.0],
        &[2.0, 1.0, 0.0, 1.0],
        Some(&[tiny, 1.0, 1.0, 1.0]),
        KendallWeightCombination::Multiplicative,
    )
    .unwrap();
    let expected = 3.0 * tiny / 6.0_f64.sqrt();
    assert!((result / expected - 1.0).abs() < 2e-14);
}

#[test]
fn additive_pair_components_survive_cancellation_of_large_weights() {
    // Signed pair masses simplify exactly to -2*small; both marginals have
    // mass 4*large+2*small. No rounded pair sum may erase the small component.
    for (large, small) in [(1e16, 1e-16), (1e150, 1e-150)] {
        let result = weighted_kendall_tau(
            &[0.0, 1.0, 2.0],
            &[1.0, 2.0, 0.0],
            Some(&[large, large, small]),
            KendallWeightCombination::Additive,
        )
        .unwrap();
        let expected = -small / (2.0 * large);
        assert!((result / expected - 1.0).abs() < 3e-14);
    }
    let result = weighted_kendall_tau(
        &[0.0_f32, 1.0, 2.0],
        &[1.0, 2.0, 0.0],
        Some(&[1e16, 1e16, 1e-16]),
        KendallWeightCombination::Additive,
    )
    .unwrap();
    assert!((result as f64 / -5e-33 - 1.0).abs() < 2e-7);
}
#[test]
fn finite_nulls() {
    let x = [1., 2., 3.];
    let y = [4., 5., 6.];
    let u = mann_whitney_test_with_options(&x, &y, Default::default()).unwrap();
    close(u.statistic(), 0.);
    close(u.p_value(), 0.1);
    let signed =
        wilcoxon_test_with_options(&x, SignedRankZeros::Discard, Default::default()).unwrap();
    close(signed.statistic(), 0.);
    close(signed.p_value(), 0.25);
    for zeros in [
        SignedRankZeros::Discard,
        SignedRankZeros::Pratt,
        SignedRankZeros::Split,
    ] {
        let a = wilcoxon_test_with_options(&[0., 0., 0.], zeros, Default::default()).unwrap();
        close(a.p_value(), 1.);
    }
    let a =
        wilcoxon_test_with_options(&[1., 1., -1.], SignedRankZeros::Discard, Default::default())
            .unwrap();
    close(a.p_value(), 1.);
    let r = spearman_test_with_options(&x, &y, Default::default()).unwrap();
    close(r.statistic(), 1.);
    close(r.p_value(), 1. / 3.);
    let k = kendall_test_with_options(&x, &y, KendallVariant::TauB, Default::default()).unwrap();
    close(k.statistic(), 1.);
    close(k.p_value(), 1. / 3.);
    let options = ClassicalOptions::default().with_alternative(TestAlternative::Greater);
    let data = Matrix::from_vec(2, 3, vec![1., 2., 3., 1., 2., 3.]).unwrap();
    let p = page_trend_test_with_options(data.view(), &[0, 1, 2], options).unwrap();
    close(p.statistic(), 28.);
    close(p.p_value(), 1. / 36.);
}
#[test]
fn symmetric_and_tied_cases() {
    let x = [1., 1., 2., 4.];
    let y = [1., 2., 2., 3.];
    let a = mann_whitney_test_with_options(&x, &y, Default::default()).unwrap();
    let b = mann_whitney_test_with_options(&y, &x, Default::default()).unwrap();
    close(a.statistic() + b.statistic(), 16.);
    close(a.p_value(), b.p_value());
    let b = bws_test_with_options(&x, &y, Default::default()).unwrap();
    assert!((0.0..=1.0).contains(&b.p_value()));
    let a = weighted_kendall_tau(&x, &y, None, KendallWeightCombination::Additive).unwrap();
    let b = weighted_kendall_tau(&y, &x, None, KendallWeightCombination::Additive).unwrap();
    close(a, b);

    let a = weighted_kendall_tau(
        &x,
        &y,
        Some(&[1., 1., 1., 1.]),
        KendallWeightCombination::Additive,
    )
    .unwrap();
    let b = kendall_test_with_options(&x, &y, KendallVariant::TauB, Default::default()).unwrap();
    close(a, b.statistic());
    assert!(rank_sum_test_with_options(&x, &y, Default::default()).is_err());
}
#[test]
fn methods_and_errors() {
    let options = ClassicalOptions::default()
        .with_calibration(TestCalibration::Exact)
        .with_max_work(1);
    assert!(mann_whitney_test_with_options(&[1., 2.], &[3., 4.], options).is_err());
    assert!(wilcoxon_test_with_options(&[1., 2.], SignedRankZeros::Discard, options).is_err());
    assert!(spearman_test_with_options(&[1., 2.], &[3., 4.], options).is_err());
    let a = mann_whitney_test_with_options(
        &[1., 2.],
        &[3., 4.],
        ClassicalOptions::default().with_max_work(1),
    )
    .unwrap();
    assert_eq!(a.method(), ClassicalMethod::Normal);
    assert!(
        weighted_kendall_tau(
            &[1., 2.],
            &[3., 4.],
            Some(&[0., 0.]),
            KendallWeightCombination::Additive
        )
        .is_err()
    );
}
#[test]
fn mgc_geometry() {
    let x = Matrix::from_vec(5, 1, vec![0., 1., 2., 3., 4.]).unwrap();
    let plan = MgcPlan::from_samples(x.view(), x.view(), ClassicalOptions::default()).unwrap();
    let mut scratch = plan.workspace().unwrap();
    close(plan.statistic(None, &mut scratch).unwrap().correlation, 1.);
    let tight = MgcPlan::from_samples(
        x.view(),
        x.view(),
        ClassicalOptions::default().with_max_workspace_bytes(1),
    )
    .unwrap();
    assert!(matches!(
        tight.workspace().unwrap_err().view(),
        scioxide_core::SciOxideErrorView::ResourceLimit { required, limit: 1, .. } if required > 1
    ));
    assert!(
        plan.statistic(Some(&[0, 1, 2, 3, 3]), &mut scratch)
            .is_err()
    );
    let zero = Matrix::<f64>::zeros(5, 1).unwrap();
    let plan = MgcPlan::from_samples(x.view(), zero.view(), ClassicalOptions::default()).unwrap();
    let mut scratch = plan.workspace().unwrap();
    close(plan.statistic(None, &mut scratch).unwrap().correlation, 0.);
}

#[test]
fn kendall_rounding_preserves_coefficient_bounds() {
    for n in 3..20 {
        let x: Vec<_> = (0..n).map(|i| i as f64).collect();
        let y: Vec<_> = x.iter().map(|&v| -v).collect();
        let options = ClassicalOptions::default().with_calibration(TestCalibration::Asymptotic);
        for variant in [KendallVariant::TauB, KendallVariant::TauC] {
            for values in [&x, &y] {
                let result = kendall_test_with_options(&x, values, variant, options).unwrap();
                assert!((-1.0..=1.0).contains(&result.statistic()));
            }
        }
    }
}

#[test]
fn signed_zero_ties_are_shared_by_every_rank_family() {
    // Each zero sign is mixed across both samples; canonical copies use +0.
    let x = [0.0_f64, -0.0, 1.0, 2.5, -0.0, 4.0];
    let y = [-0.0_f64, 0.5, 0.0, 3.0, -1.0, 0.0];
    let canonical = |values: &[f64]| values.iter().map(|&v| v + 0.0).collect::<Vec<_>>();
    let (cx, cy) = (canonical(&x), canonical(&y));
    let pooled: Vec<_> = x.iter().chain(&y).copied().collect();
    for method in [RankMethod::Average, RankMethod::Dense, RankMethod::Ordinal] {
        let signed = rank_slice(&pooled, method).unwrap();
        let positive = rank_slice(&canonical(&pooled), method).unwrap();
        assert_eq!(signed, positive, "{method:?}");
    }
    let average = rank_slice(&pooled, RankMethod::Average).unwrap();
    // Six zeros occupy ranks 2..=7 after the single -1.
    for (rank, &value) in average.iter().zip(&pooled) {
        if value == 0.0 {
            assert_eq!(rank.bounds(), (2, 7));
        }
    }
    let options = ClassicalOptions::default();
    let signed = mann_whitney_test_with_options(&x, &y, options).unwrap();
    let positive = mann_whitney_test_with_options(&cx, &cy, options).unwrap();
    assert_eq!(signed.statistic(), positive.statistic());
    assert_eq!(signed.p_value(), positive.p_value());
    for test in [
        ansari_bradley_test_with_options::<f64>,
        mood_scale_test_with_options::<f64>,
    ] {
        let signed = test(&x, &y, options).unwrap();
        let positive = test(&cx, &cy, options).unwrap();
        assert_eq!(signed.statistic(), positive.statistic());
        assert_eq!(signed.p_value(), positive.p_value());
    }
    let signed = kruskal_wallis_test_with_options(&[&x, &y], StatisticsOptions::new()).unwrap();
    let positive = kruskal_wallis_test_with_options(&[&cx, &cy], StatisticsOptions::new()).unwrap();
    assert_eq!(signed.statistic(), positive.statistic());
    assert_eq!(signed.p_value(), positive.p_value());
}
