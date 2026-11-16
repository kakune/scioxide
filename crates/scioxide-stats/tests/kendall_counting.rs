#![allow(clippy::unwrap_used)]
use scioxide_core::RealScalar;
use scioxide_stats::*;

fn close<S: RealScalar>(actual: S, expected: f64) {
    assert!(
        (actual.into_f64() - expected).abs() <= 32.0 * S::EPSILON.into_f64(),
        "{} != {expected}",
        actual.into_f64()
    );
}

// Deliberately enumerate individual pairs, independently of sorted counting.
fn reference(x: &[f64], y: &[f64], weights: &[f64], additive: bool) -> [f64; 3] {
    let mut sums = [0.0; 3];
    for i in 0..x.len() {
        for j in 0..i {
            let w = if additive {
                weights[i] + weights[j]
            } else {
                weights[i] * weights[j]
            };
            if x[i] != x[j] {
                sums[1] += w;
            }
            if y[i] != y[j] {
                sums[2] += w;
            }
            if x[i] != x[j] && y[i] != y[j] {
                sums[0] += if (x[i] > x[j]) == (y[i] > y[j]) {
                    w
                } else {
                    -w
                };
            }
        }
    }
    sums
}

fn exhaustive<S: RealScalar>() {
    let values = |mut code: usize| {
        core::array::from_fn::<_, 4, _>(|i| {
            let digit = code % 3;
            code /= 3;
            if digit == 0 && i % 2 == 0 {
                -0.0
            } else {
                digit as f64
            }
        })
    };
    let options = ClassicalOptions::default().with_calibration(TestCalibration::Asymptotic);
    for a in 0..81 {
        for b in 0..81 {
            let (x, y) = (values(a), values(b));
            let xs = x.map(|v| S::try_from_f64_lossy(v).unwrap());
            let ys = y.map(|v| S::try_from_f64_lossy(v).unwrap());
            let [score, vx, vy] = reference(&x, &y, &[1.0; 4], false);
            for variant in [KendallVariant::TauB, KendallVariant::TauC] {
                let result = kendall_test_with_options(&xs, &ys, variant, options);
                if vx == 0.0 || vy == 0.0 {
                    assert!(result.is_err());
                    continue;
                }
                let distinct =
                    |v: &[f64]| (0..v.len()).filter(|&i| !v[..i].contains(&v[i])).count();
                let m = distinct(&x).min(distinct(&y)) as f64;
                let expected = match variant {
                    KendallVariant::TauB => score / (vx * vy).sqrt(),
                    KendallVariant::TauC => 2.0 * score / (16.0 * (m - 1.0) / m),
                };
                close(result.unwrap().statistic(), expected);
            }
            for mode in [
                KendallWeightCombination::Additive,
                KendallWeightCombination::Multiplicative,
            ] {
                let weights = [0.0, 1.0, 2.0, 4.0];
                let [score, vx, vy] =
                    reference(&x, &y, &weights, mode == KendallWeightCombination::Additive);
                let result = weighted_kendall_tau(&xs, &ys, Some(&weights), mode);
                if vx == 0.0 || vy == 0.0 {
                    assert!(result.is_err());
                    continue;
                }
                let result = result.unwrap();
                close(result, score / (vx * vy).sqrt());
            }
        }
    }
}

#[test]
fn exact_pair_enumeration_f32() {
    exhaustive::<f32>();
}
#[test]
fn exact_pair_enumeration_f64() {
    exhaustive::<f64>();
}

#[test]
fn default_importance_retains_joint_tie_index_order() {
    let x = [-0.0, 1.0, 0.0, 2.0, 1.0, 1.0];
    let y = [1.0, 2.0, 1.0, 0.0, 2.0, -0.0];
    for mode in [
        KendallWeightCombination::Additive,
        KendallWeightCombination::Multiplicative,
    ] {
        let mut expected = 0.0;
        for (a, b) in [(&x, &y), (&y, &x)] {
            let mut order: Vec<_> = (0..x.len()).collect();
            order.sort_by(|&i, &j| {
                a[j].partial_cmp(&a[i])
                    .unwrap()
                    .then(b[j].partial_cmp(&b[i]).unwrap())
                    .then(i.cmp(&j))
            });
            let mut weights = [0.0; 6];
            for (rank, &i) in order.iter().enumerate() {
                weights[i] = 1.0 / (rank + 1) as f64;
            }
            let [score, vx, vy] =
                reference(&x, &y, &weights, mode == KendallWeightCombination::Additive);
            expected += score / (vx * vy).sqrt() / 2.0;
        }
        close(weighted_kendall_tau(&x, &y, None, mode).unwrap(), expected);
    }
}

#[test]
fn multiplicative_tree_corrections_survive_cancellation() {
    for (large, small) in [(1e16, 1e-16), (1e150, 1e-150)] {
        let result = weighted_kendall_tau(
            &[0.0, 1.0, 2.0],
            &[1.0, 2.0, 0.0],
            Some(&[large, small, small]),
            KendallWeightCombination::Multiplicative,
        )
        .unwrap();
        let expected = -small / (2.0 * large);
        assert!((result / expected - 1.0).abs() < 3e-14);
    }
}

#[test]
fn exact_directional_inference_includes_tied_pairings() {
    let x = [-0.0, 0.0, 1.0, 2.0];
    let y = [0.0, 2.0, 1.0, 1.0];
    let observed = reference(&x, &y, &[1.0; 4], false)[0];
    let mut null = Vec::new();
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let indices = [a, b, c, d];
                    if indices
                        .iter()
                        .enumerate()
                        .any(|(i, j)| indices[..i].contains(j))
                    {
                        continue;
                    }
                    null.push(reference(&x, &indices.map(|i| y[i]), &[1.0; 4], false)[0]);
                }
            }
        }
    }
    for alternative in [
        TestAlternative::Less,
        TestAlternative::Greater,
        TestAlternative::TwoSided,
    ] {
        let count = null
            .iter()
            .filter(|&&s| match alternative {
                TestAlternative::Less => s <= observed,
                TestAlternative::Greater => s >= observed,
                TestAlternative::TwoSided => s.abs() >= observed.abs(),
            })
            .count();
        for variant in [KendallVariant::TauB, KendallVariant::TauC] {
            let result = kendall_test_with_options(
                &x,
                &y,
                variant,
                ClassicalOptions::default()
                    .with_alternative(alternative)
                    .with_calibration(TestCalibration::Exact),
            )
            .unwrap();
            close(result.p_value(), count as f64 / 24.0);
        }
    }
}

#[test]
fn large_pair_counts_and_directional_normal_tails() {
    let x: Vec<_> = (0..100_000).map(|i| i as f64).collect();
    let y: Vec<_> = x.iter().map(|v| -v).collect();
    for alternative in [TestAlternative::Less, TestAlternative::Greater] {
        let result = kendall_test_with_options(
            &x,
            &y,
            KendallVariant::TauB,
            ClassicalOptions::default()
                .with_alternative(alternative)
                .with_calibration(TestCalibration::Asymptotic),
        )
        .unwrap();
        close(result.statistic(), -1.0);
        close(
            result.p_value(),
            if alternative == TestAlternative::Less {
                0.0
            } else {
                1.0
            },
        );
    }
}
