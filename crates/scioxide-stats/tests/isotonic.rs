#![allow(clippy::unwrap_used, clippy::expect_used)]
use scioxide_core::RealScalar;
use scioxide_stats::*;

// Exhaust all contiguous equality sets of the convex program. Candidate means,
// order and objectives use exact bounded rational arithmetic, not PAVA.
#[derive(Clone, Copy)]
struct Rational(i128, i128);
impl Rational {
    fn add(self, b: Self) -> Self {
        Self(self.0 * b.1 + b.0 * self.1, self.1 * b.1)
    }
    fn less(self, b: Self) -> bool {
        self.0 * b.1 < b.0 * self.1
    }
    fn number(self) -> f64 {
        self.0 as f64 / self.1 as f64
    }
}
fn optimum(y: &[i128], w: &[i128], decreasing: bool) -> Vec<f64> {
    let mut best: Option<(Rational, Vec<Rational>)> = None;
    for cuts in 0..1usize << (y.len() - 1) {
        let mut candidate = Vec::new();
        let mut start = 0;
        for end in 1..=y.len() {
            if end == y.len() || cuts & (1 << (end - 1)) != 0 {
                let mass: i128 = w[start..end].iter().sum();
                let sum: i128 = y[start..end]
                    .iter()
                    .zip(&w[start..end])
                    .map(|(y, w)| y * w)
                    .sum();
                candidate.extend(core::iter::repeat_n(Rational(sum, mass), end - start));
                start = end;
            }
        }
        if candidate.windows(2).any(|p| {
            if decreasing {
                p[0].less(p[1])
            } else {
                p[1].less(p[0])
            }
        }) {
            continue;
        }
        // Sum by distinct blocks to keep denominators small (n <= 5).
        let cost = y
            .iter()
            .zip(w)
            .zip(&candidate)
            .fold(Rational(0, 1), |cost, ((y, w), x)| {
                let residual = y * x.1 - x.0;
                cost.add(Rational(w * residual * residual, x.1 * x.1))
            });
        if best.as_ref().is_none_or(|(b, _)| cost.less(*b)) {
            best = Some((cost, candidate));
        }
    }
    best.unwrap().1.into_iter().map(Rational::number).collect()
}
fn numerical<S: RealScalar>() {
    let cast = |x| S::try_from_f64_lossy(x).unwrap();
    let eps = S::EPSILON.try_to_f64_exact().unwrap();
    for code in 0..243 {
        let mut digits = code;
        let y: Vec<i128> = (0..5)
            .map(|_| {
                let x = digits % 3 - 1;
                digits /= 3;
                x
            })
            .collect();
        let w = [1, 2, 1, 3, 2];
        let values: Vec<_> = y.iter().map(|&x| cast(x as f64)).collect();
        let weights: Vec<_> = w.iter().map(|&x| cast(x as f64)).collect();
        for decreasing in [false, true] {
            let direction = if decreasing {
                IsotonicDirection::Decreasing
            } else {
                IsotonicDirection::Increasing
            };
            let fit = isotonic_regression_with_options(
                &values,
                Weights::<_, Mass>::new(&weights),
                (direction, StatisticsOptions::new()).0,
                (direction, StatisticsOptions::new()).1,
            )
            .unwrap();
            let expected = optimum(&y, &w, decreasing);
            let x: Vec<_> = fit
                .fitted()
                .iter()
                .map(|x| x.try_to_f64_exact().unwrap())
                .collect();
            for (a, b) in x.iter().zip(expected) {
                assert!((a - b).abs() <= 16.0 * eps);
            }
            let mut multiplier: f64 = 0.0;
            for i in 0..5 {
                multiplier +=
                    (y[i] as f64 - x[i]) * w[i] as f64 * if decreasing { -1.0 } else { 1.0 };
                assert!(multiplier >= -128.0 * eps);
                if i < 4 {
                    assert!(if decreasing {
                        x[i] >= x[i + 1]
                    } else {
                        x[i] <= x[i + 1]
                    });
                    assert!((multiplier * (x[i + 1] - x[i])).abs() <= 128.0 * eps);
                }
            }
            assert!(multiplier.abs() <= 128.0 * eps);
            let loss: f64 = y
                .iter()
                .zip(&x)
                .zip(w)
                .map(|((&y, &x), w)| w as f64 * (y as f64 - x).powi(2))
                .sum();
            assert!(
                (fit.diagnostics()
                    .squared_error
                    .unwrap()
                    .try_to_f64_exact()
                    .unwrap()
                    - loss)
                    .abs()
                    <= 128.0 * eps
            );
            assert_eq!(fit.blocks().first().unwrap().start, 0);
            assert_eq!(fit.blocks().last().unwrap().end, 5);
            for pair in fit.blocks().windows(2) {
                assert_eq!(pair[0].end, pair[1].start);
                assert_ne!(pair[0].value, pair[1].value);
            }
        }
    }
}
#[test]
fn convex_optima_and_kkt_f32() {
    numerical::<f32>();
}
#[test]
fn convex_optima_and_kkt_f64() {
    numerical::<f64>();
}

#[test]
fn replication_scale_and_zero_extension() {
    let y = [3.0, 1.0, 4.0, 2.0];
    let w = [1.0, 3.0, 2.0, 2.0];
    let fit = isotonic_regression_with_options(
        &y,
        Weights::<_, Mass>::new(&w),
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    )
    .unwrap();
    let replicated = isotonic_regression_with_options(
        &[3.0, 1.0, 1.0, 1.0, 4.0, 4.0, 2.0, 2.0],
        Unweighted,
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert_eq!(fit.fitted(), vec![1.5, 1.5, 3.0, 3.0]);
    assert_eq!(
        replicated.fitted(),
        vec![1.5, 1.5, 1.5, 1.5, 3.0, 3.0, 3.0, 3.0]
    );
    for factor in [f64::MIN_POSITIVE, 1e300] {
        assert_eq!(
            isotonic_regression_with_options(
                &y,
                Weights::<_, Mass>::new(&w.map(|w| w * factor)),
                IsotonicDirection::Increasing,
                StatisticsOptions::new()
            )
            .unwrap()
            .fitted(),
            fit.fitted()
        );
    }
    let gaps = isotonic_regression_with_options(
        &[f64::NAN, 3.0, 99.0, 1.0, 0.0, 5.0, f64::INFINITY],
        Weights::<_, Mass>::new(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert_eq!(gaps.fitted(), vec![2.0, 2.0, 2.0, 2.0, 2.0, 5.0, 5.0]);
    assert_eq!(gaps.diagnostics().excluded, 4);
    assert_eq!(gaps.blocks()[0].end, 5);
}
fn extremes<S: RealScalar>() {
    let max = S::MAX;
    let tiny = S::MIN_POSITIVE * S::EPSILON;
    let fit = isotonic_regression_with_options(
        &[max, -max],
        Weights::<_, Mass>::new(&[max, max]),
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert_eq!(fit.fitted(), vec![S::ZERO, S::ZERO]);
    assert_eq!(fit.diagnostics().weight, None);
    assert_eq!(fit.diagnostics().squared_error, None);
    let same = isotonic_regression_with_options(
        &[max, max],
        Weights::<_, Mass>::new(&[max, max]),
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert_eq!(same.fitted(), vec![max, max]);
    assert_eq!(same.blocks().len(), 1);
    let subnormal = isotonic_regression_with_options(
        &[tiny, tiny],
        Weights::<_, Mass>::new(&[tiny, tiny]),
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert_eq!(subnormal.fitted(), vec![tiny, tiny]);
    let concentrated = isotonic_regression_with_options(
        &[S::ONE, S::ZERO],
        Weights::<_, Mass>::new(&[tiny, max]),
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert!(concentrated.fitted().iter().all(|x| x.is_finite()));
    let zeros = isotonic_regression_with_options(
        &[-S::ZERO, S::ZERO],
        Unweighted,
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert_eq!(zeros.blocks().len(), 1);
    assert!(zeros.fitted().iter().all(|x| !x.is_sign_negative()));
}
#[test]
fn extremes_f32() {
    extremes::<f32>();
}
#[test]
fn extremes_f64() {
    extremes::<f64>();
}
#[test]
fn cancellation_in_accumulated_block() {
    let fit = isotonic_regression_with_options(
        &[f64::MAX, 1.0, -f64::MAX],
        Unweighted,
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert_eq!(fit.fitted(), vec![1.0 / 3.0; 3]);
}

#[test]
fn native_cancellation_and_complete_pool_promotion() {
    let large = 2.0_f64.powi(64);
    let fit = isotonic_regression_with_options(
        &[large, 1.0, -large],
        Unweighted,
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert_eq!(fit.fitted(), vec![1.0 / 3.0; 3]);
    // The first pool already has a low correction when a wider observation
    // promotes it. Later cancellation must recover that correction.
    let wider = 2.0_f64.powi(66);
    let fit = isotonic_regression_with_options(
        &[large, 1.0, -wider, wider, -large],
        Unweighted,
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    )
    .unwrap();
    let expected = [-large, -large, -large, 1.5 * large, 1.5 * large];
    for (&actual, expected) in fit.fitted().iter().zip(expected) {
        assert!((actual - expected).abs() <= 2.0 * f64::EPSILON * expected.abs());
    }
    let fit = isotonic_regression_with_options(
        &[large, 1.0, -large, 0.0],
        Weights::<_, Mass>::new(&[1.0, 1.0, 1.0, 2.0_f64.powi(65)]),
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    )
    .unwrap();
    let expected = 1.0 / (2.0_f64.powi(65) + 3.0);
    assert!(
        fit.fitted()
            .iter()
            .all(|&v| (v / expected - 1.0).abs() < 8.0 * f64::EPSILON)
    );
}

fn guarded_frames<S: RealScalar>() {
    let cast = |v| S::try_from_f64_lossy(v).unwrap();
    let y = [3.0, -2.0, 4.0, -5.0, 1.0, 2.0];
    let w = [1.0, 2.0, 3.0, 1.0, 4.0, 2.0];
    for direction in [IsotonicDirection::Increasing, IsotonicDirection::Decreasing] {
        let options = (direction, StatisticsOptions::new());
        let base = isotonic_regression_with_options(
            &y.map(cast),
            Weights::<_, Mass>::new(&w.map(cast)),
            (options).0,
            (options).1,
        )
        .unwrap();
        for exponent in [-65, -64, 64, 65] {
            let factor = 2.0_f64.powi(exponent);
            for weighted in [false, true] {
                let values = y.map(|v| cast(if weighted { v } else { v * factor }));
                let weights = w.map(|v| cast(if weighted { v * factor } else { v }));
                let result = isotonic_regression_with_options(
                    &values,
                    Weights::<_, Mass>::new(&weights),
                    (options).0,
                    (options).1,
                )
                .unwrap();
                assert_eq!(result.blocks().len(), base.blocks().len());
                for (actual, expected) in result.fitted().iter().zip(base.fitted()) {
                    let actual = actual.into_f64() / if weighted { 1.0 } else { factor };
                    assert!((actual - expected.into_f64()).abs() <= 16.0 * S::EPSILON.into_f64());
                }
            }
        }
    }
}

#[test]
fn ordinary_and_extended_frames_f32() {
    guarded_frames::<f32>();
}
#[test]
fn ordinary_and_extended_frames_f64() {
    guarded_frames::<f64>();
}

#[test]
fn rounded_blocks_and_long_order_patterns() {
    let previous = f32::from_bits(1.0_f32.to_bits() - 1);
    let fit = isotonic_regression_with_options(
        &[1.0_f32, previous, 1.0],
        Unweighted,
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
    )
    .unwrap();
    assert_eq!(fit.fitted(), vec![1.0; 3]);
    assert_eq!(fit.blocks().len(), 1);
    assert_eq!(fit.blocks()[0].weight, Some(3.0));
    for direction in [IsotonicDirection::Increasing, IsotonicDirection::Decreasing] {
        let sign = if direction == IsotonicDirection::Increasing {
            1.0
        } else {
            -1.0
        };
        let options = (direction, StatisticsOptions::new());
        let monotone: Vec<_> = (0..4096).map(|i| sign * i as f64).collect();
        let fit = isotonic_regression_with_options(&monotone, Unweighted, (options).0, (options).1)
            .unwrap();
        assert_eq!(fit.fitted(), monotone);
        assert_eq!(fit.blocks().len(), monotone.len());
        let alternating: Vec<_> = (0..4096)
            .map(|i| sign * if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let fit =
            isotonic_regression_with_options(&alternating, Unweighted, (options).0, (options).1)
                .unwrap();
        assert_eq!(fit.fitted(), vec![0.0; alternating.len()]);
        assert_eq!(fit.blocks().len(), 1);
        assert_eq!(fit.diagnostics().squared_error, Some(4096.0));
    }
}
#[test]
fn errors_atomicity_empty_and_reuse() {
    let mut workspace = IsotonicWorkspace::default();
    let mut output = [77.0; 2];
    let mut blocks = [IsotonicBlock::default(); 2];
    assert!(
        isotonic_regression_into(
            &[2.0, 1.0],
            Unweighted,
            IsotonicDirection::Increasing,
            StatisticsOptions::new(),
            &mut output,
            &mut blocks,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    workspace
        .try_reserve(&IsotonicWorkspace::scalar_requirements(2, StatisticsOptions::new()).unwrap())
        .unwrap();
    let sentinel = blocks;
    for (y, w) in [
        ([2.0, f64::NAN], [1.0, 1.0]),
        ([2.0, 1.0], [0.0, 0.0]),
        ([2.0, 1.0], [1.0, -1.0]),
        ([2.0, 1.0], [1.0, f64::INFINITY]),
    ] {
        assert!(
            isotonic_regression_into(
                &y,
                Weights::<_, Mass>::new(&w),
                IsotonicDirection::Increasing,
                StatisticsOptions::new(),
                &mut output,
                &mut blocks,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(output, [77.0; 2]);
        assert_eq!(blocks, sentinel);
    }
    assert!(
        isotonic_regression_into(
            &[1.0],
            Unweighted,
            IsotonicDirection::Increasing,
            StatisticsOptions::new(),
            &mut output,
            &mut blocks,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert!(
        isotonic_regression_into(
            &[1.0, 2.0],
            Weights::<_, Mass>::new(&[1.0]),
            IsotonicDirection::Increasing,
            StatisticsOptions::new(),
            &mut output,
            &mut blocks,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert!(
        isotonic_regression_into(
            &[1.0, 2.0],
            Unweighted,
            IsotonicDirection::Increasing,
            StatisticsOptions::new(),
            &mut output,
            &mut [],
            &mut workspace,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    let d = isotonic_regression_into(
        &[2.0, 1.0],
        Unweighted,
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
        &mut output,
        &mut blocks,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output, [1.5; 2]);
    assert_eq!(d.block_count, 1);
    assert_eq!(
        isotonic_regression_with_options(
            &[] as &[f64],
            Unweighted,
            IsotonicDirection::Increasing,
            StatisticsOptions::new()
        )
        .unwrap()
        .diagnostics(),
        IsotonicDiagnostics::default()
    );
    assert_eq!(
        isotonic_regression_with_options(
            &[4.0],
            Unweighted,
            IsotonicDirection::Increasing,
            StatisticsOptions::new()
        )
        .unwrap()
        .fitted(),
        vec![4.0]
    );
    assert!(
        matches ! (& (isotonic_regression_with_options(& [f64 :: NAN], Unweighted, IsotonicDirection::Increasing, StatisticsOptions::new())) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
    );
    let skip = (
        IsotonicDirection::Increasing,
        StatisticsOptions::new().with_non_finite_policy(NonFinitePolicy::Skip),
    );
    assert_eq!(
        isotonic_regression_with_options(
            &[f64::NAN, 2.0, f64::INFINITY, 1.0],
            Unweighted,
            (skip).0,
            (skip).1
        )
        .unwrap()
        .fitted(),
        vec![1.5; 4]
    );
}
