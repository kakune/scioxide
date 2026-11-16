#![allow(clippy::unwrap_used)]
use scioxide_core::{ExecutionContext, RealScalar, StridedLayout};
use scioxide_stats::*;

fn decreasing_runs<S: RealScalar>() {
    let cast = |x| S::try_from_f64_lossy(x).unwrap();
    let eps = S::EPSILON.into_f64();
    for n in [255, 256, 257, 511, 512, 513, 4096] {
        for direction in [IsotonicDirection::Increasing, IsotonicDirection::Decreasing] {
            let sign = if direction == IsotonicDirection::Increasing {
                1.0
            } else {
                -1.0
            };
            let values: Vec<_> = (0..n).map(|i| cast(sign * (n - i) as f64)).collect();
            let weights: Vec<_> = (0..n).map(|i| cast((1 + i % 7) as f64)).collect();
            for weighted in [false, true] {
                let result = if weighted {
                    isotonic_regression_with_options(
                        &values,
                        Weights::<_, Mass>::new(&weights),
                        direction,
                        StatisticsOptions::new(),
                    )
                } else {
                    isotonic_regression_with_options(
                        &values,
                        Unweighted,
                        direction,
                        StatisticsOptions::new(),
                    )
                }
                .unwrap();
                // All observations must pool. Integer moments give an independent
                // weighted optimum and loss at the published, rounded scalar.
                let (mass, sum, squares) = (0..n).fold((0_i128, 0_i128, 0_i128), |(m, s, q), i| {
                    let w = if weighted { 1 + i % 7 } else { 1 } as i128;
                    let y = (n - i) as i128;
                    (m + w, s + w * y, q + w * y * y)
                });
                let expected = cast(sign * sum as f64 / mass as f64).into_f64();
                assert!(
                    result
                        .fitted()
                        .iter()
                        .all(|&x| (x.into_f64() - expected).abs() <= 8.0 * eps * expected.abs())
                );
                assert_eq!(result.blocks().len(), 1);
                assert_eq!((result.blocks()[0].start, result.blocks()[0].end), (0, n));
                let fit = result.fitted()[0].into_f64();
                let loss = squares as f64 - 2.0 * sign * fit * sum as f64 + fit * fit * mass as f64;
                let diagnostics = result.diagnostics();
                assert_eq!(diagnostics.retained, n);
                assert_eq!(diagnostics.excluded, 0);
                assert_eq!(diagnostics.weight.unwrap().into_f64(), mass as f64);
                assert_eq!(result.blocks()[0].weight, diagnostics.weight);
                assert!(
                    (diagnostics.squared_error.unwrap().into_f64() - loss).abs()
                        <= 32.0 * eps * loss
                );
            }
        }
    }
}

#[test]
fn long_weighted_and_unweighted_runs_f32() {
    decreasing_runs::<f32>();
}

#[test]
fn long_weighted_and_unweighted_runs_f64() {
    decreasing_runs::<f64>();
}

fn corrections<S: RealScalar>() {
    let cast = |x| S::try_from_f64_lossy(x).unwrap();
    let large = 2.0_f64.powi(64);
    for sign in [1.0, -1.0] {
        let values = [large, 2.0, -large, large, 1.0, -large].map(|x| cast(sign * x));
        let direction = if sign > 0.0 {
            IsotonicDirection::Increasing
        } else {
            IsotonicDirection::Decreasing
        };
        let result = isotonic_regression_with_options(
            &values,
            Unweighted,
            direction,
            StatisticsOptions::new(),
        )
        .unwrap();
        assert_eq!(result.fitted(), &[cast(sign * 0.5); 6]);
        assert_eq!(result.blocks().len(), 1);
        // Leading, interior and trailing omitted observations do not affect the
        // run's support or mean, including when their source values are NaN.
        let mut extended = vec![S::NAN];
        let mut weights = vec![S::ZERO];
        for value in values {
            extended.extend([value, S::NAN]);
            weights.extend([S::ONE, S::ZERO]);
        }
        let omitted = isotonic_regression_with_options(
            &extended,
            Weights::<_, Mass>::new(&weights),
            direction,
            StatisticsOptions::new(),
        )
        .unwrap();
        assert_eq!(omitted.fitted(), vec![cast(sign * 0.5); extended.len()]);
        assert_eq!(omitted.diagnostics().retained, 6);
        assert_eq!(omitted.diagnostics().excluded, 7);
    }
}

#[test]
fn corrections_survive_run_merges_and_gaps() {
    corrections::<f32>();
    corrections::<f64>();
}

#[test]
fn native_wide_and_rejected_lanes_reuse_one_workspace() {
    let mut workspace = IsotonicWorkspace::default();
    workspace
        .try_reserve(&IsotonicWorkspace::scalar_requirements(3, StatisticsOptions::new()).unwrap())
        .unwrap();
    let mut output = [0.0; 3];
    let mut blocks = [IsotonicBlock::default(); 3];
    for values in [[3.0, 1.0, 2.0], [f64::MAX, 1.0, -f64::MAX], [3.0, 1.0, 2.0]] {
        let diagnostics = isotonic_regression_into(
            &values,
            Unweighted,
            IsotonicDirection::Increasing,
            StatisticsOptions::new(),
            &mut output,
            &mut blocks,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        let expected = if values[0] == f64::MAX {
            1.0 / 3.0
        } else {
            2.0
        };
        assert_eq!(output, [expected; 3]);
        assert_eq!(diagnostics.block_count, 1);
        let sentinel = (output, blocks);
        for invalid in [[f64::MAX, f64::NAN, 0.0], [3.0, f64::NAN, 0.0]] {
            assert!(
                isotonic_regression_into(
                    &invalid,
                    Unweighted,
                    IsotonicDirection::Increasing,
                    StatisticsOptions::new(),
                    &mut output,
                    &mut blocks,
                    &mut workspace,
                    ExecutionContext::sequential()
                )
                .is_err()
            );
            assert_eq!((output, blocks), sentinel);
        }
    }
    // The same worker must reset its admission decision between axis lanes.
    let values = [3.0, 1.0, 2.0, f64::MAX, 1.0, -f64::MAX, 3.0, 1.0, 2.0];
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[3, 3], &[3, 1]).unwrap(),
        Axis::Index(1),
    )
    .unwrap();
    let requirements = IsotonicWorkspace::workspace_requirements(
        view,
        StatisticsOptions::new(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    workspace.try_reserve(&requirements).unwrap();
    let mut fitted = [0.0; 9];
    let mut block_output = [IsotonicBlock::default(); 9];
    let mut lanes = [IsotonicLaneResult::default(); 3];
    isotonic_regression_axes_into(
        view,
        IsotonicDirection::Increasing,
        StatisticsOptions::new(),
        IsotonicBatchOutput {
            fitted: &mut fitted,
            blocks: &mut block_output,
            lanes: &mut lanes,
        },
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(
        fitted,
        [
            2.0,
            2.0,
            2.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            2.0,
            2.0,
            2.0
        ]
    );
    assert!(lanes.iter().all(|lane| lane.diagnostics.block_count == 1));
}
