//! Bitwise scalar references, exceptional blocks, and execution boundaries.
#![allow(clippy::unwrap_used)]

use core::num::NonZeroUsize;
use scioxide_core::{
    ExecutionContext, ExecutionPolicy, NdView, RealScalar, SciOxideError, StridedLayout,
};
use scioxide_spatial::*;

fn points<'a, S: RealScalar>(
    data: &'a [S],
    shape: &'a [usize; 2],
    strides: &'a [usize; 2],
) -> PointSet<'a, S> {
    PointSet::new(NdView::new(data, StridedLayout::new_broadcast(shape, strides).unwrap()).unwrap())
        .unwrap()
}

fn row<S: RealScalar>(points: &PointSet<'_, S>, i: usize) -> Vec<S> {
    (0..points.dimension())
        .map(|k| points.coordinate(i, k).unwrap())
        .collect()
}

fn bits<S: RealScalar>(values: &[S]) -> Vec<u64> {
    values.iter().map(|x| x.into_f64().to_bits()).collect()
}

struct Partitions;
impl scioxide_core::Executor for Partitions {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: scioxide_core::ExecutionPlan,
        operation: &scioxide_core::PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        scioxide_core::Executor::execute(
            &scioxide_core::SequentialExecutor,
            context,
            plan,
            operation,
        )
    }
}

fn compare<S: RealScalar>(
    metric: &PreparedDistance,
    left: &PointSet<'_, S>,
    right: &PointSet<'_, S>,
) {
    let cross: Result<Vec<_>, _> = (0..left.len())
        .flat_map(|i| (0..right.len()).map(move |j| metric.distance(&row(left, i), &row(right, j))))
        .collect();
    let upper: Result<Vec<_>, _> = (0..left.len())
        .flat_map(|i| {
            (i + 1..left.len()).map(move |j| metric.distance(&row(left, i), &row(left, j)))
        })
        .collect();
    for policy in [
        ExecutionPolicy::Sequential,
        ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap()),
    ] {
        let options = DistanceOptions::new().with_execution_policy(policy);
        for (condensed, expected) in [(false, &cross), (true, &upper)] {
            let request = if condensed {
                DistanceWorkspaceRequest::condensed(left.len(), options)
            } else {
                DistanceWorkspaceRequest::pairwise(left.len(), right.len(), options)
            };
            let len = DistanceWorkspace::<S>::workspace_requirements(request)
                .unwrap()
                .scalar_len();
            let mut workspace = DistanceWorkspace::workspace(request).unwrap();
            let mut out = vec![S::NEG_ZERO; len + 2];
            let result = if condensed {
                metric.condensed_into(
                    left,
                    options,
                    &mut out,
                    &mut workspace,
                    ExecutionContext::new(&Partitions),
                )
            } else {
                metric.pairwise_into(
                    left,
                    right,
                    options,
                    &mut out,
                    &mut workspace,
                    ExecutionContext::new(&Partitions),
                )
            };
            match expected {
                Ok(expected) => {
                    result.unwrap();
                    assert_eq!(bits(&out[..len]), bits(expected), "{metric:?}, {policy:?}");
                }
                Err(expected) => {
                    assert_eq!(&result.unwrap_err(), expected, "{metric:?}, {policy:?}");
                    assert_eq!(bits(&out[..len]), bits(&vec![S::NEG_ZERO; len]));
                }
            }
            assert_eq!(bits(&out[len..]), bits(&[S::NEG_ZERO; 2]));
        }
    }
}

fn all_metrics<S: RealScalar>() {
    use DistanceMetric::*;
    for dimension in [0, 1, 3, 7] {
        let shape = [33, dimension];
        let target_shape = [35, dimension];
        let matrix: Vec<_> = (0..dimension * dimension)
            .map(|i| {
                if i / dimension == i % dimension {
                    2.0
                } else {
                    0.0
                }
            })
            .collect();
        let mut metrics: Vec<_> = [
            BrayCurtis,
            Canberra,
            Chebyshev,
            CityBlock,
            Correlation,
            Cosine,
            Euclidean,
            SquaredEuclidean,
            Minkowski(0.5),
            Minkowski(1.0),
            Minkowski(1.5),
            Minkowski(2.0),
            Minkowski(3.0),
            JensenShannon,
            Hamming,
        ]
        .into_iter()
        .map(|m| (PreparedDistance::new(m).unwrap(), false))
        .collect();
        metrics.extend(
            [Dice, Jaccard, RogersTanimoto, RussellRao, SokalSneath, Yule]
                .into_iter()
                .map(|m| (PreparedDistance::new(m).unwrap(), true)),
        );
        metrics.push((
            PreparedDistance::standardized(&vec![2.0; dimension]).unwrap(),
            false,
        ));
        metrics.push((
            PreparedDistance::mahalanobis(
                scioxide_linalg::MatrixView::from_slice(dimension, dimension, &matrix).unwrap(),
            )
            .unwrap(),
            false,
        ));
        for (metric, boolean) in metrics {
            for (a_stride, b_stride) in [
                ([dimension, 1], [dimension, 1]),
                ([dimension + 2, 1], [dimension + 3, 1]),
                ([2 * dimension + 1, 2], [2 * dimension + 1, 2]),
                ([dimension, 1], [2 * dimension + 1, 2]),
                ([0, 1], [dimension, 1]),
                ([1, 0], [0, 0]),
            ] {
                let storage = |rows: usize, strides: [usize; 2], shift: usize| {
                    let mut data = vec![S::NAN; rows * strides[0] + dimension * strides[1] + 1];
                    for i in 0..rows {
                        for k in 0..dimension {
                            let x = if boolean {
                                if k == 0 || (i + k + shift).is_multiple_of(3) {
                                    1.0
                                } else {
                                    0.0
                                }
                            } else {
                                0.1 + ((i * 17 + k * 11 + shift) % 43) as f64 / 13.0
                            };
                            data[i * strides[0] + k * strides[1]] =
                                S::try_from_f64_lossy(x).unwrap();
                        }
                    }
                    data
                };
                let a = storage(33, a_stride, 1);
                let b = storage(35, b_stride, 5);
                compare(
                    &metric,
                    &points(&a, &shape, &a_stride),
                    &points(&b, &target_shape, &b_stride),
                );
            }
        }
    }
}

#[test]
fn every_metric_width_and_layout_matches_scalar_bits() {
    all_metrics::<f32>();
    all_metrics::<f64>();
}

fn zero_and_duplicate_rows<S: RealScalar>() {
    for dimension in [0, 1, 3, 7] {
        let shape = [129, dimension];
        for strides in [
            [dimension, 1],
            [dimension + 2, 1],
            [2 * dimension, 2],
            [0, 1],
        ] {
            let mut data = vec![S::ZERO; 129 * strides[0] + dimension * strides[1] + 1];
            for i in 0..129 {
                for axis in 0..dimension {
                    let x = match (i + axis) % 5 {
                        0 => S::NEG_ZERO,
                        1 => S::ZERO,
                        k => S::try_from_f64_lossy((k - 2) as f64 * 0.5).unwrap(),
                    };
                    data[i * strides[0] + axis * strides[1]] = x;
                }
            }
            let p = points(&data, &shape, &strides);
            for metric in [
                DistanceMetric::Euclidean,
                DistanceMetric::SquaredEuclidean,
                DistanceMetric::Minkowski(2.0),
                DistanceMetric::CityBlock,
                DistanceMetric::Chebyshev,
            ] {
                compare(&PreparedDistance::new(metric).unwrap(), &p, &p);
            }
            assert_eq!(
                directed_hausdorff(&p, &p).unwrap(),
                hausdorff_reference(&p, &p).unwrap()
            );
        }
    }
}

#[test]
fn zero_and_duplicate_rows_preserve_bits_and_witnesses() {
    zero_and_duplicate_rows::<f32>();
    zero_and_duplicate_rows::<f64>();
}

#[test]
fn zero_bounds_do_not_hide_tiny_nonzero_hausdorff_pairs() {
    // A zero, an underflowed square and a positive subnormal square all share
    // a block with a normal square. Min/max alone cannot certify that block.
    for strides in [[3, 1], [6, 2]] {
        let shape = [65, 3];
        let mut data = vec![0.0_f64; 65 * strides[0]];
        data[strides[0]] = 1e-200;
        data[2 * strides[0]] = 1e-160;
        data[3 * strides[0]] = 1.0;
        let p = points(&data, &shape, &strides);
        let target_shape = [1, 3];
        let zeros = [0.0; 6];
        let target = points(&zeros, &target_shape, &strides);
        assert_eq!(
            directed_hausdorff(&target, &p).unwrap(),
            hausdorff_reference(&target, &p).unwrap()
        );
        assert_eq!(
            directed_hausdorff(&p, &target).unwrap(),
            hausdorff_reference(&p, &target).unwrap()
        );
        for metric in [DistanceMetric::Euclidean, DistanceMetric::SquaredEuclidean] {
            compare(&PreparedDistance::new(metric).unwrap(), &p, &target);
        }
    }
}

fn conversion_partitions<S: RealScalar>() {
    for n in [0, 1, 2, 17, 33, 65] {
        let upper: Vec<_> = (0..n)
            .flat_map(|i| {
                (i + 1..n).map(move |j| {
                    if (i + j) % 7 == 0 {
                        S::NEG_ZERO
                    } else {
                        S::try_from_f64_lossy(((i * 17 + j * 11) % 43) as f64 * 0.5).unwrap()
                    }
                })
            })
            .collect();
        let mut square = vec![S::ZERO; n * n];
        for i in 0..n {
            for j in i + 1..n {
                let value = upper[condensed_index(n, i, j).unwrap()];
                square[i * n + j] = value;
                square[j * n + i] = value;
            }
        }
        let row_stride = 2 * n + 3;
        let mut padded = vec![S::NAN; n * row_stride];
        for i in 0..n {
            for j in 0..n {
                padded[i * row_stride + 2 * j] = square[i * n + j];
            }
        }
        let matrix =
            scioxide_linalg::MatrixView::from_slice_with_strides(n, n, &padded, row_stride, 2)
                .unwrap();
        for policy in [
            ExecutionPolicy::Sequential,
            ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap()),
        ] {
            let options = DistanceOptions::new().with_execution_policy(policy);
            let mut workspace =
                DistanceWorkspace::workspace(DistanceWorkspaceRequest::square(n, options)).unwrap();
            let mut output = vec![S::NEG_ZERO; n * n + 2];
            condensed_to_square_into(
                n,
                &upper,
                options,
                &mut output,
                &mut workspace,
                ExecutionContext::new(&Partitions),
            )
            .unwrap();
            assert_eq!(bits(&output[..n * n]), bits(&square));
            assert_eq!(bits(&output[n * n..]), bits(&[S::NEG_ZERO; 2]));
            output.fill(S::NEG_ZERO);
            square_to_condensed_into(
                matrix,
                S::ZERO,
                options,
                &mut output,
                &mut workspace,
                ExecutionContext::new(&Partitions),
            )
            .unwrap();
            assert_eq!(bits(&output[..upper.len()]), bits(&upper));
            assert!(
                output[upper.len()..]
                    .iter()
                    .all(|x| x.into_f64().to_bits() == S::NEG_ZERO.into_f64().to_bits())
            );
        }
    }
}

#[test]
fn conversions_preserve_bits_across_partial_rows_layouts_and_empty_partitions() {
    conversion_partitions::<f32>();
    conversion_partitions::<f64>();
}

fn exceptional<S: RealScalar>(scales: &[f64]) {
    let shape = [65, 3];
    let stride = [3, 1];
    for &scale in scales {
        for position in [0, 1, 15, 31, 32, 33, 63, 64] {
            let mut data = vec![S::ZERO; 65 * 3];
            data[3 * position] = S::try_from_f64_lossy(3.0 * scale).unwrap();
            data[3 * position + 1] = S::try_from_f64_lossy(4.0 * scale).unwrap();
            let p = points(&data, &shape, &stride);
            for metric in [
                DistanceMetric::Euclidean,
                DistanceMetric::SquaredEuclidean,
                DistanceMetric::Minkowski(2.0),
                DistanceMetric::CityBlock,
                DistanceMetric::Chebyshev,
            ] {
                compare(&PreparedDistance::new(metric).unwrap(), &p, &p);
            }
            compare(
                &PreparedDistance::standardized(&[1.0, 1.0, 1.0]).unwrap(),
                &p,
                &p,
            );
        }
    }
}

#[test]
fn retries_and_overflow_match_scalar_bits_and_errors() {
    exceptional::<f64>(&[1e200, 1e-200, 1e-160, f64::MIN_POSITIVE, 1.0]);
    exceptional::<f32>(&[1e37, 1e-40, f32::MIN_POSITIVE as f64, 1.0]);
    let shape = [3, 1];
    let stride = [1, 1];
    let data = [0.0_f32, f32::MAX, -f32::MAX];
    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::SquaredEuclidean,
        DistanceMetric::CityBlock,
        DistanceMetric::Chebyshev,
    ] {
        let metric = PreparedDistance::new(metric).unwrap();
        let p = points(&data, &shape, &stride);
        compare(&metric, &p, &p);
        let wide = [0.0_f64, f64::MAX, -f64::MAX];
        let p = points(&wide, &shape, &stride);
        compare(&metric, &p, &p);
    }
    // Subtracting a tiny negative coordinate gives a binary64 value above MAX
    // that still rounds to finite f32::MAX. A conservative range check is wrong.
    let data = [f32::MAX, -f32::from_bits(0x6a80_0000), 0.0];
    let p = points(&data, &shape, &stride);
    compare(
        &PreparedDistance::new(DistanceMetric::Euclidean).unwrap(),
        &p,
        &p,
    );
}

fn output_block_edges<S: RealScalar>(scales: &[f64]) {
    let shape = [259, 3];
    for strides in [[3, 1], [8, 2]] {
        for &scale in scales {
            for position in [255, 256, 257, 258] {
                let mut data = vec![S::ZERO; 259 * strides[0]];
                data[position * strides[0]] = S::try_from_f64_lossy(scale).unwrap();
                let p = points(&data, &shape, &strides);
                for kind in [DistanceMetric::Euclidean, DistanceMetric::SquaredEuclidean] {
                    compare(&PreparedDistance::new(kind).unwrap(), &p, &p);
                }
            }
        }
    }
}

#[test]
fn extended_output_blocks_preserve_underflow_and_overflow_at_their_edges() {
    output_block_edges::<f64>(&[1e-200, 1e200]);
    output_block_edges::<f32>(&[1e-40, 1e38]);
}

#[test]
fn workspace_recovers_after_late_numerical_or_domain_failure() {
    let shape = [65, 3];
    let strides = [3, 1];
    let healthy = [1.0_f64, 0.0, 1.0].repeat(65);
    let good = points(&healthy, &shape, &strides);
    let options = DistanceOptions::new();
    for (kind, bad_row) in [
        (DistanceMetric::SquaredEuclidean, [f64::MAX, f64::MAX, 0.0]),
        (DistanceMetric::Cosine, [0.0; 3]),
    ] {
        let metric = PreparedDistance::new(kind).unwrap();
        let mut data = healthy.clone();
        data[64 * 3..].copy_from_slice(&bad_row);
        let bad = points(&data, &shape, &strides);
        let mut workspace =
            DistanceWorkspace::workspace(DistanceWorkspaceRequest::pairwise(65, 65, options))
                .unwrap();
        let mut output = vec![-1.0; 65 * 65];
        for condensed in [false, true] {
            output.fill(-1.0);
            if condensed {
                assert!(
                    metric
                        .condensed_into(
                            &bad,
                            options,
                            &mut output,
                            &mut workspace,
                            ExecutionContext::sequential()
                        )
                        .is_err()
                );
                assert!(output.iter().all(|&x| x == -1.0));
                metric
                    .condensed_into(
                        &good,
                        options,
                        &mut output,
                        &mut workspace,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
            } else {
                assert!(
                    metric
                        .pairwise_into(
                            &bad,
                            &bad,
                            options,
                            &mut output,
                            &mut workspace,
                            ExecutionContext::sequential()
                        )
                        .is_err()
                );
                assert!(output.iter().all(|&x| x == -1.0));
                metric
                    .pairwise_into(
                        &good,
                        &good,
                        options,
                        &mut output,
                        &mut workspace,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
            }
            let expected = metric.distance(&healthy[..3], &healthy[..3]).unwrap();
            let len = if condensed {
                condensed_len(65).unwrap()
            } else {
                65 * 65
            };
            assert!(
                output[..len]
                    .iter()
                    .all(|x| x.to_bits() == expected.to_bits())
            );
        }
    }
}

fn domain_positions<S: RealScalar>() {
    let shape = [65, 3];
    let stride = [3, 1];
    use DistanceMetric::*;
    for metric in [
        BrayCurtis,
        Correlation,
        Cosine,
        JensenShannon,
        Dice,
        Jaccard,
        RogersTanimoto,
        RussellRao,
        SokalSneath,
        Yule,
    ] {
        for position in [0, 1, 15, 31, 32, 33, 63, 64] {
            let good = [S::ONE, S::ZERO, S::ONE];
            let mut data = good.repeat(65);
            let bad = match metric {
                BrayCurtis => [-S::ONE, S::ZERO, -S::ONE],
                Dice | Jaccard | RogersTanimoto | RussellRao | SokalSneath | Yule => {
                    [S::ONE, S::try_from_f64_lossy(0.5).unwrap(), S::ONE]
                }
                _ => [S::ZERO; 3],
            };
            data[3 * position..3 * position + 3].copy_from_slice(&bad);
            let p = points(&data, &shape, &stride);
            compare(&PreparedDistance::new(metric).unwrap(), &p, &p);
        }
    }
}

#[test]
fn domain_failures_at_block_positions_preserve_output() {
    domain_positions::<f32>();
    domain_positions::<f64>();
}

#[test]
fn simple_metrics_retain_direct_formula_operation_order() {
    let a = [0.0_f64; 7];
    let b = [1.0_f64, 1.0, 1e-9, 3e-9, 1e-10, 0.0, -0.0];
    let shape = [1, 7];
    let strides = [7, 1];
    let square = b.iter().fold(0.0, |sum, x| sum + x * x);
    let norm = square.sqrt();
    assert_ne!((norm * norm).to_bits(), square.to_bits());
    for (metric, expected) in [
        (DistanceMetric::Euclidean, norm),
        (DistanceMetric::SquaredEuclidean, norm * norm),
        (
            DistanceMetric::CityBlock,
            b.iter().fold(0.0, |sum, x| sum + x.abs()),
        ),
        (DistanceMetric::Chebyshev, 1.0),
    ] {
        let output = PreparedDistance::new(metric)
            .unwrap()
            .pairwise(&points(&a, &shape, &strides), &points(&b, &shape, &strides))
            .unwrap()
            .as_slice()
            .to_vec();
        assert_eq!(output[0].to_bits(), expected.to_bits());
    }
}

fn shuffle(n: usize, mut state: u64) -> Vec<usize> {
    let mut indices: Vec<_> = (0..n).collect();
    for i in (1..n).rev() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let j = (u128::from(state.wrapping_mul(0x2545_f491_4f6c_dd1d)) % (i as u128 + 1)) as usize;
        indices.swap(i, j);
    }
    indices
}

// Straightforward checked traversal preserves which errors the early break
// observes. The production implementation batches arithmetic and validation.
fn hausdorff_reference<S: RealScalar>(
    source: &PointSet<'_, S>,
    target: &PointSet<'_, S>,
) -> Result<HausdorffResult<S>, SciOxideError> {
    let metric = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    let mut best = HausdorffResult {
        distance: S::ZERO,
        source: usize::MAX,
        target: 0,
    };
    for i in shuffle(source.len(), 0x243f_6a88_85a3_08d3) {
        let mut nearest = S::MAX;
        let mut witness = usize::MAX;
        let mut pruned = false;
        for j in shuffle(target.len(), 0x1319_8a2e_0370_7344) {
            let d = metric.distance(&row(source, i), &row(target, j))?;
            if d < best.distance {
                pruned = true;
                break;
            }
            if d < nearest || (d == nearest && j < witness) {
                nearest = d;
                witness = j;
            }
        }
        if !pruned && (nearest > best.distance || (nearest == best.distance && i < best.source)) {
            best = HausdorffResult {
                distance: nearest,
                source: i,
                target: witness,
            };
        }
    }
    Ok(best)
}

fn hausdorff_cases<S: RealScalar>(scales: &[f64]) {
    for n in [1, 2, 31, 32, 33, 65] {
        let shape = [n, 3];
        let target_shape = [n + 2, 3];
        for strides in [[3, 1], [5, 1], [7, 2], [0, 1], [1, 0], [0, 0]] {
            for &scale in scales {
                let data: Vec<_> = (0..(n + 2) * 7)
                    .map(|i| S::try_from_f64_lossy(((i * 17 % 23) as f64 - 11.0) * scale).unwrap())
                    .collect();
                let source = points(&data, &shape, &strides);
                let target = points(&data[1..], &target_shape, &strides);
                let expected = hausdorff_reference(&source, &target).unwrap();
                let actual = directed_hausdorff(&source, &target).unwrap();
                assert_eq!(bits(&[actual.distance]), bits(&[expected.distance]));
                assert_eq!(
                    (actual.source, actual.target),
                    (expected.source, expected.target)
                );
            }
        }
    }
    let source_order = shuffle(2, 0x243f_6a88_85a3_08d3);
    let target_order = shuffle(2, 0x1319_8a2e_0370_7344);
    let mut a = [S::ZERO; 2];
    a[source_order[1]] = S::MAX;
    let half = S::try_from_f64_lossy(0.5).unwrap();
    let mut b = [S::ZERO; 2];
    b[target_order[0]] = S::MAX * S::try_from_f64_lossy(0.75).unwrap();
    b[target_order[1]] = -S::MAX * half;
    let shape = [2, 1];
    let strides = [1, 1];
    let source = points(&a, &shape, &strides);
    let target = points(&b, &shape, &strides);
    let expected = hausdorff_reference(&source, &target).unwrap();
    assert_eq!(directed_hausdorff(&source, &target).unwrap(), expected);
    b.swap(0, 1);
    let target = points(&b, &shape, &strides);
    assert_eq!(
        directed_hausdorff(&source, &target).unwrap_err(),
        hausdorff_reference(&source, &target).unwrap_err()
    );
}

#[test]
fn hausdorff_blocks_preserve_bits_witnesses_and_observed_errors() {
    hausdorff_cases::<f64>(&[0.0, 0.1, 1e-200, 1e-160, 1e200]);
    hausdorff_cases::<f32>(&[0.0, 0.1, 1e-40, 1e36]);
    let shape = [2, 2];
    let one = [1, 2];
    let strides = [2, 1];
    let source_data = [0.0_f32, 1e-4, 0.0, 0.0];
    let target_data = [1.0_f32, 0.0];
    let actual = directed_hausdorff(
        &points(&source_data, &shape, &strides),
        &points(&target_data, &one, &strides),
    )
    .unwrap();
    assert_eq!(
        (actual.distance.to_bits(), actual.source),
        (1.0_f32.to_bits(), 0)
    );
    let actual = directed_hausdorff(
        &points(&target_data, &one, &strides),
        &points(&source_data, &shape, &strides),
    )
    .unwrap();
    assert_eq!(
        (actual.distance.to_bits(), actual.target),
        (1.0_f32.to_bits(), 0)
    );

    // Different binary64 square sums can have the same rounded square root.
    // The smaller raw sum must not displace the lower original target index.
    let source_data = [0.0_f64; 3];
    let target_data = [1.0, 1.0 + f64::EPSILON, 0.0, 1.0, 1.0, 0.0];
    let source_shape = [1, 3];
    let target_shape = [2, 3];
    let strides = [3, 1];
    let actual = directed_hausdorff(
        &points(&source_data, &source_shape, &strides),
        &points(&target_data, &target_shape, &strides),
    )
    .unwrap();
    assert_eq!(
        (actual.distance.to_bits(), actual.target),
        (2.0_f64.sqrt().to_bits(), 0)
    );
}
