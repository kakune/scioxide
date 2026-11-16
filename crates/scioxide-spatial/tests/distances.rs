//! Independent distance formulas and borrowed-layout contracts.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use scioxide_core::{NdView, RealScalar, StridedLayout};
use scioxide_spatial::*;

fn points<'a, S: RealScalar>(
    data: &'a [S],
    shape: &'a [usize; 2],
    strides: &'a [usize; 2],
) -> PointSet<'a, S> {
    PointSet::new(NdView::new(data, StridedLayout::new(shape, strides).unwrap()).unwrap()).unwrap()
}
fn check<S: RealScalar>(metric: DistanceMetric, a: &[f64], b: &[f64], expected: f64) {
    let a: Vec<S> = a
        .iter()
        .map(|&x| S::try_from_f64_lossy(x).unwrap())
        .collect();
    let b: Vec<S> = b
        .iter()
        .map(|&x| S::try_from_f64_lossy(x).unwrap())
        .collect();
    let d = PreparedDistance::new(metric)
        .unwrap()
        .distance(&a, &b)
        .unwrap()
        .into_f64();
    assert!(
        (d - expected).abs() <= 64.0 * S::EPSILON.into_f64() * expected.abs().max(1.0),
        "{metric:?}: {d} != {expected}"
    );
}
fn formulas<S: RealScalar>() {
    use DistanceMetric::*;
    for (metric, expected) in [
        (Euclidean, 5.0),
        (SquaredEuclidean, 25.0),
        (CityBlock, 7.0),
        (Chebyshev, 4.0),
        (Minkowski(3.0), 91.0_f64.cbrt()),
        (Canberra, 2.0),
        (BrayCurtis, 1.0),
    ] {
        check::<S>(metric, &[0.0, 0.0], &[3.0, 4.0], expected);
    }
    check::<S>(Cosine, &[1.0, 0.0], &[0.0, 1.0], 1.0);
    check::<S>(Correlation, &[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0], 2.0);
    check::<S>(JensenShannon, &[2.0, 0.0], &[0.0, 7.0], 2.0_f64.ln().sqrt());
    // Contingency table: tt=2, tf=1, ft=2, ff=1.
    let a = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let b = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    for (metric, expected) in [
        (Dice, 3.0 / 7.0),
        (Hamming, 0.5),
        (Jaccard, 0.6),
        (RogersTanimoto, 2.0 / 3.0),
        (RussellRao, 2.0 / 3.0),
        (SokalSneath, 0.75),
        (Yule, 1.0),
    ] {
        check::<S>(metric, &a, &b, expected);
    }
    check::<S>(Hamming, &[2.0, 3.0], &[2.0, 4.0], 0.5);
    check::<S>(Jaccard, &[0.0, 0.0], &[0.0, 0.0], 0.0);
    check::<S>(Yule, &[1.0, 1.0], &[0.0, 0.0], 0.0);
}
#[test]
fn formulas_both_widths() {
    formulas::<f32>();
    formulas::<f64>();
}
#[test]
fn parameterized_metrics() {
    let metric = PreparedDistance::standardized(&[4.0_f64, 9.0]).unwrap();
    assert!((metric.distance(&[0.0, 0.0], &[2.0, 3.0]).unwrap() - 2.0_f64.sqrt()).abs() < 1e-14);
    let data = [4.0, 2.0, 2.0, 3.0];
    let shape = [2, 2];
    let stride = [2, 1];
    let matrix = matrix_view(&data, &shape, &stride);
    let metric = PreparedDistance::mahalanobis(matrix).unwrap();
    // [1,2] A [1,2]^T = 4+8+12 = 24.
    assert!((metric.distance(&[0.0, 0.0], &[1.0, 2.0]).unwrap() - 24.0_f64.sqrt()).abs() < 1e-14);
    assert!(PreparedDistance::standardized(&[0.0]).is_err());
    let singular = [1.0, 1.0, 1.0, 1.0];
    assert!(PreparedDistance::mahalanobis(matrix_view(&singular, &shape, &stride)).is_err());
}
#[test]
fn extreme_scales_and_domains() {
    let euclidean = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    assert!(
        (euclidean
            .distance(&[0.0, 0.0], &[3e200_f64, 4e200])
            .unwrap()
            / 5e200
            - 1.0)
            .abs()
            < 1e-14
    );
    assert!(
        (euclidean
            .distance(&[0.0, 0.0], &[3e-200_f64, 4e-200])
            .unwrap()
            / 5e-200
            - 1.0)
            .abs()
            < 1e-14
    );
    assert!(euclidean.distance(&[f64::MAX], &[-f64::MAX]).is_err());
    assert!(euclidean.distance(&[f64::NAN], &[0.0]).is_err());
    assert_eq!(euclidean.distance::<f64>(&[], &[]).unwrap(), 0.0);
    for metric in [
        DistanceMetric::Cosine,
        DistanceMetric::Correlation,
        DistanceMetric::BrayCurtis,
        DistanceMetric::Dice,
        DistanceMetric::SokalSneath,
    ] {
        assert!(
            PreparedDistance::new(metric)
                .unwrap()
                .distance(&[0.0], &[0.0])
                .is_err()
        );
    }
    assert!(PreparedDistance::new(DistanceMetric::Minkowski(0.0)).is_err());
    assert!(
        PreparedDistance::new(DistanceMetric::JensenShannon)
            .unwrap()
            .distance(&[-1.0], &[1.0])
            .is_err()
    );
    check::<f64>(DistanceMetric::Cosine, &[1e300, 0.0], &[0.0, 1e300], 1.0);
}
// For p < 1 the scaled root `sum^(1/p)` overflows long before the norm does.
#[test]
fn small_minkowski_powers_overflow_only_with_the_norm() {
    for (p, coordinate, expected) in [
        (1e-3, 1e-300, 1e-300 * 3.0_f64.powi(500) * 3.0_f64.powi(500)),
        (1e-3, 1e-200, 1e-200 * 3.0_f64.powi(500) * 3.0_f64.powi(500)),
        (0.5, 2.0, 18.0),
    ] {
        let metric = PreparedDistance::new(DistanceMetric::Minkowski(p)).unwrap();
        let actual: f64 = metric.distance(&[coordinate; 3], &[0.0; 3]).unwrap();
        assert!(
            (actual / expected - 1.0).abs() < 1e-12,
            "p={p}: {actual} vs {expected}"
        );
    }
    let metric = PreparedDistance::new(DistanceMetric::Minkowski(1e-3)).unwrap();
    let error = metric.distance(&[1e-100_f64; 3], &[0.0; 3]).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::NumericalInstability
    );
    let narrow = PreparedDistance::new(DistanceMetric::Minkowski(0.05)).unwrap();
    let actual: f32 = narrow.distance(&[1e-30_f32; 2], &[0.0; 2]).unwrap();
    assert!((f64::from(actual) / (1e-30 * 2.0_f64.powi(20)) - 1.0).abs() < 1e-6);
}
#[test]
fn strided_pairwise_condensed_and_transaction() {
    let data = [0.0, 99.0, 0.0, 99.0, 3.0, 99.0, 4.0, 99.0, 6.0, 99.0, 8.0];
    let shape = [3, 2];
    let strides = [4, 2];
    let pts = points(&data, &shape, &strides);
    let metric = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    assert_eq!(
        metric.pairwise(&pts, &pts).unwrap().as_slice().to_vec(),
        [0.0, 5.0, 10.0, 5.0, 0.0, 5.0, 10.0, 5.0, 0.0]
    );
    let mut values = [-1.0; 3];
    let mut scratch = distance_workspace(3);
    metric
        .condensed_into(
            &pts,
            DistanceOptions::new(),
            &mut values,
            &mut scratch,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(values, [5.0, 10.0, 5.0]);
    let mut square = [0.0; 9];
    condensed_to_square_into(
        3,
        &values,
        DistanceOptions::new(),
        &mut square,
        &mut distance_workspace(3 * 3),
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    let matrix_shape = [3, 3];
    let matrix_strides = [3, 1];
    let matrix = matrix_view(&square, &matrix_shape, &matrix_strides);
    let mut roundtrip = [0.0; 3];
    square_to_condensed_into(
        matrix,
        0.0,
        DistanceOptions::new(),
        &mut roundtrip,
        &mut distance_workspace(condensed_len(matrix.rows()).unwrap()),
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(values, roundtrip);
    let undefined = PreparedDistance::new(DistanceMetric::Cosine).unwrap();
    let mut output = [42.0; 9];
    let mut work = distance_workspace(9);
    assert!(
        undefined
            .pairwise_into(
                &pts,
                &pts,
                DistanceOptions::new(),
                &mut output,
                &mut work,
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(output, [42.0; 9]);
    assert!(validate_condensed(3, &[1.0, -1.0, 2.0]).is_err());
    assert!(condensed_len(usize::MAX).is_err());
    for n in 0..20 {
        let mut expected = 0;
        for i in 0..n {
            for j in i + 1..n {
                assert_eq!(condensed_index(n, i, j).unwrap(), expected);
                expected += 1;
            }
        }
        assert_eq!(condensed_len(n).unwrap(), expected);
    }
}
#[test]
fn hausdorff_independent_witness() {
    let a = [0.0, 0.0, 3.0, 4.0];
    let b = [0.0, 0.0];
    let s = [2, 2];
    let t = [1, 2];
    let stride = [2, 1];
    let result = directed_hausdorff(&points(&a, &s, &stride), &points(&b, &t, &stride)).unwrap();
    assert_eq!(
        result,
        HausdorffResult {
            distance: 5.0,
            source: 1,
            target: 0
        }
    );
    assert_eq!(
        directed_hausdorff(&points(&b, &t, &stride), &points(&a, &s, &stride))
            .unwrap()
            .distance,
        0.0
    );
}

#[test]
fn jensen_shannon_near_coincident_has_no_cancellation() {
    let delta = 1e-8_f64;
    let a = [0.5 + delta, 0.5 - delta];
    let b = [0.5 - delta, 0.5 + delta];
    let distance = PreparedDistance::new(DistanceMetric::JensenShannon)
        .unwrap()
        .distance(&a, &b)
        .unwrap();
    // Leading positive term is sqrt(2)*delta; the next relative term is O(delta²).
    let actual_delta = (a[0] - b[0]) * 0.5;
    assert!((distance / (2.0_f64.sqrt() * actual_delta) - 1.0).abs() < 1e-14);
}

#[test]
fn metric_properties_and_dissimilarity_counterexample() {
    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::CityBlock,
        DistanceMetric::Chebyshev,
        DistanceMetric::Minkowski(1.5),
    ] {
        let prepared = PreparedDistance::new(metric).unwrap();
        for i in 0..30 {
            let a = [i as f64 / 7.0, ((i * 11) % 17) as f64];
            let b = [((i * 3) % 13) as f64, i as f64 / 11.0];
            let c = [1.0, 2.0];
            let ab = prepared.distance(&a, &b).unwrap();
            let bc = prepared.distance(&b, &c).unwrap();
            let ac = prepared.distance(&a, &c).unwrap();
            assert!(ab >= 0.0);
            assert_eq!(ab, prepared.distance(&b, &a).unwrap());
            assert_eq!(prepared.distance(&a, &a).unwrap(), 0.0);
            assert!(ac <= ab + bc + 1e-12);
        }
    }
    let squared = PreparedDistance::new(DistanceMetric::SquaredEuclidean).unwrap();
    assert!(
        squared.distance(&[0.0], &[2.0]).unwrap()
            > squared.distance(&[0.0], &[1.0]).unwrap() + squared.distance(&[1.0], &[2.0]).unwrap()
    );
}

#[test]
fn parameterized_f32_and_invalid_shapes() {
    let metric = PreparedDistance::standardized(&[4.0_f32, 9.0]).unwrap();
    assert!((metric.distance(&[0.0_f32, 0.0], &[2.0, 3.0]).unwrap() - 2.0_f32.sqrt()).abs() < 1e-6);
    let data = [4.0_f32, 2.0, 2.0, 3.0];
    let shape = [2, 2];
    let strides = [2, 1];
    let metric = PreparedDistance::mahalanobis(matrix_view(&data, &shape, &strides)).unwrap();
    assert!(
        (metric.distance(&[0.0_f32, 0.0], &[1.0, 2.0]).unwrap() - 24.0_f32.sqrt()).abs() < 1e-6
    );
    let asymmetric = [1.0, 0.5, 0.0, 1.0];
    assert!(PreparedDistance::mahalanobis(matrix_view(&asymmetric, &shape, &strides)).is_err());
    let indefinite = [1.0, 2.0, 2.0, 1.0];
    assert!(PreparedDistance::mahalanobis(matrix_view(&indefinite, &shape, &strides)).is_err());
    assert!(metric.distance(&[0.0_f32], &[0.0]).is_err());
    let asymmetric = [0.0, 1.0, 2.0, 0.0];
    assert!(validate_distance_matrix(matrix_view(&asymmetric, &shape, &strides), 0.0).is_err());
}

#[test]
fn largest_representable_condensed_index_avoids_intermediate_overflow() {
    // Integer bisection avoids assuming a pointer width or using float sqrt.
    let mut low = 1usize;
    let mut high = usize::MAX;
    while low < high {
        let mid = low + (high - low) / 2 + (high - low) % 2;
        if condensed_len(mid).is_ok() {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    assert_eq!(
        condensed_index(low, low - 2, low - 1).unwrap(),
        condensed_len(low).unwrap() - 1
    );
}

#[test]
fn row_distance_reports_the_invalid_index() {
    let data = [0.0_f64];
    let shape = [1, 1];
    let strides = [1, 1];
    let pts = points(&data, &shape, &strides);
    let metric = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    assert!(
        matches ! (& (metric . between (& pts , 1 , & pts , 0)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: OutOfBounds { .. }))
    );
    assert!(
        matches ! (& (metric . between (& pts , 0 , & pts , 2)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: OutOfBounds { .. }))
    );
}

fn close<S: RealScalar>(actual: S, expected: S) {
    assert!(
        (actual - expected).abs()
            <= S::try_from_f64_lossy(64.0).unwrap() * S::EPSILON * expected.abs().max(S::ONE),
        "{actual:?} != {expected:?}"
    );
}

fn bulk_metrics<S: RealScalar>() {
    use DistanceMetric::*;
    let shape = [33, 3];
    let target_shape = [35, 3];
    let matrix_shape = [3, 3];
    let matrix_strides = [3, 1];
    let vi: Vec<S> = [4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0]
        .map(|x| S::try_from_f64_lossy(x).unwrap())
        .into();
    let mut metrics: Vec<_> = [
        BrayCurtis,
        Canberra,
        Chebyshev,
        CityBlock,
        Correlation,
        Cosine,
        Euclidean,
        SquaredEuclidean,
        Minkowski(1.5),
        Minkowski(2.0),
        Minkowski(3.0),
        JensenShannon,
        Dice,
        Hamming,
        Jaccard,
        RogersTanimoto,
        RussellRao,
        SokalSneath,
        Yule,
    ]
    .into_iter()
    .map(|m| PreparedDistance::new(m).unwrap())
    .collect();
    metrics.push(PreparedDistance::standardized(&[4.0, 9.0, 16.0]).unwrap());
    metrics.push(
        PreparedDistance::mahalanobis(matrix_view(&vi, &matrix_shape, &matrix_strides)).unwrap(),
    );
    for strides in [[3, 1], [7, 2]] {
        // Nonlogical padding deliberately contains NaN and must not be scanned.
        let mut a = vec![S::NAN; 33 * strides[0]];
        let mut b = vec![S::ZERO; 35 * strides[0]];
        for row in 0..33 {
            for (axis, x) in [1.0, 0.0, (row % 2) as f64].into_iter().enumerate() {
                a[row * strides[0] + axis * strides[1]] = S::try_from_f64_lossy(x).unwrap();
            }
        }
        for row in 0..35 {
            for (axis, x) in [0.0, 1.0, (row % 2) as f64].into_iter().enumerate() {
                b[row * strides[0] + axis * strides[1]] = S::try_from_f64_lossy(x).unwrap();
            }
        }
        let left = points(&a, &shape, &strides);
        let right = points(&b, &target_shape, &strides);
        let sentinel = S::try_from_f64_lossy(42.0).unwrap();
        for metric in &metrics {
            let mut output = vec![sentinel; 33 * 35 + 2];
            let mut scratch = distance_workspace(output.len());
            metric
                .pairwise_into(
                    &left,
                    &right,
                    DistanceOptions::new(),
                    &mut output,
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
            for row in 0..33 {
                let a = [
                    left.coordinate(row, 0).unwrap(),
                    left.coordinate(row, 1).unwrap(),
                    left.coordinate(row, 2).unwrap(),
                ];
                for column in 0..35 {
                    let b = [
                        right.coordinate(column, 0).unwrap(),
                        right.coordinate(column, 1).unwrap(),
                        right.coordinate(column, 2).unwrap(),
                    ];
                    close(output[row * 35 + column], metric.distance(&a, &b).unwrap());
                }
            }
            assert_eq!(&output[33 * 35..], &[sentinel; 2]);
            let len = condensed_len(33).unwrap();
            metric
                .condensed_into(
                    &left,
                    DistanceOptions::new(),
                    &mut output,
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
            for row in 0..33 {
                for column in row + 1..33 {
                    close(
                        output[condensed_index(33, row, column).unwrap()],
                        metric.between(&left, row, &left, column).unwrap(),
                    );
                }
            }
            assert!(output[..len].iter().all(|x| x.is_finite()));
        }
    }
}

#[test]
fn bulk_metrics_both_widths_and_layouts() {
    bulk_metrics::<f32>();
    bulk_metrics::<f64>();
}

#[test]
fn independent_block_boundaries_and_broadcasts() {
    for dimension in [0, 1, 2, 3, 4, 16] {
        for n in [0, 1, 31, 32, 33, 65] {
            let data: Vec<_> = (0..n)
                .flat_map(|i| (1..=dimension).map(move |axis| (i * axis) as f64))
                .collect();
            let shape = [n, dimension];
            let strides = [dimension, 1];
            let pts = points(&data, &shape, &strides);
            for (metric, factor) in [
                (
                    DistanceMetric::Euclidean,
                    (1..=dimension)
                        .map(|axis| (axis * axis) as f64)
                        .sum::<f64>()
                        .sqrt(),
                ),
                (
                    DistanceMetric::CityBlock,
                    (1..=dimension).map(|axis| axis as f64).sum(),
                ),
            ] {
                let prepared = PreparedDistance::new(metric).unwrap();
                let matrix = prepared.pairwise(&pts, &pts).unwrap().as_slice().to_vec();
                let mut condensed = vec![0.0; condensed_len(n).unwrap()];
                let mut scratch = distance_workspace(condensed.len());
                prepared
                    .condensed_into(
                        &pts,
                        DistanceOptions::new(),
                        &mut condensed,
                        &mut scratch,
                        scioxide_core::ExecutionContext::sequential(),
                    )
                    .unwrap();
                for i in 0..n {
                    for j in 0..n {
                        close(matrix[i * n + j], i.abs_diff(j) as f64 * factor);
                        if i < j {
                            close(
                                condensed[condensed_index(n, i, j).unwrap()],
                                i.abs_diff(j) as f64 * factor,
                            );
                        }
                    }
                }
            }
        }
    }
    let shape = [33, 3];
    for strides in [[0, 1], [1, 0], [0, 0]] {
        let data: Vec<_> = (0..35).map(|i| i as f64).collect();
        let layout = StridedLayout::new_broadcast(&shape, &strides).unwrap();
        let pts = PointSet::new(NdView::new(&data, layout).unwrap()).unwrap();
        let metric = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
        let actual = metric.pairwise(&pts, &pts).unwrap().as_slice().to_vec();
        for i in 0..33 {
            for j in 0..33 {
                let expected = (0..3)
                    .map(|axis| {
                        let d = data[i * strides[0] + axis * strides[1]]
                            - data[j * strides[0] + axis * strides[1]];
                        d * d
                    })
                    .sum::<f64>()
                    .sqrt();
                close(actual[i * 33 + j], expected);
            }
        }
    }
}

#[test]
fn scaled_fallbacks_in_bulk_and_hausdorff() {
    let metric = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    let identity_data = [1.0_f64, 0.0, 0.0, 1.0];
    let identity_shape = [2, 2];
    let strides = [2, 1];
    let mahalanobis =
        PreparedDistance::mahalanobis(matrix_view(&identity_data, &identity_shape, &strides))
            .unwrap();
    for scale in [1e200_f64, 1e-200, 1e-160] {
        let data = [0.0, 0.0, 3.0 * scale, 4.0 * scale];
        let pts = points(&data, &identity_shape, &strides);
        let origin_shape = [1, 2];
        let origin = points(&data[..2], &origin_shape, &strides);
        for metric in [&metric, &mahalanobis] {
            let mut output = [-1.0; 4];
            let mut scratch = distance_workspace(4);
            metric
                .pairwise_into(
                    &pts,
                    &origin,
                    DistanceOptions::new(),
                    &mut output,
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
            assert!((output[1] / (5.0 * scale) - 1.0).abs() < 1e-14);
            metric
                .condensed_into(
                    &pts,
                    DistanceOptions::new(),
                    &mut output,
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
            assert!((output[0] / (5.0 * scale) - 1.0).abs() < 1e-14);
        }
        let result = directed_hausdorff(&pts, &origin).unwrap();
        assert!((result.distance / (5.0 * scale) - 1.0).abs() < 1e-14);
        assert_eq!((result.source, result.target), (1, 0));
    }
}

#[test]
fn invalid_inputs_and_late_failures_are_atomic() {
    let shape = [3, 1];
    let strides = [1, 1];
    let euclidean = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let data = [0.0, 1.0, bad];
        let mut output = [42.0; 9];
        let mut scratch = distance_workspace(9);
        let result = PointSet::new(
            NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
        )
        .and_then(|pts| {
            euclidean.pairwise_into(
                &pts,
                &pts,
                DistanceOptions::new(),
                &mut output,
                &mut scratch,
                scioxide_core::ExecutionContext::sequential(),
            )
        });
        assert!(matches!(
            result.unwrap_err().view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
        assert_eq!(output, [42.0; 9]);
    }
    for (data, metric) in [
        ([0.0, f64::MAX, -f64::MAX], DistanceMetric::Euclidean),
        ([1.0, 1.0, 0.0], DistanceMetric::Cosine),
        ([0.0, 1.0, 2.0], DistanceMetric::Dice),
    ] {
        let pts = points(&data, &shape, &strides);
        let metric = PreparedDistance::new(metric).unwrap();
        let mut output = [42.0; 9];
        let mut scratch = distance_workspace(9);
        assert!(
            metric
                .pairwise_into(
                    &pts,
                    &pts,
                    DistanceOptions::new(),
                    &mut output,
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential()
                )
                .is_err()
        );
        assert_eq!(output, [42.0; 9]);
        assert!(
            metric
                .condensed_into(
                    &pts,
                    DistanceOptions::new(),
                    &mut output,
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential()
                )
                .is_err()
        );
        assert_eq!(output, [42.0; 9]);
    }
    // Finite f32 inputs can have a Euclidean norm outside the f32 result range.
    let data = [0.0_f32, f32::MAX, -f32::MAX];
    let pts = points(&data, &shape, &strides);
    let mut output = [42.0_f32; 9];
    let mut scratch = distance_workspace(9);
    assert!(
        euclidean
            .pairwise_into(
                &pts,
                &pts,
                DistanceOptions::new(),
                &mut output,
                &mut scratch,
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(output, [42.0; 9]);
    assert!(
        euclidean
            .condensed_into(
                &pts,
                DistanceOptions::new(),
                &mut output,
                &mut scratch,
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(output, [42.0; 9]);
}

fn exhaustive_hausdorff<S: RealScalar>(
    source: &PointSet<'_, S>,
    target: &PointSet<'_, S>,
) -> HausdorffResult<S> {
    let mut best = HausdorffResult {
        distance: S::ZERO,
        source: 0,
        target: 0,
    };
    for i in 0..source.len() {
        let mut nearest = S::MAX;
        let mut witness = 0;
        for j in 0..target.len() {
            let distance = S::try_from_f64_lossy(
                (0..source.dimension())
                    .map(|axis| {
                        let d = source.coordinate(i, axis).unwrap().into_f64()
                            - target.coordinate(j, axis).unwrap().into_f64();
                        d * d
                    })
                    .sum::<f64>()
                    .sqrt(),
            )
            .unwrap();
            if distance < nearest {
                nearest = distance;
                witness = j;
            }
        }
        if i == 0 || nearest > best.distance {
            best = HausdorffResult {
                distance: nearest,
                source: i,
                target: witness,
            };
        }
    }
    best
}

fn hausdorff_ties<S: RealScalar>() {
    let strides = [2, 1];
    for n in [1, 2, 31, 32, 33, 65] {
        let shape = [n, 2];
        let target_shape = [n + 2, 2];
        let data: Vec<S> = (0..n)
            .flat_map(|i| [((i * 17) % 11) as f64, ((i * 7) % 13) as f64])
            .map(|x| S::try_from_f64_lossy(x).unwrap())
            .collect();
        let target: Vec<S> = (0..n + 2)
            .flat_map(|i| [((i * 5) % 7) as f64, ((i * 19) % 17) as f64])
            .map(|x| S::try_from_f64_lossy(x).unwrap())
            .collect();
        let a = points(&data, &shape, &strides);
        let b = points(&target, &target_shape, &strides);
        let expected = exhaustive_hausdorff(&a, &b);
        for _ in 0..3 {
            let actual = directed_hausdorff(&a, &b).unwrap();
            close(actual.distance, expected.distance);
            assert_eq!(
                (actual.source, actual.target),
                (expected.source, expected.target)
            );
        }
        let reversed: Vec<_> = target
            .rchunks_exact(2)
            .flat_map(|row| row.iter().copied())
            .collect();
        let b = points(&reversed, &target_shape, &strides);
        let expected = exhaustive_hausdorff(&a, &b);
        let actual = directed_hausdorff(&a, &b).unwrap();
        close(actual.distance, expected.distance);
        assert_eq!(
            (actual.source, actual.target),
            (expected.source, expected.target)
        );
        let identical = directed_hausdorff(&a, &a).unwrap();
        assert_eq!(
            identical,
            HausdorffResult {
                distance: S::ZERO,
                source: 0,
                target: 0
            }
        );
    }
    let a: Vec<S> = [0.0, 0.0, 0.0, 0.0, 2.0, 0.0]
        .map(|x| S::try_from_f64_lossy(x).unwrap())
        .into();
    let b: Vec<S> = [-1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        .map(|x| S::try_from_f64_lossy(x).unwrap())
        .into();
    let shape = [3, 2];
    assert_eq!(
        directed_hausdorff(&points(&a, &shape, &strides), &points(&b, &shape, &strides)).unwrap(),
        HausdorffResult {
            distance: S::ONE,
            source: 0,
            target: 0
        }
    );
}

#[test]
fn hausdorff_shuffle_preserves_original_index_ties() {
    hausdorff_ties::<f32>();
    hausdorff_ties::<f64>();
}

#[test]
fn empty_dimensions_and_capacity_errors() {
    let data = [0.0_f64, 1.0, 2.0];
    let shape = [3, 1];
    let stride = [1, 1];
    let pts = points(&data, &shape, &stride);
    let metric = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    let mut output = [42.0; 9];
    let mut scratch = distance_workspace(9);
    assert!(
        metric
            .pairwise_into(
                &pts,
                &pts,
                DistanceOptions::new(),
                &mut output[..8],
                &mut scratch,
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
    );
    assert!(
        metric
            .pairwise_into(
                &pts,
                &pts,
                DistanceOptions::new(),
                &mut output,
                &mut distance_workspace(8),
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
    );
    assert!(
        metric
            .condensed_into(
                &pts,
                DistanceOptions::new(),
                &mut output[..2],
                &mut scratch,
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
    );
    assert!(
        metric
            .condensed_into(
                &pts,
                DistanceOptions::new(),
                &mut output,
                &mut distance_workspace(2),
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(output, [42.0; 9]);
    let zero_shape = [3, 0];
    let zero_stride = [usize::MAX, usize::MAX];
    let zero = points(&[], &zero_shape, &zero_stride);
    metric
        .pairwise_into(
            &zero,
            &zero,
            DistanceOptions::new(),
            &mut output,
            &mut scratch,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output, [0.0; 9]);
    assert_eq!(
        directed_hausdorff(&zero, &zero).unwrap(),
        HausdorffResult {
            distance: 0.0,
            source: 0,
            target: 0
        }
    );
    output.fill(42.0);
    assert!(
        metric
            .pairwise_into(
                &pts,
                &zero,
                DistanceOptions::new(),
                &mut output,
                &mut scratch,
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(output, [42.0; 9]);
    assert!(directed_hausdorff(&pts, &zero).is_err());
    let empty_shape = [0, 1];
    let empty = points(&[], &empty_shape, &stride);
    assert!(
        metric
            .pairwise(&pts, &empty)
            .unwrap()
            .as_slice()
            .to_vec()
            .is_empty()
    );
    assert!(directed_hausdorff(&pts, &empty).is_err());
}

fn distance_workspace<S: scioxide_core::RealScalar>(len: usize) -> DistanceWorkspace<S> {
    DistanceWorkspace::workspace(DistanceWorkspaceRequest::pairwise(
        1,
        len,
        DistanceOptions::new(),
    ))
    .unwrap()
}

fn matrix_view<'a, S: scioxide_core::RealScalar>(
    data: &'a [S],
    shape: &[usize; 2],
    strides: &[usize; 2],
) -> scioxide_linalg::MatrixView<'a, S> {
    scioxide_linalg::MatrixView::from_slice_with_strides(
        shape[0], shape[1], data, strides[0], strides[1],
    )
    .unwrap()
}

#[test]
fn normalized_distances_preserve_extreme_mass_ratios() {
    let js = PreparedDistance::new(DistanceMetric::JensenShannon).unwrap();
    let actual = js.distance(&[1e308_f64, 1e-308], &[1e308, 0.0]).unwrap();
    let expected = 1e-308 * (0.5 * 2.0_f64.ln()).sqrt();
    assert!((actual / expected - 1.0).abs() < 1e-13);
    let metric = PreparedDistance::new(DistanceMetric::Minkowski(0.01)).unwrap();
    let actual = metric.distance(&[1e308_f64, 1e-308], &[0.0; 2]).unwrap();
    let expected = 1e308 * (1.0 + 10.0_f64.powf(-6.16)).powi(100);
    assert!((actual / expected - 1.0).abs() < 1e-13);
    let error = PreparedDistance::new(DistanceMetric::Euclidean)
        .unwrap()
        .distance(&[f32::MAX], &[-f32::MAX])
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::NumericalInstability
    );
}
