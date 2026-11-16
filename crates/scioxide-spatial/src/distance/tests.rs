//! Count actual staging work between cancellation observations, without timing.
#![allow(clippy::unwrap_used)]

use super::*;
use alloc::{vec, vec::Vec};
use core::sync::atomic::{AtomicUsize, Ordering};
use scioxide_core::{
    Cancellation, CancellationToken, ExecutionContext, SciOxideErrorKind, StridedLayout,
};

#[derive(Default)]
struct CancelAfterObservation {
    token: CancellationToken,
    polls: AtomicUsize,
    arm_on: usize,
}

impl Cancellation for CancelAfterObservation {
    fn is_cancelled(&self) -> bool {
        let cancelled = self.token.is_cancelled();
        if self.polls.fetch_add(1, Ordering::Relaxed) + 1 >= self.arm_on {
            self.token.cancel();
        }
        cancelled
    }
}

fn poll(cancellation: &CancelAfterObservation) -> DistancePoll<'_> {
    DistancePoll::new(ExecutionContext::sequential().with_cancellation(cancellation))
}

fn assert_stopped(error: SciOxideError, cancellation: &CancelAfterObservation, work: usize) {
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    assert_eq!(cancellation.polls.load(Ordering::Relaxed), 2);
    assert!(
        work > 0 && work <= DISTANCE_POLL_INTERVAL,
        "actual work: {work}"
    );
}

#[test]
fn pairwise_polling_counts_pairs_across_wide_and_short_rows_and_partitions() {
    for (rows, columns, offset) in [(3, 1025, 0), (1025, 1, 0), (67, 35, 17)] {
        let left_data = vec![0.0; rows];
        let right_data = vec![1.0; columns];
        let left_shape = [rows, 1];
        let right_shape = [columns, 1];
        let strides = [1, 1];
        let left = PointSet::new(
            NdView::new(
                &left_data,
                StridedLayout::new(&left_shape, &strides).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let right = PointSet::new(
            NdView::new(
                &right_data,
                StridedLayout::new(&right_shape, &strides).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let mut output = vec![-1.0; rows * columns - 2 * offset];
        let work = AtomicUsize::new(0);
        let cancellation = CancelAfterObservation::default();
        let error = fill_pairwise::<f64, true>(
            &left,
            &right,
            offset,
            &mut output,
            |a, points, first, values| {
                blocks::fill(DistanceMetric::Euclidean, a, points, first, values)?;
                work.fetch_add(values.len(), Ordering::Relaxed);
                Ok(())
            },
            &mut poll(&cancellation),
        )
        .unwrap_err();
        let changed = output.iter().filter(|&&x| x != -1.0).count();
        assert_eq!(changed, work.load(Ordering::Relaxed));
        assert_stopped(error, &cancellation, changed);
    }
}

#[test]
fn condensed_polling_carries_work_across_rows_and_partial_partitions() {
    for (n, offset) in [(25, 0), (65, 17), (1025, 7)] {
        let data: Vec<_> = (0..n).map(|i| i as f64).collect();
        let shape = [n, 1];
        let strides = [1, 1];
        let points = PointSet::new(
            NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
        )
        .unwrap();
        let mut output = vec![-1.0; condensed_len(n).unwrap() - 2 * offset];
        let work = AtomicUsize::new(0);
        let cancellation = CancelAfterObservation::default();
        let error = fill_condensed::<f64, true>(
            &points,
            offset,
            &mut output,
            |a, points, first, values| {
                blocks::fill(DistanceMetric::CityBlock, a, points, first, values)?;
                work.fetch_add(values.len(), Ordering::Relaxed);
                Ok(())
            },
            &mut poll(&cancellation),
        )
        .unwrap_err();
        let changed = output.iter().filter(|&&x| x != -1.0).count();
        assert_eq!(changed, work.load(Ordering::Relaxed));
        assert_stopped(error, &cancellation, changed);
    }
}

#[test]
fn both_conversion_kernels_copy_at_most_256_entries_between_polls() {
    let n = 35;
    let total = condensed_len(n).unwrap();
    let upper = vec![1.0; total];
    let square: Vec<_> = (0..n * n)
        .map(|i| if i / n == i % n { 0.0 } else { 1.0 })
        .collect();
    let matrix = MatrixView::from_slice(n, n, &square).unwrap();
    for offset in [0, 17, 31] {
        for expand in [false, true] {
            let mut output = vec![
                -1.0;
                if expand {
                    n * n - offset
                } else {
                    total - offset
                }
            ];
            let cancellation = CancelAfterObservation::default();
            let result = if expand {
                fill_square(n, &upper, offset, &mut output, &mut poll(&cancellation))
            } else {
                fill_upper(matrix, total, offset, &mut output, &mut poll(&cancellation))
            };
            let changed = output.iter().filter(|&&x| x != -1.0).count();
            assert_eq!(changed, DISTANCE_POLL_INTERVAL);
            assert_stopped(result.unwrap_err(), &cancellation, changed);
        }
    }
}

#[test]
fn conversion_validation_observes_cancellation_before_the_next_entry_block() {
    let n = 35;
    let mut values = vec![1.0; condensed_len(n).unwrap()];
    // An invalid entry just beyond the work bound must never be inspected.
    values[DISTANCE_POLL_INTERVAL] = f64::NAN;
    let cancellation = CancelAfterObservation::default();
    let error = validate_condensed_polled(n, &values, &mut poll(&cancellation)).unwrap_err();
    assert_stopped(error, &cancellation, DISTANCE_POLL_INTERVAL);

    for bad in [f64::NAN, -1.0] {
        let mut square = vec![0.0; n * n];
        // With 35-entry rows, seven rows fit before the next observation.
        square[7 * n] = bad;
        let matrix = MatrixView::from_slice(n, n, &square).unwrap();
        let cancellation = CancelAfterObservation::default();
        let error = validate_matrix(matrix, 0.0, &mut poll(&cancellation)).unwrap_err();
        assert_stopped(error, &cancellation, 7 * n);
    }

    // Let the finite scan finish, then cancel during the domain scan. Its
    // next row contains a negative entry, which must remain unobserved.
    let mut square = vec![0.0; n * n];
    square[7 * n + 7] = -1.0;
    let matrix = MatrixView::from_slice(n, n, &square).unwrap();
    let cancellation = CancelAfterObservation {
        arm_on: 6,
        ..Default::default()
    };
    let error = validate_matrix(matrix, 0.0, &mut poll(&cancellation)).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    assert_eq!(cancellation.polls.load(Ordering::Relaxed), 7);
}

#[test]
fn genuine_zero_sums_are_ordinary_for_both_scalar_widths() {
    for metric in [DistanceMetric::Euclidean, DistanceMetric::SquaredEuclidean] {
        let mut output = [f64::NAN; 3];
        let status = blocks::store(metric, &[0.0, -0.0, f64::MIN_POSITIVE], &mut output);
        assert!(status.ordinary && status.zero);
        assert_eq!(output[0].to_bits(), 0.0_f64.to_bits());
        let mut output = [f32::NAN; 2];
        assert!(blocks::store(metric, &[0.0, -0.0], &mut output).ordinary);
        assert_eq!(output[0].to_bits(), 0.0_f32.to_bits());
        assert!(!blocks::store(metric, &[f64::from_bits(1), 0.0], &mut output).ordinary);
        assert_eq!(
            blocks::raw(metric, &[0.0_f64, -0.0, 1.0], &[-0.0, 0.0, 1.0]).to_bits(),
            0
        );
    }
}

#[test]
fn zero_certificate_distinguishes_duplicates_from_underflow() {
    for dimension in [0, 1, 3, 7] {
        for row_stride in [dimension, dimension + 2] {
            let shape = [3, dimension];
            let strides = [row_stride, 1];
            let mut data = vec![0.0_f64; 3 * row_stride + dimension];
            if dimension != 0 {
                data[row_stride] = -0.0;
            }
            let points = PointSet::new(
                NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
            )
            .unwrap();
            assert!(blocks::exact_zeros(
                points.row::<true>(0),
                &points,
                0,
                &[0.0; 3]
            ));
            if dimension == 0 {
                continue;
            }
            data[2 * row_stride] = 1e-200;
            let points = PointSet::new(
                NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
            )
            .unwrap();
            assert!(!blocks::exact_zeros(
                points.row::<true>(0),
                &points,
                0,
                &[0.0; 3]
            ));
            let mut output = [-1.0; 3];
            blocks::fill(
                DistanceMetric::Euclidean,
                points.row::<true>(0),
                &points,
                0,
                &mut output,
            )
            .unwrap();
            assert_eq!(output[2].to_bits(), 1e-200_f64.to_bits());
        }
    }
}
