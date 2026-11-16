//! Mask semantics, owned and `_into` axis forms, shapes and allocation.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::*;

fn assert_shape(error: &SciOxideError, expected: &[usize], actual: &[usize]) {
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    assert!(
        matches!(error.view(), SciOxideErrorView::DimensionMismatch {
         expected: wanted, actual: found,
    } if wanted == expected && found == actual)
    );
}

fn semantics<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let mask = S::NEG_INFINITY;
    let one = S::ONE;
    let zero = S::ZERO;
    for values in [&[][..], &[mask][..], &[mask, mask][..]] {
        assert_eq!(logsumexp(values).unwrap(), mask);
        assert!(logsumexp_value(values).unwrap().log_magnitude().is_none());
        assert_eq!(
            softmax(values).unwrap_err().kind(),
            SciOxideErrorKind::DomainError
        );
        assert_eq!(
            log_softmax(values).unwrap_err().kind(),
            SciOxideErrorKind::DomainError
        );
        let weights = [one, -one];
        assert_eq!(
            weighted_logsumexp(values, &weights[..values.len()])
                .unwrap()
                .sign(),
            zero
        );
        let values: Vec<_> = values.iter().map(|&re| Complex::new(re, zero)).collect();
        let log = logsumexp(&values).unwrap();
        assert_eq!(log.re, mask);
        assert_eq!(log.im.into_f64().to_bits(), 0);
        assert!(logsumexp_value(&values).unwrap().log_magnitude().is_none());
    }
    assert_eq!(logaddexp(mask, mask).unwrap(), mask);
    assert_eq!(logaddexp(mask, one).unwrap(), one);
    assert_eq!(logaddexp(one, mask).unwrap(), one);
    assert_eq!(logsumexp(&[mask, one, mask]).unwrap(), one);
    let values = [mask, zero, mask];
    let probabilities = softmax(&values).unwrap();
    assert_eq!(probabilities, [zero, one, zero]);
    assert_eq!(probabilities[0].into_f64().to_bits(), 0);
    assert_eq!(log_softmax(&values).unwrap(), [mask, zero, mask]);
    assert_eq!(
        weighted_logsumexp(&[mask, zero], &[-one, -one])
            .unwrap()
            .sign(),
        -one
    );
    assert!(
        weighted_logsumexp(&[zero, zero, mask], &[one, -one, one])
            .unwrap()
            .log_magnitude()
            .is_none()
    );
    let weights = [
        Complex::new(one, zero),
        Complex::new(-one, zero),
        Complex::new(one, one),
    ];
    let z = weighted_logsumexp(&[zero, zero, mask], &weights).unwrap();
    assert_eq!(z.scaled_phase().round().unwrap(), Complex::new(zero, zero));
    let z = weighted_logsumexp(&[mask, zero], &[weights[0], Complex::new(zero, one)]).unwrap();
    assert_eq!(z.scaled_phase().round().unwrap(), Complex::new(zero, one));
    for bad in [S::NAN, S::INFINITY] {
        assert!(logsumexp(&[mask, bad]).is_err());
        assert!(weighted_logsumexp(&[bad], &[zero]).is_err());
        assert!(weighted_logsumexp(&[bad], &[Complex::new(zero, zero)]).is_err());
    }
    for bad in [S::NAN, S::INFINITY, mask] {
        assert!(weighted_logsumexp(&[mask], &[bad]).is_err());
        assert!(weighted_logsumexp(&[mask], &[Complex::new(zero, bad)]).is_err());
        assert!(logsumexp(&[Complex::new(mask, bad)]).is_err());
        assert!(LogValue::from_log(one, bad).is_err());
        assert!(LogValue::from_value(bad).is_err());
    }
    // Failure occurs after a valid first lane; publication must still be atomic.
    let layout = StridedLayout::new(&[2, 3], &[3, 1]).unwrap();
    for last in [[mask; 3], [zero, S::NAN, mask], [zero, S::INFINITY, mask]] {
        let input = [zero, one, mask, last[0], last[1], last[2]];
        for logarithmic in [false, true] {
            let mut out = [S::NEG_ZERO; 6];
            let mut scratch = [zero; 6];
            let output = NdViewMut::new(&mut out, layout).unwrap();
            let input = NdView::new(&input, layout).unwrap();
            let policy = ExecutionPolicy::Sequential;
            let cx = ExecutionContext::sequential();
            let result = if logarithmic {
                log_softmax_axis_into(input, 1, policy, output, &mut scratch, cx)
            } else {
                softmax_axis_into(input, 1, policy, output, &mut scratch, cx)
            };
            assert!(result.is_err());
            assert!(
                out.iter()
                    .all(|x| x.into_f64().to_bits() == (-0.0_f64).to_bits())
            );
        }
    }
}

#[test]
fn semantic_table_both_widths() {
    semantics::<f32>();
    semantics::<f64>();
}

fn axis_forms<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let shape = [2, 3];
    let layout = StridedLayout::new(&shape, &[1, 2]).unwrap();
    let contiguous = StridedLayout::new(&shape, &[3, 1]).unwrap();
    let reduced = StridedLayout::new(&[2], &[1]).unwrap();
    let values = [
        S::ZERO,
        S::ONE,
        S::NEG_INFINITY,
        S::ZERO,
        S::ONE,
        S::NEG_INFINITY,
    ];
    let input = NdView::new(&values, layout).unwrap();
    let weight_layout = StridedLayout::new_broadcast(&shape, &[0, 0]).unwrap();
    let policy = ExecutionPolicy::Sequential;
    let cx = ExecutionContext::sequential();
    let expected = softmax_axis(input, 1).unwrap();
    let expected_log = log_softmax_axis(input, 1).unwrap();
    {
        let mut out = [S::ZERO; 6];
        let mut scratch = out;
        let output = NdViewMut::new(&mut out, contiguous).unwrap();
        softmax_axis_into(input, 1, policy, output, &mut scratch, cx).unwrap();
        assert_eq!(out.as_slice(), expected);
        let output = NdViewMut::new(&mut out, contiguous).unwrap();
        log_softmax_axis_into(input, 1, policy, output, &mut scratch, cx).unwrap();
        assert_eq!(out.as_slice(), expected_log);
        let old = out;
        let output = NdViewMut::new(&mut out, contiguous).unwrap();
        assert!(softmax_axis_into(input, 1, policy, output, &mut scratch[..5], cx).is_err());
        assert_eq!(out, old);
        let mut logs = [S::ZERO; 2];
        let output = NdViewMut::new(&mut logs, reduced).unwrap();
        logsumexp_axis_into(input, 1, policy, output, &mut scratch[..2], cx).unwrap();
        for (i, &actual) in logs.iter().enumerate() {
            assert_eq!(
                actual,
                logsumexp(&[values[i], values[i + 2], values[i + 4]]).unwrap()
            );
        }
        let weights = [S::ONE];
        let weights = NdView::new(&weights, weight_layout).unwrap();
        let mut result = [LogValue::<S, S>::ZERO; 2];
        let mut work = result;
        let output = NdViewMut::new(&mut result, reduced).unwrap();
        weighted_logsumexp_axis_into(input, weights, 1, policy, output, &mut work, cx).unwrap();
        for (value, expected) in result.iter().zip(logs) {
            assert_eq!(value.log_magnitude(), Some(expected));
        }
        let values = values.map(|re| Complex::new(re, S::ZERO));
        let mut out = [Complex::new(S::ZERO, S::ZERO); 2];
        let mut work = out;
        let input = NdView::new(&values, layout).unwrap();
        let output = NdViewMut::new(&mut out, reduced).unwrap();
        logsumexp_axis_into(input, 1, policy, output, &mut work, cx).unwrap();
        assert_eq!(out.map(|v| v.re), logs);
    };
}

#[test]
fn owned_strided_and_into_forms_agree_on_masked_lanes() {
    axis_forms::<f32>();
    axis_forms::<f64>();
}

#[test]
fn scratch_lengths_cover_singleton_and_zero_extents_without_allocation() {
    for (shape, count) in [([1, 1], 1), ([0, 3], 0), ([2, 0], 2)] {
        let layout = StridedLayout::new(&shape, &[0, 0]).unwrap();
        {
            assert_eq!(layout.axis(1).unwrap().len(), count);
            assert_eq!(layout.shape().len(), shape[0] * shape[1]);
        };
    }
}

#[test]
fn empty_batches_are_distinct_from_empty_lanes() {
    for (shape, succeeds) in [([0, 3], true), ([2, 0], false)] {
        let layout = StridedLayout::new(&shape, &[0, 0]).unwrap();
        let input = NdView::<f64>::new(&[], layout).unwrap();
        for result in [softmax_axis(input, 1), log_softmax_axis(input, 1)] {
            match result {
                Ok(values) => assert!(succeeds && values.is_empty()),
                Err(error) => {
                    assert!(!succeeds);
                    assert_eq!(error.kind(), SciOxideErrorKind::DomainError);
                }
            }
        }
        let count = shape[0];
        assert_eq!(
            logsumexp_axis(input, 1).unwrap(),
            vec![f64::NEG_INFINITY; count]
        );
        let reduced = StridedLayout::new(&shape[..1], &[1]).unwrap();
        let mut out = [1.0; 2];
        let mut scratch = [0.0; 2];
        logsumexp_axis_into(
            input,
            1,
            ExecutionPolicy::Sequential,
            NdViewMut::new(&mut out[..count], reduced).unwrap(),
            &mut scratch[..count],
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert!(out[..count].iter().all(|&v| v == f64::NEG_INFINITY));
        let input = NdView::<Complex64>::new(&[], layout).unwrap();
        assert!(
            logsumexp_axis(input, 1)
                .unwrap()
                .iter()
                .all(|&v| v == Complex64::new(f64::NEG_INFINITY, 0.0))
        );
        let mut out = [Complex64::new(1.0, 1.0); 2];
        let mut scratch = out;
        logsumexp_axis_into(
            input,
            1,
            ExecutionPolicy::Sequential,
            NdViewMut::new(&mut out[..count], reduced).unwrap(),
            &mut scratch[..count],
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert!(
            out[..count]
                .iter()
                .all(|&v| v == Complex64::new(f64::NEG_INFINITY, 0.0))
        );
    }
}

const SHAPE: [usize; 3] = [2, 3, 4];
// Strides, storage length and whether the storage has masked entries: a
// contiguous, a padded and a broadcast layout of `SHAPE`. Broadcast lanes repeat
// one element, so its storage has no mask; elsewhere at most one entry per
// lane is masked.
const LAYOUTS: [([usize; 3], usize, bool); 3] = [
    ([12, 4, 1], 24, true),
    ([20, 5, 1], 34, true),
    ([0, 4, 1], 12, false),
];

fn scalar<S: RealScalar>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}

fn logits<S: RealScalar>(len: usize, masked: bool) -> Vec<S> {
    (0..len)
        .map(|i| {
            if masked && i % 13 == 5 {
                S::NEG_INFINITY
            } else {
                scalar(((i * 7) % 11) as f64 * 0.375 - 1.5)
            }
        })
        .collect()
}

fn bits<S: RealScalar>(values: &[S]) -> Vec<u64> {
    values.iter().map(|v| v.into_f64().to_bits()).collect()
}

fn complex_bits<S: RealScalar>(values: &[Complex<S>]) -> Vec<[u64; 2]> {
    values
        .iter()
        .map(|v| [v.re.into_f64().to_bits(), v.im.into_f64().to_bits()])
        .collect()
}

fn lane<T: Copy>(view: NdView<'_, '_, T>, axis: usize, index: usize) -> Vec<T> {
    let lane = view.layout().axis(axis).unwrap().lane(index).unwrap();
    lane.offsets()
        .map(|offset| view.storage()[offset])
        .collect()
}

fn reduced_shape(extents: &[usize], axis: usize) -> Vec<usize> {
    let mut reduced = extents.to_vec();
    reduced.remove(axis);
    reduced
}

fn normalize_into<S: RealScalar>(
    input: NdView<'_, '_, S>,
    axis: usize,
    logarithmic: bool,
    (policy, cx): (ExecutionPolicy, ExecutionContext<'_>),
) -> Vec<S> {
    let n = input.layout().shape().len();
    let mut output = vec![S::NAN; n];
    let mut scratch = vec![S::ZERO; n];
    let target = NdViewMut::row_major(&mut output, input.layout().shape().extents()).unwrap();
    if logarithmic {
        log_softmax_axis_into(input, axis, policy, target, &mut scratch, cx).unwrap();
    } else {
        softmax_axis_into(input, axis, policy, target, &mut scratch, cx).unwrap();
    }
    output
}

fn logsumexp_into<T: NumericScalar>(
    input: NdView<'_, '_, T>,
    axis: usize,
    (policy, cx): (ExecutionPolicy, ExecutionContext<'_>),
) -> Vec<T> {
    let lanes = input.layout().axis(axis).unwrap().len();
    let reduced = reduced_shape(input.layout().shape().extents(), axis);
    let mut output = vec![T::ONE; lanes];
    let mut scratch = vec![T::ZERO; lanes];
    let target = NdViewMut::row_major(&mut output, &reduced).unwrap();
    logsumexp_axis_into(input, axis, policy, target, &mut scratch, cx).unwrap();
    output
}

fn weighted_into<T: NumericScalar>(
    input: NdView<'_, '_, T::Real>,
    weights: NdView<'_, '_, T>,
    axis: usize,
    (policy, cx): (ExecutionPolicy, ExecutionContext<'_>),
) -> Vec<LogValue<T, T::Real>> {
    let lanes = input.layout().axis(axis).unwrap().len();
    let reduced = reduced_shape(input.layout().shape().extents(), axis);
    let mut output = vec![LogValue::ZERO; lanes];
    let mut scratch = output.clone();
    let target = NdViewMut::row_major(&mut output, &reduced).unwrap();
    weighted_logsumexp_axis_into(input, weights, axis, policy, target, &mut scratch, cx).unwrap();
    output
}

fn assert_same_log_values<T: NumericScalar>(
    actual: &[LogValue<T, T::Real>],
    expected: &[LogValue<T, T::Real>],
) {
    assert_eq!(actual, expected);
    let log = |values: &[LogValue<T, T::Real>]| -> Vec<Option<u64>> {
        values
            .iter()
            .map(|value| value.log_magnitude().map(|log| log.into_f64().to_bits()))
            .collect()
    };
    assert_eq!(log(actual), log(expected));
}

fn owned_and_into_forms_agree<S: RealScalar>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let sequential = (
        ExecutionPolicy::Sequential,
        ExecutionContext::sequential_native(),
    );
    #[cfg(feature = "std")]
    let pool = ThreadExecutor::new(core::num::NonZeroUsize::new(4).unwrap());
    #[cfg(feature = "std")]
    let runs = [
        sequential,
        (
            ExecutionPolicy::Parallel(core::num::NonZeroUsize::new(4).unwrap()),
            ExecutionContext::native(&pool),
        ),
    ];
    #[cfg(not(feature = "std"))]
    let runs = [sequential];
    let weights: Vec<S> = (0..24)
        .map(|i| scalar(((i * 5) % 7) as f64 * 0.25 - 0.75))
        .collect();
    let complex_weights: Vec<Complex<S>> = weights
        .iter()
        .map(|&w| Complex::new(w, w * scalar(0.5) + S::ONE))
        .collect();
    let weights = NdView::row_major(&weights, &SHAPE).unwrap();
    let complex_weights = NdView::row_major(&complex_weights, &SHAPE).unwrap();
    for (strides, storage, masked) in LAYOUTS {
        let data = logits::<S>(storage, masked);
        let complex: Vec<Complex<S>> = data
            .iter()
            .enumerate()
            .map(|(i, &re)| Complex::new(re, scalar(0.125 * i as f64)))
            .collect();
        let layout = StridedLayout::new_broadcast(&SHAPE, &strides).unwrap();
        let input = NdView::new(&data, layout).unwrap();
        let complex = NdView::new(&complex, layout).unwrap();
        for axis in 0..SHAPE.len() {
            let probabilities = softmax_axis(input, axis).unwrap();
            let logs = log_softmax_axis(input, axis).unwrap();
            let sums = logsumexp_axis(input, axis).unwrap();
            let complex_sums = logsumexp_axis(complex, axis).unwrap();
            let weighted = weighted_logsumexp_axis(input, weights, axis).unwrap();
            let complex_weighted = weighted_logsumexp_axis(input, complex_weights, axis).unwrap();
            for &run in &runs {
                let into = normalize_into(input, axis, false, run);
                assert_eq!(bits(&probabilities), bits(&into));
                assert_eq!(bits(&logs), bits(&normalize_into(input, axis, true, run)));
                assert_eq!(bits(&sums), bits(&logsumexp_into(input, axis, run)));
                assert_eq!(
                    complex_bits(&complex_sums),
                    complex_bits(&logsumexp_into(complex, axis, run))
                );
                let into = weighted_into(input, weights, axis, run);
                assert_same_log_values(&weighted, &into);
                let into = weighted_into(input, complex_weights, axis, run);
                assert_same_log_values(&complex_weighted, &into);
            }
            for (index, sum) in sums.iter().enumerate() {
                let expected = logsumexp(&lane(input, axis, index)).unwrap();
                assert_eq!(sum.into_f64().to_bits(), expected.into_f64().to_bits());
                let expected = logsumexp(&lane(complex, axis, index)).unwrap();
                assert_eq!(
                    complex_bits(&[complex_sums[index]]),
                    complex_bits(&[expected])
                );
                let values = lane(input, axis, index);
                let expected = weighted_logsumexp(&values, &lane(weights, axis, index)).unwrap();
                assert_same_log_values(&weighted[index..=index], &[expected]);
                let expected =
                    weighted_logsumexp(&values, &lane(complex_weights, axis, index)).unwrap();
                assert_same_log_values(&complex_weighted[index..=index], &[expected]);
            }
        }
    }
}

#[test]
fn owned_axis_forms_equal_into_forms_bit_for_bit() {
    owned_and_into_forms_agree::<f32>();
    owned_and_into_forms_agree::<f64>();
}

fn slice_forms_agree<S: RealScalar>() {
    let x = logits::<S>(7, true);
    let view = NdView::row_major(&x, &[7]).unwrap();
    let mut output = vec![S::NAN; 7];
    let mut scratch = vec![S::ZERO; 7];
    softmax_into(&x, &mut output, &mut scratch).unwrap();
    let probabilities = softmax(&x).unwrap();
    assert_eq!(bits(&probabilities), bits(&softmax_axis(view, 0).unwrap()));
    assert_eq!(bits(&probabilities), bits(&output));
    log_softmax_into(&x, &mut output, &mut scratch).unwrap();
    let logs = log_softmax(&x).unwrap();
    assert_eq!(bits(&logs), bits(&log_softmax_axis(view, 0).unwrap()));
    assert_eq!(bits(&logs), bits(&output));
}

#[test]
fn slice_forms_equal_rank_one_axis_forms() {
    slice_forms_agree::<f32>();
    slice_forms_agree::<f64>();
}

fn rank_zero_and_axis_errors<S: RealScalar>() {
    let x = [S::ONE; 6];
    let scalar_view = NdView::row_major(&x, &[]).unwrap();
    let matrix = NdView::row_major(&x, &[2, 3]).unwrap();
    for (view, axis) in [(scalar_view, 0), (matrix, 2)] {
        for error in [
            softmax_axis(view, axis).unwrap_err(),
            log_softmax_axis(view, axis).unwrap_err(),
            logsumexp_axis(view, axis).unwrap_err(),
            weighted_logsumexp_axis(view, view, axis).unwrap_err(),
        ] {
            assert_eq!(error.kind(), SciOxideErrorKind::OutOfBounds);
        }
    }
    let policy = ExecutionPolicy::Sequential;
    let cx = ExecutionContext::sequential();
    let sentinel = scalar::<S>(77.0);
    let mut output = [sentinel; 6];
    let mut scratch = [sentinel; 6];
    let target = NdViewMut::row_major(&mut output, &[2, 3]).unwrap();
    let error = softmax_axis_into(matrix, 2, policy, target, &mut scratch, cx).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::OutOfBounds);
    let target = NdViewMut::row_major(&mut output, &[2]).unwrap();
    let error = logsumexp_axis_into(matrix, 2, policy, target, &mut scratch, cx).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::OutOfBounds);
    let mut logs = [LogValue::<S, S>::from_value(sentinel).unwrap(); 2];
    let mut log_scratch = logs;
    let target = NdViewMut::row_major(&mut logs, &[2]).unwrap();
    let error =
        weighted_logsumexp_axis_into(matrix, matrix, 2, policy, target, &mut log_scratch, cx)
            .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::OutOfBounds);
    assert_eq!(bits(&output), bits(&[sentinel; 6]));
    assert_eq!(bits(&scratch), bits(&[sentinel; 6]));
    assert_eq!(logs, log_scratch);
}

#[test]
fn rank_zero_and_missing_axes_are_out_of_bounds() {
    rank_zero_and_axis_errors::<f32>();
    rank_zero_and_axis_errors::<f64>();
}

fn invalid_values_and_storage<S: RealScalar>() {
    let policy = ExecutionPolicy::Sequential;
    let cx = ExecutionContext::sequential();
    let sentinel = scalar::<S>(77.0);
    let finite = logits::<S>(6, false);
    let mut nan = finite.clone();
    nan[4] = S::NAN;
    let mut masked_lane = finite.clone();
    masked_lane[3..].fill(S::NEG_INFINITY);
    let shape = [2, 3];
    for (data, kind) in [
        (&nan, SciOxideErrorKind::NonFinite),
        (&masked_lane, SciOxideErrorKind::DomainError),
    ] {
        let input = NdView::row_major(data, &shape).unwrap();
        for logarithmic in [false, true] {
            let mut output = [sentinel; 6];
            let mut scratch = [S::ZERO; 6];
            let target = NdViewMut::row_major(&mut output, &shape).unwrap();
            let error = if logarithmic {
                log_softmax_axis_into(input, 1, policy, target, &mut scratch, cx)
            } else {
                softmax_axis_into(input, 1, policy, target, &mut scratch, cx)
            }
            .unwrap_err();
            assert_eq!(error.kind(), kind);
            assert_eq!(bits(&output), bits(&[sentinel; 6]));
        }
        assert_eq!(softmax_axis(input, 1).unwrap_err().kind(), kind);
        assert_eq!(log_softmax_axis(input, 1).unwrap_err().kind(), kind);
    }
    let input = NdView::row_major(&nan, &shape).unwrap();
    assert_eq!(
        logsumexp_axis(input, 1).unwrap_err().kind(),
        SciOxideErrorKind::NonFinite
    );
    let mut output = [sentinel; 2];
    let mut scratch = [S::ZERO; 2];
    let target = NdViewMut::row_major(&mut output, &[2]).unwrap();
    let error = logsumexp_axis_into(input, 1, policy, target, &mut scratch, cx).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
    assert_eq!(bits(&output), bits(&[sentinel; 2]));
    // A fully masked lane is an empty sum, not an error, for log-sum-exp.
    let masked = NdView::row_major(&masked_lane, &shape).unwrap();
    let weights = NdView::row_major(&finite, &shape).unwrap();
    assert_eq!(logsumexp_axis(masked, 1).unwrap()[1], S::NEG_INFINITY);
    let weighted = weighted_logsumexp_axis(masked, weights, 1).unwrap();
    assert!(weighted[1].log_magnitude().is_none());

    // Storage errors precede any write to `output` or `scratch`.
    let input = NdView::row_major(&finite, &shape).unwrap();
    let other = NdView::row_major(&finite, &[3, 2]).unwrap();
    let error = weighted_logsumexp_axis(input, other, 1).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    let log_sentinel = LogValue::<S, S>::from_value(sentinel).unwrap();
    for (output_shape, scratch_len) in [(&[2][..], 1), (&[3][..], 2)] {
        let mut logs = [log_sentinel; 3];
        let mut log_scratch = [log_sentinel; 2];
        let target = NdViewMut::row_major(&mut logs, output_shape).unwrap();
        let scratch = &mut log_scratch[..scratch_len];
        let error =
            weighted_logsumexp_axis_into(input, other, 1, policy, target, scratch, cx).unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
        let target = NdViewMut::row_major(&mut logs, output_shape).unwrap();
        let scratch = &mut log_scratch[..scratch_len];
        let error =
            weighted_logsumexp_axis_into(input, input, 1, policy, target, scratch, cx).unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
        assert_eq!(logs, [log_sentinel; 3]);
        assert_eq!(log_scratch, [log_sentinel; 2]);
        let mut output = [sentinel; 3];
        let mut scratch = [sentinel; 2];
        let target = NdViewMut::row_major(&mut output, output_shape).unwrap();
        let error = logsumexp_axis_into(input, 1, policy, target, &mut scratch[..scratch_len], cx)
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
        assert_eq!(bits(&output), bits(&[sentinel; 3]));
        assert_eq!(bits(&scratch), bits(&[sentinel; 2]));
    }
    for (output_shape, scratch_len) in [(&[2, 3][..], 5), (&[3, 2][..], 6)] {
        for logarithmic in [false, true] {
            let mut output = [sentinel; 6];
            let mut scratch = [sentinel; 6];
            let target = NdViewMut::row_major(&mut output, output_shape).unwrap();
            let scratch_slice = &mut scratch[..scratch_len];
            let error = if logarithmic {
                log_softmax_axis_into(input, 1, policy, target, scratch_slice, cx)
            } else {
                softmax_axis_into(input, 1, policy, target, scratch_slice, cx)
            }
            .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
            assert_eq!(bits(&output), bits(&[sentinel; 6]));
            assert_eq!(bits(&scratch), bits(&[sentinel; 6]));
        }
    }
}

#[test]
fn invalid_values_and_storage_leave_output_unchanged() {
    invalid_values_and_storage::<f32>();
    invalid_values_and_storage::<f64>();
}

fn independent_normalization<S: RealScalar>() {
    let width = if S::MANTISSA_DIGITS == 24 { "32" } else { "64" };
    let decode = |bits: &str| {
        let bits = u64::from_str_radix(bits, 16).unwrap();
        let value = if width == "32" {
            f32::from_bits(bits as u32) as f64
        } else {
            f64::from_bits(bits)
        };
        if value == f64::NEG_INFINITY {
            S::NEG_INFINITY
        } else {
            S::try_from_f64_lossy(value).unwrap()
        }
    };
    let mut strata = std::collections::BTreeSet::new();
    for line in include_str!("data/normalization.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields[0] != width {
            continue;
        }
        strata.insert(fields[1]);
        assert_eq!(fields.len(), 3 + 4 * fields[2].parse::<usize>().unwrap());
        let mut values = Vec::new();
        let mut probabilities = Vec::new();
        let mut logs = Vec::new();
        for group in fields[3..].chunks_exact(4) {
            let count = group[0].parse::<usize>().unwrap();
            values.extend(core::iter::repeat_n(decode(group[1]), count));
            probabilities.extend(core::iter::repeat_n(decode(group[2]), count));
            logs.extend(core::iter::repeat_n(decode(group[3]), count));
        }
        for (actual, expected) in [
            (softmax(&values).unwrap(), probabilities),
            (log_softmax(&values).unwrap(), logs),
        ] {
            for (index, (a, e)) in actual.into_iter().zip(expected).enumerate() {
                if !e.is_finite() || e == S::ZERO {
                    assert_eq!(
                        a.into_f64().to_bits(),
                        e.into_f64().to_bits(),
                        "{line}: lane {index}"
                    );
                } else {
                    let ulp = (e.abs().next_after(S::INFINITY) - e.abs()).into_f64();
                    assert!(
                        (a.into_f64() - e.into_f64()).abs() <= 16.0 * ulp,
                        "{line}: lane {index}, {a:?} != {e:?}"
                    );
                }
            }
        }
    }
    assert_eq!(
        strata.into_iter().collect::<Vec<_>>(),
        [
            "centered",
            "dominant-subnormal",
            "equal",
            "masked",
            "singleton"
        ]
    );
}

#[test]
fn independent_normalization_both_widths() {
    independent_normalization::<f32>();
    independent_normalization::<f64>();
}

#[test]
fn arithmetic_overflow_preserves_log_normalization_output() {
    let mut output = [77.0; 2];
    let mut scratch = [0.0; 2];
    let result = log_softmax_into(&[f64::MAX, -f64::MAX], &mut output, &mut scratch);
    assert_eq!(
        result.unwrap_err().kind(),
        SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(output, [77.0; 2]);
    // A vanished correction is still negative; a mathematical singleton is +0.
    assert_eq!(
        log_softmax(&[0.0_f64, -1200.0]).unwrap()[0].to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        log_softmax(&[0.0, f64::NEG_INFINITY]).unwrap()[0].to_bits(),
        0
    );
}

fn real_axis_forms_without_complex_capability<S: RealScalar>() {
    let layout = StridedLayout::new(&[2], &[1]).unwrap();
    let mut output = [S::ZERO; 2];
    let mut scratch = output;
    {
        softmax_axis_into(
            NdView::new(&[S::ZERO, S::NEG_INFINITY], layout).unwrap(),
            0,
            ExecutionPolicy::Auto,
            NdViewMut::new(&mut output, layout).unwrap(),
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap();
    };
    assert_eq!(output, [S::ONE, S::ZERO]);
}

#[test]
fn real_axis_forms_require_only_real_scalars() {
    real_axis_forms_without_complex_capability::<f32>();
    real_axis_forms_without_complex_capability::<f64>();
}

#[test]
fn small_rank_shape_errors_are_inline_and_transactional() {
    let layout = StridedLayout::new(&[4], &[1]).unwrap();
    let input = [0.0; 4];
    let view = NdView::new(&input, layout).unwrap();
    let policy = ExecutionPolicy::Sequential;
    let cx = ExecutionContext::sequential();
    let scalar = StridedLayout::new(&[], &[]).unwrap();
    {
        for actual in [3, 5] {
            let extents = [actual];
            let wrong = StridedLayout::new(&extents, &[1]).unwrap();
            let mut output = [77.0; 5];
            let mut scratch = [88.0; 4];
            let target = NdViewMut::new(&mut output, wrong).unwrap();
            let error = softmax_axis_into(view, 0, policy, target, &mut scratch, cx).unwrap_err();
            assert_shape(&error, &[4], &extents);
            assert_eq!(output, [77.0; 5]);
            assert_eq!(scratch, [88.0; 4]);
            let mut output = [LogValue::<f64, f64>::ZERO];
            let mut scratch = output;
            let error = weighted_logsumexp_axis_into(
                view,
                NdView::new(&[1.0; 5], wrong).unwrap(),
                0,
                policy,
                NdViewMut::new(&mut output, scalar).unwrap(),
                &mut scratch,
                cx,
            )
            .unwrap_err();
            assert_shape(&error, &[4], &extents);
            assert_eq!(output, [LogValue::ZERO]);
        }
        let mut output = [77.0];
        let mut scratch = [88.0];
        let target = NdViewMut::new(&mut output, StridedLayout::new(&[1], &[1]).unwrap()).unwrap();
        let error = logsumexp_axis_into(view, 0, policy, target, &mut scratch, cx).unwrap_err();
        assert_shape(&error, &[], &[1]);
        assert_eq!(output, [77.0]);
        assert_eq!(scratch, [88.0]);
        // A scalar weight has no axis zero: reject its shape before extracting lanes.
        let mut output = [LogValue::<f64, f64>::ZERO];
        let mut scratch = output;
        let error = weighted_logsumexp_axis_into(
            view,
            NdView::new(&[1.0], scalar).unwrap(),
            0,
            policy,
            NdViewMut::new(&mut output, scalar).unwrap(),
            &mut scratch,
            cx,
        )
        .unwrap_err();
        assert_shape(&error, &[4], &[]);
        assert_eq!(output, [LogValue::ZERO]);
        let matrix = StridedLayout::new(&[2, 2], &[2, 1]).unwrap();
        let wrong_matrix = StridedLayout::new(&[4, 1], &[1, 1]).unwrap();
        let mut output = [77.0; 4];
        let mut scratch = [88.0; 4];
        let error = softmax_axis_into(
            NdView::new(&input, matrix).unwrap(),
            1,
            policy,
            NdViewMut::new(&mut output, wrong_matrix).unwrap(),
            &mut scratch,
            cx,
        )
        .unwrap_err();
        assert_shape(&error, &[2, 2], &[4, 1]);
        let tensor = StridedLayout::new(&[2, 2, 1], &[2, 1, 1]).unwrap();
        let error = logsumexp_axis_into(
            NdView::new(&input, tensor).unwrap(),
            2,
            policy,
            NdViewMut::new(&mut output, wrong_matrix).unwrap(),
            &mut scratch,
            cx,
        )
        .unwrap_err();
        assert_shape(&error, &[2, 2], &[4, 1]);
        assert_eq!(output, [77.0; 4]);
        assert_eq!(scratch, [88.0; 4]);
    };
}

#[test]
fn high_rank_reduced_dimensions_are_preserved() {
    let layout = StridedLayout::new(&[2, 1, 2, 1], &[2, 2, 1, 1]).unwrap();
    let mut output = [77.0; 4];
    let mut scratch = [88.0; 4];
    let wrong = StridedLayout::new(&[4, 1], &[1, 1]).unwrap();
    let error = logsumexp_axis_into(
        NdView::new(&[0.0; 4], layout).unwrap(),
        3,
        ExecutionPolicy::Sequential,
        NdViewMut::new(&mut output, wrong).unwrap(),
        &mut scratch,
        ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert_shape(&error, &[2, 1, 2], &[4, 1]);
    assert_eq!(output, [77.0; 4]);
}
