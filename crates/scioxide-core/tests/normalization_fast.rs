//! Independent fixtures for batched normalization and transactional workspace reuse.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::*;

#[derive(Debug, Clone, Copy)]
enum Normalization {
    Probability,
    Log,
}

fn normalize_axis_into<S: RealScalar>(
    normalization: Normalization,
    input: NdView<'_, '_, S>,
    output: NdViewMut<'_, '_, S>,
    scratch: &mut [S],
    cx: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let policy = ExecutionPolicy::Sequential;
    match normalization {
        Normalization::Probability => softmax_axis_into(input, 1, policy, output, scratch, cx),
        Normalization::Log => log_softmax_axis_into(input, 1, policy, output, scratch, cx),
    }
}

fn decode<S: RealScalar>(bits: &str) -> S {
    let bits = u64::from_str_radix(bits, 16).unwrap();
    let value = if S::MANTISSA_DIGITS == 24 {
        f64::from(f32::from_bits(bits as u32))
    } else {
        f64::from_bits(bits)
    };
    if value == f64::NEG_INFINITY {
        S::NEG_INFINITY
    } else {
        S::try_from_f64_lossy(value).unwrap()
    }
}

fn assert_reference<S: RealScalar>(actual: S, expected: S, row: &str) {
    if expected == S::ZERO || expected == S::NEG_INFINITY {
        assert_eq!(
            actual.into_f64().to_bits(),
            expected.into_f64().to_bits(),
            "{row}"
        );
    } else {
        let ulp = (expected.abs().next_after(S::INFINITY) - expected.abs()).into_f64();
        assert!(
            (actual.into_f64() - expected.into_f64()).abs() <= 16.0 * ulp,
            "{row}: {actual:?} != {expected:?}"
        );
    }
}

fn fixtures<S: RealScalar>() {
    let width = if S::MANTISSA_DIGITS == 24 { "32" } else { "64" };
    let sentinel = S::try_from_f64_lossy(77.0).unwrap();
    let mut strata = std::collections::BTreeSet::new();
    for row in include_str!("data/normalization_fast.txt")
        .lines()
        .filter(|row| !row.starts_with('#'))
    {
        let fields: Vec<_> = row.split_whitespace().collect();
        if fields[0] != width {
            continue;
        }
        strata.insert(fields[1]);
        assert_eq!(fields.len(), 3 + 4 * fields[2].parse::<usize>().unwrap());
        let mut input = Vec::new();
        let mut probability = Vec::new();
        let mut logarithm = Vec::new();
        for group in fields[3..].chunks_exact(4) {
            let count = group[0].parse::<usize>().unwrap();
            input.extend(core::iter::repeat_n(decode::<S>(group[1]), count));
            probability.extend(core::iter::repeat_n(decode::<S>(group[2]), count));
            logarithm.extend(core::iter::repeat_n(decode::<S>(group[3]), count));
        }
        let n = input.len();
        for (normalization, expected) in [
            (Normalization::Probability, &probability),
            (Normalization::Log, &logarithm),
        ] {
            let mut output = vec![sentinel; n + 2];
            let mut scratch = vec![sentinel; n + 2];
            {
                match normalization {
                    Normalization::Probability => {
                        softmax_into(&input, &mut output[..n], &mut scratch)
                    }
                    Normalization::Log => log_softmax_into(&input, &mut output[..n], &mut scratch),
                }
                .unwrap();
            };
            for (&actual, &expected) in output[..n].iter().zip(expected) {
                assert_reference(actual, expected, row);
            }
            assert_eq!(&output[n..], &[sentinel; 2]);
            assert_eq!(&scratch[n..], &[sentinel; 2]);
            for (input_strides, output_strides) in
                [([n, 1], [n, 1]), ([1, 2], [1, 2]), ([0, 1], [2 * n, 2])]
            {
                let shape = [2, n];
                let input_layout = StridedLayout::new_broadcast(&shape, &input_strides).unwrap();
                let output_layout = StridedLayout::new(&shape, &output_strides).unwrap();
                let mut values = vec![S::ZERO; input_layout.required_span()];
                for lane in 0..2 {
                    for (coordinate, &value) in input.iter().enumerate() {
                        values[lane * input_strides[0] + coordinate * input_strides[1]] = value;
                    }
                }
                let mut output = vec![sentinel; output_layout.required_span() + 2];
                let mut scratch = vec![sentinel; 2 * n + 2];
                {
                    normalize_axis_into(
                        normalization,
                        NdView::new(&values, input_layout).unwrap(),
                        NdViewMut::new(&mut output, output_layout).unwrap(),
                        &mut scratch,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                };
                for lane in 0..2 {
                    for (coordinate, &expected) in expected.iter().enumerate() {
                        assert_reference(
                            output[lane * output_strides[0] + coordinate * output_strides[1]],
                            expected,
                            row,
                        );
                    }
                }
                if output_strides == [2 * n, 2] {
                    assert!(
                        output[..output_layout.required_span()]
                            .iter()
                            .skip(1)
                            .step_by(2)
                            .all(|&value| value == sentinel)
                    );
                }
                assert!(
                    output[output_layout.required_span()..]
                        .iter()
                        .all(|&value| value == sentinel)
                );
                assert_eq!(&scratch[2 * n..], &[sentinel; 2]);
            }
        }
    }
    assert_eq!(
        strata.into_iter().collect::<Vec<_>>(),
        [
            "block",
            "centered",
            "centered-boundary",
            "certificate-admit",
            "certificate-reject",
            "shift-residual",
            "signed-zero",
            "table-boundary",
            "tail",
            "tied-maximum"
        ]
    );
}

#[test]
fn independent_batched_normalization_in_both_widths_and_layouts() {
    fixtures::<f32>();
    fixtures::<f64>();
}

struct CompleteThenStop<'a> {
    cancellation: &'a CancellationToken,
    fail: bool,
}

impl Executor for CompleteThenStop<'_> {
    fn worker_budget(&self) -> core::num::NonZeroUsize {
        core::num::NonZeroUsize::MIN
    }

    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        work: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        SequentialExecutor.execute(context, plan, work)?;
        if self.fail {
            Err(SciOxideError::execution_failure(
                None,
                ExecutionFailureReason::ProviderFailure,
                "injected failure after staging",
            ))
        } else {
            self.cancellation.cancel();
            Ok(())
        }
    }
}

fn publication_failure<S: RealScalar>() {
    let input = [0.0, -0.25, -1.0, -2.0, f64::NEG_INFINITY].map(|value| {
        if value == f64::NEG_INFINITY {
            S::NEG_INFINITY
        } else {
            S::try_from_f64_lossy(value).unwrap()
        }
    });
    let sentinel = S::try_from_f64_lossy(77.0).unwrap();
    let layout = StridedLayout::new_broadcast(&[2, 5], &[0, 1]).unwrap();
    let output_layout = StridedLayout::new(&[2, 5], &[10, 2]).unwrap();
    for normalization in [Normalization::Probability, Normalization::Log] {
        let mut output = [sentinel; 20];
        let mut scratch = [sentinel; 12];
        for fail in [false, true] {
            output.fill(sentinel);
            scratch.fill(sentinel);
            let cancellation = CancellationToken::new();
            let executor = CompleteThenStop {
                cancellation: &cancellation,
                fail,
            };
            {
                let error = normalize_axis_into(
                    normalization,
                    NdView::new(&input, layout).unwrap(),
                    NdViewMut::new(&mut output, output_layout).unwrap(),
                    &mut scratch,
                    ExecutionContext::new(&executor).with_cancellation(&cancellation),
                )
                .unwrap_err();
                if fail {
                    assert!(matches!(
                        error.view(),
                        SciOxideErrorView::ExecutionFailure {
                            reason: ExecutionFailureReason::ProviderFailure,
                            ..
                        }
                    ));
                } else {
                    assert!(matches!(
                        error.view(),
                        SciOxideErrorView::ExecutionStopped {
                            reason: ExecutionStopReason::Cancelled,
                            ..
                        }
                    ));
                }
            };
            assert_eq!(output, [sentinel; 20]);
            assert!(scratch[..10].iter().all(|&value| value != sentinel));
            assert_eq!(&scratch[10..], &[sentinel; 2]);
            let expected = scratch;
            {
                normalize_axis_into(
                    normalization,
                    NdView::new(&input, layout).unwrap(),
                    NdViewMut::new(&mut output, output_layout).unwrap(),
                    &mut scratch,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            };
            for (index, &value) in expected[..10].iter().enumerate() {
                assert_eq!(
                    output[2 * index].into_f64().to_bits(),
                    value.into_f64().to_bits()
                );
                assert_eq!(output[2 * index + 1], sentinel);
            }
        }
        for invalid in [S::NAN, S::INFINITY] {
            let mut rejected = input;
            rejected[4] = invalid;
            output.fill(sentinel);
            {
                assert!(
                    normalize_axis_into(
                        normalization,
                        NdView::new(&rejected, layout).unwrap(),
                        NdViewMut::new(&mut output, output_layout).unwrap(),
                        &mut scratch,
                        ExecutionContext::sequential()
                    )
                    .is_err()
                );
            };
            assert_eq!(output, [sentinel; 20]);
        }
    }
}

#[test]
fn completed_staging_is_transactional_and_reusable_in_both_widths() {
    publication_failure::<f32>();
    publication_failure::<f64>();
}
