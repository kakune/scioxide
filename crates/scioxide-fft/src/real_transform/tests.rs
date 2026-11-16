use super::*;
use scioxide_core::{SciOxideErrorKind, backend::KernelBackend};

const KINDS: [RealTransformKind; 8] = [
    RealTransformKind::Dct(DctType::I),
    RealTransformKind::Dct(DctType::II),
    RealTransformKind::Dct(DctType::III),
    RealTransformKind::Dct(DctType::IV),
    RealTransformKind::Dst(DstType::I),
    RealTransformKind::Dst(DstType::II),
    RealTransformKind::Dst(DstType::III),
    RealTransformKind::Dst(DstType::IV),
];

fn native_lane_bits<S: scioxide_core::RealScalar>() {
    use crate::lanes::{LaneGeometry, Placement};
    let cx = ExecutionContext::sequential_native();
    for backend in crate::test_support::fft_backends(cx.processor()) {
        for n in [6, 64, 256, 512, 514] {
            for count in [3, 4, 5] {
                for axis in [0, 1] {
                    let shape = if axis == 0 { [n, count] } else { [count, n] };
                    let strides = [shape[1], 1];
                    let geometry = LaneGeometry::new::<S>(&shape, &strides, axis);
                    let input: Vec<_> = (0..n * count)
                        .map(|i| {
                            if i % n == 0 {
                                -S::ZERO
                            } else if i % n == 1 {
                                S::MIN_SUBNORMAL
                            } else {
                                S::try_from_f64_lossy(((i * 17 % 97) as f64 - 48.0) / 128.0)
                                    .unwrap()
                            }
                        })
                        .collect();
                    for kind in [DctType::II.into(), DstType::III.into(), DctType::IV.into()] {
                        for direction in [FftDirection::Forward, FftDirection::Inverse] {
                            for norm in [Normalization::Unscaled, Normalization::Orthonormal] {
                                let plan = RealTransformPlan::<S>::new(
                                    RealTransformPlanSpec::new(n, kind, direction, norm)
                                        .with_backend_strategy(
                                            crate::FftBackendStrategy::Required(backend),
                                        ),
                                )
                                .unwrap();
                                let mut alone = plan.clone();
                                alone.blocks = None;
                                for mode in [LaneMode::Bounded, LaneMode::Checked] {
                                    let execute = |plan: &RealTransformPlan<S>| {
                                        let mut values = input.clone();
                                        let mut scratch = vec![
                                            Complex::new(S::ZERO, S::ZERO);
                                            plan.lane_scratch_len()
                                        ];
                                        let mut staging =
                                            vec![
                                                S::ZERO;
                                                geometry.staging_len().max(plan.lane_staging_len())
                                            ];
                                        plan.transform_lanes(
                                            Lanes::new(
                                                &mut values,
                                                Placement::new(geometry, &shape, &strides),
                                                &mut staging,
                                            ),
                                            &mut scratch,
                                            mode,
                                            cx,
                                        )
                                        .unwrap();
                                        values
                                    };
                                    for (a, b) in execute(&plan).into_iter().zip(execute(&alone)) {
                                        assert_eq!(
                                            a.try_to_f64_exact().unwrap().to_bits(),
                                            b.try_to_f64_exact().unwrap().to_bits(),
                                            "{backend:?} {shape:?} axis={axis} {kind:?} {direction:?} {norm:?} {mode:?}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn native_blocks_equal_individual_trigonometric_lanes() {
    native_lane_bits::<f32>();
    native_lane_bits::<f64>();
}

fn checked_two_ranges<S: scioxide_core::RealScalar>(common_frame: bool) {
    use crate::lanes::{LaneGeometry, LanePairs, Placement};
    let context = ExecutionContext::sequential_native();
    for backend in crate::test_support::fft_backends(context.processor()) {
        for n in [6, 64] {
            for count in [4, 5] {
                for axis in [0, 1] {
                    let shape = if axis == 0 { [n, count] } else { [count, n] };
                    let strides = [shape[1], 1];
                    let geometry = LaneGeometry::new::<S>(&shape, &strides, axis);
                    let input: Vec<_> = (0..n * count)
                        .map(|i| {
                            let lane = if axis == 0 { i % count } else { i / n };
                            let value =
                                S::try_from_f64_lossy(((i * 17 % 19) as f64 - 9.0) / 16.0).unwrap();
                            if common_frame {
                                let point = if axis == 0 { i / count } else { i % n };
                                return if lane % 4 == 0 {
                                    S::NEG_ZERO
                                } else if point == 0 {
                                    S::try_from_f64_lossy(-0.75).unwrap()
                                } else {
                                    value
                                };
                            }
                            match lane % 4 {
                                0 => S::NEG_ZERO,
                                1 => value
                                    .scalbn(crate::scaling::range_exponent(S::MIN_SUBNORMAL) + 3),
                                2 => value.scalbn(crate::scaling::range_exponent(S::MAX) - 12),
                                _ => value,
                            }
                        })
                        .collect();
                    for (kind, direction) in [
                        (DctType::II.into(), FftDirection::Forward),
                        (DstType::II.into(), FftDirection::Forward),
                        (DctType::III.into(), FftDirection::Inverse),
                        (DstType::III.into(), FftDirection::Inverse),
                    ] {
                        let plan = RealTransformPlan::<S>::new(
                            RealTransformPlanSpec::new(
                                n,
                                kind,
                                direction,
                                Normalization::Orthonormal,
                            )
                            .with_backend_strategy(crate::FftBackendStrategy::Required(backend)),
                        )
                        .unwrap();
                        let mut individual = plan.clone();
                        individual.blocks = None;
                        let execute = |plan: &RealTransformPlan<S>, borrowed: bool| {
                            let mut values = input.clone();
                            let mut scratch =
                                vec![Complex::new(S::ZERO, S::ZERO); plan.lane_scratch_len()];
                            let mut staging =
                                vec![S::ZERO; geometry.staging_len().max(plan.lane_staging_len())];
                            let placement = Placement::new(geometry, &shape, &strides);
                            if borrowed {
                                plan.transform_source_lanes(
                                    LanePairs::new(
                                        (&input, placement),
                                        (&mut values, placement),
                                        &mut staging,
                                    ),
                                    &mut scratch,
                                    context,
                                )
                                .unwrap();
                            } else {
                                plan.transform_lanes(
                                    Lanes::new(
                                        &mut values,
                                        Placement::new(geometry, &shape, &strides),
                                        &mut staging,
                                    ),
                                    &mut scratch,
                                    LaneMode::Checked,
                                    context,
                                )
                                .unwrap();
                            }
                            values
                        };
                        for borrowed in [false, true] {
                            for (a, b) in execute(&plan, borrowed)
                                .into_iter()
                                .zip(execute(&individual, false))
                            {
                                assert_eq!(
                                    a.try_to_f64_exact().unwrap().to_bits(),
                                    b.try_to_f64_exact().unwrap().to_bits()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn checked_two_blocks_preserve_mixed_lane_ranges_and_tails() {
    for common_frame in [false, true] {
        checked_two_ranges::<f32>(common_frame);
        checked_two_ranges::<f64>(common_frame);
    }
}

#[test]
fn native_blocks_preserve_lane_error_order() {
    use crate::lanes::{LaneGeometry, LanePairs, Placement};
    let cx = ExecutionContext::sequential_native();
    for backend in crate::test_support::fft_backends(cx.processor()) {
        let plan = RealTransformPlan::<f64>::new(
            RealTransformPlanSpec::new(
                64,
                DctType::II.into(),
                FftDirection::Forward,
                Normalization::Unscaled,
            )
            .with_backend_strategy(crate::FftBackendStrategy::Required(backend)),
        )
        .unwrap();
        assert!(plan.native_lanes());
        // Competing failures in a full block and in a partial block after a
        // successful block must report the earlier lane's error.
        for count in [4, 6] {
            for axis in [0, 1] {
                let shape = if axis == 0 { [64, count] } else { [count, 64] };
                let strides = [shape[1], 1];
                let geometry = LaneGeometry::new::<f64>(&shape, &strides, axis);
                let placement = Placement::new(geometry, &shape, &strides);
                for (first, second, expected) in [
                    (f64::MAX, f64::NAN, SciOxideErrorKind::NumericalInstability),
                    (f64::NAN, f64::MAX, SciOxideErrorKind::NonFinite),
                ] {
                    let input: Vec<_> = (0..64 * count)
                        .map(|i| {
                            let lane = if axis == 0 { i % count } else { i / 64 };
                            if lane == count - 2 {
                                first
                            } else if lane == count - 1 {
                                second
                            } else {
                                0.25
                            }
                        })
                        .collect();
                    let mut scratch = vec![Complex::new(0.0, 0.0); plan.lane_scratch_len()];
                    let mut staging = vec![0.0; geometry.staging_len()];
                    let mut output = vec![99.0; input.len()];
                    let error = plan
                        .transform_source_lanes(
                            LanePairs::new(
                                (&input, placement),
                                (&mut output, placement),
                                &mut staging,
                            ),
                            &mut scratch,
                            cx,
                        )
                        .unwrap_err();
                    assert_eq!(error.kind(), expected);
                    let mut values = input;
                    let error = plan
                        .transform_lanes(
                            Lanes::new(&mut values, placement, &mut staging),
                            &mut scratch,
                            LaneMode::Checked,
                            cx,
                        )
                        .unwrap_err();
                    assert_eq!(error.kind(), expected);
                }
            }
        }
    }
}

#[test]
fn unavailable_backends_are_rejected_during_planning() {
    // AVX-512 has no FFT leaves and is rejected on every host; other
    // backends only where the host does not verify them.
    let context = scioxide_core::ExecutionContext::sequential_native();
    for backend in core::iter::once(KernelBackend::Avx512).chain(
        [KernelBackend::Avx2Fma, KernelBackend::Neon]
            .into_iter()
            .filter(|backend| !backend.is_available(context.processor())),
    ) {
        for kind in KINDS {
            let spec =
                RealTransformPlanSpec::new(8, kind, FftDirection::Forward, Normalization::Inverse)
                    .with_execution_policy(ExecutionPolicy::Sequential)
                    .with_backend_strategy(crate::FftBackendStrategy::Required(backend));
            let error = RealTransformPlan::<f64>::new(spec).unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
        }
    }
}

#[test]
fn planning_checks_minimum_lengths_and_size_overflow() {
    for direction in [FftDirection::Forward, FftDirection::Inverse] {
        for kind in KINDS {
            let minimum = if kind == RealTransformKind::Dct(DctType::I) {
                2
            } else {
                1
            };
            let spec = RealTransformPlanSpec::new(minimum, kind, direction, Normalization::Inverse);
            let plan = RealTransformPlan::<f64>::new(spec).unwrap();
            assert_eq!(plan.spec().length(), minimum);
            for length in 0..minimum {
                let error = RealTransformPlan::<f64>::new(RealTransformPlanSpec::new(
                    length,
                    kind,
                    direction,
                    Normalization::Inverse,
                ))
                .unwrap_err();
                assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
            }
        }
        for kind in [DctType::IV.into(), DstType::I.into()] {
            let error = RealTransformPlan::<f64>::new(RealTransformPlanSpec::new(
                usize::MAX,
                kind,
                direction,
                Normalization::Inverse,
            ))
            .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::SizeOverflow);
        }
    }
}

#[test]
fn plan_queries_and_failures_are_transactional() {
    for kind in KINDS {
        for direction in [FftDirection::Forward, FftDirection::Inverse] {
            let spec = RealTransformPlanSpec::new(5, kind, direction, Normalization::Inverse)
                .with_execution_policy(ExecutionPolicy::Sequential);
            let plan = RealTransformPlan::new(spec).unwrap();
            assert_eq!(plan.spec().length(), 5);
            assert_eq!(plan.spec().kind(), kind);
            assert_eq!(plan.spec().direction(), direction);
            assert_eq!(plan.spec().normalization(), Normalization::Inverse);
            assert_eq!(plan.spec().execution_policy(), ExecutionPolicy::Sequential);
            assert_eq!(
                plan.spec().backend_strategy(),
                crate::FftBackendStrategy::Automatic
            );
            assert!(plan.scratch_len > 0);

            let input = [1.0_f64, -2.0, 3.0, -4.0, 5.0];
            let mut output = [99.0; 5];
            let mut workspace = plan.workspace(()).unwrap();
            let short_scratch_len = plan.scratch_len - 1;
            let error = plan
                .execute_into(
                    &input,
                    &mut output,
                    &mut workspace[..short_scratch_len],
                    scioxide_core::ExecutionContext::sequential_native(),
                )
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
            assert_eq!(output, [99.0; 5]);

            for value in [f64::NAN, f64::INFINITY] {
                let mut non_finite = input;
                non_finite[2] = value;
                let error = plan
                    .execute_into(
                        &non_finite,
                        &mut output,
                        &mut workspace,
                        scioxide_core::ExecutionContext::sequential_native(),
                    )
                    .unwrap_err();
                assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
                assert_eq!(output, [99.0; 5]);
            }

            let mut short_output = [99.0; 4];
            let error = plan
                .execute_into(
                    &input,
                    &mut short_output,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
            assert_eq!(short_output, [99.0; 4]);
        }
    }
}

fn fused_two_matches_split<S: scioxide_core::RealScalar>() {
    for n in [4, 6, 10, 12, 18, 64, 66, 130] {
        for kind in [DctType::II.into(), DstType::II.into()] {
            for normalization in [
                Normalization::Unscaled,
                Normalization::Forward,
                Normalization::Inverse,
                Normalization::Orthonormal,
            ] {
                let fused = RealTransformPlan::<S>::new(RealTransformPlanSpec::new(
                    n,
                    kind,
                    FftDirection::Forward,
                    normalization,
                ))
                .unwrap();
                let mut split = fused.clone();
                let Kernel::EvenHalf(real) = &fused.kernel else {
                    unreachable!()
                };
                split.kernel = Kernel::Real(real.clone());
                let input: Vec<_> = (0..n)
                    .map(|i| S::try_from_f64_lossy(((i * 29 % 61) as f64 - 30.0) / 32.0).unwrap())
                    .collect();
                let mut a = vec![S::ZERO; n];
                let mut b = a.clone();
                let mut work = fused.workspace(()).unwrap();
                let cx = ExecutionContext::sequential_native();
                fused.execute_into(&input, &mut a, &mut work, cx).unwrap();
                split.execute_into(&input, &mut b, &mut work, cx).unwrap();
                for (a, b) in a.into_iter().zip(b) {
                    assert_eq!(a.decompose(), b.decompose());
                }
            }
        }
    }
}

#[test]
fn fused_even_two_preserves_split_then_rotate_bits() {
    fused_two_matches_split::<f32>();
    fused_two_matches_split::<f64>();
}

#[test]
fn checked_fallback_matches_direct_publication_and_is_transactional() {
    // Power-of-two input scaling commutes exactly with range scaling, so
    // inputs near the overflow threshold (candidate fallback) must equal
    // the scaled direct publication bit for bit.
    let context = scioxide_core::ExecutionContext::sequential_native();
    for n in [3, 5, 6, 8, 10, 12] {
        for kind in KINDS {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                for normalization in [
                    Normalization::Unscaled,
                    Normalization::Forward,
                    Normalization::Inverse,
                    Normalization::Orthonormal,
                ] {
                    let spec = RealTransformPlanSpec::new(n, kind, direction, normalization);
                    let plan = RealTransformPlan::<f64>::new(spec).unwrap();
                    let mut workspace = plan.workspace(()).unwrap();
                    let shape: Vec<_> = (0..n)
                        .map(|j| ((j * 7 % 5) as f64 - 2.0) / (8.0 * n as f64))
                        .collect();
                    let small: Vec<_> = shape.iter().map(|x| x * 8.0).collect();
                    let large: Vec<_> = small.iter().map(|x| x * 2.0_f64.powi(1021)).collect();
                    let maximum = crate::scaling::real_max(large.iter().copied()).unwrap();
                    let exponent = crate::scaling::range_exponent(maximum) + 2;
                    assert!(
                        normalization != Normalization::Unscaled
                            || !scioxide_core::RealScalar::scalbn(plan.output_bound, exponent)
                                .is_finite(),
                        "{kind:?} {direction:?}"
                    );
                    let mut expected = vec![0.0; n];
                    plan.execute_into(&small, &mut expected, &mut workspace, context)
                        .unwrap();
                    let mut actual = vec![0.0; n];
                    plan.execute_into(&large, &mut actual, &mut workspace, context)
                        .unwrap();
                    let scaled: Vec<_> = actual.iter().map(|y| y.to_bits()).collect();
                    let reference: Vec<_> = expected
                        .iter()
                        .map(|y| (y * 2.0_f64.powi(1021)).to_bits())
                        .collect();
                    assert_eq!(
                        scaled, reference,
                        "{kind:?} {direction:?} {normalization:?}"
                    );

                    if normalization == Normalization::Unscaled {
                        let mut sentinel = vec![99.0; n];
                        let error = plan
                            .execute_into(
                                &vec![f64::MAX; n],
                                &mut sentinel,
                                &mut workspace,
                                context,
                            )
                            .unwrap_err();
                        assert!(matches!(
                            error.view(),
                            scioxide_core::SciOxideErrorView::NumericalInstability {
                                issue: scioxide_core::NumericalIssue::Overflow,
                                ..
                            }
                        ));
                        assert_eq!(sentinel, vec![99.0; n]);
                    }
                }
            }
        }
    }
}

/// Independent per-bin equations keep the even adapters' expression order
/// pinned without repeating the shared mirror traversal. Both scalar and
/// four-lane paths cover endpoints and odd/even halves.
fn even_adapter_bits<S: scioxide_core::RealScalar, const L: usize>() {
    let cx = ExecutionContext::sequential_native();
    for backend in crate::test_support::fft_backends(cx.processor()) {
        for n in [4, 6, 10, 12, 64, 66] {
            for kind in [
                DctType::II.into(),
                DctType::III.into(),
                DstType::II.into(),
                DstType::III.into(),
            ] {
                for direction in [FftDirection::Forward, FftDirection::Inverse] {
                    for normalization in [
                        Normalization::Unscaled,
                        Normalization::Forward,
                        Normalization::Inverse,
                        Normalization::Orthonormal,
                    ] {
                        let plan = RealTransformPlan::<S>::new(
                            RealTransformPlanSpec::new(n, kind, direction, normalization)
                                .with_backend_strategy(crate::FftBackendStrategy::Required(
                                    backend,
                                )),
                        )
                        .unwrap();
                        let Kernel::EvenHalf(real) = &plan.kernel else {
                            unreachable!()
                        };
                        let (_, twiddles) = real.even_parts().unwrap();
                        let m = n / 2;
                        let value = |i: usize, lane: usize| {
                            S::try_from_f64_lossy(
                                ((i * 29 + lane * 13) % 61) as f64 / 32.0 - 0.9375,
                            )
                            .unwrap()
                        };
                        if plan.raw_type() == KernelType::Two {
                            let raw: Vec<_> = (0..m)
                                .map(|i| {
                                    Block::new(
                                        Reals(core::array::from_fn(|lane| value(2 * i, lane))),
                                        Reals(core::array::from_fn(|lane| value(2 * i + 1, lane))),
                                    )
                                })
                                .collect();
                            let mut actual = vec![Reals([S::NAN; L]); n];
                            plan.publish_even(&Get(|i| raw[i]), &mut Put(|i, v| actual[i] = v));
                            let mut expected = vec![Reals([S::NAN; L]); n];
                            for k in 0..=m {
                                for lane in 0..L {
                                    let z = raw[0].lane(lane);
                                    let bin = if k == 0 {
                                        Complex::new(z.re + z.im, S::ZERO)
                                    } else if k == m {
                                        Complex::new(z.re - z.im, S::ZERO)
                                    } else {
                                        let left = raw[k].lane(lane);
                                        let right = raw[m - k].lane(lane);
                                        ((left + right.conj())
                                            - (left - right.conj())
                                                * (Complex::new(S::ZERO, S::ONE) * twiddles[k]))
                                            * S::HALF
                                    } * plan.phases[k];
                                    let index = |j| {
                                        if matches!(kind, RealTransformKind::Dst(_)) {
                                            n - 1 - j
                                        } else {
                                            j
                                        }
                                    };
                                    expected[index(k)].0[lane] = bin.re
                                        * if k == 0 {
                                            plan.scaling.output_first
                                        } else {
                                            plan.scaling.output_scale
                                        };
                                    if k != 0 && k != m {
                                        expected[index(n - k)].0[lane] =
                                            -bin.im * plan.scaling.output_scale;
                                    }
                                }
                            }
                            for (actual, expected) in actual.iter().zip(expected) {
                                for (actual, expected) in actual.0.iter().zip(expected.0) {
                                    assert_eq!(
                                        actual.decompose(),
                                        expected.decompose(),
                                        "{backend:?} n={n} {kind:?} {direction:?}"
                                    );
                                    assert_eq!(
                                        actual.is_sign_negative(),
                                        expected.is_sign_negative()
                                    );
                                }
                            }
                        } else {
                            let input: Vec<Reals<S, L>> = (0..n)
                                .map(|i| Reals(core::array::from_fn(|lane| value(i, lane))))
                                .collect();
                            let mut actual = vec![Block::real(Reals::splat(S::ZERO)); m];
                            plan.stage_even(
                                &Get(|i| input[i]),
                                &Map(|v| v),
                                &mut Put(|i, v| actual[i] = v),
                            );
                            for (k, actual) in actual.iter().enumerate() {
                                for lane in 0..L {
                                    let sample = |i: usize| {
                                        input[if matches!(kind, RealTransformKind::Dst(_)) {
                                            n - 1 - i
                                        } else {
                                            i
                                        }]
                                        .0[lane]
                                    };
                                    let bin = |j: usize| {
                                        if j == 0 {
                                            Complex::new(
                                                sample(0) * plan.scaling.input_first,
                                                S::ZERO,
                                            )
                                        } else if j == m {
                                            Complex::new(
                                                sample(m) * (S::TWO * plan.phases[m].re),
                                                S::ZERO,
                                            )
                                        } else {
                                            Complex::new(sample(j), -sample(n - j))
                                                * plan.phases[j].conj()
                                        }
                                    };
                                    let (left, right) = (bin(k), bin(m - k));
                                    let expected = ((left + right.conj())
                                        + (left - right.conj())
                                            * (Complex::new(S::ZERO, S::ONE) * twiddles[k].conj()))
                                        * S::HALF;
                                    let actual = actual.lane(lane);
                                    for (actual, expected) in
                                        [(actual.re, expected.re), (actual.im, expected.im)]
                                    {
                                        assert_eq!(
                                            actual.decompose(),
                                            expected.decompose(),
                                            "{backend:?} n={n} {kind:?} {direction:?}"
                                        );
                                        assert_eq!(
                                            actual.is_sign_negative(),
                                            expected.is_sign_negative()
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn even_two_three_adapters_preserve_scalar_and_block_bits() {
    even_adapter_bits::<f32, 1>();
    even_adapter_bits::<f32, WIDTH>();
    even_adapter_bits::<f64, 1>();
    even_adapter_bits::<f64, WIDTH>();
}
