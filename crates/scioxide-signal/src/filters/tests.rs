#![allow(clippy::unwrap_used)]
use super::*;
use crate::util::error_kind;
use scioxide_core::SciOxideErrorKind;

fn native() -> ExecutionContext<'static> {
    ExecutionContext::sequential_native()
}

fn regression<S: RealScalar>(input: &[S], radius: usize, kind: LocalFilter<S>) {
    let plan = LocalFilterPlan::new(radius, kind).unwrap();
    let mut actual = alloc::vec![S::ZERO; input.len()];
    let mut indices = alloc::vec![usize::MAX; input.len()];
    let mut scratch = plan.workspace(input.len()).unwrap();
    let count = plan
        .filter_into(input, &mut actual, &mut indices, &mut scratch, native())
        .unwrap();
    let mut expected_indices = Vec::new();
    for (i, (&x, &y)) in input.iter().zip(&actual).enumerate() {
        let mut window: Vec<_> = plan
            .offsets
            .iter()
            .map(|&o| input[clamped_index(i, o, input.len())])
            .collect();
        let expected = plan.reference(x, &mut window).unwrap();
        if matches!(kind, LocalFilter::Wiener { .. }) {
            let scale = input.iter().fold(S::ZERO, |m, v| m.max(v.abs()));
            assert!(
                (y - expected).abs()
                    <= S::try_from_f64_lossy(512.).unwrap() * S::EPSILON * scale
                        + S::MIN_SUBNORMAL * S::try_from_f64_lossy(32.).unwrap(),
                "{i}: {y:?} != {expected:?}"
            );
        } else {
            assert_eq!(y, expected, "{i}");
            assert_eq!(
                y.is_sign_negative(),
                expected.is_sign_negative(),
                "zero sign at {i}"
            );
        }
        if matches!(kind, LocalFilter::Hampel { .. }) && expected != x {
            expected_indices.push(i);
        }
    }
    assert_eq!(&indices[..count], expected_indices);
}

fn cases<S: RealScalar>() {
    let s = |v| S::try_from_f64_lossy(v).unwrap();
    let random: Vec<_> = (0..257)
        .map(|i| s((((i * 37 + 11) % 113) as f64 - 56.) / 19.))
        .collect();
    let ties: Vec<_> = (0..71)
        .map(|i| [S::NEG_ZERO, S::ZERO, S::ONE, -S::ONE, S::TWO][i % 5])
        .collect();
    let extremes = [
        S::MAX,
        -S::MAX,
        S::MIN_SUBNORMAL,
        -S::MIN_SUBNORMAL,
        S::ONE,
        S::ZERO,
        S::NEG_ZERO,
    ];
    for input in [
        &random[..],
        &ties,
        &extremes,
        &[S::ONE; 3],
        &[S::MAX; 5],
        &[-S::MAX; 5],
        &[S::MIN_SUBNORMAL; 5],
        &[S::ZERO, S::NEG_ZERO],
        &[],
    ] {
        for radius in [0, 1, 2, 8, 17, 40] {
            for kind in [
                LocalFilter::Median,
                LocalFilter::Rank { rank: 0 },
                LocalFilter::Rank { rank: 2 * radius },
                LocalFilter::Hampel { threshold: S::ZERO },
                LocalFilter::Hampel { threshold: s(3.) },
                LocalFilter::Wiener {
                    noise_variance: s(0.1),
                },
                LocalFilter::Wiener {
                    noise_variance: S::ZERO,
                },
            ] {
                regression(input, radius, kind);
            }
        }
    }
}

#[test]
fn windows_match_original_definition() {
    cases::<f32>();
    cases::<f64>();
}

#[test]
fn sparse_offsets_all_ranks() {
    let input = [3_f64, -0., 0., -1., 8., 8., -2., 5.];
    for offsets in [&[-4, -2, 0, 3][..], &[1, 2, 5], &[-5, -3, -2], &[0]] {
        for rank in 0..offsets.len() {
            let plan = LocalFilterPlan::order(offsets, rank).unwrap();
            let mut out = [0.; 8];
            let mut scratch = plan.workspace(input.len()).unwrap();
            plan.filter_into(&input, &mut out, &mut [], &mut scratch, native())
                .unwrap();
            for (i, &y) in out.iter().enumerate() {
                let mut window: Vec<_> = offsets
                    .iter()
                    .map(|&o| input[clamped_index(i, o, input.len())])
                    .collect();
                let expected: f64 = plan.reference(input[i], &mut window).unwrap();
                assert_eq!(y.to_bits(), expected.to_bits());
            }
        }
    }
}

#[test]
fn many_run_footprints_match_independent_windows_in_batches_and_streams() {
    // Footprints with more offset runs than log2(w) re-sort every window.
    let input: Vec<f64> = (0..300_u32)
        .map(|i| f64::from((i * 7919) % 23) - 11.0 + if i % 5 == 0 { -0.0 } else { 0.25 })
        .collect();
    let every_other: Vec<isize> = (-20..=20).step_by(2).collect();
    let every_third: Vec<isize> = (-30..=30).step_by(3).collect();
    let mixed = [-17, -16, -11, -9, -8, -3, 0, 2, 5, 6, 7, 13, 21];
    for offsets in [&every_other[..], &every_third, &mixed] {
        for rank in [0, offsets.len() / 3, offsets.len() / 2, offsets.len() - 1] {
            let plan = LocalFilterPlan::order(offsets, rank).unwrap();
            assert!(plan.resort);
            let mut out = alloc::vec![0.; input.len()];
            let mut scratch = plan.workspace(input.len()).unwrap();
            plan.filter_into(&input, &mut out, &mut [], &mut scratch, native())
                .unwrap();
            for (i, &y) in out.iter().enumerate() {
                let mut window: Vec<_> = offsets
                    .iter()
                    .map(|&o| input[clamped_index(i, o, input.len())])
                    .collect();
                let expected: f64 = plan.reference(input[i], &mut window).unwrap();
                assert_eq!(
                    y.to_bits(),
                    expected.to_bits(),
                    "offsets {offsets:?} rank {rank}"
                );
            }
            chunks(&input, plan);
        }
    }
    assert!(
        !LocalFilterPlan::<f64>::new(20, LocalFilter::Median)
            .unwrap()
            .resort
    );
}

#[test]
fn hampel_threshold_neighbors_match_extended_decision() {
    // At the central window median=0 and MAD=1; move the multiplier through
    // the rounded equality boundary without changing either order statistic.
    let input = [-3., -1., 2., 0., 1.];
    let boundary = 2. / 1.482602218505602_f64;
    for threshold in [
        f64::from_bits(boundary.to_bits() - 1),
        boundary,
        f64::from_bits(boundary.to_bits() + 1),
        f64::MAX,
    ] {
        regression(&input, 2, LocalFilter::Hampel { threshold });
    }
    let input = [
        f64::MIN_POSITIVE,
        0.,
        f64::MIN_POSITIVE * 2.,
        f64::from_bits(1),
        -f64::MIN_POSITIVE,
    ];
    for threshold in [0., f64::from_bits(1), 1., 3., f64::MAX] {
        regression(&input, 2, LocalFilter::Hampel { threshold });
    }
}

fn chunks<S: RealScalar>(input: &[S], plan: LocalFilterPlan<S>) {
    let mut batch = alloc::vec![S::ZERO; input.len()];
    let mut indices = alloc::vec![0; input.len()];
    let mut scratch = plan.workspace(input.len()).unwrap();
    let changed = plan
        .filter_into(input, &mut batch, &mut indices, &mut scratch, native())
        .unwrap();
    for chunk_size in [1, 2, 7, 19, input.len() + 1] {
        let mut stream = plan.stream().unwrap();
        let mut scratch = stream.workspace(chunk_size).unwrap();
        let mut results = Vec::new();
        let mut replacements = Vec::new();
        for chunk in input.chunks(chunk_size).chain(core::iter::once(&[][..])) {
            let finish = if chunk.is_empty() {
                StreamChunk::Final
            } else {
                StreamChunk::Partial
            };
            let count = stream.output_len(chunk.len(), finish).unwrap();
            let mut out = alloc::vec![S::TWO; count + 1];
            let mut indices = alloc::vec![usize::MAX; count + 1];
            let progress = stream
                .process_into(
                    chunk,
                    finish,
                    &mut out,
                    &mut indices,
                    &mut scratch,
                    native(),
                )
                .unwrap();
            assert_eq!(progress.first_output_index, results.len());
            assert_eq!(out[count], S::TWO);
            assert_eq!(indices[count], usize::MAX);
            results.extend_from_slice(&out[..count]);
            replacements.extend_from_slice(&indices[..progress.replacements]);
        }
        assert_eq!(results, batch);
        for (a, b) in results.iter().zip(&batch) {
            assert_eq!(a.is_sign_negative(), b.is_sign_negative());
        }
        assert_eq!(replacements, indices[..changed]);
        stream.reset();
        assert_eq!(
            stream.output_len(input.len(), StreamChunk::Final).unwrap(),
            input.len()
        );
    }
}

#[test]
fn persistent_state_is_independent_of_chunking_and_flush() {
    let input: Vec<_> = (0..91)
        .map(|i| match i % 17 {
            0 => -0.,
            1 => 0.,
            _ => ((i * 47 % 51) as f64 - 25.) / 3.,
        })
        .collect();
    for kind in [
        LocalFilter::Median,
        LocalFilter::Rank { rank: 2 },
        LocalFilter::Hampel { threshold: 3. },
        LocalFilter::Wiener {
            noise_variance: 0.1,
        },
    ] {
        chunks(&input, LocalFilterPlan::new(8, kind).unwrap());
    }
    chunks(
        &input,
        LocalFilterPlan::order(&[-8, -5, -4, 1, 7], 2).unwrap(),
    );
    chunks(
        &input[..3],
        LocalFilterPlan::new(8, LocalFilter::Median).unwrap(),
    );
    let extreme = [
        f32::MAX,
        1.,
        f32::MIN_POSITIVE,
        -f32::MAX,
        2.,
        2.,
        2.,
        2.,
        2.,
    ];
    chunks(
        &extreme,
        LocalFilterPlan::new(
            1,
            LocalFilter::Wiener {
                noise_variance: 0.1,
            },
        )
        .unwrap(),
    );
    chunks(
        &extreme,
        LocalFilterPlan::new(2, LocalFilter::Hampel { threshold: 3. }).unwrap(),
    );
}

#[test]
fn failed_chunk_preserves_incremental_history_and_indices() {
    for kind in [
        LocalFilter::Hampel { threshold: 3. },
        LocalFilter::Wiener {
            noise_variance: 0.1,
        },
    ] {
        let plan = LocalFilterPlan::new(1, kind).unwrap();
        let mut stream = plan.stream().unwrap();
        let mut scratch = stream.workspace(2).unwrap();
        let mut output = [42.; 3];
        let mut indices = [99; 3];
        stream
            .process_into(
                &[1., 1.],
                StreamChunk::Partial,
                &mut output,
                &mut indices,
                &mut scratch,
                native(),
            )
            .unwrap();
        output.fill(42.);
        let history = stream.ring.to_vec();
        let order = stream.sorted.to_vec();
        let emitted = stream.emitted;
        assert_eq!(
            error_kind(stream.process_into(
                &[f64::NAN],
                StreamChunk::Partial,
                &mut output,
                &mut indices,
                &mut scratch,
                native()
            )),
            SciOxideErrorKind::NonFinite
        );
        assert_eq!(stream.ring, history);
        assert_eq!(stream.sorted, order);
        assert_eq!(stream.emitted, emitted);
        assert_eq!(output, [42.; 3]);
        assert_eq!(indices, [99; 3]);
        if matches!(kind, LocalFilter::Hampel { .. }) {
            assert_eq!(
                error_kind(stream.process_into(
                    &[9., 1.],
                    StreamChunk::Final,
                    &mut output,
                    &mut [],
                    &mut scratch,
                    native()
                )),
                SciOxideErrorKind::DimensionMismatch
            );
            assert_eq!(stream.ring, history);
            assert_eq!(stream.sorted, order);
            assert_eq!(output, [42.; 3]);
            assert_eq!(stream.emitted, emitted);
        }
        let result = stream
            .process_into(
                &[9., 1.],
                StreamChunk::Final,
                &mut output,
                &mut indices,
                &mut scratch,
                native(),
            )
            .unwrap();
        assert_eq!(result.produced, 3);
    }
}

#[test]
fn bounded_moments_recover_after_extreme_and_track_noise() {
    for (offset, amplitude) in [(0., 1.), (1e12, 1.), (0., 1e-140)] {
        let mut input: Vec<_> = (0..129)
            .map(|i| offset + amplitude * ((i * 31 % 23) as f64 - 11.))
            .collect();
        input[1] = f64::MAX;
        let plan = LocalFilterPlan::new(
            2,
            LocalFilter::Wiener {
                noise_variance: 0.1,
            },
        )
        .unwrap();
        let mut out = vec![0.; input.len()];
        let mut work = plan.workspace(input.len()).unwrap();
        plan.filter_into(&input, &mut out, &mut [], &mut work, native())
            .unwrap();
        for (i, &value) in out.iter().enumerate().skip(5) {
            let mut window: Vec<_> = (-2..=2)
                .map(|o| input[clamped_index(i, o, input.len())])
                .collect();
            let expected = plan.reference(input[i], &mut window).unwrap();
            let scale = window.iter().fold(0_f64, |m, x| m.max(x.abs()));
            assert!(
                (value - expected).abs() <= 512. * f64::EPSILON * scale + 32. * f64::from_bits(1)
            );
        }
        input[1] = offset;
        let mut reference = ScaledAccumulator::ZERO;
        for i in 0..input.len() {
            let (_, variance) = local_moments(
                (-2..=2).map(|o| input[clamped_index(i, o, input.len())] - offset),
                5,
            )
            .unwrap();
            reference.add_scaled(variance).unwrap();
        }
        let expected = reference
            .total()
            .unwrap()
            .div(Scaled::from_value(input.len() as f64).unwrap())
            .unwrap()
            .round()
            .unwrap();
        let actual = estimate_wiener_noise(&input, 2).unwrap();
        assert!(
            (actual - expected).abs()
                <= 128. * f64::EPSILON * expected.abs() + 32. * f64::from_bits(1),
            "offset={offset} amplitude={amplitude}: {actual} != {expected}"
        );
    }
    assert_eq!(estimate_wiener_noise(&[7.; 19], 8).unwrap(), 0.);
    assert_eq!(
        error_kind(estimate_wiener_noise::<f64>(&[], 0)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error_kind(estimate_wiener_noise(&[f64::NAN], 0)),
        SciOxideErrorKind::NonFinite
    );
}

#[test]
fn every_centered_rank_matches_selection() {
    let input: Vec<_> = (0..71).map(|i| ((i * 31 % 37) as f64 - 18.) / 3.).collect();
    for rank in 0..17 {
        regression(&input, 8, LocalFilter::Rank { rank });
    }
    assert_eq!(estimate_wiener_noise(&[f64::MAX; 7], 8).unwrap(), 0.);
}

#[test]
fn storage_queries_and_atomic_batch_validation() {
    let plan = LocalFilterPlan::new(1, LocalFilter::Hampel { threshold: 3. }).unwrap();
    assert_eq!(
        error_kind(plan.workspace_requirements(usize::MAX)),
        SciOxideErrorKind::SizeOverflow
    );
    let mut output = [42.; 4];
    let mut indices = [99; 4];
    let mut short = plan.workspace(2).unwrap();
    assert_eq!(
        plan.filter_into(
            &[1., 9., 1.],
            &mut output,
            &mut indices,
            &mut short,
            native()
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    let mut work = plan.workspace(3).unwrap();
    assert_eq!(
        error_kind(plan.filter_into(
            &[1., f64::INFINITY, 1.],
            &mut output,
            &mut indices,
            &mut work,
            native()
        )),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(
        error_kind(plan.filter_into(
            &[1., 9., 1.],
            &mut output[..2],
            &mut indices,
            &mut work,
            native()
        )),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        error_kind(plan.filter_into(&[1., 9., 1.], &mut output, &mut [], &mut work, native())),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, [42.; 4]);
    assert_eq!(indices, [99; 4]);
    let count = plan
        .filter_into(
            &[1., 9., 1.],
            &mut output,
            &mut indices,
            &mut work,
            native(),
        )
        .unwrap();
    assert_eq!(count, 1);
    assert_eq!(output, [1., 1., 1., 42.]);
    assert_eq!(indices, [1, 99, 99, 99]);
}

#[test]
fn finite_bit_patterns_preserve_order_and_hampel_rounding() {
    let mut seed = 0x29a7_53d9_18cf_604b_u64;
    let input: Vec<_> = (0..193)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let magnitude = (seed >> 1) % 0x7ff0_0000_0000_0000;
            f64::from_bits(magnitude | (seed & (1 << 63)))
        })
        .collect();
    for kind in [
        LocalFilter::Median,
        LocalFilter::Rank { rank: 3 },
        LocalFilter::Hampel { threshold: 3. },
    ] {
        regression(&input, 8, kind);
    }
    let input = [-3_f32, -1., 2., 0., 1.];
    let boundary = 2. / 1.482_602_2_f32;
    for threshold in [
        f32::from_bits(boundary.to_bits() - 1),
        boundary,
        f32::from_bits(boundary.to_bits() + 1),
    ] {
        regression(&input, 2, LocalFilter::Hampel { threshold });
    }
}
