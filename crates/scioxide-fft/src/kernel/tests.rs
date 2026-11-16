use super::*;
use alloc::vec;
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::KernelBackend;

const LENGTH: Argument = scioxide_core::argument!("n");

fn tables<S: RealScalar>(radix: usize, width: usize) -> (Vec<Complex<S>>, Vec<Complex<S>>) {
    let root = |k, n| crate::numeric::reduced_root(k as u128, n as u128, false, LENGTH).unwrap();
    let twiddles = (1..radix)
        .flat_map(|b| (0..width).map(move |k| root(b * k, radix * width)))
        .collect();
    let roots = if matches!(radix, 3 | 5 | 7) {
        (0..radix).map(|k| root(k, radix)).collect()
    } else {
        Vec::new()
    };
    (twiddles, roots)
}

fn stage<S: RealScalar>(dispatch: KernelDispatch, radix: usize, width: usize) -> Stage<S> {
    let (twiddles, roots) = tables(radix, width);
    Stage::new(dispatch, radix, width, twiddles, roots, LENGTH).unwrap()
}

// Any in-range permutation exercises the gather leaves; reversal is simple.
fn plan<S: RealScalar>(dispatch: KernelDispatch, radices: &[usize]) -> StagePlan<S> {
    let mut width = 1;
    let stages = radices
        .iter()
        .map(|&radix| {
            let stage = stage(dispatch, radix, width);
            width *= radix;
            stage
        })
        .collect();
    let permutation = Permutation::new((0..width).rev().collect()).unwrap();
    StagePlan::new(width, permutation, stages, dispatch, LENGTH).unwrap()
}

fn signal<S: RealScalar>(n: usize) -> Vec<Complex<S>> {
    (0..n)
        .map(|i| {
            Complex::new(
                S::try_from_f64_lossy(((i * 17 % 97) as f64 - 48.0) / 64.0).unwrap(),
                S::try_from_f64_lossy(((i * 31 % 97) as f64 - 48.0) / 64.0).unwrap(),
            )
        })
        .collect()
}

fn close<S: RealScalar>(actual: &[Complex<S>], expected: &[Complex<S>]) {
    assert_eq!(actual.len(), expected.len());
    let (error, norm) = actual
        .iter()
        .zip(expected)
        .fold((0.0, 0.0), |(error, norm), (a, e)| {
            let re = (a.re - e.re).try_to_f64_exact().unwrap();
            let im = (a.im - e.im).try_to_f64_exact().unwrap();
            let er = e.re.try_to_f64_exact().unwrap();
            let ei = e.im.try_to_f64_exact().unwrap();
            (error + re * re + im * im, norm + er * er + ei * ei)
        });
    crate::test_support::assert_close(
        (error / norm).sqrt(),
        8.0 * S::EPSILON.try_to_f64_exact().unwrap(),
        "kernel relative L2",
    );
}

/// Gather, middle stages and scaled final stage, as smooth plans execute.
fn transform<S: RealScalar>(
    plan: &StagePlan<S>,
    input: &[Complex<S>],
    inverse: bool,
    scale: S,
) -> Vec<Complex<S>> {
    let mut scratch = vec![Complex::new(S::ZERO, S::ZERO); input.len()];
    let mut output = scratch.clone();
    let completed = plan.gather(input, &mut scratch, inverse, false);
    plan.run(&mut scratch, completed..plan.stage_count() - 1, inverse);
    plan.finish(&scratch, &mut output, inverse, scale);
    output
}

fn plan_parity<S: RealScalar>() {
    let context = scioxide_core::ExecutionContext::sequential_native();
    let scalar = KernelDispatch::for_backend(KernelBackend::Scalar, context.processor()).unwrap();
    // Every radix at widths 1 through 4096, with radix-4 runs at every
    // power-of-four width (fused and constant-width leaves), fused gather
    // pairs and triples, radix-4 pairs at widths without fused leaves, and
    // gather groups and widths that leave a scalar remainder for f32 lanes.
    let chains: [&[usize]; 15] = [
        &[2; 7],
        &[3; 5],
        &[5; 4],
        &[7; 3],
        &[4; 7],
        &[4, 4],
        &[4, 4, 2],
        &[2, 4, 4, 4],
        &[3, 4, 4, 5],
        &[4, 4, 4, 3, 7],
        &[5, 2, 3, 4],
        &[2, 3, 4, 4, 4, 4, 4, 4],
        &[4, 5],
        &[4, 3, 4],
        &[4, 4, 4, 4, 2],
    ];
    for backend in crate::test_support::fft_backends(context.processor()) {
        let dispatch = KernelDispatch::for_backend(backend, context.processor()).unwrap();
        for radices in chains {
            let expected_plan = plan::<S>(scalar, radices);
            let actual_plan = plan::<S>(dispatch, radices);
            let n = radices.iter().product();
            let input = signal::<S>(n);
            for inverse in [false, true] {
                let mut unscaled = vec![Complex::new(S::ZERO, S::ZERO); n];
                let completed = actual_plan.gather(&input, &mut unscaled, inverse, true);
                actual_plan.run(&mut unscaled, completed..actual_plan.stage_count(), inverse);
                close(
                    &unscaled,
                    &transform(&expected_plan, &input, inverse, S::ONE),
                );
                for scale in [S::ONE, S::try_from_f64_lossy(0.125).unwrap()] {
                    close(
                        &transform(&actual_plan, &input, inverse, scale),
                        &transform(&expected_plan, &input, inverse, scale),
                    );
                }
                // Partitions are whole blocks of the range's last stage.
                let end = actual_plan.stage_count();
                if end > 1 {
                    let mut whole = input.clone();
                    let mut chunked = input.clone();
                    actual_plan.run(&mut whole, 0..end - 1, inverse);
                    for chunk in chunked.chunks_exact_mut(actual_plan.block(end - 2)) {
                        actual_plan.run(chunk, 0..end - 1, inverse);
                    }
                    close(&chunked, &whole);
                }
            }
        }
    }
}

#[test]
fn every_backend_matches_scalar_leaves_in_both_precisions_and_directions() {
    plan_parity::<f32>();
    plan_parity::<f64>();
}

fn radix5_summation_order<S: RealScalar>() {
    let context = scioxide_core::ExecutionContext::sequential_native();
    let (_, roots) = tables::<S>(5, 4);
    let two = S::ONE + S::ONE;
    let small = S::EPSILON * S::EPSILON;
    for backend in crate::test_support::fft_backends(context.processor()) {
        let dispatch = KernelDispatch::for_backend(backend, context.processor()).unwrap();
        // Width four puts index zero in a full register on every SIMD lane.
        // Equal symmetric inputs cancel the sine terms. The first addition
        // cancels exactly; regrouping the small final term loses it entirely.
        let plan = plan::<S>(dispatch, &[4, 5]);
        for (first, branch, expected) in [
            (-two * roots[1].re, 1, (two * small) * roots[2].re),
            (-two * roots[2].re, 2, (two * small) * roots[1].re),
            (-two, 0, two * small),
        ] {
            let mut data = vec![Complex::new(S::ZERO, S::ZERO); 20];
            for (values, value) in
                data.chunks_exact_mut(4)
                    .zip([first, S::ONE, small, small, S::ONE])
            {
                values.fill(Complex::new(value, S::ZERO));
            }
            for inverse in [false, true] {
                let mut actual = data.clone();
                plan.run(&mut actual, 1..2, inverse);
                let value = actual[branch * 4];
                let error = ((value.re - expected) / expected)
                    .abs()
                    .try_to_f64_exact()
                    .unwrap();
                crate::test_support::assert_close(
                    error,
                    2.0 * S::EPSILON.try_to_f64_exact().unwrap(),
                    "radix-5 left-associated cancellation",
                );
                assert!(value.im.abs() <= S::EPSILON * expected.abs());
            }
        }
    }
}

#[test]
fn radix5_backends_preserve_scalar_summation_order() {
    radix5_summation_order::<f32>();
    radix5_summation_order::<f64>();
}

#[test]
fn preparation_rejects_invalid_stage_geometry() {
    let dispatch =
        KernelDispatch::for_backend(KernelBackend::Scalar, &scioxide_core::PortableProcessor)
            .unwrap();
    let (twiddles, roots) = tables::<f64>(3, 4);
    for (radix, width, twiddles, roots) in [
        (6, 1, vec![Complex::new(1.0, 0.0); 5], Vec::new()),
        (4, 0, Vec::new(), Vec::new()),
        (4, usize::MAX, Vec::new(), Vec::new()),
        (3, 4, twiddles[..7].to_vec(), roots.clone()),
        (3, 4, twiddles.clone(), Vec::new()),
        (4, 4, twiddles, roots),
    ] {
        assert!(Stage::<f64>::new(dispatch, radix, width, twiddles, roots, LENGTH).is_err());
    }
    let stages = || vec![stage::<f64>(dispatch, 4, 1), stage(dispatch, 4, 4)];
    let permutation = |n: usize| Permutation::new((0..n).collect()).unwrap();
    assert!(StagePlan::new(16, permutation(16), stages(), dispatch, LENGTH).is_ok());
    assert!(StagePlan::new(16, permutation(8), stages(), dispatch, LENGTH).is_err());
    assert!(StagePlan::new(32, permutation(32), stages(), dispatch, LENGTH).is_err());
    let broken = vec![stage::<f64>(dispatch, 4, 1), stage(dispatch, 4, 2)];
    assert!(StagePlan::new(16, permutation(16), broken, dispatch, LENGTH).is_err());
    assert!(Permutation::new(vec![16; 16]).is_err());
}

fn rejects_geometry(call: impl FnOnce()) {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(call));
    let panic = outcome.unwrap_err();
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"invalid FFT kernel geometry")
    );
}

#[test]
fn mismatched_borrows_never_reach_a_leaf_or_change_output() {
    let dispatch =
        KernelDispatch::for_backend(KernelBackend::Scalar, &scioxide_core::PortableProcessor)
            .unwrap();
    let plan = plan::<f64>(dispatch, &[4, 4, 2]);
    let input = signal::<f64>(32);
    let short = signal::<f64>(16);
    let mut output = input.clone();
    rejects_geometry(|| {
        plan.gather(&short, &mut output, false, false);
    });
    rejects_geometry(|| plan.run(&mut output[..12], 0..2, false));
    rejects_geometry(|| plan.run(&mut output, 0..4, false));
    rejects_geometry(|| plan.run(&mut output, Range { start: 2, end: 1 }, false));
    rejects_geometry(|| plan.finish(&short, &mut output, false, 1.0));
    rejects_geometry(|| plan.finish(&input, &mut output[..16], false, 1.0));
    rejects_geometry(|| plan.finish(&input, &mut output, false, f64::NAN));
    rejects_geometry(|| {
        plan.transform_lanes(
            &mut output,
            crate::kernel::block::Run {
                first: 0,
                step: 1,
                count: 1,
                len: 16,
                stride: 1,
            },
            (false, 1.0),
            &mut rows::<f64>(),
        );
    });
    assert_eq!(output, input);
}

#[test]
fn finish_requires_a_stage_even_for_identity_plans() {
    let dispatch =
        KernelDispatch::for_backend(KernelBackend::Scalar, &scioxide_core::PortableProcessor)
            .unwrap();
    let identity = plan::<f64>(dispatch, &[]);
    let mut output = [Complex::new(17.0, -3.0)];
    rejects_geometry(|| identity.finish(&[Complex::new(1.0, 0.0)], &mut output, false, 1.0));
    assert_eq!(output, [Complex::new(17.0, -3.0)]);
}

fn bounded_gather_contract<S: RealScalar>() {
    let context = scioxide_core::ExecutionContext::sequential_native();
    for backend in crate::test_support::fft_backends(context.processor()) {
        let dispatch = KernelDispatch::for_backend(backend, context.processor()).unwrap();
        // Permutation, scalar gather, vector gather with a tail, fused pairs
        // and triples all read the original bounded values exactly once.
        for radices in [&[2][..], &[5, 7], &[4, 3], &[4, 4], &[4, 4, 4]] {
            let plan = plan::<S>(dispatch, radices);
            for include_last in [false, true] {
                for inverse in [false, true] {
                    let input = signal::<S>(plan.n);
                    let mut expected = input.clone();
                    let mut actual = input.clone();
                    let completed = plan.gather(&input, &mut expected, inverse, include_last);
                    assert_eq!(
                        plan.gather_bounded(&input, &mut actual, inverse, include_last),
                        completed
                    );
                    close(&actual, &expected);
                    for index in [0, plan.n / 2, plan.n - 1] {
                        for bad in [
                            S::NAN,
                            S::INFINITY,
                            -S::INFINITY,
                            S::ONE + S::EPSILON,
                            -S::ONE - S::EPSILON,
                        ] {
                            for imaginary in [false, true] {
                                let mut input = input.clone();
                                input[index] = if imaginary {
                                    Complex::new(S::ZERO, bad)
                                } else {
                                    Complex::new(bad, S::ZERO)
                                };
                                assert!(
                                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                        plan.gather_bounded(
                                            &input,
                                            &mut actual,
                                            inverse,
                                            include_last,
                                        );
                                    }))
                                    .is_err(),
                                    "{backend:?} index={index}"
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
fn every_gather_leaf_enforces_bounded_inputs_in_both_widths() {
    bounded_gather_contract::<f32>();
    bounded_gather_contract::<f64>();
}

fn odd_codelet_reference<S: RealScalar>() {
    let context = scioxide_core::ExecutionContext::sequential_native();
    for backend in crate::test_support::fft_backends(context.processor()) {
        let dispatch = KernelDispatch::for_backend(backend, context.processor()).unwrap();
        for radix in [5, 7] {
            let plan = StagePlan::new(
                radix,
                Permutation::new((0..radix).collect()).unwrap(),
                vec![stage::<S>(dispatch, radix, 1)],
                dispatch,
                LENGTH,
            )
            .unwrap();
            for impulse in 0..=radix {
                let input: Vec<_> = (0..radix)
                    .map(|k| {
                        if impulse == radix {
                            Complex::new(
                                if k.is_multiple_of(2) { S::ONE } else { -S::ONE },
                                S::HALF,
                            )
                        } else {
                            Complex::new(if k == impulse { S::ONE } else { S::ZERO }, S::ZERO)
                        }
                    })
                    .collect();
                let reference: Vec<_> = input
                    .iter()
                    .map(|z| {
                        Complex::new(
                            z.re.try_to_f64_exact().unwrap(),
                            z.im.try_to_f64_exact().unwrap(),
                        )
                    })
                    .collect();
                for inverse in [false, true] {
                    let expected: Vec<_> = crate::test_support::direct_dft(&reference, inverse)
                        .into_iter()
                        .map(|z| {
                            Complex::new(
                                S::try_from_f64_lossy(z.re).unwrap(),
                                S::try_from_f64_lossy(z.im).unwrap(),
                            )
                        })
                        .collect();
                    let mut actual = input.clone();
                    plan.gather_bounded(&input, &mut actual, inverse, true);
                    close(&actual, &expected);
                }
            }
        }
    }
}

#[test]
fn symmetric_odd_codelets_match_independent_defining_sums() {
    odd_codelet_reference::<f32>();
    odd_codelet_reference::<f64>();
}

fn bounded_tiles<S: RealScalar>() {
    let context = scioxide_core::ExecutionContext::sequential_native();
    for backend in crate::test_support::fft_backends(context.processor()) {
        let dispatch = KernelDispatch::for_backend(backend, context.processor()).unwrap();
        for radices in [&[][..], &[4, 2], &[5, 7], &[3, 3, 5, 5]] {
            let plan = plan::<S>(dispatch, radices);
            for (step, stride) in [(1, 3), (plan.n, 1)] {
                let run = crate::kernel::block::Run {
                    first: 0,
                    step,
                    count: 3,
                    len: plan.n,
                    stride,
                };
                for inverse in [false, true] {
                    for bad in [S::NAN, S::INFINITY, S::ONE + S::EPSILON] {
                        let mut values = signal::<S>(3 * plan.n);
                        let index = 2 * step + (plan.n - 1) * stride;
                        values[index].im = bad;
                        assert!(
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                plan.transform_lanes(
                                    &mut values,
                                    run,
                                    (inverse, S::ONE),
                                    &mut rows::<S>(),
                                );
                            }))
                            .is_err(),
                            "{backend:?} n={}",
                            plan.n
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn bounded_tiles_admit_the_last_partial_lane_in_both_widths() {
    bounded_tiles::<f32>();
    bounded_tiles::<f64>();
}

fn rows<S: RealScalar>() -> [block::Block<S, { block::WIDTH }>; block::ROWS] {
    let zero = block::Reals::splat(S::ZERO);
    [block::Block::new(zero, zero); block::ROWS]
}

fn tile_parity<S: RealScalar>() {
    let context = scioxide_core::ExecutionContext::sequential_native();
    // Smooth lengths through the across-lane codelets, with 11 lanes (two
    // whole blocks and a partial one), interleaved and lane-major.
    let chains: [&[usize]; 18] = [
        &[2],
        &[3],
        &[4],
        &[5],
        &[3, 2],
        &[7],
        &[4, 2],
        &[3, 3],
        &[5, 2],
        &[4, 3],
        &[7, 2],
        &[5, 3],
        &[4, 4],
        &[2, 2, 2, 2],
        &[4, 4, 2],
        &[4, 4, 4],
        &[5, 5, 3, 3],
        &[4, 4, 4, 4],
    ];
    let lanes = 11;
    for backend in crate::test_support::fft_backends(context.processor()) {
        let dispatch = KernelDispatch::for_backend(backend, context.processor()).unwrap();
        for radices in chains {
            let plan = plan::<S>(dispatch, radices);
            let n: usize = radices.iter().product();
            assert!(n <= block::ROWS);
            for (step, stride) in [(1, lanes), (n, 1)] {
                let values = signal::<S>(lanes * n);
                let run = crate::kernel::block::Run {
                    first: 0,
                    step,
                    count: lanes,
                    len: n,
                    stride,
                };
                for inverse in [false, true] {
                    for scale in [S::ONE, S::try_from_f64_lossy(0.125).unwrap()] {
                        let mut tiled = values.clone();
                        plan.transform_lanes(&mut tiled, run, (inverse, scale), &mut rows());
                        // Architecture builds of the portable body give identical bits.
                        let mut portable = values.clone();
                        super::tile::transform_run(
                            &plan,
                            &mut portable,
                            run,
                            (inverse, scale),
                            &mut rows(),
                        );
                        assert_eq!(tiled, portable);
                        for lane in 0..lanes {
                            let at = |i: usize| lane * step + i * stride;
                            let input: Vec<_> = (0..n).map(|i| values[at(i)]).collect();
                            let actual: Vec<_> = (0..n).map(|i| tiled[at(i)]).collect();
                            close(&actual, &transform(&plan, &input, inverse, scale));
                        }
                    }
                }
            }
        }
        // Length one is the scaled identity.
        let plan = plan::<S>(dispatch, &[]);
        let mut values = signal::<S>(3);
        let expected: Vec<_> = values.iter().map(|&v| v * S::HALF).collect();
        let run = crate::kernel::block::Run {
            first: 0,
            step: 1,
            count: 3,
            len: 1,
            stride: 1,
        };
        plan.transform_lanes(&mut values, run, (false, S::HALF), &mut rows());
        assert_eq!(values, expected);
    }
}

#[test]
fn tiles_match_single_lane_leaves_on_every_backend() {
    tile_parity::<f32>();
    tile_parity::<f64>();
}

fn native_batch_bits<S: RealScalar>() {
    let context = scioxide_core::ExecutionContext::sequential_native();
    let chains: [&[usize]; 20] = [
        &[],
        &[2],
        &[3],
        &[4],
        &[5],
        &[7],
        &[3, 2],
        &[4, 4],
        &[4, 4, 2],
        &[4, 4, 4],
        &[5, 5, 3, 3],
        &[4, 4, 4, 4],
        &[4, 3, 2],
        &[4, 5, 2],
        &[4, 7, 2],
        &[3, 4, 4],
        &[2, 4, 4],
        &[4, 4, 3],
        &[3, 3, 3],
        &[7, 3, 2],
    ];
    for backend in crate::test_support::fft_backends(context.processor()) {
        let dispatch = KernelDispatch::for_backend(backend, context.processor()).unwrap();
        for chain in chains {
            let plan = plan::<S>(dispatch, chain);
            for count in [1, 3, 4, 5] {
                let mut input = signal::<S>(count * plan.n);
                input[0] = Complex::new(-S::ZERO, S::MIN_SUBNORMAL);
                if count > 1 {
                    input[plan.n..2 * plan.n].fill(Complex::new(-S::ZERO, S::ZERO));
                }
                for inverse in [false, true] {
                    for scale in [S::ONE, S::HALF] {
                        let mut actual = input.clone();
                        let mut scratch = vec![Complex::new(S::ZERO, S::ZERO); input.len()];
                        plan.transform_batch(&mut actual, &mut scratch, inverse, scale);
                        if backend != KernelBackend::Neon {
                            for first in (0..count).step_by(block::WIDTH) {
                                let mut rows = rows();
                                for (row, &index) in rows.iter_mut().zip(plan.permutation.iter()) {
                                    *row = block::Block::from_fn(|lane| {
                                        input[(first + lane).min(count - 1) * plan.n + index]
                                    });
                                }
                                super::tile::transform_rows(&plan, &mut rows, inverse, true);
                                for lane in 0..block::WIDTH.min(count - first) {
                                    for (i, row) in rows[..plan.n].iter().enumerate() {
                                        let b = if plan.n == 1 {
                                            crate::scaling::scale_singleton(row.lane(lane), scale)
                                        } else {
                                            row.lane(lane) * scale
                                        };
                                        let a = actual[(first + lane) * plan.n + i];
                                        for (a, b) in [(a.re, b.re), (a.im, b.im)] {
                                            assert_eq!(
                                                a.try_to_f64_exact().unwrap().to_bits(),
                                                b.try_to_f64_exact().unwrap().to_bits(),
                                                "native rows {backend:?} {chain:?} inverse={inverse} lane={lane} index={i} scale={scale:?}"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        let mut digit = vec![Complex::new(S::ZERO, S::ZERO); input.len()];
                        for (from, to) in input
                            .chunks_exact(plan.n)
                            .zip(digit.chunks_exact_mut(plan.n))
                        {
                            for (to, &index) in to.iter_mut().zip(plan.permutation.iter()) {
                                *to = from[index];
                            }
                        }
                        plan.transform_staged_batch(&mut digit, &mut scratch, inverse, scale);
                        digit.copy_from_slice(&scratch);
                        for (a, b) in actual.iter().zip(&digit) {
                            for (a, b) in [(a.re, b.re), (a.im, b.im)] {
                                assert_eq!(
                                    a.try_to_f64_exact().unwrap().to_bits(),
                                    b.try_to_f64_exact().unwrap().to_bits(),
                                    "digit {backend:?} {chain:?} inverse={inverse}"
                                );
                            }
                        }
                        for (source, actual) in
                            input.chunks_exact(plan.n).zip(actual.chunks_exact(plan.n))
                        {
                            let expected = if plan.n == 1 {
                                vec![crate::scaling::scale_singleton(source[0], scale)]
                            } else {
                                transform(&plan, source, inverse, scale)
                            };
                            for (a, b) in actual.iter().zip(expected) {
                                for (a, b) in [(a.re, b.re), (a.im, b.im)] {
                                    assert_eq!(
                                        a.try_to_f64_exact().unwrap().to_bits(),
                                        b.try_to_f64_exact().unwrap().to_bits(),
                                        "{backend:?} {chain:?} {count} inverse={inverse}"
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

#[test]
fn native_batches_preserve_single_lane_bits() {
    native_batch_bits::<f32>();
    native_batch_bits::<f64>();
}

fn nufft_rows<S: RealScalar>() {
    let context = scioxide_core::ExecutionContext::sequential_native();
    let scalar = KernelDispatch::for_backend(KernelBackend::Scalar, context.processor()).unwrap();
    let convert = |x| S::try_from_f64_lossy(x).unwrap();
    let value = Complex::new(convert(0.75), convert(-0.375));
    let guard = Complex::new(convert(17.0), convert(-23.0));
    for backend in KernelBackend::available(context.processor()) {
        let dispatch = KernelDispatch::for_backend(backend, context.processor()).unwrap();
        for width in 4..19 {
            let side = width + 7;
            let weights: Vec<_> = (0..3 * width)
                .map(|j| Complex::new(convert((j % 7 + 1) as f64 / 8.0), S::NAN))
                .collect();
            let mut expected = vec![Complex::ZERO; side * side * side];
            let mut actual = expected.clone();
            for axes in [
                [(0, width); 3],
                [(7, width); 3],
                [(7, 1), (0, width), (7, 1)],
            ] {
                nufft_tile(scalar, &mut expected, &weights, value, side, axes);
                nufft_tile(dispatch, &mut actual, &weights, value, side, axes);
            }
            for (a, b) in actual.iter().zip(expected) {
                assert!((*a - b).norm() <= convert(64.0) * S::EPSILON);
            }
        }
        for len in 0..35 {
            let weights: Vec<_> = (0..len)
                .map(|j| Complex::new(convert((j % 7 + 1) as f64 / 8.0), S::NAN))
                .collect();
            {
                let mut expected = vec![guard; len + 2];
                let mut actual = expected.clone();
                let mut ec = vec![Complex::ZERO; len];
                let mut ac = ec.clone();
                for sign in [1.0, -1.0, 0.5] {
                    nufft_spread(
                        scalar,
                        &mut expected[1..len + 1],
                        &mut ec,
                        &weights,
                        value * convert(sign),
                    );
                    nufft_spread(
                        dispatch,
                        &mut actual[1..len + 1],
                        &mut ac,
                        &weights,
                        value * convert(sign),
                    );
                }
                assert_eq!(actual[0], guard);
                assert_eq!(actual[len + 1], guard);
                for (a, b) in actual.iter().zip(&expected).chain(ac.iter().zip(&ec)) {
                    assert!((*a - *b).norm() <= convert(64.0) * S::EPSILON);
                }
            }
        }
    }
}

#[test]
fn nufft_rows_cover_native_tails_and_compensated_residuals() {
    nufft_rows::<f32>();
    nufft_rows::<f64>();
}

#[test]
fn nufft_native_interpolation_wraps_and_preserves_cancellation() {
    let context = scioxide_core::ExecutionContext::sequential_native();
    for backend in KernelBackend::available(context.processor()) {
        let dispatch = KernelDispatch::for_backend(backend, context.processor()).unwrap();
        for width in 4..19 {
            let n = width + 3;
            let grid: Vec<_> = (0..n * n * n)
                .map(|j| {
                    Complex::new(
                        if j % 3 == 0 { 1e4_f64 } else { -0.125 },
                        (j % 19) as f64 - 9.0,
                    )
                })
                .collect();
            let weights: Vec<_> = (0..3 * width)
                .map(|j| Complex::new((j % 7 + 1) as f64 / 8.0, f64::NAN))
                .collect();
            for starts in [[0, 0, 0], [n - 1, n - 2, n - 1], [n / 2; 3]] {
                for supports in [[width; 3], [1, width, 1]] {
                    let axes = core::array::from_fn(|d| (starts[d], n, supports[d]));
                    let Some(actual) = nufft_interpolate3(dispatch, &grid, &weights, axes, width)
                    else {
                        continue;
                    };
                    let (mut sum, mut correction) =
                        (Complex::new(0.0, 0.0), Complex::new(0.0, 0.0));
                    let mut mass = 0.0;
                    for i in 0..supports[0] {
                        for j in 0..supports[1] {
                            for k in 0..supports[2] {
                                let index = (((starts[0] + i) % n) * n + (starts[1] + j) % n) * n
                                    + (starts[2] + k) % n;
                                let term = grid[index]
                                    * (weights[i].re
                                        * weights[width + j].re
                                        * weights[2 * width + k].re);
                                mass += term.norm();
                                crate::numeric::compensated_term(
                                    &mut sum.re,
                                    &mut correction.re,
                                    term.re,
                                );
                                crate::numeric::compensated_term(
                                    &mut sum.im,
                                    &mut correction.im,
                                    term.im,
                                );
                            }
                        }
                    }
                    assert!((actual - (sum + correction)).norm() <= 16.0 * f64::EPSILON * mass);
                }
            }
        }
    }
}
