//! Shared cosine/sine mappings, minimal storage, and tensor composition.
#![expect(clippy::unwrap_used)]

#[path = "support/numerics.rs"]
mod numerics;

use scioxide_core::{
    Capacity, Complex, ExecutionContext, NdView, NdViewMut, RealScalar, SciOxideErrorKind,
    StridedLayout,
};
use scioxide_fft::*;

#[test]
fn odd_decimated_real_workspace_follows_the_recursive_formula() {
    // R(5) = 5 + A(5); R(15) = 8 + 5 + R(5) + A(5); R(45) = 23 + 15 + R(15) + A(15).
    for (n, required) in [(5, 10), (15, 28), (45, 81), (49, 25 + 21 + 14 + 7)] {
        let plan = FftPlan::<f64>::new(FftPlanSpec::new(
            n,
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Real,
        ))
        .unwrap();
        assert_eq!(
            plan.workspace_requirements(()).unwrap().complex_len(),
            required,
            "n={n}"
        );
    }
}

#[test]
fn native_workspace_formulas_cover_size_boundaries() {
    for (n, algorithm) in [
        (4, 2),
        (16, 8),
        (32, 16),
        (64, 32),
        (4096, 2048),
        (8192, 4096),
    ] {
        let real = FftPlan::<f64>::new(FftPlanSpec::new(
            n,
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Real,
        ))
        .unwrap();
        assert_eq!(
            real.workspace_requirements(()).unwrap().complex_len(),
            n / 2 + 1 + algorithm
        );
        for kind in [DctType::II, DctType::III, DctType::IV] {
            let plan = RealTransformPlan::<f64>::new(RealTransformPlanSpec::new(
                n,
                kind.into(),
                FftDirection::Forward,
                Normalization::Orthonormal,
            ))
            .unwrap();
            let required = n + n / 2 + algorithm + usize::from(kind != DctType::IV);
            assert_eq!(
                plan.workspace_requirements(()).unwrap().complex_len(),
                required
            );
            let input = vec![0.25; n];
            let mut output = vec![7.0; n];
            let mut scratch = vec![Complex::new(f64::NAN, f64::NAN); required];
            let cx = scioxide_core::ExecutionContext::sequential();
            assert!(
                plan.execute_into(&input, &mut output, &mut scratch[..required - 1], cx)
                    .is_err()
            );
            assert!(output.iter().all(|&v| v == 7.0));
            plan.execute_into(&input, &mut output, &mut scratch, cx)
                .unwrap();
            assert!(output.iter().all(|v| v.is_finite()));
        }
    }
}

fn close<S: RealScalar>(actual: &[S], expected: &[S]) {
    let bound = S::EPSILON.try_to_f64_exact().unwrap() * 32.0 * actual.len() as f64;
    numerics::assert_real_close(actual, expected, |_| bound);
}

fn mappings<S: RealScalar>() {
    let cx = FftPlanningContext::<S>::default().execution();
    for n in [1, 2, 5, 8, 11, 30] {
        let input: Vec<S> = (0..n)
            .map(|i| S::try_from_f64_lossy((i % 7) as f64 / 8.0 - 0.5).unwrap())
            .collect();
        for (ct, st) in [
            (DctType::II, DstType::II),
            (DctType::III, DstType::III),
            (DctType::IV, DstType::IV),
        ] {
            for norm in [
                Normalization::Unscaled,
                Normalization::Forward,
                Normalization::Inverse,
                Normalization::Orthonormal,
            ] {
                let cosine = RealTransformPlan::new(RealTransformPlanSpec::new(
                    n,
                    ct.into(),
                    scioxide_fft::FftDirection::Forward,
                    norm,
                ))
                .unwrap();
                let cosine_inverse = RealTransformPlan::new(RealTransformPlanSpec::new(
                    n,
                    ct.into(),
                    scioxide_fft::FftDirection::Inverse,
                    norm,
                ))
                .unwrap();
                let sine = RealTransformPlan::new(RealTransformPlanSpec::new(
                    n,
                    st.into(),
                    scioxide_fft::FftDirection::Forward,
                    norm,
                ))
                .unwrap();
                let sine_inverse = RealTransformPlan::new(RealTransformPlanSpec::new(
                    n,
                    st.into(),
                    scioxide_fft::FftDirection::Inverse,
                    norm,
                ))
                .unwrap();
                assert_eq!(
                    cosine.workspace_requirements(()).unwrap().bytes(),
                    sine.workspace_requirements(()).unwrap().bytes()
                );
                let mut scratch = sine.workspace(()).unwrap();
                let mut expected = vec![S::ZERO; n];
                let mut actual = vec![S::ZERO; n];
                for inverse in [false, true] {
                    let raw_two =
                        matches!((ct, inverse), (DctType::II, false) | (DctType::III, true));
                    let mapped: Vec<_> = if raw_two {
                        input
                            .iter()
                            .enumerate()
                            .map(|(i, &x)| if i % 2 == 0 { x } else { -x })
                            .collect()
                    } else {
                        input.iter().rev().copied().collect()
                    };
                    if inverse {
                        cosine_inverse
                            .execute_into(&mapped, &mut expected, &mut scratch, cx)
                            .unwrap();
                        sine_inverse
                            .execute_into(&input, &mut actual, &mut scratch, cx)
                            .unwrap();
                    } else {
                        cosine
                            .execute_into(&mapped, &mut expected, &mut scratch, cx)
                            .unwrap();
                        sine.execute_into(&input, &mut actual, &mut scratch, cx)
                            .unwrap();
                    }
                    if raw_two {
                        expected.reverse();
                    } else {
                        expected
                            .iter_mut()
                            .skip(1)
                            .step_by(2)
                            .for_each(|x| *x = -*x);
                    }
                    close(&actual, &expected);
                }
            }
        }
    }
}

fn storage<S: RealScalar>() {
    let cx = FftPlanningContext::<S>::default().execution();
    for n in [1, 2, 5, 6, 15, 1024] {
        let input = vec![S::HALF; n];
        for kind in [
            DctType::II.into(),
            DctType::III.into(),
            DctType::IV.into(),
            DstType::II.into(),
            DstType::III.into(),
            DstType::IV.into(),
        ] {
            let plan = RealTransformPlan::new(RealTransformPlanSpec::new(
                n,
                kind,
                scioxide_fft::FftDirection::Forward,
                Normalization::Orthonormal,
            ))
            .unwrap();
            let plan_inverse = RealTransformPlan::new(RealTransformPlanSpec::new(
                n,
                kind,
                scioxide_fft::FftDirection::Inverse,
                Normalization::Orthonormal,
            ))
            .unwrap();
            let requirement = plan.workspace_requirements(()).unwrap();
            let mut scratch = vec![Complex::new(S::NAN, S::NAN); requirement.complex_len()];
            let mut output = vec![S::TEN; n];
            let short = scratch.len() - 1;
            assert_eq!(
                plan.execute_into(&input, &mut output, &mut scratch[..short], cx)
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::DimensionMismatch
            );
            assert!(output.iter().all(|&x| x == S::TEN));
            plan.execute_into(&input, &mut output, &mut scratch, cx)
                .unwrap();
            plan_inverse
                .execute_into(&input, &mut output, &mut scratch, cx)
                .unwrap();
        }
    }
}

#[test]
fn shared_mappings_and_storage_f32() {
    mappings::<f32>();
    storage::<f32>();
}

#[test]
fn shared_mappings_and_storage_f64() {
    mappings::<f64>();
    storage::<f64>();
}

#[test]
fn tensor_composes_cosine_and_sine_axes() {
    let shape = [3, 4];
    let layout = StridedLayout::new(&shape, &[4, 1]).unwrap();
    let input: Vec<_> = (0..12).map(|i| (i as f64 - 5.0) / 8.0).collect();
    let kinds = [DstType::III.into(), DctType::IV.into()];
    let plan = RealTransformNdPlan::new(
        RealTransformNdPlanSpec::new(
            &shape,
            &[1, 0],
            &kinds,
            scioxide_fft::FftDirection::Forward,
            Normalization::Orthonormal,
        )
        .unwrap(),
    )
    .unwrap();
    let inverse = RealTransformNdPlan::new(
        RealTransformNdPlanSpec::new(
            &shape,
            &[1, 0],
            &kinds,
            scioxide_fft::FftDirection::Inverse,
            Normalization::Orthonormal,
        )
        .unwrap(),
    )
    .unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let mut actual = vec![0.0; 12];
    plan.execute_into(
        NdView::new(&input, layout).unwrap(),
        NdViewMut::new(&mut actual, layout).unwrap(),
        &mut workspace,
        FftPlanningContext::<f64>::default().execution(),
    )
    .unwrap();
    let rows: Vec<_> = input
        .chunks_exact(4)
        .flat_map(|row| dst(row, DstType::III, Normalization::Orthonormal).unwrap())
        .collect();
    let mut expected = vec![0.0; 12];
    for column in 0..4 {
        let lane: Vec<_> = (0..3).map(|row| rows[4 * row + column]).collect();
        for (row, value) in dct(&lane, DctType::IV, Normalization::Orthonormal)
            .unwrap()
            .into_iter()
            .enumerate()
        {
            expected[4 * row + column] = value;
        }
    }
    close(&actual, &expected);
    inverse
        .execute_in_place(
            NdViewMut::new(&mut actual, layout).unwrap(),
            &mut workspace,
            FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
    close(&actual, &input);
}

/// Integer reduction avoids large-angle libm error in the direct sums.
fn cosine_table(n: usize) -> Vec<f64> {
    let period = 8 * n;
    (0..period)
        .map(|i| {
            if i == 0 {
                1.0
            } else if i == period / 2 {
                -1.0
            } else if i == period / 4 || i == 3 * period / 4 {
                0.0
            } else {
                let centered = if i > period / 2 {
                    i as f64 - period as f64
                } else {
                    i as f64
                };
                (core::f64::consts::TAU * centered / period as f64).cos()
            }
        })
        .collect()
}

fn check_definition<S: RealScalar>(spec: RealTransformPlanSpec, input: &[f64], expected: &[f64]) {
    let s = |v| S::try_from_f64_lossy(v).unwrap();
    let input: Vec<_> = input.iter().copied().map(s).collect();
    let expected: Vec<_> = expected.iter().copied().map(s).collect();
    let plan = RealTransformPlan::<S>::new(spec).unwrap();
    let mut output = vec![S::ZERO; input.len()];
    plan.execute_into(
        &input,
        &mut output,
        &mut plan.workspace(()).unwrap(),
        FftPlanningContext::<S>::default().execution(),
    )
    .unwrap();
    numerics::assert_real_close(&output, &expected, |_| {
        64.0 * S::EPSILON.into_f64() * input.len() as f64
    });
}

#[test]
fn even_two_three_match_definitions_in_every_direction_and_normalization() {
    for n in [4, 6, 10, 12, 18, 64, 66, 130] {
        let input: Vec<_> = (0..n)
            .map(|i| ((i * 29 % 61) as f64 - 30.0) / 32.0)
            .collect();
        let table = cosine_table(n);
        let cosine = |i: usize| table[i % table.len()];
        let sine = |i: usize| table[(2 * n + table.len() - i % table.len()) % table.len()];
        for is_sine in [false, true] {
            for configured_two in [false, true] {
                let kind = match (is_sine, configured_two) {
                    (false, true) => DctType::II.into(),
                    (false, false) => DctType::III.into(),
                    (true, true) => DstType::II.into(),
                    (true, false) => DstType::III.into(),
                };
                for direction in [FftDirection::Forward, FftDirection::Inverse] {
                    let two = configured_two == (direction == FftDirection::Forward);
                    for norm in [
                        Normalization::Unscaled,
                        Normalization::Forward,
                        Normalization::Inverse,
                        Normalization::Orthonormal,
                    ] {
                        let ortho = norm == Normalization::Orthonormal;
                        let factor = if ortho {
                            if two {
                                (2.0 / n as f64).sqrt()
                            } else {
                                (2.0 * n as f64).sqrt().recip()
                            }
                        } else if (norm == Normalization::Forward
                            && direction == FftDirection::Forward)
                            || (norm == Normalization::Inverse
                                && direction == FftDirection::Inverse)
                        {
                            1.0 / n as f64
                        } else {
                            1.0
                        };
                        let expected: Vec<_> = (0..n)
                            .map(|k| {
                                let total: f64 = input
                                    .iter()
                                    .enumerate()
                                    .map(|(j, &x)| {
                                        if two {
                                            x * if is_sine {
                                                sine(2 * (2 * j + 1) * (k + 1))
                                            } else {
                                                cosine(2 * (2 * j + 1) * k)
                                            }
                                        } else if (!is_sine && j == 0) || (is_sine && j == n - 1) {
                                            x * if ortho { 2.0_f64.sqrt() } else { 1.0 }
                                                * if is_sine && k % 2 == 1 { -1.0 } else { 1.0 }
                                        } else {
                                            2.0 * x
                                                * if is_sine {
                                                    sine(2 * (j + 1) * (2 * k + 1))
                                                } else {
                                                    cosine(2 * j * (2 * k + 1))
                                                }
                                        }
                                    })
                                    .sum();
                                let endpoint =
                                    two && ortho && if is_sine { k == n - 1 } else { k == 0 };
                                total * factor * if endpoint { 0.5_f64.sqrt() } else { 1.0 }
                            })
                            .collect();
                        let spec = RealTransformPlanSpec::new(n, kind, direction, norm);
                        check_definition::<f64>(spec, &input, &expected);
                        check_definition::<f32>(spec, &input, &expected);
                    }
                }
            }
        }
    }
}

#[test]
fn odd_decimated_kernels_match_orthonormal_definitions() {
    for n in [9, 15, 25, 35, 49, 105, 225, 3375] {
        let input: Vec<f64> = (0..n)
            .map(|i| ((i * 29 % 61) as f64 - 30.0) / 32.0)
            .collect();
        let scale = (2.0 / n as f64).sqrt();
        let table = cosine_table(n);
        let cosine = |i: usize| table[i % table.len()];
        let sine = |i: usize| table[(2 * n + table.len() - i % table.len()) % table.len()];
        for is_sine in [false, true] {
            for (ct, st) in [
                (DctType::II, DstType::II),
                (DctType::III, DstType::III),
                (DctType::IV, DstType::IV),
            ] {
                let expected: Vec<_> = (0..n)
                    .map(|k| {
                        input
                            .iter()
                            .enumerate()
                            .map(|(j, &x)| {
                                x * match (ct, is_sine) {
                                    (DctType::II, false) => {
                                        let weight = if k == 0 { 0.5_f64.sqrt() } else { 1.0 };
                                        weight * scale * cosine(2 * k * (2 * j + 1))
                                    }
                                    (DctType::II, true) => {
                                        let weight = if k == n - 1 { 0.5_f64.sqrt() } else { 1.0 };
                                        weight * scale * sine(2 * (k + 1) * (2 * j + 1))
                                    }
                                    (DctType::III, false) if j == 0 => (n as f64).sqrt().recip(),
                                    (DctType::III, false) => scale * cosine(2 * j * (2 * k + 1)),
                                    (DctType::III, true) if j == n - 1 => {
                                        (n as f64).sqrt().recip()
                                            * if k % 2 == 0 { 1.0 } else { -1.0 }
                                    }
                                    (DctType::III, true) => scale * sine(2 * (j + 1) * (2 * k + 1)),
                                    (_, false) => scale * cosine((2 * j + 1) * (2 * k + 1)),
                                    (_, true) => scale * sine((2 * j + 1) * (2 * k + 1)),
                                }
                            })
                            .sum()
                    })
                    .collect();
                let kind = if is_sine { st.into() } else { ct.into() };
                let spec = RealTransformPlanSpec::new(
                    n,
                    kind,
                    FftDirection::Forward,
                    Normalization::Orthonormal,
                );
                check_definition::<f64>(spec, &input, &expected);
                check_definition::<f32>(spec, &input, &expected);
            }
        }
    }
}

#[test]
fn real_tensor_in_place_forms_reuse_both_buffers() {
    use scioxide_core::{NdView, NdViewMut, StridedLayout};
    use scioxide_fft::{
        DctType, DstType, RealTransformNdPlan, RealTransformNdPlanSpec, RealTransformNdWorkspace,
    };
    let shape = [2, 3];
    let strides = [3, 1];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let input: [f64; 6] = [0.25, -1.0, 0.5, 0.75, 2.0, -0.125];
    macro_rules! check {
        ($plan:ident,$spec:ident,$kind:ident) => {{
            let plan = $plan::build(
                $spec::new(
                    &shape,
                    &[0, 1],
                    &[$kind::II.into(); 2],
                    FftDirection::Forward,
                    Normalization::Orthonormal,
                )
                .unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap();
            let inverse = $plan::build(
                $spec::new(
                    &shape,
                    &[0, 1],
                    &[$kind::II.into(); 2],
                    FftDirection::Inverse,
                    Normalization::Orthonormal,
                )
                .unwrap(),
                ExecutionContext::sequential(),
            )
            .unwrap();
            let requirements = plan.workspace_requirements(()).unwrap();
            let mut workspace = RealTransformNdWorkspace::new();
            assert_eq!(
                workspace.check(&requirements).unwrap(),
                Capacity::NeedsGrowth
            );
            workspace.try_reserve(&requirements).unwrap();
            assert_eq!(
                workspace.check(&requirements).unwrap(),
                Capacity::Sufficient
            );
            let mut expected = [0.0; 6];
            plan.execute_into(
                NdView::new(&input, layout).unwrap(),
                NdViewMut::new(&mut expected, layout).unwrap(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
            let mut actual = input;
            plan.execute_in_place(
                NdViewMut::new(&mut actual, layout).unwrap(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (*a - b).abs() < 1e-12)
            );
            inverse
                .execute_in_place(
                    NdViewMut::new(&mut actual, layout).unwrap(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(
                actual
                    .iter()
                    .zip(input)
                    .all(|(a, b)| (*a - b).abs() < 1e-12)
            );
        }};
    }
    check!(RealTransformNdPlan, RealTransformNdPlanSpec, DctType);
    check!(RealTransformNdPlan, RealTransformNdPlanSpec, DstType);
}
