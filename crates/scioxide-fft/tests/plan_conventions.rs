//! Preparation capabilities, directional plans and uniform execution errors.
#![expect(clippy::unwrap_used)]

#[path = "support/numerics.rs"]
mod numerics;
use numerics::scalar;
use scioxide_core::{
    AdjointOperator, Complex, ExecutionContext, ExecutionPolicy, LinearOperator, NdView, NdViewMut,
    RealScalar, SciOxideErrorKind, StridedLayout, Tolerance,
};
use scioxide_fft::*;

fn traits<T: Clone + core::fmt::Debug + Send + Sync>() {}

fn capabilities<S: RealScalar>() {
    traits::<FftPlan<S>>();
    traits::<RealTransformPlan<S>>();
    traits::<FftNdPlan<S>>();
    traits::<RealTransformNdPlan<S>>();
    traits::<CztPlan<S>>();
    traits::<NufftPlan<S>>();
    traits::<DiscreteHankelPlan<S>>();
    traits::<FftLogPlan<S>>();
    let native = FftPlanningContext::<S>::default();
    let sequential = ExecutionContext::sequential();
    let input: Vec<_> = (0..8)
        .map(|i| Complex::new(scalar::<S>((i as f64 * 0.7).sin()), S::ZERO))
        .collect();
    let real: Vec<_> = input.iter().map(|x| x.re).collect();
    let mut output = input.clone();
    let mut reference = input.clone();
    let tensor = StridedLayout::new(&[2, 4], &[4, 1]).unwrap();
    for direction in [FftDirection::Forward, FftDirection::Inverse] {
        for explicit in [false, true] {
            let spec = FftPlanSpec::new(8, direction, Normalization::Inverse, FftLayout::Complex);
            assert_eq!(spec.execution_policy(), ExecutionPolicy::Sequential);
            let plan = if explicit {
                FftPlan::build(spec, native)
            } else {
                FftPlan::new(spec)
            }
            .unwrap();
            let backend = plan.resolved_backends().next().unwrap();
            let mut work = plan.workspace(()).unwrap();
            plan.execute_into(&input, &mut output, &mut work, sequential)
                .unwrap();
            plan.execute_into(&input, &mut reference, &mut work, native.execution())
                .unwrap();
            numerics::assert_complex_close(&output, &reference, Complex::norm, |z| {
                512.0 * S::EPSILON.try_to_f64_exact().unwrap() * (1.0 + z.norm())
            });
            let spec = FftPlanSpec::new(8, direction, Normalization::Inverse, FftLayout::Real);
            let plan = if explicit {
                FftPlan::build(spec, native)
            } else {
                FftPlan::new(spec)
            }
            .unwrap();
            let mut work = plan.workspace(()).unwrap();
            let mut packed = HermitianSpectrum::zeros(8).unwrap();
            let mut real_output = real.clone();
            if direction == FftDirection::Forward {
                plan.execute_into(
                    &real,
                    FftOutput::Hermitian {
                        logical_len: 8,
                        bins: scioxide_linalg::VectorViewMut::from_slice(packed.bins_mut()),
                    },
                    &mut work,
                    sequential,
                )
                .unwrap();
            } else {
                plan.execute_into(
                    FftInput::Hermitian {
                        logical_len: 8,
                        bins: packed.bins().into(),
                    },
                    &mut real_output,
                    &mut work,
                    sequential,
                )
                .unwrap();
            }
            for kind in [DctType::II.into(), DstType::III.into()] {
                let spec =
                    RealTransformPlanSpec::new(8, kind, direction, Normalization::Orthonormal);
                let plan = if explicit {
                    RealTransformPlan::build(spec, native)
                } else {
                    RealTransformPlan::new(spec)
                }
                .unwrap();
                assert_eq!(plan.spec().execution_policy(), ExecutionPolicy::Sequential);
                plan.execute_into(
                    &real,
                    &mut real_output,
                    &mut plan.workspace(()).unwrap(),
                    sequential,
                )
                .unwrap();
                let spec = RealTransformNdPlanSpec::new(
                    &[2, 4],
                    &[0, 1],
                    &[kind; 2],
                    direction,
                    Normalization::Orthonormal,
                )
                .unwrap();
                let plan = if explicit {
                    RealTransformNdPlan::build(spec, native)
                } else {
                    RealTransformNdPlan::new(spec)
                }
                .unwrap();
                plan.execute_into(
                    NdView::new(&real, tensor).unwrap(),
                    NdViewMut::new(&mut real_output, tensor).unwrap(),
                    &mut plan.workspace(()).unwrap(),
                    sequential,
                )
                .unwrap();
            }
            let spec = FftNdPlanSpec::new(
                &[2, 4],
                &[0, 1],
                direction,
                Normalization::Inverse,
                FftLayout::Complex,
            )
            .unwrap();
            let plan = if explicit {
                FftNdPlan::build(spec, native)
            } else {
                FftNdPlan::new(spec)
            }
            .unwrap();
            plan.execute_into(
                FftNdInput::Complex(NdView::new(&input, tensor).unwrap()),
                FftNdOutput::Complex(NdViewMut::new(&mut output, tensor).unwrap()),
                &mut plan.workspace(()).unwrap(),
                sequential,
            )
            .unwrap();

            let tensor_packed = rfftn(&real, &[2, 4], &[0, 1], Normalization::Inverse).unwrap();
            let mut bins = tensor_packed.bins().to_vec();
            let packed_layout = StridedLayout::new(&[2, 3], &[3, 1]).unwrap();
            let spec = FftNdPlanSpec::new(
                &[2, 4],
                &[0, 1],
                direction,
                Normalization::Inverse,
                FftLayout::Real,
            )
            .unwrap();
            let plan = if explicit {
                FftNdPlan::build(spec, native)
            } else {
                FftNdPlan::new(spec)
            }
            .unwrap();
            let mut work = plan.workspace(()).unwrap();
            if direction == FftDirection::Forward {
                plan.execute_into(
                    FftNdInput::Real(NdView::new(&real, tensor).unwrap()),
                    FftNdOutput::Complex(NdViewMut::new(&mut bins, packed_layout).unwrap()),
                    &mut work,
                    sequential,
                )
                .unwrap();
                numerics::assert_complex_close(&bins, tensor_packed.bins(), Complex::norm, |z| {
                    512.0 * S::EPSILON.try_to_f64_exact().unwrap() * (1.0 + z.norm())
                });
            } else {
                plan.execute_into(
                    FftNdInput::Complex(NdView::new(&bins, packed_layout).unwrap()),
                    FftNdOutput::Real(NdViewMut::new(&mut real_output, tensor).unwrap()),
                    &mut work,
                    sequential,
                )
                .unwrap();
                numerics::assert_complex_close(
                    &input,
                    &real_output
                        .iter()
                        .map(|&x| Complex::new(x, S::ZERO))
                        .collect::<Vec<_>>(),
                    Complex::norm,
                    |z| 512.0 * S::EPSILON.try_to_f64_exact().unwrap() * (1.0 + z.norm()),
                );
            }
            for contour in [
                CztContour::dft(8).unwrap(),
                CztContour::zoom(S::ZERO, S::HALF, 8, S::ONE, false).unwrap(),
            ] {
                let spec = CztPlanSpec::new(8, 8, contour, Normalization::Inverse);
                let plan = if explicit {
                    CztPlan::build(spec, native)
                } else {
                    CztPlan::new(spec)
                }
                .unwrap();
                let mut work = plan.workspace(()).unwrap();
                plan.execute_into(&input, &mut output, &mut work, sequential)
                    .unwrap();
                plan.execute_into(&real, &mut reference, &mut work, sequential)
                    .unwrap();
                numerics::assert_complex_close(&output, &reference, Complex::norm, |z| {
                    512.0 * S::EPSILON.try_to_f64_exact().unwrap() * (1.0 + z.norm())
                });
                let before = output.clone();
                assert_eq!(
                    plan.execute_into(
                        FftInput::Hermitian {
                            logical_len: 8,
                            bins: packed.bins().into()
                        },
                        &mut output,
                        &mut work,
                        sequential
                    )
                    .unwrap_err()
                    .kind(),
                    SciOxideErrorKind::InvalidInput
                );
                assert_eq!(output, before);
            }
            let points: Vec<_> = (0..8).map(|i| scalar::<S>(i as f64 / 8.0)).collect();
            let geometry = StridedLayout::new(&[8, 1], &[1, 1]).unwrap();
            for ty in 1..=3 {
                let coordinates = NdView::new(&points, geometry).unwrap();
                let transform = match ty {
                    1 => NufftTransform::Type1 {
                        points: coordinates,
                        modes: &[8],
                    },
                    2 => NufftTransform::Type2 {
                        points: coordinates,
                        modes: &[8],
                    },
                    _ => NufftTransform::Type3 {
                        points: coordinates,
                        frequencies: NdView::new(&points, geometry).unwrap(),
                    },
                };
                let spec =
                    NufftPlanSpec::new(direction, Normalization::Unscaled, scalar::<S>(0.01));
                assert_eq!(spec.execution_policy(), ExecutionPolicy::Sequential);
                let plan = if explicit {
                    NufftPlan::build(transform, spec, native)
                } else {
                    NufftPlan::new(transform, spec)
                }
                .unwrap();
                let mut work = plan.workspace(()).unwrap();
                plan.execute_into(&input, &mut output, &mut work, sequential)
                    .unwrap();
                plan.execute_into(&real, &mut reference, &mut work, sequential)
                    .unwrap();
                numerics::assert_complex_close(&output, &reference, Complex::norm, |z| {
                    512.0 * S::EPSILON.try_to_f64_exact().unwrap() * (1.0 + z.norm())
                });
                operator_contract(&plan, &input);
            }
            let zeros: Vec<_> = (1..=9)
                .map(|i| scalar::<S>(i as f64 * core::f64::consts::PI))
                .collect();
            let spec = DiscreteHankelPlanSpec::new(S::HALF, scalar::<S>(4.0), direction)
                .with_tolerance(Tolerance::new(scalar::<S>(1e-4), scalar::<S>(1e-4)).unwrap());
            let plan = if explicit {
                DiscreteHankelPlan::build(&zeros, spec, native.execution())
            } else {
                DiscreteHankelPlan::new(&zeros, spec)
            }
            .unwrap();
            let mut work = plan.workspace(()).unwrap();
            plan.execute_into(&input, &mut output, &mut work, sequential)
                .unwrap();
            plan.apply_into(&input, &mut reference, &mut work).unwrap();
            numerics::assert_complex_close(&output, &reference, Complex::norm, |z| {
                512.0 * S::EPSILON.try_to_f64_exact().unwrap() * (1.0 + z.norm())
            });
            operator_contract(&plan, &input);

            let spec = FftLogPlanSpec::new(
                8,
                scalar::<S>(0.125),
                S::ZERO,
                S::ZERO,
                Complex::new(S::ZERO, S::ZERO),
                direction,
            );
            let plan = if explicit {
                FftLogPlan::build(spec, native, sequential)
            } else {
                FftLogPlan::new(spec)
            }
            .unwrap();
            assert_eq!(plan.resolved_backends().next().unwrap(), backend);
            let mut work = plan.workspace(()).unwrap();
            plan.execute_into(&input, &mut output, &mut work, sequential)
                .unwrap();
            plan.apply_into(&input, &mut reference, &mut work).unwrap();
            numerics::assert_complex_close(&output, &reference, Complex::norm, |z| {
                512.0 * S::EPSILON.try_to_f64_exact().unwrap() * (1.0 + z.norm())
            });
            let owned = if direction == FftDirection::Forward {
                fht(&real, scalar::<S>(0.125), S::ZERO, S::ZERO, S::ZERO)
            } else {
                ifht(&real, scalar::<S>(0.125), S::ZERO, S::ZERO, S::ZERO)
            }
            .unwrap();
            numerics::assert_complex_close(
                &output,
                &owned
                    .iter()
                    .map(|&x| Complex::new(x, S::ZERO))
                    .collect::<Vec<_>>(),
                Complex::norm,
                |z| 512.0 * S::EPSILON.try_to_f64_exact().unwrap() * (1.0 + z.norm()),
            );
            operator_contract(&plan, &input);
        }
    }
}

fn operator_contract<S: RealScalar>(plan: &impl AdjointOperator<Complex<S>>, input: &[Complex<S>]) {
    let y: Vec<_> = input
        .iter()
        .rev()
        .map(|&z| z + Complex::new(S::HALF, S::ONE))
        .collect();
    let mut applied = input.to_vec();
    let mut adjoint = y.clone();
    let mut scratch = vec![
        Complex::new(S::ZERO, S::ZERO);
        plan.apply_workspace_len()
            .unwrap()
            .max(plan.apply_adjoint_workspace_len().unwrap())
    ];
    plan.apply_into(input, &mut applied, &mut scratch).unwrap();
    plan.apply_adjoint_into(&y, &mut adjoint, &mut scratch)
        .unwrap();
    let mut state = scioxide_core::SolveState::new();
    let mut budgets = scioxide_core::SolveBudgets::unbounded();
    let identity = scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(8),
        scioxide_core::algorithm_id!("fft", "plan-conventions", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let mut scoped = input.to_vec();
    plan.apply_into_scoped(
        input,
        &mut scoped,
        &mut scratch,
        &mut scioxide_core::OperatorContext::new(&mut context),
    )
    .and_then(scioxide_core::Invocation::require)
    .unwrap();
    numerics::assert_complex_close(&scoped, &applied, Complex::norm, |z| {
        512.0 * S::EPSILON.try_to_f64_exact().unwrap() * (1.0 + z.norm())
    });
    plan.apply_adjoint_into_scoped(
        &y,
        &mut scoped,
        &mut scratch,
        &mut scioxide_core::OperatorContext::new(&mut context),
    )
    .and_then(scioxide_core::Invocation::require)
    .unwrap();
    numerics::assert_complex_close(&scoped, &adjoint, Complex::norm, |z| {
        512.0 * S::EPSILON.try_to_f64_exact().unwrap() * (1.0 + z.norm())
    });
    let lhs = applied
        .iter()
        .zip(&y)
        .fold(Complex::new(S::ZERO, S::ZERO), |sum, (&a, &b)| {
            sum + a.conj() * b
        });
    let rhs = input
        .iter()
        .zip(&adjoint)
        .fold(Complex::new(S::ZERO, S::ZERO), |sum, (&a, &b)| {
            sum + a.conj() * b
        });
    numerics::assert_complex_close(&[lhs], &[rhs], Complex::norm, |z| {
        512.0 * S::EPSILON.try_to_f64_exact().unwrap() * (1.0 + z.norm())
    });
    let before = applied.clone();
    assert_eq!(
        plan.apply_into(input, &mut applied, &mut [])
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(applied, before);
    assert_eq!(
        plan.apply_adjoint_into(input, &mut applied, &mut [])
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(applied, before);
}

#[test]
fn default_and_native_preparation_f32() {
    capabilities::<f32>();
}
#[test]
fn default_and_native_preparation_f64() {
    capabilities::<f64>();
}

#[test]
fn validation_belongs_to_preparation_and_short_real_storage_is_dimensional() {
    let spec = FftNdPlanSpec::new(
        &[2, 4],
        &[2],
        FftDirection::Forward,
        Normalization::Inverse,
        FftLayout::Complex,
    )
    .unwrap();
    assert_eq!(
        FftNdPlan::<f64>::new(spec).unwrap_err().kind(),
        SciOxideErrorKind::OutOfBounds
    );
    let spec = FftNdPlanSpec::new(
        &[2, 4],
        &[],
        FftDirection::Forward,
        Normalization::Inverse,
        FftLayout::Real,
    )
    .unwrap();
    assert_eq!(
        FftNdPlan::<f64>::new(spec).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    let spec = CztPlanSpec::new(0, 8, CztContour::dft(8).unwrap(), Normalization::Inverse);
    assert_eq!(
        CztPlan::<f64>::new(spec).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    let spec = CztPlanSpec::new(
        usize::MAX,
        8,
        CztContour::dft(8).unwrap(),
        Normalization::Inverse,
    );
    assert_eq!(
        CztPlan::<f64>::new(spec).unwrap_err().kind(),
        SciOxideErrorKind::SizeOverflow
    );
    for kind in [DctType::II.into(), DstType::III.into()] {
        let plan = RealTransformNdPlan::new(
            RealTransformNdPlanSpec::new(
                &[2, 4],
                &[0, 1],
                &[kind; 2],
                FftDirection::Forward,
                Normalization::Inverse,
            )
            .unwrap(),
        )
        .unwrap();
        let layout = StridedLayout::new(&[2, 4], &[4, 1]).unwrap();
        let input = [1.0_f64; 8];
        let mut output = [99.0; 8];
        let short_real = RealTransformNdPlan::<f64>::new(
            RealTransformNdPlanSpec::new(
                &[4],
                &[0],
                &[kind],
                FftDirection::Forward,
                Normalization::Inverse,
            )
            .unwrap(),
        )
        .unwrap()
        .workspace(())
        .unwrap();
        let short_complex = RealTransformNdPlan::<f64>::new(
            RealTransformNdPlanSpec::new(
                &[16],
                &[],
                &[],
                FftDirection::Forward,
                Normalization::Inverse,
            )
            .unwrap(),
        )
        .unwrap()
        .workspace(())
        .unwrap();
        for mut work in [RealTransformNdWorkspace::new(), short_real, short_complex] {
            assert_eq!(
                plan.execute_into(
                    NdView::new(&input, layout).unwrap(),
                    NdViewMut::new(&mut output, layout).unwrap(),
                    &mut work,
                    ExecutionContext::sequential()
                )
                .unwrap_err()
                .kind(),
                SciOxideErrorKind::DimensionMismatch
            );
            assert_eq!(output, [99.0; 8]);
            assert_eq!(
                plan.execute_in_place(
                    NdViewMut::new(&mut output, layout).unwrap(),
                    &mut work,
                    ExecutionContext::sequential()
                )
                .unwrap_err()
                .kind(),
                SciOxideErrorKind::DimensionMismatch
            );
            assert_eq!(output, [99.0; 8]);
        }
    }
}
