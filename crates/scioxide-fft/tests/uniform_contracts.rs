//! Cross-family contract regressions for preparation, introspection and accounting.
#![expect(clippy::unwrap_used)]
use scioxide_core::backend::KernelBackend;
use scioxide_core::{
    Complex, ExecutionContext, NdView, SciOxideError, SciOxideErrorKind, StridedLayout,
};
use scioxide_fft::*;

#[track_caller]
fn nonfinite<T>(result: Result<T, SciOxideError>) {
    let error = result.err().unwrap();
    assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Input,
            ..
        }
    ));
}

#[test]
fn inaccurate_supplied_bessel_zero_is_invalid_input() {
    let zeros = [core::f64::consts::PI + 0.01, 2.0 * core::f64::consts::PI];
    let error = DiscreteHankelPlan::new(
        &zeros,
        DiscreteHankelPlanSpec::new(0.5, 1.0, FftDirection::Forward)
            .with_tolerance(scioxide_core::Tolerance::new(1e-10, 1e-10).unwrap()),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    assert!(
        matches!(error.view(), scioxide_core::SciOxideErrorView::InvalidInput { argument, .. }
        if argument == scioxide_core::argument!("zeros"))
    );
}

#[test]
fn preparation_range_failures_are_numerical() {
    use scioxide_core::{NumericalIssue, SciOxideErrorView};
    let overflow = |error: SciOxideError| {
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::NumericalInstability {
                    issue: NumericalIssue::Overflow,
                    ..
                }
            ),
            "{error:?}"
        )
    };
    let layout = StridedLayout::new(&[1, 1], &[1, 1]).unwrap();
    overflow(
        NufftPlan::new(
            NufftTransform::Type3 {
                points: NdView::new(&[1e200_f64], layout).unwrap(),
                frequencies: NdView::new(&[1e200_f64], layout).unwrap(),
            },
            NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, 1e-6),
        )
        .unwrap_err(),
    );
    let zeros = [core::f32::consts::PI, 2.0 * core::f32::consts::PI];
    overflow(
        DiscreteHankelPlan::new(
            &zeros,
            DiscreteHankelPlanSpec::new(0.5, 1e-39, FftDirection::Forward)
                .with_tolerance(scioxide_core::Tolerance::new(1e-5, 1e-5).unwrap()),
        )
        .unwrap_err(),
    );
    let log = FftLogPlan::new(
        FftLogPlanSpec::new(
            16,
            f64::MAX / 4.0,
            0.0,
            0.0,
            Complex::new(0.0, 0.0),
            FftDirection::Forward,
        )
        .with_log_radius_center(f64::MAX),
    )
    .unwrap();
    overflow(log.log_radius(15).unwrap_err());
}

#[test]
fn preparation_preserves_extended_scales_and_rejects_destroyed_modes() {
    use scioxide_core::{NumericalIssue, SciOxideErrorView};
    let underflow = |error: SciOxideError| {
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::NumericalInstability {
                    issue: NumericalIssue::Underflow,
                    ..
                }
            ),
            "{error:?}"
        )
    };
    let zeros = [core::f32::consts::PI, 2.0 * core::f32::consts::PI];
    let radius = 1e12_f32;
    let plan = DiscreteHankelPlan::new(
        &zeros,
        DiscreteHankelPlanSpec::new(0.5, radius, FftDirection::Forward)
            .with_tolerance(scioxide_core::Tolerance::new(1e-5, 1e-5).unwrap()),
    )
    .unwrap();
    let mut output = [Complex::new(0.0, 0.0)];
    plan.execute_into(
        &[Complex::new(1.0, 0.0)],
        &mut output,
        &mut plan.workspace(()).unwrap(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    // For n=1 and nu=1/2, the defining quadrature is R²/(2pi).
    let expected = (radius as f64).powi(2) / (2.0 * core::f64::consts::PI);
    assert!((output[0].re as f64 - expected).abs() <= 24.0 * f32::EPSILON as f64 * expected);
    assert_eq!(output[0].im, 0.0);
    underflow(
        FftLogPlan::<f32>::new(FftLogPlanSpec::new(
            16,
            1e-8,
            0.0,
            0.0,
            Complex::new(8.0, 0.0),
            FftDirection::Forward,
        ))
        .unwrap_err(),
    );
}

#[test]
fn transforms_and_shifts_share_axis_errors() {
    let (input, shape) = ([Complex::new(1.0_f64, 0.0); 4], [2, 2]);
    for (axes, kind) in [
        (&[2][..], SciOxideErrorKind::OutOfBounds),
        (&[0, 0][..], SciOxideErrorKind::InvalidInput),
    ] {
        for error in [
            fftn(&input, &shape, axes, Normalization::Unscaled).unwrap_err(),
            fftshift_nd(&input, &shape, axes).unwrap_err(),
            ifftshift_nd(&input, &shape, axes).unwrap_err(),
        ] {
            assert_eq!(error.kind(), kind);
        }
    }
    let empty: &[f64] = &[];
    assert!(fftshift_nd(empty, &[0, 2], &[0]).unwrap().is_empty());
    assert!(ifftshift_nd(empty, &[0, 2], &[0]).unwrap().is_empty());
}

#[test]
fn nonfinite_parameters_have_one_input_error_kind() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for i in 0..4 {
            let mut p = [0.0, 0.0, 0.0, 0.25];
            p[i] = bad;
            nonfinite(CztContour::new(p[0], p[1], p[2], p[3]));
            let mut p = [0.1, 0.0, 0.0, 0.0];
            p[i] = bad;
            nonfinite(fftlog_low_ringing_offset(p[0], p[1], p[2], p[3]));
        }
        for i in 0..3 {
            let mut p = [0.0, 1.0, 2.0];
            p[i] = bad;
            nonfinite(CztContour::zoom(p[0], p[1], 4, p[2], false));
        }
        for z in [Complex::new(bad, 0.0), Complex::new(1.0, bad)] {
            nonfinite(CztContour::from_wa(Complex::new(1.0, 0.0), z));
            nonfinite(CztContour::from_wa(z, Complex::new(1.0, 0.0)));
        }
        for i in 0..6 {
            // [log_spacing, order, offset, bias.re, bias.im, log_radius_center]
            let mut p = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
            p[i] = bad;
            nonfinite(FftLogPlan::new(
                FftLogPlanSpec::new(
                    4,
                    p[0],
                    p[1],
                    p[2],
                    Complex::new(p[3], p[4]),
                    FftDirection::Forward,
                )
                .with_log_radius_center(p[5]),
            ));
        }
        let zeros = [core::f64::consts::PI, 2.0 * core::f64::consts::PI];
        let tolerance = scioxide_core::Tolerance::new(1e-10, 1e-10).unwrap();
        for (order, extent) in [(bad, 1.0), (0.5, bad)] {
            nonfinite(DiscreteHankelPlan::new(
                &zeros,
                DiscreteHankelPlanSpec::new(order, extent, FftDirection::Forward)
                    .with_tolerance(tolerance),
            ));
        }
        nonfinite(DiscreteHankelPlan::new(
            &[bad, zeros[1]],
            DiscreteHankelPlanSpec::new(0.5, 1.0, FftDirection::Forward).with_tolerance(tolerance),
        ));
        let layout = StridedLayout::new(&[1, 1], &[1, 1]).unwrap();
        let points = [0.0];
        let bad_points = [bad];
        let spec = NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, 1e-6);
        for spec in [
            NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, bad),
            spec.with_oversampling(bad),
        ] {
            nonfinite(NufftPlan::new(
                NufftTransform::Type1 {
                    points: NdView::new(&points, layout).unwrap(),
                    modes: &[3],
                },
                spec,
            ));
        }
        for ty in [1, 2, 3] {
            let bad_view = NdView::new(&bad_points, layout).unwrap();
            let transform = match ty {
                1 => NufftTransform::Type1 {
                    points: bad_view,
                    modes: &[3],
                },
                2 => NufftTransform::Type2 {
                    points: bad_view,
                    modes: &[3],
                },
                _ => NufftTransform::Type3 {
                    points: NdView::new(&points, layout).unwrap(),
                    frequencies: bad_view,
                },
            };
            nonfinite(NufftPlan::new(transform, spec));
        }
    }
    assert_eq!(
        fftlog_low_ringing_offset(0.0, 0.0, 0.0, 0.0)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        CztContour::zoom(0.0, 1.0, 4, 0.0, false)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
}

#[test]
fn required_backend_is_validated_even_without_nd_children() {
    let cx = FftPlanningContext::<f64>::default();
    // AVX-512 has no FFT leaves and is rejected on every host; other backends
    // only where the host does not verify them.
    for backend in core::iter::once(KernelBackend::Avx512).chain(
        [KernelBackend::Avx2Fma, KernelBackend::Neon]
            .into_iter()
            .filter(|b| !b.is_available(cx.execution().processor())),
    ) {
        for batch in [0, 1] {
            for layout in [FftLayout::Complex, FftLayout::Real, FftLayout::Hermitian] {
                let spec = FftNdPlanSpec::new(
                    &[batch, 8],
                    &[1],
                    FftDirection::Forward,
                    Normalization::Inverse,
                    layout,
                )
                .unwrap()
                .with_backend_strategy(FftBackendStrategy::Required(backend));
                assert!(matches!(
                    FftNdPlan::build(spec, cx).unwrap_err().view(),
                    scioxide_core::SciOxideErrorView::InvalidInput {
                        reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                        ..
                    }
                ));
            }
            let spec = RealTransformNdPlanSpec::new(
                &[batch, 8],
                &[1],
                &[DctType::II.into()],
                FftDirection::Forward,
                Normalization::Inverse,
            )
            .unwrap()
            .with_backend_strategy(FftBackendStrategy::Required(backend));
            assert_eq!(
                RealTransformNdPlan::build(spec, cx).unwrap_err().kind(),
                SciOxideErrorKind::InvalidInput
            );
        }
    }
}

#[test]
fn nufft_scoped_operators_share_inherent_charges_and_budget_denial() {
    use scioxide_core::{
        AdjointOperator, Budget, BudgetKind, EvaluationRole, LinearOperator, SolveBudgets,
        SolveState,
    };
    let points = [0.0, 0.5];
    let plan = NufftPlan::new(
        NufftTransform::Type1 {
            points: NdView::new(&points, StridedLayout::new(&[2, 1], &[1, 1]).unwrap()).unwrap(),
            modes: &[4],
        },
        NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, 1e-6),
    )
    .unwrap();
    // Inherent execution, the scoped operator and its scoped adjoint.
    for mode in 0..3 {
        let adjoint = mode == 2;
        let input = vec![Complex::new(1.0, 0.25); if adjoint { 4 } else { 2 }];
        let mut output = vec![Complex::new(91.0, -13.0); if adjoint { 2 } else { 4 }];
        let mut work = plan.workspace(()).unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(1),
            scioxide_core::algorithm_id!("test", "parent", 1),
            0,
            0,
        );
        let mut parent = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let mut run = |output: &mut [Complex<f64>],
                       parent: &mut scioxide_core::SolveContext<'_, '_>| {
            match mode {
                0 => plan.execute_into(&input, output, &mut work, parent),
                1 => plan
                    .apply_into_scoped(
                        &input,
                        output,
                        &mut work,
                        &mut scioxide_core::OperatorContext::new(parent),
                    )
                    .and_then(scioxide_core::Invocation::require),
                _ => plan
                    .apply_adjoint_into_scoped(
                        &input,
                        output,
                        &mut work,
                        &mut scioxide_core::OperatorContext::new(parent),
                    )
                    .and_then(scioxide_core::Invocation::require),
            }
        };
        run(&mut output, &mut parent).unwrap();
        assert_eq!(
            parent
                .counts()
                .nested()
                .provider(EvaluationRole::LinearOperator),
            1
        );
        assert_eq!(parent.counts().own().iterations(), 0);
        assert_eq!(parent.counts().nested().iterations(), 0);
        let before = output.clone();
        let error = run(&mut output, &mut parent).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::EvaluationLimit,
                ..
            }
        ));
        assert_eq!(output, before);
        assert_eq!(
            parent
                .counts()
                .nested()
                .provider(EvaluationRole::LinearOperator),
            1
        );
    }
}

#[test]
fn every_plan_exposes_spec_backends_bytes_and_limits() {
    let cx = ExecutionContext::sequential();
    let strategy = FftBackendStrategy::Required(KernelBackend::Scalar);
    macro_rules! surface {
        ($plan:expr, $ty:ty, $children:expr) => {{
            let plan = $plan;
            let _: &$ty = plan.spec();
            assert!(plan.plan_bytes() >= size_of_val(&plan));
            let backends: Vec<FftBackend> = plan.resolved_backends().collect();
            assert_eq!(
                backends,
                vec![FftBackend::Cpu(KernelBackend::Scalar); $children]
            );
            let requirements = plan.workspace_requirements(()).unwrap();
            assert_eq!(requirements.byte_limit(), usize::MAX);
            assert!(
                plan.workspace(()).unwrap().check(&requirements).unwrap()
                    == scioxide_core::Capacity::Sufficient
            );
            plan
        }};
    }
    let _ = surface!(
        FftPlan::<f64>::build(
            FftPlanSpec::new(
                4,
                FftDirection::Forward,
                Normalization::Inverse,
                FftLayout::Complex
            )
            .with_backend_strategy(strategy),
            cx
        )
        .unwrap(),
        FftPlanSpec,
        1
    );
    let real = surface!(
        RealTransformPlan::<f64>::build(
            RealTransformPlanSpec::new(
                4,
                DctType::II.into(),
                FftDirection::Inverse,
                Normalization::Orthonormal
            )
            .with_backend_strategy(strategy),
            cx
        )
        .unwrap(),
        RealTransformPlanSpec,
        1
    );
    assert_eq!(real.spec().direction(), FftDirection::Inverse);
    assert_eq!(real.spec().backend_strategy(), strategy);
    surface!(
        FftNdPlan::<f64>::build(
            FftNdPlanSpec::new(
                &[4, 4],
                &[0, 1],
                FftDirection::Forward,
                Normalization::Inverse,
                FftLayout::Complex
            )
            .unwrap()
            .with_backend_strategy(strategy),
            cx
        )
        .unwrap(),
        FftNdPlanSpec,
        2
    );
    let real_nd = surface!(
        RealTransformNdPlan::<f64>::build(
            RealTransformNdPlanSpec::new(
                &[2, 4],
                &[1],
                &[DstType::II.into()],
                FftDirection::Forward,
                Normalization::Inverse
            )
            .unwrap()
            .with_backend_strategy(strategy),
            cx
        )
        .unwrap(),
        RealTransformNdPlanSpec,
        1
    );
    assert_eq!(real_nd.spec().shape(), &[2, 4]);
    assert_eq!(real_nd.spec().axes(), &[1]);
    assert_eq!(real_nd.spec().types(), &[DstType::II.into()]);
    surface!(
        CztPlan::build(
            CztPlanSpec::new(
                4,
                4,
                CztContour::<f64>::dft(4).unwrap(),
                Normalization::Inverse
            )
            .with_backend_strategy(strategy),
            cx
        )
        .unwrap(),
        CztPlanSpec<f64>,
        1
    );
    surface!(
        CztPlan::build(
            CztPlanSpec::new(
                4,
                4,
                CztContour::new(800.0_f64, 0.0, 0.0, 0.0).unwrap(),
                Normalization::Unscaled
            )
            .with_backend_strategy(strategy),
            cx
        )
        .unwrap(),
        CztPlanSpec<f64>,
        0
    );
    let points = [0.0, 0.5];
    let nufft = surface!(
        NufftPlan::build(
            NufftTransform::Type1 {
                points: NdView::new(&points, StridedLayout::new(&[2, 1], &[1, 1]).unwrap())
                    .unwrap(),
                modes: &[4],
            },
            NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, 1e-6)
                .with_backend_strategy(strategy),
            cx
        )
        .unwrap(),
        NufftPlanSpec<f64>,
        1
    );
    assert_eq!(nufft.spec().backend_strategy(), strategy);
    let fftlog = surface!(
        FftLogPlan::build(
            FftLogPlanSpec::new(
                4,
                0.1,
                0.0,
                0.0,
                Complex::new(0.0, 0.0),
                FftDirection::Forward
            )
            .with_backend_strategy(strategy),
            cx,
            cx
        )
        .unwrap(),
        FftLogPlanSpec<f64>,
        1
    );
    assert_eq!(fftlog.spec().backend_strategy(), strategy);
    let zeros = [core::f64::consts::PI, 2.0 * core::f64::consts::PI];
    let hankel = surface!(
        DiscreteHankelPlan::new(
            &zeros,
            DiscreteHankelPlanSpec::new(0.5, 4.0, FftDirection::Forward)
                .with_tolerance(scioxide_core::Tolerance::new(1e-10, 1e-10).unwrap())
        )
        .unwrap(),
        DiscreteHankelPlanSpec<f64>,
        0
    );
    assert_eq!(hankel.spec().order(), 0.5);
    assert_eq!(hankel.spec().extent(), 4.0);
    let capped = CztPlan::new(
        CztPlanSpec::new(
            3,
            3,
            CztContour::<f64>::dft(3).unwrap(),
            Normalization::Unscaled,
        )
        .with_limits(CztLimits {
            workspace_bytes: 4096,
            ..CztLimits::default()
        }),
    )
    .unwrap();
    assert_eq!(
        capped.workspace_requirements(()).unwrap().byte_limit(),
        4096
    );
}

#[test]
fn fft_running_specs_enforce_required_backend_selection() {
    let cx = ExecutionContext::sequential();
    // AVX-512 has no FFT leaves and is rejected on every host; other backends
    // only where the host does not verify them.
    for backend in core::iter::once(KernelBackend::Avx512).chain(
        [KernelBackend::Avx2Fma, KernelBackend::Neon]
            .into_iter()
            .filter(|b| !b.is_available(cx.processor())),
    ) {
        let strategy = FftBackendStrategy::Required(backend);
        let points = [0.0, 0.5];
        let error = NufftPlan::build(
            NufftTransform::Type1 {
                points: NdView::new(&points, StridedLayout::new(&[2, 1], &[1, 1]).unwrap())
                    .unwrap(),
                modes: &[4],
            },
            NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, 1e-6)
                .with_backend_strategy(strategy),
            cx,
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
        let error = NufftPlan::<f64>::build(
            NufftTransform::Type1 {
                points: NdView::new(&[], StridedLayout::new(&[1, 0], &[0, 0]).unwrap()).unwrap(),
                modes: &[],
            },
            NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, 1e-6)
                .with_backend_strategy(strategy),
            cx,
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
        let error = FftLogPlan::build(
            FftLogPlanSpec::new(
                4,
                0.1,
                0.0,
                0.0,
                Complex::new(0.0, 0.0),
                FftDirection::Forward,
            )
            .with_backend_strategy(strategy),
            cx,
            cx,
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    }
}
