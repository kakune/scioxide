//! Borrowed transform layouts, workspace capacity, capabilities and cancellation.
#[path = "support/numerics.rs"]
mod numerics;
use scioxide_core::backend::KernelBackend;
use scioxide_core::{
    CancellationToken, Complex, ExecutionContext, ExecutionPolicy, ExecutionStopReason,
    SciOxideError,
};
use scioxide_fft::{
    DctType, DstType, FftDirection, FftInput, FftLayout, FftOutput, FftPlan, FftPlanSpec,
    Normalization, RealTransformPlan, fft,
};
use scioxide_linalg::{MatrixView, VectorView, VectorViewMut};

#[test]
fn complex_plan_accepts_strided_and_conjugated_views() -> Result<(), SciOxideError> {
    let logical = [
        Complex::new(1.0_f64, 0.5),
        Complex::new(-2.0, 1.0),
        Complex::new(0.25, -0.75),
        Complex::new(3.0, 0.0),
    ];
    let mut backing = [Complex::new(99.0, 99.0); 7];
    logical
        .iter()
        .copied()
        .enumerate()
        .for_each(|(index, value)| backing[index * 2] = value);
    let input = VectorView::from_slice_with_stride(&backing, logical.len(), 2)?.conjugate();
    let expected_input = logical.map(|value| value.conj());
    let expected = fft(&expected_input, Normalization::Inverse)?;

    let plan = FftPlan::<f64>::build(
        FftPlanSpec::new(
            logical.len(),
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Complex,
        )
        .with_execution_policy(ExecutionPolicy::Sequential),
        scioxide_fft::FftPlanningContext::default(),
    )?;
    let mut scratch = plan.workspace(())?;
    let sentinel = Complex::new(-17.0, 8.0);
    let mut output_backing = [sentinel; 7];
    let output = VectorViewMut::from_slice_with_stride(&mut output_backing, logical.len(), 2)?;
    plan.execute_into(
        scioxide_fft::FftInput::Complex(input),
        scioxide_fft::FftOutput::Complex(output),
        &mut scratch,
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )?;

    output_backing
        .iter()
        .step_by(2)
        .zip(expected)
        .for_each(|(&actual, expected)| {
            numerics::assert_complex_close(&[actual], &[expected], numerics::component_max, |_| {
                1e-11
            })
        });
    output_backing
        .iter()
        .skip(1)
        .step_by(2)
        .for_each(|&value| assert_eq!(value, sentinel));
    Ok(())
}

#[test]
fn real_plan_shares_kernels_for_strided_forward_and_inverse() -> Result<(), SciOxideError> {
    for logical in [
        vec![1.0_f64, -2.0, 0.5, 3.0],
        vec![1.0_f64, -2.0, 0.5, 3.0, -1.25],
    ] {
        let spec = |direction| {
            FftPlanSpec::new(
                logical.len(),
                direction,
                Normalization::Inverse,
                FftLayout::Real,
            )
            .with_execution_policy(ExecutionPolicy::Sequential)
        };
        let forward = FftPlan::<f64>::build(
            spec(FftDirection::Forward),
            scioxide_fft::FftPlanningContext::default(),
        )?;
        let inverse = FftPlan::<f64>::build(
            spec(FftDirection::Inverse),
            scioxide_fft::FftPlanningContext::default(),
        )?;
        let spectrum_len = logical.len() / 2 + 1;
        let mut input_backing = vec![41.0; logical.len() * 2 - 1];
        logical
            .iter()
            .copied()
            .enumerate()
            .for_each(|(index, value)| input_backing[index * 2] = value);
        let input = VectorView::from_slice_with_stride(&input_backing, logical.len(), 2)?;

        let mut spectrum_backing = vec![Complex::new(23.0, -9.0); spectrum_len * 2 - 1];
        let spectrum_output =
            VectorViewMut::from_slice_with_stride(&mut spectrum_backing, spectrum_len, 2)?;
        let mut scratch = forward.workspace(())?;
        forward.execute_into(
            scioxide_fft::FftInput::Real(input),
            scioxide_fft::FftOutput::Hermitian {
                logical_len: forward.spec().length(),
                bins: spectrum_output,
            },
            &mut scratch,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )?;

        let spectrum = VectorView::from_slice_with_stride(&spectrum_backing, spectrum_len, 2)?;
        let mut restored_backing = vec![71.0; logical.len() * 2 - 1];
        let restored =
            VectorViewMut::from_slice_with_stride(&mut restored_backing, logical.len(), 2)?;
        inverse.execute_into(
            scioxide_fft::FftInput::Hermitian {
                logical_len: inverse.spec().length(),
                bins: spectrum,
            },
            scioxide_fft::FftOutput::Real(restored),
            &mut scratch,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )?;

        restored_backing
            .iter()
            .step_by(2)
            .zip(logical)
            .for_each(|(&actual, expected)| assert!((actual - expected).abs() <= 1e-11));
        restored_backing
            .iter()
            .skip(1)
            .step_by(2)
            .for_each(|&value| assert_eq!(value, 71.0));
    }
    Ok(())
}

#[test]
fn layout_shape_and_workspace_failures_are_typed_and_transactional() -> Result<(), SciOxideError> {
    let mut aliased = [0.0_f64; 1];
    let error = VectorViewMut::from_slice_with_stride(&mut aliased, 2, 0).unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);

    let plan = FftPlan::<f64>::build(
        FftPlanSpec::new(
            4,
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Complex,
        ),
        scioxide_fft::FftPlanningContext::default(),
    )?;
    let input_values = [Complex::new(1.0_f64, 0.0); 4];
    let input = VectorView::from_slice(&input_values);
    let sentinel = Complex::new(13.0, -4.0);
    let mut output = [sentinel; 4];
    let mut short_scratch =
        vec![Complex::new(0.0, 0.0); plan.workspace_requirements(()).unwrap().complex_len() - 1];
    let error = plan
        .execute_into(
            scioxide_fft::FftInput::Complex(input),
            scioxide_fft::FftOutput::Complex(VectorViewMut::from_slice(&mut output)),
            &mut short_scratch,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .expect_err("short workspace must be rejected");
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, [sentinel; 4]);

    let short_input_values = [Complex::new(1.0_f64, 0.0); 3];
    let short_input = VectorView::from_slice(&short_input_values);
    let mut scratch = plan.workspace(())?;
    let error = plan
        .execute_into(
            scioxide_fft::FftInput::Complex(short_input),
            scioxide_fft::FftOutput::Complex(VectorViewMut::from_slice(&mut output)),
            &mut scratch,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .expect_err("shape mismatch must be rejected");
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, [sentinel; 4]);
    Ok(())
}

#[test]
fn matrix_view_transforms_match_owned_transpose() -> Result<(), SciOxideError> {
    let backing = [
        Complex::new(1.0_f64, 0.0),
        Complex::new(2.0, -0.5),
        Complex::new(3.0, 0.25),
        Complex::new(4.0, 0.75),
        Complex::new(5.0, -1.0),
        Complex::new(6.0, 0.0),
    ];
    let transposed = MatrixView::from_slice(2, 3, &backing)?.transpose();
    let owned = transposed.try_to_matrix()?;
    let shape = [transposed.rows(), transposed.cols()];
    let strides = [transposed.row_stride(), transposed.col_stride()];
    let plan = scioxide_fft::FftNdPlan::<f64>::new(scioxide_fft::FftNdPlanSpec::new(
        &shape,
        &[1, 0],
        FftDirection::Forward,
        Normalization::Inverse,
        FftLayout::Complex,
    )?)?;
    let mut actual = [Complex::new(0.0, 0.0); 6];
    plan.execute_into(
        scioxide_core::NdView::new(
            &backing,
            scioxide_core::StridedLayout::new(&shape, &strides)?,
        )?,
        scioxide_core::NdViewMut::row_major(&mut actual, &shape)?,
        &mut plan.workspace(())?,
        ExecutionContext::sequential(),
    )?;
    let expected = scioxide_fft::fftn(owned.as_slice(), &shape, &[1, 0], Normalization::Inverse)?;
    actual
        .as_slice()
        .iter()
        .copied()
        .zip(expected.as_slice().iter().copied())
        .for_each(|(actual, expected)| {
            numerics::assert_complex_close(&[actual], &[expected], numerics::component_max, |_| {
                1e-11
            })
        });
    Ok(())
}

#[test]
fn plan_construction_uses_injected_processor_capabilities() -> Result<(), SciOxideError> {
    let context = ExecutionContext::sequential();
    let spec = |layout| {
        FftPlanSpec::new(8, FftDirection::Forward, Normalization::Inverse, layout)
            .with_execution_policy(ExecutionPolicy::Sequential)
    };
    let complex = FftPlan::<f64>::build(spec(FftLayout::Complex), context)?;
    let real = FftPlan::<f64>::build(spec(FftLayout::Real), context)?;
    let dct = RealTransformPlan::<f64>::build(
        scioxide_fft::RealTransformPlanSpec::new(
            8,
            (DctType::II).into(),
            scioxide_fft::FftDirection::Forward,
            Normalization::Inverse,
        )
        .with_execution_policy(ExecutionPolicy::Sequential),
        context,
    )?;
    let dst = RealTransformPlan::<f64>::build(
        scioxide_fft::RealTransformPlanSpec::new(
            8,
            (DstType::II).into(),
            scioxide_fft::FftDirection::Forward,
            Normalization::Inverse,
        )
        .with_execution_policy(ExecutionPolicy::Sequential),
        context,
    )?;

    [
        complex.resolved_backends().next().unwrap(),
        real.resolved_backends().next().unwrap(),
        dct.resolved_backends().next().unwrap(),
        dst.resolved_backends().next().unwrap(),
    ]
    .into_iter()
    .for_each(|backend| {
        assert_eq!(
            backend,
            scioxide_fft::FftBackend::Cpu(KernelBackend::Scalar)
        )
    });
    Ok(())
}

#[test]
fn cancellation_is_typed_and_preserves_every_consumer_output() -> Result<(), SciOxideError> {
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let context = ExecutionContext::sequential().with_cancellation(&cancellation);

    let fft_plan = FftPlan::<f64>::build(
        FftPlanSpec::new(
            4,
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Complex,
        ),
        scioxide_fft::FftPlanningContext::default(),
    )?;
    let fft_input = [Complex::new(1.0_f64, 0.0); 4];
    let fft_sentinel = Complex::new(19.0, -3.0);
    let mut fft_output = [fft_sentinel; 4];
    let mut fft_scratch = fft_plan.workspace(())?;
    let fft_error = fft_plan
        .execute_into(
            FftInput::from(&fft_input),
            FftOutput::from(&mut fft_output),
            &mut fft_scratch,
            context,
        )
        .expect_err("cancelled FFT must stop");
    assert_cancelled(fft_error);
    assert_eq!(fft_output, [fft_sentinel; 4]);

    let real_plan = FftPlan::<f64>::build(
        FftPlanSpec::new(
            4,
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Real,
        ),
        scioxide_fft::FftPlanningContext::default(),
    )?;
    let mut real_output = [Complex::new(29.0, 2.0); 3];
    let mut real_scratch = real_plan.workspace(())?;
    let real_error = real_plan
        .execute_into(
            FftInput::from(&[1.0, 2.0, 3.0, 4.0]),
            FftOutput::Hermitian {
                logical_len: 4,
                bins: VectorViewMut::from_slice(&mut real_output),
            },
            &mut real_scratch,
            context,
        )
        .expect_err("cancelled real FFT must stop");
    assert_cancelled(real_error);
    assert_eq!(real_output, [Complex::new(29.0, 2.0); 3]);

    let dct_plan = RealTransformPlan::new(scioxide_fft::RealTransformPlanSpec::new(
        4,
        (DctType::II).into(),
        scioxide_fft::FftDirection::Forward,
        Normalization::Inverse,
    ))?;
    let mut dct_output = [31.0; 4];
    let mut dct_scratch = dct_plan.workspace(())?;
    let dct_error = dct_plan
        .execute_into(
            &[1.0, 2.0, 3.0, 4.0],
            &mut dct_output,
            &mut dct_scratch,
            context,
        )
        .expect_err("cancelled DCT must stop");
    assert_cancelled(dct_error);
    assert_eq!(dct_output, [31.0; 4]);

    let dst_plan = RealTransformPlan::new(scioxide_fft::RealTransformPlanSpec::new(
        4,
        (DstType::II).into(),
        scioxide_fft::FftDirection::Forward,
        Normalization::Inverse,
    ))?;
    let mut dst_output = [37.0; 4];
    let mut dst_scratch = dst_plan.workspace(())?;
    let dst_error = dst_plan
        .execute_into(
            &[1.0, 2.0, 3.0, 4.0],
            &mut dst_output,
            &mut dst_scratch,
            context,
        )
        .expect_err("cancelled DST must stop");
    assert_cancelled(dst_error);
    assert_eq!(dst_output, [37.0; 4]);

    Ok(())
}

fn assert_cancelled(error: SciOxideError) {
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped {
            reason: ExecutionStopReason::Cancelled,
            ..
        }
    ));
}

#[test]
fn requirements_capacity_and_borrowed_failure_are_checked() {
    use scioxide_core::{Capacity, SciOxideErrorKind};
    use scioxide_fft::FftWorkspace;
    let plan = FftPlan::build(
        FftPlanSpec::new(
            8,
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Complex,
        ),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let requirements = plan.workspace_requirements(()).unwrap();
    let mut workspace = FftWorkspace::new();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    workspace.try_reserve(&requirements).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    let input = [Complex::new(1.0, 0.25); 8];
    let mut output = [Complex::new(93.0, 17.0); 8];
    let before = output;
    assert_eq!(
        plan.execute_into(&input, &mut output, &mut [], ExecutionContext::sequential())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, before);
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    assert_eq!(
        plan.execute_in_place(
            &mut output,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ExecutionStopped
    );
    assert_eq!(output, before);
}
