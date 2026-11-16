//! Public crate-root coverage for every supported FFT lifecycle.

use scioxide_core::Complex;
use scioxide_core::{ExecutionPolicy, SciOxideError};
use scioxide_fft::{
    DctType, DstType, FftDirection, FftInput, FftLayout, FftOutput, FftPlan, FftPlanSpec,
    Normalization, RealTransformPlan, dct, dst, fft, fftfreq, fftshift, idct, idst, ifft,
    ifftshift, irfft, rfft, rfftfreq,
};
use scioxide_linalg::Matrix;

#[test]
fn allocating_root_transforms_restore_inputs_and_frequency_grids() -> Result<(), SciOxideError> {
    let normalization = Normalization::Inverse;
    let complex = [
        Complex::new(1.0_f64, 0.0),
        Complex::new(2.0, -1.0),
        Complex::new(-0.5, 0.25),
        Complex::new(0.0, 1.0),
    ];
    let spectrum = fft(&complex, normalization)?;
    assert!(
        ifft(&spectrum, normalization)?
            .iter()
            .zip(complex)
            .all(|(a, b)| (*a - b).norm() < 1e-12)
    );

    let real = [1.0_f64, -2.0, 0.5, 3.0];
    let packed = rfft(&real, normalization)?;
    assert!(
        irfft(&packed, normalization)?
            .iter()
            .zip(real)
            .all(|(a, b)| (*a - b).abs() < 1e-12)
    );

    let cosine = dct(&real, DctType::II, normalization)?;
    assert!(
        idct(&cosine, DctType::II, normalization)?
            .iter()
            .zip(real)
            .all(|(a, b)| (*a - b).abs() < 1e-12)
    );
    let sine = dst(&real, DstType::II, normalization)?;
    assert!(
        idst(&sine, DstType::II, normalization)?
            .iter()
            .zip(real)
            .all(|(a, b)| (*a - b).abs() < 1e-12)
    );
    assert_eq!(fftfreq::<f64>(4, 1.0)?, [0.0, 0.25, -0.5, -0.25]);
    assert_eq!(rfftfreq::<f64>(4, 1.0)?, [0.0, 0.25, 0.5]);
    let shifted = fftshift(&[0, 1, 2, 3])?;
    assert_eq!(ifftshift(&shifted)?, [0, 1, 2, 3]);
    Ok(())
}

#[test]
fn allocating_root_multidimensional_transforms_restore_inputs() -> Result<(), SciOxideError> {
    let normalization = Normalization::Inverse;
    let complex = Matrix::from_vec(
        2,
        2,
        vec![
            Complex::new(1.0_f64, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ],
    )?;
    let shape = [2, 2];
    let spectrum = scioxide_fft::fftn(complex.as_slice(), &shape, &[1, 0], normalization)?;
    let recovered = scioxide_fft::ifftn(&spectrum, &shape, &[1, 0], normalization)?;
    assert!(
        recovered
            .iter()
            .zip(complex.as_slice())
            .all(|(a, b)| (*a - *b).norm() < 1e-12)
    );
    let real = [1.0_f64, 2.0, 3.0, 4.0];
    let packed = scioxide_fft::rfftn(&real, &shape, &[0, 1], normalization)?;
    let recovered = scioxide_fft::irfftn(&packed, normalization)?;
    assert!(
        recovered
            .iter()
            .zip(real)
            .all(|(a, b)| (*a - b).abs() < 1e-12)
    );

    Ok(())
}

#[test]
fn reusable_root_plans_restore_complex_and_real_inputs() -> Result<(), SciOxideError> {
    let normalization = Normalization::Inverse;
    let expected = [
        Complex::new(1.0_f64, 0.25),
        Complex::new(-2.0, 0.5),
        Complex::new(0.125, -1.0),
        Complex::new(3.0, 2.0),
    ];
    let mut complex = expected.to_vec();
    let complex_spec = |direction| {
        FftPlanSpec::new(4, direction, normalization, FftLayout::Complex)
            .with_execution_policy(ExecutionPolicy::Sequential)
    };
    let fft_forward = FftPlan::<f64>::build(
        complex_spec(FftDirection::Forward),
        scioxide_fft::FftPlanningContext::default(),
    )?;
    let fft_inverse = FftPlan::<f64>::build(
        complex_spec(FftDirection::Inverse),
        scioxide_fft::FftPlanningContext::default(),
    )?;
    let mut fft_scratch = fft_forward.workspace(())?;
    fft_forward.execute_in_place(
        &mut complex,
        &mut fft_scratch,
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )?;
    fft_inverse.execute_in_place(
        &mut complex,
        &mut fft_scratch,
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )?;

    assert!(
        complex
            .iter()
            .zip(expected)
            .all(|(a, b)| (*a - b).norm() < 1e-12)
    );

    let input = [1.0_f64, -2.0, 0.5, 3.0];
    let real_spec = |direction| {
        FftPlanSpec::new(4, direction, normalization, FftLayout::Real)
            .with_execution_policy(ExecutionPolicy::Sequential)
    };
    let real_forward = FftPlan::<f64>::build(
        real_spec(FftDirection::Forward),
        scioxide_fft::FftPlanningContext::default(),
    )?;
    let real_inverse = FftPlan::<f64>::build(
        real_spec(FftDirection::Inverse),
        scioxide_fft::FftPlanningContext::default(),
    )?;
    let mut packed = scioxide_fft::HermitianSpectrum::zeros(4)?;
    let mut restored = vec![0.0; 4];
    let mut real_scratch = real_forward.workspace(())?;
    real_forward.execute_into(
        &input,
        &mut packed,
        &mut real_scratch,
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )?;
    real_inverse.execute_into(
        &packed,
        &mut restored,
        &mut real_scratch,
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )?;
    assert!(
        restored
            .iter()
            .zip(input)
            .all(|(a, b)| (*a - b).abs() < 1e-12)
    );
    let dct_plan = RealTransformPlan::new(
        scioxide_fft::RealTransformPlanSpec::new(
            input.len(),
            (DctType::II).into(),
            scioxide_fft::FftDirection::Forward,
            normalization,
        )
        .with_execution_policy(ExecutionPolicy::Sequential),
    )?;
    let dct_plan_inverse = RealTransformPlan::new(
        scioxide_fft::RealTransformPlanSpec::new(
            input.len(),
            (DctType::II).into(),
            scioxide_fft::FftDirection::Inverse,
            normalization,
        )
        .with_execution_policy(ExecutionPolicy::Sequential),
    )?;
    let mut dct_output = vec![0.0; dct_plan.spec().length()];
    let mut dct_scratch = dct_plan.workspace(())?;
    dct_plan.execute_into(
        &input,
        &mut dct_output,
        &mut dct_scratch,
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )?;
    dct_plan_inverse.execute_into(
        &dct_output,
        &mut restored,
        &mut dct_scratch,
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )?;

    assert!(
        restored
            .iter()
            .zip(input)
            .all(|(a, b)| (*a - b).abs() < 1e-12)
    );
    let dst_plan = RealTransformPlan::new(
        scioxide_fft::RealTransformPlanSpec::new(
            input.len(),
            (DstType::II).into(),
            scioxide_fft::FftDirection::Forward,
            normalization,
        )
        .with_execution_policy(ExecutionPolicy::Sequential),
    )?;
    let dst_plan_inverse = RealTransformPlan::new(
        scioxide_fft::RealTransformPlanSpec::new(
            input.len(),
            (DstType::II).into(),
            scioxide_fft::FftDirection::Inverse,
            normalization,
        )
        .with_execution_policy(ExecutionPolicy::Sequential),
    )?;
    assert_eq!(dst_plan.spec().length(), input.len());
    assert_eq!(dst_plan.spec().kind(), (DstType::II).into());
    assert_eq!(dst_plan.spec().normalization(), normalization);
    assert_eq!(
        dst_plan.spec().execution_policy(),
        ExecutionPolicy::Sequential
    );
    let _ = dst_plan.resolved_backends().next().unwrap();
    let mut dst_output = vec![0.0; dst_plan.spec().length()];
    let mut dst_scratch = dst_plan.workspace(())?;
    dst_plan.execute_into(
        &input,
        &mut dst_output,
        &mut dst_scratch,
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )?;
    dst_plan_inverse.execute_into(
        &dst_output,
        &mut restored,
        &mut dst_scratch,
        scioxide_fft::FftPlanningContext::<f64>::default().execution(),
    )?;

    assert!(
        restored
            .iter()
            .zip(input)
            .all(|(a, b)| (*a - b).abs() < 1e-12)
    );
    Ok(())
}

fn real_plan(n: usize, direction: FftDirection) -> Result<FftPlan<f64>, SciOxideError> {
    FftPlan::new(FftPlanSpec::new(
        n,
        direction,
        Normalization::Inverse,
        FftLayout::Real,
    ))
}

#[test]
fn packed_conversions_carry_the_logical_length() -> Result<(), SciOxideError> {
    let cx = scioxide_core::ExecutionContext::sequential();
    let samples = [1.0_f64, -2.0, 0.5, 3.0, 0.25, -1.5, 2.0, 0.75];
    let spectrum = rfft(&samples, Normalization::Inverse)?;
    let inverse = real_plan(8, FftDirection::Inverse)?;
    let mut workspace = inverse.workspace(())?;
    let hand_built = FftInput::Hermitian {
        logical_len: 8,
        bins: scioxide_linalg::VectorView::from_slice(spectrum.bins()),
    };
    let mut expected = [0.0; 8];
    inverse.execute_into(hand_built, &mut expected, &mut workspace, cx)?;
    for input in [
        FftInput::from(&spectrum),
        FftInput::hermitian(8, spectrum.bins()),
    ] {
        let mut actual = [0.0; 8];
        inverse.execute_into(input, &mut actual, &mut workspace, cx)?;
        assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
    }

    let forward = real_plan(8, FftDirection::Forward)?;
    let mut packed = scioxide_fft::HermitianSpectrum::zeros(8)?;
    forward.execute_into(&samples, &mut packed, &mut forward.workspace(())?, cx)?;
    assert_eq!(packed, spectrum);
    let mut bins = [Complex::new(0.0, 0.0); 5];
    let output = FftOutput::hermitian(8, &mut bins);
    forward.execute_into(&samples, output, &mut forward.workspace(())?, cx)?;
    assert_eq!(bins.as_slice(), spectrum.bins());

    // Logical lengths 8 and 9 both pack into five bins.
    let long = scioxide_fft::HermitianSpectrum::from_vec(9, spectrum.bins().to_vec())?;
    let mut untouched = [7.0; 8];
    let error = inverse
        .execute_into(&long, &mut untouched, &mut workspace, cx)
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(untouched, [7.0; 8]);
    let mut long_output = scioxide_fft::HermitianSpectrum::zeros(9)?;
    let error = forward
        .execute_into(&samples, &mut long_output, &mut workspace, cx)
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert!(
        long_output
            .bins()
            .iter()
            .all(|z| *z == Complex::new(0.0, 0.0))
    );
    Ok(())
}

#[test]
fn nd_views_convert_to_plan_arguments() -> Result<(), SciOxideError> {
    use scioxide_core::{NdView, NdViewMut, StridedLayout};
    use scioxide_fft::{FftNdInput, FftNdOutput, FftNdPlan, FftNdPlanSpec};
    let cx = scioxide_core::ExecutionContext::sequential();
    let (shape, axes) = ([2, 3], [0, 1]);
    let nd_plan = |layout| {
        let spec = FftNdPlanSpec::new(
            &shape,
            &axes,
            FftDirection::Forward,
            Normalization::Inverse,
            layout,
        )?;
        FftNdPlan::<f64>::new(spec)
    };
    let layout = StridedLayout::new(&shape, &[3, 1])?;
    let packed_layout = StridedLayout::new(&[2, 2], &[2, 1])?;
    let real = [1.0_f64, -2.0, 0.5, 3.0, 0.25, -1.5];
    let complex = real.map(|x| Complex::new(x, 0.5 * x));

    let plan = nd_plan(FftLayout::Complex)?;
    let mut workspace = plan.workspace(())?;
    let mut expected = [Complex::new(0.0, 0.0); 6];
    plan.execute_into(
        FftNdInput::Complex(NdView::new(&complex, layout)?),
        FftNdOutput::Complex(NdViewMut::new(&mut expected, layout)?),
        &mut workspace,
        cx,
    )?;
    let mut actual = [Complex::new(0.0, 0.0); 6];
    let (input, output) = (
        NdView::new(&complex, layout)?,
        NdViewMut::new(&mut actual, layout)?,
    );
    plan.execute_into(input, output, &mut workspace, cx)?;
    assert_eq!(actual, expected);
    let mut untouched = [Complex::new(7.0, 0.0); 6];
    let error = plan
        .execute_into(
            NdView::new(&real, layout)?,
            NdViewMut::new(&mut untouched, layout)?,
            &mut workspace,
            cx,
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::InvalidInput {
            argument,
            reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            ..
        } if argument == scioxide_core::argument!("input")
    ));
    assert_eq!(untouched, [Complex::new(7.0, 0.0); 6]);

    let plan = nd_plan(FftLayout::Real)?;
    let mut workspace = plan.workspace(())?;
    let mut expected = [Complex::new(0.0, 0.0); 4];
    plan.execute_into(
        FftNdInput::Real(NdView::new(&real, layout)?),
        FftNdOutput::Complex(NdViewMut::new(&mut expected, packed_layout)?),
        &mut workspace,
        cx,
    )?;
    let mut actual = [Complex::new(0.0, 0.0); 4];
    let (input, output) = (
        NdView::new(&real, layout)?,
        NdViewMut::new(&mut actual, packed_layout)?,
    );
    plan.execute_into(input, output, &mut workspace, cx)?;
    assert_eq!(actual, expected);
    Ok(())
}
