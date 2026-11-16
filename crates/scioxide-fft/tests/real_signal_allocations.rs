//! Prepared real, tensor and chirp-z execution reuses caller storage without allocation.

use scioxide_core::Complex;
use scioxide_core::ExecutionPolicy;
use scioxide_fft::{
    DctType, DstType, FftDirection, FftLayout, FftPlan, FftPlanSpec, HermitianSpectrum,
    Normalization, RealTransformPlan,
};

fn signal(len: usize, phase: f64) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let position = index as f64;
            (position * 0.013 + phase).sin() + 0.25 * (position * 0.031 - phase).cos()
        })
        .collect()
}

#[test]
fn supplied_real_signal_storage_has_zero_steady_state_allocations() {
    let len = 1_024;
    let a = signal(len, 0.2);
    let spec = |direction| {
        FftPlanSpec::new(len, direction, Normalization::Inverse, FftLayout::Real)
            .with_execution_policy(ExecutionPolicy::Sequential)
    };
    let real_forward = FftPlan::<f64>::build(
        spec(FftDirection::Forward),
        scioxide_fft::FftPlanningContext::default(),
    )
    .unwrap();
    let real_inverse = FftPlan::<f64>::build(
        spec(FftDirection::Inverse),
        scioxide_fft::FftPlanningContext::default(),
    )
    .unwrap();
    let mut real_scratch = real_forward.workspace(()).unwrap();
    let mut spectrum = HermitianSpectrum::zeros(len).unwrap();
    let mut restored = vec![0.0; len];

    real_forward
        .execute_into(
            &a,
            &mut spectrum,
            &mut real_scratch,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
    real_inverse
        .execute_into(
            &spectrum,
            &mut restored,
            &mut real_scratch,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
}

#[test]
fn supplied_dct_dst_storage_has_zero_steady_state_allocations() {
    let input = signal(1_024, 0.35);

    for ty in [DctType::I, DctType::II, DctType::III, DctType::IV] {
        let plan = RealTransformPlan::new(
            scioxide_fft::RealTransformPlanSpec::new(
                input.len(),
                (ty).into(),
                scioxide_fft::FftDirection::Forward,
                Normalization::Inverse,
            )
            .with_execution_policy(ExecutionPolicy::Sequential),
        )
        .unwrap();
        let plan_inverse = RealTransformPlan::new(
            scioxide_fft::RealTransformPlanSpec::new(
                input.len(),
                (ty).into(),
                scioxide_fft::FftDirection::Inverse,
                Normalization::Inverse,
            )
            .with_execution_policy(ExecutionPolicy::Sequential),
        )
        .unwrap();
        let mut scratch = plan.workspace(()).unwrap();
        let mut output = vec![0.0; plan.spec().length()];
        plan.execute_into(
            &input,
            &mut output,
            &mut scratch,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
        plan_inverse
            .execute_into(
                &input,
                &mut output,
                &mut scratch,
                scioxide_fft::FftPlanningContext::<f64>::default().execution(),
            )
            .unwrap();
    }

    for ty in [DstType::I, DstType::II, DstType::III, DstType::IV] {
        let plan = RealTransformPlan::new(
            scioxide_fft::RealTransformPlanSpec::new(
                input.len(),
                (ty).into(),
                scioxide_fft::FftDirection::Forward,
                Normalization::Inverse,
            )
            .with_execution_policy(ExecutionPolicy::Sequential),
        )
        .unwrap();
        let plan_inverse = RealTransformPlan::new(
            scioxide_fft::RealTransformPlanSpec::new(
                input.len(),
                (ty).into(),
                scioxide_fft::FftDirection::Inverse,
                Normalization::Inverse,
            )
            .with_execution_policy(ExecutionPolicy::Sequential),
        )
        .unwrap();
        let mut scratch = plan.workspace(()).unwrap();
        let mut output = vec![0.0; plan.spec().length()];
        plan.execute_into(
            &input,
            &mut output,
            &mut scratch,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
        plan_inverse
            .execute_into(
                &input,
                &mut output,
                &mut scratch,
                scioxide_fft::FftPlanningContext::<f64>::default().execution(),
            )
            .unwrap();
    }
}

#[test]
fn nd_prepared_transforms_execute() {
    nd_allocation_case([3, 5, 4]);
    nd_allocation_case([2, 3, 514]);
    // Even smooth axes take the lane-block paths: real and trig blocks, and
    // complex rows of up to 32 points.
    nd_allocation_case([6, 3, 64]);
    nd_allocation_case([16, 2, 256]);
}

#[test]
fn single_axis_and_short_lane_transforms_execute() {
    use scioxide_core::{NdView, NdViewMut, StridedLayout};
    use scioxide_fft::{
        FftNdInput, FftNdOutput, FftNdPlan, FftNdPlanSpec, RealTransformNdPlan,
        RealTransformNdPlanSpec,
    };
    let cx = scioxide_core::ExecutionContext::sequential();
    // Axis 0 is strided: one selected axis makes every lane a batch, and the
    // short axis of the two-axis plan runs across lanes.
    let shape = [6, 129];
    let layout = StridedLayout::new(&shape, &[129, 1]).unwrap();
    let packed_shape = [4, 129];
    let packed_layout = StridedLayout::new(&packed_shape, &[129, 1]).unwrap();
    let real = signal(774, 0.3);
    let complex: Vec<_> = real.iter().map(|&v| Complex::new(v, 0.5 * v)).collect();
    for axes in [&[0][..], &[0, 1]] {
        let plan = FftNdPlan::new(
            FftNdPlanSpec::new(
                &shape,
                axes,
                FftDirection::Forward,
                Normalization::Orthonormal,
                FftLayout::Complex,
            )
            .unwrap(),
        )
        .unwrap();
        let mut workspace = plan.workspace(()).unwrap();
        let mut output = complex.clone();
        (|| {
            plan.execute_into(
                FftNdInput::Complex(NdView::new(&complex, layout)?),
                FftNdOutput::Complex(NdViewMut::new(&mut output, layout)?),
                &mut workspace,
                cx,
            )
        })()
        .unwrap();
    }
    let forward = FftNdPlan::new(
        FftNdPlanSpec::new(
            &shape,
            &[0],
            FftDirection::Forward,
            Normalization::Inverse,
            FftLayout::Real,
        )
        .unwrap(),
    )
    .unwrap();
    let inverse = FftNdPlan::new(
        FftNdPlanSpec::new(
            &shape,
            &[0],
            FftDirection::Inverse,
            Normalization::Inverse,
            FftLayout::Real,
        )
        .unwrap(),
    )
    .unwrap();
    let mut bins = vec![Complex::new(0.0, 0.0); 4 * 129];
    let mut samples = real.clone();
    let mut workspace = forward.workspace(()).unwrap();
    workspace
        .try_reserve(&inverse.workspace_requirements(()).unwrap())
        .unwrap();
    (|| {
        forward.execute_into(
            FftNdInput::Real(NdView::new(&real, layout)?),
            FftNdOutput::Complex(NdViewMut::new(&mut bins, packed_layout)?),
            &mut workspace,
            cx,
        )?;
        inverse.execute_into(
            FftNdInput::Complex(NdView::new(&bins, packed_layout)?),
            FftNdOutput::Real(NdViewMut::new(&mut samples, layout)?),
            &mut workspace,
            cx,
        )
    })()
    .unwrap();
    let cosine = RealTransformNdPlan::new(
        RealTransformNdPlanSpec::new(
            &shape,
            &[0],
            &[DctType::II.into()],
            FftDirection::Forward,
            Normalization::Orthonormal,
        )
        .unwrap(),
    )
    .unwrap();
    let mut workspace = cosine.workspace(()).unwrap();
    let mut output = real.clone();
    (|| {
        cosine.execute_into(
            NdView::new(&real, layout)?,
            NdViewMut::new(&mut output, layout)?,
            &mut workspace,
            cx,
        )
    })()
    .unwrap();
}

#[expect(clippy::unwrap_used)]
fn nd_allocation_case(shape: [usize; 3]) {
    use scioxide_core::{NdView, NdViewMut, StridedLayout};
    use scioxide_fft::{
        FftLayout, FftNdInput, FftNdOutput, FftNdPlan, FftNdPlanSpec, RealTransformNdPlan,
    };
    let strides = [shape[1] * shape[2], shape[2], 1];
    let packed_shape = [shape[0], shape[1], shape[2] / 2 + 1];
    let packed_strides = [shape[1] * packed_shape[2], packed_shape[2], 1];
    let len = shape.iter().product();
    let input = signal(len, 0.2);
    let mut real_output = vec![0.0; len];
    let mut packed = vec![Complex::new(0.0, 0.0); packed_shape.iter().product()];
    let layout = StridedLayout::new(&shape, &strides).unwrap();
    let packed_layout = StridedLayout::new(&packed_shape, &packed_strides).unwrap();
    let complex: Vec<_> = input.iter().map(|&v| Complex::new(v, -v)).collect();
    let plan = FftNdPlan::new(
        FftNdPlanSpec::new(
            &shape,
            &[0, 2],
            FftDirection::Forward,
            Normalization::Orthonormal,
            FftLayout::Complex,
        )
        .unwrap(),
    )
    .unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let mut output = complex.clone();
    for _ in 0..2 {
        (|| {
            plan.execute_into(
                FftNdInput::Complex(NdView::new(&complex, layout)?),
                FftNdOutput::Complex(NdViewMut::new(&mut output, layout)?),
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
        })()
        .unwrap();
    }
    for family in [FftLayout::Real, FftLayout::Hermitian] {
        let forward_direction = if family == FftLayout::Real {
            FftDirection::Forward
        } else {
            FftDirection::Inverse
        };
        let inverse_direction = if family == FftLayout::Real {
            FftDirection::Inverse
        } else {
            FftDirection::Forward
        };
        let forward = FftNdPlan::new(
            FftNdPlanSpec::new(
                &shape,
                &[0, 2],
                forward_direction,
                Normalization::Inverse,
                family,
            )
            .unwrap(),
        )
        .unwrap();
        let inverse = FftNdPlan::new(
            FftNdPlanSpec::new(
                &shape,
                &[0, 2],
                inverse_direction,
                Normalization::Inverse,
                family,
            )
            .unwrap(),
        )
        .unwrap();
        let mut scratch = vec![
            Complex::new(0.0, 0.0);
            forward
                .workspace_requirements(())
                .unwrap()
                .complex_len()
                .max(inverse.workspace_requirements(()).unwrap().complex_len())
        ];
        (|| {
            forward.execute_into(
                FftNdInput::Real(NdView::new(&input, layout)?),
                FftNdOutput::Complex(NdViewMut::new(&mut packed, packed_layout)?),
                &mut scratch,
                scioxide_fft::FftPlanningContext::<f64>::default().execution(),
            )
        })()
        .unwrap();
        (|| {
            inverse.execute_into(
                FftNdInput::Complex(NdView::new(&packed, packed_layout)?),
                FftNdOutput::Real(NdViewMut::new(&mut real_output, layout)?),
                &mut scratch,
                scioxide_fft::FftPlanningContext::<f64>::default().execution(),
            )
        })()
        .unwrap();
    }
    let cosine = RealTransformNdPlan::new(
        scioxide_fft::RealTransformNdPlanSpec::new(
            &shape,
            &[0, 2],
            &[DctType::II, DctType::IV].map(Into::into),
            scioxide_fft::FftDirection::Forward,
            Normalization::Orthonormal,
        )
        .unwrap(),
    )
    .unwrap();
    let sine = RealTransformNdPlan::new(
        scioxide_fft::RealTransformNdPlanSpec::new(
            &shape,
            &[0, 2],
            &[DstType::IV, DstType::III].map(Into::into),
            scioxide_fft::FftDirection::Inverse,
            Normalization::Orthonormal,
        )
        .unwrap(),
    )
    .unwrap();
    let mut workspace = cosine.workspace(()).unwrap();
    workspace
        .try_reserve(&sine.workspace_requirements(()).unwrap())
        .unwrap();
    (|| {
        cosine.execute_into(
            NdView::new(&input, layout)?,
            NdViewMut::new(&mut real_output, layout)?,
            &mut workspace,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
    })()
    .unwrap();
    (|| {
        sine.execute_into(
            NdView::new(&input, layout)?,
            NdViewMut::new(&mut real_output, layout)?,
            &mut workspace,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
    })()
    .unwrap();
}

#[test]
fn empty_axis_shifts_need_no_storage() {
    scioxide_fft::fftshift_nd::<f64>(&[], &[usize::MAX, 0], &[0, 1]).unwrap();
}

#[test]
fn czt_execution_and_batches_execute() {
    use scioxide_core::{ExecutionContext, NdView, NdViewMut, StridedLayout};
    use scioxide_fft::{CztContour, CztPlan, CztPlanSpec};
    for (radius, step, phase, amplitude) in [
        (0.0, 0.0, 0.125, 1.0),
        (0.25, 0.0, 0.0, 1.0),
        (0.0, -0.999_f64.ln(), 0.0, 1.0),
        (800.0, 0.0, 0.0, 1.0),
        (0.25, 0.0, 0.0, f64::from_bits(1)),
    ] {
        let plan = CztPlan::build(
            CztPlanSpec::new(
                17,
                23,
                CztContour::new(radius, step, phase, 0.0).unwrap(),
                Normalization::Unscaled,
            ),
            ExecutionContext::sequential(),
        )
        .unwrap();
        let input = vec![amplitude; 34];
        let mut output = vec![scioxide_core::Complex::new(0.0, 0.0); 46];
        let mut scratch = plan.workspace(()).unwrap();
        plan.execute_into(
            &input[..17],
            &mut output[..23],
            &mut scratch,
            scioxide_fft::FftPlanningContext::<f64>::default().execution(),
        )
        .unwrap();
        let mut work =
            scioxide_fft::FftWorkspace::from_vec(vec![
                scioxide_core::Complex::new(0.0, 0.0);
                plan.workspace_requirements(
                    scioxide_fft::AxisWorkspaceRequest::Axis {
                        shape: &[2, 17],
                        axis: 1
                    }
                )
                .unwrap()
                .complex_len()
            ]);
        plan.execute_axis_into(
            NdView::new(&input, StridedLayout::new(&[2, 17], &[17, 1]).unwrap()).unwrap(),
            1,
            NdViewMut::new(&mut output, StridedLayout::new(&[2, 23], &[23, 1]).unwrap()).unwrap(),
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    }
}

#[test]
fn czt_f32_complex_admitted_fallback_and_subnormal_batches_execute() {
    use scioxide_core::{ExecutionContext, NdView, NdViewMut, StridedLayout};
    use scioxide_fft::{AxisWorkspaceRequest, CztContour, CztPlan, CztPlanSpec};
    for (radius, amplitude) in [(0.25_f32, 1.0_f32), (800.0, 1.0), (0.25, f32::from_bits(1))] {
        let plan = CztPlan::new(CztPlanSpec::new(
            17,
            23,
            CztContour::new(radius, 0.0, 0.0, 0.0).unwrap(),
            Normalization::Unscaled,
        ))
        .unwrap();
        let input = vec![Complex::new(amplitude, amplitude); 34];
        let mut output = vec![Complex::new(0.0, 0.0); 46];
        let mut plain = plan.workspace(()).unwrap();
        plan.execute_into(
            &input[..17],
            &mut output[..23],
            &mut plain,
            ExecutionContext::sequential(),
        )
        .unwrap();
        let mut work = plan
            .workspace(AxisWorkspaceRequest::Axis {
                shape: &[2, 17],
                axis: 1,
            })
            .unwrap();
        let shape = [2, 17];
        let out_shape = [2, 23];
        let layout = StridedLayout::new(&shape, &[17, 1]).unwrap();
        let out_layout = StridedLayout::new(&out_shape, &[23, 1]).unwrap();
        plan.execute_axis_into(
            NdView::new(&input, layout).unwrap(),
            1,
            NdViewMut::new(&mut output, out_layout).unwrap(),
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    }
}
