//! The four selected-axis families share one transactional layout contract.
use scioxide_core::{
    Complex, ExecutionContext, NdView, NdViewMut, SciOxideErrorKind, StridedLayout,
};
use scioxide_fft::*;

macro_rules! check_axis {
    ($plan:expr, $n:expr, $m:expr) => {{
        let plan = $plan;
        let (n, m) = ($n, $m);
        let cx = ExecutionContext::sequential();
        let plain = plan.workspace_requirements(()).unwrap();
        for axis in 0..3 {
            let mut shape = [2, 2, 2];
            shape[axis] = n;
            let mut out_shape = shape;
            out_shape[axis] = m;
            let strides = [shape[1] * shape[2] * 2, shape[2] * 2, 2];
            let out_strides = [out_shape[1] * out_shape[2] * 3, out_shape[2] * 3, 3];
            let layout = StridedLayout::new(&shape, &strides).unwrap();
            let out_layout = StridedLayout::new(&out_shape, &out_strides).unwrap();
            let input: Vec<_> = (0..shape.iter().product::<usize>() * 2)
                .map(|j| Complex::new((j as f64 * 0.3).sin(), 0.25))
                .collect();
            let sentinel = Complex::new(91.0, -13.0);
            let mut output = vec![sentinel; out_shape.iter().product::<usize>() * 3];
            let request = AxisWorkspaceRequest::Axis {
                shape: &shape,
                axis,
            };
            let requirements = plan.workspace_requirements(request).unwrap();
            // Whole output, one input lane and the plan's own transactional scratch.
            assert_eq!(requirements.complex_len(), 4 * m + n + plain.complex_len());
            let mut workspace = plan.workspace(request).unwrap();
            assert_eq!(workspace.len(), requirements.complex_len());
            plan.execute_axis_into(
                NdView::new(&input, layout).unwrap(),
                axis,
                NdViewMut::new(&mut output, out_layout).unwrap(),
                &mut workspace,
                cx,
            )
            .unwrap();
            let mut lane_workspace = plan.workspace(()).unwrap();
            for batch in 0..4 {
                let lane: Vec<_> = layout
                    .axis(axis)
                    .unwrap()
                    .lane(batch)
                    .unwrap()
                    .offsets()
                    .map(|i| input[i])
                    .collect();
                let mut expected = vec![Complex::new(0.0, 0.0); m];
                plan.execute_into(&lane, &mut expected, &mut lane_workspace, cx)
                    .unwrap();
                for (offset, expected) in out_layout
                    .axis(axis)
                    .unwrap()
                    .lane(batch)
                    .unwrap()
                    .offsets()
                    .zip(expected)
                {
                    assert!((output[offset] - expected).norm() <= 1e-9 * (1.0 + expected.norm()));
                }
            }
            assert!(
                output
                    .iter()
                    .enumerate()
                    .all(|(i, &v)| i % 3 == 0 || v == sentinel)
            );
            let before = output.clone();
            // A parent solve scope nests the same transaction.
            let mut state = scioxide_core::SolveState::new();
            let mut budgets = scioxide_core::SolveBudgets::unbounded();
            let mut parent = state
                .begin_solve(
                    cx,
                    scioxide_core::SolveIdentity::new(
                        scioxide_core::ProblemToken::new(0),
                        scioxide_core::algorithm_id!("fft", "axis", 1),
                        0,
                        0,
                    ),
                    &mut budgets,
                )
                .unwrap();
            plan.execute_axis_into(
                NdView::new(&input, layout).unwrap(),
                axis,
                NdViewMut::new(&mut output, out_layout).unwrap(),
                &mut workspace,
                &mut parent,
            )
            .unwrap();
            assert_eq!(output, before);
            let mut bad = input.clone();
            let last = layout
                .axis(axis)
                .unwrap()
                .lane(3)
                .unwrap()
                .offsets()
                .last()
                .unwrap();
            bad[last].re = f64::NAN;
            let error = plan
                .execute_axis_into(
                    NdView::new(&bad, layout).unwrap(),
                    axis,
                    NdViewMut::new(&mut output, out_layout).unwrap(),
                    &mut workspace,
                    cx,
                )
                .unwrap_err();
            assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
            assert_eq!(output, before);
            let mut short = FftWorkspace::from_vec(vec![
                Complex::new(0.0, 0.0);
                requirements.complex_len() - 1
            ]);
            assert_eq!(
                plan.execute_axis_into(
                    NdView::new(&input, layout).unwrap(),
                    axis,
                    NdViewMut::new(&mut output, out_layout).unwrap(),
                    &mut short,
                    cx
                )
                .unwrap_err()
                .kind(),
                SciOxideErrorKind::DimensionMismatch
            );
            assert_eq!(output, before);
            let cancellation = scioxide_core::CancellationToken::new();
            cancellation.cancel();
            assert_eq!(
                plan.execute_axis_into(
                    NdView::new(&input, layout).unwrap(),
                    axis,
                    NdViewMut::new(&mut output, out_layout).unwrap(),
                    &mut workspace,
                    cx.with_cancellation(&cancellation)
                )
                .unwrap_err()
                .kind(),
                SciOxideErrorKind::ExecutionStopped
            );
            assert_eq!(output, before);
            // Real inputs can broadcast outside the selected axis.
            let mut broadcast = [0, 0, 0];
            broadcast[axis] = 1;
            let real: Vec<_> = (0..n).map(|i| i as f64 * 0.25).collect();
            let broadcast = StridedLayout::new_broadcast(&shape, &broadcast).unwrap();
            plan.execute_axis_into(
                NdView::new(&real, broadcast).unwrap(),
                axis,
                NdViewMut::new(&mut output, out_layout).unwrap(),
                &mut workspace,
                cx,
            )
            .unwrap();
        }
        let shape = [0, n];
        let out_shape = [0, m];
        let request = AxisWorkspaceRequest::Axis {
            shape: &shape,
            axis: 1,
        };
        let empty = plan.workspace_requirements(request).unwrap();
        assert_eq!(empty.complex_len(), 0);
        assert_eq!(empty.byte_limit(), plain.byte_limit());
        let mut work = plan.workspace(request).unwrap();
        assert_eq!(work.len(), 0);
        plan.execute_axis_into(
            NdView::<f64>::new(&[], StridedLayout::new(&shape, &[0, 0]).unwrap()).unwrap(),
            1,
            NdViewMut::new(&mut [], StridedLayout::new(&out_shape, &[0, 0]).unwrap()).unwrap(),
            &mut work,
            cx,
        )
        .unwrap();
    }};
}

#[test]
fn czt_axes_are_atomic_and_execute() {
    check_axis!(
        CztPlan::new(CztPlanSpec::new(
            3,
            5,
            CztContour::dft(5).unwrap(),
            Normalization::Unscaled
        ))
        .unwrap(),
        3,
        5
    );
}
#[test]
fn nufft_axes_are_atomic_and_execute() {
    let points = [0.0, 0.25, 0.75];
    check_axis!(
        NufftPlan::new(
            NufftTransform::Type1 {
                points: NdView::new(&points, StridedLayout::new(&[3, 1], &[1, 1]).unwrap())
                    .unwrap(),
                modes: &[5],
            },
            NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, 1e-6)
        )
        .unwrap(),
        3,
        5
    );
}
#[test]
fn hankel_axes_are_atomic_and_execute() {
    let zeros: Vec<_> = (1..=4).map(|i| i as f64 * core::f64::consts::PI).collect();
    check_axis!(
        DiscreteHankelPlan::new(
            &zeros,
            DiscreteHankelPlanSpec::new(0.5, 4.0, FftDirection::Forward)
                .with_tolerance(scioxide_core::Tolerance::new(1e-10, 1e-10).unwrap())
        )
        .unwrap(),
        3,
        3
    );
}
#[test]
fn fftlog_axes_are_atomic_and_execute() {
    check_axis!(
        FftLogPlan::new(FftLogPlanSpec::new(
            3,
            0.1,
            0.0,
            0.0,
            Complex::new(0.0, 0.0),
            FftDirection::Forward
        ))
        .unwrap(),
        3,
        3
    );
}

#[test]
fn empty_nufft_records_still_produce_zero_output_lanes() {
    let plan = NufftPlan::<f64>::new(
        NufftTransform::Type1 {
            points: NdView::new(&[], StridedLayout::new(&[0, 1], &[0, 0]).unwrap()).unwrap(),
            modes: &[3],
        },
        NufftPlanSpec::new(FftDirection::Forward, Normalization::Unscaled, 1e-6),
    )
    .unwrap();
    let shape = [0, 2];
    let out_shape = [3, 2];
    let mut output = [Complex::new(91.0, -13.0); 6];
    let mut work = plan
        .workspace(AxisWorkspaceRequest::Axis {
            shape: &shape,
            axis: 0,
        })
        .unwrap();
    plan.execute_axis_into(
        NdView::<f64>::new(&[], StridedLayout::new(&shape, &[0, 0]).unwrap()).unwrap(),
        0,
        NdViewMut::new(
            &mut output,
            StridedLayout::new(&out_shape, &[2, 1]).unwrap(),
        )
        .unwrap(),
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output, [Complex::new(0.0, 0.0); 6]);
}
