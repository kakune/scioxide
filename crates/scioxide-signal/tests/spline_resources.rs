//! Allocation, execution admission and transactional view tests for spline smoothing.
#![allow(clippy::unwrap_used)]
#[path = "support/errors.rs"]
mod errors;
use errors::error_kind;

use scioxide_core::{ExecutionContext, NdView, NdViewMut, SciOxideErrorKind, StridedLayout};
use scioxide_signal::{
    SignalExecutionOptions, SignalLayout, SplineDegree, SplineFilter, SplineRequest,
    WhittakerHendersonPlan,
};

#[test]
fn axis_failures_preserve_every_lane_and_padding() {
    let shape = [3, 4];
    let layout = StridedLayout::new(&shape, &[4, 1]).unwrap();
    let output_layout = StridedLayout::new(&shape, &[8, 2]).unwrap();
    let mut input = [1.; 12];
    input[11] = f64::NAN;
    let mut output = [43.; 24];
    let spline = SplineFilter::new(SplineDegree::Cubic, SignalExecutionOptions::new()).unwrap();
    let mut scratch = spline
        .workspace(SplineRequest::Coefficients(SignalLayout::Axis {
            shape: &shape,
            axis: 1,
        }))
        .unwrap();
    assert_eq!(
        error_kind(spline.coefficients_axis_into(
            NdView::new(&input, layout).unwrap(),
            1,
            NdViewMut::new(&mut output, output_layout).unwrap(),
            &mut scratch,
            ExecutionContext::sequential()
        )),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(output, [43.; 24]);
    let wh = WhittakerHendersonPlan::new(
        4,
        2,
        0.5,
        None,
        scioxide_signal::SignalExecutionOptions::new(),
    )
    .unwrap();
    let mut workspace = wh.workspace(3).unwrap();
    assert_eq!(
        error_kind(wh.smooth_axis_into(
            NdView::new(&input, layout).unwrap(),
            1,
            NdViewMut::new(&mut output, output_layout).unwrap(),
            &mut workspace,
            ExecutionContext::sequential()
        )),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(output, [43.; 24]);
    input[11] = 1.;
    // Three lanes exceed a two-lane workspace before any lane is solved.
    let mut two = wh.workspace(2).unwrap();
    assert_eq!(
        wh.smooth_axis_into(
            NdView::new(&input, layout).unwrap(),
            1,
            NdViewMut::new(&mut output, output_layout).unwrap(),
            &mut two,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, [43.; 24]);
    wh.smooth_axis_into(
        NdView::new(&input, layout).unwrap(),
        1,
        NdViewMut::new(&mut output, output_layout).unwrap(),
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
}

#[test]
fn impossible_plan_sizes_are_rejected_before_implicit_weight_traversal() {
    for n in [usize::MAX, usize::MAX / 4] {
        assert_eq!(
            error_kind(WhittakerHendersonPlan::<f64>::new(
                n,
                2,
                1.,
                None,
                scioxide_signal::SignalExecutionOptions::new()
            )),
            SciOxideErrorKind::SizeOverflow
        );
    }
}
