//! Action requirements include every retained buffer, including nested factors.
#![allow(clippy::unwrap_used)]

use crate::support::action_operator::Operator;

use scioxide_core::{ProblemToken, RankThreshold, SciOxideError, SciOxideErrorKind};
use scioxide_linalg::*;

#[test]
fn overflowing_requirements_and_foreign_families_are_rejected() -> Result<(), SciOxideError> {
    let op = Operator::new(usize::MAX, 0);
    assert_eq!(
        OneNormEstimator::new()
            .workspace_requirements::<f64>(&op)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::SizeOverflow
    );
    let plan = RandomizedLowRank::new(1);
    assert_eq!(
        plan.workspace_requirements::<f64>(&op, 1)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::SizeOverflow
    );
    Ok(())
}

#[test]
fn option_deadlines_stop_with_accepted_progress_without_allocating() -> Result<(), SciOxideError> {
    use crate::support::clocks::Frozen;
    use scioxide_core::{
        BlockVectorView, BlockVectorViewMut, ClockBinding, ExecutionContext, ExecutionStopReason,
    };
    let clock = Frozen;
    let binding = ClockBinding::new(&clock);
    let op = Operator::new(3, 0);
    let options = ExponentialAction::new(1.).with_execution_options(
        scioxide_linalg::LinalgExecutionOptions::new().with_time_limit(core::time::Duration::ZERO),
    );
    let plan = options;
    let req = plan.workspace_requirements(&op, 1)?;
    let mut w = ExponentialActionWorkspace::new();
    w.try_reserve(&req)?;
    let mut output = [42.; 3];
    let report = (|| {
        plan.apply_into(
            &op,
            scioxide_linalg::PhiFunction::phi(0, 1.),
            BlockVectorView::new(&[1.; 3], 3, 1)?,
            BlockVectorViewMut::new(&mut output, 3, 1)?,
            &mut w,
            ExecutionContext::sequential().with_clock(&binding)?,
        )
    })()?;
    assert_eq!(
        report.termination(),
        SpectralTermination::ExecutionStopped(ExecutionStopReason::DeadlineReached)
    );
    assert_eq!(output, [1.; 3]);
    assert_eq!(op.calls.get(), 0);
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::Stopped
    );
    let plan = FunctionAction::default().with_execution_options(
        scioxide_linalg::LinalgExecutionOptions::new().with_time_limit(core::time::Duration::ZERO),
    );
    let mut w = plan.workspace(&op, 1)?;
    let report = (|| {
        plan.apply_into(
            &op,
            MatrixFunction::Sqrt,
            BlockVectorView::new(&[1.; 3], 3, 1)?,
            BlockVectorViewMut::new(&mut output, 3, 1)?,
            &mut w,
            ExecutionContext::sequential().with_clock(&binding)?,
        )
    })()?;
    assert_eq!(
        report.termination(),
        SpectralTermination::ExecutionStopped(ExecutionStopReason::DeadlineReached)
    );
    assert_eq!(op.calls.get(), 0);
    assert_eq!(output, [0.; 3]);
    Ok(())
}

#[test]
fn projected_function_time_limits_require_a_clock_before_providers() -> Result<(), SciOxideError> {
    use scioxide_core::{BlockVectorView, BlockVectorViewMut, ExecutionContext};
    let op = Operator::new(3, 0);
    let plan = FunctionAction::default().with_execution_options(
        scioxide_linalg::LinalgExecutionOptions::new()
            .with_time_limit(core::time::Duration::from_secs(1)),
    );
    let mut w = plan.workspace(&op, 1)?;
    let mut output = [42.; 3];
    let error = plan
        .apply_into(
            &op,
            MatrixFunction::Sqrt,
            BlockVectorView::new(&[1.; 3], 3, 1)?,
            BlockVectorViewMut::new(&mut output, 3, 1)?,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::CapabilityUnavailable);
    assert_eq!(op.calls.get(), 0);
    assert_eq!(output, [42.; 3]);
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::PreflightRejected
    );
    Ok(())
}

#[test]
fn unreserved_actions_report_capacity_shortfalls() -> Result<(), SciOxideError> {
    use scioxide_core::{BlockVectorView, BlockVectorViewMut, ExecutionContext, SciOxideErrorView};
    let op = Operator::new(3, 0);
    let input = [1.; 3];
    let mut output = [42.; 3];
    let cx = ExecutionContext::sequential();
    macro_rules! short {
        ($call:expr) => {{
            let error = $call.unwrap_err();
            assert!(matches!(error.view(),SciOxideErrorView::DimensionMismatch {expected,actual,..} if expected.len() == 1 && expected[0] > 0 && actual == [0]),"{error}");
        }};
    }
    short!(ExponentialAction::new(1.).apply_into(
        &op,
        scioxide_linalg::PhiFunction::phi(0, 1.),
        BlockVectorView::new(&input, 3, 1)?,
        BlockVectorViewMut::new(&mut output, 3, 1)?,
        &mut ExponentialActionWorkspace::new(),
        cx
    ));
    short!(FunctionAction::default().apply_into(
        &op,
        MatrixFunction::Sqrt,
        BlockVectorView::new(&input, 3, 1)?,
        BlockVectorViewMut::new(&mut output, 3, 1)?,
        &mut FunctionActionWorkspace::new(),
        cx
    ));
    short!(PartialSpectrumSolver::new(1).solve_into(
        &op,
        input.as_slice(),
        &mut PartialSpectrumBuffer::new(3, 1)?,
        &mut PartialSpectrumWorkspace::new(),
        cx
    ));
    short!(PartialSvdSolver::new(1).solve_into(
        &op,
        &input,
        &mut PartialSvdBuffer::new(3, 3, 1)?,
        &mut PartialSvdWorkspace::new(),
        cx
    ));
    short!(LobpcgSolver::new(1).solve_into(
        &op,
        BlockVectorView::new(&input, 3, 1)?,
        BlockVectorView::new(&[], 3, 0)?,
        &mut LobpcgBuffer::new(3, 1)?,
        &mut LobpcgWorkspace::new(),
        cx
    ));
    let sketch = Matrix::from_vec(3, 1, input.to_vec())?;
    short!(RandomizedLowRank::new(1).svd_into(
        &op,
        sketch.view(),
        sketch.view(),
        &mut RandomizedSvdBuffer::new(3, 3, 1)?,
        &mut RandomizedLowRankWorkspace::new(),
        cx
    ));
    short!(OneNormEstimator::new().estimate_scoped(&op, &mut OneNormWorkspace::<f64>::new(), cx));
    let mut space = KrylovRecycleSpace::<f64>::new(3, 2, ProblemToken::new(0))?;
    short!(space.refresh(
        &op,
        ProblemToken::new(1),
        RankThreshold::default(),
        &LinalgExecutionOptions::new(),
        &mut KrylovRecycleWorkspace::new(),
        cx
    ));
    assert_eq!(output, [42.; 3]);
    assert_eq!(op.calls.get(), 0);
    Ok(())
}

/// Identity-shaped operator whose every application returns NaN.
struct NanOperator;
impl scioxide_core::LinearOperator<f64> for NanOperator {
    fn rows(&self) -> usize {
        3
    }
    fn cols(&self) -> usize {
        3
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, _: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        y.fill(f64::NAN);
        Ok(())
    }
}
impl scioxide_core::AdjointOperator<f64> for NanOperator {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
    ) -> Result<(), SciOxideError> {
        scioxide_core::LinearOperator::apply_into(self, x, y, w)
    }
}

#[test]
fn nonfinite_provider_output_has_callback_origin() -> Result<(), SciOxideError> {
    use scioxide_core::{AttemptOutcome, ExecutionContext, NonFiniteOrigin, SciOxideErrorView};
    let callback = |error: SciOxideError| {
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Callback,
                    ..
                }
            ),
            "{error:?}"
        );
    };
    let plan = OneNormEstimator::new();
    let mut w = plan.workspace(&NanOperator)?;
    callback(
        plan.estimate_scoped(&NanOperator, &mut w, ExecutionContext::sequential())
            .unwrap_err(),
    );
    assert_eq!(w.last_attempt().unwrap().outcome(), AttemptOutcome::Failed);
    let plan = PartialSpectrumSolver::new(1);
    let mut w = plan.workspace(&NanOperator)?;
    let mut spectrum = PartialSpectrumBuffer::new(3, 1)?;
    callback(
        plan.solve_into(
            &NanOperator,
            &[1., 1., 1.][..],
            &mut spectrum,
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap_err(),
    );
    assert_eq!(w.last_attempt().unwrap().outcome(), AttemptOutcome::Failed);
    Ok(())
}
