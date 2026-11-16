//! Owned answers for initial-value problems.
use crate::termination::{Terminal, stop_error};
use crate::{
    DormandPrince5, OdeFn, OdeMethod, OdeOptions, OdeOutputSchedule, OdeOutputSink, OdeProblem,
    OdeProgress, OdeState,
};
use scioxide_core::{
    ExecutionContext, InvalidInputReason, NumericScalar, PartialFailure, RealScalar, SciOxideError,
    argument, quantity,
};
use scioxide_linalg::{Matrix, Vector};

/// Integrates `y′ = rhs(t, y)` from `(t0, y0)` to `t1` and returns `y(t1)`.
///
/// Uses Dormand–Prince 5(4), RMS error control with absolute tolerance `1e-6`
/// and relative tolerance `1e-3`. Backward integration is supported. Storage is
/// linear in the state length; work follows the number of adaptive steps.
/// For stiff equations, use a method value such as [`crate::RadauIia`].
///
/// # Errors
/// Returns `InvalidInput` for an empty state or equal endpoints, `NonFinite` for
/// non-finite inputs or callback values, allocation and callback errors, and
/// `ConvergenceFailure` when integration stops before `t1`. Accepted progress is
/// retained in the failure.
///
/// # Examples
/// ```
/// use scioxide_ode::solve_ivp;
/// let y = solve_ivp(|_t: f64, y: &[f64], dy: &mut [f64]| {
///     dy[0] = -y[0]; Ok(())
/// }, 0.0, 1.0, &[1.0]).map_err(scioxide_core::SciOxideError::from)?;
/// assert!((y.as_slice()[0] - (-1.0_f64).exp()).abs() < 1e-3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
/// # References
/// Dormand and Prince (1980); Hairer, Nørsett and Wanner,
/// *Solving Ordinary Differential Equations I*, II.4–II.6.
#[allow(
    clippy::result_large_err,
    reason = "owned evolution retains accepted progress"
)]
pub fn solve_ivp<S, T, F>(
    rhs: F,
    t0: S,
    t1: S,
    y0: &[T],
) -> Result<Vector<T>, PartialFailure<OdeProgress<S, T>>>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    F: Fn(S, &[T], &mut [T]) -> Result<(), SciOxideError>,
{
    validate_initial(t0, t1, y0, argument!("t1")).map_err(PartialFailure::new)?;
    let system = OdeFn::new(y0.len(), rhs).map_err(PartialFailure::new)?;
    let problem = OdeProblem::new(&system, t1);
    let options = OdeOptions::default();
    let method = DormandPrince5::new();
    let mut state = OdeState::from_slice(t0, y0).map_err(PartialFailure::new)?;
    let mut ws = method
        .workspace(&problem, &options)
        .map_err(PartialFailure::new)?;
    match method.advance(
        &problem,
        &options,
        &mut state,
        &mut (),
        &mut ws,
        ExecutionContext::sequential_native(),
    ) {
        Ok(report) => {
            if let Some(stop) = report.termination().and_then(Terminal::stop) {
                return Err(ws.shell.into_failure(stop_error(stop, report.accounting())));
            }
            Ok(state.into_parts().1)
        }
        Err(error) => Err(ws.shell.into_failure(error)),
    }
}

/// Integrates `y′ = rhs(t, y)` at every entry of `times`, returning one row per time.
///
/// The first row is `y0`. Times are monotone in either direction; repeated times
/// give identical rows, and the endpoints must differ. Uses the Dormand–Prince
/// 5(4) method and tolerances of [`solve_ivp`], with its fourth-order continuous
/// extension. Storage is `O(times.len() * y0.len())`; sampling does not alter steps.
///
/// # Errors
/// Returns the errors of [`solve_ivp`], and `InvalidInput` naming `times` for fewer
/// than two entries, equal endpoints or a non-monotone grid.
///
/// # Examples
/// ```
/// use scioxide_ode::solve_ivp_at;
/// let y = solve_ivp_at(|_t: f64, y: &[f64], dy: &mut [f64]| {
///     dy[0] = -y[0]; Ok(())
/// }, &[0.0, 0.5, 1.0], &[1.0]).map_err(scioxide_core::SciOxideError::from)?;
/// assert!((y.view().get(2, 0)? - (-1.0_f64).exp()).abs() < 1e-3);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
/// # References
/// Dormand and Prince (1980); Hairer, Nørsett and Wanner,
/// *Solving Ordinary Differential Equations I*, II.4–II.6.
#[allow(
    clippy::result_large_err,
    reason = "owned evolution retains accepted progress"
)]
pub fn solve_ivp_at<S, T, F>(
    rhs: F,
    times: &[S],
    y0: &[T],
) -> Result<Matrix<T>, PartialFailure<OdeProgress<S, T>>>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    F: Fn(S, &[T], &mut [T]) -> Result<(), SciOxideError>,
{
    if times.len() < 2 {
        return Err(PartialFailure::new(SciOxideError::invalid_input(
            argument!("times"),
            InvalidInputReason::Empty,
            "at least two times are required",
        )));
    }
    let (t0, t1) = (times[0], times[times.len() - 1]);
    validate_initial(t0, t1, y0, argument!("times")).map_err(PartialFailure::new)?;
    super::driver::validate_requested_times(times, t0, t1, argument!("times"))
        .map_err(PartialFailure::new)?;
    let system = OdeFn::new(y0.len(), rhs).map_err(PartialFailure::new)?;
    let problem = OdeProblem::new(&system, t1);
    let options = OdeOptions::default().with_output_schedule(OdeOutputSchedule::times(
        scioxide_core::backend::try_copy_slice(times, quantity!("times"))
            .map_err(PartialFailure::new)?,
    ));
    let method = DormandPrince5::new();
    let mut state = OdeState::from_slice(t0, y0).map_err(PartialFailure::new)?;
    let mut ws = method
        .workspace(&problem, &options)
        .map_err(PartialFailure::new)?;
    let mut sink = Samples(Matrix::zeros(times.len(), y0.len()).map_err(PartialFailure::new)?);
    match method.advance(
        &problem,
        &options,
        &mut state,
        &mut sink,
        &mut ws,
        ExecutionContext::sequential_native(),
    ) {
        Ok(report) => {
            if let Some(stop) = report.termination().and_then(Terminal::stop) {
                return Err(ws.shell.into_failure(stop_error(stop, report.accounting())));
            }
            Ok(sink.0)
        }
        Err(error) => Err(ws.shell.into_failure(error)),
    }
}
fn validate_initial<S: RealScalar, T>(
    t0: S,
    t1: S,
    y0: &[T],
    endpoint: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if y0.is_empty() {
        return Err(SciOxideError::invalid_input(
            argument!("y0"),
            InvalidInputReason::Empty,
            "initial state must not be empty",
        ));
    }
    if t0.is_finite() && t1.is_finite() && t0 == t1 {
        return Err(SciOxideError::invalid_input(
            endpoint,
            InvalidInputReason::Inconsistent,
            "endpoints must differ",
        ));
    }
    Ok(())
}
struct Samples<T: NumericScalar>(Matrix<T>);
impl<S: RealScalar, T: NumericScalar<Real = S>> OdeOutputSink<S, T> for Samples<T> {
    fn emit(&mut self, _: S, _: &[T]) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn emit_requested(&mut self, index: usize, _: S, state: &[T]) -> Result<(), SciOxideError> {
        for (j, &value) in state.iter().enumerate() {
            self.0.view_mut().set(index, j, value)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::termination::Stop;
    use scioxide_core::{
        Budget, BudgetKind, ConvergenceReason, ExecutionStopReason, SciOxideErrorView, SolveCounts,
    };
    #[test]
    fn every_shared_stop_maps_to_its_reason_and_counts() {
        for (stop, expected) in [
            (
                Stop::Budget(BudgetKind::Evaluation),
                ConvergenceReason::EvaluationLimit,
            ),
            (
                Stop::Budget(BudgetKind::Iteration),
                ConvergenceReason::IterationLimit,
            ),
            (
                Stop::Budget(BudgetKind::Proposal),
                ConvergenceReason::ResourceLimit(BudgetKind::Proposal),
            ),
            (
                Stop::Budget(BudgetKind::Subdivision),
                ConvergenceReason::ResourceLimit(BudgetKind::Subdivision),
            ),
            (
                Stop::Budget(BudgetKind::Sample),
                ConvergenceReason::ResourceLimit(BudgetKind::Sample),
            ),
            (
                Stop::Budget(BudgetKind::Node),
                ConvergenceReason::ResourceLimit(BudgetKind::Node),
            ),
            (Stop::StepSize, ConvergenceReason::StepSizeLimit),
        ] {
            let counts = SolveCounts::new();
            let error = stop_error(stop, counts);
            assert!(
                matches!(error.view(),SciOxideErrorView::ConvergenceFailure{reason,diagnostics,..} if reason==expected && diagnostics.counts()==counts)
            );
            assert_eq!(crate::OdeTermination::from_stop(stop).stop(), Some(stop));
        }
        for reason in [
            ExecutionStopReason::Cancelled,
            ExecutionStopReason::DeadlineReached,
        ] {
            let error = stop_error(Stop::Execution(reason), SolveCounts::new());
            assert!(
                matches!(error.view(),SciOxideErrorView::ExecutionStopped{reason:actual,..} if reason==actual)
            );
        }
    }
    #[test]
    fn a_proposal_stop_maps_with_the_last_accepted_point() {
        let system = OdeFn::new(1, |_: f64, y: &[f64], dy: &mut [f64]| {
            dy[0] = -y[0];
            Ok(())
        })
        .unwrap();
        let problem = OdeProblem::new(&system, 1.);
        let method = DormandPrince5::new();
        let options = OdeOptions::default().with_max_steps(Budget::ONE);
        let mut state = OdeState::from_slice(0., &[1.]).unwrap();
        let mut ws = method.workspace(&problem, &options).unwrap();
        let report = method
            .advance(
                &problem,
                &options,
                &mut state,
                &mut (),
                &mut ws,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert_eq!(
            report.termination(),
            Some(crate::OdeTermination::BudgetLimit(BudgetKind::Proposal))
        );
        let failure = ws.shell.into_failure(stop_error(
            report.termination().unwrap().stop().unwrap(),
            report.accounting(),
        ));
        assert_eq!(failure.progress().unwrap().time(), state.time());
        assert_eq!(failure.progress().unwrap().state(), state.values());
        assert!(
            matches!(failure.error().view(),SciOxideErrorView::ConvergenceFailure{reason:ConvergenceReason::ResourceLimit(BudgetKind::Proposal),diagnostics,..} if diagnostics.counts()==report.accounting())
        );
    }
}
