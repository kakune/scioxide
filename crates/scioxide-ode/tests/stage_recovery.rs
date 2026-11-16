//! Trial failures reject atomically; explicit provider errors remain failures.
//!
//! Injected trial failures, minimum-step and proposal-budget retention and SDIRK inner
//! iteration limits use crate-private steppers and are covered by the crate's internal
//! driver tests.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    AttemptOutcome, ErrorTolerance, EvaluationRole, ExecutionContext, NormSelection, SciOxideError,
    SolveContext,
};
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::{
    AdamsBdf, Bdf, MultistepOrder, OdeMethod, OdeOptions, OdeProblem, OdeState, OdeSystem,
    OdeTermination,
};

struct ProviderError;
impl OdeSystem<f64> for ProviderError {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        _: &[f64],
        _: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        Err(SciOxideError::singular(None))
    }
}
#[test]
fn provider_numerical_errors_are_not_retried() {
    let plan = (
        AdamsBdf::adams_bdf().unwrap(),
        OdeOptions::new().with_initial_step(0.1),
    );
    let problem = OdeProblem::new(&ProviderError, 1.);
    let mut workspace = plan.0.workspace(&problem, &plan.1).unwrap();
    let mut state = OdeState::new(0., Vector::from_vec(vec![1.]));
    let error = plan
        .0
        .advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::Singular { .. }
    ));
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    assert_eq!(attempt.counts().own().rejected(), 0);
    assert_eq!(attempt.progress().unwrap().rejected_steps(), 0);
    assert_eq!(state.values(), &[1.]);
}
struct Growth;
impl OdeSystem<f64> for Growth {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = 1000. * y[0];
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        _: &[f64],
        j: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        j.set(0, 0, 1000.)?;
        Ok(Some(()))
    }
}
#[test]
fn singular_iteration_matrix_refreshes_and_retries() {
    let plan = (
        Bdf::with_order_policy(MultistepOrder::Automatic(5)).unwrap(),
        OdeOptions::new()
            .with_initial_step(0.001)
            .with_error_tolerance(
                ErrorTolerance::scalar(1e-10, 1e-7, NormSelection::RootMeanSquare).unwrap(),
            ),
    );
    let solution = plan
        .0
        .solve(
            &OdeProblem::new(&Growth, 0.001),
            &plan.1,
            OdeState::new(0., Vector::from_vec(vec![1.])),
        )
        .unwrap();
    let report = solution.report();
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert!(report.counts().rejected() > 0);
    assert!((solution.state().values()[0] - 1_f64.exp()).abs() < 3e-6);
    assert!(report.counts().provider(EvaluationRole::Jacobian) > 1);
}
