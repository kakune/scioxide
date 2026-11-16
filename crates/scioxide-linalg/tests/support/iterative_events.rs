//! Provider faults shared across every matrix-free iterative family.
use scioxide_core::{AdjointOperator, ExecutionStopReason, LinearOperator, SciOxideError};
use std::cell::Cell;

pub struct Provider {
    pub calls: Cell<usize>,
    pub at: usize,
    pub stop: bool,
}
impl LinearOperator<f64> for Provider {
    fn rows(&self) -> usize {
        16
    }
    fn cols(&self) -> usize {
        16
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        self.calls.set(self.calls.get() + 1);
        if self.calls.get() == self.at && self.stop {
            return Err(SciOxideError::execution_stopped(
                ExecutionStopReason::Cancelled,
            ));
        }
        for (i, (out, x)) in y.iter_mut().zip(x).enumerate() {
            *out = (i + 1) as f64 * x;
        }
        if self.calls.get() == self.at {
            y[0] = f64::NAN;
        }
        Ok(())
    }
}
impl AdjointOperator<f64> for Provider {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        w: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.apply_into(x, y, w)
    }
}

#[allow(unused_macros)]
macro_rules! iterative_event_contract {
    ($name:ident, $solver:expr $(, $inverse:expr)?) => {
        #[test]
        fn $name() {
            use scioxide_core::{
                AttemptOutcome, CancellationToken, ExecutionContext, ExecutionStopReason,
                SciOxideErrorView, NonFiniteOrigin,
            };
            use scioxide_linalg::{
                IterativeWorkspace, LinearSolveTermination, Vector,
            };
            let solver = $solver;
            let b = Vector::from_vec(vec![1.; 16]);
            let mut workspace = IterativeWorkspace::new();
            for at in [1, 2, 4, 7] {
                let provider = $crate::support::iterative_events::Provider {
                    calls: Cell::new(0),
                    at,
                    stop: true,
                };
                let mut x = Vector::from_vec(vec![0.5; 16]);
                let report = solver
                    .solve_into(
                        &provider,
                        &b,
                        $($inverse,)?
                        x.view_mut(),
                        &mut workspace,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                assert_eq!(
                    report.termination(),
                    LinearSolveTermination::ExecutionStopped(ExecutionStopReason::Cancelled)
                );
                assert_eq!(
                    workspace.last_attempt().unwrap().outcome(),
                    AttemptOutcome::Stopped
                );
                assert_eq!(workspace.last_accepted().unwrap().solution(), x.as_slice());
                if let Some(residual) = report.diagnostic_residual_norm() {
                    let actual = x
                        .as_slice()
                        .iter()
                        .enumerate()
                        .fold(0_f64, |n, (i, x)| n.hypot(1. - (i + 1) as f64 * x));
                    assert!((actual - residual).abs() <= 1e-12 * actual.max(1.));
                } else {
                    assert_eq!(x.as_slice(), &[0.5; 16]);
                    assert!(workspace.last_accepted().unwrap().history().is_none());
                }
                provider.calls.set(0);
                let error = solver
                    .solve_with_options(&provider, &b, $($inverse,)? None)
                    .unwrap()
                    .into_converged_solution()
                    .unwrap_err();
                assert!(matches!(
                    error.view(),
                    SciOxideErrorView::ExecutionStopped {
                        reason: ExecutionStopReason::Cancelled,
                        ..
                    }
                ));
            }
            for at in [1, 3] {
                let provider = $crate::support::iterative_events::Provider {
                    calls: Cell::new(0),
                    at,
                    stop: false,
                };
                let mut x = Vector::from_vec(vec![0.5; 16]);
                let error = solver
                    .solve_into(
                        &provider,
                        &b,
                        $($inverse,)?
                        x.view_mut(),
                        &mut workspace,
                        ExecutionContext::sequential(),
                    )
                    .unwrap_err();
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
                assert_eq!(x.as_slice(), &[0.5; 16]);
                assert_eq!(
                    workspace.last_attempt().unwrap().outcome(),
                    AttemptOutcome::Failed
                );
                assert!(workspace.last_accepted().is_none());
            }
            let cancellation = CancellationToken::new();
            cancellation.cancel();
            let provider = $crate::support::iterative_events::Provider {
                calls: Cell::new(0),
                at: usize::MAX,
                stop: false,
            };
            let mut x = Vector::from_vec(vec![0.5; 16]);
            let report = solver
                .solve_into(
                    &provider,
                    &b,
                    $($inverse,)?
                    x.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential().with_cancellation(&cancellation),
                )
                .unwrap();
            assert_eq!(
                report.termination(),
                LinearSolveTermination::ExecutionStopped(ExecutionStopReason::Cancelled)
            );
            assert_eq!(provider.calls.get(), 0);
            assert_eq!(report.iterations(), 0);
            assert!(report.true_residual().is_none());
            assert_eq!(x.as_slice(), &[0.5; 16]);
            assert_eq!(
                workspace.last_attempt().unwrap().outcome(),
                AttemptOutcome::Stopped
            );
        }
    };
}

#[allow(unused_imports)]
pub(crate) use iterative_event_contract;
