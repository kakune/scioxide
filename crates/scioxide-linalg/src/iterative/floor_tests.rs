//! Checkpoint traces for iterates returned at the rounding floor.
#![allow(clippy::unwrap_used)]
use super::test_support::TraceGuard;
use crate::*;
use alloc::vec;
use scioxide_core::{
    AdjointOperator, BlockVectorView, Budget, LinearOperator, ResidualTolerance, SciOxideError,
};

struct Poisson(usize);
impl LinearOperator<f64> for Poisson {
    fn rows(&self) -> usize {
        self.0 * self.0
    }
    fn cols(&self) -> usize {
        self.rows()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        for (i, out) in y.iter_mut().enumerate() {
            *out = 4. * x[i];
            if i % self.0 > 0 {
                *out -= x[i - 1];
            }
            if i % self.0 + 1 < self.0 {
                *out -= x[i + 1];
            }
            if i >= self.0 {
                *out -= x[i - self.0];
            }
            if i + self.0 < x.len() {
                *out -= x[i + self.0];
            }
        }
        Ok(())
    }
}
impl AdjointOperator<f64> for Poisson {
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
fn assert_trace(trace: super::test_support::CheckpointTrace, iterations: usize, residual: f64) {
    let minimum = trace
        .samples
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min);
    assert_eq!(minimum, residual);
    let first = trace.samples.iter().find(|p| p.1 == minimum).unwrap().0;
    assert!(
        iterations <= 2 * first + 50,
        "stopped at {iterations}; minimum first at {first}"
    );
}
macro_rules! scalar {
    ($name:ident, $solver:ident $(, $p:expr)?) => {
        #[test]
        fn $name() {
            let a = Poisson(128);
            let b = Vector::from_vec(vec![1.; a.rows()]);
            let solver = $solver::new(ResidualTolerance::new(0., 1e-16).unwrap(), Budget::try_from(4000).unwrap());
            let trace = TraceGuard::start(a.0, b.as_slice().to_vec());
            let result = solver.solve_with_options(&a, &b, $($p,)? None).unwrap();
            let report = result.report();
            assert!(matches!(report.termination(), LinearSolveTermination::Stagnation | LinearSolveTermination::Breakdown(LinearSolveBreakdown::RecurrenceDenominator)), "{report:?}");
            assert_trace(trace.finish(), report.iterations(), report.diagnostic_residual_norm().unwrap());
        }
    };
}
scalar!(cg, CgSolver, Preconditioning::None);
scalar!(bicgstab, BicgstabSolver, Preconditioning::None);
scalar!(gmres, GmresSolver, Preconditioning::None);
scalar!(lgmres, LgmresSolver, Preconditioning::None);
scalar!(gcrot, GcrotSolver, Preconditioning::None);
scalar!(minres, MinresSolver, Preconditioning::None);
scalar!(lsqr, LsqrSolver);
scalar!(lsmr, LsmrSolver);
scalar!(bicg, BicgSolver, Preconditioning::None);
scalar!(qmr, QmrSolver, Preconditioning::None);
scalar!(tfqmr, TfqmrSolver, Preconditioning::None);

/// CGS squares the BiCG residual polynomial, so its updated residuals grow far
/// beyond `‖b‖` and their rounding errors keep the true residual above the
/// floor at the best iterate (Greenbaum 1997). It then runs to the iteration
/// limit and returns that best iterate.
#[test]
fn cgs() {
    let a = Poisson(128);
    let b = Vector::from_vec(vec![1.; a.rows()]);
    let solver = CgsSolver::new(
        ResidualTolerance::new(0., 1e-16).unwrap(),
        Budget::try_from(4000).unwrap(),
    );
    let trace = TraceGuard::start(a.0, b.as_slice().to_vec());
    let result = solver
        .solve_with_options(&a, &b, Preconditioning::None, None)
        .unwrap();
    let report = result.report();
    assert!(
        matches!(
            report.termination(),
            LinearSolveTermination::Stagnation
                | LinearSolveTermination::IterationLimit
                | LinearSolveTermination::Breakdown(LinearSolveBreakdown::RecurrenceDenominator)
        ),
        "{report:?}"
    );
    let trace = trace.finish();
    let minimum = trace
        .samples
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min);
    assert_eq!(minimum, report.diagnostic_residual_norm().unwrap());
}

macro_rules! block {
    ($name:ident, $solver:ident) => {
        #[test]
        fn $name() {
            let a = Poisson(128);
            let b = vec![1.; 2 * a.rows()];
            let solver = $solver::new(
                ResidualTolerance::new(0., 1e-16).unwrap(),
                Budget::try_from(4000).unwrap(),
            );
            let trace = TraceGuard::start(a.0, b.clone());
            let result = solver
                .solve(
                    &a,
                    BlockVectorView::new(&b, a.rows(), 2).unwrap(),
                    Preconditioning::None,
                    None,
                )
                .unwrap();
            let report = result.report();
            assert_eq!(
                report.termination(),
                LinearSolveTermination::Stagnation,
                "{report:?}"
            );
            assert_trace(
                trace.finish(),
                report.iterations(),
                result.residuals().iter().copied().fold(0., f64::max),
            );
        }
    };
}
block!(block_gmres, BlockGmresSolver);
block!(block_lgmres, BlockLgmresSolver);
block!(block_gcrot, BlockGcrotSolver);

struct Interruptible {
    a: Poisson,
    calls: core::cell::Cell<usize>,
    stop: usize,
}
impl LinearOperator<f64> for Interruptible {
    fn rows(&self) -> usize {
        self.a.rows()
    }
    fn cols(&self) -> usize {
        self.a.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], w: &mut [f64]) -> Result<(), SciOxideError> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if call == self.stop {
            return Err(SciOxideError::execution_stopped(
                scioxide_core::ExecutionStopReason::Cancelled,
            ));
        }
        self.a.apply_into(x, y, w)
    }
}
impl AdjointOperator<f64> for Interruptible {
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

#[test]
fn interrupted_scalar_solves_publish_the_minimum_completed_checkpoint() {
    use scioxide_core::{
        BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let tolerance = ResidualTolerance::new(0., 1e-30).unwrap();
    let limit = Budget::try_from(800).unwrap();
    macro_rules! check {
        ($solver:ident $(, $inverse:expr)?) => {{
            for (calls, provider_stop) in [(1, true), (2, true), (51, true), (103, true), (151, false), (251, false)] {
                let a = Interruptible { a: Poisson(16), calls: core::cell::Cell::new(0), stop: if provider_stop { calls } else { usize::MAX } };
                let b = vec![1.; a.rows()];
                let mut x = Vector::zeros(a.rows()).unwrap();
                let solver = $solver::new(tolerance, limit);
                let mut workspace = solver.workspace(&IterativeWorkspaceRequest::adjoint(&a).unwrap()).unwrap();
                let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(if provider_stop { 10000 } else { calls }).unwrap());
                let mut state = SolveState::new();
                let mut context = state.begin_solve(ExecutionContext::sequential(), SolveIdentity::new(ProblemToken::new(0), scioxide_core::algorithm_id!("test", "floor", 1), 0, 0), &mut budgets).unwrap();
                let trace = TraceGuard::start(16, b.clone());
                let report = solver.solve_into(&a, &b[..], $($inverse,)? x.view_mut(), &mut workspace, &mut context).unwrap();
                let trace = trace.finish();
                if trace.samples.is_empty() {
                    assert!(report.diagnostic_residual_norm().is_none());
                    assert!(x.as_slice().iter().all(|&v| v == 0.));
                } else {
                    assert_eq!(report.diagnostic_residual_norm().unwrap(), trace.samples.iter().map(|p| p.1).fold(f64::INFINITY, f64::min), "{}: {report:?}", stringify!($solver));
                }
            }
        }};
    }
    check!(CgSolver, Preconditioning::None);
    check!(BicgstabSolver, Preconditioning::None);
    check!(GmresSolver, Preconditioning::None);
    check!(LgmresSolver, Preconditioning::None);
    check!(GcrotSolver, Preconditioning::None);
    check!(MinresSolver, Preconditioning::None);
    check!(LsqrSolver);
    check!(LsmrSolver);
    check!(BicgSolver, Preconditioning::None);
    check!(CgsSolver, Preconditioning::None);
    check!(QmrSolver, Preconditioning::None);
    check!(TfqmrSolver, Preconditioning::None);
}

#[test]
fn interrupted_blocks_publish_the_minimum_complete_checkpoint() {
    use scioxide_core::{
        BlockVectorViewMut, BudgetKind, ExecutionContext, ProblemToken, SolveBudgets,
        SolveIdentity, SolveState,
    };
    macro_rules! check {
        ($solver:ident) => {{
            for (calls, provider_stop) in [
                (2, true),
                (17, true),
                (51, true),
                (103, true),
                (151, false),
                (251, false),
            ] {
                let a = Interruptible {
                    a: Poisson(16),
                    calls: core::cell::Cell::new(0),
                    stop: if provider_stop { calls } else { usize::MAX },
                };
                let b = vec![1.; 2 * a.rows()];
                let mut x = vec![0.; b.len()];
                let solver = $solver::new(
                    ResidualTolerance::new(0., 1e-30).unwrap(),
                    Budget::try_from(800).unwrap(),
                );
                let mut workspace = solver
                    .workspace(
                        &IterativeWorkspaceRequest::operator(&a)
                            .unwrap()
                            .with_right_hand_sides(2),
                    )
                    .unwrap();
                let mut budgets = SolveBudgets::unbounded().with_limit(
                    BudgetKind::Evaluation,
                    Budget::try_from(if provider_stop { 10000 } else { calls }).unwrap(),
                );
                let mut state = SolveState::new();
                let mut context = state
                    .begin_solve(
                        ExecutionContext::sequential(),
                        SolveIdentity::new(
                            ProblemToken::new(0),
                            scioxide_core::algorithm_id!("test", "floor", 1),
                            0,
                            0,
                        ),
                        &mut budgets,
                    )
                    .unwrap();
                let trace = TraceGuard::start(16, b.clone());
                let _ = solver
                    .solve_into(
                        &a,
                        BlockVectorView::new(&b, a.rows(), 2).unwrap(),
                        Preconditioning::None,
                        BlockVectorViewMut::new(&mut x, a.rows(), 2).unwrap(),
                        &mut workspace,
                        &mut context,
                    )
                    .unwrap();
                let trace = trace.finish();
                if trace.samples.is_empty() {
                    assert!(x.iter().all(|&v| v == 0.));
                    assert!(workspace.last_accepted().unwrap().columns().is_none());
                } else {
                    let residual = workspace
                        .last_accepted()
                        .unwrap()
                        .columns()
                        .unwrap()
                        .residuals()
                        .iter()
                        .copied()
                        .fold(0., f64::max);
                    assert_eq!(
                        residual,
                        trace
                            .samples
                            .iter()
                            .map(|p| p.1)
                            .fold(f64::INFINITY, f64::min)
                    );
                }
            }
        }};
    }
    check!(BlockGmresSolver);
    check!(BlockLgmresSolver);
    check!(BlockGcrotSolver);
}
