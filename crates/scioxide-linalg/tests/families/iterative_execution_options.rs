//! Execution options composed by iterative solvers: workspace byte limits,
//! capped iteration allowances and time limits.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    AdjointOperator, Budget, Capacity, ExecutionContext, LinearOperator, ResidualTolerance,
    SciOxideError, SciOxideErrorKind,
};
use scioxide_linalg::{
    BicgstabSolver, CgSolver, CgsSolver, GcrotSolver, GmresSolver, IterativeWorkspace,
    IterativeWorkspaceRequest, LinalgExecutionOptions, LinearSolveTermination, MinresSolver,
    Preconditioning, RelaxationWeight, RichardsonSolver, TfqmrSolver, VectorView, VectorViewMut,
};

const N: usize = 6;

/// `tridiag(-1, 4, -1)` that also declares provider scratch.
struct Tridiagonal;
impl LinearOperator<f64> for Tridiagonal {
    fn rows(&self) -> usize {
        N
    }
    fn cols(&self) -> usize {
        N
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(3)
    }
    fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        for (i, out) in y.iter_mut().enumerate() {
            let left = if i > 0 { x[i - 1] } else { 0. };
            let right = x.get(i + 1).copied().unwrap_or(0.);
            *out = 4. * x[i] - left - right;
        }
        Ok(())
    }
}
impl AdjointOperator<f64> for Tridiagonal {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(5)
    }
    fn apply_adjoint_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.apply_into(x, y, scratch)
    }
}

fn tolerance() -> ResidualTolerance<f64> {
    ResidualTolerance::new(0., 1e-12).unwrap()
}
fn budget(limit: usize) -> Budget {
    Budget::try_from(limit).unwrap()
}
fn limited(bytes: usize) -> LinalgExecutionOptions {
    LinalgExecutionOptions::new().with_byte_limit(bytes)
}

/// A fresh reservation retains exactly the bytes its requirements declare.

#[test]
fn byte_limits_count_capacity_retained_from_earlier_reservations() {
    let a = Tridiagonal;
    let solver = CgSolver::<f64>::new(tolerance(), budget(20));
    let staged = IterativeWorkspaceRequest::operator(&a)
        .unwrap()
        .with_right_hand_sides(4);
    let mut workspace = solver.workspace(&staged).unwrap();
    let single = IterativeWorkspaceRequest::operator(&a)
        .unwrap()
        .with_right_hand_sides(1);
    let required = solver
        .workspace_requirements(&single)
        .unwrap()
        .required_bytes();
    let limited = solver
        .with_execution_options(limited(required))
        .workspace_requirements(&single)
        .unwrap();
    assert_eq!(
        IterativeWorkspace::new().check(&limited).unwrap(),
        Capacity::NeedsGrowth
    );
    for result in [
        workspace.check(&limited).map(|_| ()),
        workspace.try_reserve(&limited),
    ] {
        assert_eq!(result.unwrap_err().kind(), SciOxideErrorKind::ResourceLimit);
    }
}

#[test]
fn execution_iteration_budget_caps_the_solver_limit() {
    let a = Tridiagonal;
    let b = [1., 0., 0., 0., 0., 1.];
    let solve = |solver: CgSolver<f64>| {
        let mut x = [0.; N];
        let mut workspace = IterativeWorkspace::new();
        solver
            .solve_into(
                &a,
                VectorView::from_slice(&b),
                Preconditioning::None,
                VectorViewMut::from_slice(&mut x),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap()
    };
    let capped = solve(
        CgSolver::new(tolerance(), budget(50))
            .with_execution_options(LinalgExecutionOptions::new().with_iteration_budget(budget(2))),
    );
    assert_eq!(capped.termination(), LinearSolveTermination::IterationLimit);
    assert_eq!(capped.iterations(), 2);
    let positional = solve(
        CgSolver::new(tolerance(), budget(2)).with_execution_options(
            LinalgExecutionOptions::new().with_iteration_budget(budget(50)),
        ),
    );
    assert_eq!(
        positional.termination(),
        LinearSolveTermination::IterationLimit
    );
    assert_eq!(positional.iterations(), 2);
    assert!(solve(CgSolver::new(tolerance(), budget(50))).is_success());

    let history = CgSolver::<f64>::new(tolerance(), budget(50))
        .with_history(true)
        .with_execution_options(LinalgExecutionOptions::new().with_iteration_budget(budget(4)))
        .workspace_requirements(&IterativeWorkspaceRequest::operator(&a).unwrap())
        .unwrap();
    assert_eq!(history.history_capacity(), 5);
}

#[test]
fn owned_solves_reject_time_limits_without_a_clock() {
    let solver = GmresSolver::<f64>::new(tolerance(), budget(20)).with_execution_options(
        LinalgExecutionOptions::new().with_time_limit(core::time::Duration::from_secs(1)),
    );
    let error = solver
        .solve_with_options(&Tridiagonal, &[1.; N][..], Preconditioning::None, None)
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::CapabilityUnavailable);
    assert_eq!(
        solver.execution_options().time_limit(),
        Some(core::time::Duration::from_secs(1))
    );
}

#[test]
fn caller_iteration_cap_survives_reborrowing_and_solver_delegation() {
    use scioxide_linalg::{IterativeLinearSolver, LinalgExecutionContext};
    for own_limit in [1, 100] {
        let solvers: Vec<Box<dyn IterativeLinearSolver<f64>>> = vec![
            Box::new(CgSolver::new(tolerance(), budget(own_limit))),
            Box::new(GmresSolver::new(tolerance(), budget(own_limit))),
        ];
        for solver in solvers {
            let mut context = LinalgExecutionContext::sequential()
                .with_iteration_budget(budget(2))
                .with_iteration_budget(budget(4));
            for _ in 0..2 {
                let mut x = [0.0; N];
                let b = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
                let report = solver
                    .solve_into(
                        &Tridiagonal,
                        VectorView::from_slice(&b),
                        Preconditioning::None,
                        VectorViewMut::from_slice(&mut x),
                        &mut IterativeWorkspace::new(),
                        context.reborrow(),
                    )
                    .unwrap();
                assert_eq!(report.termination(), LinearSolveTermination::IterationLimit);
                assert_eq!(report.iterations(), own_limit.min(2));
            }
        }
    }
}

#[test]
fn caller_iteration_cap_publishes_the_same_iterate_as_the_solver_limit() {
    use scioxide_linalg::{IterativeLinearSolver, LinalgExecutionContext};
    let b = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
    type Make = fn(usize) -> Box<dyn IterativeLinearSolver<f64>>;
    let solvers: [Make; 8] = [
        |k| Box::new(CgSolver::new(tolerance(), budget(k))),
        |k| Box::new(GmresSolver::new(tolerance(), budget(k))),
        |k| Box::new(BicgstabSolver::new(tolerance(), budget(k))),
        |k| Box::new(MinresSolver::new(tolerance(), budget(k))),
        |k| Box::new(CgsSolver::new(tolerance(), budget(k))),
        |k| Box::new(TfqmrSolver::new(tolerance(), budget(k))),
        |k| Box::new(GcrotSolver::new(tolerance(), budget(k))),
        |k| {
            Box::new(RichardsonSolver::new(
                tolerance(),
                budget(k),
                RelaxationWeight::new(0.2).unwrap(),
            ))
        },
    ];
    for make in solvers {
        let solve = |solver: Box<dyn IterativeLinearSolver<f64>>, cap: usize| {
            let mut x = [0.0; N];
            let report = solver
                .solve_into(
                    &Tridiagonal,
                    VectorView::from_slice(&b),
                    Preconditioning::None,
                    VectorViewMut::from_slice(&mut x),
                    &mut IterativeWorkspace::new(),
                    LinalgExecutionContext::sequential().with_iteration_budget(budget(cap)),
                )
                .unwrap();
            (x, report)
        };
        let (own, own_report) = solve(make(2), 100);
        let (capped, capped_report) = solve(make(100), 2);
        assert_eq!(capped_report.termination(), own_report.termination());
        assert_eq!(capped_report.iterations(), 2);
        assert!(own.iter().any(|&value| value != 0.0));
        for (left, right) in own.iter().zip(&capped) {
            assert!((left - right).abs() <= 1e-14 * left.abs().max(1.0));
        }
    }
}
