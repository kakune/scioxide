use super::*;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use scioxide_core::{PartialResultTrust, ResidualTolerance};
use scioxide_linalg::{
    IterativeLinearSolver, IterativeRequirements, IterativeSolveReport, IterativeWorkspace,
    IterativeWorkspaceRequest, LinalgExecutionContext, LinearSolveTermination, Preconditioning,
    VectorView, VectorViewMut,
};
use std::sync::Mutex;

/// Injection mode, received initial guesses and last iteration count, shared
/// between the test and the solver owned by the workspace.
#[derive(Default)]
struct Observations {
    mode: AtomicU8,
    guesses: Mutex<Vec<Vec<f64>>>,
    iterations: AtomicUsize,
}
impl Observations {
    fn guesses(&self) -> std::sync::MutexGuard<'_, Vec<Vec<f64>>> {
        self.guesses.lock().unwrap()
    }
}
struct ObservedSolver {
    inner: scioxide_linalg::GmresSolver<f64>,
    observed: Arc<Observations>,
}
impl IterativeLinearSolver<f64> for ObservedSolver {
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("test", "observed", 1)
    }
    fn workspace_requirements(
        &self,
        request: &IterativeWorkspaceRequest,
    ) -> Result<IterativeRequirements<f64>, SciOxideError> {
        self.inner.workspace_requirements(request)
    }
    fn solve_into(
        &self,
        operator: &dyn LinearOperator<f64>,
        rhs: VectorView<'_, f64>,
        inverse: Preconditioning<'_, f64>,
        mut output: VectorViewMut<'_, f64>,
        work: &mut IterativeWorkspace<f64>,
        ctx: LinalgExecutionContext<'_, '_>,
    ) -> Result<IterativeSolveReport<f64>, SciOxideError> {
        self.observed
            .guesses()
            .push(output.iter().copied().collect());
        match self.observed.mode.load(Ordering::Relaxed) {
            1 => {
                output.fill(777.);
                return Err(invalid(
                    scioxide_core::argument!("fixture"),
                    "injected linear failure",
                ));
            }
            2 => {
                output.fill(0.);
                return Ok(IterativeSolveReport::new(
                    LinearSolveTermination::Converged,
                    0,
                    None,
                    PartialResultTrust::DiagnosticOnly,
                ));
            }
            _ => {}
        }
        let report = self
            .inner
            .solve_into(operator, rhs, inverse, output, work, ctx)?;
        self.observed
            .iterations
            .store(report.iterations(), Ordering::Relaxed);
        Ok(report)
    }
}

#[test]
fn injected_transient_solver_warm_starts_and_rolls_back_failed_candidates() {
    let p = plan();
    let observed = Arc::new(Observations::default());
    let solver = ObservedSolver {
        inner: scioxide_linalg::GmresSolver::new(
            ResidualTolerance::new(1e-11, 1e-11).unwrap(),
            Budget::try_from(500).unwrap(),
        )
        .with_restart(Budget::try_from(80).unwrap()),
        observed: Arc::clone(&observed),
    };
    let options = IncompressibleSolveOptions::new()
        .with_residual(ResidualTolerance::new(1e-10, 1e-10).unwrap());
    let mut work = IncompressibleTimeWorkspace::new(&p, &options)
        .unwrap()
        .with_linear_solver(Box::new(solver));
    let forcing = |x: [f64; 2], _: f64| Ok([0.2 * x[1], 0.]);
    let system = IncompressibleSystem::stokes(&p, &mut work, Some(&forcing), 0.001).unwrap();
    let exact = problem::state(&p, 0.04, 0.3).unwrap();
    let y = p.free.iter().map(|&i| exact[i]).collect::<Vec<_>>();
    let mut out = vec![0.; y.len()];
    ephemeral(|ctx| system.rhs(0., &y, &mut out, ctx)).unwrap();
    assert!(observed.iterations.load(Ordering::Relaxed) > 0);
    ephemeral(|ctx| system.rhs(0., &y, &mut out, ctx)).unwrap();
    assert_eq!(observed.iterations.load(Ordering::Relaxed), 0);
    assert!(observed.guesses()[0].iter().all(|&v| v == 0.));
    let warm = observed.guesses()[1].clone();
    assert!(warm.iter().any(|v| v.abs() > 1e-6));
    for mode in [1, 2] {
        observed.mode.store(mode, Ordering::Relaxed);
        out.fill(42.);
        assert!(ephemeral(|ctx| system.rhs(0., &y, &mut out, ctx)).is_err());
        assert!(out.iter().all(|&v| v == 42.));
        observed.mode.store(0, Ordering::Relaxed);
        ephemeral(|ctx| system.rhs(0., &y, &mut out, ctx)).unwrap();
        assert_eq!(observed.guesses().last().unwrap(), &warm);
        assert_eq!(observed.iterations.load(Ordering::Relaxed), 0);
    }
    // Explicit projection has its own warm cache and cannot replace acceleration's.
    let mut projected = vec![0.; y.len()];
    ephemeral(|ctx| system.project_velocity_into(&y, &mut projected, ctx)).unwrap();
    ephemeral(|ctx| system.rhs(0., &y, &mut out, ctx)).unwrap();
    assert_eq!(observed.guesses().last().unwrap(), &warm);
}
