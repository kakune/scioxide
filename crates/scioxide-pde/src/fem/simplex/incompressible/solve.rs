//! Preconditioned steady flow solves with workspace-owned solvers.
use super::*;
use alloc::boxed::Box;
use core::cell::RefCell;
use scioxide_core::{Preconditioner, ResidualTolerance};
use scioxide_linalg::{
    GmresSolver, IterativeLinearSolver, IterativeSolveReport, IterativeSolveResult,
    IterativeWorkspace, IterativeWorkspaceRequest, Preconditioning, Vector, VectorView,
    VectorViewMut,
};
use scioxide_rootfind::{
    NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovResult, NewtonKrylovSolver,
    NewtonKrylovWorkspace, NonlinearLinearSolver,
};

/// Linear, nonlinear and library-preconditioner controls of flow solves.
#[derive(Clone, Debug)]
#[must_use]
pub struct IncompressibleSolveOptions<R: RealScalar> {
    residual: ResidualTolerance<R>,
    iterations: Budget,
    restart: Budget,
    preconditioner: IncompressiblePreconditionerOptions<R>,
    nonlinear: NewtonKrylovOptions<R>,
}
impl<R: RealScalar> Default for IncompressibleSolveOptions<R> {
    fn default() -> Self {
        Self {
            residual: ResidualTolerance::from_tolerance(Tolerance::from_sqrt_epsilon()),
            iterations: Budget::ONE.saturating_add(999),
            restart: Budget::ONE.saturating_add(49),
            preconditioner: IncompressiblePreconditionerOptions::default(),
            nonlinear: NewtonKrylovOptions::default(),
        }
    }
}
impl<R: RealScalar> IncompressibleSolveOptions<R> {
    /// Square-root-epsilon true residual, 1,000 GMRES(50) iterations, the default
    /// library preconditioner and default Newton–Krylov controls.
    pub fn new() -> Self {
        Self::default()
    }
    /// Absolute/relative true-residual threshold of linear and saddle solves.
    pub const fn with_residual(mut self, residual: ResidualTolerance<R>) -> Self {
        self.residual = residual;
        self
    }
    /// Total Krylov iterations of the default solver.
    pub const fn with_iterations(mut self, iterations: Budget) -> Self {
        self.iterations = iterations;
        self
    }
    /// GMRES restart length of the default solver.
    pub const fn with_restart(mut self, restart: Budget) -> Self {
        self.restart = restart;
        self
    }
    /// Sparse library preconditioner preparation.
    pub const fn with_preconditioner(
        mut self,
        preconditioner: IncompressiblePreconditionerOptions<R>,
    ) -> Self {
        self.preconditioner = preconditioner;
        self
    }
    /// Newton–Krylov controls of Navier–Stokes solves.
    pub fn with_nonlinear(mut self, nonlinear: NewtonKrylovOptions<R>) -> Self {
        self.nonlinear = nonlinear;
        self
    }
    /// True-residual threshold.
    pub const fn residual(&self) -> ResidualTolerance<R> {
        self.residual
    }
    /// Krylov iterations.
    pub const fn iterations(&self) -> Budget {
        self.iterations
    }
    /// GMRES restart length.
    pub const fn restart(&self) -> Budget {
        self.restart
    }
    /// Library preconditioner preparation.
    pub const fn preconditioner(&self) -> &IncompressiblePreconditionerOptions<R> {
        &self.preconditioner
    }
    /// Newton–Krylov controls.
    pub const fn nonlinear(&self) -> &NewtonKrylovOptions<R> {
        &self.nonlinear
    }
    pub(super) fn solver<V: NumericScalar<Real = R>>(&self) -> GmresSolver<V> {
        GmresSolver::new(self.residual, self.iterations).with_restart(self.restart)
    }
}

/// Linear solver, saddle preconditioner and Krylov storage of steady Stokes solves.
///
/// [`StokesPlan::workspace`] installs restarted GMRES and the library viscous
/// inverse; `with_*` replaces either for every later call.
pub struct StokesWorkspace<V: NumericScalar> {
    solver: Box<dyn IterativeLinearSolver<V> + Send>,
    preconditioner: Box<dyn Preconditioner<V> + Send>,
    linear: IterativeWorkspace<V>,
}
impl<V: NumericScalar> core::fmt::Debug for StokesWorkspace<V> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("StokesWorkspace")
            .field("solver", &self.solver.algorithm_id())
            .finish_non_exhaustive()
    }
}
impl<V: NumericScalar> StokesWorkspace<V> {
    /// Solve with `solver`, which must accept nonsymmetric indefinite systems.
    #[must_use]
    pub fn with_linear_solver(mut self, solver: Box<dyn IterativeLinearSolver<V> + Send>) -> Self {
        self.solver = solver;
        self
    }
    /// Precondition with `preconditioner` instead of the library inverse.
    #[must_use]
    pub fn with_preconditioner(
        mut self,
        preconditioner: Box<dyn Preconditioner<V> + Send>,
    ) -> Self {
        self.preconditioner = preconditioner;
        self
    }
    /// Krylov storage, including its last attempt.
    pub const fn linear(&self) -> &IterativeWorkspace<V> {
        &self.linear
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> StokesPlan<R, V, D>
where
    IncompressiblePreconditioner<V>: Send,
{
    /// Restarted GMRES and the library viscous inverse, prepared under `options`.
    /// # Errors
    /// Returns sparse preparation, resource and allocation failures.
    pub fn workspace(
        &self,
        options: &IncompressibleSolveOptions<R>,
    ) -> Result<StokesWorkspace<V>, SciOxideError> {
        Ok(StokesWorkspace {
            solver: Box::new(options.solver::<V>()),
            preconditioner: Box::new(IncompressiblePreconditioner::new(
                self,
                IncompressibleSaddle::Viscous,
                *options.preconditioner(),
            )?),
            linear: IterativeWorkspace::new(),
        })
    }

    /// Solve the eliminated Stokes system into `solution`, which holds the initial
    /// guess, as one epoch of `scope`. Krylov storage grows before any callback; the
    /// solver's transactional contract leaves `solution` unchanged on error. The
    /// returned report is authoritative, including unsuccessful termination.
    /// # Errors
    /// Returns shape, preconditioner, provider and numerical failures.
    pub fn solve_into<'scope, 'capability>(
        &self,
        solution: VectorViewMut<'_, V>,
        workspace: &mut StokesWorkspace<V>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<IterativeSolveReport<R>, SciOxideError> {
        let StokesWorkspace {
            solver,
            preconditioner,
            linear,
        } = workspace;
        let request = IterativeWorkspaceRequest::operator(self)?
            .with_preconditioner(Preconditioning::Inverse(&**preconditioner))?;
        linear.try_reserve(&solver.workspace_requirements(&request)?)?;
        solver.solve_into(
            self,
            VectorView::from_slice(self.rhs()),
            Preconditioning::Inverse(&**preconditioner),
            solution,
            linear,
            scope.into().into(),
        )
    }

    /// Solve Stokes flow with default GMRES tolerances and block preconditioning.
    /// # Errors
    /// Returns preparation and numerical errors, and rejects every unsuccessful termination.
    pub fn solve(&self) -> Result<alloc::vec::Vec<V>, SciOxideError> {
        Ok(self
            .solve_with_options(&IncompressibleSolveOptions::default())?
            .into_converged_solution()?
            .into_vec())
    }
    /// Solve from zero with library block preconditioning and restarted GMRES under
    /// sequential native execution.
    /// # Errors
    /// Returns sparse preparation, allocation, provider and numerical failures.
    ///
    /// ```
    /// # fn solve(plan: &scioxide_pde::StokesPlan<f64, f64, 2>) -> Result<(), scioxide_core::SciOxideError> {
    /// let result = plan.solve_with_options(&scioxide_pde::IncompressibleSolveOptions::new())?;
    /// assert!(result.is_success());
    /// # Ok(()) }
    /// ```
    pub fn solve_with_options(
        &self,
        options: &IncompressibleSolveOptions<R>,
    ) -> Result<IterativeSolveResult<V>, SciOxideError> {
        let inverse = IncompressiblePreconditioner::new(
            self,
            IncompressibleSaddle::Viscous,
            *options.preconditioner(),
        )?;
        options.solver::<V>().solve_with_options(
            self,
            VectorView::from_slice(self.rhs()),
            Preconditioning::Inverse(&inverse),
            None,
        )
    }
}

/// Adapter from a workspace-owned iterative solver to Newton–Krylov corrections.
/// Injected solvers must meet the requested nonlinear forcing accuracy; the
/// Newton driver independently checks every correction.
struct Correction<'a, V: NumericScalar> {
    solver: &'a dyn IterativeLinearSolver<V>,
    preconditioner: &'a dyn Preconditioner<V>,
    workspace: RefCell<&'a mut IterativeWorkspace<V>>,
}
impl<V: NumericScalar> NonlinearLinearSolver<V> for Correction<'_, V> {
    fn solve_into(
        &self,
        operator: &scioxide_rootfind::NonlinearLinearization<'_, V>,
        rhs: &Vector<V>,
        output: &mut Vector<V>,
        _tolerance: ResidualTolerance<V::Real>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<IterativeSolveReport<V::Real>, SciOxideError> {
        let scope = scioxide_core::SolveScope::from(context);
        let operator = operator.operator();
        let mut workspace = self.workspace.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("workspace"),
                "linear correction workspace already borrowed",
            )
        })?;
        output.as_mut_slice().fill(V::ZERO);
        self.solver.solve_into(
            operator,
            rhs.view(),
            Preconditioning::Inverse(self.preconditioner),
            output.view_mut(),
            &mut workspace,
            scope.into(),
        )
    }
}

/// Newton–Krylov plan, frozen saddle preconditioner, optional injected correction
/// solver and storage of steady Navier–Stokes solves.
pub struct NavierStokesWorkspace<R: RealScalar> {
    newton: NewtonKrylovSolver<R>,
    nonlinear: NewtonKrylovWorkspace<R>,
    preconditioner: Box<dyn Preconditioner<R> + Send>,
    solver: Option<Box<dyn IterativeLinearSolver<R> + Send>>,
    correction: IterativeWorkspace<R>,
}
impl<R: RealScalar> core::fmt::Debug for NavierStokesWorkspace<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("NavierStokesWorkspace")
            .field("injected_solver", &self.solver.is_some())
            .finish_non_exhaustive()
    }
}
impl<R: RealScalar> NavierStokesWorkspace<R> {
    /// Solve Newton corrections with `solver` instead of the built-in Krylov method.
    /// It must be configured tightly enough for the nonlinear forcing checks.
    #[must_use]
    pub fn with_linear_solver(mut self, solver: Box<dyn IterativeLinearSolver<R> + Send>) -> Self {
        self.solver = Some(solver);
        self
    }
    /// Precondition corrections with `preconditioner` instead of the frozen library inverse.
    #[must_use]
    pub fn with_preconditioner(
        mut self,
        preconditioner: Box<dyn Preconditioner<R> + Send>,
    ) -> Self {
        self.preconditioner = preconditioner;
        self
    }
    /// Newton–Krylov storage, including its last attempt.
    pub const fn nonlinear(&self) -> &NewtonKrylovWorkspace<R> {
        &self.nonlinear
    }
}

impl<R: RealScalar, const D: usize> NavierStokesPlan<R, D>
where
    IncompressiblePreconditioner<R>: Send,
{
    /// The Newton–Krylov plan of `options` and the frozen library Stokes inverse.
    /// # Errors
    /// Returns option, sparse preparation and allocation failures.
    pub fn workspace(
        &self,
        options: &IncompressibleSolveOptions<R>,
    ) -> Result<NavierStokesWorkspace<R>, SciOxideError> {
        let newton = NewtonKrylovSolver::new(options.nonlinear().clone())?;
        let preconditioner = IncompressiblePreconditioner::new(
            self.stokes(),
            IncompressibleSaddle::Viscous,
            *options.preconditioner(),
        )?;
        let nonlinear = newton.workspace(
            &NewtonKrylovProblem::with_jacobian(self).with_preconditioner(&preconditioner),
        )?;
        Ok(NavierStokesWorkspace {
            newton,
            nonlinear,
            preconditioner: Box::new(preconditioner),
            solver: None,
            correction: IterativeWorkspace::new(),
        })
    }

    /// Solve from `initial` into `output` with analytic Newton–Krylov as one epoch
    /// of `scope`. Globalization, nonlinear tolerances and diagnostics use the root
    /// solver's contracts; no Reynolds-independent convergence is promised. An
    /// error preserves `output`.
    /// # Errors
    /// Returns preparation, shape, callback, budget and arithmetic failures.
    pub fn solve_into<'scope, 'capability>(
        &self,
        initial: &[R],
        output: &mut NewtonKrylovResult<R>,
        workspace: &mut NavierStokesWorkspace<R>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let NavierStokesWorkspace {
            newton,
            nonlinear,
            preconditioner,
            solver,
            correction,
        } = workspace;
        let problem =
            NewtonKrylovProblem::with_jacobian(self).with_preconditioner(&**preconditioner);
        match solver {
            None => {
                nonlinear.try_reserve(&newton.workspace_requirements(&problem)?)?;
                newton.solve_into(&problem, initial, output, nonlinear, scope)
            }
            Some(solver) => {
                let tangent = crate::PdeTangent::new(self, initial)?;
                let request = IterativeWorkspaceRequest::operator(&tangent)?
                    .with_preconditioner(Preconditioning::Inverse(&**preconditioner))?;
                correction.try_reserve(&solver.workspace_requirements(&request)?)?;
                let correction = Correction {
                    solver: &**solver,
                    preconditioner: &**preconditioner,
                    workspace: RefCell::new(correction),
                };
                let problem =
                    NewtonKrylovProblem::with_jacobian(self).with_linear_solver(&correction);
                nonlinear.try_reserve(&newton.workspace_requirements(&problem)?)?;
                newton.solve_into(&problem, initial, output, nonlinear, scope)
            }
        }
    }

    /// Solve Navier–Stokes from `initial` with default Newton–Krylov settings.
    /// # Errors
    /// Returns preparation, callback and numerical failures, including unsuccessful terminations.
    pub fn solve(&self, initial: &[R]) -> Result<alloc::vec::Vec<R>, SciOxideError> {
        let result = self.solve_with_options(initial, &IncompressibleSolveOptions::default())?;
        require_nonlinear_success(result.termination(), result.counts())?;
        scioxide_core::backend::try_copy_slice(
            result.point(),
            scioxide_core::quantity!("flow_solution"),
        )
    }
    /// Solve from `initial` with the default workspace of `options` under
    /// sequential native execution.
    /// # Errors
    /// Returns preparation, allocation, shape, callback and arithmetic failures.
    pub fn solve_with_options(
        &self,
        initial: &[R],
        options: &IncompressibleSolveOptions<R>,
    ) -> Result<NewtonKrylovResult<R>, SciOxideError> {
        let mut workspace = self.workspace(options)?;
        let mut output = NewtonKrylovResult::new(self.unknowns(), options.nonlinear())?;
        self.solve_into(
            initial,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}

pub(super) fn require_nonlinear_success(
    termination: scioxide_rootfind::RootTermination,
    counts: scioxide_core::EvaluationCounts,
) -> Result<(), SciOxideError> {
    use scioxide_core::{ConvergenceReason as Reason, ExecutionStopReason};
    use scioxide_rootfind::RootTermination as Stop;
    if termination.is_success() {
        return Ok(());
    }
    let reason = match termination {
        Stop::Cancelled => {
            return Err(SciOxideError::execution_stopped(
                ExecutionStopReason::Cancelled,
            ));
        }
        Stop::TimeLimit => {
            return Err(SciOxideError::execution_stopped(
                ExecutionStopReason::DeadlineReached,
            ));
        }
        Stop::IterationLimit => Reason::IterationLimit,
        Stop::Stagnation => Reason::Stagnation,
        Stop::WorkLimit(kind) => Reason::ResourceLimit(kind),
        Stop::LinearFailure(_) => Reason::NestedSolver,
        _ => Reason::NoSuccessfulCandidate,
    };
    Err(SciOxideError::convergence_failure(
        reason,
        scioxide_core::ConvergenceDiagnostics::new().with_counts(counts.into()),
        "Navier–Stokes solve did not converge",
    ))
}
