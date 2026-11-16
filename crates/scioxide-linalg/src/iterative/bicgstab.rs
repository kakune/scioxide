//! Biconjugate gradient stabilized method (BiCGSTAB) for nonsymmetric systems.

use super::support::{true_residual_into_scoped, validate_square};
use super::workspace::{IterativeProgress, Job, Shape, StorageKind};
use crate::Preconditioning;
use crate::kernel::finite;
use alloc::vec::Vec;

use core::fmt;

use crate::dense::Vector;
use crate::iterative::safety::{
    Step, StepFault, checked_divide, inner_product_is_small, shadow_product_is_small,
    should_recompute_true_residual,
};
use crate::result::{IterativeSolveReport, IterativeSolveResult, IterativeSolveSummary};
use crate::traits::IterativeLinearSolver;
use crate::{
    IterativeRequirements, IterativeWorkspace, IterativeWorkspaceRequest, LinalgExecutionContext,
    VectorView, VectorViewMut,
};
use scioxide_core::AlgebraicScalar;
use scioxide_core::LinearOperator;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{
    Budget, ExecutionContext, NumericScalar, PartialResultTrust, ResidualTolerance, SciOxideError,
};

/// Biconjugate gradient stabilized method (BiCGSTAB) for general square systems.
///
/// `A` may be nonsymmetric. BiCGSTAB combines each BiCG step with a one-step
/// minimal residual (GMRES(1)) update, which smooths the convergence of BiCG
/// without products with `Aᴴ`. The preconditioner is applied from the right and
/// must be fixed during a solve. Each iteration costs two products with `A` and
/// two applications of `M⁻¹`; the workspace holds ten vectors of length `n`.
/// Use [`GmresSolver`](crate::GmresSolver) when memory allows and convergence
/// must be monotone.
///
/// The solve converges when `‖b − A x‖₂ ≤ atol + rtol·‖b‖₂`, with `‖b‖₂`
/// replaced by one when `b = 0`. The recursively updated residual is replaced by
/// the true residual `b − A x` every 50 iterations and before convergence is
/// accepted; the shadow residual `r̂ = r₀` and the recurrence are kept.
///
/// The shadow products `r̂ᴴ r` and `r̂ᴴ v` count as breakdown only when their
/// magnitude is at most `ε² ‖r̂‖₂ ‖r‖₂` (respectively `‖v‖₂`), where `ε` is the
/// machine epsilon. The stabilization step uses a threshold that grows with the
/// dimension. Breakdown ends the solve with
/// [`Breakdown(RecurrenceDenominator)`](crate::LinearSolveBreakdown::RecurrenceDenominator)
/// unless [`Self::with_breakdown_restarts`] allows restarts.
///
/// # Examples
///
/// ```rust
/// use scioxide_core::{Budget, SciOxideError, ResidualTolerance};
/// use scioxide_linalg::BicgstabSolver;
///
/// # fn main() -> Result<(), SciOxideError> {
/// let solver = BicgstabSolver::<f64>::new(
///     ResidualTolerance::new(1e-10, 0.0)?,
///     Budget::try_from(500)?,
/// )
/// .with_history(true);
/// # let _ = solver;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
#[doc = r#"

```
use scioxide_core::{Budget, ResidualTolerance};
use scioxide_linalg::*;
let a = Matrix::<f64>::identity(2)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let tolerance = ResidualTolerance::new(1e-10, 0.0)?;
let solver = BicgstabSolver::new(tolerance, Budget::try_from(20_usize)?);
let result = solver.solve(&a.view().try_finite()?, &rhs)?;
assert!((result.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```

Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.

# References

- H. A. van der Vorst, "Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG
  for the solution of nonsymmetric linear systems", *SIAM J. Sci. Stat. Comput.*
  13(2), 1992."#]
pub struct BicgstabSolver<S: NumericScalar> {
    execution: crate::LinalgExecutionOptions,
    tolerance: ResidualTolerance<S::Real>,
    max_iterations: Budget,
    record_history: bool,
    breakdown_restarts: usize,
}

/// BiCGSTAB recurrence vectors, provider scratch and residual history.
#[derive(Debug)]
pub(super) struct BicgstabStorage<S: NumericScalar> {
    /// The workspace's staged iterate, lent for the duration of a solve and
    /// empty otherwise.
    solution: Vector<S>,
    verified_solution: Vector<S>,
    best: super::support::BestIterate<S>,
    residual: Vector<S>,
    shadow_residual: Vector<S>,
    operator_direction: Vector<S>,
    direction: Vector<S>,
    preconditioned_direction: Vector<S>,
    intermediate_residual: Vector<S>,
    preconditioned_intermediate: Vector<S>,
    operator_intermediate: Vector<S>,
    operator_workspace: Vec<S>,
    preconditioner_workspace: Vec<S>,
    history: Option<Vec<S::Real>>,
}

impl<S: NumericScalar> BicgstabStorage<S> {
    /// Exact payload of [`Self::new`]: ten vectors, provider scratch and history.
    pub(super) fn bytes(shape: &Shape, history: usize) -> Result<usize, SciOxideError> {
        super::workspace::payload::<S>(
            super::workspace::total([
                super::workspace::times(10, shape.cols)?,
                shape.operator_scratch,
                shape.preconditioner_scratch,
            ])?,
            history,
        )
    }

    pub(super) fn new(shape: &Shape, history: usize) -> Result<Self, SciOxideError> {
        let dimension = shape.cols;
        Ok(Self {
            solution: Vector::zeros(0)?,
            verified_solution: Vector::zeros(dimension)?,
            best: super::support::BestIterate::new(dimension)?,
            residual: Vector::zeros(dimension)?,
            shadow_residual: Vector::zeros(dimension)?,
            operator_direction: Vector::zeros(dimension)?,
            direction: Vector::zeros(dimension)?,
            preconditioned_direction: Vector::zeros(dimension)?,
            intermediate_residual: Vector::zeros(dimension)?,
            preconditioned_intermediate: Vector::zeros(dimension)?,
            operator_intermediate: Vector::zeros(dimension)?,
            operator_workspace: try_filled(
                shape.operator_scratch,
                S::ZERO,
                scioxide_core::quantity!("bicgstab_operator_workspace"),
            )?,
            preconditioner_workspace: try_filled(
                shape.preconditioner_scratch,
                S::ZERO,
                scioxide_core::quantity!("bicgstab_preconditioner_workspace"),
            )?,
            history: (history > 0)
                .then(|| {
                    try_with_capacity(
                        history,
                        scioxide_core::quantity!("bicgstab_residual_history"),
                    )
                })
                .transpose()?,
        })
    }

    pub(super) fn history(&self) -> Option<&[S::Real]> {
        self.history.as_deref()
    }
}

impl<S: NumericScalar> fmt::Debug for BicgstabSolver<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BicgstabSolver")
            .field("tolerance", &self.tolerance)
            .field("max_iterations", &self.max_iterations)
            .field("execution", &self.execution)
            .field("record_history", &self.record_history)
            .field("breakdown_restarts", &self.breakdown_restarts)
            .finish()
    }
}

fn reliable_stabilization_product<S: NumericScalar>(
    value: scioxide_core::ScaledValue<S>,
    left_norm: S::Real,
    right_norm: S::Real,
    dimension: usize,
) -> Result<bool, SciOxideError> {
    let value = if left_norm == S::Real::ZERO || right_norm == S::Real::ZERO {
        S::ZERO
    } else {
        value
            .div(scioxide_core::ScaledValue::from_value(S::from_real(
                left_norm,
            ))?)?
            .div(scioxide_core::ScaledValue::from_value(S::from_real(
                right_norm,
            ))?)?
            .round()?
    };
    Ok(!inner_product_is_small(
        value,
        S::Real::ONE,
        S::Real::ONE,
        dimension,
    )?)
}

impl<S: NumericScalar> BicgstabSolver<S> {
    /// Iteration allowance: `max_iterations`, capped by an explicit execution budget.
    fn limit(&self) -> Budget {
        self.execution.cap_iterations(self.max_iterations)
    }

    /// Returns the execution options.
    pub const fn execution_options(&self) -> crate::LinalgExecutionOptions {
        self.execution
    }

    /// Sets the execution options: policy, product accuracy, time limit, byte
    /// limit and iteration budget.
    ///
    /// An iteration budget in `execution` caps `max_iterations`. The accuracy
    /// setting applies to dense operator products; the recurrence safeguards
    /// always use `Standard` accuracy.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Creates a BiCGSTAB solver with residual tolerance `tolerance` and at most
    /// `max_iterations` iterations.
    ///
    /// No history is recorded and breakdown restarts are disabled.
    #[must_use]
    pub fn new(tolerance: ResidualTolerance<S::Real>, max_iterations: Budget) -> Self {
        Self {
            execution: crate::LinalgExecutionOptions::new(),
            tolerance,
            max_iterations,
            record_history: false,
            breakdown_restarts: 0,
        }
    }

    /// Enables or disables recording of the residual norm after each iteration.
    #[must_use]
    pub const fn with_history(mut self, record: bool) -> Self {
        self.record_history = record;
        self
    }

    /// Allows at most `limit` restarts after breakdown; the default is none.
    ///
    /// A restart continues from the last verified iterate and uses its true
    /// residual as the new shadow residual. A second breakdown before any
    /// accepted step ends the solve. Restarts count toward the same iteration
    /// and evaluation limits, keep the history and need no extra workspace.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::{Budget, ResidualTolerance};
    /// use scioxide_linalg::BicgstabSolver;
    /// let solver = BicgstabSolver::<f64>::new(
    ///     ResidualTolerance::new(0., 1e-8)?,
    ///     Budget::try_from(4000)?,
    /// )
    /// .with_breakdown_restarts(Budget::try_from(2)?);
    /// # let _ = solver;
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    #[must_use]
    pub const fn with_breakdown_restarts(mut self, limit: Budget) -> Self {
        self.breakdown_restarts = limit.get();
        self
    }

    /// Returns the storage a solve of `request` needs, including preconditioner scratch.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size or byte count overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the byte limit of the execution options.
    pub fn workspace_requirements(
        &self,
        request: &IterativeWorkspaceRequest,
    ) -> Result<IterativeRequirements<S>, SciOxideError> {
        IterativeRequirements::new(
            Shape {
                kind: StorageKind::Bicgstab,
                rows: request.rows(),
                cols: request.cols(),
                operator_scratch: request.operator_scratch(),
                preconditioner_scratch: request.preconditioner_scratch(),
                columns: 1,
                cycle: 0,
                retained: 0,
            },
            self.record_history.then_some(self.limit()),
            Some(request),
            self.execution.byte_limit(),
        )
    }

    /// Creates a workspace reserved for `request`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if memory cannot be reserved.
    pub fn workspace(
        &self,
        request: &IterativeWorkspaceRequest,
    ) -> Result<IterativeWorkspace<S>, SciOxideError> {
        let mut workspace = IterativeWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Solves `A x = b` from `x = 0` without preconditioning and returns `x`.
    ///
    /// A solve that ends without meeting the tolerance, at the iteration or
    /// evaluation limit or by a breakdown, is an error. Use
    /// [`solve_with_options`](Self::solve_with_options) for a preconditioner, a
    /// start vector or the iterate of an unsuccessful solve.
    ///
    /// # Errors
    ///
    /// - The errors of [`solve_with_options`](Self::solve_with_options).
    /// - `ConvergenceFailure` (`NestedSolver`) if the solve does not converge; its
    ///   diagnostics name the solver and carry the iteration count and the
    ///   residual norm.
    pub fn solve<'b>(
        &self,
        a: &dyn LinearOperator<S>,
        b: impl Into<VectorView<'b, S>>,
    ) -> core::result::Result<crate::Vector<S>, scioxide_core::SciOxideError> {
        self.solve_with_options(a, b, crate::Preconditioning::None, None)?
            .into_converged_solution()
    }
    /// Solves `A x = b` from the initial guess `x0`, or from zero when `x0` is `None`, and returns
    /// the iterate with its report.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::solve_into`].
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn solve_with_options<'b>(
        &self,
        a: &dyn LinearOperator<S>,
        b: impl Into<VectorView<'b, S>>,
        preconditioning: Preconditioning<'_, S>,
        x0: Option<VectorView<'_, S>>,
    ) -> Result<IterativeSolveResult<S>, SciOxideError> {
        let b = b.into();
        IterativeWorkspace::solve_owned(a.cols(), x0, self.algorithm_id(), |x, workspace| {
            self.solve_into(
                a,
                b,
                preconditioning,
                x,
                workspace,
                ExecutionContext::sequential_native(),
            )
        })
    }

    /// Solves `A x = b` in place, starting from the values in `x`.
    ///
    /// Inner products and their ratios are formed with an extended exponent
    /// range, so intermediate quantities such as `‖r‖₂²` may overflow or
    /// underflow the scalar type without failing the solve. Diagnostics are read
    /// from [`IterativeWorkspace::last_attempt`] and [`IterativeWorkspace::last_accepted`].
    ///
    /// # Errors
    ///
    /// The errors listed on
    /// [`IterativeLinearSolver::solve_into`](crate::IterativeLinearSolver::solve_into).
    pub fn solve_into<'b, 'scope, 'capability>(
        &self,
        a: &dyn LinearOperator<S>,
        b: impl Into<VectorView<'b, S>>,
        preconditioning: Preconditioning<'_, S>,
        mut x: VectorViewMut<'_, S>,
        workspace: &mut IterativeWorkspace<S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<IterativeSolveReport<S::Real>, SciOxideError> {
        workspace.reject();
        validate_square(a)?;
        preconditioning.validate_dimension(a.cols())?;
        let preconditioner = preconditioning;
        let requirements = self.workspace_requirements(
            &IterativeWorkspaceRequest::operator(a)?.with_preconditioner(preconditioning)?,
        )?;
        workspace.run(
            context.into().with_options(&self.execution)?,
            &Job {
                algorithm: self.algorithm_id(),
                limit: self.limit(),
                requirements,
            },
            b.into(),
            &mut x,
            |storage, b, x, kernels, solve| {
                self.solve_kernel(preconditioner, a, b, x, storage, kernels, solve)
            },
            IterativeProgress::of,
        )
    }

    fn precondition_into(
        &self,
        preconditioner: Preconditioning<'_, S>,
        v: &Vector<S>,
        output: &mut Vector<S>,
        workspace: &mut [S],
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<bool, SciOxideError> {
        super::support::precondition(
            preconditioner,
            false,
            v,
            output,
            workspace,
            kernels,
            context,
        )
    }
}

crate::traits::impl_iterative_solver!(
    BicgstabSolver,
    _plan,
    scioxide_core::algorithm_id!("linear", "BiCGSTAB", 1)
);

impl<S: NumericScalar> BicgstabSolver<S> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    fn solve_kernel(
        &self,
        preconditioner: Preconditioning<'_, S>,
        a: &dyn LinearOperator<S>,
        b: &Vector<S>,
        x: &mut Vector<S>,
        workspace: &mut BicgstabStorage<S>,
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<IterativeSolveReport<S::Real>, SciOxideError> {
        // Iterate on the staged `x` itself: its buffer is lent to the storage and
        // returned on every path.
        core::mem::swap(x, &mut workspace.solution);
        let report = (|| -> Result<IterativeSolveReport<S::Real>, SciOxideError> {
            let mut diagnostics = crate::KrylovDiagnostics {
                normal_residual: None,
                condition_estimate: None,
                rank_loss_detected: false,
                restarts: 0,
            };
            let mut summary = match self.initialize_recurrence(a, b, workspace, kernels, context)? {
                core::ops::ControlFlow::Continue(mut state) => {
                    let outcome = self.iterate_recurrence(
                        preconditioner,
                        a,
                        b,
                        workspace,
                        &mut state,
                        kernels,
                        context,
                    );
                    diagnostics.restarts = state.restarts;
                    diagnostics.rank_loss_detected = state.denominator_lost;
                    match outcome {
                        Ok(summary) => summary,
                        Err(error) => {
                            let termination = super::support::execution_stop(error)?;
                            super::support::stop_at_verified(
                                super::support::AcceptedIterate {
                                    solution: &mut workspace.solution,
                                    verified: &workspace.verified_solution,
                                    history: &mut workspace.history,
                                    residual: state.verified_residual,
                                    reference: state.problem_scale,
                                },
                                context,
                                termination,
                            )?
                        }
                    }
                }
                core::ops::ControlFlow::Break(summary) => summary,
            };
            if !summary.termination.is_success() {
                super::support::restore_verified_solution(
                    &mut workspace.solution,
                    &workspace.best.solution,
                    &mut workspace.history,
                    workspace.best.residual,
                )?;
                summary.raw_residual_norm = workspace.best.residual;
            }
            Ok(IterativeSolveReport::from_summary(summary)?.with_krylov(diagnostics))
        })();
        core::mem::swap(x, &mut workspace.solution);
        report
    }
}

struct BicgstabRecurrence<S: NumericScalar> {
    problem_scale: S::Real,
    floor: super::safety::ResidualFloor<S::Real>,
    verified_residual: S::Real,
    rho: scioxide_core::ScaledValue<S>,
    alpha: S,
    omega: S,
    first_step: bool,
    half_step: bool,
    accepted_steps: usize,
    last_restart: Option<usize>,
    restarts: usize,
    denominator_lost: bool,
}

impl<S: NumericScalar> BicgstabSolver<S> {
    fn initialize_recurrence(
        &self,
        a: &dyn LinearOperator<S>,
        b: &Vector<S>,
        workspace: &mut BicgstabStorage<S>,
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<
        core::ops::ControlFlow<IterativeSolveSummary<S::Real>, BicgstabRecurrence<S>>,
        SciOxideError,
    > {
        let BicgstabStorage {
            best,
            solution: x,
            verified_solution,
            residual: r,
            shadow_residual: r_hat,
            operator_direction: v,
            direction: p,
            preconditioned_direction: _p_hat,
            intermediate_residual: _s,
            preconditioned_intermediate: _s_hat,
            operator_intermediate: t,
            operator_workspace,
            preconditioner_workspace: _,
            history,
        } = workspace;
        if let Some(samples) = history.as_mut() {
            samples.clear();
        }

        let rhs_norm = finite::norm(b.as_slice())?;
        let problem_scale = super::residual_reference(rhs_norm);
        let Some(initial_norm) = true_residual_into_scoped(
            super::support::KrylovProblem {
                operator: a,
                rhs: b,
            },
            x,
            r,
            t,
            operator_workspace,
            kernels,
            context,
        )?
        else {
            return Err(crate::iterative::evaluation_limit_error(context));
        };
        finite::copy(verified_solution.as_mut_slice(), x.as_slice())?;
        best.reset(x.as_slice(), initial_norm);
        let verified_residual = initial_norm;
        if let Some(samples) = history.as_mut() {
            samples.push(initial_norm);
        }
        if self.tolerance.is_satisfied(initial_norm, problem_scale)? {
            context.commit_accepted(PartialResultTrust::ResidualChecked)?;
            return Ok(core::ops::ControlFlow::Break(
                crate::iterative::solve_summary(
                    context,
                    crate::LinearSolveTermination::Converged,
                    crate::iterative::TrueResidual::against(initial_norm, problem_scale),
                ),
            ));
        }
        context.assess_accepted(PartialResultTrust::Admissible)?;

        // Right-preconditioned BiCGSTAB (van der Vorst) with shadow residual
        // `r_hat = r_0`. Three scalars can break down: `rho = r_hat^H r`,
        // `r_hat^H v`, and the stabilization pair `t^H s`, `t^H t`. Each is
        // checked relative to its operand norms; on breakdown the true
        // residual is computed first, since the iterate may already have
        // converged, and only otherwise is the breakdown reported.
        // The first step initializes `p = r`; subsequent steps retain the direction.
        finite::copy(r_hat.as_mut_slice(), r.as_slice())?;
        v.view_mut().fill(S::ZERO);
        p.view_mut().fill(S::ZERO);
        let rho = scioxide_core::ScaledValue::ONE;
        let alpha = S::ONE;
        let omega = S::ONE;
        let first_step = true;

        Ok(core::ops::ControlFlow::Continue(BicgstabRecurrence {
            floor: super::safety::ResidualFloor::new(rhs_norm),
            problem_scale,
            verified_residual,
            rho,
            alpha,
            omega,
            first_step,
            half_step: false,
            accepted_steps: 0,
            last_restart: None,
            restarts: 0,
            denominator_lost: false,
        }))
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps its operands, storage and control parameters explicit"
    )]
    fn iterate_recurrence(
        &self,
        preconditioner: Preconditioning<'_, S>,
        a: &dyn LinearOperator<S>,
        b: &Vector<S>,
        workspace: &mut BicgstabStorage<S>,
        state: &mut BicgstabRecurrence<S>,
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<IterativeSolveSummary<S::Real>, SciOxideError> {
        let problem = super::support::KrylovProblem {
            operator: a,
            rhs: b,
        };
        for k in 0..super::support::iteration_limit(context) {
            if let Some(reason) = context.stop_reason()? {
                return super::support::stop_at_verified(
                    super::support::AcceptedIterate {
                        solution: &mut workspace.solution,
                        verified: &workspace.verified_solution,
                        history: &mut workspace.history,
                        residual: state.verified_residual,
                        reference: state.problem_scale,
                    },
                    context,
                    crate::LinearSolveTermination::ExecutionStopped(reason),
                );
            }
            if !context.record_iteration()?.is_granted() {
                return super::support::stop_at_verified(
                    super::support::AcceptedIterate {
                        solution: &mut workspace.solution,
                        verified: &workspace.verified_solution,
                        history: &mut workspace.history,
                        residual: state.verified_residual,
                        reference: state.problem_scale,
                    },
                    context,
                    crate::LinearSolveTermination::IterationLimit,
                );
            }

            let step = (|| {
                let rho_new = match self.begin_iteration(
                    preconditioner,
                    problem,
                    workspace,
                    state,
                    context,
                    kernels,
                )? {
                    core::ops::ControlFlow::Continue(rho) => rho,
                    core::ops::ControlFlow::Break(summary) => {
                        return Ok(core::ops::ControlFlow::Break(summary));
                    }
                };
                if let core::ops::ControlFlow::Break(summary) = self.advance_intermediate(
                    preconditioner,
                    problem,
                    workspace,
                    state,
                    context,
                    kernels,
                )? {
                    return Ok(core::ops::ControlFlow::Break(summary));
                }
                self.accept_iteration(
                    preconditioner,
                    problem,
                    workspace,
                    state,
                    (k + 1, rho_new),
                    kernels,
                    context,
                )
            })();
            let progress = match step {
                Ok(progress) => progress,
                Err(StepFault::Breakdown) => {
                    self.resolve_breakdown(problem, workspace, state, kernels, context)?
                }
                Err(StepFault::Error(error)) => return Err(error),
            };
            if let core::ops::ControlFlow::Break(summary) = progress {
                return Ok(summary);
            }
        }
        let summary = super::support::finish_iterations(
            super::support::KrylovProblem {
                operator: a,
                rhs: b,
            },
            super::support::AcceptedIterate {
                solution: &mut workspace.solution,
                verified: &workspace.verified_solution,
                history: &mut workspace.history,
                residual: state.verified_residual,
                reference: state.problem_scale,
            },
            super::support::ResidualBuffers {
                residual: &mut workspace.residual,
                image: &mut workspace.operator_intermediate,
                provider: &mut workspace.operator_workspace,
            },
            self.tolerance,
            kernels,
            context,
        )?;
        workspace.best.observe(
            workspace.solution.as_slice(),
            summary.raw_residual_norm,
            context.counts().own().iterations(),
        );
        Ok(summary)
    }
    fn begin_iteration(
        &self,
        preconditioner: Preconditioning<'_, S>,
        problem: super::support::KrylovProblem<'_, S>,
        workspace: &mut BicgstabStorage<S>,
        state: &mut BicgstabRecurrence<S>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
        kernels: crate::KernelContext<'_>,
    ) -> Step<core::ops::ControlFlow<IterativeSolveSummary<S::Real>, scioxide_core::ScaledValue<S>>>
    {
        let BicgstabStorage {
            best: _,
            solution: x,
            verified_solution,
            residual: r,
            shadow_residual: r_hat,
            operator_direction: v,
            direction: p,
            preconditioned_direction: p_hat,
            intermediate_residual: _s,
            preconditioned_intermediate: _s_hat,
            operator_intermediate: _t,
            operator_workspace,
            preconditioner_workspace,
            history,
        } = workspace;

        let a = problem.operator;
        let products = crate::kernel::recurrence::products(
            r.as_slice(),
            r_hat.as_slice(),
            kernels.dispatch(),
        )?;
        let r_norm = products.left.norm(r.as_slice())?;
        let r_hat_norm = products.right.norm(r_hat.as_slice())?;
        let rho_new = products.scaled(r.as_slice(), r_hat.as_slice())?;
        if shadow_product_is_small(rho_new, r_hat_norm, r_norm)? {
            return Err(StepFault::Breakdown);
        }

        if state.first_step {
            finite::copy(p.as_mut_slice(), r.as_slice())?;
        } else {
            let rho_ratio = rho_new.div(state.rho)?.round()?;
            let alpha_ratio = checked_divide(state.alpha, state.omega)?;
            let beta = crate::checked::mul(rho_ratio, alpha_ratio)?;
            finite::update(
                p.as_mut_slice(),
                -state.omega,
                v.as_slice(),
                S::ONE,
                kernels.dispatch(),
            )?;
            finite::update(
                p.as_mut_slice(),
                S::ONE,
                r.as_slice(),
                beta,
                kernels.dispatch(),
            )?;
        }

        if !matches!(preconditioner, Preconditioning::None)
            && !self.precondition_into(
                preconditioner,
                p,
                p_hat,
                preconditioner_workspace,
                kernels,
                context,
            )?
        {
            return super::support::stop_at_verified(
                super::support::AcceptedIterate {
                    solution: x,
                    verified: verified_solution,
                    history,
                    residual: state.verified_residual,
                    reference: state.problem_scale,
                },
                context,
                crate::LinearSolveTermination::EvaluationLimit,
            )
            .map(core::ops::ControlFlow::Break)
            .map_err(StepFault::from);
        }
        let effective_direction = if matches!(preconditioner, Preconditioning::None) {
            &*p
        } else {
            &*p_hat
        };
        if super::safety::apply_operator_scoped(
            a,
            effective_direction.as_slice(),
            finite::maximum(effective_direction.as_slice(), kernels.dispatch())?,
            v.as_mut_slice(),
            operator_workspace,
            context,
        )?
        .is_exhausted()
        {
            return super::support::stop_at_verified(
                super::support::AcceptedIterate {
                    solution: x,
                    verified: verified_solution,
                    history,
                    residual: state.verified_residual,
                    reference: state.problem_scale,
                },
                context,
                crate::LinearSolveTermination::EvaluationLimit,
            )
            .map(core::ops::ControlFlow::Break)
            .map_err(StepFault::from);
        }
        let products = crate::kernel::recurrence::products(
            v.as_slice(),
            r_hat.as_slice(),
            kernels.dispatch(),
        )?;
        let v_norm = products.left.norm(v.as_slice())?;
        let r_hat_v = products.scaled(v.as_slice(), r_hat.as_slice())?;
        if shadow_product_is_small(r_hat_v, r_hat_norm, v_norm)? {
            return Err(StepFault::Breakdown);
        }
        state.alpha = rho_new.div(r_hat_v)?.round()?;

        Ok(core::ops::ControlFlow::Continue(rho_new))
    }
    fn advance_intermediate(
        &self,
        preconditioner: Preconditioning<'_, S>,
        problem: super::support::KrylovProblem<'_, S>,
        workspace: &mut BicgstabStorage<S>,
        state: &mut BicgstabRecurrence<S>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
        kernels: crate::KernelContext<'_>,
    ) -> Step<core::ops::ControlFlow<IterativeSolveSummary<S::Real>>> {
        let BicgstabStorage {
            best,
            solution: x,
            verified_solution,
            residual: r,
            shadow_residual: _r_hat,
            operator_direction: v,
            direction: p,
            preconditioned_direction: p_hat,
            intermediate_residual: s,
            preconditioned_intermediate: s_hat,
            operator_intermediate: t,
            operator_workspace,
            preconditioner_workspace,
            history,
        } = workspace;

        let a = problem.operator;
        let b = problem.rhs;
        let n = x.len();
        // Half step: `s = r - alpha v` is the residual of the BiCG iterate
        // `x + alpha p_hat`. If it already meets the tolerance (confirmed
        // with the true residual) the stabilization step is skipped,
        // which also avoids a breakdown in `t^H t` when s is tiny.
        finite::copy(s.as_mut_slice(), r.as_slice())?;
        let s_moments = finite::update(
            s.as_mut_slice(),
            -state.alpha,
            v.as_slice(),
            S::ONE,
            kernels.dispatch(),
        )?
        .moments;
        let effective_direction = if matches!(preconditioner, Preconditioning::None) {
            &*p
        } else {
            &*p_hat
        };
        finite::update(
            x.as_mut_slice(),
            state.alpha,
            effective_direction.as_slice(),
            S::ONE,
            kernels.dispatch(),
        )?;
        state.half_step = true;
        let mut s_norm = s_moments.norm(s.as_slice())?;

        if self.tolerance.is_satisfied(s_norm, state.problem_scale)? {
            let Some(true_norm) = true_residual_into_scoped(
                super::support::KrylovProblem {
                    operator: a,
                    rhs: b,
                },
                x,
                s,
                t,
                operator_workspace,
                kernels,
                context,
            )?
            else {
                return super::support::stop_at_verified(
                    super::support::AcceptedIterate {
                        solution: x,
                        verified: verified_solution,
                        history,
                        residual: state.verified_residual,
                        reference: state.problem_scale,
                    },
                    context,
                    crate::LinearSolveTermination::EvaluationLimit,
                )
                .map(core::ops::ControlFlow::Break)
                .map_err(StepFault::from);
            };
            let best_before = best.residual;
            best.observe(x.as_slice(), true_norm, context.counts().own().iterations());
            s_norm = true_norm;
            let changed = x.as_slice() != verified_solution.as_slice();
            if changed {
                state.accepted_steps += 1;
            }
            finite::copy(verified_solution.as_mut_slice(), x.as_slice())?;
            state.verified_residual = true_norm;
            let termination = if self.tolerance.is_satisfied(s_norm, state.problem_scale)? {
                context.commit_accepted(PartialResultTrust::ResidualChecked)?;
                Some(crate::LinearSolveTermination::Converged)
            } else if !changed {
                Some(crate::LinearSolveTermination::Stagnation)
            } else if super::safety::floor_test_due(s_norm, best_before) {
                // The preconditioned intermediate and its image are recomputed below.
                match state.floor.reached(
                    a,
                    best.solution.as_slice(),
                    best.residual,
                    super::safety::FloorProbe {
                        input: s_hat.as_mut_slice(),
                        image: t.as_mut_slice(),
                        workspace: operator_workspace,
                    },
                    kernels,
                    context,
                )? {
                    None => Some(crate::LinearSolveTermination::EvaluationLimit),
                    Some(true) => Some(crate::LinearSolveTermination::Stagnation),
                    Some(false) => None,
                }
            } else {
                None
            };
            if let Some(termination) = termination {
                if let Some(samples) = history.as_mut() {
                    samples.push(s_norm);
                }
                return Ok(core::ops::ControlFlow::Break(
                    crate::iterative::solve_summary(
                        context,
                        termination,
                        crate::iterative::TrueResidual::against(s_norm, state.problem_scale),
                    ),
                ));
            }
        }

        if !matches!(preconditioner, Preconditioning::None)
            && !self.precondition_into(
                preconditioner,
                s,
                s_hat,
                preconditioner_workspace,
                kernels,
                context,
            )?
        {
            return super::support::stop_at_verified(
                super::support::AcceptedIterate {
                    solution: x,
                    verified: verified_solution,
                    history,
                    residual: state.verified_residual,
                    reference: state.problem_scale,
                },
                context,
                crate::LinearSolveTermination::EvaluationLimit,
            )
            .map(core::ops::ControlFlow::Break)
            .map_err(StepFault::from);
        }
        let effective_intermediate = if matches!(preconditioner, Preconditioning::None) {
            &*s
        } else {
            &*s_hat
        };
        if super::safety::apply_operator_scoped(
            a,
            effective_intermediate.as_slice(),
            finite::maximum(effective_intermediate.as_slice(), kernels.dispatch())?,
            t.as_mut_slice(),
            operator_workspace,
            context,
        )?
        .is_exhausted()
        {
            return super::support::stop_at_verified(
                super::support::AcceptedIterate {
                    solution: x,
                    verified: verified_solution,
                    history,
                    residual: state.verified_residual,
                    reference: state.problem_scale,
                },
                context,
                crate::LinearSolveTermination::EvaluationLimit,
            )
            .map(core::ops::ControlFlow::Break)
            .map_err(StepFault::from);
        }
        let products =
            crate::kernel::recurrence::products(s.as_slice(), t.as_slice(), kernels.dispatch())?;
        let t_norm = products.right.norm(t.as_slice())?;
        let t_dot_s = products.scaled(s.as_slice(), t.as_slice())?;
        let t_dot_t = products.right.squared(t.as_slice())?;
        if !reliable_stabilization_product(t_dot_t, t_norm, t_norm, n)?
            || !reliable_stabilization_product(t_dot_s, t_norm, s_norm, n)?
        {
            return Err(StepFault::Breakdown);
        }
        state.omega = t_dot_s.div(t_dot_t)?.round()?;

        Ok(core::ops::ControlFlow::Continue(()))
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "recurrence state, providers and resolved execution are separate borrows"
    )]
    fn accept_iteration(
        &self,
        preconditioner: Preconditioning<'_, S>,
        problem: super::support::KrylovProblem<'_, S>,
        workspace: &mut BicgstabStorage<S>,
        state: &mut BicgstabRecurrence<S>,
        step: (usize, scioxide_core::ScaledValue<S>),
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Step<core::ops::ControlFlow<IterativeSolveSummary<S::Real>>> {
        let BicgstabStorage {
            best,
            solution: x,
            verified_solution,
            residual: r,
            shadow_residual: _,
            operator_direction: _v,
            direction: _p,
            preconditioned_direction: _p_hat,
            intermediate_residual: s,
            preconditioned_intermediate: s_hat,
            operator_intermediate: t,
            operator_workspace,
            preconditioner_workspace: _,
            history,
        } = workspace;

        let a = problem.operator;
        let b = problem.rhs;
        let (iteration, rho_new) = step;
        let effective_intermediate = if matches!(preconditioner, Preconditioning::None) {
            &*s
        } else {
            &*s_hat
        };
        finite::update(
            x.as_mut_slice(),
            state.omega,
            effective_intermediate.as_slice(),
            S::ONE,
            kernels.dispatch(),
        )?;
        finite::copy(r.as_mut_slice(), s.as_slice())?;
        let residual_moments = finite::update(
            r.as_mut_slice(),
            -state.omega,
            t.as_slice(),
            S::ONE,
            kernels.dispatch(),
        )?
        .moments;

        state.half_step = false;
        let estimate = residual_moments.norm(r.as_slice())?;
        let estimate_converged = self.tolerance.is_satisfied(estimate, state.problem_scale)?;
        let refreshed = estimate_converged || should_recompute_true_residual(iteration);
        let best_before = best.residual;
        let residual_norm = if refreshed {
            match true_residual_into_scoped(
                super::support::KrylovProblem {
                    operator: a,
                    rhs: b,
                },
                x,
                r,
                t,
                operator_workspace,
                kernels,
                context,
            )? {
                Some(residual) => residual,
                None => {
                    return super::support::stop_at_verified(
                        super::support::AcceptedIterate {
                            solution: x,
                            verified: verified_solution,
                            history,
                            residual: state.verified_residual,
                            reference: state.problem_scale,
                        },
                        context,
                        crate::LinearSolveTermination::EvaluationLimit,
                    )
                    .map(core::ops::ControlFlow::Break)
                    .map_err(StepFault::from);
                }
            }
        } else {
            estimate
        };
        let mut changed = true;
        if refreshed {
            best.observe(x.as_slice(), residual_norm, iteration);
            changed = x.as_slice() != verified_solution.as_slice();
            if changed {
                state.accepted_steps += 1;
            }
            finite::copy(verified_solution.as_mut_slice(), x.as_slice())?;
            state.verified_residual = residual_norm;
        }
        if let Some(samples) = history.as_mut() {
            samples.push(residual_norm);
        }
        if self
            .tolerance
            .is_satisfied(residual_norm, state.problem_scale)?
        {
            context.commit_accepted(PartialResultTrust::ResidualChecked)?;
            return Ok(core::ops::ControlFlow::Break(
                crate::iterative::solve_summary(
                    context,
                    crate::LinearSolveTermination::Converged,
                    crate::iterative::TrueResidual::against(residual_norm, state.problem_scale),
                ),
            ));
        }
        if refreshed {
            context.commit_accepted(PartialResultTrust::Admissible)?;
            // The intermediate vectors are recomputed in the next iteration.
            let stagnation = if !changed {
                Some(crate::LinearSolveTermination::Stagnation)
            } else if iteration < super::support::iteration_limit(context)
                && super::safety::floor_test_due(residual_norm, best_before)
            {
                match state.floor.reached(
                    a,
                    best.solution.as_slice(),
                    best.residual,
                    super::safety::FloorProbe {
                        input: s_hat.as_mut_slice(),
                        image: t.as_mut_slice(),
                        workspace: operator_workspace,
                    },
                    kernels,
                    context,
                )? {
                    None => Some(crate::LinearSolveTermination::EvaluationLimit),
                    Some(true) => Some(crate::LinearSolveTermination::Stagnation),
                    Some(false) => None,
                }
            } else {
                None
            };
            if let Some(termination) = stagnation {
                return Ok(core::ops::ControlFlow::Break(
                    crate::iterative::solve_summary(
                        context,
                        termination,
                        crate::iterative::TrueResidual::against(residual_norm, state.problem_scale),
                    ),
                ));
            }
        }
        // Keep the fixed shadow and the old direction/image pair across replacement.
        state.rho = rho_new;
        state.first_step = false;
        Ok(core::ops::ControlFlow::Continue(()))
    }
}

impl<S: NumericScalar> BicgstabSolver<S> {
    fn resolve_breakdown(
        &self,
        problem: super::support::KrylovProblem<'_, S>,
        w: &mut BicgstabStorage<S>,
        state: &mut BicgstabRecurrence<S>,
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<core::ops::ControlFlow<IterativeSolveSummary<S::Real>>, SciOxideError> {
        state.denominator_lost = true;
        let Some(true_norm) = true_residual_into_scoped(
            problem,
            &w.solution,
            &mut w.residual,
            &mut w.operator_intermediate,
            &mut w.operator_workspace,
            kernels,
            context,
        )?
        else {
            return super::support::stop_at_verified(
                super::support::AcceptedIterate {
                    solution: &mut w.solution,
                    verified: &w.verified_solution,
                    history: &mut w.history,
                    residual: state.verified_residual,
                    reference: state.problem_scale,
                },
                context,
                crate::LinearSolveTermination::EvaluationLimit,
            )
            .map(core::ops::ControlFlow::Break);
        };
        // Verification may accept a completed or half-step candidate. Merely
        // verifying the same iterate again must not permit another retry.
        if w.solution.as_slice() != w.verified_solution.as_slice() {
            state.accepted_steps += 1;
        }
        w.verified_solution
            .as_mut_slice()
            .copy_from_slice(w.solution.as_slice());
        w.best.observe(
            w.solution.as_slice(),
            true_norm,
            context.counts().own().iterations(),
        );
        state.verified_residual = true_norm;
        let satisfied = self
            .tolerance
            .is_satisfied(true_norm, state.problem_scale)?;
        context.commit_accepted(super::checkpoint_trust(satisfied))?;
        if let Some(samples) = w.history.as_mut() {
            if state.half_step {
                samples.push(true_norm);
            } else if let Some(last) = samples.last_mut() {
                *last = true_norm;
            }
        }
        let termination = if satisfied {
            crate::LinearSolveTermination::Converged
        } else if state.restarts >= self.breakdown_restarts
            || state.last_restart == Some(state.accepted_steps)
        {
            crate::LinearSolveTermination::Breakdown(
                crate::LinearSolveBreakdown::RecurrenceDenominator,
            )
        } else {
            w.shadow_residual
                .as_mut_slice()
                .copy_from_slice(w.residual.as_slice());
            w.direction.view_mut().fill(S::ZERO);
            w.operator_direction.view_mut().fill(S::ZERO);
            state.rho = scioxide_core::ScaledValue::ONE;
            state.alpha = S::ONE;
            state.omega = S::ONE;
            state.first_step = true;
            state.half_step = false;
            state.last_restart = Some(state.accepted_steps);
            state.restarts += 1;
            return Ok(core::ops::ControlFlow::Continue(()));
        };
        Ok(core::ops::ControlFlow::Break(
            crate::iterative::solve_summary(
                context,
                termination,
                crate::iterative::TrueResidual::against(true_norm, state.problem_scale),
            ),
        ))
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::dense::Matrix;
    use crate::iterative::safety::tests::true_residual;
    use scioxide_core::Complex;
    use scioxide_sparse::{CsrMatrix, JacobiPreconditioner};

    use super::super::test_support::{FailingPreconditioner, max_iter_behavior};

    struct DriftingOperator {
        calls: Cell<usize>,
    }

    impl LinearOperator<f64> for DriftingOperator {
        fn rows(&self) -> usize {
            1
        }

        fn cols(&self) -> usize {
            1
        }

        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(1)
        }

        fn apply_into(
            &self,
            input: &[f64],
            output: &mut [f64],
            workspace: &mut [f64],
        ) -> Result<(), SciOxideError> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            workspace[0] = if call == 1 { input[0] } else { 2.0 * input[0] };
            output.copy_from_slice(workspace);
            Ok(())
        }
    }

    fn solver(tolerance: f64, max_iterations: usize) -> BicgstabSolver<f64> {
        BicgstabSolver::new(
            ResidualTolerance::new(tolerance, 0.0).unwrap(),
            Budget::try_from(max_iterations).unwrap(),
        )
    }

    /// Helper: build a dense matrix from flat data.
    fn mat_op(data: &[f64], rows: usize, cols: usize) -> Matrix<f64> {
        Matrix::from_vec(rows, cols, data.to_vec()).unwrap()
    }

    #[test]
    fn test_solve_nonsymmetric() {
        // A = [[2, 1], [0, 3]], b = [5, 9] => x = [1, 3]
        let a = mat_op(&[2.0, 1.0, 0.0, 3.0], 2, 2);
        let b = Vector::<f64>::from_vec(vec![5.0, 9.0]);
        let solver = solver(1e-10, 200);
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();
        assert!((result.solution().view().get(0).unwrap() - 1.0).abs() < 1e-8);
        assert!((result.solution().view().get(1).unwrap() - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_solve_identity() {
        let a = Matrix::<f64>::identity(3).unwrap();
        let b = Vector::<f64>::from_vec(vec![7.0, 8.0, 9.0]);
        let solver = solver(1e-10, 200);
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();
        for i in 0..3 {
            assert!(
                (result.solution().view().get(i).unwrap() - b.view().get(i).unwrap()).abs() < 1e-8
            );
        }
    }

    #[test]
    fn complex_left_linear_inner_product_reproducer_converges() {
        let imaginary = Complex::new(0.0_f64, 1.0);
        let a = Matrix::from_vec(1, 1, vec![imaginary]).unwrap();
        let b = Vector::from_vec(vec![Complex::new(1.0, 0.0)]);
        let solver = BicgstabSolver::<Complex<f64>>::new(
            ResidualTolerance::new(1.0e-12, 0.0).unwrap(),
            Budget::try_from(4).unwrap(),
        );

        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();
        assert!(result.is_success());
        assert!(result.report().diagnostic_residual_norm().unwrap() <= 1.0e-12);
        assert!((result.solution().as_slice()[0] + imaginary).norm() <= 1.0e-12);
    }

    #[test]
    fn deterministic_complex_residual_fixtures_converge() {
        for hermitian in [true, false] {
            let (matrix, right_hand_side) =
                crate::iterative::test_support::deterministic_complex_system(hermitian);
            let solver = BicgstabSolver::<Complex<f64>>::new(
                ResidualTolerance::new(1.0e-11, 0.0).unwrap(),
                Budget::try_from(64).unwrap(),
            );

            let result = solver
                .solve_with_options(
                    &matrix.view().try_finite().unwrap(),
                    &right_hand_side,
                    Preconditioning::None,
                    None,
                )
                .unwrap();
            let (_, residual) = true_residual(
                &matrix.view().try_finite().unwrap(),
                &right_hand_side,
                result.solution(),
            )
            .unwrap();
            assert!(result.is_success());
            assert!(
                residual <= 1.0e-11,
                "Hermitian={hermitian}, residual={residual}"
            );
        }
    }

    #[test]
    fn test_convergence() {
        let a = mat_op(&[4.0, 1.0, 1.0, 3.0], 2, 2);
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let solver = solver(1e-10, 200);
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();
        assert!(result.is_success());
        assert!(result.report().diagnostic_residual_norm().unwrap() < 1e-10);
        let (_, true_norm) =
            true_residual(&a.view().try_finite().unwrap(), &b, result.solution()).unwrap();
        assert_eq!(
            result.report().diagnostic_residual_norm().unwrap(),
            true_norm
        );
    }

    #[test]
    fn test_with_preconditioner() {
        let a_data = [4.0, 1.0, 1.0, 3.0];
        let a = mat_op(&a_data, 2, 2);
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let a_mat = mat_op(&a_data, 2, 2);
        let sparse = CsrMatrix::from_raw(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            a_mat.as_slice().to_vec(),
        )
        .unwrap();
        let pc = JacobiPreconditioner::new(&sparse).unwrap();
        let solver = solver(1e-10, 200);
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::Inverse(&pc),
                None,
            )
            .unwrap();
        assert!(result.is_success());
        assert!(result.report().diagnostic_residual_norm().unwrap() < 1e-10);
    }

    #[test]
    fn ill_conditioned_nonsingular_system_does_not_spuriously_break_down() {
        // The matrix is badly conditioned but nonsingular, so it remains a
        // valid domain-level solve rather than a structural breakdown.
        let a = mat_op(&[1e-12, 1.0, 0.0, 1e-12], 2, 2);
        let b = Vector::<f64>::from_vec(vec![1.0, 1.0]);
        let solver = solver(1e-6, 500);
        let result = solver.solve_with_options(
            &a.view().try_finite().unwrap(),
            &b,
            Preconditioning::None,
            None,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_history_recording() {
        let a = mat_op(&[4.0, 1.0, 1.0, 3.0], 2, 2);
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let solver = solver(1e-10, 200).with_history(true);
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();
        assert!(result.history().is_some());
        let hist = result.history().unwrap();
        assert_eq!(hist.len(), result.report().iterations() + 1);
        assert_eq!(
            hist.last().copied(),
            Some(result.report().diagnostic_residual_norm().unwrap())
        );
    }

    #[test]
    fn workspace_reuses_storage_and_borrows_history() {
        let a = mat_op(&[4.0, 1.0, 0.0, 3.0], 2, 2);
        let b = Vector::<f64>::from_vec(vec![5.0, 6.0]);
        let solver = solver(1e-10, 20).with_history(true);
        let mut workspace = IterativeWorkspace::new();
        let mut x = Vector::from_vec(vec![0.0, 0.0]);

        {
            let report = solver
                .solve_into(
                    &a.view().try_finite().unwrap(),
                    &b,
                    Preconditioning::None,
                    x.view_mut(),
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.is_success());
            assert_eq!(
                workspace.last_accepted().unwrap().history().unwrap().len(),
                report.iterations() + 1
            );
            assert_eq!(
                workspace.last_accepted().unwrap().history().unwrap().last(),
                Some(&report.diagnostic_residual_norm().unwrap())
            );
        }

        x.view_mut().fill(0.0);
        let report = solver
            .solve_into(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                x.view_mut(),
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert!((x.view().get(0).unwrap() - 0.75).abs() < 1e-8);
        assert!((x.view().get(1).unwrap() - 2.0).abs() < 1e-8);
    }

    #[test]
    fn workspace_structural_error_leaves_caller_solution_unchanged() {
        let a = Matrix::<f64>::identity(2).unwrap();
        let b = Vector::from_vec(vec![1.0, 2.0]);
        let solver = solver(1e-10, 2);
        let mut workspace = IterativeWorkspace::new();
        let mut x = Vector::from_vec(vec![7.0, 8.0]);

        assert!(matches!(
            (solver.solve_into(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::Inverse(&FailingPreconditioner),
                x.view_mut(),
                &mut workspace,
                scioxide_core::ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
        assert_eq!(x.as_slice(), &[7.0, 8.0]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = mat_op(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let solver = solver(1e-10, 100);
        assert!(matches!(
            (solver.solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_max_iter_behavior() {
        let _ = max_iter_behavior(&solver(1e-15, 1));
    }

    #[test]
    fn singular_system_reports_breakdown() {
        let a = mat_op(&[0., 0., 0., 0.], 2, 2);
        let b = Vector::from_vec(vec![1., 0.]);
        let result = solver(0., 4)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();
        assert_eq!(
            result.report().termination(),
            crate::LinearSolveTermination::Breakdown(
                crate::LinearSolveBreakdown::RecurrenceDenominator
            )
        );
        assert_eq!(result.report().diagnostic_residual_norm(), Some(1.));
    }

    #[test]
    fn scale_aware_recurrence_accepts_small_system() {
        let a = mat_op(&[1.0e-100], 1, 1);
        let b = Vector::from_vec(vec![1.0e-100]);
        let result = solver(0.0, 2)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();

        assert!(result.is_success());
        assert!((result.solution().view().get(0).unwrap() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn zero_rhs_and_exact_guess_converge_without_preconditioning() {
        let a = Matrix::<f64>::identity(2).unwrap();
        let zero = Vector::from_vec(vec![0.0, 0.0]);
        let zero_result = solver(0.0, 2)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &zero,
                Preconditioning::Inverse(&FailingPreconditioner),
                None,
            )
            .unwrap();
        assert_eq!(zero_result.report().iterations(), 0);

        let exact = Vector::from_vec(vec![2.0, -3.0]);
        let exact_result = solver(0.0, 2)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &exact,
                Preconditioning::Inverse(&FailingPreconditioner),
                Some(exact.view()),
            )
            .unwrap();
        assert_eq!(exact_result.report().iterations(), 0);
    }

    #[test]
    fn false_recursive_convergence_is_corrected_by_true_residual() {
        let operator = DriftingOperator {
            calls: Cell::new(0),
        };
        let b = Vector::from_vec(vec![1.0]);
        let result = solver(0.0, 2)
            .solve_with_options(&operator, &b, Preconditioning::None, None)
            .unwrap();

        assert!(result.is_success());
        assert!((result.solution().view().get(0).unwrap() - 0.5).abs() < 1.0e-12);
        assert_eq!(result.report().diagnostic_residual_norm().unwrap(), 0.0);
    }

    #[test]
    fn preconditioner_error_context_is_preserved() {
        let a = Matrix::<f64>::identity(2).unwrap();
        let b = Vector::from_vec(vec![1.0, 2.0]);
        let error = solver(0.0, 2)
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::Inverse(&FailingPreconditioner),
                None,
            )
            .unwrap_err();

        assert!(
            matches!(error.view(), scioxide_core::SciOxideErrorView::InvalidInput { argument, reason, detail, .. } if argument == scioxide_core::argument!("injected_preconditioner") && reason == scioxide_core::InvalidInputReason::UnsupportedConfiguration && detail == "injected preconditioner failure")
        );
    }
}
