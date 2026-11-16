//! Preconditioned conjugate gradient method for Hermitian positive definite systems.

use super::support::{true_residual_into_scoped, validate_square};
use super::workspace::{Job, Shape, StorageKind};
use crate::Preconditioning;
use crate::kernel::finite;
use alloc::vec::Vec;

use core::fmt;

use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{
    Budget, ExecutionContext, LinearOperator, NumericScalar, ResidualTolerance, SciOxideError,
};

use crate::dense::Vector;
use crate::iterative::evaluation_limit_error;
use crate::iterative::safety::{HermitianForms, should_recompute_true_residual};
use crate::result::{IterativeSolveReport, IterativeSolveResult, IterativeSolveSummary};
use crate::traits::IterativeLinearSolver;
use crate::{
    IterativeRequirements, IterativeWorkspace, IterativeWorkspaceRequest, LinalgExecutionContext,
    VectorView, VectorViewMut,
};

/// Preconditioned conjugate gradient method (CG) for Hermitian positive definite systems.
///
/// `A` must be Hermitian positive definite, and so must the preconditioner
/// `M⁻¹` when one is given. CG minimizes the `A`-norm of the error over the
/// Krylov subspace with short recurrences: `r₀ = b − A x₀`, `z = M⁻¹ r`, and
/// the search direction `p` is updated from `z` in each step. Each iteration
/// costs one product with `A` and one application of `M⁻¹`. The workspace holds
/// six vectors of length `n`.
///
/// The solve converges when `‖b − A x‖₂ ≤ atol + rtol·‖b‖₂`, with `‖b‖₂`
/// replaced by one when `b = 0`. The recursively updated residual is replaced by
/// the true residual `b − A x` every 50 iterations and before convergence is
/// accepted; the search direction is kept. A direction with `pᴴ A p ≤ 0` stops
/// the solve with
/// [`Breakdown(NonPositiveCurvature)`](crate::LinearSolveBreakdown::NonPositiveCurvature).
/// Use [`MinresSolver`](crate::MinresSolver) for Hermitian indefinite systems.
#[derive(Clone)]
#[doc = r#"

# Examples

```
use scioxide_core::{Budget, ResidualTolerance};
use scioxide_linalg::*;
let a = Matrix::<f64>::identity(2)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let tolerance = ResidualTolerance::new(1e-10, 0.0)?;
let solver = CgSolver::new(tolerance, Budget::try_from(20_usize)?);
let result = solver.solve(&a.view().try_finite()?, &rhs)?;
assert!((result.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```

Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.

# References

- M. R. Hestenes and E. Stiefel, "Methods of conjugate gradients for solving linear
  systems", *J. Res. Nat. Bur. Standards* 49(6), 1952.
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003, §9.2."#]
pub struct CgSolver<S: NumericScalar> {
    execution: crate::LinalgExecutionOptions,
    tolerance: ResidualTolerance<S::Real>,
    max_iterations: Budget,
    record_history: bool,
}

/// Conjugate-gradient recurrence vectors, provider scratch and residual history.
#[derive(Debug)]
pub(super) struct CgStorage<S: NumericScalar> {
    best: super::support::BestIterate<S>,
    verified_solution: Vector<S>,
    residual: Vector<S>,
    preconditioned: Vector<S>,
    direction: Vector<S>,
    operator_output: Vector<S>,
    operator_workspace: Vec<S>,
    preconditioner_workspace: Vec<S>,
    history: Option<Vec<S::Real>>,
}

impl<S: NumericScalar> CgStorage<S> {
    /// Exact payload of [`Self::new`]: six vectors, provider scratch and history.
    pub(super) fn bytes(shape: &Shape, history: usize) -> Result<usize, SciOxideError> {
        super::workspace::payload::<S>(
            super::workspace::total([
                super::workspace::times(6, shape.cols)?,
                shape.operator_scratch,
                shape.preconditioner_scratch,
            ])?,
            history,
        )
    }

    pub(super) fn new(shape: &Shape, history: usize) -> Result<Self, SciOxideError> {
        let n = shape.cols;
        Ok(Self {
            best: super::support::BestIterate::new(n)?,
            verified_solution: Vector::zeros(n)?,
            residual: Vector::zeros(n)?,
            preconditioned: Vector::zeros(n)?,
            direction: Vector::zeros(n)?,
            operator_output: Vector::zeros(n)?,
            operator_workspace: try_filled(
                shape.operator_scratch,
                S::ZERO,
                scioxide_core::quantity!("cg_operator_workspace"),
            )?,
            preconditioner_workspace: try_filled(
                shape.preconditioner_scratch,
                S::ZERO,
                scioxide_core::quantity!("cg_preconditioner_workspace"),
            )?,
            history: (history > 0)
                .then(|| {
                    try_with_capacity(history, scioxide_core::quantity!("cg_residual_history"))
                })
                .transpose()?,
        })
    }

    pub(super) fn history(&self) -> Option<&[S::Real]> {
        self.history.as_deref()
    }
}

impl<S: NumericScalar> fmt::Debug for CgSolver<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CgSolver")
            .field("tolerance", &self.tolerance)
            .field("max_iterations", &self.max_iterations)
            .field("execution", &self.execution)
            .field("record_history", &self.record_history)
            .finish()
    }
}

impl<S: NumericScalar> CgSolver<S> {
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

    /// Creates a CG solver with residual tolerance `tolerance` and at most
    /// `max_iterations` iterations.
    ///
    /// The solve converges when `‖b − A x‖₂ ≤ atol + rtol·‖b‖₂`. No residual
    /// history is recorded.
    #[must_use]
    pub fn new(tolerance: ResidualTolerance<S::Real>, max_iterations: Budget) -> Self {
        Self {
            execution: crate::LinalgExecutionOptions::new(),
            tolerance,
            max_iterations,
            record_history: false,
        }
    }

    /// Enables or disables recording of the residual norm after each iteration.
    ///
    /// The history is returned by [`IterativeSolveResult::history`] and
    /// [`IterativeWorkspace::last_accepted`].
    #[must_use]
    pub const fn with_history(mut self, record: bool) -> Self {
        self.record_history = record;
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
                kind: StorageKind::Cg,
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
    /// Inner products are formed with an extended exponent range, so the solve
    /// does not fail when `‖r‖₂²` overflows the scalar type but `x` does not.
    /// Non-positive curvature and a preconditioner that is not positive definite
    /// return `Ok` with a `Breakdown` status and the best verified iterate.
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
                IterativeSolveReport::from_summary(self.solve_kernel(
                    preconditioner,
                    a,
                    b,
                    x,
                    storage,
                    kernels,
                    solve,
                )?)
            },
            super::workspace::IterativeProgress::of,
        )
    }
}

crate::traits::impl_iterative_solver!(
    CgSolver,
    _plan,
    scioxide_core::algorithm_id!("linear", "CG", 1)
);

impl<S: NumericScalar> CgSolver<S> {
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
        workspace: &mut CgStorage<S>,
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<IterativeSolveSummary<S::Real>, SciOxideError> {
        use crate::kernel::recurrence::{products, update};
        use crate::{LinearSolveBreakdown as Breakdown, LinearSolveTermination as Termination};
        let CgStorage {
            best,
            verified_solution,
            residual: r,
            preconditioned: z,
            direction: p,
            operator_output: ap,
            operator_workspace,
            preconditioner_workspace,
            history,
        } = workspace;
        if let Some(samples) = history.as_mut() {
            samples.clear();
        }
        let rhs_norm = finite::norm(b.as_slice())?;
        let scale = super::residual_reference(rhs_norm);
        let Some(mut verified) = true_residual_into_scoped(
            super::support::KrylovProblem {
                operator: a,
                rhs: b,
            },
            x,
            r,
            ap,
            operator_workspace,
            kernels,
            context,
        )?
        else {
            return Err(evaluation_limit_error(context));
        };
        finite::copy(verified_solution.as_mut_slice(), x.as_slice())?;
        best.reset(x.as_slice(), verified);
        let floor = super::safety::ResidualFloor::new(rhs_norm);
        let satisfied = self.tolerance.is_satisfied(verified, scale)?;
        context.assess_accepted(super::checkpoint_trust(satisfied))?;
        if let Some(samples) = history.as_mut() {
            samples.push(verified);
        }
        let outcome = (|| -> Result<crate::LinearSolveTermination, SciOxideError> {
            if satisfied {
                return Ok(Termination::Converged);
            }
            let unpreconditioned = matches!(preconditioner, Preconditioning::None);
            let dispatch = kernels.dispatch();
            let mut residual_moments = products(r.as_slice(), r.as_slice(), dispatch)?.left;
            let mut solution_maximum = finite::maximum(x.as_slice(), kernels.dispatch())?;
            if !unpreconditioned
                && !super::support::precondition(
                    preconditioner,
                    false,
                    r,
                    z,
                    preconditioner_workspace,
                    kernels,
                    context,
                )?
            {
                return Ok(Termination::EvaluationLimit);
            }
            let effective = if unpreconditioned { &*r } else { &*z };
            let mut operator_forms = HermitianForms::new(x.len());
            let mut preconditioner_forms = HermitianForms::new(x.len());
            let initial = products(effective.as_slice(), r.as_slice(), dispatch)?;
            let Some(mut rz) = positive_form(
                &mut preconditioner_forms,
                initial.scaled(effective.as_slice(), r.as_slice())?,
                verified,
                initial.left.norm(effective.as_slice())?,
            )?
            else {
                return Ok(Termination::Breakdown(Breakdown::Preconditioner));
            };
            let mut direction_moments = initial.left;
            finite::copy(p.as_mut_slice(), effective.as_slice())?;
            let limit = super::support::iteration_limit(context);
            for iteration in 1..=limit {
                if let Some(reason) = context.stop_reason()? {
                    return Ok(Termination::ExecutionStopped(reason));
                }
                if !context.record_iteration()?.is_granted() {
                    return Ok(Termination::IterationLimit);
                }
                if super::safety::apply_operator_scoped(
                    a,
                    p.as_slice(),
                    direction_moments.maximum,
                    ap.as_mut_slice(),
                    operator_workspace,
                    context,
                )?
                .is_exhausted()
                {
                    return Ok(Termination::EvaluationLimit);
                }
                let product = products(ap.as_slice(), p.as_slice(), dispatch)?;
                let direction_norm = product.right.norm(p.as_slice())?;
                let image_norm = product.left.norm(ap.as_slice())?;
                let Some(curvature) = positive_form(
                    &mut operator_forms,
                    product.scaled(ap.as_slice(), p.as_slice())?,
                    direction_norm,
                    image_norm,
                )?
                else {
                    return Ok(Termination::Breakdown(Breakdown::NonPositiveCurvature));
                };
                let alpha = rz.div(curvature)?.round()?;
                let (changed, solution_moments) = update(
                    x.as_mut_slice(),
                    alpha,
                    p.as_slice(),
                    S::ONE,
                    (direction_moments.maximum, solution_maximum),
                    dispatch,
                )?;
                solution_maximum = solution_moments.maximum;
                let (_, next_residual) = update(
                    r.as_mut_slice(),
                    -alpha,
                    ap.as_slice(),
                    S::ONE,
                    (product.left.maximum, residual_moments.maximum),
                    dispatch,
                )?;
                residual_moments = next_residual;
                let estimate = residual_moments.norm(r.as_slice())?;
                let candidate = self.tolerance.is_satisfied(estimate, scale)?;
                let at_limit = iteration == limit;
                let checkpoint =
                    candidate || should_recompute_true_residual(iteration) || !changed || at_limit;
                let (residual, best_before) = if checkpoint {
                    let Some(residual) = true_residual_into_scoped(
                        super::support::KrylovProblem {
                            operator: a,
                            rhs: b,
                        },
                        x,
                        r,
                        ap,
                        operator_workspace,
                        kernels,
                        context,
                    )?
                    else {
                        return Ok(Termination::EvaluationLimit);
                    };
                    residual_moments = products(r.as_slice(), r.as_slice(), dispatch)?.left;
                    let best_before = best.residual;
                    best.observe(x.as_slice(), residual, iteration);
                    verified = residual;
                    finite::copy(verified_solution.as_mut_slice(), x.as_slice())?;
                    context.commit_accepted(super::checkpoint_trust(
                        self.tolerance.is_satisfied(residual, scale)?,
                    ))?;
                    (residual, best_before)
                } else {
                    (estimate, best.residual)
                };
                if let Some(samples) = history.as_mut() {
                    samples.push(residual);
                }
                if checkpoint && self.tolerance.is_satisfied(residual, scale)? {
                    return Ok(Termination::Converged);
                }
                if !changed {
                    return Ok(Termination::Stagnation);
                }
                if at_limit {
                    return Ok(Termination::IterationLimit);
                }
                // The preconditioned residual and the operator image are
                // recomputed below, so they hold the floor probe.
                if checkpoint && super::safety::floor_test_due(residual, best_before) {
                    match floor.reached(
                        a,
                        best.solution.as_slice(),
                        best.residual,
                        super::safety::FloorProbe {
                            input: z.as_mut_slice(),
                            image: ap.as_mut_slice(),
                            workspace: operator_workspace,
                        },
                        kernels,
                        context,
                    )? {
                        None => return Ok(Termination::EvaluationLimit),
                        Some(true) => return Ok(Termination::Stagnation),
                        Some(false) => {}
                    }
                }
                if !unpreconditioned
                    && !super::support::precondition(
                        preconditioner,
                        false,
                        r,
                        z,
                        preconditioner_workspace,
                        kernels,
                        context,
                    )?
                {
                    return Ok(Termination::EvaluationLimit);
                }
                let effective = if unpreconditioned { &*r } else { &*z };
                let (next, preconditioned_moments) = if unpreconditioned {
                    (residual_moments.squared(r.as_slice())?, residual_moments)
                } else {
                    let product = products(z.as_slice(), r.as_slice(), dispatch)?;
                    (product.scaled(z.as_slice(), r.as_slice())?, product.left)
                };
                let Some(next) = positive_form(
                    &mut preconditioner_forms,
                    next,
                    residual,
                    preconditioned_moments.norm(effective.as_slice())?,
                )?
                else {
                    return Ok(Termination::Breakdown(Breakdown::Preconditioner));
                };
                // Residual replacement corrects rounding drift without discarding conjugacy.
                let beta = next.div(rz)?.round()?;
                let (_, next_direction) = update(
                    p.as_mut_slice(),
                    S::ONE,
                    effective.as_slice(),
                    beta,
                    (preconditioned_moments.maximum, direction_moments.maximum),
                    dispatch,
                )?;
                direction_moments = next_direction;
                rz = next;
            }
            Ok(Termination::IterationLimit)
        })();
        let termination = match outcome {
            Ok(t) => t,
            Err(error) => super::support::execution_stop(error)?,
        };
        super::support::stop_at_verified(
            super::support::AcceptedIterate {
                solution: x,
                verified: if termination.is_success() {
                    verified_solution
                } else {
                    &best.solution
                },
                history,
                residual: if termination.is_success() {
                    verified
                } else {
                    best.residual
                },
                reference: scale,
            },
            context,
            termination,
        )
    }
}

/// Positive real part of a CG form `pᴴAp` or `rᴴM⁻¹r`; `None` is breakdown.
fn positive_form<S: NumericScalar>(
    forms: &mut HermitianForms<S::Real>,
    value: scioxide_core::ScaledValue<S>,
    input: S::Real,
    image: S::Real,
) -> Result<Option<scioxide_core::ScaledValue<S>>, SciOxideError> {
    Ok(forms
        .positive_part(value, input, image)?
        .map(|real| scioxide_core::ScaledValue::from_components(real, scioxide_core::Scaled::ZERO)))
}
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::dense::Matrix;
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

    fn solver(tolerance: f64, max_iterations: usize) -> CgSolver<f64> {
        CgSolver::new(
            ResidualTolerance::new(tolerance, 0.0).unwrap(),
            Budget::try_from(max_iterations).unwrap(),
        )
    }

    /// Build a 2x2 SPD matrix [[4, 1], [1, 3]].
    fn spd_2x2() -> Matrix<f64> {
        Matrix::from_vec(2, 2, vec![4.0, 1.0, 1.0, 3.0]).unwrap()
    }

    #[test]
    fn test_solve_spd_system() {
        // Solve [[4,1],[1,3]] * x = [1,2]
        // Exact solution: x = [1/11, 7/11] ≈ [0.0909..., 0.6363...]
        let a = spd_2x2();
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let solver = solver(1e-10, 100);
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();

        assert!(result.is_success());

        // Verify residual: ||b - A*x|| < 1e-10
        let ax = a.view().mul_vec(result.solution()).unwrap();
        let residual = b.view().try_sub(&ax).unwrap();
        let res_norm = residual
            .view()
            .norm2_scoped(
                &crate::LinalgExecutionOptions::new(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(res_norm < 1e-10, "residual norm too large: {res_norm}");

        // Verify solution values
        let x0 = result.solution().view().get(0).unwrap();
        let x1 = result.solution().view().get(1).unwrap();
        assert!(
            (x0 - 1.0 / 11.0).abs() < 1e-10,
            "x[0] = {x0}, expected {}",
            1.0 / 11.0
        );
        assert!(
            (x1 - 7.0 / 11.0).abs() < 1e-10,
            "x[1] = {x1}, expected {}",
            7.0 / 11.0
        );
    }

    #[test]
    fn test_solve_identity() {
        // Identity matrix: x = b
        let a = Matrix::<f64>::identity(3).unwrap();
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let solver = solver(1e-10, 100);
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();

        assert!(result.is_success());
        for i in 0..3 {
            let xi = result.solution().view().get(i).unwrap();
            let bi = b.view().get(i).unwrap();
            assert!((xi - bi).abs() < 1e-10, "x[{i}] = {xi}, expected {bi}");
        }
    }

    #[test]
    fn test_convergence_check() {
        let a = spd_2x2();
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let solver = solver(1e-6, 100);
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();

        assert!(result.is_success());
        assert!(result.report().diagnostic_residual_norm().unwrap() < 1e-6);
        assert!(result.report().iterations() > 0);
    }

    #[test]
    fn test_with_jacobi_preconditioner() {
        let a = spd_2x2();
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let sparse =
            CsrMatrix::from_raw(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], a.as_slice().to_vec())
                .unwrap();
        let pc = JacobiPreconditioner::new(&sparse).unwrap();
        let solver = solver(1e-10, 100);
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
    fn test_max_iter_behavior() {
        let result = max_iter_behavior(&solver(1e-15, 1));
        assert_eq!(result.report().iterations(), 1);
    }

    #[test]
    fn test_history_recording() {
        let a = spd_2x2();
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let solver = solver(1e-10, 100).with_history(true);
        let result = solver
            .solve_with_options(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                None,
            )
            .unwrap();

        assert!(result.is_success());
        let hist = result.history().unwrap();
        assert!(!hist.is_empty());
        assert_eq!(hist.len(), result.report().iterations() + 1);
        assert_eq!(
            hist.last().copied(),
            Some(result.report().diagnostic_residual_norm().unwrap())
        );
    }

    fn sequential() -> ExecutionContext<'static> {
        ExecutionContext::sequential()
    }

    #[test]
    fn workspace_reuses_storage_and_records_history() {
        let a = spd_2x2();
        let a = a.view().try_finite().unwrap();
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0]);
        let solver = solver(1e-10, 20).with_history(true);
        let mut workspace = solver
            .workspace(&IterativeWorkspaceRequest::operator(&a).unwrap())
            .unwrap();
        let mut x = Vector::from_vec(vec![0.0, 0.0]);

        let report = solver
            .solve_into(
                &a,
                &b,
                Preconditioning::None,
                x.view_mut(),
                &mut workspace,
                sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        let accepted = workspace.last_accepted().unwrap();
        assert_eq!(accepted.history().unwrap().len(), report.iterations() + 1);
        assert_eq!(
            accepted.history().unwrap().last(),
            Some(&report.diagnostic_residual_norm().unwrap())
        );
        assert_eq!(accepted.solution(), x.as_slice());

        x.view_mut().fill(0.0);
        let report = solver
            .solve_into(
                &a,
                &b,
                Preconditioning::None,
                x.view_mut(),
                &mut workspace,
                sequential(),
            )
            .unwrap();
        assert!(report.is_success());
        assert!((x.view().get(0).unwrap() - 1.0 / 11.0).abs() < 1e-8);
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
                sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
        assert_eq!(x.as_slice(), &[7.0, 8.0]);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            scioxide_core::AttemptOutcome::Failed
        );
        assert!(workspace.last_accepted().is_none());
    }

    #[test]
    fn workspace_dimension_error_precedes_output_mutation() {
        let a = Matrix::<f64>::identity(2).unwrap();
        let b = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let solver = solver(1e-10, 2);
        let mut workspace = IterativeWorkspace::new();
        let mut x = Vector::from_vec(vec![7.0, 8.0]);

        assert!(matches!(
            (solver.solve_into(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                x.view_mut(),
                &mut workspace,
                sequential(),
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
        assert_eq!(x.as_slice(), &[7.0, 8.0]);
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(
            attempt.outcome(),
            scioxide_core::AttemptOutcome::PreflightRejected
        );
        assert!(attempt.epoch().is_none());
    }

    #[test]
    fn workspace_iteration_limit_is_an_ok_outcome_with_partial_solution() {
        let a = spd_2x2();
        let b = Vector::from_vec(vec![1.0, 2.0]);
        let solver = solver(1e-15, 1);
        let mut workspace = IterativeWorkspace::new();
        let mut x = Vector::from_vec(vec![0.0, 0.0]);

        let report = solver
            .solve_into(
                &a.view().try_finite().unwrap(),
                &b,
                Preconditioning::None,
                x.view_mut(),
                &mut workspace,
                sequential(),
            )
            .unwrap();

        assert_eq!(
            report.termination(),
            crate::LinearSolveTermination::IterationLimit
        );
        assert_ne!(x.as_slice(), &[0.0, 0.0]);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            scioxide_core::AttemptOutcome::Stopped
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = spd_2x2();
        let b = Vector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let solver = solver(1e-10, 100);
        let result = solver.solve_with_options(
            &a.view().try_finite().unwrap(),
            &b,
            Preconditioning::None,
            None,
        );

        assert!(matches!(
            (result).as_ref().err().unwrap().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn indefinite_operator_returns_verified_breakdown() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]).unwrap();
        let b = Vector::from_vec(vec![0.0, 1.0]);
        let result = solver(1e-12, 10)
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
                crate::LinearSolveBreakdown::NonPositiveCurvature
            )
        );
        assert_eq!(result.solution().as_slice(), &[0., 0.]);
        assert_eq!(result.report().diagnostic_residual_norm(), Some(1.));
    }

    #[test]
    fn scale_aware_curvature_accepts_small_well_conditioned_system() {
        let a = Matrix::from_vec(1, 1, vec![1.0e-100]).unwrap();
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
        assert!(result.report().diagnostic_residual_norm().unwrap() <= f64::EPSILON * 1.0e-100);
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
    fn convergence_is_confirmed_with_the_true_residual() {
        let operator = DriftingOperator {
            calls: Cell::new(0),
        };
        let b = Vector::from_vec(vec![1.0]);
        let result = solver(0.0, 1)
            .solve_with_options(&operator, &b, Preconditioning::None, None)
            .unwrap();

        assert_eq!(
            result.report().termination(),
            crate::LinearSolveTermination::IterationLimit
        );
        assert_eq!(result.report().diagnostic_residual_norm().unwrap(), 1.0);
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
