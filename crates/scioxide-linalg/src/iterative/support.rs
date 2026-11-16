//! Shape checks, recurrence helpers and true-residual evaluation shared by iterative solvers.
use crate::Vector;
use crate::kernel::finite;
use alloc::vec::Vec;
use scioxide_core::{AlgebraicScalar, LinearOperator, NumericScalar, SciOxideError};

/// Best verified iterate, independent of the latest recurrence checkpoint.
#[derive(Debug)]
pub(super) struct BestIterate<S: NumericScalar> {
    pub(super) solution: Vector<S>,
    pub(super) residual: S::Real,
}

impl<S: NumericScalar> BestIterate<S> {
    pub(super) fn new(n: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            solution: Vector::zeros(n)?,
            residual: S::Real::ZERO,
        })
    }

    pub(super) fn reset(&mut self, solution: &[S], residual: S::Real) {
        self.solution.as_mut_slice().copy_from_slice(solution);
        self.residual = residual;
        #[cfg(test)]
        super::test_support::checkpoint(solution, residual, 0);
    }

    pub(super) fn observe(&mut self, solution: &[S], residual: S::Real, _iteration: usize) -> bool {
        #[cfg(test)]
        super::test_support::checkpoint(solution, residual, _iteration);
        if residual <= self.residual {
            self.solution.as_mut_slice().copy_from_slice(solution);
            self.residual = residual;
            true
        } else {
            false
        }
    }
}

/// Iteration allowance of the running solve.
///
/// The solve scope holds the solver's own limit capped by the execution
/// context's iteration budget, so drivers that compare against this value take
/// their final verified checkpoint under either cap.
pub(super) fn iteration_limit(context: &scioxide_core::SolveContext<'_, '_>) -> usize {
    match context
        .budget_report(scioxide_core::BudgetKind::Iteration)
        .limit()
    {
        scioxide_core::BudgetLimit::Finite(limit) => limit,
        scioxide_core::BudgetLimit::Unbounded => usize::MAX,
    }
}

/// Reject a non-square operator before any storage or epoch is touched.
pub(super) fn validate_square<S: NumericScalar>(
    a: &dyn LinearOperator<S>,
) -> Result<(), SciOxideError> {
    crate::checked::square(a.rows(), a.cols())
}
pub(super) fn restore_verified_solution<S: NumericScalar>(
    solution: &mut Vector<S>,
    verified_solution: &Vector<S>,
    history: &mut Option<Vec<S::Real>>,
    verified_residual: S::Real,
) -> Result<(), SciOxideError> {
    finite::copy(solution.as_mut_slice(), verified_solution.as_slice())?;
    if let Some(last) = history.as_mut().and_then(|samples| samples.last_mut()) {
        *last = verified_residual;
    }
    Ok(())
}

pub(super) fn true_residual_into_scoped<S: NumericScalar>(
    problem: KrylovProblem<'_, S>,
    x: &Vector<S>,
    residual: &mut Vector<S>,
    operator_output: &mut Vector<S>,
    operator_workspace: &mut [S],
    kernels: crate::KernelContext<'_>,
    context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
) -> Result<Option<S::Real>, SciOxideError> {
    let applied = super::safety::apply_operator_scoped(
        problem.operator,
        x.as_slice(),
        finite::maximum(x.as_slice(), kernels.dispatch())?,
        operator_output.as_mut_slice(),
        operator_workspace,
        context,
    )?;
    if applied.is_exhausted() {
        return Ok(None);
    }
    finite::copy(residual.as_mut_slice(), problem.rhs.as_slice())?;
    let moments = finite::update(
        residual.as_mut_slice(),
        -S::ONE,
        operator_output.as_slice(),
        S::ONE,
        kernels.dispatch(),
    )?
    .moments;
    moments.norm(residual.as_slice()).map(Some)
}

/// Buffers and diagnostics of the last verified iterate, used by every early-stop path.
pub(super) struct AcceptedIterate<'a, S: NumericScalar> {
    pub(super) solution: &'a mut Vector<S>,
    pub(super) verified: &'a Vector<S>,
    pub(super) history: &'a mut Option<Vec<S::Real>>,
    pub(super) residual: S::Real,
    pub(super) reference: S::Real,
}

pub(super) fn stop_at_verified<S: NumericScalar>(
    accepted: AcceptedIterate<'_, S>,
    context: &scioxide_core::OperatorContext<'_, '_, '_, S>,
    termination: crate::LinearSolveTermination,
) -> Result<crate::result::IterativeSolveSummary<S::Real>, SciOxideError> {
    restore_verified_solution(
        accepted.solution,
        accepted.verified,
        accepted.history,
        accepted.residual,
    )?;
    Ok(super::solve_summary(
        context,
        termination,
        super::TrueResidual::against(accepted.residual, accepted.reference),
    ))
}

#[derive(Clone, Copy)]
pub(super) struct KrylovProblem<'a, S: NumericScalar> {
    pub(super) operator: &'a dyn LinearOperator<S>,
    pub(super) rhs: &'a Vector<S>,
}
pub(super) struct ResidualBuffers<'a, S: NumericScalar> {
    pub(super) residual: &'a mut Vector<S>,
    pub(super) image: &'a mut Vector<S>,
    pub(super) provider: &'a mut [S],
}

pub(super) fn finish_iterations<S: NumericScalar>(
    problem: KrylovProblem<'_, S>,
    accepted: AcceptedIterate<'_, S>,
    buffers: ResidualBuffers<'_, S>,
    tolerance: scioxide_core::ResidualTolerance<S::Real>,
    kernels: crate::KernelContext<'_>,
    context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
) -> Result<crate::result::IterativeSolveSummary<S::Real>, SciOxideError> {
    let final_residual = true_residual_into_scoped(
        problem,
        accepted.solution,
        buffers.residual,
        buffers.image,
        buffers.provider,
        kernels,
        context,
    )?;
    let residual = final_residual.unwrap_or(accepted.residual);
    if final_residual.is_none() {
        restore_verified_solution(
            accepted.solution,
            accepted.verified,
            accepted.history,
            accepted.residual,
        )?;
    }
    if let Some(last) = accepted
        .history
        .as_mut()
        .and_then(|samples| samples.last_mut())
    {
        *last = residual;
    }
    let termination = if final_residual.is_none() {
        crate::LinearSolveTermination::EvaluationLimit
    } else if tolerance.is_satisfied(residual, accepted.reference)? {
        crate::LinearSolveTermination::Converged
    } else {
        crate::LinearSolveTermination::IterationLimit
    };
    if final_residual.is_some() {
        context.commit_accepted(super::checkpoint_trust(termination.is_success()))?;
    }
    Ok(super::solve_summary(
        context,
        termination,
        super::TrueResidual::against(residual, accepted.reference),
    ))
}

/// Apply a real provider or copy the identity without charging a provider call.
pub(super) fn precondition<S: NumericScalar>(
    provider: crate::Preconditioning<'_, S>,
    adjoint: bool,
    input: &Vector<S>,
    output: &mut Vector<S>,
    scratch: &mut [S],
    kernels: crate::KernelContext<'_>,
    context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
) -> Result<bool, SciOxideError> {
    use crate::Preconditioning;
    if matches!(provider, Preconditioning::None) {
        finite::copy(output.as_mut_slice(), input.as_slice())?;
        return Ok(true);
    }
    if !context
        .record_provider(scioxide_core::EvaluationRole::Preconditioner)?
        .is_granted()
    {
        return Ok(false);
    }
    match provider {
        Preconditioning::Inverse(p) if !adjoint => {
            p.precondition_into_scoped(input.as_slice(), output.as_mut_slice(), scratch, context)?
        }
        Preconditioning::InverseAdjoint(p) => {
            if adjoint {
                let n = p.apply_adjoint_workspace_len()?;
                p.apply_adjoint_finite_into_scoped(
                    input.as_slice(),
                    finite::maximum(input.as_slice(), kernels.dispatch())?,
                    output.as_mut_slice(),
                    &mut scratch[..n],
                    context,
                )
                .and_then(scioxide_core::Invocation::require)?;
            } else {
                let n = p.apply_workspace_len()?;
                p.apply_finite_into_scoped(
                    input.as_slice(),
                    finite::maximum(input.as_slice(), kernels.dispatch())?,
                    output.as_mut_slice(),
                    &mut scratch[..n],
                    context,
                )
                .and_then(scioxide_core::Invocation::require)?;
            }
        }
        _ => {
            return Err(crate::checked::unsupported(
                scioxide_core::argument!("preconditioning"),
                "adjoint inverse capability required",
            ));
        }
    }
    crate::checked::provider_output(output.as_slice())?;
    Ok(true)
}

/// Single mapping for stops raised by execution checks or nested providers.
pub(super) fn execution_stop(
    error: SciOxideError,
) -> Result<crate::LinearSolveTermination, SciOxideError> {
    match error.view() {
        scioxide_core::SciOxideErrorView::ExecutionStopped { reason, .. } => {
            Ok(crate::LinearSolveTermination::ExecutionStopped(reason))
        }
        _ => Err(error),
    }
}
