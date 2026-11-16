//! Discrete PDE derivatives; spatial semantics stay here, time traversal stays in ODE.
//!
//! Linear plans differentiate their own data through [`PdeParameterization`]:
//! [`PdeEllipticAdjointPlan`] for steady problems and [`PdeTransientAdjointPlan`]
//! for method-of-lines problems. [`PdeDiscretizationDerivatives`] differentiates any
//! residual whose derivatives the caller binds, such as nonlinear PDE residuals.
use crate::validation::finite_slice;
use crate::validation::invalid;
use crate::validation::poison;

use alloc::vec::Vec;
use scioxide_core::ScaledNorm;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{
    AdjointOperator, ConvergenceDiagnostics, ConvergenceReason, EvaluationRole, LinearOperator,
    NumericScalar, RealScalar, SciOxideError, SolveContext,
};
use scioxide_ode::{AdjointObjective, AdjointObjectivePart};
use scioxide_spatial::MeshToken;
mod elliptic;
mod parameterization;
mod steady;
mod transfer;
mod transient;
pub use elliptic::{PdeEllipticAdjointPlan, PdeEllipticAdjointWorkspace};
pub use parameterization::{
    DifferentiableDiscretization, PdeConditionTangent, PdeParameterTangent, PdeParameterization,
};
pub(crate) use parameterization::{Sealed, parameterize};
pub use steady::{
    PdeDiscretizationDerivatives, PdeLinearSolve, PdeSteadyAdjointOptions, PdeSteadyAdjointReport,
    PdeSteadyAdjointResult, PdeSteadyAdjointWorkspace,
};
pub use transfer::{PdeTransferDerivatives, PdeTransferWorkspace};
pub use transient::{PdeTransientAdjointPlan, PdeTransientAdjointWorkspace};
/// Admit one provider evaluation.
pub(crate) fn admit(
    context: &mut SolveContext<'_, '_>,
    role: EvaluationRole,
) -> Result<(), SciOxideError> {
    context.execution().check()?;
    match context.record_provider(role)? {
        scioxide_core::Admission::Granted(()) => Ok(()),
        scioxide_core::Admission::Denied(report) => Err(SciOxideError::convergence_failure(
            ConvergenceReason::EvaluationLimit,
            ConvergenceDiagnostics::new()
                .with_counts(context.counts())
                .with_budget(report),
            "derivative work budget exhausted",
        )),
    }
}
pub(crate) fn norm<T: NumericScalar>(values: &[T]) -> Result<T::Real, SciOxideError> {
    let mut sum = ScaledNorm::new();
    for v in values {
        sum.add(v.abs())?;
    }
    sum.finish()
}
pub(crate) fn difference<T: NumericScalar>(
    actual: &[T],
    expected: &[T],
) -> Result<T::Real, SciOxideError> {
    let mut sum = ScaledNorm::new();
    for (&a, &b) in actual.iter().zip(expected) {
        sum.add((a - b).abs())?;
    }
    sum.finish()
}
/// Borrowed transpose application; linear solves remain caller-owned.
struct Transpose<'a, T: NumericScalar>(&'a dyn AdjointOperator<T>);
impl<T: NumericScalar> LinearOperator<T> for Transpose<'_, T> {
    fn rows(&self) -> usize {
        self.0.cols()
    }
    fn cols(&self) -> usize {
        self.0.rows()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.0.apply_adjoint_workspace_len()
    }
    fn apply_into(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.0.apply_adjoint_into(input, output, scratch)
    }
    fn apply_into_scoped(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, T>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.0
            .apply_adjoint_into_scoped(input, output, scratch, context)
    }
}

use steady::PdeSteadyAdjointPlan;
