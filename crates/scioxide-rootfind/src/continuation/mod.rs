//! Numerical continuation of solution curves and deflation of known roots.
use crate::RootTermination;
use crate::context::{KernelFault, KernelResult, admit, check_stop};
use crate::nonlinear_math::{StepRounding, affine, affine_into, difference, dot, norm};
use crate::{
    NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovResult, NewtonKrylovSolver,
    NewtonKrylovWorkspace, NonlinearLinearSolver,
};
use alloc::vec::Vec;
use scioxide_core::{
    Argument, Budget, EvaluationCounts, EvaluationRole, InvalidInputReason, JacobianOperator,
    LinearOperator, NdView, NonFiniteOrigin, PartialResultTrust, ProblemToken, Quantity,
    RealScalar, ResidualOperator, ResidualTolerance, SciOxideError, SolveBudgets,
    SolveCheckpointMetadata, SolveContext, SolveIdentity, SolveState, StepTolerance, Tolerance,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    GmresSolver, IterativeLinearSolver, IterativeWorkspace, IterativeWorkspaceRequest,
    LinearSolveTermination, Vector,
};
mod adapters;
mod checkpoint;
mod deflation;
mod diagnostics;
mod options;
mod plan;
mod predictor;
mod problem;
mod state;
mod workspace;
use crate::invalid;
use crate::residual::finite;
pub use adapters::{LinearHomotopy, RealifiedProblem};
pub use checkpoint::ContinuationCheckpoint;
pub use deflation::{
    DeflatedProblem, DeflatedRootResult, DeflationOptions, DeflationSet, DeflationWorkspace,
    DeflationWorkspaceRequirements,
};
pub use diagnostics::{ContinuationDiagnostic, ContinuationDiagnosticProvider};
pub use options::{ContinuationObservation, ContinuationOptions, ContinuationPath};
pub use plan::{Continuation, ContinuationReport};
use predictor::*;
pub use problem::ContinuationProblem;
use problem::{ProviderShape, provider_shape};
use scioxide_core::backend::try_copy_slice;
pub use state::ContinuationState;
pub use workspace::{ContinuationWorkspace, ContinuationWorkspaceRequirements};
/// Returns `a + b`, or a size-overflow error for `quantity`.
fn checked_size(a: usize, b: usize, quantity: Quantity) -> Result<usize, SciOxideError> {
    a.checked_add(b)
        .ok_or_else(|| SciOxideError::size_overflow(quantity))
}

#[cfg(test)]
use scioxide_core::ExecutionContext;
#[cfg(test)]
mod guard_tests;
#[cfg(test)]
mod tests;
