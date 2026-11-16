//! Matrix-free inexact Newton–Krylov method.
use crate::RootTermination;
use crate::nonlinear_math::{affine, difference, dot, norm};
use alloc::vec::Vec;
use core::cell::RefCell;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    Budget, BudgetKind, EvaluationCounts, EvaluationRole, InvalidInputReason, JacobianOperator,
    LinearOperator, NdView, NonFiniteOrigin, NumericScalar, PartialResultTrust, Preconditioner,
    RealScalar, ResidualOperator, ResidualReport, ResidualTolerance, SciOxideError, SolveBudgets,
    SolveContext, StepTolerance, Tolerance,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    GmresSolver, IterativeSolveReport, IterativeWorkspace, IterativeWorkspaceRequest,
    LinearSolveTermination, Vector,
};

mod input;
mod kernels;
mod options;
mod plan;
mod problem;
mod result;
mod workspace;
pub use input::NewtonKrylovInput;
use kernels::*;
pub use options::{NewtonKrylovForcing, NewtonKrylovOptions};
pub use plan::NewtonKrylovSolver;
pub use problem::{
    NewtonKrylovProblem, NonlinearLinearSolver, NonlinearLinearization, NonlinearScaling,
};
pub use result::{NewtonKrylovObservation, NewtonKrylovResult};
use workspace::DifferenceStorage;
pub use workspace::{NewtonKrylovWorkspace, NewtonKrylovWorkspaceRequirements};

fn tenth<R: RealScalar>() -> R {
    let two = R::TWO;
    let five = two + two + R::ONE;
    R::ONE / (five + five)
}
use crate::invalid;
use crate::residual::finite;
