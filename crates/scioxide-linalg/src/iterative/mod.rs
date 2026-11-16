//! Krylov and stationary iterative solvers for linear systems.
//!
//! Every solver tests convergence on the true residual `b − A x` of the
//! original system. CG and BiCGSTAB replace their recursive residual every 50
//! steps; the other Krylov solvers recompute the true residual every 50 steps,
//! at convergence candidates and at the end, and GMRES also at each restart.
//! Block and stationary solvers recompute it after every step.
//!
//! A solve stagnates when an update leaves the iterate unchanged, or when the
//! best verified residual lies within the rounding floor of
//! `safety::ResidualFloor` at a verification whose recursive residual claimed
//! convergence or that does not improve on the best residual. Solvers that
//! verify every step test the floor only at the periodic checkpoints, and the
//! GMRES family judges stagnation at the end of a cycle.

mod preconditioning;
pub use preconditioning::Preconditioning;

mod arnoldi;
mod bicgstab;
mod cg;
mod flexible;
mod gmres;
pub use arnoldi::KrylovRecycleSpace;
pub use flexible::{
    BlockGcrotSolver, BlockGmresSolver, BlockKrylovReport, BlockKrylovResult, BlockLgmresSolver,
    GcrotSolver, KrylovRecycleRequirements, KrylovRecycleWorkspace, KrylovRecycleWorkspaceRequest,
    LgmresSolver,
};
mod krylov;
mod safety;
pub use krylov::{LsmrSolver, LsqrSolver, MinresSolver};
mod workspace;
pub use workspace::{
    IterativeAccepted, IterativeProgress, IterativeRequirements, IterativeWorkspace,
    IterativeWorkspaceRequest, KrylovColumns,
};

pub use bicgstab::BicgstabSolver;
pub use cg::CgSolver;
pub use gmres::GmresSolver;

#[derive(Debug, Clone, Copy)]
pub(crate) struct TrueResidual<R> {
    raw: R,
    reference: R,
}

/// Convergence and report reference for a right-hand side of norm `rhs_norm`:
/// `‖b‖₂`, or one for a zero right-hand side so that the relative tolerance
/// still applies. Every family tests convergence against it and scales its
/// reports by it; rounding models use the raw `‖b‖₂`.
pub(crate) fn residual_reference<R: scioxide_core::RealScalar>(rhs_norm: R) -> R {
    if rhs_norm > R::ZERO { rhs_norm } else { R::ONE }
}

impl<R: scioxide_core::RealScalar> TrueResidual<R> {
    pub(crate) fn against(raw: R, problem_scale: R) -> Self {
        Self {
            raw,
            reference: residual_reference(problem_scale),
        }
    }
}

/// Trust of a verified iterate: `ResidualChecked` when its true residual
/// passed the tolerance, `Admissible` otherwise.
pub(crate) const fn checkpoint_trust(satisfied: bool) -> scioxide_core::PartialResultTrust {
    if satisfied {
        scioxide_core::PartialResultTrust::ResidualChecked
    } else {
        scioxide_core::PartialResultTrust::Admissible
    }
}

/// Summary of a solve that returns a verified iterate with true residual
/// `residual`; its trust follows from `termination`.
pub(crate) fn solve_summary<R: scioxide_core::RealScalar>(
    context: &scioxide_core::SolveContext<'_, '_>,
    termination: crate::LinearSolveTermination,
    residual: TrueResidual<R>,
) -> crate::result::IterativeSolveSummary<R> {
    crate::result::IterativeSolveSummary {
        termination,
        iterations: context.counts().own().iterations(),
        raw_residual_norm: residual.raw,
        residual_reference: residual.reference,
        trust: checkpoint_trust(termination.is_success()),
    }
}

pub(crate) fn evaluation_limit_error(
    context: &scioxide_core::SolveContext<'_, '_>,
) -> scioxide_core::SciOxideError {
    scioxide_core::SciOxideError::convergence_failure(
        scioxide_core::ConvergenceReason::EvaluationLimit,
        scioxide_core::ConvergenceDiagnostics::new().with_counts(context.counts()),
        "operator evaluation budget was exhausted before a residual was available",
    )
}

mod biorthogonal;
pub use biorthogonal::{BicgSolver, CgsSolver, QmrSolver, TfqmrSolver};

mod stationary;
pub use stationary::{
    GaussSeidelSolver, JacobiSolver, RelaxationWeight, RichardsonSolver, SorSolver, SorWeight,
    StationaryMatrix,
};

#[cfg(test)]
mod test_support;

mod support;

#[cfg(test)]
mod floor_tests;
