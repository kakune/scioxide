//! Layouts and requirements of the operator-action workspaces.
//!
//! Each workspace accepts only its own requirements:
//! ```compile_fail
//! use scioxide_linalg::{ExponentialActionWorkspace, Matrix, OneNormEstimator};
//! let matrix = Matrix::<f64>::identity(2).unwrap();
//! let operator = matrix.view().try_finite().unwrap();
//! let plan = OneNormEstimator::new();
//! let requirements = plan.workspace_requirements(&operator).unwrap();
//! ExponentialActionWorkspace::<f64>::new().try_reserve(&requirements).unwrap();
//! ```
use crate::workspace::accounting::requirements;
use scioxide_core::NumericScalar;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ShiftLayout {
    pub a: usize,
    pub mass: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExponentialLayout {
    pub(crate) n: usize,
    pub(crate) columns: usize,
    pub(crate) phi: usize,
    pub(crate) provider: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FunctionLayout {
    pub(crate) n: usize,
    pub(crate) columns: usize,
    pub(crate) restart: usize,
    pub(crate) projection: usize,
    pub(crate) provider: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LobpcgLayout {
    pub(crate) n: usize,
    pub(crate) count: usize,
    pub(crate) constraints: usize,
    pub(crate) provider: usize,
    pub(crate) mass: usize,
    pub(crate) preconditioner: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpectrumLayout {
    pub(crate) n: usize,
    pub(crate) basis: usize,
    pub(crate) count: usize,
    pub(crate) structure: super::OperatorStructure,
    pub(crate) provider: usize,
    pub(crate) shift: Option<ShiftLayout>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SvdLayout {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) basis: usize,
    pub(crate) count: usize,
    pub(crate) forward: usize,
    pub(crate) adjoint: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RandomizedLayout {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) width: usize,
    pub(crate) rank: usize,
    pub(crate) forward: usize,
    pub(crate) adjoint: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OneNormLayout {
    pub(crate) n: usize,
    pub(crate) provider: usize,
}

requirements!(
    ExponentialActionRequirements,
    ExponentialLayout,
    "Workspace storage and byte limit of a [`ExponentialAction`](crate::ExponentialAction). Returned by [`ExponentialAction::workspace_requirements`](crate::ExponentialAction::workspace_requirements) and accepted only by [`ExponentialActionWorkspace`](crate::ExponentialActionWorkspace)."
);
requirements!(
    FunctionActionRequirements,
    FunctionLayout,
    "Workspace storage and byte limit of a [`FunctionAction`](crate::FunctionAction). Returned by [`FunctionAction::workspace_requirements`](crate::FunctionAction::workspace_requirements) and accepted only by [`FunctionActionWorkspace`](crate::FunctionActionWorkspace)."
);
requirements!(
    LobpcgRequirements,
    LobpcgLayout,
    "Workspace storage and byte limit of a [`LobpcgSolver`](crate::LobpcgSolver). Returned by [`LobpcgSolver::workspace_requirements`](crate::LobpcgSolver::workspace_requirements) and accepted only by [`LobpcgWorkspace`](crate::LobpcgWorkspace)."
);
requirements!(
    PartialSpectrumRequirements,
    SpectrumLayout,
    "Workspace storage and byte limit of a [`PartialSpectrumSolver`](crate::PartialSpectrumSolver). Returned by [`PartialSpectrumSolver::workspace_requirements`](crate::PartialSpectrumSolver::workspace_requirements) and accepted only by [`PartialSpectrumWorkspace`](crate::PartialSpectrumWorkspace)."
);
requirements!(
    PartialSvdRequirements,
    SvdLayout,
    "Workspace storage and byte limit of a [`PartialSvdSolver`](crate::PartialSvdSolver). Returned by [`PartialSvdSolver::workspace_requirements`](crate::PartialSvdSolver::workspace_requirements) and accepted only by [`PartialSvdWorkspace`](crate::PartialSvdWorkspace)."
);
requirements!(
    RandomizedLowRankRequirements,
    RandomizedLayout,
    "Workspace storage and byte limit of a [`RandomizedLowRank`](crate::RandomizedLowRank). Returned by [`RandomizedLowRank::workspace_requirements`](crate::RandomizedLowRank::workspace_requirements) and accepted only by [`RandomizedLowRankWorkspace`](crate::RandomizedLowRankWorkspace)."
);
requirements!(
    OneNormRequirements,
    OneNormLayout,
    "Workspace storage and byte limit of a [`OneNormEstimator`](crate::OneNormEstimator). Returned by [`OneNormEstimator::workspace_requirements`](crate::OneNormEstimator::workspace_requirements) and accepted only by [`OneNormWorkspace`](crate::OneNormWorkspace)."
);
