//! Conservative finite-volume laws, numerical fluxes and prepared execution.
use crate::validation::finite;
use crate::validation::invalid;

use scioxide_core::{RealScalar, SciOxideError};

mod arithmetic;
mod flux;
pub use flux::{
    Burgers, ConservationLaw, Godunov, Hllc, Hlle, IdealGasEuler, LinearAdvection, NumericalFlux,
    RiemannData, RiemannReport, Rusanov,
};

/// Inconsistent-input error against `argument`.
pub(super) fn number<R: RealScalar>(value: usize) -> Result<R, SciOxideError> {
    R::try_from_usize_exact(value)
}
mod plan;
pub use plan::{
    ConservationBoundary, ConservationLimits, ConservationOptions, ConservationReconstruction,
    ConservationReport, ConservationWorkspace, SimplexConservationPlan, StructuredConservationPlan,
};
mod time;
pub use time::ConservationSystem;
#[cfg(test)]
mod tests;
