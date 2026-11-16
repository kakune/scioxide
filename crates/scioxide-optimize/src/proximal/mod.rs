//! Proximal gradient methods for composite problems and ADMM.

mod admm;
mod gradient;
#[cfg(test)]
mod tests;

pub use admm::{
    Admm, AdmmBlock, AdmmBlockUpdate, AdmmOptions, AdmmProblem, AdmmProgress, AdmmResult,
    AdmmStart, AdmmUpdateInput, AdmmUpdateReport, AdmmWorkspace, AdmmWorkspaceRequirements,
};
pub use gradient::{
    CompositeProblem, Fista, ProximalGradient, ProximalMap, ProximalOptions, ProximalProgress,
    ProximalResult, ProximalWorkspace, ProximalWorkspaceRequirements, SmoothGradient,
};

use scioxide_core::{NumericScalar, RealScalar, ScaledNorm, SciOxideError};

pub(super) use crate::validation::shape;

pub(super) use crate::validation::finite_numeric as finite;

fn stop(
    context: &scioxide_core::SolveContext<'_, '_>,
) -> Result<Option<crate::OptimizeTermination>, SciOxideError> {
    Ok(context.stop_reason()?.map(|reason| match reason {
        scioxide_core::ExecutionStopReason::DeadlineReached => {
            crate::OptimizeTermination::TimeLimit
        }
        _ => crate::OptimizeTermination::Cancelled,
    }))
}

pub(super) fn norm<S: NumericScalar>(values: &[S]) -> Result<S::Real, SciOxideError> {
    let mut scaled = ScaledNorm::new();
    for value in values {
        scaled.add_magnitude(value.abs())?;
    }
    scaled.finish()
}

/// Require a finite positive `value` supplied as the entry point's `argument`.
pub(super) fn positive<R: RealScalar>(
    argument: scioxide_core::Argument,
    value: R,
) -> Result<(), SciOxideError> {
    if !value.is_finite() || value <= R::ZERO {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "expected a finite positive value",
        ));
    }
    Ok(())
}
