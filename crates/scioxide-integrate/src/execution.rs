//! All-or-nothing one-dimensional execution and evaluation accounting.
use crate::IntegrationTermination;
#[derive(Debug, Clone, Copy)]
pub(crate) struct RegionalProgress {
    pub(crate) termination: IntegrationTermination,
    pub(crate) measured: bool,
    pub(crate) num_evals: usize,
    pub(crate) subdivisions: usize,
}

use crate::workspace::{Attempts, Outcome};
use crate::{IntegrationErrorComponents, IntegrationReport};
use scioxide_core::{
    AffineValueMut, AlgorithmId, ErrorTolerance, ExecutionContext, PartialResultTrust, RealScalar,
    SciOxideError, SolveBudgets, SolveContext, SolveScope,
};

/// Input checks, in a fixed order, for one-dimensional calls.
#[inline(always)]
pub(crate) fn preflight<S: RealScalar, O: AffineValueMut<Scale = S> + ?Sized>(
    attempts: &mut Attempts<IntegrationReport<S>>,
    published: &mut bool,
    out: &O,
    components: usize,
    tolerance: Option<&ErrorTolerance<S>>,
    validate: impl FnOnce() -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    *published = false;
    let result = scioxide_core::check_value_shape(out, components)
        .and_then(|()| {
            tolerance.map_or(Ok(()), |t| crate::workspace::check_tolerance(t, components))
        })
        .and_then(|()| validate());
    match result {
        Ok(()) => Ok(()),
        Err(error) => attempts.reject(error),
    }
}

/// Attach evaluation counts and expose component errors only when a value was written.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn integrate<'capability, S: RealScalar>(
    attempts: &mut Attempts<IntegrationReport<S>>,
    published: &mut bool,
    scope: SolveScope<'_, 'capability>,
    algorithm: AlgorithmId,
    shape: usize,
    budgets: SolveBudgets,
    kernel: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<Outcome<S>, SciOxideError>,
) -> Result<IntegrationReport<S>, SciOxideError> {
    *published = false;
    let report = attempts.integrate(scope, algorithm, shape, budgets, kernel)?;
    *published = report.error_components().is_some();
    Ok(report)
}

/// Write exact zero without calling the integrand or estimating an error.
pub(crate) fn zero<S: RealScalar, O: AffineValueMut<Scale = S> + ?Sized>(
    values: &mut [S],
    out: &mut O,
    termination: IntegrationTermination,
    context: &mut SolveContext<'_, '_>,
) -> Result<Outcome<S>, SciOxideError> {
    values.fill(S::ZERO);
    context.commit_accepted(PartialResultTrust::Admissible)?;
    scioxide_core::write_value(values, out)?;
    Ok(Outcome::Published {
        termination,
        subdivisions: 0,
        errors: None,
    })
}

/// Owned-value form that allocates a workspace and runs the prepared form.
pub(crate) fn owned<S: RealScalar, V: AffineValueMut<Scale = S>, W>(
    mut out: V,
    prepare: impl FnOnce(usize) -> Result<W, SciOxideError>,
    execute: impl FnOnce(
        &mut V,
        &mut W,
        ExecutionContext<'static>,
    ) -> Result<IntegrationReport<S>, SciOxideError>,
) -> Result<(V, IntegrationReport<S>), SciOxideError> {
    let mut workspace = prepare(out.dimension()?)?;
    let report = execute(
        &mut out,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    Ok((out, report))
}

pub(crate) fn component_errors<S: RealScalar>(
    published: bool,
    errors: &[IntegrationErrorComponents<S>],
) -> Option<&[IntegrationErrorComponents<S>]> {
    published.then_some(errors)
}
