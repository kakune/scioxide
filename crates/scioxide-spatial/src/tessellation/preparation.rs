//! Shared resource checks for construction, and scoped construction.
use crate::tessellation::storage::Storage;
use crate::tessellation::work::Work;
use crate::tessellation::{
    GeometryOptions, GeometryWorkspace, GeometryWorkspaceRequirements, Result,
};
use scioxide_core::{SolveBudgets, SolveContext, SolveScope};
pub(super) fn scoped<'scope, 'capability, T>(
    scope: impl Into<SolveScope<'scope, 'capability>>,
    run: impl FnOnce(&mut SolveContext<'_, 'capability>) -> Result<T>,
) -> Result<T> {
    crate::context::scoped(
        scope,
        scioxide_core::algorithm_id!("spatial", "tessellation", 1),
        SolveBudgets::unbounded(),
        run,
    )
}
pub(super) fn prepare<'scope, 'capability, T>(
    dimension: usize,
    point_count: usize,
    source: scioxide_core::Argument,
    options: GeometryOptions,
    scope: Option<SolveScope<'scope, 'capability>>,
    operation: impl FnOnce(&mut Work<'_, '_, '_, 'capability>) -> Result<T>,
) -> Result<T> {
    let requirements = GeometryWorkspaceRequirements::checked(dimension, options, source)?;
    super::limit(
        scioxide_core::quantity!("geometry_points"),
        point_count,
        options.limits.max_points,
    )?;
    let storage = Storage::new(options.limits.max_storage_bytes, requirements.bytes())?;
    let run = |context: Option<&mut SolveContext<'_, 'capability>>| {
        let mut workspace = GeometryWorkspace::prepared(requirements)?;
        let mut work = Work::prepared(&mut workspace, options, context, &storage)?;
        let result = operation(&mut work)?;
        if let Some(context) = work.context {
            publish(context)?;
        }
        Ok(result)
    };
    match scope {
        Some(scope) => scoped(scope, |context| run(Some(context))),
        None => run(None),
    }
}

fn publish(context: &mut SolveContext<'_, '_>) -> Result<()> {
    context.execution().check()?;
    context.commit_accepted(scioxide_core::PartialResultTrust::Admissible)?;
    Ok(())
}
