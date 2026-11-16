//! Enumerates nearest and radius cases for shared execution-contract tests.
use scioxide_spatial::*;
#[derive(Clone, Copy)]
pub enum QueryCase<S> {
    Nearest { k: usize },
    Radius { radius: S },
}
#[allow(clippy::too_many_arguments)]
pub fn run_batch<'scope, 'capability, S: scioxide_core::RealScalar>(
    tree: &KdTree<'_, S>,
    queries: &PointSet<'_, S>,
    kind: QueryCase<S>,
    options: SearchOptions,
    output: &mut [Neighbor<S>],
    counts: &mut [usize],
    workspace: &mut SearchWorkspace<S>,
    context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
) -> Result<usize, scioxide_core::SciOxideError> {
    match kind {
        QueryCase::Nearest { k } => {
            let total = tree.nearest_batch_into(queries, k, options, output, workspace, context)?;
            counts
                .iter_mut()
                .take(queries.len())
                .for_each(|count| *count = k.min(tree.len()));
            Ok(total)
        }
        QueryCase::Radius { radius } => {
            tree.radius_batch_into(queries, radius, options, output, counts, workspace, context)
        }
    }
}
