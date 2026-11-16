//! Workspace sizes, tile scratch and batch evaluation of a KDE at query points.
use super::RawGaussianKde;
use crate::error_helpers::invalid;
use crate::estimation::{execute_ranges, worker_count};
use crate::{ObservationData, StatisticsOptions};
use alloc::vec::Vec;
use scioxide_core::backend::try_filled;
use scioxide_core::backend::try_with_capacity;
use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};

pub(super) const QUERY_TILE: usize = 8;

/// Storage sizes for evaluating a KDE at query points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KdeWorkspaceRequirements {
    /// Coordinates per point.
    pub dimensions: usize,
    /// Total number of query points in all batches.
    pub queries: usize,
    /// Number of workers.
    pub workers: usize,
}
impl KdeWorkspaceRequirements {
    /// Returns the heap bytes the workspace needs, excluding allocator overhead and the output.
    ///
    /// # Errors
    /// Returns `SizeOverflow` if the byte count overflows `usize`.
    pub(crate) fn required_bytes<S: RealScalar>(self) -> Result<usize, SciOxideError> {
        let scratch = crate::estimation::storage_bytes(&[(
            self.coordinates()?,
            core::mem::size_of::<S>() + core::mem::size_of::<f64>(),
        )])?;
        crate::estimation::storage_bytes(&[
            (self.workers, scratch),
            (self.workers, core::mem::size_of::<QueryWorker<S>>()),
            (self.queries, core::mem::size_of::<S>()),
        ])
    }
    fn coordinates(self) -> Result<usize, SciOxideError> {
        self.dimensions
            .checked_mul(QUERY_TILE)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("kde_query_tile")))
    }
    /// Computes the storage sizes for this KDE and query layout.
    ///
    /// # Errors
    /// Fails on a dimension mismatch, weighted queries, or size overflow.
    pub(crate) fn new<S: RealScalar>(
        kde: &RawGaussianKde<S>,
        queries: ObservationData<'_, S>,
        options: StatisticsOptions,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        if queries.dimensions() != kde.dimensions || queries.weights.is_some() {
            return Err(invalid(
                scioxide_core::argument!("queries"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "KDE query dimensions or weights invalid",
            ));
        }
        let n = queries
            .samples()
            .checked_mul(queries.batches())
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("kde_queries")))?;
        Ok(Self {
            dimensions: kde.dimensions,
            queries: n,
            workers: worker_count(options, context, n)?,
        })
    }
}
struct QueryWorker<S> {
    point: Vec<S>,
    scratch: Vec<f64>,
}
/// Scratch for tiles of eight query points, triangular-solve scratch and result buffers.
pub(crate) struct RawKdeWorkspace<S: RealScalar> {
    workers: Vec<QueryWorker<S>>,
    staged: Vec<S>,
}
impl<S: RealScalar> Default for RawKdeWorkspace<S> {
    fn default() -> Self {
        Self {
            workers: Vec::new(),
            staged: Vec::new(),
        }
    }
}
impl<S: RealScalar> RawKdeWorkspace<S> {
    pub(crate) fn sufficient(&self, r: KdeWorkspaceRequirements) -> bool {
        self.workers.len() >= r.workers
            && self.workers.iter().take(r.workers).all(|w| {
                w.point.len() >= r.coordinates().unwrap_or(usize::MAX)
                    && w.scratch.len() >= r.coordinates().unwrap_or(usize::MAX)
            })
            && self.staged.len() >= r.queries
    }
    /// Allocates storage of the given sizes. On error the previous storage is unchanged.
    ///
    /// # Errors
    /// Returns size overflow or allocation failure.
    pub(crate) fn reserve(&mut self, r: KdeWorkspaceRequirements) -> Result<(), SciOxideError> {
        r.required_bytes::<S>()?;
        let coordinates = r.coordinates()?;
        let mut workers =
            try_with_capacity(r.workers, scioxide_core::quantity!("kde_query_workers"))?;
        for _ in 0..r.workers {
            workers.push(QueryWorker {
                point: try_filled(coordinates, S::ZERO, scioxide_core::quantity!("kde_point"))?,
                scratch: try_filled(coordinates, 0., scioxide_core::quantity!("kde_scratch"))?,
            });
        }
        *self = Self {
            workers,
            staged: try_filled(r.queries, S::ZERO, scioxide_core::quantity!("kde_staged"))?,
        };
        Ok(())
    }
}
/// Evaluates the density or log density at all query points, in batch-major order.
/// Query coordinates must be present and finite; queries are never skipped.
/// Each pass over the centers serves a tile of eight query points.
///
/// # Errors
/// Fails on invalid or missing query points, a workspace that is too small, or an
/// execution failure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kde_into<S: RealScalar>(
    kde: &RawGaussianKde<S>,
    queries: ObservationData<'_, S>,
    scale: crate::KdeScale,
    options: StatisticsOptions,
    context: ExecutionContext<'_>,
    workspace: &mut RawKdeWorkspace<S>,
    output: &mut [S],
) -> Result<(), SciOxideError> {
    if scale == crate::KdeScale::LogDensity {
        queries_into::<S, true>(kde, queries, options, context, workspace, output)
    } else {
        queries_into::<S, false>(kde, queries, options, context, workspace, output)
    }
}

fn queries_into<S: RealScalar, const LOG: bool>(
    kde: &RawGaussianKde<S>,
    queries: ObservationData<'_, S>,
    options: StatisticsOptions,
    context: ExecutionContext<'_>,
    workspace: &mut RawKdeWorkspace<S>,
    output: &mut [S],
) -> Result<(), SciOxideError> {
    let r = KdeWorkspaceRequirements::new(kde, queries, options, context)?;
    let coordinates = r.coordinates()?;
    if output.len() != r.queries
        || workspace.workers.len() < r.workers
        || workspace
            .workers
            .iter()
            .take(r.workers)
            .any(|w| w.point.len() < coordinates || w.scratch.len() < coordinates)
        || workspace.staged.len() < r.queries
    {
        return Err(invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient KDE output/workspace",
        ));
    }
    execute_ranges(
        options,
        context,
        r.queries,
        1,
        &mut workspace.workers,
        &mut workspace.staged,
        |range, w, out| {
            for (tile, output) in out.chunks_mut(QUERY_TILE).enumerate() {
                context.check()?;
                let count = output.len() * r.dimensions;
                for (lane, point) in w.point[..count].chunks_exact_mut(r.dimensions).enumerate() {
                    let i = range.start + tile * QUERY_TILE + lane;
                    let batch = i / queries.samples();
                    let sample = i % queries.samples();
                    let origin = queries.batch_origin(batch);
                    if queries
                        .admit_count_by(
                            batch,
                            sample,
                            StatisticsOptions::default(),
                            scioxide_core::argument!("queries"),
                            |j| queries.value_at_origin(origin, sample, j),
                        )?
                        .is_none()
                    {
                        return Err(invalid(
                            scioxide_core::argument!("queries"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "missing KDE query",
                        ));
                    }
                    for (j, value) in point.iter_mut().enumerate() {
                        *value = queries.value_at_origin(origin, sample, j);
                    }
                }
                kde.evaluate_tile::<LOG>(
                    &w.point[..count],
                    &mut w.scratch[..count],
                    context,
                    output,
                )?;
            }
            Ok(())
        },
    )?;
    output.copy_from_slice(&workspace.staged[..r.queries]);
    Ok(())
}
