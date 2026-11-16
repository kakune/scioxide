//! Options and workspaces for KD-tree queries.

use crate::search::Axis;
use crate::workspace::{add, bytes, grow, limit, require};
use crate::{KdTree, Neighbor};
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use scioxide_core::{Capacity, ExecutionPolicy, RealScalar, SciOxideError};

pub(crate) const MAX_WORKERS: usize = 32;

/// Options for KD-tree queries: approximation, execution policy and a
/// workspace byte limit.
#[derive(Debug, Clone, Copy)]
pub struct SearchOptions {
    epsilon: f64,
    policy: ExecutionPolicy,
    max_workspace_bytes: usize,
}
impl Default for SearchOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl SearchOptions {
    /// Returns exact, sequential search without a byte limit.
    pub const fn new() -> Self {
        Self {
            epsilon: 0.0,
            policy: ExecutionPolicy::Sequential,
            max_workspace_bytes: usize::MAX,
        }
    }
    /// Sets the relative approximation `ε ≥ 0`. With `ε > 0`, distances may be
    /// off by a factor up to `1 + ε`, as described on
    /// [`KdTree::nearest_into`](crate::KdTree::nearest_into) and
    /// [`KdTree::radius_into`](crate::KdTree::radius_into). The default is
    /// zero, exact search.
    pub const fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
    /// Sets the execution policy of [`KdTree::nearest_batch_into`] and [`KdTree::radius_batch_into`].
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Sets the limit on the bytes a [`SearchWorkspace`] may hold, including
    /// capacity from earlier, larger reservations.
    pub const fn with_max_workspace_bytes(mut self, maximum: usize) -> Self {
        self.max_workspace_bytes = maximum;
        self
    }
    /// Returns the relative approximation `ε`.
    pub const fn epsilon(self) -> f64 {
        self.epsilon
    }
    /// Returns the execution policy.
    pub const fn execution_policy(self) -> ExecutionPolicy {
        self.policy
    }
    /// Returns the workspace byte limit.
    pub const fn max_workspace_bytes(self) -> usize {
        self.max_workspace_bytes
    }
    pub(crate) fn factor(self) -> Result<f64, SciOxideError> {
        if !self.epsilon.is_finite() || self.epsilon < 0.0 {
            return Err(crate::validation::out_of_range(
                scioxide_core::argument!("options.epsilon"),
                "epsilon must be finite and nonnegative",
            ));
        }
        crate::distance::finite(1.0 + self.epsilon)
    }
}

/// The sizes a [`SearchWorkspace`] must support.
///
/// A single nearest-neighbor query needs `min(k, tree.len())` neighbor slots;
/// a single radius query needs a bound on the number of matches you expect.
/// A batch nearest query needs the same per worker. A batch radius query needs
/// none per worker, because it counts the matches before it stores them.
#[derive(Debug, Clone, Copy)]
pub struct SearchWorkspaceRequest {
    neighbors: usize,
    queries: Option<usize>,
    output: usize,
    workers: NonZeroUsize,
    options: SearchOptions,
}
impl SearchWorkspaceRequest {
    /// Requests space for single queries with up to `neighbors` results.
    pub const fn new(neighbors: usize, options: SearchOptions) -> Self {
        Self {
            neighbors,
            queries: None,
            output: 0,
            workers: NonZeroUsize::MIN,
            options,
        }
    }
    /// Adds space for [`KdTree::nearest_batch_into`](crate::KdTree::nearest_batch_into)
    /// and [`KdTree::radius_batch_into`](crate::KdTree::radius_batch_into) with
    /// `queries` queries, `neighbors` results in total over all queries, and
    /// up to `workers` workers, at most 32.
    ///
    /// The number of workers that run is chosen by the options and the
    /// execution context; this only sizes the workspace.
    pub const fn with_batch(
        mut self,
        queries: usize,
        neighbors: usize,
        workers: NonZeroUsize,
    ) -> Self {
        self.queries = Some(queries);
        self.output = neighbors;
        self.workers = workers;
        self
    }
}

/// Workspace sizes for KD-tree queries, and the byte limit.
#[derive(Debug, Clone, Copy)]
pub struct SearchWorkspaceRequirements {
    pub(crate) dimension: usize,
    // Per-worker query row, loaded only by batches; scalar queries borrow theirs.
    pub(crate) query: usize,
    pub(crate) neighbors: usize,
    pub(crate) workers: usize,
    pub(crate) counts: usize,
    pub(crate) output: usize,
    max_workspace_bytes: usize,
}
impl SearchWorkspaceRequirements {
    /// Returns the number of workers to reserve for.
    pub const fn workers(self) -> usize {
        self.workers
    }
    /// Returns the neighbor slots per worker.
    pub const fn neighbors_per_worker(self) -> usize {
        self.neighbors
    }
    /// Returns the total neighbor slots for batch results.
    pub const fn output_len(self) -> usize {
        self.output
    }
    /// Returns the slots for per-query counts: one more than the number of
    /// batch queries, or zero without a nonempty batch.
    pub const fn count_len(self) -> usize {
        self.counts
    }
    /// Returns the bytes of per-worker storage for scalar type `S`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size exceeds `usize`.
    pub fn worker_bytes<S: RealScalar>(self) -> Result<usize, SciOxideError> {
        if self.workers == 0 {
            return Ok(0);
        }
        let per_worker = add(
            bytes::<SearchWorker<S>>(1)?,
            add(
                bytes::<Neighbor<S>>(self.neighbors)?,
                add(bytes::<Axis>(self.dimension)?, bytes::<S>(self.query)?)?,
            )?,
        )?;
        self.workers
            .checked_mul(per_worker)
            .ok_or_else(crate::workspace::overflow)
    }
    /// Returns the bytes of batch result storage for scalar type `S`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size exceeds `usize`.
    pub fn output_bytes<S: RealScalar>(self) -> Result<usize, SciOxideError> {
        add(
            bytes::<Neighbor<S>>(self.output)?,
            bytes::<usize>(self.counts)?,
        )
    }
    /// Returns the total bytes for scalar type `S`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size exceeds `usize`.
    /// - `ResourceLimit` if it exceeds the byte limit.
    pub fn bytes<S: RealScalar>(self) -> Result<usize, SciOxideError> {
        let total = add(self.worker_bytes::<S>()?, self.output_bytes::<S>()?)?;
        limit(total, self.max_workspace_bytes)?;
        Ok(total)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SearchWorker<S: RealScalar> {
    pub(crate) candidates: Vec<Neighbor<S>>,
    pub(crate) axes: Vec<Axis>,
    pub(crate) query: Vec<S>,
}
impl<S: RealScalar> SearchWorker<S> {
    const fn new() -> Self {
        Self {
            candidates: Vec::new(),
            axes: Vec::new(),
            query: Vec::new(),
        }
    }
}

/// Workspace for KD-tree queries: scratch for each worker and storage for
/// batch results.
///
/// Results are written here first and copied to the output only after the
/// query succeeds, so the output is unchanged on error. Queries never grow
/// the workspace.
///
/// ```
/// use scioxide_spatial::{
///     KdTree, KdTreeOptions, Neighbor, PointSet, SearchOptions, SearchWorkspaceRequest,
///     SpatialExecutionContext, TreeMetric,
/// };
/// let points = PointSet::from_points(&[[0.0_f64], [1.0]])?;
/// let tree = KdTree::new(&points)?;
/// let options = SearchOptions::new();
/// let mut workspace = tree.workspace(SearchWorkspaceRequest::new(1, options))?;
/// let mut out = [Neighbor::default(); 1];
/// tree.nearest_into(
///     &[0.5],
///     1,
///     options,
///     &mut out,
///     &mut workspace,
///     SpatialExecutionContext::sequential(),
/// )?;
/// assert_eq!(out[0].index, 0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct SearchWorkspace<S: RealScalar> {
    pub(crate) workers: Vec<SearchWorker<S>>,
    pub(crate) output: Vec<Neighbor<S>>,
    pub(crate) counts: Vec<usize>,
}
impl<S: RealScalar> Default for SearchWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> SearchWorkspace<S> {
    /// Returns an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            workers: Vec::new(),
            output: Vec::new(),
            counts: Vec::new(),
        }
    }
    fn retained(
        &self,
        r: &SearchWorkspaceRequirements,
        projected: bool,
    ) -> Result<usize, SciOxideError> {
        let bound = |actual: usize, needed: usize| {
            if projected {
                actual.max(needed)
            } else {
                actual
            }
        };
        let mut total = add(
            bytes::<SearchWorker<S>>(bound(self.workers.capacity(), r.workers))?,
            add(
                bytes::<Neighbor<S>>(bound(self.output.capacity(), r.output))?,
                bytes::<usize>(bound(self.counts.capacity(), r.counts))?,
            )?,
        )?;
        for index in 0..bound(self.workers.len(), r.workers) {
            let worker = self.workers.get(index);
            let needed = projected && index < r.workers;
            let dimension = if needed { r.dimension } else { 0 };
            let query = if needed { r.query } else { 0 };
            let neighbors = if needed { r.neighbors } else { 0 };
            total = add(
                total,
                bytes::<Neighbor<S>>(worker.map_or(0, |w| w.candidates.capacity()).max(neighbors))?,
            )?;
            total = add(
                total,
                bytes::<Axis>(worker.map_or(0, |w| w.axes.capacity()).max(dimension))?,
            )?;
            total = add(
                total,
                bytes::<S>(worker.map_or(0, |w| w.query.capacity()).max(query))?,
            )?;
        }
        Ok(total)
    }
    /// Reports whether the workspace meets `r`.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size exceeds `usize`.
    /// - `ResourceLimit` if the capacity already held exceeds the byte limit.
    pub fn check(&self, r: &SearchWorkspaceRequirements) -> Result<Capacity, SciOxideError> {
        r.bytes::<S>()?;
        // Actual retained allocations already fit the address space. Sum them
        // only for a finite ceiling; requirement arithmetic remains checked.
        if r.max_workspace_bytes != usize::MAX {
            limit(self.retained(r, false)?, r.max_workspace_bytes)?;
        }
        Ok(
            if self.workers.len() >= r.workers
                && self.output.len() >= r.output
                && self.counts.len() >= r.counts
                && self.workers[..r.workers].iter().all(|w| {
                    w.axes.len() >= r.dimension
                        && w.query.len() >= r.query
                        && w.candidates.len() >= r.neighbors
                })
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
    /// Grows the workspace to meet `r`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the byte limit.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   reserved. The workspace may then keep part of the new capacity.
    pub fn try_reserve(&mut self, r: &SearchWorkspaceRequirements) -> Result<(), SciOxideError> {
        r.bytes::<S>()?;
        limit(self.retained(r, true)?, r.max_workspace_bytes)?;
        grow(&mut self.workers, r.workers, SearchWorker::new())?;
        for worker in &mut self.workers[..r.workers] {
            grow(&mut worker.axes, r.dimension, Axis::default())?;
            grow(&mut worker.query, r.query, S::ZERO)?;
            grow(&mut worker.candidates, r.neighbors, Neighbor::default())?;
        }
        grow(&mut self.output, r.output, Neighbor::default())?;
        grow(&mut self.counts, r.counts, 0)?;
        let _ = self.check(r)?;
        Ok(())
    }
    pub(crate) fn require(&self, r: &SearchWorkspaceRequirements) -> Result<(), SciOxideError> {
        if self.check(r)? == Capacity::NeedsGrowth {
            require(self.workers.len(), r.workers)?;
            require(self.output.len(), r.output)?;
            require(self.counts.len(), r.counts)?;
            for worker in &self.workers[..r.workers] {
                require(worker.axes.len(), r.dimension)?;
                require(worker.query.len(), r.query)?;
                require(worker.candidates.len(), r.neighbors)?;
            }
        }
        Ok(())
    }
}

impl<S: RealScalar> KdTree<'_, S> {
    /// Returns the workspace sizes for queries on this tree.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the `ε` of the options is negative or not finite.
    /// - `SizeOverflow` if a size exceeds `usize`.
    /// - `ResourceLimit` if the workspace would exceed the byte limit.
    pub fn workspace_requirements(
        &self,
        request: SearchWorkspaceRequest,
    ) -> Result<SearchWorkspaceRequirements, SciOxideError> {
        request.options.factor()?;
        self.query_requirements(request)
    }
    // Execution has already validated approximation while preparing its
    // pruning factor. Keep requirement geometry on the same checked path.
    pub(crate) fn query_requirements(
        &self,
        request: SearchWorkspaceRequest,
    ) -> Result<SearchWorkspaceRequirements, SciOxideError> {
        let r = SearchWorkspaceRequirements {
            dimension: self.dimension(),
            query: if request.queries.is_some() {
                self.dimension()
            } else {
                0
            },
            neighbors: request.neighbors.min(self.len()),
            workers: if request.queries == Some(0) {
                0
            } else {
                request.workers.get().min(MAX_WORKERS)
            },
            counts: request
                .queries
                .map(|n| if n == 0 { Ok(0) } else { add(n, 1) })
                .transpose()?
                .unwrap_or(0),
            output: request.output,
            max_workspace_bytes: request.options.max_workspace_bytes,
        };
        r.bytes::<S>()?;
        Ok(r)
    }
    /// Allocates a workspace for queries on this tree.
    ///
    /// # Errors
    ///
    /// As [`Self::workspace_requirements`], and `AllocationFailure` if the
    /// storage cannot be reserved.
    pub fn workspace(
        &self,
        request: SearchWorkspaceRequest,
    ) -> Result<SearchWorkspace<S>, SciOxideError> {
        let mut workspace = SearchWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }
}
