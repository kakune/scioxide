//! KD trees for nearest-neighbor and radius queries.
//!
//! Euclidean leaf candidates compare sums of squares against a conservative
//! cutoff before taking square roots. Coordinates of extreme magnitude use the
//! rescaled norm of the distance functions, so ties and overflow errors match.

use crate::distance::{PointSet, finite, norm};
use crate::validation::out_of_range as invalid;
use crate::workspace::require;
use crate::{SearchOptions, SearchWorkspace, SearchWorkspaceRequest, SpatialExecutionContext};
use alloc::vec::Vec;
use core::cmp::Ordering;
use scioxide_core::backend::try_filled;
use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};

#[cfg(test)]
mod tests;

const NODE_POLL_INTERVAL: usize = 64;
const LEAF_POLL_INTERVAL: usize = 256;

// One budget per scalar call or batch partition, shared across query rows.
// Callers check at entry and before publication, including empty operations.
pub(crate) struct SearchPoll<'a> {
    context: ExecutionContext<'a>,
    remaining: usize,
    #[cfg(test)]
    visits: usize,
    #[cfg(test)]
    points: usize,
}
impl<'a> SearchPoll<'a> {
    pub(crate) fn new(context: ExecutionContext<'a>) -> Self {
        Self {
            context,
            remaining: NODE_POLL_INTERVAL,
            #[cfg(test)]
            visits: 0,
            #[cfg(test)]
            points: 0,
        }
    }
    #[inline]
    pub(crate) fn visit(&mut self) -> Result<(), SciOxideError> {
        if self.remaining == 0 {
            self.check()?;
        }
        self.remaining -= 1;
        #[cfg(test)]
        {
            self.visits += 1;
        }
        Ok(())
    }
    fn check(&mut self) -> Result<(), SciOxideError> {
        self.context.check()?;
        self.remaining = NODE_POLL_INTERVAL;
        Ok(())
    }
}

/// Distance used by a [`KdTree`].
///
/// These are the metrics for which the distance to an axis-aligned box gives
/// a lower bound for the distance to every point in it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TreeMetric {
    /// Euclidean distance `|u − v|`.
    Euclidean,
    /// City-block (Manhattan) distance `Σ|uᵢ − vᵢ|`.
    CityBlock,
    /// Chebyshev distance `maxᵢ |uᵢ − vᵢ|`.
    Chebyshev,
    /// Minkowski distance `(Σ|uᵢ − vᵢ|ᵖ)^(1/p)` for a finite `p ≥ 1`.
    Minkowski(f64),
}
impl TreeMetric {
    // Norm power of the `metric` argument.
    fn power(self) -> Result<f64, SciOxideError> {
        match self {
            Self::Euclidean => Ok(2.0),
            Self::CityBlock => Ok(1.0),
            Self::Chebyshev => Ok(f64::INFINITY),
            Self::Minkowski(p) if p.is_finite() && p >= 1.0 => Ok(p),
            _ => Err(invalid(
                scioxide_core::argument!("metric"),
                "tree Minkowski exponent must be finite and at least one",
            )),
        }
    }
}
/// Options for building a [`KdTree`]: leaf size and an optional periodic box.
///
/// In a periodic box with side lengths `Lₖ`, every coordinate must lie in
/// `[0, Lₖ)`, and each coordinate difference is taken as the shortest one
/// modulo `Lₖ`.
#[derive(Debug, Clone, Copy)]
pub struct KdTreeOptions<'a, S: RealScalar> {
    /// Maximum points per leaf; positive. Default 16.
    leaf_size: usize,
    /// Optional finite positive lengths, one per coordinate axis.
    periodic_box: Option<&'a [S]>,
}
impl<S: RealScalar> Default for KdTreeOptions<'_, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, S: RealScalar> KdTreeOptions<'a, S> {
    /// Creates options with leaf size 16 and no periodic box.
    pub const fn new() -> Self {
        Self {
            leaf_size: 16,
            periodic_box: None,
        }
    }
    /// Sets the maximum number of points in a leaf; the default is 16.
    pub const fn with_leaf_size(mut self, size: usize) -> Self {
        self.leaf_size = size;
        self
    }
    /// Sets the side lengths of a periodic box, one per axis, or `None` for
    /// no periodicity.
    pub const fn with_periodic_box(mut self, lengths: Option<&'a [S]>) -> Self {
        self.periodic_box = lengths;
        self
    }
    /// Returns the maximum number of points in a leaf.
    pub const fn leaf_size(self) -> usize {
        self.leaf_size
    }
    /// Returns the side lengths of the periodic box, if any.
    pub const fn periodic_box(self) -> Option<&'a [S]> {
        self.periodic_box
    }
}

// Permutation slot. `split` caches the coordinate on the current split axis,
// so the median selection's comparator never re-reads strided borrowed
// storage.
#[derive(Debug, Clone, Copy)]
struct Entry {
    index: usize,
    split: f64,
}
// A node owns the contiguous `entries[start..end]` range. Children split it
// at the median, so each subtree is a subrange and no point lists are stored.
#[derive(Debug, Clone, Copy)]
struct Node {
    start: usize,
    end: usize,
    children: Option<Split>,
}

#[derive(Debug, Clone, Copy)]
struct Split {
    axis: usize,
    left: usize,
    right: usize,
    left_high: f64,
    right_low: f64,
}

// Query coordinates and inherited rectangle limits, prepared once per query.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Axis {
    query: f64,
    seam: f64,
    length: f64,
    low: f64,
    high: f64,
    delta: f64,
}
impl Axis {
    #[inline]
    fn interval_distance<const PERIODIC: bool>(self) -> f64 {
        if self.query < self.low {
            let direct = self.low - self.query;
            if PERIODIC {
                direct.min((self.length - self.high) + self.query)
            } else {
                direct
            }
        } else if self.query > self.high {
            let direct = self.query - self.high;
            if PERIODIC {
                direct.min(self.seam + self.low)
            } else {
                direct
            }
        } else {
            0.0
        }
    }
    #[inline]
    fn point_distance<const PERIODIC: bool>(self, point: f64) -> f64 {
        let delta = (point - self.query).abs();
        if PERIODIC {
            // Use the rectangle-bound association, so a node bound never exceeds
            // a contained point's distance. Whenever the seam image is nearer,
            // it is below L/2, so its first subtraction is exact (Sterbenz) and
            // the image is correctly rounded. `L - |p - q|` would cancel.
            // Branch-free: `L - max` is the seam offset of the larger coordinate.
            delta.min((self.length - point.max(self.query)) + point.min(self.query))
        } else {
            delta
        }
    }
}

// KIND: 0 = maximum, 1 = sum, 2 = squares, 3 = general p-power.
// Contributions only increase as a rectangle shrinks. Rescale upward before
// taking powers, so finite coordinate distances cannot overflow a power.
#[derive(Clone, Copy)]
struct Bound {
    scale: f64,
    sum: f64,
}
impl Bound {
    fn zero() -> Self {
        Self {
            scale: 0.0,
            sum: 0.0,
        }
    }
    #[inline]
    fn increase<const KIND: u8>(self, old: f64, new: f64, p: f64) -> Self {
        if KIND == 0 {
            return Self {
                scale: 1.0,
                sum: self.sum.max(new),
            };
        }
        if new <= old {
            return self;
        }
        if !new.is_finite() {
            return Self {
                scale: f64::INFINITY,
                sum: 1.0,
            };
        }
        if !self.scale.is_finite() {
            return self;
        }
        let scale = self.scale.max(new);
        let power = |x: f64| {
            if KIND == 1 {
                x
            } else if KIND == 2 {
                x * x
            } else {
                RealScalar::powf(x, p)
            }
        };
        let sum = self.sum * power(self.scale / scale)
            + (power(new / scale) - power(old / scale)).max(0.0);
        Self { scale, sum }
    }
    #[inline]
    fn distance<const KIND: u8>(self, p: f64) -> f64 {
        if KIND == 0 {
            self.sum
        } else if KIND == 1 {
            self.scale * self.sum
        } else if KIND == 2 {
            self.scale * RealScalar::sqrt(self.sum)
        } else {
            self.scale * RealScalar::powf(self.sum, 1.0 / p)
        }
    }
}

/// A neighbor found by a query: a point index and its distance.
///
/// Queries return the number of neighbors they wrote; there are no
/// placeholder entries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor<S: RealScalar> {
    /// Row of the point in the indexed [`PointSet`].
    pub index: usize,
    /// Distance from the query, in the metric of the tree.
    pub distance: S,
}
impl<S: RealScalar> Default for Neighbor<S> {
    fn default() -> Self {
        Self {
            index: 0,
            distance: S::ZERO,
        }
    }
}
pub(crate) fn order<S: RealScalar>(a: &Neighbor<S>, b: &Neighbor<S>) -> Ordering {
    a.distance
        .into_f64()
        .total_cmp(&b.distance.into_f64())
        .then_with(|| a.index.cmp(&b.index))
}

/// KD tree over a borrowed [`PointSet`] for nearest-neighbor and radius
/// queries.
///
/// Each node splits its points at the median of the coordinate with the
/// widest spread, so the depth is at most `⌈log₂ n⌉`. Building costs
/// `O(d n log n)` time and `O(n + d)` storage for `n` points in `d` dimensions;
/// the tree stores indices, not coordinates, and borrows the points for its
/// lifetime. Queries prune subtrees by their distance to the query. In high
/// dimensions a query may still visit every point.
///
/// Results are sorted by distance, then by point index; ties are decided on
/// the rounded distances. A distance that is not representable is ignored
/// unless a nearest-neighbor query needs it to return `k` results, and radius
/// queries never include it. Queries check for cancellation at least every 64
/// tree nodes and every 256 points of a large leaf.
///
/// ```
/// use scioxide_spatial::{KdTree, KdTreeOptions, PointSet, SearchOptions, TreeMetric};
///
/// let data = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 2.0, 5.0, 5.0];
/// let points = PointSet::from_rows(&data, 2)?;
/// let tree = KdTree::new(&points)?;
/// let found = tree.nearest(&[0.9, 0.1], 2)?;
/// // (1, 0) is nearest, at distance √0.02; then (0, 0), at distance √0.82.
/// assert_eq!((found[0].index, found[1].index), (1, 0));
/// assert!((found[0].distance - 0.02_f64.sqrt()).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// The tree borrows the coordinates, so they cannot change while it exists:
///
/// ```compile_fail
/// use scioxide_spatial::{KdTree, KdTreeOptions, PointSet, TreeMetric};
/// let mut data = [0.0_f64, 1.0];
/// let points =
///     PointSet::from_rows(&data, 1)
///         .unwrap();
/// let tree = KdTree::new(&points).unwrap();
/// data[0] = 10.0;
/// assert_eq!(tree.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct KdTree<'data, S: RealScalar> {
    points: PointSet<'data, S>,
    entries: Vec<Entry>,
    nodes: Vec<Node>,
    bounds: Vec<(f64, f64)>,
    periodic: Option<Vec<f64>>,
    p: f64,
    // Median splitting bounds depth by ceil(log2(n)); checked 2*n fits usize,
    // so every possible depth fits this fixed array, without query allocation.
    pruning_factors: [f64; usize::BITS as usize],
    leaf_size: usize,
}
impl<'data, S: RealScalar> KdTree<'data, S> {
    /// Builds a tree over `points` for the given metric.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a Minkowski exponent is below one or not finite,
    ///   the leaf size is zero, the periodic box has the wrong length or a
    ///   side that is not finite and positive, or a point lies outside the
    ///   periodic box.
    /// - `SizeOverflow` or `AllocationFailure` if the tree cannot be stored.
    pub fn with_options(
        points: &PointSet<'data, S>,
        metric: TreeMetric,
        options: KdTreeOptions<'_, S>,
    ) -> Result<Self, SciOxideError> {
        const PERIODIC_BOX: scioxide_core::Argument =
            scioxide_core::argument!("options.periodic_box");
        let p = metric.power()?;
        if options.leaf_size == 0 {
            return Err(invalid(
                scioxide_core::argument!("options.leaf_size"),
                "leaf size must be positive",
            ));
        }
        let periodic = if let Some(lengths) = options.periodic_box {
            if lengths.len() != points.dimension() {
                return Err(invalid(PERIODIC_BOX, "periodic box dimension mismatch"));
            }
            let mut owned = try_filled(
                lengths.len(),
                0.0,
                scioxide_core::quantity!("periodic_lengths"),
            )?;
            for (axis, (&length, out)) in lengths.iter().zip(&mut owned).enumerate() {
                if !length.is_finite() || length <= S::ZERO {
                    return Err(invalid(
                        PERIODIC_BOX,
                        "box lengths must be finite and positive",
                    ));
                }
                *out = length.into_f64();
                for row in 0..points.len() {
                    let x = points.coordinate(row, axis)?;
                    if x < S::ZERO || x >= length {
                        return Err(invalid(
                            scioxide_core::argument!("points"),
                            "indexed periodic coordinates must lie in [0, length)",
                        ));
                    }
                }
            }
            Some(owned)
        } else {
            None
        };
        // Every leaf is nonempty, so a binary tree over n points has fewer
        // than 2n nodes. With exact reservations, the recursive build never
        // reallocates and allocation failure surfaces before any work.
        let n = points.len();
        let max_nodes = n
            .checked_mul(2)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("tree_nodes")))?;
        let max_bounds = if n == 0 { 0 } else { points.dimension() };
        let mut entries = try_filled(
            n,
            Entry {
                index: 0,
                split: 0.0,
            },
            scioxide_core::quantity!("tree_entries"),
        )?;
        for (i, entry) in entries.iter_mut().enumerate() {
            entry.index = i;
        }
        let mut nodes = Vec::new();
        nodes.try_reserve_exact(max_nodes).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("tree_nodes"), max_nodes)
        })?;
        let mut bounds = Vec::new();
        bounds.try_reserve_exact(max_bounds).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("tree_bounds"), max_bounds)
        })?;
        let mut tree = Self {
            points: points.clone(),
            entries,
            nodes,
            bounds,
            periodic,
            p,
            pruning_factors: core::array::from_fn(|depth| {
                1.0 - (64.0
                    * (points.dimension() as f64 + depth as f64 + 1.0)
                    * S::EPSILON.into_f64())
                .min(1.0)
            }),
            leaf_size: options.leaf_size,
        };
        if n > 0 {
            if points.contiguous() {
                tree.build::<true>(0, n, options.leaf_size);
            } else {
                tree.build::<false>(0, n, options.leaf_size);
            }
        }
        Ok(tree)
    }
    /// Returns the number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }
    /// Returns the indexed points.
    pub const fn points(&self) -> &PointSet<'data, S> {
        &self.points
    }
    /// Returns whether the tree has no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
    /// Returns the coordinate dimension.
    pub fn dimension(&self) -> usize {
        self.points.dimension()
    }
    // Coincident with the query on every axis. An underflowed square of a
    // nonzero difference is not coincident and keeps the scaled evaluation.
    #[cold]
    #[inline(never)]
    fn coincident<const CONTIGUOUS: bool, const PERIODIC: bool>(
        axes: &[Axis],
        row: crate::distance::Row<'_, S>,
    ) -> bool {
        axes.iter()
            .enumerate()
            .all(|(i, axis)| axis.point_distance::<PERIODIC>(row.coordinate(i)) == 0.0)
    }
    // Auto planning estimate for one query: a root-to-leaf descent, one leaf
    // and the candidate selection, each touching every coordinate.
    pub(crate) fn query_work(&self, neighbors: usize) -> core::num::NonZeroUsize {
        let depth = usize::BITS - self.len().leading_zeros();
        crate::batch::work(
            self.dimension().max(1).saturating_mul(
                (depth as usize)
                    .saturating_add(self.leaf_size)
                    .saturating_add(neighbors),
            ),
        )
    }
    /// Returns the number of tree nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    /// Returns the heap storage of the tree in bytes, without the borrowed
    /// coordinates.
    pub fn storage_bytes(&self) -> usize {
        self.entries.capacity() * core::mem::size_of::<Entry>()
            + self.nodes.capacity() * core::mem::size_of::<Node>()
            + self.bounds.capacity() * core::mem::size_of::<(f64, f64)>()
            + self
                .periodic
                .as_ref()
                .map_or(0, |p| p.capacity() * core::mem::size_of::<f64>())
    }
    // Compact bounds choose the split during construction; only the root
    // rectangle is retained. Search inherits child rectangles along split axes.
    fn build<const CONTIGUOUS: bool>(
        &mut self,
        start: usize,
        end: usize,
        leaf_size: usize,
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            start,
            end,
            children: None,
        });
        let mut widest = 0;
        let mut spread = -1.0;
        for axis in 0..self.dimension() {
            let mut low = f64::INFINITY;
            let mut high = f64::NEG_INFINITY;
            for entry in &self.entries[start..end] {
                let x = self.points.row::<CONTIGUOUS>(entry.index).coordinate(axis);
                low = low.min(x);
                high = high.max(x);
            }
            if id == 0 {
                self.bounds.push((low, high));
            }
            // Halving before subtraction avoids overflow for opposite extremes,
            // but halving adjacent subnormals can round them equal, so finite
            // differences are taken directly.
            let width = if (high - low).is_finite() {
                high - low
            } else {
                high * 0.5 - low * 0.5
            };
            if width > spread {
                spread = width;
                widest = axis;
            }
        }
        // Split on the widest axis at the median, which bounds depth by
        // log2(n). Coincident points (zero spread) stay in one oversized
        // leaf, because no split could separate them. The (value, index)
        // order makes the unstable selection deterministic.
        if end - start > leaf_size && self.dimension() > 0 && spread > 0.0 {
            for entry in &mut self.entries[start..end] {
                entry.split = self
                    .points
                    .row::<CONTIGUOUS>(entry.index)
                    .coordinate(widest);
            }
            let middle = start + (end - start) / 2;
            self.entries[start..end].select_nth_unstable_by(middle - start, |a, b| {
                a.split
                    .total_cmp(&b.split)
                    .then_with(|| a.index.cmp(&b.index))
            });
            let left_high = self.entries[start..middle]
                .iter()
                .map(|e| e.split)
                .fold(f64::NEG_INFINITY, f64::max);
            let right_low = self.entries[middle..end]
                .iter()
                .map(|e| e.split)
                .fold(f64::INFINITY, f64::min);
            let left = self.build::<CONTIGUOUS>(start, middle, leaf_size);
            let right = self.build::<CONTIGUOUS>(middle, end, leaf_size);
            self.nodes[id].children = Some(Split {
                axis: widest,
                left,
                right,
                left_high,
                right_low,
            });
        }
        id
    }
    // Validate query shape and finiteness before touching staging.
    fn preflight(&self, query: &[S], options: SearchOptions) -> Result<f64, SciOxideError> {
        if query.len() != self.dimension() {
            return Err(SciOxideError::dimension_mismatch(
                [self.dimension()],
                [query.len()],
            ));
        }
        if query.iter().any(|x| !x.is_finite()) {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "query coordinates must be finite",
            ));
        }
        options.factor()
    }
    fn query_coordinate(&self, query: &[S], axis: usize) -> f64 {
        let x = query[axis].into_f64();
        if let Some(lengths) = &self.periodic {
            // Wrap into [0, L). For a tiny negative remainder, r + L can round
            // up to exactly L, which lies outside the half-open box; it is
            // mapped to the seam value 0 instead.
            let r = x % lengths[axis];
            if r < 0.0 {
                let wrapped = r + lengths[axis];
                if wrapped == lengths[axis] {
                    0.0
                } else {
                    wrapped
                }
            } else {
                r
            }
        } else {
            x
        }
    }
    /// Finds the `k` points nearest to `query`, writes them to `output` sorted
    /// by distance, and returns how many were written, `min(k, self.len())`.
    ///
    /// With [`SearchOptions::with_epsilon`] set to `ε > 0`, the `i`-th result is
    /// within a factor `1 + ε` of the true `i`-th nearest distance. `output`
    /// needs `min(k, self.len())` entries, and `workspace` must meet a
    /// [`SearchWorkspaceRequest`] for that many neighbors. The query does not
    /// allocate.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `query` has the wrong length, or `output` or
    ///   `workspace` is too small.
    /// - `NonFinite` if `query` contains NaN or infinity.
    /// - `InvalidInput` if `ε` is negative or not finite.
    /// - `NumericalInstability` if fewer than `min(k, self.len())` distances are
    ///   representable.
    /// - `ExecutionStopped` on cancellation or deadline.
    #[inline]
    pub fn nearest_into<'scope, 'capability>(
        &self,
        query: &[S],
        k: usize,
        options: SearchOptions,
        output: &mut [Neighbor<S>],
        workspace: &mut SearchWorkspace<S>,
        context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
    ) -> Result<usize, SciOxideError> {
        let factor = self.preflight(query, options)?;
        let count = k.min(self.len());
        require(output.len(), count)?;
        workspace
            .require(&self.query_requirements(SearchWorkspaceRequest::new(count, options))?)?;
        let context = context.into();
        let execution = context.execution();
        execution.check()?;
        let worker = &mut workspace.workers[0];
        let mut results = SearchState::new(&mut worker.axes, &mut worker.candidates[..count]);
        if count > 0 {
            self.search(
                query,
                Query::Nearest { count, factor },
                &mut results,
                &mut SearchPoll::new(execution),
            )?;
        }
        results.values[..results.len].sort_unstable_by(order);
        execution.check()?;
        output[..count].copy_from_slice(&results.values[..count]);
        Ok(count)
    }
    /// Finds the points within distance `radius` of `query`, inclusive, writes
    /// them to `output` sorted by distance, and returns how many were written.
    ///
    /// With [`SearchOptions::with_epsilon`] set to `ε > 0`, every point within
    /// `radius / (1 + ε)` is returned, no returned point is farther than
    /// `radius`, and points in between may be omitted. Size `output` and the
    /// workspace for the number of matches you expect, not for all points.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `query` has the wrong length.
    /// - `NonFinite` if `query` contains NaN or infinity.
    /// - `InvalidInput` if `radius` is negative or not finite, or `ε` is invalid.
    /// - `ResourceLimit` if there are more matches than `output` or the
    ///   workspace holds.
    /// - `ExecutionStopped` on cancellation or deadline.
    #[inline]
    pub fn radius_into<'scope, 'capability>(
        &self,
        query: &[S],
        radius: S,
        options: SearchOptions,
        output: &mut [Neighbor<S>],
        workspace: &mut SearchWorkspace<S>,
        context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
    ) -> Result<usize, SciOxideError> {
        let factor = self.preflight(query, options)?;
        validate_radius(radius)?;
        workspace.require(&self.query_requirements(SearchWorkspaceRequest::new(0, options))?)?;
        let context = context.into();
        let execution = context.execution();
        execution.check()?;
        let worker = &mut workspace.workers[0];
        let capacity = output.len().min(worker.candidates.len());
        let mut results = SearchState::new(&mut worker.axes, &mut worker.candidates[..capacity]);
        if !self.is_empty() {
            self.search(
                query,
                Query::Radius {
                    radius: radius.into_f64(),
                    factor,
                },
                &mut results,
                &mut SearchPoll::new(execution),
            )?;
        }
        results.values[..results.len].sort_unstable_by(order);
        execution.check()?;
        output[..results.len].copy_from_slice(&results.values[..results.len]);
        Ok(results.len)
    }
    /// Returns the `min(k, self.len())` points nearest to `query`, as
    /// [`Self::nearest_into`] does.
    ///
    /// Runs on the calling thread and allocates the result and a workspace.
    ///
    /// # Errors
    ///
    /// As [`Self::nearest_into`], and `AllocationFailure`.
    pub fn nearest(&self, query: &[S], k: usize) -> Result<Vec<Neighbor<S>>, SciOxideError> {
        let options = SearchOptions::new();
        let count = k.min(self.len());
        let mut out = try_filled(
            count,
            Neighbor::default(),
            scioxide_core::quantity!("neighbor_output"),
        )?;
        let mut workspace = self.workspace(SearchWorkspaceRequest::new(count, options))?;
        self.nearest_into(
            query,
            k,
            options,
            &mut out,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(out)
    }
    /// Returns the points within distance `radius` of `query`, as
    /// [`Self::radius_into`] does.
    ///
    /// Runs on the calling thread. It counts the matches first, then allocates
    /// storage for exactly that many.
    ///
    /// # Errors
    ///
    /// As [`Self::radius_into`], and `AllocationFailure`.
    pub fn radius(&self, query: &[S], radius: S) -> Result<Vec<Neighbor<S>>, SciOxideError> {
        let options = SearchOptions::new();
        let factor = self.preflight(query, options)?;
        validate_radius(radius)?;
        let mut workspace = self.workspace(SearchWorkspaceRequest::new(0, options))?;
        let mut results = SearchState::count(&mut workspace.workers[0].axes);
        if !self.is_empty() {
            self.search(
                query,
                Query::Radius {
                    radius: radius.into_f64(),
                    factor,
                },
                &mut results,
                &mut SearchPoll::new(ExecutionContext::sequential_native()),
            )?;
        }
        let count = results.len;
        workspace.try_reserve(
            &self.workspace_requirements(SearchWorkspaceRequest::new(count, options))?,
        )?;
        let mut out = try_filled(
            count,
            Neighbor::default(),
            scioxide_core::quantity!("neighbor_output"),
        )?;
        self.radius_into(
            query,
            radius,
            options,
            &mut out,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(out)
    }
    #[inline]
    pub(crate) fn search(
        &self,
        query: &[S],
        kind: Query,
        results: &mut SearchState<'_, S>,
        poll: &mut SearchPoll<'_>,
    ) -> Result<(), SciOxideError> {
        results.update_cutoffs(kind);
        for (axis, state) in results.axes[..self.dimension()].iter_mut().enumerate() {
            let q = self.query_coordinate(query, axis);
            let length = self.periodic.as_ref().map_or(0.0, |v| v[axis]);
            let (low, high) = self.bounds[axis];
            *state = Axis {
                query: q,
                seam: length - q,
                length,
                low,
                high,
                delta: 0.0,
            };
        }
        macro_rules! run {
            ($kind:literal) => {
                match (self.points.contiguous(), self.periodic.is_some()) {
                    (true, false) => self.start::<$kind, true, false>(kind, results, poll),
                    (true, true) => self.start::<$kind, true, true>(kind, results, poll),
                    (false, false) => self.start::<$kind, false, false>(kind, results, poll),
                    (false, true) => self.start::<$kind, false, true>(kind, results, poll),
                }
            };
        }
        if self.p == 2.0 {
            run!(2)
        } else if self.p == 1.0 {
            run!(1)
        } else if self.p == f64::INFINITY {
            run!(0)
        } else {
            run!(3)
        }?;
        results.finish(kind)
    }
    fn start<const KIND: u8, const CONTIGUOUS: bool, const PERIODIC: bool>(
        &self,
        kind: Query,
        results: &mut SearchState<'_, S>,
        poll: &mut SearchPoll<'_>,
    ) -> Result<(), SciOxideError> {
        let mut bound = Bound::zero();
        for axis in &mut results.axes[..self.dimension()] {
            axis.delta = axis.interval_distance::<PERIODIC>();
            bound = bound.increase::<KIND>(0.0, axis.delta, self.p);
        }
        self.visit::<KIND, CONTIGUOUS, PERIODIC>(
            0,
            bound,
            bound.distance::<KIND>(self.p),
            0,
            kind,
            results,
            poll,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn visit<const KIND: u8, const CONTIGUOUS: bool, const PERIODIC: bool>(
        &self,
        id: usize,
        bound: Bound,
        distance: f64,
        depth: usize,
        kind: Query,
        results: &mut SearchState<'_, S>,
        poll: &mut SearchPoll<'_>,
    ) -> Result<(), SciOxideError> {
        poll.visit()?;
        // Include accumulated update/reduction, output rounding, and
        // destination-subnormal quantization error.
        // Ambiguous boundaries are visited, preserving inclusive radius and ties.
        // An overflowed bound has lost its rounding information. Visit it
        // conservatively instead of rejecting a potentially representable result.
        let lower = if distance.is_finite() {
            (distance * self.pruning_factors[depth] - 2.0 * S::MIN_SUBNORMAL.into_f64()).max(0.0)
        } else {
            0.0
        };
        if lower > results.pruning_cutoff {
            return Ok(());
        }
        let node = self.nodes[id];
        if let Some(split) = node.children {
            let parent = results.axes[split.axis];
            let mut left = parent;
            left.high = split.left_high;
            left.delta = left.interval_distance::<PERIODIC>();
            let mut right = parent;
            right.low = split.right_low;
            right.delta = right.interval_distance::<PERIODIC>();
            let child_bound = |delta| {
                if delta == parent.delta {
                    (bound, distance)
                } else {
                    let updated = bound.increase::<KIND>(parent.delta, delta, self.p);
                    (updated, updated.distance::<KIND>(self.p))
                }
            };
            let (left_bound, left_distance) = child_bound(left.delta);
            let (right_bound, right_distance) = child_bound(right.delta);
            let children = if left_distance <= right_distance {
                [
                    (split.left, left, left_bound, left_distance),
                    (split.right, right, right_bound, right_distance),
                ]
            } else {
                [
                    (split.right, right, right_bound, right_distance),
                    (split.left, left, left_bound, left_distance),
                ]
            };
            for (child, axis, child_bound, child_distance) in children {
                results.axes[split.axis] = axis;
                self.visit::<KIND, CONTIGUOUS, PERIODIC>(
                    child,
                    child_bound,
                    child_distance,
                    depth + 1,
                    kind,
                    results,
                    poll,
                )?;
            }
            results.axes[split.axis] = parent;
        } else {
            for (chunk, entries) in self.entries[node.start..node.end]
                .chunks(LEAF_POLL_INTERVAL)
                .enumerate()
            {
                if chunk > 0 {
                    poll.check()?;
                }
                for entry in entries {
                    #[cfg(test)]
                    {
                        poll.points += 1;
                    }
                    let axes = &results.axes[..self.dimension()];
                    let row = self.points.row::<CONTIGUOUS>(entry.index);
                    let mut sum: f64 = 0.0;
                    for (i, axis) in axes.iter().enumerate() {
                        let delta = axis.point_distance::<PERIODIC>(row.coordinate(i));
                        if KIND == 0 || KIND == 3 {
                            sum = sum.max(delta);
                        } else if KIND == 1 {
                            sum += delta;
                        } else {
                            sum += delta * delta;
                        }
                    }
                    // For p >= 1 the scaled p-norm is at least its largest
                    // component: the scaled sum is at least one, so neither its
                    // root nor the final product rounds below that component.
                    if KIND == 3 && sum > results.candidate_cutoff {
                        continue;
                    }
                    let distance = if KIND == 0 || KIND == 1 {
                        sum
                    } else if KIND == 2 && sum.is_finite() && sum >= f64::MIN_POSITIVE {
                        if sum > results.squared_candidate_cutoff {
                            continue;
                        }
                        RealScalar::sqrt(sum)
                    } else if KIND == 2
                        && sum == 0.0
                        && Self::coincident::<CONTIGUOUS, PERIODIC>(axes, row)
                    {
                        0.0
                    } else {
                        // General p and extreme square ranges share the distance
                        // kernel's allocation-free, two-pass scaled evaluation.
                        match norm(axes.len(), self.p, |i| {
                            Ok(axes[i].point_distance::<PERIODIC>(row.coordinate(i)))
                        }) {
                            Ok(distance) => distance,
                            Err(error) => {
                                results.range_error.get_or_insert(error);
                                continue;
                            }
                        }
                    };
                    // Admit rounded-distance ties, but do not convert candidates
                    // proved to lose, including overflowed distances.
                    if distance > results.candidate_cutoff {
                        continue;
                    }
                    let distance = match finite(distance).and_then(S::try_from_f64_lossy) {
                        Ok(distance) => distance,
                        Err(error) => {
                            // All range failures exceed every representable
                            // distance. A later candidate may still fill the heap.
                            results.range_error.get_or_insert(error);
                            continue;
                        }
                    };
                    let candidate = Neighbor {
                        index: entry.index,
                        distance,
                    };
                    match kind {
                        Query::Nearest { count, .. } => {
                            if heap_insert(results.values, &mut results.len, candidate, count) {
                                results.update_cutoffs(kind);
                            }
                        }
                        Query::Radius { radius, .. } => {
                            if candidate.distance.into_f64() <= radius {
                                results.push(candidate)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
#[derive(Clone, Copy)]
pub(crate) enum Query {
    Nearest { count: usize, factor: f64 },
    Radius { radius: f64, factor: f64 },
}
// Bounded binary max-heap under the total (distance, index) order. The root
// is the worst retained neighbor. A candidate replaces it only when strictly
// better, so ties at the k-th distance keep the lowest indices, independent
// of visit order. The caller's capacity check guarantees `push` never grows.
#[inline]
fn heap_insert<S: RealScalar>(
    heap: &mut [Neighbor<S>],
    len: &mut usize,
    candidate: Neighbor<S>,
    count: usize,
) -> bool {
    if *len < count {
        heap[*len] = candidate;
        *len += 1;
        let mut child = *len - 1;
        while child > 0 {
            let parent = (child - 1) / 2;
            if order(&heap[parent], &heap[child]) != Ordering::Less {
                break;
            }
            heap.swap(parent, child);
            child = parent;
        }
        true
    } else if order(&candidate, &heap[0]) == Ordering::Less {
        heap[0] = candidate;
        let mut parent = 0;
        loop {
            let left = parent * 2 + 1;
            if left >= *len {
                break;
            }
            let right = left + 1;
            let child = if right < *len && order(&heap[left], &heap[right]) == Ordering::Less {
                right
            } else {
                left
            };
            if order(&heap[parent], &heap[child]) != Ordering::Less {
                break;
            }
            heap.swap(parent, child);
            parent = child;
        }
        true
    } else {
        false
    }
}

pub(crate) fn validate_radius<S: RealScalar>(radius: S) -> Result<(), SciOxideError> {
    if !radius.is_finite() || radius < S::ZERO {
        return Err(invalid(
            scioxide_core::argument!("radius"),
            "radius must be finite and nonnegative",
        ));
    }
    Ok(())
}

pub(crate) struct SearchState<'a, S: RealScalar> {
    axes: &'a mut [Axis],
    pub(crate) values: &'a mut [Neighbor<S>],
    pub(crate) len: usize,
    count_only: bool,
    pruning_cutoff: f64,
    candidate_cutoff: f64,
    squared_candidate_cutoff: f64,
    range_error: Option<SciOxideError>,
}
impl<'a, S: RealScalar> SearchState<'a, S> {
    pub(crate) fn new(axes: &'a mut [Axis], values: &'a mut [Neighbor<S>]) -> Self {
        Self {
            axes,
            values,
            len: 0,
            count_only: false,
            pruning_cutoff: f64::INFINITY,
            candidate_cutoff: f64::INFINITY,
            squared_candidate_cutoff: f64::INFINITY,
            range_error: None,
        }
    }
    pub(crate) fn count(axes: &'a mut [Axis]) -> Self {
        Self {
            count_only: true,
            ..Self::new(axes, &mut [])
        }
    }
    fn finish(&mut self, kind: Query) -> Result<(), SciOxideError> {
        if matches!(kind, Query::Nearest { count, .. } if self.len < count)
            && let Some(error) = self.range_error.take()
        {
            return Err(error);
        }
        Ok(())
    }
    #[inline]
    fn update_cutoffs(&mut self, kind: Query) {
        let (cutoff, factor) = match kind {
            Query::Nearest { count, factor } => (
                if self.len < count {
                    f64::INFINITY
                } else {
                    self.values[0].distance.into_f64()
                },
                factor,
            ),
            Query::Radius { radius, factor } => (radius, factor),
        };
        self.pruning_cutoff = cutoff / factor;
        // A full relative epsilon plus destination-subnormal slack admits all
        // values rounding to cutoff. Overflow merely disables early rejection.
        self.candidate_cutoff =
            cutoff + cutoff * S::EPSILON.into_f64() + 2.0 * S::MIN_SUBNORMAL.into_f64();
        // The next root covers sqrt rounding down to candidate_cutoff; the
        // outward-rounded square covers multiplication rounding and underflow.
        // Overflow yields infinity and disables comparison-domain rejection.
        let root = self.candidate_cutoff.next_up();
        self.squared_candidate_cutoff = (root * root).next_up();
    }
    fn push(&mut self, candidate: Neighbor<S>) -> Result<(), SciOxideError> {
        // Every indexed point is visited at most once; construction checked
        // 2 * tree.len(), so this result count cannot overflow usize.
        if !self.count_only {
            let capacity = self.values.len();
            // Only radius staging can fill; nearest selection replaces in place.
            let slot = self.values.get_mut(self.len).ok_or_else(|| {
                SciOxideError::resource_limit(
                    scioxide_core::quantity!("radius_neighbors"),
                    self.len + 1,
                    capacity,
                )
            })?;
            *slot = candidate;
        }
        self.len += 1;
        Ok(())
    }
}

impl<'data, S: RealScalar> KdTree<'data, S> {
    /// Builds a Euclidean KD tree with leaf size 16 and no periodic box.
    /// Uses [`Self::with_options`] with its construction cost and storage.
    ///
    /// # Errors
    /// As [`Self::with_options`].
    ///
    /// ```
    /// use scioxide_spatial::{KdTree, PointSet};
    /// let sites = [[0.0_f64, 0.0], [3.0, 4.0]];
    /// let tree = KdTree::new(&PointSet::from_points(&sites)?)?;
    /// let nearest = tree.nearest(&[2.9, 4.0], 1)?;
    /// assert!((nearest[0].distance - 0.1).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn new(points: &PointSet<'data, S>) -> Result<Self, SciOxideError> {
        Self::with_options(points, TreeMetric::Euclidean, KdTreeOptions::new())
    }
    /// Returns the nearest `min(k, self.len())` neighbors for each query row.
    /// Rows are concatenated in query order, each sorted as [`Self::nearest_into`].
    /// Uses default options, allocating output and a workspace for sequential native execution.
    ///
    /// # Errors
    /// As [`Self::nearest_batch_into`], plus `SizeOverflow` or `AllocationFailure` for storage.
    pub fn nearest_batch(
        &self,
        queries: &PointSet<'_, S>,
        k: usize,
    ) -> Result<Vec<Neighbor<S>>, SciOxideError> {
        let options = SearchOptions::new();
        let count = k.min(self.len());
        let total = scioxide_core::backend::checked_mul(
            queries.len(),
            count,
            scioxide_core::quantity!("neighbor_output"),
        )?;
        let mut workspace =
            self.workspace(SearchWorkspaceRequest::new(count, options).with_batch(
                queries.len(),
                total,
                core::num::NonZeroUsize::MIN,
            ))?;
        let mut output = try_filled(
            total,
            Neighbor::default(),
            scioxide_core::quantity!("neighbor_output"),
        )?;
        self.nearest_batch_into(
            queries,
            k,
            options,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Writes the nearest neighbors of each query in consecutive sorted rows.
    /// Returns `queries.len() * min(k, self.len())`; uses the caller's batch workspace without allocation.
    /// Cost is the sum of individual tree searches, with identical accuracy.
    /// The workspace request is `SearchWorkspaceRequest::new(c, options)` with
    /// `.with_batch(q, q * c, workers)`, where `q = queries.len()` and
    /// `c = min(k, self.len())`. Reserve for the maximum executor worker count.
    /// Only the returned prefix is written; output is unchanged on failure.
    ///
    /// # Errors
    /// - `DimensionMismatch` for query dimensions or insufficient output/workspace.
    /// - `InvalidInput` for negative or non-finite approximation epsilon.
    /// - `ResourceLimit` for workspace limits.
    /// - `NumericalInstability` as in [`Self::nearest_into`].
    /// - `ExecutionStopped` on cancellation or deadline, and errors of the executor.
    #[allow(clippy::too_many_arguments)]
    pub fn nearest_batch_into<'scope, 'capability>(
        &self,
        queries: &PointSet<'_, S>,
        k: usize,
        options: SearchOptions,
        output: &mut [Neighbor<S>],
        workspace: &mut SearchWorkspace<S>,
        context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
    ) -> Result<usize, SciOxideError> {
        crate::batch::query_batch(
            self,
            queries,
            crate::batch::QueryKind::Nearest { k },
            options,
            output,
            None,
            workspace,
            context,
        )
    }
    /// Writes all neighbors within `radius` of each query, with per-query counts.
    /// Concatenates sorted rows; uses the caller's workspace without allocation and returns the total.
    /// Performs a counting pass followed by searches with the accuracy of [`Self::radius_into`].
    /// Reserve `SearchWorkspaceRequest::new(0, options).with_batch(q, total, workers)`,
    /// where `q = queries.len()` and `total` bounds the combined matches. Each row
    /// is sorted as in [`Self::radius_into`]; unused output and count suffixes stay unchanged.
    ///
    /// # Errors
    /// As [`Self::nearest_batch_into`], plus `InvalidInput` for an invalid radius and
    /// `DimensionMismatch` for insufficient count storage. Outputs are unchanged on error.
    #[allow(clippy::too_many_arguments)]
    pub fn radius_batch_into<'scope, 'capability>(
        &self,
        queries: &PointSet<'_, S>,
        radius: S,
        options: SearchOptions,
        output: &mut [Neighbor<S>],
        counts: &mut [usize],
        workspace: &mut SearchWorkspace<S>,
        context: impl Into<SpatialExecutionContext<'scope, 'capability>>,
    ) -> Result<usize, SciOxideError> {
        crate::batch::query_batch(
            self,
            queries,
            crate::batch::QueryKind::Radius { radius },
            options,
            output,
            Some(counts),
            workspace,
            context,
        )
    }
}
