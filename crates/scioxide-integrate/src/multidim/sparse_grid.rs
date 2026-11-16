//! Dimension-adaptive sparse-grid quadrature with nested Clenshaw–Curtis rules.

const ALGORITHM: scioxide_core::AlgorithmId =
    scioxide_core::algorithm_id!("integrate", "SparseGridPlan", 1);

use alloc::vec::Vec;

use scioxide_core::backend::{checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AffineValue, AffineValueMut, Budget, ErrorTolerance, ExecutionContext, ExecutionPolicy,
    Interval, PartialResultTrust, RealScalar, SciOxideError, SolveContext, SolveScope,
    check_value_shape, write_value,
};
use scioxide_core::{Scaled, ScaledAccumulator};

use super::sampling::{BatchSampler, ValueSampler};
use crate::kernel::AffineMap1D;
use crate::workspace::admit;
use crate::{
    ClenshawCurtis, ErrorEstimate, ErrorReliability, FixedRule, IntegrationErrorComponents,
    IntegrationReport, IntegrationTermination,
};

const EMPTY: usize = usize::MAX;

/// Options of [`SparseGridPlan`]: tolerance, level, index and evaluation
/// limits, and execution policy.
///
/// | option | default |
/// |---|---|
/// | [`Self::tolerance`] | `√ε` absolute and relative, root-mean-square norm |
/// | [`Self::max_level`] | 4 |
/// | [`Self::max_indices`] | 64 |
/// | [`Self::max_evaluations`] | 4096 |
/// | [`Self::execution_policy`] | [`ExecutionPolicy::Sequential`] |
///
/// Here `ε` is the machine epsilon of `S`. The limits also size the workspace.
///
/// # Examples
///
/// ```
/// use scioxide_integrate::SparseGridOptions;
///
/// let options = SparseGridOptions::<f64>::default().with_max_level(3);
/// assert_eq!(options.max_level(), 3);
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct SparseGridOptions<S: RealScalar> {
    tolerance: ErrorTolerance<S>,
    max_level: usize,
    max_indices: Budget,
    max_evaluations: Budget,
    policy: ExecutionPolicy,
}

impl<S: RealScalar> SparseGridOptions<S> {
    /// Replaces the tolerance.
    pub fn with_tolerance(mut self, value: ErrorTolerance<S>) -> Self {
        self.tolerance = value;
        self
    }
    /// Sets the highest level of the one-dimensional rule on any axis.
    pub const fn with_max_level(mut self, value: usize) -> Self {
        self.max_level = value;
        self
    }
    /// Sets the largest number of multi-indices, accepted and active, per call.
    pub const fn with_max_indices(mut self, value: Budget) -> Self {
        self.max_indices = value;
        self
    }
    /// Sets the largest number of distinct nodes evaluated per call.
    pub const fn with_max_evaluations(mut self, value: Budget) -> Self {
        self.max_evaluations = value;
        self
    }
    fn validate(&self) -> Result<(), SciOxideError> {
        if self.max_level > 52 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("max_level"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "nested Clenshaw-Curtis level exceeds dyadic key precision",
            ));
        }
        let nodes = if self.max_level == 0 {
            1
        } else {
            1_usize
                .checked_shl(self.max_level as u32)
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("rule_nodes"))
                })?
        };
        if nodes > self.max_evaluations.get() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("max_evaluations"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "maximum one-dimensional rule exceeds evaluation budget",
            ));
        }
        Ok(())
    }
    /// Sets how the integrand evaluations of one call are distributed over
    /// workers. The result does not depend on the policy.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Returns the execution policy of integrand evaluations.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }
    /// Returns the tolerance.
    pub const fn tolerance(&self) -> &ErrorTolerance<S> {
        &self.tolerance
    }
    /// Returns the highest level of the one-dimensional rule on any axis.
    pub const fn max_level(&self) -> usize {
        self.max_level
    }
    /// Returns the largest number of multi-indices, accepted and active, per
    /// call.
    pub const fn max_indices(&self) -> Budget {
        self.max_indices
    }
    /// Returns the largest number of distinct nodes evaluated per call.
    pub const fn max_evaluations(&self) -> Budget {
        self.max_evaluations
    }
}

#[derive(Debug, Clone)]
struct RuleLevel<S: RealScalar> {
    keys: Vec<u64>,
    nodes: Vec<S>,
    differences: Vec<S>,
    full_weights: Vec<S>,
}

// Canonical identity of a nested Clenshaw-Curtis node. Node `index` of level `level`
// (2^level + 1 points) is also node `index / 2` of the previous level when `index` is
// even, so factors of two are stripped until the node's first level is reached; the
// level-0 midpoint is the centre of level 1. Equal keys therefore mean the same abscissa
// across levels, which lets the cache deduplicate points without comparing floats.
// Keys pack the reduced level above bit 56 and the odd index below it.
fn node_key(level: usize, index: usize) -> u64 {
    let (mut level, mut index) = if level == 0 { (1, 1) } else { (level, index) };
    while level > 0 && index.is_multiple_of(2) {
        level -= 1;
        index /= 2;
    }
    ((level as u64) << 56) | index as u64
}

/// Dimension-adaptive sparse-grid quadrature over a box.
///
/// The one-dimensional rules are nested: `Q₀` is the midpoint rule and `Q_l`
/// for `l ≥ 1` is the Clenshaw–Curtis rule with `2^l + 1` nodes. With the
/// difference rules `Δ_l = Q_l − Q_{l−1}` (`Q_{−1} = 0`), a multi-index
/// `k = (k₁, …, k_d)` has the surplus `Δ_k f = (Δ_{k₁} ⊗ … ⊗ Δ_{k_d}) f`, and
/// the estimate is
///
/// ```text
/// I ≈ Σ_{k ∈ A} Δ_k f,
/// ```
///
/// where `A` is the set of accepted indices. The plan starts from
/// `k = (0, …, 0)`. At each step it accepts the active index with the largest
/// surplus relative to the tolerance and makes active each forward neighbour
/// `k + e_j` whose backward neighbours are all accepted. Each distinct node is
/// evaluated once per call, so the grid grows only along the axes with large
/// surpluses.
///
/// # Error estimates
///
/// The error estimate is `4 Σ |Δ_k f|` over the active indices and the
/// accepted indices with unexplored forward neighbours. It has
/// [`crate::ErrorReliability::Heuristic`]: a feature that the sampled surpluses
/// miss is not detected. The plan reports
/// [`IntegrationTermination::HeuristicToleranceReached`] when the estimate
/// meets the tolerance and the first refinement of every axis is accepted. See
/// [Error estimates](crate#error-estimates).
///
/// # Termination
///
/// After the first estimate, reaching the evaluation limit returns `Ok` with
/// [`IntegrationTermination::BudgetLimit`]. Reaching the index limit, or
/// running out of indices below the level limit, returns `Ok` with
/// [`IntegrationTermination::RefinementLimit`]. Each of these results carries
/// the error estimate. A box with a zero-width axis gives zero with
/// [`IntegrationTermination::DegenerateDomain`]. See
/// [Termination](crate#termination).
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_integrate::{SparseGridOptions, SparseGridPlan};
///
/// let plan = SparseGridPlan::<f64>::new(2, SparseGridOptions::default())?;
/// let square = [Interval::new(0.0, 1.0)?; 2];
/// let f = |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]);
/// let (value, report) = plan.integrate(&f, &square)?;
/// assert!(report.is_success());
/// // ∫∫ (x² + y²) dx dy = 2/3 over the unit square.
/// assert!((value - 2.0 / 3.0).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - S. A. Smolyak, "Quadrature and interpolation formulas for tensor products
///   of certain classes of functions", *Soviet Math. Dokl.* 4, 240–243, 1963.
/// - T. Gerstner and M. Griebel, "Dimension-adaptive tensor-product
///   quadrature", *Computing* 71(1), 65–87, 2003.
///   <https://doi.org/10.1007/s00607-003-0015-5>
/// - C. W. Clenshaw and A. R. Curtis, "A method for numerical integration on
///   an automatic computer", *Numer. Math.* 2, 197–205, 1960.
#[derive(Debug, Clone)]
#[must_use]
pub struct SparseGridPlan<S: RealScalar> {
    dimension: usize,
    options: SparseGridOptions<S>,
    levels: Vec<RuleLevel<S>>,
}

impl<S: RealScalar> SparseGridPlan<S> {
    /// Prepares the difference rules up to [`SparseGridOptions::max_level`] for
    /// boxes of `dimension` axes.
    ///
    /// For `dimension = 0` the plan evaluates the integrand once at the empty
    /// point.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if the maximum level exceeds 52, or if
    ///   the rule of the maximum level has more than
    ///   [`SparseGridOptions::max_evaluations`] nodes.
    /// - The errors of [`ClenshawCurtis::new`].
    /// - `AllocationFailure` if the rules cannot be stored.
    pub fn new(dimension: usize, options: SparseGridOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        let count = options
            .max_level
            .checked_add(1)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("rule_levels")))?;
        let mut levels = try_with_capacity(count, scioxide_core::quantity!("rule_levels"))?;
        let mut keys = try_with_capacity(1, scioxide_core::quantity!("rule_keys"))?;
        keys.push(node_key(0, 0));
        let mut nodes = try_with_capacity(1, scioxide_core::quantity!("rule_nodes"))?;
        nodes.push(S::ZERO);
        let mut weights = try_with_capacity(1, scioxide_core::quantity!("rule_weights"))?;
        weights.push(S::TWO);
        let mut differences = try_with_capacity(1, scioxide_core::quantity!("rule_differences"))?;
        differences.push(S::TWO);
        levels.push(RuleLevel {
            keys,
            nodes,
            differences,
            full_weights: weights,
        });
        // Level 0 is the one-point midpoint rule (weight 2 on [-1, 1]); level l >= 1 is the
        // nested Clenshaw-Curtis rule with 2^l + 1 nodes. Smolyak's construction uses the
        // difference rules Delta_l = Q_l - Q_(l-1), so `differences` stores each node's
        // weight minus its weight in the previous level (zero there for new nodes).
        let canonical = Interval::new(-S::ONE, S::ONE)?;
        for level in 1..=options.max_level {
            let degree = 1_usize.checked_shl(level as u32).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("rule_nodes"))
            })?;
            let nodes = degree.checked_add(1).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("rule_nodes"))
            })?;
            let rule = ClenshawCurtis::<S>::new(degree)?;
            let mut keys = try_with_capacity(nodes, scioxide_core::quantity!("rule_keys"))?;
            let mut points = try_with_capacity(nodes, scioxide_core::quantity!("rule_nodes"))?;
            let mut differences =
                try_with_capacity(nodes, scioxide_core::quantity!("rule_differences"))?;
            let mut full_weights =
                try_with_capacity(nodes, scioxide_core::quantity!("rule_weights"))?;
            let mut metadata = scioxide_core::backend::try_filled(
                nodes,
                crate::FixedRuleNode::new(S::ZERO, S::ZERO, None),
                scioxide_core::quantity!("rule_metadata"),
            )?;
            rule.map_nodes_into(canonical, &mut metadata)?;
            for (index, metadata) in metadata.into_iter().enumerate() {
                let previous = if level == 1 {
                    (index == 1).then_some(levels[0].full_weights[0])
                } else if index.is_multiple_of(2) {
                    Some(levels[level - 1].full_weights[index / 2])
                } else {
                    None
                };
                keys.push(node_key(level, index));
                points.push(metadata.point());
                full_weights.push(metadata.primary_weight());
                differences.push(metadata.primary_weight() - previous.unwrap_or(S::ZERO));
            }
            levels.push(RuleLevel {
                keys,
                nodes: points,
                differences,
                full_weights,
            });
        }
        Ok(Self {
            dimension,
            options,
            levels,
        })
    }

    /// Returns the number of axes `d`.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Allocates a workspace for an integrand with `components` real components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the storage size overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn workspace(&self, components: usize) -> Result<SparseGridWorkspace<S>, SciOxideError> {
        SparseGridWorkspace::allocate(&self.workspace_requirements(components)?)
    }
    /// Returns the options.
    pub const fn options(&self) -> &SparseGridOptions<S> {
        &self.options
    }
    /// Returns the identifier of the method, recorded in solve diagnostics.
    pub const fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        ALGORITHM
    }
    /// Returns the workspace size for an integrand with `components` real
    /// components.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if `max_evaluations · components` overflows `usize`.
    pub fn workspace_requirements(
        &self,
        components: usize,
    ) -> Result<crate::IntegrationRequirements, SciOxideError> {
        checked_mul(
            self.options.max_evaluations.get(),
            components,
            scioxide_core::quantity!("sparse_grid_values"),
        )?;
        Ok(crate::IntegrationRequirements::new(
            crate::workspace::Family::SparseGrid,
            components,
            [
                self.options.max_indices.get(),
                self.options.max_evaluations.get(),
            ],
        )
        .with_dimension(self.dimension))
    }
}

/// Workspace of a [`SparseGridPlan`]: the index set and the values at the
/// evaluated nodes.
///
/// A workspace is made for one dimension and one number of value components by
/// [`SparseGridPlan::workspace`]. It holds `O(I (d + c) + E (d + c))` values
/// for `I` indices, `E` evaluations, `d` dimensions and `c` components. Any
/// plan of the same dimension and component count with no larger index and
/// evaluation limits can use it; the limits of the plan apply. Node values are
/// not kept from one call to the next.
pub struct SparseGridWorkspace<S: RealScalar> {
    requirements: crate::IntegrationRequirements,
    attempts: crate::workspace::Attempts<IntegrationReport<S>>,
    component_errors: Vec<IntegrationErrorComponents<S>>,
    published: bool,
    maps: Vec<AffineMap1D<S>>,
    index_levels: Vec<usize>,
    index_state: Vec<u8>,
    index_pending: Vec<usize>,
    index_surplus: Vec<S>,
    index_count: usize,
    counters: Vec<usize>,
    candidate: Vec<usize>,
    key_buffer: Vec<u64>,
    point_buffer: Vec<S>,
    cache_keys: Vec<u64>,
    cache_points: Vec<S>,
    cache_values: Vec<S>,
    slots: Vec<usize>,
    staged_slots: Vec<usize>,
    cache_used: usize,
    value: Vec<S>,
    errors: Vec<S>,
    scales: Vec<S>,
    directional: Vec<S>,
    sums: Vec<ScaledAccumulator<S>>,
    total_sums: Vec<ScaledAccumulator<S>>,
}

impl<S: RealScalar> SparseGridWorkspace<S> {
    /// Allocates storage for `requirements`.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure`.
    fn allocate(requirements: &crate::IntegrationRequirements) -> Result<Self, SciOxideError> {
        let dimension = requirements.dimension();
        let components = requirements.components();
        let [max_indices, max_evaluations] = requirements.units();
        // Open-addressing hash table over cached node ids: a power-of-two size of at least
        // twice the evaluation budget keeps the load factor <= 1/2, so linear probing always
        // finds an empty slot and the mask replaces a modulo.
        let slot_count = max_evaluations
            .checked_mul(2)
            .and_then(|n| n.checked_next_power_of_two())
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("cache_slots")))?;
        Ok(Self {
            requirements: *requirements,
            attempts: crate::workspace::Attempts::default(),
            component_errors: try_filled(
                components,
                IntegrationErrorComponents::default(),
                scioxide_core::quantity!("component_errors"),
            )?,
            published: false,
            maps: try_with_capacity(dimension, scioxide_core::quantity!("axis_maps"))?,
            index_levels: try_filled(
                checked_mul(
                    max_indices,
                    dimension,
                    scioxide_core::quantity!("index_levels"),
                )?,
                0,
                scioxide_core::quantity!("index_levels"),
            )?,
            index_state: try_filled(max_indices, 0, scioxide_core::quantity!("index_state"))?,
            index_pending: try_filled(max_indices, 0, scioxide_core::quantity!("index_pending"))?,
            index_surplus: try_filled(
                checked_mul(
                    max_indices,
                    components,
                    scioxide_core::quantity!("index_surplus"),
                )?,
                S::ZERO,
                scioxide_core::quantity!("index_surplus"),
            )?,
            index_count: 0,
            counters: try_filled(dimension, 0, scioxide_core::quantity!("index_counters"))?,
            candidate: try_filled(dimension, 0, scioxide_core::quantity!("candidate_levels"))?,
            key_buffer: try_filled(dimension, 0, scioxide_core::quantity!("key_buffer"))?,
            point_buffer: try_filled(dimension, S::ZERO, scioxide_core::quantity!("point_buffer"))?,
            cache_keys: try_filled(
                checked_mul(
                    max_evaluations,
                    dimension,
                    scioxide_core::quantity!("cache_keys"),
                )?,
                0,
                scioxide_core::quantity!("cache_keys"),
            )?,
            cache_points: try_filled(
                checked_mul(
                    max_evaluations,
                    dimension,
                    scioxide_core::quantity!("cache_points"),
                )?,
                S::ZERO,
                scioxide_core::quantity!("cache_points"),
            )?,
            cache_values: try_filled(
                checked_mul(
                    max_evaluations,
                    components,
                    scioxide_core::quantity!("cache_values"),
                )?,
                S::ZERO,
                scioxide_core::quantity!("cache_values"),
            )?,
            slots: try_filled(slot_count, EMPTY, scioxide_core::quantity!("cache_slots"))?,
            staged_slots: try_with_capacity(
                max_evaluations,
                scioxide_core::quantity!("staged_slots"),
            )?,
            cache_used: 0,
            value: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("component_values"),
            )?,
            errors: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("component_errors"),
            )?,
            scales: try_filled(
                components,
                S::ZERO,
                scioxide_core::quantity!("component_scales"),
            )?,
            directional: try_filled(
                dimension,
                S::ZERO,
                scioxide_core::quantity!("directional_indicators"),
            )?,
            sums: try_filled(
                components,
                ScaledAccumulator::ZERO,
                scioxide_core::quantity!("surplus_sums"),
            )?,
            total_sums: try_filled(
                components,
                ScaledAccumulator::ZERO,
                scioxide_core::quantity!("total_sums"),
            )?,
        })
    }

    /// Returns, for each axis `j`, the sum of `4 max_c |Δ_k f|` over the
    /// indices of the error estimate with `k_j > 0`, from the last call.
    ///
    /// Large values mark the axes along which the error estimate is largest.
    /// A zero does not mean that no error remains along that axis.
    pub fn directional_indicators(&self) -> &[S] {
        &self.directional
    }

    /// Returns the number of distinct nodes evaluated by the last call.
    pub const fn cached_nodes(&self) -> usize {
        self.cache_used
    }

    fn reset(&mut self) {
        self.index_count = 0;
        self.index_state.fill(0);
        self.cache_used = 0;
        self.slots.fill(EMPTY);
        self.staged_slots.clear();
        self.value.fill(S::ZERO);
        self.errors.fill(S::ZERO);
        self.directional.fill(S::ZERO);
        self.total_sums.fill(ScaledAccumulator::ZERO);
        self.maps.clear();
    }

    fn find_index(&self, levels: &[usize]) -> Option<usize> {
        (0..self.index_count).find(|&i| {
            self.index_levels
                [i * self.requirements.dimension()..(i + 1) * self.requirements.dimension()]
                == *levels
        })
    }

    // `index_state`: 0 = unused, 2 = active (evaluated, its surplus is only an error
    // indicator), 1 = old (accepted into the integral). Indices are never removed.
    fn push_candidate(&mut self, max_indices: usize) -> Option<usize> {
        if self.index_count >= max_indices {
            return None;
        }
        let id = self.index_count;
        self.index_levels
            [id * self.requirements.dimension()..(id + 1) * self.requirements.dimension()]
            .copy_from_slice(&self.candidate);
        self.index_state[id] = 2;
        self.index_count += 1;
        Some(id)
    }

    // Track unresolved forward directions once per acceptance, rather than
    // searching the entire index set during every error reduction.
    fn accept_index(&mut self, id: usize) {
        self.index_state[id] = 1;
        self.index_pending[id] = self.requirements.dimension();
        for axis in 0..self.requirements.dimension() {
            self.candidate.copy_from_slice(
                &self.index_levels
                    [id * self.requirements.dimension()..(id + 1) * self.requirements.dimension()],
            );
            self.candidate[axis] += 1;
            if self.find_index(&self.candidate).is_some() {
                self.index_pending[id] -= 1;
            }
        }
    }

    // FNV-1a over the per-axis node keys, then linear probing. Returns the slot where the
    // key lives (or would be inserted) and the cached node id if present.
    fn cache_slot(&self, keys: &[u64]) -> (usize, Option<usize>) {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for &key in keys {
            hash = (hash ^ key).wrapping_mul(0x100_0000_01b3);
        }
        let mask = self.slots.len() - 1;
        let mut slot = (hash as usize) & mask;
        loop {
            let id = self.slots[slot];
            if id == EMPTY {
                return (slot, None);
            }
            if self.cache_keys
                [id * self.requirements.dimension()..(id + 1) * self.requirements.dimension()]
                == *keys
            {
                return (slot, Some(id));
            }
            slot = (slot + 1) & mask;
        }
    }

    // New cache entries are appended after `prior`, so undoing a refused stage only needs
    // to clear the hash slots it claimed and truncate the used count.
    fn rollback_stage(&mut self, prior: usize) {
        for &slot in &self.staged_slots {
            self.slots[slot] = EMPTY;
        }
        self.staged_slots.clear();
        self.cache_used = prior;
    }
}

struct RunReport {
    termination: IntegrationTermination,
    measured: bool,
    num_evals: usize,
    refinements: usize,
    has_error: bool,
}

fn advance(indices: &mut [usize], plan: &SparseGridPlan<impl RealScalar>, levels: &[usize]) {
    for axis in (0..indices.len()).rev() {
        indices[axis] += 1;
        if indices[axis] < plan.levels[levels[axis]].keys.len() {
            return;
        }
        indices[axis] = 0;
    }
}

impl<S: RealScalar> SparseGridPlan<S> {
    fn evaluate_index<B: BatchSampler<S>>(
        &self,
        id: usize,
        ws: &mut SparseGridWorkspace<S>,
        sampler: &mut B,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        let d = self.dimension;
        let c = ws.requirements.components();
        let levels = &ws.index_levels[id * d..(id + 1) * d];
        // Admit each mapped one-dimensional rule before staging its tensor
        // nodes. Distinct canonical cache keys do not prove that the physical
        // coordinates remain distinct after rounding.
        for (axis, &level) in levels.iter().enumerate() {
            let map = ws.maps[axis];
            let rule = &self.levels[level];
            let interval = Interval::new(map.point(-S::ONE)?, map.point(S::ONE)?)?;
            let mut validator = crate::fixed::Geometry::Interval {
                interval,
                single_endpoint: false,
            }
            .validator(rule.nodes.len());
            for &node in &rule.nodes {
                if !validator.push(map.point(node)?)? {
                    return Ok(Some(IntegrationTermination::GeometryFailure));
                }
            }
        }
        let count = levels.iter().try_fold(1usize, |n, &level| {
            checked_mul(
                n,
                self.levels[level].keys.len(),
                scioxide_core::quantity!("index_nodes"),
            )
        })?;
        // Pass 1: enumerate the tensor grid of this multi-index and stage every node not yet
        // in the cache. New nodes occupy a contiguous block of the cache, so they can be
        // admitted as one unit and evaluated in a single (possibly parallel) batch; on
        // refusal the stage is rolled back and nothing was evaluated.
        let prior = ws.cache_used;
        ws.counters.fill(0);
        ws.staged_slots.clear();
        for _ in 0..count {
            for (axis, &level) in levels.iter().enumerate() {
                let rule = &self.levels[level];
                let offset = ws.counters[axis];
                ws.key_buffer[axis] = rule.keys[offset];
                ws.point_buffer[axis] = ws.maps[axis].point(rule.nodes[offset])?;
            }
            let (slot, found) = ws.cache_slot(&ws.key_buffer);
            if found.is_none() {
                if ws.cache_used == self.options.max_evaluations.get() {
                    ws.rollback_stage(prior);
                    return Ok(Some(IntegrationTermination::BudgetLimit(
                        scioxide_core::BudgetKind::Evaluation,
                    )));
                }
                let new_id = ws.cache_used;
                ws.cache_keys[new_id * d..(new_id + 1) * d].copy_from_slice(&ws.key_buffer);
                ws.cache_points[new_id * d..(new_id + 1) * d].copy_from_slice(&ws.point_buffer);
                ws.slots[slot] = new_id;
                ws.staged_slots.push(slot);
                ws.cache_used += 1;
            }
            advance(&mut ws.counters, self, levels);
        }
        let fresh = ws.cache_used - prior;
        if let Some(stop) = admit(context, fresh, id != 0)? {
            ws.rollback_stage(prior);
            return Ok(Some(stop));
        }
        if fresh > 0 {
            sampler.sample(
                &ws.cache_points[prior * d..ws.cache_used * d],
                &mut ws.cache_values[prior * c..ws.cache_used * c],
                d,
                c,
                context,
            )?;
        }
        // Pass 2: the hierarchical surplus of this index is the tensor product of the
        // difference rules applied to (cached) values; old and new nodes are read alike.
        ws.staged_slots.clear();
        ws.counters.fill(0);
        ws.sums.fill(ScaledAccumulator::ZERO);
        for _ in 0..count {
            let mut weight = Scaled::from_value(S::ONE)?;
            for (axis, &level) in levels.iter().enumerate() {
                let rule = &self.levels[level];
                let offset = ws.counters[axis];
                ws.key_buffer[axis] = rule.keys[offset];
                weight = weight.mul(ws.maps[axis].scaled_weight(rule.differences[offset])?)?;
            }
            let (_, node) = ws.cache_slot(&ws.key_buffer);
            let node = node.ok_or_else(|| {
                SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "staged sparse-grid node missing from cache",
                )
            })?;
            for (component, sum) in ws.sums.iter_mut().enumerate() {
                sum.add_product(
                    weight,
                    Scaled::from_value(ws.cache_values[node * c + component])?,
                )?;
            }
            advance(&mut ws.counters, self, levels);
        }
        for (component, sum) in ws.sums.iter().enumerate() {
            ws.index_surplus[id * c + component] = sum.round()?;
        }
        Ok(None)
    }

    fn add_forward<B: BatchSampler<S>>(
        &self,
        parent: usize,
        ws: &mut SparseGridWorkspace<S>,
        sampler: &mut B,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Option<IntegrationTermination>, SciOxideError> {
        // Gerstner-Griebel dimension adaptivity: after `parent` becomes old, each forward
        // neighbour parent + e_axis is added as active only if all of its backward
        // neighbours are already old. This keeps the index set downward closed, which the
        // Smolyak combination of difference rules requires to telescope correctly.
        let d = self.dimension;
        for axis in 0..d {
            ws.candidate
                .copy_from_slice(&ws.index_levels[parent * d..(parent + 1) * d]);
            if ws.candidate[axis] == self.options.max_level {
                continue;
            }
            ws.candidate[axis] += 1;
            if ws.find_index(&ws.candidate).is_some() {
                continue;
            }
            let mut admissible = true;
            for back in 0..d {
                if ws.candidate[back] > 0 {
                    ws.candidate[back] -= 1;
                    admissible &= ws
                        .find_index(&ws.candidate)
                        .is_some_and(|index| ws.index_state[index] == 1);
                    ws.candidate[back] += 1;
                }
            }
            if !admissible {
                continue;
            }
            let Some(id) = ws.push_candidate(self.options.max_indices.get()) else {
                return Ok(Some(IntegrationTermination::RefinementLimit));
            };
            if let Some(stop) = self.evaluate_index(id, ws, sampler, context)? {
                // An unevaluated candidate has no surplus; its direction stays
                // pending, so the parent's surplus remains frontier evidence.
                ws.index_state[id] = 0;
                ws.index_count -= 1;
                return Ok(Some(stop));
            }
            for back in 0..d {
                ws.candidate
                    .copy_from_slice(&ws.index_levels[id * d..(id + 1) * d]);
                if ws.candidate[back] > 0 {
                    ws.candidate[back] -= 1;
                    if let Some(previous) = ws.find_index(&ws.candidate) {
                        ws.index_pending[previous] -= 1;
                    }
                }
            }
        }
        Ok(None)
    }

    fn run<B: BatchSampler<S>>(
        &self,
        bounds: &[Interval<S>],
        ws: &mut SparseGridWorkspace<S>,
        sampler: &mut B,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<RunReport, SciOxideError> {
        if bounds.len() != self.dimension || ws.requirements.dimension() != self.dimension {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.dimension),
                scioxide_core::Dimensions::one(bounds.len()),
            ));
        }
        ws.reset();
        let mut report = RunReport {
            termination: IntegrationTermination::RefinementLimit,
            measured: false,
            num_evals: 0,
            refinements: 0,
            has_error: false,
        };
        if self.dimension == 0 {
            if let Some(stop) = admit(context, 1, false)? {
                report.termination = stop;
                return Ok(report);
            }
            sampler.sample(&[], &mut ws.value, 0, ws.requirements.components(), context)?;
            ws.cache_values[..ws.requirements.components()].copy_from_slice(&ws.value);
            ws.cache_used = 1;
            report.measured = true;
            report.num_evals = 1;
            report.termination = IntegrationTermination::FixedRuleCompleted;
            return Ok(report);
        }
        if bounds.iter().any(|bound| bound.lower() == bound.upper()) {
            report.measured = true;
            report.termination = IntegrationTermination::DegenerateDomain;
            return Ok(report);
        }
        if ws.requirements.components() == 0 {
            report.measured = true;
            report.termination = IntegrationTermination::FixedRuleCompleted;
            return Ok(report);
        }
        for &bound in bounds {
            let Some(map) = AffineMap1D::from_interval(bound)? else {
                report.termination = IntegrationTermination::GeometryFailure;
                return Ok(report);
            };
            ws.maps.push(map);
        }
        ws.candidate.fill(0);
        let root = ws
            .push_candidate(self.options.max_indices.get())
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("sparse_indices"))
            })?;
        if let Some(stop) = self.evaluate_index(root, ws, sampler, context)? {
            report.termination = stop;
            return Ok(report);
        }
        ws.accept_index(root);
        ws.value
            .copy_from_slice(&ws.index_surplus[..ws.requirements.components()]);
        for component in 0..ws.requirements.components() {
            ws.total_sums[component].add_product(
                Scaled::from_value(S::ONE)?,
                Scaled::from_value(ws.value[component])?,
            )?;
        }
        report.measured = true;
        // The integral is the sum of old surpluses; the active surpluses are not added but
        // summed in magnitude as the (heuristic) error. Each step accepts the active index
        // with the largest surplus relative to the tolerance and activates its admissible
        // forward neighbours. Every published stop carries that frontier evidence.
        let mut stop = self.add_forward(root, ws, sampler, context)?;
        loop {
            report.num_evals = ws.cache_used;
            let best = self.frontier(ws)?;
            report.has_error = true;
            if let Some(stop) = stop {
                report.termination = stop;
                return Ok(report);
            }
            let Some(selected) = best else {
                report.termination = IntegrationTermination::RefinementLimit;
                return Ok(report);
            };
            // Ids 1..=d are the first-level refinements along each axis (the root's forward
            // neighbours). Requiring them all to be accepted before stopping guards against
            // a tiny surplus by coincidence (e.g. symmetry) hiding an unexplored direction.
            if ws.index_state[1..=self.dimension]
                .iter()
                .all(|&state| state == 1)
                && self
                    .options
                    .tolerance
                    .is_satisfied(&ws.errors, &ws.scales)?
            {
                report.termination = IntegrationTermination::HeuristicToleranceReached;
                return Ok(report);
            }
            ws.accept_index(selected);
            for component in 0..ws.requirements.components() {
                ws.total_sums[component].add_product(
                    Scaled::from_value(S::ONE)?,
                    Scaled::from_value(
                        ws.index_surplus[selected * ws.requirements.components() + component],
                    )?,
                )?;
                ws.value[component] = ws.total_sums[component].round()?;
            }
            report.refinements += 1;
            stop = self.add_forward(selected, ws, sampler, context)?;
        }
    }

    /// Recomputes the error estimate of the current estimate, `4·|surplus|` summed
    /// over active indices and accepted indices with unexplored forward directions,
    /// with its directional indicators. Returns the active index with the largest
    /// surplus relative to the tolerance, if any remains.
    fn frontier(&self, ws: &mut SparseGridWorkspace<S>) -> Result<Option<usize>, SciOxideError> {
        let c = ws.requirements.components();
        ws.errors.fill(S::ZERO);
        ws.directional.fill(S::ZERO);
        for (scale, &value) in ws.scales.iter_mut().zip(&ws.value) {
            *scale = value.abs();
        }
        let mut best = None;
        let mut best_score = S::ZERO;
        for id in 0..ws.index_count {
            let active = ws.index_state[id] == 2;
            // An accepted index still represents unexplored descendants while
            // admissibility blocks its forward neighbours. Dropping its
            // surplus here can leave a zero frontier around a large mixed
            // contribution. Level exhaustion also leaves an unresolved tail.
            let blocked = !active && ws.index_pending[id] != 0;
            if !active && !blocked {
                continue;
            }
            let mut axis_magnitude = S::ZERO;
            let mut score = S::ZERO;
            for component in 0..c {
                let magnitude = crate::kernel::checked_intermediate(
                    S::TWO * S::TWO * ws.index_surplus[id * c + component].abs(),
                )?;
                ws.errors[component] =
                    crate::kernel::checked_intermediate(ws.errors[component] + magnitude)?;
                axis_magnitude = axis_magnitude.max(magnitude);
                let threshold =
                    self.options
                        .tolerance
                        .threshold(component, c, ws.scales[component])?;
                let ratio = if threshold == S::ZERO {
                    if magnitude == S::ZERO {
                        S::ZERO
                    } else {
                        S::MAX
                    }
                } else {
                    (magnitude / threshold).min(S::MAX)
                };
                score = score.max(ratio);
            }
            for axis in 0..self.dimension {
                if ws.index_levels[id * self.dimension + axis] > 0 {
                    ws.directional[axis] =
                        crate::kernel::checked_intermediate(ws.directional[axis] + axis_magnitude)?;
                }
            }
            if active && (best.is_none() || score > best_score) {
                best = Some(id);
                best_score = score;
            }
        }
        Ok(best)
    }

    /// Integrates `function` over the box `bounds` and writes the estimate to
    /// `out`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// `scope` is an [`ExecutionContext`] or a solve scope (see
    /// [Execution contexts](scioxide_core#execution-contexts)).
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `bounds` does not have `d` intervals, or if
    ///   `out` or a per-component tolerance has a different number of
    ///   components than the workspace.
    /// - `InvalidInput` (`Inconsistent`) if the workspace was made for another
    ///   dimension or for insufficient index or evaluation capacity.
    /// - `NumericalInstability` (`Breakdown`) if the initial midpoint is not
    ///   representable strictly inside an axis. Collapsed nodes during later
    ///   refinement return `GeometryFailure` with the last complete estimate.
    /// - `ConvergenceFailure` or `ExecutionStopped` if a limit, cancellation or
    ///   a deadline stops the call before the first estimate.
    /// - The errors of `function`, and `NonFinite` if it returns NaN or
    ///   infinity (see [Integrands](crate#integrands)).
    pub fn integrate_into<'scope, 'capability, F, V, O>(
        &self,
        function: F,
        bounds: &[Interval<S>],
        out: &mut O,
        ws: &mut SparseGridWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<IntegrationReport<S>, SciOxideError>
    where
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValue<Scale = S>,
        O: AffineValueMut<Scale = S> + ?Sized,
    {
        let function = &function;
        ws.published = false;
        let components = ws.requirements.components();
        if let Err(error) = check_value_shape(out, components)
            .and_then(|()| crate::workspace::check_tolerance(self.options.tolerance(), components))
            .and_then(|()| crate::kernel::shape(self.dimension, bounds.len()))
            .and_then(|()| {
                if ws.check(&self.workspace_requirements(components)?)?
                    == scioxide_core::Capacity::NeedsGrowth
                {
                    return Err(crate::kernel::invalid(
                        scioxide_core::argument!("workspace"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "insufficient sparse-grid storage",
                    ));
                }
                Ok::<(), SciOxideError>(())
            })
        {
            return ws.attempts.reject(error);
        }
        let mut attempts = core::mem::take(&mut ws.attempts);
        let result = attempts.integrate(
            scope.into(),
            ALGORITHM,
            self.dimension,
            scioxide_core::SolveBudgets::unbounded().with_limit(
                scioxide_core::BudgetKind::Evaluation,
                self.options.max_evaluations,
            ),
            |cx| {
                let report = self.run(
                    bounds,
                    ws,
                    &mut ValueSampler {
                        function,
                        policy: self.options.policy,
                    },
                    cx,
                )?;
                if !report.measured {
                    return Ok(crate::workspace::Outcome::Unpublished(report.termination));
                }
                for (error, &magnitude) in ws.component_errors.iter_mut().zip(&ws.errors) {
                    *error = if report.has_error {
                        heuristic_components(magnitude)
                    } else {
                        IntegrationErrorComponents::default()
                    };
                }
                cx.commit_accepted(PartialResultTrust::Admissible)?;
                write_value(&ws.value, out)?;
                ws.published = report.has_error;
                Ok(crate::workspace::Outcome::Published {
                    termination: report.termination,
                    subdivisions: report.refinements,
                    errors: if report.has_error {
                        IntegrationErrorComponents::envelope(ws.component_errors.iter())
                    } else {
                        None
                    },
                })
            },
        );
        ws.attempts = attempts;
        result
    }
    /// Integrates a real, complex or fixed-size vector integrand over the box
    /// `bounds`.
    ///
    /// Accepts closures by value; pass `&f` to reuse an integrand.
    ///
    /// Allocates a workspace and runs on
    /// [`ExecutionContext::sequential_native`].
    ///
    /// # Errors
    ///
    /// - The errors of [`SparseGridPlan::workspace`] and
    ///   [`SparseGridPlan::integrate_into`].
    pub fn integrate<F, V>(
        &self,
        function: F,
        bounds: &[Interval<S>],
    ) -> Result<(V, IntegrationReport<S>), SciOxideError>
    where
        F: Fn(&[S]) -> Result<V, SciOxideError> + Sync,
        V: AffineValueMut<Scale = S> + Copy + Default,
    {
        let function = &function;
        let mut out = V::default();
        let mut ws = self.workspace(out.dimension()?)?;
        let report = self.integrate_into(
            function,
            bounds,
            &mut out,
            &mut ws,
            ExecutionContext::sequential_native(),
        )?;
        Ok((out, report))
    }
}

fn heuristic_components<S: RealScalar>(magnitude: S) -> IntegrationErrorComponents<S> {
    IntegrationErrorComponents {
        discretization: Some(ErrorEstimate {
            magnitude,
            reliability: ErrorReliability::Heuristic,
        }),
        roundoff: None,
        inner: None,
        moments: None,
        extrapolation: None,
        tail: None,
        tail_diagnostics: None,
    }
}
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    #[cfg(feature = "std")]
    use core::num::NonZeroUsize;

    use scioxide_core::{
        BudgetKind, Complex, ExecutionContext, NormSelection, ProblemToken, SolveBudgets,
        SolveIdentity, SolveState, Tolerance,
    };

    fn budget(value: usize) -> Budget {
        Budget::try_from(value).unwrap()
    }

    fn context_with_execution<T>(
        execution: ExecutionContext<'_>,
        budgets: SolveBudgets,
        f: impl FnOnce(&mut SolveContext<'_, '_>) -> T,
    ) -> T {
        let mut state = SolveState::new();
        let identity = SolveIdentity::new(
            ProblemToken::new(593),
            scioxide_core::algorithm_id!("integrate", "sparse-grid-test", 1),
            0,
            0,
        );
        let mut budgets = budgets;
        let mut context = state
            .begin_solve(execution, identity, &mut budgets)
            .unwrap();
        f(&mut context)
    }

    fn context_run<T>(budgets: SolveBudgets, f: impl FnOnce(&mut SolveContext<'_, '_>) -> T) -> T {
        context_with_execution(ExecutionContext::sequential(), budgets, f)
    }

    fn plan<S: RealScalar + Default>(
        dimension: usize,
        level: usize,
        indices: usize,
        evals: usize,
    ) -> SparseGridPlan<S> {
        let tolerance = ErrorTolerance::from_tolerance(
            Tolerance::from_decimal_places(7, 7),
            NormSelection::Infinity,
        );
        SparseGridPlan::new(
            dimension,
            SparseGridOptions::default()
                .with_tolerance(tolerance)
                .with_max_level(level)
                .with_max_indices(budget(indices))
                .with_max_evaluations(budget(evals)),
        )
        .unwrap()
    }

    #[test]
    fn polynomial_and_mixed_terms_have_independent_integrals() {
        let plan = plan::<f64>(2, 3, 20, 100);
        let bounds = [Interval::new(0.0, 1.0).unwrap(); 2];
        let function = |x: &[f64]| Ok(x[0] * x[0] + x[0] * x[1] + x[1] * x[1]);
        let result = plan.integrate(&function, &bounds).unwrap();
        assert!((result.0 - (2.0 / 3.0 + 0.25)).abs() < 1e-12);
        assert_eq!(
            result.1.termination(),
            IntegrationTermination::HeuristicToleranceReached
        );
        assert_eq!(
            result
                .1
                .error_components()
                .unwrap()
                .discretization
                .unwrap()
                .reliability,
            ErrorReliability::Heuristic
        );
    }

    #[test]
    fn separable_dimension_sweep_matches_analytic_product() {
        for dimension in 1..=4 {
            let plan = plan::<f64>(dimension, 2, 64, 256);
            let bounds = alloc::vec![Interval::new(0.0, 1.0).unwrap(); dimension];
            let function =
                |x: &[f64]| Ok(x.iter().map(|&value| 1.0 + value * value).product::<f64>());
            let result = plan.integrate(&function, &bounds).unwrap();
            let expected = (4.0_f64 / 3.0).powi(dimension as i32);
            assert!((result.0 - expected).abs() < 1e-11);
        }
    }

    #[test]
    fn interaction_hidden_from_each_axis_probe_is_discovered() {
        let plan = plan::<f64>(2, 2, 18, 64);
        let bounds = [Interval::new(0.0, 1.0).unwrap(); 2];
        let function = |x: &[f64]| Ok((x[0] - 0.5).powi(2) * (x[1] - 0.5).powi(2));
        let result = plan.integrate(&function, &bounds).unwrap();
        assert!((result.0 - 1.0 / 144.0).abs() < 1e-12);
    }

    #[test]
    fn f32_nested_nodes_are_evaluated_once() {
        let plan = plan::<f32>(1, 2, 8, 16);
        let bounds = [Interval::new(0.0f32, 1.0).unwrap()];
        let function = |x: &[f32]| Ok(x[0].powi(4));
        let mut workspace = plan.workspace(1).unwrap();
        let result = (|| {
            let mut value = bounds
                .first()
                .map(|b| b.lower())
                .unwrap_or_else(|| <_ as scioxide_core::AlgebraicScalar>::ZERO);
            let report = plan.integrate_into(
                &function,
                &bounds,
                &mut value,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential_native(),
            )?;
            Ok::<_, scioxide_core::SciOxideError>((value, report))
        })()
        .unwrap();
        assert!((result.0 - 0.2).abs() < 2e-6);
        assert_eq!(result.1.evaluations(), 5);
        assert_eq!(workspace.cached_nodes(), 5);
    }

    #[test]
    fn callback_budget_retains_only_complete_base_estimate() {
        let plan = plan::<f64>(2, 1, 4, 3);
        let bounds = [Interval::new(0.0, 1.0).unwrap(); 2];
        let function = |x: &[f64]| Ok(x[0] + x[1]);
        let result = context_run(
            SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, budget(1)),
            |context| {
                (|| {
                    let mut ws = plan.workspace(1)?;
                    let mut value = bounds
                        .first()
                        .map(|b| b.lower())
                        .unwrap_or_else(|| <_ as scioxide_core::AlgebraicScalar>::ZERO);
                    let report =
                        plan.integrate_into(&function, &bounds, &mut value, &mut ws, context)?;
                    Ok::<_, scioxide_core::SciOxideError>((value, report))
                })()
            },
        )
        .unwrap();
        assert_eq!(
            result.1.termination(),
            IntegrationTermination::BudgetLimit(BudgetKind::Evaluation)
        );
        assert_eq!(result.1.evaluations(), 1);
        assert_eq!(result.0, 1.0);
        // The root's unexplored directions keep its surplus as frontier evidence.
        assert_eq!(result.1.error_estimate(), Some(4.0));
    }

    #[test]
    fn directional_frontier_detects_anisotropy() {
        let tolerance = ErrorTolerance::from_tolerance(
            Tolerance::from_decimal_places(0, 0),
            NormSelection::Infinity,
        );
        let options = SparseGridOptions::default()
            .with_tolerance(tolerance)
            .with_max_level(2)
            .with_max_indices(budget(10))
            .with_max_evaluations(budget(30));
        let plan = SparseGridPlan::new(2, options).unwrap();
        let bounds = [Interval::new(0.0, 1.0).unwrap(); 2];
        let function = |x: &[f64]| Ok(x[0].powi(4) + 0.2 * x[1].powi(2));
        let mut workspace = plan.workspace(1).unwrap();
        let result = (|| {
            let mut value = bounds
                .first()
                .map(|b| b.lower())
                .unwrap_or_else(|| <_ as scioxide_core::AlgebraicScalar>::ZERO);
            let report = plan.integrate_into(
                &function,
                &bounds,
                &mut value,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential_native(),
            )?;
            Ok::<_, scioxide_core::SciOxideError>((value, report))
        })()
        .unwrap();
        assert_eq!(
            result.1.termination(),
            IntegrationTermination::HeuristicToleranceReached
        );
        assert!(
            workspace.directional_indicators()[0] > 10.0 * workspace.directional_indicators()[1],
            "indicators: {:?}",
            workspace.directional_indicators()
        );
    }

    #[test]
    fn zero_dimensional_and_degenerate_domains_follow_measure_contract() {
        let empty = plan::<f64>(0, 0, 1, 1);
        let function = |_: &[f64]| Ok(3.25);
        let result = empty.integrate(&function, &[]).unwrap();
        assert_eq!(result.0, 3.25);
        assert_eq!(result.1.evaluations(), 1);
        let plan = plan::<f64>(2, 1, 4, 10);
        let function = |_: &[f64]| -> Result<f64, SciOxideError> {
            Err(SciOxideError::size_overflow(scioxide_core::quantity!(
                "result_size"
            )))
        };
        let bounds = [
            Interval::new(0.0, 0.0).unwrap(),
            Interval::new(0.0, 1.0).unwrap(),
        ];
        let result = plan.integrate(&function, &bounds).unwrap();
        assert_eq!(
            result.1.termination(),
            IntegrationTermination::DegenerateDomain
        );
        assert_eq!(result.1.evaluations(), 0);
    }

    #[test]
    fn complex_affine_and_executor_paths_agree() {
        let plan = plan::<f64>(2, 2, 12, 40);
        let bounds = [Interval::new(0.0, 1.0).unwrap(); 2];
        let sample = Complex::new(0.0, 0.0);
        let mut sequential = Complex::new(7.0, 8.0);
        let mut parallel = Complex::new(7.0, 8.0);
        let _seq_errors = [heuristic_components(0.0); 2];
        let _par_errors = [heuristic_components(0.0); 2];
        let mut seq_ws = plan.workspace(2).unwrap();
        let mut par_ws = plan.workspace(2).unwrap();
        let seq = context_run(SolveBudgets::unbounded(), |context| {
            plan.integrate_into(
                |point| {
                    let mut lane = sample;
                    scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                        *v = <_ as scioxide_core::RealScalar>::NAN
                    });
                    {
                        let value = &mut lane;

                        *value = Complex::new(point[0] * point[0], point[1] * point[1]);
                        Ok::<(), SciOxideError>(())
                    }?;
                    Ok(lane)
                },
                &bounds,
                &mut sequential,
                &mut seq_ws,
                context,
            )
        })
        .unwrap();
        let par = context_run(SolveBudgets::unbounded(), |context| {
            plan.integrate_into(
                |point| {
                    let mut lane = [<_ as scioxide_core::RealScalar>::NAN; 2];
                    {
                        let row = &mut lane;

                        row[0] = point[0] * point[0];
                        row[1] = point[1] * point[1];
                        Ok::<(), SciOxideError>(())
                    }?;
                    Ok(lane)
                },
                &bounds,
                &mut parallel,
                &mut par_ws,
                context,
            )
        })
        .unwrap();
        assert_eq!(seq.evaluations(), par.evaluations());
        assert_eq!(sequential, parallel);
        assert!((sequential.re - 1.0 / 3.0).abs() < 1e-12);
        assert!((sequential.im - 1.0 / 3.0).abs() < 1e-12);
        #[cfg(feature = "std")]
        {
            let workers = NonZeroUsize::new(3).unwrap();
            let pool = scioxide_core::ThreadExecutor::new(workers);
            let mut threaded = Complex::new(7.0, 8.0);
            let threaded_report = context_with_execution(
                ExecutionContext::new(&pool),
                SolveBudgets::unbounded(),
                |context| {
                    plan.integrate_into(
                        |point| {
                            let mut lane = [<_ as scioxide_core::RealScalar>::NAN; 2];
                            {
                                let row = &mut lane;

                                row[0] = point[0] * point[0];
                                row[1] = point[1] * point[1];
                                Ok::<(), SciOxideError>(())
                            }?;
                            Ok(lane)
                        },
                        &bounds,
                        &mut threaded,
                        &mut par_ws,
                        context,
                    )
                },
            )
            .unwrap();
            assert_eq!(threaded, sequential);
            assert_eq!(threaded_report.evaluations(), seq.evaluations());
        }
    }

    #[test]
    fn callback_failure_does_not_publish_affine_output() {
        let plan = plan::<f64>(2, 1, 6, 10);
        let bounds = [Interval::new(0.0, 1.0).unwrap(); 2];
        let sample = [0.0; 2];
        let mut output = [7.0, 8.0];
        let original = heuristic_components(5.0);
        let errors = [original; 2];
        let mut workspace = plan.workspace(2).unwrap();
        let result = context_run(SolveBudgets::unbounded(), |context| {
            plan.integrate_into(
                |_point| {
                    let mut lane = sample;
                    scioxide_core::AffineValueMut::for_each_component_mut(&mut lane, |v| {
                        *v = <_ as scioxide_core::RealScalar>::NAN
                    });
                    {
                        Err(SciOxideError::size_overflow(scioxide_core::quantity!(
                            "result_size"
                        )))
                    }?;
                    Ok(lane)
                },
                &bounds,
                &mut output,
                &mut workspace,
                context,
            )
        });
        assert!(result.is_err());
        assert_eq!(output, [7.0, 8.0]);
        assert_eq!(errors[0].discretization.unwrap().magnitude, 5.0);
    }
}

crate::workspace::workspace_protocol!(SparseGridWorkspace, IntegrationReport<S>);
impl<S: RealScalar> SparseGridWorkspace<S> {
    /// Returns the error estimate of each real component from the last
    /// successful call.
    ///
    /// Returns `None` if the last call failed or made no error estimate (a
    /// degenerate box or `d = 0`).
    pub fn component_errors(&self) -> Option<&[IntegrationErrorComponents<S>]> {
        self.published.then_some(&self.component_errors)
    }
}

impl<S: RealScalar> Default for SparseGridOptions<S> {
    fn default() -> Self {
        Self {
            tolerance: ErrorTolerance::default(),
            max_level: 4,
            max_indices: Budget::ONE.saturating_add(63),
            max_evaluations: Budget::ONE.saturating_add(4095),
            policy: ExecutionPolicy::Sequential,
        }
    }
}
