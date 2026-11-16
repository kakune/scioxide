//! Simplicial homology global optimization (SHGO) over a finite box.

use alloc::vec::Vec;

use scioxide_core::backend::{
    PartitionedSolve, checked_add, checked_mul, try_filled, try_with_capacity,
};
use scioxide_core::{
    AttemptKind, AttemptRecord, Budget, BudgetKind, Capacity, EvaluationRole, ExecutionPolicy,
    NumericScalar, PartialResultTrust, RealScalar, SciOxideError, SolveBudgets, SolveScope,
    SolveState,
};
use scioxide_random::{SeedableRng, Xoshiro256StarStar};
use scioxide_spatial::{Delaunay, GeometryOptions, PointSet};

use super::local::{
    LocalPool, LocalSolverFactory, SearchLocalSolver, budget_termination, local_identity,
    recoverable, validate_local,
};
use super::multistart::{identity, validate_point};
use super::{StartDesign, shared_objective};
use crate::attempt::{Accepted, AcceptedState, Entry, Shell, Staged};
use crate::context::{self, SolveSession};
use crate::{
    BoundedObjective, FiniteBounds, MinimizerAccepted, MinimizerProgress, ObjectiveFunction,
    OptimizeResult, OptimizeTermination, SearchRandom,
};

/// Configures [`Shgo`]: the number of samples, the triangulation limits,
/// the number of local refinements, the seed and the execution policy.
///
/// The setters do not check their arguments. [`Shgo::workspace_requirements`]
/// checks the triangulation limits for a given dimension. See
/// [Termination](crate#termination) for how the search ends.
#[derive(Debug, Clone, Copy)]
pub struct ShgoOptions {
    samples: Budget,
    max_refinements: Option<Budget>,
    geometry: GeometryOptions,
    seed: u64,
    local_budgets: SolveBudgets,
    execution: ExecutionPolicy,
}
impl ShgoOptions {
    /// Creates options that sample all `2ⁿ` corners of the box plus `samples`
    /// points of the design, for `n` variables.
    ///
    /// `geometry` limits the number of points, the dimension and the work of
    /// the Delaunay triangulation; since the corners alone grow as `2ⁿ`, it
    /// also limits the dimension in practice. The other options default to
    /// refining every candidate, seed 42, no limit on each local solve and
    /// sequential execution.
    pub const fn new(samples: Budget, geometry: GeometryOptions) -> Self {
        Self {
            samples,
            max_refinements: None,
            geometry,
            seed: 42,
            local_budgets: SolveBudgets::unbounded(),
            execution: ExecutionPolicy::Sequential,
        }
    }
    /// Sets the maximum number of candidates refined by the local solver.
    pub const fn with_max_refinements(mut self, limit: Budget) -> Self {
        self.max_refinements = Some(limit);
        self
    }
    /// Sets the seed of the sample design and the local solves.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    /// Sets the limits of each local solve. The limits of the whole search
    /// still apply.
    pub const fn with_local_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.local_budgets = budgets;
        self
    }
    /// Sets whether the local refinements run one after another or in
    /// parallel, with fixed per-refinement shares of the limits.
    ///
    /// Sampling and triangulation always run sequentially. A policy other than
    /// `Sequential` needs a problem built with [`BoundedObjective::new_sync`].
    pub const fn with_execution(mut self, execution: ExecutionPolicy) -> Self {
        self.execution = execution;
        self
    }
    /// Returns the number of design points, in addition to the corners.
    pub const fn samples(&self) -> Budget {
        self.samples
    }
    /// Returns the maximum number of local refinements, or `None` to refine
    /// every candidate.
    pub const fn max_refinements(&self) -> Option<Budget> {
        self.max_refinements
    }
    /// Returns the triangulation options.
    pub const fn geometry(&self) -> GeometryOptions {
        self.geometry
    }
    /// Returns the seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }
    /// Returns the limits of each local solve.
    pub const fn local_budgets(&self) -> SolveBudgets {
        self.local_budgets
    }
    /// Returns the execution policy of the local refinements.
    pub const fn execution(&self) -> ExecutionPolicy {
        self.execution
    }
}
fn corner_count(dimension: usize) -> Result<usize, SciOxideError> {
    let dimension = u32::try_from(dimension)
        .map_err(|_| SciOxideError::size_overflow(scioxide_core::quantity!("shgo_box_corners")))?;
    1_usize
        .checked_shl(dimension)
        .filter(|&n| n != 0)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("shgo_box_corners")))
}

/// Reports the local refinement of one [`Shgo`] candidate.
#[derive(Debug)]
pub struct ShgoRefinement<S: RealScalar> {
    vertex: usize,
    result: OptimizeResult<S>,
    completed: bool,
    error: Option<SciOxideError>,
    counts: scioxide_core::SolveCounts,
}
impl<S: RealScalar> ShgoRefinement<S> {
    /// Returns the sample index of the candidate. The corners come first, then
    /// the design points.
    pub const fn vertex(&self) -> usize {
        self.vertex
    }
    /// Returns the local result, if the local solve produced a finite point
    /// inside the box.
    pub fn result(&self) -> Option<&OptimizeResult<S>> {
        self.completed.then_some(&self.result)
    }
    /// Returns the error of the local solve, if it failed.
    pub const fn error(&self) -> Option<&SciOxideError> {
        self.error.as_ref()
    }
    /// Returns the evaluations and iterations of the local solve, including
    /// failed callbacks.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    fn reserved(dimension: usize) -> Result<Self, SciOxideError> {
        let mut result = OptimizeResult::empty();
        result.try_reserve(dimension)?;
        Ok(Self {
            vertex: 0,
            result,
            completed: false,
            error: None,
            counts: scioxide_core::SolveCounts::new(),
        })
    }
    fn copy_from(&mut self, source: &Self) -> Result<(), SciOxideError> {
        self.vertex = source.vertex;
        self.result.try_copy_from(&source.result)?;
        self.completed = source.completed;
        self.error.clone_from(&source.error);
        self.counts = source.counts;
        Ok(())
    }
}

/// Result of a [`Shgo`] search: the samples, the candidates, their local
/// refinements and the termination.
///
/// Start with [`Self::empty`]; the search sizes it on first use.
#[derive(Debug)]
pub struct ShgoResult<S: RealScalar> {
    points: Vec<S>,
    values: Vec<S>,
    candidates: Vec<usize>,
    refinements: Vec<ShgoRefinement<S>>,
    sampled: usize,
    refined: usize,
    best_sample: Option<usize>,
    best_local: Option<usize>,
    simplices: usize,
    termination: OptimizeTermination,
    counts: scioxide_core::SolveCounts,
    dimension: usize,
}
impl<S: RealScalar> Default for ShgoResult<S> {
    fn default() -> Self {
        Self::empty()
    }
}
impl<S: RealScalar> ShgoResult<S> {
    /// Returns an empty result without allocating.
    pub const fn empty() -> Self {
        Self {
            points: Vec::new(),
            values: Vec::new(),
            candidates: Vec::new(),
            refinements: Vec::new(),
            sampled: 0,
            refined: 0,
            best_sample: None,
            best_local: None,
            simplices: 0,
            termination: OptimizeTermination::NotStarted,
            counts: scioxide_core::SolveCounts::new(),
            dimension: 0,
        }
    }
    /// Returns the sampled points, one row of length `n` per point.
    pub fn points(&self) -> &[S] {
        &self.points[..self.sampled * self.dimension]
    }
    /// Returns the objective values at the sampled points.
    pub fn values(&self) -> &[S] {
        &self.values[..self.sampled]
    }
    /// Returns the sample indices of the candidates, sorted by value and then
    /// by index. A candidate is a sample with no lower neighbor in the
    /// triangulation.
    pub fn candidates(&self) -> &[usize] {
        &self.candidates
    }
    /// Returns the local refinements that ran, in candidate order.
    pub fn refinements(&self) -> &[ShgoRefinement<S>] {
        &self.refinements[..self.refined]
    }
    /// Returns the sampled point with the lowest value, and the value.
    pub fn best_sample(&self) -> Option<(&[S], S)> {
        self.best_sample.map(|i| {
            (
                &self.points[i * self.dimension..(i + 1) * self.dimension],
                self.values[i],
            )
        })
    }
    /// Returns the lowest point among the samples and the local results,
    /// whether or not the local solves succeeded.
    pub fn best_observed(&self) -> Option<(&[S], S)> {
        let mut best = self.best_sample();
        for refinement in self.refinements() {
            if let Some(local) = refinement.result()
                && best.is_none_or(|(_, value)| local.fun() < value)
            {
                best = Some((local.x(), local.fun()));
            }
        }
        best
    }
    /// Returns the lowest result of a successful local solve.
    pub fn best_refined(&self) -> Option<&OptimizeResult<S>> {
        self.best_local.map(|i| &self.refinements[i].result)
    }
    /// Returns the number of simplices of the triangulation; zero before it is
    /// built.
    pub const fn simplex_count(&self) -> usize {
        self.simplices
    }
    /// Returns why the search stopped. Completing the search gives
    /// [`OptimizeTermination::SearchBudgetCompleted`], which is not a success.
    pub const fn termination(&self) -> OptimizeTermination {
        self.termination
    }
    /// Returns the evaluations and iterations of the whole search.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    fn fits(&self, sites: usize, refinements: usize, dimension: usize) -> bool {
        self.dimension == dimension
            && self.values.len() == sites
            && self.points.len() == sites * dimension
            && self.candidates.capacity() >= sites
            && self.refinements.len() == refinements
            && self
                .refinements
                .iter()
                .all(|refinement| refinement.result.fits(dimension))
    }
    fn try_reserve(
        &mut self,
        sites: usize,
        refinements: usize,
        dimension: usize,
    ) -> Result<(), SciOxideError> {
        if self.fits(sites, refinements, dimension) {
            return Ok(());
        }
        let mut slots =
            try_with_capacity(refinements, scioxide_core::quantity!("shgo_refinements"))?;
        for _ in 0..refinements {
            slots.push(ShgoRefinement::reserved(dimension)?);
        }
        *self = Self {
            points: try_filled(
                checked_mul(sites, dimension, scioxide_core::quantity!("shgo_points"))?,
                S::ZERO,
                scioxide_core::quantity!("shgo_points"),
            )?,
            values: try_filled(sites, S::ZERO, scioxide_core::quantity!("shgo_values"))?,
            candidates: try_with_capacity(sites, scioxide_core::quantity!("shgo_candidates"))?,
            refinements: slots,
            dimension,
            ..Self::empty()
        };
        Ok(())
    }
}
impl<S: RealScalar> Staged for ShgoResult<S> {
    fn reset(&mut self) {
        self.sampled = 0;
        self.refined = 0;
        self.best_sample = None;
        self.best_local = None;
        self.simplices = 0;
        self.termination = OptimizeTermination::NotStarted;
        self.counts = scioxide_core::SolveCounts::new();
        self.candidates.clear();
        for r in &mut self.refinements {
            r.completed = false;
            r.error = None;
            r.counts = scioxide_core::SolveCounts::new();
        }
    }
    fn publish(&mut self, source: &Self) -> Result<(), SciOxideError> {
        if !self.fits(
            source.values.len(),
            source.refinements.len(),
            source.dimension,
        ) {
            let mut fresh = Self::empty();
            fresh.try_reserve(
                source.values.len(),
                source.refinements.len(),
                source.dimension,
            )?;
            fresh.publish(source)?;
            *self = fresh;
            return Ok(());
        }
        self.points.copy_from_slice(&source.points);
        self.values.copy_from_slice(&source.values);
        self.candidates.clear();
        self.candidates.extend_from_slice(&source.candidates);
        for (refinement, source) in self.refinements.iter_mut().zip(&source.refinements) {
            refinement.copy_from(source)?;
        }
        self.sampled = source.sampled;
        self.refined = source.refined;
        self.best_sample = source.best_sample;
        self.best_local = source.best_local;
        self.simplices = source.simplices;
        self.termination = source.termination;
        self.counts = source.counts;
        Ok(())
    }
    /// Sampled searches never report success.
    fn succeeded(&self) -> bool {
        false
    }
}
impl<S: RealScalar> Accepted<S> for ShgoResult<S> {
    fn accepted(&self) -> Option<AcceptedState<'_, S>> {
        let (point, value) = self.best_observed()?;
        Some(AcceptedState {
            point,
            value,
            gradient: None,
            iterations: self.sampled,
        })
    }
}

/// Workspace size of a [`Shgo`] search; it borrows the method's local factory.
#[derive(Debug)]
pub struct ShgoWorkspaceRequirements<'a, F> {
    factory: &'a F,
    token: u128,
    dimension: usize,
    sites: usize,
    refinements: usize,
    seed: u64,
}
impl<F> ShgoWorkspaceRequirements<'_, F> {
    /// Returns the number of variables.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    /// Returns the number of sample points: the corners plus the design points.
    pub const fn sites(&self) -> usize {
        self.sites
    }
    /// Returns the number of local solvers.
    pub const fn refinements(&self) -> usize {
        self.refinements
    }
}

/// Seeds, child slots, graph scratch, box copy and isolated local solvers.
pub(crate) struct ShgoStorage<S: RealScalar, L> {
    complex: Option<Delaunay<S>>,
    seeds: Vec<u64>,
    seed: u64,
    children: Vec<PartitionedSolve>,
    lower: Vec<bool>,
    parent: Vec<usize>,
    plateau_lower: Vec<bool>,
    pool: LocalPool<L>,
    bounds: FiniteBounds<S>,
    token: u128,
}

/// Workspace of [`Shgo`]: the samples, the triangulation, the local
/// solvers and the record of the last call.
pub struct ShgoWorkspace<S: RealScalar, L> {
    storage: ShgoStorage<S, L>,
    staging: ShgoResult<S>,
    shell: Shell<MinimizerProgress<S>>,
}
impl<S: RealScalar, L> core::fmt::Debug for ShgoWorkspace<S, L> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ShgoWorkspace")
            .field("sites", &self.storage.seeds.len())
            .field("refinements", &self.storage.pool.locals.len())
            .finish_non_exhaustive()
    }
}
impl<S: RealScalar, L> Default for ShgoWorkspace<S, L> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar, L> ShgoWorkspace<S, L> {
    /// Creates an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            storage: ShgoStorage {
                complex: None,
                seeds: Vec::new(),
                seed: 0,
                children: Vec::new(),
                lower: Vec::new(),
                parent: Vec::new(),
                plateau_lower: Vec::new(),
                pool: LocalPool::new(),
                bounds: FiniteBounds::unassigned(),
                token: 0,
            },
            staging: ShgoResult::empty(),
            shell: Shell::new(),
        }
    }
    /// Returns the Delaunay triangulation of the last call, or `None` before
    /// it is built or if building it failed.
    pub const fn complex(&self) -> Option<&Delaunay<S>> {
        self.storage.complex.as_ref()
    }
    /// Checks whether the workspace already fits `requirements`.
    ///
    /// # Errors
    ///
    /// Never returns an error; the `Result` matches the other workspaces.
    pub fn check<F>(
        &self,
        requirements: &ShgoWorkspaceRequirements<'_, F>,
    ) -> Result<Capacity, SciOxideError> {
        let storage = &self.storage;
        let sites = requirements.sites;
        Ok(
            if storage.pool.fits(
                requirements.refinements,
                requirements.dimension,
                requirements.token,
            ) && storage.seeds.len() == sites
                && storage.seed == requirements.seed
                && storage.children.len() == requirements.refinements
                && storage.lower.len() == sites
                && storage.bounds.fits(requirements.dimension)
                && self
                    .staging
                    .fits(sites, requirements.refinements, requirements.dimension)
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
    /// Allocates the local solvers, seeds and result storage for
    /// `requirements`.
    ///
    /// # Errors
    ///
    /// Returns the local factory's errors, `SizeOverflow` or
    /// `AllocationFailure`.
    pub fn try_reserve<F>(
        &mut self,
        requirements: &ShgoWorkspaceRequirements<'_, F>,
    ) -> Result<(), SciOxideError>
    where
        S: NumericScalar<Real = S>,
        F: LocalSolverFactory<S, Solver = L>,
    {
        if self.check(requirements)? == Capacity::Sufficient {
            return Ok(());
        }
        let sites = requirements.sites;
        let storage = &mut self.storage;
        storage.pool.prepare(
            requirements.factory,
            requirements.refinements,
            requirements.dimension,
            requirements.token,
        )?;
        if storage.seeds.len() != sites || storage.seed != requirements.seed {
            storage.seeds = scioxide_random::seed_sequence(requirements.seed, sites)?;
            storage.seed = requirements.seed;
        }
        if storage.children.len() != requirements.refinements {
            let mut slots = try_with_capacity(
                requirements.refinements,
                scioxide_core::quantity!("shgo_child_solves"),
            )?;
            slots.resize_with(requirements.refinements, PartitionedSolve::default);
            storage.children = slots;
        }
        if storage.lower.len() != sites {
            storage.lower = try_filled(
                sites,
                false,
                scioxide_core::quantity!("shgo_lower_neighbors"),
            )?;
            storage.parent =
                try_filled(sites, 0, scioxide_core::quantity!("shgo_plateau_parents"))?;
            storage.plateau_lower = try_filled(
                sites,
                false,
                scioxide_core::quantity!("shgo_plateau_status"),
            )?;
        }
        storage.bounds.try_reserve(requirements.dimension)?;
        storage.token = requirements.token;
        self.staging
            .try_reserve(sites, requirements.refinements, requirements.dimension)
    }
    /// Returns the record of the most recent call; its progress holds the
    /// best point.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<MinimizerProgress<S>>> {
        self.shell.last_attempt()
    }
    /// Returns the best point of the most recent call, if it found one.
    pub fn last_accepted(&self) -> Option<MinimizerAccepted<'_, S>>
    where
        S: NumericScalar<Real = S>,
    {
        crate::attempt::last_accepted(&self.shell, &self.staging)
    }
}

/// Minimizes an objective over a finite box with simplicial homology global
/// optimization (SHGO).
///
/// The search evaluates the objective at all `2ⁿ` corners of the box and at
/// the points of a [`StartDesign`](crate::StartDesign), for `n` variables, and
/// builds the Delaunay triangulation of these samples. Each sample with no
/// lower neighbor in the triangulation is a candidate; neighbors with equal
/// values count as one plateau. A local solver from a
/// [`LocalSolverFactory`](crate::LocalSolverFactory) refines the candidates in
/// order of value. The cost is `2ⁿ` plus the number of design points
/// evaluations, the triangulation, and one local solve per candidate, so the
/// method suits a small number of variables.
///
/// The search does not prove that it found the global minimum, and
/// completing it ends with [`OptimizeTermination::SearchBudgetCompleted`],
/// which is not a success.
///
/// # References
///
/// - S. C. Endres, C. Sandrock and W. W. Focke, "A simplicial homology
///   algorithm for Lipschitz optimisation", *J. Global Optim.* 72(2),
///   181–217, 2018.
#[derive(Debug, Clone)]
pub struct Shgo<S: RealScalar, D, F> {
    options: ShgoOptions,
    design: D,
    factory: F,
    scalar: core::marker::PhantomData<fn() -> S>,
}
impl<S, D, F> Shgo<S, D, F>
where
    S: RealScalar + NumericScalar<Real = S>,
    D: StartDesign<S>,
    F: LocalSolverFactory<S>,
{
    /// Creates a method from options, a sample design and a local solver factory.
    pub const fn new(options: ShgoOptions, design: D, factory: F) -> Self {
        Self {
            options,
            design,
            factory,
            scalar: core::marker::PhantomData,
        }
    }
    /// Returns the options.
    pub const fn options(&self) -> &ShgoOptions {
        &self.options
    }
    /// Returns the workspace size for `dimension` variables.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the corners plus design points exceed the point
    ///   limit of the triangulation options, or `dimension` exceeds their
    ///   dimension limit.
    /// - `SizeOverflow` if `2^dimension` or the size overflows.
    /// - The design's error if it cannot supply the samples.
    pub fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<ShgoWorkspaceRequirements<'_, F>, SciOxideError> {
        let corners = corner_count(dimension)?;
        let sites = checked_add(
            corners,
            self.options.samples.get(),
            scioxide_core::quantity!("shgo_sites"),
        )?;
        checked_mul(
            sites,
            dimension,
            scioxide_core::quantity!("shgo_coordinates"),
        )?;
        let limits = self.options.geometry.limits();
        if sites > limits.max_points() {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("geometry_points"),
                sites,
                limits.max_points(),
            ));
        }
        if dimension > limits.max_dimension() {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("geometry_dimension"),
                dimension,
                limits.max_dimension(),
            ));
        }
        self.design
            .validate(dimension, self.options.samples.get())?;
        Ok(ShgoWorkspaceRequirements {
            factory: &self.factory,
            token: context::options_token(&self.factory),
            dimension,
            sites,
            refinements: self
                .options
                .max_refinements
                .map_or(sites, Budget::get)
                .min(sites),
            seed: self.options.seed,
        })
    }
    /// Allocates a workspace for `dimension` variables.
    ///
    /// # Errors
    ///
    /// As [`Self::workspace_requirements`] and [`ShgoWorkspace::try_reserve`].
    pub fn workspace(
        &self,
        dimension: usize,
    ) -> Result<ShgoWorkspace<S, F::Solver>, SciOxideError> {
        let mut workspace = ShgoWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }
    /// Samples the box, triangulates the samples, refines the candidates and
    /// writes everything to `output`.
    ///
    /// A local solve that fails with `ConvergenceFailure`,
    /// `NumericalInstability` or `Singular` is recorded in its refinement
    /// report, and the search continues. When a limit, cancellation or a
    /// deadline stops the search during the refinements, `output` holds the
    /// work done so far.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a variable lacks a lower or upper bound or has
    ///   `lo ≥ hi`, a design point lies outside the box, the policy is not
    ///   sequential and the problem was not built with
    ///   [`BoundedObjective::new_sync`], or a local solver returns a point
    ///   outside the box or a non-finite value.
    /// - `ResourceLimit` and `SizeOverflow` as for
    ///   [`Self::workspace_requirements`].
    /// - `NonFinite` if the objective returns NaN or infinity at a sample.
    /// - Errors of the triangulation, the objective, a local solver other
    ///   than those recorded per refinement, and the executor.
    pub fn minimize_into<'scope, 'capability>(
        &self,
        problem: &BoundedObjective<'_, S>,
        output: &mut ShgoResult<S>,
        workspace: &mut ShgoWorkspace<S, F::Solver>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        let requirements = self.workspace_requirements(problem.dim())?;
        workspace.try_reserve(&requirements)?;
        workspace.storage.bounds.assign(problem.bounds)?;
        let shared = shared_objective(problem, self.options.execution)?;
        crate::attempt::preflight(&scope, None)?;
        let entry = Entry::Call {
            identity: identity(
                scioxide_core::algorithm_id!("optimize", "SHGO", 1),
                problem.dim(),
                &self.options,
                requirements.token,
            ),
            budgets: SolveBudgets::unbounded(),
        };
        super::attempt(
            &mut workspace.shell,
            &mut workspace.storage,
            &mut workspace.staging,
            output,
            scope,
            entry,
            |context, storage, staged| {
                storage.complex = None;
                let session = SolveSession::new(context, scioxide_core::argument!("scope"));
                let outcome = self.sample_and_select(problem, storage, staged, &session);
                let outcome = match (outcome, shared) {
                    (Ok(()), None) => self.refine_sequential(problem, storage, staged, &session),
                    (Ok(()), Some(objective)) => {
                        self.refine_partitioned(problem, objective, storage, staged, &session)
                    }
                    (Err(error), _) => Err(error),
                };
                finish(outcome, staged, &session)
            },
        )
    }
    /// Runs the search with a new workspace and returns the result.
    ///
    /// # Errors
    ///
    /// As [`Self::workspace`] and [`Self::minimize_into`].
    pub fn minimize(
        &self,
        problem: &BoundedObjective<'_, S>,
    ) -> Result<ShgoResult<S>, SciOxideError> {
        let mut workspace = self.workspace(problem.dim())?;
        let mut output = ShgoResult::empty();
        self.minimize_into(
            problem,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    fn sample_and_select(
        &self,
        problem: &BoundedObjective<'_, S>,
        work: &mut ShgoStorage<S, F::Solver>,
        staged: &mut ShgoResult<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let bounds = &work.bounds;
        let d = bounds.len();
        let corners = corner_count(d)?;
        let total = staged.values.len();
        for index in 0..total {
            session.proposal()?;
            let point = &mut staged.points[index * d..(index + 1) * d];
            if index < corners {
                for (axis, (x, &(lo, hi))) in point.iter_mut().zip(bounds.as_slice()).enumerate() {
                    *x = if (index >> axis) & 1 == 0 { lo } else { hi };
                }
            } else {
                let mut generator = Xoshiro256StarStar::from_seed(work.seeds[index]);
                let mut admission = session;
                self.design.point(
                    index - corners,
                    bounds,
                    &mut SearchRandom::new(&mut generator, &mut admission),
                    point,
                )?;
            }
            validate_point(point, bounds, scioxide_core::argument!("design"))?;
            let value = session.invoke(EvaluationRole::Objective, || problem.eval(point))?;
            if !value.is_finite() {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Output,
                    "sample value must be finite",
                ));
            }
            staged.values[index] = value;
            staged.sampled = index + 1;
            if staged
                .best_sample
                .is_none_or(|best| value < staged.values[best])
            {
                staged.best_sample = Some(index);
                session.accept(PartialResultTrust::Admissible)?;
            }
        }
        let points = PointSet::from_rows(&staged.points[..total * d], d)?;
        let geometry_result =
            session.with_context(|ctx| Delaunay::build(&points, self.options.geometry, ctx))?;
        let complex = match geometry_result {
            Ok(complex) => complex,
            Err(error) => {
                match error.view() {
                    scioxide_core::SciOxideErrorView::ConvergenceFailure { reason, .. } => {
                        match reason {
                            scioxide_core::ConvergenceReason::IterationLimit => {
                                session.set_stopped(OptimizeTermination::IterationLimit)
                            }
                            scioxide_core::ConvergenceReason::ResourceLimit(BudgetKind::Node) => {
                                session.set_stopped(OptimizeTermination::NodeLimit)
                            }
                            _ => {}
                        }
                    }
                    _ => session.record_execution_stop(&error),
                }
                return Err(error);
            }
        };
        staged.simplices = complex.simplex_count();
        // Delaunay uses canonical site order and may merge exact duplicate samples.
        let unique = complex.points().len() / d;
        let lower = &mut work.lower[..unique];
        lower.fill(false);
        let parent = &mut work.parent[..unique];
        for (i, value) in parent.iter_mut().enumerate() {
            *value = i;
        }
        let representatives = complex.representatives();
        for simplex_index in 0..complex.simplex_count() {
            let simplex = complex.simplex(simplex_index)?;
            for (i, &a) in simplex.iter().enumerate() {
                for &b in &simplex[i + 1..] {
                    let va = staged.values[representatives[a]];
                    let vb = staged.values[representatives[b]];
                    if va > vb {
                        lower[a] = true;
                    } else if vb > va {
                        lower[b] = true;
                    } else {
                        unite(parent, a, b);
                    }
                }
            }
        }
        let plateau_lower = &mut work.plateau_lower[..unique];
        plateau_lower.fill(false);
        for (i, &flag) in lower.iter().enumerate() {
            if flag {
                plateau_lower[root(parent, i)] = true;
            }
        }
        for i in 0..unique {
            if root(parent, i) == i && !plateau_lower[i] {
                let mut chosen = representatives[i];
                for (j, &original) in representatives.iter().enumerate() {
                    if root(parent, j) == i {
                        chosen = chosen.min(original);
                    }
                }
                staged.candidates.push(chosen);
            }
        }
        let values = &staged.values;
        staged.candidates.sort_unstable_by(|&a, &b| {
            values[a]
                .partial_cmp(&values[b])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        work.complex = Some(complex);
        Ok(())
    }
    fn refine_sequential(
        &self,
        problem: &BoundedObjective<'_, S>,
        work: &mut ShgoStorage<S, F::Solver>,
        staged: &mut ShgoResult<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let d = work.bounds.len();
        let count = staged.candidates.len().min(work.pool.locals.len());
        for slot in 0..count {
            let vertex = staged.candidates[slot];
            session.check()?;
            let previous_best = staged
                .best_local
                .map(|best| staged.refinements[best].result.fun());
            let report = &mut staged.refinements[slot];
            report.vertex = vertex;
            let start = &staged.points[vertex * d..(vertex + 1) * d];
            let mut state = SolveState::new();
            let local = &mut work.pool.locals[slot];
            let identity = local_identity(vertex, d, work.token);
            let seed = work.seeds[vertex];
            let outcome =
                session.with_child(&mut state, identity, self.options.local_budgets, |ctx| {
                    let outcome =
                        local.minimize_into(problem, start, seed, &mut report.result, ctx);
                    report.counts = ctx.counts();
                    outcome
                });
            staged.refined = slot + 1;
            let local_stop = match &outcome {
                Ok(()) => Some(report.result.termination()),
                Err(error) => budget_termination(error),
            };
            if local_stop.is_some_and(|stop| {
                matches!(
                    stop,
                    OptimizeTermination::IterationLimit
                        | OptimizeTermination::ProposalLimit
                        | OptimizeTermination::SampleLimit
                        | OptimizeTermination::EvaluationLimit
                )
            }) {
                session.inherit_child_stop(local_stop)?;
            }
            match outcome {
                Ok(()) => {
                    validate_local(&report.result, &work.bounds)?;
                    report.completed = true;
                    if report.result.is_success()
                        && previous_best.is_none_or(|best| report.result.fun() < best)
                    {
                        staged.best_local = Some(slot);
                        session.accept(PartialResultTrust::Admissible)?;
                    }
                }
                Err(error) if recoverable(&error) => report.error = Some(error),
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
    fn refine_partitioned(
        &self,
        problem: &BoundedObjective<'_, S>,
        objective: &(dyn ObjectiveFunction<S> + Sync),
        work: &mut ShgoStorage<S, F::Solver>,
        staged: &mut ShgoResult<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        use scioxide_core::backend::{PartitionSlot, with_partitioned_solves};
        let count = staged.candidates.len().min(work.pool.locals.len());
        if count == 0 {
            return Ok(());
        }
        session.check()?;
        let d = work.bounds.len();
        let bounds = &work.bounds;
        let box_bounds = problem.bounds;
        let token = work.token;
        let points = &staged.points;
        let candidates = &staged.candidates;
        let seeds = &work.seeds;
        let reports = &mut staged.refinements[..count];
        let locals = &mut work.pool.locals[..count];
        let children = &mut work.children[..count];
        let options = &self.options;
        session.with_context(|ctx| {
            with_partitioned_solves(
                ctx,
                children,
                options.local_budgets,
                |capabilities, children| {
                    let plan = capabilities.plan(
                        options.execution,
                        count,
                        scioxide_core::PlanLimits {
                            work_per_item: core::num::NonZeroUsize::MIN,
                            auto_threshold: core::num::NonZeroUsize::MIN,
                            worker_limit: core::num::NonZeroUsize::MIN.saturating_add(63),
                        },
                    );
                    let mut slots: [_; 64] = core::array::from_fn(|_| PartitionSlot::empty());
                    let mut remaining_reports = reports;
                    let mut remaining_locals = locals;
                    let mut remaining_children = children;
                    for partition in plan.partitions() {
                        let length = partition.range().len();
                        let (r, rt) = remaining_reports.split_at_mut(length);
                        remaining_reports = rt;
                        let (l, lt) = remaining_locals.split_at_mut(length);
                        remaining_locals = lt;
                        let (c, tail) = remaining_children.split_at(length)?;
                        remaining_children = tail;
                        slots[partition.worker_index()] = PartitionSlot::new((r, l, c));
                    }
                    capabilities.execute_partitioned(
                        plan,
                        &slots[..plan.worker_count()],
                        &|partition, (reports, locals, children), _| {
                            let problem = BoundedObjective::new_sync(objective, box_bounds)?;
                            for (slot, ((report, local), child)) in partition
                                .range()
                                .zip(reports.iter_mut().zip(locals).zip(children))
                            {
                                let vertex = candidates[slot];
                                report.vertex = vertex;
                                let start = &points[vertex * d..(vertex + 1) * d];
                                let identity = local_identity(vertex, d, token);
                                report.counts = scioxide_core::SolveCounts::new();
                                let outcome = child.run(capabilities, identity, |ctx| {
                                    let outcome = local.minimize_into(
                                        &problem,
                                        start,
                                        seeds[vertex],
                                        &mut report.result,
                                        ctx,
                                    );
                                    report.counts = ctx.counts();
                                    outcome
                                });
                                match outcome {
                                    Ok(()) => {
                                        validate_local(&report.result, bounds)?;
                                        report.completed = true;
                                    }
                                    Err(error) if recoverable(&error) => report.error = Some(error),
                                    Err(error) => return Err(error),
                                }
                            }
                            Ok(())
                        },
                    )
                },
            )
        })??;
        staged.refined = count;
        for slot in 0..count {
            let report = &staged.refinements[slot];
            if report.completed
                && report.result.is_success()
                && staged
                    .best_local
                    .is_none_or(|best| report.result.fun() < staged.refinements[best].result.fun())
            {
                staged.best_local = Some(slot);
                session.accept(PartialResultTrust::Admissible)?;
            }
        }
        for stop in [
            OptimizeTermination::EvaluationLimit,
            OptimizeTermination::IterationLimit,
            OptimizeTermination::ProposalLimit,
            OptimizeTermination::SampleLimit,
        ] {
            session.inherit_child_stop(Some(stop))?;
        }
        Ok(())
    }
}

fn finish<S: RealScalar>(
    outcome: Result<(), SciOxideError>,
    staged: &mut ShgoResult<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    staged.counts = session.accounting()?;
    match outcome {
        Ok(()) => {
            staged.termination = session
                .termination()?
                .unwrap_or(OptimizeTermination::SearchBudgetCompleted);
        }
        Err(error)
            if session.stopped().is_some()
                && (matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
                ) || matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
                )) =>
        {
            staged.termination = session.stopped().unwrap_or(OptimizeTermination::Cancelled);
        }
        Err(error) => return Err(error),
    }
    Ok(())
}

fn root(parent: &[usize], mut index: usize) -> usize {
    while parent[index] != index {
        index = parent[index];
    }
    index
}
fn unite(parent: &mut [usize], a: usize, b: usize) {
    let a = root(parent, a);
    let b = root(parent, b);
    if a < b {
        parent[b] = a;
    } else if b < a {
        parent[a] = b;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    #[test]
    fn corner_count_rejects_unrepresentable_boxes() {
        assert_eq!(super::corner_count(3).unwrap(), 8);
        assert!(super::corner_count(usize::BITS as usize).is_err());
    }
}
