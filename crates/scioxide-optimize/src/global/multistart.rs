//! Multistart search, its options, result, workspace and shared finish.
use alloc::vec::Vec;

use scioxide_core::backend::{PartitionedSolve, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AttemptKind, AttemptRecord, Budget, Capacity, ExecutionPolicy, NumericScalar,
    PartialResultTrust, RealScalar, SciOxideError, SolveBudgets, SolveContext, SolveIdentity,
    SolveScope,
};
use scioxide_random::{SeedableRng, Xoshiro256StarStar};

use super::local::{
    LocalPool, LocalSolverFactory, SearchLocalSolver, budget_termination, local_identity,
    recoverable, run_local, validate_local,
};
use super::{StartDesign, shared_objective};
use crate::attempt::{Accepted, AcceptedState, Entry, Shell, Staged};
use crate::context::{self, SolveSession};
use crate::validation::invalid;
use crate::{
    BoundedObjective, FiniteBounds, MinimizerAccepted, MinimizerProgress, ObjectiveFunction,
    OptimizeResult, OptimizeTermination, SearchRandom,
};

/// Configures [`MultiStart`], [`BasinHopping`](crate::BasinHopping)
/// and [`DualAnnealing`](crate::DualAnnealing): the number of starts,
/// the seed, limits, basin tolerances and execution policy.
///
/// The setters do not check their arguments; the methods check the basin
/// tolerances when they are created. See [Termination](crate#termination) for
/// how a search ends.
#[derive(Debug, Clone)]
pub struct GlobalSearchOptions<S: RealScalar> {
    starts: Budget,
    seed: u64,
    time_limit: Option<core::time::Duration>,
    local_budget: SolveBudgets,
    coordinate_tolerance: S,
    objective_tolerance: S,
    execution: ExecutionPolicy,
}
impl<S: RealScalar> Default for GlobalSearchOptions<S> {
    /// Returns ten starts, seed 42, no limit on each local solve, basin
    /// tolerances `√ε` and sequential execution.
    fn default() -> Self {
        Self {
            starts: Budget::ONE.saturating_add(9),
            seed: 42,
            time_limit: None,
            local_budget: SolveBudgets::unbounded(),
            coordinate_tolerance: S::SQRT_EPSILON,
            objective_tolerance: S::SQRT_EPSILON,
            execution: ExecutionPolicy::Sequential,
        }
    }
}
impl<S: RealScalar> GlobalSearchOptions<S> {
    /// Sets the number of starts. For basin hopping and dual annealing it is
    /// the number of chain steps, including the initial point.
    pub const fn with_starts(mut self, starts: Budget) -> Self {
        self.starts = starts;
        self
    }
    /// Sets the seed from which every start's random stream is derived.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    /// Sets a time limit for the whole search. It needs a clock in the
    /// [execution context](scioxide_core#execution-contexts).
    pub const fn with_time_limit(mut self, limit: core::time::Duration) -> Self {
        self.time_limit = Some(limit);
        self
    }
    /// Sets the limits of each local solve. The limits of the whole search
    /// still apply.
    pub const fn with_local_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.local_budget = budgets;
        self
    }
    /// Sets the relative tolerances under which two successful local results
    /// count as the same basin.
    ///
    /// Two results are in the same basin when their objective values and every
    /// coordinate pair `x`, `y` satisfy `|x/s − y/s| ≤ tol`, with
    /// `s = max(1, |x|, |y|)`. Methods reject NaN, infinite or nonpositive
    /// tolerances.
    pub const fn with_basin_tolerances(mut self, coordinate: S, objective: S) -> Self {
        self.coordinate_tolerance = coordinate;
        self.objective_tolerance = objective;
        self
    }
    /// Sets whether starts run one after another or in parallel.
    ///
    /// With [`ExecutionPolicy::Sequential`], the starts draw from the search's
    /// limits in start order. Any other policy splits each finite remaining
    /// limit into fixed per-start shares before the starts run, and needs a
    /// problem built with [`BoundedObjective::new_sync`]. Unused shares are not
    /// passed to other starts, so with finite limits the result can differ
    /// from sequential execution; it does not depend on the number of workers.
    /// Basin hopping and dual annealing accept only the sequential policy.
    pub const fn with_execution(mut self, execution: ExecutionPolicy) -> Self {
        self.execution = execution;
        self
    }
    /// Returns the seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }
    /// Returns the time limit of the whole search.
    pub const fn time_limit(&self) -> Option<core::time::Duration> {
        self.time_limit
    }
    /// Returns the number of starts.
    pub const fn starts(&self) -> Budget {
        self.starts
    }
    /// Returns the limits of each local solve.
    pub const fn local_budgets(&self) -> SolveBudgets {
        self.local_budget
    }
    /// Returns the coordinate and objective basin tolerances.
    pub const fn basin_tolerances(&self) -> (S, S) {
        (self.coordinate_tolerance, self.objective_tolerance)
    }
    /// Returns the execution policy.
    pub const fn execution(&self) -> ExecutionPolicy {
        self.execution
    }
    /// Reject NaN, infinite or nonpositive basin tolerances.
    pub(super) fn validate(&self) -> Result<(), SciOxideError> {
        let (coordinate, objective) = self.basin_tolerances();
        if !coordinate.is_finite()
            || coordinate <= S::ZERO
            || !objective.is_finite()
            || objective <= S::ZERO
        {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "basin tolerances must be finite and positive",
            ));
        }
        Ok(())
    }
}

/// Reports one start of a multistart search or one step of a chain.
///
/// A failed local solve keeps its error here; it never counts as a success.
#[derive(Debug)]
pub struct SearchStartReport<S: RealScalar> {
    pub(super) seed: u64,
    pub(super) start: Vec<S>,
    pub(super) candidate: OptimizeResult<S>,
    pub(super) refinement: OptimizeResult<S>,
    pub(super) local_done: bool,
    pub(super) local_in_refinement: bool,
    pub(super) temperature: Option<S>,
    pub(super) reannealed: bool,
    pub(super) has_candidate: bool,
    pub(super) error: Option<SciOxideError>,
    pub(super) counts: scioxide_core::SolveCounts,
    pub(super) basin: Option<usize>,
    pub(super) accepted: bool,
}
impl<S: RealScalar> SearchStartReport<S> {
    /// Returns the seed derived for this start.
    pub const fn seed(&self) -> u64 {
        self.seed
    }
    /// Returns the start point.
    pub fn start(&self) -> &[S] {
        &self.start
    }
    /// Returns the result of this start, if the start produced a finite point
    /// inside the box.
    pub fn candidate(&self) -> Option<&OptimizeResult<S>> {
        self.has_candidate.then_some(&self.candidate)
    }
    /// Returns the local solver's result, if a local solve finished.
    ///
    /// For dual annealing this can differ from [`Self::candidate`], which keeps
    /// the lower of the annealing sample and the local result.
    pub fn local_result(&self) -> Option<&OptimizeResult<S>> {
        self.local_done.then_some(if self.local_in_refinement {
            &self.refinement
        } else {
            &self.candidate
        })
    }
    /// Returns the temperature of the chain step; `None` for multistart.
    pub const fn temperature(&self) -> Option<S> {
        self.temperature
    }
    /// Returns whether this step restarted the annealing temperature and
    /// position.
    pub const fn reannealed(&self) -> bool {
        self.reannealed
    }
    /// Returns the error of this start's local solve, if any.
    pub const fn error(&self) -> Option<&SciOxideError> {
        self.error.as_ref()
    }
    /// Returns the evaluations and iterations of this start, including failed
    /// callbacks.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the index of the first successful start in the same basin.
    ///
    /// Basins are grouped by the tolerances of
    /// [`GlobalSearchOptions::with_basin_tolerances`]; the grouping is a
    /// heuristic, not a proof that the points share a basin of attraction.
    pub const fn basin(&self) -> Option<usize> {
        self.basin
    }
    /// Returns whether this point became the chain's current point, or for
    /// multistart, the best point so far.
    pub const fn accepted(&self) -> bool {
        self.accepted
    }

    fn reserved(dimension: usize) -> Result<Self, SciOxideError> {
        let mut candidate = OptimizeResult::empty();
        candidate.try_reserve(dimension)?;
        let mut refinement = OptimizeResult::empty();
        refinement.try_reserve(dimension)?;
        Ok(Self {
            seed: 0,
            start: try_filled(dimension, S::ZERO, scioxide_core::quantity!("search_start"))?,
            candidate,
            refinement,
            local_done: false,
            local_in_refinement: false,
            temperature: None,
            reannealed: false,
            has_candidate: false,
            error: None,
            counts: scioxide_core::SolveCounts::new(),
            basin: None,
            accepted: false,
        })
    }
    fn fits(&self, dimension: usize) -> bool {
        self.start.len() == dimension
            && self.candidate.fits(dimension)
            && self.refinement.fits(dimension)
    }
    /// Copy into storage that already fits the source.
    fn copy_from(&mut self, source: &Self) -> Result<(), SciOxideError> {
        self.seed = source.seed;
        self.start.copy_from_slice(&source.start);
        self.candidate.try_copy_from(&source.candidate)?;
        self.refinement.try_copy_from(&source.refinement)?;
        self.local_done = source.local_done;
        self.local_in_refinement = source.local_in_refinement;
        self.temperature = source.temperature;
        self.reannealed = source.reannealed;
        self.has_candidate = source.has_candidate;
        self.error.clone_from(&source.error);
        self.counts = source.counts;
        self.basin = source.basin;
        self.accepted = source.accepted;
        Ok(())
    }
}

/// Result of a multistart, basin-hopping or dual-annealing search: a report
/// per start, the best point found and the termination.
///
/// Start with [`Self::empty`]; the search sizes it on first use.
#[derive(Debug)]
pub struct GlobalSearchResult<S: RealScalar> {
    pub(super) reports: Vec<SearchStartReport<S>>,
    dimension: usize,
    pub(super) attempted: usize,
    pub(super) best: Option<usize>,
    pub(super) successful: Option<usize>,
    pub(super) termination: OptimizeTermination,
    pub(super) counts: scioxide_core::SolveCounts,
}
impl<S: RealScalar> Default for GlobalSearchResult<S> {
    fn default() -> Self {
        Self::empty()
    }
}
impl<S: RealScalar> GlobalSearchResult<S> {
    /// Returns an empty result without allocating.
    pub const fn empty() -> Self {
        Self {
            reports: Vec::new(),
            dimension: 0,
            attempted: 0,
            best: None,
            successful: None,
            termination: OptimizeTermination::NotStarted,
            counts: scioxide_core::SolveCounts::new(),
        }
    }
    /// Returns the reports of the starts that ran, in start order.
    pub fn starts(&self) -> &[SearchStartReport<S>] {
        &self.reports[..self.attempted]
    }
    /// Returns the number of requested starts.
    pub fn requested(&self) -> usize {
        self.reports.len()
    }
    /// Returns the result with the lowest objective value, whether or not its
    /// local solve succeeded.
    pub fn best(&self) -> Option<&OptimizeResult<S>> {
        self.best.map(|i| &self.reports[i].candidate)
    }
    /// Returns the lowest result whose local solve succeeded. It is not
    /// proven to be the global minimum.
    pub fn best_successful(&self) -> Option<&OptimizeResult<S>> {
        self.successful.and_then(|i| self.reports[i].local_result())
    }
    /// Returns why the search stopped. Running all starts gives
    /// [`OptimizeTermination::SearchBudgetCompleted`], which is not a success.
    pub const fn termination(&self) -> OptimizeTermination {
        self.termination
    }
    /// Returns the evaluations and iterations of the whole search.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the number of starts that produced a result.
    pub fn completed(&self) -> usize {
        self.starts().iter().filter(|r| r.has_candidate).count()
    }
    /// Returns the number of starts whose local solve succeeded.
    pub fn successes(&self) -> usize {
        self.starts()
            .iter()
            .filter(|r| r.local_result().is_some_and(OptimizeResult::is_success))
            .count()
    }
    /// Returns the number of starts whose local solve returned an error.
    pub fn failures(&self) -> usize {
        self.starts().iter().filter(|r| r.error.is_some()).count()
    }
    /// Returns the number of distinct basins among the successful starts.
    pub fn basins(&self) -> usize {
        self.starts()
            .iter()
            .enumerate()
            .filter(|(i, r)| r.basin == Some(*i))
            .count()
    }
    fn fits(&self, starts: usize, dimension: usize) -> bool {
        self.dimension == dimension
            && self.reports.len() == starts
            && self.reports.iter().all(|report| report.fits(dimension))
    }
    /// Reserve history for `starts` attempts in `dimension`; on failure the existing storage stays in place.
    pub(super) fn try_reserve(
        &mut self,
        starts: usize,
        dimension: usize,
    ) -> Result<(), SciOxideError> {
        if self.fits(starts, dimension) {
            return Ok(());
        }
        checked_mul(
            starts,
            dimension,
            scioxide_core::quantity!("search_history"),
        )?;
        let mut reports = try_with_capacity(starts, scioxide_core::quantity!("search_reports"))?;
        for _ in 0..starts {
            reports.push(SearchStartReport::reserved(dimension)?);
        }
        *self = Self {
            reports,
            dimension,
            ..Self::empty()
        };
        Ok(())
    }
    pub(super) fn reset(&mut self) {
        self.attempted = 0;
        self.best = None;
        self.successful = None;
        self.termination = OptimizeTermination::NotStarted;
        self.counts = scioxide_core::SolveCounts::new();
        for r in &mut self.reports {
            r.has_candidate = false;
            r.local_done = false;
            r.local_in_refinement = false;
            r.temperature = None;
            r.reannealed = false;
            r.error = None;
            r.basin = None;
            r.accepted = false;
            r.counts = scioxide_core::SolveCounts::new();
        }
    }
    /// Seed every start before a search.
    pub(super) fn seed(&mut self, seeds: &[u64]) {
        for (report, &seed) in self.reports.iter_mut().zip(seeds) {
            report.seed = seed;
        }
    }
    pub(super) fn record(&mut self, index: usize, options: &GlobalSearchOptions<S>) {
        let candidate = &self.reports[index].candidate;
        if !self.reports[index].has_candidate {
            return;
        }
        if self
            .best
            .is_none_or(|i| candidate.fun() < self.reports[i].candidate.fun())
        {
            self.best = Some(index);
        }
        if let Some(local) = self.reports[index]
            .local_result()
            .filter(|r| r.is_success())
        {
            if self.successful.is_none_or(|i| {
                self.reports[i]
                    .local_result()
                    .is_none_or(|old| local.fun() < old.fun())
            }) {
                self.successful = Some(index);
            }
            let basin = (0..index)
                .find(|&i| {
                    self.reports[i].basin == Some(i)
                        && self.reports[i]
                            .local_result()
                            .is_some_and(|other| same_basin(local, other, options))
                })
                .unwrap_or(index);
            self.reports[index].basin = Some(basin);
        }
    }
}
impl<S: RealScalar> Staged for GlobalSearchResult<S> {
    fn reset(&mut self) {
        Self::reset(self);
    }
    fn publish(&mut self, source: &Self) -> Result<(), SciOxideError> {
        if !self.fits(source.reports.len(), source.dimension) {
            let mut fresh = Self::empty();
            fresh.try_reserve(source.reports.len(), source.dimension)?;
            fresh.publish(source)?;
            *self = fresh;
            return Ok(());
        }
        for (report, source) in self.reports.iter_mut().zip(&source.reports) {
            report.copy_from(source)?;
        }
        self.attempted = source.attempted;
        self.best = source.best;
        self.successful = source.successful;
        self.termination = source.termination;
        self.counts = source.counts;
        Ok(())
    }
    /// Heuristic searches never report success.
    fn succeeded(&self) -> bool {
        false
    }
}
impl<S: RealScalar> Accepted<S> for GlobalSearchResult<S> {
    fn accepted(&self) -> Option<AcceptedState<'_, S>> {
        let best = &self.reports[self.best?].candidate;
        Some(AcceptedState {
            point: best.x(),
            value: best.fun(),
            gradient: None,
            iterations: self.attempted,
        })
    }
}
fn same_basin<S: RealScalar>(
    a: &OptimizeResult<S>,
    b: &OptimizeResult<S>,
    options: &GlobalSearchOptions<S>,
) -> bool {
    let close = |x: S, y: S, tolerance: S| {
        let scale = S::ONE.max(x.abs()).max(y.abs());
        (x / scale - y / scale).abs() <= tolerance
    };
    close(a.fun(), b.fun(), options.objective_tolerance)
        && a.x()
            .iter()
            .zip(b.x())
            .all(|(&x, &y)| close(x, y, options.coordinate_tolerance))
}

/// Workspace size of a multistart, basin-hopping or dual-annealing search.
///
/// It borrows the method's local factory, which creates one solver per start.
#[derive(Debug)]
pub struct GlobalSearchWorkspaceRequirements<'a, F> {
    factory: &'a F,
    pub(super) token: u128,
    dimension: usize,
    starts: usize,
    seed: u64,
}
impl<F> GlobalSearchWorkspaceRequirements<'_, F> {
    /// Returns the number of variables.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    /// Returns the number of starts, each with its own local solver.
    pub const fn starts(&self) -> usize {
        self.starts
    }
}
pub(super) fn requirements<'a, S: RealScalar, F: core::fmt::Debug>(
    options: &GlobalSearchOptions<S>,
    factory: &'a F,
    dimension: usize,
) -> Result<GlobalSearchWorkspaceRequirements<'a, F>, SciOxideError> {
    checked_mul(
        options.starts.get(),
        dimension,
        scioxide_core::quantity!("search_history"),
    )?;
    Ok(GlobalSearchWorkspaceRequirements {
        factory,
        token: context::options_token(factory),
        dimension,
        starts: options.starts.get(),
        seed: options.seed,
    })
}

/// Prepared local states, seeds, child slots, box copy and staged history.
pub(super) struct SearchStorage<S: RealScalar, L> {
    pub(super) bounds: FiniteBounds<S>,
    pub(super) seeds: Vec<u64>,
    seed: u64,
    pub(super) pool: LocalPool<L>,
    pub(super) children: Vec<PartitionedSolve>,
    pub(super) token: u128,
}

/// Workspace of a multistart, basin-hopping or dual-annealing search: the
/// local solvers, per-start seeds and the record of the last call.
///
/// Created by [`MultiStart::workspace`], [`crate::BasinHopping::workspace`]
/// or [`crate::DualAnnealing::workspace`]. A call with another dimension,
/// number of starts, seed or local solver configuration reallocates it.
pub struct GlobalSearchWorkspace<S: RealScalar, L> {
    pub(super) storage: SearchStorage<S, L>,
    pub(super) staging: GlobalSearchResult<S>,
    pub(super) shell: Shell<MinimizerProgress<S>>,
}
impl<S: RealScalar, L> core::fmt::Debug for GlobalSearchWorkspace<S, L> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("GlobalSearchWorkspace")
            .field("starts", &self.storage.seeds.len())
            .field("dimension", &self.staging.dimension)
            .finish_non_exhaustive()
    }
}
impl<S: RealScalar, L> Default for GlobalSearchWorkspace<S, L> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar, L> GlobalSearchWorkspace<S, L> {
    /// Creates an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            storage: SearchStorage {
                bounds: FiniteBounds::unassigned(),
                seeds: Vec::new(),
                seed: 0,
                pool: LocalPool::new(),
                children: Vec::new(),
                token: 0,
            },
            staging: GlobalSearchResult::empty(),
            shell: Shell::new(),
        }
    }
    /// Checks whether the workspace already fits `requirements`.
    ///
    /// # Errors
    ///
    /// Never returns an error; the `Result` matches the other workspaces.
    pub fn check<F>(
        &self,
        requirements: &GlobalSearchWorkspaceRequirements<'_, F>,
    ) -> Result<Capacity, SciOxideError> {
        let storage = &self.storage;
        Ok(
            if storage.pool.fits(
                requirements.starts,
                requirements.dimension,
                requirements.token,
            ) && storage.seeds.len() == requirements.starts
                && storage.seed == requirements.seed
                && storage.children.len() == requirements.starts
                && storage.bounds.fits(requirements.dimension)
                && self
                    .staging
                    .fits(requirements.starts, requirements.dimension)
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
        requirements: &GlobalSearchWorkspaceRequirements<'_, F>,
    ) -> Result<(), SciOxideError>
    where
        S: NumericScalar<Real = S>,
        F: LocalSolverFactory<S, Solver = L>,
    {
        if self.check(requirements)? == Capacity::Sufficient {
            return Ok(());
        }
        let storage = &mut self.storage;
        storage.pool.prepare(
            requirements.factory,
            requirements.starts,
            requirements.dimension,
            requirements.token,
        )?;
        if storage.seeds.len() != requirements.starts || storage.seed != requirements.seed {
            storage.seeds = scioxide_random::seed_sequence(requirements.seed, requirements.starts)?;
            storage.seed = requirements.seed;
        }
        if storage.children.len() != requirements.starts {
            let mut children = try_with_capacity(
                requirements.starts,
                scioxide_core::quantity!("search_child_contexts"),
            )?;
            children.resize_with(requirements.starts, PartitionedSolve::default);
            storage.children = children;
        }
        storage.bounds.try_reserve(requirements.dimension)?;
        storage.token = requirements.token;
        self.staging
            .try_reserve(requirements.starts, requirements.dimension)
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

/// Validate the problem box into prepared storage before an attempt.
pub(super) fn bind<S, L, F>(
    problem: &BoundedObjective<'_, S>,
    requirements: &GlobalSearchWorkspaceRequirements<'_, F>,
    workspace: &mut GlobalSearchWorkspace<S, L>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: LocalSolverFactory<S, Solver = L>,
{
    workspace.try_reserve(requirements)?;
    workspace.storage.bounds.assign(problem.bounds)
}

/// Identity of a global epoch: method, dimension and configuration.
pub(super) fn identity(
    algorithm: scioxide_core::AlgorithmId,
    dimension: usize,
    options: &impl core::fmt::Debug,
    local: u128,
) -> SolveIdentity {
    SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        algorithm,
        dimension as u128,
        context::options_token(options) ^ local.rotate_left(1),
    )
}

/// Minimizes an objective over a finite box by running a local solver from
/// several start points.
///
/// A [`StartDesign`](crate::StartDesign) generates the start points:
/// [`UniformStartDesign`](crate::UniformStartDesign) draws them uniformly,
/// [`BorrowedStartDesign`](crate::BorrowedStartDesign) takes them from you, and
/// a [`QmcPlan`](scioxide_random::QmcPlan) uses a quasi-random sequence. A
/// [`LocalSolverFactory`](crate::LocalSolverFactory), such as
/// [`PowellLocalFactory`](crate::PowellLocalFactory), refines each start. The
/// starts are independent, so they can run in parallel. The result reports
/// every start, groups the successful ones into basins, and keeps the lowest
/// point. The cost is the sum of the local solves.
///
/// The search does not prove that it found the global minimum, and running
/// all starts ends with
/// [`OptimizeTermination::SearchBudgetCompleted`], which is not a success. See
/// [Randomness and parallelism](crate#randomness-and-parallelism) for seeds
/// and execution policies.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Bounds, Budget, ExecutionContext, SciOxideError};
/// use scioxide_optimize::{
///     BoundedObjective, GlobalSearchOptions, GlobalSearchResult, MultiStart, ObjectiveFn,
///     PowellLocalFactory, UniformStartDesign,
/// };
/// # fn main() -> Result<(), SciOxideError> {
/// let objective = ObjectiveFn::new(1, |x: &[f64]| Ok((x[0] - 0.5).powi(2)));
/// let bounds = Bounds::new(vec![(Some(-2.0), Some(2.0))])?;
/// let problem = BoundedObjective::new(&objective, &bounds)?;
/// let method = MultiStart::new(
///     GlobalSearchOptions::default().with_starts(Budget::try_from(3)?),
///     UniformStartDesign,
///     PowellLocalFactory::default(),
/// )?;
/// let mut workspace = method.workspace(1)?;
/// let mut result = GlobalSearchResult::empty();
/// method.minimize_into(&problem, &mut result, &mut workspace, ExecutionContext::sequential())?;
/// assert!(result.best().unwrap().fun() < 1e-8);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MultiStart<S: RealScalar, Design, Factory> {
    options: GlobalSearchOptions<S>,
    design: Design,
    factory: Factory,
}
impl<S, D, F> MultiStart<S, D, F>
where
    S: RealScalar + NumericScalar<Real = S>,
    D: StartDesign<S>,
    F: LocalSolverFactory<S>,
{
    /// Creates a method from options, a start design and a local solver factory.
    ///
    /// # Errors
    ///
    /// `InvalidInput` if a basin tolerance is NaN, infinite or not positive.
    pub fn new(
        options: GlobalSearchOptions<S>,
        design: D,
        factory: F,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        Ok(Self {
            options,
            design,
            factory,
        })
    }
    /// Returns the options.
    pub const fn options(&self) -> &GlobalSearchOptions<S> {
        &self.options
    }
    /// Returns the workspace size for `dimension` variables.
    ///
    /// # Errors
    ///
    /// Returns the design's error if it cannot supply the requested starts in
    /// `dimension` variables, and `SizeOverflow` if the size overflows.
    pub fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<GlobalSearchWorkspaceRequirements<'_, F>, SciOxideError> {
        self.design.validate(dimension, self.options.starts.get())?;
        requirements(&self.options, &self.factory, dimension)
    }
    /// Allocates a workspace for `dimension` variables.
    ///
    /// # Errors
    ///
    /// As [`Self::workspace_requirements`], plus the local factory's errors and
    /// `AllocationFailure`.
    pub fn workspace(
        &self,
        dimension: usize,
    ) -> Result<GlobalSearchWorkspace<S, F::Solver>, SciOxideError> {
        let mut workspace = GlobalSearchWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }
    /// Runs every start and writes the reports and best point to `output`.
    ///
    /// A local solve that fails with `ConvergenceFailure`,
    /// `NumericalInstability` or `Singular` is recorded in its start's report,
    /// and the search continues. When a limit, cancellation or a deadline
    /// stops the search, `output` holds the starts that ran and the
    /// termination names the reason.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a variable lacks a lower or upper bound or has
    ///   `lo ≥ hi`, the design
    ///   does not fit the problem, a local solver returns a point outside the
    ///   box or a non-finite value, or the policy is not sequential and the
    ///   problem was not built with [`BoundedObjective::new_sync`].
    /// - `CapabilityUnavailable` if a time limit is set and the context has no
    ///   clock.
    /// - Errors of the objective, of a local solver other than those recorded
    ///   per start, and of the executor.
    pub fn minimize_into<'scope, 'capability>(
        &self,
        problem: &BoundedObjective<'_, S>,
        output: &mut GlobalSearchResult<S>,
        workspace: &mut GlobalSearchWorkspace<S, F::Solver>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        let requirements = self.workspace_requirements(problem.dim())?;
        let entry = Entry::Call {
            identity: identity(
                scioxide_core::algorithm_id!("optimize", "MultiStart", 1),
                problem.dim(),
                &self.options,
                requirements.token,
            ),
            budgets: SolveBudgets::unbounded(),
        };
        bind(problem, &requirements, workspace)?;
        let shared = shared_objective(problem, self.options.execution)?;
        crate::attempt::preflight(&scope, self.options.time_limit)?;
        super::attempt(
            &mut workspace.shell,
            &mut workspace.storage,
            &mut workspace.staging,
            output,
            scope,
            entry,
            |context, storage, staged| match shared {
                None => self.sequential(problem, storage, staged, context),
                Some(objective) => self.partitioned(problem, objective, storage, staged, context),
            },
        )
    }
    /// Runs the search sequentially with a new workspace and returns the
    /// result.
    ///
    /// # Errors
    ///
    /// As [`Self::workspace`] and [`Self::minimize_into`].
    pub fn minimize(
        &self,
        problem: &BoundedObjective<'_, S>,
    ) -> Result<GlobalSearchResult<S>, SciOxideError> {
        let mut workspace = self.workspace(problem.dim())?;
        let mut output = GlobalSearchResult::empty();
        self.minimize_into(
            problem,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Form one indexed start from its own seeded stream and charged samples.
    fn design_point(
        &self,
        index: usize,
        report: &mut SearchStartReport<S>,
        bounds: &FiniteBounds<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        session.proposal()?;
        let mut generator = Xoshiro256StarStar::from_seed(report.seed);
        let mut admission = session;
        self.design.point(
            index,
            bounds,
            &mut SearchRandom::new(&mut generator, &mut admission),
            &mut report.start,
        )?;
        validate_point(&report.start, bounds, scioxide_core::argument!("design"))
    }

    fn sequential(
        &self,
        problem: &BoundedObjective<'_, S>,
        storage: &mut SearchStorage<S, F::Solver>,
        staged: &mut GlobalSearchResult<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if let Some(limit) = self.options.time_limit {
            context.limit_time(limit)?;
        }
        staged.seed(&storage.seeds);
        let session = SolveSession::new(context, scioxide_core::argument!("scope"));
        let outcome = (|| {
            for index in 0..self.options.starts.get() {
                self.design_point(index, &mut staged.reports[index], &storage.bounds, &session)?;
                staged.attempted = index + 1;
                run_local(
                    &mut storage.pool.locals[index],
                    &mut staged.reports[index],
                    index,
                    problem,
                    &storage.bounds,
                    self.options.local_budget,
                    storage.token,
                    &session,
                )?;
                staged.record(index, &self.options);
                if staged.best == Some(index) {
                    staged.reports[index].accepted = true;
                    session.accept(PartialResultTrust::Admissible)?;
                }
            }
            Ok(())
        })();
        finish(outcome, staged, &session)
    }

    /// Independent starts with stable per-start budget shares and start-index commits.
    ///
    /// Every finite ancestor remainder is divided across starts before dispatch;
    /// unused shares remain with the parent and are not transferred between
    /// starts, so finite-budget results can differ from the sequential pool.
    /// Worker count does not change the shares. Child work already counted
    /// stays counted even if the executor fails after partial dispatch.
    fn partitioned(
        &self,
        problem: &BoundedObjective<'_, S>,
        objective: &(dyn ObjectiveFunction<S> + Sync),
        storage: &mut SearchStorage<S, F::Solver>,
        staged: &mut GlobalSearchResult<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        use scioxide_core::backend::{PartitionSlot, with_partitioned_solves};
        if let Some(limit) = self.options.time_limit {
            context.limit_time(limit)?;
        }
        staged.seed(&storage.seeds);
        // All designs are formed in index order before workers see any local state.
        {
            let session = SolveSession::new(context, scioxide_core::argument!("scope"));
            let outcome = (|| {
                for (index, report) in staged.reports.iter_mut().enumerate() {
                    self.design_point(index, report, &storage.bounds, &session)?;
                }
                Ok(())
            })();
            if outcome.is_err() {
                return finish(outcome, staged, &session);
            }
        }
        let options = &self.options;
        let bounds = &storage.bounds;
        let box_bounds = problem.bounds;
        let token = storage.token;
        let result = with_partitioned_solves(
            context,
            &mut storage.children,
            options.local_budget,
            |capabilities, mut children| {
                let limit = core::num::NonZeroUsize::MIN.saturating_add(63);
                let plan = capabilities.plan(
                    options.execution,
                    children.len(),
                    scioxide_core::PlanLimits {
                        work_per_item: core::num::NonZeroUsize::MIN,
                        auto_threshold: core::num::NonZeroUsize::MIN,
                        worker_limit: limit,
                    },
                );
                let mut slots: [_; 64] = core::array::from_fn(|_| PartitionSlot::empty());
                let mut reports = staged.reports.as_mut_slice();
                let mut locals = storage.pool.locals.as_mut_slice();
                for partition in plan.partitions() {
                    let length = partition.range().len();
                    let (r, rt) = reports.split_at_mut(length);
                    reports = rt;
                    let (l, lt) = locals.split_at_mut(length);
                    locals = lt;
                    let (c, tail) = children.split_at(length)?;
                    children = tail;
                    slots[partition.worker_index()] = PartitionSlot::new((r, l, c));
                }
                capabilities.execute_partitioned(
                    plan,
                    &slots[..plan.worker_count()],
                    &|partition, (reports, locals, children), _| {
                        let problem = BoundedObjective::new_sync(objective, box_bounds)?;
                        for (index, ((report, local), child)) in partition
                            .range()
                            .zip(reports.iter_mut().zip(locals).zip(children))
                        {
                            let identity = local_identity(index, bounds.len(), token);
                            report.counts = scioxide_core::SolveCounts::new();
                            let result = child.run(capabilities, identity, |context| {
                                let result = local.minimize_into(
                                    &problem,
                                    &report.start,
                                    report.seed,
                                    &mut report.candidate,
                                    context,
                                );
                                report.counts = context.counts();
                                result
                            });
                            match result {
                                Ok(()) => {
                                    validate_local(&report.candidate, bounds)?;
                                    report.has_candidate = true;
                                    report.local_done = true;
                                }
                                Err(error) if recoverable(&error) => report.error = Some(error),
                                Err(error) => return Err(error),
                            }
                        }
                        Ok(())
                    },
                )
            },
        );
        let session = SolveSession::new(context, scioxide_core::argument!("scope"));
        if result.is_ok() {
            staged.attempted = options.starts.get();
            for index in 0..options.starts.get() {
                staged.record(index, options);
                if staged.best == Some(index) {
                    staged.reports[index].accepted = true;
                    session.accept(PartialResultTrust::Admissible)?;
                }
            }
            for stop in [
                OptimizeTermination::EvaluationLimit,
                OptimizeTermination::IterationLimit,
                OptimizeTermination::ProposalLimit,
            ] {
                session.inherit_child_stop(Some(stop))?;
            }
        }
        finish(result, staged, &session)
    }
}

/// Check a point supplied through `argument` against the search box.
pub(super) fn validate_point<S: RealScalar>(
    x: &[S],
    bounds: &FiniteBounds<S>,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if x.len() != bounds.len() {
        return Err(invalid(argument, "point dimension differs"));
    }
    if !bounds.contains(x) {
        return Err(invalid(
            argument,
            "point must be finite and inside search bounds",
        ));
    }
    Ok(())
}

/// Close the staged history: budget and execution stops keep the work done so far.
pub(super) fn finish<S: RealScalar>(
    outcome: Result<(), SciOxideError>,
    staged: &mut GlobalSearchResult<S>,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    staged.counts = session.accounting()?;
    staged.termination = match outcome {
        Ok(()) => session
            .termination()?
            .unwrap_or(OptimizeTermination::SearchBudgetCompleted),
        Err(error) => match session.stopped() {
            Some(stop)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
                ) || budget_termination(&error).is_some() =>
            {
                stop
            }
            _ => return Err(error),
        },
    };
    Ok(())
}
