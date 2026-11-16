//! Differential evolution over a finite box.

use alloc::vec::Vec;
use core::time::Duration;

use scioxide_core::backend::checked_mul;
use scioxide_core::{
    AttemptKind, Budget, BudgetKind, ExecutionPolicy, NumericScalar, RealScalar, ResidualTolerance,
    SciOxideError, SolveBudgets, SolveScope, checked_affine, checked_difference,
};
use scioxide_random::{BitGenerator, SeedableRng, Xoshiro256StarStar};

use super::shared_objective;
use crate::attempt::{Accepted, AcceptedState, Entry, Shape};
use crate::context::{Observation, SolveSession};
use crate::geometry::{normalized_coordinate, repair};
use crate::result::{OptimalityDiagnostics, OptimizeResult, OptimizeTermination};
use crate::traits::ObjectiveFunction;
use crate::{BoundedObjective, FiniteBounds, SearchRandom};

/// Default population size multiplier (per dimension).
const DEFAULT_POP_PER_DIM: usize = 15;

/// Minimum population size.
const MIN_POPULATION: usize = 4;

/// Selects the mutation strategy of [`DifferentialEvolution`].
///
/// For member `xᵢ`, `r₁, …, r₅` are distinct random members other than `i`,
/// `x_best` is the best member and `F` is the mutation factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeStrategy {
    /// DE/rand/1: `v = x_r₁ + F (x_r₂ − x_r₃)`. Needs a population of 4.
    #[default]
    Rand1,
    /// DE/best/1: `v = x_best + F (x_r₁ − x_r₂)`. Needs a population of 4.
    Best1,
    /// DE/rand/2: `v = x_r₁ + F (x_r₂ − x_r₃) + F (x_r₄ − x_r₅)`. Needs a
    /// population of 6.
    Rand2,
    /// DE/current-to-best/1: `v = xᵢ + F (x_best − xᵢ) + F (x_r₁ − x_r₂)`.
    /// Needs a population of 4.
    CurrentToBest1,
}

impl DeStrategy {
    const fn minimum_population(self) -> usize {
        match self {
            Self::Rand2 => 6,
            Self::Rand1 | Self::Best1 | Self::CurrentToBest1 => 4,
        }
    }
}

/// Selects how [`DifferentialEvolution`] moves a trial coordinate that falls
/// outside the box back inside.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DifferentialEvolutionBoundary {
    /// Sets the coordinate to the nearest bound.
    #[default]
    Clip,
    /// Reflects the coordinate at the bounds, as often as needed.
    Reflect,
}

/// Selects when [`DifferentialEvolution`] replaces population members.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UpdatingMode {
    /// Replaces a member as soon as its trial wins, so later trials in the
    /// same generation see it.
    #[default]
    Immediate,
    /// Replaces members at the end of the generation, so all trials see the
    /// population of the generation start. Needed for parallel evaluation.
    Deferred,
}

/// Configures [`DifferentialEvolution`].
///
/// The setters do not check their arguments; [`DifferentialEvolution::new`]
/// checks the whole configuration. See [Termination](crate#termination) for
/// how the search ends.
#[derive(Debug, Clone)]
pub struct DifferentialEvolutionOptions<S: RealScalar> {
    boundary: DifferentialEvolutionBoundary,
    updating: UpdatingMode,
    fitness_tolerance: ResidualTolerance<S>,
    max_generations: Budget,
    population_size: Option<Budget>,
    mutation_factor: S,
    crossover_rate: S,
    strategy: DeStrategy,
    seed: u64,
    time_limit: Option<Duration>,
    execution: ExecutionPolicy,
}

impl<S: RealScalar> Default for DifferentialEvolutionOptions<S> {
    /// Returns DE/rand/1 with `F = 0.8`, `CR = 0.9`, clipping, immediate
    /// updates, 1000 generations, a population of `max(15 n, 4)` for `n`
    /// variables, seed 42 and sequential execution.
    fn default() -> Self {
        let one = S::ONE;
        let two = S::TWO;
        let five = two + two + one;
        let ten = five + five;
        Self {
            boundary: DifferentialEvolutionBoundary::Clip,
            updating: UpdatingMode::Immediate,
            fitness_tolerance: ResidualTolerance::default(),
            max_generations: Budget::ONE.saturating_add(999),
            population_size: None,
            mutation_factor: (ten - two) / ten,
            crossover_rate: (ten - one) / ten,
            strategy: DeStrategy::default(),
            seed: 42,
            time_limit: None,
            execution: ExecutionPolicy::Sequential,
        }
    }
}

impl<S: RealScalar> DifferentialEvolutionOptions<S> {
    /// Sets how trial coordinates outside the box are moved back inside.
    #[must_use]
    pub const fn with_boundary(mut self, boundary: DifferentialEvolutionBoundary) -> Self {
        self.boundary = boundary;
        self
    }

    /// Sets when population members are replaced.
    #[must_use]
    pub const fn with_updating(mut self, updating: UpdatingMode) -> Self {
        self.updating = updating;
        self
    }

    /// Sets the tolerance on the spread `f_max − f_min` of the population's
    /// objective values, relative to `max(|f_min|, |f_max|)`. A smaller spread
    /// stops the search with
    /// [`OptimizeTermination::PopulationStagnation`].
    #[must_use]
    pub const fn with_fitness_tolerance(mut self, fitness_tolerance: ResidualTolerance<S>) -> Self {
        self.fitness_tolerance = fitness_tolerance;
        self
    }

    /// Sets the maximum number of generations.
    #[must_use]
    pub const fn with_max_generations(mut self, budget: Budget) -> Self {
        self.max_generations = budget;
        self
    }

    /// Sets the population size. It must be at least the size the strategy
    /// needs (see [`DeStrategy`]).
    #[must_use]
    pub const fn with_population_size(mut self, size: Budget) -> Self {
        self.population_size = Some(size);
        self
    }

    /// Sets the mutation factor `F`, in `(0, 2]`.
    #[must_use]
    pub const fn with_mutation_factor(mut self, factor: S) -> Self {
        self.mutation_factor = factor;
        self
    }

    /// Sets the binomial crossover rate `CR`, in `[0, 1]`.
    #[must_use]
    pub const fn with_crossover_rate(mut self, rate: S) -> Self {
        self.crossover_rate = rate;
        self
    }

    /// Sets the mutation strategy.
    #[must_use]
    pub const fn with_strategy(mut self, strategy: DeStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the seed of the random stream.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets a time limit. It needs a clock in the
    /// [execution context](scioxide_core#execution-contexts).
    #[must_use]
    pub const fn with_time_limit(mut self, time_limit: Duration) -> Self {
        self.time_limit = Some(time_limit);
        self
    }

    /// Sets whether the trials of a generation are evaluated in parallel.
    ///
    /// A policy other than `Sequential` needs [`UpdatingMode::Deferred`] and a
    /// problem built with [`BoundedObjective::new_sync`]. Trials are generated
    /// and selected in member order, so the result does not depend on the
    /// number of workers.
    #[must_use]
    pub const fn with_execution(mut self, execution: ExecutionPolicy) -> Self {
        self.execution = execution;
        self
    }

    /// Returns how trial coordinates outside the box are moved back inside.
    pub const fn boundary(&self) -> DifferentialEvolutionBoundary {
        self.boundary
    }

    /// Returns when population members are replaced.
    pub const fn updating(&self) -> UpdatingMode {
        self.updating
    }

    /// Returns the tolerance on the spread of objective values.
    pub const fn fitness_tolerance(&self) -> ResidualTolerance<S> {
        self.fitness_tolerance
    }

    /// Returns the maximum number of generations.
    pub const fn max_generations(&self) -> Budget {
        self.max_generations
    }

    /// Returns the population size, or `None` for the default `max(15 n, 4)`.
    pub const fn population_size(&self) -> Option<Budget> {
        self.population_size
    }

    /// Returns the mutation factor `F`.
    pub const fn mutation_factor(&self) -> S {
        self.mutation_factor
    }

    /// Returns the crossover rate `CR`.
    pub const fn crossover_rate(&self) -> S {
        self.crossover_rate
    }

    /// Returns the mutation strategy.
    pub const fn strategy(&self) -> DeStrategy {
        self.strategy
    }

    /// Returns the seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the time limit, if set.
    pub const fn time_limit(&self) -> Option<Duration> {
        self.time_limit
    }

    /// Returns the execution policy.
    pub const fn execution(&self) -> ExecutionPolicy {
        self.execution
    }
}

/// Minimizes an objective over a finite box by differential evolution.
///
/// The population starts with `x0` and uniform random points in the box. In
/// each generation, every member `xᵢ` gets a mutant `v` from its
/// [`DeStrategy`], and binomial crossover forms a trial that takes each
/// coordinate from `v` with probability `CR` (and at least one coordinate
/// always). Coordinates outside the box are moved back by the
/// [`DifferentialEvolutionBoundary`] rule. The trial replaces `xᵢ` if its
/// objective value is not larger. Each generation costs one objective
/// evaluation per member, and storage is O(population × n) for `n`
/// variables.
///
/// The search stops with [`OptimizeTermination::PopulationStagnation`] when
/// the spread of objective values falls below the fitness tolerance, and with
/// [`OptimizeTermination::SearchBudgetCompleted`] after the last generation.
/// Neither is a success, and the search does not prove that it found the
/// global minimum. The result holds the best point found. One random stream
/// seeded by [`DifferentialEvolutionOptions::seed`] supplies every draw; see
/// [Randomness and parallelism](crate#randomness-and-parallelism).
///
/// # Examples
///
/// ```
/// use scioxide_core::{Bounds, ExecutionContext, SciOxideError};
/// use scioxide_optimize::{
///     BoundedObjective, DifferentialEvolution, DifferentialEvolutionOptions, ObjectiveFn,
///     OptimizeResult,
/// };
/// # fn main() -> Result<(), SciOxideError> {
/// let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]));
/// let bounds = Bounds::new(vec![(Some(-5.0), Some(5.0)); 2])?;
/// let problem = BoundedObjective::new(&objective, &bounds)?;
/// let solver = DifferentialEvolution::new(DifferentialEvolutionOptions::default())?;
/// let mut workspace = solver.workspace(2)?;
/// let mut output = OptimizeResult::empty();
/// let cx = ExecutionContext::sequential();
/// solver.minimize_into(&problem, &[1.0, 1.0], &mut output, &mut workspace, cx)?;
/// // The minimum of x² + y² is 0 at the origin.
/// assert!(output.fun() < 1e-6);
/// # Ok(())
/// # }
/// ```
///
/// # References
///
/// R. Storn and K. Price, "Differential evolution – a simple and efficient
/// heuristic for global optimization over continuous spaces", *J. Global
/// Optim.* 11(4), 341–359, 1997.
#[derive(Debug, Clone)]
pub struct DifferentialEvolution<S: RealScalar> {
    options: DifferentialEvolutionOptions<S>,
}

impl<S: RealScalar> Default for DifferentialEvolution<S> {
    fn default() -> Self {
        Self {
            options: DifferentialEvolutionOptions::default(),
        }
    }
}

impl<S: RealScalar> DifferentialEvolution<S> {
    /// Creates a solver after checking `options`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if the mutation factor or crossover rate is NaN or
    ///   infinite.
    /// - `InvalidInput` if the mutation factor is outside `(0, 2]`, the
    ///   crossover rate is outside `[0, 1]`, the population is smaller than
    ///   the strategy needs, or the policy is not sequential and updating is
    ///   [`UpdatingMode::Immediate`].
    pub fn new(options: DifferentialEvolutionOptions<S>) -> Result<Self, SciOxideError> {
        if options
            .population_size
            .is_some_and(|size| size.get() < options.strategy.minimum_population())
        {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("options"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                alloc::format!(
                    "{:?} requires a population of at least {}",
                    options.strategy,
                    options.strategy.minimum_population()
                ),
            ));
        }
        if !options.mutation_factor.is_finite() || !options.crossover_rate.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "mutation factor and crossover rate must be finite",
            ));
        }
        if options.mutation_factor <= S::ZERO || options.mutation_factor > S::TWO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("options"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "mutation factor must lie in (0, 2]",
            ));
        }
        if options.crossover_rate < S::ZERO || options.crossover_rate > S::ONE {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("options"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "crossover rate must lie in [0, 1]",
            ));
        }
        if options.execution != ExecutionPolicy::Sequential
            && options.updating != UpdatingMode::Deferred
        {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("options"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "a non-sequential execution policy requires deferred updates",
            ));
        }
        Ok(Self { options })
    }

    /// Returns the options.
    pub const fn options(&self) -> &DifferentialEvolutionOptions<S> {
        &self.options
    }
}

/// Pick a random index in `[0, n)` that is not in `exclude`.
fn pick_index<R: BitGenerator>(
    rng: &mut R,
    n: usize,
    exclude: &[usize],
) -> Result<usize, SciOxideError> {
    let mut rank = scioxide_random::uniform_index(n - exclude.len(), rng)?;
    // At most five exclusions: constant-size rank mapping, no population scan.
    let mut sorted = [0; 5];
    sorted[..exclude.len()].copy_from_slice(exclude);
    sorted[..exclude.len()].sort_unstable();
    for &index in &sorted[..exclude.len()] {
        if index <= rank {
            rank += 1;
        }
    }
    Ok(rank)
}

/// Reports one completed generation of [`DifferentialEvolution`].
#[derive(Debug, Clone, Copy)]
pub struct DifferentialEvolutionGeneration<S: RealScalar> {
    generation: usize,
    best_fitness: S,
    worst_fitness: S,
    accepted: usize,
    rejected: usize,
    repaired_coordinates: usize,
    counts: scioxide_core::SolveCounts,
}

impl<S: RealScalar> DifferentialEvolutionGeneration<S> {
    /// Returns the generation number, starting at 1.
    pub const fn generation(&self) -> usize {
        self.generation
    }
    /// Returns the lowest objective value in the population.
    pub const fn best_fitness(&self) -> S {
        self.best_fitness
    }
    /// Returns the highest objective value in the population.
    pub const fn worst_fitness(&self) -> S {
        self.worst_fitness
    }
    /// Returns the number of trials that replaced their member.
    pub const fn accepted(&self) -> usize {
        self.accepted
    }
    /// Returns the number of trials that did not replace their member.
    pub const fn rejected(&self) -> usize {
        self.rejected
    }
    /// Returns the number of trial coordinates moved back into the box.
    pub const fn repaired_coordinates(&self) -> usize {
        self.repaired_coordinates
    }
    /// Returns the evaluations and iterations so far.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
}

/// Access population member `i` from the flat buffer.
fn pop_member<S: Copy>(pop: &[S], i: usize, dim: usize) -> &[S] {
    &pop[i * dim..(i + 1) * dim]
}

fn de_diagnostics<S: RealScalar + NumericScalar<Real = S>>() -> OptimalityDiagnostics<S> {
    OptimalityDiagnostics {
        bound_violation: Some(<S as scioxide_core::AlgebraicScalar>::ZERO),
        ..OptimalityDiagnostics::unavailable()
    }
}

/// Population, physical points, fitness, trial and report storage.
#[derive(Debug)]
pub(crate) struct DifferentialEvolutionStorage<S: RealScalar> {
    dimension: usize,
    population_size: usize,
    bounds: FiniteBounds<S>,
    staged_population: Vec<S>,
    staged_points: Vec<S>,
    staged_fitness: Vec<S>,
    reports: Vec<DifferentialEvolutionGeneration<S>>,
    population: Vec<S>,
    points: Vec<S>,
    fitness: Vec<S>,
    trial: Vec<S>,
    physical_trial: Vec<S>,
    best: Vec<S>,
    best_value: Option<S>,
    generations: usize,
}
impl<S: RealScalar> Default for DifferentialEvolutionStorage<S> {
    fn default() -> Self {
        Self {
            dimension: 0,
            population_size: 0,
            bounds: FiniteBounds::unassigned(),
            staged_population: Vec::new(),
            staged_points: Vec::new(),
            staged_fitness: Vec::new(),
            reports: Vec::new(),
            population: Vec::new(),
            points: Vec::new(),
            fitness: Vec::new(),
            trial: Vec::new(),
            physical_trial: Vec::new(),
            best: Vec::new(),
            best_value: None,
            generations: 0,
        }
    }
}
impl<S: RealScalar> Accepted<S> for DifferentialEvolutionStorage<S> {
    fn accepted(&self) -> Option<AcceptedState<'_, S>> {
        Some(AcceptedState {
            point: &self.best,
            value: self.best_value?,
            gradient: None,
            iterations: self.generations,
        })
    }
}

/// Workspace size of a [`DifferentialEvolution`] solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DifferentialEvolutionWorkspaceRequirements {
    dimension: usize,
    population: usize,
    generations: usize,
}
impl DifferentialEvolutionWorkspaceRequirements {
    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }
    /// Returns the population size.
    pub const fn population(self) -> usize {
        self.population
    }
    /// Returns the number of generation reports kept.
    pub const fn generations(self) -> usize {
        self.generations
    }
}
impl<S: RealScalar> Shape<DifferentialEvolutionStorage<S>>
    for DifferentialEvolutionWorkspaceRequirements
{
    fn fits(&self, storage: &DifferentialEvolutionStorage<S>) -> bool {
        storage.dimension == self.dimension
            && storage.population_size == self.population
            && storage.reports.capacity() >= self.generations
            && storage.bounds.fits(self.dimension)
    }
    fn prepare(&self, storage: &mut DifferentialEvolutionStorage<S>) -> Result<(), SciOxideError> {
        let length = checked_mul(
            self.dimension,
            self.population,
            scioxide_core::quantity!("differential_evolution_population_storage"),
        )?;
        let mut next = DifferentialEvolutionStorage::default();
        for (buffer, size) in [
            (&mut next.population, length),
            (&mut next.staged_population, length),
            (&mut next.staged_points, length),
            (&mut next.staged_fitness, self.population),
            (&mut next.points, length),
            (&mut next.fitness, self.population),
            (&mut next.trial, self.dimension),
            (&mut next.physical_trial, self.dimension),
            (&mut next.best, self.dimension),
        ] {
            *buffer = scioxide_core::backend::try_filled(
                size,
                S::ZERO,
                scioxide_core::quantity!("differential_evolution_storage"),
            )?;
        }
        scioxide_core::backend::try_reserve(
            &mut next.reports,
            self.generations,
            scioxide_core::quantity!("de_generation_reports"),
        )?;
        next.bounds.try_reserve(self.dimension)?;
        next.dimension = self.dimension;
        next.population_size = self.population;
        *storage = next;
        Ok(())
    }
}

crate::attempt::local_workspace!(
    /// Workspace of [`DifferentialEvolution`]: the population, the generation
    /// reports and the record of the last call. Its size is
    /// O(population × n) for `n` variables.
    DifferentialEvolutionWorkspace,
    DifferentialEvolutionStorage<S>,
    DifferentialEvolutionWorkspaceRequirements
);

impl<S: RealScalar + NumericScalar<Real = S>> DifferentialEvolutionWorkspace<S> {
    /// Returns the reports of the generations completed by the most recent
    /// call.
    pub fn generations(&self) -> &[DifferentialEvolutionGeneration<S>] {
        &self.storage.reports
    }
}

fn observe_fitness<S, F>(
    f: &F,
    point: &[S],
    best: &mut [S],
    best_value: &mut Option<S>,
) -> Result<S, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    let value = f.eval(point)?;
    crate::utils::validate_finite(&[value])?;
    if best_value.is_none_or(|best| value < best) {
        best.copy_from_slice(point);
        *best_value = Some(value);
    }
    Ok(value)
}

type Evaluate<'a, S> =
    dyn Fn(&[S], &mut [S], &SolveSession<'_, '_, '_>) -> Result<(), SciOxideError> + 'a;

impl<S: RealScalar + NumericScalar<Real = S>> DifferentialEvolution<S> {
    fn population_size(&self, dimension: usize) -> Result<usize, SciOxideError> {
        self.options.population_size().map_or_else(
            || {
                checked_mul(
                    DEFAULT_POP_PER_DIM,
                    dimension,
                    scioxide_core::quantity!("differential_evolution_default_population"),
                )
                .map(|size| size.max(MIN_POPULATION))
            },
            |size| Ok(size.get()),
        )
    }

    /// Returns the workspace size for `dimension` variables.
    ///
    /// # Errors
    ///
    /// `SizeOverflow` if the size overflows.
    pub fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<DifferentialEvolutionWorkspaceRequirements, SciOxideError> {
        let population = self.population_size(dimension)?;
        checked_mul(
            dimension,
            population,
            scioxide_core::quantity!("differential_evolution_population_storage"),
        )?;
        Ok(DifferentialEvolutionWorkspaceRequirements {
            dimension,
            population,
            generations: self.options.max_generations().get(),
        })
    }

    /// Allocates a workspace for `dimension` variables.
    ///
    /// # Errors
    ///
    /// `SizeOverflow` or `AllocationFailure`.
    pub fn workspace(
        &self,
        dimension: usize,
    ) -> Result<DifferentialEvolutionWorkspace<S>, SciOxideError> {
        let mut workspace = DifferentialEvolutionWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }

    /// Minimizes `problem` from `x0` and writes the best point to `output`.
    ///
    /// When a limit, cancellation or a deadline stops the search after the
    /// first evaluation, `output` holds the best point found so far with an
    /// unsuccessful termination.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x0` does not match the problem dimension.
    /// - `NonFinite` if `x0` has a NaN or infinite entry, the objective returns
    ///   NaN or infinity, or a trial coordinate is not finite.
    /// - `InvalidInput` if a variable lacks a lower or upper bound or has
    ///   `lo ≥ hi`, `x0` lies outside the box, or the policy is not sequential
    ///   and the problem was not built with [`BoundedObjective::new_sync`].
    /// - `CapabilityUnavailable` if a time limit is set and the context has no
    ///   clock.
    /// - `ConvergenceFailure` or `ExecutionStopped` if the search stops before
    ///   the first evaluation, and errors of the objective and the executor.
    pub fn minimize_into<'scope, 'capability>(
        &self,
        problem: &BoundedObjective<'_, S>,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut DifferentialEvolutionWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        crate::utils::validate_x0(problem, x0)?;
        workspace.try_reserve(&self.workspace_requirements(x0.len())?)?;
        workspace.storage.bounds.assign(problem.bounds)?;
        if !workspace.storage.bounds.contains(x0) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("x0"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "initial point lies outside the problem bounds",
            ));
        }
        let shared = shared_objective(problem, self.options.execution)?;
        crate::attempt::preflight(&scope, self.options.time_limit)?;
        let entry = Entry::Call {
            identity: crate::context::local_identity(
                scioxide_core::algorithm_id!("optimize", "DifferentialEvolution", 1),
                x0.len(),
                &self.options,
            ),
            budgets: SolveBudgets::unbounded()
                .with_limit(BudgetKind::Iteration, self.options.max_generations),
        };
        crate::attempt::minimize(
            &mut workspace.shell,
            &mut workspace.storage,
            &mut workspace.staged,
            output,
            scope,
            entry,
            |context, storage, staged| {
                if let Some(limit) = self.options.time_limit {
                    context.limit_time(limit)?;
                }
                let session = SolveSession::new(context, scioxide_core::argument!("scope"));
                match shared {
                    None => self.run(
                        problem,
                        x0,
                        staged,
                        storage,
                        &session,
                        &|points, values, session| {
                            for (point, value) in points.chunks_exact(x0.len()).zip(values) {
                                *value = session
                                    .invoke(scioxide_core::EvaluationRole::Objective, || {
                                        problem.eval(point)
                                    })?;
                                crate::utils::validate_finite(&[*value])?;
                            }
                            Ok(())
                        },
                    ),
                    Some(objective) => self.run(
                        problem,
                        x0,
                        staged,
                        storage,
                        &session,
                        &|points, values, session| {
                            evaluate_partitioned(
                                objective,
                                points,
                                values,
                                x0.len(),
                                self.options.execution,
                                session,
                            )
                        },
                    ),
                }
            },
        )
    }

    fn run(
        &self,
        problem: &BoundedObjective<'_, S>,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        storage: &mut DifferentialEvolutionStorage<S>,
        session: &SolveSession<'_, '_, '_>,
        evaluate: &Evaluate<'_, S>,
    ) -> Result<(), SciOxideError> {
        storage.best_value = None;
        storage.generations = 0;
        storage.reports.clear();
        let mut generator = Xoshiro256StarStar::from_seed(self.options.seed);
        let outcome = self.evolve(
            &session.observe(problem),
            x0,
            storage,
            session,
            &mut generator,
            evaluate,
        );
        let termination = match outcome {
            Ok(termination) => termination,
            Err(error) => match (session.stopped(), storage.best_value) {
                (Some(termination), Some(_)) => termination,
                _ => return Err(error),
            },
        };
        let value = storage.best_value.ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("workspace"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "no validated population member",
            )
        })?;
        crate::utils::commit_optimize_result(
            output,
            &storage.best,
            value,
            None,
            storage.generations,
            termination,
            "differential evolution search completed",
        )?;
        output.diagnostics = de_diagnostics();
        crate::utils::synchronize_metadata(
            output,
            session,
            scioxide_core::PartialResultTrust::Admissible,
        )
    }

    fn evolve<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x0: &[S],
        storage: &mut DifferentialEvolutionStorage<S>,
        session: &SolveSession<'_, '_, '_>,
        generator: &mut dyn BitGenerator,
        evaluate: &Evaluate<'_, S>,
    ) -> Result<OptimizeTermination, SciOxideError> {
        let dim = x0.len();
        let pop_size = storage.population_size;
        let mut admission = session;
        let mut rng = SearchRandom::new(generator, &mut admission);
        let bounds = &storage.bounds;
        for i in 0..pop_size {
            session.proposal()?;
            if i == 0 {
                storage.points[..dim].copy_from_slice(x0);
                for (coordinate, (&value, &(lower, upper))) in storage.population[..dim]
                    .iter_mut()
                    .zip(x0.iter().zip(bounds.as_slice()))
                {
                    *coordinate = normalized_coordinate(value, lower, upper)?;
                }
            } else {
                for (j, &(lower, upper)) in bounds.as_slice().iter().enumerate() {
                    let u: S = scioxide_random::decode_u64(rng.next_uniform_u64()?);
                    storage.population[i * dim + j] = u;
                    storage.points[i * dim + j] = checked_affine(lower, upper, u)?;
                }
            }
            let value = observe_fitness(
                f,
                pop_member(&storage.points, i, dim),
                &mut storage.best,
                &mut storage.best_value,
            )?;
            session.accept(scioxide_core::PartialResultTrust::Admissible)?;
            storage.fitness[i] = value;
        }
        let mut best_idx = find_best_index(&storage.fitness)?;
        for generation in 0..self.options.max_generations().get() {
            let mut accepted = 0;
            let mut repaired_coordinates = 0;
            if let Some(stop) = session.iteration()? {
                return Ok(stop);
            }
            let (minimum, maximum) = fitness_spread(&storage.fitness)?;
            let spread = checked_difference(maximum, minimum)?;
            let scale = NumericScalar::abs(minimum).max(NumericScalar::abs(maximum));
            if self
                .options
                .fitness_tolerance()
                .is_satisfied(spread, scale)?
            {
                return Ok(OptimizeTermination::PopulationStagnation);
            }
            for i in 0..pop_size {
                session.proposal()?;
                mutate(
                    &mut storage.trial,
                    &storage.population,
                    i,
                    best_idx,
                    dim,
                    pop_size,
                    self.options.mutation_factor(),
                    self.options.strategy(),
                    &mut rng,
                )?;
                let j_rand = scioxide_random::uniform_index(dim, &mut rng)?;
                let current = pop_member(&storage.population, i, dim);
                for (j, (value, &current)) in storage.trial.iter_mut().zip(current).enumerate() {
                    let u: S = scioxide_random::decode_u64(rng.next_uniform_u64()?);
                    if u >= self.options.crossover_rate() && j != j_rand {
                        *value = current;
                    }
                    if !value.is_finite() {
                        return Err(SciOxideError::non_finite(
                            scioxide_core::NonFiniteOrigin::Intermediate,
                            "non-finite normalized trial",
                        ));
                    }
                    let repaired = repair(*value, self.options.boundary());
                    repaired_coordinates += usize::from(repaired != *value);
                    *value = repaired;
                    let (lower, upper) = bounds.as_slice()[j];
                    storage.physical_trial[j] = checked_affine(lower, upper, *value)?;
                }
                if self.options.updating() == UpdatingMode::Deferred {
                    storage.staged_population[i * dim..(i + 1) * dim]
                        .copy_from_slice(&storage.trial);
                    storage.staged_points[i * dim..(i + 1) * dim]
                        .copy_from_slice(&storage.physical_trial);
                    continue;
                }
                let value = observe_fitness(
                    f,
                    &storage.physical_trial,
                    &mut storage.best,
                    &mut storage.best_value,
                )?;
                if value <= storage.fitness[i] {
                    accepted += 1;
                    session.accept(scioxide_core::PartialResultTrust::Admissible)?;
                    storage.population[i * dim..(i + 1) * dim].copy_from_slice(&storage.trial);
                    storage.points[i * dim..(i + 1) * dim].copy_from_slice(&storage.physical_trial);
                    storage.fitness[i] = value;
                    if value < storage.fitness[best_idx] {
                        best_idx = i;
                    }
                } else {
                    session.reject()?;
                }
            }
            if self.options.updating() == UpdatingMode::Deferred {
                session.require_evaluations(pop_size)?;
                evaluate(&storage.staged_points, &mut storage.staged_fitness, session)?;
                for i in 0..pop_size {
                    let value = storage.staged_fitness[i];
                    if value <= storage.fitness[i] {
                        session.accept(scioxide_core::PartialResultTrust::Admissible)?;
                        accepted += 1;
                        storage.population[i * dim..(i + 1) * dim].copy_from_slice(pop_member(
                            &storage.staged_population,
                            i,
                            dim,
                        ));
                        storage.points[i * dim..(i + 1) * dim].copy_from_slice(pop_member(
                            &storage.staged_points,
                            i,
                            dim,
                        ));
                        storage.fitness[i] = value;
                    } else {
                        session.reject()?;
                    }
                }
                best_idx = find_best_index(&storage.fitness)?;
                storage
                    .best
                    .copy_from_slice(pop_member(&storage.points, best_idx, dim));
                storage.best_value = Some(storage.fitness[best_idx]);
            }
            storage.generations = generation + 1;
            let (best_fitness, worst_fitness) = fitness_spread(&storage.fitness)?;
            storage.reports.push(DifferentialEvolutionGeneration {
                generation: generation + 1,
                best_fitness,
                worst_fitness,
                accepted,
                rejected: pop_size - accepted,
                repaired_coordinates,
                counts: session.accounting()?,
            });
        }
        Ok(OptimizeTermination::SearchBudgetCompleted)
    }
}

/// Evaluate a deferred trial batch on executor partitions after counting every call.
fn evaluate_partitioned<S: RealScalar + NumericScalar<Real = S>>(
    objective: &(dyn ObjectiveFunction<S> + Sync),
    points: &[S],
    values: &mut [S],
    dimension: usize,
    execution: ExecutionPolicy,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    let context = session.execution()?;
    let limit = core::num::NonZeroUsize::MIN.saturating_add(63);
    let plan = context.plan(
        execution,
        values.len(),
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: core::num::NonZeroUsize::MIN,
            worker_limit: limit,
        },
    );
    for _ in 0..values.len() {
        session.admit(scioxide_core::EvaluationRole::Objective)?;
    }
    let mut slots: [_; 64] =
        core::array::from_fn(|_| scioxide_core::backend::PartitionSlot::empty());
    let mut remaining = values;
    for partition in plan.partitions() {
        let (out, tail) = remaining.split_at_mut(partition.range().len());
        remaining = tail;
        slots[partition.worker_index()] = scioxide_core::backend::PartitionSlot::new(out);
    }
    context.execute_partitioned(plan, &slots[..plan.worker_count()], &|partition, out, _| {
        for (index, value) in partition.range().zip(out) {
            *value = objective.eval(pop_member(points, index, dimension))?;
            crate::utils::validate_finite(&[*value])?;
        }
        Ok(())
    })?;
    session.check()
}

impl<S: RealScalar + NumericScalar<Real = S>> DifferentialEvolution<S> {
    /// Minimizes the objective with this method's settings and owned workspace.
    ///
    /// Returns the complete result, including unsuccessful termination. Call
    /// `require_success()` to require the stopping test to pass.
    ///
    /// # Errors
    ///
    /// Returns the input, callback, allocation and numerical errors of `minimize_into`.
    pub fn minimize(
        &self,
        problem: &BoundedObjective<'_, S>,
        x0: &[S],
    ) -> Result<OptimizeResult<S>, SciOxideError> {
        let mut workspace = self.workspace(x0.len())?;
        let mut output = OptimizeResult::empty();
        self.minimize_into(
            problem,
            x0,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}

impl<'a, S: RealScalar + NumericScalar<Real = S>>
    crate::traits::Optimizer<S, BoundedObjective<'a, S>> for DifferentialEvolution<S>
{
    fn minimize(
        &self,
        problem: &BoundedObjective<'a, S>,
        x0: &[S],
    ) -> Result<OptimizeResult<S>, SciOxideError> {
        self.minimize(problem, x0)
    }
}

/// Find the index of the member with the lowest fitness.
fn find_best_index<S: RealScalar>(fitness: &[S]) -> Result<usize, SciOxideError> {
    if fitness.is_empty() {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Empty,
            "fitness vector is empty",
        ));
    }
    if fitness.iter().any(|v| !RealScalar::is_finite(*v)) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Intermediate,
            "non-finite value encountered",
        ));
    }

    let mut best_idx = 0usize;
    let mut best_val = fitness[0];
    for (idx, &value) in fitness.iter().enumerate().skip(1) {
        if value < best_val {
            best_val = value;
            best_idx = idx;
        }
    }
    Ok(best_idx)
}

/// Compute the minimum and maximum fitness values.
fn fitness_spread<S: RealScalar>(fitness: &[S]) -> Result<(S, S), SciOxideError> {
    if fitness.is_empty() {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Empty,
            "fitness vector is empty",
        ));
    }
    if fitness.iter().any(|v| !RealScalar::is_finite(*v)) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Intermediate,
            "non-finite value encountered",
        ));
    }
    let mut f_min = fitness[0];
    let mut f_max = fitness[0];
    for &v in &fitness[1..] {
        if v < f_min {
            f_min = v;
        }
        if v > f_max {
            f_max = v;
        }
    }
    Ok((f_min, f_max))
}

/// Compute the mutant vector according to the chosen strategy.
#[allow(clippy::too_many_arguments)]
fn mutate<S: RealScalar + NumericScalar<Real = S>, R: BitGenerator>(
    trial: &mut [S],
    pop: &[S],
    i: usize,
    best_idx: usize,
    dim: usize,
    pop_size: usize,
    mutation_f: S,
    strategy: DeStrategy,
    rng: &mut R,
) -> Result<(), SciOxideError> {
    match strategy {
        DeStrategy::Rand1 => {
            let r1 = pick_index(rng, pop_size, &[i])?;
            let r2 = pick_index(rng, pop_size, &[i, r1])?;
            let r3 = pick_index(rng, pop_size, &[i, r1, r2])?;
            let xr1 = pop_member(pop, r1, dim);
            let xr2 = pop_member(pop, r2, dim);
            let xr3 = pop_member(pop, r3, dim);
            for j in 0..dim {
                trial[j] = xr1[j] + mutation_f * (xr2[j] - xr3[j]);
            }
        }
        DeStrategy::Best1 => {
            let r1 = pick_index(rng, pop_size, &[i])?;
            let r2 = pick_index(rng, pop_size, &[i, r1])?;
            let xb = pop_member(pop, best_idx, dim);
            let xr1 = pop_member(pop, r1, dim);
            let xr2 = pop_member(pop, r2, dim);
            for j in 0..dim {
                trial[j] = xb[j] + mutation_f * (xr1[j] - xr2[j]);
            }
        }
        DeStrategy::Rand2 => {
            let r1 = pick_index(rng, pop_size, &[i])?;
            let r2 = pick_index(rng, pop_size, &[i, r1])?;
            let r3 = pick_index(rng, pop_size, &[i, r1, r2])?;
            let r4 = pick_index(rng, pop_size, &[i, r1, r2, r3])?;
            let r5 = pick_index(rng, pop_size, &[i, r1, r2, r3, r4])?;
            let xr1 = pop_member(pop, r1, dim);
            let xr2 = pop_member(pop, r2, dim);
            let xr3 = pop_member(pop, r3, dim);
            let xr4 = pop_member(pop, r4, dim);
            let xr5 = pop_member(pop, r5, dim);
            for j in 0..dim {
                trial[j] = xr1[j] + mutation_f * (xr2[j] - xr3[j]) + mutation_f * (xr4[j] - xr5[j]);
            }
        }
        DeStrategy::CurrentToBest1 => {
            let r1 = pick_index(rng, pop_size, &[i])?;
            let r2 = pick_index(rng, pop_size, &[i, r1])?;
            let xi = pop_member(pop, i, dim);
            let xb = pop_member(pop, best_idx, dim);
            let xr1 = pop_member(pop, r1, dim);
            let xr2 = pop_member(pop, r2, dim);
            for j in 0..dim {
                trial[j] = xi[j] + mutation_f * (xb[j] - xi[j]) + mutation_f * (xr1[j] - xr2[j]);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::suboptimal_flops)]
mod tests {
    use super::*;
    use alloc::{format, vec};
    use scioxide_core::Bounds;

    fn boxed(bounds: f64, dimension: usize) -> Bounds<f64> {
        Bounds::new(vec![(Some(-bounds), Some(bounds)); dimension]).unwrap()
    }

    #[test]
    fn donor_ranks_are_uniform_and_distinct_at_minimum_populations() {
        for size in [4, 6, 7] {
            for target in 0..size {
                let mut rng = Xoshiro256StarStar::from_seed(610);
                let mut counts = vec![0_usize; size];
                for _ in 0..30_000 {
                    let first = pick_index(&mut rng, size, &[target]).unwrap();
                    counts[first] += 1;
                    let second = pick_index(&mut rng, size, &[target, first]).unwrap();
                    let third = pick_index(&mut rng, size, &[target, first, second]).unwrap();
                    assert!(![target, first].contains(&second));
                    assert!(![target, first, second].contains(&third));
                }
                assert_eq!(counts[target], 0);
                let expected = 30_000. / (size - 1) as f64;
                for (index, count) in counts.into_iter().enumerate() {
                    if index != target {
                        assert!((count as f64 - expected).abs() < 6. * expected.sqrt());
                    }
                }
            }
        }
    }

    // Sphere function: f(x) = sum(x_i^2)
    struct Sphere {
        dim: usize,
    }

    impl ObjectiveFunction<f64> for Sphere {
        fn dim(&self) -> usize {
            self.dim
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x.iter().map(|&xi| xi * xi).sum())
        }
    }

    struct Sphere32;

    impl ObjectiveFunction<f32> for Sphere32 {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f32]) -> Result<f32, SciOxideError> {
            Ok(x.iter().map(|&xi| xi * xi).sum())
        }
    }

    // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    struct Rosenbrock;

    impl ObjectiveFunction<f64> for Rosenbrock {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            Ok(a * a + 100.0 * b * b)
        }
    }

    struct NonFiniteObjective;

    impl ObjectiveFunction<f64> for NonFiniteObjective {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(f64::NAN)
        }
    }

    fn default_options() -> DifferentialEvolutionOptions<f64> {
        DifferentialEvolutionOptions::default()
            .with_fitness_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
                scioxide_core::Tolerance::new(1e-8, 1e-8).unwrap(),
            ))
            .with_max_generations(scioxide_core::Budget::try_from(1000).unwrap())
    }

    fn solve(
        options: DifferentialEvolutionOptions<f64>,
        objective: &dyn ObjectiveFunction<f64>,
        x0: &[f64],
    ) -> Result<OptimizeResult<f64>, SciOxideError> {
        let bounds = boxed(5.0, objective.dim());
        let problem = BoundedObjective::new(objective, &bounds)?;
        DifferentialEvolution::new(options)?.minimize(&problem, x0)
    }

    #[test]
    fn test_de_sphere_converges() {
        let result = solve(
            default_options().with_seed(123),
            &Sphere { dim: 2 },
            &[0.0, 0.0],
        )
        .unwrap();
        // Should find minimum near origin
        assert!(result.fun < 1e-4, "fun = {}", result.fun);
        assert_eq!(
            result.termination(),
            OptimizeTermination::PopulationStagnation
        );
        assert!(!result.is_success());
    }

    #[test]
    fn test_de_rosenbrock() {
        let options = DifferentialEvolutionOptions::default()
            .with_seed(42)
            .with_strategy(DeStrategy::Best1)
            .with_fitness_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
                scioxide_core::Tolerance::new(1e-10, 1e-10).unwrap(),
            ))
            .with_max_generations(scioxide_core::Budget::try_from(2000).unwrap());
        let result = solve(options, &Rosenbrock, &[0.0, 0.0]).unwrap();
        assert!(result.fun < 1.0, "fun = {}", result.fun);
    }

    #[test]
    fn every_strategy_reduces_the_sphere() {
        for (strategy, population) in [
            (DeStrategy::Rand1, None),
            (DeStrategy::Best1, None),
            (DeStrategy::Rand2, Some(30)),
            (DeStrategy::CurrentToBest1, None),
        ] {
            let mut options = default_options().with_strategy(strategy).with_seed(99);
            if let Some(size) = population {
                options =
                    options.with_population_size(scioxide_core::Budget::try_from(size).unwrap());
            }
            let result = solve(options, &Sphere { dim: 2 }, &[0.0, 0.0]).unwrap();
            assert!(result.fun < 1.0, "{strategy:?}");
        }
    }

    #[test]
    fn plan_construction_validates_cross_field_requirements() {
        let small = scioxide_core::Budget::try_from(4).unwrap();
        for options in [
            default_options().with_population_size(scioxide_core::Budget::try_from(3).unwrap()),
            default_options()
                .with_population_size(small)
                .with_strategy(DeStrategy::Rand2),
            default_options().with_mutation_factor(0.0),
            default_options().with_mutation_factor(2.5),
            default_options().with_crossover_rate(-0.1),
            default_options().with_crossover_rate(1.1),
            default_options().with_execution(ExecutionPolicy::Auto),
        ] {
            assert!(matches!(
                DifferentialEvolution::new(options).map_err(|error| error.kind()),
                Err(scioxide_core::SciOxideErrorKind::InvalidInput)
            ));
        }
        for options in [
            default_options().with_mutation_factor(f64::NAN),
            default_options().with_crossover_rate(f64::INFINITY),
        ] {
            assert!(matches!(
                DifferentialEvolution::new(options).map_err(|error| error.kind()),
                Err(scioxide_core::SciOxideErrorKind::NonFinite)
            ));
        }
        for options in [
            default_options().with_mutation_factor(2.0),
            default_options().with_crossover_rate(0.0),
            default_options().with_crossover_rate(1.0),
            default_options()
                .with_updating(UpdatingMode::Deferred)
                .with_execution(ExecutionPolicy::Auto),
        ] {
            assert!(DifferentialEvolution::new(options).is_ok());
        }
    }

    #[test]
    fn test_de_population_storage_overflow_is_structural() {
        let options = default_options()
            .with_population_size(scioxide_core::Budget::try_from(usize::MAX).unwrap());
        let result = solve(options, &Sphere { dim: 2 }, &[0.0, 0.0]);
        assert!(
            matches!(result, Err(error) if match error.view() { scioxide_core::SciOxideErrorView::SizeOverflow {  context: _view_context } => _view_context == scioxide_core::quantity!("differential_evolution_population_storage"), _ => false })
        );
    }

    #[test]
    fn test_de_algorithm_identity() {
        let result = solve(default_options(), &Sphere { dim: 2 }, &[1.0, 1.0]).unwrap();
        assert_eq!(
            result.algorithm().unwrap().method(),
            "DifferentialEvolution"
        );
    }

    #[test]
    fn test_de_f32() {
        let options = DifferentialEvolutionOptions::default()
            .with_seed(42)
            .with_fitness_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
                scioxide_core::Tolerance::new(1e-4, 1e-4).unwrap(),
            ))
            .with_max_generations(scioxide_core::Budget::try_from(500).unwrap());
        let bounds = Bounds::new(vec![(Some(-5.0_f32), Some(5.0_f32)); 2]).unwrap();
        let problem = BoundedObjective::new(&Sphere32, &bounds).unwrap();
        let result = DifferentialEvolution::new(options)
            .unwrap()
            .minimize(&problem, &[0.0_f32, 0.0])
            .unwrap();
        assert!(result.fun < 1.0);
    }

    #[test]
    fn test_de_dimension_mismatch() {
        let bounds = boxed(5.0, 3);
        assert!(matches!(
            BoundedObjective::new(&Sphere { dim: 2 }, &bounds).map(|_| ()),
            Err(error) if error.kind() == scioxide_core::SciOxideErrorKind::DimensionMismatch
        ));
        let bounds = boxed(5.0, 2);
        let problem = BoundedObjective::new(&Sphere { dim: 2 }, &bounds).unwrap();
        let result = DifferentialEvolution::new(default_options())
            .unwrap()
            .minimize(&problem, &[0.0, 0.0, 0.0]);
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_de_rejects_unbounded_and_outside_starts() {
        let bounds = Bounds::new(vec![(None, Some(1.0)), (Some(-1.0), Some(1.0))]).unwrap();
        let problem = BoundedObjective::new(&Sphere { dim: 2 }, &bounds).unwrap();
        let solver = DifferentialEvolution::new(default_options()).unwrap();
        assert!(solver.minimize(&problem, &[0.0, 0.0]).is_err());
        let result = solve(default_options(), &Sphere { dim: 2 }, &[6.0, 0.0]);
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
        ));
    }

    #[test]
    fn test_de_rejects_non_finite_fitness() {
        let result = solve(default_options(), &NonFiniteObjective, &[0.0, 0.0]);
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
    }

    #[test]
    fn test_de_strategy_enum_default() {
        assert_eq!(DeStrategy::default(), DeStrategy::Rand1);
        let s = DeStrategy::Best1;
        assert!(format!("{s:?}").contains("Best1"));
    }

    struct ExtremeBoundObjective;

    impl ObjectiveFunction<f64> for ExtremeBoundObjective {
        fn dim(&self) -> usize {
            1
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            assert!(x[0].is_finite());
            assert!((-f64::MAX..=f64::MAX).contains(&x[0]));
            Ok(x[0] / f64::MAX)
        }
    }

    #[test]
    fn extreme_finite_bounds_use_safe_affine_candidates() {
        let bounds = Bounds::new(vec![(Some(-f64::MAX), Some(f64::MAX))]).unwrap();
        let options = DifferentialEvolutionOptions::default()
            .with_population_size(scioxide_core::Budget::try_from(4).unwrap())
            .with_max_generations(scioxide_core::Budget::ONE)
            .with_fitness_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
                scioxide_core::Tolerance::new(0.0, 0.0).unwrap(),
            ))
            .with_seed(17);
        let problem = BoundedObjective::new(&ExtremeBoundObjective, &bounds).unwrap();
        let result = DifferentialEvolution::new(options)
            .unwrap()
            .minimize(&problem, &[0.0])
            .unwrap();

        assert!(result.x()[0].is_finite());
        assert!((-f64::MAX..=f64::MAX).contains(&result.x()[0]));
        assert_eq!(
            result.termination(),
            OptimizeTermination::SearchBudgetCompleted
        );
        assert_eq!(result.bound_violation(), Some(0.0));
    }
}
