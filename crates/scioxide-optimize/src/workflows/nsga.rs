//! NSGA-II for constrained multiobjective optimization.
use super::*;
use super::{
    pareto::{Population, constrained_dominates, crowding, violation},
    sampling::{Batch, SampleStream, VariationAdmission},
};
use crate::{
    Constraints, ConstraintsWorkspace, FiniteBounds, SearchRandom, StartDesign, UniformStartDesign,
};
use alloc::vec::Vec;

use core::time::Duration;
use scioxide_core::{
    AttemptKind, AttemptRecord, Budget, EvaluationRole, ExecutionPolicy, FeasibilityTolerance,
    PartialResultTrust, SolveBudgets, SolveCheckpointMetadata, SolveScope,
    backend::{checked_add, checked_mul, try_filled},
};

use crate::attempt::{Entry, Shell, Staged, preflight};

/// Borrows a multiobjective problem for [`Nsga2`]: a sampled vector objective
/// on a finite box, optional constraints, and the sampler of its sample tokens.
pub struct Nsga2Problem<'a, R: RealScalar, O: ?Sized, Sm: ?Sized = PhiloxSampler> {
    objective: &'a O,
    bounds: &'a FiniteBounds<R>,
    constraints: Option<&'a Constraints<R>>,
    sampler: &'a Sm,
}
impl<R: RealScalar, O: ?Sized, Sm: ?Sized> Clone for Nsga2Problem<'_, R, O, Sm> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R: RealScalar, O: ?Sized, Sm: ?Sized> Copy for Nsga2Problem<'_, R, O, Sm> {}
impl<'a, R: RealScalar, O: SampledObjective<R> + ?Sized> Nsga2Problem<'a, R, O> {
    /// Creates a problem from an objective and its box, drawing sample tokens
    /// with [`PhiloxSampler`].
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if the objective has no components or its
    /// dimension differs from the box.
    pub fn new(objective: &'a O, bounds: &'a FiniteBounds<R>) -> Result<Self, SciOxideError> {
        if objective.objectives() == 0 || objective.dimension() != bounds.len() {
            return Err(invalid(
                scioxide_core::argument!("objective"),
                "NSGA-II objective count or dimension",
            ));
        }
        Ok(Self {
            objective,
            bounds,
            constraints: None,
            sampler: &PhiloxSampler,
        })
    }
}
impl<'a, R: RealScalar, O: ?Sized, Sm: ?Sized> Nsga2Problem<'a, R, O, Sm> {
    /// Adds constraints, evaluated once per candidate point.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if the constraint dimension differs from the box.
    pub fn with_constraints(self, constraints: &'a Constraints<R>) -> Result<Self, SciOxideError> {
        if constraints.dimension() != self.bounds.len() {
            return Err(invalid(
                scioxide_core::argument!("constraints"),
                "NSGA-II constraint dimension",
            ));
        }
        Ok(Self {
            constraints: Some(constraints),
            ..self
        })
    }
    /// Replaces the sampler. A checkpoint can be resumed only with a sampler of
    /// the same configuration.
    pub const fn with_sampler<T: ?Sized>(self, sampler: &'a T) -> Nsga2Problem<'a, R, O, T> {
        Nsga2Problem {
            objective: self.objective,
            bounds: self.bounds,
            constraints: self.constraints,
            sampler,
        }
    }
    /// Returns the objective.
    pub const fn objective(&self) -> &'a O {
        self.objective
    }
    /// Returns the search box.
    pub const fn bounds(&self) -> &'a FiniteBounds<R> {
        self.bounds
    }
    /// Returns the constraints, if any.
    pub const fn constraints(&self) -> Option<&'a Constraints<R>> {
        self.constraints
    }
    /// Returns the sampler.
    pub const fn sampler(&self) -> &'a Sm {
        self.sampler
    }
}

/// Configures [`Nsga2`]: population size, archive capacity, generation limit
/// and the variation operators.
///
/// See [Termination](crate#termination) for the statuses.
#[derive(Debug, Clone)]
pub struct Nsga2Options<R: RealScalar> {
    population: usize,
    archive: usize,
    generations: Budget,
    replicates: Budget,
    feasibility: FeasibilityTolerance<R>,
    seed: u64,
    crossover: R,
    crossover_index: R,
    mutation: Option<R>,
    mutation_index: R,
    execution: ExecutionPolicy,
    time_limit: Option<Duration>,
}
impl<R: RealScalar> Nsga2Options<R> {
    /// Creates options with the given population size, archive capacity and
    /// generation limit.
    ///
    /// The defaults are: crossover probability 0.9 with distribution index 15,
    /// mutation probability `1/n` (for `n` variables) with distribution index
    /// 20, one sample per objective evaluation, seed 42, the default
    /// feasibility tolerance and sequential execution. The solver constructor
    /// requires a population of at least four and a positive archive capacity.
    pub fn new(population: usize, archive: usize, generations: Budget) -> Self {
        let ten = R::TWO * R::TWO * R::TWO + R::TWO;
        Self {
            population,
            archive,
            generations,
            replicates: Budget::ONE,
            feasibility: FeasibilityTolerance::default(),
            seed: 42,
            crossover: R::ONE - R::ONE / ten,
            crossover_index: ten + ten * R::HALF,
            mutation: None,
            mutation_index: ten + ten,
            execution: ExecutionPolicy::Sequential,
            time_limit: None,
        }
    }
    /// Sets the number of samples averaged per objective evaluation.
    /// Constraints are evaluated once per candidate.
    #[must_use]
    pub const fn with_replicates(mut self, count: Budget) -> Self {
        self.replicates = count;
        self
    }
    /// Sets the root seed of the variation operators and the objective
    /// samples.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    /// Sets the feasibility tolerance. It is applied with reference scale 1,
    /// so scale the constraint functions yourself if their magnitudes differ.
    #[must_use]
    pub const fn with_feasibility(mut self, tolerance: FeasibilityTolerance<R>) -> Self {
        self.feasibility = tolerance;
        self
    }
    /// Sets the crossover probability and distribution index of simulated
    /// binary crossover, and the per-variable mutation probability and
    /// distribution index of polynomial mutation. Larger indices give children
    /// closer to their parents. The solver constructor requires probabilities
    /// in `[0, 1]` and positive, finite indices.
    #[must_use]
    pub const fn with_variation(
        mut self,
        crossover: R,
        crossover_index: R,
        mutation: R,
        mutation_index: R,
    ) -> Self {
        self.crossover = crossover;
        self.crossover_index = crossover_index;
        self.mutation = Some(mutation);
        self.mutation_index = mutation_index;
        self
    }
    /// Sets the execution policy for evaluating replicated samples. The
    /// default is sequential. Results do not depend on the number of workers.
    #[must_use]
    pub const fn with_execution(mut self, execution: ExecutionPolicy) -> Self {
        self.execution = execution;
        self
    }
    /// Sets a time limit. A solve then needs an execution context with a clock.
    #[must_use]
    pub const fn with_time_limit(mut self, limit: Duration) -> Self {
        self.time_limit = Some(limit);
        self
    }
    /// Returns the population size.
    pub const fn population(&self) -> usize {
        self.population
    }
    /// Returns the archive capacity.
    pub const fn archive(&self) -> usize {
        self.archive
    }
    /// Returns the generation limit, which counts the generations of a resumed
    /// run together with those before the checkpoint.
    pub const fn generations(&self) -> Budget {
        self.generations
    }
    /// Returns the number of samples per objective evaluation.
    pub const fn replicates(&self) -> Budget {
        self.replicates
    }
    /// Returns the root seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }
    /// Returns the execution policy.
    pub const fn execution(&self) -> ExecutionPolicy {
        self.execution
    }
    /// Returns the time limit, if set.
    pub const fn time_limit(&self) -> Option<Duration> {
        self.time_limit
    }
    fn validate(&self) -> Result<(), SciOxideError> {
        if self.population < 4 || self.archive == 0 {
            return Err(invalid(
                scioxide_core::argument!("population"),
                "NSGA-II needs population >= 4 and a positive archive capacity",
            ));
        }
        for v in [Some(self.crossover), self.mutation].into_iter().flatten() {
            if !v.is_finite() || v < R::ZERO || v > R::ONE {
                return Err(invalid(
                    scioxide_core::argument!("crossover"),
                    "variation probability outside [0,1]",
                ));
            }
        }
        for (argument, v) in [
            (
                scioxide_core::argument!("crossover_index"),
                self.crossover_index,
            ),
            (
                scioxide_core::argument!("mutation_index"),
                self.mutation_index,
            ),
        ] {
            crate::proximal::positive(argument, v)?;
        }
        Ok(())
    }
}
/// Holds the result of an [`Nsga2`] solve: the final population and the
/// archive of nondominated feasible points.
///
/// The archive is a finite approximation of the Pareto front; it is not
/// guaranteed to contain Pareto-optimal points.
#[derive(Debug, Clone)]
pub struct Nsga2Result<R: RealScalar> {
    population: Population<R>,
    archive: ParetoArchive<R>,
    least: Population<R>,
    generations: usize,
    cursor: u64,
    next_id: u64,
    counts: scioxide_core::SolveCounts,
    termination: OptimizeTermination,
}
impl<R: RealScalar> Nsga2Result<R> {
    /// Returns the archive of feasible, mutually nondominated points found.
    pub fn archive(&self) -> &ParetoArchive<R> {
        &self.archive
    }
    /// Returns the current population, which may include infeasible points.
    pub fn population(&self) -> impl Iterator<Item = ParetoEntry<'_, R>> {
        (0..self.population.len).map(|i| self.population.entry(i))
    }
    /// Returns the evaluated point with the smallest total constraint
    /// violation; ties go to the earliest point. Useful when no feasible point
    /// was found.
    pub fn least_violating(&self) -> Option<ParetoEntry<'_, R>> {
        (self.least.len > 0).then(|| self.least.entry(0))
    }
    /// Returns the number of completed generations, not counting the initial
    /// population.
    pub const fn generations(&self) -> usize {
        self.generations
    }
    /// Returns the sequence number of the next sample.
    pub const fn sample_cursor(&self) -> u64 {
        self.cursor
    }
    /// Returns the iteration, sample and evaluation counts of the latest call.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns why the solver stopped. A run that completes its generations
    /// stops with
    /// [`SearchBudgetCompleted`](crate::OptimizeTermination::SearchBudgetCompleted);
    /// NSGA-II never reports success.
    pub const fn termination(&self) -> OptimizeTermination {
        self.termination
    }
    fn copy_from(&mut self, other: &Self) {
        self.population.copy_from(&other.population);
        self.archive.copy_from(&other.archive);
        self.least.copy_from(&other.least);
        self.generations = other.generations;
        self.cursor = other.cursor;
        self.next_id = other.next_id;
        self.counts = other.counts;
        self.termination = other.termination;
    }
    fn try_clone(&self) -> Result<Self, SciOxideError> {
        Ok(Self {
            population: self.population.try_clone()?,
            archive: self.archive.try_clone()?,
            least: self.least.try_clone()?,
            ..*self
        })
    }
    fn fits(&self, requirements: &Nsga2WorkspaceRequirements<R>) -> bool {
        let (n, m, c) = (
            requirements.dimension,
            requirements.objectives,
            requirements.constraint_rows,
        );
        let shaped = |p: &Population<R>| (p.n, p.m, p.c) == (n, m, c);
        shaped(&self.population)
            && shaped(&self.archive.data)
            && shaped(&self.least)
            && self.population.capacity() == requirements.population
            && self.archive.capacity() == requirements.archive
            && self.archive.equalities == requirements.equalities
            && self.archive.tolerance == requirements.feasibility
    }
    fn state_token(&self) -> u128 {
        crate::context::options_token(&(
            self.population.state_token(),
            self.archive.state_token(),
            self.least.state_token(),
            (self.generations, self.cursor, self.next_id),
        ))
    }
}
impl<R: RealScalar> Staged for Nsga2Result<R> {
    fn reset(&mut self) {
        self.termination = OptimizeTermination::NotStarted;
    }
    fn publish(&mut self, source: &Self) -> Result<(), SciOxideError> {
        self.copy_from(source);
        Ok(())
    }
    fn succeeded(&self) -> bool {
        self.termination.is_success()
    }
}
/// Stores the accepted population and archive of an [`Nsga2`] solve, to
/// continue it later with [`Nsga2::resume_into`].
#[derive(Debug, Clone)]
pub struct Nsga2Checkpoint<R: RealScalar> {
    state: Nsga2Result<R>,
    metadata: SolveCheckpointMetadata,
}
impl<R: RealScalar> Nsga2Checkpoint<R> {
    /// Returns the archive of feasible, mutually nondominated points.
    pub fn archive(&self) -> &ParetoArchive<R> {
        &self.state.archive
    }
    /// Returns the number of completed generations.
    pub const fn generations(&self) -> usize {
        self.state.generations
    }
    /// Returns the sequence number of the next sample. A resumed run continues
    /// from it, so it never reuses a sample.
    pub const fn sample_cursor(&self) -> u64 {
        self.state.cursor
    }
    /// Returns the solve metadata stored with the checkpoint.
    pub const fn metadata(&self) -> SolveCheckpointMetadata {
        self.metadata
    }
}
/// Describes the workspace shape of an [`Nsga2`] solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nsga2WorkspaceRequirements<R: RealScalar> {
    dimension: usize,
    objectives: usize,
    equalities: usize,
    inequalities: usize,
    constraint_rows: usize,
    // Even empty constraints need a prepared constraint workspace.
    constrained: bool,
    population: usize,
    archive: usize,
    replicates: usize,
    feasibility: FeasibilityTolerance<R>,
}
impl<R: RealScalar> Nsga2WorkspaceRequirements<R> {
    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }
    /// Returns the number of objectives.
    pub const fn objectives(self) -> usize {
        self.objectives
    }
    /// Returns the numbers of equality and inequality constraints.
    pub const fn constraints(self) -> (usize, usize) {
        (self.equalities, self.inequalities)
    }
    /// Returns the population size. The workspace holds twice as many
    /// candidates, parents and children.
    pub const fn population(self) -> usize {
        self.population
    }
}
/// Population, sorting, constraint and replicate scratch of one problem shape.
/// Storage is bounded by population², archive size, dimensions and replicate count.
struct Storage<R: RealScalar + NumericScalar<Real = R>> {
    state: Nsga2Result<R>,
    combined: Population<R>,
    batch: Batch<R>,
    constraints: Option<ConstraintsWorkspace<R>>,
    ranks: Vec<usize>,
    distance: Vec<R>,
    dominated: Vec<usize>,
    edges: Vec<bool>,
    front: Vec<usize>,
    next: Vec<usize>,
    order: Vec<usize>,
    equalities: usize,
    requirements: Nsga2WorkspaceRequirements<R>,
}
/// Holds the storage of an [`Nsga2`] solve, a record of the last call, and the
/// state for [`Self::checkpoint`].
///
/// Its size grows with the square of the population size, because
/// nondominated sorting stores the dominance relation between candidates.
pub struct Nsga2Workspace<R: RealScalar + NumericScalar<Real = R>> {
    storage: Option<Storage<R>>,
    staged: Option<Nsga2Result<R>>,
    shell: Shell<Nsga2Progress>,
    checkpoint_metadata: Option<SolveCheckpointMetadata>,
}
impl<R: RealScalar + NumericScalar<Real = R>> Default for Nsga2Workspace<R> {
    fn default() -> Self {
        Self::new()
    }
}
impl<R: RealScalar + NumericScalar<Real = R>> Nsga2Workspace<R> {
    /// Creates an empty workspace. Use [`Nsga2::workspace`] to create one sized
    /// for a problem.
    pub const fn new() -> Self {
        Self {
            storage: None,
            staged: None,
            shell: Shell::new(),
            checkpoint_metadata: None,
        }
    }
    /// Returns whether the workspace is large enough for `requirements`.
    ///
    /// # Errors
    ///
    /// Never fails; this workspace has no byte limit.
    pub fn check(
        &self,
        requirements: &Nsga2WorkspaceRequirements<R>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        Ok(
            if self
                .storage
                .as_ref()
                .is_some_and(|w| w.requirements == *requirements)
            {
                scioxide_core::Capacity::Sufficient
            } else {
                scioxide_core::Capacity::NeedsGrowth
            },
        )
    }
    /// Returns the record of the most recent call or resume, or `None`.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<Nsga2Progress>> {
        self.shell.last_attempt()
    }
    /// Returns the population and archive accepted by the most recent call, or
    /// `None`.
    pub fn last_accepted(&self) -> Option<&Nsga2Result<R>> {
        self.shell.progress()?;
        Some(&self.storage.as_ref()?.state)
    }
    /// Copies the accepted state of the most recent call into a checkpoint.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the most recent call accepted no state.
    /// - `AllocationFailure` if the checkpoint cannot be allocated.
    pub fn checkpoint(&self) -> Result<Nsga2Checkpoint<R>, SciOxideError> {
        let missing = || {
            SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Malformed,
                "no accepted state in the current attempt",
            )
        };
        let metadata = self.checkpoint_metadata.ok_or_else(missing)?;
        let state = self.last_accepted().ok_or_else(missing)?.try_clone()?;
        Ok(Nsga2Checkpoint { state, metadata })
    }
}
/// Summarizes the last accepted generation of an [`Nsga2`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Nsga2Progress {
    generations: usize,
    archived: usize,
}
impl Nsga2Progress {
    /// Returns the number of completed generations.
    pub const fn generations(self) -> usize {
        self.generations
    }
    /// Returns the number of points in the archive.
    pub const fn archived(self) -> usize {
        self.archived
    }
}
/// Approximates the Pareto front of a constrained multiobjective problem with
/// the NSGA-II evolutionary algorithm.
///
/// The problem is to minimize `F(x) = (f₁(x), …, f_m(x))` over a finite box,
/// subject to optional constraints. Each generation selects parents by binary
/// tournament, creates children by simulated binary crossover and polynomial
/// mutation (in coordinates scaled to `[0, 1]` and kept in the box), and keeps
/// the best half of parents and children. Candidates are ranked by
/// constrained dominance (a feasible point beats an infeasible one, and of two
/// infeasible points the one with the smaller total violation wins), then by
/// nondominated front, then by crowding distance. Evaluated points are also
/// offered to a [`ParetoArchive`], which keeps the feasible nondominated ones.
/// Nondominated sorting costs
/// `O(m N²)` per generation for population size `N`. Samples come from the
/// problem's sampler, so a run is reproducible from its seed and does not
/// depend on the number of workers. The initial population comes from a
/// [`StartDesign`](crate::StartDesign), uniformly random by default.
///
/// A run that completes its generations stops with
/// [`SearchBudgetCompleted`](crate::OptimizeTermination::SearchBudgetCompleted);
/// NSGA-II is a heuristic and never reports success.
///
/// # Errors
///
/// - `InvalidInput` from [`Self::new`] for invalid options.
/// - `InvalidInput` if the start design cannot supply the population.
/// - `NonFinite` if an objective sample or constraint value is not finite.
/// - `CapabilityUnavailable` if a time limit is set and the context has no
///   clock.
/// - `ExecutionStopped`, or the error of the exhausted limit, if the solve
///   stops before the first point is evaluated.
/// - Any error returned by the objective, the constraints or the sampler.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, SciOxideError};
/// use scioxide_optimize::{
///     FiniteBounds, NoiseModel, Nsga2, Nsga2Options, Nsga2Problem, SampledObjective,
/// };
///
/// // Minimize x² and (x − 1)² on [0, 1]; every x in [0, 1] is Pareto optimal.
/// struct Pair;
/// impl SampledObjective<f64> for Pair {
///     fn dimension(&self) -> usize {
///         1
///     }
///     fn objectives(&self) -> usize {
///         2
///     }
///     fn noise_model(&self) -> NoiseModel {
///         NoiseModel::Deterministic
///     }
///     fn sample_into(&self, x: &[f64], _: u64, out: &mut [f64]) -> Result<(), SciOxideError> {
///         out[0] = x[0] * x[0];
///         out[1] = (x[0] - 1.0).powi(2);
///         Ok(())
///     }
/// }
///
/// let bounds = FiniteBounds::new(vec![(0.0, 1.0)])?;
/// let solver = Nsga2::new(Nsga2Options::new(8, 8, Budget::try_from(5)?))?;
/// let result = solver.solve(&Nsga2Problem::new(&Pair, &bounds)?)?;
/// assert!(!result.archive().is_empty());
/// # Ok::<(), SciOxideError>(())
/// ```
///
/// # References
///
/// - K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist
///   multiobjective genetic algorithm: NSGA-II", *IEEE Trans. Evol. Comput.*
///   6(2), 182–197, 2002. <https://doi.org/10.1109/4235.996017>
/// - K. Deb and R. B. Agrawal, "Simulated binary crossover for continuous
///   search space", *Complex Systems* 9(2), 115–148, 1995.
#[derive(Debug, Clone)]
pub struct Nsga2<R: RealScalar, D = UniformStartDesign> {
    options: Nsga2Options<R>,
    starts: D,
}
impl<R: RealScalar + NumericScalar<Real = R>> Nsga2<R> {
    /// Creates a solver with checked options and a uniformly random initial
    /// population.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if the population is smaller than four, the
    /// archive capacity is zero, a probability is outside `[0, 1]`, or a
    /// distribution index is not positive and finite.
    pub fn new(options: Nsga2Options<R>) -> Result<Self, SciOxideError> {
        options.validate()?;
        Ok(Self {
            options,
            starts: UniformStartDesign,
        })
    }
}
impl<R: RealScalar + NumericScalar<Real = R>, D: StartDesign<R>> Nsga2<R, D> {
    /// Replaces the design that generates the initial population.
    pub fn with_start_design<T: StartDesign<R>>(self, starts: T) -> Nsga2<R, T> {
        Nsga2 {
            options: self.options,
            starts,
        }
    }
    /// Returns the options.
    pub const fn options(&self) -> &Nsga2Options<R> {
        &self.options
    }
    /// Returns the workspace shape for `problem`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if the start design cannot supply the population,
    /// and `SizeOverflow` if a size does not fit in `usize`.
    pub fn workspace_requirements<O, Sm>(
        &self,
        problem: &Nsga2Problem<'_, R, O, Sm>,
    ) -> Result<Nsga2WorkspaceRequirements<R>, SciOxideError>
    where
        O: SampledObjective<R> + ?Sized,
        Sm: ?Sized,
    {
        let o = &self.options;
        let dimension = problem.bounds.len();
        self.starts.validate(dimension, o.population)?;
        let profile = problem.constraints.map(Constraints::profile);
        let equalities = profile.map_or(0, |p| p.equality_count());
        let inequalities = profile.map_or(0, |p| p.inequality_count());
        let constraint_rows = checked_add(
            equalities,
            inequalities,
            scioxide_core::quantity!("constraint_rows"),
        )?;
        let combined = checked_mul(
            2,
            o.population,
            scioxide_core::quantity!("combined_population"),
        )?;
        checked_mul(
            combined,
            combined,
            scioxide_core::quantity!("workspace.edges"),
        )?;
        Ok(Nsga2WorkspaceRequirements {
            dimension,
            objectives: problem.objective.objectives(),
            equalities,
            inequalities,
            constraint_rows,
            constrained: problem.constraints.is_some(),
            population: o.population,
            archive: o.archive,
            replicates: o.replicates.get(),
            feasibility: o.feasibility,
        })
    }
    /// Allocates a workspace for `problem`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::workspace_requirements`] and
    /// `AllocationFailure`.
    pub fn workspace<O, Sm>(
        &self,
        problem: &Nsga2Problem<'_, R, O, Sm>,
    ) -> Result<Nsga2Workspace<R>, SciOxideError>
    where
        O: SampledObjective<R> + ?Sized,
        Sm: ?Sized,
    {
        let mut workspace = Nsga2Workspace::new();
        self.reserve(problem, &mut workspace)?;
        Ok(workspace)
    }
    /// Allocates a result for `problem`, to pass to [`Self::solve_into`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::workspace_requirements`] and
    /// `AllocationFailure`.
    pub fn result<O, Sm>(
        &self,
        problem: &Nsga2Problem<'_, R, O, Sm>,
    ) -> Result<Nsga2Result<R>, SciOxideError>
    where
        O: SampledObjective<R> + ?Sized,
        Sm: ?Sized,
    {
        self.state(&self.workspace_requirements(problem)?)
    }
    fn state(
        &self,
        requirements: &Nsga2WorkspaceRequirements<R>,
    ) -> Result<Nsga2Result<R>, SciOxideError> {
        let o = &self.options;
        let (n, m, c) = (
            requirements.dimension,
            requirements.objectives,
            requirements.constraint_rows,
        );
        Ok(Nsga2Result {
            population: Population::new(n, m, c, o.population)?,
            archive: ParetoArchive::prepare(
                n,
                m,
                c,
                requirements.equalities,
                o.archive,
                o.feasibility,
            )?,
            least: Population::new(n, m, c, 1)?,
            generations: 0,
            cursor: 0,
            next_id: 0,
            counts: scioxide_core::SolveCounts::new(),
            termination: OptimizeTermination::NotStarted,
        })
    }
    fn reserve<O, Sm>(
        &self,
        problem: &Nsga2Problem<'_, R, O, Sm>,
        workspace: &mut Nsga2Workspace<R>,
    ) -> Result<Nsga2WorkspaceRequirements<R>, SciOxideError>
    where
        O: SampledObjective<R> + ?Sized,
        Sm: ?Sized,
    {
        let requirements = self.workspace_requirements(problem)?;
        if workspace.check(&requirements)? == scioxide_core::Capacity::Sufficient {
            return Ok(requirements);
        }
        let size = 2 * requirements.population;
        let (n, m, c) = (
            requirements.dimension,
            requirements.objectives,
            requirements.constraint_rows,
        );
        let staged = self.state(&requirements)?;
        let storage = Storage {
            state: self.state(&requirements)?,
            combined: Population::new(n, m, c, size)?,
            batch: Batch::new(m, requirements.replicates)?,
            constraints: problem
                .constraints
                .map(ConstraintsWorkspace::new)
                .transpose()?,
            ranks: try_filled(size, 0, scioxide_core::quantity!("workspace.ranks"))?,
            distance: try_filled(
                size,
                R::ZERO,
                scioxide_core::quantity!("workspace.distance"),
            )?,
            dominated: try_filled(size, 0, scioxide_core::quantity!("workspace.dominated"))?,
            edges: try_filled(
                checked_mul(size, size, scioxide_core::quantity!("workspace.edges"))?,
                false,
                scioxide_core::quantity!("workspace.edges"),
            )?,
            front: try_filled(size, 0, scioxide_core::quantity!("workspace.front"))?,
            next: try_filled(size, 0, scioxide_core::quantity!("workspace.next"))?,
            order: try_filled(size, 0, scioxide_core::quantity!("workspace.order"))?,
            equalities: requirements.equalities,
            requirements,
        };
        (workspace.storage, workspace.staged) = (Some(storage), Some(staged));
        Ok(requirements)
    }
    fn identity<O, Sm>(&self, problem: &Nsga2Problem<'_, R, O, Sm>) -> scioxide_core::SolveIdentity
    where
        O: SampledObjective<R> + ?Sized,
        Sm: OptimizationSampler + ?Sized,
    {
        // Results are independent of the worker count and the time limit, so a
        // checkpoint resumes under any execution policy or deadline.
        let options = Nsga2Options {
            execution: ExecutionPolicy::Sequential,
            time_limit: None,
            ..self.options.clone()
        };
        crate::context::local_identity(
            scioxide_core::algorithm_id!("optimize", "Nsga2", 1),
            problem.bounds.len(),
            &(
                &options,
                problem.bounds,
                problem.objective.objectives(),
                problem.constraints.map(Constraints::profile),
                problem.objective.noise_model(),
                problem.sampler.configuration_token(),
                core::any::type_name::<D>(),
            ),
        )
    }
    /// Runs the search on the calling thread and returns a new result.
    ///
    /// # Errors
    ///
    /// As [`Self::solve_into`].
    pub fn solve<O, Sm>(
        &self,
        problem: &Nsga2Problem<'_, R, O, Sm>,
    ) -> Result<Nsga2Result<R>, SciOxideError>
    where
        O: SampledObjective<R> + ?Sized,
        Sm: OptimizationSampler + ?Sized,
    {
        let mut workspace = self.workspace(problem)?;
        let mut output = self.result(problem)?;
        self.solve_into(
            problem,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Runs the search and writes the result to `output`.
    ///
    /// Each generation counts as one iteration. A limit, cancellation or
    /// deadline after the first point is evaluated ends the run with the last
    /// complete generation, and [`Nsga2Workspace::checkpoint`] can then save it
    /// for [`Self::resume_into`]. A stop before that is an error. On error,
    /// `output` is unchanged.
    ///
    /// # Errors
    ///
    /// See the type-level `# Errors` section.
    pub fn solve_into<'scope, 'capability, O, Sm>(
        &self,
        problem: &Nsga2Problem<'_, R, O, Sm>,
        output: &mut Nsga2Result<R>,
        workspace: &mut Nsga2Workspace<R>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError>
    where
        O: SampledObjective<R> + ?Sized,
        Sm: OptimizationSampler + ?Sized,
    {
        workspace.shell.reject(AttemptKind::Call);
        workspace.checkpoint_metadata = None;
        let scope = scope.into();
        let entry = Entry::Call {
            identity: self.identity(problem),
            budgets: self.budgets(),
        };
        self.attempt(problem, None, output, workspace, scope, entry)
    }
    /// Continues a run from `checkpoint` and writes the result to `output`.
    ///
    /// The generation limit counts the generations before the checkpoint. An
    /// interrupted generation is discarded, and sampling continues after the
    /// last sample used, so no sample is reused. The checkpoint is not changed.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the checkpoint comes from other options, another
    ///   problem shape, noise model, sampler or start design. The checkpoint,
    ///   `output` and the parent context are then unchanged.
    /// - The errors of [`Self::solve_into`].
    pub fn resume_into<'scope, 'capability, O, Sm>(
        &self,
        problem: &Nsga2Problem<'_, R, O, Sm>,
        checkpoint: &Nsga2Checkpoint<R>,
        output: &mut Nsga2Result<R>,
        workspace: &mut Nsga2Workspace<R>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError>
    where
        O: SampledObjective<R> + ?Sized,
        Sm: OptimizationSampler + ?Sized,
    {
        workspace.shell.reject(AttemptKind::Resume);
        workspace.checkpoint_metadata = None;
        let scope = scope.into();
        let entry = Entry::Resume {
            identity: self.identity(problem),
            state_token: checkpoint.state.state_token(),
            metadata: checkpoint.metadata,
            budgets: self.budgets(),
        };
        self.attempt(problem, Some(checkpoint), output, workspace, scope, entry)
    }
    fn budgets(&self) -> SolveBudgets {
        SolveBudgets::unbounded().with_limit(
            scioxide_core::BudgetKind::Iteration,
            self.options.generations,
        )
    }
    fn attempt<'scope, 'capability, O, Sm>(
        &self,
        problem: &Nsga2Problem<'_, R, O, Sm>,
        checkpoint: Option<&Nsga2Checkpoint<R>>,
        output: &mut Nsga2Result<R>,
        workspace: &mut Nsga2Workspace<R>,
        scope: SolveScope<'scope, 'capability>,
        entry: Entry,
    ) -> Result<(), SciOxideError>
    where
        O: SampledObjective<R> + ?Sized,
        Sm: OptimizationSampler + ?Sized,
    {
        let requirements = self.reserve(problem, workspace)?;
        if !output.fits(&requirements)
            || checkpoint
                .is_some_and(|c| !c.state.fits(&requirements) || c.state.population.len == 0)
        {
            return Err(invalid(
                scioxide_core::argument!("output"),
                "NSGA-II result or checkpoint shape differs from the problem",
            ));
        }
        preflight(&scope, self.options.time_limit)?;
        let Nsga2Workspace {
            storage: Some(storage),
            staged: Some(staged),
            shell,
            checkpoint_metadata,
        } = workspace
        else {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "unprepared workspace",
            ));
        };
        crate::attempt::run(
            shell,
            storage,
            staged,
            output,
            scope,
            entry,
            |context, w, staged| {
                let s = &mut w.state;
                if let Some(checkpoint) = checkpoint {
                    s.copy_from(&checkpoint.state);
                } else {
                    s.population.len = 0;
                    s.least.len = 0;
                    s.generations = 0;
                    s.cursor = 0;
                    s.next_id = 0;
                    // Reset bounded archive counters along with visible entries.
                    s.archive.reset();
                }
                if let Some(limit) = self.options.time_limit {
                    context.limit_time(limit)?;
                }
                let result = run(&self.options, problem, &self.starts, w, context);
                if w.state.population.len > 0 {
                    let snapshot = context.checkpoint(w.state.state_token());
                    *checkpoint_metadata = if checkpoint.is_some() {
                        snapshot.ok()
                    } else {
                        Some(snapshot?)
                    };
                }
                finish(result, staged, &mut w.state, context)
            },
            |w, _| {
                (w.state.population.len > 0).then(|| Nsga2Progress {
                    generations: w.state.generations,
                    archived: w.state.archive.len(),
                })
            },
        )
    }
}
fn finish<R: RealScalar>(
    result: Flow<OptimizeTermination>,
    out: &mut Nsga2Result<R>,
    state: &mut Nsga2Result<R>,
    context: &SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    let termination = match result {
        Ok(t) | Err(FlowError::Stop(t)) => t,
        Err(FlowError::Failure(e)) => return Err(e),
    };
    if state.population.len == 0 {
        return Err(initial_stop(termination, context));
    }
    state.termination = termination;
    state.counts = context.counts();
    out.copy_from(state);
    Ok(())
}
/// A refused variation draw stops the search with its reason; other failures propagate.
fn flow<U>(random: &SearchRandom<'_>, result: Result<U, SciOxideError>) -> Flow<U> {
    result.map_err(|e| {
        random
            .stopped()
            .map_or(FlowError::Failure(e), FlowError::Stop)
    })
}
fn draw_uniform<R: RealScalar>(random: &mut SearchRandom<'_>) -> Flow<R> {
    let result = random.uniform();
    flow(random, result)
}
fn draw_index(random: &mut SearchRandom<'_>, n: usize) -> Flow<usize> {
    let result = scioxide_random::uniform_index(n, random);
    flow(random, result)
}
#[allow(clippy::too_many_arguments)]
fn observe<R: RealScalar + NumericScalar<Real = R>>(
    objective: &(impl SampledObjective<R> + ?Sized),
    constraints: Option<&Constraints<R>>,
    constraint_work: &mut Option<ConstraintsWorkspace<R>>,
    pop: &mut Population<R>,
    i: usize,
    eq: usize,
    batch: &mut Batch<R>,
    (options, bounds): (&Nsga2Options<R>, &FiniteBounds<R>),
    sampler: &(impl OptimizationSampler + ?Sized),
    cursor: &mut u64,
    policy: ExecutionPolicy,
    context: &mut SolveContext<'_, '_>,
) -> Flow<()> {
    let x = &pop.points[i * pop.n..(i + 1) * pop.n];
    finite_vector(x)?;
    if x.iter()
        .zip(bounds.as_slice())
        .any(|(&x, &(lo, hi))| x < lo || x > hi)
    {
        return Err(invalid(
            scioxide_core::argument!("starts"),
            "start design returned an out-of-bounds coordinate",
        )
        .into());
    }
    batch.reset();
    admitted(batch.extend(
        options.replicates.get(),
        objective.noise_model(),
        options.seed,
        1,
        cursor,
        sampler,
        EvaluationRole::Objective,
        policy,
        context,
        &|token, out| objective.sample_into(x, token, out),
    )?)?;
    if let Some(constraints) = constraints {
        let session =
            crate::context::SolveSession::new(context, scioxide_core::argument!("context"));
        let (equalities, inequalities) =
            pop.constraints[i * pop.c..(i + 1) * pop.c].split_at_mut(eq);
        let result = constraints.evaluate_observed(
            x,
            equalities,
            inequalities,
            constraint_work.as_mut().ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("work"),
                    "missing constraint workspace",
                )
            })?,
            Some(&session),
        );
        if let Err(e) = result {
            return Err(session
                .stopped()
                .map_or(FlowError::Failure(e), FlowError::Stop));
        }
    }
    pop.values[i * pop.m..(i + 1) * pop.m].copy_from_slice(&batch.mean);
    pop.errors[i * pop.m..(i + 1) * pop.m].copy_from_slice(&batch.error);
    pop.variances[i * pop.m..(i + 1) * pop.m].copy_from_slice(&batch.variance);
    pop.samples[i] = batch.count;
    pop.uncertainty[i] = batch.error_available;
    pop.violations[i] = violation(
        &pop.constraints[i * pop.c..(i + 1) * pop.c],
        eq,
        options.feasibility,
    )?;
    Ok(())
}
fn retain<R: RealScalar>(
    state: &mut Nsga2Result<R>,
    entry: ParetoEntry<'_, R>,
) -> Result<(), SciOxideError> {
    state.archive.insert_estimate(entry)?;
    if state.least.len == 0 || entry.violation < state.least.violations[0] {
        state.least.put(0, entry);
        state.least.len = 1;
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn rank<R: RealScalar>(
    pop: &Population<R>,
    ranks: &mut [usize],
    distance: &mut [R],
    dominated: &mut [usize],
    edges: &mut [bool],
    front: &mut [usize],
    next: &mut [usize],
    order: &mut [usize],
) -> Result<(), SciOxideError> {
    let n = pop.len;
    dominated[..n].fill(0);
    edges[..n * n].fill(false);
    for a in 0..n {
        for b in a + 1..n {
            let (winner, loser) = if constrained_dominates(pop.entry(a), pop.entry(b)) {
                (a, b)
            } else if constrained_dominates(pop.entry(b), pop.entry(a)) {
                (b, a)
            } else {
                continue;
            };
            edges[winner * n + loser] = true;
            dominated[loser] += 1;
        }
    }
    let mut len = 0;
    for (i, &count) in dominated[..n].iter().enumerate() {
        if count == 0 {
            front[len] = i;
            len += 1;
        }
    }
    let mut rank = 0;
    let mut assigned = 0;
    while len > 0 {
        crowding(pop, &mut front[..len], distance)?;
        let mut next_len = 0;
        for &a in &front[..len] {
            ranks[a] = rank;
            assigned += 1;
            for b in 0..n {
                if edges[a * n + b] {
                    dominated[b] -= 1;
                    if dominated[b] == 0 {
                        next[next_len] = b;
                        next_len += 1;
                    }
                }
            }
        }
        front[..next_len].copy_from_slice(&next[..next_len]);
        len = next_len;
        rank += 1;
    }
    if assigned != n {
        return Err(invalid(
            scioxide_core::argument!("constraints"),
            "cyclic constrained dominance",
        ));
    }
    for (i, item) in order[..n].iter_mut().enumerate() {
        *item = i;
    }
    order[..n].sort_unstable_by(|&a, &b| {
        ranks[a]
            .cmp(&ranks[b])
            .then_with(|| {
                distance[b]
                    .partial_cmp(&distance[a])
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .then(pop.ids[a].cmp(&pop.ids[b]))
    });
    Ok(())
}
fn tournament<R: RealScalar>(
    rng: &mut SearchRandom<'_>,
    pop: &Population<R>,
    ranks: &[usize],
    distance: &[R],
) -> Flow<usize> {
    let a = draw_index(rng, pop.len)?;
    let b = draw_index(rng, pop.len)?;
    Ok(
        if ranks[a] < ranks[b]
            || (ranks[a] == ranks[b]
                && (distance[a] > distance[b]
                    || (distance[a] == distance[b] && pop.ids[a] < pop.ids[b])))
        {
            a
        } else {
            b
        },
    )
}
fn crossover<R: RealScalar>(a: R, b: R, u: R, choose: R, index: R) -> Result<R, SciOxideError> {
    if a == b {
        return Ok(a);
    }
    let lo = a.min(b);
    let hi = a.max(b);
    let delta = hi - lo;
    let two = R::ONE + R::ONE;
    let power = index + <_ as scioxide_core::AlgebraicScalar>::ONE;
    let inverse = R::ONE / power;
    let side = if choose < number(0.5)? {
        lo
    } else {
        R::ONE - hi
    };
    // Reciprocal beta avoids overflow for adjacent parents near a box endpoint.
    let alpha = two - (delta / (delta + two * side)).powf(power);
    let beta = if u <= R::ONE / alpha {
        (u * alpha).powf(inverse)
    } else {
        (R::ONE / (two - u * alpha)).powf(inverse)
    };
    let signed = if choose < number(0.5)? { -beta } else { beta };
    finite(
        ((lo + hi + signed * delta) / two)
            .max(<_ as scioxide_core::AlgebraicScalar>::ZERO)
            .min(<_ as scioxide_core::AlgebraicScalar>::ONE),
    )
}
fn mutate<R: RealScalar>(value: R, u: R, index: R) -> Result<R, SciOxideError> {
    let two = R::ONE + R::ONE;
    let power = index + <_ as scioxide_core::AlgebraicScalar>::ONE;
    let inverse = R::ONE / power;
    let delta = if u < number(0.5)? {
        (two * u + (R::ONE - two * u) * (R::ONE - value).powf(power)).powf(inverse)
            - <_ as scioxide_core::AlgebraicScalar>::ONE
    } else {
        R::ONE - (two * (R::ONE - u) + two * (u - number(0.5)?) * value.powf(power)).powf(inverse)
    };
    finite(
        (value + delta)
            .max(<_ as scioxide_core::AlgebraicScalar>::ZERO)
            .min(<_ as scioxide_core::AlgebraicScalar>::ONE),
    )
}
#[allow(clippy::too_many_arguments)]
fn run<R, O, Sm>(
    o: &Nsga2Options<R>,
    problem: &Nsga2Problem<'_, R, O, Sm>,
    starts: &(impl StartDesign<R> + ?Sized),
    w: &mut Storage<R>,
    context: &mut SolveContext<'_, '_>,
) -> Flow<OptimizeTermination>
where
    R: RealScalar + NumericScalar<Real = R>,
    O: SampledObjective<R> + ?Sized,
    Sm: OptimizationSampler + ?Sized,
{
    let (objective, constraints, sampler) =
        (problem.objective, problem.constraints, problem.sampler);
    let (bounds, policy) = (problem.bounds, o.execution);
    let p = o.population;
    let n = bounds.len();
    let mutation = match o.mutation {
        Some(mutation) => mutation,
        None => R::ONE / R::try_from_usize_exact(n)?,
    };
    while w.state.population.len < p {
        admitted(stop(context)?)?;
        let i = w.state.population.len;
        let mut stream = SampleStream {
            sampler,
            seed: o.seed,
            cursor: &mut w.state.cursor,
        };
        let mut admission = VariationAdmission { context };
        let mut rng = SearchRandom::new(&mut stream, &mut admission);
        w.combined.points[..n].fill(<R as RealScalar>::NAN);
        let result = starts.point(i, bounds, &mut rng, &mut w.combined.points[..n]);
        flow(&rng, result)?;
        let id = w.state.next_id;
        w.state.next_id = id.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("state.next_id"))
        })?;
        w.combined.ids[0] = id;
        observe(
            objective,
            constraints,
            &mut w.constraints,
            &mut w.combined,
            0,
            w.equalities,
            &mut w.batch,
            (o, bounds),
            sampler,
            &mut w.state.cursor,
            policy,
            context,
        )?;
        retain(&mut w.state, w.combined.entry(0))?;
        w.state.population.put(i, w.combined.entry(0));
        w.state.population.len += 1;
        context.commit_accepted(PartialResultTrust::Admissible)?;
    }
    while w.state.generations < o.generations.get() {
        admitted(stop(context)?)?;
        // A generation is charged when it commits, so a discarded partial
        // generation does not consume the total retained by checkpoints.
        if !context
            .can_consume(scioxide_core::BudgetKind::Iteration, 1)?
            .is_granted()
        {
            return Ok(OptimizeTermination::IterationLimit);
        }
        rank(
            &w.state.population,
            &mut w.ranks,
            &mut w.distance,
            &mut w.dominated,
            &mut w.edges,
            &mut w.front,
            &mut w.next,
            &mut w.order,
        )?;
        w.combined.copy_from(&w.state.population);
        for i in p..2 * p {
            if !context.record_proposal()?.is_granted() {
                return Ok(OptimizeTermination::ProposalLimit);
            }
            let mut stream = SampleStream {
                sampler,
                seed: o.seed,
                cursor: &mut w.state.cursor,
            };
            let mut admission = VariationAdmission { context };
            let mut rng = SearchRandom::new(&mut stream, &mut admission);
            let a = tournament(&mut rng, &w.state.population, &w.ranks, &w.distance)?;
            let b = tournament(&mut rng, &w.state.population, &w.ranks, &w.distance)?;
            let cross = draw_uniform::<R>(&mut rng)? < o.crossover;
            for j in 0..n {
                let (lo, hi) = bounds.as_slice()[j];
                let a = crate::geometry::normalized_coordinate(
                    w.state.population.points[a * n + j],
                    lo,
                    hi,
                )?;
                let b = crate::geometry::normalized_coordinate(
                    w.state.population.points[b * n + j],
                    lo,
                    hi,
                )?;
                let mut value = if cross && draw_uniform::<R>(&mut rng)? < number(0.5)? {
                    crossover(
                        a,
                        b,
                        draw_uniform(&mut rng)?,
                        draw_uniform(&mut rng)?,
                        o.crossover_index,
                    )?
                } else {
                    a
                };
                if draw_uniform::<R>(&mut rng)? < mutation {
                    value = mutate(value, draw_uniform(&mut rng)?, o.mutation_index)?;
                }
                w.combined.points[i * n + j] = scioxide_core::checked_affine(lo, hi, value)?;
            }
            let id = w.state.next_id;
            w.state.next_id = id.checked_add(1).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("state.next_id"))
            })?;
            w.combined.ids[i] = id;
            observe(
                objective,
                constraints,
                &mut w.constraints,
                &mut w.combined,
                i,
                w.equalities,
                &mut w.batch,
                (o, bounds),
                sampler,
                &mut w.state.cursor,
                policy,
                context,
            )?;
        }
        w.combined.len = 2 * p;
        rank(
            &w.combined,
            &mut w.ranks,
            &mut w.distance,
            &mut w.dominated,
            &mut w.edges,
            &mut w.front,
            &mut w.next,
            &mut w.order,
        )?;
        if !context.record_iteration()?.is_granted() {
            return Ok(OptimizeTermination::IterationLimit);
        }
        // All fallible observations finished. Archive and population form one generation commit.
        for i in p..2 * p {
            retain(&mut w.state, w.combined.entry(i))?;
        }
        for (i, &selected) in w.order[..p].iter().enumerate() {
            w.state.population.put(i, w.combined.entry(selected));
        }
        w.state.generations += 1;
        context.commit_accepted(PartialResultTrust::Admissible)?;
    }
    Ok(OptimizeTermination::SearchBudgetCompleted)
}
