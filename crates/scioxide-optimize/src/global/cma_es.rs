//! CMA-ES over a finite box.

use alloc::vec::Vec;
use core::cmp::Ordering;

use scioxide_core::backend::{checked_mul, try_filled, try_reserve};
use scioxide_core::{
    AttemptKind, AttemptRecord, Budget, Capacity, EvaluationRole, ExecutionPolicy,
    InvalidInputReason, NumericScalar, PartialResultTrust, RealScalar, ResidualTolerance,
    SciOxideError, SolveBudgets, SolveScope, SolveState, StepTolerance,
};
use scioxide_linalg::{
    HermitianEigenDecomposition, HermitianEigenOptions, HermitianEigenWorkspaceRequest,
    LinalgWorkspace, Matrix,
};
use scioxide_random::{BitGenerator, Distribution, Normal, SeedableRng, Xoshiro256StarStar};

use super::local::{LocalPool, SearchLocalSolver, local_identity, recoverable, validate_local};
use super::{StartDesign, UniformStartDesign, shared_objective};
use crate::attempt::{Accepted, AcceptedState, Entry, Shell};
use crate::context::{Observation, SolveSession};
use crate::geometry::{normalized_coordinate, repair};
use crate::{
    BoundedObjective, DifferentialEvolutionBoundary, FiniteBounds, LocalSolverFactory,
    MinimizerAccepted, MinimizerProgress, ObjectiveFunction, OptimalityDiagnostics, OptimizeResult,
    OptimizeTermination, PowellLocalFactory, PowellSearchLocal, SearchRandom,
};

fn invalid(argument: scioxide_core::Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(argument, InvalidInputReason::OutOfRange, detail)
}

fn finite<S: RealScalar>(value: S, detail: &'static str) -> Result<S, SciOxideError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            detail,
        ))
    }
}

/// Selects whether [`CmaEs`] restarts after its population stagnates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CmaEsRestartPolicy {
    /// Runs once, without restarts.
    #[default]
    None,
    /// Restarts with twice the population each time a run stagnates
    /// (IPOP-CMA-ES), up to `max_restarts` times.
    IncreasingPopulation {
        /// Maximum number of restarts after the first run.
        max_restarts: usize,
    },
}

impl CmaEsRestartPolicy {
    const fn max_restarts(self) -> usize {
        match self {
            Self::None => 0,
            Self::IncreasingPopulation { max_restarts } => max_restarts,
        }
    }
}

/// Configures [`CmaEs`].
///
/// Step sizes are in box units: the box is scaled to `[0, 1]` in every
/// coordinate. The setters do not check their arguments; [`CmaEs::new`] checks
/// the whole configuration. See [Termination](crate#termination) for how the
/// search ends.
#[derive(Debug, Clone)]
pub struct CmaEsOptions<S: RealScalar> {
    initial_step: S,
    fitness_tolerance: ResidualTolerance<S>,
    step_tolerance: StepTolerance<S>,
    max_generations: Budget,
    population_size: Option<Budget>,
    restart: CmaEsRestartPolicy,
    seed: u64,
    local_budget: SolveBudgets,
    execution: ExecutionPolicy,
}

impl<S: RealScalar> Default for CmaEsOptions<S> {
    /// Returns the standard weighted CMA-ES parameters with initial step 0.3,
    /// 1000 generations, no restarts, seed 42, no limit on the local
    /// refinement and sequential execution.
    fn default() -> Self {
        Self {
            initial_step: (S::TWO + S::ONE) / S::TEN,
            fitness_tolerance: ResidualTolerance::default(),
            step_tolerance: StepTolerance::default(),
            max_generations: Budget::ONE.saturating_add(999),
            population_size: None,
            restart: CmaEsRestartPolicy::None,
            seed: 42,
            local_budget: SolveBudgets::unbounded(),
            execution: ExecutionPolicy::Sequential,
        }
    }
}

impl<S: RealScalar> CmaEsOptions<S> {
    /// Sets the initial step size `σ₀` in box units; it must be finite and
    /// positive.
    pub const fn with_initial_step(mut self, value: S) -> Self {
        self.initial_step = value;
        self
    }

    /// Sets the number of offspring per generation, `λ ≥ 4`.
    pub const fn with_population_size(mut self, value: Budget) -> Self {
        self.population_size = Some(value);
        self
    }

    /// Sets the maximum number of generations, summed over all restarts.
    pub const fn with_max_generations(mut self, value: Budget) -> Self {
        self.max_generations = value;
        self
    }

    /// Sets the tolerance on the spread `f_max − f_min` of a generation's
    /// objective values, relative to `max(|f_min|, |f_max|)`. A smaller spread
    /// ends the run.
    pub const fn with_fitness_tolerance(mut self, value: ResidualTolerance<S>) -> Self {
        self.fitness_tolerance = value;
        self
    }

    /// Sets the tolerance on the step size `σ √λ_max(C)` in box units, where
    /// `λ_max(C)` is the largest eigenvalue of the covariance matrix. A smaller
    /// step ends the run.
    pub const fn with_step_tolerance(mut self, value: StepTolerance<S>) -> Self {
        self.step_tolerance = value;
        self
    }

    /// Sets the restart policy.
    pub const fn with_restart(mut self, value: CmaEsRestartPolicy) -> Self {
        self.restart = value;
        self
    }

    /// Sets the seed of the random stream that generates the offspring and the
    /// restart points.
    pub const fn with_seed(mut self, value: u64) -> Self {
        self.seed = value;
        self
    }

    /// Sets the limits of the final local refinement. The limits of the whole
    /// search still apply.
    pub const fn with_local_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.local_budget = budgets;
        self
    }

    /// Sets whether the offspring of a generation are evaluated in parallel.
    ///
    /// A policy other than `Sequential` needs a problem built with
    /// [`BoundedObjective::new_sync`]. Offspring are generated and ranked in
    /// index order, so for a deterministic objective the result does not
    /// depend on the number of workers.
    pub const fn with_execution(mut self, execution: ExecutionPolicy) -> Self {
        self.execution = execution;
        self
    }

    /// Returns the initial step size in box units.
    pub const fn initial_step(&self) -> S {
        self.initial_step
    }
    /// Returns the number of offspring, or `None` for the default
    /// `4 + ⌊3 ln n⌋` for `n` variables.
    pub const fn population_size(&self) -> Option<Budget> {
        self.population_size
    }
    /// Returns the maximum number of generations.
    pub const fn max_generations(&self) -> Budget {
        self.max_generations
    }
    /// Returns the tolerance on the spread of objective values.
    pub const fn fitness_tolerance(&self) -> ResidualTolerance<S> {
        self.fitness_tolerance
    }
    /// Returns the tolerance on the step size.
    pub const fn step_tolerance(&self) -> StepTolerance<S> {
        self.step_tolerance
    }
    /// Returns the restart policy.
    pub const fn restart(&self) -> CmaEsRestartPolicy {
        self.restart
    }
    /// Returns the seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }
    /// Returns the limits of the final local refinement.
    pub const fn local_budgets(&self) -> SolveBudgets {
        self.local_budget
    }
    /// Returns the execution policy.
    pub const fn execution(&self) -> ExecutionPolicy {
        self.execution
    }
}

/// Reports one completed generation of [`CmaEs`].
#[derive(Debug, Clone, Copy)]
pub struct CmaEsGeneration<S: RealScalar> {
    generation: usize,
    restart: usize,
    best_fitness: S,
    worst_fitness: S,
    covariance_rank: usize,
    covariance_log_condition: S,
    step_size: S,
    repaired_coordinates: usize,
    counts: scioxide_core::SolveCounts,
}

impl<S: RealScalar> CmaEsGeneration<S> {
    /// Returns the generation number, starting at 1 and counted across
    /// restarts.
    pub const fn generation(&self) -> usize {
        self.generation
    }
    /// Returns the run this generation belongs to: 0 for the first run, `k`
    /// for restart `k`.
    pub const fn restart(&self) -> usize {
        self.restart
    }
    /// Returns the lowest objective value of the generation.
    pub const fn best_fitness(&self) -> S {
        self.best_fitness
    }
    /// Returns the highest objective value of the generation.
    pub const fn worst_fitness(&self) -> S {
        self.worst_fitness
    }
    /// Returns the numerical rank of the covariance matrix at the start of the
    /// generation.
    pub const fn covariance_rank(&self) -> usize {
        self.covariance_rank
    }
    /// Returns `ln(λ_max/λ_min)` of the covariance matrix, over its positive
    /// eigenvalues.
    pub const fn covariance_log_condition(&self) -> S {
        self.covariance_log_condition
    }
    /// Returns the step size `σ` after the update.
    pub const fn step_size(&self) -> S {
        self.step_size
    }
    /// Returns the number of offspring coordinates reflected back into the box.
    pub const fn repaired_coordinates(&self) -> usize {
        self.repaired_coordinates
    }
    /// Returns the evaluations and iterations so far.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
}

/// Numerical search state: population, covariance factors, paths and reports.
#[derive(Debug)]
struct CmaEsState<S: RealScalar + NumericScalar<Real = S>> {
    covariance: Matrix<S>,
    next_covariance: Matrix<S>,
    eig: HermitianEigenDecomposition<S>,
    eig_work: LinalgWorkspace<'static, S>,
    mean: Vec<S>,
    mean_step: Vec<S>,
    path_sigma: Vec<S>,
    path_covariance: Vec<S>,
    normal: Vec<S>,
    physical: Vec<S>,
    best: Vec<S>,
    points: Vec<S>,
    steps: Vec<S>,
    fitness: Vec<S>,
    order: Vec<usize>,
    weights: Vec<S>,
    reports: Vec<CmaEsGeneration<S>>,
    best_value: Option<S>,
    restarts: usize,
    last_covariance_rank: usize,
    last_covariance_log_condition: S,
}

impl<S: RealScalar + NumericScalar<Real = S>> CmaEsState<S> {
    fn prepare(n: usize, population: usize, generations: usize) -> Result<Self, SciOxideError> {
        let cells = checked_mul(
            population,
            n,
            scioxide_core::quantity!("cma_es_population_storage"),
        )?;
        let mut order = Vec::new();
        try_reserve(
            &mut order,
            population,
            scioxide_core::quantity!("cma_es_order"),
        )?;
        order.resize(population, 0);
        let mut reports = Vec::new();
        try_reserve(
            &mut reports,
            generations,
            scioxide_core::quantity!("cma_es_reports"),
        )?;
        let mut eig = HermitianEigenDecomposition::empty();
        let mut eig_work = HermitianEigenDecomposition::<S>::workspace(
            &HermitianEigenWorkspaceRequest::new(n, &HermitianEigenOptions::default()),
        )?;
        // Warm the eigen output on the identity, which is already diagonal, so
        // the first covariance factorization allocates nothing. Every run
        // resets the covariance before use.
        let mut covariance = Matrix::zeros(n, n)?;
        covariance
            .as_mut_slice()
            .iter_mut()
            .step_by(n + 1)
            .for_each(|value| *value = S::ONE);
        HermitianEigenDecomposition::decompose_into(
            &covariance,
            scioxide_linalg::HermitianInput::Exact,
            &HermitianEigenOptions::default(),
            &mut eig,
            &mut eig_work,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        Ok(Self {
            covariance,
            next_covariance: Matrix::zeros(n, n)?,
            eig,
            eig_work,
            mean: try_filled(n, S::ZERO, scioxide_core::quantity!("cma_es_mean"))?,
            mean_step: try_filled(n, S::ZERO, scioxide_core::quantity!("cma_es_mean_step"))?,
            path_sigma: try_filled(n, S::ZERO, scioxide_core::quantity!("cma_es_sigma_path"))?,
            path_covariance: try_filled(
                n,
                S::ZERO,
                scioxide_core::quantity!("cma_es_covariance_path"),
            )?,
            normal: try_filled(n, S::ZERO, scioxide_core::quantity!("cma_es_normal"))?,
            physical: try_filled(
                n,
                S::ZERO,
                scioxide_core::quantity!("cma_es_physical_point"),
            )?,
            best: try_filled(n, S::ZERO, scioxide_core::quantity!("cma_es_best"))?,
            points: try_filled(cells, S::ZERO, scioxide_core::quantity!("cma_es_points"))?,
            steps: try_filled(cells, S::ZERO, scioxide_core::quantity!("cma_es_steps"))?,
            fitness: try_filled(
                population,
                S::ZERO,
                scioxide_core::quantity!("cma_es_fitness"),
            )?,
            order,
            weights: try_filled(
                population / 2,
                S::ZERO,
                scioxide_core::quantity!("cma_es_weights"),
            )?,
            reports,
            best_value: None,
            restarts: 0,
            last_covariance_rank: n,
            last_covariance_log_condition: S::ZERO,
        })
    }

    fn fits(&self, n: usize, population: usize, generations: usize) -> bool {
        self.mean.len() == n
            && self.order.len() == population
            && self.reports.capacity() >= generations
    }
}

/// Search state, box copy and the optional prepared local solver.
pub(crate) struct CmaEsStorage<S: RealScalar + NumericScalar<Real = S>, L> {
    state: Option<CmaEsState<S>>,
    bounds: FiniteBounds<S>,
    local: LocalPool<L>,
    local_result: OptimizeResult<S>,
    local_error: Option<SciOxideError>,
}

impl<S: RealScalar + NumericScalar<Real = S>, L> Accepted<S> for CmaEsStorage<S, L> {
    fn accepted(&self) -> Option<AcceptedState<'_, S>> {
        let state = self.state.as_ref()?;
        Some(AcceptedState {
            point: &state.best,
            value: state.best_value?,
            gradient: None,
            iterations: state.reports.len(),
        })
    }
}

/// Workspace size of a [`CmaEs`] solve, sized for the largest restart.
///
/// It borrows the solver's local factory, if any.
#[derive(Debug)]
pub struct CmaEsWorkspaceRequirements<'a, F> {
    factory: Option<&'a F>,
    token: u128,
    dimension: usize,
    population: usize,
    generations: usize,
}

impl<F> CmaEsWorkspaceRequirements<'_, F> {
    /// Returns the number of variables.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    /// Returns the number of offspring of the largest restart.
    pub const fn population(&self) -> usize {
        self.population
    }
    /// Returns the number of generation reports kept.
    pub const fn generations(&self) -> usize {
        self.generations
    }
}

/// Workspace of [`CmaEs`]: the population, covariance matrix and its
/// eigendecomposition, evolution paths, generation reports, the local solver
/// and the record of the last call.
pub struct CmaEsWorkspace<S: RealScalar + NumericScalar<Real = S>, L = PowellSearchLocal<S>> {
    storage: CmaEsStorage<S, L>,
    staged: OptimizeResult<S>,
    shell: Shell<MinimizerProgress<S>>,
}

impl<S: RealScalar + NumericScalar<Real = S>, L> core::fmt::Debug for CmaEsWorkspace<S, L> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CmaEsWorkspace")
            .field("state", &self.storage.state)
            .field("local_error", &self.storage.local_error)
            .finish_non_exhaustive()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, L> Default for CmaEsWorkspace<S, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, L> CmaEsWorkspace<S, L> {
    /// Creates an empty workspace without allocating.
    pub fn new() -> Self {
        Self {
            storage: CmaEsStorage {
                state: None,
                bounds: FiniteBounds::unassigned(),
                local: LocalPool::new(),
                local_result: OptimizeResult::empty(),
                local_error: None,
            },
            staged: OptimizeResult::empty(),
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
        requirements: &CmaEsWorkspaceRequirements<'_, F>,
    ) -> Result<Capacity, SciOxideError> {
        let storage = &self.storage;
        Ok(
            if storage.state.as_ref().is_some_and(|state| {
                state.fits(
                    requirements.dimension,
                    requirements.population,
                    requirements.generations,
                )
            }) && storage.local.fits(
                usize::from(requirements.factory.is_some()),
                requirements.dimension,
                requirements.token,
            ) && storage.bounds.fits(requirements.dimension)
                && storage.local_result.fits(requirements.dimension)
                && self.staged.fits(requirements.dimension)
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    /// Allocates storage for `requirements`, including the largest restart.
    ///
    /// # Errors
    ///
    /// Returns the local factory's errors, eigensolver setup errors,
    /// `SizeOverflow` or `AllocationFailure`.
    pub fn try_reserve<F>(
        &mut self,
        requirements: &CmaEsWorkspaceRequirements<'_, F>,
    ) -> Result<(), SciOxideError>
    where
        F: LocalSolverFactory<S, Solver = L>,
    {
        if self.check(requirements)? == Capacity::Sufficient {
            return Ok(());
        }
        let (n, population, generations) = (
            requirements.dimension,
            requirements.population,
            requirements.generations,
        );
        let storage = &mut self.storage;
        if !storage
            .state
            .as_ref()
            .is_some_and(|state| state.fits(n, population, generations))
        {
            storage.state = Some(CmaEsState::prepare(n, population, generations)?);
        }
        match requirements.factory {
            Some(factory) => storage.local.prepare(factory, 1, n, requirements.token)?,
            None => storage.local = LocalPool::new(),
        }
        storage.bounds.try_reserve(n)?;
        storage.local_result.try_reserve(n)?;
        self.staged.try_reserve(n)
    }

    /// Returns the record of the most recent call.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<MinimizerProgress<S>>> {
        self.shell.last_attempt()
    }

    /// Returns the best point of the most recent call, if it found one.
    pub fn last_accepted(&self) -> Option<MinimizerAccepted<'_, S>> {
        crate::attempt::last_accepted(&self.shell, &self.storage)
    }

    /// Returns the reports of the generations completed by the most recent
    /// call.
    pub fn generations(&self) -> &[CmaEsGeneration<S>] {
        self.storage
            .state
            .as_ref()
            .map_or(&[], |state| &state.reports)
    }

    /// Returns the number of restarts in the most recent call.
    pub fn restarts(&self) -> usize {
        self.storage
            .state
            .as_ref()
            .map_or(0, |state| state.restarts)
    }

    /// Returns the error of the final local refinement, if it failed. A failed
    /// refinement does not fail the search.
    pub const fn local_error(&self) -> Option<&SciOxideError> {
        self.storage.local_error.as_ref()
    }

    /// Returns the last measured numerical rank of the covariance matrix. A
    /// rank below the dimension ends the run.
    pub fn covariance_rank(&self) -> usize {
        self.storage
            .state
            .as_ref()
            .map_or(0, |state| state.last_covariance_rank)
    }

    /// Returns the last finite value of `ln(λ_max/λ_min)` of the covariance
    /// matrix.
    pub fn covariance_log_condition(&self) -> S {
        self.storage
            .state
            .as_ref()
            .map_or(S::ZERO, |state| state.last_covariance_log_condition)
    }
}

type Evaluate<'a, S> = dyn Fn(&mut CmaEsState<S>, usize, usize, &SolveSession<'_, '_, '_>) -> Result<(), SciOxideError>
    + 'a;

/// Minimizes an objective over a finite box with the covariance matrix
/// adaptation evolution strategy (CMA-ES).
///
/// Each generation samples `λ` offspring from a normal distribution
/// `N(m, σ² C)` in box units, reflects coordinates that leave the box, and
/// moves the mean `m` to the weighted mean of the best `μ = ⌊λ/2⌋` offspring.
/// It then
/// adapts the full covariance matrix `C` (rank-one and rank-μ updates) and the
/// step size `σ` (cumulative step-size adaptation). Each generation costs `λ`
/// objective evaluations, an eigendecomposition of `C` in O(n³) and O(λ n²)
/// for sampling, for `n` variables.
///
/// A run ends when the spread of objective values or the step size falls
/// below its tolerance, or the covariance matrix loses rank. With
/// [`CmaEsRestartPolicy::IncreasingPopulation`] the search then restarts with
/// twice the population. The first run starts at `x0`, restart `k` at point
/// `k − 1` of the start design (uniform by default; see [`Self::with_design`]).
/// The search ends with [`OptimizeTermination::PopulationStagnation`] after the
/// last run and with [`OptimizeTermination::SearchBudgetCompleted`] after the
/// last generation. Neither is a success, and the search does not prove that
/// it found the global minimum. A local solver set with [`Self::with_local`]
/// refines the best point at the end and replaces it only if it is lower.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Bounds, ExecutionContext, SciOxideError};
/// use scioxide_optimize::{BoundedObjective, CmaEs, CmaEsOptions, ObjectiveFn, OptimizeResult};
/// # fn main() -> Result<(), SciOxideError> {
/// let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]));
/// let bounds = Bounds::new(vec![(Some(-1.0), Some(1.0)); 2])?;
/// let problem = BoundedObjective::new(&objective, &bounds)?;
/// let solver = CmaEs::new(CmaEsOptions::default())?;
/// let mut workspace = solver.workspace(2)?;
/// let mut output = OptimizeResult::empty();
/// let cx = ExecutionContext::sequential();
/// solver.minimize_into(&problem, &[0.8, 0.8], &mut output, &mut workspace, cx)?;
/// // The minimum of x² + y² is 0 at the origin.
/// assert!(output.fun() < 1e-8);
/// # Ok(())
/// # }
/// ```
///
/// # References
///
/// - N. Hansen and A. Ostermeier, "Completely derandomized self-adaptation in
///   evolution strategies", *Evol. Comput.* 9(2), 159–195, 2001.
/// - A. Auger and N. Hansen, "A restart CMA evolution strategy with increasing
///   population size", *IEEE Congress on Evolutionary Computation*, 2005.
/// - N. Hansen, "The CMA evolution strategy: a tutorial", arXiv:1604.00772,
///   2016.
#[derive(Debug, Clone)]
pub struct CmaEs<S: RealScalar, Design = UniformStartDesign, Factory = PowellLocalFactory<S>> {
    options: CmaEsOptions<S>,
    design: Design,
    local: Option<Factory>,
}

impl<S: RealScalar> Default for CmaEs<S> {
    fn default() -> Self {
        Self {
            options: CmaEsOptions::default(),
            design: UniformStartDesign,
            local: None,
        }
    }
}

impl<S: RealScalar> CmaEs<S> {
    /// Creates a solver after checking `options`. Restarts use uniform random
    /// points, and no local refinement runs.
    ///
    /// # Errors
    ///
    /// `InvalidInput` if the initial step is NaN, infinite or not positive, or
    /// the population is smaller than 4.
    pub fn new(options: CmaEsOptions<S>) -> Result<Self, SciOxideError> {
        if !options.initial_step.is_finite() || options.initial_step <= S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "initial step must be finite and positive",
            ));
        }
        if options.population_size.is_some_and(|size| size.get() < 4) {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "population must contain at least four offspring",
            ));
        }
        Ok(Self {
            options,
            design: UniformStartDesign,
            local: None,
        })
    }
}

impl<S: RealScalar, D, F> CmaEs<S, D, F> {
    /// Uses `design` for the restart points; design point 0 starts the first
    /// restart.
    pub fn with_design<Design>(self, design: Design) -> CmaEs<S, Design, F> {
        CmaEs {
            options: self.options,
            design,
            local: self.local,
        }
    }

    /// Refines the final best point with a local solver from `factory`.
    pub fn with_local<Factory>(self, factory: Factory) -> CmaEs<S, D, Factory> {
        CmaEs {
            options: self.options,
            design: self.design,
            local: Some(factory),
        }
    }

    /// Returns the options.
    pub const fn options(&self) -> &CmaEsOptions<S> {
        &self.options
    }
}

impl<S, D, F> CmaEs<S, D, F>
where
    S: RealScalar + NumericScalar<Real = S>,
    D: StartDesign<S>,
    F: LocalSolverFactory<S>,
{
    fn initial_population(&self, dimension: usize) -> usize {
        self.options.population_size.map_or_else(
            || 4 + (3.0 * (dimension as f64).ln()).floor() as usize,
            Budget::get,
        )
    }

    fn maximum_population(&self, dimension: usize) -> Result<usize, SciOxideError> {
        let shifts = u32::try_from(self.options.restart.max_restarts()).map_err(|_| {
            SciOxideError::size_overflow(scioxide_core::quantity!("cma_es_restart_count"))
        })?;
        let multiplier = 1usize.checked_shl(shifts).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("cma_es_restart_count"))
        })?;
        self.initial_population(dimension)
            .checked_mul(multiplier)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!(
                    "cma_es_restarted_population"
                ))
            })
    }

    /// Returns the workspace size for `dimension` variables, including the
    /// largest restart.
    ///
    /// # Errors
    ///
    /// Returns the design's error if it cannot supply the restart points, and
    /// `SizeOverflow` if the size overflows.
    pub fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<CmaEsWorkspaceRequirements<'_, F>, SciOxideError> {
        self.design
            .validate(dimension, self.options.restart.max_restarts())?;
        let population = self.maximum_population(dimension)?;
        checked_mul(
            population,
            dimension,
            scioxide_core::quantity!("cma_es_population_storage"),
        )?;
        Ok(CmaEsWorkspaceRequirements {
            factory: self.local.as_ref(),
            token: self.local.as_ref().map_or(0, crate::context::options_token),
            dimension,
            population,
            generations: self.options.max_generations.get(),
        })
    }

    /// Allocates a workspace for `dimension` variables.
    ///
    /// # Errors
    ///
    /// As [`Self::workspace_requirements`] and [`CmaEsWorkspace::try_reserve`].
    pub fn workspace(
        &self,
        dimension: usize,
    ) -> Result<CmaEsWorkspace<S, F::Solver>, SciOxideError> {
        let mut workspace = CmaEsWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }

    /// Minimizes `problem` from `x0` and writes the best point to `output`.
    ///
    /// One random stream seeded by [`CmaEsOptions::seed`] generates the
    /// offspring and restart points; see
    /// [Randomness and parallelism](crate#randomness-and-parallelism). When a
    /// limit, cancellation or a deadline stops the search after the first
    /// evaluation, `output` holds the best point found so far with an
    /// unsuccessful termination.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x0` does not match the problem dimension.
    /// - `NonFinite` if `x0` has a NaN or infinite entry, or the objective
    ///   returns NaN or infinity.
    /// - `InvalidInput` if a variable lacks a lower or upper bound or has
    ///   `lo ≥ hi`, `x0` lies outside the box, the policy is not sequential and
    ///   the problem was not built with [`BoundedObjective::new_sync`], or the
    ///   local solver returns a point outside the box or a non-finite value.
    /// - `NumericalInstability` if a sample or the step size overflows.
    /// - `ConvergenceFailure` or `ExecutionStopped` if the search stops before
    ///   the first evaluation, and errors of the objective, the eigensolver and
    ///   the executor.
    pub fn minimize_into<'scope, 'capability>(
        &self,
        problem: &BoundedObjective<'_, S>,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut CmaEsWorkspace<S, F::Solver>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        crate::utils::validate_x0(problem, x0)?;
        let requirements = self.workspace_requirements(x0.len())?;
        workspace.try_reserve(&requirements)?;
        workspace.storage.bounds.assign(problem.bounds)?;
        if !workspace.storage.bounds.contains(x0) {
            return Err(invalid(
                scioxide_core::argument!("x0"),
                "initial point must be finite and inside bounds",
            ));
        }
        let shared = shared_objective(problem, self.options.execution)?;
        crate::attempt::preflight(&scope, None)?;
        let entry = Entry::Call {
            identity: super::multistart::identity(
                scioxide_core::algorithm_id!("optimize", "CmaEs", 1),
                x0.len(),
                &self.options,
                requirements.token,
            ),
            budgets: SolveBudgets::unbounded(),
        };
        let token = requirements.token;
        let initial = self.initial_population(x0.len());
        crate::attempt::minimize(
            &mut workspace.shell,
            &mut workspace.storage,
            &mut workspace.staged,
            output,
            scope,
            entry,
            |context, storage, staged| {
                let session = SolveSession::new(context, scioxide_core::argument!("scope"));
                let mut generator = Xoshiro256StarStar::from_seed(self.options.seed);
                storage.local_error = None;
                let CmaEsStorage { state, bounds, .. } = storage;
                let state = state.as_mut().ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("workspace"),
                        "unprepared workspace",
                    )
                })?;
                let result = match shared {
                    None => self.run(
                        problem,
                        x0,
                        initial,
                        bounds,
                        state,
                        &session,
                        &mut generator,
                        &|state, n, population, session| {
                            evaluate_sequential(problem, state, n, population, session)
                        },
                    ),
                    Some(objective) => self.run(
                        problem,
                        x0,
                        initial,
                        bounds,
                        state,
                        &session,
                        &mut generator,
                        &|state, n, population, session| {
                            evaluate_partitioned(
                                objective,
                                state,
                                n,
                                population,
                                session,
                                self.options.execution,
                            )
                        },
                    ),
                };
                let termination = match result {
                    Ok(stop) => stop,
                    Err(error) => match session.stopped() {
                        Some(stop) if state.best_value.is_some() => stop,
                        _ => return Err(error),
                    },
                };
                publish(staged, state, &session, termination)?;
                self.refine(problem, staged, storage, token, &session)
            },
        )
    }

    /// Refine the staged search best with the prepared local solver, if configured.
    fn refine(
        &self,
        problem: &BoundedObjective<'_, S>,
        staged: &mut OptimizeResult<S>,
        storage: &mut CmaEsStorage<S, F::Solver>,
        token: u128,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        let CmaEsStorage {
            state,
            bounds,
            local,
            local_result,
            local_error,
        } = storage;
        let Some(solver) = local.locals.first_mut() else {
            return Ok(());
        };
        let mut child = SolveState::new();
        let identity = local_identity(0, bounds.len(), token);
        let outcome =
            session.with_child(&mut child, identity, self.options.local_budget, |context| {
                solver.minimize_into(
                    problem,
                    staged.x(),
                    self.options.seed,
                    local_result,
                    context,
                )
            });
        let accepted = match outcome {
            Ok(()) => {
                validate_local(local_result, bounds)?;
                local_result.fun() < staged.fun()
            }
            Err(error) if recoverable(&error) => {
                *local_error = Some(error);
                false
            }
            Err(error) => return Err(error),
        };
        if accepted {
            staged.x.copy_from_slice(local_result.x());
            staged.fun = local_result.fun();
            if let Some(state) = state {
                state.best.copy_from_slice(local_result.x());
                state.best_value = Some(local_result.fun());
            }
            session.accept(PartialResultTrust::Admissible)?;
        }
        staged.counts = session.accounting()?;
        staged.fun_evals = staged.counts.total()?.provider(EvaluationRole::Objective);
        staged.grad_evals = staged.counts.total()?.provider(EvaluationRole::Gradient);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        problem: &BoundedObjective<'_, S>,
        x0: &[S],
        initial: usize,
        bounds: &FiniteBounds<S>,
        work: &mut CmaEsState<S>,
        session: &SolveSession<'_, '_, '_>,
        generator: &mut dyn BitGenerator,
        evaluate: &Evaluate<'_, S>,
    ) -> Result<OptimizeTermination, SciOxideError> {
        let n = x0.len();
        work.reports.clear();
        work.best_value = None;
        work.restarts = 0;
        work.last_covariance_rank = n;
        work.last_covariance_log_condition = S::ZERO;
        let mut population = initial;
        let normal = Normal::new(S::ZERO, S::ONE)?;
        for restart in 0..=self.options.restart.max_restarts() {
            if restart == 0 {
                for (coordinate, (&value, &(lower, upper))) in
                    work.mean.iter_mut().zip(x0.iter().zip(bounds.as_slice()))
                {
                    *coordinate = normalized_coordinate(value, lower, upper)?;
                }
                work.physical.copy_from_slice(x0);
            } else {
                let mut admission = session;
                self.design.point(
                    restart - 1,
                    bounds,
                    &mut SearchRandom::new(&mut *generator, &mut admission),
                    &mut work.physical,
                )?;
                for (coordinate, (&value, &(lower, upper))) in work
                    .mean
                    .iter_mut()
                    .zip(work.physical.iter().zip(bounds.as_slice()))
                {
                    if !value.is_finite() || value < lower || value > upper {
                        return Err(invalid(
                            scioxide_core::argument!("design"),
                            "restart design returned a point outside bounds",
                        ));
                    }
                    *coordinate = normalized_coordinate(value, lower, upper)?;
                }
                work.restarts = restart;
            }
            work.covariance.as_mut_slice().fill(S::ZERO);
            for i in 0..n {
                work.covariance.as_mut_slice()[i * n + i] = S::ONE;
            }
            work.path_sigma.fill(S::ZERO);
            work.path_covariance.fill(S::ZERO);
            let mut sigma = self.options.initial_step;
            let mut path_decay = S::ONE;
            let parameters = Parameters::new(n, population, &mut work.weights)?;
            let start_value =
                session.invoke(EvaluationRole::Objective, || problem.eval(&work.physical))?;
            crate::utils::validate_finite(&[start_value])?;
            record_best(work, start_value, session)?;
            loop {
                if work.reports.len() >= self.options.max_generations.get() {
                    return Ok(OptimizeTermination::SearchBudgetCompleted);
                }
                if let Some(stop) = session.iteration()? {
                    return Ok(stop);
                }
                session.require_evaluations(population)?;
                let (rank, condition, largest) = factor_covariance(work, n)?;
                work.last_covariance_rank = rank;
                work.last_covariance_log_condition = condition;
                if rank < n || sigma == S::ZERO {
                    break;
                }
                let repaired = sample_population(
                    bounds, work, n, population, sigma, &normal, session, generator,
                )?;
                evaluate(work, n, population, session)?;
                work.order[..population]
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, value)| *value = i);
                work.order[..population].sort_unstable_by(|&left, &right| {
                    work.fitness[left]
                        .partial_cmp(&work.fitness[right])
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| left.cmp(&right))
                });
                let best_fitness = work.fitness[work.order[0]];
                let worst_fitness = work.fitness[work.order[population - 1]];
                update_distribution(work, n, &parameters, &mut sigma, &mut path_decay)?;
                work.reports.push(CmaEsGeneration {
                    generation: work.reports.len() + 1,
                    restart,
                    best_fitness,
                    worst_fitness,
                    covariance_rank: rank,
                    covariance_log_condition: condition,
                    step_size: sigma,
                    repaired_coordinates: repaired,
                    counts: session.accounting()?,
                });
                let spread = finite(worst_fitness - best_fitness, "fitness spread overflow")?;
                let scale = NumericScalar::abs(best_fitness).max(NumericScalar::abs(worst_fitness));
                let effective_step =
                    finite(sigma * RealScalar::sqrt(largest), "step scale overflow")?;
                if self.options.fitness_tolerance.is_satisfied(spread, scale)?
                    || self
                        .options
                        .step_tolerance
                        .is_satisfied(effective_step, S::ONE)?
                {
                    break;
                }
            }
            if restart == self.options.restart.max_restarts() {
                return Ok(OptimizeTermination::PopulationStagnation);
            }
            population = population.checked_mul(2).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("cma_es_restart_population"))
            })?;
        }
        Ok(OptimizeTermination::PopulationStagnation)
    }
}

fn publish<S: RealScalar + NumericScalar<Real = S>>(
    output: &mut OptimizeResult<S>,
    work: &CmaEsState<S>,
    session: &SolveSession<'_, '_, '_>,
    termination: OptimizeTermination,
) -> Result<(), SciOxideError> {
    let best = work.best_value.ok_or_else(|| {
        invalid(
            scioxide_core::argument!("scope"),
            "no objective was observed",
        )
    })?;
    let counts = session.combined_counts()?;
    let algorithm = session.algorithm()?;
    output.x.clear();
    output.method = crate::OptimizeDiagnostics::Unavailable;
    output.x.extend_from_slice(&work.best);
    output.fun = best;
    output.grad = None;
    output.iterations = work.reports.len();
    output.fun_evals = counts.provider(EvaluationRole::Objective);
    output.grad_evals = counts.provider(EvaluationRole::Gradient);
    output.termination = termination;
    output.message.clear();
    output
        .message
        .push_str("CMA-ES search result; global optimality is unverified");
    output.counts = session.accounting()?;
    output.algorithm = Some(algorithm);
    output.trust = PartialResultTrust::Admissible;
    output.diagnostics = OptimalityDiagnostics {
        bound_violation: Some(S::ZERO),
        ..OptimalityDiagnostics::unavailable()
    };
    Ok(())
}

impl<S, D, F> CmaEs<S, D, F>
where
    S: RealScalar + NumericScalar<Real = S>,
    D: StartDesign<S>,
    F: LocalSolverFactory<S>,
{
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

impl<'a, S, D, F> crate::traits::Optimizer<S, BoundedObjective<'a, S>> for CmaEs<S, D, F>
where
    S: RealScalar + NumericScalar<Real = S>,
    D: StartDesign<S>,
    F: LocalSolverFactory<S>,
{
    fn minimize(
        &self,
        problem: &BoundedObjective<'a, S>,
        x0: &[S],
    ) -> Result<OptimizeResult<S>, SciOxideError> {
        self.minimize(problem, x0)
    }
}

struct Parameters<S: RealScalar> {
    mu: usize,
    mu_effective: S,
    c_sigma: S,
    d_sigma: S,
    c_covariance: S,
    c_one: S,
    c_mu: S,
    expected_norm: S,
}

impl<S: RealScalar> Parameters<S> {
    fn new(dimension: usize, population: usize, weights: &mut [S]) -> Result<Self, SciOxideError> {
        let number = S::try_from_f64_lossy;
        let n = S::try_from_usize_exact(dimension)?;
        let mu = population / 2;
        let half = number(0.5)?;
        let peak = RealScalar::ln(S::try_from_usize_exact(mu)? + half);
        let mut total = S::ZERO;
        for (index, weight) in weights[..mu].iter_mut().enumerate() {
            *weight = peak - RealScalar::ln(S::try_from_usize_exact(index + 1)?);
            total += *weight;
        }
        for weight in &mut weights[..mu] {
            *weight /= total;
        }
        let squared = weights[..mu]
            .iter()
            .fold(S::ZERO, |sum, weight| sum + *weight * *weight);
        let mu_effective = S::ONE / squared;
        let c_sigma = (mu_effective + number(2.)?) / (n + mu_effective + number(5.)?);
        let d_sigma = S::ONE
            + number(2.)?
                * (RealScalar::sqrt((mu_effective - S::ONE) / (n + S::ONE)) - S::ONE).max(S::ZERO)
            + c_sigma;
        let c_covariance =
            (number(4.)? + mu_effective / n) / (n + number(4.)? + number(2.)? * mu_effective / n);
        let c_one = number(2.)? / ((n + number(1.3)?).powi(2) + mu_effective);
        let c_mu = (number(2.)? * (mu_effective - number(2.)? + S::ONE / mu_effective)
            / ((n + number(2.)?).powi(2) + mu_effective))
            .min(S::ONE - c_one);
        let expected_norm = RealScalar::sqrt(n)
            * (S::ONE - S::ONE / (number(4.)? * n) + S::ONE / (number(21.)? * n * n));
        for value in [
            mu_effective,
            c_sigma,
            d_sigma,
            c_covariance,
            c_one,
            c_mu,
            expected_norm,
        ] {
            finite(value, "adaptation parameter overflow")?;
        }
        Ok(Self {
            mu,
            mu_effective,
            c_sigma,
            d_sigma,
            c_covariance,
            c_one,
            c_mu,
            expected_norm,
        })
    }
}

fn record_best<S: RealScalar + NumericScalar<Real = S>>(
    work: &mut CmaEsState<S>,
    value: S,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    if work.best_value.is_none_or(|best| value < best) {
        work.best.copy_from_slice(&work.physical);
        work.best_value = Some(value);
        session.accept(PartialResultTrust::Admissible)?;
    }
    Ok(())
}

fn factor_covariance<S: RealScalar + NumericScalar<Real = S>>(
    work: &mut CmaEsState<S>,
    n: usize,
) -> Result<(usize, S, S), SciOxideError> {
    HermitianEigenDecomposition::decompose_into(
        &work.covariance,
        scioxide_linalg::HermitianInput::Exact,
        &HermitianEigenOptions::default(),
        &mut work.eig,
        &mut work.eig_work,
        scioxide_core::ExecutionContext::sequential(),
    )?;
    let values = work.eig.eigenvalues();
    let largest = values.iter().copied().fold(S::ZERO, RealScalar::max);
    let cutoff = S::try_from_usize_exact(n)? * S::EPSILON * largest;
    let allowance = S::try_from_f64_lossy(8.)? * cutoff;
    if values.iter().any(|&value| value < -allowance) {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "covariance has a materially negative eigenvalue",
        ));
    }
    let smallest = values
        .iter()
        .copied()
        .filter(|&value| value > cutoff)
        .fold(largest, RealScalar::min);
    let rank = values.iter().filter(|&&value| value > cutoff).count();
    let log_condition = if rank == 0 {
        S::ZERO
    } else {
        finite(
            RealScalar::ln(largest) - RealScalar::ln(smallest),
            "covariance condition overflow",
        )?
    };
    Ok((rank, log_condition, largest))
}

#[allow(clippy::too_many_arguments)]
fn sample_population<S>(
    bounds: &FiniteBounds<S>,
    work: &mut CmaEsState<S>,
    n: usize,
    population: usize,
    sigma: S,
    normal: &Normal<S>,
    session: &SolveSession<'_, '_, '_>,
    generator: &mut dyn BitGenerator,
) -> Result<usize, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    let mut repaired = 0;
    for member in 0..population {
        session.proposal()?;
        let mut admission = session;
        let mut random = SearchRandom::new(&mut *generator, &mut admission);
        for value in &mut work.normal {
            *value = normal.sample(&mut random)?;
        }
        for coordinate in 0..n {
            let mut direction = S::ZERO;
            for column in 0..n {
                let variance = work.eig.eigenvalues()[column].max(S::ZERO);
                direction += work.eig.eigenvectors().as_slice()[coordinate * n + column]
                    * RealScalar::sqrt(variance)
                    * work.normal[column];
            }
            let proposal = finite(
                work.mean[coordinate] + sigma * direction,
                "sample coordinate overflow",
            )?;
            let unit = repair(proposal, DifferentialEvolutionBoundary::Reflect);
            repaired += usize::from(unit != proposal);
            work.steps[member * n + coordinate] = finite(
                (unit - work.mean[coordinate]) / sigma,
                "sample step overflow",
            )?;
            let (lower, upper) = bounds.as_slice()[coordinate];
            work.physical[coordinate] = scioxide_core::checked_affine(lower, upper, unit)?;
        }
        work.points[member * n..(member + 1) * n].copy_from_slice(&work.physical);
    }
    Ok(repaired)
}

fn evaluate_sequential<S: RealScalar + NumericScalar<Real = S>>(
    f: &dyn ObjectiveFunction<S>,
    work: &mut CmaEsState<S>,
    n: usize,
    population: usize,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    for member in 0..population {
        let point = &work.points[member * n..(member + 1) * n];
        let value = session.invoke(EvaluationRole::Objective, || f.eval(point))?;
        crate::utils::validate_finite(&[value])?;
        work.physical.copy_from_slice(point);
        record_best(work, value, session)?;
        work.fitness[member] = value;
    }
    Ok(())
}

fn evaluate_partitioned<S: RealScalar + NumericScalar<Real = S>>(
    f: &(dyn ObjectiveFunction<S> + Sync),
    work: &mut CmaEsState<S>,
    n: usize,
    population: usize,
    session: &SolveSession<'_, '_, '_>,
    policy: ExecutionPolicy,
) -> Result<(), SciOxideError> {
    let execution = session.execution()?;
    let limit = core::num::NonZeroUsize::MIN.saturating_add(63);
    let plan = execution.plan(
        policy,
        population,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: core::num::NonZeroUsize::MIN,
            worker_limit: limit,
        },
    );
    for _ in 0..population {
        session.admit(EvaluationRole::Objective)?;
    }
    let mut slots: [_; 64] =
        core::array::from_fn(|_| scioxide_core::backend::PartitionSlot::empty());
    let mut remaining = &mut work.fitness[..population];
    for partition in plan.partitions() {
        let (values, tail) = remaining.split_at_mut(partition.range().len());
        remaining = tail;
        slots[partition.worker_index()] = scioxide_core::backend::PartitionSlot::new(values);
    }
    let points = &work.points[..population * n];
    execution.execute_partitioned(
        plan,
        &slots[..plan.worker_count()],
        &|partition, values, _| {
            for (index, value) in partition.range().zip(values.iter_mut()) {
                *value = f.eval(&points[index * n..(index + 1) * n])?;
                crate::utils::validate_finite(&[*value])?;
            }
            Ok(())
        },
    )?;

    for member in 0..population {
        work.physical
            .copy_from_slice(&work.points[member * n..(member + 1) * n]);
        record_best(work, work.fitness[member], session)?;
    }
    Ok(())
}

fn update_distribution<S: RealScalar + NumericScalar<Real = S>>(
    work: &mut CmaEsState<S>,
    n: usize,
    parameters: &Parameters<S>,
    sigma: &mut S,
    path_decay: &mut S,
) -> Result<(), SciOxideError> {
    work.mean_step.fill(S::ZERO);
    for (rank, &member) in work.order[..parameters.mu].iter().enumerate() {
        let row = &work.steps[member * n..(member + 1) * n];
        for (coordinate, &step) in row.iter().enumerate() {
            work.mean_step[coordinate] = work.mean_step[coordinate] + work.weights[rank] * step;
        }
    }
    // Whiten the selected step using the same factor that generated the population.
    work.normal.fill(S::ZERO);
    for column in 0..n {
        let eigenvalue = work.eig.eigenvalues()[column];
        let mut projection = S::ZERO;
        for row in 0..n {
            projection +=
                work.eig.eigenvectors().as_slice()[row * n + column] * work.mean_step[row];
        }
        let scaled = projection / RealScalar::sqrt(eigenvalue);
        for row in 0..n {
            work.normal[row] =
                work.normal[row] + work.eig.eigenvectors().as_slice()[row * n + column] * scaled;
        }
    }
    let two = S::TWO;
    let sigma_gain =
        RealScalar::sqrt(parameters.c_sigma * (two - parameters.c_sigma) * parameters.mu_effective);
    for row in 0..n {
        work.path_sigma[row] = finite(
            (S::ONE - parameters.c_sigma) * work.path_sigma[row] + sigma_gain * work.normal[row],
            "sigma path overflow",
        )?;
    }
    *path_decay *= (S::ONE - parameters.c_sigma).powi(2);
    let sigma_norm = RealScalar::sqrt(
        work.path_sigma
            .iter()
            .fold(S::ZERO, |sum, value| sum + *value * *value),
    );
    let normalized_path =
        sigma_norm / RealScalar::sqrt(S::ONE - *path_decay) / parameters.expected_norm;
    let n_scalar = S::try_from_usize_exact(n)?;
    let h_sigma = normalized_path < S::try_from_f64_lossy(1.4)? + two / (n_scalar + S::ONE);
    let covariance_gain = RealScalar::sqrt(
        parameters.c_covariance * (two - parameters.c_covariance) * parameters.mu_effective,
    );
    for row in 0..n {
        work.path_covariance[row] = finite(
            (S::ONE - parameters.c_covariance) * work.path_covariance[row]
                + if h_sigma {
                    covariance_gain * work.mean_step[row]
                } else {
                    S::ZERO
                },
            "covariance path overflow",
        )?;
        work.mean[row] = finite(
            work.mean[row] + *sigma * work.mean_step[row],
            "mean overflow",
        )?
        .max(S::ZERO)
        .min(S::ONE);
    }
    let retention = S::ONE - parameters.c_one - parameters.c_mu
        + if h_sigma {
            S::ZERO
        } else {
            parameters.c_one * parameters.c_covariance * (two - parameters.c_covariance)
        };
    for row in 0..n {
        for column in 0..=row {
            let mut value = retention * work.covariance.as_slice()[row * n + column]
                + parameters.c_one * work.path_covariance[row] * work.path_covariance[column];
            for (rank, &member) in work.order[..parameters.mu].iter().enumerate() {
                let step = &work.steps[member * n..(member + 1) * n];
                value += parameters.c_mu * work.weights[rank] * step[row] * step[column];
            }
            value = finite(value, "covariance update overflow")?;
            work.next_covariance.as_mut_slice()[row * n + column] = value;
            work.next_covariance.as_mut_slice()[column * n + row] = value;
        }
    }
    core::mem::swap(&mut work.covariance, &mut work.next_covariance);
    let multiplier = RealScalar::exp(
        (parameters.c_sigma / parameters.d_sigma)
            * (sigma_norm / parameters.expected_norm - S::ONE),
    );
    *sigma = finite(*sigma * multiplier, "step-size update overflow")?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn covariance_rank_loss_and_negative_spectrum_are_distinct() {
        let mut work = CmaEsState::<f64>::prepare(2, 6, 1).unwrap();
        work.covariance
            .as_mut_slice()
            .copy_from_slice(&[1., 0., 0., 0.]);
        let (rank, condition, largest) = factor_covariance(&mut work, 2).unwrap();
        assert_eq!(rank, 1);
        assert!(condition.abs() <= 1e-12);
        assert!((largest - 1.0).abs() <= 1e-12);
        work.covariance
            .as_mut_slice()
            .copy_from_slice(&[1., 0., 0., -0.01]);
        assert!(factor_covariance(&mut work, 2).is_err());
    }

    #[test]
    fn excessive_restart_capacity_fails_before_observation() {
        let solver = CmaEs::new(CmaEsOptions::<f64>::default().with_restart(
            CmaEsRestartPolicy::IncreasingPopulation {
                max_restarts: usize::MAX,
            },
        ))
        .unwrap();
        assert!(solver.workspace(1).is_err());
    }

    #[test]
    fn construction_validates_step_and_population() {
        let options = CmaEsOptions::<f64>::default();
        assert!(CmaEs::new(options.clone().with_initial_step(0.0)).is_err());
        assert!(CmaEs::new(options.clone().with_initial_step(f64::NAN)).is_err());
        assert!(
            CmaEs::new(
                options
                    .clone()
                    .with_population_size(Budget::try_from(3).unwrap())
            )
            .is_err()
        );
        assert!(CmaEs::new(options).is_ok());
        assert_eq!(CmaEsOptions::<f64>::default().initial_step(), 0.3);
        assert_eq!(CmaEsOptions::<f32>::default().initial_step(), 0.3);
    }
}
