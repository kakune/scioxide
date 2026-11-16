//! Basin hopping and dual annealing.
use scioxide_core::{
    AttemptKind, EvaluationRole, NumericScalar, PartialResultTrust, RealScalar, SciOxideError,
    SolveBudgets, SolveContext, SolveScope,
};
use scioxide_random::{Distribution, SeedableRng, StudentT, Xoshiro256StarStar};

use super::local::{LocalSolverFactory, run_local};
use super::multistart::{
    GlobalSearchWorkspaceRequirements, SearchStorage, bind, finish, identity, requirements,
    validate_point,
};
use super::{StartDesign, UniformStartDesign, sequential_only};
use crate::context::SolveSession;
use crate::geometry::{normalized_coordinate, repair};
use crate::validation::invalid;
use crate::{
    BoundedObjective, DifferentialEvolutionBoundary, FiniteBounds, GlobalSearchOptions,
    GlobalSearchResult, GlobalSearchWorkspace, ObjectiveFunction, OptimizeTermination,
    SearchRandom,
};

/// Proposes the next point of a basin-hopping chain from the current point.
pub trait BasinProposal<S: RealScalar> {
    /// Writes a point inside `bounds`, displaced from `current`, to `output`.
    ///
    /// Implementations must draw random numbers only from `random`, so that
    /// the chain is reproducible from the seed.
    ///
    /// # Errors
    ///
    /// Returns proposal errors, and the errors of `random` when a draw is
    /// refused or the generator fails. A point outside the box is rejected by
    /// the chain with `InvalidInput`.
    fn propose(
        &self,
        bounds: &FiniteBounds<S>,
        current: &[S],
        random: &mut SearchRandom<'_>,
        output: &mut [S],
    ) -> Result<(), SciOxideError>;
}

/// Displaces each coordinate by an independent uniform amount in
/// `[−step, step]` times the box width, reflected at the bounds.
#[derive(Debug, Clone, Copy)]
pub struct UniformBasinProposal<S: RealScalar> {
    step: S,
}
impl<S: RealScalar> UniformBasinProposal<S> {
    /// Creates a proposal with step `step`, as a fraction of each box width.
    ///
    /// # Errors
    ///
    /// `InvalidInput` if `step` is NaN, infinite or not positive.
    pub fn new(step: S) -> Result<Self, SciOxideError> {
        if !step.is_finite() || step <= S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("step"),
                "basin step must be finite and positive",
            ));
        }
        Ok(Self { step })
    }
    /// Returns the step as a fraction of the box width.
    pub const fn step(&self) -> S {
        self.step
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> BasinProposal<S> for UniformBasinProposal<S> {
    fn propose(
        &self,
        bounds: &FiniteBounds<S>,
        current: &[S],
        random: &mut SearchRandom<'_>,
        output: &mut [S],
    ) -> Result<(), SciOxideError> {
        for ((x, &old), &(lo, hi)) in output.iter_mut().zip(current).zip(bounds.as_slice()) {
            let unit = normalized_coordinate(old, lo, hi)?;
            let delta = (random.uniform::<S>()? * (S::TWO) - S::ONE) * self.step;
            // Fold the displacement first, avoiding overflow in unit + delta.
            let next = repair(
                unit + delta % (S::TWO),
                DifferentialEvolutionBoundary::Reflect,
            );
            *x = scioxide_core::checked_affine(lo, hi, next)?;
        }
        Ok(())
    }
}

/// Configures [`BasinHopping`]: the search options and the Metropolis
/// temperature.
///
/// Temperature zero accepts only downhill steps (monotonic basin hopping). See
/// [Termination](crate#termination) for how the chain ends.
#[derive(Debug, Clone)]
pub struct BasinHoppingOptions<S: RealScalar> {
    multistart: GlobalSearchOptions<S>,
    temperature: S,
}
impl<S: RealScalar> BasinHoppingOptions<S> {
    /// Creates options with a Metropolis temperature in objective units.
    ///
    /// # Errors
    ///
    /// `InvalidInput` if `temperature` is NaN, infinite or negative.
    pub fn new(multistart: GlobalSearchOptions<S>, temperature: S) -> Result<Self, SciOxideError> {
        if !temperature.is_finite() || temperature < S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("temperature"),
                "invalid basin temperature",
            ));
        }
        Ok(Self {
            multistart,
            temperature,
        })
    }
    /// Returns the search options.
    pub const fn multistart(&self) -> &GlobalSearchOptions<S> {
        &self.multistart
    }
    /// Returns the Metropolis temperature in objective units.
    pub const fn temperature(&self) -> S {
        self.temperature
    }
}

/// Configures [`DualAnnealing`]: the search options, the temperature
/// schedule, the visiting and acceptance parameters and local refinement.
///
/// Coordinates are scaled to `[0, 1)` across the box, and a visiting step that
/// leaves the box wraps around to the other side. See
/// [Termination](crate#termination) for how the chain ends.
#[derive(Debug, Clone)]
pub struct DualAnnealingOptions<S: RealScalar> {
    multistart: GlobalSearchOptions<S>,
    temperature: S,
    visit: S,
    accept: S,
    restart: S,
    local: bool,
}
impl<S: RealScalar> Default for DualAnnealingOptions<S> {
    fn default() -> Self {
        Self::new(GlobalSearchOptions::default())
    }
}
impl<S: RealScalar> DualAnnealingOptions<S> {
    /// Creates options with initial temperature `T₀ = 5230`, visiting
    /// parameter `q_v = 2.62`, acceptance parameter `q_a = −5`, restart ratio
    /// `2·10⁻⁵` and local refinement enabled.
    pub fn new(multistart: GlobalSearchOptions<S>) -> Self {
        let five = S::TWO + S::TWO + S::ONE;
        let ten = five + five;
        let hundred = ten * ten;
        Self {
            multistart,
            temperature: (five * ten + S::TWO) * hundred + (S::TWO + S::ONE) * ten,
            visit: (S::TWO * hundred + (five + S::ONE) * ten + S::TWO) / hundred,
            accept: -five,
            restart: S::TWO / (hundred * hundred * ten),
            local: true,
        }
    }
    /// Sets the initial temperature `T₀ > 0`, the visiting parameter
    /// `1 < q_v < 3`, the acceptance parameter `q_a < 1` and the restart ratio
    /// in `(0, 1)`.
    ///
    /// The chain reanneals, restarting at `T₀` from a uniform random point,
    /// when the temperature falls to `T₀` times the restart ratio.
    pub fn with_schedule(mut self, temperature: S, visit: S, accept: S, restart: S) -> Self {
        self.temperature = temperature;
        self.visit = visit;
        self.accept = accept;
        self.restart = restart;
        self
    }
    /// Enables or disables local refinement. When enabled, the local solver
    /// refines the initial point, each new best point and each reannealed
    /// point.
    pub const fn with_local_search(mut self, enabled: bool) -> Self {
        self.local = enabled;
        self
    }
    /// Returns the search options.
    pub const fn multistart(&self) -> &GlobalSearchOptions<S> {
        &self.multistart
    }
    /// Returns whether local refinement is enabled.
    pub const fn local_search(&self) -> bool {
        self.local
    }
}

/// Minimizes an objective over a finite box by basin hopping.
///
/// Each step displaces the current point with a [`BasinProposal`], minimizes
/// from there with a local solver, and accepts the local minimum with the
/// Metropolis probability `min(1, exp(−(f_new − f_cur)/T))`, where `T` is the
/// temperature of [`BasinHoppingOptions`]. Step 0 refines `x0`. The number of
/// steps is [`GlobalSearchOptions::with_starts`](crate::GlobalSearchOptions::with_starts),
/// and each step costs one local solve. The chain is sequential.
///
/// The result has the same form as for [`MultiStart`](crate::MultiStart).
/// The search does not prove that it found the global minimum.
///
/// # References
///
/// - D. J. Wales and J. P. K. Doye, "Global optimization by basin-hopping and
///   the lowest energy structures of Lennard-Jones clusters containing up to
///   110 atoms", *J. Phys. Chem. A* 101(28), 5111–5116, 1997.
#[derive(Debug, Clone)]
pub struct BasinHopping<S: RealScalar, Proposal, Factory> {
    options: BasinHoppingOptions<S>,
    proposal: Proposal,
    factory: Factory,
}
impl<S, P, F> BasinHopping<S, P, F>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: BasinProposal<S>,
    F: LocalSolverFactory<S>,
{
    /// Creates a method from options, a proposal and a local solver factory.
    ///
    /// # Errors
    ///
    /// `InvalidInput` if a basin tolerance is NaN, infinite or not positive, or
    /// the execution policy is not sequential.
    pub fn new(
        options: BasinHoppingOptions<S>,
        proposal: P,
        factory: F,
    ) -> Result<Self, SciOxideError> {
        options.multistart.validate()?;
        sequential_only(
            options.multistart.execution(),
            "basin hopping is a sequential chain",
        )?;
        Ok(Self {
            options,
            proposal,
            factory,
        })
    }
    /// Returns the options.
    pub const fn options(&self) -> &BasinHoppingOptions<S> {
        &self.options
    }
    /// Returns the workspace size for `dimension` variables.
    ///
    /// # Errors
    ///
    /// `SizeOverflow` if the size overflows.
    pub fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<GlobalSearchWorkspaceRequirements<'_, F>, SciOxideError> {
        requirements(&self.options.multistart, &self.factory, dimension)
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
    /// Runs the chain from `x0` and writes the step reports and best point to
    /// `output`.
    ///
    /// A failed local solve is recorded in its step's report and does not
    /// replace the current point. When a limit, cancellation or a deadline
    /// stops the chain, `output` holds the steps that ran.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a variable lacks a lower or upper bound or has
    ///   `lo ≥ hi`, `x0` is not finite and inside the box, a proposal leaves
    ///   the box, or a local solver returns a point outside the box or a
    ///   non-finite value.
    /// - `CapabilityUnavailable` if a time limit is set and the context has no
    ///   clock.
    /// - Errors of the objective, the proposal, and of a local solver other
    ///   than those recorded per step.
    pub fn minimize_into<'scope, 'capability>(
        &self,
        problem: &BoundedObjective<'_, S>,
        x0: &[S],
        output: &mut GlobalSearchResult<S>,
        workspace: &mut GlobalSearchWorkspace<S, F::Solver>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let kernel = ChainKernel::Basin {
            proposal: &self.proposal,
            temperature: self.options.temperature,
        };
        let algorithm = scioxide_core::algorithm_id!("optimize", "BasinHopping", 1);
        let requirements = self.workspace_requirements(problem.dim());
        run(
            &self.options.multistart,
            &self.options,
            algorithm,
            requirements,
            kernel,
            problem,
            x0,
            output,
            workspace,
            scope.into(),
        )
    }
    /// Runs the chain with a new workspace and returns the result.
    ///
    /// # Errors
    ///
    /// As [`Self::workspace`] and [`Self::minimize_into`].
    pub fn minimize(
        &self,
        problem: &BoundedObjective<'_, S>,
        x0: &[S],
    ) -> Result<GlobalSearchResult<S>, SciOxideError> {
        let mut workspace = self.workspace(problem.dim())?;
        let mut output = GlobalSearchResult::empty();
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

/// Minimizes an objective over a finite box by dual annealing: generalized
/// simulated annealing with optional local refinement.
///
/// The chain visits new points with a heavy-tailed distribution (Tsallis
/// q-Gaussian with parameter `q_v`, sampled through a Student t distribution)
/// whose width shrinks with the temperature
///
/// ```text
/// T(k) = T₀ (2^(q_v − 1) − 1) / ((k + 2)^(q_v − 1) − 1),   k = 0, 1, 2, …
/// ```
///
/// where `k` counts sweeps of `2n` steps and `n` is the number of variables.
/// In a sweep, the first `n` steps move all coordinates and the next `n`
/// steps move one coordinate each. A move to a higher value is accepted with
/// the generalized Metropolis probability of parameter `q_a`. When `T(k)`
/// falls to `T₀` times the restart ratio, the chain reanneals from a uniform
/// random point. With local refinement enabled, a local solver refines the
/// first point, each new best point and each reannealed point. Each step
/// costs one objective evaluation plus any local solve. The chain is
/// sequential.
///
/// The result has the same form as for [`MultiStart`](crate::MultiStart).
/// The search does not prove that it found the global minimum.
///
/// # References
///
/// - C. Tsallis and D. A. Stariolo, "Generalized simulated annealing",
///   *Physica A* 233(1–2), 395–406, 1996.
/// - Y. Xiang, D. Y. Sun, W. Fan and X. G. Gong, "Generalized simulated
///   annealing algorithm and its application to the Thomson model",
///   *Phys. Lett. A* 233(3), 216–220, 1997.
#[derive(Debug, Clone)]
pub struct DualAnnealing<S: RealScalar, Factory> {
    options: DualAnnealingOptions<S>,
    factory: Factory,
    visiting: StudentT<S>,
}
impl<S, F> DualAnnealing<S, F>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: LocalSolverFactory<S>,
{
    /// Creates a method from options and a local solver factory.
    ///
    /// # Errors
    ///
    /// `InvalidInput` if a basin tolerance is NaN, infinite or not positive, or
    /// the execution policy is not sequential. Also returns `InvalidInput` with
    /// argument `temperature` for a non-finite schedule, `T₀ ≤ 0`, `q_v` outside
    /// `(1, 3)`, `q_a ≥ 1`, or a restart ratio outside `(0, 1)`.
    pub fn new(options: DualAnnealingOptions<S>, factory: F) -> Result<Self, SciOxideError> {
        options.multistart.validate()?;
        sequential_only(
            options.multistart.execution(),
            "dual annealing is a sequential chain",
        )?;
        let three = S::TWO + S::ONE;
        if [
            options.temperature,
            options.visit,
            options.accept,
            options.restart,
        ]
        .iter()
        .any(|x| !x.is_finite())
            || options.temperature <= S::ZERO
            || options.visit <= S::ONE
            || options.visit >= three
            || options.accept >= S::ONE
            || options.restart <= S::ZERO
            || options.restart >= S::ONE
        {
            return Err(invalid(
                scioxide_core::argument!("temperature"),
                "invalid generalized annealing schedule",
            ));
        }
        let visiting = StudentT::new((three - options.visit) / (options.visit - S::ONE))?;
        Ok(Self {
            options,
            factory,
            visiting,
        })
    }
    /// Returns the options.
    pub const fn options(&self) -> &DualAnnealingOptions<S> {
        &self.options
    }
    /// Returns the workspace size for `dimension` variables.
    ///
    /// # Errors
    ///
    /// `SizeOverflow` if the size overflows.
    pub fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<GlobalSearchWorkspaceRequirements<'_, F>, SciOxideError> {
        requirements(&self.options.multistart, &self.factory, dimension)
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
    /// Runs the chain from `x0` and writes the step reports and best point to
    /// `output`.
    ///
    /// A failed or worse local refinement does not replace the annealing
    /// point. When a limit, cancellation or a deadline stops the chain,
    /// `output` holds the steps that ran.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a variable lacks a lower or upper bound or has
    ///   `lo ≥ hi`, `x0` is not finite and inside the box, or a local solver
    ///   returns a point outside the box or a non-finite value.
    /// - `NonFinite` if the objective returns NaN or infinity at a visited
    ///   point.
    /// - `NumericalInstability` (`Overflow`) if a visiting step overflows.
    /// - `CapabilityUnavailable` if a time limit is set and the context has no
    ///   clock.
    /// - Errors of the objective and of a local solver other than those
    ///   recorded per step.
    pub fn minimize_into<'scope, 'capability>(
        &self,
        problem: &BoundedObjective<'_, S>,
        x0: &[S],
        output: &mut GlobalSearchResult<S>,
        workspace: &mut GlobalSearchWorkspace<S, F::Solver>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let kernel = ChainKernel::Anneal {
            options: &self.options,
            visiting: &self.visiting,
        };
        let algorithm = scioxide_core::algorithm_id!("optimize", "DualAnnealing", 1);
        let requirements = self.workspace_requirements(problem.dim());
        run(
            &self.options.multistart,
            &self.options,
            algorithm,
            requirements,
            kernel,
            problem,
            x0,
            output,
            workspace,
            scope.into(),
        )
    }
    /// Runs the chain with a new workspace and returns the result.
    ///
    /// # Errors
    ///
    /// As [`Self::workspace`] and [`Self::minimize_into`].
    pub fn minimize(
        &self,
        problem: &BoundedObjective<'_, S>,
        x0: &[S],
    ) -> Result<GlobalSearchResult<S>, SciOxideError> {
        let mut workspace = self.workspace(problem.dim())?;
        let mut output = GlobalSearchResult::empty();
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

enum ChainKernel<'a, S: RealScalar> {
    Basin {
        proposal: &'a dyn BasinProposal<S>,
        temperature: S,
    },
    Anneal {
        options: &'a DualAnnealingOptions<S>,
        visiting: &'a StudentT<S>,
    },
}

/// Validate, prepare and run one chain attempt.
#[allow(clippy::too_many_arguments)]
fn run<'scope, 'capability, S, F>(
    options: &GlobalSearchOptions<S>,
    configuration: &impl core::fmt::Debug,
    algorithm: scioxide_core::AlgorithmId,
    requirements: Result<GlobalSearchWorkspaceRequirements<'_, F>, SciOxideError>,
    kernel: ChainKernel<'_, S>,
    problem: &BoundedObjective<'_, S>,
    x0: &[S],
    output: &mut GlobalSearchResult<S>,
    workspace: &mut GlobalSearchWorkspace<S, F::Solver>,
    scope: SolveScope<'scope, 'capability>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: LocalSolverFactory<S>,
{
    workspace.shell.reject(AttemptKind::Call);
    let requirements = requirements?;
    let entry = crate::attempt::Entry::Call {
        identity: identity(algorithm, problem.dim(), configuration, requirements.token),
        budgets: SolveBudgets::unbounded(),
    };
    bind(problem, &requirements, workspace)?;
    validate_point(
        x0,
        &workspace.storage.bounds,
        scioxide_core::argument!("x0"),
    )?;
    crate::attempt::preflight(&scope, options.time_limit())?;
    super::attempt(
        &mut workspace.shell,
        &mut workspace.storage,
        &mut workspace.staging,
        output,
        scope,
        entry,
        |context, storage, staged| {
            run_chain(options, kernel, problem, x0, storage, staged, context)
        },
    )
}

/// Probability calculation uses log space; very large uphill costs reject without exponent overflow.
fn metropolis<S: RealScalar>(old: S, new: S, temperature: S, u: S) -> bool {
    if new <= old {
        return true;
    }
    if temperature == S::ZERO {
        return false;
    }
    let difference = new - old;
    let delta = if difference.is_finite() {
        difference / temperature
    } else {
        (new / temperature) - (old / temperature)
    };
    if !delta.is_finite() {
        return false;
    }
    u == S::ZERO || u.ln() <= -delta
}
fn generalized_accept<S: RealScalar>(old: S, new: S, temperature: S, q: S, u: S) -> bool {
    if new <= old {
        return true;
    }
    let difference = new - old;
    let delta = if difference.is_finite() {
        difference / temperature
    } else {
        (new / temperature) - (old / temperature)
    };
    let cost = (S::ONE - q) * delta;
    if !cost.is_finite() || cost >= S::ONE {
        return false;
    }
    u == S::ZERO || u.ln() <= (-cost).ln_1p() / (S::ONE - q)
}
fn temperature<S: RealScalar>(
    options: &DualAnnealingOptions<S>,
    sweep: usize,
) -> Result<S, SciOxideError> {
    let power = options.visit - S::ONE;
    let time = S::try_from_usize_exact(sweep.checked_add(2).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("annealing_sweep"))
    })?)?;
    let two = S::TWO;
    Ok(options.temperature * (power * two.ln()).exp_m1() / (power * time.ln()).exp_m1())
}
fn visit<S: RealScalar>(
    law: &StudentT<S>,
    q: S,
    t: S,
    random: &mut SearchRandom<'_>,
) -> Result<S, SciOxideError> {
    let three = S::TWO + S::ONE;
    let scale = (t.ln() / (three - q) - (three - q).ln() / (S::TWO)).exp();
    periodic_product(law.sample(random)?, scale)
}
// Reduce both parts of an error-free product before adding to a unit coordinate.
// Rounding the large product first would erase the fractional phase in f32.
fn periodic_product<S: RealScalar>(sample: S, scale: S) -> Result<S, SciOxideError> {
    let value = sample * scale;
    if !value.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "visiting displacement exceeds scalar range",
        ));
    }
    let residual = sample.mul_add(scale, -value);
    Ok((value % S::ONE + residual % S::ONE) % S::ONE)
}

fn record_raw<S: RealScalar + NumericScalar<Real = S>>(
    report: &mut crate::SearchStartReport<S>,
    value: S,
    raw_evaluations: usize,
    session: &SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    crate::utils::commit_optimize_result(
        &mut report.candidate,
        &report.start,
        value,
        None,
        0,
        OptimizeTermination::SearchBudgetCompleted,
        "annealing observation",
    )?;
    report.candidate.counts = scioxide_core::SolveCounts::from_own(
        scioxide_core::EvaluationCounts::new()
            .with_provider_count(EvaluationRole::Objective, raw_evaluations),
    );
    report.candidate.fun_evals = report
        .candidate
        .counts
        .total()?
        .provider(EvaluationRole::Objective);
    report.candidate.grad_evals = report
        .candidate
        .counts
        .total()?
        .provider(EvaluationRole::Gradient);
    report.candidate.algorithm = Some(session.algorithm()?);
    report.candidate.trust = PartialResultTrust::Admissible;
    report.has_candidate = true;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_chain<S, L>(
    options: &GlobalSearchOptions<S>,
    kernel: ChainKernel<'_, S>,
    problem: &BoundedObjective<'_, S>,
    x0: &[S],
    storage: &mut SearchStorage<S, L>,
    staged: &mut GlobalSearchResult<S>,
    context: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: crate::SearchLocalSolver<S>,
{
    if let Some(limit) = options.time_limit() {
        context.limit_time(limit)?;
    }
    staged.seed(&storage.seeds);
    let bounds = &storage.bounds;
    let locals = &mut storage.pool.locals;
    let token = storage.token;
    let session = SolveSession::new(context, scioxide_core::argument!("scope"));
    let mut current: Option<usize> = None;
    let mut sweep = 0usize;
    let n = bounds.len();
    let cycle =
        scioxide_core::backend::checked_mul(n, 2, scioxide_core::quantity!("annealing_cycle"))?;
    let outcome = (|| {
        for (index, local) in locals.iter_mut().enumerate().take(options.starts().get()) {
            if let Some(stop) = session.trial_iteration()? {
                session.set_stopped(stop);
                break;
            }
            let best_value = staged.best().map(|r| r.fun());
            let (previous, next) = staged.reports.split_at_mut(index);
            let report = &mut next[0];
            let old = current.map(|i| previous[i].candidate.x()).unwrap_or(x0);
            let old_value = current.map(|i| previous[i].candidate.fun());
            let mut generator = Xoshiro256StarStar::from_seed(report.seed);
            let mut admission = &session;
            let mut random = SearchRandom::new(&mut generator, &mut admission);
            let mut reannealed = false;
            let t;
            match &kernel {
                ChainKernel::Basin {
                    proposal,
                    temperature,
                } => {
                    t = *temperature;
                    if index == 0 {
                        report.start.copy_from_slice(x0);
                    } else {
                        proposal.propose(bounds, old, &mut random, &mut report.start)?;
                    }
                }
                ChainKernel::Anneal {
                    options: anneal,
                    visiting,
                } => {
                    if index > 0 && (index - 1).is_multiple_of(cycle) {
                        sweep = sweep.checked_add(1).ok_or_else(|| {
                            SciOxideError::size_overflow(scioxide_core::quantity!(
                                "annealing_sweep"
                            ))
                        })?;
                    }
                    let candidate_temperature = temperature(anneal, sweep)?;
                    if candidate_temperature <= anneal.temperature * anneal.restart
                        || !candidate_temperature.is_finite()
                        || candidate_temperature <= S::ZERO
                    {
                        sweep = 0;
                        t = anneal.temperature;
                        reannealed = true;
                    } else {
                        t = candidate_temperature;
                    }
                    if index == 0 {
                        report.start.copy_from_slice(x0);
                    } else if reannealed {
                        UniformStartDesign.point(index, bounds, &mut random, &mut report.start)?;
                    } else {
                        report.start.copy_from_slice(old);
                        let phase = (index - 1) % cycle;
                        for (axis, (x, &(lo, hi))) in
                            report.start.iter_mut().zip(bounds.as_slice()).enumerate()
                        {
                            if phase < n || axis == phase - n {
                                let unit = normalized_coordinate(*x, lo, hi)?;
                                let delta = visit(visiting, anneal.visit, t, &mut random)? % S::ONE;
                                let next = (unit + delta) % S::ONE;
                                let next = if next < S::ZERO { next + S::ONE } else { next };
                                *x = scioxide_core::checked_affine(lo, hi, next)?;
                            }
                        }
                    }
                }
            }
            report.temperature = Some(t);
            report.reannealed = reannealed;
            validate_point(&report.start, bounds, scioxide_core::argument!("proposal"))?;
            staged.attempted = index + 1;
            if let ChainKernel::Anneal {
                options: anneal, ..
            } = &kernel
            {
                let before = session.counts()?.provider(EvaluationRole::Objective);
                let value =
                    session.invoke(EvaluationRole::Objective, || problem.eval(&report.start))?;
                let raw_evaluations =
                    session.counts()?.provider(EvaluationRole::Objective) - before;
                crate::utils::validate_finite(&[value])?;
                record_raw(report, value, raw_evaluations, &session)?;
                if anneal.local
                    && (index == 0 || reannealed || best_value.is_none_or(|best| value < best))
                {
                    // Stage refinement separately so an unsuccessful/poorer local result cannot erase a raw best.
                    run_local(
                        local,
                        report,
                        index,
                        problem,
                        bounds,
                        options.local_budgets(),
                        token,
                        &session,
                    )?;
                    if report.candidate.fun() > value || report.error.is_some() {
                        if report.local_done {
                            core::mem::swap(&mut report.candidate, &mut report.refinement);
                            report.local_in_refinement = true;
                        }
                        record_raw(report, value, raw_evaluations, &session)?;
                    }
                }
            } else {
                run_local(
                    local,
                    report,
                    index,
                    problem,
                    bounds,
                    options.local_budgets(),
                    token,
                    &session,
                )?;
            }
            if report.has_candidate {
                let accept = if reannealed || old_value.is_none() {
                    true
                } else {
                    let old = old_value.ok_or_else(|| {
                        invalid(scioxide_core::argument!("self"), "missing chain value")
                    })?;
                    let u = random.uniform()?;
                    match &kernel {
                        ChainKernel::Basin { .. } => metropolis(old, report.candidate.fun(), t, u),
                        ChainKernel::Anneal {
                            options: anneal, ..
                        } => generalized_accept(
                            old,
                            report.candidate.fun(),
                            t / S::try_from_usize_exact(sweep + 1)?,
                            anneal.accept,
                            u,
                        ),
                    }
                };
                if accept {
                    report.accepted = true;
                    current = Some(index);
                    session.accept(PartialResultTrust::Admissible)?;
                } else {
                    session.reject()?;
                }
            }
            staged.record(index, options);
            if session.termination()?.is_some() {
                break;
            }
        }
        Ok(())
    })();
    if outcome.is_ok() || session.stopped().is_some() {
        for index in 0..staged.attempted {
            staged.record(index, options);
        }
    }
    finish(outcome, staged, &session)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn acceptance_boundaries() {
        assert!(metropolis(1.0, 0.0, 0.0, 0.99));
        assert!(!metropolis(0.0, 1.0, 0.0, 0.0));
        assert!(metropolis(0.0, 1.0, 1.0, 0.3));
        assert!(!metropolis(0.0, 1.0, 1.0, 0.4));
        assert!(!generalized_accept(0.0, 1.0, 1.0, -5.0, 0.0));
        assert!(generalized_accept(0.0, 0.01, 1.0, -5.0, 0.9));
    }
    #[test]
    fn large_visits_retain_fractional_phase() {
        let sample = 0.1_f32;
        let scale = 1e10_f32;
        let oracle = ((sample as f64) * (scale as f64)) % 1.0;
        assert_eq!((sample * scale) % 1.0, 0.0);
        assert!((periodic_product(sample, scale).unwrap() as f64 - oracle).abs() < 1e-6);
        let epsilon = f64::EPSILON;
        let phase = periodic_product(1.0 + epsilon, 4503599627370497.0_f64).unwrap();
        assert_eq!(phase, epsilon);
    }
    #[test]
    fn cooling_matches_definition() {
        let o = DualAnnealingOptions::new(GlobalSearchOptions::<f64>::default());
        assert!((temperature(&o, 0).unwrap() - 5230.0).abs() < 1e-10);
        assert!(temperature(&o, 1).unwrap() < temperature(&o, 0).unwrap());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod default_tests {
    use super::*;
    fn check<S: RealScalar>() {
        let options = DualAnnealingOptions::<S>::default();
        for (actual, expected) in [
            (options.temperature, 5230.0),
            (options.visit, 2.62),
            (options.accept, -5.0),
            (options.restart, 2e-5),
        ] {
            assert_eq!(actual, S::try_from_f64_lossy(expected).unwrap());
        }
    }
    #[test]
    fn schedule_constants_are_correctly_rounded() {
        check::<f32>();
        check::<f64>();
    }
}
