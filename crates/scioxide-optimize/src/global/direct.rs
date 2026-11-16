//! DIRECT (dividing rectangles) search over a finite box.

use alloc::vec::Vec;

use scioxide_core::Scaled;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{
    AttemptKind, AttemptRecord, Bounds, Budget, BudgetKind, Capacity, EvaluationRole,
    ExecutionPolicy, InvalidInputReason, NonFiniteOrigin, NumericScalar, PartialResultTrust,
    RealScalar, SciOxideError, SolveBudgets, SolveContext, SolveScope,
};

use super::sequential_only;
use crate::attempt::{Accepted, AcceptedState, Entry, Shell, Staged};
use crate::context::SolveSession;
use crate::{
    BoundedObjective, MinimizerAccepted, MinimizerProgress, ObjectiveFunction, OptimizeTermination,
};

/// Selects the rule by which [`Direct`] chooses the potentially optimal
/// rectangles to divide. Both variants sample and divide rectangles the same
/// way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectVariant {
    /// DIRECT of Jones, Perttunen and Stuckman (1993): rectangles are measured
    /// by the distance from center to vertex, and the improvement threshold
    /// `ε` applies.
    Original,
    /// DIRECT-L of Gablonsky and Kelley (2001): rectangles are grouped by their
    /// longest side, at most one rectangle per size is selected, and `ε = 0`.
    /// This biases the search toward local refinement.
    LocallyBiased,
}

/// Configures [`Direct`]: the variant and the limits on evaluations,
/// rectangles, storage and sweeps.
///
/// The setters do not check their arguments. [`Direct::new`] checks the
/// options, and [`Direct::workspace_requirements`] checks the storage
/// limit for a given dimension. See [Termination](crate#termination) for how
/// the search ends.
#[derive(Debug, Clone)]
pub struct DirectOptions<S: RealScalar> {
    variant: DirectVariant,
    max_evaluations: Budget,
    max_rectangles: Budget,
    max_storage_bytes: Budget,
    max_sweeps: Budget,
    epsilon: S,
    execution: ExecutionPolicy,
}

impl<S: RealScalar> DirectOptions<S> {
    /// Creates options with limits on objective evaluations, on the number of
    /// rectangles kept, and on the bytes of storage the search may allocate.
    ///
    /// The storage limit covers every buffer of the workspace, including the
    /// result. The other options default to [`DirectVariant::Original`], a
    /// sweep limit equal to `max_evaluations`, `ε = 10⁻⁴` and sequential
    /// execution.
    pub fn new(max_evaluations: Budget, max_rectangles: Budget, max_storage_bytes: Budget) -> Self {
        let ten = S::TEN;
        Self {
            variant: DirectVariant::Original,
            max_evaluations,
            max_rectangles,
            max_storage_bytes,
            max_sweeps: max_evaluations,
            epsilon: S::ONE / (ten * ten * ten * ten),
            execution: ExecutionPolicy::Sequential,
        }
    }

    /// Sets the selection rule.
    pub const fn with_variant(mut self, variant: DirectVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Sets the maximum number of sweeps. A sweep selects the potentially
    /// optimal rectangles and divides each of them.
    pub const fn with_max_sweeps(mut self, max_sweeps: Budget) -> Self {
        self.max_sweeps = max_sweeps;
        self
    }

    /// Sets the improvement threshold `ε ≥ 0` of [`DirectVariant::Original`].
    ///
    /// A rectangle is potentially optimal only if it could improve the best
    /// value by at least `ε |f_min|`. [`DirectVariant::LocallyBiased`] always
    /// uses `ε = 0`.
    pub const fn with_epsilon(mut self, epsilon: S) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Sets the execution policy. Each sweep depends on all earlier samples, so
    /// [`Direct::new`] accepts only [`ExecutionPolicy::Sequential`].
    pub const fn with_execution(mut self, execution: ExecutionPolicy) -> Self {
        self.execution = execution;
        self
    }

    /// Returns the selection rule.
    pub const fn variant(&self) -> DirectVariant {
        self.variant
    }
    /// Returns the maximum number of objective evaluations.
    pub const fn max_evaluations(&self) -> Budget {
        self.max_evaluations
    }
    /// Returns the maximum number of rectangles kept at once.
    pub const fn max_rectangles(&self) -> Budget {
        self.max_rectangles
    }
    /// Returns the maximum number of bytes the workspace may allocate.
    pub const fn max_storage_bytes(&self) -> Budget {
        self.max_storage_bytes
    }
    /// Returns the maximum number of sweeps.
    pub const fn max_sweeps(&self) -> Budget {
        self.max_sweeps
    }
    /// Returns the improvement threshold `ε` of [`DirectVariant::Original`].
    pub const fn epsilon(&self) -> S {
        self.epsilon
    }
    /// Returns the execution policy; sequential for every valid method.
    pub const fn execution(&self) -> ExecutionPolicy {
        self.execution
    }

    /// Prepared payload bytes for dimension `d`.
    fn storage_bytes(&self, d: usize) -> Result<usize, SciOxideError> {
        let n = self.max_rectangles.get();
        let pair = checked_mul(d, 2, scioxide_core::quantity!("direct_pair_count"))?;
        let coordinates = checked_add(
            checked_mul(
                n,
                d,
                scioxide_core::quantity!("direct_rectangle_coordinates"),
            )?,
            checked_mul(
                pair,
                d,
                scioxide_core::quantity!("direct_trial_coordinates"),
            )?,
            scioxide_core::quantity!("direct_coordinates"),
        )?;
        let scalar_count = checked_add(
            checked_mul(
                coordinates,
                3,
                scioxide_core::quantity!("direct_scalar_buffers"),
            )?,
            checked_add(
                checked_mul(n, 2, scioxide_core::quantity!("direct_rectangle_scalars"))?,
                checked_add(pair, d, scioxide_core::quantity!("direct_trial_scalars"))?,
                scioxide_core::quantity!("direct_scalar_count"),
            )?,
            scioxide_core::quantity!("direct_scalar_count"),
        )?;
        let scalar_bytes = checked_mul(
            scalar_count,
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("direct_scalar_bytes"),
        )?;
        let index_count = checked_add(
            checked_mul(n, 2, scioxide_core::quantity!("direct_index_buffers"))?,
            d,
            scioxide_core::quantity!("direct_index_count"),
        )?;
        let index_bytes = checked_mul(
            index_count,
            core::mem::size_of::<usize>(),
            scioxide_core::quantity!("direct_index_bytes"),
        )?;
        checked_add(
            checked_add(
                scalar_bytes,
                index_bytes,
                scioxide_core::quantity!("direct_storage"),
            )?,
            checked_add(n, pair, scioxide_core::quantity!("direct_flags"))?,
            scioxide_core::quantity!("direct_storage"),
        )
    }
}

/// Result of a [`Direct`] search: the best sample, the search statistics
/// and the termination.
///
/// Start with [`Self::empty`]; the search sizes it on first use.
#[derive(Debug, Clone)]
pub struct DirectResult<S: RealScalar> {
    best_x: Vec<S>,
    best_value: Option<S>,
    evaluations: usize,
    sweeps: usize,
    rectangles: usize,
    stalled_rectangles: usize,
    max_live_half_side: S,
    termination: OptimizeTermination,
    counts: scioxide_core::SolveCounts,
}

impl<S: RealScalar> Default for DirectResult<S> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<S: RealScalar> DirectResult<S> {
    /// Returns an empty result without allocating.
    pub const fn empty() -> Self {
        Self {
            best_x: Vec::new(),
            best_value: None,
            evaluations: 0,
            sweeps: 0,
            rectangles: 0,
            stalled_rectangles: 0,
            max_live_half_side: S::ZERO,
            termination: OptimizeTermination::NotStarted,
            counts: scioxide_core::SolveCounts::new(),
        }
    }
    /// Returns the best sampled point and its value; `None` only before the
    /// first evaluation.
    pub fn best_sample(&self) -> Option<(&[S], S)> {
        self.best_value.map(|value| (self.best_x.as_slice(), value))
    }
    /// Returns the number of objective evaluations. A point sampled twice is
    /// evaluated once.
    pub const fn evaluations(&self) -> usize {
        self.evaluations
    }
    /// Returns the number of completed sweeps.
    pub const fn sweeps(&self) -> usize {
        self.sweeps
    }
    /// Returns the number of rectangles, including those that cannot be
    /// divided further.
    pub const fn rectangles(&self) -> usize {
        self.rectangles
    }
    /// Returns the number of rectangles that cannot be divided further
    /// because their new sample points round to existing points.
    pub const fn stalled_rectangles(&self) -> usize {
        self.stalled_rectangles
    }
    /// Returns the largest half side of the rectangles, in box units.
    ///
    /// It measures how finely the search has covered the box. It is not a
    /// bound on the error in the objective value.
    pub const fn max_live_half_side(&self) -> S {
        self.max_live_half_side
    }
    /// Returns why the search stopped. No termination is a success; the search
    /// does not prove that it found the global minimum.
    pub const fn termination(&self) -> OptimizeTermination {
        self.termination
    }
    /// Returns the evaluations and iterations, including failed objective calls.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }

    fn try_reserve(&mut self, dimension: usize) -> Result<(), SciOxideError> {
        if self.best_x.len() != dimension {
            self.best_x = try_filled(
                dimension,
                S::ZERO,
                scioxide_core::quantity!("direct_result_point"),
            )?;
        }
        Ok(())
    }
}

impl<S: RealScalar> Staged for DirectResult<S> {
    fn reset(&mut self) {
        self.best_value = None;
        self.evaluations = 0;
        self.sweeps = 0;
        self.rectangles = 0;
        self.stalled_rectangles = 0;
        self.max_live_half_side = S::ZERO;
        self.termination = OptimizeTermination::NotStarted;
        self.counts = scioxide_core::SolveCounts::new();
    }
    fn publish(&mut self, source: &Self) -> Result<(), SciOxideError> {
        if self.best_x.len() != source.best_x.len() {
            let mut point = try_filled(
                source.best_x.len(),
                S::ZERO,
                scioxide_core::quantity!("direct_result_point"),
            )?;
            point.copy_from_slice(&source.best_x);
            self.best_x = point;
        } else {
            self.best_x.copy_from_slice(&source.best_x);
        }
        self.best_value = source.best_value;
        self.evaluations = source.evaluations;
        self.sweeps = source.sweeps;
        self.rectangles = source.rectangles;
        self.stalled_rectangles = source.stalled_rectangles;
        self.max_live_half_side = source.max_live_half_side;
        self.termination = source.termination;
        self.counts = source.counts;
        Ok(())
    }
    /// Sampled estimates never report success.
    fn succeeded(&self) -> bool {
        false
    }
}

impl<S: RealScalar> Accepted<S> for DirectResult<S> {
    fn accepted(&self) -> Option<AcceptedState<'_, S>> {
        Some(AcceptedState {
            point: &self.best_x,
            value: self.best_value?,
            gradient: None,
            iterations: self.sweeps,
        })
    }
}

/// Workspace size of a [`Direct`] search, within its storage limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectWorkspaceRequirements {
    dimension: usize,
    rectangles: usize,
    bytes: usize,
}

impl DirectWorkspaceRequirements {
    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }
    /// Returns the maximum number of rectangles.
    pub const fn rectangles(self) -> usize {
        self.rectangles
    }
    /// Returns the number of bytes the workspace allocates.
    pub const fn bytes(self) -> usize {
        self.bytes
    }
}

/// Bounded rectangle queue and all solver-owned scratch.
#[derive(Debug)]
pub(crate) struct DirectStorage<S: RealScalar> {
    dimension: usize,
    capacity: usize,
    centers: Vec<S>,
    halves: Vec<S>,
    physical: Vec<S>,
    values: Vec<S>,
    sizes: Vec<S>,
    stalled: Vec<bool>,
    selected: Vec<usize>,
    groups: Vec<usize>,
    axes: Vec<usize>,
    pair_index: Vec<usize>,
    trials: Vec<S>,
    trial_values: Vec<S>,
    trial_fresh: Vec<bool>,
}

impl<S: RealScalar> DirectStorage<S> {
    const fn empty() -> Self {
        Self {
            dimension: 0,
            capacity: 0,
            centers: Vec::new(),
            halves: Vec::new(),
            physical: Vec::new(),
            values: Vec::new(),
            sizes: Vec::new(),
            stalled: Vec::new(),
            selected: Vec::new(),
            groups: Vec::new(),
            axes: Vec::new(),
            pair_index: Vec::new(),
            trials: Vec::new(),
            trial_values: Vec::new(),
            trial_fresh: Vec::new(),
        }
    }
    fn fits(&self, requirements: DirectWorkspaceRequirements) -> bool {
        self.dimension == requirements.dimension && self.capacity == requirements.rectangles
    }
    fn prepare(requirements: DirectWorkspaceRequirements) -> Result<Self, SciOxideError> {
        let n = requirements.rectangles;
        let d = requirements.dimension;
        let nd = checked_mul(
            n,
            d,
            scioxide_core::quantity!("direct_rectangle_coordinates"),
        )?;
        let pair = checked_mul(d, 2, scioxide_core::quantity!("direct_trial_count"))?;
        Ok(Self {
            dimension: d,
            capacity: n,
            centers: try_filled(nd, S::ZERO, scioxide_core::quantity!("direct_centers"))?,
            halves: try_filled(nd, S::ZERO, scioxide_core::quantity!("direct_half_sides"))?,
            physical: try_filled(
                nd,
                S::ZERO,
                scioxide_core::quantity!("direct_physical_centers"),
            )?,
            values: try_filled(n, S::ZERO, scioxide_core::quantity!("direct_values"))?,
            sizes: try_filled(n, S::ZERO, scioxide_core::quantity!("direct_sizes"))?,
            stalled: try_filled(n, false, scioxide_core::quantity!("direct_stalled_flags"))?,
            selected: try_with_capacity(n, scioxide_core::quantity!("direct_selected_rectangles"))?,
            groups: try_with_capacity(n, scioxide_core::quantity!("direct_size_groups"))?,
            axes: try_with_capacity(d, scioxide_core::quantity!("direct_split_axes"))?,
            pair_index: try_filled(
                d,
                usize::MAX,
                scioxide_core::quantity!("direct_pair_indices"),
            )?,
            trials: try_filled(
                checked_mul(pair, d, scioxide_core::quantity!("direct_trials"))?,
                S::ZERO,
                scioxide_core::quantity!("direct_trials"),
            )?,
            trial_values: try_filled(
                pair,
                S::ZERO,
                scioxide_core::quantity!("direct_trial_values"),
            )?,
            trial_fresh: try_filled(pair, false, scioxide_core::quantity!("direct_trial_flags"))?,
        })
    }
}

/// Workspace of [`Direct`]: the rectangles, scratch storage and the record
/// of the last call.
#[derive(Debug)]
pub struct DirectWorkspace<S: RealScalar> {
    storage: DirectStorage<S>,
    staging: DirectResult<S>,
    shell: Shell<MinimizerProgress<S>>,
}

impl<S: RealScalar> Default for DirectWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar> DirectWorkspace<S> {
    /// Creates an empty workspace without allocating.
    pub fn new() -> Self {
        Self {
            storage: DirectStorage::empty(),
            staging: DirectResult::empty(),
            shell: Shell::new(),
        }
    }
    /// Checks whether the workspace already fits `requirements`.
    ///
    /// # Errors
    ///
    /// Never returns an error; the storage limit is checked by
    /// [`Direct::workspace_requirements`].
    pub fn check(
        &self,
        requirements: &DirectWorkspaceRequirements,
    ) -> Result<Capacity, SciOxideError> {
        Ok(
            if self.storage.fits(*requirements)
                && self.staging.best_x.len() == requirements.dimension
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
    /// Allocates storage for `requirements`. On failure the existing storage
    /// is unchanged.
    ///
    /// # Errors
    ///
    /// `SizeOverflow` or `AllocationFailure`.
    pub fn try_reserve(
        &mut self,
        requirements: &DirectWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        if self.check(requirements)? == Capacity::Sufficient {
            return Ok(());
        }
        if !self.storage.fits(*requirements) {
            self.storage = DirectStorage::prepare(*requirements)?;
        }
        self.staging.try_reserve(requirements.dimension)
    }
    /// Returns the record of the most recent call; its progress holds the best
    /// sample.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<MinimizerProgress<S>>> {
        self.shell.last_attempt()
    }
    /// Returns the best sample of the most recent call, if it has one.
    pub fn last_accepted(&self) -> Option<MinimizerAccepted<'_, S>>
    where
        S: NumericScalar<Real = S>,
    {
        crate::attempt::last_accepted(&self.shell, &self.staging)
    }
}

/// Minimizes an objective over a finite box with the DIRECT (dividing
/// rectangles) algorithm.
///
/// DIRECT is deterministic and uses objective values only. It scales the box
/// to the unit cube and samples its center. Each sweep selects the potentially
/// optimal rectangles, those with the lowest center value for some Lipschitz
/// constant, according to the [`DirectVariant`]. It divides each selected
/// rectangle into thirds along its longest sides, sampling the two new centers
/// `c ± δ/3` on each such side, where `δ` is the side length. A variable with
/// equal lower and upper bounds is fixed. Storage is bounded by the limits of
/// [`DirectOptions`]: O(r n) for `r` rectangles and `n` variables.
///
/// The search stops at the evaluation or sweep limit, when the rectangle limit
/// would be exceeded, or when no rectangle can be divided. No termination is
/// a success, and the search does not prove that it found the global minimum.
/// DIRECT is sequential.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Bounds, Budget, ExecutionContext, SciOxideError};
/// use scioxide_optimize::{BoundedObjective, DirectOptions, Direct, DirectResult, ObjectiveFn};
/// # fn main() -> Result<(), SciOxideError> {
/// let objective = ObjectiveFn::new(1, |x: &[f64]| Ok((x[0] - 0.3).powi(2)));
/// let bounds = Bounds::new(vec![(Some(0.0), Some(1.0))])?;
/// let problem = BoundedObjective::new(&objective, &bounds)?;
/// let method = Direct::new(DirectOptions::new(
///     Budget::try_from(50)?,
///     Budget::try_from(101)?,
///     Budget::try_from(1 << 20)?,
/// ))?;
/// let mut workspace = method.workspace(1)?;
/// let mut result = DirectResult::empty();
/// method.minimize_into(&problem, &mut result, &mut workspace, ExecutionContext::sequential())?;
/// // The minimum of (x − 0.3)² on [0, 1] is 0 at x = 0.3.
/// assert!(result.best_sample().unwrap().1 < 1e-3);
/// # Ok(())
/// # }
/// ```
///
/// # References
///
/// - D. R. Jones, C. D. Perttunen and B. E. Stuckman, "Lipschitzian
///   optimization without the Lipschitz constant", *J. Optim. Theory Appl.*
///   79(1), 157–181, 1993.
/// - J. M. Gablonsky and C. T. Kelley, "A locally-biased form of the DIRECT
///   algorithm", *J. Global Optim.* 21(1), 27–37, 2001.
#[derive(Debug, Clone)]
pub struct Direct<S: RealScalar> {
    options: DirectOptions<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Direct<S> {
    /// Creates a method after checking `options`.
    ///
    /// # Errors
    ///
    /// `InvalidInput` if `ε` is NaN, infinite or negative, or the execution
    /// policy is not sequential.
    pub fn new(options: DirectOptions<S>) -> Result<Self, SciOxideError> {
        if !options.epsilon.is_finite() || options.epsilon < S::ZERO {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("options"),
                InvalidInputReason::OutOfRange,
                "epsilon must be finite and nonnegative",
            ));
        }
        sequential_only(options.execution, "DIRECT sweeps are sequential")?;
        Ok(Self { options })
    }
    /// Returns the options.
    pub const fn options(&self) -> &DirectOptions<S> {
        &self.options
    }
    /// Returns the workspace size for `dimension` variables.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace would exceed the storage limit.
    /// - `SizeOverflow` if the size overflows.
    pub fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<DirectWorkspaceRequirements, SciOxideError> {
        let bytes = self.options.storage_bytes(dimension)?;
        if bytes > self.options.max_storage_bytes.get() {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("direct_storage_bytes"),
                bytes,
                self.options.max_storage_bytes.get(),
            ));
        }
        Ok(DirectWorkspaceRequirements {
            dimension,
            rectangles: self.options.max_rectangles.get(),
            bytes,
        })
    }
    /// Allocates a workspace for `dimension` variables.
    ///
    /// # Errors
    ///
    /// As [`Self::workspace_requirements`], plus `AllocationFailure`.
    pub fn workspace(&self, dimension: usize) -> Result<DirectWorkspace<S>, SciOxideError> {
        let mut workspace = DirectWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }
    /// Searches the problem's box and writes the best sample and statistics to
    /// `output`.
    ///
    /// The sweep and evaluation limits of the options apply in addition to
    /// those of `scope`. When a limit, cancellation or a deadline stops the
    /// search after the first evaluation, `output` holds the best sample found
    /// so far.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a variable lacks a lower or upper bound.
    /// - `ResourceLimit` if the workspace would exceed the storage limit.
    /// - `NonFinite` if the objective returns NaN or infinity.
    /// - `ConvergenceFailure` or `ExecutionStopped` if the search stops before
    ///   the first evaluation, and errors of the objective.
    pub fn minimize_into<'scope, 'capability>(
        &self,
        problem: &BoundedObjective<'_, S>,
        output: &mut DirectResult<S>,
        workspace: &mut DirectWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        crate::geometry::finite_box(problem.bounds, false)?;
        workspace.try_reserve(&self.workspace_requirements(problem.dim())?)?;
        crate::attempt::preflight(&scope, None)?;
        let entry = Entry::Call {
            identity: crate::context::local_identity(
                scioxide_core::algorithm_id!("optimize", "DIRECT", 1),
                problem.dim(),
                &self.options,
            ),
            budgets: SolveBudgets::unbounded()
                .with_limit(BudgetKind::Iteration, self.options.max_sweeps)
                .with_limit(BudgetKind::Evaluation, self.options.max_evaluations),
        };
        super::attempt(
            &mut workspace.shell,
            &mut workspace.storage,
            &mut workspace.staging,
            output,
            scope,
            entry,
            |context, storage, staged| self.kernel(problem, storage, staged, context),
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
    ) -> Result<DirectResult<S>, SciOxideError> {
        let mut workspace = self.workspace(problem.dim())?;
        let mut output = DirectResult::empty();
        self.minimize_into(
            problem,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    fn kernel(
        &self,
        problem: &BoundedObjective<'_, S>,
        work: &mut DirectStorage<S>,
        staged: &mut DirectResult<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let session = SolveSession::new(context, scioxide_core::argument!("scope"));
        let outcome = self.run(problem, work, staged, &session);
        let d = problem.dim();
        let termination = match outcome {
            Ok(stop) => stop,
            Err(error) => match session.stopped() {
                Some(stop) if staged.best_value.is_some() => stop,
                _ => return Err(error),
            },
        };
        staged.termination = termination;
        staged.counts = session.accounting()?;
        staged.max_live_half_side = work.halves[..staged.rectangles * d]
            .iter()
            .copied()
            .fold(S::ZERO, |a, b| a.max(b));
        Ok(())
    }

    fn run(
        &self,
        problem: &BoundedObjective<'_, S>,
        work: &mut DirectStorage<S>,
        staged: &mut DirectResult<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<OptimizeTermination, SciOxideError> {
        let bounds = problem.bounds;
        let d = bounds.len();
        session.proposal()?;
        let root = &mut work.physical[..d];
        for (axis, &(lower, upper)) in bounds.as_slice().iter().enumerate() {
            let (lower, upper) = endpoints(lower, upper)?;
            work.centers[axis] = S::HALF;
            work.halves[axis] = if lower == upper { S::ZERO } else { S::HALF };
            root[axis] = scioxide_core::checked_affine(lower, upper, S::HALF)?;
        }
        let value = session.invoke(EvaluationRole::Objective, || problem.eval(root))?;
        finite_value(value)?;
        work.values[0] = value;
        work.sizes[0] = rectangle_size(&work.halves[..d], self.options.variant)?;
        work.stalled[0] = false;
        staged.rectangles = 1;
        staged.evaluations = 1;
        staged.best_x.copy_from_slice(root);
        staged.best_value = Some(value);
        session.accept(PartialResultTrust::Admissible)?;
        if work.halves[..d].iter().all(|&h| h == S::ZERO) {
            return Ok(OptimizeTermination::SearchBudgetCompleted);
        }
        loop {
            if staged.evaluations >= self.options.max_evaluations.get() {
                return Ok(OptimizeTermination::EvaluationLimit);
            }
            if staged.sweeps >= self.options.max_sweeps.get() {
                return Ok(OptimizeTermination::SearchBudgetCompleted);
            }
            if let Some(stop) = session.iteration()? {
                return Ok(stop);
            }
            select_rectangles(work, staged, &self.options)?;
            if work.selected.is_empty() {
                return Ok(OptimizeTermination::Stagnation);
            }
            let selected_count = work.selected.len();
            let mut divided = false;
            for slot in 0..selected_count {
                let index = work.selected[slot];
                let stop = self.divide(problem, bounds, work, staged, session, index)?;
                if let Some(stop) = stop {
                    return Ok(stop);
                }
                divided = true;
            }
            staged.sweeps += 1;
            if !divided {
                return Ok(OptimizeTermination::Stagnation);
            }
        }
    }

    fn divide(
        &self,
        problem: &BoundedObjective<'_, S>,
        bounds: &Bounds<S>,
        work: &mut DirectStorage<S>,
        staged: &mut DirectResult<S>,
        session: &SolveSession<'_, '_, '_>,
        index: usize,
    ) -> Result<Option<OptimizeTermination>, SciOxideError> {
        let d = bounds.len();
        let base = index * d;
        let longest = work.halves[base..base + d]
            .iter()
            .copied()
            .fold(S::ZERO, |a, b| a.max(b));
        if longest == S::ZERO {
            work.stalled[index] = true;
            staged.stalled_rectangles += 1;
            return Ok(None);
        }
        work.axes.clear();
        work.pair_index.fill(usize::MAX);
        // Every longest active direction gets a paired probe. Physical rounding
        // may make a direction unsplittable; exclude it before admitting work.
        for axis in 0..d {
            if work.halves[base + axis] != longest {
                continue;
            }
            let step = work.halves[base + axis] / S::try_from_f64_lossy(3.0)?;
            let shift = step + step;
            let center = work.centers[base + axis];
            let minus = center - shift;
            let plus = center + shift;
            if minus == center || plus == center || minus < S::ZERO || plus > S::ONE {
                continue;
            }
            let pos = work.axes.len() * 2;
            for (side, normalized) in [minus, plus].into_iter().enumerate() {
                let trial = &mut work.trials[(pos + side) * d..(pos + side + 1) * d];
                trial.copy_from_slice(&work.physical[base..base + d]);
                let (lower, upper) = bounds.as_slice()[axis];
                let (lower, upper) = endpoints(lower, upper)?;
                trial[axis] = scioxide_core::checked_affine(lower, upper, normalized)?;
            }
            if same_point(
                &work.trials[pos * d..(pos + 1) * d],
                &work.physical[base..base + d],
            ) || same_point(
                &work.trials[(pos + 1) * d..(pos + 2) * d],
                &work.physical[base..base + d],
            ) || same_point(
                &work.trials[pos * d..(pos + 1) * d],
                &work.trials[(pos + 1) * d..(pos + 2) * d],
            ) {
                continue;
            }
            work.pair_index[axis] = work.axes.len();
            work.axes.push(axis);
        }
        if work.axes.is_empty() {
            work.stalled[index] = true;
            staged.stalled_rectangles += 1;
            return Ok(None);
        }
        let needed_rectangles = checked_add(
            staged.rectangles,
            work.axes.len() * 2,
            scioxide_core::quantity!("direct_rectangles"),
        )?;
        if needed_rectangles > self.options.max_rectangles.get() {
            return Ok(Some(OptimizeTermination::MemoryLimit));
        }
        let trial_count = work.axes.len() * 2;
        let mut fresh_count = 0;
        for trial_index in 0..trial_count {
            let point = &work.trials[trial_index * d..(trial_index + 1) * d];
            let queued = (0..staged.rectangles)
                .any(|j| same_point(&work.physical[j * d..(j + 1) * d], point));
            let previous =
                (0..trial_index).any(|j| same_point(&work.trials[j * d..(j + 1) * d], point));
            work.trial_fresh[trial_index] = !queued && !previous;
            fresh_count += usize::from(work.trial_fresh[trial_index]);
        }
        if checked_add(
            staged.evaluations,
            fresh_count,
            scioxide_core::quantity!("direct_evaluations"),
        )? > self.options.max_evaluations.get()
        {
            return Ok(Some(OptimizeTermination::EvaluationLimit));
        }
        session.require_evaluations(fresh_count)?;
        session.node()?;
        for trial_index in 0..trial_count {
            let point = &work.trials[trial_index * d..(trial_index + 1) * d];
            if work.trial_fresh[trial_index] {
                session.proposal()?;
                let value = session.invoke(EvaluationRole::Objective, || problem.eval(point))?;
                finite_value(value)?;
                work.trial_values[trial_index] = value;
                staged.evaluations += 1;
                if staged.best_value.is_none_or(|best| value < best) {
                    staged.best_x.copy_from_slice(point);
                    staged.best_value = Some(value);
                    session.accept(PartialResultTrust::Admissible)?;
                }
            } else {
                let queued = (0..staged.rectangles)
                    .find(|&j| same_point(&work.physical[j * d..(j + 1) * d], point));
                work.trial_values[trial_index] = if let Some(j) = queued {
                    work.values[j]
                } else {
                    let j = (0..trial_index)
                        .find(|&j| same_point(&work.trials[j * d..(j + 1) * d], point))
                        .ok_or_else(|| {
                            invalid(scioxide_core::argument!("work"), "DIRECT duplicate lookup")
                        })?;
                    work.trial_values[j]
                };
                session.cache_hit()?;
            }
        }
        // Paired values determine the split order. Stable ties use axis order.
        work.axes.sort_by(|&left, &right| {
            let lp = work.pair_index[left] * 2;
            let rp = work.pair_index[right] * 2;
            let l = work.trial_values[lp].min(work.trial_values[lp + 1]);
            let r = work.trial_values[rp].min(work.trial_values[rp + 1]);
            l.partial_cmp(&r)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(left.cmp(&right))
        });
        // The trials retain original axis order while the split order changes.
        for &axis in &work.axes {
            let pair = work.pair_index[axis];
            let small = work.halves[base + axis] / S::try_from_f64_lossy(3.0)?;
            let shift = small + small;
            work.halves[base + axis] = small;
            for side in 0..2 {
                let child = staged.rectangles;
                let child_base = child * d;
                work.centers.copy_within(base..base + d, child_base);
                work.halves.copy_within(base..base + d, child_base);
                work.physical[child_base..child_base + d].copy_from_slice(
                    &work.trials[(pair * 2 + side) * d..(pair * 2 + side + 1) * d],
                );
                work.centers[child_base + axis] = if side == 0 {
                    work.centers[base + axis] - shift
                } else {
                    work.centers[base + axis] + shift
                };
                work.values[child] = work.trial_values[pair * 2 + side];
                work.sizes[child] = rectangle_size(
                    &work.halves[child_base..child_base + d],
                    self.options.variant,
                )?;
                work.stalled[child] = false;
                staged.rectangles += 1;
            }
        }
        work.sizes[index] = rectangle_size(&work.halves[base..base + d], self.options.variant)?;
        Ok(None)
    }
}

use crate::validation::invalid;
fn endpoints<S: RealScalar>(lower: Option<S>, upper: Option<S>) -> Result<(S, S), SciOxideError> {
    match (lower, upper) {
        (Some(lower), Some(upper)) => Ok((lower, upper)),
        _ => Err(invalid(
            scioxide_core::argument!("problem"),
            "missing bound",
        )),
    }
}
fn finite_value<S: RealScalar>(value: S) -> Result<(), SciOxideError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(
            NonFiniteOrigin::Output,
            "objective value must be finite",
        ))
    }
}
fn same_point<S: RealScalar>(left: &[S], right: &[S]) -> bool {
    left.iter()
        .zip(right)
        .all(|(&a, &b)| a == b && (!a.is_zero() || a.is_sign_negative() == b.is_sign_negative()))
}
fn rectangle_size<S: RealScalar>(halves: &[S], variant: DirectVariant) -> Result<S, SciOxideError> {
    match variant {
        DirectVariant::LocallyBiased => Ok(halves.iter().copied().fold(S::ZERO, |a, b| a.max(b))),
        DirectVariant::Original => scioxide_core::checked_scaled_norm(halves),
    }
}
fn select_rectangles<S: RealScalar>(
    work: &mut DirectStorage<S>,
    staged: &DirectResult<S>,
    options: &DirectOptions<S>,
) -> Result<(), SciOxideError> {
    let n = staged.rectangles;
    work.groups.clear();
    work.selected.clear();
    for i in 0..n {
        if work.stalled[i] || work.sizes[i] == S::ZERO {
            continue;
        }
        if let Some(position) = work
            .groups
            .iter()
            .position(|&j| work.sizes[j] == work.sizes[i])
        {
            let old = work.groups[position];
            if work.values[i] < work.values[old] {
                work.groups[position] = i;
            }
        } else {
            work.groups.push(i);
        }
    }
    let best = Scaled::from_value(
        staged
            .best_value
            .ok_or_else(|| invalid(scioxide_core::argument!("work"), "no sampled value"))?,
    )?;
    for &j in &work.groups {
        let dj = Scaled::from_value(work.sizes[j])?;
        let fj = Scaled::from_value(work.values[j])?;
        let mut lower = Scaled::ZERO;
        let mut upper = None;
        for &i in &work.groups {
            if i == j {
                continue;
            }
            let di = Scaled::from_value(work.sizes[i])?;
            let fi = Scaled::from_value(work.values[i])?;
            let slope = fj.sub(fi)?.div(dj.sub(di)?)?;
            if dj.cmp_value(di).is_gt() {
                if slope.cmp_value(lower).is_gt() {
                    lower = slope;
                }
            } else {
                if upper.is_none_or(|u: Scaled<S>| slope.cmp_value(u).is_lt()) {
                    upper = Some(slope);
                }
            }
        }
        let epsilon = if options.variant == DirectVariant::LocallyBiased {
            Scaled::ZERO
        } else {
            Scaled::from_value(options.epsilon)?
        };
        let improvement = fj.sub(best)?.add(epsilon.mul(best.abs())?)?.div(dj)?;
        if improvement.cmp_value(lower).is_gt() {
            lower = improvement;
        }
        if upper.is_none_or(|u| lower.cmp_value(u).is_le()) {
            work.selected.push(j);
            if options.variant == DirectVariant::Original {
                for i in 0..n {
                    if i != j
                        && !work.stalled[i]
                        && work.sizes[i] == work.sizes[j]
                        && work.values[i] == work.values[j]
                    {
                        work.selected.push(i);
                    }
                }
            }
        }
    }
    work.selected.sort_unstable();
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn prepared(variant: DirectVariant) -> (Direct<f64>, DirectWorkspace<f64>) {
        let options = DirectOptions::new(
            Budget::try_from(8).unwrap(),
            Budget::try_from(8).unwrap(),
            Budget::try_from(10_000).unwrap(),
        )
        .with_variant(variant);
        let plan = Direct::new(options).unwrap();
        let work = plan.workspace(1).unwrap();
        (plan, work)
    }

    #[test]
    fn lower_hull_rejects_dominated_size_and_variant_controls_ties() {
        for (variant, expected) in [
            (DirectVariant::Original, &[0, 2, 3][..]),
            (DirectVariant::LocallyBiased, &[0, 2][..]),
        ] {
            let (plan, mut work) = prepared(variant);
            work.staging.rectangles = 4;
            work.staging.best_value = Some(0.0);
            work.storage.values[..4].copy_from_slice(&[0.0, 0.5, 0.0, 0.0]);
            work.storage.sizes[..4].copy_from_slice(&[0.1, 0.2, 0.4, 0.4]);
            select_rectangles(&mut work.storage, &work.staging, plan.options()).unwrap();
            assert_eq!(work.storage.selected, expected);
        }
    }

    #[test]
    fn extreme_finite_fitness_differences_do_not_overflow_selection() {
        let (plan, mut work) = prepared(DirectVariant::Original);
        work.staging.rectangles = 2;
        work.staging.best_value = Some(-f64::MAX);
        work.storage.values[..2].copy_from_slice(&[-f64::MAX, f64::MAX]);
        work.storage.sizes[..2].copy_from_slice(&[0.1, 0.5]);
        select_rectangles(&mut work.storage, &work.staging, plan.options()).unwrap();
        assert!(!work.storage.selected.is_empty());
    }

    #[test]
    fn duplicate_identity_preserves_signed_zero() {
        assert!(!same_point(&[-0.0_f64], &[0.0_f64]));
        assert!(same_point(&[-0.0_f64], &[-0.0_f64]));
    }

    #[test]
    fn construction_and_storage_caps_are_validated() {
        let options = DirectOptions::<f64>::new(
            Budget::try_from(8).unwrap(),
            Budget::try_from(8).unwrap(),
            Budget::try_from(16).unwrap(),
        );
        assert_eq!(options.epsilon(), 1e-4);
        assert_eq!(
            DirectOptions::<f32>::new(Budget::ONE, Budget::ONE, Budget::ONE).epsilon(),
            1e-4
        );
        assert!(Direct::new(options.clone().with_epsilon(-1.0)).is_err());
        assert!(Direct::new(options.clone().with_execution(ExecutionPolicy::Auto)).is_err());
        let plan = Direct::new(options).unwrap();
        assert_eq!(
            plan.workspace_requirements(1).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
    }
}
