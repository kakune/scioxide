//! Powell's conjugate-direction method.

use crate::attempt::{Accepted, AcceptedState, Entry, Shape};
use crate::context::SolveSession;
use crate::scalar::ScalarMinimumWorkspace;
use crate::{
    ObjectiveFunction, OptimizeResult, OptimizeTermination, ScalarMinimizer, ScalarMinimumMethod,
    ScalarMinimumOptions, ScalarMinimumStart,
};
use alloc::vec::Vec;
use core::cell::RefCell;
use scioxide_core::Scaled;
use scioxide_core::backend::{checked_mul, try_reserve};
use scioxide_core::checked_sum;
use scioxide_core::{
    AttemptKind, Bounds, Budget, BudgetKind, EvaluationRole, ExecutionContext, ExecutionPolicy,
    Interval, NumericScalar, PartialResultTrust, ProblemToken, RankThreshold, RealScalar,
    ResidualTolerance, SciOxideError, SolveBudgets, SolveContext, SolveIdentity, SolveScope,
    SolveState, StepTolerance,
};
use scioxide_linalg::{
    HouseholderDecomposition, HouseholderOrientation, LinalgWorkspace, Matrix, RankPolicy,
    RankReport,
};

use crate::validation::invalid;

/// Sets the stopping tolerances, limits, line minimization and initial directions of [`Powell`].
///
/// The defaults are the default [`StepTolerance`] and [`ResidualTolerance`],
/// the default [`RankThreshold`], 1000 sweeps, 2000 objective evaluations, the
/// default [`ScalarMinimumOptions`](crate::ScalarMinimumOptions), no time
/// limit and the coordinate directions. The setters do not check their
/// arguments; [`Powell::new`] does. For bounds, wrap the objective in a
/// [`BoundedObjective`](crate::BoundedObjective). See also
/// [Termination](crate#termination).
#[derive(Debug, Clone)]
pub struct PowellOptions<S: RealScalar> {
    step: StepTolerance<S>,
    function: ResidualTolerance<S>,
    rank: RankThreshold<S>,
    sweeps: Budget,
    evaluations: Budget,
    scalar: ScalarMinimumOptions<S>,
    time: Option<core::time::Duration>,
    directions: Option<Matrix<S>>,
}
impl<S: RealScalar> Default for PowellOptions<S> {
    fn default() -> Self {
        Self {
            step: StepTolerance::default(),
            function: ResidualTolerance::default(),
            rank: RankThreshold::default(),
            sweeps: Budget::ONE.saturating_add(999),
            evaluations: Budget::ONE.saturating_add(1999),
            scalar: ScalarMinimumOptions::default(),
            time: None,
            directions: None,
        }
    }
}
impl<S: RealScalar> PowellOptions<S> {
    /// Sets the tolerance on the coordinate movement of a sweep.
    pub const fn with_step_tolerance(mut self, value: StepTolerance<S>) -> Self {
        self.step = value;
        self
    }
    /// Sets the tolerance on the objective decrease of a sweep.
    pub const fn with_function_tolerance(mut self, value: ResidualTolerance<S>) -> Self {
        self.function = value;
        self
    }
    /// Sets the threshold for the numerical rank of the normalized search
    /// directions, restricted to the coordinates that are not fixed.
    pub const fn with_rank_threshold(mut self, value: RankThreshold<S>) -> Self {
        self.rank = value;
        self
    }
    /// Sets the limit on sweeps. An iteration limit of a parent solve also
    /// counts the iterations of the line minimizations.
    pub const fn with_max_iterations(mut self, value: Budget) -> Self {
        self.sweeps = value;
        self
    }
    /// Sets the objective evaluation limit, which includes the line minimizations.
    pub const fn with_max_evaluations(mut self, value: Budget) -> Self {
        self.evaluations = value;
        self
    }
    /// Sets the options of the line minimizations.
    pub const fn with_scalar_options(mut self, value: ScalarMinimumOptions<S>) -> Self {
        self.scalar = value;
        self
    }
    /// Sets a wall-clock time limit. The execution context must provide a clock.
    pub const fn with_time_limit(mut self, value: core::time::Duration) -> Self {
        self.time = Some(value);
        self
    }
    /// Sets the initial search directions as the columns of an `n × n` matrix.
    ///
    /// Without this setting the directions are the coordinate directions.
    #[must_use]
    pub fn with_directions(mut self, directions: Matrix<S>) -> Self {
        self.directions = Some(directions);
        self
    }
    /// Returns the tolerance on the coordinate movement.
    pub const fn step_tolerance(&self) -> StepTolerance<S> {
        self.step
    }
    /// Returns the tolerance on the objective decrease.
    pub const fn function_tolerance(&self) -> ResidualTolerance<S> {
        self.function
    }
    /// Returns the threshold for the numerical rank of the directions.
    pub const fn rank_threshold(&self) -> RankThreshold<S> {
        self.rank
    }
    /// Returns the limit on sweeps.
    pub const fn max_iterations(&self) -> Budget {
        self.sweeps
    }
    /// Returns the objective evaluation limit.
    pub const fn max_evaluations(&self) -> Budget {
        self.evaluations
    }
    /// Returns the options of the line minimizations.
    pub const fn scalar_options(&self) -> ScalarMinimumOptions<S> {
        self.scalar
    }
    /// Returns the time limit, if set.
    pub const fn time_limit(&self) -> Option<core::time::Duration> {
        self.time
    }
    /// Returns the initial directions, if set.
    pub fn directions(&self) -> Option<scioxide_linalg::MatrixView<'_, S>> {
        self.directions.as_ref().map(Matrix::view)
    }
}

/// Reports the search directions and the last sweep of a [`Powell`] solve.
///
/// These values describe the samples; they do not prove that the point is a
/// minimum. A value is `None` when no sweep completed.
#[derive(Debug, Clone, Copy)]
pub struct PowellDiagnostics<S: RealScalar> {
    free: usize,
    rank: Option<RankReport<S>>,
    replacements: usize,
    rejected: usize,
    step: Option<S>,
    step_scale: Option<S>,
    change: Option<S>,
    function_scale: Option<S>,
}
impl<S: RealScalar> Default for PowellDiagnostics<S> {
    fn default() -> Self {
        Self {
            free: 0,
            rank: None,
            replacements: 0,
            rejected: 0,
            step: None,
            step_scale: None,
            change: None,
            function_scale: None,
        }
    }
}
impl<S: RealScalar> PowellDiagnostics<S> {
    /// Returns the number of coordinates that are not fixed by equal bounds.
    pub const fn free_dimension(self) -> usize {
        self.free
    }
    /// Returns the numerical rank of the current directions, or of the
    /// rejected initial directions.
    pub const fn rank(self) -> Option<RankReport<S>> {
        self.rank
    }
    /// Returns the number of directions replaced by a sweep displacement.
    pub const fn replacements(self) -> usize {
        self.replacements
    }
    /// Returns the number of replacements skipped because the directions
    /// would become numerically dependent.
    pub const fn rejected_replacements(self) -> usize {
        self.rejected
    }
    /// Returns the largest coordinate movement of the last complete sweep.
    pub const fn step(self) -> Option<S> {
        self.step
    }
    /// Returns the coordinate magnitude that scales the step tolerance.
    pub const fn step_scale(self) -> Option<S> {
        self.step_scale
    }
    /// Returns the objective decrease of the last complete sweep.
    pub const fn function_change(self) -> Option<S> {
        self.change
    }
    /// Returns the objective magnitude that scales the objective tolerance.
    pub const fn function_scale(self) -> Option<S> {
        self.function_scale
    }
    /// Returns the coordinate movement with its scale, `max(1, step_scale)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the scaled value is not representable.
    pub fn step_report(self) -> Result<Option<scioxide_core::ResidualReport<S>>, SciOxideError> {
        self.step
            .zip(self.step_scale)
            .map(|(raw, scale)| report(raw, scale))
            .transpose()
    }
    /// Returns the objective decrease with its scale, `max(1, function_scale)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the scaled value is not representable.
    pub fn function_report(
        self,
    ) -> Result<Option<scioxide_core::ResidualReport<S>>, SciOxideError> {
        self.change
            .zip(self.function_scale)
            .map(|(raw, scale)| report(raw, scale))
            .transpose()
    }
}

fn report<S: RealScalar>(
    raw: S,
    scale: S,
) -> Result<scioxide_core::ResidualReport<S>, SciOxideError> {
    scioxide_core::ResidualReport::new(
        raw,
        scioxide_core::NormSelection::Infinity,
        scioxide_core::ResidualScaling::new(
            scioxide_core::ResidualScalingRule::Reference,
            scale.max(S::ONE),
        )?,
    )
}

/// O(n²) direction, QR and trial storage.
#[derive(Debug)]
pub(crate) struct PowellStorage<S: RealScalar> {
    dimension: usize,
    directions: Matrix<S>,
    candidate_basis: Matrix<S>,
    qr: HouseholderDecomposition<S>,
    qr_work: LinalgWorkspace<'static, S>,
    point: Vec<S>,
    origin: Vec<S>,
    direction: Vec<S>,
    trial: RefCell<Vec<S>>,
    best: RefCell<Vec<S>>,
    best_value: core::cell::Cell<Option<S>>,
    accepted: Vec<S>,
    scalar: ScalarMinimumWorkspace<S>,
    child: SolveState,
    diagnostics: PowellDiagnostics<S>,
    sweeps: usize,
    line_step: S,
}
impl<S: RealScalar> Default for PowellStorage<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> PowellStorage<S> {
    /// Construct empty storage without allocating.
    fn new() -> Self {
        Self {
            dimension: 0,
            directions: Matrix::empty(),
            candidate_basis: Matrix::empty(),
            qr: HouseholderDecomposition::empty(),
            qr_work: LinalgWorkspace::new(),
            point: Vec::new(),
            origin: Vec::new(),
            direction: Vec::new(),
            trial: RefCell::new(Vec::new()),
            best: RefCell::new(Vec::new()),
            best_value: core::cell::Cell::new(None),
            accepted: Vec::new(),
            scalar: ScalarMinimumWorkspace::new(),
            child: SolveState::new(),
            diagnostics: PowellDiagnostics::default(),
            sweeps: 0,
            line_step: S::ZERO,
        }
    }
    fn fits(&self, dimension: usize) -> bool {
        self.dimension == dimension
            && self.point.len() == dimension
            && self.directions.rows() == dimension
            && dimension != 0
    }
    /// Prepare all storage, including both QR buffers.
    fn try_reserve(&mut self, dimension: usize) -> Result<(), SciOxideError> {
        checked_mul(
            dimension,
            dimension,
            scioxide_core::quantity!("powell_directions"),
        )?;
        if self.fits(dimension) {
            return Ok(());
        }
        self.directions.try_resize_flat(dimension, dimension)?;
        self.candidate_basis.try_resize_flat(dimension, dimension)?;
        self.candidate_basis.as_mut_slice().fill(S::ZERO);
        for buffer in [
            &mut self.point,
            &mut self.origin,
            &mut self.direction,
            &mut self.accepted,
            self.trial.get_mut(),
            self.best.get_mut(),
        ] {
            try_reserve(
                buffer,
                dimension.saturating_sub(buffer.len()),
                scioxide_core::quantity!("powell_vectors"),
            )?;
            buffer.resize(dimension, S::ZERO);
        }
        // Factor outputs swap with QR staging: warm both sides explicitly.
        for _ in 0..2 {
            HouseholderDecomposition::factor_into(
                &self.candidate_basis,
                HouseholderOrientation::Qr,
                scioxide_linalg::HouseholderPivoting::Column,
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential),
                &mut self.qr,
                &mut self.qr_work,
                scioxide_core::ExecutionContext::sequential(),
            )?;
        }
        self.dimension = dimension;
        Ok(())
    }
}

impl<S: RealScalar> Accepted<S> for PowellStorage<S> {
    fn accepted(&self) -> Option<AcceptedState<'_, S>> {
        Some(AcceptedState {
            point: &self.accepted,
            value: self.best_value.get()?,
            gradient: None,
            iterations: self.sweeps,
        })
    }
}

/// Describes the workspace that a [`Powell`] solve needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowellWorkspaceRequirements {
    dimension: usize,
}

impl PowellWorkspaceRequirements {
    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }
}

impl<S: RealScalar> Shape<PowellStorage<S>> for PowellWorkspaceRequirements {
    fn fits(&self, storage: &PowellStorage<S>) -> bool {
        storage.fits(self.dimension)
    }
    fn prepare(&self, storage: &mut PowellStorage<S>) -> Result<(), SciOxideError> {
        storage.try_reserve(self.dimension)
    }
}

crate::attempt::local_workspace!(
    /// Holds the O(n²) storage and the record of the last call for [`Powell`].
    PowellWorkspace,
    PowellStorage<S>,
    PowellWorkspaceRequirements
);

/// Minimizes a function without derivatives by Powell's conjugate-direction method.
///
/// Each sweep minimizes the objective along each of `n` search directions in
/// turn, starting from the coordinate directions or
/// [`PowellOptions::with_directions`]. Each line minimization uses
/// [`ScalarMinimizer`](crate::ScalarMinimizer) with Brent's method. After a
/// sweep, the net displacement replaces the direction of largest decrease
/// when Powell's test predicts that this keeps the directions well spread;
/// a replacement that would make the directions numerically dependent is
/// skipped. The solve converges when both the largest coordinate movement
/// and the objective decrease of a sweep are below their tolerances.
///
/// A [`BoundedObjective`](crate::BoundedObjective) restricts the search to
/// its box. Every bound must be finite, equal bounds fix a coordinate, and
/// line searches stay inside the box. A bounded line search first minimizes
/// over the whole segment of the line inside the box. If that finds no value
/// below the current one, it evaluates one step on each side of the current
/// point, of length 1 or up to the end of the segment, and minimizes between
/// the current point and a lower probe, or over the bracket that two higher
/// probes form around the current point. A decrease next to the current point
/// is therefore not missed. A line search
/// that stalls or reaches its own iteration limit ends that line, not the
/// solve.
///
/// Each sweep costs O(n²) arithmetic plus the objective evaluations of up to
/// `n + 1` line searches. Each direction replacement that passes Powell's
/// test adds a column-pivoted QR factorization of the n×n direction matrix,
/// O(n³) arithmetic.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Bounds, ExecutionContext, SciOxideError};
/// use scioxide_optimize::{BoundedObjective, ObjectiveFn, OptimizeResult, Powell};
///
/// # fn main() -> Result<(), SciOxideError> {
/// // On [−1, 1]², (x₀ − 2)² + x₁² is smallest at (1, 0).
/// let objective = ObjectiveFn::new(2, |x: &[f64]| Ok((x[0] - 2.0).powi(2) + x[1] * x[1]));
/// let bounds = Bounds::new(vec![(Some(-1.0), Some(1.0)), (Some(-1.0), Some(1.0))])?;
/// let problem = BoundedObjective::new(&objective, &bounds)?;
/// let solver = Powell::<f64>::default();
/// let mut workspace = solver.workspace(2)?;
/// let mut output = OptimizeResult::<f64>::empty();
/// solver.minimize_into(
///     &problem,
///     &[0.0, 0.5],
///     &mut output,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert!((output.x()[0] - 1.0).abs() < 1e-8);
/// # Ok(())
/// # }
/// ```
///
/// # References
///
/// - M. J. D. Powell, "An efficient method for finding the minimum of a
///   function of several variables without calculating derivatives",
///   *Comput. J.* 7(2), 155–162, 1964.
/// - R. P. Brent, *Algorithms for Minimization without Derivatives*,
///   Prentice-Hall, 1973, ch. 7.
#[derive(Debug, Clone)]
pub struct Powell<S: RealScalar> {
    options: PowellOptions<S>,
}
impl<S: RealScalar> Default for Powell<S> {
    fn default() -> Self {
        Self {
            options: PowellOptions::default(),
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> Powell<S> {
    /// Creates the solver after checking `options`.
    ///
    /// The rank of the directions is checked by each solve.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the initial directions are not square.
    /// - `NonFinite` if they contain NaN or infinity.
    pub fn new(options: PowellOptions<S>) -> Result<Self, SciOxideError> {
        if let Some(directions) = &options.directions {
            if directions.rows() != directions.cols() {
                return Err(invalid(
                    scioxide_core::argument!("directions"),
                    "directions must be square",
                ));
            }
            crate::utils::validate_finite_origin(
                directions.as_slice(),
                scioxide_core::NonFiniteOrigin::Input,
            )?;
        }
        Ok(Self { options })
    }
    /// Returns the options.
    pub const fn options(&self) -> &PowellOptions<S> {
        &self.options
    }
    /// Returns the workspace requirements for `dimension` variables.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` if the size of the direction matrix overflows `usize`.
    pub fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<PowellWorkspaceRequirements, SciOxideError> {
        checked_mul(
            dimension,
            dimension,
            scioxide_core::quantity!("powell_directions"),
        )?;
        Ok(PowellWorkspaceRequirements { dimension })
    }
    /// Allocates a workspace for `dimension` variables.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// reserved.
    pub fn workspace(&self, dimension: usize) -> Result<PowellWorkspace<S>, SciOxideError> {
        let mut workspace = PowellWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }
    /// Minimizes `f` from `x0` and writes the result to `output`.
    ///
    /// `f` may be a [`BoundedObjective`](crate::BoundedObjective). `scope` is an
    /// [execution context](scioxide_core#execution-contexts) or a parent
    /// [`SolveContext`](scioxide_core::SolveContext). When a limit, cancellation
    /// or deadline stops the solve after it has a finite sample, the call
    /// returns the best sample with an unsuccessful termination. On error,
    /// `output` is unchanged and [`PowellWorkspace::last_attempt`] describes
    /// the failed call. Callbacks run on the calling thread.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x0`, the bounds or the initial directions do
    ///   not have `f.dim()` entries.
    /// - `InvalidInput` if a bound is infinite, `x0` lies outside the bounds,
    ///   or the initial directions do not have full rank on the coordinates
    ///   that are not fixed.
    /// - `NonFinite` if `x0` contains NaN or infinity.
    /// - `CapabilityUnavailable` if a time limit is set and `scope` has no clock.
    /// - `ConvergenceFailure` or `ExecutionStopped` if a limit, cancellation
    ///   or deadline stops the solve before it has a finite sample.
    /// - `SizeOverflow`, `ResourceLimit` or `AllocationFailure` if the
    ///   workspace cannot grow.
    ///
    /// Errors returned by `f` are returned unchanged.
    pub fn minimize_into<'scope, 'capability, F>(
        &self,
        f: &F,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut PowellWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError>
    where
        F: ObjectiveFunction<S> + ?Sized,
    {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        workspace.try_reserve(&self.workspace_requirements(x0.len())?)?;
        self.prepare(f, x0, &mut workspace.storage)?;
        crate::attempt::preflight(&scope, self.options.time)?;
        let entry = Entry::Call {
            identity: self.identity(ProblemToken::new(0), x0.len()),
            budgets: SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, self.options.evaluations),
        };
        crate::attempt::minimize(
            &mut workspace.shell,
            &mut workspace.storage,
            &mut workspace.staged,
            output,
            scope,
            entry,
            |context, storage, staged| self.minimize_in_context(f, staged, storage, context),
        )
    }
    /// Identity of a Powell epoch: method, problem, dimension and options.
    pub(crate) fn identity(&self, problem: ProblemToken, dimension: usize) -> SolveIdentity {
        SolveIdentity::new(
            problem,
            scioxide_core::algorithm_id!("optimize", "Powell", 1),
            dimension as u128,
            crate::context::options_token(&self.options),
        )
    }
    /// Validate the start and bounds, reserve storage and build the initial basis.
    pub(crate) fn prepare<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x0: &[S],
        work: &mut PowellStorage<S>,
    ) -> Result<(), SciOxideError> {
        crate::utils::validate_x0(f, x0)?;
        self.validate(x0, f.bounds())?;
        work.try_reserve(x0.len())?;
        self.initialize(x0, work, f.bounds())
    }
    /// Run a prepared solve under an epoch the caller has already begun.
    pub(crate) fn minimize_in_context<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        output: &mut OptimizeResult<S>,
        work: &mut PowellStorage<S>,
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        output.try_reserve_without_gradient(work.dimension)?;
        if let Some(time) = self.options.time {
            ctx.limit_time(time)?;
        }
        let session = SolveSession::new(ctx, scioxide_core::argument!("scope"));
        let outcome = self.run(f, work, &session);
        work.accepted.copy_from_slice(&work.best.borrow());
        let termination = match outcome {
            Ok(stop) => stop,
            Err(error) => match (session.stopped(), work.best_value.get()) {
                (Some(stop), Some(_)) => stop,
                _ => return Err(error),
            },
        };
        let value = work.best_value.get().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("f"),
                "no finite Powell observation",
            )
        })?;
        crate::utils::commit_optimize_result(
            output,
            &work.accepted,
            value,
            None,
            work.sweeps,
            termination,
            "Powell solve completed",
        )?;
        output.method = crate::OptimizeDiagnostics::Powell(work.diagnostics);
        crate::utils::synchronize_metadata(output, &session, PartialResultTrust::Admissible)
    }
    fn validate(&self, x0: &[S], bounds: Option<&Bounds<S>>) -> Result<(), SciOxideError> {
        let n = x0.len();
        for (_label, actual) in [
            (
                "Powell direction dimension",
                self.options.directions.as_ref().map(Matrix::rows),
            ),
            ("Powell bound dimension", bounds.map(Bounds::len)),
        ] {
            if let Some(actual) = actual.filter(|&actual| actual != n) {
                return Err(SciOxideError::dimension_mismatch(
                    scioxide_core::Dimensions::one(n),
                    scioxide_core::Dimensions::one(actual),
                ));
            }
        }
        if let Some(bounds) = bounds {
            if bounds
                .as_slice()
                .iter()
                .any(|&(lo, hi)| lo.is_none() || hi.is_none())
            {
                return Err(invalid(
                    scioxide_core::argument!("f"),
                    "Powell bounds require two finite endpoints per variable",
                ));
            }
            for (&x, &(lo, hi)) in x0.iter().zip(bounds.as_slice()) {
                if lo.is_some_and(|lo| x < lo) || hi.is_some_and(|hi| x > hi) {
                    return Err(invalid(
                        scioxide_core::argument!("x0"),
                        "initial point is outside bounds",
                    ));
                }
            }
        }
        Ok(())
    }
    fn initialize(
        &self,
        x0: &[S],
        w: &mut PowellStorage<S>,
        bounds: Option<&Bounds<S>>,
    ) -> Result<(), SciOxideError> {
        let fixed = |index: usize| {
            bounds.is_some_and(|b| {
                let (lo, hi) = b.as_slice()[index];
                lo == hi
            })
        };
        let n = x0.len();
        w.diagnostics = PowellDiagnostics {
            free: (0..n).filter(|&i| !fixed(i)).count(),
            ..PowellDiagnostics::default()
        };
        w.sweeps = 0;
        w.best_value.set(None);
        w.point.copy_from_slice(x0);
        for col in 0..n {
            for (row, value) in w.direction.iter_mut().enumerate() {
                *value = if fixed(row) {
                    S::ZERO
                } else if let Some(matrix) = &self.options.directions {
                    matrix.as_slice()[row * n + col]
                } else if row == col {
                    S::ONE
                } else {
                    S::ZERO
                };
            }
            normalize(&mut w.direction);
            for (row, &value) in w.direction.iter().enumerate() {
                w.directions.as_mut_slice()[row * n + col] = value;
            }
        }
        w.candidate_basis
            .as_mut_slice()
            .copy_from_slice(w.directions.as_slice());
        let rank = self.rank(w)?;
        w.diagnostics.rank = Some(rank);
        if rank.rank() != w.diagnostics.free {
            return Err(invalid(
                scioxide_core::argument!("self"),
                "initial directions do not span the free variables",
            ));
        }
        Ok(())
    }
    fn rank(&self, w: &mut PowellStorage<S>) -> Result<RankReport<S>, SciOxideError> {
        HouseholderDecomposition::factor_into(
            &w.candidate_basis,
            HouseholderOrientation::Qr,
            scioxide_linalg::HouseholderPivoting::Column,
            &scioxide_linalg::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            &mut w.qr,
            &mut w.qr_work,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        w.qr.analyze_rank(RankPolicy::Threshold(self.options.rank))
    }
    fn remaining(&self, session: &SolveSession<'_, '_, '_>) -> Result<usize, SciOxideError> {
        Ok(self.options.evaluations.get().saturating_sub(
            session
                .combined_counts()?
                .provider(EvaluationRole::Objective),
        ))
    }
    /// Observe one point; a non-finite value rejects it and returns `None`.
    fn observe<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        point: &[S],
        w: &PowellStorage<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<Option<S>, SciOxideError> {
        if self.remaining(session)? == 0 {
            session.set_stopped(OptimizeTermination::EvaluationLimit);
            return Err(session.stop_error(OptimizeTermination::EvaluationLimit)?);
        }
        session.proposal()?;
        let value = session.invoke(EvaluationRole::Objective, || f.eval(point))?;
        if !value.is_finite() {
            session.reject()?;
            return Ok(None);
        }
        let improved = w.best_value.get().is_none_or(|best| value < best);
        remember(point, value, &w.best, &w.best_value)?;
        if improved {
            session.accept(PartialResultTrust::Admissible)?;
        } else {
            session.reject()?;
        }
        Ok(Some(value))
    }
    fn line<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        w: &mut PowellStorage<S>,
        session: &SolveSession<'_, '_, '_>,
        value: &mut S,
    ) -> Result<Option<OptimizeTermination>, SciOxideError> {
        w.line_step = S::ZERO;
        if w.direction.iter().all(|v| v.is_zero()) {
            return Ok(None);
        }
        let interval = match f.bounds() {
            Some(bounds) => {
                let (lo, hi) = line_interval(&w.point, &w.direction, bounds)?;
                if lo == hi {
                    return Ok(None);
                }
                Some((lo, hi))
            }
            None => None,
        };
        let start = match interval {
            Some((lo, hi)) => ScalarMinimumStart::Bounded(Interval::new(lo, hi)?),
            None => ScalarMinimumStart::Seeds(S::ZERO, S::ONE),
        };
        let Some(mut best) = self.scalar_line(f, w, session, start)? else {
            return Ok(Some(OptimizeTermination::EvaluationLimit));
        };
        let mut stop = line_stop(session, best.2)?;
        // A search over the whole box interval can settle in another basin.
        // Then a local search around the current point finds any decrease
        // that this line still offers.
        if stop.is_none()
            && let Some((lo, hi)) = interval
            && best.1 >= *value
        {
            match self.local_line(f, w, session, lo, hi, *value)? {
                Ok(Some(local)) => {
                    stop = line_stop(session, local.2)?;
                    if local.1 < best.1 {
                        best = local;
                    }
                }
                Ok(None) => {}
                Err(limit) => return Ok(Some(limit)),
            }
        }
        let (alpha, fun, _) = best;
        if fun < *value {
            let trial = w.trial.get_mut();
            if !trial_point(&w.point, &w.direction, alpha, f.bounds(), trial)? {
                return Err(invalid(
                    scioxide_core::argument!("work"),
                    "evaluated Powell line minimum became unrepresentable",
                ));
            }
            let movement =
                trial
                    .iter()
                    .zip(&w.point)
                    .try_fold(S::ZERO, |largest, (&x, &old)| {
                        Ok::<S, SciOxideError>(largest.max(difference(x, old)?.abs().round()?))
                    })?;
            w.line_step = movement;
            session.accept(PartialResultTrust::Admissible)?;
            w.point.copy_from_slice(trial);
            *value = fun;
        }
        Ok(stop)
    }
    /// Probe one step on each side of the current point, of length at most 1
    /// inside `[lo, hi]`, and search the segment or bracket they establish. Returns the outer
    /// `Err` for an exhausted evaluation budget.
    #[allow(clippy::too_many_arguments)]
    fn local_line<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        w: &mut PowellStorage<S>,
        session: &SolveSession<'_, '_, '_>,
        lo: S,
        hi: S,
        value: S,
    ) -> Result<Result<Option<LineMinimum<S>>, OptimizeTermination>, SciOxideError> {
        let ends = [
            (hi > S::ZERO).then(|| hi.min(S::ONE)),
            (lo < S::ZERO).then(|| lo.max(-S::ONE)),
        ];
        let mut samples = [None; 2];
        for (end, sample) in ends.iter().zip(&mut samples) {
            if let Some(alpha) = *end {
                if self.remaining(session)? == 0 {
                    return Ok(Err(OptimizeTermination::EvaluationLimit));
                }
                let representable = {
                    let mut trial = w.trial.try_borrow_mut().map_err(|_| {
                        invalid(scioxide_core::argument!("work"), "Powell trial borrow")
                    })?;
                    trial_point(&w.point, &w.direction, alpha, f.bounds(), &mut trial)?
                };
                if representable {
                    let trial = w.trial.try_borrow().map_err(|_| {
                        invalid(scioxide_core::argument!("work"), "Powell trial borrow")
                    })?;
                    *sample = self.observe(f, &trial, w, session)?.map(|v| (alpha, v));
                }
            }
        }
        let lower = samples
            .iter()
            .flatten()
            .filter(|(_, v)| *v < value)
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))
            .copied();
        let start = if let Some((alpha, _)) = lower {
            ScalarMinimumStart::Bounded(Interval::new(alpha.min(S::ZERO), alpha.max(S::ZERO))?)
        } else if let [Some((upper, fu)), Some((lower, fl))] = samples {
            match crate::MinimumBracket::new([lower, S::ZERO, upper], [fl, value, fu]) {
                Ok(bracket) => ScalarMinimumStart::Bracket(bracket),
                Err(_) => return Ok(Ok(None)),
            }
        } else if let Some((alpha, _)) = samples.iter().flatten().next().copied() {
            ScalarMinimumStart::Bounded(Interval::new(alpha.min(S::ZERO), alpha.max(S::ZERO))?)
        } else {
            return Ok(Ok(None));
        };
        let Some(result) = self.scalar_line(f, w, session, start)? else {
            return Ok(Err(OptimizeTermination::EvaluationLimit));
        };
        Ok(Ok(Some(match lower {
            Some((alpha, v)) if v < result.1 => (alpha, v, result.2),
            _ => result,
        })))
    }
    /// One Brent line minimization in a child solve. `None` means the parent
    /// evaluation budget was exhausted before the search started.
    fn scalar_line<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        w: &mut PowellStorage<S>,
        session: &SolveSession<'_, '_, '_>,
        start: ScalarMinimumStart<S>,
    ) -> Result<Option<LineMinimum<S>>, SciOxideError> {
        let remaining = self.remaining(session)?;
        if remaining == 0 {
            return Ok(None);
        }
        session.check()?;
        let scalar = ScalarMinimizer::new(ScalarMinimumMethod::Brent, self.options.scalar);
        let identity = scalar.identity(start)?;
        let objective = |alpha: S| -> Result<S, SciOxideError> {
            let mut trial = w
                .trial
                .try_borrow_mut()
                .map_err(|_| invalid(scioxide_core::argument!("work"), "Powell trial borrow"))?;
            // An unrepresentable point or a non-finite value rejects the trial;
            // the scalar search ranks it above every finite observation.
            if !trial_point(&w.point, &w.direction, alpha, f.bounds(), &mut trial)? {
                return Ok(S::INFINITY);
            }
            let observed = f.eval(&trial)?;
            if observed.is_finite() {
                remember(&trial, observed, &w.best, &w.best_value)?;
            }
            Ok(observed)
        };
        let mut denied = false;
        let result = session.with_child(
            &mut w.child,
            identity,
            SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(remaining)?),
            |ctx| {
                if !ctx.can_consume(BudgetKind::Evaluation, 1)?.is_granted() {
                    denied = true;
                    return Err(crate::context::budget_error(
                        ctx,
                        BudgetKind::Evaluation,
                        "parent evaluation budget exhausted before line initialization",
                    ));
                }
                scalar.minimize_in_context(&objective, start, &mut w.scalar, ctx)
            },
        );
        match result {
            Ok(result) => Ok(Some((result.x(), result.fun(), result.termination()))),
            Err(_) if denied => Ok(None),
            Err(error) => {
                session.record_execution_stop(&error);
                Err(error)
            }
        }
    }
    fn run<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        w: &mut PowellStorage<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<OptimizeTermination, SciOxideError> {
        let mut value = self.observe(f, &w.point, w, session)?.ok_or_else(|| {
            SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Callback,
                "objective returned a non-finite value at the start",
            )
        })?;
        session.check()?;
        if w.diagnostics.free == 0 {
            return Ok(OptimizeTermination::Converged);
        }
        let n = w.dimension;
        for sweep in 0..self.options.sweeps.get() {
            if let Some(stop) = session.iteration()? {
                return Ok(stop);
            }
            w.origin.copy_from_slice(&w.point);
            let initial = value;
            let mut largest = Scaled::ZERO;
            let mut replace = 0;
            let mut movement = S::ZERO;
            for col in 0..n {
                for (row, d) in w.direction.iter_mut().enumerate() {
                    *d = w.directions.as_slice()[row * n + col];
                }
                let before = value;
                if let Some(stop) = self.line(f, w, session, &mut value)? {
                    return Ok(stop);
                }
                let decrease = difference(before, value)?;
                if decrease.cmp_value(largest).is_gt() {
                    largest = decrease;
                    replace = col;
                }
                // The per-line maximum is populated by the line shell.
                movement = movement.max(w.line_step);
            }
            for (d, (&x, &old)) in w.direction.iter_mut().zip(w.point.iter().zip(&w.origin)) {
                *d = checked_sum(&[x, -old])?;
            }
            movement = movement.max(max_abs(&w.direction));
            let scale = max_abs(&w.point).max(max_abs(&w.origin));
            let change = difference(initial, value)?.abs().round()?;
            let function_scale = initial.abs().max(value.abs());
            w.sweeps = sweep + 1;
            w.diagnostics.step = Some(movement);
            w.diagnostics.step_scale = Some(scale);
            w.diagnostics.change = Some(change);
            w.diagnostics.function_scale = Some(function_scale);
            if self.options.step.is_satisfied(movement, scale)?
                && self.options.function.is_satisfied(change, function_scale)?
            {
                return Ok(OptimizeTermination::Converged);
            }
            if w.direction.iter().all(|d| d.is_zero()) {
                return Ok(OptimizeTermination::Stagnation);
            }
            // Powell's test is defined at x + (x - sweep_origin). A clipped
            // extrapolation is not the same model: skip the test if infeasible.
            let feasible = if let Some(bounds) = f.bounds() {
                let (_, upper) = line_interval(&w.point, &w.direction, bounds)?;
                upper >= S::ONE
            } else {
                true
            };
            if !feasible {
                continue;
            }
            let representable = {
                let mut trial = w.trial.try_borrow_mut().map_err(|_| {
                    invalid(scioxide_core::argument!("work"), "Powell trial borrow")
                })?;
                trial_point(&w.point, &w.direction, S::ONE, f.bounds(), &mut trial)?
            };
            // A rejected extrapolation keeps the current directions.
            let extrapolated = if representable {
                let trial = w.trial.try_borrow().map_err(|_| {
                    invalid(scioxide_core::argument!("work"), "Powell trial borrow")
                })?;
                self.observe(f, &trial, w, session)?
            } else {
                None
            };
            let Some(extrapolated) = extrapolated else {
                continue;
            };
            if !replacement_test(initial, value, extrapolated, largest)? {
                continue;
            }
            normalize(&mut w.direction);
            w.candidate_basis
                .as_mut_slice()
                .copy_from_slice(w.directions.as_slice());
            for (row, &d) in w.direction.iter().enumerate() {
                w.candidate_basis.as_mut_slice()[row * n + replace] = d;
            }
            let rank = self.rank(w)?;
            if rank.rank() != w.diagnostics.free {
                w.diagnostics.rejected += 1;
                continue;
            }
            if let Some(stop) = self.line(f, w, session, &mut value)? {
                return Ok(stop);
            }
            // Standard modified Powell ordering: move the last old direction
            // into the replaced slot and append the displacement direction.
            for row in 0..n {
                let last = w.directions.as_slice()[row * n + n - 1];
                w.directions.as_mut_slice()[row * n + replace] = last;
                w.directions.as_mut_slice()[row * n + n - 1] = w.direction[row];
            }
            w.diagnostics.rank = Some(rank);
            w.diagnostics.replacements += 1;
        }
        Ok(OptimizeTermination::IterationLimit)
    }
}
impl<S> Powell<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Minimizes the objective with this method's settings and owned workspace.
    ///
    /// Returns the complete result, including unsuccessful termination. Call
    /// `require_success()` to require the stopping test to pass.
    ///
    /// # Errors
    ///
    /// Returns the input, callback, allocation and numerical errors of `minimize_into`.
    pub fn minimize<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x0: &[S],
    ) -> Result<OptimizeResult<S>, SciOxideError> {
        let mut workspace = self.workspace(x0.len())?;
        let mut output = OptimizeResult::empty();
        self.minimize_into(
            f,
            x0,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}

impl<S, F> crate::traits::Optimizer<S, F> for Powell<S>
where
    S: RealScalar + NumericScalar<Real = S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    fn minimize(&self, f: &F, x0: &[S]) -> Result<OptimizeResult<S>, SciOxideError> {
        self.minimize(f, x0)
    }
}
fn max_abs<S: RealScalar>(values: &[S]) -> S {
    values.iter().fold(S::ZERO, |m, x| m.max(x.abs()))
}
fn normalize<S: RealScalar>(direction: &mut [S]) {
    let scale = max_abs(direction);
    if !scale.is_zero() {
        direction.iter_mut().for_each(|d| *d /= scale);
    }
}
fn difference<S: RealScalar>(a: S, b: S) -> Result<Scaled<S>, SciOxideError> {
    Scaled::from_value(a)?.sub(Scaled::from_value(b)?)
}
fn replacement_test<S: RealScalar>(
    initial: S,
    value: S,
    extrapolated: S,
    largest: Scaled<S>,
) -> Result<bool, SciOxideError> {
    if extrapolated >= initial || largest.is_zero() {
        return Ok(false);
    }
    let two = Scaled::from_value(S::TWO)?;
    let change = difference(initial, value)?;
    let curvature = change.add(difference(extrapolated, value)?)?;
    let rest = change.sub(largest)?;
    let extrap_change = difference(initial, extrapolated)?;
    let left = two.mul(curvature)?.mul(rest)?.mul(rest)?;
    let right = largest.mul(extrap_change)?.mul(extrap_change)?;
    Ok(left.cmp_value(right).is_lt())
}
fn remember<S: RealScalar>(
    point: &[S],
    value: S,
    best: &RefCell<Vec<S>>,
    best_value: &core::cell::Cell<Option<S>>,
) -> Result<(), SciOxideError> {
    crate::utils::validate_finite(&[value])?;
    if best_value.get().is_none_or(|old| value < old) {
        best.try_borrow_mut()
            .map_err(|_| invalid(scioxide_core::argument!("work"), "Powell best borrow"))?
            .copy_from_slice(point);
        best_value.set(Some(value));
    }
    Ok(())
}
/// Step, value and status of one line minimization.
type LineMinimum<S> = (S, S, OptimizeTermination);

/// Map the status of a line search to the solve: a converged, stalled or
/// line-capped search ends only this line; exhausted shared budgets, deadlines
/// and cancellation end the solve.
fn line_stop(
    session: &SolveSession<'_, '_, '_>,
    termination: OptimizeTermination,
) -> Result<Option<OptimizeTermination>, SciOxideError> {
    match termination {
        termination if termination.is_success() => Ok(None),
        OptimizeTermination::Stagnation => Ok(None),
        termination => {
            session.inherit_child_stop(Some(termination))?;
            Ok(session.stopped())
        }
    }
}
fn line_interval<S: RealScalar>(
    point: &[S],
    direction: &[S],
    bounds: &Bounds<S>,
) -> Result<(S, S), SciOxideError> {
    let mut lower = -S::MAX;
    let mut upper = S::MAX;
    for ((&x, &d), &(lo, hi)) in point.iter().zip(direction).zip(bounds.as_slice()) {
        if d.is_zero() {
            continue;
        }
        let (lo, hi) = (
            lo.ok_or_else(|| invalid(scioxide_core::argument!("self"), "missing bound"))?,
            hi.ok_or_else(|| invalid(scioxide_core::argument!("self"), "missing bound"))?,
        );
        let ratio = |bound| -> Result<S, SciOxideError> {
            let ratio = difference(bound, x)?.div(Scaled::from_value(d)?)?;
            if ratio.abs_cmp(Scaled::from_value(S::MAX)?).is_gt() {
                Ok(if ratio.is_sign_negative() {
                    -S::MAX
                } else {
                    S::MAX
                })
            } else {
                ratio.round()
            }
        };
        let a = ratio(lo)?;
        let b = ratio(hi)?;
        lower = lower.max(a.min(b));
        upper = upper.min(a.max(b));
    }
    Ok((lower, upper))
}
/// Write the projected trial point; `false` when a coordinate overflows.
fn trial_point<S: RealScalar>(
    point: &[S],
    direction: &[S],
    alpha: S,
    bounds: Option<&Bounds<S>>,
    output: &mut [S],
) -> Result<bool, SciOxideError> {
    if !crate::utils::fill_exact_trial(output, point, alpha, direction)? {
        return Ok(false);
    }
    if let Some(bounds) = bounds {
        for (trial, &side) in output.iter_mut().zip(bounds.as_slice()) {
            *trial = crate::geometry::project_component(*trial, side);
        }
    }
    Ok(true)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn unsuccessful_extrapolation_is_rejected_without_an_accepted_state() {
        let objective = crate::ObjectiveFn::new(1, |x: &[f64]| Ok(x[0] * x[0]));
        let solver = Powell::default();
        let work = solver.workspace(1).unwrap().storage;
        crate::context::sequential_with_budgets(
            solver.identity(ProblemToken::new(1), 1),
            SolveBudgets::unbounded(),
            |ctx| {
                let session = SolveSession::new(ctx, scioxide_core::argument!("ctx"));
                solver.observe(&objective, &[1.0], &work, &session)?;
                solver.observe(&objective, &[2.0], &work, &session)?;
                assert_eq!(session.counts()?.accepted(), 1);
                assert_eq!(session.counts()?.rejected(), 1);
                assert_eq!(
                    session
                        .combined_counts()?
                        .provider(EvaluationRole::Objective),
                    2
                );
                assert_eq!(work.best_value.get(), Some(1.0));
                Ok(())
            },
        )
        .unwrap();
    }
    #[test]
    fn replacement_comparison_is_invariant_under_extreme_objective_scaling() {
        // f0=10, f1=4, f2=2, largest=5 gives 8 < 320.
        // largest=1 gives 200 > 64, so that replacement must be rejected.
        for scale in [1e-300_f64, 1.0, 1e300] {
            assert!(
                replacement_test(
                    10.0 * scale,
                    4.0 * scale,
                    2.0 * scale,
                    Scaled::from_value(5.0 * scale).unwrap(),
                )
                .unwrap()
            );
            assert!(
                !replacement_test(
                    10.0 * scale,
                    4.0 * scale,
                    2.0 * scale,
                    Scaled::from_value(scale).unwrap(),
                )
                .unwrap()
            );
        }
        for scale in [1e-30_f32, 1.0, 1e30] {
            assert!(
                replacement_test(
                    10.0 * scale,
                    4.0 * scale,
                    2.0 * scale,
                    Scaled::from_value(5.0 * scale).unwrap(),
                )
                .unwrap()
            );
            assert!(
                !replacement_test(
                    10.0 * scale,
                    4.0 * scale,
                    2.0 * scale,
                    Scaled::from_value(scale).unwrap(),
                )
                .unwrap()
            );
        }
    }
    #[test]
    fn finite_interval_handles_opposite_extreme_bounds_and_tiny_directions() {
        let bounds = Bounds::new(alloc::vec![(Some(-f64::MAX), Some(f64::MAX))]).unwrap();
        let (lower, upper) = line_interval(&[f64::MAX], &[-1.0], &bounds).unwrap();
        assert_eq!(lower, 0.0);
        assert_eq!(upper, f64::MAX);
        let (_, upper) = line_interval(&[0.0], &[f64::MIN_POSITIVE], &bounds).unwrap();
        assert_eq!(upper, f64::MAX);
    }
    #[test]
    fn bounded_line_finds_the_decrease_next_to_the_current_point() {
        // Himmelblau's function; on the line x = 3.5441 the whole-interval
        // search of y ∈ [−5, 5] settles in a basin above the current value,
        // while y ≈ −1.85 is lower.
        let objective = crate::ObjectiveFn::new(2, |x: &[f64]| {
            Ok((x[0] * x[0] + x[1] - 11.0).powi(2) + (x[0] + x[1] * x[1] - 7.0).powi(2))
        });
        let bounds = Bounds::new(alloc::vec![(Some(-5.0), Some(5.0)); 2]).unwrap();
        let problem = crate::BoundedObjective::new(&objective, &bounds).unwrap();
        let result = Powell::default()
            .minimize(&problem, &[-3.988575, -1.318513])
            .unwrap();
        assert_eq!(result.termination(), OptimizeTermination::Converged);
        assert!(result.fun() < 1e-10);
        assert!((result.x()[0] - 3.584428).abs() < 1e-5);
        assert!((result.x()[1] + 1.848126).abs() < 1e-5);
    }
}
