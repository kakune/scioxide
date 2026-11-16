//! The continuation method value and its `advance`, `follow` and `resume` calls.
use super::*;
use crate::NewtonKrylovWorkspaceRequirements;
use crate::attempt::Entry;
use scioxide_core::{
    AttemptKind, AttemptRecord, Capacity, ExecutionContext, PartialFailure, SolveCounts, SolveScope,
};

/// Follows a solution curve of `F(x, λ) = 0` by predictor–corrector continuation.
///
/// A `Continuation` is a method value: it holds a validated
/// [`ContinuationPath`] and [`ContinuationOptions`], and construction only
/// validates them. The current point, tangent and history of accepted points
/// live in a [`ContinuationState`], so one value can follow several paths.
/// Create the state with [`Self::state`] and pass it to [`Self::follow`], which
/// follows the path to its end. To observe each point, stop early or run in a
/// caller's execution context, create the workspace with [`Self::workspace`]
/// and call [`Self::advance`] while the report's termination is
/// [`RootTermination::Advanced`].
///
/// The first call corrects the starting point with `λ` fixed and computes the
/// initial tangent. Each later call accepts at most one new point: it predicts
/// along the unit tangent (Euler predictor), corrects with a Newton–Krylov solve
/// of the bordered system, evaluates `F` again at the corrected point to check
/// the residual, applies the branch guards and computes the new tangent. Each
/// trial evaluates `F` at the predicted and corrected points and runs one
/// corrector solve. A trial that passes the residual test also costs two
/// Jacobian–vector products for the Jacobian-change guard and one tangent solve,
/// plus the optional diagnostic.
///
/// The path ends with one of these terminations:
///
/// - `Complete`: every listed parameter, or the full arclength, was reached.
/// - `HistoryLimit`: the state holds `max_points` points.
/// - `StepLimit`: the trials for one point were rejected down to the minimum
///   step or the rejection limit.
/// - `TangentFailure`: the tangent could not be computed to its tolerance, or
///   a natural path reached a fold, where the parameter component of the unit
///   tangent is at most `ε` in magnitude.
/// - `WorkLimit`, `Cancelled` and `TimeLimit`: a work limit, cancellation or
///   deadline stopped the call. [`Self::resume`] can continue from a checkpoint.
///
/// See [Termination](crate#termination) for the shared statuses.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, SciOxideError};
/// use scioxide_rootfind::{Continuation, ContinuationOptions, ContinuationPath};
/// use scioxide_rootfind::{ContinuationProblem, ResidualFn};
///
/// // F(x, λ) = x − λ: the solution curve is the line x = λ.
/// let line = ResidualFn::with_parameter(1, |y: &[f64], out: &mut [f64]| {
///     out[0] = y[0] - y[1];
///     Ok(())
/// })
/// .with_jvp(|_: &[f64], v: &[f64], out: &mut [f64]| {
///     out[0] = v[0] - v[1];
///     Ok(())
/// });
/// let problem = ContinuationProblem::new(&line)?;
/// let options = ContinuationOptions::new(0.1, 1e-6, 0.2, Budget::try_from(32)?)?;
/// let continuation = Continuation::new(ContinuationPath::Parameters(vec![1.0]), options)?;
/// let path = continuation.follow(&problem, continuation.state(&problem, &[0.0, 0.0])?)?;
/// // On the exact curve, λ = 1 gives x = 1.
/// assert!(path.is_success());
/// assert!((path.point()[0] - 1.0).abs() < 1e-8);
/// # Ok::<(), SciOxideError>(())
/// ```
///
/// # References
///
/// - E. L. Allgower and K. Georg, *Introduction to Numerical Continuation
///   Methods*, SIAM, 2003.
/// - H. B. Keller, "Numerical solution of bifurcation and nonlinear eigenvalue
///   problems", in *Applications of Bifurcation Theory*, Academic Press, 1977.
#[derive(Clone, Debug)]
pub struct Continuation<R: RealScalar> {
    path: ContinuationPath<R>,
    options: ContinuationOptions<R>,
}
/// Reports the outcome of one call to [`Continuation::advance`] or
/// [`Continuation::resume`].
///
/// The accepted points themselves are in the [`ContinuationState`].
#[derive(Clone, Copy, Debug)]
pub struct ContinuationReport<R: RealScalar> {
    termination: RootTermination,
    counts: SolveCounts,
    observation: Option<ContinuationObservation<R>>,
}
impl<R: RealScalar> ContinuationReport<R> {
    /// Returns the termination status of the path after this call.
    pub const fn termination(&self) -> RootTermination {
        self.termination
    }
    /// Returns whether the whole requested path has been followed (`Complete`).
    pub fn is_success(&self) -> bool {
        self.termination == RootTermination::Complete
    }
    /// Returns the work done by continuation itself, summed over all calls on
    /// this path, including the calls before a checkpoint.
    pub const fn counts(&self) -> EvaluationCounts {
        self.counts.own()
    }
    /// Returns the number of trial steps on this path, including rejected ones.
    pub const fn iterations(&self) -> usize {
        self.counts.own().iterations()
    }
    /// Returns `‖F(y)‖₂` at the point accepted by this call, if any.
    pub fn residual_norm(&self) -> Option<R> {
        self.observation.map(|o| o.residual_norm())
    }
    /// Returns the work of continuation and of its nested corrector and tangent
    /// solves, summed over all calls on this path.
    pub const fn accounting(&self) -> SolveCounts {
        self.counts
    }
    /// Returns the point accepted by this call, if any.
    pub const fn observation(&self) -> Option<ContinuationObservation<R>> {
        self.observation
    }
}

impl<R: RealScalar> Continuation<R> {
    /// Validates `path` and `options` and stores them; no numerical work is done.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the parameter list is empty or the arclength is not
    ///   positive and finite.
    /// - `NonFinite` if a listed parameter is NaN or infinite.
    /// - `InvalidInput` if a step size, scale, residual scale or rank threshold
    ///   is not positive and finite, the step sizes are out of order,
    ///   `max_points < 2`, a branch-guard limit is outside `(0, 1)`, or the
    ///   corrector options are invalid.
    pub fn new(
        path: ContinuationPath<R>,
        options: ContinuationOptions<R>,
    ) -> Result<Self, SciOxideError> {
        match &path {
            ContinuationPath::Parameters(values) => {
                if values.is_empty() {
                    return Err(invalid(
                        scioxide_core::argument!("path"),
                        InvalidInputReason::Inconsistent,
                        "empty parameter path",
                    ));
                }
                finite(values, NonFiniteOrigin::Input)?;
            }
            ContinuationPath::PseudoArclength(distance)
                if !distance.is_finite() || *distance <= R::ZERO =>
            {
                return Err(invalid(
                    scioxide_core::argument!("path"),
                    InvalidInputReason::Inconsistent,
                    "arclength must be positive and finite",
                ));
            }
            _ => {}
        }
        if [
            options.initial_step,
            options.minimum_step,
            options.maximum_step,
            options.state_scale,
            options.parameter_scale,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= R::ZERO)
            || options.minimum_step > options.initial_step
            || options.initial_step > options.maximum_step
            || options.max_points.get() < 2
            || options
                .residual_scale
                .is_some_and(|v| !v.is_finite() || v <= R::ZERO)
            || (options.dense_diagnostics
                && (!options.rank_tolerance.is_finite() || options.rank_tolerance <= R::ZERO))
        {
            return Err(invalid(
                scioxide_core::argument!("options"),
                InvalidInputReason::Inconsistent,
                "invalid continuation step, scaling, history or diagnostic controls",
            ));
        }
        if [
            options.minimum_tangent_cosine,
            options.maximum_correction_ratio,
            options.maximum_correction_contraction,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= R::ZERO || *v >= R::ONE)
        {
            return Err(invalid(
                scioxide_core::argument!("options"),
                InvalidInputReason::Inconsistent,
                "tangent cosine, correction ratio and contraction must lie in (0, 1)",
            ));
        }
        let _ = NewtonKrylovSolver::<R>::new(options.corrector.clone())?;
        Ok(Self { path, options })
    }
    /// Returns the requested path.
    pub const fn path(&self) -> &ContinuationPath<R> {
        &self.path
    }
    /// Returns the validated options.
    pub const fn options(&self) -> &ContinuationOptions<R> {
        &self.options
    }
    /// Creates the state of a new path starting from `seed = (x₀, λ₀)`.
    ///
    /// The state reserves room for `max_points` points. The seed need not solve
    /// the equation: the first [`Self::advance`] corrects it. This call does not
    /// evaluate `F`. The initial tangent orientation is `+λ`; use
    /// [`ContinuationState::with_initial_tangent`] to choose another.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `seed.len() != n + 1`, or if the problem has an
    ///   injected diagnostic and the options request the dense one.
    /// - `NonFinite` if `seed` contains NaN or infinity.
    /// - `SizeOverflow` or `AllocationFailure` if the history cannot be reserved.
    pub fn state(
        &self,
        problem: &ContinuationProblem<'_, R>,
        seed: &[R],
    ) -> Result<ContinuationState<R>, SciOxideError> {
        let path = match &self.path {
            ContinuationPath::Parameters(v) => {
                ContinuationPath::Parameters(try_copy_slice(v, scioxide_core::quantity!("path"))?)
            }
            ContinuationPath::PseudoArclength(v) => ContinuationPath::PseudoArclength(*v),
        };
        ContinuationState::new(problem, path, seed, self.options.clone())
    }
    /// Returns the workspace size needed for `problem`: corrector, tangent solve
    /// and optional dense diagnostic.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the problem has an injected diagnostic and the
    ///   options request the dense one.
    /// - `SizeOverflow` if the size overflows `usize`.
    /// - `ResourceLimit` if it exceeds
    ///   [`ContinuationOptions::with_workspace_byte_limit`].
    /// - Any error from the provider's workspace-length queries.
    pub fn workspace_requirements(
        &self,
        problem: &ContinuationProblem<'_, R>,
    ) -> Result<ContinuationWorkspaceRequirements<R>, SciOxideError> {
        use scioxide_core::backend::{checked_add, checked_mul};
        let shape = provider_shape(problem)?;
        if shape.diagnostic && self.options.dense_diagnostics {
            return Err(invalid(
                scioxide_core::argument!("problem"),
                InvalidInputReason::Inconsistent,
                "select dense or injected diagnostics",
            ));
        }
        let q = scioxide_core::quantity!("continuation_workspace");
        let m = checked_add(shape.dimension, 1, q)?;
        let corrector = NewtonKrylovWorkspaceRequirements::augmented(
            m,
            shape.residual,
            shape.jvp,
            shape.linear,
            &self.options.corrector,
        )?;
        let tangent = if shape.linear {
            None
        } else {
            let operator = crate::residual::LinearShape::<R>::new(m, shape.jvp);
            Some(
                tangent_solver(&self.options)
                    .workspace_requirements(&IterativeWorkspaceRequest::operator(&operator)?)?,
            )
        };
        let mut bytes = checked_add(
            checked_mul(
                checked_add(
                    checked_add(checked_mul(m, 10, q)?, m - 1, q)?,
                    shape.residual.max(shape.jvp),
                    q,
                )?,
                core::mem::size_of::<R>(),
                q,
            )?,
            corrector.required_bytes(),
            q,
        )?;
        bytes = checked_add(
            bytes,
            NewtonKrylovResult::<R>::required_bytes(m, &self.options.corrector)?,
            q,
        )?;
        bytes = checked_add(bytes, tangent.as_ref().map_or(0, |r| r.required_bytes()), q)?;
        if self.options.dense_diagnostics {
            bytes = checked_add(
                bytes,
                diagnostics::DenseDiagnostic::<R>::required_bytes(m)?,
                q,
            )?;
        }
        if bytes > self.options.byte_limit {
            return Err(SciOxideError::resource_limit(
                q,
                bytes,
                self.options.byte_limit,
            ));
        }
        Ok(ContinuationWorkspaceRequirements {
            shape,
            corrector,
            corrector_options: self.options.corrector.clone(),
            tangent,
            dense: self.options.dense_diagnostics,
            bytes,
            limit: self.options.byte_limit,
        })
    }
    /// Allocates a workspace for `problem`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace(
        &self,
        problem: &ContinuationProblem<'_, R>,
    ) -> Result<ContinuationWorkspace<R>, SciOxideError> {
        ContinuationWorkspace::allocate(&self.workspace_requirements(problem)?)
    }
    /// Advances the path by at most one accepted point.
    ///
    /// The first call corrects the seed and accepts it as the first point. Each
    /// accepted point is passed to `sink` together with its
    /// [`ContinuationObservation`]; `|_, _| Ok(())` ignores them. Repeat the call
    /// while the report's termination is [`RootTermination::Advanced`].
    ///
    /// Points accepted before an error stay in `state`, and an error does not
    /// undo calls already made to `sink`. If the call fails its input checks,
    /// `state` is unchanged and no work is counted. The workspace records the
    /// attempt in `last_attempt()`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `state` was created by a different continuation or
    ///   problem shape, or `workspace` is too small for `problem`.
    /// - `ConvergenceFailure` if the first call cannot correct the seed to the
    ///   residual tolerance or compute its tangent.
    /// - `ConvergenceFailure` or `ExecutionStopped` if a limit, cancellation or
    ///   deadline stops the first call before the seed is accepted.
    /// - `NonFinite` if `F` or a Jacobian–vector product returns NaN or infinity.
    /// - Errors returned by the problem's callbacks or by `sink`, unchanged.
    pub fn advance<'scope, 'capability>(
        &self,
        problem: &ContinuationProblem<'_, R>,
        state: &mut ContinuationState<R>,
        mut sink: impl FnMut(&[R], &ContinuationObservation<R>) -> Result<(), SciOxideError>,
        workspace: &mut ContinuationWorkspace<R>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<ContinuationReport<R>, SciOxideError> {
        self.drive(problem, state, &mut sink, workspace, scope.into(), None)
    }
    /// Follows the path from `state` to its end and returns the final state.
    ///
    /// This is [`Self::advance`] with a new workspace from [`Self::workspace`],
    /// no sink and [`ExecutionContext::sequential_native`], repeated while the
    /// termination is [`RootTermination::Advanced`]. The repetition is finite
    /// because a state that holds `max_points` points ends with `HistoryLimit`.
    /// An unsuccessful termination (`StepLimit`, `TangentFailure`,
    /// `HistoryLimit`, `WorkLimit`) is not an error; read it with
    /// [`ContinuationState::termination`] and [`ContinuationState::is_success`].
    /// Use [`Self::advance`] to observe each point, stop early or charge a
    /// caller's solve context.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`] and [`Self::advance`]. When the state
    /// holds at least one accepted point, the [`PartialFailure`] carries it as
    /// progress, with every point accepted before the error; otherwise it
    /// carries none. `?` converts the failure to its [`SciOxideError`].
    ///
    /// # Examples
    ///
    /// See [`Continuation`].
    #[allow(
        clippy::result_large_err,
        reason = "owned conveniences return progress inside PartialFailure"
    )]
    pub fn follow(
        &self,
        problem: &ContinuationProblem<'_, R>,
        mut state: ContinuationState<R>,
    ) -> Result<ContinuationState<R>, PartialFailure<ContinuationState<R>>> {
        fn failure<R: RealScalar>(
            error: SciOxideError,
            state: ContinuationState<R>,
        ) -> PartialFailure<ContinuationState<R>> {
            if state.history().is_empty() {
                PartialFailure::new(error)
            } else {
                PartialFailure::with_progress(error, || Ok(state))
            }
        }
        let mut workspace = match self.workspace(problem) {
            Ok(workspace) => workspace,
            Err(error) => return Err(failure(error, state)),
        };
        loop {
            match self.advance(
                problem,
                &mut state,
                |_, _| Ok(()),
                &mut workspace,
                ExecutionContext::sequential_native(),
            ) {
                Ok(report) if report.termination() == RootTermination::Advanced => {}
                Ok(_) => return Ok(state),
                Err(error) => return Err(failure(error, state)),
            }
        }
    }
    /// Restores `state` from `checkpoint`, then advances by at most one point.
    ///
    /// Use this to continue a path after a limit, cancellation or deadline, or to
    /// go back to an earlier point. Work recorded in the checkpoint still counts
    /// toward the path's limits from [`ContinuationOptions::with_budgets`]; the
    /// execution context and any parent limits come from `scope`. If the
    /// checkpoint does not match this continuation, neither it nor `state`
    /// changes.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the checkpoint or `state` belongs to a different
    ///   continuation or problem shape, or `state` has less history capacity
    ///   than the checkpoint holds.
    /// - The errors of [`Self::advance`].
    pub fn resume<'scope, 'capability>(
        &self,
        problem: &ContinuationProblem<'_, R>,
        checkpoint: &ContinuationCheckpoint<R>,
        state: &mut ContinuationState<R>,
        mut sink: impl FnMut(&[R], &ContinuationObservation<R>) -> Result<(), SciOxideError>,
        workspace: &mut ContinuationWorkspace<R>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<ContinuationReport<R>, SciOxideError> {
        self.drive(
            problem,
            state,
            &mut sink,
            workspace,
            scope.into(),
            Some(checkpoint),
        )
    }
    fn compatible(
        &self,
        state: &ContinuationState<R>,
        shape: ProviderShape,
    ) -> Result<(), SciOxideError> {
        if self.path != state.path
            || self.options != state.options
            || shape != state.provider_shape
            || state.point.len() != shape.dimension + 1
        {
            return Err(invalid(
                scioxide_core::argument!("state"),
                InvalidInputReason::Inconsistent,
                "state or checkpoint configuration differs from this continuation and problem shape",
            ));
        }
        Ok(())
    }
    fn drive(
        &self,
        problem: &ContinuationProblem<'_, R>,
        state: &mut ContinuationState<R>,
        sink: &mut impl FnMut(&[R], &ContinuationObservation<R>) -> Result<(), SciOxideError>,
        workspace: &mut ContinuationWorkspace<R>,
        scope: SolveScope<'_, '_>,
        checkpoint: Option<&ContinuationCheckpoint<R>>,
    ) -> Result<ContinuationReport<R>, SciOxideError> {
        let kind = if checkpoint.is_some() {
            AttemptKind::Resume
        } else {
            AttemptKind::Call
        };
        // This attempt's accepted observation, kept even if a later provider
        // or sink failure ends the epoch.
        let accepted = core::cell::Cell::new(None);
        let result = crate::attempt::attempt(
            workspace,
            state,
            scope,
            kind,
            |workspace, state| self.entry(problem, state, workspace, checkpoint),
            |workspace, state, context| {
                self.epoch(
                    problem, state, sink, workspace, checkpoint, context, &accepted,
                )
            },
            |_, state, _| crate::attempt::Conclusion {
                succeeded: state.is_success() || state.status == RootTermination::Advanced,
                progress: accepted.get(),
            },
        );
        result?;
        Ok(ContinuationReport {
            termination: state.status,
            counts: workspace
                .last_attempt()
                .map_or_else(SolveCounts::new, AttemptRecord::counts),
            observation: accepted.get(),
        })
    }
    /// Checks the workspace, state and checkpoint, and selects how the solve starts.
    fn entry(
        &self,
        problem: &ContinuationProblem<'_, R>,
        state: &ContinuationState<R>,
        workspace: &ContinuationWorkspace<R>,
        checkpoint: Option<&ContinuationCheckpoint<R>>,
    ) -> Result<Entry, SciOxideError> {
        let requirements = self.workspace_requirements(problem)?;
        self.compatible(state, requirements.shape)?;
        if workspace.check(&requirements)? != Capacity::Sufficient {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                InvalidInputReason::Inconsistent,
                "reserve matching continuation storage",
            ));
        }
        let source = checkpoint.map_or(state, |c| &c.state);
        self.compatible(source, requirements.shape)?;
        if let Some(checkpoint) = checkpoint {
            checkpoint.validate()?;
            if state.points.capacity() < source.points.len()
                || state.history.capacity() < source.history.len()
            {
                return Err(invalid(
                    scioxide_core::argument!("state"),
                    InvalidInputReason::Inconsistent,
                    "prepare state history capacity before resuming",
                ));
            }
        }
        let identity = source.identity();
        let budgets = self.options.budgets;
        Ok(match checkpoint.map(|c| c.metadata).or(state.metadata) {
            Some(metadata) => Entry::Resume {
                identity,
                state_token: source.state_token(),
                metadata,
                budgets,
            },
            None => Entry::call(identity, budgets),
        })
    }
    /// Starts the path or advances it by one accepted point.
    ///
    /// A newly accepted observation is stored in `accepted` before any later
    /// failure, then passed to `sink`.
    #[expect(
        clippy::too_many_arguments,
        reason = "One epoch borrows the problem, caller state, sink, storage and context."
    )]
    fn epoch(
        &self,
        problem: &ContinuationProblem<'_, R>,
        state: &mut ContinuationState<R>,
        sink: &mut impl FnMut(&[R], &ContinuationObservation<R>) -> Result<(), SciOxideError>,
        workspace: &mut ContinuationWorkspace<R>,
        checkpoint: Option<&ContinuationCheckpoint<R>>,
        context: &mut SolveContext<'_, '_>,
        accepted: &core::cell::Cell<Option<ContinuationObservation<R>>>,
    ) -> Result<(), SciOxideError> {
        if let Some(checkpoint) = checkpoint {
            state.restore(&checkpoint.state);
        }
        // Execution capabilities and ancestor allowances are supplied anew.
        // Local consumption still comes from the validated checkpoint.
        if matches!(
            state.status,
            RootTermination::IterationLimit
                | RootTermination::WorkLimit(_)
                | RootTermination::Cancelled
                | RootTermination::TimeLimit
        ) {
            state.status = RootTermination::Advanced;
        }
        let before = state.history.len();
        let outcome = if state.metadata.is_none() {
            state
                .start(problem, workspace, context)
                .map(|()| state.status)
        } else {
            state.step(problem, workspace, context)
        };
        if state.metadata.is_some() {
            workspace.accepted.copy_from_slice(&state.point);
            state.metadata = Some(context.checkpoint(state.state_token())?);
        }
        if state.history.len() > before {
            accepted.set(state.history.last().copied());
        }
        outcome?;
        match accepted.get() {
            Some(observation) => sink(&state.point, &observation),
            None => Ok(()),
        }
    }
}

impl<R: RealScalar> ContinuationState<R> {
    fn restore(&mut self, source: &Self) {
        self.point.copy_from_slice(&source.point);
        self.residual.copy_from_slice(&source.residual);
        self.report = source.report;
        self.tangent.copy_from_slice(&source.tangent);
        self.points.clear();
        self.points.extend_from_slice(&source.points);
        self.history.clear();
        self.history.extend_from_slice(&source.history);
        self.history_token = source.history_token;
        self.next_target = source.next_target;
        self.distance = source.distance;
        self.next_step = source.next_step;
        self.status = source.status;
        self.metadata = source.metadata;
    }
    /// Creates the state of a new path from a strided one-dimensional seed.
    ///
    /// This is [`Continuation::state`] for a seed given as an [`NdView`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `seed` is not one-dimensional.
    /// - The errors of [`Continuation::state`].
    pub fn from_view(
        continuation: &Continuation<R>,
        problem: &ContinuationProblem<'_, R>,
        seed: NdView<'_, '_, R>,
    ) -> Result<Self, SciOxideError> {
        if seed.layout().shape().rank() != 1 {
            return Err(invalid(
                scioxide_core::argument!("seed"),
                InvalidInputReason::Inconsistent,
                "seed must have rank one",
            ));
        }
        let mut packed = scioxide_core::backend::try_with_capacity(
            seed.iter().len(),
            scioxide_core::quantity!("seed"),
        )?;
        packed.extend(seed.iter().copied());
        continuation.state(problem, &packed)
    }
}
