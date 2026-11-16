//! Bounded operator actions, norm estimation and partial spectral computations.
mod workspace;
use crate::kernel::finite;
use crate::workspace::accounting::{
    PlannedStorage, StorageBytes, StoragePlan, bytes, product, resize, sum,
};
use workspace::OneNormLayout;
pub use workspace::{
    ExponentialActionRequirements, FunctionActionRequirements, LobpcgRequirements,
    OneNormRequirements, PartialSpectrumRequirements, PartialSvdRequirements,
    RandomizedLowRankRequirements,
};
mod exponential;
mod function;
mod lobpcg;
mod randomized;
mod spectrum;
mod svd;
use crate::workspace::SolveSlot;
use alloc::vec::Vec;
pub use exponential::*;
pub use function::*;
pub use lobpcg::*;
pub use randomized::*;
use scioxide_core::ScaledValue;
use scioxide_core::backend::checked_add;
use scioxide_core::{
    AdjointOperator, AttemptOutcome, AttemptRecord, Budget, EvaluationRole, ExecutionContext,
    LinearOperator, NumericScalar, ProblemToken, RealScalar, SciOxideError, SolveContext,
    SolveIdentity,
};
pub use spectrum::*;
pub use svd::*;
/// Termination status of operator actions, one-norm estimates and partial spectral solvers.
///
/// Only `Converged` means that every requested result passed its accuracy check.
/// The other variants come with an `Ok` result that holds the accepted partial
/// state, as described in [Iterative solves](crate#iterative-solves). Each report
/// type states what that partial state contains. Invalid input and numerical
/// failure are returned as `Err`.
///
/// # Examples
///
/// A budget of one operator application stops the one-norm estimator early:
///
/// ```
/// use scioxide_core::{Budget, ExecutionContext};
/// use scioxide_linalg::{Matrix, OneNormEstimator, SpectralTermination};
/// let a = Matrix::<f64>::identity(2)?;
/// let a = a.view().try_finite()?;
/// let method = OneNormEstimator::new().with_applications(Budget::ONE);
/// let mut workspace = method.workspace(&a)?;
/// let report = method.estimate_scoped(&a, &mut workspace, ExecutionContext::sequential_native())?;
/// assert_eq!(report.termination(), SpectralTermination::EvaluationLimit);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralTermination {
    /// Every requested result passed its residual or accuracy check.
    Converged,
    /// The operator-application limit of the options, or the evaluation limit of
    /// an enclosing solve, was reached.
    EvaluationLimit,
    /// An iteration, restart-cycle, Taylor-degree or projection-size limit was
    /// reached before convergence.
    IterationLimit,
    /// The Krylov subspace became invariant, or reached full dimension, before
    /// every requested pair passed its residual check.
    InvariantSubspace,
    /// A complete approximation was computed but failed its accuracy check.
    AccuracyLimit,
    /// Cancellation or a deadline stopped the computation.
    ExecutionStopped(scioxide_core::ExecutionStopReason),
}

/// Classify cooperative execution and budget stops consistently in every phase.
/// Callers write their accepted state after this classification succeeds.
fn action_termination(
    outcome: Result<(), SciOxideError>,
    termination: SpectralTermination,
) -> Result<SpectralTermination, SciOxideError> {
    match outcome {
        Ok(()) => Ok(termination),
        Err(error) => match error.view() {
            scioxide_core::SciOxideErrorView::ExecutionStopped { reason, .. } => {
                Ok(SpectralTermination::ExecutionStopped(reason))
            }
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            } => Ok(SpectralTermination::IterationLimit),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::EvaluationLimit,
                ..
            } => Ok(SpectralTermination::EvaluationLimit),
            _ => Err(error),
        },
    }
}
/// Validate room for the kept targets and two fresh restart directions.
fn validate_restart_capacity(
    dimension: usize,
    count: usize,
    basis: Budget,
) -> Result<(), SciOxideError> {
    let available = basis.get().min(dimension);
    if count > dimension {
        return Err(crate::checked::out_of_range(
            scioxide_core::argument!("self.count"),
            "target count exceeds dimension",
        ));
    }
    if available < count || (available < dimension && available - count < 2) {
        return Err(crate::checked::inconsistent(
            scioxide_core::argument!("self.basis"),
            "restart needs count+2 capacity or a complete basis",
        ));
    }
    Ok(())
}

/// Structure of an operator that [`PartialSpectrumSolver`] may exploit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OperatorStructure {
    /// No structure is assumed; eigenvalues may be complex.
    #[default]
    General,
    /// The operator is Hermitian (real symmetric for real scalars), so its
    /// eigenvalues are real.
    Hermitian,
}

/// Preserve action controls while giving projected dense work an independent
/// allowance. Dense APIs create child scopes, so sweeps appear only as nested
/// iterations of the action. `Budget` is finite; its maximum represents no
/// practical inner cap without changing the public execution-options contract.
fn projected_execution_options(
    options: crate::LinalgExecutionOptions,
) -> crate::LinalgExecutionOptions {
    options.with_iteration_budget(Budget::ONE.saturating_add(usize::MAX))
}

/// Provider budget checks, local application limit and execution checks of one action.
pub(crate) struct Work<'s, 'a, 'b> {
    execution: ExecutionContext<'b>,
    kernels: crate::KernelContext<'b>,
    solve: &'s mut SolveContext<'a, 'b>,
    limit: usize,
    count: usize,
}
impl Work<'_, '_, '_> {
    /// Charge one action iteration. Every iteration applies the operator, so an
    /// exhausted application allowance returns `false` before charging a step
    /// that could not run; the caller then stops with `EvaluationLimit`.
    fn iteration(&mut self) -> Result<bool, SciOxideError> {
        self.execution.check()?;
        if self.count >= self.limit {
            return Ok(false);
        }
        if !self.solve.record_iteration()?.is_granted() {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::IterationLimit,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(self.solve.counts()),
                "action iteration budget exhausted",
            ));
        }
        Ok(true)
    }
    fn adjoint<S: NumericScalar>(
        &mut self,
        op: &dyn AdjointOperator<S>,
        x: &[S],
        y: &mut [S],
        scratch: &mut [S],
    ) -> Result<bool, SciOxideError> {
        self.execution.check()?;
        if self.count >= self.limit
            || !self
                .solve
                .record_provider(EvaluationRole::LinearOperator)?
                .is_granted()
        {
            return Ok(false);
        }
        self.count += 1;
        op.apply_adjoint_finite_into_scoped(
            x,
            finite::maximum(x, self.kernels.dispatch())?,
            y,
            scratch,
            &mut scioxide_core::OperatorContext::with_kernels(self.solve, &self.kernels),
        )
        .and_then(scioxide_core::Invocation::require)?;
        crate::checked::provider_output(y)?;
        Ok(true)
    }
    fn apply<S: NumericScalar>(
        &mut self,
        op: &dyn LinearOperator<S>,
        x: &[S],
        y: &mut [S],
        scratch: &mut [S],
    ) -> Result<bool, SciOxideError> {
        self.execution.check()?;
        if self.count >= self.limit
            || !self
                .solve
                .record_provider(EvaluationRole::LinearOperator)?
                .is_granted()
        {
            return Ok(false);
        }
        self.count += 1;
        op.apply_finite_into_scoped(
            x,
            finite::maximum(x, self.kernels.dispatch())?,
            y,
            scratch,
            &mut scioxide_core::OperatorContext::with_kernels(self.solve, &self.kernels),
        )
        .and_then(scioxide_core::Invocation::require)?;
        crate::checked::provider_output(y)?;
        Ok(true)
    }
    fn precondition<S: NumericScalar>(
        &mut self,
        preconditioner: &dyn scioxide_core::Preconditioner<S>,
        x: &[S],
        y: &mut [S],
        scratch: &mut [S],
    ) -> Result<bool, SciOxideError> {
        self.execution.check()?;
        if self.count >= self.limit
            || !self
                .solve
                .record_provider(EvaluationRole::Preconditioner)?
                .is_granted()
        {
            return Ok(false);
        }
        self.count += 1;
        preconditioner.precondition_into_scoped(
            x,
            y,
            scratch,
            &mut scioxide_core::OperatorContext::with_kernels(self.solve, &self.kernels),
        )?;
        crate::checked::provider_output(y)?;
        Ok(true)
    }
}
/// Workspaces that own the solve state and attempt record of one action method.
pub(crate) trait ActionSlot {
    fn slot(&mut self) -> &mut SolveSlot<()>;
}

/// Run an action in a fresh epoch of the workspace's solve state.
///
/// Entries mark the attempt rejected and validate every call argument before
/// this call, so preflight failures never start an epoch.
/// An execution context starts a root epoch and a solve context a child that
/// charges provider calls to its ancestors. The attempt is `Succeeded` when
/// `converged` holds, `Stopped` for other returned outcomes and `Failed` on
/// error; the local application `limit` comes from the method settings.
pub(crate) fn run_action<'c, W: ActionSlot, T>(
    workspace: &mut W,
    context: crate::LinalgExecutionContext<'_, 'c>,
    algorithm: scioxide_core::AlgorithmId,
    limit: Budget,
    options: &crate::LinalgExecutionOptions,
    run: impl FnOnce(&mut W, Work<'_, '_, 'c>) -> Result<T, SciOxideError>,
    converged: impl FnOnce(&T) -> bool,
) -> Result<T, SciOxideError> {
    let context = context.with_options(options)?;
    let kernels = context.kernel();
    let mut slot = core::mem::take(workspace.slot());
    let result = slot.run(
        context.into_scope(),
        SolveIdentity::new(ProblemToken::new(0), algorithm, 0, 0),
        options.solve_budgets(),
        |solve| {
            let execution = solve.execution();
            run(
                workspace,
                Work {
                    execution,
                    kernels,
                    solve,
                    limit: limit.get(),
                    count: 0,
                },
            )
        },
        |value| {
            let Ok(value) = value else {
                return (AttemptOutcome::Failed, None);
            };
            (
                if converged(value) {
                    AttemptOutcome::Succeeded
                } else {
                    AttemptOutcome::Stopped
                },
                None,
            )
        },
    );
    *workspace.slot() = slot;
    result
}

/// Fixed-shift composition `(A - shift M)^-1 M`; M defaults to identity.
/// The caller supplies the inverse operator, including its factorization/solve policy.
/// One counted application covers both the mass multiplication and inverse.
/// Providers charge only additional nested work, such as their own residual probes.
struct ShiftInvertOperator<'a, S: NumericScalar> {
    inverse: &'a dyn LinearOperator<S>,
    mass: Option<&'a dyn LinearOperator<S>>,
}
impl<'a, S: NumericScalar> ShiftInvertOperator<'a, S> {
    /// Compose a caller-declared shifted inverse and optional mass operator.
    /// # Errors
    /// Returns `DimensionMismatch` for a nonsquare inverse or a mass operator of
    /// another shape, or provider workspace errors.
    fn new(
        inverse: &'a dyn LinearOperator<S>,
        mass: Option<&'a dyn LinearOperator<S>>,
    ) -> Result<Self, SciOxideError> {
        crate::checked::square(inverse.rows(), inverse.cols())?;
        if let Some(m) = mass {
            crate::checked::expect_shape([inverse.rows(), inverse.cols()], [m.rows(), m.cols()])?;
        }
        let result = Self { inverse, mass };
        result.apply_workspace_len()?;
        Ok(result)
    }
}
impl<S: NumericScalar> ShiftInvertOperator<'_, S> {
    fn validate_storage(
        &self,
        input: &[S],
        output: &[S],
        scratch: &[S],
    ) -> Result<(), SciOxideError> {
        let n = self.inverse.rows();
        crate::checked::expect_shape([n], [input.len()])?;
        crate::checked::expect_shape([n], [output.len()])?;
        crate::checked::expect_shape([self.apply_workspace_len()?], [scratch.len()])
    }
}
impl<S: NumericScalar> LinearOperator<S> for ShiftInvertOperator<'_, S> {
    fn rows(&self) -> usize {
        self.inverse.rows()
    }
    fn cols(&self) -> usize {
        self.inverse.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        checked_add(
            checked_add(
                self.rows(),
                self.rows(),
                scioxide_core::quantity!("shift_invert_vectors"),
            )?,
            self.inverse
                .apply_workspace_len()?
                .max(self.mass.map_or(Ok(0), |m| m.apply_workspace_len())?),
            scioxide_core::quantity!("apply_workspace"),
        )
    }
    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        // `y = (A - sigma M)^{-1} M x`, with the caller-supplied inverse. The
        // scratch is split into the right-hand side `M x`, a staging buffer
        // for the solve (so a failing solve cannot leave `output` half
        // written), and the providers' own scratch, which they use in turn.
        self.validate_storage(input, output, scratch)?;
        let n = self.rows();
        let (rhs, rest) = crate::workspace::split_mut(scratch, n)?;
        let (candidate, provider) = crate::workspace::split_mut(rest, n)?;
        if let Some(m) = self.mass {
            m.apply_into(input, rhs, &mut provider[..m.apply_workspace_len()?])?;
        } else {
            rhs.copy_from_slice(input);
        }
        crate::checked::provider_output(rhs)?;
        self.inverse.apply_into(
            rhs,
            candidate,
            &mut provider[..self.inverse.apply_workspace_len()?],
        )?;
        crate::checked::provider_output(candidate)?;
        output.copy_from_slice(candidate);
        Ok(())
    }
    fn apply_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.validate_storage(input, output, scratch)?;
        let n = self.rows();
        let (rhs, rest) = crate::workspace::split_mut(scratch, n)?;
        let (candidate, provider) = crate::workspace::split_mut(rest, n)?;
        if let Some(m) = self.mass {
            let outcome = m.apply_into_scoped(
                input,
                rhs,
                &mut provider[..m.apply_workspace_len()?],
                context,
            )?;
            if !outcome.is_evaluated() {
                return Ok(outcome);
            }
        } else {
            rhs.copy_from_slice(input);
        }
        crate::checked::provider_output(rhs)?;
        let outcome = self.inverse.apply_into_scoped(
            rhs,
            candidate,
            &mut provider[..self.inverse.apply_workspace_len()?],
            context,
        )?;
        if !outcome.is_evaluated() {
            return Ok(outcome);
        }
        crate::checked::provider_output(candidate)?;
        output.copy_from_slice(candidate);
        Ok(scioxide_core::Invocation::Evaluated(()))
    }
}
/// Estimator of the one-norm `‖A‖₁ = maxⱼ Σᵢ |aᵢⱼ|` of a square operator.
///
/// The operator is used only through products `A x` and `Aᴴ x`, so it can be
/// matrix-free. For `n ≤ 4` the estimator applies `A` to every unit vector and
/// returns the exact norm. For larger `n` it runs the Hager–Higham iteration,
/// at most eight steps of one product with `A` and one with `Aᴴ`, and then one
/// product with the alternating vector `xᵢ = (−1)ⁱ (1 + i/(n−1))`,
/// `i = 0, …, n−1`. The number of products is therefore at most 17, independent
/// of `n`. The estimate is `‖A x‖₁` for some `x` with `‖x‖₁ = 1`, so apart from
/// rounding it is a lower bound on `‖A‖₁`.
///
/// The value holds only the settings; the order `n` is read from the operator
/// of each call. The default limit is 64 applications, counting forward
/// products and adjoint products together. Each application also counts as one
/// iteration toward the iteration limit of the execution options.
///
/// For an exact norm, including rectangular operators, use
/// [`crate::operator_norm`] with [`crate::MatrixNorm::One`], which applies the
/// operator to every unit vector.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{Matrix, OneNormEstimator};
/// let a = Matrix::from_rows(&[[1.0_f64, 2.0], [3.0, 4.0]])?;
/// let report = OneNormEstimator::new().estimate(&a.view().try_finite()?)?;
/// // The column sums are 4 and 6.
/// assert!((report.estimate() - 6.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - W. W. Hager, "Condition estimates", *SIAM J. Sci. Stat. Comput.* 5(2), 1984.
/// - N. J. Higham, "FORTRAN codes for estimating the one-norm of a real or
///   complex matrix, with applications to condition estimation",
///   *ACM Trans. Math. Softw.* 14(4), 1988.
/// - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
///   SIAM, 2002, Ch. 15.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OneNormEstimator {
    execution: crate::LinalgExecutionOptions,
    applications: Budget,
}
impl OneNormEstimator {
    /// Returns the execution options.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }

    /// Sets the execution options; see [Execution](crate#execution).
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Returns the default options: at most 64 applications and default execution options.
    pub const fn new() -> Self {
        Self {
            execution: crate::LinalgExecutionOptions::new(),
            applications: Budget::ONE.saturating_add(63),
        }
    }
    /// Sets the maximum number of forward plus adjoint operator applications.
    pub const fn with_applications(mut self, applications: Budget) -> Self {
        self.applications = applications;
        self
    }
    /// Returns the maximum number of forward plus adjoint operator applications.
    pub const fn applications(&self) -> Budget {
        self.applications
    }
}
impl Default for OneNormEstimator {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OneNormCall {
    n: usize,
    options: OneNormEstimator,
}
impl OneNormCall {
    fn workspace_requirements<S: NumericScalar>(
        &self,
        op: &dyn AdjointOperator<S>,
    ) -> Result<OneNormRequirements<S>, SciOxideError> {
        crate::checked::expect_shape([self.n, self.n], [op.rows(), op.cols()])?;
        let n = self.n;
        let provider = op
            .apply_workspace_len()?
            .max(op.apply_adjoint_workspace_len()?);
        let payload = sum([
            bytes::<S>(sum([product(&[2, n])?, provider])?)?,
            bytes::<ScaledValue<scioxide_core::Complex<S::Real>>>(n)?,
        ])?;
        OneNormRequirements::new(
            OneNormLayout { n, provider },
            self.options.execution,
            payload,
            Default::default(),
            Default::default(),
        )
    }
    fn workspace<S: NumericScalar>(
        &self,
        op: &dyn AdjointOperator<S>,
    ) -> Result<OneNormWorkspace<S>, SciOxideError> {
        let mut workspace = OneNormWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(op)?)?;
        Ok(workspace)
    }
    fn estimate<S: NumericScalar>(
        &self,
        op: &dyn AdjointOperator<S>,
    ) -> Result<OneNormReport<S::Real>, SciOxideError> {
        self.estimate_scoped(
            op,
            &mut self.workspace(op)?,
            ExecutionContext::sequential_native(),
        )
    }
    fn estimate_scoped<'s, 'c, S: NumericScalar>(
        &self,
        op: &dyn AdjointOperator<S>,
        workspace: &mut OneNormWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<OneNormReport<S::Real>, SciOxideError> {
        workspace.slot.reject();
        let requirements = self.workspace_requirements(op)?;
        workspace.prepare(&requirements)?;
        run_action(
            workspace,
            context.into(),
            scioxide_core::algorithm_id!("linear", "OneNormEstimate", 1),
            self.options.applications,
            &self.options.execution,
            |w, work| norm_run(op, w, work),
            |report| report.termination == SpectralTermination::Converged,
        )
    }
}
impl OneNormEstimator {
    /// Returns the per-call value for the square operator `op`.
    fn call<S: NumericScalar>(
        &self,
        op: &dyn AdjointOperator<S>,
    ) -> Result<OneNormCall, SciOxideError> {
        crate::checked::square(op.rows(), op.cols())?;
        Ok(OneNormCall {
            n: op.rows(),
            options: *self,
        })
    }
    /// Returns the workspace storage needed to estimate the norm of `op`.
    ///
    /// The operator is not applied; only its shape and scratch lengths are read.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `op` is not square.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the byte limit of the options.
    /// - Errors from the scratch-length queries of `op`.
    pub fn workspace_requirements<S: NumericScalar>(
        &self,
        op: &dyn AdjointOperator<S>,
    ) -> Result<OneNormRequirements<S>, SciOxideError> {
        self.call(op)?.workspace_requirements(op)
    }

    /// Allocates a workspace for `op`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`], and `AllocationFailure`.
    pub fn workspace<S: NumericScalar>(
        &self,
        op: &dyn AdjointOperator<S>,
    ) -> Result<OneNormWorkspace<S>, SciOxideError> {
        self.call(op)?.workspace(op)
    }

    /// Estimates `‖A‖₁` of `op` with a new workspace.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace`] and [`Self::estimate_scoped`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{Matrix, OneNormEstimator};
    /// let a = Matrix::<f64>::from_vec(2, 2, vec![1., 2., 3., 4.])?;
    /// let report = OneNormEstimator::new()
    ///     .estimate(&a.view().try_finite()?)?;
    /// // Column sums are 4 and 6.
    /// assert!((report.estimate() - 6.).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn estimate<S: NumericScalar>(
        &self,
        op: &dyn AdjointOperator<S>,
    ) -> Result<OneNormReport<S::Real>, SciOxideError> {
        self.call(op)?.estimate(op)
    }

    /// Estimates `‖A‖₁` of `op` using `workspace`.
    ///
    /// When a limit or cancellation stops the estimator, the report holds the
    /// largest estimate completed so far, or zero if no product completed.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// - `InvalidInput` (`Inconsistent`) if the scratch length of `op` changed
    ///   after the workspace was reserved.
    /// - `NonFinite` if `op` returns a non-finite value.
    /// - `NumericalInstability` if the estimate overflows the scalar type.
    /// - Errors returned by `op`.
    pub fn estimate_scoped<'s, 'c, S: NumericScalar>(
        &self,
        op: &dyn AdjointOperator<S>,
        workspace: &mut OneNormWorkspace<S>,
        context: impl Into<crate::LinalgExecutionContext<'s, 'c>>,
    ) -> Result<OneNormReport<S::Real>, SciOxideError> {
        self.call(op)?.estimate_scoped(op, workspace, context)
    }
}

/// Workspace of [`OneNormEstimator`].
#[derive(Debug)]
pub struct OneNormWorkspace<S: NumericScalar> {
    slot: SolveSlot<()>,
    n: usize,
    x: Vec<S>,
    y: Vec<S>,
    scratch: Vec<S>,
    scaled: Vec<ScaledValue<scioxide_core::Complex<S::Real>>>,
}
impl<S: NumericScalar> ActionSlot for OneNormWorkspace<S> {
    fn slot(&mut self) -> &mut SolveSlot<()> {
        &mut self.slot
    }
}
impl<S: NumericScalar> OneNormWorkspace<S> {
    /// Returns the attempt record of the most recent estimate, or `None` before the first.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.slot.last_attempt()
    }
}
/// Result of [`OneNormEstimator::estimate_scoped`]: the estimate, its status and the
/// number of operator applications.
///
/// After a limit or cancellation, the estimate is the largest one completed so
/// far, or zero if no product completed.
///
/// # Examples
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
#[derive(Debug, Clone, Copy)]
pub struct OneNormReport<R: RealScalar> {
    estimate: R,
    applications: usize,
    termination: SpectralTermination,
}
impl<R: RealScalar> OneNormReport<R> {
    /// Returns the termination status.
    pub const fn termination(&self) -> SpectralTermination {
        self.termination
    }

    /// Returns the estimate, a lower bound on `‖A‖₁` apart from rounding.
    ///
    /// When `termination()` is `Converged` and `n ≤ 4`, it is the exact norm.
    pub const fn estimate(&self) -> R {
        self.estimate
    }
    /// Returns the number of forward plus adjoint operator applications.
    pub const fn applications(&self) -> usize {
        self.applications
    }
}

fn norm_run<S: NumericScalar>(
    op: &dyn AdjointOperator<S>,
    w: &mut OneNormWorkspace<S>,
    mut work: Work<'_, '_, '_>,
) -> Result<OneNormReport<S::Real>, SciOxideError> {
    if op.rows() != w.n
        || op.cols() != w.n
        || w.scratch.len()
            < op.apply_workspace_len()?
                .max(op.apply_adjoint_workspace_len()?)
    {
        return Err(crate::checked::inconsistent(
            scioxide_core::argument!("op"),
            "operator changed its shape or scratch requirement after preparation",
        ));
    }
    // Reuses the Hager/Higham block-free 1-norm estimator from Schur
    // reordering; the iterate lives in extended range and is rounded to the
    // working scalar only at the operator boundary.
    let mut termination = SpectralTermination::Converged;
    let estimate = crate::estimate::operator_one_norm(&mut w.scaled, |values, adjoint| {
        for (x, v) in w.x.iter_mut().zip(values.iter()) {
            let z = v.round()?;
            *x = S::from_components(z.re, z.im);
        }
        let admitted = (|| {
            if !work.iteration()? {
                return Ok(false);
            }
            if adjoint {
                work.adjoint(
                    op,
                    &w.x,
                    &mut w.y,
                    &mut w.scratch[..op.apply_adjoint_workspace_len()?],
                )
            } else {
                work.apply(
                    op,
                    &w.x,
                    &mut w.y,
                    &mut w.scratch[..op.apply_workspace_len()?],
                )
            }
        })();
        let outcome = match admitted {
            Ok(true) => Ok(()),
            Ok(false) => {
                termination = SpectralTermination::EvaluationLimit;
                return Ok(false);
            }
            Err(error) => Err(error),
        };
        termination = action_termination(outcome, termination)?;
        if termination != SpectralTermination::Converged {
            return Ok(false);
        }
        for (v, &y) in values.iter_mut().zip(&w.y) {
            *v = {
                let component_value = y;
                scioxide_core::ScaledValue::from_value(scioxide_core::Complex::new(
                    component_value.real_component(),
                    component_value.imaginary_component(),
                ))
            }?;
        }
        Ok(true)
    })?;
    termination = action_termination(work.execution.check(), termination)?;
    Ok(OneNormReport {
        estimate: estimate.value.round()?,
        applications: work.count,
        termination,
    })
}

impl<S: NumericScalar> Default for OneNormWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> OneNormWorkspace<S> {
    /// Creates an empty workspace.
    pub fn new() -> Self {
        Self {
            slot: SolveSlot::new(),
            n: 0,
            x: Vec::new(),
            y: Vec::new(),
            scratch: Vec::new(),
            scaled: Vec::new(),
        }
    }
    /// Returns whether the workspace already holds the storage that `requirements` describes.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the byte limit of
    ///   `requirements`.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    pub fn check(
        &self,
        requirements: &OneNormRequirements<S>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        PlannedStorage::check(self, requirements)
    }
    /// Grows the workspace to the storage that `requirements` describes.
    ///
    /// Buffers that are already large enough keep their capacity, and the
    /// attempt record is kept. If growth fails, some buffers may already have
    /// grown; the workspace stays usable.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the grown workspace would exceed the byte limit of
    ///   `requirements`.
    /// - `AllocationFailure` if an allocation fails.
    /// - `SizeOverflow` if a storage size overflows `usize`.
    pub fn try_reserve(
        &mut self,
        requirements: &OneNormRequirements<S>,
    ) -> Result<(), SciOxideError> {
        PlannedStorage::reserve(self, requirements)
    }
}
impl<S: NumericScalar> PlannedStorage<S> for OneNormWorkspace<S> {
    type Requirements = OneNormRequirements<S>;
    fn storage_plan(
        &self,
        requirements: &OneNormRequirements<S>,
    ) -> Result<StoragePlan, SciOxideError> {
        let OneNormLayout { n, provider } = requirements.layout;

        let mut plan = StoragePlan::new(self.storage_bytes()?);
        plan.buffer(&self.x, n)?;
        plan.buffer(&self.y, n)?;
        plan.buffer(&self.scratch, provider)?;
        plan.buffer(&self.scaled, n)?;
        Ok(plan)
    }
    fn configure(&mut self, requirements: &OneNormRequirements<S>) -> Result<(), SciOxideError> {
        let OneNormLayout { n, provider } = requirements.layout;

        resize(&mut self.x, n, S::ZERO)?;
        resize(&mut self.y, n, S::ZERO)?;
        resize(&mut self.scratch, provider, S::ZERO)?;
        resize(&mut self.scaled, n, ScaledValue::ZERO)?;
        self.n = n;
        Ok(())
    }
}
impl<S: NumericScalar> StorageBytes for OneNormWorkspace<S> {
    fn storage_bytes(&self) -> core::result::Result<usize, SciOxideError> {
        sum([
            self.x.storage_bytes()?,
            self.y.storage_bytes()?,
            self.scratch.storage_bytes()?,
            self.scaled.storage_bytes()?,
        ])
    }
}

/// Write a completed spectral prefix and zero every unused output lane.
/// Caller metadata is assigned only after the numerical work has succeeded.
fn publish_snapshot<'a, V: Copy, R: Copy, S: NumericScalar>(
    count: usize,
    values: &[V],
    residuals: &[R],
    output_values: &mut [V],
    output_residuals: &mut [R],
    vectors: impl IntoIterator<Item = (&'a crate::Matrix<S>, &'a mut crate::Matrix<S>)>,
    zero: (V, R),
) {
    output_values[..count].copy_from_slice(&values[..count]);
    output_values[count..].fill(zero.0);
    output_residuals[..count].copy_from_slice(&residuals[..count]);
    output_residuals[count..].fill(zero.1);
    for (source, output) in vectors {
        if count == output.cols() {
            output.as_mut_slice().copy_from_slice(source.as_slice());
            continue;
        }
        let width = output.cols().max(1);
        for (source, output) in source
            .as_slice()
            .chunks(width)
            .zip(output.as_mut_slice().chunks_mut(width))
        {
            output[..count].copy_from_slice(&source[..count]);
            output[count..].fill(S::ZERO);
        }
    }
}
