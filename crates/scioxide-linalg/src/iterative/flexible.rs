//! LGMRES, GCROT, the block GMRES family and Krylov recycle spaces.
use crate::Preconditioning;
use crate::workspace::accounting::{
    PlannedStorage, StorageBytes, StoragePlan, bytes, product, requirements, resize, sum,
};

use super::arnoldi::{KrylovRecycleSpace, Method, Options, Result, Storage};
use super::support::validate_square;
use super::workspace::{IterativeProgress, Job, Shape, StorageKind};
use crate::kernel::finite;
use crate::workspace::SolveSlot;
use crate::{
    IterativeRequirements, IterativeSolveReport, IterativeSolveResult, IterativeWorkspace,
    IterativeWorkspaceRequest, LinalgExecutionContext, LinearSolveTermination, Vector, VectorView,
    VectorViewMut,
};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    AttemptOutcome, AttemptRecord, BlockVectorView, BlockVectorViewMut, Budget, ExecutionContext,
    LinearOperator, NumericScalar, PartialResultTrust, ProblemToken, RankThreshold, RealScalar,
    ResidualTolerance, SciOxideError, SolveIdentity,
};

/// Termination status and diagnostics of a block Krylov solve.
///
/// Every right-hand side has its own stopping test. Per-column residuals,
/// references and convergence flags, and the residual history, are read from
/// [`IterativeWorkspace::last_accepted`](crate::IterativeWorkspace::last_accepted).
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct BlockKrylovReport<R: RealScalar> {
    termination: LinearSolveTermination,
    iterations: usize,
    max_residual: Option<R>,
    restarts: usize,
    deflated: usize,
    active_rank: usize,
    generated_directions: usize,
    retained: usize,
}

impl<R: RealScalar> super::workspace::StoppedReport for BlockKrylovReport<R> {
    fn unmeasured_stop(termination: LinearSolveTermination) -> Self {
        Self {
            termination,
            iterations: 0,
            max_residual: None,
            restarts: 0,
            deflated: 0,
            active_rank: 0,
            generated_directions: 0,
            retained: 0,
        }
    }
}

impl<R: RealScalar> BlockKrylovReport<R> {
    /// Returns the termination status; `Converged` requires every column to converge.
    pub const fn termination(&self) -> LinearSolveTermination {
        self.termination
    }
    /// Returns the number of block steps counted toward the iteration limit.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }
    /// Returns the largest true residual `‖bⱼ − A xⱼ‖₂` of the returned columns.
    ///
    /// It is `None` when the solve stopped before the first residual was computed.
    pub const fn max_residual(&self) -> Option<R> {
        self.max_residual
    }
    /// Returns the number of cycles started after the first one.
    pub const fn restarts(&self) -> usize {
        self.restarts
    }
    /// Returns the number of directions dropped by the rank threshold.
    ///
    /// A dropped direction does not establish that `A` is singular.
    pub const fn deflated(&self) -> usize {
        self.deflated
    }
    /// Returns the number of linearly independent residuals at the most recent restart.
    pub const fn active_rank(&self) -> usize {
        self.active_rank
    }
    /// Returns the number of preconditioned directions generated during this call.
    pub const fn generated_directions(&self) -> usize {
        self.generated_directions
    }
    /// Returns the number of direction/image pairs in the recycle space after this call.
    ///
    /// LGMRES pairs can be linearly dependent.
    pub const fn retained(&self) -> usize {
        self.retained
    }
    /// Returns whether every column meets the residual tolerance.
    pub const fn is_success(&self) -> bool {
        matches!(self.termination, LinearSolveTermination::Converged)
    }

    fn progress(&self) -> (AttemptOutcome, Option<IterativeProgress<R>>) {
        (
            if self.is_success() {
                AttemptOutcome::Succeeded
            } else {
                AttemptOutcome::Stopped
            },
            Some(IterativeProgress::new(
                self.termination,
                self.iterations,
                self.max_residual,
                self.restarts,
                if self.max_residual.is_some() {
                    super::checkpoint_trust(self.is_success())
                } else {
                    PartialResultTrust::DiagnosticOnly
                },
            )),
        )
    }
}

fn block_report<S: NumericScalar>(
    storage: &Storage<S>,
    termination: LinearSolveTermination,
) -> BlockKrylovReport<S::Real> {
    BlockKrylovReport {
        termination,
        iterations: storage.iterations,
        max_residual: Some(
            storage
                .residuals
                .iter()
                .copied()
                .fold(S::Real::ZERO, RealScalar::max),
        ),
        restarts: storage.restarts,
        deflated: storage.deflated,
        active_rank: storage.active_rank,
        generated_directions: storage.generated,
        retained: storage.recycle.len(),
    }
}

fn scalar_report<S: NumericScalar>(
    storage: &Storage<S>,
    termination: LinearSolveTermination,
) -> Result<IterativeSolveReport<S::Real>> {
    let report = IterativeSolveReport::from_summary(crate::result::IterativeSolveSummary {
        termination,
        iterations: storage.iterations,
        raw_residual_norm: storage.residuals[0],
        residual_reference: storage.references[0],
        trust: super::checkpoint_trust(termination.is_success()),
    })?;
    Ok(report.with_krylov(crate::KrylovDiagnostics {
        normal_residual: None,
        condition_estimate: None,
        rank_loss_detected: storage.deflated > 0,
        restarts: storage.restarts,
    }))
}

/// Storage requirements of an Arnoldi-family solve with `columns` coupled RHS.
/// The request's recycle capacity sizes the augmentation storage.
pub(super) fn requirements<S: NumericScalar>(
    options: &Options<S::Real>,
    request: &IterativeWorkspaceRequest,
    columns: usize,
    staging: Option<&IterativeWorkspaceRequest>,
) -> Result<IterativeRequirements<S>> {
    IterativeRequirements::new(
        Shape {
            kind: StorageKind::Arnoldi,
            rows: request.rows(),
            cols: request.cols(),
            operator_scratch: request.operator_scratch(),
            preconditioner_scratch: request.preconditioner_scratch(),
            columns,

            cycle: options.restart.get().min(options.limit().get()),
            retained: request.recycle_capacity(),
        },
        options.history.then_some(options.limit()),
        staging,
        options.execution.byte_limit(),
    )
}

/// Reject inputs before reservation: square operator, preconditioner and
/// recycle-space compatibility.
pub(super) fn preflight<S: NumericScalar>(
    options: &Options<S::Real>,
    a: &dyn LinearOperator<S>,
    p: Preconditioning<'_, S>,
    recycle: Option<(ProblemToken, &KrylovRecycleSpace<S>)>,
) -> Result<()> {
    validate_square(a)?;
    p.validate_dimension(a.cols())?;
    if let Some((epoch, space)) = recycle {
        crate::checked::expect_shape([a.rows()], [space.dimension])?;
        if space.operator_epoch() != epoch
            || (!space.is_empty() && space.method != Some(options.method))
        {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("recycle"),
                "recycle space has a stale operator epoch or a different method",
            ));
        }
    }
    Ok(())
}

/// Run the shared Arnoldi engine on staged inputs; `x` holds the initial
/// guess and receives the accepted iterate.
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
pub(super) fn execute<S: NumericScalar>(
    options: &Options<S::Real>,
    a: &dyn LinearOperator<S>,
    b: &[S],
    x: &mut [S],
    p: Preconditioning<'_, S>,
    recycle: Option<&mut KrylovRecycleSpace<S>>,
    storage: &mut Storage<S>,
    kernels: crate::KernelContext<'_>,
    context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
) -> Result<LinearSolveTermination> {
    // Recycled directions are only valid for the operator they were built
    // with (their images are `A d`); preflight checked the caller's epoch.
    // The caller's space is copied into workspace storage and written back
    // only after the run, so a failed solve leaves it unchanged.
    match recycle.as_deref() {
        Some(space) => storage.recycle.copy_from(space)?,
        None => storage.recycle.clear(ProblemToken::new(0)),
    }
    storage.recycle.method = Some(options.method);
    // An execution stop raised from inside a provider is converted into an
    // ordinary stopped termination when a verified iterate exists, so the
    // caller still receives that iterate and its true residual.
    let termination = match storage.run(options, a, p, b, x, kernels, context) {
        Err(error) => match error.view() {
            scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
                if storage.verified_this_call =>
            {
                super::support::execution_stop(error)?
            }
            _ => return Err(error),
        },
        Ok(termination) => termination,
    };
    if let Some(space) = recycle {
        space.copy_from(&storage.recycle)?;
    }
    x.copy_from_slice(&storage.x);
    Ok(termination)
}

/// Scalar Arnoldi solve through the shared workspace driver.
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
pub(super) fn solve_scalar<S: NumericScalar>(
    options: &Options<S::Real>,
    a: &dyn LinearOperator<S>,
    b: VectorView<'_, S>,
    p: Preconditioning<'_, S>,
    recycle: Option<(ProblemToken, &mut KrylovRecycleSpace<S>)>,
    mut x: VectorViewMut<'_, S>,
    workspace: &mut IterativeWorkspace<S>,
    context: LinalgExecutionContext<'_, '_>,
) -> Result<IterativeSolveReport<S::Real>> {
    workspace.reject();
    preflight(
        options,
        a,
        p,
        recycle.as_ref().map(|(epoch, space)| (*epoch, &**space)),
    )?;
    let mut request = IterativeWorkspaceRequest::operator(a)?.with_preconditioner(p)?;
    if let Some((_, space)) = &recycle {
        request = request.with_recycle_space(&**space)?;
    }
    let requirements = requirements(options, &request, 1, None)?;
    let recycle = recycle.map(|(_, space)| space);
    workspace.run(
        context.with_options(&options.execution)?,
        &Job {
            algorithm: options.method.algorithm_id(false),
            limit: options.limit(),
            requirements,
        },
        b,
        &mut x,
        |storage: &mut Storage<S>, b, x, kernels, solve| {
            let termination = execute(
                options,
                a,
                b.as_slice(),
                x.as_mut_slice(),
                p,
                recycle,
                storage,
                kernels,
                solve,
            )?;
            scalar_report(storage, termination)
        },
        IterativeProgress::of,
    )
}

macro_rules! scalar_solver {
    ($solver:ident, $example:literal, $method:ident, $description:literal) => {
        #[doc = $description]
        ///
        /// The preconditioner is applied from the right and may change between
        /// applications, as in [`GmresSolver`](crate::GmresSolver). A recycle space
        /// of capacity `k` holds `2k` vectors of length `n`, and the workspace needs
        /// about `4k + 2` more; to reserve them, pass the space to
        /// [`IterativeWorkspaceRequest::with_recycle_space`].
        #[derive(Debug, Clone, Copy)]
        #[doc = $example]
        pub struct $solver<S: NumericScalar> {
            pub(super) options: Options<S::Real>,
        }

        crate::traits::impl_iterative_solver!(
            $solver,
            plan,
            plan.options.method.algorithm_id(false)
        );
        impl<S: NumericScalar> $solver<S> {
            /// Returns the execution options.
            pub const fn execution_options(&self) -> crate::LinalgExecutionOptions {
                self.options.execution
            }

            /// Sets the execution options: policy, product accuracy, time limit,
            /// byte limit and iteration budget.
            ///
            /// An iteration budget in `execution` caps `max_iterations`. The
            /// accuracy setting applies to dense operator products; the
            /// recurrence safeguards always use `Standard` accuracy.
            pub const fn with_execution_options(
                mut self,
                execution: crate::LinalgExecutionOptions,
            ) -> Self {
                self.options.execution = execution;
                self
            }

            /// Creates a solver with residual tolerance `tolerance` and at most
            /// `max_iterations` steps in total.
            ///
            /// The solve converges when `‖b − A x‖₂ ≤ atol + rtol·‖b‖₂`. The restart
            /// length is 30 and both parts of the rank threshold are the machine epsilon.
            pub fn new(tolerance: ResidualTolerance<S::Real>, max_iterations: Budget) -> Self {
                Self {
                    options: Options::new(Method::$method, tolerance, max_iterations),
                }
            }
            /// Sets the restart length, the number of steps per cycle (default 30).
            pub const fn with_restart(mut self, restart: Budget) -> Self {
                self.options.restart = restart;
                self
            }
            /// Sets the threshold for dropping dependent directions.
            ///
            /// A direction is dropped when orthogonalization reduces its norm to at
            /// most `atol + rtol` times its original norm.
            pub const fn with_rank_threshold(mut self, rank: RankThreshold<S::Real>) -> Self {
                self.options.rank = rank;
                self
            }
            /// Enables or disables the residual history.
            ///
            /// Between verification checkpoints the history holds residual
            /// estimates; the other entries are true residuals.
            pub const fn with_history(mut self, history: bool) -> Self {
                self.options.history = history;
                self
            }
            /// Returns the storage a solve of `request` needs.
            ///
            /// Add the preconditioner with
            /// [`IterativeWorkspaceRequest::with_preconditioner`] and the recycle
            /// space with [`IterativeWorkspaceRequest::with_recycle_space`].
            ///
            /// # Errors
            ///
            /// - `SizeOverflow` if a size or byte count overflows `usize`.
            /// - `ResourceLimit` if the storage exceeds the byte limit of the
            ///   execution options.
            pub fn workspace_requirements(
                &self,
                request: &IterativeWorkspaceRequest,
            ) -> Result<IterativeRequirements<S>> {
                requirements(&self.options, request, 1, Some(request))
            }
            /// Creates a workspace reserved for `request`.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::workspace_requirements`].
            /// - `AllocationFailure` if memory cannot be reserved.
            pub fn workspace(
                &self,
                request: &IterativeWorkspaceRequest,
            ) -> Result<IterativeWorkspace<S>> {
                let mut workspace = IterativeWorkspace::new();
                workspace.try_reserve(&self.workspace_requirements(request)?)?;
                Ok(workspace)
            }
            /// Solves `A x = b` from `x = 0` without preconditioning and returns `x`.
            ///
            /// A solve that ends without meeting the tolerance, at the iteration or
            /// evaluation limit or by a breakdown, is an error. Use
            /// [`solve_with_options`](Self::solve_with_options) for a preconditioner, a
            /// start vector or the iterate of an unsuccessful solve.
            ///
            /// # Errors
            ///
            /// - The errors of [`solve_with_options`](Self::solve_with_options).
            /// - `ConvergenceFailure` (`NestedSolver`) if the solve does not converge; its
            ///   diagnostics name the solver and carry the iteration count and the
            ///   residual norm.
            pub fn solve<'b>(
                &self,
                a: &dyn LinearOperator<S>,
                b: impl Into<VectorView<'b, S>>,
            ) -> core::result::Result<crate::Vector<S>, scioxide_core::SciOxideError> {
                self.solve_with_options(a, b, crate::Preconditioning::None, None)?
                    .into_converged_solution()
            }
            /// Solves `A x = b` from `x0`, or from zero, without a recycle space, and returns the
            /// iterate with its report.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::solve_into`].
            /// - `AllocationFailure` if the output or workspace cannot be allocated.
            pub fn solve_with_options<'b>(
                &self,
                a: &dyn LinearOperator<S>,
                b: impl Into<VectorView<'b, S>>,
                p: Preconditioning<'_, S>,
                x0: Option<VectorView<'_, S>>,
            ) -> Result<IterativeSolveResult<S>> {
                let b = b.into();
                IterativeWorkspace::solve_owned(
                    a.cols(),
                    x0,
                    <Self as crate::IterativeLinearSolver<S>>::algorithm_id(self),
                    |x, workspace| {
                        self.solve_into(
                            a,
                            b,
                            p,
                            x,
                            workspace,
                            ExecutionContext::sequential_native(),
                        )
                    },
                )
            }
            /// Solves `A x = b` in place from the values in `x`, without a recycle space.
            ///
            /// No pairs are stored, so the iteration is restarted flexible GMRES.
            ///
            /// # Errors
            ///
            /// The errors listed on
            /// [`IterativeLinearSolver::solve_into`](crate::IterativeLinearSolver::solve_into).
            pub fn solve_into<'b, 'scope, 'capability>(
                &self,
                a: &dyn LinearOperator<S>,
                b: impl Into<VectorView<'b, S>>,
                p: Preconditioning<'_, S>,
                x: VectorViewMut<'_, S>,
                workspace: &mut IterativeWorkspace<S>,
                context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
            ) -> Result<IterativeSolveReport<S::Real>> {
                solve_scalar(
                    &self.options,
                    a,
                    b.into(),
                    p,
                    None,
                    x,
                    workspace,
                    context.into(),
                )
            }
        }
    };
}

scalar_solver!(
    LgmresSolver,
    r#"

# Examples

```
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    IterativeWorkspace, IterativeWorkspaceRequest, LgmresSolver, LinearSolveTermination, Matrix,
    Preconditioning, Vector,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let solver = LgmresSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
);
let request = IterativeWorkspaceRequest::operator(&operator)?;
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let mut solution = Vector::zeros(2)?;
let report = solver.solve_into(
    &operator,
    &rhs,
    Preconditioning::None,
    solution.view_mut(),
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```

# References

- A. H. Baker, E. R. Jessup and T. Manteuffel, "A technique for accelerating the
  convergence of restarted GMRES", *SIAM J. Matrix Anal. Appl.* 26(4), 2005."#,
    Lgmres,
    r#"LGMRES: restarted flexible GMRES augmented with the corrections of earlier cycles.

At the end of each cycle, LGMRES adds the normalized corrections `xᵢ − xᵢ₋₁`
of the most recent cycles, together with their images under `A`, to the search
space and minimizes the residual over the enlarged space. This counters the
stalling of restarted GMRES. The pairs are kept in a [`KrylovRecycleSpace`] of
capacity `k`; when it is full, the oldest pair is dropped.
[`Self::solve_recycled_into`] uses and updates a space you provide, which can
carry the pairs over to later solves with the same operator. [`Self::solve`]
and [`Self::solve_into`] store no pairs and behave like
[`GmresSolver`](crate::GmresSolver)."#
);
scalar_solver!(
    GcrotSolver,
    r#"

# Examples

```
use scioxide_core::{Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    GcrotSolver, IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, Matrix,
    Preconditioning, Vector,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let solver = GcrotSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
);
let request = IterativeWorkspaceRequest::operator(&operator)?;
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = Vector::from_vec(vec![1.0, 2.0]);
let mut solution = Vector::zeros(2)?;
let report = solver.solve_into(
    &operator,
    &rhs,
    Preconditioning::None,
    solution.view_mut(),
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution.as_slice()[1] - 2.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```

# References

- E. de Sturler, "Truncation strategies for optimal Krylov subspace methods",
  *SIAM J. Numer. Anal.* 36(3), 1999.
- J. E. Hicken and D. W. Zingg, "A simplified and flexible variant of GCROT for
  solving nonsymmetric linear systems", *SIAM J. Sci. Comput.* 32(3), 2010."#,
    Gcrot,
    r#"Flexible GCROT(m, k): restarted GMRES with an outer space of stored directions.

The outer space holds up to `k` pairs `(uᵢ, cᵢ)` with `cᵢ = A uᵢ` and
orthonormal `cᵢ`, stored in a [`KrylovRecycleSpace`] of capacity `k`. Each
cycle first minimizes the residual over `span{uᵢ}`, then runs `m` flexible
GMRES steps with an Arnoldi basis orthogonal to the `cᵢ`. The correction of the
cycle is orthogonalized against the stored `cᵢ` and added to the space; when it
is full, the oldest pair is dropped. [`Self::solve_recycled_into`] uses and
updates a space you provide, which can carry the pairs over to later solves
with the same operator. [`Self::solve`] and [`Self::solve_into`] store no pairs
and behave like [`GmresSolver`](crate::GmresSolver)."#
);

macro_rules! scalar_recycling {
    ($solver:ident) => {
        impl<S: NumericScalar> $solver<S> {
            /// Solves `A x = b` in place from the values in `x`, using and updating `recycle`.
            ///
            /// `operator_epoch` identifies `A` and must equal the epoch of
            /// `recycle`. The right-hand side may differ from earlier solves. The
            /// space is updated whenever the call returns `Ok`, including an
            /// unsuccessful termination; on error it is unchanged.
            ///
            /// # Errors
            ///
            /// - The errors listed on
            ///   [`IterativeLinearSolver::solve_into`](crate::IterativeLinearSolver::solve_into).
            /// - `DimensionMismatch` if the dimension of `recycle` differs from `a.rows()`.
            /// - `InvalidInput` (`Inconsistent`) if `operator_epoch` differs from
            ///   [`KrylovRecycleSpace::operator_epoch`], or if `recycle` is non-empty
            ///   and was filled by the other method. Both are checked before `A` is applied.
            #[expect(clippy::too_many_arguments, reason = "the numerical operation keeps its operands, storage and control parameters explicit")]
            pub fn solve_recycled_into<'b, 'scope, 'capability>(
                &self,
                a: &dyn LinearOperator<S>,
                b: impl Into<VectorView<'b, S>>,
                p: Preconditioning<'_, S>,
                operator_epoch: ProblemToken,
                recycle: &mut KrylovRecycleSpace<S>,
                x: VectorViewMut<'_, S>,
                workspace: &mut IterativeWorkspace<S>,
                context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
            ) -> Result<IterativeSolveReport<S::Real>> {
                solve_scalar(
                    &self.options,
                    a,
                    b.into(),
                    p,
                    Some((operator_epoch, recycle)),
                    x,
                    workspace,
                    context.into(),
                )
            }
        }
    };
}
scalar_recycling!(LgmresSolver);
scalar_recycling!(GcrotSolver);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecycleLayout {
    pub(crate) n: usize,
    pub(crate) count: usize,
    pub(crate) provider: usize,
}

requirements!(
    KrylovRecycleRequirements,
    RecycleLayout,
    "Storage and byte limit of a [`KrylovRecycleWorkspace`] for one recycle-space refresh."
);

/// Scratch storage for [`KrylovRecycleSpace::refresh`].
#[derive(Debug)]
pub struct KrylovRecycleWorkspace<S: NumericScalar> {
    slot: SolveSlot<()>,
    staged: KrylovRecycleSpace<S>,
    independent: KrylovRecycleSpace<S>,
    original_image: Vector<S>,
    direction: Vector<S>,
    image: Vector<S>,
    scratch: Vec<S>,
}
impl<S: NumericScalar> KrylovRecycleWorkspace<S> {
    /// Returns the record of the most recent refresh.
    ///
    /// The outcome is `Succeeded` when the space was replaced and `Stopped`
    /// when a budget, cancellation or deadline left it unchanged.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.slot.last_attempt()
    }
}
impl<S: NumericScalar> KrylovRecycleSpace<S> {
    /// Recomputes the images `cᵢ = A uᵢ` for a changed operator `a` under `operator_epoch`.
    ///
    /// For GCROT the new images are orthonormalized again. For LGMRES,
    /// directions whose new images are numerically dependent are dropped. Both
    /// use `rank` as the threshold for dependence. Each direction costs one
    /// product with `A` and counts as one iteration toward the iteration budget
    /// of `options`, which also sets the byte and time limits.
    ///
    /// Returns `Ok(true)` when the space was replaced. Returns `Ok(false)` when a
    /// budget, cancellation or deadline stopped the refresh; the space is then
    /// unchanged.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `a` is not square of the space's dimension.
    /// - `NonFinite` if `A` returns NaN or infinity.
    /// - Errors returned by `a`, unchanged.
    /// - Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// On error the space is unchanged.
    pub fn refresh<'scope, 'capability>(
        &mut self,
        a: &dyn LinearOperator<S>,
        operator_epoch: ProblemToken,
        rank: RankThreshold<S::Real>,
        options: &crate::LinalgExecutionOptions,
        workspace: &mut KrylovRecycleWorkspace<S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<bool> {
        workspace.slot.reject();
        let requirements =
            self.workspace_requirements(&KrylovRecycleWorkspaceRequest::new(a, options)?)?;
        workspace.prepare(&requirements)?;
        let context = context.into().with_options(options)?;
        let kernels = context.kernel();
        if a.rows() != self.dimension
            || a.cols() != self.dimension
            || workspace.staged.dimension != self.dimension
            || workspace.staged.capacity() != self.capacity()
            || workspace.scratch.len() != a.apply_workspace_len()?
        {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("workspace"),
                "incompatible recycle refresh workspace",
            ));
        }
        let KrylovRecycleWorkspace {
            slot,
            staged,
            independent,
            original_image,
            direction,
            image,
            scratch,
            ..
        } = workspace;
        slot.run(
            context.into_scope(),
            SolveIdentity::new(
                ProblemToken::new(0),
                scioxide_core::algorithm_id!("linear", "KrylovRecycleRefresh", 1),
                self.dimension as u128,
                0,
            ),
            options.solve_budgets(),
            |solve| match self.refresh_kernel(
                a,
                operator_epoch,
                rank,
                RefreshBuffers {
                    staged,
                    independent,
                    original_image,
                    direction,
                    image,
                    scratch,
                },
                kernels,
                &mut scioxide_core::OperatorContext::with_kernels(solve, &kernels),
            ) {
                Err(error) => match error.view() {
                    scioxide_core::SciOxideErrorView::ExecutionStopped { .. } => Ok(false),
                    _ => Err(error),
                },
                result => result,
            },
            |refreshed| {
                let Ok(refreshed) = refreshed else {
                    return (AttemptOutcome::Failed, None);
                };
                (
                    if *refreshed {
                        AttemptOutcome::Succeeded
                    } else {
                        AttemptOutcome::Stopped
                    },
                    None,
                )
            },
        )
    }
    fn refresh_kernel(
        &mut self,
        a: &dyn LinearOperator<S>,
        operator_epoch: ProblemToken,
        rank: RankThreshold<S::Real>,
        buffers: RefreshBuffers<'_, S>,
        kernels: crate::KernelContext<'_>,
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<bool> {
        let RefreshBuffers {
            staged,
            independent,
            original_image,
            direction,
            image,
            scratch,
        } = buffers;
        staged.clear(operator_epoch);
        staged.method = self.method;
        independent.clear(operator_epoch);
        for i in 0..self.len {
            if context.stop_reason()?.is_some() || !context.record_iteration()?.is_granted() {
                return Ok(false);
            }
            finite::copy(direction.as_mut_slice(), self.directions[i].as_slice())?;
            if super::safety::apply_operator_scoped(
                a,
                direction.as_slice(),
                finite::maximum(direction.as_slice(), kernels.dispatch())?,
                image.as_mut_slice(),
                scratch,
                context,
            )?
            .is_exhausted()
            {
                return Ok(false);
            }
            if self.method == Some(Method::Lgmres) {
                finite::copy(original_image.as_mut_slice(), image.as_slice())?;
                if !independent.insert(
                    direction.as_mut_slice(),
                    image.as_mut_slice(),
                    rank,
                    kernels,
                )? {
                    continue;
                }
                finite::copy(direction.as_mut_slice(), self.directions[i].as_slice())?;
                finite::copy(image.as_mut_slice(), original_image.as_slice())?;
            }
            staged.insert(
                direction.as_mut_slice(),
                image.as_mut_slice(),
                rank,
                kernels,
            )?;
        }
        self.copy_from(staged)?;
        Ok(true)
    }
}

struct RefreshBuffers<'a, S: NumericScalar> {
    staged: &'a mut KrylovRecycleSpace<S>,
    independent: &'a mut KrylovRecycleSpace<S>,
    original_image: &'a mut Vector<S>,
    direction: &'a mut Vector<S>,
    image: &'a mut Vector<S>,
    scratch: &'a mut [S],
}

/// Solution and per-column diagnostics returned by an owned block Krylov solve.
///
/// The solution is stored column by column: column `j` occupies entries
/// `j·rows .. (j + 1)·rows`.
#[derive(Debug, Clone)]
#[must_use]
pub struct BlockKrylovResult<S: NumericScalar> {
    solution: Vec<S>,
    rows: usize,
    right_hand_sides: usize,
    report: BlockKrylovReport<S::Real>,
    residuals: Vec<S::Real>,
    references: Vec<S::Real>,
    converged: Vec<bool>,
    history: Option<Vec<S::Real>>,
}

impl<S: NumericScalar> BlockKrylovResult<S> {
    /// Returns the solution, column by column.
    pub fn solution(&self) -> &[S] {
        &self.solution
    }
    /// Returns the solution, column by column, whatever the termination status.
    pub fn into_solution(self) -> Vec<S> {
        self.solution
    }
    /// Returns the number of entries in each solution column.
    pub const fn rows(&self) -> usize {
        self.rows
    }
    /// Returns the number of solution columns.
    pub const fn right_hand_sides(&self) -> usize {
        self.right_hand_sides
    }
    /// Returns the report of the solve.
    pub const fn report(&self) -> BlockKrylovReport<S::Real> {
        self.report
    }
    /// Returns the true residual `‖bⱼ − A xⱼ‖₂` of each returned column.
    pub fn residuals(&self) -> &[S::Real] {
        &self.residuals
    }
    /// Returns the reference `‖bⱼ‖₂` of each column, or one for a zero column.
    pub fn references(&self) -> &[S::Real] {
        &self.references
    }
    /// Returns whether each column meets the residual tolerance.
    pub fn converged(&self) -> &[bool] {
        &self.converged
    }
    /// Returns the largest column residual after each step, if history was enabled.
    pub fn history(&self) -> Option<&[S::Real]> {
        self.history.as_deref()
    }
    /// Returns whether every column meets the residual tolerance.
    pub const fn is_success(&self) -> bool {
        self.report.is_success()
    }
}

/// Block Arnoldi solve through the shared workspace driver.
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
fn solve_block<S: NumericScalar>(
    options: &Options<S::Real>,
    a: &dyn LinearOperator<S>,
    b: BlockVectorView<'_, S>,
    p: Preconditioning<'_, S>,
    recycle: Option<(ProblemToken, &mut KrylovRecycleSpace<S>)>,
    x: &mut BlockVectorViewMut<'_, S>,
    workspace: &mut IterativeWorkspace<S>,
    context: LinalgExecutionContext<'_, '_>,
) -> Result<BlockKrylovReport<S::Real>> {
    workspace.reject();
    preflight(
        options,
        a,
        p,
        recycle.as_ref().map(|(epoch, space)| (*epoch, &**space)),
    )?;
    let columns = b.right_hand_sides();
    crate::checked::expect_shape([a.rows(), columns], [b.rows(), columns])?;
    crate::checked::expect_shape([a.cols(), columns], [x.rows(), x.right_hand_sides()])?;
    let mut request = IterativeWorkspaceRequest::operator(a)?
        .with_right_hand_sides(columns)
        .with_preconditioner(p)?;
    if let Some((_, space)) = &recycle {
        request = request.with_recycle_space(&**space)?;
    }
    let requirements = requirements(options, &request, columns, None)?;
    let recycle = recycle.map(|(_, space)| space);
    workspace.run(
        context.with_options(&options.execution)?,
        &Job {
            algorithm: options.method.algorithm_id(true),
            limit: options.limit(),
            requirements,
        },
        VectorView::from_slice(b.as_slice()),
        &mut VectorViewMut::from_slice(x.as_mut_slice()),
        |storage: &mut Storage<S>, b, x, kernels, solve| {
            let termination = execute(
                options,
                a,
                b.as_slice(),
                x.as_mut_slice(),
                p,
                recycle,
                storage,
                kernels,
                solve,
            )?;
            Ok(block_report(storage, termination))
        },
        BlockKrylovReport::progress,
    )
}

macro_rules! block_solver {
    ($solver:ident, $example:literal, $method:ident, $description:literal) => {
        #[doc = $description]
        ///
        /// `b` and `x` hold one column per right-hand side. One block Krylov space
        /// is built from the residuals of the columns that have not converged, and
        /// each column's residual is minimized over it. A residual that is
        /// numerically in the span of the others is dropped, so dependent
        /// right-hand sides cost no extra products. A block step applies `A` and
        /// `M⁻¹` once to each column of the current block, in a fixed column order;
        /// `M⁻¹` may change between applications. Column `j` converges when
        /// `‖bⱼ − A xⱼ‖₂ ≤ atol + rtol·‖bⱼ‖₂`, and the solve succeeds only when
        /// every column converges. To solve the columns independently, use
        /// [`IterativeLinearSolver::solve_block_into`](crate::IterativeLinearSolver::solve_block_into)
        /// with a single-vector solver.
        #[derive(Debug, Clone, Copy)]
        #[doc = $example]
        pub struct $solver<S: NumericScalar> {
            options: Options<S::Real>,
        }
        impl<S: NumericScalar> $solver<S> {
            /// Returns the execution options.
            pub const fn execution_options(&self) -> crate::LinalgExecutionOptions {
                self.options.execution
            }

            /// Sets the execution options: policy, product accuracy, time limit,
            /// byte limit and iteration budget.
            ///
            /// An iteration budget in `execution` caps `max_iterations`. The
            /// accuracy setting applies to dense operator products; the
            /// recurrence safeguards always use `Standard` accuracy.
            pub const fn with_execution_options(
                mut self,
                execution: crate::LinalgExecutionOptions,
            ) -> Self {
                self.options.execution = execution;
                self
            }

            /// Creates a solver with per-column residual tolerance `tolerance` and at
            /// most `max_iterations` block steps in total.
            ///
            /// The restart length is 30 and both parts of the rank threshold are the
            /// machine epsilon.
            pub fn new(tolerance: ResidualTolerance<S::Real>, max_iterations: Budget) -> Self {
                Self {
                    options: Options::new(Method::$method, tolerance, max_iterations),
                }
            }
            /// Sets the restart length, the number of block steps per cycle (default 30).
            pub const fn with_restart(mut self, restart: Budget) -> Self {
                self.options.restart = restart;
                self
            }
            /// Sets the threshold for dropping dependent directions.
            ///
            /// A direction is dropped when orthogonalization reduces its norm to at
            /// most `atol + rtol` times its original norm.
            pub const fn with_rank_threshold(mut self, rank: RankThreshold<S::Real>) -> Self {
                self.options.rank = rank;
                self
            }
            /// Enables or disables the history of the largest column residual.
            ///
            /// With one right-hand side, the history holds residual estimates
            /// between verification checkpoints.
            pub const fn with_history(mut self, history: bool) -> Self {
                self.options.history = history;
                self
            }
            /// Returns the storage a solve of `request` needs.
            ///
            /// Set the number of columns with
            /// [`IterativeWorkspaceRequest::with_right_hand_sides`]. The storage
            /// grows with the restart length times the number of columns.
            ///
            /// # Errors
            ///
            /// - `SizeOverflow` if a size or byte count overflows `usize`.
            /// - `ResourceLimit` if the storage exceeds the byte limit of the
            ///   execution options.
            pub fn workspace_requirements(
                &self,
                request: &IterativeWorkspaceRequest,
            ) -> Result<IterativeRequirements<S>> {
                requirements(&self.options, request, request.right_hand_sides(), None)
            }
            /// Creates a workspace reserved for `request`.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::workspace_requirements`].
            /// - `AllocationFailure` if memory cannot be reserved.
            pub fn workspace(
                &self,
                request: &IterativeWorkspaceRequest,
            ) -> Result<IterativeWorkspace<S>> {
                let mut workspace = IterativeWorkspace::new();
                workspace.try_reserve(&self.workspace_requirements(request)?)?;
                Ok(workspace)
            }
            /// Solves `A X = B` from `x0`, or from zero when `x0` is `None`.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::solve_into`].
            /// - `AllocationFailure` if the output or workspace cannot be allocated.
            pub fn solve(
                &self,
                a: &dyn LinearOperator<S>,
                b: BlockVectorView<'_, S>,
                p: Preconditioning<'_, S>,
                x0: Option<BlockVectorView<'_, S>>,
            ) -> Result<BlockKrylovResult<S>> {
                let (rows, columns) = (a.cols(), b.right_hand_sides());
                let mut solution = match x0 {
                    Some(initial) => scioxide_core::backend::try_copy_slice(
                        initial.as_slice(),
                        scioxide_core::quantity!("owned_block_solution"),
                    )?,
                    None => scioxide_core::backend::try_filled(
                        scioxide_core::backend::checked_mul(
                            rows,
                            columns,
                            scioxide_core::quantity!("owned_block_solution"),
                        )?,
                        S::ZERO,
                        scioxide_core::quantity!("owned_block_solution"),
                    )?,
                };
                let mut workspace = IterativeWorkspace::new();
                let report = self.solve_into(
                    a,
                    b,
                    p,
                    BlockVectorViewMut::new(&mut solution, rows, columns)?,
                    &mut workspace,
                    ExecutionContext::sequential_native(),
                )?;
                let copy = |values: &[_]| {
                    scioxide_core::backend::try_copy_slice(
                        values,
                        scioxide_core::quantity!("owned_block_diagnostics"),
                    )
                };
                let accepted = workspace.last_accepted().ok_or_else(|| {
                    crate::checked::inconsistent(
                        scioxide_core::argument!("workspace"),
                        "block solve returned without accepted state",
                    )
                })?;
                let columns_report = accepted.columns().ok_or_else(|| {
                    crate::checked::inconsistent(
                        scioxide_core::argument!("workspace"),
                        "block solve returned without column diagnostics",
                    )
                })?;
                Ok(BlockKrylovResult {
                    solution,
                    rows,
                    right_hand_sides: columns,
                    report,
                    residuals: copy(columns_report.residuals())?,
                    references: copy(columns_report.references())?,
                    converged: scioxide_core::backend::try_copy_slice(
                        columns_report.converged(),
                        scioxide_core::quantity!("owned_block_diagnostics"),
                    )?,
                    history: accepted.history().map(copy).transpose()?,
                })
            }
            /// Solves `A X = B` in place, starting from the columns of `x`.
            ///
            /// All columns are written together. An unsuccessful solve returns the
            /// verified block whose largest column residual is smallest (the latest
            /// one on ties); the per-column diagnostics describe that block, and
            /// the iteration count includes all steps.
            ///
            /// # Errors
            ///
            /// - The errors listed on
            ///   [`IterativeLinearSolver::solve_into`](crate::IterativeLinearSolver::solve_into).
            /// - `DimensionMismatch` if `b` or `x` does not have `a.rows()` rows and
            ///   the same number of columns.
            pub fn solve_into<'scope, 'capability>(
                &self,
                a: &dyn LinearOperator<S>,
                b: BlockVectorView<'_, S>,
                p: Preconditioning<'_, S>,
                mut x: BlockVectorViewMut<'_, S>,
                workspace: &mut IterativeWorkspace<S>,
                context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
            ) -> Result<BlockKrylovReport<S::Real>> {
                solve_block(
                    &self.options,
                    a,
                    b,
                    p,
                    None,
                    &mut x,
                    workspace,
                    context.into(),
                )
            }
        }
    };
}
block_solver!(
    BlockGmresSolver,
    r#"

# Examples

```
use scioxide_core::{BlockVectorView, BlockVectorViewMut, Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    BlockGmresSolver, IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, Matrix,
    Preconditioning,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let solver = BlockGmresSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
);
let request = IterativeWorkspaceRequest::operator(&operator)?
    .with_right_hand_sides(2);
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = [1.0, 0.0, 0.0, 1.0];
let mut solution = [0.0_f64; 4];
let report = solver.solve_into(
    &operator,
    BlockVectorView::new(&rhs, 2, 2)?,
    Preconditioning::None,
    BlockVectorViewMut::new(&mut solution, 2, 2)?,
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution[3] - 1.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```"#,
    Gmres,
    "Block flexible GMRES for several right-hand sides that share one search space."
);

block_solver!(
    BlockLgmresSolver,
    r#"

# Examples

```
use scioxide_core::{BlockVectorView, BlockVectorViewMut, Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    BlockLgmresSolver, IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination,
    Matrix, Preconditioning,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let solver = BlockLgmresSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
);
let request = IterativeWorkspaceRequest::operator(&operator)?
    .with_right_hand_sides(2);
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = [1.0, 0.0, 0.0, 1.0];
let mut solution = [0.0_f64; 4];
let report = solver.solve_into(
    &operator,
    BlockVectorView::new(&rhs, 2, 2)?,
    Preconditioning::None,
    BlockVectorViewMut::new(&mut solution, 2, 2)?,
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution[3] - 1.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```"#,
    Lgmres,
    "Block LGMRES: block GMRES augmented with earlier cycle corrections, as in [`LgmresSolver`]."
);
block_solver!(
    BlockGcrotSolver,
    r#"

# Examples

```
use scioxide_core::{BlockVectorView, BlockVectorViewMut, Budget, ExecutionContext, ResidualTolerance};
use scioxide_linalg::{
    BlockGcrotSolver, IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination, Matrix,
    Preconditioning,
};
// A = I, so the solution is x = b.
let a = Matrix::<f64>::identity(2)?;
let operator = a.view().try_finite()?;
let solver = BlockGcrotSolver::new(
    ResidualTolerance::new(1e-10, 0.0)?,
    Budget::try_from(20_usize)?,
);
let request = IterativeWorkspaceRequest::operator(&operator)?
    .with_right_hand_sides(2);
let requirements = solver.workspace_requirements(&request)?;
let mut workspace = IterativeWorkspace::new();
workspace.try_reserve(&requirements)?;
let rhs = [1.0, 0.0, 0.0, 1.0];
let mut solution = [0.0_f64; 4];
let report = solver.solve_into(
    &operator,
    BlockVectorView::new(&rhs, 2, 2)?,
    Preconditioning::None,
    BlockVectorViewMut::new(&mut solution, 2, 2)?,
    &mut workspace,
    ExecutionContext::sequential_native(),
)?;
assert_eq!(report.termination(), LinearSolveTermination::Converged);
assert!((solution[3] - 1.0).abs() < 1e-10);
# Ok::<(), scioxide_core::SciOxideError>(())
```"#,
    Gcrot,
    "Block flexible GCROT: block GMRES with stored outer directions, as in [`GcrotSolver`]."
);

macro_rules! block_recycling {
    ($solver:ident) => {
        impl<S: NumericScalar> $solver<S> {
            /// Solves `A X = B` in place from the columns of `x`, using and updating `recycle`.
            ///
            /// `operator_epoch` identifies `A` and must equal the epoch of
            /// `recycle`. The space is updated whenever the call returns `Ok`; on
            /// error it is unchanged.
            ///
            /// # Errors
            ///
            /// - The errors of [`Self::solve_into`].
            /// - `DimensionMismatch` if the dimension of `recycle` differs from `a.rows()`.
            /// - `InvalidInput` (`Inconsistent`) if `operator_epoch` differs from
            ///   [`KrylovRecycleSpace::operator_epoch`], or if `recycle` is non-empty
            ///   and was filled by the other method.
            #[expect(clippy::too_many_arguments, reason = "the numerical operation keeps its operands, storage and control parameters explicit")]
            pub fn solve_recycled_into<'scope, 'capability>(
                &self,
                a: &dyn LinearOperator<S>,
                b: BlockVectorView<'_, S>,
                p: Preconditioning<'_, S>,
                operator_epoch: ProblemToken,
                recycle: &mut KrylovRecycleSpace<S>,
                mut x: BlockVectorViewMut<'_, S>,
                workspace: &mut IterativeWorkspace<S>,
                context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
            ) -> Result<BlockKrylovReport<S::Real>> {
                solve_block(
                    &self.options,
                    a,
                    b,
                    p,
                    Some((operator_epoch, recycle)),
                    &mut x,
                    workspace,
                    context.into(),
                )
            }
        }
    };
}
block_recycling!(BlockLgmresSolver);
block_recycling!(BlockGcrotSolver);

impl<S: NumericScalar> Default for KrylovRecycleWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> KrylovRecycleWorkspace<S> {
    /// Creates an empty workspace without allocating.
    pub fn new() -> Self {
        Self {
            slot: SolveSlot::new(),
            staged: KrylovRecycleSpace::empty(),
            independent: KrylovRecycleSpace::empty(),
            original_image: Vector::empty(),
            direction: Vector::empty(),
            image: Vector::empty(),
            scratch: Vec::new(),
        }
    }
    /// Checks whether the workspace covers `requirements`, without allocating.
    ///
    /// # Errors
    ///
    /// Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn check(
        &self,
        requirements: &KrylovRecycleRequirements<S>,
    ) -> Result<scioxide_core::Capacity> {
        PlannedStorage::check(self, requirements)
    }
    /// Grows the buffers that do not cover `requirements`; the attempt record is kept.
    ///
    /// # Errors
    ///
    /// Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// After a failed growth some buffers may already be larger; the workspace
    /// stays usable.
    pub fn try_reserve(&mut self, requirements: &KrylovRecycleRequirements<S>) -> Result<()> {
        PlannedStorage::reserve(self, requirements)
    }
}
impl<S: NumericScalar> PlannedStorage<S> for KrylovRecycleWorkspace<S> {
    type Requirements = KrylovRecycleRequirements<S>;
    fn storage_plan(&self, requirements: &KrylovRecycleRequirements<S>) -> Result<StoragePlan> {
        let RecycleLayout { n, count, provider } = requirements.layout;
        let mut plan = StoragePlan::new(self.storage_bytes()?);
        self.staged.action_storage(&mut plan, n, count)?;
        self.independent.action_storage(&mut plan, n, count)?;
        for vector in [&self.original_image, &self.direction, &self.image] {
            plan.capacity::<S>(vector.capacity(), n)?;
        }
        plan.buffer(&self.scratch, provider)?;
        Ok(plan)
    }
    fn configure(&mut self, requirements: &KrylovRecycleRequirements<S>) -> Result<()> {
        let RecycleLayout { n, count, provider } = requirements.layout;
        self.staged.configure_refresh(n, count)?;
        self.independent.configure_refresh(n, count)?;
        self.original_image.try_resize(n, S::ZERO)?;
        self.direction.try_resize(n, S::ZERO)?;
        self.image.try_resize(n, S::ZERO)?;
        resize(&mut self.scratch, provider, S::ZERO)
    }
}
impl<S: NumericScalar> StorageBytes for KrylovRecycleWorkspace<S> {
    fn storage_bytes(&self) -> core::result::Result<usize, SciOxideError> {
        sum([
            self.staged.storage_bytes()?,
            self.independent.storage_bytes()?,
            self.original_image.storage_bytes()?,
            self.direction.storage_bytes()?,
            self.image.storage_bytes()?,
            self.scratch.storage_bytes()?,
        ])
    }
}
/// Sizes and execution options of a [`KrylovRecycleSpace::refresh`].
///
/// A request holds the operator's shape and scratch length and the execution
/// options, not the operator itself.
///
/// # Examples
///
/// ```
/// use scioxide_core::ProblemToken;
/// use scioxide_linalg::{
///     KrylovRecycleSpace, KrylovRecycleWorkspaceRequest, LinalgExecutionOptions, Matrix,
/// };
/// let a = Matrix::<f64>::identity(3)?;
/// let a = a.view().try_finite()?;
/// let space = KrylovRecycleSpace::<f64>::new(3, 2, ProblemToken::new(1))?;
/// let request = KrylovRecycleWorkspaceRequest::new(&a, &LinalgExecutionOptions::new())?;
/// let workspace = space.workspace(&request)?;
/// assert!(workspace.last_attempt().is_none());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct KrylovRecycleWorkspaceRequest {
    rows: usize,
    cols: usize,
    provider: usize,
    options: crate::LinalgExecutionOptions,
}
impl KrylovRecycleWorkspaceRequest {
    /// Creates a request for refreshing a space against `a` under `options`.
    ///
    /// # Errors
    ///
    /// Propagates the error of `a.apply_workspace_len()`.
    pub fn new<S: NumericScalar>(
        a: &dyn LinearOperator<S>,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Self> {
        Ok(Self {
            rows: a.rows(),
            cols: a.cols(),
            provider: a.apply_workspace_len()?,
            options: *options,
        })
    }
}
impl<S: NumericScalar> KrylovRecycleSpace<S> {
    /// Returns the storage [`Self::refresh`] needs for `request`, without applying the operator.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the operator of `request` is not square of the
    ///   space's dimension.
    /// - `SizeOverflow` if a size or byte count overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the byte limit of the request's options.
    pub fn workspace_requirements(
        &self,
        request: &KrylovRecycleWorkspaceRequest,
    ) -> Result<KrylovRecycleRequirements<S>> {
        crate::checked::expect_shape(
            [self.dimension, self.dimension],
            [request.rows, request.cols],
        )?;
        let n = self.dimension;
        let count = self.capacity();
        let provider = request.provider;
        let payload = sum([
            bytes::<S>(sum([
                product(&[4, n, count])?,
                product(&[2, count])?,
                product(&[3, n])?,
                provider,
            ])?)?,
            bytes::<Vector<S>>(product(&[4, count])?)?,
        ])?;
        KrylovRecycleRequirements::new(
            RecycleLayout { n, count, provider },
            request.options,
            payload,
            Default::default(),
            Default::default(),
        )
    }
    /// Creates a workspace for [`Self::refresh`] reserved for `request`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if memory cannot be reserved.
    pub fn workspace(
        &self,
        request: &KrylovRecycleWorkspaceRequest,
    ) -> Result<KrylovRecycleWorkspace<S>> {
        let requirements = self.workspace_requirements(request)?;
        let mut w = KrylovRecycleWorkspace::new();
        w.try_reserve(&requirements)?;
        Ok(w)
    }
}
