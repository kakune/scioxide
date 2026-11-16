//! Mixed-precision iterative refinement and power-of-two equilibration.

use crate::workspace::SolveSlot;
use crate::{
    EquilibrationMetadata, LinalgExecutionContext, MatrixView, MatrixViewMut, Vector, VectorView,
    VectorViewMut,
};
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::mem::size_of;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{ScalarKind, checked_mul, try_filled};
use scioxide_core::{
    AttemptOutcome, AttemptRecord, Budget, Capacity, Complex, EvaluationRole, ExecutionContext,
    InvalidInputReason, LinearOperator, NumericScalar, PartialResultTrust, ProblemToken,
    RealScalar, ResidualTolerance, SciOxideError, SolveBudgets, SolveContext, SolveIdentity,
};
use scioxide_core::{Scaled, ScaledAccumulator, ScaledValue};
use scioxide_sparse::CsrMatrix;

type Result<T> = core::result::Result<T, SciOxideError>;

fn scaled<S: NumericScalar>(value: S) -> Result<ScaledValue<scioxide_core::Complex<S::Real>>> {
    ScaledValue::from_value(Complex::new(
        value.real_component(),
        value.imaginary_component(),
    ))
}
fn magnitude<S: NumericScalar>(value: S) -> Result<Scaled<S::Real>> {
    if value.imaginary_component() == S::Real::ZERO {
        Ok(Scaled::from_value(value.real_component().abs())?)
    } else {
        scaled(value)?.magnitude()
    }
}
fn maximum<R: RealScalar>(a: Scaled<R>, b: Scaled<R>) -> Scaled<R> {
    if a.abs_le(b) { b } else { a }
}
fn ratio<R: RealScalar>(a: Scaled<R>, b: Scaled<R>) -> Result<R> {
    if a.is_zero() {
        return Ok(R::ZERO);
    }
    if b.is_zero() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "nonzero residual with zero backward-error scale",
        ));
    }
    // Preserve a positive diagnostic if the quotient underflows.
    Ok(a.div(b)?.round()?.max(R::MIN_POSITIVE * R::EPSILON))
}

/// The original matrix `A` of a refinement problem, in the residual precision.
///
/// Residuals and backward errors are always computed with this matrix, never
/// with the low-precision factorization. An operator must keep its shape and
/// scratch length for the lifetime of the [`RefinementProblem`].
#[derive(Clone, Copy)]
pub enum RefinementSystem<'a, S: NumericScalar> {
    /// A dense matrix, including strided or conjugated views.
    Dense(MatrixView<'a, S>),
    /// A compressed sparse row (CSR) matrix; stored zeros are kept.
    Sparse(&'a CsrMatrix<S>),
    /// A matrix-free operator with an upper bound on `‖A‖∞`.
    ///
    /// The normwise backward error uses `norm_bound` in place of `‖A‖∞`, so it
    /// is only as reliable as the bound and the operator's accuracy. The
    /// componentwise backward error is not available.
    Operator {
        /// The operator `A`, in the original (unequilibrated) coordinates.
        operator: &'a (dyn LinearOperator<S> + Sync),
        /// A finite positive bound on `‖A‖∞`; refinement does not check it.
        norm_bound: S::Real,
    },
}
impl<S: NumericScalar> RefinementSystem<'_, S> {
    fn shape(self) -> (usize, usize) {
        match self {
            Self::Dense(a) => (a.rows(), a.cols()),
            Self::Sparse(a) => (a.rows(), a.cols()),
            Self::Operator { operator, .. } => (operator.rows(), operator.cols()),
        }
    }
    fn scratch(self) -> Result<usize> {
        match self {
            Self::Operator { operator, .. } => operator.apply_workspace_len(),
            _ => Ok(0),
        }
    }
    fn row(self, row: usize, mut visit: impl FnMut(usize, S) -> Result<()>) -> Result<()> {
        match self {
            Self::Dense(a) => (0..a.cols()).try_for_each(|col| visit(col, a.get_element(row, col))),
            Self::Sparse(a) => (a.row_ptr()[row]..a.row_ptr()[row + 1])
                .try_for_each(|i| visit(a.col_indices()[i], a.values()[i])),
            Self::Operator { .. } => Err(crate::checked::unsupported(
                scioxide_core::argument!("self"),
                "matrix entries unavailable for opaque operator",
            )),
        }
    }
    fn validate(self) -> Result<()> {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(rows, rows),
                scioxide_core::Dimensions::two(rows, cols),
            ));
        }
        match self {
            Self::Operator { norm_bound, .. } => {
                crate::checked::finite(
                    norm_bound,
                    scioxide_core::NonFiniteOrigin::Input,
                    "operator norm bound must be finite",
                )?;
                if norm_bound <= S::Real::ZERO {
                    return Err(SciOxideError::invalid_input(
                        scioxide_core::argument!("system"),
                        InvalidInputReason::OutOfRange,
                        "operator norm bound must be positive",
                    ));
                }
                Ok(())
            }
            _ => (0..rows).try_for_each(|row| {
                self.row(row, |_, value| {
                    crate::checked::finite(
                        value,
                        scioxide_core::NonFiniteOrigin::Input,
                        "values must be finite",
                    )
                    .map(|_| ())
                })
            }),
        }
    }
}

/// Backward error that decides convergence of refinement.
///
/// Both are computed for the original system, with `r = b − A x`:
///
/// ```text
/// normwise:       η = ‖r‖∞ / (‖A‖∞ ‖x‖∞ + ‖b‖∞)
/// componentwise:  ω = maxᵢ |rᵢ| / (|A| |x| + |b|)ᵢ
/// ```
///
/// `η` is the smallest normwise relative perturbation of `A` and `b` for which
/// `x` is an exact solution (Rigal and Gaches), and `ω` the smallest
/// componentwise one (Oettli and Prager); see Higham, *Accuracy and Stability
/// of Numerical Algorithms*, 2nd ed., §7.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinementCriterion {
    /// The normwise backward error `η`.
    Normwise,
    /// The componentwise backward error `ω`; requires a dense or sparse matrix.
    Componentwise,
}
/// Whether refinement may switch to an inverse in the residual precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RefinementFallback {
    /// Never switch; a failure of the low-precision inverse ends refinement.
    #[default]
    Disabled,
    /// Switch once to the fallback inverse of the [`RefinementProblem`] after a
    /// conversion failure, a numerical failure of the inverse, or stagnation.
    OnFailure,
}
/// Why refinement switched from the low-precision inverse to the fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinementFallbackReason {
    /// A residual component that is not negligible could not be represented in
    /// the factor precision, or a correction overflowed the residual precision.
    Conversion,
    /// The low-precision inverse returned a `NumericalInstability` error.
    NumericalFailure,
    /// A correction did not decrease the selected backward error.
    Stagnation,
}
/// Reason refinement of one right-hand side stopped.
///
/// Only `Converged` is successful.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinementTermination {
    /// The selected backward error met the tolerance.
    Converged,
    /// A correction did not decrease the selected backward error.
    Stagnation,
    /// The limit on corrections per right-hand side was reached.
    IterationLimit,
    /// The evaluation limit prevented a residual or inverse application.
    EvaluationLimit,
    /// Cancellation or a deadline stopped refinement.
    ExecutionStopped(scioxide_core::ExecutionStopReason),
    /// Conversion to the factor precision or the low-precision correction failed,
    /// and no fallback was allowed.
    CorrectionFailed,
}
/// Tolerance, stopping criterion, fallback policy and execution options of refinement.
///
/// Refinement converges when the selected backward error is at most
/// `atol + rtol`, since the backward error is already relative. The iteration
/// budget of the [`LinalgExecutionOptions`](crate::LinalgExecutionOptions) limits
/// the corrections per right-hand side, including fallback corrections; the
/// default is ten. Its byte limit bounds the workspace, and its policy,
/// accuracy and time limit apply to every refinement.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, ResidualTolerance};
/// use scioxide_linalg::{LinalgExecutionOptions, RefinementCriterion, RefinementOptions};
///
/// let options = RefinementOptions::new(ResidualTolerance::new(0.0, 1e-14)?)
///     .with_criterion(RefinementCriterion::Componentwise)
///     .with_execution_options(
///         LinalgExecutionOptions::new()
///             .with_iteration_budget(Budget::ONE.saturating_add(3))
///             .with_byte_limit(1 << 16),
///     );
/// assert_eq!(options.execution_options().byte_limit(), Some(1 << 16));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct RefinementOptions<R: RealScalar> {
    execution: crate::LinalgExecutionOptions,
    tolerance: ResidualTolerance<R>,
    criterion: RefinementCriterion,
    fallback: RefinementFallback,
}
impl<R: RealScalar> RefinementOptions<R> {
    /// Creates options with tolerance `tolerance`, the normwise criterion, no
    /// fallback and default execution options.
    pub const fn new(tolerance: ResidualTolerance<R>) -> Self {
        Self {
            execution: crate::LinalgExecutionOptions::new(),
            tolerance,
            criterion: RefinementCriterion::Normwise,
            fallback: RefinementFallback::Disabled,
        }
    }
    /// Returns the execution options.
    pub const fn execution_options(&self) -> crate::LinalgExecutionOptions {
        self.execution
    }
    /// Sets the execution options: policy, product accuracy, time limit, byte
    /// limit and iteration budget.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }
    /// Sets the backward-error tolerance.
    pub const fn with_tolerance(mut self, tolerance: ResidualTolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Sets the backward error that decides convergence.
    pub const fn with_criterion(mut self, criterion: RefinementCriterion) -> Self {
        self.criterion = criterion;
        self
    }
    /// Sets the fallback policy.
    pub const fn with_fallback(mut self, fallback: RefinementFallback) -> Self {
        self.fallback = fallback;
        self
    }
    /// Returns the backward-error tolerance.
    ///
    /// Only the backward error decides convergence; a small correction does not.
    pub const fn tolerance(&self) -> ResidualTolerance<R> {
        self.tolerance
    }
    /// Returns the backward error that decides convergence.
    pub const fn criterion(&self) -> RefinementCriterion {
        self.criterion
    }
    /// Returns the fallback policy.
    pub const fn fallback(&self) -> RefinementFallback {
        self.fallback
    }
    /// Corrections per right-hand side: the execution iteration budget, ten by
    /// default.
    fn corrections(&self) -> Budget {
        self.execution
            .iteration_budget()
            .unwrap_or(Budget::ONE.saturating_add(9))
    }
}
/// Backward errors `η` and `ω` of an iterate, as defined on [`RefinementCriterion`].
///
/// They are floating-point estimates for the original system, not rigorous
/// bounds. They bound the forward error only together with a condition number.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{BackwardError, RefinementReport};
///
/// fn normwise_error(report: &RefinementReport<f64>) -> Option<f64> {
///     report.final_error().map(BackwardError::normwise)
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct BackwardError<R: RealScalar> {
    /// Normwise backward error `η`.
    normwise: R,
    /// Componentwise backward error `ω`.
    componentwise: Option<R>,
    /// Whether `η` uses the operator bound in place of `‖A‖∞`.
    uses_operator_bound: bool,
}
impl<R: RealScalar> BackwardError<R> {
    /// Returns the normwise backward error `η`.
    ///
    /// For [`RefinementSystem::Operator`] it uses the supplied bound in place
    /// of `‖A‖∞`.
    pub const fn normwise(self) -> R {
        self.normwise
    }
    /// Returns the componentwise backward error `ω`; `None` for [`RefinementSystem::Operator`].
    pub const fn componentwise(self) -> Option<R> {
        self.componentwise
    }
    /// Returns whether `η` uses the operator bound in place of `‖A‖∞`.
    pub const fn uses_operator_bound(self) -> bool {
        self.uses_operator_bound
    }

    fn selected(self, criterion: RefinementCriterion) -> Result<R> {
        match criterion {
            RefinementCriterion::Normwise => Ok(self.normwise),
            RefinementCriterion::Componentwise => self.componentwise.ok_or_else(|| {
                crate::checked::unsupported(
                    scioxide_core::argument!("self"),
                    "componentwise error requires matrix entries",
                )
            }),
        }
    }
}
/// Termination status and diagnostics of refinement for one right-hand side.
///
/// An unsuccessful refinement returns the best iterate it accepted.
/// Cancellation and deadlines give the status `ExecutionStopped(reason)`. A
/// backward error is `None` when no residual was computed for that iterate.
#[derive(Clone, Copy, Debug)]
pub struct RefinementReport<R: RealScalar> {
    /// Termination status.
    pub(crate) termination: RefinementTermination,
    /// Attempted corrections, including unsuccessful attempts.
    pub(crate) iterations: usize,
    /// Backward error of the initial guess.
    pub(crate) initial_error: Option<BackwardError<R>>,
    /// Backward error of the returned iterate.
    pub(crate) final_error: Option<BackwardError<R>>,
    /// Reason for switching to the fallback, if a switch occurred.
    pub(crate) fallback_reason: Option<RefinementFallbackReason>,
    /// Applications of the fallback inverse.
    pub(crate) fallback_iterations: usize,
    /// Scalar representation used by the primary inverse.
    pub(crate) factor_precision: ScalarKind,
    /// Scalar representation used for residuals and solution updates.
    pub(crate) residual_precision: ScalarKind,
}

impl<R: RealScalar> RefinementReport<R> {
    /// Returns the termination status.
    pub const fn termination(&self) -> RefinementTermination {
        self.termination
    }

    /// Returns the number of attempted corrections, including rejected ones.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the backward errors of the initial guess.
    pub const fn initial_error(&self) -> Option<BackwardError<R>> {
        self.initial_error
    }

    /// Returns the backward errors of the returned iterate.
    pub const fn final_error(&self) -> Option<BackwardError<R>> {
        self.final_error
    }

    /// Returns why refinement switched to the fallback inverse, if it did.
    pub const fn fallback_reason(&self) -> Option<RefinementFallbackReason> {
        self.fallback_reason
    }

    /// Returns the number of fallback-inverse applications; zero if a limit prevented its use.
    pub const fn fallback_iterations(&self) -> usize {
        self.fallback_iterations
    }

    /// Returns the scalar type of the low-precision inverse.
    pub const fn factor_precision(&self) -> ScalarKind {
        self.factor_precision
    }

    /// Returns the scalar type of residuals and solution updates.
    pub const fn residual_precision(&self) -> ScalarKind {
        self.residual_precision
    }
}

impl<R: RealScalar> RefinementReport<R> {
    /// Returns whether the termination is `Converged`.
    pub fn is_success(&self) -> bool {
        self.termination == RefinementTermination::Converged
    }
}

/// Sizes of a refinement problem and the number of right-hand sides.
///
/// A request holds sizes and flags only, not the matrix or the inverses. See
/// [`IterativeRefinement`] for an example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct RefinementWorkspaceRequest {
    order: usize,
    right_hand_sides: usize,
    low_scratch: usize,
    high_scratch: usize,
    residual_scratch: usize,
    fallback: bool,
    opaque: bool,
}
impl RefinementWorkspaceRequest {
    /// Creates a request for one right-hand side of `problem`.
    ///
    /// The scratch lengths are those recorded when `problem` was created.
    pub fn new<L: NumericScalar, H: NumericScalar>(problem: &RefinementProblem<'_, L, H>) -> Self {
        Self {
            order: problem.order,
            right_hand_sides: 1,
            low_scratch: problem.low_scratch,
            high_scratch: problem.high_scratch,
            residual_scratch: problem.residual_scratch,
            fallback: problem.fallback.is_some(),
            opaque: matches!(problem.system, RefinementSystem::Operator { .. }),
        }
    }
    /// Sets the number of right-hand sides to `count`.
    pub const fn with_right_hand_sides(mut self, count: usize) -> Self {
        self.right_hand_sides = count;
        self
    }
    /// Returns the order `n` of the system.
    pub const fn order(&self) -> usize {
        self.order
    }
    /// Returns the number of right-hand sides.
    pub const fn right_hand_sides(&self) -> usize {
        self.right_hand_sides
    }
}

/// Storage a [`IterativeRefinement`] needs for one [`RefinementWorkspaceRequest`], with its byte
/// limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefinementRequirements<L: NumericScalar, H: NumericScalar> {
    order: usize,
    right_hand_sides: usize,
    low_scratch: usize,
    high_scratch: usize,
    residual_scratch: usize,
    bytes: usize,
    byte_limit: Option<usize>,
    scalars: PhantomData<fn() -> (L, H)>,
}
impl<L: NumericScalar, H: NumericScalar> RefinementRequirements<L, H> {
    /// Returns the number of right-hand sides.
    pub const fn right_hand_sides(&self) -> usize {
        self.right_hand_sides
    }
    /// Returns the bytes of numerical data, excluding `Vec` headers and allocator bookkeeping.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Returns the byte limit from the execution options.
    pub const fn byte_limit(&self) -> Option<usize> {
        self.byte_limit
    }
}

/// Reusable scratch storage and attempt record for [`IterativeRefinement::refine_into`].
///
/// A workspace is reserved for one system order and a number of right-hand
/// sides. See [Plans and workspaces](scioxide_core#plans-and-workspaces).
#[derive(Debug)]
pub struct RefinementWorkspace<L: NumericScalar, H: NumericScalar> {
    order: usize,
    capacity: usize,
    staged: Vec<H>,
    rhs: Vec<H>,
    current: Vec<H>,
    candidate: Vec<H>,
    residual: Vec<H>,
    next_residual: Vec<H>,
    correction: Vec<H>,
    low_rhs: Vec<L>,
    low_correction: Vec<L>,
    low_scratch: Vec<L>,
    high_scratch: Vec<H>,
    residual_scratch: Vec<H>,
    reports: Vec<RefinementReport<H::Real>>,
    slot: SolveSlot<()>,
}
impl<L: NumericScalar, H: NumericScalar> Default for RefinementWorkspace<L, H> {
    fn default() -> Self {
        Self::new()
    }
}
impl<L: NumericScalar, H: NumericScalar> RefinementWorkspace<L, H> {
    /// Creates an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            order: 0,
            capacity: 0,
            staged: Vec::new(),
            rhs: Vec::new(),
            current: Vec::new(),
            candidate: Vec::new(),
            residual: Vec::new(),
            next_residual: Vec::new(),
            correction: Vec::new(),
            low_rhs: Vec::new(),
            low_correction: Vec::new(),
            low_scratch: Vec::new(),
            high_scratch: Vec::new(),
            residual_scratch: Vec::new(),
            reports: Vec::new(),
            slot: SolveSlot::new(),
        }
    }
    /// Checks whether the workspace covers `requirements`, without allocating.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the requirements, or the storage the workspace
    ///   holds, exceed the byte limit.
    /// - `SizeOverflow` if a byte count overflows `usize`.
    pub fn check(&self, requirements: &RefinementRequirements<L, H>) -> Result<Capacity> {
        crate::workspace::admit_bytes(
            requirements.bytes,
            requirements.byte_limit,
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        let retained = payload_bytes::<L, H>(
            [
                &self.staged,
                &self.rhs,
                &self.current,
                &self.candidate,
                &self.residual,
                &self.next_residual,
                &self.correction,
                &self.high_scratch,
                &self.residual_scratch,
            ]
            .iter()
            .map(|buffer| buffer.len())
            .try_fold(0_usize, usize::checked_add),
            [&self.low_rhs, &self.low_correction, &self.low_scratch]
                .iter()
                .map(|buffer| buffer.len())
                .try_fold(0_usize, usize::checked_add),
            self.reports.len(),
        )?;
        crate::workspace::admit_bytes(
            retained,
            requirements.byte_limit,
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        Ok(
            if self.order == requirements.order
                && self.capacity >= requirements.right_hand_sides
                && self.low_scratch.len() >= requirements.low_scratch
                && self.high_scratch.len() >= requirements.high_scratch
                && self.residual_scratch.len() >= requirements.residual_scratch
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
    /// Reserves storage for `requirements`; storage that already suffices is kept.
    ///
    /// # Errors
    ///
    /// Workspace errors as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
    /// On error the workspace is unchanged.
    pub fn try_reserve(&mut self, requirements: &RefinementRequirements<L, H>) -> Result<()> {
        if self.check(requirements)? == Capacity::Sufficient {
            return Ok(());
        }
        let n = requirements.order;
        let high = |len, quantity| try_filled(len, H::ZERO, quantity);
        let low = |len, quantity| try_filled(len, L::ZERO, quantity);
        let empty = RefinementReport {
            termination: RefinementTermination::IterationLimit,
            iterations: 0,
            initial_error: None,
            final_error: None,
            fallback_reason: None,
            fallback_iterations: 0,
            factor_precision: L::kind(),
            residual_precision: H::kind(),
        };
        *self = Self {
            order: n,
            capacity: requirements.right_hand_sides,
            staged: high(
                checked_mul(
                    n,
                    requirements.right_hand_sides,
                    scioxide_core::quantity!("workspace.staged"),
                )?,
                scioxide_core::quantity!("workspace.staged"),
            )?,
            rhs: high(n, scioxide_core::quantity!("workspace.rhs"))?,
            current: high(n, scioxide_core::quantity!("workspace.current"))?,
            candidate: high(n, scioxide_core::quantity!("workspace.candidate"))?,
            residual: high(n, scioxide_core::quantity!("workspace.residual"))?,
            next_residual: high(n, scioxide_core::quantity!("workspace.next_residual"))?,
            correction: high(n, scioxide_core::quantity!("workspace.correction"))?,
            low_rhs: low(n, scioxide_core::quantity!("workspace.low_rhs"))?,
            low_correction: low(n, scioxide_core::quantity!("workspace.low_correction"))?,
            low_scratch: low(
                requirements.low_scratch,
                scioxide_core::quantity!("workspace.low_scratch"),
            )?,
            high_scratch: high(
                requirements.high_scratch,
                scioxide_core::quantity!("workspace.high_scratch"),
            )?,
            residual_scratch: high(
                requirements.residual_scratch,
                scioxide_core::quantity!("workspace.residual_scratch"),
            )?,
            reports: try_filled(
                requirements.right_hand_sides,
                empty,
                scioxide_core::quantity!("workspace.reports"),
            )?,
            slot: core::mem::take(&mut self.slot),
        };
        Ok(())
    }
    /// Returns the record of the most recent refinement call.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<()>> {
        self.slot.last_attempt()
    }
}

/// Payload bytes of `high` residual-precision scalars, `low` factor-precision
/// scalars and `reports` per-column reports.
fn payload_bytes<L: NumericScalar, H: NumericScalar>(
    high: Option<usize>,
    low: Option<usize>,
    reports: usize,
) -> Result<usize> {
    high.and_then(|high| high.checked_mul(size_of::<H>()))
        .zip(low.and_then(|low| low.checked_mul(size_of::<L>())))
        .and_then(|(high, low)| high.checked_add(low))
        .and_then(|bytes| {
            bytes.checked_add(reports.checked_mul(size_of::<RefinementReport<H::Real>>())?)
        })
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("workspace_bytes")))
}

/// Matrix, inverses and optional equilibration of one refinement problem.
///
/// The low-precision inverse applies `(Dr A Dc)⁻¹` when equilibration is given
/// and `A⁻¹` otherwise; you factor the matrix yourself, for example with
/// [`LuDecomposition`](crate::LuDecomposition) in `f32`. The optional fallback
/// applies `A⁻¹` in the residual precision. Creating the problem records the
/// shapes and scratch lengths of the inverses; they must not change while the
/// problem is in use. See [`IterativeRefinement`] for an example.
#[derive(Clone, Copy)]
pub struct RefinementProblem<'a, L: NumericScalar, H: NumericScalar> {
    system: RefinementSystem<'a, H>,
    inverse: &'a (dyn LinearOperator<L> + Sync),
    fallback: Option<&'a (dyn LinearOperator<H> + Sync)>,
    scaling: Option<&'a EquilibrationMetadata>,
    order: usize,
    low_scratch: usize,
    high_scratch: usize,
    residual_scratch: usize,
}
impl<L: NumericScalar, H: NumericScalar> core::fmt::Debug for RefinementProblem<'_, L, H> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RefinementProblem")
            .field("order", &self.order)
            .field("scaling", &self.scaling)
            .field("has_fallback", &self.fallback.is_some())
            .finish_non_exhaustive()
    }
}
impl<'a, L: NumericScalar, H: NumericScalar> RefinementProblem<'a, L, H> {
    /// Creates a problem from the matrix `system`, the low-precision `inverse`,
    /// optional `scaling` and an optional residual-precision `fallback`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `system` is not square, or if `inverse`,
    ///   `fallback` or `scaling` does not match its order.
    /// - `NonFinite` if a matrix entry or the operator norm bound is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if the operator norm bound is not positive.
    /// - Propagates the errors of the inverses' scratch-length queries.
    pub fn new(
        system: RefinementSystem<'a, H>,
        inverse: &'a (dyn LinearOperator<L> + Sync),
        scaling: Option<&'a EquilibrationMetadata>,
        fallback: Option<&'a (dyn LinearOperator<H> + Sync)>,
    ) -> Result<Self> {
        system.validate()?;
        let (n, _) = system.shape();
        crate::checked::expect_shape([n, n], [inverse.rows(), inverse.cols()])?;
        if let Some(fallback) = fallback {
            crate::checked::expect_shape([n, n], [fallback.rows(), fallback.cols()])?;
        }
        if let Some(scaling) = scaling {
            crate::checked::expect_shape(
                [n, n],
                [
                    scaling.row_exponents().len(),
                    scaling.column_exponents().len(),
                ],
            )?;
        }
        Ok(Self {
            system,
            inverse,
            fallback,
            scaling,
            order: n,
            low_scratch: inverse.apply_workspace_len()?,
            high_scratch: fallback.map_or(Ok(0), LinearOperator::apply_workspace_len)?,
            residual_scratch: system.scratch()?,
        })
    }
    /// Reject provider shape changes as `DimensionMismatch` and scratch-length
    /// changes as `InvalidInput`, before any application.
    fn validate_providers(&self) -> Result<()> {
        let n = self.order;
        let (rows, cols) = self.system.shape();
        crate::checked::expect_shape([n, n], [rows, cols])?;
        crate::checked::expect_shape([n, n], [self.inverse.rows(), self.inverse.cols()])?;
        if let Some(fallback) = self.fallback {
            crate::checked::expect_shape([n, n], [fallback.rows(), fallback.cols()])?;
        }
        if self.inverse.apply_workspace_len()? != self.low_scratch
            || self.system.scratch()? != self.residual_scratch
            || self
                .fallback
                .map_or(Ok(0), LinearOperator::apply_workspace_len)?
                != self.high_scratch
        {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("problem"),
                "provider workspace sizes changed after preparation",
            ));
        }
        Ok(())
    }
}

/// Mixed-precision iterative refinement of `A x = b` with a low-precision inverse.
///
/// `L` is the factor precision and `H` the residual precision. Supported pairs
/// are `f32`/`f32`, `f64`/`f64` and `f32`/`f64`, and their complex counterparts.
/// Each step computes
///
/// ```text
/// r = b − A x                 in H (rounded once for dense and sparse A)
/// d = (Dr A Dc)⁻¹ (Dr r)      in L, then d ← Dc d widened to H
/// x ← x + d                   in H
/// ```
///
/// where `Dr`, `Dc` are the power-of-two equilibration of the problem, or the
/// identity. Before narrowing to `L`, `Dr r` is scaled by a power of two near
/// its infinity norm. A step is kept only if it meets the tolerance or
/// decreases the selected backward error; under the componentwise criterion, a
/// step after which narrowing omitted negligible residual components is also
/// kept when `ω` is unchanged and `‖r‖∞` decreases. Otherwise refinement
/// switches once to the fallback inverse when [`RefinementFallback::OnFailure`]
/// allows it, and stops with [`RefinementTermination::Stagnation`] if not.
/// Refinement converges when `A` is not too ill-conditioned with
/// respect to the factor precision. With an `f32` LU factorization and `f64`
/// residuals it can then reach `f64` backward error at the cost of an `f32`
/// factorization plus one residual and one application of the inverse per step.
///
/// The method value holds only options and can be used with many problems. The
/// [`RefinementProblem`] is passed to each call.
///
/// # Examples
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
///
/// # References
///
/// N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
/// SIAM, 2002, Chapter 12.
#[derive(Clone, Debug)]
pub struct IterativeRefinement<L: NumericScalar, H: NumericScalar> {
    options: RefinementOptions<H::Real>,
    scalars: PhantomData<fn() -> (L, H)>,
}
impl<L: NumericScalar, H: NumericScalar> IterativeRefinement<L, H> {
    /// Checks `options` and returns the method value.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if `(L, H)` is not one of
    ///   the supported precision pairs.
    pub fn new(options: RefinementOptions<H::Real>) -> Result<Self> {
        use ScalarKind::{Complex32, Complex64, F32, F64};
        if !matches!(
            (L::kind(), H::kind()),
            (F32, F32)
                | (F64, F64)
                | (F32, F64)
                | (Complex32, Complex32)
                | (Complex64, Complex64)
                | (Complex32, Complex64)
        ) {
            return Err(crate::checked::unsupported(
                scioxide_core::argument!("inverse"),
                "unsupported factor/residual scalar pair",
            ));
        }
        Ok(Self {
            options,
            scalars: PhantomData,
        })
    }
    fn validate_problem(&self, problem: &RefinementProblem<'_, L, H>) -> Result<()> {
        problem.validate_providers()?;
        self.validate_request(&RefinementWorkspaceRequest::new(problem))
    }
    fn validate_request(&self, request: &RefinementWorkspaceRequest) -> Result<()> {
        if self.options.fallback == RefinementFallback::OnFailure && !request.fallback {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("fallback"),
                "fallback policy requires a residual-precision inverse",
            ));
        }
        if self.options.criterion == RefinementCriterion::Componentwise && request.opaque {
            return Err(crate::checked::unsupported(
                scioxide_core::argument!("options"),
                "componentwise stopping requires explicit matrix entries",
            ));
        }
        Ok(())
    }
    /// Returns the storage a refinement of `request` needs.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Inconsistent`) if the fallback policy is
    ///   [`RefinementFallback::OnFailure`] and the problem has no fallback inverse.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if the criterion is
    ///   componentwise and the system is a [`RefinementSystem::Operator`].
    /// - `SizeOverflow` if a size or byte count overflows `usize`.
    /// - `ResourceLimit` if the storage exceeds the byte limit of the options.
    pub fn workspace_requirements(
        &self,
        request: &RefinementWorkspaceRequest,
    ) -> Result<RefinementRequirements<L, H>> {
        self.validate_request(request)?;
        let (n, right_hand_sides) = (request.order, request.right_hand_sides);
        // Staged columns plus rhs, current, candidate, residual, next residual
        // and correction; the factor precision holds the narrowed pair.
        let high = n
            .checked_mul(right_hand_sides)
            .zip(n.checked_mul(6))
            .and_then(|(staged, vectors)| staged.checked_add(vectors))
            .and_then(|v| v.checked_add(request.high_scratch))
            .and_then(|v| v.checked_add(request.residual_scratch));
        let low = n
            .checked_mul(2)
            .and_then(|v| v.checked_add(request.low_scratch));
        let bytes = payload_bytes::<L, H>(high, low, right_hand_sides)?;
        let byte_limit = self.options.execution.byte_limit();
        crate::workspace::admit_bytes(
            bytes,
            byte_limit,
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        Ok(RefinementRequirements {
            order: n,
            right_hand_sides,
            low_scratch: request.low_scratch,
            high_scratch: request.high_scratch,
            residual_scratch: request.residual_scratch,
            bytes,
            byte_limit,
            scalars: PhantomData,
        })
    }

    /// Creates a workspace reserved for `request`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if memory cannot be reserved.
    pub fn workspace(
        &self,
        request: &RefinementWorkspaceRequest,
    ) -> Result<RefinementWorkspace<L, H>> {
        let mut workspace = RefinementWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }
    /// Refines the initial guess `initial` for one right-hand side and returns the result.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::refine_into`].
    /// - `AllocationFailure` if the output or workspace cannot be allocated.
    pub fn refine<'r, 'i>(
        &self,
        problem: &RefinementProblem<'_, L, H>,
        rhs: impl Into<VectorView<'r, H>>,
        initial: impl Into<VectorView<'i, H>>,
    ) -> Result<(Vector<H>, RefinementReport<H::Real>)> {
        let initial = initial.into();
        let mut solution = Vector::from_vec(try_filled(
            initial.len(),
            H::ZERO,
            scioxide_core::quantity!("solution"),
        )?);
        solution
            .as_mut_slice()
            .iter_mut()
            .zip(initial.iter())
            .for_each(|(out, value)| *out = value);
        let mut workspace = self.workspace(&RefinementWorkspaceRequest::new(problem))?;
        let report = self.refine_into(
            problem,
            rhs.into(),
            solution.view_mut(),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?[0];
        Ok((solution, report))
    }
    /// Refines each column of `solution` against the same column of `rhs`; one report per column.
    ///
    /// The columns of `solution` hold the initial guesses and are refined
    /// independently; a vector counts as a one-column matrix. The workspace
    /// grows to `rhs.cols()` columns if needed. Pass an `ExecutionContext` for
    /// a standalone call, or a solve context to count residual and inverse
    /// applications toward the parent's limits (see
    /// [Solve accounting](scioxide_core#solve-accounting)). An unsuccessful
    /// refinement, including cancellation and deadlines, returns `Ok` with the
    /// best accepted iterates and the reports.
    ///
    /// Scaling `Dr r` by a power of two adds no rounding. Narrowing to the
    /// factor precision drops only components below `ε_L ‖Dr r‖∞`, where `ε_L`
    /// is the machine epsilon of `L`. The widened correction is rounded once to
    /// `H`, with gradual underflow; a component beyond the range of `H` is a
    /// [`RefinementFallbackReason::Conversion`] failure. For dense and sparse
    /// matrices, the residual and both backward errors come from one evaluation
    /// with exact products.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rhs` or `solution` does not have `n` rows, or
    ///   if their column counts differ.
    /// - `NonFinite` if `rhs` or `solution` contains NaN or infinity, or if an
    ///   operator returns one.
    /// - `InvalidInput` (`Inconsistent`) if a scratch length of the problem's
    ///   inverses changed since the problem was created.
    /// - The errors of [`Self::workspace_requirements`].
    /// - `AllocationFailure` if the workspace cannot grow.
    /// - Errors returned by the operators, other than those that select a fallback.
    /// - `NumericalInstability` if a backward error cannot be formed, for
    ///   example a nonzero residual with a zero denominator.
    pub fn refine_into<'w, 'r, 'x, 's, 'c>(
        &self,
        problem: &RefinementProblem<'_, L, H>,
        rhs: impl Into<MatrixView<'r, H>>,
        solution: impl Into<MatrixViewMut<'x, H>>,
        workspace: &'w mut RefinementWorkspace<L, H>,
        context: impl Into<LinalgExecutionContext<'s, 'c>>,
    ) -> Result<&'w [RefinementReport<H::Real>]> {
        workspace.slot.reject();
        let context = context.into().with_options(&self.options.execution)?;
        let (rhs, mut solution) = (rhs.into(), solution.into());
        let (n, columns) = (problem.system.shape().0, rhs.cols());
        if rhs.rows() != n || solution.rows() != n || solution.cols() != columns {
            let actual = if rhs.rows() != n {
                scioxide_core::Dimensions::two(rhs.rows(), columns)
            } else {
                scioxide_core::Dimensions::two(solution.rows(), solution.cols())
            };
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(n, columns),
                actual,
            ));
        }
        self.validate_problem(problem)?;
        (0..columns)
            .flat_map(|col| (0..n).map(move |row| (row, col)))
            .flat_map(|(row, col)| [rhs.get_element(row, col), solution.get_element(row, col)])
            .try_for_each(|value| {
                crate::checked::finite(
                    value,
                    scioxide_core::NonFiniteOrigin::Input,
                    "values must be finite",
                )
                .map(|_| ())
            })?;
        workspace.try_reserve(&self.workspace_requirements(
            &RefinementWorkspaceRequest::new(problem).with_right_hand_sides(columns),
        )?)?;
        let mut slot = core::mem::take(&mut workspace.slot);
        let kernels = context.kernel();
        let outcome = slot.run(
            context.into_scope(),
            SolveIdentity::new(
                ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "refinement", 1),
                n as u128,
                0,
            ),
            SolveBudgets::unbounded(),
            |solve| {
                let w = &mut *workspace;
                for col in 0..columns {
                    for row in 0..n {
                        w.rhs[row] = rhs.get_element(row, col);
                        w.current[row] = solution.get_element(row, col);
                    }
                    w.reports[col] = self.run(problem, w, solve, kernels)?;
                    w.staged[col * n..(col + 1) * n].copy_from_slice(&w.current);
                }
                let success = w.reports[..columns]
                    .iter()
                    .all(RefinementReport::is_success);
                solve.commit_accepted(if success {
                    PartialResultTrust::ResidualChecked
                } else {
                    PartialResultTrust::DiagnosticOnly
                })?;
                Ok(success)
            },
            |success| {
                let Ok(success) = success else {
                    return (AttemptOutcome::Failed, None);
                };
                (
                    if *success {
                        AttemptOutcome::Succeeded
                    } else {
                        AttemptOutcome::Stopped
                    },
                    None,
                )
            },
        );
        workspace.slot = slot;
        outcome?;
        for col in 0..columns {
            for row in 0..n {
                solution.set_element(row, col, workspace.staged[col * n + row]);
            }
        }
        Ok(&workspace.reports[..columns])
    }
    fn evaluate(
        &self,
        problem: &RefinementProblem<'_, L, H>,
        x: &[H],
        b: &[H],
        residual: &mut [H],
        scratch: &mut [H],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, H>,
    ) -> Result<Option<ResidualEvaluation<H::Real>>> {
        if !context
            .record_provider(EvaluationRole::Residual)?
            .is_granted()
        {
            return Ok(None);
        }
        residual_error(
            problem.system,
            (x, b),
            residual,
            scratch,
            self.options.execution.execution_policy(),
            context,
        )
        .map(Some)
    }
    fn converged(&self, error: BackwardError<H::Real>) -> Result<bool> {
        self.options
            .tolerance
            .is_satisfied(error.selected(self.options.criterion)?, H::Real::ONE)
    }
    fn run(
        &self,
        problem: &RefinementProblem<'_, L, H>,
        w: &mut RefinementWorkspace<L, H>,
        context: &mut SolveContext<'_, '_>,
        kernels: crate::KernelContext<'_>,
    ) -> Result<RefinementReport<H::Real>> {
        let mut report = RefinementReport {
            termination: RefinementTermination::IterationLimit,
            iterations: 0,
            initial_error: None,
            final_error: None,
            fallback_reason: None,
            fallback_iterations: 0,
            factor_precision: L::kind(),
            residual_precision: H::kind(),
        };
        let outcome = self.run_column(problem, w, context, kernels, &mut report);
        if let Err(error) = outcome {
            match error.view() {
                scioxide_core::SciOxideErrorView::ExecutionStopped { reason, .. } => {
                    report.termination = RefinementTermination::ExecutionStopped(reason);
                }
                _ => return Err(error),
            }
        }
        Ok(report)
    }
    fn run_column(
        &self,
        problem: &RefinementProblem<'_, L, H>,
        w: &mut RefinementWorkspace<L, H>,
        context: &mut SolveContext<'_, '_>,
        kernels: crate::KernelContext<'_>,
        report: &mut RefinementReport<H::Real>,
    ) -> Result<()> {
        let context = &mut scioxide_core::OperatorContext::with_kernels(context, &kernels);
        if let Some(reason) = context.stop_reason()? {
            report.termination = RefinementTermination::ExecutionStopped(reason);
            return Ok(());
        }
        let Some(mut error) = self.evaluate(
            problem,
            &w.current,
            &w.rhs,
            &mut w.residual,
            &mut w.residual_scratch,
            context,
        )?
        else {
            report.termination = RefinementTermination::EvaluationLimit;
            return Ok(());
        };
        report.initial_error = Some(error.error);
        report.final_error = Some(error.error);
        if self.converged(error.error)? {
            report.termination = RefinementTermination::Converged;
            return Ok(());
        }
        // Mixed-precision iterative refinement: the residual `r = b - A x` is
        // formed in the high (working) precision H (exactly for explicit
        // matrices), the correction is solved with the cheap low-precision
        // factor L on the equilibrated, narrowed residual, then widened,
        // unscaled and added to x. If narrowing fails, the low solve fails,
        // or a step does not reduce the selected error, the loop can switch
        // once to the caller's high-precision fallback inverse (`high`)
        // before reporting stagnation or failure. A step is only committed
        // when it converges or decreases its verified error.
        let mut high = false;
        for _ in 0..self.options.corrections().get() {
            if let Some(reason) = context.stop_reason()? {
                report.termination = RefinementTermination::ExecutionStopped(reason);
                break;
            }
            if !context.record_iteration()?.is_granted() {
                break;
            }
            report.iterations += 1;
            let mut failure = None;
            let mut discarded = false;
            if high {
                if !context
                    .record_provider(EvaluationRole::LinearSolve)?
                    .is_granted()
                {
                    report.termination = RefinementTermination::EvaluationLimit;
                    break;
                }
                report.fallback_iterations += 1;
                problem
                    .fallback
                    .ok_or_else(|| {
                        crate::checked::inconsistent(
                            scioxide_core::argument!("self"),
                            "missing fallback inverse",
                        )
                    })?
                    .apply_into_scoped(
                        &w.residual,
                        &mut w.correction,
                        &mut w.high_scratch[..problem.high_scratch],
                        context,
                    )
                    .and_then(scioxide_core::Invocation::require)?;
                crate::checked::provider_output(&w.correction)?;
            } else {
                let exponent =
                    match narrow_residual::<L, H>(&w.residual, &mut w.low_rhs, problem.scaling) {
                        Ok((exponent, dropped)) => {
                            discarded = dropped;
                            exponent
                        }
                        Err(_) => {
                            failure = Some(RefinementFallbackReason::Conversion);
                            0
                        }
                    };
                if failure.is_none() {
                    if !context
                        .record_provider(EvaluationRole::LinearSolve)?
                        .is_granted()
                    {
                        report.termination = RefinementTermination::EvaluationLimit;
                        break;
                    }
                    match problem
                        .inverse
                        .apply_into_scoped(
                            &w.low_rhs,
                            &mut w.low_correction,
                            &mut w.low_scratch[..problem.low_scratch],
                            &mut scioxide_core::OperatorContext::with_kernels(context, &kernels),
                        )
                        .and_then(scioxide_core::Invocation::require)
                    {
                        Ok(()) => {
                            crate::checked::provider_output(&w.low_correction)?;
                            if !widen_correction(
                                &w.low_correction,
                                exponent,
                                problem.scaling,
                                &mut w.correction,
                            )? {
                                failure = Some(RefinementFallbackReason::Conversion);
                            }
                        }
                        Err(error) => match error.view() {
                            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
                            | scioxide_core::SciOxideErrorView::Singular { .. } => {
                                failure = Some(RefinementFallbackReason::NumericalFailure)
                            }
                            _ => return Err(error),
                        },
                    }
                }
            }
            if let Some(reason) = failure {
                if !high && self.options.fallback == RefinementFallback::OnFailure {
                    high = true;
                    report.fallback_reason = Some(reason);
                    continue;
                }
                report.termination = RefinementTermination::CorrectionFailed;
                break;
            }
            for ((out, value), correction) in
                w.candidate.iter_mut().zip(&w.current).zip(&w.correction)
            {
                let sum = scaled(*value)?.add(scaled(*correction)?)?.round()?;
                *out = H::from_components(sum.re, sum.im);
            }
            if let Some(reason) = context.stop_reason()? {
                report.termination = RefinementTermination::ExecutionStopped(reason);
                break;
            }
            let Some(next) = self.evaluate(
                problem,
                &w.candidate,
                &w.rhs,
                &mut w.next_residual,
                &mut w.residual_scratch,
                context,
            )?
            else {
                report.termination = RefinementTermination::EvaluationLimit;
                break;
            };
            let converged = self.converged(next.error)?;
            let selected = error.error.selected(self.options.criterion)?;
            let next_selected = next.error.selected(self.options.criterion)?;
            // A negligible lane can dominate componentwise error until a later
            // correction. Permit this plateau only when omission occurred and
            // the unrounded residual norm strictly improved.
            let recovering = discarded
                && self.options.criterion == RefinementCriterion::Componentwise
                && next_selected == selected
                && next.raw.abs_cmp(error.raw).is_lt();
            if !converged && next_selected >= selected && !recovering {
                if !high && self.options.fallback == RefinementFallback::OnFailure {
                    high = true;
                    report.fallback_reason = Some(RefinementFallbackReason::Stagnation);
                    continue;
                }
                report.termination = RefinementTermination::Stagnation;
                break;
            }
            core::mem::swap(&mut w.current, &mut w.candidate);
            core::mem::swap(&mut w.residual, &mut w.next_residual);
            error = next;
            report.final_error = Some(error.error);
            if converged {
                report.termination = RefinementTermination::Converged;
                break;
            }
        }
        Ok(())
    }
}

// Normalize Dr*r in extended binary range before the destination-width
// rounding. Only components whose narrowing erases them may be omitted, and
// only below epsilon_low * ||Dr*r||_infinity. Small representable lanes survive.
fn narrow_residual<L: NumericScalar, H: NumericScalar>(
    residual: &[H],
    output: &mut [L],
    scaling: Option<&EquilibrationMetadata>,
) -> Result<(i64, bool)> {
    let mut norm = Scaled::ZERO;
    for (i, value) in residual.iter().enumerate() {
        let exponent = i64::from(scaling.map_or(0, |s| s.row_exponents()[i]));
        let value = ScaledValue::from_parts(
            Complex::new(value.real_component(), value.imaginary_component()),
            exponent,
        )?;
        norm = maximum(norm, value.magnitude()?);
    }
    let exponent = norm.exponent();
    let epsilon = H::Real::try_from_f64_exact(L::Real::EPSILON.try_to_f64_exact()?)?;
    let negligible = norm.mul(Scaled::from_value(epsilon)?)?;
    let mut discarded = false;
    for (i, (value, out)) in residual.iter().zip(output).enumerate() {
        let row_exponent = i64::from(scaling.map_or(0, |s| s.row_exponents()[i]));
        let mut component = |value: H::Real| -> Result<L::Real> {
            let scaled = Scaled::from_parts(value, row_exponent)?;
            let normalized = Scaled::from_parts(value, row_exponent - exponent)?;
            let narrowed = normalized.round_to::<L::Real>()?;
            if !scaled.is_zero() && narrowed == L::Real::ZERO {
                if !scaled.abs_cmp(negligible).is_lt() {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Underflow,
                        "non-negligible scaled residual component is unrepresentable",
                    ));
                }
                discarded = true;
            }
            Ok(narrowed)
        };
        *out = L::from_components(
            component(value.real_component())?,
            component(value.imaginary_component())?,
        );
    }
    Ok((exponent, discarded))
}

// Strict conversion for factor preparation. Residual narrowing and correction
// widening have their own scale-aware range policies.
fn convert<A: NumericScalar, B: NumericScalar>(value: A) -> Result<B> {
    fn component<A: RealScalar, B: RealScalar>(value: A) -> Result<B> {
        let result = B::try_from_f64_lossy(value.try_to_f64_exact()?)?;
        if value != A::ZERO && result == B::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "narrowing erased a nonzero component",
            ));
        }
        Ok(result)
    }
    Ok(B::from_components(
        component(value.real_component())?,
        component(value.imaginary_component())?,
    ))
}
// Undo the residual normalization and column equilibration of a finite low-precision
// correction with one rounding into residual precision. Gradual underflow to signed
// zero is ordinary rounding of an update far below every representable step; `false`
// reports a component beyond the residual-precision range.
fn widen_correction<L: NumericScalar, H: NumericScalar>(
    low: &[L],
    exponent: i64,
    scaling: Option<&EquilibrationMetadata>,
    output: &mut [H],
) -> Result<bool> {
    for (i, (out, value)) in output.iter_mut().zip(low).enumerate() {
        let exponent = exponent + i64::from(scaling.map_or(0, |s| s.column_exponents()[i]));
        let component = |part: L::Real| Scaled::from_parts(part, exponent)?.round_to::<H::Real>();
        match (
            component(value.real_component()),
            component(value.imaginary_component()),
        ) {
            (Ok(re), Ok(im)) => *out = H::from_components(re, im),
            (Err(error), _) | (_, Err(error))
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Overflow,
                        ..
                    }
                ) =>
            {
                return Ok(false);
            }
            (Err(error), _) | (_, Err(error)) => return Err(error),
        }
    }
    Ok(true)
}
fn scale_value<S: NumericScalar>(value: S, exponent: i32) -> Result<S> {
    fn part<R: RealScalar>(value: R, exponent: i32) -> Result<R> {
        let result = value.scalbn(exponent);
        if !result.is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "equilibrated value overflows the scalar range",
            ));
        }
        if value != R::ZERO && result == R::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "equilibrated value underflows to zero",
            ));
        }
        Ok(result)
    }
    Ok(S::from_components(
        part(value.real_component(), exponent)?,
        part(value.imaginary_component(), exponent)?,
    ))
}

struct ResidualEvaluation<R: RealScalar> {
    error: BackwardError<R>,
    raw: Scaled<R>,
}

#[derive(Clone, Copy)]
struct ResidualDiagnostics<R: RealScalar> {
    raw: Scaled<R>,
    matrix_norm: Scaled<R>,
    componentwise: R,
}
impl<R: RealScalar> ResidualDiagnostics<R> {
    const ZERO: Self = Self {
        raw: Scaled::ZERO,
        matrix_norm: Scaled::ZERO,
        componentwise: R::ZERO,
    };
    fn merge(self, other: Self) -> Self {
        Self {
            raw: maximum(self.raw, other.raw),
            matrix_norm: maximum(self.matrix_norm, other.matrix_norm),
            componentwise: self.componentwise.max(other.componentwise),
        }
    }
}

fn explicit_residual<S: NumericScalar>(
    system: RefinementSystem<'_, S>,
    x: &[S],
    b: &[S],
    residual: &mut [S],
    execution: ExecutionContext<'_>,
    policy: scioxide_core::ExecutionPolicy,
) -> Result<ResidualDiagnostics<S::Real>> {
    use core::num::NonZeroUsize;
    use scioxide_core::backend::PartitionSlot;
    const WORKERS: usize = 32;
    if residual.is_empty() {
        return Ok(ResidualDiagnostics::ZERO);
    }
    let plan = execution.plan(
        policy,
        residual.len(),
        scioxide_core::PlanLimits {
            work_per_item: NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: NonZeroUsize::MIN.saturating_add(WORKERS - 1),
        },
    );
    let mut diagnostics = [ResidualDiagnostics::ZERO; WORKERS];
    {
        type Slot<'a, S> = PartitionSlot<(
            &'a mut [S],
            &'a mut ResidualDiagnostics<<S as NumericScalar>::Real>,
        )>;
        let mut slots: [Slot<'_, S>; WORKERS] = core::array::from_fn(|_| PartitionSlot::empty());
        let mut remaining = residual;
        for (part, diagnostic) in plan.partitions().zip(&mut diagnostics) {
            let (chunk, rest) = crate::workspace::split_mut(remaining, part.range().len())?;
            remaining = rest;
            slots[part.worker_index()] = PartitionSlot::new((chunk, diagnostic));
        }
        execution.execute_partitioned(
            plan,
            &slots[..plan.worker_count()],
            &|part, (chunk, diagnostic), _| {
                for (row, out) in part.range().zip(chunk) {
                    let mut sum = scioxide_core::ProductSum::new();
                    sum.add_product(b[row], S::ONE)?;
                    let mut row_norm = ScaledAccumulator::ZERO;
                    let mut denominator = ScaledAccumulator::from_scaled(magnitude(b[row])?);
                    system.row(row, |col, a| {
                        sum.subtract_product(a, x[col])?;
                        let a = magnitude(a)?;
                        row_norm.add_scaled(a)?;
                        denominator.add_scaled(a.mul(magnitude(x[col])?)?)
                    })?;
                    let raw = scioxide_core::backend::product_sum_scaled(&sum)?.magnitude()?;
                    *out = sum.finish()?;
                    *diagnostic = diagnostic.merge(ResidualDiagnostics {
                        raw,
                        matrix_norm: row_norm.total()?,
                        componentwise: ratio(raw, denominator.total()?)?,
                    });
                }
                Ok(())
            },
        )?;
    }
    Ok(diagnostics
        .into_iter()
        .fold(ResidualDiagnostics::ZERO, ResidualDiagnostics::merge))
}

fn residual_error<S: NumericScalar>(
    system: RefinementSystem<'_, S>,
    (x, b): (&[S], &[S]),
    residual: &mut [S],
    scratch: &mut [S],
    policy: scioxide_core::ExecutionPolicy,
    context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
) -> Result<ResidualEvaluation<S::Real>> {
    let mut xnorm = Scaled::ZERO;
    let mut bnorm = Scaled::ZERO;
    for (x, b) in x.iter().zip(b) {
        xnorm = maximum(xnorm, magnitude(*x)?);
        bnorm = maximum(bnorm, magnitude(*b)?);
    }
    let (rnorm, anorm, componentwise) = if let RefinementSystem::Operator {
        operator,
        norm_bound,
    } = system
    {
        if !residual.is_empty() {
            let required = operator.apply_workspace_len()?;
            let scratch = scratch.get_mut(..required).ok_or_else(|| {
                crate::checked::inconsistent(
                    scioxide_core::argument!("system"),
                    "operator workspace requirement changed during refinement",
                )
            })?;
            operator
                .apply_into_scoped(x, residual, scratch, context)
                .and_then(scioxide_core::Invocation::require)?;
            crate::checked::provider_output(residual)?;
        }
        let mut rnorm = Scaled::ZERO;
        for (r, b) in residual.iter_mut().zip(b) {
            let value = scaled(*b)?.sub(scaled(*r)?)?.round()?;
            *r = S::from_components(value.re, value.im);
            rnorm = maximum(rnorm, magnitude(*r)?);
        }
        (rnorm, Scaled::from_value(norm_bound)?, None)
    } else {
        let diagnostics = explicit_residual(system, x, b, residual, context.execution(), policy)?;
        (
            diagnostics.raw,
            diagnostics.matrix_norm,
            Some(diagnostics.componentwise),
        )
    };
    let denominator = anorm.mul(xnorm)?.add(bnorm)?;
    Ok(ResidualEvaluation {
        raw: rnorm,
        error: BackwardError {
            normwise: ratio(rnorm, denominator)?,
            componentwise,
            uses_operator_bound: matches!(system, RefinementSystem::Operator { .. }),
        },
    })
}

impl EquilibrationMetadata {
    /// Computes power-of-two row and column scaling of the entries of `system`.
    ///
    /// Each row is scaled so that its largest entry magnitude lies in `[1, 2)`,
    /// then each column of the row-scaled matrix likewise. A zero row or column
    /// gets exponent zero. The cost is `O(n²)` for dense and `O(n + nnz)` for
    /// sparse matrices, with `O(n)` storage.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) for a [`RefinementSystem::Operator`].
    /// - `DimensionMismatch` if the matrix is not square.
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - `NumericalInstability` (`Overflow` or `Underflow`) if scaling an entry
    ///   would overflow or turn a nonzero entry into zero.
    /// - `AllocationFailure` if the exponent vectors cannot be allocated.
    pub fn prepare<S: NumericScalar>(system: RefinementSystem<'_, S>) -> Result<Self> {
        system.validate()?;
        if matches!(system, RefinementSystem::Operator { .. }) {
            return Err(crate::checked::unsupported(
                scioxide_core::argument!("system"),
                "operator equilibration requires caller-supplied exponents",
            ));
        }
        let n = system.shape().0;
        let mut rows = try_filled(n, 0, scioxide_core::quantity!("row_exponents"))?;
        let mut columns = try_filled(n, 0, scioxide_core::quantity!("column_exponents"))?;
        let mut maxima = try_filled(n, S::Real::ZERO, scioxide_core::quantity!("maxima"))?;
        for (row, exponent) in rows.iter_mut().enumerate() {
            let mut largest = S::Real::ZERO;
            system.row(row, |_, value| {
                largest = largest
                    .max(value.real_component().abs())
                    .max(value.imaginary_component().abs());
                Ok(())
            })?;
            *exponent = equilibration_exponent(largest)?;
            system.row(row, |col, value| {
                let value = scale_value(value, *exponent)?;
                maxima[col] = maxima[col]
                    .max(value.real_component().abs())
                    .max(value.imaginary_component().abs());
                Ok(())
            })?;
        }
        for (exponent, maximum) in columns.iter_mut().zip(maxima) {
            *exponent = equilibration_exponent(maximum)?;
        }
        let result = Self::new(rows, columns);
        for row in 0..n {
            system.row(row, |col, value| {
                result.scale_entry(value, row, col).map(|_| ())
            })?;
        }
        Ok(result)
    }
    /// Creates scaling from given row and column exponents.
    ///
    /// Use it for operators, whose entries [`Self::prepare`] cannot read.
    /// Whether scaled values are representable is checked when the scaling is
    /// applied.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `rows` and `columns` have different lengths.
    /// - `InvalidInput` (`OutOfRange`) if an exponent lies outside `-4096..=4096`.
    pub fn from_exponents(rows: Vec<i32>, columns: Vec<i32>) -> Result<Self> {
        crate::checked::expect_shape([rows.len()], [columns.len()])?;
        for (exponents, argument) in [
            (&rows, scioxide_core::argument!("rows")),
            (&columns, scioxide_core::argument!("columns")),
        ] {
            if exponents.iter().any(|e| !(-4096..=4096).contains(e)) {
                return Err(SciOxideError::invalid_input(
                    argument,
                    InvalidInputReason::OutOfRange,
                    "equilibration exponents must lie in -4096..=4096",
                ));
            }
        }
        Ok(Self::new(rows, columns))
    }
    fn scale_entry<S: NumericScalar>(&self, value: S, row: usize, col: usize) -> Result<S> {
        crate::checked::finite(
            value,
            scioxide_core::NonFiniteOrigin::Input,
            "values must be finite",
        )
        .map(|_| ())?;
        let exponent = self.row_exponents[row]
            .checked_add(self.column_exponents[col])
            .ok_or_else(|| {
                SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::ExponentOutOfRange,
                    "equilibration exponent overflow",
                )
            })?;
        scale_value(value, exponent)
    }
    /// Writes `Dr A Dc` to `output`, converting to the scalar type of `output`.
    ///
    /// Use it to build the matrix that you factor in the low precision.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `source` or `output` does not match the scaling.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if one of the scalar types is
    ///   real and the other complex.
    /// - `NonFinite` if an entry of `source` is NaN or infinite.
    /// - `NumericalInstability` if a scaled or converted entry overflows, or a
    ///   nonzero entry becomes zero.
    pub fn scale_matrix_into<H: NumericScalar, L: NumericScalar>(
        &self,
        source: MatrixView<'_, H>,
        output: &mut MatrixViewMut<'_, L>,
    ) -> Result<()> {
        let shape = [self.row_exponents.len(), self.column_exponents.len()];
        crate::checked::expect_shape(shape, [source.rows(), source.cols()])?;
        crate::checked::expect_shape(shape, [output.rows(), output.cols()])?;
        // Real/complex domain changes are not implicit.
        if matches!(H::kind(), ScalarKind::Complex32 | ScalarKind::Complex64)
            != matches!(L::kind(), ScalarKind::Complex32 | ScalarKind::Complex64)
        {
            return Err(crate::checked::unsupported(
                scioxide_core::argument!("output"),
                "equilibration conversion changes scalar domain",
            ));
        }
        for row in 0..source.rows() {
            for col in 0..source.cols() {
                let _: L = convert(self.scale_entry(source.get_element(row, col), row, col)?)?;
            }
        }
        for row in 0..source.rows() {
            for col in 0..source.cols() {
                output.set_element(
                    row,
                    col,
                    convert(self.scale_entry(source.get_element(row, col), row, col)?)?,
                );
            }
        }
        Ok(())
    }
    /// Writes the values of `Dr A Dc` for a CSR matrix to `output`, in stored order.
    ///
    /// The row pointers and column indices of `source` describe the result;
    /// stored zeros are kept.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `source` does not match the scaling or
    ///   `output.len()` differs from the number of stored entries.
    /// - `InvalidInput` (`UnsupportedConfiguration`) if one of the scalar types is
    ///   real and the other complex.
    /// - `NonFinite` if a stored value is NaN or infinite.
    /// - `NumericalInstability` if a scaled or converted value overflows, or a
    ///   nonzero value becomes zero.
    pub fn scale_sparse_values_into<H: NumericScalar, L: NumericScalar>(
        &self,
        source: &CsrMatrix<H>,
        output: &mut [L],
    ) -> Result<()> {
        crate::checked::expect_shape(
            [self.row_exponents.len(), self.column_exponents.len()],
            [source.rows(), source.cols()],
        )?;
        crate::checked::expect_shape([source.nnz()], [output.len()])?;
        if matches!(H::kind(), ScalarKind::Complex32 | ScalarKind::Complex64)
            != matches!(L::kind(), ScalarKind::Complex32 | ScalarKind::Complex64)
        {
            return Err(crate::checked::unsupported(
                scioxide_core::argument!("output"),
                "equilibration conversion changes scalar domain",
            ));
        }
        for (row, col, value) in source.iter() {
            let _: L = convert(self.scale_entry(value, row, col)?)?;
        }
        for (out, (row, col, value)) in output.iter_mut().zip(source.iter()) {
            *out = convert(self.scale_entry(value, row, col)?)?;
        }
        Ok(())
    }
    /// Writes the scaled right-hand side `Dr b` to `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `source` or `output` does not match the scaling.
    /// - `NonFinite` if `source` contains NaN or infinity.
    /// - `NumericalInstability` if a scaled value overflows or a nonzero value
    ///   becomes zero.
    pub fn scale_rhs_into<S: NumericScalar>(
        &self,
        source: VectorView<'_, S>,
        output: &mut VectorViewMut<'_, S>,
    ) -> Result<()> {
        scale_vector(source, output, &self.row_exponents)
    }
    /// Writes `Dc y` to `output`, mapping a solution `y` of the equilibrated system back.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `source` or `output` does not match the scaling.
    /// - `NonFinite` if `source` contains NaN or infinity.
    /// - `NumericalInstability` if a scaled value overflows or a nonzero value
    ///   becomes zero.
    pub fn unscale_solution_into<S: NumericScalar>(
        &self,
        source: VectorView<'_, S>,
        output: &mut VectorViewMut<'_, S>,
    ) -> Result<()> {
        scale_vector(source, output, &self.column_exponents)
    }
}
fn scale_vector<S: NumericScalar>(
    source: VectorView<'_, S>,
    output: &mut VectorViewMut<'_, S>,
    exponents: &[i32],
) -> Result<()> {
    crate::checked::expect_shape([exponents.len()], [source.len()])?;
    crate::checked::expect_shape([exponents.len()], [output.len()])?;
    for (value, e) in source.iter().zip(exponents) {
        crate::checked::finite(
            value,
            scioxide_core::NonFiniteOrigin::Input,
            "values must be finite",
        )
        .map(|_| ())?;
        scale_value(value, *e)?;
    }
    for ((out, value), e) in output.iter_mut().zip(source.iter()).zip(exponents) {
        *out = scale_value(value, *e)?;
    }
    Ok(())
}

// Power-of-two exponent `e` with `magnitude * 2^e` in [1, 2): the top set bit
// of the integer significand gives `floor(log2(magnitude))` exactly, so no
// floating-point logarithm (and no rounding) is involved.
pub(crate) fn equilibration_exponent<R: RealScalar>(magnitude: R) -> Result<i32> {
    if magnitude == R::ZERO {
        return Ok(0);
    }
    let (significand, exponent, _) = crate::checked::finite_parts(magnitude)?;
    Ok(-(i32::from(exponent) + (u64::BITS - 1 - significand.leading_zeros()) as i32))
}
