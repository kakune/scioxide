//! Toeplitz, Hankel and circulant solves: Levinson–Durbin, GKO, circulant
//! division and an optional dense LU fallback.
use super::constructors::Pattern;
use super::direct::{admit, identity};
use super::{StructuredMatrixView, StructuredOperation, StructuredWorkspaceRequest};
use crate::workspace::ScalarRegion;
use crate::{
    ConditionEstimate, DirectSolveTermination, LinalgExecutionContext, LinalgExecutionOptions,
    LinalgRequirements, LinalgWorkspace, MatrixView, MatrixViewMut, VectorView, VectorViewMut,
};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    EstimateReliability, ExecutionContext, ExecutionPolicy, NumericScalar, RealScalar,
    SciOxideError, Tolerance,
};

/// Options for a [`StructuredSolvePlan`]: backward-error tolerance and execution options.
///
/// A solve is accepted when its backward error `η` (defined on
/// [`StructuredSolvePlan`]) satisfies `η ≤ atol + rtol` for the tolerance
/// `(atol, rtol)`. The execution options apply to preparation and to every
/// solve. The byte limit covers the coefficients stored in the plan, and for
/// a solve those coefficients plus the workspace, including the dense LU
/// fallback. The defaults are [`Tolerance::from_sqrt_epsilon`], which accepts
/// `η ≤ 2√ε`, and [`LinalgExecutionOptions::new`].
///
/// ```
/// use scioxide_core::Tolerance;
/// use scioxide_linalg::{LinalgExecutionOptions, StructuredSolveOptions};
/// let options = StructuredSolveOptions::<f64>::new()
///     .with_tolerance(Tolerance::new(0.0, 1e-12)?)
///     .with_execution_options(LinalgExecutionOptions::new().with_byte_limit(512));
/// assert_eq!(options.execution_options().byte_limit(), Some(512));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct StructuredSolveOptions<R: RealScalar> {
    tolerance: Tolerance<R>,
    execution: LinalgExecutionOptions,
}

impl<R: RealScalar> StructuredSolveOptions<R> {
    /// Returns the execution options.
    pub const fn execution_options(self) -> crate::LinalgExecutionOptions {
        self.execution
    }

    /// Sets the execution options.
    pub const fn with_execution_options(
        mut self,
        execution: crate::LinalgExecutionOptions,
    ) -> Self {
        self.execution = execution;
        self
    }

    /// Returns the default options.
    pub fn new() -> Self {
        Self {
            tolerance: Tolerance::from_sqrt_epsilon(),
            execution: LinalgExecutionOptions::new(),
        }
    }

    /// Sets the backward-error tolerance.
    pub const fn with_tolerance(mut self, tolerance: Tolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Returns the backward-error tolerance.
    pub const fn tolerance(&self) -> Tolerance<R> {
        self.tolerance
    }
}

impl<R: RealScalar> Default for StructuredSolveOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Compact {
    Toeplitz,
    HermitianToeplitz,
    Hankel,
    Circulant,
}

/// Result of a [`StructuredSolvePlan`] solve: acceptance, backward error and methods used.
///
/// With several right-hand sides, the solve is accepted only if every column
/// is accepted. [`backward_error`](Self::backward_error) is the largest over
/// all columns, rejected ones included. The method counts include only
/// accepted columns, even when another column is rejected. An empty system is
/// accepted with zero error and zero counts.
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuredSolveReport<R: RealScalar> {
    termination: DirectSolveTermination,
    backward_error: R,
    levinson: usize,
    gko: usize,
    dense_lu: usize,
    circulant: usize,
}

impl<R: RealScalar> StructuredSolveReport<R> {
    /// Returns [`DirectSolveTermination::Accepted`] or
    /// [`DirectSolveTermination::ResidualRejected`].
    pub const fn termination(self) -> DirectSolveTermination {
        self.termination
    }

    /// Returns the largest backward error `η` over all columns, including rejected ones.
    pub const fn backward_error(self) -> R {
        self.backward_error
    }
    /// Returns the number of accepted columns solved by Levinson–Durbin recursion.
    ///
    /// A `1 × 1` Hermitian Toeplitz system with a positive entry is solved by
    /// one division and counted here.
    pub const fn levinson_solves(self) -> usize {
        self.levinson
    }
    /// Returns the number of accepted columns solved by the GKO algorithm.
    ///
    /// Other `1 × 1` Toeplitz and Hankel systems are solved by one division
    /// and counted here.
    pub const fn gko_solves(self) -> usize {
        self.gko
    }
    /// Returns the number of accepted columns solved by the dense LU fallback.
    pub const fn dense_lu_solves(self) -> usize {
        self.dense_lu
    }
    /// Returns the number of accepted columns solved by circulant division.
    pub const fn circulant_solves(self) -> usize {
        self.circulant
    }

    fn empty() -> Self {
        Self {
            termination: DirectSolveTermination::Accepted,
            backward_error: R::ZERO,
            levinson: 0,
            gko: 0,
            dense_lu: 0,
            circulant: 0,
        }
    }
    fn include(&mut self, other: Self) {
        if !other.termination.is_success() {
            self.termination = other.termination;
        }
        self.backward_error = self.backward_error.max(other.backward_error);
        self.levinson += other.levinson;
        self.gko += other.gko;
        self.dense_lu += other.dense_lu;
        self.circulant += other.circulant;
    }
}

/// Per-column staging distinguishes an unavailable candidate from a measured
/// rejection. Either can select fallback, but only a measured outcome can be
/// returned as a diagnostic report when fallback is unavailable.
#[derive(Clone, Copy)]
enum ColumnOutcome<R: RealScalar> {
    Pending,
    Assessed(StructuredSolveReport<R>),
}

impl<R: RealScalar> ColumnOutcome<R> {
    fn is_accepted(self) -> bool {
        matches!(self, Self::Assessed(report) if report.termination.is_success())
    }

    fn report(self) -> Result<StructuredSolveReport<R>, SciOxideError> {
        match self {
            Self::Assessed(report) => Ok(report),
            Self::Pending => Err(kernels::instability(
                "no verified structured candidate; dense fallback unavailable",
            )),
        }
    }

    // Exact small tags and one residual retain column outcomes without adding
    // allocation or changing the two-slot parallel staging layout.
    fn store<T: NumericScalar<Real = R>>(self, slots: &mut [T]) {
        let Self::Assessed(report) = self else {
            slots[0] = T::ZERO;
            slots[1] = -T::ONE;
            return;
        };
        slots[0] = T::from_real(report.backward_error);
        slots[1] = if !report.termination.is_success() {
            -(T::ONE + T::ONE)
        } else if report.levinson == 1 {
            T::ZERO
        } else if report.dense_lu == 1 {
            T::ONE
        } else if report.gko == 1 {
            T::ONE + T::ONE + T::ONE
        } else {
            T::ONE + T::ONE
        };
    }
    fn load<T: NumericScalar<Real = R>>(slots: &[T]) -> Self {
        if slots[1] == -T::ONE {
            return Self::Pending;
        }
        Self::Assessed(StructuredSolveReport {
            termination: if slots[1] == -(T::ONE + T::ONE) {
                DirectSolveTermination::ResidualRejected
            } else {
                DirectSolveTermination::Accepted
            },
            backward_error: slots[0].real_component(),
            levinson: usize::from(slots[1] == T::ZERO),
            gko: usize::from(slots[1] == T::ONE + T::ONE + T::ONE),
            dense_lu: usize::from(slots[1] == T::ONE),
            circulant: usize::from(slots[1] == T::ONE + T::ONE),
        })
    }
}

/// Toeplitz, Hankel or circulant system `op(A) x = b`, prepared for repeated O(n²) solves.
///
/// `A` is a square [`StructuredMatrixView`] made by
/// [`toeplitz`](StructuredMatrixView::toeplitz),
/// [`hankel`](StructuredMatrixView::hankel) or
/// [`circulant`](StructuredMatrixView::circulant). The plan stores O(n)
/// values: the coefficients of `A` scaled by a power of two, DFT twiddle
/// factors and, for a circulant, its eigenvalues. Each solve chooses `op(A)`
/// with a [`StructuredOperation`].
///
/// The method depends on the structure:
///
/// - Hermitian Toeplitz (real symmetric for real `T`): Levinson–Durbin
///   recursion. If a prediction error falls to `√ε` or below, relative to the
///   largest coefficient, or the recursion produces a non-finite value or a
///   rejected solution, that column is solved by GKO instead. `A` need not be
///   positive definite.
/// - Other Toeplitz and all Hankel matrices: the GKO algorithm. A Toeplitz
///   matrix `T` has displacement rank two with respect to the cyclic shifts
///   `Z₁` and `Z₋₁`: `Z₁ T − T Z₋₁` has rank at most two. Discrete Fourier transforms
///   turn `T` into a Cauchy-like matrix with the same displacement rank,
///   which is factored by Gaussian elimination with partial pivoting. A
///   Hankel matrix is solved as the Toeplitz matrix `A J`, where `J` reverses
///   the columns.
/// - Circulant: division by the eigenvalues `λₖ = Σⱼ cⱼ exp(−2πi jk / n)`
///   of `A`, the DFT of the first column `c`.
///
/// Levinson–Durbin, GKO and circulant division cost O(n²) per right-hand side
/// (the DFTs are evaluated directly, not by an FFT). A workspace reserved
/// with [`StructuredWorkspaceRequest::with_dense_fallback`] also holds a dense
/// LU factorization with partial pivoting. Columns that the O(n²) methods do
/// not solve to the tolerance are then solved with LU, which costs O(n³) and
/// is factored at most once per call. With a parallel execution policy,
/// Levinson–Durbin and circulant columns may run in parallel; GKO and LU run
/// sequentially.
///
/// # Accuracy
///
/// Every accepted solution meets the tolerance in [`StructuredSolveOptions`]
/// for the row-wise backward error
///
/// ```text
/// η = maxᵢ |b − op(A) x|ᵢ / (|bᵢ| + ‖x‖_∞ Σⱼ |op(A)ᵢⱼ|)
/// ```
///
/// where a row with zero denominator contributes zero. This bounds the
/// backward error only. The forward error also depends on the condition of
/// `A`, which [`condition_estimate`](Self::condition_estimate) computes.
/// Coefficients and right-hand sides are scaled by powers of two to avoid
/// overflow.
///
/// # Examples
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
///
/// # References
/// - N. Levinson, "The Wiener RMS (root mean square) error criterion in filter
///   design and prediction", *J. Math. Phys.* 25 (1947), 261–278.
/// - J. Durbin, "The fitting of time-series models", *Rev. Inst. Int. Stat.* 28
///   (1960), 233–244.
/// - I. Gohberg, T. Kailath and V. Olshevsky, "Fast Gaussian elimination with
///   partial pivoting for matrices with displacement structure", *Math. Comp.*
///   64 (1995), 1557–1576.
/// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
///   Hopkins, 2013, §4.7.
#[derive(Debug, Clone)]
pub struct StructuredSolvePlan<'a, T: NumericScalar> {
    matrix: StructuredMatrixView<'a, T>,
    options: StructuredSolveOptions<T::Real>,
    coefficients: Vec<T>,
    phases: Vec<T>,
    spectrum: Vec<T>,
    matrix_exponent: i32,
    compact: Compact,
}
impl<'a, T: NumericScalar> StructuredSolvePlan<'a, T> {
    /// Prepares a plan for a square Toeplitz, Hankel or circulant matrix.
    ///
    /// Preparation runs sequentially; see [Execution](crate#execution). The
    /// plan keeps `options` as given, so later solves use their execution
    /// policy.
    ///
    /// # Errors
    /// The same conditions as [`build`](Self::build).
    pub fn new(
        matrix: StructuredMatrixView<'a, T>,
        options: &StructuredSolveOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        let mut plan = Self::build(
            matrix,
            &options.with_execution_options(
                options
                    .execution_options()
                    .with_execution_policy(ExecutionPolicy::Sequential),
            ),
            ExecutionContext::sequential_native(),
        )?;
        plan.options = *options;
        Ok(plan)
    }

    /// Prepares a plan for a square Toeplitz, Hankel or circulant matrix in an execution context.
    ///
    /// Each row of `matrix` counts as one iteration toward the iteration
    /// budget in `options`. For a circulant, preparation computes the
    /// eigenvalues by a direct DFT in O(n²) operations.
    ///
    /// ```
    /// use scioxide_linalg::{
    ///     LinalgExecutionOptions, StructuredMatrixView, StructuredSolveOptions, StructuredSolvePlan,
    /// };
    /// let column = [2.0_f64, 1.0, 0.0];
    /// let options = StructuredSolveOptions::new()
    ///     .with_execution_options(LinalgExecutionOptions::new().with_byte_limit(12 * 8));
    /// let plan = StructuredSolvePlan::build(
    ///     StructuredMatrixView::circulant(&column)?,
    ///     &options,
    ///     scioxide_core::ExecutionContext::sequential_native(),
    /// )?;
    /// assert_eq!(plan.order(), 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    /// - `DimensionMismatch` if `matrix` is not square.
    /// - `InvalidInput` if `matrix` is not a Toeplitz, Hankel or circulant matrix.
    /// - `ResourceLimit` if the stored values exceed the byte limit in `options`.
    /// - `SizeOverflow` or `AllocationFailure` if the stored values cannot be allocated.
    /// - `NumericalInstability` if an eigenvalue of a circulant overflows.
    /// - `ConvergenceFailure` if the iteration budget runs out.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    pub fn build<'scope, 'capability>(
        matrix: StructuredMatrixView<'a, T>,
        options: &StructuredSolveOptions<T::Real>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        let context = context.into().with_options(&options.execution)?;
        context.execution().check()?;
        if matrix.rows() != matrix.cols() {
            return Err(SciOxideError::dimension_mismatch(
                [matrix.rows(), matrix.rows()],
                [matrix.rows(), matrix.cols()],
            ));
        }
        let compact = match matrix.pattern {
            Pattern::Toeplitz(c, r) if c.iter().zip(r).all(|(&c, &r)| c == r.conj()) => {
                Compact::HermitianToeplitz
            }
            Pattern::Toeplitz(..) => Compact::Toeplitz,
            Pattern::Hankel(..) => Compact::Hankel,
            Pattern::Circulant(..) => Compact::Circulant,
            _ => {
                return Err(crate::checked::unsupported(
                    scioxide_core::argument!("matrix"),
                    "compact solve requires Toeplitz, Hankel or circulant",
                ));
            }
        };
        let n = matrix.rows();
        let (coefficients, phases, spectrum) = storage_lengths::<T>(n, compact)?;
        let len = scioxide_core::backend::checked_add(
            scioxide_core::backend::checked_add(
                coefficients,
                phases,
                scioxide_core::quantity!("preparation_scalars"),
            )?,
            spectrum,
            scioxide_core::quantity!("preparation_scalars"),
        )?;
        super::constructors::admit_owned::<T>(
            len,
            &options.execution,
            scioxide_core::quantity!("preparation_bytes"),
        )?;
        let mut plan = Self {
            matrix,
            options: *options,
            compact,
            coefficients: zeroed(coefficients)?,
            phases: zeroed(phases)?,
            spectrum: zeroed(spectrum)?,
            matrix_exponent: 0,
        };
        if n == 0 {
            return Ok(plan);
        }
        let mut state = scioxide_core::SolveState::new();
        context.into_scope().run(
            &mut state,
            identity(
                scioxide_core::algorithm_id!("linalg", "structured_prepare", 1),
                n,
            ),
            options.execution.solve_budgets(),
            |scope| {
                let execution = scope.execution();
                plan.prepare(execution, scope)?;
                execution.check()
            },
        )?;
        Ok(plan)
    }

    /// Returns the order `n`.
    pub fn order(&self) -> usize {
        self.matrix.rows()
    }

    /// Returns the workspace size for a solve `request`.
    ///
    /// A nonempty system needs `(3n + 2) m` scalars for `m` right-hand sides.
    /// For `n > 1` and `m > 0`, Toeplitz and Hankel plans also need
    /// `n (n + 1) / 2 + 7n` complex values for the packed triangular GKO
    /// factor and seven work vectors, shared by all columns. A complex value
    /// takes two scalars for real `T` and one for complex `T`.
    /// [`StructuredWorkspaceRequest::with_dense_fallback`] adds a dense copy of
    /// the matrix, its LU factorization, pivot indices and product scratch. A
    /// condition estimate needs the workspace of a one-column request. The
    /// byte limit is the smaller of the limits in `request` and in the plan
    /// options.
    ///
    /// ```
    /// use scioxide_linalg::{
    ///     LinalgExecutionOptions, StructuredMatrixView, StructuredSolveOptions, StructuredSolvePlan,
    ///     StructuredWorkspaceRequest,
    /// };
    /// let column = [2.0_f64, 1.0, 0.0];
    /// let plan = StructuredSolvePlan::new(
    ///     StructuredMatrixView::circulant(&column)?,
    ///     &StructuredSolveOptions::new(),
    /// )?;
    /// let request = StructuredWorkspaceRequest::solve_matrix(3, 2, &LinalgExecutionOptions::new());
    /// // (3n + 2) m = 11 · 2; a circulant needs no GKO storage.
    /// assert_eq!(plan.workspace_requirements(&request)?.scalar_len()?, 22);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    /// - `InvalidInput` if `request` is a factor request.
    /// - `DimensionMismatch` if `request` is for another order.
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `ResourceLimit` if the plan's stored values plus the workspace exceed
    ///   the byte limit.
    pub fn workspace_requirements(
        &self,
        request: &StructuredWorkspaceRequest,
    ) -> Result<LinalgRequirements, SciOxideError> {
        let (columns, byte_limit) =
            request.plan_solve(self.order(), self.options.execution.byte_limit())?;
        self.requirements(columns, byte_limit, request.dense_fallback)
    }

    fn requirements(
        &self,
        columns: usize,
        byte_limit: Option<usize>,
        dense_fallback: bool,
    ) -> Result<LinalgRequirements, SciOxideError> {
        let compact = self.compact_requirements(columns, byte_limit)?;
        self.check_limits(&compact)?;
        if self.order() <= 1 || columns == 0 || self.compact == Compact::Circulant {
            return Ok(compact);
        }
        if !dense_fallback {
            Ok(compact)
        } else {
            let mut requirements = self.dense_requirements(columns, byte_limit)?;
            requirements.structured_dense_fallback = true;
            Ok(requirements)
        }
    }

    fn admit_dense(
        &self,
        columns: usize,
        byte_limit: Option<usize>,
        workspace: &LinalgWorkspace<'_, T>,
    ) -> Result<(), SciOxideError> {
        if !workspace.structured_dense_fallback {
            return Err(kernels::instability(
                "GKO rejected; dense fallback was not declared",
            ));
        }
        let requirements = self.dense_requirements(columns, byte_limit)?;
        if let Some((required, available)) = workspace.capacity_shortfall(&requirements) {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("structured_dense_fallback"),
                required,
                available,
            ));
        }
        Ok(())
    }

    fn compact_requirements(
        &self,
        columns: usize,
        byte_limit: Option<usize>,
    ) -> Result<LinalgRequirements, SciOxideError> {
        let mut requirements = LinalgRequirements::new(byte_limit);
        let q = scioxide_core::quantity!("solve_workspace");
        *requirements.scalar_mut(ScalarRegion::Direct) = scioxide_core::backend::checked_add(
            scioxide_core::backend::checked_product(&[self.stride()?, columns], q)?,
            if columns == 0 { 0 } else { self.gko_len()? },
            q,
        )?;
        Ok(requirements)
    }

    fn dense_requirements(
        &self,
        columns: usize,
        byte_limit: Option<usize>,
    ) -> Result<LinalgRequirements, SciOxideError> {
        let n = self.order();
        let options =
            LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential);
        let mut requirements = crate::LuDecomposition::<T>::workspace_requirements(
            &crate::LuWorkspaceRequest::factor(n, &options),
        )?
        .max(&crate::LuDecomposition::<T>::workspace_requirements(
            &crate::LuWorkspaceRequest::solve_matrix(n, 1, &options),
        )?);
        let q = scioxide_core::quantity!("solve_workspace");
        *requirements.scalar_mut(ScalarRegion::Direct) = scioxide_core::backend::checked_add(
            requirements.scalar(ScalarRegion::Direct),
            scioxide_core::backend::checked_add(
                scioxide_core::backend::checked_product(&[n, n], q)?,
                self.compact_requirements(columns, byte_limit)?
                    .scalar(ScalarRegion::Direct),
                q,
            )?,
            q,
        )?;
        requirements.byte_limit = byte_limit;
        self.check_limits(&requirements)?;
        Ok(requirements)
    }

    /// Check the workspace against its own limit, and the plan's coefficient
    /// storage plus the workspace against the options' byte limit.
    fn check_limits(&self, requirements: &LinalgRequirements) -> Result<(), SciOxideError> {
        let quantity = scioxide_core::quantity!("structured_bytes");
        let workspace = requirements.required_bytes::<T>()?;
        let total = (self.coefficients.len() + self.phases.len() + self.spectrum.len())
            .checked_mul(core::mem::size_of::<T>())
            .and_then(|plan| plan.checked_add(workspace))
            .ok_or_else(|| SciOxideError::size_overflow(quantity))?;
        crate::workspace::admit_bytes(total, self.options.execution.byte_limit(), quantity)
    }

    /// Allocates a workspace for a solve `request`.
    ///
    /// # Errors
    /// The errors of [`workspace_requirements`](Self::workspace_requirements),
    /// and `AllocationFailure` if the workspace cannot be allocated.
    pub fn workspace(
        &self,
        request: &StructuredWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, T>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Solves `op(A) x = b` into `output` and reports the method used and the backward error.
    ///
    /// The solution is accepted if its backward error `η` meets the tolerance
    /// in the plan options. Otherwise the call returns `Ok` with
    /// [`DirectSolveTermination::ResidualRejected`] and `output` is not
    /// written. Each solve counts `n` iterations toward the iteration budget,
    /// including any fallback.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `b` or `output` does not have length `n`.
    /// - `NonFinite` if `b` contains NaN or an infinity.
    /// - `Singular` if a circulant eigenvalue, the entry of a `1 × 1` matrix,
    ///   a pivot of the GKO elimination or a pivot of the dense LU fallback is
    ///   exactly zero. A zero GKO pivot ends the solve even when the dense LU
    ///   fallback is reserved.
    /// - `NumericalInstability` if no method produced a finite solution and the
    ///   dense LU fallback is not reserved.
    /// - `ConvergenceFailure` if the iteration budget runs out.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn solve_into<'r, 'scope, 'capability>(
        &self,
        operation: StructuredOperation,
        rhs: impl Into<VectorView<'r, T>>,
        output: VectorViewMut<'_, T>,
        workspace: &mut LinalgWorkspace<'_, T>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<StructuredSolveReport<T::Real>, SciOxideError> {
        self.solve_matrix_into(
            operation,
            MatrixView::from(rhs.into()),
            MatrixViewMut::from(output),
            workspace,
            context,
        )
    }

    /// Solves `op(A) X = B` for all columns of `B` into `output`.
    ///
    /// The workspace is sized by [`workspace_requirements`](Self::workspace_requirements).
    /// A workspace reserved with the dense LU fallback grows, if needed, to the
    /// fallback size for the columns of `B`. Each entry of `B` counts as one
    /// iteration toward the iteration budget.
    /// Columns solved by Levinson–Durbin or circulant division may run in
    /// parallel. GKO factors once for all columns that need it.
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    ///
    /// # Errors
    /// The same conditions as [`solve_into`](Self::solve_into), with `B` and
    /// `output` required to have `n` rows and the same number of columns.
    pub fn solve_matrix_into<'r, 'scope, 'capability>(
        &self,
        operation: StructuredOperation,
        rhs: impl Into<MatrixView<'r, T>>,
        mut output: MatrixViewMut<'_, T>,
        workspace: &mut LinalgWorkspace<'_, T>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<StructuredSolveReport<T::Real>, SciOxideError> {
        workspace.begin_attempt();
        let context = context.into().with_options(&self.options.execution)?;
        context.execution().check()?;
        let n = self.order();
        let rhs = rhs.into();
        let columns = rhs.cols();
        crate::checked::check_block_shape(n, rhs, &output)?;
        let byte_limit = self.options.execution.byte_limit();
        // A workspace that declares the dense fallback grows to cover it for
        // these columns, or the solve fails with the limit error.
        let requirements =
            self.requirements(columns, byte_limit, workspace.structured_dense_fallback)?;
        // An empty order has nothing to validate, even with many empty columns.
        let rhs = if n == 0 {
            rhs
        } else {
            rhs.try_finite()?.as_view()
        };
        workspace.try_reserve(&requirements)?;
        let dense_admission = self.admit_dense(columns, byte_limit, workspace);
        let policy = self.options.execution_options().execution_policy();
        workspace.run_scope(
            context,
            identity(
                scioxide_core::algorithm_id!("linalg", "structured_displacement_solve", 1),
                n,
            ),
            self.options.execution.solve_budgets(),
            |workspace, kernels, scope| {
                if n == 0 || columns == 0 {
                    return Ok(StructuredSolveReport::empty());
                }
                let execution = kernels.execution();
                for _ in 0..n * columns {
                    admit(execution, scope)?;
                }
                let stride = self.stride()?;
                workspace.with_direct(|storage, _, indices, gemm| {
                    let (scratch, rest) = crate::workspace::split_mut(storage, stride * columns)?;
                    let (gko, dense) = crate::workspace::split_mut(rest, self.gko_len()?)?;
                    crate::kernel::execution::execute_chunks(
                        scratch,
                        stride,
                        policy,
                        execution,
                        |start, chunk| {
                            for (offset, work) in chunk.chunks_exact_mut(stride).enumerate() {
                                let outcome = self.solve_compact(
                                    operation,
                                    |i| rhs.get_element(i, start + offset),
                                    work,
                                    execution,
                                )?;
                                outcome.store(&mut work[stride - 2..]);
                            }
                            Ok(())
                        },
                    )?;
                    self.try_gko(
                        operation,
                        |i, j| rhs.get_element(i, j),
                        scratch,
                        gko,
                        execution,
                    )?;
                    if n > 1
                        && self.compact != Compact::Circulant
                        && dense_admission.is_ok()
                        && scratch
                            .chunks_exact(stride)
                            .any(|work| !ColumnOutcome::load(&work[stride - 2..]).is_accepted())
                    {
                        let factor =
                            self.factor_dense(operation, dense, indices, gemm, execution)?;
                        for (j, work) in scratch.chunks_exact_mut(stride).enumerate() {
                            if !ColumnOutcome::load(&work[stride - 2..]).is_accepted() {
                                let report = self.solve_dense(
                                    operation,
                                    &factor,
                                    |i| rhs.get_element(i, j),
                                    work,
                                    gemm,
                                    execution,
                                )?;
                                ColumnOutcome::Assessed(report).store(&mut work[stride - 2..]);
                            }
                        }
                    }
                    execution.check()?;
                    let mut report = StructuredSolveReport::empty();
                    for work in scratch.chunks_exact(stride) {
                        report.include(ColumnOutcome::load(&work[stride - 2..]).report()?);
                    }
                    if report.termination.is_success() {
                        for (j, work) in scratch.chunks_exact(stride).enumerate() {
                            for (i, z) in work[..n].iter().enumerate() {
                                output.set_element(i, j, *z);
                            }
                        }
                    }
                    Ok(report)
                })
            },
        )
    }

    /// Computes `rcond₁(A) = 1 / (‖A‖₁ ‖A⁻¹‖₁)` sequentially.
    ///
    /// # Errors
    /// The same conditions as
    /// [`condition_estimate_scoped`](Self::condition_estimate_scoped).
    pub fn condition_estimate(&self) -> Result<ConditionEstimate<T::Real>, SciOxideError> {
        self.condition_estimate_scoped(
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
    }

    /// Computes `rcond₁(A) = 1 / (‖A‖₁ ‖A⁻¹‖₁)` with a workspace and execution context.
    ///
    /// `‖A⁻¹‖₁` is computed column by column from `n` structured solves with
    /// the unit vectors, O(n³) operations in total; the inverse is not stored.
    /// The workspace is that of a one-column solve request. Reserve it with
    /// [`StructuredWorkspaceRequest::with_dense_fallback`] to make the dense
    /// LU fallback available. If a solve finds `A` singular, the result is
    /// zero. Each solve counts `n` iterations toward the iteration budget.
    ///
    /// ```
    /// use scioxide_linalg::{StructuredMatrixView, StructuredSolveOptions, StructuredSolvePlan};
    /// // A = 2 I has rcond₁ = 1.
    /// let column = [2.0_f64, 0.0, 0.0];
    /// let plan = StructuredSolvePlan::new(
    ///     StructuredMatrixView::circulant(&column)?,
    ///     &StructuredSolveOptions::new(),
    /// )?;
    /// let estimate = plan.condition_estimate_scoped(
    ///     &mut scioxide_linalg::LinalgWorkspace::new(),
    ///     scioxide_core::ExecutionContext::sequential_native(),
    /// )?;
    /// assert!((estimate.reciprocal().value() - 1.0).abs() < 1e-12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    /// - `DomainError` if `n = 0`.
    /// - `NumericalInstability` if a column of `A⁻¹` is not solved to the
    ///   tolerance and the dense LU fallback is not available.
    /// - `ConvergenceFailure` if the iteration budget runs out.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn condition_estimate_scoped<'scope, 'capability>(
        &self,
        workspace: &mut LinalgWorkspace<'_, T>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<ConditionEstimate<T::Real>, SciOxideError> {
        workspace.begin_attempt();
        let context = context.into().with_options(&self.options.execution)?;
        context.execution().check()?;
        let n = self.order();
        if n == 0 {
            return Err(crate::analysis::empty_condition());
        }
        let byte_limit = self.options.execution.byte_limit();
        workspace.try_reserve(&self.requirements(1, byte_limit, false)?)?;
        let mut dense_admission = Some(self.admit_dense(1, byte_limit, workspace));
        let normal = StructuredOperation::Normal;
        workspace.run_scope(
            context,
            identity(
                scioxide_core::algorithm_id!("linalg", "structured_displacement_condition", 1),
                n,
            ),
            self.options.execution.solve_budgets(),
            |workspace, kernels, scope| {
                let execution = kernels.execution();
                workspace.with_direct(|storage, _, mut indices, gemm| {
                    let (scratch, mut dense) =
                        crate::workspace::split_mut(storage, self.stride()?)?;
                    let (gko, rest) = crate::workspace::split_mut(dense, self.gko_len()?)?;
                    dense = rest;
                    let mut factor = None;
                    // Exact 1-norm condition number `||A||_1 ||A^{-1}||_1`: A's
                    // column sums come from the structure, and `A^{-1}` is
                    // materialized one column at a time with the structured solver
                    // (O(n) solves). A singular solve reports a reciprocal
                    // condition of zero rather than failing.
                    let zero = scioxide_core::Scaled::from_value(T::Real::ZERO)?;
                    let mut norm = zero;
                    let mut inverse = zero;
                    for j in 0..n {
                        let mut col = zero;
                        for i in 0..n {
                            col = col.add(scioxide_core::Scaled::from_value(
                                self.matrix.entry(i, j)?.abs(),
                            )?)?;
                        }
                        if norm.abs_le(col) {
                            norm = col;
                        }
                        for _ in 0..n {
                            admit(execution, scope)?;
                        }
                        let unit = |i: usize| if i == j { T::ONE } else { T::ZERO };
                        let result = self
                            .solve_compact(normal, unit, scratch, execution)
                            .and_then(|outcome| {
                                let stride = self.stride()?;
                                outcome.store(&mut scratch[stride - 2..]);
                                self.try_gko(normal, |i, _| unit(i), scratch, gko, execution)?;
                                let outcome = ColumnOutcome::load(&scratch[stride - 2..]);
                                if outcome.is_accepted() {
                                    return Ok(());
                                }
                                if n > 1 && self.compact != Compact::Circulant {
                                    if factor.is_none() {
                                        if let Some(admission) = dense_admission.take() {
                                            admission?;
                                        }
                                        factor = Some(self.factor_dense(
                                            normal,
                                            core::mem::take(&mut dense),
                                            core::mem::take(&mut indices),
                                            gemm,
                                            execution,
                                        )?);
                                    }
                                    if let Some(factor) = &factor {
                                        let report = self.solve_dense(
                                            normal, factor, unit, scratch, gemm, execution,
                                        )?;
                                        if report.termination.is_success() {
                                            return Ok(());
                                        }
                                    }
                                }
                                Err(kernels::instability(
                                    "condition estimate requires accepted inverse actions",
                                ))
                            });
                        if let Err(error) = result {
                            return match error.view() {
                                scioxide_core::SciOxideErrorView::Singular { .. } => {
                                    ConditionEstimate::new(
                                        T::Real::ZERO,
                                        EstimateReliability::ModelBased,
                                    )
                                }
                                _ => Err(error),
                            };
                        }
                        let mut col = zero;
                        for &value in &scratch[..n] {
                            col = col.add(scioxide_core::Scaled::from_value(value.abs())?)?;
                        }
                        if inverse.abs_le(col) {
                            inverse = col;
                        }
                    }
                    let reciprocal = scioxide_core::Scaled::from_value(T::Real::ONE)?
                        .div(norm.mul(inverse)?)?
                        .round()?
                        .min(T::Real::ONE);
                    execution.check()?;
                    ConditionEstimate::new(reciprocal, EstimateReliability::ModelBased)
                })
            },
        )
    }

    // Entry of `A` in the Toeplitz frame: Hankel columns are reversed.
    fn toeplitz_entry(&self, i: usize, j: usize) -> Result<T, SciOxideError> {
        self.matrix.entry(
            i,
            if self.compact == Compact::Hankel {
                self.order() - 1 - j
            } else {
                j
            },
        )
    }
}

mod gko;
mod kernels;
use kernels::{storage_lengths, zeroed};

#[cfg(test)]
mod empty_condition_tests {
    use super::*;

    #[test]
    fn every_empty_condition_path_is_undefined() -> Result<(), SciOxideError> {
        // Empty Toeplitz is vacuously Hermitian, so select the internal GKO
        // dispatch explicitly as well as the publicly constructible patterns.
        for compact in [
            Compact::HermitianToeplitz,
            Compact::Toeplitz,
            Compact::Hankel,
            Compact::Circulant,
        ] {
            for fallback in [false, true] {
                let mut plan = StructuredSolvePlan::new(
                    StructuredMatrixView::circulant(&[] as &[f64])?,
                    &StructuredSolveOptions::new(),
                )?;
                plan.compact = compact;
                let mut workspace = LinalgWorkspace::new();
                workspace.structured_dense_fallback = fallback;
                let error = plan
                    .condition_estimate_scoped(&mut workspace, ExecutionContext::sequential())
                    .err()
                    .ok_or_else(|| {
                        crate::checked::inconsistent(
                            scioxide_core::argument!("condition"),
                            "empty condition unexpectedly succeeded",
                        )
                    })?;
                assert_eq!(error, crate::analysis::empty_condition());
                assert_eq!(
                    workspace.last_attempt().map(|record| record.outcome()),
                    Some(scioxide_core::AttemptOutcome::PreflightRejected)
                );
                if compact == Compact::Circulant {
                    assert_eq!(
                        plan.spectral_condition_estimate().err(),
                        Some(crate::analysis::empty_condition())
                    );
                }
            }
        }
        Ok(())
    }
}
