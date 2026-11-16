//! Shared helpers for structured solves: workspace requests and the solve methods
//! of prepared factors.
use crate::diagnostics::Band;
use crate::kernel::CANDIDATE;
use crate::workspace::ScalarRegion;
use crate::{LinalgExecutionOptions, LinalgRequirements};
use alloc::vec::Vec;
use scioxide_core::backend::checked_product;
use scioxide_core::{
    AlgorithmId, Complex, InvalidInputReason, NumericScalar, Quantity, SciOxideError,
};
use scioxide_core::{Scaled, ScaledValue};

// Shared structured helpers report; `argument` and `quantity` name the offending input or
// resource.
pub(crate) fn size(a: usize, b: usize, quantity: Quantity) -> Result<usize, SciOxideError> {
    a.checked_mul(b)
        .ok_or_else(|| SciOxideError::size_overflow(quantity))
}
/// Grow `values` fallibly to hold `len` elements without changing its contents.
pub(crate) fn reserve_capacity<T>(
    values: &mut Vec<T>,
    len: usize,
    quantity: Quantity,
) -> Result<(), SciOxideError> {
    if values.capacity() < len {
        scioxide_core::backend::try_reserve(values, len - values.len(), quantity)?;
    }
    Ok(())
}
// Canonical core rounding permits gradual underflow, including signed zero.
pub(crate) fn wide<S: NumericScalar>(x: S) -> Result<ScaledValue<Complex<S::Real>>, SciOxideError> {
    ScaledValue::from_value(Complex::new(x.real_component(), x.imaginary_component()))
}
pub(crate) fn narrow<S: NumericScalar>(
    x: ScaledValue<Complex<S::Real>>,
) -> Result<S, SciOxideError> {
    ScaledValue::<S>::from_components(x.real(), x.imaginary()).round()
}
pub(crate) fn quotient<S: NumericScalar>(a: S, b: S) -> Result<S, SciOxideError> {
    if b == S::ZERO {
        return Err(scioxide_core::SciOxideError::singular(None));
    }
    narrow(wide(a)?.div(wide(b)?)?)
}

/// Selects `op(A)` in a structured product or solve: `A`, `Aᵀ` or `Aᴴ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredOperation {
    /// `op(A) = A`.
    Normal,
    /// `op(A) = Aᵀ`, the transpose.
    Transpose,
    /// `op(A) = Aᴴ`, the conjugate transpose, which equals `Aᵀ` for real scalars.
    Adjoint,
}
impl StructuredOperation {
    pub(crate) fn value<S: NumericScalar>(self, x: S) -> S {
        if self == Self::Adjoint { x.conj() } else { x }
    }
}

/// Normwise backward error of `op(A)` for the compact coefficients `element` of
/// `A` inside `band`, sharing the dense diagnostics in original coordinates.
pub(super) fn residual<S: NumericScalar>(
    band: Band,
    operation: StructuredOperation,
    element: impl Fn(usize, usize) -> S,
    x: &[S],
    rhs: crate::VectorView<'_, S>,
    context: crate::KernelContext<'_>,
) -> Result<scioxide_core::ResidualReport<S::Real>, SciOxideError> {
    let transpose = operation != StructuredOperation::Normal;
    let band = if transpose { band.transpose() } else { band };
    let element = |i, j| {
        if transpose {
            operation.value(element(j, i))
        } else {
            element(i, j)
        }
    };
    let norm = crate::diagnostics::one_norm(band.transpose(), |i, j| element(j, i))?;
    crate::diagnostics::backward_error(
        band,
        element,
        |_| None,
        norm,
        crate::VectorView::from_slice(x),
        rhs,
        context,
    )
}

/// Copy of the original band coefficients, used to compute residuals of the original system.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Source<S: NumericScalar> {
    data: Vec<S>,
    order: usize,
    lower: usize,
    upper: usize,
    norm: Scaled<S::Real>,
}
impl<S: NumericScalar> Source<S> {
    pub(super) const fn empty() -> Self {
        Self {
            data: Vec::new(),
            order: 0,
            lower: 0,
            upper: 0,
            norm: Scaled::ZERO,
        }
    }
    pub(super) fn reserve(
        &mut self,
        n: usize,
        lower: usize,
        upper: usize,
    ) -> Result<(), SciOxideError> {
        let count = if n == 0 {
            0
        } else {
            let width = lower
                .checked_add(upper)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("diagnostic_band"))
                })?;
            checked_product(&[n, width], scioxide_core::quantity!("diagnostic_band"))?
        };
        reserve_capacity(
            &mut self.data,
            count,
            scioxide_core::quantity!("diagnostic_band"),
        )
    }
    /// Stores the band; it cannot fail, so callers validate and reserve first.
    /// `column(j)` yields the band entries of column `j` in ascending row order.
    pub(super) fn publish<I: Iterator<Item = S>>(
        &mut self,
        n: usize,
        lower: usize,
        upper: usize,
        norm: Scaled<S::Real>,
        column: impl Fn(usize) -> I,
    ) {
        self.data.clear();
        self.order = n;
        self.lower = lower;
        self.upper = upper;
        self.norm = norm;
        if n == 0 {
            return;
        }
        let width = lower + upper + 1;
        self.data.resize(n * width, S::ZERO);
        for (col, slots) in self.data.chunks_exact_mut(width).enumerate() {
            let first = upper - col.min(upper);
            slots[first..]
                .iter_mut()
                .zip(column(col))
                .for_each(|(slot, value)| *slot = value);
        }
    }
    pub(super) fn norm(&self) -> Scaled<S::Real> {
        self.norm
    }
    fn element(&self, row: usize, col: usize) -> S {
        self.data[col * (self.lower + self.upper + 1) + self.upper + row - col]
    }
    pub(super) fn residual(
        &self,
        operation: StructuredOperation,
        x: &[S],
        rhs: crate::VectorView<'_, S>,
        context: crate::KernelContext<'_>,
    ) -> Result<scioxide_core::ResidualReport<S::Real>, SciOxideError> {
        residual(
            Band::new(self.order, self.lower, self.upper),
            operation,
            |i, j| self.element(i, j),
            x,
            rhs,
            context,
        )
    }
}

/// Factorization or right-hand-side job described by a structured request.
#[derive(Debug, Clone, Copy)]
pub(crate) enum StructuredJob {
    Factor {
        order: usize,
        lower: usize,
        upper: usize,
    },
    Solve {
        order: usize,
        columns: usize,
    },
}

/// Describes a structured factorization or solve, for sizing a workspace.
///
/// The workspace is a [`LinalgWorkspace`](crate::LinalgWorkspace).
///
/// Pass the request to the `workspace_requirements` or `workspace` method of
/// [`BandedLuDecomposition`](crate::BandedLuDecomposition),
/// [`BandedCholeskyDecomposition`](crate::BandedCholeskyDecomposition),
/// [`TridiagonalLuDecomposition`](crate::TridiagonalLuDecomposition),
/// [`TriangularPlan`](crate::TriangularPlan),
/// [`CyclicTridiagonalPlan`](crate::CyclicTridiagonalPlan) or
/// [`StructuredSolvePlan`](crate::StructuredSolvePlan).
/// A factor request gives the order and the band: a Hermitian band of
/// bandwidth `k` uses `lower = upper = k`, and a tridiagonal matrix uses
/// `lower = upper = 1`. A solve request gives the order and the number of
/// right-hand sides. Prepared plans accept only solve requests for their own
/// order. The request takes the byte limit of the options it is built from;
/// a plan applies the smaller of that limit and its own. The storage of the
/// factor itself is allocated separately. See
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct StructuredWorkspaceRequest {
    pub(crate) job: StructuredJob,
    pub(crate) byte_limit: Option<usize>,
    pub(crate) dense_fallback: bool,
}

impl StructuredWorkspaceRequest {
    /// Requests the factorization of an `order × order` band matrix.
    ///
    /// The matrix has `lower` subdiagonals and `upper` superdiagonals.
    pub const fn factor(
        order: usize,
        lower: usize,
        upper: usize,
        options: &LinalgExecutionOptions,
    ) -> Self {
        Self {
            job: StructuredJob::Factor {
                order,
                lower,
                upper,
            },
            byte_limit: options.byte_limit(),
            dense_fallback: false,
        }
    }

    /// Requests a solve with one right-hand side of length `order`.
    pub const fn solve(order: usize, options: &LinalgExecutionOptions) -> Self {
        Self::solve_matrix(order, 1, options)
    }

    /// Requests a solve with `right_hand_sides` right-hand sides of length `order`.
    pub const fn solve_matrix(
        order: usize,
        right_hand_sides: usize,
        options: &LinalgExecutionOptions,
    ) -> Self {
        Self {
            job: StructuredJob::Solve {
                order,
                columns: right_hand_sides,
            },
            byte_limit: options.byte_limit(),
            dense_fallback: false,
        }
    }

    /// Also requests storage for the dense LU fallback of a structured solve plan.
    ///
    /// This applies to [`StructuredSolvePlan`](crate::StructuredSolvePlan).
    ///
    /// A Toeplitz or Hankel plan always reserves storage for its O(n²) solvers.
    /// With this request, a workspace also holds a dense copy of the matrix
    /// and its LU factorization with partial pivoting. Columns that the O(n²)
    /// solvers cannot solve to the requested backward error are then solved
    /// with LU, at O(n³) cost. Without it, such a column makes the solve return
    /// `Ok` with
    /// [`DirectSolveTermination::ResidualRejected`](crate::DirectSolveTermination::ResidualRejected),
    /// or `NumericalInstability` if no finite solution was found. Spare
    /// capacity in a workspace reserved without this request does not enable
    /// the fallback.
    ///
    /// Reserving fails with `ResourceLimit` if the fallback exceeds the byte
    /// limit. Circulant plans and the other structured factorizations have no
    /// fallback.
    ///
    /// ```
    /// use scioxide_linalg::{LinalgExecutionOptions, StructuredWorkspaceRequest};
    /// let request = StructuredWorkspaceRequest::solve(8, &LinalgExecutionOptions::new())
    ///     .with_dense_fallback();
    /// ```
    pub const fn with_dense_fallback(mut self) -> Self {
        self.dense_fallback = true;
        self
    }

    /// Right-hand sides of a plan solve over `order`, and the stricter of the
    /// request's and the plan's byte limits.
    pub(crate) fn plan_solve(
        &self,
        order: usize,
        byte_limit: Option<usize>,
    ) -> Result<(usize, Option<usize>), SciOxideError> {
        match self.job {
            StructuredJob::Solve {
                order: requested,
                columns,
            } if requested == order => {
                Ok((columns, self.byte_limit.into_iter().chain(byte_limit).min()))
            }
            StructuredJob::Solve {
                order: requested,
                columns,
            } => Err(SciOxideError::dimension_mismatch(
                [order, columns],
                [requested, columns],
            )),
            StructuredJob::Factor { .. } => Err(SciOxideError::invalid_input(
                scioxide_core::argument!("request"),
                InvalidInputReason::UnsupportedConfiguration,
                "a prepared plan takes a solve request",
            )),
        }
    }
}

/// Requirements for `columns` staged right-hand sides of length `order`.
pub(crate) fn rhs_requirements<S: NumericScalar>(
    order: usize,
    columns: usize,
    byte_limit: Option<usize>,
) -> Result<LinalgRequirements, SciOxideError> {
    let mut requirements = LinalgRequirements::new(byte_limit);
    *requirements.scalar_mut(CANDIDATE) = checked_product(
        &[order, columns],
        scioxide_core::quantity!("structured_rhs"),
    )?;
    requirements.required_bytes::<S>()?;
    Ok(requirements)
}

/// Workspace for a solve with a rank policy, including an optional condition estimate.
pub(crate) fn solve_requirements<S: NumericScalar>(
    order: usize,
    columns: usize,
    byte_limit: Option<usize>,
) -> Result<LinalgRequirements, SciOxideError> {
    let requirements = rhs_requirements::<S>(order, columns, byte_limit)?
        .max(&crate::estimate::condition_requirements(order, byte_limit)?);
    requirements.required_bytes::<S>()?;
    Ok(requirements)
}

/// Requirements for a factor candidate of `scalars` values and `indices` pivots.
pub(crate) fn factor_requirements<S: NumericScalar>(
    scalars: usize,
    indices: usize,
    byte_limit: Option<usize>,
) -> Result<LinalgRequirements, SciOxideError> {
    let mut requirements = LinalgRequirements::new(byte_limit).with_indices(indices);
    *requirements.scalar_mut(ScalarRegion::Direct) = scalars;
    requirements.required_bytes::<S>()?;
    Ok(requirements)
}

pub(crate) fn identity(algorithm: AlgorithmId, order: usize) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        algorithm,
        order as u128,
        0,
    )
}

pub(crate) fn admit(
    execution: scioxide_core::ExecutionContext<'_>,
    solve: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<(), SciOxideError> {
    crate::solve_support::admit_iteration(
        Some(execution),
        Some(solve),
        "structured pivot/RHS budget exhausted",
    )
}

// Generate the view-based RHS solve family for the structured factor `$ty`.
// `$order` names the order accessor, `$vector` the single-vector kernel hook
// (which may route to an injected provider) and `solve_slice` the native
// column kernel. Each entry point reports `scioxide_linalg::$ty::<method>`.
// `$setup` is a doctest prelude defining a 2×2 `factor` with `A [1, 1] = [2, 3]`.
macro_rules! solve_methods {
    ($ty:ident, $order:ident, $vector:ident, $setup:literal) => {
        /// Solves `op(A) x = b` and returns `x`.
        ///
        /// `operation` selects `op(A)`. The solve uses the default
        /// [`DirectSolveOptions`](crate::DirectSolveOptions): a pivot that is
        /// exactly zero makes the factor rank deficient under
        /// [`RankPolicy::Exact`](crate::RankPolicy::Exact), and the solve then
        /// fails with `Singular`. An empty system has the empty solution.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = $setup]
        /// let x = factor.solve(StructuredOperation::Normal, &[2.0, 3.0][..])?;
        /// assert!(x.as_slice().iter().all(|x| (x - 1.0_f64).abs() < 1e-12));
        /// # Ok::<(), scioxide_core::SciOxideError>(())
        /// ```
        ///
        /// # Errors
        /// - `Singular` if the factor has an exactly zero pivot.
        /// - The other conditions of [`solve_matrix_into`](Self::solve_matrix_into).
        pub fn solve<'b>(
            &self,
            operation: super::direct::StructuredOperation,
            rhs: impl Into<crate::VectorView<'b, S>>,
        ) -> Result<crate::Vector<S>, scioxide_core::SciOxideError> {
            self.solve_with_options(operation, rhs, &crate::DirectSolveOptions::default())?
                .into_accepted()
        }
        /// Solves `op(A) x = b`, checks the result against `options` and
        /// returns it with its report.
        ///
        /// The rank is computed from the pivots of the factorization and
        /// compared with the [`RankPolicy`](crate::RankPolicy) in `options`. A
        /// rejected solve returns `Ok` with no solution; see
        /// [Direct solves](crate#direct-solves). An empty system is accepted
        /// unless `options` requests a condition estimate.
        ///
        /// # Errors
        /// The same conditions as [`solve_matrix_into`](Self::solve_matrix_into).
        pub fn solve_with_options<'b>(
            &self,
            operation: super::direct::StructuredOperation,
            rhs: impl Into<crate::VectorView<'b, S>>,
            options: &crate::DirectSolveOptions<S::Real>,
        ) -> Result<crate::DirectSolveResult<crate::Vector<S>, S::Real>, scioxide_core::SciOxideError> {
            let options = options.with_execution_options(
                options
                    .execution_options()
                    .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
            );
            let mut output = crate::Vector::zeros(self.$order())?;
            let report = self.solve_into(
                operation,
                rhs,
                &options,
                output.view_mut(),
                &mut crate::LinalgWorkspace::new(),
                scioxide_core::ExecutionContext::sequential_native(),
            )?;
            Ok(crate::DirectSolveResult::new(output, report))
        }
        /// Solves `op(A) x = b` into `output`, using `workspace` for scratch.
        ///
        /// This is [`solve_matrix_into`](Self::solve_matrix_into) with one
        /// right-hand side.
        ///
        /// # Errors
        /// The same conditions as [`solve_matrix_into`](Self::solve_matrix_into).
        pub fn solve_into<'b, 'scope, 'capability>(
            &self,
            operation: super::direct::StructuredOperation,
            rhs: impl Into<crate::VectorView<'b, S>>,
            options: &crate::DirectSolveOptions<S::Real>,
            output: crate::VectorViewMut<'_, S>,
            workspace: &mut crate::LinalgWorkspace<'_, S>,
            context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
        ) -> Result<crate::DirectSolveReport<S::Real>, scioxide_core::SciOxideError> {
            self.solve_matrix_into(
                operation,
                crate::MatrixView::from(rhs.into()),
                options,
                crate::MatrixViewMut::from(output),
                workspace,
                context,
            )
        }
        /// Solves `op(A) X = B` for all columns of `B` and returns `X`.
        ///
        /// The solve uses the default
        /// [`DirectSolveOptions`](crate::DirectSolveOptions), as
        /// [`solve`](Self::solve) does.
        ///
        /// # Errors
        /// - `Singular` if the factor has an exactly zero pivot.
        /// - The other conditions of [`solve_matrix_into`](Self::solve_matrix_into).
        pub fn solve_matrix<'b>(
            &self,
            operation: super::direct::StructuredOperation,
            rhs: impl Into<crate::MatrixView<'b, S>>,
        ) -> Result<crate::Matrix<S>, scioxide_core::SciOxideError> {
            self.solve_matrix_with_options(operation, rhs, &crate::DirectSolveOptions::default())?
                .into_accepted()
        }
        /// Solves `op(A) X = B` for all columns of `B` with one rank and
        /// condition decision under `options`, and returns `X` with its report.
        ///
        /// # Errors
        /// The same conditions as [`solve_matrix_into`](Self::solve_matrix_into).
        pub fn solve_matrix_with_options<'b>(
            &self,
            operation: super::direct::StructuredOperation,
            rhs: impl Into<crate::MatrixView<'b, S>>,
            options: &crate::DirectSolveOptions<S::Real>,
        ) -> Result<crate::DirectSolveResult<crate::Matrix<S>, S::Real>, scioxide_core::SciOxideError> {
            let options = options.with_execution_options(
                options
                    .execution_options()
                    .with_execution_policy(scioxide_core::ExecutionPolicy::Sequential),
            );
            let rhs = rhs.into();
            let mut output = crate::Matrix::zeros(self.$order(), rhs.cols())?;
            let report = self.solve_matrix_into(
                operation,
                rhs,
                &options,
                output.view_mut(),
                &mut crate::LinalgWorkspace::new(),
                scioxide_core::ExecutionContext::sequential_native(),
            )?;
            Ok(crate::DirectSolveResult::new(output, report))
        }
        /// Solves `op(A) X = B` into `output`, with one rank and condition decision.
        ///
        /// The condition test uses the estimate of `rcond₁(A)` from
        /// [`condition_estimate_scoped`](Self::condition_estimate_scoped), for
        /// `A` itself whatever `operation` is. If `options` requests a
        /// residual, the report holds the largest normwise backward error of
        /// `op(A) x = b` over the columns. Each column counts as one iteration
        /// toward the iteration budget in `options`.
        ///
        /// # Errors
        /// - `DimensionMismatch` if `B` does not have `n` rows or `output` does
        ///   not have the shape of `B`.
        /// - `NonFinite` if `B` contains NaN or an infinity.
        /// - `DomainError` if `n = 0` and `options` requests a condition estimate.
        /// - `ConvergenceFailure` if the iteration budget runs out.
        /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
        /// - `NumericalInstability` if the substitution produces a non-finite value.
        /// - Workspace errors as described in
        ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
        pub fn solve_matrix_into<'b, 'scope, 'capability>(
            &self,
            operation: super::direct::StructuredOperation,
            rhs: impl Into<crate::MatrixView<'b, S>>,
            options: &crate::DirectSolveOptions<S::Real>,
            mut output: crate::MatrixViewMut<'_, S>,
            workspace: &mut crate::LinalgWorkspace<'_, S>,
            context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
        ) -> Result<crate::DirectSolveReport<S::Real>, scioxide_core::SciOxideError> {
            workspace.begin_attempt();
            let context = context.into().with_options(&options.execution_options())?;
            context.execution().check()?;
            let rhs = rhs.into();
            let n = self.$order();
            crate::checked::check_block_shape(n, rhs, &output)?;
            let rhs = rhs
                .try_finite_dispatched(context.kernel().dispatch())?
                .as_view();
            let mut requirements = super::direct::rhs_requirements::<S>(
                n,
                rhs.cols(),
                options.execution_options().byte_limit(),
            )?;
            if options.minimum_reciprocal_condition().is_some() {
                if n == 0 {
                    return Err(crate::analysis::empty_condition());
                }
                requirements = requirements.max(&crate::estimate::condition_requirements(
                    n,
                    options.execution_options().byte_limit(),
                )?);
            }
            workspace.try_reserve(&requirements)?;
            let rank = self.rank(options.rank_policy())?;
            workspace.run_scope(
                context,
                super::direct::identity(
                    scioxide_core::algorithm_id!("linalg", "structured_solve", 1),
                    n,
                ),
                options.execution_options().solve_budgets(),
                |workspace, kernels, scope| {
                    let condition = options
                        .minimum_reciprocal_condition()
                        .map(|_| {
                            self.condition_estimate_scoped(
                                &options.execution_options(),
                                workspace,
                                kernels.scoped(&mut *scope),
                            )
                        })
                        .transpose()?;
                    let report = crate::analysis::acceptance(rank, condition, options);
                    if report.termination() != crate::DirectSolveTermination::Accepted
                        || n == 0
                        || rhs.cols() == 0
                    {
                        let residual = if n == 0 {
                            crate::analysis::empty_residual(options, rhs.cols())?
                        } else {
                            None
                        };
                        return Ok(crate::DirectSolveReport::new(
                            report.termination(),
                            rank,
                            condition,
                            residual,
                        ));
                    }
                    let staged =
                        &mut workspace.scalar_mut(crate::kernel::CANDIDATE)[..n * rhs.cols()];
                    for (j, column) in staged.chunks_exact_mut(n).enumerate() {
                        super::direct::admit(kernels.execution(), scope)?;
                        for (i, value) in column.iter_mut().enumerate() {
                            *value = rhs.get_element(i, j);
                        }
                    }
                    crate::kernel::execution::execute_chunks(
                        staged,
                        n,
                        options.execution_options().execution_policy(),
                        kernels.execution(),
                        |_, block| {
                            for column in block.chunks_exact_mut(n) {
                                kernels.execution().check()?;
                                if rhs.cols() == 1 {
                                    self.$vector(column, operation, kernels)?;
                                } else {
                                    self.solve_slice(column, operation, kernels)?;
                                }
                            }
                            Ok(())
                        },
                    )?;
                    let mut residual = None;
                    if options.residual() == crate::Estimation::Estimate {
                        for (j, column) in staged.chunks_exact(n).enumerate() {
                            crate::analysis::retain_largest_residual(
                                &mut residual,
                                self.residual(operation, column, rhs.col(j)?, kernels)?,
                            );
                        }
                    }
                    kernels.execution().check()?;
                    for (j, column) in staged.chunks_exact(n).enumerate() {
                        for (i, &value) in column.iter().enumerate() {
                            output.set_element(i, j, value);
                        }
                    }
                    Ok(crate::DirectSolveReport::new(
                        report.termination(),
                        rank,
                        condition,
                        residual,
                    ))
                },
            )
        }
        /// Estimates the reciprocal condition number `rcond₁(A) = 1 / (‖A‖₁ ‖A⁻¹‖₁)`.
        ///
        /// `‖A‖₁` is computed from the original matrix. `‖A⁻¹‖₁` is estimated
        /// with Hager's method as refined by Higham, which applies `A⁻¹` and
        /// `A⁻ᴴ` to a few vectors through the factorization. The estimate is
        /// not a bound; in the crate's reference tests it is within a factor
        /// of 10 of the exact value. A singular factor gives zero.
        ///
        /// # Errors
        /// - `DomainError` if `n = 0`.
        /// - `AllocationFailure` if the scratch cannot be allocated.
        ///
        /// # References
        /// - W. W. Hager, "Condition estimates", *SIAM J. Sci. Stat. Comput.* 5
        ///   (1984), 311–316.
        /// - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*,
        ///   2nd ed., SIAM, 2002, Ch. 15.
        pub fn condition_estimate(
            &self,
        ) -> Result<crate::ConditionEstimate<S::Real>, scioxide_core::SciOxideError> {
            self.condition_estimate_scoped(
                &crate::LinalgExecutionOptions::new(),
                &mut crate::LinalgWorkspace::new(),
                scioxide_core::ExecutionContext::sequential_native(),
            )
        }
        /// Estimates `rcond₁(A)` with a workspace and an execution context.
        ///
        /// The method is that of [`condition_estimate`](Self::condition_estimate).
        ///
        /// A workspace reserved for a solve request is large enough. `options`
        /// supplies the byte limit, time limit and iteration budget; each
        /// application of `A⁻¹` or `A⁻ᴴ` counts as one iteration.
        ///
        /// # Errors
        /// - `DomainError` if `n = 0`.
        /// - `ConvergenceFailure` if the iteration budget runs out.
        /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
        /// - Workspace errors as described in
        ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
        pub fn condition_estimate_scoped<'scope, 'capability>(
            &self,
            options: &crate::LinalgExecutionOptions,
            workspace: &mut crate::LinalgWorkspace<'_, S>,
            context: impl Into<crate::LinalgExecutionContext<'scope, 'capability>>,
        ) -> Result<crate::ConditionEstimate<S::Real>, scioxide_core::SciOxideError> {
            crate::estimate::scoped_condition(
                options,
                workspace,
                context.into(),
                || {
                    Ok(crate::estimate::ConditionQuery {
                        algorithm: scioxide_core::algorithm_id!("linalg", "structured_condition", 1),
                        order: self.$order(),
                        norm: self.diagnostic_norm()?,
                        singular: !self.rank(crate::RankPolicy::Exact)?.is_full_rank(),
                    })
                },
                |action, kernels| {
                    let adjoint = action.adjoint;
                    action.solve(
                        |values, _| {
                            self.solve_slice(
                                values,
                                if adjoint {
                                    super::direct::StructuredOperation::Adjoint
                                } else {
                                    super::direct::StructuredOperation::Normal
                                },
                                kernels,
                            )
                        },
                        |rhs, _| self.solve_scaled(rhs, adjoint),
                    )
                },
            )
        }
    };
}
pub(crate) use solve_methods;

/// Require operator application buffers: `x` of length `cols`, `y` of length
/// `rows` and at least `scratch` scratch values.
pub(crate) fn check_apply(
    [cols, rows, scratch]: [usize; 3],
    x: usize,
    y: usize,
    w: usize,
) -> Result<(), SciOxideError> {
    if x != cols || y != rows {
        return Err(SciOxideError::dimension_mismatch([cols, rows], [x, y]));
    }
    if w < scratch {
        return Err(SciOxideError::dimension_mismatch([scratch], [w]));
    }
    Ok(())
}
