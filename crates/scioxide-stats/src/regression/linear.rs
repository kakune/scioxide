//! Linear least squares with a prepared design matrix, and inference on contrasts.
use super::{TestAlternative, budget_bytes, finite, invalid, number, t_probability};
use crate::weights::RegressionRows;
use crate::{ConfidenceProbability, RegressionWeights};
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;

use scioxide_core::{
    ExecutionContext, ExecutionPolicy, NumericScalar, ProductSum, RealScalar, Scaled,
    ScaledAccumulator, SciOxideError,
};
use scioxide_linalg::{
    HouseholderDecomposition, LinalgWorkspace, Matrix, MatrixView, RankPolicy, RankReport,
    ThinSvdDecomposition, Vector,
};

/// Selects the factorization of the weighted design matrix.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RegressionFactorization {
    /// Householder QR with column pivoting. Uses the thin SVD instead when the
    /// design has more columns than rows or is numerically rank deficient.
    #[default]
    Auto,
    /// Thin SVD. Solutions are minimum-norm after truncating singular values
    /// below the rank threshold.
    Svd,
}
/// Options of a [`PreparedRegression`]: the rank policy, the factorization and
/// a byte limit.
///
/// The byte limit bounds the storage that preparation reserves, excluding the
/// borrowed inputs, and every workspace and output the plan later creates,
/// including those behind [`PreparedRegression::fit`].
#[derive(Clone, Copy, Debug)]
pub struct LinearRegressionOptions<S: RealScalar> {
    /// Scale-dependent numerical rank convention.
    rank_policy: RankPolicy<S>,
    /// Factorization choice.
    factorization: RegressionFactorization,
    max_bytes: usize,
}

impl<S: RealScalar> LinearRegressionOptions<S> {
    /// Sets the byte limit. The default is `usize::MAX`.
    pub const fn with_max_bytes(mut self, value: usize) -> Self {
        self.max_bytes = value;
        self
    }
    /// Returns the byte limit.
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }
    /// Returns the rule that sets the numerical rank threshold. The default is
    /// `RankPolicy::DimensionScaledEpsilon`.
    pub const fn rank_policy(&self) -> RankPolicy<S> {
        self.rank_policy
    }
    /// Sets the rank policy.
    pub const fn with_rank_policy(mut self, value: RankPolicy<S>) -> Self {
        self.rank_policy = value;
        self
    }
    /// Returns the factorization. The default is
    /// [`RegressionFactorization::Auto`].
    pub const fn factorization(&self) -> RegressionFactorization {
        self.factorization
    }
    /// Sets the factorization.
    pub const fn with_factorization(mut self, value: RegressionFactorization) -> Self {
        self.factorization = value;
        self
    }
}
impl<S: RealScalar> Default for LinearRegressionOptions<S> {
    fn default() -> Self {
        Self {
            rank_policy: RankPolicy::DimensionScaledEpsilon,
            factorization: RegressionFactorization::Auto,
            max_bytes: usize::MAX,
        }
    }
}
#[derive(Clone)]
enum Factor<T: NumericScalar> {
    Qr(HouseholderDecomposition<T>),
    Svd(ThinSvdDecomposition<T>),
}

/// Returns the fitted value of one row, with fixed-length paths for short rows.
#[inline]
fn fitted_value<T: NumericScalar>(row: &[T], coefficients: &[T]) -> Result<T, SciOxideError> {
    use scioxide_core::backend::checked_product_sum;
    match (row, coefficients) {
        ([a, b], [x, y]) => checked_product_sum([(*a, *x), (*b, *y)]),
        ([a, b, c], [x, y, z]) => checked_product_sum([(*a, *x), (*b, *y), (*c, *z)]),
        ([a, b, c, d], [w, x, y, z]) => {
            checked_product_sum([(*a, *w), (*b, *x), (*c, *y), (*d, *z)])
        }
        _ => checked_product_sum(row.iter().copied().zip(coefficients.iter().copied())),
    }
}

fn fitted_residual<T: NumericScalar>(
    row: &[T],
    coefficients: &[T],
    response: T,
    root_weight: T::Real,
) -> Result<(T, T, T), SciOxideError> {
    let fitted = fitted_value(row, coefficients)?;
    let residual = crate::error_helpers::finite(response - fitted)?;
    let weighted = crate::error_helpers::finite(residual * T::from_real(root_weight))?;
    Ok((fitted, residual, weighted))
}

/// Holds a design matrix and its factorization for repeated weighted least
/// squares fits.
///
/// For an `n × p` design `X`, row weights `wᵢ ≥ 0` and a response `y`, a fit
/// solves
///
/// ```text
/// minimize ‖W^(1/2) (y − X β)‖₂ over β,
/// ```
///
/// where `W = diag(w₁, …, wₙ)`. The plan factors `A = W^(1/2) X` once, so each
/// fit costs O(np) after preparation. Preparation copies the design. The model
/// has no implicit intercept and no centering: include a column of ones for an
/// intercept. Fits accept real and complex scalars, while
/// [`Self::contrast`] and [`Self::compare_nested`] need a real design.
///
/// # Examples
///
/// ```
/// // y = 1 + 2x fits the points (0, 1), (1, 3), (2, 5) exactly.
/// use scioxide_linalg::Matrix;
/// use scioxide_stats::{LinearRegressionOptions, PreparedRegression, Unweighted};
/// let design = Matrix::from_vec(3, 2, vec![1.0_f64, 0.0, 1.0, 1.0, 1.0, 2.0])?;
/// let plan = PreparedRegression::new(
///     design.view(),
///     Unweighted,
///     LinearRegressionOptions::default().with_max_bytes(1 << 20),
/// )?;
/// let fit = plan.fit(&[1.0, 3.0, 5.0])?;
/// assert!((fit.coefficients().as_slice()[1] - 2.0).abs() < 1e-12);
/// assert!(fit.residual_norm() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone)]
pub struct PreparedRegression<T: NumericScalar> {
    design: Matrix<T>,
    root_weights: Vector<T::Real>,
    factor: Factor<T>,
    rank: RankReport<T::Real>,
    options: LinearRegressionOptions<T::Real>,
    units: f64,
    frequency: bool,
    unit_weights: bool,
}
/// Holds the coefficients, fitted values, residuals and residual norms of a
/// linear least-squares fit.
///
/// Create it with [`PreparedRegression::output`] or
/// [`crate::PreparedOperatorRegression::output`] and reuse it across fits. The
/// fields are read-only.
///
/// ```compile_fail
/// fn corrupt(mut fit: scioxide_stats::LinearRegressionFit<f64>) { fit.residual_norm = 0.0; }
/// ```
pub struct LinearRegressionFit<T: NumericScalar> {
    pub(super) coefficients: Vector<T>,
    pub(super) fitted: Vector<T>,
    pub(super) residuals: Vector<T>,
    pub(super) weighted_residuals: Vector<T>,
    pub(super) rank: Option<RankReport<T::Real>>,
    pub(super) degrees_of_freedom: Option<f64>,
    pub(super) residual_norm: T::Real,
    pub(super) weighted_residual_norm: T::Real,
    pub(super) normal_residual_norm: T::Real,
}
impl<T: NumericScalar> LinearRegressionFit<T> {
    /// Returns the bytes of numerical storage for a fixed output shape.
    pub(super) fn bytes(rows: usize, columns: usize) -> Result<usize, SciOxideError> {
        rows.checked_mul(3)
            .and_then(|n| n.checked_add(columns))
            .and_then(|n| n.checked_mul(core::mem::size_of::<T>()))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("regression_output"))
            })
    }
    /// Allocates an output of a fixed shape. The caller checks the byte limit.
    pub(super) fn reserve(rows: usize, columns: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            coefficients: Vector::zeros(columns)?,
            fitted: Vector::zeros(rows)?,
            residuals: Vector::zeros(rows)?,
            weighted_residuals: Vector::zeros(rows)?,
            rank: None,
            degrees_of_freedom: None,
            residual_norm: T::Real::ZERO,
            weighted_residual_norm: T::Real::ZERO,
            normal_residual_norm: T::Real::ZERO,
        })
    }
    /// Returns the coefficient estimates `β̂`.
    ///
    /// For a rank-deficient design, `β̂` is the minimum-norm least-squares
    /// solution.
    pub fn coefficients(&self) -> &Vector<T> {
        &self.coefficients
    }
    /// Returns the fitted values `X β̂` for every row, including rows of zero
    /// weight.
    pub fn fitted(&self) -> &Vector<T> {
        &self.fitted
    }
    /// Returns the residuals `y − X β̂`.
    pub fn residuals(&self) -> &Vector<T> {
        &self.residuals
    }
    /// Returns the weighted residuals `W^(1/2) (y − X β̂)`.
    pub fn weighted_residuals(&self) -> &Vector<T> {
        &self.weighted_residuals
    }
    /// Returns the numerical rank of the weighted design and its threshold.
    ///
    /// Returns `None` before the first fit and for operator fits.
    pub fn rank(&self) -> Option<RankReport<T::Real>> {
        self.rank
    }
    /// Returns the residual degrees of freedom `n_eff − rank`.
    ///
    /// `n_eff` is the number of rows for unweighted fits, the sum of the counts
    /// for frequency weights, and the number of rows of positive weight for
    /// analytic weights. Returns `None` before the first fit or when the rank is
    /// unknown.
    pub fn degrees_of_freedom(&self) -> Option<f64> {
        self.degrees_of_freedom
    }
    /// Returns `‖y − X β̂‖₂`.
    pub fn residual_norm(&self) -> T::Real {
        self.residual_norm
    }
    /// Returns `‖W^(1/2) (y − X β̂)‖₂`.
    pub fn weighted_residual_norm(&self) -> T::Real {
        self.weighted_residual_norm
    }
    /// Returns `‖Xᴴ W (y − X β̂)‖₂`, the norm of the normal-equation residual.
    ///
    /// It is computed from the residuals after the solve and is zero at an exact
    /// least-squares solution, so it measures how well the computed `β̂`
    /// satisfies the optimality conditions.
    pub fn normal_residual_norm(&self) -> T::Real {
        self.normal_residual_norm
    }
    pub(super) fn has_shape(&self, rows: usize, columns: usize) -> bool {
        self.coefficients.len() == columns
            && self.fitted.len() == rows
            && self.residuals.len() == rows
            && self.weighted_residuals.len() == rows
    }
    pub(super) fn copy_from(&mut self, other: &Self) -> Result<(), SciOxideError> {
        if !self.has_shape(other.fitted.len(), other.coefficients.len())
            || !other.has_shape(other.fitted.len(), other.coefficients.len())
        {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "regression output shape mismatch",
            ));
        }
        self.coefficients
            .as_mut_slice()
            .copy_from_slice(other.coefficients.as_slice());
        self.fitted
            .as_mut_slice()
            .copy_from_slice(other.fitted.as_slice());
        self.residuals
            .as_mut_slice()
            .copy_from_slice(other.residuals.as_slice());
        self.weighted_residuals
            .as_mut_slice()
            .copy_from_slice(other.weighted_residuals.as_slice());
        self.rank = other.rank;
        self.degrees_of_freedom = other.degrees_of_freedom;
        self.residual_norm = other.residual_norm;
        self.weighted_residual_norm = other.weighted_residual_norm;
        self.normal_residual_norm = other.normal_residual_norm;
        Ok(())
    }
}
/// Holds the scratch storage of [`PreparedRegression::fit_into`].
///
/// Create it with [`PreparedRegression::workspace`].
pub struct RegressionWorkspace<T: NumericScalar> {
    rhs: Vector<T>,
    scratch: scioxide_linalg::LinalgWorkspace<'static, T>,
    normal: Vector<T>,
    stage: LinearRegressionFit<T>,
}
impl<T: NumericScalar> PreparedRegression<T> {
    /// Copies and factors a design for ordinary or weighted least squares.
    ///
    /// `weights` is [`crate::Unweighted`], [`crate::Frequency`] (integer
    /// replication counts) or [`crate::Analytic`] (relative inverse variances).
    /// Rows of zero weight do not affect the fit. The factorization follows
    /// `options.factorization()`. Preparation costs O(np²) for QR and the cost
    /// of a thin SVD otherwise.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the design has no rows or no columns, or the weights
    ///   have a different length than the design has rows.
    /// - `InvalidInput` if the design contains a NaN or infinite entry.
    /// - `InvalidInput` if a weight is negative or not finite, or a frequency
    ///   weight is not an integer.
    /// - `InvalidInput` if every weight is zero, or the effective sample size
    ///   exceeds 2⁵³.
    /// - `ResourceLimit` if the storage exceeds `options.max_bytes()`.
    /// - The error of the QR or SVD factorization.
    pub fn new<'w>(
        design: MatrixView<'_, T>,
        weights: impl RegressionWeights<'w, T::Real>,
        options: LinearRegressionOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        let rows = weights.rows();
        let (n, p) = (design.rows(), design.cols());
        if n == 0 || p == 0 || !rows.matches(n) {
            return Err(invalid(
                scioxide_core::argument!("design"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid design shape or weight length",
            ));
        }
        let envelope = n
            .checked_mul(p)
            .and_then(|v| p.checked_mul(p).and_then(|q| v.checked_add(q)))
            .and_then(|v| v.checked_add(n))
            .and_then(|v| v.checked_add(p))
            .and_then(|v| v.checked_mul(32))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?;
        budget_bytes(envelope, core::mem::size_of::<T>(), options.max_bytes)?;
        let owned = if let Some(values) = design.as_contiguous_slice() {
            Matrix::from_vec(
                n,
                p,
                scioxide_core::backend::try_copy_slice(values, scioxide_core::quantity!("design"))?,
            )?
        } else {
            design.try_to_matrix()?
        };
        if owned.as_slice().iter().any(|x| !x.has_finite_components()) {
            return Err(invalid(
                scioxide_core::argument!("design"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "non-finite design",
            ));
        }
        let mut root_weights = Vector::zeros(n)?;
        let mut units = n as f64;
        let mut unit_weights = true;
        let scaled = if matches!(rows, RegressionRows::Unit) {
            root_weights.as_mut_slice().fill(T::Real::ONE);
            None
        } else {
            let mut scaled = Matrix::zeros(n, p)?;
            units = 0.0;
            for i in 0..n {
                let w = rows.weight(i)?;
                let wf = w.try_to_f64_exact()?;
                units = finite(
                    units
                        + if rows.is_frequency() {
                            wf
                        } else if wf > 0.0 {
                            1.0
                        } else {
                            0.0
                        },
                )?;
                let root = w.sqrt();
                unit_weights &= root == T::Real::ONE;
                root_weights.as_mut_slice()[i] = root;
                for (out, &x) in scaled.as_mut_slice()[i * p..(i + 1) * p]
                    .iter_mut()
                    .zip(&owned.as_slice()[i * p..(i + 1) * p])
                {
                    *out = x * T::from_real(root);
                }
            }
            Some(scaled)
        };
        if units == 0.0 || units > 9007199254740992.0 {
            return Err(invalid(
                scioxide_core::argument!("weights"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "empty or unrepresentable effective sample count",
            ));
        }
        let weighted_design = scaled.as_ref().unwrap_or(&owned);
        let factor = if options.factorization == RegressionFactorization::Auto && n >= p {
            let mut qr = HouseholderDecomposition::empty();
            let mut work = LinalgWorkspace::new();
            HouseholderDecomposition::factor_into(
                weighted_design,
                scioxide_linalg::HouseholderOrientation::Qr,
                scioxide_linalg::HouseholderPivoting::Column,
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential),
                &mut qr,
                &mut work,
                scioxide_core::ExecutionContext::sequential_native(),
            )?;
            if qr.analyze_rank(options.rank_policy)?.rank() == p {
                Factor::Qr(qr)
            } else {
                Factor::Svd(scioxide_linalg::ThinSvdDecomposition::compute(
                    weighted_design,
                )?)
            }
        } else {
            Factor::Svd(scioxide_linalg::ThinSvdDecomposition::compute(
                weighted_design,
            )?)
        };
        let rank = match &factor {
            Factor::Qr(q) => q.analyze_rank(options.rank_policy)?,
            Factor::Svd(s) => s.analyze_rank(options.rank_policy)?,
        };
        Ok(Self {
            design: owned,
            root_weights,
            factor,
            rank,
            options,
            units,
            frequency: rows.is_frequency(),
            unit_weights,
        })
    }
    /// Returns the options of this plan.
    pub fn options(&self) -> LinearRegressionOptions<T::Real> {
        self.options
    }
    /// Allocates an output for [`Self::fit_into`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the output exceeds the plan's byte limit.
    pub fn output(&self) -> Result<LinearRegressionFit<T>, SciOxideError> {
        let (n, p) = (self.design.rows(), self.design.cols());
        crate::error_helpers::limit_bytes(
            LinearRegressionFit::<T>::bytes(n, p)?,
            self.options.max_bytes,
        )?;
        LinearRegressionFit::reserve(n, p)
    }
    /// Returns the copied design, without weights.
    pub fn design(&self) -> &Matrix<T> {
        &self.design
    }
    /// Returns the numerical rank of the weighted design and its threshold.
    pub fn rank(&self) -> RankReport<T::Real> {
        self.rank
    }
    /// Estimates the condition of the weighted design from its factorization.
    ///
    /// A QR factorization gives a 1-norm estimate. An SVD gives the ratio of
    /// the extreme singular values. The result does not depend on the rank
    /// threshold and is an estimate, not a bound.
    ///
    /// # Errors
    ///
    /// Returns the error of the condition estimator.
    pub fn condition(&self) -> Result<scioxide_linalg::ConditionEstimate<T::Real>, SciOxideError> {
        match &self.factor {
            Factor::Qr(q) => q.condition_estimate(),
            Factor::Svd(s) => s.condition_estimate(),
        }
    }
    /// Allocates the workspace for [`Self::fit_into`].
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the workspace exceeds the plan's byte limit.
    pub fn workspace(&self) -> Result<RegressionWorkspace<T>, SciOxideError> {
        let (n, p) = (self.design.rows(), self.design.cols());
        budget_bytes(
            n.checked_add(p)
                .and_then(|v| v.checked_mul(16))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
                })?,
            core::mem::size_of::<T>(),
            self.options.max_bytes,
        )?;
        Ok(RegressionWorkspace {
            rhs: Vector::zeros(n)?,
            scratch: self.solve_workspace()?,
            normal: Vector::zeros(p)?,
            stage: LinearRegressionFit::reserve(n, p)?,
        })
    }

    fn solve_workspace(&self) -> Result<LinalgWorkspace<'static, T>, SciOxideError> {
        let (n, p) = (self.design.rows(), self.design.cols());
        let mut scratch = LinalgWorkspace::new();
        if matches!(self.factor, Factor::Qr(_)) {
            scratch.try_reserve(
                &scioxide_linalg::HouseholderDecomposition::<T>::workspace_requirements(
                    &scioxide_linalg::HouseholderWorkspaceRequest::solve(
                        n,
                        p,
                        scioxide_linalg::HouseholderOrientation::Qr,
                        &scioxide_linalg::LinalgExecutionOptions::new(),
                    ),
                )?,
            )?;
        } else {
            scratch = ThinSvdDecomposition::<T>::workspace(
                &scioxide_linalg::SvdWorkspaceRequest::solve(
                    n,
                    p,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )?;
        }
        Ok(scratch)
    }
    fn solve(
        &self,
        rhs: &Vector<T>,
        output: &mut Vector<T>,
        scratch: &mut scioxide_linalg::LinalgWorkspace<'_, T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        execution.check()?;
        match &self.factor {
            Factor::Qr(q) => {
                let report = q.solve_into(
                    rhs,
                    &(scioxide_linalg::DirectSolveOptions::new(self.options.rank_policy))
                        .with_execution_options(scioxide_linalg::LinalgExecutionOptions::new()),
                    output.view_mut(),
                    scratch,
                    execution,
                )?;
                if report.termination() != scioxide_linalg::DirectSolveTermination::Accepted {
                    return Err(invalid(
                        scioxide_core::argument!("self"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "prepared QR no longer has full column rank",
                    ));
                }
                Ok(())
            }
            Factor::Svd(s) => s.solve_into(
                rhs,
                self.options.rank_policy,
                T::Real::ZERO,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                output.view_mut(),
                scratch,
                execution,
            ),
        }
    }
    /// Fits one response and returns a new output.
    ///
    /// Runs sequentially and allocates its workspace and output. Use
    /// [`Self::fit_into`] to reuse storage.
    ///
    /// # Errors
    ///
    /// The conditions of [`Self::fit_into`], and `ResourceLimit` if the
    /// workspace exceeds the plan's byte limit.
    pub fn fit(&self, response: &[T]) -> Result<LinearRegressionFit<T>, SciOxideError> {
        let mut work = self.workspace()?;
        self.fit_candidate(response, &mut work, ExecutionContext::sequential_native())?;
        Ok(work.stage)
    }
    /// Fits one response and writes the result to `output`.
    ///
    /// Each fitted value and each component of `Xᴴ W r` is a dot product
    /// rounded once. See
    /// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `response`, `output` or `work` does not match the
    ///   design shape.
    /// - `InvalidInput` if `response` contains a NaN or infinite value.
    /// - `NumericalInstability` if a fitted value or residual overflows.
    /// - The error of `execution` if it is cancelled or its deadline passes.
    pub fn fit_into(
        &self,
        response: &[T],
        output: &mut LinearRegressionFit<T>,
        work: &mut RegressionWorkspace<T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if !output.has_shape(self.design.rows(), self.design.cols()) {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "fit output shape mismatch",
            ));
        }
        self.fit_candidate(response, work, execution)?;
        output.copy_from(&work.stage)
    }

    fn fit_candidate(
        &self,
        response: &[T],
        work: &mut RegressionWorkspace<T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (n, p) = (self.design.rows(), self.design.cols());
        if response.len() != n
            || work.rhs.len() != n
            || work.normal.len() != p
            || !work.stage.has_shape(n, p)
        {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "fit output or workspace shape mismatch",
            ));
        }
        self.solve_response(
            response,
            &mut work.rhs,
            &mut work.stage.coefficients,
            &mut work.scratch,
            execution,
        )?;
        for (i, ((row, &y), &root)) in self
            .design
            .as_slice()
            .chunks_exact(p)
            .zip(response)
            .zip(self.root_weights.as_slice())
            .enumerate()
        {
            let (fitted, residual, weighted) =
                fitted_residual(row, work.stage.coefficients.as_slice(), y, root)?;
            work.stage.fitted.as_mut_slice()[i] = fitted;
            work.stage.residuals.as_mut_slice()[i] = residual;
            work.stage.weighted_residuals.as_mut_slice()[i] = weighted;
        }
        self.fit_diagnostics(work, execution)
    }

    fn solve_response(
        &self,
        response: &[T],
        rhs: &mut Vector<T>,
        coefficients: &mut Vector<T>,
        scratch: &mut LinalgWorkspace<'_, T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if response.len() != self.design.rows() {
            return Err(invalid(
                scioxide_core::argument!("response"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "response shape mismatch",
            ));
        }
        execution.check()?;
        for ((rhs, &y), &root) in rhs
            .as_mut_slice()
            .iter_mut()
            .zip(response)
            .zip(self.root_weights.as_slice())
        {
            if !y.has_finite_components() {
                return Err(invalid(
                    scioxide_core::argument!("response"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "non-finite response",
                ));
            }
            *rhs = y * T::from_real(root);
        }
        self.solve(rhs, coefficients, scratch, execution)
    }

    fn fit_diagnostics(
        &self,
        work: &mut RegressionWorkspace<T>,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let p = self.design.cols();
        for (j, normal) in work.normal.as_mut_slice().iter_mut().enumerate() {
            execution.check()?;
            *normal = if self.unit_weights {
                scioxide_core::backend::checked_product_sum(
                    self.design
                        .as_slice()
                        .chunks_exact(p)
                        .zip(work.stage.residuals.as_slice())
                        .map(|(row, &r)| (row[j].conj(), r)),
                )?
            } else {
                scioxide_core::backend::checked_product_sum(
                    self.design
                        .as_slice()
                        .chunks_exact(p)
                        .zip(self.root_weights.as_slice())
                        .zip(work.stage.weighted_residuals.as_slice())
                        .map(|((row, &root), &r)| (row[j].conj() * T::from_real(root), r)),
                )?
            };
        }
        work.stage.rank = Some(self.rank);
        work.stage.degrees_of_freedom = Some(self.units - self.rank.rank() as f64);
        work.stage.residual_norm = work
            .stage
            .residuals
            .view()
            .norm2_scoped(&scioxide_linalg::LinalgExecutionOptions::new(), execution)?;
        work.stage.weighted_residual_norm = if self.unit_weights {
            work.stage.residual_norm
        } else {
            work.stage
                .weighted_residuals
                .view()
                .norm2_scoped(&scioxide_linalg::LinalgExecutionOptions::new(), execution)?
        };
        work.stage.normal_residual_norm = work
            .normal
            .view()
            .norm2_scoped(&scioxide_linalg::LinalgExecutionOptions::new(), execution)?;
        execution.check()?;
        Ok(())
    }
}

/// Selects the covariance estimator for [`PreparedRegression::contrast`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RegressionCovariance {
    /// Classical estimator `σ̂² (AᵀA)⁻¹` for independent normal errors with a
    /// common variance after weighting. Inference uses Student's t
    /// distribution.
    #[default]
    Classical,
    /// HC3 heteroskedasticity-consistent sandwich estimator. Inference uses the
    /// asymptotic normal distribution.
    ///
    /// # References
    ///
    /// J. G. MacKinnon and H. White, "Some heteroskedasticity-consistent
    /// covariance matrix estimators with improved finite sample properties",
    /// *J. Econometrics* 29(3), 305–325, 1985.
    Hc3,
}
/// Holds the test of a linear contrast `cᵀβ` and its confidence interval.
///
/// The estimate, standard error, p-value and interval come from one covariance
/// estimator. The fields are read-only.
///
/// ```compile_fail
/// fn corrupt(mut c: scioxide_stats::RegressionContrast<f64>) { c.p_value = 1.0; }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RegressionContrast<S: RealScalar> {
    pub(super) estimate: S,
    pub(super) standard_error: S,
    pub(super) p_value: S,
    pub(super) interval: scioxide_core::Interval<S>,
    pub(super) covariance: RegressionCovariance,
    pub(super) degrees_of_freedom: f64,
}
impl<S: RealScalar> RegressionContrast<S> {
    /// Returns `cᵀβ̂ − c₀`, the estimate minus the null value `c₀`.
    pub const fn estimate(&self) -> S {
        self.estimate
    }
    /// Returns the standard error of `cᵀβ̂` under the selected covariance.
    pub const fn standard_error(&self) -> S {
        self.standard_error
    }
    /// Returns the p-value for the selected alternative.
    pub const fn p_value(&self) -> S {
        self.p_value
    }
    /// Returns the two-sided confidence interval for `cᵀβ`, without
    /// subtracting the null value.
    pub const fn interval(&self) -> scioxide_core::Interval<S> {
        self.interval
    }
    /// Returns the covariance estimator.
    pub const fn covariance(&self) -> RegressionCovariance {
        self.covariance
    }
    /// Returns the residual degrees of freedom `n_eff − rank`.
    ///
    /// Only the classical estimator uses them for the t distribution.
    pub const fn degrees_of_freedom(&self) -> f64 {
        self.degrees_of_freedom
    }
}
impl<S: RealScalar> PreparedRegression<S> {
    /// Tests a nested reduced model against this model with the F-test.
    ///
    /// The statistic is
    ///
    /// ```text
    /// F = ((RSS_r − RSS_f) / (r_f − r_r)) / (RSS_f / (n_eff − r_f)),
    /// ```
    ///
    /// where `RSS_f` and `RSS_r` are the weighted residual sums of squares of
    /// this model and of `reduced`, `r_f` and `r_r` their numerical ranks, and
    /// `n_eff` the effective sample size. Under normal errors and the reduced
    /// model, `F` follows the F distribution with `(r_f − r_r, n_eff − r_f)`
    /// degrees of freedom. The p-value is the upper tail. Before testing, each
    /// column of the reduced design is checked to lie in the column space of
    /// this design.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the models have different rows or weights, or the
    ///   rank of `reduced` is not smaller than the rank of this model.
    /// - `InvalidInput` if a column of the reduced design is not in the column
    ///   space of this design.
    /// - `InvalidInput` if `n_eff − r_f ≤ 0` or `RSS_f = 0`.
    /// - `InvalidInput` if `RSS_r < RSS_f` beyond rounding.
    /// - The errors of [`Self::fit`].
    pub fn compare_nested(
        &self,
        reduced: &Self,
        response: &[S],
    ) -> Result<super::GroupTest<S>, SciOxideError> {
        if self.design.rows() != reduced.design.rows()
            || self.root_weights.as_slice() != reduced.root_weights.as_slice()
            || self.frequency != reduced.frequency
            || self.rank.rank() <= reduced.rank.rank()
        {
            return Err(invalid(
                scioxide_core::argument!("reduced"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "nested comparison requires matching weights and a strictly smaller reduced rank",
            ));
        }
        let mut column = Vector::zeros(self.design.rows())?;
        for j in 0..reduced.design.cols() {
            for (i, v) in column.as_mut_slice().iter_mut().enumerate() {
                *v = reduced.design.view().get(i, j)?;
            }
            let projected = self.fit(column.as_slice())?;
            let mut weighted = Vector::zeros(column.len())?;
            for (i, v) in weighted.as_mut_slice().iter_mut().enumerate() {
                *v = column.view().get(i)? * self.root_weights.view().get(i)?;
            }
            let tolerance = number::<S>(128.0 * self.design.rows().max(self.design.cols()) as f64)?
                * S::EPSILON
                * weighted.view().norm2_scoped(
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    scioxide_core::ExecutionContext::sequential(),
                )?;
            if projected.weighted_residual_norm > tolerance {
                return Err(invalid(
                    scioxide_core::argument!("reduced"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "reduced design is not nested in the full design",
                ));
            }
        }
        let full = self.fit(response)?;
        let small = reduced.fit(response)?;
        let df = self.units - self.rank.rank() as f64;
        let delta = (self.rank.rank() - reduced.rank.rank()) as f64;
        let full_norm = full.weighted_residual_norm.try_to_f64_exact()?;
        let small_norm = small.weighted_residual_norm.try_to_f64_exact()?;
        if df <= 0.0 || full_norm == 0.0 {
            return Err(invalid(
                scioxide_core::argument!("response"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "nested inference needs positive residual variance and degrees",
            ));
        }
        let ratio = small_norm / full_norm;
        if ratio < 1.0 - 128.0 * S::EPSILON.try_to_f64_exact()? {
            return Err(invalid(
                scioxide_core::argument!("response"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "nested residual ordering is numerically inconsistent",
            ));
        }
        let statistic = finite(((ratio - 1.0) * (ratio + 1.0)).max(0.0) * df / delta)?;
        Ok(super::GroupTest {
            statistic: number(statistic)?,
            p_value: number(super::f_probability(statistic, delta, df)?)?,
            degrees_of_freedom: delta,
            denominator_degrees: Some(df),
            method: super::GroupTestMethod::NestedModels,
        })
    }

    // Factor coordinates of (A^+)ᵀ c, where A = sqrt(W) X. Thin orthonormal
    // coordinates give the classical variance directly and HC3 influence rows.
    fn contrast_coordinates(
        &self,
        contrast: &[S],
        hc3: bool,
    ) -> Result<(Vector<S>, Option<Matrix<S>>), SciOxideError> {
        let (n, p) = (self.design.rows(), self.design.cols());
        match &self.factor {
            Factor::Qr(q) => {
                let mut rhs = Vector::zeros(p)?;
                for (j, &original) in q.permutation().iter().enumerate() {
                    rhs.view_mut().set(j, contrast[original])?;
                }
                let coordinates = scioxide_linalg::TriangularPlan::from_dense(
                    q.packed_factors().block(0, 0, p, p)?,
                    scioxide_core::Triangle::Upper,
                    scioxide_core::DiagonalKind::Explicit,
                )?
                .solve(scioxide_linalg::StructuredOperation::Transpose, rhs.view())?;
                let basis = if hc3 {
                    let mut seed = Matrix::zeros(n, p)?;
                    for j in 0..p {
                        seed.view_mut().set(j, j, S::ONE)?;
                    }
                    let mut basis = Matrix::zeros(n, p)?;
                    let options = scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential);
                    let mut scratch = LinalgWorkspace::new();
                    scratch.try_reserve(&HouseholderDecomposition::<S>::workspace_requirements(
                        &scioxide_linalg::HouseholderWorkspaceRequest::apply_q(
                            n,
                            p,
                            scioxide_linalg::HouseholderOrientation::Qr,
                            p,
                            &options,
                        ),
                    )?)?;
                    q.apply_q_into(
                        seed.view(),
                        scioxide_linalg::StructuredOperation::Normal,
                        &options,
                        basis.view_mut(),
                        &mut scratch,
                        ExecutionContext::sequential(),
                    )?;
                    Some(basis)
                } else {
                    None
                };
                Ok((coordinates, basis))
            }
            Factor::Svd(s) => {
                let rank = self.rank.rank();
                let mut projections = Vector::zeros(rank)?;
                let mut coordinates = Vector::zeros(rank)?;
                for j in 0..rank {
                    let mut sum = ProductSum::new();
                    for (k, &c) in contrast.iter().enumerate() {
                        sum.add_product(s.vt_matrix().view().get(j, k)?, c)?;
                    }
                    let projection = sum.finish()?;
                    projections.view_mut().set(j, projection)?;
                    coordinates.view_mut().set(
                        j,
                        scioxide_core::checked_float_quotient(projection, s.singular_values()[j])?,
                    )?;
                }
                let norm = contrast.iter().fold(S::ZERO, |a, &b| a.max(b.abs()));
                for (k, &c) in contrast.iter().enumerate() {
                    let mut sum = ProductSum::new();
                    for (j, &z) in projections.as_slice().iter().enumerate() {
                        sum.add_product(s.vt_matrix().view().get(j, k)?, z)?;
                    }
                    if (sum.finish()? - c).abs()
                        > number::<S>(128.0 * n.max(p) as f64)? * S::EPSILON * norm
                    {
                        return Err(invalid(
                            scioxide_core::argument!("contrast"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "contrast is not estimable under selected rank policy",
                        ));
                    }
                }
                Ok((coordinates, None))
            }
        }
    }
    /// Tests `H₀: cᵀβ = c₀` for a contrast vector `c` and null value `c₀`.
    ///
    /// The method fits `response`, then computes the standard error of `cᵀβ̂`
    /// from the factorization without forming a pseudoinverse.
    /// [`RegressionCovariance::Classical`] uses `σ̂² = RSS / (n_eff − rank)` and
    /// Student's t distribution with `n_eff − rank` degrees of freedom.
    /// [`RegressionCovariance::Hc3`] scales each squared residual by
    /// `(1 − hᵢ)⁻²`, where `hᵢ` is the leverage of row `i`, and uses the normal
    /// distribution. The interval is two-sided at level `confidence`.
    /// `alternative` selects the tail of the p-value.
    ///
    /// The classical estimator needs O(n + p²) temporary storage, HC3
    /// O(np + p²). For a rank-deficient design, `c` must be estimable, that is,
    /// lie in the row space of the weighted design.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `contrast` does not have one entry per column, or
    ///   `contrast` or `null` is not finite.
    /// - `InvalidInput` if `contrast` is not estimable under the rank policy.
    /// - `InvalidInput` for HC3 with frequency weights other than one. Expand
    ///   the rows instead.
    /// - `InvalidInput` for HC3 if a row has leverage 1.
    /// - `InvalidInput` if `n_eff − rank ≤ 0`.
    /// - The errors of [`Self::fit_into`] for `response`.
    pub fn contrast(
        &self,
        response: &[S],
        contrast: &[S],
        null: S,
        covariance: RegressionCovariance,
        confidence: ConfidenceProbability,
        alternative: TestAlternative,
    ) -> Result<RegressionContrast<S>, SciOxideError> {
        let (n, p) = (self.design.rows(), self.design.cols());
        if contrast.len() != p || contrast.iter().any(|v| !v.is_finite()) || !null.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("contrast"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid contrast",
            ));
        }
        if covariance == RegressionCovariance::Hc3
            && self.frequency
            && self.root_weights.as_slice().iter().any(|&w| w != S::ONE)
        {
            return Err(invalid(
                scioxide_core::argument!("covariance"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "HC3 frequency inference requires explicit replication",
            ));
        }
        let execution = ExecutionContext::sequential_native();
        let mut rhs = Vector::zeros(n)?;
        let mut coefficients = Vector::zeros(p)?;
        let mut scratch = self.solve_workspace()?;
        self.solve_response(
            response,
            &mut rhs,
            &mut coefficients,
            &mut scratch,
            execution,
        )?;
        // The response RHS is no longer needed after solving. Reuse its storage
        // for the original-row weighted residuals required by both covariances.
        for (i, ((row, &y), &root)) in self
            .design
            .as_slice()
            .chunks_exact(p)
            .zip(response)
            .zip(self.root_weights.as_slice())
            .enumerate()
        {
            rhs.as_mut_slice()[i] = fitted_residual(row, coefficients.as_slice(), y, root)?.2;
        }
        let weighted_residual_norm = rhs
            .view()
            .norm2_scoped(&scioxide_linalg::LinalgExecutionOptions::new(), execution)?;
        let degrees = self.units - self.rank.rank() as f64;
        if degrees <= 0.0 {
            return Err(invalid(
                scioxide_core::argument!("response"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "inference requires positive residual degrees",
            ));
        }

        let (coordinates, qr_basis) =
            self.contrast_coordinates(contrast, covariance == RegressionCovariance::Hc3)?;
        let mut variance = ScaledAccumulator::ZERO;
        if covariance == RegressionCovariance::Classical {
            let scale = Scaled::from_value(weighted_residual_norm.into_f64())?
                .div(Scaled::from_value(degrees.sqrt())?)?;
            for &z in coordinates.as_slice() {
                variance.add_scaled(Scaled::from_value(z.into_f64())?.mul(scale)?.square()?)?;
            }
        } else {
            for i in 0..n {
                let mut influence = ProductSum::new();
                let mut leverage = ProductSum::new();
                for (j, &z) in coordinates.as_slice().iter().enumerate() {
                    let u = match &self.factor {
                        Factor::Qr(_) => qr_basis
                            .as_ref()
                            .ok_or_else(|| {
                                invalid(
                                    scioxide_core::argument!("self"),
                                    scioxide_core::InvalidInputReason::Inconsistent,
                                    "missing thin QR basis",
                                )
                            })?
                            .view()
                            .get(i, j)?,
                        Factor::Svd(s) => s.u_matrix().view().get(i, j)?,
                    };
                    influence.add_product(u, z)?;
                    leverage.add_product(u, u)?;
                }
                let remaining = S::ONE - leverage.finish()?;
                if remaining <= number::<S>(32.0)? * S::EPSILON {
                    return Err(invalid(
                        scioxide_core::argument!("covariance"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "HC3 is undefined at unit leverage",
                    ));
                }
                let v = Scaled::from_value(influence.finish()?.into_f64())?
                    .mul(Scaled::from_value(rhs.view().get(i)?.into_f64())?)?
                    .div(Scaled::from_value(remaining.into_f64())?)?;
                variance.add_scaled(v.square()?)?;
            }
        }
        let se: S = variance.total()?.sqrt()?.round_to()?;
        let effect = scioxide_core::backend::checked_product_sum(
            contrast
                .iter()
                .copied()
                .zip(coefficients.as_slice().iter().copied()),
        )?;
        let estimate = finite(effect - null)?;
        let ef = estimate.try_to_f64_exact()?;
        let sf = se.try_to_f64_exact()?;
        let t = if sf == 0.0 {
            if ef == 0.0 {
                0.0
            } else {
                ef.signum() * f64::MAX
            }
        } else {
            ef / sf
        };
        let (probability, critical) = if covariance == RegressionCovariance::Classical {
            (
                t_probability(t, degrees, alternative)?,
                confidence.student_t_critical_value_with_options(degrees, Default::default())?,
            )
        } else {
            let two = scioxide_special::erfc(t.abs() / core::f64::consts::SQRT_2)?;
            let prob = match alternative {
                TestAlternative::TwoSided => two,
                TestAlternative::Less => {
                    if t < 0.0 {
                        two / 2.0
                    } else {
                        1.0 - two / 2.0
                    }
                }
                TestAlternative::Greater => {
                    if t > 0.0 {
                        two / 2.0
                    } else {
                        1.0 - two / 2.0
                    }
                }
            };
            (prob, confidence.normal_critical_value::<f64>()?)
        };
        let half = finite(number::<S>(critical)? * se)?;
        Ok(RegressionContrast {
            estimate,
            standard_error: se,
            p_value: number(probability)?,
            interval: scioxide_core::Interval::new(finite(effect - half)?, finite(effect + half)?)?,
            covariance,
            degrees_of_freedom: degrees,
        })
    }
}

/// Fits several responses to one prepared design, in parallel when the
/// execution context allows it.
///
/// `output[k]` receives the fit of `responses[k]`. `options` selects the
/// execution policy. `staging` holds one output per response, and `workers`
/// one [`RegressionWorkspace`] per worker. Every fit uses `execution`, with its
/// cancellation and deadline. The missing-value policy of `options` does not
/// apply: a NaN or infinite response value is an error. See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// # Errors
///
/// - `InvalidInput` if `output` or `staging` does not have one entry per
///   response, or an entry does not match the design shape.
/// - The errors of [`PreparedRegression::fit_into`] for any response.
/// - The error of the executor.
pub fn regression_batch_into<T: NumericScalar>(
    model: &PreparedRegression<T>,
    responses: &[&[T]],
    options: crate::StatisticsOptions,
    output: &mut [LinearRegressionFit<T>],
    staging: &mut [LinearRegressionFit<T>],
    workers: &mut [RegressionWorkspace<T>],
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if output.len() != responses.len() || staging.len() != responses.len() {
        return Err(invalid(
            scioxide_core::argument!("responses"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "regression batch shape mismatch",
        ));
    }
    for fit in output.iter().chain(staging.iter()) {
        if !fit.has_shape(model.design.rows(), model.design.cols()) {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "regression batch output shape mismatch",
            ));
        }
    }
    crate::estimation::execute(
        options,
        execution,
        responses.len(),
        1,
        workers,
        staging,
        |job, work, slot| model.fit_into(responses[job], &mut slot[0], work, execution),
    )?;
    execution.check()?;
    output
        .iter_mut()
        .zip(staging)
        .try_for_each(|(out, stage)| out.copy_from(stage))
}

/// Fits ordinary least squares with unweighted observations and the default
/// [`LinearRegressionOptions`].
///
/// Uses [`PreparedRegression`] with [`RegressionFactorization::Auto`]: Householder QR with
/// column pivoting in O(np²) time, or the thin SVD with the minimum-norm solution when the
/// design has more columns than rows or is numerically rank deficient.
///
/// # Errors
///
/// The errors of [`PreparedRegression::new`] and [`PreparedRegression::fit`]. A rank-deficient
/// design is not an error.
///
/// ```
/// use scioxide_linalg::MatrixView;
/// let design = MatrixView::from_slice(4, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0])?;
/// let fit = scioxide_stats::ordinary_least_squares(design, &[1.0_f64, 3.0, 7.0, 9.0])?;
/// assert!((fit.coefficients().as_slice()[1] - 2.8).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ordinary_least_squares<T: NumericScalar>(
    design: MatrixView<'_, T>,
    response: &[T],
) -> Result<LinearRegressionFit<T>, SciOxideError> {
    PreparedRegression::new(
        design,
        crate::Unweighted,
        LinearRegressionOptions::default(),
    )?
    .fit(response)
}
impl<S: RealScalar> PreparedRegression<S> {
    /// Computes two-sided inference for each coefficient using one unit-vector contrast per column.
    ///
    /// Costs p evaluations of [`Self::contrast`] for p columns and reuses one contrast vector.
    ///
    /// # Errors
    ///
    /// Returns contrast and allocation errors.
    pub fn coefficient_inference(
        &self,
        response: &[S],
        covariance: RegressionCovariance,
        confidence: ConfidenceProbability,
    ) -> Result<Vec<RegressionContrast<S>>, SciOxideError> {
        let p = self.design().cols();
        let mut contrast = scioxide_core::backend::try_filled(
            p,
            S::ZERO,
            scioxide_core::quantity!("regression_contrast"),
        )?;
        let mut output = scioxide_core::backend::try_with_capacity(
            p,
            scioxide_core::quantity!("coefficient_inference"),
        )?;
        for j in 0..p {
            contrast[j] = S::ONE;
            output.push(self.contrast(
                response,
                &contrast,
                S::ZERO,
                covariance,
                confidence,
                TestAlternative::TwoSided,
            )?);
            contrast[j] = S::ZERO;
        }
        Ok(output)
    }
}
