//! Canonical correlation analysis.
use super::center::{CenteredData, centered};
use super::storage::{MultivariateRequirements, admit_preparation, cca_bytes, reserve_stage};
use super::{MultivariateOptions, PreciseCenter, extended, factor, project};
use crate::error_helpers::{count as number, invalid};
use alloc::vec::Vec;
use scioxide_core::{ExecutionContext, NumericScalar, ProductSum, RealScalar, SciOxideError};
use scioxide_linalg::{Matrix, MatrixView, RankReport};

/// Canonical correlation analysis of two data matrices with paired rows, fitted
/// once and applied to new data.
///
/// For centered data `X` (`n × p`) and `Y` (`n × q`), the `k`-th canonical
/// correlation `ρ_k` is the largest correlation between `X a` and `Y b` over
/// weight vectors `a`, `b` whose scores are uncorrelated with the first `k − 1`
/// pairs. The fit follows Björck and Golub: it computes the thin SVDs
///
/// ```text
/// X = U_x Σ_x V_xᴴ,   Y = U_y Σ_y V_yᴴ,   U_xᴴ U_y = P C Qᴴ,
/// ```
///
/// truncated to the numerical ranks `r_x` and `r_y` chosen by the rank policy
/// in [`MultivariateOptions`]. The correlations are the singular values in `C`,
/// and the weights are `A = √(n − 1) V_x Σ_x⁻¹ P` and
/// `B = √(n − 1) V_y Σ_y⁻¹ Q`. The plan keeps `r = min(r_x, r_y)` pairs. The
/// scores `X A` and `Y B` have unit sample variance, and their cross-covariance
/// is `diag(ρ₁, …, ρ_r)`.
///
/// # References
///
/// Å. Björck and G. H. Golub, "Numerical methods for computing angles between
/// linear subspaces", *Mathematics of Computation* 27 (1973), 579–594.
#[derive(Clone)]
pub struct PreparedCca<T: NumericScalar> {
    x_mean: Vec<T>,
    y_mean: Vec<T>,
    x_centers: Vec<PreciseCenter>,
    y_centers: Vec<PreciseCenter>,
    x_weights: Matrix<T>,
    y_weights: Matrix<T>,
    correlations: Vec<T::Real>,
    x_rank: RankReport<T::Real>,
    y_rank: RankReport<T::Real>,
    options: MultivariateOptions<T::Real>,
}

/// Scratch storage for [`PreparedCca::scores_into`].
///
/// It holds two matrices of the score shape. See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
pub struct CcaWorkspace<T: NumericScalar> {
    x: Vec<T>,
    y: Vec<T>,
}
impl<T: NumericScalar> Default for CcaWorkspace<T> {
    fn default() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
        }
    }
}
impl<T: NumericScalar> CcaWorkspace<T> {
    /// Returns the requirements for two score matrices of `rows × columns` entries.
    ///
    /// # Errors
    ///
    /// `ResourceLimit` if the scratch exceeds `options.max_bytes()`.
    pub fn workspace_requirements(
        rows: usize,
        columns: usize,
        options: MultivariateOptions<T::Real>,
    ) -> Result<MultivariateRequirements<T>, SciOxideError> {
        MultivariateRequirements::<T>::new(rows, columns, 2, options.max_bytes)
    }
    /// Reports whether the workspace is large enough for `req`, without growing it.
    ///
    /// # Errors
    ///
    /// `ResourceLimit` if the workspace already holds more bytes than the limit in `req`.
    pub fn check(
        &self,
        req: &MultivariateRequirements<T>,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        let bytes = self
            .x
            .capacity()
            .checked_add(self.y.capacity())
            .and_then(|v| v.checked_mul(core::mem::size_of::<T>()))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("cca_workspace"))
            })?;
        crate::error_helpers::limit_bytes(bytes, req.max_bytes)?;
        Ok(
            if req.copies == 2
                && self.x.len() >= req.rows * req.columns
                && self.y.len() >= req.rows * req.columns
            {
                scioxide_core::Capacity::Sufficient
            } else {
                scioxide_core::Capacity::NeedsGrowth
            },
        )
    }
    /// Grows the workspace to meet `req`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `req` was produced for a PCA workspace.
    /// - `ResourceLimit` if the workspace would exceed the limit in `req`.
    /// - `AllocationFailure` if memory cannot be reserved.
    pub fn try_reserve(&mut self, req: &MultivariateRequirements<T>) -> Result<(), SciOxideError> {
        if req.copies != 2 {
            return Err(invalid(
                scioxide_core::argument!("requirements"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "CCA workspace requirements",
            ));
        }
        if self.check(req)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        let count = req.rows * req.columns;
        let retained = self
            .x
            .capacity()
            .max(count)
            .checked_add(self.y.capacity().max(count))
            .and_then(|n| n.checked_mul(core::mem::size_of::<T>()))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("cca_workspace"))
            })?;
        crate::error_helpers::limit_bytes(retained, req.max_bytes)?;
        reserve_stage(&mut self.x, req.rows * req.columns)?;
        reserve_stage(&mut self.y, req.rows * req.columns)?;
        if self.check(req)? != scioxide_core::Capacity::Sufficient {
            return Err(SciOxideError::allocation_failure(
                scioxide_core::quantity!("cca_workspace"),
                req.required_bytes(),
            ));
        }
        Ok(())
    }
    /// Creates a workspace that meets `req`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::try_reserve`].
    pub fn workspace(req: &MultivariateRequirements<T>) -> Result<Self, SciOxideError> {
        let mut ws = Self::default();
        ws.try_reserve(req)?;
        Ok(ws)
    }
}

impl<T: NumericScalar> PreparedCca<T> {
    /// Returns the workspace requirements for scoring `rows` observations, under
    /// this plan's byte limit.
    ///
    /// # Errors
    ///
    /// `ResourceLimit` if the scratch exceeds the plan's byte limit.
    pub fn workspace_requirements(
        &self,
        rows: usize,
    ) -> Result<MultivariateRequirements<T>, SciOxideError> {
        CcaWorkspace::<T>::workspace_requirements(rows, self.correlations.len(), self.options)
    }
    /// Creates a workspace for scoring `rows` observations.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`] and
    /// [`CcaWorkspace::try_reserve`].
    pub fn workspace(&self, rows: usize) -> Result<CcaWorkspace<T>, SciOxideError> {
        CcaWorkspace::workspace(&self.workspace_requirements(rows)?)
    }

    /// Returns the canonical scores `(X A, Y B)` of paired rows `x` and `y`.
    ///
    /// Runs on the calling thread. Use [`Self::scores_into`] to reuse the outputs
    /// and workspace.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::scores_into`], and `ResourceLimit` if the outputs and
    /// scratch exceed the plan's byte limit.
    pub fn scores(
        &self,
        x: MatrixView<'_, T>,
        y: MatrixView<'_, T>,
    ) -> Result<(Matrix<T>, Matrix<T>), SciOxideError> {
        let req = self.workspace_requirements(x.rows())?;
        let bytes = req.required_bytes().checked_mul(2).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("cca_owned_scores"))
        })?;
        crate::error_helpers::limit_bytes(bytes, self.options.max_bytes)?;
        let mut work = CcaWorkspace::workspace(&req)?;
        let mut xs = Matrix::zeros(x.rows(), self.correlations.len())?;
        let mut ys = Matrix::zeros(y.rows(), self.correlations.len())?;
        self.scores_into(
            x,
            y,
            &mut xs,
            &mut ys,
            &mut work,
            ExecutionContext::sequential_native(),
        )?;
        Ok((xs, ys))
    }
    /// Fits the canonical correlations of the paired rows of `x` and `y` on the
    /// calling thread.
    ///
    /// If `x` or `y` is rank deficient, the rank policy truncates its SVD, so
    /// linearly dependent columns are allowed. The cost is three thin SVDs: of
    /// `x`, of `y`, and of the `r_x × r_y` cross product. `options.max_bytes()`
    /// limits the memory for the fit and for the plan. The limit is checked
    /// against the full thin SVDs before the ranks are known. Use [`Self::build`] to run with an execution context.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `x` and `y` have different row counts, fewer than two
    ///   rows, no columns, or more rows than the scalar type counts exactly
    ///   (2⁵³ for `f64`, 2²⁴ for `f32`).
    /// - `InvalidInput` if `x` or `y` has numerical rank zero, for example a
    ///   constant column.
    /// - `NonFinite` if `x` or `y` contains NaN or infinity.
    /// - `ResourceLimit` if the memory needed exceeds `options.max_bytes()`.
    /// - `ConvergenceFailure` if an SVD does not converge.
    /// - `NumericalInstability` if a weight overflows the scalar type.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{Matrix, RankPolicy};
    /// use scioxide_stats::{MultivariateOptions, PreparedCca};
    ///
    /// // A variable is perfectly correlated with itself.
    /// let x = Matrix::from_vec(3, 1, vec![1.0_f64, 2., 3.])?;
    /// let options = MultivariateOptions::new()
    ///     .with_rank_policy(RankPolicy::DimensionScaledEpsilon)
    ///     .with_max_bytes(4096);
    /// let fit = PreparedCca::with_options(x.view(), x.view(), options)?;
    /// assert!((fit.correlations()[0] - 1.).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn with_options(
        x: MatrixView<'_, T>,
        y: MatrixView<'_, T>,
        options: MultivariateOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::build(x, y, options, ExecutionContext::sequential_native())
    }
    /// Prepares the model with default options.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::with_options`].
    pub fn new(x: MatrixView<'_, T>, y: MatrixView<'_, T>) -> Result<Self, SciOxideError> {
        Self::with_options(x, y, MultivariateOptions::new())
    }

    /// Fits the canonical correlations of `x` and `y` with an execution context.
    ///
    /// The three SVDs use the context's executor. Centering and cross products sum
    /// rows in input order, so they do not depend on the number of workers.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`], and `ExecutionStopped` on cancellation or deadline.
    pub fn build(
        x: MatrixView<'_, T>,
        y: MatrixView<'_, T>,
        options: MultivariateOptions<T::Real>,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        let policy = options.rank_policy;
        let max_bytes = options.max_bytes;
        if x.rows() != y.rows() {
            return Err(invalid(
                scioxide_core::argument!("y"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "CCA row mismatch",
            ));
        }
        admit_preparation(
            cca_bytes::<T>(x.rows(), x.cols(), y.cols(), context)?,
            max_bytes,
        )?;
        let CenteredData {
            means: x_mean,
            centers: x_centers,
            values: xc,
        } = centered(x, max_bytes, context)?;
        let CenteredData {
            means: y_mean,
            centers: y_centers,
            values: yc,
        } = centered(y, max_bytes, context)?;
        context.check()?;
        let xf = factor(&xc, context)?;
        drop(xc);
        let yf = factor(&yc, context)?;
        drop(yc);
        let x_rank = xf.analyze_rank(policy)?;
        let y_rank = yf.analyze_rank(policy)?;
        let (rx, ry) = (x_rank.rank(), y_rank.rank());
        if rx == 0 || ry == 0 {
            return Err(invalid(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "CCA requires nonzero marginal ranks",
            ));
        }
        let mut cross = Matrix::zeros(rx, ry)?;
        for i in 0..rx {
            for j in 0..ry {
                let mut sum = ProductSum::<T>::new();
                for row in 0..x.rows() {
                    sum.add_product(
                        xf.u_matrix().view().get(row, i)?.conj(),
                        yf.u_matrix().view().get(row, j)?,
                    )?;
                }
                cross.view_mut().set(i, j, sum.finish()?)?;
            }
        }
        let cf = factor(&cross, context)?;
        drop(cross);
        let r = rx.min(ry);
        let mut x_weights = Matrix::zeros(x.cols(), r)?;
        let mut y_weights = Matrix::zeros(y.cols(), r)?;
        let norm = number::<T::Real>(x.rows() - 1)?.sqrt();
        for k in 0..r {
            for j in 0..x.cols() {
                let mut sum = T::ZERO;
                for a in 0..rx {
                    sum = sum
                        + xf.vt_matrix().view().get(a, j)?.conj()
                            * cf.u_matrix().view().get(a, k)?
                            / T::from_real(xf.singular_values()[a]);
                }
                let value = sum * T::from_real(norm);
                crate::error_helpers::finite(value)?;
                x_weights.view_mut().set(j, k, value)?;
            }
            for j in 0..y.cols() {
                let mut sum = T::ZERO;
                for a in 0..ry {
                    sum = sum
                        + yf.vt_matrix().view().get(a, j)?.conj()
                            * cf.vt_matrix().view().get(k, a)?.conj()
                            / T::from_real(yf.singular_values()[a]);
                }
                let value = sum * T::from_real(norm);
                crate::error_helpers::finite(value)?;
                y_weights.view_mut().set(j, k, value)?;
            }
        }
        let correlations = scioxide_core::backend::try_copy_slice(
            cf.singular_values(),
            scioxide_core::quantity!("multivariate_correlations"),
        )?;
        Ok(Self {
            x_mean,
            y_mean,
            x_centers,
            y_centers,
            x_weights,
            y_weights,
            correlations,
            x_rank,
            y_rank,
            options,
        })
    }
    /// Returns the canonical correlations `ρ₁ ≥ ρ₂ ≥ … ≥ ρ_r`, in `[0, 1]` up to rounding.
    pub fn correlations(&self) -> &[T::Real] {
        &self.correlations
    }
    /// Returns the `p × r` weight matrix `A` for `x`, one column per correlation.
    pub fn x_weights(&self) -> &Matrix<T> {
        &self.x_weights
    }
    /// Returns the `q × r` weight matrix `B` for `y`, one column per correlation.
    pub fn y_weights(&self) -> &Matrix<T> {
        &self.y_weights
    }
    /// Returns the numerical ranks of the centered `x` and `y` chosen by the rank policy.
    pub fn ranks(&self) -> (RankReport<T::Real>, RankReport<T::Real>) {
        (self.x_rank, self.y_rank)
    }
    /// Returns the column means of the fitted `x` and `y`, rounded to the scalar type.
    pub fn means(&self) -> (&[T], &[T]) {
        (&self.x_mean, &self.y_mean)
    }
    /// Computes the canonical scores `X A` and `Y B` of paired rows into `x_output`
    /// and `y_output`.
    ///
    /// Rows are centered by the compensated means of the fitted data. Both
    /// outputs must be `m × r` for `m` rows. On the fitted data, the scores have
    /// unit sample variance and cross-covariance `diag(ρ₁, …, ρ_r)` up to rounding.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `x` and `y` have different row counts, if their column
    ///   counts differ from the fitted data, or if an output is not `m × r`.
    /// - `NonFinite` if `x` or `y` contains NaN or infinity.
    /// - `ResourceLimit` if `work` is smaller than the outputs.
    /// - `NumericalInstability` if a score overflows the scalar type.
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn scores_into(
        &self,
        x: MatrixView<'_, T>,
        y: MatrixView<'_, T>,
        x_output: &mut Matrix<T>,
        y_output: &mut Matrix<T>,
        work: &mut CcaWorkspace<T>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let (n, r) = (x.rows(), self.correlations.len());
        if y.rows() != n
            || x.cols() != self.x_mean.len()
            || y.cols() != self.y_mean.len()
            || x_output.rows() != n
            || y_output.rows() != n
            || x_output.cols() != r
            || y_output.cols() != r
        {
            return Err(invalid(
                scioxide_core::argument!("x_output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "CCA score shape",
            ));
        }
        context.check()?;
        let req = self.workspace_requirements(n)?;
        if work.check(&req)? != scioxide_core::Capacity::Sufficient {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("cca_workspace_elements"),
                n * r,
                work.x.len().min(work.y.len()),
            ));
        }
        for i in 0..n {
            context.check()?;
            for k in 0..r {
                let mut xs = PreciseCenter::ZERO;
                let mut ys = PreciseCenter::ZERO;
                for (j, &center) in self.x_centers.iter().enumerate() {
                    xs.add(
                        center
                            .difference(extended(x.get(i, j)?)?)?
                            .mul(extended(self.x_weights.view().get(j, k)?)?)?,
                    )?;
                }
                for (j, &center) in self.y_centers.iter().enumerate() {
                    ys.add(
                        center
                            .difference(extended(y.get(i, j)?)?)?
                            .mul(extended(self.y_weights.view().get(j, k)?)?)?,
                    )?;
                }
                work.x[i * r + k] = project(xs.value()?)?;
                work.y[i * r + k] = project(ys.value()?)?;
            }
        }
        context.check()?;
        x_output.as_mut_slice().copy_from_slice(&work.x[..n * r]);
        y_output.as_mut_slice().copy_from_slice(&work.y[..n * r]);
        Ok(())
    }
}
