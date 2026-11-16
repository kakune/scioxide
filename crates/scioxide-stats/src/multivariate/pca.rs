//! Principal component analysis.
use super::center::{CenteredData, centered};
use super::storage::{MultivariateRequirements, admit_preparation, pca_bytes, reserve_stage};
use super::{Extended, MultivariateOptions, PreciseCenter, conjugate, extended, factor, project};
use crate::error_helpers::invalid;
use alloc::vec::Vec;
use scioxide_core::backend::RealBackend;
use scioxide_core::{
    AlgebraicScalar, ExecutionContext, NumericScalar, NumericalIssue, RealScalar, Scaled,
    SciOxideError,
};
use scioxide_linalg::{Matrix, MatrixView, RankReport};

/// Selects the scale of principal component scores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcaScaling {
    /// Scores in the units of the data.
    Unwhitened,
    /// Scores divided by the standard deviation of their component, so each has
    /// unit sample variance on the fitted data.
    Whitened,
}

/// Principal component analysis of a data matrix, fitted once and applied to new data.
///
/// Rows of the `n × p` data matrix `X` are observations. The fit centers `X` by
/// its column means `x̄` and computes the thin singular value decomposition
///
/// ```text
/// X − 1 x̄ᵀ = U Σ Vᴴ,
/// ```
///
/// where `1` is the vector of ones, `Σ` holds the singular values `σ₁ ≥ σ₂ ≥ …`,
/// and the columns of `V` are the principal directions. The variance of
/// component `k` is `σ_k²/(n − 1)`. Only the first `r` components are kept,
/// where `r` is the numerical rank chosen by the rank policy in
/// [`MultivariateOptions`].
///
/// The scores of the rows of a matrix `Z` are `(Z − 1 x̄ᵀ) V`. Whitened scores
/// are also divided by the component standard deviations `σ_k/√(n − 1)`.
/// [`Self::reconstruct`] maps scores back to `1 x̄ᵀ + S Vᴴ`, where `S` is the
/// score matrix. Transforms subtract the compensated means rather than the
/// rounded [`Self::mean`], and whitening divides by the standard deviation
/// before it is rounded, so a variance that underflows in [`Self::variances`]
/// does not affect the scores.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::Matrix;
/// use scioxide_stats::{MultivariateOptions, PcaScaling, PreparedPca};
///
/// // Observations 1 and 3: mean 2, variance 2, scores ±1.
/// let data = Matrix::from_vec(2, 1, vec![1.0_f64, 3.0])?;
/// let pca = PreparedPca::with_options(data.view(), MultivariateOptions::new())?;
/// assert!((pca.variances()[0] - 2.0).abs() < 1e-14);
/// let scores = pca.transform(data.view(), PcaScaling::Unwhitened)?;
/// assert!((scores.as_slice()[0].abs() - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// I. T. Jolliffe, *Principal Component Analysis*, 2nd ed., Springer, 2002.
#[derive(Clone)]
pub struct PreparedPca<T: NumericScalar> {
    mean: Vec<T>,
    centers: Vec<PreciseCenter>,
    native_centers: Vec<Option<[f64; 2]>>,
    components: Matrix<T>,
    variances: Vec<T::Real>,
    deviations: Vec<Scaled<f64>>,
    rank: RankReport<T::Real>,
    options: MultivariateOptions<T::Real>,
}

/// Scratch storage for [`PreparedPca::transform_into`] and
/// [`PreparedPca::reconstruct_into`].
///
/// It holds one matrix of the output shape. See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
pub struct PcaWorkspace<T: NumericScalar> {
    stage: Vec<T>,
}
impl<T: NumericScalar> Default for PcaWorkspace<T> {
    fn default() -> Self {
        Self { stage: Vec::new() }
    }
}
impl<T: NumericScalar> PcaWorkspace<T> {
    /// Returns the requirements for outputs of `rows × columns` entries.
    ///
    /// # Errors
    ///
    /// `ResourceLimit` if the scratch exceeds `options.max_bytes()`.
    pub fn workspace_requirements(
        rows: usize,
        columns: usize,
        options: MultivariateOptions<T::Real>,
    ) -> Result<MultivariateRequirements<T>, SciOxideError> {
        MultivariateRequirements::<T>::new(rows, columns, 1, options.max_bytes)
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
        crate::error_helpers::limit_bytes(
            self.stage
                .capacity()
                .checked_mul(core::mem::size_of::<T>())
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("multivariate_workspace"))
                })?,
            req.max_bytes,
        )?;
        Ok(
            if req.copies == 1 && self.stage.len() >= req.rows * req.columns {
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
    /// - `InvalidInput` if `req` was produced for a CCA workspace.
    /// - `ResourceLimit` if the workspace would exceed the limit in `req`.
    /// - `AllocationFailure` if memory cannot be reserved.
    pub fn try_reserve(&mut self, req: &MultivariateRequirements<T>) -> Result<(), SciOxideError> {
        if req.copies != 1 {
            return Err(invalid(
                scioxide_core::argument!("requirements"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "PCA workspace requirements",
            ));
        }
        if self.check(req)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        reserve_stage(&mut self.stage, req.rows * req.columns)?;
        if self.check(req)? != scioxide_core::Capacity::Sufficient {
            return Err(SciOxideError::allocation_failure(
                scioxide_core::quantity!("pca_workspace"),
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

impl<T: NumericScalar> PreparedPca<T> {
    /// Returns the workspace requirements for outputs of `rows × columns` entries,
    /// under this plan's byte limit.
    ///
    /// For [`Self::transform_into`], `columns` is the number of kept components.
    /// For [`Self::reconstruct_into`], it is the number of variables.
    ///
    /// # Errors
    ///
    /// `ResourceLimit` if the scratch exceeds the plan's byte limit.
    pub fn workspace_requirements(
        &self,
        rows: usize,
        columns: usize,
    ) -> Result<MultivariateRequirements<T>, SciOxideError> {
        PcaWorkspace::<T>::workspace_requirements(rows, columns, self.options)
    }
    /// Creates a workspace for outputs of `rows × columns` entries.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::workspace_requirements`] and
    /// [`PcaWorkspace::try_reserve`].
    pub fn workspace(&self, rows: usize, columns: usize) -> Result<PcaWorkspace<T>, SciOxideError> {
        PcaWorkspace::workspace(&self.workspace_requirements(rows, columns)?)
    }

    /// Returns the principal component scores of the rows of `data`.
    ///
    /// Runs on the calling thread. Use [`Self::transform_into`] to reuse the
    /// output and workspace.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::transform_into`], and `ResourceLimit` if the output
    /// and scratch exceed the plan's byte limit.
    pub fn transform(
        &self,
        data: MatrixView<'_, T>,
        scaling: PcaScaling,
    ) -> Result<Matrix<T>, SciOxideError> {
        let req = self.workspace_requirements(data.rows(), self.variances.len())?;
        let bytes = req.required_bytes().checked_mul(2).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("pca_owned_transform"))
        })?;
        crate::error_helpers::limit_bytes(bytes, self.options.max_bytes)?;
        let mut output = Matrix::zeros(data.rows(), self.variances.len())?;
        let mut work = PcaWorkspace::workspace(&req)?;
        self.transform_into(
            data,
            scaling,
            &mut output,
            &mut work,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Maps principal component scores back to the original variables.
    ///
    /// Runs on the calling thread. Use [`Self::reconstruct_into`] to reuse the
    /// output and workspace.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::reconstruct_into`], and `ResourceLimit` if the output
    /// and scratch exceed the plan's byte limit.
    pub fn reconstruct(
        &self,
        scores: MatrixView<'_, T>,
        scaling: PcaScaling,
    ) -> Result<Matrix<T>, SciOxideError> {
        let req = self.workspace_requirements(scores.rows(), self.mean.len())?;
        let bytes = req.required_bytes().checked_mul(2).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("pca_owned_reconstruct"))
        })?;
        crate::error_helpers::limit_bytes(bytes, self.options.max_bytes)?;
        let mut output = Matrix::zeros(scores.rows(), self.mean.len())?;
        let mut work = PcaWorkspace::workspace(&req)?;
        self.reconstruct_into(
            scores,
            scaling,
            &mut output,
            &mut work,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
    /// Fits the principal components of `data` on the calling thread.
    ///
    /// The cost is one thin SVD of the centered `n × p` matrix,
    /// O(np·min(n, p)) time. `options.max_bytes()` limits the memory for the fit
    /// and for the plan. The limit is checked against the full thin SVD before
    /// the rank is known. Use [`Self::build`] to run with an execution context.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `data` has fewer than two rows, no columns, or more rows
    ///   than the scalar type counts exactly (2⁵³ for `f64`, 2²⁴ for `f32`).
    /// - `NonFinite` if `data` contains NaN or infinity.
    /// - `ResourceLimit` if the memory needed exceeds `options.max_bytes()`.
    /// - `ConvergenceFailure` if the SVD does not converge.
    /// - `NumericalInstability` if a component variance overflows the scalar type,
    ///   or (`Breakdown`) if a kept component has zero variance.
    pub fn with_options(
        data: MatrixView<'_, T>,
        options: MultivariateOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::build(data, options, ExecutionContext::sequential_native())
    }
    /// Prepares the model with default options.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::with_options`].
    pub fn new(data: MatrixView<'_, T>) -> Result<Self, SciOxideError> {
        Self::with_options(data, MultivariateOptions::new())
    }

    /// Fits the principal components of `data` with an execution context.
    ///
    /// The SVD uses the context's executor. Centering sums rows in input order, so
    /// the means do not depend on the number of workers.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`], and `ExecutionStopped` on cancellation or deadline.
    pub fn build(
        data: MatrixView<'_, T>,
        options: MultivariateOptions<T::Real>,
        context: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        let policy = options.rank_policy;
        let max_bytes = options.max_bytes;
        admit_preparation(
            pca_bytes::<T>(data.rows(), data.cols(), context)?,
            max_bytes,
        )?;
        let CenteredData {
            means: mean,
            centers,
            values: centered,
        } = centered(data, max_bytes, context)?;
        context.check()?;
        let factor = factor(&centered, context)?;
        drop(centered);
        let rank = factor.analyze_rank(policy)?;
        let r = rank.rank();
        let mut components = Matrix::zeros(data.cols(), r)?;
        let mut variances = scioxide_core::backend::try_filled(
            r,
            T::Real::ZERO,
            scioxide_core::quantity!("multivariate_variances"),
        )?;
        let mut deviations = scioxide_core::backend::try_filled(
            r,
            Scaled::ZERO,
            scioxide_core::quantity!("pca_deviations"),
        )?;
        let denom = Scaled::from_value((data.rows() - 1) as f64)?;
        for (k, (variance, deviation)) in variances.iter_mut().zip(&mut deviations).enumerate() {
            // Positivity and whitening use the unrounded variance; only the
            // reported variance is projected to the scalar range.
            let exact = Scaled::from_value(factor.singular_values()[k].try_to_f64_exact()?)?
                .square()?
                .div(denom)?;
            if exact.is_zero() {
                return Err(SciOxideError::numerical_instability(
                    NumericalIssue::Breakdown,
                    "retained PCA variance is zero",
                ));
            }
            *deviation = exact.sqrt()?;
            *variance = exact.round_to()?;
            for j in 0..data.cols() {
                components
                    .view_mut()
                    .set(j, k, factor.vt_matrix().view().get(k, j)?.conj())?;
            }
        }
        let mut native_centers = scioxide_core::backend::try_filled(
            if T::IS_REAL { data.cols() } else { 0 },
            None,
            scioxide_core::quantity!("pca_native_centers"),
        )?;
        for (native, center) in native_centers.iter_mut().zip(&centers) {
            *native = center.native_real();
        }
        Ok(Self {
            mean,
            centers,
            native_centers,
            components,
            variances,
            deviations,
            rank,
            options,
        })
    }
    /// Returns the column means of the fitted data, rounded to the scalar type.
    pub fn mean(&self) -> &[T] {
        &self.mean
    }
    /// Returns the `p × r` matrix whose columns are the kept principal directions,
    /// in order of decreasing variance.
    pub fn components(&self) -> &Matrix<T> {
        &self.components
    }
    /// Returns the sample variance `σ_k²/(n − 1)` of each kept component.
    ///
    /// A variance below the smallest normal number rounds to a subnormal or zero.
    pub fn variances(&self) -> &[T::Real] {
        &self.variances
    }
    /// Returns the numerical rank chosen by the rank policy.
    pub fn rank(&self) -> RankReport<T::Real> {
        self.rank
    }
    /// Computes the principal component scores of the rows of `data` into `output`.
    ///
    /// `output` must be `m × r` for `m` rows of `data` and `r` kept components.
    /// Real scores are accumulated with exact products in `f64`, and complex or
    /// out-of-range values in extended range. The cost is O(mpr).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `data` does not have `p` columns or `output` is not `m × r`.
    /// - `NonFinite` if `data` contains NaN or infinity.
    /// - `ResourceLimit` if `work` is smaller than the output.
    /// - `NumericalInstability` if a score overflows the scalar type.
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn transform_into(
        &self,
        data: MatrixView<'_, T>,
        scaling: PcaScaling,
        output: &mut Matrix<T>,
        work: &mut PcaWorkspace<T>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if data.cols() != self.mean.len()
            || output.rows() != data.rows()
            || output.cols() != self.variances.len()
        {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "PCA transform shape",
            ));
        }
        context.check()?;
        for row in 0..data.rows() {
            if row % 256 == 0 {
                context.check()?;
            }
            for column in 0..data.cols() {
                let value = data.get(row, column)?;
                if !value.has_finite_components() {
                    extended(value)?;
                }
            }
        }
        let req = self.workspace_requirements(output.rows(), output.cols())?;
        if work.check(&req)? != scioxide_core::Capacity::Sufficient {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("pca_workspace_elements"),
                req.rows * req.columns,
                work.stage.len(),
            ));
        }
        for (k, &deviation) in self.deviations.iter().enumerate() {
            let scale = if scaling == PcaScaling::Whitened {
                deviation
            } else {
                Scaled::ONE
            };
            let component = self.components.view().col(k)?;
            for i in 0..data.rows() {
                context.check()?;
                if let Some(raw) = self.native_score(data.row(i)?, component)? {
                    let value = if scaling == PcaScaling::Unwhitened {
                        scioxide_core::backend::checked_final_rounding(
                            T::Real::from_f64_ieee(raw),
                            "PCA score exceeds the finite scalar range",
                        )?
                    } else {
                        Scaled::from_value(raw)?.div(scale)?.round_to()?
                    };
                    work.stage[i * output.cols() + k] = T::from_real(value);
                    continue;
                }
                let mut sum = PreciseCenter::ZERO;
                for (j, &center) in self.centers.iter().enumerate() {
                    sum.add(
                        center
                            .difference(extended(data.get(i, j)?)?)?
                            .mul(extended(self.components.view().get(j, k)?)?)?,
                    )?;
                }
                let raw = sum.value()?;
                let value: T = project(Extended::from_components(
                    raw.real().div(scale)?,
                    raw.imaginary().div(scale)?,
                ))?;
                work.stage[i * output.cols() + k] = value;
            }
        }
        context.check()?;
        let count = output.as_slice().len();
        output.as_mut_slice().copy_from_slice(&work.stage[..count]);
        Ok(())
    }

    // Guard before forming products: all admitted differences and loadings lie
    // within ±256 binary exponents, so exact reduction cannot lose a component
    // before whitening. f32 is narrowed only after the scaled normalization.
    fn native_score(
        &self,
        row: scioxide_linalg::VectorView<'_, T>,
        component: scioxide_linalg::VectorView<'_, T>,
    ) -> Result<Option<f64>, SciOxideError> {
        if !T::IS_REAL {
            return Ok(None);
        }
        let mut admitted = true;
        let raw = scioxide_core::backend::checked_product_sum(
            self.native_centers
                .iter()
                .zip(row.iter())
                .zip(component.iter())
                .map(|((&center, value), loading)| {
                    let Some(center) = center else {
                        admitted = false;
                        return (0.0, 0.0);
                    };
                    let value = value.real_component().into_f64();
                    let loading = loading.real_component().into_f64();
                    if !crate::arithmetic::native(value, 64)
                        || !crate::arithmetic::native(loading, 256)
                    {
                        admitted = false;
                        return (0.0, 0.0);
                    }
                    let delta = crate::arithmetic::normalized_deviation(value, center);
                    if !crate::arithmetic::native(delta, 256) {
                        admitted = false;
                        return (0.0, 0.0);
                    }
                    (delta, loading)
                }),
        )?;
        Ok(admitted.then_some(raw))
    }
    /// Maps principal component scores back to the original variables, into `output`.
    ///
    /// Computes `1 x̄ᵀ + S Vᴴ` for the score matrix `S`, first multiplying whitened
    /// scores by the component standard deviations. `scores` must have `r`
    /// columns and `output` must be `m × p` for `m` rows of `scores`. With all
    /// components kept, this inverts [`Self::transform_into`] up to rounding.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `scores` does not have `r` columns or `output` is not `m × p`.
    /// - `NonFinite` if `scores` contains NaN or infinity.
    /// - `ResourceLimit` if `work` is smaller than the output.
    /// - `NumericalInstability` if an entry overflows the scalar type.
    /// - `ExecutionStopped` on cancellation or deadline.
    pub fn reconstruct_into(
        &self,
        scores: MatrixView<'_, T>,
        scaling: PcaScaling,
        output: &mut Matrix<T>,
        work: &mut PcaWorkspace<T>,
        context: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if scores.cols() != self.variances.len()
            || output.rows() != scores.rows()
            || output.cols() != self.mean.len()
        {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "PCA reconstruction shape",
            ));
        }
        context.check()?;
        let req = self.workspace_requirements(output.rows(), output.cols())?;
        if work.check(&req)? != scioxide_core::Capacity::Sufficient {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("pca_workspace_elements"),
                req.rows * req.columns,
                work.stage.len(),
            ));
        }
        for i in 0..scores.rows() {
            context.check()?;
            for j in 0..self.mean.len() {
                let mut sum = self.centers[j];
                for k in 0..self.variances.len() {
                    let scale = if scaling == PcaScaling::Whitened {
                        self.deviations[k]
                    } else {
                        Scaled::ONE
                    };
                    sum.add(
                        extended(scores.get(i, k)?)?
                            .mul_real(scale)?
                            .mul(conjugate(extended(self.components.view().get(j, k)?)?))?,
                    )?;
                }
                let value = project(sum.value()?)?;
                work.stage[i * output.cols() + j] = value;
            }
        }
        context.check()?;
        let count = output.as_slice().len();
        output.as_mut_slice().copy_from_slice(&work.stage[..count]);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use alloc::vec;

    fn compare_projection_paths<S: RealScalar>() {
        let convert = |x| S::try_from_f64_lossy(x).unwrap();
        let tolerance = 128.0 * S::EPSILON.into_f64();
        for n in [64, 1024] {
            let offset = 2.0_f64.powi(S::MANTISSA_DIGITS as i32 - 4);
            let data = Matrix::from_vec(
                n,
                3,
                (0..n)
                    .flat_map(|i| {
                        [
                            offset + if i & 1 == 0 { -3.0 } else { 3.0 },
                            offset + if i & 2 == 0 { -2.0 } else { 2.0 },
                            offset + if i & 4 == 0 { -1.0 } else { 1.0 },
                        ]
                        .map(convert)
                    })
                    .collect(),
            )
            .unwrap();
            let native =
                PreparedPca::with_options(data.view(), MultivariateOptions::new()).unwrap();
            assert!(native.native_centers.iter().all(Option::is_some));
            let mut scaled = native.clone();
            scaled.native_centers.fill(None);
            for scaling in [PcaScaling::Unwhitened, PcaScaling::Whitened] {
                let actual = native.transform(data.view(), scaling).unwrap();
                let expected = scaled.transform(data.view(), scaling).unwrap();
                for (&a, &b) in actual.as_slice().iter().zip(expected.as_slice()) {
                    assert!((a - b).abs().into_f64() <= tolerance);
                }
            }
            // A later transform can contain values outside preparation's native
            // domain. Such a row must replay the scaled projection unchanged.
            let exponent = if S::MANTISSA_DIGITS == 24 { 70 } else { 300 };
            let wide =
                Matrix::from_vec(1, 3, vec![S::ONE.scalbn(exponent), S::ZERO, S::ONE]).unwrap();
            for scaling in [PcaScaling::Unwhitened, PcaScaling::Whitened] {
                assert_eq!(
                    native.transform(wide.view(), scaling).unwrap(),
                    scaled.transform(wide.view(), scaling).unwrap()
                );
            }
        }
    }

    #[test]
    fn native_and_scaled_scores_agree_in_both_precisions() {
        compare_projection_paths::<f32>();
        compare_projection_paths::<f64>();
    }
}
