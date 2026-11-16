//! Orthogonal distance regression: errors-in-variables fitting with known
//! measurement standard deviations.
use crate::error_helpers::invalid;
use crate::{FitDerivativeCapabilities, FitDerivatives, FitObjective};
use alloc::vec::Vec;

use crate::error_helpers::count as number;
use crate::multivariate::{CenteredData, centered, factor, reserve_multivariate};
use scioxide_core::{ExecutionContext, ProductSum, RealScalar, SciOxideError};
use scioxide_linalg::{Matrix, MatrixView, RankPolicy, RankReport};

/// An explicit model `y = f(β, x)` for orthogonal distance regression, with `k`
/// parameters `β`, `p` predictors `x` and `q` responses `y`.
///
/// All values are in the units of the data. The derivative methods return
/// exact partial derivatives of [`Self::response`]. Methods must be pure
/// functions of their arguments, because they may be called from several
/// threads.
pub trait OdrModel<S: RealScalar>: Sync {
    /// Returns the number of parameters `k`.
    fn parameter_count(&self) -> usize;
    /// Returns the number of predictors `p`.
    fn predictor_count(&self) -> usize;
    /// Returns the number of responses `q`.
    fn response_count(&self) -> usize;
    /// Returns response `coordinate` of `f(β, x)`, with `β = parameters` and
    /// `x = predictors`.
    ///
    /// # Errors
    ///
    /// An error if the model is undefined at these arguments.
    fn response(
        &self,
        parameters: &[S],
        predictors: OdrPredictors<'_, S>,
        coordinate: usize,
    ) -> Result<S, SciOxideError>;
    /// Returns `∂f_r/∂β_j` for response `r = response` and parameter `j = parameter`.
    ///
    /// # Errors
    ///
    /// An error if the derivative is undefined at these arguments.
    fn parameter_derivative(
        &self,
        parameters: &[S],
        predictors: OdrPredictors<'_, S>,
        response: usize,
        parameter: usize,
    ) -> Result<S, SciOxideError>;
    /// Returns `∂f_r/∂x_j` for response `r = response` and predictor `j = predictor`.
    ///
    /// # Errors
    ///
    /// An error if the derivative is undefined at these arguments.
    fn predictor_derivative(
        &self,
        parameters: &[S],
        predictors: OdrPredictors<'_, S>,
        response: usize,
        predictor: usize,
    ) -> Result<S, SciOxideError>;
}

/// The adjusted predictors `x + δ` of one observation, passed to an [`OdrModel`].
#[derive(Clone, Copy)]
pub struct OdrPredictors<'a, S: RealScalar> {
    observed: MatrixView<'a, S>,
    row: usize,
    corrections: &'a [S],
}
impl<S: RealScalar> OdrPredictors<'_, S> {
    /// Returns the number of predictors.
    pub fn len(&self) -> usize {
        self.corrections.len()
    }
    /// Returns `true` if there are no predictors.
    pub fn is_empty(&self) -> bool {
        self.corrections.is_empty()
    }
    /// Returns the adjusted predictor `x_j + δ_j`.
    ///
    /// # Errors
    ///
    /// `InvalidInput` if `j` is out of range or `x_j + δ_j` overflows.
    pub fn get(&self, j: usize) -> Result<S, SciOxideError> {
        let delta = *self.corrections.get(j).ok_or_else(|| {
            invalid(
                scioxide_core::argument!("j"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "predictor index",
            )
        })?;
        let value = self.observed.get(self.row, j)? + delta;
        if !value.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "non-finite adjusted predictor",
            ));
        }
        Ok(value)
    }
}

/// The affine model `y = β₀ + β₁ x₁ + … + β_p x_p` with one response.
///
/// The parameters are ordered `[β₀, β₁, …, β_p]`: intercept, then slopes.
pub struct AffineOdrModel {
    predictors: usize,
}
impl AffineOdrModel {
    /// Creates the affine model with `predictors` predictors.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `predictors` is zero.
    /// - `SizeOverflow` if `predictors + 1` overflows.
    pub fn new(predictors: usize) -> Result<Self, SciOxideError> {
        if predictors == 0 {
            return Err(invalid(
                scioxide_core::argument!("predictors"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "affine model needs at least one predictor",
            ));
        }
        if predictors.checked_add(1).is_none() {
            return Err(SciOxideError::size_overflow(scioxide_core::quantity!(
                "statistics_size"
            )));
        }
        Ok(Self { predictors })
    }
}
impl<S: RealScalar> OdrModel<S> for AffineOdrModel {
    fn parameter_count(&self) -> usize {
        self.predictors + 1
    }
    fn predictor_count(&self) -> usize {
        self.predictors
    }
    fn response_count(&self) -> usize {
        1
    }
    fn response(
        &self,
        b: &[S],
        x: OdrPredictors<'_, S>,
        coordinate: usize,
    ) -> Result<S, SciOxideError> {
        if coordinate != 0 || b.len() != self.predictors + 1 || x.len() != self.predictors {
            return Err(invalid(
                scioxide_core::argument!("b"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "affine model shape",
            ));
        }
        let mut sum = ProductSum::<S>::new();
        sum.add_product(b[0], S::ONE)?;
        for (j, &slope) in b[1..].iter().enumerate() {
            sum.add_product(slope, x.get(j)?)?;
        }
        sum.finish()
    }
    fn parameter_derivative(
        &self,
        b: &[S],
        x: OdrPredictors<'_, S>,
        response: usize,
        j: usize,
    ) -> Result<S, SciOxideError> {
        if response != 0
            || b.len() != self.predictors + 1
            || x.len() != self.predictors
            || j > self.predictors
        {
            return Err(invalid(
                scioxide_core::argument!("j"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "affine derivative index or shape",
            ));
        }
        if j == 0 { Ok(S::ONE) } else { x.get(j - 1) }
    }
    fn predictor_derivative(
        &self,
        b: &[S],
        x: OdrPredictors<'_, S>,
        response: usize,
        j: usize,
    ) -> Result<S, SciOxideError> {
        if response != 0 || b.len() != self.predictors + 1 || x.len() != self.predictors {
            return Err(invalid(
                scioxide_core::argument!("b"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "affine derivative shape",
            ));
        }
        b.get(j + 1).copied().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("j"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "affine predictor index",
            )
        })
    }
}

/// Orthogonal distance regression objective for a nonlinear explicit model.
///
/// Observation `i` has predictors `xᵢ ∈ ℝᵖ` and responses `yᵢ ∈ ℝ^q`, both
/// measured with independent normal errors of known standard deviations
/// `σ_x,ij` and `σ_y,ij`. The unknowns are the model parameters `β` and the
/// predictor corrections `δᵢ`, and the objective is
///
/// ```text
/// S(β, δ) = Σᵢ [ Σⱼ (δ_ij/σ_x,ij)² + Σⱼ ((f_j(β, xᵢ + δᵢ) − y_ij)/σ_y,ij)² ],
/// ```
///
/// without a factor ½. The parameter vector is `z = [β, δ₁, …, δₙ]`, of length
/// `k + np`. The plan implements [`FitObjective`] with `n(p + q)` residuals,
/// ordered by observation with the `p` predictor residuals first, and
/// [`FitDerivatives`] with the gradient and Jacobian. Fit it with
/// [`Self::fit_into`] and a [`FitSolver`](crate::FitSolver) such as a
/// Levenberg–Marquardt method. Correlated errors and implicit models are not
/// supported. For an affine model with the same standard deviations for every
/// observation, [`linear_odr_with_options`] solves the problem in closed form.
///
/// # References
///
/// P. T. Boggs, R. H. Byrd and R. B. Schnabel, "A stable and efficient algorithm
/// for nonlinear orthogonal distance regression", *SIAM Journal on Scientific
/// and Statistical Computing* 8 (1987), 1052–1078.
#[derive(Clone)]
pub struct PreparedOdr<'a, S: RealScalar> {
    model: &'a dyn OdrModel<S>,
    x: MatrixView<'a, S>,
    y: MatrixView<'a, S>,
    sx: MatrixView<'a, S>,
    sy: MatrixView<'a, S>,
    parameters: usize,
    residuals: usize,
    options: OdrOptions,
}
/// Numerical rank of the residual Jacobian of an ODR problem at a point.
///
/// The `np` predictor-correction columns always have full rank, so the rank
/// left for the `k` model parameters is the full rank minus `np`. The rank
/// depends on the rank policy and holds only near the point. The fields are
/// read-only:
///
/// ```compile_fail
/// fn forge<S: scioxide_core::RealScalar>(rank: &mut scioxide_stats::OdrIdentifiability<S>) {
///     rank.coefficient_rank = 0;
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct OdrIdentifiability<S: RealScalar> {
    coefficient_rank: usize,
    coefficient_count: usize,
    full_rank: RankReport<S>,
}
impl<S: RealScalar> OdrIdentifiability<S> {
    /// Returns the rank attributed to the model parameters: the full rank minus
    /// `np`, or zero if the full rank is below `np`.
    pub fn coefficient_rank(&self) -> usize {
        self.coefficient_rank
    }
    /// Returns the number of model parameters `k`.
    pub fn coefficient_count(&self) -> usize {
        self.coefficient_count
    }
    /// Returns the rank report of the full residual Jacobian.
    pub fn full_rank(&self) -> RankReport<S> {
        self.full_rank
    }
}
impl<'a, S: RealScalar> PreparedOdr<'a, S> {
    /// Creates the objective for `n` observations: predictors `x` (`n × p`),
    /// responses `y` (`n × q`), and their standard deviations `sx` and `sy` of
    /// the same shapes.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `n < 2`, if `p`, `q` or `k` is zero, if the model's
    ///   predictor or response count differs from the data, or if the shapes
    ///   disagree.
    /// - `InvalidInput` if `x` or `y` is not finite, or a standard deviation is
    ///   not finite and positive.
    pub fn new(
        model: &'a dyn OdrModel<S>,
        x: MatrixView<'a, S>,
        y: MatrixView<'a, S>,
        sx: MatrixView<'a, S>,
        sy: MatrixView<'a, S>,
        options: OdrOptions,
    ) -> Result<Self, SciOxideError> {
        let (n, p, q, k) = (x.rows(), x.cols(), y.cols(), model.parameter_count());
        if n < 2
            || p == 0
            || q == 0
            || k == 0
            || model.predictor_count() != p
            || model.response_count() != q
            || y.rows() != n
            || sx.rows() != n
            || sy.rows() != n
            || sx.cols() != p
            || sy.cols() != q
        {
            return Err(invalid(
                scioxide_core::argument!("y"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "ODR observation shapes",
            ));
        }
        let parameters = n
            .checked_mul(p)
            .and_then(|v| v.checked_add(k))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("odr_parameters"))
            })?;
        let residuals = n
            .checked_mul(p.checked_add(q).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("odr_residuals"))
            })?)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("odr_residuals"))
            })?;
        for i in 0..n {
            for j in 0..p {
                if !x.get(i, j)?.is_finite()
                    || !sx.get(i, j)?.is_finite()
                    || sx.get(i, j)? <= S::ZERO
                {
                    return Err(invalid(
                        scioxide_core::argument!("x"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "invalid predictor observation or standard deviation",
                    ));
                }
            }
            for j in 0..q {
                if !y.get(i, j)?.is_finite()
                    || !sy.get(i, j)?.is_finite()
                    || sy.get(i, j)? <= S::ZERO
                {
                    return Err(invalid(
                        scioxide_core::argument!("y"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "invalid response observation or standard deviation",
                    ));
                }
            }
        }
        Ok(Self {
            model,
            x,
            y,
            sx,
            sy,
            parameters,
            residuals,
            options,
        })
    }
    /// Returns the number of model parameters `k`, without the predictor corrections.
    pub fn coefficient_count(&self) -> usize {
        self.model.parameter_count()
    }
    /// Returns the starting point `z = [β, 0, …, 0]`: the given model parameters
    /// and zero predictor corrections.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `coefficients` does not have `k` finite entries.
    /// - `ResourceLimit` if `z` exceeds the byte limit in [`OdrOptions`].
    pub fn initial_parameters(&self, coefficients: &[S]) -> Result<Vec<S>, SciOxideError> {
        let max_bytes = self.options.max_bytes;
        if coefficients.len() != self.coefficient_count()
            || coefficients.iter().any(|x| !x.is_finite())
        {
            return Err(invalid(
                scioxide_core::argument!("coefficients"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid ODR initial coefficients",
            ));
        }
        let bytes = scioxide_core::backend::checked_mul(
            self.parameters,
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("odr_initial_bytes"),
        )?;
        crate::error_helpers::limit(
            bytes,
            max_bytes,
            scioxide_core::quantity!("odr_initial_bytes"),
        )?;
        let mut out = scioxide_core::backend::try_filled(
            self.parameters,
            S::ZERO,
            scioxide_core::quantity!("odr_initial_parameters"),
        )?;
        out[..coefficients.len()].copy_from_slice(coefficients);
        Ok(out)
    }
    fn checked<'b>(&self, z: &'b [S]) -> Result<(&'b [S], &'b [S]), SciOxideError> {
        if z.len() != self.parameters || z.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                scioxide_core::argument!("z"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "ODR parameter shape or finiteness",
            ));
        }
        Ok(z.split_at(self.coefficient_count()))
    }
    fn predictor<'b>(&self, z: &'b [S], row: usize) -> OdrPredictors<'b, S>
    where
        'a: 'b,
    {
        let p = self.x.cols();
        OdrPredictors {
            observed: self.x,
            row,
            corrections: &z[row * p..(row + 1) * p],
        }
    }
    /// Returns the negative log likelihood of the normal error model at `z`,
    /// including the normalization constants:
    ///
    /// ```text
    /// −ln L(z) = S(z)/2 + Σᵢⱼ ln σ_x,ij + Σᵢⱼ ln σ_y,ij + (n(p + q)/2) ln 2π.
    /// ```
    ///
    /// The predictor corrections are treated as parameters, not integrated out.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `z` does not have `k + np` finite entries, or an
    ///   adjusted predictor overflows.
    /// - The errors of the model.
    /// - `NumericalInstability` if the result overflows.
    pub fn gaussian_negative_log_likelihood(&self, z: &[S]) -> Result<S, SciOxideError> {
        let mut value = self.value(z, &mut ())? / S::try_from_f64_lossy(2.)?;
        let half_log_two_pi = S::try_from_f64_lossy(0.9189385332046727)?;
        for i in 0..self.x.rows() {
            for j in 0..self.x.cols() {
                value = value + self.sx.get(i, j)?.ln() + half_log_two_pi;
            }
            for j in 0..self.y.cols() {
                value = value + self.sy.get(i, j)?.ln() + half_log_two_pi;
            }
        }
        if !value.is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "non-finite Gaussian likelihood",
            ));
        }
        Ok(value)
    }
    /// Computes the numerical rank of the residual Jacobian at `z`.
    ///
    /// The Jacobian has `m = n(p + q)` rows and `d = k + np` columns. The cost
    /// is one thin SVD, O(md·min(m, d)), with O(md + d²) memory, which must fit
    /// in the byte limit of [`OdrOptions`]. A coefficient rank below `k` means
    /// that the data do not determine all model parameters near `z`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `z` does not have `k + np` finite entries.
    /// - `ResourceLimit` if the Jacobian and SVD storage exceed the byte limit.
    /// - The errors of the model, and `NumericalInstability` if a Jacobian entry
    ///   overflows.
    /// - `ConvergenceFailure` if the SVD does not converge.
    pub fn local_identifiability(
        &self,
        z: &[S],
        policy: RankPolicy<S>,
    ) -> Result<OdrIdentifiability<S>, SciOxideError> {
        let max_bytes = self.options.max_bytes;
        self.checked(z)?;
        let d = self.parameters;
        let r = self.residuals;
        let elements = r
            .checked_mul(d)
            .and_then(|v| v.checked_add(d.checked_mul(d)?))
            .and_then(|v| v.checked_mul(32))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("odr_elements"))
            })?;
        crate::error_helpers::limit(
            scioxide_core::backend::checked_mul(
                elements,
                core::mem::size_of::<S>(),
                scioxide_core::quantity!("odr_bytes"),
            )?,
            max_bytes,
            scioxide_core::quantity!("odr_bytes"),
        )?;
        let mut matrix = Matrix::zeros(r, d)?;
        self.jacobian(z, matrix.as_mut_slice())?;
        let factor = scioxide_linalg::ThinSvdDecomposition::compute(&matrix)?;
        let full_rank = factor.analyze_rank(policy)?;
        let nuisance = self.x.rows() * self.x.cols();
        Ok(OdrIdentifiability {
            coefficient_rank: full_rank.rank().saturating_sub(nuisance),
            coefficient_count: self.coefficient_count(),
            full_rank,
        })
    }
}
impl<S: RealScalar> FitObjective<S> for PreparedOdr<'_, S> {
    type Workspace = ();
    fn workspace_bytes(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn workspace(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn dimension(&self) -> usize {
        self.parameters
    }
    fn validate(&self, z: &[S], _: &mut ()) -> Result<(), SciOxideError> {
        let (b, d) = self.checked(z)?;
        for i in 0..self.x.rows() {
            let x = self.predictor(d, i);
            for j in 0..self.x.cols() {
                x.get(j)?;
            }
            for j in 0..self.y.cols() {
                let value = self.model.response(b, x, j)?;
                if !value.is_finite() {
                    return Err(invalid(
                        scioxide_core::argument!("z"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "non-finite model response",
                    ));
                }
            }
        }
        Ok(())
    }
    fn value(&self, z: &[S], _: &mut ()) -> Result<S, SciOxideError> {
        let (b, d) = self.checked(z)?;
        let mut total = ProductSum::<S>::new();
        for i in 0..self.x.rows() {
            let adjusted = self.predictor(d, i);
            for j in 0..self.x.cols() {
                let delta = d[i * self.x.cols() + j];
                adjusted.get(j)?;
                let r = delta / self.sx.get(i, j)?;
                total.add_product(r, r)?;
            }
            for j in 0..self.y.cols() {
                let r = (self.model.response(b, adjusted, j)? - self.y.get(i, j)?)
                    / self.sy.get(i, j)?;
                total.add_product(r, r)?;
            }
        }
        total.finish()
    }
    fn residual_count(&self) -> Option<usize> {
        Some(self.residuals)
    }
    fn residuals(&self, z: &[S], out: &mut [S], _: &mut ()) -> Result<(), SciOxideError> {
        let (b, d) = self.checked(z)?;
        if out.len() != self.residuals {
            return Err(invalid(
                scioxide_core::argument!("out"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "ODR residual shape",
            ));
        }
        let (p, q) = (self.x.cols(), self.y.cols());
        for i in 0..self.x.rows() {
            let adjusted = self.predictor(d, i);
            for j in 0..p {
                let delta = d[i * p + j];
                adjusted.get(j)?;
                out[i * (p + q) + j] = delta / self.sx.get(i, j)?;
            }
            for j in 0..q {
                out[i * (p + q) + p + j] = (self.model.response(b, adjusted, j)?
                    - self.y.get(i, j)?)
                    / self.sy.get(i, j)?;
            }
        }
        Ok(())
    }
}

impl<S: RealScalar> FitDerivatives<S> for PreparedOdr<'_, S> {
    fn capabilities(&self) -> FitDerivativeCapabilities {
        FitDerivativeCapabilities {
            gradient: true,
            hessian: false,
            jacobian: true,
        }
    }
    fn gradient(&self, z: &[S], out: &mut [S]) -> Result<(), SciOxideError> {
        let (b, d) = self.checked(z)?;
        if out.len() != z.len() {
            return Err(invalid(
                scioxide_core::argument!("out"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "ODR gradient shape",
            ));
        }
        out.fill(S::ZERO);
        let (n, p, q, k) = (self.x.rows(), self.x.cols(), self.y.cols(), b.len());
        for i in 0..n {
            let adjusted = self.predictor(d, i);
            for j in 0..p {
                adjusted.get(j)?;
                out[k + i * p + j] = S::try_from_f64_lossy(2.)? * d[i * p + j]
                    / (self.sx.get(i, j)? * self.sx.get(i, j)?);
            }
            for a in 0..q {
                let coefficient = S::try_from_f64_lossy(2.)?
                    * (self.model.response(b, adjusted, a)? - self.y.get(i, a)?)
                    / (self.sy.get(i, a)? * self.sy.get(i, a)?);
                for (j, entry) in out[..k].iter_mut().enumerate() {
                    *entry += coefficient * self.model.parameter_derivative(b, adjusted, a, j)?;
                }
                for j in 0..p {
                    out[k + i * p + j] +=
                        coefficient * self.model.predictor_derivative(b, adjusted, a, j)?;
                }
            }
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "non-finite ODR gradient",
            ));
        }
        Ok(())
    }
    fn jacobian(&self, z: &[S], out: &mut [S]) -> Result<(), SciOxideError> {
        let (b, d) = self.checked(z)?;
        let (n, p, q, k) = (self.x.rows(), self.x.cols(), self.y.cols(), b.len());
        if out.len()
            != self.residuals.checked_mul(self.parameters).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("odr_jacobian_elements"))
            })?
        {
            return Err(invalid(
                scioxide_core::argument!("out"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "ODR Jacobian shape",
            ));
        }
        out.fill(S::ZERO);
        for i in 0..n {
            let adjusted = self.predictor(d, i);
            for j in 0..p {
                adjusted.get(j)?;
                out[(i * (p + q) + j) * self.parameters + k + i * p + j] =
                    S::ONE / self.sx.get(i, j)?;
            }
            for a in 0..q {
                let row = (i * (p + q) + p + a) * self.parameters;
                let scale = self.sy.get(i, a)?;
                for j in 0..k {
                    out[row + j] = self.model.parameter_derivative(b, adjusted, a, j)? / scale;
                }
                for j in 0..p {
                    out[row + k + i * p + j] =
                        self.model.predictor_derivative(b, adjusted, a, j)? / scale;
                }
            }
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "non-finite ODR Jacobian",
            ));
        }
        Ok(())
    }
}

/// Options for orthogonal distance regression: a byte limit.
#[derive(Clone, Copy, Debug)]
pub struct OdrOptions {
    max_bytes: usize,
}
impl Default for OdrOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl OdrOptions {
    /// Returns the default options, with no byte limit.
    pub const fn new() -> Self {
        Self {
            max_bytes: usize::MAX,
        }
    }
    /// Sets the limit, in bytes, on the memory used for preparation, results and
    /// workspaces.
    pub const fn with_max_bytes(mut self, bytes: usize) -> Self {
        self.max_bytes = bytes;
        self
    }
    /// Returns the byte limit.
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }
}
/// Result of an orthogonal distance regression: model parameters, predictor
/// corrections and a report.
pub struct OdrFit<S: RealScalar> {
    coefficients: Vec<S>,
    predictor_corrections: Vec<S>,
    report: OdrReport<S>,
}
impl<S: RealScalar> OdrFit<S> {
    /// Returns the model parameters `β`. For an affine model these are the
    /// intercept, then the slopes.
    pub fn coefficients(&self) -> &[S] {
        &self.coefficients
    }
    /// Returns the predictor corrections `δ`, `p` per observation, observation by
    /// observation.
    pub fn predictor_corrections(&self) -> &[S] {
        &self.predictor_corrections
    }
    /// Returns the termination status, objective value and identifiability diagnostics.
    pub fn report(&self) -> &OdrReport<S> {
        &self.report
    }
}
/// Diagnostic of whether the data determine the model parameters.
#[derive(Clone, Copy, Debug)]
pub enum OdrIdentifiabilityDiagnostic<S: RealScalar> {
    /// From [`linear_odr_with_options`]: the two smallest singular values of the scaled data
    /// are separated, so the fitted hyperplane is unique.
    SingularGap {
        /// Difference between the two smallest singular values.
        gap: S,
        /// Threshold the gap had to exceed: `ε max(n, p + 1) σ₁`, where `σ₁` is the
        /// largest singular value.
        threshold: S,
    },
    /// Numerical rank of the residual Jacobian, as from
    /// [`PreparedOdr::local_identifiability`].
    WeightedJacobian(OdrIdentifiability<S>),
}
/// Status and diagnostics of an orthogonal distance regression.
#[derive(Clone, Copy, Debug)]
pub struct OdrReport<S: RealScalar> {
    termination: crate::StatsTermination,
    chi_square: Option<S>,
    counts: scioxide_core::SolveCounts,
    identifiability: Option<OdrIdentifiabilityDiagnostic<S>>,
}
impl<S: RealScalar> OdrReport<S> {
    /// Returns the termination status. [`linear_odr_with_options`] always reports `Converged`.
    pub fn termination(self) -> crate::StatsTermination {
        self.termination
    }
    /// Returns the objective `S(β, δ)` at the fit, or `None` if the fit stopped
    /// before its final evaluation.
    pub fn chi_square(self) -> Option<S> {
        self.chi_square
    }
    /// Returns the iteration and evaluation counts. They are zero for [`linear_odr_with_options`].
    pub fn counts(self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the identifiability diagnostic of [`linear_odr_with_options`], or `None` for
    /// [`PreparedOdr::fit_into`]; call [`PreparedOdr::local_identifiability`] there.
    pub fn identifiability(self) -> Option<OdrIdentifiabilityDiagnostic<S>> {
        self.identifiability
    }
}

/// Computes the result of [`linear_odr_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`linear_odr_with_options`] for the fixed default settings.
pub fn linear_odr<S: RealScalar>(
    predictors: MatrixView<'_, S>,
    response: &[S],
    predictor_errors: &[S],
    response_error: S,
) -> Result<OdrFit<S>, SciOxideError> {
    linear_odr_with_options(
        predictors,
        response,
        predictor_errors,
        response_error,
        Default::default(),
    )
    .map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("predictors"),
        )
    })
}
/// Fits the affine model `y = β₀ + β₁ x₁ + … + β_p x_p` by orthogonal distance
/// regression, in closed form.
///
/// Each predictor `j` has a known error standard deviation `σ_j` and the
/// response has `σ_y`, the same for every observation. The function scales the
/// centered columns of `[X y]` by these deviations and computes their thin SVD.
/// The right singular vector `v` of the smallest singular value `σ_min` is the
/// normal of the fitted hyperplane, which minimizes the objective of
/// [`PreparedOdr`]. The slopes are `β_j = −σ_y v_j/(σ_j v_y)`, the minimum
/// objective is `σ_min²`, and the predictor corrections move each scaled point
/// orthogonally onto the hyperplane. This is the total least squares solution
/// of the scaled problem. The cost is O(n(p + 1)²). It runs on the calling
/// thread and uses no optimizer.
///
/// # Errors
///
/// - `InvalidInput` if `p` is zero, `n ≤ p`, `response` does not have `n`
///   entries, or `predictor_errors` does not have `p` entries.
/// - `InvalidInput` if `response` is not finite, or a standard deviation is not
///   finite and positive.
/// - `NonFinite` if `predictors` contains NaN or infinity.
/// - `InvalidInput` if the hyperplane is not unique, because the two smallest
///   singular values differ by at most `ε max(n, p + 1) σ₁`, where `σ₁` is the
///   largest.
/// - `InvalidInput` if the hyperplane is nearly parallel to the response axis:
///   `|v_y| ≤ (p + 1) ε`.
/// - `ResourceLimit` if the memory needed exceeds the byte limit in `options`.
/// - `ConvergenceFailure` if the SVD does not converge.
/// - `NumericalInstability` if a coefficient, the objective or a correction
///   overflows.
///
/// # Examples
///
/// ```
/// use scioxide_linalg::MatrixView;
/// use scioxide_stats::{OdrOptions, linear_odr_with_options};
///
/// // Points on y = 1 + 2x: the slope is 2 for any error scales.
/// let x = [0.0_f64, 1.0, 2.0, 3.0];
/// let y = [1.0, 3.0, 5.0, 7.0];
/// let predictors = MatrixView::from_slice(4, 1, &x)?;
/// let fit = linear_odr_with_options(predictors, &y, &[1.0], 1.0, OdrOptions::new())?;
/// assert!((fit.coefficients()[1] - 2.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// G. H. Golub and C. F. Van Loan, "An analysis of the total least squares
/// problem", *SIAM Journal on Numerical Analysis* 17 (1980), 883–893.
pub fn linear_odr_with_options<S: RealScalar>(
    predictors: MatrixView<'_, S>,
    response: &[S],
    predictor_errors: &[S],
    response_error: S,
    options: OdrOptions,
) -> Result<OdrFit<S>, SciOxideError> {
    let context = ExecutionContext::sequential_native();
    let max_bytes = options.max_bytes;
    let (n, p) = (predictors.rows(), predictors.cols());
    if p == 0
        || n <= p
        || response.len() != n
        || predictor_errors.len() != p
        || !response_error.is_finite()
        || response_error <= S::ZERO
        || predictor_errors
            .iter()
            .any(|&s| !s.is_finite() || s <= S::ZERO)
        || response.iter().any(|&y| !y.is_finite())
    {
        return Err(invalid(
            scioxide_core::argument!("predictor_errors"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "linear ODR shape or error scales",
        ));
    }
    let columns = p.checked_add(1).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("multivariate_columns"))
    })?;
    let elements = n
        .checked_mul(columns)
        .and_then(|v| columns.checked_mul(columns).and_then(|q| v.checked_add(q)))
        .and_then(|v| n.checked_mul(n).and_then(|q| v.checked_add(q)))
        .and_then(|v| v.checked_mul(32))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("multivariate_elements"))
        })?;
    reserve_multivariate::<S>(elements, columns, 0, max_bytes)?;
    let mut augmented = Matrix::zeros(n, p + 1)?;
    for (i, &y) in response.iter().enumerate() {
        for j in 0..p {
            augmented.view_mut().set(i, j, predictors.get(i, j)?)?;
        }
        augmented.view_mut().set(i, p, y)?;
    }
    let CenteredData {
        means,
        values: mut centered,
        ..
    } = centered(augmented.view(), max_bytes, context)?;
    for i in 0..n {
        for (j, &sigma) in predictor_errors.iter().enumerate() {
            let value = centered.view().get(i, j)? / sigma;
            centered.view_mut().set(i, j, value)?;
        }
        let value = centered.view().get(i, p)? / response_error;
        centered.view_mut().set(i, p, value)?;
    }
    let factor = factor(&centered, context)?;
    let singular = factor.singular_values();
    let last = singular[p];
    let gap = singular[p - 1] - last;
    let threshold = S::EPSILON * number::<S>(n.max(p + 1))? * singular[0];
    let vy = factor.vt_matrix().view().get(p, p)?;
    if gap <= threshold || vy.abs() <= S::EPSILON * number::<S>(p + 1)? {
        return Err(invalid(
            scioxide_core::argument!("predictors"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "linear ODR hyperplane is not identifiable",
        ));
    }
    let mut slopes = scioxide_core::backend::try_filled(
        p,
        S::ZERO,
        scioxide_core::quantity!("multivariate_slopes"),
    )?;
    for (j, slope) in slopes.iter_mut().enumerate() {
        *slope =
            -response_error * factor.vt_matrix().view().get(p, j)? / (predictor_errors[j] * vy);
        if !slope.is_finite() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "unrepresentable ODR slope",
            ));
        }
    }
    let intercept = means[p]
        - slopes
            .iter()
            .zip(&means)
            .fold(S::ZERO, |sum, (&b, &x)| sum + b * x);
    let chi_square = last * last;
    if !intercept.is_finite() || !chi_square.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "unrepresentable ODR fit",
        ));
    }
    let mut coefficients = scioxide_core::backend::try_with_capacity(
        p + 1,
        scioxide_core::quantity!("odr_coefficients"),
    )?;
    coefficients.push(intercept);
    coefficients.extend_from_slice(&slopes);
    let mut predictor_corrections = scioxide_core::backend::try_filled(
        n * p,
        S::ZERO,
        scioxide_core::quantity!("odr_predictor_corrections"),
    )?;
    for i in 0..n {
        let mut projection = ProductSum::new();
        for j in 0..=p {
            projection.add_product(
                centered.view().get(i, j)?,
                factor.vt_matrix().view().get(p, j)?,
            )?;
        }
        let distance = projection.finish()?;
        for (j, &sigma) in predictor_errors.iter().enumerate() {
            let delta = -sigma * factor.vt_matrix().view().get(p, j)? * distance;
            if !delta.is_finite() {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Overflow,
                    "ODR predictor correction overflow",
                ));
            }
            predictor_corrections[i * p + j] = delta;
        }
    }
    Ok(OdrFit {
        coefficients,
        predictor_corrections,
        report: OdrReport {
            termination: crate::StatsTermination::Converged,
            chi_square: Some(chi_square),
            counts: scioxide_core::SolveCounts::new(),
            identifiability: Some(OdrIdentifiabilityDiagnostic::SingularGap { gap, threshold }),
        },
    })
}

/// Inputs of [`PreparedOdr::fit_into`] besides the plan: starting point,
/// constraints and solver.
pub struct OdrInputs<'a, S: RealScalar> {
    /// Starting point `z = [β, δ]`, within the bounds, for example from
    /// [`PreparedOdr::initial_parameters`].
    pub initial: &'a [S],
    /// Bounds on all `k + np` entries of `z`.
    pub bounds: &'a scioxide_core::Bounds<S>,
    /// Flags that mark integer entries of `z`, one per entry.
    pub integrality: &'a [bool],
    /// The solver that minimizes the objective.
    pub solver: &'a mut dyn crate::FitSolver<S>,
}
/// Output storage for [`PreparedOdr::fit_into`].
///
/// It holds a fit only after a call returns a fitted point.
pub struct OdrOutput<S: RealScalar> {
    fit: OdrFit<S>,
    published: bool,
}
impl<S: RealScalar> OdrOutput<S> {
    /// Returns the last fit written, or `None` before the first.
    pub fn fit(&self) -> Option<&OdrFit<S>> {
        self.published.then_some(&self.fit)
    }
}
/// Scratch storage for [`PreparedOdr::fit_into`].
///
/// It also records the most recent call. The solver's own scratch is not
/// included. See [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
pub struct OdrWorkspace<S: RealScalar> {
    fit: crate::FitWorkspace<S>,
    candidate: Vec<S>,
    slot: crate::execution::SolveSlot<OdrReport<S>>,
}
impl<S: RealScalar> OdrWorkspace<S> {
    /// Returns the record of the most recent call, including one rejected before
    /// the solver ran, or `None` before the first call.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<OdrReport<S>>> {
        self.slot.last_attempt()
    }
    /// Returns the report of the most recent call if it produced a fitted point.
    pub fn last_accepted(&self) -> Option<&OdrReport<S>> {
        self.slot
            .last_attempt()
            .and_then(scioxide_core::AttemptRecord::progress)
    }
}
impl<S: RealScalar> PreparedOdr<'_, S> {
    fn admit_fit_storage(&self) -> Result<(), SciOxideError> {
        let bytes = self
            .parameters
            .checked_mul(3)
            .and_then(|n| n.checked_mul(core::mem::size_of::<S>()))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("odr_fit_storage"))
            })?;
        crate::error_helpers::limit_bytes(bytes, self.options.max_bytes)
    }
    /// Creates output storage for [`Self::fit_into`].
    ///
    /// # Errors
    ///
    /// `ResourceLimit` if the storage exceeds the byte limit in [`OdrOptions`].
    pub fn output(&self) -> Result<OdrOutput<S>, SciOxideError> {
        self.admit_fit_storage()?;
        Ok(OdrOutput {
            fit: OdrFit {
                coefficients: scioxide_core::backend::try_filled(
                    self.coefficient_count(),
                    S::ZERO,
                    scioxide_core::quantity!("odr_coefficients"),
                )?,
                predictor_corrections: scioxide_core::backend::try_filled(
                    self.parameters - self.coefficient_count(),
                    S::ZERO,
                    scioxide_core::quantity!("odr_corrections"),
                )?,
                report: OdrReport {
                    termination: crate::StatsTermination::NotConverged,
                    chi_square: None,
                    counts: scioxide_core::SolveCounts::new(),
                    identifiability: None,
                },
            },
            published: false,
        })
    }
    /// Creates a workspace for [`Self::fit_into`].
    ///
    /// # Errors
    ///
    /// `ResourceLimit` if the workspace exceeds the byte limit in [`OdrOptions`].
    pub fn workspace(&self) -> Result<OdrWorkspace<S>, SciOxideError> {
        self.admit_fit_storage()?;
        Ok(OdrWorkspace {
            fit: crate::FitWorkspace::new(
                self.parameters,
                crate::StatisticsOptions::new().with_max_bytes(self.options.max_bytes),
            )?,
            candidate: scioxide_core::backend::try_filled(
                self.parameters,
                S::ZERO,
                scioxide_core::quantity!("odr_candidate"),
            )?,
            slot: Default::default(),
        })
    }
    /// Minimizes the objective with the solver in `inputs` and writes the fit to
    /// `output`, as a root or nested solve.
    ///
    /// Runs [`fit_into`](crate::fit_into) with this plan as objective and
    /// derivatives. `output` receives the fit only if the final objective
    /// evaluation completes, even when the status is not `Converged`. See
    /// [Results and termination](crate#results-and-termination).
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `inputs.initial` does not have `k + np` entries, or
    ///   `output` or `work` was created for another plan shape.
    /// - The errors of [`fit_into`](crate::fit_into).
    pub fn fit_into<'scope, 'capability>(
        &self,
        inputs: OdrInputs<'_, S>,
        output: &mut OdrOutput<S>,
        work: &mut OdrWorkspace<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<OdrReport<S>, SciOxideError> {
        work.slot.reject();
        if inputs.initial.len() != self.parameters
            || output.fit.coefficients.len() != self.coefficient_count()
            || output.fit.predictor_corrections.len() != self.parameters - self.coefficient_count()
            || work.candidate.len() != self.parameters
        {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "ODR fit shapes",
            ));
        }
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("stats", "odr-fit", 1),
            self.parameters as u128,
            0,
        );
        let mut slot = core::mem::take(&mut work.slot);
        let result = slot.run(
            scope.into(),
            identity,
            scioxide_core::SolveBudgets::unbounded(),
            |context| {
                let report = crate::fit_into(
                    crate::FitRequest {
                        objective: self,
                        derivatives: Some(self),
                        initial: inputs.initial,
                        bounds: inputs.bounds,
                        integrality: inputs.integrality,
                    },
                    inputs.solver,
                    &mut work.candidate,
                    &mut work.fit,
                    context,
                )?;
                Ok(OdrReport {
                    termination: report.termination(),
                    chi_square: report.objective(),
                    counts: report.counts(),
                    identifiability: None,
                })
            },
            |result| match result {
                Ok(report) => (
                    if report.termination == crate::StatsTermination::Converged {
                        scioxide_core::AttemptOutcome::Succeeded
                    } else {
                        scioxide_core::AttemptOutcome::Stopped
                    },
                    report.chi_square.map(|_| *report),
                ),
                Err(_) => (scioxide_core::AttemptOutcome::Failed, None),
            },
        );
        work.slot = slot;
        if let Ok(report) = &result
            && report.chi_square.is_some()
        {
            let k = self.coefficient_count();
            output
                .fit
                .coefficients
                .copy_from_slice(&work.candidate[..k]);
            output
                .fit
                .predictor_corrections
                .copy_from_slice(&work.candidate[k..]);
            output.fit.report = *report;
            output.published = true;
        }
        result
    }
    /// Minimizes the objective with the solver in `inputs` and returns the fit.
    ///
    /// Allocates the output and workspace for one call. Use [`Self::fit_into`] to
    /// reuse them.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::output`], [`Self::workspace`] and [`Self::fit_into`].
    /// - `ConvergenceFailure` if the fit stopped before its final objective evaluation.
    pub fn fit_scoped<'scope, 'capability>(
        &self,
        inputs: OdrInputs<'_, S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<OdrFit<S>, SciOxideError> {
        let mut output = self.output()?;
        let mut work = self.workspace()?;
        let report = self.fit_into(inputs, &mut output, &mut work, scope)?;
        if report.chi_square.is_none() {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::EvaluationLimit,
                scioxide_core::ConvergenceDiagnostics::new().with_counts(report.counts),
                "ODR stopped before candidate verification",
            ));
        }
        Ok(output.fit)
    }
    /// Executes [`Self::fit_scoped`] with sequential native capabilities.
    ///
    /// # Errors
    ///
    /// Returns the scoped form's input, numerical and allocation errors.
    pub fn fit(&self, inputs: OdrInputs<'_, S>) -> Result<OdrFit<S>, SciOxideError> {
        self.fit_scoped(inputs, scioxide_core::ExecutionContext::sequential_native())
    }
}
