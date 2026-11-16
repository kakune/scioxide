//! Half-deviance objective of a dense GLM, for fitting with an optimizer you provide.
use super::dense::DenseGlm;
use super::family::nonnegative_deviance;
use crate::regression::{finite, invalid};
use scioxide_core::{ProductSum, RealScalar, Scaled, ScaledAccumulator, SciOxideError};

/// Evaluates half the weighted deviance of a dense GLM and its gradient, as a
/// [`crate::FitObjective`] for [`crate::fit_into`].
///
/// For coefficients `β`, the objective is
///
/// ```text
/// f(β) = ½ Σᵢ wᵢ mᵢ d(yᵢ, μᵢ(β)),   μᵢ(β) = g⁻¹(xᵢᵀβ + oᵢ),
/// ```
///
/// where `d` is the unit deviance of the [`crate::GlmFamily`], `wᵢ` the row
/// weight, `mᵢ` the number of binomial trials (one for other families) and
/// `oᵢ` the offset. This is the negative log-likelihood at unit dispersion,
/// without the terms that do not depend on `β`. Each evaluation of `f` or its
/// gradient costs O(np) for an `n × p` design.
///
/// The unit deviances are summed in an extended exponent range and rounded
/// once. An overflowing value or gradient is an error, never clamped. On an
/// error the gradient output may be partly written. Only dense models are
/// supported. [`crate::PreparedGlm::fit_into`] with [`crate::GlmSolver::Injected`]
/// uses this objective.
///
/// # Examples
///
/// ```
/// // y = 1e308 and μ = 1/η = 1e−10: the half deviance overflows f64.
/// use scioxide_linalg::Matrix;
/// use scioxide_stats::{
///     FitObjective, GlmFamily, GlmObjective, GlmStorageOptions, PreparedGlm, Unweighted,
/// };
/// let design = Matrix::from_vec(3, 1, vec![1.0_f64; 3])?;
/// let model = PreparedGlm::new(
///     design.view(),
///     GlmFamily::Gamma,
///     Unweighted,
///     None,
///     None,
///     None,
///     GlmStorageOptions::new().with_max_bytes(4096),
/// )?;
/// let objective = GlmObjective::new(&model, &[1e308; 3])?;
/// assert!(objective.value(&[1e10], &mut ()).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct GlmObjective<'a, S: RealScalar> {
    model: &'a DenseGlm<S>,
    response: &'a [S],
}
impl<'a, S: RealScalar> GlmObjective<'a, S> {
    /// Returns the objective of `model` for the response `response`.
    ///
    /// To fit the model, use [`crate::PreparedGlm::fit_into`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `response` does not have one value per row, or a
    ///   value lies outside the family's domain.
    pub fn new(model: &'a super::PreparedGlm<S>, response: &'a [S]) -> Result<Self, SciOxideError> {
        Self::from_model(&model.raw, response)
    }

    /// Checks and borrows a response. Binomial responses are proportions of the
    /// model's trials.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` for a wrong length or a value outside the
    /// family's domain.
    fn from_model(model: &'a DenseGlm<S>, response: &'a [S]) -> Result<Self, SciOxideError> {
        if response.len() != model.design.rows()
            || response.iter().any(|&y| !model.family.valid_response(y))
        {
            return Err(invalid(
                scioxide_core::argument!("response"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "GLM fitting response shape or domain",
            ));
        }
        Ok(Self { model, response })
    }
    fn row(&self, parameters: &[S], i: usize) -> Result<(S, S, S), SciOxideError> {
        let mut eta = ProductSum::new();
        for (j, &b) in parameters.iter().enumerate() {
            eta.add_product(self.model.design.view().get(i, j)?, b)?;
        }
        let eta = finite(eta.finish()? + self.model.rows.offsets.view().get(i)?)?;
        self.model.family.mean(eta).ok_or_else(|| {
            invalid(
                scioxide_core::argument!("parameters"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "GLM fitting predictor left its domain",
            )
        })
    }
    fn shape(&self, parameters: &[S]) -> Result<(), SciOxideError> {
        if parameters.len() != self.model.design.cols() || parameters.iter().any(|b| !b.is_finite())
        {
            return Err(invalid(
                scioxide_core::argument!("parameters"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "GLM fitting parameter shape or domain",
            ));
        }
        Ok(())
    }
}
impl<S: RealScalar> crate::FitObjective<S> for GlmObjective<'_, S> {
    type Workspace = ();
    fn workspace_bytes(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn workspace(&self) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn dimension(&self) -> usize {
        self.model.design.cols()
    }
    fn validate(&self, p: &[S], _: &mut ()) -> Result<(), SciOxideError> {
        self.shape(p)?;
        for i in 0..self.response.len() {
            self.row(p, i)?;
        }
        Ok(())
    }
    fn value(&self, p: &[S], _: &mut ()) -> Result<S, SciOxideError> {
        self.shape(p)?;
        let mut total = ScaledAccumulator::ZERO;
        let half = Scaled::HALF;
        for (i, &y) in self.response.iter().enumerate() {
            let (mu, _, _) = self.row(p, i)?;
            total.add_product(
                Scaled::from_value(self.model.row_weight(i)?)?.mul(half)?,
                self.model.family.deviance(y, mu)?,
            )?;
        }
        nonnegative_deviance(total.total()?)?.round()
    }
}
impl<S: RealScalar> crate::FitDerivatives<S> for GlmObjective<'_, S> {
    fn capabilities(&self) -> crate::FitDerivativeCapabilities {
        crate::FitDerivativeCapabilities {
            gradient: true,
            ..crate::FitDerivativeCapabilities::default()
        }
    }
    fn gradient(&self, p: &[S], output: &mut [S]) -> Result<(), SciOxideError> {
        self.shape(p)?;
        if output.len() != p.len() {
            return Err(invalid(
                scioxide_core::argument!("output"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "GLM fit gradient shape",
            ));
        }
        output.fill(S::ZERO);
        for (i, &y) in self.response.iter().enumerate() {
            let (mu, _, _) = self.row(p, i)?;
            let score = (-self.model.family.score(y, mu)?)
                .mul(Scaled::from_value(self.model.row_weight(i)?)?)?
                .round()?;
            for (j, g) in output.iter_mut().enumerate() {
                let mut sum = ProductSum::new();
                sum.add_product(*g, S::ONE)?;
                sum.add_product(self.model.design.view().get(i, j)?, score)?;
                *g = sum.finish()?;
            }
        }
        Ok(())
    }
}
