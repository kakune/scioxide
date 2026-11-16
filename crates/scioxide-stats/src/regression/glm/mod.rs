//! Generalized linear models with canonical links, fitted by iteratively
//! reweighted least squares or by an optimizer you provide.
mod api;
mod dense;
mod family;
mod irls;
mod objective;
mod operator;
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;

pub use api::{
    GlmAccepted, GlmBatchInputs, GlmBatchOutput, GlmBatchWorkspace, GlmFit, GlmFitView, GlmInputs,
    GlmOperatorWorkspace, GlmOutput, GlmResult, GlmSolver, GlmStorageOptions, GlmWorkspace,
    PreparedGlm, PreparedOperatorGlm, glm_batch_into,
};
pub use family::{GlmFamily, GlmLink};
pub use objective::GlmObjective;

use crate::StatsTermination;
use scioxide_core::{Budget, GradientTolerance, RealScalar, StepTolerance};
use scioxide_linalg::RankPolicy;

/// Options of a GLM fit: stopping tolerances, iteration limits, the rank
/// policy and a byte limit.
///
/// The iteration limit of the solve scope also applies. See
/// [Solve accounting](scioxide_core#solve-accounting).
#[derive(Clone, Copy, Debug)]
pub struct GlmOptions<S: RealScalar> {
    /// Tolerance on the coefficient change between iterations.
    tolerance: StepTolerance<S>,
    /// Tolerance on the score norm.
    score_tolerance: GradientTolerance<S>,
    /// Maximum number of IRLS iterations.
    iterations: Budget,
    /// Iteration limit of each SVD or LSMR solve.
    factor_iterations: Budget,
    /// Rule that sets the numerical rank threshold.
    rank_policy: RankPolicy<S>,
    max_bytes: usize,
}

impl<S: RealScalar> GlmOptions<S> {
    /// Sets the tolerance on the coefficient change.
    ///
    /// It is compared with the largest absolute change of a coefficient,
    /// relative to the largest absolute coefficient.
    pub fn with_tolerance(mut self, value: StepTolerance<S>) -> Self {
        self.tolerance = value;
        self
    }
    /// Returns the tolerance on the coefficient change.
    pub fn tolerance(&self) -> StepTolerance<S> {
        self.tolerance
    }
    /// Sets the tolerance on the score norm.
    ///
    /// It is compared with `‖∇ℓ(β)‖₂` relative to `1 + D`, where `ℓ` is the
    /// log-likelihood at unit dispersion and `D` the deviance.
    pub fn with_score_tolerance(mut self, value: GradientTolerance<S>) -> Self {
        self.score_tolerance = value;
        self
    }
    /// Returns the tolerance on the score norm.
    pub fn score_tolerance(&self) -> GradientTolerance<S> {
        self.score_tolerance
    }
    /// Sets the maximum number of IRLS iterations. The default is 100.
    pub fn with_iterations(mut self, value: Budget) -> Self {
        self.iterations = value;
        self
    }
    /// Returns the maximum number of IRLS iterations.
    pub fn iterations(&self) -> Budget {
        self.iterations
    }
    /// Sets the iteration limit of each SVD for a dense design, or of each
    /// LSMR solve for an operator design. The default is 4096.
    pub fn with_factor_iterations(mut self, value: Budget) -> Self {
        self.factor_iterations = value;
        self
    }
    /// Returns the iteration limit of each SVD or LSMR solve.
    pub fn factor_iterations(&self) -> Budget {
        self.factor_iterations
    }
    /// Sets the rule for the numerical rank of the weighted dense design. The
    /// default is `RankPolicy::DimensionScaledEpsilon`.
    pub fn with_rank_policy(mut self, value: RankPolicy<S>) -> Self {
        self.rank_policy = value;
        self
    }
    /// Returns the rank policy.
    pub fn rank_policy(&self) -> RankPolicy<S> {
        self.rank_policy
    }
    /// Sets the byte limit of the workspace. The default is `usize::MAX`.
    pub fn with_max_bytes(mut self, value: usize) -> Self {
        self.max_bytes = value;
        self
    }
    /// Returns the byte limit of the workspace.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }
}

impl<S: RealScalar> Default for GlmOptions<S> {
    fn default() -> Self {
        Self {
            tolerance: Default::default(),
            score_tolerance: Default::default(),
            iterations: Budget::ONE.saturating_add(99),
            factor_iterations: Budget::ONE.saturating_add(4095),
            rank_policy: RankPolicy::DimensionScaledEpsilon,
            max_bytes: usize::MAX,
        }
    }
}

/// Describes how a GLM fit ended.
///
/// The report is returned even when no iterate was accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlmReport {
    /// Reason the fit stopped.
    pub termination: StatsTermination,
    /// Number of accepted iterates.
    ///
    /// IRLS counts its iterations. An injected solver gives at most one, its
    /// final point after it is checked. When it is zero, the fit wrote nothing
    /// to the output.
    pub accepted_iterations: usize,
    /// Iterations and evaluations of this fit, including nested solves and
    /// failed callbacks. Work counted by an enclosing scope before this fit is
    /// not included.
    pub counts: scioxide_core::SolveCounts,
}

/// Selects the covariance estimator of the coefficients of a converged,
/// full-rank dense GLM.
///
/// With `A = W^(1/2) X` the design weighted by the final IRLS weights and
/// `B = AᵀA` the Fisher information at unit dispersion, the estimators are
/// `φ B⁻¹` and the sandwich `B⁻¹ (Σᵢ cᵢ sᵢ sᵢᵀ) B⁻¹`, where `sᵢ` is the score
/// contribution of row `i`.
///
/// # References
///
/// J. G. MacKinnon and H. White, "Some heteroskedasticity-consistent
/// covariance matrix estimators with improved finite sample properties",
/// *J. Econometrics* 29(3), 305–325, 1985.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GlmCovariance {
    /// `φ B⁻¹`. The dispersion is `φ = 1` for the binomial and Poisson
    /// families, and otherwise the Pearson estimate `X² / (n_eff − p)`, where
    /// `X²` is the Pearson chi-square statistic, `n_eff` the effective sample
    /// size and `p` the number of coefficients.
    #[default]
    Model,
    /// Sandwich estimator with `cᵢ = 1`.
    Hc0,
    /// Sandwich estimator with `cᵢ = (1 − hᵢ)⁻²`, where `hᵢ` is the leverage
    /// of row `i` in `A`.
    Hc3,
}

/// Holds the Wald test and confidence interval for one GLM coefficient.
///
/// The standard error `se` is the square root of the diagonal entry of the
/// selected covariance. The p-value is `2(1 − Φ(|β̂ⱼ| / se))` and the interval
/// is `β̂ⱼ ± z se`, where `Φ` is the standard normal distribution function and
/// `z` its quantile for the confidence level. Both are asymptotic. The fields
/// are read-only.
///
/// ```compile_fail
/// fn corrupt(mut c: scioxide_stats::GlmCoefficientInference<f64>) { c.p_value = 1.0; }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct GlmCoefficientInference<S: RealScalar> {
    pub(super) estimate: S,
    pub(super) standard_error: S,
    pub(super) p_value: S,
    pub(super) interval: scioxide_core::Interval<S>,
    pub(super) covariance: GlmCovariance,
}
impl<S: RealScalar> GlmCoefficientInference<S> {
    /// Returns the coefficient estimate `β̂ⱼ`.
    pub const fn estimate(&self) -> S {
        self.estimate
    }
    /// Returns the standard error.
    pub const fn standard_error(&self) -> S {
        self.standard_error
    }
    /// Returns the two-sided p-value of `H₀: βⱼ = 0`.
    pub const fn p_value(&self) -> S {
        self.p_value
    }
    /// Returns the two-sided confidence interval for `βⱼ`.
    pub const fn interval(&self) -> scioxide_core::Interval<S> {
        self.interval
    }
    /// Returns the covariance estimator.
    pub const fn covariance(&self) -> GlmCovariance {
        self.covariance
    }
}

pub use api::glm_fit;
