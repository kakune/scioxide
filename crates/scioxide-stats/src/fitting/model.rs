use super::{finite, invalid, number};

use scioxide_core::{RealScalar, SciOxideError};
use scioxide_random::{LogCdf, LogDensity, LogMass, LogSurvival, Mean, Quantile, Variance};

/// One observation of a random variable `X`, exact or censored.
///
/// Censoring intervals are half-open, `(lower, upper]`, for both continuous and
/// count distributions. An exact value contributes the density of a continuous
/// distribution or the mass of a count distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FitObservation<S: RealScalar> {
    /// `X = x`, a finite value.
    Exact(S),
    /// `X ≤ x`: only an upper bound is known.
    LeftCensored(S),
    /// `X > x`: only a lower bound is known.
    RightCensored(S),
    /// `lower < X ≤ upper`.
    IntervalCensored {
        /// Lower endpoint, excluded.
        lower: S,
        /// Upper endpoint, included.
        upper: S,
    },
}
impl<S: RealScalar> FitObservation<S> {
    pub(super) fn validate(self) -> Result<(), SciOxideError> {
        match self {
            Self::Exact(x) | Self::LeftCensored(x) | Self::RightCensored(x) => {
                crate::error_helpers::input_finite(x)?;
            }
            Self::IntervalCensored { lower, upper } => {
                crate::error_helpers::input_finite(lower)?;
                crate::error_helpers::input_finite(upper)?;
                if lower >= upper {
                    return Err(invalid(
                        scioxide_core::argument!("o"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "empty censor interval",
                    ));
                }
            }
        }
        Ok(())
    }
    pub(super) fn interval(self) -> Option<(S, S)> {
        match self {
            Self::Exact(_) => None,
            Self::LeftCensored(x) => Some((S::NEG_INFINITY, x)),
            Self::RightCensored(x) => Some((x, S::INFINITY)),
            Self::IntervalCensored { lower, upper } => Some((lower, upper)),
        }
    }
}
/// Truncation interval `(lower, upper]`: only values in it can be observed.
///
/// A missing endpoint means the interval is unbounded on that side. The
/// likelihood of each observation is divided by the model probability of the
/// interval.
#[derive(Clone, Copy, Debug)]
pub struct FitTruncation<S: RealScalar> {
    /// Lower endpoint, excluded.
    pub lower: Option<S>,
    /// Upper endpoint, included.
    pub upper: Option<S>,
}
impl<S: RealScalar> FitTruncation<S> {
    pub(super) fn interval(self) -> Result<(S, S), SciOxideError> {
        if let Some(x) = self.lower {
            crate::error_helpers::input_finite(x)?;
        }
        if let Some(x) = self.upper {
            crate::error_helpers::input_finite(x)?;
        }
        let pair = (
            self.lower.unwrap_or(S::NEG_INFINITY),
            self.upper.unwrap_or(S::INFINITY),
        );
        if pair.0 >= pair.1 {
            return Err(invalid(
                scioxide_core::argument!("truncation"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "empty truncation interval",
            ));
        }
        Ok(pair)
    }
}
/// A parametric family of distributions that [`PreparedFit`](crate::PreparedFit)
/// can fit.
///
/// Parameters, moments and quantiles are in the units of the data. Methods must
/// be pure functions of their arguments, because they may be called from
/// several threads.
pub trait FitModel<S: RealScalar>: Sync {
    /// Returns the number of parameters.
    fn parameter_count(&self) -> usize;
    /// Checks that `parameters` define a member of the family, including
    /// constraints that couple parameters.
    ///
    /// # Errors
    ///
    /// An error if `parameters` has the wrong length, is not finite or lies
    /// outside the parameter space.
    fn validate(&self, parameters: &[S]) -> Result<(), SciOxideError>;
    /// Returns the log density or log mass of an exact observation, or the log
    /// probability of a censoring interval.
    ///
    /// Returns `−∞` for an observation of zero probability.
    /// [`PreparedFit`](crate::PreparedFit) reports such an observation as a
    /// `DomainError`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::validate`] and of evaluating the distribution.
    fn log_observation(
        &self,
        parameters: &[S],
        observation: FitObservation<S>,
    ) -> Result<S, SciOxideError>;
    /// Returns the raw moment `E[Xᵏ]` of order `k = order`.
    ///
    /// The default returns `NotImplemented`. Moment matching needs it.
    ///
    /// # Errors
    ///
    /// An error if the order is not supported, the moment does not exist or it
    /// overflows.
    fn moment(&self, _parameters: &[S], _order: usize) -> Result<S, SciOxideError> {
        Err(SciOxideError::not_implemented("model raw moment"))
    }
    /// Returns the quantile `Q(p) = inf{x : F(x) ≥ p}` for `0 < p < 1`.
    ///
    /// The default returns `NotImplemented`. Quantile matching and probability
    /// plots need it.
    ///
    /// # Errors
    ///
    /// An error if quantiles are not supported, `probability` is not in `(0, 1)`
    /// or the evaluation fails.
    fn quantile(&self, _parameters: &[S], _probability: S) -> Result<S, SciOxideError> {
        Err(SciOxideError::not_implemented("model quantile"))
    }
}
/// A continuous family defined by a function that builds a distribution from
/// the parameter vector.
///
/// The function returns a distribution from `scioxide_random` (or any type that
/// implements its density, distribution function, survival, mean, variance and
/// quantile traits), or an error for invalid parameters. It is called for every
/// evaluation, so it must be cheap and must not draw random numbers. Raw
/// moments of orders 1 and 2 are available, computed from the mean and
/// variance.
pub struct LawFitModel<F> {
    dimension: usize,
    constructor: F,
}
impl<F> LawFitModel<F> {
    /// Creates a family with `dimension` parameters, built by `constructor`.
    ///
    /// # Errors
    ///
    /// `InvalidInput` if `dimension` is zero.
    pub fn new(dimension: usize, constructor: F) -> Result<Self, SciOxideError> {
        if dimension == 0 {
            return Err(invalid(
                scioxide_core::argument!("dimension"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "zero model dimension",
            ));
        }
        Ok(Self {
            dimension,
            constructor,
        })
    }
}
fn continuous<S: RealScalar, L: LogDensity<S> + LogCdf<S> + LogSurvival<S>>(
    law: &L,
    o: FitObservation<S>,
) -> Result<S, SciOxideError> {
    match o {
        FitObservation::Exact(x) => law.log_pdf(x),
        _ => {
            let (a, b) = o.interval().ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("o"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "expected interval",
                )
            })?;
            scioxide_random::log_interval_probability(law, a, b)
        }
    }
}
impl<S, F, L> FitModel<S> for LawFitModel<F>
where
    S: RealScalar,
    F: Fn(&[S]) -> Result<L, SciOxideError> + Sync,
    L: LogDensity<S> + LogCdf<S> + LogSurvival<S> + Mean<S> + Variance<S> + Quantile<S>,
{
    fn parameter_count(&self) -> usize {
        self.dimension
    }
    fn validate(&self, p: &[S]) -> Result<(), SciOxideError> {
        check_parameters(p, self.dimension)?;
        (self.constructor)(p)?;
        Ok(())
    }
    fn log_observation(&self, p: &[S], o: FitObservation<S>) -> Result<S, SciOxideError> {
        self.validate(p)?;
        o.validate()?;
        continuous(&(self.constructor)(p)?, o)
    }
    fn moment(&self, p: &[S], order: usize) -> Result<S, SciOxideError> {
        self.validate(p)?;
        let law = (self.constructor)(p)?;
        raw_moment(&law, order)
    }
    fn quantile(&self, p: &[S], q: S) -> Result<S, SciOxideError> {
        self.validate(p)?;
        (self.constructor)(p)?.quantile(q)
    }
}
fn raw_moment<S: RealScalar, L: Mean<S> + Variance<S>>(
    law: &L,
    order: usize,
) -> Result<S, SciOxideError> {
    match order {
        1 => law.mean(),
        2 => {
            let m = law.mean()?;
            finite(law.variance()? + m * m)
        }
        _ => Err(SciOxideError::not_implemented(
            "built-in raw moments support orders one and two",
        )),
    }
}
fn check_parameters<S: RealScalar>(p: &[S], n: usize) -> Result<(), SciOxideError> {
    if p.len() != n {
        return Err(invalid(
            scioxide_core::argument!("p"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "parameter dimension mismatch",
        ));
    }
    p.iter().try_for_each(|&x| finite(x).map(|_| ()))
}
/// Built-in distribution families.
///
/// Each family has exactly the parameters listed. No location or scale
/// parameter is added. Raw moments of orders 1 and 2 are available.
#[derive(Clone, Copy, Debug)]
pub enum DistributionFitModel {
    /// Normal distribution with parameters `[μ, σ]`, `σ > 0`.
    Normal,
    /// Exponential distribution with parameter `[λ]`, rate `λ > 0`, on `[0, ∞)`.
    Exponential,
    /// Uniform distribution with parameters `[a, b]`, `a < b`.
    ///
    /// An exact observation at `b` has density `1/(b − a)`, so the maximum
    /// likelihood estimate `[min x, max x]` exists.
    Uniform,
    /// Poisson distribution with parameter `[λ]`, rate `λ ≥ 0`.
    ///
    /// An exact observation that is not a nonnegative integer has probability
    /// zero. Quantiles are not available.
    Poisson,
}
impl<S: RealScalar> FitModel<S> for DistributionFitModel {
    fn parameter_count(&self) -> usize {
        match self {
            Self::Normal | Self::Uniform => 2,
            _ => 1,
        }
    }
    fn validate(&self, p: &[S]) -> Result<(), SciOxideError> {
        check_parameters(p, <Self as FitModel<S>>::parameter_count(self))?;
        match self {
            Self::Normal => {
                let _ = scioxide_random::Normal::new(p[0], p[1])?;
            }
            Self::Exponential => {
                let _ = scioxide_random::Exponential::new(p[0])?;
            }
            Self::Uniform => {
                let _ = scioxide_random::Uniform::new(p[0], p[1])?;
            }
            Self::Poisson => {
                let _ = scioxide_random::Poisson::new(p[0])?;
            }
        }
        Ok(())
    }
    fn log_observation(&self, p: &[S], o: FitObservation<S>) -> Result<S, SciOxideError> {
        self.validate(p)?;
        o.validate()?;
        match self {
            Self::Normal => continuous(&scioxide_random::Normal::new(p[0], p[1])?, o),
            Self::Exponential => continuous(&scioxide_random::Exponential::new(p[0])?, o),
            Self::Uniform => {
                let law = scioxide_random::Uniform::new(p[0], p[1])?;
                // Continuous density values at a point are arbitrary. Fitting uses
                // the closed-support limiting density so the boundary MLE exists.
                if matches!(o, FitObservation::Exact(x) if x == p[1]) {
                    law.log_pdf(p[0])
                } else {
                    continuous(&law, o)
                }
            }
            Self::Poisson => {
                let law = scioxide_random::Poisson::new(p[0])?;
                match o {
                    FitObservation::Exact(x) => {
                        if x < S::ZERO || x.floor() != x {
                            return Ok(S::NEG_INFINITY);
                        }
                        law.log_pmf(count(x)?)
                    }
                    _ => {
                        let (a, b) = o.interval().ok_or_else(|| {
                            invalid(
                                scioxide_core::argument!("o"),
                                scioxide_core::InvalidInputReason::Inconsistent,
                                "expected count interval",
                            )
                        })?;
                        scioxide_random::log_interval_probability(&RealCountLaw(law), a, b)
                    }
                }
            }
        }
    }
    fn moment(&self, p: &[S], order: usize) -> Result<S, SciOxideError> {
        self.validate(p)?;
        match self {
            Self::Normal => raw_moment(&scioxide_random::Normal::new(p[0], p[1])?, order),
            Self::Exponential => raw_moment(&scioxide_random::Exponential::new(p[0])?, order),
            Self::Uniform => raw_moment(&scioxide_random::Uniform::new(p[0], p[1])?, order),
            Self::Poisson => raw_moment(&scioxide_random::Poisson::new(p[0])?, order),
        }
    }
    fn quantile(&self, p: &[S], q: S) -> Result<S, SciOxideError> {
        self.validate(p)?;
        if !q.is_finite() || q <= S::ZERO || q >= S::ONE {
            return Err(invalid(
                scioxide_core::argument!("q"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "fitting quantile probability must be interior",
            ));
        }
        match self {
            Self::Normal => scioxide_random::Normal::new(p[0], p[1])?.quantile(q),
            Self::Exponential => scioxide_random::Exponential::new(p[0])?.quantile(q),
            Self::Uniform => scioxide_random::Uniform::new(p[0], p[1])?.quantile(q),
            Self::Poisson => Err(SciOxideError::not_implemented(
                "Poisson fitting quantiles require an injected count-quantile model",
            )),
        }
    }
}
fn count<S: RealScalar>(x: S) -> Result<usize, SciOxideError> {
    let v = x.try_to_f64_exact()?;
    if v < 0.0 || v >= usize::MAX as f64 || v >= 9007199254740992.0 {
        return Err(invalid(
            scioxide_core::argument!("o"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "count outside exact supported range",
        ));
    }
    Ok(v as usize)
}
// Poisson law over real arguments with checked count conversion.
struct RealCountLaw<S: RealScalar>(scioxide_random::Poisson<S>);
impl<S: RealScalar> LogCdf<S> for RealCountLaw<S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ZERO {
            Ok(S::NEG_INFINITY)
        } else {
            self.0.log_cdf(count(x.floor())?)
        }
    }
}
impl<S: RealScalar> LogSurvival<S> for RealCountLaw<S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ZERO {
            Ok(S::ZERO)
        } else {
            self.0.log_sf(count(x.floor())?)
        }
    }
}

/// Analytic gradient and Hessian of the negative log likelihood of a built-in
/// family for exact, uncensored samples.
///
/// For `n` samples `xᵢ`, the negative log likelihood `ℓ` has these derivatives:
///
/// ```text
/// Normal [μ, σ]:   ∂ℓ/∂μ = Σ (μ − xᵢ)/σ²,   ∂ℓ/∂σ = Σ (1 − zᵢ²)/σ,
///                  ∂²ℓ/∂μ² = n/σ²,  ∂²ℓ/∂μ∂σ = −2 Σ zᵢ/σ²,  ∂²ℓ/∂σ² = Σ (3zᵢ² − 1)/σ²,
///                  with zᵢ = (μ − xᵢ)/σ,
/// Exponential [λ]: ∂ℓ/∂λ = Σ (xᵢ − 1/λ),   ∂²ℓ/∂λ² = n/λ²,
/// Poisson [λ]:     ∂ℓ/∂λ = Σ (1 − xᵢ/λ),   ∂²ℓ/∂λ² = Σ xᵢ/λ².
/// ```
///
/// The uniform family has a parameter-dependent support, so the regular
/// derivatives do not apply and are not provided. Pass this type as the
/// derivatives of a [`FitRequest`](crate::FitRequest), or use its Hessian in
/// [`fit_information`](crate::fit_information).
#[derive(Clone, Copy, Debug)]
pub struct ExactFitDerivatives<'a, S: RealScalar> {
    /// The family.
    pub model: DistributionFitModel,
    /// The exact sample values, checked on each call.
    pub samples: &'a [S],
}
impl<S: RealScalar> ExactFitDerivatives<'_, S> {
    /// Writes the gradient and the row-major Hessian of the negative log
    /// likelihood at `p`.
    ///
    /// # Errors
    ///
    /// - The errors of validating `p` for the family.
    /// - `InvalidInput` if `samples` is empty, or if `gradient` or `hessian` does
    ///   not have length `d` or `d²` for `d` parameters.
    /// - `InvalidInput` for a Poisson rate of zero, where the derivatives do not exist.
    /// - `NotImplemented` for the uniform family.
    /// - `NonFinite` if a sample is NaN or infinite, or a derivative overflows.
    /// - An error if a sample has zero probability, such as a non-integer
    ///   Poisson count.
    pub fn evaluate(
        &self,
        p: &[S],
        gradient: &mut [S],
        hessian: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.model.validate(p)?;
        let d = p.len();
        if gradient.len() != d || hessian.len() != d * d || self.samples.is_empty() {
            return Err(invalid(
                scioxide_core::argument!("gradient"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "derivative shape or empty sample",
            ));
        }
        let mut g = [S::ZERO; 2];
        let mut h = [S::ZERO; 4];
        let two = number::<S>(2.0)?;
        let three = number::<S>(3.0)?;
        for &x in self.samples {
            crate::error_helpers::input_finite(x)?;
            finite(self.model.log_observation(p, FitObservation::Exact(x))?)?;
            match self.model {
                DistributionFitModel::Normal => {
                    let z = (p[0] - x) / p[1];
                    let inv = S::ONE / p[1];
                    g[0] += z * inv;
                    g[1] += (S::ONE - z * z) * inv;
                    h[0] += inv * inv;
                    h[1] += -two * z * inv * inv;
                    h[3] += (three * z * z - S::ONE) * inv * inv;
                }
                DistributionFitModel::Exponential => {
                    g[0] = g[0] + x - S::ONE / p[0];
                    h[0] += S::ONE / p[0] / p[0];
                }
                DistributionFitModel::Poisson => {
                    if p[0] == S::ZERO {
                        return Err(invalid(
                            scioxide_core::argument!("p"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "Poisson zero-rate derivative is nonregular",
                        ));
                    }
                    g[0] = g[0] + S::ONE - x / p[0];
                    h[0] += x / p[0] / p[0];
                }
                DistributionFitModel::Uniform => {
                    return Err(SciOxideError::not_implemented(
                        "Uniform regular information",
                    ));
                }
            }
        }
        h[2] = h[1];
        for &x in g.iter().chain(h.iter()) {
            crate::error_helpers::input_finite(x)?;
        }
        gradient.copy_from_slice(&g[..d]);
        if d == 1 {
            hessian[0] = h[0];
        } else {
            hessian.copy_from_slice(&h);
        }
        Ok(())
    }
}
