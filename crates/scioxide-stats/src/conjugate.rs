//! Conjugate Bayesian updates for binomial, Poisson and normal models.
use crate::ConfidenceProbability;
use crate::arithmetic::affine_scaled;
use crate::error_helpers::invalid;
use crate::weighted::WeightedMoments;
use crate::{
    ConjugateWeights, MomentOrder, MomentsSnapshot, OnlineMoments, OnlineWeightedMoments,
    WeightSemantics, WeightedMomentsSnapshot, Weights,
};
use core::marker::PhantomData;
use scioxide_core::Scaled;
use scioxide_core::{Argument, Interval, NonFiniteOrigin, RealScalar, SciOxideError};
use scioxide_random::{
    Affine, Beta, BetaBinomial, Gamma, GeneralizedGamma, InversionOptions, LogCdf, LogSurvival,
    Mean, NegativeBinomial, NormalInverseGamma, StudentT, Support,
    invert_log_probability_with_options,
};

fn parameter<S: RealScalar>(
    x: S,
    positive: bool,
    argument: Argument,
) -> Result<Scaled<f64>, SciOxideError> {
    if !x.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "parameter must be finite",
        ));
    }
    if x < S::ZERO || (positive && x == S::ZERO) {
        return Err(invalid(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "parameter must be positive (nonnegative for explicit improper boundaries)",
        ));
    }
    Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(x))
}
fn proper(x: Scaled<f64>) -> bool {
    !x.is_zero() && !x.is_sign_negative()
}

/// Beta prior or posterior `Beta(α, β)` for the success probability of binomial observations.
///
/// `α` and `β` act as prior counts of successes and failures. Observing `k` successes in `n`
/// trials gives the posterior `Beta(α + k, β + n − k)`. Each update costs `O(1)`. The
/// parameters are stored with an extended exponent range and rounded to `S` when read.
///
/// # References
///
/// A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari and D. B. Rubin, *Bayesian
/// Data Analysis*, 3rd ed., CRC Press, 2013, ch. 2–3.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct BetaBinomialPrior<S: RealScalar> {
    alpha: Scaled<f64>,
    beta: Scaled<f64>,
    scalar: PhantomData<S>,
}
impl<S: RealScalar> BetaBinomialPrior<S> {
    /// Creates the proper prior `Beta(α, β)`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite.
    /// - `InvalidInput` if `alpha` or `beta` is not positive.
    pub fn new(alpha: S, beta: S) -> Result<Self, SciOxideError> {
        Ok(Self {
            alpha: parameter(alpha, true, scioxide_core::argument!("alpha"))?,
            beta: parameter(beta, true, scioxide_core::argument!("beta"))?,
            scalar: PhantomData,
        })
    }
    /// Creates an improper prior with density proportional to `p^(α−1) (1 − p)^(β−1)`, where
    /// `α = 0` or `β = 0`.
    ///
    /// `improper(0, 0)` is Haldane's prior. The posterior becomes proper once both parameters
    /// are positive.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite.
    /// - `InvalidInput` if `alpha` or `beta` is negative, or neither is zero.
    pub fn improper(alpha: S, beta: S) -> Result<Self, SciOxideError> {
        let result = Self {
            alpha: parameter(alpha, false, scioxide_core::argument!("alpha"))?,
            beta: parameter(beta, false, scioxide_core::argument!("beta"))?,
            scalar: PhantomData,
        };
        if result.is_proper() {
            return Err(invalid(
                scioxide_core::argument!("alpha"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "improper beta boundary requires a zero parameter",
            ));
        }
        Ok(result)
    }
    /// Returns `true` if both parameters are positive, so the distribution can be normalized.
    pub fn is_proper(self) -> bool {
        proper(self.alpha) && proper(self.beta)
    }
    /// Returns `(α, β)`, each rounded to `S`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a parameter overflows `S`.
    pub fn parameters(self) -> Result<(S, S), SciOxideError> {
        Ok((self.alpha.round_to()?, self.beta.round_to()?))
    }
    /// Returns the posterior after `successes` in `trials` binomial trials.
    ///
    /// Zero trials leave the distribution unchanged.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `successes > trials`.
    pub fn update(self, successes: usize, trials: usize) -> Result<Self, SciOxideError> {
        self.update_frequency(successes, trials, 1)
    }
    /// Returns the posterior after `frequency` independent repetitions of the same binomial
    /// observation.
    ///
    /// The result is `Beta(α + f k, β + f (n − k))` for frequency `f`. A frequency of zero
    /// leaves the distribution unchanged.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `successes > trials`.
    /// - `NumericalInstability` if a parameter overflows the extended exponent range.
    pub fn update_frequency(
        self,
        successes: usize,
        trials: usize,
        frequency: usize,
    ) -> Result<Self, SciOxideError> {
        if successes > trials {
            return Err(invalid(
                scioxide_core::argument!("successes"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "successes exceed binomial trials",
            ));
        }
        let f = Scaled::<f64>::from_value((frequency) as f64)?;
        Ok(Self {
            alpha: self
                .alpha
                .add(Scaled::<f64>::from_value((successes) as f64)?.mul(f)?)?,
            beta: self
                .beta
                .add(Scaled::<f64>::from_value((trials - successes) as f64)?.mul(f)?)?,
            ..self
        })
    }
    /// Returns the posterior after the `(successes, trials)` observations in order.
    ///
    /// # Errors
    ///
    /// - The first error of [`Self::update`].
    pub fn update_batch(self, observations: &[(usize, usize)]) -> Result<Self, SciOxideError> {
        observations
            .iter()
            .try_fold(self, |state, &(s, n)| state.update(s, n))
    }
    /// Returns the distribution `Beta(α, β)` of the success probability.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the distribution is improper.
    /// - `NumericalInstability` if a parameter overflows `S`.
    pub fn posterior_law(self) -> Result<Beta<S>, SciOxideError> {
        if !self.is_proper() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "beta posterior is improper",
            ));
        }
        let (a, b) = self.parameters()?;
        Beta::new(a, b)
    }
    /// Returns the posterior predictive distribution `BetaBinomial(n, α, β)` of the number of
    /// successes in `n = trials` future trials.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the distribution is improper.
    /// - `NumericalInstability` if a parameter overflows `S`.
    pub fn predictive(self, trials: usize) -> Result<BetaBinomial<S>, SciOxideError> {
        if !self.is_proper() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "beta posterior is improper",
            ));
        }
        let (a, b) = self.parameters()?;
        BetaBinomial::new(trials, a, b)
    }
}

/// Gamma prior or posterior for the rate `λ` of Poisson counts.
///
/// The density is
///
/// ```text
/// p(λ) = β^a / Γ(a) · λ^(a−1) e^(−βλ),   λ > 0,
/// ```
///
/// with shape `a` and rate `β`. A count `k` observed over exposure `t` has distribution
/// `Poisson(λt)` and gives the posterior shape `a + k` and rate `β + t`. Exposure is in any
/// fixed unit, and `λ` is the rate per that unit. Each update costs `O(1)`.
///
/// # References
///
/// A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari and D. B. Rubin, *Bayesian
/// Data Analysis*, 3rd ed., CRC Press, 2013, ch. 2–3.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct GammaPoissonPrior<S: RealScalar> {
    shape: Scaled<f64>,
    rate: Scaled<f64>,
    scalar: PhantomData<S>,
}
impl<S: RealScalar> GammaPoissonPrior<S> {
    /// Creates the proper prior with shape `a` and rate `β`, the reciprocal of the scale.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `shape` or `rate` is NaN or infinite.
    /// - `InvalidInput` if `shape` or `rate` is not positive.
    pub fn new(shape: S, rate: S) -> Result<Self, SciOxideError> {
        Ok(Self {
            shape: parameter(shape, true, scioxide_core::argument!("shape"))?,
            rate: parameter(rate, true, scioxide_core::argument!("rate"))?,
            scalar: PhantomData,
        })
    }
    /// Creates an improper prior with density proportional to `λ^(a−1) e^(−βλ)`, where `a = 0`
    /// or `β = 0`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `shape` or `rate` is NaN or infinite.
    /// - `InvalidInput` if `shape` or `rate` is negative, or neither is zero.
    pub fn improper(shape: S, rate: S) -> Result<Self, SciOxideError> {
        let result = Self {
            shape: parameter(shape, false, scioxide_core::argument!("shape"))?,
            rate: parameter(rate, false, scioxide_core::argument!("rate"))?,
            scalar: PhantomData,
        };
        if result.is_proper() {
            return Err(invalid(
                scioxide_core::argument!("shape"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "improper gamma boundary requires a zero parameter",
            ));
        }
        Ok(result)
    }
    /// Returns `true` if both parameters are positive, so the distribution can be normalized.
    pub fn is_proper(self) -> bool {
        proper(self.shape) && proper(self.rate)
    }
    /// Returns `(a, β)`, each rounded to `S`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a parameter overflows `S`.
    pub fn parameters(self) -> Result<(S, S), SciOxideError> {
        Ok((self.shape.round_to()?, self.rate.round_to()?))
    }
    /// Returns the posterior after observing `count` events over `exposure`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `exposure` is NaN or infinite.
    /// - `InvalidInput` if `exposure` is negative, or zero with a positive `count`.
    pub fn update(self, count: usize, exposure: S) -> Result<Self, SciOxideError> {
        self.update_frequency(count, exposure, 1)
    }
    /// Returns the posterior after `frequency` independent repetitions of the same observation.
    ///
    /// The result has shape `a + f k` and rate `β + f t` for frequency `f`. A frequency of
    /// zero leaves the distribution unchanged.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::update`].
    /// - `NumericalInstability` if a parameter overflows the extended exponent range.
    pub fn update_frequency(
        self,
        count: usize,
        exposure: S,
        frequency: usize,
    ) -> Result<Self, SciOxideError> {
        let exposure = parameter(exposure, false, scioxide_core::argument!("exposure"))?;
        if count > 0 && exposure.is_zero() {
            return Err(invalid(
                scioxide_core::argument!("exposure"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "positive Poisson count requires positive exposure",
            ));
        }
        let f = Scaled::<f64>::from_value((frequency) as f64)?;
        Ok(Self {
            shape: self
                .shape
                .add(Scaled::<f64>::from_value((count) as f64)?.mul(f)?)?,
            rate: self.rate.add(exposure.mul(f)?)?,
            ..self
        })
    }
    /// Returns the posterior after the `(count, exposure)` observations in order.
    ///
    /// # Errors
    ///
    /// - The first error of [`Self::update`].
    pub fn update_batch(self, observations: &[(usize, S)]) -> Result<Self, SciOxideError> {
        observations
            .iter()
            .try_fold(self, |state, &(k, t)| state.update(k, t))
    }
    /// Returns the Gamma distribution of `λ`, with shape `a` and scale `1/β`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the distribution is improper.
    /// - `NumericalInstability` if a parameter overflows `S`.
    pub fn posterior_law(self) -> Result<Gamma<S>, SciOxideError> {
        if !self.is_proper() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "gamma posterior is improper",
            ));
        }
        Gamma::new(
            self.shape.round_to()?,
            Scaled::<f64>::ONE.div(self.rate)?.round_to()?,
        )
    }
    /// Returns the posterior predictive distribution of a count over a future `exposure` `t`.
    ///
    /// It is the negative binomial distribution with `a` successes and success probability
    /// `β/(β + t)`. A zero exposure gives a point mass at zero.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `exposure` is NaN or infinite.
    /// - `InvalidInput` if `exposure` is negative, the distribution is improper, or a positive
    ///   `exposure` is so small that `β/(β + t)` rounds to 1.
    /// - `NumericalInstability` if a parameter overflows `S`.
    pub fn predictive(self, exposure: S) -> Result<NegativeBinomial<S>, SciOxideError> {
        let t = parameter(exposure, false, scioxide_core::argument!("exposure"))?;
        if !self.is_proper() {
            return Err(invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "gamma posterior is improper",
            ));
        }
        let p = self.rate.div(self.rate.add(t)?)?.round_to()?;
        if !t.is_zero() && p == S::ONE {
            return Err(invalid(
                scioxide_core::argument!("exposure"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "positive predictive exposure is below scalar resolution",
            ));
        }
        NegativeBinomial::new(self.shape.round_to()?, p)
    }
}

/// Normal-inverse-gamma prior or posterior for the mean `μ` and variance `σ²` of normal
/// observations.
///
/// ```text
/// σ² ~ InvGamma(a, b),   μ | σ² ~ N(m, σ²/κ),
/// ```
///
/// with location `m`, precision `κ`, shape `a` and scale `b`. Observing `n` values with mean
/// `x̄` and sum of squared deviations `Q = Σ (xᵢ − x̄)²` gives
///
/// ```text
/// κₙ = κ + n,   mₙ = (κ m + n x̄)/κₙ,   aₙ = a + n/2,
/// bₙ = b + Q/2 + κ n (x̄ − m)² / (2κₙ).
/// ```
///
/// The parameters are stored with an extended exponent range and rounded to `S` when read.
///
/// # References
///
/// A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari and D. B. Rubin, *Bayesian
/// Data Analysis*, 3rd ed., CRC Press, 2013, ch. 2–3.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct NormalInverseGammaPrior<S: RealScalar> {
    location: Scaled<f64>,
    precision: Scaled<f64>,
    shape: Scaled<f64>,
    scale: Scaled<f64>,
    scalar: PhantomData<S>,
}
impl<S: RealScalar> NormalInverseGammaPrior<S> {
    /// Creates the proper prior with location `m`, precision `κ`, shape `a` and scale `b`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a parameter is NaN or infinite.
    /// - `InvalidInput` if `precision`, `shape` or `scale` is not positive.
    pub fn new(location: S, precision: S, shape: S, scale: S) -> Result<Self, SciOxideError> {
        if !location.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "location must be finite",
            ));
        }
        Ok(Self {
            location: Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(location))?,
            precision: parameter(precision, true, scioxide_core::argument!("precision"))?,
            shape: parameter(shape, true, scioxide_core::argument!("shape"))?,
            scale: parameter(scale, true, scioxide_core::argument!("scale"))?,
            scalar: PhantomData,
        })
    }
    /// Returns the improper reference prior `p(μ, σ²) ∝ 1/σ²`.
    ///
    /// It has `m = 0`, `κ = 0`, `a = −1/2` and `b = 0`. After `n` observations the posterior
    /// has `κ = n`, `m = x̄`, `a = (n − 1)/2` and `b = Q/2`, which is proper for `n ≥ 2` and
    /// `Q > 0`.
    pub fn reference() -> Self {
        Self {
            location: Scaled::<f64>::ZERO,
            precision: Scaled::<f64>::ZERO,
            shape: -Scaled::<f64>::HALF,
            scale: Scaled::<f64>::ZERO,
            scalar: PhantomData,
        }
    }
    /// Returns `true` if `κ`, `a` and `b` are positive, so the distribution can be normalized.
    pub fn is_proper(self) -> bool {
        proper(self.precision) && proper(self.shape) && proper(self.scale)
    }
    /// Returns `(m, κ, a, b)`, each rounded to `S`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a parameter overflows `S`.
    pub fn parameters(self) -> Result<(S, S, S, S), SciOxideError> {
        Ok((
            self.location.round_to()?,
            self.precision.round_to()?,
            self.shape.round_to()?,
            self.scale.round_to()?,
        ))
    }
    fn updated(
        self,
        count: usize,
        mass: Scaled<f64>,
        mean: Scaled<f64>,
        variance: Scaled<f64>,
    ) -> Result<Self, SciOxideError> {
        if count == 0 {
            return Ok(self);
        }
        let precision = self.precision.add(mass)?;
        let (location, correction) = if self.precision.is_zero() {
            (mean, Scaled::<f64>::ZERO)
        } else {
            let p = self.precision.div(precision)?;
            let q = mass.div(precision)?;
            let delta = mean.sub(self.location)?;
            (
                affine_scaled(self.location, mean, p, q)?,
                delta.mul(delta)?.mul(self.precision)?.mul(q)?,
            )
        };
        let half = Scaled::<f64>::from_value(0.5)?;
        Ok(Self {
            location,
            precision,
            shape: self
                .shape
                .add(Scaled::<f64>::from_value((count) as f64)?.mul(half)?)?,
            scale: self
                .scale
                .add(variance.mul(mass)?.add(correction)?.mul(half)?)?,
            ..self
        })
    }
    /// Returns the posterior after the observations summarized by `moments`.
    ///
    /// The count, mean and sum of squared deviations of `moments` are sufficient statistics,
    /// so the update costs `O(1)`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a parameter overflows the extended exponent range.
    pub fn update_moments(self, moments: &OnlineMoments<S>) -> Result<Self, SciOxideError> {
        self.updated(
            moments.count(),
            Scaled::<f64>::from_value((moments.count()) as f64)?,
            moments.state.mean()?.total()?,
            moments.state.m2()?,
        )
    }
    /// Returns the posterior after the observations summarized by a moments snapshot.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a parameter overflows the extended exponent range.
    pub fn update_snapshot(self, moments: MomentsSnapshot<S>) -> Result<Self, SciOxideError> {
        self.updated(
            moments.count(),
            Scaled::<f64>::from_value((moments.count()) as f64)?,
            moments.state.mean()?.total()?,
            moments.state.m2()?,
        )
    }
    /// Returns the posterior after weighted observations.
    ///
    /// A frequency weight `wᵢ` counts observation `i` that many times. An analytic weight
    /// means that `xᵢ` has variance `σ²/wᵢ`. Then `x̄` and `Q` are weighted, `κ` grows by
    /// `Σ wᵢ` and `a` by half the number of observations, so rescaling analytic weights
    /// changes the model. Reliability weights do not define a likelihood and are not accepted;
    /// see [`ConjugateWeights`].
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if a parameter overflows the extended exponent range.
    pub fn update_weighted<M: ConjugateWeights>(
        self,
        moments: &OnlineWeightedMoments<S, M>,
    ) -> Result<Self, SciOxideError> {
        self.update_core(&moments.core)
    }
    /// Returns the posterior after weighted observations summarized by a snapshot, as in
    /// [`Self::update_weighted`].
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::update_weighted`].
    pub fn update_weighted_snapshot<M: ConjugateWeights>(
        self,
        moments: WeightedMomentsSnapshot<S, M>,
    ) -> Result<Self, SciOxideError> {
        self.update_core(&moments.accumulator)
    }
    fn update_core(self, moments: &WeightedMoments<S>) -> Result<Self, SciOxideError> {
        self.updated(
            moments.estimator_count(),
            moments.total_mass()?,
            moments.state.mean()?.total()?,
            moments.state.m2()?,
        )
    }
    /// Returns the posterior after one observation.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `value` is NaN or infinite.
    /// - `NumericalInstability` if a parameter overflows the extended exponent range.
    pub fn update(self, value: S) -> Result<Self, SciOxideError> {
        let mut moments = OnlineMoments::new(MomentOrder::Second);
        moments.observe(value)?;
        self.update_moments(&moments)
    }
    /// Returns the posterior after the observations in `values`.
    ///
    /// The cost is `O(n)` time and `O(1)` memory.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a value is NaN or infinite.
    /// - `NumericalInstability` if a parameter overflows the extended exponent range.
    pub fn update_batch(self, values: &[S]) -> Result<Self, SciOxideError> {
        self.update_moments(&OnlineMoments::from_slice(values, MomentOrder::Second)?)
    }
    /// Returns the joint normal-inverse-gamma distribution of `(μ, σ²)`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the distribution is improper.
    /// - `NumericalInstability` if a parameter overflows `S`.
    pub fn posterior_law(self) -> Result<NormalInverseGamma<S>, SciOxideError> {
        self.require_proper()?;
        let (m, k, a, b) = self.parameters()?;
        NormalInverseGamma::with_options(
            m,
            k,
            a,
            b,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
    }
    fn require_proper(self) -> Result<(), SciOxideError> {
        if self.is_proper() {
            Ok(())
        } else {
            Err(invalid(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "normal posterior is improper; reference inference requires at least two observations and positive scatter",
            ))
        }
    }
    /// Returns the marginal distributions of `μ`, `σ²` and `σ`.
    ///
    /// The mean has a Student t distribution with `2a` degrees of freedom, location `m` and
    /// scale `√(b/(aκ))`. The variance has the inverse-gamma distribution `InvGamma(a, b)`, and
    /// the standard deviation is its square root.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the distribution is improper.
    /// - `NumericalInstability` if a parameter overflows `S`.
    pub fn marginals(self) -> Result<NormalPosteriorMarginals<S>, SciOxideError> {
        self.require_proper()?;
        let a: S = self.shape.round_to()?;
        let location = self.location.round_to()?;
        let mean_scale = self
            .scale
            .div(self.shape.mul(self.precision)?)?
            .sqrt()?
            .round_to()?;
        Ok(NormalPosteriorMarginals {
            mean: Affine::new(
                StudentT::new(
                    self.shape
                        .mul(Scaled::<f64>::from_value(2.0)?)?
                        .round_to()?,
                )?,
                location,
                mean_scale,
            )?,
            variance: Affine::new(
                GeneralizedGamma::new(a, -S::ONE)?,
                S::ZERO,
                self.scale.round_to()?,
            )?,
            standard_deviation: Affine::new(
                GeneralizedGamma::new(a, -S::TWO)?,
                S::ZERO,
                self.scale.sqrt()?.round_to()?,
            )?,
            parameters: self,
        })
    }
    /// Returns the posterior predictive distribution of a new observation with variance
    /// `σ²/w`, where `w` is `precision_multiplier`.
    ///
    /// It is a Student t distribution with `2a` degrees of freedom, location `m` and scale
    /// `√(b/a · (1/w + 1/κ))`. Use `w = 1` for an observation like the data.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the distribution is improper or `precision_multiplier` is not
    ///   positive.
    /// - `NonFinite` if `precision_multiplier` is NaN or infinite.
    /// - `NumericalInstability` if a parameter overflows `S`.
    pub fn predictive(
        self,
        precision_multiplier: S,
    ) -> Result<Affine<StudentT<S>, S>, SciOxideError> {
        self.require_proper()?;
        let multiplier = parameter(
            precision_multiplier,
            true,
            scioxide_core::argument!("precision_multiplier"),
        )?;
        let factor = Scaled::<f64>::ONE
            .div(multiplier)?
            .add(Scaled::<f64>::ONE.div(self.precision)?)?;
        let scale = self
            .scale
            .div(self.shape)?
            .mul(factor)?
            .sqrt()?
            .round_to()?;
        Affine::new(
            StudentT::new(
                self.shape
                    .mul(Scaled::<f64>::from_value(2.0)?)?
                    .round_to()?,
            )?,
            self.location.round_to()?,
            scale,
        )
    }
}

/// Equal-tailed credible interval of a posterior distribution, from [`credible_interval`].
///
/// It contains posterior probability `c` and leaves `(1 − c)/2` in each tail. It is not a
/// confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct CredibleInterval<S: RealScalar> {
    bounds: Interval<S>,
    probability: ConfidenceProbability,
}
impl<S: RealScalar> CredibleInterval<S> {
    /// Returns the interval `[q_(1−c)/2, q_(1+c)/2]` between the two posterior quantiles.
    pub const fn bounds(self) -> Interval<S> {
        self.bounds
    }
    /// Returns the posterior probability `c` of the interval.
    pub const fn probability(self) -> ConfidenceProbability {
        self.probability
    }
}
/// Computes the equal-tailed credible interval with posterior probability `c` of a
/// distribution.
///
/// The lower end solves `ln F(x) = ln((1 − c)/2)` and the upper end solves
/// `ln S(x) = ln((1 − c)/2)`, where `F` is the CDF and `S` the survival function. Inverting
/// the survival function directly, rather than the CDF at `1 − (1 − c)/2`, avoids cancellation
/// in the upper tail. Each end has its own work budget from `options`, so the total cost is at
/// most twice that budget.
///
/// # Errors
///
/// - `ConvergenceFailure` if the inversion for an end does not converge.
/// - `NonFinite` if an end is infinite.
/// - Errors from evaluating the tails of `law`.
pub fn credible_interval<S: RealScalar, D: LogCdf<S> + LogSurvival<S> + Support<S> + ?Sized>(
    law: &D,
    probability: ConfidenceProbability,
    options: &InversionOptions<S>,
) -> Result<CredibleInterval<S>, SciOxideError> {
    let log_tail = S::try_from_f64_lossy(((1.0 - probability.get()) * 0.5).ln())?;
    let lower = invert_log_probability_with_options(law, log_tail, false, options)?;
    let upper = invert_log_probability_with_options(law, log_tail, true, options)?;
    if !lower.root().is_success() || !upper.root().is_success() {
        return Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::IterationLimit,
            scioxide_core::ConvergenceDiagnostics::new(),
            "credible quantile inversion did not converge",
        ));
    }
    Ok(CredibleInterval {
        bounds: Interval::new(lower.value(), upper.value())?,
        probability,
    })
}

/// Marginal posterior distributions of the mean, variance and standard deviation of normal
/// observations, from [`NormalInverseGammaPrior::marginals`].
///
/// Each distribution provides its density, tail probabilities, quantiles, moments and
/// sampling.
#[derive(Debug, Clone)]
#[must_use]
pub struct NormalPosteriorMarginals<S: RealScalar> {
    mean: Affine<StudentT<S>, S>,
    variance: Affine<GeneralizedGamma<S>, S>,
    standard_deviation: Affine<GeneralizedGamma<S>, S>,
    parameters: NormalInverseGammaPrior<S>,
}
impl<S: RealScalar> NormalPosteriorMarginals<S> {
    /// Returns the Student t distribution of the mean `μ`.
    pub fn mean_law(&self) -> &Affine<StudentT<S>, S> {
        &self.mean
    }
    /// Returns the inverse-gamma distribution of the variance `σ²`.
    pub fn variance_law(&self) -> &Affine<GeneralizedGamma<S>, S> {
        &self.variance
    }
    /// Returns the distribution of the standard deviation `σ`, the square root of an
    /// inverse-gamma variable.
    pub fn standard_deviation_law(&self) -> &Affine<GeneralizedGamma<S>, S> {
        &self.standard_deviation
    }
    /// Returns the posterior mean and equal-tailed credible interval of `μ`, `σ²` and `σ`.
    ///
    /// The posterior mean of `μ` and `σ` exists for `a > 1/2` and that of `σ²` for `a > 1`.
    /// Otherwise it is `None`.
    ///
    /// # Errors
    ///
    /// - The errors of [`credible_interval`].
    /// - Errors from computing a posterior mean.
    pub fn credible_summary(
        &self,
        probability: ConfidenceProbability,
        options: &InversionOptions<S>,
    ) -> Result<NormalCredibleSummary<S>, SciOxideError> {
        let a: f64 = self.parameters.shape.round_to()?;
        Ok(NormalCredibleSummary {
            mean: CredibleSummary {
                mean: if a > 0.5 {
                    Some(self.mean.mean()?)
                } else {
                    None
                },
                interval: credible_interval(&self.mean, probability, options)?,
            },
            variance: CredibleSummary {
                mean: if a > 1.0 {
                    Some(self.variance.mean()?)
                } else {
                    None
                },
                interval: credible_interval(&self.variance, probability, options)?,
            },
            standard_deviation: CredibleSummary {
                mean: if a > 0.5 {
                    Some(self.standard_deviation.mean()?)
                } else {
                    None
                },
                interval: credible_interval(&self.standard_deviation, probability, options)?,
            },
        })
    }
}
/// Posterior mean and equal-tailed credible interval of one parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct CredibleSummary<S: RealScalar> {
    /// Posterior mean, or `None` if it does not exist.
    pub mean: Option<S>,
    /// Equal-tailed credible interval.
    pub interval: CredibleInterval<S>,
}
/// Posterior means and credible intervals of the mean, variance and standard deviation of
/// normal observations.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct NormalCredibleSummary<S: RealScalar> {
    /// Summary of the mean `μ`.
    pub mean: CredibleSummary<S>,
    /// Summary of the variance `σ²`.
    pub variance: CredibleSummary<S>,
    /// Summary of the standard deviation `σ`.
    pub standard_deviation: CredibleSummary<S>,
}
/// Computes the marginal posterior distributions of the mean, variance and standard deviation
/// of normal data under the reference prior `p(μ, σ²) ∝ 1/σ²`.
///
/// For `n` observations with mean `x̄` and sample variance `s²`, the mean has the distribution
/// `x̄ + (s/√n) T` with `T` Student t with `n − 1` degrees of freedom, and the variance has
/// the inverse-gamma distribution `InvGamma((n − 1)/2, (n − 1)s²/2)`. These distributions are
/// exact for every `n ≥ 2`. The cost is `O(n)`.
///
/// # Errors
///
/// - `NonFinite` if a value is NaN or infinite.
/// - `InvalidInput` if there are fewer than two values or all values are equal.
/// - `NumericalInstability` if a parameter overflows `S`.
///
/// # References
///
/// A. Gelman et al., *Bayesian Data Analysis*, 3rd ed., CRC Press, 2013, §3.2.
pub fn normal_posterior_marginals<S: RealScalar>(
    values: &[S],
) -> Result<NormalPosteriorMarginals<S>, SciOxideError> {
    NormalInverseGammaPrior::reference()
        .update_batch(values)?
        .marginals()
}
/// Computes posterior means and equal-tailed credible intervals of the mean, variance and
/// standard deviation of normal data under the reference prior `p(μ, σ²) ∝ 1/σ²`.
///
/// This is [`NormalPosteriorMarginals::credible_summary`] applied to
/// [`normal_posterior_marginals`]. Each of the six interval ends has its own work budget from
/// `options`.
///
/// # Errors
///
/// - The errors of [`normal_posterior_marginals`] and
///   [`NormalPosteriorMarginals::credible_summary`].
pub fn normal_credible_summary<S: RealScalar>(
    values: &[S],
    probability: ConfidenceProbability,
    options: &InversionOptions<S>,
) -> Result<NormalCredibleSummary<S>, SciOxideError> {
    normal_posterior_marginals(values)?.credible_summary(probability, options)
}

/// Normal-inverse-gamma update of one prior from a matrix of observations, for
/// [`normal_posteriors_into`].
///
/// Every element of the matrix is an observation, read in row-major order. Without weights
/// each observation counts once. [`Self::with_weights`] adds frequency or analytic weights of
/// the same shape.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, ExecutionPolicy};
/// use scioxide_linalg::MatrixView;
/// use scioxide_stats::{
///     Analytic, NormalInverseGammaPrior, NormalPosteriorRequest, Weights, normal_posteriors_into,
/// };
/// let values = [1.0_f64, 2.0, 4.0];
/// let weights = [1.0, 2.0, 1.0];
/// let request = NormalPosteriorRequest::new(
///     NormalInverseGammaPrior::reference(),
///     MatrixView::from_slice(1, 3, &values)?,
/// )
/// .with_weights(Weights::<_, Analytic>::new(MatrixView::from_slice(1, 3, &weights)?));
/// let (mut output, mut staging) = ([None], [None]);
/// normal_posteriors_into(
///     &[request],
///     ExecutionPolicy::Sequential,
///     &mut output,
///     &mut staging,
///     ExecutionContext::sequential(),
/// )?;
/// // The posterior location is the weighted mean (1 + 2·2 + 4)/4 = 2.25.
/// assert_eq!(output[0].unwrap().parameters()?.0, 2.25);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// Reliability weights do not define a likelihood and do not compile:
///
/// ```compile_fail
/// use scioxide_linalg::MatrixView;
/// use scioxide_stats::{NormalInverseGammaPrior, NormalPosteriorRequest, Reliability, Weights};
/// let values = [1.0_f64, 2.0];
/// let view = MatrixView::from_slice(1, 2, &values).unwrap();
/// NormalPosteriorRequest::new(NormalInverseGammaPrior::reference(), view)
///     .with_weights(Weights::<_, Reliability>::new(view));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct NormalPosteriorRequest<'a, S: RealScalar> {
    prior: NormalInverseGammaPrior<S>,
    values: scioxide_linalg::MatrixView<'a, S>,
    weights: Option<(scioxide_linalg::MatrixView<'a, S>, WeightSemantics)>,
}
impl<'a, S: RealScalar> NormalPosteriorRequest<'a, S> {
    /// Creates a request to update `prior` with the unweighted observations in `values`.
    ///
    /// `values` may be strided or transposed. A posterior from the reference prior stays
    /// improper with fewer than two observations.
    pub const fn new(
        prior: NormalInverseGammaPrior<S>,
        values: scioxide_linalg::MatrixView<'a, S>,
    ) -> Self {
        Self {
            prior,
            values,
            weights: None,
        }
    }
    /// Adds frequency or analytic weights, one per observation at the same position.
    ///
    /// [`normal_posteriors_into`] checks that the shapes match. See
    /// [`NormalInverseGammaPrior::update_weighted`] for the meaning of the weights.
    pub fn with_weights<M: ConjugateWeights>(
        mut self,
        weights: Weights<scioxide_linalg::MatrixView<'a, S>, M>,
    ) -> Self {
        self.weights = Some((weights.into_storage(), M::SEMANTICS));
        self
    }
    /// Returns the distribution to update.
    pub const fn prior(&self) -> NormalInverseGammaPrior<S> {
        self.prior
    }
    /// Returns the observations.
    pub const fn values(&self) -> scioxide_linalg::MatrixView<'a, S> {
        self.values
    }
    /// Returns the weights and their kind, or `None` for unweighted observations.
    pub const fn weights(&self) -> Option<(scioxide_linalg::MatrixView<'a, S>, WeightSemantics)> {
        self.weights
    }
}
/// Computes the posterior of each request and writes it to the same position of `output`.
///
/// Requests run in parallel under `execution`, on at most 64 workers. Each request reads its
/// observations in row-major order, so the result does not depend on the number of workers.
/// `staging` is scratch storage of at least `requests.len()` entries. The cost is linear in
/// the total number of observations. On success every entry of `output` is `Some`.
///
/// # Errors
///
/// - `InvalidInput` if `output.len() ≠ requests.len()`, `staging.len() < requests.len()`, or
///   the weights of a request differ in shape from its observations.
/// - `NonFinite` if an observation or weight is NaN or infinite.
/// - `InvalidInput` if a weight is negative.
/// - `ScalarConversion` if a frequency weight is not an integer.
/// - `NumericalInstability` if a parameter overflows the extended exponent range.
/// - `ExecutionStopped` on cancellation.
pub fn normal_posteriors_into<S>(
    requests: &[NormalPosteriorRequest<'_, S>],
    execution: scioxide_core::ExecutionPolicy,
    output: &mut [Option<NormalInverseGammaPrior<S>>],
    staging: &mut [Option<NormalInverseGammaPrior<S>>],
    context: scioxide_core::ExecutionContext<'_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + scioxide_core::NumericScalar<Real = S>,
{
    if output.len() != requests.len() || staging.len() < requests.len() {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "normal posterior output/staging length",
        ));
    }
    for request in requests {
        if let Some((weights, _)) = request.weights
            && (weights.rows() != request.values.rows() || weights.cols() != request.values.cols())
        {
            return Err(invalid(
                scioxide_core::argument!("requests"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "normal weights must match observation shape",
            ));
        }
    }
    let options = crate::StatisticsOptions::new().with_execution_policy(execution);
    let mut workers = [(); 64];
    crate::estimation::execute(
        options,
        context,
        requests.len(),
        1,
        &mut workers,
        staging,
        |index, _, result| {
            let request = requests[index];
            let semantics = request
                .weights
                .map_or(WeightSemantics::Frequency, |(_, semantics)| semantics);
            let mut moments = WeightedMoments::new(semantics, MomentOrder::Second);
            for row in 0..request.values.rows() {
                for col in 0..request.values.cols() {
                    context.check()?;
                    let weight = if let Some((weights, _)) = request.weights {
                        weights.get(row, col)?
                    } else {
                        S::ONE
                    };
                    // Reuse the pure extended-range kernel without projecting
                    // the population variance: only posterior law parameters
                    // need to fit the caller's scalar.
                    moments = moments.observed(request.values.get(row, col)?, weight)?;
                }
            }
            result[0] = Some(request.prior.update_core(&moments)?);
            Ok(())
        },
    )?;
    output.copy_from_slice(&staging[..requests.len()]);
    Ok(())
}
