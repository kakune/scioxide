//! Standard errors and confidence intervals of Monte Carlo, QMC and stratified estimates.

use scioxide_core::{Interval, RealScalar, Scaled, SciOxideError};
use scioxide_stats::{
    ConfidenceProbability, MeanConfidenceInterval, MeanIntervalMethod, MomentsSnapshot,
};

use super::MonteCarloOptions;

/// Independent units from which the variance of an estimate is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarianceBasis {
    /// The `n` observations `V f(x) / q` of one run: the estimate is their mean,
    /// its variance `s²/n` with `n − 1` degrees of freedom.
    Observations,
    /// The `R` independent replicate estimates, pseudo-random or randomized QMC:
    /// the estimate is their mean, its variance `s_R²/R` with `R − 1` degrees of
    /// freedom.
    Replicates,
    /// The strata of a stratified plan: the estimate is the sum of the stratum
    /// estimates, its variance the sum of their variances, with Welch–Satterthwaite
    /// degrees of freedom.
    Strata,
}

/// Standard error and confidence interval of one real component of a Monte Carlo
/// estimate.
///
/// The confidence interval is two-sided:
///
/// ```text
/// I = estimate ± c · SE
/// ```
///
/// where `SE` is the square root of the retained estimator variance and `c` is the
/// critical value for the confidence probability `p`: the `(1 + p)/2` quantile of
/// Student's t distribution with [`degrees_of_freedom`](Self::degrees_of_freedom)
/// degrees of freedom, or of the standard normal distribution for
/// `NormalAsymptotic`. Its coverage is nominal: it holds only approximately, under
/// the assumptions of the method. The standard error stays in extended range
/// through the endpoint calculation even when [`standard_error`](Self::standard_error)
/// rounds to zero. An estimated variance cannot detect regions
/// of the integrand that no sample reached. See
/// [error estimates](crate#error-estimates).
///
/// Each real component (for example the real and imaginary parts of a complex
/// integrand) has its own statistics. The components share their observations,
/// so their intervals do not hold simultaneously.
///
/// Integration does not compute the critical value. The first call of
/// [`confidence_interval`](Self::confidence_interval) or
/// [`interval_report`](Self::interval_report) computes it and stores the result,
/// including an error, in this value. These methods take `&mut self`, so clone
/// the statistics returned by [`MonteCarloWorkspace::statistics`](crate::MonteCarloWorkspace::statistics) to call them.
/// Equality ignores the stored interval.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, Interval};
/// use scioxide_integrate::{MonteCarloOptions, MonteCarloIntegrator, RandomStreams, VarianceBasis};
/// use scioxide_random::{Philox4x64, SeedableRng};
/// let plan = MonteCarloIntegrator::new(&[Interval::new(0.0, 1.0)?], MonteCarloOptions::default())?;
/// let cx = ExecutionContext::sequential_native();
/// let mut workspace = plan.workspace(1, cx)?;
/// let root = Philox4x64::from_seed(7);
/// let (f, streams) = (|x: &[f64]| Ok(x[0]), RandomStreams::new(&root, 0));
/// let mut value = 0.0;
/// plan.integrate_into(&f, streams, &mut value, &mut workspace, cx)?;
/// let mut stats = workspace.statistics()[0].clone().expect("statistics");
/// assert_eq!(stats.basis(), VarianceBasis::Observations);
/// // 4,096 observations give 4,095 degrees of freedom.
/// assert_eq!(stats.degrees_of_freedom(), Some(4095.0));
/// assert!(stats.confidence_interval()?.lower() < value);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. B. Owen, *Monte Carlo theory, methods and examples*, 2013 (confidence
///   intervals, importance sampling, effective sample size, randomized QMC).
/// - F. E. Satterthwaite, "An approximate distribution of estimates of variance
///   components", *Biometrics Bulletin* 2(6), 1946.
#[derive(Debug, Clone)]
#[must_use]
pub struct MonteCarloStatistics<S: RealScalar> {
    estimate: Estimate<S>,
    confidence: ConfidenceProbability,
    method: MeanIntervalMethod,
    interval: Option<Result<MeanConfidenceInterval<S>, SciOxideError>>,
}

/// Inputs of one statistics record before its interval is constructed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Estimate<S: RealScalar> {
    pub(super) basis: VarianceBasis,
    pub(super) units: usize,
    pub(super) samples: usize,
    pub(super) moments: Option<MomentsSnapshot<S>>,
    pub(super) value: S,
    pub(super) estimator_variance: Scaled<S>,
    pub(super) standard_error: S,
    pub(super) degrees_of_freedom: Option<S>,
    pub(super) effective_sample_size: Option<S>,
}

impl<S: RealScalar> MonteCarloStatistics<S> {
    /// Stores the interval options; the critical value is computed on first use.
    pub(super) fn new(estimate: Estimate<S>, options: &MonteCarloOptions) -> Self {
        Self {
            estimate,
            confidence: options.confidence_probability(),
            method: options.interval_method(),
            interval: None,
        }
    }

    /// Builds the statistics of the mean of the units summarized by `moments`.
    pub(super) fn from_moments(
        basis: VarianceBasis,
        moments: MomentsSnapshot<S>,
        samples: usize,
        effective_sample_size: Option<S>,
        options: &MonteCarloOptions,
    ) -> Result<Self, SciOxideError> {
        let missing = || {
            super::invalid(
                scioxide_core::argument!("moments"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "estimator dispersion requires at least two units",
            )
        };
        // The variance is a range diagnostic: it can leave the IEEE range while
        // the estimate, its standard error and its interval stay representable.
        let estimator_variance = moments
            .scaled_variance_of_sample_mean()?
            .ok_or_else(missing)?;
        let standard_error = estimator_variance.sqrt()?.round()?;
        Ok(Self::new(
            Estimate {
                basis,
                units: moments.count(),
                samples,
                moments: Some(moments),
                value: moments.mean()?.ok_or_else(missing)?,
                estimator_variance,
                standard_error,
                degrees_of_freedom: Some(S::try_from_f64_lossy(
                    moments.count().saturating_sub(1) as f64
                )?),
                effective_sample_size,
            },
            options,
        ))
    }

    /// Returns the units from which the variance is computed.
    pub const fn basis(&self) -> VarianceBasis {
        self.estimate.basis
    }

    /// Returns the number of independent units: observations, replicates or strata.
    #[must_use]
    pub const fn units(&self) -> usize {
        self.estimate.units
    }

    /// Returns the total number of observations behind the estimate.
    #[must_use]
    pub const fn samples(&self) -> usize {
        self.estimate.samples
    }

    /// Returns the moments of the observations or replicate estimates, or `None`
    /// for strata.
    #[must_use]
    pub const fn moments(&self) -> Option<&MomentsSnapshot<S>> {
        self.estimate.moments.as_ref()
    }

    /// Returns the estimated variance of the estimate: `s²/n` for `n`
    /// observations or replicates with sample variance `s²`, and the sum of
    /// the stratum variances for strata.
    ///
    /// The variance is a [`Scaled`] value with an extended exponent range, since it
    /// can overflow or underflow `S`. It is retained independently of the rounded
    /// standard error, including when that error underflows to zero.
    /// [`Scaled::round`] converts it to `S` and returns `NumericalInstability`
    /// (`Overflow`) if it does not fit.
    #[must_use]
    pub const fn estimator_variance(&self) -> Scaled<S> {
        self.estimate.estimator_variance
    }

    /// Returns the estimated standard error of the estimate.
    #[must_use]
    pub const fn standard_error(&self) -> S {
        self.estimate.standard_error
    }

    /// Returns the degrees of freedom of the variance: `n − 1` for `n`
    /// observations, `R − 1` for `R` replicates, or the Welch–Satterthwaite value
    /// `ν = V² / Σ_c (v_c² / (n_c − 1))` for strata, where `v_c` is the variance and
    /// `n_c` the number of observations of stratum `c`, and `V = Σ_c v_c`.
    #[must_use]
    pub const fn degrees_of_freedom(&self) -> Option<S> {
        self.estimate.degrees_of_freedom
    }

    /// Returns the confidence probability of the interval.
    pub const fn confidence_probability(&self) -> ConfidenceProbability {
        self.confidence
    }

    /// Returns the estimate of the integral.
    pub const fn estimate(&self) -> S {
        self.estimate.value
    }

    /// Returns the confidence interval for the integral.
    ///
    /// The first call computes the interval; later calls return the stored result.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if an endpoint is not representable.
    /// - The error of the Student-t quantile computation if it fails.
    ///
    /// The estimate and the integration report stay valid after an error.
    pub fn confidence_interval(&mut self) -> Result<Interval<S>, SciOxideError> {
        self.interval_report().map(|interval| interval.bounds())
    }

    /// Returns the confidence interval with its method, degrees of freedom and
    /// assumptions.
    ///
    /// # Errors
    ///
    /// The errors of [`confidence_interval`](Self::confidence_interval); an error
    /// is stored and returned again by later calls.
    pub fn interval_report(&mut self) -> Result<MeanConfidenceInterval<S>, SciOxideError> {
        let degrees = match self.method {
            MeanIntervalMethod::StudentT => self.estimate.degrees_of_freedom.map(S::into_f64),
            _ => None,
        };
        self.interval
            .get_or_insert_with(|| {
                scioxide_stats::MeanIntervalPlan::new(
                    self.confidence,
                    self.method,
                    degrees,
                    Default::default(),
                )
                .and_then(|plan| {
                    plan.interval_scaled(
                        self.estimate.value,
                        self.estimate.estimator_variance.sqrt()?,
                    )
                })
            })
            .clone()
    }

    /// Returns Kish's effective sample size `(Σ wᵢ)² / Σ wᵢ²` of the importance
    /// weights `wᵢ = 1/q(yᵢ)`, or `None` unless the basis is
    /// [`VarianceBasis::Observations`].
    ///
    /// It equals the number of observations for a uniform proposal and is smaller
    /// when the weights are uneven. It is a diagnostic of the proposal, not a
    /// degrees-of-freedom value.
    #[must_use]
    pub const fn effective_sample_size(&self) -> Option<S> {
        self.estimate.effective_sample_size
    }
}

impl<S: RealScalar> PartialEq for MonteCarloStatistics<S> {
    fn eq(&self, other: &Self) -> bool {
        self.estimate == other.estimate
            && self.confidence == other.confidence
            && self.method == other.method
    }
}

/// Returns the summed stratum variance `V` and the Welch–Satterthwaite degrees of
/// freedom `V² / Σ (v_c² / (n_c − 1))`, computed in extended range; `fallback` is
/// returned when every stratum variance is zero.
pub(super) fn welch_satterthwaite<S: RealScalar>(
    strata: impl IntoIterator<Item = Result<(Scaled<S>, usize), SciOxideError>>,
    fallback: usize,
) -> Result<(Scaled<S>, S), SciOxideError> {
    let mut variance = scioxide_core::PromotingCompensatedSum::ZERO;
    let mut denominator = scioxide_core::PromotingCompensatedSum::ZERO;
    for stratum in strata {
        let (stratum, samples) = stratum?;
        let degrees = S::try_from_f64_lossy(samples.saturating_sub(1).max(1) as f64)?;
        let native = stratum
            .try_value()
            .ok()
            .filter(|value| stratum.is_zero() || value.is_finite() && *value >= S::MIN_POSITIVE);
        if let Some(value) = native {
            variance.add_value(value)?;
            let square = value * value;
            let term = square / degrees;
            if value == S::ZERO
                || square.is_finite() && square >= S::MIN_POSITIVE && term >= S::MIN_POSITIVE
            {
                denominator.add_value(term)?;
                continue;
            }
        } else {
            variance.add_scaled(stratum)?;
        }
        denominator.add_scaled(stratum.square()?.div(Scaled::from_value(degrees)?)?)?;
    }
    let variance = variance.total()?;
    let denominator = denominator.total()?;
    let degrees = if denominator.is_zero() {
        S::try_from_f64_lossy(fallback.max(1) as f64)?
    } else {
        variance.square()?.div(denominator)?.round()?
    };
    Ok((variance, degrees))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn estimator_variance_survives_standard_error_underflow() {
        fn check<S: RealScalar>() {
            let tiny = S::ZERO.next_after(S::ONE);
            let moments = scioxide_stats::OnlineMoments::from_slice(
                &[S::ZERO, tiny, S::ZERO, tiny],
                scioxide_stats::MomentOrder::Second,
            )
            .unwrap()
            .snapshot();
            let mut statistics = MonteCarloStatistics::from_moments(
                VarianceBasis::Observations,
                moments,
                4,
                None,
                &MonteCarloOptions::default(),
            )
            .unwrap();
            assert_eq!(statistics.standard_error(), S::ZERO);
            let expected = Scaled::from_value(tiny)
                .unwrap()
                .square()
                .unwrap()
                .div(Scaled::from_value(S::try_from_usize_exact(12).unwrap()).unwrap())
                .unwrap();
            let ratio = statistics
                .estimator_variance()
                .div(expected)
                .unwrap()
                .round()
                .unwrap();
            assert!((ratio - S::ONE).abs() <= S::try_from_usize_exact(8).unwrap() * S::EPSILON);
            let interval = statistics.confidence_interval().unwrap();
            assert_eq!(interval.lower(), -tiny);
            assert_eq!(interval.upper(), tiny);
        }
        check::<f32>();
        check::<f64>();
    }

    fn snapshot<S: RealScalar>(
        value: S,
        error: S,
        degrees: S,
        basis: VarianceBasis,
        options: &MonteCarloOptions,
    ) -> MonteCarloStatistics<S> {
        MonteCarloStatistics::new(
            Estimate {
                basis,
                units: 2,
                samples: 16,
                moments: None,
                value,
                estimator_variance: Scaled::from_value(error).unwrap().square().unwrap(),
                standard_error: error,
                degrees_of_freedom: Some(degrees),
                effective_sample_size: None,
            },
            options,
        )
    }

    #[test]
    fn deferred_intervals_preserve_exact_endpoints_and_snapshot_equality() {
        for confidence in [0.1, 0.95, 1.0 - f64::EPSILON] {
            for method in [
                MeanIntervalMethod::StudentT,
                MeanIntervalMethod::NormalAsymptotic,
            ] {
                let options = MonteCarloOptions::default()
                    .with_confidence_probability(ConfidenceProbability::new(confidence).unwrap())
                    .with_interval_method(method);
                for (degrees, basis) in [
                    (3.0, VarianceBasis::Observations),
                    (7.0, VarianceBasis::Replicates),
                    (1000.125, VarianceBasis::Strata),
                ] {
                    for error in [0.0, 0.1] {
                        let mut actual = snapshot(1.0_f64, error, degrees, basis, &options);
                        let original = actual.clone();
                        let expected = scioxide_stats::MeanIntervalPlan::new(
                            options.confidence_probability(),
                            method,
                            (method == MeanIntervalMethod::StudentT).then_some(degrees),
                            Default::default(),
                        )
                        .and_then(|plan| plan.interval(1.0, error))
                        .unwrap();
                        assert_eq!(actual.estimate(), 1.0);
                        assert_eq!(actual.interval_report().unwrap(), expected);
                        assert_eq!(actual.confidence_interval().unwrap(), expected.bounds());
                        assert_eq!(actual, original);
                        assert_eq!(actual.clone().interval_report().unwrap(), expected);
                        let mut narrow =
                            snapshot(1.0_f32, error as f32, degrees as f32, basis, &options);
                        let expected = scioxide_stats::MeanIntervalPlan::new(
                            options.confidence_probability(),
                            method,
                            (method == MeanIntervalMethod::StudentT)
                                .then_some((degrees as f32) as f64),
                            Default::default(),
                        )
                        .and_then(|plan| plan.interval(1.0_f32, error as f32))
                        .unwrap();
                        assert_eq!(narrow.interval_report().unwrap(), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn interval_range_failure_is_deferred_and_preserves_the_estimate() {
        let mut statistics = snapshot(
            f64::MAX,
            f64::MAX / 4.0,
            3.25,
            VarianceBasis::Strata,
            &MonteCarloOptions::default(),
        );
        let before = statistics.clone();
        assert_eq!(statistics.estimate(), f64::MAX);
        let first = statistics.interval_report().unwrap_err();
        assert_eq!(statistics.confidence_interval().unwrap_err(), first);
        assert_eq!(statistics, before);
        let mut zero = snapshot(
            f64::MAX,
            0.0,
            3.25,
            VarianceBasis::Strata,
            &MonteCarloOptions::default(),
        );
        let interval = zero.interval_report().unwrap();
        assert!(interval.is_degenerate());
        assert_eq!(interval.bounds().lower(), f64::MAX);
        assert_eq!(interval.bounds().upper(), f64::MAX);
    }
}
