//! Options of Monte Carlo evaluation.

use scioxide_core::{Budget, ExecutionPolicy, SciOxideError};
use scioxide_stats::{ConfidenceProbability, MeanIntervalMethod};

/// Options of a Monte Carlo evaluation: sample count, replicates, confidence
/// interval and execution policy.
///
/// `samples` is the number of observations per replicate, or the total over all
/// strata of a stratified plan. With one replicate, the standard error comes from
/// the variance of the observations. With `R ≥ 2` replicates, the estimate is the
/// mean of `R` independent replicate estimates and the standard error comes from
/// their spread. The confidence interval describes sampling error only; it does
/// not bound bias. See [error estimates](crate#error-estimates).
///
/// The setters do not validate. [`MonteCarloIntegrator::new`](crate::MonteCarloIntegrator::new)
/// and the other plan constructors return `InvalidInput` if `samples < 2`, if
/// `replicates` is zero, or if the interval method is
/// [`MeanIntervalMethod::NormalKnownVariance`], since the variance is estimated.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, ExecutionPolicy};
/// use scioxide_integrate::MonteCarloOptions;
/// use scioxide_stats::{ConfidenceProbability, MeanIntervalMethod};
/// let options = MonteCarloOptions::default()
///     .with_samples(Budget::try_from(1 << 14)?)
///     .with_replicates(8)
///     .with_confidence_probability(ConfidenceProbability::new(0.99)?);
/// assert_eq!(options.samples().get(), 16_384);
/// assert_eq!(options.replicates(), 8);
/// assert_eq!(options.interval_method(), MeanIntervalMethod::StudentT);
/// assert_eq!(options.execution_policy(), ExecutionPolicy::Sequential);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct MonteCarloOptions {
    samples: Budget,
    replicates: usize,
    confidence: ConfidenceProbability,
    interval: MeanIntervalMethod,
    policy: ExecutionPolicy,
}

impl Default for MonteCarloOptions {
    /// Returns 4,096 samples, one replicate, a 95% Student-t interval and
    /// sequential execution.
    fn default() -> Self {
        Self {
            samples: Budget::ONE.saturating_add(4095),
            replicates: 1,
            confidence: ConfidenceProbability::default(),
            interval: MeanIntervalMethod::StudentT,
            policy: ExecutionPolicy::Sequential,
        }
    }
}

impl MonteCarloOptions {
    /// Sets the number of observations per replicate, or the total over all strata.
    pub const fn with_samples(mut self, samples: Budget) -> Self {
        self.samples = samples;
        self
    }

    /// Sets the number of independent replicates.
    pub const fn with_replicates(mut self, replicates: usize) -> Self {
        self.replicates = replicates;
        self
    }

    /// Sets the confidence probability of the two-sided interval.
    pub const fn with_confidence_probability(mut self, confidence: ConfidenceProbability) -> Self {
        self.confidence = confidence;
        self
    }

    /// Sets the interval method: `StudentT` (default) or `NormalAsymptotic`.
    pub const fn with_interval_method(mut self, method: MeanIntervalMethod) -> Self {
        self.interval = method;
        self
    }

    /// Sets how sample blocks and replicates are run in parallel.
    ///
    /// The result does not depend on the policy; see [`crate::RandomStreams`].
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Returns the number of observations per replicate, or the total over all strata.
    pub const fn samples(&self) -> Budget {
        self.samples
    }

    /// Returns the number of independent replicates.
    #[must_use]
    pub const fn replicates(&self) -> usize {
        self.replicates
    }

    /// Returns the confidence probability of the interval.
    pub const fn confidence_probability(&self) -> ConfidenceProbability {
        self.confidence
    }

    /// Returns the interval method.
    pub const fn interval_method(&self) -> MeanIntervalMethod {
        self.interval
    }

    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }

    /// Checks the options for a plan.
    pub(super) fn validate(&self) -> Result<(), SciOxideError> {
        if self.samples.get() < 2 {
            return Err(super::invalid(
                scioxide_core::argument!("samples"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Monte Carlo integration requires at least two samples",
            ));
        }
        if self.replicates == 0 {
            return Err(super::invalid(
                scioxide_core::argument!("replicates"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "Monte Carlo integration requires at least one replicate",
            ));
        }
        if self.interval == MeanIntervalMethod::NormalKnownVariance {
            return Err(super::invalid(
                scioxide_core::argument!("interval_method"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Monte Carlo variance is estimated, never known",
            ));
        }
        Ok(())
    }
}
