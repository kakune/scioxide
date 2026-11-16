//! Classical hypothesis tests on one sample, two samples, paired data and blocks.
use crate::error_helpers::{finite, invalid, number as checked};
mod location;
pub use location::{
    TTestSummary, TTestVariance, t_test_from_summaries, t_test_from_summaries_with_options,
    t_test_independent, t_test_independent_with_options, t_test_one_sample,
    t_test_one_sample_with_options, t_test_paired, t_test_paired_with_options,
};
mod discrete;
pub use discrete::{
    binomial_test, binomial_test_with_options, poisson_means_test, poisson_means_test_with_options,
    quantile_test, quantile_test_with_options,
};
mod association;
pub use association::{
    KendallVariant, KendallWeightCombination, chatterjee_test, chatterjee_test_with_options,
    kendall_test, kendall_test_with_options, pearson_test, pearson_test_with_options,
    point_biserial_test, point_biserial_test_with_options, somers_contingency_test,
    somers_contingency_test_with_options, somers_test, somers_test_with_options, spearman_test,
    spearman_test_with_options, weighted_kendall_tau,
};
mod permutation;
pub use permutation::{
    ClassicalPermutationResult, classical_permutation_test, classical_permutation_test_into,
};
mod ranks;
pub use ranks::{
    BrunnerMunzelCalibration, SignedRankZeros, brunner_munzel_test,
    brunner_munzel_test_with_options, bws_test, bws_test_with_options, mann_whitney_test,
    mann_whitney_test_with_options, rank_sum_test, rank_sum_test_with_options, wilcoxon_test,
    wilcoxon_test_with_options,
};
mod page;
pub use page::{page_trend_test, page_trend_test_with_options};
mod mgc;
pub use mgc::{MgcPlan, MgcStatistic};
mod batch;
pub use batch::{ClassicalBatchTest, ClassicalBatchWorkspace, classical_tests_into};
mod bws;
pub use bws::BwsPlan;

use crate::{ConfidenceProbability, TestAlternative};
use scioxide_core::{Argument, Interval, RealScalar, SciOxideError};
use scioxide_random::{Cdf, Normal, StudentT};

/// Selects how a classical test computes its null distribution.
///
/// A test that does not support an explicit choice returns `InvalidInput`
/// instead of falling back to another method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TestCalibration {
    /// Use the exact null distribution when its size fits
    /// [`ClassicalOptions::max_work`], and the large-sample approximation
    /// otherwise. Tests with only one method use it.
    #[default]
    Auto,
    /// Use the exact null distribution, and fail if it exceeds the work limit.
    Exact,
    /// Use the analytic or large-sample reference distribution documented on the test.
    Asymptotic,
}

/// Holds the options shared by the classical tests.
///
/// The default is a two-sided alternative, [`TestCalibration::Auto`], 95%
/// confidence, at most 100 000 exact null states, no workspace byte limit and a
/// half-unit continuity correction.
#[derive(Debug, Clone, Copy)]
pub struct ClassicalOptions {
    /// Direction of the population effect, as documented by each test.
    alternative: TestAlternative,
    /// Requested null calibration.
    calibration: TestCalibration,
    /// Coverage of two-sided confidence intervals.
    confidence: ConfidenceProbability,
    /// Maximum exact null states, or Poisson joint mass evaluations.
    max_work: usize,
    /// Byte limit on exact and permutation scratch, excluding input and output.
    max_workspace_bytes: usize,
    /// Continuity correction of normal approximations to rank statistics.
    continuity: crate::ContinuityCorrection,
}

impl ClassicalOptions {
    /// Returns the alternative hypothesis. Each test documents the direction of
    /// its effect.
    pub const fn alternative(&self) -> TestAlternative {
        self.alternative
    }
    /// Sets the alternative hypothesis.
    pub const fn with_alternative(mut self, value: TestAlternative) -> Self {
        self.alternative = value;
        self
    }
    /// Returns the requested null calibration.
    pub const fn calibration(&self) -> TestCalibration {
        self.calibration
    }
    /// Sets the requested null calibration.
    pub const fn with_calibration(mut self, value: TestCalibration) -> Self {
        self.calibration = value;
        self
    }
    /// Returns the coverage probability of two-sided confidence intervals.
    pub const fn confidence(&self) -> ConfidenceProbability {
        self.confidence
    }
    /// Sets the coverage probability of two-sided confidence intervals.
    pub const fn with_confidence(mut self, value: ConfidenceProbability) -> Self {
        self.confidence = value;
        self
    }
    /// Returns the largest number of exact null states (label assignments or
    /// dynamic-programming updates) a test may evaluate. For the Poisson E-test
    /// it bounds the number of joint mass evaluations.
    pub const fn max_work(&self) -> usize {
        self.max_work
    }
    /// Sets the limit on exact null states.
    pub const fn with_max_work(mut self, value: usize) -> Self {
        self.max_work = value;
        self
    }
    /// Returns the byte limit on the scratch of exact and permutation null
    /// distributions, excluding input and output.
    pub const fn max_workspace_bytes(&self) -> usize {
        self.max_workspace_bytes
    }
    /// Sets the byte limit on exact and permutation scratch.
    pub const fn with_max_workspace_bytes(mut self, value: usize) -> Self {
        self.max_workspace_bytes = value;
        self
    }
    /// Returns the continuity correction applied to normal approximations of
    /// discrete rank statistics.
    pub const fn continuity(&self) -> crate::ContinuityCorrection {
        self.continuity
    }
    /// Sets the continuity correction of normal rank approximations.
    pub const fn with_continuity(mut self, value: crate::ContinuityCorrection) -> Self {
        self.continuity = value;
        self
    }
}
impl Default for ClassicalOptions {
    fn default() -> Self {
        Self {
            alternative: TestAlternative::TwoSided,
            calibration: TestCalibration::Auto,
            confidence: ConfidenceProbability::default(),
            max_work: 100_000,
            max_workspace_bytes: usize::MAX,
            continuity: crate::ContinuityCorrection::HalfUnit,
        }
    }
}

/// Reference distribution a classical test actually used for its p-value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClassicalMethod {
    /// Student's t distribution. It is exact for the one-sample, paired and
    /// pooled t-tests under normal sampling, and approximate for Welch's test,
    /// Spearman's test and the Brunner–Munzel test.
    StudentT {
        /// Degrees of freedom, possibly non-integer (Welch–Satterthwaite).
        degrees_of_freedom: f64,
    },
    /// Standard normal large-sample approximation.
    Normal,
    /// Exact discrete null distribution or exact parametric tail probability.
    Exact,
    /// Poisson E-test, which evaluates the null at an estimated common rate. It
    /// is not an exact test.
    PoissonE,
    /// Permutation distribution conditional on the observed data.
    Permutation {
        /// Whether every assignment was enumerated.
        exact: bool,
        /// Number of assignments evaluated.
        replicates: usize,
    },
}

/// Sampling design a classical test assumes.
///
/// The design is an assumption of the test. It is not checked against the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestSamplePolicy {
    /// One independent scalar sample.
    Independent,
    /// Two independent samples of independent observations.
    IndependentGroups,
    /// Independent rows with paired coordinates.
    Paired,
    /// Independent complete blocks with repeated treatments.
    Blocks,
    /// Event counts with their exposures or numbers of trials.
    Counts,
    /// Sample means, unbiased variances and sample sizes.
    Summary,
}

/// Quantity reported by [`ClassicalTestResult::effect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassicalEffect {
    /// Observed mean minus the null value, or first mean minus second mean.
    MeanDifference,
    /// Binomial success probability.
    Proportion,
    /// Population quantile estimated by the empirical inverse CDF.
    Quantile,
    /// First event rate minus second event rate.
    RateDifference,
    /// The test's correlation or association coefficient.
    Association,
    /// `P(X > Y) + P(X = Y)/2` for `X` from the first sample and `Y` from the
    /// second.
    Superiority,
    /// Matched-pairs rank-biserial correlation `(W⁺ − W⁻)/(W⁺ + W⁻)`, where
    /// `W⁺` and `W⁻` are the sums of positive and negative signed ranks.
    RankBiserial,
}

/// Result of a classical hypothesis test.
///
/// Holds the test statistic, the p-value, an optional effect estimate with its
/// confidence interval, the reference distribution used and the sample sizes.
/// Confidence intervals are statistical estimates, not guaranteed enclosures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassicalTestResult<S: RealScalar> {
    /// Test statistic, oriented as documented on the test.
    statistic: S,
    /// Tail probability under the null hypothesis.
    p_value: f64,
    /// Effect estimate and its meaning.
    effect: Option<(ClassicalEffect, S)>,
    /// Two-sided confidence interval for the effect and its coverage.
    interval: Option<(Interval<S>, ConfidenceProbability)>,
    /// Reference distribution used.
    method: ClassicalMethod,
    /// Assumed sampling design.
    sample_policy: TestSamplePolicy,
    /// Missing-value policy applied to the input.
    missing_policy: crate::MissingPolicy,
    /// Non-finite-value policy applied to the input.
    non_finite_policy: crate::NonFinitePolicy,
    /// Sample sizes used.
    retained: [usize; 2],
    /// Observations, rows or blocks left out.
    omitted: [usize; 2],
    /// Null probability mass outside a truncated summation.
    omitted_null_mass: f64,
}
impl<S: RealScalar> ClassicalTestResult<S> {
    /// Returns the test statistic, oriented as documented on the test.
    pub fn statistic(&self) -> S {
        self.statistic
    }
    /// Returns the p-value in `[0, 1]`.
    pub fn p_value(&self) -> f64 {
        self.p_value
    }
    /// Returns the effect estimate and its meaning, if the test defines one.
    pub fn effect(&self) -> Option<(ClassicalEffect, S)> {
        self.effect
    }
    /// Returns the two-sided confidence interval for the effect, if the test
    /// computes one.
    pub fn interval(&self) -> Option<Interval<S>> {
        self.interval.map(|(bounds, _)| bounds)
    }
    /// Returns the coverage of [`Self::interval`]. It is present exactly when the
    /// interval is.
    pub fn confidence(&self) -> Option<ConfidenceProbability> {
        self.interval.map(|(_, confidence)| confidence)
    }
    /// Returns the reference distribution used for the p-value.
    pub fn method(&self) -> ClassicalMethod {
        self.method
    }
    /// Returns the sampling design the test assumes.
    pub fn sample_policy(&self) -> TestSamplePolicy {
        self.sample_policy
    }
    /// Returns the missing-value policy applied to the input.
    ///
    /// Direct calls on slices report [`MissingPolicy::Reject`](crate::MissingPolicy::Reject).
    /// [`classical_tests_into`] reports the policy of its options. Paired and
    /// block designs drop whole rows.
    pub fn missing_policy(&self) -> crate::MissingPolicy {
        self.missing_policy
    }
    /// Returns the policy for NaN and infinite values applied to the input.
    pub fn non_finite_policy(&self) -> crate::NonFinitePolicy {
        self.non_finite_policy
    }
    /// Returns the sample sizes used by the test.
    ///
    /// One-sample tests report `[n, 0]` and paired tests `[n, n]`. Page's test
    /// reports the numbers of blocks and treatments, and the Poisson E-test the
    /// two event counts.
    pub fn retained(&self) -> [usize; 2] {
        self.retained
    }
    /// Returns the numbers of observations left out under
    /// [`MissingPolicy::Skip`](crate::MissingPolicy::Skip) or
    /// [`NonFinitePolicy`](crate::NonFinitePolicy).
    ///
    /// Paired designs count complete rows and Page's test counts complete blocks.
    pub fn omitted(&self) -> [usize; 2] {
        self.omitted
    }
    /// Returns the null probability mass outside a truncated summation. It is
    /// zero except for [`poisson_means_test_with_options`].
    pub fn omitted_null_mass(&self) -> f64 {
        self.omitted_null_mass
    }
    /// Splits the result into its parts, in the order of the accessors above.
    /// The interval and its coverage stay together in one `Option`.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        S,
        f64,
        Option<(ClassicalEffect, S)>,
        Option<(Interval<S>, ConfidenceProbability)>,
        ClassicalMethod,
        TestSamplePolicy,
        crate::MissingPolicy,
        crate::NonFinitePolicy,
        [usize; 2],
        [usize; 2],
        f64,
    ) {
        (
            self.statistic,
            self.p_value,
            self.effect,
            self.interval,
            self.method,
            self.sample_policy,
            self.missing_policy,
            self.non_finite_policy,
            self.retained,
            self.omitted,
            self.omitted_null_mass,
        )
    }
}

// Private helpers report.
fn sample<S: RealScalar>(x: &[S], minimum: usize, argument: Argument) -> Result<(), SciOxideError> {
    crate::error_helpers::count::<S>(x.len())?;
    if x.len() < minimum {
        return Err(invalid(
            argument,
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient observations",
        ));
    }
    x.iter()
        .try_for_each(|&v| crate::error_helpers::input_finite(v).map(|_| ()))
}
fn paired<S: RealScalar>(x: &[S], y: &[S], minimum: usize) -> Result<(), SciOxideError> {
    if x.len() != y.len() {
        return Err(SciOxideError::dimension_mismatch(x.len(), y.len()));
    }
    sample(x, minimum, scioxide_core::argument!("x"))?;
    sample(y, minimum, scioxide_core::argument!("y"))
}
fn probability(p: f64) -> Result<f64, SciOxideError> {
    if p.is_finite() && (0.0..=1.0).contains(&p) {
        Ok(p)
    } else {
        Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "calculated probability outside [0,1]",
        ))
    }
}
fn tails(lower: f64, upper: f64, alternative: TestAlternative) -> Result<f64, SciOxideError> {
    probability(match alternative {
        TestAlternative::Less => lower,
        TestAlternative::Greater => upper,
        TestAlternative::TwoSided => (2.0 * lower.min(upper)).min(1.0),
    })
}
fn normal_p(z: f64, alternative: TestAlternative) -> Result<f64, SciOxideError> {
    let law = Normal::new(0.0, 1.0)?;
    tails(law.cdf(z)?, law.cdf(-z)?, alternative)
}
fn student_p(t: f64, df: f64, alternative: TestAlternative) -> Result<f64, SciOxideError> {
    let law = StudentT::new(df)?;
    tails(law.cdf(t)?, law.cdf(-t)?, alternative)
}
fn result<S: RealScalar>(
    statistic: f64,
    p: f64,
    method: ClassicalMethod,
    sample_policy: TestSamplePolicy,
    retained: [usize; 2],
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    Ok(ClassicalTestResult {
        statistic: checked(statistic)?,
        p_value: probability(p)?,
        effect: None,
        interval: None,
        method,
        sample_policy,
        missing_policy: crate::MissingPolicy::Reject,
        non_finite_policy: crate::NonFinitePolicy::Reject,
        retained,
        omitted: [0, 0],
        omitted_null_mass: 0.0,
    })
}
