//! Power of mean tests, simulated power and sample-size search.
use crate::error_helpers::invalid;
use crate::{ConfidenceProbability, StatisticsOptions, TestAlternative};
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use scioxide_core::backend::{PartitionSlot, try_filled, try_with_capacity};
use scioxide_core::{
    ExecutionContext, Interval, InvalidInputReason, NonFiniteOrigin, RealScalar, SciOxideError,
};
use scioxide_random::{
    Cdf, NoncentralStudentT, Normal, Quantile, Rng, SplittableRng, StudentT, Survival,
};

/// Null distribution of a mean test: z-test or Student's t-test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeanPowerReference {
    /// z-test with known standard deviation. The test statistic is normal under the alternative.
    KnownVariance,
    /// Student's t-test with estimated standard deviation. The test statistic has a noncentral
    /// t distribution under the alternative.
    EstimatedVariance,
}

/// Sampling design of a mean test.
///
/// For a paired design, use [`MeanPowerDesign::OneSample`] with the standard deviation of the
/// paired differences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeanPowerDesign {
    /// One sample or paired differences. `n` is the number of observations.
    OneSample,
    /// Two independent groups of equal size. `n` is the number of observations in each group.
    EqualGroups,
}

/// Normal alternative for a one-sample, paired or two-sample mean test, evaluated by [`power`].
///
/// The observations are independent and normal with a common standard deviation `σ`.
/// `significance` holds the significance level α itself, not `1 − α`.
#[derive(Debug, Clone, Copy)]
pub struct MeanPowerModel<S: RealScalar> {
    /// Difference `μ − μ₀` between the true mean (or mean difference) and its null value.
    pub effect: S,
    /// Standard deviation `σ > 0` of one observation or one paired difference.
    pub standard_deviation: S,
    /// Sampling design.
    pub design: MeanPowerDesign,
    /// z-test or t-test.
    pub reference: MeanPowerReference,
    /// Tail of the rejection region.
    pub alternative: TestAlternative,
    /// Significance level α in `(0, 1)`.
    pub significance: ConfidenceProbability,
}

/// Power of a test: an exact probability or a simulated rejection fraction.
///
/// An analytic estimate from [`PowerEstimate::analytic`] or [`power`] has no interval and zero
/// counts. A simulated estimate from [`PowerEstimate::simulated`] or
/// [`simulated_power`] holds the number of rejections among at least two
/// independent replicates, their fraction, and the Wilson score 95% confidence interval for the
/// power. The interval describes Monte Carlo sampling error. It is not a bound on rounding
/// error.
///
/// # References
///
/// E. B. Wilson, "Probable inference, the law of succession, and statistical inference",
/// *J. Amer. Statist. Assoc.* 22 (1927), 209–212.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerEstimate {
    probability: f64,
    interval: Option<Interval<f64>>,
    replicates: usize,
    rejections: usize,
}

impl PowerEstimate {
    /// Returns an analytic estimate with the given power.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `probability` is NaN or infinite.
    /// - `InvalidInput` if `probability` is outside `[0, 1]`.
    pub fn analytic(probability: f64) -> Result<Self, SciOxideError> {
        Ok(Self {
            probability: self::probability(probability, NonFiniteOrigin::Input)?,
            interval: None,
            replicates: 0,
            rejections: 0,
        })
    }
    /// Returns a simulated estimate from `rejections` among `replicates` independent tests.
    ///
    /// The power is the fraction `p̂ = r/m` of rejections `r` among replicates `m`. The interval
    /// is the Wilson score interval
    ///
    /// ```text
    /// (p̂ + z²/(2m) ± z √(p̂(1 − p̂)/m + z²/(4m²))) / (1 + z²/m),
    /// ```
    ///
    /// with `z = 1.959963984540054`, the 0.975 quantile of the standard normal distribution.
    /// The lower end is 0 when `r = 0` and the upper end is 1 when `r = m`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `replicates < 2` or `rejections > replicates`.
    pub fn simulated(rejections: usize, replicates: usize) -> Result<Self, SciOxideError> {
        simulation_estimate(rejections, replicates)
    }
    /// Returns the analytic power or the simulated rejection fraction.
    pub const fn probability(&self) -> f64 {
        self.probability
    }
    /// Returns the Wilson 95% interval of a simulated estimate, or `None` for an analytic one.
    pub const fn interval(&self) -> Option<Interval<f64>> {
        self.interval
    }
    /// Returns the number of simulated tests, or zero for an analytic estimate.
    pub const fn replicates(&self) -> usize {
        self.replicates
    }
    /// Returns the number of simulated rejections, or zero for an analytic estimate.
    pub const fn rejections(&self) -> usize {
        self.rejections
    }
}

fn probability(value: f64, origin: NonFiniteOrigin) -> Result<f64, SciOxideError> {
    if !value.is_finite() {
        return Err(SciOxideError::non_finite(
            origin,
            "probability must be finite",
        ));
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(invalid(
            scioxide_core::argument!("probability"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "probability must be in [0, 1]",
        ));
    }
    Ok(value)
}

fn open_probability(value: f64, argument: scioxide_core::Argument) -> Result<(), SciOxideError> {
    if !value.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "probability must be finite",
        ));
    }
    if value <= 0.0 || value >= 1.0 {
        return Err(invalid(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "probability must be strictly inside (0, 1)",
        ));
    }
    Ok(())
}

fn inconsistent(detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(
        scioxide_core::argument!("estimate"),
        InvalidInputReason::Inconsistent,
        detail,
    )
}

fn simulation_estimate(
    rejections: usize,
    replicates: usize,
) -> Result<PowerEstimate, SciOxideError> {
    if replicates < 2 || rejections > replicates {
        return Err(inconsistent(
            "simulation counts require 0 <= rejections <= replicates and replicates >= 2",
        ));
    }
    let probability = rejections as f64 / replicates as f64;
    let interval = Interval::try_from(wilson(rejections, replicates))?;
    if !interval.contains(probability) {
        return Err(inconsistent(
            "Wilson interval must contain the rejection fraction",
        ));
    }
    Ok(PowerEstimate {
        probability,
        interval: Some(interval),
        replicates,
        rejections,
    })
}

/// Computes the power of a z-test or t-test for a normal mean at sample size `n`.
///
/// The power is the probability that the test rejects at level α when the alternative in
/// `model` is true. With standard error `σ √(c/n)`, where `c = 1` for one sample and `c = 2`
/// for two equal groups, the noncentrality is
///
/// ```text
/// δ = (μ − μ₀) / (σ √(c/n)).
/// ```
///
/// For the z-test, the test statistic is `N(δ, 1)` and the critical values are standard normal
/// quantiles. For the t-test, the critical values are Student t quantiles with `ν = n − 1`
/// (one sample) or `ν = 2(n − 1)` (two groups) degrees of freedom, and the power is a tail
/// probability of the noncentral t distribution with `ν` degrees of freedom and noncentrality
/// `δ`. A two-sided test splits α equally between the tails and adds both tail probabilities.
/// The result is exact for normal data. It has no interval.
///
/// # Errors
///
/// - `InvalidInput` if `n < 2`, `n > 2⁵³`, `effect` or `standard_deviation` is not finite, or
///   `standard_deviation ≤ 0`.
/// - `NumericalInstability` if `δ` overflows.
/// - Errors from evaluating the normal, Student t or noncentral t distribution.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{
///     ConfidenceProbability, MeanPowerDesign, MeanPowerModel, MeanPowerReference, TestAlternative,
///     power,
/// };
/// // One-sided z-test with δ = 0.5 √25 = 2.5: power = Φ(2.5 − 1.644854) ≈ 0.8037659.
/// let model = MeanPowerModel {
///     effect: 0.5_f64,
///     standard_deviation: 1.0,
///     design: MeanPowerDesign::OneSample,
///     reference: MeanPowerReference::KnownVariance,
///     alternative: TestAlternative::Greater,
///     significance: ConfidenceProbability::new(0.05)?,
/// };
/// let estimate = power(model, 25)?;
/// assert!((estimate.probability() - 0.803_765_9).abs() < 1e-6);
/// assert!(estimate.interval().is_none());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn power<S: RealScalar>(
    model: MeanPowerModel<S>,
    n: usize,
) -> Result<PowerEstimate, SciOxideError> {
    if n < 2
        || !model.effect.is_finite()
        || !model.standard_deviation.is_finite()
        || model.standard_deviation <= S::ZERO
        || u64::try_from(n).map_or(true, |count| count > 1_u64 << 53)
    {
        return Err(invalid(
            scioxide_core::argument!("model"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid mean model or sample size",
        ));
    }
    let contrast = match model.design {
        MeanPowerDesign::OneSample => 1.0,
        MeanPowerDesign::EqualGroups => 2.0,
    };
    let delta = model.effect.into_f64()
        / model.standard_deviation.into_f64()
        / (contrast / n as f64).sqrt();
    if !delta.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "unrepresentable noncentrality",
        ));
    }
    let alpha = model.significance.get();
    let p = match model.reference {
        MeanPowerReference::KnownVariance => {
            let null = Normal::<f64>::new(0.0, 1.0)?;
            let actual = Normal::<f64>::new(delta, 1.0)?;
            match model.alternative {
                TestAlternative::Less => actual.cdf(null.quantile(alpha)?)?,
                TestAlternative::Greater => actual.sf(-null.quantile(alpha)?)?,
                TestAlternative::TwoSided => {
                    actual.cdf(null.quantile(alpha / 2.0)?)?
                        + actual.sf(-null.quantile(alpha / 2.0)?)?
                }
            }
        }
        MeanPowerReference::EstimatedVariance => {
            let df = match model.design {
                MeanPowerDesign::OneSample => (n - 1) as f64,
                MeanPowerDesign::EqualGroups => 2.0 * (n - 1) as f64,
            };
            let null = StudentT::<f64>::new(df)?;
            let actual = NoncentralStudentT::<f64>::with_options(
                df,
                delta,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )?;
            match model.alternative {
                TestAlternative::Less => actual.cdf(null.quantile(alpha)?)?,
                TestAlternative::Greater => actual.sf(-null.quantile(alpha)?)?,
                TestAlternative::TwoSided => {
                    actual.cdf(null.quantile(alpha / 2.0)?)?
                        + actual.sf(-null.quantile(alpha / 2.0)?)?
                }
            }
        }
    };
    Ok(PowerEstimate {
        probability: probability(p, NonFiniteOrigin::Output)?,
        interval: None,
        replicates: 0,
        rejections: 0,
    })
}

/// Simulates power with an explicit split-capable random stream.
///
/// Computes one p-value per replicate, counts the replicates with p-value below
/// `significance`, and returns the rejection fraction with its Wilson score interval
/// (see [`PowerEstimate::simulated`]).
///
/// # Errors
///
/// The errors of [`simulated_power_into`], with `replicates` in place of the output length.
#[allow(clippy::too_many_arguments)]
pub fn simulated_power<S, R, F, T>(
    sampler: F,
    test: T,
    first_size: usize,
    second_size: usize,
    significance: ConfidenceProbability,
    replicates: usize,
    root: &R,
) -> Result<PowerEstimate, SciOxideError>
where
    S: RealScalar,
    R: Rng<S> + SplittableRng + Sync,
    F: Fn(&mut R, &mut [S]) -> Result<(), SciOxideError> + Sync,
    T: Fn(&[S], &[S]) -> Result<f64, SciOxideError> + Sync,
{
    let mut output = try_filled(replicates, 0.0, scioxide_core::quantity!("power_output"))?;
    simulated_power_into(
        sampler,
        test,
        first_size,
        second_size,
        significance,
        root,
        StatisticsOptions::default(),
        &mut output,
        ExecutionContext::sequential_native(),
    )
    .map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["output", "options"],
            scioxide_core::argument!("replicates"),
        )
    })
}
/// Simulates power and atomically writes each replicate's p-value.
///
/// Costs one sampler and test evaluation per output element; storage scales
/// with sample sizes and the worker count.
///
/// # Errors
///
/// - `InvalidInput` if `first_size` is zero, `output` has fewer than two elements, or `test`
///   returns a p-value outside `[0, 1]`.
/// - `NonFinite` if `sampler` writes a NaN or infinite value or `test` returns one.
/// - `SizeOverflow`, allocation or `ResourceLimit` errors for the sample and p-value storage.
/// - Errors returned by `sampler` or `test`, and execution stops.
#[allow(clippy::too_many_arguments)]
pub fn simulated_power_into<S, R, F, T>(
    sampler: F,
    test: T,
    first_size: usize,
    second_size: usize,
    significance: ConfidenceProbability,
    root: &R,
    options: StatisticsOptions,
    output: &mut [f64],
    context: ExecutionContext<'_>,
) -> Result<PowerEstimate, SciOxideError>
where
    S: RealScalar,
    R: Rng<S> + SplittableRng + Sync,
    F: Fn(&mut R, &mut [S]) -> Result<(), SciOxideError> + Sync,
    T: Fn(&[S], &[S]) -> Result<f64, SciOxideError> + Sync,
{
    let replicates = output.len();
    if first_size == 0 {
        return Err(invalid(
            scioxide_core::argument!("first_size"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "simulation requires a nonempty first sample",
        ));
    }
    if replicates < 2 || u64::try_from(replicates).is_err() {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "simulation requires at least two replicates",
        ));
    }
    first_size
        .checked_add(second_size)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("power_sample")))?;
    let count = first_size + second_size;
    let execution = context.plan(
        options.execution_policy(),
        replicates,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: context.worker_budget(),
        },
    );
    let sample_bytes = count
        .checked_mul(core::mem::size_of::<S>())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("power_sample")))?;
    let actual_bytes = execution
        .worker_count()
        .checked_mul(sample_bytes)
        .and_then(|v| {
            replicates
                .checked_mul(core::mem::size_of::<f64>())?
                .checked_add(v)
        })
        .and_then(|v| {
            execution
                .worker_count()
                .checked_mul(core::mem::size_of::<PartitionSlot<(Vec<S>, &mut [f64])>>())?
                .checked_add(v)
        })
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("power_workspace")))?;
    crate::error_helpers::limit(
        actual_bytes,
        options.max_bytes(),
        scioxide_core::quantity!("power_bytes"),
    )?;
    let mut values = try_filled(replicates, 0.0, scioxide_core::quantity!("power_p_values"))?;
    let mut slots = try_with_capacity(
        execution.worker_count(),
        scioxide_core::quantity!("power_executor_slots"),
    )?;
    let mut remaining = values.as_mut_slice();
    for partition in execution.partitions() {
        let len = partition.range().len();
        let (lane, rest) = remaining.split_at_mut(len);
        remaining = rest;
        slots.push(PartitionSlot::new((
            try_filled(count, S::ZERO, scioxide_core::quantity!("power_sample"))?,
            lane,
        )));
    }
    context.execute_partitioned(
        execution,
        &slots[..execution.worker_count()],
        &|partition, (mut sample, lane), _| {
            for (offset, value) in lane.iter_mut().enumerate() {
                context.check()?;
                let index = partition.range().start + offset;
                let mut child = root.split(index as u64)?;
                sampler(&mut child, &mut sample)?;
                if sample.iter().any(|x| !x.is_finite()) {
                    return Err(SciOxideError::non_finite(
                        NonFiniteOrigin::Callback,
                        "sampler returned nonfinite data",
                    ));
                }
                *value = probability(
                    test(&sample[..first_size], &sample[first_size..])?,
                    NonFiniteOrigin::Callback,
                )?;
            }
            Ok(())
        },
    )?;

    drop(slots);
    let significance = significance.get();
    let rejections = values.iter().filter(|&&p| p < significance).count();
    let estimate = simulation_estimate(rejections, replicates)?;
    output.copy_from_slice(&values);
    Ok(estimate)
}

fn wilson(successes: usize, count: usize) -> (f64, f64) {
    let p = successes as f64 / count as f64;
    let n = count as f64;
    let z = 1.959_963_984_540_054_f64;
    let denominator = 1.0 + z * z / n;
    let center = (p + z * z / (2.0 * n)) / denominator;
    let radius = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denominator;
    let lo = if successes == 0 {
        0.0
    } else {
        (center - radius).max(0.0)
    };
    let hi = if successes == count {
        1.0
    } else {
        (center + radius).min(1.0)
    };
    (lo, hi)
}

/// Smallest sample size found by [`sample_size`] and its power.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleSizeResult {
    /// Smallest size whose power reaches the target.
    pub size: usize,
    /// Power at `size`.
    pub estimate: PowerEstimate,
    /// Number of sizes evaluated, including `size`.
    pub evaluated: usize,
}

/// Finds the smallest sample size in `minimum..=maximum` whose power reaches `target`.
///
/// The search calls `evaluate` for `n = minimum, minimum + 1, …` and stops at the first size
/// that qualifies, so it makes at most `maximum − minimum + 1` calls. An analytic estimate
/// qualifies when its power is at least `target`. A simulated estimate qualifies when the lower
/// end of its Wilson interval is at least `target`. Returns `None` if no size in the range
/// qualifies. A larger size outside the range may still qualify.
///
/// # Errors
///
/// - `NonFinite` if `target` is NaN or infinite.
/// - `InvalidInput` if `target` is not in `(0, 1)`, `minimum < 2`, or `maximum < minimum`.
/// - The first error returned by `evaluate`.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{PowerEstimate, sample_size};
/// # fn main() -> Result<(), scioxide_core::SciOxideError> {
/// // The power reaches 0.8 first at n = 4, after evaluating n = 2, 3, 4.
/// let result = sample_size(2, 4, 0.8, |n| {
///     PowerEstimate::analytic(if n < 4 { 0.7 } else { 0.9 })
/// })?
/// .unwrap();
/// assert_eq!(result.size, 4);
/// assert_eq!(result.evaluated, 3);
/// # Ok(())
/// # }
/// ```
pub fn sample_size(
    minimum: usize,
    maximum: usize,
    target: f64,
    mut evaluate: impl FnMut(usize) -> Result<PowerEstimate, SciOxideError>,
) -> Result<Option<SampleSizeResult>, SciOxideError> {
    open_probability(target, scioxide_core::argument!("target"))?;
    if minimum < 2 || maximum < minimum {
        return Err(invalid(
            scioxide_core::argument!("minimum"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid sample size search",
        ));
    }
    for (offset, size) in (minimum..=maximum).enumerate() {
        let estimate = evaluate(size)?;
        let qualified = estimate
            .interval
            .map_or(estimate.probability, |interval| interval.lower());
        if qualified >= target {
            return Ok(Some(SampleSizeResult {
                size,
                estimate,
                evaluated: offset + 1,
            }));
        }
    }
    Ok(None)
}
