//! Student's t-tests for one sample, paired samples and two independent samples.
use super::*;
use crate::MomentOrder;
use scioxide_core::Scaled;
use scioxide_random::Quantile;

/// Selects the variance model of the two-sample t-test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TTestVariance {
    /// Welch's t-test: unequal variances, Welch–Satterthwaite degrees of freedom.
    #[default]
    Welch,
    /// Student's t-test: a common unknown variance, pooled, with `n + m − 2`
    /// degrees of freedom.
    Pooled,
}

/// Summary statistics of one sample: mean, unbiased variance and size.
#[derive(Debug, Clone, Copy)]
pub struct TTestSummary<S: RealScalar> {
    /// Sample mean.
    pub mean: S,
    /// Unbiased sample variance, with divisor `n − 1`.
    pub variance: S,
    /// Number of independent observations, at least two.
    pub count: usize,
}

fn moments<S: RealScalar>(
    x: &[S],
    argument: scioxide_core::Argument,
) -> Result<(Scaled<f64>, Scaled<f64>), SciOxideError> {
    sample(x, 2, argument)?;
    let state = crate::OnlineMoments::from_slice(x, MomentOrder::Second)?
        .snapshot()
        .state;
    Ok((
        state.mean()?.total()?,
        state
            .m2()?
            .mul(Scaled::<f64>::from_value((x.len()) as f64)?)?
            .div(Scaled::<f64>::from_value((x.len() - 1) as f64)?)?,
    ))
}
#[allow(clippy::too_many_arguments)]
fn inference<S: RealScalar>(
    effect: Scaled<f64>,
    null: Scaled<f64>,
    variance: Scaled<f64>,
    df: f64,
    options: ClassicalOptions,
    policy: TestSamplePolicy,
    sizes: [usize; 2],
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    if variance.is_zero() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "Student statistic requires positive standard error",
        ));
    }
    let se = variance.sqrt()?;
    let t = effect.sub(null)?.div(se)?.round_to::<f64>()?;
    let mut out = result(
        t,
        student_p(t, df, options.alternative)?,
        ClassicalMethod::StudentT {
            degrees_of_freedom: df,
        },
        policy,
        sizes,
    )?;
    let critical = -StudentT::new(df)?.quantile((1.0 - options.confidence.get()) / 2.0)?;
    let radius = se.mul(Scaled::<f64>::from_value(critical)?)?;
    out.effect = Some((ClassicalEffect::MeanDifference, effect.round_to()?));
    out.interval = Some((
        Interval::new(
            effect.sub(radius)?.round_to()?,
            effect.add(radius)?.round_to()?,
        )?,
        options.confidence,
    ));
    Ok(out)
}
/// Performs the one-sample t-test of the mean against `null_mean`.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`t_test_one_sample_with_options`].
///
/// # Errors
///
/// Returns the errors of [`t_test_one_sample_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let actual = t_test_one_sample::<f64>(&x, 3.)?;
/// let configured = t_test_one_sample_with_options::<f64>(&x, 3., ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn t_test_one_sample<S: RealScalar>(
    x: &[S],
    null_mean: S,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    t_test_one_sample_with_options(x, null_mean, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the one-sample t-test of the mean against `null_mean`.
///
/// ```text
/// t = (x̄ − μ₀) / (s / √n),   ν = n − 1,
/// ```
///
/// where `x̄` is the sample mean, `s²` the unbiased sample variance, `n` the
/// sample size and `μ₀ = null_mean`. The p-value comes from Student's t
/// distribution with `ν` degrees of freedom, which is exact for independent
/// normal observations. The effect is `x̄ − μ₀` with its t confidence interval.
/// The cost is O(n) time and O(1) memory.
///
/// # Errors
///
/// - `InvalidInput` if `x` has fewer than two observations.
/// - `NonFinite` if `x` contains NaN or infinity. A non-finite `null_mean` is
///   also rejected.
/// - `NumericalInstability` (`Breakdown`) if the standard error is zero.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{ClassicalOptions, t_test_one_sample_with_options};
///
/// // x̄ = 5.1, s = √0.025, so t = 0.1 / (s / √5) = √2.
/// let x = [5.1_f64, 4.9, 5.3, 5.0, 5.2];
/// let test = t_test_one_sample_with_options(&x, 5.0, ClassicalOptions::default())?;
/// assert!((test.statistic() - 2.0_f64.sqrt()).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn t_test_one_sample_with_options<S: RealScalar>(
    x: &[S],
    null_mean: S,
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    finite(null_mean)?;
    let (mean, variance) = moments(x, scioxide_core::argument!("x"))?;
    inference(
        mean.sub(Scaled::<f64>::from_value(
            scioxide_core::RealScalar::into_f64(null_mean),
        )?)?,
        Scaled::<f64>::ZERO,
        variance.div(Scaled::<f64>::from_value((x.len()) as f64)?)?,
        (x.len() - 1) as f64,
        options,
        TestSamplePolicy::Independent,
        [x.len(), 0],
    )
}
/// Performs the paired t-test: the one-sample t-test of the differences `dᵢ = xᵢ − yᵢ` against zero.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`t_test_paired_with_options`].
///
/// # Errors
///
/// Returns the errors of [`t_test_paired_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = t_test_paired::<f64>(&x, &y)?;
/// let configured = t_test_paired_with_options::<f64>(&x, &y, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn t_test_paired<S: RealScalar>(
    x: &[S],
    y: &[S],
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    t_test_paired_with_options(x, y, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the paired t-test: the one-sample t-test of the differences
/// `dᵢ = xᵢ − yᵢ` against zero.
///
/// The effect is the mean difference with its t confidence interval. The
/// differences are formed in extended range, so they do not overflow. The cost
/// is O(n) time and O(1) memory.
///
/// # Errors
///
/// - `DimensionMismatch` if `x` and `y` have different lengths.
/// - `InvalidInput` if there are fewer than two pairs.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
/// - `NumericalInstability` (`Breakdown`) if the standard error is zero.
pub fn t_test_paired_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    paired(x, y, 2)?;
    let difference = |i: usize| {
        Scaled::<f64>::from_value(x[i].into_f64())?.sub(Scaled::<f64>::from_value(y[i].into_f64())?)
    };
    let native = |i: usize| {
        let (x, y) = (x[i].into_f64(), y[i].into_f64());
        (crate::arithmetic::native(x, 64) && crate::arithmetic::native(y, 64)).then_some(x - y)
    };
    let (mean, population) = crate::arithmetic::scaled_second(x.len(), native, difference)?;
    inference(
        mean.total()?,
        Scaled::ZERO,
        population.div(Scaled::from_value((x.len() - 1) as f64)?)?,
        (x.len() - 1) as f64,
        options,
        TestSamplePolicy::Paired,
        [x.len(), y.len()],
    )
}

fn independent<S: RealScalar>(
    a: (Scaled<f64>, Scaled<f64>),
    b: (Scaled<f64>, Scaled<f64>),
    sizes: [usize; 2],
    variance: TTestVariance,
    options: ClassicalOptions,
    policy: TestSamplePolicy,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    if variance == TTestVariance::Welch && options.calibration == TestCalibration::Exact {
        return Err(invalid(
            scioxide_core::argument!("options.calibration"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Welch calibration is approximate; exact requires a justified pooled-variance model",
        ));
    }
    let [n, m] = sizes;
    let (se2, df) = match variance {
        TTestVariance::Pooled => {
            let df = (n - 1) as f64 + (m - 1) as f64;
            let pooled =
                a.1.mul(Scaled::<f64>::from_value((n - 1) as f64)?)?
                    .add(b.1.mul(Scaled::<f64>::from_value((m - 1) as f64)?)?)?
                    .div(Scaled::<f64>::from_value(df)?)?;
            (
                pooled
                    .div(Scaled::<f64>::from_value((n) as f64)?)?
                    .add(pooled.div(Scaled::<f64>::from_value((m) as f64)?)?)?,
                df,
            )
        }
        TTestVariance::Welch => {
            let v1 = a.1.div(Scaled::<f64>::from_value((n) as f64)?)?;
            let v2 = b.1.div(Scaled::<f64>::from_value((m) as f64)?)?;
            let se = v1.add(v2)?;
            if se.is_zero() {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "both sample variances vanish",
                ));
            }
            let p = v1.div(se)?.round_to::<f64>()?;
            let q = v2.div(se)?.round_to::<f64>()?;
            (se, 1.0 / (p * p / (n - 1) as f64 + q * q / (m - 1) as f64))
        }
    };
    inference(
        a.0.sub(b.0)?,
        Scaled::<f64>::ZERO,
        se2,
        df,
        options,
        policy,
        sizes,
    )
}
/// Performs the two-sample t-test of the difference of means `x̄ − ȳ`.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`t_test_independent_with_options`].
///
/// # Errors
///
/// Returns the errors of [`t_test_independent_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = t_test_independent::<f64>(&x, &y)?;
/// let configured = t_test_independent_with_options::<f64>(&x, &y, TTestVariance::Welch, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// B. L. Welch, "The generalization of 'Student's' problem when several
/// different population variances are involved", *Biometrika* 34 (1947), 28–35.
pub fn t_test_independent<S: RealScalar>(
    x: &[S],
    y: &[S],
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    t_test_independent_with_options(x, y, TTestVariance::Welch, Default::default()).map_err(
        |error| {
            crate::error_helpers::one_shot_argument(
                error,
                &["options", "variance"],
                scioxide_core::argument!("x"),
            )
        },
    )
}
/// Performs the two-sample t-test of the difference of means `x̄ − ȳ`.
///
/// ```text
/// Welch:   t = (x̄ − ȳ) / √(s²ₓ/n + s²ᵧ/m),
///          ν = (s²ₓ/n + s²ᵧ/m)² / ((s²ₓ/n)²/(n − 1) + (s²ᵧ/m)²/(m − 1))
/// Pooled:  t = (x̄ − ȳ) / (sₚ √(1/n + 1/m)),
///          sₚ² = ((n − 1)s²ₓ + (m − 1)s²ᵧ) / (n + m − 2),   ν = n + m − 2
/// ```
///
/// where `x̄`, `ȳ` are the sample means, `s²ₓ`, `s²ᵧ` the unbiased variances and
/// `n`, `m` the sizes. Alternatives refer to `x̄ − ȳ`. Prefer Welch's test unless
/// equal variances are justified. The cost is O(n + m) time and O(1) memory.
///
/// # Errors
///
/// - `InvalidInput` if either sample has fewer than two observations.
/// - `InvalidInput` if `variance` is [`TTestVariance::Welch`] and
///   `options.calibration()` is [`TestCalibration::Exact`]: Welch's test is
///   approximate.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
/// - `NumericalInstability` (`Breakdown`) if the standard error is zero.
///
/// # References
///
/// B. L. Welch, "The generalization of 'Student's' problem when several
/// different population variances are involved", *Biometrika* 34 (1947), 28–35.
pub fn t_test_independent_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    variance: TTestVariance,
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    independent(
        moments(x, scioxide_core::argument!("x"))?,
        moments(y, scioxide_core::argument!("y"))?,
        [x.len(), y.len()],
        variance,
        options,
        TestSamplePolicy::IndependentGroups,
    )
}

/// Computes the result of [`t_test_from_summaries_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`t_test_from_summaries_with_options`] for the fixed default settings.
pub fn t_test_from_summaries<S: RealScalar>(
    a: TTestSummary<S>,
    b: TTestSummary<S>,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    t_test_from_summaries_with_options(a, b, TTestVariance::Welch, Default::default()).map_err(
        |error| {
            crate::error_helpers::one_shot_argument(
                error,
                &["options", "variance"],
                scioxide_core::argument!("a"),
            )
        },
    )
}
/// Performs the two-sample t-test from summary statistics.
///
/// The test is the same as [`t_test_independent_with_options`], with each sample given by
/// its mean, unbiased variance and size.
///
/// # Errors
///
/// - `InvalidInput` if a count is below two or a variance is negative.
/// - An error if a mean or variance is NaN or infinite.
/// - `InvalidInput` for Welch's test with [`TestCalibration::Exact`].
/// - `NumericalInstability` (`Breakdown`) if the standard error is zero.
pub fn t_test_from_summaries_with_options<S: RealScalar>(
    a: TTestSummary<S>,
    b: TTestSummary<S>,
    variance: TTestVariance,
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    let validate = |s: TTestSummary<S>| {
        finite(s.mean)?;
        finite(s.variance)?;
        if s.count < 2 || s.variance < S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("a"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid sample summary",
            ));
        }
        Ok((
            Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(s.mean))?,
            Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(s.variance))?,
        ))
    };
    independent(
        validate(a)?,
        validate(b)?,
        [a.count, b.count],
        variance,
        options,
        TestSamplePolicy::Summary,
    )
}
