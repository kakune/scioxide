//! Exact binomial and quantile tests, and the Poisson E-test.
use super::*;
use scioxide_core::{CompensatedSum, Scaled};
use scioxide_random::{Binomial, InversionOptions, LogMass, Mass, Poisson, Quantile, Survival};
/// Performs the exact binomial test of a success probability.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`binomial_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`binomial_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let actual = binomial_test::<f64>(3, 8, 0.5)?;
/// let configured = binomial_test_with_options::<f64>(3, 8, 0.5, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// C. J. Clopper and E. S. Pearson, "The use of confidence or fiducial limits
/// illustrated in the case of the binomial", *Biometrika* 26 (1934), 404–413.
pub fn binomial_test<S: RealScalar>(
    successes: usize,
    trials: usize,
    null_probability: S,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    binomial_test_with_options(successes, trials, null_probability, Default::default()).map_err(
        |error| {
            crate::error_helpers::one_shot_argument(
                error,
                &["options"],
                scioxide_core::argument!("successes"),
            )
        },
    )
}
/// Performs the exact binomial test of a success probability.
///
/// The statistic and effect are the observed proportion `k/n`, with
/// `k = successes` and `n = trials`. One-sided p-values are binomial tail
/// probabilities. The two-sided p-value sums the probabilities of all outcomes
/// no more likely than `k` under the null, with a relative tolerance of
/// `64 ε` on the comparison. The result includes the Clopper–Pearson interval.
/// The p-value takes O(log n) probability evaluations and O(1) memory.
///
/// # Errors
///
/// - `InvalidInput` if `trials` is zero or `successes > trials`.
/// - `InvalidInput` if the calibration is [`TestCalibration::Asymptotic`].
/// - An error if `null_probability` is not a finite value in `[0, 1]`.
///
/// # References
///
/// C. J. Clopper and E. S. Pearson, "The use of confidence or fiducial limits
/// illustrated in the case of the binomial", *Biometrika* 26 (1934), 404–413.
pub fn binomial_test_with_options<S: RealScalar>(
    successes: usize,
    trials: usize,
    null_probability: S,
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    if trials == 0 || successes > trials {
        return Err(invalid(
            scioxide_core::argument!("successes"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid binomial counts",
        ));
    }
    if options.calibration == TestCalibration::Asymptotic {
        return Err(invalid(
            scioxide_core::argument!("options.calibration"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "binomial test is exact",
        ));
    }
    let p = finite(null_probability)?.into_f64();
    probability(p)?;
    let law = Binomial::new(trials, p)?;
    let upper = |k| if k == 0 { Ok(1.0) } else { law.sf(k - 1) };
    let p_value = match options.alternative {
        TestAlternative::Less => law.cdf(successes)?,
        TestAlternative::Greater => upper(successes)?,
        TestAlternative::TwoSided => {
            if p == 0.0 || p == 1.0 {
                law.pmf(successes)?
            } else {
                let observed = law.log_pmf(successes)?;
                // Compare in log space; retain symmetric probabilities within rounding noise.
                let threshold = observed + 64.0 * f64::EPSILON * (1.0 + observed.abs());
                let mode = ((trials as f64 + 1.0) * p).floor().min(trials as f64) as usize;
                if law.log_pmf(mode)? <= threshold {
                    1.0
                } else {
                    let mut lo = 0;
                    let mut hi = mode;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        if law.log_pmf(mid)? <= threshold {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    let left = if lo == 0 { 0.0 } else { law.cdf(lo - 1)? };
                    lo = mode;
                    hi = trials;
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        if law.log_pmf(mid)? > threshold {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    let right = if law.log_pmf(lo)? > threshold {
                        0.0
                    } else {
                        upper(lo)?
                    };
                    (left + right).min(1.0)
                }
            }
        }
    };
    let defaults = InversionOptions::<f64>::default();
    let root = defaults
        .root_options()
        .clone()
        .with_step_tolerance(scioxide_core::StepTolerance::new(0.0, 8.0 * f64::EPSILON)?);
    let limits = crate::binomial_confidence_limits_scoped(
        successes,
        trials,
        options.confidence,
        TestAlternative::TwoSided,
        &InversionOptions::default()
            .with_root_options(root)
            .with_work_budget(defaults.work_budget()),
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    let (lower, upper) = limits.set().bounds().ok_or_else(|| {
        invalid(
            scioxide_core::argument!("options.confidence"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "empty binomial confidence set",
        )
    })?;
    let lower = lower.ok_or_else(|| {
        invalid(
            scioxide_core::argument!("options.confidence"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "missing binomial lower bound",
        )
    })?;
    let upper = upper.ok_or_else(|| {
        invalid(
            scioxide_core::argument!("options.confidence"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "missing binomial upper bound",
        )
    })?;
    let estimate = successes as f64 / trials as f64;
    let mut out = result(
        estimate,
        p_value,
        ClassicalMethod::Exact,
        TestSamplePolicy::Counts,
        [trials, 0],
    )?;
    out.effect = Some((ClassicalEffect::Proportion, checked(estimate)?));
    out.interval = Some((
        Interval::new(checked(lower)?, checked(upper)?)?,
        options.confidence,
    ));
    Ok(out)
}

/// Performs the exact quantile test that the population `p`-quantile equals
/// `null_quantile`.
///
/// `p = quantile_probability`. `T₁` counts the observations below
/// `null_quantile` and `T₂` those at or below it. Under the null both are
/// compared with the binomial distribution `B(n, p)`. The alternative `Less`
/// means the quantile is below `null_quantile` and uses `T₁`, `Greater` uses
/// `T₂`, and the two-sided test uses whichever gives the smaller tail. The
/// effect is the order statistic `x₍⌈np⌉₎`. The confidence interval has order
/// statistics as endpoints, and is absent when no pair of order statistics
/// reaches the requested coverage.
///
/// Sorts an owned copy of the `n` observations in O(n log n) time and O(n) space.
///
/// # Errors
///
/// - `InvalidInput` if `x` is empty or `quantile_probability` is not in `(0, 1)`.
/// - `InvalidInput` if the calibration is [`TestCalibration::Asymptotic`].
/// - `NonFinite` if `x` contains NaN or infinity. A non-finite
///   `null_quantile` is also rejected.
///
/// # References
///
/// W. J. Conover, *Practical Nonparametric Statistics*, 3rd ed., Wiley, 1999.
pub fn quantile_test_with_options<S: RealScalar>(
    x: &[S],
    null_quantile: S,
    quantile_probability: f64,
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    sample(x, 1, scioxide_core::argument!("x"))?;
    finite(null_quantile)?;
    if !(0.0..1.0).contains(&quantile_probability) || quantile_probability == 0.0 {
        return Err(invalid(
            scioxide_core::argument!("quantile_probability"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "quantile probability must be inside (0,1)",
        ));
    }
    if options.calibration == TestCalibration::Asymptotic {
        return Err(invalid(
            scioxide_core::argument!("options.calibration"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "exact quantile test does not support asymptotic calibration",
        ));
    }
    let n = x.len();
    let strict = x.iter().filter(|&&v| v < null_quantile).count();
    let inclusive = x.iter().filter(|&&v| v <= null_quantile).count();
    let law = Binomial::new(n, quantile_probability)?;
    let less = if strict == 0 {
        1.0
    } else {
        law.sf(strict - 1)?
    };
    let greater = law.cdf(inclusive)?;
    let statistic = match options.alternative {
        TestAlternative::Less => strict,
        TestAlternative::Greater => inclusive,
        TestAlternative::TwoSided => {
            if less < greater {
                strict
            } else {
                inclusive
            }
        }
    };
    let mut out = result(
        statistic as f64,
        tails(less, greater, options.alternative)?,
        ClassicalMethod::Exact,
        TestSamplePolicy::Independent,
        [n, 0],
    )?;
    let scratch = crate::distribution_tests::sorted(x, options.max_workspace_bytes())?;
    let index = ((n as f64 * quantile_probability).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    out.effect = Some((ClassicalEffect::Quantile, scratch[index]));
    let alpha = (1.0 - options.confidence.get()) / 2.0;
    let lower = law.quantile(alpha)?;
    let upper = law.quantile(1.0 - alpha)?;
    if lower > 0 && upper < n {
        out.interval = Some((
            Interval::new(scratch[lower - 1], scratch[upper])?,
            options.confidence,
        ));
    }
    Ok(out)
}
/// Compares two Poisson rates with the Krishnamoorthy–Thomson E-test.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`poisson_means_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`poisson_means_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let actual = poisson_means_test::<f64>([3, 8], [10., 20.], 0.0)?;
/// let configured = poisson_means_test_with_options::<f64>([3, 8], [10., 20.], 0.0, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// K. Krishnamoorthy and J. Thomson, "A more powerful test for comparing two
/// Poisson means", *J. Statist. Plann. Inference* 119 (2004), 23–35.
pub fn poisson_means_test<S: RealScalar>(
    counts: [usize; 2],
    exposures: [S; 2],
    difference: S,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    poisson_means_test_with_options(counts, exposures, difference, Default::default()).map_err(
        |error| {
            crate::error_helpers::one_shot_argument(
                error,
                &["options"],
                scioxide_core::argument!("counts"),
            )
        },
    )
}
/// Compares two Poisson rates with the Krishnamoorthy–Thomson E-test.
///
/// `counts` are the event counts `k₁`, `k₂` observed over `exposures` `t₁`,
/// `t₂`. The null hypothesis is `λ₁ − λ₂ = d` with `d = difference`. The
/// statistic is the standardized rate difference. The p-value sums the joint
/// Poisson probabilities of all outcomes at least as extreme, with both rates
/// estimated under the null. The test is therefore not exact. The effect is
/// `k₁/t₁ − k₂/t₂`. If the estimated second rate is not positive, the statistic
/// is 0 and the p-value 1.
///
/// The sum covers each marginal between its `10⁻¹³` and `1 − 10⁻¹³` quantiles.
/// [`ClassicalTestResult::omitted_null_mass`] reports the probability outside
/// this region, about `4 × 10⁻¹³` at most. The number of joint probabilities
/// evaluated is the size of this region, which must not exceed
/// [`ClassicalOptions::max_work`].
///
/// # Errors
///
/// - `InvalidInput` if an exposure is not positive or the calibration is
///   [`TestCalibration::Exact`].
/// - `ConvergenceFailure` if the summation region exceeds `max_work`.
/// - An error if `exposures` or `difference` is NaN or infinite.
///
/// # Examples
///
/// ```
/// use scioxide_stats::poisson_means_test_with_options;
///
/// // 0 and 3 events over equal exposures; reference p-value of the E-test.
/// let test = poisson_means_test_with_options([0, 3], [100.0_f64; 2], 0.0, Default::default())?;
/// assert!((test.p_value() - 0.08837900945483522).abs() < 4e-13);
/// assert!(test.omitted_null_mass() < 4e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// K. Krishnamoorthy and J. Thomson, "A more powerful test for comparing two
/// Poisson means", *J. Statist. Plann. Inference* 119 (2004), 23–35.
pub fn poisson_means_test_with_options<S: RealScalar>(
    counts: [usize; 2],
    exposures: [S; 2],
    difference: S,
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    let [a, b] = [
        finite(exposures[0])?.into_f64(),
        finite(exposures[1])?.into_f64(),
    ];
    let d = finite(difference)?.into_f64();
    if a <= 0.0 || b <= 0.0 || options.calibration == TestCalibration::Exact {
        return Err(invalid(
            scioxide_core::argument!("exposures"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "E-test requires positive exposures and approximate calibration",
        ));
    }
    // Reflect the problem for negative contrasts, preserving the paper's d>=0 domain.
    if d < 0.0 {
        let alternative = match options.alternative {
            TestAlternative::Less => TestAlternative::Greater,
            TestAlternative::Greater => TestAlternative::Less,
            x => x,
        };
        let mut out = poisson_means_test_with_options(
            [counts[1], counts[0]],
            [exposures[1], exposures[0]],
            -difference,
            options.with_alternative(alternative),
        )?;
        out.statistic = -out.statistic;
        out.effect = out.effect.map(|(kind, effect)| (kind, -effect));
        out.retained = counts;
        return Ok(out);
    }
    let effect = Scaled::<f64>::from_value((counts[0]) as f64)?
        .div(Scaled::<f64>::from_value(a)?)?
        .sub(Scaled::<f64>::from_value((counts[1]) as f64)?.div(Scaled::<f64>::from_value(b)?)?)?;
    let total = Scaled::<f64>::from_value(a)?.add(Scaled::<f64>::from_value(b)?)?;
    let rate2 = Scaled::<f64>::from_value((counts[0]) as f64)?
        .add(Scaled::<f64>::from_value((counts[1]) as f64)?)?
        .sub(Scaled::<f64>::from_value(d)?.mul(Scaled::<f64>::from_value(a)?)?)?
        .div(total)?
        .round_to::<f64>()?;
    let pivot = PoissonPivot::new(a, b, d)?;
    let observed = if rate2 <= 0.0 {
        0.0
    } else {
        pivot.evaluate(
            pivot.axes[0].term(counts[0])?,
            pivot.axes[1].term(counts[1])?,
        )?
    };
    let mut out = result(
        observed,
        1.0,
        ClassicalMethod::PoissonE,
        TestSamplePolicy::Counts,
        counts,
    )?;
    out.effect = Some((ClassicalEffect::RateDifference, effect.round_to()?));
    if rate2 <= 0.0 {
        return Ok(out);
    }
    let p1 = Poisson::new(finite(a * (rate2 + d))?)?;
    let p2 = Poisson::new(finite(b * rate2)?)?;
    let bounds = |law: &Poisson<f64>| -> Result<(usize, usize, f64), SciOxideError> {
        let low = law.quantile(1e-13)?;
        let high = law.quantile(1.0 - 1e-13)?;
        Ok((
            low,
            high,
            (if low == 0 { 0.0 } else { law.cdf(low - 1)? }) + law.sf(high)?,
        ))
    };
    let (l1, h1, e1) = bounds(&p1)?;
    let (l2, h2, e2) = bounds(&p2)?;
    let width = |low: usize, high: usize| high.checked_sub(low).and_then(|n| n.checked_add(1));
    let widths = width(l1, h1).zip(width(l2, h2));
    let work = widths
        .and_then(|(n, m)| n.checked_mul(m))
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    crate::error_helpers::work(
        work,
        options.max_work,
        scioxide_core::BudgetKind::Evaluation,
        "Poisson joint support exceeds options.max_work mass evaluations",
    )?;
    let tail = PoissonTail::new(observed, options.alternative);
    const TILE: usize = 128;
    let mut tile = [(PoissonTerm::ZERO, 0.0); TILE];
    let mut sum = CompensatedSum::<f64>::ZERO;
    // Each second marginal mass/term is prepared once. First-marginal preparation
    // is repeated only between tiles, never for individual joint-support pairs.
    for start in (l2..=h2).step_by(TILE) {
        let len = (h2 - start).min(TILE - 1) + 1;
        for (offset, entry) in tile[..len].iter_mut().enumerate() {
            let j = start + offset;
            *entry = (pivot.axes[1].term(j)?, p2.pmf(j)?);
        }
        for i in l1..=h1 {
            let first = pivot.axes[0].term(i)?;
            let mass = p1.pmf(i)?;
            for &(second, second_mass) in &tile[..len] {
                if pivot.extreme(first, second, tail)? {
                    sum.add(mass * second_mass)?;
                }
            }
        }
    }
    out.p_value = probability(sum.total()?.min(1.0))?;
    out.omitted_null_mass = e1 + e2 - e1 * e2;
    Ok(out)
}

#[derive(Clone, Copy)]
struct PoissonTerm {
    rate: Scaled<f64>,
    variance: Scaled<f64>,
    native_rate: f64,
    native_variance: f64,
}
impl PoissonTerm {
    const ZERO: Self = Self {
        rate: Scaled::ZERO,
        variance: Scaled::ZERO,
        native_rate: 0.0,
        native_variance: 0.0,
    };
}

struct PoissonAxis {
    exposure: Scaled<f64>,
    squared_exposure: Scaled<f64>,
    coefficient: f64,
    squared_coefficient: f64,
}
impl PoissonAxis {
    fn term(&self, count: usize) -> Result<PoissonTerm, SciOxideError> {
        let count = count as f64;
        let scaled = Scaled::from_value(count)?;
        Ok(PoissonTerm {
            rate: scaled.div(self.exposure)?,
            variance: scaled.div(self.squared_exposure)?,
            native_rate: count * self.coefficient,
            native_variance: count * self.squared_coefficient,
        })
    }
}

#[derive(Clone, Copy)]
struct PoissonTail {
    alternative: TestAlternative,
    boundary: f64,
}
impl PoissonTail {
    fn new(observed: f64, alternative: TestAlternative) -> Self {
        let threshold = 64.0 * f64::EPSILON * (1.0 + observed.abs());
        Self {
            alternative,
            boundary: match alternative {
                TestAlternative::Less => -(observed + threshold),
                TestAlternative::Greater => observed - threshold,
                TestAlternative::TwoSided => observed.abs() - threshold,
            },
        }
    }
    fn orient(self, pivot: f64) -> f64 {
        match self.alternative {
            TestAlternative::Less => -pivot,
            TestAlternative::Greater => pivot,
            TestAlternative::TwoSided => pivot.abs(),
        }
    }
}

struct PoissonPivot {
    axes: [PoissonAxis; 2],
    difference: Scaled<f64>,
    native_difference: Option<f64>,
}
impl PoissonPivot {
    fn new(a: f64, b: f64, difference: f64) -> Result<Self, SciOxideError> {
        let scale = a.min(b);
        let axis = |exposure| -> Result<_, SciOxideError> {
            let coefficient = scale / exposure;
            let exposure = Scaled::from_value(exposure)?;
            Ok(PoissonAxis {
                exposure,
                squared_exposure: exposure.square()?,
                coefficient,
                squared_coefficient: coefficient * coefficient,
            })
        };
        let axes = [axis(a)?, axis(b)?];
        let difference = Scaled::from_value(difference)?;
        // Multiply numerator/denominator by min(a,b). Every count contribution
        // is bounded by usize::MAX, even when a or b cannot be squared in IEEE.
        // Normal nonzero coefficients/shift exclude gradual-underflow error from
        // the relative rounding bound below; unsupported ranges use Scaled.
        let native_difference = difference
            .mul(Scaled::from_value(scale)?)?
            .round_to::<f64>()
            .ok()
            .filter(|&shift| {
                (shift.is_normal() || difference.is_zero())
                    && axes.iter().all(|axis| axis.squared_coefficient.is_normal())
            });
        Ok(Self {
            axes,
            difference,
            native_difference,
        })
    }

    #[cold]
    fn evaluate(&self, first: PoissonTerm, second: PoissonTerm) -> Result<f64, SciOxideError> {
        let variance = first.variance.add(second.variance)?;
        let delta = first.rate.sub(second.rate)?.sub(self.difference)?;
        if variance.is_zero() {
            Ok(if delta.is_zero() {
                0.0
            } else {
                f64::NEG_INFINITY
            })
        } else {
            delta.div(variance.sqrt()?)?.round_to()
        }
    }

    // Keep the native comparison in the joint loop: passing the cached scaled
    // records through an out-of-line call otherwise dominates ordinary inputs.
    #[inline(always)]
    fn extreme(
        &self,
        first: PoissonTerm,
        second: PoissonTerm,
        tail: PoissonTail,
    ) -> Result<bool, SciOxideError> {
        if let Some(shift) = self.native_difference {
            let variance = first.native_variance + second.native_variance;
            if variance > 0.0 {
                let denominator = variance.sqrt();
                let delta = first.native_rate - second.native_rate - shift;
                // Compare before dividing by the positive standard deviation.
                let boundary = tail.boundary * denominator;
                // Covers coefficient rounding, count products, subtraction,
                // variance/sqrt, threshold multiplication and Scaled projection.
                // Cancellation is bounded by the absolute operand sum, not |delta|.
                let error = 64.0
                    * f64::EPSILON
                    * (first.native_rate + second.native_rate + shift.abs() + boundary.abs());
                let oriented = tail.orient(delta);
                if (oriented - boundary).abs() > error {
                    return Ok(oriented >= boundary);
                }
            }
        }
        Ok(tail.orient(self.evaluate(first, second)?) >= tail.boundary)
    }
}
/// Performs the exact quantile test that the population `p`-quantile equals `null_quantile`.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`quantile_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`quantile_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let actual = quantile_test::<f64>(&x, 3., 0.5)?;
/// let configured = quantile_test_with_options::<f64>(&x, 3., 0.5, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// W. J. Conover, *Practical Nonparametric Statistics*, 3rd ed., Wiley, 1999.
pub fn quantile_test<S: RealScalar>(
    x: &[S],
    null_quantile: S,
    quantile_probability: f64,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    quantile_test_with_options(x, null_quantile, quantile_probability, Default::default()).map_err(
        |error| {
            crate::error_helpers::one_shot_argument(
                error,
                &[
                    "options",
                    "weights",
                    "left_weights",
                    "right_weights",
                    "direction",
                    "ties",
                ],
                scioxide_core::argument!("x"),
            )
        },
    )
}
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod poisson_tests {
    use super::*;

    #[test]
    fn guarded_comparisons_agree_with_scaled_pivots() {
        for (a, b, d) in [
            (10., 20., 0.1),
            (1., 1., 6.),
            (1e-200, 2e-200, 1e199),
            (1e200, 2e200, 1e-201),
            (1e150, 1e-150, 0.),
            (1e-200, 1e-200, 1e-200),
        ] {
            let pivot = PoissonPivot::new(a, b, d).unwrap();
            for counts in [[0, 3], [3, 8], [7, 1], [3, 3]] {
                let observed = pivot
                    .evaluate(
                        pivot.axes[0].term(counts[0]).unwrap(),
                        pivot.axes[1].term(counts[1]).unwrap(),
                    )
                    .unwrap();
                for alternative in [
                    TestAlternative::TwoSided,
                    TestAlternative::Less,
                    TestAlternative::Greater,
                ] {
                    let tail = PoissonTail::new(observed, alternative);
                    for i in 0..12 {
                        let first = pivot.axes[0].term(i).unwrap();
                        for j in 0..12 {
                            let second = pivot.axes[1].term(j).unwrap();
                            let expected = tail.orient(pivot.evaluate(first, second).unwrap())
                                >= tail.boundary;
                            assert_eq!(
                                pivot.extreme(first, second, tail).unwrap(),
                                expected,
                                "{a} {b} {d} {counts:?} {alternative:?} {i} {j}"
                            );
                        }
                    }
                }
            }
        }
    }
}
