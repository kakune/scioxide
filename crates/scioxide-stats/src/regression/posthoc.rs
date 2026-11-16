//! Multiple comparisons of group means: Tukey–Kramer, Games–Howell and Dunnett.
use super::{PreparedGroups, TestAlternative, budget_bytes, finite, invalid, number};
use crate::ConfidenceProbability;
use alloc::vec::Vec;

use crate::arithmetic::center_difference;
use scioxide_core::{Budget, RealScalar, Scaled, SciOxideError, Tolerance};

/// Selects a multiple-comparison procedure that controls the familywise error
/// rate.
///
/// # References
///
/// - J. C. Hsu, *Multiple Comparisons: Theory and Methods*, Chapman & Hall,
///   1996.
/// - C. W. Dunnett, "A multiple comparison procedure for comparing several
///   treatments with a control", *J. Amer. Statist. Assoc.* 50, 1096–1121,
///   1955.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostHocMethod {
    /// All pairs, assuming equal variances. Uses the studentized range
    /// distribution with the pooled variance, and allows unequal group sizes.
    TukeyKramer,
    /// All pairs, without assuming equal variances. Uses the studentized range
    /// distribution with Welch–Satterthwaite degrees of freedom for each pair.
    GamesHowell,
    /// Each group against one control, assuming equal variances. Uses the
    /// multivariate t distribution of the single-step procedure.
    Dunnett,
}
/// Options of the multiple-comparison procedures: the probability tolerance,
/// an evaluation limit, the confidence level and a byte limit.
///
/// Defaults to absolute and relative tolerances of `1e-7`, 40 000 000
/// integrand evaluations, 95% confidence and no byte limit. The budget admits
/// 28 conditional CDF evaluations at the full 1 398 016-point refinement.
///
/// # References
///
/// J. C. Hsu, Multiple Comparisons: Theory and Methods (1996).
#[derive(Clone, Copy, Debug)]
pub struct PostHocOptions {
    /// Absolute tolerance of the probabilities. The relative part bounds the
    /// critical-value bracket.
    tolerance: Option<Tolerance<f64>>,
    /// Evaluation limit for the whole family.
    evaluations: Budget,
    /// Confidence level of the simultaneous intervals.
    confidence: ConfidenceProbability,
    max_bytes: usize,
}

impl Default for PostHocOptions {
    fn default() -> Self {
        Self {
            tolerance: None,
            evaluations: Budget::ONE.saturating_add(39_999_999),
            confidence: Default::default(),
            max_bytes: usize::MAX,
        }
    }
}
impl PostHocOptions {
    fn resolve(self) -> Result<ResolvedPostHocOptions, SciOxideError> {
        Ok(ResolvedPostHocOptions {
            tolerance: self
                .tolerance
                .map_or_else(|| Tolerance::new(1e-7, 1e-7), Ok)?,
            evaluations: self.evaluations,
            confidence: self.confidence,
            max_bytes: self.max_bytes,
        })
    }
    /// Returns the explicit probability tolerance; `None` selects `(1e-7, 1e-7)`.
    ///
    /// Its absolute part bounds the error of each computed probability and must
    /// lie in `[1e-12, 1e-3]`. Its relative part sets the width at which the
    /// critical-value search stops.
    pub const fn tolerance(&self) -> Option<Tolerance<f64>> {
        self.tolerance
    }
    /// Sets the probability tolerance.
    pub const fn with_tolerance(mut self, value: Tolerance<f64>) -> Self {
        self.tolerance = Some(value);
        self
    }
    /// Returns the evaluation limit.
    ///
    /// The limit counts integrand evaluations over all comparisons and
    /// critical-value searches of one call.
    pub const fn evaluations(&self) -> Budget {
        self.evaluations
    }
    /// Sets the evaluation limit.
    pub const fn with_evaluations(mut self, value: Budget) -> Self {
        self.evaluations = value;
        self
    }
    /// Returns the confidence level of the simultaneous intervals. The default
    /// is 0.95.
    pub const fn confidence(&self) -> ConfidenceProbability {
        self.confidence
    }
    /// Sets the confidence level.
    pub const fn with_confidence(mut self, value: ConfidenceProbability) -> Self {
        self.confidence = value;
        self
    }
    /// Returns the byte limit of the result. The default is `usize::MAX`.
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }
    /// Sets the byte limit.
    pub const fn with_max_bytes(mut self, value: usize) -> Self {
        self.max_bytes = value;
        self
    }
}
#[derive(Clone, Copy)]
struct ResolvedPostHocOptions {
    tolerance: Tolerance<f64>,
    evaluations: Budget,
    confidence: ConfidenceProbability,
    max_bytes: usize,
}
/// Holds one comparison of two group means with its adjusted p-value and
/// simultaneous confidence interval.
///
/// Group indices refer to the order of the groups in [`crate::PreparedGroups`].
/// The fields are read-only.
///
/// ```compile_fail
/// fn corrupt(mut c: scioxide_stats::PostHocComparison<f64>) { c.lower = None; }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct PostHocComparison<S: RealScalar> {
    first: usize,
    second: usize,
    difference: S,
    p_value: S,
    probability_error: S,
    lower: Option<S>,
    upper: Option<S>,
    degrees_of_freedom: f64,
}
impl<S: RealScalar> PostHocComparison<S> {
    /// Returns the index of the first group, the treatment for Dunnett's
    /// procedure.
    pub const fn first(&self) -> usize {
        self.first
    }
    /// Returns the index of the second group, the control for Dunnett's
    /// procedure.
    pub const fn second(&self) -> usize {
        self.second
    }
    /// Returns the mean of the first group minus the mean of the second.
    pub const fn difference(&self) -> S {
        self.difference
    }
    /// Returns the p-value adjusted for the whole family of comparisons.
    pub const fn p_value(&self) -> S {
        self.p_value
    }
    /// Returns an estimate of the absolute error of [`Self::p_value`]. It is
    /// not a bound.
    pub const fn probability_error(&self) -> S {
        self.probability_error
    }
    /// Returns the lower endpoint of the simultaneous interval for the
    /// difference, or `None` for the alternative `Less`.
    pub const fn lower(&self) -> Option<S> {
        self.lower
    }
    /// Returns the upper endpoint of the simultaneous interval for the
    /// difference, or `None` for the alternative `Greater`.
    pub const fn upper(&self) -> Option<S> {
        self.upper
    }
    /// Returns the degrees of freedom of the reference distribution.
    pub const fn degrees_of_freedom(&self) -> f64 {
        self.degrees_of_freedom
    }
}
/// Holds a family of comparisons and the procedure that produced it.
///
/// The fields are read-only.
///
/// ```compile_fail
/// fn corrupt(mut family: scioxide_stats::PostHocResult<f64>) { family.comparisons.clear(); }
/// ```
pub struct PostHocResult<S: RealScalar> {
    comparisons: Vec<PostHocComparison<S>>,
    method: PostHocMethod,
    alternative: TestAlternative,
    confidence: ConfidenceProbability,
    evaluations: usize,
}
impl<S: RealScalar> PostHocResult<S> {
    /// Returns the comparisons.
    ///
    /// For Tukey–Kramer and Games–Howell these are the pairs `(i, j)` with
    /// `i < j` in lexicographic order. For Dunnett's procedure they are the
    /// non-control groups in index order.
    pub fn comparisons(&self) -> &[PostHocComparison<S>] {
        &self.comparisons
    }
    /// Returns the procedure.
    pub const fn method(&self) -> PostHocMethod {
        self.method
    }
    /// Returns the alternative hypothesis.
    pub const fn alternative(&self) -> TestAlternative {
        self.alternative
    }
    /// Returns the confidence level of the intervals.
    pub const fn confidence(&self) -> ConfidenceProbability {
        self.confidence
    }
    /// Returns the number of evaluations used, counted against
    /// [`PostHocOptions::evaluations`].
    pub const fn evaluations(&self) -> usize {
        self.evaluations
    }
}

// Sixteen-point Gauss–Legendre rule. Nodes/weights from the defining Legendre
// polynomial; tensor refinement is local to the Dunnett control probability law.
const NODES: [f64; 8] = [
    0.09501250983763744,
    0.2816035507792589,
    0.4580167776572274,
    0.6178762444026438,
    0.755404408355003,
    0.8656312023878318,
    0.9445750230732326,
    0.9894009349916499,
];
const WEIGHTS: [f64; 8] = [
    0.1894506104550685,
    0.1826034150449236,
    0.16915651939500254,
    0.14959598881657673,
    0.12462897125553387,
    0.09515851168249278,
    0.06225352393864789,
    0.027152459411754095,
];
fn normal_cdf(x: f64) -> Result<f64, SciOxideError> {
    Ok(scioxide_special::erfc(-x / core::f64::consts::SQRT_2)? / 2.0)
}
fn normal_interval(low: f64, high: f64) -> Result<f64, SciOxideError> {
    if high < low {
        return Ok(0.0);
    }
    if low > 0.0 {
        Ok((scioxide_special::erfc(low / core::f64::consts::SQRT_2)?
            - scioxide_special::erfc(high / core::f64::consts::SQRT_2)?)
            / 2.0)
    } else {
        Ok(normal_cdf(high)? - normal_cdf(low)?)
    }
}
#[derive(Clone, Copy)]
struct ControlFamily<'a> {
    groups: &'a [crate::group_summary::GroupSummary],
    control: usize,
    two_sided: bool,
}
struct ControlProbabilityKernel<'a> {
    family: ControlFamily<'a>,
    degrees: f64,
    low: f64,
    high: f64,
    log_normalizer: f64,
    tail_error: f64,
}
impl<'a> ControlProbabilityKernel<'a> {
    fn new(family: ControlFamily<'a>, degrees: f64, tolerance: f64) -> Result<Self, SciOxideError> {
        if degrees <= 0.0 || !degrees.is_finite() || !(1e-12..=1e-3).contains(&tolerance) {
            return Err(invalid(
                scioxide_core::argument!("options.tolerance"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "post-hoc CDF tolerance must lie in [1e-12,1e-3], with positive degrees",
            ));
        }
        let tail = tolerance / 64.0;
        let shape = degrees / 2.0;
        let low = scioxide_special::gamma_p_inv(shape, tail)?;
        let high = scioxide_special::gamma_p_inv(shape, 1.0 - tail)?;
        Ok(Self {
            family,
            degrees,
            low: 0.5 * (low / shape).ln(),
            high: 0.5 * (high / shape).ln(),
            log_normalizer: core::f64::consts::LN_2 + shape * shape.ln()
                - scioxide_special::ln_gamma(shape)?,
            tail_error: 2.0 * tail + 3e-18,
        })
    }
    fn conditional(&self, threshold: f64, scale: f64, z: f64) -> Result<f64, SciOxideError> {
        let ControlFamily {
            groups,
            control,
            two_sided,
        } = self.family;
        let nc = groups[control].n;
        let mut probability = 1.0;
        for (i, g) in groups.iter().enumerate() {
            if i == control {
                continue;
            }
            let a = (g.n / (g.n + nc)).sqrt();
            let b = (nc / (g.n + nc)).sqrt();
            let high = (threshold * scale + a * z) / b;
            probability *= if two_sided {
                normal_interval((-threshold * scale + a * z) / b, high)?
            } else {
                normal_cdf(high)?
            };
        }
        Ok(probability)
    }

    fn cdf(
        &self,
        threshold: f64,
        options: ResolvedPostHocOptions,
        used: &mut usize,
    ) -> Result<(f64, f64), SciOxideError> {
        if threshold < 0.0 && self.family.two_sided {
            return Ok((0.0, 0.0));
        }
        let mut previous: Option<f64> = None;
        for panels in [1usize, 2, 4, 8, 16, 32, 64] {
            let cost = 256 * panels * panels;
            if used
                .checked_add(cost)
                .is_none_or(|n| n > options.evaluations.get())
            {
                return Err(SciOxideError::convergence_failure(
                    scioxide_core::ConvergenceReason::IterationLimit,
                    Default::default(),
                    "conditional evaluation budget exhausted",
                ));
            }
            *used += cost;
            let dr = (self.high - self.low) / (panels as f64);
            let dz = 18.0 / (panels as f64);
            let mut sum = 0.0;
            for rpanel in 0..panels {
                let center = self.low + (rpanel as f64 + 0.5) * dr;
                for (&node, &weight) in NODES.iter().zip(&WEIGHTS) {
                    for sign in [-1.0, 1.0] {
                        let r = center + sign * node * dr / 2.0;
                        let scale = r.exp();
                        let density = (self.log_normalizer + self.degrees * r
                            - self.degrees * scale * scale / 2.0)
                            .exp();
                        let mut conditional = 0.0;
                        for zpanel in 0..panels {
                            let zcenter = -9.0 + (zpanel as f64 + 0.5) * dz;
                            for (&zn, &zw) in NODES.iter().zip(&WEIGHTS) {
                                for zs in [-1.0, 1.0] {
                                    let z = zcenter + zs * zn * dz / 2.0;
                                    conditional += zw * (-z * z / 2.0).exp()
                                        / (2.0 * core::f64::consts::PI).sqrt()
                                        * self.conditional(threshold, scale, z)?;
                                }
                            }
                        }
                        sum += weight * density * conditional * dz / 2.0;
                    }
                }
            }
            let value = finite(sum * dr / 2.0)?;
            if let Some(old) = previous {
                let error = (value - old).abs() + self.tail_error;
                if error <= options.tolerance.absolute() {
                    return Ok((value.clamp(0.0, 1.0), error));
                }
            }
            previous = Some(value);
        }
        Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::IterationLimit,
            Default::default(),
            "quadrature refinement failed requested accuracy",
        ))
    }
    fn critical(
        &self,
        options: ResolvedPostHocOptions,
        used: &mut usize,
    ) -> Result<f64, SciOxideError> {
        let target = options.confidence.get();
        if target.min(1.0 - target) <= 8.0 * options.tolerance.absolute() {
            return Err(invalid(
                scioxide_core::argument!("options.confidence"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "confidence tails must exceed eight times the requested CDF tolerance",
            ));
        }
        let mut low = if !self.family.two_sided { -1.0 } else { 0.0 };
        let mut high = 2.0;
        while self.cdf(low, options, used)?.0 > target {
            low *= 2.0;
        }
        while self.cdf(high, options, used)?.0 < target {
            high *= 2.0;
            finite(high)?;
        }
        for _ in 0..80 {
            let mid = low + (high - low) / 2.0;
            let (p, error) = self.cdf(mid, options, used)?;
            if (p - target).abs() + error <= 4.0 * options.tolerance.absolute() {
                return Ok(mid);
            }
            if p < target {
                low = mid;
            } else {
                high = mid;
            }
            if high - low <= options.tolerance.threshold(mid.abs())? {
                return Ok(mid);
            }
        }
        Err(invalid(
            scioxide_core::argument!("options.tolerance"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "post-hoc critical value failed to converge",
        ))
    }
}

// The random crate owns the scalar law. This adapter only maintains the
// post-hoc solve's total work ledger and confidence/critical-value policy.
struct RangeKernel {
    k: usize,
    degrees: f64,
}
impl RangeKernel {
    fn tail(
        &self,
        x: f64,
        upper: bool,
        options: ResolvedPostHocOptions,
        used: &mut usize,
    ) -> Result<(f64, f64), SciOxideError> {
        if !(1e-12..=1e-3).contains(&options.tolerance.absolute()) {
            return Err(invalid(
                scioxide_core::argument!("options.tolerance"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "post-hoc CDF tolerance must lie in [1e-12,1e-3]",
            ));
        }
        let left = options
            .evaluations
            .get()
            .checked_sub(*used)
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                SciOxideError::convergence_failure(
                    scioxide_core::ConvergenceReason::IterationLimit,
                    Default::default(),
                    "conditional evaluation budget exhausted",
                )
            })?;
        let settings = scioxide_random::LawOptions::default()
            .with_work_budget(Budget::try_from(left)?)
            .with_tolerance(Tolerance::new(
                options.tolerance.absolute() / 8.0,
                options.tolerance.absolute() / 8.0,
            )?);
        let law =
            scioxide_random::StudentizedRange::with_options(self.k as f64, self.degrees, settings)?;
        let report = law.evaluate_tail(x, upper)?;
        *used += report.evaluations;
        Ok((report.log_probability.exp(), report.absolute_error))
    }
    fn critical(
        &self,
        options: ResolvedPostHocOptions,
        used: &mut usize,
    ) -> Result<f64, SciOxideError> {
        let target = 1.0 - options.confidence.get();
        if target.min(1.0 - target) <= 8.0 * options.tolerance.absolute() {
            return Err(invalid(
                scioxide_core::argument!("options.confidence"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "confidence tails must exceed eight times the requested CDF tolerance",
            ));
        }
        struct Provider<'a> {
            kernel: &'a RangeKernel,
            options: ResolvedPostHocOptions,
            used: core::cell::Cell<usize>,
        }
        impl scioxide_random::Support<f64> for Provider<'_> {
            fn support(&self) -> Result<scioxide_core::ExtendedInterval<f64>, SciOxideError> {
                scioxide_core::ExtendedInterval::new(Some(0.0), None)
            }
        }
        impl scioxide_random::LogCdf<f64> for Provider<'_> {
            fn log_cdf(&self, x: f64) -> Result<f64, SciOxideError> {
                let mut used = self.used.get();
                let p = self.kernel.tail(x, false, self.options, &mut used)?;
                self.used.set(used);
                Ok(p.0.ln())
            }
        }
        impl scioxide_random::LogSurvival<f64> for Provider<'_> {
            fn log_sf(&self, x: f64) -> Result<f64, SciOxideError> {
                let mut used = self.used.get();
                let p = self.kernel.tail(x, true, self.options, &mut used)?;
                self.used.set(used);
                Ok(p.0.ln())
            }
        }
        let provider = Provider {
            kernel: self,
            options,
            used: core::cell::Cell::new(*used),
        };
        let inversion = scioxide_random::InversionOptions::default();
        let root = inversion.root_options().clone().with_step_tolerance(
            scioxide_core::StepTolerance::new(
                options.tolerance.absolute(),
                options.tolerance.relative(),
            )?,
        );
        let result = scioxide_random::invert_log_probability_with_options(
            &provider,
            target.ln(),
            true,
            &scioxide_random::InversionOptions::default()
                .with_root_options(root)
                .with_work_budget(Budget::try_from(256)?),
        )?;
        *used = provider.used.get();
        if !result.root().is_success() {
            return Err(invalid(
                scioxide_core::argument!("options.tolerance"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "post-hoc critical value failed to converge",
            ));
        }
        Ok(result.value())
    }
}
impl<S: RealScalar> PreparedGroups<S> {
    /// Compares all pairs of group means by the Tukey–Kramer or Games–Howell
    /// procedure.
    ///
    /// For groups `i < j` with means `ȳᵢ`, sizes `nᵢ` and sample variances
    /// `sᵢ²`, the difference is `d = ȳᵢ − ȳⱼ` and its scale is
    ///
    /// ```text
    /// Tukey–Kramer:  σ = √(s² (1/nᵢ + 1/nⱼ) / 2),   ν = N − k,
    /// Games–Howell:  σ = √((sᵢ²/nᵢ + sⱼ²/nⱼ) / 2),  ν = Welch–Satterthwaite,
    /// ```
    ///
    /// where `s²` is the pooled within-group variance, `N` the total size and
    /// `k` the number of groups. The p-value is `P(Q > |d| / σ)` for the
    /// studentized range `Q` with `k` means and `ν` degrees of freedom. The
    /// interval is `d ± q σ`, where `q` is the quantile of `Q` at the
    /// confidence level. All comparisons are two-sided. The difference and
    /// scale are computed in an extended exponent range, so they stay finite
    /// when a variance overflows `S`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `method` is [`PostHocMethod::Dunnett`]. Use
    ///   [`Self::compare_control`].
    /// - `InvalidInput` for Tukey–Kramer if `N − k ≤ 0` or the pooled variance
    ///   is zero.
    /// - `InvalidInput` for Games–Howell if a group has fewer than two
    ///   observations or zero variance.
    /// - `InvalidInput` if the absolute tolerance is outside `[1e-12, 1e-3]`,
    ///   or `min(α, 1 − α)` is at most 8 times the absolute tolerance, where
    ///   `α = 1 − confidence`.
    /// - `InvalidInput` if the critical-value search does not converge.
    /// - `ConvergenceFailure` if the evaluation limit is exhausted.
    /// - `ResourceLimit` if the result exceeds `options.max_bytes()`.
    pub fn pairwise_with_options(
        &self,
        method: PostHocMethod,
        options: PostHocOptions,
    ) -> Result<PostHocResult<S>, SciOxideError> {
        let options = options.resolve()?;
        if method == PostHocMethod::Dunnett {
            return Err(invalid(
                scioxide_core::argument!("method"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "use control comparisons for Dunnett",
            ));
        }
        if method == PostHocMethod::GamesHowell {
            self.summaries.positive_variances()?;
        }
        let k = self.summaries.groups.len();
        let count = k.checked_mul(k - 1).map(|n| n / 2).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })?;
        budget_bytes(
            count,
            core::mem::size_of::<PostHocComparison<S>>(),
            options.max_bytes,
        )?;
        let mut comparisons = scioxide_core::backend::try_with_capacity(
            count,
            scioxide_core::quantity!("pairwise_comparisons"),
        )?;
        let mut used = 0;
        let pooled = if method == PostHocMethod::TukeyKramer {
            Some(self.summaries.pooled()?)
        } else {
            None
        };
        let common = if let Some((_, df)) = pooled {
            Some(RangeKernel { k, degrees: df })
        } else {
            None
        };
        let common_critical = common
            .as_ref()
            .map(|kernel| kernel.critical(options, &mut used))
            .transpose()?;
        for i in 0..k {
            for j in i + 1..k {
                let (a, b) = (self.summaries.groups[i], self.summaries.groups[j]);
                let difference = center_difference(b.mean, a.mean)?;
                let (scale, df) = if let Some((variance, df)) = pooled {
                    (
                        variance
                            .mul(Scaled::from_value(0.5 * (1.0 / a.n + 1.0 / b.n))?)?
                            .sqrt()?,
                        df,
                    )
                } else {
                    let va = a.variance.div(Scaled::from_value(a.n)?)?;
                    let vb = b.variance.div(Scaled::from_value(b.n)?)?;
                    let sum = va.add(vb)?;
                    let fraction_a = va.div(sum)?;
                    let fraction_b = vb.div(sum)?;
                    (
                        sum.mul(Scaled::HALF)?.sqrt()?,
                        Scaled::ONE
                            .div(
                                fraction_a
                                    .square()?
                                    .div(Scaled::from_value(a.n - 1.0)?)?
                                    .add(
                                        fraction_b.square()?.div(Scaled::from_value(b.n - 1.0)?)?,
                                    )?,
                            )?
                            .round()?,
                    )
                };
                let individual;
                let kernel = if let Some(ref kernel) = common {
                    kernel
                } else {
                    individual = RangeKernel { k, degrees: df };
                    &individual
                };
                let critical = if let Some(c) = common_critical {
                    c
                } else {
                    kernel.critical(options, &mut used)?
                };
                let (probability, error) = kernel.tail(
                    difference.abs().div(scale)?.round()?,
                    true,
                    options,
                    &mut used,
                )?;
                comparisons.push(PostHocComparison {
                    first: i,
                    second: j,
                    difference: difference.round_to()?,
                    p_value: number(probability)?,
                    probability_error: number(error)?,
                    lower: Some(
                        difference
                            .sub(scale.mul(Scaled::from_value(critical)?)?)?
                            .round_to()?,
                    ),
                    upper: Some(
                        difference
                            .add(scale.mul(Scaled::from_value(critical)?)?)?
                            .round_to()?,
                    ),
                    degrees_of_freedom: df,
                });
            }
        }
        Ok(PostHocResult {
            comparisons,
            method,
            alternative: TestAlternative::TwoSided,
            confidence: options.confidence,
            evaluations: used,
        })
    }
    /// Computes simultaneous comparisons with default options.
    ///
    /// # Errors
    ///
    /// Returns errors of the configured comparison.
    pub fn pairwise(&self, method: PostHocMethod) -> Result<PostHocResult<S>, SciOxideError> {
        self.pairwise_with_options(method, PostHocOptions::default())
    }

    /// Compares each group with a control group by Dunnett's single-step
    /// procedure.
    ///
    /// For group `i` and control `c`, the difference is `d = ȳᵢ − ȳ_c` and its
    /// standard error `σ = √(s² (1/nᵢ + 1/n_c))`, with the pooled variance `s²`
    /// on `ν = N − k` degrees of freedom. The statistics `dᵢ / σᵢ` jointly
    /// follow a multivariate t distribution whose correlations depend on the
    /// group sizes. The p-value and the critical value `q` come from this joint
    /// distribution, which is integrated by composite 16-point Gauss–Legendre
    /// quadrature over the normal and chi variables, refined until two
    /// successive estimates agree within the absolute tolerance.
    ///
    /// For a two-sided alternative the interval is `d ± q σ`. A one-sided
    /// alternative gives a one-sided interval: `Greater` has only a lower
    /// endpoint and `Less` only an upper endpoint.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `control` is not a group index.
    /// - `InvalidInput` if `N − k ≤ 0` or the pooled variance is zero.
    /// - `InvalidInput` if the absolute tolerance is outside `[1e-12, 1e-3]`,
    ///   or `min(α, 1 − α)` is at most 8 times the absolute tolerance, where
    ///   `α = 1 − confidence`.
    /// - `InvalidInput` if the critical-value search does not converge.
    /// - `ConvergenceFailure` if the evaluation limit is exhausted or the
    ///   quadrature does not reach the tolerance.
    /// - `ResourceLimit` if the result exceeds `options.max_bytes()`.
    pub fn compare_control_with_options(
        &self,
        control: usize,
        alternative: TestAlternative,
        options: PostHocOptions,
    ) -> Result<PostHocResult<S>, SciOxideError> {
        let options = options.resolve()?;
        if control >= self.summaries.groups.len() {
            return Err(invalid(
                scioxide_core::argument!("control"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid control group index",
            ));
        }
        let (variance, df) = self.summaries.pooled()?;
        budget_bytes(
            self.summaries.groups.len() - 1,
            core::mem::size_of::<PostHocComparison<S>>(),
            options.max_bytes,
        )?;
        let kernel = ControlProbabilityKernel::new(
            ControlFamily {
                groups: &self.summaries.groups,
                control,
                two_sided: alternative == TestAlternative::TwoSided,
            },
            df,
            options.tolerance.absolute(),
        )?;
        let mut used = 0;
        let critical = kernel.critical(options, &mut used)?;
        let mut comparisons = scioxide_core::backend::try_with_capacity(
            self.summaries.groups.len() - 1,
            scioxide_core::quantity!("control_comparisons"),
        )?;
        for (i, g) in self.summaries.groups.iter().enumerate() {
            if i == control {
                continue;
            }
            let difference = center_difference(self.summaries.groups[control].mean, g.mean)?;
            let scale = variance
                .mul(Scaled::from_value(
                    1.0 / g.n + 1.0 / self.summaries.groups[control].n,
                )?)?
                .sqrt()?;
            let t = match alternative {
                TestAlternative::TwoSided => difference.abs().div(scale)?.round()?,
                TestAlternative::Greater => difference.div(scale)?.round()?,
                TestAlternative::Less => (-difference).div(scale)?.round()?,
            };
            let (cdf_with_options, error) = kernel.cdf(t, options, &mut used)?;
            comparisons.push(PostHocComparison {
                first: i,
                second: control,
                difference: difference.round_to()?,
                p_value: number(1.0 - cdf_with_options)?,
                probability_error: number(error)?,
                lower: if alternative == TestAlternative::Less {
                    None
                } else {
                    Some(
                        difference
                            .sub(scale.mul(Scaled::from_value(critical)?)?)?
                            .round_to()?,
                    )
                },
                upper: if alternative == TestAlternative::Greater {
                    None
                } else {
                    Some(
                        difference
                            .add(scale.mul(Scaled::from_value(critical)?)?)?
                            .round_to()?,
                    )
                },
                degrees_of_freedom: df,
            });
        }
        Ok(PostHocResult {
            comparisons,
            method: PostHocMethod::Dunnett,
            alternative,
            confidence: options.confidence,
            evaluations: used,
        })
    }
    /// Computes simultaneous comparisons with default options.
    ///
    /// # Errors
    ///
    /// Returns errors of the configured comparison.
    pub fn compare_control(
        &self,
        control: usize,
        alternative: TestAlternative,
    ) -> Result<PostHocResult<S>, SciOxideError> {
        self.compare_control_with_options(control, alternative, PostHocOptions::default())
    }
}
