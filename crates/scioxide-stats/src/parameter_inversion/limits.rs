//! Confidence limits for distribution parameters by test inversion.
use super::*;
use crate::{ConfidenceProbability, TestAlternative};
use scioxide_core::ExtendedInterval;

use scioxide_random::{Binomial, ChiSquared, LawOptions, LogCdf, Poisson};

/// Confidence set for a distribution parameter, obtained by inverting a test.
///
/// For confidence level `c`, a [`TestAlternative::TwoSided`] set puts probability `(1 − c)/2`
/// in each excluded tail. [`TestAlternative::Greater`] gives a lower confidence bound, so the
/// set extends to the upper end of the parameter domain. [`TestAlternative::Less`] gives an
/// upper confidence bound, so the set extends to the lower end. The coverage is nominal and
/// holds only under the assumed sampling distribution. The set can be empty.
#[derive(Debug, Clone, Copy)]
pub struct ParameterConfidenceLimits<S: RealScalar> {
    set: ExtendedInterval<S>,
    confidence: ConfidenceProbability,
    alternative: TestAlternative,
    evaluations: usize,
    lower_report: Option<ParameterInversionReport<S>>,
    upper_report: Option<ParameterInversionReport<S>>,
}
impl<S: RealScalar> ParameterConfidenceLimits<S> {
    /// Returns the confidence set, which includes its endpoints and may be empty or unbounded.
    pub const fn set(&self) -> ExtendedInterval<S> {
        self.set
    }
    /// Returns the nominal confidence level `c` of [`Self::set`].
    pub const fn confidence(&self) -> ConfidenceProbability {
        self.confidence
    }
    /// Returns the alternative of the inverted test.
    pub const fn alternative(&self) -> TestAlternative {
        self.alternative
    }
    /// Returns the total number of tail evaluations for both endpoints.
    pub const fn evaluations(&self) -> usize {
        self.evaluations
    }
    /// Returns the solver report of the lower limit, or `None` if the limit is an end of the
    /// parameter domain or the set is empty.
    pub const fn lower_report(&self) -> Option<&ParameterInversionReport<S>> {
        self.lower_report.as_ref()
    }
    /// Returns the solver report of the upper limit, or `None` if the limit is an end of the
    /// parameter domain or the set is empty.
    pub const fn upper_report(&self) -> Option<&ParameterInversionReport<S>> {
        self.upper_report.as_ref()
    }
}
#[derive(Clone, Copy)]
// Keep successful confidence queries allocation-free; reports are caller-visible evidence.
#[allow(clippy::large_enum_variant)]
enum Endpoint<S: RealScalar> {
    Value(S, ParameterInversionReport<S>),
    Boundary(Option<S>),
    Empty,
}
#[allow(clippy::too_many_arguments)]
fn endpoint<S: RealScalar>(
    domain: ExtendedInterval<S>,
    bracket: Interval<S>,
    target: ParameterProbability<S>,
    direction: ParameterMonotonicity,
    lower: bool,
    options: &InversionOptions<S>,
    f: &impl Fn(S) -> Result<S, SciOxideError>,
    context: &mut SolveContext<'_, '_>,
) -> Result<Endpoint<S>, SciOxideError> {
    let (domain_low, domain_high) = domain.bounds().ok_or_else(|| {
        invalid(
            scioxide_core::argument!("bracket"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "empty statistical parameter domain",
        )
    })?;
    let a = query(context, || f(bracket.lower()))?;
    let b = query(context, || f(bracket.upper()))?;
    let increasing = direction == ParameterMonotonicity::Increasing;
    if (increasing && a > b) || (!increasing && a < b) {
        return Err(invalid(
            scioxide_core::argument!("bracket"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "confidence tail monotonicity violated",
        ));
    }
    let before = if increasing {
        target.log < a
    } else {
        target.log > a
    };
    let after = if increasing {
        target.log > b
    } else {
        target.log < b
    };
    if before {
        if domain_low != Some(bracket.lower()) {
            return Err(invalid(
                scioxide_core::argument!("bracket"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "confidence bracket misses a parameter endpoint",
            ));
        }
        return Ok(if lower {
            Endpoint::Boundary(domain_low)
        } else {
            Endpoint::Empty
        });
    }
    if after {
        if domain_high != Some(bracket.upper()) {
            return Err(invalid(
                scioxide_core::argument!("bracket"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "confidence bracket misses a parameter endpoint",
            ));
        }
        return Ok(if lower {
            Endpoint::Empty
        } else {
            Endpoint::Boundary(domain_high)
        });
    }
    let report = invert_parameter_scoped(f, bracket, target, direction, options, context)?;
    match report.solution {
        ParameterSolution::Unique { value, .. } if report.root.is_none_or(|r| r.is_success()) => {
            Ok(Endpoint::Value(value, report))
        }
        _ => Err(SciOxideError::convergence_failure(
            ConvergenceReason::Stagnation,
            ConvergenceDiagnostics::new(),
            "confidence endpoint is nonunique, inconclusive or unsuccessful",
        )),
    }
}
#[allow(clippy::too_many_arguments)]
fn limits<'scope, 'capability, S: RealScalar>(
    domain: ExtendedInterval<S>,
    bracket: Interval<S>,
    confidence: ConfidenceProbability,
    alternative: TestAlternative,
    direction: ParameterMonotonicity,
    options: &InversionOptions<S>,
    lower_fixed: bool,
    upper_fixed: bool,
    lower_cdf: impl Fn(S) -> Result<S, SciOxideError>,
    upper_cdf: impl Fn(S) -> Result<S, SciOxideError>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    let (domain_low, domain_high) = domain.bounds().ok_or_else(|| {
        invalid(
            scioxide_core::argument!("bracket"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "empty statistical parameter domain",
        )
    })?;
    if !matches!(
        direction,
        ParameterMonotonicity::Increasing | ParameterMonotonicity::Decreasing
    ) || !domain.contains(bracket.lower())
        || !domain.contains(bracket.upper())
    {
        return Err(invalid(
            scioxide_core::argument!("bracket"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "confidence inversion requires a monotone bracket inside the parameter domain",
        ));
    }
    let alpha = (1.0 - confidence.get())
        / if alternative == TestAlternative::TwoSided {
            2.0
        } else {
            1.0
        };
    let low_log = S::try_from_f64_lossy(alpha.ln())?;
    let high_log = S::try_from_f64_lossy((-alpha).ln_1p())?;
    let increasing = direction == ParameterMonotonicity::Increasing;
    let mut state = SolveState::new();
    scope.into().run(
        &mut state,
        identity(),
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, options.work_budget()),
        |context| {
            context.execution().check()?;
            let lower = if alternative == TestAlternative::Less || lower_fixed {
                Endpoint::Boundary(domain_low)
            } else {
                endpoint(
                    domain,
                    bracket,
                    ParameterProbability::from_log(
                        if increasing { low_log } else { high_log },
                        crate::TailSide::Lower,
                    )?,
                    direction,
                    true,
                    options,
                    &lower_cdf,
                    context,
                )?
            };
            let upper = if alternative == TestAlternative::Greater || upper_fixed {
                Endpoint::Boundary(domain_high)
            } else {
                endpoint(
                    domain,
                    bracket,
                    ParameterProbability::from_log(
                        if increasing { high_log } else { low_log },
                        crate::TailSide::Lower,
                    )?,
                    direction,
                    false,
                    options,
                    &upper_cdf,
                    context,
                )?
            };
            let (low, lower_report) = match lower {
                Endpoint::Value(v, r) => (Some(v), Some(r)),
                Endpoint::Boundary(v) => (v, None),
                Endpoint::Empty => (None, None),
            };
            let (high, upper_report) = match upper {
                Endpoint::Value(v, r) => (Some(v), Some(r)),
                Endpoint::Boundary(v) => (v, None),
                Endpoint::Empty => (None, None),
            };
            let set = if matches!(lower, Endpoint::Empty)
                || matches!(upper, Endpoint::Empty)
                || low.zip(high).is_some_and(|(a, b)| a > b)
            {
                ExtendedInterval::empty()
            } else {
                ExtendedInterval::new(low, high)?
            };
            Ok(ParameterConfidenceLimits {
                set,
                confidence,
                alternative,
                evaluations: context.counts().total()?.total_providers()?,
                lower_report,
                upper_report,
            })
        },
    )
}
/// Executes [`invert_confidence_limits_scoped`] with sequential native capabilities.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors.
pub fn invert_confidence_limits_with_options<S: RealScalar>(
    log_cdf: impl Fn(S) -> Result<S, SciOxideError>,
    domain: ExtendedInterval<S>,
    bracket: Interval<S>,
    confidence: ConfidenceProbability,
    alternative: TestAlternative,
    direction: ParameterMonotonicity,
    options: &InversionOptions<S>,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    invert_confidence_limits_scoped(
        log_cdf,
        domain,
        bracket,
        confidence,
        alternative,
        direction,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Computes confidence limits for a parameter `θ` of a continuous distribution by inverting
/// its CDF.
///
/// `log_cdf(θ)` returns `ln F(t; θ)`, where `F` is the CDF of the test statistic and `t` its
/// observed value. It must be strictly monotone in `θ` on `domain`, in the direction given by
/// `direction`. With `α = 1 − c` for a one-sided set and `α = (1 − c)/2` for a two-sided set,
/// the limits solve `F(t; θ) = α` and `F(t; θ) = 1 − α` by bisection. If an equation has no
/// solution in the domain, the limit is the end of the domain or the set is empty. `bracket`
/// must contain every limit that is not an end of the domain. The work budget of `options`
/// limits the tail evaluations of both limits together. `scope` is an execution context or a
/// solve scope that shares its budgets and cancellation with an enclosing solve.
///
/// # Errors
///
/// - `InvalidInput` if `direction` is not `Increasing` or `Decreasing`, `domain` is empty,
///   or `bracket` is not inside `domain`.
/// - `InvalidInput` if a limit lies outside `bracket` but inside `domain`, or the values of
///   `log_cdf` at the ends of `bracket` contradict `direction`.
/// - `ConvergenceFailure` if the bisection for a limit does not converge or finds no unique
///   solution.
/// - The errors of [`invert_parameter_scoped`](crate::invert_parameter_scoped).
///
/// # References
///
/// G. Casella and R. L. Berger, *Statistical Inference*, 2nd ed., Duxbury, 2002, §9.2.3.
#[allow(clippy::too_many_arguments)]
pub fn invert_confidence_limits_scoped<'scope, 'capability, S: RealScalar>(
    log_cdf: impl Fn(S) -> Result<S, SciOxideError>,
    domain: ExtendedInterval<S>,
    bracket: Interval<S>,
    confidence: ConfidenceProbability,
    alternative: TestAlternative,
    direction: ParameterMonotonicity,
    options: &InversionOptions<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    limits(
        domain,
        bracket,
        confidence,
        alternative,
        direction,
        options,
        false,
        false,
        &log_cdf,
        &log_cdf,
        scope,
    )
}

/// Executes [`binomial_confidence_limits_scoped`] with sequential native capabilities.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors.
pub fn binomial_confidence_limits_with_options<S: RealScalar>(
    successes: usize,
    trials: usize,
    confidence: ConfidenceProbability,
    alternative: TestAlternative,
    options: &InversionOptions<S>,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    binomial_confidence_limits_scoped(
        successes,
        trials,
        confidence,
        alternative,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Computes the Clopper–Pearson exact confidence limits for a binomial success probability.
///
/// For `k` successes in `n` trials, the lower limit `p_L` solves `P(X ≥ k; p_L) = α` and the
/// upper limit `p_U` solves `P(X ≤ k; p_U) = α`, where `X ~ Binomial(n, p)`, `α = 1 − c` for a
/// one-sided set and `α = (1 − c)/2` for a two-sided set. The lower limit is 0 when `k = 0`
/// and the upper limit is 1 when `k = n`. The limits are found by bisection on `[0, 1]`. The
/// coverage is at least `c` for every `p`. `scope` is an execution context or a solve scope
/// that shares its budgets and cancellation with an enclosing solve.
///
/// # Errors
///
/// - `InvalidInput` if `trials` is zero or `successes > trials`.
/// - `ConvergenceFailure` if the bisection for a limit does not converge or the work budget of
///   `options` runs out.
/// - `ExecutionStopped` on cancellation or deadline.
///
/// # Examples
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_random::InversionOptions;
/// use scioxide_stats::{ConfidenceProbability, TestAlternative, binomial_confidence_limits_scoped};
/// let limits = binomial_confidence_limits_scoped(
///     0,
///     10,
///     ConfidenceProbability::new(0.95)?,
///     TestAlternative::Less,
///     &InversionOptions::<f64>::default(),
///     ExecutionContext::sequential_native(),
/// )?;
/// let (lower, upper) = limits.set().bounds().unwrap();
/// assert_eq!(lower, Some(0.0));
/// // P(X ≤ 0; p) = (1 − p)¹⁰ = 0.05 gives the upper bound 1 − 0.05^(1/10).
/// assert!((upper.unwrap() - (1.0 - 0.05_f64.powf(0.1))).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// C. J. Clopper and E. S. Pearson, "The use of confidence or fiducial limits illustrated in
/// the case of the binomial", *Biometrika* 26 (1934), 404–413.
pub fn binomial_confidence_limits_scoped<'scope, 'capability, S: RealScalar>(
    successes: usize,
    trials: usize,
    confidence: ConfidenceProbability,
    alternative: TestAlternative,
    options: &InversionOptions<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    if trials == 0 || successes > trials {
        return Err(invalid(
            scioxide_core::argument!("successes"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid binomial confidence counts",
        ));
    }
    limits(
        ExtendedInterval::new(Some(S::ZERO), Some(S::ONE))?,
        Interval::new(S::ZERO, S::ONE)?,
        confidence,
        alternative,
        ParameterMonotonicity::Decreasing,
        options,
        successes == 0,
        successes == trials,
        |p| Binomial::new(trials, p)?.log_cdf(successes.saturating_sub(1)),
        |p| Binomial::new(trials, p)?.log_cdf(successes),
        scope,
    )
}

/// Executes [`poisson_rate_confidence_limits_scoped`] with sequential native capabilities.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors.
pub fn poisson_rate_confidence_limits_with_options<S: RealScalar>(
    count: usize,
    exposure: S,
    bracket: Interval<S>,
    confidence: ConfidenceProbability,
    alternative: TestAlternative,
    options: &InversionOptions<S>,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    poisson_rate_confidence_limits_scoped(
        count,
        exposure,
        bracket,
        confidence,
        alternative,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Computes exact confidence limits for a Poisson rate.
///
/// For `k` events over exposure `E`, the count is `X ~ Poisson(λE)`. The lower limit `λ_L`
/// solves `P(X ≥ k; λ_L) = α` and the upper limit `λ_U` solves `P(X ≤ k; λ_U) = α`, with
/// `α = 1 − c` for a one-sided set and `α = (1 − c)/2` for a two-sided set. The lower limit is
/// 0 when `k = 0`. The limits are found by bisection on the rate interval `bracket`, which must
/// contain them. `scope` is an execution context or a solve scope that shares its budgets and
/// cancellation with an enclosing solve.
///
/// # Errors
///
/// - `InvalidInput` if `exposure` is not finite and positive.
/// - The errors of [`invert_confidence_limits_scoped`], including `InvalidInput` if `bracket` has a
///   negative end or does not contain a limit.
///
/// # References
///
/// F. Garwood, "Fiducial limits for the Poisson distribution", *Biometrika* 28 (1936),
/// 437–442.
#[allow(clippy::too_many_arguments)]
pub fn poisson_rate_confidence_limits_scoped<'scope, 'capability, S: RealScalar>(
    count: usize,
    exposure: S,
    bracket: Interval<S>,
    confidence: ConfidenceProbability,
    alternative: TestAlternative,
    options: &InversionOptions<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    if !exposure.is_finite() || exposure <= S::ZERO {
        return Err(invalid(
            scioxide_core::argument!("exposure"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "positive finite Poisson exposure required",
        ));
    }
    limits(
        ExtendedInterval::new(Some(S::ZERO), None)?,
        bracket,
        confidence,
        alternative,
        ParameterMonotonicity::Decreasing,
        options,
        count == 0,
        false,
        |rate| Poisson::new(rate * exposure)?.log_cdf(count.saturating_sub(1)),
        |rate| Poisson::new(rate * exposure)?.log_cdf(count),
        scope,
    )
}

/// Executes [`normal_variance_confidence_limits_scoped`] with sequential native capabilities.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors.
pub fn normal_variance_confidence_limits_with_options<S: RealScalar>(
    scatter: S,
    degrees: S,
    bracket: Interval<S>,
    confidence: ConfidenceProbability,
    alternative: TestAlternative,
    options: &InversionOptions<S>,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    normal_variance_confidence_limits_scoped(
        scatter,
        degrees,
        bracket,
        confidence,
        alternative,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Computes confidence limits for the variance `σ²` of a normal distribution.
///
/// `scatter` is the sum of squared deviations `Q = Σ (xᵢ − x̄)²` and `degrees` its degrees of
/// freedom `ν`, usually `n − 1`. Since `Q/σ² ~ χ²(ν)`, the limits solve
/// `F_χ²(Q/σ²; ν) = α` and `F_χ²(Q/σ²; ν) = 1 − α`, which gives the interval
/// `[Q/χ²_{1−α}(ν), Q/χ²_α(ν)]` with `χ²_q(ν)` the `q` quantile. Here `α = 1 − c` for a
/// one-sided set and `α = (1 − c)/2` for a two-sided set. The limits are found by bisection on
/// the variance interval `bracket`, which must contain them. `scope` is an execution context
/// or a solve scope that shares its budgets and cancellation with an enclosing solve.
///
/// # Errors
///
/// - `InvalidInput` if `scatter` is not finite and positive, or `bracket.lower() ≤ 0`.
/// - Errors from constructing the chi-squared distribution with `degrees`.
/// - The errors of [`invert_confidence_limits_scoped`].
#[allow(clippy::too_many_arguments)]
pub fn normal_variance_confidence_limits_scoped<'scope, 'capability, S: RealScalar>(
    scatter: S,
    degrees: S,
    bracket: Interval<S>,
    confidence: ConfidenceProbability,
    alternative: TestAlternative,
    options: &InversionOptions<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    if !scatter.is_finite() || scatter <= S::ZERO || bracket.lower() <= S::ZERO {
        return Err(invalid(
            scioxide_core::argument!("scatter"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "positive scatter and variance bracket required",
        ));
    }
    let law = ChiSquared::new(degrees)?;
    invert_confidence_limits_scoped(
        |variance| law.log_cdf(scatter / variance),
        ExtendedInterval::new(Some(S::ZERO), None)?,
        bracket,
        confidence,
        alternative,
        ParameterMonotonicity::Decreasing,
        options,
        scope,
    )
}

/// Executes [`noncentrality_confidence_limits_scoped`] with sequential native capabilities.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors.
pub fn noncentrality_confidence_limits_with_options<S: RealScalar>(
    model: ProbabilityParameter<S>,
    bracket: Interval<S>,
    confidence: ConfidenceProbability,
    alternative: TestAlternative,
    options: &InversionOptions<S>,
    resources: LawOptions<S>,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    noncentrality_confidence_limits_scoped(
        model,
        bracket,
        confidence,
        alternative,
        options,
        resources,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Computes confidence limits for the noncentrality of a noncentral t, chi-squared or F
/// distribution.
///
/// `model` gives the distribution, the observed statistic `x` and the fixed degrees of
/// freedom. The limits invert the CDF at `x` as in [`invert_confidence_limits_scoped`]. The
/// parameter domain is the real line for the t distribution and `[0, ∞)` for the others. The
/// work budget of `options` covers both limits together. `resources` sets the work limits of
/// each tail evaluation. `scope` is an execution context or a solve scope that shares its
/// budgets and cancellation with an enclosing solve.
///
/// # Errors
///
/// - `InvalidInput` if `model` is not [`ProbabilityParameter::StudentTNoncentrality`],
///   [`ProbabilityParameter::ChiSquaredNoncentrality`] or
///   [`ProbabilityParameter::FNoncentrality`].
/// - `InvalidInput` if `bracket` is not finite, or has a negative end for a chi-squared or F
///   noncentrality.
/// - The errors of [`ProbabilityParameter::monotonicity`] and [`invert_confidence_limits_scoped`].
pub fn noncentrality_confidence_limits_scoped<'scope, 'capability, S: RealScalar>(
    model: ProbabilityParameter<S>,
    bracket: Interval<S>,
    confidence: ConfidenceProbability,
    alternative: TestAlternative,
    options: &InversionOptions<S>,
    resources: LawOptions<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    let low = match model {
        ProbabilityParameter::StudentTNoncentrality { .. } => None,
        ProbabilityParameter::ChiSquaredNoncentrality { .. }
        | ProbabilityParameter::FNoncentrality { .. } => Some(S::ZERO),
        _ => {
            return Err(invalid(
                scioxide_core::argument!("model"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "noncentrality confidence limits require a noncentral law",
            ));
        }
    };
    model.validate_bracket(bracket)?;
    invert_confidence_limits_scoped(
        |p| model.log_tail(p, crate::TailSide::Lower, resources),
        ExtendedInterval::new(low, None)?,
        bracket,
        confidence,
        alternative,
        model.monotonicity(crate::TailSide::Lower)?,
        options,
        scope,
    )
}

/// Inverts the parameter probability with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same bracketed algorithm and accuracy as [`invert_confidence_limits_with_options`].
///
/// # Errors
///
/// Rejects invalid data or bounds and propagates numerical and convergence errors.
pub fn invert_confidence_limits<S: RealScalar>(
    log_cdf: impl Fn(S) -> Result<S, SciOxideError>,
    domain: ExtendedInterval<S>,
    lower: S,
    upper: S,
    direction: ParameterMonotonicity,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    invert_confidence_limits_with_options(
        log_cdf,
        domain,
        Interval::new(lower, upper)?,
        ConfidenceProbability::default(),
        TestAlternative::TwoSided,
        direction,
        &InversionOptions::default(),
    )
}

/// Inverts the parameter probability with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same bracketed algorithm and accuracy as [`binomial_confidence_limits_with_options`].
///
/// # Errors
///
/// Rejects invalid data or bounds and propagates numerical and convergence errors.
pub fn binomial_confidence_limits<S: RealScalar>(
    successes: usize,
    trials: usize,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    binomial_confidence_limits_with_options(
        successes,
        trials,
        ConfidenceProbability::default(),
        TestAlternative::TwoSided,
        &InversionOptions::default(),
    )
}

/// Inverts the parameter probability with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same bracketed algorithm and accuracy as [`poisson_rate_confidence_limits_with_options`].
///
/// # Errors
///
/// Rejects invalid data or bounds and propagates numerical and convergence errors.
pub fn poisson_rate_confidence_limits<S: RealScalar>(
    count: usize,
    exposure: S,
    lower: S,
    upper: S,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    poisson_rate_confidence_limits_with_options(
        count,
        exposure,
        Interval::new(lower, upper)?,
        ConfidenceProbability::default(),
        TestAlternative::TwoSided,
        &InversionOptions::default(),
    )
}

/// Inverts the parameter probability with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same bracketed algorithm and accuracy as [`normal_variance_confidence_limits_with_options`].
///
/// # Errors
///
/// Rejects invalid data or bounds and propagates numerical and convergence errors.
pub fn normal_variance_confidence_limits<S: RealScalar>(
    scatter: S,
    degrees: S,
    lower: S,
    upper: S,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    normal_variance_confidence_limits_with_options(
        scatter,
        degrees,
        Interval::new(lower, upper)?,
        ConfidenceProbability::default(),
        TestAlternative::TwoSided,
        &InversionOptions::default(),
    )
}

/// Inverts the parameter probability with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same bracketed algorithm and accuracy as [`noncentrality_confidence_limits_with_options`].
///
/// # Errors
///
/// Rejects invalid data or bounds and propagates numerical and convergence errors.
pub fn noncentrality_confidence_limits<S: RealScalar>(
    model: ProbabilityParameter<S>,
    lower: S,
    upper: S,
) -> Result<ParameterConfidenceLimits<S>, SciOxideError> {
    noncentrality_confidence_limits_with_options(
        model,
        Interval::new(lower, upper)?,
        ConfidenceProbability::default(),
        TestAlternative::TwoSided,
        &InversionOptions::default(),
        LawOptions::default(),
    )
}
