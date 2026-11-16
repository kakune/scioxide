//! Support queries, argument checks and log-probability arithmetic for probability laws.
pub(crate) use scioxide_core::logaddexp as logadd;
use scioxide_core::{Argument, ExtendedInterval, RealScalar, SciOxideError};

/// The one bounded-work failure, carrying the attempted work, including
/// failed attempts, as shared counts.
#[cold]
#[inline(never)]
pub(crate) fn exhausted_with(counts: scioxide_core::EvaluationCounts) -> SciOxideError {
    SciOxideError::convergence_failure(
        scioxide_core::ConvergenceReason::IterationLimit,
        scioxide_core::ConvergenceDiagnostics::new().with_counts(counts.into()),
        "numerical work or sampling budget exhausted",
    )
}

/// Exhaustion for callers that track their own work units.
#[cold]
#[inline(never)]
pub(crate) fn exhausted() -> SciOxideError {
    exhausted_with(scioxide_core::EvaluationCounts::new())
}

/// Exhaustion after `used` attempted work units.
#[cold]
#[inline(never)]
pub(crate) fn iteration_limit(used: usize) -> SciOxideError {
    exhausted_with(scioxide_core::EvaluationCounts::from_iterations(used))
}

/// Count work before evaluation; if the limit refuses it, the provider is not evaluated.
pub(crate) fn charge(left: &mut usize) -> Result<(), SciOxideError> {
    *left = left.checked_sub(1).ok_or_else(exhausted)?;
    Ok(())
}

/// Run a bounded kernel within both its policy and the enclosing query allowance.
/// Reconcile attempted work even when the kernel fails; no retry gets a fresh budget.
pub(crate) fn with_work<T>(
    context: &mut scioxide_core::SolveContext<'_, '_>,
    limit: scioxide_core::Budget,
    kernel: impl FnOnce(&mut usize) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let allowance = context
        .effective_budget_report(scioxide_core::BudgetKind::Evaluation)
        .remaining()
        .unwrap_or(usize::MAX)
        .min(limit.get());
    let mut left = allowance;
    let result = kernel(&mut left);
    if !context
        .try_consume(scioxide_core::BudgetKind::Evaluation, allowance - left)?
        .is_granted()
    {
        return Err(exhausted());
    }
    result
}

/// Bridge internal remaining-work kernels to the public scoped capabilities.
pub(crate) fn scoped<T>(
    left: &mut usize,
    query: impl FnOnce(&mut scioxide_core::SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let budget = scioxide_core::Budget::try_from(*left).map_err(|_| exhausted())?;
    let mut budgets = scioxide_core::SolveBudgets::unbounded()
        .with_limit(scioxide_core::BudgetKind::Evaluation, budget);
    let mut state = scioxide_core::SolveState::new();
    let mut context = state.begin_solve(
        scioxide_core::ExecutionContext::sequential_native(),
        scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("random", "law-query", 1),
            0,
            0,
        ),
        &mut budgets,
    )?;
    let result = query(&mut context);
    *left -= context
        .budget_report(scioxide_core::BudgetKind::Evaluation)
        .consumed();
    result
}

/// Run `kernel` in one fresh root or child epoch of `scope` with the local
/// limits `allowance`. The identity is internal: callers never build one.
pub(crate) fn run_scoped<'scope, 'capability, T>(
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    algorithm: scioxide_core::AlgorithmId,
    allowance: scioxide_core::SolveBudgets,
    kernel: impl FnOnce(&mut scioxide_core::SolveContext<'_, 'capability>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    scope.into().run(
        &mut scioxide_core::SolveState::new(),
        scioxide_core::SolveIdentity::new(scioxide_core::ProblemToken::new(0), algorithm, 0, 0),
        allowance,
        kernel,
    )
}

/// One logarithmic tail query of `law` on `context`: `ln P(X > x)` if
/// `upper`, otherwise `ln P(X ≤ x)`. A cancellation or deadline seen before
/// the query ends it with `ExecutionStopped`.
pub(crate) fn tail_scoped<S: RealScalar, D: crate::LogCdf<S> + crate::LogSurvival<S> + ?Sized>(
    law: &D,
    x: S,
    upper: bool,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<S, SciOxideError> {
    if let Some(reason) = context.stop_reason()? {
        return Err(SciOxideError::execution_stopped(reason));
    }
    if upper {
        law.log_sf_scoped(x, context)
    } else {
        law.log_cdf_scoped(x, context)
    }
}

/// A fixed-cost provider is still counted before it is invoked.
pub(crate) fn provider<T>(
    context: &mut scioxide_core::SolveContext<'_, '_>,
    query: impl FnOnce() -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    if !context
        .record_provider(scioxide_core::EvaluationRole::Function)?
        .is_granted()
    {
        return Err(exhausted());
    }
    query()
}

/// Bound nested special-function work before execution and debit its actual attempts.
pub(crate) fn special<S: RealScalar>(
    left: &mut usize,
    evaluate: impl FnOnce(
        scioxide_core::Budget,
    ) -> Result<scioxide_special::SpecialFunctionResult<S>, SciOxideError>,
) -> Result<S, SciOxideError> {
    let budget = scioxide_core::Budget::try_from(*left).map_err(|_| exhausted())?;
    let report = match evaluate(budget) {
        Ok(report) => report,
        Err(error) => {
            let used = if let scioxide_core::SciOxideErrorView::ConvergenceFailure {
                diagnostics,
                ..
            } = error.view()
            {
                diagnostics.counts().total()?.iterations()
            } else {
                0
            };
            *left = left.checked_sub(used).ok_or_else(exhausted)?;
            return Err(error);
        }
    };
    *left = left
        .checked_sub(report.counts().iterations())
        .ok_or_else(exhausted)?;
    report.try_value()
}

/// Support of a continuous law: the smallest closed interval containing it, and membership of points.
///
/// The interval may contain gaps, such as the gap of a [`Mixture`](crate::Mixture)
/// of two separated uniform laws.
///
/// ```
/// use scioxide_core::ExtendedInterval;
/// use scioxide_random::{Exponential, Support};
/// let law = Exponential::new(2.0_f64)?;
/// assert_eq!(law.support()?.bounds(), Some((Some(0.0), None)));
/// assert!(law.contains(0.0)? && !law.contains(-1.0)?);
/// let clipped = law.support_in(ExtendedInterval::new(Some(-1.0), Some(3.0))?)?;
/// assert_eq!(clipped.bounds(), Some((Some(0.0), Some(3.0))));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub trait Support<S: RealScalar> {
    /// Returns the smallest closed interval containing the support; a missing endpoint is unbounded.
    ///
    /// # Errors
    /// - Returns an error if the support of a component law is invalid, or a
    ///   transformed endpoint cannot be represented.
    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError>;
    /// Returns the smallest closed interval containing the part of the support inside `interval`.
    ///
    /// The default intersects [`Self::support`] with `interval`, which is exact
    /// for a connected support. An intersection that is a single point is empty,
    /// since a continuous law has no atoms. A law with a disconnected support
    /// must override this method and [`Self::contains`].
    ///
    /// # Errors
    /// - The errors of [`Self::support`].
    fn support_in(
        &self,
        interval: ExtendedInterval<S>,
    ) -> Result<ExtendedInterval<S>, SciOxideError> {
        let clipped = self.support()?.intersection(&interval);
        Ok(
            if clipped.bounds().is_some_and(|(a, b)| a.is_some() && a == b) {
                ExtendedInterval::empty()
            } else {
                clipped
            },
        )
    }
    /// Returns whether the finite point `x` belongs to the closed support, endpoints included.
    ///
    /// # Errors
    /// - The errors of [`Self::support`].
    fn contains(&self, x: S) -> Result<bool, SciOxideError> {
        Ok(self.support()?.contains(x))
    }
}

/// Invalid-input error against `argument`.
pub(crate) fn invalid(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::OutOfRange,
        detail,
    )
}
/// Require a finite probability `p` in `[0, 1]`.
pub(crate) fn probability<S: RealScalar>(p: S, argument: Argument) -> Result<(), SciOxideError> {
    crate::util::require_finite_param(p, "probability")?;
    if p < S::ZERO || p > S::ONE {
        return Err(domain(argument, "probability must be in [0, 1]"));
    }
    Ok(())
}
/// Require a finite parameter or query coordinate named `name`.
pub(crate) fn argument<S: RealScalar>(x: S, name: &str) -> Result<(), SciOxideError> {
    crate::util::require_finite_param(x, name)
}
/// Require a finite result.
#[inline]
pub(crate) fn finite<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    scioxide_core::backend::checked_final_rounding(x, "result exceeds finite scalar range")
}

/// Preserve genuine infinite density limits while rejecting indeterminate formulas.
pub(crate) fn checked_log_density<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    if value.is_nan() {
        Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "log-density became indeterminate",
        ))
    } else {
        Ok(value)
    }
}

/// Preserve a genuine singular density, but distinguish exponentiation overflow.
pub(crate) fn density_from_log<S: RealScalar>(log: S) -> Result<S, SciOxideError> {
    if log == S::INFINITY {
        Ok(log)
    } else {
        finite(log.exp())
    }
}

/// Validate a computed positive value, keeping numerical range failures.
pub(crate) fn positive_result<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    if value.is_nan() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "computed positive value is indeterminate",
        ));
    }
    let value = finite(value)?;
    if value <= S::ZERO {
        return Err(SciOxideError::numerical_instability(
            if value == S::ZERO {
                scioxide_core::NumericalIssue::Underflow
            } else {
                scioxide_core::NumericalIssue::Breakdown
            },
            "computed positive value lost its invariant",
        ));
    }
    Ok(value)
}

/// Materialize a strictly positive draw only after all logarithmic transforms.
/// Unlike probability rounding, losing a continuous draw to zero is an error.
pub(crate) fn positive_draw<S: RealScalar>(log: f64) -> Result<S, SciOxideError> {
    use scioxide_core::NumericalIssue;
    if log.is_nan() {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Breakdown,
            "indeterminate logarithmic draw",
        ));
    }
    let value = log.exp();
    if !value.is_finite() {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Overflow,
            "draw exceeds the finite scalar range",
        ));
    }
    let value = S::try_from_f64_lossy(value).map_err(|_| {
        SciOxideError::numerical_instability(
            NumericalIssue::Overflow,
            "draw exceeds the destination scalar range",
        )
    })?;
    if value == S::ZERO {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Underflow,
            "positive draw rounds to zero",
        ));
    }
    Ok(value)
}
pub(crate) fn constant<S: RealScalar>(x: f64) -> Result<S, SciOxideError> {
    if x == f64::NEG_INFINITY {
        Ok(S::NEG_INFINITY)
    } else {
        S::try_from_f64_lossy(x)
    }
}

/// Undefined mathematical moments are domain errors, including divergent moments.
pub(crate) fn undefined_moment(detail: &'static str) -> SciOxideError {
    SciOxideError::domain_error(
        scioxide_core::argument!("self"),
        scioxide_core::DomainReason::Undefined,
        detail,
    )
}

/// Finite mathematical parameters outside their law's domain.
pub(crate) fn domain(argument: Argument, detail: &'static str) -> SciOxideError {
    SciOxideError::domain_error(argument, scioxide_core::DomainReason::OutsideDomain, detail)
}

/// Validate a positive mathematical parameter in its original scalar precision.
pub(crate) fn positive<S: RealScalar>(x: S, name: Argument) -> Result<S, SciOxideError> {
    argument(x, "x")?;
    if x <= S::ZERO {
        return Err(domain(name, "parameter must be positive"));
    }
    Ok(x)
}

/// Validate a nonnegative mathematical parameter, including the degenerate zero law.
pub(crate) fn nonnegative<S: RealScalar>(x: S, name: Argument) -> Result<S, SciOxideError> {
    argument(x, "x")?;
    if x < S::ZERO {
        return Err(domain(name, "parameter must be nonnegative"));
    }
    Ok(x)
}

/// Validate computed log probabilities in the precision used by the kernel.
/// Only positive round-off within 32 epsilons is clamped. Negative infinity
/// denotes exact zero mass; NaN, positive infinity, and larger excursions fail.
pub(crate) fn log_probability<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    if value.is_nan() || value > constant::<S>(32.)? * S::EPSILON {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "invalid logarithmic probability",
        ));
    }
    Ok(value.min(S::ZERO))
}
/// log(1-exp(x)), for x <= 0, including exact zero probabilities.
pub(crate) use scioxide_core::log1mexp;
/// ln(exp(x)-1), x >= 0, without an overflowing exponential.
pub(crate) fn log_expm1<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    if x > S::ONE {
        Ok(x + log1mexp(-x)?)
    } else {
        Ok(x.exp_m1().ln())
    }
}

/// expm1(coefficient * argument) / coefficient, with the continuous zero-coefficient limit.
/// Cancel the scale before exponentiation if the intermediate exponential overflows.
pub(crate) fn expm1_ratio<S: RealScalar>(coefficient: S, argument: S) -> S {
    let y = coefficient * argument;
    if y.abs() <= S::ONE {
        if y == S::ZERO {
            argument
        } else {
            argument * (y.exp_m1() / y)
        }
    } else if y < S::ZERO {
        y.exp_m1() / coefficient
    } else {
        let magnitude = (y + (-(-y).exp()).ln_1p() - coefficient.abs().ln()).exp();
        if coefficient < S::ZERO {
            -magnitude
        } else {
            magnitude
        }
    }
}

/// ln(ln(1+exp(x))), preserving subnormal and underflowed exp(x).
pub(crate) fn log_softplus<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    // Replacing softplus(x) by exp(x) changes its logarithm by < epsilon.
    if x < S::EPSILON.ln() {
        Ok(x)
    } else {
        Ok(scioxide_core::softplus(x)?.ln())
    }
}

/// Reject rounded support endpoints for an interior inverse.
pub(crate) fn interior_result<S: RealScalar>(
    value: S,
    low: Option<S>,
    high: Option<S>,
) -> Result<S, SciOxideError> {
    let value = finite(value)?;
    if low == Some(value) || high == Some(value) {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Underflow,
            "interior quantile rounds to a support endpoint",
        ));
    }
    Ok(value)
}

/// A tail log-probability above `ln(15/16)` is formed from the complementary
/// tail: kernels that are only absolutely accurate near one then still give
/// relative accuracy. Below it, the complement amplifies absolute error by at
/// most sixteen.
fn near_one<S: RealScalar>(log_probability: S) -> Result<bool, SciOxideError> {
    Ok(log_probability > constant::<S>(-0.064_538_521_137_571_18)?)
}

/// The one near-one tail switch of every continuous law and combinator.
///
/// `tail(upper)` evaluates one logarithmic tail, charging its own work.
/// `first_upper` selects the tail evaluated first: the requested tail, or the
/// tail that excludes an approximate median. A finite first value above
/// `ln(15/16)` is replaced by its complementary kernel before validation,
/// because a near-one estimate may be only absolutely accurate and can
/// overshoot zero. Returns the requested log tail and the validated log tail
/// that was actually computed, in the kernel's working precision.
pub(crate) fn smaller_tail_first<S: RealScalar>(
    first_upper: bool,
    upper: bool,
    mut tail: impl FnMut(bool) -> Result<S, SciOxideError>,
) -> Result<(S, S), SciOxideError> {
    let mut computed = first_upper;
    let mut value = tail(computed)?;
    if value.is_finite() && near_one(value)? {
        computed = !computed;
        value = tail(computed)?;
    }
    let value = log_probability(value)?;
    let requested = if computed == upper {
        value
    } else {
        log1mexp(value)?
    };
    Ok((requested, value))
}
/// `log(exp(a) - exp(b))` for ordered log probabilities; a disordered pair is
/// invalid input against `argument`.
pub(crate) fn logdiff<S: RealScalar>(a: S, b: S, argument: Argument) -> Result<S, SciOxideError> {
    if a < b || a.is_nan() || b.is_nan() {
        return Err(invalid(argument, "tail probabilities are not ordered"));
    }
    Ok(if a == b {
        S::NEG_INFINITY
    } else {
        a + log1mexp(b - a)?
    })
}

/// Returns `ln P(low < X ≤ high)` for a continuous law.
///
/// The probability is formed from the lower tails `ln F` or the upper tails
/// `ln(1 − F)`, whichever are smaller, so it keeps its relative accuracy when
/// both CDF values round to one. Infinite endpoints are accepted, and an empty
/// interval gives `−∞`. The cost is at most three tail evaluations. It equals
/// [`log_interval_probability_with_options`] with the default
/// [`LawOptions`](crate::LawOptions).
///
/// # Errors
///
/// The errors of [`log_interval_probability_with_options`].
///
/// # Examples
///
/// ```
/// use scioxide_random::{Normal, log_interval_probability};
/// let law = Normal::<f64>::standard();
/// let log_p = log_interval_probability(&law, -1.0, 1.0)?;
/// assert!((log_p.exp() - 0.682_689_492_137_086).abs() < 1e-12);
/// // Far in the tail, where both CDF values round to one.
/// assert!(log_interval_probability(&law, 40.0, f64::INFINITY)? < -800.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn log_interval_probability<
    S: RealScalar,
    D: crate::LogCdf<S> + crate::LogSurvival<S> + ?Sized,
>(
    law: &D,
    low: S,
    high: S,
) -> Result<S, SciOxideError> {
    log_interval_probability_with_options(law, low, high, crate::LawOptions::default())
}

/// Returns `ln P(low < X ≤ high)` as [`log_interval_probability`] does, with the work limit
/// of `options`.
///
/// Every tail evaluation, with the work the law does inside it, counts toward
/// [`LawOptions::work_budget`](crate::LawOptions::work_budget). The function runs on the
/// calling thread.
///
/// # Errors
///
/// - `InvalidInput` if an endpoint is NaN, `low > high`, or `options` has a
///   relative tolerance outside `[64ε, 0.1)`.
/// - `ConvergenceFailure` if the work limit is exhausted.
/// - Any error of the law's [`LogCdf`](crate::LogCdf) or
///   [`LogSurvival`](crate::LogSurvival).
///
/// # Examples
///
/// ```
/// use scioxide_core::Budget;
/// use scioxide_random::{LawOptions, Normal, log_interval_probability_with_options};
/// let options = LawOptions::default().with_work_budget(Budget::try_from(16)?);
/// let law = Normal::<f64>::standard();
/// let log_p = log_interval_probability_with_options(&law, 0.0, f64::INFINITY, options)?;
/// assert!((log_p - 0.5_f64.ln()).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn log_interval_probability_with_options<
    S: RealScalar,
    D: crate::LogCdf<S> + crate::LogSurvival<S> + ?Sized,
>(
    law: &D,
    low: S,
    high: S,
    options: crate::LawOptions<S>,
) -> Result<S, SciOxideError> {
    log_interval_probability_scoped(
        law,
        low,
        high,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}

/// Returns `ln P(low < X ≤ high)` as [`log_interval_probability_with_options`] does, as one
/// epoch of `scope`.
///
/// `scope` is an [`ExecutionContext`](scioxide_core::ExecutionContext) for a top-level call or
/// the `&mut SolveContext` of an enclosing solve. The call observes the cancellation and
/// deadline of the scope before each tail evaluation. The work of the law's tail evaluations
/// counts in the epoch, within [`LawOptions::work_budget`](crate::LawOptions::work_budget), and
/// in every enclosing solve, so an enclosing evaluation limit also bounds it. With a
/// sequential context the result equals that of [`log_interval_probability_with_options`] bit
/// for bit.
///
/// # Errors
///
/// - The errors of [`log_interval_probability_with_options`]; `ConvergenceFailure` also if an
///   enclosing evaluation limit is exhausted.
/// - `ExecutionStopped` if the scope is cancelled or past its deadline before a tail
///   evaluation.
/// - `AccountingInvariant` if the counts of an enclosing solve are inconsistent.
///
/// # Examples
///
/// ```
/// use scioxide_core::{CancellationToken, ExecutionContext, SciOxideErrorKind};
/// use scioxide_random::{LawOptions, Normal, log_interval_probability_scoped};
/// let law = Normal::<f64>::standard();
/// let token = CancellationToken::new();
/// let cx = ExecutionContext::sequential().with_cancellation(&token);
/// let log_p = log_interval_probability_scoped(&law, -1.0, 1.0, LawOptions::default(), cx)?;
/// assert!((log_p.exp() - 0.682_689_492_137_086).abs() < 1e-12);
/// token.cancel();
/// let error = log_interval_probability_scoped(&law, -1.0, 1.0, LawOptions::default(), cx);
/// assert_eq!(error.unwrap_err().kind(), SciOxideErrorKind::ExecutionStopped);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn log_interval_probability_scoped<
    'scope,
    'capability,
    S: RealScalar,
    D: crate::LogCdf<S> + crate::LogSurvival<S> + ?Sized,
>(
    law: &D,
    low: S,
    high: S,
    options: crate::LawOptions<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<S, SciOxideError> {
    options.validate()?;
    run_scoped(
        scope,
        scioxide_core::algorithm_id!("random", "LogIntervalProbability", 1),
        scioxide_core::SolveBudgets::unbounded()
            .with_limit(scioxide_core::BudgetKind::Evaluation, options.work_budget()),
        |context| interval(low, high, |x, upper| tail_scoped(law, x, upper, context)),
    )
}

pub(crate) fn interval_with_work<
    S: RealScalar,
    D: crate::LogCdf<S> + crate::LogSurvival<S> + ?Sized,
>(
    law: &D,
    low: S,
    high: S,
    left: &mut usize,
) -> Result<S, SciOxideError> {
    interval(low, high, |x, upper| {
        scoped(left, |context| {
            if upper {
                law.log_sf_scoped(x, context)
            } else {
                law.log_cdf_scoped(x, context)
            }
        })
    })
}

/// `ln P(low < X ≤ high)` from the logarithmic tails `tail(x, upper)`: `ln P(X > x)`
/// if `upper`, otherwise `ln P(X ≤ x)`. `tail` is only called at finite points.
fn interval<S: RealScalar>(
    low: S,
    high: S,
    mut tail: impl FnMut(S, bool) -> Result<S, SciOxideError>,
) -> Result<S, SciOxideError> {
    if low.is_nan() || high.is_nan() || low > high {
        return Err(invalid(
            scioxide_core::argument!("high"),
            "invalid probability interval",
        ));
    }
    if low == high {
        return Ok(S::NEG_INFINITY);
    }
    let mut log_tail = |x: S, upper: bool| {
        if x == S::NEG_INFINITY {
            Ok(if upper { S::ZERO } else { S::NEG_INFINITY })
        } else if x == S::INFINITY {
            Ok(if upper { S::NEG_INFINITY } else { S::ZERO })
        } else {
            tail(x, upper)
        }
    };
    // P(low < X <= high) can be formed as F(high) - F(low) or S(low) - S(high).
    // Subtract on whichever side the probabilities are small: when F(high) is
    // well below 1 the CDF difference has no cancellation, otherwise the
    // interval sits in the upper region and survival values keep precision.
    let high_cdf = log_tail(high, false)?;
    if high_cdf < -S::ONE {
        logdiff(
            high_cdf,
            log_tail(low, false)?,
            scioxide_core::argument!("law"),
        )
    } else {
        let low_sf = log_tail(low, true)?;
        logdiff(
            low_sf,
            log_tail(high, true)?,
            scioxide_core::argument!("law"),
        )
    }
}

// Borrowed providers retain exactly the same capabilities, without cloning.
impl<S: RealScalar, D: crate::Entropy<S> + ?Sized> crate::Entropy<S> for &D {
    fn entropy(&self) -> Result<S, SciOxideError> {
        D::entropy(self)
    }
    fn entropy_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        D::entropy_scoped(self, context)
    }
}
impl<S: RealScalar, X, D: crate::Cdf<S, X> + ?Sized> crate::Cdf<S, X> for &D {
    fn cdf(&self, x: X) -> Result<S, SciOxideError> {
        D::cdf(self, x)
    }
}
impl<S: RealScalar, X, D: crate::LogCdf<S, X> + ?Sized> crate::LogCdf<S, X> for &D {
    fn log_cdf(&self, x: X) -> Result<S, SciOxideError> {
        D::log_cdf(self, x)
    }

    fn log_cdf_scoped(
        &self,
        x: X,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        D::log_cdf_scoped(self, x, context)
    }
}
impl<S: RealScalar, X, D: crate::Survival<S, X> + ?Sized> crate::Survival<S, X> for &D {
    fn sf(&self, x: X) -> Result<S, SciOxideError> {
        D::sf(self, x)
    }
}
impl<S: RealScalar, X, D: crate::LogSurvival<S, X> + ?Sized> crate::LogSurvival<S, X> for &D {
    fn log_sf(&self, x: X) -> Result<S, SciOxideError> {
        D::log_sf(self, x)
    }

    fn log_sf_scoped(
        &self,
        x: X,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        D::log_sf_scoped(self, x, context)
    }
}
impl<S: RealScalar, X, D: crate::Mass<S, X> + ?Sized> crate::Mass<S, X> for &D {
    fn pmf(&self, x: X) -> Result<S, SciOxideError> {
        D::pmf(self, x)
    }
}
impl<S: RealScalar, X, D: crate::LogMass<S, X> + ?Sized> crate::LogMass<S, X> for &D {
    fn log_pmf(&self, x: X) -> Result<S, SciOxideError> {
        D::log_pmf(self, x)
    }
}
impl<S: RealScalar, X, D: crate::Density<S, X> + ?Sized> crate::Density<S, X> for &D {
    fn pdf(&self, x: X) -> Result<S, SciOxideError> {
        D::pdf(self, x)
    }
}
impl<S: RealScalar, X, D: crate::LogDensity<S, X> + ?Sized> crate::LogDensity<S, X> for &D {
    fn log_pdf(&self, x: X) -> Result<S, SciOxideError> {
        D::log_pdf(self, x)
    }

    fn log_pdf_scoped(
        &self,
        x: X,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        D::log_pdf_scoped(self, x, context)
    }
}
impl<S: RealScalar, Y, D: crate::Mean<S, Y> + ?Sized> crate::Mean<S, Y> for &D {
    fn mean(&self) -> Result<Y, SciOxideError> {
        D::mean(self)
    }

    fn mean_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Y, SciOxideError> {
        D::mean_scoped(self, context)
    }
}
impl<S: RealScalar, Y, D: crate::Variance<S, Y> + ?Sized> crate::Variance<S, Y> for &D {
    fn variance(&self) -> Result<Y, SciOxideError> {
        D::variance(self)
    }

    fn variance_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<Y, SciOxideError> {
        D::variance_scoped(self, context)
    }
}
impl<S: RealScalar, D: Support<S> + ?Sized> Support<S> for &D {
    fn support(&self) -> Result<ExtendedInterval<S>, SciOxideError> {
        D::support(self)
    }
    fn support_in(
        &self,
        interval: ExtendedInterval<S>,
    ) -> Result<ExtendedInterval<S>, SciOxideError> {
        D::support_in(self, interval)
    }
    fn contains(&self, x: S) -> Result<bool, SciOxideError> {
        D::contains(self, x)
    }
}
impl<S: RealScalar, D: crate::Distribution<S> + ?Sized> crate::Distribution<S> for &D {
    fn batch_byte_limit(&self, token: crate::traits::private::Token) -> usize {
        D::batch_byte_limit(self, token)
    }
    fn sample<R: crate::Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        D::sample(self, rng)
    }
    fn sample_scoped<R: crate::Rng<S>>(
        &self,
        rng: &mut R,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        D::sample_scoped(self, rng, context)
    }
    fn sample_n<R: crate::Rng<S>>(
        &self,
        rng: &mut R,
        n: usize,
    ) -> Result<alloc::vec::Vec<S>, SciOxideError> {
        D::sample_n(self, rng, n)
    }
    fn sample_n_into<R: crate::Rng<S>>(
        &self,
        rng: &mut R,
        policy: crate::SamplingPolicy<impl crate::StatePolicy<R>>,
        output: &mut [S],
        workspace: &mut crate::SamplingWorkspace<S>,
    ) -> Result<(), SciOxideError> {
        D::sample_n_into(self, rng, policy, output, workspace)
    }
}

impl<D: crate::CountDistribution + ?Sized> crate::CountDistribution for &D {
    type Count = D::Count;
    fn sample<R: crate::BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
    ) -> Result<Self::Count, SciOxideError> {
        D::sample(self, rng)
    }
    fn sample_n<R: crate::BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        n: usize,
    ) -> Result<alloc::vec::Vec<Self::Count>, SciOxideError> {
        D::sample_n(self, rng, n)
    }
    fn sample_n_into<R: crate::BitGenerator + ?Sized>(
        &self,
        rng: &mut R,
        policy: crate::SamplingPolicy<impl crate::StatePolicy<R>>,
        output: &mut [Self::Count],
        workspace: &mut crate::SamplingWorkspace<Self::Count>,
    ) -> Result<(), SciOxideError> {
        D::sample_n_into(self, rng, policy, output, workspace)
    }
}
impl<D: crate::CountSupport + ?Sized> crate::CountSupport for &D {
    fn count_support(&self) -> (usize, Option<usize>) {
        D::count_support(self)
    }
}
impl<D: crate::SignedCountSupport + ?Sized> crate::SignedCountSupport for &D {
    fn signed_count_support(&self) -> (Option<i64>, Option<i64>) {
        D::signed_count_support(self)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod policy_tests {
    use super::*;
    #[test]
    fn positive_draw_checks_the_final_destination_range() {
        use scioxide_core::{NumericalIssue, SciOxideErrorView};
        for (log, issue) in [
            (-2000.0, NumericalIssue::Underflow),
            (2000.0, NumericalIssue::Overflow),
            (f64::NAN, NumericalIssue::Breakdown),
        ] {
            assert!(matches!(positive_draw::<f64>(log).unwrap_err().view(),
                SciOxideErrorView::NumericalInstability { issue: actual, .. } if actual == issue));
        }
        assert_eq!(positive_draw::<f32>(0.0).unwrap(), 1.0);
        assert!(
            (positive_draw::<f64>(f64::MIN_POSITIVE.ln()).unwrap() / f64::MIN_POSITIVE - 1.0).abs()
                < 1e-13
        );
        for (log, issue) in [
            (-200.0, NumericalIssue::Underflow),
            (200.0, NumericalIssue::Overflow),
        ] {
            assert!(positive_draw::<f64>(log).unwrap() > 0.0);
            assert!(matches!(positive_draw::<f32>(log).unwrap_err().view(),
                SciOxideErrorView::NumericalInstability { issue: actual, .. } if actual == issue));
        }
    }
    fn log_probability_boundaries<S: RealScalar>() {
        let limit = constant::<S>(32.).unwrap() * S::EPSILON;
        assert_eq!(log_probability(limit).unwrap(), S::ZERO);
        assert!(log_probability(limit.next_after(S::INFINITY)).is_err());
        assert_eq!(log_probability(S::NEG_INFINITY).unwrap(), S::NEG_INFINITY);
        assert_eq!(log_probability(-S::ONE).unwrap(), -S::ONE);
        assert!(log_probability(S::NAN).is_err());
        assert!(log_probability(S::INFINITY).is_err());
        assert_eq!(
            logadd(S::NEG_INFINITY, S::NEG_INFINITY).unwrap(),
            S::NEG_INFINITY
        );
        assert_eq!(logadd(-S::ONE, S::NEG_INFINITY).unwrap(), -S::ONE);
    }
    #[test]
    fn working_precision_log_probability_policy() {
        log_probability_boundaries::<f32>();
        log_probability_boundaries::<f64>();
    }

    // Count tail evaluations of a law whose lower tail is `lower`.
    fn switched(lower: f64, first_upper: bool, upper: bool) -> (f64, f64, usize) {
        let mut calls = 0;
        let (value, computed) = smaller_tail_first(first_upper, upper, |tail_upper| {
            calls += 1;
            Ok(if tail_upper {
                log1mexp(lower.ln())?
            } else {
                lower.ln()
            })
        })
        .unwrap();
        (value, computed, calls)
    }

    #[test]
    fn complement_switch_starts_at_fifteen_sixteenths() {
        // Between 1/2 and 15/16 the requested tail is kept: one evaluation.
        let (value, computed, calls) = switched(0.6, false, false);
        assert_eq!((calls, value, computed), (1, 0.6_f64.ln(), 0.6_f64.ln()));
        // Above 15/16 the complementary tail is evaluated and complemented.
        let (value, computed, calls) = switched(0.95, false, false);
        assert_eq!(calls, 2);
        assert!((computed - 0.05_f64.ln()).abs() < 1e-15);
        assert!((value - 0.95_f64.ln()).abs() < 1e-15);
        // A median-excluding first tail answers the other side by complement.
        let (value, computed, calls) = switched(0.3, false, true);
        assert_eq!((calls, computed), (1, 0.3_f64.ln()));
        assert!((value - 0.7_f64.ln()).abs() < 1e-15);
    }

    #[test]
    fn near_one_overshoot_is_replaced_before_validation() {
        // An absolutely accurate near-one estimate may exceed zero; its
        // complementary kernel is evaluated instead of rejecting it.
        let mut calls = 0;
        let (value, _) = smaller_tail_first(false, false, |upper| {
            calls += 1;
            Ok(if upper { 1e-20_f64.ln() } else { 1e-10 })
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert!(value <= 0.0 && value > -1e-19);
        assert!(smaller_tail_first(false, false, |_| Ok(f64::NAN)).is_err());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod computed_value_tests {
    use super::*;
    use scioxide_core::NumericalIssue;
    #[test]
    fn positive_computed_normalizers_and_draws_retain_the_failure_kind() {
        for value in [0.0_f64, -0.0] {
            crate::error_assertions::numerical(
                positive_result(value).unwrap_err(),
                NumericalIssue::Underflow,
            );
        }
        for value in [-1.0_f64, f64::NAN] {
            crate::error_assertions::numerical(
                positive_result(value).unwrap_err(),
                NumericalIssue::Breakdown,
            );
        }
        for value in [f64::INFINITY, f64::NEG_INFINITY] {
            crate::error_assertions::numerical(
                positive_result(value).unwrap_err(),
                NumericalIssue::Overflow,
            );
        }
        assert_eq!(
            positive_result(f64::from_bits(1)).unwrap(),
            f64::from_bits(1)
        );
        assert_eq!(positive_result(1.0_f32).unwrap(), 1.0);
    }
}

impl<S: RealScalar, X, D: crate::Quantile<S, X> + ?Sized> crate::Quantile<S, X> for &D {
    fn quantile(&self, p: S) -> Result<X, SciOxideError> {
        D::quantile(self, p)
    }
    fn quantile_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<X, SciOxideError> {
        D::quantile_scoped(self, p, context)
    }
}

impl<S: RealScalar, X, D: crate::InverseSurvival<S, X> + ?Sized> crate::InverseSurvival<S, X>
    for &D
{
    fn isf(&self, p: S) -> Result<X, SciOxideError> {
        D::isf(self, p)
    }
    fn isf_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<X, SciOxideError> {
        D::isf_scoped(self, p, context)
    }
}
