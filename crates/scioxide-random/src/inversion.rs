//! Numerical inversion of monotone log probabilities by bracketing and bisection.
use crate::law::{exhausted, invalid, probability};
use crate::{LogCdf, LogSurvival, Support};
use core::cell::{Cell, RefCell};
use scioxide_core::{
    Budget, ExtendedInterval, Interval, RealScalar, ResidualTolerance, SciOxideError, StepTolerance,
};
use scioxide_rootfind::{
    Bisection, BracketSearchDirection, BracketSearchInput, BracketSearchOptions, RootResult,
    RootTermination, ScalarFn, ScalarRootOptions, ScalarRootSolver, find_bracket_with_options,
};

/// Work limit and stopping rule for [`invert_log_probability_with_options`] and
/// [`invert_log_probability_scoped`].
///
/// The defaults allow 4,096 iterations and 65,536 work units, the default
/// [`LawOptions::work_budget`](crate::LawOptions::work_budget) of one law operation, with an
/// absolute and relative bracket tolerance of machine epsilon. Every tail evaluation of an
/// iterative law, such as the continued fraction of a beta tail, costs many work units, so the
/// work limit is that of a law operation rather than the iteration limit. Built-in laws do not use
/// these options for their own quantiles; they use their [`LawOptions`](crate::LawOptions).
///
/// ```
/// use scioxide_core::{Budget, StepTolerance};
/// use scioxide_random::{InversionOptions, Normal, Quantile, invert_log_probability_with_options};
/// use scioxide_rootfind::ScalarRootOptions;
/// let root = ScalarRootOptions::default().with_step_tolerance(StepTolerance::new(1e-10, 0.0)?);
/// let options = InversionOptions::default()
///     .with_root_options(root)
///     .with_work_budget(Budget::try_from(200)?);
/// let law = Normal::<f64>::standard();
/// let report = invert_log_probability_with_options(&law, 0.9_f64.ln(), false, &options)?;
/// assert!(report.root().is_success() && report.evaluations() <= 200);
/// assert!((report.value() - law.quantile(0.9)?).abs() < 1e-9);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct InversionOptions<S: RealScalar> {
    root: ScalarRootOptions<S>,
    evaluations: Budget,
}
impl<S: RealScalar> InversionOptions<S> {
    /// Returns the root-finding options.
    ///
    /// Inversion uses their iteration limit and step tolerance. It ignores the
    /// residual tolerance, since the residual only records on which side of the
    /// target probability a point lies.
    pub fn root_options(&self) -> &ScalarRootOptions<S> {
        &self.root
    }
    /// Sets the root-finding options.
    #[must_use]
    pub fn with_root_options(mut self, value: ScalarRootOptions<S>) -> Self {
        self.root = value;
        self
    }
    /// Returns the work limit, shared by bracket search, bisection and the law's tail evaluations.
    pub fn work_budget(&self) -> Budget {
        self.evaluations
    }
    /// Sets the work limit.
    #[must_use]
    pub fn with_work_budget(mut self, value: Budget) -> Self {
        self.evaluations = value;
        self
    }
}
impl<S: RealScalar> Default for InversionOptions<S> {
    fn default() -> Self {
        Self {
            root: ScalarRootOptions::default()
                .with_max_iterations(Budget::ONE.saturating_add(4095))
                .with_step_tolerance(StepTolerance::from_tolerance(
                    scioxide_core::Tolerance::from_machine_epsilon(),
                )),
            evaluations: Budget::ONE.saturating_add(65_535),
        }
    }
}

/// The one mapping of an unsuccessful root solve after `used` work units:
/// caller stops are execution stops, iteration and evaluation limits and
/// stagnation keep their reason, and any other failure is a nested-solver
/// failure.
pub(crate) fn root_failure(termination: RootTermination, used: usize) -> SciOxideError {
    use scioxide_core::{ConvergenceReason, ExecutionStopReason};
    let reason = match termination {
        RootTermination::Cancelled => {
            return SciOxideError::execution_stopped(ExecutionStopReason::Cancelled);
        }
        RootTermination::TimeLimit => {
            return SciOxideError::execution_stopped(ExecutionStopReason::DeadlineReached);
        }
        RootTermination::IterationLimit
        | RootTermination::WorkLimit(scioxide_core::BudgetKind::Iteration) => {
            ConvergenceReason::IterationLimit
        }
        RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation) => {
            ConvergenceReason::EvaluationLimit
        }
        RootTermination::WorkLimit(kind) => ConvergenceReason::ResourceLimit(kind),
        RootTermination::Stagnation => ConvergenceReason::Stagnation,
        _ => ConvergenceReason::NestedSolver,
    };
    SciOxideError::convergence_failure(
        reason,
        scioxide_core::ConvergenceDiagnostics::new()
            .with_counts(scioxide_core::EvaluationCounts::from_iterations(used).into()),
        "numerical inversion did not converge",
    )
}
/// Result of [`invert_log_probability_with_options`] and [`invert_log_probability_scoped`]:
/// the inverse, the root-finding result and the work done.
///
/// Check [`RootResult::is_success`] on [`Self::root`] before using
/// [`Self::value`]. An upper-tail inversion resolves probabilities whose
/// complement rounds to one:
///
/// ```
/// use scioxide_random::{
///     InverseSurvival, InversionOptions, Normal, invert_log_probability_with_options,
/// };
/// let law = Normal::<f64>::standard();
/// let log_p = 1e-300_f64.ln();
/// let options = InversionOptions::default();
/// let report = invert_log_probability_with_options(&law, log_p, true, &options)?;
/// assert!(report.root().is_success() && report.evaluations() > 0);
/// assert!((report.value() - law.isf(1e-300)?).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct InversionReport<S: RealScalar> {
    root: RootResult<S>,
    evaluations: usize,
}
impl<S: RealScalar> InversionReport<S> {
    /// Returns the upper end of the final bracket, the generalized inverse.
    pub fn value(&self) -> S {
        // The residual is a step function, so the root is a jump location.
        // The bracket's upper end is the side that satisfies the probability
        // (F(x) >= p), which keeps quantiles right-continuous and consistent.
        self.root.bracket().map_or(self.root.point(), |b| b.upper())
    }
    /// Returns the root-finding result: bracket, residual and termination status.
    pub fn root(&self) -> &RootResult<S> {
        &self.root
    }
    /// Returns the number of tail evaluations, bracket search included.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }
}
/// Returns the point `x` with `ln F(x) = log_p` (or `ln P(X > x) = log_p` if `upper`), for
/// finite `log_p < 0`.
///
/// The method is bisection on the sign of `ln F(x) − log_p` (or of the upper
/// tail), after a bracket search over the support. It works for any law with
/// logarithmic tails and a support, including laws with gaps. Where the CDF is
/// constant at the target level, the lower inversion returns the left end of
/// that interval and the upper inversion the right end. No probability is
/// formed by subtraction, so `log_p` may be far below the smallest float
/// probability. The result is the upper end of a final bracket of width about
/// machine epsilon relative, found within 65,536 work units. It equals
/// [`InversionReport::value`] of [`invert_log_probability_with_options`] with the default
/// [`InversionOptions`].
///
/// # Errors
///
/// - The errors of [`invert_log_probability_with_options`].
/// - `ConvergenceFailure` if the root solve does not succeed, with its termination reason;
///   `ExecutionStopped` for a cancellation or deadline seen by the solve.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cdf, Normal, invert_log_probability};
/// let law = Normal::<f64>::standard();
/// let x = invert_log_probability(&law, 0.25_f64.ln(), false)?;
/// assert!((law.cdf(x)? - 0.25).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn invert_log_probability<
    S: RealScalar,
    D: LogCdf<S> + LogSurvival<S> + Support<S> + ?Sized,
>(
    law: &D,
    log_p: S,
    upper: bool,
) -> Result<S, SciOxideError> {
    report_value(&invert_log_probability_with_options(
        law,
        log_p,
        upper,
        &InversionOptions::default(),
    )?)
}

/// Inverts `ln F(x) = log_p` (or `ln P(X > x) = log_p` if `upper`) as
/// [`invert_log_probability`] does, with the work limit and stopping rule of `options`, and
/// reports the root solve.
///
/// The bracket width, set by [`InversionOptions::root_options`], controls the accuracy of
/// the result. Every tail evaluation, with the work the law does inside it, counts toward
/// [`InversionOptions::work_budget`]. The function runs on the calling thread.
///
/// # Errors
///
/// - `InvalidInput` if `log_p` is not finite and negative, or the support is
///   empty.
/// - Any error of the law's tail functions, or an invalid log probability
///   returned by them.
/// - `ConvergenceFailure` if the work limit is exhausted.
/// - Returns an error if no finite bracket can be found.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cdf, InversionOptions, Normal, invert_log_probability_with_options};
/// let law = Normal::<f64>::standard();
/// let options = InversionOptions::default();
/// let report = invert_log_probability_with_options(&law, 0.25_f64.ln(), false, &options)?;
/// assert!(report.root().is_success());
/// assert!((law.cdf(report.value())? - 0.25).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn invert_log_probability_with_options<
    S: RealScalar,
    D: LogCdf<S> + LogSurvival<S> + Support<S> + ?Sized,
>(
    law: &D,
    log_p: S,
    upper: bool,
    options: &InversionOptions<S>,
) -> Result<InversionReport<S>, SciOxideError> {
    invert_log_probability_scoped(
        law,
        log_p,
        upper,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}

/// Inverts a logarithmic tail as [`invert_log_probability_with_options`] does, as one epoch
/// of `scope`.
///
/// `scope` is an [`ExecutionContext`](scioxide_core::ExecutionContext) for a top-level call or
/// the `&mut SolveContext` of an enclosing solve. The call observes the cancellation and
/// deadline of the scope before each tail evaluation. The work of the law's tail evaluations
/// counts in the epoch, within [`InversionOptions::work_budget`], and in every enclosing
/// solve, so an enclosing evaluation limit also bounds it. With a sequential context the
/// report equals that of [`invert_log_probability_with_options`] bit for bit.
///
/// # Errors
///
/// - The errors of [`invert_log_probability_with_options`]; `ConvergenceFailure` also if an
///   enclosing evaluation limit is exhausted.
/// - `ExecutionStopped` if the scope is cancelled or past its deadline before a tail
///   evaluation.
/// - `AccountingInvariant` if the counts of an enclosing solve are inconsistent.
///
/// # Examples
///
/// ```
/// use scioxide_core::{EvaluationRole as Role, ExecutionContext, ProblemToken, SolveBudgets};
/// use scioxide_core::{SolveIdentity, SolveScope, SolveState, algorithm_id};
/// use scioxide_random::{InversionOptions, Normal, invert_log_probability_scoped};
/// let (law, options) = (Normal::<f64>::standard(), InversionOptions::default());
/// let id = SolveIdentity::new(ProblemToken::new(1), algorithm_id!("app", "outer", 1), 0, 0);
/// let scope = SolveScope::from(ExecutionContext::sequential());
/// scope.run(&mut SolveState::new(), id, SolveBudgets::unbounded(), |parent| {
///     let half = 0.5_f64.ln();
///     let report = invert_log_probability_scoped(&law, half, false, &options, &mut *parent)?;
///     assert!(report.value().abs() < 1e-12);
///     // Each tail evaluation of the law counts in the enclosing solve.
///     assert_eq!(parent.provider_count(Role::Function)?, report.evaluations());
///     Ok(())
/// })?;
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn invert_log_probability_scoped<
    'scope,
    'capability,
    S: RealScalar,
    D: LogCdf<S> + LogSurvival<S> + Support<S> + ?Sized,
>(
    law: &D,
    log_p: S,
    upper: bool,
    options: &InversionOptions<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<InversionReport<S>, SciOxideError> {
    crate::law::run_scoped(
        scope,
        scioxide_core::algorithm_id!("random", "InvertLogProbability", 1),
        scioxide_core::SolveBudgets::unbounded()
            .with_limit(scioxide_core::BudgetKind::Evaluation, options.evaluations),
        |context| {
            invert_tail(
                law.support()?,
                log_p,
                upper,
                upper,
                false,
                options,
                |x, upper| crate::law::tail_scoped(law, x, upper, context),
            )
        },
    )
}

/// The value of a successful inversion, or the failure of its root solve.
pub(crate) fn report_value<S: RealScalar>(report: &InversionReport<S>) -> Result<S, SciOxideError> {
    if report.root.is_success() {
        Ok(report.value())
    } else {
        Err(root_failure(report.root.termination(), report.evaluations))
    }
}

#[allow(clippy::too_many_arguments)]
fn invert_tail<S: RealScalar>(
    support: ExtendedInterval<S>,
    log_p: S,
    upper: bool,
    right_edge: bool,
    smooth: bool,
    options: &InversionOptions<S>,
    tail: impl FnMut(S, bool) -> Result<S, SciOxideError>,
) -> Result<InversionReport<S>, SciOxideError> {
    if !log_p.is_finite() || log_p >= S::ZERO {
        return Err(invalid(
            scioxide_core::argument!("log_p"),
            "generic inversion requires a finite negative log probability",
        ));
    }
    let calls = Cell::new(0usize);
    let tail = RefCell::new(tail);
    let residual = |x: S| {
        if calls.get() >= options.evaluations.get() {
            return Err(exhausted());
        }
        calls.set(calls.get() + 1);
        let actual = crate::law::log_probability(tail.borrow_mut()(x, upper)?)?;
        if smooth {
            return Ok((if upper {
                log_p - actual
            } else {
                actual - log_p
            })
            .max(-S::MAX)
            .min(S::MAX));
        }
        // The residual is only the sign of the comparison. Bisection on a sign
        // function needs no derivative and is immune to flat or noisy log
        // tails, and it compares logarithms directly, so probabilities far
        // below the scalar's normal range are still resolved.
        let below = if upper {
            actual > log_p || (right_edge && actual == log_p)
        } else {
            actual < log_p || (right_edge && actual == log_p)
        };
        Ok(if below { -S::ONE } else { S::ONE })
    };
    let (lower, upper_bound) = support
        .bounds()
        .ok_or_else(|| invalid(scioxide_core::argument!("law"), "empty inversion support"))?;
    let mut a = lower.unwrap_or(-S::ONE);
    let mut b = upper_bound.unwrap_or(S::ONE);
    if a >= b {
        if upper_bound.is_none() {
            b = a + S::ONE.max(a.abs());
        } else if lower.is_none() {
            a = b - S::ONE.max(b.abs());
        }
    }
    let root = solve_monotone(
        residual,
        (a, b),
        (lower.unwrap_or(-S::MAX), upper_bound.unwrap_or(S::MAX)),
        &if smooth {
            options.root.clone()
        } else {
            options
                .root
                .clone()
                .with_residual_tolerance(ResidualTolerance::new(S::ZERO, S::ZERO)?)
        },
        Bisection,
    )?;
    Ok(InversionReport {
        root,
        evaluations: calls.get(),
    })
}
/// Threshold inverses resolve the coordinate lattice; their work policy bounds the
/// entire query, while the public inversion entry point accepts explicit root options.
pub(crate) fn law_options<S: RealScalar>(
    work: Budget,
) -> Result<InversionOptions<S>, SciOxideError> {
    Ok(InversionOptions::default()
        .with_root_options(
            ScalarRootOptions::default()
                .with_max_iterations(work)
                .with_step_tolerance(StepTolerance::new(S::ZERO, S::EPSILON)?),
        )
        .with_work_budget(work))
}

/// Smooth-inversion options resolving the logarithmic tail to eight units of
/// its magnitude, the accuracy of a converged Newton step.
pub(crate) fn smooth_options(
    work: Budget,
    log_p: f64,
) -> Result<InversionOptions<f64>, SciOxideError> {
    let options = law_options::<f64>(work)?;
    let root = options
        .root
        .clone()
        .with_residual_tolerance(scioxide_core::ResidualTolerance::new(
            8.0 * f64::EPSILON * (1.0 + log_p.abs()),
            0.0,
        )?);
    Ok(options.with_root_options(root))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn inverse_remaining<S: RealScalar>(
    support: ExtendedInterval<S>,
    p: S,
    upper: bool,
    options: &InversionOptions<S>,
    tolerance: Option<scioxide_core::Tolerance<S>>,
    left: &mut usize,
    tail: impl Fn(S, bool, &mut usize) -> Result<S, SciOxideError>,
) -> Result<S, SciOxideError> {
    probability(p, scioxide_core::argument!("p"))?;
    let (low, high) = support
        .bounds()
        .ok_or_else(|| invalid(scioxide_core::argument!("law"), "empty inversion support"))?;
    if p == S::ZERO || p == S::ONE {
        return Ok(if (p == S::ZERO) != upper {
            low.unwrap_or(S::NEG_INFINITY)
        } else {
            high.unwrap_or(S::INFINITY)
        });
    }
    let allowance = *left;
    let left = RefCell::new(left);
    let options = if let Some(tolerance) = tolerance {
        InversionOptions::default()
            .with_root_options(options.root.clone().with_residual_tolerance(
                ResidualTolerance::new(
                    tolerance.relative() * crate::law::constant::<S>(0.125)?,
                    S::ZERO,
                )?,
            ))
            .with_work_budget(options.evaluations)
    } else {
        options.clone()
    };
    let report = invert_tail(
        support,
        if p > S::HALF { (-p).ln_1p() } else { p.ln() },
        upper != (p > S::HALF),
        upper,
        tolerance.is_some(),
        &options,
        |x, upper| {
            let mut left = left.borrow_mut();
            crate::law::charge(&mut left)?;
            tail(x, upper, &mut left)
        },
    )?;
    if !report.root.is_success() {
        return Err(root_failure(
            report.root.termination(),
            allowance - **left.borrow(),
        ));
    }
    let value = if tolerance.is_some() {
        report.root.point()
    } else {
        report.value()
    };
    let touches_lower =
        low.is_some_and(|low| report.root.bracket().is_some_and(|b| b.lower() == low));
    let unresolved = if let Some(tolerance) = tolerance {
        let touches_upper =
            high.is_some_and(|high| report.root.bracket().is_some_and(|b| b.upper() == high));
        (touches_lower || touches_upper)
            && report.root.residual_report().raw_norm()
                > tolerance.relative() * crate::law::constant::<S>(0.125)?
    } else {
        touches_lower
    };
    if low == Some(value) || high == Some(value) || unresolved {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Underflow,
            "interior quantile cannot be resolved at a support endpoint",
        ));
    }
    Ok(value)
}

/// Bracketed solve adapter, sharing bracket discovery with smooth inversion.
/// Coordinate transforms and residuals are supplied by the law; all solver work
/// flows through the same charged residual. Bounds include representability limits.
pub(crate) fn solve_monotone<S, F, R>(
    residual: F,
    initial: (S, S),
    bounds: (S, S),
    options: &ScalarRootOptions<S>,
    solver: R,
) -> Result<RootResult<S>, SciOxideError>
where
    S: RealScalar,
    F: Fn(S) -> Result<S, SciOxideError>,
    R: ScalarRootSolver<S, Initialization = Interval<S>>,
{
    let function = ScalarFn::new(residual);
    let bracket = find_bracket_with_options(
        &function,
        BracketSearchInput::Pair(initial.0.max(bounds.0), initial.1.min(bounds.1)),
        BracketSearchDirection::Increasing,
        &bracket_options(bounds),
    )
    .map_err(bracket_failure)?;
    solver.solve(&function, bracket.interval(), options)
}

/// Keep coordinate-range failures distinct from work exhaustion.
fn bracket_failure(error: SciOxideError) -> SciOxideError {
    if matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::Stagnation,
            ..
        }
    ) {
        SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "inverse exceeds its coordinate range",
        )
    } else {
        error
    }
}
fn bracket_options<S: RealScalar>(bounds: (S, S)) -> BracketSearchOptions<S> {
    use core::ops::Bound::{Included, Unbounded};
    // Law coordinates may use infinite representability limits. Rootfind's
    // absent bounds use Unbounded, keeping every actual callback point finite.
    BracketSearchOptions::default().with_domain(
        if bounds.0 == S::NEG_INFINITY {
            Unbounded
        } else {
            Included(bounds.0)
        },
        if bounds.1 == S::INFINITY {
            Unbounded
        } else {
            Included(bounds.1)
        },
    )
}

/// Smooth log-tail inversion in a law-supplied coordinate. The density callback
/// includes its Jacobian. Only strictly continuous laws use this path; threshold
/// inversion keeps the edge conventions needed by distributions with gaps.
/// Tail and derivative calls, including their nested work, share one budget.
#[allow(clippy::too_many_arguments)]
pub(crate) fn invert_smooth_remaining(
    log_p: f64,
    upper: bool,
    guess: f64,
    bounds: (f64, f64),
    options: &InversionOptions<f64>,
    tail: impl Fn(f64, bool, &mut usize) -> Result<f64, SciOxideError>,
    log_density: impl Fn(f64, &mut usize) -> Result<f64, SciOxideError>,
    left: &mut usize,
) -> Result<f64, SciOxideError> {
    use scioxide_differentiate::DerivativeStrategy;
    use scioxide_rootfind::{SafeguardedInitialGuess, SafeguardedNewton};

    let allowance = *left;
    let left = RefCell::new(left);
    // The bracket ends and the start are evaluated before the solver sees
    // them; a few remembered points spare their recomputation.
    let cached = Cell::new([None::<(f64, f64)>; 4]);
    let value = |x| {
        let mut points = cached.get();
        if let Some((_, value)) = points.iter().flatten().find(|(at, _)| *at == x) {
            return Ok(*value);
        }
        let mut left = left.borrow_mut();
        crate::law::charge(&mut left)?;
        let value = crate::law::log_probability(tail(x, upper, &mut left)?)?;
        points.rotate_right(1);
        points[0] = Some((x, value));
        cached.set(points);
        Ok::<_, SciOxideError>(value)
    };
    let residual = |x| {
        let delta = value(x)? - log_p;
        // Logarithmic zero is legitimate away from the root. A finite saturation
        // supplies its sign to the shared solver without forming a probability.
        Ok((if upper { -delta } else { delta }).clamp(-f64::MAX, f64::MAX))
    };
    let derivative = |x| {
        let probability = value(x)?;
        let mut left = left.borrow_mut();
        crate::law::charge(&mut left)?;
        let density = log_density(x, &mut left)?;
        if density.is_nan() || density == f64::INFINITY {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "inverse density is not representable",
            ));
        }
        let derivative = (density - probability).exp();
        // An unrepresentable log-tail slope cannot provide a Newton step;
        // a zero derivative asks the safeguarded solver to bisect instead.
        Ok(if derivative.is_finite() {
            derivative
        } else {
            0.0
        })
    };
    let guess = guess.clamp(bounds.0, bounds.1);
    let at_guess = residual(guess)?;
    // Laws retain their derivative-guided seeds; rootfind owns every expansion.
    let seed = if at_guess == 0.0 {
        guess
    } else {
        let slope = derivative(guess)?;
        let step = -at_guess / slope;
        if slope > 0.0 && step.is_finite() {
            (guess + step * 1.0625).clamp(bounds.0, bounds.1)
        } else {
            guess
        }
    };
    let input = if at_guess == 0.0 || seed != guess {
        BracketSearchInput::Pair(guess.min(seed), guess.max(seed))
    } else {
        BracketSearchInput::Pair((guess - 1.0).max(bounds.0), (guess + 1.0).min(bounds.1))
    };
    let bracket = find_bracket_with_options(
        &ScalarFn::new(&residual),
        input,
        BracketSearchDirection::Increasing,
        &bracket_options(bounds),
    )
    .map_err(bracket_failure)?;
    let start = if bracket.lower_value().abs() < bracket.upper_value().abs() {
        bracket.lower()
    } else {
        bracket.upper()
    };
    let function = ScalarFn::new(residual).with_derivative(derivative);
    let root = SafeguardedNewton::new(DerivativeStrategy::Analytic).solve(
        &function,
        SafeguardedInitialGuess::new(bracket.interval(), start)?,
        &options.root,
    )?;
    if !root.is_success() {
        return Err(root_failure(
            root.termination(),
            allowance - **left.borrow(),
        ));
    }
    Ok(root.point())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    #[test]
    fn root_failures_keep_their_termination_reason() {
        use scioxide_core::{ConvergenceReason, ExecutionStopReason, SciOxideErrorView};
        for (termination, expected) in [
            (
                RootTermination::IterationLimit,
                ConvergenceReason::IterationLimit,
            ),
            (
                RootTermination::WorkLimit(scioxide_core::BudgetKind::Proposal),
                ConvergenceReason::ResourceLimit(scioxide_core::BudgetKind::Proposal),
            ),
            (
                RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation),
                ConvergenceReason::EvaluationLimit,
            ),
            (RootTermination::Stagnation, ConvergenceReason::Stagnation),
            (
                RootTermination::ComplexStepRequired,
                ConvergenceReason::NestedSolver,
            ),
        ] {
            let error = root_failure(termination, 7);
            let SciOxideErrorView::ConvergenceFailure {
                reason,
                diagnostics,
                ..
            } = error.view()
            else {
                unreachable!("{error:?}");
            };
            assert_eq!(reason, expected);
            assert_eq!(diagnostics.counts().own().iterations(), 7);
        }
        for (termination, expected) in [
            (RootTermination::Cancelled, ExecutionStopReason::Cancelled),
            (
                RootTermination::TimeLimit,
                ExecutionStopReason::DeadlineReached,
            ),
        ] {
            assert!(matches!(
                root_failure(termination, 0).view(),
                SciOxideErrorView::ExecutionStopped { reason, .. } if reason == expected
            ));
        }
    }
    #[test]
    fn unsuccessful_reports_are_root_failures() {
        let normal = crate::Normal::<f64>::standard();
        let options = InversionOptions::default()
            .with_root_options(ScalarRootOptions::default().with_max_iterations(Budget::ONE));
        let report =
            invert_log_probability_with_options(&normal, 0.25_f64.ln(), false, &options).unwrap();
        assert!(!report.root().is_success());
        let error = report_value(&report).unwrap_err();
        let scioxide_core::SciOxideErrorView::ConvergenceFailure { reason, .. } = error.view()
        else {
            unreachable!("{error:?}");
        };
        assert_eq!(reason, scioxide_core::ConvergenceReason::IterationLimit);
        let report =
            invert_log_probability_with_options(&normal, 0.25_f64.ln(), false, &Default::default())
                .unwrap();
        assert_eq!(
            report_value(&report).unwrap().to_bits(),
            report.value().to_bits()
        );
    }
    struct Gap;
    impl Support<f64> for Gap {
        fn support(&self) -> Result<scioxide_core::ExtendedInterval<f64>, SciOxideError> {
            scioxide_core::ExtendedInterval::new(Some(0.0), Some(3.0))
        }
    }
    impl LogCdf<f64> for Gap {
        fn log_cdf(&self, x: f64) -> Result<f64, SciOxideError> {
            Ok((if x < 1.0 {
                0.75 * x.max(0.0)
            } else if x < 2.0 {
                0.75
            } else {
                0.75 + 0.25 * (x.min(3.0) - 2.0)
            })
            .ln())
        }
    }
    impl LogSurvival<f64> for Gap {
        fn log_sf(&self, x: f64) -> Result<f64, SciOxideError> {
            Ok((if x < 1.0 {
                0.25 + 0.75 * (1.0 - x.max(0.0))
            } else if x < 2.0 {
                0.25
            } else {
                0.25 * (3.0 - x.min(3.0))
            })
            .ln())
        }
    }
    // Lattice inversion of the scoped logarithmic tails within a fresh budget.
    fn inverse(law: &Gap, p: f64, upper: bool) -> Result<f64, SciOxideError> {
        let options = law_options(crate::LawOptions::<f64>::default().work_budget())?;
        inverse_remaining(
            law.support()?,
            p,
            upper,
            &options,
            None,
            &mut options.evaluations.get(),
            |x, upper, left| {
                crate::law::scoped(left, |context| {
                    if upper {
                        law.log_sf_scoped(x, context)
                    } else {
                        law.log_cdf_scoped(x, context)
                    }
                })
            },
        )
    }
    #[test]
    fn switched_tail_retains_gap_edge() {
        assert!((inverse(&Gap, 0.75, false).unwrap() - 1.0).abs() <= 4.0 * f64::EPSILON);
        assert!((inverse(&Gap, 0.25, true).unwrap() - 2.0).abs() <= 4.0 * f64::EPSILON);
    }

    #[test]
    fn nested_tail_work_cannot_reset_the_inversion_budget() {
        let nested = Cell::new(0);
        let error = inverse_remaining(
            ExtendedInterval::new(Some(0.), Some(1.)).unwrap(),
            0.3,
            false,
            &law_options(Budget::try_from(255).unwrap()).unwrap(),
            None,
            &mut 255,
            |x: f64, upper, left| {
                for _ in 0..100 {
                    crate::law::charge(left)?;
                    nested.set(nested.get() + 1);
                }
                Ok(if upper { (-x).ln_1p() } else { x.ln() })
            },
        )
        .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        // Three outer tail calls plus 252 admitted nested units exhaust exactly 255.
        assert_eq!(nested.get(), 252);
    }

    #[test]
    fn smooth_inversion_safeguards_an_unusable_derivative() {
        // Logistic CDF with a deliberately unavailable (zero) Newton slope.
        // A remote seed must still converge through bracket contraction.
        let log_p = 0.01_f64.ln();
        let mut options = InversionOptions::default();
        options.root = options
            .root
            .with_residual_tolerance(ResidualTolerance::new(0.0, 0.0).unwrap());
        let x = invert_smooth_remaining(
            log_p,
            false,
            12.0,
            (-32.0, 32.0),
            &options,
            |x, _, _| Ok(-(1.0 + (-x).exp()).ln()),
            |_, _| Ok(f64::NEG_INFINITY),
            &mut options.evaluations.get(),
        )
        .unwrap();
        assert!((x - (0.01_f64 / 0.99).ln()).abs() < 1e-12);
    }

    #[test]
    fn smooth_inversion_cannot_restart_nested_work() {
        let nested = Cell::new(0);
        let error = invert_smooth_remaining(
            0.3_f64.ln(),
            false,
            0.0,
            (-10.0, 10.0),
            &law_options(Budget::try_from(5).unwrap()).unwrap(),
            |x, _, left| {
                for _ in 0..2 {
                    crate::law::charge(left)?;
                    nested.set(nested.get() + 1);
                }
                Ok(-(1.0 + (-x).exp()).ln())
            },
            |_, _| Ok(0.0),
            &mut 5,
        )
        .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::ConvergenceFailure
        );
        // The start's tail (one outer and two nested units), its derivative and
        // the next outer admission consume the entire budget.
        assert_eq!(nested.get(), 2);
    }
}
