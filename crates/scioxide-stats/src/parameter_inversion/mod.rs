//! Solves tail-probability equations for a distribution parameter.

use crate::error_helpers::invalid;
use scioxide_core::{
    BudgetKind, ConvergenceDiagnostics, ConvergenceReason, EvaluationRole, Interval, Invocation,
    RealScalar, ResidualTolerance, SciOxideError, SolveBudgets, SolveContext, SolveIdentity,
    SolveState,
};
use scioxide_random::InversionOptions;
use scioxide_rootfind::{Bisection, RootResult, ScalarFn, ScalarRootSolver};
mod models;
pub use models::{
    IntegerParameterInversionRequest, IntegerProbabilityParameter, ParameterInversionRequest,
    ProbabilityParameter, invert_integer_parameters_into, invert_parameters_into,
};
mod limits;
pub use limits::{
    ParameterConfidenceLimits, binomial_confidence_limits, binomial_confidence_limits_scoped,
    binomial_confidence_limits_with_options, invert_confidence_limits,
    invert_confidence_limits_scoped, invert_confidence_limits_with_options,
    noncentrality_confidence_limits, noncentrality_confidence_limits_scoped,
    noncentrality_confidence_limits_with_options, normal_variance_confidence_limits,
    normal_variance_confidence_limits_scoped, normal_variance_confidence_limits_with_options,
    poisson_rate_confidence_limits, poisson_rate_confidence_limits_scoped,
    poisson_rate_confidence_limits_with_options,
};

pub(super) fn exhausted() -> SciOxideError {
    SciOxideError::convergence_failure(
        ConvergenceReason::EvaluationLimit,
        ConvergenceDiagnostics::new(),
        "parameter tail-evaluation allowance exhausted",
    )
}
pub(super) fn identity() -> SolveIdentity {
    SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("stats", "parameter-inversion", 1),
        1,
        0,
    )
}
/// Monotonicity of the log tail probability as a function of the parameter on the whole
/// bracket or domain.
///
/// The solvers trust this declaration and check it only at the endpoints. Establish it
/// analytically, not from sampled values. [`invert_parameter_scoped`] needs strict monotonicity for a
/// unique solution. [`invert_integer_parameter_scoped`] also accepts weak monotonicity with flat
/// stretches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterMonotonicity {
    /// Increasing, up to rounding error.
    Increasing,
    /// Decreasing, up to rounding error.
    Decreasing,
    /// Constant on the whole bracket.
    Constant,
    /// Not known. The solvers return an inconclusive result without evaluating the tail.
    Unknown,
}
impl ParameterMonotonicity {
    pub(super) fn reflected(self, upper: bool) -> Self {
        if upper {
            match self {
                Self::Increasing => Self::Decreasing,
                Self::Decreasing => Self::Increasing,
                x => x,
            }
        } else {
            self
        }
    }
}
/// Target tail probability of a parameter equation, stored as its natural logarithm.
///
/// Storing `ln p` keeps tail probabilities far below the smallest positive `f64`. The side
/// selects the lower tail `P(X ≤ x)` or the upper tail `P(X > x)`.
#[derive(Debug, Clone, Copy)]
pub struct ParameterProbability<S: RealScalar> {
    log: S,
    upper: bool,
}
impl<S: RealScalar> ParameterProbability<S> {
    /// Creates a target from a probability `p` in `(0, 1)`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `probability` is NaN, infinite, or outside `(0, 1)`.
    pub fn new(probability: S, side: crate::TailSide) -> Result<Self, SciOxideError> {
        if !probability.is_finite() || probability <= S::ZERO || probability >= S::ONE {
            return Err(invalid(
                scioxide_core::argument!("probability"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "parameter probability must be inside (0,1)",
            ));
        }
        Self::from_log(probability.ln(), side)
    }
    /// Creates a target from its natural logarithm `ln p < 0`.
    ///
    /// The targets `p = 0` and `p = 1` are not accepted.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `log` is NaN, infinite, or not negative.
    pub fn from_log(log: S, side: crate::TailSide) -> Result<Self, SciOxideError> {
        if !log.is_finite() || log >= S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("log"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "log probability must be finite and negative",
            ));
        }
        Ok(Self {
            log,
            upper: side == crate::TailSide::Upper,
        })
    }
    /// Returns the natural logarithm of the target probability.
    pub const fn log_probability(self) -> S {
        self.log
    }
    /// Returns the tail of the target.
    pub const fn side(self) -> crate::TailSide {
        if self.upper {
            crate::TailSide::Upper
        } else {
            crate::TailSide::Lower
        }
    }
    /// Returns `true` for an upper-tail target `P(X > x)`.
    pub const fn is_upper(self) -> bool {
        self.upper
    }
}
/// Solution set of a continuous parameter equation on the bracket, from [`invert_parameter_scoped`].
#[derive(Debug, Clone, Copy)]
pub enum ParameterSolution<S: RealScalar> {
    /// A unique solution, or the best approximation if bisection stopped early.
    ///
    /// Check [`ParameterInversionReport::root`] for the termination status before using it.
    Unique {
        /// Approximate solution.
        value: S,
        /// Final bisection bracket. Rounding error in the tail can move the true solution
        /// outside it.
        bracket: Interval<S>,
        /// Log tail at `value` minus the target. `None` if bisection stopped early.
        log_residual: Option<S>,
        /// Difference quotient of the log tail over `bracket`, an estimate of its derivative at
        /// `value`. With the default step tolerance the bracket is a few rounding units wide, so
        /// cancellation in the log tail can make the quotient inaccurate. `None` if bisection
        /// stopped early or the quotient is zero or not finite.
        log_tail_slope: Option<S>,
    },
    /// The tail is constant and equal to the target, so every parameter in the interval solves
    /// the equation.
    Multiple(Interval<S>),
    /// The target lies outside the range of the tail on the bracket.
    NoSolution,
    /// The monotonicity is unknown, or the tail has the same value at both ends of the bracket.
    Inconclusive,
}
/// Result of [`invert_parameter_scoped`]: the solution set, the bisection result and the number of
/// tail evaluations.
///
/// The termination status of [`Self::root`] tells whether a refined solution converged.
#[derive(Debug, Clone, Copy)]
pub struct ParameterInversionReport<S: RealScalar> {
    solution: ParameterSolution<S>,
    root: Option<RootResult<S>>,
    evaluations: usize,
}
impl<S: RealScalar> ParameterInversionReport<S> {
    /// Returns the solution set on the bracket.
    pub const fn solution(&self) -> ParameterSolution<S> {
        self.solution
    }
    /// Returns the bisection result, or `None` if the solution was found without bisection.
    pub const fn root(&self) -> Option<&RootResult<S>> {
        self.root.as_ref()
    }
    /// Returns the number of calls to the log tail, including the endpoint evaluations and the
    /// final residual and slope evaluations.
    pub const fn evaluations(&self) -> usize {
        self.evaluations
    }
}
fn checked_log<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    if value.is_nan() || value > S::ZERO {
        Err(invalid(
            scioxide_core::argument!("log_tail"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "provider returned an invalid log tail",
        ))
    } else {
        Ok(value)
    }
}
pub(super) fn query<S: RealScalar>(
    context: &mut SolveContext<'_, '_>,
    f: impl FnOnce() -> Result<S, SciOxideError>,
) -> Result<S, SciOxideError> {
    context.execution().check()?;
    match context.invoke_provider(EvaluationRole::Function, || checked_log(f()?))? {
        Invocation::Evaluated(v) => Ok(v),
        Invocation::ExecutionStopped(reason) => Err(SciOxideError::execution_stopped(reason)),
        Invocation::BudgetExhausted(_) => Err(exhausted()),
    }
}

/// Executes [`invert_parameter_scoped`] with sequential native capabilities.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors.
pub fn invert_parameter_with_options<S: RealScalar>(
    log_tail: impl Fn(S) -> Result<S, SciOxideError>,
    bracket: Interval<S>,
    target: ParameterProbability<S>,
    monotonicity: ParameterMonotonicity,
    options: &InversionOptions<S>,
) -> Result<ParameterInversionReport<S>, SciOxideError> {
    invert_parameter_scoped(
        log_tail,
        bracket,
        target,
        monotonicity,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Solves `log_tail(θ) = ln p` for a distribution parameter `θ` on a finite bracket.
///
/// `log_tail` returns the natural logarithm of a tail probability of the distribution with
/// parameter `θ`. It may return `−∞` for a zero probability. With
/// [`ParameterMonotonicity::Unknown`] the result is inconclusive and `log_tail` is not called.
/// With [`ParameterMonotonicity::Constant`] the result is either no solution or the whole
/// bracket. Otherwise the solver checks the endpoints and runs bisection on the sign of
/// `log_tail(θ) − ln p`, compared exactly. After convergence it evaluates the residual and the
/// slope. The work budget of `options` limits all calls. `scope` is an execution context or a
/// solve scope that shares its budgets and cancellation with an enclosing solve.
///
/// # Errors
///
/// - `InvalidInput` if an endpoint of `bracket` is infinite.
/// - `InvalidInput` if `log_tail` returns NaN or a positive value, or if the endpoint values
///   contradict `monotonicity`.
/// - `ConvergenceFailure` with `ConvergenceReason::EvaluationLimit` if the work budget runs
///   out during the endpoint or final evaluations.
/// - `ExecutionStopped` on cancellation or deadline before bisection starts.
/// - Errors returned by `log_tail`.
///
/// If the budget runs out or the computation is cancelled during bisection, the result is
/// [`ParameterSolution::Unique`] with the last bracket and an unsuccessful termination in
/// [`ParameterInversionReport::root`].
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, Interval};
/// use scioxide_random::InversionOptions;
/// use scioxide_stats::{
///     ParameterMonotonicity, ParameterProbability, ParameterSolution, TailSide, invert_parameter_scoped,
/// };
/// // Exponential rate λ with P(X ≤ 1) = 1 − e^(−λ) = 0.5, so λ = ln 2.
/// let target = ParameterProbability::new(0.5_f64, TailSide::Lower)?;
/// let report = invert_parameter_scoped(|rate: f64| Ok((-(-rate).exp()).ln_1p()), Interval::new(0.1, 2.0)?, target, ParameterMonotonicity::Increasing, &InversionOptions::default(), ExecutionContext::sequential_native())?;
/// let ParameterSolution::Unique { value, .. } = report.solution() else { unreachable!() };
/// assert!((value - core::f64::consts::LN_2).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn invert_parameter_scoped<'scope, 'capability, S: RealScalar>(
    log_tail: impl Fn(S) -> Result<S, SciOxideError>,
    bracket: Interval<S>,
    target: ParameterProbability<S>,
    monotonicity: ParameterMonotonicity,
    options: &InversionOptions<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<ParameterInversionReport<S>, SciOxideError> {
    let mut state = SolveState::new();
    scope.into().run(
        &mut state,
        identity(),
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, options.work_budget()),
        |context| solve(bracket, target, monotonicity, options, &log_tail, context),
    )
}
fn solve<S: RealScalar>(
    bracket: Interval<S>,
    target: ParameterProbability<S>,
    monotonicity: ParameterMonotonicity,
    options: &InversionOptions<S>,
    log_tail: &impl Fn(S) -> Result<S, SciOxideError>,
    context: &mut SolveContext<'_, '_>,
) -> Result<ParameterInversionReport<S>, SciOxideError> {
    if !bracket.lower().is_finite() || !bracket.upper().is_finite() {
        return Err(invalid(
            scioxide_core::argument!("bracket"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "finite parameter bracket required",
        ));
    }
    context.execution().check()?;
    let report = |solution, root, evaluations| ParameterInversionReport {
        solution,
        root,
        evaluations,
    };
    if monotonicity == ParameterMonotonicity::Unknown {
        return Ok(report(ParameterSolution::Inconclusive, None, 0));
    }
    let a = query(context, || log_tail(bracket.lower()))?;
    let b = query(context, || log_tail(bracket.upper()))?;
    if monotonicity == ParameterMonotonicity::Constant || bracket.lower() == bracket.upper() {
        if a != b {
            return Err(invalid(
                scioxide_core::argument!("log_tail"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "constant-tail contract violated",
            ));
        }
        let solution = if a != target.log {
            ParameterSolution::NoSolution
        } else if bracket.lower() == bracket.upper() {
            ParameterSolution::Unique {
                value: bracket.lower(),
                bracket,
                log_residual: Some(S::ZERO),
                log_tail_slope: None,
            }
        } else {
            ParameterSolution::Multiple(bracket)
        };
        return Ok(report(solution, None, 2));
    }
    if a == b {
        return Ok(report(ParameterSolution::Inconclusive, None, 2));
    }
    let increasing = monotonicity == ParameterMonotonicity::Increasing;
    if (a < b) != increasing {
        return Err(invalid(
            scioxide_core::argument!("log_tail"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "monotonic-tail contract violated",
        ));
    }
    if target.log < a.min(b) || target.log > a.max(b) {
        return Ok(report(ParameterSolution::NoSolution, None, 2));
    }
    // Sign comparisons remain finite even for exact zero tails (log=-infinity).
    let function = ScalarFn::new(|x| {
        let v = checked_log(log_tail(x)?)?;
        Ok(if v == target.log {
            S::ZERO
        } else if (v < target.log) == increasing {
            -S::ONE
        } else {
            S::ONE
        })
    });
    let root_options = options
        .root_options()
        .clone()
        .with_residual_tolerance(ResidualTolerance::new(S::ZERO, S::ZERO)?);
    let root = Bisection.solve_scoped(&function, bracket, &root_options, &mut *context)?;
    let bounds = root
        .bracket()
        .map_or(Interval::new(root.point(), root.point()), |b| {
            Interval::new(b.lower(), b.upper())
        })?;
    let (residual, slope) = if root.is_success() {
        let value = query(context, || log_tail(root.point()))?;
        let low = query(context, || log_tail(bounds.lower()))?;
        let high = query(context, || log_tail(bounds.upper()))?;
        let slope = (high - low) / (bounds.upper() - bounds.lower());
        (
            Some(value - target.log),
            if slope.is_finite() && slope != S::ZERO {
                Some(slope)
            } else {
                None
            },
        )
    } else {
        (None, None)
    };
    Ok(report(
        ParameterSolution::Unique {
            value: root.point(),
            bracket: bounds,
            log_residual: residual,
            log_tail_slope: slope,
        },
        Some(root),
        context.counts().total()?.total_providers()?,
    ))
}

/// Solution of an integer parameter inequality, from [`invert_integer_parameter_scoped`].
///
/// The inequality is `log_tail(n) ≥ ln p` for an increasing tail and `log_tail(n) ≤ ln p` for
/// a decreasing one. Equality need not hold at the threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerParameterSolution {
    /// The smallest integer in the domain that satisfies the inequality, and the integer
    /// before it.
    Threshold {
        /// Smallest integer that satisfies the inequality.
        first: usize,
        /// `first − 1`, if it lies in the domain.
        previous: Option<usize>,
    },
    /// Every integer in the domain satisfies the inequality.
    All {
        /// Lower end of the domain.
        lower: usize,
        /// Upper end of the domain.
        upper: usize,
    },
    /// No integer in the domain satisfies the inequality.
    None,
    /// The monotonicity is unknown.
    Inconclusive,
}
/// Result of [`invert_integer_parameter_scoped`]: the solution and the number of tail evaluations.
#[derive(Debug, Clone, Copy)]
pub struct IntegerParameterReport {
    /// Solution of the inequality on the domain.
    pub solution: IntegerParameterSolution,
    /// Number of calls to the log tail.
    pub evaluations: usize,
}

/// Executes [`invert_integer_parameter_scoped`] with sequential native capabilities.
///
/// # Errors
///
/// Returns the scoped form's input, numerical and allocation errors.
pub fn invert_integer_parameter_with_options<S: RealScalar>(
    log_tail: impl Fn(usize) -> Result<S, SciOxideError>,
    domain: (usize, usize),
    target: ParameterProbability<S>,
    monotonicity: ParameterMonotonicity,
    options: &InversionOptions<S>,
) -> Result<IntegerParameterReport, SciOxideError> {
    invert_integer_parameter_scoped(
        log_tail,
        domain,
        target,
        monotonicity,
        options,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}
/// Finds the smallest integer parameter `n` in `domain.0..=domain.1` whose log tail reaches
/// the target.
///
/// The inequality is `log_tail(n) ≥ ln p` for an increasing tail and `log_tail(n) ≤ ln p` for
/// a decreasing tail. The tail may be weakly monotone. The solver evaluates both ends of the
/// domain and then bisects, so it makes at most `⌈log₂(domain.1 − domain.0 + 1)⌉ + 2` calls.
/// With [`ParameterMonotonicity::Unknown`] the result is inconclusive and `log_tail` is not
/// called. The work budget of `options` limits the calls. `scope` is an execution context or
/// a solve scope that shares its budgets and cancellation with an enclosing solve.
///
/// # Errors
///
/// - `InvalidInput` if `domain.0 > domain.1`.
/// - `InvalidInput` if `log_tail` returns NaN or a positive value, or if the endpoint values
///   contradict `monotonicity`.
/// - `ConvergenceFailure` with `ConvergenceReason::EvaluationLimit` if the work budget runs
///   out.
/// - `ExecutionStopped` on cancellation or deadline.
/// - Errors returned by `log_tail`.
pub fn invert_integer_parameter_scoped<'scope, 'capability, S: RealScalar>(
    log_tail: impl Fn(usize) -> Result<S, SciOxideError>,
    domain: (usize, usize),
    target: ParameterProbability<S>,
    monotonicity: ParameterMonotonicity,
    options: &InversionOptions<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<IntegerParameterReport, SciOxideError> {
    if domain.0 > domain.1 {
        return Err(invalid(
            scioxide_core::argument!("domain"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "reversed integer parameter domain",
        ));
    }
    let mut state = SolveState::new();
    scope.into().run(
        &mut state,
        identity(),
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, options.work_budget()),
        |context| {
            context.execution().check()?;
            let solution = if monotonicity == ParameterMonotonicity::Unknown {
                IntegerParameterSolution::Inconclusive
            } else {
                let mut lo = domain.0;
                let mut hi = domain.1;
                let a = query(context, || log_tail(lo))?;
                let b = query(context, || log_tail(hi))?;
                let increasing = monotonicity == ParameterMonotonicity::Increasing;
                let satisfies = |v| {
                    if increasing {
                        v >= target.log
                    } else {
                        v <= target.log
                    }
                };
                if monotonicity == ParameterMonotonicity::Constant && a != b
                    || monotonicity != ParameterMonotonicity::Constant
                        && ((increasing && a > b) || (!increasing && a < b))
                {
                    return Err(invalid(
                        scioxide_core::argument!("log_tail"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "integer monotonic-tail contract violated",
                    ));
                }
                if satisfies(a) {
                    IntegerParameterSolution::All {
                        lower: lo,
                        upper: hi,
                    }
                } else if !satisfies(b) {
                    IntegerParameterSolution::None
                } else {
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        let value = query(context, || log_tail(mid))?;
                        if satisfies(value) {
                            hi = mid;
                        } else {
                            lo = mid + 1;
                        }
                    }
                    IntegerParameterSolution::Threshold {
                        first: lo,
                        previous: lo.checked_sub(1).filter(|&n| n >= domain.0),
                    }
                }
            };
            Ok(IntegerParameterReport {
                solution,
                evaluations: context.counts().total()?.total_providers()?,
            })
        },
    )
}

/// Inverts the parameter probability with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same bracketed algorithm and accuracy as [`invert_parameter_with_options`].
///
/// # Errors
///
/// - `NonFinite` if `lower` or `upper` is not finite, `InvalidInput` if `lower > upper`, and
///   the errors of [`invert_parameter_scoped`].
/// - `ConvergenceFailure` with `ConvergenceReason::Stagnation` if the bisection ends without
///   success, whatever its [`ParameterInversionReport::root`] termination.
pub fn invert_parameter<S: RealScalar>(
    log_tail: impl Fn(S) -> Result<S, SciOxideError>,
    lower: S,
    upper: S,
    target: ParameterProbability<S>,
    monotonicity: ParameterMonotonicity,
) -> Result<ParameterInversionReport<S>, SciOxideError> {
    let report = invert_parameter_with_options(
        log_tail,
        Interval::new(lower, upper)?,
        target,
        monotonicity,
        &InversionOptions::default(),
    )?;
    if report
        .root()
        .is_some_and(|root| !root.termination().is_success())
    {
        return Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::Stagnation,
            Default::default(),
            "parameter inversion did not converge",
        ));
    }
    Ok(report)
}

/// Inverts the parameter probability with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same bracketed algorithm and accuracy as [`invert_integer_parameter_with_options`].
///
/// # Errors
///
/// Rejects invalid data or bounds and propagates numerical and convergence errors.
pub fn invert_integer_parameter<S: RealScalar>(
    log_tail: impl Fn(usize) -> Result<S, SciOxideError>,
    domain: (usize, usize),
    target: ParameterProbability<S>,
    monotonicity: ParameterMonotonicity,
) -> Result<IntegerParameterReport, SciOxideError> {
    invert_integer_parameter_with_options(
        log_tail,
        domain,
        target,
        monotonicity,
        &InversionOptions::default(),
    )
}
