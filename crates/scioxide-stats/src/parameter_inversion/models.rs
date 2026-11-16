//! Tail-probability equations for named distribution parameters.
use super::*;
use scioxide_core::ExecutionContext;

use scioxide_random::{
    Beta, Binomial, ChiSquared, FisherF, Gamma, LawOptions, LogCdf, LogSurvival, NegativeBinomial,
    NoncentralChiSquared, NoncentralF, NoncentralStudentT, Poisson, StudentT,
};

/// Distribution, unknown real parameter and evaluation point of a tail-probability equation.
///
/// Each variant names one unknown parameter of a distribution and fixes the others. `x` or
/// `count` is the point at which the tail is evaluated. The lower tail is `P(X ≤ x)` and the
/// upper tail is `P(X > x)`. The Gamma rate is the reciprocal of the scale. Each variant states
/// how the lower tail varies with the unknown parameter. The upper tail varies the other way,
/// as returned by [`Self::monotonicity`].
#[derive(Debug, Clone, Copy)]
pub enum ProbabilityParameter<S: RealScalar> {
    /// Gamma shape with fixed scale. The lower tail decreases for `x > 0`.
    GammaShape {
        /// Evaluation point.
        x: S,
        /// Fixed positive scale.
        scale: S,
    },
    /// Gamma scale with fixed shape. The lower tail decreases for `x > 0`.
    GammaScale {
        /// Evaluation point.
        x: S,
        /// Fixed positive shape.
        shape: S,
    },
    /// Gamma rate with fixed shape. The lower tail increases for `x > 0`.
    GammaRate {
        /// Evaluation point.
        x: S,
        /// Fixed positive shape.
        shape: S,
    },
    /// First Beta shape `α` with fixed `β`. The lower tail decreases for `0 < x < 1`.
    BetaAlpha {
        /// Evaluation point.
        x: S,
        /// Fixed positive second shape `β`.
        beta: S,
    },
    /// Second Beta shape `β` with fixed `α`. The lower tail increases for `0 < x < 1`.
    BetaBeta {
        /// Evaluation point.
        x: S,
        /// Fixed positive first shape `α`.
        alpha: S,
    },
    /// Degrees of freedom of the chi-squared distribution. The lower tail decreases for
    /// `x > 0`.
    ChiSquaredDegrees {
        /// Evaluation point.
        x: S,
    },
    /// Degrees of freedom of Student's t distribution. The lower tail increases for `x > 0`,
    /// decreases for `x < 0`, and is 1/2 at `x = 0`.
    StudentTDegrees {
        /// Evaluation point.
        x: S,
    },
    /// Numerator degrees of freedom of the F distribution. The lower tail decreases for
    /// `x > 0` when the denominator degrees of freedom are 2. For other denominators the
    /// monotonicity is unknown and the solution is inconclusive.
    FNumeratorDegrees {
        /// Evaluation point.
        x: S,
        /// Fixed positive denominator degrees of freedom.
        denominator: S,
    },
    /// Denominator degrees of freedom of the F distribution. The lower tail increases for
    /// `x > 0` when the numerator degrees of freedom are 2. For other numerators the
    /// monotonicity is unknown and the solution is inconclusive.
    FDenominatorDegrees {
        /// Evaluation point.
        x: S,
        /// Fixed positive numerator degrees of freedom.
        numerator: S,
    },
    /// Noncentrality `λ ≥ 0` of the noncentral chi-squared distribution. The lower tail
    /// decreases for `x > 0`.
    ChiSquaredNoncentrality {
        /// Evaluation point.
        x: S,
        /// Fixed positive degrees of freedom.
        degrees: S,
    },
    /// Noncentrality `λ ≥ 0` of the noncentral F distribution. The lower tail decreases for
    /// `x > 0`.
    FNoncentrality {
        /// Evaluation point.
        x: S,
        /// Fixed positive numerator degrees of freedom.
        numerator: S,
        /// Fixed positive denominator degrees of freedom.
        denominator: S,
    },
    /// Noncentrality `δ` of the noncentral t distribution. The lower tail decreases.
    StudentTNoncentrality {
        /// Evaluation point.
        x: S,
        /// Fixed positive degrees of freedom.
        degrees: S,
    },
    /// Success probability `p ∈ [0, 1]` of the binomial distribution. The lower tail
    /// decreases when `count < trials`.
    BinomialProbability {
        /// Evaluation point: the lower tail is `P(X ≤ count)`.
        count: usize,
        /// Fixed number of trials.
        trials: usize,
    },
    /// Mean `μ ≥ 0` of the Poisson distribution, not a rate per unit exposure. The lower tail
    /// decreases.
    PoissonMean {
        /// Evaluation point: the lower tail is `P(X ≤ count)`.
        count: usize,
    },
}
fn tail<S: RealScalar, X: Copy, D: LogCdf<S, X> + LogSurvival<S, X>>(
    law: D,
    x: X,
    upper: bool,
) -> Result<S, SciOxideError> {
    if upper { law.log_sf(x) } else { law.log_cdf(x) }
}
impl<S: RealScalar> ProbabilityParameter<S> {
    /// Returns the natural logarithm of the lower or upper tail at the given parameter value.
    ///
    /// `resources` sets the work limits of the F and noncentral distributions. The other
    /// distributions use their default options.
    ///
    /// # Errors
    ///
    /// - Errors from constructing the distribution, for example `InvalidInput` for a parameter
    ///   outside its domain.
    /// - Errors from evaluating its tail.
    pub fn log_tail(
        self,
        parameter: S,
        side: crate::TailSide,
        resources: LawOptions<S>,
    ) -> Result<S, SciOxideError> {
        let upper = side == crate::TailSide::Upper;
        match self {
            Self::GammaShape { x, scale } => tail(Gamma::new(parameter, scale)?, x, upper),
            Self::GammaScale { x, shape } => tail(Gamma::new(shape, parameter)?, x, upper),
            Self::GammaRate { x, shape } => tail(Gamma::new(shape, S::ONE / parameter)?, x, upper),
            Self::BetaAlpha { x, beta } => tail(Beta::new(parameter, beta)?, x, upper),
            Self::BetaBeta { x, alpha } => tail(Beta::new(alpha, parameter)?, x, upper),
            Self::ChiSquaredDegrees { x } => tail(ChiSquared::new(parameter)?, x, upper),
            Self::StudentTDegrees { x } => tail(StudentT::new(parameter)?, x, upper),
            Self::FNumeratorDegrees { x, denominator } => tail(
                FisherF::with_options(parameter, denominator, resources)?,
                x,
                upper,
            ),
            Self::FDenominatorDegrees { x, numerator } => tail(
                FisherF::with_options(numerator, parameter, resources)?,
                x,
                upper,
            ),
            Self::ChiSquaredNoncentrality { x, degrees } => tail(
                NoncentralChiSquared::with_options(degrees, parameter, resources)?,
                x,
                upper,
            ),
            Self::FNoncentrality {
                x,
                numerator,
                denominator,
            } => tail(
                NoncentralF::with_options(numerator, denominator, parameter, resources)?,
                x,
                upper,
            ),
            Self::StudentTNoncentrality { x, degrees } => tail(
                NoncentralStudentT::with_options(degrees, parameter, resources)?,
                x,
                upper,
            ),
            Self::BinomialProbability { count, trials } => {
                tail(Binomial::new(trials, parameter)?, count, upper)
            }
            Self::PoissonMean { count } => tail(Poisson::new(parameter)?, count, upper),
        }
    }
    /// Returns how the selected tail varies with the unknown parameter.
    ///
    /// The tail is constant where the evaluation point lies outside the support, for example
    /// `x ≤ 0` for the Gamma distribution. F degrees of freedom return
    /// [`ParameterMonotonicity::Unknown`] unless the other degrees of freedom equal 2. To
    /// invert them elsewhere, call [`invert_parameter_scoped`](crate::invert_parameter_scoped) with a
    /// monotonicity you have established for your bracket.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `x` is not finite, or a fixed shape, scale or degrees of freedom is
    ///   not finite and positive.
    pub fn monotonicity(
        self,
        side: crate::TailSide,
    ) -> Result<ParameterMonotonicity, SciOxideError> {
        let upper = side == crate::TailSide::Upper;
        use ParameterMonotonicity::{Constant, Decreasing, Increasing, Unknown};
        let positive = |p: S| {
            if p.is_finite() && p > S::ZERO {
                Ok(())
            } else {
                Err(invalid(
                    scioxide_core::argument!("self"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "fixed law parameter must be finite and positive",
                ))
            }
        };
        let coordinate = |x: S| {
            if x.is_finite() {
                Ok(())
            } else {
                Err(invalid(
                    scioxide_core::argument!("self"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "finite observation required",
                ))
            }
        };
        let positive_support = |x: S| -> Result<_, SciOxideError> {
            coordinate(x)?;
            Ok(if x <= S::ZERO { Constant } else { Decreasing })
        };
        let direction = match self {
            Self::GammaShape { x, scale } => {
                positive(scale)?;
                positive_support(x)?
            }
            Self::GammaScale { x, shape } => {
                positive(shape)?;
                positive_support(x)?
            }
            Self::GammaRate { x, shape } => {
                positive(shape)?;
                positive_support(x)?.reflected(true)
            }
            Self::BetaAlpha { x, beta } => {
                positive(beta)?;
                coordinate(x)?;
                if x <= S::ZERO || x >= S::ONE {
                    Constant
                } else {
                    Decreasing
                }
            }
            Self::BetaBeta { x, alpha } => {
                positive(alpha)?;
                coordinate(x)?;
                if x <= S::ZERO || x >= S::ONE {
                    Constant
                } else {
                    Increasing
                }
            }
            Self::ChiSquaredDegrees { x } => positive_support(x)?,
            Self::StudentTDegrees { x } => {
                coordinate(x)?;
                if x == S::ZERO {
                    Constant
                } else if x > S::ZERO {
                    Increasing
                } else {
                    Decreasing
                }
            }
            Self::FNumeratorDegrees { x, denominator } => {
                positive(denominator)?;
                coordinate(x)?;
                if x <= S::ZERO {
                    Constant
                } else if denominator == S::TWO {
                    Decreasing
                } else {
                    Unknown
                }
            }
            Self::FDenominatorDegrees { x, numerator } => {
                positive(numerator)?;
                coordinate(x)?;
                if x <= S::ZERO {
                    Constant
                } else if numerator == S::TWO {
                    Increasing
                } else {
                    Unknown
                }
            }
            Self::ChiSquaredNoncentrality { x, degrees } => {
                positive(degrees)?;
                positive_support(x)?
            }
            Self::FNoncentrality {
                x,
                numerator,
                denominator,
            } => {
                positive(numerator)?;
                positive(denominator)?;
                positive_support(x)?
            }
            Self::StudentTNoncentrality { x, degrees } => {
                positive(degrees)?;
                coordinate(x)?;
                Decreasing
            }
            Self::BinomialProbability { count, trials } => {
                if count >= trials {
                    Constant
                } else {
                    Decreasing
                }
            }
            Self::PoissonMean { .. } => Decreasing,
        };
        Ok(direction.reflected(upper))
    }
    pub(super) fn validate_bracket(self, bracket: Interval<S>) -> Result<(), SciOxideError> {
        let nonnegative = matches!(
            self,
            Self::ChiSquaredNoncentrality { .. }
                | Self::FNoncentrality { .. }
                | Self::PoissonMean { .. }
                | Self::BinomialProbability { .. }
        );
        let real = matches!(self, Self::StudentTNoncentrality { .. });
        if !bracket.lower().is_finite()
            || !bracket.upper().is_finite()
            || (!real
                && (bracket.lower() < S::ZERO || (!nonnegative && bracket.lower() == S::ZERO)))
            || matches!(self, Self::BinomialProbability { .. }) && bracket.upper() > S::ONE
        {
            return Err(invalid(
                scioxide_core::argument!("bracket"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "parameter bracket outside law domain",
            ));
        }
        Ok(())
    }
}
/// Equation `tail(θ) = p` for a named distribution parameter `θ` on a bracket.
///
/// The solver searches only inside `bracket`.
#[derive(Debug, Clone)]
pub struct ParameterInversionRequest<S: RealScalar> {
    /// Distribution, unknown parameter and evaluation point.
    pub model: ProbabilityParameter<S>,
    /// Finite interval of parameter values to search.
    pub bracket: Interval<S>,
    /// Target probability and tail.
    pub target: ParameterProbability<S>,
    /// Bisection options and the limit on the total number of tail evaluations.
    pub options: InversionOptions<S>,
    /// Work limits for each evaluation of an F or noncentral distribution.
    pub tail_options: LawOptions<S>,
}
impl<S: RealScalar> ParameterInversionRequest<S> {
    /// Solves the equation with [`invert_parameter_scoped`](crate::invert_parameter_scoped), using the
    /// monotonicity from [`ProbabilityParameter::monotonicity`].
    ///
    /// `scope` is an execution context or a solve scope that shares its budgets and
    /// cancellation with an enclosing solve.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `bracket` is not finite or leaves the parameter domain: negative
    ///   values for any parameter but a t noncentrality, zero for a shape, scale, rate or
    ///   degrees of freedom, or values above 1 for a binomial probability.
    /// - The errors of [`ProbabilityParameter::monotonicity`] and
    ///   [`invert_parameter_scoped`](crate::invert_parameter_scoped).
    pub fn solve_scoped<'scope, 'capability>(
        &self,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<ParameterInversionReport<S>, SciOxideError> {
        self.model.validate_bracket(self.bracket)?;
        invert_parameter_scoped(
            |p| {
                self.model
                    .log_tail(p, self.target.side(), self.tail_options)
            },
            self.bracket,
            self.target,
            self.model.monotonicity(self.target.side())?,
            &self.options,
            scope,
        )
    }
    /// Solves the request with sequential native capabilities.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::solve_scoped`].
    pub fn solve(&self) -> Result<ParameterInversionReport<S>, SciOxideError> {
        self.solve_scoped(scioxide_core::ExecutionContext::sequential_native())
    }
}
/// Solves independent parameter equations and writes report `i` to `output[i]`.
///
/// Each request runs [`ParameterInversionRequest::solve`] with its own evaluation limit.
/// Requests run in parallel under `execution`, on at most 64 workers. `staging` is scratch
/// storage of at least `requests.len()` entries. On success every entry of `output` is
/// `Some`. A request with no solution, a solution interval or an inconclusive result is
/// a valid report, not an error.
///
/// # Errors
///
/// - `InvalidInput` if `output.len() ≠ requests.len()` or `staging.len() < requests.len()`.
/// - `ConvergenceFailure` with `ConvergenceReason::EvaluationLimit` if the bisection of a
///   request does not converge.
/// - `ExecutionStopped` on cancellation.
/// - The first error of [`ParameterInversionRequest::solve`].
pub fn invert_parameters_into<S: RealScalar>(
    requests: &[ParameterInversionRequest<S>],
    execution: scioxide_core::ExecutionPolicy,
    output: &mut [Option<ParameterInversionReport<S>>],
    staging: &mut [Option<ParameterInversionReport<S>>],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if output.len() != requests.len() || staging.len() < requests.len() {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "parameter batch output/staging lengths",
        ));
    }
    let options = crate::StatisticsOptions::new().with_execution_policy(execution);
    let mut workers = [(); 64];
    crate::estimation::execute(
        options,
        context,
        requests.len(),
        1,
        &mut workers,
        staging,
        |index, _, slot| {
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut solve = state.begin_solve(context, identity(), &mut budgets)?;
            let report = requests[index].solve_scoped(&mut solve)?;
            if report.root.is_some_and(|root| !root.is_success()) {
                return Err(exhausted());
            }
            slot[0] = Some(report);
            Ok(())
        },
    )?;
    output.copy_from_slice(&staging[..requests.len()]);
    Ok(())
}
/// Discrete distribution with an unknown integer parameter, for
/// [`IntegerProbabilityParameter::solve`].
///
/// In both variants the lower tail `P(X ≤ count)` decreases as the integer parameter grows.
#[derive(Debug, Clone, Copy)]
pub enum IntegerProbabilityParameter<S: RealScalar> {
    /// Number of trials of the binomial distribution.
    BinomialTrials {
        /// Evaluation point: the lower tail is `P(X ≤ count)`.
        count: usize,
        /// Fixed success probability in `[0, 1]`.
        probability: S,
    },
    /// Number of successes `r ≥ 1` of the negative binomial distribution of the number of
    /// failures before the `r`-th success.
    NegativeBinomialShape {
        /// Evaluation point: the lower tail is `P(X ≤ count)` failures.
        count: usize,
        /// Fixed success probability in `(0, 1]`.
        probability: S,
    },
}
impl<S: RealScalar> IntegerProbabilityParameter<S> {
    /// Finds the smallest integer parameter `n` in `domain.0..=domain.1` that satisfies the
    /// tail inequality, with [`invert_integer_parameter_scoped`](crate::invert_integer_parameter_scoped).
    ///
    /// For a lower-tail target the inequality is `P(X ≤ count) ≤ p`. For an upper-tail target
    /// it is `P(X > count) ≥ p`. `scope` is an execution context or a solve scope that shares
    /// its budgets and cancellation with an enclosing solve.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `probability` is not finite or outside `[0, 1]`.
    /// - `InvalidInput` for [`Self::NegativeBinomialShape`] if `probability` is zero,
    ///   `domain.0` is zero, or `domain.1` exceeds `2/ε` (2⁵³ for `f64`), above which `S` does
    ///   not represent every integer.
    /// - The errors of [`invert_integer_parameter_scoped`](crate::invert_integer_parameter_scoped).
    pub fn solve_scoped<'scope, 'capability>(
        self,
        domain: (usize, usize),
        target: ParameterProbability<S>,
        options: &InversionOptions<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<IntegerParameterReport, SciOxideError> {
        let (p, negative) = match self {
            Self::BinomialTrials { probability, .. } => (probability, false),
            Self::NegativeBinomialShape { probability, .. } => (probability, true),
        };
        if !p.is_finite()
            || p < S::ZERO
            || p > S::ONE
            || negative && (p == S::ZERO || domain.0 == 0)
        {
            return Err(invalid(
                scioxide_core::argument!("domain"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid integer law parameter domain",
            ));
        }
        // Compare as integers: converting 2^53+1 to f64 would round down and
        // accidentally admit an unrepresentable shape. The cast saturates on
        // narrower usize platforms, where every supported integer fits.
        let exact_limit = (2.0 / S::EPSILON.into_f64()) as usize;
        if negative && domain.1 > exact_limit {
            return Err(invalid(
                scioxide_core::argument!("domain"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "integer shape exceeds exact scalar range",
            ));
        }
        // Integer threshold contracts are weakly monotone. Keep their orientation
        // even for a degenerate law: reflecting a constant CDF must still change
        // the inequality from <= to >= for its survival tail.
        let direction = ParameterMonotonicity::Decreasing.reflected(target.upper);
        invert_integer_parameter_scoped(
            |n| match self {
                Self::BinomialTrials { count, probability } => {
                    tail(Binomial::new(n, probability)?, count, target.upper)
                }
                Self::NegativeBinomialShape { count, probability } => tail(
                    NegativeBinomial::new(S::try_from_usize_exact(n)?, probability)?,
                    count,
                    target.upper,
                ),
            },
            domain,
            target,
            direction,
            options,
            scope,
        )
    }
    /// Solves an integer probability threshold with default inversion options.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::solve_with_options`].
    pub fn solve(
        self,
        domain: (usize, usize),
        target: ParameterProbability<S>,
    ) -> Result<IntegerParameterReport, SciOxideError> {
        self.solve_with_options(domain, target, &InversionOptions::default())
    }
    /// Solves the request with sequential native capabilities.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::solve_scoped`].
    pub fn solve_with_options(
        self,
        domain: (usize, usize),
        target: ParameterProbability<S>,
        options: &InversionOptions<S>,
    ) -> Result<IntegerParameterReport, SciOxideError> {
        self.solve_scoped(
            domain,
            target,
            options,
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }
}

/// Integer parameter inequality for [`invert_integer_parameters_into`].
#[derive(Debug, Clone)]
pub struct IntegerParameterInversionRequest<S: RealScalar> {
    /// Distribution and evaluation point.
    pub model: IntegerProbabilityParameter<S>,
    /// Inclusive range of integers to search.
    pub domain: (usize, usize),
    /// Target probability and tail.
    pub target: ParameterProbability<S>,
    /// Limit on the number of tail evaluations. The bisection options are not used.
    pub options: InversionOptions<S>,
}
/// Solves independent integer parameter inequalities and writes report `i` to `output[i]`.
///
/// Each request runs [`IntegerProbabilityParameter::solve`] with its own evaluation limit.
/// Requests run in parallel under `execution`, on at most 64 workers. `staging` is scratch
/// storage of at least `requests.len()` entries. On success every entry of `output` is
/// `Some`.
///
/// # Errors
///
/// - `InvalidInput` if `output.len() ≠ requests.len()` or `staging.len() < requests.len()`.
/// - `ExecutionStopped` on cancellation.
/// - The first error of [`IntegerProbabilityParameter::solve`].
pub fn invert_integer_parameters_into<S: RealScalar>(
    requests: &[IntegerParameterInversionRequest<S>],
    execution: scioxide_core::ExecutionPolicy,
    output: &mut [Option<IntegerParameterReport>],
    staging: &mut [Option<IntegerParameterReport>],
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if output.len() != requests.len() || staging.len() < requests.len() {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "integer parameter batch output/staging lengths",
        ));
    }
    let options = crate::StatisticsOptions::new().with_execution_policy(execution);
    let mut workers = [(); 64];
    crate::estimation::execute(
        options,
        context,
        requests.len(),
        1,
        &mut workers,
        staging,
        |index, _, slot| {
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded();
            let mut solve = state.begin_solve(context, identity(), &mut budgets)?;
            let request = &requests[index];
            slot[0] = Some(request.model.solve_scoped(
                request.domain,
                request.target,
                &request.options,
                &mut solve,
            )?);
            Ok(())
        },
    )?;
    output.copy_from_slice(&staging[..requests.len()]);
    Ok(())
}
