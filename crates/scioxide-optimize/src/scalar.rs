//! Minimization of a real function of one variable.

use crate::OptimizeTermination;
use crate::context::SolveSession;
use scioxide_core::{
    AlgorithmId, Budget, BudgetKind, EvaluationRole, ExecutionContext, Interval, NonFiniteOrigin,
    PartialResultTrust, RealScalar, ResidualTolerance, SciOxideError, SolveBudgets, SolveContext,
    SolveIdentity, SolveScope, SolveState, StepTolerance,
};

/// Evaluates a real objective of one real variable.
///
/// Closures `Fn(S) -> Result<S, SciOxideError>` implement this trait. Every call
/// counts toward the evaluation limit.
pub trait ScalarObjective<S: RealScalar> {
    /// Evaluates the objective at a finite point `x`.
    ///
    /// A non-finite value rejects the trial: the search ranks it above every
    /// finite value and never returns it as a result. The initial samples must
    /// include a finite value.
    ///
    /// # Errors
    ///
    /// Errors returned here stop the search and are returned unchanged.
    fn eval(&self, x: S) -> Result<S, SciOxideError>;
}
impl<S: RealScalar, F: Fn(S) -> Result<S, SciOxideError>> ScalarObjective<S> for F {
    #[inline]
    fn eval(&self, x: S) -> Result<S, SciOxideError> {
        self(x)
    }
}

/// Stores a point and the finite objective value there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimumSample<S: RealScalar> {
    x: S,
    value: S,
}
impl<S: RealScalar> MinimumSample<S> {
    /// Returns the point.
    pub const fn x(self) -> S {
        self.x
    }
    /// Returns the objective value at the point.
    pub const fn value(self) -> S {
        self.value
    }
}

/// Stores a bracket of a local minimum: three points `a < b < c` with `f(b) ≤ f(a)`,
/// `f(b) ≤ f(c)` and `f(b)` strictly smaller than at least one of them.
///
/// A continuous function has a local minimum in `(a, c)` when such a bracket
/// exists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimumBracket<S: RealScalar> {
    samples: [MinimumSample<S>; 3],
}
impl<S: RealScalar> MinimumBracket<S> {
    /// Creates a bracket from three points and their objective values.
    ///
    /// A solve started from a bracket evaluates the objective at the three
    /// points again, so the values here only serve to check the bracket.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a point or value is NaN or infinite.
    /// - `InvalidInput` if the points are not strictly increasing or the values
    ///   do not form a bracket.
    pub fn new(points: [S; 3], values: [S; 3]) -> Result<Self, SciOxideError> {
        for value in points.into_iter().chain(values) {
            finite(value, NonFiniteOrigin::Input)?;
        }
        let samples = core::array::from_fn(|i| MinimumSample {
            x: points[i],
            value: values[i],
        });
        if !valid_bracket(samples) {
            return Err(invalid(
                scioxide_core::argument!("values"),
                "minimum bracket invariant failed",
            ));
        }
        Ok(Self { samples })
    }
    /// Returns the lower, middle and upper samples, in increasing order of the point.
    pub const fn samples(self) -> [MinimumSample<S>; 3] {
        self.samples
    }
}

/// Selects how a scalar minimum search starts.
#[derive(Debug, Clone, Copy)]
pub enum ScalarMinimumStart<S: RealScalar> {
    /// Searches the closed interval `[a, b]`. A single-point interval is evaluated once.
    Bounded(Interval<S>),
    /// Starts from a bracket; its three points are evaluated again.
    Bracket(MinimumBracket<S>),
    /// Finds a bracket by searching downhill from two distinct finite points.
    Seeds(S, S),
}

/// Selects the derivative-free method that shrinks the bracket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarMinimumMethod {
    /// Golden-section search: shrinks the bracket by the factor `(√5 − 1)/2` per iteration.
    GoldenSection,
    /// Brent's method: parabolic interpolation through three points, with
    /// golden-section steps when the parabolic step is unsafe.
    Brent,
}

/// Sets the stopping tolerances and limits of [`ScalarMinimizer`].
///
/// A search with bracket `a < x < b` converges when
///
/// ```text
/// |b − a| ≤ step.threshold(max(|a|, |b|))
/// |f(a) − f(x)|, |f(b) − f(x)| ≤ function.threshold(max(|f(a)|, |f(x)|, |f(b)|))
/// ```
///
/// where `threshold(s) = atol + rtol·s`. The defaults are the default
/// [`StepTolerance`] and [`ResidualTolerance`], 1000 iterations and 2000
/// objective evaluations. See also [Termination](crate#termination).
#[derive(Debug, Clone, Copy)]
pub struct ScalarMinimumOptions<S: RealScalar> {
    step: StepTolerance<S>,
    function: ResidualTolerance<S>,
    iterations: Budget,
    evaluations: Budget,
}
impl<S: RealScalar> Default for ScalarMinimumOptions<S> {
    fn default() -> Self {
        Self {
            step: StepTolerance::default(),
            function: ResidualTolerance::default(),
            iterations: Budget::ONE.saturating_add(999),
            evaluations: Budget::ONE.saturating_add(1999),
        }
    }
}
impl<S: RealScalar> ScalarMinimumOptions<S> {
    /// Sets the tolerance on the bracket width.
    pub const fn with_step_tolerance(mut self, value: StepTolerance<S>) -> Self {
        self.step = value;
        self
    }
    /// Sets the tolerance on the spread of the objective values in the bracket.
    pub const fn with_function_tolerance(mut self, value: ResidualTolerance<S>) -> Self {
        self.function = value;
        self
    }
    /// Sets the iteration limit, which includes bracketing.
    pub const fn with_max_iterations(mut self, value: Budget) -> Self {
        self.iterations = value;
        self
    }
    /// Sets the objective evaluation limit, which includes the initial samples.
    pub const fn with_max_evaluations(mut self, value: Budget) -> Self {
        self.evaluations = value;
        self
    }
    /// Returns the tolerance on the bracket width.
    pub const fn step_tolerance(self) -> StepTolerance<S> {
        self.step
    }
    /// Returns the tolerance on the spread of the objective values.
    pub const fn function_tolerance(self) -> ResidualTolerance<S> {
        self.function
    }
    /// Returns the iteration limit.
    pub const fn max_iterations(self) -> Budget {
        self.iterations
    }
    /// Returns the objective evaluation limit.
    pub const fn max_evaluations(self) -> Budget {
        self.evaluations
    }
}

/// Reports the result of a scalar minimum search.
///
/// An unsuccessful result still holds the best finite sample. The bracket is
/// absent when initialization or bracketing did not finish, or when an
/// endpoint had a non-finite value.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct ScalarMinimumResult<S: RealScalar> {
    best: MinimumSample<S>,
    samples: Option<[MinimumSample<S>; 3]>,
    bracket: Option<MinimumBracket<S>>,
    termination: OptimizeTermination,
    counts: scioxide_core::SolveCounts,
    algorithm: AlgorithmId,
}
impl<S: RealScalar> ScalarMinimumResult<S> {
    /// Returns the point with the smallest objective value found.
    ///
    /// Among equal values the earlier sample wins: the middle point of a
    /// supplied bracket, or the first minimum of a bounded start.
    pub const fn x(self) -> S {
        self.best.x
    }
    /// Returns the objective value at [`Self::x`].
    pub const fn fun(self) -> S {
        self.best.value
    }
    /// Returns the last lower, best and upper samples.
    pub const fn samples(self) -> Option<[MinimumSample<S>; 3]> {
        self.samples
    }
    /// Returns the final bracket, or `None` for a minimum at an interval end or
    /// when bracketing did not finish.
    pub const fn bracket(self) -> Option<MinimumBracket<S>> {
        self.bracket
    }
    /// Returns the termination status.
    pub const fn termination(self) -> OptimizeTermination {
        self.termination
    }
    /// Returns `true` if both tolerances were met, or, for
    /// [`ScalarMinimizer::bracket_minimum`], if a bracket was found.
    pub const fn is_success(self) -> bool {
        self.termination.is_success()
    }

    /// Requires the method's stopping test to have passed.
    ///
    /// # Errors
    ///
    /// Returns `ExecutionStopped` for cancellation or a deadline, `DomainError`
    /// for verified infeasibility or unboundedness, `InvalidInput` before a solve,
    /// and `ConvergenceFailure` with counts for every other unsuccessful termination.
    pub fn require_success(self) -> Result<(), SciOxideError> {
        crate::result::termination_error(
            scioxide_core::argument!("self"),
            self.termination(),
            self.counts(),
            Some(self.algorithm()),
        )
        .map_or(Ok(()), Err)
    }

    /// Returns the iteration and evaluation counts, including initialization and bracketing.
    pub const fn counts(self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the number of iterations.
    pub fn iterations(self) -> usize {
        self.counts.own().iterations()
    }
    /// Returns the number of objective evaluations.
    pub fn fun_evals(self) -> usize {
        self.counts.own().provider(EvaluationRole::Objective)
    }
    /// Returns the algorithm identifier.
    pub const fn algorithm(self) -> AlgorithmId {
        self.algorithm
    }
    /// Returns [`PartialResultTrust::Admissible`]: the result is a finite
    /// sample, not a proven minimum.
    pub const fn trust(self) -> PartialResultTrust {
        PartialResultTrust::Admissible
    }
    /// Returns the interval spanned by the last lower and upper samples.
    ///
    /// Its width may exceed the largest finite value of `S`.
    ///
    /// # Errors
    ///
    /// Returns an error if the samples do not form a valid [`Interval`].
    pub fn interval(self) -> Result<Option<Interval<S>>, SciOxideError> {
        self.samples
            .map(|s| Interval::new(s[0].x, s[2].x))
            .transpose()
    }
}

/// Fixed-size scalar observations. No reservation or heap storage is needed.
/// Each new solve clears the samples; a failed callback never stores a trial.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScalarMinimumWorkspace<S: RealScalar> {
    best: Option<MinimumSample<S>>,
    samples: Option<[MinimumSample<S>; 3]>,
}
impl<S: RealScalar> ScalarMinimumWorkspace<S> {
    /// Create an empty workspace.
    pub(crate) const fn new() -> Self {
        Self {
            best: None,
            samples: None,
        }
    }
}

/// Minimizes a real function of one variable without derivatives.
///
/// The search starts from an interval, a bracket or two seed points (see
/// [`ScalarMinimumStart`]) and shrinks a bracket of a local minimum by
/// golden-section search or Brent's method (the default). Each iteration costs
/// at most one objective evaluation and O(1) memory; the solver needs no
/// workspace. The result is a local minimum within the tolerances of
/// [`ScalarMinimumOptions`].
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_optimize::{ScalarMinimizer, ScalarMinimumStart};
///
/// // (x − 0.25)² has its minimum at x = 0.25.
/// let objective = |x: f64| Ok((x - 0.25).powi(2));
/// let solver = ScalarMinimizer::default();
/// let start = ScalarMinimumStart::Bounded(Interval::new(-1.0, 2.0)?);
/// let result = solver.minimize(&objective, start)?;
/// assert!(result.is_success());
/// assert!((result.x() - 0.25).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. P. Brent, *Algorithms for Minimization without Derivatives*,
/// Prentice-Hall, 1973, ch. 5.
///
/// J. Kiefer, "Sequential minimax search for a maximum", *Proc. Amer. Math.
/// Soc.* 4(3), 502–506, 1953.
#[derive(Debug, Clone, Copy)]
pub struct ScalarMinimizer<S: RealScalar> {
    method: ScalarMinimumMethod,
    options: ScalarMinimumOptions<S>,
}
impl<S: RealScalar> Default for ScalarMinimizer<S> {
    fn default() -> Self {
        Self::new(ScalarMinimumMethod::Brent, ScalarMinimumOptions::default())
    }
}
impl<S: RealScalar> ScalarMinimizer<S> {
    /// Creates a minimizer with the given method and options.
    pub const fn new(method: ScalarMinimumMethod, options: ScalarMinimumOptions<S>) -> Self {
        Self { method, options }
    }
    /// Minimizes `objective` from `start` on the calling thread.
    ///
    /// When an iteration or evaluation limit stops the search, the result is
    /// returned with an unsuccessful termination.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a start point is NaN or infinite, or if no initial
    ///   sample has a finite value.
    /// - `InvalidInput` if the seeds of [`ScalarMinimumStart::Seeds`] are equal.
    /// - `ConvergenceFailure` if a limit stops the search before it has a
    ///   finite sample.
    /// - Errors from `objective`, unchanged.
    pub fn minimize<F: ScalarObjective<S> + ?Sized>(
        &self,
        objective: &F,
        start: ScalarMinimumStart<S>,
    ) -> Result<ScalarMinimumResult<S>, SciOxideError> {
        self.minimize_scoped(objective, start, ExecutionContext::sequential_native())
    }

    /// Minimizes `objective` from `start` within an execution context or a parent solve.
    ///
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// parent [`SolveContext`]. Evaluations also count toward the limits of
    /// the parent (see [solve accounting](scioxide_core#solve-accounting)). A
    /// limit, cancellation or deadline that stops the search after it has a
    /// finite sample returns the best sample with an unsuccessful termination.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::minimize`], and `ExecutionStopped` on cancellation
    /// or deadline before the search has a finite sample.
    pub fn minimize_scoped<'scope, 'capability, F: ScalarObjective<S> + ?Sized>(
        &self,
        objective: &F,
        start: ScalarMinimumStart<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<ScalarMinimumResult<S>, SciOxideError> {
        let identity = self.identity(start)?;
        let mut state = SolveState::new();
        scope
            .into()
            .run(&mut state, identity, self.budgets(), |context| {
                self.minimize_in_context(
                    objective,
                    start,
                    &mut ScalarMinimumWorkspace::new(),
                    context,
                )
            })
    }

    /// Finds a bracket of a local minimum by searching downhill from `first` and `second`.
    ///
    /// The bracket is not refined further. A successful result holds the
    /// bracket in [`ScalarMinimumResult::bracket`].
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a seed is NaN or infinite, or no sample has a finite value.
    /// - `InvalidInput` if `first == second`.
    /// - `ConvergenceFailure` if a limit stops the search before it has a
    ///   finite sample.
    /// - Errors from `objective`, unchanged.
    pub fn bracket_minimum<F: ScalarObjective<S> + ?Sized>(
        &self,
        objective: &F,
        first: S,
        second: S,
    ) -> Result<ScalarMinimumResult<S>, SciOxideError> {
        self.bracket_minimum_scoped(
            objective,
            first,
            second,
            ExecutionContext::sequential_native(),
        )
    }

    /// Finds a bracket of a local minimum within an execution context or a parent solve.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::bracket_minimum`], and `ExecutionStopped` on
    /// cancellation or deadline before the search has a finite sample.
    pub fn bracket_minimum_scoped<'scope, 'capability, F: ScalarObjective<S> + ?Sized>(
        &self,
        objective: &F,
        first: S,
        second: S,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<ScalarMinimumResult<S>, SciOxideError> {
        let start = ScalarMinimumStart::Seeds(first, second);
        validate_start(start)?;
        let identity = self.epoch_identity(
            scioxide_core::algorithm_id!("optimize", "scalar-bracket", 1),
            start,
        );
        let mut state = SolveState::new();
        scope
            .into()
            .run(&mut state, identity, self.budgets(), |context| {
                self.run(
                    objective,
                    start,
                    &mut ScalarMinimumWorkspace::new(),
                    &SolveSession::new(context, scioxide_core::argument!("scope")),
                    true,
                )
            })
    }

    /// Identity of a minimization epoch; initialization and options are included.
    pub(crate) fn identity(
        &self,
        start: ScalarMinimumStart<S>,
    ) -> Result<SolveIdentity, SciOxideError> {
        validate_start(start)?;
        Ok(self.epoch_identity(
            match self.method {
                ScalarMinimumMethod::GoldenSection => {
                    scioxide_core::algorithm_id!("optimize", "scalar-golden", 1)
                }
                ScalarMinimumMethod::Brent => {
                    scioxide_core::algorithm_id!("optimize", "scalar-brent", 1)
                }
            },
            start,
        ))
    }

    // Structured scalar parts avoid formatting floating-point options on every solve.
    fn epoch_identity(
        &self,
        algorithm: AlgorithmId,
        start: ScalarMinimumStart<S>,
    ) -> SolveIdentity {
        fn mix(token: &mut u128, word: u128) {
            *token = (*token ^ word).wrapping_mul(0x100000001b3);
        }
        fn scalar<S: RealScalar>(token: &mut u128, value: S) {
            match value.decompose() {
                scioxide_core::FloatDecomposition::Finite {
                    significand,
                    exponent,
                    negative,
                } => {
                    mix(token, u128::from(significand));
                    mix(token, u128::from(exponent as u16));
                    mix(token, u128::from(negative));
                }
                scioxide_core::FloatDecomposition::Infinite { negative } => {
                    mix(token, u128::MAX - u128::from(negative))
                }
                scioxide_core::FloatDecomposition::Nan => mix(token, u128::MAX - 2),
            }
        }
        let mut token = 0xcbf29ce484222325;
        for value in [
            self.options.step.absolute(),
            self.options.step.relative(),
            self.options.function.absolute(),
            self.options.function.relative(),
        ] {
            scalar(&mut token, value);
        }
        mix(&mut token, self.options.iterations.get() as u128);
        mix(&mut token, self.options.evaluations.get() as u128);
        match start {
            ScalarMinimumStart::Bounded(interval) => {
                mix(&mut token, 0);
                scalar(&mut token, interval.lower());
                scalar(&mut token, interval.upper());
            }
            ScalarMinimumStart::Seeds(a, b) => {
                mix(&mut token, 1);
                scalar(&mut token, a);
                scalar(&mut token, b);
            }
            ScalarMinimumStart::Bracket(bracket) => {
                mix(&mut token, 2);
                for sample in bracket.samples {
                    scalar(&mut token, sample.x);
                    scalar(&mut token, sample.value);
                }
            }
        }
        SolveIdentity::new(scioxide_core::ProblemToken::new(0), algorithm, 1, token)
    }

    /// Minimize under an epoch the caller has already begun.
    pub(crate) fn minimize_in_context<F: ScalarObjective<S> + ?Sized>(
        &self,
        objective: &F,
        start: ScalarMinimumStart<S>,
        workspace: &mut ScalarMinimumWorkspace<S>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<ScalarMinimumResult<S>, SciOxideError> {
        self.run(
            objective,
            start,
            workspace,
            &SolveSession::new(context, scioxide_core::argument!("scope")),
            false,
        )
    }
    fn budgets(&self) -> SolveBudgets {
        SolveBudgets::unbounded()
            .with_limit(BudgetKind::Iteration, self.options.iterations)
            .with_limit(BudgetKind::Evaluation, self.options.evaluations)
    }
    fn run<F: ScalarObjective<S> + ?Sized>(
        &self,
        objective: &F,
        start: ScalarMinimumStart<S>,
        workspace: &mut ScalarMinimumWorkspace<S>,
        session: &SolveSession<'_, '_, '_>,
        bracket_only: bool,
    ) -> Result<ScalarMinimumResult<S>, SciOxideError> {
        *workspace = ScalarMinimumWorkspace::new();
        let mut shell = Shell {
            objective,
            workspace,
            session,
            options: self.options,
        };
        let outcome = initialize(&mut shell, start).and_then(|samples| {
            if !samples[1].value.is_finite() {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Callback,
                    "no finite objective value among the initial samples",
                ));
            }
            shell.workspace.samples = Some(samples);
            shell.workspace.best = Some(samples[1]);
            shell.session.check()?;
            if bracket_only {
                Ok(OptimizeTermination::Converged)
            } else {
                contract(&mut shell, samples, self.method)
            }
        });
        let termination = match outcome {
            Ok(value) => value,
            Err(error) => match (session.stopped(), shell.workspace.best) {
                (Some(stop), Some(_)) => stop,
                _ => return Err(error),
            },
        };
        let best = shell.workspace.best.ok_or_else(|| {
            invalid(
                scioxide_core::argument!("objective"),
                "no finite scalar observation",
            )
        })?;
        // Published observations are finite; a rejected endpoint withholds them.
        let samples = shell
            .workspace
            .samples
            .filter(|samples| samples.iter().all(|sample| sample.value.is_finite()));
        let termination =
            if bracket_only && termination == OptimizeTermination::Converged && samples.is_none() {
                OptimizeTermination::Stagnation
            } else {
                termination
            };
        Ok(ScalarMinimumResult {
            best,
            samples,
            bracket: samples
                .filter(|s| valid_bracket(*s))
                .map(|samples| MinimumBracket { samples }),
            termination,
            counts: session.accounting()?,
            algorithm: session.algorithm()?,
        })
    }
}

use crate::validation::invalid;
fn finite<S: RealScalar>(x: S, origin: NonFiniteOrigin) -> Result<(), SciOxideError> {
    if x.is_finite() {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(origin, "finite scalar required"))
    }
}
fn validate_start<S: RealScalar>(start: ScalarMinimumStart<S>) -> Result<(), SciOxideError> {
    if let ScalarMinimumStart::Seeds(a, b) = start {
        finite(a, NonFiniteOrigin::Input)?;
        finite(b, NonFiniteOrigin::Input)?;
        if a == b {
            return Err(invalid(
                scioxide_core::argument!("start"),
                "bracketing seeds must be distinct",
            ));
        }
    }
    Ok(())
}
fn valid_bracket<S: RealScalar>([a, b, c]: [MinimumSample<S>; 3]) -> bool {
    a.x < b.x
        && b.x < c.x
        && b.value <= a.value
        && b.value <= c.value
        && (b.value < a.value || b.value < c.value)
}

struct Shell<'a, 'b, 'c, 'd, 'e, S: RealScalar, F: ?Sized> {
    objective: &'a F,
    workspace: &'a mut ScalarMinimumWorkspace<S>,
    session: &'b SolveSession<'c, 'd, 'e>,
    options: ScalarMinimumOptions<S>,
}
impl<S: RealScalar, F: ScalarObjective<S> + ?Sized> Shell<'_, '_, '_, '_, '_, S, F> {
    #[inline(always)]
    fn eval(&mut self, x: S) -> Result<MinimumSample<S>, SciOxideError> {
        if self.session.provider_count(EvaluationRole::Objective)? >= self.options.evaluations.get()
        {
            self.session
                .set_stopped(OptimizeTermination::EvaluationLimit);
            return Err(self
                .session
                .stop_error(OptimizeTermination::EvaluationLimit)?);
        }
        finite(x, NonFiniteOrigin::Intermediate)?;
        let value = self
            .session
            .invoke(EvaluationRole::Objective, || self.objective.eval(x))?;
        // A non-finite value rejects the trial: it ranks above every finite
        // observation, is never the best sample and is never published.
        if !value.is_finite() {
            return Ok(MinimumSample {
                x,
                value: S::INFINITY,
            });
        }
        let sample = MinimumSample { x, value };
        if self.workspace.best.is_none_or(|best| value < best.value) {
            self.session.accept(PartialResultTrust::Admissible)?;
            self.workspace.best = Some(sample);
        }
        Ok(sample)
    }
    #[inline(always)]
    fn iteration(&self) -> Result<(), SciOxideError> {
        if self.session.counts()?.iterations() >= self.options.iterations.get() {
            self.session
                .set_stopped(OptimizeTermination::IterationLimit);
            return Err(self
                .session
                .stop_error(OptimizeTermination::IterationLimit)?);
        }
        if let Some(stop) = self.session.trial_iteration()? {
            return Err(self.session.stop_error(stop)?);
        }
        Ok(())
    }
}

fn initialize<S: RealScalar, F: ScalarObjective<S> + ?Sized>(
    shell: &mut Shell<'_, '_, '_, '_, '_, S, F>,
    start: ScalarMinimumStart<S>,
) -> Result<[MinimumSample<S>; 3], SciOxideError> {
    match start {
        ScalarMinimumStart::Seeds(a, b) => bracket(shell, a, b),
        ScalarMinimumStart::Bracket(input) => {
            let [a, b, c] = input.samples;
            let samples = [shell.eval(a.x)?, shell.eval(b.x)?, shell.eval(c.x)?];
            if !valid_bracket(samples) {
                return Err(invalid(
                    scioxide_core::argument!("start"),
                    "objective does not satisfy minimum bracket",
                ));
            }
            Ok(samples)
        }
        ScalarMinimumStart::Bounded(bounds) => {
            let a = shell.eval(bounds.lower())?;
            if bounds.lower() == bounds.upper() {
                return Ok([a, a, a]);
            }
            let c = shell.eval(bounds.upper())?;
            let middle = blend(a.x, c.x, S::HALF);
            let b = if middle > a.x && middle < c.x {
                shell.eval(middle)?
            } else if a.value <= c.value {
                a
            } else {
                c
            };
            // A bounded minimum may be an endpoint.
            let best = [b, c].into_iter().fold(a, |best, sample| {
                if sample.value < best.value {
                    sample
                } else {
                    best
                }
            });
            Ok([a, best, c])
        }
    }
}

fn bracket<S: RealScalar, F: ScalarObjective<S> + ?Sized>(
    shell: &mut Shell<'_, '_, '_, '_, '_, S, F>,
    first: S,
    second: S,
) -> Result<[MinimumSample<S>; 3], SciOxideError> {
    let mut a = shell.eval(first)?;
    let mut b = shell.eval(second)?;
    if !(a.value.is_finite() || b.value.is_finite()) {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Callback,
            "neither bracketing seed has a finite objective value",
        ));
    }
    if b.value > a.value {
        core::mem::swap(&mut a, &mut b);
    }
    loop {
        shell.iteration()?;
        // Doubling the last step avoids a precision-dependent growth constant.
        let two = S::ONE + S::ONE;
        let next = match scioxide_core::checked_sum_of_products(&[b.x, a.x], &[two + S::ONE, -two])
        {
            Ok(next) => next,
            Err(error) => {
                shell.session.set_stopped(OptimizeTermination::Stagnation);
                return Err(error);
            }
        };
        if !next.is_finite() || next == b.x {
            shell.session.set_stopped(OptimizeTermination::Stagnation);
            return Err(invalid(
                scioxide_core::argument!("objective"),
                "bracket expansion has no finite representable progress",
            ));
        }
        let c = shell.eval(next)?;
        let samples = if a.x < c.x { [a, b, c] } else { [c, b, a] };
        if valid_bracket(samples) {
            return Ok(samples);
        }
        // Three equal values leave no descent to bracket along this line.
        if a.value == b.value && b.value == c.value {
            shell.session.set_stopped(OptimizeTermination::Stagnation);
            return Err(invalid(
                scioxide_core::argument!("objective"),
                "objective is flat along the bracketing line",
            ));
        }
        a = b;
        b = c;
    }
}

fn golden<S: RealScalar>() -> S {
    (S::ONE + S::ONE + S::ONE - (S::ONE + S::ONE + S::ONE + S::ONE + S::ONE).sqrt()) * S::HALF
}
// Convex interpolation avoids overflowing opposite-sign endpoint differences.
fn blend<S: RealScalar>(a: S, b: S, t: S) -> S {
    if (a < S::ZERO) == (b < S::ZERO) {
        a + (b - a) * t
    } else {
        a * (S::ONE - t) + b * t
    }
}
fn distance<S: RealScalar>(a: S, b: S) -> S {
    (a * S::HALF - b * S::HALF).abs()
}
fn within<S: RealScalar>(a: S, b: S, threshold: S) -> bool {
    let delta = (a - b).abs();
    delta.is_finite() && delta <= threshold
}
#[inline(always)]
fn converged<S: RealScalar>(
    s: [MinimumSample<S>; 3],
    options: ScalarMinimumOptions<S>,
) -> Result<bool, SciOxideError> {
    let [a, x, b] = s;
    if !(a.value.is_finite() && b.value.is_finite()) {
        return Ok(false);
    }
    let xt = options.step.threshold(a.x.abs().max(b.x.abs()))?;
    let ft = options
        .function
        .threshold(a.value.abs().max(x.value.abs()).max(b.value.abs()))?;
    Ok(within(a.x, b.x, xt) && within(a.value, x.value, ft) && within(b.value, x.value, ft))
}

// A normalized parabola keeps both coordinate and objective products in range.
// Accept only a strict interior step shorter than half the preceding safeguard.
fn parabolic<S: RealScalar>(
    x: MinimumSample<S>,
    w: MinimumSample<S>,
    v: MinimumSample<S>,
    a: S,
    b: S,
    previous: S,
) -> Option<S> {
    let scale = x.x.abs().max(w.x.abs()).max(v.x.abs());
    let fs = x.value.abs().max(w.value.abs()).max(v.value.abs());
    if scale == S::ZERO || fs == S::ZERO {
        return None;
    }
    let dxw = x.x / scale - w.x / scale;
    let dxv = x.x / scale - v.x / scale;
    let r = dxw * (x.value / fs - v.value / fs);
    let q = dxv * (x.value / fs - w.value / fs);
    let denominator = (S::ONE + S::ONE) * (q - r);
    if denominator == S::ZERO {
        return None;
    }
    let step = (dxv * q - dxw * r) / denominator;
    let u = x.x - step * scale;
    if u.is_finite() && u > a && u < b && u != x.x && distance(u, x.x) < previous * S::HALF {
        Some(u)
    } else {
        None
    }
}

fn contract<S: RealScalar, F: ScalarObjective<S> + ?Sized>(
    shell: &mut Shell<'_, '_, '_, '_, '_, S, F>,
    mut samples: [MinimumSample<S>; 3],
    method: ScalarMinimumMethod,
) -> Result<OptimizeTermination, SciOxideError> {
    let mut w = samples[1];
    let mut v = w;
    let mut previous = distance(samples[0].x, samples[2].x);
    loop {
        shell.session.check()?;
        if converged(samples, shell.options)? {
            return Ok(OptimizeTermination::Converged);
        }
        let [a, x, b] = samples;
        let target = if distance(a.x, x.x) > distance(x.x, b.x) {
            a.x
        } else {
            b.x
        };
        let proposal = if method == ScalarMinimumMethod::Brent {
            parabolic(x, w, v, a.x, b.x, previous)
        } else {
            None
        };
        let mut u = proposal.unwrap_or_else(|| blend(x.x, target, golden::<S>()));
        if u <= a.x || u >= b.x || u == x.x {
            u = x.x.next_after(target);
        }
        if u <= a.x || u >= b.x || u == x.x {
            // The longer side can tie in rounded arithmetic; try the other side.
            let other = if target == a.x { b.x } else { a.x };
            u = x.x.next_after(other);
            if u <= a.x || u >= b.x || u == x.x {
                return Ok(OptimizeTermination::Stagnation);
            }
        }
        shell.iteration()?;
        let trial = shell.eval(u)?;
        previous = if proposal.is_some() {
            distance(x.x, u)
        } else {
            distance(x.x, target)
        };
        if trial.value < x.value {
            samples = if u < x.x {
                [a, trial, x]
            } else {
                [x, trial, b]
            };
            v = w;
            w = x;
        } else {
            samples = if u < x.x {
                [trial, x, b]
            } else {
                [a, x, trial]
            };
            if trial.value <= w.value || w.x == x.x {
                v = w;
                w = trial;
            } else if trial.value <= v.value || v.x == x.x || v.x == w.x {
                v = trial;
            }
        }
        shell.workspace.samples = Some(samples);
    }
}
