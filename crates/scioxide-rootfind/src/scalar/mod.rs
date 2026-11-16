//! Real and complex scalar root solvers.
//!
//! Bracketing methods share the loop [`solve_bracket`] and differ in their
//! [`BracketRule`]. A proposal outside the bracket is replaced by the
//! midpoint. Secant, Newton, Halley and Muller iterations, real or complex,
//! share the loop in [`open`] and differ in their proposals. The corrections
//! and the small-step test live in [`correction`]. Every callback value goes
//! through [`evaluate`].

/// Implement [`crate::ScalarRootSolver`] for a bracketing solver whose rule is
/// built by `$start` from the evaluated initial bracket.
macro_rules! bracket_solver {
    ($solver:ident, $start:expr) => {
        impl<S: scioxide_core::RealScalar> crate::ScalarRootSolver<S> for $solver {
            type Initialization = scioxide_core::Interval<S>;

            fn name(&self) -> &'static str {
                stringify!($solver)
            }

            fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
                scioxide_core::algorithm_id!("rootfind", stringify!($solver), 1)
            }

            #[inline]
            fn solve_scoped<'scope, 'capability, F: crate::ScalarFunction<S> + ?Sized>(
                &self,
                f: &F,
                init: scioxide_core::Interval<S>,
                options: &crate::ScalarRootOptions<S>,
                scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
            ) -> Result<crate::RootResult<S>, scioxide_core::SciOxideError> {
                crate::context::run(
                    scope.into(),
                    <Self as crate::ScalarRootSolver<S>>::algorithm_id(self),
                    options.budgets(),
                    |context| {
                        crate::scalar::solve_bracket(
                            f,
                            init,
                            options,
                            scioxide_core::argument!("init"),
                            $start,
                            context,
                        )
                    },
                )
            }
        }
    };
}

mod advanced_bracket;
mod bisection;
mod complex_methods;
pub(crate) mod correction;
mod derivative_methods;
mod dispatch;
mod find_root;
mod illinois;
mod muller;
mod open;
mod options;
mod regula_falsi;
mod ridder;
mod secant;

pub use advanced_bracket::{Brent, Brenth, Chandrupatla, Toms748};
pub use bisection::Bisection;
pub use complex_methods::{ComplexHalley, ComplexNewton, ComplexSecant};
pub use derivative_methods::{
    Halley, Newton, SafeguardedHalley, SafeguardedInitialGuess, SafeguardedNewton,
};
pub use dispatch::{ScalarRootInput, ScalarRootMethod};
pub use find_root::find_root;
pub use illinois::Illinois;
pub use muller::{ComplexMuller, Muller};
pub use options::ScalarRootOptions;
pub use regula_falsi::RegulaFalsi;
pub use ridder::Ridder;
pub use secant::Secant;

use scioxide_core::checked_sum_of_products;
use scioxide_core::{
    AlgebraicScalar, Complex, EvaluationRole, Interval, NonFiniteOrigin, NumericScalar, RealScalar,
    SciOxideError, SolveContext, Tolerance,
};

use crate::context::Session;
use crate::initialization::Bracket;
use crate::result::{ComplexRootResult, RootResult, RootTermination};
use crate::traits::{ScalarFunction, finite_output};

/// Counts and calls `callback` at a finite point, and checks its value.
///
/// The point must be finite before the call and every component of the value
/// finite after it.
#[inline(always)]
pub(crate) fn evaluate<T: NumericScalar>(
    session: &Session<'_, '_, '_>,
    role: EvaluationRole,
    x: T,
    callback: impl FnOnce(T, &mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    if !x.has_finite_components() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "solver candidate must be finite before callback evaluation",
        ));
    }
    finite_output(session.invoke_context(role, |context| callback(x, context))?)
}

/// An accepted sample with its evaluated sign bracket, if any.
type Accepted<T> = (Sample<T>, Option<BracketState<<T as NumericScalar>::Real>>);

/// State of one real or complex scalar solve: the last accepted sample with
/// its bracket, the best observation and the residual scale. The result's
/// [`Bracket`] is built only when the solve returns.
pub(crate) struct Run<'a, 'state, 'capability, T: NumericScalar> {
    session: Session<'a, 'state, 'capability>,
    evaluation_role: EvaluationRole,
    accepted: Option<Accepted<T>>,
    /// Best observation and its residual magnitude.
    best: Option<(Sample<T>, T::Real)>,
    pub(crate) scale: T::Real,
}

impl<'a, 'state, 'capability, T: NumericScalar> Run<'a, 'state, 'capability, T> {
    /// Validate `options` and borrow `context` for one solve.
    ///
    /// # Errors
    /// Rejects an invalid residual scale before any callback.
    pub(crate) fn new(
        context: &'a mut SolveContext<'state, 'capability>,
        options: &ScalarRootOptions<T::Real>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        Ok(Self {
            session: Session::new(context),
            evaluation_role: options.evaluation_role(),
            accepted: None,
            best: None,
            scale: options
                .residual_scale
                .unwrap_or(<T::Real as AlgebraicScalar>::ONE),
        })
    }

    /// Counts one provider call, stopping on cancellation or an exhausted limit.
    #[inline(always)]
    pub(crate) fn admit(&self, role: EvaluationRole) -> Result<(), SciOxideError> {
        self.session.admit(role)
    }

    /// Counts one further proposal within the current iteration.
    #[inline(always)]
    pub(crate) fn propose(&self) -> Result<(), SciOxideError> {
        self.session.proposal()
    }

    /// Counts the iteration after its starting state has been accepted.
    #[inline(always)]
    pub(crate) fn begin_iteration(&self) -> Result<(), SciOxideError> {
        self.session.iteration()
    }

    /// Counts, calls and checks an analytic derivative provider.
    ///
    /// # Errors
    /// Stops at a limit, propagates provider errors, reports a missing
    /// derivative as invalid input `f` and a non-finite one as a callback error.
    #[inline(always)]
    pub(crate) fn derivative(
        &self,
        role: EvaluationRole,
        provider: impl FnOnce() -> Result<Option<T>, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        self.session
            .invoke(role, provider)?
            .ok_or_else(missing_derivative)
            .and_then(finite_output)
    }

    /// Records `sample` as the accepted state. A new point advances the
    /// accepted revision; a repeated point keeps its earlier assessment.
    #[inline(always)]
    pub(crate) fn accept(
        &mut self,
        sample: Sample<T>,
        bracket: Option<BracketState<T::Real>>,
    ) -> Result<(), SciOxideError> {
        match self.accepted {
            Some((previous, _)) if previous.x != sample.x => self.session.accept()?,
            None => self.session.assess()?,
            _ => {}
        }
        self.accepted = Some((sample, bracket));
        Ok(())
    }

    /// Counted, checked evaluation of `callback` at `x`, with its residual
    /// magnitude. Every observation, including finite-difference probes, can
    /// become the best sample; the first one becomes the accepted state.
    #[inline(always)]
    pub(crate) fn sample(
        &mut self,
        x: T,
        callback: impl FnOnce(T, &mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
    ) -> Result<(Sample<T>, T::Real), SciOxideError> {
        let (sample, residual) = Sample::checked(
            x,
            evaluate(&self.session, self.evaluation_role, x, callback)?,
        )?;
        // Ties retain the earlier observation.
        if self.best.is_none_or(|(_, best)| residual < best) {
            self.best = Some((sample, residual));
        }
        if self.accepted.is_none() {
            self.accept(sample, None)?;
        }
        Ok((sample, residual))
    }

    /// Best function observation so far, or `sample` when there is none
    /// better; ties keep the earlier observation.
    pub(crate) fn best_sample(&self, sample: Sample<T>) -> Sample<T> {
        match self.best {
            Some((best, residual)) if residual <= sample.residual() => best,
            _ => sample,
        }
    }

    /// Returns `sample` with its termination and bracket as the result.
    pub(crate) fn result<R: Root<T>>(
        &mut self,
        sample: Sample<T>,
        termination: RootTermination,
        bracket: Option<BracketState<T::Real>>,
    ) -> Result<R, SciOxideError> {
        self.accept(sample, bracket)?;
        Ok(R::assemble(
            sample,
            termination,
            bracket.map(BracketState::certificate).transpose()?,
            self.session.algorithm()?,
            self.session.accounting()?,
            crate::context::residual(sample.residual(), self.scale)?,
        ))
    }

    /// Turn a budget, cancellation or deadline stop into the last accepted
    /// state; every other outcome passes through unchanged.
    #[inline(always)]
    pub(crate) fn finish<R: Root<T>>(
        &mut self,
        outcome: Result<R, SciOxideError>,
    ) -> Result<R, SciOxideError> {
        match (outcome, self.session.stopped.get(), self.accepted) {
            (Err(_), Some(reason), Some((sample, bracket))) => self.result(sample, reason, bracket),
            (outcome, _, _) => outcome,
        }
    }
}

impl<S: RealScalar> Run<'_, '_, '_, S> {
    /// Counted, checked evaluation through [`ScalarFunction::eval_scoped`].
    #[inline(always)]
    pub(crate) fn evaluate<F: ScalarFunction<S> + ?Sized>(
        &mut self,
        f: &F,
        x: S,
    ) -> Result<Sample<S>, SciOxideError> {
        Ok(self.sample(x, |x, context| f.eval_scoped(x, context))?.0)
    }
}

/// The solve entry point's callback `f` lacks a required analytic derivative.
fn missing_derivative() -> SciOxideError {
    SciOxideError::invalid_input(
        scioxide_core::argument!("f"),
        scioxide_core::InvalidInputReason::Inconsistent,
        "requested analytic derivative is unavailable",
    )
}

/// A scalar solve result with point type `T`: [`RootResult`] for real and
/// [`ComplexRootResult`] for complex solves, built once by [`Run::result`].
pub(crate) trait Root<T: NumericScalar> {
    /// Assemble the result; complex results carry no bracket.
    fn assemble(
        sample: Sample<T>,
        termination: RootTermination,
        bracket: Option<Bracket<T::Real>>,
        algorithm: scioxide_core::AlgorithmId,
        counts: scioxide_core::SolveCounts,
        residual: scioxide_core::ResidualReport<T::Real>,
    ) -> Self;
}

impl<S: RealScalar> Root<S> for RootResult<S> {
    #[inline(always)]
    fn assemble(
        sample: Sample<S>,
        termination: RootTermination,
        bracket: Option<Bracket<S>>,
        algorithm: scioxide_core::AlgorithmId,
        counts: scioxide_core::SolveCounts,
        residual: scioxide_core::ResidualReport<S>,
    ) -> Self {
        Self {
            root: sample.x,
            function_value: sample.value,
            termination,
            bracket,
            algorithm,
            counts,
            residual,
            trust: scioxide_core::PartialResultTrust::ResidualChecked,
        }
    }
}

impl<S: RealScalar> Root<Complex<S>> for ComplexRootResult<S> {
    #[inline(always)]
    fn assemble(
        sample: Sample<Complex<S>>,
        termination: RootTermination,
        _: Option<Bracket<S>>,
        algorithm: scioxide_core::AlgorithmId,
        counts: scioxide_core::SolveCounts,
        residual: scioxide_core::ResidualReport<S>,
    ) -> Self {
        Self {
            root: sample.x,
            function_value: sample.value,
            termination,
            algorithm,
            counts,
            residual,
            trust: scioxide_core::PartialResultTrust::ResidualChecked,
        }
    }
}

/// One checked observation.
///
/// The residual magnitude is derived rather than stored, so real samples,
/// bracket states and the open loop's point history stay two scalars wide.
/// Paths that compare complex residuals repeatedly carry the magnitude
/// [`Run::sample`] computed alongside the sample.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sample<T: NumericScalar> {
    pub(crate) x: T,
    pub(crate) value: T,
}

impl<T: NumericScalar> Sample<T> {
    /// Pair a finite observation with its residual magnitude.
    ///
    /// # Errors
    /// Rejects a complex modulus outside the finite range.
    #[inline(always)]
    pub(crate) fn checked(x: T, value: T) -> Result<(Self, T::Real), SciOxideError> {
        let sample = Self { x, value };
        let residual = sample.residual();
        if !T::IS_REAL {
            scioxide_core::backend::checked_final_rounding(
                residual,
                "raw residual magnitude is outside the finite range",
            )?;
        }
        Ok((sample, residual))
    }

    /// Absolute value or complex modulus of `value`.
    #[inline(always)]
    pub(crate) fn residual(self) -> T::Real {
        let (re, im) = (
            self.value.real_component(),
            self.value.imaginary_component(),
        );
        if T::IS_REAL || im == <T::Real as AlgebraicScalar>::ZERO {
            re.abs()
        } else {
            re.hypot(im)
        }
    }
}

impl<S: RealScalar> Sample<S> {
    /// Real observation.
    pub(crate) const fn new(x: S, value: S) -> Self {
        Self { x, value }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BracketState<S: RealScalar> {
    pub(crate) lower: Sample<S>,
    pub(crate) upper: Sample<S>,
}

/// Sign bracket of a solver's initial interval; a missing sign change is
/// reported against `argument`, the caller's name for the interval.
impl<S: RealScalar> BracketState<S> {
    #[inline(always)]
    pub(crate) fn evaluate<F: ScalarFunction<S> + ?Sized>(
        f: &F,
        interval: Interval<S>,
        run: &mut Run<'_, '_, '_, S>,
        argument: scioxide_core::Argument,
    ) -> Result<Self, SciOxideError> {
        let lower = run.evaluate(f, interval.lower())?;
        let upper = if interval.lower() == interval.upper() {
            lower
        } else {
            run.evaluate(f, interval.upper())?
        };
        crate::initialization::validate_sign_change(lower.value, upper.value, argument)?;
        Ok(Self { lower, upper })
    }

    pub(crate) fn certificate(self) -> Result<Bracket<S>, SciOxideError> {
        Bracket::from_evaluated(
            Interval::new(self.lower.x, self.upper.x)?,
            self.lower.value,
            self.upper.value,
            scioxide_core::argument!("init"),
        )
    }

    #[inline(always)]
    pub(crate) fn best(self) -> Sample<S> {
        if self.lower.value.abs() <= self.upper.value.abs() {
            self.lower
        } else {
            self.upper
        }
    }

    /// Extended-range width, including brackets wider than the scalar range.
    pub(crate) fn width(self) -> Result<scioxide_core::Scaled<S>, SciOxideError> {
        scioxide_core::Scaled::from_value(self.upper.x)?
            .sub(scioxide_core::Scaled::from_value(self.lower.x)?)
    }

    /// Half-width contraction without extended arithmetic in the normal range.
    pub(crate) fn half_width_contracted(self, start: Self) -> Result<bool, SciOxideError> {
        let width = self.upper.x - self.lower.x;
        let original = start.upper.x - start.lower.x;
        if width.is_finite() && original.is_normal() && (original * S::HALF).is_normal() {
            return Ok(width <= original * S::HALF);
        }
        Ok(self
            .width()?
            .abs_le(start.width()?.mul(scioxide_core::Scaled::HALF)?))
    }

    #[inline(always)]
    pub(crate) fn midpoint(self) -> Result<S, SciOxideError> {
        let sum = self.lower.x + self.upper.x;
        if sum.is_finite() && (sum == S::ZERO || sum.abs() >= S::MIN_POSITIVE / S::EPSILON) {
            Ok(sum * S::HALF)
        } else {
            // Opposite maximum-scale endpoints and gradual underflow retain
            // the shared correctly rounded midpoint contract.
            scioxide_core::checked_midpoint(self.lower.x, self.upper.x)
        }
    }

    #[inline(always)]
    pub(crate) fn width_ok(self, tolerance: &Tolerance<S>) -> Result<bool, SciOxideError> {
        correction::within(tolerance, self.lower.x, self.upper.x)
    }

    /// One termination policy for every evaluated sign bracket.
    #[inline(always)]
    pub(crate) fn termination(
        self,
        options: &ScalarRootOptions<S>,
        scale: S,
    ) -> Result<Option<RootTermination>, SciOxideError> {
        if let Some(reason) = options.residual_termination(self.best().value, scale, false, true)? {
            return Ok(Some(reason));
        }
        Ok(self
            .width_ok(&options.step_tolerance.tolerance())?
            .then_some(RootTermination::BracketTolerance))
    }

    pub(crate) fn contains_strict(self, x: S) -> bool {
        x > self.lower.x && x < self.upper.x
    }

    /// The safeguarded step: `proposal` when it lies strictly inside the
    /// bracket, otherwise the midpoint. The flag reports an accepted proposal.
    #[inline(always)]
    pub(crate) fn safeguard(self, proposal: Option<S>) -> Result<(S, bool), SciOxideError> {
        match proposal.filter(|&x| self.contains_strict(x)) {
            Some(x) => Ok((x, true)),
            None => Ok((self.midpoint()?, false)),
        }
    }

    #[inline(always)]
    pub(crate) fn update(self, candidate: Sample<S>) -> Result<Self, SciOxideError> {
        use crate::initialization::brackets_root;
        if candidate.value == S::ZERO {
            return Ok(Self {
                lower: candidate,
                upper: candidate,
            });
        }
        if brackets_root(self.lower.value, candidate.value) {
            Ok(Self {
                lower: self.lower,
                upper: candidate,
            })
        } else if brackets_root(candidate.value, self.upper.value) {
            Ok(Self {
                lower: candidate,
                upper: self.upper,
            })
        } else {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("init"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "finite candidate did not preserve the root certificate",
            ))
        }
    }
}

/// One bracketing method: its candidates within an iteration and the history
/// it keeps between them.
pub(crate) trait BracketRule<S: RealScalar> {
    /// Candidate for `step` of the zero-based iteration `cycle`, or `None` to
    /// end the iteration. Step zero always proposes; `start` is the bracket the
    /// iteration began with. A candidate outside `bracket` ends the solve with
    /// [`RootTermination::Stagnation`].
    fn propose(
        &mut self,
        bracket: BracketState<S>,
        start: BracketState<S>,
        cycle: usize,
        step: usize,
        options: &ScalarRootOptions<S>,
    ) -> Result<Option<S>, SciOxideError>;

    /// Observe the evaluated `sample` that moved the bracket from `old` to `new`.
    fn observe(
        &mut self,
        old: BracketState<S>,
        new: BracketState<S>,
        sample: Sample<S>,
    ) -> Result<(), SciOxideError> {
        let _ = (old, new, sample);
        Ok(())
    }
}

/// The bracketing loop shared by every bracketing method.
///
/// Each iteration counts itself, then evaluates the rule's candidates in order;
/// candidates after the first count as further proposals. Every evaluation
/// updates the bracket, is accepted and is followed by the termination test.
#[inline(always)]
pub(crate) fn solve_bracket<S, F, R>(
    f: &F,
    interval: Interval<S>,
    options: &ScalarRootOptions<S>,
    argument: scioxide_core::Argument,
    start: impl FnOnce(BracketState<S>) -> R,
    context: &mut SolveContext<'_, '_>,
) -> Result<RootResult<S>, SciOxideError>
where
    S: RealScalar,
    F: ScalarFunction<S> + ?Sized,
    R: BracketRule<S>,
{
    let mut run = Run::new(context, options)?;
    let outcome = (|| {
        let mut bracket = BracketState::evaluate(f, interval, &mut run, argument)?;
        let scale = options
            .residual_scale
            .unwrap_or_else(|| residual_scale(&[bracket.lower.value, bracket.upper.value]));
        run.scale = scale;
        let mut rule = start(bracket);
        run.accept(bracket.best(), Some(bracket))?;
        if let Some(reason) = bracket.termination(options, scale)? {
            return run.result(bracket.best(), reason, Some(bracket));
        }
        for cycle in 0..options.max_iterations.get() {
            run.begin_iteration()?;
            let initial = bracket;
            for step in 0.. {
                let Some(x) = rule.propose(bracket, initial, cycle, step, options)? else {
                    break;
                };
                if step > 0 {
                    run.propose()?;
                }
                if !bracket.contains_strict(x) {
                    return run.result(bracket.best(), RootTermination::Stagnation, Some(bracket));
                }
                let sample = run.evaluate(f, x)?;
                let next = bracket.update(sample)?;
                rule.observe(bracket, next, sample)?;
                bracket = next;
                run.accept(bracket.best(), Some(bracket))?;
                if let Some(reason) = bracket.termination(options, scale)? {
                    return run.result(bracket.best(), reason, Some(bracket));
                }
            }
        }
        run.result(
            bracket.best(),
            RootTermination::IterationLimit,
            Some(bracket),
        )
    })();
    run.finish(outcome)
}

pub(crate) fn residual_scale<S: RealScalar>(values: &[S]) -> S {
    values
        .iter()
        .fold(S::ZERO, |scale, value| scale.max(value.abs()))
}

pub(crate) fn false_position<S: RealScalar>(
    lower: Sample<S>,
    upper: Sample<S>,
) -> Result<S, SciOxideError> {
    let scale = lower.value.abs().max(upper.value.abs());
    let lower_weight = lower.value.abs() / scale;
    let upper_weight = upper.value.abs() / scale;
    let total = lower_weight + upper_weight;
    let candidate = (upper_weight / total) * lower.x + (lower_weight / total) * upper.x;
    if candidate.is_finite() {
        return Ok(candidate);
    }
    checked_sum_of_products(
        &[upper_weight / total, lower_weight / total],
        &[lower.x, upper.x],
    )
}
