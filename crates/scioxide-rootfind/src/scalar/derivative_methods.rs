//! Newton and Halley iterations, open or safeguarded by a bracket.
use super::open::{self, OpenMethod, Step};
use super::{BracketState, Run, Sample, correction};
use crate::{RootResult, RootTermination, ScalarFunction, ScalarRootOptions, ScalarRootSolver};
use scioxide_core::{
    EvaluationRole, ExtendedInterval, Interval, NumericScalar, RealScalar, SciOxideError,
    SolveContext,
};
use scioxide_differentiate::DerivativeStrategy;

/// Holds an interval `[a, b]` and a starting point `x₀ ∈ [a, b]` for
/// [`SafeguardedNewton`] and [`SafeguardedHalley`].
///
/// The solver evaluates `f(a)` and `f(b)` and requires opposite signs or a
/// zero.
#[derive(Debug, Clone, Copy)]
pub struct SafeguardedInitialGuess<S: RealScalar> {
    interval: Interval<S>,
    guess: S,
}
impl<S: RealScalar> SafeguardedInitialGuess<S> {
    /// Combines `interval` with the starting point `guess`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` (origin `Input`) if `guess` is NaN or infinite.
    /// - `InvalidInput` if `guess` lies outside `interval`.
    pub fn new(interval: Interval<S>, guess: S) -> Result<Self, SciOxideError> {
        if !guess.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "initial point must be finite",
            ));
        }
        if guess < interval.lower() || guess > interval.upper() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("guess"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "initial point must be inside the interval",
            ));
        }
        Ok(Self { interval, guess })
    }
    /// Returns the interval `[a, b]`.
    pub const fn interval(&self) -> Interval<S> {
        self.interval
    }
    /// Returns the starting point `x₀`.
    pub const fn guess(&self) -> S {
        self.guess
    }
}

/// Finds a zero of `f` by Newton's method, starting from one point `x₀`.
///
/// Each iteration computes
///
/// ```text
/// xₖ₊₁ = xₖ − f(xₖ) / f′(xₖ),
/// ```
///
/// with `f′` from the [`DerivativeStrategy`] given to [`Newton::new`]: an
/// analytic derivative, automatic differentiation, or a forward or central
/// difference. [`Newton::default`] uses `AnalyticOrForward`: the analytic
/// derivative when the function provides one and a forward difference
/// otherwise. Each iteration costs one evaluation of `f` and one of `f′`, or
/// one or two more evaluations of `f` for a difference quotient. Convergence
/// to a simple zero is quadratic when `x₀` is close enough and `f′` is exact.
/// The iteration is not safeguarded; use [`SafeguardedNewton`] when a bracket
/// is known.
///
/// The solve stops as described in [`ScalarRootOptions`]. It ends with
/// `Stagnation` when `f′(xₖ) = 0` or a step repeats a point. After
/// `Stagnation` or `IterationLimit`, the result is the evaluated point with
/// the smallest `|f|`, including points evaluated for difference quotients.
///
/// # Errors
///
/// - `InvalidInput` if no source in the strategy provides `f′`.
/// - The errors listed on [`ScalarRootSolver::solve`].
///
/// # Examples
///
/// ```
/// use scioxide_rootfind::{Newton, ScalarFn, ScalarRootOptions, ScalarRootSolver};
///
/// // x³ − 2 = 0 has the real root ∛2.
/// let cube = ScalarFn::new(|x: f64| Ok(x * x * x - 2.0)).with_derivative(|x| Ok(3.0 * x * x));
/// let options = ScalarRootOptions::default();
/// let result = Newton::default().solve(&cube, 1.0, &options)?;
/// assert!(result.is_success());
/// assert!((result.point() - 2.0_f64.cbrt()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Newton {
    strategy: DerivativeStrategy,
}
/// Finds a zero of `f` in `[a, b]` by Newton's method safeguarded with
/// bisection.
///
/// The solve evaluates `f(a)` and `f(b)`, which must have opposite signs or
/// include a zero, and starts the Newton iteration at the point `x₀` of the
/// [`SafeguardedInitialGuess`]. Every evaluation updates the bracket. A
/// Newton step is accepted when it lands strictly inside the bracket and is at
/// most half as long as the step before last; otherwise the iteration bisects.
/// A Newton point that rounds onto an endpoint is moved to the adjacent
/// floating-point number inside. Near a simple zero the iteration converges
/// quadratically, and far from it the bisection steps keep shrinking the
/// bracket. Difference quotients only sample points in `[a, b]`.
/// [`SafeguardedNewton::default`] takes `f′` from the `AnalyticOrForward`
/// [`DerivativeStrategy`], as [`Newton::default`] does.
///
/// The solve stops as described for bracketing methods in
/// [`ScalarRootOptions`]. The result is the bracket endpoint with the smaller
/// `|f|`, and [`RootResult::bracket`] returns the final bracket.
///
/// # Errors
///
/// - `InvalidInput` if `f(a)` and `f(b)` have the same nonzero sign, or no
///   source in the strategy provides `f′`.
/// - The errors listed on [`ScalarRootSolver::solve`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SafeguardedNewton {
    strategy: DerivativeStrategy,
}
macro_rules! strategy {
    ($name:ident) => {
        impl $name {
            /// Creates the solver with the source of `f′` to use.
            pub const fn new(strategy: DerivativeStrategy) -> Self {
                Self { strategy }
            }
            /// Returns the source of `f′`.
            pub const fn derivative_strategy(&self) -> DerivativeStrategy {
                self.strategy
            }
        }
    };
}
strategy!(Newton);
strategy!(SafeguardedNewton);
/// Finds a zero of `f` by Halley's method, starting from one point `x₀`.
///
/// Each iteration computes
///
/// ```text
/// xₖ₊₁ = xₖ − (f/f′) / (1 − f f″ / (2 f′²)),
/// ```
///
/// evaluated at `xₖ`, and takes the Newton step `xₖ − f/f′` instead when
/// `|f f″ / (2 f′²)| ≥ 1`. It requires the analytic derivatives
/// [`ScalarFunction::derivative`] and [`ScalarFunction::second_derivative`].
/// Each iteration costs one evaluation each of `f`, `f′` and `f″`. Convergence
/// to a simple zero is cubic when `x₀` is close enough. The iteration is not
/// safeguarded; use [`SafeguardedHalley`] when a bracket is known.
///
/// The solve stops as described in [`ScalarRootOptions`]. It ends with
/// `Stagnation` when `f′(xₖ) = 0` or a step repeats a point. After
/// `Stagnation` or `IterationLimit`, the result is the evaluated point with
/// the smallest `|f|`.
///
/// # Errors
///
/// - `InvalidInput` if `f` does not provide `f′` or `f″`.
/// - The errors listed on [`ScalarRootSolver::solve`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Halley;
/// Finds a zero of `f` in `[a, b]` by Halley's method safeguarded with
/// bisection.
///
/// The iteration is that of [`SafeguardedNewton`] with the Halley step of
/// [`Halley`], and requires analytic `f′` and `f″`.
///
/// # Errors
///
/// - `InvalidInput` if `f(a)` and `f(b)` have the same nonzero sign, or `f`
///   does not provide `f′` or `f″`.
/// - The errors listed on [`ScalarRootSolver::solve`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SafeguardedHalley;

macro_rules! real_solver {
    ($name:ident, $init:ty, $get:expr, $policy:expr, $halley:expr, $guarded:expr) => {
        impl<S: RealScalar> ScalarRootSolver<S> for $name {
            type Initialization = $init;
            fn name(&self) -> &'static str {
                stringify!($name)
            }
            fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
                scioxide_core::algorithm_id!("rootfind", stringify!($name), 1)
            }

            fn solve_scoped<'scope, 'capability, F: ScalarFunction<S> + ?Sized>(
                &self,
                f: &F,
                init: Self::Initialization,
                options: &ScalarRootOptions<S>,
                scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
            ) -> Result<RootResult<S>, SciOxideError> {
                crate::context::run(
                    scope.into(),
                    <Self as ScalarRootSolver<S>>::algorithm_id(self),
                    options.budgets(),
                    |context| {
                        solve_real::<S, F, $halley, $guarded>(
                            f,
                            ($get)(init),
                            options,
                            ($policy)(self),
                            context,
                        )
                    },
                )
            }
        }
    };
}
real_solver!(
    Newton,
    S,
    |x| (x, None),
    |solver: &Newton| solver.strategy,
    false,
    false
);
real_solver!(
    Halley,
    S,
    |x| (x, None),
    |_: &Halley| DerivativeStrategy::Analytic,
    true,
    false
);
real_solver!(
    SafeguardedNewton,
    SafeguardedInitialGuess<S>,
    |x: SafeguardedInitialGuess<S>| (x.guess, Some(x.interval)),
    |solver: &SafeguardedNewton| solver.strategy,
    false,
    true
);
real_solver!(
    SafeguardedHalley,
    SafeguardedInitialGuess<S>,
    |x: SafeguardedInitialGuess<S>| (x.guess, Some(x.interval)),
    |_: &SafeguardedHalley| DerivativeStrategy::Analytic,
    true,
    true
);

/// First and, for Halley, second derivatives of a callback at an accepted point.
pub(super) trait Derivatives<T: NumericScalar> {
    /// Counted, checked derivatives at `current`; difference-quotient samples
    /// are evaluated through `run`.
    fn at(
        &self,
        run: &mut Run<'_, '_, '_, T>,
        current: Sample<T>,
    ) -> Result<(T, Option<T>), SciOxideError>;
}

/// Open Newton or Halley iteration over any derivative source, for real and
/// complex points.
pub(super) struct NewtonStep<D>(pub(super) D);

impl<T: NumericScalar, D: Derivatives<T>> OpenMethod<T, 1> for NewtonStep<D> {
    #[inline(always)]
    fn propose(
        &mut self,
        run: &mut Run<'_, '_, '_, T>,
        [current]: &[Sample<T>; 1],
    ) -> Result<Step<T>, SciOxideError> {
        let (first, second) = self.0.at(run, *current)?;
        correction::newton(current.x, current.value, first, second).map(Step::proposed)
    }
}

/// Real derivatives from the configured source of `f′`; Halley's second
/// derivative is always analytic. Difference-quotient samples stay inside
/// `probes`.
struct RealDerivatives<'f, F: ?Sized, S: RealScalar, const HALLEY: bool> {
    f: &'f F,
    strategy: DerivativeStrategy,
    probes: ExtendedInterval<S>,
}

impl<S, F, const HALLEY: bool> Derivatives<S> for RealDerivatives<'_, F, S, HALLEY>
where
    S: RealScalar,
    F: ScalarFunction<S> + ?Sized,
{
    #[inline(always)]
    fn at(
        &self,
        run: &mut Run<'_, '_, '_, S>,
        current: Sample<S>,
    ) -> Result<(S, Option<S>), SciOxideError> {
        let first = crate::derivative::scalar_derivative(
            self.f,
            current.x,
            current.value,
            self.strategy,
            run,
            self.probes,
        )?;
        let second = if HALLEY {
            Some(run.derivative(EvaluationRole::Hessian, || {
                self.f.second_derivative(current.x)
            })?)
        } else {
            None
        };
        Ok((first, second))
    }
}

/// Real Newton or Halley solve: open, or safeguarded by the sign bracket on
/// `interval`, which also bounds difference-quotient samples.
fn solve_real<
    S: RealScalar,
    F: ScalarFunction<S> + ?Sized,
    const HALLEY: bool,
    const GUARDED: bool,
>(
    f: &F,
    (guess, interval): (S, Option<Interval<S>>),
    options: &ScalarRootOptions<S>,
    strategy: DerivativeStrategy,
    context: &mut SolveContext<'_, '_>,
) -> Result<RootResult<S>, SciOxideError> {
    // Keep the method's fixed geometry inside the kernel, rather than
    // capturing an Option whose known absence is hidden by a call boundary.
    let interval = if GUARDED { interval } else { None };
    let derivatives = RealDerivatives::<F, S, HALLEY> {
        f,
        strategy,
        probes: interval.map_or_else(ExtendedInterval::entire, ExtendedInterval::from),
    };
    let callback = |x: S, context: &mut SolveContext<'_, '_>| f.eval_scoped(x, context);
    let Some(interval) = interval else {
        return open::solve(context, options, [guess], callback, NewtonStep(derivatives));
    };
    let mut run = Run::new(context, options)?;
    let outcome = (|| {
        let bracket =
            BracketState::evaluate(f, interval, &mut run, scioxide_core::argument!("init"))?;
        let scale = options
            .residual_scale
            .unwrap_or(bracket.lower.value.abs().max(bracket.upper.value.abs()));
        run.scale = scale;
        run.accept(bracket.best(), Some(bracket))?;
        if let Some(reason) = bracket.termination(options, scale)? {
            return run.result(bracket.best(), reason, Some(bracket));
        }
        open::iterate(
            &mut run,
            [guess],
            options,
            &callback,
            Safeguarded {
                bracket,
                scale,
                half_step: None,
                prior_half_step: None,
                derivatives,
            },
        )
    })();
    run.finish(outcome)
}

/// Newton or Halley proposals safeguarded by a sign bracket, with the last
/// two step lengths; bisection resets them.
struct Safeguarded<S: RealScalar, D> {
    bracket: BracketState<S>,
    scale: S,
    half_step: Option<S>,
    prior_half_step: Option<S>,
    derivatives: D,
}

impl<S: RealScalar, D: Derivatives<S>> OpenMethod<S, 1> for Safeguarded<S, D> {
    #[inline(always)]
    fn propose(
        &mut self,
        run: &mut Run<'_, '_, '_, S>,
        [current]: &[Sample<S>; 1],
    ) -> Result<Step<S>, SciOxideError> {
        let b = self.bracket;
        // The history starts from the bracket after the initial observation.
        let bisection = half_distance(b.upper.x, b.lower.x);
        let half_step = *self.half_step.get_or_insert(bisection);
        let limit = *self.prior_half_step.get_or_insert(bisection);
        let (first, second) = self.derivatives.at(run, *current)?;
        let proposal = correction::newton(current.x, current.value, first, second)
            .ok()
            .flatten()
            .map(|x| {
                if x == b.lower.x {
                    x.next_after(b.upper.x)
                } else if x == b.upper.x {
                    x.next_after(b.lower.x)
                } else {
                    x
                }
            })
            .filter(|&x| scioxide_core::backend::absolute_difference_le(x, current.x, limit));
        let (candidate, accepted) = b.safeguard(proposal)?;
        if accepted {
            self.prior_half_step = Some(half_step);
            self.half_step = Some(half_distance(candidate, current.x));
        } else {
            // A short move to the midpoint must not poison subsequent
            // derivative proposals that return toward the root.
            self.half_step = Some(bisection);
            self.prior_half_step = Some(bisection);
        }
        Ok(if candidate == current.x || !b.contains_strict(candidate) {
            Step::Stop(RootTermination::Stagnation)
        } else {
            Step::Candidate(candidate)
        })
    }

    fn assess(
        &mut self,
        sample: Sample<S>,
        _: S,
        _: bool,
        scale: S,
        options: &ScalarRootOptions<S>,
    ) -> Result<Option<(RootTermination, Sample<S>)>, SciOxideError> {
        self.bracket = self.bracket.update(sample)?;
        Ok(self
            .bracket
            .termination(options, scale)?
            .map(|reason| (reason, self.bracket.best())))
    }

    fn certificate(&self) -> Option<BracketState<S>> {
        Some(self.bracket)
    }

    fn fallback(&self, _: Sample<S>) -> Sample<S> {
        self.bracket.best()
    }

    fn known(&self, x: S) -> Option<Sample<S>> {
        [self.bracket.lower, self.bracket.upper]
            .into_iter()
            .find(|endpoint| endpoint.x == x)
    }

    fn scale(&self) -> Option<S> {
        Some(self.scale)
    }
}

/// Half a finite coordinate distance, including opposite maximum-scale points.
fn half_distance<S: RealScalar>(a: S, b: S) -> S {
    let difference = a - b;
    if difference.is_finite() {
        difference.abs() * S::HALF
    } else {
        (a * S::HALF - b * S::HALF).abs()
    }
}
