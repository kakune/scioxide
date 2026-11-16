//! Run-time selection of a real scalar root solver.
use crate::{
    Bisection, Brent, Brenth, Chandrupatla, Halley, Illinois, Muller, Newton, RegulaFalsi, Ridder,
    RootResult, SafeguardedHalley, SafeguardedInitialGuess, SafeguardedNewton, ScalarFunction,
    ScalarRootOptions, ScalarRootSolver, Secant, Toms748,
};
use scioxide_core::{Interval, RealScalar, SciOxideError};
use scioxide_differentiate::DerivativeStrategy;

/// Selects a real scalar root solver at run time.
///
/// Each variant names one solver type, such as [`Brent`] or [`Newton`], and
/// solving with the variant gives the same result, name and identifier as
/// that solver. The starting data is a [`ScalarRootInput`] whose variant must
/// match the method. The method is never chosen by inspecting `f`.
///
/// # Errors
///
/// - `InvalidInput` if the [`ScalarRootInput`] variant does not match the
///   method, for example a `Guess` for `Brent`.
/// - The errors of the selected solver.
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_rootfind::{
///     ScalarFn, ScalarRootInput, ScalarRootMethod, ScalarRootOptions, ScalarRootSolver,
/// };
/// // x² − 2 = 0 on [0, 2] has the root √2.
/// let f = ScalarFn::new(|x: f64| Ok(x * x - 2.0));
/// let options = ScalarRootOptions::default();
/// for method in [ScalarRootMethod::Brent, ScalarRootMethod::Toms748] {
///     let result = method.solve(
///         &f,
///         ScalarRootInput::Bracket(Interval::new(0.0, 2.0)?),
///         &options,
///     )?;
///     assert!((result.point() - 2.0_f64.sqrt()).abs() < 1e-12);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub enum ScalarRootMethod {
    /// [`Bisection`], with a `Bracket` input.
    Bisection,
    /// [`RegulaFalsi`], with a `Bracket` input.
    RegulaFalsi,
    /// [`Illinois`], with a `Bracket` input.
    Illinois,
    /// [`Ridder`], with a `Bracket` input.
    Ridder,
    /// [`Brent`], with a `Bracket` input.
    Brent,
    /// [`Brenth`], with a `Bracket` input.
    Brenth,
    /// [`Chandrupatla`], with a `Bracket` input.
    Chandrupatla,
    /// [`Toms748`], with a `Bracket` input.
    Toms748,
    /// [`Newton`] with the given source of `f′`, with a `Guess` input.
    Newton(DerivativeStrategy),
    /// [`Halley`], with a `Guess` input.
    Halley,
    /// [`SafeguardedNewton`] with the given source of `f′`, with a
    /// `Safeguarded` input.
    SafeguardedNewton(DerivativeStrategy),
    /// [`SafeguardedHalley`], with a `Safeguarded` input.
    SafeguardedHalley,
    /// [`Secant`], with a `Pair` input.
    Secant,
    /// [`Muller`], with a `Triple` input.
    Muller,
}
/// Starting data for a [`ScalarRootMethod`]; the variant must match the method.
#[derive(Debug, Clone, Copy)]
pub enum ScalarRootInput<S: RealScalar> {
    /// An interval `[a, b]` for a bracketing method.
    Bracket(Interval<S>),
    /// One starting point for Newton or Halley.
    Guess(S),
    /// Two starting points for the secant method.
    Pair(S, S),
    /// Three starting points for Muller's method.
    Triple(S, S, S),
    /// An interval and a starting point for the safeguarded methods.
    Safeguarded(SafeguardedInitialGuess<S>),
}
impl<S: RealScalar> ScalarRootSolver<S> for ScalarRootMethod {
    type Initialization = ScalarRootInput<S>;
    fn name(&self) -> &'static str {
        match self {
            Self::Bisection => "Bisection",
            Self::RegulaFalsi => "RegulaFalsi",
            Self::Illinois => "Illinois",
            Self::Ridder => "Ridder",
            Self::Brent => "Brent",
            Self::Brenth => "Brenth",
            Self::Chandrupatla => "Chandrupatla",
            Self::Toms748 => "Toms748",
            Self::Newton(_) => "Newton",
            Self::Halley => "Halley",
            Self::SafeguardedNewton(_) => "SafeguardedNewton",
            Self::SafeguardedHalley => "SafeguardedHalley",
            Self::Secant => "Secant",
            Self::Muller => "Muller",
        }
    }
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        match self {
            Self::Bisection => <Bisection as ScalarRootSolver<S>>::algorithm_id(&Bisection),
            Self::RegulaFalsi => <RegulaFalsi as ScalarRootSolver<S>>::algorithm_id(&RegulaFalsi),
            Self::Illinois => <Illinois as ScalarRootSolver<S>>::algorithm_id(&Illinois),
            Self::Ridder => <Ridder as ScalarRootSolver<S>>::algorithm_id(&Ridder),
            Self::Brent => <Brent as ScalarRootSolver<S>>::algorithm_id(&Brent),
            Self::Brenth => <Brenth as ScalarRootSolver<S>>::algorithm_id(&Brenth),
            Self::Chandrupatla => {
                <Chandrupatla as ScalarRootSolver<S>>::algorithm_id(&Chandrupatla)
            }
            Self::Toms748 => <Toms748 as ScalarRootSolver<S>>::algorithm_id(&Toms748),
            Self::Newton(strategy) => {
                <Newton as ScalarRootSolver<S>>::algorithm_id(&Newton::new(*strategy))
            }
            Self::Halley => <Halley as ScalarRootSolver<S>>::algorithm_id(&Halley),
            Self::SafeguardedNewton(strategy) => {
                <SafeguardedNewton as ScalarRootSolver<S>>::algorithm_id(&SafeguardedNewton::new(
                    *strategy,
                ))
            }
            Self::SafeguardedHalley => {
                <SafeguardedHalley as ScalarRootSolver<S>>::algorithm_id(&SafeguardedHalley)
            }
            Self::Secant => <Secant as ScalarRootSolver<S>>::algorithm_id(&Secant),
            Self::Muller => <Muller as ScalarRootSolver<S>>::algorithm_id(&Muller),
        }
    }

    fn solve_scoped<'scope, 'capability, F: ScalarFunction<S> + ?Sized>(
        &self,
        f: &F,
        init: Self::Initialization,
        options: &ScalarRootOptions<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<RootResult<S>, SciOxideError> {
        match (*self, init) {
            (Self::Bisection, ScalarRootInput::Bracket(x)) => {
                Bisection.solve_scoped(f, x, options, scope)
            }
            (Self::RegulaFalsi, ScalarRootInput::Bracket(x)) => {
                RegulaFalsi.solve_scoped(f, x, options, scope)
            }
            (Self::Illinois, ScalarRootInput::Bracket(x)) => {
                Illinois.solve_scoped(f, x, options, scope)
            }
            (Self::Ridder, ScalarRootInput::Bracket(x)) => {
                Ridder.solve_scoped(f, x, options, scope)
            }
            (Self::Brent, ScalarRootInput::Bracket(x)) => Brent.solve_scoped(f, x, options, scope),
            (Self::Brenth, ScalarRootInput::Bracket(x)) => {
                Brenth.solve_scoped(f, x, options, scope)
            }
            (Self::Chandrupatla, ScalarRootInput::Bracket(x)) => {
                Chandrupatla.solve_scoped(f, x, options, scope)
            }
            (Self::Toms748, ScalarRootInput::Bracket(x)) => {
                Toms748.solve_scoped(f, x, options, scope)
            }
            (Self::Newton(s), ScalarRootInput::Guess(x)) => {
                Newton::new(s).solve_scoped(f, x, options, scope)
            }
            (Self::Halley, ScalarRootInput::Guess(x)) => Halley.solve_scoped(f, x, options, scope),
            (Self::SafeguardedNewton(s), ScalarRootInput::Safeguarded(x)) => {
                SafeguardedNewton::new(s).solve_scoped(f, x, options, scope)
            }
            (Self::SafeguardedHalley, ScalarRootInput::Safeguarded(x)) => {
                SafeguardedHalley.solve_scoped(f, x, options, scope)
            }
            (Self::Secant, ScalarRootInput::Pair(a, b)) => {
                Secant.solve_scoped(f, (a, b), options, scope)
            }
            (Self::Muller, ScalarRootInput::Triple(a, b, c)) => {
                Muller.solve_scoped(f, (a, b, c), options, scope)
            }
            _ => Err(SciOxideError::invalid_input(
                scioxide_core::argument!("init"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "input geometry does not match the selected method",
            )),
        }
    }
}
