//! Function and solver traits, and the closure adapters for scalar functions.

use scioxide_core::{NumericScalar, RealScalar, ResidualOperator, SciOxideError};

use crate::ScalarRootOptions;
use crate::residual::DenseJacobian;
use crate::result::{ComplexRootResult, RootResult, SystemRootResult};

// ---------------------------------------------------------------------------
// Function traits
// ---------------------------------------------------------------------------

/// Defines a real function `f: ℝ → ℝ` whose zeros a scalar solver finds.
///
/// Implement [`eval`](ScalarFunction::eval). The derivative methods are optional
/// and return `Ok(None)` by default. [`Newton`](crate::Newton) and
/// [`SafeguardedNewton`](crate::SafeguardedNewton) ask for `f′` in the order of
/// their [`DerivativeStrategy`](scioxide_differentiate::DerivativeStrategy);
/// [`Halley`](crate::Halley) and [`SafeguardedHalley`](crate::SafeguardedHalley)
/// require [`derivative`](ScalarFunction::derivative) and
/// [`second_derivative`](ScalarFunction::second_derivative). The trait is object
/// safe, so solvers accept `&dyn ScalarFunction<S>`. [`ScalarFn`] wraps closures.
///
/// Solvers check every value and derivative they receive: NaN or infinity is a
/// `NonFinite` error with origin `Callback`, and an error you return stops the
/// solve and is returned unchanged.
pub trait ScalarFunction<S: RealScalar> {
    /// Evaluates `f(x)`.
    ///
    /// # Errors
    ///
    /// Implementations return any error that describes the failure, for example
    /// `DomainError` when `x` lies outside the domain of `f`.
    fn eval(&self, x: S) -> Result<S, SciOxideError>;

    /// Evaluates `f(x)` inside a solve, with access to the solve's `context`.
    ///
    /// Override this when evaluating `f` runs another iterative solver: pass
    /// `context` to that solver's scoped form so its work counts toward this
    /// solve's limits (see [Solve accounting](scioxide_core#solve-accounting)).
    /// The solver already counts the call itself. The default calls
    /// [`Self::eval`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::eval`] and of any nested solve.
    #[inline(always)]
    fn eval_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        let _ = context;
        self.eval(x)
    }

    /// Evaluates the analytic first derivative `f′(x)`, or returns `Ok(None)`
    /// when none is provided (the default).
    ///
    /// # Errors
    ///
    /// Implementations return any error that describes the failure.
    fn derivative(&self, x: S) -> Result<Option<S>, SciOxideError> {
        let _ = x;
        Ok(None)
    }

    /// Evaluates the analytic second derivative `f″(x)`, or returns `Ok(None)`
    /// when none is provided (the default).
    ///
    /// # Errors
    ///
    /// Implementations return any error that describes the failure.
    fn second_derivative(&self, _x: S) -> Result<Option<S>, SciOxideError> {
        Ok(None)
    }

    /// Evaluates `f′(x)` by automatic differentiation, or returns `Ok(None)`
    /// when this is not supported (the default).
    ///
    /// Only the strategies that name automatic differentiation call it.
    ///
    /// # Errors
    ///
    /// Implementations return any error that describes the failure.
    fn automatic_derivative(&self, x: S) -> Result<Option<S>, SciOxideError> {
        let _ = x;
        Ok(None)
    }
}

/// Wraps a closure `Fn(S) -> Result<S, SciOxideError>` as a [`ScalarFunction`].
///
/// The adapter returns a `NonFinite` error with origin `Callback` when the
/// closure returns NaN or infinity. Attach derivatives with
/// [`with_derivative`](ScalarFn::with_derivative).
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_rootfind::{Brent, ScalarFn, ScalarRootOptions, ScalarRootSolver};
///
/// // x² − 2 = 0 on [0, 2] has the root √2.
/// let f = ScalarFn::new(|x: f64| Ok(x * x - 2.0));
/// let result = Brent.solve(&f, Interval::new(0.0, 2.0)?, &ScalarRootOptions::default())?;
/// assert!((result.point() - 2.0_f64.sqrt()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ScalarFn<F> {
    evaluate: F,
}

impl<F> ScalarFn<F> {
    /// Wraps the closure that evaluates `f(x)`.
    #[must_use]
    pub const fn new(evaluate: F) -> Self {
        Self { evaluate }
    }

    /// Adds a closure that evaluates the analytic derivative `f′(x)`.
    #[must_use]
    pub fn with_derivative<D>(self, derivative: D) -> ScalarFnWithDerivative<F, D> {
        ScalarFnWithDerivative {
            evaluate: self.evaluate,
            derivative,
        }
    }
}

impl<S: RealScalar, F: Fn(S) -> Result<S, SciOxideError>> ScalarFunction<S> for ScalarFn<F> {
    fn eval(&self, x: S) -> Result<S, SciOxideError> {
        finite_output((self.evaluate)(x)?)
    }
}

/// Wraps closures for `f` and `f′` as a [`ScalarFunction`].
///
/// Created by [`ScalarFn::with_derivative`]. Both outputs are checked for NaN
/// and infinity as in [`ScalarFn`].
#[derive(Debug, Clone, Copy)]
pub struct ScalarFnWithDerivative<F, D> {
    evaluate: F,
    derivative: D,
}

impl<S, F, D> ScalarFunction<S> for ScalarFnWithDerivative<F, D>
where
    S: RealScalar,
    F: Fn(S) -> Result<S, SciOxideError>,
    D: Fn(S) -> Result<S, SciOxideError>,
{
    fn eval(&self, x: S) -> Result<S, SciOxideError> {
        finite_output((self.evaluate)(x)?)
    }

    fn derivative(&self, x: S) -> Result<Option<S>, SciOxideError> {
        finite_output((self.derivative)(x)?).map(Some)
    }
}

/// Wraps closures for `f`, `f′` and `f″` as a [`ScalarFunction`].
///
/// Created by [`ScalarFnWithDerivative::with_second_derivative`]. All outputs
/// are checked for NaN and infinity as in [`ScalarFn`].
#[derive(Debug, Clone, Copy)]
pub struct ScalarFnWithDerivatives<F, D, D2> {
    first: ScalarFnWithDerivative<F, D>,
    second: D2,
}
impl<F, D> ScalarFnWithDerivative<F, D> {
    /// Adds a closure that evaluates the analytic second derivative `f″(x)`.
    pub fn with_second_derivative<D2>(self, second: D2) -> ScalarFnWithDerivatives<F, D, D2> {
        ScalarFnWithDerivatives {
            first: self,
            second,
        }
    }
}
impl<S: RealScalar, F, D, D2> ScalarFunction<S> for ScalarFnWithDerivatives<F, D, D2>
where
    F: Fn(S) -> Result<S, SciOxideError>,
    D: Fn(S) -> Result<S, SciOxideError>,
    D2: Fn(S) -> Result<S, SciOxideError>,
{
    fn eval(&self, x: S) -> Result<S, SciOxideError> {
        self.first.eval(x)
    }
    fn derivative(&self, x: S) -> Result<Option<S>, SciOxideError> {
        self.first.derivative(x)
    }
    fn second_derivative(&self, x: S) -> Result<Option<S>, SciOxideError> {
        finite_output((self.second)(x)?).map(Some)
    }
}

/// Defines a complex function `f: ℂ → ℂ` whose zeros a complex solver finds.
///
/// You supply the complex extension of your function; a [`ScalarFunction`] is
/// never converted to one. Derivatives are holomorphic derivatives and are
/// optional: [`ComplexNewton`](crate::ComplexNewton) requires the first and
/// [`ComplexHalley`](crate::ComplexHalley) both. Solvers check every value and
/// derivative and reject a NaN or infinite component as a `NonFinite` error
/// with origin `Callback`. [`ComplexScalarFn`] wraps closures.
pub trait ComplexScalarFunction<S: RealScalar> {
    /// Evaluates `f(z)`.
    ///
    /// # Errors
    ///
    /// Implementations return any error that describes the failure, for example
    /// `DomainError` when `z` lies outside the domain of `f`.
    fn eval(
        &self,
        x: scioxide_core::Complex<S>,
    ) -> Result<scioxide_core::Complex<S>, SciOxideError>;
    /// Evaluates `f(z)` inside a solve, with access to the solve's `context`.
    ///
    /// Use it as described for [`ScalarFunction::eval_scoped`]. The default
    /// calls [`Self::eval`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::eval`] and of any nested solve.
    #[inline(always)]
    fn eval_scoped(
        &self,
        x: scioxide_core::Complex<S>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<scioxide_core::Complex<S>, SciOxideError> {
        let _ = context;
        self.eval(x)
    }
    /// Evaluates the holomorphic derivative `f′(z)`, or returns `Ok(None)` when
    /// none is provided (the default).
    ///
    /// # Errors
    ///
    /// Implementations return any error that describes the failure.
    fn derivative(
        &self,
        _x: scioxide_core::Complex<S>,
    ) -> Result<Option<scioxide_core::Complex<S>>, SciOxideError> {
        Ok(None)
    }
    /// Evaluates the second holomorphic derivative `f″(z)`, or returns
    /// `Ok(None)` when none is provided (the default).
    ///
    /// # Errors
    ///
    /// Implementations return any error that describes the failure.
    fn second_derivative(
        &self,
        _x: scioxide_core::Complex<S>,
    ) -> Result<Option<scioxide_core::Complex<S>>, SciOxideError> {
        Ok(None)
    }
}

/// Wraps a closure `Fn(Complex<S>) -> Result<Complex<S>, SciOxideError>` as a
/// [`ComplexScalarFunction`].
///
/// The adapter checks every output, including attached derivatives: a NaN or
/// infinite component is a `NonFinite` error with origin `Callback`.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_rootfind::{ComplexScalarFn, ComplexScalarFunction};
///
/// // f(z) = z², with a derivative closure that returns NaN.
/// let f = ComplexScalarFn::new(|z: Complex<f64>| Ok(z * z))
///     .with_derivative(|_| Ok(Complex::new(f64::NAN, 0.0)));
/// assert_eq!(f.eval(Complex::new(2.0, 0.0))?, Complex::new(4.0, 0.0));
/// assert_eq!(
///     f.derivative(Complex::new(2.0, 0.0)).unwrap_err().kind(),
///     scioxide_core::SciOxideErrorKind::NonFinite
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ComplexScalarFn<F> {
    evaluate: F,
}

impl<F> ComplexScalarFn<F> {
    /// Wraps the closure that evaluates `f(z)`.
    pub const fn new(evaluate: F) -> Self {
        Self { evaluate }
    }

    /// Adds a closure that evaluates the holomorphic derivative `f′(z)`.
    #[must_use]
    pub fn with_derivative<D>(self, derivative: D) -> ComplexScalarFnWithDerivative<F, D> {
        ComplexScalarFnWithDerivative {
            evaluate: self.evaluate,
            derivative,
        }
    }
}

impl<S: RealScalar, F> ComplexScalarFunction<S> for ComplexScalarFn<F>
where
    F: Fn(scioxide_core::Complex<S>) -> Result<scioxide_core::Complex<S>, SciOxideError>,
{
    fn eval(
        &self,
        x: scioxide_core::Complex<S>,
    ) -> Result<scioxide_core::Complex<S>, SciOxideError> {
        finite_output((self.evaluate)(x)?)
    }
}

/// Wraps closures for `f` and `f′` as a [`ComplexScalarFunction`].
///
/// Created by [`ComplexScalarFn::with_derivative`].
#[derive(Debug, Clone, Copy)]
pub struct ComplexScalarFnWithDerivative<F, D> {
    evaluate: F,
    derivative: D,
}
/// Wraps closures for `f`, `f′` and `f″` as a [`ComplexScalarFunction`].
///
/// Created by [`ComplexScalarFnWithDerivative::with_second_derivative`].
#[derive(Debug, Clone, Copy)]
pub struct ComplexScalarFnWithDerivatives<F, D, D2> {
    first: ComplexScalarFnWithDerivative<F, D>,
    second: D2,
}
impl<F, D> ComplexScalarFnWithDerivative<F, D> {
    /// Adds a closure that evaluates the second holomorphic derivative `f″(z)`.
    #[must_use]
    pub fn with_second_derivative<D2>(
        self,
        second: D2,
    ) -> ComplexScalarFnWithDerivatives<F, D, D2> {
        ComplexScalarFnWithDerivatives {
            first: self,
            second,
        }
    }
}
impl<S: RealScalar, F, D> ComplexScalarFunction<S> for ComplexScalarFnWithDerivative<F, D>
where
    F: Fn(scioxide_core::Complex<S>) -> Result<scioxide_core::Complex<S>, SciOxideError>,
    D: Fn(scioxide_core::Complex<S>) -> Result<scioxide_core::Complex<S>, SciOxideError>,
{
    fn eval(
        &self,
        x: scioxide_core::Complex<S>,
    ) -> Result<scioxide_core::Complex<S>, SciOxideError> {
        finite_output((self.evaluate)(x)?)
    }
    fn derivative(
        &self,
        x: scioxide_core::Complex<S>,
    ) -> Result<Option<scioxide_core::Complex<S>>, SciOxideError> {
        finite_output((self.derivative)(x)?).map(Some)
    }
}
impl<S: RealScalar, F, D, D2> ComplexScalarFunction<S> for ComplexScalarFnWithDerivatives<F, D, D2>
where
    F: Fn(scioxide_core::Complex<S>) -> Result<scioxide_core::Complex<S>, SciOxideError>,
    D: Fn(scioxide_core::Complex<S>) -> Result<scioxide_core::Complex<S>, SciOxideError>,
    D2: Fn(scioxide_core::Complex<S>) -> Result<scioxide_core::Complex<S>, SciOxideError>,
{
    fn eval(
        &self,
        x: scioxide_core::Complex<S>,
    ) -> Result<scioxide_core::Complex<S>, SciOxideError> {
        self.first.eval(x)
    }
    fn derivative(
        &self,
        x: scioxide_core::Complex<S>,
    ) -> Result<Option<scioxide_core::Complex<S>>, SciOxideError> {
        self.first.derivative(x)
    }
    fn second_derivative(
        &self,
        x: scioxide_core::Complex<S>,
    ) -> Result<Option<scioxide_core::Complex<S>>, SciOxideError> {
        finite_output((self.second)(x)?).map(Some)
    }
}

// ---------------------------------------------------------------------------
// Solver traits
// ---------------------------------------------------------------------------

/// Finds a zero of a real function `f: ℝ → ℝ`.
///
/// The starting data of a method is the associated type
/// [`Initialization`](Self::Initialization): an
/// [`Interval`](scioxide_core::Interval) for bracketing methods, one point for
/// Newton and Halley, two for the secant method, three for Muller, and a
/// [`SafeguardedInitialGuess`](crate::SafeguardedInitialGuess) for the
/// safeguarded methods. A mismatch is a compile error:
///
/// ```compile_fail
/// use scioxide_core::Interval;
/// use scioxide_rootfind::{ScalarFn, ScalarRootOptions, ScalarRootSolver, Secant};
///
/// let function = ScalarFn::new(|x: f64| Ok(x * x - 2.0));
/// let bracket = Interval::new(1.0, 2.0).unwrap();
/// let _ = Secant.solve(&function, bracket, &ScalarRootOptions::default());
/// ```
///
/// [`ScalarRootMethod`](crate::ScalarRootMethod) selects a method at run time.
/// Stopping tests and limits come from [`ScalarRootOptions`]; the result's
/// [`termination`](RootResult::termination) says whether the solve succeeded
/// (see [Termination](crate#termination)). Solves do not allocate.
pub trait ScalarRootSolver<S: RealScalar> {
    /// Starting data of this method.
    type Initialization;

    /// Returns the method name, for example `"Brent"`.
    fn name(&self) -> &'static str;

    /// Returns the identifier of this method, as recorded in its results.
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId;

    /// Finds a zero of `f` on the calling thread, without a clock or
    /// cancellation.
    ///
    /// When an iteration or work limit stops the solve after at least one
    /// evaluation, the result holds the best point found so far and an
    /// unsuccessful termination.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `options` are invalid, the bracket endpoints have
    ///   the same nonzero sign, or `f` lacks a derivative the method requires.
    /// - `NonFinite` if a starting point is NaN or infinite (origin `Input`),
    ///   or `f` or a derivative returns NaN or infinity (origin `Callback`).
    /// - `NumericalInstability` if the next iterate overflows.
    /// - Errors returned by `f` are returned unchanged.
    #[inline(always)]
    fn solve<F: ScalarFunction<S> + ?Sized>(
        &self,
        f: &F,
        init: Self::Initialization,
        options: &ScalarRootOptions<S>,
    ) -> Result<RootResult<S>, SciOxideError> {
        self.solve_scoped(
            f,
            init,
            options,
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }

    /// Finds a zero of `f` in an [execution context](scioxide_core#execution-contexts)
    /// or as a nested solve inside a parent [`SolveContext`](scioxide_core::SolveContext).
    ///
    /// A nested solve counts its evaluations toward the parent's limits (see
    /// [Solve accounting](scioxide_core#solve-accounting)). Cancellation, a
    /// deadline or a work limit after at least one evaluation returns the best
    /// point so far with an unsuccessful termination.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::solve`]. A limit, cancellation or deadline
    /// reached before the first evaluation is a `ConvergenceFailure` or
    /// `ExecutionStopped` error.
    fn solve_scoped<'scope, 'capability, F: ScalarFunction<S> + ?Sized>(
        &self,
        f: &F,
        init: Self::Initialization,
        options: &ScalarRootOptions<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<RootResult<S>, SciOxideError>;
}

/// Finds a zero of a complex function `f: ℂ → ℂ`.
///
/// The complex counterpart of [`ScalarRootSolver`], implemented by
/// [`ComplexNewton`](crate::ComplexNewton), [`ComplexHalley`](crate::ComplexHalley),
/// [`ComplexSecant`](crate::ComplexSecant) and [`ComplexMuller`](crate::ComplexMuller).
/// The function must be a [`ComplexScalarFunction`]. Residual tolerances
/// compare the modulus `|f(z)|`, and step tolerances compare `|Δz|`.
pub trait ComplexRootSolver<S: RealScalar> {
    /// Starting data of this method.
    type Initialization;

    /// Returns the method name, for example `"ComplexNewton"`.
    fn name(&self) -> &'static str;

    /// Returns the identifier of this method, as recorded in its results.
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId;

    /// Finds a zero of `f` on the calling thread, without a clock or
    /// cancellation.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `options` are invalid or `f` lacks a derivative the
    ///   method requires.
    /// - `NonFinite` if a starting point has a NaN or infinite component (origin
    ///   `Input`), or `f` or a derivative returns one (origin `Callback`).
    /// - `NumericalInstability` if the next iterate or `|f(z)|` overflows.
    /// - Errors returned by `f` are returned unchanged.
    #[inline(always)]
    fn solve<F: ComplexScalarFunction<S> + ?Sized>(
        &self,
        f: &F,
        init: Self::Initialization,
        options: &ScalarRootOptions<S>,
    ) -> Result<ComplexRootResult<S>, SciOxideError> {
        self.solve_scoped(
            f,
            init,
            options,
            scioxide_core::ExecutionContext::sequential_native(),
        )
    }

    /// Finds a zero of `f` in an execution context or as a nested solve, as
    /// [`ScalarRootSolver::solve_scoped`] does.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::solve`]. A limit, cancellation or deadline
    /// reached before the first evaluation is a `ConvergenceFailure` or
    /// `ExecutionStopped` error.
    fn solve_scoped<'scope, 'capability, F: ComplexScalarFunction<S> + ?Sized>(
        &self,
        f: &F,
        init: Self::Initialization,
        options: &ScalarRootOptions<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<ComplexRootResult<S>, SciOxideError>;
}

/// Solves a square nonlinear system `F(x) = 0` with `F: ℝⁿ → ℝⁿ`.
///
/// Implemented by [`NewtonSystem`](crate::NewtonSystem),
/// [`Broyden`](crate::Broyden) and [`TrustRegionSystem`](crate::TrustRegionSystem).
/// A solver object holds validated options and can solve many problems. Both
/// solve methods take the starting point `x₀` as a slice of `n` components,
/// where `n` is the dimension of `function`, and check it before the first
/// evaluation. [`solve`](Self::solve) returns an owned result;
/// [`solve_into`](Self::solve_into) reuses a workspace and an output you
/// provide (see [Reusing workspaces](crate#reusing-workspaces)).
pub trait SystemRootSolver<
    S: RealScalar + NumericScalar<Real = S>,
    F: DenseJacobian<S> + ?Sized = dyn DenseJacobian<S>,
>
{
    /// Scratch storage reused across solves.
    type Workspace;
    /// Returns the method name.
    fn name(&self) -> &'static str;
    /// Returns the identifier of this method, as recorded in its results.
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId;
    /// Returns the workspace size this solver needs for `function`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` or `DimensionMismatch` if `function` has dimension zero
    ///   or is not square.
    /// - `SizeOverflow` or `ResourceLimit` if the size overflows `usize` or
    ///   exceeds the configured memory limit.
    fn workspace_requirements(
        &self,
        function: &F,
    ) -> Result<crate::SystemWorkspaceRequirements<S>, SciOxideError>;
    /// Allocates a workspace for `function`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::workspace_requirements`], and
    /// `AllocationFailure` if memory cannot be reserved.
    fn workspace(&self, function: &F) -> Result<Self::Workspace, SciOxideError>;
    /// Solves `F(x) = 0` from `initial` on the calling thread and returns an
    /// owned result.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`function`, `OutOfRange`) if `function` has dimension
    ///   zero.
    /// - `DimensionMismatch` if `function` is not square or `initial` does not
    ///   have its dimension.
    /// - `NonFinite` (origin `Input`) if a component of `initial` is NaN or
    ///   infinite.
    /// - `AllocationFailure` if the workspace or result cannot be allocated.
    /// - The conditions listed on each solver, and errors returned by
    ///   `function`, which are returned unchanged.
    fn solve(&self, function: &F, initial: &[S]) -> Result<SystemRootResult<S>, SciOxideError>;
    /// Solves `F(x) = 0` from `initial` into `output`, using `workspace` and
    /// an [execution context](scioxide_core#execution-contexts) or parent
    /// [`SolveContext`](scioxide_core::SolveContext).
    ///
    /// Reserve `output` with [`SystemRootResult::try_reserve`] first. A limit,
    /// cancellation or deadline after the first accepted point writes that
    /// point with an unsuccessful termination.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::solve`], and an error if `output` or
    /// `workspace` was not prepared for this problem. On error `output` is
    /// unchanged (see [Output on error](scioxide_core#output-on-error)).
    fn solve_into<'scope, 'capability>(
        &self,
        function: &F,
        initial: &[S],
        output: &mut SystemRootResult<S>,
        workspace: &mut Self::Workspace,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError>;
}

/// Returns `value` if every component is finite, and otherwise a `NonFinite`
/// error with origin `Callback`.
///
/// Closure adapters and solvers apply it to every callback value and
/// derivative.
pub(crate) fn finite_output<T: NumericScalar>(value: T) -> Result<T, SciOxideError> {
    if value.has_finite_components() {
        Ok(value)
    } else {
        Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Callback,
            "callback produced a non-finite value",
        ))
    }
}

// ---------------------------------------------------------------------------
// Line-search trait
// ---------------------------------------------------------------------------

/// Chooses a step length along a search direction for a nonlinear system
/// solver.
///
/// Given the iterate `x`, a direction `d` and the slope `φ′(0)` of the merit
/// function
///
/// ```text
/// φ(α) = ½ ‖F(x + α d)‖₂²,
/// ```
///
/// an implementation finds `α > 0` with sufficient decrease in `φ`.
/// [`BacktrackingLineSearch`](crate::BacktrackingLineSearch) is the provided
/// implementation. For Newton's method the slope is `F(x)ᵀ J(x) d`; for
/// Broyden's method it is `F(x)ᵀ B d` with the secant approximation `B`, so a
/// decrease of the model need not decrease `φ`. The trait is object safe.
///
/// A trial residual can be NaN or infinite, or `F` can return a `DomainError`
/// with [`OutsideDomain`](scioxide_core::DomainReason::OutsideDomain) at the
/// trial point. An implementation treats such a trial as rejected and shortens
/// the step; it accepts only a finite point with a finite residual.
pub trait RootLineSearch<S: RealScalar + NumericScalar<Real = S>> {
    /// Searches along `direction` from `x` and returns the accepted step
    /// length `α`, or `None` when no trial is accepted.
    ///
    /// `current_residual` is `F(x)` and `merit_slope` is the strictly negative
    /// slope `φ′(0)`, stored with extended exponent range. On success,
    /// `trial_point` and `trial_residual` hold `x + α d` and `F(x + α d)`; after
    /// `None` their contents are unspecified. `scratch` holds at least
    /// `f.residual_workspace_len()` values.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if a slice length differs from the dimension of
    ///   `f`; the outputs are then unchanged.
    /// - Errors returned by `f`, other than the rejected trials above, are
    ///   returned unchanged.
    ///
    /// The solver treats an error as fatal and `None` as a failed search.
    #[expect(
        clippy::too_many_arguments,
        reason = "Numerical execution borrows separate caller-owned buffers and solve state."
    )]
    fn search_into(
        &self,
        f: &dyn ResidualOperator<S>,
        x: &[S],
        direction: &[S],
        current_residual: &[S],
        merit_slope: scioxide_core::Scaled<S>,
        trial_point: &mut [S],
        trial_residual: &mut [S],
        scratch: &mut [S],
    ) -> Result<Option<S>, SciOxideError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::SciOxideError;

    // A simple test function f(x) = x^2 - 2 with derivative 2x
    struct Sqrt2Fn;

    impl ScalarFunction<f64> for Sqrt2Fn {
        fn eval(&self, x: f64) -> Result<f64, SciOxideError> {
            Ok(x * x - 2.0)
        }

        fn derivative(&self, x: f64) -> Result<Option<f64>, SciOxideError> {
            Ok(Some(2.0 * x))
        }
    }

    // A function with no derivative
    struct NoDerivFn;

    impl ScalarFunction<f64> for NoDerivFn {
        fn eval(&self, x: f64) -> Result<f64, SciOxideError> {
            Ok(x * x - 1.0)
        }
    }

    #[test]
    fn test_scalar_function_eval() {
        let f = Sqrt2Fn;
        let val = f.eval(1.0);
        assert!(val.is_ok());
        assert!((val.unwrap_or(f64::NAN) - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_scalar_function_derivative() {
        let f = Sqrt2Fn;
        let d = f.derivative(3.0);
        assert!(d.is_ok());
        assert!((d.unwrap_or(None).unwrap_or(f64::NAN) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_scalar_function_default_derivative() {
        let f = NoDerivFn;
        let d = f.derivative(1.0);
        assert!(d.is_ok());
        assert!(d.unwrap_or(Some(0.0)).is_none());
    }

    #[test]
    fn test_scalar_function_dyn_dispatch() {
        let f: &dyn ScalarFunction<f64> = &Sqrt2Fn;
        let val = f.eval(2.0);
        assert!(val.is_ok());
        assert!((val.unwrap_or(f64::NAN) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn scalar_closure_adapter_exposes_optional_derivative() {
        let function =
            ScalarFn::new(|x: f64| Ok(x * x - 2.0)).with_derivative(|x: f64| Ok(2.0 * x));
        assert!((function.eval(2.0).unwrap() - 2.0).abs() < 1e-12);
        assert!((function.derivative(2.0).unwrap().unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn scalar_closure_adapter_rejects_non_finite_output() {
        let function = ScalarFn::new(|_x: f64| Ok(f64::NAN));
        assert_eq!(
            function.eval(0.0).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::NonFinite
        );
    }

    #[test]
    fn closure_adapters_propagate_callback_errors() {
        let function = ScalarFn::new(|_x: f64| {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("callback_sentinel"),
                scioxide_core::InvalidInputReason::Malformed,
                "intentional callback failure",
            ))
        });
        let error = function.eval(0.0).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput { argument, .. }
                if argument == scioxide_core::argument!("callback_sentinel")
        ));
    }
}
