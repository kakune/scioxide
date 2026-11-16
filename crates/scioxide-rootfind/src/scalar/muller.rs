//! Muller's method for real and complex functions.

use core::ops::Neg;
use scioxide_core::ScaledValue;
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideError, SolveContext};

use super::Sample;
use super::open::{self, Step};
use crate::{
    ComplexRootResult, ComplexRootSolver, ComplexScalarFunction, RootResult, RootTermination,
    ScalarFunction, ScalarRootOptions, ScalarRootSolver,
};

/// Finds a zero of a real function by Muller's method, starting from three
/// points `(x₀, x₁, x₂)`.
///
/// Each iteration fits the quadratic through the last three points and moves
/// to its zero nearest the newest point. With `a`, `b`, `c` the coefficients
/// of the quadratic in `x − xₖ` (so `c = f(xₖ)`), the step is
///
/// ```text
/// xₖ₊₁ = xₖ − 2c / (b ± √(b² − 4ac)),
/// ```
///
/// with the sign that gives the denominator the larger magnitude (`+` on a
/// tie). The step is computed with an extended exponent range, so
/// intermediate overflow does not stop the iteration. Each iteration costs one
/// evaluation; equal starting points are evaluated once. Convergence to a
/// simple zero has order about 1.84.
///
/// When `b² − 4ac < 0` the next point is complex, and the solve ends with
/// [`RootTermination::ComplexStepRequired`] at the newest point. Use
/// [`ComplexMuller`] with a [`ComplexScalarFunction`] to continue into the
/// complex plane. The solve stops as described in [`ScalarRootOptions`]. It
/// ends with `Stagnation` when two points coincide or the step is undefined.
/// After `Stagnation` or `IterationLimit`, the result is the evaluated point
/// with the smallest `|f|`.
///
/// # Errors
///
/// The errors listed on [`ScalarRootSolver::solve`].
///
/// # References
///
/// D. E. Muller, "A method for solving algebraic equations using an automatic
/// computer", *Math. Tables Aids Comput.* 10(56), 208–215 (1956).
#[derive(Debug, Clone, Copy, Default)]
pub struct Muller;

/// Finds a zero of a complex function by Muller's method, starting from three
/// points.
///
/// The iteration is that of [`Muller`] in complex arithmetic, with the
/// principal square root. It can reach a nonreal zero from real starting
/// points, which you pass as complex numbers with zero imaginary part. The
/// result is not projected onto the real axis. Residual tolerances compare
/// `|f(z)|` and step tolerances compare `|Δz|`. Near a multiple zero a small
/// residual does not imply a small error in `z`.
///
/// # Errors
///
/// The errors listed on [`ComplexRootSolver::solve`].
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex, ResidualTolerance, Tolerance};
/// use scioxide_rootfind::{
///     ComplexMuller, ComplexRootSolver, ComplexScalarFn, ComplexScalarFunction, ScalarRootOptions,
/// };
///
/// // z² + 1 has no real zero; from real starting points Muller reaches ±i.
/// let f = ComplexScalarFn::new(|z: Complex<f64>| Ok(z * z + Complex::new(1.0, 0.0)));
/// let tolerance = ResidualTolerance::from_tolerance(Tolerance::new(1e-12, 0.0)?);
/// let options = ScalarRootOptions::default().with_residual_tolerance(tolerance);
/// let real = |x: f64| Complex::new(x, 0.0);
/// let start = (real(-1.0), real(0.0), real(1.0));
/// let result = ComplexMuller.solve(&f, start, &options)?;
/// assert!(result.is_success());
/// assert!(f.eval(result.point())?.norm() <= 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// D. E. Muller, "A method for solving algebraic equations using an automatic
/// computer", *Math. Tables Aids Comput.* 10(56), 208–215 (1956).
#[derive(Debug, Clone, Copy, Default)]
pub struct ComplexMuller;

impl<S: RealScalar> ScalarRootSolver<S> for Muller {
    type Initialization = (S, S, S);

    fn name(&self) -> &'static str {
        "Muller"
    }
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("rootfind", "Muller", 1)
    }

    fn solve_scoped<'scope, 'capability, F: ScalarFunction<S> + ?Sized>(
        &self,
        f: &F,
        init: (S, S, S),
        options: &ScalarRootOptions<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<RootResult<S>, SciOxideError> {
        crate::context::run(
            scope.into(),
            <Self as ScalarRootSolver<S>>::algorithm_id(self),
            options.budgets(),
            |context| {
                solve(
                    |x, context| f.eval_scoped(x, context),
                    [init.0, init.1, init.2],
                    options,
                    context,
                )
            },
        )
    }
}

impl<S: RealScalar> ComplexRootSolver<S> for ComplexMuller {
    type Initialization = (Complex<S>, Complex<S>, Complex<S>);

    fn name(&self) -> &'static str {
        "ComplexMuller"
    }
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("rootfind", "ComplexMuller", 1)
    }

    fn solve_scoped<'scope, 'capability, F: ComplexScalarFunction<S> + ?Sized>(
        &self,
        f: &F,
        init: (Complex<S>, Complex<S>, Complex<S>),
        options: &ScalarRootOptions<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<ComplexRootResult<S>, SciOxideError> {
        crate::context::run(
            scope.into(),
            <Self as ComplexRootSolver<S>>::algorithm_id(self),
            options.budgets(),
            |context| {
                solve(
                    |x, context| f.eval_scoped(x, context),
                    [init.0, init.1, init.2],
                    options,
                    context,
                )
            },
        )
    }
}

/// Muller iteration; a real solve stops instead of taking a complex step.
fn solve<T: NumericScalar, R: super::Root<T>>(
    callback: impl Fn(T, &mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
    init: [T; 3],
    options: &ScalarRootOptions<T::Real>,
    context: &mut SolveContext<'_, '_>,
) -> Result<R, SciOxideError> {
    open::solve(
        context,
        options,
        init,
        callback,
        open::proposal(|_, points: &[Sample<T>; 3]| step(*points)),
    )
}

/// One Muller step, in extended exponent range until the final rounding. Real
/// points stop with [`RootTermination::ComplexStepRequired`] instead of
/// taking a complex step.
#[inline]
fn step<T: NumericScalar>(points: [Sample<T>; 3]) -> Result<Step<T>, SciOxideError> {
    let real_only = T::IS_REAL;
    let [a, b, c] = points.map(|point| ScaledValue::from_value(point.x));
    let x = [a?, b?, c?];
    let [a, b, c] = points.map(|point| ScaledValue::from_value(point.value));
    let y = [a?, b?, c?];
    let h1 = x[1].sub(x[0])?;
    let h2 = x[2].sub(x[1])?;
    let span = x[2].sub(x[0])?;
    if h1.is_zero() || h2.is_zero() || span.is_zero() {
        return Ok(Step::Stop(RootTermination::Stagnation));
    }
    let d1 = y[1].sub(y[0])?.div(h1)?;
    let d2 = y[2].sub(y[1])?.div(h2)?;
    let a = d2.sub(d1)?.div(span)?;
    let b = a.mul(h2)?.add(d2)?;
    let correction = if a.is_zero() {
        if b.is_zero() {
            return Ok(Step::Stop(RootTermination::Stagnation));
        }
        y[2].neg().div(b)?
    } else {
        let twice_a = ScaledValue::from_components(a.real().shifted(1)?, a.imaginary().shifted(1)?);
        let twice_c =
            ScaledValue::from_components(y[2].real().shifted(1)?, y[2].imaginary().shifted(1)?);
        let discriminant = b.difference_of_products(b, twice_a, twice_c)?;
        if real_only && discriminant.real().is_sign_negative() && !discriminant.real().is_zero() {
            return Ok(Step::Stop(RootTermination::ComplexStepRequired));
        }
        let square_root = discriminant.sqrt()?;
        let plus = b.add(square_root)?;
        let minus = b.sub(square_root)?;
        let denominator = if minus.magnitude()?.abs_le(plus.magnitude()?) {
            plus
        } else {
            minus
        };
        if denominator.is_zero() {
            return Ok(Step::Stop(RootTermination::Stagnation));
        }
        let numerator = ScaledValue::from_components(
            y[2].real().shifted(1)?.neg(),
            y[2].imaginary().shifted(1)?.neg(),
        );
        numerator.div(denominator)?
    };
    // Test the unrounded imaginary part: underflow must not silently cross domains.
    if real_only && !correction.imaginary().is_zero() {
        return Ok(Step::Stop(RootTermination::ComplexStepRequired));
    }
    x[2].add(correction)?.round().map(Step::Candidate)
}
