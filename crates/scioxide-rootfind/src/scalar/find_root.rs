//! One-shot real root of a scalar function on a bracket.

use scioxide_core::{ExecutionContext, Interval, RealScalar, SciOxideError};

use crate::{Brent, ScalarFn, ScalarRootOptions, ScalarRootSolver};

/// Finds a zero of `f` in `[lower, upper]` by Brent's method and returns it.
///
/// `f(lower)` and `f(upper)` must have opposite signs or include a zero. The
/// iteration is that of [`Brent`] with the default [`ScalarRootOptions`]: it
/// keeps a sign change of `f`, evaluates `f` once per iteration and stops when
/// the bracket is at most `4ε + 4ε·max(|a|, |b|)` wide or `f` is exactly zero,
/// where `ε` is the machine epsilon of `S`, within 100 iterations. The returned
/// point is the bracket endpoint with the smaller `|f|`; it equals
/// `Brent.solve(..).point()` bit for bit. Convergence to a simple zero is
/// superlinear. The call does not allocate.
///
/// Use [`Brent`] or another [`ScalarRootSolver`] with [`ScalarRootOptions`]
/// for other tolerances, budgets or methods, or to read the final bracket and
/// work counts.
///
/// # Errors
///
/// - `NonFinite` (origin `Input`) if `lower` or `upper` is NaN or infinite.
/// - `InvalidInput` (`lower`, `Inconsistent`) if `lower > upper`, or if
///   `f(lower)` and `f(upper)` have the same nonzero sign.
/// - `NonFinite` (origin `Callback`) if `f` returns NaN or infinity; errors
///   returned by `f` are returned unchanged.
/// - `NumericalInstability` if an iterate leaves the floating-point range.
/// - `ConvergenceFailure` (`IterationLimit` or `Stagnation`) if the bracket
///   does not reach the tolerance (see the [crate errors](crate#errors)).
///
/// # Examples
///
/// ```
/// // Wallis' cubic x³ − 2x − 5 = 0 has one real root in [2, 3].
/// let x = scioxide_rootfind::find_root(|x: f64| Ok(x * x * x - 2.0 * x - 5.0), 2.0, 3.0)?;
/// assert!((x - 2.094_551_481_542_326_5).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. P. Brent, *Algorithms for Minimization without Derivatives*,
/// Prentice-Hall, 1973, chapter 4.
pub fn find_root<S, F>(f: F, lower: S, upper: S) -> Result<S, SciOxideError>
where
    S: RealScalar,
    F: Fn(S) -> Result<S, SciOxideError>,
{
    let interval = Interval::new(lower, upper)?;
    let options = ScalarRootOptions::default();
    let result = crate::context::run(
        ExecutionContext::sequential_native().into(),
        ScalarRootSolver::<S>::algorithm_id(&Brent),
        options.budgets(),
        |context| {
            super::solve_bracket(
                &ScalarFn::new(f),
                interval,
                &options,
                scioxide_core::argument!("lower"),
                super::advanced_bracket::brent,
                context,
            )
        },
    )?;
    crate::termination::require_success(result.termination(), result.accounting())?;
    Ok(result.point())
}
