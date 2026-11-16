//! Secant method.

use scioxide_core::{RealScalar, SciOxideError, SolveContext};

use super::open::{self, Step};
use super::{Sample, ScalarRootOptions, correction};
use crate::result::RootResult;
use crate::traits::{ScalarFunction, ScalarRootSolver};

/// Finds a zero of `f` by the secant method, starting from two points
/// `(x₀, x₁)`.
///
/// Each iteration evaluates `f` once at
///
/// ```text
/// xₖ₊₁ = xₖ − f(xₖ) (xₖ − xₖ₋₁) / (f(xₖ) − f(xₖ₋₁)).
/// ```
///
/// Convergence to a simple zero is superlinear, of order `(1 + √5)/2 ≈ 1.618`,
/// when the starting points are close enough. The iteration is not
/// safeguarded: from poor starting points it can diverge. Use a bracketing
/// method such as [`Brent`](crate::Brent) when a sign change is known.
/// Intermediate range loss retries the intersection in extended exponent range,
/// retaining small function values until after their coordinate products.
///
/// The solve stops as described in [`ScalarRootOptions`](crate::ScalarRootOptions).
/// It ends with `Stagnation` when `f(xₖ) = f(xₖ₋₁)` or a step repeats a point.
/// After `Stagnation` or `IterationLimit`, the result is the evaluated point
/// with the smallest `|f|`.
///
/// # Errors
///
/// The errors listed on [`ScalarRootSolver::solve`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Secant;

impl<S: RealScalar> ScalarRootSolver<S> for Secant {
    type Initialization = (S, S);

    fn name(&self) -> &'static str {
        "Secant"
    }
    fn algorithm_id(&self) -> scioxide_core::AlgorithmId {
        scioxide_core::algorithm_id!("rootfind", "Secant", 1)
    }

    fn solve_scoped<'scope, 'capability, F: ScalarFunction<S> + ?Sized>(
        &self,
        f: &F,
        init: (S, S),
        options: &ScalarRootOptions<S>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<RootResult<S>, SciOxideError> {
        crate::context::run(
            scope.into(),
            <Self as ScalarRootSolver<S>>::algorithm_id(self),
            options.budgets(),
            |context| solve(f, init, options, context),
        )
    }
}

/// Secant iteration from two starting points.
fn solve<S: RealScalar, F: ScalarFunction<S> + ?Sized>(
    f: &F,
    init: (S, S),
    options: &ScalarRootOptions<S>,
    context: &mut SolveContext<'_, '_>,
) -> Result<RootResult<S>, SciOxideError> {
    open::solve(
        context,
        options,
        [init.0, init.1],
        |x, context| f.eval_scoped(x, context),
        open::proposal(|_, [previous, current]: &[Sample<S>; 2]| {
            correction::secant(*previous, *current).map(Step::proposed)
        }),
    )
}
