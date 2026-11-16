//! Bisection method.

use scioxide_core::{RealScalar, SciOxideError};

use super::ScalarRootOptions;
use crate::scalar::{BracketRule, BracketState};

/// Finds a zero of `f` in `[a, b]` by bisection.
///
/// Each iteration evaluates `f` at the midpoint and keeps the half on which
/// `f` changes sign, so the width halves with every evaluation. About
/// `log₂((b − a)/τ)` evaluations reach a bracket of width `τ`. Bisection
/// converges linearly but never more slowly; use [`Brent`](crate::Brent) or
/// [`Toms748`](crate::Toms748) for faster convergence on smooth functions.
///
/// The solve starts from an [`Interval`](scioxide_core::Interval) `[a, b]` whose
/// endpoint values have opposite signs or include a zero, keeps a bracket
/// around the zero, and stops as described in [`ScalarRootOptions`](crate::ScalarRootOptions).
/// The result is the bracket endpoint with the smaller `|f|`, and
/// [`RootResult::bracket`](crate::RootResult::bracket) returns the final bracket.
///
/// # Errors
///
/// - `InvalidInput` if `f(a)` and `f(b)` have the same nonzero sign.
/// - The errors listed on [`ScalarRootSolver::solve`](crate::ScalarRootSolver::solve).
#[derive(Debug, Clone, Copy, Default)]
pub struct Bisection;

/// One midpoint per iteration.
struct Halving;

impl<S: RealScalar> BracketRule<S> for Halving {
    #[inline(always)]
    fn propose(
        &mut self,
        bracket: BracketState<S>,
        _: BracketState<S>,
        _: usize,
        step: usize,
        _: &ScalarRootOptions<S>,
    ) -> Result<Option<S>, SciOxideError> {
        (step == 0).then(|| bracket.midpoint()).transpose()
    }
}

bracket_solver!(Bisection, |_| Halving);
