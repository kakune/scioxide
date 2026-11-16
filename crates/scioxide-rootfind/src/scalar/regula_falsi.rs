//! Regula falsi (false position) method.

use scioxide_core::{RealScalar, SciOxideError};

use super::ScalarRootOptions;
use crate::scalar::{BracketRule, BracketState, false_position};

/// Finds a zero of `f` in `[a, b]` by regula falsi (the method of false
/// position).
///
/// Each iteration evaluates `f` at the zero of the line through `(a, f(a))`
/// and `(b, f(b))`,
///
/// ```text
/// x = (a f(b) − b f(a)) / (f(b) − f(a)),
/// ```
///
/// and keeps the subinterval on which `f` changes sign. A point that is not
/// strictly inside the bracket is replaced by the midpoint. On a convex or
/// concave `f` one endpoint never moves and convergence is linear and often
/// slow; [`Illinois`](crate::Illinois) avoids this.
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
pub struct RegulaFalsi;

/// One false-position candidate per iteration, bisecting unusable ones.
struct FalsePosition;

impl<S: RealScalar> BracketRule<S> for FalsePosition {
    fn propose(
        &mut self,
        bracket: BracketState<S>,
        _: BracketState<S>,
        _: usize,
        step: usize,
        _: &ScalarRootOptions<S>,
    ) -> Result<Option<S>, SciOxideError> {
        if step > 0 {
            return Ok(None);
        }
        let interpolation = false_position(bracket.lower, bracket.upper)?;
        let midpoint = bracket.midpoint()?;
        Ok(Some(if bracket.contains_strict(interpolation) {
            interpolation
        } else {
            midpoint
        }))
    }
}

bracket_solver!(RegulaFalsi, |_| FalsePosition);
