//! Ridders' method.

use scioxide_core::{RealScalar, SciOxideError};

use super::ScalarRootOptions;
use crate::scalar::{BracketRule, BracketState, Sample, residual_scale};

/// Finds a zero of `f` in `[a, b]` by Ridders' method.
///
/// Each iteration evaluates `f` at the midpoint `m` and then at
///
/// ```text
/// x = m + (m − a) sign(f(a) − f(b)) f(m) / √(f(m)² − f(a) f(b)),
/// ```
///
/// the zero of the line through the values of `f(t) e^(αt)` at `a`, `m` and
/// `b`, with `α` chosen to make them collinear. Both points update the
/// bracket. When `x` is not strictly inside the bracket, the iteration keeps
/// the midpoint step only. Convergence to a simple zero is quadratic per
/// iteration, or of order `√2` per evaluation.
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
///
/// # References
///
/// C. J. F. Ridders, "A new algorithm for computing a single root of a real
/// continuous function", *IEEE Trans. Circuits Syst.* 26(11), 979–980 (1979).
#[derive(Debug, Clone, Copy, Default)]
pub struct Ridder;

/// Each iteration evaluates the midpoint, then the exponential candidate built
/// from the midpoint and the iteration's starting bracket.
struct Exponential<S: RealScalar> {
    candidate: Option<S>,
    awaiting_midpoint: bool,
}

impl<S: RealScalar> BracketRule<S> for Exponential<S> {
    fn propose(
        &mut self,
        bracket: BracketState<S>,
        _: BracketState<S>,
        _: usize,
        step: usize,
        _: &ScalarRootOptions<S>,
    ) -> Result<Option<S>, SciOxideError> {
        match step {
            0 => {
                self.awaiting_midpoint = true;
                bracket.midpoint().map(Some)
            }
            // The midpoint already supplies a certified contraction, even if
            // interpolation underflows or rounds onto an endpoint.
            1 => Ok(self
                .candidate
                .take()
                .filter(|&x| bracket.contains_strict(x))),
            _ => Ok(None),
        }
    }

    fn observe(
        &mut self,
        old: BracketState<S>,
        _: BracketState<S>,
        midpoint: Sample<S>,
    ) -> Result<(), SciOxideError> {
        if !core::mem::take(&mut self.awaiting_midpoint) {
            return Ok(());
        }
        let value_scale = residual_scale(&[old.lower.value, old.upper.value, midpoint.value]);
        let lower_value = old.lower.value / value_scale;
        let upper_value = old.upper.value / value_scale;
        let midpoint_value = midpoint.value / value_scale;
        let radicand = midpoint_value * midpoint_value - lower_value * upper_value;
        let direction = if old.lower.value < old.upper.value {
            -S::ONE
        } else {
            S::ONE
        };
        self.candidate = if radicand > S::ZERO && radicand.is_finite() {
            let ratio = direction * midpoint_value / radicand.sqrt();
            let candidate = midpoint.x + ratio * (midpoint.x - old.lower.x);
            candidate.is_finite().then_some(candidate)
        } else {
            None
        };
        Ok(())
    }
}

bracket_solver!(Ridder, |_| Exponential {
    candidate: None,
    awaiting_midpoint: false,
});
