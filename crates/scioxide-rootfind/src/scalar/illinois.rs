//! Illinois method (modified false position).

use scioxide_core::{RealScalar, SciOxideError};

use super::ScalarRootOptions;
use crate::scalar::{BracketRule, BracketState, Sample, false_position};

/// Finds a zero of `f` in `[a, b]` by the Illinois method, a modified regula
/// falsi.
///
/// Each iteration takes a false-position step as in
/// [`RegulaFalsi`](crate::RegulaFalsi). When the same endpoint is kept twice
/// in a row, its function value is halved in the next interpolation, so that
/// endpoint moves too. Convergence to a simple zero is superlinear, of order
/// about 1.44 (Dowell & Jarratt, 1971). Every fourth iteration checks that the
/// last four steps at least halved the bracket and otherwise bisects; a point
/// outside the bracket is also replaced by the midpoint.
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
/// M. Dowell and P. Jarratt, "A modified regula falsi method for computing the
/// root of an equation", *BIT* 11, 168–174 (1971).
#[derive(Debug, Clone, Copy, Default)]
pub struct Illinois;

/// Damped endpoint values, the side kept by the last interpolation, and the
/// bracket at the last contraction check.
struct Damping<S: RealScalar> {
    effective_lower: S,
    effective_upper: S,
    retained_side: i8,
    contraction_start: BracketState<S>,
    interpolated: bool,
}

impl<S: RealScalar> Damping<S> {
    fn new(bracket: BracketState<S>) -> Self {
        Self {
            effective_lower: bracket.lower.value,
            effective_upper: bracket.upper.value,
            retained_side: 0,
            contraction_start: bracket,
            interpolated: false,
        }
    }
}

impl<S: RealScalar> BracketRule<S> for Damping<S> {
    fn propose(
        &mut self,
        bracket: BracketState<S>,
        _: BracketState<S>,
        cycle: usize,
        step: usize,
        _: &ScalarRootOptions<S>,
    ) -> Result<Option<S>, SciOxideError> {
        if step > 0 {
            return Ok(None);
        }
        // Four Illinois proposals must at least halve the bracket; otherwise
        // take a bisection and start fresh damping history.
        let check = (cycle + 1).is_multiple_of(4);
        let insufficient_contraction =
            check && !bracket.half_width_contracted(self.contraction_start)?;
        if check {
            self.contraction_start = bracket;
        }
        let interpolation = false_position(
            Sample::new(bracket.lower.x, self.effective_lower),
            Sample::new(bracket.upper.x, self.effective_upper),
        )
        .ok()
        .filter(|&x| !insufficient_contraction && bracket.contains_strict(x));
        self.interpolated = interpolation.is_some();
        interpolation
            .map_or_else(|| bracket.midpoint(), Ok)
            .map(Some)
    }

    fn observe(
        &mut self,
        old: BracketState<S>,
        new: BracketState<S>,
        sample: Sample<S>,
    ) -> Result<(), SciOxideError> {
        if !self.interpolated {
            self.effective_lower = new.lower.value;
            self.effective_upper = new.upper.value;
            self.retained_side = 0;
        } else if new.lower.x == old.lower.x {
            self.effective_lower = if self.retained_side == -1 {
                self.effective_lower * S::HALF
            } else {
                old.lower.value
            };
            self.effective_upper = sample.value;
            self.retained_side = -1;
        } else {
            self.effective_lower = sample.value;
            self.effective_upper = if self.retained_side == 1 {
                self.effective_upper * S::HALF
            } else {
                old.upper.value
            };
            self.retained_side = 1;
        }
        Ok(())
    }
}

bracket_solver!(Illinois, Damping::new);
