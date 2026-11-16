//! Line-search outcomes.

use core::fmt;

use scioxide_core::{Argument, RealScalar, SciOxideError};

/// Gives the reason a line search stopped without accepting a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub enum LineSearchFailure {
    /// The search reached its trial limit.
    IterationLimit,
    /// An evaluation limit was reached.
    EvaluationLimit,
    /// The next step would be below the minimum step, or the bracket is
    /// narrower than the configured step tolerance.
    StepSizeLimit,
    /// The bracket cannot shrink further in floating-point arithmetic.
    Stagnation,
}

impl fmt::Display for LineSearchFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IterationLimit => formatter.write_str("iteration limit reached"),
            Self::EvaluationLimit => formatter.write_str("evaluation limit reached"),
            Self::StepSizeLimit => formatter.write_str("step-size limit reached"),
            Self::Stagnation => formatter.write_str("search stagnated"),
        }
    }
}

/// Says whether a line search accepted a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[must_use]
pub enum LineSearchTermination {
    /// A step satisfies the acceptance conditions of the search.
    Accepted,
    /// No step was accepted, for the given reason.
    Unsuccessful(LineSearchFailure),
}

impl fmt::Display for LineSearchTermination {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accepted => formatter.write_str("Accepted"),
            Self::Unsuccessful(reason) => write!(formatter, "Unsuccessful({reason})"),
        }
    }
}

impl LineSearchTermination {
    /// Returns `true` if the search accepted a step.
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Accepted)
    }

    /// Returns the reason the search failed, or `None` if it accepted a step.
    pub const fn failure(self) -> Option<LineSearchFailure> {
        match self {
            Self::Accepted => None,
            Self::Unsuccessful(reason) => Some(reason),
        }
    }
}

/// Describes the outcome of [`crate::LineSearch::search_into`].
///
/// The accepted point, or the best finite trial of a failed search, is written
/// to the trial slice passed to the search. When [`Self::gradient_available`]
/// is `true`, its gradient is written to the gradient slice. The counts cover
/// evaluations made inside the search; the value and gradient at the starting
/// point, which the caller supplies, are not counted.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct LineSearchResult<S: RealScalar> {
    termination: LineSearchTermination,
    has_trial: bool,
    step: S,
    value: S,
    gradient_available: bool,
    directional_derivative: Option<S>,
    fun_evals: usize,
    grad_evals: usize,
    rejected_trials: usize,
}

impl<S: RealScalar> LineSearchResult<S> {
    /// Creates the result of a search that accepted step `step` with objective value `value`.
    ///
    /// Custom [`crate::LineSearch`] implementations use this constructor.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `step`, `value` or `directional_derivative` is NaN or
    ///   infinite.
    /// - `InvalidInput` if `step ≤ 0`.
    pub fn try_accepted(
        step: S,
        value: S,
        gradient_available: bool,
        directional_derivative: Option<S>,
        fun_evals: usize,
        grad_evals: usize,
    ) -> Result<Self, SciOxideError> {
        validate_line_search_trial(
            step,
            value,
            directional_derivative,
            scioxide_core::argument!("step"),
        )?;
        Ok(Self::accepted(
            step,
            value,
            gradient_available,
            directional_derivative,
            fun_evals,
            grad_evals,
        ))
    }

    /// Creates the result of a failed search, with the best finite trial
    /// `(step, value)` if there is one.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if the step or value of `best_trial` is NaN or infinite.
    /// - `InvalidInput` if the step of `best_trial` is not positive.
    pub fn try_unsuccessful(
        failure: LineSearchFailure,
        best_trial: Option<(S, S)>,
        fun_evals: usize,
        grad_evals: usize,
    ) -> Result<Self, SciOxideError> {
        if let Some((step, value)) = best_trial {
            validate_line_search_trial(step, value, None, scioxide_core::argument!("best_trial"))?;
        }
        Ok(Self::unsuccessful(
            failure, best_trial, fun_evals, grad_evals,
        ))
    }

    /// Creates the result of an accepted step.
    pub(crate) const fn accepted(
        step: S,
        value: S,
        gradient_available: bool,
        directional_derivative: Option<S>,
        fun_evals: usize,
        grad_evals: usize,
    ) -> Self {
        Self {
            termination: LineSearchTermination::Accepted,
            has_trial: true,
            step,
            value,
            gradient_available,
            directional_derivative,
            fun_evals,
            grad_evals,
            rejected_trials: 0,
        }
    }

    /// Creates the result of a failed search with an optional best finite trial.
    pub(crate) fn unsuccessful(
        failure: LineSearchFailure,
        best_trial: Option<(S, S)>,
        fun_evals: usize,
        grad_evals: usize,
    ) -> Self {
        let (has_trial, step, value) = match best_trial {
            Some((step, value)) => (true, step, value),
            None => (false, S::ZERO, S::ZERO),
        };
        Self {
            termination: LineSearchTermination::Unsuccessful(failure),
            has_trial,
            step,
            value,
            gradient_available: false,
            directional_derivative: None,
            fun_evals,
            grad_evals,
            rejected_trials: fun_evals,
        }
    }

    /// Returns `true` if the search accepted a step.
    pub const fn is_success(self) -> bool {
        self.termination.is_success()
    }

    /// Returns the termination status.
    pub const fn termination(self) -> LineSearchTermination {
        self.termination
    }

    /// Returns the reason the search failed, or `None` if it accepted a step.
    pub const fn failure(self) -> Option<LineSearchFailure> {
        self.termination.failure()
    }

    /// Returns `true` if [`Self::step`], [`Self::value`] and the trial slice hold a trial point.
    pub const fn has_trial(self) -> bool {
        self.has_trial
    }

    /// Returns the accepted step, or the best finite step of a failed search.
    pub const fn step(self) -> S {
        self.step
    }

    /// Returns the objective value at [`Self::step`].
    pub const fn value(self) -> S {
        self.value
    }

    /// Returns `true` if the gradient slice holds the gradient at the trial point.
    pub const fn gradient_available(self) -> bool {
        self.gradient_available
    }

    /// Returns the directional derivative `∇f(x + α d)ᵀd` at the trial point, if evaluated.
    pub const fn directional_derivative(self) -> Option<S> {
        self.directional_derivative
    }

    /// Returns the number of objective evaluations made by the search.
    pub const fn fun_evals(self) -> usize {
        self.fun_evals
    }

    /// Returns the number of gradient evaluations made by the search.
    pub const fn grad_evals(self) -> usize {
        self.grad_evals
    }

    /// Returns the number of trials that the search rejected.
    pub const fn rejected_trials(self) -> usize {
        self.rejected_trials
    }

    pub(crate) const fn with_rejected_trials(mut self, rejected_trials: usize) -> Self {
        self.rejected_trials = rejected_trials;
        self
    }
}

fn validate_line_search_trial<S: RealScalar>(
    step: S,
    value: S,
    directional_derivative: Option<S>,
    argument: Argument,
) -> Result<(), SciOxideError> {
    if !step.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "step must be finite",
        ));
    }
    if step <= S::ZERO {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "line-search trial step must be positive",
        ));
    }
    if !value.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "objective value must be finite",
        ));
    }
    if directional_derivative.is_some_and(|derivative| !derivative.is_finite()) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "directional derivative must be finite",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn termination_display_names_the_variant_and_reason() {
        assert_eq!(LineSearchTermination::Accepted.to_string(), "Accepted");
        assert_eq!(
            LineSearchTermination::Unsuccessful(LineSearchFailure::Stagnation).to_string(),
            "Unsuccessful(search stagnated)"
        );
    }

    #[test]
    fn line_search_result_separates_accepted_and_diagnostic_trials() {
        let accepted =
            LineSearchResult::try_accepted(0.5_f64, 0.25, true, Some(-1.0), 2, 1).unwrap();

        assert!(accepted.is_success());
        assert!(accepted.has_trial());
        assert_eq!(accepted.step(), 0.5);
        assert!(accepted.gradient_available());
        assert_eq!(accepted.fun_evals(), 2);
        assert_eq!(accepted.grad_evals(), 1);

        let unsuccessful = LineSearchResult::try_unsuccessful(
            LineSearchFailure::IterationLimit,
            Some((0.25, 0.5)),
            4,
            0,
        )
        .unwrap();

        assert!(!unsuccessful.is_success());
        assert_eq!(unsuccessful.step(), 0.25);
        assert!(!unsuccessful.gradient_available());
        assert!(matches!(
            unsuccessful.termination(),
            LineSearchTermination::Unsuccessful(LineSearchFailure::IterationLimit)
        ));
        let empty = LineSearchResult::<f64>::try_unsuccessful(
            LineSearchFailure::IterationLimit,
            None,
            0,
            0,
        )
        .unwrap();
        assert!(!empty.has_trial());
    }

    #[test]
    fn line_search_result_rejects_invalid_trial_metadata() {
        assert!(matches!(
            (LineSearchResult::try_accepted(0.0_f64, 1.0, false, None, 0, 0))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
        ));

        for result in [
            LineSearchResult::try_accepted(f64::NAN, 1.0, false, None, 0, 0),
            LineSearchResult::try_accepted(1.0, f64::INFINITY, false, None, 0, 0),
            LineSearchResult::try_accepted(1.0, 1.0, false, Some(f64::NEG_INFINITY), 0, 0),
            LineSearchResult::try_unsuccessful(
                LineSearchFailure::IterationLimit,
                Some((1.0, f64::NAN)),
                0,
                0,
            ),
        ] {
            assert!(matches!(
                (result)
                    .as_ref()
                    .map_err(scioxide_core::SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
            ));
        }
    }
}
