//! Shared backtracking progression and trust-radius policies.
use scioxide_core::{Budget, RealScalar, Scaled, SciOxideError};

pub(crate) const TRIALS: Budget = Budget::ONE.saturating_add(39);
pub(crate) fn sufficient_decrease<S: RealScalar>() -> S {
    let ten = S::TWO * (S::TWO + S::TWO + S::ONE);
    S::ONE / (ten * ten * ten * ten)
}

/// A callback returns a completed outcome, or requests the next contracted step.
/// Method-specific merit evaluation and correction steps stay in the callback.
pub(crate) fn backtrack<S: RealScalar, T>(
    initial: S,
    minimum: S,
    contraction: S,
    trials: Budget,
    mut trial: impl FnMut(usize, S) -> Result<Option<T>, SciOxideError>,
) -> Result<Result<T, crate::LineSearchFailure>, SciOxideError> {
    let mut alpha = initial;
    for attempt in 0..trials.get() {
        if alpha < minimum {
            return Ok(Err(crate::LineSearchFailure::StepSizeLimit));
        }
        if let Some(value) = trial(attempt, alpha)? {
            return Ok(Ok(value));
        }
        alpha *= contraction;
    }
    Ok(Err(crate::LineSearchFailure::IterationLimit))
}

pub(crate) fn next_radius<S: RealScalar>(
    radius: S,
    ratio: Scaled<S>,
    boundary: bool,
) -> Result<S, SciOxideError> {
    if ratio.cmp_value(Scaled::from_value(S::HALF * S::HALF)?) == core::cmp::Ordering::Less {
        Ok(radius * S::HALF * S::HALF)
    } else if boundary
        && ratio.cmp_value(Scaled::from_value(S::ONE - S::HALF * S::HALF)?)
            == core::cmp::Ordering::Greater
    {
        let maximum = S::try_from_f64_lossy(1e8)?;
        Ok(radius + radius.min((maximum - radius).max(S::ZERO)))
    } else {
        Ok(radius)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    #[test]
    fn trust_radius_strict_boundaries_and_cap() {
        for (ratio, boundary, expected) in [
            (0.2, false, 0.25),
            (0.25, true, 1.0),
            (0.75, true, 1.0),
            (0.8, false, 1.0),
            (0.8, true, 2.0),
        ] {
            assert_eq!(
                next_radius(1.0, Scaled::from_value(ratio).unwrap(), boundary).unwrap(),
                expected
            );
        }
        assert_eq!(next_radius(1e8, Scaled::ONE, true).unwrap(), 1e8);
    }
    #[test]
    fn backtracking_limits_and_callback_failure() {
        let mut steps = alloc::vec::Vec::new();
        let result = backtrack(1.0, 0.3, 0.5, TRIALS, |_, a| {
            steps.push(a);
            Ok(None::<()>)
        })
        .unwrap();
        assert_eq!(steps, &[1.0, 0.5]);
        assert_eq!(result, Err(crate::LineSearchFailure::StepSizeLimit));
        let error = backtrack(1.0, 0.0, 0.5, TRIALS, |_, _| {
            Err::<Option<()>, _>(SciOxideError::singular(None))
        })
        .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::Singular { .. }
        ));
    }
}
