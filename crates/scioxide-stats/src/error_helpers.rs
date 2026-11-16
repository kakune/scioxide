//! Error constructors, finiteness checks and conversions used across the crate.
use crate::KernelWeights;
use alloc::borrow::ToOwned;
use alloc::vec::Vec;
use scioxide_core::{
    Argument, InvalidInputReason, NonFiniteOrigin, NumericScalar, RealScalar, Scaled, SciOxideError,
};

pub(crate) fn invalid(
    argument: Argument,
    reason: InvalidInputReason,
    detail: &'static str,
) -> SciOxideError {
    SciOxideError::invalid_input(argument, reason, detail)
}

/// Relabels fixed configuration arguments at a one-shot boundary.
pub(crate) fn one_shot_argument(
    error: SciOxideError,
    omitted: &[&str],
    argument: Argument,
) -> SciOxideError {
    if let scioxide_core::SciOxideErrorView::InvalidInput {
        argument: original,
        reason,
        detail,
    } = error.view()
        && omitted.iter().any(|name| {
            original.as_str() == *name
                || original
                    .as_str()
                    .strip_prefix(name)
                    .is_some_and(|suffix| suffix.starts_with('.'))
        })
    {
        return SciOxideError::invalid_input(argument, reason, detail.to_owned());
    }
    error
}

pub(crate) fn input_finite<T: NumericScalar>(value: T) -> Result<T, SciOxideError> {
    finite_origin(value, NonFiniteOrigin::Input)
}

pub(crate) fn callback_finite<T: NumericScalar>(value: T) -> Result<T, SciOxideError> {
    finite_origin(value, NonFiniteOrigin::Callback)
}

fn finite_origin<T: NumericScalar>(value: T, origin: NonFiniteOrigin) -> Result<T, SciOxideError> {
    if value.has_finite_components() {
        Ok(value)
    } else {
        Err(SciOxideError::non_finite(
            origin,
            "non-finite statistical value",
        ))
    }
}

pub(crate) fn finite<T: NumericScalar>(value: T) -> Result<T, SciOxideError> {
    if !value.has_finite_components()
        && !value.real_component().is_nan()
        && !value.imaginary_component().is_nan()
    {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "statistical result overflow",
        ));
    }
    finite_origin(value, NonFiniteOrigin::Output)
}

pub(crate) fn number<S: RealScalar>(value: f64) -> Result<S, SciOxideError> {
    finite(value)?;
    finite(S::try_from_f64_lossy(value).map_err(|_| {
        SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "statistical scalar overflow",
        )
    })?)
}

pub(crate) fn count<S: RealScalar>(value: usize) -> Result<S, SciOxideError> {
    S::try_from_usize_exact(value)
}

pub(crate) fn zeros<S: RealScalar>(length: usize) -> Result<Vec<S>, SciOxideError> {
    scioxide_core::backend::try_filled(
        length,
        S::ZERO,
        scioxide_core::quantity!("statistical_storage"),
    )
}

pub(crate) fn limit(
    required: usize,
    limit: usize,
    quantity: scioxide_core::Quantity,
) -> Result<(), SciOxideError> {
    if required > limit {
        Err(SciOxideError::resource_limit(quantity, required, limit))
    } else {
        Ok(())
    }
}

/// Checks the workspace size in bytes against the configured limit.
pub(crate) fn limit_bytes(bytes: usize, limit_bytes: usize) -> Result<(), SciOxideError> {
    limit(
        bytes,
        limit_bytes,
        scioxide_core::quantity!("statistics_workspace_bytes"),
    )
}

/// Checks the required work against the configured work limit.
///
/// Too small a limit is a convergence failure, not a `ResourceLimit` error.
pub(crate) fn work(
    required: usize,
    allowance: usize,
    kind: scioxide_core::BudgetKind,
    detail: &'static str,
) -> Result<(), SciOxideError> {
    if required <= allowance {
        return Ok(());
    }
    Err(work_exhausted(allowance, kind, detail))
}

/// Returns the convergence failure for an exhausted work limit of `kind`.
///
/// A positive limit is included in the budget report.
pub(crate) fn work_exhausted(
    allowance: usize,
    kind: scioxide_core::BudgetKind,
    detail: &'static str,
) -> SciOxideError {
    use scioxide_core::{BudgetKind, ConvergenceDiagnostics, ConvergenceReason};
    let diagnostics = scioxide_core::Budget::try_from(allowance).map_or_else(
        |_| ConvergenceDiagnostics::new(),
        |budget| {
            ConvergenceDiagnostics::new().with_budget(
                scioxide_core::SolveBudgets::unbounded()
                    .with_limit(kind, budget)
                    .report(kind),
            )
        },
    );
    let reason = match kind {
        BudgetKind::Iteration => ConvergenceReason::IterationLimit,
        BudgetKind::Evaluation => ConvergenceReason::EvaluationLimit,
        kind => ConvergenceReason::ResourceLimit(kind),
    };
    SciOxideError::convergence_failure(reason, diagnostics, detail)
}

/// Checks that a sample is nonempty and finite.
pub(crate) fn sample<S: RealScalar>(values: &[S], argument: Argument) -> Result<(), SciOxideError> {
    count::<S>(values.len())?;
    if values.is_empty() {
        return Err(invalid(
            argument,
            scioxide_core::InvalidInputReason::Inconsistent,
            "empty retained sample",
        ));
    }
    if !values.iter().fold(true, |finite, x| finite & x.is_finite()) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "nonfinite sample",
        ));
    }
    Ok(())
}
pub(crate) fn mass<S: RealScalar>(
    weights: Option<KernelWeights<'_, S>>,
    i: usize,
    argument: Argument,
) -> Result<Scaled<f64>, SciOxideError> {
    Scaled::<f64>::from_value(mass_value(weights, i, argument)?)
}
/// Checks a mass and returns it as an `f64` before the arithmetic is chosen.
#[inline]
pub(crate) fn mass_value<S: RealScalar>(
    weights: Option<KernelWeights<'_, S>>,
    i: usize,
    argument: Argument,
) -> Result<f64, SciOxideError> {
    match weights {
        None => Ok(1.0),
        Some(KernelWeights::Frequency(w)) => Ok(w[i] as f64),
        Some(KernelWeights::Mass(w)) => {
            if !w[i].is_finite() || w[i] < S::ZERO {
                return Err(invalid(
                    argument,
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "mass must be finite and nonnegative",
                ));
            }
            Ok(scioxide_core::RealScalar::into_f64(w[i]))
        }
    }
}
pub(crate) fn check_weights<S: RealScalar>(
    weights: Option<KernelWeights<'_, S>>,
    n: usize,
    argument: Argument,
) -> Result<(), SciOxideError> {
    let len = match weights {
        None => n,
        Some(KernelWeights::Mass(w)) => w.len(),
        Some(KernelWeights::Frequency(w)) => w.len(),
    };
    if len != n {
        return Err(invalid(
            argument,
            scioxide_core::InvalidInputReason::Inconsistent,
            "weight length mismatch",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use scioxide_core::{NonFiniteOrigin, NumericalIssue, SciOxideErrorKind, SciOxideErrorView};
    #[test]
    fn origins_and_output_overflow_are_distinct() -> Result<(), SciOxideError> {
        for (result, expected) in [
            (input_finite(f64::NAN), NonFiniteOrigin::Input),
            (callback_finite(f64::INFINITY), NonFiniteOrigin::Callback),
            (finite(f64::NAN), NonFiniteOrigin::Output),
        ] {
            assert!(
                matches!(result.unwrap_err().view(), SciOxideErrorView::NonFinite { origin, .. } if origin == expected)
            );
        }
        assert!(matches!(
            finite(f64::INFINITY).err().map(|e| e.kind()),
            Some(SciOxideErrorKind::NumericalInstability)
        ));
        assert!(matches!(
            number::<f32>(f64::MAX).unwrap_err().view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
        assert_eq!(finite(-0.0_f64)?.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(finite(f64::from_bits(1))?, f64::from_bits(1));
        Ok(())
    }
    #[test]
    fn exact_counts_and_size_boundaries() -> Result<(), SciOxideError> {
        assert_eq!(count::<f32>(1 << 24)?, 16_777_216.);
        assert!(count::<f32>((1 << 24) + 1).is_err());
        assert_eq!(
            scioxide_core::backend::checked_mul(
                0,
                usize::MAX,
                scioxide_core::quantity!("test_bytes")
            )?,
            0
        );
        assert_eq!(
            scioxide_core::backend::checked_mul(
                usize::MAX,
                2,
                scioxide_core::quantity!("test_bytes")
            )
            .err()
            .map(|e| e.kind()),
            Some(SciOxideErrorKind::SizeOverflow)
        );
        Ok(())
    }
}
