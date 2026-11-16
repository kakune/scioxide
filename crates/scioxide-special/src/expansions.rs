//! Shared budgeted Chebyshev evaluation, real conversion and nonnegative-domain validation.

use crate::SpecialFunctionTermination;
use crate::result::{Bound, Evaluation};
use scioxide_core::{RealScalar, SciOxideError};

pub(crate) fn real<S: RealScalar>(value: f64) -> Result<S, SciOxideError> {
    S::try_from_f64_lossy(value)
}

/// Require finite nonnegative `argument`.
pub(crate) fn nonnegative<S: RealScalar>(
    x: S,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if !x.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "x must be finite",
        ));
    }
    if x < S::ZERO {
        return Err(SciOxideError::domain_error(
            argument,
            scioxide_core::DomainReason::OutsideDomain,
            "x must be nonnegative",
        ));
    }
    Ok(())
}

/// Clenshaw recurrence with the half-weight convention `c[0]/2` for the constant term.
/// Omitted coefficients bound an interrupted expansion on [-1,1]. One
/// coefficient consumes one work unit, including the constant coefficient.
#[inline]
pub(crate) fn chebyshev<S: RealScalar>(
    x: S,
    coefficients: &[f64],
    work: &mut crate::support::Work,
) -> Result<(S, S), SciOxideError> {
    let result =
        crate::support::polynomial::chebyshev(x, coefficients, true, false, real::<S>, work);
    work.check()?;
    result
}

/// A bounded expansion evaluated.
///
/// An interrupted expansion keeps its value with a heuristic bound and
/// `IterationLimit`; without a useful bound no value is returned.
#[inline]
pub(crate) fn bounded<S: RealScalar>(
    value: S,
    bound: Bound<S>,
    algorithm: scioxide_core::AlgorithmId,
    work: usize,
    complete: bool,
) -> Result<Evaluation<S>, SciOxideError> {
    if !complete && bound.absolute == S::MAX {
        return Err(crate::result::exhausted_work(work));
    }
    let value = crate::support::check_finite(value)?;
    let (bound, termination) = if !complete {
        (
            Bound::heuristic(bound.absolute),
            SpecialFunctionTermination::IterationLimit,
        )
    } else if work == 0 && bound.absolute == S::ZERO {
        (bound, SpecialFunctionTermination::Exact)
    } else {
        // Final subnormal rounding of the reported value.
        (
            bound.with_subnormal_rounding(value.abs(), real::<S>(4.0)?),
            SpecialFunctionTermination::Converged,
        )
    };
    Ok(Evaluation::finite(
        value,
        bound,
        None,
        algorithm,
        work,
        termination,
    ))
}
