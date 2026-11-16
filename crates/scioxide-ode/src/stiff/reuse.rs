//! Pure reuse decisions shared by ODE stages and residual DAE Newton solves.
use super::newton::NewtonReport;
use scioxide_core::{AlgebraicScalar, NumericScalar, RealScalar, SciOxideError};

/// Unit floors avoid refreshing a linearization at tolerance-sized movements.
pub(crate) fn movement<S: RealScalar>(old: S, new: S) -> S {
    (new - old).abs() / S::ONE.max(old.abs()).max(new.abs())
}

pub(crate) fn movement_limit<S: RealScalar>() -> Result<S, SciOxideError> {
    S::try_from_f64_lossy(0.5)
}

pub(crate) fn within_movement<T: NumericScalar>(old: &[T], new: &[T], limit: T::Real) -> bool {
    old.iter()
        .zip(new)
        .all(|(&old, &new)| (new - old).abs() / T::Real::ONE.max(old.abs()).max(new.abs()) <= limit)
}

/// A nearby factor is a chord approximation, subject to Newton rate checks.
pub(crate) fn nearby<S: RealScalar>(ratio: S) -> Result<bool, SciOxideError> {
    Ok(ratio >= S::try_from_f64_lossy(0.8)? && ratio <= S::try_from_f64_lossy(1.2)?)
}

pub(crate) fn slow<S: RealScalar>(report: NewtonReport<S>) -> Result<bool, SciOxideError> {
    let limit = S::try_from_f64_lossy(0.3)?;
    Ok(report.corrections > 2 && report.rate.is_some_and(|rate| rate > limit))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn factor_window_is_closed_and_rejects_nonfinite_ratios() {
        for ratio in [0.8_f64, 1., 1.2] {
            assert!(nearby(ratio).unwrap());
        }
        for ratio in [0.799_f64, 1.201, -1., f64::NAN, f64::INFINITY] {
            assert!(!nearby(ratio).unwrap());
        }
    }

    #[test]
    fn movement_uses_unit_floors_and_both_endpoints() {
        assert!(within_movement(&[0_f64, 2.], &[0.5, 4.], 0.5));
        assert!(!within_movement(&[0_f64, 2.], &[0.501, 4.], 0.5));
        assert_eq!(movement(2_f64, 4.), 0.5);
    }
}
