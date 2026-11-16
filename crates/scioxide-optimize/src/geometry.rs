//! Projection, feasibility, nonsmooth merit and finite boxes.
use alloc::vec::Vec;

use scioxide_core::{
    Bounds, InvalidInputReason, NumericScalar, RealScalar, SciOxideError, checked_difference,
    checked_float_quotient,
};

pub(crate) fn project_component<S: RealScalar>(value: S, bounds: (Option<S>, Option<S>)) -> S {
    let value = bounds.0.map_or(value, |lower| value.max(lower));
    bounds.1.map_or(value, |upper| value.min(upper))
}
pub(crate) fn project<S: RealScalar>(point: &mut [S], bounds: &[(Option<S>, Option<S>)]) {
    for (value, &bounds) in point.iter_mut().zip(bounds) {
        *value = project_component(*value, bounds);
    }
}
pub(crate) fn violation<S: RealScalar>(values: &[S], equalities: usize) -> S {
    values
        .iter()
        .enumerate()
        .fold(S::ZERO, |maximum, (row, &value)| {
            maximum.max(if row < equalities { value.abs() } else { value })
        })
}
pub(crate) fn bound_violation<S: RealScalar>(point: &[S], bounds: &[(Option<S>, Option<S>)]) -> S {
    point
        .iter()
        .zip(bounds)
        .fold(S::ZERO, |maximum, (&value, &(lower, upper))| {
            maximum
                .max(lower.map_or(S::ZERO, |b| (b - value).max(S::ZERO)))
                .max(upper.map_or(S::ZERO, |b| (value - b).max(S::ZERO)))
        })
}
pub(crate) fn absolute_directional<S: RealScalar>(value: S, derivative: S) -> S {
    if value > S::ZERO {
        derivative
    } else if value < S::ZERO {
        -derivative
    } else {
        derivative.abs()
    }
}
pub(crate) fn positive_directional<S: RealScalar>(value: S, derivative: S) -> S {
    if value > S::ZERO {
        derivative
    } else if value < S::ZERO {
        S::ZERO
    } else {
        derivative.max(S::ZERO)
    }
}

/// Holds a box `lᵢ ≤ xᵢ ≤ uᵢ` with finite endpoints and `lᵢ < uᵢ`.
///
/// The global searches convert the bounds of their
/// [`BoundedObjective`](crate::BoundedObjective) into this form and pass it to
/// start designs and basin proposals. [`Nsga2Problem`](crate::Nsga2Problem)
/// takes it as its search box.
#[derive(Debug, Clone, PartialEq)]
pub struct FiniteBounds<S: RealScalar> {
    entries: Vec<(S, S)>,
}

impl<S: RealScalar> FiniteBounds<S> {
    /// Creates the box from `(lower, upper)` pairs.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`Empty`) if `entries` is empty.
    /// - `NonFinite` if an endpoint is NaN or infinite.
    /// - `InvalidInput` (`OutOfRange`) if a lower endpoint is not less than its
    ///   upper endpoint.
    pub fn new(entries: Vec<(S, S)>) -> Result<Self, SciOxideError> {
        if entries.is_empty() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("bounds"),
                InvalidInputReason::Empty,
                "finite bounds must describe at least one variable",
            ));
        }
        for (index, &(lower, upper)) in entries.iter().enumerate() {
            if !lower.is_finite() || !upper.is_finite() {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Input,
                    alloc::format!("bounds for variable {index} must be finite"),
                ));
            }
            if lower >= upper {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("bounds"),
                    InvalidInputReason::OutOfRange,
                    alloc::format!(
                        "lower bound must be less than upper bound for variable {index}"
                    ),
                ));
            }
        }
        Ok(Self { entries })
    }

    /// Returns the number of variables.
    #[must_use]
    #[allow(
        clippy::len_without_is_empty,
        reason = "FiniteBounds is validated as non-empty"
    )]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns the `(lower, upper)` pairs.
    #[must_use]
    pub fn as_slice(&self) -> &[(S, S)] {
        &self.entries
    }

    /// Returns the `(lower, upper)` pairs by value.
    #[must_use]
    pub fn into_entries(self) -> Vec<(S, S)> {
        self.entries
    }

    /// Unvalidated empty storage that a workspace fills with [`Self::assign`].
    pub(crate) const fn unassigned() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Whether `dimension` pairs fit without growing.
    pub(crate) fn fits(&self, dimension: usize) -> bool {
        self.entries.capacity() >= dimension
    }

    /// Reserve storage for `dimension` pairs.
    pub(crate) fn try_reserve(&mut self, dimension: usize) -> Result<(), SciOxideError> {
        let additional = dimension.saturating_sub(self.entries.len());
        scioxide_core::backend::try_reserve(
            &mut self.entries,
            additional,
            scioxide_core::quantity!("finite_bounds"),
        )
    }

    /// Validate problem `bounds` as a strict finite box and copy them here.
    ///
    /// Validates before writing, and reuses the reserved capacity.
    pub(crate) fn assign(&mut self, bounds: &Bounds<S>) -> Result<(), SciOxideError> {
        finite_box(bounds, true)?;
        self.try_reserve(bounds.len())?;
        self.entries.clear();
        self.entries.extend(
            bounds
                .as_slice()
                .iter()
                .map(|&(lower, upper)| (lower.unwrap_or(S::ZERO), upper.unwrap_or(S::ZERO))),
        );
        Ok(())
    }

    /// Whether `point` has this dimension and finite coordinates inside the box.
    pub(crate) fn contains(&self, point: &[S]) -> bool {
        point.len() == self.entries.len()
            && point
                .iter()
                .zip(&self.entries)
                .all(|(&value, &(lower, upper))| {
                    value.is_finite() && value >= lower && value <= upper
                })
    }
}

/// Require finite endpoints on every coordinate of a global search box.
///
/// `strict` additionally requires `lower < upper`; otherwise equal endpoints
/// fix a coordinate.
///
/// # Errors
/// An absent endpoint is `UnsupportedConfiguration`; a degenerate interval
/// under `strict` is `OutOfRange`.
pub(crate) fn finite_box<S: RealScalar>(
    bounds: &Bounds<S>,
    strict: bool,
) -> Result<(), SciOxideError> {
    for &(lower, upper) in bounds.as_slice() {
        let (Some(lower), Some(upper)) = (lower, upper) else {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("problem"),
                InvalidInputReason::UnsupportedConfiguration,
                "global search requires finite lower and upper bounds",
            ));
        };
        if strict && lower >= upper {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("problem"),
                InvalidInputReason::OutOfRange,
                "global search requires every lower bound below its upper bound",
            ));
        }
    }
    Ok(())
}

/// Map an in-box coordinate to `[0, 1]` without overflowing extreme boxes.
pub(crate) fn normalized_coordinate<S: RealScalar + NumericScalar<Real = S>>(
    value: S,
    lower: S,
    upper: S,
) -> Result<S, SciOxideError> {
    if value == lower {
        return Ok(<S as scioxide_core::AlgebraicScalar>::ZERO);
    }
    if value == upper {
        return Ok(<S as scioxide_core::AlgebraicScalar>::ONE);
    }
    let scale = NumericScalar::abs(lower).max(NumericScalar::abs(upper));
    let scaled_value = checked_float_quotient(value, scale)?;
    let scaled_lower = checked_float_quotient(lower, scale)?;
    let scaled_upper = checked_float_quotient(upper, scale)?;
    let numerator = checked_difference(scaled_value, scaled_lower)?;
    let denominator = checked_difference(scaled_upper, scaled_lower)?;
    let coordinate = checked_float_quotient(numerator, denominator)?;
    if coordinate < <S as scioxide_core::AlgebraicScalar>::ZERO
        || coordinate > <S as scioxide_core::AlgebraicScalar>::ONE
    {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "an in-bounds initial coordinate was not representable in [0, 1]",
        ));
    }
    Ok(coordinate)
}

/// Return a finite normalized coordinate to `[0, 1]` by clipping or reflection.
pub(crate) fn repair<S: RealScalar>(value: S, policy: crate::DifferentialEvolutionBoundary) -> S {
    let zero = S::ZERO;
    let one = S::ONE;
    match policy {
        crate::DifferentialEvolutionBoundary::Clip => value.max(zero).min(one),
        crate::DifferentialEvolutionBoundary::Reflect => {
            let two = S::TWO;
            let folded = value % two;
            let positive = if folded < zero { folded + two } else { folded };
            if positive > one {
                two - positive
            } else {
                positive
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection_violation_and_merit_kinks() {
        let bounds = [
            (Some(1.0), Some(1.0)),
            (None, Some(2.0)),
            (Some(-1.0), None),
        ];
        let mut point = [0.0, 3.0, -2.0];
        assert_eq!(bound_violation(&point, &bounds), 1.0);
        project(&mut point, &bounds);
        assert_eq!(point, [1.0, 2.0, -1.0]);
        assert_eq!(bound_violation(&point, &bounds), 0.0);
        assert_eq!(violation(&[-2.0, -3.0, 1.0], 1), 2.0);
        assert_eq!(absolute_directional(0.0, -2.0), 2.0);
        assert_eq!(positive_directional(0.0, -2.0), 0.0);
        assert_eq!(positive_directional(-1.0, 2.0), 0.0);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod box_tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn finite_bounds_require_strict_finite_intervals() {
        assert!(FiniteBounds::new(vec![(0.0_f64, 1.0)]).is_ok());
        assert!(matches!(
            (FiniteBounds::new(vec![(1.0_f64, 1.0)]))
                .as_ref()
                .map_err(SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
        ));
        assert!(matches!(
            (FiniteBounds::new(vec![(0.0_f64, f64::INFINITY)]))
                .as_ref()
                .map_err(SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
        assert!(FiniteBounds::<f64>::new(Vec::new()).is_err());
    }

    #[test]
    fn problem_boxes_are_validated_before_assignment() {
        let mut storage = FiniteBounds::unassigned();
        storage
            .assign(&Bounds::new(vec![(Some(-1.0_f64), Some(2.0))]).unwrap())
            .unwrap();
        assert_eq!(storage.as_slice(), &[(-1.0, 2.0)]);
        assert!(storage.contains(&[2.0]));
        assert!(!storage.contains(&[2.5]));
        assert!(!storage.contains(&[f64::NAN]));
        for bounds in [
            Bounds::new(vec![(None, Some(1.0_f64))]).unwrap(),
            Bounds::new(vec![(Some(1.0_f64), Some(1.0))]).unwrap(),
        ] {
            assert!(storage.assign(&bounds).is_err());
            assert_eq!(storage.as_slice(), &[(-1.0, 2.0)]);
        }
        let fixed = Bounds::new(vec![(Some(1.0_f64), Some(1.0))]).unwrap();
        assert!(finite_box(&fixed, false).is_ok());
    }

    #[test]
    fn reflection_folds_multiple_crossings_exactly() {
        use crate::DifferentialEvolutionBoundary;
        for (input, expected) in [
            (-4.25, 0.25),
            (-3.75, 0.25),
            (-1., 1.),
            (-0.25, 0.25),
            (0., 0.),
            (1., 1.),
            (1.25, 0.75),
            (4.25, 0.25),
        ] {
            assert_eq!(
                repair(input, DifferentialEvolutionBoundary::Reflect),
                expected
            );
            assert_eq!(
                repair(input as f32, DifferentialEvolutionBoundary::Reflect),
                expected as f32
            );
            assert_eq!(
                repair(input, DifferentialEvolutionBoundary::Clip),
                input.clamp(0., 1.)
            );
        }
    }

    #[test]
    fn extreme_boxes_normalize_without_overflow() {
        let (lower, upper) = (-f64::MAX, f64::MAX);
        assert_eq!(normalized_coordinate(lower, lower, upper).unwrap(), 0.0);
        assert_eq!(normalized_coordinate(0.0, lower, upper).unwrap(), 0.5);
        assert_eq!(normalized_coordinate(upper, lower, upper).unwrap(), 1.0);
    }
}
