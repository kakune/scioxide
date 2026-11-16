//! Crate-private input validation, with one error kind per condition.
//!
//! | Condition | Error |
//! |---|---|
//! | Required collection empty | `InvalidInput(Empty)` |
//! | Nonempty but fewer entries than the method needs | `InvalidInput(OutOfRange)` |
//! | Length disagreeing with its partner | `DimensionMismatch` |
//! | Non-finite input coordinate, value or weight | `NonFinite(Input)` |
//! | Decreasing or coincident nodes | `InvalidInput(Inconsistent)` |
//! | Negative (or, where required, nonpositive) weight | `InvalidInput(OutOfRange)` |
//! | No positive observation weight | `InvalidInput(Inconsistent)` |
//! | Scattered points coinciding under the separation policy | `InvalidInput(Inconsistent)` |
//!
//! Knot vectors are validated by `BSplineBasis`, and conflicting duplicate
//! sites by `ScatteredValues`; both use the kinds above.

use scioxide_core::{Argument, NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError};

use crate::error::{empty, invalid, out_of_range, shape};
use crate::protocol::NodeSeparation;

/// Require `len >= minimum` entries.
pub(crate) fn validate_count(
    len: usize,
    minimum: usize,
    argument: Argument,
) -> Result<(), SciOxideError> {
    if len == 0 && minimum > 0 {
        Err(empty(argument, "a nonempty input is required"))
    } else if len < minimum {
        Err(out_of_range(
            argument,
            "fewer entries than required by this interpolation method",
        ))
    } else {
        Ok(())
    }
}

/// Require a length to equal its partner's.
pub(crate) fn validate_length(expected: usize, actual: usize) -> Result<(), SciOxideError> {
    if expected == actual {
        Ok(())
    } else {
        Err(shape(expected, actual))
    }
}

/// Validate at least `min_len` finite, strictly increasing nodes that are
/// distinct under `separation`.
///
/// Ordering is checked before separation, so a decreasing pair is an ordering
/// error; equal nodes are rejected by every policy.
pub(crate) fn validate_nodes<S: RealScalar>(
    x: &[S],
    min_len: usize,
    separation: NodeSeparation<S>,
    argument: Argument,
) -> Result<(), SciOxideError> {
    validate_count(x.len(), min_len, argument)?;
    validate_finite(x)?;
    for w in x.windows(2) {
        if w[1] < w[0] {
            return Err(invalid(
                argument,
                "nodes must be strictly monotonically increasing",
            ));
        }
        if separation.coincides(w[0], w[1])? {
            return Err(invalid(
                argument,
                "nodes are not distinct under the configured separation policy",
            ));
        }
    }
    Ok(())
}

/// Validate finite real or complex input values.
pub(crate) fn validate_finite<'a, C: NumericScalar + 'a>(
    values: impl IntoIterator<Item = &'a C>,
) -> Result<(), SciOxideError> {
    validate_finite_origin(values, NonFiniteOrigin::Input)
}

/// Validate finite values, reporting a failure with the given origin.
pub(crate) fn validate_finite_origin<'a, C: NumericScalar + 'a>(
    values: impl IntoIterator<Item = &'a C>,
    origin: NonFiniteOrigin,
) -> Result<(), SciOxideError> {
    if values
        .into_iter()
        .all(|value| value.has_finite_components())
    {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(
            origin,
            "encountered a non-finite value",
        ))
    }
}

/// Sign requirement of a weight or positive-parameter vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Weights {
    /// Every weight is strictly positive.
    Positive,
    /// Every weight is nonnegative.
    Nonnegative,
    /// Nonnegative with at least one positive observation.
    Observed,
}

/// Validate `expected` finite weights (or positive parameters) against `requirement`.
pub(crate) fn validate_weights<S: RealScalar>(
    weights: &[S],
    expected: usize,
    requirement: Weights,
    argument: Argument,
) -> Result<(), SciOxideError> {
    validate_length(expected, weights.len())?;
    validate_finite(weights)?;
    if requirement == Weights::Positive && weights.iter().any(|&w| w <= S::ZERO) {
        return Err(out_of_range(argument, "entries must be strictly positive"));
    }
    if weights.iter().any(|&w| w < S::ZERO) {
        return Err(out_of_range(argument, "entries must be nonnegative"));
    }
    if requirement == Weights::Observed && weights.iter().all(|&w| w == S::ZERO) {
        return Err(invalid(argument, "at least one entry must be positive"));
    }
    Ok(())
}

/// Validate that no two finite points coincide under `separation`, comparing
/// Euclidean distances at the larger coordinate magnitude. `O(n²)` work,
/// polling `check` once per point.
pub(crate) fn validate_distinct_points<S: RealScalar, const D: usize>(
    points: &[[S; D]],
    separation: NodeSeparation<S>,
    argument: Argument,
    mut check: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    let scale = |point: &[S; D]| point.iter().fold(S::ZERO, |m, x| m.max(x.abs()));
    for (i, left) in points.iter().enumerate() {
        check()?;
        for right in &points[..i] {
            let coincides = match crate::geometry::distance(left, right) {
                Ok(distance) => {
                    separation.is_zero_at_scale(distance, scale(left).max(scale(right)))?
                }
                Err(_) => {
                    let distance = crate::geometry::extended_distance(left, right, &[S::ONE; D])?;
                    match separation {
                        NodeSeparation::Exact => distance.is_zero(),
                        NodeSeparation::Tolerant(tolerance) => {
                            // Keep the existing validation of representability of
                            // the configured scalar tolerance threshold.
                            let threshold = tolerance.threshold(scale(left).max(scale(right)))?;
                            distance.abs_le(scioxide_core::Scaled::from_value(threshold)?)
                        }
                    }
                }
            };
            if coincides {
                return Err(invalid(
                    argument,
                    "points are not distinct under the configured separation policy",
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::{InvalidInputReason, SciOxideErrorKind, SciOxideErrorView, Tolerance};

    fn reason(result: Result<(), SciOxideError>) -> InvalidInputReason {
        match result.unwrap_err().view() {
            SciOxideErrorView::InvalidInput { reason, .. } => Some(reason),
            _ => None,
        }
        .unwrap()
    }

    fn kind(result: Result<(), SciOxideError>) -> SciOxideErrorKind {
        result.unwrap_err().kind()
    }

    const X: Argument = scioxide_core::argument!("x");

    #[test]
    fn counts_distinguish_empty_from_too_few() {
        assert!(validate_count(0, 0, X).is_ok());
        assert!(validate_count(2, 2, X).is_ok());
        assert_eq!(reason(validate_count(0, 1, X)), InvalidInputReason::Empty);
        assert_eq!(
            reason(validate_count(1, 2, X)),
            InvalidInputReason::OutOfRange
        );
        assert_eq!(
            kind(validate_length(2, 3)),
            SciOxideErrorKind::DimensionMismatch
        );
        assert!(validate_length(3, 3).is_ok());
    }

    #[test]
    fn nodes_report_count_finiteness_order_and_separation() {
        let exact = NodeSeparation::Exact;
        assert!(validate_nodes(&[0.0_f64, 1.0], 2, exact, X).is_ok());
        assert_eq!(
            reason(validate_nodes::<f64>(&[], 1, exact, X)),
            InvalidInputReason::Empty
        );
        assert_eq!(
            reason(validate_nodes(&[0.0_f64], 2, exact, X)),
            InvalidInputReason::OutOfRange
        );
        assert!(matches!(
            validate_nodes(&[0.0, f64::NAN], 1, exact, X)
                .unwrap_err()
                .view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
        assert_eq!(
            reason(validate_nodes(&[1.0_f64, 0.0], 1, exact, X)),
            InvalidInputReason::Inconsistent
        );
        assert_eq!(
            reason(validate_nodes(&[1.0_f64, 1.0], 1, exact, X)),
            InvalidInputReason::Inconsistent
        );
        let tolerant = NodeSeparation::Tolerant(Tolerance::new(1e-9, 0.0).unwrap());
        assert_eq!(
            reason(validate_nodes(&[0.0_f64, 1e-12], 1, tolerant, X)),
            InvalidInputReason::Inconsistent
        );
        // Opposite-sign extremes are separated without overflowing the gap.
        assert!(validate_nodes(&[-f64::MAX, f64::MAX], 2, tolerant, X).is_ok());
    }

    #[test]
    fn weights_report_shape_finiteness_sign_and_observation() {
        for requirement in [Weights::Positive, Weights::Nonnegative, Weights::Observed] {
            assert!(validate_weights(&[1.0_f64, 2.0], 2, requirement, X).is_ok());
            assert_eq!(
                kind(validate_weights(&[1.0_f64], 2, requirement, X)),
                SciOxideErrorKind::DimensionMismatch
            );
            assert!(matches!(
                validate_weights(&[f64::INFINITY], 1, requirement, X)
                    .unwrap_err()
                    .view(),
                SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Input,
                    ..
                }
            ));
            assert_eq!(
                reason(validate_weights(&[-1.0_f64], 1, requirement, X)),
                InvalidInputReason::OutOfRange
            );
        }
        assert_eq!(
            reason(validate_weights(&[0.0_f64, 1.0], 2, Weights::Positive, X)),
            InvalidInputReason::OutOfRange
        );
        assert!(validate_weights(&[0.0_f64, 0.0], 2, Weights::Nonnegative, X).is_ok());
        assert!(validate_weights(&[0.0_f64, 1.0], 2, Weights::Observed, X).is_ok());
        assert_eq!(
            reason(validate_weights(&[0.0_f64, 0.0], 2, Weights::Observed, X)),
            InvalidInputReason::Inconsistent
        );
    }

    #[test]
    fn distinct_points_use_scaled_separation_and_poll() {
        let exact = NodeSeparation::Exact;
        let mut polls = 0;
        let points = [[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0]];
        validate_distinct_points(&points, exact, X, || {
            polls += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(polls, 3);
        let duplicate = [[0.5_f64, 0.5], [1.0, 0.0], [0.5, 0.5]];
        assert_eq!(
            reason(validate_distinct_points(&duplicate, exact, X, || Ok(()))),
            InvalidInputReason::Inconsistent
        );
        let tolerant = NodeSeparation::Tolerant(Tolerance::new(0.0, 1e-9).unwrap());
        let close = [[1e6_f64, 0.0], [1e6, 1e-4]];
        assert_eq!(
            reason(validate_distinct_points(&close, tolerant, X, || Ok(()))),
            InvalidInputReason::Inconsistent
        );
        assert!(
            validate_distinct_points(
                &[[-f64::MAX / 2.0, 0.0], [f64::MAX / 2.0, 0.0]],
                exact,
                X,
                || Ok(())
            )
            .is_ok()
        );
    }

    fn node_validation<S: RealScalar>() {
        use crate::test_support::assert_error;
        use scioxide_core::SciOxideErrorKind;
        let argument = scioxide_core::argument!("nodes");
        let result = validate_nodes::<S>(&[], 1, NodeSeparation::Exact, argument);
        assert!(matches!(
            assert_error(&result, SciOxideErrorKind::InvalidInput).view(),
            SciOxideErrorView::InvalidInput {
                reason: InvalidInputReason::Empty,
                ..
            }
        ));
        assert!(validate_nodes(&[S::ZERO], 1, NodeSeparation::Exact, argument).is_ok());
        assert_error(
            &validate_nodes(&[S::ZERO], 2, NodeSeparation::Exact, argument),
            SciOxideErrorKind::InvalidInput,
        );
        for nodes in [[S::ONE, S::ZERO], [S::ZERO, S::ZERO]] {
            let result = validate_nodes(&nodes, 2, NodeSeparation::Exact, argument);
            assert!(matches!(
                assert_error(&result, SciOxideErrorKind::InvalidInput).view(),
                SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::Inconsistent,
                    ..
                }
            ));
        }
        let near = S::ZERO.next_after(S::ONE);
        assert!(validate_nodes(&[S::ZERO, near], 2, NodeSeparation::Exact, argument).is_ok());
        let tolerant = NodeSeparation::Tolerant(Tolerance::new(S::HALF, S::ZERO).unwrap());
        assert_error(
            &validate_nodes(&[S::ZERO, near], 2, tolerant, argument),
            SciOxideErrorKind::InvalidInput,
        );
        for bad in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
            let result = validate_nodes(&[S::ZERO, bad], 2, NodeSeparation::Exact, argument);
            assert!(matches!(
                assert_error(&result, SciOxideErrorKind::NonFinite).view(),
                SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
    }

    #[test]
    fn node_validator_f32() {
        node_validation::<f32>();
    }

    #[test]
    fn node_validator_f64() {
        node_validation::<f64>();
    }

    #[test]
    fn finite_validators_preserve_origin_and_accept_empty_slices() {
        use crate::test_support::assert_error;
        use scioxide_core::SciOxideErrorKind;
        assert!(validate_finite::<f64>(&[]).is_ok());
        assert!(validate_finite(&[0.0_f64, -1., f64::MAX]).is_ok());
        for origin in [
            NonFiniteOrigin::Input,
            NonFiniteOrigin::Intermediate,
            NonFiniteOrigin::Output,
        ] {
            for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let result = validate_finite_origin(&[0., bad], origin);
                assert!(
                    matches!(assert_error(&result,SciOxideErrorKind::NonFinite).view(),
                    SciOxideErrorView::NonFinite { origin: actual, .. } if actual == origin)
                );
            }
        }
        assert!(validate_finite(&[f32::NAN]).is_err());
    }
}
