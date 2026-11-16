//! Checked scalar arithmetic that reports where a non-finite value arose.

use scioxide_core::{
    NonFiniteOrigin, NumericScalar, NumericalIssue, RealScalar, SciOxideError, checked_midpoint,
};
use scioxide_poly::backend::polynomial_basis::normalized_linear_difference;

pub(crate) fn finite<C: NumericScalar>(
    value: C,
    origin: NonFiniteOrigin,
    detail: &'static str,
) -> Result<C, SciOxideError> {
    if value.has_finite_components() {
        Ok(value)
    } else {
        Err(SciOxideError::non_finite(origin, detail))
    }
}

/// Reject a non-finite real or complex intermediate.
pub(crate) fn intermediate<C: NumericScalar>(value: C) -> Result<C, SciOxideError> {
    finite(
        value,
        NonFiniteOrigin::Intermediate,
        "intermediate arithmetic produced a non-finite value",
    )
}

/// Reject a non-finite result before it is written to the output.
pub(crate) fn output<C: NumericScalar>(value: C) -> Result<C, SciOxideError> {
    finite(
        value,
        NonFiniteOrigin::Output,
        "interpolation result is non-finite",
    )
}

pub(crate) fn add<S: RealScalar>(left: S, right: S) -> Result<S, SciOxideError> {
    intermediate(left + right)
}

pub(crate) fn sub<S: RealScalar>(left: S, right: S) -> Result<S, SciOxideError> {
    intermediate(left - right)
}

pub(crate) fn mul<S: RealScalar>(left: S, right: S) -> Result<S, SciOxideError> {
    intermediate(left * right)
}

pub(crate) fn div<S: RealScalar>(numerator: S, denominator: S) -> Result<S, SciOxideError> {
    intermediate(numerator / denominator)
}

/// Centre and half-width scale of `[lower, upper]`, without forming the raw
/// width: the scale is the larger one-sided distance from the correctly
/// rounded midpoint, so `(x - centre) / scale` maps the interval onto
/// (approximately) `[-1, 1]`.
pub(crate) fn interval_normalization<S: RealScalar>(
    lower: S,
    upper: S,
) -> Result<(S, S), SciOxideError> {
    let center = checked_midpoint(lower, upper)?;
    let scale = normalized_linear_difference(center, lower, S::ONE)?
        .abs()
        .max(normalized_linear_difference(upper, center, S::ONE)?.abs());
    if scale == S::ZERO {
        return Err(crate::error::instability(
            NumericalIssue::Breakdown,
            "normalization scale is zero",
        ));
    }
    Ok((center, scale))
}

pub(crate) fn interval_coordinate<S: RealScalar>(
    value: S,
    lower: S,
    upper: S,
) -> Result<S, SciOxideError> {
    // Keep the numerator as a difference from the lower endpoint. Subtracting
    // separately centered coordinates loses small offsets on wide intervals.
    scioxide_poly::backend::polynomial_basis::linear_difference_ratio(value, lower, upper, lower)
}

/// Whether a value is small or large enough for the fast native-arithmetic kernels.
/// A factor of 64 leaves headroom for cubic transforms and weighted sums.
/// Nonzero data outside this band use the rescaled arithmetic that avoids
/// overflow and underflow.
pub(crate) fn ordinary<S: RealScalar>(value: S) -> bool {
    let magnitude = value.abs();
    let margin = S::ONE.scalbn(6);
    magnitude == S::ZERO
        || (magnitude >= S::MIN_POSITIVE.sqrt() * margin && magnitude <= S::MAX.sqrt() / margin)
}

pub(crate) fn ordinary_nodes<S: RealScalar>(nodes: &[S]) -> bool {
    nodes.iter().all(|&node| ordinary(node))
        && nodes
            .windows(2)
            .all(|pair| pair[1] > pair[0] && ordinary(pair[1] - pair[0]))
}

/// For an accepted interval and an interior query the quotient is in [0, 1].
#[inline]
pub(crate) fn coordinate<S: RealScalar>(
    value: S,
    lower: S,
    upper: S,
    native: bool,
) -> Result<S, SciOxideError> {
    if native && value >= lower && value <= upper {
        Ok((value - lower) / (upper - lower))
    } else {
        interval_coordinate(value, lower, upper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::assert_error;

    #[test]
    fn admission_keeps_extreme_nodes_on_the_original_path() {
        assert!(ordinary_nodes(&[-1.0_f64, 0.0, 0.5, 2.0]));
        assert!(ordinary_nodes(&[-1.0_f32, 0.0, 0.5, 2.0]));
        for nodes in [
            [-f64::MAX, f64::MAX],
            [0.0, f64::from_bits(1)],
            [0.0, 1e-200],
            [0.0, 1e200],
            [1.0, 1.0],
        ] {
            assert!(!ordinary_nodes(&nodes));
        }
        assert!(!ordinary(f32::from_bits(1)));
        assert!(!ordinary(f32::MAX));
        assert!(!ordinary(f64::NAN));
        let margin = 64.0;
        assert!(ordinary(f64::MIN_POSITIVE.sqrt() * margin));
        assert!(ordinary(f64::MAX.sqrt() / margin));
    }

    #[test]
    fn non_finite_provenance_is_not_relabelled() {
        for origin in [
            NonFiniteOrigin::Input,
            NonFiniteOrigin::Callback,
            NonFiniteOrigin::Intermediate,
            NonFiniteOrigin::Output,
        ] {
            assert!(matches!(assert_error(
    &finite(f64::NAN, origin, "non-finite"), scioxide_core::SciOxideErrorKind::NonFinite
).view(), scioxide_core::SciOxideErrorView::NonFinite { origin: actual, .. } if actual == origin));
        }
    }
}
