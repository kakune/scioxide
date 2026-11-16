//! Euclidean norms, distances and bounding boxes for scattered and local data.

use scioxide_core::{Argument, Interval, RealScalar, Scaled, ScaledAccumulator, SciOxideError};
use scioxide_poly::backend::polynomial_basis::normalized_linear_difference;

use crate::checked::{intermediate, mul};
use crate::error::empty;
use crate::radial::split_product;

/// Euclidean norm of checked components, folded with `hypot` so no component
/// is squared.
pub(crate) fn norm<S: RealScalar>(
    components: impl IntoIterator<Item = Result<S, SciOxideError>>,
) -> Result<S, SciOxideError> {
    components.into_iter().try_fold(S::ZERO, |norm, component| {
        intermediate(norm.hypot(component?))
    })
}

/// Euclidean distance between finite points.
///
/// Raw coordinate differences are used when all are finite; otherwise every
/// difference is divided by the largest coordinate magnitude first and the
/// norm is rescaled, so extreme but finite coordinates stay representable.
pub(crate) fn distance<S: RealScalar, const D: usize>(
    a: &[S; D],
    b: &[S; D],
) -> Result<S, SciOxideError> {
    if a.iter().zip(b).all(|(&x, &y)| (x - y).is_finite()) {
        return norm(a.iter().zip(b).map(|(&x, &y)| Ok(x - y)));
    }
    let scale = a.iter().chain(b).fold(S::ZERO, |m, x| m.max(x.abs()));
    let normalized = norm(
        a.iter()
            .zip(b)
            .map(|(&x, &y)| normalized_linear_difference(x, y, scale)),
    )?;
    mul(normalized, scale)
}

/// Extended Euclidean distance after per-axis normalization. This is also the
/// recovery path for radial kernels: neither differences nor normalized
/// components are projected before the complete norm is known.
pub(crate) fn extended_distance<S: RealScalar, const D: usize>(
    a: &[S; D],
    b: &[S; D],
    scales: &[S; D],
) -> Result<Scaled<S>, SciOxideError> {
    extended_distance_parts(a, b, scales)?.total()
}

/// Compensate subtraction, division, squared reduction and square root before
/// a radial exponential magnifies their residuals at the zero-tail boundary.
pub(crate) fn extended_distance_parts<S: RealScalar, const D: usize>(
    a: &[S; D],
    b: &[S; D],
    scales: &[S; D],
) -> Result<ScaledAccumulator<S>, SciOxideError> {
    let mut squares = ScaledAccumulator::ZERO;
    for ((&x, &y), &scale) in a.iter().zip(b).zip(scales) {
        let scale = Scaled::from_value(scale)?;
        let mut difference = ScaledAccumulator::from_value(x)?;
        difference.sub_scaled(Scaled::from_value(y)?)?;
        let component = difference.total()?.div(scale)?;
        let (product, residual) = split_product(component, scale)?;
        difference.sub_scaled(product)?;
        difference.sub_scaled(residual)?;
        let correction = difference.total()?.div(scale)?;
        let (square, residual) = split_product(component, component)?;
        squares.add_scaled(square)?;
        squares.add_scaled(residual)?;
        squares.add_scaled(component.mul(correction)?.shifted(1)?)?;
        squares.add_scaled(correction.square()?)?;
    }
    let root = squares.total()?.sqrt()?;
    if root.is_zero() {
        return Ok(ScaledAccumulator::ZERO);
    }
    let (square, residual) = split_product(root, root)?;
    squares.sub_scaled(square)?;
    squares.sub_scaled(residual)?;
    let correction = squares.total()?.div(root.shifted(1)?)?;
    Ok(ScaledAccumulator::from_parts(root, correction))
}

/// Closed per-axis bounding intervals of nonempty points.
pub(crate) fn bounds<S: RealScalar, const D: usize>(
    points: impl IntoIterator<Item = [S; D]>,
    argument: Argument,
) -> Result<[Interval<S>; D], SciOxideError> {
    let mut points = points.into_iter();
    let first = points
        .next()
        .ok_or_else(|| empty(argument, "samples must be nonempty"))?;
    let (lower, upper) = points.fold((first, first), |(mut lower, mut upper), point| {
        for axis in 0..D {
            lower[axis] = lower[axis].min(point[axis]);
            upper[axis] = upper[axis].max(point[axis]);
        }
        (lower, upper)
    });
    let mut result = [Interval::new(S::ZERO, S::ZERO)?; D];
    for (axis, domain) in result.iter_mut().enumerate() {
        *domain = Interval::new(lower[axis], upper[axis])?;
    }
    Ok(result)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::{InvalidInputReason, SciOxideErrorView};

    #[test]
    fn norm_and_distance_avoid_squaring_and_raw_overflow() {
        assert_eq!(norm([Ok(3.0_f64), Ok(4.0)]).unwrap(), 5.0);
        assert_eq!(norm::<f64>([]).unwrap(), 0.0);
        assert!(
            (norm([Ok(1e200_f64), Ok(1e200)]).unwrap() - 1e200 * 2.0_f64.sqrt()).abs() <= 1e186
        );
        assert_eq!(distance(&[0.0_f64, 0.0], &[3.0, 4.0]).unwrap(), 5.0);
        assert_eq!(distance(&[1.0_f32], &[1.0]).unwrap(), 0.0);
        let wide = distance(&[-f64::MAX / 2.0, 0.0], &[f64::MAX / 2.0, 0.0]).unwrap();
        assert_eq!(wide, f64::MAX);
        assert!(distance(&[-f64::MAX, 0.0], &[f64::MAX, 0.0]).is_err());
        assert!(norm([Ok(1.0_f64), Ok(f64::NAN)]).is_err());
    }

    #[test]
    fn bounds_cover_points_and_reject_empty_input() {
        let [x, y] = bounds(
            [[0.0_f64, 2.0], [-1.0, 3.0], [4.0, -2.0]],
            scioxide_core::argument!("points"),
        )
        .unwrap();
        assert_eq!(
            (x.lower(), x.upper(), y.lower(), y.upper()),
            (-1.0, 4.0, -2.0, 3.0)
        );
        let [single] = bounds([[5.0_f64]], scioxide_core::argument!("points")).unwrap();
        assert_eq!((single.lower(), single.upper()), (5.0, 5.0));
        let error = bounds::<f64, 2>([], scioxide_core::argument!("points")).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::InvalidInput {
                reason: InvalidInputReason::Empty,
                ..
            }
        ));
    }
}
