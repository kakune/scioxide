//! Centered cardinal B-splines on integer knots.

use scioxide_core::{NonFiniteOrigin, RealScalar, SciOxideError};

use super::basis::cardinal_basis;

/// Evaluates the centered cardinal B-spline `Bₚ(x)` of degree `p` from zero to three.
///
/// `Bₚ` is the B-spline of degree `p` on the uniform knots
/// `−(p + 1)/2, −(p − 1)/2, …, (p + 1)/2`. Its support is the half-open interval
/// `[−(p + 1)/2, (p + 1)/2)`, it is zero outside, and it integrates to one.
/// For example `B₁(x) = 1 − |x|` on `[−1, 1)` and `B₃(0) = 2/3`. Each degree
/// is evaluated from its closed-form polynomial pieces in O(1) operations.
///
/// # Errors
///
/// - `InvalidInput` (`OutOfRange`) if `degree` is greater than three.
/// - `NonFinite` if `x` is NaN or infinite.
///
/// # Examples
///
/// ```
/// use scioxide_interp::cardinal_bspline;
/// assert_eq!(cardinal_bspline(1, 0.25_f64)?, 0.75);
/// assert!((cardinal_bspline(3, 0.0_f64)? - 2.0 / 3.0).abs() < 1e-15);
/// assert_eq!(cardinal_bspline(2, 1.5_f64)?, 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// C. de Boor, *A Practical Guide to Splines*, rev. ed., Springer, 2001, chapter IX.
pub fn cardinal_bspline<S: RealScalar>(degree: usize, x: S) -> Result<S, SciOxideError> {
    if degree > 3 {
        return Err(crate::error::out_of_range(
            scioxide_core::argument!("degree"),
            "supported degrees are zero through three",
        ));
    }
    if !x.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "cardinal B-spline coordinate must be finite",
        ));
    }
    let half = S::try_from_usize_exact(degree + 1)? * S::HALF;
    if x < -half || x >= half {
        return Ok(S::ZERO);
    }
    Ok(cardinal_basis(degree, x))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn closed_forms_and_errors() {
        assert_eq!(cardinal_bspline(0, -0.5_f64).unwrap(), 1.0);
        assert_eq!(cardinal_bspline(0, 0.5_f64).unwrap(), 0.0);
        assert!((cardinal_bspline(2, 0.0_f64).unwrap() - 0.75).abs() < 1e-15);
        assert!((cardinal_bspline(3, 1.0_f32).unwrap() - 1.0 / 6.0).abs() < 1e-7);
        assert!(cardinal_bspline(4, 0.0_f64).is_err());
        assert!(cardinal_bspline(1, f64::NAN).is_err());
    }
}
