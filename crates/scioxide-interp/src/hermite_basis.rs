//! The cubic Hermite basis on the unit interval.
//!
//! One coefficient table drives every cubic Hermite encoding in the crate:
//! basis values and derivatives (piecewise cubics, spherical colatitude
//! factors), basis antiderivatives (piecewise integrals) and the Hermite→power
//! transform (piecewise and bicubic coefficients). All arithmetic is checked.

use scioxide_core::{RealScalar, SciOxideError};

use crate::checked::{add, div, mul};

/// `h00, h01, h10, h11` in ascending powers of `s`: `p(s) = y0 h00 + y1 h01 +
/// H0 h10 + H1 h11` with `p(0) = y0`, `p(1) = y1`, `p'(0) = H0`, `p'(1) = H1`.
const BASIS: [[i8; 4]; 4] = [[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]];

fn integer<S: RealScalar>(value: usize) -> Result<S, SciOxideError> {
    S::try_from_usize_exact(value)
}

fn coefficient<S: RealScalar>(value: i8) -> Result<S, SciOxideError> {
    let magnitude = integer::<S>(usize::from(value.unsigned_abs()))?;
    Ok(if value < 0 { -magnitude } else { magnitude })
}

/// `order`-th `s`-derivatives `[h00, h01, h10, h11]` at `s`; zero above order 3.
///
/// Order-zero values at `s = 0` and `s = 1` are the exact unit vectors.
pub(crate) fn basis<S: RealScalar>(s: S, order: usize) -> Result<[S; 4], SciOxideError> {
    if order == 0 && (s == S::ZERO || s == S::ONE) {
        let mut exact = [S::ZERO; 4];
        exact[usize::from(s == S::ONE)] = S::ONE;
        return Ok(exact);
    }
    let mut result = [S::ZERO; 4];
    if order > 3 {
        return Ok(result);
    }
    for (slot, row) in result.iter_mut().zip(BASIS) {
        // Horner on the differentiated row: the s^k coefficient gains the
        // falling factorial k (k - 1) ... (k - order + 1).
        *slot = (order..4).rev().try_fold(S::ZERO, |value, power| {
            let falling = (power + 1 - order..=power).product::<usize>();
            add(
                mul(value, s)?,
                mul(coefficient(row[power])?, integer(falling)?)?,
            )
        })?;
    }
    Ok(result)
}

/// Antiderivatives `∫_0^s [h00, h01, h10, h11]`.
pub(crate) fn basis_integral<S: RealScalar>(s: S) -> Result<[S; 4], SciOxideError> {
    let mut result = [S::ZERO; 4];
    for (slot, row) in result.iter_mut().zip(BASIS) {
        let primitive = (0..4).rev().try_fold(S::ZERO, |value, power| {
            add(
                mul(value, s)?,
                div(coefficient(row[power])?, integer(power + 1)?)?,
            )
        })?;
        *slot = mul(primitive, s)?;
    }
    Ok(result)
}

/// Ascending power coefficients on `[0, 1]` from `[y0, Δ, H0, H1]`, where
/// `Δ = y1 - y0` and `H0`, `H1` are endpoint derivatives in `s`.
///
/// Using `h00 + h01 = 1` the value enters only through `y0` and `Δ`, so a
/// constant has zero higher coefficients exactly.
pub(crate) fn power<S: RealScalar>(data: [S; 4]) -> Result<[S; 4], SciOxideError> {
    let [y0, delta, h0, h1] = data;
    let mut result = [S::ZERO; 4];
    for (power, slot) in result.iter_mut().enumerate() {
        *slot = [(delta, 1), (h0, 2), (h1, 3)].into_iter().try_fold(
            if power == 0 { y0 } else { S::ZERO },
            |sum, (value, row)| match BASIS[row][power] {
                0 => Ok(sum),
                1 => add(sum, value),
                -1 => add(sum, -value),
                weight => add(sum, mul(coefficient(weight)?, value)?),
            },
        )?;
    }
    Ok(result)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn explicit(s: f64) -> [[f64; 4]; 4] {
        let (s2, s3) = (s * s, s * s * s);
        [
            [
                2.0 * s3 - 3.0 * s2 + 1.0,
                3.0 * s2 - 2.0 * s3,
                s3 - 2.0 * s2 + s,
                s3 - s2,
            ],
            [
                6.0 * s2 - 6.0 * s,
                6.0 * s - 6.0 * s2,
                3.0 * s2 - 4.0 * s + 1.0,
                3.0 * s2 - 2.0 * s,
            ],
            [12.0 * s - 6.0, 6.0 - 12.0 * s, 6.0 * s - 4.0, 6.0 * s - 2.0],
            [12.0, -12.0, 6.0, 6.0],
        ]
    }

    #[test]
    fn table_matches_explicit_basis_and_derivatives() {
        for s in [0.0, 0.125, 0.5, 0.75, 1.0, -0.5, 1.5] {
            for (order, expected) in explicit(s).iter().enumerate() {
                let actual = basis(s, order).unwrap();
                for (a, e) in actual.iter().zip(expected) {
                    assert!((a - e).abs() <= 1e-14 * e.abs().max(1.0), "{s} {order}");
                }
            }
            assert_eq!(basis(s, 4).unwrap(), [0.0; 4]);
        }
        assert_eq!(basis(0.0_f64, 0).unwrap(), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(basis(1.0_f32, 0).unwrap(), [0.0, 1.0, 0.0, 0.0]);
        assert!(basis(f64::MAX, 1).is_err());
    }

    #[test]
    fn integrals_differentiate_back_to_the_basis() {
        assert_eq!(basis_integral(0.0_f64).unwrap(), [0.0; 4]);
        let at_one = basis_integral(1.0_f64).unwrap();
        for (a, e) in at_one.iter().zip([0.5, 0.5, 1.0 / 12.0, -1.0 / 12.0]) {
            assert!((a - e).abs() <= 1e-15);
        }
        let (s, h) = (0.375_f64, 1e-6);
        let upper = basis_integral(s + h).unwrap();
        let lower = basis_integral(s - h).unwrap();
        let values = basis(s, 0).unwrap();
        for k in 0..4 {
            assert!(((upper[k] - lower[k]) / (2.0 * h) - values[k]).abs() <= 1e-9);
        }
    }

    #[test]
    fn power_transform_reproduces_endpoint_data_and_constants() {
        let [y0, y1, h0, h1] = [1.5_f64, -2.0, 0.25, 3.0];
        let p = power([y0, y1 - y0, h0, h1]).unwrap();
        let value = |s: f64| ((p[3] * s + p[2]) * s + p[1]) * s + p[0];
        let slope = |s: f64| (3.0 * p[3] * s + 2.0 * p[2]) * s + p[1];
        assert_eq!(value(0.0), y0);
        assert!((value(1.0) - y1).abs() <= 1e-15);
        assert_eq!(slope(0.0), h0);
        assert!((slope(1.0) - h1).abs() <= 1e-14);
        assert_eq!(
            power([7.0_f32, 0.0, 0.0, 0.0]).unwrap(),
            [7.0, 0.0, 0.0, 0.0]
        );
        assert!(power([0.0, f64::MAX, 0.0, 0.0]).is_err());
    }
}
