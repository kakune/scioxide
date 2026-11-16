//! Nested Chebyshev interpolation by the defining discrete cosine transform.
//! Endpoint samples and endpoint coefficients carry half weights (DCT-I).
//! The fixed 12/24-degree transforms require O(n²) arithmetic and stack storage.

use scioxide_core::RealScalar;

/// Accumulate one cosine projection with Neumaier compensation.
fn project<S: RealScalar>(values: &[S], row: &[S; 25], stride: usize) -> S {
    let mut sum = S::ZERO;
    let mut correction = S::ZERO;
    for j in (0..25).step_by(stride) {
        let weight = if j == 0 || j == 24 { S::HALF } else { S::ONE };
        let term = (values[j] * weight) * row[j];
        let next = sum + term;
        correction += if sum.abs() >= term.abs() {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        sum = next;
    }
    sum + correction
}

/// The degree-12 projection uses only the even-indexed samples.
#[inline(always)]
pub(super) fn coarse_coefficients<S: RealScalar>(values: &[S], cosines: &[[S; 25]; 25]) -> [S; 13] {
    core::array::from_fn(|k| {
        let divisor = S::from_f64_ieee(if k == 0 || k == 12 { 12.0 } else { 6.0 });
        project(values, &cosines[k], 2) / divisor
    })
}

/// Form both projections with identical coarse arithmetic in either entry point.
#[inline(always)]
pub(super) fn coefficients<S: RealScalar>(
    values: &[S],
    cosines: &[[S; 25]; 25],
) -> ([S; 13], [S; 25]) {
    let coarse = coarse_coefficients(values, cosines);
    let fine = core::array::from_fn(|k| {
        let divisor = S::from_f64_ieee(if k == 0 || k == 24 { 24.0 } else { 12.0 });
        project(values, &cosines[k], 1) / divisor
    });
    (coarse, fine)
}

type Coefficients<S> = (
    [scioxide_core::PromotingProduct<S>; 13],
    [scioxide_core::PromotingProduct<S>; 25],
);

/// Direct DCT-I in extended range. Normalizes before rounding and keeps
/// coefficients beyond the range of `S` when their moment products can still fit.
pub(super) fn extended_coefficients<S: RealScalar>(
    values: &[S],
    cosines: &[[S; 25]; 25],
) -> Result<Coefficients<S>, scioxide_core::SciOxideError> {
    use scioxide_core::{PromotingProduct, Scaled, ScaledAccumulator};
    let mut coarse = [PromotingProduct::ONE; 13];
    let mut fine = [PromotingProduct::ONE; 25];
    for (k, cosine_row) in cosines.iter().enumerate() {
        let mut high = ScaledAccumulator::ZERO;
        let mut low = ScaledAccumulator::ZERO;
        for (j, &value) in values.iter().enumerate() {
            let value =
                Scaled::from_value(value)?.shifted(if j == 0 || j == 24 { -1 } else { 0 })?;
            let cosine = Scaled::from_value(cosine_row[j])?;
            high.add_product(value, cosine)?;
            if k <= 12 && j % 2 == 0 {
                low.add_product(value, cosine)?;
            }
        }
        fine[k] = PromotingProduct::from_scaled(high.total()?.div(Scaled::from_value(
            S::from_f64_ieee(if k == 0 || k == 24 { 24.0 } else { 12.0 }),
        )?)?);
        if k <= 12 {
            coarse[k] = PromotingProduct::from_scaled(low.total()?.div(Scaled::from_value(
                S::from_f64_ieee(if k == 0 || k == 12 { 12.0 } else { 6.0 }),
            )?)?);
        }
    }
    Ok((coarse, fine))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn basis<S: RealScalar>() {
        let cosines = crate::weighted::moments::cosines::<S>().unwrap();
        for (degree, values) in cosines.iter().enumerate() {
            let (coarse, fine) = coefficients(values, &cosines);
            let allowance = S::try_from_usize_exact(128).unwrap() * S::EPSILON;
            for (k, &value) in fine.iter().enumerate() {
                let expected = if k == degree { S::ONE } else { S::ZERO };
                assert!(
                    (value - expected).abs() <= allowance,
                    "degree {degree}, coefficient {k}: {value:?}"
                );
            }
            let alias = if degree <= 12 { degree } else { 24 - degree };
            for (k, &value) in coarse.iter().enumerate() {
                let expected = if k == alias { S::ONE } else { S::ZERO };
                assert!(
                    (value - expected).abs() <= allowance,
                    "aliased degree {degree}, coefficient {k}: {value:?}"
                );
            }
        }
    }

    #[test]
    fn polynomial_basis_and_nested_aliasing_in_both_precisions() {
        basis::<f64>();
        basis::<f32>();
    }

    #[test]
    fn coarse_coefficients_repeat_the_full_transform_bit_for_bit() {
        fn check<S: RealScalar>() {
            let cosines = crate::weighted::moments::cosines::<S>().unwrap();
            let mut values: [S; 25] =
                core::array::from_fn(|j| (S::try_from_usize_exact(j + 1).unwrap()).ln() - S::HALF);
            for _ in 0..2 {
                let coarse = coarse_coefficients(&values, &cosines);
                let (full, _) = coefficients(&values, &cosines);
                for (a, b) in coarse.iter().zip(&full) {
                    assert_eq!(a.into_f64().to_bits(), b.into_f64().to_bits());
                }
                // Odd-indexed values never reach the degree-12 coefficients.
                for value in values.iter_mut().skip(1).step_by(2) {
                    *value = S::ZERO;
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }
}
