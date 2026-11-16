//! Gamma-series coefficients and shared evaluation for scalar, wide, and cancellation kernels.
pub(crate) use super::gamma_coefficients::BERNOULLI_OVER_FACTORIAL;
use super::gamma_coefficients::STIRLING;
use super::wide::Wide;
use core::convert::Infallible;
use scioxide_core::{RealScalar, SciOxideError};

/// Stirling correction from the first `terms` coefficients, in reverse Horner order.
/// Arithmetic and work accounting are statically supplied by the representation.
#[inline(always)]
pub(crate) fn stirling<T: Copy, E>(
    inverse: T,
    square: T,
    terms: usize,
    zero: T,
    mut step: impl FnMut(T, T, [f64; 3]) -> Result<T, E>,
    multiply: impl FnOnce(T, T) -> Result<T, E>,
) -> Result<T, E> {
    let sum = STIRLING[..terms]
        .iter()
        .rev()
        .try_fold(zero, |sum, &c| step(sum, square, c))?;
    multiply(inverse, sum)
}

/// Twelve terms for x >= 16; the first omitted term is below 2e-27.
#[inline(always)]
pub(crate) fn stirling_wide(x: Wide) -> Wide {
    let inverse = Wide::from(1.0) / x;
    let result = stirling(
        inverse,
        inverse * inverse,
        12,
        Wide::ZERO,
        |sum, square, [hi, lo, _]| {
            Ok::<_, Infallible>(super::polynomial::compensated_step(sum, square, hi, lo))
        },
        |a, b| Ok(a * b),
    );
    match result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

#[inline(always)]
pub(crate) fn stirling_f64(inverse: f64, square: f64) -> f64 {
    let result = stirling(
        inverse,
        square,
        12,
        0.0,
        |sum, square, [hi, _, _]| Ok::<_, Infallible>(scioxide_core::fast_mul_add(sum, square, hi)),
        |a, b| Ok(a * b),
    );
    match result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

use super::reciprocal_gamma::RECIPROCAL_GAMMA;
use super::wide;
use super::wide::Wide as D;
const RECIPROCAL_TERMS: usize = 26;
/// Even and odd parts E(u^2) + u O(u^2) of 1/Gamma(1+u), |u| <= 1/2; the two
/// leading coefficients of each keep their low parts.
pub(crate) fn reciprocal_gamma_parts(u: f64) -> (D, D) {
    let square = wide::two_product(u, u);
    let s = square.hi;
    let (even, odd) = RECIPROCAL_GAMMA[4..RECIPROCAL_TERMS].chunks(2).rev().fold(
        (0.0_f64, 0.0_f64),
        |(even, odd), pair| {
            (
                even * s + pair[0].0,
                pair.get(1).map_or(odd, |&(hi, _)| odd * s + hi),
            )
        },
    );
    let head = |tail: f64, first: (f64, f64), second: (f64, f64)| {
        let inner = wide::sum(
            D {
                hi: second.0,
                lo: second.1,
            },
            D::from(tail * s),
        );
        wide::sum(
            D {
                hi: first.0,
                lo: first.1,
            },
            wide::product(inner, square),
        )
    };
    (
        head(even, RECIPROCAL_GAMMA[0], RECIPROCAL_GAMMA[2]),
        head(odd, RECIPROCAL_GAMMA[1], RECIPROCAL_GAMMA[3]),
    )
}

/// 1/Gamma(1+u) for |u| <= 1/2.
pub(crate) fn reciprocal_gamma(u: f64) -> D {
    let (even, odd) = reciprocal_gamma_parts(u);
    wide::sum(even, wide::scale(odd, u))
}

/// Full reciprocal-gamma polynomial for |offset| <= 1/2, with one coefficient source.
/// The triple-word path keeps the checked third residuals; ordinary wide paths ignore them.
#[inline(always)]
pub(crate) fn reciprocal_gamma_full<T, E>(
    offset: T,
    zero: T,
    mut step: impl FnMut(T, &T, [f64; 3]) -> Result<T, E>,
) -> Result<T, E> {
    use super::reciprocal_gamma::RECIPROCAL_GAMMA_THIRD;
    RECIPROCAL_GAMMA
        .iter()
        .enumerate()
        .rev()
        .try_fold(zero, |sum, (i, &(hi, lo))| {
            step(
                sum,
                &offset,
                [
                    hi,
                    lo,
                    RECIPROCAL_GAMMA_THIRD.get(i).copied().unwrap_or(0.0),
                ],
            )
        })
}

/// Full wide even/odd Temme seeds. Work accounting is unchanged at u=0.
pub(crate) fn reciprocal_gamma_temme(
    u: Wide,
    square: Wide,
    work: &mut super::Work,
) -> Result<(Wide, Wide), scioxide_core::SciOxideError> {
    if u.hi == 0.0 {
        let (hi, lo) = RECIPROCAL_GAMMA[1];
        return Ok((-Wide { hi, lo }, Wide::from(1.0)));
    }
    let mut odd = Wide::ZERO;
    let mut even = Wide::ZERO;
    for (index, &(hi, lo)) in RECIPROCAL_GAMMA.iter().enumerate().rev() {
        work.try_step()?;
        let coefficient = Wide { hi, lo };
        if index % 2 == 0 {
            even = wide::product(even, square) + coefficient;
        } else {
            odd = wide::product(odd, square) - coefficient;
        }
    }
    Ok((odd, even))
}

/// Keep recurrence products and the reduced argument separate for logarithmic cancellation.
#[inline(always)]
pub(crate) fn gamma_recurrence<T: Copy, E>(
    mut argument: T,
    one: T,
    mut needs_shift: impl FnMut(T) -> bool,
    mut subtract: impl FnMut(T, T) -> Result<T, E>,
    mut multiply: impl FnMut(T, T) -> Result<T, E>,
) -> Result<(T, T), E> {
    let mut product = one;
    while needs_shift(argument) {
        argument = subtract(argument, one)?;
        product = multiply(product, argument)?;
    }
    Ok((argument, product))
}

#[inline(always)]
pub(crate) fn gamma_recurrence_wide(x: Wide) -> (Wide, Wide) {
    let result = gamma_recurrence(
        x,
        Wide::from(1.0),
        |x| x.hi > 1.5,
        |a, b| Ok::<_, Infallible>(a - b),
        |a, b| Ok(a * b),
    );
    match result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

/// Reciprocal Gamma(1+offset)-1; the constant is kept outside compensated evaluation.
#[inline(always)]
pub(crate) fn reciprocal_gamma_minus_one(offset: Wide, coefficients: &[(f64, f64)]) -> Wide {
    offset * super::polynomial::compensated(offset, coefficients, 0.0)
}

/// Log Gamma(1+order) from the existing reciprocal-gamma polynomial and
/// an exact order shift. Binary rescaling keeps the product finite.
#[inline]
pub(crate) fn gamma_log_shifted(order: Wide, w: &mut super::Work) -> Result<Wide, SciOxideError> {
    let index = order.hi.round().max(0.0);
    if index > w.remaining as f64 {
        w.complete = false;
        w.check()?;
    }
    let n = index.try_to_usize_exact()?;
    let u = order - Wide::from(index);
    let reciprocal = reciprocal_gamma_full(u, Wide::ZERO, |sum, u, [hi, lo, _]| {
        w.try_step()?;
        Ok::<_, SciOxideError>(sum * *u + Wide { hi, lo })
    })?;
    let mut product = Wide::from(1.0);
    let mut exponent = 0.0;
    for k in 1..=n {
        w.try_step()?;
        if product.hi > 1e150 {
            product = product.mul_f64(f64::from_bits(511_u64 << 52));
            exponent += 512.0;
        }
        product = product * (u + Wide::from(k as f64));
    }
    w.try_step()?;
    Ok((product / reciprocal).ln() + super::constants::LN_2.mul_f64(exponent))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_gamma_even_odd_and_full_precision_agree_at_reduction_edges() {
        for u in [-0.5, -0.25, 0.0, 0.25, 0.5] {
            let full = reciprocal_gamma_full(Wide::from(u), Wide::ZERO, |sum, u, [hi, lo, _]| {
                Ok::<_, Infallible>(sum * *u + Wide { hi, lo })
            })
            .unwrap();
            let mut work = super::super::Work::new(1000);
            let offset = Wide::from(u);
            let (odd, even) = reciprocal_gamma_temme(offset, offset * offset, &mut work).unwrap();
            let temme = even - offset * odd;
            assert!((temme.value() - full.value()).abs() <= 2.0 * f64::EPSILON * full.hi.abs());
            assert!(
                (reciprocal_gamma(u).value() - full.value()).abs()
                    <= 2.0 * f64::EPSILON * full.hi.abs()
            );
            assert_eq!(work.used, if u == 0.0 { 0 } else { 41 });
        }
    }

    #[test]
    fn stirling_uses_the_requested_precision_and_term_count() {
        let inverse = 1.0 / 32.0;
        let expected = inverse * (1.0 / 12.0 - inverse * inverse / 360.0);
        let mut terms = 0;
        let value = stirling(
            inverse,
            inverse * inverse,
            2,
            0.0,
            |sum, square, [hi, _, _]| {
                terms += 1;
                Ok::<_, Infallible>(sum * square + hi)
            },
            |a, b| Ok(a * b),
        )
        .unwrap();
        assert!((value - expected).abs() < 1e-18);
        assert_eq!(terms, 2);
        assert!(
            (stirling_f64(inverse, inverse * inverse) - stirling_wide(Wide::from(32.0)).value())
                .abs()
                < 1e-18
        );
    }
}
