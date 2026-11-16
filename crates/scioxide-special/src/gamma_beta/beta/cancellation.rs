//! Log-beta near zero needs more than 106 bits when O(1) logarithms cancel.
//! Three components, accumulated with core's exact products, are confined to
//! this rare region. No general-purpose extended scalar is exposed.

use crate::support::wide::Wide;
use scioxide_core::{RangeAccumulator, RealScalar, SciOxideError};

type Parts = [f64; 3];
const ONE: Parts = [1.0, 0.0, 0.0];

// Keep extraction in the two arithmetic callers: otherwise every three-word
// update crosses an additional Result boundary in the cancellation series.
#[inline(always)]
fn extract(mut sum: RangeAccumulator<f64>) -> Result<Parts, SciOxideError> {
    let mut parts = [0.0; 3];
    for part in &mut parts[..2] {
        *part = sum.finish()?;
        sum.add(-*part)?;
    }
    parts[2] = sum.finish()?;
    Ok(parts)
}

fn add(a: Parts, b: Parts) -> Result<Parts, SciOxideError> {
    let mut sum = RangeAccumulator::new();
    for x in a.into_iter().chain(b) {
        sum.add(x)?;
    }
    extract(sum)
}

fn neg(a: Parts) -> Parts {
    a.map(|x| -x)
}

fn mul(a: Parts, b: Parts) -> Result<Parts, SciOxideError> {
    let mut sum = RangeAccumulator::new();
    for x in a {
        for y in b {
            sum.add_product(x, y)?;
        }
    }
    extract(sum)
}

fn div(a: Parts, b: Parts) -> Result<Parts, SciOxideError> {
    let mut residual = a;
    let mut result = [0.0; 3];
    for _ in 0..3 {
        let next = [residual[0] / b[0], 0.0, 0.0];
        result = add(result, next)?;
        residual = add(residual, neg(mul(b, next)?))?;
    }
    Ok(result)
}

fn log(x: Parts) -> Result<Parts, SciOxideError> {
    let (_, mut exponent) = crate::support::wide::finite_parts(x[0]);
    let mut reduced = x.map(|v| v.scalbn(-exponent));
    if reduced[0] > core::f64::consts::SQRT_2 {
        reduced = reduced.map(|v| v * 0.5);
        exponent += 1;
    }
    let z = div(add(reduced, neg(ONE))?, add(reduced, ONE)?)?;
    let square = mul(z, z)?;
    let mut term = z;
    let mut sum = z;
    // |z| < 0.172; the omitted term is below 2e-51.
    for k in 1..33 {
        term = mul(term, square)?;
        sum = add(sum, div(term, [(2 * k + 1) as f64, 0.0, 0.0])?)?;
    }
    add(
        mul(sum, [2.0, 0.0, 0.0])?,
        mul(
            [
                crate::support::constants::LN_2.hi,
                crate::support::constants::LN_2.lo,
                5.707708438416212e-34,
            ],
            [f64::from(exponent), 0.0, 0.0],
        )?,
    )
}

fn log_gamma(mut a: Parts) -> Result<Parts, SciOxideError> {
    let (reduced, product) = crate::support::gamma_series::gamma_recurrence(
        a,
        ONE,
        |x| x[0] > 1.5,
        |x, one| add(x, neg(one)),
        mul,
    )?;
    a = reduced;
    // Third residuals of the existing reciprocal-gamma coefficients. Terms
    // beyond degree 15 contribute less than 1e-45 on |a-1| <= 1/2.
    let offset = if a[0] < 0.5 { a } else { add(a, neg(ONE))? };
    let reciprocal = crate::support::gamma_series::reciprocal_gamma_full(
        offset,
        [0.0; 3],
        |sum, offset, coefficient| add(mul(sum, *offset)?, coefficient),
    )?;
    let result = add(log(product)?, neg(log(reciprocal)?))?;
    if a[0] < 0.5 {
        add(result, neg(log(a)?))
    } else {
        Ok(result)
    }
}

fn stirling(x: Parts) -> Result<Parts, SciOxideError> {
    let inverse = div(ONE, x)?;
    let square = mul(inverse, inverse)?;
    crate::support::gamma_series::stirling(
        inverse,
        square,
        16,
        [0.0; 3],
        |sum, square, coefficient| add(mul(sum, square)?, coefficient),
        mul,
    )
}

/// Called only for severe cancellation in ln B(a,b), with a <= b.
pub(super) fn ln_beta(a: f64, b: f64) -> Result<Wide, SciOxideError> {
    let (a, b) = ([a, 0.0, 0.0], [b, 0.0, 0.0]);
    let sum = add(a, b)?;
    if b[0] < 32.0 {
        let value = add(add(log_gamma(a)?, log_gamma(b)?)?, neg(log_gamma(sum)?))?;
        return Ok(Wide {
            hi: value[0],
            lo: value[1] + value[2],
        });
    }
    let r = div(a, b)?;
    let mut series = [0.0; 3];
    // (r - log1p(r))/r², |r| <= 1/32. Keeping the leading
    // cancellation analytic also handles subnormal a/b without amplification.
    for k in (2..34).rev() {
        series = add(mul(series, neg(r))?, div(ONE, [k as f64, 0.0, 0.0])?)?;
    }
    let half_inverse = div([0.5, 0.0, 0.0], b)?;
    let difference = mul(
        a,
        add(
            half_inverse,
            mul(add(ONE, neg(half_inverse))?, mul(r, series)?)?,
        )?,
    )?;
    let value = add(
        add(log_gamma(a)?, neg(mul(a, log(sum)?)?))?,
        add(difference, add(stirling(b)?, neg(stirling(sum)?))?)?,
    )?;
    Ok(Wide {
        hi: value[0],
        lo: value[1] + value[2],
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn double_word_beta_ratio_cannot_resolve_the_unit_shape_zero() {
        // ln B(1-d,1+d) = zeta(2)*d² + O(d⁴); the omitted term is
        // below 0.01 output ULP. The ratio rounds around one before ln.
        let d = 2.0_f64.powi(-40);
        let expected = 1.644_934_066_848_226_4 * d * d;
        let bound = 4.0 * (expected.next_up() - expected);
        let double =
            crate::support::gamma_real::beta_moderate(Wide::from(1.0 - d), Wide::from(1.0 + d))
                .ln()
                .value();
        assert!((double - expected).abs() > bound);
        assert!((ln_beta(1.0 - d, 1.0 + d).unwrap().value() - expected).abs() <= bound);
    }
}
