//! Guard native finite-volume expressions before recomputing unsafe intermediates.
use scioxide_core::{RealScalar, Scaled, SciOxideError};

#[inline]
pub(super) fn difference_quotient<R: RealScalar>(
    a: R,
    b: R,
    divisor: R,
) -> Result<R, SciOxideError> {
    let difference = a - b;
    let quotient = difference / divisor;
    if difference.is_finite() && (quotient.is_normal() || difference == R::ZERO) {
        return Ok(quotient);
    }
    Scaled::from_value(a)?
        .sub(Scaled::from_value(b)?)?
        .div(Scaled::from_value(divisor)?)?
        .round()
}

#[inline]
pub(super) fn trace<R: RealScalar, const D: usize, const N: usize>(
    average: &[R; N],
    gradient: &[[R; D]; N],
    offset: [R; D],
    scale: R,
) -> Result<[R; N], SciOxideError> {
    let mut state = *average;
    for j in 0..N {
        let mut jump = R::ZERO;
        let mut safe = true;
        for (a, b) in gradient[j].iter().zip(offset) {
            let product = *a * b;
            safe &= product.is_normal() || *a == R::ZERO || b == R::ZERO;
            jump += product;
        }
        let change = scale * jump;
        let value = average[j] + change;
        if safe && jump.is_finite() && change.is_finite() && value.is_finite() {
            state[j] = value;
        } else {
            state[j] = trace_extended(average[j], &gradient[j], offset, scale)?;
        }
    }
    Ok(state)
}

#[cold]
fn trace_extended<R: RealScalar, const D: usize>(
    average: R,
    gradient: &[R; D],
    offset: [R; D],
    scale: R,
) -> Result<R, SciOxideError> {
    let mut jump = Scaled::ZERO;
    for (&a, b) in gradient.iter().zip(offset) {
        jump = jump.add(Scaled::from_value(a)?.mul(Scaled::from_value(b)?)?)?;
    }
    Scaled::from_value(average)?
        .add(jump.mul(Scaled::from_value(scale)?)?)?
        .round()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reconstructs_from_original_operands_at_extreme_scales() {
        let slope = difference_quotient(f64::MAX, -f64::MAX, 2.).unwrap();
        assert!((slope / f64::MAX - 1.).abs() < 1e-15);
        let state = trace(&[0.], &[[f64::MAX]], [2.], 0.25).unwrap();
        assert!((state[0] / (f64::MAX * 0.5) - 1.).abs() < 1e-15);
        let tiny = 2_f64.powi(-600);
        let state = trace(&[0.], &[[tiny]], [tiny], 1. / tiny).unwrap();
        assert!((state[0] / tiny - 1.).abs() < 1e-15);
        assert!(difference_quotient(f64::MAX, -f64::MAX, 0.5).is_err());
    }
}
