//! Bounded expansion arithmetic for cancellation in the real Kummer series.
//!
//! Only addition and multiplication/division by exact two-word parameters are
//! needed. Core's exact accumulator keeps every product and subtraction;
//! successive rounded residuals give a nonoverlapping expansion. The discarded
//! residual supplies an absolute error, including for division. There is no
//! heap allocation and no precision-dependent public scalar contract.
use crate::support::constants::LN_2;
use crate::support::wide::Wide as D;
use crate::support::{HYPERGEOMETRIC_ERRORS as ERRORS, Work};
use scioxide_core::{RangeAccumulator, RealScalar, SciOxideError};

#[derive(Clone, Copy)]
struct Expansion<const N: usize>([f64; N]);

impl<const N: usize> Expansion<N> {
    fn one() -> Self {
        let mut parts = [0.0; N];
        parts[0] = 1.0;
        Self(parts)
    }

    fn accumulator(self) -> Result<RangeAccumulator<f64>, SciOxideError> {
        let mut sum = RangeAccumulator::new();
        for part in self.0 {
            sum.add(part)?;
        }
        Ok(sum)
    }

    fn extract(mut sum: RangeAccumulator<f64>) -> Result<(Self, f64), SciOxideError> {
        let mut parts = [0.0; N];
        for part in &mut parts {
            *part = sum.finish()?;
            sum.add(-*part)?;
        }
        let discarded = sum.finish()?.abs();
        let error = if discarded == 0.0 && sum.finish_scaled::<f64>()?.is_zero() {
            0.0
        } else {
            discarded * (1.0 + f64::EPSILON) + f64::from_bits(1)
        };
        Ok((Self(parts), error))
    }

    fn add(self, other: Self) -> Result<(Self, f64), SciOxideError> {
        let mut sum = self.accumulator()?;
        for part in other.0 {
            sum.add(part)?;
        }
        Self::extract(sum)
    }

    fn multiply(self, factor: D) -> Result<(Self, f64), SciOxideError> {
        let mut sum = RangeAccumulator::new();
        for part in self.0 {
            sum.add_product(part, factor.hi)?;
            sum.add_product(part, factor.lo)?;
        }
        Self::extract(sum)
    }

    fn divide(self, divisor: D) -> Result<(Self, f64), SciOxideError> {
        let mut residual = self.accumulator()?;
        let mut quotient = [0.0; N];
        for part in &mut quotient {
            *part = residual.finish_quotient(divisor.hi)?;
            residual.subtract_product(*part, divisor.hi)?;
            residual.subtract_product(*part, divisor.lo)?;
        }
        let error =
            residual.finish_quotient(divisor.hi)?.abs() / (1.0 - (divisor.lo / divisor.hi).abs());
        let error = if error == 0.0 && residual.finish_scaled::<f64>()?.is_zero() {
            0.0
        } else {
            error * (1.0 + 4.0 * f64::EPSILON) + f64::from_bits(1)
        };
        Ok((Self(quotient), error))
    }

    fn scale(self, exponent: i32) -> Self {
        Self(self.0.map(|part| part.scalbn(exponent)))
    }

    fn leading(self) -> f64 {
        self.0[0]
    }

    fn wide(self) -> D {
        D {
            hi: self.0[0],
            lo: self.0[1],
        }
    }
}

/// Mantissa, exact-power logarithmic scale, and its absolute error. `None`
/// means the selected expansion cannot resolve the sum and needs more words.
fn series<const N: usize>(
    a: D,
    b: D,
    z: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<(D, D, f64)>, SciOxideError> {
    let mut term = Expansion::<N>::one();
    let mut sum = term;
    let mut term_error = 0.0;
    let mut error = 0.0;
    let mut magnitude = 1.0;
    let mut exponent = 0.0;
    let end = if a.lo == 0.0 && a.hi <= 0.0 && a.hi.fract() == 0.0 {
        -a.hi
    } else {
        f64::INFINITY
    };
    for n in 0usize.. {
        let k = n as f64;
        if k == end {
            return Ok((error <= f64::EPSILON * sum.leading().abs()).then_some((
                sum.wide(),
                LN_2.mul_f64(exponent),
                error,
            )));
        }
        // A work unit is a recurrence operation, independent of the private
        // number of words. Every retry shares the caller's remaining budget.
        w.steps(4)?;
        for factor in [a + D::from(k), D::from(z)] {
            let (next, rounding) = term.multiply(factor)?;
            term = next;
            term_error = term_error * (factor.hi.abs() + factor.lo.abs()) + rounding;
        }
        for divisor in [b + D::from(k), D::from(k + 1.0)] {
            let (next, rounding) = term.divide(divisor)?;
            term = next;
            term_error = term_error / (divisor.hi.abs() - divisor.lo.abs()) + rounding;
        }
        let (next, rounding) = sum.add(term)?;
        sum = next;
        error += term_error + rounding;
        magnitude += term.leading().abs();
        if !magnitude.is_finite() || !error.is_finite() {
            return Err(ERRORS.unstable());
        }
        if magnitude > 1e100 {
            term = term.scale(-256);
            sum = sum.scale(-256);
            term_error = term_error.scalbn(-256);
            error = error.scalbn(-256);
            magnitude = magnitude.scalbn(-256);
            exponent += 256.0;
        }
        let next_k = k + 1.0;
        if next_k + b.hi <= 0.0 || next_k + a.hi <= 0.0 {
            continue;
        }
        let ratio = z.abs() / (next_k + 1.0)
            * ((next_k + a.hi.abs() + a.lo.abs()) / (next_k + b.hi - b.lo.abs())).max(1.0);
        if !(0.0..1.0).contains(&ratio) {
            continue;
        }
        let tail = term.leading().abs() * ratio / (1.0 - ratio);
        let target = f64::EPSILON / 32.0 * sum.leading().abs();
        if tail <= target {
            let bound = error * (1.0 + 16.0 * (k + 1.0) * f64::EPSILON) + 2.0 * tail;
            return Ok((bound <= f64::EPSILON * sum.leading().abs()).then_some((
                sum.wide(),
                LN_2.mul_f64(exponent),
                bound,
            )));
        }
    }
    Err(ERRORS.unstable())
}

/// Resolve an exact-input Kummer series with a bounded adaptive expansion.
pub(super) fn kummer(
    a: D,
    b: D,
    z: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<(D, D, f64), SciOxideError> {
    if let Some(result) = series::<4>(a, b, z, w)? {
        return Ok(result);
    }
    if let Some(result) = series::<8>(a, b, z, w)? {
        return Ok(result);
    }
    series::<16>(a, b, z, w)?.ok_or_else(|| ERRORS.unstable())
}
