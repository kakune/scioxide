//! Positive Hurwitz summation, with one remainder test for all precisions.
//!
//! Powers may be normalized by q^-s or evaluated directly when integer powers
//! cannot overflow. This changes neither the Euler--Maclaurin recurrence nor its
//! first-omitted-term bound (s > 1, q > 0). Callers choose a rounding budget;
//! the driver extends the prefix if the asymptotic corrections start growing.

use super::{Work, gamma_series::BERNOULLI_OVER_FACTORIAL, wide::Wide};
use core::ops::{Add, Div, Mul, Sub};
use scioxide_core::{RealScalar, SciOxideError};

pub(crate) trait Arithmetic:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    fn number(value: f64) -> Result<Self, SciOxideError>;
    fn coefficient(value: Wide) -> Result<Self, SciOxideError>;
    fn magnitude(self) -> f64;
    fn accumulate(sum: Self, correction: Self, term: Self) -> (Self, Self);
    fn total(sum: Self, correction: Self) -> Self;
    fn negligible_tail(s: Self, x: Self, base: Self, direct: Self, relative: f64) -> bool;
    fn correction_bound(derivative: Self, coefficient: Wide) -> Result<f64, SciOxideError>;
    fn tail_start(
        s: Self,
        x: Self,
        base: Self,
        prefix: (Self, Self),
    ) -> Result<(Self, Self), SciOxideError>;
    fn derivative(s: Self, base: Self, inverse: Self) -> Self;
    fn advance(
        s: Self,
        index: usize,
        derivative: Self,
        inverse: Self,
    ) -> Result<Self, SciOxideError>;
}

impl<S: RealScalar> Arithmetic for S {
    #[inline(always)]
    fn number(value: f64) -> Result<Self, SciOxideError> {
        S::try_from_f64_lossy(value)
    }
    #[inline(always)]
    fn coefficient(value: Wide) -> Result<Self, SciOxideError> {
        Self::number(value.hi)
    }
    #[inline(always)]
    fn magnitude(self) -> f64 {
        self.abs().into_f64()
    }
    #[inline(always)]
    fn accumulate(sum: Self, correction: Self, term: Self) -> (Self, Self) {
        let next = sum + term;
        let residual = if sum.abs() >= term.abs() {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        (next, correction + residual)
    }
    #[inline(always)]
    fn total(sum: Self, correction: Self) -> Self {
        sum + correction
    }
    #[inline(always)]
    fn negligible_tail(s: Self, x: Self, base: Self, direct: Self, relative: f64) -> bool {
        (base + base * x / (s - S::ONE)).magnitude() <= relative * direct.magnitude()
    }
    #[inline(always)]
    fn correction_bound(derivative: Self, coefficient: Wide) -> Result<f64, SciOxideError> {
        Ok((Self::coefficient(coefficient)? * derivative).magnitude())
    }
    #[inline(always)]
    fn tail_start(
        s: Self,
        x: Self,
        base: Self,
        prefix: (Self, Self),
    ) -> Result<(Self, Self), SciOxideError> {
        let (sum, correction) = Self::accumulate(prefix.0, prefix.1, base * x / (s - S::ONE));
        Ok(Self::accumulate(sum, correction, S::HALF * base))
    }
    #[inline(always)]
    fn derivative(s: Self, base: Self, inverse: Self) -> Self {
        s * base * inverse
    }
    #[inline(always)]
    fn advance(
        s: Self,
        index: usize,
        derivative: Self,
        inverse: Self,
    ) -> Result<Self, SciOxideError> {
        let shift = Self::number((2 * index + 1) as f64)?;
        Ok(derivative * ((s + shift) * inverse) * ((s + shift + S::ONE) * inverse))
    }
}

impl Arithmetic for Wide {
    #[inline(always)]
    fn number(value: f64) -> Result<Self, SciOxideError> {
        Ok(Self::from(value))
    }
    #[inline(always)]
    fn coefficient(value: Wide) -> Result<Self, SciOxideError> {
        Ok(value)
    }
    #[inline(always)]
    fn magnitude(self) -> f64 {
        self.hi.abs()
    }
    #[inline(always)]
    fn accumulate(sum: Self, correction: Self, term: Self) -> (Self, Self) {
        (sum + term, correction)
    }
    #[inline(always)]
    fn total(sum: Self, _correction: Self) -> Self {
        sum
    }
    #[inline(always)]
    fn negligible_tail(_s: Self, _x: Self, _base: Self, _direct: Self, _relative: f64) -> bool {
        // Twofold callers always retain the tail. A redundant wide division
        // just to discard it costs more than the first correction.
        false
    }
    #[inline(always)]
    fn correction_bound(derivative: Self, coefficient: Wide) -> Result<f64, SciOxideError> {
        Ok((derivative.hi.abs() + derivative.lo.abs())
            * (coefficient.hi.abs() + coefficient.lo.abs()))
    }
    #[inline(always)]
    fn tail_start(
        s: Self,
        x: Self,
        base: Self,
        prefix: (Self, Self),
    ) -> Result<(Self, Self), SciOxideError> {
        let integral = if s.lo == 0.0 && s.hi == s.hi.floor() {
            x.div_f64(s.hi - 1.0)
        } else {
            x / (s - Self::from(1.0))
        };
        Ok((prefix.0 + base * (integral + Self::from(0.5)), prefix.1))
    }
    #[inline(always)]
    fn derivative(s: Self, base: Self, inverse: Self) -> Self {
        if s.lo == 0.0 {
            base.mul_f64(s.hi) * inverse
        } else {
            s * base * inverse
        }
    }
    #[inline(always)]
    fn advance(
        s: Self,
        index: usize,
        derivative: Self,
        inverse: Self,
    ) -> Result<Self, SciOxideError> {
        let first = s + Self::from((2 * index + 1) as f64);
        // Polygamma orders are exact integers: their product fits binary64
        // throughout the work-bounded order range. Retain the original short
        // multiplication chain instead of two unnecessary wide products.
        Ok(
            if first.lo == 0.0 && first.hi == first.hi.floor() && first.hi.abs() < 67_108_864.0 {
                derivative.mul_f64(first.hi * (first.hi + 1.0)) * inverse * inverse
            } else {
                derivative * (first * inverse) * ((first + Self::from(1.0)) * inverse)
            },
        )
    }
}

/// A precision target in the scale selected by `power`. The first block may
/// be empty; later attempts add twelve terms without recomputing the prefix.
#[inline]
pub(crate) fn positive<T: Arithmetic>(
    s: T,
    q: T,
    first_block: usize,
    relative: f64,
    power: impl Fn(usize, T) -> Result<T, SciOxideError>,
    step: &mut impl FnMut() -> Result<(), SciOxideError>,
) -> Result<T, SciOxideError> {
    let zero = T::number(0.0)?;
    let mut sum = zero;
    let mut compensation = zero;
    let mut prefix = 0;
    let mut block = first_block;
    loop {
        for _ in 0..block {
            step()?;
            let x = q + T::number(prefix as f64)?;
            (sum, compensation) = T::accumulate(sum, compensation, power(prefix, x)?);
            prefix += 1;
        }
        let x = q + T::number(prefix as f64)?;
        let base = power(prefix, x)?;
        let direct = T::total(sum, compensation);
        if T::negligible_tail(s, x, base, direct, relative) {
            return finite(direct);
        }
        if let Some((value, _)) = tail(s, x, base, (sum, compensation), relative, 1, step)? {
            return finite(value);
        }
        block = 12;
    }
}

/// Euler--Maclaurin corrections shared by the positive sum and large real
/// shifts. `minimum` enforces s+2m>1 before using the omitted-term bound.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tail<T: Arithmetic>(
    s: T,
    x: T,
    base: T,
    prefix: (T, T),
    relative: f64,
    minimum: usize,
    step: &mut impl FnMut() -> Result<(), SciOxideError>,
) -> Result<Option<(T, f64)>, SciOxideError> {
    let one = T::number(1.0)?;
    let inverse = one / x;
    let (mut value, mut correction) = T::tail_start(s, x, base, prefix)?;
    let mut derivative = T::derivative(s, base, inverse);
    let mut previous = f64::INFINITY;
    for (index, pair) in BERNOULLI_OVER_FACTORIAL.windows(2).enumerate() {
        step()?;
        let term = T::coefficient(pair[0])? * derivative;
        if index + 1 >= minimum && term.magnitude() > previous {
            break;
        }
        (value, correction) = T::accumulate(value, correction, term);
        previous = term.magnitude();
        derivative = T::advance(s, index, derivative, inverse)?;
        let bound = T::correction_bound(derivative, pair[1])?;
        let result = T::total(value, correction);
        if index + 1 >= minimum && bound <= relative * result.magnitude() {
            return Ok(Some((result, bound)));
        }
    }
    Ok(None)
}

fn finite<T: Arithmetic>(value: T) -> Result<T, SciOxideError> {
    if value.magnitude().is_finite() {
        Ok(value)
    } else {
        Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "the positive Hurwitz sum exceeds the working range",
        ))
    }
}

/// Native real powers keep log1p when q+k rounds to q.
#[inline]
pub(crate) fn native<S: RealScalar>(
    s: S,
    q: S,
    step: &mut impl FnMut() -> Result<(), SciOxideError>,
) -> Result<S, SciOxideError> {
    positive(
        s,
        q,
        12,
        S::EPSILON.into_f64(),
        |offset, _| Ok((-s * (S::try_from_f64_lossy(offset as f64)? / q).ln_1p()).exp()),
        step,
    )
}

/// Work-owning adapter for ordinary and logarithmic real evaluations.
pub(crate) fn sum<S: RealScalar>(s: S, q: S, work: &mut Work) -> Result<(S, usize), SciOxideError> {
    let start = work.used;
    let value = native(s, q, &mut || work.try_step())?;
    Ok((value, work.used - start))
}
