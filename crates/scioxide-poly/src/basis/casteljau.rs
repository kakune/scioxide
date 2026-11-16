//! De Casteljau kernels shared by univariate Bernstein bases and tensor nets.
//!
//! Convex blends run natively while operands stay in the sqrt range, and replay
//! a level in scaled arithmetic only when a result leaves that range.
//! Tensor-product nets call these kernels; nothing here depends on them.
use crate::admission::native_value;
use crate::util::{invalid, minimum_len, round_complex, scaled_complex};
use alloc::vec::Vec;
use scioxide_core::backend::try_copy_slice;
use scioxide_core::backend::try_filled;
use scioxide_core::{Interval, NumericScalar, RealScalar, Scaled, ScaledValue, SciOxideError};

/// Prepared convex weights; scaled arithmetic is confined to range fallback.
pub(crate) struct Blend<C: NumericScalar> {
    point: C,
    complement: C,
    native: bool,
}
impl<C: NumericScalar> Blend<C> {
    pub(crate) fn new(point: C) -> Self {
        let complement = C::ONE - point;
        Self {
            point,
            complement,
            native: native_value(point) & native_value(complement),
        }
    }
    #[inline]
    pub(crate) fn apply(&self, a: C, b: C) -> Result<C, SciOxideError> {
        if self.point == C::ZERO {
            return Ok(a);
        }
        if self.point == C::ONE {
            return Ok(b);
        }
        // Sqrt-range admission leaves headroom for all complex products/sums.
        // Admit the result before using it as another product operand; this also
        // retries cancellation into tiny components in extended range.
        if self.native && native_value(a) & native_value(b) {
            let value = a * self.complement + b * self.point;
            if native_value(value) {
                return Ok(value);
            }
        }
        self.extended(a, b)
    }
    #[cold]
    #[inline(never)]
    pub(crate) fn extended(&self, a: C, b: C) -> Result<C, SciOxideError> {
        let t = scaled_complex(self.point)?;
        round_complex(
            scaled_complex(a)?
                .mul(scaled_complex(C::ONE)?.sub(t)?)?
                .add(scaled_complex(b)?.mul(t)?)?,
        )
    }
}

/// One triangular kernel. Observers borrow each level for derivatives/subdivision.
pub(crate) fn triangle<C: NumericScalar>(
    work: &mut [C],
    weights: &Blend<C>,
    mut observe: impl FnMut(&[C]),
) -> Result<C, SciOxideError> {
    observe(work);
    // Admit the initial level once. Every native result is checked before the
    // next level, so admitted levels need no repeated operand classification.
    // Endpoints keep their exact-control path, including signed zeros.
    let mut admitted = weights.native
        && weights.point != C::ZERO
        && weights.point != C::ONE
        && work
            .iter()
            .copied()
            .fold(true, |safe, c| safe & native_value(c));
    for remaining in (1..work.len()).rev() {
        let mut next_admitted =
            weights.native && weights.point != C::ZERO && weights.point != C::ONE;
        if admitted {
            for i in 0..remaining {
                let a = work[i];
                let b = work[i + 1];
                let value = a * weights.complement + b * weights.point;
                let safe = native_value(value);
                work[i] = if safe { value } else { weights.extended(a, b)? };
                next_admitted &= safe;
            }
        } else {
            for i in 0..remaining {
                work[i] = weights.apply(work[i], work[i + 1])?;
                next_admitted &= native_value(work[i]);
            }
        }
        admitted = next_admitted;
        observe(&work[..remaining]);
    }
    Ok(work[0])
}
pub(crate) fn de_casteljau_in_place<C: NumericScalar>(
    work: &mut [C],
    point: C,
) -> Result<C, SciOxideError> {
    if point == C::ZERO {
        return Ok(work[0]);
    }
    if point == C::ONE {
        return Ok(work[work.len() - 1]);
    }
    triangle(work, &Blend::new(point), |_| {})
}

/// One copy and one triangle yield both requested outputs. No derivative net.
pub(crate) fn value_and_derivative<C: NumericScalar>(
    coefficients: &[C],
    point: C,
    domain: Interval<C::Real>,
    scratch: &mut [C],
) -> Result<(C, C, C), SciOxideError> {
    minimum_len(coefficients.len(), scratch.len())?;
    if coefficients.len() == 1 {
        return Ok((coefficients[0], C::ZERO, C::ZERO));
    }
    let t = normalized(point, domain)?;
    let last = coefficients.len() - 1;
    let (value, pair) = if t == C::ZERO {
        (coefficients[0], (coefficients[0], coefficients[1]))
    } else if t == C::ONE {
        (
            coefficients[last],
            (coefficients[last - 1], coefficients[last]),
        )
    } else {
        let work = &mut scratch[..coefficients.len()];
        work.copy_from_slice(coefficients);
        let mut pair = (C::ZERO, C::ZERO);
        let value = triangle(work, &Blend::new(t), |level| {
            if level.len() == 2 {
                pair = (level[0], level[1]);
            }
        })?;
        (value, pair)
    };
    let width = Scaled::from_value(domain.upper())?.sub(Scaled::from_value(domain.lower())?)?;
    let degree = Scaled::from_value(C::Real::try_from_usize_exact(last)?)?;
    let derivative = round_complex(
        scaled_complex(pair.1)?
            .sub(scaled_complex(pair.0)?)?
            .mul_real(degree)?
            .div(ScaledValue::from_components(width, Scaled::ZERO))?,
    )?;
    Ok((value, derivative, t))
}

pub(crate) fn normalized<C: NumericScalar>(
    point: C,
    domain: Interval<C::Real>,
) -> Result<C, SciOxideError> {
    crate::util::finite_input(point, "nonfinite value")?;
    if point == C::from_real(domain.lower()) {
        return Ok(C::ZERO);
    }
    if point == C::from_real(domain.upper()) {
        return Ok(C::ONE);
    }
    let width = domain.upper() - domain.lower();
    let numerator = point - C::from_real(domain.lower());
    if width.is_finite()
        && width >= C::Real::MIN_POSITIVE
        && native_value(numerator)
        && crate::admission::native_range(width)
    {
        let value = numerator / C::from_real(width);
        if native_value(value) {
            return Ok(value);
        }
    }
    let width = Scaled::from_value(domain.upper())?.sub(Scaled::from_value(domain.lower())?)?;
    round_complex(
        scaled_complex(point)?
            .sub(scaled_complex(C::from_real(domain.lower()))?)?
            .div(ScaledValue::from_components(width, Scaled::ZERO))?,
    )
}
/// Normalized interior split parameter; the caller's parameter is `point`.
pub(crate) fn split_parameter<C: NumericScalar>(
    point: C::Real,
    domain: Interval<C::Real>,
) -> Result<C, SciOxideError> {
    // Endpoints would produce a zero-width piece, which `Interval`-based domains
    // (and the normalization above) cannot represent.
    crate::util::finite_input(point, "subdivision point must be finite")?;
    if point <= domain.lower() || point >= domain.upper() {
        return Err(invalid(
            scioxide_core::argument!("point"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "subdivision point must be strictly interior",
        ));
    }
    normalized(C::from_real(point), domain)
}
pub(crate) fn split_curve<C: NumericScalar>(
    coefficients: &[C],
    t: C,
) -> Result<(Vec<C>, Vec<C>), SciOxideError> {
    let n = coefficients.len();
    let mut work = try_copy_slice(coefficients, scioxide_core::quantity!("de_casteljau_work"))?;
    let mut left = try_filled(n, C::ZERO, scioxide_core::quantity!("left_controls"))?;
    let mut right = try_filled(n, C::ZERO, scioxide_core::quantity!("right_controls"))?;
    triangle(&mut work, &Blend::new(t), |level| {
        left[n - level.len()] = level[0];
        right[level.len() - 1] = level[level.len() - 1];
    })?;
    Ok((left, right))
}

/// Elevate inside target-sized storage; descending writes preserve unread neighbors.
pub(crate) fn elevate_curve<C: NumericScalar>(
    work: &mut [C],
    length: usize,
) -> Result<(), SciOxideError> {
    for n in length..work.len() {
        work[n] = work[n - 1];
        let denominator = C::Real::try_from_usize_exact(n)?;
        for i in (1..n).rev() {
            let alpha = C::Real::try_from_usize_exact(i)? / denominator;
            work[i] = Blend::new(C::from_real(alpha)).apply(work[i], work[i - 1])?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blend_boundaries<R: RealScalar>() {
        let weights = Blend::new(R::HALF);
        for edge in [R::MIN_POSITIVE.sqrt(), R::MAX.sqrt() / (R::TWO + R::TWO)] {
            for a in [edge.next_after(R::ZERO), edge, edge.next_after(R::MAX)] {
                let actual = weights.apply(a, -a).unwrap();
                assert_eq!(actual, R::ZERO);
                assert_eq!(weights.apply(a, a).unwrap(), a);
            }
        }
        let outside = Blend::new(R::TWO);
        assert_eq!(outside.apply(R::MAX, R::MAX).unwrap(), R::MAX);
        assert!(
            Blend::new(R::ZERO)
                .apply(-R::ZERO, R::MAX)
                .unwrap()
                .is_sign_negative()
        );
    }

    fn triangle_range_transition<R: RealScalar>() {
        let large = R::MAX.sqrt() / (R::TWO + R::TWO);
        // The first level [2*large, large] leaves native admission, then the
        // final extrapolating blend cancels exactly. Its inputs must be replayed
        // in scaled arithmetic even though the final value is in native range.
        let mut work = [R::ZERO, large, large];
        assert_eq!(
            triangle(&mut work, &Blend::new(R::TWO), |_| {}).unwrap(),
            R::ZERO
        );
        let small = R::MIN_POSITIVE.sqrt();
        let mut work = [small, R::ZERO, R::ZERO];
        assert_eq!(
            triangle(&mut work, &Blend::new(R::HALF), |_| {}).unwrap(),
            small / (R::TWO + R::TWO)
        );
    }

    #[test]
    fn triangle_checks_each_level_before_reusing_native_operands() {
        triangle_range_transition::<f32>();
        triangle_range_transition::<f64>();
    }

    #[test]
    fn convex_blends_at_native_admission_boundaries_and_range_fallback() {
        blend_boundaries::<f32>();
        blend_boundaries::<f64>();
    }
}
