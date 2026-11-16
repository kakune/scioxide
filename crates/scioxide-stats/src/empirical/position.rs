//! Hyndman–Fan quantile positions over sorted and frequency-weighted data.

use crate::arithmetic::interpolate;
use crate::ranking::finite_total_cmp;
use core::cmp::Ordering;

use scioxide_core::{
    InvalidInputReason, NonFiniteOrigin, RealScalar, Scaled, ScaledAccumulator, SciOxideError,
};

use super::exact_mass::{EXACT_MASS_MIN_EXPONENT, ExactMass, probability_parts};
use super::{QuantileConvention, QuantileMethod, QuantileProbability, QuantileResult};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct QuantilePosition {
    pub(super) lower: usize,
    pub(super) upper: usize,
    pub(super) fraction: f64,
}

impl QuantilePosition {
    /// Interpolates between the two order statistics returned by `value`.
    #[inline]
    pub(crate) fn evaluate<S: RealScalar>(
        self,
        value: impl Fn(usize) -> S,
    ) -> Result<S, SciOxideError> {
        if self.lower == self.upper {
            return Ok(value(self.lower));
        }
        interpolate(value(self.lower), value(self.upper), self.fraction)
    }
}

pub(crate) fn quantile_position(
    len: usize,
    probability: QuantileProbability,
    method: QuantileMethod,
) -> Result<QuantilePosition, SciOxideError> {
    if len == 0 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("samples"),
            InvalidInputReason::Empty,
            "an empirical distribution requires at least one observation",
        ));
    }

    match method {
        QuantileMethod::NearestRank => nearest_rank_position(len, probability),
        QuantileMethod::Linear => linear_position(len, probability),
        _ => general_position(len, probability, method),
    }
}

fn nearest_rank_position(
    len: usize,
    probability: QuantileProbability,
) -> Result<QuantilePosition, SciOxideError> {
    if probability.get() == 0.0 {
        return Ok(QuantilePosition {
            lower: 0,
            upper: 0,
            fraction: 0.0,
        });
    }

    let (floor, fraction) = scaled_probability(len, probability)?;
    let rank = if fraction == 0.0 {
        floor
    } else {
        floor.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("nearest_rank_quantile_index"))
        })?
    };
    let index = rank.checked_sub(1).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("nearest_rank_zero_based_index"))
    })?;
    if index >= len {
        return Err(SciOxideError::out_of_bounds(
            scioxide_core::quantity!("nearest_rank_quantile"),
            index,
            len,
        ));
    }
    Ok(QuantilePosition {
        lower: index,
        upper: index,
        fraction: 0.0,
    })
}

fn linear_position(
    len: usize,
    probability: QuantileProbability,
) -> Result<QuantilePosition, SciOxideError> {
    let span = len.checked_sub(1).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("linear_quantile_sample_span"))
    })?;
    let (lower, fraction) = scaled_probability(span, probability)?;
    let upper = if fraction == 0.0 {
        lower
    } else {
        lower.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("linear_quantile_upper_index"))
        })?
    };
    if upper >= len {
        return Err(SciOxideError::out_of_bounds(
            scioxide_core::quantity!("linear_quantile"),
            upper,
            len,
        ));
    }
    Ok(QuantilePosition {
        lower,
        upper,
        fraction,
    })
}

pub(super) fn scaled_probability(
    factor: usize,
    probability: QuantileProbability,
) -> Result<(usize, f64), SciOxideError> {
    let value = probability.get();
    if factor == 0 || value == 0.0 {
        return Ok((0, 0.0));
    }
    if value == 1.0 {
        return Ok((factor, 0.0));
    }

    let (significand, denominator_shift) = probability_parts(probability);
    let numerator = (factor as u128)
        .checked_mul(u128::from(significand))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("quantile_probability_numerator"))
        })?;
    if denominator_shift >= u128::BITS {
        return Ok((0, value * factor as f64));
    }

    let floor_u128 = numerator >> denominator_shift;
    let floor = usize::try_from(floor_u128).map_err(|_| {
        SciOxideError::size_overflow(scioxide_core::quantity!(
            "quantile_probability_integer_part"
        ))
    })?;
    let remainder_mask = (1_u128 << denominator_shift) - 1;
    let remainder = numerator & remainder_mask;
    let shift = i32::try_from(denominator_shift).map_err(|_| {
        SciOxideError::size_overflow(scioxide_core::quantity!("quantile_fraction_exponent"))
    })?;
    let fraction = remainder as f64 / 2.0_f64.powi(shift);
    Ok((floor, fraction))
}

// H&F equation (1), evaluated with exact dyadic ranks. Existing type 1/7
// kernels remain specialized fast paths for their simpler positions.
fn general_position(
    len: usize,
    p: QuantileProbability,
    method: QuantileMethod,
) -> Result<QuantilePosition, SciOxideError> {
    let endpoint = |index| QuantilePosition {
        lower: index,
        upper: index,
        fraction: 0.0,
    };
    if p.get() == 0.0 || len == 1 {
        return Ok(endpoint(0));
    }
    if p.get() == 1.0 {
        return Ok(endpoint(len - 1));
    }
    let (a, b, c, d, subtract) = match method {
        QuantileMethod::ClosestObservation => (2, 0, 1, 2, true),
        QuantileMethod::Hazen => (2, 0, 1, 2, false),
        QuantileMethod::Weibull => (1, 1, 0, 1, false),
        QuantileMethod::MedianUnbiased => (3, 1, 1, 3, false),
        QuantileMethod::NormalUnbiased => (8, 2, 3, 8, false),
        _ => (1, 0, 0, 1, false),
    };
    let (significand, shift) = probability_parts(p);
    let mut coefficient = ExactMass::count(len).multiplied_by_u64(a);
    coefficient.add_assign(ExactMass::count(b));
    let mut numerator = coefficient.multiplied_by_u64(significand);
    let offset = ExactMass::count(c).shifted_left(shift as usize);
    if subtract {
        if numerator.compare(offset) != Ordering::Greater {
            return Ok(endpoint(0));
        }
        numerator.subtract_assign(offset);
    } else {
        numerator.add_assign(offset);
    }
    let denominator = ExactMass::count(d).shifted_left(shift as usize);
    let quotient = numerator.divided_by_usize(d);
    let floor = quotient
        .shifted_right(shift as usize + (-EXACT_MASS_MIN_EXPONENT) as usize)
        .to_usize()
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("quantile_rank")))?;
    if floor >= len {
        return Ok(endpoint(len - 1));
    }
    numerator.subtract_assign(denominator.multiplied_by_u64(floor as u64));
    let exact = numerator.highest_bit().is_none();
    let fraction = match method {
        QuantileMethod::AveragedInverseCdf => {
            if exact {
                0.5
            } else {
                1.0
            }
        }
        QuantileMethod::ClosestObservation => {
            if exact && floor % 2 == 0 {
                0.0
            } else {
                1.0
            }
        }
        _ => numerator
            .to_scaled(0)?
            .div(denominator.to_scaled(0)?)?
            .try_value_to()
            .ok()
            .unwrap_or(0.0),
    };
    if floor == 0 {
        return Ok(endpoint(0));
    }
    Ok(QuantilePosition {
        lower: floor - 1,
        upper: floor,
        fraction,
    })
}

/// Returns a frequency-weighted quantile without expanding the observations.
///
/// Pairs are `(value, count)`, and pairs with zero count are ignored. The
/// buffer is checked first and then sorted in place in O(n log n) time.
///
/// # Errors
/// Fails for a non-finite value with positive count, a zero total count,
/// total-count overflow, or an interpolated result that is not representable.
/// Only the last failure occurs after the buffer is reordered.
pub(crate) fn frequency_quantile_in_place<S: RealScalar>(
    samples: &mut [(S, u64)],
    p: QuantileProbability,
    method: QuantileMethod,
) -> Result<QuantileResult<S>, SciOxideError> {
    let count = samples.iter().try_fold(0_usize, |total, &(value, count)| {
        if count > 0 && !value.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "nonfinite support",
            ));
        }
        total
            .checked_add(crate::weights::frequency_index(count)?)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("frequency_count"))
            })
    })?;
    let position = quantile_position(count, p, method)?;
    samples.sort_unstable_by(|a, b| match (a.1 == 0, b.1 == 0) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => finite_total_cmp(&a.0, &b.0),
    });
    let mut cumulative = 0;
    let mut lower = S::ZERO;
    let mut upper = S::ZERO;
    for &(value, count) in samples.iter() {
        let next = cumulative + count;
        if cumulative <= position.lower as u64 && (position.lower as u64) < next {
            lower = value;
        }
        if cumulative <= position.upper as u64 && (position.upper as u64) < next {
            upper = value;
            break;
        }
        cumulative = next;
    }
    let value = interpolate(lower, upper, position.fraction)?;
    Ok(QuantileResult {
        value,
        probability: p,
        convention: QuantileConvention::Frequency(method),
    })
}

pub(crate) fn quantile_sorted_by<S: RealScalar>(
    len: usize,
    p: QuantileProbability,
    method: QuantileMethod,
    value: impl Fn(usize) -> S,
) -> Result<QuantileResult<S>, SciOxideError> {
    let position = quantile_position(len, p, method)?;
    let result = position.evaluate(value)?;
    Ok(QuantileResult::unweighted(result, p, method))
}

/// Returns the quantile of sorted scaled values as an unrounded two-part sum,
/// so a midpoint of two order statistics is exact.
pub(crate) fn quantile_scaled_sorted_by(
    len: usize,
    p: QuantileProbability,
    method: QuantileMethod,
    value: impl Fn(usize) -> Result<Scaled<f64>, SciOxideError>,
) -> Result<ScaledAccumulator<f64>, SciOxideError> {
    let position = quantile_position(len, p, method)?;
    crate::arithmetic::interpolate_scaled(
        value(position.lower)?,
        value(position.upper)?,
        Scaled::<f64>::from_value(position.fraction)?,
    )
}
