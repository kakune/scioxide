//! Empirical tail means with a fractionally included boundary observation.

use crate::ranking::finite_total_cmp;
use core::cmp::Ordering;

use scioxide_core::{InvalidInputReason, NumericalIssue, RealScalar, Scaled, SciOxideError};

use super::exact_mass::{ExactMass, ExactTarget, ExactWeightedMean, validated_exact_mass};
use super::{
    QuantileProbability, TailMeanResult, TailSide, validate_samples, validate_weighted_samples,
    weighted_total_cmp,
};
#[cfg(doc)]
use super::{RawEmpiricalDistribution, RawWeightedEmpiricalDistribution};

/// Returns a tail mean by sorting a work buffer in O(n log n) time.
///
/// The buffer and the tail are checked before the buffer is reordered.
///
/// # Errors
///
/// Fails as [`RawEmpiricalDistribution::tail_mean`] and for an empty or
/// non-finite buffer.
pub(crate) fn tail_mean_in_place<S: RealScalar>(
    work_buffer: &mut [S],
    probability: QuantileProbability,
    side: TailSide,
) -> Result<TailMeanResult<S>, SciOxideError> {
    validate_samples(work_buffer, scioxide_core::argument!("work_buffer"))?;
    let plan = unweighted_tail_plan(work_buffer.len(), probability, side)?;
    work_buffer.sort_unstable_by(finite_total_cmp);
    evaluate_unweighted_tail(work_buffer, probability, side, plan)
}

/// Returns a weighted tail mean by sorting `(observation, weight)` pairs in
/// O(n log n) time.
///
/// The pairs, their total weight and the tail are checked before the buffer
/// is reordered.
///
/// # Errors
///
/// Fails as [`RawWeightedEmpiricalDistribution::tail_mean`] and for invalid
/// pairs.
pub(crate) fn weighted_tail_mean_in_place<S: RealScalar>(
    work_buffer: &mut [(S, S)],
    probability: QuantileProbability,
    side: TailSide,
) -> Result<TailMeanResult<S>, SciOxideError> {
    let summary = validate_weighted_samples(work_buffer, scioxide_core::argument!("work_buffer"))?;
    validate_tail_probability(probability, side)?;
    work_buffer.sort_unstable_by(weighted_total_cmp);
    evaluate_weighted_tail(work_buffer, summary.total_mass, probability, side)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct UnweightedTailPlan {
    pub(super) boundary_index: usize,
    pub(super) boundary_mass: ExactTarget,
    tail_mass: ExactTarget,
}

fn validate_tail_probability(
    probability: QuantileProbability,
    side: TailSide,
) -> Result<(), SciOxideError> {
    let empty = match side {
        TailSide::Lower => probability.get() == 0.0,
        TailSide::Upper => probability.get() == 1.0,
    };
    if empty {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("probability"),
            InvalidInputReason::OutOfRange,
            "the selected probability defines an empty empirical tail",
        ));
    }
    Ok(())
}

pub(super) fn unweighted_tail_plan(
    len: usize,
    probability: QuantileProbability,
    side: TailSide,
) -> Result<UnweightedTailPlan, SciOxideError> {
    validate_tail_probability(probability, side)?;
    let total_mass = ExactMass::count(len);
    let tail_mass = match side {
        TailSide::Lower => ExactTarget::lower(total_mass, probability),
        TailSide::Upper => ExactTarget::upper(total_mass, probability),
    };
    let included = tail_mass.ceil_count()?;
    if included == 0 || included > len {
        return Err(SciOxideError::out_of_bounds(
            scioxide_core::quantity!("empirical_tail_included_observations"),
            included,
            len,
        ));
    }
    let interior_count = included.checked_sub(1).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("empirical_tail_interior_count"))
    })?;
    let mut boundary_mass = tail_mass;
    boundary_mass.subtract_integer(ExactMass::count(interior_count));
    let boundary_index = match side {
        TailSide::Lower => interior_count,
        TailSide::Upper => len.checked_sub(included).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!(
                "upper_empirical_tail_boundary_index"
            ))
        })?,
    };
    Ok(UnweightedTailPlan {
        boundary_index,
        boundary_mass,
        tail_mass,
    })
}

pub(super) fn evaluate_unweighted_tail<S: RealScalar>(
    sorted: &[S],
    probability: QuantileProbability,
    side: TailSide,
    plan: UnweightedTailPlan,
) -> Result<TailMeanResult<S>, SciOxideError> {
    let boundary_value = sorted[plan.boundary_index];
    let interior = match side {
        TailSide::Lower => &sorted[..plan.boundary_index],
        TailSide::Upper => &sorted[plan.boundary_index + 1..],
    };
    let unit_mass =
        ExactTarget::from_integer(ExactMass::count(1), plan.tail_mass.denominator_shift);
    let state = interior.iter().copied().try_fold(
        ExactWeightedMean::zero(plan.tail_mass.denominator_shift),
        |state, value| state.add(value, unit_mass),
    )?;
    let state = state.add(boundary_value, plan.boundary_mass)?;

    let tied_interior = interior
        .iter()
        .filter(|value| finite_total_cmp(*value, &boundary_value) == Ordering::Equal)
        .count();
    let boundary_mass = plan.boundary_mass.add(ExactTarget::from_integer(
        ExactMass::count(tied_interior),
        plan.tail_mass.denominator_shift,
    ));
    let boundary_contribution = exact_boundary_contribution(boundary_value, boundary_mass);
    Ok(TailMeanResult::new(
        state.mean,
        probability,
        side,
        plan.tail_mass.to_optional_scalar(),
        boundary_value,
        boundary_mass.to_optional_scalar(),
        boundary_contribution,
    ))
}

pub(super) fn evaluate_weighted_tail<S: RealScalar>(
    sorted: &[(S, S)],
    total_mass: ExactMass,
    probability: QuantileProbability,
    side: TailSide,
) -> Result<TailMeanResult<S>, SciOxideError> {
    validate_tail_probability(probability, side)?;
    let target = match side {
        TailSide::Lower => ExactTarget::lower(total_mass, probability),
        TailSide::Upper => ExactTarget::upper(total_mass, probability),
    };
    if target.is_zero() {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Underflow,
            "a non-empty weighted tail has zero exact mass",
        ));
    }

    let mut state = ExactWeightedMean::zero(target.denominator_shift);
    let mut remaining = target;
    let mut boundary = None;
    match side {
        TailSide::Lower => {
            let mut start = 0_usize;
            while start < sorted.len() {
                let end = equal_value_run_end(sorted, start);
                let value = sorted[start].0;
                let group_mass =
                    validated_exact_mass(sorted[start..end].iter().map(|(_, weight)| *weight));
                if group_mass.highest_bit().is_none() {
                    start = end;
                    continue;
                }
                if remaining.compare_integer(group_mass) != Ordering::Greater {
                    state = state.add(value, remaining)?;
                    boundary = Some((value, remaining));
                    break;
                }
                state = state.add(
                    value,
                    ExactTarget::from_integer(group_mass, target.denominator_shift),
                )?;
                remaining.subtract_integer(group_mass);
                start = end;
            }
        }
        TailSide::Upper => {
            let mut end = sorted.len();
            while end > 0 {
                let start = equal_value_run_start(sorted, end);
                let value = sorted[start].0;
                let group_mass =
                    validated_exact_mass(sorted[start..end].iter().map(|(_, weight)| *weight));
                if group_mass.highest_bit().is_none() {
                    end = start;
                    continue;
                }
                if remaining.compare_integer(group_mass) != Ordering::Greater {
                    state = state.add(value, remaining)?;
                    boundary = Some((value, remaining));
                    break;
                }
                state = state.add(
                    value,
                    ExactTarget::from_integer(group_mass, target.denominator_shift),
                )?;
                remaining.subtract_integer(group_mass);
                end = start;
            }
        }
    }

    let (boundary_value, boundary_mass) = boundary.ok_or_else(|| {
        SciOxideError::numerical_instability(
            NumericalIssue::Breakdown,
            "positive support was exhausted before reaching target tail mass",
        )
    })?;
    let boundary_contribution = exact_boundary_contribution(boundary_value, boundary_mass);
    Ok(TailMeanResult::new(
        state.mean,
        probability,
        side,
        target.to_optional_scalar(),
        boundary_value,
        boundary_mass.to_optional_scalar(),
        boundary_contribution,
    ))
}

fn equal_value_run_end<S: RealScalar>(sorted: &[(S, S)], start: usize) -> usize {
    let value = sorted[start].0;
    sorted[start + 1..]
        .iter()
        .position(|(candidate, _)| finite_total_cmp(candidate, &value) != Ordering::Equal)
        .map_or(sorted.len(), |offset| start + 1 + offset)
}

fn equal_value_run_start<S: RealScalar>(sorted: &[(S, S)], end: usize) -> usize {
    let value = sorted[end - 1].0;
    sorted[..end - 1]
        .iter()
        .rposition(|(candidate, _)| finite_total_cmp(candidate, &value) != Ordering::Equal)
        .map_or(0, |index| index + 1)
}

fn exact_boundary_contribution<S: RealScalar>(
    boundary_value: S,
    boundary_mass: ExactTarget,
) -> Option<S> {
    boundary_mass
        .to_scaled()
        .ok()?
        .mul(Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(boundary_value)).ok()?)
        .ok()?
        .try_value_to()
        .ok()
}
