//! Quantile selection in a work buffer, for unweighted and weighted samples.

use crate::arithmetic::interpolate;
use crate::ranking::finite_total_cmp;
use core::cmp::Ordering;

use scioxide_core::{NumericalIssue, RealScalar, SciOxideError};

#[cfg(doc)]
use super::RawWeightedEmpiricalDistribution;
use super::exact_mass::{ExactMass, ExactTarget, validated_exact_mass};
use super::position::quantile_position;
use super::{
    QuantileMethod, QuantileProbability, QuantileResult, validate_samples,
    validate_weighted_samples,
};

/// Returns a sample quantile by partial selection in a work buffer.
///
/// The buffer is checked and the order-statistic indices are computed before
/// the buffer is reordered. Selection takes expected O(n) time. The second
/// order statistic is the extremum of the smaller partition. On success the
/// buffer holds a permutation of its input.
///
/// # Errors
///
/// Fails for an empty or non-finite buffer, index overflow, or an
/// interpolated result that is not representable. Only the last failure
/// occurs after the buffer is reordered.
pub(crate) fn quantile_in_place<S: RealScalar>(
    work_buffer: &mut [S],
    probability: QuantileProbability,
    method: QuantileMethod,
) -> Result<QuantileResult<S>, SciOxideError> {
    validate_samples(work_buffer, scioxide_core::argument!("work_buffer"))?;
    let position = quantile_position(work_buffer.len(), probability, method)?;
    let [lower, upper] = select_adjacent_by(
        work_buffer,
        position.lower,
        position.upper,
        finite_total_cmp,
    )?;
    let value = if position.lower == position.upper {
        lower
    } else {
        interpolate(lower, upper, position.fraction)?
    };

    Ok(QuantileResult::unweighted(value, probability, method))
}

/// Selects the order statistics at equal or adjacent zero-based ranks in a
/// checked buffer, in expected O(n) time.
///
/// Returning both values lets callers interpolate in extended precision.
///
/// # Errors
/// Fails, before reordering, if the ranks are not equal or adjacent or lie
/// outside the buffer.
pub(crate) fn select_adjacent_by<T: Copy>(
    work_buffer: &mut [T],
    lower: usize,
    upper: usize,
    compare: impl Fn(&T, &T) -> Ordering,
) -> Result<[T; 2], SciOxideError> {
    if lower > upper || upper >= work_buffer.len() || upper - lower > 1 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("ranks"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "selection requires equal or adjacent ranks within the buffer",
        ));
    }
    if lower == upper {
        let (_, selected, _) = work_buffer.select_nth_unstable_by(lower, compare);
        Ok([*selected; 2])
    } else if work_buffer.len() - 1 - lower < upper {
        // The adjacent endpoint is an extremum of one partition. Scan the
        // smaller side: upper-tail quantiles need only the short right tail.
        let (_, lower, upper_partition) = work_buffer.select_nth_unstable_by(lower, &compare);
        let upper = upper_partition
            .iter()
            .copied()
            .min_by(compare)
            .ok_or_else(|| {
                SciOxideError::numerical_instability(
                    NumericalIssue::Breakdown,
                    "selection did not retain the adjacent upper order statistic",
                )
            })?;
        Ok([*lower, upper])
    } else {
        let (lower_partition, upper, _) = work_buffer.select_nth_unstable_by(upper, &compare);
        let lower = lower_partition
            .iter()
            .copied()
            .max_by(compare)
            .ok_or_else(|| {
                SciOxideError::numerical_instability(
                    NumericalIssue::Breakdown,
                    "selection did not retain the adjacent lower order statistic",
                )
            })?;
        Ok([lower, *upper])
    }
}

/// Returns the weighted inverse-ECDF quantile by selection in a buffer of
/// `(observation, weight)` pairs.
///
/// The pairs are checked and the exact target weight `p W` is formed before
/// the buffer is reordered. Selection is a deterministic three-way
/// introselect, O(n) in the worst case. On success the buffer holds a
/// permutation of its input, not necessarily sorted.
///
/// # Errors
///
/// Fails as
/// [`RawWeightedEmpiricalDistribution::weighted_inverse_ecdf_quantile`] and
/// for invalid pairs. Every failure occurs before the buffer is reordered.
pub(crate) fn weighted_inverse_ecdf_quantile_in_place<S: RealScalar>(
    work_buffer: &mut [(S, S)],
    probability: QuantileProbability,
) -> Result<QuantileResult<S>, SciOxideError> {
    let summary = validate_weighted_samples(work_buffer, scioxide_core::argument!("work_buffer"))?;
    let target = weighted_quantile_target(summary.total_mass, probability);
    let value = match target.kind {
        WeightedQuantileTargetKind::Minimum => positive_support_endpoint(work_buffer, true),
        WeightedQuantileTargetKind::Maximum => positive_support_endpoint(work_buffer, false),
        WeightedQuantileTargetKind::Mass => weighted_introselect(work_buffer, target.mass),
    };
    Ok(QuantileResult::weighted(value, probability))
}

#[derive(Debug, Clone, Copy)]
enum WeightedQuantileTargetKind {
    Minimum,
    Maximum,
    Mass,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct WeightedQuantileTarget {
    kind: WeightedQuantileTargetKind,
    mass: ExactMass,
}

/// Selection work counted on the current thread, for unit tests.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(super) struct WeightedSelectionMetrics {
    comparisons: u64,
    classified_pairs: u64,
    partitions: u64,
    fallback_activations: u64,
}

#[cfg(test)]
std::thread_local! {
    static SELECTION_METRICS: core::cell::Cell<WeightedSelectionMetrics> =
        const { core::cell::Cell::new(WeightedSelectionMetrics::ZERO) };
}

#[cfg(test)]
impl WeightedSelectionMetrics {
    const ZERO: Self = Self {
        comparisons: 0,
        classified_pairs: 0,
        partitions: 0,
        fallback_activations: 0,
    };

    /// Runs `operation` and returns its result with the selection work it
    /// performed on the current thread.
    pub(super) fn record<T>(operation: impl FnOnce() -> T) -> (T, Self) {
        SELECTION_METRICS.set(Self::ZERO);
        let result = operation();
        (result, SELECTION_METRICS.replace(Self::ZERO))
    }

    /// Returns the number of comparisons plus classified pairs.
    #[must_use]
    pub(super) const fn work_units(self) -> u64 {
        self.comparisons.saturating_add(self.classified_pairs)
    }

    /// Returns the number of three-way partitions, including fallback pivot
    /// selection.
    #[must_use]
    pub(super) const fn partitions(self) -> u64 {
        self.partitions
    }

    /// Returns the number of switches to the worst-case fallback.
    #[must_use]
    pub(super) const fn fallback_activations(self) -> u64 {
        self.fallback_activations
    }

    fn update(update: impl FnOnce(&mut Self)) {
        let mut metrics = SELECTION_METRICS.get();
        update(&mut metrics);
        SELECTION_METRICS.set(metrics);
    }
}

// Work-counting hooks: their bodies compile only in unit tests, so
// production selection carries no counter state or argument.
#[inline(always)]
fn count_comparison() {
    #[cfg(test)]
    WeightedSelectionMetrics::update(|metrics| {
        metrics.comparisons = metrics.comparisons.saturating_add(1);
    });
}

#[inline(always)]
fn count_classified_pair() {
    #[cfg(test)]
    WeightedSelectionMetrics::update(|metrics| {
        metrics.classified_pairs = metrics.classified_pairs.saturating_add(1);
    });
}

#[inline(always)]
fn count_partition() {
    #[cfg(test)]
    WeightedSelectionMetrics::update(|metrics| {
        metrics.partitions = metrics.partitions.saturating_add(1);
    });
}

#[inline(always)]
fn count_fallback() {
    #[cfg(test)]
    WeightedSelectionMetrics::update(|metrics| {
        metrics.fallback_activations = metrics.fallback_activations.saturating_add(1);
    });
}

pub(super) fn weighted_quantile_target(
    total_mass: ExactMass,
    probability: QuantileProbability,
) -> WeightedQuantileTarget {
    if probability.get() == 0.0 {
        return WeightedQuantileTarget {
            kind: WeightedQuantileTargetKind::Minimum,
            mass: ExactMass::zero(),
        };
    }
    if probability.get() == 1.0 {
        return WeightedQuantileTarget {
            kind: WeightedQuantileTargetKind::Maximum,
            mass: ExactMass::zero(),
        };
    }

    WeightedQuantileTarget {
        kind: WeightedQuantileTargetKind::Mass,
        mass: ExactTarget::lower(total_mass, probability).ceil_mass(),
    }
}

pub(super) fn evaluate_sorted_weighted_quantile<S: RealScalar>(
    sorted: &[(S, S)],
    target: WeightedQuantileTarget,
) -> Result<S, SciOxideError> {
    let first_positive = sorted.iter().find(|(_, weight)| *weight > S::ZERO);
    let last_positive = sorted.iter().rev().find(|(_, weight)| *weight > S::ZERO);
    match target.kind {
        WeightedQuantileTargetKind::Minimum => {
            first_positive.map(|(value, _)| *value).ok_or_else(|| {
                SciOxideError::numerical_instability(
                    NumericalIssue::Breakdown,
                    "validated weighted support contains no positive mass",
                )
            })
        }
        WeightedQuantileTargetKind::Maximum => {
            last_positive.map(|(value, _)| *value).ok_or_else(|| {
                SciOxideError::numerical_instability(
                    NumericalIssue::Breakdown,
                    "validated weighted support contains no positive mass",
                )
            })
        }
        WeightedQuantileTargetKind::Mass => {
            let mut cumulative_mass = ExactMass::zero();
            for (value, weight) in sorted {
                if *weight == S::ZERO {
                    continue;
                }
                cumulative_mass.add_validated_scalar(*weight);
                if cumulative_mass.compare(target.mass) != Ordering::Less {
                    return Ok(*value);
                }
            }
            last_positive.map(|(value, _)| *value).ok_or_else(|| {
                SciOxideError::numerical_instability(
                    NumericalIssue::Breakdown,
                    "cumulative weight did not reach the requested target",
                )
            })
        }
    }
}

fn positive_support_endpoint<S: RealScalar>(samples: &[(S, S)], minimum: bool) -> S {
    let mut endpoint = samples[0].0;
    let mut seen_positive = false;
    for (value, _) in samples.iter().filter(|(_, weight)| *weight > S::ZERO) {
        if !seen_positive
            || (minimum && finite_total_cmp(value, &endpoint) == Ordering::Less)
            || (!minimum && finite_total_cmp(value, &endpoint) == Ordering::Greater)
        {
            endpoint = *value;
            seen_positive = true;
        }
    }
    endpoint
}

fn weighted_introselect<S: RealScalar>(work_buffer: &mut [(S, S)], mut target: ExactMass) -> S {
    let mut active = work_buffer;
    let mut fast_work_budget = active.len().saturating_mul(2);
    let mut use_fallback = false;
    loop {
        if active.len() == 1 {
            return active[0].0;
        }
        if !use_fallback && fast_work_budget < active.len() {
            use_fallback = true;
            count_fallback();
        }
        let pivot = if use_fallback {
            median_of_medians_value(active)
        } else {
            fast_work_budget -= active.len();
            sampled_pivot_value(active)
        };
        let (less_end, equal_end) = three_way_partition(active, pivot);
        let less_mass = validated_exact_mass(
            active[..less_end]
                .iter()
                .map(|(_, candidate_weight)| *candidate_weight),
        );
        if target.compare(less_mass) != Ordering::Greater {
            active = &mut active[..less_end];
            continue;
        }
        target.subtract_assign(less_mass);
        let equal_mass = validated_exact_mass(
            active[less_end..equal_end]
                .iter()
                .map(|(_, candidate_weight)| *candidate_weight),
        );
        if target.compare(equal_mass) != Ordering::Greater {
            return pivot;
        }
        target.subtract_assign(equal_mass);
        active = &mut active[equal_end..];
    }
}

fn median_of_three_value<S: RealScalar>(samples: &[(S, S)]) -> S {
    let mut left = samples[0].0;
    let mut middle = samples[samples.len() / 2].0;
    let mut right = samples[samples.len() - 1].0;
    if observed_value_cmp(&left, &middle) == Ordering::Greater {
        core::mem::swap(&mut left, &mut middle);
    }
    if observed_value_cmp(&middle, &right) == Ordering::Greater {
        core::mem::swap(&mut middle, &mut right);
    }
    if observed_value_cmp(&left, &middle) == Ordering::Greater {
        core::mem::swap(&mut left, &mut middle);
    }
    middle
}

fn sampled_pivot_value<S: RealScalar>(samples: &[(S, S)]) -> S {
    if samples.len() < 9 {
        return median_of_three_value(samples);
    }
    let last = samples.len() - 1;
    let mut candidates = [samples[0]; 9];
    for (index, candidate) in candidates.iter_mut().enumerate() {
        *candidate = samples[(last / 8) * index + (last % 8) * index / 8];
    }
    insertion_sort_pairs(&mut candidates);
    candidates[4].0
}

fn median_of_medians_value<S: RealScalar>(samples: &mut [(S, S)]) -> S {
    if samples.len() <= 5 {
        insertion_sort_pairs(samples);
        return samples[samples.len() / 2].0;
    }
    let group_count = samples.len().div_ceil(5);
    for group in 0..group_count {
        let start = group * 5;
        let end = (start + 5).min(samples.len());
        insertion_sort_pairs(&mut samples[start..end]);
        samples.swap(group, start + (end - start) / 2);
    }
    deterministic_select_value(&mut samples[..group_count], group_count / 2)
}

fn deterministic_select_value<S: RealScalar>(samples: &mut [(S, S)], index: usize) -> S {
    if samples.len() <= 5 {
        insertion_sort_pairs(samples);
        return samples[index].0;
    }
    let pivot = median_of_medians_value(samples);
    let (less_end, equal_end) = three_way_partition(samples, pivot);
    if index < less_end {
        deterministic_select_value(&mut samples[..less_end], index)
    } else if index < equal_end {
        pivot
    } else {
        deterministic_select_value(&mut samples[equal_end..], index - equal_end)
    }
}

fn insertion_sort_pairs<S: RealScalar>(samples: &mut [(S, S)]) {
    for index in 1..samples.len() {
        let mut current = index;
        while current > 0
            && observed_value_cmp(&samples[current].0, &samples[current - 1].0) == Ordering::Less
        {
            samples.swap(current, current - 1);
            current -= 1;
        }
    }
}

fn three_way_partition<S: RealScalar>(samples: &mut [(S, S)], pivot: S) -> (usize, usize) {
    count_partition();
    let mut less = 0_usize;
    let mut current = 0_usize;
    let mut greater = samples.len();
    while current < greater {
        count_classified_pair();
        match observed_value_cmp(&samples[current].0, &pivot) {
            Ordering::Less => {
                samples.swap(less, current);
                less += 1;
                current += 1;
            }
            Ordering::Equal => current += 1,
            Ordering::Greater => {
                greater -= 1;
                samples.swap(current, greater);
            }
        }
    }
    (less, greater)
}

fn observed_value_cmp<S: RealScalar>(left: &S, right: &S) -> Ordering {
    count_comparison();
    finite_total_cmp(left, right)
}
