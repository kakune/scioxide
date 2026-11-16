//! Sorted empirical distributions and their CDF, quantile and tail-mean queries.

use crate::ranking::finite_total_cmp;
use alloc::vec::Vec;

use scioxide_core::backend::try_with_capacity;
use scioxide_core::{RealScalar, Scaled, SciOxideError};

use super::exact_mass::{ExactMass, validated_exact_mass};
use super::position::quantile_sorted_by;
use super::selection::{evaluate_sorted_weighted_quantile, weighted_quantile_target};
use super::tail::{evaluate_unweighted_tail, evaluate_weighted_tail, unweighted_tail_plan};
use super::{
    QuantileMethod, QuantileProbability, QuantileResult, TailMeanResult, TailSide,
    validate_samples, validate_weighted_samples, weighted_total_cmp,
};

/// Sorted copy of an unweighted sample for repeated queries.
///
/// Construction takes O(n log n) time and O(n) memory. Quantile queries take
/// O(1) time, CDF queries O(log n) and tail means O(n).
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub(crate) struct RawEmpiricalDistribution<S: RealScalar> {
    sorted: Vec<S>,
}

impl<S: RealScalar> RawEmpiricalDistribution<S> {
    /// Copies, checks and sorts a sample.
    ///
    /// # Errors
    ///
    /// Fails for an empty sample, a non-finite observation, or allocation
    /// failure.
    pub(crate) fn new(samples: &[S]) -> Result<Self, SciOxideError> {
        validate_samples(samples, scioxide_core::argument!("samples"))?;
        let mut sorted = try_with_capacity(
            samples.len(),
            scioxide_core::quantity!("empirical_observations"),
        )?;
        sorted.extend_from_slice(samples);
        sorted.sort_unstable_by(finite_total_cmp);
        Ok(Self { sorted })
    }

    /// Returns the number of observations.
    #[must_use]
    #[allow(
        clippy::len_without_is_empty,
        reason = "validated empirical distributions are never empty"
    )]
    pub(crate) const fn len(&self) -> usize {
        self.sorted.len()
    }

    /// Returns the empirical CDF `P(X ≤ value)`.
    ///
    /// # Errors
    /// Fails if `value` is NaN. Infinite values give 0 or 1.
    pub(crate) fn cdf(&self, value: S) -> Result<S, SciOxideError> {
        validate_cdf_query(value)?;
        crate::arithmetic::count_ratio(self.sorted.partition_point(|x| *x <= value), self.len())?
            .round_to()
    }

    /// Returns the empirical survival function `P(X > value)`, counted
    /// directly rather than as `1 − P(X ≤ value)`.
    ///
    /// # Errors
    /// Fails if `value` is NaN.
    pub(crate) fn survival(&self, value: S) -> Result<S, SciOxideError> {
        validate_cdf_query(value)?;
        crate::arithmetic::count_ratio(
            self.len() - self.sorted.partition_point(|x| *x <= value),
            self.len(),
        )?
        .round_to()
    }

    /// Returns the CDF estimate that inverts the quantile definition `method`.
    ///
    /// # Errors
    /// Fails if `value` is NaN or the result is not representable in `S`.
    pub(crate) fn estimated_cdf(
        &self,
        value: S,
        method: QuantileMethod,
    ) -> Result<S, SciOxideError> {
        estimated_cdf_sorted(&self.sorted, value, method)
    }

    /// Returns the sample quantile at `probability` with the Hyndman–Fan
    /// definition `method`.
    ///
    /// # Errors
    ///
    /// Fails if the interpolated result is not representable in `S`.
    pub(crate) fn quantile(
        &self,
        probability: QuantileProbability,
        method: QuantileMethod,
    ) -> Result<QuantileResult<S>, SciOxideError> {
        quantile_sorted_by(self.sorted.len(), probability, method, |index| {
            self.sorted[index]
        })
    }

    /// Returns the mean of the lower tail of weight `n p` or the upper tail of
    /// weight `n (1 − p)`, including the boundary order statistic fractionally.
    ///
    /// # Errors
    ///
    /// Fails for an empty tail (lower at `p = 0`, upper at `p = 1`) or a mean
    /// that is not representable in `S`.
    pub(crate) fn tail_mean(
        &self,
        probability: QuantileProbability,
        side: TailSide,
    ) -> Result<TailMeanResult<S>, SciOxideError> {
        let plan = unweighted_tail_plan(self.sorted.len(), probability, side)?;
        evaluate_unweighted_tail(&self.sorted, probability, side, plan)
    }
}

/// Sorted copy of `(observation, weight)` pairs for repeated queries.
///
/// Pairs with zero weight are kept but contribute nothing.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub(crate) struct RawWeightedEmpiricalDistribution<S: RealScalar> {
    sorted: Vec<(S, S)>,
    total_weight: Option<S>,
    total_mass: ExactMass,
}

impl<S: RealScalar> RawWeightedEmpiricalDistribution<S> {
    /// Copies, checks and sorts weighted observations.
    ///
    /// # Errors
    ///
    /// Fails for an empty input, a non-finite observation or weight, a
    /// negative weight, zero total weight, or allocation failure.
    pub(crate) fn new(samples: &[(S, S)]) -> Result<Self, SciOxideError> {
        let summary = validate_weighted_samples(samples, scioxide_core::argument!("samples"))?;

        let mut sorted = try_with_capacity(
            samples.len(),
            scioxide_core::quantity!("weighted_empirical_observations"),
        )?;
        sorted.extend_from_slice(samples);
        sorted.sort_unstable_by(weighted_total_cmp);
        Ok(Self {
            sorted,
            total_weight: summary.total_weight,
            total_mass: summary.total_mass,
        })
    }

    /// Returns the number of pairs, including those with zero weight.
    #[must_use]
    #[allow(
        clippy::len_without_is_empty,
        reason = "validated weighted empirical distributions are never empty"
    )]
    pub(crate) const fn len(&self) -> usize {
        self.sorted.len()
    }

    /// Returns the total weight, or `None` if it is not representable in `S`.
    #[must_use]
    pub(crate) const fn total_weight(&self) -> Option<S> {
        self.total_weight
    }

    /// Returns the weighted empirical CDF `P(X ≤ value)`.
    ///
    /// # Errors
    /// Fails if `value` is NaN or the probability is not representable.
    pub(crate) fn cdf(&self, value: S) -> Result<S, SciOxideError> {
        self.probability(value, false)
    }

    /// Returns the weighted survival function `P(X > value)`, summed directly.
    ///
    /// # Errors
    /// Fails if `value` is NaN or the probability is not representable.
    pub(crate) fn survival(&self, value: S) -> Result<S, SciOxideError> {
        self.probability(value, true)
    }

    fn probability(&self, value: S, upper: bool) -> Result<S, SciOxideError> {
        validate_cdf_query(value)?;
        let mass = validated_exact_mass(
            self.sorted
                .iter()
                .filter(|(x, _)| (*x > value) == upper)
                .map(|(_, weight)| *weight),
        );
        mass.to_scaled(0)?
            .div(self.total_mass.to_scaled(0)?)?
            .round_to()
    }

    /// Returns the left-continuous inverse of the weighted empirical CDF.
    ///
    /// At `p = 0` this is the smallest observation with positive weight. For
    /// `p > 0` it is the first observation whose cumulative weight reaches
    /// `p W`.
    ///
    /// # Errors
    ///
    /// Fails only on an internal inconsistency between the support and its
    /// cumulative weight.
    pub(crate) fn weighted_inverse_ecdf_quantile(
        &self,
        probability: QuantileProbability,
    ) -> Result<QuantileResult<S>, SciOxideError> {
        let target = weighted_quantile_target(self.total_mass, probability);
        let value = evaluate_sorted_weighted_quantile(&self.sorted, target)?;
        Ok(QuantileResult::weighted(value, probability))
    }

    /// Returns the mean of the lower tail of weight `p W` or the upper tail of
    /// weight `(1 − p) W`, including the boundary weight fractionally.
    ///
    /// # Errors
    ///
    /// Fails for an empty tail or a mean that is not representable in `S`.
    pub(crate) fn tail_mean(
        &self,
        probability: QuantileProbability,
        side: TailSide,
    ) -> Result<TailMeanResult<S>, SciOxideError> {
        evaluate_weighted_tail(&self.sorted, self.total_mass, probability, side)
    }
}

fn validate_cdf_query<S: RealScalar>(value: S) -> Result<(), SciOxideError> {
    if value.is_nan() {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("value"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "NaN query",
        ));
    }
    Ok(())
}

fn estimated_cdf_sorted<S: RealScalar>(
    sorted: &[S],
    value: S,
    method: QuantileMethod,
) -> Result<S, SciOxideError> {
    validate_cdf_query(value)?;
    estimated_cdf_by(
        sorted.len(),
        sorted.partition_point(|x| *x <= value),
        value,
        method,
        |i| sorted[i],
    )
}

pub(crate) fn estimated_cdf_by<S: RealScalar>(
    n: usize,
    rank: usize,
    value: S,
    method: QuantileMethod,
    at: impl Fn(usize) -> S,
) -> Result<S, SciOxideError> {
    validate_cdf_query(value)?;
    if n == 0 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("self"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "empty sample",
        ));
    }
    if value < at(0) {
        return Ok(S::ZERO);
    }
    if value > at(n - 1) {
        return Ok(S::ONE);
    }
    let probability = match method {
        QuantileMethod::NearestRank => rank as f64 / n as f64,
        QuantileMethod::AveragedInverseCdf => (rank as f64 - 0.5) / n as f64,
        QuantileMethod::ClosestObservation => (rank as f64 + 0.5) / n as f64,
        _ => {
            let upper = rank.clamp(usize::from(n > 1), n - 1);
            let lower = upper.saturating_sub(1);
            let delta = if at(upper) > at(lower) {
                Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(value))?
                    .sub(Scaled::<f64>::from_value(
                        scioxide_core::RealScalar::into_f64(at(lower)),
                    )?)?
                    .div(
                        Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(at(upper)))?
                            .sub(Scaled::<f64>::from_value(
                                scioxide_core::RealScalar::into_f64(at(lower)),
                            )?)?,
                    )?
                    .round_to::<f64>()?
            } else {
                1.0
            };
            let (a, b) = match method {
                QuantileMethod::InterpolatedInverseCdf => (0.0, 1.0),
                QuantileMethod::Hazen => (0.5, 0.5),
                QuantileMethod::Weibull => (0.0, 0.0),
                QuantileMethod::MedianUnbiased => (1.0 / 3.0, 1.0 / 3.0),
                QuantileMethod::NormalUnbiased => (3.0 / 8.0, 3.0 / 8.0),
                _ => (1.0, 1.0),
            };
            let denominator = n as f64 + 1.0 - a - b;
            if denominator == 0.0 {
                1.0
            } else {
                (upper as f64 + delta - a) / denominator
            }
        }
    };
    S::try_from_f64_lossy(probability.clamp(0.0, 1.0))
}
