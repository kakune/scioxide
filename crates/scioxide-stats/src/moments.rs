//! Online moments of one variable that can be merged across partitions.

use crate::arithmetic::Shares;
use scioxide_core::{NonFiniteOrigin, RealScalar, SciOxideError};

use crate::MomentOrder;
use crate::central::{Central, compatible};
use crate::{ConfidenceProbability, MeanConfidenceInterval};
use scioxide_core::Scaled;

/// Accumulates the count, mean, extrema and central moments of a stream of observations.
///
/// Each observation counts as one independent unit. [`Self::observe`] updates
/// the state in O(1) time, and [`Self::merge`] combines accumulators of
/// disjoint partitions with the pairwise formulas of Pébay et al. The
/// [`MomentOrder`] selects the highest tracked central moment. The mean is kept
/// as a compensated two-part value, and values outside the normal `f64` range
/// switch to extended-range arithmetic instead of overflowing.
/// [`Self::snapshot`] returns the estimates.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{MomentOrder, OnlineMoments};
/// let mut left = OnlineMoments::from_slice(&[2.0_f64, 4.0, 4.0, 4.0], MomentOrder::Second)?;
/// let right = OnlineMoments::from_slice(&[5.0, 5.0, 7.0, 9.0], MomentOrder::Second)?;
/// left.merge(&right)?;
/// let snapshot = left.snapshot();
/// let mean = snapshot.mean()?.unwrap_or(f64::NAN);
/// let variance = snapshot.sample_variance()?.unwrap_or(f64::NAN);
/// // The eight values have mean 5 and sum of squared deviations 32.
/// assert!((mean - 5.0).abs() < 1e-14);
/// assert!((variance - 32.0 / 7.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. Pébay, T. B. Terriberry, H. Kolla and J. Bennett, "Numerically stable,
/// scalable formulas for parallel and online computation of higher-order
/// multivariate central moments with arbitrary weights", *Computational
/// Statistics* 31, 1305–1325, 2016. doi:10.1007/s00180-015-0637-z
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct OnlineMoments<S: RealScalar> {
    count: usize,
    pub(crate) state: Central<S>,
    order: MomentOrder,
}

impl<S: RealScalar> OnlineMoments<S> {
    /// Creates an empty accumulator that tracks central moments up to `order`.
    pub fn new(order: MomentOrder) -> Self {
        Self {
            count: 0,
            state: Central::empty(),
            order,
        }
    }

    /// Computes the moments of a complete slice.
    ///
    /// When all values lie in the native `f64` range, it makes two compensated
    /// passes: one for the mean about a shifted origin, one for the central
    /// powers. Otherwise it observes the values one by one in extended range.
    /// The cost is O(n) time and O(1) memory. The result can differ in the last
    /// bits from observing the same values one at a time.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::observe`].
    #[inline]
    pub fn from_slice(values: &[S], order: MomentOrder) -> Result<Self, SciOxideError> {
        if let Some((count, central)) = crate::batch::moments(values, order)? {
            return Ok(Self::from_parts(count, central, order));
        }
        values
            .iter()
            .try_fold(Self::new(order), |mut state, &value| {
                state.observe(value)?;
                Ok(state)
            })
    }

    /// Returns the highest tracked moment order.
    pub const fn moment_order(&self) -> MomentOrder {
        self.order
    }

    /// Returns whether `other` tracks the same moment order and so can be merged.
    pub fn is_merge_compatible(&self, other: &Self) -> bool {
        self.order == other.order
    }

    pub(crate) fn from_parts(count: usize, state: Central<S>, order: MomentOrder) -> Self {
        Self {
            count,
            state,
            order,
        }
    }

    /// Returns the number of observations.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Returns whether the accumulator has no observations.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Adds one observation.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `value` is NaN or infinite.
    /// - `SizeOverflow` if the count overflows `usize`.
    /// - `NumericalInstability` if an even central moment becomes negative
    ///   through cancellation, or a range error of the extended arithmetic.
    #[inline(always)]
    pub fn observe(&mut self, value: S) -> Result<(), SciOxideError> {
        if !value.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "observation must be finite",
            ));
        }
        let next_count = self.count.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!(
                "onlinemoments_observation_count"
            ))
        })?;

        let state = if self.is_empty() {
            Central::singleton(value)?
        } else {
            self.state.observed(value, next_count, self.order)?
        };

        *self = Self {
            count: next_count,
            state,
            order: self.order,
        };
        Ok(())
    }

    /// Merges the accumulator of a disjoint partition into this one.
    ///
    /// Merging an empty accumulator changes nothing.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the moment orders differ.
    /// - `SizeOverflow` if the combined count overflows `usize`.
    /// - The numerical errors of [`Self::observe`].
    pub fn merge(&mut self, other: &Self) -> Result<(), SciOxideError> {
        compatible(self.is_merge_compatible(other))?;
        if other.is_empty() {
            return Ok(());
        }
        if self.is_empty() {
            *self = *other;
            return Ok(());
        }
        let count = self.count.checked_add(other.count).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("onlinemoments_merged_count"))
        })?;
        let shares = Shares::counts(self.count, other.count, count)?;
        let state = self.state.merged(other.state, shares, self.order)?;
        *self = Self {
            count,
            state,
            order: self.order,
        };
        Ok(())
    }

    /// Returns the current estimates as an immutable snapshot.
    ///
    /// The snapshot keeps the extended-range state. Each estimate is rounded to
    /// `S` only when you ask for it, so one estimate can overflow while others
    /// remain available.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_stats::{MomentOrder, OnlineMoments};
    /// let snapshot = OnlineMoments::from_slice(&[-1e154_f64, 1e154], MomentOrder::Second)?
    ///     .snapshot();
    /// assert_eq!(snapshot.mean()?, Some(0.0));
    /// // The sample variance 2·10³⁰⁸ overflows f64; the standard error 10¹⁵⁴ does not.
    /// assert!(snapshot.sample_variance().is_err());
    /// assert!(snapshot.standard_error_of_sample_mean()?.is_some());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn snapshot(&self) -> MomentsSnapshot<S> {
        MomentsSnapshot {
            state: self.state,
            order: self.order,
            count: self.count,
        }
    }
}

/// Immutable estimates from an [`OnlineMoments`] accumulator.
///
/// Estimates that are undefined for the current count return `Ok(None)`; see
/// [Results and termination](crate#results-and-termination).
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct MomentsSnapshot<S: RealScalar> {
    pub(crate) state: Central<S>,
    pub(crate) order: MomentOrder,
    count: usize,
}

impl<S: RealScalar> MomentsSnapshot<S> {
    /// Returns the number of observations `n`.
    #[must_use]
    pub const fn count(self) -> usize {
        self.count
    }

    /// Returns the arithmetic mean `x̄ = Σ xᵢ / n`, or `None` if `n = 0`.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the mean cannot be represented in `S`.
    pub fn mean(self) -> Result<Option<S>, SciOxideError> {
        if self.count == 0 {
            Ok(None)
        } else if let Some([high, low]) = self.state.native_center_parts() {
            S::try_from_f64_lossy(high + low).map(Some)
        } else {
            self.state.mean()?.total()?.round_to().map(Some)
        }
    }

    /// Returns the population variance `Σ (xᵢ − x̄)² / n`, or `None` if `n = 0`.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the variance overflows `S`.
    pub fn population_variance(self) -> Result<Option<S>, SciOxideError> {
        self.variance(0)
    }

    /// Returns the unbiased sample variance `s² = Σ (xᵢ − x̄)² / (n − 1)`, or `None` if `n < 2`.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the variance overflows `S`.
    pub fn sample_variance(self) -> Result<Option<S>, SciOxideError> {
        self.variance(1)
    }

    /// Returns the estimated variance of the sample mean, `s² / n`, or `None` if `n < 2`.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the result overflows `S`.
    pub fn variance_of_sample_mean(self) -> Result<Option<S>, SciOxideError> {
        self.mean_variance_scaled()?
            .map(Scaled::round_to)
            .transpose()
    }

    /// Computes the estimated variance of the sample mean, `s²/n`, with an
    /// extended binary exponent, or `None` if `n < 2`.
    ///
    /// Here `n` is the observation count and `s²` is the unbiased sample
    /// variance. Divides the retained second central moment in `O(1)` work
    /// without allocating. Only the significand is rounded to `S`; the variance
    /// remains nonzero even when its square root rounds to zero in `S`.
    /// Accuracy follows the accumulated moments; conversion of the normalized
    /// significand contributes at most half an ulp.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if extended-range arithmetic fails.
    ///
    /// ```
    /// use scioxide_stats::{MomentOrder, OnlineMoments};
    /// let moments = OnlineMoments::from_slice(&[-1e200_f64, 1e200], MomentOrder::Second)?;
    /// let variance = moments.snapshot().scaled_variance_of_sample_mean()?.unwrap();
    /// assert!(variance.round().is_err());
    /// assert!((variance.sqrt()?.round()? / 1e200 - 1.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// G. Casella and R. L. Berger, *Statistical Inference*, 2nd ed.,
    /// Duxbury, 2002, Section 5.2.
    pub fn scaled_variance_of_sample_mean(self) -> Result<Option<Scaled<S>>, SciOxideError> {
        self.mean_variance_scaled()?
            .map(|value| {
                Scaled::from_parts(
                    S::try_from_f64_lossy(value.significand())?,
                    value.exponent(),
                )
            })
            .transpose()
    }

    fn mean_variance_scaled(self) -> Result<Option<Scaled<f64>>, SciOxideError> {
        self.variance_scaled(1)?
            .map(|value| value.div(Scaled::from_value(self.count as f64)?))
            .transpose()
    }

    /// Returns the standard error of the sample mean, `s / √n`, or `None` if `n < 2`.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the result overflows `S`.
    pub fn standard_error_of_sample_mean(self) -> Result<Option<S>, SciOxideError> {
        self.standard_error(1)
    }

    /// Computes the result of [`Self::confidence_interval_of_mean_with_options`] with the [one-shot defaults](crate#one-shot-forms).
    ///
    /// Uses the same numerical method, cost and accuracy as the configured form.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::confidence_interval_of_mean_with_options`] for the fixed default settings.
    pub fn confidence_interval_of_mean(
        self,
    ) -> Result<Option<MeanConfidenceInterval<S>>, SciOxideError> {
        self.confidence_interval_of_mean_with_options(
            ConfidenceProbability::default(),
            crate::MeanIntervalMethod::StudentT,
            Default::default(),
        )
    }
    /// Computes a two-sided confidence interval `x̄ ± c·s/√n` for the mean.
    ///
    /// [`MeanIntervalMethod::StudentT`](crate::MeanIntervalMethod::StudentT)
    /// uses `n − 1` degrees of freedom and assumes independent normal
    /// observations. `options` controls the inverse incomplete beta function
    /// for the Student's t critical value. Returns `Ok(None)` if `n < 2`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `method` is
    ///   [`NormalKnownVariance`](crate::MeanIntervalMethod::NormalKnownVariance), because the
    ///   variance is estimated.
    /// - The errors of [`Self::standard_error_of_sample_mean`].
    /// - The errors of [`MeanIntervalPlan::new`](crate::MeanIntervalPlan::new) and
    ///   [`MeanIntervalPlan::interval`](crate::MeanIntervalPlan::interval).
    pub fn confidence_interval_of_mean_with_options(
        self,
        probability: ConfidenceProbability,
        method: crate::MeanIntervalMethod,
        options: scioxide_special::BetaOptions,
    ) -> Result<Option<MeanConfidenceInterval<S>>, SciOxideError> {
        if method == crate::MeanIntervalMethod::NormalKnownVariance {
            return Err(crate::error_helpers::invalid(
                scioxide_core::argument!("method"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "estimated variance cannot use known-variance inference",
            ));
        }

        self.mean()?
            .zip(self.standard_error_of_sample_mean()?)
            .map(|(mean, se)| {
                crate::confidence::estimated_interval(
                    mean,
                    se,
                    probability,
                    method,
                    (self.count - 1) as f64,
                    crate::MeanIntervalAssumptions::IndependentNormal,
                    options,
                )
            })
            .transpose()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {

    use super::*;

    #[test]
    fn empty_and_singleton_undefined_estimates_are_absent() {
        let mut moments = OnlineMoments::<f64>::new(crate::MomentOrder::Second);
        let empty = moments.snapshot();
        assert_eq!(empty.mean().unwrap(), None);
        assert_eq!(empty.population_variance().unwrap(), None);
        assert_eq!(empty.sample_variance().unwrap(), None);
        assert_eq!(empty.variance_of_sample_mean().unwrap(), None);
        assert_eq!(empty.scaled_variance_of_sample_mean().unwrap(), None);

        moments.observe(4.0).unwrap();
        let singleton = moments.snapshot();
        assert_eq!(singleton.mean().unwrap(), Some(4.0));
        assert_eq!(singleton.population_variance().unwrap(), Some(0.0));
        assert_eq!(singleton.sample_variance().unwrap(), None);
        assert_eq!(singleton.standard_error_of_sample_mean().unwrap(), None);
        assert_eq!(singleton.scaled_variance_of_sample_mean().unwrap(), None);
    }

    #[test]
    fn scaled_mean_variance_keeps_the_range_of_the_central_moment() {
        fn check<S: RealScalar>() {
            for scale in [
                S::ZERO,
                S::ZERO.next_after(S::ONE),
                S::ONE,
                S::MAX * S::HALF,
            ] {
                let moments = OnlineMoments::from_slice(&[-scale, scale], MomentOrder::Second)
                    .unwrap()
                    .snapshot();
                let variance = moments.scaled_variance_of_sample_mean().unwrap().unwrap();
                if scale == S::ZERO {
                    assert!(variance.is_zero());
                } else {
                    let expected = Scaled::from_value(scale).unwrap().square().unwrap();
                    let ratio = variance.div(expected).unwrap().round().unwrap();
                    assert!(
                        (ratio - S::ONE).abs() <= S::try_from_usize_exact(4).unwrap() * S::EPSILON
                    );
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn published_moment_example_distinguishes_all_variances() {
        let mut moments = OnlineMoments::new(crate::MomentOrder::Second);
        [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
            .into_iter()
            .try_for_each(|value| moments.observe(value))
            .unwrap();
        let snapshot = moments.snapshot();
        assert_eq!(snapshot.mean().unwrap(), Some(5.0));
        assert_eq!(snapshot.population_variance().unwrap(), Some(4.0));
        assert!((snapshot.sample_variance().unwrap().unwrap() - 32.0 / 7.0).abs() < 1.0e-14);
        assert!((snapshot.variance_of_sample_mean().unwrap().unwrap() - 4.0 / 7.0).abs() < 1.0e-14);
        assert!(
            (snapshot.standard_error_of_sample_mean().unwrap().unwrap() - (4.0_f64 / 7.0).sqrt())
                .abs()
                < 1.0e-14
        );
    }

    #[test]
    fn partition_merge_agrees_with_single_pass() {
        let values = [1.0_f64, 2.0, 4.0, 8.0, 16.0, 32.0];
        let mut single = OnlineMoments::new(crate::MomentOrder::Second);
        values
            .into_iter()
            .try_for_each(|value| single.observe(value))
            .unwrap();

        let mut left = OnlineMoments::new(crate::MomentOrder::Second);
        values[..3]
            .iter()
            .copied()
            .try_for_each(|value| left.observe(value))
            .unwrap();
        let mut right = OnlineMoments::new(crate::MomentOrder::Second);
        values[3..]
            .iter()
            .copied()
            .try_for_each(|value| right.observe(value))
            .unwrap();
        left.merge(&right).unwrap();

        let expected = single.snapshot();
        let actual = left.snapshot();
        assert!(
            (actual.mean().unwrap().unwrap() - expected.mean().unwrap().unwrap()).abs() < 1.0e-14
        );
        assert!(
            (actual.sample_variance().unwrap().unwrap()
                - expected.sample_variance().unwrap().unwrap())
            .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn invalid_observation_and_overflow_do_not_mutate() {
        let mut moments = OnlineMoments::new(crate::MomentOrder::Second);
        moments.observe(2.0_f64).unwrap();
        let before = moments;
        assert!(
            matches ! (& (moments . observe (f64 :: NAN)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
        );
        assert_eq!(moments, before);

        let mut overflow = OnlineMoments {
            count: usize::MAX,
            state: Central::singleton(2.0_f64).unwrap(),
            order: MomentOrder::Second,
        };
        let before = overflow;
        assert!(
            matches ! (& (overflow . observe (2.0)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: SizeOverflow { .. }))
        );
        assert_eq!(overflow, before);
    }

    #[test]
    fn unrepresentable_variance_is_rejected_only_on_query() {
        let mut moments = OnlineMoments::new(crate::MomentOrder::Fourth);
        moments.observe(-f64::MAX).unwrap();
        moments.observe(f64::MAX).unwrap();
        let snapshot = moments.snapshot();
        assert_eq!(snapshot.count(), 2);
        assert_eq!(snapshot.mean().unwrap(), Some(0.0));
        assert!(snapshot.population_variance().is_err());
        assert!(snapshot.variance_of_sample_mean().is_err());
        assert!(snapshot.standard_deviation(1).is_err());
        assert_eq!(snapshot.standard_deviation(0).unwrap(), Some(f64::MAX));
        assert_eq!(snapshot.standard_error(1).unwrap(), Some(f64::MAX));
    }

    #[test]
    fn extreme_singletons_merge_without_forming_delta_squared() {
        let mut single = OnlineMoments::new(crate::MomentOrder::Second);
        [0.0_f64, 1.5e154]
            .into_iter()
            .try_for_each(|value| single.observe(value))
            .unwrap();
        let mut left = OnlineMoments::new(crate::MomentOrder::Second);
        left.observe(0.0).unwrap();
        let mut right = OnlineMoments::new(crate::MomentOrder::Second);
        right.observe(1.5e154).unwrap();
        left.merge(&right).unwrap();
        assert_eq!(left.snapshot(), single.snapshot());
    }

    #[test]
    fn high_counts_do_not_require_exact_scalar_conversion() {
        let mut high_count = OnlineMoments {
            count: 16_777_216,
            state: Central::singleton(2.0_f32).unwrap(),
            order: MomentOrder::Second,
        };
        high_count.observe(2.0).unwrap();
        assert_eq!(high_count.count(), 16_777_217);
        assert_eq!(
            high_count.snapshot().population_variance().unwrap(),
            Some(0.0)
        );

        if let Ok(count) = usize::try_from(1_u64 << 53) {
            let mut high_count = OnlineMoments {
                count,
                state: Central::singleton(2.0_f64).unwrap(),
                order: MomentOrder::Second,
            };
            high_count.observe(2.0).unwrap();
            assert_eq!(high_count.count(), count + 1);
            assert_eq!(
                high_count.snapshot().population_variance().unwrap(),
                Some(0.0)
            );
        }

        let mut overflow = OnlineMoments {
            count: usize::MAX,
            state: Central::singleton(2.0_f64).unwrap(),
            order: MomentOrder::Second,
        };
        let other = OnlineMoments {
            count: 1,
            state: Central::singleton(3.0).unwrap(),
            order: MomentOrder::Second,
        };
        let before = overflow;
        assert!(
            matches ! (& (overflow . merge (& other)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: SizeOverflow { .. }))
        );
        assert_eq!(overflow, before);
    }

    #[test]
    fn rounded_zero_second_moments_retain_a_representable_standard_error() {
        let minimum_subnormal = f32::from_bits(1);
        let mut moments = OnlineMoments::new(crate::MomentOrder::Second);
        [0.0_f32, minimum_subnormal + minimum_subnormal]
            .into_iter()
            .try_for_each(|value| moments.observe(value))
            .unwrap();
        let snapshot = moments.snapshot();
        assert_eq!(snapshot.population_variance().unwrap(), Some(0.0));
        assert_eq!(snapshot.sample_variance().unwrap(), Some(0.0));
        assert_eq!(snapshot.variance_of_sample_mean().unwrap(), Some(0.0));
        assert_eq!(
            snapshot.standard_error_of_sample_mean().unwrap(),
            Some(minimum_subnormal)
        );
    }
}
