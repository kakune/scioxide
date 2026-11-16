//! Online covariance of two variables that can be merged across partitions.

use scioxide_core::{NonFiniteOrigin, NumericalIssue, RealScalar, SciOxideError};

use crate::arithmetic::Shares;
use crate::arithmetic::center_difference;
use crate::arithmetic::count_ratio;
use crate::central::{Central, compatible, merge_cross};
use crate::{MomentOrder, MomentsSnapshot, OnlineMoments};
use scioxide_core::Scaled;

/// Accumulates the means, marginal moments and covariance of a stream of pairs `(x, y)`.
///
/// Each pair counts as one independent unit. The accumulator tracks the
/// marginal states of `x` and `y`, as [`OnlineMoments`] does, and the
/// co-moment `c = Σ (xᵢ − x̄)(yᵢ − ȳ) / n`. Updates and merges use the pairwise
/// formulas of Pébay et al. (2016). [`Self::snapshot`] returns the estimates.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{MomentOrder, OnlineCovariance};
/// let x = [2.1_f64, 2.5, 4.0, 3.6];
/// let y = [8.0, 12.0, 14.0, 10.0];
/// let covariance = OnlineCovariance::from_slices(&x, &y, MomentOrder::Second)?;
/// let snapshot = covariance.snapshot();
/// let sample = snapshot.sample_covariance()?.unwrap_or(f64::NAN);
/// // Σ (xᵢ − 3.05)(yᵢ − 11) = 4.6, divided by n − 1 = 3.
/// assert!((sample - 4.6 / 3.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct OnlineCovariance<S: RealScalar> {
    count: usize,
    x: Central<S>,
    y: Central<S>,
    order: MomentOrder,
    population_covariance: Scaled<f64>,
}

impl<S: RealScalar> OnlineCovariance<S> {
    /// Creates an empty accumulator whose marginals track central moments up to `order`.
    pub fn new(order: MomentOrder) -> Self {
        Self {
            count: 0,
            x: Central::empty(),
            y: Central::empty(),
            order,
            population_covariance: Scaled::<f64>::ZERO,
        }
    }

    /// Computes the paired moments of two complete slices.
    ///
    /// When all values lie in the native `f64` range, it makes two compensated
    /// passes: one for the means, one for the central powers and co-moment.
    /// If the co-moment suffers severe cancellation, it is recomputed with exact
    /// products. Otherwise it observes the pairs one by one in extended range.
    /// The cost is O(n) time and O(1) memory. The result can differ in the last
    /// bits from observing the pairs one at a time.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `x` and `y` differ in length.
    /// - The errors of [`Self::observe`]. Overflow of a marginal estimate is
    ///   reported only when that estimate is queried.
    pub fn from_slices(x: &[S], y: &[S], order: MomentOrder) -> Result<Self, SciOxideError> {
        if x.len() != y.len() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(x.len()),
                scioxide_core::Dimensions::one(y.len()),
            ));
        }
        if let Some(batch) = crate::batch::paired(x, y, order)? {
            return Ok(Self {
                count: x.len(),
                x: batch.x,
                y: batch.y,
                order,
                population_covariance: batch.covariance,
            });
        }
        x.iter()
            .zip(y)
            .try_fold(Self::new(order), |mut state, (&x, &y)| {
                state.observe(x, y)?;
                Ok(state)
            })
    }
    /// Returns the highest tracked marginal moment order.
    pub const fn moment_order(&self) -> MomentOrder {
        self.order
    }
    /// Returns whether `other` tracks the same moment order and so can be merged.
    pub fn is_merge_compatible(&self, other: &Self) -> bool {
        self.order == other.order
    }
    /// Returns the number of pairs.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Returns whether the accumulator has no pairs.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Adds one pair `(x, y)`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `x` or `y` is NaN or infinite.
    /// - `SizeOverflow` if the count overflows `usize`.
    /// - The numerical errors of [`OnlineMoments::observe`].
    #[inline(always)]
    pub fn observe(&mut self, x: S, y: S) -> Result<(), SciOxideError> {
        validate_pair(x, y)?;
        let next_count = self.count.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!(
                "onlinecovariance_observation_count"
            ))
        })?;
        let singleton = Self {
            count: 1,
            x: Central::singleton(x)?,
            y: Central::singleton(y)?,
            order: self.order,
            population_covariance: Scaled::<f64>::ZERO,
        };
        let candidate = if self.is_empty() {
            singleton
        } else {
            self.merged(&singleton, next_count)?
        };
        *self = candidate;
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
            SciOxideError::size_overflow(scioxide_core::quantity!("covariance_merged_count"))
        })?;
        *self = self.merged(other, count)?;
        Ok(())
    }
    #[inline(always)]
    fn merged(&self, other: &Self, count: usize) -> Result<Self, SciOxideError> {
        let shares = Shares::counts(self.count, other.count, count)?;
        let [p, q] = [shares.p[0], shares.q[0]];
        let covariance = merge_cross(
            self.population_covariance,
            other.population_covariance,
            (self.x.mean()?, self.y.mean()?),
            (other.x.mean()?, other.y.mean()?),
            p,
            q,
        )?;
        let x = self.x.merged(other.x, shares, self.order)?;
        let y = self.y.merged(other.y, shares, self.order)?;

        Ok(Self {
            count,
            x,
            y,
            order: self.order,
            population_covariance: covariance,
        })
    }
    /// Returns the current estimates as an immutable snapshot.
    ///
    /// Each estimate is rounded to `S` only when you ask for it.
    pub const fn snapshot(&self) -> CovarianceSnapshot<S> {
        CovarianceSnapshot {
            x: self.x,
            y: self.y,
            order: self.order,
            covariance: self.population_covariance,
            count: self.count,
        }
    }
}

/// Immutable estimates from an [`OnlineCovariance`] accumulator.
///
/// `n` is the number of pairs and `c = Σ (xᵢ − x̄)(yᵢ − ȳ) / n` the co-moment.
/// Estimates that are undefined for the current count return `Ok(None)`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct CovarianceSnapshot<S: RealScalar> {
    x: Central<S>,
    y: Central<S>,
    order: MomentOrder,
    pub(crate) covariance: Scaled<f64>,
    count: usize,
}

impl<S: RealScalar> CovarianceSnapshot<S> {
    /// Returns the least-squares slope and intercept of `y` on `x` in extended range.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` for fewer than two pairs or a constant `x`.
    pub(crate) fn least_squares_line(self) -> Result<(Scaled<f64>, Scaled<f64>), SciOxideError> {
        let x = self.marginal_x().state;
        if self.count() < 2 || x.m2()?.is_zero() {
            return Err(crate::error_helpers::invalid(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "constant predictor",
            ));
        }
        let slope = self.covariance.div(x.m2()?)?;
        let intercept = center_difference(
            x.mean()?.multiplied(slope)?,
            self.marginal_y().state.mean()?,
        )?;
        Ok((slope, intercept))
    }
    /// Returns the moments of `x` over the same pairs.
    pub fn marginal_x(self) -> MomentsSnapshot<S> {
        OnlineMoments::from_parts(self.count, self.x, self.order).snapshot()
    }

    /// Returns the moments of `y` over the same pairs.
    pub fn marginal_y(self) -> MomentsSnapshot<S> {
        OnlineMoments::from_parts(self.count, self.y, self.order).snapshot()
    }

    /// Returns which covariance estimates are defined for the current count.
    pub const fn validity(self) -> CovarianceValidity {
        match self.count {
            0 => CovarianceValidity::Empty,
            1 => CovarianceValidity::PopulationOnly,
            _ => CovarianceValidity::SampleDefined,
        }
    }

    /// Returns whether the correlation is defined and, if not, why.
    pub fn correlation_validity(self) -> CorrelationValidity {
        correlation_validity(self.count, self.x.is_constant(), self.y.is_constant())
    }

    /// Returns the Pearson correlation `r = c / √(m₂ₓ m₂ᵧ)`.
    ///
    /// `m₂ₓ` and `m₂ᵧ` are the population variances of `x` and `y`. The ratio
    /// is formed from the unrounded moments and clamped to `[−1, 1]`. Returns
    /// `None` unless [`Self::correlation_validity`] is
    /// [`CorrelationValidity::Defined`].
    ///
    /// # Errors
    ///
    /// Returns `NumericalInstability` if `|r|` exceeds `1 + 128ε`, which means
    /// the co-moment lost accuracy to cancellation.
    pub fn correlation(self) -> Result<Option<S>, SciOxideError> {
        correlation(self.count, self.x.m2()?, self.y.m2()?, self.covariance)
    }
    /// Returns the number of pairs `n`.
    #[must_use]
    pub const fn count(self) -> usize {
        self.count
    }

    /// Returns the mean `x̄`, or `None` if `n = 0`.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the mean cannot be represented in `S`.
    pub fn mean_x(self) -> Result<Option<S>, SciOxideError> {
        self.marginal_x().mean()
    }

    /// Returns the mean `ȳ`, or `None` if `n = 0`.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the mean cannot be represented in `S`.
    pub fn mean_y(self) -> Result<Option<S>, SciOxideError> {
        self.marginal_y().mean()
    }

    /// Returns the population covariance `c`, or `None` if `n = 0`.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if `c` overflows `S`.
    pub fn population_covariance(self) -> Result<Option<S>, SciOxideError> {
        self.covariance_scaled(0)?
            .map(|value| value.round_to())
            .transpose()
    }

    /// Returns the unbiased sample covariance `n c / (n − 1)`, or `None` if `n < 2`.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the result overflows `S`.
    pub fn sample_covariance(self) -> Result<Option<S>, SciOxideError> {
        self.covariance_scaled(1)?
            .map(|value| value.round_to())
            .transpose()
    }

    /// Returns the estimated covariance of the sample means `x̄` and `ȳ`, `c / (n − 1)`.
    ///
    /// This is the sample covariance divided by `n`. Returns `None` if `n < 2`.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the result overflows `S`.
    pub fn covariance_of_sample_means(self) -> Result<Option<S>, SciOxideError> {
        self.covariance_scaled(1)?
            .map(|value| {
                value
                    .div(Scaled::from_value(self.count as f64)?)?
                    .round_to()
            })
            .transpose()
    }
    fn covariance_scaled(self, ddof: usize) -> Result<Option<Scaled<f64>>, SciOxideError> {
        if self.count <= ddof {
            return Ok(None);
        }
        self.covariance
            .mul(count_ratio(self.count, self.count - ddof)?)
            .map(Some)
    }
}

/// Which covariance estimates are defined for a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovarianceValidity {
    /// There are no pairs.
    Empty,
    /// The population covariance is defined. The sample covariance is not,
    /// because there are too few pairs or the weight model has none.
    PopulationOnly,
    /// The sample covariance is defined for the weight model.
    SampleDefined,
}

/// Whether a Pearson correlation is defined and, if not, why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationValidity {
    /// There are fewer than two pairs.
    InsufficientPairs,
    /// Both coordinates are constant.
    ConstantBoth,
    /// The first coordinate is constant.
    ConstantX,
    /// The second coordinate is constant.
    ConstantY,
    /// Both marginal variances are positive.
    Defined,
}

pub(crate) fn correlation_validity(count: usize, x: bool, y: bool) -> CorrelationValidity {
    if count < 2 {
        return CorrelationValidity::InsufficientPairs;
    }
    match (x, y) {
        (true, true) => CorrelationValidity::ConstantBoth,
        (true, false) => CorrelationValidity::ConstantX,
        (false, true) => CorrelationValidity::ConstantY,
        (false, false) => CorrelationValidity::Defined,
    }
}

pub(crate) fn correlation<S: RealScalar>(
    count: usize,
    x: Scaled<f64>,
    y: Scaled<f64>,
    covariance: Scaled<f64>,
) -> Result<Option<S>, SciOxideError> {
    if correlation_validity(count, x.is_zero(), y.is_zero()) != CorrelationValidity::Defined {
        return Ok(None);
    }
    let r: S = covariance.div(x.sqrt()?)?.div(y.sqrt()?)?.round_to()?;
    // A scalar-precision rounding envelope, not a relaxation for invalid covariance.
    let envelope = S::try_from_f64_lossy(128.0)? * <S as RealScalar>::EPSILON;
    if r.abs() > S::ONE + envelope {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Cancellation,
            "covariance violates the Cauchy-Schwarz bound",
        ));
    }
    Ok(Some(r.max(-S::ONE).min(S::ONE)))
}

fn validate_pair<S: RealScalar>(x: S, y: S) -> Result<(), SciOxideError> {
    if !x.is_finite() || !y.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "both coordinates must be finite",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {

    use super::*;

    #[test]
    fn empty_and_singleton_covariances_are_explicit() {
        let mut covariance = OnlineCovariance::<f64>::new(crate::MomentOrder::Second);
        let empty = covariance.snapshot();
        assert_eq!(empty.mean_x().unwrap(), None);
        assert_eq!(empty.population_covariance().unwrap(), None);

        covariance.observe(2.0, 4.0).unwrap();
        let singleton = covariance.snapshot();
        assert_eq!(singleton.mean_x().unwrap(), Some(2.0));
        assert_eq!(singleton.mean_y().unwrap(), Some(4.0));
        assert_eq!(singleton.population_covariance().unwrap(), Some(0.0));
        assert_eq!(singleton.sample_covariance().unwrap(), None);
    }

    #[test]
    fn published_covariance_example_matches_reference() {
        let mut covariance = OnlineCovariance::new(crate::MomentOrder::Second);
        [(2.1_f64, 8.0), (2.5, 12.0), (4.0, 14.0), (3.6, 10.0)]
            .into_iter()
            .try_for_each(|(x, y)| covariance.observe(x, y))
            .unwrap();
        let snapshot = covariance.snapshot();
        assert!((snapshot.mean_x().unwrap().unwrap() - 3.05).abs() < 1.0e-14);
        assert!((snapshot.mean_y().unwrap().unwrap() - 11.0).abs() < 1.0e-14);
        assert!(
            (snapshot.sample_covariance().unwrap().unwrap() - 1.533_333_333_333_333_4).abs()
                < 1.0e-14
        );
        assert!(
            (snapshot.covariance_of_sample_means().unwrap().unwrap() - 0.383_333_333_333_333_36)
                .abs()
                < 1.0e-14
        );
    }

    #[test]
    fn swapping_coordinates_preserves_covariance() {
        let pairs = [(-2.0_f64, 5.0), (1.0, 3.0), (7.0, -4.0), (9.0, 8.0)];
        let mut xy = OnlineCovariance::new(crate::MomentOrder::Second);
        let mut yx = OnlineCovariance::new(crate::MomentOrder::Second);
        pairs
            .into_iter()
            .try_for_each(|(x, y)| {
                xy.observe(x, y)?;
                yx.observe(y, x)
            })
            .unwrap();
        assert_eq!(
            xy.snapshot().sample_covariance().unwrap(),
            yx.snapshot().sample_covariance().unwrap()
        );
    }

    #[test]
    fn zero_covariance_survives_an_unrepresentable_coordinate_delta() {
        let mut covariance = OnlineCovariance::new(crate::MomentOrder::Second);
        [(-f64::MAX, 1.0), (f64::MAX, 1.0)]
            .into_iter()
            .try_for_each(|(x, y)| covariance.observe(x, y))
            .unwrap();
        let snapshot = covariance.snapshot();
        assert_eq!(snapshot.mean_x().unwrap(), Some(0.0));
        assert_eq!(snapshot.population_covariance().unwrap(), Some(0.0));
        assert_eq!(snapshot.sample_covariance().unwrap(), Some(0.0));
    }

    #[test]
    fn partition_merge_matches_single_pass() {
        let pairs = [(1.0_f64, 2.0), (2.0, 3.0), (5.0, -1.0), (8.0, 7.0)];
        let mut single = OnlineCovariance::new(crate::MomentOrder::Second);
        pairs
            .into_iter()
            .try_for_each(|(x, y)| single.observe(x, y))
            .unwrap();
        let mut left = OnlineCovariance::new(crate::MomentOrder::Second);
        pairs[..2]
            .iter()
            .copied()
            .try_for_each(|(x, y)| left.observe(x, y))
            .unwrap();
        let mut right = OnlineCovariance::new(crate::MomentOrder::Second);
        pairs[2..]
            .iter()
            .copied()
            .try_for_each(|(x, y)| right.observe(x, y))
            .unwrap();
        left.merge(&right).unwrap();
        let expected = single.snapshot().sample_covariance().unwrap().unwrap();
        let actual = left.snapshot().sample_covariance().unwrap().unwrap();
        assert!((actual - expected).abs() < 1.0e-14);
    }

    #[test]
    fn invalid_pair_and_overflow_leave_state_unchanged() {
        let mut covariance = OnlineCovariance::new(crate::MomentOrder::Second);
        covariance.observe(1.0_f64, 2.0).unwrap();
        let before = covariance;
        assert!(
            matches ! (& (covariance . observe (f64 :: NAN , 3.0)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
        );
        assert_eq!(covariance, before);

        let mut overflow = OnlineCovariance {
            count: usize::MAX,
            ..covariance
        };
        let before = overflow;
        assert!(
            matches ! (& (overflow . observe (3.0 , 4.0)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: SizeOverflow { .. }))
        );
        assert_eq!(overflow, before);
    }

    #[test]
    fn merge_count_overflow_leaves_state_unchanged() {
        let mut overflow = OnlineCovariance {
            count: usize::MAX,
            x: Central::singleton(1.0_f64).unwrap(),
            y: Central::singleton(2.0).unwrap(),
            order: MomentOrder::Second,
            population_covariance: Scaled::<f64>::ZERO,
        };
        let other = OnlineCovariance {
            count: 1,
            x: Central::singleton(3.0).unwrap(),
            y: Central::singleton(4.0).unwrap(),
            order: MomentOrder::Second,
            population_covariance: Scaled::<f64>::ZERO,
        };
        let before = overflow;
        assert!(
            matches ! (& (overflow . merge (& other)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: SizeOverflow { .. }))
        );
        assert_eq!(overflow, before);
    }

    #[test]
    fn high_counts_do_not_require_scalar_conversion() {
        let mut f32_state = OnlineCovariance {
            count: 16_777_216,
            x: Central::singleton(2.0_f32).unwrap(),
            y: Central::singleton(3.0).unwrap(),
            order: MomentOrder::Second,
            population_covariance: Scaled::<f64>::ZERO,
        };
        f32_state.observe(2.0, 3.0).unwrap();
        assert_eq!(f32_state.count(), 16_777_217);

        if let Ok(count) = usize::try_from(1_u64 << 53) {
            let mut f64_state = OnlineCovariance {
                count,
                x: Central::singleton(2.0_f64).unwrap(),
                y: Central::singleton(3.0).unwrap(),
                order: MomentOrder::Second,
                population_covariance: Scaled::<f64>::ZERO,
            };
            f64_state.observe(2.0, 3.0).unwrap();
            assert_eq!(f64_state.count(), count + 1);
        }
    }

    #[test]
    fn covariance_underflow_preserves_the_correct_zero_sign() {
        let minimum_subnormal = f32::from_bits(1);
        let delta = minimum_subnormal + minimum_subnormal;
        let mut covariance = OnlineCovariance::new(crate::MomentOrder::Second);
        [(0.0_f32, 0.0_f32), (delta, -delta)]
            .into_iter()
            .try_for_each(|(x, y)| covariance.observe(x, y))
            .unwrap();
        let snapshot = covariance.snapshot();
        for value in [
            snapshot.population_covariance().unwrap().unwrap(),
            snapshot.sample_covariance().unwrap().unwrap(),
            snapshot.covariance_of_sample_means().unwrap().unwrap(),
        ] {
            assert_eq!(value, 0.0);
            assert!(value.is_sign_negative());
        }
    }
}
