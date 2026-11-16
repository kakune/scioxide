//! Weighted central moments for the frequency, reliability and analytic weight models.

use crate::arithmetic::Arithmetic;
use crate::arithmetic::native_delta;
use crate::central::{Central, compatible};
use crate::error_helpers::invalid;
use crate::{
    Analytic, BiasCorrection, ConfidenceProbability, Frequency, KurtosisConvention,
    MeanConfidenceInterval, MomentOrder, MomentWeights, WeightSemantics, Weights,
};
#[cfg(test)]
use alloc::vec::Vec;
use core::marker::PhantomData;
use scioxide_core::Scaled;
use scioxide_core::{NonFiniteOrigin, NumericalIssue, RealScalar, SciOxideError};

/// Accumulates weighted moments of a stream of observations under the weight model `M`.
///
/// With weights `wᵢ ≥ 0`, the accumulator tracks the total weight
/// `W = Σ wᵢ`, the weighted mean `x̄ = Σ wᵢ xᵢ / W`, the extrema and the
/// central moments `mₖ = Σ wᵢ (xᵢ − x̄)ᵏ / W` up to the [`MomentOrder`].
/// The model fixes how the weights enter the sample estimators:
///
/// - [`Frequency`]: integer replication counts. `N = Σ wᵢ` is the sample size.
/// - [`crate::Reliability`]: relative importance. The effective sample size is
///   Kish's `W² / Q`, with `Q = Σ wᵢ²`.
/// - [`Analytic`]: inverse variances up to one common unknown factor, the
///   noise scale.
///
/// Zero weights add nothing. Accumulators of different models are different
/// types and cannot be merged. See
/// [Data, weights and missing values](crate#data-weights-and-missing-values).
/// Updates and merges use the pairwise formulas of Pébay et al. (2016).
///
/// # Examples
///
/// ```
/// use scioxide_stats::{Analytic, MomentOrder, OnlineWeightedMoments, Weights};
/// let moments = OnlineWeightedMoments::from_slices(
///     &[1.0_f64, 2.0, 4.0],
///     Weights::<_, Analytic>::new(&[1.0, 2.0, 1.0]),
///     MomentOrder::Second,
/// )?;
/// // (1·1 + 2·2 + 1·4) / 4 = 2.25
/// assert_eq!(moments.snapshot().mean()?, Some(2.25));
/// assert!(moments.snapshot().noise_scale()?.is_some());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// The noise scale exists only for analytic weights:
///
/// ```compile_fail
/// use scioxide_stats::{MomentOrder, OnlineWeightedMoments, Reliability};
/// let moments = OnlineWeightedMoments::<f64, Reliability>::new(MomentOrder::Second);
/// moments.snapshot().noise_scale();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct OnlineWeightedMoments<S: RealScalar, M: MomentWeights> {
    pub(crate) core: WeightedMoments<S>,
    model: PhantomData<M>,
}

impl<S: RealScalar, M: MomentWeights> OnlineWeightedMoments<S, M> {
    /// Creates an empty accumulator that tracks central moments up to `order`.
    pub fn new(order: MomentOrder) -> Self {
        Self::from_core(WeightedMoments::new(M::SEMANTICS, order))
    }
    pub(crate) const fn from_core(core: WeightedMoments<S>) -> Self {
        Self {
            core,
            model: PhantomData,
        }
    }
    /// Computes the weighted moments of a complete slice.
    ///
    /// When all values and weights lie in the native `f64` range, it makes two
    /// compensated passes. Otherwise it observes the pairs one by one in
    /// extended range. The cost is O(n) time and O(1) memory. The result can
    /// differ in the last bits from observing the pairs one at a time or from
    /// another merge order.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `values` and `weights` differ in length.
    /// - The errors of [`Self::observe`].
    pub fn from_slices<W: AsRef<[S]> + ?Sized>(
        values: &[S],
        weights: Weights<&W, M>,
        order: MomentOrder,
    ) -> Result<Self, SciOxideError> {
        WeightedMoments::from_slices(values, weights.into_storage().as_ref(), M::SEMANTICS, order)
            .map(Self::from_core)
    }
    /// Returns the number of observations with positive weight.
    ///
    /// This is not the replicated or effective sample size.
    pub const fn observation_count(&self) -> usize {
        self.core.observation_count()
    }
    /// Returns whether no observation with positive weight has been added.
    pub const fn is_empty(&self) -> bool {
        self.core.is_empty()
    }
    /// Returns the weight model `M` as a [`WeightSemantics`] value.
    pub const fn weight_semantics(&self) -> WeightSemantics {
        M::SEMANTICS
    }
    /// Returns the highest tracked moment order.
    pub const fn moment_order(&self) -> MomentOrder {
        self.core.moment_order()
    }
    /// Returns whether `other` tracks the same moment order and so can be merged.
    pub fn is_merge_compatible(&self, other: &Self) -> bool {
        self.core.is_merge_compatible(&other.core)
    }
    /// Adds one observation `value` with weight `weight ≥ 0`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `value` or `weight` is NaN or infinite.
    /// - `InvalidInput` if `weight` is negative.
    /// - `ScalarConversion` if a [`Frequency`] weight is not an integer that fits `usize`.
    /// - `SizeOverflow` if a count overflows `usize`.
    /// - `NumericalInstability` if the weight state or an even central moment
    ///   breaks down, or a range error of the extended arithmetic.
    #[inline(always)]
    pub fn observe(&mut self, value: S, weight: S) -> Result<(), SciOxideError> {
        self.core.observe(value, weight)
    }
    /// Merges the accumulator of a disjoint partition into this one.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the moment orders differ, even if one accumulator is empty.
    /// - `SizeOverflow` if a combined count overflows `usize`.
    /// - The numerical errors of [`Self::observe`].
    pub fn merge(&mut self, other: &Self) -> Result<(), SciOxideError> {
        self.core.merge(&other.core)
    }
    /// Returns the current estimates as an immutable snapshot.
    ///
    /// Each estimate is rounded to `S` only when you ask for it.
    pub const fn snapshot(&self) -> WeightedMomentsSnapshot<S, M> {
        self.core.snapshot()
    }
}

/// Weighted moment state with the weight model as a runtime value.
///
/// The public types fix the model in their type parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WeightedMoments<S: RealScalar> {
    observation_count: usize,
    frequency_count: usize,
    semantics: WeightSemantics,
    order: MomentOrder,
    mass: WeightMass<S>,
    pub(crate) state: Central<S>,
}

impl<S: RealScalar> WeightedMoments<S> {
    /// Creates an empty accumulator for the given weight model and order.
    pub(crate) fn new(semantics: WeightSemantics, order: MomentOrder) -> Self {
        Self {
            observation_count: 0,
            frequency_count: 0,
            semantics,
            order,
            mass: WeightMass::Native(NativeWeightMass {
                scale: S::ZERO,
                sum: 0.0,
                degrees_fraction: 0.0,
                concentration: 1.0,
            }),
            state: Central::empty(),
        }
    }

    /// Computes the weighted moments of a complete slice; see
    /// [`OnlineWeightedMoments::from_slices`].
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` for unequal lengths, or the errors of [`Self::observe`].
    pub(crate) fn from_slices(
        values: &[S],
        weights: &[S],
        semantics: WeightSemantics,
        order: MomentOrder,
    ) -> Result<Self, SciOxideError> {
        if values.len() != weights.len() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(values.len()),
                scioxide_core::Dimensions::one(weights.len()),
            ));
        }
        if let Some(batch) = crate::batch::from_slices(values, Some(weights), semantics, order)? {
            return Ok(Self {
                observation_count: batch.count,
                frequency_count: batch.frequency_count,
                semantics,
                order,
                state: batch.central,
                mass: WeightMass::from_extended(ExtendedWeightMass {
                    scale: batch.scale,
                    sum: batch.sum,
                    degrees_fraction: batch.degrees_fraction,
                    concentration: batch.concentration,
                })?,
            });
        }
        values.iter().zip(weights).try_fold(
            Self::new(semantics, order),
            |mut state, (&value, &weight)| {
                state.observe(value, weight)?;
                Ok(state)
            },
        )
    }
    /// Returns the number of observations with positive weight.
    pub(crate) const fn observation_count(&self) -> usize {
        self.observation_count
    }
    /// Returns whether no observation with positive weight has been added.
    pub(crate) const fn is_empty(&self) -> bool {
        self.observation_count == 0
    }
    /// Returns the weight model.
    pub(crate) const fn weight_semantics(&self) -> WeightSemantics {
        self.semantics
    }
    /// Returns the highest tracked moment order.
    pub(crate) const fn moment_order(&self) -> MomentOrder {
        self.order
    }
    /// Returns whether `other` has the same order and weight model.
    pub(crate) fn is_merge_compatible(&self, other: &Self) -> bool {
        self.order == other.order && self.semantics == other.semantics
    }
    /// Adds one observation with a nonnegative weight.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`OnlineWeightedMoments::observe`].
    #[inline(always)]
    pub(crate) fn observe(&mut self, value: S, weight: S) -> Result<(), SciOxideError> {
        let candidate = self.observed(value, weight)?;
        *self = candidate;
        Ok(())
    }
    #[inline(always)]
    pub(crate) fn observed(&self, value: S, weight: S) -> Result<Self, SciOxideError> {
        if !value.is_finite() || !weight.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "value and weight must be finite",
            ));
        }
        if weight < S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("weight"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "weight must be nonnegative",
            ));
        }
        if weight == S::ZERO {
            return Ok(*self);
        }
        let frequency_count = if self.semantics == WeightSemantics::Frequency {
            weight.try_to_usize_exact()?
        } else {
            0
        };
        let singleton = Self {
            observation_count: 1,
            frequency_count,
            semantics: self.semantics,
            order: self.order,
            mass: WeightMass::singleton(weight)?,
            state: Central::singleton(value)?,
        };
        self.combined(&singleton).map(|(state, _)| state)
    }
    /// Merges a disjoint partition with the same order and weight model.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if the order or weight model differs, even for an
    /// empty state, or a count or numerical error.
    pub(crate) fn merge(&mut self, other: &Self) -> Result<(), SciOxideError> {
        let (candidate, _) = self.combined(other)?;
        *self = candidate;
        Ok(())
    }
    pub(crate) fn merged(
        &self,
        other: &Self,
    ) -> Result<(Self, Scaled<f64>, Scaled<f64>), SciOxideError> {
        let (state, fractions) = self.combined(other)?;
        let (p, q) = fractions.scaled()?;
        Ok((state, p, q))
    }
    #[inline]
    fn combined(&self, other: &Self) -> Result<(Self, WeightFractions), SciOxideError> {
        compatible(self.is_merge_compatible(other))?;
        if other.is_empty() {
            return Ok((*self, WeightFractions::Native(1.0, 0.0)));
        }
        if self.is_empty() {
            return Ok((*other, WeightFractions::Native(0.0, 1.0)));
        }
        let observation_count = self
            .observation_count
            .checked_add(other.observation_count)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("weighted_observation_count"))
            })?;
        let frequency_count = self
            .frequency_count
            .checked_add(other.frequency_count)
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!(
                    "replicated_observation_count"
                ))
            })?;
        let (mass, state, fractions) = if let Some((mass, p, q)) =
            self.mass.merged_native(other.mass)
        {
            (
                mass,
                self.state
                    .merged_native_fractions(other.state, p, q, self.order)?,
                WeightFractions::Native(p, q),
            )
        } else {
            let weights = merge_normalized_weights(self.mass.extended()?, other.mass.extended()?)?;
            let state = self.state.merged(
                other.state,
                crate::arithmetic::Shares::rounded(weights.left_fraction, weights.right_fraction),
                self.order,
            )?;
            (
                WeightMass::Extended(ExtendedWeightMass {
                    scale: weights.scale,
                    sum: weights.sum,
                    degrees_fraction: weights.degrees_fraction,
                    concentration: weights.concentration,
                }),
                state,
                WeightFractions::Extended(weights.left_fraction, weights.right_fraction),
            )
        };
        Ok((
            Self {
                observation_count,
                frequency_count,
                semantics: self.semantics,
                order: self.order,
                mass,
                state,
            },
            fractions,
        ))
    }
    /// Returns the current estimates as an immutable snapshot.
    pub(crate) const fn snapshot<M: MomentWeights>(&self) -> WeightedMomentsSnapshot<S, M> {
        WeightedMomentsSnapshot {
            accumulator: *self,
            model: PhantomData,
        }
    }
    pub(crate) fn total_mass(&self) -> Result<Scaled<f64>, SciOxideError> {
        Scaled::<f64>::from_value(scioxide_core::RealScalar::into_f64(
            self.mass.extended()?.scale,
        ))?
        .mul(self.mass.extended()?.sum)
    }
    pub(crate) fn sample_factor(&self) -> Result<Option<Scaled<f64>>, SciOxideError> {
        match self.semantics {
            WeightSemantics::Frequency if self.frequency_count > 1 => {
                Scaled::<f64>::from_value((self.frequency_count) as f64)?
                    .div(Scaled::<f64>::from_value(
                        (self.frequency_count - 1) as f64,
                    )?)
                    .map(Some)
            }
            WeightSemantics::Reliability if self.observation_count > 1 => Scaled::<f64>::ONE
                .div(self.mass.extended()?.degrees_fraction)
                .map(Some),
            _ => Ok(None),
        }
    }
    fn sample_scaled(&self) -> Result<Option<Scaled<f64>>, SciOxideError> {
        self.sample_factor()?
            .map(|f| self.state.m2()?.mul(f))
            .transpose()
    }
    pub(crate) fn mean_variance_factor(&self) -> Result<Option<Scaled<f64>>, SciOxideError> {
        if self.semantics == WeightSemantics::Analytic {
            if self.observation_count < 2 {
                return Ok(None);
            }
            return Scaled::<f64>::ONE
                .div(Scaled::<f64>::from_value(
                    (self.observation_count - 1) as f64,
                )?)
                .map(Some);
        }
        self.sample_factor()?
            .map(|f| match self.semantics {
                WeightSemantics::Frequency => {
                    f.div(Scaled::<f64>::from_value((self.frequency_count) as f64)?)
                }
                _ => f.mul(self.mass.extended()?.concentration),
            })
            .transpose()
    }
    fn mean_variance_scaled(&self) -> Result<Option<Scaled<f64>>, SciOxideError> {
        self.mean_variance_factor()?
            .map(|f| self.state.m2()?.mul(f))
            .transpose()
    }
    pub(crate) fn estimator_count(&self) -> usize {
        if self.semantics == WeightSemantics::Frequency {
            self.frequency_count
        } else {
            self.observation_count
        }
    }

    fn corrected_count(&self, correction: BiasCorrection) -> Result<usize, SciOxideError> {
        if correction == BiasCorrection::Corrected && self.semantics != WeightSemantics::Frequency {
            return Err(SciOxideError::not_implemented(
                "unbiased higher moments require independent units or frequency replication",
            ));
        }
        Ok(if self.semantics == WeightSemantics::Frequency {
            self.frequency_count
        } else {
            self.observation_count
        })
    }
}

/// Immutable estimates from an [`OnlineWeightedMoments`] accumulator with weight model `M`.
///
/// The notation is that of [`OnlineWeightedMoments`]: `W = Σ wᵢ`,
/// `Q = Σ wᵢ²`, `n` the number of observations with positive weight and
/// `N = Σ wᵢ` for frequency weights. Some estimators exist only for one model:
/// [`Self::noise_scale`] for [`Analytic`] weights and the k-statistics for
/// [`Frequency`] weights. Estimates that are undefined for the current data
/// return `Ok(None)`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct WeightedMomentsSnapshot<S: RealScalar, M: MomentWeights> {
    pub(crate) accumulator: WeightedMoments<S>,
    model: PhantomData<M>,
}
impl<S: RealScalar, M: MomentWeights> WeightedMomentsSnapshot<S, M> {
    /// Returns the number `n` of observations with positive weight.
    pub const fn observation_count(self) -> usize {
        self.accumulator.observation_count
    }
    /// Returns the replicated sample size `N` for frequency weights, or `None` for other models.
    pub fn frequency_count(self) -> Option<usize> {
        (M::SEMANTICS == WeightSemantics::Frequency).then_some(self.accumulator.frequency_count)
    }
    /// Returns the weight model `M` as a [`WeightSemantics`] value.
    pub const fn weight_semantics(self) -> WeightSemantics {
        M::SEMANTICS
    }
    /// Returns the highest tracked moment order.
    pub const fn moment_order(self) -> MomentOrder {
        self.accumulator.order
    }
    /// Returns the total weight `W`, which is zero for an empty snapshot.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if `W` overflows `S`.
    pub fn total_weight(self) -> Result<S, SciOxideError> {
        self.accumulator.total_mass()?.round_to()
    }
    /// Returns the effective sample size, or `None` if the snapshot is empty.
    ///
    /// It is `N` for frequency weights and Kish's `W² / Q` for the other
    /// models. Inference with analytic weights uses `n` instead; see
    /// [`Self::degrees_of_freedom`].
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the result overflows `S`.
    ///
    /// # References
    ///
    /// L. Kish, *Survey Sampling*, Wiley, 1965.
    pub fn effective_sample_size(self) -> Result<Option<S>, SciOxideError> {
        let a = self.accumulator;
        if a.is_empty() {
            return Ok(None);
        }
        (if a.semantics == WeightSemantics::Frequency {
            Scaled::from_value(a.frequency_count as f64)?
        } else {
            Scaled::ONE.div(a.mass.extended()?.concentration)?
        })
        .round_to()
        .map(Some)
    }
    /// Returns the degrees of freedom of the variance estimate, or `None` if the snapshot is empty.
    ///
    /// It is `N − 1` for frequency weights, `W²/Q − 1` for reliability weights
    /// and `n − 1` for analytic weights.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the result cannot be represented in `S`.
    pub fn degrees_of_freedom(self) -> Result<Option<S>, SciOxideError> {
        if self.accumulator.is_empty() {
            return Ok(None);
        }
        let a = self.accumulator;
        let degrees = match a.semantics {
            WeightSemantics::Frequency => {
                Scaled::<f64>::from_value((a.frequency_count - 1) as f64)?
            }
            WeightSemantics::Reliability => a
                .mass
                .extended()?
                .degrees_fraction
                .div(a.mass.extended()?.concentration)?,
            WeightSemantics::Analytic => {
                Scaled::<f64>::from_value((a.observation_count - 1) as f64)?
            }
        };
        degrees.round_to().map(Some)
    }
    /// Returns the divisor `d` of `Σ wᵢ (xᵢ − x̄)²` in the sample variance or noise scale.
    ///
    /// It is `N − 1` for frequency weights, `W − Q/W` for reliability weights
    /// and `n − 1` for analytic weights. Returns `None` if `N < 2` for
    /// frequency weights or `n < 2` for the other models.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if `d` overflows `S`.
    pub fn variance_divisor(self) -> Result<Option<S>, SciOxideError> {
        let a = self.accumulator;
        let divisor = match a.semantics {
            WeightSemantics::Frequency if a.frequency_count > 1 => {
                Some(Scaled::<f64>::from_value((a.frequency_count - 1) as f64)?)
            }
            WeightSemantics::Reliability if a.observation_count > 1 => {
                Some(a.total_mass()?.mul(a.mass.extended()?.degrees_fraction)?)
            }
            WeightSemantics::Analytic if a.observation_count > 1 => {
                Some(Scaled::<f64>::from_value((a.observation_count - 1) as f64)?)
            }
            _ => None,
        };
        divisor.map(|value| value.round_to()).transpose()
    }
    /// Returns the weighted mean `x̄ = Σ wᵢ xᵢ / W`, or `None` if the snapshot is empty.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the mean cannot be represented in `S`.
    pub fn mean(self) -> Result<Option<S>, SciOxideError> {
        if self.accumulator.is_empty() {
            Ok(None)
        } else {
            self.accumulator.state.mean()?.total()?.round_to().map(Some)
        }
    }
    /// Returns the weighted population variance `m₂ = Σ wᵢ (xᵢ − x̄)² / W`, or `None` if empty.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if `m₂` overflows `S`.
    pub fn weighted_population_variance(self) -> Result<Option<S>, SciOxideError> {
        self.central_moment(2)
    }
    /// Returns the unbiased sample variance `s² = Σ wᵢ (xᵢ − x̄)² / d`.
    ///
    /// `d` is [`Self::variance_divisor`]. Returns `None` if `N < 2` for
    /// frequency weights or `n < 2` for reliability weights. Always returns
    /// `None` for analytic weights; use
    /// [`noise_scale`](WeightedMomentsSnapshot::noise_scale) instead.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if `s²` overflows `S`.
    pub fn sample_variance(self) -> Result<Option<S>, SciOxideError> {
        self.accumulator
            .sample_scaled()?
            .map(|value| value.round_to())
            .transpose()
    }
    /// Returns the estimated variance of the weighted mean.
    ///
    /// It is `s²/N` for frequency weights, `s² Q/W²` for reliability weights
    /// and `σ̂²/W` for analytic weights, where `σ̂²` is the noise scale. Returns
    /// `None` if `N < 2` for frequency weights or `n < 2` for the other models.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the result overflows `S`.
    pub fn variance_of_weighted_mean(self) -> Result<Option<S>, SciOxideError> {
        self.accumulator
            .mean_variance_scaled()?
            .map(|value| value.round_to())
            .transpose()
    }
    /// Returns the standard error of the weighted mean, the square root of
    /// [`Self::variance_of_weighted_mean`].
    ///
    /// The square root is taken before rounding to `S`.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the result overflows `S`.
    pub fn standard_error_of_weighted_mean(self) -> Result<Option<S>, SciOxideError> {
        self.accumulator
            .mean_variance_scaled()?
            .map(|value| value.sqrt()?.round_to())
            .transpose()
    }
    /// Returns the weighted central moment `mₖ = Σ wᵢ (xᵢ − x̄)ᵏ / W` of order `k = order`.
    ///
    /// Orders 0 and 1 give 1 and 0. Returns `None` if the snapshot is empty.
    ///
    /// # Errors
    ///
    /// - `NotImplemented` if `order` exceeds the tracked [`MomentOrder`].
    /// - A numerical error if `mₖ` overflows `S`.
    pub fn central_moment(self, order: usize) -> Result<Option<S>, SciOxideError> {
        let v = self
            .accumulator
            .state
            .moment(order, self.accumulator.order)?;
        if self.accumulator.is_empty() {
            Ok(None)
        } else {
            v.round_to().map(Some)
        }
    }
    /// Returns the weighted skewness `g₁ = m₃ / m₂^(3/2)`, or its corrected form.
    ///
    /// See [`BiasCorrection`] for the formulas. The correction uses `N` and is
    /// available only for frequency weights. Returns `None` if the snapshot is
    /// empty, `m₂ = 0`, or the corrected form has `N < 3`.
    ///
    /// # Errors
    ///
    /// - `NotImplemented` if the tracked order is below
    ///   [`MomentOrder::Third`], or if `correction` is
    ///   [`BiasCorrection::Corrected`] for reliability or analytic weights.
    /// - A numerical error if the result overflows `S`.
    pub fn skewness(self, correction: BiasCorrection) -> Result<Option<S>, SciOxideError> {
        let a = self.accumulator;
        a.state
            .skewness(a.corrected_count(correction)?, a.order, correction)
    }
    /// Returns the weighted kurtosis in the given convention, or its corrected form.
    ///
    /// See [`BiasCorrection`] and [`KurtosisConvention`] for the formulas. The
    /// correction uses `N` and is available only for frequency weights.
    /// Returns `None` if the snapshot is empty, `m₂ = 0`, or the corrected form
    /// has `N < 4`.
    ///
    /// # Errors
    ///
    /// - `NotImplemented` if the tracked order is below
    ///   [`MomentOrder::Fourth`], or if `correction` is
    ///   [`BiasCorrection::Corrected`] for reliability or analytic weights.
    /// - A numerical error if the result overflows `S`.
    pub fn kurtosis(
        self,
        correction: BiasCorrection,
        convention: KurtosisConvention,
    ) -> Result<Option<S>, SciOxideError> {
        let a = self.accumulator;
        a.state.kurtosis(
            a.corrected_count(correction)?,
            a.order,
            correction,
            convention,
        )
    }
    /// Computes the result of [`Self::confidence_interval_of_weighted_mean_with_options`] with the [one-shot defaults](crate#one-shot-forms).
    ///
    /// Uses the same numerical method, cost and accuracy as the configured form.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::confidence_interval_of_weighted_mean_with_options`] for the fixed default settings.
    pub fn confidence_interval_of_weighted_mean(
        self,
    ) -> Result<Option<MeanConfidenceInterval<S>>, SciOxideError> {
        self.confidence_interval_of_weighted_mean_with_options(
            ConfidenceProbability::default(),
            crate::MeanIntervalMethod::StudentT,
            Default::default(),
        )
    }
    /// Computes a two-sided confidence interval for the weighted mean.
    ///
    /// The interval is `x̄ ± c·se` with the standard error of
    /// [`Self::standard_error_of_weighted_mean`]. For
    /// [`MeanIntervalMethod::StudentT`](crate::MeanIntervalMethod::StudentT)
    /// the degrees of freedom are `N − 1` for frequency weights, `n − 1` for
    /// analytic weights and `W²/Q − 1` for reliability weights. The reliability
    /// interval is an approximation. The result reports the matching
    /// [`MeanIntervalAssumptions`](crate::MeanIntervalAssumptions) and the
    /// weight model. Returns `Ok(None)` if the standard error is undefined.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `method` is
    ///   [`NormalKnownVariance`](crate::MeanIntervalMethod::NormalKnownVariance), because the
    ///   variance is estimated.
    /// - The errors of [`Self::standard_error_of_weighted_mean`].
    /// - The errors of [`MeanIntervalPlan::new`](crate::MeanIntervalPlan::new) and
    ///   [`MeanIntervalPlan::interval`](crate::MeanIntervalPlan::interval).
    pub fn confidence_interval_of_weighted_mean_with_options(
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
            .zip(self.standard_error_of_weighted_mean()?)
            .map(|(mean, se)| {
                let a = self.accumulator;
                let (degrees, assumptions) = match a.semantics {
                    WeightSemantics::Frequency => (
                        (a.frequency_count - 1) as f64,
                        crate::MeanIntervalAssumptions::FrequencyNormal,
                    ),
                    WeightSemantics::Analytic => (
                        (a.observation_count - 1) as f64,
                        crate::MeanIntervalAssumptions::AnalyticNormal,
                    ),
                    WeightSemantics::Reliability => (
                        a.mass
                            .extended()?
                            .degrees_fraction
                            .div(a.mass.extended()?.concentration)?
                            .round_to::<f64>()?,
                        crate::MeanIntervalAssumptions::EffectiveSampleApproximation,
                    ),
                };
                crate::confidence::estimated_interval(
                    mean,
                    se,
                    probability,
                    method,
                    degrees,
                    assumptions,
                    options,
                )
                .map(|interval| interval.with_weight_semantics(a.semantics))
            })
            .transpose()
    }
}
impl<S: RealScalar> WeightedMomentsSnapshot<S, Analytic> {
    /// Returns the noise scale `σ̂² = Σ wᵢ (xᵢ − x̄)² / (n − 1)`, or `None` if `n < 2`.
    ///
    /// Analytic weights model `Var(xᵢ) = σ²/wᵢ` with one unknown `σ²`.
    /// Multiplying all weights by a constant multiplies `σ̂²` by it but leaves
    /// the variance of the weighted mean unchanged.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if `σ̂²` overflows `S`.
    pub fn noise_scale(self) -> Result<Option<S>, SciOxideError> {
        let a = self.accumulator;
        if a.observation_count < 2 {
            return Ok(None);
        }
        a.state
            .m2()?
            .mul(a.total_mass()?)?
            .div(Scaled::<f64>::from_value((a.observation_count - 1) as f64)?)?
            .round_to()
            .map(Some)
    }
}
impl<S: RealScalar> WeightedMomentsSnapshot<S, Frequency> {
    /// Returns the k-statistic `kᵣ`, the unbiased estimator of the cumulant of order `r = order`.
    ///
    /// The sample is the frequency-replicated sample of size `N`:
    ///
    /// ```text
    /// k₁ = x̄,
    /// k₂ = N m₂ / (N − 1),
    /// k₃ = N² m₃ / ((N − 1)(N − 2)),
    /// k₄ = N² ((N + 1) m₄ − 3 (N − 1) m₂²) / ((N − 1)(N − 2)(N − 3)).
    /// ```
    ///
    /// Returns `None` if `N < r`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `order` is not in `1..=4`.
    /// - `NotImplemented` if `order` exceeds the tracked [`MomentOrder`].
    /// - A numerical error if the result overflows `S`.
    pub fn k_statistic(self, order: usize) -> Result<Option<S>, SciOxideError> {
        let a = self.accumulator;
        a.state
            .k_statistic(order, a.frequency_count, a.order)?
            .map(|v| v.round_to())
            .transpose()
    }
    /// Returns the unbiased estimate `vᵣ` of `Var(kᵣ)` for `r = order` in `{1, 2}`.
    ///
    /// ```text
    /// v₁ = k₂ / N,
    /// v₂ = (2 N k₂² + (N − 1) k₄) / (N (N + 1)).
    /// ```
    ///
    /// The order-two estimate can be negative for a particular sample. Returns
    /// `None` if a required k-statistic is undefined.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `order` is not 1 or 2.
    /// - `NotImplemented` if the tracked order is too low: order 2 needs
    ///   [`MomentOrder::Fourth`].
    /// - A numerical error if the result overflows `S`.
    pub fn k_statistic_variance(self, order: usize) -> Result<Option<S>, SciOxideError> {
        let a = self.accumulator;
        a.state
            .k_statistic_variance(order, a.frequency_count, a.order)?
            .map(|value| value.round_to())
            .transpose()
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
struct NormalizedWeights<S: RealScalar> {
    scale: S,
    sum: Scaled<f64>,
    degrees_fraction: Scaled<f64>,
    concentration: Scaled<f64>,
    left_fraction: Scaled<f64>,
    right_fraction: Scaled<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum WeightMass<S: RealScalar> {
    Native(NativeWeightMass<S>),
    Extended(ExtendedWeightMass<S>),
}
#[derive(Debug, Clone, Copy, PartialEq)]
struct NativeWeightMass<S: RealScalar> {
    scale: S,
    sum: f64,
    degrees_fraction: f64,
    concentration: f64,
}
enum WeightFractions {
    Native(f64, f64),
    Extended(Scaled<f64>, Scaled<f64>),
}
impl WeightFractions {
    fn scaled(self) -> Result<(Scaled<f64>, Scaled<f64>), SciOxideError> {
        match self {
            Self::Native(p, q) => Ok((Scaled::from_value(p)?, Scaled::from_value(q)?)),
            Self::Extended(p, q) => Ok((p, q)),
        }
    }
}
impl<S: RealScalar> WeightMass<S> {
    fn singleton(scale: S) -> Result<Self, SciOxideError> {
        if native_delta(S::ZERO, scale).is_some() {
            Ok(Self::Native(NativeWeightMass {
                scale,
                sum: 1.0,
                degrees_fraction: 0.0,
                concentration: 1.0,
            }))
        } else {
            Ok(Self::Extended(ExtendedWeightMass {
                scale,
                sum: Scaled::ONE,
                degrees_fraction: Scaled::ZERO,
                concentration: Scaled::ONE,
            }))
        }
    }
    fn from_extended(value: ExtendedWeightMass<S>) -> Result<Self, SciOxideError> {
        let native = || -> Option<NativeWeightMass<S>> {
            native_delta(S::ZERO, value.scale)?;
            Some(NativeWeightMass {
                scale: value.scale,
                sum: crate::arithmetic::native_scaled(value.sum, 64)?,
                degrees_fraction: crate::arithmetic::native_scaled(value.degrees_fraction, 64)?,
                concentration: crate::arithmetic::native_scaled(value.concentration, 64)?,
            })
        };
        Ok(native().map_or(Self::Extended(value), Self::Native))
    }
    fn extended(self) -> Result<ExtendedWeightMass<S>, SciOxideError> {
        match self {
            Self::Extended(v) => Ok(v),
            Self::Native(v) => Ok(ExtendedWeightMass {
                scale: v.scale,
                sum: Scaled::from_value(v.sum)?,
                degrees_fraction: Scaled::from_value(v.degrees_fraction)?,
                concentration: Scaled::from_value(v.concentration)?,
            }),
        }
    }
    #[inline]
    fn merged_native(self, right: Self) -> Option<(Self, f64, f64)> {
        let (Self::Native(a), Self::Native(b)) = (self, right) else {
            return None;
        };
        let scale = a.scale.max(b.scale);
        let [sum, degrees_fraction, concentration, p, q] = merge_weight_powers(
            [
                a.scale.into_f64(),
                a.sum,
                a.degrees_fraction,
                a.concentration,
            ],
            [
                b.scale.into_f64(),
                b.sum,
                b.degrees_fraction,
                b.concentration,
            ],
            scale.into_f64(),
        )
        .ok()?;
        if [sum, degrees_fraction, concentration]
            .into_iter()
            .any(|x| !crate::arithmetic::native(x, 64))
            || sum <= 0.0
            || !(0.0..1.0).contains(&degrees_fraction)
            || concentration <= 0.0
            || concentration > 1.0
        {
            return None;
        }
        Some((
            Self::Native(NativeWeightMass {
                scale,
                sum,
                degrees_fraction,
                concentration,
            }),
            p,
            q,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ExtendedWeightMass<S: RealScalar> {
    scale: S,
    sum: Scaled<f64>,
    degrees_fraction: Scaled<f64>,
    concentration: Scaled<f64>,
}

#[inline(always)]
fn merge_normalized_weights<S: RealScalar>(
    left: ExtendedWeightMass<S>,
    right: ExtendedWeightMass<S>,
) -> Result<NormalizedWeights<S>, SciOxideError> {
    let scale = left.scale.max(right.scale);
    let [
        total,
        degrees_fraction,
        concentration,
        left_fraction,
        right_fraction,
    ] = merge_weight_powers(
        [
            Scaled::from_value(left.scale.into_f64())?,
            left.sum,
            left.degrees_fraction,
            left.concentration,
        ],
        [
            Scaled::from_value(right.scale.into_f64())?,
            right.sum,
            right.degrees_fraction,
            right.concentration,
        ],
        Scaled::from_value(scale.into_f64())?,
    )?;
    if total.is_zero()
        || degrees_fraction.is_sign_negative()
        || degrees_fraction.abs_cmp(Scaled::<f64>::ONE) != core::cmp::Ordering::Less
        || concentration.is_zero()
        || concentration.is_sign_negative()
        || concentration.abs_cmp(Scaled::<f64>::ONE) == core::cmp::Ordering::Greater
    {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Breakdown,
            "normalized reliability-weight state is outside its valid range",
        ));
    }
    Ok(NormalizedWeights {
        scale,
        sum: total,
        degrees_fraction,
        concentration,
        left_fraction,
        right_fraction,
    })
}

#[inline(always)]
fn merge_weight_powers<N: Arithmetic>(
    left: [N; 4],
    right: [N; 4],
    scale: N,
) -> Result<[N; 5], SciOxideError> {
    let left_total = left[1].mul(left[0].div(scale)?)?;
    let right_total = right[1].mul(right[0].div(scale)?)?;
    let total = left_total.add(right_total)?;
    let (p, q) = if left_total.abs_le(right_total) {
        let p = left_total.div(total)?;
        (p, N::number(1.0)?.sub(p)?)
    } else {
        let q = right_total.div(total)?;
        (N::number(1.0)?.sub(q)?, q)
    };
    let degrees = p
        .mul(q)?
        .mul(N::number(2.0)?)?
        .add(p.square()?.mul(left[2])?)?
        .add(q.square()?.mul(right[2])?)?;
    let concentration = p.square()?.mul(left[3])?.add(q.square()?.mul(right[3])?)?;
    Ok([total, degrees, concentration, p, q])
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::OnlineMoments;

    #[test]
    fn empty_and_singleton_weighted_estimates_are_explicit() {
        let mut moments =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        let empty = moments.snapshot();
        assert_eq!(empty.observation_count(), 0);
        assert_eq!(empty.effective_sample_size().unwrap(), None);
        assert_eq!(empty.mean().unwrap(), None);

        moments.observe(4.0, 3.0).unwrap();
        let singleton = moments.snapshot();
        assert_eq!(singleton.effective_sample_size().unwrap(), Some(1.0));
        assert_eq!(singleton.mean().unwrap(), Some(4.0));
        assert_eq!(singleton.weighted_population_variance().unwrap(), Some(0.0));
        assert_eq!(singleton.sample_variance().unwrap(), None);
    }

    #[test]
    fn reliability_weight_formulas_are_distinct() {
        let mut moments =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        [(1.0_f64, 1.0), (3.0, 2.0), (5.0, 1.0)]
            .into_iter()
            .try_for_each(|(value, weight)| moments.observe(value, weight))
            .unwrap();
        let snapshot = moments.snapshot();
        assert_eq!(snapshot.observation_count(), 3);
        assert_eq!(snapshot.total_weight().unwrap(), 4.0);
        assert!((snapshot.effective_sample_size().unwrap().unwrap() - 8.0 / 3.0).abs() < 1.0e-14);
        assert_eq!(snapshot.mean().unwrap(), Some(3.0));
        assert!((snapshot.weighted_population_variance().unwrap().unwrap() - 2.0).abs() < 1.0e-14);
        assert!((snapshot.sample_variance().unwrap().unwrap() - 3.2).abs() < 1.0e-14);
        assert!((snapshot.variance_of_weighted_mean().unwrap().unwrap() - 1.2).abs() < 1.0e-14);
    }

    #[test]
    fn unit_weights_agree_with_unweighted_estimators() {
        let values = [1.0_f64, 2.0, 7.0, 8.0];
        let mut plain = OnlineMoments::new(crate::MomentOrder::Second);
        let mut weighted =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        values
            .into_iter()
            .try_for_each(|value| {
                plain.observe(value)?;
                weighted.observe(value, 1.0)
            })
            .unwrap();
        let plain = plain.snapshot();
        let weighted = weighted.snapshot();
        assert_eq!(weighted.mean().unwrap(), plain.mean().unwrap());
        assert_eq!(
            weighted.weighted_population_variance().unwrap(),
            plain.population_variance().unwrap()
        );
        assert!(
            (weighted.sample_variance().unwrap().unwrap()
                - plain.sample_variance().unwrap().unwrap())
            .abs()
                <= 4.0 * f64::EPSILON * plain.sample_variance().unwrap().unwrap()
        );
        assert!(
            (weighted.variance_of_weighted_mean().unwrap().unwrap()
                - plain.variance_of_sample_mean().unwrap().unwrap())
            .abs()
                <= 4.0 * f64::EPSILON * plain.variance_of_sample_mean().unwrap().unwrap()
        );
    }

    #[test]
    fn integer_weights_match_repeated_population_moments() {
        let mut repeated = OnlineMoments::new(crate::MomentOrder::Second);
        let mut weighted =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        [(2.0_f64, 3_usize), (8.0, 1)]
            .into_iter()
            .try_for_each(|(value, repetitions)| {
                (0..repetitions).try_for_each(|_| repeated.observe(value))?;
                weighted.observe(value, repetitions as f64)
            })
            .unwrap();
        let repeated = repeated.snapshot();
        let weighted = weighted.snapshot();
        assert_eq!(weighted.mean().unwrap(), repeated.mean().unwrap());
        assert_eq!(
            weighted.weighted_population_variance().unwrap(),
            repeated.population_variance().unwrap()
        );
    }

    #[test]
    fn invalid_weight_and_overflow_leave_state_unchanged() {
        let mut moments =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        moments.observe(2.0_f64, 1.0).unwrap();
        for weight in [-1.0, -f64::MAX] {
            let before = moments;
            assert!(
                matches ! (& (moments . observe (3.0 , weight)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
            );
            assert_eq!(moments, before);
        }
        let before = moments;
        assert!(
            matches ! (& (moments . observe (3.0 , f64 :: NAN)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
        );
        assert_eq!(moments, before);

        let mut overflow = moments;
        overflow.core.observation_count = usize::MAX;
        let before = overflow;
        assert!(
            matches ! (& (overflow . observe (3.0 , 1.0)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: SizeOverflow { .. }))
        );
        assert_eq!(overflow, before);
    }

    #[test]
    fn extreme_dynamic_range_weights_remain_supported() {
        let large = 2.0_f64.powi(54);
        let mut snapshots = Vec::new();
        for observations in [[(1.0_f64, 1.0), (2.0, large)], [(1.0, large), (2.0, 1.0)]] {
            let mut moments =
                OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
            observations
                .into_iter()
                .try_for_each(|(value, weight)| moments.observe(value, weight))
                .unwrap();
            snapshots.push(moments.snapshot());
        }
        assert_eq!(snapshots[0].mean().unwrap(), Some(2.0));
        assert_eq!(snapshots[1].mean().unwrap(), Some(1.0));
        assert!(snapshots.iter().all(|snapshot| {
            snapshot.effective_sample_size().unwrap().unwrap() >= 1.0
                && snapshot.weighted_population_variance().unwrap().unwrap() >= 0.0
        }));
    }

    #[test]
    fn reliability_state_is_invariant_to_extreme_common_weight_scaling() {
        let observations = [(1.0_f64, 1.0), (4.0, 2.0), (9.0, 3.0)];
        let snapshots = [1.0, 1.0e-200, 1.0e200].map(|scale| {
            observations
                .into_iter()
                .try_fold(
                    OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second),
                    |mut state, (value, weight)| {
                        state.observe(value, weight * scale)?;
                        Ok::<_, SciOxideError>(state)
                    },
                )
                .unwrap()
                .snapshot()
        });
        for snapshot in &snapshots[1..] {
            assert!(
                (snapshot.mean().unwrap().unwrap() - snapshots[0].mean().unwrap().unwrap()).abs()
                    < 1.0e-14
            );
            assert!(
                (snapshot.weighted_population_variance().unwrap().unwrap()
                    - snapshots[0]
                        .weighted_population_variance()
                        .unwrap()
                        .unwrap())
                .abs()
                    < 1.0e-13
            );
            assert!(
                (snapshot.effective_sample_size().unwrap().unwrap()
                    - snapshots[0].effective_sample_size().unwrap().unwrap())
                .abs()
                    < 1.0e-14
            );
        }

        for weight in [f64::from_bits(1), f64::MIN_POSITIVE, 1.0, 1.0e200, f64::MAX] {
            let mut singleton =
                OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
            singleton.observe(3.0, weight).unwrap();
            assert_eq!(
                singleton
                    .snapshot()
                    .effective_sample_size()
                    .unwrap()
                    .unwrap()
                    .to_bits(),
                1.0_f64.to_bits()
            );
        }
    }

    #[test]
    fn unrepresentable_total_weight_fails_independently() {
        let mut moments =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        moments.observe(1.0_f64, f64::MAX).unwrap();
        moments.observe(1.0, f64::MAX).unwrap();
        let snapshot = moments.snapshot();
        assert_eq!(snapshot.mean().unwrap(), Some(1.0));
        assert!(snapshot.total_weight().is_err());
    }

    #[test]
    fn partition_merge_matches_single_pass() {
        let observations = [(1.0_f64, 0.5), (2.0, 1.5), (4.0, 2.0), (9.0, 0.75)];
        let mut single =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        observations
            .into_iter()
            .try_for_each(|(value, weight)| single.observe(value, weight))
            .unwrap();
        let mut left =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        observations[..2]
            .iter()
            .copied()
            .try_for_each(|(value, weight)| left.observe(value, weight))
            .unwrap();
        let mut right =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        observations[2..]
            .iter()
            .copied()
            .try_for_each(|(value, weight)| right.observe(value, weight))
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
                < 1.0e-13
        );
    }

    #[test]
    fn extreme_singletons_merge_without_forming_delta_squared() {
        let mut single =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        [(0.0_f64, 1.0), (1.5e154, 1.0)]
            .into_iter()
            .try_for_each(|(value, weight)| single.observe(value, weight))
            .unwrap();
        let mut left =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        left.observe(0.0, 1.0).unwrap();
        let mut right =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        right.observe(1.5e154, 1.0).unwrap();
        left.merge(&right).unwrap();
        assert_eq!(left.snapshot(), single.snapshot());
    }

    #[test]
    fn rounded_zero_weighted_moments_retain_a_representable_standard_error() {
        let minimum_subnormal = f32::from_bits(1);
        let mut moments =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        [
            (0.0_f32, 1.0_f32),
            (minimum_subnormal + minimum_subnormal, 1.0),
        ]
        .into_iter()
        .try_for_each(|(value, weight)| moments.observe(value, weight))
        .unwrap();
        let snapshot = moments.snapshot();
        assert_eq!(snapshot.weighted_population_variance().unwrap(), Some(0.0));
        assert_eq!(snapshot.sample_variance().unwrap(), Some(0.0));
        assert_eq!(snapshot.variance_of_weighted_mean().unwrap(), Some(0.0));
        assert_eq!(
            snapshot.standard_error_of_weighted_mean().unwrap(),
            Some(minimum_subnormal)
        );
    }

    #[test]
    fn merge_count_overflow_leaves_state_unchanged() {
        let mut overflow =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        overflow.observe(2.0_f64, 1.0).unwrap();
        overflow.core.observation_count = usize::MAX;
        let mut other =
            OnlineWeightedMoments::<_, crate::Reliability>::new(crate::MomentOrder::Second);
        other.observe(3.0, 1.0).unwrap();
        let before = overflow;
        assert!(
            matches ! (& (overflow . merge (& other)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: SizeOverflow { .. }))
        );
        assert_eq!(overflow, before);
    }
}
