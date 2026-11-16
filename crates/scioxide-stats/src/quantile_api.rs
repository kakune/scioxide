//! Empirical distributions, and quantiles and tail means computed in a work buffer.
use crate::weights::WeightData;
use crate::{
    BatchWeights, EmpiricalWeights, Frequency, Mass, QuantileProbability, QuantileResult,
    QuantileRule, Unweighted, WeightModel, Weights,
};
use alloc::vec::Vec;
use core::marker::PhantomData;
use scioxide_core::{RealScalar, SciOxideError};

mod selection {
    use super::*;
    pub trait Sealed<S: RealScalar> {
        type Model: WeightModel;
        fn select(
            self,
            p: QuantileProbability,
            rule: QuantileRule<Self::Model>,
        ) -> Result<QuantileResult<S>, SciOxideError>;
        fn tail(
            self,
            _p: QuantileProbability,
            _side: crate::TailSide,
        ) -> Result<crate::TailMeanResult<S>, SciOxideError>
        where
            Self: Sized,
        {
            Err(crate::error_helpers::invalid(
                scioxide_core::argument!("weights"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid tail capability",
            ))
        }
    }
}
/// Work buffer accepted by [`quantile_in_place`] and [`tail_mean_in_place`].
///
/// The buffer is reordered in place. It is one of:
///
/// - `&mut [S]`, `&mut [S; N]` or `&mut Vec<S>`: unweighted observations.
/// - [`Weights<&mut B, Mass>`](crate::Weights) with `B: AsMut<[(S, S)]>`:
///   `(observation, weight)` pairs.
/// - [`Weights<&mut B, Frequency>`](crate::Weights) with `B: AsMut<[(S, u64)]>`:
///   `(observation, count)` pairs.
///
/// Pairs move as units. The trait is sealed.
pub trait SelectionSamples<S: RealScalar>: selection::Sealed<S> {}
impl<S: RealScalar, D: selection::Sealed<S>> SelectionSamples<S> for D {}
impl<S: RealScalar> selection::Sealed<S> for &mut [S] {
    fn tail(
        self,
        p: QuantileProbability,
        side: crate::TailSide,
    ) -> Result<crate::TailMeanResult<S>, SciOxideError> {
        crate::empirical::tail_mean_in_place(self, p, side)
    }

    type Model = Unweighted;
    fn select(
        self,
        p: QuantileProbability,
        rule: QuantileRule<Unweighted>,
    ) -> Result<QuantileResult<S>, SciOxideError> {
        crate::empirical::quantile_in_place(self, p, rule.method)
    }
}
impl<S: RealScalar, const N: usize> selection::Sealed<S> for &mut [S; N] {
    fn tail(
        self,
        p: QuantileProbability,
        side: crate::TailSide,
    ) -> Result<crate::TailMeanResult<S>, SciOxideError> {
        crate::empirical::tail_mean_in_place(self, p, side)
    }

    type Model = Unweighted;
    fn select(
        self,
        p: QuantileProbability,
        rule: QuantileRule<Unweighted>,
    ) -> Result<QuantileResult<S>, SciOxideError> {
        crate::empirical::quantile_in_place(self, p, rule.method)
    }
}
impl<S: RealScalar> selection::Sealed<S> for &mut Vec<S> {
    fn tail(
        self,
        p: QuantileProbability,
        side: crate::TailSide,
    ) -> Result<crate::TailMeanResult<S>, SciOxideError> {
        crate::empirical::tail_mean_in_place(self, p, side)
    }

    type Model = Unweighted;
    fn select(
        self,
        p: QuantileProbability,
        rule: QuantileRule<Unweighted>,
    ) -> Result<QuantileResult<S>, SciOxideError> {
        crate::empirical::quantile_in_place(self, p, rule.method)
    }
}
impl<S: RealScalar, B: AsMut<[(S, S)]> + ?Sized> selection::Sealed<S> for Weights<&mut B, Mass> {
    fn tail(
        self,
        p: QuantileProbability,
        side: crate::TailSide,
    ) -> Result<crate::TailMeanResult<S>, SciOxideError> {
        crate::empirical::weighted_tail_mean_in_place(self.into_storage().as_mut(), p, side)
    }

    type Model = Mass;
    fn select(
        self,
        p: QuantileProbability,
        _rule: QuantileRule<Mass>,
    ) -> Result<QuantileResult<S>, SciOxideError> {
        crate::empirical::weighted_inverse_ecdf_quantile_in_place(self.into_storage().as_mut(), p)
    }
}
impl<S: RealScalar, B: AsMut<[(S, u64)]> + ?Sized> selection::Sealed<S>
    for Weights<&mut B, Frequency>
{
    type Model = Frequency;
    fn select(
        self,
        p: QuantileProbability,
        rule: QuantileRule<Frequency>,
    ) -> Result<QuantileResult<S>, SciOxideError> {
        crate::empirical::frequency_quantile_in_place(self.into_storage().as_mut(), p, rule.method)
    }
}
/// Computes a sample quantile by reordering a work buffer, without copying it.
///
/// The definitions are given on [`QuantileMethod`](crate::QuantileMethod) and
/// [`QuantileConvention`](crate::QuantileConvention). The buffer `samples`
/// selects the method:
///
/// - Unweighted observations accept any Hyndman–Fan definition and use
///   partial selection in expected O(n) time.
/// - Mass-weighted pairs take
///   [`QuantileRule::inverse_ecdf`](crate::QuantileRule::inverse_ecdf) and use
///   a deterministic selection that is O(n) in the worst case.
/// - Frequency-weighted pairs accept any Hyndman–Fan definition and are sorted
///   in O(n log n) time.
///
/// On success the buffer holds a permutation of its input. For repeated
/// queries on the same data, build an [`EmpiricalDistribution`] instead.
///
/// The buffer is checked before it is reordered, so an invalid buffer is left
/// unchanged.
///
/// # Errors
///
/// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if the buffer
///   is empty, a weight is negative, or the total weight or count is zero.
/// - [`NonFinite`](scioxide_core::SciOxideErrorKind::NonFinite) if an
///   observation or weight is NaN or infinite. Observations with a zero
///   frequency count are not checked.
/// - [`SizeOverflow`](scioxide_core::SciOxideErrorKind::SizeOverflow) if the total
///   frequency count exceeds `usize::MAX`.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{QuantileMethod, QuantileProbability, QuantileRule, quantile_in_place};
///
/// let mut values = [3.0_f64, 1.0, 2.0];
/// let median = quantile_in_place(
///     &mut values,
///     QuantileProbability::new(0.5)?,
///     QuantileRule::new(QuantileMethod::Linear),
/// )?;
/// // The type 7 median of 1, 2, 3 is the middle order statistic.
/// assert_eq!(median.value(), 2.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn quantile_in_place<S: RealScalar, D: SelectionSamples<S>>(
    samples: D,
    p: QuantileProbability,
    rule: QuantileRule<<D as selection::Sealed<S>>::Model>,
) -> Result<QuantileResult<S>, SciOxideError> {
    selection::Sealed::select(samples, p, rule)
}

// Keep the exact-mass state inline rather than adding an allocation to preparation.
#[allow(clippy::large_enum_variant)]
enum Sorted<S: RealScalar> {
    Unweighted(crate::RawEmpiricalDistribution<S>),
    Mass(crate::RawWeightedEmpiricalDistribution<S>),
    Frequency {
        cumulative: Vec<(S, usize)>,
        count: usize,
    },
}
/// Empirical distribution of a finite sample, stored sorted for repeated
/// quantile, CDF and tail-mean queries.
///
/// The weight model `M` is [`Unweighted`], [`Mass`] or [`Frequency`]. It fixes
/// the quantile definitions that apply (see
/// [`QuantileConvention`](crate::QuantileConvention)). Construction copies and
/// sorts the sample in O(n log n) time and O(n) memory. For unweighted data,
/// quantiles then take O(1) time and CDF queries O(log n).
///
/// # Examples
///
/// ```
/// use scioxide_stats::{
///     EmpiricalDistribution, QuantileMethod, QuantileProbability, QuantileRule, Unweighted,
/// };
///
/// let distribution = EmpiricalDistribution::new(&[3.0_f64, 1.0, 4.0, 2.0], Unweighted)?;
/// let rule = QuantileRule::new(QuantileMethod::Linear);
/// let median = distribution.quantile(QuantileProbability::new(0.5)?, rule)?;
/// // Type 7 at p = 1/2 interpolates halfway between 2 and 3.
/// assert!((median.value() - 2.5).abs() < 1e-15);
/// // Two of the four observations are at most 2.
/// assert!((distribution.cdf(2.0)? - 0.5).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct EmpiricalDistribution<S: RealScalar, M: EmpiricalWeights = Unweighted> {
    sorted: Sorted<S>,
    model: PhantomData<M>,
}
impl<S: RealScalar, M: EmpiricalWeights> EmpiricalDistribution<S, M> {
    /// Checks, copies and sorts `values` with their `weights`.
    ///
    /// Observations with zero weight or zero count are ignored by every query.
    ///
    /// # Errors
    ///
    /// - [`DimensionMismatch`](scioxide_core::SciOxideErrorKind::DimensionMismatch)
    ///   if `weights` and `values` differ in length.
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if
    ///   `values` is empty, a mass is negative, or the total mass or count is
    ///   zero.
    /// - [`NonFinite`](scioxide_core::SciOxideErrorKind::NonFinite) if an
    ///   observation or mass is NaN or infinite. Observations with a zero
    ///   frequency count are not checked.
    /// - [`SizeOverflow`](scioxide_core::SciOxideErrorKind::SizeOverflow) if the
    ///   total frequency count exceeds `usize::MAX`.
    pub fn new<'a, W: BatchWeights<'a, S, Model = M>>(
        values: &[S],
        weights: W,
    ) -> Result<Self, SciOxideError> {
        let weights = crate::weights::batch_sealed::Sealed::data(weights);
        if let Some(n) = weights.len()
            && n != values.len()
        {
            return Err(SciOxideError::dimension_mismatch([values.len()], [n]));
        }
        let sorted = match weights {
            WeightData::None => Sorted::Unweighted(crate::RawEmpiricalDistribution::new(values)?),
            WeightData::Mass(masses) => {
                let mut paired = scioxide_core::backend::try_with_capacity(
                    values.len(),
                    scioxide_core::quantity!("empirical_pairs"),
                )?;
                paired.extend(values.iter().copied().zip(masses.iter().copied()));
                Sorted::Mass(crate::RawWeightedEmpiricalDistribution::new(&paired)?)
            }
            WeightData::Frequency(counts) => {
                let mut cumulative = scioxide_core::backend::try_with_capacity(
                    values.len(),
                    scioxide_core::quantity!("empirical_counts"),
                )?;
                for (&value, &count) in values.iter().zip(counts) {
                    if count == 0 {
                        continue;
                    }
                    if !value.is_finite() {
                        return Err(SciOxideError::non_finite(
                            scioxide_core::NonFiniteOrigin::Input,
                            "nonfinite positive-count support",
                        ));
                    }
                    cumulative.push((value, crate::weights::frequency_index(count)?));
                }
                cumulative.sort_unstable_by(|a, b| crate::ranking::finite_total_cmp(&a.0, &b.0));
                let mut count = 0usize;
                for (_, n) in &mut cumulative {
                    count = count.checked_add(*n).ok_or_else(|| {
                        SciOxideError::size_overflow(scioxide_core::quantity!("empirical_count"))
                    })?;
                    *n = count;
                }
                crate::empirical::quantile_position(
                    count,
                    QuantileProbability::new(0.0)?,
                    crate::QuantileMethod::NearestRank,
                )?;
                Sorted::Frequency { cumulative, count }
            }
            _ => {
                return Err(crate::error_helpers::invalid(
                    scioxide_core::argument!("weights"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "invalid empirical capability",
                ));
            }
        };
        Ok(Self {
            sorted,
            model: PhantomData,
        })
    }
    /// Returns the number of stored observations.
    ///
    /// This counts every observation without weights or with masses, and the
    /// observations with a positive count with frequency weights.
    pub fn len(&self) -> usize {
        match &self.sorted {
            Sorted::Unweighted(d) => d.len(),
            Sorted::Mass(d) => d.len(),
            Sorted::Frequency { cumulative, .. } => cumulative.len(),
        }
    }
    /// Returns `false`: construction rejects an empty sample.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns the total weight `W`: the number of observations, the sum of the
    /// masses, or the total count.
    ///
    /// Returns `None` if `W` is not representable in `S`. Counts must be
    /// representable exactly.
    pub fn total_weight(&self) -> Option<S> {
        match &self.sorted {
            Sorted::Unweighted(d) => S::try_from_usize_exact(d.len()).ok(),
            Sorted::Mass(d) => d.total_weight(),
            Sorted::Frequency { count, .. } => S::try_from_usize_exact(*count).ok(),
        }
    }
    /// Returns the sample quantile at `p` with the definition in `rule`.
    ///
    /// # Errors
    ///
    /// Fails only if the interpolated value is not representable in `S`.
    pub fn quantile(
        &self,
        p: QuantileProbability,
        rule: QuantileRule<M>,
    ) -> Result<QuantileResult<S>, SciOxideError> {
        match &self.sorted {
            Sorted::Unweighted(d) => d.quantile(p, rule.method),
            Sorted::Mass(d) => d.weighted_inverse_ecdf_quantile(p),
            Sorted::Frequency { cumulative, count } => {
                let position = crate::empirical::quantile_position(*count, p, rule.method)?;
                let value = position.evaluate(|rank| {
                    cumulative[cumulative.partition_point(|&(_, n)| n <= rank)].0
                })?;
                Ok(QuantileResult::frequency(value, p, rule.method))
            }
        }
    }
    /// Returns the empirical CDF `F(x) = P(X ≤ x)`, the weight of the
    /// observations at most `value` divided by `W`.
    ///
    /// `F` is a right-continuous step function. Infinite `value` gives 0 or 1.
    ///
    /// # Errors
    ///
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if
    ///   `value` is NaN.
    pub fn cdf(&self, value: S) -> Result<S, SciOxideError> {
        self.probability(value, false)
    }
    /// Returns the empirical survival function `P(X > x)`.
    ///
    /// The weight above `value` is summed directly instead of forming
    /// `1 − F(x)`, so small upper-tail probabilities keep their relative
    /// accuracy.
    ///
    /// # Errors
    ///
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if
    ///   `value` is NaN.
    pub fn survival(&self, value: S) -> Result<S, SciOxideError> {
        self.probability(value, true)
    }
    fn probability(&self, value: S, upper: bool) -> Result<S, SciOxideError> {
        match &self.sorted {
            Sorted::Unweighted(d) => {
                if upper {
                    d.survival(value)
                } else {
                    d.cdf(value)
                }
            }
            Sorted::Mass(d) => {
                if upper {
                    d.survival(value)
                } else {
                    d.cdf(value)
                }
            }
            Sorted::Frequency { cumulative, count } => {
                if value.is_nan() {
                    return Err(crate::error_helpers::invalid(
                        scioxide_core::argument!("value"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "NaN empirical query",
                    ));
                }
                let i = cumulative.partition_point(|&(x, _)| x <= value);
                let lower = if i == 0 { 0 } else { cumulative[i - 1].1 };
                crate::arithmetic::count_ratio(if upper { count - lower } else { lower }, *count)?
                    .round_to()
            }
        }
    }
    /// Returns the CDF estimate that corresponds to the quantile definition in
    /// `rule`.
    ///
    /// The estimate is 0 below the smallest observation and 1 above the
    /// largest. In between, with `k` the number of observations at most `x`:
    ///
    /// - Types 1, 2 and 3 give `k/n`, `(k − 1/2)/n` and `(k + 1/2)/n`.
    /// - Types 4–9 invert the piecewise-linear quantile function. For
    ///   `x₍ₖ₎ ≤ x ≤ x₍ₖ₊₁₎`, with `h = k + (x − x₍ₖ₎)/(x₍ₖ₊₁₎ − x₍ₖ₎)`, the
    ///   estimate is the plotting position `(h − α)/(n + 1 − α − β)`.
    ///
    /// Here `(α, β)` is `(0, 1)`, `(1/2, 1/2)`, `(0, 0)`, `(1, 1)`,
    /// `(1/3, 1/3)` or `(3/8, 3/8)` for types 4 to 9, and the result is clamped
    /// to `[0, 1]`. Frequency weights use the replicated sample. With masses,
    /// the estimate is [`cdf`](Self::cdf).
    ///
    /// # Errors
    ///
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if
    ///   `value` is NaN.
    pub fn estimated_cdf(&self, value: S, rule: QuantileRule<M>) -> Result<S, SciOxideError> {
        match &self.sorted {
            Sorted::Unweighted(d) => d.estimated_cdf(value, rule.method),
            Sorted::Mass(d) => d.cdf(value),
            Sorted::Frequency { cumulative, count } => crate::empirical::estimated_cdf_by(
                *count,
                {
                    let i = cumulative.partition_point(|&(x, _)| x <= value);
                    if i == 0 { 0 } else { cumulative[i - 1].1 }
                },
                value,
                rule.method,
                |rank| cumulative[cumulative.partition_point(|&(_, n)| n <= rank)].0,
            ),
        }
    }
}
/// Weight models for which tail means are defined: [`Unweighted`] and
/// [`Mass`].
pub trait TailWeights: EmpiricalWeights {}
impl TailWeights for Unweighted {}
impl TailWeights for Mass {}
impl<S: RealScalar, M: TailWeights> EmpiricalDistribution<S, M> {
    /// Returns the mean of the lower tail of weight `p W` or the upper tail of
    /// weight `(1 − p) W`, in O(n) time.
    ///
    /// The boundary observation is included with the part of its weight that
    /// lies inside the tail. See [`TailMeanResult`](crate::TailMeanResult) for
    /// the definition.
    ///
    /// # Errors
    ///
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if the
    ///   tail is empty: `p = 0` with [`TailSide::Lower`](crate::TailSide::Lower)
    ///   or `p = 1` with [`TailSide::Upper`](crate::TailSide::Upper).
    /// - [`NumericalInstability`](scioxide_core::SciOxideErrorKind::NumericalInstability)
    ///   if the tail weight of a nonempty mass-weighted tail underflows to zero.
    pub fn tail_mean(
        &self,
        p: QuantileProbability,
        side: crate::TailSide,
    ) -> Result<crate::TailMeanResult<S>, SciOxideError> {
        match &self.sorted {
            Sorted::Unweighted(d) => d.tail_mean(p, side),
            Sorted::Mass(d) => d.tail_mean(p, side),
            Sorted::Frequency { .. } => Err(crate::error_helpers::invalid(
                scioxide_core::argument!("weights"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid tail capability",
            )),
        }
    }
}

/// Computes the mean of one tail of a sample by sorting a work buffer in
/// O(n log n) time.
///
/// The lower tail at probability `p` holds the lowest observations of total
/// weight `p W` and the upper tail the highest of weight `(1 − p) W`. The
/// boundary observation is included with the part of its weight that lies
/// inside the tail. See [`TailMeanResult`](crate::TailMeanResult) for the
/// definition. The buffer holds unweighted observations or mass-weighted
/// pairs. It is checked before it is sorted, so an invalid buffer is left
/// unchanged.
///
/// # Errors
///
/// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if the
///   buffer is empty, a mass is negative, the total mass is zero, or the tail
///   is empty (`p = 0` for the lower tail, `p = 1` for the upper tail).
/// - [`NonFinite`](scioxide_core::SciOxideErrorKind::NonFinite) if an
///   observation or mass is NaN or infinite.
/// - [`NumericalInstability`](scioxide_core::SciOxideErrorKind::NumericalInstability)
///   if the tail weight of a nonempty mass-weighted tail underflows to zero.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{QuantileProbability, TailSide, tail_mean_in_place};
///
/// let mut values = [4.0_f64, 1.0, 3.0, 2.0];
/// let p = QuantileProbability::new(0.5)?;
/// let upper = tail_mean_in_place(&mut values, p, TailSide::Upper)?;
/// // The upper half of the sample is {3, 4}.
/// assert!((upper.mean() - 3.5).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn tail_mean_in_place<S: RealScalar, D: SelectionSamples<S>>(
    samples: D,
    p: QuantileProbability,
    side: crate::TailSide,
) -> Result<crate::TailMeanResult<S>, SciOxideError>
where
    <D as selection::Sealed<S>>::Model: TailWeights,
{
    selection::Sealed::tail(samples, p, side)
}
