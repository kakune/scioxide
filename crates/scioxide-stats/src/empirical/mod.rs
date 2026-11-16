//! Sample quantiles, empirical CDFs and tail means of a finite sample.
//!
//! The quantile definitions are documented on [`QuantileMethod`], the weighted
//! inverse empirical CDF on [`QuantileConvention::MassInverseEcdf`], and tail
//! means on [`TailMeanResult`].

use crate::ranking::finite_total_cmp;
use core::cmp::Ordering;

use scioxide_core::{InvalidInputReason, NonFiniteOrigin, RealScalar, SciOxideError};

mod distribution;
mod exact_mass;
mod position;
mod selection;
mod tail;
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;

pub(crate) use distribution::{
    RawEmpiricalDistribution, RawWeightedEmpiricalDistribution, estimated_cdf_by,
};
pub(crate) use exact_mass::ExactMass;
pub(crate) use position::{
    frequency_quantile_in_place, quantile_position, quantile_scaled_sorted_by, quantile_sorted_by,
};
pub(crate) use selection::{
    quantile_in_place, select_adjacent_by, weighted_inverse_ecdf_quantile_in_place,
};
pub(crate) use tail::{tail_mean_in_place, weighted_tail_mean_in_place};

/// A probability `p` in the closed interval `[0, 1]` at which a quantile or
/// tail mean is evaluated.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[must_use]
pub struct QuantileProbability(f64);

impl QuantileProbability {
    /// Checks that `probability` lies in `[0, 1]` and wraps it.
    ///
    /// Both endpoints are accepted. Every quantile definition in this crate
    /// returns the smallest observation at `p = 0` and the largest at `p = 1`.
    ///
    /// # Errors
    ///
    /// - [`NonFinite`](scioxide_core::SciOxideErrorKind::NonFinite) if `probability`
    ///   is NaN or infinite.
    /// - [`InvalidInput`](scioxide_core::SciOxideErrorKind::InvalidInput) if
    ///   `probability` lies outside `[0, 1]`.
    pub fn new(probability: f64) -> Result<Self, SciOxideError> {
        if !probability.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "quantile probability must be finite",
            ));
        }
        if !(0.0..=1.0).contains(&probability) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("probability"),
                InvalidInputReason::OutOfRange,
                "quantile probability must lie in [0, 1]",
            ));
        }
        Ok(Self(probability))
    }

    /// Returns `p` as an `f64`.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// One of the nine sample quantile definitions of Hyndman and Fan (1996).
///
/// For sorted observations `x₍₁₎ ≤ … ≤ x₍ₙ₎` and a probability `p`, every
/// definition is a weighted average of two adjacent order statistics:
///
/// ```text
/// Q(p) = (1 − γ) x₍ⱼ₎ + γ x₍ⱼ₊₁₎,   j = ⌊n p + m⌋,   g = n p + m − j,
/// ```
///
/// where the offset `m` and the weight `γ ∈ [0, 1]` depend on the definition.
/// Order statistics with index below 1 or above `n` are replaced by `x₍₁₎` and
/// `x₍ₙ₎`, so `Q(0) = x₍₁₎` and `Q(1) = x₍ₙ₎` for every definition.
///
/// | Variant | Type | `m` | `γ` |
/// |---|---|---|---|
/// | [`NearestRank`](Self::NearestRank) | 1 | `0` | `0` if `g = 0`, else `1` |
/// | [`AveragedInverseCdf`](Self::AveragedInverseCdf) | 2 | `0` | `1/2` if `g = 0`, else `1` |
/// | [`ClosestObservation`](Self::ClosestObservation) | 3 | `−1/2` | `0` if `g = 0` and `j` is even, else `1` |
/// | [`InterpolatedInverseCdf`](Self::InterpolatedInverseCdf) | 4 | `0` | `g` |
/// | [`Hazen`](Self::Hazen) | 5 | `1/2` | `g` |
/// | [`Weibull`](Self::Weibull) | 6 | `p` | `g` |
/// | [`Linear`](Self::Linear) | 7 | `1 − p` | `g` |
/// | [`MedianUnbiased`](Self::MedianUnbiased) | 8 | `(p + 1)/3` | `g` |
/// | [`NormalUnbiased`](Self::NormalUnbiased) | 9 | `p/4 + 3/8` | `g` |
///
/// Types 1–3 are step functions of `p`. Types 4–9 interpolate linearly
/// between order statistics at the one-based position `h = n p + m`. The
/// index `j` is computed exactly from the binary value of `p`. Only the
/// fraction `g` is rounded.
///
/// With frequency weights, the definitions apply to the sample in which each
/// observation is repeated by its count, without forming that sample.
///
/// # References
///
/// R. J. Hyndman and Y. Fan, "Sample quantiles in statistical packages",
/// *The American Statistician* 50(4), 361–365, 1996.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum QuantileMethod {
    /// Type 1: the inverse of the empirical CDF, `x₍⌈n p⌉₎` for `p > 0`.
    NearestRank,
    /// Type 2: the inverse of the empirical CDF, averaging `x₍ₖ₎` and `x₍ₖ₊₁₎`
    /// where `n p = k` is an integer.
    AveragedInverseCdf,
    /// Type 3: the order statistic nearest to `n p`, taking the even index when
    /// `n p` lies halfway between two indices.
    ClosestObservation,
    /// Type 4: linear interpolation of the empirical CDF, at position `h = n p`.
    InterpolatedInverseCdf,
    /// Type 5: linear interpolation at position `h = n p + 1/2`.
    Hazen,
    /// Type 6: linear interpolation at position `h = (n + 1) p`.
    Weibull,
    /// Type 8: linear interpolation at position `h = (n + 1/3) p + 1/3`.
    /// The result is approximately median-unbiased for any continuous
    /// distribution.
    MedianUnbiased,
    /// Type 9: linear interpolation at position `h = (n + 1/4) p + 3/8`.
    /// The result is approximately unbiased for normally distributed data.
    NormalUnbiased,
    /// Type 7: linear interpolation at position `h = (n − 1) p + 1`.
    Linear,
}

/// Quantile definition used to compute a [`QuantileResult`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum QuantileConvention {
    /// Unweighted observations, with the given Hyndman–Fan definition.
    Unweighted(QuantileMethod),
    /// Mass-weighted observations, with the left-continuous inverse of the
    /// weighted empirical CDF.
    ///
    /// For observations `xᵢ` with weights `wᵢ ≥ 0` and total weight
    /// `W = Σ wᵢ > 0`, the quantile is the smallest observation `x` with
    /// positive weight such that `Σ_{xᵢ ≤ x} wᵢ ≥ p W`. At `p = 0` it is the
    /// smallest observation with positive weight. Observations with zero
    /// weight are ignored. With equal weights this is Hyndman–Fan type 1.
    MassInverseEcdf,
    /// Frequency-weighted observations, with the given Hyndman–Fan definition
    /// applied to the sample in which each observation is repeated by its count.
    Frequency(QuantileMethod),
}

/// A sample quantile with the probability and definition used to compute it.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct QuantileResult<S: RealScalar> {
    value: S,
    probability: QuantileProbability,
    convention: QuantileConvention,
}

impl<S: RealScalar> QuantileResult<S> {
    pub(crate) const fn unweighted(
        value: S,
        probability: QuantileProbability,
        method: QuantileMethod,
    ) -> Self {
        Self {
            value,
            probability,
            convention: QuantileConvention::Unweighted(method),
        }
    }

    pub(crate) const fn weighted(value: S, probability: QuantileProbability) -> Self {
        Self {
            value,
            probability,
            convention: QuantileConvention::MassInverseEcdf,
        }
    }

    pub(crate) const fn frequency(
        value: S,
        probability: QuantileProbability,
        method: QuantileMethod,
    ) -> Self {
        Self {
            value,
            probability,
            convention: QuantileConvention::Frequency(method),
        }
    }
    /// Returns the quantile.
    #[must_use]
    pub const fn value(self) -> S {
        self.value
    }

    /// Returns the probability `p`.
    pub const fn probability(self) -> QuantileProbability {
        self.probability
    }

    /// Returns the quantile definition.
    pub const fn convention(self) -> QuantileConvention {
        self.convention
    }
}

/// Tail of an empirical distribution whose mean is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum TailSide {
    /// The lowest observations, of total weight `p W`.
    Lower,
    /// The highest observations, of total weight `(1 − p) W`.
    Upper,
}

/// Definition of the tail used by a [`TailMeanResult`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum TailMeanConvention {
    /// The mean of the inverse empirical CDF over `[0, p]` or `[p, 1]`. The
    /// observation at the tail boundary contributes only the part of its
    /// weight that lies inside the tail.
    FractionalInverseEcdf,
}

/// Mean of one tail of an empirical distribution, with the tail weight and the
/// boundary observation.
///
/// For observations `xᵢ` with weights `wᵢ` (all `1` without weights) and total
/// weight `W`, the lower tail at probability `p` consists of the lowest
/// observations of total weight `p W`, and the upper tail of the highest
/// observations of total weight `(1 − p) W`. The tail mean is
///
/// ```text
/// T = (Σ_{i ∈ I} wᵢ xᵢ + b x_b) / M,
/// ```
///
/// where `x_b` is the value at which the tail ends, `b` is the part of the
/// combined weight of all observations equal to `x_b` that lies inside the
/// tail, `I` holds the other observations of the tail, and `M` is the tail
/// weight `p W` or `(1 − p) W`. Equivalently, `T` is the mean of the inverse
/// empirical CDF over `[0, p]` or `[p, 1]`.
///
/// The lower tail at `p = 0` and the upper tail at `p = 1` are empty, and
/// computing them is an error. The lower tail at `p = 1` and the upper tail at
/// `p = 0` are the whole sample.
///
/// Weights are measured in observation counts without weights and in the units
/// of the given weights otherwise. The weight accessors return `None` when the
/// value is not representable in `S`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct TailMeanResult<S: RealScalar> {
    mean: S,
    probability: QuantileProbability,
    side: TailSide,
    convention: TailMeanConvention,
    tail_mass: Option<S>,
    boundary_value: S,
    boundary_mass: Option<S>,
    boundary_contribution: Option<S>,
}

impl<S: RealScalar> TailMeanResult<S> {
    const fn new(
        mean: S,
        probability: QuantileProbability,
        side: TailSide,
        tail_mass: Option<S>,
        boundary_value: S,
        boundary_mass: Option<S>,
        boundary_contribution: Option<S>,
    ) -> Self {
        Self {
            mean,
            probability,
            side,
            convention: TailMeanConvention::FractionalInverseEcdf,
            tail_mass,
            boundary_value,
            boundary_mass,
            boundary_contribution,
        }
    }

    /// Returns the tail mean `T`.
    #[must_use]
    pub const fn mean(self) -> S {
        self.mean
    }

    /// Returns the probability `p` that defines the tail.
    pub const fn probability(self) -> QuantileProbability {
        self.probability
    }

    /// Returns the tail side.
    pub const fn side(self) -> TailSide {
        self.side
    }

    /// Returns the tail definition.
    pub const fn convention(self) -> TailMeanConvention {
        self.convention
    }

    /// Returns the tail weight `M`.
    #[must_use]
    pub const fn tail_mass(self) -> Option<S> {
        self.tail_mass
    }

    /// Returns the boundary value `x_b`.
    #[must_use]
    pub const fn boundary_value(self) -> S {
        self.boundary_value
    }

    /// Returns the weight `b` of the boundary value inside the tail.
    #[must_use]
    pub const fn boundary_mass(self) -> Option<S> {
        self.boundary_mass
    }

    /// Returns the boundary contribution `b x_b` to the numerator of `T`.
    #[must_use]
    pub const fn boundary_contribution(self) -> Option<S> {
        self.boundary_contribution
    }
}

#[derive(Debug, Clone, Copy)]
struct WeightedMassSummary<S: RealScalar> {
    total_weight: Option<S>,
    total_mass: ExactMass,
}

fn validate_samples<S: RealScalar>(
    samples: &[S],
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if samples.is_empty() {
        return Err(SciOxideError::invalid_input(
            argument,
            InvalidInputReason::Empty,
            "an empirical distribution requires at least one observation",
        ));
    }
    if samples.iter().any(|value| !value.is_finite()) {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "empirical observations must be finite",
        ));
    }
    Ok(())
}

fn validate_weighted_samples<S: RealScalar>(
    samples: &[(S, S)],
    argument: scioxide_core::Argument,
) -> Result<WeightedMassSummary<S>, SciOxideError> {
    if samples.is_empty() {
        return Err(SciOxideError::invalid_input(
            argument,
            InvalidInputReason::Empty,
            "a weighted empirical distribution requires at least one pair",
        ));
    }
    let mut total_mass = ExactMass::zero();
    let mut positive_count = 0_usize;
    for &(value, weight) in samples {
        if !value.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "weighted empirical observations must be finite",
            ));
        }
        if !weight.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "empirical weights must be finite",
            ));
        }
        if weight < S::ZERO {
            return Err(SciOxideError::invalid_input(
                argument,
                InvalidInputReason::OutOfRange,
                "empirical weights must be non-negative",
            ));
        }
        if weight == S::ZERO {
            continue;
        }
        total_mass.try_add_scalar(weight)?;
        positive_count = positive_count.checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("positive_empirical_support"))
        })?;
    }
    if positive_count == 0 {
        return Err(SciOxideError::invalid_input(
            argument,
            InvalidInputReason::OutOfRange,
            "weighted empirical total weight must be positive",
        ));
    }
    Ok(WeightedMassSummary {
        total_weight: total_mass.to_optional_scalar(),
        total_mass,
    })
}

fn weighted_total_cmp<S: RealScalar>(left: &(S, S), right: &(S, S)) -> Ordering {
    finite_total_cmp(&left.0, &right.0).then_with(|| finite_total_cmp(&left.1, &right.1))
}
