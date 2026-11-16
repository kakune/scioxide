//! Weighted covariance of two variables for the frequency, reliability and analytic weight models.
use crate::central::merge_cross;
use crate::central::{Central, compatible};
use crate::covariance::{correlation, correlation_validity};
use crate::weighted::WeightedMoments;
use crate::{
    Analytic, CorrelationValidity, CovarianceValidity, MomentOrder, MomentWeights, WeightSemantics,
    WeightedMomentsSnapshot,
};
use core::marker::PhantomData;
use scioxide_core::Scaled;
use scioxide_core::{NonFiniteOrigin, RealScalar, SciOxideError};

/// Accumulates the weighted means, marginal moments and covariance of a stream of pairs `(x, y)`.
///
/// Each pair has one weight `wᵢ ≥ 0`, interpreted by the weight model `M` as in
/// [`OnlineWeightedMoments`](crate::OnlineWeightedMoments). The accumulator
/// tracks the weighted co-moment `c = Σ wᵢ (xᵢ − x̄)(yᵢ − ȳ) / W`, with
/// `W = Σ wᵢ`, and the marginal moments of `x` and `y` over the same pairs.
/// Overflow of a marginal estimate does not prevent computing the covariance
/// or correlation.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{Analytic, MomentOrder, OnlineWeightedCovariance};
/// let mut pairs = OnlineWeightedCovariance::<f64, Analytic>::new(MomentOrder::Second);
/// for (x, y, w) in [(1.0, 2.0, 1.0), (2.0, 1.0, 2.0), (4.0, 0.0, 1.0)] {
///     pairs.observe(x, y, w)?;
/// }
/// // Σ wᵢ (xᵢ − x̄)(yᵢ − ȳ) / (n − 1) = −3 / 2 with x̄ = 2.25 and ȳ = 1.
/// assert!(pairs.snapshot().noise_covariance()?.unwrap() < 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// Noise covariance is defined only for analytic weights:
///
/// ```compile_fail
/// use scioxide_stats::{Frequency, MomentOrder, OnlineWeightedCovariance};
/// let pairs = OnlineWeightedCovariance::<f64, Frequency>::new(MomentOrder::Second);
/// pairs.snapshot().noise_covariance();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct OnlineWeightedCovariance<S: RealScalar, M: MomentWeights> {
    pub(crate) core: WeightedCovariance<S>,
    model: PhantomData<M>,
}
impl<S: RealScalar, M: MomentWeights> OnlineWeightedCovariance<S, M> {
    /// Creates an empty accumulator whose marginals track central moments up to `order`.
    pub fn new(order: MomentOrder) -> Self {
        Self {
            core: WeightedCovariance::new(M::SEMANTICS, order),
            model: PhantomData,
        }
    }
    /// Returns the number of pairs with positive weight.
    pub const fn observation_count(&self) -> usize {
        self.core.x.observation_count()
    }
    /// Returns whether no pair with positive weight has been added.
    pub const fn is_empty(&self) -> bool {
        self.core.x.is_empty()
    }
    /// Returns the weight model `M` as a [`WeightSemantics`] value.
    pub const fn weight_semantics(&self) -> WeightSemantics {
        M::SEMANTICS
    }
    /// Returns the highest tracked marginal moment order.
    pub const fn moment_order(&self) -> MomentOrder {
        self.core.x.moment_order()
    }
    /// Returns whether `other` tracks the same moment order and so can be merged.
    pub fn is_merge_compatible(&self, other: &Self) -> bool {
        self.core.x.is_merge_compatible(&other.core.x)
    }
    /// Adds one pair `(x, y)` with weight `weight ≥ 0`.
    ///
    /// A zero weight adds nothing, but all three inputs are still checked.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `x`, `y` or `weight` is NaN or infinite.
    /// - The errors of
    ///   [`OnlineWeightedMoments::observe`](crate::OnlineWeightedMoments::observe).
    pub fn observe(&mut self, x: S, y: S, weight: S) -> Result<(), SciOxideError> {
        self.core.observe(x, y, weight)
    }
    /// Merges the accumulator of a disjoint partition into this one.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the moment orders differ.
    /// - `SizeOverflow` if a combined count overflows `usize`.
    /// - A numerical error of the extended-range arithmetic.
    pub fn merge(&mut self, other: &Self) -> Result<(), SciOxideError> {
        self.core.merge(&other.core)
    }
    /// Returns the current estimates as an immutable snapshot.
    ///
    /// Each estimate is rounded to `S` only when you ask for it.
    pub const fn snapshot(&self) -> WeightedCovarianceSnapshot<S, M> {
        self.core.snapshot()
    }
}

/// Weighted covariance state with the weight model as a runtime value.
///
/// The public types fix the model in their type parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WeightedCovariance<S: RealScalar> {
    x: WeightedMoments<S>,
    y: Central<S>,
    covariance: Scaled<f64>,
}
impl<S: RealScalar> WeightedCovariance<S> {
    pub(crate) fn new(semantics: WeightSemantics, order: MomentOrder) -> Self {
        Self {
            x: WeightedMoments::new(semantics, order),
            y: Central::empty(),
            covariance: Scaled::<f64>::ZERO,
        }
    }
    pub(crate) fn observe(&mut self, x: S, y: S, weight: S) -> Result<(), SciOxideError> {
        if !y.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "coordinates must be finite",
            ));
        }
        let singleton = Self {
            x: WeightedMoments::new(self.x.weight_semantics(), self.x.moment_order())
                .observed(x, weight)?,
            y: Central::singleton(y)?,
            covariance: Scaled::<f64>::ZERO,
        };
        self.merge(&singleton)
    }
    pub(crate) fn merge(&mut self, other: &Self) -> Result<(), SciOxideError> {
        compatible(self.x.is_merge_compatible(&other.x))?;
        if other.x.is_empty() {
            return Ok(());
        }
        if self.x.is_empty() {
            *self = *other;
            return Ok(());
        }
        let (x, p, q) = self.x.merged(&other.x)?;
        let y = self.y.merged(
            other.y,
            crate::arithmetic::Shares::rounded(p, q),
            self.x.moment_order(),
        )?;
        let covariance = merge_cross(
            self.covariance,
            other.covariance,
            (self.x.state.mean()?, self.y.mean()?),
            (other.x.state.mean()?, other.y.mean()?),
            p,
            q,
        )?;
        *self = Self { x, y, covariance };
        Ok(())
    }
    pub(crate) const fn snapshot<M: MomentWeights>(&self) -> WeightedCovarianceSnapshot<S, M> {
        WeightedCovarianceSnapshot {
            state: *self,
            model: PhantomData,
        }
    }
}

/// Immutable estimates from an [`OnlineWeightedCovariance`] accumulator with weight model `M`.
///
/// The notation is that of [`WeightedMomentsSnapshot`]. `c` is the weighted
/// co-moment `Σ wᵢ (xᵢ − x̄)(yᵢ − ȳ) / W`. [`Self::noise_covariance`] exists only
/// for [`Analytic`] weights.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct WeightedCovarianceSnapshot<S: RealScalar, M: MomentWeights> {
    state: WeightedCovariance<S>,
    model: PhantomData<M>,
}
impl<S: RealScalar, M: MomentWeights> WeightedCovarianceSnapshot<S, M> {
    /// Returns the number `n` of pairs with positive weight.
    pub const fn observation_count(self) -> usize {
        self.state.x.observation_count()
    }
    /// Returns the weight model `M` as a [`WeightSemantics`] value.
    pub const fn weight_semantics(self) -> WeightSemantics {
        M::SEMANTICS
    }
    /// Returns the weighted moments of `x` over the same pairs.
    pub fn marginal_x(self) -> WeightedMomentsSnapshot<S, M> {
        self.state.x.snapshot()
    }
    /// Returns the weighted moments of `y` over the same pairs and weights.
    pub fn marginal_y(self) -> WeightedMomentsSnapshot<S, M> {
        let mut y = self.state.x;
        y.state = self.state.y;
        y.snapshot()
    }
    /// Returns which covariance estimates are defined.
    ///
    /// Analytic weights never define a sample covariance. They report
    /// [`CovarianceValidity::PopulationOnly`] and use
    /// [`noise_covariance`](WeightedCovarianceSnapshot::noise_covariance) instead.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the variance divisor cannot be computed.
    pub fn validity(self) -> Result<CovarianceValidity, SciOxideError> {
        Ok(if self.state.x.is_empty() {
            CovarianceValidity::Empty
        } else if self.state.x.sample_factor()?.is_some() {
            CovarianceValidity::SampleDefined
        } else {
            CovarianceValidity::PopulationOnly
        })
    }
    /// Returns the weighted population covariance `c`, or `None` if the snapshot is empty.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if `c` overflows `S`.
    pub fn population_covariance(self) -> Result<Option<S>, SciOxideError> {
        if self.state.x.is_empty() {
            Ok(None)
        } else {
            self.state.covariance.round_to().map(Some)
        }
    }
    /// Returns the unbiased sample covariance `W c / d`.
    ///
    /// `d` is the divisor of [`WeightedMomentsSnapshot::variance_divisor`].
    /// Returns `None` if there are too few pairs, and always for analytic
    /// weights, which use
    /// [`noise_covariance`](WeightedCovarianceSnapshot::noise_covariance) instead.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the result overflows `S`.
    pub fn sample_covariance(self) -> Result<Option<S>, SciOxideError> {
        self.state
            .x
            .sample_factor()?
            .map(|f| self.state.covariance.mul(f)?.round_to())
            .transpose()
    }
    /// Returns the estimated covariance of the weighted means `x̄` and `ȳ`.
    ///
    /// It uses the same factor as
    /// [`WeightedMomentsSnapshot::variance_of_weighted_mean`], applied to `c`.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the result overflows `S`.
    pub fn covariance_of_weighted_means(self) -> Result<Option<S>, SciOxideError> {
        self.state
            .x
            .mean_variance_factor()?
            .map(|f| self.state.covariance.mul(f)?.round_to())
            .transpose()
    }
    /// Returns whether the correlation is defined and, if not, why.
    ///
    /// The pair count is `N` for frequency weights and `n` otherwise.
    pub fn correlation_validity(self) -> CorrelationValidity {
        correlation_validity(
            self.state.x.estimator_count(),
            self.state.x.state.is_constant(),
            self.state.y.is_constant(),
        )
    }
    /// Returns the weighted Pearson correlation `r = c / √(m₂ₓ m₂ᵧ)`.
    ///
    /// `m₂ₓ` and `m₂ᵧ` are the weighted population variances. The result does
    /// not depend on the variance divisor and is clamped to `[−1, 1]`. Returns
    /// `None` unless [`Self::correlation_validity`] is
    /// [`CorrelationValidity::Defined`].
    ///
    /// # Errors
    ///
    /// Returns `NumericalInstability` if `|r|` exceeds `1 + 128ε`, which means
    /// the co-moment lost accuracy to cancellation.
    pub fn correlation(self) -> Result<Option<S>, SciOxideError> {
        correlation(
            self.state.x.estimator_count(),
            self.state.x.state.m2()?,
            self.state.y.m2()?,
            self.state.covariance,
        )
    }
}
impl<S: RealScalar> WeightedCovarianceSnapshot<S, Analytic> {
    /// Returns the noise covariance `Σ wᵢ (xᵢ − x̄)(yᵢ − ȳ) / (n − 1)`, or `None` if `n < 2`.
    ///
    /// It is the cross term of the noise scale of
    /// [`WeightedMomentsSnapshot::noise_scale`].
    ///
    /// # Errors
    ///
    /// Returns a numerical error if the result overflows `S`.
    pub fn noise_covariance(self) -> Result<Option<S>, SciOxideError> {
        if self.observation_count() < 2 {
            return Ok(None);
        }
        self.state
            .covariance
            .mul(self.state.x.total_mass()?)?
            .div(Scaled::<f64>::from_value(
                (self.observation_count() - 1) as f64,
            )?)?
            .round_to()
            .map(Some)
    }
}
