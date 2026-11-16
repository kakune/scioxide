//! Descriptive statistics computed from a moments snapshot.

use crate::{BiasCorrection, KurtosisConvention, MomentOrder, MomentsSnapshot};
use scioxide_core::Scaled;
use scioxide_core::{RealScalar, SciOxideError};

impl<S: RealScalar> MomentsSnapshot<S> {
    /// Returns the highest accumulated moment order.
    pub const fn moment_order(self) -> MomentOrder {
        self.order
    }

    /// Returns the smallest observation, or `None` for an empty sample.
    pub fn minimum(self) -> Option<S> {
        (self.count() > 0).then_some(self.state.minimum())
    }

    /// Returns the largest observation, or `None` for an empty sample.
    pub fn maximum(self) -> Option<S> {
        (self.count() > 0).then_some(self.state.maximum())
    }

    /// Computes the population central moment `m_k = (1/n) Σ (xᵢ − x̄)^k` of
    /// order `k = order`.
    ///
    /// `m₀ = 1` and `m₁ = 0`. Returns `None` for an empty sample.
    ///
    /// # Errors
    ///
    /// - `NotImplemented` if `order` exceeds [`Self::moment_order`].
    /// - `NumericalInstability` (`Overflow`) if the result overflows `S`.
    pub fn central_moment(self, order: usize) -> Result<Option<S>, SciOxideError> {
        let value = self.state.moment(order, self.order)?;
        if self.count() == 0 {
            Ok(None)
        } else {
            value.round_to().map(Some)
        }
    }

    /// Computes the Fisher–Pearson skewness `g₁ = m₃ / m₂^(3/2)`, or its
    /// adjusted form `G₁` (see [`BiasCorrection`]).
    ///
    /// Returns `None` for an empty or constant sample, and for fewer than
    /// three observations with [`BiasCorrection::Corrected`].
    ///
    /// # Errors
    ///
    /// - `NotImplemented` if the third moment was not accumulated.
    /// - `NumericalInstability` (`Overflow`) if the result overflows `S`.
    pub fn skewness(self, correction: BiasCorrection) -> Result<Option<S>, SciOxideError> {
        self.state.skewness(self.count(), self.order, correction)
    }

    /// Computes the kurtosis `m₄ / m₂²`, as excess kurtosis or not according to
    /// [`KurtosisConvention`], and optionally adjusted (see
    /// [`BiasCorrection`]).
    ///
    /// Returns `None` for an empty or constant sample, and for fewer than four
    /// observations with [`BiasCorrection::Corrected`].
    ///
    /// # Errors
    ///
    /// - `NotImplemented` if the fourth moment was not accumulated.
    /// - `NumericalInstability` (`Overflow`) if the result overflows `S`.
    pub fn kurtosis(
        self,
        correction: BiasCorrection,
        convention: KurtosisConvention,
    ) -> Result<Option<S>, SciOxideError> {
        self.state
            .kurtosis(self.count(), self.order, correction, convention)
    }

    /// Computes the k-statistic `k_r`, the unbiased estimator of the cumulant
    /// `κ_r`, for `r = order` in 1–4.
    ///
    /// The estimators use central moments:
    ///
    /// ```text
    /// k₁ = x̄
    /// k₂ = n m₂ / (n − 1)
    /// k₃ = n² m₃ / ((n − 1)(n − 2))
    /// k₄ = n² ((n + 1) m₄ − 3 (n − 1) m₂²) / ((n − 1)(n − 2)(n − 3))
    /// ```
    ///
    /// where `n` is the number of observations and `m_k` are the population
    /// central moments. Returns `None` for fewer than `order` observations.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) if `order` is not in 1–4.
    /// - `NotImplemented` if moment `order` was not accumulated.
    /// - `NumericalInstability` (`Overflow`) if the result overflows `S`.
    ///
    /// # References
    ///
    /// M. G. Kendall and A. Stuart, *The Advanced Theory of Statistics*,
    /// Vol. 1.
    pub fn k_statistic(self, order: usize) -> Result<Option<S>, SciOxideError> {
        self.state
            .k_statistic(order, self.count(), self.order)?
            .map(|value| value.round_to())
            .transpose()
    }

    /// Computes an unbiased estimate of the variance of `k₁` or `k₂`.
    ///
    /// ```text
    /// V̂(k₁) = k₂ / n
    /// V̂(k₂) = (2 n k₂² + (n − 1) k₄) / (n (n + 1))
    /// ```
    ///
    /// where `V̂` denotes the variance estimate. Returns `None` for fewer than
    /// two observations (order 1) or four observations (order 2). The order-2
    /// estimate can be negative for an individual sample.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `order` is not 1 or 2.
    /// - `NotImplemented` if order 2 is requested and the fourth moment was
    ///   not accumulated.
    /// - `NumericalInstability` (`Overflow`) if the result overflows `S`.
    pub fn k_statistic_variance(self, order: usize) -> Result<Option<S>, SciOxideError> {
        self.state
            .k_statistic_variance(order, self.count(), self.order)?
            .map(|value| value.round_to())
            .transpose()
    }

    /// Computes the variance `Σ (xᵢ − x̄)² / (n − ddof)`.
    ///
    /// Use `ddof = 0` for the population variance and `ddof = 1` for the
    /// unbiased sample variance. Returns `None` when `n ≤ ddof`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if the result overflows `S`.
    pub fn variance(self, ddof: usize) -> Result<Option<S>, SciOxideError> {
        self.variance_scaled(ddof)?
            .map(|value| value.round_to())
            .transpose()
    }

    /// Computes the standard deviation, the square root of
    /// [`Self::variance`]`(ddof)`.
    ///
    /// Returns `None` when `n ≤ ddof`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if the result overflows `S`.
    pub fn standard_deviation(self, ddof: usize) -> Result<Option<S>, SciOxideError> {
        if let Some(variance) = self.native_variance(ddof) {
            return Scaled::<f64>::from_value(variance.sqrt())?
                .round_to()
                .map(Some);
        }
        self.variance_scaled(ddof)?
            .map(|value| value.sqrt()?.round_to())
            .transpose()
    }

    /// Computes the standard error of the mean, `s / √n`, where `s` is the
    /// standard deviation with divisor `n − ddof`.
    ///
    /// The usual choice is `ddof = 1`. Returns `None` for fewer than two
    /// observations or when `n ≤ ddof`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if the result overflows `S`.
    pub fn standard_error(self, ddof: usize) -> Result<Option<S>, SciOxideError> {
        if self.count() < 2 {
            return Ok(None);
        }
        if let Some(variance) = self.native_variance(ddof) {
            return Scaled::<f64>::from_value((variance / self.count() as f64).sqrt())?
                .round_to()
                .map(Some);
        }
        self.variance_scaled(ddof)?
            .map(|value| {
                value
                    .div(Scaled::<f64>::from_value((self.count()) as f64)?)?
                    .sqrt()?
                    .round_to()
            })
            .transpose()
    }

    /// Computes the coefficient of variation `s / x̄`, where `s` is the standard
    /// deviation with divisor `n − ddof` and `x̄` is the signed mean.
    ///
    /// The usual choice is `ddof = 0`. The ratio is formed before rounding to
    /// `S`, so it is defined even when `x̄` or `s` alone underflows to zero.
    /// Returns `None` for an empty sample, a zero mean, or `n ≤ ddof`.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if the result overflows `S`.
    pub fn coefficient_of_variation(self, ddof: usize) -> Result<Option<S>, SciOxideError> {
        if self.count() == 0 || self.state.mean()?.total()?.is_zero() {
            return Ok(None);
        }
        self.variance_scaled(ddof)?
            .map(|value| value.sqrt()?.div(self.state.mean()?.total()?)?.round_to())
            .transpose()
    }

    // Native centered states bound powers by 2^257 and counts by 2^53 here,
    // so products, divisions and square roots stay normal in binary64.
    fn native_variance(self, ddof: usize) -> Option<f64> {
        if self.count() <= ddof || self.count() as u128 > (1_u128 << 53) {
            return None;
        }
        self.state
            .native_second()
            .map(|m2| m2 * self.count() as f64 / (self.count() - ddof) as f64)
    }

    pub(crate) fn variance_scaled(self, ddof: usize) -> Result<Option<Scaled<f64>>, SciOxideError> {
        if self.count() <= ddof {
            return Ok(None);
        }
        if let Some(variance) = self.native_variance(ddof) {
            return Scaled::<f64>::from_value(variance).map(Some);
        }
        self.state
            .m2()?
            .mul(Scaled::<f64>::from_value((self.count()) as f64)?)?
            .div(Scaled::<f64>::from_value((self.count() - ddof) as f64)?)
            .map(Some)
    }
}
