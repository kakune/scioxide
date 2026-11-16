//! Selection and evaluation of scalar statistics.
use crate::{
    Analytic, CircularOptions, CircularSummary, DifferentialEntropyOptions, EntropyOptions,
    Frequency, InformationValue, Mass, OrderResult, PercentileKind, QuantileMethod,
    QuantileProbability, Reliability, TrimSide, Unweighted, WeightModel,
};
use core::marker::PhantomData;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

fn real_identity<S>(values: &[S]) -> &[S] {
    values
}

/// A quantile definition that is valid for weight model `M`.
///
/// Unweighted and frequency-weighted samples accept all nine Hyndman–Fan
/// definitions through [`QuantileRule::new`]; frequency counts apply the
/// definition to the expanded sample. Relative masses have only the inverse of
/// the weighted empirical CDF, [`QuantileRule::inverse_ecdf`], so other
/// definitions do not compile:
///
/// ```compile_fail
/// use scioxide_stats::{Mass, QuantileMethod, QuantileRule};
/// // Relative mass has an inverse ECDF, rather than replicated interpolation ranks.
/// let rule = QuantileRule::<Mass>::new(QuantileMethod::Linear);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantileRule<M: WeightModel = Unweighted> {
    pub(crate) method: QuantileMethod,
    model: PhantomData<M>,
}
impl<M: WeightModel> QuantileRule<M> {
    /// Returns the Hyndman–Fan definition, or [`QuantileMethod::NearestRank`]
    /// for the inverse empirical CDF of relative masses.
    pub const fn method(self) -> QuantileMethod {
        self.method
    }
}
/// Weight models that accept all nine Hyndman–Fan quantile definitions:
/// [`Unweighted`] and [`Frequency`].
pub trait InterpolatedQuantiles: WeightModel {}
impl InterpolatedQuantiles for Unweighted {}
impl InterpolatedQuantiles for Frequency {}
impl<M: InterpolatedQuantiles> QuantileRule<M> {
    /// Selects one of the nine Hyndman–Fan definitions.
    pub const fn new(method: QuantileMethod) -> Self {
        Self {
            method,
            model: PhantomData,
        }
    }
}
impl QuantileRule<Mass> {
    /// Selects the inverse of the weighted empirical CDF, the quantile
    /// definition for relative masses.
    pub const fn inverse_ecdf() -> Self {
        Self {
            method: QuantileMethod::NearestRank,
            model: PhantomData,
        }
    }
}

type RealProjection<T> = fn(&[T]) -> &[<T as NumericScalar>::Real];

/// Proof that a [`Statistic`] is defined for scalar type `T` and weight model
/// `M`.
///
/// Only the [`Statistic`] constructors create it, so every `Statistic` value
/// names a supported combination.
#[derive(Clone, Copy, Debug)]
pub struct StatisticCapability<T: NumericScalar, M: WeightModel, const K: u8> {
    real: Option<RealProjection<T>>,
    model: PhantomData<M>,
}
impl<T: NumericScalar, M: WeightModel, const K: u8> StatisticCapability<T, M, K> {
    const fn numeric() -> Self {
        Self {
            real: None,
            model: PhantomData,
        }
    }
}
impl<S: RealScalar, M: WeightModel, const K: u8> StatisticCapability<S, M, K> {
    const fn real() -> Self {
        Self {
            real: Some(real_identity::<S>),
            model: PhantomData,
        }
    }
}

/// A scalar statistic of a sample, for scalar type `T` and weight model `M`.
///
/// Evaluate it with [`statistic`](fn@statistic),
/// [`StatisticsWorkspace::evaluate`](crate::StatisticsWorkspace::evaluate),
/// [`statistics_into`](crate::statistics_into) or
/// [`rolling_statistics_into`](crate::rolling_statistics_into). Each
/// constructor exists only where the statistic is defined, so an unsupported
/// combination does not compile:
///
/// | Constructors | `T` | `M` |
/// |---|---|---|
/// | [`sum`](Self::sum), [`mean`](Self::mean) | real or complex | [`Unweighted`]; for real `T`, every weight model |
/// | [`variance`](Self::variance), [`standard_deviation`](Self::standard_deviation) | real or complex | [`Unweighted`] |
/// | [`geometric_mean`](Self::geometric_mean), [`harmonic_mean`](Self::harmonic_mean), [`power_mean`](Self::power_mean), [`quantile`](Self::quantile), [`circular`](Self::circular) | real | [`Unweighted`], [`Frequency`], [`Mass`] |
/// | [`expectile`](Self::expectile) | real | [`Unweighted`], [`Mass`] |
/// | all others | real | [`Unweighted`] |
///
/// The median is `Statistic::quantile(QuantileProbability::new(0.5)?, rule)`.
/// For complex data, the variance uses the squared modulus `|xᵢ − x̄|²`.
/// Invalid observations and undefined values are errors at evaluation.
///
/// ```
/// use scioxide_stats::{Statistic, statistic};
///
/// // (1 + 2 + 3) / 3 = 2
/// let value = statistic(&[1.0_f64, 2.0, 3.0], scioxide_stats::Unweighted, Statistic::mean())?;
/// assert_eq!(value.location(), Some(2.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// ```compile_fail
/// use scioxide_stats::{Mass, Statistic};
/// // A sample variance estimator is not defined for arbitrary relative masses.
/// let statistic = Statistic::<f64, Mass>::variance(1);
/// ```
///
/// ```compile_fail
/// use scioxide_core::Complex;
/// use scioxide_stats::Statistic;
/// let statistic = Statistic::<Complex<f64>>::minimum();
/// ```
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Statistic<T: NumericScalar, M: WeightModel = Unweighted> {
    /// Sum `Σ xᵢ`, or `Σ wᵢ xᵢ` with weights.
    Sum(StatisticCapability<T, M, 0>),
    /// Arithmetic mean, or weighted mean `Σ wᵢ xᵢ / Σ wᵢ`.
    Mean(StatisticCapability<T, M, 0>),
    /// Variance `Σ |xᵢ − x̄|² / (n − ddof)`.
    Variance(usize, StatisticCapability<T, M, 1>),
    /// Standard deviation, the square root of the variance.
    StandardDeviation(usize, StatisticCapability<T, M, 1>),
    /// Smallest observation.
    Minimum(StatisticCapability<T, M, 2>),
    /// Largest observation.
    Maximum(StatisticCapability<T, M, 2>),
    /// Geometric mean of nonnegative observations.
    GeometricMean(StatisticCapability<T, M, 3>),
    /// Harmonic mean of nonnegative observations.
    HarmonicMean(StatisticCapability<T, M, 3>),
    /// Power mean of order `p` of nonnegative observations.
    PowerMean(T::Real, StatisticCapability<T, M, 3>),
    /// Geometric standard deviation `exp(sd(log x))`.
    GeometricStandardDeviation(usize, StatisticCapability<T, M, 2>),
    /// Quantile of probability `p` under a quantile rule.
    Quantile(
        QuantileProbability,
        QuantileRule<M>,
        StatisticCapability<T, M, 4>,
    ),
    /// Median absolute deviation divided by a scale.
    MedianAbsoluteDeviation(QuantileMethod, T::Real, StatisticCapability<T, M, 2>),
    /// Trimmed mean.
    TrimmedMean(T::Real, TrimSide, StatisticCapability<T, M, 2>),
    /// Unbiased sample L-moment, or L-moment ratio.
    LMoment(
        usize,
        crate::LMomentNormalization,
        StatisticCapability<T, M, 2>,
    ),
    /// Most frequent value and its count.
    Mode(StatisticCapability<T, M, 2>),
    /// Expectile, the asymmetric least-squares location.
    Expectile(QuantileProbability, StatisticCapability<T, M, 7>),
    /// Interquartile range.
    Iqr(QuantileMethod, StatisticCapability<T, M, 2>),
    /// Percentage of observations relative to a score.
    PercentileOfScore(T::Real, PercentileKind, StatisticCapability<T, M, 2>),
    /// Tie correction factor of rank tests.
    TieCorrection(StatisticCapability<T, M, 2>),
    /// Circular mean direction and dispersion.
    Circular(CircularOptions<T::Real>, StatisticCapability<T, M, 5>),
    /// Shannon entropy of the observations read as probability masses.
    Entropy(EntropyOptions<T::Real>, StatisticCapability<T, M, 6>),
    /// Spacing estimate of the differential entropy.
    DifferentialEntropy(
        DifferentialEntropyOptions<T::Real>,
        StatisticCapability<T, M, 6>,
    ),
}
/// Weight models for which [`Statistic::sum`] and [`Statistic::mean`] are
/// defined: [`Unweighted`] for real and complex data, and every weight model
/// for real data.
pub trait LocationModel<T: NumericScalar>: WeightModel {
    /// Returns the real view of the observations used by weighted sums and
    /// means.
    #[doc(hidden)]
    fn real_projection() -> Option<RealProjection<T>>;
}
impl<T: NumericScalar> LocationModel<T> for Unweighted {
    fn real_projection() -> Option<RealProjection<T>> {
        None
    }
}
macro_rules! weighted_locations {
    ($($model:ty),*) => {$ (
        impl<S: RealScalar> LocationModel<S> for $model {
            fn real_projection() -> Option<RealProjection<S>> { Some(real_identity::<S>) }
        }
    )*};
}
weighted_locations!(Frequency, Mass, Reliability, Analytic);
impl<T: NumericScalar, M: LocationModel<T>> Statistic<T, M> {
    /// Selects the sum `Σ xᵢ`, or `Σ wᵢ xᵢ` with weights.
    ///
    /// The sum of an empty unweighted sample is zero.
    pub fn sum() -> Self {
        Self::Sum(StatisticCapability {
            real: M::real_projection(),
            model: PhantomData,
        })
    }
    /// Selects the arithmetic mean `(1/n) Σ xᵢ`, or the weighted mean
    /// `Σ wᵢ xᵢ / Σ wᵢ`. The sample must not be empty.
    pub fn mean() -> Self {
        Self::Mean(StatisticCapability {
            real: M::real_projection(),
            model: PhantomData,
        })
    }
}
impl<T: NumericScalar> Statistic<T> {
    /// Selects the variance `Σ |xᵢ − x̄|² / (n − ddof)`.
    ///
    /// Use `ddof = 1` for the unbiased sample variance and `ddof = 0` for the
    /// population variance. Evaluation fails when `n ≤ ddof`.
    pub const fn variance(ddof: usize) -> Self {
        Self::Variance(ddof, StatisticCapability::numeric())
    }
    /// Selects the standard deviation, the square root of
    /// [`Self::variance`]`(ddof)`.
    pub const fn standard_deviation(ddof: usize) -> Self {
        Self::StandardDeviation(ddof, StatisticCapability::numeric())
    }
}
impl<S: RealScalar, M: crate::EmpiricalWeights> Statistic<S, M> {
    /// Selects the geometric mean `exp(Σ wᵢ log xᵢ / Σ wᵢ)` of nonnegative
    /// observations. A zero observation with positive weight gives zero.
    pub const fn geometric_mean() -> Self {
        Self::GeometricMean(StatisticCapability::real())
    }
    /// Selects the harmonic mean `Σ wᵢ / Σ (wᵢ / xᵢ)` of nonnegative
    /// observations. A zero observation with positive weight gives zero.
    pub const fn harmonic_mean() -> Self {
        Self::HarmonicMean(StatisticCapability::real())
    }
    /// Selects the power mean `(Σ wᵢ xᵢ^p / Σ wᵢ)^(1/p)` of nonnegative
    /// observations.
    ///
    /// `p = 0` gives the geometric mean, and `p = ±∞` the largest and
    /// smallest observation.
    pub const fn power_mean(power: S) -> Self {
        Self::PowerMean(power, StatisticCapability::real())
    }
    /// Selects the quantile of probability `p` under `rule`. `p = 0.5` gives
    /// the median.
    pub const fn quantile(p: QuantileProbability, rule: QuantileRule<M>) -> Self {
        Self::Quantile(p, rule, StatisticCapability::real())
    }
    /// Selects the circular mean direction and dispersion of angles.
    pub const fn circular(options: CircularOptions<S>) -> Self {
        Self::Circular(options, StatisticCapability::real())
    }
}
impl<S: RealScalar> Statistic<S> {
    /// Selects the smallest observation.
    pub const fn minimum() -> Self {
        Self::Minimum(StatisticCapability::real())
    }
    /// Selects the largest observation.
    pub const fn maximum() -> Self {
        Self::Maximum(StatisticCapability::real())
    }
    /// Selects the geometric standard deviation `exp(s)`, where `s` is the
    /// standard deviation of `log xᵢ` with divisor `n − ddof`.
    pub const fn geometric_standard_deviation(ddof: usize) -> Self {
        Self::GeometricStandardDeviation(ddof, StatisticCapability::real())
    }
    /// Selects the median absolute deviation `median |xᵢ − median x| / scale`.
    ///
    /// Both medians use `method`. `scale = 1` gives the raw MAD. Evaluation
    /// fails unless `scale` is positive and finite.
    pub const fn median_absolute_deviation(method: QuantileMethod, scale: S) -> Self {
        Self::MedianAbsoluteDeviation(method, scale, StatisticCapability::real())
    }
    /// Selects the trimmed mean: the mean after removing `⌊proportion · n⌋`
    /// observations from each end chosen by `side`.
    ///
    /// Evaluation fails unless `proportion` is in `[0, 1)`.
    pub const fn trimmed_mean(proportion: S, side: TrimSide) -> Self {
        Self::TrimmedMean(proportion, side, StatisticCapability::real())
    }
    /// Selects the unbiased sample L-moment `λᵣ` of order `r = order`, or the
    /// L-moment ratio `τᵣ = λᵣ / λ₂` for `r ≥ 3` with
    /// [`LMomentNormalization::Standardized`](crate::LMomentNormalization::Standardized).
    ///
    /// Evaluation fails unless `1 ≤ order ≤ n`.
    ///
    /// # References
    ///
    /// J. R. M. Hosking, "L-moments: analysis and estimation of distributions
    /// using linear combinations of order statistics", *Journal of the Royal
    /// Statistical Society B* 52(1), 105–124, 1990.
    pub const fn l_moment(order: usize, standardize: crate::LMomentNormalization) -> Self {
        Self::LMoment(order, standardize, StatisticCapability::real())
    }
    /// Selects the most frequent value and its count. Among equally frequent
    /// values the smallest is returned.
    pub const fn mode() -> Self {
        Self::Mode(StatisticCapability::real())
    }
    /// Selects the interquartile range `Q(0.75) − Q(0.25)`, with both quartiles
    /// under `method`.
    pub const fn iqr(method: QuantileMethod) -> Self {
        Self::Iqr(method, StatisticCapability::real())
    }
    /// Selects the percentage of observations below or at `score`, under the
    /// convention `kind`.
    pub const fn percentile_of_score(score: S, kind: PercentileKind) -> Self {
        Self::PercentileOfScore(score, kind, StatisticCapability::real())
    }
    /// Selects the tie correction factor `1 − Σ (tⱼ³ − tⱼ) / (n³ − n)` of rank
    /// tests, where `tⱼ` is the size of the `j`-th group of tied values.
    pub const fn tie_correction() -> Self {
        Self::TieCorrection(StatisticCapability::real())
    }
    /// Selects the Shannon entropy `−Σ pᵢ log pᵢ`, where the observations are
    /// the probability masses `pᵢ`. The logarithm base and the normalization
    /// come from `options`.
    pub const fn entropy(options: EntropyOptions<S>) -> Self {
        Self::Entropy(options, StatisticCapability::real())
    }
    /// Selects a spacing estimate of the differential entropy of a continuous
    /// distribution from its sample.
    pub const fn differential_entropy(options: DifferentialEntropyOptions<S>) -> Self {
        Self::DifferentialEntropy(options, StatisticCapability::real())
    }
}
/// Weight models accepted by [`Statistic::expectile`]: [`Unweighted`] and
/// [`Mass`].
pub trait ExpectileWeights: WeightModel {}
impl ExpectileWeights for Unweighted {}
impl ExpectileWeights for Mass {}
impl<S: RealScalar, M: ExpectileWeights> Statistic<S, M> {
    /// Selects the expectile of level `τ = p`, the asymmetric least-squares
    /// location.
    ///
    /// The expectile `e` solves
    ///
    /// ```text
    /// τ Σ_{xᵢ > e} wᵢ (xᵢ − e) = (1 − τ) Σ_{xᵢ < e} wᵢ (e − xᵢ)
    /// ```
    ///
    /// with unit weights `wᵢ = 1` for unweighted data. `τ = 0.5` gives the mean,
    /// and `τ = 0` and `τ = 1` give the smallest and largest observation with
    /// positive weight. Zero-weight observations are skipped before their values
    /// are read.
    ///
    /// The solver selects the bracketing observations by median selection with
    /// compensated sums, without sorting and without an iteration tolerance.
    /// The returned [`ExpectileResult`](crate::ExpectileResult) reports the
    /// balance residual at the rounded result. It is a diagnostic, not an error
    /// bound.
    ///
    /// Evaluation fails if no observation has positive weight, a weight is
    /// negative or non-finite, an observation with positive weight is not
    /// finite, or the computation breaks down.
    ///
    /// ```
    /// use scioxide_stats::{QuantileProbability, Statistic, Unweighted, statistic};
    ///
    /// // 0.75 · (10 − e) = 0.25 · (e − 0) gives e = 7.5.
    /// let p = QuantileProbability::new(0.75)?;
    /// let result = statistic(&[0.0_f64, 10.0], Unweighted, Statistic::expectile(p))?;
    /// assert_eq!(result.scalar()?, 7.5);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// W. K. Newey and J. L. Powell, "Asymmetric least squares estimation and
    /// testing", *Econometrica* 55(4), 819–847, 1987.
    pub const fn expectile(p: QuantileProbability) -> Self {
        Self::Expectile(p, StatisticCapability::real())
    }
}

/// The value of a [`Statistic`], in the form that matches the statistic.
///
/// Use [`Self::scalar`] to read an ordinary real value, or match on the
/// variant to read the conventions and diagnostics of order, circular and
/// information results.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StatisticValue<T: NumericScalar> {
    /// A value in the scalar type `T`: sums, means, extrema, geometric,
    /// harmonic and power means, geometric standard deviation, median absolute
    /// deviation, trimmed means and L-moments.
    Location(T),
    /// A variance or standard deviation, which is real and nonnegative also
    /// for complex data.
    Dispersion(T::Real),
    /// A quantile, mode, expectile, interquartile range, percentile of score
    /// or tie correction, with its convention and diagnostics.
    Order(OrderResult<T::Real>),
    /// A circular summary, including whether the mean direction is defined.
    Circular(CircularSummary<T::Real>),
    /// An entropy, which may be finite or divergent.
    Information(InformationValue<T::Real>),
}
impl<T: NumericScalar> StatisticValue<T> {
    /// Returns the order result, or `None` for other variants.
    pub const fn ordered(self) -> Option<OrderResult<T::Real>> {
        match self {
            Self::Order(x) => Some(x),
            _ => None,
        }
    }
    /// Returns the location, or `None` for other variants.
    pub const fn location(self) -> Option<T> {
        match self {
            Self::Location(x) => Some(x),
            _ => None,
        }
    }
    /// Returns the dispersion, or `None` for other variants.
    pub const fn dispersion(self) -> Option<T::Real> {
        match self {
            Self::Dispersion(x) => Some(x),
            _ => None,
        }
    }
}

pub(crate) fn legacy_weights<S: RealScalar>(
    weights: crate::weights::WeightData<'_, S>,
) -> Option<crate::KernelWeights<'_, S>> {
    match weights {
        crate::weights::WeightData::None => None,
        crate::weights::WeightData::Frequency(w) => Some(crate::KernelWeights::Frequency(w)),
        crate::weights::WeightData::Mass(w)
        | crate::weights::WeightData::Reliability(w)
        | crate::weights::WeightData::Analytic(w) => Some(crate::KernelWeights::Mass(w)),
    }
}

pub(crate) fn evaluate<T: NumericScalar, M: WeightModel>(
    values: &[T],
    weights: crate::weights::WeightData<'_, T::Real>,
    statistic: Statistic<T, M>,
    workspace: &mut crate::RawDescriptiveWorkspace<T::Real>,
) -> Result<StatisticValue<T>, SciOxideError> {
    use crate::OrderedStatistic as O;
    use crate::RobustStatistic as R;
    use Statistic as D;
    let unsupported = || {
        crate::error_helpers::invalid(
            scioxide_core::argument!("statistic"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid statistic capability",
        )
    };
    let (real, request) = match statistic {
        D::Sum(proof) | D::Mean(proof) => {
            let sum = matches!(statistic, D::Sum(_));
            if matches!(weights, crate::weights::WeightData::None) {
                return crate::numeric_statistic(
                    values,
                    if sum {
                        crate::ArithmeticStatistic::Sum
                    } else {
                        crate::ArithmeticStatistic::Mean
                    },
                )
                .map(|x| match x {
                    crate::ArithmeticValue::Location(x) => StatisticValue::Location(x),
                    crate::ArithmeticValue::Dispersion(x) => StatisticValue::Dispersion(x),
                });
            }
            (
                proof.real.ok_or_else(unsupported)?,
                if sum { R::Sum } else { R::Mean },
            )
        }
        D::Variance(ddof, _) | D::StandardDeviation(ddof, _) => {
            return crate::numeric_statistic(
                values,
                if matches!(statistic, D::Variance(..)) {
                    crate::ArithmeticStatistic::Variance(ddof)
                } else {
                    crate::ArithmeticStatistic::StandardDeviation(ddof)
                },
            )
            .map(|x| match x {
                crate::ArithmeticValue::Location(x) => StatisticValue::Location(x),
                crate::ArithmeticValue::Dispersion(x) => StatisticValue::Dispersion(x),
            });
        }
        D::Minimum(proof) => (proof.real.ok_or_else(unsupported)?, R::Minimum),
        D::Maximum(proof) => (proof.real.ok_or_else(unsupported)?, R::Maximum),
        D::GeometricMean(proof) => (proof.real.ok_or_else(unsupported)?, R::GeometricMean),
        D::HarmonicMean(proof) => (proof.real.ok_or_else(unsupported)?, R::HarmonicMean),
        D::PowerMean(p, proof) => (proof.real.ok_or_else(unsupported)?, R::PowerMean(p)),
        D::GeometricStandardDeviation(d, proof) => (
            proof.real.ok_or_else(unsupported)?,
            R::GeometricStandardDeviation(d),
        ),
        D::MedianAbsoluteDeviation(method, scale, proof) => (
            proof.real.ok_or_else(unsupported)?,
            R::MedianAbsoluteDeviation { method, scale },
        ),
        D::TrimmedMean(proportion, side, proof) => (
            proof.real.ok_or_else(unsupported)?,
            R::TrimmedMean { proportion, side },
        ),
        D::LMoment(order, standardize, proof) => (
            proof.real.ok_or_else(unsupported)?,
            R::LMoment { order, standardize },
        ),
        D::Quantile(p, rule, proof) => {
            return order_evaluate(
                (proof.real.ok_or_else(unsupported)?)(values),
                weights,
                O::Quantile(p, rule.method),
                workspace,
            )
            .map(StatisticValue::Order);
        }
        D::Mode(proof) => {
            return order_evaluate(
                (proof.real.ok_or_else(unsupported)?)(values),
                weights,
                O::Mode,
                workspace,
            )
            .map(StatisticValue::Order);
        }
        D::Expectile(p, proof) => {
            return order_evaluate(
                (proof.real.ok_or_else(unsupported)?)(values),
                weights,
                O::Expectile(p),
                workspace,
            )
            .map(StatisticValue::Order);
        }
        D::Iqr(method, proof) => {
            return order_evaluate(
                (proof.real.ok_or_else(unsupported)?)(values),
                weights,
                O::Iqr(method),
                workspace,
            )
            .map(StatisticValue::Order);
        }
        D::PercentileOfScore(score, kind, proof) => {
            return order_evaluate(
                (proof.real.ok_or_else(unsupported)?)(values),
                weights,
                O::PercentileOfScore(score, kind),
                workspace,
            )
            .map(StatisticValue::Order);
        }
        D::TieCorrection(proof) => {
            return order_evaluate(
                (proof.real.ok_or_else(unsupported)?)(values),
                weights,
                O::TieCorrection,
                workspace,
            )
            .map(StatisticValue::Order);
        }
        D::Circular(options, proof) => {
            return crate::circular_statistics(
                (proof.real.ok_or_else(unsupported)?)(values),
                legacy_weights(weights),
                options,
            )
            .map(StatisticValue::Circular);
        }
        D::Entropy(options, proof) => {
            return crate::entropy_with_options(
                (proof.real.ok_or_else(unsupported)?)(values),
                None,
                options,
            )
            .map(StatisticValue::Information);
        }
        D::DifferentialEntropy(options, proof) => {
            return workspace
                .differential_entropy_with_options(
                    (proof.real.ok_or_else(unsupported)?)(values),
                    options,
                )
                .map(StatisticValue::Information);
        }
    };
    workspace
        .evaluate(real(values), legacy_weights(weights), request)
        .map(|x| StatisticValue::Location(T::from_real(x)))
}

fn order_evaluate<S: RealScalar>(
    values: &[S],
    weights: crate::weights::WeightData<'_, S>,
    request: crate::OrderedStatistic<S>,
    workspace: &mut crate::RawDescriptiveWorkspace<S>,
) -> Result<OrderResult<S>, SciOxideError> {
    let order = &mut workspace.order;
    match weights {
        crate::weights::WeightData::None => order.evaluate(values, request),
        crate::weights::WeightData::Frequency(counts) => {
            order.begin(values.len(), scioxide_core::argument!("workspace"))?;
            for (i, (&value, &count)) in values.iter().zip(counts).enumerate() {
                order.push_frequency(
                    value,
                    crate::weights::frequency_index(count)?,
                    i,
                    scioxide_core::argument!("values"),
                )?;
            }
            order.evaluate_weighted(request, true)
        }
        crate::weights::WeightData::Mass(masses) => {
            order.begin(values.len(), scioxide_core::argument!("workspace"))?;
            for (i, (&value, &mass)) in values.iter().zip(masses).enumerate() {
                order.push(value, mass, i, scioxide_core::argument!("values"))?;
            }
            order.evaluate_weighted(request, false)
        }
        _ => Err(crate::error_helpers::invalid(
            scioxide_core::argument!("weights"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid order capability",
        )),
    }
}

/// Computes one statistic of a real or complex sample, with optional weights.
///
/// `weights` is [`Unweighted`] or a [`Weights`](crate::Weights) slice with one
/// weight per observation. Sums, means, variances, standard deviations,
/// extrema, geometric, harmonic and power means, circular statistics and
/// entropy run without scratch. Other statistics allocate scratch for one
/// call; use [`StatisticsWorkspace::evaluate`](crate::StatisticsWorkspace::evaluate)
/// to reuse it. For a strided array, a mask or several batches, use
/// [`statistics_into`](crate::statistics_into).
///
/// # Errors
///
/// - `DimensionMismatch` if the number of weights differs from `values.len()`.
/// - `InvalidInput` if the sample is empty (except for an unweighted sum), has
///   too few observations for the statistic, or a parameter of the statistic
///   is out of range.
/// - An error if an observation or weight is NaN or infinite, or a weight is
///   negative.
/// - `NumericalInstability` (`Overflow`) if the result overflows.
/// - `AllocationFailure` if scratch cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{
///     QuantileMethod, QuantileProbability, QuantileRule, Statistic, Unweighted, statistic,
/// };
///
/// // The median of 3, 1, 2 is 2.
/// let rule = QuantileRule::new(QuantileMethod::Linear);
/// let median = Statistic::quantile(QuantileProbability::new(0.5)?, rule);
/// let value = statistic(&[3.0_f64, 1.0, 2.0], Unweighted, median)?;
/// assert!((value.scalar()? - 2.0).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn statistic<'a, T: NumericScalar, W: crate::BatchWeights<'a, T::Real>>(
    values: &[T],
    weights: W,
    request: Statistic<T, W::Model>,
) -> Result<StatisticValue<T>, SciOxideError> {
    let weights = crate::weights::batch_sealed::Sealed::data(weights);
    if let Some(len) = weights.len()
        && len != values.len()
    {
        return Err(SciOxideError::dimension_mismatch([values.len()], [len]));
    }
    let direct = match &request {
        Statistic::Variance(ddof, _) => Some(crate::ArithmeticStatistic::Variance(*ddof)),
        Statistic::StandardDeviation(ddof, _) => {
            Some(crate::ArithmeticStatistic::StandardDeviation(*ddof))
        }
        Statistic::Sum(_) if matches!(weights, crate::weights::WeightData::None) => {
            Some(crate::ArithmeticStatistic::Sum)
        }
        Statistic::Mean(_) if matches!(weights, crate::weights::WeightData::None) => {
            Some(crate::ArithmeticStatistic::Mean)
        }
        _ => None,
    };
    if let Some(direct) = direct {
        return crate::numeric_statistic(values, direct).map(|value| match value {
            crate::ArithmeticValue::Location(value) => StatisticValue::Location(value),
            crate::ArithmeticValue::Dispersion(value) => StatisticValue::Dispersion(value),
        });
    }
    let mut workspace = crate::RawDescriptiveWorkspace::default();
    if !matches!(
        request,
        Statistic::Sum(_)
            | Statistic::Mean(_)
            | Statistic::Variance(..)
            | Statistic::StandardDeviation(..)
            | Statistic::Minimum(_)
            | Statistic::Maximum(_)
            | Statistic::GeometricMean(_)
            | Statistic::HarmonicMean(_)
            | Statistic::PowerMean(..)
            | Statistic::Circular(..)
            | Statistic::Entropy(..)
    ) {
        workspace.reserve(values.len())?;
    }
    evaluate(values, weights, request, &mut workspace)
}

impl<S: RealScalar> StatisticValue<S> {
    /// Returns the value as a real scalar.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` for circular and information values.
    /// - An error if an order result, such as a percentage, is not
    ///   representable in `S`.
    #[inline]
    pub fn scalar(self) -> Result<S, SciOxideError> {
        match self {
            Self::Location(x) | Self::Dispersion(x) => Ok(x),
            Self::Order(x) => x.scalar(),
            _ => Err(crate::error_helpers::invalid(
                scioxide_core::argument!("value"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "result is not an ordinary scalar",
            )),
        }
    }
}
