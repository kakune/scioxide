//! Scalar summaries with conventional unweighted definitions.
use crate::{QuantileMethod, QuantileProbability, QuantileRule, Statistic, Unweighted, statistic};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
fn not_scalar() -> SciOxideError {
    crate::error_helpers::invalid(
        scioxide_core::argument!("values"),
        scioxide_core::InvalidInputReason::Inconsistent,
        "result is not an ordinary scalar",
    )
}

/// Computes the unweighted arithmetic mean.
///
/// Uses [`statistic`] with the [one-shot conventions](crate#one-shot-forms).
/// Costs O(n) time with no allocation. Accuracy follows [`statistic`].
///
/// # Errors
///
/// Rejects empty or non-finite samples.
/// Propagates numerical overflow and allocation errors from [`statistic`].
///
/// ```
/// let value = scioxide_stats::mean(&[1.0_f64, 2.0, 3.0, 4.0])?;
/// assert!((value - 2.5).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// D. E. Knuth, The Art of Computer Programming, vol. 2, §4.2.2.
pub fn mean<T: NumericScalar>(values: &[T]) -> Result<T, SciOxideError> {
    statistic(values, Unweighted, Statistic::mean())?
        .location()
        .ok_or_else(not_scalar)
}

/// Computes the unweighted sample variance with divisor `n − 1`.
///
/// Uses [`statistic`] with the [one-shot conventions](crate#one-shot-forms).
/// Costs O(n) time with no allocation. Accuracy follows [`statistic`].
///
/// # Errors
///
/// Rejects empty or non-finite samples, or fewer than two observations.
/// Propagates numerical overflow and allocation errors from [`statistic`].
///
/// ```
/// let value = scioxide_stats::sample_variance(&[1.0_f64, 2.0, 3.0, 4.0])?;
/// assert!((value - 5.0 / 3.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// D. E. Knuth, The Art of Computer Programming, vol. 2, §4.2.2.
pub fn sample_variance<T: NumericScalar>(values: &[T]) -> Result<T::Real, SciOxideError> {
    statistic(values, Unweighted, Statistic::variance(1))?
        .dispersion()
        .ok_or_else(not_scalar)
}

/// Computes the unweighted sample standard deviation with divisor `n − 1`.
///
/// Uses [`statistic`] with the [one-shot conventions](crate#one-shot-forms).
/// Costs O(n) time with no allocation. Accuracy follows [`statistic`].
///
/// # Errors
///
/// Rejects empty or non-finite samples, or fewer than two observations.
/// Propagates numerical overflow and allocation errors from [`statistic`].
///
/// ```
/// let value = scioxide_stats::sample_standard_deviation(&[1.0_f64, 2.0, 3.0, 4.0])?;
/// assert!((value - (5.0_f64 / 3.0).sqrt()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// D. E. Knuth, The Art of Computer Programming, vol. 2, §4.2.2.
pub fn sample_standard_deviation<T: NumericScalar>(values: &[T]) -> Result<T::Real, SciOxideError> {
    statistic(values, Unweighted, Statistic::standard_deviation(1))?
        .dispersion()
        .ok_or_else(not_scalar)
}

/// Computes the unweighted median using Hyndman–Fan type 7 interpolation.
///
/// Uses [`statistic`] with the [one-shot conventions](crate#one-shot-forms).
/// Sorting costs O(n log n) time and O(n) scratch. Accuracy follows [`statistic`].
///
/// # Errors
///
/// Rejects empty or non-finite samples.
/// Propagates numerical overflow and allocation errors from [`statistic`].
///
/// ```
/// let value = scioxide_stats::median(&[1.0_f64, 2.0, 3.0, 4.0])?;
/// assert!((value - 2.5).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. J. Hyndman and Y. Fan, “Sample quantiles in statistical packages”, American Statistician 50 (1996), 361–365.
pub fn median<S: RealScalar>(values: &[S]) -> Result<S, SciOxideError> {
    statistic(
        values,
        Unweighted,
        Statistic::quantile(
            QuantileProbability::new(0.5)?,
            QuantileRule::new(QuantileMethod::Linear),
        ),
    )?
    .scalar()
}

/// Computes the unweighted quantile using Hyndman–Fan type 7 interpolation.
///
/// Uses [`statistic`] with the [one-shot conventions](crate#one-shot-forms).
/// Sorting costs O(n log n) time and O(n) scratch. Accuracy follows [`statistic`].
///
/// # Errors
///
/// Rejects empty or non-finite samples, and probabilities outside `[0, 1]`.
/// Propagates numerical overflow and allocation errors from [`statistic`].
///
/// ```
/// let value = scioxide_stats::quantile(&[1.0_f64, 2.0, 3.0, 4.0], 0.5)?;
/// assert!((value - 2.5).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. J. Hyndman and Y. Fan, “Sample quantiles in statistical packages”, American Statistician 50 (1996), 361–365.
pub fn quantile<S: RealScalar>(values: &[S], probability: f64) -> Result<S, SciOxideError> {
    statistic(
        values,
        Unweighted,
        Statistic::quantile(
            QuantileProbability::new(probability)?,
            QuantileRule::new(QuantileMethod::Linear),
        ),
    )?
    .scalar()
}

/// Computes the unweighted minimum.
///
/// Uses [`statistic`] with the [one-shot conventions](crate#one-shot-forms).
/// Costs O(n) time with no allocation. Accuracy follows [`statistic`].
///
/// # Errors
///
/// Rejects empty or non-finite samples.
/// Propagates numerical overflow and allocation errors from [`statistic`].
///
/// ```
/// let value = scioxide_stats::minimum(&[1.0_f64, 2.0, 3.0, 4.0])?;
/// assert!((value - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// D. E. Knuth, The Art of Computer Programming, vol. 2, §4.2.2.
pub fn minimum<S: RealScalar>(values: &[S]) -> Result<S, SciOxideError> {
    statistic(values, Unweighted, Statistic::minimum())?.scalar()
}

/// Computes the unweighted maximum.
///
/// Uses [`statistic`] with the [one-shot conventions](crate#one-shot-forms).
/// Costs O(n) time with no allocation. Accuracy follows [`statistic`].
///
/// # Errors
///
/// Rejects empty or non-finite samples.
/// Propagates numerical overflow and allocation errors from [`statistic`].
///
/// ```
/// let value = scioxide_stats::maximum(&[1.0_f64, 2.0, 3.0, 4.0])?;
/// assert!((value - 4.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// D. E. Knuth, The Art of Computer Programming, vol. 2, §4.2.2.
pub fn maximum<S: RealScalar>(values: &[S]) -> Result<S, SciOxideError> {
    statistic(values, Unweighted, Statistic::maximum())?.scalar()
}

/// Computes the two-sided 95% Student t confidence interval of a sample mean.
///
/// Uses `n − 1` degrees of freedom and assumes independent normal observations.
/// Costs O(n) time without allocation; critical-value accuracy follows
/// [`crate::ConfidenceProbability::student_t_critical_value`].
///
/// # Errors
///
/// Rejects fewer than two or non-finite observations and propagates numerical errors.
///
/// ```
/// let interval = scioxide_stats::mean_confidence_interval(&[1.0_f64, 2.0, 3.0, 4.0])?;
/// assert!((interval.estimate() - 2.5).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// Student, “The probable error of a mean”, Biometrika 6 (1908), 1–25.
pub fn mean_confidence_interval<S: RealScalar>(
    values: &[S],
) -> Result<crate::MeanConfidenceInterval<S>, SciOxideError> {
    crate::OnlineMoments::from_slice(values, crate::MomentOrder::Second)?
        .snapshot()
        .confidence_interval_of_mean()?
        .ok_or_else(|| {
            crate::error_helpers::invalid(
                scioxide_core::argument!("values"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "insufficient observations",
            )
        })
}
