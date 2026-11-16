//! Normality tests based on sample skewness and kurtosis.
use super::*;
use crate::{BiasCorrection, KurtosisConvention, MomentOrder, OnlineMoments};

fn standardized_moments<S: RealScalar>(
    x: &[S],
    minimum: usize,
) -> Result<(f64, f64), SciOxideError> {
    sample(x, minimum, scioxide_core::argument!("x"))?;
    let snapshot = OnlineMoments::from_slice(x, MomentOrder::Fourth)?.snapshot();
    let skew = snapshot
        .skewness(BiasCorrection::None)?
        .ok_or_else(|| {
            invalid(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "normality statistic undefined for constant data",
            )
        })?
        .try_to_f64_exact()?;
    let kurtosis = snapshot
        .kurtosis(BiasCorrection::None, KurtosisConvention::Pearson)?
        .ok_or_else(|| {
            invalid(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "normality statistic undefined for constant data",
            )
        })?
        .try_to_f64_exact()?;
    Ok((finite(skew)?, finite(kurtosis)?))
}

// D'Agostino et al. (1990), transformed sample skewness.
fn skew_z(skew: f64, n: f64) -> Result<f64, SciOxideError> {
    let y = skew * ((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0))).sqrt();
    let beta2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
        / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
    let w2 = -1.0 + (2.0 * (beta2 - 1.0)).sqrt();
    let delta = 1.0 / (0.5 * w2.ln()).sqrt();
    let alpha = (2.0 / (w2 - 1.0)).sqrt();
    finite(delta * (y / alpha).asinh())
}

// Anscombe–Glynn (1983), transformed sample kurtosis.
fn kurtosis_z(kurtosis: f64, n: f64) -> Result<f64, SciOxideError> {
    let expected = 3.0 * (n - 1.0) / (n + 1.0);
    let variance = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0).powi(2) * (n + 3.0) * (n + 5.0));
    let x = (kurtosis - expected) / variance.sqrt();
    let beta = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0))
        * (6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0))).sqrt();
    let a = 6.0 + 8.0 / beta * (2.0 / beta + (1.0 + 4.0 / (beta * beta)).sqrt());
    let denominator = 1.0 + x * (2.0 / (a - 4.0)).sqrt();
    if denominator == 0.0 {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "kurtosis transform is singular",
        ));
    }
    let term = denominator.signum() * ((1.0 - 2.0 / a) / denominator.abs()).cbrt();
    finite((1.0 - 2.0 / (9.0 * a) - term) / (2.0 / (9.0 * a)).sqrt())
}
/// Performs D'Agostino's test of zero skewness.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`skew_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`skew_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let actual = skew_test::<f64>(&x)?;
/// let configured = skew_test_with_options::<f64>(&x, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. B. D'Agostino, A. Belanger and R. B. D'Agostino Jr., "A suggestion for
/// using powerful and informative tests of normality", *Amer. Statist.* 44
/// (1990), 316–321.
pub fn skew_test<S: RealScalar>(x: &[S]) -> Result<DistributionTestResult<S>, SciOxideError> {
    skew_test_with_options(x, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs D'Agostino's test of zero skewness.
///
/// The sample skewness `√b₁ = m₃ / m₂^(3/2)`, with `mₖ` the `k`-th central
/// sample moment, is transformed to an approximately standard normal `Z`. The
/// statistic is `Z` and the p-value its normal tail. The test needs `n ≥ 8`.
/// The cost is O(n) time and O(1) memory.
///
/// # Errors
///
/// - `InvalidInput` if `x` has fewer than eight observations or is constant.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`].
/// - `NonFinite` if `x` contains NaN or infinity.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{ClassicalOptions, skew_test_with_options};
///
/// // Reference values of D'Agostino's skewness test for this sample.
/// let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8000.0];
/// let test = skew_test_with_options(&x, ClassicalOptions::default())?;
/// assert!((test.statistic() - 3.571773510360407).abs() < 1e-10);
/// assert!((test.p_value() - 0.0003545719905823133).abs() < 1e-10);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. B. D'Agostino, A. Belanger and R. B. D'Agostino Jr., "A suggestion for
/// using powerful and informative tests of normality", *Amer. Statist.* 44
/// (1990), 316–321.
pub fn skew_test_with_options<S: RealScalar>(
    x: &[S],
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    require_asymptotic(options)?;
    let (skew, _) = standardized_moments(x, 8)?;
    let z = skew_z(skew, x.len() as f64)?;
    result(
        z,
        normal_tail(z, options.alternative())?,
        DistributionMethod::Asymptotic,
        NullParameters::Estimated,
        x.len(),
        1,
    )
}
/// Performs the Anscombe–Glynn test of normal kurtosis.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`kurtosis_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`kurtosis_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let actual = kurtosis_test::<f64>(&x)?;
/// let configured = kurtosis_test_with_options::<f64>(&x, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// F. J. Anscombe and W. J. Glynn, "Distribution of the kurtosis statistic b₂
/// for normal samples", *Biometrika* 70 (1983), 227–234.
pub fn kurtosis_test<S: RealScalar>(x: &[S]) -> Result<DistributionTestResult<S>, SciOxideError> {
    kurtosis_test_with_options(x, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the Anscombe–Glynn test of normal kurtosis.
///
/// The sample kurtosis `b₂ = m₄ / m₂²` is standardized with its exact normal
/// mean and variance and transformed to an approximately standard normal `Z`.
/// The statistic is `Z` and the p-value its normal tail. The test needs `n ≥ 5`,
/// and the approximation is recommended for `n ≥ 20`. The cost is O(n) time and
/// O(1) memory.
///
/// # Errors
///
/// - `InvalidInput` if `x` has fewer than five observations or is constant.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`], or the
///   transformation is singular.
/// - `NonFinite` if `x` contains NaN or infinity.
///
/// # References
///
/// F. J. Anscombe and W. J. Glynn, "Distribution of the kurtosis statistic b₂
/// for normal samples", *Biometrika* 70 (1983), 227–234.
pub fn kurtosis_test_with_options<S: RealScalar>(
    x: &[S],
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    require_asymptotic(options)?;
    let (_, kurtosis) = standardized_moments(x, 5)?;
    let z = kurtosis_z(kurtosis, x.len() as f64)?;
    result(
        z,
        normal_tail(z, options.alternative())?,
        DistributionMethod::Asymptotic,
        NullParameters::Estimated,
        x.len(),
        1,
    )
}
/// Performs the D'Agostino–Pearson omnibus test of normality.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`normality_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`normality_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let actual = normality_test::<f64>(&x)?;
/// let configured = normality_test_with_options::<f64>(&x, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// R. B. D'Agostino and E. S. Pearson, "Tests for departure from normality.
/// Empirical results for the distributions of b₂ and √b₁", *Biometrika* 60
/// (1973), 613–622.
pub fn normality_test<S: RealScalar>(x: &[S]) -> Result<DistributionTestResult<S>, SciOxideError> {
    normality_test_with_options(x, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the D'Agostino–Pearson omnibus test of normality.
///
/// The statistic is `K² = Z_s² + Z_k²`, where `Z_s` and `Z_k` are the
/// statistics of [`skew_test_with_options`] and [`kurtosis_test_with_options`]. The p-value is the upper
/// tail of the chi-square distribution with 2 degrees of freedom. The test needs
/// `n ≥ 8`. The cost is O(n) time and O(1) memory.
///
/// # Errors
///
/// - `InvalidInput` if `x` has fewer than eight observations or is constant.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`] or the
///   alternative is not two-sided.
/// - `NonFinite` if `x` contains NaN or infinity.
///
/// # References
///
/// R. B. D'Agostino and E. S. Pearson, "Tests for departure from normality.
/// Empirical results for the distributions of b₂ and √b₁", *Biometrika* 60
/// (1973), 613–622.
pub fn normality_test_with_options<S: RealScalar>(
    x: &[S],
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    require_asymptotic(options)?;
    require_omnibus(options)?;
    let (skew, kurtosis) = standardized_moments(x, 8)?;
    let n = x.len() as f64;
    let z_skew = skew_z(skew, n)?;
    let z_kurtosis = kurtosis_z(kurtosis, n)?;
    let statistic = finite(z_skew * z_skew + z_kurtosis * z_kurtosis)?;
    result(
        statistic,
        chi_tail(statistic, 2.0)?,
        DistributionMethod::Asymptotic,
        NullParameters::Estimated,
        x.len(),
        1,
    )
}
/// Performs the Jarque–Bera test of normality.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`jarque_bera_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`jarque_bera_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let actual = jarque_bera_test::<f64>(&x)?;
/// let configured = jarque_bera_test_with_options::<f64>(&x, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// C. M. Jarque and A. K. Bera, "A test for normality of observations and
/// regression residuals", *Int. Statist. Rev.* 55 (1987), 163–172.
pub fn jarque_bera_test<S: RealScalar>(
    x: &[S],
) -> Result<DistributionTestResult<S>, SciOxideError> {
    jarque_bera_test_with_options(x, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the Jarque–Bera test of normality.
///
/// ```text
/// JB = n/6 (b₁ + (b₂ − 3)²/4),
/// ```
///
/// where `b₁` is the squared sample skewness and `b₂` the sample kurtosis. The
/// p-value is the upper tail of the chi-square distribution with 2 degrees of
/// freedom, which is accurate only for large `n`. The cost is O(n) time and O(1)
/// memory.
///
/// # Errors
///
/// - `InvalidInput` if `x` has fewer than four observations or is constant.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`] or the
///   alternative is not two-sided.
/// - `NonFinite` if `x` contains NaN or infinity.
///
/// # References
///
/// C. M. Jarque and A. K. Bera, "A test for normality of observations and
/// regression residuals", *Int. Statist. Rev.* 55 (1987), 163–172.
pub fn jarque_bera_test_with_options<S: RealScalar>(
    x: &[S],
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    require_asymptotic(options)?;
    require_omnibus(options)?;
    let (skew, kurtosis) = standardized_moments(x, 4)?;
    let statistic = finite(x.len() as f64 / 6.0 * (skew * skew + (kurtosis - 3.0).powi(2) / 4.0))?;
    result(
        statistic,
        chi_tail(statistic, 2.0)?,
        DistributionMethod::Asymptotic,
        NullParameters::Estimated,
        x.len(),
        1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_moment_examples() {
        let x: [f64; 8] = [1., 2., 3., 4., 5., 6., 7., 8000.];
        let r = skew_test_with_options(&x, ClassicalOptions::default()).unwrap();
        assert!((r.statistic - 3.571773510360407).abs() < 2e-12);
        assert!((r.p_value - 0.0003545719905823133).abs() < 2e-12);
        let y: [f64; 20] = core::array::from_fn(|i| i as f64);
        let r = kurtosis_test_with_options(&y, ClassicalOptions::default()).unwrap();
        assert!((r.statistic + 1.7058104152122062).abs() < 2e-12);
        assert!((r.p_value - 0.08804338332528348).abs() < 2e-12);
    }

    #[test]
    fn precision_and_invalid_samples() {
        let x: [f32; 8] = [1., 2., 3., 4., 5., 6., 7., 8000.];
        assert!(
            (skew_test_with_options(&x, ClassicalOptions::default())
                .unwrap()
                .statistic
                - 3.5717735)
                .abs()
                < 2e-5
        );
        assert!(skew_test_with_options(&x[..7], ClassicalOptions::default()).is_err());
        assert!(normality_test_with_options(&[1.0_f64; 8], ClassicalOptions::default()).is_err());
        assert!(
            jarque_bera_test_with_options(
                &[1.0_f64, 2., f64::NAN, 4.],
                ClassicalOptions::default()
            )
            .is_err()
        );
    }
}
