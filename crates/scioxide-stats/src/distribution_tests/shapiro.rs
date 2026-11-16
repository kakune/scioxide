//! Shapiro–Wilk test of normality.
use super::*;

use scioxide_core::{Scaled, backend::try_with_capacity};
use scioxide_random::Quantile;

fn polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |value, &coefficient| value * x + coefficient)
}

fn shapiro_w<S: RealScalar>(x: &[S], max_bytes: usize) -> Result<f64, SciOxideError> {
    sample(x, 3, scioxide_core::argument!("x"))?;
    let half = x.len() / 2;
    let bytes = x
        .len()
        .checked_mul(core::mem::size_of::<S>())
        .and_then(|b| {
            half.checked_mul(core::mem::size_of::<f64>())
                .and_then(|a| a.checked_add(b))
        })
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    crate::error_helpers::limit(bytes, max_bytes, scioxide_core::quantity!("shapiro_bytes"))?;
    let sorted = super::one_sample::sorted(x, max_bytes)?;
    if sorted[0] == sorted[sorted.len() - 1] {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Shapiro W undefined for constant data",
        ));
    }
    let n = x.len() as f64;
    let mut weights =
        try_with_capacity(half, scioxide_core::quantity!("shapiro_order_coefficients"))?;
    if x.len() == 3 {
        weights.push(core::f64::consts::FRAC_1_SQRT_2);
    } else {
        let normal = Normal::new(0.0, 1.0)?;
        let mut sumsq = 0.0;
        for i in 0..half {
            let q = normal.quantile((i as f64 + 0.625) / (n + 0.25))?;
            weights.push(q);
            sumsq += 2.0 * q * q;
        }
        let scale = sumsq.sqrt();
        let root = 1.0 / n.sqrt();
        let first = polynomial(
            &[0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056],
            root,
        ) - weights[0] / scale;
        let start = if x.len() > 5 {
            let second = polynomial(
                &[0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633],
                root,
            ) - weights[1] / scale;
            let remaining =
                ((sumsq - 2.0 * weights[0] * weights[0] - 2.0 * weights[1] * weights[1])
                    / (1.0 - 2.0 * first * first - 2.0 * second * second))
                    .sqrt();
            weights[1] = second;
            for weight in &mut weights[2..] {
                *weight /= -remaining;
            }
            2
        } else {
            let remaining =
                ((sumsq - 2.0 * weights[0] * weights[0]) / (1.0 - 2.0 * first * first)).sqrt();
            for weight in &mut weights[1..] {
                *weight /= -remaining;
            }
            1
        };
        weights[0] = first;
        if start == 2 && weights[1] <= 0.0 {
            return Err(invalid(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Shapiro coefficient breakdown",
            ));
        }
    }
    let origin = Scaled::from_value(sorted[0].try_to_f64_exact()?)?;
    let range = Scaled::from_value(sorted[sorted.len() - 1].try_to_f64_exact()?)?.sub(origin)?;
    // Subtract the origin before projection: the uncentered quotient loses
    // narrow spreads, while the unscaled endpoint difference can overflow.
    // Extended-range subtraction/division also preserves subnormal spreads.
    let normalize = |value: S| {
        Scaled::from_value(value.try_to_f64_exact()?)?
            .sub(origin)?
            .div(range)?
            .round()
    };
    let mut mean = 0.0;
    for &value in &sorted {
        mean += normalize(value)?;
    }
    mean /= n;
    let mut denominator = 0.0;
    for &value in &sorted {
        denominator += (normalize(value)? - mean).powi(2);
    }
    if denominator <= 0.0 {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Shapiro sample variance is zero",
        ));
    }
    let mut numerator = 0.0;
    for (i, &weight) in weights.iter().enumerate() {
        numerator += weight * (normalize(sorted[sorted.len() - 1 - i])? - normalize(sorted[i])?);
    }
    Ok(finite(numerator * numerator / denominator)?.clamp(0.0, 1.0))
}
/// Computes the Shapiro–Wilk statistic `W` for any `n ≥ 3`.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`shapiro_wilk_statistic_with_options`].
///
/// # Errors
///
/// Returns the errors of [`shapiro_wilk_statistic_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let actual = shapiro_wilk_statistic::<f64>(&x)?;
/// let configured = shapiro_wilk_statistic_with_options::<f64>(&x, ClassicalOptions::default())?;
/// assert!((actual - configured).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// S. S. Shapiro and M. B. Wilk, "An analysis of variance test for normality
/// (complete samples)", *Biometrika* 52 (1965), 591–611.
///
/// P. Royston, "Remark AS R94: A remark on algorithm AS 181: The W-test for
/// normality", *Appl. Statist.* 44 (1995), 547–551.
pub fn shapiro_wilk_statistic<S: RealScalar>(x: &[S]) -> Result<S, SciOxideError> {
    shapiro_wilk_statistic_with_options(x, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Computes the Shapiro–Wilk statistic `W` for any `n ≥ 3`.
///
/// ```text
/// W = (Σᵢ aᵢ x₍ᵢ₎)² / Σᵢ (xᵢ − x̄)²,
/// ```
///
/// where `x₍₁₎ ≤ … ≤ x₍ₙ₎` is the sorted sample and `aᵢ` are Royston's
/// approximations to the Shapiro–Wilk coefficients. `W` lies in `(0, 1]`, and
/// small values indicate departure from normality. The sample is shifted by its
/// minimum and divided by its range in extended range, so samples near the
/// overflow or underflow thresholds are accepted. `W` does not change under
/// translation or nonzero scaling, up to rounding.
///
/// The cost is O(n log n) time. The scratch of `n` values and `n/2`
/// coefficients is bounded by [`ClassicalOptions::max_workspace_bytes`], the
/// only option used. For `n > 5000` this function gives the statistic for a
/// Monte Carlo p-value; see [`shapiro_wilk_test_with_options`].
///
/// # Errors
///
/// - `InvalidInput` if `x` has fewer than three observations or is constant.
/// - `NonFinite` if `x` contains NaN or infinity.
/// - `ResourceLimit` if the scratch exceeds `max_workspace_bytes`.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{ClassicalOptions, shapiro_wilk_statistic_with_options};
///
/// // Three equally spaced points fit a normal sample exactly: W = 1.
/// let options = ClassicalOptions::default().with_max_workspace_bytes(1024);
/// let w = shapiro_wilk_statistic_with_options(&[-1e308_f64, 0.0, 1e308], options)?;
/// assert!((w - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// S. S. Shapiro and M. B. Wilk, "An analysis of variance test for normality
/// (complete samples)", *Biometrika* 52 (1965), 591–611.
///
/// P. Royston, "Remark AS R94: A remark on algorithm AS 181: The W-test for
/// normality", *Appl. Statist.* 44 (1995), 547–551.
pub fn shapiro_wilk_statistic_with_options<S: RealScalar>(
    x: &[S],
    options: ClassicalOptions,
) -> Result<S, SciOxideError> {
    S::try_from_f64_lossy(shapiro_w(x, options.max_workspace_bytes())?)
}
/// Performs the Shapiro–Wilk test of normality for `3 ≤ n ≤ 5000`.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`shapiro_wilk_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`shapiro_wilk_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let actual = shapiro_wilk_test::<f64>(&x)?;
/// let configured = shapiro_wilk_test_with_options::<f64>(&x, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. Royston, "Approximating the Shapiro–Wilk W-test for non-normality",
/// *Statist. Comput.* 2 (1992), 117–119.
pub fn shapiro_wilk_test<S: RealScalar>(
    x: &[S],
) -> Result<DistributionTestResult<S>, SciOxideError> {
    shapiro_wilk_test_with_options(x, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the Shapiro–Wilk test of normality for `3 ≤ n ≤ 5000`.
///
/// The statistic is `W` of [`shapiro_wilk_statistic_with_options`]. The p-value uses
/// Royston's normalizing transformation of `W`, valid for `3 ≤ n ≤ 5000`. For
/// larger samples, compute `W` with [`shapiro_wilk_statistic_with_options`] and simulate its
/// null distribution from normal samples. Samples with ties are accepted. The
/// cost is O(n log n) time and O(n) memory.
///
/// # Errors
///
/// - `InvalidInput` if `n < 3`, `n > 5000` or `x` is constant.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`] or the
///   alternative is not two-sided.
/// - `NonFinite` if `x` contains NaN or infinity.
/// - `ResourceLimit` if the scratch exceeds
///   [`ClassicalOptions::max_workspace_bytes`].
///
/// # Examples
///
/// ```
/// use scioxide_stats::{ClassicalOptions, shapiro_wilk_test_with_options};
///
/// // W is translation invariant, even for a shift of 2⁵⁰.
/// let x = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 10.0];
/// let shifted = x.map(|value| value + 2.0_f64.powi(50));
/// let original = shapiro_wilk_test_with_options(&x, ClassicalOptions::default())?;
/// let translated = shapiro_wilk_test_with_options(&shifted, ClassicalOptions::default())?;
/// assert!((original.statistic() - translated.statistic()).abs() < 1e-14);
/// assert!((original.p_value() - translated.p_value()).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// P. Royston, "Approximating the Shapiro–Wilk W-test for non-normality",
/// *Statist. Comput.* 2 (1992), 117–119.
pub fn shapiro_wilk_test_with_options<S: RealScalar>(
    x: &[S],
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    require_asymptotic(options)?;
    require_omnibus(options)?;
    if x.len() > 5000 {
        return Err(invalid(
            scioxide_core::argument!("options.calibration"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Shapiro p-value above n=5000 requires Monte Carlo calibration",
        ));
    }
    let w = shapiro_w(x, options.max_workspace_bytes())?;
    let n = x.len() as f64;
    let p = if x.len() == 3 {
        (1.0 - 6.0 / core::f64::consts::PI * w.sqrt().acos()).clamp(0.0, 1.0)
    } else if w >= 1.0 {
        1.0
    } else {
        let mut y = (1.0 - w).ln();
        let (mean, scale) = if x.len() <= 11 {
            let gamma = polynomial(&[-2.273, 0.459], n);
            if y >= gamma {
                return result(
                    w,
                    0.0,
                    DistributionMethod::Asymptotic,
                    NullParameters::Estimated,
                    x.len(),
                    1,
                );
            }
            y = -(gamma - y).ln();
            (
                polynomial(&[0.5440, -0.39978, 0.025054, -0.0006714], n),
                polynomial(&[1.3822, -0.77857, 0.062767, -0.0020322], n).exp(),
            )
        } else {
            let l = n.ln();
            (
                polynomial(&[-1.5861, -0.31082, -0.083751, 0.0038915], l),
                polynomial(&[-0.4803, -0.082676, 0.0030302], l).exp(),
            )
        };
        Normal::new(0.0, 1.0)?.sf((y - mean) / scale)?
    };
    result(
        w,
        p,
        DistributionMethod::Asymptotic,
        NullParameters::Estimated,
        x.len(),
        1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn affine_invariance<S: RealScalar>(shift: f64, large_scale: f64, tiny_scale: f64) {
        let base = [0., 1., 2., 3., 4., 10.];
        let convert = |value| S::try_from_f64_lossy(value).unwrap();
        let options = ClassicalOptions::default();
        let original = shapiro_wilk_test_with_options(&base.map(convert), options).unwrap();
        for (origin, scale) in [
            (shift, 1.),
            (-shift, 1.),
            (0., large_scale),
            (0., tiny_scale),
            (0., -1.),
        ] {
            let transformed = base.map(|value| convert(origin + scale * value));
            let actual = shapiro_wilk_test_with_options(&transformed, options).unwrap();
            assert!(
                (actual.statistic.try_to_f64_exact().unwrap()
                    - original.statistic.try_to_f64_exact().unwrap())
                .abs()
                    <= 16. * f64::EPSILON
            );
            assert!((actual.p_value - original.p_value).abs() <= 128. * f64::EPSILON);
        }
    }

    #[test]
    fn translation_scaling_and_reflection_preserve_normality() {
        affine_invariance::<f64>(2_f64.powi(50), 2_f64.powi(1000), f64::from_bits(1));
        affine_invariance::<f32>(
            2_f64.powi(20),
            2_f64.powi(120),
            f64::from(f32::from_bits(1)),
        );
    }

    #[test]
    fn opposite_large_endpoints_are_not_degenerate() {
        for values in [[-1e308_f64, 0., 1e308], [-f64::MAX, 0., f64::MAX]] {
            let actual =
                shapiro_wilk_test_with_options(&values, ClassicalOptions::default()).unwrap();
            assert!((actual.statistic - 1.).abs() <= f64::EPSILON);
            assert!((actual.p_value - 1.).abs() <= f64::EPSILON);
        }
    }

    #[test]
    fn three_point_exact_p_and_boundaries() {
        let r = shapiro_wilk_test_with_options(&[1.0_f64, 2.0, 3.0], ClassicalOptions::default())
            .unwrap();
        assert!((r.statistic - 1.0).abs() < 1e-14);
        assert!((r.p_value - 1.0).abs() < 1e-12);
        let r = shapiro_wilk_test_with_options(&[1.0_f64, 1.0, 2.0], ClassicalOptions::default())
            .unwrap();
        assert!((r.statistic - 0.75).abs() < 1e-12);
        assert!(r.p_value < 1e-6);
        assert!(
            shapiro_wilk_test_with_options(&[1.0_f64; 3], ClassicalOptions::default()).is_err()
        );
        let men = [
            148.0_f64, 154.0, 158.0, 160.0, 161.0, 162.0, 166.0, 170.0, 182.0, 195.0, 236.0,
        ];
        let published = shapiro_wilk_test_with_options(&men, ClassicalOptions::default()).unwrap();
        assert!((published.statistic - 0.7888).abs() < 0.0001);
        assert!((published.p_value - 0.006704).abs() < 0.00005);
    }
}
