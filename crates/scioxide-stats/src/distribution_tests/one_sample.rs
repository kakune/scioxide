//! One-sample Kolmogorov–Smirnov, Cramér–von Mises and Anderson–Darling statistics.
use super::*;
use alloc::vec::Vec;

use scioxide_core::{Budget, backend::try_with_capacity};
use scioxide_random::{
    Kolmogorov, KolmogorovSmirnovOneSided, KolmogorovSmirnovTwoSided, LawOptions,
};

pub(crate) fn sorted<S: RealScalar>(x: &[S], max_bytes: usize) -> Result<Vec<S>, SciOxideError> {
    sample(x, 1, scioxide_core::argument!("x"))?;
    crate::error_helpers::limit(
        scioxide_core::backend::checked_mul(
            x.len(),
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("ordering_bytes"),
        )?,
        max_bytes,
        scioxide_core::quantity!("ordering_bytes"),
    )?;
    let mut out = try_with_capacity(
        x.len(),
        scioxide_core::quantity!("distribution_test_ordering"),
    )?;
    out.extend_from_slice(x);
    crate::ranking::sort_finite(&mut out);
    Ok(out)
}

fn cdf_at<S: RealScalar>(
    cdf: impl Fn(S) -> Result<S, SciOxideError>,
    value: S,
) -> Result<f64, SciOxideError> {
    let p = crate::error_helpers::callback_finite(cdf(value)?.try_to_f64_exact()?)?;
    if !(0.0..=1.0).contains(&p) {
        return Err(invalid(
            scioxide_core::argument!("cdf"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "CDF output outside [0,1]",
        ));
    }
    Ok(p)
}

fn ks_statistic_sorted<S: RealScalar>(
    values: &[S],
    cdf: impl Fn(S) -> Result<S, SciOxideError>,
) -> Result<(f64, f64), SciOxideError> {
    let n = values.len() as f64;
    let mut plus: f64 = 0.0;
    let mut minus: f64 = 0.0;
    let mut previous = 0.0;
    for (index, &value) in values.iter().enumerate() {
        let p = cdf_at(&cdf, value)?;
        if p < previous {
            return Err(invalid(
                scioxide_core::argument!("cdf"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "CDF is not monotone on observations",
            ));
        }
        previous = p;
        plus = plus.max((index + 1) as f64 / n - p);
        minus = minus.max(p - index as f64 / n);
    }
    Ok((plus, minus))
}
/// Computes the one-sample Kolmogorov–Smirnov statistics `(D⁺, D⁻)` against the CDF `F`.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`ks_one_sample_statistic_with_options`].
///
/// # Errors
///
/// Returns the errors of [`ks_one_sample_statistic_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let cdf = |v: f64| Ok(v / 10.0);
/// let actual = ks_one_sample_statistic::<f64>(&x, cdf)?;
/// let configured = ks_one_sample_statistic_with_options::<f64>(&x, cdf, ClassicalOptions::default())?;
/// assert!((actual.0 - configured.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ks_one_sample_statistic<S: RealScalar>(
    x: &[S],
    cdf: impl Fn(S) -> Result<S, SciOxideError>,
) -> Result<(S, S), SciOxideError> {
    ks_one_sample_statistic_with_options(x, cdf, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Computes the one-sample Kolmogorov–Smirnov statistics `(D⁺, D⁻)` against
/// the CDF `F`.
///
/// ```text
/// D⁺ = maxᵢ (i/n − F(x₍ᵢ₎)),   D⁻ = maxᵢ (F(x₍ᵢ₎) − (i − 1)/n),
/// ```
///
/// where `x₍₁₎ ≤ … ≤ x₍ₙ₎` is the sorted sample. The two-sided statistic is
/// `D = max(D⁺, D⁻)`. Sorting costs O(n log n) time and a copy of `x`. Of the
/// options, only [`ClassicalOptions::max_workspace_bytes`] is used, as the byte
/// limit of that copy.
///
/// # Errors
///
/// - `InvalidInput` if `x` is empty, or `cdf` returns a value outside `[0, 1]`
///   or decreases along the sorted sample.
/// - `NonFinite` if `x` or a value of `cdf` is NaN or infinite.
/// - `ResourceLimit` if the sorted copy exceeds `max_workspace_bytes`.
/// - Any error returned by `cdf`.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{ClassicalOptions, ks_one_sample_statistic_with_options};
///
/// // Uniform CDF: D⁺ = 2/3 − 0.4 and D⁻ = 0.8 − 2/3.
/// let cdf = |x: f64| Ok(x);
/// let options = ClassicalOptions::default();
/// let (plus, minus) = ks_one_sample_statistic_with_options(&[0.1, 0.4, 0.8], &cdf, options)?;
/// assert!((plus - 4.0 / 15.0).abs() < 1e-15 && (minus - 2.0 / 15.0).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ks_one_sample_statistic_with_options<S: RealScalar>(
    x: &[S],
    cdf: impl Fn(S) -> Result<S, SciOxideError>,
    options: ClassicalOptions,
) -> Result<(S, S), SciOxideError> {
    let values = sorted(x, options.max_workspace_bytes())?;
    let (plus, minus) = ks_statistic_sorted(&values, cdf)?;
    Ok((S::try_from_f64_lossy(plus)?, S::try_from_f64_lossy(minus)?))
}
/// Performs the one-sample Kolmogorov–Smirnov test against a fully specified continuous CDF.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`ks_one_sample_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`ks_one_sample_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let cdf = |v: f64| Ok(v / 10.0);
/// let actual = ks_one_sample_test::<f64>(&x, cdf)?;
/// let configured = ks_one_sample_test_with_options::<f64>(&x, cdf, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ks_one_sample_test<S: RealScalar>(
    x: &[S],
    cdf: impl Fn(S) -> Result<S, SciOxideError>,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    ks_one_sample_test_with_options(x, cdf, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the one-sample Kolmogorov–Smirnov test against a fully specified
/// continuous CDF.
///
/// The statistic is `D⁺` for [`TestAlternative::Greater`], `D⁻` for `Less` and
/// `D = max(D⁺, D⁻)` for a two-sided test, as defined in
/// [`ks_one_sample_statistic_with_options`]. The exact p-value uses the finite-sample
/// distribution of `D` or `D⁺`, evaluated within [`ClassicalOptions::max_work`] work units.
/// Under [`TestCalibration::Auto`] it is tried when `n²` is at most `max_work`, and the test
/// falls back to the asymptotic law if the exact evaluation exhausts `max_work` or
/// `max_workspace_bytes`; [`DistributionTestResult::method`] reports which was used. The asymptotic
/// two-sided test uses the Kolmogorov distribution of `√n D` and the one-sided tests use
/// `exp(−2 n D²)`.
///
/// The null distribution assumes a continuous `F` with no parameters estimated
/// from `x`. For estimated parameters or discrete distributions, use
/// [`goodness_of_fit_test_into`].
///
/// # Errors
///
/// - `InvalidInput` if `x` has ties.
/// - The errors of [`ks_one_sample_statistic_with_options`].
/// - An error if the exact distribution exceeds `max_work` or
///   `max_workspace_bytes` under [`TestCalibration::Exact`].
pub fn ks_one_sample_test_with_options<S: RealScalar>(
    x: &[S],
    cdf: impl Fn(S) -> Result<S, SciOxideError>,
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    let values = sorted(x, options.max_workspace_bytes())?;
    if values.windows(2).any(|w| w[0] == w[1]) {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "ties require calibrated discrete-null Monte Carlo inference",
        ));
    }
    let (plus, minus) = ks_statistic_sorted(&values, cdf)?;
    let statistic = match options.alternative() {
        TestAlternative::Greater => plus,
        TestAlternative::Less => minus,
        TestAlternative::TwoSided => plus.max(minus),
    };
    let try_exact = options.calibration() == TestCalibration::Exact
        || (options.calibration() == TestCalibration::Auto
            && x.len()
                .checked_mul(x.len())
                .is_some_and(|w| w <= options.max_work()));
    let exact_p = || -> Result<f64, SciOxideError> {
        let sort_bytes = x
            .len()
            .checked_mul(core::mem::size_of::<S>())
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?;
        let law_options = LawOptions::default()
            .with_work_budget(Budget::try_from(options.max_work())?)
            .with_max_bytes(options.max_workspace_bytes() - sort_bytes);
        let d = S::try_from_f64_lossy(statistic)?;
        match options.alternative() {
            TestAlternative::TwoSided => {
                KolmogorovSmirnovTwoSided::with_options(x.len(), law_options)?.sf(d)?
            }
            _ => KolmogorovSmirnovOneSided::with_options(x.len(), law_options)?.sf(d)?,
        }
        .try_to_f64_exact()
    };
    // Under `Auto`, an exact evaluation that runs out of work or storage falls back to the
    // asymptotic law.
    let exact = if try_exact {
        match exact_p() {
            Ok(p) => Some(p),
            Err(error)
                if options.calibration() == TestCalibration::Auto
                    && matches!(
                        error.kind(),
                        scioxide_core::SciOxideErrorKind::ConvergenceFailure
                            | scioxide_core::SciOxideErrorKind::ResourceLimit
                    ) =>
            {
                None
            }
            Err(error) => return Err(error),
        }
    } else {
        None
    };
    let p = match exact {
        Some(p) => p,
        None => {
            let n = x.len() as f64;
            match options.alternative() {
                TestAlternative::TwoSided => Kolmogorov::<f64>::with_options(
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )?
                .sf(n.sqrt() * statistic)?,
                _ => (-2.0 * n * statistic * statistic).exp(),
            }
        }
    };
    result(
        statistic,
        p,
        if exact.is_some() {
            DistributionMethod::Exact
        } else {
            DistributionMethod::Asymptotic
        },
        NullParameters::Known,
        x.len(),
        1,
    )
}
/// Computes the one-sample Cramér–von Mises statistic `W²` against the CDF `F`.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`cramer_von_mises_statistic_with_options`].
///
/// # Errors
///
/// Returns the errors of [`cramer_von_mises_statistic_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let cdf = |v: f64| Ok(v / 10.0);
/// let actual = cramer_von_mises_statistic::<f64>(&x, cdf)?;
/// let configured = cramer_von_mises_statistic_with_options::<f64>(&x, cdf, ClassicalOptions::default())?;
/// assert!((actual - configured).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn cramer_von_mises_statistic<S: RealScalar>(
    x: &[S],
    cdf: impl Fn(S) -> Result<S, SciOxideError>,
) -> Result<S, SciOxideError> {
    cramer_von_mises_statistic_with_options(x, cdf, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Computes the one-sample Cramér–von Mises statistic `W²` against the CDF `F`.
///
/// ```text
/// W² = 1/(12n) + Σᵢ (F(x₍ᵢ₎) − (2i − 1)/(2n))²,
/// ```
///
/// where `x₍₁₎ ≤ … ≤ x₍ₙ₎` is the sorted sample. For a p-value use
/// [`cramer_von_mises_test_with_options`] or, with estimated parameters,
/// [`goodness_of_fit_test_into`]. Sorting costs O(n log n) time and a copy of `x`,
/// bounded by [`ClassicalOptions::max_workspace_bytes`], the only option used.
///
/// # Errors
///
/// - `InvalidInput` if `x` is empty, or `cdf` returns a value outside `[0, 1]`
///   or decreases along the sorted sample.
/// - `NonFinite` if `x` or a value of `cdf` is NaN or infinite.
/// - `ResourceLimit` if the sorted copy exceeds `max_workspace_bytes`.
/// - Any error returned by `cdf`.
pub fn cramer_von_mises_statistic_with_options<S: RealScalar>(
    x: &[S],
    cdf: impl Fn(S) -> Result<S, SciOxideError>,
    options: ClassicalOptions,
) -> Result<S, SciOxideError> {
    let values = sorted(x, options.max_workspace_bytes())?;
    S::try_from_f64_lossy(cvm_statistic_sorted(&values, cdf)?)
}

fn cvm_statistic_sorted<S: RealScalar>(
    values: &[S],
    cdf: impl Fn(S) -> Result<S, SciOxideError>,
) -> Result<f64, SciOxideError> {
    let n = values.len() as f64;
    let mut previous = 0.0;
    let mut sum = 1.0 / (12.0 * n);
    for (index, &value) in values.iter().enumerate() {
        let p = cdf_at(&cdf, value)?;
        if p < previous {
            return Err(invalid(
                scioxide_core::argument!("cdf"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "CDF is not monotone on observations",
            ));
        }
        previous = p;
        sum += (p - (index as f64 + 0.5) / n).powi(2);
    }
    finite(sum)
}

// Csörgő–Faraway (1996), limiting W² CDF. The half-gamma ratio is advanced
// recursively to avoid factorial overflow. Each term uses SciOxide's Bessel K.
pub(super) fn cvm_cdf_limit(x: f64, max_work: usize) -> Result<f64, SciOxideError> {
    if x <= 0.0 {
        return Ok(0.0);
    }
    if max_work == 0 {
        return Err(invalid(
            scioxide_core::argument!("options.max_work"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "Cramér–von Mises series needs positive work",
        ));
    }
    let mut ratio = 1.0;
    let mut sum = 0.0;
    for k in 0..max_work {
        let odd = (4 * k + 1) as f64;
        let q = odd * odd / (16.0 * x);
        let term = ratio * odd.sqrt() * (-q).exp() * scioxide_special::bessel_k(0.25_f64, q)?
            / (core::f64::consts::PI * x.sqrt());
        sum += term;
        if k > 0 && term < 1e-12 {
            return Ok(sum.clamp(0.0, 1.0));
        }
        ratio *= (k as f64 + 0.5) / (k as f64 + 1.0);
    }
    Err(crate::error_helpers::work_exhausted(
        max_work,
        scioxide_core::BudgetKind::Iteration,
        "Cramér–von Mises series exhausted options.max_work terms",
    ))
}
/// Performs the one-sample Cramér–von Mises test against a fully specified continuous CDF.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`cramer_von_mises_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`cramer_von_mises_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let cdf = |v: f64| Ok(v / 10.0);
/// let actual = cramer_von_mises_test::<f64>(&x, cdf)?;
/// let configured = cramer_von_mises_test_with_options::<f64>(&x, cdf, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// S. Csörgő and J. J. Faraway, "The exact and asymptotic distributions of
/// Cramér–von Mises statistics", *J. Roy. Statist. Soc. Ser. B* 58 (1996),
/// 221–234.
pub fn cramer_von_mises_test<S: RealScalar>(
    x: &[S],
    cdf: impl Fn(S) -> Result<S, SciOxideError>,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    cramer_von_mises_test_with_options(x, cdf, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the one-sample Cramér–von Mises test against a fully specified
/// continuous CDF.
///
/// The statistic is `W²` of [`cramer_von_mises_statistic_with_options`]. The p-value comes
/// from the limiting distribution of `W²`, summed as the series of Csörgő and
/// Faraway until a term falls below `10⁻¹²`. At most
/// [`ClassicalOptions::max_work`] terms are summed. For estimated parameters
/// or discrete distributions, use [`goodness_of_fit_test_into`].
///
/// # Errors
///
/// - `InvalidInput` if `x` has ties.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`] or the
///   alternative is not two-sided.
/// - The errors of [`cramer_von_mises_statistic_with_options`].
/// - `ConvergenceFailure` if the series needs more than `max_work` terms.
///
/// # References
///
/// S. Csörgő and J. J. Faraway, "The exact and asymptotic distributions of
/// Cramér–von Mises statistics", *J. Roy. Statist. Soc. Ser. B* 58 (1996),
/// 221–234.
pub fn cramer_von_mises_test_with_options<S: RealScalar>(
    x: &[S],
    cdf: impl Fn(S) -> Result<S, SciOxideError>,
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    require_asymptotic(options)?;
    require_omnibus(options)?;
    let values = sorted(x, options.max_workspace_bytes())?;
    if values.windows(2).any(|w| w[0] == w[1]) {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "ties require Monte Carlo calibration",
        ));
    }
    let statistic = cvm_statistic_sorted(&values, cdf)?;
    let p = 1.0 - cvm_cdf_limit(statistic, options.max_work())?;
    result(
        statistic,
        p.clamp(0.0, 1.0),
        DistributionMethod::Asymptotic,
        NullParameters::Known,
        x.len(),
        1,
    )
}
/// Computes the one-sample Anderson–Darling statistic `A²`.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`anderson_darling_statistic_with_options`].
///
/// # Errors
///
/// Returns the errors of [`anderson_darling_statistic_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// use scioxide_core::RealScalar;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let cdf = |v: f64| Ok(v / 10.0);
/// let actual = anderson_darling_statistic::<f64>( &x, |v: f64| cdf(v).map(RealScalar::ln), |v: f64| cdf(v).map(|p| (1.0 - p).ln()) )?;
/// let configured = anderson_darling_statistic_with_options::<f64>( &x, |v: f64| cdf(v).map(RealScalar::ln), |v: f64| cdf(v).map(|p| (1.0 - p).ln()), ClassicalOptions::default() )?;
/// assert!((actual - configured).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// T. W. Anderson and D. A. Darling, "A test of goodness of fit", *J. Amer.
/// Statist. Assoc.* 49 (1954), 765–769.
pub fn anderson_darling_statistic<S: RealScalar>(
    x: &[S],
    log_cdf: impl Fn(S) -> Result<S, SciOxideError>,
    log_survival: impl Fn(S) -> Result<S, SciOxideError>,
) -> Result<S, SciOxideError> {
    anderson_darling_statistic_with_options(x, log_cdf, log_survival, Default::default()).map_err(
        |error| {
            crate::error_helpers::one_shot_argument(
                error,
                &["options"],
                scioxide_core::argument!("x"),
            )
        },
    )
}
/// Computes the one-sample Anderson–Darling statistic `A²`.
///
/// ```text
/// A² = −n − (1/n) Σᵢ (2i − 1) [ln F(x₍ᵢ₎) + ln(1 − F(x₍ₙ₊₁₋ᵢ₎))],
/// ```
///
/// where `x₍₁₎ ≤ … ≤ x₍ₙ₎` is the sorted sample. `log_cdf` returns `ln F(x)` and
/// `log_survival` returns `ln(1 − F(x))`. Passing the logarithms keeps the
/// tails accurate. Both must be finite, so `0 < F(x) < 1` at every observation.
/// For a p-value use [`goodness_of_fit_test_into`]. Sorting costs O(n log n) time and
/// a copy of `x`, bounded by [`ClassicalOptions::max_workspace_bytes`], the only
/// option used.
///
/// # Errors
///
/// - `InvalidInput` if `x` is empty or a log tail is positive.
/// - `NonFinite` if `x` contains NaN or infinity.
/// - An error if a log tail is not finite.
/// - `ResourceLimit` if the sorted copy exceeds `max_workspace_bytes`.
/// - Any error returned by `log_cdf` or `log_survival`.
///
/// # References
///
/// T. W. Anderson and D. A. Darling, "A test of goodness of fit", *J. Amer.
/// Statist. Assoc.* 49 (1954), 765–769.
pub fn anderson_darling_statistic_with_options<S: RealScalar>(
    x: &[S],
    log_cdf: impl Fn(S) -> Result<S, SciOxideError>,
    log_survival: impl Fn(S) -> Result<S, SciOxideError>,
    options: ClassicalOptions,
) -> Result<S, SciOxideError> {
    let values = sorted(x, options.max_workspace_bytes())?;
    let n = values.len() as f64;
    let mut sum = 0.0;
    for (index, &value) in values.iter().enumerate() {
        let lo = finite(log_cdf(value)?.try_to_f64_exact()?)?;
        let hi = finite(log_survival(values[values.len() - 1 - index])?.try_to_f64_exact()?)?;
        if lo > 0.0 || hi > 0.0 {
            return Err(invalid(
                scioxide_core::argument!("log_cdf"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "log-CDF tails must be nonpositive",
            ));
        }
        sum += (2.0 * index as f64 + 1.0) * (lo + hi);
    }
    S::try_from_f64_lossy(finite(-n - sum / n)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_order_statistics() {
        let cdf = |x: f64| Ok(x);
        let options = ClassicalOptions::default().with_max_workspace_bytes(24);
        let (plus, minus) =
            ks_one_sample_statistic_with_options(&[0.1, 0.4, 0.8], cdf, options).unwrap();
        assert!((plus - 4.0 / 15.0).abs() < 1e-15);
        assert!((minus - 2.0 / 15.0).abs() < 1e-15);
        let cvm = cramer_von_mises_statistic_with_options(&[0.2, 0.5, 0.8], cdf, options).unwrap();
        assert!((cvm - 0.03_f64).abs() < 1e-15);
        let ad = anderson_darling_statistic_with_options(
            &[0.2, 0.5, 0.8],
            |x: f64| Ok(x.ln()),
            |x: f64| Ok((1.0 - x).ln()),
            options,
        )
        .unwrap();
        assert!(ad > 0.19 && ad < 0.21);
        // The options budget is the only byte limit: one byte short is a
        // resource limit carrying the required size, not the limit.
        let short = options.with_max_workspace_bytes(23);
        for error in [
            ks_one_sample_statistic_with_options(&[0.1, 0.4, 0.8], cdf, short).map(|_| ()),
            cramer_von_mises_statistic_with_options(&[0.2, 0.5, 0.8], cdf, short).map(|_| ()),
            anderson_darling_statistic_with_options(
                &[0.2, 0.5, 0.8],
                |x: f64| Ok(x.ln()),
                |x: f64| Ok((1.0 - x).ln()),
                short,
            )
            .map(|_| ()),
        ] {
            let error = error.unwrap_err();
            assert_eq!(
                error.kind(),
                scioxide_core::SciOxideErrorKind::ResourceLimit
            );
        }
    }

    #[test]
    fn exact_one_observation_and_rejections() {
        let p =
            ks_one_sample_test_with_options(&[0.25_f64], Ok, ClassicalOptions::default()).unwrap();
        assert!((p.statistic - 0.75).abs() < 1e-15);
        assert!((p.p_value - 0.5).abs() < 1e-13);
        assert_eq!(p.method, DistributionMethod::Exact);
        assert!(
            ks_one_sample_test_with_options(&[0.25_f64, 0.25], Ok, ClassicalOptions::default())
                .is_err()
        );
        assert!(
            ks_one_sample_test_with_options(&[0.25_f64], |_| Ok(1.5), ClassicalOptions::default())
                .is_err()
        );
        let cvm = cramer_von_mises_test_with_options(
            &[0.2_f64, 0.5, 0.8],
            Ok,
            ClassicalOptions::default(),
        )
        .unwrap();
        assert!((cvm.statistic - 0.03).abs() < 1e-14);
        assert!((0.0..=1.0).contains(&cvm.p_value));
    }

    #[test]
    fn finite_ks_calibration_and_limits() {
        let n = 512;
        let d = 0.03125;
        // This grid has the known maximum CDF-minus-ECDF difference d at x[0].
        let x: Vec<f64> = (0..n)
            .map(|i| d + (1.0 - d) * i as f64 / (n - 1) as f64)
            .collect();
        let cdf = |x: f64| Ok(x);
        let options = ClassicalOptions::default()
            .with_calibration(TestCalibration::Exact)
            .with_max_work(1 << 24);
        let result = ks_one_sample_test_with_options(&x, cdf, options).unwrap();
        assert_eq!(result.method, DistributionMethod::Exact);
        assert!((result.statistic - d).abs() < 1e-15);
        let law = KolmogorovSmirnovTwoSided::with_options(
            n,
            LawOptions::default().with_work_budget(Budget::try_from(options.max_work()).unwrap()),
        )
        .unwrap();
        assert!((result.p_value - law.sf(d).unwrap()).abs() < 1e-13);
        assert!(ks_one_sample_test_with_options(&x, cdf, options.with_max_work(1)).is_err());
        let automatic = ks_one_sample_test_with_options(
            &x,
            cdf,
            options
                .with_calibration(TestCalibration::Auto)
                .with_max_work(1),
        )
        .unwrap();
        assert_eq!(automatic.method, DistributionMethod::Asymptotic);
        assert!(
            ks_one_sample_test_with_options(&x, cdf, options.with_max_workspace_bytes(n * 8))
                .is_err()
        );
    }
}
