//! Correlation tests: Pearson, point-biserial, Spearman, Kendall, Chatterjee and Somers.
use super::*;
use crate::{MomentOrder, OnlineCovariance};
use crate::{PermutationKind, RankMethod};
use scioxide_core::backend::try_with_capacity;
mod kendall;
/// Tests for zero Pearson correlation between paired samples.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`pearson_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`pearson_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = pearson_test::<f64>(&x, &y)?;
/// let configured = pearson_test_with_options::<f64>(&x, &y, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn pearson_test<S: RealScalar>(
    x: &[S],
    y: &[S],
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    pearson_test_with_options(x, y, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Tests for zero Pearson correlation between paired samples.
///
/// The statistic is the sample correlation `r`. The p-value uses
///
/// ```text
/// t = r √((n − 2)/(1 − r²)),   ν = n − 2,
/// ```
///
/// which has Student's t distribution under independent bivariate normal
/// sampling. For `n > 3` the result includes the approximate Fisher-z
/// confidence interval `tanh(atanh r ± z_(1−α/2)/√(n − 3))`. The cost is O(n)
/// time and O(1) memory.
///
/// # Errors
///
/// - `DimensionMismatch` if `x` and `y` have different lengths.
/// - `InvalidInput` if there are fewer than two pairs, or either sample is
///   constant.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
pub fn pearson_test_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    paired(x, y, 2)?;
    let r = OnlineCovariance::from_slices(x, y, MomentOrder::Second)?
        .snapshot()
        .correlation()?
        .ok_or_else(|| {
            invalid(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "constant sample has undefined correlation",
            )
        })?
        .into_f64();
    let p = if x.len() == 2 {
        match options.alternative {
            TestAlternative::TwoSided => 1.0,
            _ => 0.5,
        }
    } else if r.abs() == 1.0 {
        match options.alternative {
            TestAlternative::TwoSided => 0.0,
            TestAlternative::Less => {
                if r < 0.0 {
                    0.0
                } else {
                    1.0
                }
            }
            TestAlternative::Greater => {
                if r > 0.0 {
                    0.0
                } else {
                    1.0
                }
            }
        }
    } else {
        let df = (x.len() - 2) as f64;
        student_p(
            r * ((df / (1.0 - r)) / (1.0 + r)).sqrt(),
            df,
            options.alternative,
        )?
    };
    let mut out = result(
        r,
        p,
        ClassicalMethod::Exact,
        TestSamplePolicy::Paired,
        [x.len(), y.len()],
    )?;
    out.effect = Some((ClassicalEffect::Association, checked(r)?));
    if x.len() > 3 {
        let (lower, upper) = if r.abs() == 1.0 {
            (r, r)
        } else {
            let z = 0.5 * ((1.0 + r).ln() - (1.0 - r).ln());
            let radius =
                options.confidence.normal_critical_value::<f64>()? / ((x.len() - 3) as f64).sqrt();
            ((z - radius).tanh(), (z + radius).tanh())
        };
        out.interval = Some((
            Interval::new(checked(lower)?, checked(upper)?)?,
            options.confidence,
        ));
    }
    Ok(out)
}

/// Computes the result of [`point_biserial_test_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`point_biserial_test_with_options`] for the fixed default settings.
pub fn point_biserial_test<S: RealScalar>(
    binary: &[S],
    values: &[S],
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    point_biserial_test_with_options(binary, values, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("binary"),
        )
    })
}
/// Tests for zero point-biserial correlation between a binary and a continuous
/// variable.
///
/// The point-biserial correlation is the Pearson correlation with a 0/1
/// variable, so the test is [`pearson_test_with_options`] applied to `binary` and `values`.
///
/// # Errors
///
/// - `InvalidInput` if `binary` contains a value other than 0 or 1.
/// - Any error of [`pearson_test_with_options`].
pub fn point_biserial_test_with_options<S: RealScalar>(
    binary: &[S],
    values: &[S],
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    if binary.iter().any(|&x| x != S::ZERO && x != S::ONE) {
        return Err(invalid(
            scioxide_core::argument!("binary"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "point-biserial predictor must contain only zero and one",
        ));
    }
    pearson_test_with_options(binary, values, options)
}

// Used by rank tests; preserve direct normal tails for very small probabilities.
fn association_normal<S: RealScalar>(
    statistic: f64,
    z: f64,
    options: ClassicalOptions,
    n: usize,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    let mut out = result(
        statistic,
        normal_p(z, options.alternative)?,
        ClassicalMethod::Normal,
        TestSamplePolicy::Paired,
        [n, n],
    )?;
    out.effect = Some((ClassicalEffect::Association, checked(statistic)?));
    Ok(out)
}

/// Selects the normalization of Kendall's tau.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KendallVariant {
    /// Tau-b: `τ_b = S / √((n₀ − n₁)(n₀ − n₂))`, where `n₀ = n(n − 1)/2` and
    /// `n₁`, `n₂` count the pairs tied in `x` and in `y`. Suited to square
    /// tables.
    TauB,
    /// Stuart's tau-c: `τ_c = 2S / (n² (m − 1)/m)`, where `m` is the smaller
    /// number of distinct values in `x` and `y`. Suited to rectangular tables.
    TauC,
}
/// Tests for zero Kendall rank correlation between paired samples.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`kendall_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`kendall_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = kendall_test::<f64>(&x, &y, KendallVariant::TauB)?;
/// let configured = kendall_test_with_options::<f64>(&x, &y, KendallVariant::TauB, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. G. Kendall, "The treatment of ties in ranking problems", *Biometrika* 33
/// (1945), 239–251.
///
/// W. R. Knight, "A computer method for calculating Kendall's tau with
/// ungrouped data", *J. Amer. Statist. Assoc.* 61 (1966), 436–439.
pub fn kendall_test<S: RealScalar>(
    x: &[S],
    y: &[S],
    variant: KendallVariant,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    kendall_test_with_options(x, y, variant, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Tests for zero Kendall rank correlation between paired samples.
///
/// `S` is the number of concordant minus discordant pairs. Pairs tied in `x` or
/// in `y` count as neither. The statistic is tau-b or tau-c, selected by
/// `variant`. Counting `S` costs O(n log n) time and O(n) memory.
///
/// The exact p-value enumerates all `n!` pairings of `x` with `y`, at O(n²)
/// per pairing, and is valid with ties. Under [`TestCalibration::Auto`] the
/// test is exact when `n!` is at most [`ClassicalOptions::max_work`].
/// Otherwise it uses the normal approximation to `S` with the tie-corrected
/// variance.
///
/// # Errors
///
/// - `DimensionMismatch` if `x` and `y` have different lengths.
/// - `InvalidInput` if there are fewer than two pairs, or fewer than three for
///   the normal approximation.
/// - `InvalidInput` if `x` or `y` is constant, or the null variance of `S` is
///   zero.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
/// - `ConvergenceFailure` if the calibration is [`TestCalibration::Exact`] and
///   `n!` exceeds `max_work`.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{KendallVariant, kendall_test_with_options};
///
/// // Every pair is discordant, so τ_b = −1.
/// let x = [1.0_f64, 2.0, 3.0];
/// let y = [3.0, 2.0, 1.0];
/// let result = kendall_test_with_options(&x, &y, KendallVariant::TauB, Default::default())?;
/// assert!((result.statistic() + 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. G. Kendall, "The treatment of ties in ranking problems", *Biometrika* 33
/// (1945), 239–251.
///
/// W. R. Knight, "A computer method for calculating Kendall's tau with
/// ungrouped data", *J. Amer. Statist. Assoc.* 61 (1966), 436–439.
pub fn kendall_test_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    variant: KendallVariant,
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    paired(x, y, 2)?;
    let counts = kendall::Counts::new(x, y)?;
    let score = counts.score;
    let [xt, yt] = counts.ties;
    let (tx, ty) = (xt.sums.pairs / 2.0, yt.sums.pairs / 2.0);
    let n = x.len() as f64;
    let pairs = n * (n - 1.0) / 2.0;
    if tx == pairs || ty == pairs {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "constant marginal has undefined Kendall coefficient",
        ));
    }
    let (ax, bx, cx, dx) = (xt.sums.pairs, xt.sums.triples, xt.sums.kendall, xt.distinct);
    let (ay, by, cy, dy) = (yt.sums.pairs, yt.sums.triples, yt.sums.kendall, yt.distinct);
    drop(counts);
    let statistic = (match variant {
        KendallVariant::TauB => score / (pairs - tx).sqrt() / (pairs - ty).sqrt(),
        KendallVariant::TauC => {
            let m = dx.min(dy) as f64;
            2.0 * score / (n * n * (m - 1.0) / m)
        }
    })
    .clamp(-1.0, 1.0);
    if permutation::use_exact(scioxide_core::checked_factorial(x.len()), options)? {
        let callback = |s: crate::ResampledIndices<'_>| -> Result<f64, SciOxideError> {
            let indices = s.group(0)?;
            let mut score = 0.0;
            for (i, &a) in x.iter().enumerate() {
                for (j, &b) in x[..i].iter().enumerate() {
                    let c = y[indices[i]];
                    let d = y[indices[j]];
                    score += if a == b || c == d {
                        0.0
                    } else if (a > b) == (c > d) {
                        1.0
                    } else {
                        -1.0
                    };
                }
            }
            Ok(score)
        };
        let p = permutation::exact(&[x.len()], PermutationKind::Pairings, options, &callback)?.1;
        let mut out = result(
            statistic,
            p,
            ClassicalMethod::Exact,
            TestSamplePolicy::Paired,
            [x.len(), y.len()],
        )?;
        out.effect = Some((ClassicalEffect::Association, checked(statistic)?));
        return Ok(out);
    }
    if x.len() < 3 {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "Kendall normal inference requires at least three pairs",
        ));
    }
    let variance = (n * (n - 1.0) * (2.0 * n + 5.0) - cx - cy) / 18.0
        + ax * ay / (2.0 * n * (n - 1.0))
        + bx * by / (9.0 * n * (n - 1.0) * (n - 2.0));
    if variance <= 0.0 {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "undefined Kendall null variance",
        ));
    }
    association_normal(statistic, score / variance.sqrt(), options, x.len())
}
/// Tests for zero Spearman rank correlation between paired samples.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`spearman_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`spearman_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = spearman_test::<f64>(&x, &y)?;
/// let configured = spearman_test_with_options::<f64>(&x, &y, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn spearman_test<S: RealScalar>(
    x: &[S],
    y: &[S],
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    spearman_test_with_options(x, y, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Tests for zero Spearman rank correlation between paired samples.
///
/// The statistic `ρ` is the Pearson correlation of the midranks of `x` and `y`.
/// The exact p-value enumerates all `n!` pairings and is valid with ties. Under
/// [`TestCalibration::Auto`] the test is exact when `n!` is at most
/// [`ClassicalOptions::max_work`]. Otherwise it uses
/// `t = ρ √((n − 2)/(1 − ρ²))` with Student's t distribution on `n − 2`
/// degrees of freedom.
///
/// # Errors
///
/// - `DimensionMismatch` if `x` and `y` have different lengths.
/// - `InvalidInput` if there are fewer than two pairs, or fewer than three for
///   the t approximation.
/// - `InvalidInput` if `x` or `y` is constant.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
/// - `ConvergenceFailure` if the calibration is [`TestCalibration::Exact`] and
///   `n!` exceeds `max_work`.
pub fn spearman_test_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    paired(x, y, 2)?;
    let rx = ranks::ranks(x, RankMethod::Average)?;
    let ry = ranks::ranks(y, RankMethod::Average)?;
    let n = x.len() as f64;
    let center = (n + 1.0) / 2.0;
    let sx = rx.iter().map(|v| (v - center) * (v - center)).sum::<f64>();
    let sy = ry.iter().map(|v| (v - center) * (v - center)).sum::<f64>();
    if sx == 0.0 || sy == 0.0 {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "constant marginal has undefined Spearman coefficient",
        ));
    }
    let denominator = sx.sqrt() * sy.sqrt();
    let rho = rx
        .iter()
        .zip(&ry)
        .map(|(a, b)| (a - center) * (b - center))
        .sum::<f64>()
        / denominator;
    let (p, method) = if permutation::use_exact(scioxide_core::checked_factorial(x.len()), options)?
    {
        let callback = |s: crate::ResampledIndices<'_>| -> Result<f64, SciOxideError> {
            Ok(rx
                .iter()
                .zip(s.group(0)?)
                .map(|(a, &i)| (a - center) * (ry[i] - center))
                .sum::<f64>()
                / denominator)
        };
        (
            permutation::exact(&[x.len()], PermutationKind::Pairings, options, &callback)?.1,
            ClassicalMethod::Exact,
        )
    } else {
        if x.len() < 3 {
            return Err(invalid(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "Spearman asymptotic inference requires three pairs",
            ));
        }
        let df = n - 2.0;
        let p = if rho.abs() >= 1.0 {
            match options.alternative {
                TestAlternative::TwoSided => 0.0,
                TestAlternative::Less => {
                    if rho < 0.0 {
                        0.0
                    } else {
                        1.0
                    }
                }
                TestAlternative::Greater => {
                    if rho > 0.0 {
                        0.0
                    } else {
                        1.0
                    }
                }
            }
        } else {
            student_p(
                rho * ((df / (1.0 - rho)) / (1.0 + rho)).sqrt(),
                df,
                options.alternative,
            )?
        };
        (
            p,
            ClassicalMethod::StudentT {
                degrees_of_freedom: df,
            },
        )
    };
    let mut out = result(
        rho.clamp(-1.0, 1.0),
        p,
        method,
        TestSamplePolicy::Paired,
        [x.len(), y.len()],
    )?;
    out.effect = Some((ClassicalEffect::Association, checked(rho.clamp(-1.0, 1.0))?));
    Ok(out)
}
/// Tests independence with Chatterjee's rank correlation `ξₙ`.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`chatterjee_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`chatterjee_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = chatterjee_test::<f64>(&x, &y)?;
/// let configured = chatterjee_test_with_options::<f64>(&x, &y, ClassicalOptions::default().with_alternative(TestAlternative::Greater))?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// S. Chatterjee, "A new coefficient of correlation", *J. Amer. Statist.
/// Assoc.* 116 (2021), 2009–2022.
pub fn chatterjee_test<S: RealScalar>(
    x: &[S],
    y: &[S],
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    chatterjee_test_with_options(
        x,
        y,
        ClassicalOptions::default().with_alternative(TestAlternative::Greater),
    )
    .map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Tests independence with Chatterjee's rank correlation `ξₙ`.
///
/// ```text
/// ξₙ = 1 − n Σᵢ |rᵢ₊₁ − rᵢ| / (2 Σᵢ lᵢ (n − lᵢ)),
/// ```
///
/// where the pairs are sorted by `x`, `y₍ᵢ₎` is the response of the `i`-th
/// sorted pair, `rᵢ` is the number of `j` with `yⱼ ≤ y₍ᵢ₎` and `lᵢ` the number
/// with `yⱼ ≥ y₍ᵢ₎`. Ties in `x` are broken by input order. `ξₙ` estimates a
/// measure of dependence that is 0 under independence and 1 when `y` is a
/// measurable function of `x`. The alternative must be
/// [`TestAlternative::Greater`].
///
/// The exact p-value enumerates all `n!` pairings. Under
/// [`TestCalibration::Auto`] the test is exact when `n!` is at most
/// [`ClassicalOptions::max_work`]. Otherwise it uses Chatterjee's normal
/// approximation with the variance for possibly tied `y`. The cost is
/// O(n log n) time and O(n) memory.
///
/// # Errors
///
/// - `DimensionMismatch` if `x` and `y` have different lengths.
/// - `InvalidInput` if there are fewer than three pairs, `y` is constant or
///   the alternative is not `Greater`.
/// - `InvalidInput` if the asymptotic variance is zero.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
/// - `ConvergenceFailure` if the calibration is [`TestCalibration::Exact`] and
///   `n!` exceeds `max_work`.
///
/// # References
///
/// S. Chatterjee, "A new coefficient of correlation", *J. Amer. Statist.
/// Assoc.* 116 (2021), 2009–2022.
pub fn chatterjee_test_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    paired(x, y, 3)?;
    if options.alternative != TestAlternative::Greater {
        return Err(invalid(
            scioxide_core::argument!("options.alternative"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "xi independence inference uses the upper tail",
        ));
    }
    let r = ranks::ranks(y, RankMethod::Maximum)?;
    let min = ranks::ranks(y, RankMethod::Minimum)?;
    let n = x.len() as f64;
    let denominator = 2.0
        * min
            .iter()
            .map(|&v| {
                let l = n + 1.0 - v;
                l * (n - l)
            })
            .sum::<f64>();
    if denominator == 0.0 {
        return Err(invalid(
            scioxide_core::argument!("y"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "constant response has undefined xi",
        ));
    }
    let mut order = try_with_capacity(x.len(), scioxide_core::quantity!("xi_predictor_order"))?;
    order.extend(0..x.len());
    order.sort_unstable_by(|&i, &j| crate::ranking::finite_cmp(&x[i], &x[j]).then(i.cmp(&j)));
    let score = |indices: &[usize]| {
        1.0 - n * order
            .windows(2)
            .map(|pair| (r[indices[pair[1]]] - r[indices[pair[0]]]).abs())
            .sum::<f64>()
            / denominator
    };
    let mut identity = try_with_capacity(x.len(), scioxide_core::quantity!("xi_identity"))?;
    identity.extend(0..x.len());
    let statistic = score(&identity);
    let (p, method) = if permutation::use_exact(scioxide_core::checked_factorial(x.len()), options)?
    {
        let callback = |s: crate::ResampledIndices<'_>| -> Result<f64, SciOxideError> {
            Ok(score(s.group(0)?))
        };
        (
            permutation::exact(&[x.len()], PermutationKind::Pairings, options, &callback)?.1,
            ClassicalMethod::Exact,
        )
    } else {
        let mut sorted = r;
        crate::ranking::sort_finite(&mut sorted);
        let (mut an, mut bn, mut cn, mut cumulative) = (0.0, 0.0, 0.0, 0.0);
        for (index, &rank) in sorted.iter().enumerate() {
            let i = (index + 1) as f64;
            let u = rank / n;
            cumulative += u;
            let weight = (2.0 * n - 2.0 * i + 1.0) / n;
            an += weight * u * u / n;
            cn += weight * u / n;
            let v = (cumulative + (n - i) * u) / n;
            bn += v * v / n;
        }
        let dn = denominator / (2.0 * n * n * n);
        let variance = (an - 2.0 * bn + cn * cn) / (dn * dn * n);
        if variance <= 0.0 {
            return Err(invalid(
                scioxide_core::argument!("y"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "xi asymptotic variance vanishes",
            ));
        }
        (
            normal_p(statistic / variance.sqrt(), TestAlternative::Greater)?,
            ClassicalMethod::Normal,
        )
    };
    let mut out = result(
        statistic,
        p,
        method,
        TestSamplePolicy::Paired,
        [x.len(), y.len()],
    )?;
    out.effect = Some((ClassicalEffect::Association, checked(statistic)?));
    Ok(out)
}

/// Selects how [`weighted_kendall_tau`] combines the weights of a pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KendallWeightCombination {
    /// Weight the pair by `wᵢ + wⱼ`.
    #[default]
    Additive,
    /// Weight the pair by `wᵢ wⱼ`.
    Multiplicative,
}

/// Computes the weighted Kendall rank correlation coefficient.
///
/// Each pair `(i, j)` contributes `+w`, `−w` or 0 to the score as it is
/// concordant, discordant or tied, where `w` combines the observation weights
/// as selected by `combination`. The score is normalized like tau-b, with tied
/// pairs left out of each marginal denominator. With `importance = Some(w)` the
/// weights are `w`, one non-negative value per observation. With `None` the
/// result is the mean of two coefficients with hyperbolic weights `1/(r + 1)`,
/// where `r` is the rank in decreasing lexicographic order of `(x, y)` and of
/// `(y, x)`. Pair weights are accumulated in extended range, so weights near
/// the overflow and underflow thresholds are accepted.
///
/// Returns the coefficient without a null-distribution calculation.
/// The cost is O(n log n) time and O(n)
/// memory.
///
/// # Errors
///
/// - `DimensionMismatch` if `x` and `y` have different lengths.
/// - `InvalidInput` if there are fewer than two pairs.
/// - `InvalidInput` if `importance` has the wrong length, a negative entry or
///   no positive entry.
/// - `InvalidInput` if a weighted denominator is zero, for example because `x`
///   or `y` is constant.
/// - An error if `x`, `y` or `importance` contains NaN or infinity.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{KendallWeightCombination, weighted_kendall_tau};
///
/// // One concordant pair gives a coefficient of 1 for any positive weights.
/// let weights = [1e300, 1e-300];
/// let combination = KendallWeightCombination::Multiplicative;
/// let result = weighted_kendall_tau(&[0.0_f64, 1.0], &[0.0, 1.0], Some(&weights), combination)?;
/// assert!((result - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// S. Vigna, "A weighted correlation index for rankings with ties", *Proc. 24th
/// International Conference on World Wide Web* (2015).
pub fn weighted_kendall_tau<S: RealScalar>(
    x: &[S],
    y: &[S],
    importance: Option<&[f64]>,
    combination: KendallWeightCombination,
) -> Result<S, SciOxideError> {
    paired(x, y, 2)?;
    let counts = kendall::Counts::new(x, y)?;
    let coefficient = |weights: &[f64]| -> Result<f64, SciOxideError> {
        if weights.len() != x.len() {
            return Err(invalid(
                scioxide_core::argument!("importance"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "importance length mismatch",
            ));
        }
        let (positive, native) =
            weights
                .iter()
                .try_fold((false, true), |(positive, native), &w| {
                    finite(w)?;
                    if w < 0.0 {
                        return Err(invalid(
                            scioxide_core::argument!("importance"),
                            scioxide_core::InvalidInputReason::Inconsistent,
                            "negative importance",
                        ));
                    }
                    Ok((
                        positive || w > 0.0,
                        native && crate::arithmetic::native(w, 64),
                    ))
                })?;
        if !positive {
            return Err(invalid(
                scioxide_core::argument!("importance"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "all importance weights are zero",
            ));
        }
        use scioxide_core::Scaled;
        let native = if native {
            let sums = counts.weighted::<f64>(weights, combination)?;
            // A conservative gamma bound for the tree and final reductions.
            // Near cancellation, recompute from original weights in extended range.
            let error = 16.0 * x.len() as f64 * (x.len().ilog2() + 1) as f64 * f64::EPSILON;
            (error < 0.25 && sums[0].abs() > error / (1.0 - error) * (sums[1] + sums[2]))
                .then_some(sums)
        } else {
            None
        };
        let [score, vx, vy] = if let Some(sums) = native {
            crate::arithmetic::encode_components(sums)?
        } else {
            counts.weighted::<Scaled<f64>>(weights, combination)?
        };
        if vx.is_zero() || vy.is_zero() {
            return Err(invalid(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "weighted Kendall denominator vanishes",
            ));
        }
        Ok(score
            .div(vx.sqrt()?)?
            .div(vy.sqrt()?)?
            .round()?
            .clamp(-1.0, 1.0))
    };
    let statistic = if let Some(w) = importance {
        coefficient(w)?
    } else {
        let mut weights = scioxide_core::backend::try_filled(
            x.len(),
            0.0,
            scioxide_core::quantity!("kendall_importance"),
        )?;
        let mut sum = 0.0;
        for axis in 0..2 {
            counts.importance(axis, &mut weights);
            sum += coefficient(&weights)?;
        }
        sum / 2.0
    };
    checked(statistic)
}
/// Tests for zero Somers' `D(Y|X)` between paired samples.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`somers_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`somers_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = somers_test::<f64>(&x, &y)?;
/// let configured = somers_test_with_options::<f64>(&x, &y, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. B. Brown and J. K. Benedetti, "Sampling behavior of tests for correlation
/// in two-way contingency tables", *J. Amer. Statist. Assoc.* 72 (1977),
/// 309–315.
pub fn somers_test<S: RealScalar>(
    x: &[S],
    y: &[S],
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    somers_test_with_options(x, y, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Tests for zero Somers' `D(Y|X)` between paired samples.
///
/// ```text
/// D(Y|X) = S / (n(n − 1)/2 − Tₓ),
/// ```
///
/// where `S` is the number of concordant minus discordant pairs and `Tₓ` the
/// number of pairs tied in `x`. `x` is the predictor and `y` the response. The
/// p-value uses the normal approximation with the asymptotic variance of Brown
/// and Benedetti. The cost is O(n²) time and O(n) memory.
///
/// # Errors
///
/// - `DimensionMismatch` if `x` and `y` have different lengths.
/// - `InvalidInput` if there are fewer than two pairs or `x` is constant.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`].
/// - `InvalidInput` if the estimated variance is zero.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
///
/// # References
///
/// M. B. Brown and J. K. Benedetti, "Sampling behavior of tests for correlation
/// in two-way contingency tables", *J. Amer. Statist. Assoc.* 72 (1977),
/// 309–315.
pub fn somers_test_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    paired(x, y, 2)?;
    if options.calibration == TestCalibration::Exact {
        return Err(invalid(
            scioxide_core::argument!("options.calibration"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "use pairing permutations for exact independence inference",
        ));
    }
    let counts = kendall::Counts::new(x, y)?;
    let (score, tx) = (counts.score, counts.ties[0].sums.pairs / 2.0);
    drop(counts);
    let n = x.len() as f64;
    let denominator = n * (n - 1.0) / 2.0 - tx;
    if denominator == 0.0 {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "constant predictor has undefined Somers D",
        ));
    }
    let influences = x.iter().zip(y).map(|(&a, &b)| {
        x.iter()
            .zip(y)
            .map(|(&c, &d)| {
                if a == c || b == d {
                    0.0
                } else if (a > c) == (b > d) {
                    1.0
                } else {
                    -1.0
                }
            })
            .sum::<f64>()
    });
    let centered = 2.0 * score / n;
    let variance = influences
        .map(|v| (v - centered) * (v - centered))
        .sum::<f64>();
    if variance == 0.0 {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Somers normal variance vanishes; use conditional permutation inference",
        ));
    }
    association_normal(
        score / denominator,
        score / variance.sqrt(),
        options,
        x.len(),
    )
}

/// Tests for zero Somers' `D(column|row)` in an ordinal contingency table.
///
/// `table` holds the counts of an `R × C` table. Rows and columns
/// are ordered categories: the row is the predictor and the column the
/// response. The statistic is the [`somers_test_with_options`] statistic of the expanded
/// observations, computed from the table without expanding it. The p-value
/// uses a normal approximation. Empty rows and columns are allowed. The cost is
/// O((R C)²) time and O(1) memory for `R` rows and `C` columns.
///
/// # Errors
///
/// - `InvalidInput` if the table has fewer than two rows or columns, or it
///   holds fewer than two observations.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`].
/// - `InvalidInput` if all observations fall in one row, or the estimated
///   variance is zero.
/// - `InvalidInput` if the total count exceeds `usize::MAX`.
pub fn somers_contingency_test_with_options(
    table: crate::ContingencyView<'_>,
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<f64>, SciOxideError> {
    let counts = table.counts();
    let columns = table.columns();
    if table.rows() < 2 || table.columns() < 2 || options.calibration == TestCalibration::Exact {
        return Err(invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid Somers contingency shape/calibration",
        ));
    }
    let count = usize::try_from(counts.iter().try_fold(0_u128, |a, &b| {
        a.checked_add(b as u128).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })
    })?)
    .map_err(|_| {
        invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Somers retained count exceeds usize",
        )
    })?;
    if count < 2 {
        return Err(invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "insufficient contingency observations",
        ));
    }
    let n = count as f64;
    let score_at = |i: usize| {
        counts
            .iter()
            .enumerate()
            .map(|(j, &weight)| {
                let a = i / columns;
                let b = i % columns;
                let c = j / columns;
                let d = j % columns;
                if a == c || b == d {
                    0.0
                } else if (a > c) == (b > d) {
                    weight as f64
                } else {
                    -(weight as f64)
                }
            })
            .sum::<f64>()
    };
    let score = counts
        .iter()
        .enumerate()
        .map(|(i, &w)| w as f64 * score_at(i))
        .sum::<f64>();
    let row_squares = counts
        .chunks_exact(columns)
        .map(|row| {
            let w = row.iter().map(|&v| v as f64).sum::<f64>();
            w * w
        })
        .sum::<f64>();
    let denominator = n * n - row_squares;
    if denominator <= 0.0 {
        return Err(invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "constant contingency predictor",
        ));
    }
    let variance = counts
        .iter()
        .enumerate()
        .map(|(i, &w)| {
            let d = score_at(i) - score / n;
            w as f64 * d * d
        })
        .sum::<f64>();
    if variance <= 0.0 {
        return Err(invalid(
            scioxide_core::argument!("table"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Somers contingency variance vanishes",
        ));
    }
    let mut out = association_normal(
        score / denominator,
        score / (2.0 * variance.sqrt()),
        options,
        count,
    )?;
    out.sample_policy = TestSamplePolicy::Counts;
    Ok(out)
}

/// Computes the result of [`somers_contingency_test_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the configured form's errors for these defaults.
pub fn somers_contingency_test(
    table: crate::ContingencyView<'_>,
) -> Result<ClassicalTestResult<f64>, SciOxideError> {
    somers_contingency_test_with_options(table, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &[
                "options",
                "weights",
                "left_weights",
                "right_weights",
                "direction",
                "ties",
            ],
            scioxide_core::argument!("table"),
        )
    })
}
