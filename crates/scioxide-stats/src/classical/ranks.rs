//! Rank tests: Mann–Whitney, rank-sum, Wilcoxon signed-rank, Brunner–Munzel and BWS.
use super::*;
use crate::ranking::RankPool;
use crate::{PermutationKind, RankMethod, rankdata};
use alloc::vec::Vec;
use scioxide_core::backend::{try_filled, try_with_capacity};

pub(super) fn ranks<S: RealScalar>(x: &[S], method: RankMethod) -> Result<Vec<f64>, SciOxideError> {
    let ranks = rankdata(x, method)?;
    let mut out = try_with_capacity(x.len(), scioxide_core::quantity!("classical_ranks"))?;
    for r in ranks {
        out.push(r.to_f64()?);
    }
    Ok(out)
}
pub(super) fn pooled_ranks<S: RealScalar>(x: &[S], y: &[S]) -> Result<Vec<f64>, SciOxideError> {
    pooled_rank_data(x, y).map(|(ranks, _)| ranks)
}

/// Sorts the pooled samples, keeping each value's position so its sample label
/// is known.
fn ordered_pool<S: RealScalar>(x: &[S], y: &[S]) -> Result<RankPool<S, usize>, SciOxideError> {
    RankPool::positions(&[x, y], scioxide_core::quantity!("pooled_sample"))
}

fn pooled_rank_data<S: RealScalar>(x: &[S], y: &[S]) -> Result<(Vec<f64>, bool), SciOxideError> {
    ordered_pool(x, y)?.midranks(scioxide_core::quantity!("classical_ranks"))
}
/// Performs the Mann–Whitney U test of two independent samples.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`mann_whitney_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`mann_whitney_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = mann_whitney_test::<f64>(&x, &y)?;
/// let configured = mann_whitney_test_with_options::<f64>(&x, &y, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// H. B. Mann and D. R. Whitney, "On a test of whether one of two random
/// variables is stochastically larger than the other", *Ann. Math. Statist.* 18
/// (1947), 50–60.
pub fn mann_whitney_test<S: RealScalar>(
    x: &[S],
    y: &[S],
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    mann_whitney_test_with_options(x, y, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the Mann–Whitney U test of two independent samples.
///
/// ```text
/// U = R₁ − n(n + 1)/2,
/// ```
///
/// where `R₁` is the sum of the pooled midranks of `x` and `n`, `m` are the
/// sample sizes. Large `U` means `x` tends to be larger. The effect is
/// `U / (n m)`, an estimate of `P(X > Y) + P(X = Y)/2`.
///
/// The exact p-value enumerates all `C(n + m, n)` label assignments of the
/// observed midranks, so it is valid with ties. Under
/// [`TestCalibration::Auto`] the test is exact when `C(n + m, n)` is at most
/// [`ClassicalOptions::max_work`]. Otherwise it uses the normal approximation
/// with tie-corrected variance and the configured continuity correction.
/// Ranking costs O(N log N) time and O(N) memory, `N = n + m`.
///
/// # Errors
///
/// - `InvalidInput` if either sample is empty.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
/// - `ConvergenceFailure` if the calibration is [`TestCalibration::Exact`] and
///   `C(n + m, n)` exceeds `max_work`.
///
/// # References
///
/// H. B. Mann and D. R. Whitney, "On a test of whether one of two random
/// variables is stochastically larger than the other", *Ann. Math. Statist.* 18
/// (1947), 50–60.
pub fn mann_whitney_test_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    sample(x, 1, scioxide_core::argument!("x"))?;
    sample(y, 1, scioxide_core::argument!("y"))?;
    let pool = pooled_ranks(x, y)?;
    let n = x.len() as f64;
    let m = y.len() as f64;
    let u = pool[..x.len()].iter().sum::<f64>() - n * (n + 1.0) / 2.0;
    let exact = permutation::use_exact(
        scioxide_core::checked_binomial(pool.len(), x.len()),
        options,
    )?;
    let (p, method) = if exact {
        let callback = |s: crate::ResampledIndices<'_>| -> Result<f64, SciOxideError> {
            Ok(s.group(0)?.iter().map(|&i| pool[i]).sum::<f64>() - n * (n + 1.0) / 2.0)
        };
        (
            permutation::exact(
                &[x.len(), y.len()],
                PermutationKind::Independent,
                options,
                &callback,
            )?
            .1,
            ClassicalMethod::Exact,
        )
    } else {
        let center = (pool.len() + 1) as f64 / 2.0;
        let variance = pool
            .iter()
            .map(|r| (r - center) * (r - center))
            .sum::<f64>()
            * n
            * m
            / ((n + m) * (n + m - 1.0));
        let p = if variance == 0.0 {
            1.0
        } else {
            rank_normal(u - n * m / 2.0, variance.sqrt(), options)?
        };
        (p, ClassicalMethod::Normal)
    };
    let mut out = result(
        u,
        p,
        method,
        TestSamplePolicy::IndependentGroups,
        [x.len(), y.len()],
    )?;
    out.effect = Some((ClassicalEffect::Superiority, checked(u / (n * m))?));
    Ok(out)
}

pub(super) fn rank_normal(
    centered: f64,
    se: f64,
    options: ClassicalOptions,
) -> Result<f64, SciOxideError> {
    let correction = if options.continuity == crate::ContinuityCorrection::HalfUnit {
        0.5
    } else {
        0.0
    };
    let adjusted = match options.alternative {
        TestAlternative::Less => centered + correction,
        TestAlternative::Greater => centered - correction,
        TestAlternative::TwoSided => centered.signum() * (centered.abs() - correction).max(0.0),
    };
    normal_p(adjusted / se, options.alternative)
}
/// Performs the Wilcoxon rank-sum test with the normal approximation.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`rank_sum_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`rank_sum_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = rank_sum_test::<f64>(&x, &y)?;
/// let configured = rank_sum_test_with_options::<f64>(&x, &y, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn rank_sum_test<S: RealScalar>(
    x: &[S],
    y: &[S],
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    rank_sum_test_with_options(x, y, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the Wilcoxon rank-sum test with the normal approximation.
///
/// ```text
/// z = (U − n m/2) / √(n m (n + m + 1)/12),
/// ```
///
/// where `U` is the Mann–Whitney statistic of [`mann_whitney_test_with_options`] and `n`, `m`
/// are the sample sizes. The statistic is `z`, with no tie or continuity
/// correction, so the data must be untied. For tied data or an exact p-value,
/// use [`mann_whitney_test_with_options`]. The cost is O(N log N) time and O(N) memory,
/// `N = n + m`.
///
/// # Errors
///
/// - `InvalidInput` if either sample is empty or the pooled sample has ties.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`].
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
///
/// # Examples
///
/// ```
/// use scioxide_stats::rank_sum_test_with_options;
///
/// // U = 3 is below its null mean n m/2 = 4.5, so z < 0.
/// let test = rank_sum_test_with_options(&[1.0_f64, 3.0, 5.0], &[2.0, 4.0, 6.0], Default::default())?;
/// assert!(test.statistic() < 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn rank_sum_test_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    if options.calibration == TestCalibration::Exact {
        return Err(invalid(
            scioxide_core::argument!("options.calibration"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "rank-sum normal test has no exact calibration; use Mann–Whitney",
        ));
    }
    sample(x, 1, scioxide_core::argument!("x"))?;
    sample(y, 1, scioxide_core::argument!("y"))?;
    let (pool, tied) = pooled_rank_data(x, y)?;
    if tied {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "rank-sum normal test requires no ties",
        ));
    }
    let n = x.len() as f64;
    let m = y.len() as f64;
    let u = pool[..x.len()].iter().sum::<f64>() - n * (n + 1.0) / 2.0;
    let z = (u - n * m / 2.0) / (n * m * (n + m + 1.0) / 12.0).sqrt();
    let mut out = result(
        z,
        normal_p(z, options.alternative)?,
        ClassicalMethod::Normal,
        TestSamplePolicy::IndependentGroups,
        [x.len(), y.len()],
    )?;
    out.effect = Some((ClassicalEffect::Superiority, checked(u / (n * m))?));
    Ok(out)
}

/// Selects how the Wilcoxon signed-rank test treats zero differences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SignedRankZeros {
    /// Wilcoxon's method: drop zeros before ranking the absolute differences.
    #[default]
    Discard,
    /// Pratt's method: rank zeros with the other differences, then leave their
    /// ranks out of both signed-rank sums.
    Pratt,
    /// Rank zeros and add half of their rank sum to each signed-rank sum.
    Split,
}
/// Performs the Wilcoxon signed-rank test that independent differences are symmetric about zero.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`wilcoxon_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`wilcoxon_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let actual = wilcoxon_test::<f64>(&x)?;
/// let configured = wilcoxon_test_with_options::<f64>(&x, SignedRankZeros::Discard, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// F. Wilcoxon, "Individual comparisons by ranking methods", *Biometrics
/// Bulletin* 1 (1945), 80–83.
///
/// J. W. Pratt, "Remarks on zeros and ties in the Wilcoxon signed rank
/// procedures", *J. Amer. Statist. Assoc.* 54 (1959), 655–667.
pub fn wilcoxon_test<S: RealScalar>(
    differences: &[S],
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    wilcoxon_test_with_options(differences, SignedRankZeros::Discard, Default::default()).map_err(
        |error| {
            crate::error_helpers::one_shot_argument(
                error,
                &["options", "zeros"],
                scioxide_core::argument!("differences"),
            )
        },
    )
}
/// Performs the Wilcoxon signed-rank test that independent differences are
/// symmetric about zero.
///
/// The absolute differences are ranked with midranks. `W⁺` and `W⁻` are the
/// sums of the ranks of positive and negative differences. The statistic is
/// `min(W⁺, W⁻)` for a two-sided test and `W⁺` otherwise, plus half the zero
/// ranks under [`SignedRankZeros::Split`]. The effect is the matched-pairs
/// rank-biserial correlation `(W⁺ − W⁻)/(W⁺ + W⁻)`. If every difference is zero,
/// the p-value is 1.
///
/// The exact null distribution comes from a subset-sum recursion on doubled
/// midranks, so it is exact with ties. It costs O(K n) time and O(K) memory,
/// where `K` is the sum of the doubled ranks. Under [`TestCalibration::Auto`]
/// the test is exact when `K n` is at most [`ClassicalOptions::max_work`], and
/// uses the normal approximation otherwise.
///
/// # Errors
///
/// - `InvalidInput` if `differences` is empty.
/// - `NonFinite` if `differences` contains NaN or infinity.
/// - `ConvergenceFailure` if the calibration is [`TestCalibration::Exact`] and
///   `K n` exceeds `max_work`.
/// - `ResourceLimit` if the exact distribution needs more than
///   [`ClassicalOptions::max_workspace_bytes`].
///
/// # References
///
/// F. Wilcoxon, "Individual comparisons by ranking methods", *Biometrics
/// Bulletin* 1 (1945), 80–83.
///
/// J. W. Pratt, "Remarks on zeros and ties in the Wilcoxon signed rank
/// procedures", *J. Amer. Statist. Assoc.* 54 (1959), 655–667.
pub fn wilcoxon_test_with_options<S: RealScalar>(
    differences: &[S],
    zeros: SignedRankZeros,
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    sample(differences, 1, scioxide_core::argument!("differences"))?;
    let mut absolute =
        try_with_capacity(differences.len(), scioxide_core::quantity!("signed_ranks"))?;
    let mut signs = try_with_capacity(differences.len(), scioxide_core::quantity!("signed_ranks"))?;
    for &d in differences {
        if d != S::ZERO || zeros != SignedRankZeros::Discard {
            absolute.push(d.abs());
            signs.push(d > S::ZERO);
        }
    }
    let ranks = if absolute.is_empty() {
        Vec::new()
    } else {
        ranks(&absolute, RankMethod::Average)?
    };
    let (mut positive, mut total, mut square, mut zero_mass) = (0.0, 0.0, 0.0, 0.0);
    let mut weights = try_with_capacity(
        absolute.len(),
        scioxide_core::quantity!("doubled_signed_ranks"),
    )?;
    for ((&value, &sign), &rank) in absolute.iter().zip(&signs).zip(&ranks) {
        if value == S::ZERO {
            zero_mass += rank;
        } else {
            total += rank;
            square += rank * rank;
            if sign {
                positive += rank;
            }
            weights.push((2.0 * rank) as usize);
        }
    }
    let offset = if zeros == SignedRankZeros::Split {
        zero_mass / 2.0
    } else {
        0.0
    };
    let negative = total - positive;
    let statistic = if options.alternative == TestAlternative::TwoSided {
        positive.min(negative) + offset
    } else {
        positive + offset
    };
    let states = weights.iter().try_fold(1usize, |a, &b| {
        a.checked_add(b).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })
    })?;
    let work = states
        .checked_mul(weights.len())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")));
    let exact = weights.is_empty() || permutation::use_exact(work, options)?;
    let (p, method) = if exact {
        let bytes = states
            .checked_mul(core::mem::size_of::<f64>())
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?;
        crate::error_helpers::limit(
            bytes,
            options.max_workspace_bytes,
            scioxide_core::quantity!("ranks_bytes"),
        )?;
        let mut dp = try_filled(states, 0.0, scioxide_core::quantity!("signed_rank_null"))?;
        dp[0] = 1.0;
        let mut top = 0;
        for &w in &weights {
            for j in (0..=top + w).rev() {
                dp[j] = 0.5 * dp[j] + if j >= w { 0.5 * dp[j - w] } else { 0.0 };
            }
            top += w;
        }
        let observed = (positive * 2.0) as usize;
        (
            tails(
                dp[..=observed].iter().sum(),
                dp[observed..].iter().sum(),
                options.alternative,
            )?,
            ClassicalMethod::Exact,
        )
    } else {
        (
            rank_normal(positive - total / 2.0, square.sqrt() / 2.0, options)?,
            ClassicalMethod::Normal,
        )
    };
    let mut out = result(
        statistic,
        p,
        method,
        TestSamplePolicy::Paired,
        [differences.len(), differences.len()],
    )?;
    out.effect = Some((
        ClassicalEffect::RankBiserial,
        checked(if total == 0.0 {
            0.0
        } else {
            (positive - negative) / total
        })?,
    ));
    Ok(out)
}

/// Selects the reference distribution of the Brunner–Munzel test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BrunnerMunzelCalibration {
    /// Student's t with Satterthwaite-type estimated degrees of freedom, for
    /// small samples.
    #[default]
    Student,
    /// Large-sample normal approximation.
    Normal,
}
/// Performs the Brunner–Munzel test of `p = P(X > Y) + P(X = Y)/2 = 1/2`.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`brunner_munzel_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`brunner_munzel_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = brunner_munzel_test::<f64>(&x, &y)?;
/// let configured = brunner_munzel_test_with_options::<f64>(&x, &y, BrunnerMunzelCalibration::Student, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// E. Brunner and U. Munzel, "The nonparametric Behrens–Fisher problem:
/// asymptotic theory and a small-sample approximation", *Biometrical Journal*
/// 42 (2000), 17–25.
pub fn brunner_munzel_test<S: RealScalar>(
    x: &[S],
    y: &[S],
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    brunner_munzel_test_with_options(x, y, BrunnerMunzelCalibration::Student, Default::default())
        .map_err(|error| {
            crate::error_helpers::one_shot_argument(
                error,
                &["options", "calibration"],
                scioxide_core::argument!("x"),
            )
        })
}
/// Performs the Brunner–Munzel test of `p = P(X > Y) + P(X = Y)/2 = 1/2`.
///
/// The statistic is `(p̂ − 1/2)/σ̂`, where `p̂` is computed from pooled midranks
/// and `σ̂` is its placement-based standard error. It is positive when `x` tends
/// to be larger. Unlike [`mann_whitney_test_with_options`], the null hypothesis allows the
/// two distributions to differ in shape and spread. The effect is `p̂`. The cost
/// is O(N log N) time and O(N) memory, `N = n + m`.
///
/// # Errors
///
/// - `InvalidInput` if either sample has fewer than two observations.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`].
/// - `InvalidInput` if the estimated variance `σ̂²` is zero, for example when
///   the samples do not overlap.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{BrunnerMunzelCalibration, brunner_munzel_test_with_options};
///
/// // p̂ = P(X > Y) = 6/9 > 1/2, so the statistic is positive.
/// let x = [2.0_f64, 4.0, 6.0];
/// let y = [1.0, 3.0, 5.0];
/// let calibration = BrunnerMunzelCalibration::Student;
/// let test = brunner_munzel_test_with_options(&x, &y, calibration, Default::default())?;
/// assert!(test.statistic() > 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// E. Brunner and U. Munzel, "The nonparametric Behrens–Fisher problem:
/// asymptotic theory and a small-sample approximation", *Biometrical Journal*
/// 42 (2000), 17–25.
pub fn brunner_munzel_test_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    calibration: BrunnerMunzelCalibration,
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    sample(x, 2, scioxide_core::argument!("x"))?;
    sample(y, 2, scioxide_core::argument!("y"))?;
    if options.calibration == TestCalibration::Exact {
        return Err(invalid(
            scioxide_core::argument!("calibration"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Brunner–Munzel heteroscedastic null has no exact label calibration",
        ));
    }
    let n = x.len() as f64;
    let m = y.len() as f64;
    let pool = ordered_pool(x, y)?;
    let mut placements = try_filled(
        pool.len(),
        0.0,
        scioxide_core::quantity!("brunner_munzel_placements"),
    )?;
    let (mut below_x, mut below_y) = (0, 0);
    for (_, group) in pool.runs() {
        let tied_x = group.iter().filter(|&&(_, i)| i < x.len()).count();
        let tied_y = group.len() - tied_x;
        let px = (below_y as f64 + tied_y as f64 * 0.5) / m;
        let py = ((x.len() - below_x - tied_x) as f64 + tied_x as f64 * 0.5) / n;
        for &(_, i) in group {
            placements[i] = if i < x.len() { px } else { py };
        }
        below_x += tied_x;
        below_y += tied_y;
    }
    let (px, py) = placements.split_at(x.len());
    let p = px.iter().sum::<f64>() / n;
    let vx = px.iter().map(|v| (v - p) * (v - p)).sum::<f64>() / ((n - 1.0) * n);
    let vy = py.iter().map(|v| (v - p) * (v - p)).sum::<f64>() / ((m - 1.0) * m);
    let v = vx + vy;
    if v <= 0.0 {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Brunner–Munzel placement variance vanishes",
        ));
    }
    let statistic = (p - 0.5) / v.sqrt();
    let (probability, method) = match calibration {
        BrunnerMunzelCalibration::Normal => (
            normal_p(statistic, options.alternative)?,
            ClassicalMethod::Normal,
        ),
        BrunnerMunzelCalibration::Student => {
            let df = v * v / (vx * vx / (n - 1.0) + vy * vy / (m - 1.0));
            (
                student_p(statistic, df, options.alternative)?,
                ClassicalMethod::StudentT {
                    degrees_of_freedom: df,
                },
            )
        }
    };
    let mut out = result(
        statistic,
        probability,
        method,
        TestSamplePolicy::IndependentGroups,
        [x.len(), y.len()],
    )?;
    out.effect = Some((ClassicalEffect::Superiority, checked(p)?));
    Ok(out)
}

/// Computes the BWS statistic from pooled midranks. The two-sided form squares
/// each discrepancy and the directional form keeps its sign.
pub(super) fn bws_score(pool: &[f64], a: &[usize], b: &[usize], two_sided: bool) -> f64 {
    let part = |indices: &[usize], other: usize| {
        let n = indices.len() as f64;
        let m = other as f64;
        indices
            .iter()
            .enumerate()
            .map(|(position, &j)| {
                let i = 1 + indices
                    .iter()
                    .enumerate()
                    .filter(|&(k, &l)| pool[l] < pool[j] || (pool[l] == pool[j] && k < position))
                    .count();
                let i = i as f64;
                let delta = pool[j] - (n + m) * i / n;
                let numerator = if two_sided {
                    delta * delta
                } else {
                    delta * delta.abs()
                };
                numerator / (i / (n + 1.0) * (1.0 - i / (n + 1.0)) * m * (m + n))
            })
            .sum::<f64>()
    };
    let x = part(a, b.len());
    let y = part(b, a.len());
    if two_sided {
        (x + y) / 2.0
    } else {
        (x - y) / 2.0
    }
}
/// Performs the exact Baumgartner–Weiss–Schindler (BWS) two-sample test.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`bws_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`bws_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = bws_test::<f64>(&x, &y)?;
/// let configured = bws_test_with_options::<f64>(&x, &y, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// W. Baumgartner, P. Weiß and H. Schindler, "A nonparametric test for the
/// general two-sample problem", *Biometrics* 54 (1998), 1129–1135.
pub fn bws_test<S: RealScalar>(x: &[S], y: &[S]) -> Result<ClassicalTestResult<S>, SciOxideError> {
    bws_test_with_options(x, y, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the exact Baumgartner–Weiss–Schindler (BWS) two-sample test.
///
/// The two-sided statistic `B` measures the weighted squared distance between
/// the pooled ranks of each sample and their expected positions. Large `B`
/// indicates different distributions. The directional alternatives use a
/// signed form of the same discrepancy.
///
/// The p-value enumerates all `C(n + m, n)` label assignments of the pooled
/// midranks, so it is valid with ties. The test has no large-sample
/// approximation. For larger samples, use [`BwsPlan`] with a randomized
/// permutation plan.
///
/// # Errors
///
/// - `InvalidInput` if either sample has fewer than two observations.
/// - `InvalidInput` if the calibration is [`TestCalibration::Asymptotic`].
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
/// - An error if `C(n + m, n)` exceeds [`ClassicalOptions::max_work`].
///
/// # References
///
/// W. Baumgartner, P. Weiß and H. Schindler, "A nonparametric test for the
/// general two-sample problem", *Biometrics* 54 (1998), 1129–1135.
pub fn bws_test_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    sample(x, 2, scioxide_core::argument!("x"))?;
    sample(y, 2, scioxide_core::argument!("y"))?;
    if options.calibration == TestCalibration::Asymptotic {
        return Err(invalid(
            scioxide_core::argument!("options.calibration"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "BWS requires permutation calibration",
        ));
    }
    let pool = pooled_ranks(x, y)?;
    let two_sided = options.alternative == TestAlternative::TwoSided;
    let callback = |s: crate::ResampledIndices<'_>| -> Result<f64, SciOxideError> {
        Ok(bws_score(&pool, s.group(0)?, s.group(1)?, two_sided))
    };
    let alt = if two_sided {
        TestAlternative::Greater
    } else {
        options.alternative
    };
    let (stat, p) = permutation::exact(
        &[x.len(), y.len()],
        PermutationKind::Independent,
        options.with_alternative(alt),
        &callback,
    )?;
    result(
        stat,
        p,
        ClassicalMethod::Exact,
        TestSamplePolicy::IndependentGroups,
        [x.len(), y.len()],
    )
}
