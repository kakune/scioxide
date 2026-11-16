//! Tests of equal variances: Bartlett, Levene, Brown–Forsythe and Fligner–Killeen.
use super::*;
use crate::arithmetic::{center_difference, centered, native, shift, shifted_center, sum_mean};
use crate::empirical::select_adjacent_by;
use crate::group_summary::{GroupSummaries, GroupSummary};
use crate::ranking::{RankPool, finite_cmp};
use alloc::vec::Vec;
use core::cmp::Ordering;
use scioxide_core::{CompensatedSum, Scaled, ScaledAccumulator, compensated_reduce};

use scioxide_core::backend::try_with_capacity;
use scioxide_random::Quantile;

/// Selects the group center subtracted in the Levene and Fligner–Killeen tests.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum VarianceCenter {
    /// Group mean: Levene's original test. Suited to symmetric, moderate-tailed
    /// distributions.
    Mean,
    /// Group median: the Brown–Forsythe test. This is the default and is robust
    /// to skewed distributions.
    #[default]
    Median,
    /// Trimmed group mean: `⌊p n⌋` observations are removed from each tail of a
    /// group, with `p = proportion`. The removed observations are dropped from the
    /// test, not only from the center: the deviations and the group size
    /// `n − 2⌊p n⌋` use the retained observations alone.
    Trimmed {
        /// Fraction removed from each tail, in `[0, 0.5)`.
        proportion: f64,
    },
}
enum Deviations {
    Native(Vec<Vec<f64>>),
    Extended(Vec<Vec<Scaled<f64>>>),
}

fn validate_groups<S: RealScalar>(
    groups: &[&[S]],
    center: VarianceCenter,
) -> Result<bool, SciOxideError> {
    if groups.len() < 2 {
        return Err(invalid(
            scioxide_core::argument!("groups"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "at least two independent groups required",
        ));
    }
    let count = groups.iter().try_fold(0usize, |n, g| {
        n.checked_add(g.len()).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })
    })?;
    if count as u128 > (1u128 << 53) {
        return Err(invalid(
            scioxide_core::argument!("groups"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "group count is not exactly representable",
        ));
    }
    if let VarianceCenter::Trimmed { proportion } = center
        && (!proportion.is_finite() || !(0.0..0.5).contains(&proportion))
    {
        return Err(invalid(
            scioxide_core::argument!("center"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "invalid trim proportion",
        ));
    }
    let mut ordinary = true;
    for &group in groups {
        sample(group, 2, scioxide_core::argument!("groups"))?;
        if group.len() - 2 * trim_count(group.len(), center) < 2 {
            return Err(invalid(
                scioxide_core::argument!("center"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "trim leaves fewer than two observations",
            ));
        }
        ordinary &= group.iter().try_fold(true, |admitted, &value| {
            Ok::<_, SciOxideError>(admitted & native(value.try_to_f64_exact()?, 64))
        })?;
    }
    Ok(ordinary)
}

fn group_data<S: RealScalar, T>(
    groups: &[&[S]],
    convert: impl Fn(S) -> Result<T, SciOxideError>,
) -> Result<Vec<Vec<T>>, SciOxideError> {
    let mut rows = try_with_capacity(groups.len(), scioxide_core::quantity!("variance_groups"))?;
    for &group in groups {
        let mut row = try_with_capacity(
            group.len(),
            scioxide_core::quantity!("variance_group_observations"),
        )?;
        for &value in group {
            row.push(convert(value)?);
        }
        rows.push(row);
    }
    Ok(rows)
}

fn trim_count(n: usize, center: VarianceCenter) -> usize {
    match center {
        VarianceCenter::Trimmed { proportion } => (proportion * n as f64).floor() as usize,
        _ => 0,
    }
}

// Both arithmetic paths select the retained middle without sorting either tail.
fn trim<T>(row: &mut Vec<T>, cut: usize, compare: impl Fn(&T, &T) -> Ordering) {
    if cut != 0 {
        let end = row.len() - cut;
        row.select_nth_unstable_by(end, &compare);
        row.truncate(end);
        row.select_nth_unstable_by(cut, compare);
        row.drain(..cut);
    }
}

fn median<T: Copy>(
    row: &mut [T],
    compare: impl Fn(&T, &T) -> Ordering,
) -> Result<[T; 2], SciOxideError> {
    let n = row.len();
    select_adjacent_by(row, (n - 1) / 2, n / 2, compare)
}

fn native_deviations<S: RealScalar>(
    groups: &[&[S]],
    center: VarianceCenter,
) -> Result<Vec<Vec<f64>>, SciOxideError> {
    let mut rows = group_data(groups, S::try_to_f64_exact)?;
    for row in &mut rows {
        let cut = trim_count(row.len(), center);
        trim(row, cut, finite_cmp);
        let c = if center == VarianceCenter::Median {
            let [lower, upper] = median(row, finite_cmp)?;
            // Admission keeps both halvings exact. Keep the midpoint residual
            // until after centering, so symmetric deviations remain tied.
            let (high, low) = scioxide_core::backend::two_sum(lower * 0.5, upper * 0.5);
            [high, low]
        } else {
            let origin = row[0];
            shifted_center(
                origin,
                compensated_reduce(row.iter().map(|&value| shift(value, origin)))?,
                row.len() as f64,
            )?
        };
        for value in row {
            *value = centered(*value, c)?.abs();
        }
    }
    Ok(rows)
}

fn extended_deviations<S: RealScalar>(
    groups: &[&[S]],
    center: VarianceCenter,
) -> Result<Vec<Vec<Scaled<f64>>>, SciOxideError> {
    let mut rows = group_data(groups, |value| {
        Scaled::from_value(value.try_to_f64_exact()?)
    })?;
    for row in &mut rows {
        let compare = |a: &Scaled<f64>, b: &Scaled<f64>| a.cmp_value(*b);
        let cut = trim_count(row.len(), center);
        trim(row, cut, compare);
        let c = if center == VarianceCenter::Median {
            let [lower, upper] = median(row, compare)?;
            crate::arithmetic::interpolate_scaled(lower, upper, Scaled::HALF)?
        } else {
            GroupSummary::from_scaled(row)?.mean
        };
        for value in row {
            *value = center_difference(c, ScaledAccumulator::from_scaled(*value))?.abs();
        }
    }
    Ok(rows)
}

fn deviations<S: RealScalar>(
    groups: &[&[S]],
    center: VarianceCenter,
) -> Result<Deviations, SciOxideError> {
    if validate_groups(groups, center)? {
        native_deviations(groups, center).map(Deviations::Native)
    } else {
        extended_deviations(groups, center).map(Deviations::Extended)
    }
}

fn levene_statistic<T>(
    rows: Vec<Vec<T>>,
    max_bytes: usize,
    summarize: impl Fn(&[T]) -> Result<GroupSummary, SciOxideError>,
) -> Result<(Scaled<f64>, usize), SciOxideError> {
    let count = rows.iter().map(Vec::len).sum::<usize>();
    let summaries = GroupSummaries::from_summaries(
        rows.len(),
        count,
        max_bytes,
        rows.iter().map(|row| summarize(row)),
    )?;
    let (f, _) = summaries.anova()?;
    Ok((f, count))
}
/// Performs Bartlett's test of equal variances in `k` independent normal samples.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`bartlett_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`bartlett_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let groups = [&x[..], &y[..]];
/// let actual = bartlett_test::<f64>(&groups)?;
/// let configured = bartlett_test_with_options::<f64>(&groups, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. S. Bartlett, "Properties of sufficiency and statistical tests", *Proc.
/// Roy. Soc. London Ser. A* 160 (1937), 268–282.
pub fn bartlett_test<S: RealScalar>(
    groups: &[&[S]],
) -> Result<DistributionTestResult<S>, SciOxideError> {
    bartlett_test_with_options(groups, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("groups"),
        )
    })
}
/// Performs Bartlett's test of equal variances in `k` independent normal
/// samples.
///
/// ```text
/// B = ((N − k) ln sₚ² − Σᵢ (nᵢ − 1) ln sᵢ²) / C,
/// C = 1 + (Σᵢ 1/(nᵢ − 1) − 1/(N − k)) / (3(k − 1)),
/// ```
///
/// where `nᵢ` and `sᵢ²` are the size and unbiased variance of group `i`,
/// `N = Σ nᵢ`, and `sₚ²` is the pooled variance. The p-value is the upper tail
/// of the chi-square distribution with `k − 1` degrees of freedom. The test is
/// sensitive to non-normality; [`levene_test_with_options`] and [`fligner_test_with_options`] are robust
/// alternatives. Variances are accumulated in extended range. The cost is O(N)
/// time and O(k) memory.
///
/// # Errors
///
/// - `InvalidInput` if there are fewer than two groups, or a group has fewer
///   than two observations or zero variance.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`] or the
///   alternative is not two-sided.
/// - `NonFinite` if a group contains NaN or infinity.
/// - `ResourceLimit` if the group summaries exceed
///   [`ClassicalOptions::max_workspace_bytes`].
///
/// # References
///
/// M. S. Bartlett, "Properties of sufficiency and statistical tests", *Proc.
/// Roy. Soc. London Ser. A* 160 (1937), 268–282.
pub fn bartlett_test_with_options<S: RealScalar>(
    groups: &[&[S]],
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    require_asymptotic(options)?;
    require_omnibus(options)?;
    let summaries = GroupSummaries::prepare(groups, options.max_workspace_bytes())?;
    summaries.positive_variances()?;
    let k = summaries.groups.len() as f64;
    let (pooled, total_df) = summaries.pooled()?;
    let mut numerator = ScaledAccumulator::ZERO;
    let mut log_magnitude = 0.0;
    let mut reciprocal_degrees = 0.0;
    for group in &summaries.groups {
        // Compare dimensionless variances, avoiding cancellation of large logs.
        let logarithm = crate::arithmetic::scaled_log_ratio(pooled, group.variance)?;
        let term = (group.n - 1.0) * logarithm;
        numerator.add_value(term)?;
        log_magnitude += term.abs();
        reciprocal_degrees += 1.0 / (group.n - 1.0);
    }
    let numerator = numerator.total()?.round()?;
    if numerator < -64.0 * f64::EPSILON * (1.0 + log_magnitude) {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Cancellation,
            "negative Bartlett numerator",
        ));
    }
    let correction = 1.0 + (reciprocal_degrees - 1.0 / total_df) / (3.0 * (k - 1.0));
    let statistic = scioxide_core::backend::checked_final_rounding(
        numerator.max(0.0) / correction,
        "unrepresentable Bartlett statistic",
    )?;
    result(
        statistic,
        chi_tail(statistic, k - 1.0)?,
        DistributionMethod::Asymptotic,
        NullParameters::Exchangeable,
        summaries.total,
        groups.len(),
    )
}
/// Performs Levene's test of equal variances in `k` independent samples.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`levene_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`levene_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let groups = [&x[..], &y[..]];
/// let actual = levene_test::<f64>(&groups)?;
/// let configured = levene_test_with_options::<f64>(&groups, VarianceCenter::Median, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// H. Levene, "Robust tests for equality of variances", in I. Olkin (ed.),
/// *Contributions to Probability and Statistics*, Stanford University Press,
/// 1960, 278–292.
///
/// M. B. Brown and A. B. Forsythe, "Robust tests for the equality of
/// variances", *J. Amer. Statist. Assoc.* 69 (1974), 364–367.
pub fn levene_test<S: RealScalar>(
    groups: &[&[S]],
) -> Result<DistributionTestResult<S>, SciOxideError> {
    levene_test_with_options(groups, VarianceCenter::Median, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options", "center"],
            scioxide_core::argument!("groups"),
        )
    })
}
/// Performs Levene's test of equal variances in `k` independent samples.
///
/// The test is the one-way analysis of variance F test applied to the absolute
/// deviations `|xᵢⱼ − cᵢ|`, where `cᵢ` is the center of group `i` selected by
/// `center`. With the median it is the Brown–Forsythe test. The p-value is the
/// upper tail of the F distribution with `k − 1` and `N − k` degrees of
/// freedom, where `N` is the total size. Medians and trimmed means use
/// linear-time selection, so the cost is O(N) time and O(N) memory.
/// Deviations switch to extended range when they would overflow or underflow
/// in `f64`.
///
/// # Errors
///
/// - `InvalidInput` if there are fewer than two groups or a group has fewer
///   than two observations, also after trimming.
/// - `InvalidInput` if a trim proportion is outside `[0, 0.5)`.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`] or the
///   alternative is not two-sided.
/// - `NonFinite` if a group contains NaN or infinity.
/// - `ResourceLimit` if the deviations exceed
///   [`ClassicalOptions::max_workspace_bytes`].
/// - An error if the absolute deviations have no within-group variation.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{ClassicalOptions, VarianceCenter, levene_test_with_options};
///
/// // Brown–Forsythe test of two groups with similar spread.
/// let groups = [&[1.0_f64, 2.0, 4.0, 8.0][..], &[2.0, 3.0, 5.0, 6.0]];
/// let test = levene_test_with_options(&groups, VarianceCenter::Median, ClassicalOptions::default())?;
/// assert!((0.0..=1.0).contains(&test.p_value()));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// H. Levene, "Robust tests for equality of variances", in I. Olkin (ed.),
/// *Contributions to Probability and Statistics*, Stanford University Press,
/// 1960, 278–292.
///
/// M. B. Brown and A. B. Forsythe, "Robust tests for the equality of
/// variances", *J. Amer. Statist. Assoc.* 69 (1974), 364–367.
pub fn levene_test_with_options<S: RealScalar>(
    groups: &[&[S]],
    center: VarianceCenter,
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    require_asymptotic(options)?;
    require_omnibus(options)?;
    let count = groups.iter().try_fold(0usize, |total, g| {
        total.checked_add(g.len()).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })
    })?;
    let bytes = count
        .checked_mul(core::mem::size_of::<Scaled<f64>>())
        .and_then(|b| {
            groups
                .len()
                .checked_mul(
                    core::mem::size_of::<Vec<Scaled<f64>>>() + core::mem::size_of::<GroupSummary>(),
                )
                .and_then(|headers| b.checked_add(headers))
        })
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    crate::error_helpers::limit(
        bytes,
        options.max_workspace_bytes(),
        scioxide_core::quantity!("variance_bytes"),
    )?;
    let (f, count) = match deviations(groups, center)? {
        Deviations::Native(rows) => levene_statistic(
            rows,
            options.max_workspace_bytes(),
            GroupSummary::from_native,
        )?,
        Deviations::Extended(rows) => levene_statistic(
            rows,
            options.max_workspace_bytes(),
            GroupSummary::from_scaled,
        )?,
    };
    let k = groups.len();
    f.round_to::<S>()?;
    let f = f.round()?;
    result(
        f,
        f_tail(f, (k - 1) as f64, (count - k) as f64)?,
        DistributionMethod::Asymptotic,
        NullParameters::Exchangeable,
        count,
        k,
    )
}
/// Performs the Fligner–Killeen test of equal variances in `k` independent samples.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`fligner_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`fligner_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let groups = [&x[..], &y[..]];
/// let actual = fligner_test::<f64>(&groups)?;
/// let configured = fligner_test_with_options::<f64>(&groups, VarianceCenter::Median, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. A. Fligner and T. J. Killeen, "Distribution-free two-sample tests for
/// scale", *J. Amer. Statist. Assoc.* 71 (1976), 210–213.
///
/// W. J. Conover, M. E. Johnson and M. M. Johnson, "A comparative study of
/// tests for homogeneity of variances, with applications to the outer
/// continental shelf bidding data", *Technometrics* 23 (1981), 351–361.
pub fn fligner_test<S: RealScalar>(
    groups: &[&[S]],
) -> Result<DistributionTestResult<S>, SciOxideError> {
    fligner_test_with_options(groups, VarianceCenter::Median, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options", "center"],
            scioxide_core::argument!("groups"),
        )
    })
}
/// Performs the Fligner–Killeen test of equal variances in `k` independent
/// samples.
///
/// The absolute deviations `|xᵢⱼ − cᵢ|` from the group centers are ranked over
/// all groups with midranks `r`, and each is replaced by the normal score
/// `a = Φ⁻¹(1/2 + r/(2(N + 1)))`. The statistic is
///
/// ```text
/// X² = Σᵢ nᵢ (āᵢ − ā)² / s²,
/// ```
///
/// where `āᵢ` is the mean score of group `i`, `ā` and `s²` the mean and
/// variance of all `N` scores, and `nᵢ` the group size. The p-value is the upper
/// tail of the chi-square distribution with `k − 1` degrees of freedom. The test
/// is robust to non-normality. The cost is O(N log N) time and O(N) memory.
///
/// # Errors
///
/// - `InvalidInput` if there are fewer than two groups or a group has fewer
///   than two observations, also after trimming.
/// - `InvalidInput` if a trim proportion is outside `[0, 0.5)`.
/// - `InvalidInput` if every absolute deviation is equal.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`] or the
///   alternative is not two-sided.
/// - `NonFinite` if a group contains NaN or infinity.
/// - `ResourceLimit` if the deviations and scores exceed
///   [`ClassicalOptions::max_workspace_bytes`].
///
/// # Examples
///
/// ```
/// use scioxide_stats::{ClassicalOptions, VarianceCenter, fligner_test_with_options};
///
/// // Median-centered Fligner–Killeen test of two groups with similar spread.
/// let groups = [&[1.0_f64, 2.0, 4.0, 8.0][..], &[2.0, 3.0, 5.0, 6.0]];
/// let test = fligner_test_with_options(&groups, VarianceCenter::Median, ClassicalOptions::default())?;
/// assert!((0.0..=1.0).contains(&test.p_value()));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. A. Fligner and T. J. Killeen, "Distribution-free two-sample tests for
/// scale", *J. Amer. Statist. Assoc.* 71 (1976), 210–213.
///
/// W. J. Conover, M. E. Johnson and M. M. Johnson, "A comparative study of
/// tests for homogeneity of variances, with applications to the outer
/// continental shelf bidding data", *Technometrics* 23 (1981), 351–361.
pub fn fligner_test_with_options<S: RealScalar>(
    groups: &[&[S]],
    center: VarianceCenter,
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    require_asymptotic(options)?;
    require_omnibus(options)?;
    let requested = groups.iter().try_fold(0usize, |total, g| {
        total.checked_add(g.len()).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })
    })?;
    let requested_bytes = requested
        .checked_mul(
            core::mem::size_of::<Scaled<f64>>()
                + core::mem::size_of::<f64>()
                + core::mem::size_of::<(Scaled<f64>, usize)>(),
        )
        .and_then(|b| {
            groups
                .len()
                .checked_mul(core::mem::size_of::<Vec<Scaled<f64>>>() + core::mem::size_of::<f64>())
                .and_then(|headers| b.checked_add(headers))
        })
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    crate::error_helpers::limit(
        requested_bytes,
        options.max_workspace_bytes(),
        scioxide_core::quantity!("variance_bytes"),
    )?;
    let (statistic, n) = match deviations(groups, center)? {
        Deviations::Native(rows) => fligner_statistic(rows, finite_cmp)?,
        Deviations::Extended(rows) => fligner_statistic(rows, |a, b| a.cmp_value(*b))?,
    };
    result(
        statistic,
        chi_tail(statistic, (groups.len() - 1) as f64)?,
        DistributionMethod::Asymptotic,
        NullParameters::Exchangeable,
        n,
        groups.len(),
    )
}

fn fligner_statistic<T: PartialEq>(
    rows: Vec<Vec<T>>,
    compare: impl Fn(&T, &T) -> Ordering,
) -> Result<(f64, usize), SciOxideError> {
    let n = rows.iter().map(Vec::len).sum::<usize>();
    let k = rows.len();
    let mut counts = try_with_capacity(k, scioxide_core::quantity!("fligner_group_counts"))?;
    let mut ranked = try_with_capacity(n, scioxide_core::quantity!("fligner_ranks"))?;
    for (group, row) in rows.into_iter().enumerate() {
        counts.push(row.len());
        ranked.extend(row.into_iter().map(|value| (value, group)));
    }
    let ranked = RankPool::sorted_by(ranked, compare);
    // Drop the observation rows before reserving scores and compensated group
    // sums. These phases fit the original all-live scaled-width byte bound.
    let mut scores = try_with_capacity(n, scioxide_core::quantity!("fligner_scores"))?;
    let mut sums = scioxide_core::backend::try_filled(
        k,
        CompensatedSum::ZERO,
        scioxide_core::quantity!("fligner_group_scores"),
    )?;
    let normal = Normal::new(0.0, 1.0)?;
    for (run, members) in ranked.runs() {
        let score = normal.quantile(0.5 + run.midrank()? / (2.0 * (n + 1) as f64))?;
        for (_, group) in members {
            sums[*group].add(score)?;
            scores.push(score);
        }
    }
    let [total] = compensated_reduce(scores.iter().map(|&score| [score]))?;
    let grand = sum_mean(total.parts(), n as f64);
    let [squares] = compensated_reduce(scores.iter().map(|&score| [(score - grand).powi(2)]))?;
    let variance = squares.total()? / (n - 1) as f64;
    if variance <= 0.0 {
        return Err(invalid(
            scioxide_core::argument!("groups"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Fligner statistic undefined for identical deviations",
        ));
    }
    let [between] = compensated_reduce(sums.iter().zip(&counts).map(|(&sum, &count)| {
        [count as f64 * (sum_mean(sum.parts(), count as f64) - grand).powi(2)]
    }))?;
    Ok((finite(between.total()? / variance)?, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_and_unequal_scales() {
        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = [11.0_f64, 12.0, 13.0, 14.0, 15.0];
        let equal = [&x[..], &y[..]];
        let b = bartlett_test_with_options(&equal, ClassicalOptions::default()).unwrap();
        assert!(b.statistic.abs() < 1e-12);
        assert!((b.p_value - 1.0).abs() < 1e-12);
        let wide = [0.0_f64, 4.0, 8.0, 12.0, 16.0];
        let unequal = [&x[..], &wide[..]];
        assert!(
            bartlett_test_with_options(&unequal, ClassicalOptions::default())
                .unwrap()
                .p_value
                < 0.05
        );
        assert!(
            levene_test_with_options(
                &unequal,
                VarianceCenter::Median,
                ClassicalOptions::default()
            )
            .unwrap()
            .statistic
                > 0.0
        );
        assert!(
            fligner_test_with_options(
                &unequal,
                VarianceCenter::Median,
                ClassicalOptions::default()
            )
            .unwrap()
            .statistic
                > 0.0
        );
    }

    #[test]
    fn rejects_collapse_and_workspace_limit() {
        let x = [1.0_f32; 3];
        let groups = [&x[..], &x[..]];
        assert!(bartlett_test_with_options(&groups, ClassicalOptions::default()).is_err());
        assert!(
            levene_test_with_options(&groups, VarianceCenter::Median, ClassicalOptions::default())
                .is_err()
        );
        let options = ClassicalOptions::default().with_max_workspace_bytes(8);
        assert!(fligner_test_with_options(&groups, VarianceCenter::Median, options).is_err());
    }

    #[test]
    fn independent_three_group_reference() {
        let a = [
            8.88_f64, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99,
        ];
        let b = [
            8.88_f64, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05,
        ];
        let c = [
            8.95_f64, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98,
        ];
        let groups = [&a[..], &b[..], &c[..]];
        let options = ClassicalOptions::default();
        let bartlett = bartlett_test_with_options(&groups, options).unwrap();
        let levene = levene_test_with_options(&groups, VarianceCenter::Median, options).unwrap();
        let fligner = fligner_test_with_options(&groups, VarianceCenter::Median, options).unwrap();
        assert!((bartlett.p_value - 1.1254782518834628e-5).abs() < 1e-10);
        assert!((levene.p_value - 0.002431505967249681).abs() < 1e-9);
        // Independent rational median/deviation ordering of the binary64 inputs,
        // followed by normal scores and the df=2 chi-square tail. Rounding the
        // median first spuriously splits symmetric ties in the first group.
        assert!(
            (fligner.p_value - 0.0045468894265999725).abs() < 1e-9,
            "{}",
            fligner.p_value
        );
    }

    fn scale_family<S: RealScalar>(exponents: &[i32], translated: (i32, i32)) {
        let a = [-4., -2., -1., 0., 0., 1., 2., 4.];
        let b = [-6., -4., -2., 0., 0., 2., 4., 6.];
        let c = [-8., -3., -1., 0., 1., 2., 5., 8.];
        let options = ClassicalOptions::default();
        let ordinary = [&a[..], &b, &c];
        let bartlett = bartlett_test_with_options(&ordinary, options).unwrap();
        let mut frames = exponents
            .iter()
            .map(|&e| (S::ONE.scalbn(e), S::ZERO))
            .collect::<Vec<_>>();
        frames.push((S::ONE.scalbn(translated.1), S::ONE.scalbn(translated.0)));
        let tolerance = if core::mem::size_of::<S>() == 4 {
            3e-6
        } else {
            3e-12
        };
        for (scale, shift) in frames {
            for sign in [S::ONE, -S::ONE] {
                let transform = |values: [f64; 8]| {
                    values.map(|v| shift + sign * scale * S::try_from_f64_lossy(v).unwrap())
                };
                let (x, y, z) = (transform(a), transform(b), transform(c));
                let groups = [&x[..], &y, &z];
                let result = bartlett_test_with_options(&groups, options).unwrap();
                assert!((result.statistic.into_f64() - bartlett.statistic).abs() < tolerance);
                assert!((result.p_value - bartlett.p_value).abs() < tolerance);
                for center in [
                    VarianceCenter::Mean,
                    VarianceCenter::Median,
                    VarianceCenter::Trimmed { proportion: 0.125 },
                ] {
                    for (actual, expected) in [
                        (
                            levene_test_with_options(&groups, center, options).unwrap(),
                            levene_test_with_options(&ordinary, center, options).unwrap(),
                        ),
                        (
                            fligner_test_with_options(&groups, center, options).unwrap(),
                            fligner_test_with_options(&ordinary, center, options).unwrap(),
                        ),
                    ] {
                        assert!(
                            (actual.statistic.into_f64() - expected.statistic).abs() < tolerance,
                            "stat {} != {}, scale {}",
                            actual.statistic.into_f64(),
                            expected.statistic,
                            scale.into_f64()
                        );
                        assert!((actual.p_value - expected.p_value).abs() < tolerance);
                        assert_eq!(actual.observations, expected.observations);
                        assert_eq!(actual.groups, expected.groups);
                    }
                }
            }
        }
    }

    #[test]
    fn variance_tests_preserve_units_translation_and_subnormal_spreads() {
        scale_family::<f64>(&[-1074, -700, 0, 510, 1000], (1000, 950));
        scale_family::<f32>(&[-149, -80, 0, 65, 120], (100, 80));
    }

    #[test]
    fn deviations_may_exceed_the_scalar_range() {
        let a = [-f64::MAX, f64::MAX, f64::MAX, f64::MAX];
        let b = [-f64::MAX, -f64::MAX, f64::MAX, f64::MAX];
        let ordinary = [&[-1., 1., 1., 1.][..], &[-1., -1., 1., 1.]];
        let options = ClassicalOptions::default();
        for center in [
            VarianceCenter::Mean,
            VarianceCenter::Median,
            VarianceCenter::Trimmed { proportion: 0.0 },
        ] {
            for (actual, expected) in [
                (
                    levene_test_with_options(&[&a, &b], center, options).unwrap(),
                    levene_test_with_options(&ordinary, center, options).unwrap(),
                ),
                (
                    fligner_test_with_options(&[&a, &b], center, options).unwrap(),
                    fligner_test_with_options(&ordinary, center, options).unwrap(),
                ),
            ] {
                assert!((actual.statistic - expected.statistic).abs() < 1e-12);
                assert!((actual.p_value - expected.p_value).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn scaled_workspace_limits_include_all_live_buffers() {
        let a = [-4., -2., -1., 0., 0., 1., 2., 4.];
        let b = [-6., -4., -2., 0., 0., 2., 4., 6.];
        let groups = [&a[..], &b];
        let summaries = 2 * core::mem::size_of::<GroupSummary>();
        let rows =
            16 * core::mem::size_of::<Scaled<f64>>() + 2 * core::mem::size_of::<Vec<Scaled<f64>>>();
        let fligner = rows
            + 16 * (core::mem::size_of::<(Scaled<f64>, usize)>() + core::mem::size_of::<f64>())
            + 2 * core::mem::size_of::<f64>();
        for (test, bytes) in [(0, summaries), (1, rows + summaries), (2, fligner)] {
            for allowed in [bytes - 1, bytes] {
                let options = ClassicalOptions::default().with_max_workspace_bytes(allowed);
                let result = match test {
                    0 => bartlett_test_with_options(&groups, options),
                    1 => levene_test_with_options(&groups, VarianceCenter::Median, options),
                    _ => fligner_test_with_options(&groups, VarianceCenter::Median, options),
                };
                assert_eq!(result.is_ok(), allowed == bytes);
            }
        }
    }

    #[test]
    fn independent_references_cover_centers_and_unequal_group_lengths() {
        let a = [16.0_f64, 0., 2., 2., 3., -4., 11.];
        let b = [17., -0., 4., 4., 6., -8., 2., 10.];
        let c = [15., 0., 3., 1., 1., -2., 7.];
        let groups = [&a[..], &b, &c];
        // SciPy 1.18.1. The trimmed references explicitly trimboth(0.25)
        // before the mean-centered tests, matching our retained-sample policy.
        // Odd/even medians, duplicates and unequal retained group counts.
        for (center, n, levene, fligner) in [
            (
                VarianceCenter::Mean,
                22,
                (0.10983380808536199, 0.8965477357739058),
                (0.18670644007979587, 0.9108717053271728),
            ),
            (
                VarianceCenter::Median,
                22,
                (0.09990337386262982, 0.9053970260712738),
                (0.07808032755254747, 0.961712082381757),
            ),
            (
                VarianceCenter::Trimmed { proportion: 0.25 },
                14,
                (1.0929768555466879, 0.3690169039404783),
                (1.5654400822252086, 0.45716082235849587),
            ),
            (
                VarianceCenter::Trimmed { proportion: 0.0 },
                22,
                (0.10983380808536199, 0.8965477357739058),
                (0.18670644007979587, 0.9108717053271728),
            ),
        ] {
            let options = ClassicalOptions::default();
            for (actual, expected) in [
                (
                    levene_test_with_options(&groups, center, options).unwrap(),
                    levene,
                ),
                (
                    fligner_test_with_options(&groups, center, options).unwrap(),
                    fligner,
                ),
            ] {
                assert!(
                    (actual.statistic - expected.0).abs() < 2e-12,
                    "{center:?}: {actual:?} != {expected:?}"
                );
                assert!(
                    (actual.p_value - expected.1).abs() < 2e-12,
                    "{center:?}: {actual:?} != {expected:?}"
                );
                assert_eq!(actual.observations, n);
                assert_eq!(actual.groups, 3);
            }
            let native = native_deviations(&groups, center).unwrap();
            let extended = extended_deviations(&groups, center).unwrap();
            let (native_f, _) =
                levene_statistic(native, usize::MAX, GroupSummary::from_native).unwrap();
            let (extended_f, _) =
                levene_statistic(extended, usize::MAX, GroupSummary::from_scaled).unwrap();
            assert!((native_f.round().unwrap() - extended_f.round().unwrap()).abs() < 2e-12);
            let (native_f, _) =
                fligner_statistic(native_deviations(&groups, center).unwrap(), finite_cmp).unwrap();
            let (extended_f, _) =
                fligner_statistic(extended_deviations(&groups, center).unwrap(), |a, b| {
                    a.cmp_value(*b)
                })
                .unwrap();
            assert!((native_f - extended_f).abs() < 2e-12);
        }
    }

    #[test]
    fn native_guard_boundaries_preserve_scale_inference() {
        let a = [1., 1.125, 1.25, 1.5, 1.75, 1.875];
        let b = [1., 1.25, 1.5, 1.625, 1.75, 1.875];
        let options = ClassicalOptions::default();
        for exponent in [-65, -64, 64, 65] {
            let scale = 1.0_f64.scalbn(exponent);
            let x = a.map(|v| v * scale);
            let y = b.map(|v| v * scale);
            for center in [
                VarianceCenter::Mean,
                VarianceCenter::Median,
                VarianceCenter::Trimmed { proportion: 0.2 },
            ] {
                let groups = [&x[..], &y];
                assert_eq!(
                    matches!(deviations(&groups, center).unwrap(), Deviations::Native(_)),
                    (-64..=64).contains(&exponent),
                );
                for (actual, expected) in [
                    (
                        levene_test_with_options(&groups, center, options).unwrap(),
                        levene_test_with_options(&[&a, &b], center, options).unwrap(),
                    ),
                    (
                        fligner_test_with_options(&groups, center, options).unwrap(),
                        fligner_test_with_options(&[&a, &b], center, options).unwrap(),
                    ),
                ] {
                    assert!((actual.statistic - expected.statistic).abs() < 2e-12);
                    assert!((actual.p_value - expected.p_value).abs() < 2e-12);
                }
            }
        }
    }

    #[test]
    fn scale_tests_reject_invalid_inputs_and_collapsed_deviations() {
        use scioxide_core::SciOxideErrorKind;
        let a = [0., 1., 3., 8.];
        let b = [-2., 2., 4., 9.];
        let options = ClassicalOptions::default();
        for groups in [
            &[][..],
            &[&a[..]][..],
            &[&a[..], &[]][..],
            &[&a[..], &[1.]][..],
        ] {
            assert_eq!(
                levene_test_with_options(groups, VarianceCenter::Median, options)
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::InvalidInput,
            );
            assert_eq!(
                fligner_test_with_options(groups, VarianceCenter::Median, options)
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::InvalidInput,
            );
        }
        for proportion in [-0.1, 0.5, 1.0, f64::NAN, f64::INFINITY, 0.49] {
            let center = VarianceCenter::Trimmed { proportion };
            // At 0.49 the odd group retains only its median.
            let groups = [&a[..], &[0., 1., 2.]];
            assert_eq!(
                levene_test_with_options(&groups, center, options)
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::InvalidInput
            );
            assert_eq!(
                fligner_test_with_options(&groups, center, options)
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::InvalidInput
            );
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let groups = [&a[..], &[0., bad, 2.]];
            assert_eq!(
                levene_test_with_options(&groups, VarianceCenter::Median, options)
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::NonFinite
            );
            assert_eq!(
                fligner_test_with_options(&groups, VarianceCenter::Median, options)
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::NonFinite
            );
        }
        let collapsed = [&[-0.0, 0.0][..], &[-0.0, 0.0]];
        for center in [
            VarianceCenter::Mean,
            VarianceCenter::Median,
            VarianceCenter::Trimmed { proportion: 0.0 },
        ] {
            assert!(levene_test_with_options(&collapsed, center, options).is_err());
            assert!(fligner_test_with_options(&collapsed, center, options).is_err());
            for invalid_options in [
                options.with_calibration(TestCalibration::Exact),
                options.with_alternative(TestAlternative::Less),
            ] {
                assert!(levene_test_with_options(&[&a, &b], center, invalid_options).is_err());
                assert!(fligner_test_with_options(&[&a, &b], center, invalid_options).is_err());
            }
        }
    }
}
