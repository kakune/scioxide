//! Mood and Ansari–Bradley rank tests of equal scale in two samples.
use super::*;

use scioxide_core::backend::try_with_capacity;

fn score_tail(
    scores: &[f64],
    n: usize,
    observed: f64,
    options: ClassicalOptions,
) -> Result<(f64, DistributionMethod), SciOxideError> {
    let total = scores.len();
    let m = total - n;
    let expected = n as f64 * scores.iter().sum::<f64>() / total as f64;
    let count = scioxide_core::checked_binomial(total, n).ok();
    let exact = options.calibration() == TestCalibration::Exact
        || (options.calibration() == TestCalibration::Auto
            && count.is_some_and(|v| v <= options.max_work()));
    if exact {
        let count = count.ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })?;
        crate::error_helpers::work(
            count,
            options.max_work(),
            scioxide_core::BudgetKind::Node,
            "exact scale-rank assignments exceed options.max_work",
        )?;
        let bytes = n
            .checked_mul(core::mem::size_of::<usize>())
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })?;
        crate::error_helpers::limit(
            bytes,
            options.max_workspace_bytes(),
            scioxide_core::quantity!("scale_rank_bytes"),
        )?;
        let mut chosen = try_with_capacity(n, scioxide_core::quantity!("rank_assignments"))?;
        chosen.extend(0..n);
        let (mut lower, mut upper) = (0usize, 0usize);
        loop {
            let value = chosen.iter().map(|&index| scores[index]).sum::<f64>();
            let tolerance = 64.0 * f64::EPSILON * (1.0 + value.abs() + observed.abs());
            if value <= observed + tolerance {
                lower += 1;
            }
            if value + tolerance >= observed {
                upper += 1;
            }
            let mut index = n;
            while index > 0 {
                index -= 1;
                if chosen[index] < total - n + index {
                    break;
                }
            }
            if index == 0 && chosen[0] == total - n {
                break;
            }
            chosen[index] += 1;
            for next in index + 1..n {
                chosen[next] = chosen[next - 1] + 1;
            }
        }
        let p = match options.alternative() {
            TestAlternative::Less => lower as f64 / count as f64,
            TestAlternative::Greater => upper as f64 / count as f64,
            TestAlternative::TwoSided => (2.0 * (lower.min(upper)) as f64 / count as f64).min(1.0),
        };
        Ok((p, DistributionMethod::Exact))
    } else {
        let mean = scores.iter().sum::<f64>() / total as f64;
        let sumsq = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>();
        let variance = n as f64 * m as f64 / (total * (total - 1)) as f64 * sumsq;
        if variance <= 0.0 {
            return Err(invalid(
                scioxide_core::argument!("x"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "rank scale statistic has zero null variance",
            ));
        }
        let z = (observed - expected) / variance.sqrt();
        Ok((
            normal_tail(z, options.alternative())?,
            DistributionMethod::Asymptotic,
        ))
    }
}
/// Performs Mood's squared-rank test of equal scale in two independent samples.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`mood_scale_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`mood_scale_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = mood_scale_test::<f64>(&x, &y)?;
/// let configured = mood_scale_test_with_options::<f64>(&x, &y, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. M. Mood, "On the asymptotic efficiency of certain nonparametric
/// two-sample tests", *Ann. Math. Statist.* 25 (1954), 514–522.
pub fn mood_scale_test<S: RealScalar>(
    x: &[S],
    y: &[S],
) -> Result<DistributionTestResult<S>, SciOxideError> {
    mood_scale_test_with_options(x, y, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs Mood's squared-rank test of equal scale in two independent
/// samples.
///
/// Each pooled midrank `r` gets the score `(r − (N + 1)/2)²`, with `N = n + m`.
/// `M` is the sum of the scores of `x`. The reported statistic is
/// `z = (M − E M)/√(Var M)`, with the null moments of `M` computed from the
/// actual scores, so ties are handled. Positive `z` means `x` is more
/// dispersed, and the alternatives refer to the scale of `x` relative to `y`.
///
/// The exact p-value enumerates all `C(N, n)` label assignments. Under
/// [`TestCalibration::Auto`] it is used when `C(N, n)` is at most
/// [`ClassicalOptions::max_work`], and the normal approximation to `z`
/// otherwise. Ranking costs O(N log N) time and O(N) memory.
///
/// # Errors
///
/// - `InvalidInput` if either sample has fewer than two observations.
/// - `InvalidInput` if the scores have zero null variance.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
/// - `ConvergenceFailure` if the calibration is [`TestCalibration::Exact`] and
///   `C(N, n)` exceeds `max_work`.
/// - `ResourceLimit` if the pooled ranks exceed
///   [`ClassicalOptions::max_workspace_bytes`].
///
/// # References
///
/// A. M. Mood, "On the asymptotic efficiency of certain nonparametric
/// two-sample tests", *Ann. Math. Statist.* 25 (1954), 514–522.
pub fn mood_scale_test_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    let (ranks, labels, _) = super::two_sample::pooled_ranks(x, y, options.max_workspace_bytes())?;
    let center = (ranks.len() + 1) as f64 / 2.0;
    let mut scores = try_with_capacity(ranks.len(), scioxide_core::quantity!("mood_rank_scores"))?;
    scores.extend(ranks.iter().map(|r| (r - center).powi(2)));
    let observed = scores
        .iter()
        .zip(&labels)
        .filter_map(|(&s, &is_x)| is_x.then_some(s))
        .sum::<f64>();
    let average = scores.iter().sum::<f64>() / ranks.len() as f64;
    let mean = x.len() as f64 * average;
    let variance = x.len() as f64 * y.len() as f64 / (ranks.len() * (ranks.len() - 1)) as f64
        * scores.iter().map(|s| (s - average).powi(2)).sum::<f64>();
    if variance <= 0.0 {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Mood rank variance is zero",
        ));
    }
    let z = (observed - mean) / variance.sqrt();
    let (p, method) = score_tail(&scores, x.len(), observed, options)?;
    result(z, p, method, NullParameters::Exchangeable, ranks.len(), 2)
}
/// Performs the Ansari–Bradley test of equal scale in two independent samples.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`ansari_bradley_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`ansari_bradley_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = ansari_bradley_test::<f64>(&x, &y)?;
/// let configured = ansari_bradley_test_with_options::<f64>(&x, &y, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. R. Ansari and R. A. Bradley, "Rank-sum tests for dispersions", *Ann.
/// Math. Statist.* 31 (1960), 1174–1189.
pub fn ansari_bradley_test<S: RealScalar>(
    x: &[S],
    y: &[S],
) -> Result<DistributionTestResult<S>, SciOxideError> {
    ansari_bradley_test_with_options(x, y, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the Ansari–Bradley test of equal scale in two independent samples.
///
/// Each pooled midrank `r` gets the score `min(r, N + 1 − r)`, with `N = n + m`.
/// `AB` is the sum of the scores of `x`. The reported statistic is
/// `z = (E AB − AB)/√(Var AB)`, with the null moments computed from the actual
/// scores, so ties are handled. Positive `z` means `x` is more dispersed, and
/// the alternatives refer to the scale of `x` relative to `y`. The test assumes
/// the two distributions have the same median.
///
/// The exact p-value enumerates all `C(N, n)` label assignments. Under
/// [`TestCalibration::Auto`] it is used when `C(N, n)` is at most
/// [`ClassicalOptions::max_work`], and the normal approximation to `z`
/// otherwise. Ranking costs O(N log N) time and O(N) memory.
///
/// # Errors
///
/// - `InvalidInput` if either sample has fewer than two observations.
/// - `InvalidInput` if the scores have zero null variance.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
/// - `ConvergenceFailure` if the calibration is [`TestCalibration::Exact`] and
///   `C(N, n)` exceeds `max_work`.
/// - `ResourceLimit` if the pooled ranks exceed
///   [`ClassicalOptions::max_workspace_bytes`].
///
/// # References
///
/// A. R. Ansari and R. A. Bradley, "Rank-sum tests for dispersions", *Ann.
/// Math. Statist.* 31 (1960), 1174–1189.
pub fn ansari_bradley_test_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    let (ranks, labels, _) = super::two_sample::pooled_ranks(x, y, options.max_workspace_bytes())?;
    let total = ranks.len() as f64;
    let mut scores =
        try_with_capacity(ranks.len(), scioxide_core::quantity!("ansari_rank_scores"))?;
    scores.extend(ranks.iter().map(|&r| r.min(total + 1.0 - r)));
    let observed = scores
        .iter()
        .zip(&labels)
        .filter_map(|(&s, &is_x)| is_x.then_some(s))
        .sum::<f64>();
    let average = scores.iter().sum::<f64>() / total;
    let variance = x.len() as f64 * y.len() as f64 / (ranks.len() * (ranks.len() - 1)) as f64
        * scores.iter().map(|s| (s - average).powi(2)).sum::<f64>();
    if variance <= 0.0 {
        return Err(invalid(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Ansari rank variance is zero",
        ));
    }
    let z = (x.len() as f64 * average - observed) / variance.sqrt();
    // The score sum decreases as the first sample becomes more dispersed.
    let swapped = options.with_alternative(match options.alternative() {
        TestAlternative::Less => TestAlternative::Greater,
        TestAlternative::Greater => TestAlternative::Less,
        TestAlternative::TwoSided => TestAlternative::TwoSided,
    });
    let (p, method) = score_tail(&scores, x.len(), observed, swapped)?;
    result(z, p, method, NullParameters::Exchangeable, ranks.len(), 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn enumerated_four_point_scale_null() {
        let x = [1.0_f64, 4.0];
        let y = [2.0_f64, 3.0];
        let mood = mood_scale_test_with_options(&x, &y, ClassicalOptions::default()).unwrap();
        let ansari = ansari_bradley_test_with_options(&x, &y, ClassicalOptions::default()).unwrap();
        assert_eq!(mood.method, DistributionMethod::Exact);
        assert_eq!(ansari.method, DistributionMethod::Exact);
        assert!(mood.statistic > 0.0 && ansari.statistic > 0.0);
        assert!((mood.p_value - 1.0 / 3.0).abs() < 1e-15);
        assert!((ansari.p_value - 1.0 / 3.0).abs() < 1e-15);
    }
}
