//! Mood's median test for `k` independent samples.
use super::*;
use alloc::vec::Vec;

use scioxide_core::backend::try_with_capacity;

/// Selects how [`median_test_with_options`] counts observations equal to the pooled median.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MedianTies {
    /// Count them as below the median.
    #[default]
    Below,
    /// Count them as above the median.
    Above,
    /// Leave them out of the table.
    Ignore,
}

/// Result of [`median_test_with_options`]: the chi-square test, the pooled median and the
/// `2 × k` contingency table.
#[derive(Clone, Debug, PartialEq)]
pub struct MedianTestOutcome<S: RealScalar> {
    /// Pearson chi-square test of the `2 × k` table.
    pub test: DistributionTestResult<S>,
    /// Median of all observations pooled.
    pub grand_median: S,
    /// `[above, below]` counts of each group, after tie handling.
    pub table: Vec<[usize; 2]>,
}

/// Performs Mood's median test that `k` independent samples have the same
/// median.
///
/// Each observation is classified as above or below the median of all `N`
/// observations pooled, with ties at the median handled by `ties`. The statistic
/// is Pearson's chi-square of the resulting `2 × k` table, and the p-value its
/// upper tail with `k − 1` degrees of freedom. For two groups,
/// [`ClassicalOptions::continuity`] selects Yates's continuity correction.
/// The cost is O(N log N) time and O(N) memory.
///
/// # Errors
///
/// - `InvalidInput` if there are fewer than two groups or a group is empty.
/// - `InvalidInput` if `ties` removes every observation of a group, or every
///   observation is on one side of the median.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`] or the
///   alternative is not two-sided.
/// - `NonFinite` if a group contains NaN or infinity.
/// - `ResourceLimit` if the pooled copy exceeds
///   [`ClassicalOptions::max_workspace_bytes`].
///
/// # References
///
/// A. M. Mood, *Introduction to the Theory of Statistics*, McGraw-Hill, 1950.
pub fn median_test_with_options<S: RealScalar>(
    groups: &[&[S]],
    ties: MedianTies,
    options: ClassicalOptions,
) -> Result<MedianTestOutcome<S>, SciOxideError> {
    require_asymptotic(options)?;
    require_omnibus(options)?;
    if groups.len() < 2 {
        return Err(invalid(
            scioxide_core::argument!("groups"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "median test needs at least two groups",
        ));
    }
    let count = groups.iter().try_fold(0usize, |n, g| {
        n.checked_add(g.len()).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })
    })?;
    let bytes = count
        .checked_mul(core::mem::size_of::<S>())
        .and_then(|b| {
            groups
                .len()
                .checked_mul(core::mem::size_of::<[usize; 2]>())
                .and_then(|t| b.checked_add(t))
        })
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    crate::error_helpers::limit(
        bytes,
        options.max_workspace_bytes(),
        scioxide_core::quantity!("median_bytes"),
    )?;
    let mut pooled = try_with_capacity(
        count,
        scioxide_core::quantity!("pooled_median_observations"),
    )?;
    for &group in groups {
        sample(group, 1, scioxide_core::argument!("groups"))?;
        pooled.extend_from_slice(group);
    }
    crate::ranking::sort_finite(&mut pooled);
    let median = crate::empirical::quantile_sorted_by(
        count,
        crate::QuantileProbability::new(0.5)?,
        crate::QuantileMethod::Linear,
        |i| pooled[i],
    )?
    .value();
    let mut table = try_with_capacity(
        groups.len(),
        scioxide_core::quantity!("median_contingency_table"),
    )?;
    for &group in groups {
        let (mut above, mut below) = (0usize, 0usize);
        for &value in group {
            if value > median || (value == median && ties == MedianTies::Above) {
                above += 1;
            } else if value < median || (value == median && ties == MedianTies::Below) {
                below += 1;
            }
        }
        if above + below == 0 {
            return Err(invalid(
                scioxide_core::argument!("ties"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "group removed entirely by median tie policy",
            ));
        }
        table.push([above, below]);
    }
    let top = table.iter().map(|v| v[0]).sum::<usize>();
    let bottom = table.iter().map(|v| v[1]).sum::<usize>();
    if top == 0 || bottom == 0 {
        return Err(invalid(
            scioxide_core::argument!("groups"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "median contingency table has an empty row",
        ));
    }
    let kept = (top + bottom) as f64;
    let mut chi = 0.0;
    for counts in &table {
        let column = (counts[0] + counts[1]) as f64;
        for (observed, row) in [(counts[0], top), (counts[1], bottom)] {
            let expected = column * row as f64 / kept;
            let difference = (observed as f64 - expected).abs();
            let corrected = if options.continuity() == crate::ContinuityCorrection::HalfUnit
                && groups.len() == 2
            {
                (difference - 0.5).max(0.0)
            } else {
                difference
            };
            chi += corrected * corrected / expected;
        }
    }
    let test = result(
        chi,
        chi_tail(chi, (groups.len() - 1) as f64)?,
        DistributionMethod::Asymptotic,
        NullParameters::Exchangeable,
        top + bottom,
        groups.len(),
    )?;
    Ok(MedianTestOutcome {
        test,
        grand_median: median,
        table,
    })
}
/// Performs Mood's median test that `k` independent samples have the same median.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`median_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`median_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let groups = [&x[..], &y[..]];
/// let actual = median_test::<f64>(&groups)?;
/// let configured = median_test_with_options::<f64>(&groups, MedianTies::Below, ClassicalOptions::default())?;
/// assert!((actual.test.statistic() - configured.test.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. M. Mood, *Introduction to the Theory of Statistics*, McGraw-Hill, 1950.
pub fn median_test<S: RealScalar>(groups: &[&[S]]) -> Result<MedianTestOutcome<S>, SciOxideError> {
    median_test_with_options(groups, MedianTies::Below, Default::default()).map_err(|error| {
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
            scioxide_core::argument!("groups"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contingency_counts_and_tie_policy() {
        let x = [1.0_f64, 2.0, 3.0];
        let y = [4.0_f64, 5.0, 6.0];
        let r = median_test_with_options(
            &[&x, &y],
            MedianTies::Below,
            ClassicalOptions::default().with_continuity(crate::ContinuityCorrection::None),
        )
        .unwrap();
        assert_eq!(r.grand_median, 3.5);
        assert_eq!(r.table, alloc::vec![[0, 3], [3, 0]]);
        assert!((r.test.statistic - 6.0).abs() < 1e-15);
        assert!(r.test.p_value < 0.02);
        let tied = [3.5_f64, 3.5, 3.5];
        assert!(
            median_test_with_options(
                &[&tied, &tied],
                MedianTies::Ignore,
                ClassicalOptions::default().with_continuity(crate::ContinuityCorrection::None)
            )
            .is_err()
        );
    }
}
