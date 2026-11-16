//! Page's trend test for ordered alternatives in complete blocks.
use super::*;
use crate::{PermutationKind, RankMethod};
use scioxide_core::backend::try_filled;
use scioxide_linalg::MatrixView;

/// Computes the result of [`page_trend_test_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`page_trend_test_with_options`] for the fixed default settings.
pub fn page_trend_test<S: RealScalar>(
    data: MatrixView<'_, S>,
    treatment_order: &[usize],
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    page_trend_test_with_options(
        data,
        treatment_order,
        ClassicalOptions::default().with_alternative(TestAlternative::Greater),
    )
    .map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("data"),
        )
    })
}
/// Performs Page's trend test for an ordered alternative in a randomized
/// complete block design.
///
/// ```text
/// L = Σⱼ j Rⱼ,
/// ```
///
/// where `Rⱼ` is the sum over blocks of the within-block midranks of the
/// treatment in position `j` of `treatment_order` (`j = 1, …, k`). Rows of `data`
/// are the `b` blocks and columns the `k` treatments. `treatment_order` lists
/// column indices from smallest to largest expected response. Large `L` supports
/// the ordered alternative, so the alternative must be
/// [`TestAlternative::Greater`].
///
/// The exact p-value enumerates all `(k!)ᵇ` within-block permutations. Under
/// [`TestCalibration::Auto`] the test is exact when `(k!)ᵇ` is at most
/// [`ClassicalOptions::max_work`]. Otherwise it uses the normal approximation
/// with tie-corrected variance.
///
/// # Errors
///
/// - `InvalidInput` if `b < 2` or `k < 2`, or the alternative is not
///   `Greater`.
/// - `InvalidInput` if `treatment_order` is not a permutation of `0..k`.
/// - `InvalidInput` if every block is tied.
/// - `ConvergenceFailure` if the calibration is [`TestCalibration::Exact`] and
///   `(k!)ᵇ` exceeds `max_work`.
///
/// # References
///
/// E. B. Page, "Ordered hypotheses for multiple treatments: a significance test
/// for linear ranks", *J. Amer. Statist. Assoc.* 58 (1963), 216–230.
pub fn page_trend_test_with_options<S: RealScalar>(
    data: MatrixView<'_, S>,
    treatment_order: &[usize],
    options: ClassicalOptions,
) -> Result<ClassicalTestResult<S>, SciOxideError> {
    let b = data.rows();
    let k = data.cols();
    if b < 2
        || k < 2
        || treatment_order.len() != k
        || options.alternative != TestAlternative::Greater
    {
        return Err(invalid(
            scioxide_core::argument!("data"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Page needs complete blocks, a treatment order, and the upper alternative",
        ));
    }
    let total = b
        .checked_mul(k)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let mut weights = try_filled(k, 0.0, scioxide_core::quantity!("page_order"))?;
    for (position, &column) in treatment_order.iter().enumerate() {
        if column >= k || weights[column] != 0.0 {
            return Err(invalid(
                scioxide_core::argument!("treatment_order"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "treatment order must be a permutation of columns",
            ));
        }
        weights[column] = (position + 1) as f64;
    }
    let mut pooled = try_filled(total, 0.0, scioxide_core::quantity!("page_block_ranks"))?;
    let mut row = try_filled(k, S::ZERO, scioxide_core::quantity!("page_row"))?;
    let center = (k + 1) as f64 / 2.0;
    let mut rank_variance = 0.0;
    for block in 0..b {
        for (column, value) in row.iter_mut().enumerate() {
            *value = data.get(block, column)?;
        }
        let ranked = ranks::ranks(&row, RankMethod::Average)?;
        for (column, &rank) in ranked.iter().enumerate() {
            pooled[column * b + block] = rank;
            rank_variance += (rank - center) * (rank - center);
        }
    }
    if rank_variance == 0.0 {
        return Err(invalid(
            scioxide_core::argument!("data"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "all Page blocks are tied",
        ));
    }
    let statistic = pooled
        .chunks_exact(b)
        .zip(&weights)
        .map(|(column, &w)| w * column.iter().sum::<f64>())
        .sum::<f64>();
    let ways = scioxide_core::checked_factorial(k).and_then(|ways| {
        (0..b).try_fold(1usize, |count, _| {
            count.checked_mul(ways).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
            })
        })
    });
    let (p, method) = if permutation::use_exact(ways, options)? {
        let lengths = try_filled(k, b, scioxide_core::quantity!("page_permutation_groups"))?;
        let callback = |s: crate::ResampledIndices<'_>| -> Result<f64, SciOxideError> {
            weights
                .iter()
                .enumerate()
                .try_fold(0.0, |sum, (column, &w)| {
                    Ok(sum + w * s.group(column)?.iter().map(|&i| pooled[i]).sum::<f64>())
                })
        };
        (
            permutation::exact(&lengths, PermutationKind::PairedLabels, options, &callback)?.1,
            ClassicalMethod::Exact,
        )
    } else {
        let mean = b as f64 * k as f64 * center * center;
        let weight_variance = weights
            .iter()
            .map(|w| (w - center) * (w - center))
            .sum::<f64>();
        let variance = rank_variance * weight_variance / (k - 1) as f64;
        (
            normal_p(
                (statistic - mean) / variance.sqrt(),
                TestAlternative::Greater,
            )?,
            ClassicalMethod::Normal,
        )
    };
    result(statistic, p, method, TestSamplePolicy::Blocks, [b, k])
}
