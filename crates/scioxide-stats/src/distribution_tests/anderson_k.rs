//! Scholz–Stephens k-sample Anderson–Darling test.
use super::*;
use crate::ranking::RankPool;
use scioxide_core::backend::try_filled;

/// Selects the form of the k-sample Anderson–Darling statistic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AndersonKVariant {
    /// `A²ₐₖₙ`, based on midranks. Valid for continuous and discrete
    /// distributions; the default.
    #[default]
    Midrank,
    /// `A²ₖₙ` with right-continuous empirical distributions, for discrete
    /// distributions.
    Right,
    /// The form for continuous distributions. The pooled sample must have no
    /// ties.
    Continuous,
}

fn critical_probability(a2: f64, k: usize) -> f64 {
    let root = ((k - 1) as f64).sqrt();
    let m = (k - 1) as f64;
    let b0 = [0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085];
    let b1 = [-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615];
    let b2 = [-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154];
    let significance = [0.25_f64, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001];
    let mut critical = [0.0; 7];
    for i in 0..7 {
        critical[i] = b0[i] + b1[i] / root + b2[i] / m;
    }
    if a2 <= critical[0] {
        return 0.25;
    }
    if a2 >= critical[6] {
        return 0.001;
    }
    for i in 0..6 {
        if a2 <= critical[i + 1] {
            let t = (a2 - critical[i]) / (critical[i + 1] - critical[i]);
            return (significance[i].ln() + t * (significance[i + 1].ln() - significance[i].ln()))
                .exp();
        }
    }
    0.001
}
/// Performs the Scholz–Stephens k-sample Anderson–Darling test that `k ≥ 2` independent samples come from the same distribution.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`anderson_k_sample_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`anderson_k_sample_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let groups = [&x[..], &y[..]];
/// let actual = anderson_k_sample_test::<f64>(&groups)?;
/// let configured = anderson_k_sample_test_with_options::<f64>(&groups, AndersonKVariant::Midrank, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// F. W. Scholz and M. A. Stephens, "K-sample Anderson–Darling tests", *J.
/// Amer. Statist. Assoc.* 82 (1987), 918–924.
pub fn anderson_k_sample_test<S: RealScalar>(
    groups: &[&[S]],
) -> Result<DistributionTestResult<S>, SciOxideError> {
    anderson_k_sample_test_with_options(groups, AndersonKVariant::Midrank, Default::default())
        .map_err(|error| {
            crate::error_helpers::one_shot_argument(
                error,
                &["options", "variant"],
                scioxide_core::argument!("groups"),
            )
        })
}
/// Performs the Scholz–Stephens k-sample Anderson–Darling test that `k ≥ 2`
/// independent samples come from the same distribution.
///
/// The reported statistic is the standardized `(A² − (k − 1))/σₙ`, where `A²`
/// is the variant selected by `variant` and `σₙ²` its finite-sample null
/// variance. The p-value is interpolated log-linearly between the tabulated
/// critical values for significance levels 0.25, 0.1, 0.05, 0.025, 0.01, 0.005
/// and 0.001. It is clamped to `[0.001, 0.25]`, so values at the ends only
/// bound the p-value. For a precise p-value, use a permutation test. The cost is
/// O(N log N) time and O(N + k) memory for `N` observations.
///
/// # Errors
///
/// - `InvalidInput` if there are fewer than two groups, a group is empty, or
///   there are fewer than four observations.
/// - `InvalidInput` if all observations are equal, or `variant` is
///   [`AndersonKVariant::Continuous`] and the pooled sample has ties.
/// - `InvalidInput` if the calibration is [`TestCalibration::Exact`] or the
///   alternative is not two-sided.
/// - `NonFinite` if a group contains NaN or infinity.
/// - `ResourceLimit` if the pooled sample exceeds
///   [`ClassicalOptions::max_workspace_bytes`].
///
/// # References
///
/// F. W. Scholz and M. A. Stephens, "K-sample Anderson–Darling tests", *J.
/// Amer. Statist. Assoc.* 82 (1987), 918–924.
pub fn anderson_k_sample_test_with_options<S: RealScalar>(
    groups: &[&[S]],
    variant: AndersonKVariant,
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    require_asymptotic(options)?;
    require_omnibus(options)?;
    if groups.len() < 2 {
        return Err(invalid(
            scioxide_core::argument!("groups"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "Anderson k-sample needs at least two groups",
        ));
    }
    let total = groups.iter().try_fold(0usize, |a, g| {
        a.checked_add(g.len()).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })
    })?;
    if total < 4 {
        return Err(invalid(
            scioxide_core::argument!("groups"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "Anderson k-sample approximation needs at least four observations",
        ));
    }
    let bytes = total
        .checked_mul(core::mem::size_of::<(S, usize)>())
        .and_then(|b| {
            groups
                .len()
                .checked_mul(2 * core::mem::size_of::<usize>())
                .and_then(|g| b.checked_add(g))
        })
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    crate::error_helpers::limit(
        bytes,
        options.max_workspace_bytes(),
        scioxide_core::quantity!("anderson_k_bytes"),
    )?;
    for &values in groups {
        sample(values, 1, scioxide_core::argument!("groups"))?;
    }
    let pool = RankPool::samples(
        groups,
        scioxide_core::quantity!("anderson_pooled_observations"),
    )?;
    let ordered = pool.entries();
    if ordered[0].0 == ordered[total - 1].0 {
        return Err(invalid(
            scioxide_core::argument!("groups"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Anderson pool must contain distinct values",
        ));
    }
    let k = groups.len();
    let mut counts = try_filled(k, 0usize, scioxide_core::quantity!("anderson_group_counts"))?;
    let mut block = try_filled(k, 0usize, scioxide_core::quantity!("anderson_tie_block"))?;
    let nf = total as f64;
    let mut sum = 0.0;
    for (run, members) in pool.runs() {
        let (left, right, width) = (run.start, run.end, run.len());
        if variant == AndersonKVariant::Continuous && width > 1 {
            return Err(invalid(
                scioxide_core::argument!("variant"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "continuous Anderson variant requires no ties",
            ));
        }
        block.fill(0);
        for &(_, group) in members {
            block[group] += 1;
        }
        if variant == AndersonKVariant::Right {
            if right < total {
                let b = right as f64;
                for (group, &size) in groups.iter().enumerate() {
                    let m = (counts[group] + block[group]) as f64;
                    let difference = nf * m - b * size.len() as f64;
                    sum += width as f64 / nf * difference * difference
                        / (b * (nf - b))
                        / size.len() as f64;
                }
            }
        } else {
            let b = (left as f64 + right as f64) / 2.0;
            let denominator = b * (nf - b) - nf * width as f64 / 4.0;
            if denominator <= 0.0 {
                return Err(invalid(
                    scioxide_core::argument!("groups"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "Anderson midrank denominator collapsed",
                ));
            }
            for (group, &size) in groups.iter().enumerate() {
                let m = counts[group] as f64 + block[group] as f64 / 2.0;
                let difference = nf * m - b * size.len() as f64;
                sum +=
                    width as f64 / nf * difference * difference / denominator / size.len() as f64;
            }
        }
        for group in 0..k {
            counts[group] += block[group];
        }
    }
    if variant != AndersonKVariant::Right {
        sum *= (nf - 1.0) / nf;
    }
    let harmonic = groups.iter().map(|g| 1.0 / g.len() as f64).sum::<f64>();
    let mut h = 1.0;
    let mut partial = 0.0;
    let mut g = 0.0;
    for j in 2..total {
        partial += 1.0 / (total + 1 - j) as f64;
        g += partial / j as f64;
    }
    for j in 2..total {
        h += 1.0 / j as f64;
    }
    let kf = k as f64;
    let a = (4.0 * g - 6.0) * (kf - 1.0) + (10.0 - 6.0 * g) * harmonic;
    let b = (2.0 * g - 4.0) * kf * kf + 8.0 * h * kf + (2.0 * g - 14.0 * h - 4.0) * harmonic
        - 8.0 * h
        + 4.0 * g
        - 6.0;
    let c = (6.0 * h + 2.0 * g - 2.0) * kf * kf
        + (4.0 * h - 4.0 * g + 6.0) * kf
        + (2.0 * h - 6.0) * harmonic
        + 4.0 * h;
    let d = (2.0 * h + 6.0) * kf * kf - 4.0 * h * kf;
    let sigma2 =
        (a * nf.powi(3) + b * nf * nf + c * nf + d) / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0));
    if sigma2 <= 0.0 {
        return Err(invalid(
            scioxide_core::argument!("groups"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "Anderson normalization variance is not positive",
        ));
    }
    let standardized = (sum - (kf - 1.0)) / sigma2.sqrt();
    result(
        standardized,
        critical_probability(standardized, k),
        DistributionMethod::Asymptotic,
        NullParameters::Exchangeable,
        total,
        k,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn midrank_and_discrete_boundary() {
        let x = [1.0_f64, 2.0, 3.0, 4.0];
        let y = [1.0_f64, 2.0, 5.0, 6.0];
        let r = anderson_k_sample_test_with_options(
            &[&x, &y],
            AndersonKVariant::Midrank,
            ClassicalOptions::default(),
        )
        .unwrap();
        assert!(r.statistic.is_finite() && (0.001..=0.25).contains(&r.p_value));
        assert!(
            anderson_k_sample_test_with_options(
                &[&x, &y],
                AndersonKVariant::Continuous,
                ClassicalOptions::default()
            )
            .is_err()
        );
        assert!(
            anderson_k_sample_test_with_options(
                &[&x, &y],
                AndersonKVariant::Right,
                ClassicalOptions::default()
            )
            .is_ok()
        );
        let reversed = anderson_k_sample_test_with_options(
            &[&y, &x],
            AndersonKVariant::Midrank,
            ClassicalOptions::default(),
        )
        .unwrap();
        assert!((r.statistic - reversed.statistic).abs() < 1e-12);
        assert!((r.p_value - reversed.p_value).abs() < 1e-12);
        let x32 = [1.0_f32, 2.0, 3.0, 4.0];
        let y32 = [1.0_f32, 2.0, 5.0, 6.0];
        let r32 = anderson_k_sample_test_with_options(
            &[&x32, &y32],
            AndersonKVariant::Midrank,
            ClassicalOptions::default(),
        )
        .unwrap();
        assert!((r32.statistic as f64 - r.statistic).abs() < 1e-5);
    }
}
