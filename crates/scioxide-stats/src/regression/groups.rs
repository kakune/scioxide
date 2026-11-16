//! One-way analysis of variance and rank tests for several groups.
use super::{budget_bytes, f_probability, finite, invalid, number, zeros};
use crate::StatisticsOptions;
use crate::arithmetic::center_difference;
use crate::group_summary::{GroupSummaries, GroupSummary};
use crate::ranking::{TieSums, finite_cmp, tie_runs};
use crate::{RankMethod, rankdata};

use scioxide_core::{RealScalar, Scaled, ScaledAccumulator, SciOxideError};
use scioxide_linalg::MatrixView;

/// Names the test that produced a [`GroupTest`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupTestMethod {
    /// One-way ANOVA F-test, assuming normal groups with equal variances.
    Anova,
    /// Welch's ANOVA, an approximate F-test for unequal variances.
    Welch,
    /// Alexander–Govern test for unequal variances, with an approximate
    /// chi-square distribution.
    AlexanderGovern,
    /// Kruskal–Wallis H-test with tie correction and the asymptotic
    /// chi-square distribution.
    KruskalWallis,
    /// Friedman test for matched blocks with tie correction and the
    /// asymptotic chi-square distribution.
    Friedman,
    /// F-test of nested linear models.
    NestedModels,
}
/// Holds the statistic, p-value and degrees of freedom of an F or chi-square
/// test.
///
/// The fields are read-only.
///
/// ```compile_fail
/// fn corrupt(mut test: scioxide_stats::GroupTest<f64>) { test.p_value = 1.0; }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct GroupTest<S: RealScalar> {
    pub(super) statistic: S,
    pub(super) p_value: S,
    pub(super) degrees_of_freedom: f64,
    pub(super) denominator_degrees: Option<f64>,
    pub(super) method: GroupTestMethod,
}
impl<S: RealScalar> GroupTest<S> {
    /// Returns the test statistic, which is finite and nonnegative.
    pub const fn statistic(&self) -> S {
        self.statistic
    }
    /// Returns the upper-tail p-value of the statistic under the null
    /// hypothesis.
    pub const fn p_value(&self) -> S {
        self.p_value
    }
    /// Returns the numerator degrees of freedom of an F-test, or the degrees of
    /// freedom of a chi-square test.
    pub const fn degrees_of_freedom(&self) -> f64 {
        self.degrees_of_freedom
    }
    /// Returns the denominator degrees of freedom of an F-test, or `None` for a
    /// chi-square test.
    pub const fn denominator_degrees(&self) -> Option<f64> {
        self.denominator_degrees
    }
    /// Returns the test.
    pub const fn method(&self) -> GroupTestMethod {
        self.method
    }
}
/// Holds the size, mean and variance of each of `k ≥ 2` independent samples,
/// for tests of equal means.
///
/// Only the summaries are stored, in O(k) space, not the observations. Means
/// and variances are kept in an extended exponent range, so data near the
/// limits of `S` do not overflow. The tests are invariant under translation
/// and nonzero scaling of the data, up to rounding of the input.
///
/// [`Self::anova`] and [`Self::alexander_govern`] test equal means.
/// [`Self::pairwise`] and [`Self::compare_control`] compare individual means.
///
/// # Examples
///
/// ```
/// // Means 0 and 1e154. The sums of squares exceed f64::MAX, yet F = 1.5.
/// use scioxide_stats::{GroupTestMethod, PreparedGroups, StatisticsOptions};
/// let a = [-1e154_f64, -1e154, 1e154, 1e154];
/// let b = [0.0, 0.0, 2e154, 2e154];
/// let groups = PreparedGroups::new(&[&a, &b], StatisticsOptions::new())?;
/// assert!((groups.anova(GroupTestMethod::Anova)?.statistic() - 1.5).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct PreparedGroups<S: RealScalar> {
    pub(super) summaries: GroupSummaries,
    marker: core::marker::PhantomData<S>,
}
impl<S: RealScalar> Clone for PreparedGroups<S> {
    fn clone(&self) -> Self {
        Self {
            summaries: GroupSummaries {
                groups: self.summaries.groups.clone(),
                total: self.summaries.total,
                within: self.summaries.within,
            },
            marker: core::marker::PhantomData,
        }
    }
}
impl<S: RealScalar> PreparedGroups<S> {
    /// Summarizes each sample in `samples` as one group.
    ///
    /// A group of one observation contributes zero to the within-group sum of
    /// squares. Only `options.max_bytes()` applies.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if there are fewer than two groups, or the total
    ///   number of observations exceeds 2⁵³.
    /// - `InvalidInput` if a group is empty.
    /// - `NonFinite` if an observation is NaN or infinite.
    /// - `ResourceLimit` if the summaries exceed `options.max_bytes()`.
    pub fn new(samples: &[&[S]], options: StatisticsOptions) -> Result<Self, SciOxideError> {
        Ok(Self {
            summaries: GroupSummaries::prepare(samples, options.max_bytes())?,
            marker: core::marker::PhantomData,
        })
    }
    /// Returns the number of groups.
    pub fn len(&self) -> usize {
        self.summaries.groups.len()
    }
    /// Returns `false`, since there are always at least two groups.
    pub fn is_empty(&self) -> bool {
        self.summaries.groups.is_empty()
    }
    /// Tests equal group means by one-way ANOVA or Welch's ANOVA.
    ///
    /// With group sizes `nᵢ`, means `ȳᵢ`, variances `sᵢ²`, total size `N` and
    /// `k` groups:
    ///
    /// ```text
    /// Anova:  F = [Σ nᵢ (ȳᵢ − ȳ)² / (k − 1)] / [Σ (nᵢ − 1) sᵢ² / (N − k)],
    ///         ν₁ = k − 1,  ν₂ = N − k,
    /// Welch:  F = [Σ wᵢ (ȳᵢ − ȳ_w)² / (k − 1)] / [1 + 2(k − 2) Λ / (k² − 1)],
    ///         ν₁ = k − 1,  ν₂ = (k² − 1) / (3 Λ),
    /// ```
    ///
    /// where `ȳ` is the grand mean, `wᵢ = nᵢ / sᵢ²`, `W = Σ wᵢ`,
    /// `ȳ_w = Σ wᵢ ȳᵢ / W` and `Λ = Σ (1 − wᵢ/W)² / (nᵢ − 1)`. The p-value is
    /// the upper tail of the F distribution with `(ν₁, ν₂)` degrees of freedom.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `method` is not [`GroupTestMethod::Anova`] or
    ///   [`GroupTestMethod::Welch`].
    /// - `InvalidInput` for `Anova` if `N − k ≤ 0` or the pooled variance is
    ///   zero.
    /// - `InvalidInput` for `Welch` if a group has fewer than two observations
    ///   or zero variance.
    /// - `NumericalInstability` if the statistic overflows `S`.
    ///
    /// # References
    ///
    /// B. L. Welch, "On the comparison of several mean values: an alternative
    /// approach", *Biometrika* 38, 330–336, 1951.
    pub fn anova(&self, method: GroupTestMethod) -> Result<GroupTest<S>, SciOxideError> {
        let k = self.summaries.groups.len() as f64;
        let (statistic, df) = match method {
            GroupTestMethod::Anova => {
                let (statistic, df) = self.summaries.anova()?;
                (statistic.round()?, df)
            }
            GroupTestMethod::Welch => {
                self.summaries.positive_variances()?;
                let (mean, sum) = self.summaries.weighted_center(GroupSummary::precision)?;
                let mut correction = ScaledAccumulator::ZERO;
                let mut numerator = ScaledAccumulator::ZERO;
                for &group in &self.summaries.groups {
                    let weight = group.precision()?;
                    correction.add_scaled(
                        Scaled::ONE
                            .sub(weight.div(sum)?)?
                            .square()?
                            .div(Scaled::from_value(group.n - 1.0)?)?,
                    )?;
                    numerator
                        .add_scaled(weight.mul(center_difference(mean, group.mean)?.square()?)?)?;
                }
                let correction = correction.total()?.round()?;
                (
                    numerator
                        .total()?
                        .div(Scaled::from_value(k - 1.0)?)?
                        .div(Scaled::from_value(
                            1.0 + 2.0 * (k - 2.0) / (k * k - 1.0) * correction,
                        )?)?
                        .round()?,
                    (k * k - 1.0) / (3.0 * correction),
                )
            }
            _ => {
                return Err(invalid(
                    scioxide_core::argument!("method"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "select classical or Welch ANOVA",
                ));
            }
        };
        Ok(GroupTest {
            statistic: Scaled::from_value(finite(statistic)?)?.round_to()?,
            p_value: number(f_probability(statistic, k - 1.0, df)?)?,
            degrees_of_freedom: k - 1.0,
            denominator_degrees: Some(df),
            method,
        })
    }
    /// Tests equal group means by the Alexander–Govern test, which does not
    /// assume equal variances.
    ///
    /// Each group's t statistic against the precision-weighted mean is
    /// transformed to an approximate standard normal deviate `zᵢ` by a
    /// normalizing transformation. The statistic `A = Σ zᵢ²`
    /// approximately follows the chi-square distribution with `k − 1` degrees
    /// of freedom, and the p-value is its upper tail.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a group has fewer than two observations or zero
    ///   variance.
    /// - `NumericalInstability` if the statistic is not finite.
    ///
    /// # References
    ///
    /// R. A. Alexander and D. M. Govern, "A new and simpler approximation for
    /// ANOVA under variance heterogeneity", *J. Educational Statist.* 19(2),
    /// 91–101, 1994.
    pub fn alexander_govern(&self) -> Result<GroupTest<S>, SciOxideError> {
        self.summaries.positive_variances()?;
        let (mean, _) = self.summaries.weighted_center(GroupSummary::precision)?;
        let mut statistic = ScaledAccumulator::ZERO;
        for &group in &self.summaries.groups {
            let t2 = center_difference(mean, group.mean)?
                .square()?
                .mul(group.precision()?)?;
            let v = group.n - 1.0;
            let ratio = t2.div(Scaled::from_value(v)?)?;
            // log1p near zero retains tiny contrasts; large ratios stay scaled.
            let logarithm = crate::arithmetic::scaled_ln_1p(ratio)?;
            let a = v - 0.5;
            let c_squared = logarithm.mul(Scaled::from_value(a)?)?;
            let b = 48.0 * a * a;
            let c2 = c_squared.round()?;
            let factor = 1.0 + (c2 + 3.0) / b
                - (4.0 * c2 * c2 * c2 + 33.0 * c2 * c2 + 240.0 * c2 + 855.0)
                    / (10.0 * b * b + 8.0 * b * c2 * c2 + 1000.0 * b);
            statistic.add_scaled(c_squared.mul(Scaled::from_value(finite(factor * factor)?)?)?)?;
        }
        chi_test(
            statistic.total()?.round()?,
            self.summaries.groups.len() as f64 - 1.0,
            GroupTestMethod::AlexanderGovern,
        )
    }
}
fn chi_test<S: RealScalar>(
    statistic: f64,
    degrees: f64,
    method: GroupTestMethod,
) -> Result<GroupTest<S>, SciOxideError> {
    let statistic = finite(statistic)?;
    if statistic < 0.0 || degrees <= 0.0 {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "invalid chi-square statistic",
        ));
    }
    Ok(GroupTest {
        statistic: Scaled::from_value(statistic)?.round_to()?,
        p_value: number(scioxide_special::gamma_q(degrees / 2.0, statistic / 2.0)?)?,
        degrees_of_freedom: degrees,
        denominator_degrees: None,
        method,
    })
}
/// Runs one test on each of several group sets, in parallel when the execution
/// context allows it.
///
/// `output[j]` receives the test of `plans[j]`. `method` is
/// [`GroupTestMethod::Anova`], [`GroupTestMethod::Welch`] or
/// [`GroupTestMethod::AlexanderGovern`]. `options` selects the execution
/// policy, and `staging` holds one entry per plan. The results equal those of
/// sequential calls. See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
///
/// # Errors
///
/// - `InvalidInput` if `output` or `staging` does not have one entry per plan.
/// - The errors of [`PreparedGroups::anova`] or
///   [`PreparedGroups::alexander_govern`] for any plan.
/// - The error of the executor.
pub fn group_tests_into<S: RealScalar>(
    plans: &[PreparedGroups<S>],
    method: GroupTestMethod,
    options: StatisticsOptions,
    output: &mut [GroupTest<S>],
    staging: &mut [GroupTest<S>],
    context: scioxide_core::ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    if output.len() != plans.len() || staging.len() != plans.len() {
        return Err(invalid(
            scioxide_core::argument!("output"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "group batch shape mismatch",
        ));
    }
    let mut workers = [(); 64];
    crate::estimation::execute(
        options,
        context,
        plans.len(),
        1,
        &mut workers,
        staging,
        |job, _, slot| {
            slot[0] = if method == GroupTestMethod::AlexanderGovern {
                plans[job].alexander_govern()?
            } else {
                plans[job].anova(method)?
            };
            Ok(())
        },
    )?;
    context.check()?;
    output.copy_from_slice(staging);
    Ok(())
}
/// Computes the Kruskal–Wallis H-test that `k` independent samples come from the same distribution.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`kruskal_wallis_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`kruskal_wallis_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let groups = [&x[..], &y[..]];
/// let actual = kruskal_wallis_test::<f64>(&groups)?;
/// let configured = kruskal_wallis_test_with_options::<f64>(&groups, Default::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// W. H. Kruskal and W. A. Wallis, "Use of ranks in one-criterion variance
/// analysis", *J. Amer. Statist. Assoc.* 47, 583–621, 1952.
pub fn kruskal_wallis_test<S: RealScalar>(groups: &[&[S]]) -> Result<GroupTest<S>, SciOxideError> {
    kruskal_wallis_test_with_options(groups, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("groups"),
        )
    })
}
/// Computes the Kruskal–Wallis H-test that `k` independent samples come from
/// the same distribution.
///
/// The observations are pooled and ranked, with average ranks for ties. With
/// `Rᵢ` the rank sum and `nᵢ` the size of group `i`, and `N` the total size,
///
/// ```text
/// H = [12 / (N (N + 1)) Σ Rᵢ² / nᵢ − 3 (N + 1)] / (1 − Σ (t³ − t) / (N³ − N)),
/// ```
///
/// where the sum in the denominator runs over groups of `t` tied values. The
/// p-value is the upper tail of the chi-square distribution with `k − 1`
/// degrees of freedom. This asymptotic distribution is not exact for small
/// samples. The test allocates O(N) storage, bounded by
/// `options.max_bytes()`. No other option applies.
///
/// # Errors
///
/// - `InvalidInput` if there are fewer than two groups or a group is empty.
/// - `InvalidInput` if all observations are tied.
/// - `NonFinite` if an observation is NaN or infinite.
/// - `ResourceLimit` if the storage exceeds `options.max_bytes()`.
///
/// # Examples
///
/// ```
/// // Ranks 1, 2, 3 and 4, 5, 6: H = 12/42 (36 + 225)/3 − 21 = 27/7.
/// use scioxide_stats::{StatisticsOptions, kruskal_wallis_test_with_options};
/// let test = kruskal_wallis_test_with_options(&[&[1.0_f64, 2.0, 3.0], &[4.0, 5.0, 6.0]], StatisticsOptions::new())?;
/// assert!((test.statistic() - 27.0 / 7.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// W. H. Kruskal and W. A. Wallis, "Use of ranks in one-criterion variance
/// analysis", *J. Amer. Statist. Assoc.* 47, 583–621, 1952.
pub fn kruskal_wallis_test_with_options<S: RealScalar>(
    groups: &[&[S]],
    options: StatisticsOptions,
) -> Result<GroupTest<S>, SciOxideError> {
    if groups.len() < 2 || groups.iter().any(|g| g.is_empty()) {
        return Err(invalid(
            scioxide_core::argument!("groups"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "nonempty independent groups required",
        ));
    }
    let n = groups.iter().try_fold(0usize, |a, g| {
        a.checked_add(g.len()).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size"))
        })
    })?;
    budget_bytes(n, core::mem::size_of::<S>() * 16, options.max_bytes())?;
    let mut values =
        scioxide_core::backend::try_with_capacity(n, scioxide_core::quantity!("pooled_ranks"))?;
    for &g in groups {
        values.extend_from_slice(g);
    }
    let ranks = rankdata(&values, RankMethod::Average)?;
    let mut offset = 0;
    let mut squares = 0.0;
    for &g in groups {
        let sum = ranks[offset..offset + g.len()]
            .iter()
            .try_fold(0.0, |a, r| Ok::<_, SciOxideError>(a + r.to_f64()?))?;
        squares += sum * sum / g.len() as f64;
        offset += g.len();
    }
    let nf = n as f64;
    let correction = 1.0 - TieSums::of_values(&mut values).cubes / (nf * (nf * nf - 1.0));
    if correction <= 0.0 {
        return Err(invalid(
            scioxide_core::argument!("groups"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Kruskal-Wallis is undefined for all-tied observations",
        ));
    }
    chi_test(
        ((12.0 * squares / (nf * (nf + 1.0)) - 3.0 * (nf + 1.0)) / correction).max(0.0),
        groups.len() as f64 - 1.0,
        GroupTestMethod::KruskalWallis,
    )
}
/// Computes the result of [`friedman_test_with_options`] with the [one-shot defaults](crate#one-shot-forms).
///
/// Uses the same numerical method, cost and accuracy as the configured form.
///
/// # Errors
///
/// Returns the errors of [`friedman_test_with_options`] for the fixed default settings.
pub fn friedman_test<S: RealScalar>(
    samples: MatrixView<'_, S>,
) -> Result<GroupTest<S>, SciOxideError> {
    friedman_test_with_options(samples, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["options"],
            scioxide_core::argument!("samples"),
        )
    })
}
/// Computes the Friedman test that `k` treatments have the same effect in
/// `n` matched blocks.
///
/// Rows of `samples` are blocks and columns are treatments. Each block is
/// ranked separately, with average ranks for ties. With `Rⱼ` the rank sum of
/// treatment `j`,
///
/// ```text
/// Q = [12 / (n k (k + 1)) Σ Rⱼ² − 3 n (k + 1)] / (1 − Σ (t³ − t) / (n k (k² − 1))),
/// ```
///
/// where the sum in the denominator runs over groups of `t` tied values within
/// each block. The p-value is the upper tail of the chi-square distribution
/// with `k − 1` degrees of freedom. The test uses O(k) storage, bounded by
/// `options.max_bytes()`. No other option applies.
///
/// # Errors
///
/// - `InvalidInput` if there are fewer than two blocks or three treatments.
/// - `InvalidInput` if every block is entirely tied.
/// - `NonFinite` if an observation is NaN or infinite.
/// - `ResourceLimit` if the storage exceeds `options.max_bytes()`.
///
/// # Examples
///
/// ```
/// // Rank sums 2, 5, 5 over two blocks: Q = 12/24 · 54 − 24 = 3.
/// use scioxide_linalg::MatrixView;
/// use scioxide_stats::{StatisticsOptions, friedman_test_with_options};
/// let blocks = [1.0_f64, 2.0, 3.0, 1.0, 3.0, 2.0];
/// let test = friedman_test_with_options(MatrixView::from_slice(2, 3, &blocks)?, StatisticsOptions::new())?;
/// assert!((test.statistic() - 3.0).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// M. Friedman, "The use of ranks to avoid the assumption of normality
/// implicit in the analysis of variance", *J. Amer. Statist. Assoc.* 32,
/// 675–701, 1937.
pub fn friedman_test_with_options<S: RealScalar>(
    samples: MatrixView<'_, S>,
    options: StatisticsOptions,
) -> Result<GroupTest<S>, SciOxideError> {
    let (n, k) = (samples.rows(), samples.cols());
    if n < 2 || k < 3 {
        return Err(invalid(
            scioxide_core::argument!("samples"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Friedman needs two blocks and three treatments",
        ));
    }
    budget_bytes(
        k,
        core::mem::size_of::<f64>() + core::mem::size_of::<S>() + core::mem::size_of::<usize>(),
        options.max_bytes(),
    )?;
    let mut sums = zeros::<f64>(k)?;
    let mut row = zeros::<S>(k)?;
    let mut order =
        scioxide_core::backend::try_filled(k, 0usize, scioxide_core::quantity!("ranks"))?;
    let mut ties = 0.0;
    for i in 0..n {
        for (j, v) in row.iter_mut().enumerate() {
            *v = samples.get(i, j)?;
            if !v.is_finite() {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Input,
                    "nonfinite observation",
                ));
            }
        }
        for (j, slot) in order.iter_mut().enumerate() {
            *slot = j;
        }
        order.sort_unstable_by(|&a, &b| finite_cmp(&row[a], &row[b]));
        for (run, members) in tie_runs(&order, |&a, &b| row[a] == row[b]) {
            let rank = run.midrank()?;
            for &column in members {
                sums[column] += rank;
            }
        }
        ties += TieSums::of_values(&mut row).cubes;
    }
    let nf = n as f64;
    let kf = k as f64;
    let correction = 1.0 - ties / (nf * kf * (kf * kf - 1.0));
    if correction <= 0.0 {
        return Err(invalid(
            scioxide_core::argument!("samples"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "Friedman is undefined when all blocks are tied",
        ));
    }
    let statistic = (12.0 * sums.iter().map(|v| v * v).sum::<f64>() / (nf * kf * (kf + 1.0))
        - 3.0 * nf * (kf + 1.0))
        / correction;
    chi_test(statistic.max(0.0), kf - 1.0, GroupTestMethod::Friedman)
}
