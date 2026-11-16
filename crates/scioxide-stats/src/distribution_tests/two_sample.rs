//! Two-sample Kolmogorov–Smirnov and Cramér–von Mises tests.
use super::*;
use crate::ranking::RankPool;
use crate::{
    PermutationKind, ResamplingInference, ResamplingMethod, ResamplingPlan, ResamplingProblem,
    ResamplingResult, ResamplingWorkspace, StatsTermination,
};
use alloc::vec::Vec;

use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_random::Kolmogorov;

fn sorted_pair<S: RealScalar>(
    x: &[S],
    y: &[S],
    bytes: usize,
) -> Result<(Vec<S>, Vec<S>), SciOxideError> {
    let total = x
        .len()
        .checked_add(y.len())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    crate::error_helpers::limit(
        scioxide_core::backend::checked_mul(
            total,
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("ordering_bytes"),
        )?,
        bytes,
        scioxide_core::quantity!("ordering_bytes"),
    )?;
    Ok((
        super::one_sample::sorted(x, bytes)?,
        super::one_sample::sorted(y, bytes)?,
    ))
}

fn ks_counts<S: RealScalar>(x: &[S], y: &[S]) -> (usize, usize, bool) {
    let (n, m) = (x.len(), y.len());
    let (mut i, mut j, mut plus, mut minus) = (0usize, 0usize, 0usize, 0usize);
    let mut tied = false;
    while i < n || j < m {
        let next = if j == m || (i < n && x[i] < y[j]) {
            x[i]
        } else {
            y[j]
        };
        let old_i = i;
        let old_j = j;
        while i < n && x[i] == next {
            i += 1;
        }
        while j < m && y[j] == next {
            j += 1;
        }
        tied |= i - old_i + j - old_j > 1;
        let diff = i * m;
        let other = j * n;
        plus = plus.max(diff.saturating_sub(other));
        minus = minus.max(other.saturating_sub(diff));
    }
    (plus, minus, tied)
}

fn exact_ks_tail(
    n: usize,
    m: usize,
    observed: usize,
    alternative: TestAlternative,
    max_bytes: usize,
) -> Result<f64, SciOxideError> {
    if observed == 0 {
        return Ok(1.0);
    }
    let bytes = (m + 1)
        .checked_mul(core::mem::size_of::<f64>())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    crate::error_helpers::limit(
        bytes,
        max_bytes,
        scioxide_core::quantity!("two_sample_bytes"),
    )?;
    let mut row = try_filled(m + 1, 0.0, scioxide_core::quantity!("two_sample_ks_dp"))?;
    row[0] = 1.0;
    let mut tail = 0.0;
    for i in 0..=n {
        for j in 0..=m {
            if i == 0 && j == 0 {
                continue;
            }
            let signed = (i as i128) * (m as i128) - (j as i128) * (n as i128);
            let distance = match alternative {
                TestAlternative::Greater => signed.max(0),
                TestAlternative::Less => (-signed).max(0),
                TestAlternative::TwoSided => signed.abs(),
            };
            let remaining = (n + m - i - j + 1) as f64;
            let from_x = if i > 0 {
                row[j] * (n - i + 1) as f64 / remaining
            } else {
                0.0
            };
            let from_y = if j > 0 {
                row[j - 1] * (m - j + 1) as f64 / remaining
            } else {
                0.0
            };
            let mass = from_x + from_y;
            if distance >= observed as i128 {
                tail += mass;
                row[j] = 0.0;
            } else {
                row[j] = mass;
            }
        }
    }
    Ok(tail.clamp(0.0, 1.0))
}
/// Performs the two-sample Kolmogorov–Smirnov test for continuous data.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`ks_two_sample_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`ks_two_sample_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = ks_two_sample_test::<f64>(&x, &y)?;
/// let configured = ks_two_sample_test_with_options::<f64>(&x, &y, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ks_two_sample_test<S: RealScalar>(
    x: &[S],
    y: &[S],
) -> Result<DistributionTestResult<S>, SciOxideError> {
    ks_two_sample_test_with_options(x, y, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the two-sample Kolmogorov–Smirnov test for continuous data.
///
/// With `Fₙ` and `Gₘ` the empirical CDFs of `x` and `y`, the statistic is
/// `D⁺ = maxₜ (Fₙ(t) − Gₘ(t))` for [`TestAlternative::Greater`],
/// `D⁻ = maxₜ (Gₘ(t) − Fₙ(t))` for `Less`, and `D = max(D⁺, D⁻)` for a
/// two-sided test.
///
/// The exact p-value counts lattice paths in O(n m) time and O(m) memory.
/// Under [`TestCalibration::Auto`] it is used when `n m` is at most
/// [`ClassicalOptions::max_work`]. Otherwise the two-sided test uses the
/// Kolmogorov distribution of `√(n m/(n + m)) D` and the one-sided tests use
/// `exp(−2 n m D²/(n + m))`. Both assume no ties; for tied or discrete data,
/// use [`ks_two_sample_permutation_test`].
///
/// # Errors
///
/// - `InvalidInput` if either sample is empty or the pooled sample has ties.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
/// - `ResourceLimit` if the sorted copies exceed
///   [`ClassicalOptions::max_workspace_bytes`].
/// - `ConvergenceFailure` if the calibration is [`TestCalibration::Exact`] and
///   `n m` exceeds `max_work`.
///
/// # Examples
///
/// ```
/// use scioxide_stats::{ClassicalOptions, ks_two_sample_test_with_options};
///
/// // Disjoint samples give D = 1. Two of the C(4, 2) = 6 label assignments
/// // reach D = 1, so p = 1/3.
/// let test = ks_two_sample_test_with_options(&[1.0_f64, 2.0], &[3.0, 4.0], ClassicalOptions::default())?;
/// assert!((test.statistic() - 1.0).abs() < 1e-15);
/// assert!((test.p_value() - 1.0 / 3.0).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn ks_two_sample_test_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    let (x, y) = sorted_pair(x, y, options.max_workspace_bytes())?;
    let (n, m) = (x.len(), y.len());
    let work = n
        .checked_mul(m)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let (plus, minus, tied) = ks_counts(&x, &y);
    if tied {
        return Err(invalid(
            scioxide_core::argument!("options.calibration"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "ties require conditional permutation calibration",
        ));
    }
    let observed = match options.alternative() {
        TestAlternative::Greater => plus,
        TestAlternative::Less => minus,
        TestAlternative::TwoSided => plus.max(minus),
    };
    let d = observed as f64 / work as f64;
    let exact = options.calibration() == TestCalibration::Exact
        || (options.calibration() == TestCalibration::Auto && work <= options.max_work());
    if exact {
        crate::error_helpers::work(
            work,
            options.max_work(),
            scioxide_core::BudgetKind::Node,
            "exact KS lattice exceeds options.max_work",
        )?;
    }
    let p = if exact {
        exact_ks_tail(
            n,
            m,
            observed,
            options.alternative(),
            options.max_workspace_bytes() - (n + m) * core::mem::size_of::<S>(),
        )?
    } else {
        let en = (work as f64 / (n + m) as f64).sqrt();
        match options.alternative() {
            TestAlternative::TwoSided => Kolmogorov::<f64>::with_options(
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )?
            .sf(en * d)?,
            _ => (-2.0 * (en * d).powi(2)).exp(),
        }
    };
    result(
        d,
        p,
        if exact {
            DistributionMethod::Exact
        } else {
            DistributionMethod::Asymptotic
        },
        NullParameters::Exchangeable,
        n + m,
        2,
    )
}

/// Result of [`ks_two_sample_permutation_test`]: the test result and the
/// resampling report.
///
/// The test result is present exactly when the resampling run produced a
/// p-value.
#[derive(Clone, Copy, Debug)]
pub struct KsPermutationResult<S: RealScalar> {
    pub(crate) test: Option<DistributionTestResult<S>>,
    pub(crate) resampling: ResamplingResult<S>,
}
impl<S: RealScalar> KsPermutationResult<S> {
    /// Returns the test result, or `None` if a limit or a stop request ended the
    /// run early.
    pub const fn test(&self) -> Option<&DistributionTestResult<S>> {
        self.test.as_ref()
    }
    /// Returns the resampling report: termination status, tail counts and Monte
    /// Carlo uncertainty.
    pub const fn resampling(&self) -> &ResamplingResult<S> {
        &self.resampling
    }
}

/// Performs the two-sample Kolmogorov–Smirnov test with a permutation p-value.
///
/// The statistic is that of [`ks_two_sample_test_with_options`], selected by `alternative`.
/// The p-value comes from permutations of the group labels, so it is valid for
/// tied and discrete data. `plan` must permute labels between two independent
/// groups of sizes `n` and `m`, use the upper tail, and have at least `n + m`
/// statistic scratch entries. `replicates`, `output`, `workspace` and `scope`
/// have the same meaning as in [`ResamplingPlan::evaluate_into`]. Preparation costs
/// O(N log N) time and each replicate O(N), `N = n + m`.
///
/// # Errors
///
/// - `InvalidInput` if either sample is empty.
/// - `InvalidInput` if `plan` is not an upper-tail permutation plan for
///   independent groups of sizes `n` and `m`, or its statistic scratch is
///   shorter than `n + m`.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
/// - `ResourceLimit` if the pooled sample and the plan's workspace together
///   exceed the plan's
///   [`ResamplingOptions::max_workspace_bytes`](crate::ResamplingOptions::max_workspace_bytes).
/// - Any error of [`ResamplingPlan::evaluate_into`].
#[allow(clippy::too_many_arguments)]
pub fn ks_two_sample_permutation_test_into<'scope, 'capability, S: RealScalar>(
    x: &[S],
    y: &[S],
    alternative: TestAlternative,
    plan: &ResamplingPlan<S>,
    replicates: impl crate::ReplicateExecution<S>,
    output: &mut [S],
    workspace: &mut ResamplingWorkspace<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<KsPermutationResult<S>, SciOxideError> {
    workspace.reject_attempt();
    sample(x, 1, scioxide_core::argument!("x"))?;
    sample(y, 1, scioxide_core::argument!("y"))?;
    if !matches!(
        plan.method(),
        ResamplingMethod::Permutation {
            kind: PermutationKind::Independent,
            alternative: TestAlternative::Greater,
            ..
        }
    ) {
        return Err(invalid(
            scioxide_core::argument!("plan"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "KS conditional test needs an independent upper-tail permutation plan",
        ));
    }
    let n = x.len();
    let m = y.len();
    let total = n
        .checked_add(m)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let product = n
        .checked_mul(m)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    if plan.statistic_scratch_elements() < total {
        return Err(invalid(
            scioxide_core::argument!("workspace"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "KS permutation scratch too short",
        ));
    }
    let runner = plan.workspace_bytes()?;
    let bytes = total
        .checked_mul(core::mem::size_of::<(S, usize)>())
        .and_then(|pool| pool.checked_add(runner))
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    crate::error_helpers::limit(
        bytes,
        plan.options.max_workspace_bytes(),
        scioxide_core::quantity!("two_sample_bytes"),
    )?;
    let pool = RankPool::positions(&[x, y], scioxide_core::quantity!("ks_pooled_observations"))?;
    let statistic =
        |selection: crate::ResampledIndices<'_>, scratch: &mut [S]| -> Result<S, SciOxideError> {
            if selection.groups() != 2
                || selection.group(0)?.len() != n
                || selection.group(1)?.len() != m
            {
                return Err(invalid(
                    scioxide_core::argument!("plan"),
                    scioxide_core::InvalidInputReason::Inconsistent,
                    "KS permutation group lengths do not match inputs",
                ));
            }
            scratch[..total].fill(S::ZERO);
            for &index in selection.group(0)? {
                if index >= total {
                    return Err(invalid(
                        scioxide_core::argument!("plan"),
                        scioxide_core::InvalidInputReason::OutOfRange,
                        "KS assignment index outside pooled sample",
                    ));
                }
                scratch[index] = S::ONE;
            }
            let (mut x_seen, mut y_seen, mut plus, mut minus) = (0usize, 0usize, 0usize, 0usize);
            for (_, members) in pool.runs() {
                for &(_, index) in members {
                    if scratch[index] == S::ONE {
                        x_seen += 1;
                    } else {
                        y_seen += 1;
                    }
                }
                let a = x_seen * m;
                let b = y_seen * n;
                plus = plus.max(a.saturating_sub(b));
                minus = minus.max(b.saturating_sub(a));
            }
            let value = match alternative {
                TestAlternative::Greater => plus,
                TestAlternative::Less => minus,
                TestAlternative::TwoSided => plus.max(minus),
            };
            S::try_from_f64_lossy(value as f64 / product as f64)
        };
    let report = plan.evaluate_into(
        ResamplingProblem::IndexedWorkspace(&statistic),
        replicates,
        output,
        workspace,
        scope,
    )?;
    let test = if report.termination == StatsTermination::Completed {
        let Some(ResamplingInference::Test(inference)) = report.inference else {
            return Err(invalid(
                scioxide_core::argument!("plan"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "KS permutation completed without inference",
            ));
        };
        let mut test = result(
            report
                .observed
                .ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("plan"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "KS observed statistic missing",
                    )
                })?
                .try_to_f64_exact()?,
            inference.p_value,
            DistributionMethod::Permutation {
                exact: inference.exact,
                replicates: report.completed,
            },
            NullParameters::Exchangeable,
            total,
            2,
        )?;
        test.monte_carlo_interval = inference.tail_uncertainty.map(|v| v[1]);
        Some(test)
    } else {
        None
    };
    Ok(KsPermutationResult {
        test,
        resampling: report,
    })
}
/// Evaluates the permutation test with owned replicate storage.
///
/// # Errors
///
/// Returns allocation errors and the errors of the caller-output form.
pub fn ks_two_sample_permutation_test<S: RealScalar>(
    x: &[S],
    y: &[S],
    alternative: TestAlternative,
    plan: &ResamplingPlan<S>,
    replicates: impl crate::ReplicateExecution<S>,
) -> Result<KsPermutationResult<S>, SciOxideError> {
    let mut output = scioxide_core::backend::try_filled(
        plan.replicates(),
        S::ZERO,
        scioxide_core::quantity!("permutation_output"),
    )?;
    let mut workspace = plan.workspace()?;
    ks_two_sample_permutation_test_into(
        x,
        y,
        alternative,
        plan,
        replicates,
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}

pub(super) fn pooled_ranks<S: RealScalar>(
    x: &[S],
    y: &[S],
    max_bytes: usize,
) -> Result<(Vec<f64>, Vec<bool>, usize), SciOxideError> {
    sample(x, 2, scioxide_core::argument!("x"))?;
    sample(y, 2, scioxide_core::argument!("y"))?;
    let total = x
        .len()
        .checked_add(y.len())
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    let bytes = total
        .checked_mul(
            core::mem::size_of::<(S, usize)>()
                + core::mem::size_of::<f64>()
                + core::mem::size_of::<bool>(),
        )
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    crate::error_helpers::limit(
        bytes,
        max_bytes,
        scioxide_core::quantity!("two_sample_bytes"),
    )?;
    let pool = RankPool::samples(
        &[x, y],
        scioxide_core::quantity!("two_sample_pooled_observations"),
    )?;
    let mut ranks = try_with_capacity(total, scioxide_core::quantity!("pooled_midranks"))?;
    let mut labels = try_with_capacity(total, scioxide_core::quantity!("pooled_labels"))?;
    let mut tied = 0;
    for (run, members) in pool.runs() {
        tied += usize::from(run.len() > 1);
        let rank = run.midrank()?;
        for &(_, group) in members {
            ranks.push(rank);
            labels.push(group == 0);
        }
    }
    Ok((ranks, labels, tied))
}

fn cvm_two_statistic(ranks: &[f64], labels: &[bool], n: usize) -> f64 {
    let m = ranks.len() - n;
    let (mut ix, mut iy, mut u) = (0usize, 0usize, 0.0);
    for (&rank, &is_x) in ranks.iter().zip(labels) {
        if is_x {
            ix += 1;
            u += n as f64 * (rank - ix as f64).powi(2);
        } else {
            iy += 1;
            u += m as f64 * (rank - iy as f64).powi(2);
        }
    }
    let total = (n + m) as f64;
    u / (n * m) as f64 / total - (4 * n * m - 1) as f64 / (6.0 * total)
}

fn cvm_exact_tail(
    ranks: &[f64],
    n: usize,
    observed: f64,
    max_work: usize,
    max_bytes: usize,
) -> Result<f64, SciOxideError> {
    let total = ranks.len();
    let combinations = scioxide_core::checked_binomial(total, n)?;
    crate::error_helpers::work(
        combinations,
        max_work,
        scioxide_core::BudgetKind::Node,
        "exact Cramér–von Mises assignments exceed options.max_work",
    )?;
    let bytes = total
        .checked_mul(core::mem::size_of::<bool>())
        .and_then(|v| v.checked_add(n * core::mem::size_of::<usize>()))
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    crate::error_helpers::limit(
        bytes,
        max_bytes,
        scioxide_core::quantity!("two_sample_bytes"),
    )?;
    let mut chosen = try_with_capacity(n, scioxide_core::quantity!("cram_r_von_mises_labels"))?;
    chosen.extend(0..n);
    let mut labels = try_filled(
        total,
        false,
        scioxide_core::quantity!("cram_r_von_mises_assignment"),
    )?;
    let mut extreme = 0usize;
    loop {
        labels.fill(false);
        for &index in &chosen {
            labels[index] = true;
        }
        let value = cvm_two_statistic(ranks, &labels, n);
        if value + 64.0 * f64::EPSILON * (1.0 + value.abs() + observed.abs()) >= observed {
            extreme += 1;
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
    Ok(extreme as f64 / combinations as f64)
}
/// Performs the two-sample Cramér–von Mises test.
///
/// Uses the configured method with the [one-shot defaults](crate#one-shot-forms).
/// Cost and accuracy are those of [`cramer_von_mises_two_sample_test_with_options`].
///
/// # Errors
///
/// Returns the errors of [`cramer_von_mises_two_sample_test_with_options`] for these defaults.
///
/// ```
/// use scioxide_stats::*;
/// let x = [0.2_f64, 1.0, 1.8, 2.5, 4.0, 5.0, 6.2, 8.0];
/// let y = [0.8_f64, 4.1, 1.3, 7.0, 2.8, 8.2, 4.4, 6.0];
/// let actual = cramer_von_mises_two_sample_test::<f64>(&x, &y)?;
/// let configured = cramer_von_mises_two_sample_test_with_options::<f64>(&x, &y, ClassicalOptions::default())?;
/// assert!((actual.statistic() - configured.statistic()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// T. W. Anderson, "On the distribution of the two-sample Cramér–von Mises
/// criterion", *Ann. Math. Statist.* 33 (1962), 1148–1159.
pub fn cramer_von_mises_two_sample_test<S: RealScalar>(
    x: &[S],
    y: &[S],
) -> Result<DistributionTestResult<S>, SciOxideError> {
    cramer_von_mises_two_sample_test_with_options(x, y, Default::default()).map_err(|error| {
        crate::error_helpers::one_shot_argument(error, &["options"], scioxide_core::argument!("x"))
    })
}
/// Performs the two-sample Cramér–von Mises test.
///
/// ```text
/// T = U / (n m N) − (4 n m − 1) / (6 N),
/// U = n Σᵢ (rᵢ − i)² + m Σⱼ (sⱼ − j)²,
/// ```
///
/// where `rᵢ` and `sⱼ` are the sorted pooled midranks of `x` and `y`, and
/// `N = n + m`. The exact p-value enumerates all `C(N, n)` label assignments
/// in O(C(N, n) N) time and is valid with ties. Under [`TestCalibration::Auto`]
/// it is used when `C(N, n)` is at most [`ClassicalOptions::max_work`].
/// Otherwise the p-value comes from Anderson's approximation to the limiting
/// distribution, which requires untied data.
///
/// # Errors
///
/// - `InvalidInput` if either sample has fewer than two observations, or the
///   alternative is not two-sided.
/// - `InvalidInput` if the large-sample approximation is used on tied data.
/// - `NonFinite` if `x` or `y` contains NaN or infinity.
/// - `ConvergenceFailure` if the calibration is [`TestCalibration::Exact`] and
///   `C(N, n)` exceeds `max_work`.
///
/// # References
///
/// T. W. Anderson, "On the distribution of the two-sample Cramér–von Mises
/// criterion", *Ann. Math. Statist.* 33 (1962), 1148–1159.
pub fn cramer_von_mises_two_sample_test_with_options<S: RealScalar>(
    x: &[S],
    y: &[S],
    options: ClassicalOptions,
) -> Result<DistributionTestResult<S>, SciOxideError> {
    require_omnibus(options)?;
    let n = x.len();
    let m = y.len();
    let (ranks, labels, ties) = pooled_ranks(x, y, options.max_workspace_bytes())?;
    let statistic = finite(cvm_two_statistic(&ranks, &labels, n))?.max(0.0);
    let count = scioxide_core::checked_binomial(n + m, n).ok();
    let exact = options.calibration() == TestCalibration::Exact
        || (options.calibration() == TestCalibration::Auto
            && count.is_some_and(|v| v <= options.max_work()));
    let p = if exact {
        cvm_exact_tail(
            &ranks,
            n,
            statistic,
            options.max_work(),
            options.max_workspace_bytes(),
        )?
    } else {
        if ties > 0 {
            return Err(invalid(
                scioxide_core::argument!("options.calibration"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "tied samples require conditional permutation calibration",
            ));
        }
        let total = (n + m) as f64;
        let product = (n * m) as f64;
        let expected = (1.0 + 1.0 / total) / 6.0;
        let variance = (total + 1.0)
            * (4.0 * product * total - 3.0 * ((n * n + m * m) as f64) - 2.0 * product)
            / (180.0 * total * total * product);
        let transformed = 1.0 / 6.0 + (statistic - expected) / (45.0 * variance).sqrt();
        if transformed <= 0.0 {
            1.0
        } else {
            1.0 - super::one_sample::cvm_cdf_limit(transformed, options.max_work())?
        }
    };
    result(
        statistic,
        p.clamp(0.0, 1.0),
        if exact {
            DistributionMethod::Exact
        } else {
            DistributionMethod::Asymptotic
        },
        NullParameters::Exchangeable,
        n + m,
        2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PermutationMode, ResamplingOptions};
    use scioxide_core::{ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState};
    use scioxide_random::{OutputPolicy, SamplingPolicy, SeedableRng, SplitMix64};
    #[test]
    fn exact_ks_small_paths() {
        let x = [1.0_f64, 2.0];
        let y = [3.0_f64, 4.0];
        let r = ks_two_sample_test_with_options(&x, &y, ClassicalOptions::default()).unwrap();
        assert!((r.statistic - 1.0).abs() < 1e-15);
        assert!((r.p_value - 1.0 / 3.0).abs() < 1e-15);
        assert_eq!(r.method, DistributionMethod::Exact);
        let alias = ks_two_sample_test_with_options(&x, &y, ClassicalOptions::default()).unwrap();
        assert_eq!(r, alias);
        assert!(
            ks_two_sample_test_with_options(&[1.0_f64, 1.0], &y, ClassicalOptions::default())
                .is_err()
        );
        let cvm =
            cramer_von_mises_two_sample_test_with_options(&x, &y, ClassicalOptions::default())
                .unwrap();
        assert!((cvm.statistic - 0.375).abs() < 1e-15);
        assert!((cvm.p_value - 1.0 / 3.0).abs() < 1e-15);
        let x32 = [1.0_f32, 2.0];
        let y32 = [3.0_f32, 4.0];
        let cvm32 =
            cramer_von_mises_two_sample_test_with_options(&x32, &y32, ClassicalOptions::default())
                .unwrap();
        assert!((cvm32.statistic as f64 - 0.375).abs() < 1e-6);
    }

    #[test]
    fn tied_ks_uses_conditional_labels() {
        let x = [1.0_f64, 1.0];
        let y = [1.0_f64, 2.0];
        let plan_with = |options: ResamplingOptions<f64>| {
            ResamplingPlan::new(
                &[2, 2],
                ResamplingMethod::Permutation {
                    kind: PermutationKind::Independent,
                    mode: PermutationMode::Exact,
                    alternative: TestAlternative::Greater,
                },
                options,
            )
            .unwrap()
            .with_statistic_scratch(4)
            .unwrap()
        };
        let plan = plan_with(
            ResamplingOptions::default()
                .with_replicates(6)
                .with_batch_size(1),
        );
        let mut workspace = plan.workspace().unwrap();
        let mut output = [0.0; 6];
        let mut rng = SplitMix64::from_seed(547);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                SolveIdentity::new(
                    ProblemToken::new(547),
                    scioxide_core::algorithm_id!("stats", "ks-permutation-test", 1),
                    0,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        let result = ks_two_sample_permutation_test_into(
            &x,
            &y,
            TestAlternative::TwoSided,
            &plan,
            crate::ReplicateOptions::sequential(
                &mut rng,
                SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
            ),
            &mut output,
            &mut workspace,
            &mut context,
        )
        .unwrap();
        let test = result.test().unwrap();
        assert_eq!(
            test.method(),
            DistributionMethod::Permutation {
                exact: true,
                replicates: 6
            }
        );
        assert!((test.statistic() - 0.5).abs() < 1e-15);
        assert!((test.p_value() - 1.0).abs() < 1e-15);
        assert_eq!(
            result.resampling().termination(),
            StatsTermination::Completed
        );
        // The pooled preparation (four value/position pairs) and the runner
        // workspace are admitted against the plan's own byte budget; there is no
        // separate positional limit.
        let pairs = 4 * core::mem::size_of::<(f64, usize)>() + plan.workspace_bytes().unwrap();
        let tight = plan_with(
            ResamplingOptions::default()
                .with_replicates(6)
                .with_batch_size(1)
                .with_max_workspace_bytes(pairs - 1),
        );
        let mut workspace = tight.workspace().unwrap();
        let error = ks_two_sample_permutation_test_into(
            &x,
            &y,
            TestAlternative::TwoSided,
            &tight,
            crate::ReplicateOptions::sequential(
                &mut rng,
                SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
            ),
            &mut output,
            &mut workspace,
            &mut context,
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ResourceLimit { required, limit, .. }
                if required == pairs && limit == pairs - 1
        ));
    }

    #[test]
    fn exact_ks_tiny_tail_avoids_complement_cancellation() {
        let x: [f64; 30] = core::array::from_fn(|i| i as f64);
        let y: [f64; 30] = core::array::from_fn(|i| (i + 30) as f64);
        let r = ks_two_sample_test_with_options(&x, &y, ClassicalOptions::default()).unwrap();
        let expected = 2.0 / scioxide_core::checked_binomial(60, 30).unwrap() as f64;
        assert!(r.p_value > 0.0);
        assert!((r.p_value / expected - 1.0).abs() < 2e-14);
    }
}
