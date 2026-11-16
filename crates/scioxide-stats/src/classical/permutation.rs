//! Permutation p-values for classical test statistics.
use super::*;
use crate::{
    IndexStatistic, PermutationKind, PermutationMode, ResamplingInference, ResamplingMethod,
    ResamplingOptions, ResamplingPlan, ResamplingProblem, ResamplingResult, ResamplingWorkspace,
    StatsTermination,
};
use scioxide_random::SamplingPolicy;

/// Result of [`classical_permutation_test`]: the test result with a permutation
/// p-value, and the resampling report.
///
/// The test result is present exactly when the resampling run produced a
/// p-value.
#[derive(Debug, Clone, Copy)]
pub struct ClassicalPermutationResult<S: RealScalar> {
    pub(crate) test: Option<ClassicalTestResult<S>>,
    pub(crate) resampling: ResamplingResult<S>,
}
impl<S: RealScalar> ClassicalPermutationResult<S> {
    /// Returns the test result with the permutation p-value and
    /// [`ClassicalMethod::Permutation`]. It is `None` if the run stopped early,
    /// so a p-value from an incomplete run is never reported.
    pub const fn test(&self) -> Option<&ClassicalTestResult<S>> {
        self.test.as_ref()
    }
    /// Returns the resampling report: termination status, tail counts and Monte
    /// Carlo uncertainty.
    pub const fn resampling(&self) -> &ResamplingResult<S> {
        &self.resampling
    }
}

fn completed<S: RealScalar>(
    base: ClassicalTestResult<S>,
    report: ResamplingResult<S>,
) -> Result<ClassicalPermutationResult<S>, SciOxideError> {
    let test = if let Some(ResamplingInference::Test(inference)) = report.inference {
        let mut test = base;
        test.p_value = inference.p_value;
        test.method = ClassicalMethod::Permutation {
            exact: inference.exact,
            replicates: report.completed,
        };
        Some(test)
    } else {
        None
    };
    Ok(ClassicalPermutationResult {
        test,
        resampling: report,
    })
}

/// Replaces the p-value of a classical test result with a permutation p-value.
///
/// `base` is a result from one of the classical tests. `statistic` must compute
/// the same statistic as `base` from the permuted group indices, under the
/// exchangeability model of `plan`. On the identity permutation it must return
/// exactly `base.statistic()`. Use an upper-tail plan
/// ([`TestAlternative::Greater`](crate::TestAlternative::Greater)) for omnibus
/// statistics such as BWS. `replicates`, `output`, `workspace` and `scope`
/// have the same meaning as in [`ResamplingPlan::evaluate_into`].
///
/// # Errors
///
/// - `InvalidInput` if `plan` is not a permutation plan.
/// - `InvalidInput` if `statistic` on the identity permutation differs from
///   `base.statistic()`.
/// - Any error of [`ResamplingPlan::evaluate_into`] or of `statistic`.
#[allow(clippy::too_many_arguments)]
pub fn classical_permutation_test_into<'scope, 'capability, S: RealScalar>(
    base: ClassicalTestResult<S>,
    plan: &ResamplingPlan<S>,
    statistic: &IndexStatistic<'_, S>,
    replicates: impl crate::ReplicateExecution<S>,
    output: &mut [S],
    workspace: &mut ResamplingWorkspace<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<ClassicalPermutationResult<S>, SciOxideError> {
    workspace.reject_attempt();
    if !matches!(plan.method(), ResamplingMethod::Permutation { .. }) {
        return Err(invalid(
            scioxide_core::argument!("plan"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "classical permutation requires a permutation plan",
        ));
    }
    let checked_statistic = |indices: crate::ResampledIndices<'_>| {
        let value = statistic(indices)?;
        if indices.indices().iter().enumerate().all(|(i, &j)| i == j) && value != base.statistic {
            return Err(invalid(
                scioxide_core::argument!("statistic"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "base and permutation observed statistic differ",
            ));
        }
        Ok(value)
    };
    let report = plan.evaluate_into(
        ResamplingProblem::Indexed(&checked_statistic),
        replicates,
        output,
        workspace,
        scope,
    )?;
    completed(base, report)
}
/// Evaluates the permutation test with owned replicate storage.
///
/// # Errors
///
/// Returns allocation errors and the errors of the caller-output form.
pub fn classical_permutation_test<S: RealScalar>(
    base: ClassicalTestResult<S>,
    plan: &ResamplingPlan<S>,
    statistic: &IndexStatistic<'_, S>,
    replicates: impl crate::ReplicateExecution<S>,
) -> Result<ClassicalPermutationResult<S>, SciOxideError> {
    let mut output = scioxide_core::backend::try_filled(
        plan.replicates(),
        S::ZERO,
        scioxide_core::quantity!("permutation_output"),
    )?;
    let mut workspace = plan.workspace()?;
    classical_permutation_test_into(
        base,
        plan,
        statistic,
        replicates,
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
}

pub(super) fn exact(
    groups: &[usize],
    kind: PermutationKind,
    options: ClassicalOptions,
    statistic: &IndexStatistic<'_, f64>,
) -> Result<(f64, f64), SciOxideError> {
    let plan = ResamplingPlan::new(
        groups,
        ResamplingMethod::Permutation {
            kind,
            mode: PermutationMode::Exact,
            alternative: options.alternative,
        },
        (ResamplingOptions::default()
            .with_replicates(options.max_work)
            .with_batch_size(1))
        .with_max_workspace_bytes(options.max_workspace_bytes),
    )?;
    // Exact plans do not draw; use the runner's deterministic provider.
    let report = plan
        .evaluate(
            crate::ResamplingProblem::Indexed(statistic),
            crate::ReplicateOptions::sequential(
                &mut crate::resampling::NoRandom(scioxide_core::argument!("statistic")),
                SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic),
            ),
        )?
        .report;
    match (report.termination, report.observed, report.inference) {
        (StatsTermination::Completed, Some(stat), Some(ResamplingInference::Test(test))) => {
            Ok((stat, test.p_value))
        }
        _ => Err(invalid(
            scioxide_core::argument!("options.max_work"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "exact inference did not complete",
        )),
    }
}

pub(super) fn use_exact(
    count: Result<usize, SciOxideError>,
    options: ClassicalOptions,
) -> Result<bool, SciOxideError> {
    if options.calibration == TestCalibration::Asymptotic {
        return Ok(false);
    }
    match count {
        Ok(n) if n <= options.max_work => Ok(true),
        _ if options.calibration == TestCalibration::Auto => Ok(false),
        Ok(_) => Err(crate::error_helpers::work_exhausted(
            options.max_work,
            scioxide_core::BudgetKind::Node,
            "exact null states exceed options.max_work",
        )),
        Err(error) => Err(error),
    }
}
