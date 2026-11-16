//! Resampling functions that allocate their workspace and output.
use super::*;
use scioxide_core::backend::try_filled;
use scioxide_random::SamplingPolicy;

/// Holds replicate statistics and the termination report from [`ResamplingPlan::evaluate`].
#[derive(Debug)]
pub struct ResamplingOutput<S: RealScalar> {
    pub(crate) report: ResamplingResult<S>,
    pub(crate) distribution: Vec<S>,
}
impl<S: RealScalar> ResamplingOutput<S> {
    /// Returns the result, with the termination and the inference.
    pub const fn report(&self) -> &ResamplingResult<S> {
        &self.report
    }
    /// Returns the replicate statistics in replicate order. After an early
    /// stop it holds only the completed replicates. The jackknife deletions of
    /// BCa are not included.
    pub fn distribution(&self) -> &[S] {
        &self.distribution
    }
    /// Returns the result and the replicate statistics.
    pub fn into_parts(self) -> (ResamplingResult<S>, Vec<S>) {
        (self.report, self.distribution)
    }
}
impl<S: RealScalar> ResamplingPlan<S> {
    /// Evaluates the prepared resampling design with owned storage.
    ///
    /// Returns completed replicate values and the complete termination report.
    /// The computation and accuracy are those of [`Self::evaluate_into`].
    ///
    /// # Errors
    ///
    /// Returns preparation, allocation, callback and execution errors.
    pub fn evaluate(
        &self,
        problem: ResamplingProblem<'_, S>,
        replicates: impl crate::ReplicateExecution<S>,
    ) -> Result<ResamplingOutput<S>, SciOxideError> {
        let mut workspace = self.workspace()?;
        let mut distribution = try_filled(
            self.replicates(),
            S::ZERO,
            scioxide_core::quantity!("owned_resampling_output"),
        )?;
        // The owned result is unpublished until return, so staging policies are observationally equivalent on errors.
        let report = self.evaluate_into(
            problem,
            replicates,
            &mut distribution,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        if report.termination != StatsTermination::Completed {
            distribution[..report.completed]
                .copy_from_slice(&workspace.principal_values()[..report.completed]);
            distribution.truncate(report.completed);
        }
        Ok(ResamplingOutput {
            report,
            distribution,
        })
    }
}
fn inference<S: RealScalar>(
    output: ResamplingOutput<S>,
) -> Result<ResamplingInference<S>, SciOxideError> {
    if let Some(error) = output.report.termination.one_shot_error(
        scioxide_core::ConvergenceReason::ResourceLimit(scioxide_core::BudgetKind::Sample),
        scioxide_core::ConvergenceDiagnostics::new(),
    ) {
        return Err(error);
    }
    output.report.inference.ok_or_else(|| {
        invalid(
            scioxide_core::argument!("statistic"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "completed resampling has no inference",
        )
    })
}

/// Computes a two-sided 95% BCa bootstrap interval.
///
/// Uses the [one-shot defaults](crate#one-shot-forms), with no allocation per callback.
/// The plan evaluates 9999 replicates; accuracy follows the returned inference type.
///
/// # Errors
///
/// Rejects invalid samples, propagates callback errors, and fails if inference
/// cannot be formed or the run stops before completion.
///
/// # References
///
/// A. C. Davison and D. V. Hinkley, Bootstrap Methods and Their Application (1997), chapters 4–5.
///
/// ```
/// use scioxide_random::{SeedableRng, SplitMix64};
/// use scioxide_stats::{bootstrap, mean};
/// let mut rng = SplitMix64::from_seed(7);
/// let interval = bootstrap(mean, &[2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], &mut rng)?;
/// assert!(interval.bounds.contains(5.0));
/// assert!((interval.standard_error - 2.0 / 8.0_f64.sqrt()).abs() < 0.03);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn bootstrap<S, F, R>(
    statistic: F,
    sample: &[S],
    rng: &mut R,
) -> Result<BootstrapInterval<S>, SciOxideError>
where
    S: RealScalar,
    F: Fn(&[S]) -> Result<S, SciOxideError> + Sync,
    R: scioxide_random::BitGenerator,
{
    let plan = ResamplingPlan::new(
        &[sample.len()],
        ResamplingMethod::Bootstrap {
            design: BootstrapDesign::Independent,
            method: BootstrapMethod::Bca,
            confidence: crate::ConfidenceProbability::default(),
        },
        ResamplingOptions::default(),
    )
    .map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["lengths", "method", "problem", "options"],
            scioxide_core::argument!("sample"),
        )
    })?
    .with_statistic_scratch(sample.len())?;
    let adapter = |indices: ResampledIndices<'_>, scratch: &mut [S]| {
        for (out, &index) in scratch.iter_mut().zip(indices.indices()) {
            *out = sample[index];
        }
        statistic(&scratch[..indices.indices().len()])
    };

    let output = plan.evaluate(
        ResamplingProblem::IndexedWorkspace(&adapter),
        crate::ReplicateOptions::sequential(
            rng,
            SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic),
        ),
    )?;
    match inference(output)? {
        ResamplingInference::Bootstrap(value) => Ok(value),
        _ => Err(invalid(
            scioxide_core::argument!("statistic"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "unexpected resampling inference",
        )),
    }
}

/// Computes delete-one jackknife bias and standard error.
///
/// Uses the [one-shot defaults](crate#one-shot-forms), with no allocation per callback.
/// The plan evaluates n deletions; accuracy follows the returned inference type.
///
/// # Errors
///
/// Rejects invalid samples, propagates callback errors, and fails if inference
/// cannot be formed or the run stops before completion.
///
/// # References
///
/// A. C. Davison and D. V. Hinkley, Bootstrap Methods and Their Application (1997), chapters 4–5.
///
/// ```
/// use scioxide_stats::{jackknife, mean};
/// let estimate = jackknife(mean, &[1.0_f64, 2.0, 3.0])?;
/// assert!(estimate.bias.abs() < 1e-14);
/// assert!((estimate.standard_error - (1.0_f64 / 3.0).sqrt()).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn jackknife<S, F>(statistic: F, sample: &[S]) -> Result<JackknifeEstimate<S>, SciOxideError>
where
    S: RealScalar,
    F: Fn(&[S]) -> Result<S, SciOxideError> + Sync,
{
    let plan = ResamplingPlan::new(
        &[sample.len()],
        ResamplingMethod::Jackknife {
            design: BootstrapDesign::Independent,
        },
        ResamplingOptions::default().with_replicates(sample.len()),
    )
    .map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["lengths", "method", "problem", "options"],
            scioxide_core::argument!("sample"),
        )
    })?
    .with_statistic_scratch(sample.len())?;
    let adapter = |indices: ResampledIndices<'_>, scratch: &mut [S]| {
        for (out, &index) in scratch.iter_mut().zip(indices.indices()) {
            *out = sample[index];
        }
        statistic(&scratch[..indices.indices().len()])
    };
    let mut rng = runner::NoRandom(scioxide_core::argument!("statistic"));
    let output = plan.evaluate(
        ResamplingProblem::IndexedWorkspace(&adapter),
        crate::ReplicateOptions::sequential(
            &mut rng,
            SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic),
        ),
    )?;
    match inference(output)? {
        ResamplingInference::Jackknife(value) => Ok(value),
        _ => Err(invalid(
            scioxide_core::argument!("statistic"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "unexpected resampling inference",
        )),
    }
}

/// Computes a two-sided randomized permutation-test p-value.
///
/// Uses the [one-shot defaults](crate#one-shot-forms), with no allocation per callback.
/// The plan evaluates 9999 replicates; accuracy follows the returned inference type.
///
/// # Errors
///
/// Rejects invalid samples, propagates callback errors, and fails if inference
/// cannot be formed or the run stops before completion.
///
/// # References
///
/// A. C. Davison and D. V. Hinkley, Bootstrap Methods and Their Application (1997), chapters 4–5.
pub fn permutation_test<S, F, R>(
    statistic: F,
    x: &[S],
    y: &[S],
    rng: &mut R,
) -> Result<ResamplingTest, SciOxideError>
where
    S: RealScalar,
    F: Fn(&[S], &[S]) -> Result<S, SciOxideError> + Sync,
    R: scioxide_random::BitGenerator,
{
    let plan = ResamplingPlan::new(
        &[x.len(), y.len()],
        ResamplingMethod::Permutation {
            kind: PermutationKind::Independent,
            mode: PermutationMode::Randomized,
            alternative: TestAlternative::TwoSided,
        },
        ResamplingOptions::default(),
    )
    .map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["lengths", "method", "problem", "options"],
            scioxide_core::argument!("x"),
        )
    })?
    .with_statistic_scratch(x.len().checked_add(y.len()).ok_or_else(|| {
        SciOxideError::size_overflow(scioxide_core::quantity!("resampling_sample"))
    })?)?;
    let adapter = |indices: ResampledIndices<'_>, scratch: &mut [S]| {
        for (out, &index) in scratch.iter_mut().zip(indices.indices()) {
            *out = if index < x.len() {
                x[index]
            } else {
                y[index - x.len()]
            }
        }
        let first = indices.group(0)?.len();
        statistic(&scratch[..first], &scratch[first..indices.indices().len()])
    };

    let output = plan.evaluate(
        ResamplingProblem::IndexedWorkspace(&adapter),
        crate::ReplicateOptions::sequential(
            rng,
            SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic),
        ),
    )?;
    match inference(output)? {
        ResamplingInference::Test(value) => Ok(value),
        _ => Err(invalid(
            scioxide_core::argument!("statistic"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "unexpected resampling inference",
        )),
    }
}

/// Computes a two-sided Monte Carlo p-value.
///
/// Uses the [one-shot defaults](crate#one-shot-forms), with no allocation per callback.
/// The plan evaluates 9999 replicates; accuracy follows the returned inference type.
///
/// # Errors
///
/// Rejects invalid samples, propagates callback errors, and fails if inference
/// cannot be formed or the run stops before completion.
///
/// # References
///
/// A. C. Davison and D. V. Hinkley, Bootstrap Methods and Their Application (1997), chapters 4–5.
pub fn monte_carlo_test<S, F, G, R>(
    statistic: F,
    sampler: G,
    observed: &[S],
    rng: &mut R,
) -> Result<ResamplingTest, SciOxideError>
where
    S: RealScalar,
    F: Fn(&[S]) -> Result<S, SciOxideError> + Sync,
    G: Fn(&mut dyn Rng<S>, &mut [S]) -> Result<(), SciOxideError> + Sync,
    R: scioxide_random::BitGenerator,
{
    let plan = ResamplingPlan::new(
        &[observed.len()],
        ResamplingMethod::MonteCarlo {
            alternative: TestAlternative::TwoSided,
        },
        ResamplingOptions::default(),
    )
    .map_err(|error| {
        crate::error_helpers::one_shot_argument(
            error,
            &["lengths", "method", "problem", "options"],
            scioxide_core::argument!("observed"),
        )
    })?;

    let output = plan.evaluate(
        ResamplingProblem::MonteCarlo {
            observed,
            sampler: &sampler,
            statistic: &statistic,
        },
        crate::ReplicateOptions::sequential(
            rng,
            SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic),
        ),
    )?;
    match inference(output)? {
        ResamplingInference::Test(value) => Ok(value),
        _ => Err(invalid(
            scioxide_core::argument!("statistic"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "unexpected resampling inference",
        )),
    }
}
