//! Goodness-of-fit tests with parametric bootstrap p-values.
use super::*;
use crate::{
    ResamplingInference, ResamplingMethod, ResamplingPlan, ResamplingProblem, ResamplingResult,
    ResamplingWorkspace, StatsTermination,
};

use scioxide_random::Rng;

/// Selects the EDF statistic of [`goodness_of_fit_test_into`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoodnessOfFitStatistic {
    /// Kolmogorov–Smirnov `D`: the largest absolute difference between the
    /// empirical and model CDFs.
    KolmogorovSmirnov,
    /// Cramér–von Mises `W²`: the integrated squared difference.
    CramerVonMises,
    /// Anderson–Darling `A²`: the squared difference weighted by
    /// `1/(F(1 − F))`, which emphasizes the tails.
    AndersonDarling,
}

/// Parametric distribution family tested by [`goodness_of_fit_test_into`].
///
/// The family is given by its log-CDF, log-survival function and sampler, each
/// taking a parameter vector. Without `fit` the null hypothesis is the single
/// distribution with `parameters`. With `fit` the parameters are estimated from
/// each sample. `fit` receives a copy of `parameters` and must overwrite every
/// estimated component. It must give the same result regardless of the number
/// of worker threads. The callbacks may allocate; their memory does not count
/// toward any workspace limit.
#[derive(Clone, Copy)]
#[expect(
    clippy::type_complexity,
    reason = "preserve the public callback field signatures"
)]
pub struct GoodnessOfFitModel<'a, S: RealScalar> {
    /// Known parameter values and starting guesses for estimated ones.
    pub parameters: &'a [S],
    /// Estimates the parameters from a sample: `fit(sample, parameters, scratch)`.
    /// `None` tests a fully specified distribution.
    pub fit: Option<&'a (dyn Fn(&[S], &mut [S], &mut [S]) -> Result<(), SciOxideError> + Sync)>,
    /// Returns `ln F(x; θ)` for a value `x` and parameters `θ`.
    pub log_cdf: &'a (dyn Fn(S, &[S]) -> Result<S, SciOxideError> + Sync),
    /// Returns `ln(1 − F(x; θ))`. Only the Anderson–Darling statistic uses it.
    pub log_survival: &'a (dyn Fn(S, &[S]) -> Result<S, SciOxideError> + Sync),
    /// Fills the output slice with one sample drawn from the distribution with
    /// the given parameters.
    pub sampler: &'a (dyn Fn(&mut dyn Rng<S>, &[S], &mut [S]) -> Result<(), SciOxideError> + Sync),
    /// Number of scratch entries `fit` needs per call.
    pub fit_scratch_elements: usize,
}

/// Result of [`goodness_of_fit_test_into`]: the test result and the resampling
/// report.
///
/// The test result is present exactly when every replicate was evaluated.
#[derive(Clone, Copy, Debug)]
pub struct GoodnessOfFitResult<S: RealScalar> {
    test: Option<DistributionTestResult<S>>,
    resampling: ResamplingResult<S>,
}
impl<S: RealScalar> GoodnessOfFitResult<S> {
    /// Returns the test result, or `None` if the run stopped before every
    /// replicate was evaluated.
    pub const fn test(&self) -> Option<&DistributionTestResult<S>> {
        self.test.as_ref()
    }
    /// Returns the resampling report: termination status, replicate counts and
    /// Monte Carlo uncertainty.
    pub const fn resampling(&self) -> &ResamplingResult<S> {
        &self.resampling
    }
}

fn prepare_model<S: RealScalar>(
    observed: &[S],
    model: GoodnessOfFitModel<'_, S>,
    fitted: &mut [S],
    work: &mut [S],
    plan: &ResamplingPlan<S>,
) -> Result<(), SciOxideError> {
    sample(observed, 1, scioxide_core::argument!("observed"))?;
    if !matches!(
        plan.method(),
        ResamplingMethod::MonteCarlo {
            alternative: TestAlternative::Greater
        }
    ) {
        return Err(invalid(
            scioxide_core::argument!("plan"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "GOF requires an upper-tail Monte Carlo plan",
        ));
    }
    if fitted.len() != model.parameters.len() || work.len() < model.fit_scratch_elements {
        return Err(invalid(
            scioxide_core::argument!("fitted"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "GOF parameter or fitting scratch length mismatch",
        ));
    }
    let needed = observed
        .len()
        .checked_add(model.parameters.len())
        .and_then(|v| v.checked_add(model.fit_scratch_elements))
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    if plan.statistic_scratch_elements() < needed {
        return Err(invalid(
            scioxide_core::argument!("plan"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "GOF plan scratch too short",
        ));
    }
    fitted.copy_from_slice(model.parameters);
    if let Some(fit) = model.fit {
        fit(observed, fitted, &mut work[..model.fit_scratch_elements])?;
    }
    fitted.iter().try_for_each(|p| {
        if p.is_finite() {
            Ok(())
        } else {
            Err(invalid(
                scioxide_core::argument!("model"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "fitter returned nonfinite parameter",
            ))
        }
    })
}

fn finished<S: RealScalar>(
    report: ResamplingResult<S>,
    model: GoodnessOfFitModel<'_, S>,
    n: usize,
) -> Result<GoodnessOfFitResult<S>, SciOxideError> {
    let test = if report.termination == StatsTermination::Completed {
        let Some(ResamplingInference::Test(inference)) = report.inference else {
            return Err(invalid(
                scioxide_core::argument!("plan"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "GOF completed without test inference",
            ));
        };
        let mut r = result(
            report
                .observed
                .ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("plan"),
                        scioxide_core::InvalidInputReason::Inconsistent,
                        "GOF missing observed statistic",
                    )
                })?
                .try_to_f64_exact()?,
            inference.p_value,
            DistributionMethod::MonteCarlo {
                replicates: report.completed,
            },
            if model.fit.is_some() {
                NullParameters::Estimated
            } else {
                NullParameters::Known
            },
            n,
            1,
        )?;
        r.monte_carlo_interval = inference.tail_uncertainty.map(|v| v[1]);
        Some(r)
    } else {
        None
    };
    Ok(GoodnessOfFitResult {
        test,
        resampling: report,
    })
}

fn statistic<S: RealScalar>(
    sample: &[S],
    scratch: &mut [S],
    initial: &[S],
    model: GoodnessOfFitModel<'_, S>,
    kind: GoodnessOfFitStatistic,
) -> Result<S, SciOxideError> {
    let needed = sample
        .len()
        .checked_add(initial.len())
        .and_then(|v| v.checked_add(model.fit_scratch_elements))
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("statistics_size")))?;
    if scratch.len() < needed {
        return Err(invalid(
            scioxide_core::argument!("plan"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "GOF statistic scratch too short",
        ));
    }
    let (ordered, rest) = scratch.split_at_mut(sample.len());
    let (params, fit_work) = rest.split_at_mut(initial.len());
    params.copy_from_slice(initial);
    if let Some(fit) = model.fit {
        fit(sample, params, &mut fit_work[..model.fit_scratch_elements])?;
    }
    params.iter().try_for_each(|p| {
        if p.is_finite() {
            Ok(())
        } else {
            Err(invalid(
                scioxide_core::argument!("model"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "fitter returned nonfinite parameter",
            ))
        }
    })?;
    ordered.copy_from_slice(sample);
    crate::ranking::sort_finite(ordered);
    let n = ordered.len() as f64;
    let (mut plus, mut minus, mut cvm, mut ad) = (0.0_f64, 0.0_f64, 1.0 / (12.0 * n), 0.0_f64);
    let mut previous = 0.0;
    for (i, &value) in ordered.iter().enumerate() {
        let lo = (model.log_cdf)(value, params)?.into_f64();
        if lo > 0.0 || lo.is_nan() {
            return Err(invalid(
                scioxide_core::argument!("model"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "invalid log-CDF tail",
            ));
        }
        let p = lo.exp();
        if p < previous {
            return Err(invalid(
                scioxide_core::argument!("model"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "CDF is not monotone on observations",
            ));
        }
        previous = p;
        plus = plus.max((i + 1) as f64 / n - p);
        minus = minus.max(p - i as f64 / n);
        cvm += (p - (i as f64 + 0.5) / n).powi(2);
        if kind == GoodnessOfFitStatistic::AndersonDarling {
            let hi = (model.log_survival)(ordered[ordered.len() - 1 - i], params)?.into_f64();
            if !lo.is_finite() || !hi.is_finite() || hi > 0.0 {
                return Err(invalid(
                    scioxide_core::argument!("model"),
                    scioxide_core::InvalidInputReason::OutOfRange,
                    "Anderson log-CDF tails must be finite and nonpositive",
                ));
            }
            ad += (2 * i + 1) as f64 * (lo + hi);
        }
    }
    let value = match kind {
        GoodnessOfFitStatistic::KolmogorovSmirnov => plus.max(minus),
        GoodnessOfFitStatistic::CramerVonMises => cvm,
        GoodnessOfFitStatistic::AndersonDarling => -n - ad / n,
    };
    S::try_from_f64_lossy(finite(value)?)
}

/// Tests whether `observed` comes from a continuous parametric family, with a
/// parametric bootstrap p-value.
///
/// The parameters `θ̂` are fitted to `observed` (or taken from
/// `model.parameters` when `model.fit` is `None`) and written to
/// `fitted_parameters`. Each replicate draws a sample of the same size from
/// `F(·; θ̂)`, refits the parameters to it, and computes the statistic selected
/// by `kind`. The p-value is the Monte Carlo estimate of the probability that
/// a replicate is at least as large as the observed statistic. Refitting on
/// every replicate makes the p-value valid when parameters are estimated,
/// where the fixed-parameter [`ks_one_sample_test_with_options`] and
/// [`cramer_von_mises_test_with_options`] are not.
///
/// `plan` must be a Monte Carlo plan with the upper-tail alternative over one
/// group of `observed.len()` values. Its statistic scratch must hold at least
/// `observed.len() + parameters.len() + fit_scratch_elements` entries.
/// `fit_scratch` holds at least `fit_scratch_elements` entries for the initial
/// fit. `replicates`, `output`, `workspace` and `scope` have the same meaning
/// as in [`ResamplingPlan::evaluate_into`]. Each replicate costs O(n log n) time plus
/// one fit.
///
/// # Errors
///
/// - `InvalidInput` if `observed` is empty.
/// - `InvalidInput` if `plan` is not an upper-tail Monte Carlo plan or its
///   statistic scratch is too short.
/// - `InvalidInput` if `fitted_parameters.len() != parameters.len()` or
///   `fit_scratch` is too short.
/// - `InvalidInput` if `fit` returns a non-finite parameter, or `log_cdf`
///   returns a positive or NaN value or decreases along a sorted sample.
/// - `InvalidInput` for Anderson–Darling if a log tail is not finite.
/// - `NonFinite` if `observed` contains NaN or infinity.
/// - Any error of [`ResamplingPlan::evaluate_into`] or of the model callbacks.
#[allow(clippy::too_many_arguments)]
pub fn goodness_of_fit_test_into<'scope, 'capability, S: RealScalar>(
    observed: &[S],
    kind: GoodnessOfFitStatistic,
    model: GoodnessOfFitModel<'_, S>,
    plan: &ResamplingPlan<S>,
    replicates: impl crate::ReplicateExecution<S>,
    fitted_parameters: &mut [S],
    output: &mut [S],
    fit_scratch: &mut [S],
    workspace: &mut ResamplingWorkspace<S>,
    scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
) -> Result<GoodnessOfFitResult<S>, SciOxideError> {
    workspace.reject_attempt();
    prepare_model(observed, model, fitted_parameters, fit_scratch, plan)?;
    let fitted: &[S] = fitted_parameters;
    let sampler = |random: &mut dyn Rng<S>, out: &mut [S]| (model.sampler)(random, fitted, out);
    let statistic =
        |sample: &[S], scratch: &mut [S]| statistic(sample, scratch, model.parameters, model, kind);
    let report = plan.evaluate_into(
        ResamplingProblem::MonteCarloWorkspace {
            observed,
            sampler: &sampler,
            statistic: &statistic,
        },
        replicates,
        output,
        workspace,
        scope,
    )?;
    finished(report, model, observed.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::ExecutionPolicy;
    use scioxide_random::{OutputPolicy, SamplingPolicy};

    use crate::ResamplingOptions;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use scioxide_core::{ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState};
    use scioxide_random::{Philox4x64, SeedableRng, SplitMix64};

    #[test]
    fn fitted_null_refits_every_replicate() {
        let observed = [0.1_f64, 0.2, 0.4, 0.8];
        let fits = AtomicUsize::new(0);
        let fit = |sample: &[f64], parameters: &mut [f64], _scratch: &mut [f64]| {
            fits.fetch_add(1, Ordering::Relaxed);
            parameters[0] = sample.iter().copied().fold(0.0, f64::max);
            Ok(())
        };
        let cdf = |x: f64, p: &[f64]| Ok((x / p[0]).ln());
        let survival = |x: f64, p: &[f64]| Ok((1.0 - x / p[0]).ln());
        let sampler = |rng: &mut dyn Rng<f64>, p: &[f64], out: &mut [f64]| {
            for value in out {
                *value = rng.next_float()? * p[0];
            }
            Ok(())
        };
        let model = GoodnessOfFitModel {
            parameters: &[1.0],
            fit: Some(&fit),
            log_cdf: &cdf,
            log_survival: &survival,
            sampler: &sampler,
            fit_scratch_elements: 0,
        };
        let plan = ResamplingPlan::new(
            &[observed.len()],
            ResamplingMethod::MonteCarlo {
                alternative: TestAlternative::Greater,
            },
            ResamplingOptions::default()
                .with_replicates(9)
                .with_batch_size(1),
        )
        .unwrap()
        .with_statistic_scratch(observed.len() + 1)
        .unwrap();
        let mut workspace = plan.workspace().unwrap();
        let mut output = [0.0; 9];
        let mut fitted = [0.0];
        let mut rng = SplitMix64::from_seed(547);
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                SolveIdentity::new(
                    ProblemToken::new(547),
                    scioxide_core::algorithm_id!("stats", "gof-test", 1),
                    0,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        let report = goodness_of_fit_test_into(
            &observed,
            GoodnessOfFitStatistic::KolmogorovSmirnov,
            model,
            &plan,
            crate::ReplicateOptions::sequential(
                &mut rng,
                SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
            ),
            &mut fitted,
            &mut output,
            &mut [],
            &mut workspace,
            &mut context,
        )
        .unwrap();
        assert_eq!(fits.load(Ordering::Relaxed), 11);
        assert_eq!(report.resampling.completed, 9);
        assert_eq!(
            report.test.unwrap().method,
            DistributionMethod::MonteCarlo { replicates: 9 }
        );
        assert!(report.test.unwrap().p_value >= 0.1);
        let root = Philox4x64::from_seed(547);
        let mut indexed_output = [0.0; 9];
        let mut indexed_workspace = plan.workspace().unwrap();
        let mut indexed_state = SolveState::new();
        let mut indexed_budgets = SolveBudgets::unbounded();
        let mut indexed_context = indexed_state
            .begin_solve(
                ExecutionContext::sequential(),
                SolveIdentity::new(
                    ProblemToken::new(548),
                    scioxide_core::algorithm_id!("stats", "gof-indexed-test", 1),
                    0,
                    0,
                ),
                &mut indexed_budgets,
            )
            .unwrap();
        let indexed = goodness_of_fit_test_into(
            &observed,
            GoodnessOfFitStatistic::KolmogorovSmirnov,
            model,
            &plan,
            crate::ReplicateOptions::indexed(
                &root,
                100,
                ExecutionPolicy::Sequential,
                scioxide_random::OutputPolicy::Atomic,
            ),
            &mut fitted,
            &mut indexed_output,
            &mut [],
            &mut indexed_workspace,
            &mut indexed_context,
        )
        .unwrap();
        assert_eq!(fits.load(Ordering::Relaxed), 22);
        assert_eq!(indexed.resampling.completed, 9);
        assert!(indexed.test.unwrap().p_value >= 0.1);
        let model32 = GoodnessOfFitModel::<f32> {
            parameters: &[],
            fit: None,
            log_cdf: &|x: f32, _: &[f32]| Ok(x.ln()),
            log_survival: &|x: f32, _: &[f32]| Ok((1.0 - x).ln()),
            sampler: &|_, _, _| Ok(()),
            fit_scratch_elements: 0,
        };
        let mut scratch32 = [0.0_f32; 3];
        let cvm32 = statistic(
            &[0.2_f32, 0.5, 0.8],
            &mut scratch32,
            &[],
            model32,
            GoodnessOfFitStatistic::CramerVonMises,
        )
        .unwrap();
        assert!((cvm32 - 0.03).abs() < 2e-6);
    }
}
