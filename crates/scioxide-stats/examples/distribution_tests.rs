//! Fixed and fitted distribution nulls with explicit Monte Carlo resources.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_random::{OutputPolicy, Rng, SamplingPolicy, SeedableRng, SplitMix64};
use scioxide_stats::{
    ClassicalOptions, GoodnessOfFitModel, GoodnessOfFitStatistic, ResamplingMethod,
    ResamplingOptions, ResamplingPlan, TestAlternative, bartlett_test_with_options,
    goodness_of_fit_test_into, ks_one_sample_test_with_options, shapiro_wilk_test_with_options,
};

fn main() -> Result<(), SciOxideError> {
    let sample = [0.1_f64, 0.2, 0.4, 0.8];
    let fixed = ks_one_sample_test_with_options(&sample, Ok, ClassicalOptions::default())?;
    println!(
        "fixed uniform KS: D={}, p={}",
        fixed.statistic(),
        fixed.p_value()
    );

    let normality = shapiro_wilk_test_with_options(
        &[1.0_f64, 2.0, 2.5, 3.0, 4.0],
        ClassicalOptions::default(),
    )?;
    println!(
        "Shapiro W={}, p={}",
        normality.statistic(),
        normality.p_value()
    );
    let variance = bartlett_test_with_options(
        &[&[1.0_f64, 2.0, 3.0], &[4.0, 5.0, 6.0]],
        ClassicalOptions::default(),
    )?;
    println!(
        "Bartlett statistic={}, p={}",
        variance.statistic(),
        variance.p_value()
    );

    // Composite Uniform(0, upper) null. `fit` re-estimates upper on the observed
    // sample and every generated sample; the sampler uses the observed estimate.
    let fit = |observations: &[f64], parameters: &mut [f64], _scratch: &mut [f64]| {
        parameters[0] = observations.iter().copied().fold(0.0, f64::max);
        Ok(())
    };
    let model = GoodnessOfFitModel {
        parameters: &[1.0],
        fit: Some(&fit),
        log_cdf: &|x: f64, p: &[f64]| Ok((x / p[0]).ln()),
        log_survival: &|x: f64, p: &[f64]| Ok((1.0 - x / p[0]).ln()),
        sampler: &|rng: &mut dyn Rng<f64>, p: &[f64], out: &mut [f64]| {
            for value in out {
                *value = rng.next_float()? * p[0];
            }
            Ok(())
        },
        fit_scratch_elements: 0,
    };
    let plan = ResamplingPlan::new(
        &[sample.len()],
        ResamplingMethod::MonteCarlo {
            alternative: TestAlternative::Greater,
        },
        ResamplingOptions::default()
            .with_replicates(199)
            .with_batch_size(1),
    )?
    .with_statistic_scratch(sample.len() + 1)?;
    let mut workspace = plan.workspace()?;
    let mut output = [0.0_f64; 199];
    let mut fitted = [0.0_f64];
    let mut rng = SplitMix64::from_seed(547);

    let calibrated = goodness_of_fit_test_into(
        &sample,
        GoodnessOfFitStatistic::KolmogorovSmirnov,
        model,
        &plan,
        scioxide_stats::ReplicateOptions::sequential(
            &mut rng,
            SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
        ),
        &mut fitted,
        &mut output,
        &mut [],
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    if let Some(test) = calibrated.test() {
        println!(
            "fitted uniform upper={}, KS p={}",
            fitted[0],
            test.p_value()
        );
    } else {
        println!(
            "Monte Carlo stopped after {} draws",
            calibrated.resampling().completed()
        );
    }
    Ok(())
}
