//! Small model power, Sobol and posterior analysis example.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_linalg::MatrixView;
use scioxide_random::{Philox4x64, QmcPlan, QmcReplicateOptions, QmcReplicatePlan, SeedableRng};
use scioxide_stats::{
    ConfidenceProbability, MeanPowerDesign, MeanPowerModel, MeanPowerReference, PosteriorOptions,
    StatisticsOptions, TestAlternative, posterior_summary_with_options, power, sample_size,
    sobol_indices_qmc_scoped,
};

fn main() -> Result<(), SciOxideError> {
    let alternative = MeanPowerModel {
        effect: 0.5_f64,
        standard_deviation: 1.0,
        design: MeanPowerDesign::OneSample,
        reference: MeanPowerReference::EstimatedVariance,
        alternative: TestAlternative::Greater,
        significance: ConfidenceProbability::new(0.05)?,
    };
    let size = sample_size(2, 128, 0.8, |n| power(alternative, n))?;
    println!("first model-qualified size: {size:?}");

    let root = Philox4x64::from_seed(558);
    let qmc = QmcReplicatePlan::new(
        &QmcPlan::sobol(4)?,
        &root,
        QmcReplicateOptions::default()
            .with_replicates(4)
            .with_points_per_replicate(256),
    )?;
    let sensitivity = sobol_indices_qmc_scoped(
        &qmc,
        2,
        256,
        |_, u: f64| Ok(2.0 * u - 1.0),
        |x| Ok(x[0] + 2.0 * x[1]),
        StatisticsOptions::new().with_max_bytes(32_768),
        ExecutionContext::sequential(),
    )?;
    println!("first-order Sobol indices: {:?}", sensitivity.first_order());

    let draws: Vec<f64> = (0..4 * 32)
        .map(|i| ((i * 17 % 37) as f64 - 18.0) / 10.0)
        .collect();
    let chains = MatrixView::from_slice(4, 32, &draws)?;
    let posterior = posterior_summary_with_options(chains, PosteriorOptions::default())?;
    println!(
        "posterior mean={}, R-hat={:?}",
        posterior.mean, posterior.rhat
    );
    Ok(())
}
