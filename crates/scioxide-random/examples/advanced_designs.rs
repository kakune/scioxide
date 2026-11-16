//! Prepared finite designs and independent randomized integration estimates.
use scioxide_core::SciOxideError;
use scioxide_random::{
    DiscrepancyMethod, LatinHypercubeOptions, Philox4x64, QmcPlan, QmcReplicateOptions,
    QmcReplicatePlan, SeedableRng, SplitMix64, discrepancy, latin_hypercube_with_options,
};
fn main() -> Result<(), SciOxideError> {
    let (points, report) = latin_hypercube_with_options::<f64, _>(
        32,
        3,
        &mut SplitMix64::from_seed(42),
        LatinHypercubeOptions::default().with_max_proposals(100),
    )?;
    println!(
        "Centered discrepancy: {} after {} swap proposals",
        discrepancy(&points, 3, DiscrepancyMethod::Centered)?,
        report.proposals()
    );
    let replicates = QmcReplicatePlan::new(
        &QmcPlan::sobol(3)?,
        &Philox4x64::from_key([42, 17]),
        QmcReplicateOptions::default()
            .with_replicates(16)
            .with_points_per_replicate(256),
    )?;
    let estimate = replicates.estimate(|x: &[f64]| Ok(x.iter().product()))?;
    println!(
        "Integral: {} (analytic 0.125), empirical standard error: {}",
        estimate.mean, estimate.standard_error
    );
    Ok(())
}
