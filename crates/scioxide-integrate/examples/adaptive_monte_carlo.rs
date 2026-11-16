//! Train frozen maps and strata, then evaluate them with independent samples.
use scioxide_core::{Budget, ExecutionContext, Interval, SciOxideError};
use scioxide_integrate::{
    MiserOptions, MiserTrainer, MonteCarloIntegrator, MonteCarloOptions, RandomStreams,
    VegasOptions, VegasTrainer,
};
use scioxide_random::{Philox4x64, QmcPlan, QmcReplicateOptions, QmcReplicatePlan, SeedableRng};

fn main() -> Result<(), SciOxideError> {
    let bounds = [Interval::new(0.0, 1.0)?; 2];
    let root = Philox4x64::from_seed(594);
    let function = |x: &[f64]| {
        let y = (-30.0 * x[0] - 3.0 * x[1]).exp();
        Ok([y, -2.0 * y])
    };

    // Training writes a separate product from its own stream; the plan never changes.
    let vegas = VegasTrainer::new(
        &bounds,
        VegasOptions::default()
            .with_bins(16)
            .with_iterations(Budget::try_from(4)?),
    )?;
    let (map, trained) = vegas.train(&function, RandomStreams::new(&root, 0))?;
    println!("VEGAS training: {trained:?}");

    // Randomized QMC replicates sample the trained map.
    let qmc = QmcReplicatePlan::new(
        &QmcPlan::sobol(2)?,
        &Philox4x64::from_seed(595),
        QmcReplicateOptions::default()
            .with_replicates(8)
            .with_points_per_replicate(1024),
    )?;
    let replicated = MonteCarloIntegrator::importance(
        &bounds,
        map,
        MonteCarloOptions::default()
            .with_samples(Budget::try_from(1024)?)
            .with_replicates(8),
    )?;
    let mut workspace = replicated.workspace(2, ExecutionContext::sequential_native())?;
    let mut output = [0.0; 2];
    let report = replicated.integrate_into(
        &function,
        &qmc,
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!(
        "QMC + VEGAS: {output:?}, report={report:?}, statistics={:?}",
        workspace.statistics()
    );

    let miser = MiserTrainer::new(
        &bounds,
        MiserOptions::default()
            .with_evaluation(MonteCarloOptions::default().with_samples(Budget::try_from(8192)?))
            .with_max_leaves(Budget::try_from(8)?)
            .with_max_depth(4),
    )?;
    let (strata, trained) = miser.train(&function, RandomStreams::new(&root, 1))?;
    let stratified = MonteCarloIntegrator::stratified(strata)?;
    let mut workspace = stratified.workspace(2, ExecutionContext::sequential_native())?;
    let report = stratified.integrate_into(
        &function,
        RandomStreams::new(&root, 2),
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!(
        "MISER: {output:?}, training={trained:?}, report={report:?}, statistics={:?}",
        workspace.statistics()
    );
    Ok(())
}
