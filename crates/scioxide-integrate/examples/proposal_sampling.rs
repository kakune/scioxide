//! Normalized importance sampling on a finite box with caller-owned resources.
use scioxide_core::{Budget, ExecutionContext, Interval, SciOxideError};
use scioxide_integrate::{
    MonteCarloIntegrator, MonteCarloOptions, MonteCarloProposal, ProposalUniforms, RandomStreams,
};
use scioxide_random::{Pcg64, SeedableRng};

/// Density `q(x) = (1 + 2x) / 2` through its stable inverse CDF.
struct LinearProposal;
impl MonteCarloProposal<f64> for LinearProposal {
    fn dimension(&self) -> usize {
        1
    }
    fn uniform_count(&self) -> usize {
        1
    }
    fn sample(&self, u: ProposalUniforms<'_, f64>, x: &mut [f64]) -> Result<f64, SciOxideError> {
        let u = u.as_slice();
        x[0] = 4.0 * u[0] / ((1.0 + 8.0 * u[0]).sqrt() + 1.0);
        Ok((1.0 + 2.0 * x[0]) / 2.0)
    }
}

fn main() -> Result<(), SciOxideError> {
    let plan = MonteCarloIntegrator::importance(
        &[Interval::new(0.0, 1.0)?],
        LinearProposal,
        MonteCarloOptions::default().with_samples(Budget::try_from(10_000)?),
    )?;
    let mut workspace = plan.workspace(1, ExecutionContext::sequential_native())?;
    let root = Pcg64::from_seed(590);
    let mut value = 0.0;
    let report = plan.integrate_into(
        |x: &[f64]| Ok(x[0] * x[0]),
        RandomStreams::new(&root, 0),
        &mut value,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!(
        "Integral of x²: {value}; analytic value: 1/3 ({:?})",
        report.termination()
    );
    if let Some(mut stats) = workspace.statistics()[0].clone() {
        println!(
            "samples={}, ESS={:?}, standard error={}, interval={:?}",
            stats.samples(),
            stats.effective_sample_size(),
            stats.standard_error(),
            stats.interval_report()?
        );
    }
    Ok(())
}
