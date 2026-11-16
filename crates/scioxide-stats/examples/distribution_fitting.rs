//! Injected finite-grid example; production optimization adapters are separate.
use scioxide_core::*;
use scioxide_stats::*;
#[path = "../tests/support/fit_solver.rs"]
mod reference;
fn main() -> Result<(), SciOxideError> {
    let data = [
        FitObservation::Exact(1.0),
        FitObservation::Exact(2.0),
        FitObservation::RightCensored(3.0),
    ];
    let bounds = Bounds::new(vec![(Some(0.1), Some(1.0))])?;
    let problem = PreparedFit::new(
        &DistributionFitModel::Exponential,
        &data,
        FitMethod::Likelihood,
        None,
        &bounds,
        &[false],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(0),
    )?;
    println!(
        "NLL at rate 1/3 = {}",
        problem.value(&[1.0 / 3.0], &mut ())?
    );

    let result = scioxide_stats::fit_with_options(
        FitRequest {
            objective: &problem,
            derivatives: None,
            bounds: &bounds,
            integrality: &[false],
            initial: &[0.5],
        },
        &mut reference::GridSolver { intervals: 90 },
        scioxide_stats::StatisticsOptions::new().with_max_bytes(128),
    )?;
    println!(
        "Best finite-grid rate: {:?}; report: {:?}",
        result.parameters(),
        result.report()
    );
    println!(
        "Box-Cox log limit at 2 = {}",
        scioxide_stats::PowerTransform::BoxCox.apply(2.0, 0.0)?
    );
    let plot = probplot_with_options(
        &[1., 2., 3.],
        &DistributionFitModel::Normal,
        &[0., 1.],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(1024),
    )?;
    println!("Probability-plot correlation: {:?}", plot.fit().correlation);
    Ok(())
}
