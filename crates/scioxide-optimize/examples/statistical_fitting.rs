//! Explicit solver injection keeps statistical models in stats.
use scioxide_core::{Bounds, SciOxideError};
use scioxide_optimize::{LbfgsB, LbfgsBFitSolver, SimplexSolver, SimplexTransportSolver};
use scioxide_stats::{
    DistributionFitModel, ExactFitDerivatives, FitMethod, FitObservation, FitRequest, PreparedFit,
};
fn main() -> Result<(), SciOxideError> {
    let samples = [-1., -1., 3., 3.];
    let observations = samples.map(FitObservation::Exact);
    let bounds = Bounds::new(vec![(Some(-4.), Some(4.)), (Some(0.2), Some(5.))])?;
    let model = DistributionFitModel::Normal;
    let objective = PreparedFit::new(
        &model,
        &observations,
        FitMethod::Likelihood,
        None,
        &bounds,
        &[false; 2],
        scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
    )?;
    let derivatives = ExactFitDerivatives {
        model,
        samples: &samples,
    };
    let mut solver = LbfgsBFitSolver::new(LbfgsB::default(), 2)?;

    let result = scioxide_stats::fit_with_options(
        FitRequest {
            objective: &objective,
            derivatives: Some(&derivatives),
            bounds: &bounds,
            integrality: &[false; 2],
            initial: &[0., 1.],
        },
        &mut solver,
        scioxide_stats::StatisticsOptions::new().with_max_bytes(4096),
    )?;
    println!(
        "Normal parameters: {:?}; {:?}",
        result.parameters(),
        result.report().termination()
    );

    let mut transport = SimplexTransportSolver::new(SimplexSolver::default(), 4, 4096);
    let coordinates = [0.0, 1.0];
    let layout = scioxide_core::StridedLayout::new(&[2], &[1])?;
    let left =
        scioxide_stats::BatchView::new(&coordinates, layout, scioxide_stats::Axis::Index(0))?
            .with_weights(scioxide_stats::Weights::<_, scioxide_stats::Mass>::new(&[
                0.75, 0.25,
            ]))?;
    let right =
        scioxide_stats::BatchView::new(&coordinates, layout, scioxide_stats::Axis::Index(0))?
            .with_weights(scioxide_stats::Weights::<_, scioxide_stats::Mass>::new(&[
                0.25, 0.75,
            ]))?;
    let result = scioxide_stats::wasserstein_distance_nd(
        left,
        right,
        Default::default(),
        Default::default(),
        &mut transport,
    )?;
    println!(
        "Transport distance: {:?}; {:?}",
        result.distance, result.termination
    );
    Ok(())
}
