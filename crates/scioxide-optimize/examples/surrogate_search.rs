//! Fixed-covariance surrogate proposals for a bounded expensive objective.
use scioxide_core::{Budget, SciOxideError};
use scioxide_interp::{Covariance, CovarianceKernel, LocalModelOptions};
use scioxide_optimize::{
    FiniteBounds, GaussianAcquisition, KrigingSurrogateFitter, PowellLocalFactory,
    SurrogateHistory, SurrogateObservation, SurrogateSearch, SurrogateSearchOptions,
    UniformStartDesign,
};

fn main() -> Result<(), SciOxideError> {
    let bounds = FiniteBounds::new(vec![(-2.0, 2.0)])?;
    let mut history = SurrogateHistory::<f64, 1>::new(bounds, 12)?;
    for x in [-1.5, -0.5, 0.5, 1.5] {
        history.push(SurrogateObservation::new([x], (x - 0.4_f64).powi(2), 0.01))?;
    }
    let options = SurrogateSearchOptions::default()
        .with_acquisition_starts(Budget::try_from(8)?)
        .with_seed(41)
        .with_separation(0.01)
        .with_batch_size(Budget::try_from(2)?);
    let fitter = KrigingSurrogateFitter::new(
        Covariance::new(CovarianceKernel::Matern32, 1.0, [0.7])?,
        LocalModelOptions::default(),
    );
    let search = SurrogateSearch::new(
        options,
        fitter,
        GaussianAcquisition::ExpectedImprovement { margin: 0.0 },
        |x: &[f64]| Ok(x[0] >= -1.75),
        UniformStartDesign,
        PowellLocalFactory::default(),
    )?;
    let batch = search.propose_batch(&history, 2)?;
    println!("batch proposals: {batch:?}");
    let report = search.minimize(&mut history, |x| Ok(((x[0] - 0.4).powi(2), 0.01)))?;
    println!(
        "real objective calls: {}; total observations: {}",
        report.objective_evaluations(),
        history.observations().len()
    );
    Ok(())
}
