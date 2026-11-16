//! Explicit QMC multistart and stochastic chain factories.
use scioxide_core::{Bounds, Budget, SciOxideError};
use scioxide_optimize::{
    BasinHopping, BasinHoppingOptions, BoundedObjective, DualAnnealing, DualAnnealingOptions,
    GlobalSearchOptions, MultiStart, ObjectiveFn, PowellLocalFactory, UniformBasinProposal,
};
fn main() -> Result<(), SciOxideError> {
    let options = GlobalSearchOptions::default().with_starts(Budget::try_from(16)?);
    let objective = ObjectiveFn::new(2, |x: &[f64]| {
        Ok(x.iter().map(|v| (v * v - 1.0).powi(2)).sum())
    });
    let bounds = Bounds::new(vec![(Some(-3.0), Some(3.0)); 2])?;
    let problem = BoundedObjective::new(&objective, &bounds)?;
    let multistart = MultiStart::new(
        options.clone(),
        scioxide_random::QmcPlan::halton(2)?,
        PowellLocalFactory::default(),
    )?;
    let result = multistart.minimize(&problem)?;
    println!(
        "{} attempts, {} local successes, {} observed basins; {:?}",
        result.starts().len(),
        result.successes(),
        result.basins(),
        result.termination()
    );
    if let Some(best) = result.best() {
        println!("best observed: {:?}, f={}", best.x(), best.fun());
    }
    let basin = BasinHopping::new(
        BasinHoppingOptions::new(options.clone(), 0.1)?,
        UniformBasinProposal::new(0.3)?,
        PowellLocalFactory::default(),
    )?;
    let anneal = DualAnnealing::new(
        DualAnnealingOptions::new(options),
        PowellLocalFactory::default(),
    )?;
    for result in [
        basin.minimize(&problem, &[2.0, 2.0])?,
        anneal.minimize(&problem, &[2.0, 2.0])?,
    ] {
        println!(
            "chain: {:?}, best={:?}",
            result.termination(),
            result.best().map(|r| r.fun())
        );
    }
    Ok(())
}
