//! Follow a fold, solve a homotopy and explicitly exclude a known solution.
#[path = "support/continuation.rs"]
mod support;
use scioxide_core::{Budget, SciOxideError};
use scioxide_rootfind::{
    Continuation, ContinuationOptions, ContinuationPath, ContinuationProblem, DeflatedProblem,
    DeflationOptions, DeflationSet, LinearHomotopy, NewtonKrylovOptions,
};
fn main() -> Result<(), SciOxideError> {
    let options = ContinuationOptions::new(0.1, 1e-6, 0.15, Budget::try_from(200)?)?
        .with_dense_diagnostics(1e-8);
    let fold = support::fold();
    let problem = ContinuationProblem::new(&fold)?;
    let arclength = Continuation::new(ContinuationPath::PseudoArclength(3.), options.clone())?;
    let seed = arclength
        .state(&problem, &[1., 1.])?
        .with_initial_tangent(&[-1., -2.])?;
    let branch = arclength.follow(&problem, seed)?;
    println!(
        "fold: {:?}; endpoint {:?}; turning candidates {}",
        branch.termination(),
        branch.point(),
        branch
            .history()
            .iter()
            .filter(|h| h.turning_point_candidate())
            .count()
    );
    let checkpoint = branch.checkpoint()?;
    println!("restart metadata: {:?}", checkpoint.metadata());
    let start = support::square(1.);
    let target = support::square(4.);
    let homotopy = LinearHomotopy::new(&start, &target)?;
    let problem = ContinuationProblem::new(&homotopy)?;
    let natural = Continuation::new(ContinuationPath::Parameters(vec![1.]), options)?;
    let path = natural.follow(&problem, natural.state(&problem, &[1., 0.])?)?;
    println!(
        "homotopy: {:?}; endpoint {:?}",
        path.termination(),
        path.point()
    );
    let mut known = DeflationSet::new(1, Budget::try_from(2)?, DeflationOptions::default())?;
    known.insert(&start, &[1.])?;
    let deflated = DeflatedProblem::new(
        &start,
        &known,
        NewtonKrylovOptions::default()
            .with_step_tolerance(scioxide_core::StepTolerance::new(1e-15, 0.)?)
            .with_residual_tolerance(scioxide_core::ResidualTolerance::new(1e-12, 0.)?),
    )?;

    let result = deflated.solve(&[0.2])?;
    println!(
        "deflation: {:?}; root {:?}; original residual {:?}",
        result.termination(),
        result.point(),
        result.residual_norm()
    );
    assert!(branch.is_success() && path.is_success() && result.is_success());
    Ok(())
}
