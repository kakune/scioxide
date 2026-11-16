//! Multivariate laws with injected streams, matrix output and bounded joint probabilities.
use scioxide_core::{SciOxideError, Tolerance};
use scioxide_linalg::Matrix;
use scioxide_random::{
    Dirichlet, MatrixNormal, MultivariateNormal, MultivariateProbabilityOptions, Philox4x64,
    RandomTable, SeedableRng, SplitMix64,
};
fn main() -> Result<(), SciOxideError> {
    let mut rng = SplitMix64::from_seed(542);
    let weights = Dirichlet::with_options(
        &[1.0, 2.0, 3.0],
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )?
    .sample(&mut rng)?;
    let table = RandomTable::<f64>::with_options(
        &[3, 4],
        &[2, 3, 2],
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )?
    .sample(&mut rng)?;
    let mean = Matrix::<f64>::zeros(2, 2)?;
    let identity = Matrix::identity(2)?;
    // Matrix-valued samples are row-major rows of the law's `shape()`.
    let law = MatrixNormal::with_options(
        &mean,
        &identity,
        &identity,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )?;
    let (rows, cols) = law.shape();
    let matrix = Matrix::from_vec(rows, cols, law.sample(&mut rng)?)?;
    let covariance = Matrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 1.0])?;
    let gaussian = MultivariateNormal::new(&[0.0, 0.0], &covariance)?;
    let options = MultivariateProbabilityOptions::default()
        .with_points_per_replicate(4096)
        .with_tolerance(Tolerance::new(1e-4, 1e-4)?);
    let plan = gaussian.prepare_probability(options, &Philox4x64::from_seed(542))?;
    let report = plan.rectangle(&[f64::NEG_INFINITY; 2], &[0.0; 2])?;
    println!("simplex: {weights:?}; fixed-margin table: {table:?}");
    println!(
        "matrix: {:?}; joint probability: {report:?}",
        matrix.as_slice()
    );
    Ok(())
}
