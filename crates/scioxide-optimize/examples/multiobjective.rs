//! A finite approximation to a constrained analytic Pareto front.
use scioxide_core::{Budget, SciOxideError};
use scioxide_optimize::{
    ConstraintFn, Constraints, FiniteBounds, NoiseModel, Nsga2, Nsga2Options, Nsga2Problem,
    SampledObjective,
};
struct Objectives;
impl SampledObjective<f64> for Objectives {
    fn dimension(&self) -> usize {
        1
    }
    fn objectives(&self) -> usize {
        2
    }
    fn noise_model(&self) -> NoiseModel {
        NoiseModel::Deterministic
    }
    fn sample_into(&self, x: &[f64], _: u64, out: &mut [f64]) -> Result<(), SciOxideError> {
        out[0] = x[0] * x[0];
        out[1] = (x[0] - 2.).powi(2);
        Ok(())
    }
}
fn main() -> Result<(), SciOxideError> {
    let constraints =
        Constraints::new(1)?.inequality(ConstraintFn::new(1, |x: &[f64]| Ok(0.5 - x[0])))?;
    let bounds = FiniteBounds::new(vec![(-1., 3.)])?;
    let solver = Nsga2::new(Nsga2Options::new(32, 24, Budget::try_from(40)?).with_seed(42))?;
    let problem = Nsga2Problem::new(&Objectives, &bounds)?.with_constraints(&constraints)?;
    let approximation = solver.solve(&problem)?;
    println!(
        "{} retained feasible points; stop={}; observed nondominance={}",
        approximation.archive().len(),
        approximation.termination(),
        approximation.archive().verify()?
    );
    for entry in approximation.archive().entries().take(5) {
        println!(
            "x={:?}, objectives={:?}, constraints={:?}",
            entry.point(),
            entry.objectives(),
            entry.constraints()
        );
    }
    Ok(())
}
