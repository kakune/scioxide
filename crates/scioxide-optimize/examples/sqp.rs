//! Projected starts, dependent equalities, direct bounds, and reusable SQP state.
use scioxide_core::{Bounds, ExecutionContext, SciOxideError};
use scioxide_optimize::{
    ConstrainedProblem, ConstraintFunction, Constraints, ObjectiveFunction, OptimizeResult, Sqp,
    SqpOptions,
};
struct Bowl;
impl ObjectiveFunction<f64> for Bowl {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((x[0] - 2.0).powi(2) + x[1] * x[1])
    }
    fn gradient_into(&self, x: &[f64], g: &mut [f64]) -> Result<bool, SciOxideError> {
        g[0] = 2.0 * (x[0] - 2.0);
        g[1] = 2.0 * x[1];
        Ok(true)
    }
}
struct Sum(f64);
impl ConstraintFunction<f64> for Sum {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(self.0 * (x[0] + x[1] - 1.0))
    }
    fn jacobian_into(&self, _: &[f64], g: &mut [f64]) -> Result<bool, SciOxideError> {
        g.fill(self.0);
        Ok(true)
    }
}
fn main() -> Result<(), SciOxideError> {
    let constraints = Constraints::new(2)?
        .equality(Sum(1.0))?
        .equality(Sum(2.0))?
        .bounds(Bounds::new(vec![(Some(0.0), Some(1.0)); 2])?)?;
    let problem = ConstrainedProblem::new(&Bowl, &constraints)?;
    let solver = Sqp::new(SqpOptions::default());
    let mut workspace = solver.workspace(&problem)?;
    let mut result = OptimizeResult::empty();
    solver.minimize_into(
        &problem,
        &[-3.0, 4.0],
        &mut result,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    assert!(result.is_success());
    let x = result.x();
    assert!((x[0] + x[1] - 1.0).abs() < 1e-5);
    assert!(x.iter().all(|v| (0.0..=1.0).contains(v)));
    let dual = workspace.state();
    let equality_dual = dual.multipliers()[0] + 2.0 * dual.multipliers()[1];
    for (j, gradient) in [2.0 * (x[0] - 2.0), 2.0 * x[1]].into_iter().enumerate() {
        let lower = dual.lower_multipliers()[j];
        let upper = dual.upper_multipliers()[j];
        assert!((gradient + equality_dual + upper - lower).abs() < 1e-5);
        assert!(lower >= 0.0 && upper >= 0.0);
        assert!((lower * x[j]).abs() < 1e-5 && (upper * (x[j] - 1.0)).abs() < 1e-5);
    }
    println!(
        "{}: x={:?}, KKT={:?}",
        result.termination(),
        result.x(),
        result.diagnostics()
    );
    println!(
        "nonlinear multipliers={:?}, lower={:?}, upper={:?}",
        workspace.state().multipliers(),
        workspace.state().lower_multipliers(),
        workspace.state().upper_multipliers()
    );
    let state = workspace.state().clone();
    let point = result.x().to_vec();
    workspace.set_warm_start(&state)?;
    solver.minimize_into(
        &problem,
        &point,
        &mut result,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    assert!(result.is_success());
    assert!((result.x()[0] - 1.0).abs() < 1e-5 && result.x()[1].abs() < 1e-5);
    Ok(())
}
