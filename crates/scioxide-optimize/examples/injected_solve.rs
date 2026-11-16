//! Reuse BFGS storage while starting a fresh local solve epoch.

use scioxide_core::{EvaluationRole, ExecutionContext, GradientTolerance, SciOxideError};
use scioxide_optimize::{Bfgs, BfgsOptions, ObjectiveFunction, OptimizeResult};

struct Quadratic;

impl ObjectiveFunction<f64> for Quadratic {
    fn dim(&self) -> usize {
        1
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] * x[0])
    }
    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output[0] = 2.0 * x[0];
        Ok(true)
    }
}

fn main() -> Result<(), SciOxideError> {
    let solver = Bfgs::new(
        BfgsOptions::default().with_gradient_tolerance(GradientTolerance::new(1e-10, 1e-10)?),
    );
    let mut workspace = solver.workspace(1)?;
    let mut result = OptimizeResult::<f64>::empty();
    result.try_reserve(1)?;
    for initial in [1.0, 2.0] {
        solver.minimize_into(
            &Quadratic,
            &[initial],
            &mut result,
            &mut workspace,
            ExecutionContext::sequential(),
        )?;
        println!(
            "{}: x={:?}, objective calls={}, accepted={}",
            result.termination(),
            result.x(),
            result.counts().own().provider(EvaluationRole::Objective),
            result.counts().own().accepted()
        );
    }
    Ok(())
}
