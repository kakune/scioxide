//! Minimize a derivative-free objective with finite/fixed bounds and explicit directions.
use scioxide_core::{Bounds, ExecutionContext, SciOxideError};
use scioxide_linalg::Matrix;
use scioxide_optimize::{BoundedObjective, ObjectiveFn, OptimizeResult, Powell, PowellOptions};

fn main() -> Result<(), SciOxideError> {
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok((x[0] - 0.25).abs() + x[1] * x[1]));
    let bounds = Bounds::new(vec![(Some(-2.0), Some(2.0)), (Some(1.0), Some(1.0))])?;
    let problem = BoundedObjective::new(&objective, &bounds)?;
    let directions = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0])?;
    let solver = Powell::new(PowellOptions::default().with_directions(directions))?;
    let mut workspace = solver.workspace(2)?;
    let mut result = OptimizeResult::empty();
    result.try_reserve(2)?;
    solver.minimize_into(
        &problem,
        &[0.0, 1.0],
        &mut result,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    println!(
        "x={:?}, f={}, termination={:?}, evaluations={}",
        result.x(),
        result.fun(),
        result.termination(),
        result.fun_evals()
    );
    println!("direction diagnostics: {:?}", result.method_diagnostics());
    Ok(())
}
