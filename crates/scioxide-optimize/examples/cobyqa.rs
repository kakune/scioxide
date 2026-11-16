//! Derivative-free minimization with a nonlinear inequality.
use scioxide_core::SciOxideError;
use scioxide_optimize::{
    Cobyqa, CobyqaOptions, ConstrainedProblem, ConstraintFn, Constraints, ObjectiveFn,
};

fn main() -> Result<(), SciOxideError> {
    let objective = ObjectiveFn::new(1, |x: &[f64]| Ok((x[0] - 2.0).powi(2)));
    let constraints =
        Constraints::new(1)?.inequality(ConstraintFn::new(1, |x: &[f64]| Ok(1.0 - x[0])))?;
    let problem = ConstrainedProblem::new(&objective, &constraints)?;
    let solver = Cobyqa::new(CobyqaOptions::default())?;
    let result = solver.minimize(&problem, &[0.0])?;
    assert!(result.is_success());
    assert!((result.x()[0] - 2.0).abs() < 1e-2);
    assert!(result.primal_feasibility().unwrap_or(f64::INFINITY) < 1e-4);
    Ok(())
}
