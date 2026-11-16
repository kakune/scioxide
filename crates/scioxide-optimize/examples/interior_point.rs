//! Mixed constraints, an infeasible start, exact curvature and injected KKT factors.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_linalg::MatrixViewMut;
use scioxide_optimize::{
    ConstrainedProblem, ConstraintFunction, Constraints, DenseKktProvider, InteriorPoint,
    InteriorPointOptions, LagrangianHessian, ObjectiveFunction, OptimizeResult,
};
struct Objective;
impl ObjectiveFunction<f64> for Objective {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((x[0] - 2.).powi(2) + (x[1] - 1.).powi(2))
    }
    fn gradient_into(&self, x: &[f64], g: &mut [f64]) -> Result<bool, SciOxideError> {
        g[0] = 2. * (x[0] - 2.);
        g[1] = 2. * (x[1] - 1.);
        Ok(true)
    }
}
impl LagrangianHessian<f64> for Objective {
    fn hessian_into(
        &self,
        _: &[f64],
        _: &[f64],
        mut h: MatrixViewMut<'_, f64>,
    ) -> Result<(), SciOxideError> {
        for i in 0..2 {
            for j in 0..2 {
                h.set(i, j, if i == j { 2. } else { 0. })?;
            }
        }
        Ok(())
    }
}
struct Row(bool);
impl ConstraintFunction<f64> for Row {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(if self.0 {
            x[0] + x[1] - 1.
        } else {
            0.25 - x[1]
        })
    }
    fn jacobian_into(&self, _: &[f64], j: &mut [f64]) -> Result<bool, SciOxideError> {
        j.copy_from_slice(if self.0 { &[1., 1.] } else { &[0., -1.] });
        Ok(true)
    }
}
fn main() -> Result<(), SciOxideError> {
    let constraints = Constraints::new(2)?
        .equality(Row(true))?
        .inequality(Row(false))?;
    let problem =
        ConstrainedProblem::new(&Objective, &constraints)?.with_lagrangian_hessian(&Objective);
    let solver = InteriorPoint::new(InteriorPointOptions::default());
    // The workspace owns the injected KKT factorization; its order is
    // variables plus nonlinear rows plus bound sides.
    let mut workspace = solver
        .workspace(&problem)?
        .with_kkt_provider(DenseKktProvider::new(4)?);
    let mut result = OptimizeResult::empty();
    solver.minimize_into(
        &problem,
        &[-3., 4.],
        &mut result,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    assert!(result.is_success());
    let x = result.x();
    let l = workspace.state().multipliers();
    assert!((x[0] - 0.75).abs() < 1e-5 && (x[1] - 0.25).abs() < 1e-5);
    assert!((x[0] + x[1] - 1.).abs() < 1e-5 && x[1] >= 0.25 - 1e-5);
    assert!((2. * (x[0] - 2.) + l[0]).abs() < 1e-5);
    assert!((2. * (x[1] - 1.) + l[0] - l[1]).abs() < 1e-5);
    assert!(l[1] >= 0. && (l[1] * (0.25 - x[1])).abs() < 1e-5);
    println!("x={x:?}, multipliers={l:?}, {:?}", result.termination());
    Ok(())
}
