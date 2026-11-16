#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
use super::*;
use scioxide_core::{Complex, ResidualTolerance};
use scioxide_rootfind::{
    NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovResult, NewtonKrylovSolver,
};
#[path = "../../examples/support/nonlinear_pde.rs"]
#[allow(
    unreachable_pub,
    reason = "shared example fixture with crate-external visibility"
)]
mod model;
use model::{ReactionDiffusion, real};

mod failures;
mod reuse;
mod spatial;
mod transient;

#[path = "../../examples/support/nonlinear_fixture.rs"]
#[allow(
    unreachable_pub,
    reason = "shared example fixture with crate-external visibility"
)]
mod fixtures;
use fixtures::{block_pattern, fixture};

/// Run direct callback evaluations in a private, unbounded root epoch.
fn context<R>(
    run: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<R, SciOxideError>,
) -> Result<R, SciOxideError> {
    unscoped(run)
}
fn options<R: RealScalar>() -> NewtonKrylovOptions<R> {
    let tol = if R::EPSILON > real(1e-10) {
        real(1e-5)
    } else {
        real(1e-11)
    };
    NewtonKrylovOptions::default()
        .with_residual_tolerance(ResidualTolerance::new(tol, R::ZERO).unwrap())
        .with_step_tolerance(scioxide_core::StepTolerance::new(R::EPSILON, R::EPSILON).unwrap())
}
struct Inverse<T: NumericScalar>(Vec<T>);
impl<T: NumericScalar> Preconditioner<T> for Inverse<T> {
    fn dimension(&self) -> usize {
        self.0.len()
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn precondition_into(&self, x: &[T], out: &mut [T], _: &mut [T]) -> Result<(), SciOxideError> {
        for ((v, &x), &d) in out.iter_mut().zip(x).zip(&self.0) {
            *v = x * d;
        }
        Ok(())
    }
}

struct Scalar;
impl OdeSystem<f64> for Scalar {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        x: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = x[0] * x[0] - 4.;
        Ok(())
    }
    fn jacobian(
        &self,
        _: f64,
        x: &[f64],
        out: &mut scioxide_linalg::MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.set(0, 0, 2. * x[0])?;
        Ok(Some(()))
    }
}
#[test]
fn dense_adapter_composes_with_newton() {
    let fields = PdeFieldLayout::new(&[1]).unwrap();
    let plan_fields = fields;
    let plan_mode = PdeJacobianMode::Dense;
    let mut work = NonlinearPdeProblem::new(&Scalar, 0.0, &plan_fields, &plan_mode)
        .workspace()
        .unwrap();
    let residual = NonlinearPdeProblem::new(&Scalar, 0., &plan_fields, &plan_mode)
        .bind(&mut work)
        .unwrap();
    let problem = scioxide_rootfind::NewtonKrylovProblem::with_jacobian(&residual);
    let result = scioxide_rootfind::NewtonKrylovSolver::new(
        scioxide_rootfind::NewtonKrylovOptions::default(),
    )
    .and_then(|root_plan| root_plan.solve(&problem, &[1.]))
    .unwrap();
    assert!(result.is_success());
    assert!((result.point()[0] - 2.).abs() < 1e-7);
}
