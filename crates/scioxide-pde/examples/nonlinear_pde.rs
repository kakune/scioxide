//! Steady and transient coupled reaction–diffusion using separate model/solver state.
#[path = "support/nonlinear_fixture.rs"]
mod fixtures;
#[path = "support/nonlinear_pde.rs"]
mod model;
use scioxide_core::{
    Budget, ErrorTolerance, ExecutionContext, NormSelection, ResidualTolerance, SciOxideError,
};
use scioxide_linalg::Vector;
use scioxide_ode::{
    OdeMethod, OdeOptions, OdeOutputSchedule, OdeProblem, OdeStageBackend, OdeState,
    RosenbrockMethod, RosenbrockW,
};
use scioxide_pde::{MethodOfLinesPlan, NonlinearPdeProblem, PdeFieldLayout, PdeJacobianMode};
use scioxide_rootfind::{NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovSolver};

fn main() -> Result<(), SciOxideError> {
    let (diffusion, _, mass, target, laplace, _) = fixtures::fixture::<f64>(15);
    let mut equation = model::ReactionDiffusion::new(&diffusion, &mass, &target, &laplace)?;
    let plan_fields = PdeFieldLayout::new(&[15, 15])?;
    let plan_mode = PdeJacobianMode::Operator {
        operator_scratch: 0,
    };
    let problem = NonlinearPdeProblem::new(&equation, 0.0, &plan_fields, &plan_mode);
    let mut workspace = problem.workspace()?;
    let residual = problem.bind(&mut workspace)?;
    let options = NewtonKrylovOptions::default()
        .with_step_tolerance(scioxide_core::StepTolerance::new(
            f64::EPSILON,
            f64::EPSILON,
        )?)
        .with_residual_tolerance(ResidualTolerance::new(1e-10, 0.)?);
    let result = NewtonKrylovSolver::new(options).and_then(|root_plan| {
        root_plan.solve(
            &NewtonKrylovProblem::with_jacobian(&residual),
            &vec![0.; 30],
        )
    })?;
    let steady_error = result
        .point()
        .iter()
        .zip(&target)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f64::max);
    assert!(result.is_success() && steady_error < 1e-7);
    println!(
        "steady {:?}: error={steady_error:e}, first field has {} free values",
        result.termination(),
        plan_fields.field(result.point(), 0)?.len()
    );

    equation.decay = 0.8;
    let physical_mass: Vec<_> = mass.iter().chain(&mass).copied().collect();
    let normalization = MethodOfLinesPlan::new(30, Some(&physical_mass), None)?;
    let mut workspace = normalization.workspace()?;
    let system = normalization.bind(&equation, &mut workspace)?;
    let tolerance = ErrorTolerance::try_vector(&[1e-8; 30], 1e-8, NormSelection::Infinity)?;
    let backend = OdeStageBackend::Krylov {
        iterations: Budget::try_from(100)?,
        restart: Budget::try_from(30)?,
        operator_scratch: 0,
        preconditioner_scratch: 0,
    };
    let ode = (
        RosenbrockW::new(RosenbrockMethod::Ros34Pw2)?.with_backend(backend),
        OdeOptions::new()
            .with_output_schedule(OdeOutputSchedule::AcceptedSteps)
            .with_error_tolerance(normalization.error_tolerance(&tolerance)?)
            .with_initial_step(0.01),
    );
    let problem = OdeProblem::new(&system, 0.2);
    let mut state = OdeState::new(0., Vector::from_vec(target.clone()));
    let mut observations = 0;
    let report = ode.0.advance(
        &problem,
        &ode.1,
        &mut state,
        &mut |_, _: &[f64]| {
            observations += 1;
            Ok(())
        },
        &mut ode.0.workspace(&problem, &ode.1)?,
        ExecutionContext::sequential(),
    )?;
    let mut expected = target.clone();
    equation.exact_into(0.2, &mut expected);
    let error = state
        .values()
        .iter()
        .zip(&expected)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f64::max);
    assert!(report.is_success() && error < 1e-6);
    println!(
        "transient {:?}: error={error:e}, streamed observations={observations}",
        report.termination()
    );
    Ok(())
}
