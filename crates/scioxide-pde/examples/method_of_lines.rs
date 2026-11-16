//! Complex semidiscrete residuals with explicit mass, field scales and sparse stages.
use scioxide_core::{
    Complex64, ErrorTolerance, ExecutionContext, NormSelection, SciOxideError, SolveContext,
    StructuralPattern, StructuralPatternView,
};
use scioxide_linalg::Vector;
use scioxide_ode::{
    OdeMethod, OdeOptions, OdeProblem, OdeStageBackend, OdeState, OdeSystem, RosenbrockMethod,
    RosenbrockW,
};
use scioxide_pde::{
    MethodOfLinesPlan, PdeObservation, PdeOutputSink, PdeSolveReport, PdeTrajectory,
};
use scioxide_sparse::SparseLuLimits;
struct Residual;
impl OdeSystem<f64, Complex64> for Residual {
    fn dimension(&self) -> usize {
        2
    }
    fn jacobian_is_constant(&self) -> bool {
        true
    }
    fn time_derivative(
        &self,
        _: f64,
        _: &[Complex64],
        output: &mut [Complex64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.fill(Complex64::new(0., 0.));
        Ok(Some(()))
    }
    fn rhs(
        &self,
        _: f64,
        y: &[Complex64],
        out: &mut [Complex64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = y[0] * Complex64::new(-200., 0.);
        out[1] = y[1] * Complex64::new(-4., 0.);
        Ok(())
    }
    fn sparse_jacobian(
        &self,
        _: f64,
        _: &[Complex64],
        _: StructuralPatternView<'_>,
        out: &mut [Complex64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        out.copy_from_slice(&[Complex64::new(-200., 0.), Complex64::new(-4., 0.)]);
        Ok(Some(()))
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = MethodOfLinesPlan::new(2, Some(&[2., 4.]), Some(&[0.5, 8.]))?;
    let mut work = plan.workspace()?;
    let system = plan.bind(&Residual, &mut work)?;
    let mut initial = vec![Complex64::new(0., 0.); 2];
    plan.encode_into(
        &[Complex64::new(1., 1.), Complex64::new(1., -1.)],
        &mut initial,
    )?;
    let solver =
        RosenbrockW::new(RosenbrockMethod::Ros34Pw2)?.with_backend(OdeStageBackend::Sparse {
            pattern: StructuralPattern::new(2, 2, vec![0, 1, 2], vec![0, 1])?,
            limits: SparseLuLimits::new(8, 16_384),
        });
    let options = OdeOptions::new()
        .with_error_tolerance(plan.error_tolerance(&ErrorTolerance::scalar(
            1e-9,
            1e-8,
            NormSelection::Infinity,
        )?)?)
        .with_initial_step(0.001);
    let ode = (solver, options);
    let problem = OdeProblem::new(&system, 0.1);
    let mut state = OdeState::new(0., Vector::from_vec(initial));
    let mut physical = vec![Complex64::new(0., 0.); 2];
    let mut trajectory = PdeTrajectory::<f64, Complex64>::new();
    let raw = ode.0.advance(
        &problem,
        &ode.1,
        &mut state,
        &mut |t, y: &[Complex64]| {
            plan.decode_into(y, &mut physical)?;
            trajectory.observe(PdeObservation::new(t, &physical, None)?)
        },
        &mut ode.0.workspace(&problem, &ode.1)?,
        ExecutionContext::sequential(),
    )?;
    let report = PdeSolveReport::try_from(&raw)?;
    let mut values = vec![Complex64::new(0., 0.); system.field_len()];
    system.reconstruct_into(state.time(), state.values(), &mut values)?;
    println!(
        "{} at t={}: {:?} ({} streamed states)",
        report.termination(),
        report.final_time(),
        values,
        trajectory.times().len()
    );
    Ok(())
}
