//! A genuine singular-mass system: x'=-x, 0=z-x*x-t.
use scioxide_core::{ExecutionContext, SciOxideError, SolveContext};
use scioxide_linalg::MatrixViewMut;
use scioxide_ode::{
    DaeBdf, DaeInitialization, DaeOptions, DaePartition, DaeProblem, DaeState, DaeSystem,
    OdeStageBackend,
};
struct Decay;
impl DaeSystem<f64> for Decay {
    fn dimension(&self) -> usize {
        2
    }
    fn residual(
        &self,
        t: f64,
        y: &[f64],
        d: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = d[0] + y[0];
        out[1] = y[1] - y[0] * y[0] - t;
        Ok(())
    }
    fn derivatives(
        &self,
        _: f64,
        y: &[f64],
        _: &[f64],
        a: &mut MatrixViewMut<'_, f64>,
        b: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        a.set(0, 0, 1.)?;
        a.set(0, 1, 0.)?;
        a.set(1, 0, -2. * y[0])?;
        a.set(1, 1, 1.)?;
        b.set(0, 0, 1.)?;
        b.set(0, 1, 0.)?;
        b.set(1, 0, 0.)?;
        b.set(1, 1, 0.)?;
        Ok(Some(()))
    }
}
fn main() -> Result<(), SciOxideError> {
    let partition = DaePartition::new(&[true, false], &[true, false])?;
    // Project the inconsistent guess onto the constraint manifold, then integrate.
    let plan = (
        DaeBdf::new(partition, OdeStageBackend::Dense)?,
        DaeOptions::new()
            .with_numerics(
                scioxide_ode::DaeNumericalOptions::new()
                    .with_initialization(DaeInitialization::Project),
            )
            .with_initial_step(0.05),
    );
    let mut workspace = plan.0.workspace()?;
    let mut state = DaeState::from_slices(0., &[1., 0.], &[0., 0.])?;
    let report = plan.0.advance(
        &DaeProblem::new(&Decay, 0.25),
        &plan.1,
        &mut state,
        &mut |t: f64, y: &[f64], _: &[f64]| {
            println!("accepted t={t:.4}: {y:?}");
            Ok(())
        },
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    if let Some(consistency) = report.consistency() {
        println!(
            "projection corrected the algebraic state by {}",
            consistency.state_correction()
        );
    }
    println!(
        "{:?}: t={}, y={:?}, algebraic residual={}",
        report.termination(),
        report.final_time(),
        state.values(),
        report.algebraic_residual()
    );
    Ok(())
}
