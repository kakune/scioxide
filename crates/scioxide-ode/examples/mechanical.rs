//! Fixed-step oscillator and constrained free motion on a circle.
use scioxide_core::{ExecutionContext, ResidualTolerance, SciOxideError, SolveContext};
use scioxide_linalg::{Matrix, MatrixViewMut};
use scioxide_ode::{
    HolonomicConstraints, MechanicalOptions, PhaseSpace, Rattle, RattleOptions, RattleProblem,
    SeparableHamiltonian, StormerVerlet,
};
struct Oscillator;
impl SeparableHamiltonian<f64> for Oscillator {
    fn dimension(&self) -> usize {
        2
    }
    fn gradient(
        &self,
        q: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(q);
        Ok(())
    }
    fn velocity(
        &self,
        p: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(p);
        Ok(())
    }
}
struct Circle;
impl HolonomicConstraints<f64> for Circle {
    fn dimension(&self) -> usize {
        2
    }
    fn constraints(&self) -> usize {
        1
    }
    fn gradient(
        &self,
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.fill(0.);
        Ok(())
    }
    fn values(
        &self,
        q: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = (q[0] * q[0] + q[1] * q[1] - 1.) / 2.;
        Ok(())
    }
    fn jacobian(
        &self,
        q: &[f64],
        out: &mut MatrixViewMut<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.set(0, 0, q[0])?;
        out.set(0, 1, q[1])
    }
}
fn main() -> Result<(), SciOxideError> {
    let options = (0.01, 100, MechanicalOptions::new());
    let initial = PhaseSpace::new(vec![1., 0.], vec![0., 1.])?;
    let solution = {
        let (step, steps, options) = options;
        StormerVerlet.solve(&Oscillator, step, steps, &options, initial.clone())
    }?;
    println!(
        "Verlet: {:?}, q={:?}",
        solution.report().termination(),
        solution.state().position()
    );

    let mass = Matrix::identity(2)?;
    let inverse_mass = mass.view().try_finite()?;
    let problem = RattleProblem::new(&Circle, &inverse_mass);
    let tolerance = ResidualTolerance::new(1e-13, 0.)?;
    let plan = (
        options.0,
        options.1,
        RattleOptions::new()
            .with_integration(options.2)
            .with_position_tolerance(tolerance, 1.)
            .with_momentum_tolerance(tolerance, 1.),
    );
    // Successive calls continue the trajectory and stream every committed step.
    let mut workspace = Rattle.workspace(&problem)?;
    let mut state = initial;
    let mut furthest = 0.0_f64;
    for _ in 0..3 {
        let report = Rattle.advance(
            &problem,
            plan.0,
            plan.1,
            &plan.2,
            &mut state,
            &mut |_: usize, q: &[f64], _: &[f64]| {
                furthest = furthest.max(q[1]);
                Ok(())
            },
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        println!(
            "RATTLE: {:?}, q={:?}, residual={:?}",
            report.termination(),
            state.position(),
            report.position_residual()
        );
    }
    println!("largest q[1] = {furthest}");
    Ok(())
}
