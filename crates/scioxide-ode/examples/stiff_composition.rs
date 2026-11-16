//! Native complex matrix-free Rosenbrock integration and real Adams/BDF switching.
use scioxide_core::{Budget, Complex, SciOxideError, SolveContext};
use scioxide_linalg::Vector;
use scioxide_ode::*;
struct ComplexDecay;
impl OdeSystem<f64, Complex<f64>> for ComplexDecay {
    fn dimension(&self) -> usize {
        1
    }
    fn time_derivative(
        &self,
        _: f64,
        _: &[Complex<f64>],
        output: &mut [Complex<f64>],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.fill(Complex::new(0., 0.));
        Ok(Some(()))
    }
    fn rhs(
        &self,
        _: f64,
        y: &[Complex<f64>],
        out: &mut [Complex<f64>],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = Complex::new(-100., 10.) * y[0];
        Ok(())
    }
    fn jacobian_product(
        &self,
        t: f64,
        _: &[Complex<f64>],
        v: &[Complex<f64>],
        out: &mut [Complex<f64>],
        _: &mut [Complex<f64>],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        self.rhs(t, v, out, cx).map(Some)
    }
}
fn main() -> Result<(), SciOxideError> {
    let method =
        RosenbrockW::new(RosenbrockMethod::Ros34Pw2)?.with_backend(OdeStageBackend::Krylov {
            iterations: Budget::try_from(20)?,
            restart: Budget::try_from(4)?,
            operator_scratch: 0,
            preconditioner_scratch: 0,
        });
    let plan = (method, OdeOptions::new());
    let solution = plan.0.solve(
        &OdeProblem::new(&ComplexDecay, 0.1),
        &plan.1,
        OdeState::new(0., Vector::from_vec(vec![Complex::new(1., 0.)])),
    )?;
    println!(
        "complex result {:?}: {:?}",
        solution.report().termination(),
        solution.state().values()
    );
    let real = OdeFn::new(1, |_: f64, y: &[f64], out: &mut [f64]| {
        out[0] = -100. * y[0];
        Ok(())
    })?
    .with_jacobian(
        |_: f64, _: &[f64], j: &mut scioxide_linalg::MatrixViewMut<'_, f64>| j.set(0, 0, -100.),
    );
    let automatic = (
        AdamsBdf::adams_bdf()?,
        OdeOptions::new().with_initial_step(0.05),
    );
    let solution = automatic.0.solve(
        &OdeProblem::new(&real, 0.1),
        &automatic.1,
        OdeState::new(0., Vector::from_vec(vec![1.])),
    )?;
    println!(
        "Adams/BDF {:?}, switches={}",
        solution.report().termination(),
        solution.report().method_switches()
    );
    Ok(())
}
