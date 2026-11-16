//! Context-free callbacks preserve protocol results and prepared allocation bounds.
#![allow(clippy::unwrap_used)]
use scioxide_core::{SciOxideError, SolveContext};
use scioxide_linalg::Matrix;
use scioxide_ode::*;

struct Model;
impl DaeSystem<f64> for Model {
    fn dimension(&self) -> usize {
        1
    }
    fn residual(
        &self,
        _: f64,
        y: &[f64],
        d: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = d[0] + y[0];
        Ok(())
    }
}
impl SdeSystem<f64> for Model {
    fn dimension(&self) -> usize {
        1
    }
    fn noise_dimension(&self) -> usize {
        1
    }
    fn drift(
        &self,
        _: f64,
        y: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = -y[0];
        Ok(())
    }
    fn diffusion_product(
        &self,
        _: f64,
        _: &[f64],
        dw: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = dw[0];
        Ok(())
    }
}
impl SeparableHamiltonian<f64> for Model {
    fn dimension(&self) -> usize {
        1
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
    fn gradient(
        &self,
        q: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out.copy_from_slice(q);
        Ok(())
    }
}
impl BvpSystem<f64> for Model {
    fn dimension(&self) -> usize {
        1
    }
    fn rhs(
        &self,
        _: f64,
        _: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = 1.;
        Ok(())
    }
    fn boundary(
        &self,
        a: &[f64],
        _: &[f64],
        _: &[f64],
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        out[0] = a[0];
        Ok(())
    }
}

#[test]
fn bvp_adapter_matches_manual_system() {
    let system = BvpFn::new(
        1,
        |_: f64, _: &[f64], dy: &mut [f64]| {
            dy[0] = 1.;
            Ok(())
        },
        |a: &[f64], _: &[f64], r: &mut [f64]| {
            r[0] = a[0];
            Ok(())
        },
    );
    assert_eq!(system.parameter_dimension(), 0);
    let plan = BvpPlan::new(1, 0, &[0., 1.], BvpOptions::default()).unwrap();
    let guess = Matrix::zeros(2, 1).unwrap();
    let adapter = plan
        .solve(&BvpProblem::numerical(&system), guess.view(), &[])
        .unwrap();
    let manual = plan
        .solve(&BvpProblem::numerical(&Model), guess.view(), &[])
        .unwrap();
    for t in [0., 0.25, 0.5, 1.] {
        let mut a = [0.];
        let mut b = [0.];
        adapter.evaluate_into(t, &mut a).unwrap();
        manual.evaluate_into(t, &mut b).unwrap();
        assert_eq!(a, b);
        assert!((a[0] - t).abs() < 1e-12);
    }
}
