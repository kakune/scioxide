//! A scalar Lasso solved directly and through the explicit split `x-z=0`.
use scioxide_core::{AdjointOperator, LinearOperator, SciOxideError, SolveContext};
use scioxide_optimize::{
    Admm, AdmmBlock, AdmmBlockUpdate, AdmmOptions, AdmmProblem, AdmmStart, AdmmUpdateInput,
    AdmmUpdateReport, CompositeProblem, Fista, ProximalMap, ProximalOptions, SmoothGradient,
};

struct Quadratic;
impl SmoothGradient<f64> for Quadratic {
    fn dimension(&self) -> usize {
        1
    }
    fn value(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(0.5 * (x[0] - 3.0).powi(2))
    }
    fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<(), SciOxideError> {
        out[0] = x[0] - 3.0;
        Ok(())
    }
}
struct L1;
impl ProximalMap<f64> for L1 {
    fn apply_into(&self, point: &[f64], step: f64, out: &mut [f64]) -> Result<(), SciOxideError> {
        out[0] = point[0].signum() * (point[0].abs() - step).max(0.0);
        Ok(())
    }
}
struct Scale(f64);
impl LinearOperator<f64> for Scale {
    fn rows(&self) -> usize {
        1
    }
    fn cols(&self) -> usize {
        1
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(
        &self,
        input: &[f64],
        out: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        out[0] = self.0 * input[0];
        Ok(())
    }
}
impl AdjointOperator<f64> for Scale {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_adjoint_into(
        &self,
        input: &[f64],
        out: &mut [f64],
        work: &mut [f64],
    ) -> Result<(), SciOxideError> {
        self.apply_into(input, out, work)
    }
}
struct XUpdate;
impl AdmmBlockUpdate<f64> for XUpdate {
    fn update_into(
        &self,
        input: AdmmUpdateInput<'_, f64>,
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdmmUpdateReport<f64>, SciOxideError> {
        assert_eq!(input.block(), AdmmBlock::X);
        out[0] =
            (3.0 + input.rho() * (input.other()[0] - input.scaled_dual()[0])) / (1.0 + input.rho());
        Ok(AdmmUpdateReport::new(true, 0.0))
    }
}
struct ZUpdate;
impl AdmmBlockUpdate<f64> for ZUpdate {
    fn update_into(
        &self,
        input: AdmmUpdateInput<'_, f64>,
        out: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<AdmmUpdateReport<f64>, SciOxideError> {
        assert_eq!(input.block(), AdmmBlock::Z);
        L1.apply_into(
            &[input.other()[0] + input.scaled_dual()[0]],
            1.0 / input.rho(),
            out,
        )?;
        Ok(AdmmUpdateReport::new(true, 0.0))
    }
}

fn main() -> Result<(), SciOxideError> {
    let fista = Fista::new(ProximalOptions::new(1.0))?;
    let direct = fista.minimize(&CompositeProblem::new(&Quadratic, &L1), &[0.0])?;
    assert!(direct.is_success());

    // Original objective: (x-3)^2/2 + |z|, subject to x-z=0.
    let a = Scale(1.0);
    let b = Scale(-1.0);
    let rhs = [0.0];
    let problem = AdmmProblem::new(&a, &b, &rhs, &XUpdate, &ZUpdate)?;
    let split = Admm::new(AdmmOptions::new(1.0))?
        .solve(&problem, AdmmStart::new(&[0.0], &[0.0], &[0.0]))?;
    assert!(split.is_success());
    println!(
        "FISTA x={}, ADMM x={}, primal residual={}",
        direct.x()[0],
        split.x()[0],
        split.primal_norm().unwrap_or(f64::NAN)
    );
    Ok(())
}
