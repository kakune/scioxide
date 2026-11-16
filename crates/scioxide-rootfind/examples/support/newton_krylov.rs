//! Manufactured tridiagonal system with the exact root x_i = 1.
use scioxide_core::{JacobianOperator, Preconditioner, ResidualOperator, SciOxideError};

pub struct Tridiagonal(pub usize);
fn diagonal(i: usize) -> f64 {
    f64::from((i % 8) as u8) + 1.0
}
impl ResidualOperator<f64> for Tridiagonal {
    fn input_dimension(&self) -> usize {
        self.0
    }
    fn output_dimension(&self) -> usize {
        self.0
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
        for (i, out) in y.iter_mut().enumerate() {
            *out = diagonal(i) * (x[i] * x[i] - 1.);
            if i > 0 {
                *out += 0.05 * (x[i - 1] - 1.);
            }
            if i + 1 < self.0 {
                *out += 0.05 * (x[i + 1] - 1.);
            }
        }
        Ok(())
    }
}
impl JacobianOperator<f64> for Tridiagonal {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(
        &self,
        x: &[f64],
        v: &[f64],
        y: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        for (i, out) in y.iter_mut().enumerate() {
            *out = 2. * diagonal(i) * x[i] * v[i];
            if i > 0 {
                *out += 0.05 * v[i - 1];
            }
            if i + 1 < self.0 {
                *out += 0.05 * v[i + 1];
            }
        }
        Ok(())
    }
}
impl Preconditioner<f64> for Tridiagonal {
    fn dimension(&self) -> usize {
        self.0
    }
    fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn precondition_into(
        &self,
        x: &[f64],
        y: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), SciOxideError> {
        y.iter_mut()
            .zip(x)
            .enumerate()
            .for_each(|(i, (y, x))| *y = x / (2. * diagonal(i)));
        Ok(())
    }
}
