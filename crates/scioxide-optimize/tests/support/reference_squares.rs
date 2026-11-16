//! Supplied MGH objective/residual derivatives from independent test programs.
use super::reference_program::{Jet, Program, Token};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_linalg::Matrix;
use scioxide_optimize::{LeastSquaresProblem, ObjectiveFunction};
pub struct SumOfSquares<S> {
    pub dimension: usize,
    pub residuals: Vec<Program<S>>,
}
impl<S: RealScalar> SumOfSquares<S> {
    pub fn new(dimension: usize, residuals: &[Vec<Token>]) -> Self {
        Self {
            dimension,
            residuals: residuals
                .iter()
                .map(|p| Program::new(p, dimension))
                .collect(),
        }
    }
    pub fn value(&self, x: &[S], first: Option<usize>, second: Option<usize>) -> Jet<S> {
        self.residuals
            .iter()
            .fold(Jet::constant(S::ZERO), |sum, p| {
                let r = p.evaluate(x, first, second);
                sum + r * r
            })
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> ObjectiveFunction<S> for SumOfSquares<S> {
    fn dim(&self) -> usize {
        self.dimension
    }
    fn eval(&self, x: &[S]) -> Result<S, SciOxideError> {
        Ok(self.value(x, None, None).value)
    }
    fn gradient_into(&self, x: &[S], out: &mut [S]) -> Result<bool, SciOxideError> {
        for (i, v) in out.iter_mut().enumerate() {
            *v = self.value(x, Some(i), None).first;
        }
        Ok(true)
    }
    fn hessian_into(&self, x: &[S], out: &mut Matrix<S>) -> Result<bool, SciOxideError> {
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                out.as_mut_slice()[i * self.dimension + j] = self.value(x, Some(i), Some(j)).mixed;
            }
        }
        Ok(true)
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> LeastSquaresProblem<S> for SumOfSquares<S> {
    fn param_dim(&self) -> usize {
        self.dimension
    }
    fn residual_dim(&self) -> usize {
        self.residuals.len()
    }
    fn residual_into(&self, x: &[S], out: &mut [S]) -> Result<(), SciOxideError> {
        for (v, p) in out.iter_mut().zip(&self.residuals) {
            *v = p.evaluate(x, None, None).value;
        }
        Ok(())
    }
    fn jacobian_into(&self, x: &[S], out: &mut Matrix<S>) -> Result<bool, SciOxideError> {
        for (row, p) in self.residuals.iter().enumerate() {
            for j in 0..self.dimension {
                out.as_mut_slice()[row * self.dimension + j] = p.evaluate(x, Some(j), None).first;
            }
        }
        Ok(true)
    }
}
