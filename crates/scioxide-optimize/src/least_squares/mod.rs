//! Nonlinear least squares: minimizes `½ ‖r(x)‖²`, optionally with a robust loss,
//! by Gauss–Newton or Levenberg–Marquardt steps.

mod complex;
mod engine;
mod kernels;
mod operator;
mod options;
mod result;
mod solver;
mod storage;
pub use complex::ComplexLeastSquares;
pub use options::{
    LeastSquaresGlobalization, LeastSquaresLinearSolver, LeastSquaresLoss, LeastSquaresOptions,
    LeastSquaresStopping,
};
pub use result::LeastSquaresResult;
pub use solver::{
    LeastSquares, LeastSquaresAccepted, LeastSquaresProgress, LeastSquaresWorkspace,
    LeastSquaresWorkspaceRequest, LeastSquaresWorkspaceRequirements,
};
pub(crate) use storage::{LeastSquaresShape, LeastSquaresStorage};

use crate::DerivativeWorkspace;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::Matrix;

use crate::traits::LeastSquaresProblem;

pub(crate) fn jacobian_into<S, P>(
    problem: &P,
    x: &[S],
    residual: &[S],
    strategy: DerivativeStrategy,
    output: &mut Matrix<S>,
    finite_difference: &mut DerivativeWorkspace<S>,
    execution: scioxide_core::ExecutionContext<'_>,
) -> Result<usize, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    P: LeastSquaresProblem<S> + ?Sized,
{
    let rows = problem.residual_dim();
    let columns = problem.param_dim();
    if x.len() != columns
        || residual.len() != rows
        || output.rows() != rows
        || output.cols() != columns
    {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(rows, columns),
            scioxide_core::Dimensions::two(output.rows(), output.cols()),
        ));
    }
    crate::derivatives::matrix_into(
        strategy,
        x,
        residual,
        output,
        finite_difference,
        |analytic, out| {
            if analytic {
                problem.jacobian_into(x, out)
            } else {
                problem.automatic_jacobian_into(x, out)
            }
        },
        |point, out| problem.residual_into(point, out),
        execution,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::suboptimal_flops)]
mod tests {
    use super::*;
    use scioxide_core::SciOxideError;

    struct LinearLSP;

    impl LeastSquaresProblem<f64> for LinearLSP {
        fn param_dim(&self) -> usize {
            2
        }
        fn residual_dim(&self) -> usize {
            3
        }
        fn residual_into(&self, x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
            // r(x) = Ax - b where A=[[1,0],[0,1],[1,1]], b=[1,2,3]
            output.copy_from_slice(&[x[0] - 1.0, x[1] - 2.0, x[0] + x[1] - 3.0]);
            Ok(())
        }
    }

    #[test]
    fn test_finite_difference_jacobian_ls_linear() {
        let problem = LinearLSP;
        let x = [0.5, 1.5];
        let r_x = problem.residual(&x).unwrap();
        let mut jac = Matrix::zeros(3, 2).unwrap();
        let mut workspace = crate::DerivativeWorkspace::new(
            scioxide_differentiate::DerivativeOperation::Jacobian,
            2,
            3,
        )
        .unwrap();
        jacobian_into(
            &problem,
            &x,
            &r_x,
            DerivativeStrategy::Forward,
            &mut jac,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();

        assert_eq!(jac.rows(), 3);
        assert_eq!(jac.cols(), 2);

        // Expected Jacobian: [[1,0],[0,1],[1,1]]
        assert!((jac.view().get(0, 0).unwrap() - 1.0).abs() < 1e-6);
        assert!((jac.view().get(0, 1).unwrap()).abs() < 1e-6);
        assert!((jac.view().get(1, 0).unwrap()).abs() < 1e-6);
        assert!((jac.view().get(1, 1).unwrap() - 1.0).abs() < 1e-6);
        assert!((jac.view().get(2, 0).unwrap() - 1.0).abs() < 1e-6);
        assert!((jac.view().get(2, 1).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_finite_difference_jacobian_ls_at_origin() {
        let problem = LinearLSP;
        let x = [0.0, 0.0];
        let r_x = problem.residual(&x).unwrap();
        let mut jac = Matrix::zeros(3, 2).unwrap();
        let mut workspace = crate::DerivativeWorkspace::new(
            scioxide_differentiate::DerivativeOperation::Jacobian,
            2,
            3,
        )
        .unwrap();
        jacobian_into(
            &problem,
            &x,
            &r_x,
            DerivativeStrategy::Forward,
            &mut jac,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();

        assert!((jac.view().get(0, 0).unwrap() - 1.0).abs() < 1e-6);
        assert!((jac.view().get(1, 1).unwrap() - 1.0).abs() < 1e-6);
    }
}
