#![allow(clippy::unwrap_used)]

use scioxide_core::Complex;
use scioxide_core::ExecutionContext;
use scioxide_core::NumericScalar;
use scioxide_core::{Budget, RealScalar};
use scioxide_linalg::{GeneralEigenOptions, GeneralEigenvaluesBuffer, LinalgWorkspace, Matrix};
fn cast<R: RealScalar>(value: f64) -> R {
    R::try_from_f64_lossy(value).unwrap()
}

fn complex_matrix<R: RealScalar>(order: usize, values: &[(f64, f64)]) -> Matrix<Complex<R>>
where
    Complex<R>: NumericScalar<Real = R>,
{
    Matrix::from_vec(
        order,
        order,
        values
            .iter()
            .map(|&(real, imaginary)| Complex::new(cast(real), cast(imaginary)))
            .collect(),
    )
    .unwrap()
}

#[test]
fn shared_complex_kernel_nonconvergence_preserves_eigen_result_and_options() {
    let seed = complex_matrix::<f64>(2, &[(1.0, 1.0), (2.0, -1.0), (0.0, 0.0), (3.0, 0.5)]);
    let mut result = scioxide_linalg::GeneralEigenvaluesResult::compute(&seed)
        .unwrap()
        .into_parts()
        .0;
    let result_before = result.eigenvalues().to_vec();
    let matrix = complex_matrix::<f64>(
        4,
        &[
            (4.0, 1.0),
            (2.0, -1.0),
            (1.0, 0.5),
            (3.0, 1.0),
            (1.0, -0.5),
            (3.0, 2.0),
            (2.0, -1.5),
            (1.0, 0.25),
            (3.0, 1.0),
            (1.0, -1.0),
            (2.0, 0.5),
            (4.0, -0.5),
            (2.0, -1.0),
            (1.0, 0.75),
            (3.0, 1.25),
            (1.0, -2.0),
        ],
    );
    let options = GeneralEigenOptions::default().with_execution_options(
        scioxide_linalg::LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE),
    );
    let options_before = options;
    let mut workspace = LinalgWorkspace::new();

    assert!(matches!(
        (GeneralEigenvaluesBuffer::compute_into(
            &matrix,
            &options,
            &mut result,
            &mut workspace,
            ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
    ));
    assert_eq!(result.eigenvalues(), result_before);
    assert_eq!(options, options_before);
}
