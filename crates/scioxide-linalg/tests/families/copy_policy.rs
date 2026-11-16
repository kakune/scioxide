//! Copy policy, native scratch, and contiguous borrowed preparation contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{ExecutionContext, SciOxideError, SciOxideErrorKind};
use scioxide_linalg::{
    CopyPolicy, DenseWorkspaceRequest, LinalgExecutionOptions, LinalgWorkspace, MatrixView,
    MatrixViewMut, VectorView, VectorViewMut,
};

#[test]
fn copy_rejects_mismatches_and_handles_zero_extents() -> Result<(), SciOxideError> {
    for policy in [CopyPolicy::Preserve, CopyPolicy::RequireFinite] {
        let mut output = [3.0_f64; 2];
        let error = MatrixView::from_slice(1, 1, &[1.0])?
            .copy_into(
                policy,
                &LinalgExecutionOptions::new(),
                MatrixViewMut::from_slice(1, 2, &mut output)?,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
        assert_eq!(output, [3.0; 2]);
        MatrixView::<f64>::from_slice(usize::MAX, 0, &[])?.copy_into(
            policy,
            &LinalgExecutionOptions::new(),
            MatrixViewMut::from_slice(usize::MAX, 0, &mut [])?,
            ExecutionContext::sequential(),
        )?;
    }
    Ok(())
}

#[test]
fn partial_spectrum_accepts_strided_initial_values_and_ignores_padding() -> Result<(), SciOxideError>
{
    use scioxide_linalg::{
        Matrix, OperatorStructure, PartialSpectrumBuffer, PartialSpectrumSolver,
    };
    let matrix = Matrix::from_vec(3, 3, vec![1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])?;
    let plan = PartialSpectrumSolver::new(1).with_structure(OperatorStructure::Hermitian);
    let operator = matrix.view().try_finite()?;
    let mut workspace = plan.workspace(&operator)?;
    let mut expected = PartialSpectrumBuffer::new(3, 1)?;
    let mut actual = PartialSpectrumBuffer::new(3, 1)?;
    let expected_report = plan.solve_into(
        &operator,
        &[1.0, 1.0, 1.0][..],
        &mut expected,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    let initial = [1.0, f64::NAN, 1.0, f64::NAN, 1.0];
    let actual_report = plan.solve_into(
        &operator,
        VectorView::from_slice_with_stride(&initial, 3, 2)?,
        &mut actual,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    assert_eq!(
        actual_report.termination(),
        scioxide_linalg::SpectralTermination::Converged
    );
    assert_eq!(actual_report.termination(), expected_report.termination());
    assert_eq!(actual.eigenvalues().len(), 1);
    for (actual, expected) in actual.eigenvalues().iter().zip(expected.eigenvalues()) {
        assert!((actual.re - expected.re).abs() <= 8.0 * f64::EPSILON * expected.re.abs().max(1.0));
        assert!((actual.im - expected.im).abs() <= 8.0 * f64::EPSILON);
    }
    Ok(())
}

#[test]
fn empty_triangular_products_and_solves_do_not_traverse_unbounded_empty_rows()
-> Result<(), SciOxideError> {
    use scioxide_core::{DiagonalKind, Triangle};
    use scioxide_linalg::{MatrixSide, StructuredOperation};
    let triangle = MatrixView::<f64>::from_slice(0, 0, &[])?;
    for solve in [false, true] {
        let output = MatrixViewMut::from_slice(usize::MAX, 0, &mut [])?;
        let mut workspace = LinalgWorkspace::new();
        let options = LinalgExecutionOptions::new().with_byte_limit(0);
        if solve {
            triangle.triangular_solve_mat_in_place(
                MatrixSide::Right,
                Triangle::Upper,
                DiagonalKind::Unit,
                StructuredOperation::Normal,
                1.0,
                &options,
                output,
                &mut workspace,
                ExecutionContext::sequential(),
            )?;
        } else {
            triangle.triangular_mul_mat_in_place(
                MatrixSide::Right,
                Triangle::Upper,
                DiagonalKind::Unit,
                StructuredOperation::Normal,
                1.0,
                &options,
                output,
                &mut workspace,
                ExecutionContext::sequential(),
            )?;
        }
    }
    Ok(())
}

#[test]
fn native_copy_and_swap_still_enforce_retained_workspace_limits() -> Result<(), SciOxideError> {
    let options = LinalgExecutionOptions::new();
    let mut workspace =
        LinalgWorkspace::<f64>::workspace(&DenseWorkspaceRequest::vector(2, &options))?;
    let limited = options.with_byte_limit(0);
    let mut x = [1.0, 2.0];
    let mut y = [3.0, 4.0];
    for policy in [CopyPolicy::RequireFinite, CopyPolicy::Preserve] {
        assert_eq!(
            VectorView::from_slice(&x)
                .copy_into(
                    policy,
                    &limited,
                    VectorViewMut::from_slice(&mut y),
                    &mut workspace,
                    ExecutionContext::sequential()
                )
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::ResourceLimit
        );
    }
    assert_eq!(
        VectorViewMut::swap_in_place(
            &limited,
            VectorViewMut::from_slice(&mut x),
            VectorViewMut::from_slice(&mut y),
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(x, [1.0, 2.0]);
    assert_eq!(y, [3.0, 4.0]);
    Ok(())
}
