//! Public requirements, execution and transactional output contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Capacity, ExecutionContext, ExecutionPolicy, SciOxideError, SciOxideErrorKind,
};
use scioxide_linalg::*;

fn sequential() -> LinalgExecutionOptions {
    LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential)
}

#[test]
fn factor_extraction_accepts_strides_and_preserves_output_on_overflow() -> Result<(), SciOxideError>
{
    use scioxide_linalg::{LuDecomposition, MatrixViewMut};
    let ordinary = Matrix::from_vec(2, 2, vec![2.0_f64, 1.0, 1.0, 3.0])?;
    let factor = LuDecomposition::factor_with_options(&ordinary, &sequential())?;
    let expected = factor.to_l_matrix()?;
    let mut storage = [7.0; 8];
    factor.l_matrix_into(MatrixViewMut::from_slice_with_strides(
        2,
        2,
        &mut storage,
        4,
        2,
    )?)?;
    assert_eq!(
        [storage[0], storage[2], storage[4], storage[6]],
        expected.as_slice()
    );
    assert_eq!([storage[1], storage[3], storage[5], storage[7]], [7.0; 4]);
    let extreme = Matrix::from_vec(2, 2, vec![f64::MAX, f64::MAX, f64::MAX, -f64::MAX])?;
    let factor = LuDecomposition::factor_with_options(&extreme, &sequential())?;
    let before = storage;
    assert!(
        factor
            .u_matrix_into(MatrixViewMut::from_slice_with_strides(
                2,
                2,
                &mut storage,
                4,
                2
            )?)
            .is_err()
    );
    assert_eq!(storage, before);
    Ok(())
}

#[test]
fn lu_requirements_cover_factor_and_solve_capacity_and_atomic_failures() -> Result<(), SciOxideError>
{
    use scioxide_linalg::{LuDecomposition, LuWorkspaceRequest};
    let options = sequential();
    let request = LuWorkspaceRequest::factor(3, &options);
    let requirements = LuDecomposition::<f64>::workspace_requirements(&request)?.max(
        &LuDecomposition::<f64>::workspace_requirements(&LuWorkspaceRequest::solve(3, &options))?,
    );
    let mut workspace = LinalgWorkspace::new();
    assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
    workspace.try_reserve(&requirements)?;
    assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    let matrix = Matrix::<f64>::identity(3)?;
    let mut factor = LuDecomposition::factor_with_options(&matrix, &options)?;
    let before = factor.determinant()?;
    let limited = options.with_byte_limit(1);
    let error = LuDecomposition::factor_into(
        &matrix,
        &limited,
        &mut factor,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
    assert_eq!(factor.determinant()?, before);
    assert_eq!(
        LuDecomposition::<f64>::workspace_requirements(&LuWorkspaceRequest::factor(
            usize::MAX,
            &options
        ))
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::SizeOverflow
    );
    let mut short = [0.0; 1];
    let mut reserved = LinalgWorkspace::from_scalar_slice(&mut short);
    assert_eq!(
        LuDecomposition::factor_into(
            &matrix,
            &options,
            &mut factor,
            &mut reserved,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(factor.determinant()?, before);
    let singular = Matrix::<f64>::zeros(3, 3)?;
    assert_eq!(
        LuDecomposition::factor_into(
            &singular,
            &options,
            &mut factor,
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::Singular
    );
    assert_eq!(factor.determinant()?, before);
    Ok(())
}

#[test]
fn cholesky_triangle_ignores_unused_entries_and_projection_preserves_subnormals()
-> Result<(), SciOxideError> {
    use scioxide_core::Triangle;
    use scioxide_linalg::{
        CholeskyDecomposition, CholeskyPivoting, HermitianProjectionPolicy, VectorViewMut,
    };
    let data = [4.0, f64::NAN, f64::NAN, f64::NAN, 0.0, f64::NAN, 9.0];
    let input = MatrixView::from_slice_with_strides(2, 2, &data, 4, 2)?;
    let options = sequential();
    let factor = CholeskyDecomposition::factor_with_options(
        input,
        scioxide_linalg::HermitianInput::Triangle(Triangle::Lower),
        CholeskyPivoting::None,
        &options,
    )?;
    let mut out = [17.0, 23.0, 19.0];
    factor
        .solve_into(
            &[8.0, 27.0][..],
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            VectorViewMut::from_slice_with_stride(&mut out, 2, 2)?,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential(),
        )
        .and_then(|report| {
            if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                Ok(())
            } else {
                Err(scioxide_core::SciOxideError::singular(None))
            }
        })?;
    assert!((out[0] - 2.0).abs() < 1e-14);
    assert_eq!(out[1], 23.0);
    assert!((out[2] - 3.0).abs() < 1e-14);
    let tiny = [f64::from_bits(1)];
    let factor = CholeskyDecomposition::factor_with_options(
        MatrixView::from_slice(1, 1, &tiny)?,
        scioxide_linalg::HermitianInput::Projected(HermitianProjectionPolicy::Always),
        CholeskyPivoting::None,
        &options,
    )?;
    assert!(factor.l_matrix().as_slice()[0] > 0.0);
    assert_eq!(
        factor.projection_report().unwrap().absolute_frobenius(),
        0.0
    );
    assert_eq!(
        CholeskyDecomposition::factor_with_options(
            MatrixView::from_slice(1, 1, &[1.0])?,
            scioxide_linalg::HermitianInput::Exact,
            CholeskyPivoting::None,
            &options.with_time_limit(core::time::Duration::from_secs(1))
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    Ok(())
}

#[test]
fn matrix_solve_dimension_errors_identify_the_invalid_rhs() -> Result<(), SciOxideError> {
    use scioxide_linalg::{
        CholeskyDecomposition, CholeskyPivoting, HouseholderDecomposition, HouseholderOrientation,
        HouseholderPivoting, LdlDecomposition, LuDecomposition, RankPolicy,
    };
    let a = Matrix::<f64>::identity(2)?;
    let rhs = Matrix::from_vec(1, 1, vec![1.0])?;
    let lu = LuDecomposition::factor_with_options(&a, &sequential())?;
    let cholesky = CholeskyDecomposition::factor_with_options(
        &a,
        scioxide_linalg::HermitianInput::Exact,
        CholeskyPivoting::None,
        &sequential(),
    )?;
    let ldl = LdlDecomposition::factor_with_options(
        &a,
        scioxide_linalg::HermitianInput::Exact,
        &sequential(),
    )?;
    let qr = HouseholderDecomposition::factor_with_options(
        &a,
        HouseholderOrientation::Qr,
        HouseholderPivoting::None,
        &sequential(),
    )?;
    let mut output = Matrix::from_vec(2, 1, vec![7.0; 2])?;
    let mut workspace = LinalgWorkspace::new();
    for family in 0..4 {
        let error = match family {
            0 => lu
                .solve_matrix_into(
                    &rhs,
                    &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
                    output.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .and_then(|report| {
                    if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                        Ok(())
                    } else {
                        Err(scioxide_core::SciOxideError::singular(None))
                    }
                }),
            1 => cholesky
                .solve_matrix_into(
                    &rhs,
                    &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
                    output.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .and_then(|report| {
                    if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                        Ok(())
                    } else {
                        Err(scioxide_core::SciOxideError::singular(None))
                    }
                }),
            2 => ldl
                .solve_matrix_into(
                    &rhs,
                    &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
                    output.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .and_then(|report| {
                    if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                        Ok(())
                    } else {
                        Err(scioxide_core::SciOxideError::singular(None))
                    }
                }),
            _ => qr
                .solve_matrix_into(
                    &rhs,
                    &(scioxide_linalg::DirectSolveOptions::new(RankPolicy::Exact))
                        .with_execution_options(sequential()),
                    output.view_mut(),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .map(|_| ()),
        }
        .unwrap_err();
        assert!(
            matches!(error.view(), scioxide_core::SciOxideErrorView::DimensionMismatch { expected, actual, .. } if expected == [2, 1] && actual == [1, 1])
        );
        assert_eq!(output.as_slice(), &[7.0; 2]);
    }
    Ok(())
}
