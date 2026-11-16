//! Direct-factor view layouts and clockless owned computation.

use core::time::Duration;
use scioxide_core::{Complex, ExecutionContext, SciOxideError, SciOxideErrorKind};
use scioxide_linalg::{
    LinalgExecutionOptions, LinalgWorkspace, LuDecomposition, Matrix, MatrixView, MatrixViewMut,
    SvdValues, VectorView, VectorViewMut,
};

#[test]
fn lu_factors_and_solves_conjugated_strides_without_touching_padding() -> Result<(), SciOxideError>
{
    let nan = Complex::new(f64::NAN, f64::NAN);
    let zero = Complex::new(0.0, 0.0);
    let one = Complex::new(1.0, 0.0);
    let storage = [
        Complex::new(2.0, 1.0),
        nan,
        Complex::new(1.0, -2.0),
        nan,
        zero,
        nan,
        Complex::new(3.0, -1.0),
    ];
    let a = MatrixView::from_slice_with_strides(2, 2, &storage, 4, 2)?.conjugate();
    let factor = LuDecomposition::factor(a)?;
    let expected = [Complex::new(1.0, 2.0), Complex::new(-2.0, 1.0)];
    let rhs = [
        a.get(0, 0)? * expected[0] + a.get(0, 1)? * expected[1],
        nan,
        a.get(1, 0)? * expected[0] + a.get(1, 1)? * expected[1],
    ];
    let sentinel = Complex::new(99.0, -99.0);
    let mut out = [sentinel; 3];
    let mut workspace = LinalgWorkspace::new();
    factor
        .solve_into(
            VectorView::from_slice_with_stride(&rhs, 2, 2)?,
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            VectorViewMut::from_slice_with_stride(&mut out, 2, 2)?,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .and_then(|report| {
            if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                Ok(())
            } else {
                Err(scioxide_core::SciOxideError::singular(None))
            }
        })?;
    assert!((out[0] - expected[0]).norm() < 1e-13);
    assert!((out[2] - expected[1]).norm() < 1e-13);
    assert_eq!(out[1], sentinel);

    let mut matrix_out = [sentinel; 6];
    factor
        .solve_matrix_into(
            a,
            &scioxide_linalg::DirectSolveOptions::new(scioxide_linalg::RankPolicy::Exact),
            MatrixViewMut::from_slice_with_strides(2, 2, &mut matrix_out, 3, 1)?,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .and_then(|report| {
            if report.termination() == scioxide_linalg::DirectSolveTermination::Accepted {
                Ok(())
            } else {
                Err(scioxide_core::SciOxideError::singular(None))
            }
        })?;
    for (index, expected) in [(0, one), (1, zero), (3, zero), (4, one)] {
        assert!((matrix_out[index] - expected).norm() < 1e-13);
    }
    assert_eq!(matrix_out[2], sentinel);
    assert_eq!(matrix_out[5], sentinel);
    Ok(())
}

#[test]
fn owned_direct_computation_rejects_clockless_time_limits() -> Result<(), SciOxideError> {
    let matrix = Matrix::<f64>::identity(2)?;
    let lu = LuDecomposition::factor_with_options(
        &matrix,
        &LinalgExecutionOptions::new().with_time_limit(Duration::from_secs(1)),
    );
    assert!(matches!(lu, Err(error) if error.kind() == SciOxideErrorKind::CapabilityUnavailable));
    let svd = SvdValues::compute_with_options(
        &matrix,
        &LinalgExecutionOptions::new().with_time_limit(Duration::from_secs(1)),
    );
    assert!(matches!(svd, Err(error) if error.kind() == SciOxideErrorKind::CapabilityUnavailable));
    Ok(())
}
