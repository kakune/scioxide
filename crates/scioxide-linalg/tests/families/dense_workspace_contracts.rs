//! Public requirements, execution and transactional output contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{ExecutionContext, ExecutionPolicy, SciOxideError, SciOxideErrorKind};
use scioxide_linalg::*;

fn sequential() -> LinalgExecutionOptions {
    LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential)
}

#[test]
fn matrix_vector_short_borrowed_scratch_preserves_output() -> Result<(), SciOxideError> {
    use scioxide_linalg::VectorViewMut;
    let matrix = MatrixView::from_slice(3, 1, &[1.0_f64, 2.0, 3.0])?;
    let mut output = [7.0; 3];
    let mut scratch = [0.0; 2];
    let result = matrix.mul_vec_into(
        &[1.0][..],
        &sequential(),
        VectorViewMut::from_slice(&mut output),
        &mut LinalgWorkspace::from_scalar_slice(&mut scratch),
        ExecutionContext::sequential(),
    );
    assert!(matches!(result, Err(error) if error.kind() == SciOxideErrorKind::DimensionMismatch));
    assert_eq!(output, [7.0; 3]);
    Ok(())
}

#[test]
fn product_preflight_preserves_output_on_resource_and_clock_failures() -> Result<(), SciOxideError>
{
    let matrix = Matrix::<f64>::identity(2)?;
    let mut output = Matrix::from_vec(2, 2, vec![19.0; 4])?;
    let mut workspace = LinalgWorkspace::new();
    for (options, expected) in [
        (
            sequential().with_byte_limit(1),
            SciOxideErrorKind::ResourceLimit,
        ),
        (
            sequential().with_time_limit(core::time::Duration::from_secs(1)),
            SciOxideErrorKind::CapabilityUnavailable,
        ),
    ] {
        let result = matrix.view().mul_mat_into(
            &matrix,
            &options,
            output.view_mut(),
            &mut workspace,
            ExecutionContext::sequential(),
        );
        assert!(matches!(result, Err(error) if error.kind() == expected));
        assert_eq!(output.as_slice(), &[19.0; 4]);
        assert_eq!(workspace.retained_bytes()?, 0);
    }
    Ok(())
}

#[test]
fn empty_products_do_not_read_unused_nonfinite_operands() -> Result<(), SciOxideError> {
    let left = MatrixView::<f64>::from_slice(0, 1, &[])?;
    let right = MatrixView::from_slice(1, 1, &[f64::NAN])?;
    let output = left.mul_mat_with_options(right, &sequential())?;
    assert_eq!((output.rows(), output.cols()), (0, 1));
    Ok(())
}
