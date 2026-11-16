//! Checked byte counts for banded spline construction and parametric trials.

use scioxide_core::backend::{checked_add, checked_mul};
use scioxide_core::{RealScalar, SciOxideError};
use scioxide_linalg::{
    BandedCholeskyDecomposition, LinalgExecutionOptions, StructuredWorkspaceRequest,
};

use super::penalized::SplineFitDiagnostics;

/// Scalars for the ordinary and integral knot vectors that a basis stores.
pub(super) fn basis_scalars(controls: usize, degree: usize) -> Result<usize, SciOxideError> {
    let q = scioxide_core::quantity!("spline_preparation_bytes");
    let knots = checked_add(checked_add(controls, degree, q)?, 1, q)?;
    checked_add(checked_mul(2, knots, q)?, 2, q)
}

/// Conservative constructor payload, including transferred input storage.
pub(super) fn penalized_bytes<S: RealScalar>(
    controls: usize,
    degree: usize,
    samples: usize,
    order: usize,
) -> Result<usize, SciOxideError> {
    let q = scioxide_core::quantity!("spline_preparation_bytes");
    let k = degree.max(order);
    // Normal matrix and factor: two lower bands. Independent diagnostics retain
    // both halves of the original band. The factor candidate lives in scratch.
    let bands = checked_add(checked_mul(4, k, q)?, 3, q)?;
    let differences = if order == 0 {
        0
    } else {
        checked_add(order, 1, q)?
    };
    let scalars = [
        basis_scalars(controls, degree)?,
        checked_mul(controls, bands, q)?,
        checked_mul(2, samples, q)?,
        checked_add(degree, 1, q)?,
        differences,
    ]
    .into_iter()
    .try_fold(0, |total, count| checked_add(total, count, q))?;
    let scratch = BandedCholeskyDecomposition::<S>::workspace_requirements(
        &StructuredWorkspaceRequest::factor(controls, k, k, &LinalgExecutionOptions::new()),
    )?
    .required_bytes::<S>()?;
    checked_add(
        checked_mul(scalars, core::mem::size_of::<S>(), q)?,
        scratch,
        q,
    )
}

/// Bytes for one cubic trial while the last feasible fit is still stored. The
/// limit is checked before the basis is built, so this takes only sizes.
pub(crate) fn parametric_bytes<S: RealScalar>(
    samples: usize,
    controls: usize,
    dimension: usize,
) -> Result<usize, SciOxideError> {
    let q = scioxide_core::quantity!("spline_workspace_bytes");
    let preparation = penalized_bytes::<S>(controls, 3, samples, 2)?;
    // Parameters/weights and their copies, residuals, breakpoints, two retained
    // spline bases and component buffers for workspace, trial and feasible fit.
    let scalars = [
        checked_mul(8, samples, q)?,
        checked_mul(8, checked_mul(controls, dimension, q)?, q)?,
        checked_mul(3, basis_scalars(controls, 3)?, q)?,
    ]
    .into_iter()
    .try_fold(0, |total, count| checked_add(total, count, q))?;
    let solve = BandedCholeskyDecomposition::<S>::workspace_requirements(
        &StructuredWorkspaceRequest::solve_matrix(
            controls,
            dimension,
            &LinalgExecutionOptions::new(),
        ),
    )?
    .required_bytes::<S>()?;
    let diagnostics = checked_mul(
        checked_mul(2, dimension, q)?,
        core::mem::size_of::<SplineFitDiagnostics<S>>(),
        q,
    )?;
    checked_add(
        checked_add(
            preparation,
            checked_mul(scalars, core::mem::size_of::<S>(), q)?,
            q,
        )?,
        checked_add(solve, diagnostics, q)?,
        q,
    )
}
