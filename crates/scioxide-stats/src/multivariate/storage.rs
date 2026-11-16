//! Memory limits and workspace requirements for the multivariate estimators.
use super::{Extended, PreciseCenter};
use alloc::vec::Vec;
use scioxide_core::{ExecutionContext, NumericScalar, SciOxideError};
use scioxide_linalg::{LinalgExecutionOptions, SvdWorkspaceRequest, ThinSvdDecomposition};

/// Checks that `elements` scalars, `centers` compensated centers and
/// `extended_entries` extended-range entries fit in `limit` bytes.
pub(crate) fn reserve_multivariate<T: NumericScalar>(
    elements: usize,
    centers: usize,
    extended_entries: usize,
    limit: usize,
) -> Result<(), SciOxideError> {
    let extra = centers
        .checked_mul(core::mem::size_of::<PreciseCenter>())
        .and_then(|v| {
            extended_entries
                .checked_mul(core::mem::size_of::<Extended>())
                .and_then(|q| v.checked_add(q))
        })
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("multivariate_bytes"))
        })?;
    let payload = scioxide_core::backend::checked_mul(
        elements,
        core::mem::size_of::<T>(),
        scioxide_core::quantity!("multivariate_bytes"),
    )?;
    let required = scioxide_core::backend::checked_add(
        extra,
        payload,
        scioxide_core::quantity!("multivariate_bytes"),
    )?;
    crate::error_helpers::limit(
        required,
        limit,
        scioxide_core::quantity!("multivariate_bytes"),
    )
}

pub(super) fn byte_sum(parts: impl IntoIterator<Item = usize>) -> Result<usize, SciOxideError> {
    parts.into_iter().try_fold(0_usize, |sum, part| {
        sum.checked_add(part).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("multivariate_bytes"))
        })
    })
}

pub(super) fn payload<T>(dimensions: &[usize]) -> Result<usize, SciOxideError> {
    scioxide_core::backend::checked_product(
        dimensions,
        scioxide_core::quantity!("multivariate_bytes"),
    )?
    .checked_mul(core::mem::size_of::<T>())
    .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("multivariate_bytes")))
}

fn center_bytes<T: NumericScalar>(columns: usize) -> Result<usize, SciOxideError> {
    byte_sum([
        payload::<T>(&[columns])?,
        payload::<PreciseCenter>(&[columns])?,
    ])
}

fn factor_bytes<T: NumericScalar>(rows: usize, cols: usize) -> Result<usize, SciOxideError> {
    let k = rows.min(cols);
    byte_sum([
        payload::<T>(&[rows, k])?,
        payload::<T>(&[k, cols])?,
        payload::<T::Real>(&[k])?,
    ])
}

pub(super) fn factor_work_bytes<T: NumericScalar>(
    rows: usize,
    cols: usize,
    context: ExecutionContext<'_>,
) -> Result<usize, SciOxideError> {
    ThinSvdDecomposition::<T>::workspace_requirements(
        &SvdWorkspaceRequest::new(rows, cols, &LinalgExecutionOptions::default())
            .with_worker_bound(context.worker_budget()),
    )?
    .required_bytes::<T>()
}

// These pure shape bounds follow the explicit drop points in the constructors.
// Factors retain their full thin capacities even when analyze_rank truncates them.
pub(super) fn pca_bytes<T: NumericScalar>(
    n: usize,
    p: usize,
    context: ExecutionContext<'_>,
) -> Result<usize, SciOxideError> {
    let k = n.min(p);
    let fixed = byte_sum([center_bytes::<T>(p)?, factor_bytes::<T>(n, p)?])?;
    let preparation = byte_sum([
        payload::<T>(&[n, p])?,
        factor_work_bytes::<T>(n, p, context)?,
    ])?;
    let output = byte_sum([
        payload::<T>(&[p, k])?,
        payload::<T::Real>(&[k])?,
        payload::<scioxide_core::Scaled<f64>>(&[k])?,
        payload::<Option<[f64; 2]>>(&[if T::IS_REAL { p } else { 0 }])?,
    ])?;
    byte_sum([fixed, preparation.max(output)])
}

pub(super) fn cca_bytes<T: NumericScalar>(
    n: usize,
    p: usize,
    q: usize,
    context: ExecutionContext<'_>,
) -> Result<usize, SciOxideError> {
    let (kx, ky) = (n.min(p), n.min(q));
    let k = kx.min(ky);
    let centers = byte_sum([center_bytes::<T>(p)?, center_bytes::<T>(q)?])?;
    let (xf, yf) = (factor_bytes::<T>(n, p)?, factor_bytes::<T>(n, q)?);
    let x_stage = byte_sum([
        payload::<T>(&[n, p])?,
        payload::<T>(&[n, q])?,
        xf,
        factor_work_bytes::<T>(n, p, context)?,
    ])?;
    let y_stage = byte_sum([
        payload::<T>(&[n, q])?,
        xf,
        yf,
        factor_work_bytes::<T>(n, q, context)?,
    ])?;
    let cross = byte_sum([
        xf,
        yf,
        payload::<T>(&[kx, ky])?,
        factor_bytes::<T>(kx, ky)?,
        factor_work_bytes::<T>(kx, ky, context)?,
    ])?;
    let output = byte_sum([
        xf,
        yf,
        factor_bytes::<T>(kx, ky)?,
        payload::<T>(&[p, k])?,
        payload::<T>(&[q, k])?,
        payload::<T::Real>(&[k])?,
    ])?;
    byte_sum([centers, x_stage.max(y_stage).max(cross).max(output)])
}

pub(super) fn admit_preparation(bytes: usize, limit: usize) -> Result<(), SciOxideError> {
    if bytes > limit {
        return Err(SciOxideError::resource_limit(
            scioxide_core::quantity!("multivariate_bytes"),
            bytes,
            limit,
        ));
    }
    Ok(())
}

/// Workspace requirements of a PCA or CCA transform: the output shape and the
/// scratch bytes needed.
///
/// Obtain it from the `workspace_requirements` method of [`PreparedPca`](crate::PreparedPca),
/// [`PreparedCca`](crate::PreparedCca), [`PcaWorkspace`](crate::PcaWorkspace) or
/// [`CcaWorkspace`](crate::CcaWorkspace). See
/// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
#[derive(Clone, Copy, Debug)]
pub struct MultivariateRequirements<T: NumericScalar> {
    scalar: core::marker::PhantomData<T>,
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) copies: usize,
    bytes: usize,
    pub(super) max_bytes: usize,
}
impl<T: NumericScalar> MultivariateRequirements<T> {
    pub(super) fn new(
        rows: usize,
        columns: usize,
        copies: usize,
        max_bytes: usize,
    ) -> Result<Self, SciOxideError> {
        let bytes = rows
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(copies))
            .and_then(|n| n.checked_mul(core::mem::size_of::<T>()))
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("multivariate_workspace"))
            })?;
        crate::error_helpers::limit_bytes(bytes, max_bytes)?;
        Ok(Self {
            scalar: core::marker::PhantomData,
            rows,
            columns,
            copies,
            bytes,
            max_bytes,
        })
    }
    /// Returns the scratch bytes needed, excluding the output matrices.
    pub const fn required_bytes(self) -> usize {
        self.bytes
    }
}
pub(super) fn reserve_stage<T: NumericScalar>(
    values: &mut Vec<T>,
    count: usize,
) -> Result<(), SciOxideError> {
    if values.capacity() < count {
        values
            .try_reserve_exact(count.saturating_sub(values.len()))
            .map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("multivariate_workspace"),
                    count,
                )
            })?;
    }
    values.resize(count, T::ZERO);
    Ok(())
}
