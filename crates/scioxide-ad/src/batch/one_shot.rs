//! One-shot means over a table of items, with and without the parameter gradient.

use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};
use scioxide_linalg::Vector;

use super::{BatchEstimate, BatchFunction, BatchOptions, BatchOutputs, BatchPlan, BatchReduction};

/// Checks the table, builds the plan and evaluates the mean.
fn mean<S: RealScalar, F: BatchFunction<S>>(
    function: &F,
    parameters: &[S],
    data: &[S],
    items: usize,
    gradient: Option<&mut [S]>,
) -> Result<BatchEstimate<S>, SciOxideError> {
    let width = data.len().checked_div(items).unwrap_or(0);
    let plan = BatchPlan::<8>::new(
        items,
        parameters.len(),
        0,
        width,
        BatchReduction::Mean,
        BatchOptions::new(),
    )?;
    if width * items != data.len() {
        return Err(SciOxideError::dimension_mismatch(
            [items, data.len().div_ceil(items)],
            data.len(),
        ));
    }
    let mut workspace = plan.workspace::<S>(1)?;
    let outputs = match gradient {
        Some(gradient) => BatchOutputs::new().with_parameter_gradient(gradient),
        None => BatchOutputs::new(),
    };
    let none: &[S] = &[];
    plan.evaluate_into(
        function,
        parameters,
        none,
        data,
        outputs,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )
}

/// Returns the mean of `f(θ, dᵢ)` over `items` items, with `θ = parameters` and
/// `dᵢ` row `i` of the row-major `items × width` table `data`.
///
/// This is [`BatchPlan`] with eight lanes, [`BatchOptions::new`] (blocks of 256
/// items, `KinkPolicy::Midpoint`, `NonFinitePolicy::Propagate`), no per-item active
/// inputs, the mean, one worker and `ExecutionContext::sequential_native()`; the
/// estimate has the bits that configuration returns and reports `BatchMode::Value`.
/// Sums, per-item inputs or values, gradient errors, other lane counts and workers
/// go through [`BatchPlan`]. The call allocates one workspace.
///
/// # Errors
///
/// - `InvalidInput` (argument `items`, `Empty`) if `items` is zero.
/// - `DimensionMismatch` if `data.len()` is not a multiple of `items`.
/// - The errors of `function`, unchanged; `AllocationFailure` and `SizeOverflow`
///   of the plan and workspace.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{BatchFunction, Real, batch_mean};
/// use scioxide_core::SciOxideError;
///
/// /// θ·zᵢ²: the mean over z = (1, 2, 3) is 14θ/3.
/// struct Scaled;
/// impl BatchFunction for Scaled {
///     type Shared<R: Real> = R;
///     fn prepare<R: Real>(&self, p: &[R]) -> Result<R, SciOxideError> { Ok(p[0]) }
///     fn item<R: Real>(&self, t: &R, _: &[R], z: &[R::Base]) -> Result<R, SciOxideError> {
///         Ok(*t * R::passive(z[0]) * R::passive(z[0]))
///     }
/// }
/// let estimate = batch_mean(&Scaled, &[2.0], &[1.0, 2.0, 3.0], 3)?;
/// assert!((estimate.value() - 28.0 / 3.0).abs() < 1e-14);
/// # Ok::<(), SciOxideError>(())
/// ```
pub fn batch_mean<S, F>(
    function: &F,
    parameters: &[S],
    data: &[S],
    items: usize,
) -> Result<BatchEstimate<S>, SciOxideError>
where
    S: RealScalar,
    F: BatchFunction<S>,
{
    mean(function, parameters, data, items, None)
}

/// Returns the mean of [`batch_mean`] and its gradient with respect to the
/// parameters.
///
/// The configuration is that of [`batch_mean`]; the gradient comes from forward
/// mode for at most four parameters and from reverse mode beyond, and the value has
/// the same bits in every mode. With no parameters the gradient is empty and the
/// estimate reports `BatchMode::Value`. The call allocates one workspace, its
/// tapes and the gradient.
///
/// # Errors
///
/// The errors of [`batch_mean`]; `ResourceLimit` for a recording beyond 2²⁸ words.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{BatchFunction, Real, batch_mean_and_grad};
/// use scioxide_core::SciOxideError;
///
/// /// sin(θ)·zᵢ: the mean over z = (0.5, 1.5, 1.0) is sin θ, its gradient cos θ.
/// struct Sample;
/// impl BatchFunction for Sample {
///     type Shared<R: Real> = R;
///     fn prepare<R: Real>(&self, p: &[R]) -> Result<R, SciOxideError> { Ok(p[0].sin()) }
///     fn item<R: Real>(&self, s: &R, _: &[R], z: &[R::Base]) -> Result<R, SciOxideError> {
///         Ok(*s * R::passive(z[0]))
///     }
/// }
/// let (estimate, gradient) = batch_mean_and_grad(&Sample, &[0.4], &[0.5, 1.5, 1.0], 3)?;
/// assert!((estimate.value() - 0.4_f64.sin()).abs() < 1e-15);
/// assert!((gradient.as_slice()[0] - 0.4_f64.cos()).abs() < 1e-15);
/// # Ok::<(), SciOxideError>(())
/// ```
pub fn batch_mean_and_grad<S, F>(
    function: &F,
    parameters: &[S],
    data: &[S],
    items: usize,
) -> Result<(BatchEstimate<S>, Vector<S>), SciOxideError>
where
    S: RealScalar,
    F: BatchFunction<S>,
{
    let mut gradient = Vector::zeros(parameters.len())?;
    let estimate = mean(
        function,
        parameters,
        data,
        items,
        Some(gradient.as_mut_slice()),
    )?;
    Ok((estimate, gradient))
}
