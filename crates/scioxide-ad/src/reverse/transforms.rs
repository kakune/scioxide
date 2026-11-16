//! One-shot reverse transforms with default options.
//!
//! Each builds a fresh tape, uses `ExecutionContext::sequential_native()` and
//! allocates its result. The empty tape cannot hold the first recording, so the
//! function runs twice: once to size the tape and once to record. Reuse a
//! [`Tape`] to run it once and to avoid the allocations.

use alloc::vec::Vec;

use scioxide_core::{ExecutionContext, RealScalar, SciOxideError};
use scioxide_linalg::{Matrix, Vector};
use scioxide_sparse::{CsrMatrix, derivative_coloring};

use super::{Tape, TapeOptions, Var};
use crate::Dual;

fn zeros<S: RealScalar>(len: usize) -> Result<Vec<S>, SciOxideError> {
    scioxide_core::backend::try_filled(len, S::ZERO, scioxide_core::quantity!("result"))
}

/// Returns the value and gradient of a scalar function at `point`.
///
/// One reverse sweep after the two runs of the function that every one-shot
/// makes (see the module docs). A caller with a generic function writes
/// `value_and_grad(|x| Ok(f(x)), &point)`.
///
/// ```
/// use core::sync::atomic::{AtomicUsize, Ordering};
/// use scioxide_ad::{Real, value_and_grad};
/// let runs = AtomicUsize::new(0);
/// let count = || runs.fetch_add(1, Ordering::Relaxed);
/// value_and_grad(|x| { count(); Ok(x[0] * x[0]) }, &[3.0])?;
/// assert_eq!(runs.into_inner(), 2); // size the fresh tape, then record
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// The errors of [`Tape::gradient_into`], and `AllocationFailure` for the result.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, value_and_grad};
///
/// fn rosenbrock<R: Real>(x: &[R]) -> R {
///     x.windows(2).map(|w| (w[1] - w[0] * w[0]).powi(2) * 100.0 + (-w[0] + 1.0).powi(2)).sum()
/// }
/// let (value, gradient) = value_and_grad(|x| Ok(rosenbrock(x)), &[1.0, 1.0, 1.0])?;
/// assert_eq!(value, 0.0);
/// assert!(gradient.as_slice().iter().all(|g| g.abs() < 1e-14));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn value_and_grad<S, F>(function: F, point: &[S]) -> Result<(S, Vector<S>), SciOxideError>
where
    S: RealScalar,
    F: for<'t> Fn(&[Var<'t, S>]) -> Result<Var<'t, S>, SciOxideError>,
{
    let mut tape = Tape::new(TapeOptions::new());
    let mut gradient = zeros(point.len())?;
    let value = tape.gradient_into(
        function,
        point,
        &mut gradient,
        ExecutionContext::sequential_native(),
    )?;
    Ok((value, Vector::from_vec(gradient)))
}

/// Returns the gradient of a scalar function at `point`.
///
/// # Errors
///
/// The errors of [`value_and_grad`].
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, grad};
/// let g = grad(|x| Ok(x[0] * x[1].exp()), &[2.0, 0.0])?;
/// assert_eq!(g.as_slice(), [1.0, 2.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn grad<S, F>(function: F, point: &[S]) -> Result<Vector<S>, SciOxideError>
where
    S: RealScalar,
    F: for<'t> Fn(&[Var<'t, S>]) -> Result<Var<'t, S>, SciOxideError>,
{
    value_and_grad(function, point).map(|(_, gradient)| gradient)
}

/// Returns the reverse product `Jᵀ·cotangent` of a vector function at `point`.
///
/// `function` pushes its outputs onto the vector it receives.
///
/// # Errors
///
/// - `DimensionMismatch` if `cotangent` does not have one entry per output.
/// - The errors of [`Tape::linearize`] and [`Linearization::vjp_into`](crate::Linearization::vjp_into).
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, vjp};
/// let g = vjp(|x, y| { y.push(x[0] * x[1]); y.push(x[0].sin()); Ok(()) }, &[0.0, 3.0], &[1.0, 2.0])?;
/// assert_eq!(g.as_slice(), [5.0, 0.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn vjp<S, F>(function: F, point: &[S], cotangent: &[S]) -> Result<Vector<S>, SciOxideError>
where
    S: RealScalar,
    F: for<'t> Fn(&[Var<'t, S>], &mut Vec<Var<'t, S>>) -> Result<(), SciOxideError>,
{
    let context = ExecutionContext::sequential_native();
    let mut tape = Tape::new(TapeOptions::new());
    let mut linear = tape.linearize(function, point, context)?;
    let mut out = zeros(point.len())?;
    linear.vjp_into(cotangent, &mut out, context)?;
    Ok(Vector::from_vec(out))
}

/// Returns the dense Jacobian (outputs × inputs) of a vector function at `point`.
///
/// # Errors
///
/// The errors of [`Linearization::jacobian_into`](crate::Linearization::jacobian_into).
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, jacobian};
/// let j = jacobian(|x, y| { y.push(x[0] * x[1]); Ok(()) }, &[2.0, 5.0])?;
/// assert_eq!((j.rows(), j.cols()), (1, 2));
/// assert_eq!(j.as_slice(), [5.0, 2.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn jacobian<S, F>(function: F, point: &[S]) -> Result<Matrix<S>, SciOxideError>
where
    S: RealScalar,
    F: for<'t> Fn(&[Var<'t, S>], &mut Vec<Var<'t, S>>) -> Result<(), SciOxideError>,
{
    let context = ExecutionContext::sequential_native();
    let mut tape = Tape::new(TapeOptions::new());
    let mut linear = tape.linearize(function, point, context)?;
    let rows = linear.outputs();
    let mut out = zeros(rows * point.len())?;
    linear.jacobian_into(&mut out, context)?;
    Matrix::from_vec(rows, point.len(), out)
}

/// Returns the Hessian-vector product `H·direction` of a scalar function.
///
/// The function is recorded on duals seeded `Dual::new(xᵢ, vᵢ)`, so one sweep
/// returns the gradient in the values and `H·v` in the derivatives.
///
/// # Errors
///
/// The errors of [`Tape::hvp_into`], and `AllocationFailure` if the result cannot
/// be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, hvp};
/// let h = hvp(|x| Ok(x[0].powi(3) * x[1]), &[1.0, 2.0], &[1.0, 1.0])?;
/// // H = [[6xy, 3x²], [3x², 0]] = [[12, 3], [3, 0]]
/// assert_eq!(h.as_slice(), [15.0, 3.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn hvp<S, F>(function: F, point: &[S], direction: &[S]) -> Result<Vector<S>, SciOxideError>
where
    S: RealScalar,
    F: for<'t> Fn(&[Var<'t, Dual<S>>]) -> Result<Var<'t, Dual<S>>, SciOxideError>,
{
    let mut tape = Tape::new(TapeOptions::new());
    let mut product = zeros(point.len())?;
    tape.hvp_into(
        function,
        point,
        direction,
        &mut product,
        ExecutionContext::sequential_native(),
    )?;
    Ok(Vector::from_vec(product))
}

/// Returns the dense Hessian of a scalar function at `point`, one recording per
/// unit direction.
///
/// For `n > 0` inputs, the function runs `n + 1` times: the first direction sizes the
/// fresh tape and records again, and the tape then holds every later recording.
/// With no inputs, it records twice to validate the function and returns a
/// `0 × 0` matrix.
///
/// # Errors
///
/// The errors of [`Tape::hvp_into`]; `SizeOverflow` if the matrix size overflows,
/// and `AllocationFailure` if the result cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, hessian};
/// let h = hessian(|x| Ok(x[0] * x[0] * x[1]), &[3.0, 2.0])?;
/// assert_eq!(h.as_slice(), [4.0, 6.0, 6.0, 0.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn hessian<S, F>(function: F, point: &[S]) -> Result<Matrix<S>, SciOxideError>
where
    S: RealScalar,
    F: for<'t> Fn(&[Var<'t, Dual<S>>]) -> Result<Var<'t, Dual<S>>, SciOxideError>,
{
    let n = point.len();
    let context = ExecutionContext::sequential_native();
    let mut tape = Tape::new(TapeOptions::new());
    let len = scioxide_core::backend::checked_mul(n, n, scioxide_core::quantity!("hessian"))?;
    let mut out = zeros(len)?;
    if n == 0 {
        tape.hvp_seeded(&function, point, |_| S::ZERO, |_, _| {}, context)?;
    }
    for column in 0..n {
        tape.hvp_seeded(
            &function,
            point,
            |k| if k == column { S::ONE } else { S::ZERO },
            |_, adjoints| {
                for (row, adjoint) in adjoints.iter().enumerate() {
                    out[row * n + column] = adjoint.derivative();
                }
            },
            context,
        )?;
    }
    Matrix::from_vec(n, n, out)
}

/// Returns the sparse Jacobian (outputs × inputs) of a vector function at `point` as
/// a CSR matrix.
///
/// `function` pushes its outputs, as for [`jacobian`]. The call records the
/// function on a fresh tape, takes
/// [`Linearization::jacobian_pattern`](crate::Linearization::jacobian_pattern), colors it
/// with `scioxide_sparse::derivative_coloring` and runs one tangent sweep per color
/// ([`Linearization::sparse_jacobian_into`](crate::Linearization::sparse_jacobian_into)),
/// so the stored values equal those of a dense Jacobian built by tangent sweeps bit
/// for bit. The pattern is valid for the points that take the same branches. For
/// repeated use keep the pattern and coloring and call the `_into` form. A function
/// with no outputs gives a 0 × n matrix.
///
/// # Errors
///
/// - The errors of [`Tape::linearize`] and of the `_into` form.
/// - `NonFinite` if a stored derivative is NaN or infinite, whatever the
///   [`NonFinitePolicy`](crate::NonFinitePolicy); the `_into` form writes it.
/// - `AllocationFailure` and `SizeOverflow` for the tape, the coloring and the
///   matrix.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, sparse_jacobian};
/// // y = (x₀x₁, x₁ + x₂) at (2, 3, 4): J = [[3, 2, 0], [0, 1, 1]].
/// let j = sparse_jacobian(|x, y| { y.extend([x[0] * x[1], x[1] + x[2]]); Ok(()) }, &[2.0, 3.0, 4.0])?;
/// assert_eq!(j.iter().collect::<Vec<_>>(), [(0, 0, 3.0), (0, 1, 2.0), (1, 1, 1.0), (1, 2, 1.0)]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn sparse_jacobian<S, F>(function: F, point: &[S]) -> Result<CsrMatrix<S>, SciOxideError>
where
    S: RealScalar,
    F: for<'t> Fn(&[Var<'t, S>], &mut Vec<Var<'t, S>>) -> Result<(), SciOxideError>,
{
    let cx = ExecutionContext::sequential_native();
    let mut tape = Tape::<S>::new(TapeOptions::new());
    let mut linear = tape.linearize(function, point, cx)?;
    let pattern = linear.jacobian_pattern()?;
    let coloring = derivative_coloring(pattern.view())?;
    let mut values = zeros(pattern.view().col_indices().len())?;
    linear.sparse_jacobian_into(&coloring, &mut values, cx)?;
    CsrMatrix::from_pattern(pattern, values)
}

/// Returns the sparse Hessian of a scalar function at `point` as a CSR matrix.
///
/// `function` is the closure of [`hessian`]. The call takes
/// [`Tape::hessian_pattern`] on a fresh `Tape<Dual<S>>`, colors it with
/// `scioxide_sparse::derivative_coloring` and records the function once per color
/// ([`Tape::sparse_hessian_into`]), so the stored values equal [`hessian`] bit for
/// bit. The pattern is valid for the points that take the same branches. For
/// repeated use keep the pattern and coloring and call the `_into` form.
///
/// # Errors
///
/// - The errors of [`Tape::hessian_pattern`] and [`Tape::sparse_hessian_into`].
/// - `NonFinite` if a stored derivative is NaN or infinite, whatever the
///   [`NonFinitePolicy`](crate::NonFinitePolicy); the `_into` form writes it.
/// - `AllocationFailure` and `SizeOverflow` for the tape, the coloring and the
///   matrix.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, sparse_hessian};
/// // f = x₀²x₁ + x₂² at (1, 2, 3): H = [[4, 2, 0], [2, 0, 0], [0, 0, 2]].
/// let h = sparse_hessian(|x| Ok(x[0] * x[0] * x[1] + x[2] * x[2]), &[1.0, 2.0, 3.0])?;
/// assert_eq!(h.iter().collect::<Vec<_>>(), [(0, 0, 4.0), (0, 1, 2.0), (1, 0, 2.0), (2, 2, 2.0)]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn sparse_hessian<S, F>(function: F, point: &[S]) -> Result<CsrMatrix<S>, SciOxideError>
where
    S: RealScalar,
    F: for<'t> Fn(&[Var<'t, Dual<S>>]) -> Result<Var<'t, Dual<S>>, SciOxideError>,
{
    let cx = ExecutionContext::sequential_native();
    let mut tape = Tape::<Dual<S>>::new(TapeOptions::new());
    let pattern = tape.hessian_pattern(&function, point, cx)?;
    let coloring = derivative_coloring(pattern.view())?;
    let mut values = zeros(pattern.view().col_indices().len())?;
    tape.sparse_hessian_into(&function, point, &coloring, &mut values, cx)?;
    CsrMatrix::from_pattern(pattern, values)
}
