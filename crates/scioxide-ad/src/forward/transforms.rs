//! Forward-mode one-shots: a Jacobian-vector product and a Taylor expansion.

use alloc::vec::Vec;

use scioxide_core::backend::try_with_capacity;
use scioxide_core::{RealScalar, SciOxideError, quantity};
use scioxide_linalg::Vector;

use super::{Dual, Jet};

/// Returns the Jacobian-vector product `J(p)·d` of `function` at `point` in the
/// direction `direction`.
///
/// `function` maps the inputs `xᵢ = pᵢ + dᵢ ε` ([`Dual`]) to its outputs, which it
/// pushes to the vector it receives (empty on entry); output `k` of the result is
/// the derivative of output `k` of `function`. One forward pass costs a small
/// constant times one evaluation, and the derivatives are those of `Dual`, exact up
/// to the rounding of each rule. The call allocates its inputs and outputs.
///
/// # Errors
///
/// - `DimensionMismatch` if `direction.len() != point.len()`.
/// - `AllocationFailure` if the inputs or the result cannot be allocated.
/// - The first error returned by `function`, unchanged.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, jvp};
/// // f(x, y) = (x·y, sin x); J·(1, 2) = (y + 2x, cos x).
/// let tangent = jvp(
///     |x, out| {
///         out.extend([x[0] * x[1], x[0].sin()]);
///         Ok(())
///     },
///     &[0.5, 3.0],
///     &[1.0, 2.0],
/// )?;
/// assert!((tangent.as_slice()[0] - 4.0).abs() < 1e-15);
/// assert!((tangent.as_slice()[1] - 0.5_f64.cos()).abs() < 1e-15);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// A. Griewank and A. Walther, *Evaluating Derivatives*, 2nd ed., SIAM, 2008, ch. 3.
pub fn jvp<S: RealScalar, F>(
    function: F,
    point: &[S],
    direction: &[S],
) -> Result<Vector<S>, SciOxideError>
where
    F: Fn(&[Dual<S>], &mut Vec<Dual<S>>) -> Result<(), SciOxideError>,
{
    crate::support::check_len(direction.len(), point.len())?;
    let mut inputs = try_with_capacity(point.len(), quantity!("inputs"))?;
    inputs.extend(
        point
            .iter()
            .zip(direction)
            .map(|(&value, &tangent)| Dual::new(value, tangent)),
    );
    let mut outputs = Vec::new();
    function(&inputs, &mut outputs)?;
    let mut tangent = try_with_capacity(outputs.len(), quantity!("tangent"))?;
    tangent.extend(outputs.iter().map(|output| output.derivative()));
    Ok(Vector::from_vec(tangent))
}

/// Returns the normalized Taylor coefficients `cₖ = (1/k!) dᵏ/dtᵏ f(p + t·d)` at
/// `t = 0`, `k < N`, of `function` along the line through `point` in the direction
/// `direction`.
///
/// Each input is the [`Jet`] `xᵢ = [pᵢ, dᵢ, 0, …]`. `c₀` is the value, `c₁` the
/// directional derivative and `2·c₂` the curvature `dᵀ∇²f d`. A nonlinear operation
/// costs `O(N²)` operations; the accuracy is that of [`Jet`]. The call allocates
/// its inputs.
///
/// # Errors
///
/// - `DimensionMismatch` if `direction.len() != point.len()`.
/// - `AllocationFailure` if the inputs cannot be allocated.
/// - The first error returned by `function`, unchanged.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, jet};
/// // f(x, y) = x·eʸ along (1, 0) + t·(1, 1): (1 + t)·eᵗ = 1 + 2t + 1.5t² + …
/// let c: [f64; 3] = jet(|x| Ok(x[0] * x[1].exp()), &[1.0, 0.0], &[1.0, 1.0])?;
/// for (actual, expected) in c.into_iter().zip([1.0, 2.0, 1.5]) {
///     assert!((actual - expected).abs() < 1e-15);
/// }
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - A. Griewank and A. Walther, *Evaluating Derivatives*, 2nd ed., SIAM, 2008,
///   ch. 13.
pub fn jet<S: RealScalar, F, const N: usize>(
    function: F,
    point: &[S],
    direction: &[S],
) -> Result<[S; N], SciOxideError>
where
    F: Fn(&[Jet<S, N>]) -> Result<Jet<S, N>, SciOxideError>,
{
    crate::support::check_len(direction.len(), point.len())?;
    let mut inputs = try_with_capacity(point.len(), quantity!("inputs"))?;
    inputs.extend(point.iter().zip(direction).map(|(&value, &tangent)| {
        Jet::new(core::array::from_fn(|k| match k {
            0 => value,
            1 => tangent,
            _ => S::ZERO,
        }))
    }));
    Ok(function(&inputs)?.coefficients())
}
