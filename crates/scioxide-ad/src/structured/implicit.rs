//! Derivatives of implicitly defined solutions `F(y, p) = 0`.

use alloc::vec::Vec;

use scioxide_core::{InvalidInputReason, RealScalar, ResidualTolerance, SciOxideError};

use crate::{BaseReal, Dual, Real, solve_into};

/// A residual `F(y, p)` with `n` unknowns `y` and `q` parameters `p`, written once
/// for every number type with scalar masks.
pub trait Residual<S: RealScalar = f64> {
    /// Writes `F(solution, parameters)` (`n` values) to `out`.
    ///
    /// # Errors
    ///
    /// Any error of the residual, returned unchanged.
    fn residual<R: Real<S, Mask = bool>>(
        &self,
        solution: &[R],
        parameters: &[R],
        out: &mut [R],
    ) -> Result<(), SciOxideError>;
}

fn buffer<T: Clone>(len: usize, fill: T) -> Result<Vec<T>, SciOxideError> {
    scioxide_core::backend::try_filled(len, fill, scioxide_core::quantity!("implicit_scratch"))
}

/// Writes the solution `y(p)` of `F(y, p) = 0` with derivatives, given a solution
/// `solution` at the base values of `parameters`.
///
/// The derivatives follow the implicit function theorem `∂y/∂p = −F_y⁻¹·F_p`,
/// applied recursively on the derivative level: the solution one level down is
/// computed by the same function, `F_y` and `F_p` are evaluated on `Dual` with unit
/// tangents, and the result is built with [`Real::combine`]. For nested duals,
/// `MultiDual` and tapes it is exact to the nesting order, so Hessian-vector
/// products through an implicit solution work; a [`Jet`](crate::Jet) gets `c₀` and
/// `c₁`, and NaN above, because `combine` carries one derivative order. The
/// residual is accepted when `‖F(y, p)‖₂` satisfies `tolerance` relative to
/// `‖y‖₂`, both computed without overflow. Cost per level: `n + q` residual
/// evaluations, one `n × n` factorization and `O(n(n + q))` allocated scratch.
///
/// # Errors
///
/// - `DimensionMismatch` if `out` has not `solution.len()` entries.
/// - `InvalidInput` (argument `solution`, `OutOfRange`) if `‖F(y, p)‖₂` misses
///   `tolerance` at the base level, or the residual or the solution is not
///   finite.
/// - `Singular` if `F_y` is singular, at derivative levels only: a call on plain
///   numbers does not form `F_y`.
/// - `SizeOverflow` if a derivative matrix size overflows `usize`.
/// - `AllocationFailure` if the scratch cannot be allocated; the errors of the
///   residual.
///
/// # Examples
///
/// ```
/// use scioxide_ad::{Real, Residual, grad, implicit_solution_into};
/// use scioxide_core::{SciOxideError, ResidualTolerance};
///
/// /// y² − p = 0, so y = √p.
/// struct Root;
/// impl Residual for Root {
///     fn residual<R: Real<f64, Mask = bool>>(&self, y: &[R], p: &[R], out: &mut [R]) -> Result<(), SciOxideError> {
///         out[0] = y[0] * y[0] - p[0];
///         Ok(())
///     }
/// }
/// let tolerance = ResidualTolerance::new(1e-12, 0.0)?;
/// let g = grad(|p| { let mut y = [p[0]]; implicit_solution_into(&Root, &[2.0], p, tolerance, &mut y)?; Ok(y[0]) }, &[4.0])?;
/// assert!((g.as_slice()[0] - 0.25).abs() < 1e-15);
/// # Ok::<(), SciOxideError>(())
/// ```
pub fn implicit_solution_into<S, R, F>(
    residual: &F,
    solution: &[S],
    parameters: &[R],
    tolerance: ResidualTolerance<S>,
    out: &mut [R],
) -> Result<(), SciOxideError>
where
    S: RealScalar,
    R: Real<S, Mask = bool>,
    F: Residual<S>,
{
    let (n, q) = (solution.len(), parameters.len());
    crate::support::check_len(out.len(), n)?;
    if R::LEVEL == 0 {
        let mut y = buffer(n, R::zero())?;
        for (target, &value) in y.iter_mut().zip(solution) {
            *target = R::constant(value);
        }
        let mut f = buffer(n, R::zero())?;
        residual.residual(&y, parameters, &mut f)?;
        // Euclidean norms by `hypot`, so no square overflows.
        let norm = f
            .iter()
            .fold(S::ZERO, |s, v| s.hypot(v.base().lane(0).unwrap_or(S::NAN)));
        let scale = solution.iter().fold(S::ZERO, |s, &v| s.hypot(v));
        let missed = || {
            SciOxideError::invalid_input(
                scioxide_core::argument!("solution"),
                InvalidInputReason::OutOfRange,
                "the residual misses the tolerance at the given solution",
            )
        };
        if !norm.is_finite() || !scale.is_finite() || !tolerance.is_satisfied(norm, scale)? {
            return Err(missed());
        }
        out.copy_from_slice(&y);
        return Ok(());
    }
    let square =
        scioxide_core::backend::checked_mul(n, n, scioxide_core::quantity!("implicit_scratch"))?;
    let rectangle =
        scioxide_core::backend::checked_mul(n, q, scioxide_core::quantity!("implicit_scratch"))?;
    let columns =
        scioxide_core::backend::checked_add(n, q, scioxide_core::quantity!("implicit_scratch"))?;
    let mut primal = buffer(q, <R::Primal>::zero())?;
    for (target, p) in primal.iter_mut().zip(parameters) {
        *target = p.primal();
    }
    let mut y = buffer(n, <R::Primal>::zero())?;
    implicit_solution_into(residual, solution, &primal, tolerance, &mut y)?;
    // Columns of F_y (n) and F_p (q), by unit tangents on duals.
    let mut fy = buffer(square, <R::Primal>::zero())?;
    let mut fp = buffer(rectangle, <R::Primal>::zero())?;
    let mut f = buffer(n, Dual::lift(<R::Primal>::zero()))?;
    let mut y_dual = buffer(n, Dual::lift(<R::Primal>::zero()))?;
    let mut p_dual = buffer(q, Dual::lift(<R::Primal>::zero()))?;
    for column in 0..columns {
        let unit = |k: usize| {
            if k == column {
                <R::Primal>::one()
            } else {
                <R::Primal>::zero()
            }
        };
        for (k, (target, &v)) in y_dual.iter_mut().zip(&y).enumerate() {
            *target = Dual::new(v, unit(k));
        }
        for (k, (target, &v)) in p_dual.iter_mut().zip(&primal).enumerate() {
            *target = Dual::new(v, unit(n + k));
        }
        residual.residual(&y_dual, &p_dual, &mut f)?;
        for (row, value) in f.iter().enumerate() {
            if column < n {
                fy[row * n + column] = value.derivative();
            } else {
                fp[row * q + column - n] = value.derivative();
            }
        }
    }
    // M = −F_y⁻¹·F_p.
    let mut m = buffer(rectangle, <R::Primal>::zero())?;
    solve_into(&fy, n, &fp, q, &mut m)?;
    let mut partials = buffer(q, <R::Primal>::zero())?;
    for (k, target) in out.iter_mut().enumerate() {
        for (j, partial) in partials.iter_mut().enumerate() {
            *partial = -m[k * q + j];
        }
        *target = R::combine(parameters, y[k], &partials)?;
    }
    Ok(())
}
