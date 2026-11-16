//! Monotone bicubic Hermite derivatives: PCHIP slopes reduced by sufficient
//! Bernstein sign conditions, with zero mixed derivatives.
use crate::checked::{add, div, mul, sub};
use crate::piecewise::pchip_slopes;
use alloc::vec::Vec;
use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{RealScalar, SciOxideError};
use scioxide_poly::backend::polynomial_basis::linear_multiply_difference;

/// Returns the first partial derivatives at the nodes; mixed partials are zero.
/// The caller has checked the coordinate and value shapes. Monotonicity errors
/// name the grid values `z`.
pub(super) fn derivatives<S: RealScalar>(
    x: &[S],
    y: &[S],
    z: &[S],
) -> Result<(Vec<S>, Vec<S>), SciOxideError> {
    let nx = x.len();
    let ny = y.len();
    // Monotonicity is required globally per axis: every row (resp. column) must
    // be non-decreasing, or every one non-increasing. The detected sign lets
    // the limiter below treat both cases as the increasing one; an axis that
    // is constant everywhere defaults to +1.
    let direction = |axis: usize| -> Result<S, SciOxideError> {
        let mut sign = S::ZERO;
        for j in 0..ny {
            for i in 0..nx {
                let next = if axis == 0 && i + 1 < nx {
                    Some(j * nx + i + 1)
                } else if axis == 1 && j + 1 < ny {
                    Some((j + 1) * nx + i)
                } else {
                    None
                };
                if let Some(next) = next {
                    let a = z[j * nx + i];
                    let b = z[next];
                    let current = if b > a {
                        S::ONE
                    } else if b < a {
                        -S::ONE
                    } else {
                        S::ZERO
                    };
                    if current != S::ZERO {
                        if sign != S::ZERO && sign != current {
                            return Err(crate::error::invalid(
                                scioxide_core::argument!("z"),
                                "grid must have a consistent monotonicity direction along each axis",
                            ));
                        }
                        sign = current;
                    }
                }
            }
        }
        Ok(if sign == S::ZERO { S::ONE } else { sign })
    };
    let sx = direction(0)?;
    let sy = direction(1)?;
    // Start from 1-D PCHIP slopes along every row (fx) and column (fy); they
    // already keep each grid line monotone, and the per-line factors below only
    // shrink them further.
    let mut fx = try_with_capacity(z.len(), scioxide_core::quantity!("x_derivatives"))?;
    for row in z.chunks_exact(nx) {
        fx.extend(pchip_slopes(x, row)?);
    }
    let mut fy = try_filled(z.len(), S::ZERO, scioxide_core::quantity!("y_derivatives"))?;
    let mut column = try_with_capacity(ny, scioxide_core::quantity!("grid_column"))?;
    for i in 0..nx {
        column.clear();
        column.extend((0..ny).map(|j| z[j * nx + i]));
        for (j, d) in pchip_slopes(y, &column)?.into_iter().enumerate() {
            fy[j * nx + i] = d;
        }
    }
    let mut ax = try_filled(nx, S::ONE, scioxide_core::quantity!("x_slope_factors"))?;
    let mut ay = try_filled(ny, S::ONE, scioxide_core::quantity!("y_slope_factors"))?;
    let three = S::TWO + S::ONE;
    // Independent nonnegative terms are limited against the available edge
    // difference. Ignoring helpful transverse terms makes all later reductions
    // safe. Each x-column / y-row shares its factor, preserving equal-slope
    // cancellation and C1 continuity across all adjoining patches.
    // For each patch edge along x, `budget` is the oriented value increase,
    // `a` and `b` are the Bernstein control offsets h f_x / 3 contributed by
    // the end slopes, and `adverse` is the part of the transverse-slope
    // difference that can push an interior Bernstein difference negative. If
    // their sum exceeds the budget, all involved factors are reduced to the
    // ratio, which keeps every Bernstein difference nonnegative (a sufficient
    // monotonicity condition). The second loop is the same for edges along y.
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            for top in [false, true] {
                let row = j + usize::from(top);
                let l = row * nx + i;
                let r = l + 1;
                let budget = mul(sx, sub(z[r], z[l])?)?;
                let a = mul(
                    sx,
                    linear_multiply_difference(fx[l] / three, x[i + 1], x[i])?,
                )?;
                let b = mul(
                    sx,
                    linear_multiply_difference(fx[r] / three, x[i + 1], x[i])?,
                )?;
                let transverse = sub(fy[l] / three, fy[r] / three)?;
                let adverse = mul(
                    sx,
                    linear_multiply_difference(
                        if top { -transverse } else { transverse },
                        y[j + 1],
                        y[j],
                    )?,
                )?
                .max(S::ZERO);
                let total = add(add(a, b)?, adverse)?;
                if total > budget {
                    let factor = div(budget, total)?;
                    ax[i] = ax[i].min(factor);
                    ax[i + 1] = ax[i + 1].min(factor);
                    if adverse > S::ZERO {
                        ay[row] = ay[row].min(factor);
                    }
                }
            }
            for right in [false, true] {
                let col = i + usize::from(right);
                let b = j * nx + col;
                let t = b + nx;
                let budget = mul(sy, sub(z[t], z[b])?)?;
                let a = mul(
                    sy,
                    linear_multiply_difference(fy[b] / three, y[j + 1], y[j])?,
                )?;
                let c = mul(
                    sy,
                    linear_multiply_difference(fy[t] / three, y[j + 1], y[j])?,
                )?;
                let transverse = sub(fx[b] / three, fx[t] / three)?;
                let adverse = mul(
                    sy,
                    linear_multiply_difference(
                        if right { -transverse } else { transverse },
                        x[i + 1],
                        x[i],
                    )?,
                )?
                .max(S::ZERO);
                let total = add(add(a, c)?, adverse)?;
                if total > budget {
                    let factor = div(budget, total)?;
                    ay[j] = ay[j].min(factor);
                    ay[j + 1] = ay[j + 1].min(factor);
                    if adverse > S::ZERO {
                        ax[col] = ax[col].min(factor);
                    }
                }
            }
        }
    }
    for (j, (xs, ys)) in fx
        .chunks_exact_mut(nx)
        .zip(fy.chunks_exact_mut(nx))
        .enumerate()
    {
        for (i, (dx, dy)) in xs.iter_mut().zip(ys).enumerate() {
            *dx = mul(*dx, ax[i])?;
            *dy = mul(*dy, ay[j])?;
        }
    }
    Ok((fx, fy))
}
