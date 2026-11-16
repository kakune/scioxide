//! Small dense diagonal-block kernels; no dense whole-matrix fallback.
use super::*;
use crate::factor_arithmetic::Arithmetic;
use scioxide_core::AlgebraicScalar;
use scioxide_core::Complex;
use scioxide_core::{Scaled, ScaledValue};
// Extended-range Frobenius norm.
pub(super) fn norm<S: NumericScalar>(x: &[S]) -> Result<Scaled<S::Real>, SciOxideError> {
    let mut sum = Scaled::ZERO;
    for v in x {
        for component in [v.real_component(), v.imaginary_component()] {
            let s = Scaled::from_value(component)?;
            sum = sum.add(s.mul(s)?)?;
        }
    }
    sum.sqrt()
}
/// C -= A B, or C -= A B^H.
pub(super) fn update<S: NumericScalar>(
    c: &mut [S],
    a: &[S],
    b: &[S],
    side: usize,
    adjoint: bool,
    arithmetic: Arithmetic,
) -> Result<(), SciOxideError> {
    for i in 0..side {
        for j in 0..side {
            c[i * side + j] = arithmetic.subtract_products(
                c[i * side + j],
                (0..side).map(|k| {
                    (
                        a[i * side + k],
                        if adjoint {
                            b[j * side + k].conj()
                        } else {
                            b[k * side + j]
                        },
                    )
                }),
            )?;
        }
    }
    Ok(())
}
// Applies the pivot policy to one pivot.
fn repair<S: NumericScalar>(
    value: S,
    floor: Scaled<S::Real>,
    positive: bool,
    index: usize,
    count: &mut usize,
) -> Result<S, SciOxideError> {
    let magnitude = norm(&[value])?;
    let bad = if positive {
        value.real_component() <= S::Real::ZERO
    } else {
        value == S::ZERO
    };
    if positive && value.imaginary_component() != S::Real::ZERO {
        return Err(invalid(
            scioxide_core::argument!("matrix"),
            "Cholesky diagonal must be real",
        ));
    }
    if bad || magnitude.abs_cmp(floor).is_lt() {
        if floor.abs_cmp(Scaled::ZERO).is_eq() {
            return Err(SciOxideError::singular(Some(index)));
        }
        *count += 1;
        if positive || value == S::ZERO {
            return Ok(S::from_real(floor.round()?));
        }
        let z = ScaledValue::from_value(Complex::new(
            value.real_component(),
            value.imaginary_component(),
        ))?
        .mul(ScaledValue::from_components(
            floor.div(magnitude)?,
            Scaled::ZERO,
        ))?
        .round()?;
        return Ok(S::from_components(z.re, z.im));
    }
    Ok(value)
}
// Factors one dense diagonal block.
#[allow(clippy::too_many_arguments)]
pub(super) fn factor<S: NumericScalar>(
    a: &mut [S],
    pivots: &mut [usize],
    b: usize,
    chol: bool,
    floor: Scaled<S::Real>,
    offset: usize,
    repairs: &mut usize,
    arithmetic: Arithmetic,
) -> Result<(), SciOxideError> {
    for k in 0..b {
        if chol {
            let diagonal = arithmetic.subtract_products(
                a[k * b + k],
                (0..k).map(|j| (a[k * b + j], a[k * b + j].conj())),
            )?;
            let d = repair(diagonal, floor, true, offset + k, repairs)?;
            a[k * b + k] = S::from_real(d.real_component().sqrt());
            for i in k + 1..b {
                let residual = arithmetic.subtract_products(
                    a[i * b + k],
                    (0..k).map(|j| (a[i * b + j], a[k * b + j].conj())),
                )?;
                a[i * b + k] = arithmetic.divide(
                    residual,
                    a[k * b + k],
                    crate::helpers::Projection::Strict,
                )?;
            }
            for j in k + 1..b {
                a[k * b + j] = S::ZERO;
            }
        } else {
            let mut pivot = k;
            let mut largest = norm(&[a[k * b + k]])?;
            for i in k + 1..b {
                let mag = norm(&[a[i * b + k]])?;
                if mag.abs_cmp(largest).is_gt() {
                    pivot = i;
                    largest = mag;
                }
            }
            pivots[k] = pivot;
            for j in 0..b {
                a.swap(k * b + j, pivot * b + j);
            }
            a[k * b + k] = repair(a[k * b + k], floor, false, offset + k, repairs)?;
            for i in k + 1..b {
                a[i * b + k] = arithmetic.divide(
                    a[i * b + k],
                    a[k * b + k],
                    crate::helpers::Projection::Strict,
                )?;
                for j in k + 1..b {
                    a[i * b + j] =
                        arithmetic.subtract_product(a[i * b + j], a[i * b + k], a[k * b + j])?;
                }
            }
        }
    }
    Ok(())
}
// Dense LU solve, or a Cholesky triangular solve. trans=true means transpose; conjugation is
// applied independently, allowing all inverse operations.
#[allow(clippy::too_many_arguments)]
pub(super) fn solve<S: NumericScalar>(
    a: &[S],
    p: &[usize],
    x: &mut [S],
    b: usize,
    chol: bool,
    trans: bool,
    conj: bool,
    arithmetic: Arithmetic,
) -> Result<(), SciOxideError> {
    let at = |i: usize, j: usize| {
        let v = a[i * b + j];
        if conj { v.conj() } else { v }
    };
    if chol {
        if !trans {
            for i in 0..b {
                for j in 0..i {
                    x[i] = arithmetic.subtract_product(x[i], at(i, j), x[j])?;
                }
                x[i] = arithmetic.divide(x[i], at(i, i), crate::helpers::Projection::Strict)?;
            }
        } else {
            for i in (0..b).rev() {
                for j in i + 1..b {
                    x[i] = arithmetic.subtract_product(x[i], at(j, i), x[j])?;
                }
                x[i] = arithmetic.divide(x[i], at(i, i), crate::helpers::Projection::Strict)?;
            }
        }
    } else if !trans {
        for (k, &pivot) in p.iter().enumerate() {
            x.swap(k, pivot);
        }
        for i in 0..b {
            for j in 0..i {
                x[i] = arithmetic.subtract_product(x[i], at(i, j), x[j])?;
            }
        }
        for i in (0..b).rev() {
            for j in i + 1..b {
                x[i] = arithmetic.subtract_product(x[i], at(i, j), x[j])?;
            }
            x[i] = arithmetic.divide(x[i], at(i, i), crate::helpers::Projection::Strict)?;
        }
    } else {
        for i in 0..b {
            for j in 0..i {
                x[i] = arithmetic.subtract_product(x[i], at(j, i), x[j])?;
            }
            x[i] = arithmetic.divide(x[i], at(i, i), crate::helpers::Projection::Strict)?;
        }
        for i in (0..b).rev() {
            for j in i + 1..b {
                x[i] = arithmetic.subtract_product(x[i], at(j, i), x[j])?;
            }
        }
        for (k, &pivot) in p.iter().enumerate().rev() {
            x.swap(k, pivot);
        }
    }
    Ok(())
}
// Right block solve.
pub(super) fn right_solve<S: NumericScalar>(
    a: &mut [S],
    diagonal: &[S],
    p: &[usize],
    b: usize,
    chol: bool,
    arithmetic: Arithmetic,
) -> Result<(), SciOxideError> {
    for row in a.chunks_exact_mut(b) {
        // X D = A for LU; X L^H = A for Cholesky.
        solve(diagonal, p, row, b, chol, !chol, chol, arithmetic)?;
    }
    Ok(())
}
