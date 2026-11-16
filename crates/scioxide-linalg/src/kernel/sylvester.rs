//! Block triangular Sylvester and Stein substitution leaves.
//! Extended-range core arithmetic keeps scaling explicit without perturbing pivots.
use super::frames::{Tile, WIDE, exponent_sum};
use super::frames::{extended, scalar};
use super::small::{Arithmetic, ExtendedArithmetic, Ordinary, Singularity, eliminate};
use crate::{MatrixView, kernel::standard, workspace::Extended};
use scioxide_core::ScaledValue;
use scioxide_core::{AlgebraicScalar, Complex, NumericScalar};
use scioxide_core::{RealScalar, SciOxideError};

/// Operands and orientation shared by continuous and discrete substitution.
#[derive(Clone, Copy)]
pub(crate) struct Recurrence<'a, S: NumericScalar> {
    pub(crate) a: MatrixView<'a, S>,
    pub(crate) b: MatrixView<'a, S>,
    pub(crate) sign: S,
    pub(crate) left_adjoint: bool,
    pub(crate) right_adjoint: bool,
    pub(crate) discrete: bool,
}

struct Substitution<'a, V> {
    x: &'a mut [V],
    cached: &'a mut [V],
    coefficient_shift: i32,
    singularity: Singularity,
}

/// One oriented recurrence for continuous Sylvester and discrete Stein leaves.
fn substitute<S: NumericScalar, A: Arithmetic<S>>(
    arithmetic: &A,
    recurrence: Recurrence<'_, S>,
    storage: Substitution<'_, A::Value>,
    admit: &mut impl FnMut() -> Result<(), SciOxideError>,
) -> Result<bool, SciOxideError> {
    let Recurrence {
        a,
        b,
        sign,
        left_adjoint,
        right_adjoint,
        discrete,
    } = recurrence;
    let Substitution {
        x,
        cached,
        coefficient_shift,
        singularity,
    } = storage;
    let (m, n) = (a.rows(), b.rows());
    let av = if left_adjoint { a.adjoint() } else { a };
    let bv = if right_adjoint { b.adjoint() } else { b };
    let coefficient = |x: S| {
        let scaled = standard::scale(x, coefficient_shift);
        if coefficient_shift != 0 && super::frames::lost(x, scaled) {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "local coefficient alignment loses range",
            ));
        }
        arithmetic.value(scaled)
    };
    if discrete {
        cached.fill(A::ZERO);
    }
    let mut col = if right_adjoint { n } else { 0 };
    while if right_adjoint { col > 0 } else { col < n } {
        let (c, q) = block(b, col, right_adjoint);
        let mut row = if left_adjoint { 0 } else { m };
        while if left_adjoint { row < m } else { row > 0 } {
            admit()?;
            let (r, p) = block(a, row, !left_adjoint);
            let mut rhs = [A::ZERO; 4];
            let mut coefficients = [A::ZERO; 16];
            for i in 0..p {
                for j in 0..q {
                    let index = i * q + j;
                    let mut value = x[(r + i) * n + c + j];
                    let cols = if right_adjoint { c + q..n } else { 0..c };
                    for k in cols {
                        value = arithmetic.subtract_product(
                            value,
                            if discrete {
                                cached[(r + i) * n + k]
                            } else {
                                x[(r + i) * n + k]
                            },
                            coefficient(if discrete {
                                bv.get_element(k, c + j)
                            } else {
                                sign * bv.get_element(k, c + j)
                            })?,
                        )?;
                    }
                    for v in 0..if discrete { q } else { 1 } {
                        let rows = if left_adjoint { 0..r } else { r + p..m };
                        if discrete {
                            let mut sum = A::ZERO;
                            for k in rows {
                                sum = arithmetic.add(
                                    sum,
                                    arithmetic.mul(
                                        coefficient(av.get_element(r + i, k))?,
                                        x[k * n + c + v],
                                    )?,
                                )?;
                            }
                            value = arithmetic.subtract_product(
                                value,
                                sum,
                                coefficient(bv.get_element(c + v, c + j))?,
                            )?;
                        } else {
                            for k in rows {
                                value = arithmetic.subtract_product(
                                    value,
                                    coefficient(av.get_element(r + i, k))?,
                                    x[k * n + c + j],
                                )?;
                            }
                        }
                    }
                    rhs[index] = value;
                    if discrete {
                        for u in 0..p {
                            for v in 0..q {
                                let mut value = arithmetic.mul(
                                    coefficient(av.get_element(r + i, r + u))?,
                                    coefficient(bv.get_element(c + v, c + j))?,
                                )?;
                                if i == u && j == v {
                                    value = arithmetic.add(value, arithmetic.value(-S::ONE)?)?;
                                }
                                coefficients[index * 4 + u * q + v] = value;
                            }
                        }
                    } else {
                        for u in 0..p {
                            coefficients[index * 4 + u * q + j] =
                                coefficient(av.get_element(r + i, r + u))?;
                        }
                        for v in 0..q {
                            let k = index * 4 + i * q + v;
                            coefficients[k] = arithmetic.add(
                                coefficients[k],
                                coefficient(sign * bv.get_element(c + v, c + j))?,
                            )?;
                        }
                    }
                }
            }
            if !eliminate::<S, A>(arithmetic, &mut coefficients, &mut rhs, p * q, singularity)? {
                return Ok(false);
            }
            for i in 0..p {
                for j in 0..q {
                    x[(r + i) * n + c + j] = rhs[i * q + j];
                }
            }
            row = if left_adjoint { r + p } else { r };
        }
        if discrete {
            for i in 0..m {
                for j in c..c + q {
                    let mut sum = A::ZERO;
                    for k in 0..m {
                        sum = arithmetic.add(
                            sum,
                            arithmetic.mul(coefficient(av.get_element(i, k))?, x[k * n + j])?,
                        )?;
                    }
                    cached[i * n + j] = sum;
                }
            }
        }
        col = if right_adjoint { c } else { c + q };
    }
    Ok(true)
}

pub(crate) fn solve<S: NumericScalar>(
    t: &impl crate::workspace::storage::MatrixStorage<S>,
    m: usize,
    x: &mut [Extended<S>],
    adjoint: bool,
) -> Result<bool, SciOxideError> {
    pair(
        t.try_view()?.block(0, 0, m, m)?,
        t.try_view()?.block(m, m, t.rows() - m, t.rows() - m)?,
        -S::ONE,
        x,
        adjoint,
        || Ok(()),
    )
}
pub(crate) fn pair<S: NumericScalar>(
    a: MatrixView<'_, S>,
    b: MatrixView<'_, S>,
    sign: S,
    x: &mut [Extended<S>],
    adjoint: bool,
    admit: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<bool, SciOxideError> {
    pair_oriented(
        Recurrence {
            a,
            b,
            sign,
            left_adjoint: adjoint,
            right_adjoint: adjoint,
            discrete: false,
        },
        x,
        &mut [],
        Singularity::Reject,
        admit,
    )
}

/// Adapter for eigen and matrix-function consumers with complex extended storage.
pub(crate) fn pair_oriented<S: NumericScalar>(
    recurrence: Recurrence<'_, S>,
    x: &mut [Extended<S>],
    values: &mut [S],
    singularity: Singularity,
    mut admit: impl FnMut() -> Result<(), SciOxideError>,
) -> Result<bool, SciOxideError> {
    let n = recurrence.a.rows() * recurrence.b.rows();
    let ordinary = Ordinary::<S::Real>::new()?;
    if !values.is_empty() {
        let mut valid = true;
        for (value, &source) in values[..n].iter_mut().zip(x.iter()) {
            match scalar(source) {
                Ok(v) => {
                    *value = v;
                    valid &= ordinary.check(v).is_ok();
                }
                Err(_) => valid = false,
            }
        }
        if valid {
            match substitute::<S, Ordinary<S::Real>>(
                &ordinary,
                recurrence,
                Substitution {
                    x: &mut values[..n],
                    cached: &mut [],
                    coefficient_shift: 0,
                    singularity,
                },
                &mut admit,
            ) {
                Ok(result) => {
                    for (out, &value) in x.iter_mut().zip(values.iter()) {
                        *out = extended(value)?;
                    }
                    return Ok(result);
                }
                Err(e) if e.kind() == scioxide_core::SciOxideErrorKind::NumericalInstability => (),
                Err(e) => return Err(e),
            }
        }
    }
    substitute::<S, ExtendedArithmetic<Complex<S::Real>>>(
        &ExtendedArithmetic(core::marker::PhantomData),
        recurrence,
        Substitution {
            x,
            cached: &mut [],
            coefficient_shift: 0,
            singularity,
        },
        &mut admit,
    )
}

/// Solve one frame; retry only this leaf in scalar-domain extended arithmetic.
pub(crate) fn framed<S: NumericScalar>(
    recurrence: Recurrence<'_, S>,
    tile: &mut Tile<'_, S>,
    values: &mut [S],
    cache: &mut [S],
    wide_cache: &mut [ScaledValue<S>],
    admit: &mut impl FnMut() -> Result<(), SciOxideError>,
) -> Result<bool, SciOxideError> {
    let ordinary = Ordinary::<S::Real>::new()?;
    let count = tile.rows * tile.cols;
    let (frame, recovery) = tile.input()?.pack(values)?;
    let mut maximum = S::Real::ZERO;
    for v in [recurrence.a, recurrence.b] {
        for i in 0..v.rows() {
            for j in 0..v.cols() {
                maximum = maximum.max(standard::magnitude(v.get_element(i, j)));
            }
        }
    }
    let bound = super::frames::range_frame_bound::<S::Real>()?;
    let e = standard::exponent(maximum)?;
    let shift = if !recurrence.discrete && e.abs() > bound {
        -e
    } else {
        0
    };
    if !recovery {
        match substitute::<S, Ordinary<S::Real>>(
            &ordinary,
            recurrence,
            Substitution {
                x: &mut values[..count],
                cached: cache,
                coefficient_shift: shift,
                singularity: Singularity::Reject,
            },
            admit,
        ) {
            Ok(true) => {
                tile.write_plain(values, exponent_sum(frame, i64::from(shift))?);
                return Ok(true);
            }
            Ok(false) => (),
            Err(e) if e.kind() == scioxide_core::SciOxideErrorKind::NumericalInstability => (),
            Err(e) => return Err(e),
        }
    }
    tile.promote()?;
    // Leaf storage is contiguous, unlike whole-matrix tile views.
    debug_assert_eq!(tile.stride, tile.cols);
    let result = substitute::<S, ExtendedArithmetic<S>>(
        &ExtendedArithmetic(core::marker::PhantomData),
        recurrence,
        Substitution {
            x: &mut tile.wide[..count],
            cached: wide_cache,
            coefficient_shift: 0,
            singularity: Singularity::Reject,
        },
        admit,
    )?;
    let (frame, recovery) = tile.input()?.pack(values)?;
    if !recovery {
        tile.write_plain(values, frame);
    } else {
        *tile.scale = WIDE;
    }
    Ok(result)
}

pub(crate) fn block<S: NumericScalar>(
    a: crate::MatrixView<'_, S>,
    cursor: usize,
    reverse: bool,
) -> (usize, usize) {
    if reverse {
        let pair = S::IS_REAL && cursor >= 2 && a.get_element(cursor - 1, cursor - 2) != S::ZERO;
        let size = if pair { 2 } else { 1 };
        (cursor - size, size)
    } else {
        let pair =
            S::IS_REAL && cursor + 1 < a.rows() && a.get_element(cursor + 1, cursor) != S::ZERO;
        (cursor, if pair { 2 } else { 1 })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::Matrix;
    use scioxide_core::AlgebraicScalar;
    use scioxide_core::Complex;

    fn duality<S: NumericScalar>() {
        let c = |r, i| {
            S::from_components(
                S::Real::try_from_f64_lossy(r).unwrap(),
                S::Real::try_from_f64_lossy(i).unwrap(),
            )
        };
        let mut data = [c(0., 0.); 16];
        for i in 0..4 {
            data[i * 4 + i] = c(if i < 2 { 1. } else { 5. }, (i + 1) as f64);
        }
        data[1] = c(2., -1.);
        data[2] = c(1., 1.);
        data[3] = c(2., -2.);
        data[6] = c(3., 2.);
        data[7] = c(4., 1.);
        data[11] = c(3., 1.);
        if S::IS_REAL {
            data[4] = c(-1., 0.);
            data[14] = c(-2., 0.);
        }
        let t = Matrix::from_vec(4, 4, data.to_vec()).unwrap();
        let x = [c(1., 2.), c(3., -1.), c(-2., 4.), c(2., 1.)];
        let y = [c(2., -1.), c(-3., 2.), c(1., 3.), c(4., -2.)];
        let mut forward = x.map(|v| extended(v).unwrap());
        let mut adjoint = y.map(|v| extended(v).unwrap());
        assert!(solve(&t, 2, &mut forward, false).unwrap());
        assert!(solve(&t, 2, &mut adjoint, true).unwrap());
        let lhs = (0..4).fold(S::ZERO, |sum, i| {
            sum + scalar::<S>(forward[i]).unwrap().conj() * y[i]
        });
        let rhs = (0..4).fold(S::ZERO, |sum, i| {
            sum + x[i].conj() * scalar::<S>(adjoint[i]).unwrap()
        });
        let tolerance =
            S::Real::try_from_usize_exact(256).unwrap() * <S::Real as RealScalar>::EPSILON;
        assert!((lhs - rhs).abs() < tolerance * lhs.abs().max(S::Real::ONE));
        let a = t.view().block(0, 0, 2, 2).unwrap();
        let b = t.view().block(2, 2, 2, 2).unwrap();
        for left in [false, true] {
            for right in [false, true] {
                let av = if left { a.adjoint() } else { a };
                let bv = if right { b.adjoint() } else { b };
                for sign in [S::ONE, -S::ONE] {
                    for ordinary in [false, true] {
                        let mut rhs = core::array::from_fn::<_, 4, _>(|index| {
                            let (i, j) = (index / 2, index % 2);
                            extended((0..2).fold(S::ZERO, |sum, k| {
                                sum + av.get_element(i, k) * x[k * 2 + j]
                                    + sign * x[i * 2 + k] * bv.get_element(k, j)
                            }))
                            .unwrap()
                        });
                        let mut values = [S::ZERO; 4];
                        let scratch = if ordinary { &mut values[..] } else { &mut [] };
                        assert!(
                            pair_oriented(
                                Recurrence {
                                    a,
                                    b,
                                    sign,
                                    left_adjoint: left,
                                    right_adjoint: right,
                                    discrete: false
                                },
                                &mut rhs,
                                scratch,
                                Singularity::Reject,
                                || Ok(())
                            )
                            .unwrap()
                        );
                        for (actual, expected) in rhs.into_iter().zip(x) {
                            assert!(
                                (scalar::<S>(actual).unwrap() - expected).abs()
                                    <= tolerance * expected.abs().max(S::Real::ONE)
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn real32() {
        duality::<f32>();
    }
    #[test]
    fn real64() {
        duality::<f64>();
    }
    #[test]
    fn complex32() {
        duality::<Complex<f32>>();
    }
    #[test]
    fn complex64() {
        duality::<Complex<f64>>();
    }
}
