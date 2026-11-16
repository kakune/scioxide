//! Prepared vector substitution for `Standard` accuracy.

use core::ops::Range;
use scioxide_core::backend::KernelDispatch;
use scioxide_core::{AlgebraicScalar, NumericScalar, RealScalar, SciOxideError};
use scioxide_core::{DiagonalKind, Triangle};

use super::{GemmOp, standard};
use crate::{MatrixView, PackedMatrixView};

/// Immutable coefficient storage. Layout is resolved once per substituted row.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Storage<'a, S: NumericScalar> {
    Dense(MatrixView<'a, S>),
    /// Physical column-major band, including LU fill rows. The diagonal
    /// offset is independent of the logical bandwidth and matrix order.
    Band {
        data: &'a [S],
        stride: usize,
        diagonal: usize,
    },
    Packed(PackedMatrixView<'a, S>),
}

/// A row has constant stride except in column-packed storage, where successive
/// column lengths change the stride by one. No coefficient accessor is called
/// by the arithmetic loop.
#[derive(Clone, Copy)]
struct Row<'a, S> {
    data: &'a [S],
    position: usize,
    stride: usize,
    stride_change: isize,
    remaining: usize,
}

impl<S: Copy> Iterator for Row<'_, S> {
    type Item = S;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }

    #[inline]
    fn next(&mut self) -> Option<S> {
        if self.remaining == 0 {
            return None;
        }
        let value = self.data[self.position];
        self.remaining -= 1;
        if self.remaining != 0 {
            self.position += self.stride;
            self.stride = self.stride.wrapping_add_signed(self.stride_change);
        }
        Some(value)
    }
}

impl<S: Copy> ExactSizeIterator for Row<'_, S> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<'a, S: NumericScalar> Storage<'a, S> {
    fn row(self, index: usize, known: Range<usize>, transpose: bool) -> Row<'a, S> {
        // Empty rows never index storage (in particular, empty packed triangles).
        let remaining = known.len();
        let (i, j) = if transpose {
            (known.start, index)
        } else {
            (index, known.start)
        };
        let (data, position, stride, stride_change) = match self {
            Self::Dense(a) => (
                a.data,
                i * a.row_stride + j * a.col_stride,
                if transpose {
                    a.row_stride
                } else {
                    a.col_stride
                },
                0,
            ),
            Self::Band {
                data,
                stride,
                diagonal,
            } => (
                data,
                if remaining == 0 {
                    0
                } else {
                    j * stride + diagonal + i - j
                },
                if transpose {
                    1
                } else {
                    stride.saturating_sub(1)
                },
                0,
            ),
            Self::Packed(a) => {
                let n = a.order();
                let position = if remaining == 0 {
                    0
                } else {
                    match a.triangle() {
                        Triangle::Upper => j * (j + 1) / 2 + i,
                        Triangle::Lower => a.as_slice().len() - (n - j) * (n - j + 1) / 2 + i - j,
                    }
                };
                let (stride, change) = if transpose {
                    (1, 0)
                } else {
                    match a.triangle() {
                        Triangle::Upper => (j + 1, 1),
                        Triangle::Lower => (n.saturating_sub(j + 1), -1),
                    }
                };
                (a.as_slice(), position, stride, change)
            }
        };
        Row {
            data,
            position,
            stride,
            stride_change,
            remaining,
        }
    }

    /// Measured at preparation, never on the repeated solve path. Unit slots
    /// and the unselected triangle are not read.
    pub(crate) fn maximum(
        self,
        n: usize,
        triangle: Triangle,
        diagonal: DiagonalKind,
        bandwidth: usize,
    ) -> S::Real {
        let mut maximum = if diagonal == DiagonalKind::Unit {
            S::Real::ONE
        } else {
            S::Real::ZERO
        };
        for i in 0..n {
            let range = match triangle {
                Triangle::Lower => {
                    i.saturating_sub(bandwidth)..i + usize::from(diagonal == DiagonalKind::Explicit)
                }
                Triangle::Upper => {
                    i + usize::from(diagonal == DiagonalKind::Unit)
                        ..i.saturating_add(bandwidth).saturating_add(1).min(n)
                }
            };
            for value in self.row(i, range, false) {
                maximum = maximum.max(standard::magnitude(value));
            }
        }
        maximum
    }
}

/// The caller has validated coefficients and dimensions and owns the RHS
/// staging. Pivots are checked at their point of use, without a second scan.
pub(crate) fn solve<S: NumericScalar>(
    operand: crate::kernel::TriangularOperand<Storage<'_, S>>,
    maximum: S::Real,
    values: &mut [S],
    bandwidth: usize,
    dispatch: KernelDispatch,
) -> Result<(), SciOxideError> {
    let crate::kernel::TriangularOperand {
        view,
        triangle,
        op,
        diagonal,
    } = operand;

    let conjugated = matches!(view, Storage::Dense(a) if a.conjugated) ^ (op == GemmOp::Adjoint);
    if let Storage::Dense(a) = view
        && a.col_stride == 1
        && !conjugated
        && op == GemmOp::None
        && bandwidth >= values.len().saturating_sub(1)
    {
        return match (triangle, diagonal) {
            (Triangle::Lower, DiagonalKind::Unit) => {
                contiguous::<S, true, true, true>(a, maximum, values, dispatch)
            }
            (Triangle::Lower, DiagonalKind::Explicit) => {
                contiguous::<S, true, false, true>(a, maximum, values, dispatch)
            }
            (Triangle::Upper, DiagonalKind::Unit) => {
                contiguous::<S, false, true, true>(a, maximum, values, dispatch)
            }
            (Triangle::Upper, DiagonalKind::Explicit) => {
                contiguous::<S, false, false, true>(a, maximum, values, dispatch)
            }
        };
    }
    if let Storage::Dense(a) = view
        && a.col_stride == 1
        && op != GemmOp::None
        && bandwidth >= values.len().saturating_sub(1)
    {
        return transpose_dense(
            a,
            operand.triangle,
            operand.diagonal,
            maximum,
            values,
            conjugated,
        );
    }
    if conjugated {
        substitute::<S, true>(
            crate::kernel::TriangularOperand {
                view,
                triangle,
                op,
                diagonal,
            },
            maximum,
            values,
            bandwidth,
            dispatch,
        )
    } else {
        substitute::<S, false>(
            crate::kernel::TriangularOperand {
                view,
                triangle,
                op,
                diagonal,
            },
            maximum,
            values,
            bandwidth,
            dispatch,
        )
    }
}

/// Substitute short packed LU rows without per-row vector-kernel dispatch.
/// The same scalar reduction and range replay serve the general portable path.
pub(crate) fn solve_small_lu<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    maximum: S::Real,
    values: &mut [S],
    dispatch: KernelDispatch,
) -> Result<(), SciOxideError> {
    contiguous::<S, true, true, false>(matrix, maximum, values, dispatch)?;
    contiguous::<S, false, false, false>(matrix, maximum, values, dispatch)
}

/// Resolve dense layout and triangular orientation outside the substitution loop.
/// Row arithmetic and range-safe replay match the general strided kernel.
fn contiguous<S: NumericScalar, const LOWER: bool, const UNIT: bool, const VECTORIZE: bool>(
    matrix: MatrixView<'_, S>,
    maximum: S::Real,
    values: &mut [S],
    dispatch: KernelDispatch,
) -> Result<(), SciOxideError> {
    let n = values.len();
    let mut solved_maximum = S::Real::ZERO;
    for step in 0..n {
        let index = if LOWER { step } else { n - 1 - step };
        let known = if LOWER { 0..index } else { index + 1..n };
        let row = &matrix.data[index * matrix.row_stride..index * matrix.row_stride + n];
        let coefficients = &row[known.clone()];
        let x = &values[known];
        let pivot = if UNIT { S::ONE } else { row[index] };
        if pivot == S::ZERO {
            return Err(SciOxideError::singular(Some(index)));
        }
        let prior = values[index];
        let residual = if x.is_empty() {
            prior
        } else if VECTORIZE
            && S::IS_REAL
            && let Some(sum) = super::arch::dot(coefficients, x, dispatch)?
        {
            prior - sum
        } else {
            let mut sums = [prior, S::ZERO, S::ZERO, S::ZERO];
            let mut a = coefficients.chunks_exact(4);
            let mut b = x.chunks_exact(4);
            for (a, b) in a.by_ref().zip(b.by_ref()) {
                sums[0] = sums[0] - a[0] * b[0];
                sums[1] = sums[1] - a[1] * b[1];
                sums[2] = sums[2] - a[2] * b[2];
                sums[3] = sums[3] - a[3] * b[3];
            }
            for (j, (&a, &b)) in a.remainder().iter().zip(b.remainder()).enumerate() {
                sums[j] = sums[j] - a * b;
            }
            (sums[0] + sums[1]) + (sums[2] + sums[3])
        };
        let value = standard::finish_substitution(
            prior,
            pivot,
            residual,
            coefficients.iter().copied().zip(x.iter().copied()),
            Some((maximum, solved_maximum)),
        )?;
        values[index] = value;
        solved_maximum = solved_maximum.max(standard::magnitude(value));
    }
    Ok(())
}

/// Tile the output coordinates so an adjoint streams physical rows. The
/// original RHS remains available until each row is finished, including the
/// rescaled retry in `finish_substitution`.
fn transpose_dense<S: NumericScalar>(
    a: MatrixView<'_, S>,
    triangle: Triangle,
    diagonal: DiagonalKind,
    maximum: S::Real,
    values: &mut [S],
    conjugated: bool,
) -> Result<(), SciOxideError> {
    let n = values.len();
    let lower = triangle == Triangle::Upper;
    let mut solved_maximum = S::Real::ZERO;
    for tile in 0..n.div_ceil(32) {
        let (start, end) = if lower {
            (tile * 32, ((tile + 1) * 32).min(n))
        } else {
            (n.saturating_sub((tile + 1) * 32), n - tile * 32)
        };
        let mut residuals = [S::ZERO; 32];
        residuals[..end - start].copy_from_slice(&values[start..end]);
        let known = if lower { 0..start } else { end..n };
        for row in known {
            let coefficients = &a.data[row * a.row_stride + start..row * a.row_stride + end];
            let x = values[row];
            for (r, &coefficient) in residuals.iter_mut().zip(coefficients) {
                let coefficient = if conjugated {
                    coefficient.conj()
                } else {
                    coefficient
                };
                *r = *r - coefficient * x;
            }
        }
        for step in 0..end - start {
            let i = if lower { start + step } else { end - 1 - step };
            let local = if lower { start..i } else { i + 1..end };
            let coefficient = |j: usize| {
                let value = a.data[j * a.row_stride + i];
                if conjugated { value.conj() } else { value }
            };
            let mut residual = residuals[i - start];
            for j in local {
                residual = residual - coefficient(j) * values[j];
            }
            let pivot = if diagonal == DiagonalKind::Unit {
                S::ONE
            } else {
                coefficient(i)
            };
            let known = if lower { 0..i } else { i + 1..n };
            let value = standard::finish_substitution(
                values[i],
                pivot,
                residual,
                known.map(|j| (coefficient(j), values[j])),
                Some((maximum, solved_maximum)),
            )?;
            values[i] = value;
            solved_maximum = solved_maximum.max(standard::magnitude(value));
        }
    }
    Ok(())
}

fn substitute<S: NumericScalar, const CONJ: bool>(
    operand: crate::kernel::TriangularOperand<Storage<'_, S>>,
    maximum: S::Real,
    values: &mut [S],
    bandwidth: usize,
    dispatch: KernelDispatch,
) -> Result<(), SciOxideError> {
    let crate::kernel::TriangularOperand {
        view,
        triangle,
        op,
        diagonal,
    } = operand;

    let n = values.len();
    let transpose = op != GemmOp::None;
    let lower = (triangle == Triangle::Lower) != transpose;
    let mut solved_maximum = S::Real::ZERO;
    for step in 0..n {
        let index = if lower { step } else { n - 1 - step };
        let known = if lower {
            index.saturating_sub(bandwidth)..index
        } else {
            index + 1..index.saturating_add(bandwidth).saturating_add(1).min(n)
        };
        let row = view.row(index, known.clone(), transpose);
        let pivot = if diagonal == DiagonalKind::Unit {
            S::ONE
        } else {
            let diagonal_row = view.row(index, index..index + 1, false);
            let pivot = diagonal_row.data[diagonal_row.position];
            if CONJ { pivot.conj() } else { pivot }
        };
        if pivot == S::ZERO {
            return Err(SciOxideError::singular(Some(index)));
        }
        let prior = values[index];
        let x = &values[known];
        let residual = residual::<S, CONJ>(row, x, prior, dispatch)?;
        let value = standard::finish_substitution(
            prior,
            pivot,
            residual,
            row.zip(x)
                .map(|(a, &x)| (if CONJ { a.conj() } else { a }, x)),
            Some((maximum, solved_maximum)),
        )?;
        values[index] = value;
        solved_maximum = solved_maximum.max(standard::magnitude(value));
    }
    Ok(())
}

fn residual<S: NumericScalar, const CONJ: bool>(
    row: Row<'_, S>,
    x: &[S],
    prior: S,
    dispatch: KernelDispatch,
) -> Result<S, SciOxideError> {
    if row.remaining == 0 {
        return Ok(prior);
    }
    if row.stride == 1 && row.stride_change == 0 {
        let a = &row.data[row.position..row.position + row.remaining];
        if S::IS_REAL
            && let Some(sum) = super::arch::dot(a, x, dispatch)?
        {
            return Ok(prior - sum);
        }
    }
    let mut sums = [prior, S::ZERO, S::ZERO, S::ZERO];
    for (index, (a, &b)) in row.zip(x).enumerate() {
        let a = if CONJ { a.conj() } else { a };
        let lane = index % sums.len();
        sums[lane] = sums[lane] - a * b;
    }
    Ok((sums[0] + sums[1]) + (sums[2] + sums[3]))
}
