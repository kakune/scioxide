//! Scalar-domain block frames. Extended entries are touched only on recovery.
use crate::{MatrixView, kernel::standard};
use scioxide_core::{
    AlgebraicScalar, NumericScalar, RealScalar, Scaled, ScaledValue, SciOxideError,
};

// A reserved frame denotes individually scaled entries. That exponent can still
// be represented in those entries; it is never rounded or used as a binary shift.
pub(crate) const WIDE: i64 = i64::MIN;

pub(crate) fn exponent_sum(a: i64, b: i64) -> Result<i64, SciOxideError> {
    a.checked_add(b).filter(|&e| e != WIDE).ok_or_else(|| {
        SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::ExponentOutOfRange,
            "matrix equation block exponent is outside its range",
        )
    })
}

pub(crate) fn coordinate<R: RealScalar>(part: Scaled<R>, exponent: i64) -> R {
    let shift = (i128::from(part.exponent()) - i128::from(exponent))
        .clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32;
    part.significand().scalbn(shift)
}

pub(crate) fn shift<S: NumericScalar>(value: S, exponent: i128) -> S {
    standard::scale(
        value,
        exponent.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32,
    )
}

pub(crate) fn lost<S: NumericScalar>(original: S, packed: S) -> bool {
    (original.real_component() != S::Real::ZERO
        && packed.real_component().abs() < S::Real::MIN_POSITIVE)
        || (!S::IS_REAL
            && original.imaginary_component() != S::Real::ZERO
            && packed.imaginary_component().abs() < S::Real::MIN_POSITIVE)
}

#[derive(Clone, Copy)]
pub(crate) struct Grid<'a> {
    pub(crate) rows: &'a [usize],
    pub(crate) cols: &'a [usize],
}
impl Grid<'_> {
    pub(crate) fn shape(self) -> (usize, usize) {
        (
            self.rows[self.rows.len() - 1],
            self.cols[self.cols.len() - 1],
        )
    }
    pub(crate) fn blocks(self) -> usize {
        (self.rows.len() - 1) * (self.cols.len() - 1)
    }
    fn index(self, row: usize, col: usize) -> usize {
        let r = self.rows.partition_point(|&x| x <= row) - 1;
        let c = self.cols.partition_point(|&x| x <= col) - 1;
        r * (self.cols.len() - 1) + c
    }
}

/// Row-major mantissas, one exponent per Schur-aligned tile, and reserved recovery.
pub(crate) struct Framed<'a, S: NumericScalar> {
    pub(crate) values: &'a mut [S],
    pub(crate) wide: &'a mut [ScaledValue<S>],
    pub(crate) scales: &'a mut [i64],
    pub(crate) grid: Grid<'a>,
}
impl<'a, S: NumericScalar> Framed<'a, S> {
    pub(crate) fn new(
        values: &'a mut [S],
        wide: &'a mut [ScaledValue<S>],
        scales: &'a mut [i64],
        grid: Grid<'a>,
    ) -> Self {
        scales.fill(0);
        Self {
            values,
            wide,
            scales,
            grid,
        }
    }
    pub(crate) fn input(&self) -> Result<Input<'_, S>, SciOxideError> {
        let (rows, cols) = self.grid.shape();
        Ok(if self.scales.iter().all(|&e| e == 0) {
            Input::Plain(MatrixView::from_slice_with_strides(
                rows,
                cols,
                self.values,
                cols,
                1,
            )?)
        } else {
            Input::Framed {
                values: self.values,
                wide: self.wide,
                scales: self.scales,
                grid: self.grid,
                row: 0,
                col: 0,
                rows,
                cols,
            }
        })
    }
    pub(crate) fn tile(&mut self, row: usize, col: usize) -> Tile<'_, S> {
        let stride = self.grid.shape().1;
        let offset = self.grid.rows[row] * stride + self.grid.cols[col];
        Tile {
            values: &mut self.values[offset..],
            wide: &mut self.wide[offset..],
            scale: &mut self.scales[row * (self.grid.cols.len() - 1) + col],
            rows: self.grid.rows[row + 1] - self.grid.rows[row],
            cols: self.grid.cols[col + 1] - self.grid.cols[col],
            stride,
        }
    }
    pub(crate) fn negate(&mut self) {
        for r in 0..self.grid.rows.len() - 1 {
            for c in 0..self.grid.cols.len() - 1 {
                let tile = self.tile(r, c);
                for i in 0..tile.rows {
                    for j in 0..tile.cols {
                        let k = i * tile.stride + j;
                        if *tile.scale == WIDE {
                            tile.wide[k] = -tile.wide[k];
                        } else {
                            tile.values[k] = -tile.values[k];
                        }
                    }
                }
            }
        }
    }
    pub(crate) fn publish(&self, out: &mut [S]) -> Result<(), SciOxideError> {
        if self.scales.iter().all(|&e| e == 0) {
            out.copy_from_slice(self.values);
            return Ok(());
        }
        let input = self.input()?;
        let (_, n) = input.shape();
        for (k, value) in out.iter_mut().enumerate() {
            *value = input.get(k / n, k % n)?.round()?;
        }
        Ok(())
    }
}

pub(crate) struct Tile<'a, S: NumericScalar> {
    pub(crate) values: &'a mut [S],
    pub(crate) wide: &'a mut [ScaledValue<S>],
    pub(crate) scale: &'a mut i64,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) stride: usize,
}
impl<S: NumericScalar> Tile<'_, S> {
    pub(crate) fn input(&self) -> Result<Input<'_, S>, SciOxideError> {
        Ok(if *self.scale == WIDE {
            Input::Wide {
                values: self.wide,
                rows: self.rows,
                cols: self.cols,
                stride: self.stride,
                offset: 0,
            }
        } else {
            Input::Scaled(
                MatrixView::from_slice_with_strides(
                    self.rows,
                    self.cols,
                    self.values,
                    self.stride,
                    1,
                )?,
                *self.scale,
            )
        })
    }
    pub(crate) fn clear(&mut self) {
        *self.scale = 0;
        for i in 0..self.rows {
            self.values[i * self.stride..i * self.stride + self.cols].fill(S::ZERO);
        }
    }
    pub(crate) fn promote(&mut self) -> Result<(), SciOxideError> {
        if *self.scale != WIDE {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let k = i * self.stride + j;
                    self.wide[k] = ScaledValue::from_parts(self.values[k], *self.scale)?;
                }
            }
            *self.scale = WIDE;
        }
        Ok(())
    }
    pub(crate) fn write_plain(&mut self, values: &[S], scale: i64) {
        for i in 0..self.rows {
            self.values[i * self.stride..i * self.stride + self.cols]
                .copy_from_slice(&values[i * self.cols..(i + 1) * self.cols]);
        }
        *self.scale = scale;
    }
    pub(crate) fn copy(
        &mut self,
        input: Input<'_, S>,
        packing: &mut [S],
    ) -> Result<(), SciOxideError> {
        let (exponent, recovery) = input.pack(packing)?;
        if !recovery {
            self.write_plain(packing, exponent);
        } else {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    self.wide[i * self.stride + j] = input.get(i, j)?;
                }
            }
            *self.scale = WIDE;
        }
        Ok(())
    }
    /// Align before addition; restart in individual scales if alignment loses range.
    pub(crate) fn add_plain(
        &mut self,
        values: &[S],
        exponent: i64,
        sign: S,
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        if *self.scale != WIDE {
            let empty = (0..self.rows).all(|i| {
                self.values[i * self.stride..i * self.stride + self.cols]
                    .iter()
                    .all(|&v| v == S::ZERO)
            });
            let common = if empty {
                exponent
            } else {
                exponent.max(*self.scale)
            };
            let mut recovery = false;
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let k = i * self.cols + j;
                    let old = self.values[i * self.stride + j];
                    let a = shift(old, i128::from(*self.scale) - i128::from(common));
                    let b = shift(values[k], i128::from(exponent) - i128::from(common));
                    scratch[k] = a + sign * b;
                    recovery |=
                        !scratch[k].has_finite_components() || lost(old, a) || lost(values[k], b);
                }
            }
            if !recovery {
                self.write_plain(scratch, common);
                return Ok(());
            }
            self.promote()?;
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = ScaledValue::from_parts(sign * values[i * self.cols + j], exponent)?;
                let dest = &mut self.wide[i * self.stride + j];
                *dest = dest.add(value)?;
            }
        }
        Ok(())
    }
    pub(crate) fn add(
        &mut self,
        input: Input<'_, S>,
        sign: S,
        packing: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        let (exponent, recovery) = input.pack(packing)?;
        if !recovery {
            return self.add_plain(packing, exponent, sign, scratch);
        }
        self.promote()?;
        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = input.get(i, j)?;
                let dest = &mut self.wide[i * self.stride + j];
                *dest = if sign == S::ONE {
                    dest.add(value)?
                } else {
                    dest.sub(value)?
                };
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Input<'a, S: NumericScalar> {
    Plain(MatrixView<'a, S>),
    Scaled(MatrixView<'a, S>, i64),
    Wide {
        values: &'a [ScaledValue<S>],
        rows: usize,
        cols: usize,
        stride: usize,
        offset: usize,
    },
    Framed {
        values: &'a [S],
        wide: &'a [ScaledValue<S>],
        scales: &'a [i64],
        grid: Grid<'a>,
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },
}
impl<'a, S: NumericScalar> Input<'a, S> {
    pub(crate) fn shape(self) -> (usize, usize) {
        match self {
            Self::Plain(v) | Self::Scaled(v, _) => (v.rows(), v.cols()),
            Self::Wide { rows, cols, .. } | Self::Framed { rows, cols, .. } => (rows, cols),
        }
    }
    pub(crate) fn block(
        self,
        r: usize,
        c: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, SciOxideError> {
        Ok(match self {
            Self::Plain(v) => Self::Plain(v.block(r, c, rows, cols)?),
            Self::Scaled(v, e) => Self::Scaled(v.block(r, c, rows, cols)?, e),
            Self::Wide {
                values,
                stride,
                offset,
                ..
            } => Self::Wide {
                values,
                rows,
                cols,
                stride,
                offset: offset + r * stride + c,
            },
            Self::Framed {
                values,
                wide,
                scales,
                grid,
                row,
                col,
                ..
            } => {
                let (row, col) = (row + r, col + c);
                let index = grid.index(row, col);
                if rows > 0 && cols > 0 && grid.index(row + rows - 1, col + cols - 1) == index {
                    let stride = grid.shape().1;
                    if scales[index] == WIDE {
                        Self::Wide {
                            values: wide,
                            rows,
                            cols,
                            stride,
                            offset: row * stride + col,
                        }
                    } else {
                        Self::Scaled(
                            MatrixView::from_slice(grid.shape().0, stride, values)?
                                .block(row, col, rows, cols)?,
                            scales[index],
                        )
                    }
                } else {
                    Self::Framed {
                        values,
                        wide,
                        scales,
                        grid,
                        row,
                        col,
                        rows,
                        cols,
                    }
                }
            }
        })
    }
    pub(crate) fn get(self, i: usize, j: usize) -> Result<ScaledValue<S>, SciOxideError> {
        match self {
            Self::Plain(v) => ScaledValue::from_value(v.get_element(i, j)),
            Self::Scaled(v, e) => ScaledValue::from_parts(v.get_element(i, j), e),
            Self::Wide {
                values,
                stride,
                offset,
                ..
            } => Ok(values[offset + i * stride + j]),
            Self::Framed {
                values,
                wide,
                scales,
                grid,
                row,
                col,
                ..
            } => {
                let (r, c) = (row + i, col + j);
                let k = r * grid.shape().1 + c;
                let e = scales[grid.index(r, c)];
                if e == WIDE {
                    Ok(wide[k])
                } else {
                    ScaledValue::from_parts(values[k], e)
                }
            }
        }
    }
    /// Quarter-range packing leaves headroom for products and their cancellation.
    pub(crate) fn pack(self, output: &mut [S]) -> Result<(i64, bool), SciOxideError> {
        let (rows, cols) = self.shape();
        let bound = range_frame_bound::<S::Real>()?;
        let minimum = S::Real::ONE.scalbn(-bound);
        if let Self::Plain(v) | Self::Scaled(v, _) = self {
            let base = if let Self::Scaled(_, e) = self { e } else { 0 };
            let mut maximum = S::Real::ZERO;
            for i in 0..rows {
                for j in 0..cols {
                    maximum = maximum.max(standard::magnitude(v.get_element(i, j)));
                }
            }
            let e = standard::exponent(maximum)?;
            let shift = if e.abs() > bound { e } else { 0 };
            let mut recovery = false;
            for i in 0..rows {
                for j in 0..cols {
                    let original = v.get_element(i, j);
                    let value = standard::scale(original, -shift);
                    output[i * cols + j] = value;
                    recovery |= (original.real_component() != S::Real::ZERO
                        && value.real_component().abs() < minimum)
                        || (!S::IS_REAL
                            && original.imaginary_component() != S::Real::ZERO
                            && value.imaginary_component().abs() < minimum);
                }
            }
            return Ok((exponent_sum(base, i64::from(shift))?, recovery));
        }
        let mut maximum = None;
        for i in 0..rows {
            for j in 0..cols {
                let value = self.get(i, j)?;
                if !value.is_zero() {
                    maximum = Some(maximum.map_or(value.scale_exponent(), |e: i64| {
                        e.max(value.scale_exponent())
                    }));
                }
            }
        }
        let mut exponent = maximum.unwrap_or(0);
        if exponent.unsigned_abs() <= bound as u64 {
            exponent = 0;
        }
        let mut recovery = exponent == WIDE;
        for i in 0..rows {
            for j in 0..cols {
                let value = self.get(i, j)?;
                let re = coordinate(value.real(), exponent);
                let im = coordinate(value.imaginary(), exponent);
                output[i * cols + j] = S::from_components(re, im);
                recovery |= (!value.real().is_zero() && re.abs() < minimum)
                    || (!value.imaginary().is_zero() && im.abs() < minimum);
            }
        }
        Ok((exponent, recovery))
    }
}

/// Quarter-range exponent bound leaves headroom for local products and elimination.
pub(crate) fn range_frame_bound<R: RealScalar>() -> Result<i32, SciOxideError> {
    Ok((standard::exponent(R::MAX)? - 8) / 4)
}

pub(crate) fn extended<S: NumericScalar>(
    value: S,
) -> Result<ScaledValue<scioxide_core::Complex<S::Real>>, SciOxideError> {
    ScaledValue::from_value(scioxide_core::Complex::new(
        value.real_component(),
        value.imaginary_component(),
    ))
}
pub(crate) fn scalar<S: NumericScalar>(
    value: ScaledValue<scioxide_core::Complex<S::Real>>,
) -> Result<S, SciOxideError> {
    let z = value.round()?;
    Ok(S::from_components(z.re, z.im))
}
