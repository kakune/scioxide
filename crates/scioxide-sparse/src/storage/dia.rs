use super::*;

/// Stores a sparse matrix in diagonal (DIA) format.
///
/// The matrix keeps a strictly increasing list of diagonal offsets
/// `d = column − row`, and for each offset `cols` values aligned by column: the
/// entry `a[r, c]` on diagonal `k` is `values()[k · cols + c]`. Positions of a
/// diagonal that fall outside the matrix are padding and hold zero. Every value
/// is finite. The format suits banded matrices; it implements
/// [`LinearOperator`] and [`AdjointOperator`] with
/// [`Accuracy::Standard`](scioxide_core::Accuracy::Standard).
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{CsrMatrix, DiaMatrix, ExplicitZeroPolicy};
///
/// // [[1, 2, 0],
/// //  [0, 3, 4]]
/// let values = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let a = CsrMatrix::from_raw(2, 3, vec![0, 2, 4], vec![0, 1, 1, 2], values)?;
/// let d = DiaMatrix::from_sparse(a.view())?;
/// assert_eq!(d.offsets(), &[0, 1]);
/// assert_eq!(d.values(), &[1.0, 3.0, 0.0, 0.0, 2.0, 4.0]);
/// assert_eq!(d.to_csr(ExplicitZeroPolicy::Drop)?, a);
/// // Slot 0 of diagonal 1 lies outside the matrix and must be zero.
/// let padded = DiaMatrix::from_raw(2, 3, vec![1], vec![9.0_f64, 2.0, 4.0]);
/// assert_eq!(padded.unwrap_err().kind(), SciOxideErrorKind::InvalidInput);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DiaMatrix<S: NumericScalar> {
    rows: usize,
    cols: usize,
    offsets: Vec<isize>,
    values: Vec<S>,
}
// Signed diagonal offset `col - row`.
fn offset(row: usize, col: usize) -> Result<isize, SciOxideError> {
    if col >= row {
        isize::try_from(col - row)
            .map_err(|_| SciOxideError::size_overflow(scioxide_core::quantity!("dia_offset")))
    } else {
        if row - col == isize::MIN.unsigned_abs() {
            return Ok(isize::MIN);
        }
        isize::try_from(row - col)
            .map(|v| -v)
            .map_err(|_| SciOxideError::size_overflow(scioxide_core::quantity!("dia_offset")))
    }
}
pub(super) fn diagonal_row(rows: usize, col: usize, offset: isize) -> Option<usize> {
    let r = if offset >= 0 {
        col.checked_sub(offset as usize)
    } else {
        col.checked_add(offset.unsigned_abs())
    };
    r.filter(|&r| r < rows)
}
impl<S: NumericScalar> DiaMatrix<S> {
    /// Creates a `rows × cols` matrix from diagonal offsets and column-aligned
    /// values.
    ///
    /// `offsets` must be strictly increasing, and `values` holds `cols` values
    /// for each offset, with zero in the padding positions.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `values.len()` is not `offsets.len() · cols`.
    /// - `InvalidInput` if the offsets are not strictly increasing, a diagonal
    ///   lies entirely outside the matrix, or a padding position is nonzero.
    /// - `NonFinite` if a value has a NaN or infinite component.
    /// - `SizeOverflow` if `offsets.len() · cols` overflows `usize`.
    pub fn from_raw(
        rows: usize,
        cols: usize,
        offsets: Vec<isize>,
        values: Vec<S>,
    ) -> Result<Self, SciOxideError> {
        crate::helpers::exact(
            values.len(),
            checked_mul(offsets.len(), cols, scioxide_core::quantity!("dia_values"))?,
        )?;

        if offsets.windows(2).any(|p| p[0] >= p[1]) {
            return Err(invalid(
                scioxide_core::argument!("offsets"),
                "malformed DIA arrays",
            ));
        }
        finite(&values)?;
        for (d, &offset) in offsets.iter().enumerate() {
            if rows == 0
                || cols == 0
                || (offset >= 0 && offset as usize >= cols)
                || (offset < 0 && offset.unsigned_abs() >= rows)
            {
                return Err(invalid(
                    scioxide_core::argument!("offsets"),
                    "DIA offset does not intersect shape",
                ));
            }
            for c in 0..cols {
                if diagonal_row(rows, c, offset).is_none() && values[d * cols + c] != S::ZERO {
                    return Err(invalid(
                        scioxide_core::argument!("values"),
                        "nonzero DIA padding",
                    ));
                }
            }
        }
        Ok(Self {
            rows,
            cols,
            offsets,
            values,
        })
    }
    /// Returns a borrowed [`SparseView`] of the matrix.
    pub fn view(&self) -> SparseView<'_, S> {
        SparseView {
            rows: self.rows,
            cols: self.cols,
            values: &self.values,
            layout: Layout::Dia {
                offsets: &self.offsets,
            },
        }
    }
    /// Returns the diagonal offsets `column − row`, in increasing order.
    pub fn offsets(&self) -> &[isize] {
        &self.offsets
    }
    /// Returns the storage that [`from_sparse`](Self::from_sparse) would need
    /// for `source`.
    ///
    /// The call collects the occupied diagonals, which takes a temporary list
    /// of one offset per stored entry, but it does not allocate the values.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if an offset does not fit in `isize` or the value
    ///   count overflows `usize`.
    /// - `AllocationFailure` if the offset list cannot be allocated.
    pub fn conversion_requirements(
        source: SparseView<'_, S>,
    ) -> Result<SparseStorageRequirements, SciOxideError> {
        let offsets = diagonal_keys(source)?;
        Ok(SparseStorageRequirements {
            pointers: 0,
            traversal_pointers: 0,
            traversal_indices: 0,
            indices: offsets.len(),
            values: checked_mul(
                offsets.len(),
                source.cols,
                scioxide_core::quantity!("dia_values"),
            )?,
        })
    }
    /// Converts a matrix in any format to DIA.
    ///
    /// Every diagonal that contains a stored entry is kept, and its positions
    /// inside the matrix that the source does not store become explicit zeros.
    /// The storage is `cols` values per occupied diagonal, so DIA is compact
    /// only when few diagonals are occupied.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if an offset does not fit in `isize` or the value
    ///   count overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn from_sparse(source: SparseView<'_, S>) -> Result<Self, SciOxideError> {
        let offsets = diagonal_keys(source)?;
        let mut values = try_filled(
            checked_mul(
                offsets.len(),
                source.cols,
                scioxide_core::quantity!("dia_values"),
            )?,
            S::ZERO,
            scioxide_core::quantity!("dia_values"),
        )?;
        for (r, c, v) in source.iter() {
            // Every occupied offset is a key, so its insertion point is its position.
            let key = offset(r, c)?;
            let d = offsets.partition_point(|&o| o < key);
            values[d * source.cols + c] = v;
        }
        // Offsets are sorted, unique and occupied; padding was never written.
        Ok(Self {
            rows: source.rows,
            cols: source.cols,
            offsets,
            values,
        })
    }
}
// Sorted occupied diagonal offsets.
fn diagonal_keys<S: NumericScalar>(source: SparseView<'_, S>) -> Result<Vec<isize>, SciOxideError> {
    let mut offsets =
        try_with_capacity(source.values.len(), scioxide_core::quantity!("dia_offsets"))?;
    for (r, c, _) in source.iter() {
        offsets.push(offset(r, c)?);
    }
    offsets.sort_unstable();
    offsets.dedup();
    Ok(offsets)
}

impl<S: NumericScalar> DiaMatrix<S> {
    /// Returns the number of rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }
    /// Returns the number of columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }
    /// Returns the column-aligned values, `cols` per diagonal, including
    /// padding.
    pub fn values(&self) -> &[S] {
        &self.values
    }
    /// Converts to a [`CsrMatrix`].
    ///
    /// Padding is never copied. With [`ExplicitZeroPolicy::Preserve`] the
    /// zeros inside the matrix stay as entries; with
    /// [`ExplicitZeroPolicy::Drop`] they are left out. The conversion does not
    /// sort.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn to_csr(&self, zeros: ExplicitZeroPolicy) -> Result<CsrMatrix<S>, SciOxideError> {
        self.view().to_csr(zeros)
    }
    /// Returns the transpose `Aᵀ` in DIA format.
    ///
    /// Offset `d` becomes `−d`, and each diagonal of `Aᵀ` holds `rows` values.
    /// The cost is `O(k · (rows + cols))` for `k` diagonals, without sorting.
    ///
    /// ```
    /// use scioxide_sparse::{DiaMatrix, ExplicitZeroPolicy};
    ///
    /// // [[1, 2, 0],
    /// //  [0, 3, 4]]
    /// let a = DiaMatrix::from_raw(2, 3, vec![0, 1], vec![1.0_f64, 3.0, 0.0, 0.0, 2.0, 4.0])?;
    /// let t = a.transpose()?;
    /// assert_eq!(((t.rows(), t.cols()), t.offsets()), ((3, 2), &[-1, 0][..]));
    /// assert_eq!(t.values(), &[2.0, 4.0, 1.0, 3.0]);
    /// assert_eq!(t.to_csr(ExplicitZeroPolicy::Drop)?.get(2, 1)?, 4.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if an offset cannot be negated in `isize` or the value
    ///   count overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn transpose(&self) -> Result<Self, SciOxideError> {
        self.transformed(false)
    }
    /// Returns the conjugate transpose `Aᴴ` in DIA format, at the cost of
    /// [`transpose`](Self::transpose).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if an offset cannot be negated in `isize` or the value
    ///   count overflows `usize`.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn adjoint(&self) -> Result<Self, SciOxideError> {
        self.transformed(true)
    }
    fn transformed(&self, conjugate: bool) -> Result<Self, SciOxideError> {
        let quantity = scioxide_core::quantity!("dia_transpose");
        let mut offsets = try_with_capacity(self.offsets.len(), quantity)?;
        for &offset in self.offsets.iter().rev() {
            offsets.push(offset.checked_neg().ok_or(SciOxideError::size_overflow(
                scioxide_core::quantity!("dia_offset"),
            ))?);
        }
        let mut values = try_filled(
            checked_mul(self.offsets.len(), self.rows, quantity)?,
            S::ZERO,
            quantity,
        )?;
        for (d, &offset) in self.offsets.iter().enumerate() {
            let target = (self.offsets.len() - 1 - d) * self.rows;
            let source = &self.values[d * self.cols..(d + 1) * self.cols];
            for (c, &v) in source.iter().enumerate() {
                if let Some(r) = diagonal_row(self.rows, c, offset) {
                    values[target + r] = if conjugate { v.conj() } else { v };
                }
            }
        }
        Ok(Self {
            rows: self.cols,
            cols: self.rows,
            offsets,
            values,
        })
    }
}
impl<S: NumericScalar> LinearOperator<S> for DiaMatrix<S> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.rows)
    }
    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.view().apply_into(input, output, scratch)
    }
}
impl<S: NumericScalar> AdjointOperator<S> for DiaMatrix<S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.cols)
    }
    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.view().apply_adjoint_into(input, output, scratch)
    }
}
impl<S: NumericScalar> BlockOperator<S> for DiaMatrix<S> {}
