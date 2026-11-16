use super::*;

/// Stores a sparse matrix in block sparse row (BSR) format with dense
/// `br × bc` blocks.
///
/// The matrix is divided into `br × bc` blocks, so `br` must divide `rows` and
/// `bc` must divide `cols`. Only blocks with at least one stored entry are
/// kept, each as `br · bc` values in row-major order, zeros included. Block row
/// `i` holds the block-column indices `col_indices()[row_ptr()[i]..row_ptr()[i + 1]]`
/// in strictly increasing order. Every value is finite. Like [`CsrMatrix`], the
/// matrix keeps a block-column index for transposed products, and it implements
/// [`LinearOperator`] and [`AdjointOperator`] with
/// [`Accuracy::Standard`](scioxide_core::Accuracy::Standard).
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{BsrMatrix, CsrMatrix};
///
/// // [[1, 0, 0, 3],
/// //  [0, 2, 0, 0]] in 2 x 2 blocks.
/// let a = CsrMatrix::from_raw(2, 4, vec![0, 2, 3], vec![0, 3, 1], vec![1.0_f64, 3.0, 2.0])?;
/// let b = BsrMatrix::from_sparse(a.view(), 2, 2)?;
/// assert_eq!((b.block_shape(), b.block_count(), b.col_indices()), ((2, 2), 2, &[0, 1][..]));
/// assert_eq!(b.values(), &[1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0]);
/// let raw = BsrMatrix::from_raw(2, 4, 2, 2, vec![0, 2], vec![0, 1], b.values().to_vec())?;
/// assert_eq!(raw, b);
/// let err = BsrMatrix::from_sparse(a.view(), 2, 3).unwrap_err();
/// assert_eq!(err.kind(), SciOxideErrorKind::InvalidInput);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct BsrMatrix<S: NumericScalar> {
    rows: usize,
    cols: usize,
    br: usize,
    bc: usize,
    ptr: Vec<usize>,
    col: Vec<usize>,
    values: Vec<S>,
    opposite_ptr: Vec<usize>,
    opposite_order: Vec<usize>,
    block_rows: Vec<usize>,
}
// Checked block area; each block extent is paired
// with the caller's parameter that supplied it.
fn block_shape(
    rows: usize,
    cols: usize,
    (br, br_argument): (usize, Argument),
    (bc, bc_argument): (usize, Argument),
) -> Result<usize, SciOxideError> {
    if br == 0 || bc == 0 || !rows.is_multiple_of(br) || !cols.is_multiple_of(bc) {
        let argument = if br == 0 || !rows.is_multiple_of(br) {
            br_argument
        } else {
            bc_argument
        };
        return Err(invalid(
            argument,
            "BSR requires positive blocks dividing the shape",
        ));
    }
    checked_mul(br, bc, scioxide_core::quantity!("block_area"))
}
impl<S: NumericScalar> BsrMatrix<S> {
    /// Creates a `rows × cols` matrix of `block_rows × block_cols` blocks from
    /// BSR arrays.
    ///
    /// `row_ptr` has `rows / block_rows + 1` entries. Block row `i` holds the
    /// block-column indices `col_indices[row_ptr[i]..row_ptr[i + 1]]` in
    /// strictly increasing order, and block `k` occupies
    /// `values[k · block_rows · block_cols..(k + 1) · block_rows · block_cols]`
    /// in row-major order.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a block dimension is zero or does not divide the
    ///   matrix dimension, `row_ptr` does not run nondecreasing from 0 to
    ///   `col_indices.len()`, a block-column index is out of range, or the
    ///   indices of a block row are not strictly increasing.
    /// - `DimensionMismatch` if `row_ptr` or `values` has the wrong length.
    /// - `NonFinite` if a value has a NaN or infinite component.
    /// - `SizeOverflow` or `AllocationFailure` if the block-column index cannot
    ///   be allocated.
    pub fn from_raw(
        rows: usize,
        cols: usize,
        block_rows: usize,
        block_cols: usize,
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<S>,
    ) -> Result<Self, SciOxideError> {
        let malformed = |argument| Err(invalid(argument, "malformed BSR arrays"));
        let area = block_shape(
            rows,
            cols,
            (block_rows, scioxide_core::argument!("block_rows")),
            (block_cols, scioxide_core::argument!("block_cols")),
        )?;
        crate::helpers::exact(
            row_ptr.len(),
            checked_add(
                rows / block_rows,
                1,
                scioxide_core::quantity!("bsr_row_pointers"),
            )?,
        )?;
        crate::helpers::exact(
            values.len(),
            checked_mul(
                col_indices.len(),
                area,
                scioxide_core::quantity!("bsr_values"),
            )?,
        )?;
        if row_ptr.first() != Some(&0)
            || row_ptr.last() != Some(&col_indices.len())
            || row_ptr.windows(2).any(|p| p[0] > p[1])
            || row_ptr.iter().any(|&p| p > col_indices.len())
        {
            return malformed(scioxide_core::argument!("row_ptr"));
        }
        if col_indices.iter().any(|&c| c >= cols / block_cols) {
            return malformed(scioxide_core::argument!("col_indices"));
        }
        if row_ptr
            .windows(2)
            .any(|p| col_indices[p[0]..p[1]].windows(2).any(|c| c[0] >= c[1]))
        {
            return Err(invalid(
                scioxide_core::argument!("col_indices"),
                "noncanonical BSR columns",
            ));
        }
        finite(&values)?;
        Self::from_accepted(
            (rows, cols),
            (block_rows, block_cols),
            row_ptr,
            col_indices,
            values,
        )
    }
    // Canonical block arrays whose values are already proven finite; derives
    // the cached opposite traversal without revalidating or rescanning values.
    fn from_accepted(
        (rows, cols): (usize, usize),
        (br, bc): (usize, usize),
        ptr: Vec<usize>,
        col: Vec<usize>,
        values: Vec<S>,
    ) -> Result<Self, SciOxideError> {
        let block_rows = crate::compressed::major_indices(rows / br, &ptr, col.len())?;
        let (opposite_ptr, opposite_order) = if values.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            crate::compressed::transpose_index(cols / bc, &col)?
        };
        Ok(Self {
            rows,
            cols,
            br,
            bc,
            ptr,
            col,
            values,
            opposite_ptr,
            opposite_order,
            block_rows,
        })
    }
    /// Returns a borrowed [`SparseView`] of the matrix.
    pub fn view(&self) -> SparseView<'_, S> {
        SparseView {
            rows: self.rows,
            cols: self.cols,
            values: &self.values,
            layout: Layout::Bsr {
                ptr: &self.ptr,
                col: &self.col,
                br: self.br,
                bc: self.bc,
                opposite: (&self.opposite_ptr, &self.opposite_order, &self.block_rows),
            },
        }
    }
    /// Returns the block dimensions `(br, bc)`.
    pub const fn block_shape(&self) -> (usize, usize) {
        (self.br, self.bc)
    }
    /// Returns the `rows / br + 1` block-row pointers.
    pub fn row_ptr(&self) -> &[usize] {
        &self.ptr
    }
    /// Returns the block-column index of each stored block, block row by
    /// block row.
    pub fn col_indices(&self) -> &[usize] {
        &self.col
    }
    /// Returns the number of stored blocks.
    pub fn block_count(&self) -> usize {
        self.col.len()
    }
    /// Returns the storage that [`from_sparse`](Self::from_sparse) would need
    /// for `source` with `br × bc` blocks.
    ///
    /// The call finds the occupied blocks, which takes a temporary list of one
    /// block coordinate per stored entry, but it does not allocate the block
    /// values.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `br` or `bc` is zero or does not divide the matrix
    ///   dimension.
    /// - `SizeOverflow` or `AllocationFailure` if the block list cannot be
    ///   allocated.
    pub fn conversion_requirements(
        source: SparseView<'_, S>,
        br: usize,
        bc: usize,
    ) -> Result<SparseStorageRequirements, SciOxideError> {
        let area = block_shape(
            source.rows,
            source.cols,
            (br, scioxide_core::argument!("br")),
            (bc, scioxide_core::argument!("bc")),
        )?;
        let keys = block_keys(source, br, bc)?;
        Ok(SparseStorageRequirements {
            pointers: checked_add(
                source.rows / br,
                1,
                scioxide_core::quantity!("bsr_row_pointers"),
            )?,
            indices: keys.len(),
            traversal_pointers: if keys.is_empty() {
                0
            } else {
                checked_add(
                    source.cols / bc,
                    1,
                    scioxide_core::quantity!("bsr_column_pointers"),
                )?
            },
            traversal_indices: checked_mul(
                keys.len(),
                2,
                scioxide_core::quantity!("bsr_traversal_indices"),
            )?,
            values: checked_mul(keys.len(), area, scioxide_core::quantity!("bsr_values"))?,
        })
    }
    /// Converts a matrix in any format to BSR with `br × bc` blocks.
    ///
    /// Every block that contains a stored entry is kept, even if all its
    /// values are zero, and positions of such a block that the source does not
    /// store become explicit zeros. Sorting the block coordinates costs
    /// `O(nnz log nnz)`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `br` or `bc` is zero or does not divide the matrix
    ///   dimension.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn from_sparse(
        source: SparseView<'_, S>,
        br: usize,
        bc: usize,
    ) -> Result<Self, SciOxideError> {
        let area = block_shape(
            source.rows,
            source.cols,
            (br, scioxide_core::argument!("br")),
            (bc, scioxide_core::argument!("bc")),
        )?;
        let keys = block_keys(source, br, bc)?;
        let mut values = try_filled(
            checked_mul(keys.len(), area, scioxide_core::quantity!("bsr_values"))?,
            S::ZERO,
            scioxide_core::quantity!("bsr_values"),
        )?;
        let mut col = try_filled(
            keys.len(),
            0usize,
            scioxide_core::quantity!("bsr_column_indices"),
        )?;
        let ptr = crate::helpers::counting_transpose(
            source.rows / br,
            keys.iter().copied(),
            |slot, c| col[slot] = c,
        )?;
        for (r, c, v) in source.iter() {
            // Every occupied block is a key, so its insertion point is its position.
            let position = keys.partition_point(|&key| key < (r / br, c / bc));
            values[position * area + (r % br) * bc + c % bc] = v;
        }
        Self::from_accepted((source.rows, source.cols), (br, bc), ptr, col, values)
    }
}
// Sorted occupied block coordinates.
fn block_keys<S: NumericScalar>(
    source: SparseView<'_, S>,
    br: usize,
    bc: usize,
) -> Result<Vec<(usize, usize)>, SciOxideError> {
    let mut keys = try_with_capacity(
        source.values.len(),
        scioxide_core::quantity!("bsr_block_keys"),
    )?;
    keys.extend(source.iter().map(|(r, c, _)| (r / br, c / bc)));
    keys.sort_unstable();
    keys.dedup();
    Ok(keys)
}

impl<S: NumericScalar> BsrMatrix<S> {
    /// Returns the number of rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }
    /// Returns the number of columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }
    /// Returns the block values, `br · bc` per block in row-major order,
    /// including zeros inside stored blocks.
    pub fn values(&self) -> &[S] {
        &self.values
    }
    /// Converts to a [`CsrMatrix`].
    ///
    /// With [`ExplicitZeroPolicy::Preserve`] every position of every stored
    /// block becomes an entry, zeros included; with
    /// [`ExplicitZeroPolicy::Drop`] exact zeros are left out. The conversion
    /// does not sort.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn to_csr(&self, zeros: ExplicitZeroPolicy) -> Result<CsrMatrix<S>, SciOxideError> {
        self.view().to_csr(zeros)
    }
    /// Returns the transpose `Aᵀ` in BSR format with `bc × br` blocks.
    ///
    /// The block-column index of `A` gives the block rows of `Aᵀ` directly, so
    /// the cost is `O(b · br · bc + rows / br + cols / bc)` for `b` stored
    /// blocks, without sorting. Every stored block is kept, zeros included.
    ///
    /// ```
    /// use scioxide_sparse::{BsrMatrix, ExplicitZeroPolicy};
    ///
    /// // [[1, 2, 0, 0],
    /// //  [0, 0, 3, 4]] in 1 × 2 blocks.
    /// let values = vec![1.0_f64, 2.0, 3.0, 4.0];
    /// let a = BsrMatrix::from_raw(2, 4, 1, 2, vec![0, 1, 2], vec![0, 1], values)?;
    /// let t = a.transpose()?;
    /// assert_eq!(((t.rows(), t.cols()), t.block_shape()), ((4, 2), (2, 1)));
    /// assert_eq!((t.row_ptr(), t.col_indices()), (&[0, 1, 2][..], &[0, 1][..]));
    /// assert_eq!(t.to_csr(ExplicitZeroPolicy::Drop)?.get(3, 1)?, 4.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn transpose(&self) -> Result<Self, SciOxideError> {
        self.transformed(false)
    }
    /// Returns the conjugate transpose `Aᴴ` in BSR format with `bc × br`
    /// blocks, at the cost of [`transpose`](Self::transpose).
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn adjoint(&self) -> Result<Self, SciOxideError> {
        self.transformed(true)
    }
    fn transformed(&self, conjugate: bool) -> Result<Self, SciOxideError> {
        let (br, bc) = (self.br, self.bc);
        let area = br * bc;
        let quantity = scioxide_core::quantity!("bsr_transpose");
        let blocks = self.col.len();
        let mut values = try_with_capacity(self.values.len(), quantity)?;
        let mut col = try_with_capacity(blocks, quantity)?;
        let mut block_rows = try_with_capacity(blocks, quantity)?;
        // Transposed block k is source block opposite_order[k], in block-column order.
        for &block in &self.opposite_order {
            col.push(self.block_rows[block]);
            block_rows.push(self.col[block]);
            let source = &self.values[block * area..(block + 1) * area];
            for c in 0..bc {
                values.extend(
                    source[c..]
                        .iter()
                        .step_by(bc)
                        .map(|&v| if conjugate { v.conj() } else { v }),
                );
            }
        }
        // Source native order is the transposed column-major traversal.
        let mut opposite_order = try_filled(blocks, 0usize, quantity)?;
        for (target, &source) in self.opposite_order.iter().enumerate() {
            opposite_order[source] = target;
        }
        let (ptr, opposite_ptr) = if blocks == 0 {
            let pointers = checked_add(self.cols / bc, 1, quantity)?;
            (try_filled(pointers, 0usize, quantity)?, Vec::new())
        } else {
            (
                try_copy_slice(&self.opposite_ptr, quantity)?,
                try_copy_slice(&self.ptr, quantity)?,
            )
        };
        Ok(Self {
            rows: self.cols,
            cols: self.rows,
            br: bc,
            bc: br,
            ptr,
            col,
            values,
            opposite_ptr,
            opposite_order,
            block_rows,
        })
    }
}
impl<S: NumericScalar> LinearOperator<S> for BsrMatrix<S> {
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
impl<S: NumericScalar> AdjointOperator<S> for BsrMatrix<S> {
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
impl<S: NumericScalar> BlockOperator<S> for BsrMatrix<S> {}
