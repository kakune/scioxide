use alloc::vec::Vec;

use scioxide_core::backend::{checked_mul, try_copy_slice, try_filled, try_with_capacity};
use scioxide_core::{
    AdjointOperator, BlockOperator, LinearOperator, NumericScalar, SciOxideError,
    StructuralPattern, StructuralPatternView,
};

use crate::build::{CooMatrix, Entry};
use crate::{SparseTraversal, SparseTraversalView, transpose_pattern};

// Builds the opposite traversal.
pub(crate) fn transpose_index(
    minor_dimension: usize,
    indices: &[usize],
) -> Result<(Vec<usize>, Vec<usize>), SciOxideError> {
    let mut order = try_filled(
        indices.len(),
        0usize,
        scioxide_core::quantity!("sparse_transpose_order"),
    )?;
    let transpose_pointers = crate::helpers::counting_transpose(
        minor_dimension,
        indices.iter().enumerate().map(|(pos, &minor)| (minor, pos)),
        |slot, pos| order[slot] = pos,
    )?;
    Ok((transpose_pointers, order))
}

// Expands pointers to per-entry major indices.
pub(crate) fn major_indices(
    major_dimension: usize,
    pointers: &[usize],
    nnz: usize,
) -> Result<Vec<usize>, SciOxideError> {
    let mut major_indices = try_filled(
        nnz,
        0_usize,
        scioxide_core::quantity!("sparse_major_indices"),
    )?;
    for major in 0..major_dimension {
        major_indices[pointers[major]..pointers[major + 1]].fill(major);
    }
    Ok(major_indices)
}

// Validated native orientation, its opposite traversal, and native numerical order.
#[derive(Debug, Clone, PartialEq)]
struct CompressedStorage<S: NumericScalar> {
    native: StructuralPattern,
    opposite: SparseTraversal,
    values: Vec<S>,
}
impl<S: NumericScalar> CompressedStorage<S> {
    fn new(native: StructuralPattern, values: Vec<S>) -> Result<Self, SciOxideError> {
        crate::helpers::exact(native.view().col_indices().len(), values.len())?;
        let opposite = transpose_pattern(native.view())?;
        Ok(Self {
            native,
            opposite,
            values,
        })
    }
    // Input entries are unique and row-major; counting scatter also accepts CSC order.
    fn from_entries(
        rows: usize,
        cols: usize,
        entries: impl ExactSizeIterator<Item = (usize, usize, S)> + Clone,
        column_major: bool,
    ) -> Result<Self, SciOxideError> {
        let mut indices = try_filled(
            entries.len(),
            0usize,
            scioxide_core::quantity!("compressed_indices"),
        )?;
        let mut values = try_filled(
            entries.len(),
            S::ZERO,
            scioxide_core::quantity!("compressed_values"),
        )?;
        let (major, minor) = if column_major {
            (cols, rows)
        } else {
            (rows, cols)
        };
        let pointers = crate::helpers::counting_transpose(
            major,
            entries.map(|(row, col, v)| {
                if column_major {
                    (col, (row, v))
                } else {
                    (row, (col, v))
                }
            }),
            |slot, (i, v)| {
                indices[slot] = i;
                values[slot] = v;
            },
        )?;
        Self::new(
            StructuralPattern::new(major, minor, pointers, indices)?,
            values,
        )
    }
    // Copy native order for a mathematical transpose, optionally conjugating values.
    fn copy_native(&self, conjugate: bool) -> Result<Self, SciOxideError> {
        let mut values = try_copy_slice(
            &self.values,
            scioxide_core::quantity!("sparse_transpose_values"),
        )?;
        if conjugate {
            values.iter_mut().for_each(|v| *v = v.conj());
        }
        Ok(Self {
            native: self.native.view().to_owned()?,
            opposite: self.opposite.try_copy()?,
            values,
        })
    }
    // Exchange cached orientations for a format conversion; invert the value permutation.
    fn copy_opposite(&self) -> Result<Self, SciOxideError> {
        let mut values = try_with_capacity(
            self.values.len(),
            scioxide_core::quantity!("sparse_conversion_values"),
        )?;
        values.extend(self.opposite.positions.iter().map(|&p| self.values[p]));
        let mut positions = try_filled(
            values.len(),
            0usize,
            scioxide_core::quantity!("sparse_inverse_positions"),
        )?;
        for (target, &source) in self.opposite.positions.iter().enumerate() {
            positions[source] = target;
        }
        Ok(Self {
            native: self.opposite.pattern.view().to_owned()?,
            opposite: SparseTraversal {
                pattern: self.native.view().to_owned()?,
                positions,
            },
            values,
        })
    }
}

/// Stores a sparse matrix in compressed sparse row (CSR) format.
///
/// Row `i` holds the column indices `col_indices()[row_ptr()[i]..row_ptr()[i + 1]]`
/// in strictly increasing order, with the matching entries of `values()`. Every
/// value is finite, and explicitly stored zeros belong to the pattern. The
/// pattern is fixed after construction; the values can change through
/// `update_values`. The matrix also keeps a column-wise index of its entries
/// (see [`opposite_traversal`](Self::opposite_traversal)), so transposed
/// products and conversion to CSC do not search the rows. See
/// [Storage](crate#storage) for the other formats.
///
/// `CsrMatrix` implements [`LinearOperator`] and [`AdjointOperator`] with
/// [`Accuracy::Standard`](scioxide_core::Accuracy::Standard); their workspaces
/// need `rows` and `cols` scalars. Use [`view`](Self::view) to select another
/// accuracy.
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::CsrMatrix;
///
/// // [[1, 0, 2],
/// //  [0, 3, 0]]
/// let a = CsrMatrix::from_raw(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0_f64, 2.0, 3.0])?;
/// assert_eq!((a.get(0, 2)?, a.get(1, 0)?), (2.0, 0.0));
/// assert_eq!(a.get(2, 0).unwrap_err().kind(), SciOxideErrorKind::OutOfBounds);
/// assert_eq!(a.iter().collect::<Vec<_>>(), [(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0)]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct CsrMatrix<S: NumericScalar> {
    storage: CompressedStorage<S>,
}

impl<S: NumericScalar> CsrMatrix<S> {
    /// Creates a `rows × cols` matrix from CSR arrays.
    ///
    /// `row_ptr` has `rows + 1` entries, and row `i` holds the column indices
    /// `col_ind[row_ptr[i]..row_ptr[i + 1]]` in strictly increasing order, with
    /// values in the same positions of `values`. The arrays are checked once
    /// and then moved into the matrix. Use [`SparseBuilder`](crate::SparseBuilder)
    /// for unsorted triplets.
    ///
    /// ```
    /// use scioxide_core::SciOxideErrorKind;
    /// use scioxide_sparse::CsrMatrix;
    ///
    /// let unsorted = CsrMatrix::from_raw(1, 3, vec![0, 2], vec![2, 0], vec![1.0_f64, 2.0]);
    /// assert_eq!(unsorted.unwrap_err().kind(), SciOxideErrorKind::InvalidInput);
    /// let out_of_range = CsrMatrix::from_raw(1, 3, vec![0, 1], vec![3], vec![1.0_f64]);
    /// assert_eq!(out_of_range.unwrap_err().kind(), SciOxideErrorKind::InvalidInput);
    /// let non_finite = CsrMatrix::from_raw(1, 3, vec![0, 1], vec![0], vec![f64::NAN]);
    /// assert_eq!(non_finite.unwrap_err().kind(), SciOxideErrorKind::NonFinite);
    /// ```
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `row_ptr` does not have `rows + 1` nondecreasing
    ///   entries from 0 to `col_ind.len()`, a column index is not less than
    ///   `cols`, or the indices of a row are not strictly increasing.
    /// - `DimensionMismatch` if `values` and `col_ind` have different lengths.
    /// - `NonFinite` if a value has a NaN or infinite component.
    /// - `SizeOverflow` or `AllocationFailure` if the column index cannot be
    ///   allocated.
    pub fn from_raw(
        rows: usize,
        cols: usize,
        row_ptr: Vec<usize>,
        col_ind: Vec<usize>,
        values: Vec<S>,
    ) -> Result<Self, SciOxideError> {
        Self::from_pattern(
            StructuralPattern::new(rows, cols, row_ptr, col_ind)?,
            values,
        )
    }

    /// Creates the `rows × cols` matrix `A = Σₖ vₖ e_{iₖ} e_{jₖ}ᵀ` from the
    /// triplets `(iₖ, jₖ, vₖ) = (row_indices[k], col_indices[k], values[k])`,
    /// given in any order.
    ///
    /// Values at the same coordinate are added exactly and rounded once, so the
    /// result does not depend on the order of the triplets. The pattern is the
    /// set of distinct coordinates whatever the values: a sum that cancels stays
    /// as a stored zero. This is [`SparseBuilder`](crate::SparseBuilder) with
    /// [`DuplicatePolicy::Sum`](crate::DuplicatePolicy::Sum) and
    /// [`ExplicitZeroPolicy::Preserve`](crate::ExplicitZeroPolicy::Preserve);
    /// the builder takes other policies. Sorting the `t` triplets costs
    /// `O(t log t)`.
    ///
    /// ```
    /// use scioxide_sparse::CsrMatrix;
    ///
    /// // The two triplets at (1, 2) sum to 2 + 3 = 5.
    /// let a = CsrMatrix::from_triplets(2, 3, &[1, 0, 1], &[2, 0, 2], &[2.0_f64, 1.0, 3.0])?;
    /// assert_eq!(a.iter().collect::<Vec<_>>(), [(0, 0, 1.0), (1, 2, 5.0)]);
    /// let y = a.apply(&[1.0, 2.0, 3.0])?;
    /// assert!((y[0] - 1.0).abs() < 1e-15 && (y[1] - 15.0).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the three slices have different lengths.
    /// - `OutOfBounds` if an index lies outside the matrix.
    /// - `NonFinite` if a value has a NaN or infinite component.
    /// - `NumericalInstability` if a sum of values at one coordinate overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        row_indices: &[usize],
        col_indices: &[usize],
        values: &[S],
    ) -> Result<Self, SciOxideError> {
        let mut builder = crate::SparseBuilder::with_capacity(rows, cols, values.len())?;
        builder.extend(row_indices, col_indices, values)?;
        builder.finish_csr(
            crate::DuplicatePolicy::Sum,
            crate::ExplicitZeroPolicy::Preserve,
        )
    }

    /// Creates the identity matrix of order `order`, with a stored 1 at each
    /// diagonal position and no other entries.
    ///
    /// ```
    /// use scioxide_sparse::CsrMatrix;
    ///
    /// let eye = CsrMatrix::<f64>::identity(3)?;
    /// assert_eq!((eye.nnz(), eye.get(1, 1)?, eye.get(0, 1)?), (3, 1.0, 0.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn identity(order: usize) -> Result<Self, SciOxideError> {
        let pointers = scioxide_core::quantity!("identity_row_pointers");
        let mut row_ptr = try_with_capacity(
            scioxide_core::backend::checked_add(order, 1, pointers)?,
            pointers,
        )?;
        row_ptr.extend(0..=order);
        let mut col_ind = try_with_capacity(order, scioxide_core::quantity!("identity_columns"))?;
        col_ind.extend(0..order);
        let values = try_filled(order, S::ONE, scioxide_core::quantity!("identity_values"))?;
        Self::from_finite(
            StructuralPattern::new(order, order, row_ptr, col_ind)?,
            values,
        )
    }

    /// Creates a matrix from a checked [`StructuralPattern`] and its values in
    /// storage order.
    ///
    /// The pattern is not checked again; only the number of values and their
    /// finiteness are. Explicit zeros stay in the pattern.
    ///
    /// ```
    /// use scioxide_core::{SciOxideErrorKind, StructuralPattern};
    /// use scioxide_sparse::CsrMatrix;
    ///
    /// // [[1, 0, 2],
    /// //  [0, 0, 0]] with a stored zero at (1, 1).
    /// let pattern = StructuralPattern::new(2, 3, vec![0, 2, 3], vec![0, 2, 1])?;
    /// let a = CsrMatrix::from_pattern(pattern.clone(), vec![1.0_f64, 2.0, 0.0])?;
    /// assert_eq!(a.structural_pattern(), pattern.view());
    /// assert_eq!(a.iter().collect::<Vec<_>>(), [(0, 0, 1.0), (0, 2, 2.0), (1, 1, 0.0)]);
    /// let short = CsrMatrix::from_pattern(pattern.clone(), vec![1.0_f64]);
    /// assert_eq!(short.unwrap_err().kind(), SciOxideErrorKind::DimensionMismatch);
    /// let non_finite = CsrMatrix::from_pattern(pattern, vec![1.0_f64, f64::NAN, 0.0]);
    /// assert_eq!(non_finite.unwrap_err().kind(), SciOxideErrorKind::NonFinite);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the number of values differs from the number
    ///   of stored positions in `pattern`.
    /// - `NonFinite` if a value has a NaN or infinite component.
    /// - `SizeOverflow` or `AllocationFailure` if the column index cannot be
    ///   allocated.
    pub fn from_pattern(pattern: StructuralPattern, values: Vec<S>) -> Result<Self, SciOxideError> {
        crate::helpers::exact(values.len(), pattern.view().col_indices().len())?;
        crate::helpers::finite(&values)?;
        Self::from_finite(pattern, values)
    }

    /// Replaces the shape and pattern of the matrix with `pattern` and sets
    /// every value to zero.
    ///
    /// The existing storage is reused, so the call allocates only when the new
    /// pattern has more entries than any earlier one. `scratch` must hold
    /// exactly `cols + 1 + 2·nnz` indices of the new pattern. If storage
    /// cannot grow, the matrix is unchanged. Analyses and factors made for the
    /// old pattern do not apply to the new one.
    ///
    /// ```
    /// use scioxide_core::StructuralPattern;
    /// use scioxide_sparse::CsrMatrix;
    ///
    /// let mut a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64; 2])?;
    /// let p = StructuralPattern::new(1, 1, vec![0, 1], vec![0])?;
    /// // Scratch length: cols + 1 + 2·nnz = 1 + 1 + 2 = 4.
    /// a.try_assign_pattern(p.view(), &mut [0; 4])?;
    /// assert_eq!(a.values(), &[0.0]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `scratch` does not have exactly
    ///   `cols + 1 + 2·nnz` entries.
    /// - `SizeOverflow` if that length overflows `usize`.
    /// - `AllocationFailure` if the storage cannot grow.
    pub fn try_assign_pattern(
        &mut self,
        pattern: StructuralPatternView<'_>,
        scratch: &mut [usize],
    ) -> Result<(), SciOxideError> {
        let q = scioxide_core::quantity!("sparse_pattern_scratch");
        let pointers = scioxide_core::backend::checked_add(pattern.cols(), 1, q)?;
        let z = pattern.col_indices().len();
        let required = scioxide_core::backend::checked_add(pointers, checked_mul(z, 2, q)?, q)?;
        crate::helpers::exact(scratch.len(), required)?;
        let (ptr, rest) = scratch.split_at_mut(pointers);
        let (indices, positions) = rest.split_at_mut(z);
        crate::helpers::transpose_into(pattern, ptr, indices, positions);
        // Reserve every owned buffer before changing accepted structure or values.
        self.storage.native.try_reserve(pattern.rows(), z)?;
        self.storage
            .opposite
            .pattern
            .try_reserve(pattern.cols(), z)?;
        let retained_positions = &mut self.storage.opposite.positions;
        let additional = z.saturating_sub(retained_positions.len());
        scioxide_core::backend::try_reserve(retained_positions, additional, q)?;
        let additional = z.saturating_sub(self.storage.values.len());
        scioxide_core::backend::try_reserve(&mut self.storage.values, additional, q)?;
        self.storage.native.try_assign(pattern)?;
        self.storage
            .opposite
            .pattern
            .try_assign(StructuralPatternView::new(
                pattern.cols(),
                pattern.rows(),
                ptr,
                indices,
            )?)?;
        crate::helpers::resize(&mut self.storage.opposite.positions, z, 0)?;
        self.storage.opposite.positions.copy_from_slice(positions);
        crate::helpers::resize(&mut self.storage.values, z, S::ZERO)?;
        self.storage.values.fill(S::ZERO);
        Ok(())
    }

    // Takes values already proven finite by an accepted source or a checked kernel.
    pub(crate) fn from_finite(
        pattern: StructuralPattern,
        values: Vec<S>,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            storage: CompressedStorage::new(pattern, values)?,
        })
    }

    // Publishes a complete candidate already proven finite by a checked kernel;
    // the caller has matched its length to the accepted pattern.
    pub(crate) fn publish_finite(&mut self, values: &[S]) {
        self.storage.values.copy_from_slice(values);
    }

    pub(crate) fn from_entries(
        rows: usize,
        cols: usize,
        entries: Vec<Entry<S>>,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            storage: CompressedStorage::from_entries(
                rows,
                cols,
                entries.iter().map(|e| (e.row, e.col, e.value)),
                false,
            )?,
        })
    }
    pub(crate) fn from_coo(coo: &CooMatrix<S>) -> Result<Self, SciOxideError> {
        Ok(Self {
            storage: CompressedStorage::from_entries(coo.rows(), coo.cols(), coo.iter(), false)?,
        })
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.storage.native.view().rows()
    }

    /// Returns the number of columns.
    pub fn cols(&self) -> usize {
        self.storage.native.view().cols()
    }

    /// Returns the number of stored entries, including explicit zeros.
    pub fn nnz(&self) -> usize {
        self.storage.values.len()
    }

    /// Returns the sparsity pattern in CSR form, including explicit zeros.
    ///
    /// The pattern does not change when the values change. Symbolic analyses
    /// such as [`SparseLuSymbolic::analyze`](crate::SparseLuSymbolic::analyze)
    /// take it as input.
    pub fn structural_pattern(&self) -> StructuralPatternView<'_> {
        self.storage.native.view()
    }

    /// Returns the `rows + 1` row pointers.
    pub fn row_ptr(&self) -> &[usize] {
        self.storage.native.view().row_ptr()
    }

    /// Returns the column index of each stored entry, row by row.
    pub fn col_indices(&self) -> &[usize] {
        self.storage.native.view().col_indices()
    }

    /// Returns the stored values in storage order.
    pub fn values(&self) -> &[S] {
        &self.storage.values
    }

    /// Returns the entry `a[row, col]`, or zero if it is not stored.
    ///
    /// A binary search in one row or column costs `O(log k)` for `k` stored
    /// entries in it.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `row` or `col` lies outside the matrix.
    pub fn get(&self, row: usize, col: usize) -> Result<S, SciOxideError> {
        if row >= self.rows() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("csr_rows"),
                row,
                self.rows(),
            ));
        }
        if col >= self.cols() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("csr_columns"),
                col,
                self.cols(),
            ));
        }
        let segment = self.row_ptr()[row]..self.row_ptr()[row + 1];
        Ok(self.col_indices()[segment.clone()]
            .binary_search(&col)
            .map_or(S::ZERO, |offset| {
                self.storage.values[segment.start + offset]
            }))
    }

    /// Iterates over the stored `(row, column, value)` triplets in row-major
    /// order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, S)> + '_ {
        (0..self.rows()).flat_map(move |row| {
            (self.row_ptr()[row]..self.row_ptr()[row + 1]).map(move |position| {
                (
                    row,
                    self.col_indices()[position],
                    self.storage.values[position],
                )
            })
        })
    }

    /// Converts to a [`CooMatrix`] with the same entries.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn to_coo(&self) -> Result<CooMatrix<S>, SciOxideError> {
        let mut entries =
            try_with_capacity(self.nnz(), scioxide_core::quantity!("csr_to_coo_entries"))?;
        entries.extend(
            self.iter()
                .map(|(row, col, value)| Entry { row, col, value }),
        );
        CooMatrix::from_entries(self.rows(), self.cols(), entries)
    }

    /// Converts to a [`CscMatrix`] with the same entries.
    ///
    /// The column-wise index kept by the matrix becomes the new storage order,
    /// so the conversion costs `O(nnz + rows + cols)` and does not sort.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the new storage cannot be allocated.
    pub fn to_csc(&self) -> Result<CscMatrix<S>, SciOxideError> {
        Ok(CscMatrix {
            storage: self.storage.copy_opposite()?,
        })
    }

    /// Returns the transpose `Aᵀ`, stored as a [`CscMatrix`].
    ///
    /// The CSR arrays of `A` are the CSC arrays of `Aᵀ`, so this copies them
    /// without reordering.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the new storage cannot be allocated.
    pub fn transpose(&self) -> Result<CscMatrix<S>, SciOxideError> {
        Ok(CscMatrix {
            storage: self.storage.copy_native(false)?,
        })
    }

    /// Returns the conjugate transpose `Aᴴ`, stored as a [`CscMatrix`].
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the new storage cannot be allocated.
    pub fn adjoint(&self) -> Result<CscMatrix<S>, SciOxideError> {
        Ok(CscMatrix {
            storage: self.storage.copy_native(true)?,
        })
    }

    /// Writes the matrix as a dense row-major array of `rows · cols` values to
    /// `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output.len()` is not `rows · cols`.
    /// - `SizeOverflow` if `rows · cols` overflows `usize`.
    pub fn to_dense_into(&self, output: &mut [S]) -> Result<(), SciOxideError> {
        let expected = checked_mul(
            self.rows(),
            self.cols(),
            scioxide_core::quantity!("dense_sparse_conversion"),
        )?;
        crate::helpers::exact(output.len(), expected)?;

        output.fill(S::ZERO);
        self.iter().for_each(|(row, col, value)| {
            output[row * self.cols() + col] = value;
        });
        Ok(())
    }
}

impl<S: NumericScalar> LinearOperator<S> for CsrMatrix<S> {
    fn rows(&self) -> usize {
        self.rows()
    }

    fn cols(&self) -> usize {
        self.cols()
    }

    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.rows())
    }

    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        crate::helpers::minimum(workspace.len(), self.rows())?;
        self.view().apply_checked(
            input,
            output,
            workspace,
            false,
            false,
            scioxide_core::Accuracy::Standard,
        )
    }
}

impl<S: NumericScalar> AdjointOperator<S> for CsrMatrix<S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.cols())
    }

    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        crate::helpers::minimum(workspace.len(), self.cols())?;
        self.view().apply_checked(
            input,
            output,
            workspace,
            true,
            true,
            scioxide_core::Accuracy::Standard,
        )
    }
}

impl<S: NumericScalar> BlockOperator<S> for CsrMatrix<S> {}

/// Stores a sparse matrix in compressed sparse column (CSC) format.
///
/// Column `j` holds the row indices `row_indices()[col_ptr()[j]..col_ptr()[j + 1]]`
/// in strictly increasing order, with the matching entries of `values()`. The
/// other rules are those of [`CsrMatrix`], with rows and columns exchanged: the
/// matrix keeps a row-wise index of its entries, and its
/// [`LinearOperator`] and [`AdjointOperator`] implementations use
/// [`Accuracy::Standard`](scioxide_core::Accuracy::Standard).
///
/// ```
/// use scioxide_sparse::{CscMatrix, CsrMatrix};
///
/// // [[1, 0, 2],
/// //  [0, 3, 0]] stored by columns.
/// let a = CscMatrix::from_raw(2, 3, vec![0, 1, 2, 3], vec![0, 1, 0], vec![1.0_f64, 3.0, 2.0])?;
/// assert_eq!(a.get(0, 2)?, 2.0);
/// assert_eq!(a.iter().collect::<Vec<_>>(), [(0, 0, 1.0), (1, 1, 3.0), (0, 2, 2.0)]);
/// let csr = CsrMatrix::from_raw(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0])?;
/// assert_eq!(a.to_csr()?, csr);
/// // Column pointers need `cols + 1` entries.
/// assert!(CscMatrix::from_raw(2, 3, vec![0, 1], vec![0], vec![1.0_f64]).is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct CscMatrix<S: NumericScalar> {
    storage: CompressedStorage<S>,
}

impl<S: NumericScalar> CscMatrix<S> {
    /// Creates a `rows × cols` matrix from CSC arrays.
    ///
    /// `col_ptr` has `cols + 1` entries, and column `j` holds the row indices
    /// `row_ind[col_ptr[j]..col_ptr[j + 1]]` in strictly increasing order, with
    /// values in the same positions of `values`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `col_ptr` does not have `cols + 1` nondecreasing
    ///   entries from 0 to `row_ind.len()`, a row index is not less than
    ///   `rows`, or the indices of a column are not strictly increasing.
    /// - `DimensionMismatch` if `values` and `row_ind` have different lengths.
    /// - `NonFinite` if a value has a NaN or infinite component.
    /// - `SizeOverflow` or `AllocationFailure` if the row index cannot be
    ///   allocated.
    pub fn from_raw(
        rows: usize,
        cols: usize,
        col_ptr: Vec<usize>,
        row_ind: Vec<usize>,
        values: Vec<S>,
    ) -> Result<Self, SciOxideError> {
        crate::helpers::finite(&values)?;
        let native = StructuralPattern::new(cols, rows, col_ptr, row_ind)?;
        Ok(Self {
            storage: CompressedStorage::new(native, values)?,
        })
    }

    pub(crate) fn from_entries(
        rows: usize,
        cols: usize,
        entries: Vec<Entry<S>>,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            storage: CompressedStorage::from_entries(
                rows,
                cols,
                entries.iter().map(|e| (e.row, e.col, e.value)),
                true,
            )?,
        })
    }
    pub(crate) fn from_coo(coo: &CooMatrix<S>) -> Result<Self, SciOxideError> {
        Ok(Self {
            storage: CompressedStorage::from_entries(coo.rows(), coo.cols(), coo.iter(), true)?,
        })
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.storage.native.view().cols()
    }

    /// Returns the number of columns.
    pub fn cols(&self) -> usize {
        self.storage.native.view().rows()
    }

    /// Returns the number of stored entries, including explicit zeros.
    pub fn nnz(&self) -> usize {
        self.storage.values.len()
    }

    /// Returns the sparsity pattern in CSR form, including explicit zeros.
    ///
    /// The pattern does not change when the values change. Symbolic analyses
    /// such as [`SparseLuSymbolic::analyze`](crate::SparseLuSymbolic::analyze)
    /// take it as input.
    pub fn structural_pattern(&self) -> StructuralPatternView<'_> {
        self.storage.opposite.pattern.view()
    }

    /// Returns the `cols + 1` column pointers.
    pub fn col_ptr(&self) -> &[usize] {
        self.storage.native.view().row_ptr()
    }

    /// Returns the row index of each stored entry, column by column.
    pub fn row_indices(&self) -> &[usize] {
        self.storage.native.view().col_indices()
    }

    /// Returns the stored values in storage order.
    pub fn values(&self) -> &[S] {
        &self.storage.values
    }

    /// Returns the entry `a[row, col]`, or zero if it is not stored.
    ///
    /// A binary search in one row or column costs `O(log k)` for `k` stored
    /// entries in it.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `row` or `col` lies outside the matrix.
    pub fn get(&self, row: usize, col: usize) -> Result<S, SciOxideError> {
        if row >= self.rows() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("csc_rows"),
                row,
                self.rows(),
            ));
        }
        if col >= self.cols() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("csc_columns"),
                col,
                self.cols(),
            ));
        }
        let segment = self.col_ptr()[col]..self.col_ptr()[col + 1];
        Ok(self.row_indices()[segment.clone()]
            .binary_search(&row)
            .map_or(S::ZERO, |offset| {
                self.storage.values[segment.start + offset]
            }))
    }

    /// Iterates over the stored `(row, column, value)` triplets in
    /// column-major order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, S)> + '_ {
        (0..self.cols()).flat_map(move |col| {
            (self.col_ptr()[col]..self.col_ptr()[col + 1]).map(move |position| {
                (
                    self.row_indices()[position],
                    col,
                    self.storage.values[position],
                )
            })
        })
    }

    /// Converts to a [`CooMatrix`] with the same entries in row-major order.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn to_coo(&self) -> Result<CooMatrix<S>, SciOxideError> {
        let mut entries =
            try_with_capacity(self.nnz(), scioxide_core::quantity!("csc_to_coo_entries"))?;
        let traversal = self.opposite_traversal();
        let pattern = traversal.pattern();
        for (row, bounds) in pattern.row_ptr().windows(2).enumerate() {
            entries.extend((bounds[0]..bounds[1]).map(|k| Entry {
                row,
                col: pattern.col_indices()[k],
                value: self.values()[traversal.positions()[k]],
            }));
        }
        CooMatrix::from_entries(self.rows(), self.cols(), entries)
    }

    /// Converts to a [`CsrMatrix`] with the same entries, in
    /// `O(nnz + rows + cols)` without sorting.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the new storage cannot be allocated.
    pub fn to_csr(&self) -> Result<CsrMatrix<S>, SciOxideError> {
        Ok(CsrMatrix {
            storage: self.storage.copy_opposite()?,
        })
    }

    /// Returns the transpose `Aᵀ`, stored as a [`CsrMatrix`].
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the new storage cannot be allocated.
    pub fn transpose(&self) -> Result<CsrMatrix<S>, SciOxideError> {
        self.transposed(false)
    }

    /// Returns the conjugate transpose `Aᴴ`, stored as a [`CsrMatrix`].
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the new storage cannot be allocated.
    pub fn adjoint(&self) -> Result<CsrMatrix<S>, SciOxideError> {
        self.transposed(true)
    }

    fn transposed(&self, conjugate: bool) -> Result<CsrMatrix<S>, SciOxideError> {
        Ok(CsrMatrix {
            storage: self.storage.copy_native(conjugate)?,
        })
    }

    /// Writes the matrix as a dense row-major array of `rows · cols` values to
    /// `output`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output.len()` is not `rows · cols`.
    /// - `SizeOverflow` if `rows · cols` overflows `usize`.
    pub fn to_dense_into(&self, output: &mut [S]) -> Result<(), SciOxideError> {
        let expected = checked_mul(
            self.rows(),
            self.cols(),
            scioxide_core::quantity!("dense_sparse_conversion"),
        )?;
        crate::helpers::exact(output.len(), expected)?;

        output.fill(S::ZERO);
        self.iter().for_each(|(row, col, value)| {
            output[row * self.cols() + col] = value;
        });
        Ok(())
    }
}

impl<S: NumericScalar> LinearOperator<S> for CscMatrix<S> {
    fn rows(&self) -> usize {
        self.rows()
    }

    fn cols(&self) -> usize {
        self.cols()
    }

    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.rows())
    }

    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        crate::helpers::minimum(workspace.len(), self.rows())?;
        self.view().apply_checked(
            input,
            output,
            workspace,
            false,
            false,
            scioxide_core::Accuracy::Standard,
        )
    }
}

impl<S: NumericScalar> AdjointOperator<S> for CscMatrix<S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.cols())
    }

    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        crate::helpers::minimum(workspace.len(), self.cols())?;
        self.view().apply_checked(
            input,
            output,
            workspace,
            true,
            true,
            scioxide_core::Accuracy::Standard,
        )
    }
}

impl<S: NumericScalar> BlockOperator<S> for CscMatrix<S> {}

impl<S: NumericScalar> CsrMatrix<S> {
    /// Returns the column-wise index of the entries: the pattern of `Aᵀ` and,
    /// for each of its entries, the position of the value in
    /// [`values`](Self::values).
    ///
    /// See [`SparseTraversalView`] for an example.
    pub fn opposite_traversal(&self) -> SparseTraversalView<'_> {
        self.storage.opposite.view()
    }
}
impl<S: NumericScalar> CscMatrix<S> {
    /// Returns the row-wise index of the entries: the CSR pattern of `A` and,
    /// for each of its entries, the position of the value in
    /// [`values`](Self::values).
    ///
    /// See [`SparseTraversalView`] for an example.
    pub fn opposite_traversal(&self) -> SparseTraversalView<'_> {
        self.storage.opposite.view()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use scioxide_core::{
        AdjointOperator, BlockOperator, BlockVectorView, BlockVectorViewMut, Complex64,
        LinearOperator,
    };

    use crate::{CsrMatrix, DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};

    fn matrix() -> CsrMatrix<f64> {
        let mut builder = SparseBuilder::new(3, 3);
        [(0, 0, 2.0), (0, 2, 1.0), (1, 1, -1.0), (2, 0, 4.0)]
            .into_iter()
            .try_for_each(|(row, col, value)| builder.push(row, col, value))
            .unwrap();
        builder
            .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
            .unwrap()
    }

    #[test]
    fn csr_csc_round_trip_and_spmv_match_dense() {
        let csr = matrix();
        let csc = csr.to_csc().unwrap();
        assert_eq!(csc.to_csr().unwrap(), csr);
        let mut dense = [9.0; 9];
        csr.to_dense_into(&mut dense).unwrap();
        assert_eq!(dense, [2.0, 0.0, 1.0, 0.0, -1.0, 0.0, 4.0, 0.0, 0.0]);
        let mut csr_output = [0.0; 3];
        let mut csc_output = [0.0; 3];
        let mut workspace = [0.0; 3];
        csr.apply_into(&[1.0, 2.0, 3.0], &mut csr_output, &mut workspace)
            .unwrap();
        csc.apply_into(&[1.0, 2.0, 3.0], &mut csc_output, &mut workspace)
            .unwrap();
        assert_eq!(csr_output, [5.0, -2.0, 4.0]);
        assert_eq!(csc_output, csr_output);
    }

    #[test]
    fn transpose_adjoint_block_and_zero_extents_are_supported() {
        let complex = CsrMatrix::from_raw(
            1,
            2,
            vec![0, 2],
            vec![0, 1],
            vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
        )
        .unwrap();
        let mut transpose = [Complex64::new(0.0, 0.0); 2];
        let mut adjoint = transpose;
        let mut workspace = transpose;
        complex
            .view()
            .apply_transpose_into(&[Complex64::new(2.0, 0.0)], &mut transpose, &mut workspace)
            .unwrap();
        complex
            .apply_adjoint_into(&[Complex64::new(2.0, 0.0)], &mut adjoint, &mut workspace)
            .unwrap();
        assert_eq!(transpose[0], Complex64::new(2.0, 4.0));
        assert_eq!(adjoint[0], Complex64::new(2.0, -4.0));

        let csr = matrix();
        let input_values = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0];
        let mut output_values = [0.0; 6];
        let input = BlockVectorView::new(&input_values, 3, 2).unwrap();
        let mut output = BlockVectorViewMut::new(&mut output_values, 3, 2).unwrap();
        let mut block_workspace = [0.0; 9];
        csr.apply_block_into(input, &mut output, &mut block_workspace)
            .unwrap();
        assert_eq!(output_values, [5.0, -2.0, 4.0, 7.0, -2.0, 12.0]);

        let empty = CsrMatrix::<f64>::from_raw(0, 3, vec![0], vec![], vec![]).unwrap();
        empty
            .apply_into(&[1.0, 2.0, 3.0], &mut [], &mut [])
            .unwrap();
    }

    #[test]
    fn undersized_workspace_leaves_output_unchanged() {
        let csr = matrix();
        let mut output = [7.0; 3];
        assert!(
            csr.apply_into(&[1.0; 3], &mut output, &mut [0.0; 2])
                .is_err()
        );
        assert_eq!(output, [7.0; 3]);
    }
}

crate::values::impl_value_updates!(CsrMatrix, storage.values);

crate::values::impl_value_updates!(CscMatrix, storage.values);
