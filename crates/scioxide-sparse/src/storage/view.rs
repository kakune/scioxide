use super::*;

type Traversal<'a> = (&'a [usize], &'a [usize], &'a [usize]);

#[derive(Clone, Copy, Debug)]
pub(super) enum Layout<'a> {
    Coo {
        row: &'a [usize],
        col: &'a [usize],
        ptr: &'a [usize],
        opposite: Traversal<'a>,
    },
    Csr {
        ptr: &'a [usize],
        col: &'a [usize],
        opposite: SparseTraversalView<'a>,
    },
    Csc {
        ptr: &'a [usize],
        row: &'a [usize],
        opposite: SparseTraversalView<'a>,
    },
    Bsr {
        ptr: &'a [usize],
        col: &'a [usize],
        br: usize,
        bc: usize,
        opposite: Traversal<'a>,
    },
    Dia {
        offsets: &'a [isize],
    },
}

/// Borrows a sparse matrix of any format.
///
/// Every matrix type has a `view` method that returns a `SparseView` without
/// copying. Functions that accept any format, such as
/// [`sparse_dense_into`](crate::sparse_dense_into) and the symbolic analyses,
/// take a view. The view iterates over the stored entries, explicit zeros
/// included but DIA padding excluded, and implements [`LinearOperator`] and
/// [`AdjointOperator`] with [`Accuracy::Standard`]. Use
/// [`with_accuracy`](Self::with_accuracy) to select another accuracy.
///
/// ```
/// use scioxide_sparse::{CsrMatrix, ExplicitZeroPolicy};
///
/// // [[1, 0 (stored)],
/// //  [0, 4]]
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0_f64, 0.0, 4.0])?;
/// let view = a.view();
/// assert_eq!((view.rows(), view.cols()), (2, 2));
/// assert_eq!(view.iter().collect::<Vec<_>>(), [(0, 0, 1.0), (0, 1, 0.0), (1, 1, 4.0)]);
/// assert_eq!(view.to_csr(ExplicitZeroPolicy::Drop)?.values(), &[1.0, 4.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SparseView<'a, S: NumericScalar> {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) values: &'a [S],
    pub(super) layout: Layout<'a>,
}
impl<'a, S: NumericScalar> SparseView<'a, S> {
    /// Borrows the values when the view is CSR with exactly the given pattern.
    pub(crate) fn csr_values_on(
        self,
        pattern: scioxide_core::StructuralPatternView<'_>,
    ) -> Option<&'a [S]> {
        match self.layout {
            Layout::Csr { ptr, col, .. }
                if self.rows == pattern.rows()
                    && self.cols == pattern.cols()
                    && ptr == pattern.row_ptr()
                    && col == pattern.col_indices() =>
            {
                Some(self.values)
            }
            _ => None,
        }
    }
    /// Returns a [`SparseOperator`] that applies `A`, `Aᵀ` and `Aᴴ` with the
    /// given [`Accuracy`].
    ///
    /// ```
    /// use scioxide_core::{Accuracy, LinearOperator};
    /// use scioxide_sparse::CsrMatrix;
    ///
    /// let a = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![2.0_f64])?;
    /// a.view().with_accuracy(Accuracy::Exact).apply_into(&[3.0], &mut [0.0], &mut [0.0])?;
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn with_accuracy(self, accuracy: Accuracy) -> SparseOperator<'a, S> {
        SparseOperator {
            view: self,
            accuracy,
        }
    }
    /// Returns the number of rows.
    pub const fn rows(self) -> usize {
        self.rows
    }
    /// Returns the number of columns.
    pub const fn cols(self) -> usize {
        self.cols
    }
    /// Returns the stored values in storage order, including zeros inside BSR
    /// blocks and DIA padding.
    pub const fn values(self) -> &'a [S] {
        self.values
    }
    /// Iterates over the stored `(row, column, value)` triplets in storage
    /// order, skipping DIA padding.
    pub fn iter(self) -> SparseEntries<'a, S> {
        SparseEntries {
            view: self,
            position: 0,
            end: self.values.len(),
            major: 0,
            column_filter: None,
            mapped: None,
        }
    }
    pub(crate) fn column(self, column: usize) -> SparseEntries<'a, S> {
        let mut entries = self.iter();
        entries.column_filter = Some(column);
        if self.values.is_empty() {
            return entries;
        }
        match self.layout {
            Layout::Csc { ptr, .. } => {
                entries.position = ptr[column];
                entries.end = ptr[column + 1];
                entries.major = column;
            }
            Layout::Csr { opposite, .. } => {
                let ptr = opposite.pattern().row_ptr();
                entries.position = ptr[column];
                entries.end = ptr[column + 1];
            }
            Layout::Coo {
                opposite: (ptr, order, majors),
                ..
            } => {
                entries.position = ptr[column];
                entries.end = ptr[column + 1];
                entries.mapped = Some((order, majors));
            }
            Layout::Dia { offsets } => {
                entries.end = offsets.len();
            }
            Layout::Bsr {
                br,
                bc,
                opposite: (ptr, _, _),
                ..
            } => {
                entries.position = ptr[column / bc] * br;
                entries.end = ptr[column / bc + 1] * br;
            }
        }
        entries
    }
    /// Converts the viewed matrix to a [`CsrMatrix`].
    ///
    /// Each format is read in row-major order without sorting, so the copy
    /// costs `O(rows + cols + s)` for `s` stored values. With
    /// [`ExplicitZeroPolicy::Drop`] stored zeros, including zeros inside BSR
    /// blocks, are left out.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn to_csr(self, zeros: ExplicitZeroPolicy) -> Result<CsrMatrix<S>, SciOxideError> {
        let keep = |value: S| zeros == ExplicitZeroPolicy::Preserve || value != S::ZERO;
        let quantity = scioxide_core::quantity!("csr_conversion");
        let mut row_ptr = try_filled(checked_add(self.rows, 1, quantity)?, 0usize, quantity)?;
        self.for_each_row_major(|row, _, value| {
            if keep(value) {
                row_ptr[row + 1] += 1;
            }
        });
        let mut total = 0;
        for pointer in &mut row_ptr {
            total += *pointer;
            *pointer = total;
        }
        let mut col_indices = try_with_capacity(total, quantity)?;
        let mut values = try_with_capacity(total, quantity)?;
        self.for_each_row_major(|_, col, value| {
            if keep(value) {
                col_indices.push(col);
                values.push(value);
            }
        });
        CsrMatrix::from_finite(
            StructuralPattern::new(self.rows, self.cols, row_ptr, col_indices)?,
            values,
        )
    }
    // Visits structural entries in canonical row-major order; within a row,
    // columns increase. O(rows + stored slots), without allocation or sorting.
    fn for_each_row_major(self, mut visit: impl FnMut(usize, usize, S)) {
        if self.values.is_empty() {
            return;
        }
        match self.layout {
            Layout::Coo { row, col, .. } => row
                .iter()
                .zip(col)
                .zip(self.values)
                .for_each(|((&r, &c), &v)| visit(r, c, v)),
            Layout::Csr { ptr, col, .. } => {
                for (r, bounds) in ptr.windows(2).enumerate() {
                    let range = bounds[0]..bounds[1];
                    col[range.clone()]
                        .iter()
                        .zip(&self.values[range])
                        .for_each(|(&c, &v)| visit(r, c, v));
                }
            }
            Layout::Csc { opposite, .. } => {
                let pattern = opposite.pattern();
                for (r, bounds) in pattern.row_ptr().windows(2).enumerate() {
                    let range = bounds[0]..bounds[1];
                    pattern.col_indices()[range.clone()]
                        .iter()
                        .zip(&opposite.positions()[range])
                        .for_each(|(&c, &p)| visit(r, c, self.values[p]));
                }
            }
            Layout::Bsr {
                ptr, col, br, bc, ..
            } => {
                for (block_row, bounds) in ptr.windows(2).enumerate() {
                    for r in 0..br {
                        let row = block_row * br + r;
                        for (block, &block_col) in
                            col.iter().enumerate().take(bounds[1]).skip(bounds[0])
                        {
                            let start = block * br * bc + r * bc;
                            self.values[start..start + bc]
                                .iter()
                                .enumerate()
                                .for_each(|(c, &v)| visit(row, block_col * bc + c, v));
                        }
                    }
                }
            }
            Layout::Dia { offsets } => {
                // Active diagonals of a row form one contiguous offset range.
                let (mut start, mut end) = (offsets.len(), offsets.len());
                for r in 0..self.rows {
                    while start > 0 && diagonal_column(r, offsets[start - 1]).is_some() {
                        start -= 1;
                    }
                    while end > start
                        && diagonal_column(r, offsets[end - 1]).is_none_or(|c| c >= self.cols)
                    {
                        end -= 1;
                    }
                    for (p, &offset) in offsets.iter().enumerate().take(end).skip(start) {
                        if let Some(c) = diagonal_column(r, offset).filter(|&c| c < self.cols) {
                            visit(r, c, self.values[p * self.cols + c]);
                        }
                    }
                }
            }
        }
    }
    /// Computes `y = Aᵀ x` (transpose without conjugation) into `output`, with
    /// [`Accuracy::Standard`].
    ///
    /// `scratch` needs at least `cols` entries.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` does not have `rows` entries, `output`
    ///   does not have `cols` entries, or `scratch` has fewer than `cols`.
    /// - `NonFinite` if `input` has a NaN or infinite component.
    /// - `NumericalInstability` if a result overflows.
    pub fn apply_transpose_into(
        self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.apply_checked(input, output, scratch, true, false, Accuracy::Standard)
    }
    // One coordinate, with layout dispatch outside the reduction loop.
    pub(crate) fn dot(
        self,
        coordinate: usize,
        input: &[S],
        (transpose, conjugate): (bool, bool),
        accuracy: Accuracy,
    ) -> Result<S, SciOxideError> {
        if self.values.is_empty() {
            return Ok(S::ZERO);
        }
        let reduce =
            |index: usize, value: S| (if conjugate { value.conj() } else { value }, input[index]);
        match (self.layout, transpose) {
            (
                Layout::Csr {
                    ptr, col: indices, ..
                },
                false,
            )
            | (
                Layout::Csc {
                    ptr, row: indices, ..
                },
                true,
            )
            | (
                Layout::Coo {
                    ptr, col: indices, ..
                },
                false,
            ) => {
                let range = ptr[coordinate]..ptr[coordinate + 1];
                scioxide_core::backend::standard::sum_products(
                    indices[range.clone()]
                        .iter()
                        .copied()
                        .zip(self.values[range].iter().copied())
                        .map(|(i, v)| reduce(i, v)),
                    accuracy,
                )
            }
            (Layout::Csr { opposite, .. }, true) | (Layout::Csc { opposite, .. }, false) => {
                let pattern = opposite.pattern();
                scioxide_core::backend::standard::sum_products(
                    (pattern.row_ptr()[coordinate]..pattern.row_ptr()[coordinate + 1]).map(|k| {
                        reduce(
                            pattern.col_indices()[k],
                            self.values[opposite.positions()[k]],
                        )
                    }),
                    accuracy,
                )
            }
            (
                Layout::Coo {
                    opposite: (ptr, order, indices),
                    ..
                },
                true,
            ) => scioxide_core::backend::standard::sum_products(
                order[ptr[coordinate]..ptr[coordinate + 1]]
                    .iter()
                    .map(|&p| reduce(indices[p], self.values[p])),
                accuracy,
            ),
            (
                Layout::Bsr {
                    ptr, col, br, bc, ..
                },
                false,
            ) => scioxide_core::backend::standard::sum_products(
                (ptr[coordinate / br]..ptr[coordinate / br + 1]).flat_map(|block| {
                    (0..bc).map(move |c| {
                        reduce(
                            col[block] * bc + c,
                            self.values[block * br * bc + (coordinate % br) * bc + c],
                        )
                    })
                }),
                accuracy,
            ),
            (
                Layout::Bsr {
                    br,
                    bc,
                    opposite: (ptr, order, majors),
                    ..
                },
                true,
            ) => scioxide_core::backend::standard::sum_products(
                order[ptr[coordinate / bc]..ptr[coordinate / bc + 1]]
                    .iter()
                    .flat_map(|&block| {
                        (0..br).map(move |r| {
                            reduce(
                                majors[block] * br + r,
                                self.values[block * br * bc + r * bc + coordinate % bc],
                            )
                        })
                    }),
                accuracy,
            ),
            (Layout::Dia { offsets }, _) => {
                self.diagonal_dot(coordinate, input, transpose, conjugate, accuracy, offsets)
            }
        }
    }
    fn diagonal_dot(
        self,
        coordinate: usize,
        input: &[S],
        transpose: bool,
        conjugate: bool,
        accuracy: Accuracy,
        offsets: &[isize],
    ) -> Result<S, SciOxideError> {
        scioxide_core::backend::standard::sum_products(
            offsets.iter().enumerate().filter_map(|(p, &d)| {
                let (row, col) = if transpose {
                    (diagonal_row(self.rows, coordinate, d)?, coordinate)
                } else {
                    let col = diagonal_column(coordinate, d)?;
                    if col >= self.cols {
                        return None;
                    }
                    (coordinate, col)
                };
                let v = self.values[p * self.cols + col];
                Some((
                    if conjugate { v.conj() } else { v },
                    input[if transpose { row } else { col }],
                ))
            }),
            accuracy,
        )
    }

    // All writes target unpublished caller scratch.
    pub(crate) fn apply_staged(
        self,
        input: &[S],
        staged: &mut [S],
        transpose: bool,
        conjugate: bool,
        accuracy: Accuracy,
    ) -> Result<(), SciOxideError> {
        if self.values.is_empty() {
            staged.fill(S::ZERO);
            return Ok(());
        }
        // DIA uses moving active diagonal bounds: each padding slot is skipped
        // without scanning all diagonals separately for every output coordinate.
        if let Layout::Dia { offsets } = self.layout {
            let (mut start, mut end) = (0, 0);
            if !transpose {
                start = offsets.len();
                end = offsets.len();
                for (r, slot) in staged.iter_mut().enumerate() {
                    while start > 0 && diagonal_column(r, offsets[start - 1]).is_some() {
                        start -= 1;
                    }
                    while end > start
                        && diagonal_column(r, offsets[end - 1]).is_none_or(|c| c >= self.cols)
                    {
                        end -= 1;
                    }
                    *slot = scioxide_core::backend::standard::sum_products(
                        (start..end).filter_map(|p| {
                            let c = diagonal_column(r, offsets[p])?;
                            (c < self.cols).then(|| {
                                let v = self.values[p * self.cols + c];
                                (if conjugate { v.conj() } else { v }, input[c])
                            })
                        }),
                        accuracy,
                    )?;
                }
            } else {
                for (c, slot) in staged.iter_mut().enumerate() {
                    while end < offsets.len() && (offsets[end] <= 0 || offsets[end] as usize <= c) {
                        end += 1;
                    }
                    while start < end && diagonal_row(self.rows, c, offsets[start]).is_none() {
                        start += 1;
                    }
                    *slot = scioxide_core::backend::standard::sum_products(
                        (start..end).filter_map(|p| {
                            let r = diagonal_row(self.rows, c, offsets[p])?;
                            let v = self.values[p * self.cols + c];
                            Some((if conjugate { v.conj() } else { v }, input[r]))
                        }),
                        accuracy,
                    )?;
                }
            }
            return Ok(());
        }
        if let Layout::Bsr {
            ptr, col, br, bc, ..
        } = self.layout
            && !transpose
            && accuracy == Accuracy::Standard
        {
            staged.fill(S::ZERO);
            let mut retry = false;
            for (block_row, bounds) in ptr.windows(2).enumerate() {
                for block in bounds[0]..bounds[1] {
                    for (r, slot) in staged[block_row * br..(block_row + 1) * br]
                        .iter_mut()
                        .enumerate()
                    {
                        for (c, &x) in input[col[block] * bc..(col[block] + 1) * bc]
                            .iter()
                            .enumerate()
                        {
                            let a = self.values[block * br * bc + r * bc + c];
                            let p = a * x;
                            retry |= scioxide_core::backend::standard::product_needs_retry(a, x, p);
                            *slot = *slot + p;
                        }
                    }
                }
            }
            if !retry && staged.iter().all(|v| v.has_finite_components()) {
                return Ok(());
            }
        }
        // Dispatch once per application. Compressed rows borrow only their
        // scalar/index slices; they do not copy or branch on the full view.
        match (self.layout, transpose) {
            (
                Layout::Csr {
                    ptr, col: indices, ..
                },
                false,
            )
            | (
                Layout::Csc {
                    ptr, row: indices, ..
                },
                true,
            )
            | (
                Layout::Coo {
                    ptr, col: indices, ..
                },
                false,
            ) => {
                for (bounds, slot) in ptr.windows(2).zip(staged) {
                    let range = bounds[0]..bounds[1];
                    *slot = scioxide_core::backend::standard::sum_products(
                        indices[range.clone()]
                            .iter()
                            .zip(&self.values[range])
                            .map(|(&i, &v)| (if conjugate { v.conj() } else { v }, input[i])),
                        accuracy,
                    )?;
                }
            }
            (Layout::Csr { opposite, .. }, true) | (Layout::Csc { opposite, .. }, false) => {
                let pattern = opposite.pattern();
                for (bounds, slot) in pattern.row_ptr().windows(2).zip(staged) {
                    *slot = scioxide_core::backend::standard::sum_products(
                        (bounds[0]..bounds[1]).map(|k| {
                            let v = self.values[opposite.positions()[k]];
                            (
                                if conjugate { v.conj() } else { v },
                                input[pattern.col_indices()[k]],
                            )
                        }),
                        accuracy,
                    )?;
                }
            }
            (
                Layout::Coo {
                    opposite: (ptr, order, indices),
                    ..
                },
                true,
            ) => {
                for (bounds, slot) in ptr.windows(2).zip(staged) {
                    *slot = scioxide_core::backend::standard::sum_products(
                        order[bounds[0]..bounds[1]].iter().map(|&p| {
                            let v = self.values[p];
                            (if conjugate { v.conj() } else { v }, input[indices[p]])
                        }),
                        accuracy,
                    )?;
                }
            }
            _ => {
                for (r, slot) in staged.iter_mut().enumerate() {
                    *slot = self.dot(r, input, (transpose, conjugate), accuracy)?;
                }
            }
        }
        Ok(())
    }
    // Owned `y = op(A) x` with the checks of `apply_checked` and no scratch.
    fn apply_owned(
        self,
        input: &[S],
        transpose: bool,
        conjugate: bool,
        accuracy: Accuracy,
    ) -> Result<Vec<S>, SciOxideError> {
        let (rows, cols) = if transpose {
            (self.cols, self.rows)
        } else {
            (self.rows, self.cols)
        };
        crate::helpers::exact(input.len(), cols)?;
        finite(input)?;
        let mut output = try_filled(rows, S::ZERO, scioxide_core::quantity!("sparse_product"))?;
        self.apply_staged(input, &mut output, transpose, conjugate, accuracy)?;
        Ok(output)
    }
    /// Returns `y = A x` in a new vector, with [`Accuracy::Standard`].
    ///
    /// Each entry of `y` is the sum of the products of a stored row of `A`
    /// with `x`, in `O(nnz + rows)` operations. The result equals
    /// [`apply_into`](LinearOperator::apply_into) with the same input.
    ///
    /// ```
    /// use scioxide_core::SciOxideErrorKind;
    /// use scioxide_sparse::CsrMatrix;
    ///
    /// // A = [[1, 0, 2], [0, 3, 0]]
    /// let a = CsrMatrix::from_raw(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0_f64, 2.0, 3.0])?;
    /// let y = a.view().apply(&[1.0, 1.0, 1.0])?;
    /// assert!((y[0] - 3.0).abs() < 1e-15 && (y[1] - 3.0).abs() < 1e-15);
    /// let x = a.view().apply_transpose(&[1.0, 1.0])?;
    /// assert!(
    ///     x.iter()
    ///         .zip([1.0, 3.0, 2.0])
    ///         .all(|(v, e)| (v - e).abs() < 1e-15)
    /// );
    /// let short = a.view().apply(&[1.0]).unwrap_err();
    /// assert_eq!(short.kind(), SciOxideErrorKind::DimensionMismatch);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` does not have `cols` entries.
    /// - `NonFinite` if `input` has a NaN or infinite component.
    /// - `NumericalInstability` if a result overflows.
    /// - `AllocationFailure` if `y` cannot be allocated.
    pub fn apply(self, input: &[S]) -> Result<Vec<S>, SciOxideError> {
        self.apply_owned(input, false, false, Accuracy::Standard)
    }
    /// Returns `y = Aᵀ x` (transpose without conjugation) in a new vector, with
    /// [`Accuracy::Standard`]. See [`apply`](Self::apply) for an example.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` does not have `rows` entries.
    /// - `NonFinite` if `input` has a NaN or infinite component.
    /// - `NumericalInstability` if a result overflows.
    /// - `AllocationFailure` if `y` cannot be allocated.
    pub fn apply_transpose(self, input: &[S]) -> Result<Vec<S>, SciOxideError> {
        self.apply_owned(input, true, false, Accuracy::Standard)
    }
    /// Returns `y = Aᴴ x` (conjugate transpose) in a new vector, with
    /// [`Accuracy::Standard`]. See [`apply`](Self::apply) for an example.
    ///
    /// # Errors
    ///
    /// - The errors of [`apply_transpose`](Self::apply_transpose).
    pub fn apply_adjoint(self, input: &[S]) -> Result<Vec<S>, SciOxideError> {
        self.apply_owned(input, true, true, Accuracy::Standard)
    }
    pub(crate) fn apply_checked(
        self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
        transpose: bool,
        conjugate: bool,
        accuracy: Accuracy,
    ) -> Result<(), SciOxideError> {
        let (rows, cols) = if transpose {
            (self.cols, self.rows)
        } else {
            (self.rows, self.cols)
        };
        crate::helpers::exact(input.len(), cols)?;
        crate::helpers::exact(output.len(), rows)?;
        crate::helpers::minimum(scratch.len(), rows)?;

        finite(input)?;
        self.apply_staged(input, &mut scratch[..rows], transpose, conjugate, accuracy)?;
        output.copy_from_slice(&scratch[..rows]);
        Ok(())
    }
}
fn diagonal_column(row: usize, offset: isize) -> Option<usize> {
    if offset >= 0 {
        row.checked_add(offset as usize)
    } else {
        row.checked_sub(offset.unsigned_abs())
    }
}

/// Applies a borrowed sparse matrix as a linear operator with a chosen
/// [`Accuracy`].
///
/// Created by [`SparseView::with_accuracy`]. It implements [`LinearOperator`]
/// (`y = A x`, scratch of `rows`), [`AdjointOperator`] (`y = Aᴴ x`, scratch of
/// `cols`) and [`BlockOperator`].
///
/// ```
/// use scioxide_core::{Accuracy, SciOxideErrorKind, LinearOperator};
/// use scioxide_sparse::CsrMatrix;
///
/// let a = CsrMatrix::from_raw(1, 2, vec![0, 2], vec![0, 1], vec![2.0_f64, 3.0])?;
/// let op = a.view().with_accuracy(Accuracy::Exact);
/// let (mut y, mut x) = ([0.0], [0.0; 2]);
/// op.apply_into(&[1.0, 4.0], &mut y, &mut [0.0])?;
/// op.apply_transpose_into(&[2.0], &mut x, &mut [0.0; 2])?;
/// assert_eq!((y, x), ([14.0], [4.0, 6.0]));
/// let err = op.apply_into(&[1.0], &mut y, &mut [0.0]).unwrap_err();
/// assert_eq!(err.kind(), SciOxideErrorKind::DimensionMismatch);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SparseOperator<'a, S: NumericScalar> {
    view: SparseView<'a, S>,
    accuracy: Accuracy,
}
impl<S: NumericScalar> SparseOperator<'_, S> {
    /// Returns `y = A x` in a new vector, with the selected accuracy.
    ///
    /// # Errors
    ///
    /// - The errors of [`SparseView::apply`].
    pub fn apply(self, input: &[S]) -> Result<Vec<S>, SciOxideError> {
        self.view.apply_owned(input, false, false, self.accuracy)
    }
    /// Returns `y = Aᵀ x` in a new vector, with the selected accuracy.
    ///
    /// # Errors
    ///
    /// - The errors of [`SparseView::apply_transpose`].
    pub fn apply_transpose(self, input: &[S]) -> Result<Vec<S>, SciOxideError> {
        self.view.apply_owned(input, true, false, self.accuracy)
    }
    /// Returns `y = Aᴴ x` in a new vector, with the selected accuracy.
    ///
    /// # Errors
    ///
    /// - The errors of [`SparseView::apply_adjoint`].
    pub fn apply_adjoint(self, input: &[S]) -> Result<Vec<S>, SciOxideError> {
        self.view.apply_owned(input, true, true, self.accuracy)
    }
    /// Computes `y = Aᵀ x` (transpose without conjugation) into `output`, with
    /// the selected accuracy.
    ///
    /// `scratch` needs at least `cols` entries.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `input` does not have `rows` entries, `output`
    ///   does not have `cols` entries, or `scratch` has fewer than `cols`.
    /// - `NonFinite` if `input` has a NaN or infinite component.
    /// - `NumericalInstability` if a result overflows.
    pub fn apply_transpose_into(
        self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.view
            .apply_checked(input, output, scratch, true, false, self.accuracy)
    }
}
impl<S: NumericScalar> LinearOperator<S> for SparseOperator<'_, S> {
    fn rows(&self) -> usize {
        self.view.rows
    }
    fn cols(&self) -> usize {
        self.view.cols
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.view.rows)
    }
    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.view
            .apply_checked(input, output, scratch, false, false, self.accuracy)
    }
}
impl<S: NumericScalar> AdjointOperator<S> for SparseOperator<'_, S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.view.cols)
    }
    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.view
            .apply_checked(input, output, scratch, true, true, self.accuracy)
    }
}
impl<S: NumericScalar> BlockOperator<S> for SparseOperator<'_, S> {}

/// Iterates over the stored `(row, column, value)` triplets of a
/// [`SparseView`], explicit zeros included.
///
/// Created by [`SparseView::iter`]. Entries follow storage order, and DIA
/// padding is skipped.
///
/// ```
/// use scioxide_sparse::DiaMatrix;
///
/// // Diagonals 0 and 1 of a 2 × 2 matrix; slot 0 of diagonal 1 is padding.
/// let a = DiaMatrix::from_raw(2, 2, vec![0, 1], vec![1.0_f64, 0.0, 0.0, 5.0])?;
/// let entries: Vec<_> = a.view().iter().collect();
/// assert_eq!(entries, [(0, 0, 1.0), (1, 1, 0.0), (0, 1, 5.0)]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct SparseEntries<'a, S: NumericScalar> {
    view: SparseView<'a, S>,
    position: usize,
    end: usize,
    major: usize,
    column_filter: Option<usize>,
    mapped: Option<(&'a [usize], &'a [usize])>,
}
impl<S: NumericScalar> Iterator for SparseEntries<'_, S> {
    type Item = (usize, usize, S);
    fn next(&mut self) -> Option<Self::Item> {
        while self.position < self.end {
            let mut p = self.position;
            self.position += 1;
            if let Layout::Dia { offsets } = self.view.layout
                && let Some(col) = self.column_filter
            {
                let Some(row) = diagonal_row(self.view.rows, col, offsets[p]) else {
                    continue;
                };
                return Some((row, col, self.view.values[p * self.view.cols + col]));
            }
            if let (
                Layout::Bsr {
                    br,
                    bc,
                    opposite: (_, order, majors),
                    ..
                },
                Some(column),
            ) = (self.view.layout, self.column_filter)
            {
                let block = order[p / br];
                let row = majors[block] * br + p % br;
                return Some((
                    row,
                    column,
                    self.view.values[block * br * bc + (p % br) * bc + column % bc],
                ));
            }
            if let (Layout::Csr { opposite, .. }, Some(column)) =
                (self.view.layout, self.column_filter)
            {
                return Some((
                    opposite.pattern().col_indices()[p],
                    column,
                    self.view.values[opposite.positions()[p]],
                ));
            }
            if let Some((mapped, majors)) = self.mapped {
                p = mapped[p];
                self.major = majors[p];
            }
            let (r, c) = match self.view.layout {
                Layout::Coo { row, col, .. } => (row[p], col[p]),
                Layout::Csr { ptr, col, .. } => {
                    while ptr[self.major + 1] <= p {
                        self.major += 1;
                    }
                    (self.major, col[p])
                }
                Layout::Csc { ptr, row, .. } => {
                    while ptr[self.major + 1] <= p {
                        self.major += 1;
                    }
                    (row[p], self.major)
                }
                Layout::Bsr {
                    ptr, col, br, bc, ..
                } => {
                    let block = p / (br * bc);
                    while ptr[self.major + 1] <= block {
                        self.major += 1;
                    }
                    (
                        self.major * br + (p % (br * bc)) / bc,
                        col[block] * bc + p % bc,
                    )
                }
                Layout::Dia { offsets } => {
                    let c = p % self.view.cols;
                    let offset = offsets[p / self.view.cols];
                    let r = if offset >= 0 {
                        c.checked_sub(offset as usize)
                    } else {
                        c.checked_add(offset.unsigned_abs())
                    };
                    let Some(r) = r.filter(|&r| r < self.view.rows) else {
                        continue;
                    };
                    (r, c)
                }
            };
            if self.column_filter.is_none_or(|col| col == c) {
                return Some((r, c, self.view.values[p]));
            }
        }
        None
    }
}
impl<S: NumericScalar> LinearOperator<S> for SparseView<'_, S> {
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
        self.apply_checked(input, output, scratch, false, false, Accuracy::Standard)
    }
}
impl<S: NumericScalar> AdjointOperator<S> for SparseView<'_, S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.cols)
    }
    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        scratch: &mut [S],
    ) -> Result<(), SciOxideError> {
        self.apply_checked(input, output, scratch, true, true, Accuracy::Standard)
    }
}
impl<S: NumericScalar> BlockOperator<S> for SparseView<'_, S> {}

impl<S: NumericScalar> CooMatrix<S> {
    /// Returns a borrowed [`SparseView`] of the matrix.
    pub fn view(&self) -> SparseView<'_, S> {
        SparseView {
            rows: self.rows(),
            cols: self.cols(),
            values: self.values(),
            layout: Layout::Coo {
                row: self.row_indices(),
                col: self.col_indices(),
                ptr: &self.row_ptr,
                opposite: (&self.column_ptr, &self.column_order, self.row_indices()),
            },
        }
    }
}

impl<S: NumericScalar> CsrMatrix<S> {
    /// Returns a borrowed [`SparseView`] of the matrix.
    pub fn view(&self) -> SparseView<'_, S> {
        SparseView {
            rows: self.rows(),
            cols: self.cols(),
            values: self.values(),
            layout: Layout::Csr {
                ptr: self.row_ptr(),
                col: self.col_indices(),
                opposite: self.opposite_traversal(),
            },
        }
    }
}

impl<S: NumericScalar> CscMatrix<S> {
    /// Returns a borrowed [`SparseView`] of the matrix.
    pub fn view(&self) -> SparseView<'_, S> {
        SparseView {
            rows: self.rows(),
            cols: self.cols(),
            values: self.values(),
            layout: Layout::Csc {
                ptr: self.col_ptr(),
                row: self.row_indices(),
                opposite: self.opposite_traversal(),
            },
        }
    }
}

/// Generates the owned products `apply` and `apply_adjoint` of an owned
/// matrix type from its [`SparseView`].
macro_rules! impl_owned_products {
    ($($name:ident),* $(,)?) => {$(
        impl<S: NumericScalar> $name<S> {
            /// Returns `y = A x` in a new vector, with [`Accuracy::Standard`],
            /// as [`SparseView::apply`] does.
            ///
            /// # Errors
            ///
            /// - The errors of [`SparseView::apply`].
            pub fn apply(&self, input: &[S]) -> Result<Vec<S>, SciOxideError> {
                self.view().apply(input)
            }
            /// Returns `y = Aᴴ x` in a new vector, with [`Accuracy::Standard`],
            /// as [`SparseView::apply_adjoint`] does.
            ///
            /// # Errors
            ///
            /// - The errors of [`SparseView::apply_adjoint`].
            pub fn apply_adjoint(&self, input: &[S]) -> Result<Vec<S>, SciOxideError> {
                self.view().apply_adjoint(input)
            }
        }
    )*};
}
impl_owned_products!(CsrMatrix, CscMatrix, BsrMatrix, DiaMatrix);
