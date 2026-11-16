use alloc::vec::Vec;

use scioxide_core::{
    InvalidInputReason, NonFiniteOrigin, NumericScalar, ProductSum, SciOxideError,
};

use crate::{CscMatrix, CsrMatrix};

/// Selects how [`SparseBuilder`] treats several triplets with the same
/// coordinate.
///
/// Duplicates are resolved first, and the [`ExplicitZeroPolicy`] then sees the
/// resolved value. See [`SparseBuilder`] for an example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum DuplicatePolicy {
    /// Adds the values. The sum is accumulated exactly and rounded once, so it
    /// does not depend on the order of the triplets.
    Sum,
    /// Rejects the matrix with `InvalidInput` at the first repeated coordinate
    /// in row-major order.
    Reject,
}

/// Selects whether entries whose value is exactly zero stay in the sparsity
/// pattern.
///
/// The policy applies after duplicates are summed. See [`SparseBuilder`] for an
/// example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum ExplicitZeroPolicy {
    /// Keeps exact zeros as stored entries.
    Preserve,
    /// Removes exact zeros from the pattern.
    Drop,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Entry<S> {
    pub(crate) row: usize,
    pub(crate) col: usize,
    pub(crate) value: S,
}

// Checks one triplet before appending it.
fn validate_entry<S: NumericScalar>(
    rows: usize,
    cols: usize,
    row: usize,
    col: usize,
    value: S,
) -> Result<(), SciOxideError> {
    validate_coordinate(rows, cols, row, col)?;
    if !value.has_finite_components() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "sparse values must have finite components",
        ));
    }
    Ok(())
}

fn validate_coordinate(
    rows: usize,
    cols: usize,
    row: usize,
    col: usize,
) -> Result<(), SciOxideError> {
    if row >= rows {
        return Err(SciOxideError::out_of_bounds(
            scioxide_core::quantity!("sparse_rows"),
            row,
            rows,
        ));
    }
    if col >= cols {
        return Err(SciOxideError::out_of_bounds(
            scioxide_core::quantity!("sparse_columns"),
            col,
            cols,
        ));
    }
    Ok(())
}

// Sorts and resolves builder triplets.
fn canonical_entries<S: NumericScalar>(
    mut entries: Vec<Entry<S>>,
    duplicates: DuplicatePolicy,
    zeros: ExplicitZeroPolicy,
) -> Result<Vec<Entry<S>>, SciOxideError> {
    entries.sort_unstable_by_key(|entry| (entry.row, entry.col));
    let mut written = 0;
    let mut cursor = 0;
    while cursor < entries.len() {
        let coordinate = (entries[cursor].row, entries[cursor].col);
        let mut end = cursor + 1;
        while end < entries.len() && (entries[end].row, entries[end].col) == coordinate {
            end += 1;
        }
        if end - cursor > 1 && duplicates == DuplicatePolicy::Reject {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                InvalidInputReason::Malformed,
                alloc::format!(
                    "duplicate coordinate ({}, {}) is rejected by policy",
                    coordinate.0,
                    coordinate.1
                ),
            ));
        }

        let value = if end - cursor == 1 {
            entries[cursor].value
        } else {
            let mut sum = ProductSum::new();
            entries[cursor..end]
                .iter()
                .try_for_each(|entry| sum.add_product(entry.value, S::ONE))?;
            sum.finish()?
        };
        if zeros == ExplicitZeroPolicy::Preserve || value != S::ZERO {
            entries[written] = Entry {
                row: coordinate.0,
                col: coordinate.1,
                value,
            };
            written += 1;
        }
        cursor = end;
    }
    entries.truncate(written);
    Ok(entries)
}

/// Collects `(row, column, value)` triplets in any order and creates a sparse
/// matrix from them.
///
/// Each triplet is checked when it is added: its indices must lie inside the
/// matrix and its value must be finite. The `finish_*` methods sort the
/// triplets in row-major order, which costs `O(t log t)` for `t` triplets,
/// resolve duplicates by a [`DuplicatePolicy`] and then exact zeros by an
/// [`ExplicitZeroPolicy`], and create the matrix in the requested format.
/// [`CsrMatrix::from_triplets`] does the same in one call with
/// `(Sum, Preserve)`.
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};
///
/// // The two triplets at (0, 1) sum to 2 − 2 = 0.
/// let mut builder = SparseBuilder::new(1, 2);
/// builder.extend(&[0, 0, 0], &[1, 0, 1], &[2.0_f64, 1.0, -2.0])?;
/// let kept = builder
///     .clone()
///     .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)?;
/// assert_eq!(kept.values(), &[1.0, 0.0]);
/// let dropped = builder
///     .clone()
///     .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)?;
/// assert_eq!(dropped.values(), &[1.0]);
/// let err = builder.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve);
/// assert_eq!(err.unwrap_err().kind(), SciOxideErrorKind::InvalidInput);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct SparseBuilder<S: NumericScalar> {
    rows: usize,
    cols: usize,
    entries: Vec<Entry<S>>,
}

impl<S: NumericScalar> SparseBuilder<S> {
    /// Creates an empty builder for a `rows × cols` matrix.
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: Vec::new(),
        }
    }

    /// Creates an empty builder for a `rows × cols` matrix with room for
    /// `capacity` triplets.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the triplet storage cannot be reserved.
    pub fn with_capacity(rows: usize, cols: usize, capacity: usize) -> Result<Self, SciOxideError> {
        let mut entries = Vec::new();
        entries.try_reserve_exact(capacity).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("sparse_builder_entries"),
                capacity,
            )
        })?;
        Ok(Self {
            rows,
            cols,
            entries,
        })
    }

    /// Returns the number of triplets added so far, counting duplicates and
    /// zeros.
    pub fn triplet_count(&self) -> usize {
        self.entries.len()
    }

    /// Adds the triplet `(row, col, value)`.
    ///
    /// On error the builder is unchanged.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `row` or `col` lies outside the matrix.
    /// - `NonFinite` if `value` has a NaN or infinite component.
    /// - `SizeOverflow` or `AllocationFailure` if the triplet storage cannot grow.
    pub fn push(&mut self, row: usize, col: usize, value: S) -> Result<(), SciOxideError> {
        validate_entry(self.rows, self.cols, row, col, value)?;
        self.append(row, col, value)
    }

    // Appends a value already proven finite by an accepted source or a checked
    // kernel; only the coordinate is checked.
    pub(crate) fn push_finite(
        &mut self,
        row: usize,
        col: usize,
        value: S,
    ) -> Result<(), SciOxideError> {
        validate_coordinate(self.rows, self.cols, row, col)?;
        self.append(row, col, value)
    }

    fn append(&mut self, row: usize, col: usize, value: S) -> Result<(), SciOxideError> {
        let required = self.entries.len().checked_add(1).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("sparse_builder_entries"))
        })?;
        self.entries.try_reserve(1).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("sparse_builder_entries"),
                required,
            )
        })?;
        self.entries.push(Entry { row, col, value });
        Ok(())
    }

    /// Adds the triplets `(rows[k], cols[k], values[k])`.
    ///
    /// All triplets are checked before any is added, so on error the builder
    /// is unchanged.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the three slices have different lengths.
    /// - `OutOfBounds` if an index lies outside the matrix.
    /// - `NonFinite` if a value has a NaN or infinite component.
    /// - `SizeOverflow` or `AllocationFailure` if the triplet storage cannot grow.
    pub fn extend(
        &mut self,
        rows: &[usize],
        cols: &[usize],
        values: &[S],
    ) -> Result<(), SciOxideError> {
        if rows.len() != cols.len() || rows.len() != values.len() {
            return Err(SciOxideError::dimension_mismatch(
                [rows.len(), rows.len()],
                [cols.len(), values.len()],
            ));
        }
        rows.iter()
            .copied()
            .zip(cols.iter().copied())
            .zip(values.iter().copied())
            .try_for_each(|((row, col), value)| {
                validate_entry(self.rows, self.cols, row, col, value)
            })?;
        let required = self
            .entries
            .len()
            .checked_add(values.len())
            .ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("sparse_builder_entries"))
            })?;
        self.entries.try_reserve(values.len()).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("sparse_builder_entries"),
                required,
            )
        })?;
        self.entries.extend(
            rows.iter()
                .copied()
                .zip(cols.iter().copied())
                .zip(values.iter().copied())
                .map(|((row, col), value)| Entry { row, col, value }),
        );
        Ok(())
    }

    /// Creates a [`CooMatrix`] from the triplets, resolving duplicates by
    /// `duplicates` and then exact zeros by `zeros`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a coordinate repeats under [`DuplicatePolicy::Reject`].
    /// - `NumericalInstability` if a sum of duplicates overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn finish_coo(
        self,
        duplicates: DuplicatePolicy,
        zeros: ExplicitZeroPolicy,
    ) -> Result<CooMatrix<S>, SciOxideError> {
        CooMatrix::from_entries(
            self.rows,
            self.cols,
            canonical_entries(self.entries, duplicates, zeros)?,
        )
    }
    /// Creates a [`CsrMatrix`] from the triplets, without an intermediate COO
    /// matrix, resolving duplicates by `duplicates` and then exact zeros by
    /// `zeros`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a coordinate repeats under [`DuplicatePolicy::Reject`].
    /// - `NumericalInstability` if a sum of duplicates overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn finish_csr(
        self,
        duplicates: DuplicatePolicy,
        zeros: ExplicitZeroPolicy,
    ) -> Result<CsrMatrix<S>, SciOxideError> {
        CsrMatrix::from_entries(
            self.rows,
            self.cols,
            canonical_entries(self.entries, duplicates, zeros)?,
        )
    }

    /// Creates a [`CscMatrix`] from the triplets, without an intermediate COO
    /// matrix, resolving duplicates by `duplicates` and then exact zeros by
    /// `zeros`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if a coordinate repeats under [`DuplicatePolicy::Reject`].
    /// - `NumericalInstability` if a sum of duplicates overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn finish_csc(
        self,
        duplicates: DuplicatePolicy,
        zeros: ExplicitZeroPolicy,
    ) -> Result<CscMatrix<S>, SciOxideError> {
        CscMatrix::from_entries(
            self.rows,
            self.cols,
            canonical_entries(self.entries, duplicates, zeros)?,
        )
    }
}

/// Stores a sparse matrix as `(row, column, value)` triplets.
///
/// The triplets are sorted by row, then column, and no coordinate repeats.
/// Every value is finite. The coordinates are fixed after construction; the
/// values can change through `update_values`. A `CooMatrix` is created by
/// [`SparseBuilder::finish_coo`].
///
/// ```
/// use scioxide_sparse::{DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};
///
/// let mut builder = SparseBuilder::new(2, 2);
/// builder.extend(&[1, 0], &[0, 1], &[3.0_f64, 2.0])?;
/// let coo = builder.finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)?;
/// // The triplets come back in row-major order.
/// assert_eq!(
///     (coo.row_indices(), coo.col_indices()),
///     (&[0, 1][..], &[1, 0][..])
/// );
/// assert_eq!(coo.iter().collect::<Vec<_>>(), [(0, 1, 2.0), (1, 0, 3.0)]);
/// assert_eq!(coo.to_csr()?.get(1, 0)?, 3.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct CooMatrix<S: NumericScalar> {
    rows: usize,
    cols: usize,
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<S>,
    pub(crate) row_ptr: Vec<usize>,
    pub(crate) column_ptr: Vec<usize>,
    pub(crate) column_order: Vec<usize>,
}

impl<S: NumericScalar> CooMatrix<S> {
    // Moves canonical entries into COO storage.
    pub(crate) fn from_entries(
        rows: usize,
        cols: usize,
        entries: Vec<Entry<S>>,
    ) -> Result<Self, SciOxideError> {
        let allocation_error =
            |quantity| SciOxideError::allocation_failure(quantity, entries.len());
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        row_indices
            .try_reserve_exact(entries.len())
            .map_err(|_| allocation_error(scioxide_core::quantity!("coo_row_indices")))?;
        col_indices
            .try_reserve_exact(entries.len())
            .map_err(|_| allocation_error(scioxide_core::quantity!("coo_column_indices")))?;
        values
            .try_reserve_exact(entries.len())
            .map_err(|_| allocation_error(scioxide_core::quantity!("coo_values")))?;
        entries.into_iter().for_each(|entry| {
            row_indices.push(entry.row);
            col_indices.push(entry.col);
            values.push(entry.value);
        });
        let row_ptr = if values.is_empty() {
            Vec::new()
        } else {
            crate::helpers::counting_transpose(
                rows,
                row_indices.iter().copied().map(|r| (r, ())),
                |_, ()| {},
            )?
        };
        let (column_ptr, column_order) = if values.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            crate::compressed::transpose_index(cols, &col_indices)?
        };
        Ok(Self {
            rows,
            cols,
            row_indices,
            col_indices,
            values,
            row_ptr,
            column_ptr,
            column_order,
        })
    }

    /// Number of rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the number of stored entries, including explicit zeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Returns the row index of each stored entry.
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Returns the column index of each stored entry.
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Returns the value of each stored entry.
    pub fn values(&self) -> &[S] {
        &self.values
    }

    /// Iterates over the stored `(row, column, value)` triplets in row-major
    /// order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (usize, usize, S)> + Clone + '_ {
        self.row_indices
            .iter()
            .copied()
            .zip(self.col_indices.iter().copied())
            .zip(self.values.iter().copied())
            .map(|((row, col), value)| (row, col, value))
    }

    /// Converts to a [`CsrMatrix`] with the same entries, in `O(nnz + rows + cols)`
    /// time.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn to_csr(&self) -> Result<CsrMatrix<S>, SciOxideError> {
        CsrMatrix::from_coo(self)
    }

    /// Converts to a [`CscMatrix`] with the same entries, in `O(nnz + rows + cols)`
    /// time.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
    pub fn to_csc(&self) -> Result<CscMatrix<S>, SciOxideError> {
        CscMatrix::from_coo(self)
    }
}

#[cfg(test)]
mod tests {
    use super::{DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};

    #[test]
    fn canonicalization_is_permutation_independent_and_policy_driven() {
        let build = |order: &[usize], zeros| {
            let rows = [1, 0, 1, 0];
            let cols = [0, 1, 0, 1];
            let values = [3.0_f64, 2.0, -3.0, 4.0];
            let mut builder = SparseBuilder::new(2, 2);
            order
                .iter()
                .try_for_each(|&index| builder.push(rows[index], cols[index], values[index]))
                .unwrap();
            builder.finish_coo(DuplicatePolicy::Sum, zeros).unwrap()
        };
        let forward = build(&[0, 1, 2, 3], ExplicitZeroPolicy::Preserve);
        let reversed = build(&[3, 2, 1, 0], ExplicitZeroPolicy::Preserve);
        assert_eq!(forward, reversed);
        assert_eq!(forward.values(), &[6.0, 0.0]);
        assert_eq!(
            build(&[0, 1, 2, 3], ExplicitZeroPolicy::Drop).values(),
            &[6.0]
        );
    }

    #[test]
    fn duplicate_rejection_and_range_safe_sum_are_explicit() {
        let mut builder = SparseBuilder::new(1, 1);
        builder.push(0, 0, 1.0_f64).unwrap();
        builder.push(0, 0, 2.0).unwrap();
        assert!(
            builder
                .clone()
                .finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
                .is_err()
        );

        let mut extreme = SparseBuilder::new(1, 1);
        extreme.push(0, 0, f64::MAX).unwrap();
        extreme.push(0, 0, f64::MAX).unwrap();
        assert!(
            extreme
                .finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
                .is_err()
        );
    }
}

crate::values::impl_value_updates!(CooMatrix, values);
