//! Transposed sparsity patterns with the positions of the original values.
use alloc::vec::Vec;
use scioxide_core::backend::{try_copy_slice, try_filled};
use scioxide_core::{SciOxideError, StructuralPattern, StructuralPatternView};

/// Holds the transpose of a sparsity pattern and, for each of its entries, the
/// position of the corresponding value in the original matrix.
///
/// Created by [`transpose_pattern`]. No values are copied: entry `k` of the
/// transposed pattern takes its value from position `positions[k]` of the
/// original values.
///
/// ```
/// use scioxide_core::StructuralPattern;
/// use scioxide_sparse::transpose_pattern;
///
/// // Rows {0, 2} and {1} of a 2 × 3 pattern.
/// let p = StructuralPattern::new(2, 3, vec![0, 2, 3], vec![0, 2, 1])?;
/// let (transposed, positions) = transpose_pattern(p.view())?.into_parts();
/// let t = transposed.view();
/// assert_eq!((t.rows(), t.cols(), t.row_ptr()), (3, 2, &[0, 1, 2, 3][..]));
/// assert_eq!(t.col_indices(), &[0, 1, 0]);
/// assert_eq!(positions, [0, 2, 1]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct SparseTraversal {
    pub(crate) pattern: StructuralPattern,
    pub(crate) positions: Vec<usize>,
}
impl SparseTraversal {
    /// Returns a borrowed view of the pattern and the positions.
    pub fn view(&self) -> SparseTraversalView<'_> {
        SparseTraversalView {
            pattern: self.pattern.view(),
            positions: &self.positions,
        }
    }
    /// Returns the transposed pattern and the positions, consuming `self`.
    pub fn into_parts(self) -> (StructuralPattern, Vec<usize>) {
        (self.pattern, self.positions)
    }
    pub(crate) fn try_copy(&self) -> Result<Self, SciOxideError> {
        Ok(Self {
            pattern: self.pattern.view().to_owned()?,
            positions: try_copy_slice(
                &self.positions,
                scioxide_core::quantity!("sparse_traversal_positions"),
            )?,
        })
    }
}

/// Borrows a sparsity pattern in CSR form together with the positions of its
/// values in another storage order.
///
/// Entry `k` of [`pattern`](Self::pattern) takes its value from
/// `source.values()[positions()[k]]`, and the positions are a permutation of
/// all stored entries, explicit zeros included. For a
/// [`CsrMatrix`](crate::CsrMatrix) the pattern is that of `Aᵀ`, so its rows are
/// the columns of `A`. For a [`CscMatrix`](crate::CscMatrix) the pattern is the
/// CSR pattern of `A`.
///
/// ```
/// use scioxide_sparse::CsrMatrix;
///
/// let a = CsrMatrix::from_raw(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![2., 4., 3.])?;
/// let columns = a.opposite_traversal();
/// assert_eq!(columns.pattern().row_ptr(), &[0, 1, 2, 3]);
/// assert_eq!(columns.pattern().col_indices(), &[0, 1, 0]);
/// let values: Vec<_> = columns.positions().iter().map(|&p| a.values()[p]).collect();
/// assert_eq!(values, [2., 3., 4.]);
/// let csc = a.to_csc()?;
/// let rows = csc.opposite_traversal();
/// assert_eq!(rows.pattern(), a.structural_pattern());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct SparseTraversalView<'a> {
    pattern: StructuralPatternView<'a>,
    positions: &'a [usize],
}
impl<'a> SparseTraversalView<'a> {
    /// Returns the pattern in CSR form; the indices within each row are
    /// strictly increasing.
    pub fn pattern(self) -> StructuralPatternView<'a> {
        self.pattern
    }
    /// Returns, for each entry of [`pattern`](Self::pattern), the position of
    /// its value in the source matrix.
    pub fn positions(self) -> &'a [usize] {
        self.positions
    }
}

/// Computes the transpose of a sparsity pattern and the position of each
/// transposed entry in the original storage order.
///
/// A counting sort by column costs `O(rows + cols + nnz)` time. Stored zeros
/// count as entries like any other.
///
/// ```
/// use scioxide_core::StructuralPattern;
/// use scioxide_sparse::transpose_pattern;
///
/// let p = StructuralPattern::new(2, 3, vec![0, 2, 3], vec![0, 2, 1])?;
/// let transposed = transpose_pattern(p.view())?;
/// assert_eq!(transposed.view().pattern().col_indices(), &[0, 1, 0]);
/// assert_eq!(transposed.view().positions(), &[0, 2, 1]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `SizeOverflow` or `AllocationFailure` if the transposed pattern cannot be
///   allocated.
pub fn transpose_pattern(
    pattern: StructuralPatternView<'_>,
) -> Result<SparseTraversal, SciOxideError> {
    let nnz = pattern.col_indices().len();
    let mut indices = try_filled(
        nnz,
        0usize,
        scioxide_core::quantity!("transpose_structural_indices"),
    )?;
    let mut positions = try_filled(
        nnz,
        0usize,
        scioxide_core::quantity!("transpose_structural_positions"),
    )?;
    let pointers = crate::helpers::counting_transpose(
        pattern.cols(),
        pattern
            .row_ptr()
            .windows(2)
            .enumerate()
            .flat_map(|(row, bounds)| {
                (bounds[0]..bounds[1]).map(move |p| (pattern.col_indices()[p], (row, p)))
            }),
        |slot, (row, p)| {
            indices[slot] = row;
            positions[slot] = p;
        },
    )?;
    Ok(SparseTraversal {
        pattern: StructuralPattern::new(pattern.cols(), pattern.rows(), pointers, indices)?,
        positions,
    })
}
