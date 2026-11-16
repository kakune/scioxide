//! BSR and DIA storage, borrowed views of every format, and their products.
use crate::helpers::finite;
use crate::helpers::invalid;
use crate::{CooMatrix, CscMatrix, CsrMatrix, ExplicitZeroPolicy, SparseTraversalView};
use alloc::vec::Vec;
use scioxide_core::backend::{
    checked_add, checked_mul, try_copy_slice, try_filled, try_with_capacity,
};
use scioxide_core::{
    Accuracy, AdjointOperator, Argument, BlockOperator, LinearOperator, NumericScalar,
    SciOxideError, StructuralPattern,
};

mod bsr;
mod dia;
mod view;
pub use bsr::BsrMatrix;
pub use dia::DiaMatrix;
use dia::diagonal_row;
use view::Layout;
pub use view::{SparseEntries, SparseOperator, SparseView};

/// Holds the array lengths, in elements rather than bytes, that a conversion to
/// BSR or DIA would allocate.
///
/// Returned by [`BsrMatrix::conversion_requirements`] and
/// [`DiaMatrix::conversion_requirements`], so you can compare formats before
/// converting.
///
/// ```
/// use scioxide_sparse::{BsrMatrix, CsrMatrix, DiaMatrix};
///
/// // [[1, 0, 0, 3],
/// //  [0, 2, 0, 0]] occupies two 2 × 2 blocks and diagonals 0 and 3.
/// let a = CsrMatrix::from_raw(2, 4, vec![0, 2, 3], vec![0, 3, 1], vec![1.0_f64, 3.0, 2.0])?;
/// let bsr = BsrMatrix::conversion_requirements(a.view(), 2, 2)?;
/// assert_eq!((bsr.pointers(), bsr.indices(), bsr.values()), (2, 2, 8));
/// assert_eq!((bsr.traversal_pointers(), bsr.traversal_indices()), (3, 4));
/// let dia = DiaMatrix::conversion_requirements(a.view())?;
/// assert_eq!((dia.pointers(), dia.indices(), dia.values()), (0, 2, 8));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseStorageRequirements {
    /// Compressed pointers, or zero for DIA.
    pointers: usize,
    /// Block-column indices or diagonal offsets.
    indices: usize,
    /// Scalar slots, including introduced zeros and padding.
    values: usize,
    /// Block-column pointers of the column-wise index, or zero when not needed.
    traversal_pointers: usize,
    /// Order and block-row indices of the column-wise index, or zero for DIA.
    traversal_indices: usize,
}

impl SparseStorageRequirements {
    /// Returns the number of block-row pointers, or zero for DIA.
    pub const fn pointers(self) -> usize {
        self.pointers
    }
    /// Returns the number of block-column indices or diagonal offsets.
    pub const fn indices(self) -> usize {
        self.indices
    }
    /// Returns the number of scalar values, including zeros added inside
    /// blocks and diagonal padding.
    pub const fn values(self) -> usize {
        self.values
    }
    /// Returns the number of block-column pointers of the column-wise index
    /// kept by BSR, or zero.
    pub const fn traversal_pointers(self) -> usize {
        self.traversal_pointers
    }
    /// Returns the number of entries of the column-wise index kept by BSR:
    /// two per stored block, or zero.
    pub const fn traversal_indices(self) -> usize {
        self.traversal_indices
    }
}
