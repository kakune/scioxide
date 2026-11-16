//! Entrywise sparse arithmetic, dropping of small values, and block assembly.
use crate::helpers::{finite, invalid};
use crate::{
    BsrMatrix, CsrMatrix, DiaMatrix, DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder, SparseView,
};
use scioxide_core::Accuracy;
use scioxide_core::backend::scaled::scaled_tolerance_threshold;
use scioxide_core::backend::standard::{product, sum_products};
use scioxide_core::backend::{checked_add, checked_mul, try_with_capacity};
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideError, Tolerance};
use scioxide_core::{Scaled, ScaledValue};

/// Holds the accuracy and the zero policy of entrywise sparse arithmetic.
///
/// Used by [`SparseView::scaled_with_options`],
/// [`CsrMatrix::add_with_options`], [`CsrMatrix::subtract_with_options`] and
/// [`CsrMatrix::hadamard_with_options`]. The zero policy decides whether
/// results that are exactly zero, for example from cancellation, stay in the
/// pattern. The default, which the forms without options use, is
/// [`ExplicitZeroPolicy::Preserve`] and [`Accuracy::Standard`]: the pattern of
/// the result then depends only on the patterns of the operands.
///
/// ```
/// use scioxide_core::Accuracy;
/// use scioxide_sparse::{ExplicitZeroPolicy, SparseArithmeticOptions};
///
/// let options = SparseArithmeticOptions::new()
///     .with_zero_policy(ExplicitZeroPolicy::Drop)
///     .with_accuracy(Accuracy::Exact);
/// assert_eq!(options.accuracy(), Accuracy::Exact);
/// assert_eq!(options.zero_policy(), ExplicitZeroPolicy::Drop);
/// assert_eq!(
///     SparseArithmeticOptions::default().zero_policy(),
///     ExplicitZeroPolicy::Preserve
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseArithmeticOptions {
    zeros: ExplicitZeroPolicy,
    accuracy: Accuracy,
}
impl Default for SparseArithmeticOptions {
    fn default() -> Self {
        Self::new()
    }
}
impl SparseArithmeticOptions {
    /// Creates options with [`ExplicitZeroPolicy::Preserve`] and
    /// [`Accuracy::Standard`].
    pub const fn new() -> Self {
        Self {
            zeros: ExplicitZeroPolicy::Preserve,
            accuracy: Accuracy::Standard,
        }
    }
    /// Sets the zero policy applied to the computed values.
    pub const fn with_zero_policy(mut self, zeros: ExplicitZeroPolicy) -> Self {
        self.zeros = zeros;
        self
    }
    /// Sets the accuracy of products and sums.
    pub const fn with_accuracy(mut self, accuracy: Accuracy) -> Self {
        self.accuracy = accuracy;
        self
    }
    /// Returns the accuracy.
    pub const fn accuracy(self) -> Accuracy {
        self.accuracy
    }
    /// Returns the zero policy.
    pub const fn zero_policy(self) -> ExplicitZeroPolicy {
        self.zeros
    }
}
/// Selects which small stored values a pruning removes.
///
/// A value `v` is dropped when `|v| ≤ absolute + relative · reference_scale`,
/// where `absolute` and `relative` come from the [`Tolerance`]. Unlike
/// [`ExplicitZeroPolicy`], which acts only on exact zeros, this removes
/// nonzero values. See [`SparseView::pruned`].
///
/// ```
/// use scioxide_core::{SciOxideErrorKind, Tolerance};
/// use scioxide_sparse::SparsePruning;
///
/// let tolerance = Tolerance::new(1e-12_f64, 1e-6)?;
/// // Drops values with modulus <= 1e-12 + 1e-6 * 4.0.
/// assert!(SparsePruning::new(tolerance, 4.0).is_ok());
/// let err = SparsePruning::new(tolerance, -1.0).unwrap_err();
/// assert_eq!(err.kind(), SciOxideErrorKind::InvalidInput);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SparsePruning<S: RealScalar> {
    tolerance: Tolerance<S>,
    scale: S,
}
impl<S: RealScalar> SparsePruning<S> {
    /// Creates a pruning policy from a tolerance and a reference scale, such
    /// as a norm of the matrix.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `reference_scale` is negative, NaN or infinite.
    pub fn new(tolerance: Tolerance<S>, reference_scale: S) -> Result<Self, SciOxideError> {
        if !reference_scale.is_finite() || reference_scale < S::ZERO {
            return Err(invalid(
                scioxide_core::argument!("reference_scale"),
                "invalid pruning reference scale",
            ));
        }
        Ok(Self {
            tolerance,
            scale: reference_scale,
        })
    }
    // Drop decision for one value.
    fn drops<T: NumericScalar<Real = S>>(self, value: T) -> Result<bool, SciOxideError> {
        let magnitude = ScaledValue::from_value(Complex::new(
            value.real_component(),
            value.imaginary_component(),
        ))?
        .magnitude()?;
        Ok(magnitude.abs_le(scaled_tolerance_threshold(
            &self.tolerance,
            Scaled::from_value(self.scale)?,
        )?))
    }
}
impl<S: NumericScalar> SparseView<'_, S> {
    /// Returns `scale · A` as a [`CsrMatrix`] with the pattern of `A`.
    ///
    /// Each stored entry is multiplied by `scale` with
    /// [`Accuracy::Standard`], and a product that is zero stays stored. This is
    /// [`scaled_with_options`](Self::scaled_with_options) with
    /// [`SparseArithmeticOptions::new`].
    ///
    /// ```
    /// use scioxide_core::SciOxideErrorKind;
    /// use scioxide_sparse::CsrMatrix;
    ///
    /// let a = CsrMatrix::from_raw(1, 3, vec![0, 2], vec![0, 2], vec![1.5_f64, -2.0])?;
    /// assert_eq!(a.view().scaled(2.0)?.values(), &[3.0, -4.0]);
    /// assert_eq!(a.view().scaled(0.0)?.nnz(), 2);
    /// let err = a.view().scaled(f64::NAN).unwrap_err();
    /// assert_eq!(err.kind(), SciOxideErrorKind::NonFinite);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `scale` is NaN or infinite.
    /// - `NumericalInstability` if a product overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the result cannot be
    ///   allocated.
    pub fn scaled(self, scale: S) -> Result<CsrMatrix<S>, SciOxideError> {
        self.scaled_with_options(scale, SparseArithmeticOptions::new())
    }
    /// Returns `scale · A` as a [`CsrMatrix`], with the accuracy and zero
    /// policy of `options`.
    ///
    /// Only stored entries are scaled, so the result has at most the pattern of
    /// `A`. With a zero scale, [`ExplicitZeroPolicy::Drop`] gives an empty
    /// pattern.
    ///
    /// ```
    /// use scioxide_sparse::{CsrMatrix, ExplicitZeroPolicy, SparseArithmeticOptions};
    ///
    /// let a = CsrMatrix::from_raw(1, 3, vec![0, 2], vec![0, 2], vec![1.5_f64, -2.0])?;
    /// let drop = SparseArithmeticOptions::new().with_zero_policy(ExplicitZeroPolicy::Drop);
    /// assert_eq!(a.view().scaled_with_options(0.0, drop)?.nnz(), 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - The errors of [`scaled`](Self::scaled).
    pub fn scaled_with_options(
        self,
        scale: S,
        options: SparseArithmeticOptions,
    ) -> Result<CsrMatrix<S>, SciOxideError> {
        finite(&[scale])?;
        let mut builder =
            SparseBuilder::with_capacity(self.rows(), self.cols(), self.values().len())?;
        for (r, c, v) in self.iter() {
            builder.push_finite(r, c, product(v, scale, options.accuracy)?)?;
        }
        builder.finish_csr(DuplicatePolicy::Sum, options.zeros)
    }
    /// Returns a [`CsrMatrix`] without the stored values that `policy` drops.
    ///
    /// The source is unchanged. Moduli and the threshold are compared with
    /// rescaling, so a complex modulus larger than the largest finite real is
    /// still compared correctly.
    ///
    /// ```
    /// use scioxide_core::Tolerance;
    /// use scioxide_sparse::{CsrMatrix, SparsePruning};
    ///
    /// let a = CsrMatrix::from_raw(1, 3, vec![0, 3], vec![0, 1, 2], vec![1.0_f64, 1e-9, -0.5])?;
    /// // Drop values with modulus <= 1e-6 * 1.0.
    /// let policy = SparsePruning::new(Tolerance::new(0.0, 1e-6)?, 1.0)?;
    /// let pruned = a.view().pruned(policy)?;
    /// assert_eq!(pruned.iter().collect::<Vec<_>>(), [(0, 0, 1.0), (0, 2, -0.5)]);
    /// assert_eq!(a.nnz(), 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the threshold overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the result cannot be
    ///   allocated.
    pub fn pruned(self, policy: SparsePruning<S::Real>) -> Result<CsrMatrix<S>, SciOxideError> {
        let mut builder =
            SparseBuilder::with_capacity(self.rows(), self.cols(), self.values().len())?;
        for (r, c, v) in self.iter() {
            if !policy.drops(v)? {
                builder.push_finite(r, c, v)?;
            }
        }
        builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
    }
}
#[derive(Clone, Copy)]
enum Arithmetic {
    Add,
    Subtract,
    Multiply,
}
// Merges `left` and `right` (the public method's `self` and `other`).
fn combine<S: NumericScalar>(
    left: &CsrMatrix<S>,
    right: &CsrMatrix<S>,
    options: SparseArithmeticOptions,
    arithmetic: Arithmetic,
) -> Result<CsrMatrix<S>, SciOxideError> {
    crate::helpers::shape((right.rows(), right.cols()), (left.rows(), left.cols()))?;

    let capacity = if matches!(arithmetic, Arithmetic::Multiply) {
        left.nnz().min(right.nnz())
    } else {
        checked_add(
            left.nnz(),
            right.nnz(),
            scioxide_core::quantity!("combined_entries"),
        )?
    };
    let mut builder = SparseBuilder::with_capacity(left.rows(), left.cols(), capacity)?;
    let mut a = left.iter().peekable();
    let mut b = right.iter().peekable();
    while a.peek().is_some() || b.peek().is_some() {
        let ak = a.peek().map(|&(r, c, _)| (r, c));
        let bk = b.peek().map(|&(r, c, _)| (r, c));
        let key = match (ak, bk) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => break,
        };
        let av = if ak == Some(key) {
            a.next().map(|(_, _, v)| v)
        } else {
            None
        };
        let bv = if bk == Some(key) {
            b.next().map(|(_, _, v)| v)
        } else {
            None
        };
        let value = match arithmetic {
            Arithmetic::Multiply => {
                let (Some(a), Some(b)) = (av, bv) else {
                    continue;
                };
                product(a, b, options.accuracy)?
            }
            Arithmetic::Add | Arithmetic::Subtract => sum_products(
                av.map(|a| (a, S::ONE)).into_iter().chain(bv.map(|b| {
                    (
                        b,
                        if matches!(arithmetic, Arithmetic::Subtract) {
                            -S::ONE
                        } else {
                            S::ONE
                        },
                    )
                })),
                options.accuracy,
            )?,
        };
        builder.push_finite(key.0, key.1, value)?;
    }
    builder.finish_csr(DuplicatePolicy::Sum, options.zeros)
}
impl<S: NumericScalar> CsrMatrix<S> {
    /// Returns `A + B`, where `A` is `self` and `B` is `other`.
    ///
    /// The result has an entry at every coordinate stored in `A` or `B`, so its
    /// pattern is the union of the two patterns whatever the values: an exact
    /// cancellation stays as a stored zero. Each entry is computed with
    /// [`Accuracy::Standard`]. The rows are merged in `O(nnz(A) + nnz(B))`.
    /// This is [`add_with_options`](Self::add_with_options) with
    /// [`SparseArithmeticOptions::new`].
    ///
    /// ```
    /// use scioxide_core::SciOxideErrorKind;
    /// use scioxide_sparse::CsrMatrix;
    ///
    /// let a = CsrMatrix::from_raw(1, 3, vec![0, 2], vec![0, 1], vec![1.0_f64, 2.0])?;
    /// let b = CsrMatrix::from_raw(1, 3, vec![0, 2], vec![1, 2], vec![-2.0, 5.0])?;
    /// // The cancelled coordinate (0, 1) stays as a stored zero.
    /// assert_eq!(a.add(&b)?.values(), &[1.0, 0.0, 5.0]);
    /// assert_eq!(a.subtract(&b)?.values(), &[1.0, 4.0, -5.0]);
    /// assert_eq!(a.hadamard(&b)?.values(), &[-4.0]);
    /// let c = CsrMatrix::from_raw(3, 1, vec![0, 1, 1, 1], vec![0], vec![1.0])?;
    /// assert_eq!(
    ///     a.add(&c).unwrap_err().kind(),
    ///     SciOxideErrorKind::DimensionMismatch
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the matrices have different shapes.
    /// - `NumericalInstability` if a result overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the result cannot be
    ///   allocated.
    pub fn add(&self, other: &Self) -> Result<Self, SciOxideError> {
        self.add_with_options(other, SparseArithmeticOptions::new())
    }
    /// Returns `A + B` with the accuracy and zero policy of `options`.
    ///
    /// The zero policy decides whether exact cancellations stay in the
    /// pattern; see [`add`](Self::add) for the method.
    ///
    /// ```
    /// use scioxide_sparse::{CsrMatrix, ExplicitZeroPolicy, SparseArithmeticOptions};
    ///
    /// let options = SparseArithmeticOptions::new().with_zero_policy(ExplicitZeroPolicy::Drop);
    /// let a = CsrMatrix::from_raw(1, 3, vec![0, 2], vec![0, 1], vec![1.0_f64, 2.0])?;
    /// let b = CsrMatrix::from_raw(1, 3, vec![0, 2], vec![1, 2], vec![-2.0, 5.0])?;
    /// let sum = a.add_with_options(&b, options)?;
    /// assert_eq!(sum.iter().collect::<Vec<_>>(), [(0, 0, 1.0), (0, 2, 5.0)]);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - The errors of [`add`](Self::add).
    pub fn add_with_options(
        &self,
        other: &Self,
        options: SparseArithmeticOptions,
    ) -> Result<Self, SciOxideError> {
        combine(self, other, options, Arithmetic::Add)
    }
    /// Returns `A − B` on the union of the two patterns, like
    /// [`add`](Self::add), which has an example.
    ///
    /// # Errors
    ///
    /// - The errors of [`add`](Self::add).
    pub fn subtract(&self, other: &Self) -> Result<Self, SciOxideError> {
        self.subtract_with_options(other, SparseArithmeticOptions::new())
    }
    /// Returns `A − B` with the accuracy and zero policy of `options`, like
    /// [`add_with_options`](Self::add_with_options).
    ///
    /// # Errors
    ///
    /// - The errors of [`add`](Self::add).
    pub fn subtract_with_options(
        &self,
        other: &Self,
        options: SparseArithmeticOptions,
    ) -> Result<Self, SciOxideError> {
        combine(self, other, options, Arithmetic::Subtract)
    }
    /// Returns the Hadamard (entrywise) product `A ∘ B`.
    ///
    /// The result has an entry at every coordinate stored in both `A` and `B`,
    /// a zero product included. See [`add`](Self::add) for an example.
    ///
    /// # Errors
    ///
    /// - The errors of [`add`](Self::add).
    pub fn hadamard(&self, other: &Self) -> Result<Self, SciOxideError> {
        self.hadamard_with_options(other, SparseArithmeticOptions::new())
    }
    /// Returns `A ∘ B` with the accuracy and zero policy of `options`, like
    /// [`add_with_options`](Self::add_with_options).
    ///
    /// # Errors
    ///
    /// - The errors of [`add`](Self::add).
    pub fn hadamard_with_options(
        &self,
        other: &Self,
        options: SparseArithmeticOptions,
    ) -> Result<Self, SciOxideError> {
        combine(self, other, options, Arithmetic::Multiply)
    }
}
impl<S: NumericScalar> BsrMatrix<S> {
    /// Returns a copy without the values that `policy` drops.
    ///
    /// A block is removed only when all its values are dropped. The other
    /// blocks stay complete, with dropped values set to zero.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the threshold overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the result cannot be
    ///   allocated.
    pub fn pruned(&self, policy: SparsePruning<S::Real>) -> Result<Self, SciOxideError> {
        let csr = self.view().pruned(policy)?;
        let (br, bc) = self.block_shape();
        Self::from_sparse(csr.view(), br, bc)
    }
}
impl<S: NumericScalar> DiaMatrix<S> {
    /// Returns a copy without the values that `policy` drops.
    ///
    /// A diagonal is removed only when all its values are dropped. The other
    /// diagonals stay complete, with dropped values set to zero.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the threshold overflows.
    /// - `SizeOverflow` or `AllocationFailure` if the result cannot be
    ///   allocated.
    pub fn pruned(&self, policy: SparsePruning<S::Real>) -> Result<Self, SciOxideError> {
        Self::from_sparse(self.view().pruned(policy)?.view())
    }
}

/// Creates a `rows × cols` [`CsrMatrix`] from its diagonals.
///
/// Diagonal `k` has offset `d = offsets[k] = column − row` and holds exactly
/// the values inside the matrix, starting at `(0, d)` for `d ≥ 0` or at
/// `(−d, 0)` for `d < 0`. Every given value is stored, zeros included, and the
/// values of a repeated offset are added exactly and rounded once: the policy
/// `(Sum, Preserve)` of [`CsrMatrix::from_triplets`]. Remove zeros afterwards
/// with [`SparseView::to_csr`] and [`ExplicitZeroPolicy::Drop`].
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::sparse_diagonals;
///
/// // Main diagonal (length 3) and first superdiagonal (length 2) of a 3 x 3 matrix.
/// let a = sparse_diagonals(3, 3, &[0, 1], &[&[2.0_f64, 2.0, 2.0], &[-1.0, -1.0]])?;
/// assert_eq!((a.nnz(), a.get(1, 2)?, a.get(2, 1)?), (5, -1.0, 0.0));
/// let short = sparse_diagonals(3, 3, &[1], &[&[-1.0_f64]]);
/// assert_eq!(
///     short.unwrap_err().kind(),
///     SciOxideErrorKind::DimensionMismatch
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if `offsets` and `diagonals` have different lengths,
///   or a diagonal does not have the number of values that fit in the matrix.
/// - `InvalidInput` if a diagonal lies outside the matrix.
/// - `NonFinite` if a value is NaN or infinite.
/// - `NumericalInstability` if a sum of repeated diagonals overflows.
/// - `SizeOverflow` or `AllocationFailure` if the result cannot be allocated.
pub fn sparse_diagonals<S: NumericScalar>(
    rows: usize,
    cols: usize,
    offsets: &[isize],
    diagonals: &[&[S]],
) -> Result<CsrMatrix<S>, SciOxideError> {
    crate::helpers::exact(diagonals.len(), offsets.len())?;

    let mut builder = SparseBuilder::new(rows, cols);
    for (&offset, &values) in offsets.iter().zip(diagonals) {
        let (row, col) = if offset >= 0 {
            (0, offset as usize)
        } else {
            (offset.unsigned_abs(), 0)
        };
        if row >= rows || col >= cols {
            return Err(invalid(
                scioxide_core::argument!("offsets"),
                "diagonal does not intersect shape",
            ));
        }
        crate::helpers::exact(values.len(), (rows - row).min(cols - col))?;

        for (i, &v) in values.iter().enumerate() {
            builder.push(row + i, col + i, v)?;
        }
    }
    builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
}
/// Creates a [`CsrMatrix`] from a grid of sparse blocks.
///
/// The grid has `row_sizes.len()` block rows and `col_sizes.len()` block
/// columns, and `blocks` lists its cells in row-major order. Block `(i, j)`
/// must be `row_sizes[i] × col_sizes[j]`; `None` is a zero block. The blocks
/// may have any storage format, and every stored entry of every block is
/// stored in the result, explicit zeros included.
///
/// ```
/// use scioxide_core::SciOxideErrorKind;
/// use scioxide_sparse::{CsrMatrix, sparse_block_grid};
///
/// let one = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![1.0_f64])?;
/// let row = CsrMatrix::from_raw(1, 2, vec![0, 2], vec![0, 1], vec![2.0, 3.0])?;
/// // [[one, row], [0, 0]] with row sizes [1, 1] and column sizes [1, 2].
/// let blocks = [Some(one.view()), Some(row.view()), None, None];
/// let a = sparse_block_grid(&[1, 1], &[1, 2], &blocks)?;
/// assert_eq!((a.rows(), a.cols()), (2, 3));
/// assert_eq!(
///     a.iter().collect::<Vec<_>>(),
///     [(0, 0, 1.0), (0, 1, 2.0), (0, 2, 3.0)]
/// );
/// let swapped = sparse_block_grid(&[1, 1], &[2, 1], &blocks);
/// assert_eq!(
///     swapped.unwrap_err().kind(),
///     SciOxideErrorKind::DimensionMismatch
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `DimensionMismatch` if `blocks` does not have one cell per grid
///   position, or a block does not have the size its row and column give.
/// - `SizeOverflow` if the total dimensions overflow `usize`.
/// - `AllocationFailure` if the result cannot be allocated.
pub fn sparse_block_grid<S: NumericScalar>(
    row_sizes: &[usize],
    col_sizes: &[usize],
    blocks: &[Option<SparseView<'_, S>>],
) -> Result<CsrMatrix<S>, SciOxideError> {
    crate::helpers::exact(
        blocks.len(),
        checked_mul(
            row_sizes.len(),
            col_sizes.len(),
            scioxide_core::quantity!("block_grid_cells"),
        )?,
    )?;

    let offsets = |sizes: &[usize]| -> Result<alloc::vec::Vec<usize>, SciOxideError> {
        let mut result = try_with_capacity(
            checked_add(
                sizes.len(),
                1,
                scioxide_core::quantity!("block_grid_offsets"),
            )?,
            scioxide_core::quantity!("block_grid_offsets"),
        )?;
        result.push(0);
        let mut sum = 0;
        for &size in sizes {
            sum = checked_add(sum, size, scioxide_core::quantity!("block_grid_extent"))?;
            result.push(sum);
        }
        Ok(result)
    };
    let rows = offsets(row_sizes)?;
    let cols = offsets(col_sizes)?;
    let mut builder = SparseBuilder::new(rows[row_sizes.len()], cols[col_sizes.len()]);
    for (index, block) in blocks.iter().enumerate() {
        if let Some(block) = block {
            let r = index / col_sizes.len();
            let c = index % col_sizes.len();
            crate::helpers::shape((block.rows(), block.cols()), (row_sizes[r], col_sizes[c]))?;

            for (i, j, v) in block.iter() {
                builder.push_finite(rows[r] + i, cols[c] + j, v)?;
            }
        }
    }
    builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
}
/// Creates the block-diagonal [`CsrMatrix`] `diag(B₁, B₂, …)` from the
/// given blocks.
///
/// Blocks may be rectangular or empty, and of any storage format. The block
/// `Bₖ` starts at the row and column that follow the previous block. Every
/// stored entry of every block is stored in the result, explicit zeros
/// included.
///
/// ```
/// use scioxide_sparse::{CsrMatrix, sparse_block_diagonal};
///
/// let a = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![2.0_f64])?;
/// let b = CsrMatrix::from_raw(2, 1, vec![0, 1, 2], vec![0, 0], vec![3.0, 4.0])?;
/// let d = sparse_block_diagonal(&[a.view(), b.view()])?;
/// assert_eq!((d.rows(), d.cols()), (3, 2));
/// assert_eq!(
///     d.iter().collect::<Vec<_>>(),
///     [(0, 0, 2.0), (1, 1, 3.0), (2, 1, 4.0)]
/// );
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `SizeOverflow` if the total dimensions overflow `usize`.
/// - `AllocationFailure` if the result cannot be allocated.
pub fn sparse_block_diagonal<S: NumericScalar>(
    blocks: &[SparseView<'_, S>],
) -> Result<CsrMatrix<S>, SciOxideError> {
    let (rows, cols) = blocks.iter().try_fold((0, 0), |(r, c), b| {
        Ok::<_, SciOxideError>((
            checked_add(r, b.rows(), scioxide_core::quantity!("block_diagonal_rows"))?,
            checked_add(
                c,
                b.cols(),
                scioxide_core::quantity!("block_diagonal_columns"),
            )?,
        ))
    })?;
    let mut builder = SparseBuilder::new(rows, cols);
    let (mut row, mut col) = (0, 0);
    for block in blocks {
        for (r, c, v) in block.iter() {
            builder.push_finite(row + r, col + c, v)?;
        }
        row += block.rows();
        col += block.cols();
    }
    builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
}
