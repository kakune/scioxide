//! Band, Hermitian band and packed triangular storage.
use super::direct::size;
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::Triangle;
use scioxide_core::{
    AdjointOperator, Argument, BlockOperator, Dimensions, LinearOperator, NumericScalar,
    SciOxideError,
};

/// Meaning of the triangle that a [`PackedMatrix`] does not store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedKind {
    /// Triangular matrix: the unstored triangle is zero.
    Triangular,
    /// Hermitian matrix: `a[j][i] = conj(a[i][j])`, with a real diagonal.
    ///
    /// For real scalars the matrix is symmetric.
    Hermitian,
}

fn bandwidth(n: usize, k: usize, argument: Argument) -> Result<(), SciOxideError> {
    if (n == 0 && k != 0) || (n != 0 && k >= n) {
        return Err(crate::checked::out_of_range(
            argument,
            "bandwidth exceeds order",
        ));
    }
    Ok(())
}
fn length(actual: usize, expected: usize) -> Result<(), SciOxideError> {
    if actual != expected {
        return Err(SciOxideError::dimension_mismatch([expected], [actual]));
    }
    Ok(())
}
fn bounds(n: usize, i: usize, j: usize) -> Result<(), SciOxideError> {
    if i >= n {
        return Err(SciOxideError::out_of_bounds(
            scioxide_core::quantity!("matrix_row"),
            i,
            n,
        ));
    }
    if j >= n {
        return Err(SciOxideError::out_of_bounds(
            scioxide_core::quantity!("matrix_column"),
            j,
            n,
        ));
    }
    Ok(())
}
fn diagonal<S: NumericScalar>(value: S, argument: Argument) -> Result<(), SciOxideError> {
    if value.imaginary_component() != S::Real::ZERO {
        return Err(crate::checked::out_of_range(
            argument,
            "Hermitian diagonal must be real",
        ));
    }
    Ok(())
}

/// General `m × n` band matrix in column-major band storage.
///
/// A band matrix with `lower` subdiagonals and `upper` superdiagonals has
/// `a[i][j] = 0` for `i > j + lower` and for `j > i + upper`. Each column `j`
/// occupies `leading_dimension ≥ lower + upper + 1` consecutive elements of
/// `data`, and a band entry is stored at
///
/// ```text
/// a[i][j] = data[j · leading_dimension + upper + i − j],   j − upper ≤ i ≤ j + lower
/// ```
///
/// The diagonal entry of column `j` is at position `upper` of that column.
/// Positions that fall outside the matrix, and the last
/// `leading_dimension − lower − upper − 1` positions of each column, are
/// padding and are never read. A matrix with zero rows or columns has empty
/// storage and `lower = upper = leading_dimension = 0`. Construction checks the
/// layout, not the values.
///
/// [`BandedLuDecomposition`](crate::BandedLuDecomposition),
/// [`BandedCholeskyDecomposition`](crate::BandedCholeskyDecomposition) and
/// [`TriangularPlan`](crate::TriangularPlan) require a square matrix.
///
/// ```
/// use scioxide_linalg::{BandedMatrix, StructuredOperation, Vector};
///
/// // A = [1 2 0; 3 4 5] with one subdiagonal and one superdiagonal. Column j
/// // stores rows j − 1, j and j + 1; positions outside A hold padding (0.0).
/// let a = BandedMatrix::new(2, 3, 1, 1, 3, vec![0.0, 1.0, 3.0, 2.0, 4.0, 0.0, 5.0, 0.0, 0.0])?;
/// let y = a.view().mul_vec(
///     StructuredOperation::Normal,
///     &Vector::from_vec(vec![1.0, 2.0, 3.0]))?;
/// // A [1, 2, 3]ᵀ = [5, 26]ᵀ, exact in floating point.
/// assert_eq!(y.as_slice(), &[5.0, 26.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct BandedMatrix<S: NumericScalar> {
    rows: usize,
    cols: usize,
    lower: usize,
    upper: usize,
    leading_dimension: usize,
    data: Vec<S>,
}

/// Borrowed general band matrix with the storage layout of [`BandedMatrix`].
///
/// The view implements [`LinearOperator`] and [`AdjointOperator`].
#[derive(Debug, Clone, Copy)]
pub struct BandedMatrixView<'a, S: NumericScalar> {
    rows: usize,
    cols: usize,
    lower: usize,
    upper: usize,
    leading_dimension: usize,
    data: &'a [S],
}

impl<S: NumericScalar> BandedMatrix<S> {
    /// Creates a band matrix from column-major band storage `data`.
    ///
    /// # Errors
    /// - `InvalidInput` if `lower ≥ rows`, `upper ≥ cols`, or
    ///   `leading_dimension < lower + upper + 1` for a nonempty matrix, or if any
    ///   of `lower`, `upper`, `leading_dimension` is nonzero for an empty matrix.
    /// - `SizeOverflow` if `lower + upper + 1` or `cols · leading_dimension`
    ///   overflows `usize`.
    /// - `DimensionMismatch` if `data.len()` is not `cols · leading_dimension`
    ///   (zero for an empty matrix).
    pub fn new(
        rows: usize,
        cols: usize,
        lower: usize,
        upper: usize,
        leading_dimension: usize,
        data: Vec<S>,
    ) -> Result<Self, SciOxideError> {
        validate_layout(rows, cols, lower, upper, leading_dimension, data.len())?;
        Ok(Self {
            rows,
            cols,
            lower,
            upper,
            leading_dimension,
            data,
        })
    }
    /// Returns the number of rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }
    /// Returns the number of columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }
    /// Returns the number of storage elements per column.
    pub const fn leading_dimension(&self) -> usize {
        self.leading_dimension
    }
    /// Borrows the matrix as a [`BandedMatrixView`].
    pub fn view(&self) -> BandedMatrixView<'_, S> {
        BandedMatrixView {
            rows: self.rows,
            cols: self.cols,
            lower: self.lower,
            upper: self.upper,
            leading_dimension: self.leading_dimension,
            data: &self.data,
        }
    }
    /// Returns the column-major band storage, including padding.
    pub fn as_slice(&self) -> &[S] {
        &self.data
    }
}

impl<'a, S: NumericScalar> BandedMatrixView<'a, S> {
    /// Creates a view of column-major band storage `data`.
    ///
    /// The layout is described on [`BandedMatrix`].
    ///
    /// # Errors
    /// The same conditions as [`BandedMatrix::new`].
    pub fn new(
        rows: usize,
        cols: usize,
        lower: usize,
        upper: usize,
        leading_dimension: usize,
        data: &'a [S],
    ) -> Result<Self, SciOxideError> {
        validate_layout(rows, cols, lower, upper, leading_dimension, data.len())?;
        Ok(Self {
            rows,
            cols,
            lower,
            upper,
            leading_dimension,
            data,
        })
    }
    /// Returns the number of rows.
    pub const fn rows(self) -> usize {
        self.rows
    }
    /// Returns the number of columns.
    pub const fn cols(self) -> usize {
        self.cols
    }
    /// Returns the number of subdiagonals.
    pub const fn lower(self) -> usize {
        self.lower
    }
    /// Returns the number of superdiagonals.
    pub const fn upper(self) -> usize {
        self.upper
    }
    /// Returns the number of storage elements per column.
    pub const fn leading_dimension(self) -> usize {
        self.leading_dimension
    }
    /// Returns the column-major band storage, including padding.
    pub const fn as_slice(self) -> &'a [S] {
        self.data
    }
    /// Returns entry `(row, col)`, which is zero outside the band.
    ///
    /// # Errors
    /// - `OutOfBounds` if `row ≥ rows` or `col ≥ cols`.
    pub fn get(self, row: usize, col: usize) -> Result<S, SciOxideError> {
        if row >= self.rows {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("band_row"),
                row,
                self.rows,
            ));
        }
        if col >= self.cols {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("band_column"),
                col,
                self.cols,
            ));
        }
        Ok(self.element(row, col))
    }
    /// Stored entries of column `col` in ascending row order: rows
    /// `col - upper ..= col + lower`, clipped to the matrix.
    pub(crate) fn column(self, col: usize) -> &'a [S] {
        let first = col.saturating_sub(self.upper);
        let end = col
            .saturating_add(self.lower)
            .saturating_add(1)
            .min(self.rows);
        let start = col * self.leading_dimension + self.upper + first - col;
        &self.data[start..start + end.saturating_sub(first)]
    }
    pub(crate) fn element(self, row: usize, col: usize) -> S {
        if row > col && row - col > self.lower || col > row && col - row > self.upper {
            S::ZERO
        } else {
            let offset = if row >= col {
                self.upper + (row - col)
            } else {
                self.upper - (col - row)
            };
            self.data[col * self.leading_dimension + offset]
        }
    }
}

impl<'a, S: NumericScalar> From<&'a BandedMatrix<S>> for BandedMatrixView<'a, S> {
    fn from(matrix: &'a BandedMatrix<S>) -> Self {
        matrix.view()
    }
}

fn validate_layout(
    rows: usize,
    cols: usize,
    lower: usize,
    upper: usize,
    leading_dimension: usize,
    data_len: usize,
) -> Result<(), SciOxideError> {
    let empty = rows == 0 || cols == 0;
    if empty && (lower != 0 || upper != 0 || leading_dimension != 0)
        || !empty && (lower >= rows || upper >= cols)
    {
        let argument = if empty && lower == 0 && upper == 0 {
            scioxide_core::argument!("leading_dimension")
        } else if (empty && lower != 0) || (!empty && lower >= rows) {
            scioxide_core::argument!("lower")
        } else {
            scioxide_core::argument!("upper")
        };
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::Inconsistent,
            "bandwidth exceeds matrix extent",
        ));
    }
    let minimum = lower
        .checked_add(upper)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("band_leading_dimension"))
        })?;
    if !empty && leading_dimension < minimum {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("leading_dimension"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "leading dimension is smaller than the stored band",
        ));
    }
    let expected = if empty {
        0
    } else {
        cols.checked_mul(leading_dimension)
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("band_storage")))?
    };
    if data_len != expected {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(expected),
            Dimensions::one(data_len),
        ));
    }
    Ok(())
}

/// Hermitian band matrix that stores one triangle of its band.
///
/// An `n × n` Hermitian matrix with bandwidth `k` has `a[i][j] = 0` for
/// `|i − j| > k`. The selected triangle is stored column by column with `k + 1`
/// elements per column, `n (k + 1)` elements in total:
///
/// ```text
/// Triangle::Lower:  a[i][j] = data[j (k + 1) + i − j],       j ≤ i ≤ j + k
/// Triangle::Upper:  a[i][j] = data[j (k + 1) + k + i − j],   j − k ≤ i ≤ j
/// ```
///
/// In lower storage the diagonal is the first element of each column; in upper
/// storage it is the last. The other triangle is `a[j][i] = conj(a[i][j])`.
/// Positions that fall outside the matrix are padding and are never read.
/// Diagonal entries must have zero imaginary part. For real scalars this is a
/// symmetric band matrix. Construction checks the layout and the diagonal;
/// numerical operations check the entries they use for finiteness.
///
/// [`HermitianBandMatrixView`] implements [`LinearOperator`] and
/// [`AdjointOperator`]. [`BandedCholeskyDecomposition`](crate::BandedCholeskyDecomposition)
/// factors positive definite matrices in this storage.
#[derive(Debug, Clone)]
pub struct HermitianBandMatrix<S: NumericScalar> {
    order: usize,
    bandwidth: usize,
    triangle: Triangle,
    data: Vec<S>,
}
/// Borrowed Hermitian band matrix with the storage layout of [`HermitianBandMatrix`].
#[derive(Debug, Clone, Copy)]
pub struct HermitianBandMatrixView<'a, S: NumericScalar> {
    order: usize,
    bandwidth: usize,
    triangle: Triangle,
    data: &'a [S],
}
impl<'a, S: NumericScalar> From<&'a HermitianBandMatrix<S>> for HermitianBandMatrixView<'a, S> {
    fn from(matrix: &'a HermitianBandMatrix<S>) -> Self {
        matrix.view()
    }
}
impl<S: NumericScalar> HermitianBandMatrix<S> {
    /// Creates a Hermitian band matrix from one stored triangle.
    ///
    /// `order` is `n`, `bandwidth_value` is `k`, and `data` holds `triangle`
    /// in the layout described on [`HermitianBandMatrix`].
    ///
    /// # Errors
    /// The same conditions as [`HermitianBandMatrixView::new`].
    pub fn new(
        order: usize,
        bandwidth_value: usize,
        triangle: Triangle,
        data: Vec<S>,
    ) -> Result<Self, SciOxideError> {
        HermitianBandMatrixView::new(order, bandwidth_value, triangle, &data)?;
        Ok(Self {
            order,
            bandwidth: bandwidth_value,
            triangle,
            data,
        })
    }
    /// Borrows the matrix as a [`HermitianBandMatrixView`].
    pub fn view(&self) -> HermitianBandMatrixView<'_, S> {
        HermitianBandMatrixView {
            order: self.order,
            bandwidth: self.bandwidth,
            triangle: self.triangle,
            data: &self.data,
        }
    }
    /// Returns the band storage, including padding.
    pub fn as_slice(&self) -> &[S] {
        &self.data
    }
    /// Returns the order `n`.
    pub fn order(&self) -> usize {
        self.order
    }
}
impl<'a, S: NumericScalar> HermitianBandMatrixView<'a, S> {
    /// Creates a view of the `triangle` of a Hermitian band matrix stored in `data`.
    ///
    /// The layout is described on [`HermitianBandMatrix`].
    ///
    /// # Errors
    /// - `InvalidInput` if `bandwidth_value ≥ order` (or nonzero for `order = 0`),
    ///   or if a diagonal entry has a nonzero imaginary part.
    /// - `SizeOverflow` if `order · (bandwidth_value + 1)` overflows `usize`.
    /// - `DimensionMismatch` if `data.len()` is not `order · (bandwidth_value + 1)`.
    pub fn new(
        order: usize,
        bandwidth_value: usize,
        triangle: Triangle,
        data: &'a [S],
    ) -> Result<Self, SciOxideError> {
        bandwidth(
            order,
            bandwidth_value,
            scioxide_core::argument!("bandwidth_value"),
        )?;
        length(
            data.len(),
            size(
                order,
                bandwidth_value + 1,
                scioxide_core::quantity!("band_storage"),
            )?,
        )?;
        let result = Self {
            order,
            bandwidth: bandwidth_value,
            triangle,
            data,
        };
        for i in 0..order {
            diagonal(result.element(i, i), scioxide_core::argument!("data"))?;
        }
        Ok(result)
    }
    /// Returns the order `n`.
    pub fn order(self) -> usize {
        self.order
    }
    /// Returns the bandwidth `k`.
    pub fn bandwidth(self) -> usize {
        self.bandwidth
    }
    /// Returns the stored triangle.
    pub fn triangle(self) -> Triangle {
        self.triangle
    }
    /// Returns the band storage, including padding.
    pub fn as_slice(self) -> &'a [S] {
        self.data
    }
    /// Returns entry `(i, j)`: zero outside the band, `conj(a[j][i])` in the
    /// unstored triangle.
    ///
    /// # Errors
    /// - `OutOfBounds` if `i ≥ n` or `j ≥ n`.
    pub fn get(self, i: usize, j: usize) -> Result<S, SciOxideError> {
        bounds(self.order, i, j)?;
        Ok(self.element(i, j))
    }
    /// Stored entries of column `col` in ascending row order: rows
    /// `col ..= col + bandwidth` of a lower triangle or `col - bandwidth ..= col`
    /// of an upper one, clipped to the matrix.
    pub(crate) fn stored_column(self, col: usize) -> &'a [S] {
        let width = self.bandwidth + 1;
        let (start, len) = match self.triangle {
            Triangle::Lower => (col * width, (self.order - col).min(width)),
            Triangle::Upper => {
                let len = col.min(self.bandwidth) + 1;
                (col * width + width - len, len)
            }
        };
        &self.data[start..start + len]
    }
    /// Interpreted entries of column `col` in ascending row order over rows
    /// `col - bandwidth ..= col + bandwidth`, clipped to the matrix.
    pub(crate) fn column(self, col: usize) -> impl Iterator<Item = S> + Clone + 'a {
        let [above, diagonal, below] = self.segments(col);
        above.chain(diagonal).chain(below)
    }
    /// Interpreted entries of rows `col ..= col + bandwidth` of column `col`,
    /// clipped to the matrix.
    pub(crate) fn lower_column(self, col: usize) -> impl Iterator<Item = S> + Clone + 'a {
        let [_, diagonal, below] = self.segments(col);
        diagonal.chain(below)
    }
    /// Rows above, on and below the diagonal of column `col`. The reflected
    /// triangle is a conjugated stored row, `bandwidth` scalars apart.
    fn segments(self, col: usize) -> [Strided<'a, S>; 3] {
        let k = self.bandwidth;
        let (above, below) = (col.min(k), (self.order - col - 1).min(k));
        let diagonal = col * (k + 1)
            + match self.triangle {
                Triangle::Lower => 0,
                Triangle::Upper => k,
            };
        match self.triangle {
            // Row `p < col` holds `conj(a[col, p])`, slot `col - p` of column `p`.
            Triangle::Lower => [
                Strided::new(self.data, (col - above) * k + col, k, above, true),
                Strided::new(self.data, diagonal, 1, 1, false),
                Strided::new(self.data, diagonal + 1, 1, below, false),
            ],
            // Row `i > col` holds `conj(a[col, i])`, slot `k + col - i` of column `i`.
            Triangle::Upper => [
                Strided::new(self.data, diagonal - above, 1, above, false),
                Strided::new(self.data, diagonal, 1, 1, false),
                Strided::new(self.data, diagonal + k, k, below, true),
            ],
        }
    }
    pub(crate) fn element(self, i: usize, j: usize) -> S {
        // Only one triangle is stored; an entry from the other triangle is
        // read from its mirror position and conjugated (`a_ij = conj(a_ji)`).
        let (r, c, conjugate) = match self.triangle {
            Triangle::Lower if i < j => (j, i, true),
            Triangle::Upper if i > j => (j, i, true),
            _ => (i, j, false),
        };
        if r.abs_diff(c) > self.bandwidth {
            return S::ZERO;
        }
        let offset = match self.triangle {
            Triangle::Lower => r - c,
            Triangle::Upper => self.bandwidth + r - c,
        };
        let value = self.data[c * (self.bandwidth + 1) + offset];
        if conjugate { value.conj() } else { value }
    }
}

/// `remaining` entries `data[next]`, `data[next + stride]`, ..., conjugated
/// when `conjugate`.
#[derive(Debug, Clone)]
struct Strided<'a, S> {
    data: &'a [S],
    next: usize,
    stride: usize,
    remaining: usize,
    conjugate: bool,
}

impl<'a, S> Strided<'a, S> {
    const fn new(
        data: &'a [S],
        next: usize,
        stride: usize,
        remaining: usize,
        conjugate: bool,
    ) -> Self {
        Self {
            data,
            next,
            stride,
            remaining,
            conjugate,
        }
    }
}

impl<S: NumericScalar> Iterator for Strided<'_, S> {
    type Item = S;
    fn next(&mut self) -> Option<S> {
        (self.remaining > 0).then(|| {
            let value = self.data[self.next];
            self.next += self.stride;
            self.remaining -= 1;
            if self.conjugate { value.conj() } else { value }
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

/// Triangular or Hermitian matrix in packed column-major storage.
///
/// One triangle of an `n × n` matrix is stored column by column without gaps,
/// `n (n + 1) / 2` elements in total:
///
/// ```text
/// Triangle::Upper:  a[i][j] = data[j (j + 1) / 2 + i],                    i ≤ j
/// Triangle::Lower:  a[i][j] = data[len − (n − j)(n − j + 1) / 2 + i − j], i ≥ j
/// ```
///
/// where `len = n (n + 1) / 2`. Upper storage holds `a[0][0], a[0][1], a[1][1],
/// a[0][2], …`; lower storage holds `a[0][0], a[1][0], …, a[n−1][0], a[1][1], …`.
/// [`PackedKind`] says whether the other triangle is zero or the conjugate
/// transpose. Construction checks the length and, for Hermitian matrices, that
/// the diagonal is real.
///
/// [`PackedMatrixView`] implements [`LinearOperator`] and [`AdjointOperator`].
#[derive(Debug, Clone)]
pub struct PackedMatrix<S: NumericScalar> {
    order: usize,
    triangle: Triangle,
    kind: PackedKind,
    data: Vec<S>,
}
/// Borrowed packed matrix with the storage layout of [`PackedMatrix`].
#[derive(Debug, Clone, Copy)]
pub struct PackedMatrixView<'a, S: NumericScalar> {
    order: usize,
    triangle: Triangle,
    kind: PackedKind,
    data: &'a [S],
}
/// Mutable borrowed packed matrix with the storage layout of [`PackedMatrix`].
#[derive(Debug)]
pub struct PackedMatrixViewMut<'a, S: NumericScalar> {
    order: usize,
    triangle: Triangle,
    kind: PackedKind,
    data: &'a mut [S],
}
impl<S: NumericScalar> PackedMatrix<S> {
    /// Creates a packed matrix of order `order` from the `triangle` stored in `data`.
    ///
    /// # Errors
    /// The same conditions as [`PackedMatrixView::new`].
    pub fn new(
        order: usize,
        triangle: Triangle,
        kind: PackedKind,
        data: Vec<S>,
    ) -> Result<Self, SciOxideError> {
        validate_packed(order, triangle, kind, &data)?;
        Ok(Self {
            order,
            triangle,
            kind,
            data,
        })
    }
    /// Borrows the matrix as a [`PackedMatrixView`].
    pub fn view(&self) -> PackedMatrixView<'_, S> {
        PackedMatrixView {
            order: self.order,
            triangle: self.triangle,
            kind: self.kind,
            data: &self.data,
        }
    }
    /// Borrows the matrix as a [`PackedMatrixViewMut`].
    pub fn view_mut(&mut self) -> PackedMatrixViewMut<'_, S> {
        PackedMatrixViewMut {
            order: self.order,
            triangle: self.triangle,
            kind: self.kind,
            data: &mut self.data,
        }
    }
    /// Returns the packed storage.
    pub fn as_slice(&self) -> &[S] {
        &self.data
    }
    /// Returns the order `n`.
    pub fn order(&self) -> usize {
        self.order
    }
}
impl<'a, S: NumericScalar> PackedMatrixViewMut<'a, S> {
    /// Creates a mutable view of the `triangle` of a packed matrix stored in `data`.
    ///
    /// # Errors
    /// The same conditions as [`PackedMatrixView::new`].
    pub fn new(
        order: usize,
        triangle: Triangle,
        kind: PackedKind,
        data: &'a mut [S],
    ) -> Result<Self, SciOxideError> {
        validate_packed(order, triangle, kind, data)?;
        Ok(Self {
            order,
            triangle,
            kind,
            data,
        })
    }
    /// Returns an immutable view of the same matrix.
    pub fn as_view(&self) -> PackedMatrixView<'_, S> {
        PackedMatrixView {
            order: self.order,
            triangle: self.triangle,
            kind: self.kind,
            data: self.data,
        }
    }
    /// Reborrows the view for a shorter lifetime, so that `self` stays usable afterwards.
    pub fn reborrow(&mut self) -> PackedMatrixViewMut<'_, S> {
        PackedMatrixViewMut {
            order: self.order,
            triangle: self.triangle,
            kind: self.kind,
            data: self.data,
        }
    }
    /// Returns the order `n`.
    pub const fn order(&self) -> usize {
        self.order
    }
    /// Returns the stored triangle.
    pub const fn triangle(&self) -> Triangle {
        self.triangle
    }
    /// Returns whether the matrix is triangular or Hermitian.
    pub const fn kind(&self) -> PackedKind {
        self.kind
    }
    /// Returns entry `(row, col)`, which is zero or `conj(a[col][row])` in the
    /// unstored triangle.
    ///
    /// # Errors
    /// - `OutOfBounds` if `row ≥ n` or `col ≥ n`.
    pub fn get(&self, row: usize, col: usize) -> Result<S, SciOxideError> {
        self.as_view().get(row, col)
    }
    /// Sets entry `(row, col)` of the stored triangle to `value`.
    ///
    /// # Errors
    /// - `OutOfBounds` if `row ≥ n` or `col ≥ n`.
    /// - `InvalidInput` if `(row, col)` lies in the unstored triangle, or if the
    ///   matrix is Hermitian, `row = col` and `value` has a nonzero imaginary part.
    /// - `NonFinite` if `value` has a NaN or infinite component.
    pub fn set(&mut self, row: usize, col: usize, value: S) -> Result<(), SciOxideError> {
        bounds(self.order, row, col)?;
        let selected = match self.triangle {
            Triangle::Lower => row >= col,
            Triangle::Upper => row <= col,
        };
        if !selected {
            return Err(crate::checked::out_of_range(
                scioxide_core::argument!("row"),
                "packed write outside stored triangle",
            ));
        }
        crate::checked::validate_input(value, "value")?;
        if row == col && self.kind == PackedKind::Hermitian {
            diagonal(value, scioxide_core::argument!("value"))?;
        }
        self.set_element(row, col, value);
        Ok(())
    }
    pub(crate) fn set_element(&mut self, row: usize, col: usize, value: S) {
        let offset = match self.triangle {
            Triangle::Upper => triangle_len(col) + row,
            Triangle::Lower => self.data.len() - triangle_len(self.order - col) + row - col,
        };
        self.data[offset] = value;
    }
}
impl<'a, S: NumericScalar> PackedMatrixView<'a, S> {
    /// Creates a view of the `triangle` of a packed matrix stored in `data`.
    ///
    /// The layout is described on [`PackedMatrix`].
    ///
    /// # Errors
    /// - `SizeOverflow` if `order (order + 1) / 2` overflows `usize`.
    /// - `DimensionMismatch` if `data.len()` is not `order (order + 1) / 2`.
    /// - `InvalidInput` if `kind` is [`PackedKind::Hermitian`] and a diagonal
    ///   entry has a nonzero imaginary part.
    pub fn new(
        order: usize,
        triangle: Triangle,
        kind: PackedKind,
        data: &'a [S],
    ) -> Result<Self, SciOxideError> {
        validate_packed(order, triangle, kind, data)?;
        Ok(Self {
            order,
            triangle,
            kind,
            data,
        })
    }
    /// Returns the order `n`.
    pub fn order(self) -> usize {
        self.order
    }
    /// Returns the stored triangle.
    pub fn triangle(self) -> Triangle {
        self.triangle
    }
    /// Returns whether the matrix is triangular or Hermitian.
    pub fn kind(self) -> PackedKind {
        self.kind
    }
    /// Returns the packed storage.
    pub fn as_slice(self) -> &'a [S] {
        self.data
    }
    /// Returns entry `(i, j)`, which is zero or `conj(a[j][i])` in the
    /// unstored triangle.
    ///
    /// # Errors
    /// - `OutOfBounds` if `i ≥ n` or `j ≥ n`.
    pub fn get(self, i: usize, j: usize) -> Result<S, SciOxideError> {
        bounds(self.order, i, j)?;
        Ok(self.element(i, j))
    }
    pub(crate) fn element(self, i: usize, j: usize) -> S {
        let outside = match self.triangle {
            Triangle::Lower => i < j,
            Triangle::Upper => i > j,
        };
        if outside && self.kind == PackedKind::Triangular {
            return S::ZERO;
        }
        let (r, c) = if outside { (j, i) } else { (i, j) };
        let offset = match self.triangle {
            Triangle::Upper => triangle_len(c) + r,
            Triangle::Lower => self.data.len() - triangle_len(self.order - c) + r - c,
        };
        let value = self.data[offset];
        if outside { value.conj() } else { value }
    }
}

// Arguments are bounded by the constructor-validated triangular storage length.
fn triangle_len(n: usize) -> usize {
    if n.is_multiple_of(2) {
        (n / 2) * (n + 1)
    } else {
        n * (n / 2 + 1)
    }
}

fn validate_packed<S: NumericScalar>(
    order: usize,
    triangle: Triangle,
    kind: PackedKind,
    data: &[S],
) -> Result<(), SciOxideError> {
    let next = order
        .checked_add(1)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("packed_storage")))?;
    let len = if order.is_multiple_of(2) {
        size(order / 2, next, scioxide_core::quantity!("packed_storage"))?
    } else {
        size(order, next / 2, scioxide_core::quantity!("packed_storage"))?
    };
    length(data.len(), len)?;

    if kind == PackedKind::Hermitian {
        for i in 0..order {
            let offset = match triangle {
                Triangle::Upper => triangle_len(i) + i,
                Triangle::Lower => data.len() - triangle_len(order - i),
            };
            diagonal(data[offset], scioxide_core::argument!("data"))?;
        }
    }
    Ok(())
}

// Operator protocols are implemented once, on the views; owned matrices lend
// a view. Standard products use native sequential dispatch unless a scoped
// caller supplies its resolved dispatch. Workspace holds the output candidate.
macro_rules! view_operator {
    ($view:ident, $rows:ident, $cols:ident, |$matrix:ident, $adjoint:ident| $operation:expr) => {
        impl<S: NumericScalar> LinearOperator<S> for $view<'_, S> {
            fn rows(&self) -> usize {
                self.$rows
            }
            fn cols(&self) -> usize {
                self.$cols
            }
            fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
                Ok(self.$rows)
            }
            fn apply_into(
                &self,
                input: &[S],
                output: &mut [S],
                workspace: &mut [S],
            ) -> Result<(), SciOxideError> {
                let ($matrix, $adjoint) = (*self, false);
                apply_view(
                    input,
                    output,
                    workspace,
                    crate::KernelContext::sequential_native(),
                    $operation,
                )
            }
            fn apply_into_scoped(
                &self,
                input: &[S],
                output: &mut [S],
                workspace: &mut [S],
                context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
            ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
                let ($matrix, $adjoint) = (*self, false);
                apply_view(
                    input,
                    output,
                    workspace,
                    crate::KernelContext::for_operator(context),
                    $operation,
                )
                .map(scioxide_core::Invocation::Evaluated)
            }
        }
        impl<S: NumericScalar> AdjointOperator<S> for $view<'_, S> {
            fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
                Ok(self.$cols)
            }
            fn apply_adjoint_into(
                &self,
                input: &[S],
                output: &mut [S],
                workspace: &mut [S],
            ) -> Result<(), SciOxideError> {
                let ($matrix, $adjoint) = (*self, true);
                apply_view(
                    input,
                    output,
                    workspace,
                    crate::KernelContext::sequential_native(),
                    $operation,
                )
            }
            fn apply_adjoint_into_scoped(
                &self,
                input: &[S],
                output: &mut [S],
                workspace: &mut [S],
                context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
            ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
                let ($matrix, $adjoint) = (*self, true);
                apply_view(
                    input,
                    output,
                    workspace,
                    crate::KernelContext::for_operator(context),
                    $operation,
                )
                .map(scioxide_core::Invocation::Evaluated)
            }
        }
        impl<S: NumericScalar> BlockOperator<S> for $view<'_, S> {}
    };
}

// `multiply(x, y, workspace, options, context)` computes `y := op(A) x`.
fn apply_view<S: NumericScalar>(
    input: &[S],
    output: &mut [S],
    workspace: &mut [S],
    context: crate::KernelContext<'_>,
    multiply: impl FnOnce(
        crate::VectorView<'_, S>,
        crate::VectorViewMut<'_, S>,
        &mut crate::LinalgWorkspace<'_, S>,
        &crate::LinalgExecutionOptions,
        crate::KernelContext<'_>,
    ) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    multiply(
        crate::VectorView::from_slice(input),
        crate::VectorViewMut::from_slice(output),
        &mut crate::LinalgWorkspace::from_scalar_slice(workspace),
        &crate::LinalgExecutionOptions::new(),
        context,
    )
}

fn operation(adjoint: bool) -> crate::StructuredOperation {
    if adjoint {
        crate::StructuredOperation::Adjoint
    } else {
        crate::StructuredOperation::Normal
    }
}

view_operator!(BandedMatrixView, rows, cols, |matrix, adjoint| {
    |x, y, workspace, options, context| {
        matrix.mul_vec_in_place(
            operation(adjoint),
            S::ONE,
            x,
            S::ZERO,
            options,
            y,
            workspace,
            crate::LinalgExecutionContext::unscoped(context),
        )
    }
});
// A Hermitian band is its own adjoint.
view_operator!(HermitianBandMatrixView, order, order, |matrix, _adjoint| {
    |x, y, workspace, options, context| {
        matrix.mul_vec_in_place(
            S::ONE,
            x,
            S::ZERO,
            options,
            y,
            workspace,
            crate::LinalgExecutionContext::unscoped(context),
        )
    }
});
view_operator!(PackedMatrixView, order, order, |matrix, adjoint| {
    |x, y, workspace, options, context| {
        matrix.mul_vec_in_place(
            operation(adjoint),
            S::ONE,
            x,
            S::ZERO,
            options,
            y,
            workspace,
            crate::LinalgExecutionContext::unscoped(context),
        )
    }
});
