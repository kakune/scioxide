//! Structured matrices described by a few parameters and evaluated entry by entry.
use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    AdjointOperator, BlockOperator, Complex, LinearOperator, NumericScalar, RealScalar,
    SciOxideError,
};
fn real_from_usize<R: RealScalar>(n: usize) -> Result<R, SciOxideError> {
    crate::checked::rounded_count::<R>(n)
}
use super::direct::{check_apply, narrow, quotient, size, wide};
use crate::Matrix;

/// Selects a Hilbert or Pascal matrix or its inverse, whose entries have closed forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixForm {
    /// The matrix itself.
    Direct,
    /// The inverse, evaluated from its closed-form entries.
    Inverse,
}

/// Selects the rows of a Helmert matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelmertKind {
    /// All `n` rows, starting with the constant row `1/√n`.
    Full,
    /// The `n − 1` rows after the constant row.
    Reduced,
}

/// Layout of a companion matrix of a polynomial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompanionKind {
    /// Ones on the subdiagonal and `−cᵢ / cₙ` in the last column.
    Frobenius,
    /// Pentadiagonal companion matrix of Fiedler (2003).
    Fiedler,
}

/// Sign of the exponent of a DFT matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DftDirection {
    /// Entries `exp(-2πi jk/n)`.
    Forward,
    /// Entries `exp(+2πi jk/n)`.
    Inverse,
}

/// Scale factor of a DFT matrix of order `n`.
#[derive(Debug, Clone, Copy)]
pub enum DftNormalization {
    /// No scaling.
    None,
    /// Divide every entry by `n`.
    ByOrder,
    /// Divide every entry by `√n`, which makes the matrix unitary.
    Unitary,
}
/// Selects the triangular or symmetric Pascal matrix.
#[derive(Debug, Clone, Copy)]
pub enum PascalKind {
    /// Lower triangular `L[i][j] = C(i, j)`, zero for `j > i`.
    Lower,
    /// Upper triangular `Lᵀ`.
    Upper,
    /// Symmetric `S[i][j] = C(i + j, i)`, which equals `L Lᵀ`.
    Symmetric,
}
/// Selects which outputs of a full linear convolution a convolution matrix returns.
///
/// With kernel length `m` and input length `N`, full convolution has
/// `m + N − 1` outputs.
#[derive(Debug, Clone, Copy)]
pub enum ConvolutionMode {
    /// All `m + N − 1` outputs.
    Full,
    /// The central `max(m, N)` outputs, starting at offset `(min(m, N) − 1) / 2`.
    Same,
    /// The outputs where the shorter sequence lies entirely inside the longer one.
    ///
    /// There are `max(m, N) − min(m, N) + 1` of them.
    Valid,
}
/// Checks `scalars` values of `T` against the byte limit of `options` before
/// allocation. A time limit is `CapabilityUnavailable`, because these owned
/// forms run without a clock. Exceeding the byte limit is `ResourceLimit`.
pub(super) fn admit_owned<T>(
    scalars: usize,
    options: &crate::LinalgExecutionOptions,
    quantity: scioxide_core::Quantity,
) -> Result<(), SciOxideError> {
    let _ =
        crate::LinalgExecutionContext::new(scioxide_core::ExecutionContext::sequential_native())
            .with_options(options)?;
    let bytes = scalars
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| SciOxideError::size_overflow(quantity))?;
    crate::workspace::admit_bytes(bytes, options.byte_limit(), quantity)
}
#[derive(Debug, Clone, Copy)]
pub(super) enum Pattern<'a, T: NumericScalar> {
    Toeplitz(&'a [T], &'a [T]),
    Hankel(&'a [T], &'a [T]),
    Circulant(&'a [T]),
    Hilbert(MatrixForm),
    Pascal(PascalKind, MatrixForm),
    Hadamard,
    Helmert(HelmertKind),
    Leslie(&'a [T], &'a [T]),
    Fiedler(&'a [T]),
    Companion(&'a [T], CompanionKind),
    Convolution(&'a [T], usize),
    Dft(DftDirection, DftNormalization),
}
/// Structured matrix defined by a few parameters, with entries evaluated on demand.
///
/// A view either borrows the defining coefficients (Toeplitz, Hankel,
/// circulant, Leslie, Fiedler, companion and convolution matrices) or stores
/// only the order (Hilbert, Pascal, Hadamard, Helmert and DFT matrices). No
/// matrix entries are stored. [`get`](Self::get) evaluates one entry, and
/// [`materialize`](Self::materialize) builds a dense [`Matrix`]. Indices start
/// at zero. Constructors check that the coefficients are finite.
///
/// The view implements [`LinearOperator`] and [`AdjointOperator`]. A product
/// evaluates every entry once. An entry costs O(1) operations, so a product
/// costs O(m n) for an `m × n` matrix, except for Pascal matrices and the
/// inverse Hilbert matrix of order `n`: their entries are built from binomial
/// coefficients in O(n) operations, so a product costs O(n³).
/// [`StructuredSolvePlan`](crate::StructuredSolvePlan) solves Toeplitz,
/// Hankel and circulant systems in O(n²) operations.
///
/// ```
/// use scioxide_linalg::{CompanionKind, HelmertKind, MatrixForm, PascalKind, StructuredMatrixView};
///
/// // inv([1 1/2; 1/2 1/3]) = [4 −6; −6 12].
/// let inverse = StructuredMatrixView::<f64>::hilbert(2, MatrixForm::Inverse)?;
/// assert_eq!(inverse.get(0, 0)?, 4.0);
/// // C(2, 1) = 2.
/// let pascal = StructuredMatrixView::<f64>::pascal(3, PascalKind::Lower, MatrixForm::Direct)?;
/// assert_eq!(pascal.get(2, 1)?, 2.0);
/// let helmert = StructuredMatrixView::<f64>::helmert(3, HelmertKind::Reduced)?;
/// assert_eq!((helmert.rows(), helmert.cols()), (2, 3));
/// // x² − 3x + 2 has the Frobenius companion matrix [0 −2; 1 3].
/// let companion =
///     StructuredMatrixView::companion(&[2.0_f64, -3.0, 1.0], CompanionKind::Frobenius)?;
/// assert_eq!(companion.get(0, 1)?, -2.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
/// M. Fiedler, "A note on companion matrices", *Linear Algebra Appl.* 372 (2003),
/// 325–331.
#[derive(Debug, Clone, Copy)]
pub struct StructuredMatrixView<'a, T: NumericScalar> {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) pattern: Pattern<'a, T>,
}
impl<'a, T: NumericScalar> StructuredMatrixView<'a, T> {
    fn checked(rows: usize, cols: usize, pattern: Pattern<'a, T>) -> Result<Self, SciOxideError> {
        size(rows, cols, scioxide_core::quantity!("structured_elements"))?;
        Ok(Self {
            rows,
            cols,
            pattern,
        })
    }
    /// Creates the Toeplitz matrix with first column `column` and first row `row`.
    ///
    /// The matrix is `m × n` with `m = column.len()` and `n = row.len()`:
    ///
    /// ```text
    /// A[i][j] = column[i − j]   for i ≥ j
    /// A[i][j] = row[j − i]      for i < j
    /// ```
    ///
    /// # Errors
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `InvalidInput` if both slices are nonempty and `column[0] ≠ row[0]`.
    /// - `SizeOverflow` if `m · n` overflows `usize`.
    pub fn toeplitz(column: &'a [T], row: &'a [T]) -> Result<Self, SciOxideError> {
        crate::checked::validate_inputs(column)?;
        crate::checked::validate_inputs(row)?;
        if !column.is_empty() && !row.is_empty() && column[0] != row[0] {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("row"),
                "Toeplitz corners differ",
            ));
        }
        Self::checked(column.len(), row.len(), Pattern::Toeplitz(column, row))
    }
    /// Creates the Hankel matrix with first column `column` and last row `last_row`.
    ///
    /// The matrix is `m × n` with `m = column.len()` and `n = last_row.len()`,
    /// and `A[i][j]` depends only on `i + j`:
    ///
    /// ```text
    /// A[i][j] = column[i + j]              for i + j < m
    /// A[i][j] = last_row[i + j − m + 1]    for i + j ≥ m
    /// ```
    ///
    /// # Errors
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `InvalidInput` if both slices are nonempty and the last entry of
    ///   `column` differs from `last_row[0]`.
    /// - `SizeOverflow` if `m · n` overflows `usize`.
    pub fn hankel(column: &'a [T], last_row: &'a [T]) -> Result<Self, SciOxideError> {
        crate::checked::validate_inputs(column)?;
        crate::checked::validate_inputs(last_row)?;
        if let (Some(c), Some(r)) = (column.last(), last_row.first())
            && c != r
        {
            return Err(crate::checked::inconsistent(
                scioxide_core::argument!("last_row"),
                "Hankel corners differ",
            ));
        }
        Self::checked(
            column.len(),
            last_row.len(),
            Pattern::Hankel(column, last_row),
        )
    }
    /// Creates the `n × n` circulant matrix `A[i][j] = column[(i − j) mod n]`.
    ///
    /// # Errors
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `SizeOverflow` if `n²` overflows `usize`.
    pub fn circulant(column: &'a [T]) -> Result<Self, SciOxideError> {
        crate::checked::validate_inputs(column)?;
        Self::checked(column.len(), column.len(), Pattern::Circulant(column))
    }
    /// Creates the Hilbert matrix `H[i][j] = 1 / (i + j + 1)` of order `n`, or its inverse.
    ///
    /// The inverse has the integer entries
    ///
    /// ```text
    /// H⁻¹[i][j] = (−1)^(i+j) (i + j + 1) C(n + i, n − j − 1) C(n + j, n − i − 1) C(i + j, i)²
    /// ```
    ///
    /// They grow quickly with `n`; an entry that overflows `T` is an error
    /// when it is evaluated.
    ///
    /// # Errors
    /// - `SizeOverflow` if `n²` overflows `usize`.
    pub fn hilbert(order: usize, form: MatrixForm) -> Result<Self, SciOxideError> {
        Self::checked(order, order, Pattern::Hilbert(form))
    }
    /// Creates the Pascal matrix of order `n`, or its inverse.
    ///
    /// [`PascalKind`] defines the entries. The inverses have closed forms:
    /// `L⁻¹[i][j] = (−1)^(i−j) C(i, j)` for the triangular forms and
    /// `S⁻¹[i][j] = (−1)^(i+j) Σₖ C(k, i) C(k, j)`, summed over
    /// `max(i, j) ≤ k < n`, for the symmetric form. An entry that overflows `T`
    /// is an error when it is evaluated.
    ///
    /// # Errors
    /// - `SizeOverflow` if `n²` overflows `usize`.
    pub fn pascal(order: usize, kind: PascalKind, form: MatrixForm) -> Result<Self, SciOxideError> {
        Self::checked(order, order, Pattern::Pascal(kind, form))
    }
    /// Creates the Sylvester–Hadamard matrix of order `n`, without normalization.
    ///
    /// The entries are `H[i][j] = (−1)^popcount(i & j)`, so `H Hᵀ = n I`.
    /// Order zero gives an empty matrix.
    ///
    /// # Errors
    /// - `InvalidInput` if `n` is nonzero and not a power of two.
    /// - `SizeOverflow` if `n²` overflows `usize`.
    pub fn hadamard(order: usize) -> Result<Self, SciOxideError> {
        if order != 0 && !order.is_power_of_two() {
            return Err(crate::checked::out_of_range(
                scioxide_core::argument!("order"),
                "Hadamard order must be a power of two",
            ));
        }
        Self::checked(order, order, Pattern::Hadamard)
    }
    /// Creates the orthonormal Helmert matrix of order `n`, or its last `n − 1` rows.
    ///
    /// Row 0 is constant, `1/√n`. For `k ≥ 1`, row `k` is
    ///
    /// ```text
    /// H[k][j] = 1/√(k(k+1))     for j < k
    /// H[k][k] = −k/√(k(k+1))
    /// H[k][j] = 0               for j > k
    /// ```
    ///
    /// [`HelmertKind::Reduced`] gives the `(n − 1) × n` matrix of rows `1..n`,
    /// whose rows are orthonormal contrasts.
    ///
    /// # Errors
    /// - `SizeOverflow` if the number of entries overflows `usize`.
    pub fn helmert(order: usize, kind: HelmertKind) -> Result<Self, SciOxideError> {
        Self::checked(
            if kind == HelmertKind::Full {
                order
            } else {
                order.saturating_sub(1)
            },
            order,
            Pattern::Helmert(kind),
        )
    }
    /// Creates the `n × n` Leslie matrix with first row `fertility` and subdiagonal `survival`.
    ///
    /// `A[0][j] = fertility[j]`, `A[j + 1][j] = survival[j]`, and all other
    /// entries are zero. `n = fertility.len()`.
    ///
    /// # Errors
    /// - `NonFinite` if a value is NaN or infinite.
    /// - `DimensionMismatch` if `survival.len()` is not `n − 1` (or 0 for `n = 0`).
    /// - `SizeOverflow` if `n²` overflows `usize`.
    pub fn leslie(fertility: &'a [T], survival: &'a [T]) -> Result<Self, SciOxideError> {
        crate::checked::validate_inputs(fertility)?;
        crate::checked::validate_inputs(survival)?;
        let subdiagonal = fertility.len().saturating_sub(1);
        if survival.len() != subdiagonal {
            return Err(SciOxideError::dimension_mismatch(
                [subdiagonal],
                [survival.len()],
            ));
        }
        Self::checked(
            fertility.len(),
            fertility.len(),
            Pattern::Leslie(fertility, survival),
        )
    }
    /// Creates the Fiedler matrix `A[i][j] = |values[i] − values[j]|`.
    ///
    /// For complex scalars `|·|` is the modulus, stored as a real value of `T`.
    ///
    /// # Errors
    /// - `NonFinite` if a value is NaN or infinite.
    /// - `SizeOverflow` if `n²` overflows `usize`.
    pub fn fiedler(values: &'a [T]) -> Result<Self, SciOxideError> {
        crate::checked::validate_inputs(values)?;
        Self::checked(values.len(), values.len(), Pattern::Fiedler(values))
    }
    /// Creates a companion matrix of `p(x) = c₀ + c₁ x + … + cₙ xⁿ`.
    ///
    /// `coefficients` holds `c₀, …, cₙ` in ascending order of power. The
    /// matrix is `n × n` and its eigenvalues are the roots of `p`. A nonzero
    /// constant (`n = 0`) gives an empty matrix. [`CompanionKind`] selects the
    /// layout.
    ///
    /// # Errors
    /// - `NonFinite` if a coefficient is NaN or infinite.
    /// - `InvalidInput` if `coefficients` is empty or `cₙ = 0`.
    /// - `SizeOverflow` if `n²` overflows `usize`.
    pub fn companion(coefficients: &'a [T], kind: CompanionKind) -> Result<Self, SciOxideError> {
        crate::checked::validate_inputs(coefficients)?;
        if coefficients.last().is_none_or(|v| *v == T::ZERO) {
            return Err(crate::checked::out_of_range(
                scioxide_core::argument!("coefficients"),
                "companion requires a nonzero leading coefficient",
            ));
        }
        let n = coefficients.len() - 1;
        Self::checked(n, n, Pattern::Companion(coefficients, kind))
    }
    /// Creates the matrix `A` such that `A x` is the convolution of `kernel` with `x`.
    ///
    /// `A` has `input_len` columns and the number of rows given by `mode`. With
    /// the row offset `s` of the mode (zero for [`ConvolutionMode::Full`]),
    /// `A[i][j] = kernel[i + s − j]`, and zero where that index is out of range.
    /// An empty kernel or `input_len = 0` gives zero rows and `input_len` columns.
    ///
    /// # Errors
    /// - `NonFinite` if a kernel value is NaN or infinite.
    /// - `SizeOverflow` if the number of rows or entries overflows `usize`.
    pub fn convolution(
        kernel: &'a [T],
        input_len: usize,
        mode: ConvolutionMode,
    ) -> Result<Self, SciOxideError> {
        crate::checked::validate_inputs(kernel)?;
        let m = kernel.len();
        let (rows, offset) = if m == 0 || input_len == 0 {
            (0, 0)
        } else {
            let full = m.checked_add(input_len - 1).ok_or_else(|| {
                SciOxideError::size_overflow(scioxide_core::quantity!("convolution"))
            })?;
            match mode {
                ConvolutionMode::Full => (full, 0),
                ConvolutionMode::Same => (m.max(input_len), (m.min(input_len) - 1) / 2),
                ConvolutionMode::Valid => (
                    m.max(input_len) - m.min(input_len) + 1,
                    m.min(input_len) - 1,
                ),
            }
        };
        Self::checked(rows, input_len, Pattern::Convolution(kernel, offset))
    }
    /// Returns the number of rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }
    /// Returns the number of columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }
    /// Returns entry `(i, j)`.
    ///
    /// # Errors
    /// - `OutOfBounds` if `i ≥ rows` or `j ≥ cols`.
    /// - `NumericalInstability` if the entry overflows `T`.
    pub fn get(&self, i: usize, j: usize) -> Result<T, SciOxideError> {
        if i >= self.rows {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("structured_row"),
                i,
                self.rows,
            ));
        }
        if j >= self.cols {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("structured_column"),
                j,
                self.cols,
            ));
        }
        self.entry(i, j)
    }
    pub(super) fn entry(&self, i: usize, j: usize) -> Result<T, SciOxideError> {
        let one = T::ONE;
        let zero = T::ZERO;
        let n = self.cols;
        let value = match self.pattern {
            Pattern::Toeplitz(c, r) => {
                if i >= j {
                    c[i - j]
                } else {
                    r[j - i]
                }
            }
            Pattern::Hankel(c, r) => {
                if i + j < c.len() {
                    c[i + j]
                } else {
                    r[i + j - c.len() + 1]
                }
            }
            Pattern::Circulant(c) => c[if i >= j { i - j } else { n - (j - i) }],
            Pattern::Hilbert(MatrixForm::Direct) => quotient(one, number(i + j + 1)?)?,
            Pattern::Hilbert(MatrixForm::Inverse) => {
                let mut v = wide(number::<T>(i + j + 1)?)?;
                for x in [
                    binomial::<T>(n + i, n - j - 1)?,
                    binomial(n + j, n - i - 1)?,
                    binomial(i + j, i)?,
                    binomial(i + j, i)?,
                ] {
                    v = v.mul(wide(x)?)?;
                }
                let v: T = narrow(v)?;
                if (i + j).is_multiple_of(2) { v } else { -v }
            }
            Pattern::Pascal(kind, form) => {
                let (a, b) = if matches!(kind, PascalKind::Upper) {
                    (j, i)
                } else {
                    (i, j)
                };
                if matches!(kind, PascalKind::Symmetric) {
                    if form == MatrixForm::Inverse {
                        // Σₖ C(k, i) C(k, j) over k ≥ max(i, j), advancing both
                        // binomials by C(k + 1, m) = C(k, m) (k + 1) / (k + 1 − m).
                        let start = i.max(j);
                        let mut left = wide(binomial::<T>(start, i)?)?;
                        let mut right = wide(binomial::<T>(start, j)?)?;
                        let mut sum = left.mul(right)?;
                        for k in start + 1..n {
                            let next = wide(number::<T>(k)?)?;
                            left = left.mul(next)?.div(wide(number::<T>(k - i)?)?)?;
                            right = right.mul(next)?.div(wide(number::<T>(k - j)?)?)?;
                            sum = sum.add(left.mul(right)?)?;
                        }
                        let v: T = narrow(sum)?;
                        if (i + j).is_multiple_of(2) { v } else { -v }
                    } else {
                        binomial(i + j, i)?
                    }
                } else if a < b {
                    zero
                } else {
                    let v: T = binomial(a, b)?;
                    if form == MatrixForm::Inverse && (a - b) % 2 == 1 {
                        -v
                    } else {
                        v
                    }
                }
            }
            Pattern::Hadamard => {
                if (i & j).count_ones().is_multiple_of(2) {
                    one
                } else {
                    -one
                }
            }
            Pattern::Helmert(kind) => {
                let k = if kind == HelmertKind::Full { i } else { i + 1 };
                if k == 0 {
                    T::from_real(T::Real::ONE / real_from_usize::<T::Real>(n)?.sqrt())
                } else if j > k {
                    zero
                } else {
                    let a = real_from_usize::<T::Real>(k)?;
                    let b = real_from_usize::<T::Real>(k + 1)?;
                    T::from_real(if j < k {
                        T::Real::ONE / (a * b).sqrt()
                    } else {
                        -a / (a * b).sqrt()
                    })
                }
            }
            Pattern::Leslie(f, s) => {
                if i == 0 {
                    f[j]
                } else if i == j + 1 {
                    s[j]
                } else {
                    zero
                }
            }
            Pattern::Fiedler(v) => T::from_real(narrow::<T>(wide(v[i])?.sub(wide(v[j])?)?)?.abs()),
            Pattern::Companion(c, CompanionKind::Frobenius) => {
                if j == n - 1 {
                    -quotient(c[i], c[n])?
                } else if i == j + 1 {
                    one
                } else {
                    zero
                }
            }
            Pattern::Companion(c, CompanionKind::Fiedler) => {
                if (i == 0 && j == 0)
                    || (i.is_multiple_of(2) && j == i + 1)
                    || (i > 0 && i.is_multiple_of(2) && j == i - 1)
                {
                    let power = if i == 0 && j == 0 {
                        n - 1
                    } else if j == i + 1 {
                        n - i - 2
                    } else {
                        n - i - 1
                    };
                    -quotient(c[power], c[n])?
                } else if (i == 1 && j == 0)
                    || (i.is_multiple_of(2) && j == i + 2)
                    || (i >= 3 && i % 2 == 1 && j == i - 2)
                {
                    one
                } else {
                    zero
                }
            }
            Pattern::Convolution(k, offset) => (i + offset)
                .checked_sub(j)
                .and_then(|a| k.get(a))
                .copied()
                .unwrap_or(zero),
            Pattern::Dft(direction, norm) => {
                let phase = phase::<T::Real>(i, j, n, direction)?;
                let scale = match norm {
                    DftNormalization::None => T::Real::ONE,
                    DftNormalization::ByOrder => real_from_usize(n)?,
                    DftNormalization::Unitary => real_from_usize::<T::Real>(n)?.sqrt(),
                };
                T::from_components(phase.re / scale, phase.im / scale)
            }
        };
        crate::checked::finite_candidate(value)
    }
    /// Returns the matrix as a new dense [`Matrix`], evaluating each entry once.
    ///
    /// To write into an existing matrix, use [`crate::materialize_operator_into`].
    ///
    /// ```
    /// use scioxide_linalg::{MatrixForm, StructuredMatrixView};
    /// let hilbert = StructuredMatrixView::<f64>::hilbert(3, MatrixForm::Direct)?;
    /// // Hilbert entries are 1 / (i + j + 1).
    /// assert_eq!(hilbert.materialize()?.view().get(1, 2)?, 0.25);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    /// # Errors
    /// - `SizeOverflow` if the size of the result overflows `usize`.
    /// - `AllocationFailure` if the result cannot be allocated.
    /// - `NumericalInstability` if an entry overflows `T`.
    pub fn materialize(&self) -> Result<Matrix<T>, SciOxideError> {
        self.materialize_with_options(&crate::LinalgExecutionOptions::default())
    }

    /// Runs [`materialize`](Self::materialize) with the byte limit of `options` applied to the
    /// `rows × cols` result.
    ///
    /// ```
    /// use scioxide_core::SciOxideErrorKind;
    /// use scioxide_linalg::{LinalgExecutionOptions, MatrixForm, StructuredMatrixView};
    /// let hilbert = StructuredMatrixView::<f64>::hilbert(3, MatrixForm::Direct)?;
    /// let limit = |bytes| LinalgExecutionOptions::new().with_byte_limit(bytes);
    /// let dense = hilbert.materialize_with_options(&limit(9 * 8))?;
    /// assert_eq!(dense.view().get(1, 2)?, 0.25);
    /// let capped = hilbert.materialize_with_options(&limit(9 * 8 - 1));
    /// assert_eq!(capped.unwrap_err().kind(), SciOxideErrorKind::ResourceLimit);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The errors of [`materialize`](Self::materialize), and:
    ///
    /// - `ResourceLimit` if the result exceeds the byte limit in `options`.
    pub fn materialize_with_options(
        &self,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Matrix<T>, SciOxideError> {
        let quantity = scioxide_core::quantity!("materialized_bytes");
        admit_owned::<T>(size(self.rows, self.cols, quantity)?, options, quantity)?;
        let mut out = Matrix::zeros(self.rows, self.cols)?;
        for (index, value) in out.as_mut_slice().iter_mut().enumerate() {
            *value = self.entry(index / self.cols, index % self.cols)?;
        }
        Ok(out)
    }
}
impl<'a, R: RealScalar> StructuredMatrixView<'a, Complex<R>>
where
    Complex<R>: NumericScalar<Real = R>,
{
    /// Creates the DFT matrix of order `n`.
    ///
    /// ```text
    /// F[j][k] = exp(∓2πi jk / n) / s
    /// ```
    ///
    /// The sign is negative for [`DftDirection::Forward`] and positive for
    /// [`DftDirection::Inverse`]. The scale `s` is 1, `n` or `√n` for the
    /// three [`DftNormalization`] values. `F x` with `Forward` and no scaling is
    /// the DFT `X[j] = Σₖ x[k] exp(−2πi jk / n)`. The product `jk` is reduced
    /// modulo `n` in integer arithmetic before the exponential is evaluated.
    ///
    /// # Errors
    /// - `SizeOverflow` if `n²` overflows `usize`.
    pub fn dft(
        order: usize,
        direction: DftDirection,
        normalization: DftNormalization,
    ) -> Result<Self, SciOxideError> {
        Self::checked(order, order, Pattern::Dft(direction, normalization))
    }
}
pub(super) fn number<T: NumericScalar>(n: usize) -> Result<T, SciOxideError> {
    Ok(T::from_real(real_from_usize(n)?))
}
fn binomial<T: NumericScalar>(n: usize, k: usize) -> Result<T, SciOxideError> {
    let mut v = wide(T::ONE)?;
    for j in 1..=k.min(n - k) {
        v = v
            .mul(wide(number::<T>(n - j + 1)?)?)?
            .div(wide(number::<T>(j)?)?)?;
    }
    narrow(v)
}
pub(super) fn phase<R: RealScalar>(
    i: usize,
    j: usize,
    n: usize,
    direction: DftDirection,
) -> Result<Complex<R>, SciOxideError> {
    // DFT twiddle `exp(-+2 pi i ij / n)`: the phase index is reduced modulo n
    // exactly in integers before conversion, so the argument passed to
    // `sin_cos_pi` is in [0, 2) and large `i * j` loses no accuracy.
    let product = (i as u128 * j as u128) % n as u128;
    let x = real_from_usize::<R>(product as usize)? / real_from_usize::<R>(n)?;
    let (sin, cos) = scioxide_core::sin_cos_pi(x + x)?;
    Ok(Complex::new(
        cos,
        match direction {
            DftDirection::Forward => -sin,
            DftDirection::Inverse => sin,
        },
    ))
}
impl<T: NumericScalar> LinearOperator<T> for StructuredMatrixView<'_, T> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.rows)
    }
    fn apply_into(&self, x: &[T], out: &mut [T], scratch: &mut [T]) -> Result<(), SciOxideError> {
        self.apply(false, x, out, scratch)
    }
}
impl<T: NumericScalar> AdjointOperator<T> for StructuredMatrixView<'_, T> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.cols)
    }
    fn apply_adjoint_into(
        &self,
        x: &[T],
        out: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        self.apply(true, x, out, scratch)
    }
}
impl<T: NumericScalar> BlockOperator<T> for StructuredMatrixView<'_, T> {}
impl<T: NumericScalar> StructuredMatrixView<'_, T> {
    fn apply(
        &self,
        adj: bool,
        x: &[T],
        out: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), SciOxideError> {
        let (m, n) = if adj {
            (self.cols, self.rows)
        } else {
            (self.rows, self.cols)
        };
        check_apply([n, m, m], x.len(), out.len(), scratch.len())?;
        crate::checked::validate_inputs(x)?;
        for (i, v) in scratch[..m].iter_mut().enumerate() {
            let mut sum = wide(T::ZERO)?;
            for (j, &xj) in x.iter().enumerate() {
                let a = if adj {
                    self.entry(j, i)?.conj()
                } else {
                    self.entry(i, j)?
                };
                sum = sum.add(wide(a)?.mul(wide(xj)?)?)?;
            }
            *v = narrow(sum)?;
        }
        out.copy_from_slice(&scratch[..m]);
        Ok(())
    }
}
/// Toeplitz, Hankel or circulant matrix that owns its defining coefficients.
///
/// [`view`](Self::view) returns a [`StructuredMatrixView`] with the same
/// entries.
///
/// ```
/// use scioxide_linalg::{LinalgExecutionOptions, StructuredMatrix, StructuredMatrixView};
/// let column = [1.0_f64, 2.0, 3.0];
/// let owned = StructuredMatrix::from_view_with_options(
///     StructuredMatrixView::circulant(&column)?,
///     &LinalgExecutionOptions::new().with_byte_limit(3 * 8),
/// )?;
/// // A[0][1] = column[(0 − 1) mod 3] = column[2].
/// assert_eq!(owned.view().get(0, 1)?, 3.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct StructuredMatrix<T: NumericScalar> {
    coefficients: Compact<Vec<T>>,
    rows: usize,
    cols: usize,
}
/// Compact coefficient storage of one pattern.
#[derive(Debug, Clone)]
enum Compact<C> {
    Toeplitz { column: C, row: C },
    Hankel { column: C, last_row: C },
    Circulant(C),
}
impl<C> Compact<C> {
    fn try_map<D>(
        self,
        mut f: impl FnMut(C) -> Result<D, SciOxideError>,
    ) -> Result<Compact<D>, SciOxideError> {
        Ok(match self {
            Self::Toeplitz { column, row } => Compact::Toeplitz {
                column: f(column)?,
                row: f(row)?,
            },
            Self::Hankel { column, last_row } => Compact::Hankel {
                column: f(column)?,
                last_row: f(last_row)?,
            },
            Self::Circulant(column) => Compact::Circulant(f(column)?),
        })
    }
}
impl<T: NumericScalar> StructuredMatrix<T> {
    /// Copies the coefficients of a Toeplitz, Hankel or circulant view.
    ///
    /// An empty matrix stores no coefficients.
    ///
    /// # Errors
    /// - `InvalidInput` if `view` is not a Toeplitz, Hankel or circulant matrix.
    /// - `SizeOverflow` if the number of coefficients overflows `usize`.
    /// - `AllocationFailure` if the copy cannot be allocated.
    pub fn from_view(view: StructuredMatrixView<'_, T>) -> Result<Self, SciOxideError> {
        Self::from_view_with_options(view, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`from_view`](Self::from_view) with the byte limit of `options` applied to the copied
    /// coefficients.
    ///
    /// # Errors
    ///
    /// The errors of [`from_view`](Self::from_view), and:
    ///
    /// - `ResourceLimit` if the coefficients exceed the byte limit in `options`.
    pub fn from_view_with_options(
        view: StructuredMatrixView<'_, T>,
        options: &crate::LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let quantity = scioxide_core::quantity!("compact_coefficient_bytes");
        let compact = match view.pattern {
            Pattern::Toeplitz(column, row) => Compact::Toeplitz { column, row },
            Pattern::Hankel(column, last_row) => Compact::Hankel { column, last_row },
            Pattern::Circulant(column) => Compact::Circulant(column),
            _ => {
                return Err(crate::checked::unsupported(
                    scioxide_core::argument!("view"),
                    "owned compact storage requires Toeplitz, Hankel or circulant",
                ));
            }
        };
        let empty = view.rows == 0 || view.cols == 0;
        let mut len = 0usize;
        let compact = compact.try_map(|values| {
            let values = if empty { &[][..] } else { values };
            len = len
                .checked_add(values.len())
                .ok_or_else(|| SciOxideError::size_overflow(quantity))?;
            Ok(values)
        })?;
        admit_owned::<T>(len, options, quantity)?;
        Ok(Self {
            coefficients: compact.try_map(|values| {
                let mut copied = scioxide_core::backend::try_with_capacity(values.len(), quantity)?;
                copied.extend_from_slice(values);
                Ok(copied)
            })?,
            rows: view.rows,
            cols: view.cols,
        })
    }
    /// Borrows the matrix as a [`StructuredMatrixView`].
    pub fn view(&self) -> StructuredMatrixView<'_, T> {
        StructuredMatrixView {
            rows: self.rows,
            cols: self.cols,
            pattern: match &self.coefficients {
                Compact::Toeplitz { column, row } => Pattern::Toeplitz(column, row),
                Compact::Hankel { column, last_row } => Pattern::Hankel(column, last_row),
                Compact::Circulant(column) => Pattern::Circulant(column),
            },
        }
    }
}
