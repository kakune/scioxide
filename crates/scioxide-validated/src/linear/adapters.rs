//! Interval operators for dense and sparse matrices and LU factors.
use super::*;
use scioxide_core::backend::{RealBackend, RealProductSum};
use scioxide_core::{LinearOperator, ProductSum, StructuralPatternView};
use scioxide_linalg::{
    LinalgExecutionOptions, LinalgWorkspace, LuDecomposition, LuWorkspaceRequest, Matrix,
    MatrixView,
};
use scioxide_sparse::{CsrMatrix, SparseLu};

/// Dense interval matrix `[L, U]`, borrowed from two matrix views: the set of
/// matrices `A` with `Lᵢⱼ ≤ Aᵢⱼ ≤ Uᵢⱼ`, applied to the real and imaginary parts
/// separately.
///
/// A point matrix uses the same view for both bounds. The views may be strided
/// or conjugated; their logical entries are used. An enclosure costs O(n²)
/// operations for an `n × n` matrix.
///
/// ```
/// use scioxide_linalg::MatrixView;
/// use scioxide_validated::DenseIntervalOperator;
///
/// // Every matrix with diagonal entries in [1.9, 2.1] and zero elsewhere.
/// let (lower, upper) = ([1.9_f64, 0.0, 0.0, 1.9], [2.1_f64, 0.0, 0.0, 2.1]);
/// let a = DenseIntervalOperator::new(
///     MatrixView::from_slice(2, 2, &lower)?,
///     MatrixView::from_slice(2, 2, &upper)?,
/// )?;
/// # let _ = a;
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy)]
pub struct DenseIntervalOperator<'a, S: NumericScalar> {
    lower: MatrixView<'a, S>,
    upper: MatrixView<'a, S>,
}
impl<'a, S: NumericScalar> DenseIntervalOperator<'a, S> {
    /// Creates the interval matrix with entrywise bounds `lower` and `upper`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the views differ in shape.
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - `InvalidInput` if a real or imaginary part of `lower` exceeds that of
    ///   `upper`.
    pub fn new(lower: MatrixView<'a, S>, upper: MatrixView<'a, S>) -> Result<Self> {
        crate::shell::exact(lower.rows(), upper.rows())?;
        crate::shell::exact(lower.cols(), upper.cols())?;
        let result = Self { lower, upper };
        for row in 0..lower.rows() {
            result.visit(row, |_, lo, hi| interval_enclosure(lo, hi).map(|_| ()))?;
        }
        Ok(result)
    }
    /// Creates the operator of the single matrix `matrix`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if an entry is NaN or infinite.
    pub fn point(matrix: MatrixView<'a, S>) -> Result<Self> {
        Self::new(matrix, matrix)
    }
    fn visit(&self, row: usize, mut f: impl FnMut(usize, S, S) -> Result<()>) -> Result<()> {
        (0..self.lower.cols())
            .try_for_each(|j| f(j, self.lower.get(row, j)?, self.upper.get(row, j)?))
    }
    fn column(&self, row: usize, column: usize) -> Result<IntervalBox<S::Real>> {
        interval_enclosure(self.lower.get(row, column)?, self.upper.get(row, column)?)
    }
    /// Direct coefficients only for contiguous point views; other layouts keep
    /// their logical strided/conjugated traversal.
    fn point_coefficients(&self, row: usize) -> Option<&[S]> {
        let lower = self.lower.as_contiguous_slice()?;
        let upper = self.upper.as_contiguous_slice()?;
        core::ptr::eq(lower, upper)
            .then(|| &lower[row * self.lower.cols()..(row + 1) * self.lower.cols()])
    }
}
/// Sparse interval matrix in compressed sparse row (CSR) form, borrowed.
///
/// The lower and upper bounds share one sparsity pattern from
/// `scioxide-sparse`. Entries outside the pattern are exactly zero. An enclosure
/// costs O(nnz) operations for `nnz` stored entries.
#[derive(Clone, Copy)]
pub struct SparseIntervalOperator<'a, S: NumericScalar> {
    pattern: StructuralPatternView<'a>,
    lower: &'a [S],
    upper: &'a [S],
}
impl<'a, S: NumericScalar> SparseIntervalOperator<'a, S> {
    /// Creates the interval matrix with the stored entries `lower` and `upper`
    /// on `pattern`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `lower` or `upper` does not have one value per
    ///   stored entry.
    /// - `NonFinite` if a value is NaN or infinite.
    /// - `InvalidInput` if a real or imaginary part of `lower` exceeds that of
    ///   `upper`.
    pub fn new(pattern: StructuralPatternView<'a>, lower: &'a [S], upper: &'a [S]) -> Result<Self> {
        crate::shell::exact(pattern.col_indices().len(), lower.len())?;
        crate::shell::exact(lower.len(), upper.len())?;
        lower
            .iter()
            .zip(upper)
            .try_for_each(|(&a, &b)| interval_enclosure(a, b).map(|_| ()))?;
        Ok(Self {
            pattern,
            lower,
            upper,
        })
    }
    /// Creates the operator of the single matrix `matrix`, without copying it.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a value is NaN or infinite.
    pub fn point(matrix: &'a CsrMatrix<S>) -> Result<Self> {
        Self::new(
            matrix.structural_pattern(),
            matrix.values(),
            matrix.values(),
        )
    }
    fn visit(&self, row: usize, mut f: impl FnMut(usize, S, S) -> Result<()>) -> Result<()> {
        (self.pattern.row_ptr()[row]..self.pattern.row_ptr()[row + 1])
            .try_for_each(|k| f(self.pattern.col_indices()[k], self.lower[k], self.upper[k]))
    }
    fn column(&self, row: usize, column: usize) -> Result<IntervalBox<S::Real>> {
        let start = self.pattern.row_ptr()[row];
        let end = self.pattern.row_ptr()[row + 1];
        match self.pattern.col_indices()[start..end].binary_search(&column) {
            Ok(k) => interval_enclosure(self.lower[start + k], self.upper[start + k]),
            Err(_) => zero(),
        }
    }
    fn point_coefficients(&self, _: usize) -> Option<&[S]> {
        None
    }
}

/// Exact basis columns need no reduction or rounding-error allowance.
fn unit_column<R: RealScalar>(input: &[IntervalBox<R>]) -> Option<usize> {
    let mut column = None;
    for (j, &x) in input.iter().enumerate() {
        let x = unpack(x);
        if !x.im_is_zero() {
            return None;
        }
        let (lo, hi) = x.re.bounds()?;
        if lo == R::ONE && hi == R::ONE && column.is_none() {
            column = Some(j);
        } else if lo != R::ZERO || hi != R::ZERO {
            return None;
        }
    }
    column
}
/// Check the arguments of the enclosure application.
fn application_shape<S: NumericScalar>(
    rows: usize,
    cols: usize,
    input: &[IntervalBox<S::Real>],
    addend: Option<&[IntervalBox<S::Real>]>,
    output: &[IntervalBox<S::Real>],
    workspace: &[IntervalBox<S::Real>],
    needed: usize,
) -> Result<()> {
    crate::environment::check()?;
    crate::shell::exact(cols, input.len())?;
    crate::shell::exact(rows, output.len())?;
    if let Some(b) = addend {
        crate::shell::exact(rows, b.len())?;
    }
    crate::shell::at_least(needed, workspace.len())?;
    validate_boxes::<S>(input, true, scioxide_core::argument!("input"))?;
    if let Some(b) = addend {
        validate_boxes::<S>(b, true, scioxide_core::argument!("addend"))?;
    }
    Ok(())
}
macro_rules! matrix_operator {
    ($name:ident,$rows:expr,$cols:expr) => {
        impl<S: NumericScalar> LinearOperator<S> for $name<'_, S> {
            fn rows(&self) -> usize {
                ($rows)(self)
            }
            fn cols(&self) -> usize {
                ($cols)(self)
            }
            fn apply_workspace_len(&self) -> Result<usize> {
                Ok(self.rows())
            }
            fn apply_into(&self, input: &[S], output: &mut [S], workspace: &mut [S]) -> Result<()> {
                crate::shell::exact(self.cols(), input.len())?;
                crate::shell::exact(self.rows(), output.len())?;
                crate::shell::at_least(self.rows(), workspace.len())?;
                if input.iter().any(|x| !x.has_finite_components()) {
                    return Err(SciOxideError::non_finite(
                        scioxide_core::NonFiniteOrigin::Input,
                        "application input must be finite",
                    ));
                }
                for (row, slot) in workspace[..self.rows()].iter_mut().enumerate() {
                    if let Some(coefficients) = self.point_coefficients(row) {
                        // This is an untrusted ordinary application. The proof
                        // encloses residuals independently; exceptional native
                        // ranges retain exact accumulation below.
                        let candidate = scioxide_core::backend::standard::dot(
                            coefficients.iter().copied().zip(input.iter().copied()),
                        );
                        if candidate.has_finite_components() {
                            *slot = candidate;
                            continue;
                        }
                        let mut re = RealProductSum::new();
                        let mut im = RealProductSum::new();
                        for (&a, &x) in coefficients.iter().zip(input) {
                            re.add_product(a.real_component(), x.real_component())?;
                            if !S::IS_REAL {
                                re.add_product(-a.imaginary_component(), x.imaginary_component())?;
                                im.add_product(a.real_component(), x.imaginary_component())?;
                                im.add_product(a.imaginary_component(), x.real_component())?;
                            }
                        }
                        *slot = S::from_components(
                            re.finish()?,
                            if !S::IS_REAL {
                                im.finish()?
                            } else {
                                S::Real::ZERO
                            },
                        );
                        continue;
                    }
                    let mut sum = ProductSum::new();
                    self.visit(row, |j, lo, hi| {
                        let re = scioxide_core::checked_midpoint(
                            lo.real_component(),
                            hi.real_component(),
                        )?;
                        let im = scioxide_core::checked_midpoint(
                            lo.imaginary_component(),
                            hi.imaginary_component(),
                        )?;
                        sum.add_product(S::from_components(re, im), input[j])
                    })?;
                    *slot = sum.finish()?;
                }
                output.copy_from_slice(&workspace[..self.rows()]);
                Ok(())
            }
        }
        impl<S: NumericScalar> IntervalLinearOperator<S> for $name<'_, S> {
            fn enclosure_workspace_len(&self) -> Result<usize> {
                Ok(self.rows())
            }
            fn apply_enclosure_into(
                &self,
                input: &[IntervalBox<S::Real>],
                addend: Option<&[IntervalBox<S::Real>]>,
                output: &mut [IntervalBox<S::Real>],
                workspace: &mut [IntervalBox<S::Real>],
            ) -> Result<()> {
                application_shape::<S>(
                    self.rows(),
                    self.cols(),
                    input,
                    addend,
                    output,
                    workspace,
                    self.rows(),
                )?;
                if let Some(column) = addend.is_none().then(|| unit_column(input)).flatten() {
                    for (row, slot) in workspace[..self.rows()].iter_mut().enumerate() {
                        *slot = self.column(row, column)?;
                    }
                    output.copy_from_slice(&workspace[..self.rows()]);
                    return Ok(());
                }
                for (row, slot) in workspace[..self.rows()].iter_mut().enumerate() {
                    if let (Some(coefficients), Some(bias)) = (self.point_coefficients(row), addend)
                        && let Some(value) = tight_point_row(coefficients, input, bias[row])?
                    {
                        *slot = value;
                        continue;
                    }
                    let mut sum = BoxSum::new(
                        if let Some(b) = addend {
                            unpack(b[row])
                        } else {
                            Box2::zero()?
                        },
                        scioxide_core::argument!("input"),
                    )?;
                    self.visit(row, |j, lo, hi| {
                        sum.product(unpack(interval_enclosure(lo, hi)?), unpack(input[j]), false)
                    })?;
                    *slot = pack(sum.finish()?);
                }
                output.copy_from_slice(&workspace[..self.rows()]);
                Ok(())
            }
        }
    };
}
matrix_operator!(
    DenseIntervalOperator,
    |s: &DenseIntervalOperator<'_, S>| s.lower.rows(),
    |s: &DenseIntervalOperator<'_, S>| s.lower.cols()
);
matrix_operator!(
    SparseIntervalOperator,
    |s: &SparseIntervalOperator<'_, S>| s.pattern.rows(),
    |s: &SparseIntervalOperator<'_, S>| s.pattern.cols()
);

enum Factor<'a, S: NumericScalar> {
    Dense(Matrix<S>),
    Sparse {
        factor: &'a SparseLu<S>,
        pointers: Vec<usize>,
        entries: Vec<(usize, usize, S)>,
    },
}
/// Preconditioner `R ≈ A⁻¹` from an LU factorization, for
/// [`LinearVerifier`](crate::LinearVerifier).
///
/// [`Self::from_dense`] forms the approximate inverse of a dense LU
/// factorization once and stores it, in `n²` scalars for order `n`; each
/// enclosure then costs O(n²). [`Self::from_sparse`] keeps the sparse factors
/// and applies `R` by forward and back substitution in interval arithmetic,
/// using O(n + F) storage for `F` factor entries and never forming a dense
/// inverse. Interval substitution can overestimate (the wrapping effect), so
/// a sparse preconditioner may fail to prove a system that its dense inverse
/// would prove; no bound on the order or condition number guarantees success.
///
/// The verifier does not trust `R`: a poor approximate inverse makes the proof
/// inconclusive, never wrong. An enclosure needs `n` scratch boxes for a dense
/// and `2n` for a sparse preconditioner.
pub struct LuIntervalInverse<'a, S: NumericScalar> {
    factor: Factor<'a, S>,
}
impl<'a, S: NumericScalar> LuIntervalInverse<'a, S> {
    /// Returns the peak storage of [`Self::from_dense`] for order `order`, in
    /// bytes, including the temporary storage of the inversion. The stored
    /// inverse takes `order² · size_of::<S>()` bytes.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`, including in the
    ///   workspace requirements of the inversion.
    pub fn dense_storage_bytes(order: usize) -> Result<usize> {
        let bytes = scioxide_core::quantity!("dense_storage_bytes");
        let retained = checked_mul(
            checked_mul(order, order, bytes)?,
            core::mem::size_of::<S>(),
            bytes,
        )?;
        let options = LinalgExecutionOptions::new();
        let request = LuWorkspaceRequest::solve_matrix(order, order, &options);
        checked_add(
            retained,
            LuDecomposition::<S>::workspace_requirements(&request)?.required_bytes::<S>()?,
            bytes,
        )
    }
    /// Computes and stores the approximate inverse of the LU factorization
    /// `factor`.
    ///
    /// The peak storage, [`Self::dense_storage_bytes`], must fit the byte limit
    /// of `options`.
    ///
    /// ```
    /// use scioxide_linalg::{LinalgExecutionOptions, LuDecomposition, Matrix};
    /// use scioxide_validated::{LuIntervalInverse, ValidatedOptions};
    ///
    /// let a = Matrix::<f64>::identity(3)?;
    /// let lu = LuDecomposition::factor(&a)?;
    /// let limit = LuIntervalInverse::<f64>::dense_storage_bytes(3)?;
    /// let inverse = LuIntervalInverse::from_dense(&lu, ValidatedOptions::new().with_byte_limit(limit))?;
    /// assert!(
    ///     LuIntervalInverse::from_dense(&lu, ValidatedOptions::new().with_byte_limit(limit - 1)).is_err()
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the peak storage exceeds the byte limit.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be sized or
    ///   reserved.
    /// - Errors of the inversion of `factor`.
    pub fn from_dense(factor: &LuDecomposition<S>, options: ValidatedOptions) -> Result<Self> {
        let n = factor.packed_factors().rows();
        let limit = options.byte_limit();
        crate::shell::within_limit(Self::dense_storage_bytes(n)?, limit)?;
        let mut inverse = Matrix::zeros(n, n)?;
        let retained = checked_mul(
            checked_mul(n, n, scioxide_core::quantity!("dense_inverse"))?,
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("dense_inverse"),
        )?;
        let options = LinalgExecutionOptions::new().with_byte_limit(limit - retained);
        let mut workspace = LinalgWorkspace::new();
        factor.inverse_into(
            &options,
            inverse.view_mut(),
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        Ok(Self {
            factor: Factor::Dense(inverse),
        })
    }
    /// Returns the storage of [`Self::from_sparse`] for `factor`, in bytes,
    /// excluding the borrowed factor itself.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size overflows `usize`.
    pub fn sparse_storage_bytes(factor: &SparseLu<S>) -> Result<usize> {
        const BYTES: Quantity = scioxide_core::quantity!("sparse_storage_bytes");
        let add = |a, b| checked_add(a, b, BYTES);
        let mul = |a, b| checked_mul(a, b, BYTES);
        add(
            mul(
                add(factor.symbolic().dimension(), 1)?,
                core::mem::size_of::<usize>(),
            )?,
            mul(
                factor.factor_entries(),
                core::mem::size_of::<(usize, usize, S)>(),
            )?,
        )
    }
    /// Creates the preconditioner of the sparse LU factorization `factor`,
    /// which it borrows together with its row and column permutations.
    ///
    /// The storage, [`Self::sparse_storage_bytes`], must fit the byte limit of
    /// `options`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the storage exceeds the byte limit.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be sized or
    ///   reserved.
    pub fn from_sparse(factor: &'a SparseLu<S>, options: ValidatedOptions) -> Result<Self> {
        crate::shell::within_limit(Self::sparse_storage_bytes(factor)?, options.byte_limit())?;
        let n = factor.symbolic().dimension();
        let mut entries = scioxide_core::backend::try_with_capacity(
            factor.factor_entries(),
            scioxide_core::quantity!("entries"),
        )?;
        entries.extend(factor.lower_entries().chain(factor.upper_entries()));
        entries.sort_unstable_by_key(|e| (e.0, e.1));
        const POINTERS: Quantity = scioxide_core::quantity!("pointers");
        let mut pointers = try_filled(checked_add(n, 1, POINTERS)?, 0, POINTERS)?;
        for (row, ptr) in pointers.iter_mut().enumerate() {
            *ptr = entries.partition_point(|e| e.0 < row);
        }
        Ok(Self {
            factor: Factor::Sparse {
                factor,
                pointers,
                entries,
            },
        })
    }
}
impl<S: NumericScalar> LinearOperator<S> for LuIntervalInverse<'_, S> {
    fn rows(&self) -> usize {
        match &self.factor {
            Factor::Dense(f) => f.rows(),
            Factor::Sparse { factor, .. } => factor.symbolic().dimension(),
        }
    }
    fn cols(&self) -> usize {
        self.rows()
    }
    fn apply_workspace_len(&self) -> Result<usize> {
        match &self.factor {
            Factor::Dense(f) => Ok(f.rows()),
            Factor::Sparse { factor, .. } => factor.apply_workspace_len(),
        }
    }
    fn apply_into(&self, input: &[S], output: &mut [S], workspace: &mut [S]) -> Result<()> {
        match &self.factor {
            Factor::Dense(f) => DenseIntervalOperator {
                lower: f.view(),
                upper: f.view(),
            }
            .apply_into(input, output, workspace),
            Factor::Sparse { factor, .. } => factor.apply_into(input, output, workspace),
        }
    }
    fn apply_into_scoped(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, S>,
    ) -> Result<scioxide_core::Invocation<()>> {
        match &self.factor {
            Factor::Dense(f) => DenseIntervalOperator {
                lower: f.view(),
                upper: f.view(),
            }
            .apply_into_scoped(input, output, workspace, context),
            Factor::Sparse { factor, .. } => {
                factor.apply_into_scoped(input, output, workspace, context)
            }
        }
    }
}
impl<S: NumericScalar> IntervalLinearOperator<S> for LuIntervalInverse<'_, S> {
    fn enclosure_workspace_len(&self) -> Result<usize> {
        match &self.factor {
            Factor::Dense(matrix) => Ok(matrix.rows()),
            Factor::Sparse { .. } => checked_mul(
                2,
                self.rows(),
                scioxide_core::quantity!("enclosure_workspace_len"),
            ),
        }
    }
    fn apply_enclosure_into(
        &self,
        input: &[IntervalBox<S::Real>],
        addend: Option<&[IntervalBox<S::Real>]>,
        output: &mut [IntervalBox<S::Real>],
        workspace: &mut [IntervalBox<S::Real>],
    ) -> Result<()> {
        let n = self.rows();
        application_shape::<S>(
            n,
            n,
            input,
            addend,
            output,
            workspace,
            self.enclosure_workspace_len()?,
        )?;
        match &self.factor {
            Factor::Dense(matrix) => {
                if addend.is_none() && input.iter().all(|&x| unpack(x).is_zero()) {
                    output.fill(zero()?);
                    return Ok(());
                }
                let column = if addend.is_none() {
                    unit_column(input)
                } else {
                    None
                };
                for (row, slot) in workspace[..n].iter_mut().enumerate() {
                    if let Some(column) = column {
                        *slot = point_enclosure(matrix.view().get(row, column)?)?;
                        continue;
                    }
                    let coefficients = &matrix.as_slice()[row * n..(row + 1) * n];
                    if let Some(value) = point_row(coefficients, input, addend.map(|b| b[row]))? {
                        *slot = value;
                        continue;
                    }
                    let mut sum = BoxSum::new(
                        addend.map_or(Box2::zero()?, |b| unpack(b[row])),
                        scioxide_core::argument!("input"),
                    )?;
                    for (&a, &x) in coefficients.iter().zip(input) {
                        sum.product(Box2::point(a)?, unpack(x), false)?;
                    }
                    *slot = pack(sum.finish()?);
                }
                output.copy_from_slice(&workspace[..n]);
                Ok(())
            }
            Factor::Sparse {
                factor,
                pointers,
                entries,
            } => {
                let (work, staged) = workspace[..2 * n].split_at_mut(n);
                for (i, slot) in work.iter_mut().enumerate() {
                    *slot = input[factor.row_permutation()[i]];
                }
                // Stateful triangular substitutions retain fixed row/column order.
                for i in 0..n {
                    let mut value = unpack(work[i]);
                    for &(_, j, a) in &entries[pointers[i]..pointers[i + 1]] {
                        if j < i {
                            value = value.sub(Box2::point(a)?.mul(unpack(work[j]))?)?;
                        }
                    }
                    work[i] = pack(value);
                }
                for i in (0..n).rev() {
                    let mut value = unpack(work[i]);
                    let mut diagonal = None;
                    for &(_, j, a) in &entries[pointers[i]..pointers[i + 1]] {
                        if j > i {
                            value = value.sub(Box2::point(a)?.mul(unpack(work[j]))?)?;
                        } else if j == i {
                            diagonal = Some(a);
                        }
                    }
                    let d =
                        diagonal.ok_or_else(|| crate::shell::breakdown("missing LU diagonal"))?;
                    if d == S::ZERO {
                        return Err(crate::shell::breakdown("zero LU diagonal"));
                    }
                    work[i] = pack(divide_point(value, d)?);
                }
                for (i, &source) in work.iter().enumerate() {
                    let destination = factor.symbolic().column_permutation().new_to_old()[i];
                    let value = unpack(source);
                    staged[destination] = pack(if let Some(b) = addend {
                        value.add(unpack(b[destination]))?
                    } else {
                        value
                    });
                }
                output.copy_from_slice(staged);
                Ok(())
            }
        }
    }
}

/// Exact point residual with compact integer sums. The exact remainder's sign
/// selects the tight outward endpoints, including subnormal results. Overflowing
/// and nonpoint rows fall back to BoxSum replay.
fn tight_point_row<S: NumericScalar>(
    coefficients: &[S],
    input: &[IntervalBox<S::Real>],
    bias: IntervalBox<S::Real>,
) -> Result<Option<IntervalBox<S::Real>>> {
    fn point<S: NumericScalar>(x: IntervalBox<S::Real>) -> Option<Complex<S::Real>> {
        let (Some(rl), Some(rh)) = x.re.bounds()? else {
            return None;
        };
        let (il, ih) = if S::IS_REAL {
            (S::Real::ZERO, S::Real::ZERO)
        } else {
            let (Some(il), Some(ih)) = x.im.bounds()? else {
                return None;
            };
            (il, ih)
        };
        (rl == rh && il == ih).then_some(Complex::new(rl, il))
    }
    fn endpoints<R: RealScalar>(sum: &mut RealProductSum<R>) -> Result<Option<(R, R)>> {
        let value = match sum.finish() {
            Ok(value) => value,
            Err(error)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability {
                        issue: scioxide_core::NumericalIssue::Overflow,
                        ..
                    }
                ) =>
            {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        sum.add_value(-value)?;
        Ok(Some(match sum.sign() {
            core::cmp::Ordering::Less => (value.next_after(-R::INFINITY), value),
            core::cmp::Ordering::Greater => (value, value.next_after(R::INFINITY)),
            core::cmp::Ordering::Equal => (value, value),
        }))
    }
    let Some(bias) = point::<S>(bias) else {
        return Ok(None);
    };
    let mut re = RealProductSum::new();
    let mut im = RealProductSum::new();
    re.add_value(bias.re)?;
    if !S::IS_REAL {
        im.add_value(bias.im)?;
    }
    for (&a, &x) in coefficients.iter().zip(input) {
        let Some(x) = point::<S>(x) else {
            return Ok(None);
        };
        re.add_product(a.real_component(), x.re)?;
        if !S::IS_REAL {
            re.add_product(-a.imaginary_component(), x.im)?;
            im.add_product(a.real_component(), x.im)?;
            im.add_product(a.imaginary_component(), x.re)?;
        }
    }
    let Some((rl, rh)) = endpoints(&mut re)? else {
        return Ok(None);
    };
    let (il, ih) = if !S::IS_REAL {
        let Some(bounds) = endpoints(&mut im)? else {
            return Ok(None);
        };
        bounds
    } else {
        (S::Real::ZERO, S::Real::ZERO)
    };
    if [rl, rh, il, ih].iter().any(|x| !x.is_finite()) {
        return Ok(None);
    }
    Ok(Some(interval_enclosure(
        S::from_components(rl, il),
        S::from_components(rh, ih),
    )?))
}

/// Native endpoint dot products with a rigorous forward-error allowance.
/// Each component sums at most 2n products. Its error is at most
/// gamma_(2n+1) times the exact absolute product mass plus underflow.
/// With (n+1)*epsilon < 1/64, 8*(n+1)*epsilon times the computed mass
/// covers this bound, the mass reduction and bound-evaluation rounding.
/// 8*(n+1)*eta additionally covers all underflowing products/additions.
/// Nonfinite native ranges replay through exact endpoint accumulation.
fn point_row<S: NumericScalar>(
    coefficients: &[S],
    input: &[IntervalBox<S::Real>],
    bias: Option<IntervalBox<S::Real>>,
) -> Result<Option<IntervalBox<S::Real>>> {
    let count = S::Real::from_f64_ieee((coefficients.len() + 1) as f64);
    let eps = count * S::Real::EPSILON;
    if eps >= S::Real::from_f64_ieee(1.0 / 64.0) {
        return Ok(None);
    }
    let mut sums = [S::Real::ZERO; 4];
    let mut mass = S::Real::ZERO;
    if let Some(b) = bias {
        let (rl, rh) = bounded(unpack(b).re, scioxide_core::argument!("addend"))?;
        let (il, ih) = bounded(unpack(b).im, scioxide_core::argument!("addend"))?;
        sums = [rl, rh, il, ih];
        mass = rl.abs().max(rh.abs()) + il.abs().max(ih.abs());
    }
    for (&a, &x) in coefficients.iter().zip(input) {
        let (rl, rh) = bounded(unpack(x).re, scioxide_core::argument!("input"))?;
        let (il, ih) = bounded(unpack(x).im, scioxide_core::argument!("input"))?;
        let re = a.real_component();
        let im = a.imaginary_component();
        let product = |a: S::Real, lo, hi| {
            if a < S::Real::ZERO {
                (a * hi, a * lo)
            } else {
                (a * lo, a * hi)
            }
        };
        let (ar, br) = product(re, rl, rh);
        let (ai, bi) = product(-im, il, ih);
        sums[0] = (sums[0] + ar) + ai;
        sums[1] = (sums[1] + br) + bi;
        let (ar, br) = product(im, rl, rh);
        let (ai, bi) = product(re, il, ih);
        sums[2] = (sums[2] + ar) + ai;
        sums[3] = (sums[3] + br) + bi;
        mass += (re.abs() + im.abs()) * (rl.abs().max(rh.abs()) + il.abs().max(ih.abs()));
    }
    let eta = S::Real::ZERO.next_after(S::Real::ONE);
    let eight = S::Real::from_f64_ieee(8.0);
    let error = (mass * (eight * eps) + eta * (eight * count)).next_after(S::Real::INFINITY);
    if !error.is_finite() || sums.iter().any(|s| !s.is_finite()) {
        return Ok(None);
    }
    let lo = |x: S::Real| (x - error).next_after(-S::Real::INFINITY);
    let hi = |x: S::Real| (x + error).next_after(S::Real::INFINITY);
    let lower = S::from_components(
        lo(sums[0]),
        if !S::IS_REAL {
            lo(sums[2])
        } else {
            S::Real::ZERO
        },
    );
    let upper = S::from_components(
        hi(sums[1]),
        if !S::IS_REAL {
            hi(sums[3])
        } else {
            S::Real::ZERO
        },
    );
    if !lower.has_finite_components() || !upper.has_finite_components() {
        return Ok(None);
    }
    Ok(Some(interval_enclosure(lower, upper)?))
}
