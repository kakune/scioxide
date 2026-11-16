//! Tridiagonal LU with partial pivoting, and cyclic tridiagonal solves.
//!
//! Partial pivoting on a tridiagonal matrix compares only rows `k` and `k + 1`
//! at step `k`. An interchange brings the superdiagonal entry of row `k + 1`
//! into the pivot row, so `U` gains a second superdiagonal, which is stored.
//! A cyclic tridiagonal matrix is reordered into a band matrix with two
//! subdiagonals and two superdiagonals and factored by band LU.

use alloc::vec::Vec;
use scioxide_core::AlgebraicScalar;

use scioxide_core::backend::try_with_capacity;
use scioxide_core::{
    AdjointOperator, BlockOperator, Dimensions, ExecutionContext, ExecutionPolicy, LinearOperator,
    NumericScalar, ProductSum, RealScalar, SciOxideError,
};

use super::band::BandedLuDecomposition;
use super::band::kernels::multiplier;
use super::direct::{
    StructuredJob, StructuredOperation, StructuredWorkspaceRequest, admit, factor_requirements,
    identity, reserve_capacity, solve_methods, solve_requirements,
};
use crate::checked::{mul, sub};
use crate::kernel::CANDIDATE;
use crate::kernel::standard::substitute_row;
use crate::{
    KernelContext, LinalgExecutionContext, LinalgExecutionOptions, LinalgRequirements,
    LinalgWorkspace, Matrix, MatrixView, MatrixViewMut, Vector, VectorView, VectorViewMut,
};

/// LU factorization of a real tridiagonal matrix for the low-level kernels.
///
/// The rules of the [`backend`](crate::backend) module apply.
///
/// The factorization is Gaussian elimination with partial pivoting between
/// adjacent rows, as in [`TridiagonalLuDecomposition`]. If the matrix is
/// symmetric with nonnegative off-diagonal entries, every diagonal entry is
/// positive and at least twice the sum of the off-diagonal entries in its row,
/// and all entries are well inside the floating-point range, no pivoting is
/// needed: the factor then stores reciprocal pivots and solves with
/// multiplications. The factor keeps no copy of the matrix and computes no
/// diagnostics. Factorization and each solve cost O(n) operations.
///
/// ```
/// use scioxide_core::ExecutionContext;
/// use scioxide_linalg::backend::tridiagonal::TridiagonalFactor;
/// // [2 1; 1 2] x = [3, 3] has the solution x = [1, 1].
/// let execution = ExecutionContext::sequential_native();
/// let factor = TridiagonalFactor::factor(vec![1.0_f64], vec![2.0, 2.0], vec![1.0], execution)?;
/// let mut rhs = [3.0, 3.0];
/// factor.solve_in_place(&mut rhs, execution)?;
/// assert!((rhs[0] - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct TridiagonalFactor<S: RealScalar> {
    lower: Vec<S>,
    main: Vec<S>,
    upper: Vec<S>,
    storage: CompactStorage<S>,
    ready: bool,
}

#[derive(Debug, Clone)]
enum CompactStorage<S: RealScalar> {
    Pivoted {
        second: Vec<S>,
        pivots: Vec<usize>,
    },
    Dominant {
        inverse: Vec<S>,
        minimum: S,
        upper: S,
    },
}

impl<S: RealScalar> TridiagonalFactor<S> {
    /// Computes the factorization of the matrix with diagonals `lower`, `main` and `upper`.
    ///
    /// `lower` is the subdiagonal and `upper` the superdiagonal. The vectors
    /// are taken over and reused as factor storage. `execution` supplies
    /// cancellation and deadlines, checked before and after the elimination;
    /// no iterations are counted. Pass
    /// [`ExecutionContext::sequential_native`] when the call needs neither.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `lower` or `upper` does not have length
    ///   `n − 1` (zero for `n = 0`).
    /// - `NonFinite` if an entry is NaN or infinite, or an entry of the
    ///   factor is not finite.
    /// - `Singular` if a pivot is exactly zero.
    /// - `NumericalInstability` if a multiplier or pivot overflows.
    /// - `AllocationFailure` if the pivot storage cannot be allocated.
    /// - `ExecutionStopped` if `execution` is cancelled or its deadline passes.
    pub fn factor(
        mut lower: Vec<S>,
        mut main: Vec<S>,
        mut upper: Vec<S>,
        execution: ExecutionContext<'_>,
    ) -> Result<Self, SciOxideError> {
        execution.check()?;
        let n = main.len();
        validate_layout(lower.len(), n, upper.len())?;
        let margin = S::TWO * S::TWO * S::TWO * S::TWO * S::TWO * S::TWO;
        let ordinary = |v: S| {
            v == S::ZERO
                || (v.abs() >= S::MIN_POSITIVE.sqrt() * margin && v.abs() <= S::MAX.sqrt() / margin)
        };
        // For symmetric nonnegative off-diagonals and d >= 2 * row_sum,
        // elimination needs no pivot: every remaining pivot is at least half
        // its original diagonal and both substitution ratios are <= 1/2.
        let admitted = n > 0
            && lower == upper
            && lower.iter().all(|&v| v >= S::ZERO && ordinary(v))
            && main.iter().enumerate().all(|(i, &v)| {
                let left = if i > 0 { upper[i - 1] } else { S::ZERO };
                let right = upper.get(i).copied().unwrap_or(S::ZERO);
                v > S::ZERO && ordinary(v) && v >= S::TWO * (left + right)
            });
        let storage = if admitted {
            for i in 0..n - 1 {
                if i % 128 == 0 {
                    execution.check()?;
                }
                lower[i] /= main[i];
                main[i + 1] -= lower[i] * upper[i];
            }
            if lower.iter().chain(&main).any(|v| !v.is_finite()) {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Intermediate,
                    "compact tridiagonal factor is non-finite",
                ));
            }
            let mut inverse =
                try_with_capacity(n, scioxide_core::quantity!("tridiagonal_inverse_pivots"))?;
            inverse.extend(main.iter().map(|&pivot| S::ONE / pivot));
            CompactStorage::Dominant {
                inverse,
                minimum: main.iter().copied().fold(S::MAX, |a, b| a.min(b)),
                upper: upper.iter().copied().fold(S::ZERO, |a, b| a.max(b)),
            }
        } else {
            // Native admission already proves every input finite. Other
            // matrices retain the checked input path before any mutation.
            for values in [&lower, &main, &upper] {
                validate_finite(values)?;
            }
            let mut second = scioxide_core::backend::try_filled(
                n.saturating_sub(2),
                S::ZERO,
                scioxide_core::quantity!("tridiagonal_second"),
            )?;
            let mut pivots = try_with_capacity(n, scioxide_core::quantity!("tridiagonal_pivots"))?;
            pivots.extend(0..n);
            for i in 0..n.saturating_sub(1) {
                if i % 128 == 0 {
                    execution.check()?;
                }
                factor_step(
                    &mut lower,
                    &mut main,
                    &mut upper,
                    &mut second,
                    &mut pivots,
                    i,
                )?;
            }
            if n > 0 {
                ensure_pivot(main[n - 1], n - 1)?;
            }
            CompactStorage::Pivoted { second, pivots }
        };
        execution.check()?;
        Ok(Self {
            lower,
            main,
            upper,
            storage,
            ready: true,
        })
    }

    /// Recomputes the factorization for new diagonals, reusing the storage of `self`.
    ///
    /// This always pivots; it does not use the reciprocal-pivot form of
    /// [`factor`](Self::factor). If it fails, `self` cannot solve until a
    /// later `refactor` succeeds, and solves return `InvalidInput`.
    ///
    /// ```
    /// use scioxide_core::ExecutionContext;
    /// use scioxide_linalg::backend::tridiagonal::TridiagonalFactor;
    /// let execution = ExecutionContext::sequential();
    /// let mut factor = TridiagonalFactor::factor(vec![], vec![1.0_f64], vec![], execution)?;
    /// factor.refactor(&[-1.0], &[2.0, 2.0], &[-1.0], execution)?;
    /// let mut rhs = [1.0, 1.0];
    /// factor.solve_in_place(&mut rhs, execution)?;
    /// assert!((rhs[0] - 1.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    /// # Errors
    /// - `DimensionMismatch` if `lower` or `upper` does not have length `n − 1`.
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - `Singular` if a pivot is exactly zero.
    /// - `NumericalInstability` if a multiplier or pivot overflows.
    /// - `AllocationFailure` if the storage cannot grow.
    /// - `ExecutionStopped` if `execution` is cancelled or its deadline passes.
    pub fn refactor(
        &mut self,
        lower: &[S],
        main: &[S],
        upper: &[S],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let n = main.len();
        validate_layout(lower.len(), n, upper.len())?;
        for values in [lower, main, upper] {
            validate_finite(values)?;
        }
        execution.check()?;
        self.ready = false;
        for (target, source) in [
            (&mut self.lower, lower),
            (&mut self.main, main),
            (&mut self.upper, upper),
        ] {
            reserve_capacity(
                target,
                source.len(),
                scioxide_core::quantity!("tridiagonal_factor"),
            )?;
            target.clear();
            target.extend_from_slice(source);
        }
        if !matches!(self.storage, CompactStorage::Pivoted { .. }) {
            let old = core::mem::replace(
                &mut self.storage,
                CompactStorage::Pivoted {
                    second: Vec::new(),
                    pivots: Vec::new(),
                },
            );
            if let CompactStorage::Dominant { inverse, .. } = old {
                self.storage = CompactStorage::Pivoted {
                    second: inverse,
                    pivots: Vec::new(),
                };
            }
        }
        let CompactStorage::Pivoted { second, pivots } = &mut self.storage else {
            unreachable!("pivot storage prepared above")
        };
        let count = n.saturating_sub(2);
        reserve_capacity(
            second,
            count,
            scioxide_core::quantity!("tridiagonal_second"),
        )?;
        second.clear();
        second.resize(count, S::ZERO);
        reserve_capacity(pivots, n, scioxide_core::quantity!("tridiagonal_pivots"))?;
        pivots.clear();
        pivots.extend(0..n);
        for i in 0..n.saturating_sub(1) {
            if i % 128 == 0 {
                execution.check()?;
            }
            factor_step(
                &mut self.lower,
                &mut self.main,
                &mut self.upper,
                second,
                pivots,
                i,
            )?;
        }
        if n > 0 {
            ensure_pivot(self.main[n - 1], n - 1)?;
        }
        execution.check()?;
        self.ready = true;
        Ok(())
    }

    /// Solves `A x = b` in place: `values` holds `b` on entry and `x` on return.
    ///
    /// `execution` is checked for cancellation and deadlines before the
    /// substitution. On error, `values` may be partly overwritten.
    ///
    /// # Errors
    /// - `InvalidInput` if the last [`refactor`](Self::refactor) failed.
    /// - `DimensionMismatch` if `values` does not have length `n`.
    /// - `NonFinite` if `values` contains NaN or an infinity, or the solution
    ///   is not finite.
    /// - `NumericalInstability` if the substitution overflows.
    /// - `ExecutionStopped` if `execution` is cancelled or its deadline passes.
    pub fn solve_in_place(
        &self,
        values: &mut [S],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if !self.ready {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("self"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "a failed compact refactor must be replaced before solving",
            ));
        }
        execution.check()?;
        if values.len() != self.main.len() {
            return Err(SciOxideError::dimension_mismatch(
                [self.main.len()],
                [values.len()],
            ));
        }
        let mut maximum = S::ZERO;
        for &value in values.iter() {
            if !value.is_finite() {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Input,
                    "tridiagonal RHS is non-finite",
                ));
            }
            maximum = maximum.max(value.abs());
        }
        match &self.storage {
            CompactStorage::Pivoted { second, pivots } => solve_normal_with_execution(
                &self.lower,
                &self.main,
                &self.upper,
                second,
                pivots,
                values,
                execution,
            ),
            CompactStorage::Dominant {
                inverse,
                minimum,
                upper,
            } => {
                let four = S::TWO + S::TWO;
                // Forward growth <= 2; backward growth <= 2/minimum.
                // Four in each bound absorbs scalar and reciprocal rounding.
                let forward = maximum * four;
                let bound = (forward / *minimum) * four;
                let product = bound * *upper;
                let limit = S::MAX / (four * four);
                let native = forward.is_finite()
                    && bound.is_finite()
                    && product.is_finite()
                    && forward <= limit
                    && bound <= limit
                    && product <= limit;
                for i in 0..values.len().saturating_sub(1) {
                    if i % 128 == 0 {
                        execution.check()?;
                    }
                    values[i + 1] = if native {
                        values[i + 1] - self.lower[i] * values[i]
                    } else {
                        substitute_row(
                            values[i + 1],
                            S::ONE,
                            [(self.lower[i], values[i])].into_iter(),
                        )?
                    };
                }
                for i in (0..values.len()).rev() {
                    if i % 128 == 0 {
                        execution.check()?;
                    }
                    let prior = values[i];
                    let residual = if i + 1 < values.len() {
                        prior - self.upper[i] * values[i + 1]
                    } else {
                        prior
                    };
                    let tiny = residual.abs() < S::MIN_POSITIVE && self.main[i] < S::ONE;
                    let recover = tiny
                        && !(prior == S::ZERO
                            && (i + 1 == values.len()
                                || self.upper[i] == S::ZERO
                                || values[i + 1] == S::ZERO));
                    values[i] = if native && !recover {
                        residual * inverse[i]
                    } else {
                        // Reconstruct tiny cancellation or a failed admission
                        // with the original pivots and Standard row arithmetic.
                        substitute_row(
                            prior,
                            self.main[i],
                            (i + 1..(i + 2).min(values.len())).map(|j| (self.upper[i], values[j])),
                        )?
                    };
                }
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(SciOxideError::non_finite(
                        scioxide_core::NonFiniteOrigin::Intermediate,
                        "compact tridiagonal solution is non-finite",
                    ));
                }
                execution.check()
            }
        }
    }
}

/// LU factorization of a real cyclic tridiagonal matrix, for the low-level kernels.
///
/// The rules of the [`backend`](crate::backend) module apply.
///
/// The matrix `A` of order `n ≥ 3` is a tridiagonal matrix plus
/// `corner_upper` at `(0, n − 1)` and `corner_lower` at `(n − 1, 0)`. As in
/// [`CyclicTridiagonalPlan`], the symmetric reordering `0, n − 1, 1, n − 2, …`
/// turns `A` into a band matrix with two subdiagonals and two superdiagonals,
/// which is factored by Gaussian elimination with partial pivoting. This is
/// backward stable for every nonsingular `A`; the Sherman–Morrison formula is
/// not used, because its rank-one split can be ill-conditioned when `A` is
/// well-conditioned. Factorization and each solve cost O(n) operations, and
/// the factor takes O(n) storage. It computes no diagnostics.
///
/// ```
/// use scioxide_linalg::backend::tridiagonal::CyclicTridiagonalFactor;
/// // Every row of A sums to 4 − 1 − 1 = 2, so A x = [2, 2, 2] has x = [1, 1, 1].
/// let factor = CyclicTridiagonalFactor::factor(
///     vec![-1.0_f64; 2],
///     vec![4.0; 3],
///     vec![-1.0; 2],
///     -1.0,
///     -1.0,
/// )?;
/// let mut rhs = [2.0; 3];
/// factor.solve_in_place(&mut rhs)?;
/// assert!(rhs.iter().all(|value| (value - 1.0).abs() < 1e-14));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
/// Hopkins, 2013, §4.3.
#[derive(Debug, Clone)]
pub struct CyclicTridiagonalFactor<S: RealScalar> {
    band: BandedLuDecomposition<S>,
    swaps: Vec<usize>,
}

impl<S: RealScalar> CyclicTridiagonalFactor<S> {
    /// Computes the factorization from the tridiagonal part and the two corner entries.
    ///
    /// `lower`, `main` and `upper` are the diagonals of the tridiagonal part.
    ///
    /// # Errors
    /// - `InvalidInput` if `n < 3`.
    /// - `DimensionMismatch` if `lower` or `upper` does not have length `n − 1`.
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - `Singular` if a pivot of the band elimination is exactly zero.
    /// - `NumericalInstability` if an entry of the band factor overflows.
    /// - `AllocationFailure` if the storage cannot be allocated.
    pub fn factor(
        lower: Vec<S>,
        main: Vec<S>,
        upper: Vec<S>,
        corner_upper: S,
        corner_lower: S,
    ) -> Result<Self, SciOxideError> {
        let matrix = TridiagonalMatrixView::new(&lower, &main, &upper)?;
        validate_finite(&[corner_upper, corner_lower])?;
        let (band, swaps) = fold_cyclic(matrix, corner_upper, corner_lower)?;
        let n = main.len();
        let mut factor = BandedLuDecomposition::empty();
        BandedLuDecomposition::factor_into(
            super::storage::BandedMatrixView::new(n, n, 2, 2, 5, &band)?,
            &LinalgExecutionOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
            &mut factor,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(Self {
            band: factor,
            swaps,
        })
    }

    /// Solves `A x = b` in place: `values` holds `b` on entry and `x` on return.
    ///
    /// On error, `values` may be partly overwritten.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `values` does not have length `n`.
    /// - `NonFinite` if `values` contains NaN or an infinity.
    /// - `NumericalInstability` if an entry of the solution overflows.
    pub fn solve_in_place(&self, values: &mut [S]) -> Result<(), SciOxideError> {
        if values.len() != self.band.order() {
            return Err(SciOxideError::dimension_mismatch(
                [self.band.order()],
                [values.len()],
            ));
        }
        crate::checked::validate_inputs(values)?;
        fold_vector(values, &self.swaps);
        self.band.solve_slice(
            values,
            StructuredOperation::Normal,
            KernelContext::sequential_native(),
        )?;
        unfold_vector(values, &self.swaps);
        Ok(())
    }
}

/// Square tridiagonal matrix stored as its three diagonals.
///
/// For order `n`, `lower` holds the subdiagonal `A[i + 1][i]`, `main` the
/// diagonal `A[i][i]` and `upper` the superdiagonal `A[i][i + 1]`. The
/// diagonal has length `n` and the off-diagonals `n − 1`, so an empty matrix
/// has three empty diagonals and a `1 × 1` matrix has empty off-diagonals.
/// All entries are finite.
///
/// [`TridiagonalMatrixView`] implements [`LinearOperator`] and
/// [`AdjointOperator`].
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct TridiagonalMatrix<S: NumericScalar> {
    lower: Vec<S>,
    main: Vec<S>,
    upper: Vec<S>,
}

impl<S: NumericScalar> TridiagonalMatrix<S> {
    /// Creates a tridiagonal matrix from its subdiagonal, diagonal and superdiagonal.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `lower` or `upper` does not have length
    ///   `main.len() − 1` (zero for an empty `main`).
    /// - `NonFinite` if an entry is NaN or infinite.
    pub fn new(lower: Vec<S>, main: Vec<S>, upper: Vec<S>) -> Result<Self, SciOxideError> {
        validate_layout(lower.len(), main.len(), upper.len())?;
        validate_finite(&lower)?;
        validate_finite(&main)?;
        validate_finite(&upper)?;
        Ok(Self { lower, main, upper })
    }

    /// Returns the `0 × 0` matrix.
    pub const fn empty() -> Self {
        Self {
            lower: Vec::new(),
            main: Vec::new(),
            upper: Vec::new(),
        }
    }

    /// Returns the order `n`.
    pub const fn len(&self) -> usize {
        self.main.len()
    }

    /// Returns `true` if `n = 0`.
    pub const fn is_empty(&self) -> bool {
        self.main.is_empty()
    }

    /// Returns the subdiagonal `A[i + 1][i]`.
    pub fn lower(&self) -> &[S] {
        &self.lower
    }

    /// Returns the diagonal `A[i][i]`.
    pub fn main(&self) -> &[S] {
        &self.main
    }

    /// Returns the superdiagonal `A[i][i + 1]`.
    pub fn upper(&self) -> &[S] {
        &self.upper
    }

    /// Borrows the matrix as a [`TridiagonalMatrixView`].
    pub fn view(&self) -> TridiagonalMatrixView<'_, S> {
        TridiagonalMatrixView {
            lower: &self.lower,
            main: &self.main,
            upper: &self.upper,
        }
    }

    /// Returns the diagonals `(lower, main, upper)`, consuming the matrix.
    pub fn into_diagonals(self) -> (Vec<S>, Vec<S>, Vec<S>) {
        (self.lower, self.main, self.upper)
    }
}

/// Borrowed square tridiagonal matrix with the layout of [`TridiagonalMatrix`].
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct TridiagonalMatrixView<'a, S: NumericScalar> {
    lower: &'a [S],
    main: &'a [S],
    upper: &'a [S],
}

impl<'a, S: NumericScalar> TridiagonalMatrixView<'a, S> {
    /// Creates a view of the tridiagonal matrix with the given three diagonals.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `lower` or `upper` does not have length
    ///   `main.len() − 1` (zero for an empty `main`).
    /// - `NonFinite` if an entry is NaN or infinite.
    pub fn new(lower: &'a [S], main: &'a [S], upper: &'a [S]) -> Result<Self, SciOxideError> {
        validate_layout(lower.len(), main.len(), upper.len())?;
        validate_finite(lower)?;
        validate_finite(main)?;
        validate_finite(upper)?;
        Ok(Self { lower, main, upper })
    }

    /// Returns the order `n`.
    pub const fn len(self) -> usize {
        self.main.len()
    }

    /// Returns `true` if `n = 0`.
    pub const fn is_empty(self) -> bool {
        self.main.is_empty()
    }

    /// Returns the subdiagonal `A[i + 1][i]`.
    pub const fn lower(self) -> &'a [S] {
        self.lower
    }

    /// Returns the diagonal `A[i][i]`.
    pub const fn main(self) -> &'a [S] {
        self.main
    }

    /// Returns the superdiagonal `A[i][i + 1]`.
    pub const fn upper(self) -> &'a [S] {
        self.upper
    }
}

fn apply_tridiagonal<S: NumericScalar>(
    matrix: TridiagonalMatrixView<'_, S>,
    input: &[S],
    output: &mut [S],
    workspace: &mut [S],
    adjoint: bool,
) -> Result<(), SciOxideError> {
    let dimension = matrix.main.len();
    for (_argument, actual) in [
        ("tridiagonal operator input", input.len()),
        ("tridiagonal operator output", output.len()),
        ("tridiagonal operator workspace", workspace.len()),
    ] {
        if actual != dimension {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(dimension),
                Dimensions::one(actual),
            ));
        }
    }
    for (row, slot) in workspace.iter_mut().enumerate() {
        let mut sum = ProductSum::new();
        let diagonal = if adjoint {
            matrix.main[row].conj()
        } else {
            matrix.main[row]
        };
        sum.add_product(diagonal, input[row])?;
        if row > 0 {
            let lower = if adjoint {
                matrix.upper[row - 1].conj()
            } else {
                matrix.lower[row - 1]
            };
            sum.add_product(lower, input[row - 1])?;
        }
        if row + 1 < dimension {
            let upper = if adjoint {
                matrix.lower[row].conj()
            } else {
                matrix.upper[row]
            };
            sum.add_product(upper, input[row + 1])?;
        }
        *slot = sum.finish()?;
    }
    output.copy_from_slice(workspace);
    Ok(())
}

impl<S: NumericScalar> LinearOperator<S> for TridiagonalMatrixView<'_, S> {
    fn rows(&self) -> usize {
        self.main.len()
    }

    fn cols(&self) -> usize {
        self.main.len()
    }

    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.main.len())
    }

    fn apply_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        apply_tridiagonal(*self, input, output, workspace, false)
    }
}

impl<S: NumericScalar> AdjointOperator<S> for TridiagonalMatrixView<'_, S> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.main.len())
    }

    fn apply_adjoint_into(
        &self,
        input: &[S],
        output: &mut [S],
        workspace: &mut [S],
    ) -> Result<(), SciOxideError> {
        apply_tridiagonal(*self, input, output, workspace, true)
    }
}

impl<S: NumericScalar> BlockOperator<S> for TridiagonalMatrixView<'_, S> {}

impl<'a, S: NumericScalar> From<&'a TridiagonalMatrix<S>> for TridiagonalMatrixView<'a, S> {
    fn from(matrix: &'a TridiagonalMatrix<S>) -> Self {
        matrix.view()
    }
}

/// LU factorization of a square tridiagonal matrix with partial pivoting.
///
/// At step `k` the only pivot candidates are the diagonal entry and the
/// subdiagonal entry of row `k + 1`. Rows `k` and `k + 1` are interchanged if
/// the subdiagonal entry is larger in modulus, so every multiplier has
/// modulus at most 1. An interchange brings the superdiagonal entry of row
/// `k + 1` into the pivot row, which gives `U` a second superdiagonal. The
/// factorization stores
///
/// - `U`: its diagonal, its superdiagonal and its
///   [`second_superdiagonal`](Self::second_superdiagonal) of length `n − 2`;
/// - the multipliers `lₖ` in elimination order, as in
///   [`BandedLuDecomposition`](crate::BandedLuDecomposition);
/// - the [`pivots`](Self::pivots): `pivots()[k]` is `k` or `k + 1`.
///
/// Factorization costs O(n) operations and storage, and a solve O(n) per
/// right-hand side. Factorization fails only on an exactly zero pivot or when
/// an entry of `U` overflows; see [Direct solves](crate#direct-solves). A
/// multiplier below the normal range rounds to a subnormal number or to zero.
/// Substitution rescales by powers of two to avoid
/// overflow, as in [`TriangularPlan`](crate::TriangularPlan).
/// For a cyclic tridiagonal matrix, use [`CyclicTridiagonalPlan`].
///
/// # Examples
///
/// ```
/// use scioxide_linalg::{StructuredOperation, TridiagonalLuDecomposition, TridiagonalMatrix};
///
/// // The second-difference matrix tridiag(−1, 2, −1) maps [1, 1, 1] to [1, 0, 1].
/// let matrix = TridiagonalMatrix::new(
///     vec![-1.0, -1.0],
///     vec![2.0, 2.0, 2.0],
///     vec![-1.0, -1.0],
/// )?;
/// let factor = TridiagonalLuDecomposition::factor(&matrix)?;
/// let solution = factor
///     .solve(
///         StructuredOperation::Normal,
///         &[1.0, 0.0, 1.0][..])?;
/// assert!(solution.as_slice().iter().all(|&value| (value - 1.0_f64).abs() < 1.0e-14));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
/// Hopkins, 2013, §4.3.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct TridiagonalLuDecomposition<S: NumericScalar> {
    lower: Vec<S>,
    main: Vec<S>,
    upper: Vec<S>,
    second_upper: Vec<S>,
    pivots: Vec<usize>,
    exact_scale: S::Real,
    source: super::direct::Source<S>,
}

impl<S: NumericScalar> Default for TridiagonalLuDecomposition<S> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<S: NumericScalar> TridiagonalLuDecomposition<S> {
    /// Returns the factorization of a `0 × 0` matrix.
    ///
    /// Use it as reusable output of [`factor_into`](Self::factor_into).
    pub const fn empty() -> Self {
        Self {
            lower: Vec::new(),
            main: Vec::new(),
            upper: Vec::new(),
            second_upper: Vec::new(),
            pivots: Vec::new(),
            exact_scale: S::Real::ZERO,
            source: super::direct::Source::empty(),
        }
    }

    /// Returns the order `n`.
    pub const fn order(&self) -> usize {
        self.main.len()
    }

    /// Returns `true` if `n = 0`.
    pub const fn is_empty(&self) -> bool {
        self.main.is_empty()
    }

    /// Returns the pivot rows: step `k` interchanged rows `k` and `pivots()[k]`,
    /// which is `k` or `k + 1`.
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }

    /// Returns the second superdiagonal `U[k][k + 2]` of `U`, of length `n − 2`.
    pub fn second_superdiagonal(&self) -> &[S] {
        &self.second_upper
    }

    /// Returns the workspace size for a factor or solve `request`.
    ///
    /// A factor request must have `lower = upper = 1`; it needs about `4n`
    /// scalars for the factor diagonals and `n` indices. A solve request needs
    /// `n m` scalars for `m` right-hand sides, and the scratch of a condition
    /// estimate.
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    ///
    /// # Errors
    /// - `InvalidInput` if a factor request has `lower` or `upper` other than 1.
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `ResourceLimit` if the workspace exceeds the byte limit of `request`.
    pub fn workspace_requirements(
        request: &StructuredWorkspaceRequest,
    ) -> Result<LinalgRequirements, SciOxideError> {
        match request.job {
            StructuredJob::Factor {
                order,
                lower: 1,
                upper: 1,
            } => {
                let off = order.saturating_sub(1);
                let scalars = [off, order, off, order.saturating_sub(2)]
                    .into_iter()
                    .try_fold(0usize, |total, len| {
                        total.checked_add(len).ok_or_else(|| {
                            SciOxideError::size_overflow(scioxide_core::quantity!(
                                "tridiagonal_factor"
                            ))
                        })
                    })?;
                factor_requirements::<S>(scalars, order, request.byte_limit)
            }
            StructuredJob::Factor { .. } => Err(SciOxideError::invalid_input(
                scioxide_core::argument!("request"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "a tridiagonal factor has unit bandwidths",
            )),
            StructuredJob::Solve { order, columns } => {
                solve_requirements::<S>(order, columns, request.byte_limit)
            }
        }
    }

    /// Allocates a workspace for a factor or solve `request`.
    ///
    /// ```
    /// use scioxide_core::Capacity;
    /// use scioxide_linalg::{
    ///     LinalgExecutionOptions, StructuredWorkspaceRequest, TridiagonalLuDecomposition,
    /// };
    /// let request = StructuredWorkspaceRequest::solve_matrix(16, 4, &LinalgExecutionOptions::new());
    /// let workspace = TridiagonalLuDecomposition::<f64>::workspace(&request)?;
    /// assert_eq!(
    ///     workspace.check(&TridiagonalLuDecomposition::<f64>::workspace_requirements(&request)?)?,
    ///     Capacity::Sufficient
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    /// The errors of [`workspace_requirements`](Self::workspace_requirements),
    /// and `AllocationFailure` if the workspace cannot be allocated.
    pub fn workspace(
        request: &StructuredWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Computes the LU factorization of a square tridiagonal matrix.
    ///
    /// See [Execution](crate#execution) for how this form runs.
    ///
    /// # Errors
    /// The same conditions as [`factor_into`](Self::factor_into).
    pub fn factor<'m>(
        matrix: impl Into<TridiagonalMatrixView<'m, S>>,
    ) -> Result<Self, SciOxideError> {
        Self::factor_with_options(matrix, &crate::LinalgExecutionOptions::default())
    }

    /// Runs [`factor`](Self::factor) with `options`, which set the product accuracy and the limits
    /// described in [Execution](crate#execution).
    ///
    /// # Errors
    ///
    /// The errors of [`factor`](Self::factor), and limit errors from `options` as described in
    /// [Execution](crate#execution).
    pub fn factor_with_options<'m>(
        matrix: impl Into<TridiagonalMatrixView<'m, S>>,
        options: &LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::factor_into(
            matrix,
            &options.with_execution_policy(ExecutionPolicy::Sequential),
            &mut output,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Computes the LU factorization of `matrix` into `output`, reusing its storage.
    ///
    /// The scratch is described by a factor request with `lower = upper = 1`.
    /// The storage of `output` grows if needed. Each pivot counts as one
    /// iteration toward the iteration budget in `options`.
    ///
    /// ```
    /// use scioxide_linalg::{
    ///     LinalgExecutionOptions, LinalgWorkspace, TridiagonalLuDecomposition, TridiagonalMatrixView,
    /// };
    /// let a = TridiagonalMatrixView::new(&[1.0], &[1.0, 2.0], &[1.0])?;
    /// let mut factor = TridiagonalLuDecomposition::empty();
    /// let mut workspace = LinalgWorkspace::new();
    /// TridiagonalLuDecomposition::factor_into(
    ///     a,
    ///     &LinalgExecutionOptions::new(),
    ///     &mut factor,
    ///     &mut workspace,
    ///     scioxide_core::ExecutionContext::sequential_native(),
    /// )?;
    /// assert_eq!(factor.order(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    /// - `Singular` if a pivot is exactly zero; the error carries its index.
    /// - `NumericalInstability` if a multiplier or pivot overflows.
    /// - `ConvergenceFailure` if the iteration budget runs out.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces), and
    ///   `AllocationFailure` if the storage of `output` cannot grow.
    pub fn factor_into<'m, 'scope, 'capability>(
        matrix: impl Into<TridiagonalMatrixView<'m, S>>,
        options: &LinalgExecutionOptions,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        workspace.begin_attempt();
        let context = context.into().with_options(options)?;
        context.execution().check()?;
        let matrix = matrix.into();
        let n = matrix.len();
        let off = n.saturating_sub(1);
        let second = n.saturating_sub(2);
        let requirements =
            Self::workspace_requirements(&StructuredWorkspaceRequest::factor(n, 1, 1, options))?;
        workspace.try_reserve(&requirements)?;
        for (values, len) in [
            (&mut output.lower, off),
            (&mut output.main, n),
            (&mut output.upper, off),
            (&mut output.second_upper, second),
        ] {
            reserve_capacity(values, len, scioxide_core::quantity!("factor.diagonals"))?;
        }
        reserve_capacity(
            &mut output.pivots,
            n,
            scioxide_core::quantity!("factor.pivots"),
        )?;
        output.source.reserve(n, 1, 1)?;
        let element = |i: usize, j: usize| {
            if i == j {
                matrix.main[i]
            } else if i > j {
                matrix.lower[j]
            } else {
                matrix.upper[i]
            }
        };
        let norm = crate::diagnostics::one_norm(crate::diagnostics::Band::new(n, 1, 1), element)?;
        workspace.run_scope(
            context,
            identity(
                scioxide_core::algorithm_id!("linalg", "tridiagonal_lu", 1),
                n,
            ),
            options.solve_budgets(),
            |workspace, kernels, scope| {
                workspace.with_direct(|storage, _, indices, _| {
                    let (lower, rest) = crate::workspace::split_mut(storage, off)?;
                    let (main, rest) = crate::workspace::split_mut(rest, n)?;
                    let (upper, rest) = crate::workspace::split_mut(rest, off)?;
                    let second_upper = &mut rest[..second];
                    let pivots = &mut indices[..n];
                    lower.copy_from_slice(matrix.lower);
                    main.copy_from_slice(matrix.main);
                    upper.copy_from_slice(matrix.upper);
                    second_upper.fill(S::ZERO);
                    pivots.iter_mut().enumerate().for_each(|(i, p)| *p = i);
                    let execution = kernels.execution();
                    for index in 0..off {
                        admit(execution, scope)?;
                        factor_step::<S>(lower, main, upper, second_upper, pivots, index)?;
                    }
                    if n >= 1 {
                        admit(execution, scope)?;
                        ensure_pivot(main[n - 1], n - 1)?;
                    }
                    // A successful factor has no zero pivots. Retain its Exact
                    // diagnostic once, rather than rescanning on every solve.
                    let exact_scale = if S::IS_REAL {
                        main.iter().fold(S::Real::ZERO, |scale, v| {
                            scale.max(v.real_component().abs())
                        })
                    } else {
                        crate::analysis::analyze_scaled_magnitudes(
                            main.iter()
                                .map(|&x| scioxide_core::ScaledValue::from_value(x)?.magnitude()),
                            n,
                            crate::RankPolicy::Exact,
                        )?
                        .scale()
                    };
                    execution.check()?;
                    for (target, source) in [
                        (&mut output.lower, &*lower),
                        (&mut output.main, &*main),
                        (&mut output.upper, &*upper),
                        (&mut output.second_upper, &*second_upper),
                    ] {
                        target.clear();
                        target.extend_from_slice(source);
                    }
                    output.source.publish(n, 1, 1, norm, |col| {
                        (col.saturating_sub(1)..(col + 2).min(n)).map(move |row| element(row, col))
                    });
                    output.pivots.clear();
                    output.pivots.extend_from_slice(pivots);
                    output.exact_scale = exact_scale;
                    Ok(())
                })
            },
        )
    }

    solve_methods!(
        TridiagonalLuDecomposition,
        order,
        solve_slice,
        "use scioxide_linalg::{LinalgExecutionOptions, StructuredOperation, TridiagonalLuDecomposition, TridiagonalMatrixView};
let a = TridiagonalMatrixView::new(&[1.0], &[1.0, 2.0], &[1.0])?;
let factor = TridiagonalLuDecomposition::factor(a)?;"
    );

    fn solve_slice(
        &self,
        x: &mut [S],
        op: StructuredOperation,
        _: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.solve_in_place(x, op)
    }

    fn solve_in_place(
        &self,
        values: &mut [S],
        op: StructuredOperation,
    ) -> Result<(), SciOxideError> {
        match op {
            StructuredOperation::Normal => solve_normal::<S>(
                &self.lower,
                &self.main,
                &self.upper,
                &self.second_upper,
                &self.pivots,
                values,
            ),
            StructuredOperation::Transpose => self.solve_transposed(values, false),
            StructuredOperation::Adjoint => self.solve_transposed(values, true),
        }
    }

    fn solve_transposed(&self, values: &mut [S], conjugate: bool) -> Result<(), SciOxideError> {
        // `A^T = U^T L^T P`: substitute U first, then undo L and its swaps.
        let n = self.order();
        let coefficient = |value: S| if conjugate { value.conj() } else { value };
        for index in 0..n {
            values[index] = crate::kernel::standard::substitute_row(
                values[index],
                coefficient(self.main[index]),
                (index.saturating_sub(2)..index).map(|j| {
                    (
                        coefficient(if j + 1 == index {
                            self.upper[j]
                        } else {
                            self.second_upper[j]
                        }),
                        values[j],
                    )
                }),
            )?;
        }
        for index in (0..n.saturating_sub(1)).rev() {
            values[index] = crate::kernel::standard::substitute_row(
                values[index],
                S::ONE,
                [(coefficient(self.lower[index]), values[index + 1])].into_iter(),
            )?;
            if self.pivots[index] != index {
                values.swap(index, index + 1);
            }
        }
        Ok(())
    }
}

/// Cyclic tridiagonal system `op(A) x = b`, prepared for repeated O(n) solves.
///
/// `A` has order `n ≥ 3`. It is a tridiagonal matrix plus `corner_upper` at
/// `(0, n − 1)` and `corner_lower` at `(n − 1, 0)`, as in finite differences
/// with periodic boundary conditions. The symmetric reordering
/// `0, n − 1, 1, n − 2, …` turns `A` into a band matrix with two subdiagonals
/// and two superdiagonals, and preparation factors it with
/// [`BandedLuDecomposition`](crate::BandedLuDecomposition), in O(n)
/// operations and storage. Gaussian elimination with partial pivoting is
/// backward stable for every nonsingular `A`, unlike the Sherman–Morrison
/// formula, whose rank-one split can be ill-conditioned when `A` is not. A
/// solve permutes the right-hand side, substitutes with the band factors and
/// permutes back, in O(n) operations per right-hand side. The same factor
/// gives the rank, the condition estimate and the residuals of `A`.
///
/// For real scalars, an adjoint solve is a transposed solve. A rejected solve
/// returns `Ok` with no solution; see [Direct solves](crate#direct-solves).
///
/// ```
/// use scioxide_linalg::{
///     CyclicTridiagonalPlan, LinalgExecutionOptions, StructuredOperation, TridiagonalMatrixView,
/// };
/// // Every row of A sums to 4 − 1 − 1 = 2, so A x = [2, 2, 2] has x = [1, 1, 1].
/// let ordinary = TridiagonalMatrixView::new(&[-1.0; 2], &[4.0; 3], &[-1.0; 2])?;
/// let plan = CyclicTridiagonalPlan::new(ordinary, -1.0, -1.0, &LinalgExecutionOptions::new())?;
/// let x = plan.solve(StructuredOperation::Normal, &[2.0, 2.0, 2.0][..])?;
/// assert!(x.as_slice().iter().all(|value| (value - 1.0_f64).abs() < 1e-12));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
/// G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
/// Hopkins, 2013, §4.3 and §4.7.
#[derive(Debug, Clone)]
#[must_use]
pub struct CyclicTridiagonalPlan<S: NumericScalar> {
    band: BandedLuDecomposition<S>,
    swaps: Vec<usize>,
    byte_limit: Option<usize>,
}

impl<S: NumericScalar> CyclicTridiagonalPlan<S> {
    /// Prepares a plan for a cyclic tridiagonal matrix.
    ///
    /// `matrix` is the tridiagonal part, and the corners are the entries at
    /// `(0, n − 1)` and `(n − 1, 0)`.
    ///
    /// Preparation runs sequentially; see [Execution](crate#execution). The
    /// plan keeps the byte limit of `options`; solves apply the smaller of it
    /// and their own limit. Other options are given per solve.
    ///
    /// # Errors
    /// The same conditions as [`build`](Self::build).
    pub fn new<'m>(
        matrix: impl Into<TridiagonalMatrixView<'m, S>>,
        corner_upper: S,
        corner_lower: S,
        options: &LinalgExecutionOptions,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            matrix,
            corner_upper,
            corner_lower,
            &options.with_execution_policy(ExecutionPolicy::Sequential),
            ExecutionContext::sequential_native(),
        )
    }

    /// Prepares a plan for a cyclic tridiagonal matrix in an execution context.
    ///
    /// Each pivot column of the band factorization counts as one iteration
    /// toward the iteration budget in `options`.
    ///
    /// ```
    /// use scioxide_linalg::{CyclicTridiagonalPlan, LinalgExecutionOptions, TridiagonalMatrixView};
    /// let ordinary = TridiagonalMatrixView::new(&[-1.0; 2], &[4.0; 3], &[-1.0; 2])?;
    /// let plan = CyclicTridiagonalPlan::build(
    ///     ordinary,
    ///     -1.0,
    ///     -1.0,
    ///     &LinalgExecutionOptions::new(),
    ///     scioxide_core::ExecutionContext::sequential_native(),
    /// )?;
    /// assert_eq!(plan.len(), 3);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    /// - `InvalidInput` if `n < 3`.
    /// - `NonFinite` if a corner entry is NaN or infinite.
    /// - `Singular` if the band LU of the reordered matrix meets an exactly
    ///   zero pivot.
    /// - `NumericalInstability` if an entry of the band factor overflows.
    /// - `ConvergenceFailure` if the iteration budget runs out.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors of the factorization, as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn build<'m, 'scope, 'capability>(
        matrix: impl Into<TridiagonalMatrixView<'m, S>>,
        corner_upper: S,
        corner_lower: S,
        options: &LinalgExecutionOptions,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        let matrix = matrix.into();
        validate_finite(&[corner_upper, corner_lower])?;
        let (band, swaps) = fold_cyclic(matrix, corner_upper, corner_lower)?;
        let mut factor = BandedLuDecomposition::empty();
        BandedLuDecomposition::factor_into(
            super::storage::BandedMatrixView::new(matrix.len(), matrix.len(), 2, 2, 5, &band)?,
            options,
            &mut factor,
            &mut LinalgWorkspace::new(),
            context,
        )?;
        Ok(Self {
            band: factor,
            swaps,
            byte_limit: options.byte_limit(),
        })
    }

    /// Returns the order `n`.
    pub const fn len(&self) -> usize {
        self.band.order()
    }

    /// Returns `false`: a plan always has order at least 3.
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Returns the workspace size for a solve `request`.
    ///
    /// The workspace holds the `m` right-hand sides, two work vectors for the
    /// residuals and the scratch of a condition estimate. The byte limit is the smaller of the
    /// limits of `request` and of the plan.
    ///
    /// ```
    /// use scioxide_linalg::{
    ///     CyclicTridiagonalPlan, LinalgExecutionOptions, StructuredWorkspaceRequest,
    ///     TridiagonalMatrixView,
    /// };
    /// let options = LinalgExecutionOptions::new();
    /// let plan = CyclicTridiagonalPlan::new(
    ///     TridiagonalMatrixView::new(&[-1.0; 2], &[4.0; 3], &[-1.0; 2])?,
    ///     -1.0,
    ///     -1.0,
    ///     &options,
    /// )?;
    /// let requirements =
    ///     plan.workspace_requirements(&StructuredWorkspaceRequest::solve_matrix(3, 2, &options))?;
    /// assert_eq!(requirements.scalar_len()?, 18);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    /// - `InvalidInput` if `request` is a factor request.
    /// - `DimensionMismatch` if `request` is for another order.
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `ResourceLimit` if the workspace exceeds the byte limit.
    pub fn workspace_requirements(
        &self,
        request: &StructuredWorkspaceRequest,
    ) -> Result<LinalgRequirements, SciOxideError> {
        let (columns, byte_limit) = request.plan_solve(self.len(), self.byte_limit)?;
        self.requirements(columns, byte_limit)
    }

    /// Allocates a workspace for a solve `request`.
    ///
    /// # Errors
    /// The errors of [`workspace_requirements`](Self::workspace_requirements),
    /// and `AllocationFailure` if the workspace cannot be allocated.
    pub fn workspace(
        &self,
        request: &StructuredWorkspaceRequest,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    fn requirements(
        &self,
        columns: usize,
        byte_limit: Option<usize>,
    ) -> Result<LinalgRequirements, SciOxideError> {
        let staged = columns.checked_add(2).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("structured_rhs"))
        })?;
        solve_requirements::<S>(self.len(), staged, byte_limit)
    }

    /// Solves `op(A) x = b` and returns `x`.
    ///
    /// The solve uses the default [`DirectSolveOptions`](crate::DirectSolveOptions):
    /// an exactly zero pivot of the factorization rejects it.
    ///
    /// # Errors
    /// - `Singular` if a pivot is exactly zero.
    /// - The other conditions of [`solve_matrix_into`](Self::solve_matrix_into).
    pub fn solve<'b>(
        &self,
        operation: StructuredOperation,
        rhs: impl Into<VectorView<'b, S>>,
    ) -> Result<Vector<S>, SciOxideError> {
        self.solve_with_options(operation, rhs, &crate::DirectSolveOptions::default())?
            .into_accepted()
    }

    /// Solves `op(A) x = b`, checks the result against `options` and returns it with its
    /// [`DirectSolveReport`](crate::DirectSolveReport).
    ///
    /// A solve that the checks reject returns `Ok` without a solution; see
    /// [Direct solves](crate#direct-solves).
    ///
    /// # Errors
    /// The same conditions as [`solve_matrix_into`](Self::solve_matrix_into).
    pub fn solve_with_options<'b>(
        &self,
        operation: StructuredOperation,
        rhs: impl Into<VectorView<'b, S>>,
        options: &crate::DirectSolveOptions<S::Real>,
    ) -> Result<crate::DirectSolveResult<Vector<S>, S::Real>, SciOxideError> {
        let options = options.with_execution_options(
            options
                .execution_options()
                .with_execution_policy(ExecutionPolicy::Sequential),
        );
        let mut output = Vector::zeros(self.len())?;
        let report = self.solve_into(
            operation,
            rhs,
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(crate::DirectSolveResult::new(output, report))
    }

    /// Solves `op(A) x = b` into `output`, using `workspace` for scratch.
    ///
    /// This is [`solve_matrix_into`](Self::solve_matrix_into) with one
    /// right-hand side.
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    ///
    /// # Errors
    /// The same conditions as [`solve_matrix_into`](Self::solve_matrix_into).
    pub fn solve_into<'b, 'scope, 'capability>(
        &self,
        operation: StructuredOperation,
        rhs: impl Into<VectorView<'b, S>>,
        options: &crate::DirectSolveOptions<S::Real>,
        output: VectorViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<crate::DirectSolveReport<S::Real>, SciOxideError> {
        self.solve_matrix_into(
            operation,
            crate::MatrixView::from(rhs.into()),
            options,
            crate::MatrixViewMut::from(output),
            workspace,
            context,
        )
    }

    /// Solves `op(A) X = B` for all columns of `B` and returns `X`.
    ///
    /// The solve uses the default [`DirectSolveOptions`](crate::DirectSolveOptions),
    /// as [`solve`](Self::solve) does.
    ///
    /// # Errors
    /// - `Singular` if a pivot is exactly zero.
    /// - The other conditions of [`solve_matrix_into`](Self::solve_matrix_into).
    pub fn solve_matrix<'b>(
        &self,
        operation: StructuredOperation,
        rhs: impl Into<MatrixView<'b, S>>,
    ) -> Result<Matrix<S>, SciOxideError> {
        self.solve_matrix_with_options(operation, rhs, &crate::DirectSolveOptions::default())?
            .into_accepted()
    }

    /// Solves as [`solve_matrix`](Self::solve_matrix) does, checks the solution against `options`
    /// and returns it with its [`DirectSolveReport`](crate::DirectSolveReport).
    ///
    /// A solve that the checks reject returns `Ok` without a solution; see [Direct
    /// solves](crate#direct-solves).
    ///
    /// # Errors
    ///
    /// The errors of [`solve_matrix`](Self::solve_matrix) other than a rejection, and limit errors
    /// from `options` as described in [Execution](crate#execution).
    pub fn solve_matrix_with_options<'b>(
        &self,
        operation: StructuredOperation,
        rhs: impl Into<MatrixView<'b, S>>,
        options: &crate::DirectSolveOptions<S::Real>,
    ) -> Result<crate::DirectSolveResult<Matrix<S>, S::Real>, SciOxideError> {
        let options = options.with_execution_options(
            options
                .execution_options()
                .with_execution_policy(ExecutionPolicy::Sequential),
        );
        let rhs = rhs.into();
        let mut output = Matrix::zeros(self.len(), rhs.cols())?;
        let report = self.solve_matrix_into(
            operation,
            rhs,
            &options,
            output.view_mut(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(crate::DirectSolveResult::new(output, report))
    }

    /// Solves `op(A) X = B` into `output`, with one rank and condition decision for all columns.
    ///
    /// The rank comes from [`rank`](Self::rank) and the condition test from
    /// [`condition_estimate_scoped`](Self::condition_estimate_scoped), both for
    /// `A` itself whatever `operation` is. If `options` requests a residual,
    /// the report holds the largest normwise backward error of `op(A) x = b`
    /// over the columns. Each column counts as one iteration toward the
    /// iteration budget. With a parallel execution policy, columns may be
    /// solved in parallel.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `B` does not have `n` rows or `output` does
    ///   not have the shape of `B`.
    /// - `NonFinite` if `B` contains NaN or an infinity.
    /// - `NumericalInstability` if an entry of the solution overflows.
    /// - `ConvergenceFailure` if the iteration budget runs out.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn solve_matrix_into<'b, 'scope, 'capability>(
        &self,
        operation: StructuredOperation,
        rhs: impl Into<MatrixView<'b, S>>,
        options: &crate::DirectSolveOptions<S::Real>,
        mut output: MatrixViewMut<'_, S>,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<crate::DirectSolveReport<S::Real>, SciOxideError> {
        workspace.begin_attempt();
        let execution_options = self.execution_options(&options.execution_options());
        let context = context.into().with_options(&execution_options)?;
        let kernels = context.kernel();
        kernels.execution().check()?;
        let n = self.len();
        let rhs = rhs.into();
        let columns = rhs.cols();
        crate::checked::check_block_shape(n, rhs, &output)?;
        let requirements = self.requirements(columns, execution_options.byte_limit())?;
        let rhs = rhs.try_finite_dispatched(kernels.dispatch())?.as_view();
        workspace.try_reserve(&requirements)?;
        let operation = if S::IS_REAL && operation == StructuredOperation::Adjoint {
            StructuredOperation::Transpose
        } else {
            operation
        };
        let policy = execution_options.execution_policy();
        workspace.run_scope(
            context,
            identity(
                scioxide_core::algorithm_id!("linalg", "cyclic_tridiagonal_solve", 1),
                n,
            ),
            execution_options.solve_budgets(),
            |workspace, kernels, scope| {
                let rank = self.rank(options.rank_policy())?;
                let condition = options
                    .minimum_reciprocal_condition()
                    .map(|_| {
                        self.condition_estimate_scoped(
                            &execution_options,
                            workspace,
                            kernels.scoped(&mut *scope),
                        )
                    })
                    .transpose()?;
                let report = crate::analysis::acceptance(rank, condition, options);
                if report.termination() != crate::DirectSolveTermination::Accepted || columns == 0 {
                    return Ok(report);
                }
                let execution = kernels.execution();
                let (scratch, staged) = crate::workspace::split_mut(
                    &mut workspace.scalar_mut(CANDIDATE)[..n * (columns + 2)],
                    2 * n,
                )?;
                let (folded, folded_rhs) = crate::workspace::split_mut(scratch, n)?;
                for (j, column) in staged.chunks_exact_mut(n).enumerate() {
                    admit(execution, scope)?;
                    for (i, slot) in column.iter_mut().enumerate() {
                        *slot = rhs.get_element(i, j);
                    }
                }
                solve_columns(staged, n, policy, execution, &|column| {
                    fold_vector(column, &self.swaps);
                    self.band.solve_slice(column, operation, kernels)?;
                    unfold_vector(column, &self.swaps);
                    Ok(())
                })?;
                let mut residual = None;
                if options.residual() == crate::Estimation::Estimate {
                    for (j, column) in staged.chunks_exact(n).enumerate() {
                        folded.copy_from_slice(column);
                        for (i, value) in folded_rhs.iter_mut().enumerate() {
                            *value = rhs.get_element(i, j);
                        }
                        fold_vector(folded, &self.swaps);
                        fold_vector(folded_rhs, &self.swaps);
                        crate::analysis::retain_largest_residual(
                            &mut residual,
                            self.band.residual(
                                operation,
                                folded,
                                VectorView::from_slice(folded_rhs),
                                kernels,
                            )?,
                        );
                    }
                }
                execution.check()?;
                for (j, column) in staged.chunks_exact(n).enumerate() {
                    for (i, &value) in column.iter().enumerate() {
                        output.set_element(i, j, value);
                    }
                }
                Ok(crate::DirectSolveReport::new(
                    report.termination(),
                    rank,
                    condition,
                    residual,
                ))
            },
        )
    }

    /// Returns the numerical rank of `A` under `policy`, from the pivots of its band LU.
    ///
    /// ```
    /// use scioxide_linalg::{
    ///     CyclicTridiagonalPlan, LinalgExecutionOptions, RankPolicy, TridiagonalMatrixView,
    /// };
    /// let matrix = TridiagonalMatrixView::new(&[-1.0; 2], &[4.0; 3], &[-1.0; 2])?;
    /// let plan = CyclicTridiagonalPlan::new(matrix, -1.0, -1.0, &LinalgExecutionOptions::new())?;
    /// assert!(plan.rank(RankPolicy::Exact)?.is_full_rank());
    /// assert!(plan.condition_estimate()?.reciprocal().value() > 0.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    /// - `NumericalInstability` if the rank threshold of `policy` overflows.
    pub fn rank(
        &self,
        policy: crate::RankPolicy<S::Real>,
    ) -> Result<crate::RankReport<S::Real>, SciOxideError> {
        self.band.rank(policy)
    }

    /// Estimates the reciprocal condition number `rcond₁(A) = 1 / (‖A‖₁ ‖A⁻¹‖₁)`.
    ///
    /// `‖A⁻¹‖₁` is estimated from the band LU with Hager's method as refined
    /// by Higham. The symmetric reordering does not change `‖A‖₁` or
    /// `‖A⁻¹‖₁`. The estimate is not a bound.
    ///
    /// # Errors
    /// The same conditions as
    /// [`condition_estimate_scoped`](Self::condition_estimate_scoped).
    ///
    /// # References
    /// - W. W. Hager, "Condition estimates", *SIAM J. Sci. Stat. Comput.* 5
    ///   (1984), 311–316.
    /// - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*,
    ///   2nd ed., SIAM, 2002, Ch. 15.
    pub fn condition_estimate(&self) -> Result<crate::ConditionEstimate<S::Real>, SciOxideError> {
        self.condition_estimate_scoped(
            &LinalgExecutionOptions::new(),
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )
    }

    /// Estimates `rcond₁(A)` with a workspace and an execution context.
    ///
    /// The method is that of [`condition_estimate`](Self::condition_estimate).
    ///
    /// A workspace reserved for a solve request is large enough. The byte
    /// limit is the smaller of the plan's and that of `options`. Each
    /// application of `A⁻¹` or `A⁻ᴴ` counts as one iteration.
    ///
    /// # Errors
    /// - `ConvergenceFailure` if the iteration budget runs out.
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    pub fn condition_estimate_scoped<'scope, 'capability>(
        &self,
        options: &LinalgExecutionOptions,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<LinalgExecutionContext<'scope, 'capability>>,
    ) -> Result<crate::ConditionEstimate<S::Real>, SciOxideError> {
        self.band
            .condition_estimate_scoped(&self.execution_options(options), workspace, context)
    }

    fn execution_options(&self, options: &LinalgExecutionOptions) -> LinalgExecutionOptions {
        self.byte_limit
            .into_iter()
            .chain(options.byte_limit())
            .min()
            .map_or(*options, |limit| options.with_byte_limit(limit))
    }
}

// Fold the cycle into a path with edges of distance at most two. This also
// handles nonsingular cyclic matrices for which every rank-one split is singular.
// Returns the column-major band storage of the reordered matrix, five entries
// per column, and the interchanges that reorder a vector in place.
fn fold_cyclic<S: NumericScalar>(
    matrix: TridiagonalMatrixView<'_, S>,
    corner_upper: S,
    corner_lower: S,
) -> Result<(Vec<S>, Vec<usize>), SciOxideError> {
    let n = matrix.len();
    if n < 3 {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("matrix"),
            scioxide_core::InvalidInputReason::OutOfRange,
            "a cyclic tridiagonal matrix requires at least three rows",
        ));
    }
    let original = |i: usize| {
        if i.is_multiple_of(2) {
            i / 2
        } else {
            n - 1 - i / 2
        }
    };
    let len =
        scioxide_core::backend::checked_product(&[n, 5], scioxide_core::quantity!("cyclic_band"))?;
    let mut data = try_with_capacity(len, scioxide_core::quantity!("cyclic_band"))?;
    data.resize(len, S::ZERO);
    for j in 0..n {
        for i in j.saturating_sub(2)..(j + 3).min(n) {
            let (r, c) = (original(i), original(j));
            data[j * 5 + 2 + i - j] = if r == c {
                matrix.main[r]
            } else if r + 1 == c {
                matrix.upper[r]
            } else if c + 1 == r {
                matrix.lower[c]
            } else if r == 0 && c == n - 1 {
                corner_upper
            } else if c == 0 && r == n - 1 {
                corner_lower
            } else {
                S::ZERO
            };
        }
    }
    let mut swaps = try_with_capacity(n, scioxide_core::quantity!("cyclic_permutation"))?;
    let mut positions = try_with_capacity(n, scioxide_core::quantity!("cyclic_permutation"))?;
    let mut values = try_with_capacity(n, scioxide_core::quantity!("cyclic_permutation"))?;
    positions.extend(0..n);
    values.extend(0..n);
    for i in 0..n {
        let j = positions[original(i)];
        swaps.push(j);
        positions[values[i]] = j;
        positions[values[j]] = i;
        values.swap(i, j);
    }
    Ok((data, swaps))
}

/// Reorder a vector from cyclic order into the folded band order.
fn fold_vector<T>(values: &mut [T], swaps: &[usize]) {
    for (i, &j) in swaps.iter().enumerate() {
        values.swap(i, j);
    }
}

/// Undo [`fold_vector`].
fn unfold_vector<T>(values: &mut [T], swaps: &[usize]) {
    for (i, &j) in swaps.iter().enumerate().rev() {
        values.swap(i, j);
    }
}

// Solve staged columns of length `order`, in parallel when allowed.
fn solve_columns<S: NumericScalar>(
    staged: &mut [S],
    order: usize,
    policy: ExecutionPolicy,
    execution: ExecutionContext<'_>,
    solve: &(dyn Fn(&mut [S]) -> Result<(), SciOxideError> + Sync),
) -> Result<(), SciOxideError> {
    crate::kernel::execution::execute_chunks(staged, order, policy, execution, |_, block| {
        for column in block.chunks_exact_mut(order) {
            execution.check()?;
            solve(column)?;
        }
        Ok(())
    })
}

fn validate_layout(lower: usize, main: usize, upper: usize) -> Result<(), SciOxideError> {
    let expected = main.saturating_sub(1);
    if lower != expected || upper != expected {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(expected, expected),
            scioxide_core::Dimensions::two(lower, upper),
        ));
    }
    Ok(())
}

// Validate input entries. Computed factors and solutions need no rescan:
// elimination, substitution and checked operations reject a non-finite
// result where it arises.
fn validate_finite<S: NumericScalar>(values: &[S]) -> Result<(), SciOxideError> {
    if values.iter().any(|value| !value.has_finite_components()) {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "tridiagonal entry has a non-finite component",
        ));
    }
    Ok(())
}

fn ensure_pivot<S: NumericScalar>(pivot: S, index: usize) -> Result<(), SciOxideError> {
    if !pivot.has_finite_components() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Intermediate,
            "pivot is non-finite",
        ));
    }
    if pivot == S::ZERO {
        return Err(SciOxideError::singular(Some(index)));
    }
    Ok(())
}

impl<S: NumericScalar> TridiagonalLuDecomposition<S> {
    fn rank(
        &self,
        policy: crate::RankPolicy<S::Real>,
    ) -> Result<crate::RankReport<S::Real>, SciOxideError> {
        if matches!(policy, crate::RankPolicy::Exact) {
            return Ok(crate::RankReport::from_factor(
                self.order(),
                self.order(),
                S::Real::ZERO,
                self.exact_scale,
            ));
        }
        crate::analysis::analyze_scaled_magnitudes(
            self.main
                .iter()
                .map(|&x| scioxide_core::ScaledValue::from_value(x)?.magnitude()),
            self.order(),
            policy,
        )
    }
    fn diagnostic_norm(&self) -> Result<scioxide_core::Scaled<S::Real>, SciOxideError> {
        Ok(self.source.norm())
    }
    fn residual(
        &self,
        op: StructuredOperation,
        x: &[S],
        b: crate::VectorView<'_, S>,
        context: crate::KernelContext<'_>,
    ) -> Result<scioxide_core::ResidualReport<S::Real>, SciOxideError> {
        self.source.residual(op, x, b, context)
    }
    fn solve_scaled(
        &self,
        x: &mut [scioxide_core::ScaledValue<S>],
        adjoint: bool,
    ) -> Result<(), SciOxideError> {
        use scioxide_core::ScaledValue;
        if !adjoint {
            for j in 0..self.order().saturating_sub(1) {
                x.swap(j, self.pivots[j]);
                x[j + 1] = x[j + 1].sub(ScaledValue::from_value(self.lower[j])?.mul(x[j])?)?;
            }
        }
        crate::diagnostics::substitute(
            x,
            scioxide_core::Triangle::Upper,
            scioxide_core::DiagonalKind::Explicit,
            2,
            adjoint,
            |i, j| {
                if i == j {
                    self.main[i]
                } else if j == i + 1 {
                    self.upper[i]
                } else {
                    self.second_upper[i]
                }
            },
        )?;
        if adjoint {
            for j in (0..self.order().saturating_sub(1)).rev() {
                x[j] = x[j].sub(ScaledValue::from_value(self.lower[j].conj())?.mul(x[j + 1])?)?;
                x.swap(j, self.pivots[j]);
            }
        }
        Ok(())
    }
}

#[inline(always)]
fn factor_step<S: NumericScalar>(
    lower: &mut [S],
    main: &mut [S],
    upper: &mut [S],
    second_upper: &mut [S],
    pivots: &mut [usize],
    index: usize,
) -> Result<(), SciOxideError> {
    // One step of Gaussian elimination with partial pivoting restricted to
    // the two candidate rows: row `index + 1` is only
    // nonzero in columns `index..=index + 1`, so the pivot is either the
    // diagonal or the subdiagonal entry. Choosing the larger bounds the
    // multiplier by 1. A row swap moves the next row's entries up and makes
    // U gain a second superdiagonal, stored in `second_upper`; `pivots`
    // records the swap for the solves.
    if main[index].abs() >= lower[index].abs() {
        ensure_pivot(main[index], index)?;
        let multiplier = multiplier::<S>(lower[index], main[index])?;
        lower[index] = multiplier;
        main[index + 1] = substitute_row(
            main[index + 1],
            S::ONE,
            [(multiplier, upper[index])].into_iter(),
        )?;
    } else {
        ensure_pivot(lower[index], index)?;
        let multiplier = multiplier::<S>(main[index], lower[index])?;
        main[index] = lower[index];
        lower[index] = multiplier;
        let original_upper = upper[index];
        upper[index] = main[index + 1];
        main[index + 1] = substitute_row(
            original_upper,
            S::ONE,
            [(multiplier, main[index + 1])].into_iter(),
        )?;
        if index < second_upper.len() {
            second_upper[index] = upper[index + 1];
            upper[index + 1] = sub::<S>(S::ZERO, mul::<S>(multiplier, upper[index + 1])?)?;
        }
        pivots[index] = index + 1;
    }
    Ok(())
}

fn solve_normal<S: NumericScalar>(
    lower: &[S],
    main: &[S],
    upper: &[S],
    second_upper: &[S],
    pivots: &[usize],
    values: &mut [S],
) -> Result<(), SciOxideError> {
    solve_normal_with_execution(
        lower,
        main,
        upper,
        second_upper,
        pivots,
        values,
        ExecutionContext::sequential_native(),
    )
}

fn solve_normal_with_execution<S: NumericScalar>(
    lower: &[S],
    main: &[S],
    upper: &[S],
    second_upper: &[S],
    pivots: &[usize],
    values: &mut [S],
    execution: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let n = main.len();
    for i in 0..n.saturating_sub(1) {
        if i % 128 == 0 {
            execution.check()?;
        }
        if pivots[i] != i {
            values.swap(i, i + 1);
        }
        let product = lower[i] * values[i];
        let candidate = values[i + 1] - product;
        values[i + 1] = if candidate.has_finite_components()
            && !scioxide_core::backend::standard::product_needs_retry(lower[i], values[i], product)
        {
            candidate
        } else {
            substitute_row(values[i + 1], S::ONE, [(lower[i], values[i])].into_iter())?
        };
    }
    for i in (0..n).rev() {
        if i % 128 == 0 {
            execution.check()?;
        }
        let a = if i + 1 < n { upper[i] } else { S::ZERO };
        let x = values.get(i + 1).copied().unwrap_or(S::ZERO);
        let b = if i + 2 < n { second_upper[i] } else { S::ZERO };
        let y = values.get(i + 2).copied().unwrap_or(S::ZERO);
        let ax = a * x;
        let by = b * y;
        let residual = (values[i] - ax) - by;
        let candidate = residual / main[i];
        let tiny = residual.abs() < S::Real::MIN_POSITIVE && main[i].abs() < S::Real::ONE;
        values[i] = if candidate.has_finite_components()
            && !tiny
            && !scioxide_core::backend::standard::product_needs_retry(a, x, ax)
            && !scioxide_core::backend::standard::product_needs_retry(b, y, by)
        {
            candidate
        } else {
            substitute_row(values[i], main[i], [(a, x), (b, y)].into_iter())?
        };
    }
    execution.check()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::Matrix;

    use super::*;
    use scioxide_core::Complex;

    type Complex64 = Complex<f64>;

    #[test]
    fn real_rank_preserves_subnormal_pivots_and_range_boundaries() {
        macro_rules! check {
            ($real:ty) => {
                let main = [<$real>::from_bits(1), <$real>::MAX];
                let matrix = TridiagonalMatrixView::new(&[0.], &main, &[0.]).unwrap();
                let factor = TridiagonalLuDecomposition::factor(matrix).unwrap();
                let exact = factor.rank(crate::RankPolicy::Exact).unwrap();
                assert_eq!(exact.rank(), 2);
                assert_eq!(exact.scale(), <$real>::MAX);
                let threshold = factor
                    .rank(crate::RankPolicy::DimensionScaledEpsilon)
                    .unwrap();
                assert_eq!(threshold.rank(), 1);
                assert!(threshold.threshold() > 0. && threshold.threshold().is_finite());
            };
        }
        check!(f32);
        check!(f64);
    }

    fn residual(matrix: TridiagonalMatrixView<'_, f64>, x: &[f64], rhs: &[f64]) -> f64 {
        (0..matrix.len())
            .map(|row| {
                let mut value = matrix.main[row] * x[row];
                if row > 0 {
                    value += matrix.lower[row - 1] * x[row - 1];
                }
                if row + 1 < matrix.len() {
                    value += matrix.upper[row] * x[row + 1];
                }
                (value - rhs[row]).abs()
            })
            .fold(0.0, f64::max)
    }

    fn factor_into(
        matrix: &TridiagonalMatrix<f64>,
        factor: &mut TridiagonalLuDecomposition<f64>,
        workspace: &mut LinalgWorkspace<'_, f64>,
    ) -> Result<(), SciOxideError> {
        TridiagonalLuDecomposition::factor_into(
            matrix,
            &LinalgExecutionOptions::new(),
            factor,
            workspace,
            ExecutionContext::sequential(),
        )
    }

    #[test]
    fn failed_decomposition_keeps_the_published_factor() {
        let mut factor = TridiagonalLuDecomposition::<f64>::empty();
        assert!(factor.is_empty());
        assert!(
            factor
                .solve(StructuredOperation::Normal, &[][..])
                .unwrap()
                .is_empty()
        );
        let regular = TridiagonalMatrix::new(vec![0.0], vec![2.0, 3.0], vec![0.0]).unwrap();
        let singular = TridiagonalMatrix::new(vec![0.0], vec![0.0, 0.0], vec![0.0]).unwrap();
        let mut workspace = LinalgWorkspace::new();
        assert_eq!(
            factor_into(&singular, &mut factor, &mut workspace)
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::Singular
        );
        assert!(factor.is_empty());
        factor_into(&regular, &mut factor, &mut workspace).unwrap();
        let rhs = [2.0, 3.0];
        let solve = |factor: &TridiagonalLuDecomposition<f64>| {
            factor
                .solve(StructuredOperation::Normal, &rhs[..])
                .unwrap()
                .into_vec()
        };
        assert_eq!(solve(&factor), [1.0, 1.0]);
        assert!(factor_into(&singular, &mut factor, &mut workspace).is_err());
        assert_eq!(solve(&factor), [1.0, 1.0]);
        factor_into(&regular, &mut factor, &mut workspace).unwrap();
        assert_eq!(solve(&factor), [1.0, 1.0]);
    }

    #[test]
    fn layout_rules_cover_empty_and_singleton_matrices() {
        let empty = TridiagonalMatrix::<f64>::new(vec![], vec![], vec![]).unwrap();
        let empty_factor = TridiagonalLuDecomposition::factor(&empty).unwrap();
        assert!(
            empty_factor
                .solve(StructuredOperation::Normal, &[][..])
                .unwrap()
                .is_empty()
        );

        let singleton = TridiagonalMatrix::new(vec![], vec![2.0], vec![]).unwrap();
        let solution = TridiagonalLuDecomposition::factor(&singleton)
            .unwrap()
            .solve(StructuredOperation::Normal, &[6.0][..])
            .unwrap();
        assert!(f64::abs(solution.as_slice()[0] - 3.0) < 1.0e-14);

        assert!(matches!(
            (TridiagonalMatrix::new(vec![1.0], vec![2.0], vec![]))
                .as_ref()
                .err()
                .unwrap()
                .view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn adjacent_pivoting_stores_second_superdiagonal() {
        let matrix = TridiagonalMatrix::new(
            vec![2.0, 3.0, 4.0],
            vec![0.25, 0.5, 5.0, 6.0],
            vec![1.0, 7.0, 8.0],
        )
        .unwrap();
        let rhs = [1.0, 2.0, 3.0, 4.0];
        let factor = TridiagonalLuDecomposition::factor(&matrix).unwrap();
        assert_eq!(factor.pivots()[0], 1);
        assert_ne!(factor.second_superdiagonal()[0], 0.0);
        let solution = factor.solve(StructuredOperation::Normal, &rhs[..]).unwrap();
        assert!(residual(matrix.view(), solution.as_slice(), &rhs) < 1.0e-12);
    }

    #[test]
    fn singularity_and_non_finite_inputs_are_diagnostic() {
        let singular = TridiagonalMatrix::new(vec![0.0], vec![0.0, 0.0], vec![0.0]).unwrap();
        assert!(matches!(
            TridiagonalLuDecomposition::factor(&singular)
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::Singular { pivot, .. } if pivot == Some(0)
        ));
        assert!(matches!(
            (TridiagonalMatrix::new(vec![], vec![f64::NAN], vec![]))
                .as_ref()
                .err()
                .unwrap()
                .view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }

    #[test]
    fn factor_and_output_storage_can_be_reused_for_f32() {
        let first =
            TridiagonalMatrix::new(vec![-1.0_f32; 3], vec![4.0_f32; 4], vec![-1.0; 3]).unwrap();
        let second =
            TridiagonalMatrix::new(vec![0.5_f32; 3], vec![3.0_f32; 4], vec![0.25; 3]).unwrap();
        let rhs = [1.0_f32, 2.0, 3.0, 4.0];
        let options = LinalgExecutionOptions::new();
        let mut factor = TridiagonalLuDecomposition::factor(&first).unwrap();
        let mut output = [0.0_f32; 4];
        let mut workspace = LinalgWorkspace::new();
        let execution = ExecutionContext::sequential();
        factor
            .solve_into(
                StructuredOperation::Normal,
                &rhs[..],
                &crate::DirectSolveOptions::new(crate::RankPolicy::Exact),
                VectorViewMut::from_slice(&mut output),
                &mut workspace,
                execution,
            )
            .and_then(|report| {
                if report.termination() == crate::DirectSolveTermination::Accepted {
                    Ok(())
                } else {
                    Err(scioxide_core::SciOxideError::singular(None))
                }
            })
            .unwrap();
        TridiagonalLuDecomposition::factor_into(
            &second,
            &options,
            &mut factor,
            &mut workspace,
            execution,
        )
        .unwrap();
        factor
            .solve_into(
                StructuredOperation::Normal,
                &rhs[..],
                &crate::DirectSolveOptions::new(crate::RankPolicy::Exact),
                VectorViewMut::from_slice(&mut output),
                &mut workspace,
                execution,
            )
            .and_then(|report| {
                if report.termination() == crate::DirectSolveTermination::Accepted {
                    Ok(())
                } else {
                    Err(scioxide_core::SciOxideError::singular(None))
                }
            })
            .unwrap();
        let max_residual = (0..4)
            .map(|row| {
                let mut value = second.main()[row] * output[row];
                if row > 0 {
                    value += second.lower()[row - 1] * output[row - 1];
                }
                if row + 1 < 4 {
                    value += second.upper()[row] * output[row + 1];
                }
                (value - rhs[row]).abs()
            })
            .fold(0.0_f32, f32::max);
        assert!(max_residual < 1.0e-5);
    }

    #[test]
    fn transpose_and_multi_rhs_solutions_match_residuals() {
        let matrix =
            TridiagonalMatrix::new(vec![-1.0, 2.0], vec![4.0, 5.0, 6.0], vec![3.0, -2.0]).unwrap();
        let factor = TridiagonalLuDecomposition::factor(&matrix).unwrap();
        let rhs = Matrix::from_vec(3, 2, vec![1.0, 3.0, 2.0, 2.0, 4.0, 1.0]).unwrap();
        let solutions = factor
            .solve_matrix(StructuredOperation::Normal, &rhs)
            .unwrap();
        for column in 0..2 {
            let right: Vec<f64> = (0..3)
                .map(|row| rhs.view().get(row, column).unwrap())
                .collect();
            let solution: Vec<f64> = (0..3)
                .map(|row| solutions.view().get(row, column).unwrap())
                .collect();
            assert!(residual(matrix.view(), &solution, &right) < 1.0e-12);
        }

        let transpose_rhs = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let transpose_solution = factor
            .solve(StructuredOperation::Transpose, &transpose_rhs)
            .unwrap();
        let dense =
            Matrix::from_vec(3, 3, vec![4.0, -1.0, 0.0, 3.0, 5.0, 2.0, 0.0, -2.0, 6.0]).unwrap();
        let mut product = Vector::zeros(3).unwrap();
        dense
            .view()
            .mul_vec_into(
                &transpose_solution,
                &LinalgExecutionOptions::new(),
                product.view_mut(),
                &mut LinalgWorkspace::new(),
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(
            product
                .as_slice()
                .iter()
                .zip(transpose_rhs.as_slice())
                .all(|(actual, expected)| (*actual - *expected).abs() < 1.0e-12)
        );
    }

    #[test]
    fn complex_adjoint_solve_is_correct() {
        let i = Complex64::new(0.0, 1.0);
        let matrix = TridiagonalMatrix::new(
            vec![Complex64::new(1.0, -1.0), Complex64::new(-0.5, 0.25)],
            vec![
                Complex64::new(4.0, 1.0),
                Complex64::new(5.0, 0.5),
                Complex64::new(3.0, -1.0),
            ],
            vec![i, Complex64::new(0.75, -0.5)],
        )
        .unwrap();
        let factor = TridiagonalLuDecomposition::factor(&matrix).unwrap();
        let rhs = [Complex64::new(1.0, 2.0), i, Complex64::new(3.0, -1.0)];
        let solution = factor
            .solve(StructuredOperation::Adjoint, &rhs[..])
            .unwrap();
        for (row, expected) in rhs.iter().enumerate() {
            let mut value = matrix.main[row].conj() * solution.as_slice()[row];
            if row > 0 {
                value += matrix.upper[row - 1].conj() * solution.as_slice()[row - 1];
            }
            if row + 1 < 3 {
                value += matrix.lower[row].conj() * solution.as_slice()[row + 1];
            }
            assert!((value - *expected).norm() < 1.0e-12);
        }
    }

    #[test]
    fn cyclic_plan_solves_and_rejects_small_dimensions() {
        let ordinary = TridiagonalMatrix::new(
            vec![-1.0, -1.0, -1.0],
            vec![4.0, 4.0, 4.0, 4.0],
            vec![-1.0, -1.0, -1.0],
        )
        .unwrap();
        let plan =
            CyclicTridiagonalPlan::new(&ordinary, -0.5, -0.25, &LinalgExecutionOptions::new())
                .unwrap();
        let rhs = [1.0, 2.0, 3.0, 4.0];
        let solution = plan.solve(StructuredOperation::Normal, &rhs[..]).unwrap();
        let solution = solution.as_slice();
        for row in 0..4 {
            let mut value = ordinary.main[row] * solution[row];
            if row > 0 {
                value += ordinary.lower[row - 1] * solution[row - 1];
            }
            if row + 1 < 4 {
                value += ordinary.upper[row] * solution[row + 1];
            }
            if row == 0 {
                value += -0.5 * solution[3];
            }
            if row == 3 {
                value += -0.25 * solution[0];
            }
            assert!(f64::abs(value - rhs[row]) < 1.0e-12);
        }
        // The transposed cyclic matrix swaps the off-diagonals and corners.
        let transposed = plan
            .solve(StructuredOperation::Transpose, &rhs[..])
            .unwrap();
        let swapped = CyclicTridiagonalPlan::new(
            TridiagonalMatrixView::new(&ordinary.upper, &ordinary.main, &ordinary.lower).unwrap(),
            -0.25,
            -0.5,
            &LinalgExecutionOptions::new(),
        )
        .unwrap()
        .solve(StructuredOperation::Normal, &rhs[..])
        .unwrap();
        for (a, b) in transposed.as_slice().iter().zip(swapped.as_slice()) {
            assert!((a - b).abs() < 1.0e-12);
        }

        let too_small = TridiagonalMatrix::new(vec![1.0], vec![2.0, 2.0], vec![1.0]).unwrap();
        assert!(matches!(
            CyclicTridiagonalPlan::new(&too_small, 1.0, 1.0, &LinalgExecutionOptions::new())
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));

        let singular_cyclic =
            TridiagonalMatrix::new(vec![0.0, 0.0], vec![1.0, 1.0, 1.0], vec![0.0, 0.0]).unwrap();
        assert!(matches!(
            CyclicTridiagonalPlan::new(&singular_cyclic, 1.0, 1.0, &LinalgExecutionOptions::new())
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::Singular { .. }
        ));
    }

    #[test]
    fn cyclic_plan_solves_matrix_whose_rank_one_split_is_singular() {
        // With γ = −a₀₀ the Sherman–Morrison denominator of this matrix is
        // within rounding of zero; the band LU needs no split.
        let delta = 4.0 * f64::EPSILON;
        let matrix = TridiagonalMatrixView::new(&[0.0; 2], &[1.0; 3], &[0.0; 2]).unwrap();
        let execution = LinalgExecutionOptions::new();
        let plan = CyclicTridiagonalPlan::new(matrix, 1.0, 1.0 - delta, &execution).unwrap();
        let threshold =
            crate::RankPolicy::Threshold(scioxide_core::RankThreshold::new(1e-8, 0.0).unwrap());
        assert_eq!(plan.rank(threshold).unwrap().rank(), 2);
        let options = crate::DirectSolveOptions::new(crate::RankPolicy::Exact)
            .with_residual(crate::Estimation::Estimate);
        for op in [
            StructuredOperation::Normal,
            StructuredOperation::Transpose,
            StructuredOperation::Adjoint,
        ] {
            let rhs = if op == StructuredOperation::Normal {
                [0.0, 2.0, delta]
            } else {
                [-delta, 2.0, 0.0]
            };
            let result = plan.solve_with_options(op, &rhs[..], &options).unwrap();
            assert_eq!(
                result.report().termination(),
                crate::DirectSolveTermination::Accepted
            );
            assert!(result.report().residual().unwrap().scaled_norm() < 1e-15);
            for (&actual, expected) in result
                .solution()
                .unwrap()
                .as_slice()
                .iter()
                .zip([-1.0, 2.0, 1.0])
            {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cyclic_gamma_handles_scalar_range_boundaries() {
        macro_rules! check {
            ($real:ty, $tolerance:expr) => {
                for fraction in [0.5, 0.6, 1.0] {
                    for sign in [-1.0, 1.0] {
                        let scale = sign * fraction * <$real>::MAX;
                        let main = [scale; 3];
                        let matrix =
                            TridiagonalMatrixView::new(&[0.0; 2], &main, &[0.0; 2]).unwrap();
                        let plan = CyclicTridiagonalPlan::new(
                            matrix,
                            0.0,
                            0.0,
                            &LinalgExecutionOptions::new(),
                        )
                        .unwrap();
                        let rhs = [0.125 * scale, -0.25 * scale, 0.375 * scale];
                        for op in [
                            StructuredOperation::Normal,
                            StructuredOperation::Transpose,
                            StructuredOperation::Adjoint,
                        ] {
                            let x = plan.solve(op, &rhs[..]).unwrap();
                            for (&actual, expected) in
                                x.as_slice().iter().zip([0.125, -0.25, 0.375])
                            {
                                assert!((actual - expected).abs() < $tolerance);
                            }
                        }
                        assert!(
                            (plan.condition_estimate().unwrap().reciprocal().value() - 1.0).abs()
                                < $tolerance
                        );
                        let plan = CyclicTridiagonalPlan::new(
                            matrix,
                            0.125 * scale,
                            0.25 * scale,
                            &LinalgExecutionOptions::new(),
                        )
                        .unwrap();
                        for op in [
                            StructuredOperation::Normal,
                            StructuredOperation::Transpose,
                            StructuredOperation::Adjoint,
                        ] {
                            let rhs = if op == StructuredOperation::Normal {
                                [0.171875 * scale, -0.25 * scale, 0.40625 * scale]
                            } else {
                                [0.21875 * scale, -0.25 * scale, 0.390625 * scale]
                            };
                            let x = plan.solve(op, &rhs[..]).unwrap();
                            for (&actual, expected) in
                                x.as_slice().iter().zip([0.125, -0.25, 0.375])
                            {
                                assert!((actual - expected).abs() < $tolerance);
                            }
                        }
                    }
                }
            };
        }
        check!(f32, 1e-5);
        check!(f64, 1e-12);
    }

    #[test]
    fn cyclic_band_keeps_tiny_multipliers_and_inverse_actions() {
        fn check<S: NumericScalar<Real = f64>>() {
            let n = 1200;
            let lower_value = S::from_components(0.75, if S::IS_REAL { 0.0 } else { -0.5 });
            let upper_value = S::from_components(1.0, if S::IS_REAL { 0.0 } else { 0.25 });
            let lower = vec![lower_value; n - 1];
            let upper = vec![upper_value; n - 1];
            let main = vec![S::from_real(4.0); n];
            let execution = LinalgExecutionOptions::new();
            // The multipliers of the folded band underflow gradually.
            let plan = CyclicTridiagonalPlan::new(
                TridiagonalMatrixView::new(&lower, &main, &upper).unwrap(),
                lower_value,
                upper_value,
                &execution,
            )
            .unwrap();
            let options = crate::DirectSolveOptions::new(crate::RankPolicy::Exact)
                .with_residual(crate::Estimation::Estimate)
                .with_minimum_reciprocal_condition(
                    crate::ReciprocalConditionThreshold::new(0.0).unwrap(),
                );
            let mut workspace = plan
                .workspace(&StructuredWorkspaceRequest::solve(n, &execution))
                .unwrap();
            let mut output = vec![S::ZERO; n];
            for op in [
                StructuredOperation::Normal,
                StructuredOperation::Transpose,
                StructuredOperation::Adjoint,
            ] {
                let b = S::from_real(4.0) + lower_value + upper_value;
                let rhs = vec![
                    if op == StructuredOperation::Adjoint {
                        b.conj()
                    } else {
                        b
                    };
                    n
                ];
                let report = plan
                    .solve_into(
                        op,
                        &rhs[..],
                        &options,
                        VectorViewMut::from_slice(&mut output),
                        &mut workspace,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap();
                assert!(report.termination().is_success());
                assert_eq!(report.rank().rank(), n);
                assert!(report.condition().unwrap().reciprocal().value() > 0.1);
                assert!(report.residual().unwrap().scaled_norm() < 1e-13);
                assert!(output.iter().all(|&x| (x - S::ONE).abs() < 1e-12));
            }
        }
        check::<f64>();
        check::<Complex64>();
    }

    #[test]
    fn cyclic_plan_solves_every_operation_of_a_nearly_singular_split() {
        // A cyclic matrix whose Sherman–Morrison split is nearly singular for
        // the transposed operation.
        let lower = [-0.9839305015979414, -0.9245022042907793];
        let main = [1.2203696402340427, 1.067150903504919, 1.2719239079885056];
        let upper = [-0.9448393286926058, -0.7487469272068175];
        let matrix = TridiagonalMatrixView::new(&lower, &main, &upper).unwrap();
        let plan = CyclicTridiagonalPlan::new(
            matrix,
            -0.791571233093336,
            0.7027660735921679,
            &LinalgExecutionOptions::new(),
        )
        .unwrap();
        let rhs = [1.0, 2.0, 3.0];
        let mut output = [7.0, 8.0, 9.0];
        let mut workspace = plan
            .workspace(&StructuredWorkspaceRequest::solve(
                plan.len(),
                &LinalgExecutionOptions::new(),
            ))
            .unwrap();
        let report = plan
            .solve_into(
                StructuredOperation::Normal,
                &rhs[..],
                &crate::DirectSolveOptions::new(crate::RankPolicy::Exact),
                VectorViewMut::from_slice(&mut output),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.termination().is_success());
        let options = crate::DirectSolveOptions::new(crate::RankPolicy::Exact)
            .with_residual(crate::Estimation::Estimate);
        let report = plan
            .solve_into(
                StructuredOperation::Transpose,
                &rhs[..],
                &options,
                VectorViewMut::from_slice(&mut output),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.termination().is_success());
        assert!(report.residual().unwrap().scaled_norm() < 1e-15);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            scioxide_core::AttemptOutcome::Succeeded
        );
    }

    #[test]
    fn cyclic_condition_test_is_scale_invariant() {
        for exponent in [-400, 0, 400] {
            let scale = 2.0_f64.powi(exponent);
            let matrix =
                TridiagonalMatrix::new(vec![0.0, 0.0], vec![scale; 3], vec![0.0, 0.0]).unwrap();
            let plan =
                CyclicTridiagonalPlan::new(&matrix, 0.0, 0.0, &LinalgExecutionOptions::new())
                    .unwrap();
            let rhs = [scale, -scale, 0.5 * scale];
            let mut output = [7.0, 8.0, 9.0];
            let mut workspace = plan
                .workspace(&StructuredWorkspaceRequest::solve(
                    plan.len(),
                    &LinalgExecutionOptions::new(),
                ))
                .unwrap();
            let report = plan
                .solve_into(
                    StructuredOperation::Normal,
                    &rhs[..],
                    &crate::DirectSolveOptions::new(crate::RankPolicy::Exact),
                    VectorViewMut::from_slice(&mut output),
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert!(report.termination().is_success());
            assert!(
                output
                    .iter()
                    .zip([1.0, -1.0, 0.5])
                    .all(|(&actual, expected)| (actual - expected).abs() < 1.0e-12)
            );
        }

        let large =
            TridiagonalMatrix::new(vec![0.0, 0.0], vec![1.0e100; 3], vec![0.0, 0.0]).unwrap();
        assert!(
            CyclicTridiagonalPlan::new(&large, 0.0, 0.0, &LinalgExecutionOptions::new()).is_ok()
        );

        let zero_first =
            TridiagonalMatrix::new(vec![1.0, 1.0], vec![0.0, 4.0, 4.0], vec![1.0, 1.0]).unwrap();
        assert!(
            CyclicTridiagonalPlan::new(&zero_first, 0.25, -0.5, &LinalgExecutionOptions::new())
                .is_ok()
        );
    }

    #[test]
    fn tridiagonal_failures_preserve_factor_and_output() {
        let regular = TridiagonalMatrix::new(vec![0.0], vec![2.0, 3.0], vec![0.0]).unwrap();
        let overflowing =
            TridiagonalMatrix::new(vec![f64::MAX], vec![f64::MAX, f64::MAX], vec![-f64::MAX])
                .unwrap();
        let mut factor = TridiagonalLuDecomposition::factor(&regular).unwrap();
        let mut workspace = LinalgWorkspace::new();
        assert!(matches!(
            factor_into(&overflowing, &mut factor, &mut workspace)
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
        let solution = factor
            .solve(StructuredOperation::Normal, &[2.0, 3.0][..])
            .unwrap();
        assert_eq!(solution.as_slice(), &[1.0, 1.0]);

        let tiny = TridiagonalMatrix::new(
            vec![0.0],
            vec![f64::MIN_POSITIVE, f64::MIN_POSITIVE],
            vec![0.0],
        )
        .unwrap();
        let tiny_factor = TridiagonalLuDecomposition::factor(&tiny).unwrap();
        let mut output = [11.0, 13.0];
        assert!(matches!(
            tiny_factor
                .solve_into(
                    StructuredOperation::Normal,
                    &[f64::MAX, 1.0][..],
                    &crate::DirectSolveOptions::new(crate::RankPolicy::Exact),
                    VectorViewMut::from_slice(&mut output),
                    &mut workspace,
                    ExecutionContext::sequential()
                )
                .and_then(|report| {
                    if report.termination() == crate::DirectSolveTermination::Accepted {
                        Ok(())
                    } else {
                        Err(scioxide_core::SciOxideError::singular(None))
                    }
                })
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
        assert_eq!(output, [11.0, 13.0]);
    }
}
