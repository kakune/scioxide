use super::active::Active;
use super::*;
use crate::SparseSolveOptions;
use crate::SparseView;
use crate::backend::hermitian::block_values;
use crate::factor_arithmetic::{Arithmetic, ordinary_product, ordinary_sum};
use crate::helpers::admit;
use core::cmp::Ordering;
use scioxide_core::AlgebraicScalar;
use scioxide_core::{Accuracy, RealScalar};
use scioxide_core::{Scaled, ScaledValue};

#[derive(Debug, Clone, Copy)]
pub(super) struct Entry<S> {
    pub row: usize,
    pub col: usize,
    pub value: S,
}
#[derive(Debug)]
pub(super) struct Data<S: NumericScalar> {
    pub entries: Vec<Entry<S>>,
    pub ptr: Vec<usize>,
    pub diagonal: Vec<S>,
    pub off: Vec<S>,
    pub blocks: Vec<LdlPivotBlock>,
    pub p: Vec<usize>,
    pub inverse: Vec<usize>,
}
const MATRIX: Argument = scioxide_core::argument!("matrix");
impl<S: NumericScalar> Data<S> {
    // Reserves factor storage.
    fn new(plan: &SparseSymmetricSymbolic) -> Result<Self, SciOxideError> {
        Self::reserve_shape(plan.dimension(), plan.limits.max_factor_entries)
    }
    fn reserve_shape(n: usize, entries: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            entries: try_with_capacity(entries, scioxide_core::quantity!("factor_entries"))?,
            ptr: try_filled(
                add(n, 1, scioxide_core::quantity!("factor_column_pointers"))?,
                0,
                scioxide_core::quantity!("factor_column_pointers"),
            )?,
            diagonal: try_filled(n, S::ZERO, scioxide_core::quantity!("factor_diagonal"))?,
            off: try_filled(n, S::ZERO, scioxide_core::quantity!("factor_off_diagonal"))?,
            blocks: try_with_capacity(n, scioxide_core::quantity!("pivot_blocks"))?,
            p: try_filled(n, 0, scioxide_core::quantity!("symmetric_permutation"))?,
            inverse: try_filled(n, 0, scioxide_core::quantity!("inverse_permutation"))?,
        })
    }
    fn reset(&mut self, plan: &SparseSymmetricSymbolic) {
        self.entries.clear();
        self.ptr.fill(0);
        self.diagonal.fill(S::ZERO);
        self.off.fill(S::ZERO);
        self.blocks.clear();
        self.p.copy_from_slice(plan.permutation.new_to_old());
        self.inverse.copy_from_slice(plan.permutation.old_to_new());
    }
    fn swap(&mut self, i: usize, j: usize) {
        self.p.swap(i, j);
        self.inverse[self.p[i]] = i;
        self.inverse[self.p[j]] = j;
    }
    // Appends one factor entry.
    fn push(
        &mut self,
        (row, col): (usize, usize),
        value: S,
        limit: usize,
    ) -> Result<(), SciOxideError> {
        if self.entries.len() == limit {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("factor_entries"),
                self.entries.len() + 1,
                limit,
            ));
        }
        self.entries.push(Entry { row, col, value });
        Ok(())
    }
}
/// Holds the storage for recomputing a sparse Cholesky or `LDLᴴ` factorization
/// without allocation.
///
/// The workspace holds a second factor, scratch, and for `LDLᴴ` the entries of
/// the Schur complement. `refactor` computes into it and replaces the current
/// factor only on success. All of this storage counts toward the byte limit of
/// the analysis. Create a workspace with [`SparseSymmetricSymbolic::workspace`],
/// or with [`new`](Self::new) and [`try_reserve`](Self::try_reserve).
///
/// ```
/// use scioxide_core::{Capacity, ExecutionContext, Triangle};
/// use scioxide_sparse::{
///     CsrMatrix, SparseCholesky, SparseOrdering, SparseSolveOptions, SparseSymmetricLimits,
///     SparseSymmetricSymbolic, SparseSymmetricWorkspace,
/// };
/// let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![4.0_f64, 9.])?;
/// let plan = SparseSymmetricSymbolic::analyze(
///     a.structural_pattern(),
///     SparseOrdering::Natural,
///     Some(Triangle::Lower),
///     SparseSymmetricLimits::new(1, 2, 1 << 16),
/// )?;
/// let requirements = plan.workspace_requirements::<f64>(())?;
/// let mut workspace = SparseSymmetricWorkspace::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// let options = SparseSolveOptions::new();
/// let scope = ExecutionContext::sequential();
/// let mut factor = SparseCholesky::factor_scoped(plan, a.view(), options, scope)?;
/// let b = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 4.])?;
/// factor.refactor(b.view(), options, &mut workspace, scope)?;
/// assert_eq!(factor.diagonal(), &[1., 2.]);
/// assert!(workspace.last_attempt().is_some());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug)]
pub struct SparseSymmetricWorkspace<S: NumericScalar> {
    epoch: crate::factor_solve::Epoch,
    pub(super) candidate: Data<S>,
    source: Vec<S>,
    seen: Vec<bool>,
    x: Vec<S>,
    y: Vec<S>,
    mark: Vec<usize>,
    reach: Vec<usize>,
    cursor: Vec<usize>,
    active: Active<S>,
    // Reserved strict-lower entry capacity of the candidate factor.
    factor_entries: usize,
}
impl<S: NumericScalar> SparseSymmetricWorkspace<S> {
    /// Reserves all scratch and a second factor within the plan's total byte limit.
    /// # Errors
    /// Returns `SizeOverflow`, `ResourceLimit` or `AllocationFailure`.
    fn reserve(r: &SparseSymmetricRequirements) -> Result<Self, SciOxideError> {
        let n = r.n;
        let z = r.z;
        Ok(Self {
            epoch: crate::factor_solve::Epoch::new(),
            candidate: Data::reserve_shape(n, r.limits.max_factor_entries)?,
            source: try_filled(z, S::ZERO, scioxide_core::quantity!("source_values"))?,
            seen: try_filled(z, false, scioxide_core::quantity!("source_seen"))?,
            x: try_filled(n, S::ZERO, scioxide_core::quantity!("dense_column"))?,
            y: try_filled(n, S::ZERO, scioxide_core::quantity!("second_dense_column"))?,
            mark: try_filled(n, n, scioxide_core::quantity!("reach_marks"))?,
            reach: try_filled(n, 0, scioxide_core::quantity!("reach"))?,
            cursor: try_filled(n, 0, scioxide_core::quantity!("column_cursors"))?,
            active: Active::new(n, r.limits.max_active_entries)?,
            factor_entries: r.limits.max_factor_entries,
        })
    }
    // Exact reserved shape and entry capacities of `r`; the byte ceiling is
    // admitted separately. A mismatch reports `DimensionMismatch` with the
    // requested extent first.
    fn fits(&self, r: &SparseSymmetricRequirements) -> Result<(), SciOxideError> {
        crate::helpers::exact(self.x.len(), r.n)?;
        crate::helpers::exact(self.candidate.ptr.len(), r.n + 1)?;
        crate::helpers::exact(self.source.len(), r.z)?;
        crate::helpers::exact(self.factor_entries, r.limits.max_factor_entries)?;
        crate::helpers::exact(self.active.limit(), r.limits.max_active_entries)
    }
}
macro_rules! factor {
    ($name:ident,$chol:expr,$description:literal) => {
        #[doc=$description]
        #[derive(Debug)]
        pub struct $name<S: NumericScalar> {
            pub(super) symbolic: SparseSymmetricSymbolic,
            pub(super) data: Data<S>,
        }
        impl<S: NumericScalar> $name<S> {
            /// Computes the factorization of `matrix`, whose pattern `symbolic`
            /// analyzed, with the default [`SparseSolveOptions`] in a fresh
            /// scope.
            ///
            /// This is [`factor_scoped`](Self::factor_scoped) with
            /// [`SparseSolveOptions::new`] and
            /// [`ExecutionContext::sequential_native`](scioxide_core::ExecutionContext::sequential_native).
            ///
            /// # Errors
            ///
            /// - The errors of [`factor_scoped`](Self::factor_scoped).
            pub fn factor(
                symbolic: SparseSymmetricSymbolic,
                matrix: SparseView<'_, S>,
            ) -> Result<Self, SciOxideError> {
                Self::factor_scoped(
                    symbolic,
                    matrix,
                    SparseSolveOptions::new(),
                    scioxide_core::ExecutionContext::sequential_native(),
                )
            }
            /// Computes the factorization of `matrix`, whose pattern `symbolic`
            /// analyzed, inside the caller's scope.
            ///
            /// This allocates the factor and a temporary workspace. To factor a
            /// sequence of matrices without allocation, use
            /// [`refactor`](Self::refactor).
            ///
            /// # Errors
            ///
            /// - `ResourceLimit` if the factor and workspace exceed the byte limit,
            ///   and `SizeOverflow` or `AllocationFailure` if they cannot be
            ///   allocated.
            /// - The errors of [`refactor`](Self::refactor).
            pub fn factor_scoped<'scope, 'capability>(
                symbolic: SparseSymmetricSymbolic,
                matrix: SparseView<'_, S>,
                options: SparseSolveOptions,
                scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
            ) -> Result<Self, SciOxideError> {
                let mut w = symbolic.workspace(())?;
                let mut result = Self {
                    data: Data::new(&symbolic)?,
                    symbolic,
                };
                result.refactor(matrix, options, &mut w, scope)?;
                Ok(result)
            }
            /// Returns the symbolic analysis.
            pub fn symbolic(&self) -> &SparseSymmetricSymbolic {
                &self.symbolic
            }
            /// Returns the number of stored entries of `L` strictly below the
            /// diagonal. The diagonal is stored separately.
            pub fn factor_entries(&self) -> usize {
                self.data.entries.len()
            }
            /// Returns the permutation `p`, with `p[new] = old`, including the
            /// interchanges of `LDLᴴ` pivoting.
            pub fn permutation(&self) -> &[usize] {
                self.data.p.as_slice()
            }
            /// Returns the entries `(i, j, L[i, j])` of `L` strictly below the
            /// diagonal, in permuted coordinates.
            pub fn lower_entries(&self) -> impl Iterator<Item = (usize, usize, S)> + '_ {
                self.data
                    .entries
                    .iter()
                    .map(|e| (self.data.inverse[e.row], e.col, e.value))
            }
            /// Returns the diagonal of `L` for Cholesky, or the diagonal of `D` for
            /// `LDLᴴ`.
            pub fn diagonal(&self) -> &[S] {
                &self.data.diagonal
            }
            /// Recomputes the factorization for new values with the analyzed
            /// pattern.
            ///
            /// `refactor` does not allocate. On error the previous factor is
            /// unchanged and stays usable.
            ///
            /// # Errors
            ///
            /// - `InvalidInput` if `matrix` does not have exactly the analyzed
            ///   pattern, a diagonal entry is not real, or, when the analysis reads
            ///   both triangles, the values are not exactly Hermitian.
            /// - `DimensionMismatch` if `matrix` has another shape, or `w` was
            ///   reserved for another analysis.
            /// - `ResourceLimit` if the factor needs more than
            ///   `max_factor_entries` entries, or `LDLᴴ` needs more than
            ///   `max_active_entries` Schur complement entries.
            /// - `NumericalInstability` if an entry overflows or underflows, or a
            ///   Cholesky pivot is not positive.
            /// - `Singular` if an `LDLᴴ` 2×2 pivot block has a zero determinant.
            /// - `ConvergenceFailure` if the iteration limit of `options` or
            ///   `scope` is reached before every row is factored.
            /// - `ExecutionStopped` or `CapabilityUnavailable` from the
            ///   [execution context](scioxide_core#execution-contexts).
            pub fn refactor<'scope, 'capability>(
                &mut self,
                matrix: SparseView<'_, S>,
                options: SparseSolveOptions,
                w: &mut SparseSymmetricWorkspace<S>,
                scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
            ) -> Result<(), SciOxideError> {
                w.epoch.reject();
                prepare(&self.symbolic, matrix, w)?;
                let mut epoch = core::mem::take(&mut w.epoch);
                let result = epoch.run(scope, options, |context| {
                    let arithmetic = Arithmetic(options.accuracy());
                    if $chol {
                        cholesky(&self.symbolic, w, arithmetic, &mut || admit(context))
                    } else {
                        ldl(&self.symbolic, w, arithmetic, &mut || admit(context))
                    }
                });
                w.epoch = epoch;
                result?;
                core::mem::swap(&mut self.data, &mut w.candidate);
                Ok(())
            }
        }
    };
}
factor!(
    SparseCholesky,
    true,
    r#"Holds the sparse Cholesky factorization of a Hermitian positive definite
matrix.

The factor satisfies

```text
A[p[i], p[j]] = (L Lᴴ)[i, j],
```

where `L` is lower triangular with a positive real diagonal and `p` is the
permutation of the [analysis](SparseSymmetricSymbolic). The factorization is
up-looking: row `k` of `L` comes from a sparse triangular solve whose pattern
follows the elimination tree, so the fill is the one the analysis predicts.
Factorization is sequential, and each row counts as one iteration toward the
iteration limit. A matrix that is not positive definite is rejected with
`NumericalInstability` at the first pivot that is not positive; use
[`SparseLdl`] for indefinite matrices.

Solve with `solve`, `solve_into` or `solve_block_into`. The factor also
implements [`LinearOperator`](scioxide_core::LinearOperator) as the action of
`A⁻¹`. See [Direct solvers](crate#direct-solvers) for the analyze–factor–solve
sequence.

# Examples

```
use scioxide_core::{SciOxideErrorKind, Triangle};
use scioxide_sparse::{
    CsrMatrix, SparseCholesky, SparseOrdering, SparseSymmetricLimits, SparseSymmetricSymbolic,
};
// Lower triangle of [[4, 2], [2, 5]] = L Lᵀ with L = [[2, 0], [1, 2]].
let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 3], vec![0, 0, 1], vec![4.0_f64, 2., 5.])?;
let (lower, limits) = (Some(Triangle::Lower), SparseSymmetricLimits::new(1, 3, 1 << 16));
let plan = SparseSymmetricSymbolic::analyze(a.structural_pattern(), SparseOrdering::Natural, lower, limits)?;
let factor = SparseCholesky::factor(plan.try_clone()?, a.view())?;
assert_eq!(factor.diagonal(), &[2., 2.]);
assert!(factor.solve(&[6., 7.])?.iter().all(|x| (x - 1.0).abs() < 1e-15));
// [[1, 2], [2, 1]] is indefinite.
let indefinite = CsrMatrix::from_raw(2, 2, vec![0, 1, 3], vec![0, 0, 1], vec![1.0_f64, 2., 1.])?;
let error = SparseCholesky::factor(plan, indefinite.view()).unwrap_err();
assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);
# Ok::<(), scioxide_core::SciOxideError>(())
```
"#
);
factor!(
    SparseLdl,
    false,
    r#"Holds the sparse `LDLᴴ` factorization of a Hermitian matrix, which may be
indefinite or singular.

The factor satisfies

```text
A[p[i], p[j]] = (L D Lᴴ)[i, j],
```

where `L` is unit lower triangular, `D` is Hermitian block diagonal with 1×1
and 2×2 blocks, and `p` combines the permutation of the
[analysis](SparseSymmetricSymbolic) with the interchanges of pivoting.

Pivots are chosen by rook pivoting with `α = (1 + √17)/8`. A diagonal entry
is a 1×1 pivot if its magnitude is at least `α` times the largest
off-diagonal magnitude in its column. Otherwise the search alternates between
columns until a diagonal entry passes this test or a 2×2 pivot is found. Ties
go to the smallest index. Interchanges can create fill outside the Cholesky
pattern of the analysis. That fill counts toward `max_factor_entries`, and the
Schur complement toward `max_active_entries`; exceeding either is a
`ResourceLimit` error. Each row of the factor counts as one iteration toward
the iteration limit.

A singular matrix factors with exact zero pivots, which
[`inertia`](Self::inertia) counts, but solving with such a factor fails with
`Singular`. Solve with `solve`, `solve_into` or `solve_block_into`.

# Examples

```
use scioxide_core::SciOxideErrorKind;
use scioxide_sparse::{
    CsrMatrix, LdlInertia, SparseLdl, SparseOrdering, SparseSymmetricLimits,
    SparseSymmetricSymbolic,
};
let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![1, 0], vec![2.0_f64, 2.0])?;
let limits = SparseSymmetricLimits::new(1, 1, 4096);
let plan = SparseSymmetricSymbolic::analyze(a.structural_pattern(), SparseOrdering::Natural, None, limits)?;
let factor = SparseLdl::factor(plan.try_clone()?, a.view())?;
assert_eq!(factor.inertia()?, LdlInertia::new(1, 1, 0)?);
let x = factor.solve(&[2., 4.])?;
assert!((x[0] - 2.0).abs() < 1e-15 && (x[1] - 1.0).abs() < 1e-15);
// A singular matrix factors with zero pivots, but solving it fails.
let zero = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![1, 0], vec![0.0_f64, 0.])?;
let singular = SparseLdl::factor(plan, zero.view())?;
assert_eq!(singular.inertia()?, LdlInertia::new(0, 0, 2)?);
assert_eq!(singular.solve(&[1., 1.]).unwrap_err().kind(), SciOxideErrorKind::Singular);
# Ok::<(), scioxide_core::SciOxideError>(())
```
"#
);
// Analysis of both triangles with the fill-reducing ordering and no limits;
// the one-shot constructors replace the limits before each factorization.
fn analyze_unbounded<S: NumericScalar>(
    matrix: &crate::CsrMatrix<S>,
) -> Result<SparseSymmetricSymbolic, SciOxideError> {
    SparseSymmetricSymbolic::analyze(
        matrix.structural_pattern(),
        crate::SparseOrdering::FillReducing,
        None,
        SparseSymmetricLimits::new(usize::MAX, usize::MAX, usize::MAX),
    )
}
impl<S: NumericScalar> SparseCholesky<S> {
    /// Computes the Cholesky factorization `A = L Lᴴ` of the Hermitian positive
    /// definite matrix `A = matrix`, which stores both triangles.
    ///
    /// The analysis uses
    /// [`SparseOrdering::FillReducing`](crate::SparseOrdering::FillReducing)
    /// (approximate minimum degree), reads both triangles (`triangle = None`)
    /// and checks that the values are exactly Hermitian. The factor stores
    /// exactly the [`cholesky_entries`](SparseSymmetricSymbolic::cholesky_entries)
    /// that the analysis predicts, with no byte limit. The result equals
    /// [`analyze`](SparseSymmetricSymbolic::analyze) followed by
    /// [`factor`](Self::factor) with those limits.
    ///
    /// ```
    /// use scioxide_sparse::{CsrMatrix, SparseCholesky};
    ///
    /// // A = [[4, 2], [2, 5]] = L Lᵀ with L = [[2, 0], [1, 2]].
    /// let a = CsrMatrix::from_triplets(2, 2, &[0, 0, 1, 1], &[0, 1, 0, 1], &[4.0_f64, 2., 2., 5.])?;
    /// let x = SparseCholesky::new(&a)?.solve(&[6., 7.])?;
    /// assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-15));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `matrix` is not square.
    /// - `InvalidInput` if a diagonal entry is not real or the values are not
    ///   exactly Hermitian.
    /// - `NumericalInstability` if a pivot is not positive, or an entry
    ///   overflows or underflows.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn new(matrix: &crate::CsrMatrix<S>) -> Result<Self, SciOxideError> {
        let mut symbolic = analyze_unbounded(matrix)?;
        symbolic.limits = SparseSymmetricLimits::new(symbolic.cholesky_entries(), 0, usize::MAX);
        Self::factor(symbolic, matrix.view())
    }
}
impl<S: NumericScalar> SparseLdl<S> {
    /// Computes the `LDLᴴ` factorization of the Hermitian matrix `A = matrix`,
    /// which stores both triangles and may be indefinite or singular.
    ///
    /// The analysis uses
    /// [`SparseOrdering::FillReducing`](crate::SparseOrdering::FillReducing)
    /// (approximate minimum degree), reads both triangles (`triangle = None`)
    /// and checks that the values are exactly Hermitian. The factor storage `c`
    /// starts at the [`cholesky_entries`](SparseSymmetricSymbolic::cholesky_entries)
    /// of the analysis, with `min(c + n, n(n + 1)/2)` active entries and no
    /// byte limit, and doubles after each `ResourceLimit` up to the
    /// `n(n − 1)/2` entries of a dense strict lower triangle. Without pivot
    /// interchanges, as for every positive definite matrix, the first attempt
    /// succeeds; interchanges can add fill and then cost further
    /// factorizations. The result equals
    /// [`analyze`](SparseSymmetricSymbolic::analyze) with sufficient limits
    /// followed by [`factor`](Self::factor).
    ///
    /// ```
    /// use scioxide_sparse::{CsrMatrix, LdlInertia, SparseLdl};
    ///
    /// // A = [[0, 2], [2, 0]] is indefinite and needs a 2×2 pivot.
    /// let a = CsrMatrix::from_triplets(2, 2, &[0, 1], &[1, 0], &[2.0_f64, 2.0])?;
    /// let factor = SparseLdl::new(&a)?;
    /// assert_eq!(factor.inertia()?, LdlInertia::new(1, 1, 0)?);
    /// let x = factor.solve(&[2., 4.])?;
    /// assert!((x[0] - 2.0).abs() < 1e-15 && (x[1] - 1.0).abs() < 1e-15);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `matrix` is not square.
    /// - `InvalidInput` if a diagonal entry is not real or the values are not
    ///   exactly Hermitian.
    /// - `NumericalInstability` if an entry overflows or underflows.
    /// - `Singular` if a 2×2 pivot block has a zero determinant.
    /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be
    ///   allocated.
    pub fn new(matrix: &crate::CsrMatrix<S>) -> Result<Self, SciOxideError> {
        let symbolic = analyze_unbounded(matrix)?;
        let n = symbolic.dimension();
        let active = crate::capacity::triangle(n);
        crate::capacity::with_capacity(
            symbolic.cholesky_entries(),
            crate::capacity::strict_triangle(n),
            |entries| {
                let mut attempt = symbolic.try_clone()?;
                attempt.limits = SparseSymmetricLimits::new(
                    entries,
                    entries.saturating_add(n).min(active),
                    usize::MAX,
                );
                Self::factor(attempt, matrix.view())
            },
        )
    }
    /// Returns the subdiagonal of `D`: entry `k` is `D[k + 1, k]` where a 2×2
    /// block starts at `k`, and zero elsewhere.
    pub fn off_diagonal(&self) -> &[S] {
        &self.data.off
    }
    /// Returns the 1×1 and 2×2 diagonal blocks of `D`, in order.
    pub fn pivot_blocks(&self) -> impl Iterator<Item = LdlPivotBlock> + '_ {
        self.data.blocks.iter().copied()
    }
    /// Returns the inertia of `D`: the numbers of positive, negative and zero
    /// eigenvalues of its blocks.
    ///
    /// By Sylvester's law of inertia this is the inertia of `A` when the
    /// factorization is exact. Eigenvalues are compared with exact zero, so
    /// rounding errors can move an eigenvalue near zero to either side; use
    /// [`inertia_with_threshold`](Self::inertia_with_threshold) to count small
    /// eigenvalues as zero.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if an eigenvalue of a 2×2 block is not
    ///   representable.
    pub fn inertia(&self) -> Result<LdlInertia, SciOxideError> {
        self.classify(Scaled::ZERO)
    }
    /// Returns the inertia of `D`, counting eigenvalues of magnitude at most
    /// `threshold.threshold(scale)` as zero.
    ///
    /// The threshold applies to the eigenvalues of the blocks of `D`, not to
    /// the eigenvalues of `A`. Congruence does not preserve this count, so it
    /// estimates the numerical rank but does not bound it.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if `scale` is negative, and `NonFinite` if it is NaN or
    ///   infinite.
    /// - `NumericalInstability` if the threshold or an eigenvalue of a 2×2 block
    ///   is not representable.
    pub fn inertia_with_threshold(
        &self,
        threshold: scioxide_core::RankThreshold<S::Real>,
        scale: S::Real,
    ) -> Result<LdlInertia, SciOxideError> {
        self.classify(Scaled::from_value(threshold.threshold(scale)?)?)
    }
    // Shared inertia count.
    fn classify(&self, cutoff: Scaled<S::Real>) -> Result<LdlInertia, SciOxideError> {
        let mut result = LdlInertia::default();
        let mut visit = |v: Scaled<S::Real>| -> Result<(), SciOxideError> {
            let sign = if v.abs_cmp(cutoff) != Ordering::Greater {
                Ordering::Equal
            } else if v.is_sign_negative() {
                Ordering::Less
            } else {
                Ordering::Greater
            };
            result = result.checked_add(LdlInertia::from_sign(sign))?;
            Ok(())
        };
        for block in self.pivot_blocks() {
            let k = block.checked_range(self.data.diagonal.len())?.start;
            if block.size() == 1 {
                visit(Scaled::from_value(self.data.diagonal[k].real_component())?)?;
            } else {
                for v in block_values(
                    self.data.diagonal[k],
                    self.data.off[k],
                    self.data.diagonal[k + 1],
                )? {
                    visit(v)?;
                }
            }
        }
        Ok(result)
    }
}
// Scatters the caller's `matrix` into `w`.
fn prepare<S: NumericScalar>(
    p: &SparseSymmetricSymbolic,
    a: SparseView<'_, S>,
    w: &mut SparseSymmetricWorkspace<S>,
) -> Result<(), SciOxideError> {
    let n = p.dimension();
    let pattern = p.pattern();
    crate::helpers::shape((a.rows(), a.cols()), (n, n))?;
    // The same byte admission and reserved shape that `check` reports as sufficient.
    let requirements = p.workspace_requirements::<S>(())?;
    let _ = w.check(&requirements)?;
    w.fits(&requirements)?;
    w.seen.fill(false);
    for (row, col, v) in a.iter() {
        let pos = position(pattern, row, col)
            .ok_or_else(|| invalid(MATRIX, "numeric pattern differs from analysis"))?;
        if w.seen[pos] {
            return Err(invalid(MATRIX, "duplicate numeric coordinate"));
        }
        w.seen[pos] = true;
        w.source[pos] = v;
    }
    if w.seen.iter().any(|&s| !s) {
        return Err(invalid(
            MATRIX,
            "numeric pattern omits analyzed coordinates",
        ));
    }
    for row in 0..n {
        let d = value(p, &w.source, row, row);
        if d.imaginary_component() != S::Real::ZERO {
            return Err(invalid(MATRIX, "Hermitian diagonal must be real"));
        }
        if p.triangle.is_none() {
            for pos in pattern.row_ptr()[row]..pattern.row_ptr()[row + 1] {
                let col = pattern.col_indices()[pos];
                if w.source[pos] != raw(pattern, &w.source, col, row).conj() {
                    return Err(invalid(MATRIX, "input is not exactly Hermitian"));
                }
            }
        }
    }
    w.candidate.reset(p);
    w.active.clear();
    w.x.fill(S::ZERO);
    w.y.fill(S::ZERO);
    w.mark.fill(n);
    Ok(())
}
fn position(p: StructuralPatternView<'_>, row: usize, col: usize) -> Option<usize> {
    let start = p.row_ptr()[row];
    p.col_indices()[start..p.row_ptr()[row + 1]]
        .binary_search(&col)
        .ok()
        .map(|i| start + i)
}
fn raw<S: NumericScalar>(p: StructuralPatternView<'_>, source: &[S], row: usize, col: usize) -> S {
    position(p, row, col).map_or(S::ZERO, |i| source[i])
}
fn value<S: NumericScalar>(p: &SparseSymmetricSymbolic, source: &[S], row: usize, col: usize) -> S {
    if selected(p.triangle, row, col) {
        raw(p.pattern(), source, row, col)
    } else {
        raw(p.pattern(), source, col, row).conj()
    }
}
// Up-looking Cholesky into the candidate.
fn cholesky<S: NumericScalar>(
    p: &SparseSymmetricSymbolic,
    w: &mut SparseSymmetricWorkspace<S>,
    arithmetic: Arithmetic,
    tick: &mut dyn FnMut() -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    let n = p.dimension();
    let d = &mut w.candidate;
    d.ptr.copy_from_slice(&p.lower_ptr);
    w.cursor.copy_from_slice(&p.lower_ptr[..n]);
    for col in 0..n {
        for &row in &p.lower_rows[p.lower_ptr[col]..p.lower_ptr[col + 1]] {
            d.push((d.p[row], col), S::ZERO, p.limits.max_factor_entries)?;
        }
    }
    for k in 0..n {
        tick()?;
        let mut len = 0;
        visit_reach(
            k,
            &p.upper[p.upper_ptr[k]..p.upper_ptr[k + 1]],
            &p.parent,
            &mut w.mark,
            |i| {
                w.reach[len] = i;
                len += 1;
            },
        );
        w.reach[..len].sort_unstable();
        for &i in &w.reach[..len] {
            w.x[i] = S::ZERO;
        }
        for &(_, i) in &p.upper[p.upper_ptr[k]..p.upper_ptr[k + 1]] {
            if i < k {
                w.x[i] = value(p, &w.source, d.p[k], d.p[i]);
            }
        }
        for &i in &w.reach[..len] {
            let l = arithmetic.divide(w.x[i], d.diagonal[i], crate::helpers::Projection::Strict)?;
            for e in &d.entries[d.ptr[i]..w.cursor[i]] {
                let row = d.inverse[e.row];
                w.x[row] = arithmetic.subtract_product(w.x[row], l, e.value.conj())?;
            }
            d.entries[w.cursor[i]].value = l;
            w.cursor[i] += 1;
        }
        d.diagonal[k] = arithmetic.cholesky_diagonal(
            value(p, &w.source, d.p[k], d.p[k]).real_component(),
            w.reach[..len]
                .iter()
                .map(|&i| d.entries[w.cursor[i] - 1].value),
        )?;
    }
    Ok(())
}
fn maximum<S: NumericScalar>(
    active: &Active<S>,
    d: &Data<S>,
    (k, col): (usize, usize),
    arithmetic: Arithmetic,
) -> Result<(usize, S), SciOxideError> {
    let mut best = col;
    let mut maximum = S::ZERO;
    let old = d.p[col];
    for (other, value) in active.column(old) {
        let row = d.inverse[other];
        if row < k || row == col {
            continue;
        }
        let comparison = arithmetic.compare_magnitude(value, maximum)?;
        if comparison == Ordering::Greater || (comparison == Ordering::Equal && row < best) {
            best = row;
            maximum = value;
        }
    }
    Ok((best, maximum))
}
// Bounded rook pivot choice at step `k`.
#[derive(Clone, Copy)]
enum PivotChoice {
    One(usize),
    Two(usize, usize),
}

fn choose<S: NumericScalar>(
    active: &Active<S>,
    d: &Data<S>,
    k: usize,
    arithmetic: Arithmetic,
) -> Result<PivotChoice, SciOxideError> {
    let alpha = (S::Real::ONE + (S::Real::TEN + S::Real::TEN - S::Real::TWO - S::Real::ONE).sqrt())
        / (S::Real::TWO * S::Real::TWO * S::Real::TWO);
    let sufficient = |i, maximum: S| -> Result<bool, SciOxideError> {
        let diagonal = active.get(d.p[i], d.p[i]);
        if arithmetic.0 == Accuracy::Standard {
            let (diagonal, maximum) = (diagonal.abs(), maximum.abs());
            if diagonal.is_finite()
                && maximum.is_finite()
                && let Some(threshold) = ordinary_product(alpha, maximum)
            {
                return Ok(diagonal >= threshold);
            }
        }
        Ok(crate::helpers::scaled(diagonal)?.magnitude()?.abs_cmp(
            crate::helpers::scaled(maximum)?
                .magnitude()?
                .mul(Scaled::from_value(alpha)?)?,
        ) != Ordering::Less)
    };
    let (mut imax, mut colmax) = maximum(active, d, (k, k), arithmetic)?;
    if colmax == S::ZERO || sufficient(k, colmax)? {
        return Ok(PivotChoice::One(k));
    }
    let mut prev = k;
    loop {
        let (jmax, rowmax) = maximum(active, d, (k, imax), arithmetic)?;
        if sufficient(imax, rowmax)? {
            return Ok(PivotChoice::One(imax));
        }
        if jmax == prev || arithmetic.compare_magnitude(rowmax, colmax)? != Ordering::Greater {
            return Ok(PivotChoice::Two(prev, imax));
        }
        prev = imax;
        imax = jmax;
        colmax = rowmax;
    }
}
// Bounded-rook LDL into the candidate.
fn ldl<S: NumericScalar>(
    p: &SparseSymmetricSymbolic,
    w: &mut SparseSymmetricWorkspace<S>,
    arithmetic: Arithmetic,
    tick: &mut dyn FnMut() -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    let n = p.dimension();
    for &(col, row) in &p.upper {
        let i = p.permutation.new_to_old()[row];
        let j = p.permutation.new_to_old()[col];
        w.active.set(i, j, value(p, &w.source, i, j))?;
    }
    let d = &mut w.candidate;
    let mut k = 0;
    while k < n {
        tick()?;
        let (first, second, block) = match choose(&w.active, d, k, arithmetic)? {
            PivotChoice::One(first) => (first, first, LdlPivotBlock::One { start: k }),
            PivotChoice::Two(first, second) => (first, second, LdlPivotBlock::Two { start: k }),
        };
        let width = block.checked_range(n)?.len();
        d.swap(k, first);
        if width == 2 {
            tick()?;
            let second = if second == k { first } else { second };
            d.swap(k + 1, second);
        }
        let a = w.active.get(d.p[k], d.p[k]);
        let (b, c) = if width == 2 {
            (
                w.active.get(d.p[k + 1], d.p[k]),
                w.active.get(d.p[k + 1], d.p[k + 1]),
            )
        } else {
            (S::ZERO, S::ZERO)
        };
        d.diagonal[k] = a;
        d.blocks.push(block);
        if width == 2 {
            d.diagonal[k + 1] = c;
            d.off[k] = b;
        }
        let mut count = 0;
        for pivot in k..k + width {
            for (old, value) in w.active.column(d.p[pivot]) {
                let i = d.inverse[old];
                if i < k + width {
                    continue;
                }
                if w.mark[i] != k {
                    w.mark[i] = k;
                    w.x[i] = S::ZERO;
                    w.y[i] = S::ZERO;
                    w.reach[count] = i;
                    count += 1;
                }
                if pivot == k {
                    w.x[i] = value;
                } else {
                    w.y[i] = value;
                }
            }
        }
        w.reach[..count].sort_unstable();
        for &i in &w.reach[..count] {
            let (x, y) = (w.x[i], w.y[i]);
            let (l, m) = if width == 2 {
                let (l, m) = block_solve(a, b, c, x.conj(), y.conj(), arithmetic)?;
                (l.conj(), m.conj())
            } else {
                (
                    arithmetic.divide(x, a, crate::helpers::Projection::Strict)?,
                    S::ZERO,
                )
            };
            w.x[i] = l;
            w.y[i] = m;
        }
        d.ptr[k] = d.entries.len();
        for &i in &w.reach[..count] {
            if w.x[i] != S::ZERO {
                d.push((d.p[i], k), w.x[i], p.limits.max_factor_entries)?;
            }
        }
        d.ptr[k + 1] = d.entries.len();
        if width == 2 {
            for &i in &w.reach[..count] {
                if w.y[i] != S::ZERO {
                    d.push((d.p[i], k + 1), w.y[i], p.limits.max_factor_entries)?;
                }
            }
            d.ptr[k + 2] = d.entries.len();
        }
        for &old in &d.p[k..k + width] {
            w.active.eliminate(old);
        }
        for (ii, &i) in w.reach[..count].iter().enumerate() {
            for &j in &w.reach[..=ii] {
                let prior = w.active.get(d.p[i], d.p[j]);
                let ordinary = if arithmetic.0 == Accuracy::Standard {
                    (|| {
                        let u = ordinary_sum(
                            ordinary_product(a, w.x[j].conj())?,
                            ordinary_product(b.conj(), w.y[j].conj())?,
                        )?;
                        let v = ordinary_sum(
                            ordinary_product(b, w.x[j].conj())?,
                            ordinary_product(c, w.y[j].conj())?,
                        )?;
                        let update = ordinary_sum(
                            ordinary_product(w.x[i], u)?,
                            ordinary_product(w.y[i], v)?,
                        )?;
                        let result = ordinary_sum(prior, -update)?;
                        // Hermitian diagonals are real: only that component is judged.
                        if i == j {
                            let real = result.real_component();
                            normal_or_zero(real).then(|| S::from_real(real))
                        } else {
                            normal_components(result).then_some(result)
                        }
                    })()
                } else {
                    None
                };
                let result = if let Some(result) = ordinary {
                    result
                } else {
                    let u = crate::helpers::scaled(a)?
                        .mul(crate::helpers::scaled(w.x[j].conj())?)?
                        .add(
                            crate::helpers::scaled(b.conj())?
                                .mul(crate::helpers::scaled(w.y[j].conj())?)?,
                        )?;
                    let v = crate::helpers::scaled(b)?
                        .mul(crate::helpers::scaled(w.x[j].conj())?)?
                        .add(
                            crate::helpers::scaled(c)?
                                .mul(crate::helpers::scaled(w.y[j].conj())?)?,
                        )?;
                    let update = crate::helpers::scaled(w.x[i])?
                        .mul(u)?
                        .add(crate::helpers::scaled(w.y[i])?.mul(v)?)?;
                    let result =
                        crate::helpers::scaled(w.active.get(d.p[i], d.p[j]))?.sub(update)?;
                    let result = if i == j {
                        ScaledValue::from_components(result.real(), Scaled::ZERO)
                    } else {
                        result
                    };
                    crate::helpers::project(result, crate::helpers::Projection::Strict)?
                };
                w.active.set(d.p[i], d.p[j], result)?;
            }
        }
        k += width;
    }
    Ok(())
}
// Solves one Hermitian 2x2 pivot block.
pub(super) fn block_solve<S: NumericScalar>(
    a: S,
    b: S,
    c: S,
    x: S,
    y: S,
    arithmetic: Arithmetic,
) -> Result<(S, S), SciOxideError> {
    if arithmetic.0 == Accuracy::Standard {
        let candidate = (|| {
            // With real diagonals, `a*c` and `b*conj(b)` have exactly zero
            // imaginary parts, so the determinant is judged by its real part.
            let det = ordinary_sum(ordinary_product(a, c)?, -ordinary_product(b, b.conj())?)?;
            let u = ordinary_sum(ordinary_product(c, x)?, -ordinary_product(b.conj(), y)?)?;
            let v = ordinary_sum(ordinary_product(a, y)?, -ordinary_product(b, x)?)?;
            // A zero or subnormal determinant and subnormal numerators replay
            // Cramer's rule before division can amplify a lost bit.
            (det.real_component().abs() >= S::Real::MIN_POSITIVE
                && normal_components(u)
                && normal_components(v))
            .then_some((u, v, det))
        })();
        if let Some((u, v, det)) = candidate {
            return Ok((
                arithmetic.divide(u, det, crate::helpers::Projection::Strict)?,
                arithmetic.divide(v, det, crate::helpers::Projection::Strict)?,
            ));
        }
    }
    let [x, y] = crate::backend::hermitian::solve_block(a, b, c, x, y)?;
    Ok((
        crate::helpers::project(x, crate::helpers::Projection::Strict)?,
        crate::helpers::project(y, crate::helpers::Projection::Strict)?,
    ))
}

// Accept a component of an ordinary result built from underflow-screened
// products. A sum that cancels exactly to zero commits no rounding, so zero
// keeps the composed ordinary bound; subnormal components take the scaled path.
fn normal_or_zero<R: RealScalar>(value: R) -> bool {
    value == R::ZERO || value.abs() >= R::MIN_POSITIVE
}
fn normal_components<S: NumericScalar>(value: S) -> bool {
    normal_or_zero(value.real_component())
        && (S::IS_REAL || normal_or_zero(value.imaginary_component()))
}

impl<S: NumericScalar> Data<S> {
    fn retained_bytes(&self) -> usize {
        self.entries.capacity() * size_of::<Entry<S>>()
            + (self.diagonal.capacity() + self.off.capacity()) * size_of::<S>()
            + self.blocks.capacity() * size_of::<LdlPivotBlock>()
            + (self.ptr.capacity() + self.p.capacity() + self.inverse.capacity())
                * size_of::<usize>()
    }
}

impl<S: NumericScalar> Default for SparseSymmetricWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: NumericScalar> SparseSymmetricWorkspace<S> {
    /// Creates an empty workspace. Reserve it with
    /// [`try_reserve`](Self::try_reserve) before use.
    pub fn new() -> Self {
        Self {
            epoch: crate::factor_solve::Epoch::new(),
            candidate: Data {
                entries: Vec::new(),
                ptr: Vec::new(),
                diagonal: Vec::new(),
                off: Vec::new(),
                blocks: Vec::new(),
                p: Vec::new(),
                inverse: Vec::new(),
            },
            source: Vec::new(),
            seen: Vec::new(),
            x: Vec::new(),
            y: Vec::new(),
            mark: Vec::new(),
            reach: Vec::new(),
            cursor: Vec::new(),
            active: Active::empty(),
            factor_entries: 0,
        }
    }
    /// Returns the bytes this workspace holds, including unused capacity: the
    /// second factor, scratch and the `LDLᴴ` Schur complement storage.
    /// [`check`](Self::check) adds the analysis and one factor.
    pub fn retained_bytes(&self) -> usize {
        self.candidate.retained_bytes()
            + (self.source.capacity() + self.x.capacity() + self.y.capacity()) * size_of::<S>()
            + self.seen.capacity()
            + (self.mark.capacity() + self.reach.capacity() + self.cursor.capacity())
                * size_of::<usize>()
            + self.active.retained_bytes()
    }
    /// Returns the record of the most recent factorization that used this
    /// workspace.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.epoch.record()
    }
    /// Checks whether this workspace fits the requirements `r`.
    ///
    /// Returns `Sufficient` when the order, the number of pattern entries and
    /// the entry limits match `r` exactly. Analyses that differ only in their
    /// byte limit share a workspace. `refactor` rejects a workspace that needs
    /// growth with `DimensionMismatch`.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the bytes held by the workspace, plus the analysis
    ///   and one factor, exceed the byte limit of `r`.
    /// - `SizeOverflow` if a size overflows `usize`.
    pub fn check(
        &self,
        r: &SparseSymmetricRequirements,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        let bytes = add(
            add(
                r.symbolic_bytes,
                r.factor_bytes,
                scioxide_core::quantity!("workspace_bytes"),
            )?,
            self.retained_bytes(),
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        if bytes > r.limits.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                bytes,
                r.limits.max_workspace_bytes,
            ));
        }
        Ok(if self.fits(r).is_ok() {
            scioxide_core::Capacity::Sufficient
        } else {
            scioxide_core::Capacity::NeedsGrowth
        })
    }
    /// Replaces the storage of this workspace with storage that fits `r`.
    ///
    /// If this fails, the workspace keeps its earlier storage.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if `r.total_bytes()`, or the current and new storage
    ///   together, exceed the byte limit of `r`.
    /// - `SizeOverflow` if a size overflows `usize`.
    /// - `AllocationFailure` if memory cannot be allocated.
    pub fn try_reserve(&mut self, r: &SparseSymmetricRequirements) -> Result<(), SciOxideError> {
        if r.total_bytes > r.limits.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                r.total_bytes,
                r.limits.max_workspace_bytes,
            ));
        }
        if self.check(r)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        let projected = add(
            r.total_bytes,
            self.retained_bytes(),
            scioxide_core::quantity!("workspace_bytes"),
        )?;
        if projected > r.limits.max_workspace_bytes {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("workspace_bytes"),
                projected,
                r.limits.max_workspace_bytes,
            ));
        }
        let replacement = Self::reserve(r)?;
        let _ = replacement.check(r)?;
        *self = replacement;
        Ok(())
    }
}
