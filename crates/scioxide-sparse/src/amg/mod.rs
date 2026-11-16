//! Smoothed-aggregation algebraic multigrid: hierarchy setup, cycles and the
//! owning preconditioner.
//!
use crate::helpers::admit;
use crate::helpers::finite;
use crate::helpers::invalid;
use crate::{CsrMatrix, SparseSolveOptions, SparseView};
use alloc::vec::Vec;
use core::mem::size_of;
use scioxide_core::backend::{
    checked_add as add, checked_mul as mul, try_copy_slice, try_filled, try_with_capacity,
};
use scioxide_core::{
    Argument, BlockVectorView, ExecutionContext, LinearOperator, NumericScalar, ProductSum,
    RealScalar, SciOxideError, SolveScope,
};
mod cycle;
mod owned;
mod setup;
pub use cycle::{AmgCycle, AmgCycleKind, AmgCycleOptions, AmgLevelView, AmgSmootherView};
pub use owned::AmgPreconditioner;
fn scalar<S: NumericScalar>(value: f64) -> Result<S, SciOxideError> {
    Ok(S::from_real(S::Real::try_from_f64_lossy(value)?))
}
/// Holds the depth and storage limits of an algebraic multigrid setup.
///
/// There is no default; every limit must be chosen. [`AmgHierarchy::build`]
/// checks them.
///
/// # Examples
///
/// ```
/// use scioxide_sparse::AmgLimits;
/// let limits = AmgLimits::new(8, 10_000, 1 << 20).with_max_levels(4);
/// assert_eq!((limits.max_levels(), limits.max_entries()), (4, 10_000));
/// let default = AmgLimits::default();
/// assert_eq!(
///     (default.max_levels(), default.max_setup_bytes()),
///     (64, usize::MAX)
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmgLimits {
    max_levels: usize,
    max_entries: usize,
    max_setup_bytes: usize,
}
/// Returns limits that only bound the hierarchy depth: 64 levels, the largest
/// accepted, and no entry or byte limit (`usize::MAX`). Setup compares the
/// limits and never sizes storage by them.
impl Default for AmgLimits {
    fn default() -> Self {
        Self::new(64, usize::MAX, usize::MAX)
    }
}
impl AmgLimits {
    /// Creates limits on the number of levels, on the entries of each setup
    /// matrix and on setup storage in bytes.
    pub const fn new(max_levels: usize, max_entries: usize, max_setup_bytes: usize) -> Self {
        Self {
            max_levels,
            max_entries,
            max_setup_bytes,
        }
    }
    /// Returns the largest number of levels, the finest and coarsest included.
    /// It must be between 1 and 64.
    pub const fn max_levels(self) -> usize {
        self.max_levels
    }
    /// Sets the largest number of levels.
    pub const fn with_max_levels(mut self, value: usize) -> Self {
        self.max_levels = value;
        self
    }
    /// Returns the limit on stored entries during setup.
    ///
    /// The fine matrix, every tentative basis (`rows · modes` entries) and every
    /// prolongation hold at most this many entries. Each sparse product `A B`
    /// must satisfy `nnz(A) + nnz(B) + paths ≤ max_entries`, where `paths` counts
    /// the scalar products `a_ik b_kj` before equal positions are summed.
    pub const fn max_entries(self) -> usize {
        self.max_entries
    }
    /// Sets the limit on stored entries.
    pub const fn with_max_entries(mut self, value: usize) -> Self {
        self.max_entries = value;
        self
    }
    /// Returns the limit on bytes allocated during setup.
    ///
    /// Setup adds an upper bound of every allocation, temporaries included, so
    /// the limit also bounds peak storage. Inputs and allocator overhead are not
    /// counted.
    pub const fn max_setup_bytes(self) -> usize {
        self.max_setup_bytes
    }
    /// Sets the limit on setup bytes.
    pub const fn with_max_setup_bytes(mut self, value: usize) -> Self {
        self.max_setup_bytes = value;
        self
    }
}
/// Holds the coarsening options of [`AmgHierarchy::build`].
///
/// The [`Accuracy`](scioxide_core::Accuracy) of setup and its iteration limit
/// come from the [`SparseSolveOptions`] passed to `build`.
///
/// # Examples
///
/// ```
/// use scioxide_sparse::AmgSetupOptions;
/// let options = AmgSetupOptions::<f64>::new()
///     .with_coarse_size(4)
///     .with_strength(0.25);
/// assert_eq!((options.coarse_size(), options.strength()), (4, 0.25));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmgSetupOptions<R: RealScalar> {
    coarse_size: usize,
    strength: R,
}
impl<R: RealScalar> Default for AmgSetupOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}
impl<R: RealScalar> AmgSetupOptions<R> {
    /// Returns the default options: coarse size 16 and strength threshold
    /// `1/16`.
    pub fn new() -> Self {
        let four = R::TWO * R::TWO;
        Self {
            coarse_size: 16,
            strength: R::ONE / (four * four),
        }
    }
    /// Sets the dimension at or below which coarsening stops. It must be
    /// positive.
    pub const fn with_coarse_size(mut self, value: usize) -> Self {
        self.coarse_size = value;
        self
    }
    /// Returns the coarse size.
    pub const fn coarse_size(self) -> usize {
        self.coarse_size
    }
    /// Sets the strength threshold `θ`, a finite number in `[0, 1]`.
    ///
    /// Node `j` is strongly connected to node `i` if
    /// `|a_ij| / (√a_ii · √a_jj) ≥ θ`. Only strong connections join nodes into
    /// aggregates.
    pub const fn with_strength(mut self, value: R) -> Self {
        self.strength = value;
        self
    }
    /// Returns the strength threshold.
    pub const fn strength(self) -> R {
        self.strength
    }
}
/// Reports why [`AmgHierarchy::build`] stopped coarsening.
///
/// Only [`CoarseSize`](Self::CoarseSize) guarantees that the coarsest matrix
/// is no larger than the requested coarse size. In the other cases the
/// coarsest matrix can be large; check its size before you factor it.
///
/// # Examples
///
/// ```
/// use scioxide_core::{BlockVectorView, ExecutionContext};
/// use scioxide_sparse::{
///     AmgHierarchy, AmgLimits, AmgSetupOptions, AmgSetupTermination, CsrMatrix, SparseSolveOptions,
/// };
/// let chain = CsrMatrix::from_raw(
///     3,
///     3,
///     vec![0, 2, 5, 7],
///     vec![0, 1, 0, 1, 2, 1, 2],
///     vec![2.0_f64, -1., -1., 2., -1., -1., 2.],
/// )?;
/// let diagonal = CsrMatrix::from_raw(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![2.0_f64; 3])?;
/// let modes = BlockVectorView::new(&[1.0; 3], 3, 1)?;
/// let termination = |a: &CsrMatrix<f64>, levels| {
///     AmgHierarchy::build(
///         a.view(),
///         modes,
///         AmgSetupOptions::new().with_coarse_size(1),
///         AmgLimits::new(levels, 1000, 1 << 20),
///         SparseSolveOptions::new(),
///         ExecutionContext::sequential(),
///     )
///     .map(|h| h.diagnostics().termination())
/// };
/// assert_eq!(termination(&chain, 4)?, AmgSetupTermination::CoarseSize);
/// // One level leaves the 3 × 3 matrix to the coarse solver.
/// assert_eq!(termination(&chain, 1)?, AmgSetupTermination::LevelLimit);
/// // Without strong connections every node is its own aggregate.
/// assert_eq!(termination(&diagonal, 4)?, AmgSetupTermination::Stagnation);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AmgSetupTermination {
    /// The coarsest matrix has at most the requested coarse size.
    CoarseSize,
    /// The level limit was reached before the coarse size.
    LevelLimit,
    /// Aggregation did not reduce the dimension, because there were not enough
    /// strong connections.
    Stagnation,
}
/// Reports the size of a multigrid hierarchy and the storage of its setup.
///
/// The complexities describe the cost of a cycle. They do not predict its
/// convergence rate.
///
/// # Examples
///
/// ```
/// use scioxide_core::{BlockVectorView, ExecutionContext};
/// use scioxide_sparse::{
///     AmgHierarchy, AmgLimits, AmgSetupOptions, AmgSetupTermination, CsrMatrix, SparseSolveOptions,
/// };
/// let a = CsrMatrix::from_raw(
///     3,
///     3,
///     vec![0, 2, 5, 7],
///     vec![0, 1, 0, 1, 2, 1, 2],
///     vec![2.0_f64, -1., -1., 2., -1., -1., 2.],
/// )?;
/// let h = AmgHierarchy::build(
///     a.view(),
///     BlockVectorView::new(&[1.0; 3], 3, 1)?,
///     AmgSetupOptions::new().with_coarse_size(1),
///     AmgLimits::new(4, 1000, 1 << 20),
///     SparseSolveOptions::new(),
///     ExecutionContext::sequential(),
/// )?;
/// let d = h.diagnostics();
/// // Three fine rows form one aggregate: 7 + 1 operator entries, P and R with 3 each.
/// assert_eq!(
///     (d.levels(), d.total_rows(), d.operator_entries()),
///     (2, 4, 8)
/// );
/// assert_eq!(
///     (d.transfer_entries(), d.termination()),
///     (6, AmgSetupTermination::CoarseSize)
/// );
/// assert!((d.operator_complexity().unwrap() - 8.0 / 7.0).abs() < 1e-15);
/// assert!(d.setup_bytes_bound() <= 1 << 20);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmgDiagnostics {
    levels: usize,
    total_rows: usize,
    operator_entries: usize,
    coarsening_ratio: Option<f64>,
    operator_complexity: Option<f64>,
    transfer_entries: usize,
    setup_bytes_bound: usize,
    termination: AmgSetupTermination,
}
impl AmgDiagnostics {
    /// Returns the number of levels.
    pub const fn levels(self) -> usize {
        self.levels
    }
    /// Returns the sum of the dimensions of all levels.
    pub const fn total_rows(self) -> usize {
        self.total_rows
    }
    /// Returns the number of stored entries of all level matrices, without the
    /// transfer operators.
    pub const fn operator_entries(self) -> usize {
        self.operator_entries
    }
    /// Returns the fine dimension divided by the dimension of the second level,
    /// which counts one row per aggregate and mode. Returns `None` for a
    /// single level.
    pub const fn coarsening_ratio(self) -> Option<f64> {
        self.coarsening_ratio
    }
    /// Returns the operator complexity: the stored entries of all level
    /// matrices divided by those of the fine matrix.
    ///
    /// It is 1 for a single nonempty level and `None` if the fine matrix has no
    /// entries.
    pub const fn operator_complexity(self) -> Option<f64> {
        self.operator_complexity
    }
    /// Returns the number of stored entries of all prolongations and
    /// restrictions.
    pub const fn transfer_entries(self) -> usize {
        self.transfer_entries
    }
    /// Returns the upper bound on setup bytes that was checked against
    /// [`AmgLimits::max_setup_bytes`].
    pub const fn setup_bytes_bound(self) -> usize {
        self.setup_bytes_bound
    }
    /// Returns why coarsening stopped.
    pub const fn termination(self) -> AmgSetupTermination {
        self.termination
    }
}
struct Level<S: NumericScalar> {
    a: CsrMatrix<S>,
    inverse: Vec<S>,
    weight: S::Real,
    modes: Vec<S>,
    aggregates: Vec<usize>,
    tentative: Option<CsrMatrix<S>>,
    p: Option<CsrMatrix<S>>,
    r: Option<CsrMatrix<S>>,
}
/// Holds a smoothed-aggregation multigrid hierarchy for a Hermitian positive
/// definite matrix.
///
/// Each level `ℓ` has a matrix `A_ℓ`. Below the finest level,
///
/// ```text
/// P_ℓ = (I − ω D_ℓ⁻¹ A_ℓ) T_ℓ,   R_ℓ = P_ℓᴴ,   A_(ℓ+1) = R_ℓ A_ℓ P_ℓ,
/// ```
///
/// where `D_ℓ` is the diagonal of `A_ℓ`, `T_ℓ` is the tentative prolongation
/// and `ω = 4 / (3 ρ̂)` with `ρ̂ = max(1, maxᵢ Σⱼ |a_ij| / a_ii)`, an upper bound
/// of the spectral radius of `D_ℓ⁻¹ A_ℓ`. Each coarse matrix is made exactly
/// Hermitian.
///
/// Nodes are grouped into aggregates of strongly connected nodes (see
/// [`AmgSetupOptions::with_strength`]) in three passes over the rows in
/// ascending order. The first pass makes aggregates from disjoint strong
/// neighborhoods. The second attaches each remaining node to the aggregate of
/// its first strong neighbor from the first pass. The third groups the rest;
/// an isolated node forms its own aggregate. With the default strength, a
/// path-graph Laplacian with `n` nodes has `⌈n/3⌉` aggregates. On each
/// aggregate, `T_ℓ` holds an orthonormal basis of the near-nullspace modes
/// restricted to it (modified Gram–Schmidt, applied twice), so the next level
/// has one row per aggregate and mode.
///
/// The hierarchy does not include a coarse solver. Use
/// [`AmgPreconditioner`], which factors the coarsest matrix by Cholesky, or
/// [`cycle`](Self::cycle) with a coarse solver of your own.
///
/// # Examples
///
/// ```
/// use scioxide_core::{BlockVectorView, ExecutionContext};
/// use scioxide_sparse::{AmgHierarchy, AmgLimits, AmgSetupOptions, CsrMatrix, SparseSolveOptions};
/// # fn main() -> Result<(), scioxide_core::SciOxideError> {
/// let a = CsrMatrix::from_raw(
///     3,
///     3,
///     vec![0, 2, 5, 7],
///     vec![0, 1, 0, 1, 2, 1, 2],
///     vec![2., -1., -1., 2., -1., -1., 2.],
/// )?;
/// let modes = BlockVectorView::new(&[1.; 3], 3, 1)?;
/// let h = AmgHierarchy::build(
///     a.view(),
///     modes,
///     AmgSetupOptions::new().with_coarse_size(1),
///     AmgLimits::new(4, 1000, 1_000_000),
///     SparseSolveOptions::new(),
///     ExecutionContext::sequential(),
/// )?;
/// // The three nodes form one aggregate, so the coarsening ratio is 3.
/// assert_eq!(h.diagnostics().coarsening_ratio(), Some(3.));
/// assert!(h.diagnostics().operator_complexity().is_some());
/// // A zero coarse size is rejected.
/// let invalid = AmgSetupOptions::new().with_coarse_size(0);
/// assert!(
///     AmgHierarchy::build(
///         a.view(),
///         modes,
///         invalid,
///         AmgLimits::new(4, 1000, 1_000_000),
///         SparseSolveOptions::new(),
///         ExecutionContext::sequential()
///     )
///     .is_err()
/// );
/// # Ok(())
/// # }
/// ```
///
/// # References
///
/// - P. Vaněk, J. Mandel and M. Brezina, "Algebraic multigrid by smoothed
///   aggregation for second and fourth order elliptic problems", *Computing* 56
///   (1996), 179–196.
/// - U. Trottenberg, C. W. Oosterlee and A. Schüller, *Multigrid*, Academic
///   Press, 2001.
pub struct AmgHierarchy<S: NumericScalar> {
    levels: Vec<Level<S>>,
    mode_count: usize,
    diagnostics: AmgDiagnostics,
}
impl<S: NumericScalar> AmgHierarchy<S> {
    /// Builds the hierarchy for `matrix` from the near-nullspace `modes`, one
    /// column per mode.
    ///
    /// For a scalar diffusion problem, `modes` is the constant vector; for
    /// elasticity it holds the rigid-body modes. Coarsening stops at the coarse
    /// size, at the level limit or when aggregation stalls, and
    /// [`AmgDiagnostics::termination`] reports which. The matrix must be
    /// Hermitian positive definite. Setup checks that it is exactly Hermitian
    /// with a positive real diagonal, but it does not check definiteness.
    ///
    /// `options` sets the [`Accuracy`](scioxide_core::Accuracy) of the Galerkin
    /// products and the iteration limit. Setup runs sequentially. It counts one
    /// iteration at the start and one at the end, one per row of each level for
    /// the diagonal, and, on each coarsened level, one per row for each of the
    /// four aggregation passes and for smoothing `T`, one per aggregate for `T`,
    /// and one per output row plus one for each of the products `A T`, `A P` and
    /// `R (A P)`.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `matrix` is not square or `modes` does not have
    ///   one row per matrix row.
    /// - `InvalidInput` if `modes` has no columns, the coarse size is zero, the
    ///   strength is not in `[0, 1]`, the level limit is not in `1..=64`, the
    ///   matrix is not exactly Hermitian or has a diagonal entry that is not
    ///   real and positive, or the modes vanish on an aggregate.
    /// - `NonFinite` if `modes` contains NaN or infinity.
    /// - `ResourceLimit` if a matrix or product exceeds the entry limit or setup
    ///   exceeds the byte limit.
    /// - `NumericalInstability` if a value overflows or underflows.
    /// - `ConvergenceFailure` if the iteration limit is reached;
    ///   `ExecutionStopped` on cancellation or a deadline.
    /// - `AllocationFailure` if storage cannot be allocated.
    pub fn build<'s, 'c>(
        matrix: SparseView<'_, S>,
        modes: BlockVectorView<'_, S>,
        setup: AmgSetupOptions<S::Real>,
        limits: AmgLimits,
        options: SparseSolveOptions,
        scope: impl Into<SolveScope<'s, 'c>>,
    ) -> Result<Self, SciOxideError> {
        crate::factor_solve::Epoch::new().run(scope, options, |context| {
            setup::build(
                matrix,
                modes,
                setup,
                limits,
                options.accuracy(),
                &mut || admit(context),
            )
        })
    }
    /// Returns the setup diagnostics.
    pub const fn diagnostics(&self) -> AmgDiagnostics {
        self.diagnostics
    }
    /// Returns the number of near-nullspace modes, which is the same on every
    /// level.
    pub const fn mode_count(&self) -> usize {
        self.mode_count
    }
    /// Returns the matrix of level `level`, or `None` if there is no such level.
    pub fn matrix(&self, level: usize) -> Option<&CsrMatrix<S>> {
        self.levels.get(level).map(|l| &l.a)
    }
    /// Returns the matrix of the coarsest level, to be solved by a coarse
    /// solver such as [`SparseCholesky`](crate::SparseCholesky).
    pub fn coarse_matrix(&self) -> &CsrMatrix<S> {
        &self.levels[self.levels.len() - 1].a
    }
    /// Returns the near-nullspace modes of level `level` in column-major order,
    /// or `None` if there is no such level.
    pub fn modes(&self, level: usize) -> Option<&[S]> {
        self.levels.get(level).map(|l| l.modes.as_slice())
    }
    /// Returns the aggregate index of each row of level `level`. The slice is
    /// empty on the coarsest level, and `None` is returned if there is no such
    /// level.
    pub fn aggregates(&self, level: usize) -> Option<&[usize]> {
        self.levels.get(level).map(|l| l.aggregates.as_slice())
    }
    /// Returns the tentative prolongation `T` of level `level`, or `None` on the
    /// coarsest level or if there is no such level.
    pub fn tentative(&self, level: usize) -> Option<&CsrMatrix<S>> {
        self.levels.get(level).and_then(|l| l.tentative.as_ref())
    }
    /// Returns the smoothed prolongation `P` of level `level`, or `None` on the
    /// coarsest level or if there is no such level.
    pub fn prolongation(&self, level: usize) -> Option<&CsrMatrix<S>> {
        self.levels.get(level).and_then(|l| l.p.as_ref())
    }
    /// Returns level `level` as an [`AmgLevelView`] with the damped Jacobi
    /// smoother `ω D⁻¹`, or `None` if there is no such level.
    pub fn level(&self, level: usize) -> Option<AmgLevelView<'_, S>> {
        self.levels.get(level).map(|l| AmgLevelView {
            operator: &l.a,
            prolongation: l.p.as_ref().map(|p| p as &(dyn LinearOperator<S> + Sync)),
            restriction: l.r.as_ref().map(|r| r as &(dyn LinearOperator<S> + Sync)),
            smoother: AmgSmootherView::Jacobi {
                inverse_diagonal: &l.inverse,
                weight: l.weight,
            },
        })
    }
    /// Creates a multigrid cycle on this hierarchy, with `coarse` applying the
    /// inverse of the coarsest matrix.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `coarse` does not have the shape of the coarsest
    ///   matrix.
    /// - `SizeOverflow` if the scratch length overflows `usize`.
    pub fn cycle<'a>(
        &'a self,
        coarse: &'a (dyn LinearOperator<S> + Sync),
        options: AmgCycleOptions,
    ) -> Result<AmgCycle<'a, S>, SciOxideError> {
        AmgCycle::from_owned(self, coarse, options)
    }
}
