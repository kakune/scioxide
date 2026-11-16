//! Workspace, workspace requests and attempt records shared by the iterative solvers.

use super::arnoldi;
use super::bicgstab::BicgstabStorage;
use super::biorthogonal::{self, Bicg, Cgs, Qmr, Tfqmr};
use super::cg::CgStorage;
use super::krylov;
use super::stationary::{Richardson, StationaryStorage};
use crate::Preconditioning;
use crate::workspace::SolveSlot;
use crate::{
    IterativeSolveReport, LinalgExecutionContext, LinearSolveTermination, Vector, VectorView,
    VectorViewMut,
};
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::mem::size_of;
use scioxide_core::backend::{checked_add, checked_mul, try_reserve};
use scioxide_core::{
    AdjointOperator, AttemptOutcome, AttemptRecord, Budget, BudgetKind, Capacity, LinearOperator,
    NumericScalar, PartialResultTrust, ProblemToken, RealScalar, SciOxideError, SolveBudgets,
    SolveIdentity,
};

type Result<T> = core::result::Result<T, SciOxideError>;

/// Sizes of an iterative solve: operator shape, scratch lengths and right-hand-side count.
///
/// A request holds sizes only. It does not borrow the operator or preconditioner,
/// so one request can size workspaces for several solvers. Pass it to a solver's
/// `workspace_requirements` or `workspace`.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Budget, ResidualTolerance};
/// use scioxide_linalg::{CgSolver, IterativeWorkspaceRequest, Matrix};
/// let a = Matrix::from_vec(2, 2, vec![4.0, 1.0, 1.0, 3.0])?;
/// let a = a.view().try_finite()?;
/// let solver = CgSolver::<f64>::new(ResidualTolerance::new(1e-12, 0.0)?, Budget::try_from(10)?);
/// let workspace = solver.workspace(&IterativeWorkspaceRequest::operator(&a)?)?;
/// assert!(workspace.last_attempt().is_none());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IterativeWorkspaceRequest {
    rows: usize,
    cols: usize,
    operator_scratch: usize,
    preconditioner_scratch: usize,
    right_hand_sides: usize,
    staged: bool,
    recycle: usize,
}

impl IterativeWorkspaceRequest {
    /// Creates a request for a solver that applies `a` only in the forward direction.
    ///
    /// # Errors
    ///
    /// Propagates the error of `a.apply_workspace_len()`.
    pub fn operator<S: NumericScalar>(a: &dyn LinearOperator<S>) -> Result<Self> {
        Ok(Self::sized(a.rows(), a.cols(), a.apply_workspace_len()?))
    }

    /// Creates a request for a solver that applies both `a` and `Aᴴ`.
    ///
    /// Use it for [`BicgSolver`](crate::BicgSolver) and [`QmrSolver`](crate::QmrSolver).
    /// The operator scratch is the larger of the forward and adjoint lengths.
    ///
    /// # Errors
    ///
    /// Propagates the errors of the operator's two scratch-length queries.
    pub fn adjoint<S: NumericScalar>(a: &dyn AdjointOperator<S>) -> Result<Self> {
        Ok(Self::sized(
            a.rows(),
            a.cols(),
            a.apply_workspace_len()?
                .max(a.apply_adjoint_workspace_len()?),
        ))
    }

    /// Creates a request for a stationary iteration on the matrix `a`.
    ///
    /// # Errors
    ///
    /// Propagates the error of the matrix's scratch-length query.
    pub fn matrix<S: NumericScalar>(a: crate::StationaryMatrix<'_, S>) -> Result<Self> {
        let operator = a.operator();
        Ok(Self::sized(
            operator.rows(),
            operator.cols(),
            operator.apply_workspace_len()?,
        ))
    }

    /// Adds the scratch length of the preconditioner `p` to the request.
    ///
    /// An adjoint preconditioner contributes the larger of its forward and
    /// adjoint scratch lengths. Only the length is stored.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `p` does not act on vectors of length `self.cols()`.
    /// - Propagates the error of the preconditioner's scratch-length query.
    pub fn with_preconditioner<S: NumericScalar>(
        mut self,
        p: Preconditioning<'_, S>,
    ) -> Result<Self> {
        p.validate_dimension(self.cols)?;
        self.preconditioner_scratch = p.workspace_len()?;
        Ok(self)
    }

    /// Sets the number of right-hand sides to `count`.
    ///
    /// Block solvers iterate on all `count` columns together. Single-vector
    /// solvers reserve a buffer for `count` columns, which
    /// [`IterativeLinearSolver::solve_block_into`](crate::IterativeLinearSolver::solve_block_into)
    /// solves one at a time.
    pub const fn with_right_hand_sides(mut self, count: usize) -> Self {
        self.right_hand_sides = count;
        self.staged = true;
        self
    }

    /// Adds room for the augmentation vectors stored in `space`.
    ///
    /// [`GcrotSolver`](crate::GcrotSolver), [`LgmresSolver`](crate::LgmresSolver)
    /// and their block forms use it. Other solvers ignore it. Only the capacity
    /// of `space` is stored.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if the dimension of `space` differs from the
    ///   operator's rows or columns.
    pub fn with_recycle_space<S: NumericScalar>(
        mut self,
        space: &super::KrylovRecycleSpace<S>,
    ) -> Result<Self> {
        if self.rows != space.dimension || self.cols != space.dimension {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(space.dimension, space.dimension),
                scioxide_core::Dimensions::two(self.rows, self.cols),
            ));
        }
        self.recycle = space.capacity();
        Ok(self)
    }

    const fn sized(rows: usize, cols: usize, operator_scratch: usize) -> Self {
        Self {
            rows,
            cols,
            operator_scratch,
            preconditioner_scratch: 0,
            right_hand_sides: 1,
            staged: false,
            recycle: 0,
        }
    }

    /// Returns the number of operator rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of operator columns, the length of the solution.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the number of right-hand sides; one unless set.
    pub const fn right_hand_sides(&self) -> usize {
        self.right_hand_sides
    }

    pub(super) const fn operator_scratch(&self) -> usize {
        self.operator_scratch
    }

    pub(super) const fn preconditioner_scratch(&self) -> usize {
        self.preconditioner_scratch
    }

    pub(super) const fn recycle_capacity(&self) -> usize {
        self.recycle
    }
}

/// Recurrence storage selected by a solver family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StorageKind {
    Empty,
    Cg,
    Bicgstab,
    Arnoldi,
    Krylov,
    Bicg,
    Cgs,
    Qmr,
    Tfqmr,
    Richardson,
    Stationary,
}

/// Exact geometry of one family's recurrence storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Shape {
    pub(super) kind: StorageKind,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) operator_scratch: usize,
    pub(super) preconditioner_scratch: usize,
    /// Columns iterated together; staging holds `rows × columns` and `cols × columns`.
    pub(super) columns: usize,
    /// Arnoldi steps per restart cycle.
    pub(super) cycle: usize,
    /// Recycle-space capacity.
    pub(super) retained: usize,
}

/// Storage a solver needs for one [`IterativeWorkspaceRequest`], with its byte limit.
///
/// Recurrence vectors are sized for the exact operator shape. A workspace may
/// hold more residual-history and block-staging capacity than requested. The
/// byte limit comes from the solver's [`LinalgExecutionOptions`](crate::LinalgExecutionOptions)
/// and bounds all storage the workspace holds, including capacity left from
/// earlier reservations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IterativeRequirements<S: NumericScalar> {
    shape: Shape,
    history: usize,
    block: usize,
    storage_bytes: usize,
    bytes: usize,
    byte_limit: Option<usize>,
    scalar: PhantomData<fn() -> S>,
}

impl<S: NumericScalar> IterativeRequirements<S> {
    /// Requirements of solvers without recurrence vectors: only block staging.
    pub(crate) fn staging_only(request: &IterativeWorkspaceRequest) -> Result<Self> {
        Self::new(
            Shape {
                kind: StorageKind::Empty,
                rows: 0,
                cols: 0,
                operator_scratch: 0,
                preconditioner_scratch: 0,
                columns: 1,
                cycle: 0,
                retained: 0,
            },
            None,
            Some(request),
            None,
        )
    }

    /// Requirements of one family. `limit` sizes residual history when
    /// enabled; `staging` reserves `solve_block_into` columns.
    /// # Errors
    /// Returns `SizeOverflow`, or `ResourceLimit` when the storage exceeds
    /// `byte_limit`.
    pub(super) fn new(
        shape: Shape,
        history: Option<Budget>,
        staging: Option<&IterativeWorkspaceRequest>,
        byte_limit: Option<usize>,
    ) -> Result<Self> {
        let history = history.map_or(Ok(0), |limit| {
            checked_add(
                limit.get(),
                1,
                scioxide_core::quantity!("iterative_residual_history"),
            )
        })?;
        let block = match staging {
            Some(request) if request.staged => checked_mul(
                request.cols,
                request.right_hand_sides,
                scioxide_core::quantity!("iterative_block_staging"),
            )?,
            _ => 0,
        };
        let storage_bytes = storage_bytes::<S>(&shape, history)?;
        let bytes = total([storage_bytes, scalar_bytes::<S>(block)?])?;
        crate::workspace::admit_bytes(bytes, byte_limit, WORKSPACE_BYTES)?;
        Ok(Self {
            shape,
            history,
            block,
            storage_bytes,
            bytes,
            byte_limit,
            scalar: PhantomData,
        })
    }

    /// Returns the number of residual-history samples; zero when history is disabled.
    pub const fn history_capacity(&self) -> usize {
        self.history
    }

    /// Returns the number of scalars reserved for the columns of `solve_block_into`.
    pub const fn block_staging(&self) -> usize {
        self.block
    }

    /// Returns the bytes of numerical data in a fresh workspace.
    ///
    /// `Vec` headers and allocator bookkeeping are not counted.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }

    /// Returns the limit on all workspace bytes, from the solver's execution options.
    pub const fn byte_limit(&self) -> Option<usize> {
        self.byte_limit
    }
}

pub(super) const WORKSPACE_BYTES: scioxide_core::Quantity =
    scioxide_core::quantity!("iterative_workspace_bytes");

/// Bytes of `count` working scalars.
pub(super) fn scalar_bytes<S: NumericScalar>(count: usize) -> Result<usize> {
    bytes_of::<S>(count)
}

/// Bytes of `count` values of `T`.
pub(super) fn bytes_of<T>(count: usize) -> Result<usize> {
    checked_mul(count, size_of::<T>(), WORKSPACE_BYTES)
}

/// Checked element count of `count` buffers of `len` values.
pub(super) fn times(count: usize, len: usize) -> Result<usize> {
    checked_mul(count, len, WORKSPACE_BYTES)
}

/// Checked sum of byte counts.
pub(super) fn total(parts: impl IntoIterator<Item = usize>) -> Result<usize> {
    parts
        .into_iter()
        .try_fold(0, |sum, part| checked_add(sum, part, WORKSPACE_BYTES))
}

/// Payload of `scalars` working scalars plus `reals` real values.
pub(super) fn payload<S: NumericScalar>(scalars: usize, reals: usize) -> Result<usize> {
    total([scalar_bytes::<S>(scalars)?, bytes_of::<S::Real>(reals)?])
}

/// Exact bytes of one family's recurrence storage and the staged
/// right-hand side and solution, as built by [`Storage::build`].
fn storage_bytes<S: NumericScalar>(shape: &Shape, history: usize) -> Result<usize> {
    let staging = total([
        checked_mul(shape.rows, shape.columns, WORKSPACE_BYTES)?,
        checked_mul(shape.cols, shape.columns, WORKSPACE_BYTES)?,
    ])?;
    let recurrence = match shape.kind {
        StorageKind::Empty => 0,
        StorageKind::Cg => CgStorage::<S>::bytes(shape, history)?,
        StorageKind::Bicgstab => BicgstabStorage::<S>::bytes(shape, history)?,
        StorageKind::Arnoldi => arnoldi::Storage::<S>::bytes(shape, history)?,
        StorageKind::Krylov => krylov::Storage::<S>::bytes(shape, history)?,
        StorageKind::Bicg => biorthogonal::Storage::<S, Bicg<S>>::bytes(shape, history)?,
        StorageKind::Cgs => biorthogonal::Storage::<S, Cgs<S>>::bytes(shape, history)?,
        StorageKind::Qmr => biorthogonal::Storage::<S, Qmr<S>>::bytes(shape, history)?,
        StorageKind::Tfqmr => biorthogonal::Storage::<S, Tfqmr<S>>::bytes(shape, history)?,
        StorageKind::Richardson => {
            biorthogonal::Storage::<S, Richardson<S>>::bytes(shape, history)?
        }
        StorageKind::Stationary => StationaryStorage::<S>::bytes(shape, history)?,
    };
    total([scalar_bytes::<S>(staging)?, recurrence])
}

/// Diagnostics of a solve that succeeded or stopped with a usable iterate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterativeProgress<R: RealScalar> {
    termination: LinearSolveTermination,
    iterations: usize,
    residual: Option<R>,
    restarts: usize,
    trust: PartialResultTrust,
}

impl<R: RealScalar> IterativeProgress<R> {
    pub(super) const fn new(
        termination: LinearSolveTermination,
        iterations: usize,
        residual: Option<R>,
        restarts: usize,
        trust: PartialResultTrust,
    ) -> Self {
        Self {
            termination,
            iterations,
            residual,
            restarts,
            trust,
        }
    }

    /// Outcome and accepted progress of a returned report.
    pub(super) fn of(report: &IterativeSolveReport<R>) -> (AttemptOutcome, Option<Self>) {
        (
            if report.is_success() {
                AttemptOutcome::Succeeded
            } else {
                AttemptOutcome::Stopped
            },
            Some(Self {
                termination: report.termination(),
                iterations: report.iterations(),
                residual: report.diagnostic_residual_norm(),
                restarts: report.krylov().map_or(0, |krylov| krylov.restarts),
                trust: report.trust(),
            }),
        )
    }

    /// Returns the termination status of the attempt.
    pub const fn termination(&self) -> LinearSolveTermination {
        self.termination
    }

    /// Returns the number of iterations the attempt counted toward its iteration limit.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the true residual norm `‖b − A x‖₂` of the returned iterate.
    ///
    /// For block solves it is the largest column residual. It is `None` when
    /// the solve stopped before the first residual was computed.
    pub const fn residual(&self) -> Option<R> {
        self.residual
    }

    /// Returns the number of recurrence or Arnoldi restarts.
    ///
    /// Periodic residual verification does not count as a restart.
    pub const fn restarts(&self) -> usize {
        self.restarts
    }

    /// Returns how far the returned iterate was checked.
    pub const fn trust(&self) -> PartialResultTrust {
        self.trust
    }
}

/// Per-column residuals of the iterate returned by a GMRES-family solve.
#[derive(Debug, Clone, Copy)]
pub struct KrylovColumns<'a, R: RealScalar> {
    pub(super) residuals: &'a [R],
    pub(super) references: &'a [R],
    pub(super) converged: &'a [bool],
}

impl<'a, R: RealScalar> KrylovColumns<'a, R> {
    /// Returns the true residual norm `‖bⱼ − A xⱼ‖₂` of each column `j`.
    pub const fn residuals(&self) -> &'a [R] {
        self.residuals
    }

    /// Returns the reference `‖bⱼ‖₂` of each column, or one for a zero column.
    pub const fn references(&self) -> &'a [R] {
        self.references
    }

    /// Returns whether each column meets the residual tolerance.
    pub const fn converged(&self) -> &'a [bool] {
        self.converged
    }
}

/// Returned iterate, residual history and diagnostics of the most recent solve.
///
/// Obtain it from [`IterativeWorkspace::last_accepted`].
#[derive(Debug, Clone, Copy)]
pub struct IterativeAccepted<'a, S: NumericScalar> {
    solution: &'a [S],
    history: Option<&'a [S::Real]>,
    columns: Option<KrylovColumns<'a, S::Real>>,
    progress: IterativeProgress<S::Real>,
}

impl<'a, S: NumericScalar> IterativeAccepted<'a, S> {
    /// Returns the iterate written to the output; block solves store it column by column.
    pub const fn solution(&self) -> &'a [S] {
        self.solution
    }

    /// Returns the residual history, when the solver records it.
    ///
    /// The first sample is the true initial residual, followed by one sample
    /// per completed iteration. When the report has a true residual, the last
    /// sample is the residual of the returned iterate.
    pub const fn history(&self) -> Option<&'a [S::Real]> {
        self.history
    }

    /// Returns per-column residuals of GMRES-family solves; `None` for other solvers.
    pub const fn columns(&self) -> Option<KrylovColumns<'a, S::Real>> {
        self.columns
    }

    /// Returns the scalar diagnostics of the attempt.
    pub const fn progress(&self) -> IterativeProgress<S::Real> {
        self.progress
    }
}

/// Recurrence storage of the most recently reserved family.
#[derive(Debug)]
// Families hold differently sized recurrence vectors; boxing would add an
// allocation per reservation without changing retained capacity.
#[cfg_attr(
    target_pointer_width = "64",
    expect(
        clippy::large_enum_variant,
        reason = "retained solver variants avoid extra allocations when a workspace changes family"
    )
)]
pub(super) enum Storage<S: NumericScalar> {
    Empty,
    Cg(CgStorage<S>),
    Bicgstab(BicgstabStorage<S>),
    Arnoldi(arnoldi::Storage<S>),
    Krylov(krylov::Storage<S>),
    Bicg(biorthogonal::Storage<S, Bicg<S>>),
    Cgs(biorthogonal::Storage<S, Cgs<S>>),
    Qmr(biorthogonal::Storage<S, Qmr<S>>),
    Tfqmr(biorthogonal::Storage<S, Tfqmr<S>>),
    Richardson(biorthogonal::Storage<S, Richardson<S>>),
    Stationary(StationaryStorage<S>),
}

impl<S: NumericScalar> Storage<S> {
    fn build(shape: &Shape, history: usize) -> Result<Self> {
        Ok(match shape.kind {
            StorageKind::Empty => Self::Empty,
            StorageKind::Cg => Self::Cg(CgStorage::new(shape, history)?),
            StorageKind::Bicgstab => Self::Bicgstab(BicgstabStorage::new(shape, history)?),
            StorageKind::Arnoldi => Self::Arnoldi(arnoldi::Storage::new(shape, history)?),
            StorageKind::Krylov => Self::Krylov(krylov::Storage::new(shape, history)?),
            StorageKind::Bicg => Self::Bicg(biorthogonal::Storage::new(shape, history)?),
            StorageKind::Cgs => Self::Cgs(biorthogonal::Storage::new(shape, history)?),
            StorageKind::Qmr => Self::Qmr(biorthogonal::Storage::new(shape, history)?),
            StorageKind::Tfqmr => Self::Tfqmr(biorthogonal::Storage::new(shape, history)?),
            StorageKind::Richardson => {
                Self::Richardson(biorthogonal::Storage::new(shape, history)?)
            }
            StorageKind::Stationary => Self::Stationary(StationaryStorage::new(shape, history)?),
        })
    }

    fn history(&self) -> Option<&[S::Real]> {
        match self {
            Self::Empty => None,
            Self::Cg(storage) => storage.history(),
            Self::Bicgstab(storage) => storage.history(),
            Self::Arnoldi(storage) => storage.recorded_history(),
            Self::Krylov(storage) => storage.history(),
            Self::Bicg(storage) => storage.history(),
            Self::Cgs(storage) => storage.history(),
            Self::Qmr(storage) => storage.history(),
            Self::Tfqmr(storage) => storage.history(),
            Self::Richardson(storage) => storage.history(),
            Self::Stationary(storage) => storage.history(),
        }
    }

    fn columns(&self) -> Option<KrylovColumns<'_, S::Real>> {
        match self {
            Self::Arnoldi(storage) => Some(storage.columns()),
            _ => None,
        }
    }
}

/// Reports whose initial state can be returned before any residual is measured.
pub(super) trait StoppedReport: Copy {
    fn unmeasured_stop(termination: LinearSolveTermination) -> Self;
}
impl<R: RealScalar> StoppedReport for IterativeSolveReport<R> {
    fn unmeasured_stop(termination: LinearSolveTermination) -> Self {
        Self::new(termination, 0, None, PartialResultTrust::DiagnosticOnly)
    }
}

/// Typed access to one family's storage after reservation.
pub(super) trait Family<S: NumericScalar>: Sized {
    fn select(storage: &mut Storage<S>) -> Option<&mut Self>;
}

macro_rules! family {
    ($variant:ident, $type:ty) => {
        impl<S: NumericScalar> Family<S> for $type {
            fn select(storage: &mut Storage<S>) -> Option<&mut Self> {
                match storage {
                    Storage::$variant(storage) => Some(storage),
                    _ => None,
                }
            }
        }
    };
}
family!(Cg, CgStorage<S>);
family!(Bicgstab, BicgstabStorage<S>);
family!(Arnoldi, arnoldi::Storage<S>);
family!(Krylov, krylov::Storage<S>);
family!(Bicg, biorthogonal::Storage<S, Bicg<S>>);
family!(Cgs, biorthogonal::Storage<S, Cgs<S>>);
family!(Qmr, biorthogonal::Storage<S, Qmr<S>>);
family!(Tfqmr, biorthogonal::Storage<S, Tfqmr<S>>);
family!(Richardson, biorthogonal::Storage<S, Richardson<S>>);
family!(Stationary, StationaryStorage<S>);

/// Reusable scratch storage and attempt record for the iterative solvers.
///
/// One workspace serves every iterative solver of this crate. It holds the
/// recurrence vectors of one solver family for one operator shape, plus
/// residual history and block staging. For another family or shape,
/// [`Self::check`] reports `NeedsGrowth`, and [`Self::try_reserve`] or the next
/// solve replaces the recurrence vectors. History and staging capacity is
/// kept. The byte limit of the solver's execution options bounds everything
/// the workspace holds and is checked before any allocation. See
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// [`Self::last_attempt`] records the outcome of every solve call. A call
/// rejected during input validation has no counts, and a failed call has
/// counts but no returned iterate. [`Self::last_accepted`] returns the iterate
/// and residual history of a solve that succeeded or stopped with a usable
/// iterate.
#[derive(Debug)]
pub struct IterativeWorkspace<S: NumericScalar> {
    slot: SolveSlot<IterativeProgress<S::Real>>,
    reserved: Option<Reserved>,
    storage: Storage<S>,
    rhs: Vector<S>,
    solution: Vector<S>,
    block: Vec<S>,
}

/// Shape, history capacity and bytes of the reserved recurrence storage.
#[derive(Debug, Clone, Copy)]
struct Reserved {
    shape: Shape,
    capacity_shape: Shape,
    capacity_history: usize,
    history: usize,
    bytes: usize,
}

impl<S: NumericScalar> Default for IterativeWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed metadata and storage requirements of one solve through the shared driver.
pub(super) struct Job<S: NumericScalar> {
    pub(super) algorithm: scioxide_core::AlgorithmId,
    pub(super) limit: Budget,
    pub(super) requirements: IterativeRequirements<S>,
}

impl<S: NumericScalar> IterativeWorkspace<S> {
    /// Returns the bytes the workspace holds in recurrence and block-staging storage.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    pub fn retained_bytes(&self) -> Result<usize> {
        total([
            self.reserved.map_or(0, |r| r.bytes),
            scalar_bytes::<S>(self.block.capacity())?,
        ])
    }

    /// Creates an empty workspace without allocating.
    pub const fn new() -> Self {
        Self {
            slot: SolveSlot::new(),
            reserved: None,
            storage: Storage::Empty,
            rhs: Vector::empty(),
            solution: Vector::empty(),
            block: Vec::new(),
        }
    }

    /// Checks whether the reserved storage covers `requirements`, without allocating.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit` if the requirements, or the storage the workspace would
    ///   hold after reserving them, exceed the byte limit of `requirements`.
    /// - `SizeOverflow` if a byte count overflows `usize`.
    pub fn check(&self, requirements: &IterativeRequirements<S>) -> Result<Capacity> {
        self.admit(requirements)?;
        Ok(
            if self.covers(requirements) && self.block.capacity() >= requirements.block {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    /// Reserves storage for `requirements`.
    ///
    /// Storage for another solver family is built completely before it replaces
    /// the current storage. GMRES-family storage without a recycle space keeps
    /// its Krylov basis capacity when the operator dimension changes. If that
    /// growth fails, the workspace stays usable and the next reservation or
    /// solve allocates again.
    ///
    /// # Errors
    ///
    /// - `ResourceLimit`, before any allocation, if the storage the workspace
    ///   would hold exceeds the byte limit of `requirements`.
    /// - `SizeOverflow` if a size or byte count overflows `usize`.
    /// - `AllocationFailure` if memory cannot be reserved.
    pub fn try_reserve(&mut self, requirements: &IterativeRequirements<S>) -> Result<()> {
        self.admit(requirements)?;
        if !self.covers(requirements) {
            let shape = requirements.shape;
            let mut capacity_shape = shape;
            let mut retained_storage_bytes = requirements.storage_bytes;
            let mut capacity_history = requirements.history;
            if let (Some(old), Storage::Arnoldi(storage)) = (self.reserved, &mut self.storage)
                && shape.kind == StorageKind::Arnoldi
                && shape.retained == 0
                && old.shape.retained == 0
                && shape.columns == old.shape.columns
            {
                // A dominating shape admits both retained bases and new dimensions.
                let mut retained = shape;
                retained.rows = retained.rows.max(old.capacity_shape.rows);
                retained.cols = retained.cols.max(old.capacity_shape.cols);
                retained.cycle = retained.cycle.max(old.capacity_shape.cycle);
                retained.operator_scratch = retained
                    .operator_scratch
                    .max(old.capacity_shape.operator_scratch);
                retained.preconditioner_scratch = retained
                    .preconditioner_scratch
                    .max(old.capacity_shape.preconditioner_scratch);
                capacity_history = requirements.history.max(old.capacity_history);
                let retained_bytes = storage_bytes::<S>(&retained, capacity_history)?;
                crate::workspace::admit_bytes(
                    total([
                        retained_bytes,
                        scalar_bytes::<S>(self.block.capacity().max(requirements.block))?,
                    ])?,
                    requirements.byte_limit,
                    WORKSPACE_BYTES,
                )?;
                capacity_shape = retained;
                retained_storage_bytes = retained_bytes;
                self.reserved = None;
                self.slot.reject();
                storage.resize_unrecycled(&shape, requirements.history)?;
            } else {
                let storage = Storage::build(&shape, requirements.history)?;
                let rhs = Vector::zeros(times(shape.rows, shape.columns)?)?;
                let solution = Vector::zeros(times(shape.cols, shape.columns)?)?;
                self.storage = storage;
                self.rhs = rhs;
                self.solution = solution;
            }
            self.rhs.try_resize(
                checked_mul(
                    shape.rows,
                    shape.columns,
                    scioxide_core::quantity!("iterative_rhs_staging"),
                )?,
                S::ZERO,
            )?;
            self.solution.try_resize(
                checked_mul(
                    shape.cols,
                    shape.columns,
                    scioxide_core::quantity!("iterative_solution_staging"),
                )?,
                S::ZERO,
            )?;
            self.reserved = Some(Reserved {
                shape,
                capacity_shape,
                capacity_history,
                history: requirements.history,
                bytes: retained_storage_bytes,
            });
        }
        let additional = requirements.block.saturating_sub(self.block.len());
        try_reserve(
            &mut self.block,
            additional,
            scioxide_core::quantity!("iterative_block_staging"),
        )
    }

    /// Checks the requirements and the bytes held after reserving them
    /// (kept or replacement recurrence storage plus grown block staging)
    /// against the byte limit.
    fn admit(&self, requirements: &IterativeRequirements<S>) -> Result<()> {
        let limit = requirements.byte_limit;
        crate::workspace::admit_bytes(requirements.bytes, limit, WORKSPACE_BYTES)?;
        let storage = match self.reserved {
            Some(reserved) if self.covers(requirements) => reserved.bytes,
            _ => requirements.storage_bytes,
        };
        let block = scalar_bytes::<S>(self.block.capacity().max(requirements.block))?;
        crate::workspace::admit_bytes(total([storage, block])?, limit, WORKSPACE_BYTES)
    }

    /// Returns the record of the most recent solve call, or `None` before the first call.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<IterativeProgress<S::Real>>> {
        self.slot.last_attempt()
    }

    /// Returns the iterate and diagnostics of the most recent solve.
    ///
    /// It is `None` when that solve failed with an error or was rejected
    /// during input validation.
    pub fn last_accepted(&self) -> Option<IterativeAccepted<'_, S>> {
        let progress = *self.slot.last_attempt()?.progress()?;
        Some(IterativeAccepted {
            solution: self.solution.as_slice(),
            history: progress.residual.and_then(|_| self.storage.history()),
            columns: progress.residual.and_then(|_| self.storage.columns()),
            progress,
        })
    }

    fn covers(&self, requirements: &IterativeRequirements<S>) -> bool {
        self.reserved.is_some_and(|reserved| {
            reserved.shape == requirements.shape
                && (reserved.history == 0) == (requirements.history == 0)
                && reserved.history >= requirements.history
        })
    }

    /// Take the block staging buffer, leaving an empty vector in place.
    pub(crate) fn take_block(&mut self) -> Vec<S> {
        core::mem::take(&mut self.block)
    }

    /// Return block staging taken by [`Self::take_block`].
    pub(crate) fn restore_block(&mut self, block: Vec<S>) {
        self.block = block;
    }

    /// Record a call rejected before its epoch began.
    pub(crate) fn reject(&mut self) {
        self.slot.reject();
    }

    /// Allocate output from `x0` (zero when absent) and a fresh workspace,
    /// solve, and own the recorded residual history.
    pub(crate) fn solve_owned(
        cols: usize,
        x0: Option<VectorView<'_, S>>,
        algorithm: scioxide_core::AlgorithmId,
        solve: impl FnOnce(VectorViewMut<'_, S>, &mut Self) -> Result<IterativeSolveReport<S::Real>>,
    ) -> Result<crate::IterativeSolveResult<S>> {
        let mut x = match x0 {
            Some(initial) => initial.try_to_vector()?,
            None => Vector::zeros(cols)?,
        };
        let mut workspace = Self::new();
        let report = solve(x.view_mut(), &mut workspace)?;
        let history = workspace
            .last_accepted()
            .and_then(|accepted| accepted.history())
            .map(|samples| {
                scioxide_core::backend::try_copy_slice(
                    samples,
                    scioxide_core::quantity!("iterative_owned_history"),
                )
            })
            .transpose()?;
        Ok(crate::IterativeSolveResult::new(
            algorithm, x, report, history,
        ))
    }

    /// Run one family's solve: stage `b` and `x`, iterate in a fresh epoch,
    /// record the attempt, then write `x` on `Ok`.
    ///
    /// Callers validate inputs after [`Self::reject`] and before this call.
    pub(super) fn run<'capability, F: Family<S>, T: StoppedReport>(
        &mut self,
        context: LinalgExecutionContext<'_, 'capability>,
        job: &Job<S>,
        b: VectorView<'_, S>,
        x: &mut VectorViewMut<'_, S>,
        kernel: impl FnOnce(
            &mut F,
            &Vector<S>,
            &mut Vector<S>,
            crate::KernelContext<'capability>,
            &mut scioxide_core::OperatorContext<'_, '_, 'capability, S>,
        ) -> Result<T>,
        classify: impl FnOnce(&T) -> (AttemptOutcome, Option<IterativeProgress<S::Real>>),
    ) -> Result<T> {
        let requirements = &job.requirements;
        let shape = requirements.shape;
        let (rows, cols) = (
            checked_mul(
                shape.rows,
                shape.columns,
                scioxide_core::quantity!("iterative_rhs_staging"),
            )?,
            checked_mul(
                shape.cols,
                shape.columns,
                scioxide_core::quantity!("iterative_solution_staging"),
            )?,
        );
        if b.len() != rows || x.len() != cols {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::two(rows, cols),
                scioxide_core::Dimensions::two(b.len(), x.len()),
            ));
        }
        self.try_reserve(requirements)?;
        stage(&mut self.rhs, b.iter())?;
        stage(&mut self.solution, x.iter().copied())?;
        let kernels = context.kernel();
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            job.algorithm,
            ((shape.rows as u128) << 64) | shape.cols as u128,
            0,
        );
        let allowance = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Iteration, context.cap_iterations(job.limit));
        let Self {
            slot,
            storage,
            rhs,
            solution,
            ..
        } = self;
        let storage = F::select(storage).ok_or_else(|| {
            crate::checked::inconsistent(
                scioxide_core::argument!("workspace"),
                "reserved storage belongs to another solver family",
            )
        })?;
        let result = slot.run(
            context.into_scope(),
            identity,
            allowance,
            |solve| {
                if let Some(reason) = solve.stop_reason()? {
                    return Ok(T::unmeasured_stop(
                        LinearSolveTermination::ExecutionStopped(reason),
                    ));
                }
                match kernel(
                    storage,
                    rhs,
                    solution,
                    kernels,
                    &mut scioxide_core::OperatorContext::with_kernels(solve, &kernels),
                ) {
                    Ok(report) => Ok(report),
                    // Family drivers restore their verified snapshot before returning a
                    // stopped report. A stop escaping here preceded a verified state.
                    Err(error) => super::support::execution_stop(error).map(T::unmeasured_stop),
                }
            },
            |result| match result {
                Ok(report) => classify(report),
                Err(_) => (scioxide_core::AttemptOutcome::Failed, None),
            },
        )?;
        // Publication may stride the caller view; the private copy kernel keeps bits.
        crate::kernel::level2::copy_into(
            crate::VectorView::from(&*solution).into(),
            x.reborrow().into(),
            crate::CopyPolicy::Preserve,
        )?;
        Ok(result)
    }
}

fn stage<S: NumericScalar>(target: &mut Vector<S>, values: impl Iterator<Item = S>) -> Result<()> {
    for (slot, value) in target.as_mut_slice().iter_mut().zip(values) {
        crate::checked::validate_input(value, "solve input")?;
        *slot = value;
    }
    Ok(())
}
