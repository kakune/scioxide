//! Scratch storage for dense operations, sized from typed requests.

pub(crate) mod accounting;
pub(crate) mod storage;

use crate::{Accuracy, LinalgExecutionOptions};
use alloc::vec::Vec;
use core::mem::size_of;
use core::num::NonZeroUsize;
use scioxide_core::{
    Capacity, Complex, NumericScalar, RangeAccumulator, ScaledValue, SciOxideError,
};

/// Split reserved scratch, reporting a shortfall as `DimensionMismatch` instead of panicking.
pub(crate) fn split_mut<T>(
    values: &mut [T],
    len: usize,
) -> Result<(&mut [T], &mut [T]), SciOxideError> {
    let available = values.len();
    values
        .split_at_mut_checked(len)
        .ok_or_else(|| SciOxideError::dimension_mismatch([len], [available]))
}

/// Take a prefix from a cursor over reserved scratch.
pub(crate) fn take_mut<'a, T>(
    rest: &mut &'a mut [T],
    len: usize,
) -> Result<&'a mut [T], SciOxideError> {
    let (head, tail) = split_mut(core::mem::take(rest), len)?;
    *rest = tail;
    Ok(head)
}

/// Take a row-major `n × n` matrix from a cursor over reserved scratch.
pub(crate) fn take_square<'a, S: NumericScalar>(
    rest: &mut &'a mut [S],
    n: usize,
) -> Result<storage::ScratchMatrix<'a, S>, SciOxideError> {
    let len = scioxide_core::backend::checked_mul(
        n,
        n,
        scioxide_core::quantity!("square_scratch_elements"),
    )?;
    storage::ScratchMatrix::new(take_mut(rest, len)?, n, n)
}

/// Extended-range complex value stored by the extended arena.
pub(crate) type Extended<S> = ScaledValue<Complex<<S as NumericScalar>::Real>>;

/// Description of one operation's scratch needs, accepted by [`LinalgWorkspace`].
///
/// Pass a request to [`LinalgWorkspace::workspace_requirements`] or
/// [`LinalgWorkspace::workspace`]. It is implemented by
/// [`DenseWorkspaceRequest`], [`OperatorWorkspaceRequest`](crate::OperatorWorkspaceRequest),
/// [`SubspaceBasisWorkspaceRequest`](crate::SubspaceBasisWorkspaceRequest),
/// [`SubspaceAnglesWorkspaceRequest`](crate::SubspaceAnglesWorkspaceRequest)
/// and [`ProcrustesWorkspaceRequest`](crate::ProcrustesWorkspaceRequest).
/// The other `*WorkspaceRequest` types, of factorizations, eigensolvers,
/// matrix functions and plans, are passed to the `workspace_requirements` and
/// `workspace` functions of their own type, such as
/// [`LuDecomposition::workspace_requirements`](crate::LuDecomposition::workspace_requirements).
/// The trait is sealed: it cannot be implemented outside this crate.
pub trait LinalgWorkspaceRequest<S: NumericScalar>: request::Sealed<S> {}
impl<S: NumericScalar, T: Request<S>> LinalgWorkspaceRequest<S> for T {}

mod request {
    use super::*;
    pub trait Sealed<S: NumericScalar> {
        fn requirements(&self) -> Result<LinalgRequirements, SciOxideError>;
    }
}
pub(crate) use request::Sealed as Request;

/// Scratch request for a dense kernel: a vector operation or a matrix product.
///
/// Each constructor names an operation and its dimensions. The
/// [`LinalgExecutionOptions`] supply the accuracy level and byte limit, which
/// change the requirements. By default a request covers as many workers as the
/// execution policy allows and no BLAS provider. Adjust this with
/// [`for_context`](Self::for_context), [`with_worker_bound`](Self::with_worker_bound)
/// or [`with_provider_packing`](Self::with_provider_packing). Product scratch is
/// large enough for every native kernel and for either storage order of the
/// operands.
#[derive(Debug, Clone, Copy)]
pub struct DenseWorkspaceRequest {
    pub(crate) shape: DenseShape,
    pub(crate) options: LinalgExecutionOptions,
    pub(crate) workers: NonZeroUsize,
    pub(crate) packing: Packing,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DenseShape {
    Vector(usize),
    VectorPair(usize),
    InPlaceVector(usize),
    RankTwo {
        order: usize,
        shared: usize,
    },
    Product {
        rows: usize,
        shared: usize,
        cols: usize,
    },
    TriangularProduct {
        rows: usize,
        order: usize,
        cols: usize,
    },
    TriangularSolve {
        rows: usize,
        shared: usize,
        cols: usize,
    },
    Structured {
        rows: usize,
        shared: usize,
        cols: usize,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Packing {
    Native,
    Provider,
}

impl DenseWorkspaceRequest {
    /// Requests scratch for a vector operation of length `len`.
    ///
    /// Examples are `axpby` and matrix-vector products. The requirement is
    /// `len` scalars.
    pub const fn vector(len: usize, options: &LinalgExecutionOptions) -> Self {
        Self::new(DenseShape::Vector(len), options)
    }

    /// Requests scratch for an in-place update of `len` elements, such as a scaling.
    ///
    /// Native execution needs no scratch. With provider packing the requirement
    /// is `len` scalars.
    pub const fn in_place_vector(len: usize, options: &LinalgExecutionOptions) -> Self {
        Self::new(DenseShape::InPlaceVector(len), options)
    }
    /// Requests scratch for an operation on two vectors of length `len`.
    ///
    /// Examples are swaps and plane rotations. The requirement is `2 len`
    /// scalars.
    pub const fn vector_pair(len: usize, options: &LinalgExecutionOptions) -> Self {
        Self::new(DenseShape::VectorPair(len), options)
    }
    /// Requests scratch for the product of a `rows × shared` matrix and a `shared × cols` matrix.
    pub const fn product(
        rows: usize,
        shared: usize,
        cols: usize,
        options: &LinalgExecutionOptions,
    ) -> Self {
        Self::new(DenseShape::Product { rows, shared, cols }, options)
    }

    /// Requests scratch for a product of structured operands, including dense copies of both.
    pub const fn structured(
        rows: usize,
        shared: usize,
        cols: usize,
        options: &LinalgExecutionOptions,
    ) -> Self {
        Self::new(DenseShape::Structured { rows, shared, cols }, options)
    }

    /// Requests scratch for the product of a triangular `order × order` matrix and a dense block.
    ///
    /// The output block is `rows × cols`. Native execution needs two buffers of
    /// `rows cols` scalars. With provider packing the triangle is also copied,
    /// which adds `order²` scalars.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{LinalgExecutionOptions, DenseWorkspaceRequest, LinalgWorkspace};
    /// let request = DenseWorkspaceRequest::triangular_product(2, 2, 3, &LinalgExecutionOptions::new());
    /// let requirements = LinalgWorkspace::<f64>::workspace_requirements(&request)?;
    /// assert_eq!(requirements.scalar_len()?, 12);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn triangular_product(
        rows: usize,
        order: usize,
        cols: usize,
        options: &LinalgExecutionOptions,
    ) -> Self {
        Self::new(DenseShape::TriangularProduct { rows, order, cols }, options)
    }

    /// Requests scratch for a triangular solve with an `order × order` matrix.
    ///
    /// The right-hand side and solution are `rows × cols`.
    pub const fn triangular_solve(
        rows: usize,
        order: usize,
        cols: usize,
        options: &LinalgExecutionOptions,
    ) -> Self {
        Self::new(
            DenseShape::TriangularSolve {
                rows,
                shared: order,
                cols,
            },
            options,
        )
    }
    /// Narrows the request to the workers and BLAS provider of `context`.
    ///
    /// The accuracy level still comes from the request's options.
    pub fn for_context(self, context: &crate::LinalgExecutionContext<'_, '_>) -> Self {
        self.for_kernels(context.kernel())
    }
    pub(crate) fn for_kernels(mut self, context: crate::KernelContext<'_>) -> Self {
        self.workers = context.execution().worker_budget();
        #[cfg(feature = "blas")]
        if context.blas().is_some() {
            self = self.with_provider_packing();
        }
        self
    }
    /// Requests scratch for a symmetric or Hermitian rank-2k update of an `order × order` matrix.
    ///
    /// `shared` is the number of columns of the two factors.
    pub const fn rank_two(order: usize, shared: usize, options: &LinalgExecutionOptions) -> Self {
        Self::new(DenseShape::RankTwo { order, shared }, options)
    }

    const fn new(shape: DenseShape, options: &LinalgExecutionOptions) -> Self {
        Self {
            shape,
            options: *options,
            workers: NonZeroUsize::MAX,
            packing: Packing::Native,
        }
    }

    /// Include the storage required for this accuracy level.
    pub(crate) const fn with_product_accuracy(mut self, accuracy: Accuracy) -> Self {
        self.options = self.options.with_accuracy(accuracy);
        self
    }

    /// Limits the request to at most `workers` workers.
    ///
    /// The kernel's own worker limit also applies.
    pub const fn with_worker_bound(mut self, workers: NonZeroUsize) -> Self {
        self.workers = workers;
        self
    }

    /// Adds scratch for copying whole operands for a BLAS provider.
    ///
    /// The native kernels can still run with this scratch.
    pub const fn with_provider_packing(mut self) -> Self {
        self.packing = Packing::Provider;
        self
    }
}

/// Buffer lengths that a [`LinalgWorkspace`] needs for an operation, and its byte limit.
///
/// Get them from [`LinalgWorkspace::workspace_requirements`]. Use
/// [`max`](Self::max) to size one workspace for several operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinalgRequirements {
    scalars: [usize; 8],
    reals: [usize; 2],
    pub(crate) indices: usize,
    pub(crate) accumulators: usize,
    pub(crate) extended: usize,
    pub(crate) scaled: usize,
    pub(crate) exponents: usize,
    pub(crate) byte_limit: Option<usize>,
    pub(crate) structured_dense_fallback: bool,
}

/// Named partitions of the scalar workspace. Numeric layout is private here.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ScalarRegion {
    PackedA,
    PackedB,
    Candidate,
    StructuredA,
    StructuredB,
    StructuredOutput,
    ExactTiles,
    Direct,
}

/// Exact packing scratch and family-specific real scratch.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RealRegion {
    Exact,
    Direct,
}

impl LinalgRequirements {
    pub(crate) fn new(byte_limit: Option<usize>) -> Self {
        Self {
            byte_limit,
            ..Self::default()
        }
    }
    pub(crate) fn scalar(&self, region: ScalarRegion) -> usize {
        self.scalars[region as usize]
    }
    pub(crate) fn scalar_mut(&mut self, region: ScalarRegion) -> &mut usize {
        &mut self.scalars[region as usize]
    }
    pub(crate) fn real(&self, region: RealRegion) -> usize {
        self.reals[region as usize]
    }
    pub(crate) fn real_mut(&mut self, region: RealRegion) -> &mut usize {
        &mut self.reals[region as usize]
    }
    pub(crate) fn with_scalar(mut self, region: ScalarRegion, len: usize) -> Self {
        *self.scalar_mut(region) = len;
        self
    }

    pub(crate) fn with_real(mut self, region: RealRegion, len: usize) -> Self {
        *self.real_mut(region) = len;
        self
    }
    pub(crate) fn with_scaled(mut self, len: usize) -> Self {
        self.scaled = len;
        self
    }
    pub(crate) fn with_indices(mut self, len: usize) -> Self {
        self.indices = len;
        self
    }
    /// Returns requirements that cover both `self` and `other`.
    ///
    /// Each buffer length is the larger of the two, and the byte limit is the
    /// smaller. The combined byte count is checked later, by
    /// [`LinalgWorkspace::check`] and [`LinalgWorkspace::try_reserve`].
    pub fn max(&self, other: &Self) -> Self {
        Self {
            scalars: core::array::from_fn(|i| self.scalars[i].max(other.scalars[i])),
            reals: core::array::from_fn(|i| self.reals[i].max(other.reals[i])),
            indices: self.indices.max(other.indices),
            accumulators: self.accumulators.max(other.accumulators),
            extended: self.extended.max(other.extended),
            scaled: self.scaled.max(other.scaled),
            exponents: self.exponents.max(other.exponents),
            structured_dense_fallback: self.structured_dense_fallback
                || other.structured_dense_fallback,
            byte_limit: match (self.byte_limit, other.byte_limit) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (limit, None) | (None, limit) => limit,
            },
        }
    }

    /// Returns the number of scalars required.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the total does not fit in `usize`.
    pub fn scalar_len(&self) -> Result<usize, SciOxideError> {
        arena_len(self.scalars)
    }
    /// Returns the bytes required for scalar type `S`, excluding fixed-size metadata.
    ///
    /// Use it to include this scratch in a larger memory budget.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count does not fit in `usize`.
    /// - `ResourceLimit` if it exceeds the byte limit.
    pub fn required_bytes<S: NumericScalar>(&self) -> Result<usize, SciOxideError> {
        let bytes = self.bytes::<S>()?;
        admit_bytes(
            bytes,
            self.byte_limit,
            scioxide_core::quantity!("linalg_workspace_bytes"),
        )?;
        Ok(bytes)
    }

    fn bytes<S: NumericScalar>(&self) -> Result<usize, SciOxideError> {
        [
            (arena_len(self.scalars)?, size_of::<S>()),
            (arena_len(self.reals)?, size_of::<S::Real>()),
            (self.indices, size_of::<usize>()),
            (self.accumulators, size_of::<RangeAccumulator<S::Real>>()),
            (self.extended, size_of::<Extended<S>>()),
            (self.scaled, size_of::<ScaledValue<S>>()),
            (self.exponents, size_of::<i64>()),
        ]
        .into_iter()
        .try_fold(0usize, |total, (count, width)| {
            count
                .checked_mul(width)
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(|| {
                    SciOxideError::size_overflow(scioxide_core::quantity!("linalg_workspace_bytes"))
                })
        })
    }
}

/// Reusable scratch storage for dense kernels, factorizations and dense plans.
///
/// A workspace holds several typed buffers: scalars, real values, indices,
/// accumulators for `Exact` products, and values with an extended exponent
/// range. Size it from a request. [`workspace_requirements`](Self::workspace_requirements)
/// returns the [`LinalgRequirements`] of one operation, [`check`](Self::check)
/// tells whether they fit, and [`try_reserve`](Self::try_reserve) grows the
/// buffers. Operations that take a workspace call `try_reserve` themselves, so
/// a workspace from [`new`](Self::new) grows on first use.
///
/// Reservation never shrinks a buffer. A byte limit applies to all owned
/// buffers, including capacity left from an earlier, larger operation or
/// another accuracy level. Create a new workspace to release memory.
/// [`from_scalar_slice`](Self::from_scalar_slice) uses a buffer you provide
/// instead; it never grows.
///
/// A workspace stores no inputs or results between calls. See
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// # Examples
///
/// ```
/// use scioxide_core::Capacity;
/// use scioxide_linalg::{DenseWorkspaceRequest, LinalgExecutionOptions, MatrixView, LinalgWorkspace};
/// let request = DenseWorkspaceRequest::product(4, 4, 4, &LinalgExecutionOptions::new());
/// let requirements = LinalgWorkspace::<f64>::workspace_requirements(&request)?;
/// let mut workspace = LinalgWorkspace::<f64>::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct LinalgWorkspace<'a, S: NumericScalar> {
    scalars: PartitionedArena<'a, S, 8>,
    scopes: ScopeStorage<'a>,
    scope_depth: usize,
    reals: PartitionedArena<'a, S::Real, 2>,
    indices: Arena<'a, usize>,
    accumulators: Arena<'a, RangeAccumulator<S::Real>>,
    extended: Arena<'a, Extended<S>>,
    scaled: Arena<'a, ScaledValue<S>>,
    exponents: Arena<'a, i64>,
    pub(crate) structured_dense_fallback: bool,
}

/// Internal fixed-arena lending for nested kernels and white-box tests.
#[derive(Debug)]
pub(crate) struct LinalgSlices<'a, S: NumericScalar> {
    /// Scalar arena, partitioned by the requirements' scalar regions.
    pub scalars: &'a mut [S],
    /// Real arena, partitioned by the requirements' real regions.
    pub reals: &'a mut [S::Real],
    /// Index arena.
    pub indices: &'a mut [usize],
    /// Exact product accumulators.
    pub accumulators: &'a mut [RangeAccumulator<S::Real>],
    /// Extended-range complex values.
    pub extended: &'a mut [ScaledValue<Complex<S::Real>>],
    /// Scalar-domain extended values for equation range recovery.
    pub scaled: &'a mut [ScaledValue<S>],
    /// Signed binary exponents; contents are scratch, not caller configuration.
    pub exponents: &'a mut [i64],
}

impl<S: NumericScalar> Default for LinalgSlices<'_, S> {
    fn default() -> Self {
        Self {
            scalars: &mut [],
            reals: &mut [],
            indices: &mut [],
            accumulators: &mut [],
            extended: &mut [],
            scaled: &mut [],
            exponents: &mut [],
        }
    }
}

/// One reusable solve state and the record of its most recent attempt.
///
/// Every workspace that starts root or child epochs embeds one slot. A new
/// call replaces only the attempt record; the state keeps its epoch lineage.
#[derive(Debug, Clone)]
pub(crate) struct SolveSlot<P> {
    state: scioxide_core::SolveState,
    attempt: Option<scioxide_core::AttemptRecord<P>>,
}
impl<P> SolveSlot<P> {
    pub(crate) const fn new() -> Self {
        Self {
            state: scioxide_core::SolveState::new(),
            attempt: None,
        }
    }

    /// Record of the most recent call, if any.
    pub(crate) const fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<P>> {
        self.attempt.as_ref()
    }

    /// Mark a fresh call as rejected until its epoch starts.
    pub(crate) fn reject(&mut self) {
        self.attempt = Some(scioxide_core::AttemptRecord::new(
            None,
            scioxide_core::AttemptKind::Call,
            scioxide_core::AttemptOutcome::PreflightRejected,
            scioxide_core::SolveCounts::new(),
            None,
        ));
    }

    /// Run `work` in a fresh epoch of this slot's state and record the attempt.
    ///
    /// A scope that fails before its epoch begins stays a preflight rejection.
    /// `classify` decides the outcome and accepted progress after an epoch starts.
    pub(crate) fn run<'capability, T>(
        &mut self,
        scope: scioxide_core::SolveScope<'_, 'capability>,
        identity: scioxide_core::SolveIdentity,
        allowance: scioxide_core::SolveBudgets,
        work: impl FnOnce(&mut scioxide_core::SolveContext<'_, 'capability>) -> Result<T, SciOxideError>,
        classify: impl FnOnce(&Result<T, SciOxideError>) -> (scioxide_core::AttemptOutcome, Option<P>),
    ) -> Result<T, SciOxideError> {
        let previous = self.state.epoch();
        let result = scope.run(&mut self.state, identity, allowance, work);
        let started = self.state.epoch() != previous;
        let (outcome, progress) = if started {
            classify(&result)
        } else {
            (scioxide_core::AttemptOutcome::PreflightRejected, None)
        };
        self.attempt = Some(scioxide_core::AttemptRecord::new(
            if started { self.state.epoch() } else { None },
            scioxide_core::AttemptKind::Call,
            outcome,
            self.state.counts(),
            progress,
        ));
        result
    }
}
impl<P> Default for SolveSlot<P> {
    fn default() -> Self {
        Self::new()
    }
}

// Plans nest at most four checked stages (for example Riccati, its QZ pencil,
// a pivoted Householder factor and that factor's diagnostic SVD). Numerical
// kernels borrow those capabilities without opening further scopes.
const SCOPE_DEPTH: usize = 4;

#[derive(Debug)]
// Inline owned metadata keeps even borrowed workspace construction allocation-free.
#[expect(
    clippy::large_enum_variant,
    reason = "workspace variants retain reusable factor storage without allocation during execution"
)]
enum ScopeStorage<'a> {
    Owned([SolveSlot<()>; SCOPE_DEPTH]),
    Borrowed(&'a mut [SolveSlot<()>]),
}
impl ScopeStorage<'_> {
    const fn new() -> Self {
        Self::Owned([const { SolveSlot::new() }; SCOPE_DEPTH])
    }
    fn as_slice(&self) -> &[SolveSlot<()>] {
        match self {
            Self::Owned(slots) => slots,
            Self::Borrowed(slots) => slots,
        }
    }
    fn as_mut(&mut self) -> &mut [SolveSlot<()>] {
        match self {
            Self::Owned(slots) => slots,
            Self::Borrowed(slots) => slots,
        }
    }
}
impl Clone for ScopeStorage<'_> {
    fn clone(&self) -> Self {
        Self::Owned(core::array::from_fn(|index| {
            self.as_slice().get(index).cloned().unwrap_or_default()
        }))
    }
}

#[derive(Debug)]
enum Arena<'a, T> {
    Owned(Vec<T>),
    Borrowed(&'a mut [T]),
}
impl<T: Clone> Clone for Arena<'_, T> {
    fn clone(&self) -> Self {
        Self::Owned(self.as_slice().to_vec())
    }
}
impl<T> Arena<'_, T> {
    fn as_slice(&self) -> &[T] {
        match self {
            Self::Owned(v) => v,
            Self::Borrowed(v) => v,
        }
    }
    fn as_mut(&mut self) -> &mut [T] {
        match self {
            Self::Owned(v) => v,
            Self::Borrowed(v) => v,
        }
    }
    fn capacity(&self) -> usize {
        match self {
            Self::Owned(v) => v.capacity(),
            Self::Borrowed(v) => v.len(),
        }
    }
    fn owned_capacity(&self) -> usize {
        match self {
            Self::Owned(v) => v.capacity(),
            Self::Borrowed(_) => 0,
        }
    }
    fn ensure(&mut self, len: usize, value: impl FnMut() -> T) -> Result<(), SciOxideError> {
        match self {
            Self::Owned(values) => {
                reserve(values, len)?;
                values.resize_with(values.len().max(len), value);
                Ok(())
            }
            Self::Borrowed(values) if values.len() >= len => Ok(()),
            Self::Borrowed(values) => Err(SciOxideError::dimension_mismatch([len], [values.len()])),
        }
    }
}

#[derive(Debug)]
enum PartitionedArena<'a, S, const N: usize> {
    Owned {
        values: Vec<S>,
        regions: [usize; N],
    },
    Borrowed([&'a mut [S]; N]),
    Contiguous {
        values: &'a mut [S],
        regions: [usize; N],
    },
}
impl<S: Clone, const N: usize> Clone for PartitionedArena<'_, S, N> {
    fn clone(&self) -> Self {
        let regions = self.regions();
        let mut values = Vec::new();
        for region in 0..N {
            values.extend_from_slice(self.region(region));
        }
        Self::Owned { values, regions }
    }
}
impl<S, const N: usize> PartitionedArena<'_, S, N> {
    fn regions(&self) -> [usize; N] {
        match self {
            Self::Owned { regions, .. } | Self::Contiguous { regions, .. } => *regions,
            Self::Borrowed(regions) => core::array::from_fn(|i| regions[i].len()),
        }
    }
    fn capacity(&self) -> usize {
        match self {
            Self::Owned { values, .. } => values.capacity(),
            Self::Borrowed(regions) => regions.iter().map(|region| region.len()).sum(),
            Self::Contiguous { values, .. } => values.len(),
        }
    }
    fn owned_capacity(&self) -> usize {
        match self {
            Self::Owned { values, .. } => values.capacity(),
            Self::Borrowed(_) | Self::Contiguous { .. } => 0,
        }
    }
    fn region(&self, index: usize) -> &[S] {
        match self {
            Self::Owned { values, regions } => {
                let start: usize = regions[..index].iter().sum();
                &values[start..start + regions[index]]
            }
            Self::Contiguous { values, regions } => {
                let start: usize = regions[..index].iter().sum();
                &values[start..start + regions[index]]
            }
            Self::Borrowed(regions) => regions[index],
        }
    }
    fn region_mut(&mut self, index: usize) -> &mut [S] {
        match self {
            Self::Owned { values, regions } => {
                let start: usize = regions[..index].iter().sum();
                &mut values[start..start + regions[index]]
            }
            Self::Contiguous { values, regions } => {
                let start: usize = regions[..index].iter().sum();
                &mut values[start..start + regions[index]]
            }
            Self::Borrowed(regions) => regions[index],
        }
    }
    fn parts(&mut self) -> Result<[&mut [S]; N], SciOxideError> {
        match self {
            Self::Owned { values, regions } => partition(values, *regions),
            Self::Contiguous { values, regions } => partition(values, *regions),
            Self::Borrowed(regions) => Ok(regions.each_mut().map(|region| &mut **region)),
        }
    }
}
impl<S: Copy, const N: usize> PartitionedArena<'_, S, N> {
    /// Whether reserving `requested` needs no allocation. Owned and borrowed
    /// regions must already hold it; a contiguous slice may extend its
    /// partition in place or, when `repartition` holds, restart it.
    fn admits(&self, requested: [usize; N], repartition: bool) -> bool {
        let regions = self.regions();
        if regions
            .iter()
            .zip(requested)
            .all(|(&have, need)| have >= need)
        {
            return true;
        }
        let Self::Contiguous { values, .. } = self else {
            return false;
        };
        let fits = |regions: [usize; N]| arena_len(regions).is_ok_and(|len| len <= values.len());
        fits(core::array::from_fn(|i| regions[i].max(requested[i])))
            || (repartition && fits(requested))
    }
    /// Reserve `requested` while keeping existing regions. Outside any epoch
    /// (`repartition`) no region holds live data, so a contiguous slice that
    /// cannot extend its partition restarts it for this request alone.
    fn ensure(
        &mut self,
        requested: [usize; N],
        zero: impl Fn() -> S,
        repartition: bool,
    ) -> Result<(), SciOxideError> {
        match self {
            Self::Contiguous { values, regions } => {
                let next = core::array::from_fn(|i| regions[i].max(requested[i]));
                let len = arena_len(next)?;
                if len > values.len() {
                    let exact = arena_len(requested)?;
                    if repartition && exact <= values.len() {
                        values[..exact].fill(zero());
                        *regions = requested;
                        return Ok(());
                    }
                    return Err(SciOxideError::dimension_mismatch([len], [values.len()]));
                }
                let mut old_end: usize = regions.iter().sum();
                let mut new_end = len;
                for i in (0..N).rev() {
                    let old_start = old_end - regions[i];
                    let new_start = new_end - next[i];
                    values.copy_within(old_start..old_end, new_start);
                    values[new_start + regions[i]..new_end].fill(zero());
                    old_end = old_start;
                    new_end = new_start;
                }
                *regions = next;
            }
            Self::Borrowed(regions) => {
                for (region, required) in regions.iter().zip(requested) {
                    if region.len() < required {
                        return Err(SciOxideError::dimension_mismatch(
                            [required],
                            [region.len()],
                        ));
                    }
                }
            }
            Self::Owned { values, regions } => {
                let next = core::array::from_fn(|i| regions[i].max(requested[i]));
                if next == *regions {
                    return Ok(());
                }
                let len = arena_len(next)?;
                reserve(values, len)?;
                // The producer makes zero initialization local to this
                // monomorphization, even across codegen units.
                values.resize_with(len, &zero);
                // Growth moves later regions first. Existing candidates survive
                // preparation of another region in the same atomic operation.
                let mut old_end: usize = regions.iter().sum();
                let mut new_end = len;
                if old_end == 0 {
                    // resize_with initialized every region of a fresh arena.
                    *regions = next;
                    return Ok(());
                }
                for i in (0..N).rev() {
                    let old_start = old_end - regions[i];
                    let new_start = new_end - next[i];
                    values.copy_within(old_start..old_end, new_start);
                    values[new_start + regions[i]..new_end].fill(zero());
                    old_end = old_start;
                    new_end = new_start;
                }
                *regions = next;
            }
        }
        Ok(())
    }
}

fn partition<T, const N: usize>(
    values: &mut [T],
    lengths: [usize; N],
) -> Result<[&mut [T]; N], SciOxideError> {
    let mut parts = core::array::from_fn(|_| &mut [][..]);
    let mut remaining = values;
    for (part, length) in parts.iter_mut().zip(lengths) {
        let (head, tail) = split_mut(remaining, length)?;
        *part = head;
        remaining = tail;
    }
    Ok(parts)
}

struct WorkspaceParts<'a, S: NumericScalar> {
    scalars: [&'a mut [S]; 8],
    reals: [&'a mut [S::Real]; 2],
    indices: &'a mut [usize],
    accumulators: &'a mut [RangeAccumulator<S::Real>],
    extended: &'a mut [Extended<S>],
    scaled: &'a mut [ScaledValue<S>],
    exponents: &'a mut [i64],
}

pub(crate) struct ProductParts<'a, S: NumericScalar> {
    pub(crate) packed_a: &'a mut [S],
    pub(crate) packed_b: &'a mut [S],
    pub(crate) candidate: &'a mut [S],
    pub(crate) tiles: &'a mut [S],
    pub(crate) reals: &'a mut [S::Real],
    pub(crate) accumulators: &'a mut [RangeAccumulator<S::Real>],
}

fn arena_len<const N: usize>(regions: [usize; N]) -> Result<usize, SciOxideError> {
    regions.into_iter().try_fold(0usize, |sum, n| {
        sum.checked_add(n).ok_or_else(|| {
            SciOxideError::size_overflow(scioxide_core::quantity!("scalar_arena_elements"))
        })
    })
}

impl<S: NumericScalar> Default for LinalgWorkspace<'_, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, S: NumericScalar> LinalgWorkspace<'a, S> {
    fn from_borrowed(
        parts: WorkspaceParts<'a, S>,
        scopes: ScopeStorage<'a>,
        scope_depth: usize,
    ) -> Self {
        Self {
            scalars: PartitionedArena::Borrowed(parts.scalars),
            reals: PartitionedArena::Borrowed(parts.reals),
            indices: Arena::Borrowed(parts.indices),
            accumulators: Arena::Borrowed(parts.accumulators),
            extended: Arena::Borrowed(parts.extended),
            scaled: Arena::Borrowed(parts.scaled),
            exponents: Arena::Borrowed(parts.exponents),
            structured_dense_fallback: false,
            scopes,
            scope_depth,
        }
    }
    /// Returns an empty workspace that allocates on first use.
    pub const fn new() -> Self {
        Self {
            scopes: ScopeStorage::new(),
            scope_depth: 0,
            scalars: PartitionedArena::Owned {
                values: Vec::new(),
                regions: [0; 8],
            },
            reals: PartitionedArena::Owned {
                values: Vec::new(),
                regions: [0; 2],
            },
            indices: Arena::Owned(Vec::new()),
            accumulators: Arena::Owned(Vec::new()),
            extended: Arena::Owned(Vec::new()),
            scaled: Arena::Owned(Vec::new()),
            exponents: Arena::Owned(Vec::new()),
            structured_dense_fallback: false,
        }
    }

    /// Returns a workspace that uses `values` as its scalar buffer and never allocates.
    ///
    /// Only operations whose requirements consist of scalars can use it. A
    /// reservation that needs more than `values.len()` scalars, or any other
    /// buffer, is `DimensionMismatch`. [`check`](Self::check) returns
    /// `Sufficient` exactly when the reservation would succeed.
    pub fn from_scalar_slice(values: &'a mut [S]) -> Self {
        Self {
            structured_dense_fallback: false,
            scopes: ScopeStorage::new(),
            scope_depth: 0,
            scalars: PartitionedArena::Contiguous {
                values,
                regions: [0; 8],
            },
            reals: PartitionedArena::Borrowed([&mut [], &mut []]),
            indices: Arena::Borrowed(&mut []),
            accumulators: Arena::Borrowed(&mut []),
            extended: Arena::Borrowed(&mut []),
            scaled: Arena::Borrowed(&mut []),
            exponents: Arena::Borrowed(&mut []),
        }
    }

    /// Borrow fixed buffers, partitioning scalar storage according to a request.
    /// Borrowed storage never grows, and its extra elements do not count toward
    /// owned-workspace byte limits.
    /// # Errors
    /// Returns `SizeOverflow`, `ResourceLimit`, or `DimensionMismatch` for undersized storage.
    pub(crate) fn from_slices(
        slices: LinalgSlices<'a, S>,
        requirements: &LinalgRequirements,
    ) -> Result<Self, SciOxideError> {
        let LinalgSlices {
            scalars,
            reals,
            indices,
            accumulators,
            extended,
            scaled,
            exponents,
        } = slices;
        requirements.required_bytes::<S>()?;
        let lengths = [
            arena_len(requirements.scalars)?,
            arena_len(requirements.reals)?,
            requirements.indices,
            requirements.accumulators,
            requirements.extended,
            requirements.scaled,
            requirements.exponents,
        ];
        for (required, available) in lengths.into_iter().zip([
            scalars.len(),
            reals.len(),
            indices.len(),
            accumulators.len(),
            extended.len(),
            scaled.len(),
            exponents.len(),
        ]) {
            if available < required {
                return Err(SciOxideError::dimension_mismatch([required], [available]));
            }
        }
        let regions = partition(scalars, requirements.scalars)?;
        let real_regions = partition(reals, requirements.reals)?;
        let mut workspace = Self::from_borrowed(
            WorkspaceParts {
                scalars: regions,
                reals: real_regions,
                indices,
                accumulators,
                extended,
                scaled,
                exponents,
            },
            ScopeStorage::new(),
            0,
        );
        workspace.structured_dense_fallback = requirements.structured_dense_fallback;
        Ok(workspace)
    }

    /// Returns whether the workspace already meets `requirements` or needs to grow.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the required byte count does not fit in `usize`.
    /// - `ResourceLimit` if the requirement, or the current owned storage,
    ///   exceeds the byte limit.
    pub fn check(&self, requirements: &LinalgRequirements) -> Result<Capacity, SciOxideError> {
        requirements.required_bytes::<S>()?;
        self.check_limit(requirements.byte_limit)?;
        let repartition = self.scope_depth == 0;
        Ok(
            if self.scalars.admits(requirements.scalars, repartition)
                && self.reals.admits(requirements.reals, repartition)
                && self.indices.as_slice().len() >= requirements.indices
                && self.accumulators.as_slice().len() >= requirements.accumulators
                && self.extended.as_slice().len() >= requirements.extended
                && self.scaled.as_slice().len() >= requirements.scaled
                && self.exponents.as_slice().len() >= requirements.exponents
                && (!requirements.structured_dense_fallback || self.structured_dense_fallback)
            {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }

    /// Check that every buffer meets the operation's requirements.
    pub(crate) fn require_capacity(
        &self,
        requirements: &LinalgRequirements,
    ) -> Result<(), SciOxideError> {
        if let Some((required, available)) = self.capacity_shortfall(requirements) {
            return Err(SciOxideError::dimension_mismatch([required], [available]));
        }
        Ok(())
    }

    pub(crate) fn capacity_shortfall(
        &self,
        requirements: &LinalgRequirements,
    ) -> Option<(usize, usize)> {
        self.scalars
            .regions()
            .into_iter()
            .zip(requirements.scalars)
            .chain(self.reals.regions().into_iter().zip(requirements.reals))
            .chain([
                (self.indices.as_slice().len(), requirements.indices),
                (
                    self.accumulators.as_slice().len(),
                    requirements.accumulators,
                ),
                (self.extended.as_slice().len(), requirements.extended),
                (self.scaled.as_slice().len(), requirements.scaled),
                (self.exponents.as_slice().len(), requirements.exponents),
            ])
            .find_map(|(available, required)| {
                (available < required).then_some((required, available))
            })
    }

    /// Grows the buffers to meet `requirements`.
    ///
    /// Existing capacity is kept. Does nothing if the workspace already
    /// suffices.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the grown owned storage would exceed the byte limit.
    ///   This is checked before allocation.
    /// - `AllocationFailure` if growth fails.
    /// - `DimensionMismatch` if a borrowed buffer is too small.
    pub fn try_reserve(&mut self, requirements: &LinalgRequirements) -> Result<(), SciOxideError> {
        let _ = self.check(requirements)?;
        if self.capacity_shortfall(requirements).is_none()
            && (!requirements.structured_dense_fallback || self.structured_dense_fallback)
        {
            return Ok(());
        }
        let _ = self.reservation_bytes(requirements)?;
        let repartition = self.scope_depth == 0;
        self.scalars
            .ensure(requirements.scalars, || S::ZERO, repartition)?;
        self.reals.ensure(
            requirements.reals,
            || <S::Real as scioxide_core::AlgebraicScalar>::ZERO,
            repartition,
        )?;
        self.indices.ensure(requirements.indices, || 0)?;
        self.accumulators
            .ensure(requirements.accumulators, RangeAccumulator::new)?;
        self.extended
            .ensure(requirements.extended, || ScaledValue::ZERO)?;
        self.scaled
            .ensure(requirements.scaled, || ScaledValue::ZERO)?;
        self.exponents.ensure(requirements.exponents, || 0)?;
        self.check_limit(requirements.byte_limit)?;
        self.structured_dense_fallback |= requirements.structured_dense_fallback;
        Ok(())
    }

    /// Reserve requirements whose sizes were already validated for `S` by the plan.
    /// The byte limit and capacity are still checked on every call; growth goes
    /// through [`Self::try_reserve`].
    pub(crate) fn try_reserve_prepared(
        &mut self,
        requirements: &LinalgRequirements,
    ) -> Result<(), SciOxideError> {
        self.check_limit(requirements.byte_limit)?;
        if self.capacity_shortfall(requirements).is_none()
            && (!requirements.structured_dense_fallback || self.structured_dense_fallback)
        {
            return Ok(());
        }
        self.try_reserve(requirements)
    }

    /// Returns the record of the most recent factorization or solve run in this workspace.
    ///
    /// The record holds the outcome and the iteration and evaluation counts,
    /// but no numerical values. A call rejected during input validation
    /// replaces the earlier record with one that has zero counts. Kernels such
    /// as products create no record.
    pub fn last_attempt(&self) -> Option<&scioxide_core::AttemptRecord<()>> {
        self.scopes
            .as_slice()
            .get(self.scope_depth)
            .and_then(SolveSlot::last_attempt)
    }

    pub(crate) fn begin_attempt(&mut self) {
        if let Some(slot) = self.scopes.as_mut().get_mut(self.scope_depth) {
            slot.reject();
        }
    }

    /// Run a checked numerical stage. All validation failures must propagate
    /// before the output is written; nested stages use the next scope slot.
    pub(crate) fn run_scope<'scope, 'capability, T>(
        &mut self,
        context: crate::LinalgExecutionContext<'scope, 'capability>,
        identity: scioxide_core::SolveIdentity,
        budgets: scioxide_core::SolveBudgets,
        work: impl FnOnce(
            &mut Self,
            crate::KernelContext<'capability>,
            &mut scioxide_core::SolveContext<'_, 'capability>,
        ) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let level = self.scope_depth;
        let slot = self.scopes.as_mut().get_mut(level).ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("workspace"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "nested factorization has no retained scope slot",
            )
        })?;
        let mut slot = core::mem::take(slot);
        self.scope_depth += 1;
        let kernels = context.kernel();
        let result = slot.run(
            context.into_scope(),
            identity,
            budgets,
            |solve| work(self, kernels, solve),
            |result| match result {
                Ok(_) => (scioxide_core::AttemptOutcome::Succeeded, None),
                Err(error)
                    if matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
                            | scioxide_core::SciOxideErrorView::ConvergenceFailure {
                                reason: scioxide_core::ConvergenceReason::IterationLimit
                                    | scioxide_core::ConvergenceReason::EvaluationLimit,
                                ..
                            }
                    ) =>
                {
                    (scioxide_core::AttemptOutcome::Stopped, None)
                }
                Err(_) => (scioxide_core::AttemptOutcome::Failed, None),
            },
        );
        self.scope_depth -= 1;
        self.scopes.as_mut()[level] = slot;
        result
    }

    /// Returns the owned bytes this workspace would hold after reserving `requirements`.
    ///
    /// Existing capacity is included. Borrowed buffers count as zero. Nothing
    /// is allocated, so you can use it to include the workspace in a larger
    /// memory budget.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count does not fit in `usize`.
    /// - `ResourceLimit` if the result exceeds the byte limit.
    pub fn reservation_bytes(
        &self,
        requirements: &LinalgRequirements,
    ) -> Result<usize, SciOxideError> {
        requirements.required_bytes::<S>()?;
        let mut grown = *requirements;
        let current = self.scalars.regions();
        let scalar_regions =
            core::array::from_fn::<_, 8, _>(|i| current[i].max(requirements.scalars[i]));
        let scalars = if matches!(self.scalars, PartitionedArena::Owned { .. }) {
            self.scalars.capacity().max(arena_len(scalar_regions)?)
        } else {
            0
        };
        let current = self.reals.regions();
        let real_regions =
            core::array::from_fn::<_, 2, _>(|i| current[i].max(requirements.reals[i]));
        let reals = if matches!(self.reals, PartitionedArena::Owned { .. }) {
            self.reals.capacity().max(arena_len(real_regions)?)
        } else {
            0
        };
        grown.scalars = [scalars, 0, 0, 0, 0, 0, 0, 0];
        grown.reals = [reals, 0];
        grown.indices = if matches!(self.indices, Arena::Owned(_)) {
            self.indices.capacity().max(requirements.indices)
        } else {
            0
        };
        grown.accumulators = if matches!(self.accumulators, Arena::Owned(_)) {
            self.accumulators.capacity().max(requirements.accumulators)
        } else {
            0
        };
        grown.extended = if matches!(self.extended, Arena::Owned(_)) {
            self.extended.capacity().max(requirements.extended)
        } else {
            0
        };
        grown.scaled = if matches!(self.scaled, Arena::Owned(_)) {
            self.scaled.capacity().max(requirements.scaled)
        } else {
            0
        };
        grown.exponents = if matches!(self.exponents, Arena::Owned(_)) {
            self.exponents.capacity().max(requirements.exponents)
        } else {
            0
        };
        grown.required_bytes::<S>()
    }

    /// Returns the capacity of all owned buffers in bytes, excluding fixed-size metadata.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count does not fit in `usize`.
    pub fn retained_bytes(&self) -> Result<usize, SciOxideError> {
        LinalgRequirements {
            scalars: [self.scalars.owned_capacity(), 0, 0, 0, 0, 0, 0, 0],
            reals: [self.reals.owned_capacity(), 0],
            indices: self.indices.owned_capacity(),
            accumulators: self.accumulators.owned_capacity(),
            extended: self.extended.owned_capacity(),
            scaled: self.scaled.owned_capacity(),
            exponents: self.exponents.owned_capacity(),
            byte_limit: None,
            structured_dense_fallback: self.structured_dense_fallback,
        }
        .bytes::<S>()
    }

    fn check_limit(&self, limit: Option<usize>) -> Result<(), SciOxideError> {
        if limit.is_none() {
            return Ok(());
        }
        admit_bytes(
            self.retained_bytes()?,
            limit,
            scioxide_core::quantity!("linalg_workspace_bytes"),
        )
    }

    pub(crate) fn scalar_mut(&mut self, region: ScalarRegion) -> &mut [S] {
        self.scalars.region_mut(region as usize)
    }
    pub(crate) fn product_parts(&mut self) -> Result<ProductParts<'_, S>, SciOxideError> {
        let [packed_a, packed_b, candidate, _, _, _, tiles, _] = self.scalars.parts()?;
        Ok(ProductParts {
            packed_a,
            packed_b,
            candidate,
            tiles,
            reals: self.reals.region_mut(RealRegion::Exact as usize),
            accumulators: self.accumulators.as_mut(),
        })
    }

    pub(crate) fn with_direct<T>(
        &mut self,
        work: impl FnOnce(
            &mut [S],
            &mut [S::Real],
            &mut [usize],
            &mut LinalgWorkspace<'_, S>,
        ) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let mut regions = self.scalars.parts()?;
        let direct = core::mem::take(&mut regions[7]);
        let [exact, direct_reals] = self.reals.parts()?;
        let mut child = LinalgWorkspace::from_borrowed(
            WorkspaceParts {
                scalars: regions,
                reals: [exact, &mut []],
                indices: &mut [],
                accumulators: self.accumulators.as_mut(),
                extended: self.extended.as_mut(),
                scaled: self.scaled.as_mut(),
                exponents: self.exponents.as_mut(),
            },
            ScopeStorage::Borrowed(self.scopes.as_mut()),
            self.scope_depth,
        );
        work(direct, direct_reals, self.indices.as_mut(), &mut child)
    }

    pub(crate) fn with_prefix<T>(
        &mut self,
        length: usize,
        work: impl FnOnce(&mut [S], &mut LinalgWorkspace<'_, S>) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        self.with_prefixes([length, 0, 0], |scalars, _, _, child| work(scalars, child))
    }

    /// Lend a parent stage and the remaining direct arena to a nested operation.
    #[inline(always)]
    pub(crate) fn with_prefixes<T>(
        &mut self,
        lengths: [usize; 3],
        work: impl FnOnce(
            &mut [S],
            &mut [S::Real],
            &mut [usize],
            &mut LinalgWorkspace<'_, S>,
        ) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let mut regions = self.scalars.parts()?;
        let direct = core::mem::take(&mut regions[7]);
        let (prefix, rest) = crate::workspace::split_mut(direct, lengths[0])?;
        regions[7] = rest;
        let mut real_regions = self.reals.parts()?;
        let direct_reals = core::mem::take(&mut real_regions[1]);
        let (real_prefix, real_rest) = crate::workspace::split_mut(direct_reals, lengths[1])?;
        real_regions[1] = real_rest;
        let (index_prefix, index_rest) =
            crate::workspace::split_mut(self.indices.as_mut(), lengths[2])?;
        let mut child = LinalgWorkspace::from_borrowed(
            WorkspaceParts {
                scalars: regions,
                reals: real_regions,
                indices: index_rest,
                accumulators: self.accumulators.as_mut(),
                extended: self.extended.as_mut(),
                scaled: self.scaled.as_mut(),
                exponents: self.exponents.as_mut(),
            },
            ScopeStorage::Borrowed(self.scopes.as_mut()),
            self.scope_depth,
        );
        work(prefix, real_prefix, index_prefix, &mut child)
    }

    /// Initialized extended-range scratch for leaf estimators.
    pub(crate) fn extended_mut(&mut self) -> &mut [Extended<S>] {
        self.extended.as_mut()
    }

    /// Lend scaled values and signed exponent scratch without borrowing GEMM arenas.
    pub(crate) fn with_scaled<T>(
        &mut self,
        work: impl FnOnce(
            &mut [ScaledValue<S>],
            &mut [i64],
            &mut LinalgWorkspace<'_, S>,
        ) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let mut child = LinalgWorkspace::from_borrowed(
            WorkspaceParts {
                scalars: self.scalars.parts()?,
                reals: self.reals.parts()?,
                indices: self.indices.as_mut(),
                accumulators: self.accumulators.as_mut(),
                extended: self.extended.as_mut(),
                scaled: &mut [],
                exponents: &mut [],
            },
            ScopeStorage::Borrowed(self.scopes.as_mut()),
            self.scope_depth,
        );
        work(self.scaled.as_mut(), self.exponents.as_mut(), &mut child)
    }

    pub(crate) fn with_regions<const N: usize, T>(
        &mut self,
        selected: [ScalarRegion; N],
        work: impl FnOnce([&mut [S]; N], &mut LinalgWorkspace<'_, S>) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let mut regions = self.scalars.parts()?;
        let buffers = core::array::from_fn(|i| core::mem::take(&mut regions[selected[i] as usize]));
        let mut child = LinalgWorkspace::from_borrowed(
            WorkspaceParts {
                scalars: regions,
                reals: self.reals.parts()?,
                indices: self.indices.as_mut(),
                accumulators: self.accumulators.as_mut(),
                extended: self.extended.as_mut(),
                scaled: self.scaled.as_mut(),
                exponents: self.exponents.as_mut(),
            },
            ScopeStorage::Borrowed(self.scopes.as_mut()),
            self.scope_depth,
        );
        work(buffers, &mut child)
    }
}

fn reserve<T>(values: &mut Vec<T>, required: usize) -> Result<(), SciOxideError> {
    scioxide_core::backend::try_reserve(
        values,
        required.saturating_sub(values.len()),
        scioxide_core::quantity!("linalg_workspace_elements"),
    )
}

impl<S: NumericScalar> LinalgWorkspace<'_, S> {
    /// Returns the buffer lengths and byte limit that `request` needs.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if a size does not fit in `usize`.
    /// - `ResourceLimit` if the required bytes exceed the byte limit of the request.
    pub fn workspace_requirements(
        request: &impl LinalgWorkspaceRequest<S>,
    ) -> Result<LinalgRequirements, SciOxideError> {
        request.requirements()
    }

    /// Returns a new workspace reserved for `request`.
    ///
    /// # Errors
    ///
    /// - Errors of [`workspace_requirements`](Self::workspace_requirements).
    /// - `AllocationFailure` if the buffers cannot be allocated.
    pub fn workspace(
        request: &impl LinalgWorkspaceRequest<S>,
    ) -> Result<LinalgWorkspace<'static, S>, SciOxideError> {
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&Self::workspace_requirements(request)?)?;
        Ok(workspace)
    }
}

/// Check workspace bytes against a byte limit, naming the family's quantity on failure.
pub(crate) fn admit_bytes(
    bytes: usize,
    limit: Option<usize>,
    quantity: scioxide_core::Quantity,
) -> Result<(), scioxide_core::SciOxideError> {
    if let Some(limit) = limit
        && bytes > limit
    {
        return Err(scioxide_core::SciOxideError::resource_limit(
            quantity, bytes, limit,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod scope_tests {
    use super::*;
    use scioxide_core::{AttemptOutcome, ProblemToken, SolveBudgets, SolveIdentity};

    #[test]
    fn nested_epochs_survive_prefix_lending_and_keep_disjoint_counts() {
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("test", "nested_linalg", 1),
            0,
            0,
        );
        let mut workspace = LinalgWorkspace::<f64>::new();
        let mut epochs = [None, None];
        for epoch in &mut epochs {
            workspace.begin_attempt();
            *epoch = workspace
                .run_scope(
                    crate::LinalgExecutionContext::sequential(),
                    identity,
                    SolveBudgets::unbounded(),
                    |workspace, kernels, solve| {
                        workspace.with_prefix(0, |_, child| {
                            child.begin_attempt();
                            child.run_scope(
                                kernels.scoped(solve),
                                identity,
                                SolveBudgets::unbounded(),
                                |_, _, solve| {
                                    assert!(solve.record_iteration()?.is_granted());
                                    Ok(())
                                },
                            )?;
                            Ok(child.last_attempt().unwrap().epoch())
                        })
                    },
                )
                .unwrap();
            let attempt = workspace.last_attempt().unwrap();
            assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
            assert_eq!(attempt.counts().own().iterations(), 0);
            assert_eq!(attempt.counts().nested().iterations(), 1);
        }
        assert!(epochs.iter().all(Option::is_some));
        assert_ne!(epochs[0], epochs[1]);
        workspace.begin_attempt();
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
        assert!(attempt.epoch().is_none());
        assert_eq!(attempt.counts(), scioxide_core::SolveCounts::new());
    }

    fn nest(
        workspace: &mut LinalgWorkspace<'_, f64>,
        context: crate::LinalgExecutionContext<'_, '_>,
        levels: usize,
    ) -> Result<usize, SciOxideError> {
        let identity = SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("test", "nested_linalg_depth", 1),
            0,
            0,
        );
        workspace.begin_attempt();
        workspace.run_scope(
            context,
            identity,
            SolveBudgets::unbounded(),
            |workspace, kernels, solve| {
                if levels == 1 {
                    return Ok(1);
                }
                workspace.with_prefix(0, |_, child| {
                    Ok(1 + nest(child, kernels.scoped(solve), levels - 1)?)
                })
            },
        )
    }

    #[test]
    fn four_nested_stages_fit_and_a_fifth_is_rejected() {
        let mut workspace = LinalgWorkspace::<f64>::new();
        assert_eq!(
            nest(
                &mut workspace,
                crate::LinalgExecutionContext::sequential(),
                SCOPE_DEPTH
            )
            .unwrap(),
            SCOPE_DEPTH
        );
        let error = nest(
            &mut workspace,
            crate::LinalgExecutionContext::sequential(),
            SCOPE_DEPTH + 1,
        )
        .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod layout_tests {
    use super::*;
    use scioxide_core::SciOxideErrorView;

    fn short<T>(result: Result<T, SciOxideError>) {
        assert!(result.err().is_some_and(|error| matches!(
            error.view(),
            SciOxideErrorView::DimensionMismatch { .. }
        )));
    }

    #[test]
    fn partitioned_growth_initializes_new_entries_and_preserves_regions()
    -> Result<(), SciOxideError> {
        let mut storage = [99.0_f64; 12];
        for mut arena in [
            PartitionedArena::Owned {
                values: Vec::with_capacity(12),
                regions: [0; 3],
            },
            PartitionedArena::Contiguous {
                values: &mut storage,
                regions: [0; 3],
            },
        ] {
            arena.ensure([0; 3], || 0.0, false)?;
            arena.ensure([2, 0, 3], || 0.0, false)?;
            assert_eq!(arena.region(0), &[0.0; 2]);
            assert!(arena.region(1).is_empty());
            assert_eq!(arena.region(2), &[0.0; 3]);
            arena.region_mut(0).copy_from_slice(&[1.0, 2.0]);
            arena.region_mut(2).copy_from_slice(&[3.0, 4.0, 5.0]);
            arena.ensure([4, 2, 4], || 0.0, false)?;
            assert_eq!(arena.region(0), &[1.0, 2.0, 0.0, 0.0]);
            assert_eq!(arena.region(1), &[0.0; 2]);
            assert_eq!(arena.region(2), &[3.0, 4.0, 5.0, 0.0]);
            arena.ensure([1, 0, 2], || 0.0, false)?;
            assert_eq!(arena.regions(), [4, 2, 4]);
            assert_eq!(arena.region(2), &[3.0, 4.0, 5.0, 0.0]);
        }
        Ok(())
    }

    #[test]
    fn contiguous_partitions_restart_only_outside_epochs() -> Result<(), SciOxideError> {
        let mut storage = [99.0_f64; 12];
        let mut arena = PartitionedArena::Contiguous {
            values: &mut storage,
            regions: [0; 3],
        };
        arena.ensure([10, 0, 0], || 0.0, true)?;
        assert!(!arena.admits([0, 12, 0], false));
        assert!(arena.admits([0, 12, 0], true));
        // Inside an epoch the live region must survive, so the slice is too short.
        assert_eq!(
            arena.ensure([0, 12, 0], || 0.0, false).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::DimensionMismatch
        );
        arena.ensure([0, 12, 0], || 0.0, true)?;
        assert_eq!(arena.regions(), [0, 12, 0]);
        assert_eq!(arena.region(1), &[0.0; 12]);
        Ok(())
    }

    #[test]
    fn partitions_reject_short_storage_and_accept_empty_boundaries() -> Result<(), SciOxideError> {
        let mut values = [0.0; 4];
        let [first, empty, last] = partition(&mut values, [2, 0, 2])?;
        first.fill(1.0);
        assert!(empty.is_empty());
        last.fill(2.0);
        assert_eq!(values, [1.0, 1.0, 2.0, 2.0]);
        short(partition(&mut values, [2, 3]));
        short(partition(&mut values, [usize::MAX, 1]));
        Ok(())
    }

    #[test]
    fn each_borrowed_arena_rejects_a_one_element_shortfall() -> Result<(), SciOxideError> {
        let mut requirements = LinalgRequirements::new(None).with_scalar(ScalarRegion::Direct, 2);
        *requirements.real_mut(RealRegion::Direct) = 2;
        requirements.indices = 2;
        requirements.accumulators = 2;
        requirements.extended = 2;
        requirements.scaled = 2;
        requirements.exponents = 2;
        for missing in 0..7 {
            let mut scalars = [0.0_f64; 2];
            let mut reals = [0.0_f64; 2];
            let mut indices = [0; 2];
            let mut accumulators = core::array::from_fn::<_, 2, _>(|_| RangeAccumulator::new());
            let mut extended = [ScaledValue::ZERO; 2];
            let mut scaled = [ScaledValue::ZERO; 2];
            let mut exponents = [0; 2];
            let len = |arena| if missing == arena { 1 } else { 2 };
            short(LinalgWorkspace::<f64>::from_slices(
                LinalgSlices {
                    scalars: &mut scalars[..len(0)],
                    reals: &mut reals[..len(1)],
                    indices: &mut indices[..len(2)],
                    accumulators: &mut accumulators[..len(3)],
                    extended: &mut extended[..len(4)],
                    scaled: &mut scaled[..len(5)],
                    exponents: &mut exponents[..len(6)],
                },
                &requirements,
            ));
        }
        Ok(())
    }

    #[test]
    fn prefix_shortfalls_never_enter_the_child_and_capacity_checks_never_grow()
    -> Result<(), SciOxideError> {
        let requirements = LinalgRequirements::new(None).with_scalar(ScalarRegion::Direct, 2);
        let mut workspace = LinalgWorkspace::<f64>::new();
        workspace.try_reserve(&requirements)?;
        let bytes = workspace.retained_bytes()?;
        let entered = core::cell::Cell::new(false);
        short(workspace.with_prefix(3, |_, _| {
            entered.set(true);
            Ok(())
        }));
        short(workspace.with_prefixes([0, 1, 0], |_, _, _, _| {
            entered.set(true);
            Ok(())
        }));
        short(workspace.with_prefixes([0, 0, 1], |_, _, _, _| {
            entered.set(true);
            Ok(())
        }));
        short(workspace.require_capacity(&requirements.with_scalar(ScalarRegion::Candidate, 1)));
        assert!(!entered.get());
        assert_eq!(workspace.retained_bytes()?, bytes);
        workspace.with_prefix(2, |prefix, child| {
            assert_eq!(prefix.len(), 2);
            child.with_prefix(0, |empty, _| {
                assert!(empty.is_empty());
                Ok(())
            })
        })?;
        Ok(())
    }

    #[test]
    fn requirement_union_preserves_named_regions_and_the_stricter_limit()
    -> Result<(), SciOxideError> {
        let a = LinalgRequirements::new(Some(128)).with_scalar(ScalarRegion::Direct, 3);
        let b = LinalgRequirements::new(Some(64)).with_scalar(ScalarRegion::Candidate, 4);
        let combined = a.max(&b);
        assert_eq!(combined.scalar(ScalarRegion::Direct), 3);
        assert_eq!(combined.scalar(ScalarRegion::Candidate), 4);
        assert_eq!(combined.scalar_len()?, 7);
        assert_eq!(combined.required_bytes::<f64>()?, 56);
        let excessive = combined.with_scalar(ScalarRegion::PackedA, 2);
        assert!(
            excessive
                .required_bytes::<f64>()
                .err()
                .is_some_and(|error| matches!(
                    error.view(),
                    SciOxideErrorView::ResourceLimit { .. }
                ))
        );
        Ok(())
    }
}
