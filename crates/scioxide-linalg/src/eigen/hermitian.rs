//! Dense Hermitian and real symmetric tridiagonal eigenproblems.
//!
//! Dense input is reduced to real symmetric tridiagonal form by Householder
//! similarity transformations (unblocked through order 64, width-32 panels
//! above). The tridiagonal problem is solved by Pal–Walker–Kahan QL/QR (values
//! only), implicit QL through order 32 or divide and conquer above it
//! (vectors), or Sturm bisection and inverse iteration (selected indices and
//! relative accuracy). Vectors are back-transformed with compact-WY products.
//! The method depends only on the order and the job.

use crate::workspace::{RealRegion, ScalarRegion};
use scioxide_core::AlgebraicScalar;
mod blocked;
mod divide_conquer;
pub(super) mod first_components;
mod selected;
mod values;

use crate::HermitianEigenvaluesResult;
use core::cmp::Ordering;

use scioxide_core::backend::{checked_add, checked_product};
use scioxide_core::{
    ConvergenceDiagnostics, ConvergenceReason, Dimensions, ExecutionContext, ExecutionPolicy,
    NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError, SolveContext,
};

use crate::workspace::storage::{MatrixStorage, ScratchMatrix};
use crate::{
    HermitianInput, HermitianProjectionReport, KernelContext, LinalgExecutionContext,
    LinalgRequirements, LinalgWorkspace, MatrixView,
};

use super::hermitian_types::HermitianStorage;
use super::{
    EigenvalueAccuracy, EigenvalueOrder, HermitianEigenDecomposition, HermitianEigenOptions,
    HermitianEigenWorkspaceRequest, HermitianEigenvaluesBuffer, hermitian_reduction_scale,
    normalize_slice,
};

/// Orders above this crossover reduce with blocked panels.
const BLOCKED_ORDER: usize = 64;

/// Complete vector jobs divide and conquer above this order, independent of ISA.
/// Paired tridiagonal/eigh measurements favor QL at 32 and merges from 33 upward.
const VECTOR_ORDER: usize = 32;

impl<R: RealScalar> HermitianEigenOptions<R> {
    /// Resolve one allowance for the original job, before splitting into leaves.
    pub(super) fn resolved_iteration_budget(
        self,
        order: usize,
        vectors: bool,
    ) -> Result<usize, SciOxideError> {
        if let Some(budget) = self.iteration_budget() {
            return Ok(budget.get());
        }
        let n = order.max(1);
        let per_value = if self.selects() {
            // Bisection can traverse the entire exponent range for a tiny root;
            // significand bits, endpoint padding and 16 inverse iterations follow.
            let exponent = crate::kernel::standard::exponent;
            (exponent(R::MAX)? - exponent(R::MIN_SUBNORMAL)?) as usize
                + R::MANTISSA_DIGITS as usize
                + 32
        } else if vectors && order > VECTOR_ORDER {
            let leaves = n.div_ceil(divide_conquer::LEAF);
            30 * (1 + (usize::BITS - (leaves - 1).leading_zeros()) as usize)
        } else {
            30
        };
        scioxide_core::backend::checked_mul(
            n,
            per_value,
            scioxide_core::quantity!("eigen_iteration_budget"),
        )
    }
}

/// Whether a job computes eigenvectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Job {
    Values,
    Vectors,
}

/// Dense reduction to tridiagonal form; `B` is the blocked panel scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reduction<B = ()> {
    Unblocked,
    Blocked(B),
}

/// Tridiagonal eigensolver; `D` is the divide-and-conquer scratch. Values-only
/// jobs run Pal–Walker–Kahan QL/QR where vector jobs run [`Solver::Ql`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Solver<D = ()> {
    Bisection,
    Ql,
    DivideConquer(D),
}

impl Reduction {
    /// Blocked exactly above the crossover.
    const fn for_order(order: usize) -> Self {
        if order > BLOCKED_ORDER {
            Self::Blocked(())
        } else {
            Self::Unblocked
        }
    }

    fn try_map<B>(
        self,
        blocked: impl FnOnce() -> Result<B, SciOxideError>,
    ) -> Result<Reduction<B>, SciOxideError> {
        Ok(match self {
            Self::Unblocked => Reduction::Unblocked,
            Self::Blocked(()) => Reduction::Blocked(blocked()?),
        })
    }
}

impl Solver {
    /// Selected indices and relative accuracy bisect; complete vector jobs
    /// divide and conquer above the crossover; remaining jobs use implicit
    /// tridiagonal iteration (guarded QL for vectors, QL/QR for values).
    fn new<R: RealScalar>(order: usize, job: Job, options: &HermitianEigenOptions<R>) -> Self {
        if options.selects() {
            Self::Bisection
        } else if job == Job::Vectors && order > VECTOR_ORDER {
            Self::DivideConquer(())
        } else {
            Self::Ql
        }
    }

    fn try_map<D>(
        self,
        divide_conquer: impl FnOnce() -> Result<D, SciOxideError>,
    ) -> Result<Solver<D>, SciOxideError> {
        Ok(match self {
            Self::Bisection => Solver::Bisection,
            Self::Ql => Solver::Ql,
            Self::DivideConquer(()) => Solver::DivideConquer(divide_conquer()?),
        })
    }
}

fn scratch_overflow() -> SciOxideError {
    SciOxideError::size_overflow(scioxide_core::quantity!("hermitian_eigen_scratch"))
}

fn add(a: usize, b: usize) -> Result<usize, SciOxideError> {
    checked_add(a, b, scioxide_core::quantity!("hermitian_eigen_scratch"))
}

fn square(order: usize) -> Result<usize, SciOxideError> {
    checked_product(
        &[order, order],
        scioxide_core::quantity!("hermitian_eigen_scratch"),
    )
}

/// Direct-arena geometry of a dense job: exactly the scratch of its plan.
#[derive(Debug, Clone, Copy)]
struct DenseLayout {
    order: usize,
    job: Job,
    reduction: Reduction,
    solver: Solver,
}

impl DenseLayout {
    fn new<R: RealScalar>(order: usize, job: Job, options: &HermitianEigenOptions<R>) -> Self {
        Self {
            order,
            job,
            reduction: Reduction::for_order(order),
            solver: Solver::new(order, job, options),
        }
    }

    /// Scalar, real and index elements of the direct arenas.
    fn lengths(self) -> Result<[usize; 3], SciOxideError> {
        let n = self.order;
        let matrix = square(n)?;
        let vectors = match self.job {
            Job::Values => 0,
            Job::Vectors => matrix,
        };
        let vectors_len = n.checked_mul(3).ok_or_else(scratch_overflow)?;
        let mut scalars = add(
            add(matrix, vectors)?,
            add(n.saturating_sub(2), vectors_len)?,
        )?;
        if self.reduction == Reduction::Blocked(()) {
            scalars = add(scalars, blocked::Scratch::<f64>::len(n)?)?;
        }
        let mut reals = vectors_len;
        if self.job == Job::Vectors && self.solver == Solver::Ql {
            reals = add(reals, n.checked_mul(2).ok_or_else(scratch_overflow)?)?;
        }
        let mut indices = 0;
        if self.solver == Solver::DivideConquer(()) {
            let [dc_scalars, dc_reals, dc_indices] = divide_conquer::Workspace::<f64>::lengths(n)?;
            scalars = add(scalars, dc_scalars)?;
            reals = add(reals, dc_reals)?;
            indices = dc_indices;
        }
        Ok([scalars, reals, indices])
    }
}

/// Shared scratch of the request's dense or tridiagonal job.
pub(super) fn requirements<S: NumericScalar>(
    request: &HermitianEigenWorkspaceRequest<S::Real>,
    job: Job,
) -> Result<LinalgRequirements, SciOxideError> {
    match request.storage {
        HermitianStorage::Dense => dense_requirements::<S>(request, job),
        HermitianStorage::Tridiagonal => tridiagonal_requirements::<S>(request, job),
    }
}

/// Shared scratch of a dense job's plan.
pub(super) fn dense_requirements<S: NumericScalar>(
    request: &HermitianEigenWorkspaceRequest<S::Real>,
    job: Job,
) -> Result<LinalgRequirements, SciOxideError> {
    let layout = DenseLayout::new(request.order, job, &request.options);
    let [scalars, reals, indices] = layout.lengths()?;
    let mut requirements =
        LinalgRequirements::new(request.options.execution_options().byte_limit());
    *requirements.scalar_mut(ScalarRegion::Direct) = scalars;
    *requirements.real_mut(RealRegion::Direct) = reals;
    requirements.indices = indices;
    let n = layout.order;
    if layout.reduction == Reduction::Blocked(()) {
        for (rows, shared, cols, structured) in
            blocked::Scratch::<S>::products(n, job == Job::Vectors)
        {
            requirements = requirements.max(&request.product::<S>(rows, shared, cols, structured)?);
        }
    }
    if layout.solver == Solver::DivideConquer(()) {
        requirements = requirements.max(&request.product::<S>(n, n, n, false)?);
    }
    requirements.required_bytes::<S>()?;
    Ok(requirements)
}

/// Shared scratch of a real symmetric tridiagonal job.
fn tridiagonal_requirements<S: NumericScalar>(
    request: &HermitianEigenWorkspaceRequest<S::Real>,
    job: Job,
) -> Result<LinalgRequirements, SciOxideError> {
    let n = request.order;
    let layout = TridiagonalLayout::new(n, job, &request.options)?;
    let [scalars, reals, indices] = layout.lengths()?;
    let mut requirements =
        LinalgRequirements::new(request.options.execution_options().byte_limit());
    *requirements.scalar_mut(ScalarRegion::Direct) = scalars;
    *requirements.real_mut(RealRegion::Direct) = reals;
    requirements.indices = indices;
    if layout.solver == Solver::DivideConquer(()) {
        requirements = requirements.max(&request.product::<S>(n, n, n, false)?);
    }
    requirements.required_bytes::<S>()?;
    Ok(requirements)
}

#[derive(Debug, Clone, Copy)]
struct TridiagonalLayout {
    order: usize,
    count: usize,
    job: Job,
    solver: Solver,
}

impl TridiagonalLayout {
    fn new<R: RealScalar>(
        order: usize,
        job: Job,
        options: &HermitianEigenOptions<R>,
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            order,
            count: options.resolve_indices(order)?.len(),
            job,
            solver: Solver::new(order, job, options),
        })
    }

    fn lengths(self) -> Result<[usize; 3], SciOxideError> {
        let n = self.order;
        let pair = n.checked_mul(2).ok_or_else(scratch_overflow)?;
        let mut scalars = if self.job == Job::Vectors {
            checked_product(&[n, self.count], scioxide_core::quantity!("eigen_scratch"))?
        } else {
            0
        };
        let mut reals = pair;
        let mut indices = 0;
        match self.solver {
            Solver::Bisection => {
                if self.job == Job::Vectors {
                    // Contiguous inverse-iteration rows and the transposed candidate.
                    scalars = add(add(scalars, scalars)?, pair)?;
                }
                reals = add(reals, n)?;
            }
            Solver::Ql => {
                if self.job == Job::Vectors {
                    reals = add(reals, pair)?;
                }
            }
            Solver::DivideConquer(()) => {
                let [dc_scalars, dc_reals, dc_indices] =
                    divide_conquer::Workspace::<f64>::lengths(n)?;
                scalars = add(scalars, dc_scalars)?;
                reals = add(reals, dc_reals)?;
                indices = dc_indices;
            }
        }
        Ok([scalars, reals, indices])
    }
}

/// Borrowed eigensystem candidate, valid until the job's arenas are released.
///
/// `values` holds the selected eigenvalues in output order. `vectors` holds
/// the matching columns (`order × values.len()`) or is empty for values jobs.
pub(super) struct Candidate<'c, 'a, 'g, S: NumericScalar> {
    pub(super) values: &'c mut [S::Real],
    pub(super) vectors: &'c mut ScratchMatrix<'a, S>,
    pub(super) gemm: &'c mut LinalgWorkspace<'g, S>,
    pub(super) projection: Option<HermitianProjectionReport<S::Real>>,
}

struct DenseScratch<'a, S: NumericScalar> {
    packed: ScratchMatrix<'a, S>,
    eigenvectors: ScratchMatrix<'a, S>,
    beta: &'a mut [S],
    phases: &'a mut [S],
    vector: &'a mut [S],
    work: &'a mut [S],
    reduction: Reduction<blocked::Scratch<'a, S>>,
    solver: Solver<divide_conquer::Workspace<'a, S>>,
    diagonal: &'a mut [S::Real],
    off_diagonal: &'a mut [S::Real],
    values: &'a mut [S::Real],
    rotations: &'a mut [S::Real],
}

impl<'a, S: NumericScalar> DenseScratch<'a, S> {
    fn new(
        layout: DenseLayout,
        scalars: &'a mut [S],
        reals: &'a mut [S::Real],
        indices: &'a mut [usize],
    ) -> Result<Self, SciOxideError> {
        let n = layout.order;
        let vectors = match layout.job {
            Job::Values => 0,
            Job::Vectors => n,
        };
        let (packed, scalars) = crate::workspace::split_mut(scalars, n * n)?;
        let (eigenvectors, scalars) = crate::workspace::split_mut(scalars, vectors * vectors)?;
        let (beta, scalars) = crate::workspace::split_mut(scalars, n.saturating_sub(2))?;
        let (phases, scalars) = crate::workspace::split_mut(scalars, n)?;
        let (vector, scalars) = crate::workspace::split_mut(scalars, n)?;
        let (work, mut scalars) = crate::workspace::split_mut(scalars, n)?;
        let reduction = layout.reduction.try_map(|| {
            let (blocks, rest) = blocked::Scratch::new(core::mem::take(&mut scalars), n)?;
            scalars = rest;
            Ok(blocks)
        })?;
        let (diagonal, reals) = crate::workspace::split_mut(reals, n)?;
        let (off_diagonal, reals) = crate::workspace::split_mut(reals, n)?;
        let (values, reals) = crate::workspace::split_mut(reals, n)?;
        let rotation_len = if layout.job == Job::Vectors && layout.solver == Solver::Ql {
            2 * n
        } else {
            0
        };
        let (rotations, reals) = crate::workspace::split_mut(reals, rotation_len)?;
        let solver = layout
            .solver
            .try_map(|| divide_conquer::Workspace::new(scalars, reals, indices, n))?;
        Ok(Self {
            packed: ScratchMatrix::new(packed, n, n)?,
            eigenvectors: ScratchMatrix::new(eigenvectors, vectors, vectors)?,
            beta,
            phases,
            vector,
            work,
            reduction,
            solver,
            diagonal,
            off_diagonal,
            values,
            rotations,
        })
    }
}

/// Compute a dense candidate from a validated finite Hermitian view.
///
/// The explicit interpretation is staged before reduction. Iterations are
/// charged to `solve`; dense products use `workspace`'s product regions.
#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
pub(super) fn dense_candidate<S: NumericScalar, T>(
    matrix: MatrixView<'_, S>,
    interpretation: HermitianInput<S::Real>,
    job: Job,
    options: &HermitianEigenOptions<S::Real>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
    solve: &mut SolveContext<'_, '_>,
    consume: impl FnOnce(Candidate<'_, '_, '_, S>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let projection = stage_dense_input(matrix, interpretation, workspace)?;
    prepared_dense_candidate(
        matrix.rows(),
        projection,
        job,
        options,
        workspace,
        context,
        solve,
        consume,
    )
}

fn stage_dense_input<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    interpretation: HermitianInput<S::Real>,
    workspace: &mut LinalgWorkspace<'_, S>,
) -> Result<Option<HermitianProjectionReport<S::Real>>, SciOxideError> {
    let n = matrix.rows();
    workspace.with_direct(|scalars, _, _, _| {
        crate::decompose::hermitian::prepare(
            matrix,
            interpretation,
            &mut crate::MatrixViewMut::from_slice(n, n, &mut scalars[..n * n])?,
            scioxide_core::argument!("a"),
        )
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
fn prepared_dense_candidate<S: NumericScalar, T>(
    order: usize,
    projection: Option<HermitianProjectionReport<S::Real>>,
    job: Job,
    options: &HermitianEigenOptions<S::Real>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
    solve: &mut SolveContext<'_, '_>,
    consume: impl FnOnce(Candidate<'_, '_, '_, S>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    let layout = DenseLayout::new(order, job, options);
    workspace.with_direct(|scalars, reals, indices, gemm| {
        let mut scratch = DenseScratch::new(layout, scalars, reals, indices)?;
        let scale = hermitian_reduction_scale(crate::kernel::standard::slice_maximum(
            scratch.packed.as_slice(),
        ));
        let count = compute_dense(scale, layout, &mut scratch, options, gemm, context, solve)?;
        let _ = solve.counts().total()?;
        consume(Candidate {
            values: &mut scratch.values[..count],
            vectors: &mut scratch.eigenvectors,
            gemm,
            projection,
        })
    })
}

fn compute_dense<S: NumericScalar>(
    scale: S::Real,
    layout: DenseLayout,
    scratch: &mut DenseScratch<'_, S>,
    options: &HermitianEigenOptions<S::Real>,
    gemm: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
    solve: &mut SolveContext<'_, '_>,
) -> Result<usize, SciOxideError> {
    let order = layout.order;
    let policy = options.execution_options().execution_policy();
    normalize_slice(scratch.packed.as_mut_slice(), scale);
    reduce_to_tridiagonal(
        &mut scratch.packed,
        scratch.diagonal,
        scratch.off_diagonal,
        scratch.beta,
        scratch.phases,
        scratch.vector,
        scratch.work,
        &mut scratch.reduction,
        gemm,
        policy,
        context,
    )?;
    scratch.values.copy_from_slice(scratch.diagonal);
    if layout.job == Job::Values {
        let count = match &scratch.solver {
            Solver::Bisection => selected::solve(
                scratch.diagonal,
                scratch.off_diagonal,
                scratch.values,
                None::<&mut ScratchMatrix<'_, S>>,
                scratch.vector,
                scratch.work,
                options,
                context.execution(),
                solve,
            )?,
            Solver::Ql | Solver::DivideConquer(_) => {
                values::solve(
                    scratch.values,
                    scratch.off_diagonal,
                    options,
                    context.execution(),
                    solve,
                )?;
                order
            }
        };
        let values = &mut scratch.values[..count];
        rescale_real_values(values, scale)?;
        order_real_values::<S, ScratchMatrix<'_, S>>(values, None, options.ordering())?;
        validate_real_output(values)?;
        return Ok(count);
    }

    let vectors = &mut scratch.eigenvectors;
    fill_identity(vectors);
    let count = match &mut scratch.solver {
        Solver::Bisection => selected::solve(
            scratch.diagonal,
            scratch.off_diagonal,
            scratch.values,
            Some(&mut *vectors),
            scratch.vector,
            scratch.work,
            options,
            context.execution(),
            solve,
        )?,
        Solver::DivideConquer(divide_conquer) => {
            divide_conquer.solve(
                scratch.values,
                scratch.off_diagonal,
                vectors,
                options,
                context,
                solve,
                gemm,
            )?;
            order
        }
        Solver::Ql => {
            tridiagonal_ql(
                scratch.values,
                scratch.off_diagonal,
                Some(QlVectors::new(&mut *vectors, scratch.rotations, order)?),
                options,
                context.execution(),
                solve,
            )?;
            order
        }
    };
    let values = &mut scratch.values[..count];
    rescale_real_values(values, scale)?;
    order_real_values(values, Some(&mut *vectors), options.ordering())?;
    crate::decompose::util::transpose_square_in_place(vectors);
    // Back-transform only the `count` selected columns: O(n² count), not O(n³).
    compact_columns(vectors, count)?;
    apply_phase_rows(vectors, scratch.phases);
    let blocks = match &mut scratch.reduction {
        Reduction::Blocked(blocks) => Some(blocks.lend(gemm, policy, context)),
        Reduction::Unblocked => None,
    };
    super::hessenberg::apply_q(
        &scratch.packed,
        scratch.beta,
        vectors.try_view_mut()?,
        blocks,
    )?;
    normalize_phases(vectors);
    validate_real_output(values)?;
    validate_scalar_output(vectors.as_slice())?;
    Ok(count)
}

/// Keep the leading `count` columns of a square candidate.
fn compact_columns<S: NumericScalar>(
    vectors: &mut ScratchMatrix<'_, S>,
    count: usize,
) -> Result<(), SciOxideError> {
    let order = vectors.rows();
    if count == order {
        return Ok(());
    }
    for row in 0..order {
        vectors
            .as_mut_slice()
            .copy_within(row * order..row * order + count, row * count);
    }
    vectors.try_resize_flat(order, count)
}

struct TridiagonalScratch<'a, R: RealScalar + NumericScalar<Real = R>> {
    eigenvectors: ScratchMatrix<'a, R>,
    transposed: &'a mut [R],
    pivots: &'a mut [R],
    rhs: &'a mut [R],
    solver: Solver<divide_conquer::Workspace<'a, R>>,
    values: &'a mut [R],
    off_diagonal: &'a mut [R],
    original: &'a mut [R],
    rotations: &'a mut [R],
}

impl<'a, R: RealScalar + NumericScalar<Real = R>> TridiagonalScratch<'a, R> {
    fn new(
        layout: TridiagonalLayout,
        scalars: &'a mut [R],
        reals: &'a mut [R],
        indices: &'a mut [usize],
    ) -> Result<Self, SciOxideError> {
        let n = layout.order;
        let selected = match layout.solver {
            Solver::Bisection => n,
            Solver::Ql | Solver::DivideConquer(()) => 0,
        };
        let vector_rows = if layout.job == Job::Vectors {
            layout.count
        } else {
            0
        };
        let vector_columns = if layout.job == Job::Vectors { n } else { 0 };
        let vector_len = vector_rows * vector_columns;
        let (eigenvectors, scalars) = crate::workspace::split_mut(scalars, vector_len)?;
        let transpose_len = if layout.solver == Solver::Bisection {
            vector_len
        } else {
            0
        };
        let (transposed, scalars) = crate::workspace::split_mut(scalars, transpose_len)?;
        let selected_vectors = if layout.job == Job::Vectors {
            selected
        } else {
            0
        };
        let (pivots, scalars) = crate::workspace::split_mut(scalars, selected_vectors)?;
        let (rhs, scalars) = crate::workspace::split_mut(scalars, selected_vectors)?;
        let (values, reals) = crate::workspace::split_mut(reals, n)?;
        let (off_diagonal, reals) = crate::workspace::split_mut(reals, n)?;
        let (original, reals) = crate::workspace::split_mut(reals, selected)?;
        let rotation_len = if layout.job == Job::Vectors && layout.solver == Solver::Ql {
            2 * n
        } else {
            0
        };
        let (rotations, reals) = crate::workspace::split_mut(reals, rotation_len)?;
        let solver = layout
            .solver
            .try_map(|| divide_conquer::Workspace::new(scalars, reals, indices, n))?;
        Ok(Self {
            eigenvectors: ScratchMatrix::new(eigenvectors, vector_rows, vector_columns)?,
            transposed,
            pivots,
            rhs,
            solver,
            values,
            off_diagonal,
            original,
            rotations,
        })
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
fn tridiagonal_candidate<R, T>(
    diagonal: &[R],
    off_diagonal: &[R],
    scale: R,
    job: Job,
    options: &HermitianEigenOptions<R>,
    workspace: &mut LinalgWorkspace<'_, R>,
    context: KernelContext<'_>,
    solve: &mut SolveContext<'_, '_>,
    consume: impl FnOnce(Candidate<'_, '_, '_, R>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError>
where
    R: RealScalar + NumericScalar<Real = R>,
{
    let order = diagonal.len();
    let layout = TridiagonalLayout::new(order, job, options)?;
    workspace.with_direct(|scalars, reals, indices, gemm| {
        let mut scratch = TridiagonalScratch::new(layout, scalars, reals, indices)?;
        scratch.values.copy_from_slice(diagonal);
        normalize_slice(scratch.values, scale);
        scratch.off_diagonal.fill(R::ZERO);
        scratch.off_diagonal[..off_diagonal.len()].copy_from_slice(off_diagonal);
        normalize_slice(&mut scratch.off_diagonal[..off_diagonal.len()], scale);
        if job == Job::Values {
            let count = if options.selects() {
                scratch.original.copy_from_slice(scratch.values);
                selected::solve::<R, ScratchMatrix<'_, R>>(
                    scratch.original,
                    scratch.off_diagonal,
                    scratch.values,
                    None,
                    scratch.pivots,
                    scratch.rhs,
                    options,
                    context.execution(),
                    solve,
                )?
            } else {
                values::solve(
                    scratch.values,
                    scratch.off_diagonal,
                    options,
                    context.execution(),
                    solve,
                )?;
                order
            };
            let values = &mut scratch.values[..count];
            rescale_real_values(values, scale)?;
            order_real_values::<R, ScratchMatrix<'_, R>>(values, None, options.ordering())?;
            validate_real_output(values)?;
            let _ = solve.counts().total()?;
            return consume(Candidate {
                values,
                vectors: &mut scratch.eigenvectors,
                gemm,
                projection: None,
            });
        }
        let vectors = &mut scratch.eigenvectors;
        if layout.solver != Solver::Bisection {
            fill_identity(vectors);
        }
        let count = match &mut scratch.solver {
            Solver::Bisection => {
                scratch.original.copy_from_slice(scratch.values);
                selected::solve(
                    scratch.original,
                    scratch.off_diagonal,
                    scratch.values,
                    Some(&mut *vectors),
                    scratch.pivots,
                    scratch.rhs,
                    options,
                    context.execution(),
                    solve,
                )?
            }
            Solver::DivideConquer(divide_conquer) => {
                divide_conquer.solve(
                    scratch.values,
                    scratch.off_diagonal,
                    vectors,
                    options,
                    context,
                    solve,
                    gemm,
                )?;
                order
            }
            Solver::Ql => {
                tridiagonal_ql(
                    scratch.values,
                    scratch.off_diagonal,
                    Some(QlVectors::new(&mut *vectors, scratch.rotations, order)?),
                    options,
                    context.execution(),
                    solve,
                )?;
                order
            }
        };
        let values = &mut scratch.values[..count];
        rescale_real_values(values, scale)?;
        order_real_values(values, Some(&mut *vectors), options.ordering())?;
        if layout.solver == Solver::Bisection {
            for (column, row) in vectors.as_slice().chunks_exact(order.max(1)).enumerate() {
                for (index, &value) in row.iter().enumerate() {
                    scratch.transposed[index * count + column] = value;
                }
            }
            *vectors = ScratchMatrix::new(scratch.transposed, order, count)?;
        } else {
            crate::decompose::util::transpose_square_in_place(vectors);
        }
        normalize_phases(vectors);
        validate_real_output(values)?;
        validate_scalar_output(vectors.as_slice())?;
        let _ = solve.counts().total()?;
        consume(Candidate {
            values,
            vectors,
            gemm,
            projection: None,
        })
    })
}

/// Check dense geometry and reserve scratch; interpretation stages and validates
/// the selected entries before any numerical reduction.
fn prepare_dense<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    interpretation: HermitianInput<S::Real>,
    job: Job,
    options: &HermitianEigenOptions<S::Real>,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<Option<HermitianProjectionReport<S::Real>>, SciOxideError> {
    context.execution().check()?;
    let order = matrix.rows();
    options.resolve_indices(order)?;
    if order != matrix.cols() {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::two(order, order),
            Dimensions::two(matrix.rows(), matrix.cols()),
        ));
    }
    let requirements = dense_requirements::<S>(
        &HermitianEigenWorkspaceRequest::new(order, options).for_kernels(context),
        job,
    )?;
    workspace.try_reserve(&requirements)?;
    let projection = stage_dense_input(matrix, interpretation, workspace)?;
    Ok(projection)
}

fn dense_identity(order: usize) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(0),
        scioxide_core::algorithm_id!("linalg", "hermitian_eigen", 1),
        order as u128,
        0,
    )
}

impl<R: RealScalar> HermitianEigenvaluesBuffer<R> {
    /// Computes the eigenvalues of a dense Hermitian matrix into `output`.
    ///
    /// This is the reusable form of [`HermitianEigenvaluesResult::compute`],
    /// which describes the method. Size the workspace with a
    /// [`HermitianEigenWorkspaceRequest`] built by
    /// [`HermitianEigenWorkspaceRequest::new`]. The returned report counts the
    /// iterations, which are also recorded by [`LinalgWorkspace::last_attempt`].
    ///
    /// # Errors
    ///
    /// - The errors of [`HermitianEigenvaluesResult::compute`].
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    pub fn compute_into<'scope, 'm, 'a, S: NumericScalar<Real = R>>(
        matrix: impl Into<MatrixView<'m, S>>,
        interpretation: HermitianInput<S::Real>,
        options: &HermitianEigenOptions<R>,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<super::EigenReport, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(&options.execution_options())?;
        let matrix = matrix.into();
        let order = matrix.rows();
        let projection = prepare_dense(
            matrix,
            interpretation,
            Job::Values,
            options,
            workspace,
            scope.kernel(),
        )?;
        output.reserve(order)?;
        workspace.run_scope(
            scope,
            dense_identity(order),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, kernels, solve| {
                prepared_dense_candidate(
                    order,
                    projection,
                    Job::Values,
                    options,
                    workspace,
                    kernels,
                    solve,
                    |candidate| {
                        output.eigenvalues.clear();
                        output.eigenvalues.extend_from_slice(candidate.values);
                        output.projection = candidate.projection;
                        Ok(())
                    },
                )?;
                Ok(super::EigenReport {
                    counts: solve.counts(),
                })
            },
        )
    }
}

fn publish_vectors<S: NumericScalar>(
    output: &mut HermitianEigenDecomposition<S>,
    candidate: Candidate<'_, '_, '_, S>,
) -> Result<(), SciOxideError> {
    output
        .eigenvectors
        .try_resize_flat(candidate.vectors.rows(), candidate.vectors.cols())?;
    output
        .eigenvectors
        .as_mut_slice()
        .copy_from_slice(candidate.vectors.as_slice());
    output.eigenvalues.clear();
    output.eigenvalues.extend_from_slice(candidate.values);
    output.projection = candidate.projection;
    Ok(())
}

impl<S: NumericScalar> HermitianEigenDecomposition<S> {
    /// Computes the eigenvalues and eigenvectors `A = V Λ Vᴴ` of a dense Hermitian matrix.
    ///
    /// `interpretation` defines `A`: the exact matrix, one triangle, or the
    /// projection `(A + Aᴴ) / 2` (see [`HermitianInput`]). The eigenvalues are
    /// real and sorted in ascending order; the index range and the ordering of
    /// [`HermitianEigenOptions`] apply through
    /// [`decompose_with_options`](Self::decompose_with_options). The
    /// eigenvectors are the orthonormal columns of `V` in the scalar type of
    /// `A`, with the conventions stated on [`HermitianEigenDecomposition`].
    ///
    /// `A` is scaled by its largest entry magnitude and reduced to real
    /// symmetric tridiagonal form by Householder similarity transformations.
    /// The tridiagonal problem is solved by implicit QL through order 32 and by
    /// divide and conquer above it. An index range or
    /// [`EigenvalueAccuracy::Relative`](crate::EigenvalueAccuracy::Relative)
    /// selects Sturm bisection and inverse iteration instead. The cost is
    /// O(n³) time and O(n²) memory, dominated by the reduction; for an index
    /// range of `k` eigenvalues, back-transforming their eigenvectors costs
    /// O(n² k). Use [`HermitianEigenvaluesResult::compute`]
    /// when only the eigenvalues are needed, and [`Self::decompose_into`] to
    /// reuse storage or run in parallel.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `matrix` is not square.
    /// - `NonFinite` if an entry read under `interpretation` is NaN or infinite.
    /// - `InvalidInput` if `interpretation` is [`HermitianInput::Exact`] and the
    ///   matrix is not exactly Hermitian, if a diagonal entry has a nonzero
    ///   imaginary part, or if a projection with a tolerance changes the matrix
    ///   by more than that tolerance.
    /// - `ConvergenceFailure` if the iteration budget is exhausted, or if inverse
    ///   iteration does not reach its residual bound.
    /// - `NumericalInstability` if inverse iteration loses its direction, or if
    ///   an eigenvalue overflows when scaled back to the input units.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{HermitianEigenDecomposition, Matrix};
    /// let a = Matrix::from_vec(2, 2, vec![2.0_f64, 1.0, 1.0, 2.0])?;
    /// let eigen = HermitianEigenDecomposition::decompose(
    ///     &a,
    ///     scioxide_linalg::HermitianInput::Exact)?;
    /// // The eigenvector of the largest eigenvalue, 3, is (1, 1) / √2.
    /// let v = eigen.eigenvectors();
    /// assert!((v.view().get(0, 1)? - v.view().get(1, 1)?).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
    ///   Hopkins, 2013, §8.3–8.4.
    /// - J. J. M. Cuppen, "A divide and conquer method for the symmetric
    ///   tridiagonal eigenproblem", *Numer. Math.* 36, 1981.
    /// - M. Gu and S. C. Eisenstat, "A divide-and-conquer algorithm for the
    ///   symmetric tridiagonal eigenproblem", *SIAM J. Matrix Anal. Appl.* 16,
    ///   1995.
    pub fn decompose<'m>(
        matrix: impl Into<MatrixView<'m, S>>,
        interpretation: HermitianInput<S::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::decompose_with_options(
            matrix,
            interpretation,
            &crate::HermitianEigenOptions::default(),
        )
    }

    /// Runs [`decompose`](Self::decompose) with `options`, which set the eigenvalue selection,
    /// ordering and accuracy, the deflation tolerance and the execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`decompose`](Self::decompose), and:
    ///
    /// - `InvalidInput` if the index range of `options` ends beyond the order.
    pub fn decompose_with_options<'m>(
        matrix: impl Into<MatrixView<'m, S>>,
        interpretation: HermitianInput<S::Real>,
        options: &HermitianEigenOptions<S::Real>,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::decompose_into(
            matrix,
            interpretation,
            &options.with_execution_options(
                options
                    .execution_options()
                    .with_execution_policy(ExecutionPolicy::Sequential),
            ),
            &mut output,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Computes the eigendecomposition of a dense Hermitian matrix into `output`.
    ///
    /// This is the reusable form of [`Self::decompose`], which describes the
    /// method and conventions. Size the workspace with
    /// [`HermitianEigenWorkspaceRequest::new`]. The returned report counts the
    /// iterations, which are also recorded by [`LinalgWorkspace::last_attempt`].
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::decompose`].
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    pub fn decompose_into<'scope, 'm, 'a>(
        matrix: impl Into<MatrixView<'m, S>>,
        interpretation: HermitianInput<S::Real>,
        options: &HermitianEigenOptions<S::Real>,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: impl Into<LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<super::EigenReport, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(&options.execution_options())?;
        let matrix = matrix.into();
        let order = matrix.rows();
        let projection = prepare_dense(
            matrix,
            interpretation,
            Job::Vectors,
            options,
            workspace,
            scope.kernel(),
        )?;
        output.reserve(order, options.resolve_indices(order)?.len())?;
        let report = workspace.run_scope(
            scope,
            dense_identity(order),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, kernels, solve| {
                prepared_dense_candidate(
                    order,
                    projection,
                    Job::Vectors,
                    options,
                    workspace,
                    kernels,
                    solve,
                    |candidate| publish_vectors(output, candidate),
                )?;
                Ok(super::EigenReport {
                    counts: solve.counts(),
                })
            },
        )?;
        output.report = Some(report);
        Ok(report)
    }
}

fn prepare_tridiagonal<R: RealScalar + NumericScalar<Real = R>>(
    diagonal: &[R],
    off_diagonal: &[R],
    job: Job,
    options: &HermitianEigenOptions<R>,
    workspace: &mut LinalgWorkspace<'_, R>,
    kernels: KernelContext<'_>,
) -> Result<R, SciOxideError> {
    kernels.execution().check()?;
    let scale = validate_tridiagonal(diagonal, off_diagonal, options)?;
    let requirements = tridiagonal_requirements::<R>(
        &HermitianEigenWorkspaceRequest::tridiagonal(diagonal.len(), options).for_kernels(kernels),
        job,
    )?;
    workspace.try_reserve(&requirements)?;
    Ok(scale)
}

/// Validate shared tridiagonal inputs and determine their normalization scale.
fn validate_tridiagonal<R: RealScalar>(
    diagonal: &[R],
    off_diagonal: &[R],
    options: &HermitianEigenOptions<R>,
) -> Result<R, SciOxideError> {
    let order = diagonal.len();
    let expected_off_diagonal = order.saturating_sub(1);
    if off_diagonal.len() != expected_off_diagonal {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(expected_off_diagonal),
            Dimensions::one(off_diagonal.len()),
        ));
    }
    crate::checked::validate_values(
        diagonal,
        NonFiniteOrigin::Input,
        "tridiagonal diagonal must be finite",
    )?;
    crate::checked::validate_values(
        off_diagonal,
        NonFiniteOrigin::Input,
        "tridiagonal off-diagonal must be finite",
    )?;
    options.resolve_indices(order)?;
    // A singleton is already diagonal: avoid rounding its exact eigenvalue
    // through a normalization and rescaling round trip.
    if order == 1 {
        return Ok(R::ONE);
    }
    let maximum = diagonal
        .iter()
        .chain(off_diagonal)
        .fold(R::ZERO, |scale, value| scale.max(value.abs()));
    if options.selects() {
        selection_scale(maximum)
    } else {
        Ok(hermitian_reduction_scale(maximum))
    }
}

/// Power-of-two scale for bisection and inverse iteration.
///
/// Dividing a graded matrix by its largest entry would flush its smallest
/// entries, and with them the relative accuracy of its smallest eigenvalues.
/// The scale lifts a matrix below one to `[1, 2)`, lowers one whose Gershgorin
/// bounds and residuals could overflow below `2^(E−3)`, with `E` the binary
/// exponent of the largest finite number in the `frexp` convention, and
/// otherwise leaves the entries unchanged. Both changes are exact.
fn selection_scale<R: RealScalar>(maximum: R) -> Result<R, SciOxideError> {
    if maximum == R::ZERO {
        return Ok(R::ONE);
    }
    let exponent = crate::kernel::standard::exponent(maximum)?;
    let headroom = crate::kernel::standard::exponent(R::MAX)? - 3;
    Ok(if exponent <= 0 {
        R::ONE.scalbn(exponent - 1)
    } else if exponent > headroom {
        R::ONE.scalbn(exponent - headroom)
    } else {
        R::ONE
    })
}

impl<R: RealScalar + NumericScalar<Real = R>> HermitianEigenDecomposition<R> {
    /// Computes the eigenvalues and eigenvectors of a real symmetric tridiagonal matrix.
    ///
    /// `diagonal` holds the `n` diagonal entries and `off_diagonal` the `n − 1`
    /// entries of the sub- and superdiagonal; the matrix is never formed
    /// densely. Eigenvectors are the columns of
    /// [`eigenvectors`](Self::eigenvectors), with the ordering and sign
    /// conventions of [`HermitianEigenDecomposition`]. A 1×1 matrix returns its
    /// diagonal entry exactly with eigenvector `[1]`.
    ///
    /// The complete spectrum uses implicit QL with Wilkinson shifts through
    /// order 32 and divide and conquer above it. An index range or
    /// [`EigenvalueAccuracy::Relative`](crate::EigenvalueAccuracy::Relative)
    /// uses Sturm bisection and inverse iteration, with O(n·k) storage for `k`
    /// selected eigenvectors. Use [`Self::tridiagonal_into`] to reuse storage.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `off_diagonal.len()` is not
    ///   `diagonal.len().saturating_sub(1)`.
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - `ConvergenceFailure` if the iteration budget is exhausted, or if inverse
    ///   iteration does not reach its residual bound.
    /// - `NumericalInstability` if inverse iteration loses its direction, or if
    ///   an eigenvalue overflows when scaled back to the input units.
    ///
    /// # Examples
    ///
    /// ```
    /// // The eigenvalues of [[2, 1], [1, 2]] are 1 and 3.
    /// use scioxide_linalg::HermitianEigenDecomposition;
    /// let eigen = HermitianEigenDecomposition::tridiagonal(
    ///     &[2.0_f64, 2.0],
    ///     &[1.0])?;
    /// assert!((eigen.eigenvalues()[0] - 1.0).abs() < 1e-14);
    /// assert!((eigen.eigenvalues()[1] - 3.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
    ///   Hopkins, 2013, §8.3–8.4.
    /// - M. Gu and S. C. Eisenstat, "A divide-and-conquer algorithm for the
    ///   symmetric tridiagonal eigenproblem", *SIAM J. Matrix Anal. Appl.* 16,
    ///   1995.
    pub fn tridiagonal(diagonal: &[R], off_diagonal: &[R]) -> Result<Self, SciOxideError> {
        Self::tridiagonal_with_options(
            diagonal,
            off_diagonal,
            &crate::HermitianEigenOptions::default(),
        )
    }

    /// Runs [`tridiagonal`](Self::tridiagonal) with `options`, which set the eigenvalue selection,
    /// ordering and accuracy, the deflation tolerance and the execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`tridiagonal`](Self::tridiagonal), and:
    ///
    /// - `InvalidInput` if the index range of `options` ends beyond the order.
    pub fn tridiagonal_with_options(
        diagonal: &[R],
        off_diagonal: &[R],
        options: &HermitianEigenOptions<R>,
    ) -> Result<Self, SciOxideError> {
        let mut output = Self::empty();
        Self::tridiagonal_into(
            diagonal,
            off_diagonal,
            &options.with_execution_options(
                options
                    .execution_options()
                    .with_execution_policy(ExecutionPolicy::Sequential),
            ),
            &mut output,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }

    /// Computes the eigendecomposition of a real symmetric tridiagonal matrix into `output`.
    ///
    /// This is the reusable form of [`Self::tridiagonal`], which describes the
    /// method and conventions. Size the workspace with
    /// [`HermitianEigenWorkspaceRequest::tridiagonal`]. Iterations count toward
    /// the iteration budget of
    /// [`LinalgExecutionOptions`](crate::LinalgExecutionOptions) and are
    /// reported in `counts().nested()` of a calling [`SolveContext`].
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::tridiagonal`].
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    pub fn tridiagonal_into<'scope, 'a>(
        diagonal: &[R],
        off_diagonal: &[R],
        options: &HermitianEigenOptions<R>,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, R>,
        context: impl Into<LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<super::EigenReport, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(&options.execution_options())?;
        let kernels = scope.kernel();
        let order = diagonal.len();
        let scale = prepare_tridiagonal(
            diagonal,
            off_diagonal,
            Job::Vectors,
            options,
            workspace,
            kernels,
        )?;
        output.reserve(order, options.resolve_indices(order)?.len())?;
        let report = workspace.run_scope(
            scope,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "tridiagonal_eigen", 1),
                order as u128,
                0,
            ),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, kernels, solve| {
                tridiagonal_candidate(
                    diagonal,
                    off_diagonal,
                    scale,
                    Job::Vectors,
                    options,
                    workspace,
                    kernels,
                    solve,
                    |candidate| publish_vectors(output, candidate),
                )?;
                Ok(super::EigenReport {
                    counts: solve.counts(),
                })
            },
        )?;
        output.report = Some(report);
        Ok(report)
    }
}
impl<R: RealScalar + NumericScalar<Real = R>> HermitianEigenvaluesBuffer<R> {
    /// Computes the eigenvalues of a real symmetric tridiagonal matrix into `output`.
    ///
    /// This is the reusable form of [`HermitianEigenvaluesResult::tridiagonal`],
    /// which describes the method. Size the workspace with
    /// [`HermitianEigenWorkspaceRequest::tridiagonal`]. Iterations count toward
    /// the iteration budget of
    /// [`LinalgExecutionOptions`](crate::LinalgExecutionOptions) and are
    /// reported in `counts().nested()` of a calling [`SolveContext`].
    ///
    /// # Errors
    ///
    /// - The errors of [`HermitianEigenvaluesResult::tridiagonal`].
    /// - `ExecutionStopped` if the context is cancelled or its deadline passes.
    /// - Workspace errors as described in
    ///   [Plans and workspaces](scioxide_core#plans-and-workspaces).
    ///
    /// # Examples
    ///
    /// Run `cargo run -p scioxide-linalg --example workspace_operations` for a complete caller-owned storage example.
    pub fn tridiagonal_into<'scope, 'a>(
        diagonal: &[R],
        off_diagonal: &[R],
        options: &HermitianEigenOptions<R>,
        output: &mut Self,
        workspace: &mut LinalgWorkspace<'_, R>,
        context: impl Into<LinalgExecutionContext<'scope, 'a>>,
    ) -> Result<super::EigenReport, SciOxideError> {
        workspace.begin_attempt();
        let scope = context.into().with_options(&options.execution_options())?;
        let kernels = scope.kernel();
        let order = diagonal.len();
        let scale = prepare_tridiagonal(
            diagonal,
            off_diagonal,
            Job::Values,
            options,
            workspace,
            kernels,
        )?;
        output.reserve(order)?;
        workspace.run_scope(
            scope,
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(0),
                scioxide_core::algorithm_id!("linalg", "tridiagonal_eigen", 1),
                order as u128,
                0,
            ),
            scioxide_core::SolveBudgets::unbounded(),
            |workspace, kernels, solve| {
                tridiagonal_candidate(
                    diagonal,
                    off_diagonal,
                    scale,
                    Job::Values,
                    options,
                    workspace,
                    kernels,
                    solve,
                    |candidate| {
                        output.eigenvalues.clear();
                        output.eigenvalues.extend_from_slice(candidate.values);
                        output.projection = None;
                        Ok(())
                    },
                )?;
                Ok(super::EigenReport {
                    counts: solve.counts(),
                })
            },
        )
    }
}

fn rescale_real_values<R: RealScalar>(values: &mut [R], scale: R) -> Result<(), SciOxideError> {
    if scale == R::ONE {
        return Ok(());
    }
    // Finite normalized values and a finite scale can only overflow.
    values.iter_mut().try_for_each(|value| {
        *value = crate::checked::finite_candidate(*value * scale)?;
        Ok(())
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "the numerical operation keeps its operands, storage and control parameters explicit"
)]
fn reduce_to_tridiagonal<S>(
    matrix: &mut ScratchMatrix<'_, S>,
    diagonal: &mut [S::Real],
    off_diagonal: &mut [S::Real],
    beta: &mut [S],
    phases: &mut [S],
    vector: &mut [S],
    work: &mut [S],
    reduction: &mut Reduction<blocked::Scratch<'_, S>>,
    gemm: &mut LinalgWorkspace<'_, S>,
    policy: ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError>
where
    S: NumericScalar,
{
    // Unblocked Householder tridiagonalization (lower triangle): for
    // each column, `p = beta A v`, `w = p - (beta/2)(v^H p) v`, and the
    // Hermitian rank-2 update `A -= v w^H + w v^H` is applied to the lower
    // triangle only (the upper triangle is never read). Reflector vectors
    // are stored below the subdiagonal for back-transformation.
    let order = matrix.rows();
    phases.fill(<S as scioxide_core::AlgebraicScalar>::ONE);
    if let Reduction::Blocked(blocks) = reduction {
        blocked::reduce(matrix, beta, vector, work, blocks, gemm, policy, context)?;
    } else {
        for (step, beta_step_slot) in beta.iter_mut().enumerate() {
            let start = step + 1;
            let len = order - start;
            vector[..len]
                .iter_mut()
                .enumerate()
                .for_each(|(index, slot)| *slot = matrix.get_element(start + index, step));
            let h =
                crate::reflector::generate(crate::VectorViewMut::from_slice(&mut vector[..len]))?;
            let beta_step = h.tau;
            *beta_step_slot = beta_step;
            if beta_step != S::ZERO {
                let beta_scalar = beta_step;
                work[..len].fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
                for row in 0..len {
                    let diagonal_value = matrix.get_element(start + row, start + row);
                    work[row] = work[row] + diagonal_value * vector[row];
                    for col in 0..row {
                        let value = matrix.get_element(start + row, start + col);
                        work[row] = work[row] + value * vector[col];
                        work[col] = work[col] + value.conj() * vector[row];
                    }
                }
                for value in work.iter_mut().take(len) {
                    *value = *value * beta_scalar;
                }
                let dot = (0..len)
                    .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |sum, index| {
                        sum + vector[index].conj() * work[index]
                    });
                let correction = dot * (-beta_step.conj() * S::from_real(S::Real::HALF));
                for index in 0..len {
                    work[index] = work[index] + correction * vector[index];
                }
                for row in 0..len {
                    for col in 0..=row {
                        let value = matrix.get_element(start + row, start + col)
                            - vector[row] * work[col].conj()
                            - work[row] * vector[col].conj();
                        matrix.as_mut_slice()[(start + row) * order + start + col] = value;
                    }
                }
            }
            matrix.as_mut_slice()[start * order + step] = S::from_real(h.beta);
            for (offset, &value) in vector.iter().enumerate().skip(1).take(len - 1) {
                matrix.as_mut_slice()[(start + offset) * order + step] = value;
            }
        }
    }

    // The complex tridiagonal T is made real symmetric by a diagonal unitary
    // similarity `D^H T D` with `d_{k+1} = d_k * phase(t_{k+1,k})`, so the
    // real QL iteration can be used. The phases are kept and applied to the
    // eigenvector rows afterwards.
    diagonal.iter_mut().enumerate().for_each(|(index, slot)| {
        *slot = matrix.get_element(index, index).real_component();
    });
    if order > 1 {
        for index in 0..(order - 1) {
            let sub = matrix.get_element(index + 1, index);
            let magnitude = sub.abs();
            off_diagonal[index] = magnitude;
            phases[index + 1] = if magnitude > S::Real::ZERO {
                phases[index] * sub / S::from_real(magnitude)
            } else {
                phases[index]
            };
        }
    }
    if let Some(last) = off_diagonal.last_mut() {
        *last = S::Real::ZERO;
    }
    Ok(())
}

fn apply_phase_rows<S: NumericScalar>(matrix: &mut impl MatrixStorage<S>, phases: &[S]) {
    let columns = matrix.cols();
    if columns == 0 {
        return;
    }
    matrix
        .as_mut_slice()
        .chunks_mut(columns)
        .zip(phases)
        .for_each(|(row, phase)| row.iter_mut().for_each(|value| *value = *phase * *value));
}

/// A QL sweep records adjacent rotations before applying them to transposed vectors.
struct QlVectors<'a, R, M> {
    matrix: &'a mut M,
    cosines: &'a mut [R],
    sines: &'a mut [R],
}

impl<'a, R, M> QlVectors<'a, R, M> {
    fn new(matrix: &'a mut M, scratch: &'a mut [R], order: usize) -> Result<Self, SciOxideError> {
        let (cosines, sines) = crate::workspace::split_mut(scratch, order)?;
        Ok(Self {
            matrix,
            cosines,
            sines,
        })
    }
}

fn tridiagonal_ql<R: RealScalar, S: NumericScalar<Real = R>, M: MatrixStorage<S>>(
    diagonal: &mut [R],
    off_diagonal: &mut [R],
    mut vectors: Option<QlVectors<'_, R, M>>,
    options: &HermitianEigenOptions<R>,
    context: ExecutionContext<'_>,
    solve: &mut SolveContext<'_, '_>,
) -> Result<usize, SciOxideError> {
    // Implicit QL with Wilkinson-type shift on the real symmetric tridiagonal
    // (EISPACK `tql2` / Numerical Recipes `tqli`). For each `start`, find the
    // first negligible off-diagonal `end`; if the block is larger than 1x1,
    // run one shifted QL sweep from `end` back to `start`. Each rotation uses
    // a range-guarded Euclidean norm: ordinary lanes admit native squares;
    // extreme lanes retain the unsquared hypot path. Rotations accumulate into
    // the rows of `vectors` (eigenvectors transposed): rows are contiguous in
    // row-major storage. Full-vector jobs transpose once at the end, while
    // first-component jobs retain only column zero of the transposed vectors.
    let order = diagonal.len();
    if order < 2 {
        return Ok(0);
    }
    let budget = options.resolved_iteration_budget(order, false)?;
    let mut steps = 0_usize;
    for start in 0..order {
        loop {
            context.check()?;
            let mut end = start;
            while end + 1 < order {
                let scale = diagonal[end].abs() + diagonal[end + 1].abs();
                if off_diagonal[end].abs() <= options.deflation_tolerance().threshold(scale)? {
                    break;
                }
                end += 1;
            }
            if end == start {
                break;
            }
            if steps >= budget {
                return Err(SciOxideError::convergence_failure(
                    ConvergenceReason::IterationLimit,
                    ConvergenceDiagnostics::new().with_counts(
                        (scioxide_core::EvaluationCounts::from_iterations(steps)).into(),
                    ),
                    "Hermitian tridiagonal iteration exhausted its budget",
                ));
            }
            crate::solve_support::admit_iteration(
                Some(solve.execution()),
                Some(&mut *solve),
                "solve iteration budget exhausted",
            )?;
            steps += 1;
            let two = R::TWO;
            let mut g = (diagonal[start + 1] - diagonal[start]) / (two * off_diagonal[start]);
            let mut radius = crate::rotation::hypot(g, R::ONE);
            if g < R::ZERO {
                radius = -radius;
            }
            g = diagonal[end] - diagonal[start] + off_diagonal[start] / (g + radius);
            let mut sine = R::ONE;
            let mut cosine = R::ONE;
            let mut correction = R::ZERO;
            let mut recovered_underflow = false;
            let mut first_rotation = end;
            for index in (start..end).rev() {
                let f = sine * off_diagonal[index];
                let b = cosine * off_diagonal[index];
                radius = crate::rotation::hypot(f, g);
                off_diagonal[index + 1] = radius;
                if radius == R::ZERO {
                    // Full tqli recovery: do not form 0/0 or finish a sweep
                    // whose rotation vanished. Commit its pending correction
                    // and restart deflation at this start index.
                    diagonal[index + 1] -= correction;
                    off_diagonal[end] = R::ZERO;
                    recovered_underflow = true;
                    break;
                }
                sine = f / radius;
                cosine = g / radius;
                g = diagonal[index + 1] - correction;
                radius = (diagonal[index] - g) * sine + two * cosine * b;
                correction = sine * radius;
                diagonal[index + 1] = g + correction;
                g = cosine * radius - b;
                if let Some(q) = vectors.as_mut() {
                    crate::checked::finite_candidate(S::from_real(cosine))?;
                    crate::checked::finite_candidate(S::from_real(sine))?;
                    q.cosines[index] = cosine;
                    q.sines[index] = -sine;
                    first_rotation = index;
                }
            }
            if let Some(q) = vectors.as_mut() {
                // Orthogonal rows have bounded entries: plain arithmetic is safe.
                // Flush even an interrupted sweep before its underflow restart.
                let stride = q.matrix.cols();
                crate::rotation::apply_adjacent_rows(
                    q.matrix.as_mut_slice(),
                    stride,
                    &q.cosines[first_rotation..end],
                    &q.sines[first_rotation..end],
                    first_rotation,
                )?;
            }
            if recovered_underflow {
                continue;
            }
            diagonal[start] -= correction;
            off_diagonal[start] = g;
            off_diagonal[end] = R::ZERO;
        }
    }
    Ok(steps)
}

fn fill_identity<S: NumericScalar>(matrix: &mut impl MatrixStorage<S>) {
    let order = matrix.rows();
    matrix
        .as_mut_slice()
        .fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
    (0..order).for_each(|index| {
        matrix.as_mut_slice()[index * order + index] = <S as scioxide_core::AlgebraicScalar>::ONE;
    });
}

/// Sort transposed eigenvectors by swapping contiguous rows before back-transformation.
fn order_real_values<S: NumericScalar, M: MatrixStorage<S>>(
    values: &mut [S::Real],
    mut vectors: Option<&mut M>,
    ordering: EigenvalueOrder,
) -> Result<(), SciOxideError> {
    for destination in 0..values.len() {
        let selected = (destination..values.len())
            .min_by(|&left, &right| compare_real(values[left], values[right], ordering))
            .unwrap_or(destination);
        values.swap(destination, selected);
        if let Some(matrix) = vectors.as_deref_mut()
            && selected != destination
        {
            let stride = matrix.cols();
            let (before, after) =
                crate::workspace::split_mut(matrix.as_mut_slice(), selected * stride)?;
            before[destination * stride..(destination + 1) * stride]
                .swap_with_slice(&mut after[..stride]);
        }
    }

    Ok(())
}

fn compare_real<R: RealScalar>(left: R, right: R, ordering: EigenvalueOrder) -> Ordering {
    let ascending = left.partial_cmp(&right).unwrap_or(Ordering::Equal);
    match ordering {
        EigenvalueOrder::Ascending => ascending,
        EigenvalueOrder::Descending => ascending.reverse(),
    }
}

/// Normalize columns in cache-sized tiles, choosing the last largest-magnitude
/// component for the deterministic real/nonnegative phase convention.
fn normalize_phases<S: NumericScalar>(matrix: &mut impl MatrixStorage<S>) {
    let columns = matrix.cols();
    for first in (0..columns).step_by(32) {
        let width = (columns - first).min(32);
        let mut norms = [S::Real::ZERO; 32];
        let mut largest = [S::Real::ZERO; 32];
        let mut phases = [S::ZERO; 32];
        for row in matrix.as_slice().chunks_exact(columns) {
            for (j, &value) in row[first..first + width].iter().enumerate() {
                let square = value.norm_sqr();
                norms[j] += square;
                if square >= largest[j] {
                    largest[j] = square;
                    phases[j] = value;
                }
            }
        }
        for j in 0..width {
            let norm = norms[j].sqrt();
            norms[j] = if norm == S::Real::ZERO {
                S::Real::ONE
            } else {
                S::Real::ONE / norm
            };
            let magnitude = phases[j].abs();
            phases[j] = if magnitude == S::Real::ZERO {
                S::ONE
            } else {
                phases[j].conj() / S::from_real(magnitude)
            };
        }
        for row in matrix.as_mut_slice().chunks_exact_mut(columns) {
            for (j, value) in row[first..first + width].iter_mut().enumerate() {
                *value = (*value * S::from_real(norms[j])) * phases[j];
            }
        }
    }
}

fn validate_real_output<R: RealScalar>(values: &[R]) -> Result<(), SciOxideError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or_else(|| {
            SciOxideError::non_finite(NonFiniteOrigin::Output, "computed eigenvalue is not finite")
        })
}

fn validate_scalar_output<S: NumericScalar>(values: &[S]) -> Result<(), SciOxideError> {
    values
        .iter()
        .all(|value| value.has_finite_components())
        .then_some(())
        .ok_or_else(|| {
            SciOxideError::non_finite(
                NonFiniteOrigin::Output,
                "computed eigenvector is not finite",
            )
        })
}

impl<R: RealScalar + NumericScalar<Real = R>> HermitianEigenvaluesResult<R> {
    /// Computes the eigenvalues of a real symmetric tridiagonal matrix.
    ///
    /// `diagonal` holds the `n` diagonal entries and `off_diagonal` the `n − 1`
    /// entries of the sub- and superdiagonal; the matrix is never formed
    /// densely. The eigenvalues are sorted by [`HermitianEigenOptions::ordering`].
    ///
    /// The complete spectrum uses the square-root-free Pal–Walker–Kahan QL/QR
    /// iteration in O(n²) time. An index range or
    /// [`EigenvalueAccuracy::Relative`](crate::EigenvalueAccuracy::Relative)
    /// uses Sturm bisection to full working precision. Both need O(n) memory.
    /// Use [`HermitianEigenvaluesBuffer::tridiagonal_into`] to reuse storage.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `off_diagonal.len()` is not
    ///   `diagonal.len().saturating_sub(1)`.
    /// - `NonFinite` if an entry is NaN or infinite.
    /// - `ConvergenceFailure` if the iteration budget is exhausted.
    /// - `NumericalInstability` if an eigenvalue overflows when scaled back to
    ///   the input units.
    ///
    /// # Examples
    ///
    /// ```
    /// // The eigenvalues of [[2, 1], [1, 2]] are 1 and 3.
    /// use scioxide_linalg::HermitianEigenvaluesResult;
    /// let eigen = HermitianEigenvaluesResult::tridiagonal(
    ///     &[2.0_f64, 2.0],
    ///     &[1.0])?;
    /// assert!((eigen.eigenvalues()[0] - 1.0).abs() < 1e-14);
    /// assert!((eigen.eigenvalues()[1] - 3.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - B. N. Parlett, *The Symmetric Eigenvalue Problem*, SIAM, 1998.
    pub fn tridiagonal(diagonal: &[R], off_diagonal: &[R]) -> Result<Self, SciOxideError> {
        Self::tridiagonal_with_options(
            diagonal,
            off_diagonal,
            &crate::HermitianEigenOptions::default(),
        )
    }

    /// Runs [`tridiagonal`](Self::tridiagonal) with `options`, which set the eigenvalue selection,
    /// ordering and accuracy, the deflation tolerance and the execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`tridiagonal`](Self::tridiagonal), and:
    ///
    /// - `InvalidInput` if the index range of `options` ends beyond the order.
    pub fn tridiagonal_with_options(
        diagonal: &[R],
        off_diagonal: &[R],
        options: &HermitianEigenOptions<R>,
    ) -> Result<Self, SciOxideError> {
        let mut output = HermitianEigenvaluesBuffer::empty();
        let report = HermitianEigenvaluesBuffer::tridiagonal_into(
            diagonal,
            off_diagonal,
            &options.with_execution_options(
                options
                    .execution_options()
                    .with_execution_policy(ExecutionPolicy::Sequential),
            ),
            &mut output,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(Self { output, report })
    }
}

impl<R: RealScalar> HermitianEigenvaluesResult<R> {
    /// Computes the eigenvalues of a dense Hermitian matrix.
    ///
    /// `interpretation` defines the matrix `A`: the exact matrix, one
    /// triangle, or the projection `(A + Aᴴ) / 2` (see [`HermitianInput`]). The
    /// eigenvalues are real and sorted in ascending order; the index range and
    /// the ordering of [`HermitianEigenOptions`] apply through
    /// [`compute_with_options`](Self::compute_with_options).
    ///
    /// `A` is scaled by its largest entry magnitude and reduced to real
    /// symmetric tridiagonal form by Householder similarity transformations, in
    /// O(n³) time and O(n²) memory. The complete spectrum then uses the
    /// square-root-free Pal–Walker–Kahan QL/QR iteration. An index range or
    /// [`EigenvalueAccuracy::Relative`](crate::EigenvalueAccuracy::Relative)
    /// uses Sturm bisection instead. Use [`HermitianEigenDecomposition::decompose`]
    /// for eigenvectors.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `matrix` is not square.
    /// - `NonFinite` if an entry read under `interpretation` is NaN or infinite.
    /// - `InvalidInput` if `interpretation` is [`HermitianInput::Exact`] and the
    ///   matrix is not exactly Hermitian, if a diagonal entry has a nonzero
    ///   imaginary part, or if a projection with a tolerance changes the matrix
    ///   by more than that tolerance.
    /// - `ConvergenceFailure` if the iteration budget is exhausted.
    /// - `NumericalInstability` if an eigenvalue overflows when scaled back to
    ///   the input units.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_linalg::{HermitianEigenvaluesResult, Matrix};
    /// // The eigenvalues of [[2, 1], [1, 2]] are 1 and 3.
    /// let a = Matrix::from_vec(2, 2, vec![2.0_f64, 1.0, 1.0, 2.0])?;
    /// let values = HermitianEigenvaluesResult::compute(
    ///     &a,
    ///     scioxide_linalg::HermitianInput::Exact)?;
    /// assert!((values.eigenvalues()[0] - 1.0).abs() < 1e-14);
    /// assert!((values.eigenvalues()[1] - 3.0).abs() < 1e-14);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # References
    ///
    /// - G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
    ///   Hopkins, 2013, §8.3–8.4.
    /// - B. N. Parlett, *The Symmetric Eigenvalue Problem*, SIAM, 1998.
    pub fn compute<'m, S: NumericScalar<Real = R>>(
        matrix: impl Into<MatrixView<'m, S>>,
        interpretation: HermitianInput<S::Real>,
    ) -> Result<Self, SciOxideError> {
        Self::compute_with_options(
            matrix,
            interpretation,
            &crate::HermitianEigenOptions::default(),
        )
    }

    /// Runs [`compute`](Self::compute) with `options`, which set the eigenvalue selection, ordering
    /// and accuracy, the deflation tolerance and the execution limits.
    ///
    /// # Errors
    ///
    /// The errors of [`compute`](Self::compute), and:
    ///
    /// - `InvalidInput` if the index range of `options` ends beyond the order.
    pub fn compute_with_options<'m, S: NumericScalar<Real = R>>(
        matrix: impl Into<MatrixView<'m, S>>,
        interpretation: HermitianInput<S::Real>,
        options: &HermitianEigenOptions<R>,
    ) -> Result<Self, SciOxideError> {
        let mut output = HermitianEigenvaluesBuffer::empty();
        let report = HermitianEigenvaluesBuffer::compute_into(
            matrix,
            interpretation,
            &options.with_execution_options(
                options
                    .execution_options()
                    .with_execution_policy(ExecutionPolicy::Sequential),
            ),
            &mut output,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential_native(),
        )?;
        Ok(Self { output, report })
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::Matrix;
    use scioxide_core::Complex;

    /// Run `work` inside a fresh unbounded solve epoch.
    pub(in crate::eigen) fn with_solve<T>(
        work: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
    ) -> Result<T, SciOxideError> {
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(0),
            scioxide_core::algorithm_id!("test", "hermitian", 1),
            0,
            0,
        );
        let mut state = scioxide_core::SolveState::new();
        let mut budgets = scioxide_core::SolveBudgets::unbounded();
        let mut solve =
            state.begin_solve(ExecutionContext::sequential(), identity, &mut budgets)?;
        work(&mut solve)
    }

    #[test]
    fn plans_depend_only_on_order_and_job() {
        let complete = HermitianEigenOptions::<f64>::default();
        let selected = complete.with_index_range(crate::EigenIndexRange::new(0, 1).unwrap());
        let divide_conquer = Solver::DivideConquer(());
        for (order, job, options, reduction, solver, tridiagonal) in [
            (
                32,
                Job::Vectors,
                complete,
                Reduction::Unblocked,
                Solver::Ql,
                Solver::Ql,
            ),
            (
                33,
                Job::Vectors,
                complete,
                Reduction::Unblocked,
                divide_conquer,
                divide_conquer,
            ),
            (
                64,
                Job::Vectors,
                complete,
                Reduction::Unblocked,
                divide_conquer,
                divide_conquer,
            ),
            (
                65,
                Job::Vectors,
                complete,
                Reduction::Blocked(()),
                divide_conquer,
                divide_conquer,
            ),
            (
                65,
                Job::Values,
                complete,
                Reduction::Blocked(()),
                Solver::Ql,
                divide_conquer,
            ),
            (
                65,
                Job::Vectors,
                selected,
                Reduction::Blocked(()),
                Solver::Bisection,
                Solver::Bisection,
            ),
            (
                64,
                Job::Values,
                selected,
                Reduction::Unblocked,
                Solver::Bisection,
                Solver::Bisection,
            ),
        ] {
            let layout = DenseLayout::new(order, job, &options);
            assert_eq!((layout.reduction, layout.solver), (reduction, solver));
            assert_eq!(
                TridiagonalLayout::new(order, Job::Vectors, &options)
                    .unwrap()
                    .solver,
                tridiagonal
            );
        }
    }

    #[test]
    fn blocked_reduction_and_wy_vectors_cross_panel_boundaries() {
        let n = 67;
        for scale in [1.0e-200, 1.0, 1.0e200] {
            let matrix = Matrix::from_vec(
                n,
                n,
                (0..n * n)
                    .map(|index| {
                        let row = index / n;
                        let col = index % n;
                        let real = if row == col {
                            70.0 + row as f64 * 0.1
                        } else {
                            ((row + col) % 17) as f64 * 0.01
                        };
                        Complex::new(real * scale, (row as f64 - col as f64) * 0.001 * scale)
                    })
                    .collect(),
            )
            .unwrap();
            let mut decomposition = HermitianEigenDecomposition::empty();
            HermitianEigenDecomposition::decompose_into(
                &matrix,
                crate::HermitianInput::Exact,
                &HermitianEigenOptions::default(),
                &mut decomposition,
                &mut LinalgWorkspace::new(),
                crate::kernel::tests::test_context(),
            )
            .unwrap();
            let vectors = decomposition.eigenvectors();
            let mut residual = 0.0_f64;
            let mut orthogonality = 0.0_f64;
            for row in 0..n {
                for col in 0..n {
                    let product = (0..n).fold(Complex::new(0.0, 0.0), |sum, inner| {
                        sum + matrix.get_element(row, inner) / scale
                            * vectors.get_element(inner, col)
                    });
                    let difference = product
                        - vectors.get_element(row, col)
                            * (decomposition.eigenvalues()[col] / scale);
                    residual = residual.max(difference.norm());
                    let gram = (0..n).fold(Complex::new(0.0, 0.0), |sum, inner| {
                        sum + vectors.get_element(inner, row).conj()
                            * vectors.get_element(inner, col)
                    });
                    orthogonality = orthogonality
                        .max((gram - Complex::new(if row == col { 1.0 } else { 0.0 }, 0.0)).norm());
                }
            }
            assert!(
                residual / (77.0 * n as f64 * f64::EPSILON) < 100.0,
                "residual {residual}"
            );
            assert!(
                orthogonality / (n as f64 * f64::EPSILON) < 100.0,
                "orthogonality {orthogonality}"
            );
        }
    }

    #[test]
    fn tridiagonal_subnormal_couplings_remain_finite() {
        let tiny = f64::from_bits(1);
        for diagonal in [[0.0, tiny, 2.0 * tiny], [1.0, 1.0, 2.0]] {
            let mut values = diagonal;
            let mut off = [tiny, tiny, 0.0];
            let mut vectors = Matrix::<f64>::identity(3).unwrap();
            with_solve(|solve| {
                tridiagonal_ql(
                    &mut values,
                    &mut off,
                    Some(QlVectors::new(&mut vectors, &mut [0.0; 6], 3).unwrap()),
                    &HermitianEigenOptions::default(),
                    ExecutionContext::sequential(),
                    solve,
                )
            })
            .unwrap();
            assert!(
                values
                    .iter()
                    .chain(vectors.as_slice())
                    .all(|value| value.is_finite())
            );
        }
    }

    #[test]
    fn real_symmetric_values_and_vectors_agree() {
        let matrix = Matrix::from_vec(
            3,
            3,
            vec![4.0_f64, 1.0, -2.0, 1.0, 2.0, 0.0, -2.0, 0.0, 3.0],
        )
        .unwrap();
        let values =
            HermitianEigenvaluesResult::<f64>::compute(&matrix, crate::HermitianInput::Exact)
                .unwrap();
        let decomposition =
            HermitianEigenDecomposition::decompose(&matrix, crate::HermitianInput::Exact).unwrap();
        assert_eq!(values.eigenvalues().len(), 3);
        values
            .eigenvalues()
            .iter()
            .zip(decomposition.eigenvalues())
            .for_each(|(left, right)| {
                assert!(
                    (left - right).abs() < 1.0e-10,
                    "values-only {left:?} differs from vector job {right:?}"
                );
            });
        assert_residual(&matrix, &decomposition, 1.0e-10);
    }

    #[test]
    fn complex_hermitian_vectors_have_small_residual() {
        let i = Complex::new(0.0_f64, 1.0);
        let matrix = Matrix::from_vec(
            3,
            3,
            vec![
                Complex::new(3.0, 0.0),
                i,
                Complex::new(2.0, -1.0),
                -i,
                Complex::new(4.0, 0.0),
                Complex::new(-1.0, 2.0),
                Complex::new(2.0, 1.0),
                Complex::new(-1.0, -2.0),
                Complex::new(2.0, 0.0),
            ],
        )
        .unwrap();
        let decomposition =
            HermitianEigenDecomposition::decompose(&matrix, crate::HermitianInput::Exact).unwrap();
        assert_residual(&matrix, &decomposition, 1.0e-10);
    }

    #[test]
    fn rejects_non_hermitian_without_changing_output() {
        let matrix = Matrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let mut result = HermitianEigenvaluesBuffer {
            eigenvalues: vec![9.0],
            projection: None,
        };
        let error = HermitianEigenvaluesBuffer::compute_into(
            &matrix,
            crate::HermitianInput::Exact,
            &HermitianEigenOptions::default(),
            &mut result,
            &mut LinalgWorkspace::new(),
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert!(matches!(
            (error).view(),
            scioxide_core::SciOxideErrorView::InvalidInput { .. }
        ));
        assert_eq!(result.eigenvalues(), &[9.0]);
    }

    fn assert_residual<S>(
        matrix: &Matrix<S>,
        decomposition: &HermitianEigenDecomposition<S>,
        tolerance: S::Real,
    ) where
        S: NumericScalar,
    {
        let order = matrix.rows();
        for column in 0..order {
            for row in 0..order {
                let applied =
                    (0..order).fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |sum, inner| {
                        sum + matrix.get_element(row, inner)
                            * decomposition.eigenvectors().get_element(inner, column)
                    });
                let expected = decomposition.eigenvectors().get_element(row, column)
                    * S::from_real(decomposition.eigenvalues()[column]);
                assert!(
                    (applied - expected).abs() < tolerance,
                    "residual ({row}, {column}) = {:?}, eigenvalue = {:?}, vector = {:?}",
                    (applied - expected).abs(),
                    decomposition.eigenvalues()[column],
                    decomposition.eigenvectors().as_slice(),
                );
            }
        }
    }
}
