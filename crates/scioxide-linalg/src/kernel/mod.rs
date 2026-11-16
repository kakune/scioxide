//! Checked dense kernel substrate.
//!
//! Public matrix conveniences enter through this module, where shapes, layouts,
//! backend availability, execution policy, and workspace capacity are resolved
//! before output mutation. Standard uses packed binary-scaled arithmetic; Exact
//! uses exponent-split packed products with a single final rounding.

use crate::workspace::{RealRegion, ScalarRegion};
/// Per-call scheduling and capabilities; never stored in a workspace.
#[derive(Clone, Copy)]
pub(crate) struct KernelExecution<C> {
    pub(crate) policy: ExecutionPolicy,
    pub(crate) context: C,
}
#[derive(Clone, Copy)]
pub(crate) struct MatrixOperand<V> {
    pub(crate) view: V,
    pub(crate) op: GemmOp,
}

/// Output coordinates and depth of a packed block, including global triangle coordinates.
#[derive(Clone, Copy)]
pub(crate) struct PackedBlock {
    pub(crate) row_start: usize,
    pub(crate) col_start: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) shared: usize,
    pub(crate) global_row_start: usize,
}
#[derive(Clone, Copy)]
pub(crate) struct PackedKernel<S: NumericScalar> {
    pub(crate) beta: S,
    pub(crate) spec: KernelSpec,
    pub(crate) microkernel: Option<Microkernel>,
    pub(crate) triangle: Option<Triangle>,
}
/// Register-tile coordinates, packed strides, and selected output entries.
pub(super) struct OutputTile {
    pub(super) row_start: usize,
    pub(super) col_start: usize,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) shared: usize,
    pub(super) mr: usize,
    pub(super) nr: usize,
    pub(super) global_row_start: usize,
    pub(super) triangle: Option<Triangle>,
}

impl OutputTile {
    pub(super) fn contains(&self, row: usize, col: usize) -> bool {
        self.triangle.is_none_or(|triangle| {
            triangle_contains(triangle, self.global_row_start + row, self.col_start + col)
        })
    }
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct TriangularOperand<V> {
    pub(crate) view: V,
    pub(crate) triangle: Triangle,
    pub(crate) op: GemmOp,
    pub(crate) diagonal: scioxide_core::DiagonalKind,
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct TriangularUpdate<S> {
    pub(crate) side: matrix::MatrixSide,
    pub(crate) alpha: S,
    pub(crate) policy: ExecutionPolicy,
}

pub(crate) mod arch;
mod exact;
pub(crate) mod execution;
pub(crate) mod finite;
pub(crate) mod frames;
pub(crate) mod generalized_sylvester;
pub(crate) mod gko;
pub(crate) mod level2;
pub(crate) mod level3;
pub(crate) mod matrix;
mod microkernel;
pub(crate) mod modified_rotation;
pub(crate) mod normalized;
pub(crate) mod rank;
pub(crate) mod recurrence;
pub(crate) mod small;
pub(crate) mod standard;
pub(crate) mod structured_product;
pub(crate) mod structured_vector;
pub(crate) mod subspace;
pub(crate) mod sylvester;
#[cfg(test)]
mod tile_tests;
pub(crate) mod triangular;
pub(crate) mod vector;
#[cfg(test)]
mod verified_dispatch;

use core::mem::size_of;
use core::num::NonZeroUsize;

use scioxide_core::ExecutionPlan;
use scioxide_core::backend::{KernelBackend, KernelDispatch, ScalarKind};
use scioxide_core::backend::{checked_add, checked_mul, checked_product};
use scioxide_core::{
    Dimensions, ExecutionContext, ExecutionPolicy, NumericScalar, ProductSum, RealScalar,
    SciOxideError,
};

use crate::dense::{FiniteMatrixView, MatrixView, MatrixViewMut};
use crate::{Accuracy, KernelContext};

use self::microkernel::{Microkernel, scalar_microkernel};
use scioxide_core::backend::{component_count, components, components_mut};

// Products whose every dimension fits in one 32-wide block gain nothing from
// packing: the packed panels would be mostly padding, and the direct loops
// already stay in cache. Callers use this threshold to skip packed scratch.
const SMALL_GEMM_DIMENSION: usize = 32;
fn direct_product(rows: usize, shared: usize, cols: usize) -> bool {
    (rows <= SMALL_GEMM_DIMENSION && shared <= SMALL_GEMM_DIMENSION && cols <= SMALL_GEMM_DIMENSION)
        || rows
            .checked_mul(cols)
            .is_some_and(|entries| entries <= SMALL_GEMM_DIMENSION)
}
// `Auto` stays sequential below 2^28 floating operations, where executor
// dispatch overhead is not repaid by splitting the work across workers.
pub(crate) const GEMM_AUTO_FLOPS: usize = 268_435_456;
pub(crate) const GEMM_AUTO_WORKERS: NonZeroUsize =
    NonZeroUsize::new(8).expect("resource limit must be positive");

/// Operands borrowed from a successfully validated factorization workspace.
pub(crate) struct WorkspaceGemmProblem<'a, 'b, S: NumericScalar> {
    pub(crate) left: FiniteMatrixView<'a, S>,
    pub(crate) right: FiniteMatrixView<'a, S>,
    pub(crate) output: MatrixViewMut<'b, S>,
    pub(crate) config: GemmConfig<S>,
}

pub(crate) fn gemm_workspace_with_context<'a, S: NumericScalar>(
    problem: WorkspaceGemmProblem<'_, '_, S>,
    policy: ExecutionPolicy,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: impl Into<KernelContext<'a>>,
) -> Result<(), SciOxideError> {
    let WorkspaceGemmProblem {
        left,
        right,
        output,
        config,
    } = problem;
    let left = match config.left_op {
        GemmOp::None => left,
        GemmOp::Transpose => left.transpose(),
        GemmOp::Adjoint => left.adjoint(),
    };
    let right = match config.right_op {
        GemmOp::None => right,
        GemmOp::Transpose => right.transpose(),
        GemmOp::Adjoint => right.adjoint(),
    };
    let context: crate::KernelContext<'_> = context.into();
    validate_dimensions(
        left.rows(),
        left.cols(),
        right.rows(),
        right.cols(),
        &output,
    )?;
    #[cfg(feature = "blas")]
    if context.blas().is_some() {
        return gemm_bounded(left, right, output, config, policy, workspace, context);
    }
    if context.accuracy() == Accuracy::Exact {
        gemm_bounded(left, right, output, config, policy, workspace, context)
    } else {
        normalized::workspace_bounded(left, right, output, config, policy, workspace, context)
    }
}

pub(crate) use crate::LinalgWorkspace;
use crate::workspace::{DenseShape, Packing};
use crate::{DenseWorkspaceRequest, LinalgRequirements};

pub(crate) const PACKED_A: ScalarRegion = ScalarRegion::PackedA;
pub(crate) const PACKED_B: ScalarRegion = ScalarRegion::PackedB;
pub(crate) const CANDIDATE: ScalarRegion = ScalarRegion::Candidate;
pub(crate) const STRUCTURED_A: ScalarRegion = ScalarRegion::StructuredA;
pub(crate) const STRUCTURED_B: ScalarRegion = ScalarRegion::StructuredB;
pub(crate) const STRUCTURED_OUTPUT: ScalarRegion = ScalarRegion::StructuredOutput;
const EXACT_TILES: ScalarRegion = ScalarRegion::ExactTiles;

/// Dense requests are sized by the kernels that consume them, so the shared
/// workspace never depends on kernel tiling or scheduling.
impl<S: NumericScalar> crate::workspace::Request<S> for DenseWorkspaceRequest {
    fn requirements(&self) -> Result<LinalgRequirements, SciOxideError> {
        product_requirements::<S>(self)
    }
}

/// Workers a request may need: the policy bound capped by the GEMM scheduler's
/// limit, the request's own bound and the available partitions.
fn worker_bound(request: &DenseWorkspaceRequest, work: usize, partitions: usize) -> usize {
    let workers = match request.options.execution_policy() {
        ExecutionPolicy::Sequential => 1,
        ExecutionPolicy::Auto if work < GEMM_AUTO_FLOPS => 1,
        ExecutionPolicy::Auto => GEMM_AUTO_WORKERS.get(),
        ExecutionPolicy::Parallel(workers) => workers.get().min(GEMM_AUTO_WORKERS.get()),
    };
    workers.min(request.workers.get()).min(partitions)
}

pub(crate) fn product_requirements<S: NumericScalar>(
    request: &DenseWorkspaceRequest,
) -> Result<LinalgRequirements, SciOxideError> {
    let mut requirements = LinalgRequirements::new(request.options.byte_limit());
    let (rows, shared, cols) = match request.shape {
        DenseShape::Vector(len) => {
            *requirements.scalar_mut(CANDIDATE) = len;
            requirements.required_bytes::<S>()?;
            return Ok(requirements);
        }
        DenseShape::InPlaceVector(len) => {
            if matches!(request.packing, Packing::Provider) {
                *requirements.scalar_mut(CANDIDATE) = len;
            }
            requirements.required_bytes::<S>()?;
            return Ok(requirements);
        }
        DenseShape::VectorPair(len) => {
            *requirements.scalar_mut(CANDIDATE) =
                checked_product(&[2, len], scioxide_core::quantity!("paired_vector_scratch"))?;
            requirements.required_bytes::<S>()?;
            return Ok(requirements);
        }
        DenseShape::TriangularProduct { rows, order, cols } => {
            let candidate = checked_product(
                &[rows, cols],
                scioxide_core::quantity!("triangular_product_output"),
            )?;
            *requirements.scalar_mut(CANDIDATE) = candidate;
            *requirements.scalar_mut(ScalarRegion::Direct) = candidate;
            if candidate != 0 && matches!(request.packing, Packing::Provider) {
                *requirements.scalar_mut(PACKED_A) = checked_product(
                    &[order, order],
                    scioxide_core::quantity!("provider_triangular_operand"),
                )?;
            }
            requirements.required_bytes::<S>()?;
            return Ok(requirements);
        }
        DenseShape::RankTwo { order, shared } => (order, shared, order),
        DenseShape::Product { rows, shared, cols }
        | DenseShape::TriangularSolve { rows, shared, cols }
        | DenseShape::Structured { rows, shared, cols } => (rows, shared, cols),
    };
    let candidate = checked_product(&[rows, cols], scioxide_core::quantity!("product_output"))?;
    *requirements.scalar_mut(CANDIDATE) = candidate;
    if matches!(request.shape, DenseShape::TriangularSolve { .. }) {
        *requirements.scalar_mut(ScalarRegion::Direct) = candidate;
    }
    if rows != 0 && cols != 0 {
        let work = checked_product(
            &[rows, shared, cols, 2],
            scioxide_core::quantity!("product_work"),
        )?;
        if !direct_product(rows, shared, cols) {
            for backend in KernelSpec::BACKENDS {
                let spec = KernelSpec::for_scalar::<S>(backend);
                let workers = worker_bound(request, work, rows.div_ceil(spec.mc));
                let (a, b) = spec.panel_lengths::<S>(rows, shared, cols)?;
                *requirements.scalar_mut(PACKED_A) = requirements.scalar(PACKED_A).max(
                    checked_mul(a, workers, scioxide_core::quantity!("packed_left"))?,
                );
                *requirements.scalar_mut(PACKED_B) = requirements.scalar(PACKED_B).max(
                    checked_mul(b, workers, scioxide_core::quantity!("packed_right"))?,
                );
            }
        }
        if request.options.accuracy() == Accuracy::Exact
            && (!direct_product(rows, shared, cols)
                || matches!(request.shape, DenseShape::RankTwo { .. }))
        {
            let tiles = exact::tile_count(rows, cols)?;
            let exact = exact::Layout::new::<S>(rows, cols, worker_bound(request, work, tiles))?;
            *requirements.scalar_mut(EXACT_TILES) = exact.tiles;
            *requirements.real_mut(RealRegion::Exact) = exact.reals;
            requirements.accumulators = exact.accumulators;
        }
        if matches!(request.packing, Packing::Provider) {
            *requirements.scalar_mut(PACKED_A) = requirements.scalar(PACKED_A).max(
                checked_product(&[rows, shared], scioxide_core::quantity!("provider_left"))?,
            );
            *requirements.scalar_mut(PACKED_B) = requirements.scalar(PACKED_B).max(
                checked_product(&[shared, cols], scioxide_core::quantity!("provider_right"))?,
            );
        }
        if matches!(request.packing, Packing::Provider)
            && matches!(
                request.shape,
                DenseShape::Structured { .. } | DenseShape::TriangularSolve { .. }
            )
        {
            *requirements.scalar_mut(PACKED_A) =
                requirements.scalar(PACKED_A).max(checked_product(
                    &[shared, shared],
                    scioxide_core::quantity!("provider_structured_operand"),
                )?);
            *requirements.scalar_mut(PACKED_B) = requirements.scalar(PACKED_B).max(candidate);
        }
        if matches!(request.shape, DenseShape::Structured { .. }) {
            *requirements.scalar_mut(STRUCTURED_A) =
                checked_product(&[rows, shared], scioxide_core::quantity!("structured_left"))?;
            *requirements.scalar_mut(STRUCTURED_B) = checked_product(
                &[shared, cols],
                scioxide_core::quantity!("structured_right"),
            )?;
            *requirements.scalar_mut(STRUCTURED_OUTPUT) = candidate;
        }
    }
    if matches!(request.shape, DenseShape::RankTwo { .. }) {
        *requirements.scalar_mut(STRUCTURED_OUTPUT) = candidate;
    }
    requirements.required_bytes::<S>()?;
    Ok(requirements)
}

impl<S: NumericScalar> LinalgWorkspace<'_, S> {
    pub(crate) fn require_product<'a>(
        &self,
        rows: usize,
        shared: usize,
        cols: usize,
        policy: ExecutionPolicy,
        context: impl Into<KernelContext<'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into();
        let options = crate::LinalgExecutionOptions::new().with_execution_policy(policy);
        let request = DenseWorkspaceRequest::product(rows, shared, cols, &options)
            .with_product_accuracy(context.accuracy())
            .for_kernels(context);
        self.require_capacity(&product_requirements::<S>(&request)?)
    }

    #[cfg(feature = "blas")]
    pub(crate) fn require_packed(&self, left: usize, right: usize) -> Result<(), SciOxideError> {
        let mut req = LinalgRequirements::default();
        *req.scalar_mut(PACKED_A) = left;
        *req.scalar_mut(PACKED_B) = right;
        self.require_capacity(&req)
    }

    pub(crate) fn require_gemm(&self, plan: &GemmPlan) -> Result<(), SciOxideError> {
        let mut req = LinalgRequirements::default();
        *req.scalar_mut(PACKED_A) = plan.packed_a_len;
        *req.scalar_mut(PACKED_B) = plan.packed_b_len;
        *req.scalar_mut(CANDIDATE) = checked_product(
            &[plan.rows, plan.cols],
            scioxide_core::quantity!("gemm_candidate_elements"),
        )?;
        self.require_capacity(&req)
    }

    fn require_candidate(&self, len: usize) -> Result<(), SciOxideError> {
        self.require_capacity(&LinalgRequirements::new(None).with_scalar(CANDIDATE, len))
    }

    fn exact_layout(
        &self,
        rows: usize,
        shared: usize,
        cols: usize,
        policy: ExecutionPolicy,
        context: ExecutionContext<'_>,
    ) -> Result<exact::Layout, SciOxideError> {
        let layout = exact::Layout::new::<S>(
            rows,
            cols,
            exact::plan(rows, shared, cols, policy, context)?.worker_count(),
        )?;
        let mut req = LinalgRequirements::new(None)
            .with_scalar(EXACT_TILES, layout.tiles)
            .with_real(RealRegion::Exact, layout.reals);
        req.accumulators = layout.accumulators;
        self.require_capacity(&req)?;
        Ok(layout)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemmOp {
    None,
    Transpose,
    Adjoint,
}

pub(crate) use scioxide_core::Triangle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RankKind {
    Symmetric,
    Hermitian,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GemmConfig<S: NumericScalar> {
    pub(crate) left_op: GemmOp,
    pub(crate) right_op: GemmOp,
    pub(crate) alpha: S,
    pub(crate) beta: S,
    pub(crate) backend: Option<KernelBackend>,
    pub(crate) triangle: Option<Triangle>,
    /// Repeated equation tiles amortize packing even below the small-GEMM cutoff.
    /// The caller must reserve packed workspace for the tile geometry.
    pub(crate) packed: bool,
    /// Known upper bound on the components of the output that `beta` scales,
    /// within `triangle`. It only selects the rescaling route, so any bound
    /// that is not below the true maximum is valid; `None` measures it.
    pub(crate) prior: Option<S::Real>,
    /// Single-input proof supplied exclusively by SYRK/HERK entry points.
    #[cfg(feature = "blas")]
    pub(crate) rank: Option<RankKind>,
}

impl<S: NumericScalar> GemmConfig<S> {
    pub(crate) fn ordinary() -> Self {
        Self {
            left_op: GemmOp::None,
            right_op: GemmOp::None,
            alpha: S::ONE,
            beta: S::ZERO,
            backend: None,
            triangle: None,
            packed: false,
            prior: None,
            #[cfg(feature = "blas")]
            rank: None,
        }
    }
}

/// Conservative bound on the components of the shrinking trailing matrix of a
/// right-looking factorization, kept across panels so that trailing updates
/// need not rescan it (see `GemmConfig::prior`). An update `C -= L U` of depth
/// `k` adds at most `2k max|L| max|U|` to any component, and the bound adds a
/// power of two of at least twice that. Row interchanges inside the trailing
/// rows keep it valid. Near overflow it stops serving as a prior.
#[derive(Clone, Copy)]
pub(crate) struct TrailingBound<R>(R);

impl<R: RealScalar> TrailingBound<R> {
    pub(crate) const fn new(maximum: R) -> Self {
        Self(maximum)
    }

    /// The prior of the next update, while far below overflow.
    pub(crate) fn prior(self) -> Option<R> {
        (self.0 < R::MAX.sqrt()).then_some(self.0)
    }

    /// The bound after an update of depth `depth` with operand bounds `left`, `right`.
    pub(crate) fn update(self, depth: usize, left: R, right: R) -> Self {
        let growth = R::ONE.scalbn((usize::BITS - depth.leading_zeros()) as i32 + 2);
        Self(self.0 + growth * left * right)
    }
}

// Goto/BLIS-style blocking parameters. `mr x nr` is the register tile computed
// by one microkernel call; `mc x kc` bounds a packed A block (sized for L2) and
// `kc x nc` a packed B block (sized for L3/L2). The architecture tiles below
// must match the leaves in `arch`, which assume exactly these `mr`/`nr` shapes.
// All counts are in scalars of the product; complex products run the real
// leaves on component panels (see `pack_a`/`pack_b`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct KernelSpec {
    pub(crate) mr: usize,
    pub(crate) nr: usize,
    pub(crate) mc: usize,
    pub(crate) kc: usize,
    pub(crate) nc: usize,
}

impl KernelSpec {
    pub(super) const BACKENDS: [KernelBackend; 4] = [
        KernelBackend::Scalar,
        KernelBackend::Avx2Fma,
        KernelBackend::Avx512,
        KernelBackend::Neon,
    ];
    fn panel_lengths<S: NumericScalar>(
        self,
        rows: usize,
        shared: usize,
        cols: usize,
    ) -> Result<(usize, usize), SciOxideError> {
        let packed_a_rows = checked_mul(
            rows.min(self.mc).div_ceil(self.mr),
            self.mr,
            scioxide_core::quantity!("gemm_packed_a_padded_rows"),
        )?;
        let packed_b_cols = checked_mul(
            cols.min(self.nc).div_ceil(self.nr),
            self.nr,
            scioxide_core::quantity!("gemm_packed_b_padded_columns"),
        )?;
        let packed_a_panel_len = checked_product(
            &[packed_a_rows, shared.min(self.kc)],
            scioxide_core::quantity!("gemm_packed_a_elements_per_worker"),
        )?;
        // A packed B shared index holds one real row per scalar component.
        let packed_b_panel_len = checked_product(
            &[shared.min(self.kc), packed_b_cols, component_count::<S>()],
            scioxide_core::quantity!("gemm_packed_b_elements_per_worker"),
        )?;
        // Per-worker panels are laid out back to back; one extra cache line
        // between them keeps neighbouring workers from false-sharing a line.
        let cache_line_padding = 64_usize.div_ceil(size_of::<S>());
        let packed_a_per_worker = checked_add(
            packed_a_panel_len,
            cache_line_padding,
            scioxide_core::quantity!("gemm_padded_a_scratch_stride"),
        )?;
        let packed_b_per_worker = checked_add(
            packed_b_panel_len,
            cache_line_padding,
            scioxide_core::quantity!("gemm_padded_b_scratch_stride"),
        )?;
        Ok((packed_a_per_worker, packed_b_per_worker))
    }
    const SCALAR: Self = Self {
        mr: 4,
        nr: 4,
        mc: 64,
        kc: 128,
        nc: 128,
    };

    /// Tiles of `S`: the real leaf geometry of `S::Real` with `nr`, `kc` and
    /// `nc` counted in scalars. A complex column or shared index spans two real
    /// components, so complex tiles keep the real register tile, real depth and
    /// cache footprint of their component type.
    pub(crate) fn for_scalar<S: NumericScalar>(backend: KernelBackend) -> Self {
        let real = Self::for_real(<S::Real as scioxide_core::backend::Scalar>::kind(), backend);
        let count = component_count::<S>();
        Self {
            nr: real.nr / count,
            kc: real.kc / count,
            nc: real.nc / count,
            ..real
        }
    }

    fn for_real(kind: ScalarKind, backend: KernelBackend) -> Self {
        match backend {
            KernelBackend::Avx2Fma => Self {
                mr: if kind == ScalarKind::F32 { 4 } else { 6 },
                nr: if kind == ScalarKind::F32 { 16 } else { 8 },
                mc: 120,
                kc: 256,
                nc: 256,
            },
            KernelBackend::Avx512 => Self {
                mr: 4,
                nr: if kind == ScalarKind::F32 { 32 } else { 16 },
                mc: 128,
                kc: 256,
                nc: 256,
            },
            KernelBackend::Neon => Self {
                mr: 4,
                nr: if kind == ScalarKind::F32 { 8 } else { 4 },
                mc: 96,
                kc: 256,
                nc: 256,
            },
            KernelBackend::Scalar => Self::SCALAR,
        }
    }
}

/// Factor panels share the packed kernel's cache geometry, bounded by each algorithm's cap.
pub(crate) fn panel_width<S: NumericScalar>(context: KernelContext<'_>, cap: usize) -> usize {
    (KernelSpec::for_scalar::<S>(context.dispatch().backend()).kc / 4)
        .max(32)
        .min(cap)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GemmPlan {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) spec: KernelSpec,
    pub(crate) dispatch: KernelDispatch,
    execution: ExecutionPlan,
    pub(crate) packed_a_per_worker: usize,
    pub(crate) packed_b_per_worker: usize,
    packed_a_len: usize,
    packed_b_len: usize,
}

impl GemmPlan {
    pub(crate) fn new<S: NumericScalar>(
        rows: usize,
        shared: usize,
        cols: usize,
        policy: ExecutionPolicy,
        context: ExecutionContext<'_>,
        dispatch: KernelDispatch,
    ) -> Result<Self, SciOxideError> {
        if rows == 0 || cols == 0 {
            return Ok(Self {
                rows,
                cols,
                spec: KernelSpec::for_scalar::<S>(dispatch.backend()),
                dispatch,
                execution: context.plan(
                    ExecutionPolicy::Sequential,
                    0,
                    scioxide_core::PlanLimits {
                        work_per_item: core::num::NonZeroUsize::MIN,
                        auto_threshold: NonZeroUsize::MIN,
                        worker_limit: GEMM_AUTO_WORKERS,
                    },
                ),
                packed_a_per_worker: 0,
                packed_b_per_worker: 0,
                packed_a_len: 0,
                packed_b_len: 0,
            });
        }
        let _ = checked_product(
            &[rows, cols],
            scioxide_core::quantity!("gemm_output_logical_element_count"),
        )?;
        let products = checked_product(
            &[rows, shared, cols],
            scioxide_core::quantity!("gemm_operation_count"),
        )?;
        let flops = checked_mul(
            products,
            2,
            scioxide_core::quantity!("gemm_floating_operation_count"),
        )?;
        let spec = KernelSpec::for_scalar::<S>(dispatch.backend());
        // Work is partitioned by `mc` row blocks so each worker owns disjoint
        // output rows and needs only its own packed A block.
        let row_blocks = rows.div_ceil(spec.mc);
        let effective_policy = match policy {
            ExecutionPolicy::Auto if flops < GEMM_AUTO_FLOPS => ExecutionPolicy::Sequential,
            other => other,
        };
        let execution = context.plan(
            effective_policy,
            row_blocks,
            scioxide_core::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: NonZeroUsize::MIN,
                worker_limit: GEMM_AUTO_WORKERS,
            },
        );
        // Panels are padded up to whole `mr`/`nr` tiles (the tail lanes are
        // zero-filled by the packers), so a full-tile leaf never reads past a
        // panel even for ragged edge blocks.
        let (packed_a_per_worker, packed_b_per_worker) =
            spec.panel_lengths::<S>(rows, shared, cols)?;
        let packed_a_len = checked_mul(
            packed_a_per_worker,
            execution.worker_count(),
            scioxide_core::quantity!("gemm_packed_a_workspace_elements"),
        )?;
        let packed_b_len = checked_mul(
            packed_b_per_worker,
            execution.worker_count(),
            scioxide_core::quantity!("gemm_packed_b_workspace_elements"),
        )?;
        Ok(Self {
            rows,
            cols,
            spec,
            dispatch,
            execution,
            packed_a_per_worker,
            packed_b_per_worker,
            packed_a_len,
            packed_b_len,
        })
    }
}

pub(crate) fn gemm_bounded<S: NumericScalar>(
    left_finite: FiniteMatrixView<'_, S>,
    right_finite: FiniteMatrixView<'_, S>,
    mut output: MatrixViewMut<'_, S>,
    config: GemmConfig<S>,
    policy: ExecutionPolicy,
    workspace: &mut LinalgWorkspace<'_, S>,
    context: KernelContext<'_>,
) -> Result<(), SciOxideError> {
    let left = left_finite.as_view();
    let right = right_finite.as_view();
    let (left_rows, left_cols, right_cols) = (left.rows, left.cols, right.cols);
    let dispatch = config.backend.map_or_else(
        || Ok(context.dispatch()),
        |backend| KernelDispatch::for_backend(backend, context.execution().processor()),
    )?;
    let plan = GemmPlan::new::<S>(
        left_rows,
        left_cols,
        right_cols,
        policy,
        context.execution(),
        dispatch,
    )?;
    #[cfg(feature = "blas")]
    // Empty plans have no workers; their unused provider limit uses the valid
    // minimum. Nonempty plans preserve the exact caller budget.
    let provider_workers =
        NonZeroUsize::new(plan.execution.worker_count()).unwrap_or(NonZeroUsize::MIN);

    // Transactional contract: every fallible step (capacity checks, finiteness checks,
    // provider calls, executor failures, overflow in the accumulation) happens
    // while the result lives only in `candidate`. `output` is written solely
    // in the final commit loops, so an error leaves it untouched.
    let candidate_len = checked_product(
        &[plan.rows, plan.cols],
        scioxide_core::quantity!("gemm_transactional_candidate_elements"),
    )?;

    #[cfg(feature = "blas")]
    if left_rows != 0
        && right_cols != 0
        && left_cols != 0
        && config.backend.is_none()
        && let Some(blas) = context.blas()
    {
        workspace.require_candidate(candidate_len)?;
        let published = blas_candidate(
            PreparedGemm {
                left: left_finite,
                right: right_finite,
                prior: output.as_view(),
                config,
            },
            workspace,
            blas,
            context.accuracy(),
            provider_workers,
            candidate_len,
        )?;
        // No layout means the dispatch declined (unsupported scalar or shape);
        // fall through to the selected native tier, which rebuilds the candidate
        // from scratch.
        if let Some(layout) = published {
            for row in 0..left_rows {
                for col in 0..right_cols {
                    if config
                        .triangle
                        .is_none_or(|triangle| triangle_contains(triangle, row, col))
                    {
                        output.set_element(
                            row,
                            col,
                            workspace.scalar_mut(CANDIDATE)[crate::blas::dispatch::matrix_index(
                                layout, left_rows, right_cols, row, col,
                            )],
                        );
                    }
                }
            }
            return Ok(());
        }
    }
    if context.accuracy() == Accuracy::Standard {
        return normalized::bounded(
            left_finite,
            right_finite,
            output,
            config,
            workspace,
            context,
            plan,
        );
    }
    workspace.require_candidate(candidate_len)?;
    if !direct_product(left_rows, left_cols, right_cols) {
        let exact_layout = workspace.exact_layout(
            left_rows,
            left_cols,
            right_cols,
            policy,
            context.execution(),
        )?;
        let crate::workspace::ProductParts {
            candidate,
            tiles,
            reals,
            accumulators,
            ..
        } = workspace.product_parts()?;
        let mut exact = exact::Workspace {
            tiles,
            reals,
            accumulators,
            layout: exact_layout,
        };
        let candidate = &mut candidate[..candidate_len];
        for row in 0..left_rows {
            for col in 0..right_cols {
                let selected = config
                    .triangle
                    .is_none_or(|triangle| triangle_contains(triangle, row, col));
                candidate[row * right_cols + col] = if selected && config.beta == S::ZERO {
                    S::ZERO
                } else {
                    output.get_element(row, col)
                };
                if selected && config.beta != S::ZERO {
                    crate::checked::validate_input(
                        candidate[row * right_cols + col],
                        "output entry",
                    )?;
                }
            }
        }
        exact.multiply(
            crate::kernel::exact::ExactProduct {
                left,
                right,
                config,
                addend: None,
            },
            candidate,
            policy,
            context.execution(),
            dispatch,
        )?;
        for row in 0..left_rows {
            for col in 0..right_cols {
                output.set_element(row, col, candidate[row * right_cols + col]);
            }
        }
        return Ok(());
    }

    let candidate = &mut workspace.scalar_mut(CANDIDATE)[..candidate_len];
    for row in 0..left_rows {
        for col in 0..right_cols {
            let index = row * right_cols + col;
            if config
                .triangle
                .is_some_and(|triangle| !triangle_contains(triangle, row, col))
            {
                candidate[index] = output.get_element(row, col);
                continue;
            }

            // Each entry `alpha * sum(a * b) + beta * c` is accumulated exactly
            // and rounded once, so the public product is independent of
            // summation order and backend, and intermediate overflow of a
            // representable result is impossible. `alpha == 0` / `beta == 0`
            // skip their terms entirely so that non-finite values in the
            // skipped operand cannot contaminate the result (BLAS semantics).
            let mut affine = ProductSum::new();
            if config.alpha != S::ZERO {
                for inner in 0..left_cols {
                    affine.add_triple_product(
                        config.alpha,
                        left.get_element(row, inner),
                        right.get_element(inner, col),
                    )?;
                }
            }
            if config.beta != S::ZERO {
                let prior = output.get_element(row, col);
                crate::checked::validate_input(prior, "output entry")?;
                affine.add_product(config.beta, prior)?;
            }
            candidate[index] = affine.finish()?;
        }
    }

    for row in 0..left_rows {
        for col in 0..right_cols {
            output.set_element(row, col, candidate[row * right_cols + col]);
        }
    }
    Ok(())
}

pub(crate) fn operated_view<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    op: GemmOp,
) -> MatrixView<'_, S> {
    match op {
        GemmOp::None => matrix,
        GemmOp::Transpose => matrix.transpose(),
        GemmOp::Adjoint => matrix.adjoint(),
    }
}

pub(crate) fn validate_dimensions<S: NumericScalar>(
    left_rows: usize,
    left_cols: usize,
    right_rows: usize,
    right_cols: usize,
    output: &MatrixViewMut<'_, S>,
) -> Result<(), SciOxideError> {
    if left_cols != right_rows {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(left_cols),
            Dimensions::one(right_rows),
        ));
    }
    if output.rows != left_rows || output.cols != right_cols {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::two(left_rows, right_cols),
            Dimensions::two(output.rows, output.cols),
        ));
    }
    Ok(())
}

pub(crate) const fn triangle_contains(triangle: Triangle, row: usize, col: usize) -> bool {
    match triangle {
        Triangle::Lower => row >= col,
        Triangle::Upper => row <= col,
    }
}

/// The triangle a transposed or adjoint view of `triangle` occupies.
pub(crate) const fn flip_triangle(triangle: Triangle) -> Triangle {
    match triangle {
        Triangle::Lower => Triangle::Upper,
        Triangle::Upper => Triangle::Lower,
    }
}

fn operated_get<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    op: GemmOp,
    row: usize,
    col: usize,
) -> S {
    match op {
        GemmOp::None => matrix.get_element(row, col),
        GemmOp::Transpose => matrix.get_element(col, row),
        GemmOp::Adjoint => matrix.get_element(col, row).conj(),
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PackingPanel {
    outer_start: usize,
    shared_start: usize,
    outer_len: usize,
    shared_len: usize,
    lane_count: usize,
    scale: i32,
}

impl PackingPanel {
    pub(crate) const fn new(
        outer_start: usize,
        shared_start: usize,
        outer_len: usize,
        shared_len: usize,
        lane_count: usize,
    ) -> Self {
        Self {
            outer_start,
            shared_start,
            outer_len,
            shared_len,
            lane_count,
            scale: 0,
        }
    }
    pub(crate) const fn scaled(mut self, exponent: i32) -> Self {
        self.scale = exponent;
        self
    }
}

/// Operand role of a packed panel, which fixes its real component layout.
///
/// Packed products run the real leaves (the 1m formulation). For each shared
/// index, a left (A) panel stores the real parts of its `lanes` rows followed
/// by their imaginary parts; a right (B) panel stores `re, im` per column and
/// then `-im, re`. A real leaf over `2 * depth` then produces the interleaved
/// `re, im` of every complex output entry. Real scalars keep one real row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PanelSide {
    Left,
    Right,
}

impl PanelSide {
    /// Scalars of one packed shared index of a `lanes`-wide panel.
    pub(crate) const fn depth_stride<S: NumericScalar>(self, lanes: usize) -> usize {
        match self {
            Self::Left => lanes,
            Self::Right => lanes * component_count::<S>(),
        }
    }

    /// Store `value` at `lane` of one packed shared index, given as real components.
    #[inline]
    fn store<S: NumericScalar>(self, slot: &mut [S::Real], lanes: usize, lane: usize, value: S) {
        let (re, im) = (value.real_component(), value.imaginary_component());
        match (S::IS_REAL, self) {
            (true, _) => slot[lane] = re,
            (false, Self::Left) => {
                slot[lane] = re;
                slot[lanes + lane] = im;
            }
            (false, Self::Right) => {
                slot[2 * lane] = re;
                slot[2 * lane + 1] = im;
                slot[2 * (lanes + lane)] = -im;
                slot[2 * (lanes + lane) + 1] = re;
            }
        }
    }
}

/// Pack A as consecutive `mr`-row panels with alpha folded into the entries.
pub(crate) fn pack_a<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    op: GemmOp,
    panel: PackingPanel,
    alpha: S,
    packed: &mut [S],
) {
    pack_panel(
        operated_view(matrix, op).transpose(),
        panel,
        PanelSide::Left,
        alpha,
        packed,
    );
}

/// Pack B as consecutive `nr`-column panels, zero-padding incomplete panels.
pub(crate) fn pack_b<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    op: GemmOp,
    panel: PackingPanel,
    packed: &mut [S],
) {
    pack_panel(
        operated_view(matrix, op),
        panel,
        PanelSide::Right,
        S::ONE,
        packed,
    );
}

/// Pack one `lanes`-wide panel of `depth` shared indices from `entry(inner, lane)`;
/// callers return zero for padding lanes.
pub(crate) fn pack_entries<S: NumericScalar>(
    packed: &mut [S],
    side: PanelSide,
    lanes: usize,
    depth: usize,
    entry: impl Fn(usize, usize) -> S,
) {
    let stride = side.depth_stride::<S>(lanes);
    let target = components_mut(&mut packed[..depth * stride]);
    for (inner, slot) in target
        .chunks_exact_mut(stride * component_count::<S>())
        .enumerate()
    {
        for lane in 0..lanes {
            side.store(slot, lanes, lane, entry(inner, lane));
        }
    }
}

// Both operands keep the panel's outer lanes consecutive inside each shared
// coordinate. Resolve transformations before the hot loops.
fn pack_panel<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    panel: PackingPanel,
    side: PanelSide,
    alpha: S,
    packed: &mut [S],
) {
    let conjugated = !S::IS_REAL && matrix.conjugated;
    if panel.scale == 0 && alpha == S::ONE {
        if conjugated {
            pack_panel_with(matrix, panel, side, packed, S::conj);
        } else {
            pack_panel_with(matrix, panel, side, packed, core::convert::identity);
        }
    } else {
        pack_panel_with(matrix, panel, side, packed, |value| {
            let value = if conjugated { value.conj() } else { value };
            alpha * standard::scale(value, panel.scale)
        });
    }
}

fn pack_panel_with<S: NumericScalar>(
    matrix: MatrixView<'_, S>,
    panel: PackingPanel,
    side: PanelSide,
    packed: &mut [S],
    transform: impl Fn(S) -> S,
) {
    if panel.shared_len == 0 {
        return;
    }
    let lanes = panel.lane_count;
    let depth_stride = side.depth_stride::<S>(lanes);
    let slot_len = depth_stride * component_count::<S>();
    let stride = panel.shared_len * depth_stride;
    for outer in (0..panel.outer_len).step_by(lanes) {
        let valid = lanes.min(panel.outer_len - outer);
        let start = outer / lanes * stride;
        let target = components_mut(&mut packed[start..start + stride]);
        let base = panel.shared_start * matrix.row_stride
            + (panel.outer_start + outer) * matrix.col_stride;
        if valid < lanes {
            target.fill(<S::Real as scioxide_core::AlgebraicScalar>::ZERO);
        }
        if matrix.col_stride == 1 {
            for (inner, slot) in target.chunks_exact_mut(slot_len).enumerate() {
                let offset = base + inner * matrix.row_stride;
                for (lane, &value) in matrix.data[offset..offset + valid].iter().enumerate() {
                    side.store(slot, lanes, lane, transform(value));
                }
            }
        } else {
            // Read short contiguous runs along the shared axis. Blocking keeps
            // the transposed destination in a few cache lines for either layout.
            for first in (0..panel.shared_len).step_by(8) {
                let end = (first + 8).min(panel.shared_len);
                for lane in 0..valid {
                    let offset = base + lane * matrix.col_stride;
                    for inner in first..end {
                        side.store(
                            &mut target[inner * slot_len..(inner + 1) * slot_len],
                            lanes,
                            lane,
                            transform(matrix.data[offset + inner * matrix.row_stride]),
                        );
                    }
                }
            }
        }
    }
}

pub(crate) fn apply_packed_block<S: NumericScalar>(
    packed_a: &[S],
    packed_b: &[S],
    output: &mut MatrixViewMut<'_, S>,
    block: crate::kernel::PackedBlock,
    kernel: crate::kernel::PackedKernel<S>,
) {
    let crate::kernel::PackedBlock {
        row_start,
        col_start,
        rows,
        cols,
        shared,
        global_row_start,
    } = block;
    let crate::kernel::PackedKernel {
        beta,
        spec,
        microkernel,
        triangle,
    } = kernel;

    let row_panels = rows.div_ceil(spec.mr);
    let col_panels = cols.div_ceil(spec.nr);
    let b_len = shared * PanelSide::Right.depth_stride::<S>(spec.nr);
    let mut row_panel = 0;
    while row_panel < row_panels {
        let tile_rows = (rows - row_panel * spec.mr).min(spec.mr);
        let a_start = row_panel * shared * spec.mr;
        let a_panel = components(&packed_a[a_start..a_start + shared * spec.mr]);
        let mut col_panel = 0;
        while col_panel < col_panels {
            let tile_cols = (cols - col_panel * spec.nr).min(spec.nr);
            let b_start = col_panel * b_len;
            let b_panel = components(&packed_b[b_start..b_start + b_len]);
            let tile_row_start = global_row_start + row_panel * spec.mr;
            let tile_col_start = col_start + col_panel * spec.nr;
            // Triangle tests use global coordinates (`global_row_start`)
            // because `row_start` may be relative to a partitioned view.
            // Tiles entirely outside the triangle are skipped, tiles entirely
            // inside write directly; straddling tiles publish selected entries only.
            let wholly_outside = triangle.is_some_and(|selected| match selected {
                Triangle::Lower => tile_row_start + tile_rows - 1 < tile_col_start,
                Triangle::Upper => tile_row_start > tile_col_start + tile_cols - 1,
            });
            if wholly_outside {
                col_panel += 1;
                continue;
            }
            let wholly_inside = triangle.is_none_or(|selected| match selected {
                Triangle::Lower => tile_row_start >= tile_col_start + tile_cols - 1,
                Triangle::Upper => tile_row_start + tile_rows - 1 <= tile_col_start,
            });
            let tile = OutputTile {
                row_start: row_start + row_panel * spec.mr,
                col_start: tile_col_start,
                rows: tile_rows,
                cols: tile_cols,
                shared,
                mr: spec.mr,
                nr: spec.nr,
                global_row_start: tile_row_start,
                triangle: if wholly_inside { None } else { triangle },
            };
            if !microkernel
                .is_some_and(|leaf| leaf.apply_tile(a_panel, b_panel, output, &tile, beta))
            {
                scalar_microkernel(a_panel, b_panel, output, &tile, beta);
            }
            col_panel += 1;
        }
        row_panel += 1;
    }
}

#[cfg(feature = "blas")]
struct PreparedGemm<'a, 'b, S: NumericScalar> {
    left: crate::FiniteMatrixView<'a, S>,
    right: crate::FiniteMatrixView<'a, S>,
    prior: MatrixView<'b, S>,
    config: GemmConfig<S>,
}
#[cfg(feature = "blas")]
fn blas_candidate<S: NumericScalar>(
    problem: PreparedGemm<'_, '_, S>,
    workspace: &mut LinalgWorkspace<'_, S>,
    blas: crate::BlasDispatch<'_>,
    accuracy: scioxide_core::Accuracy,
    provider_workers: NonZeroUsize,
    candidate_len: usize,
) -> Result<Option<crate::BlasLayout>, SciOxideError> {
    let PreparedGemm {
        left,
        right,
        prior: output,
        config,
    } = problem;
    let left = left.as_view();
    let right = right.as_view();
    let (left_rows, left_cols) = (left.rows, left.cols);
    let (right_rows, right_cols) = (right.rows, right.cols);
    let left_len = checked_product(
        &[left_rows, left_cols],
        scioxide_core::quantity!("blas_left_packed_elements"),
    )?;
    let right_len = checked_product(
        &[right_rows, right_cols],
        scioxide_core::quantity!("blas_right_packed_elements"),
    )?;
    workspace.require_packed(left_len, right_len)?;
    let capabilities = blas.capabilities()?;
    let layout = crate::blas::dispatch::preferred_layout(&capabilities);
    for row in 0..left_rows {
        for col in 0..right_cols {
            let selected = config
                .triangle
                .is_none_or(|triangle| triangle_contains(triangle, row, col));
            let value = if selected && config.beta != S::ZERO {
                let value = output.get_element(row, col);
                crate::checked::validate_input(value, "output entry")?;
                value
            } else {
                S::ZERO
            };
            workspace.scalar_mut(CANDIDATE)
                [crate::blas::dispatch::matrix_index(layout, left_rows, right_cols, row, col)] =
                value;
        }
    }
    let crate::workspace::ProductParts {
        packed_a,
        packed_b,
        candidate,
        ..
    } = workspace.product_parts()?;
    let route = match (config.rank, config.triangle) {
        (Some(rank), Some(triangle)) => {
            crate::blas::dispatch::rank_k(crate::blas::dispatch::BlasRankK {
                accuracy,
                input: left,
                order: left_rows,
                alpha: config.alpha,
                beta: config.beta,
                triangle,
                hermitian: rank == RankKind::Hermitian && !S::IS_REAL,
                packed_input: &mut packed_a[..left_len],
                candidate: &mut candidate[..candidate_len],
                dispatch: blas,
                capabilities: &capabilities,
                workers: provider_workers,
            })?
        }
        _ => crate::blas::dispatch::gemm(crate::blas::dispatch::BlasGemm {
            accuracy,
            left,
            right,
            alpha: config.alpha,
            beta: config.beta,
            triangle: config.triangle,
            packed_left: &mut packed_a[..left_len],
            packed_right: &mut packed_b[..right_len],
            candidate: &mut candidate[..candidate_len],
            dispatch: blas,
            capabilities: &capabilities,
            workers: provider_workers,
        })?,
    };
    Ok((route == crate::blas::dispatch::BlasRoute::Provider).then_some(layout))
}
#[cfg(feature = "blas")]
pub(crate) fn provider_worker_bound<S: NumericScalar>(
    rows: usize,
    shared: usize,
    cols: usize,
    policy: ExecutionPolicy,
    context: KernelContext<'_>,
) -> Result<NonZeroUsize, SciOxideError> {
    let plan = GemmPlan::new::<S>(
        rows,
        shared,
        cols,
        policy,
        context.execution(),
        context.dispatch(),
    )?;
    Ok(NonZeroUsize::new(plan.execution.worker_count()).unwrap_or(NonZeroUsize::MIN))
}

/// Provider worker budget for a vector or scalar routine under the context's
/// composed policy: one worker for `Sequential`, at most the requested count
/// for `Parallel`, and the executor budget for `Auto`.
#[cfg(feature = "blas")]
pub(crate) fn provider_policy_workers(context: KernelContext<'_>) -> NonZeroUsize {
    policy_workers(context.policy(), context)
}

/// Provider worker budget under an explicit `policy`.
#[cfg(feature = "blas")]
pub(crate) fn policy_workers(policy: ExecutionPolicy, context: KernelContext<'_>) -> NonZeroUsize {
    let budget = context.execution().worker_budget();
    match policy {
        ExecutionPolicy::Sequential => NonZeroUsize::MIN,
        ExecutionPolicy::Auto => budget,
        ExecutionPolicy::Parallel(workers) => workers.min(budget),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    #[cfg(not(feature = "std"))]
    use scioxide_core::PortableProcessor;
    use scioxide_core::ProcessorCapabilities;
    #[cfg(feature = "std")]
    use scioxide_core::StdProcessor;
    #[cfg(feature = "std")]
    static NATIVE_PROCESSOR: StdProcessor = StdProcessor;
    #[cfg(not(feature = "std"))]
    static NATIVE_PROCESSOR: PortableProcessor = PortableProcessor;

    pub(crate) fn reference_gemm<S: NumericScalar>(
        left: MatrixView<'_, S>,
        right: MatrixView<'_, S>,
        mut output: MatrixViewMut<'_, S>,
        config: GemmConfig<S>,
    ) -> Result<(), SciOxideError> {
        let left = operated_view(left, config.left_op);
        let right = operated_view(right, config.right_op);
        let (left_rows, left_cols) = (left.rows, left.cols);
        let (right_rows, right_cols) = (right.rows, right.cols);
        validate_dimensions(left_rows, left_cols, right_rows, right_cols, &output)?;
        for row in 0..left_rows {
            for col in 0..right_cols {
                if config
                    .triangle
                    .is_some_and(|triangle| !triangle_contains(triangle, row, col))
                {
                    continue;
                }
                let mut result = ProductSum::new();
                if config.alpha != S::ZERO {
                    for inner in 0..left_cols {
                        result.add_triple_product(
                            config.alpha,
                            left.get_element(row, inner),
                            right.get_element(inner, col),
                        )?;
                    }
                }
                if config.beta != S::ZERO {
                    result.add_product(config.beta, output.get_element(row, col))?;
                }
                output.set_element(row, col, result.finish()?);
            }
        }
        Ok(())
    }

    pub(crate) fn test_context() -> ExecutionContext<'static> {
        ExecutionContext::sequential().with_processor(test_processor())
    }

    pub(crate) fn test_processor() -> &'static dyn ProcessorCapabilities {
        &NATIVE_PROCESSOR
    }

    use core::time::Duration;

    use scioxide_core::{CapabilityError, Clock, Complex, Deadline, MonotonicInstant};
    use scioxide_core::{ExecutionPlan, Executor, PartitionFn};

    use super::*;
    use crate::dense::Matrix;

    pub(crate) fn prepare_product<'a, S: NumericScalar>(
        workspace: &mut LinalgWorkspace<'_, S>,
        rows: usize,
        shared: usize,
        cols: usize,
        policy: ExecutionPolicy,
        context: impl Into<KernelContext<'a>>,
    ) -> Result<(), SciOxideError> {
        let context = context.into();
        let options = crate::LinalgExecutionOptions::new()
            .with_execution_policy(policy)
            .with_accuracy(context.accuracy());
        workspace.try_reserve(&product_requirements::<S>(
            &DenseWorkspaceRequest::product(rows, shared, cols, &options).for_kernels(context),
        )?)
    }

    pub(crate) fn prepared_gemm<S: NumericScalar>(
        left: FiniteMatrixView<'_, S>,
        right: FiniteMatrixView<'_, S>,
        output: MatrixViewMut<'_, S>,
        config: GemmConfig<S>,
        policy: ExecutionPolicy,
        workspace: &mut LinalgWorkspace<'_, S>,
        context: KernelContext<'_>,
    ) -> Result<(), SciOxideError> {
        prepare_product(
            workspace,
            left.rows(),
            left.cols(),
            right.cols(),
            policy,
            context,
        )?;
        super::gemm_bounded(left, right, output, config, policy, workspace, context)
    }

    #[test]
    fn underdeclared_packing_fails_without_growth_or_publication() -> Result<(), SciOxideError> {
        let left = matrix_f64(35, 39, 1);
        let right = matrix_f64(39, 37, 2);
        let mut output = Matrix::from_vec(35, 37, alloc::vec![7.0; 35 * 37])?;
        let mut requirements = product_requirements::<f64>(&DenseWorkspaceRequest::product(
            35,
            39,
            37,
            &crate::LinalgExecutionOptions::new(),
        ))?;
        *requirements.scalar_mut(PACKED_A) = 0;
        let mut workspace = LinalgWorkspace::new();
        workspace.try_reserve(&requirements)?;
        let bytes = workspace.retained_bytes()?;
        let result = super::gemm_bounded(
            left.view().try_finite()?,
            right.view().try_finite()?,
            output.view_mut(),
            GemmConfig::ordinary(),
            ExecutionPolicy::Sequential,
            &mut workspace,
            KernelContext::new(test_context()),
        );
        assert!(result.err().is_some_and(|error| matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        )));
        assert_eq!(workspace.retained_bytes()?, bytes);
        assert!(output.as_slice().iter().all(|&value| value == 7.0));
        Ok(())
    }

    fn matrix_f64(rows: usize, cols: usize, phase: usize) -> Matrix<f64> {
        let values = (0..rows * cols)
            .map(|index| {
                let centered = ((index * 13 + phase * 7) % 29) as f64 - 14.0;
                centered / 17.0
            })
            .collect();
        Matrix::from_vec(rows, cols, values).unwrap()
    }

    #[test]
    fn standard_affine_gemm_checks_intermediate_scaling_and_reuses_failed_candidate() {
        for n in [2, 65] {
            let mut workspace = LinalgWorkspace::new();
            let mut output = Matrix::from_vec(n, n, alloc::vec![7.0_f64;n*n]).unwrap();
            for (a, b, alpha, expected) in [
                (1e200, 1e-200, 1e200, Some(2e200)),
                (f64::MAX, 2.0, 1.0, None),
                (2.0, 3.0, 1.0, Some(12.0)),
            ] {
                let left = Matrix::from_vec(n, 2, alloc::vec![a;n*2]).unwrap();
                let right = Matrix::from_vec(2, n, alloc::vec![b;n*2]).unwrap();
                let before = output.clone();
                let result = (|| -> Result<(), SciOxideError> {
                    let config = GemmConfig {
                        alpha,
                        ..GemmConfig::ordinary()
                    };
                    crate::kernel::tests::prepared_gemm(
                        operated_view(left.view(), config.left_op).try_finite()?,
                        operated_view(right.view(), config.right_op).try_finite()?,
                        output.view_mut(),
                        config,
                        ExecutionPolicy::Sequential,
                        &mut workspace,
                        KernelContext::from(test_context()),
                    )
                })();
                if let Some(expected) = expected {
                    result.unwrap();
                    for &value in output.as_slice() {
                        assert!((value / expected - 1.0).abs() < 1e-14);
                    }
                } else {
                    assert!(result.is_err());
                    assert_eq!(output, before);
                }
            }
        }
    }

    #[test]
    fn standard_gemm_normalizes_only_overflowing_entries() {
        // Even rows multiply MAX by MIN_POSITIVE, which is ordinary in the unit
        // frame; odd rows overflow there but cancel once normalized. Two rows
        // use the direct product and 64 rows the packed kernels.
        let right = Matrix::from_vec(4, 1, vec![f64::MIN_POSITIVE, f64::MAX, 2.0, 2.0]).unwrap();
        let standard = KernelContext::new(test_context()).with_accuracy(Accuracy::Standard);
        for rows in [2, 64] {
            let values = (0..rows)
                .flat_map(|row| {
                    if row % 2 == 0 {
                        [f64::MAX, f64::MIN_POSITIVE, 0.0, 0.0]
                    } else {
                        [0.0, 0.0, f64::MAX, -f64::MAX]
                    }
                })
                .collect();
            let left = Matrix::from_vec(rows, 4, values).unwrap();
            let check = |output: &Matrix<f64>| {
                for row in 0..rows {
                    let expected = if row % 2 == 0 {
                        0x401f_ffff_ffff_ffff
                    } else {
                        0
                    };
                    assert_eq!(
                        output.get_element(row, 0).to_bits(),
                        expected,
                        "{rows}:{row}"
                    );
                }
            };
            let mut output = Matrix::<f64>::zeros(rows, 1).unwrap();
            left.view()
                .mul_mat_into(
                    right.view(),
                    &crate::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential),
                    output.view_mut(),
                    &mut LinalgWorkspace::new(),
                    crate::LinalgExecutionContext::unscoped(standard),
                )
                .unwrap();
            check(&output);
            let mut output = Matrix::<f64>::zeros(rows, 1).unwrap();
            gemm_workspace_with_context(
                WorkspaceGemmProblem {
                    left: FiniteMatrixView::from_workspace(left.view()),
                    right: FiniteMatrixView::from_workspace(right.view()),
                    output: output.view_mut(),
                    config: GemmConfig::ordinary(),
                },
                ExecutionPolicy::Sequential,
                &mut LinalgWorkspace::workspace(&DenseWorkspaceRequest::product(
                    rows,
                    4,
                    1,
                    &crate::LinalgExecutionOptions::new(),
                ))
                .unwrap(),
                standard,
            )
            .unwrap();
            check(&output);
        }
    }

    fn matrix_complex(rows: usize, cols: usize, phase: usize) -> Matrix<Complex<f64>> {
        let values = (0..rows * cols)
            .map(|index| {
                let real = ((index * 11 + phase * 5) % 23) as f64 - 11.0;
                let imaginary = ((index * 7 + phase * 3) % 19) as f64 - 9.0;
                Complex::new(real / 13.0, imaginary / 17.0)
            })
            .collect();
        Matrix::from_vec(rows, cols, values).unwrap()
    }

    #[derive(Debug)]
    struct RejectingExecutor;

    impl Executor for RejectingExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(4).unwrap()
        }

        fn execute(
            &self,
            _context: ExecutionContext<'_>,
            _plan: ExecutionPlan,
            _operation: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            Err(SciOxideError::execution_failure(
                None,
                scioxide_core::ExecutionFailureReason::ProviderFailure,
                "injected executor rejection",
            ))
        }
    }

    #[derive(Debug)]
    struct FailingClock;

    impl Clock for FailingClock {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            Err(CapabilityError::new("injected clock failure"))
        }
    }

    fn assert_f64_close(actual: &Matrix<f64>, expected: &Matrix<f64>, tolerance: f64) {
        actual
            .as_slice()
            .iter()
            .zip(expected.as_slice())
            .for_each(|(&actual, &expected)| {
                let scale = 1.0 + expected.abs();
                assert!((actual - expected).abs() <= tolerance * scale);
            });
    }

    #[test]
    fn small_gemm_reservation_uses_only_transactional_candidate_storage() {
        let mut workspace = LinalgWorkspace::<f64>::new();
        crate::kernel::tests::prepare_product(
            &mut workspace,
            SMALL_GEMM_DIMENSION,
            SMALL_GEMM_DIMENSION,
            SMALL_GEMM_DIMENSION,
            ExecutionPolicy::Sequential,
            ExecutionContext::sequential(),
        )
        .unwrap();

        assert_eq!(workspace.scalar_mut(crate::kernel::PACKED_A).len(), 0);
        assert_eq!(workspace.scalar_mut(crate::kernel::PACKED_B).len(), 0);
    }

    fn assert_f32_close(actual: &Matrix<f32>, expected: &Matrix<f32>, tolerance: f32) {
        actual
            .as_slice()
            .iter()
            .zip(expected.as_slice())
            .for_each(|(&actual, &expected)| {
                let scale = 1.0 + expected.abs();
                assert!((actual - expected).abs() <= tolerance * scale);
            });
    }

    fn assert_complex_close(
        actual: &Matrix<Complex<f64>>,
        expected: &Matrix<Complex<f64>>,
        tolerance: f64,
    ) {
        actual
            .as_slice()
            .iter()
            .zip(expected.as_slice())
            .for_each(|(&actual, &expected)| {
                let scale = 1.0 + expected.norm();
                assert!((actual - expected).norm() <= tolerance * scale);
            });
    }

    fn stored_shape(rows: usize, cols: usize, op: GemmOp) -> (usize, usize) {
        match op {
            GemmOp::None => (rows, cols),
            GemmOp::Transpose | GemmOp::Adjoint => (cols, rows),
        }
    }

    #[test]
    fn transactional_f64_matches_reference_for_rectangular_tail() {
        let left = matrix_f64(37, 35, 1);
        let right = matrix_f64(35, 41, 2);
        let mut actual = Matrix::zeros(37, 41).unwrap();
        let mut expected = Matrix::zeros(37, 41).unwrap();
        let config = GemmConfig {
            backend: Some(KernelBackend::Scalar),
            ..GemmConfig::ordinary()
        };
        let mut workspace = LinalgWorkspace::new();

        (|| -> Result<(), SciOxideError> {
            crate::kernel::tests::prepared_gemm(
                operated_view(left.view(), config.left_op).try_finite()?,
                operated_view(right.view(), config.right_op).try_finite()?,
                actual.view_mut(),
                config,
                ExecutionPolicy::Sequential,
                &mut workspace,
                KernelContext::new(crate::kernel::tests::test_context())
                    .with_accuracy(Accuracy::Exact),
            )
        })()
        .unwrap();
        reference_gemm(left.view(), right.view(), expected.view_mut(), config).unwrap();

        assert_f64_close(&actual, &expected, 2.0e-14);
    }

    #[test]
    fn every_operation_pair_matches_complex_reference() {
        let operations = [GemmOp::None, GemmOp::Transpose, GemmOp::Adjoint];
        for left_op in operations {
            for right_op in operations {
                let (left_rows, left_cols) = stored_shape(33, 35, left_op);
                let (right_rows, right_cols) = stored_shape(35, 34, right_op);
                let left = matrix_complex(left_rows, left_cols, 3);
                let right = matrix_complex(right_rows, right_cols, 5);
                let initial = matrix_complex(33, 34, 7);
                let mut actual = initial.clone();
                let mut expected = initial.clone();
                let config = GemmConfig {
                    #[cfg(feature = "blas")]
                    rank: None,
                    left_op,
                    right_op,
                    alpha: Complex::new(0.75, -0.25),
                    beta: Complex::new(-0.5, 0.125),
                    backend: Some(KernelBackend::Scalar),
                    triangle: None,
                    packed: false,
                    prior: None,
                };
                let mut workspace = LinalgWorkspace::new();

                (|| -> Result<(), SciOxideError> {
                    crate::kernel::tests::prepared_gemm(
                        operated_view(left.view(), config.left_op).try_finite()?,
                        operated_view(right.view(), config.right_op).try_finite()?,
                        actual.view_mut(),
                        config,
                        ExecutionPolicy::Sequential,
                        &mut workspace,
                        KernelContext::new(crate::kernel::tests::test_context())
                            .with_accuracy(Accuracy::Exact),
                    )
                })()
                .unwrap();
                reference_gemm(left.view(), right.view(), expected.view_mut(), config).unwrap();

                assert_complex_close(&actual, &expected, 3.0e-14);
            }
        }
    }

    #[test]
    fn every_available_backend_matches_scalar_for_unaligned_tails() {
        let left = matrix_f64(39, 37, 11);
        let right = matrix_f64(37, 43, 13);
        let mut expected = Matrix::zeros(39, 43).unwrap();
        let scalar = GemmConfig {
            backend: Some(KernelBackend::Scalar),
            ..GemmConfig::ordinary()
        };
        reference_gemm(left.view(), right.view(), expected.view_mut(), scalar).unwrap();

        for backend in KernelBackend::available(test_processor()) {
            let mut actual = Matrix::zeros(39, 43).unwrap();
            let mut workspace = LinalgWorkspace::new();
            let config = GemmConfig {
                backend: Some(backend),
                ..GemmConfig::ordinary()
            };
            (|| -> Result<(), SciOxideError> {
                crate::kernel::tests::prepared_gemm(
                    operated_view(left.view(), config.left_op).try_finite()?,
                    operated_view(right.view(), config.right_op).try_finite()?,
                    actual.view_mut(),
                    config,
                    ExecutionPolicy::Sequential,
                    &mut workspace,
                    KernelContext::new(crate::kernel::tests::test_context())
                        .with_accuracy(Accuracy::Exact),
                )
            })()
            .unwrap();
            assert_f64_close(&actual, &expected, 5.0e-13);
        }
    }

    #[test]
    fn every_available_f32_backend_matches_scalar_reference() {
        let left = Matrix::from_vec(
            41,
            35,
            (0..41 * 35)
                .map(|index| ((index * 7 + 3) % 23) as f32 / 19.0 - 0.5)
                .collect(),
        )
        .unwrap();
        let right = Matrix::from_vec(
            35,
            38,
            (0..35 * 38)
                .map(|index| ((index * 11 + 5) % 29) as f32 / 23.0 - 0.5)
                .collect(),
        )
        .unwrap();
        let mut expected = Matrix::zeros(41, 38).unwrap();
        reference_gemm(
            left.view(),
            right.view(),
            expected.view_mut(),
            GemmConfig::ordinary(),
        )
        .unwrap();

        for backend in KernelBackend::available(test_processor()) {
            let mut actual = Matrix::zeros(41, 38).unwrap();
            let mut workspace = LinalgWorkspace::new();
            (|| -> Result<(), SciOxideError> {
                let config = GemmConfig {
                    backend: Some(backend),
                    ..GemmConfig::ordinary()
                };
                crate::kernel::tests::prepared_gemm(
                    operated_view(left.view(), config.left_op).try_finite()?,
                    operated_view(right.view(), config.right_op).try_finite()?,
                    actual.view_mut(),
                    config,
                    ExecutionPolicy::Sequential,
                    &mut workspace,
                    KernelContext::new(crate::kernel::tests::test_context())
                        .with_accuracy(Accuracy::Exact),
                )
            })()
            .unwrap();
            assert_f32_close(&actual, &expected, 2.0e-5);
        }
    }

    #[test]
    fn transactional_f64_updates_strided_rows_with_alpha_and_beta() {
        let rows = 37;
        let shared = 8;
        let cols = 35;
        let output_stride = cols + 7;
        let left = matrix_f64(rows, shared, 17);
        let right = matrix_f64(shared, cols, 19);
        let mut expected = Matrix::from_vec(rows, cols, vec![0.25; rows * cols]).unwrap();
        let config = GemmConfig {
            alpha: -0.75,
            beta: 0.5,
            ..GemmConfig::ordinary()
        };
        reference_gemm(left.view(), right.view(), expected.view_mut(), config).unwrap();

        for backend in KernelBackend::available(test_processor()) {
            let output_len = (rows - 1) * output_stride + cols;
            let mut output_data = vec![73.0; output_len];
            (0..rows).for_each(|row| {
                output_data[row * output_stride..row * output_stride + cols].fill(0.25);
            });
            let output = MatrixViewMut::from_slice_with_strides(
                rows,
                cols,
                &mut output_data,
                output_stride,
                1,
            )
            .unwrap();
            let mut workspace = LinalgWorkspace::new();
            (|| -> Result<(), SciOxideError> {
                let mut config = config;
                config.backend = Some(backend);
                crate::kernel::tests::prepared_gemm(
                    operated_view(left.view(), config.left_op).try_finite()?,
                    operated_view(right.view(), config.right_op).try_finite()?,
                    output,
                    config,
                    ExecutionPolicy::Sequential,
                    &mut workspace,
                    KernelContext::new(crate::kernel::tests::test_context())
                        .with_accuracy(Accuracy::Exact),
                )
            })()
            .unwrap();

            (0..rows).for_each(|row| {
                (0..cols).for_each(|col| {
                    let actual = output_data[row * output_stride + col];
                    let expected = expected.get_element(row, col);
                    assert!((actual - expected).abs() <= 5.0e-13 * (1.0 + expected.abs()));
                });
                if row + 1 < rows {
                    output_data[row * output_stride + cols..(row + 1) * output_stride]
                        .iter()
                        .for_each(|&value| assert_eq!(value, 73.0));
                }
            });
        }
    }

    #[test]
    fn transactional_f32_beta_zero_ignores_prior_output() {
        let rows = 39;
        let shared = 8;
        let cols = 37;
        let left = Matrix::from_vec(
            rows,
            shared,
            (0..rows * shared)
                .map(|index| ((index * 7 + 3) % 23) as f32 / 19.0 - 0.5)
                .collect(),
        )
        .unwrap();
        let right = Matrix::from_vec(
            shared,
            cols,
            (0..shared * cols)
                .map(|index| ((index * 11 + 5) % 29) as f32 / 23.0 - 0.5)
                .collect(),
        )
        .unwrap();
        let config = GemmConfig {
            alpha: 0.625,
            ..GemmConfig::ordinary()
        };
        let mut expected = Matrix::zeros(rows, cols).unwrap();
        reference_gemm(left.view(), right.view(), expected.view_mut(), config).unwrap();

        for backend in KernelBackend::available(test_processor()) {
            let output_stride = cols + 5;
            let output_len = (rows - 1) * output_stride + cols;
            let mut output_data = vec![f32::NAN; output_len];
            let output = MatrixViewMut::from_slice_with_strides(
                rows,
                cols,
                &mut output_data,
                output_stride,
                1,
            )
            .unwrap();
            let mut workspace = LinalgWorkspace::new();
            (|| -> Result<(), SciOxideError> {
                let mut config = config;
                config.backend = Some(backend);
                crate::kernel::tests::prepared_gemm(
                    operated_view(left.view(), config.left_op).try_finite()?,
                    operated_view(right.view(), config.right_op).try_finite()?,
                    output,
                    config,
                    ExecutionPolicy::Sequential,
                    &mut workspace,
                    KernelContext::new(crate::kernel::tests::test_context())
                        .with_accuracy(Accuracy::Exact),
                )
            })()
            .unwrap();

            (0..rows).for_each(|row| {
                (0..cols).for_each(|col| {
                    let actual = output_data[row * output_stride + col];
                    let expected = expected.get_element(row, col);
                    assert!((actual - expected).abs() <= 2.0e-5 * (1.0 + expected.abs()));
                });
            });
        }
    }

    #[test]
    fn transactional_complex_f32_adjoint_matches_reference() {
        let left = Matrix::from_vec(
            35,
            37,
            (0..35 * 37)
                .map(|index| {
                    Complex::new(
                        ((index * 5 + 2) % 17) as f32 / 13.0 - 0.5,
                        ((index * 7 + 1) % 19) as f32 / 17.0 - 0.5,
                    )
                })
                .collect(),
        )
        .unwrap();
        let right = Matrix::from_vec(
            35,
            34,
            (0..35 * 34)
                .map(|index| {
                    Complex::new(
                        ((index * 11 + 3) % 23) as f32 / 19.0 - 0.5,
                        ((index * 13 + 4) % 29) as f32 / 23.0 - 0.5,
                    )
                })
                .collect(),
        )
        .unwrap();
        let initial = Matrix::from_vec(37, 34, vec![Complex::new(0.25, -0.5); 37 * 34]).unwrap();
        let mut actual = initial.clone();
        let mut expected = initial;
        let config = GemmConfig {
            #[cfg(feature = "blas")]
            rank: None,
            left_op: GemmOp::Adjoint,
            right_op: GemmOp::None,
            alpha: Complex::new(0.75, -0.25),
            beta: Complex::new(-0.5, 0.125),
            backend: Some(KernelBackend::Scalar),
            triangle: None,
            packed: false,
            prior: None,
        };
        let mut workspace = LinalgWorkspace::new();
        (|| -> Result<(), SciOxideError> {
            crate::kernel::tests::prepared_gemm(
                operated_view(left.view(), config.left_op).try_finite()?,
                operated_view(right.view(), config.right_op).try_finite()?,
                actual.view_mut(),
                config,
                ExecutionPolicy::Sequential,
                &mut workspace,
                KernelContext::new(crate::kernel::tests::test_context())
                    .with_accuracy(Accuracy::Exact),
            )
        })()
        .unwrap();
        reference_gemm(left.view(), right.view(), expected.view_mut(), config).unwrap();
        actual
            .as_slice()
            .iter()
            .zip(expected.as_slice())
            .for_each(|(&actual, &expected)| {
                assert!((actual - expected).norm() <= 3.0e-5 * (1.0 + expected.norm()));
            });
    }

    #[test]
    fn transactional_strided_views_preserve_padding() {
        let rows = 37;
        let shared = 35;
        let cols = 39;
        let left_stride = shared + 3;
        let right_stride = cols + 5;
        let output_row_stride = cols * 2 + 3;
        let mut left_data = vec![f64::NAN; (rows - 1) * left_stride + shared];
        let mut right_data = vec![f64::NAN; (shared - 1) * right_stride + cols];
        for row in 0..rows {
            for col in 0..shared {
                left_data[row * left_stride + col] = ((row * 7 + col * 3) % 17) as f64 / 13.0;
            }
        }
        for row in 0..shared {
            for col in 0..cols {
                right_data[row * right_stride + col] = ((row * 11 + col * 5) % 19) as f64 / 17.0;
            }
        }
        let left =
            MatrixView::from_slice_with_strides(rows, shared, &left_data, left_stride, 1).unwrap();
        let right = MatrixView::from_slice_with_strides(shared, cols, &right_data, right_stride, 1)
            .unwrap();
        let mut expected = Matrix::zeros(rows, cols).unwrap();
        reference_gemm(left, right, expected.view_mut(), GemmConfig::ordinary()).unwrap();

        for backend in KernelBackend::available(test_processor()) {
            let mut contiguous = Matrix::zeros(rows, cols).unwrap();
            let mut workspace = LinalgWorkspace::new();
            (|| -> Result<(), SciOxideError> {
                let config = GemmConfig {
                    backend: Some(backend),
                    ..GemmConfig::ordinary()
                };
                crate::kernel::tests::prepared_gemm(
                    operated_view(left, config.left_op).try_finite()?,
                    operated_view(right, config.right_op).try_finite()?,
                    contiguous.view_mut(),
                    config,
                    ExecutionPolicy::Sequential,
                    &mut workspace,
                    KernelContext::new(crate::kernel::tests::test_context())
                        .with_accuracy(Accuracy::Exact),
                )
            })()
            .unwrap();
            assert_f64_close(&contiguous, &expected, 5.0e-13);
        }

        let output_len = (rows - 1) * output_row_stride + (cols - 1) * 2 + 1;
        let mut output_data = vec![73.0; output_len];
        let mut workspace = LinalgWorkspace::new();
        let output = MatrixViewMut::from_slice_with_strides(
            rows,
            cols,
            &mut output_data,
            output_row_stride,
            2,
        )
        .unwrap();
        left.mul_mat_into(
            right,
            &crate::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            output,
            &mut workspace,
            crate::LinalgExecutionContext::unscoped(
                KernelContext::new(crate::kernel::tests::test_context())
                    .with_accuracy(Accuracy::Exact),
            ),
        )
        .unwrap();
        for row in 0..rows {
            for col in 0..cols {
                assert!(
                    (output_data[row * output_row_stride + col * 2]
                        - expected.get_element(row, col))
                    .abs()
                        <= 5.0e-13 * (1.0 + expected.get_element(row, col).abs())
                );
            }
        }
        output_data.iter().enumerate().for_each(|(index, &value)| {
            let row = index / output_row_stride;
            let within_row = index % output_row_stride;
            let logical = row < rows && within_row < cols * 2 && within_row.is_multiple_of(2);
            if !logical {
                assert_eq!(value, 73.0);
            }
        });
    }

    #[test]
    fn adversarial_finite_scales_match_scalar_reference() {
        let dimension = 35;
        let left = Matrix::from_vec(
            dimension,
            dimension,
            (0..dimension * dimension)
                .map(|index| if index % 2 == 0 { 1.0e150 } else { -1.0e150 })
                .collect(),
        )
        .unwrap();
        let right = Matrix::from_vec(
            dimension,
            dimension,
            (0..dimension * dimension)
                .map(|index| if index % 3 == 0 { 1.0e-150 } else { -1.0e-150 })
                .collect(),
        )
        .unwrap();
        let mut expected = Matrix::zeros(dimension, dimension).unwrap();
        reference_gemm(
            left.view(),
            right.view(),
            expected.view_mut(),
            GemmConfig::ordinary(),
        )
        .unwrap();
        for backend in KernelBackend::available(test_processor()) {
            let mut actual = Matrix::zeros(dimension, dimension).unwrap();
            let mut workspace = LinalgWorkspace::new();
            (|| -> Result<(), SciOxideError> {
                let config = GemmConfig {
                    backend: Some(backend),
                    ..GemmConfig::ordinary()
                };
                crate::kernel::tests::prepared_gemm(
                    operated_view(left.view(), config.left_op).try_finite()?,
                    operated_view(right.view(), config.right_op).try_finite()?,
                    actual.view_mut(),
                    config,
                    ExecutionPolicy::Sequential,
                    &mut workspace,
                    KernelContext::new(crate::kernel::tests::test_context())
                        .with_accuracy(Accuracy::Exact),
                )
            })()
            .unwrap();
            assert_f64_close(&actual, &expected, 2.0e-14);
        }
    }

    #[test]
    fn alpha_zero_still_checks_inputs_and_beta_zero_does_not_read_output() {
        let left = matrix_f64(33, 33, 13);
        let right = matrix_f64(33, 33, 15);
        let mut output = Matrix::from_vec(33, 33, vec![2.0; 33 * 33]).unwrap();
        let mut workspace = LinalgWorkspace::new();
        let config = GemmConfig {
            alpha: 0.0,
            beta: 3.0,
            ..GemmConfig::ordinary()
        };
        (|| -> Result<(), SciOxideError> {
            crate::kernel::tests::prepared_gemm(
                operated_view(left.view(), config.left_op).try_finite()?,
                operated_view(right.view(), config.right_op).try_finite()?,
                output.view_mut(),
                config,
                ExecutionPolicy::Sequential,
                &mut workspace,
                KernelContext::new(crate::kernel::tests::test_context())
                    .with_accuracy(Accuracy::Exact),
            )
        })()
        .unwrap();
        assert!(output.as_slice().iter().all(|&value| value == 6.0));

        let non_finite = Matrix::from_vec(33, 33, vec![f64::NAN; 33 * 33]).unwrap();
        let original = output.clone();
        assert!(matches!(
            ((|| -> Result<(), SciOxideError> {
                crate::kernel::tests::prepared_gemm(
                    operated_view(non_finite.view(), config.left_op).try_finite()?,
                    operated_view(right.view(), config.right_op).try_finite()?,
                    output.view_mut(),
                    config,
                    ExecutionPolicy::Sequential,
                    &mut workspace,
                    KernelContext::new(crate::kernel::tests::test_context())
                        .with_accuracy(Accuracy::Exact),
                )
            })())
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
        assert_eq!(output, original);

        let left = matrix_f64(33, 33, 17);
        let right = matrix_f64(33, 33, 19);
        output.view_mut().fill(f64::NAN);
        left.view()
            .mul_mat_into(
                right.view(),
                &crate::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential),
                output.view_mut(),
                &mut workspace,
                crate::LinalgExecutionContext::unscoped(
                    KernelContext::new(crate::kernel::tests::test_context())
                        .with_accuracy(Accuracy::Exact),
                ),
            )
            .unwrap();
        assert!(output.as_slice().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn scaled_gemm_does_not_materialize_an_overflowing_inner_product() {
        let left = Matrix::from_vec(1, 1, vec![f64::MAX]).unwrap();
        let right = Matrix::from_vec(1, 1, vec![2.0]).unwrap();
        let mut output = Matrix::from_vec(1, 1, vec![f64::MAX]).unwrap();
        let mut workspace = LinalgWorkspace::new();
        (|| -> Result<(), SciOxideError> {
            let config = GemmConfig {
                alpha: 0.25,
                beta: -0.25,
                ..GemmConfig::ordinary()
            };
            crate::kernel::tests::prepared_gemm(
                operated_view(left.view(), config.left_op).try_finite()?,
                operated_view(right.view(), config.right_op).try_finite()?,
                output.view_mut(),
                config,
                ExecutionPolicy::Sequential,
                &mut workspace,
                KernelContext::new(crate::kernel::tests::test_context())
                    .with_accuracy(Accuracy::Exact),
            )
        })()
        .unwrap();
        assert_eq!(output.get_element(0, 0), f64::MAX / 4.0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn parallel_partition_matches_sequential() {
        let left = matrix_f64(257, 131, 23);
        let right = matrix_f64(131, 137, 29);
        let mut sequential = Matrix::zeros(257, 137).unwrap();
        let mut parallel = Matrix::zeros(257, 137).unwrap();
        let mut sequential_workspace = LinalgWorkspace::new();
        let mut parallel_workspace = LinalgWorkspace::new();
        let config = GemmConfig {
            backend: Some(KernelBackend::Scalar),
            ..GemmConfig::ordinary()
        };
        (|| -> Result<(), SciOxideError> {
            crate::kernel::tests::prepared_gemm(
                operated_view(left.view(), config.left_op).try_finite()?,
                operated_view(right.view(), config.right_op).try_finite()?,
                sequential.view_mut(),
                config,
                ExecutionPolicy::Sequential,
                &mut sequential_workspace,
                KernelContext::new(crate::kernel::tests::test_context())
                    .with_accuracy(Accuracy::Exact),
            )
        })()
        .unwrap();
        let executor = scioxide_core::ThreadExecutor::new(NonZeroUsize::new(4).unwrap());
        let context = ExecutionContext::new(&executor).with_processor(test_processor());
        (|| -> Result<(), SciOxideError> {
            crate::kernel::tests::prepared_gemm(
                operated_view(left.view(), config.left_op).try_finite()?,
                operated_view(right.view(), config.right_op).try_finite()?,
                parallel.view_mut(),
                config,
                ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap()),
                &mut parallel_workspace,
                KernelContext::new(context).with_accuracy(Accuracy::Exact),
            )
        })()
        .unwrap();
        assert_eq!(parallel.as_slice(), sequential.as_slice());
    }

    #[test]
    fn executor_rejection_precedes_gemm_commit() {
        let left = matrix_f64(257, 33, 53);
        let right = matrix_f64(33, 7, 59);
        let mut output = Matrix::from_vec(257, 7, vec![9.0; 257 * 7]).unwrap();
        let original = output.clone();
        let mut workspace = LinalgWorkspace::new();
        let executor = RejectingExecutor;
        let context = ExecutionContext::new(&executor).with_processor(test_processor());

        assert!(matches!(
            (left.view().mul_mat_into(
                right.view(),
                &crate::LinalgExecutionOptions::new().with_execution_policy(
                    ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap())
                ),
                output.view_mut(),
                &mut workspace,
                context
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::ExecutionFailure { .. }
        ));
        assert_eq!(output, original);
    }

    #[test]
    fn capability_provider_failure_precedes_gemm_commit() {
        let left = matrix_f64(3, 2, 61);
        let right = matrix_f64(2, 4, 67);
        let mut output = Matrix::from_vec(3, 4, vec![9.0; 12]).unwrap();
        let original = output.clone();
        let mut workspace = LinalgWorkspace::new();
        let clock = FailingClock;
        let clock_binding = scioxide_core::ClockBinding::new(&clock);
        let deadline = Deadline::at(
            &clock_binding,
            MonotonicInstant::from_duration_since_epoch(Duration::ZERO),
        );
        let context = ExecutionContext::sequential()
            .with_deadline(deadline)
            .unwrap();

        assert!(matches!(
            (left.view().mul_mat_into(
                right.view(),
                &crate::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential),
                output.view_mut(),
                &mut workspace,
                context
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::CapabilityFailure { .. }
        ));
        assert_eq!(output, original);
    }

    #[test]
    fn broadcast_stride_is_supported_and_workspace_overflow_is_structural() {
        let left_data = [2.0_f64];
        let left = MatrixView::from_slice_with_strides(33, 33, &left_data, 0, 0).unwrap();
        let right = matrix_f64(33, 33, 31);
        let mut output = Matrix::from_vec(33, 33, vec![7.0; 33 * 33]).unwrap();
        let original = output.clone();
        let mut workspace = LinalgWorkspace::new();
        left.mul_mat_into(
            right.view(),
            &crate::LinalgExecutionOptions::new()
                .with_execution_policy(ExecutionPolicy::Sequential),
            output.view_mut(),
            &mut workspace,
            crate::LinalgExecutionContext::unscoped(
                KernelContext::new(crate::kernel::tests::test_context())
                    .with_accuracy(Accuracy::Exact),
            ),
        )
        .unwrap();
        assert_ne!(output, original);
        assert!(
            output
                .as_slice()
                .chunks_exact(33)
                .all(|row| row == &output.as_slice()[..33])
        );
        assert!(matches!(
            (crate::kernel::tests::prepare_product(
                &mut workspace,
                usize::MAX,
                2,
                2,
                ExecutionPolicy::Sequential,
                ExecutionContext::sequential()
            ))
            .as_ref()
            .err()
            .unwrap()
            .view(),
            scioxide_core::SciOxideErrorView::SizeOverflow { .. }
        ));
    }
}
