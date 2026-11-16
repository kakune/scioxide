//! Error-driven spatial adaptation; topology and solvers retain their existing owners.
use crate::validation::finite;
use crate::validation::invalid;

use alloc::vec::Vec;
use scioxide_core::backend::{
    PartitionSlot, checked_add as add, checked_mul as size, try_filled, try_with_capacity,
};
use scioxide_core::{
    AdjointOperator, BudgetKind, ExecutionContext, ExecutionPolicy, LinearOperator, NumericScalar,
    RealScalar, SciOxideError, SolveContext,
};
use scioxide_sparse::{CsrMatrix, DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};
use scioxide_spatial::{
    AdaptiveCellKind, AdaptiveMesh, Cell, Field, FieldView, MeshToken, TopologyMap,
};

mod diffusion;
mod estimate;
mod space;
mod transfer;
pub use diffusion::{AmrDiffusionData, AmrDiffusionPlan};
pub use estimate::{
    AmrEstimator, AmrEstimatorWorkspace, AmrIndicators, AmrMarking, AmrMarks, AmrResidual,
};
pub use space::{AmrNodalField, AmrNodalFieldView, AmrOrder, AmrSpace};
pub use transfer::{
    AmrBoundaryTags, AmrCellTransfer, AmrMassSolve, AmrNodalTransfer, AmrProjectionReport,
    AmrTransferWorkspace,
};

type Result<T> = core::result::Result<T, SciOxideError>;
/// Check the length and finiteness of the caller's `argument`.
fn validate<V: NumericScalar>(
    values: &[V],
    n: usize,
    argument: scioxide_core::Argument,
) -> Result<()> {
    if values.len() != n {
        return Err(invalid(argument, "incorrect field or vector length"));
    }
    if values.iter().any(|v| !v.has_finite_components()) {
        return Err(invalid(argument, "nonfinite field"));
    }
    Ok(())
}

/// Hard ceilings for one AMR preparation. Previously prepared objects and caller inputs are
/// excluded.
///
/// Exceeding a ceiling fails with `ResourceLimit` naming the quantity, the required
/// amount and the ceiling. Entry points run in a solve scope also charge its node
/// and evaluation budgets; an exhausted budget fails with `ConvergenceFailure`
/// (`ResourceLimit(Node)` or `EvaluationLimit`) carrying the budget report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmrLimits {
    max_dofs: usize,
    max_entries: usize,
    max_quadrature_nodes: usize,
    max_work: usize,
    max_storage_bytes: usize,
}
impl Default for AmrLimits {
    fn default() -> Self {
        Self::new(usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX)
    }
}

impl AmrLimits {
    /// Ceilings on full or independent nodal/cell counts, generated constraint or
    /// matrix entries (including temporary triplets), nodes of one quadrature rule,
    /// charged visits/comparisons/contractions and cumulative requested bytes.
    pub const fn new(
        max_dofs: usize,
        max_entries: usize,
        max_quadrature_nodes: usize,
        max_work: usize,
        max_storage_bytes: usize,
    ) -> Self {
        Self {
            max_dofs,
            max_entries,
            max_quadrature_nodes,
            max_work,
            max_storage_bytes,
        }
    }
    /// Maximum full or independent nodal/cell count.
    pub const fn max_dofs(&self) -> usize {
        self.max_dofs
    }
    /// Maximum generated constraint or matrix entries.
    pub const fn max_entries(&self) -> usize {
        self.max_entries
    }
    /// Maximum nodes in a volume or face quadrature rule.
    pub const fn max_quadrature_nodes(&self) -> usize {
        self.max_quadrature_nodes
    }
    /// Maximum charged visits, comparisons and scalar contractions.
    pub const fn max_work(&self) -> usize {
        self.max_work
    }
    /// Maximum cumulative requested allocation payload.
    pub const fn max_storage_bytes(&self) -> usize {
        self.max_storage_bytes
    }
}
/// Shared numerical and scheduling choices for an immutable AMR space, transfer and
/// derived plans. The execution policy is the only worker setting; at most 32
/// workers are used.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct AmrOptions<R: RealScalar> {
    limits: AmrLimits,
    quadrature_degree: usize,
    minimum_reciprocal_condition: R,
    execution: ExecutionPolicy,
}
impl<R: RealScalar> Default for AmrOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}
impl<R: RealScalar> AmrOptions<R> {
    /// Degree-six quadrature, a `64 epsilon` geometry cutoff and sequential
    /// execution with no resource ceiling.
    pub fn new() -> Self {
        Self {
            limits: AmrLimits::default(),
            quadrature_degree: 6,
            minimum_reciprocal_condition: R::EPSILON * R::ONE.scalbn(6),
            execution: ExecutionPolicy::Sequential,
        }
    }
    /// Set preparation resource ceilings.
    pub const fn with_limits(mut self, limits: AmrLimits) -> Self {
        self.limits = limits;
        self
    }
    /// Polynomial degree integrated exactly by the cell and face rules.
    ///
    /// Simplex cells use the positive interior simplex rules of the conforming
    /// FEM ([`scioxide_integrate::SimplexRule::positive`]), degree 2 through 7;
    /// consistent mass requires at least twice the nodal order. Tensor cells use
    /// Gauss–Legendre products with `⌈(degree + 1)/2⌉` nodes per axis, degree 2
    /// through 31.
    pub const fn with_quadrature_degree(mut self, degree: usize) -> Self {
        self.quadrature_degree = degree;
        self
    }
    /// Minimum usable geometric reciprocal condition.
    pub const fn with_minimum_reciprocal_condition(mut self, cutoff: R) -> Self {
        self.minimum_reciprocal_condition = cutoff;
        self
    }
    /// Scheduling of independent rows and cells of every derived map and plan.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.execution = policy;
        self
    }
    /// Resource ceilings.
    pub const fn limits(&self) -> AmrLimits {
        self.limits
    }
    /// Polynomial exactness degree of the cell and face rules.
    pub const fn quadrature_degree(&self) -> usize {
        self.quadrature_degree
    }
    /// Geometry cutoff.
    pub const fn minimum_reciprocal_condition(&self) -> R {
        self.minimum_reciprocal_condition
    }
    /// Execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.execution
    }
}
/// Resource ledger of one AMR entry point.
struct Work<'a, 'b, 'c> {
    limits: AmrLimits,
    bytes: usize,
    work: usize,
    context: Option<&'a mut SolveContext<'b, 'c>>,
}
impl<'a, 'b, 'c> Work<'a, 'b, 'c> {
    const fn new(limits: AmrLimits, context: Option<&'a mut SolveContext<'b, 'c>>) -> Self {
        Self {
            limits,
            bytes: 0,
            work: 0,
            context,
        }
    }
    fn tick(&mut self, n: usize) -> Result<()> {
        self.work = add(self.work, n, scioxide_core::quantity!("amr_work"))?;
        if self.work > self.limits.max_work {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("amr_work"),
                self.work,
                self.limits.max_work,
            ));
        }
        if let Some(c) = self.context.as_deref_mut() {
            c.execution().check()?;
            c.try_consume(BudgetKind::Node, n)?.require()?;
        }
        Ok(())
    }
    /// Conservative comparison count for sorting `n` entries.
    fn sort(&mut self, n: usize) -> Result<()> {
        self.tick(size(n, search_work(n), scioxide_core::quantity!("work"))?)
    }
    fn charge<T>(&mut self, n: usize) -> Result<()> {
        const BYTES: scioxide_core::Quantity = scioxide_core::quantity!("storage_bytes");
        self.bytes = add(
            self.bytes,
            size(n, core::mem::size_of::<T>(), BYTES)?,
            BYTES,
        )?;
        if self.bytes > self.limits.max_storage_bytes {
            return Err(SciOxideError::resource_limit(
                BYTES,
                self.bytes,
                self.limits.max_storage_bytes,
            ));
        }
        Ok(())
    }
    fn filled<T: Clone>(
        &mut self,
        n: usize,
        value: T,
        quantity: scioxide_core::Quantity,
    ) -> Result<Vec<T>> {
        self.charge::<T>(n)?;
        try_filled(n, value, quantity)
    }
    fn vector<T>(&mut self, n: usize, quantity: scioxide_core::Quantity) -> Result<Vec<T>> {
        self.charge::<T>(n)?;
        try_with_capacity(n, quantity)
    }
    fn push<T>(
        &mut self,
        values: &mut Vec<T>,
        value: T,
        quantity: scioxide_core::Quantity,
    ) -> Result<()> {
        if values.len() == values.capacity() {
            let capacity = add(values.capacity(), values.capacity().max(4), quantity)?;
            self.charge::<T>(capacity)?;
            values
                .try_reserve_exact(capacity - values.len())
                .map_err(|_| SciOxideError::allocation_failure(quantity, capacity))?;
        }
        values.push(value);
        Ok(())
    }
    fn dofs(&self, n: usize) -> Result<()> {
        ceiling(
            scioxide_core::quantity!("amr_dofs"),
            n,
            self.limits.max_dofs,
        )
    }
    fn entries(&self, n: usize) -> Result<()> {
        ceiling(
            scioxide_core::quantity!("amr_entries"),
            n,
            self.limits.max_entries,
        )
    }
    fn provider(&mut self) -> Result<()> {
        self.tick(1)?;
        if let Some(c) = self.context.as_deref_mut() {
            c.record_provider(scioxide_core::EvaluationRole::Function)?
                .require()?;
        }
        Ok(())
    }
}

/// `ResourceLimit` error when `required` exceeds the [`AmrLimits`] ceiling `limit`.
const fn ceiling(resource: scioxide_core::Quantity, required: usize, limit: usize) -> Result<()> {
    if required > limit {
        Err(SciOxideError::resource_limit(resource, required, limit))
    } else {
        Ok(())
    }
}

/// Maximum comparisons in a binary search, including its final equality check.
fn search_work(n: usize) -> usize {
    (usize::BITS - n.leading_zeros()) as usize + 1
}

fn csr<V: NumericScalar>(
    rows: usize,
    cols: usize,
    entries: &[(usize, usize, V)],
    work: &mut Work<'_, '_, '_>,
) -> Result<CsrMatrix<V>> {
    const WORK: scioxide_core::Quantity = scioxide_core::quantity!("work");
    const BYTES: scioxide_core::Quantity = scioxide_core::quantity!("storage_bytes");
    work.dofs(rows)?;
    work.dofs(cols)?;
    work.entries(entries.len())?;
    work.tick(size(entries.len(), 2, WORK)?)?;
    work.sort(entries.len())?;
    // COO canonicalization plus CSR/transpose indexing: conservatively charge eight
    // complete entry payloads and eight index arrays, including temporary sorting.
    work.charge::<(usize, usize, V)>(size(entries.len(), 8, BYTES)?)?;
    work.charge::<usize>(size(add(add(rows, cols, BYTES)?, 2, BYTES)?, 8, BYTES)?)?;
    let mut builder = SparseBuilder::with_capacity(rows, cols, entries.len())?;
    for &(row, col, value) in entries {
        builder.push(
            row,
            col,
            finite(value, scioxide_core::NonFiniteOrigin::Intermediate)?,
        )?;
    }
    builder.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)
}

fn row<R: RealScalar>(
    matrix: &CsrMatrix<R>,
    i: usize,
) -> impl Iterator<Item = (usize, R)> + Clone + '_ {
    let range = matrix.row_ptr()[i]..matrix.row_ptr()[i + 1];
    matrix.col_indices()[range.clone()]
        .iter()
        .copied()
        .zip(matrix.values()[range].iter().copied())
}

/// A prepared real-weight map usable with matching real or complex vectors.
/// Storage is canonical shared CSR; dense materialization is explicitly caller-requested.
/// Scoped application schedules rows with the policy of the space or transfer that
/// produced it, in a fixed reduction order.
#[derive(Clone, Debug)]
pub struct AmrMap<R: RealScalar> {
    matrix: CsrMatrix<R>,
    policy: ExecutionPolicy,
}
impl<R: RealScalar> AmrMap<R> {
    pub(crate) const fn new(matrix: CsrMatrix<R>, policy: ExecutionPolicy) -> Self {
        Self { matrix, policy }
    }
    /// Borrow the real canonical sparse coefficients.
    pub fn matrix(&self) -> &CsrMatrix<R> {
        &self.matrix
    }
    fn apply<V: NumericScalar<Real = R>>(
        &self,
        input: &[V],
        output: &mut [V],
        scratch: &mut [V],
        execution: ExecutionContext<'_>,
    ) -> Result<()> {
        validate(input, self.matrix.cols(), scioxide_core::argument!("input"))?;
        rows_into(
            self.matrix.rows(),
            output,
            scratch,
            execution,
            self.policy,
            |i| {
                scioxide_core::backend::standard::sum_products(
                    row(&self.matrix, i).map(|(j, w)| (V::from_real(w), input[j])),
                    scioxide_core::Accuracy::Standard,
                )
            },
        )
    }

    fn adjoint<V: NumericScalar<Real = R>>(
        &self,
        input: &[V],
        output: &mut [V],
        scratch: &mut [V],
        execution: ExecutionContext<'_>,
    ) -> Result<()> {
        validate(input, self.matrix.rows(), scioxide_core::argument!("input"))?;
        let traversal = self.matrix.opposite_traversal();
        let pattern = traversal.pattern();
        rows_into(
            self.matrix.cols(),
            output,
            scratch,
            execution,
            self.policy,
            |row| {
                scioxide_core::backend::standard::sum_products(
                    (pattern.row_ptr()[row]..pattern.row_ptr()[row + 1]).map(|k| {
                        (
                            V::from_real(self.matrix.values()[traversal.positions()[k]]),
                            input[pattern.col_indices()[k]],
                        )
                    }),
                    scioxide_core::Accuracy::Standard,
                )
            },
        )
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>> LinearOperator<V> for AmrMap<R> {
    fn rows(&self) -> usize {
        self.matrix.rows()
    }
    fn cols(&self) -> usize {
        self.matrix.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize> {
        Ok(self.matrix.rows())
    }
    fn apply_into(&self, input: &[V], output: &mut [V], scratch: &mut [V]) -> Result<()> {
        self.apply(
            input,
            output,
            scratch,
            ExecutionContext::sequential_native(),
        )
    }
    fn apply_into_scoped(
        &self,
        input: &[V],
        output: &mut [V],
        scratch: &mut [V],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, V>,
    ) -> Result<scioxide_core::Invocation<()>> {
        self.apply(input, output, scratch, context.execution())
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>> AdjointOperator<V> for AmrMap<R> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize> {
        Ok(self.matrix.cols())
    }
    fn apply_adjoint_into(&self, input: &[V], output: &mut [V], scratch: &mut [V]) -> Result<()> {
        self.adjoint(
            input,
            output,
            scratch,
            ExecutionContext::sequential_native(),
        )
    }
    fn apply_adjoint_into_scoped(
        &self,
        input: &[V],
        output: &mut [V],
        scratch: &mut [V],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, V>,
    ) -> Result<scioxide_core::Invocation<()>> {
        self.adjoint(input, output, scratch, context.execution())
            .map(scioxide_core::Invocation::Evaluated)
    }
}

fn rows_into<V: NumericScalar>(
    n: usize,
    output: &mut [V],
    scratch: &mut [V],
    execution: ExecutionContext<'_>,
    policy: ExecutionPolicy,
    f: impl Fn(usize) -> Result<V> + Sync,
) -> Result<()> {
    if output.len() != n || scratch.len() < n {
        return Err(invalid(
            if output.len() != n {
                scioxide_core::argument!("output")
            } else {
                scioxide_core::argument!("scratch")
            },
            "output or workspace shape",
        ));
    }
    execution.check()?;
    let plan = execution.plan(
        policy,
        n,
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: core::num::NonZeroUsize::MIN,
            worker_limit: crate::validation::WORKER_LIMIT,
        },
    );
    let mut slots: [PartitionSlot<&mut [V]>; crate::validation::MAX_WORKERS] =
        core::array::from_fn(|_| PartitionSlot::empty());
    let mut remaining = &mut scratch[..n];
    for partition in plan.partitions() {
        let (part, tail) = remaining.split_at_mut(partition.range().len());
        remaining = tail;
        slots[partition.worker_index()] = PartitionSlot::new(part);
    }
    execution.execute_partitioned(
        plan,
        &slots[..plan.worker_count()],
        &|partition, part, _| {
            for (i, value) in part.iter_mut().enumerate() {
                execution.check()?;
                *value = finite(
                    f(partition.range().start + i)?,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
            Ok(())
        },
    )?;

    execution.check()?;
    output.copy_from_slice(&scratch[..n]);
    Ok(())
}
