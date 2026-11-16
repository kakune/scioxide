//! Shared prepared operators, load sites and the spatial plan protocol.
//!
//! Every discretization family (structured FDM, cell-centred FVM, simplex FEM)
//! prepares a time-invariant operator `A` and a set of load sites. Loads at time
//! `t` combine sources, natural/Robin boundary data and lifted Dirichlet values:
//! `b(t) = Σ w f(x, t) + Σ w g(x, n, t) − L g_D(t)`, with Dirichlet rows set to
//! `g_D(t)`. Dirichlet unknowns are eliminated symmetrically: constrained rows and
//! columns are decoupled with unit diagonals, so symmetric problems assemble
//! symmetric matrices.

use alloc::vec::Vec;

use scioxide_core::backend::{try_filled, try_with_capacity};
use scioxide_core::{
    Dimensions, DomainReason, ExecutionContext, LinearOperator, NonFiniteOrigin, NumericScalar,
    NumericalIssue, RealScalar, SciOxideError, Tolerance,
};
use scioxide_sparse::{CsrMatrix, DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};
use scioxide_spatial::MeshToken;

use crate::boundary::{
    BoundaryConditions, BoundaryData, BoundaryPoint, DirichletConflictPolicy, FieldData,
    FieldPoint, Prescribed, select_prescribed,
};
use crate::validation::{finite, invalid};
use crate::{SimplexFemPlan, SimplexFvmPlan, StructuredFdmPlan, StructuredFvmPlan};

/// Caller ceilings checked before a structured FDM, structured FVM or simplex FVM plan
/// allocates storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatialLimits {
    max_unknowns: usize,
    max_entries: usize,
    max_storage_bytes: usize,
}

impl Default for SpatialLimits {
    fn default() -> Self {
        Self::new(usize::MAX, usize::MAX, usize::MAX)
    }
}

impl SpatialLimits {
    /// Ceilings on algebraic unknowns (including components), emitted COO triplets
    /// before canonicalization, and bytes reserved for coefficient, geometry and
    /// load-site arrays.
    pub const fn new(max_unknowns: usize, max_entries: usize, max_storage_bytes: usize) -> Self {
        Self {
            max_unknowns,
            max_entries,
            max_storage_bytes,
        }
    }

    /// Maximum algebraic unknown count, including components.
    pub const fn max_unknowns(&self) -> usize {
        self.max_unknowns
    }

    /// Maximum emitted COO triplets before canonicalization.
    pub const fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Maximum bytes reserved for coefficient, geometry and load-site arrays.
    pub const fn max_storage_bytes(&self) -> usize {
        self.max_storage_bytes
    }

    /// Reject a preparation exceeding any ceiling.
    pub(crate) fn admit(
        self,
        unknowns: usize,
        entries: usize,
        storage: usize,
    ) -> Result<(), SciOxideError> {
        if unknowns > self.max_unknowns
            || entries > self.max_entries
            || storage > self.max_storage_bytes
        {
            return Err(invalid(
                scioxide_core::argument!("limits"),
                "spatial resource limit exceeded",
            ));
        }
        Ok(())
    }
}

/// Classify an assembled operator entry: non-finite values from finite inputs
/// are overflow in every family.
pub(crate) fn entry<V: NumericScalar>(value: V) -> Result<V, SciOxideError> {
    representable(value, "assembled operator entry is unrepresentable")
}

/// Classify a non-finite value computed from finite data as overflow.
pub(crate) fn representable<V: NumericScalar>(
    value: V,
    detail: &'static str,
) -> Result<V, SciOxideError> {
    if value.has_finite_components() {
        Ok(value)
    } else {
        Err(SciOxideError::numerical_instability(
            NumericalIssue::Overflow,
            detail,
        ))
    }
}

/// Eliminated operator, free-row/constrained-column lift and constraint mask.
pub(crate) type Eliminated<V> = (CsrMatrix<V>, CsrMatrix<V>, Vec<bool>);

/// How an [`OperatorBuilder`] treats constrained unknowns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Elimination {
    /// Symmetric elimination: constrained rows dropped, constrained columns of
    /// free rows moved into the lift, unit diagonals on constrained rows.
    Symmetric,
    /// Parameter tangent of the free residual rows: constrained rows dropped,
    /// every column kept in the matrix, no unit diagonal and an empty lift.
    Tangent,
    /// Only the free-row/constrained-column lift; the matrix is empty.
    Lift,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AssemblyDestination {
    Matrix,
    Lift,
    Discard,
}

impl Elimination {
    /// Route one structural entry; all simplex and adaptive builders share this rule.
    pub(crate) fn destination(self, row_fixed: bool, column_fixed: bool) -> AssemblyDestination {
        if row_fixed {
            return AssemblyDestination::Discard;
        }
        match (column_fixed, self) {
            (true, Self::Symmetric | Self::Lift) => AssemblyDestination::Lift,
            (false, Self::Lift) => AssemblyDestination::Discard,
            _ => AssemblyDestination::Matrix,
        }
    }
}

/// Triplet sink performing symmetric Dirichlet elimination.
pub(crate) struct OperatorBuilder<V: NumericScalar> {
    constrained: Vec<bool>,
    elimination: Elimination,
    matrix: SparseBuilder<V>,
    lift: SparseBuilder<V>,
}

impl<V: NumericScalar> OperatorBuilder<V> {
    pub(crate) fn new(constrained: Vec<bool>, entry_bound: usize) -> Result<Self, SciOxideError> {
        Self::with_elimination(constrained, entry_bound, Elimination::Symmetric)
    }

    pub(crate) fn with_elimination(
        constrained: Vec<bool>,
        entry_bound: usize,
        elimination: Elimination,
    ) -> Result<Self, SciOxideError> {
        let n = constrained.len();
        let fixed = constrained.iter().filter(|&&c| c).count();
        let matrix = if elimination == Elimination::Lift {
            0
        } else {
            entry_bound.saturating_add(fixed)
        };
        // The lift couples only rows next to constrained unknowns; it grows on demand.
        Ok(Self {
            matrix: SparseBuilder::with_capacity(n, n, matrix)?,
            lift: SparseBuilder::new(n, n),
            constrained,
            elimination,
        })
    }

    /// Route one entry: constrained rows are dropped, constrained columns of
    /// free rows move into the lift (symmetric and lift-only elimination).
    pub(crate) fn push(
        &mut self,
        row: usize,
        column: usize,
        value: V,
    ) -> Result<(), SciOxideError> {
        if value == V::ZERO || self.constrained[row] {
            return Ok(());
        }
        let value = entry(value)?;
        match self
            .elimination
            .destination(self.constrained[row], self.constrained[column])
        {
            AssemblyDestination::Lift => self.lift.push(row, column, value),
            AssemblyDestination::Discard => Ok(()),
            AssemblyDestination::Matrix => self.matrix.push(row, column, value),
        }
    }

    /// Constraint mask of the routed unknowns.
    pub(crate) fn constrained(&self) -> &[bool] {
        &self.constrained
    }

    /// Canonical eliminated operator and lift.
    pub(crate) fn finish(mut self) -> Result<Eliminated<V>, SciOxideError> {
        if self.elimination == Elimination::Symmetric {
            for (row, &fixed) in self.constrained.iter().enumerate() {
                if fixed {
                    self.matrix.push(row, row, V::ONE)?;
                }
            }
        }
        Ok((
            self.matrix
                .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)?,
            self.lift
                .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Drop)?,
            self.constrained,
        ))
    }
}

/// Prepared eliminated operator shared by structured FDM and FVM: one apply,
/// one CSR, one constant-mode compatibility check and one `LinearOperator`.
#[derive(Debug, Clone)]
pub(crate) struct AssembledOperator<R: RealScalar, V: NumericScalar<Real = R>> {
    pub(crate) matrix: CsrMatrix<V>,
    pub(crate) bounds: crate::kernels::RowBounds<R>,
    pub(crate) compact_columns: Option<Vec<u32>>,
    pub(crate) constrained: Vec<bool>,
    pub(crate) mass: Vec<R>,
    pub(crate) components: usize,
    pub(crate) constant_modes: bool,
    pub(crate) entry_bound: usize,
}

impl<R: RealScalar, V: NumericScalar<Real = R>> AssembledOperator<R, V> {
    pub(crate) fn unknowns(&self) -> usize {
        self.constrained.len()
    }

    /// Publish admitted rows directly; fallible/scoped rows use `workspace`.
    /// Output is unchanged on error, including a stop at the commit boundary.
    pub(crate) fn apply(
        &self,
        input: &[V],
        output: &mut [V],
        workspace: &mut [V],
        execution: Option<ExecutionContext<'_>>,
    ) -> Result<(), SciOxideError> {
        let n = self.unknowns();
        for actual in [input.len(), output.len()] {
            if actual != n {
                return Err(SciOxideError::dimension_mismatch(
                    Dimensions::one(n),
                    Dimensions::one(actual),
                ));
            }
        }
        if workspace.len() < n {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(n),
                Dimensions::one(workspace.len()),
            ));
        }
        let native = self.bounds.admits(input)?;
        if let Some(context) = execution {
            context.check()?;
        }
        let (pointers, columns, values) = (
            self.matrix.row_ptr(),
            self.matrix.col_indices(),
            self.matrix.values(),
        );
        if native && execution.is_none() {
            self.native_apply(input, output);
            return Ok(());
        }
        let staged = &mut workspace[..n];
        if native {
            self.native_apply(input, staged);
        } else {
            for (row, out) in staged.iter_mut().enumerate() {
                let range = pointers[row]..pointers[row + 1];
                *out = crate::kernels::checked_row(
                    V::ZERO,
                    &values[range.clone()],
                    &columns[range],
                    input,
                )?;
            }
        }
        if let Some(context) = execution {
            context.check()?;
        }
        output.copy_from_slice(staged);
        Ok(())
    }

    fn native_apply(&self, input: &[V], output: &mut [V]) {
        match &self.compact_columns {
            Some(columns) => self.native_rows(input, output, columns),
            None => self.native_rows(input, output, self.matrix.col_indices()),
        }
    }

    fn native_rows<I: crate::kernels::StencilIndex>(
        &self,
        input: &[V],
        output: &mut [V],
        columns: &[I],
    ) {
        let pointers = self.matrix.row_ptr();
        let values = self.matrix.values();
        for (row, out) in output.iter_mut().enumerate() {
            let range = pointers[row]..pointers[row + 1];
            *out = crate::kernels::row(V::ZERO, &values[range.clone()], &columns[range], input);
        }
    }

    /// Check each component's integrated load against its constant mode.
    pub(crate) fn check_constant_mode_compatibility(
        &self,
        rhs: &[V],
        tolerance: Tolerance<R>,
    ) -> Result<(), SciOxideError> {
        check_constant_modes(self.constant_modes, self.components, rhs, tolerance)
    }
}

/// Component-block load sums must vanish relative to their absolute sums.
pub(crate) fn check_constant_modes<R: RealScalar, V: NumericScalar<Real = R>>(
    known: bool,
    components: usize,
    rhs: &[V],
    tolerance: Tolerance<R>,
) -> Result<(), SciOxideError> {
    if !known {
        return Err(invalid(
            scioxide_core::argument!("self"),
            "constant modes are not structurally known",
        ));
    }
    let block = rhs.len() / components;
    for component in rhs.chunks_exact(block) {
        let (sum, scale) =
            component
                .iter()
                .try_fold((V::ZERO, R::ZERO), |(sum, scale), &value| {
                    finite(value, NonFiniteOrigin::Input)?;
                    Ok::<_, SciOxideError>((
                        finite(sum + value, NonFiniteOrigin::Intermediate)?,
                        scale + value.abs(),
                    ))
                })?;
        if sum.abs() > tolerance.absolute() + tolerance.relative() * scale {
            return Err(SciOxideError::domain_error(
                scioxide_core::argument!("rhs"),
                DomainReason::OutsideDomain,
                "load is incompatible with the known constant mode",
            ));
        }
    }
    Ok(())
}

/// Source of boundary values by condition index, so one load evaluation serves
/// bound conditions and their parameter tangents.
pub(crate) trait BoundaryValues<R, V, const D: usize> {
    /// Value of condition `condition` at a boundary site.
    fn value(
        &self,
        condition: usize,
        site: BoundaryPoint<R, D>,
        time: R,
    ) -> Result<V, SciOxideError>;
    /// Shared-node Dirichlet precedence.
    fn conflict_policy(&self) -> DirichletConflictPolicy;
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize, Region> BoundaryValues<R, V, D>
    for BoundaryConditions<R, V, D, Region>
{
    fn value(
        &self,
        condition: usize,
        site: BoundaryPoint<R, D>,
        time: R,
    ) -> Result<V, SciOxideError> {
        self.evaluate(condition, site, time)
    }
    fn conflict_policy(&self) -> DirichletConflictPolicy {
        self.conflict_policy
    }
}

/// Data of every condition in condition order, with the bound precedence.
pub(crate) struct ConditionData<'a, R, V, const D: usize> {
    pub(crate) data: &'a [BoundaryData<R, V, D>],
    pub(crate) policy: DirichletConflictPolicy,
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> BoundaryValues<R, V, D>
    for ConditionData<'_, R, V, D>
{
    fn value(
        &self,
        condition: usize,
        site: BoundaryPoint<R, D>,
        time: R,
    ) -> Result<V, SciOxideError> {
        self.data[condition].evaluate(site, time)
    }
    fn conflict_policy(&self) -> DirichletConflictPolicy {
        self.policy
    }
}

/// One strongly imposed Dirichlet candidate; sites are sorted by `(dof, rank)`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DirichletSite<R, const D: usize> {
    pub(crate) dof: usize,
    pub(crate) site: BoundaryPoint<R, D>,
    pub(crate) condition: usize,
    pub(crate) rank: (usize, usize),
}

/// Prepared load quadrature: weighted source sites, weighted boundary-data
/// sites, Dirichlet candidates and the Dirichlet lift. One evaluation path
/// serves every family and every time.
#[derive(Debug, Clone)]
pub(crate) struct PreparedLoads<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    pub(crate) preparation: alloc::sync::Arc<()>,
    unknowns: usize,
    sources: Vec<FieldPoint<R, D>>,
    source_offsets: Vec<usize>,
    source_entries: Vec<(usize, R)>,
    data: Vec<(BoundaryPoint<R, D>, usize)>,
    data_offsets: Vec<usize>,
    data_entries: Vec<(usize, R)>,
    dirichlet: Vec<DirichletSite<R, D>>,
    lift: Option<CsrMatrix<V>>,
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> PreparedLoads<R, V, D> {
    pub(crate) fn new(unknowns: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            preparation: alloc::sync::Arc::new(()),
            unknowns,
            sources: Vec::new(),
            source_offsets: try_filled(1, 0, scioxide_core::quantity!("source_offsets"))?,
            source_entries: Vec::new(),
            data: Vec::new(),
            data_offsets: try_filled(1, 0, scioxide_core::quantity!("data_offsets"))?,
            data_entries: Vec::new(),
            dirichlet: Vec::new(),
            lift: None,
        })
    }

    /// Add a source site contributing `weight * f(site)` to each listed row.
    pub(crate) fn reserve_sources(
        &mut self,
        sites: usize,
        entries: usize,
    ) -> Result<(), SciOxideError> {
        scioxide_core::backend::try_reserve(
            &mut self.sources,
            sites,
            scioxide_core::quantity!("source_sites"),
        )?;
        scioxide_core::backend::try_reserve(
            &mut self.source_offsets,
            sites,
            scioxide_core::quantity!("source_sites"),
        )?;
        scioxide_core::backend::try_reserve(
            &mut self.source_entries,
            entries,
            scioxide_core::quantity!("source_entries"),
        )
    }

    /// Add a source site contributing `weight * f(site)` to each listed row.
    pub(crate) fn push_source(
        &mut self,
        site: FieldPoint<R, D>,
        entries: impl IntoIterator<Item = (usize, R)>,
    ) -> Result<(), SciOxideError> {
        push(&mut self.sources, site)?;
        for entry in entries {
            push(&mut self.source_entries, entry)?;
        }
        push(&mut self.source_offsets, self.source_entries.len())
    }

    /// Add a boundary-data site of `condition` contributing `weight * g(site)`.
    pub(crate) fn push_data(
        &mut self,
        site: BoundaryPoint<R, D>,
        condition: usize,
        entries: impl IntoIterator<Item = (usize, R)>,
    ) -> Result<(), SciOxideError> {
        push(&mut self.data, (site, condition))?;
        for entry in entries {
            push(&mut self.data_entries, entry)?;
        }
        push(&mut self.data_offsets, self.data_entries.len())
    }

    /// Add a strongly imposed Dirichlet candidate.
    pub(crate) fn push_dirichlet(
        &mut self,
        site: DirichletSite<R, D>,
    ) -> Result<(), SciOxideError> {
        push(&mut self.dirichlet, site)
    }

    /// Attach the free-row/constrained-column lift and canonicalize candidates.
    pub(crate) fn finish(mut self, lift: Option<CsrMatrix<V>>) -> Self {
        self.dirichlet.sort_by_key(|site| (site.dof, site.rank));
        self.lift = lift;
        self
    }

    /// Unknowns with at least one Dirichlet candidate.
    pub(crate) fn dirichlet_dofs(&self) -> impl Iterator<Item = &usize> + '_ {
        self.dirichlet.iter().map(|site| &site.dof)
    }

    /// Stage a load once and commit only after every callback and reduction succeeds.
    pub(crate) fn load_transaction(
        &self,
        boundaries: &impl BoundaryValues<R, V, D>,
        source: &FieldData<R, V, D>,
        time: R,
        out: &mut [V],
        scratch: &mut [V],
    ) -> Result<(), SciOxideError> {
        crate::validation::shape(out.len(), self.unknowns)?;
        if scratch.len() < self.unknowns {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(self.unknowns),
                Dimensions::one(scratch.len()),
            ));
        }
        let staged = &mut scratch[..self.unknowns];
        self.load_into(boundaries, source, time, staged)?;
        out.copy_from_slice(staged);
        Ok(())
    }

    /// Evaluate `b(t)` into scratch. Output is unspecified on error.
    pub(crate) fn load_into(
        &self,
        boundaries: &impl BoundaryValues<R, V, D>,
        source: &FieldData<R, V, D>,
        time: R,
        out: &mut [V],
    ) -> Result<(), SciOxideError> {
        self.natural_into(boundaries, source, time, out)?;
        self.dirichlet_into(boundaries, time, out)?;
        if let Some(lift) = &self.lift {
            apply_lift(lift, out)?;
        }
        Ok(())
    }

    /// Evaluate the source and natural/Robin boundary data into `out`, before
    /// Dirichlet values and lifting. Output is unspecified on error.
    pub(crate) fn natural_into(
        &self,
        boundaries: &impl BoundaryValues<R, V, D>,
        source: &FieldData<R, V, D>,
        time: R,
        out: &mut [V],
    ) -> Result<(), SciOxideError> {
        if out.len() != self.unknowns {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(self.unknowns),
                Dimensions::one(out.len()),
            ));
        }
        if !time.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "time must be finite",
            ));
        }
        out.fill(V::ZERO);
        if source.constant_value() != Some(V::ZERO) {
            for (site, window) in self.sources.iter().zip(self.source_offsets.windows(2)) {
                let value = source.evaluate(*site, time)?;
                scatter(out, &self.source_entries[window[0]..window[1]], value)?;
            }
        }
        for (&(site, condition), window) in self.data.iter().zip(self.data_offsets.windows(2)) {
            let value = boundaries.value(condition, site, time)?;
            scatter(out, &self.data_entries[window[0]..window[1]], value)?;
        }
        Ok(())
    }

    /// Write the selected Dirichlet value of every constrained unknown.
    pub(crate) fn dirichlet_into(
        &self,
        boundaries: &impl BoundaryValues<R, V, D>,
        time: R,
        out: &mut [V],
    ) -> Result<(), SciOxideError> {
        let mut current: Option<(usize, Prescribed<V>)> = None;
        for site in &self.dirichlet {
            let candidate = Prescribed {
                value: boundaries.value(site.condition, site.site, time)?,
                rank: site.rank,
            };
            current = Some(match current {
                Some((dof, selected)) if dof == site.dof => (
                    dof,
                    select_prescribed(Some(selected), candidate, boundaries.conflict_policy())?,
                ),
                previous => {
                    if let Some((dof, selected)) = previous {
                        out[dof] = selected.value;
                    }
                    (site.dof, candidate)
                }
            });
        }
        if let Some((dof, selected)) = current {
            out[dof] = selected.value;
        }
        Ok(())
    }
}

/// Lift prescribed values into free rows: `out[r] -= Σ L[r, c] out[c]`, where
/// `out` already holds the prescribed values on the constrained rows of `lift`.
pub(crate) fn apply_lift<V: NumericScalar>(
    lift: &CsrMatrix<V>,
    out: &mut [V],
) -> Result<(), SciOxideError> {
    let (pointers, columns, values) = (lift.row_ptr(), lift.col_indices(), lift.values());
    for row in 0..lift.rows() {
        out[row] = scioxide_core::backend::standard::sum_products(
            core::iter::once((out[row], V::ONE))
                .chain((pointers[row]..pointers[row + 1]).map(|k| (-values[k], out[columns[k]]))),
            scioxide_core::Accuracy::Standard,
        )?;
    }
    Ok(())
}

fn scatter<R: RealScalar, V: NumericScalar<Real = R>>(
    out: &mut [V],
    entries: &[(usize, R)],
    value: V,
) -> Result<(), SciOxideError> {
    for &(row, weight) in entries {
        out[row] = scioxide_core::backend::standard::sum_products(
            [(out[row], V::ONE), (V::from_real(weight), value)].into_iter(),
            scioxide_core::Accuracy::Standard,
        )?;
    }
    Ok(())
}

fn push<T>(values: &mut Vec<T>, value: T) -> Result<(), SciOxideError> {
    values.try_reserve(1).map_err(|_| {
        SciOxideError::allocation_failure(scioxide_core::quantity!("load_sites"), values.len() + 1)
    })?;
    values.push(value);
    Ok(())
}

/// Allocate a zero vector of `n` values.
pub(crate) fn zeros<V: NumericScalar>(n: usize) -> Result<Vec<V>, SciOxideError> {
    try_filled(n, V::ZERO, scioxide_core::quantity!("spatial_vector"))
}

/// Reserve an empty vector for `n` values.
pub(crate) fn reserve<T>(n: usize) -> Result<Vec<T>, SciOxideError> {
    try_with_capacity(n, scioxide_core::quantity!("spatial_storage"))
}

/// Summarize a prepared plan by its mesh revision, unknowns, components and
/// constraints; coefficient and boundary callbacks are not printed.
macro_rules! plan_debug {
    ($($plan:ident),+) => {$(
        impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> core::fmt::Debug
            for $plan<R, V, D>
        where
            Self: SpatialDiscretization<R, V, D>,
        {
            fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter
                    .debug_struct(stringify!($plan))
                    .field("mesh_token", self.mesh_token())
                    .field("unknowns", &self.unknowns())
                    .field("components", &self.components())
                    .field(
                        "constrained",
                        &self.constrained().iter().filter(|&&fixed| fixed).count(),
                    )
                    .finish_non_exhaustive()
            }
        }
    )+};
}
plan_debug!(
    StructuredFdmPlan,
    StructuredFvmPlan,
    SimplexFvmPlan,
    SimplexFemPlan
);

/// Admit structural provider metadata before any indexing or data callback.
pub(crate) fn validate_structure<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    plan: &impl SpatialDiscretization<R, V, D>,
) -> Result<(), SciOxideError> {
    let n = plan.unknowns();
    crate::validation::shape(plan.rows(), n)?;
    crate::validation::shape(plan.cols(), n)?;
    crate::validation::shape(plan.constrained().len(), n)?;
    crate::validation::shape(plan.lumped_mass().len(), n)?;
    if plan.components() == 0 || !n.is_multiple_of(plan.components()) {
        return Err(invalid(
            scioxide_core::argument!("plan"),
            "unknowns must form complete component blocks",
        ));
    }
    crate::validation::finite_slice(plan.lumped_mass(), NonFiniteOrigin::Callback)?;
    if plan.lumped_mass().iter().any(|&mass| mass <= R::ZERO) {
        return Err(invalid(
            scioxide_core::argument!("plan"),
            "lumped mass must be positive",
        ));
    }
    Ok(())
}

/// Prepared linear spatial discretization `A u = b(t)` with time-invariant `A`.
///
/// Unknowns are component-major. Constrained (Dirichlet) unknowns have unit
/// rows and columns; their values come from [`Self::load_into`]. Natural data is
/// the outward conormal flux and Robin data `n·K∇u + αu = g` in every family.
/// Steady solves evaluate data at time zero.
pub trait SpatialDiscretization<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>:
    LinearOperator<V>
{
    /// Owned physical field type returned by [`Self::field`].
    type Field;

    /// Exact mesh family and revision of the unknowns.
    fn mesh_token(&self) -> &MeshToken;

    /// Number of coupled field components.
    fn components(&self) -> usize;

    /// Number of algebraic unknowns, including constrained rows.
    fn unknowns(&self) -> usize {
        self.rows()
    }

    /// Physical site and component of one unknown.
    /// # Errors
    /// Returns an out-of-bounds error for an invalid unknown.
    fn dof_point(&self, unknown: usize) -> Result<FieldPoint<R, D>, SciOxideError>;

    /// Whether each unknown is a strongly imposed Dirichlet value.
    fn constrained(&self) -> &[bool];

    /// Positive lumped mass of each unknown.
    fn lumped_mass(&self) -> &[R];

    /// Whether each component has a structurally known constant nullspace mode.
    fn has_constant_modes(&self) -> bool;

    /// Assemble the eliminated operator as canonical CSR.
    /// # Errors
    /// Propagates checked sparse construction and allocation failures.
    fn assemble_csr(&self) -> Result<CsrMatrix<V>, SciOxideError>;

    /// Time dependence of plan-owned load data, excluding the supplied source.
    /// `TimeInvariant` permits a semidiscrete system to reuse a successfully
    /// sampled load when its source is also time-invariant. The conservative
    /// default keeps evaluating custom plans at every requested time.
    fn load_time_dependence(&self) -> crate::TimeDependence {
        crate::TimeDependence::TimeDependent
    }

    /// Scalar scratch required by [`Self::load_into`], including load staging.
    fn load_workspace_len(&self) -> usize {
        self.unknowns()
    }

    /// Evaluate the load `b(t)` into caller storage without allocating.
    /// Scratch must contain at least [`Self::load_workspace_len`] scalars.
    /// Every failure preserves `out`; callbacks are sampled once per load site.
    /// # Errors
    /// Propagates callback, non-finite, conflict and shape failures.
    fn load_into(
        &self,
        source: &FieldData<R, V, D>,
        time: R,
        out: &mut [V],
        scratch: &mut [V],
    ) -> Result<(), SciOxideError>;

    /// Allocate and evaluate the load `b(t)`.
    /// # Errors
    /// Propagates [`Self::load_into`] and allocation failures.
    fn load(&self, source: &FieldData<R, V, D>, time: R) -> Result<Vec<V>, SciOxideError> {
        let mut out = zeros(self.unknowns())?;
        let mut scratch = zeros(self.load_workspace_len())?;
        crate::validation::poison(&mut out);
        self.load_into(source, time, &mut out, &mut scratch)?;
        crate::validation::finite_slice(&out, NonFiniteOrigin::Callback)?;
        Ok(out)
    }

    /// Reconstruct the owned physical field from unknown values.
    /// # Errors
    /// Rejects a length mismatch and propagates field construction failures.
    fn field(&self, values: &[V]) -> Result<Self::Field, SciOxideError>;
}
