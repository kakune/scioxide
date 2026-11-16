//! Symbolic simplex assembly and deterministic numerical reduction.
use super::*;
use crate::discretization::{AssemblyDestination, Elimination};
use scioxide_core::{StructuralPattern, quantity};

/// Holds staging values for repeated simplex CSR assembly.
///
/// Plans create this workspace with `assembly_workspace()`. Its size is one scalar
/// per structural matrix entry. Numerical assembly keeps the output pattern,
/// including explicit zeros, and publishes all values together on success.
/// Contributions add in cell order, with rounding proportional to that row's
/// number of contributions times scalar epsilon for well-scaled sums.
///
/// # References
/// T. J. R. Hughes, *The Finite Element Method* (2000), Chapter 3.
#[derive(Clone, Debug)]
pub struct SimplexCsrWorkspace<V: NumericScalar> {
    pub(super) values: Vec<V>,
}

/// Borrow a fully evaluated real tangent at one fixed state.
///
/// Products use the canonical CSR in O(stored entries) work and allocate nothing
/// with the operator's requested scratch. The state and workspace remain borrowed
/// until this view is released, so refreshing the workspace cannot stale the view.
/// Products accumulate in scalar precision; well-scaled rows have rounding of
/// order `stored entries per row × epsilon`.
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideError, LinearOperator};
/// use scioxide_pde::HyperelasticityPlan;
/// fn check(plan: &HyperelasticityPlan<f64, 2>, point: &[f64]) -> Result<(), SciOxideError> {
///     let mut workspace = plan.linearization_workspace()?;
///     let execution = ExecutionContext::sequential_native();
///     let tangent = plan.linearize(point, &mut workspace, execution)?;
///     let zero = vec![0.0; point.len()];
///     let mut result = zero.clone();
///     let mut scratch = vec![0.0; tangent.apply_workspace_len()?];
///     tangent.apply_into(&zero, &mut result, &mut scratch)?;
///     assert!(result.iter().all(|x| x.abs() < 1e-12));
///     Ok(())
/// }
/// ```
///
/// # References
/// Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed. (2003), Chapter 3.
#[derive(Clone, Copy)]
pub struct SimplexLinearization<'a, R: RealScalar> {
    pub(super) point: &'a [R],
    pub(super) matrix: &'a CsrMatrix<R>,
}
impl<'a, R: RealScalar> SimplexLinearization<'a, R> {
    /// Borrow the checked linearization point.
    pub const fn point(&self) -> &'a [R] {
        self.point
    }
    /// Borrow the evaluated tangent without copying its pattern or values.
    pub const fn matrix(&self) -> &'a CsrMatrix<R> {
        self.matrix
    }
    /// Copy numerical values into a matching reserved CSR without allocation.
    /// An error preserves the destination.
    /// # Errors
    /// Rejects a different sparsity pattern.
    pub fn assemble_csr_into(&self, output: &mut CsrMatrix<R>) -> Result<(), SciOxideError> {
        if output.structural_pattern() != self.matrix.structural_pattern() {
            return Err(invalid(
                scioxide_core::argument!("output"),
                "linearization CSR pattern mismatch",
            ));
        }
        output.copy_values_from(self.matrix.values())
    }
}
impl<R: RealScalar> LinearOperator<R> for SimplexLinearization<'_, R> {
    fn rows(&self) -> usize {
        self.matrix.rows()
    }
    fn cols(&self) -> usize {
        self.matrix.cols()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.matrix.apply_workspace_len()
    }
    fn apply_into(&self, x: &[R], out: &mut [R], scratch: &mut [R]) -> Result<(), SciOxideError> {
        self.matrix.apply_into(x, out, scratch)
    }
    fn apply_into_scoped(
        &self,
        x: &[R],
        out: &mut [R],
        scratch: &mut [R],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, R>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.matrix.apply_into_scoped(x, out, scratch, context)
    }
}
impl<R: RealScalar> scioxide_core::AdjointOperator<R> for SimplexLinearization<'_, R> {
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.matrix.apply_adjoint_workspace_len()
    }
    fn apply_adjoint_into(
        &self,
        x: &[R],
        out: &mut [R],
        scratch: &mut [R],
    ) -> Result<(), SciOxideError> {
        self.matrix.apply_adjoint_into(x, out, scratch)
    }
    fn apply_adjoint_into_scoped(
        &self,
        x: &[R],
        out: &mut [R],
        scratch: &mut [R],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, R>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.matrix
            .apply_adjoint_into_scoped(x, out, scratch, context)
    }
}

pub(crate) struct TangentStorage<R: RealScalar> {
    pub(crate) matrices: Vec<R>,
    pub(crate) matrix: CsrMatrix<R>,
    pub(crate) assembly: SimplexCsrWorkspace<R>,
}
impl<R: RealScalar> TangentStorage<R> {
    pub(crate) fn new(
        pattern: &AssemblyPattern,
        entries: usize,
        extra: &[R],
    ) -> Result<Self, SciOxideError> {
        Ok(Self {
            matrices: try_filled(entries, R::ZERO, quantity!("tangent_elements"))?,
            matrix: pattern.matrix(&|_, _, _| Ok(R::ZERO), extra)?,
            assembly: pattern.workspace()?,
        })
    }
}

#[derive(Clone)]
pub(crate) struct AssemblyPattern {
    pattern: StructuralPattern,
    offsets: Vec<usize>,
    sources: Vec<usize>,
    local: usize,
    element_entries: usize,
}

impl AssemblyPattern {
    /// Build each sorted row from its incident cells; source order is cell order.
    pub(crate) fn new<R: RealScalar, const D: usize>(
        space: &SimplexSpace<R, D>,
        layout: Layout,
        constrained: &[bool],
        elimination: Elimination,
        extra: impl IntoIterator<Item = (usize, usize)>,
    ) -> Result<Self, SciOxideError> {
        Self::with_coupling(space, layout, constrained, elimination, extra, |_, _| true)
    }

    pub(crate) fn with_coupling<R: RealScalar, const D: usize>(
        space: &SimplexSpace<R, D>,
        layout: Layout,
        constrained: &[bool],
        elimination: Elimination,
        extra: impl IntoIterator<Item = (usize, usize)>,
        coupling: impl Fn(usize, usize) -> bool,
    ) -> Result<Self, SciOxideError> {
        let rows = constrained.len();
        let physical = layout.unknowns(&space.mesh)?;
        let local = layout.local::<D>();
        let element_entries = layout.entries(&space.mesh)?;
        let destination = if elimination == Elimination::Lift {
            AssemblyDestination::Lift
        } else {
            AssemblyDestination::Matrix
        };
        let mut extras = Vec::new();
        for (index, (row, col)) in extra.into_iter().enumerate() {
            if row >= rows || col >= rows {
                return Err(invalid(
                    scioxide_core::argument!("pattern"),
                    "extra assembly index",
                ));
            }
            scioxide_core::backend::try_reserve(&mut extras, 1, quantity!("assembly_pattern"))?;
            extras.push((
                row,
                col,
                add(element_entries, index, quantity!("assembly_pattern"))?,
            ));
        }
        extras.sort_unstable();
        let bound = add(
            add(element_entries, rows, quantity!("assembly_pattern"))?,
            extras.len(),
            quantity!("assembly_pattern"),
        )?;
        let mut pointers = try_with_capacity(
            add(rows, 1, quantity!("assembly_pattern"))?,
            quantity!("assembly_pattern"),
        )?;
        let mut columns = try_with_capacity(bound, quantity!("assembly_pattern"))?;
        let mut sources = try_with_capacity(bound, quantity!("assembly_pattern"))?;
        let mut offsets = try_with_capacity(
            add(bound, 1, quantity!("assembly_pattern"))?,
            quantity!("assembly_pattern"),
        )?;
        let mut row_entries = Vec::new();
        let mut extra_index = 0;
        pointers.push(0);
        offsets.push(0);
        for row in 0..rows {
            row_entries.clear();
            if constrained[row] && elimination == Elimination::Symmetric {
                scioxide_core::backend::try_reserve(
                    &mut row_entries,
                    1,
                    quantity!("assembly_row"),
                )?;
                row_entries.push((row, usize::MAX));
            }
            if row < physical && !constrained[row] {
                for (cell, i) in space.incident(layout, row) {
                    scioxide_core::backend::try_reserve(
                        &mut row_entries,
                        local,
                        quantity!("assembly_row"),
                    )?;
                    for j in 0..local {
                        if !coupling(i, j) {
                            continue;
                        }
                        let col = space.global(layout, cell, j);
                        let keep = elimination.destination(constrained[row], constrained[col])
                            == destination;
                        if keep {
                            row_entries.push((col, (cell * local + i) * local + j));
                        }
                    }
                }
            }
            while extra_index < extras.len() && extras[extra_index].0 == row {
                let (_, col, source) = extras[extra_index];
                if elimination.destination(constrained[row], constrained[col]) == destination {
                    scioxide_core::backend::try_reserve(
                        &mut row_entries,
                        1,
                        quantity!("assembly_row"),
                    )?;
                    row_entries.push((col, source));
                }
                extra_index += 1;
            }
            row_entries.sort_unstable();
            for (index, &(col, source)) in row_entries.iter().enumerate() {
                if index == 0 || row_entries[index - 1].0 != col {
                    if index != 0 {
                        offsets.push(sources.len());
                    }
                    columns.push(col);
                }
                sources.push(source);
            }
            if !row_entries.is_empty() {
                offsets.push(sources.len());
            }
            pointers.push(columns.len());
        }
        Ok(Self {
            pattern: StructuralPattern::new(rows, rows, pointers, columns)?,
            offsets,
            sources,
            local,
            element_entries,
        })
    }

    pub(crate) fn workspace<V: NumericScalar>(
        &self,
    ) -> Result<SimplexCsrWorkspace<V>, SciOxideError> {
        Ok(SimplexCsrWorkspace {
            values: try_filled(
                self.pattern.view().col_indices().len(),
                V::ZERO,
                quantity!("assembly_values"),
            )?,
        })
    }

    pub(crate) fn matrix<V: NumericScalar>(
        &self,
        entry: &(impl Fn(usize, usize, usize) -> Result<V, SciOxideError> + Sync),
        extra: &[V],
    ) -> Result<CsrMatrix<V>, SciOxideError> {
        let mut workspace = self.workspace()?;
        self.fill(0, &mut workspace.values, entry, extra)?;
        let mut pattern = StructuralPattern::new(0, 0, vec![0], Vec::new())?;
        pattern.try_assign(self.pattern.view())?;
        CsrMatrix::from_pattern(pattern, workspace.values)
    }

    fn fill<V: NumericScalar>(
        &self,
        first: usize,
        values: &mut [V],
        entry: &(impl Fn(usize, usize, usize) -> Result<V, SciOxideError> + Sync),
        extra: &[V],
    ) -> Result<(), SciOxideError> {
        for (offset, value) in values.iter_mut().enumerate() {
            let position = first + offset;
            let mut sum = V::ZERO;
            for &source in &self.sources[self.offsets[position]..self.offsets[position + 1]] {
                let term = if source == usize::MAX {
                    V::ONE
                } else if source >= self.element_entries {
                    *extra.get(source - self.element_entries).ok_or_else(|| {
                        invalid(scioxide_core::argument!("extra"), "missing assembly value")
                    })?
                } else {
                    let n = self.local;
                    entry(source / (n * n), source / n % n, source % n)?
                };
                sum = finite(sum + term, NonFiniteOrigin::Intermediate)?;
            }
            *value = sum;
        }
        Ok(())
    }

    pub(crate) fn assemble_into<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
        &self,
        space: &SimplexSpace<R, D>,
        output: &mut CsrMatrix<V>,
        workspace: &mut SimplexCsrWorkspace<V>,
        entry: &(impl Fn(usize, usize, usize) -> Result<V, SciOxideError> + Sync),
        extra: &[V],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        if output.structural_pattern() != self.pattern.view()
            || workspace.values.len() != output.nnz()
        {
            return Err(invalid(
                scioxide_core::argument!("output"),
                "CSR assembly pattern or workspace mismatch",
            ));
        }
        execution.check()?;
        space.partitioned(&mut workspace.values, 1, execution, &|first, values| {
            self.fill(first, values, entry, extra)
        })?;
        execution.check()?;
        output.copy_values_from(&workspace.values)
    }

    pub(crate) fn bound(entries: usize, rows: usize) -> Result<space::Bytes, SciOxideError> {
        space::Bytes::new(size_of::<Self>())
            .array::<usize>(size(
                add(entries, rows, quantity!("assembly_pattern"))?,
                5,
                quantity!("assembly_pattern"),
            )?)?
            .array::<usize>(add(rows, 1, quantity!("assembly_pattern"))?)
    }
}
