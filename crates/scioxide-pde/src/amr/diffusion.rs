use super::*;
use crate::discretization::{Eliminated, Elimination, OperatorBuilder};

/// Source for `-div(k grad u) = f` with cellwise constant positive scalar `k`.
/// Boundary kinds and data come from [`crate::TaggedBoundaries`] bound with
/// [`crate::TaggedBoundaries::adaptive`]. Sampling does not certify source
/// regularity or boundary resolution.
pub trait AmrDiffusionData<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>:
    Sync
{
    /// Physical source in one active cell.
    /// # Errors
    /// Callback errors propagate unchanged.
    fn source(&self, cell: usize, point: [R; D]) -> Result<V>;
}

/// Prepared scalar diffusion forms over a constrained adaptive nodal space.
/// Matrix-free application and explicit CSR assembly use the same local forms.
/// Existing core/linalg solvers own convergence, gauges and preconditioners. Scoped
/// application schedules rows with the space's execution policy.
#[derive(Clone)]
pub struct AmrDiffusionPlan<'a, R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    space: &'a AmrSpace<R, D>,
    diffusion: Vec<R>,
    boundaries: &'a crate::TaggedBoundaries<R, V, D>,
    data: &'a dyn AmrDiffusionData<R, V, D>,
    elements: Vec<[V; 100]>,
    rhs: Vec<V>,
    prescribed: Vec<Option<V>>,
    limits: AmrLimits,
    storage: usize,
}
impl<'a, R: RealScalar, V: NumericScalar<Real = R>, const D: usize> AmrDiffusionPlan<'a, R, V, D> {
    /// Prepare with default numerical settings and no resource ceiling.
    ///
    /// # Errors
    /// Returns the geometry, coefficient, allocation and configuration errors of [`Self::with_options`].
    pub fn new(
        space: &'a AmrSpace<R, D>,
        diffusion: FieldView<'_, R, Cell, D>,
        boundaries: &'a crate::TaggedBoundaries<R, V, D>,
        data: &'a dyn AmrDiffusionData<R, V, D>,
    ) -> Result<Self> {
        Self::with_options(space, diffusion, boundaries, data, AmrLimits::default())
    }

    /// Prepare immutable forms and Dirichlet lifting with explicit allocation limits.
    /// # Errors
    /// Rejects epoch/shape mismatches, nonpositive diffusion, nonfinite providers,
    /// incompatible boundary traces/kinds, or resource exhaustion.
    pub fn with_options(
        space: &'a AmrSpace<R, D>,
        diffusion: FieldView<'_, R, Cell, D>,
        boundaries: &'a crate::TaggedBoundaries<R, V, D>,
        data: &'a dyn AmrDiffusionData<R, V, D>,
        limits: AmrLimits,
    ) -> Result<Self> {
        Self::build(
            space,
            diffusion,
            boundaries,
            data,
            limits,
            ExecutionContext::sequential_native(),
        )
    }
    /// Prepare as one epoch of `scope`, charging every attempted callback to it.
    /// # Errors
    /// Same failures as `new`, including cancellation and evaluation/node budgets.
    pub fn build<'scope, 'capability>(
        space: &'a AmrSpace<R, D>,
        diffusion: FieldView<'_, R, Cell, D>,
        boundaries: &'a crate::TaggedBoundaries<R, V, D>,
        data: &'a dyn AmrDiffusionData<R, V, D>,
        limits: AmrLimits,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<Self> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "amr-diffusion", 2),
            |context| {
                Self::prepare(
                    space,
                    diffusion,
                    boundaries,
                    data,
                    &mut Work::new(limits, Some(context)),
                )
            },
        )
    }
    fn prepare(
        space: &'a AmrSpace<R, D>,
        diffusion: FieldView<'_, R, Cell, D>,
        boundaries: &'a crate::TaggedBoundaries<R, V, D>,
        data: &'a dyn AmrDiffusionData<R, V, D>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Self> {
        check_inputs(space, &diffusion, boundaries)?;
        work.dofs(space.full_dofs())?;
        work.entries(size(
            space.cells.len(),
            size(
                space.local,
                space.local,
                scioxide_core::quantity!("entries"),
            )?,
            scioxide_core::quantity!("entries"),
        )?)?;
        let mut coefficients =
            work.vector(diffusion.len(), scioxide_core::quantity!("coefficients"))?;
        coefficients.extend_from_slice(diffusion.values());
        let mut elements = work.filled(
            space.cells.len(),
            [V::ZERO; 100],
            scioxide_core::quantity!("elements"),
        )?;
        let mut loads = work.filled(
            space.cells.len(),
            [V::ZERO; 10],
            scioxide_core::quantity!("loads"),
        )?;
        let mut prescribed = work.filled(
            space.independent_dofs(),
            None,
            scioxide_core::quantity!("prescribed"),
        )?;
        volume_forms(
            space,
            diffusion.values(),
            data,
            &mut elements,
            &mut loads,
            work,
        )?;
        boundary_forms(
            space,
            boundaries,
            &mut elements,
            &mut loads,
            &mut prescribed,
            work,
        )?;
        let rhs = lifted_rhs(space, &elements, &loads, &prescribed, work)?;
        Ok(Self {
            space,
            diffusion: coefficients,
            boundaries,
            data,
            elements,
            rhs,
            prescribed,
            limits: work.limits,
            storage: work.bytes,
        })
    }
    /// Independent unknown count, including prescribed independent DOFs.
    pub fn unknowns(&self) -> usize {
        self.rhs.len()
    }
    /// Fully lifted load with prescribed diagonal equations.
    pub fn rhs(&self) -> &[V] {
        &self.rhs
    }
    /// Essential values in independent coordinates.
    pub fn prescribed_values(&self) -> &[Option<V>] {
        &self.prescribed
    }
    /// Conservative cumulative preparation payload.
    pub fn preparation_storage_bound(&self) -> usize {
        self.storage
    }
    /// Materialize the same constrained, symmetrically eliminated operator as CSR.
    /// Dense callers may explicitly convert this shared matrix.
    /// # Errors
    /// Rejects resource/size ceilings, allocation failure or unrepresentable arithmetic.
    pub fn assemble_csr(&self) -> Result<CsrMatrix<V>> {
        let mut work = Work::new(self.limits, None);
        let constrained = self.prescribed.iter().map(Option::is_some).collect();
        assemble(
            self.space,
            &self.elements,
            constrained,
            Elimination::Symmetric,
            &mut work,
        )
        .map(|(matrix, _, _)| matrix)
    }
    /// Matrix-free application with canonical full-row reductions; output is
    /// preserved on shape, arithmetic, cancellation or executor failures.
    fn apply(
        &self,
        input: &[V],
        output: &mut [V],
        scratch: &mut [V],
        execution: ExecutionContext<'_>,
    ) -> Result<()> {
        validate(input, self.unknowns(), scioxide_core::argument!("input"))?;
        if output.len() != self.unknowns() || scratch.len() < self.apply_workspace_len()? {
            return Err(invalid(
                if output.len() != self.unknowns() {
                    scioxide_core::argument!("output")
                } else {
                    scioxide_core::argument!("scratch")
                },
                "adaptive diffusion workspace shape",
            ));
        }
        execution.check()?;
        let full = self.space.full_dofs();
        let (x, rest) = scratch.split_at_mut(full);
        let (ax, rest) = rest.split_at_mut(full);
        let (staging, temporary) = rest.split_at_mut(self.unknowns());
        for (i, value) in x.iter_mut().enumerate() {
            *value = row(&self.space.constraints.matrix, i).try_fold(V::ZERO, |s, (j, w)| {
                if self.prescribed[j].is_none() {
                    finite(
                        s + V::from_real(w) * input[j],
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )
                } else {
                    Ok(s)
                }
            })?;
        }
        rows_into(
            full,
            ax,
            &mut temporary[..full],
            execution,
            self.space.options.execution,
            |node| {
                let mut result = V::ZERO;
                for &(cell, i) in &self.space.incidence[node] {
                    for (j, &column) in self.space.cell_dofs(cell)?.iter().enumerate() {
                        result = finite(
                            result + self.elements[cell][i * self.space.local + j] * x[column],
                            scioxide_core::NonFiniteOrigin::Intermediate,
                        )?;
                    }
                }
                Ok(result)
            },
        )?;
        self.space.constraints.apply_adjoint_into(
            ax,
            staging,
            &mut temporary[..self.unknowns()],
        )?;
        for (i, value) in staging.iter_mut().enumerate() {
            if self.prescribed[i].is_some() {
                *value = input[i];
            }
        }
        execution.check()?;
        output.copy_from_slice(staging);
        Ok(())
    }
    /// Bind a validated full nodal solution to its strong residual/flux provider.
    /// The original source/boundary callbacks must remain mathematically unchanged.
    /// # Errors
    /// Rejects shape, nonfinite values or inconsistent hanging constraints.
    pub fn residual<'b>(&'b self, values: &'b [V]) -> Result<impl AmrResidual<R, V, D> + 'b> {
        self.space.validate_field(values)?;
        Ok(DiffusionResidual { plan: self, values })
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> LinearOperator<V>
    for AmrDiffusionPlan<'_, R, V, D>
{
    fn rows(&self) -> usize {
        self.unknowns()
    }
    fn cols(&self) -> usize {
        self.unknowns()
    }
    fn apply_workspace_len(&self) -> Result<usize> {
        add(
            size(
                self.space.full_dofs(),
                3,
                scioxide_core::quantity!("diffusion_scratch"),
            )?,
            self.unknowns(),
            scioxide_core::quantity!("diffusion_scratch"),
        )
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
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> AdjointOperator<V>
    for AmrDiffusionPlan<'_, R, V, D>
{
    fn apply_adjoint_workspace_len(&self) -> Result<usize> {
        self.apply_workspace_len()
    }
    fn apply_adjoint_into(&self, input: &[V], output: &mut [V], scratch: &mut [V]) -> Result<()> {
        self.apply_into(input, output, scratch)
    }
    fn apply_adjoint_into_scoped(
        &self,
        input: &[V],
        output: &mut [V],
        scratch: &mut [V],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, V>,
    ) -> Result<scioxide_core::Invocation<()>> {
        self.apply_into_scoped(input, output, scratch, context)
    }
}
/// Reject foreign meshes, vector boundaries, complex or negative Robin
/// coefficients and nonpositive or nonfinite diffusion.
fn check_inputs<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    space: &AmrSpace<R, D>,
    diffusion: &FieldView<'_, R, Cell, D>,
    boundaries: &crate::TaggedBoundaries<R, V, D>,
) -> Result<()> {
    space
        .mesh
        .mesh_token()
        .ensure_compatible(diffusion.mesh_token())?;
    space
        .mesh
        .mesh_token()
        .ensure_compatible(boundaries.mesh_token())?;
    if boundaries.components() != 1 {
        return Err(invalid(
            scioxide_core::argument!("boundaries"),
            "adaptive diffusion is scalar",
        ));
    }
    if boundaries.conditions().iter().any(|c| {
        matches!(c.kind(), crate::BoundaryKind::Robin { alpha }
            if alpha.imaginary_component() != R::ZERO || alpha.real_component() < R::ZERO)
    }) {
        return Err(invalid(
            scioxide_core::argument!("boundaries"),
            "Robin coefficient must be real nonnegative",
        ));
    }
    validate(
        diffusion.values(),
        space.mesh.cell_count(),
        scioxide_core::argument!("diffusion"),
    )?;
    if diffusion.values().iter().any(|&k| k <= R::ZERO) {
        return Err(invalid(
            scioxide_core::argument!("diffusion"),
            "diffusion must be positive",
        ));
    }
    Ok(())
}

/// Scaled stiffness and source loads of every cell.
fn volume_forms<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    space: &AmrSpace<R, D>,
    diffusion: &[R],
    data: &dyn AmrDiffusionData<R, V, D>,
    elements: &mut [[V; 100]],
    loads: &mut [[V; 10]],
    work: &mut Work<'_, '_, '_>,
) -> Result<()> {
    for (cell, element) in elements.iter_mut().enumerate() {
        work.tick(size(
            space.local,
            space.local,
            scioxide_core::quantity!("work"),
        )?)?;
        for (value, &stiffness) in element.iter_mut().zip(&space.cells[cell].stiffness) {
            *value = V::from_real(finite(
                diffusion[cell] * stiffness,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?);
        }
        for q in &space.volume {
            work.provider()?;
            let f = finite(
                data.source(cell, space.physical(cell, q.point)?)?,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
            let sample = space.reference_sample(q.point, scioxide_core::argument!("space"))?;
            let w = finite(
                q.weight * space.cells[cell].map.determinant,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
            for (i, value) in loads[cell].iter_mut().enumerate().take(space.local) {
                *value = finite(
                    *value + V::from_real(w * sample.values[i]) * f,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
    }
    Ok(())
}

/// Natural and Robin data loads, Robin mass and Dirichlet values of every exterior face.
fn boundary_forms<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    space: &AmrSpace<R, D>,
    boundaries: &crate::TaggedBoundaries<R, V, D>,
    elements: &mut [[V; 100]],
    loads: &mut [[V; 10]],
    prescribed: &mut [Option<V>],
    work: &mut Work<'_, '_, '_>,
) -> Result<()> {
    for (interface_id, interface) in space.mesh.interfaces().iter().enumerate() {
        if interface.right.is_some() {
            continue;
        }
        let face = interface.left_face;
        let condition = boundaries.slot(face, 0)?;
        let kind = boundaries.kind_at(face, 0)?;
        let cell = interface.left_cell;
        if let crate::BoundaryKind::Dirichlet = kind {
            let normal = space
                .face_sample(interface_id, 0, scioxide_core::argument!("space"))?
                .normal;
            dirichlet_values(space, boundaries, cell, face, normal, prescribed, work)?;
            continue;
        }
        let alpha = match kind {
            crate::BoundaryKind::Robin { alpha } => alpha.real_component(),
            _ => R::ZERO,
        };
        for q in 0..space.facets.len() {
            let sample = space.face_sample(interface_id, q, scioxide_core::argument!("space"))?;
            work.provider()?;
            let g = boundaries.evaluate(
                condition,
                crate::BoundaryPoint {
                    point: sample.point,
                    normal: sample.normal,
                    component: 0,
                },
                R::ZERO,
            )?;
            let basis = space.reference_sample(
                space.reference(cell, sample.point, scioxide_core::argument!("space"))?,
                scioxide_core::argument!("space"),
            )?;
            work.tick(size(
                space.local,
                space.local,
                scioxide_core::quantity!("work"),
            )?)?;
            for (i, load) in loads[cell].iter_mut().enumerate().take(space.local) {
                *load = finite(
                    *load + V::from_real(sample.weight * basis.values[i]) * g,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
                for j in 0..space.local {
                    let slot = i * space.local + j;
                    elements[cell][slot] = finite(
                        elements[cell][slot]
                            + V::from_real(
                                sample.weight * alpha * basis.values[i] * basis.values[j],
                            ),
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// Dirichlet values of the independent nodes of `cell` on its exterior `face`, under the bound conflict policy.
fn dirichlet_values<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    space: &AmrSpace<R, D>,
    boundaries: &crate::TaggedBoundaries<R, V, D>,
    cell: usize,
    face: usize,
    normal: [R; D],
    prescribed: &mut [Option<V>],
    work: &mut Work<'_, '_, '_>,
) -> Result<()> {
    let condition = boundaries.slot(face, 0)?;
    for &node in space.cell_dofs(cell)? {
        work.tick(1)?;
        if !space.on_face(node, face)? {
            continue;
        }
        work.provider()?;
        let value = boundaries.evaluate(
            condition,
            crate::BoundaryPoint {
                point: space.points[node],
                normal,
                component: 0,
            },
            R::ZERO,
        )?;
        if let Ok(i) = space.independent_nodes().binary_search(&node) {
            prescribed[i] = Some(
                crate::boundary::select_prescribed(
                    prescribed[i].map(|value| crate::boundary::Prescribed {
                        value,
                        rank: (0, 0),
                    }),
                    crate::boundary::Prescribed {
                        value,
                        rank: (0, 0),
                    },
                    boundaries.conflict_policy(),
                )?
                .value,
            );
        }
    }
    Ok(())
}

/// Route the constrained element matrices `Cᵀ Aₑ C` of every cell, in
/// independent coordinates, through one eliminating builder.
fn assemble<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    space: &AmrSpace<R, D>,
    elements: &[[V; 100]],
    constrained: Vec<bool>,
    elimination: Elimination,
    work: &mut Work<'_, '_, '_>,
) -> Result<Eliminated<V>> {
    let mut builder = OperatorBuilder::with_elimination(constrained, 0, elimination)?;
    let mut entries = 0usize;
    for (cell, element) in elements.iter().enumerate() {
        for (i, &a) in space.cell_dofs(cell)?.iter().enumerate() {
            for (j, &b) in space.cell_dofs(cell)?.iter().enumerate() {
                for (r, x) in row(&space.constraints.matrix, a) {
                    if builder.constrained()[r] {
                        continue;
                    }
                    for (c, y) in row(&space.constraints.matrix, b) {
                        entries = add(entries, 1, scioxide_core::quantity!("entries"))?;
                        work.entries(entries)?;
                        work.tick(1)?;
                        work.charge::<(usize, usize, V)>(1)?;
                        builder.push(
                            r,
                            c,
                            finite(
                                V::from_real(x * y) * element[i * space.local + j],
                                scioxide_core::NonFiniteOrigin::Intermediate,
                            )?,
                        )?;
                    }
                }
            }
        }
    }
    builder.finish()
}

/// Constrained load `Cᵀ f` lifted by the prescribed independent values.
fn lifted_rhs<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    space: &AmrSpace<R, D>,
    elements: &[[V; 100]],
    loads: &[[V; 10]],
    prescribed: &[Option<V>],
    work: &mut Work<'_, '_, '_>,
) -> Result<Vec<V>> {
    let mut rhs = work.filled(
        space.independent_dofs(),
        V::ZERO,
        scioxide_core::quantity!("rhs"),
    )?;
    for (cell, load) in loads.iter().enumerate() {
        for (i, &node) in space.cell_dofs(cell)?.iter().enumerate() {
            for (j, weight) in row(&space.constraints.matrix, node) {
                work.tick(1)?;
                rhs[j] = finite(
                    rhs[j] + V::from_real(weight) * load[i],
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
    }
    for (value, fixed) in rhs.iter_mut().zip(prescribed) {
        if let Some(fixed) = fixed {
            *value = *fixed;
        }
    }
    let constrained = prescribed.iter().map(Option::is_some).collect();
    let (_, lift, _) = assemble(space, elements, constrained, Elimination::Lift, work)?;
    crate::discretization::apply_lift(&lift, &mut rhs)?;
    Ok(rhs)
}

struct DiffusionResidual<'a, 'b, R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    plan: &'a AmrDiffusionPlan<'b, R, V, D>,
    values: &'a [V],
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> DiffusionResidual<'_, '_, R, V, D> {
    fn value(&self, cell: usize, point: [R; D]) -> Result<V> {
        let space = self.plan.space;
        let sample = space.reference_sample(
            space.reference(cell, point, scioxide_core::argument!("point"))?,
            scioxide_core::argument!("point"),
        )?;
        space
            .cell_dofs(cell)?
            .iter()
            .enumerate()
            .try_fold(V::ZERO, |sum, (i, &node)| {
                finite(
                    sum + V::from_real(sample.values[i]) * self.values[node],
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )
            })
    }
    fn laplacian(&self, cell: usize) -> Result<V> {
        let space = self.plan.space;
        if space.order != AmrOrder::P2 {
            return Ok(V::ZERO);
        }
        let laplacians = space.laplacians(cell)?;
        space
            .cell_dofs(cell)?
            .iter()
            .enumerate()
            .try_fold(V::ZERO, |sum, (i, &node)| {
                finite(
                    sum + V::from_real(laplacians[i]) * self.values[node],
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )
            })
    }
    fn gradient(&self, cell: usize, point: [R; D]) -> Result<[V; D]> {
        let space = self.plan.space;
        let sample = space.sample(
            cell,
            space.reference(cell, point, scioxide_core::argument!("point"))?,
            scioxide_core::argument!("point"),
        )?;
        let mut gradient = [V::ZERO; D];
        for (i, &node) in space.cell_dofs(cell)?.iter().enumerate() {
            for (a, g) in gradient.iter_mut().enumerate() {
                *g = finite(
                    *g + V::from_real(sample.gradients[i][a]) * self.values[node],
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        Ok(gradient)
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> AmrResidual<R, V, D>
    for DiffusionResidual<'_, '_, R, V, D>
{
    fn volume(&self, cell: usize, point: [R; D]) -> Result<V> {
        finite(
            self.plan.data.source(cell, point)?
                + V::from_real(self.plan.diffusion[cell]) * self.laplacian(cell)?,
            scioxide_core::NonFiniteOrigin::Intermediate,
        )
    }
    fn flux(&self, cell: usize, point: [R; D], normal: [R; D]) -> Result<V> {
        self.gradient(cell, point)?
            .iter()
            .zip(normal)
            .try_fold(V::ZERO, |s, (&g, n)| {
                finite(
                    s + V::from_real(self.plan.diffusion[cell] * n) * g,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )
            })
    }
    fn boundary_flux(&self, face: usize, point: [R; D], normal: [R; D]) -> Result<Option<V>> {
        let boundaries = self.plan.boundaries;
        let cell = self.plan.space.exterior_cell(face)?;
        let condition = boundaries.slot(face, 0)?;
        let site = crate::BoundaryPoint {
            point,
            normal,
            component: 0,
        };
        match boundaries.kind_at(face, 0)? {
            crate::BoundaryKind::Dirichlet => Ok(None),
            crate::BoundaryKind::Natural => {
                Ok(Some(boundaries.evaluate(condition, site, R::ZERO)?))
            }
            crate::BoundaryKind::Robin { alpha } => Ok(Some(finite(
                boundaries.evaluate(condition, site, R::ZERO)? - alpha * self.value(cell, point)?,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?)),
        }
    }
}
