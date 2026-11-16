use super::*;

/// Borrowed mixed operator with either viscous or velocity-mass leading block.
/// The divergence, pressure gauge and strongly constrained rows are identical.
pub struct IncompressibleOperator<'a, R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    pub(super) plan: &'a StokesPlan<R, V, D>,
    pub(super) mass: bool,
    pub(super) gauged: bool,
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> StokesPlan<R, V, D> {
    pub(super) fn prepare_matrices(
        &mut self,
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let l = self.space.order.nodes(D);
        let m = self.local();
        let Self {
            space,
            samples,
            matrices,
            masses,
            pressure_masses,
            gauge_weights,
            density,
            viscosity,
            ..
        } = self;
        let mut local_weights = try_filled(
            space.cell_count() * (D + 1),
            R::ZERO,
            scioxide_core::quantity!("gauge_elements"),
        )?;
        let mut blocks = try_with_capacity(
            space.cell_count(),
            scioxide_core::quantity!("flow_elements"),
        )?;
        for (((matrix, mass), pressure_mass), weights) in matrices
            .chunks_exact_mut(m * m)
            .zip(masses.chunks_exact_mut(l * l))
            .zip(pressure_masses.chunks_exact_mut((D + 1) * (D + 1)))
            .zip(local_weights.chunks_exact_mut(D + 1))
        {
            blocks.push((matrix, mass, pressure_mass, weights));
        }
        space.for_each_cell(
            &mut blocks,
            || Ok(()),
            &|cell, (matrix, mass, pressure_mass, weights), _, ctx| {
                admit(ctx, false)?;
                for s in samples.cell(cell) {
                    let reference = space.reference(s);
                    let (values, pressure) = (&reference.solution.values, &reference.linear);
                    for a in 0..D {
                        for i in 0..l {
                            for b in 0..D {
                                for j in 0..l {
                                    let dot = if a == b {
                                        s.gradients[i]
                                            .iter()
                                            .zip(&s.gradients[j])
                                            .fold(R::ZERO, |v, (&x, &y)| v + x * y)
                                    } else {
                                        R::ZERO
                                    };
                                    let k = (a * l + i) * m + b * l + j;
                                    matrix[k] = finite(
                                        matrix[k]
                                            + *viscosity
                                                * s.weight
                                                * (dot + s.gradients[i][b] * s.gradients[j][a]),
                                        scioxide_core::NonFiniteOrigin::Intermediate,
                                    )?;
                                }
                            }
                            for (j, &q) in pressure.iter().enumerate().take(D + 1) {
                                let value = finite(
                                    -s.weight * s.gradients[i][a] * q,
                                    scioxide_core::NonFiniteOrigin::Intermediate,
                                )?;
                                let index = (a * l + i) * m + D * l + j;
                                matrix[index] = finite(
                                    matrix[index] + value,
                                    scioxide_core::NonFiniteOrigin::Intermediate,
                                )?;
                                matrix[(D * l + j) * m + a * l + i] = matrix[index];
                            }
                        }
                    }
                    for i in 0..l {
                        for j in i..l {
                            mass[i * l + j] = finite(
                                mass[i * l + j] + *density * s.weight * values[i] * values[j],
                                scioxide_core::NonFiniteOrigin::Intermediate,
                            )?;
                            mass[j * l + i] = mass[i * l + j];
                        }
                    }
                    for i in 0..=D {
                        weights[i] = finite(
                            weights[i] + s.weight * pressure[i],
                            scioxide_core::NonFiniteOrigin::Intermediate,
                        )?;
                        for j in i..=D {
                            let k = i * (D + 1) + j;
                            pressure_mass[k] = finite(
                                pressure_mass[k] + s.weight * pressure[i] * pressure[j],
                                scioxide_core::NonFiniteOrigin::Intermediate,
                            )?;
                            pressure_mass[j * (D + 1) + i] = pressure_mass[k];
                        }
                    }
                }
                Ok(())
            },
            ctx,
        )?;
        drop(blocks);
        for (cell, weights) in local_weights.chunks_exact(D + 1).enumerate() {
            for (i, &weight) in weights.iter().enumerate() {
                let vertex = space.dofs(cell)[i];
                gauge_weights[vertex] = finite(
                    gauge_weights[vertex] + weight,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        Ok(())
    }
    /// Unit weights of orthogonal gauges and the targets of weighted-mean gauges.
    pub(super) fn prepare_gauges(
        &mut self,
        gauges: &[EllipticGauge<V>],
    ) -> Result<(), SciOxideError> {
        for (v, weight) in self.gauge_weights.iter_mut().enumerate() {
            let c = self.component[v];
            if matches!(
                gauges
                    .get(c)
                    .copied()
                    .unwrap_or(EllipticGauge::WeightedMean(V::ZERO)),
                EllipticGauge::Orthogonal
            ) {
                *weight = R::ONE;
            }
            if let (Some(k), EllipticGauge::WeightedMean(value)) = (
                self.gauge_index[c],
                gauges
                    .get(c)
                    .copied()
                    .unwrap_or(EllipticGauge::WeightedMean(V::ZERO)),
            ) {
                self.gauge_targets[k] = finite(
                    self.gauge_targets[k] + value * V::from_real(*weight),
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        Ok(())
    }
    /// Dirichlet-lifted physical load followed by the gauge targets.
    pub(super) fn prepare_rhs(&mut self) -> Result<(), SciOxideError> {
        let n = self.physical();
        let m = self.local();
        let mut rhs = self.space.lifted_rhs(
            Self::layout(),
            &self.prescribed,
            &self.load[..n],
            |cell, i, j| Ok(V::from_real(self.matrices[(cell * m + i) * m + j])),
        )?;
        rhs.try_reserve_exact(self.gauge_targets.len())
            .map_err(|_| {
                SciOxideError::allocation_failure(scioxide_core::quantity!("rhs"), self.load.len())
            })?;
        rhs.extend_from_slice(&self.gauge_targets);
        self.rhs = rhs;
        Ok(())
    }
    /// Gauge coupling triplets `(pressure, multiplier)` and their transposes.
    pub(super) fn gauge_entries(&self) -> impl Iterator<Item = (usize, usize, V)> + '_ {
        self.component.iter().enumerate().flat_map(move |(v, &c)| {
            self.gauge_index[c].into_iter().flat_map(move |k| {
                let (i, j) = (self.velocity_unknowns() + v, self.physical() + k);
                let w = V::from_real(self.gauge_weights[v]);
                [(i, j, w), (j, i, w)]
            })
        })
    }
    pub(super) fn validate_nullspaces(
        &self,
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.physical();
        let op = IncompressibleOperator {
            plan: self,
            mass: false,
            gauged: false,
        };
        let mut basis = try_filled(n, V::ZERO, scioxide_core::quantity!("basis"))?;
        // A maximum absolute row bound scales with the local stencil, rather
        // than summing unrelated elements as the mesh is refined.
        let mut row_bounds = try_filled(n, R::ZERO, scioxide_core::quantity!("row_bounds"))?;
        let m = self.local();
        for cell in 0..self.mesh().cell_count() {
            admit(ctx, false)?;
            for i in 0..m {
                let row = self.global(cell, i);
                if self.prescribed(row).is_some() {
                    continue;
                }
                for j in 0..m {
                    if self.prescribed(self.global(cell, j)).is_none() {
                        row_bounds[row] = finite(
                            row_bounds[row] + self.matrices[cell * m * m + i * m + j].abs(),
                            scioxide_core::NonFiniteOrigin::Intermediate,
                        )?;
                    }
                }
            }
        }
        let scale = row_bounds.into_iter().fold(R::ONE, R::max);
        for (c, k) in self.gauge_index.iter().enumerate() {
            if k.is_none() {
                continue;
            }
            admit(ctx, false)?;
            basis.fill(V::ZERO);
            let count = self.component.iter().filter(|&&v| v == c).count();
            let value = V::from_real(R::ONE / R::try_from_usize_exact(count)?.sqrt());
            for (v, &component) in self.component.iter().enumerate() {
                if component == c {
                    basis[self.velocity_unknowns() + v] = value;
                }
            }
            let view = BlockVectorView::new(&basis, n, 1)?;
            let plan = BorderedSystem::build(
                &op,
                view,
                view,
                scale,
                &scioxide_linalg::BorderedSystemOptions::new()
                    .with_validation_tolerance(self.compatibility)
                    .with_compatibility_tolerance(self.compatibility),
                &mut *ctx,
            )?;
            let mut load = try_filled(
                plan.augmented().rows(),
                V::ZERO,
                scioxide_core::quantity!("bordered_load"),
            )?;
            plan.prepare_rhs_into(
                &self.rhs[..n],
                &[V::ZERO][..],
                CompatibilityPolicy::RequireCompatible,
                scioxide_linalg::VectorViewMut::from_slice(&mut load),
                &mut scioxide_linalg::LinalgWorkspace::new(),
                &mut *ctx,
            )?;
        }
        Ok(())
    }
    pub(super) fn prescribed(&self, i: usize) -> Option<V> {
        self.prescribed.get(i).copied().flatten()
    }
    pub(super) fn check(
        &self,
        x: &[V],
        out: &[V],
        scratch: &[V],
        needed: usize,
    ) -> Result<(), SciOxideError> {
        super::space::check_buffers(self.unknowns(), x, out, scratch, needed)
    }
    /// Total unknowns, including explicit gauge multipliers.
    pub fn unknowns(&self) -> usize {
        self.rhs.len()
    }
    /// Number of packed velocity coordinates including prescribed entries.
    pub fn velocity_unknowns(&self) -> usize {
        D * self.space.nodes
    }
    /// Number of P2 nodes per velocity component.
    pub fn velocity_nodes(&self) -> usize {
        self.space.nodes
    }
    /// Borrow the exact prepared mesh.
    pub fn mesh(&self) -> &SimplexMesh<R, D> {
        &self.space.mesh
    }
    /// Physical interpolation point of a P2 velocity node.
    /// # Errors
    /// Rejects an out-of-range node.
    pub fn node_point(&self, i: usize) -> Result<[R; D], SciOxideError> {
        self.space.dof_point(i)
    }
    /// Dirichlet-lifted load and pressure constraints.
    pub fn rhs(&self) -> &[V] {
        &self.rhs
    }
    /// Original body/traction load; pressure and gauge entries are zero.
    pub fn load(&self) -> &[V] {
        &self.load
    }
    /// Scalar scratch of an operator application.
    pub(super) fn work(&self) -> usize {
        self.unknowns() + self.space.cell_count() * self.local()
    }
    /// Conservative preparation peak byte bound, excluding supplied inputs/providers.
    pub fn preparation_storage_bound(&self) -> usize {
        self.storage_bound
    }
    /// Upper bound on explicitly assembled CSR triplets.
    pub fn entry_bound(&self) -> usize {
        self.entry_bound
    }
    /// Borrow the mass/divergence saddle operator used for constrained acceleration.
    pub fn mass_system(&self) -> IncompressibleOperator<'_, R, V, D> {
        IncompressibleOperator {
            plan: self,
            mass: true,
            gauged: true,
        }
    }
    /// Borrow a complete unconstrained element's mixed viscous matrix.
    /// # Errors
    /// Rejects an invalid cell.
    pub fn element_matrix(&self, cell: usize) -> Result<&[R], SciOxideError> {
        if cell >= self.mesh().cell_count() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("cells"),
                cell,
                self.mesh().cell_count(),
            ));
        }
        let m = self.local();
        Ok(&self.matrices[cell * m * m..(cell + 1) * m * m])
    }
    /// Borrow the density-weighted scalar P2 element mass matrix.
    /// # Errors
    /// Rejects an invalid cell.
    pub fn element_velocity_mass(&self, cell: usize) -> Result<&[R], SciOxideError> {
        self.element_matrix(cell)?;
        let l = self.space.order.nodes(D);
        Ok(&self.masses[cell * l * l..(cell + 1) * l * l])
    }
    /// Borrow the P1 pressure element mass matrix for injected preconditioners.
    /// # Errors
    /// Rejects an invalid cell.
    pub fn element_pressure_mass(&self, cell: usize) -> Result<&[R], SciOxideError> {
        self.element_matrix(cell)?;
        let l = D + 1;
        Ok(&self.pressure_masses[cell * l * l..(cell + 1) * l * l])
    }
    /// Explicit CSR assembly. Dense conversion is caller-selected through CSR.
    /// # Errors
    /// Returns allocation, size and arithmetic errors.
    pub fn assemble_csr(&self) -> Result<CsrMatrix<V>, SciOxideError> {
        IncompressibleOperator {
            plan: self,
            mass: false,
            gauged: true,
        }
        .assemble_csr()
    }
    /// Reserve numerical staging for repeated gauged Stokes assembly.
    /// # Errors
    /// Returns size and allocation failures.
    pub fn assembly_workspace(&self) -> Result<SimplexCsrWorkspace<V>, SciOxideError> {
        self.assembly.workspace()
    }
    /// Refill a matching Stokes CSR without allocation, preserving output on error.
    /// # Errors
    /// Rejects pattern/workspace mismatches, nonfinite sums and execution stops.
    pub fn assemble_csr_into<'scope, 'capability>(
        &self,
        output: &mut CsrMatrix<V>,
        workspace: &mut SimplexCsrWorkspace<V>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        IncompressibleOperator {
            plan: self,
            mass: false,
            gauged: true,
        }
        .assemble_csr_into(output, workspace, scope)
    }
    /// Convert a checked packed state into shared owned fields; gauge multipliers
    /// are solver coordinates and are not physical pressure fields.
    /// # Errors
    /// Rejects shape, nonfinite values or allocation failure.
    pub fn fields(&self, x: &[V]) -> Result<IncompressibleField<V, D>, SciOxideError> {
        self.check(x, x, &[], 0)?;
        let (velocity, pressure) = self.space.unpack(Self::layout(), &x[..self.physical()])?;
        let pressure = pressure
            .ok_or_else(|| invalid(scioxide_core::argument!("x"), "flow state lacks pressure"))?;
        Ok(IncompressibleField { velocity, pressure })
    }
    /// Pack exact-mesh borrowed fields; initialize gauge multipliers to zero.
    /// # Errors
    /// Invalid shapes, identities or values leave output unchanged.
    pub fn pack_fields_into(
        &self,
        fields: IncompressibleFieldView<'_, '_, V, D>,
        out: &mut [V],
    ) -> Result<(), SciOxideError> {
        if out.len() != self.unknowns() {
            return Err(invalid(
                scioxide_core::argument!("out"),
                "flow field layout",
            ));
        }
        let n = self.physical();
        self.space.pack_into(
            Self::layout(),
            fields.velocity,
            Some(&fields.pressure),
            &mut out[..n],
        )?;
        out[n..].fill(V::ZERO);
        Ok(())
    }
    pub(super) fn values(
        &self,
        s: &Sample<R, D>,
        x: &[V],
        lift: bool,
    ) -> Result<([V; D], [[V; D]; D]), SciOxideError> {
        self.values_with(s, self.basis(s), x, lift)
    }
    /// Velocity and its gradient at a sample with P2 values `values`; `lift`
    /// replaces constrained coordinates by their prescribed values.
    pub(super) fn values_with(
        &self,
        s: &Sample<R, D>,
        values: &[R; MAX_NODES],
        x: &[V],
        lift: bool,
    ) -> Result<([V; D], [[V; D]; D]), SciOxideError> {
        let l = self.space.order.nodes(D);
        let mut u = [V::ZERO; D];
        let mut g = [[V::ZERO; D]; D];
        for a in 0..D {
            for (i, &phi) in values.iter().enumerate().take(l) {
                let row = self.global(s.cell, a * l + i);
                let v = if lift {
                    self.prescribed(row).unwrap_or(x[row])
                } else {
                    x[row]
                };
                u[a] = finite(
                    u[a] + v * V::from_real(phi),
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
                for (entry, &gradient) in g[a].iter_mut().zip(&s.gradients[i]) {
                    *entry = finite(
                        *entry + v * V::from_real(gradient),
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?;
                }
            }
        }
        Ok((u, g))
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>
    IncompressibleOperator<'_, R, V, D>
{
    pub(super) fn entry(&self, cell: usize, i: usize, j: usize) -> R {
        let p = self.plan;
        let l = p.space.order.nodes(D);
        let m = p.local();
        if self.mass && i < D * l && j < D * l {
            if i / l == j / l {
                p.masses[cell * l * l + (i % l) * l + j % l]
            } else {
                R::ZERO
            }
        } else {
            p.matrices[cell * m * m + i * m + j]
        }
    }
    /// Assemble this exact constrained operator into canonical CSR.
    /// # Errors
    /// Returns allocation, size and arithmetic errors.
    pub fn assemble_csr(&self) -> Result<CsrMatrix<V>, SciOxideError> {
        if self.gauged {
            self.plan.assembly.matrix(
                &|cell, i, j| Ok(V::from_real(self.entry(cell, i, j))),
                &self.plan.assembly_extra,
            )
        } else {
            self.assemble_with(|_, _, _| Ok(R::ZERO))
        }
    }
    /// Reserve staging values for the gauged operator's CSR assembly.
    /// # Errors
    /// Returns checked size and allocation failures.
    pub fn assembly_workspace(&self) -> Result<SimplexCsrWorkspace<V>, SciOxideError> {
        self.plan.assembly.workspace()
    }
    /// Refill a matching gauged CSR in linear work without allocation.
    /// An error preserves the output; numerical zeros remain stored.
    /// # Errors
    /// Rejects pattern/workspace mismatches, nonfinite arithmetic and execution stops.
    pub fn assemble_csr_into<'scope, 'capability>(
        &self,
        output: &mut CsrMatrix<V>,
        workspace: &mut SimplexCsrWorkspace<V>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "incompressible-csr", 1),
            |context| {
                self.plan.assembly.assemble_into(
                    &self.plan.space,
                    output,
                    workspace,
                    &|cell, i, j| Ok(V::from_real(self.entry(cell, i, j))),
                    &self.plan.assembly_extra,
                    context.execution(),
                )
            },
        )
    }
    /// Assemble the operator plus the velocity–velocity element blocks `extra`.
    pub(super) fn assemble_with(
        &self,
        mut extra: impl FnMut(usize, usize, usize) -> Result<R, SciOxideError>,
    ) -> Result<CsrMatrix<V>, SciOxideError> {
        let p = self.plan;
        let gauges = p.gauge_entries().filter(|_| self.gauged);
        p.space
            .assemble(
                StokesPlan::<R, V, D>::layout(),
                p.constrained[..self.rows()].to_vec(),
                crate::discretization::Elimination::Symmetric,
                |cell, i, j| Ok(V::from_real(self.entry(cell, i, j) + extra(cell, i, j)?)),
                gauges,
            )
            .map(|(matrix, _, _)| matrix)
    }
    pub(super) fn apply(
        &self,
        x: &[V],
        out: &mut [V],
        scratch: &mut [V],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let p = self.plan;
        let n = self.rows();
        super::space::check_buffers(n, x, out, scratch, self.apply_workspace_len()?)?;
        execution.check()?;
        let stage = &mut scratch[..n];
        let physical = p.physical();
        p.space.apply(
            StokesPlan::<R, V, D>::layout(),
            &|cell, i, j| V::from_real(self.entry(cell, i, j)),
            Some(&p.constrained[..physical]),
            x,
            &mut stage[..physical],
            execution,
        )?;
        if self.gauged {
            stage[physical..].fill(V::ZERO);
            for (i, j, w) in p.gauge_entries() {
                stage[i] = finite(
                    stage[i] + w * x[j],
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        execution.check()?;
        out.copy_from_slice(stage);
        Ok(())
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> LinearOperator<V>
    for IncompressibleOperator<'_, R, V, D>
{
    fn rows(&self) -> usize {
        if self.gauged {
            self.plan.unknowns()
        } else {
            self.plan.physical()
        }
    }
    fn cols(&self) -> usize {
        self.rows()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.rows() + self.plan.mesh().cell_count() * self.plan.local())
    }
    fn apply_into(&self, x: &[V], out: &mut [V], scratch: &mut [V]) -> Result<(), SciOxideError> {
        self.apply(x, out, scratch, ExecutionContext::sequential_native())
    }
    fn apply_into_scoped(
        &self,
        x: &[V],
        out: &mut [V],
        scratch: &mut [V],
        ctx: &mut scioxide_core::OperatorContext<'_, '_, '_, V>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply(x, out, scratch, ctx.execution())
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> AdjointOperator<V>
    for IncompressibleOperator<'_, R, V, D>
{
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.apply_workspace_len()
    }
    fn apply_adjoint_into(
        &self,
        x: &[V],
        out: &mut [V],
        scratch: &mut [V],
    ) -> Result<(), SciOxideError> {
        self.apply_into(x, out, scratch)
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> BlockOperator<V>
    for IncompressibleOperator<'_, R, V, D>
{
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> LinearOperator<V>
    for StokesPlan<R, V, D>
{
    fn rows(&self) -> usize {
        self.unknowns()
    }
    fn cols(&self) -> usize {
        self.unknowns()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.work())
    }
    fn apply_into(&self, x: &[V], out: &mut [V], scratch: &mut [V]) -> Result<(), SciOxideError> {
        IncompressibleOperator {
            plan: self,
            mass: false,
            gauged: true,
        }
        .apply_into(x, out, scratch)
    }
    fn apply_into_scoped(
        &self,
        x: &[V],
        out: &mut [V],
        scratch: &mut [V],
        ctx: &mut scioxide_core::OperatorContext<'_, '_, '_, V>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        IncompressibleOperator {
            plan: self,
            mass: false,
            gauged: true,
        }
        .apply_into_scoped(x, out, scratch, ctx)
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> AdjointOperator<V>
    for StokesPlan<R, V, D>
{
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.apply_workspace_len()
    }
    fn apply_adjoint_into(
        &self,
        x: &[V],
        out: &mut [V],
        scratch: &mut [V],
    ) -> Result<(), SciOxideError> {
        self.apply_into(x, out, scratch)
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> BlockOperator<V>
    for StokesPlan<R, V, D>
{
}
