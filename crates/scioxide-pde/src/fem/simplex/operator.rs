use super::*;

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> SimplexFemPlan<R, V, D> {
    /// Reserve one staging value per structural operator entry for repeated assembly.
    /// # Errors
    /// Returns allocation and size failures.
    pub fn assembly_workspace(&self) -> Result<SimplexCsrWorkspace<V>, SciOxideError> {
        self.assembly.workspace()
    }

    /// Assemble into an existing operator CSR in O(element entries) work without allocation.
    /// The pattern includes numerical zeros; an error preserves `output`.
    /// # Errors
    /// Rejects a different pattern, insufficient workspace, nonfinite sums or execution stops.
    pub fn assemble_csr_into<'scope, 'capability>(
        &self,
        output: &mut CsrMatrix<V>,
        workspace: &mut SimplexCsrWorkspace<V>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "simplex-csr", 1),
            |context| {
                self.assembly.assemble_into(
                    &self.space,
                    output,
                    workspace,
                    &|cell, i, j| Ok(self.matrix_entry(cell, i, j)),
                    &[],
                    context.execution(),
                )
            },
        )
    }
    /// Shared underlying simplex topology and its vertex coordinates.
    pub const fn mesh(&self) -> &SimplexMesh<R, D> {
        &self.space.mesh
    }
    /// Solution polynomial degree.
    pub const fn order(&self) -> SimplexFemOrder {
        self.space.order
    }
    /// Number of field components.
    pub const fn components(&self) -> usize {
        self.components
    }
    /// Degrees of freedom per component, vertices followed by edges for P2.
    pub const fn dofs_per_component(&self) -> usize {
        self.space.nodes
    }
    /// Global component-major scalar unknown count.
    pub fn unknowns(&self) -> usize {
        self.constrained.len()
    }
    /// Upper bound on emitted CSR triplets, including prescribed diagonals.
    pub const fn entry_bound(&self) -> usize {
        self.entry_bound
    }
    /// Conservative simultaneously-live preparation storage estimate, in bytes.
    pub const fn preparation_storage_bound(&self) -> usize {
        self.storage_bound
    }
    /// Bound boundary conditions, including their data.
    pub const fn boundaries(&self) -> &crate::TaggedBoundaries<R, V, D> {
        &self.boundaries
    }
    pub(super) const fn layout(&self) -> Layout {
        Layout {
            order: self.space.order,
            components: self.components,
            pressure: false,
        }
    }
    /// Global scalar-node indices for one cell, before component offsets.
    /// # Errors
    /// Rejects an invalid cell index.
    pub fn cell_dofs(&self, cell: usize) -> Result<&[usize], SciOxideError> {
        self.space.cell_dofs(cell)
    }
    /// Physical interpolation node, including curved edge coordinates for P2.
    /// # Errors
    /// Rejects an invalid node index or unrepresentable affine midpoint.
    pub fn node_point(&self, node: usize) -> Result<[R; D], SciOxideError> {
        self.space.dof_point(node)
    }
    /// Map a reference point, checking positive orientation and conditioning.
    /// # Errors
    /// Rejects invalid cell/point, failed Jacobian checks or unrepresentable values.
    pub fn map(
        &self,
        cell: usize,
        reference: [R; D],
    ) -> Result<SimplexFemMappedPoint<R, D>, SciOxideError> {
        self.space.map_point(cell, reference)
    }
    /// Evaluate reference basis values and gradients into exact-length outputs.
    /// Outputs are unchanged on failure; this method allocates no storage.
    /// # Errors
    /// Rejects wrong lengths or points outside the reference simplex.
    pub fn basis_into(
        &self,
        reference: [R; D],
        values: &mut [R],
        gradients: &mut [[R; D]],
    ) -> Result<(), SciOxideError> {
        let n = self.order().nodes(D);
        if values.len() != n || gradients.len() != n {
            return Err(invalid(
                if values.len() != n {
                    scioxide_core::argument!("values")
                } else {
                    scioxide_core::argument!("gradients")
                },
                "basis output shape",
            ));
        }
        let sample = self.space.basis().sample(reference)?;
        values.copy_from_slice(&sample.values[..n]);
        gradients.copy_from_slice(&sample.gradients[..n]);
        Ok(())
    }
    /// Borrow the full, unconstrained local weak-form matrix in component-major order.
    /// # Errors
    /// Rejects a cell index outside the mesh.
    pub fn element_matrix(&self, cell: usize) -> Result<&[V], SciOxideError> {
        self.space.cell_dofs(cell)?;
        let n = self.layout().local::<D>();
        self.matrices
            .get(cell * n * n..(cell + 1) * n * n)
            .ok_or_else(|| {
                SciOxideError::out_of_bounds(
                    scioxide_core::quantity!("cells"),
                    cell,
                    self.space.cell_count(),
                )
            })
    }
    /// Borrow the scalar consistent mass of one cell in local-node order.
    /// # Errors
    /// Rejects a cell index outside the mesh.
    pub fn element_mass(&self, cell: usize) -> Result<&[R], SciOxideError> {
        self.space.cell_dofs(cell)?;
        let n = self.space.order.nodes(D);
        self.masses
            .get(cell * n * n..(cell + 1) * n * n)
            .ok_or_else(|| {
                SciOxideError::out_of_bounds(
                    scioxide_core::quantity!("cells"),
                    cell,
                    self.space.cell_count(),
                )
            })
    }
    fn mass_entry(&self, cell: usize, i: usize, j: usize) -> V {
        let l = self.order().nodes(D);
        if i / l == j / l {
            V::from_real(self.masses[(cell * l + i % l) * l + j % l])
        } else {
            V::ZERO
        }
    }
    fn matrix_entry(&self, cell: usize, i: usize, j: usize) -> V {
        self.matrices[(cell * self.layout().local::<D>() + i) * self.layout().local::<D>() + j]
    }

    /// Materialize consistent mass, independently replicated for each component.
    /// This is the full unconstrained mass, with no row-sum lumping.
    /// # Errors
    /// Propagates checked size, allocation and arithmetic failures.
    pub fn assemble_mass_csr(&self) -> Result<CsrMatrix<V>, SciOxideError> {
        self.mass_pattern()
            .matrix(&|cell, i, j| Ok(self.mass_entry(cell, i, j)), &[])
    }
    fn mass_pattern(&self) -> &AssemblyPattern {
        self.mass_assembly.as_deref().unwrap_or(&self.assembly)
    }
    /// Reserve numerical staging for repeated consistent-mass assembly.
    /// # Errors
    /// Returns checked size or allocation failures.
    pub fn mass_assembly_workspace(&self) -> Result<SimplexCsrWorkspace<V>, SciOxideError> {
        self.mass_pattern().workspace()
    }
    /// Refill a matching unconstrained mass CSR without allocation in linear work.
    /// An error preserves `output`.
    /// # Errors
    /// Rejects mismatched patterns/workspaces, nonfinite sums and execution stops.
    pub fn assemble_mass_csr_into<'scope, 'capability>(
        &self,
        output: &mut CsrMatrix<V>,
        workspace: &mut SimplexCsrWorkspace<V>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "simplex-mass-csr", 1),
            |context| {
                self.mass_pattern().assemble_into(
                    &self.space,
                    output,
                    workspace,
                    &|cell, i, j| Ok(self.mass_entry(cell, i, j)),
                    &[],
                    context.execution(),
                )
            },
        )
    }

    /// Unconstrained consistent mass, without boundary elimination, as an operator.
    ///
    /// Its application uses the operator protocol's transactional scratch and the
    /// plan's execution policy, exactly as the stiffness operator does.
    pub const fn mass_operator(&self) -> SimplexFemMassOperator<'_, R, V, D> {
        SimplexFemMassOperator { plan: self }
    }
    fn work(&self) -> usize {
        self.unknowns()
    }
    fn apply_inner(
        &self,
        input: &[V],
        output: &mut [V],
        scratch: &mut [V],
        mass: bool,
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        execution.check()?;
        space::check_buffers(self.unknowns(), input, output, scratch, self.work())?;
        let staged = &mut scratch[..self.work()];
        if mass {
            self.space.apply(
                self.layout(),
                &|cell, i, j| self.mass_entry(cell, i, j),
                None,
                input,
                staged,
                execution,
            )?;
        } else {
            self.space.apply(
                self.layout(),
                &|cell, i, j| self.matrix_entry(cell, i, j),
                Some(&self.constrained),
                input,
                staged,
                execution,
            )?;
        }
        output.copy_from_slice(staged);
        Ok(())
    }

    /// Pack borrowed shared vertex/edge fields into the scalar DOF layout.
    /// All identities, lengths and values are checked before output changes.
    /// # Errors
    /// Rejects incorrect component/order layouts, foreign mesh revisions or nonfinite values.
    pub fn pack_fields_into(
        &self,
        fields: SimplexFemFieldView<'_, '_, V, D>,
        output: &mut [V],
    ) -> Result<(), SciOxideError> {
        self.space.pack_into(self.layout(), fields, None, output)
    }
    /// Create owned shared fields from the component-major scalar layout.
    /// # Errors
    /// Rejects shape/nonfinite data and allocation failures.
    pub fn fields(&self, values: &[V]) -> Result<SimplexFemField<V, D>, SciOxideError> {
        self.space
            .unpack(self.layout(), values)
            .map(|(fields, _)| fields)
    }
    /// Apply borrowed shared fields and return owned shared fields.
    /// # Errors
    /// Propagates field validation, operator and allocation failures.
    pub fn apply_fields(
        &self,
        fields: SimplexFemFieldView<'_, '_, V, D>,
    ) -> Result<SimplexFemField<V, D>, SciOxideError> {
        let mut input = try_filled(
            self.unknowns(),
            V::ZERO,
            scioxide_core::quantity!("packed_input"),
        )?;
        self.pack_fields_into(fields, &mut input)?;
        let mut output = try_filled(self.unknowns(), V::ZERO, scioxide_core::quantity!("output"))?;
        let mut scratch = try_filled(self.work(), V::ZERO, scioxide_core::quantity!("scratch"))?;
        LinearOperator::apply_into(self, &input, &mut output, &mut scratch)?;
        self.fields(&output)
    }

    /// Evaluate field values and physical gradients at one reference point.
    /// Only this cell's input DOFs are read and validated, and outputs change only
    /// after every component evaluated successfully. No scratch is needed.
    /// # Errors
    /// Rejects shapes, cell/point/geometry, nonfinite local input or arithmetic failure.
    pub fn evaluate_into(
        &self,
        cell: usize,
        reference: [R; D],
        input: &[V],
        values: &mut [V],
        gradients: &mut [[V; D]],
    ) -> Result<(), SciOxideError> {
        if input.len() != self.unknowns()
            || values.len() != self.components
            || gradients.len() != self.components
        {
            return Err(invalid(
                if input.len() != self.unknowns() {
                    scioxide_core::argument!("input")
                } else if values.len() != self.components {
                    scioxide_core::argument!("values")
                } else {
                    scioxide_core::argument!("gradients")
                },
                "FEM evaluation shape",
            ));
        }
        self.space
            .evaluate_into(cell, reference, input, values, gradients)
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> LinearOperator<V>
    for SimplexFemPlan<R, V, D>
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
    /// Apply the prepared homogeneous operator sequentially; output is unchanged on error.
    fn apply_into(
        &self,
        input: &[V],
        output: &mut [V],
        workspace: &mut [V],
    ) -> Result<(), SciOxideError> {
        self.apply_inner(
            input,
            output,
            workspace,
            false,
            ExecutionContext::sequential_native(),
        )
    }
    /// Apply with the caller's capabilities and the plan's execution policy. Every
    /// output row has the same accumulation order for all worker counts.
    fn apply_into_scoped(
        &self,
        input: &[V],
        output: &mut [V],
        workspace: &mut [V],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, V>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        // The enclosing iterative solver admits and counts this provider call.
        // Do not consume its iteration budget a second time inside an operator.
        self.apply_inner(input, output, workspace, false, context.execution())
            .map(scioxide_core::Invocation::Evaluated)
    }
}

/// Borrowed unconstrained consistent mass of a [`SimplexFemPlan`].
///
/// Obtained from [`SimplexFemPlan::mass_operator`]; the mass is real symmetric, so
/// it is its own adjoint.
#[derive(Debug, Clone, Copy)]
pub struct SimplexFemMassOperator<'a, R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    plan: &'a SimplexFemPlan<R, V, D>,
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> LinearOperator<V>
    for SimplexFemMassOperator<'_, R, V, D>
{
    fn rows(&self) -> usize {
        self.plan.unknowns()
    }
    fn cols(&self) -> usize {
        self.plan.unknowns()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.plan.work())
    }
    fn apply_into(
        &self,
        input: &[V],
        output: &mut [V],
        workspace: &mut [V],
    ) -> Result<(), SciOxideError> {
        self.plan.apply_inner(
            input,
            output,
            workspace,
            true,
            ExecutionContext::sequential_native(),
        )
    }
    fn apply_into_scoped(
        &self,
        input: &[V],
        output: &mut [V],
        workspace: &mut [V],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, V>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.plan
            .apply_inner(input, output, workspace, true, context.execution())
            .map(scioxide_core::Invocation::Evaluated)
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> scioxide_core::AdjointOperator<V>
    for SimplexFemMassOperator<'_, R, V, D>
{
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        self.apply_workspace_len()
    }
    fn apply_adjoint_into(
        &self,
        input: &[V],
        output: &mut [V],
        workspace: &mut [V],
    ) -> Result<(), SciOxideError> {
        self.apply_into(input, output, workspace)
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> scioxide_core::BlockOperator<V>
    for SimplexFemMassOperator<'_, R, V, D>
{
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>
    crate::SpatialDiscretization<R, V, D> for SimplexFemPlan<R, V, D>
{
    type Field = SimplexFemField<V, D>;

    fn mesh_token(&self) -> &MeshToken {
        self.space.mesh.mesh_token()
    }
    fn components(&self) -> usize {
        self.components
    }
    fn dof_point(&self, unknown: usize) -> Result<crate::FieldPoint<R, D>, SciOxideError> {
        if unknown >= self.unknowns() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("fem_unknown"),
                unknown,
                self.unknowns(),
            ));
        }
        Ok(crate::FieldPoint {
            point: SimplexFemPlan::node_point(self, unknown % self.space.nodes)?,
            component: unknown / self.space.nodes,
        })
    }
    fn constrained(&self) -> &[bool] {
        &self.constrained
    }
    fn lumped_mass(&self) -> &[R] {
        &self.lumped
    }
    fn has_constant_modes(&self) -> bool {
        self.constant_modes
    }
    /// Canonical CSR with symmetric Dirichlet elimination.
    fn assemble_csr(&self) -> Result<CsrMatrix<V>, SciOxideError> {
        self.assembly
            .matrix(&|cell, i, j| Ok(self.matrix_entry(cell, i, j)), &[])
    }
    fn load_time_dependence(&self) -> crate::TimeDependence {
        self.boundaries.time_dependence()
    }
    fn load_into(
        &self,
        source: &crate::FieldData<R, V, D>,
        time: R,
        out: &mut [V],
        scratch: &mut [V],
    ) -> Result<(), SciOxideError> {
        self.loads
            .load_transaction(&self.boundaries, source, time, out, scratch)
    }
    fn field(&self, values: &[V]) -> Result<Self::Field, SciOxideError> {
        self.fields(values)
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> crate::adjoint::Sealed
    for SimplexFemPlan<R, V, D>
{
    fn preparation(&self) -> &alloc::sync::Arc<()> {
        &self.loads.preparation
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>
    crate::DifferentiableDiscretization<R, V, D> for SimplexFemPlan<R, V, D>
{
    type Coefficients = dyn SimplexFemCoefficients<R, V, D>;

    fn parameterize(
        &self,
        values: &[R],
        tangents: &[crate::PdeParameterTangent<'_, R, V, D, Self::Coefficients>],
    ) -> Result<crate::PdeParameterization<R, V, D>, SciOxideError> {
        crate::adjoint::parameterize(
            self,
            &self.boundaries,
            &self.loads,
            values,
            tangents,
            |coefficients, robin| Ok((self.tangent_operator(coefficients, robin)?, None)),
        )
    }
}
