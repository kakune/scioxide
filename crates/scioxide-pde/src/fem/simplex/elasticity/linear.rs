use super::*;

/// Prepared isotropic Hooke elasticity, including stable mixed P2/P1 forms. Real moduli act on
/// real or complex displacement/load amplitudes. Shared displacement nodes use
/// [`crate::TaggedBoundaries`]' strict or last-condition Dirichlet policy.
#[derive(Clone)]
pub struct LinearElasticityPlan<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    pub(super) space: Space<R, V, D>,
    matrices: Vec<V>,
    rhs: Vec<V>,
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> LinearElasticityPlan<R, V, D> {
    /// Prepare geometry, fixed material and reference loads under sequential native
    /// execution.
    /// # Errors
    /// Unsupported domains, invalid geometry/data, resource and numerical failures.
    pub fn new(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        material: IsotropicElasticMaterial<R>,
        boundaries: &TaggedBoundaries<R, V, D>,
        body: &(dyn Fn([R; D]) -> Result<[V; D], SciOxideError> + Sync),
        options: ElasticityOptions<R>,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            mesh,
            geometry,
            material,
            boundaries,
            body,
            options,
            ExecutionContext::sequential_native(),
        )
    }
    /// Prepare as one epoch of `scope`; failures publish no plan.
    /// # Errors
    /// The [`Self::new`] failures plus budget, cancellation and deadline stops.
    pub fn build<'scope, 'capability>(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        material: IsotropicElasticMaterial<R>,
        boundaries: &TaggedBoundaries<R, V, D>,
        body: &(dyn Fn([R; D]) -> Result<[V; D], SciOxideError> + Sync),
        options: ElasticityOptions<R>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "linear-elasticity", 2),
            |context| Self::prepare(mesh, geometry, material, boundaries, body, options, context),
        )
    }
    fn prepare(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        material: IsotropicElasticMaterial<R>,
        boundaries: &TaggedBoundaries<R, V, D>,
        body: &(dyn Fn([R; D]) -> Result<[V; D], SciOxideError> + Sync),
        options: ElasticityOptions<R>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Self, SciOxideError> {
        let space = Space::prepare(
            mesh, geometry, material, boundaries, body, options, true, context,
        )?;
        let c = material::hooke(material, space.mixed(), space.kinematics)?;
        let local = space.local();
        let mut matrices = try_filled(
            space.layout.entries(&space.space.mesh)?,
            V::ZERO,
            scioxide_core::quantity!("matrices"),
        )?;
        // Mapped samples live only for one quadrature node on each worker.
        space.space.for_each_chunk(
            &mut matrices,
            local * local,
            context,
            &|cell, matrix, context| {
                admit(context, false)?;
                for reference in space.space.volume_nodes() {
                    let sample = space.space.sample(cell, reference)?;
                    space.element(&[sample], matrix, |_| Ok(c))?;
                }
                Ok(())
            },
        )?;
        let rhs = space.space.lifted_rhs(
            space.layout,
            &space.prescribed,
            &space.load,
            |cell, i, j| Ok(matrices[(cell * local + i) * local + j]),
        )?;
        Ok(Self {
            space,
            matrices,
            rhs,
        })
    }
    accessors!(V);
    /// Dirichlet-lifted load for the eliminated linear operator.
    pub fn rhs(&self) -> &[V] {
        &self.rhs
    }
    /// Borrow the full unconstrained local matrix, including pressure blocks.
    pub fn element_matrix(&self, cell: usize) -> Result<&[V], SciOxideError> {
        if cell >= self.mesh().cell_count() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("cells"),
                cell,
                self.mesh().cell_count(),
            ));
        }
        let n = self.space.local();
        Ok(&self.matrices[cell * n * n..(cell + 1) * n * n])
    }
    /// Assemble canonical CSR with symmetric Dirichlet elimination. Explicit dense conversion is caller-owned.
    pub fn assemble_csr(&self) -> Result<CsrMatrix<V>, SciOxideError> {
        self.space.csr(&self.matrices)
    }
    /// Reserve numerical CSR staging for repeated assembly.
    /// # Errors
    /// Returns size or allocation failures.
    pub fn assembly_workspace(&self) -> Result<SimplexCsrWorkspace<V>, SciOxideError> {
        self.space.assembly.workspace()
    }
    /// Refill a matching CSR without allocation, retaining explicit numerical zeros.
    /// An error preserves `output`; work is linear in the element entries.
    /// # Errors
    /// Rejects pattern/workspace mismatches, nonfinite sums and execution stops.
    pub fn assemble_csr_into<'scope, 'capability>(
        &self,
        output: &mut CsrMatrix<V>,
        workspace: &mut SimplexCsrWorkspace<V>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError> {
        let n = self.space.local();
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "elasticity-csr", 1),
            |context| {
                self.space.assembly.assemble_into(
                    &self.space.space,
                    output,
                    workspace,
                    &|cell, i, j| Ok(self.matrices[(cell * n + i) * n + j]),
                    &[],
                    context.execution(),
                )
            },
        )
    }
    fn apply(
        &self,
        x: &[V],
        output: &mut [V],
        scratch: &mut [V],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        self.space.check(x, output, scratch, self.space.work())?;
        execution.check()?;
        let n = self.space.local();
        let staged = &mut scratch[..self.unknowns()];
        self.space.space.apply(
            self.space.layout,
            &|cell, i, j| self.matrices[(cell * n + i) * n + j],
            Some(&self.space.constrained),
            x,
            staged,
            execution,
        )?;
        output.copy_from_slice(staged);
        Ok(())
    }
    /// Evaluate the physical stress using tensorial shear and the selected plane reduction.
    /// Returns errors for invalid cell, reference point, shape or nonfinite arithmetic.
    pub fn stress(
        &self,
        cell: usize,
        point: [R; D],
        x: &[V],
    ) -> Result<ElasticityStress<V>, SciOxideError> {
        self.space.check(x, x, &[], 0)?;
        let (sample, reference) = self.space.space.point_sample(cell, point)?;
        let mut h = self.space.variables(&sample, &reference.linear, x, false)?;
        let c = material::hooke(
            self.space.material,
            self.space.mixed(),
            ElasticityKinematics::ThreeDimensional,
        )?;
        if self.space.kinematics == ElasticityKinematics::PlaneStress {
            h[8] = -h
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != 8)
                .fold(V::ZERO, |s, (i, &v)| s + v * V::from_real(c[8][i]))
                / V::from_real(c[8][8]);
        }
        let mut stress = [[V::ZERO; 3]; 3];
        let mut strain = stress;
        for i in 0..3 {
            for j in 0..3 {
                strain[i][j] = finite(
                    h[3 * i + j] * V::from_real(R::HALF) + h[3 * j + i] * V::from_real(R::HALF),
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
                stress[i][j] = finite(
                    (0..10).fold(V::ZERO, |s, k| s + h[k] * V::from_real(c[3 * i + j][k])),
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        let pressure = if self.space.mixed() {
            h[9]
        } else {
            (h[0] + h[4] + h[8]) * V::from_real(self.space.material.bulk())
        };
        Ok(ElasticityStress {
            kinematic_tensor: strain,
            cauchy: stress,
            first_piola: stress,
            pressure: finite(pressure, scioxide_core::NonFiniteOrigin::Intermediate)?,
        })
    }
    /// Integrated quadratic energy and original load work for the supplied state.
    /// Mixed internal energy is a saddle functional, not positive stored energy.
    pub fn energy(&self, x: &[V]) -> Result<ElasticityEnergy<R>, SciOxideError> {
        self.space.check(x, x, &[], 0)?;
        let n = self.space.local();
        let mut energy = V::ZERO;
        for cell in 0..self.mesh().cell_count() {
            let a = self.element_matrix(cell)?;
            for i in 0..n {
                for j in 0..n {
                    energy = finite(
                        energy
                            + x[self.space.global(cell, i)].conj()
                                * a[i * n + j]
                                * x[self.space.global(cell, j)]
                                * V::from_real(R::HALF),
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?;
                }
            }
        }
        self.space.energy(energy.real_component(), x)
    }
}

pub(super) fn contract<R: RealScalar>(
    c: &[[R; 10]; 10],
    a: &[R; 10],
    b: &[R; 10],
) -> Result<R, SciOxideError> {
    let mut sum = R::ZERO;
    for (i, &x) in a.iter().enumerate().filter(|(_, x)| **x != R::ZERO) {
        for (j, &y) in b.iter().enumerate().filter(|(_, y)| **y != R::ZERO) {
            sum = finite(
                sum + x * c[i][j] * y,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
        }
    }
    Ok(sum)
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> LinearOperator<V>
    for LinearElasticityPlan<R, V, D>
{
    fn rows(&self) -> usize {
        self.unknowns()
    }
    fn cols(&self) -> usize {
        self.unknowns()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.space.work())
    }
    /// Apply the eliminated operator transactionally, sequentially.
    fn apply_into(&self, x: &[V], out: &mut [V], work: &mut [V]) -> Result<(), SciOxideError> {
        self.apply(x, out, work, ExecutionContext::sequential_native())
    }
    /// Apply with deterministic row reductions and the caller's capabilities.
    fn apply_into_scoped(
        &self,
        x: &[V],
        out: &mut [V],
        work: &mut [V],
        ctx: &mut scioxide_core::OperatorContext<'_, '_, '_, V>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        self.apply(x, out, work, ctx.execution())
            .map(scioxide_core::Invocation::Evaluated)
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> AdjointOperator<V>
    for LinearElasticityPlan<R, V, D>
{
    fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.space.work())
    }
    fn apply_adjoint_into(
        &self,
        x: &[V],
        out: &mut [V],
        work: &mut [V],
    ) -> Result<(), SciOxideError> {
        LinearOperator::apply_into(self, x, out, work)
    }
}
