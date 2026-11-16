use super::*;

/// Own reserved element tangents and CSR storage for successive hyperelastic states.
/// Construct with [`HyperelasticityPlan::linearization_workspace`]. Refreshing a
/// state allocates nothing; storage is O(cells × local unknowns² + CSR entries).
pub struct HyperelasticityLinearizationWorkspace<R: RealScalar> {
    storage: super::super::csr::TangentStorage<R>,
}

/// Prepared real neo-Hookean residual and consistent Jacobian action.
/// The plan is immutable; nonlinear iteration and accepted state belong to the caller's solver.
#[derive(Clone)]
pub struct HyperelasticityPlan<R: RealScalar, const D: usize> {
    pub(super) space: Space<R, R, D>,
    samples: CellSamples<R, D>,
}
impl<R: RealScalar, const D: usize> HyperelasticityPlan<R, D> {
    /// Reserve the element and CSR buffers used by [`Self::linearize`].
    /// # Errors
    /// Returns checked size or allocation failures.
    pub fn linearization_workspace(
        &self,
    ) -> Result<HyperelasticityLinearizationWorkspace<R>, SciOxideError> {
        Ok(HyperelasticityLinearizationWorkspace {
            storage: super::super::csr::TangentStorage::new(
                &self.space.assembly,
                self.space.layout.entries(self.mesh())?,
                &[],
            )?,
        })
    }
    /// Evaluate the consistent tangent once and borrow it for repeated products or assembly.
    /// Preparation costs O(cells × quadrature nodes × local unknowns²); subsequent
    /// products use the stored CSR. No material response is recomputed by the view.
    /// # Errors
    /// Returns invalid state/workspace, material, arithmetic, budget or execution errors.
    pub fn linearize<'a, 'scope, 'capability>(
        &'a self,
        point: &'a [R],
        workspace: &'a mut HyperelasticityLinearizationWorkspace<R>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<SimplexLinearization<'a, R>, SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "hyperelastic-linearization", 1),
            |context| self.linearize_in(point, &mut workspace.storage, context),
        )?;
        Ok(SimplexLinearization {
            point,
            matrix: &workspace.storage.matrix,
        })
    }
    fn linearize_in(
        &self,
        point: &[R],
        storage: &mut super::super::csr::TangentStorage<R>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.space.check(point, point, &[], 0)?;
        if storage.matrices.len() != self.space.layout.entries(self.mesh())? {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "hyperelastic linearization size",
            ));
        }
        let n = self.space.local();
        storage.matrices.fill(R::ZERO);
        self.space.space.for_each_chunk(
            &mut storage.matrices,
            n * n,
            context,
            &|cell, matrix, context| {
                self.space
                    .element(self.samples.cell(cell), matrix, |sample| {
                        let linear = &self.space.space.reference(sample).linear;
                        Ok(self.response(sample, linear, point, true, context)?.tangent)
                    })
            },
        )?;
        self.space.assembly.assemble_into(
            &self.space.space,
            &mut storage.matrix,
            &mut storage.assembly,
            &|cell, i, j| Ok(storage.matrices[(cell * n + i) * n + j]),
            &[],
            context.execution(),
        )
    }
    /// Prepare fixed geometry, material and dead reference loads under sequential
    /// native execution.
    /// # Errors
    /// Invalid geometry, unsupported forms, nonfinite data and exceeded limits.
    pub fn new(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        material: IsotropicElasticMaterial<R>,
        boundaries: &TaggedBoundaries<R, R, D>,
        body: &(dyn Fn([R; D]) -> Result<[R; D], SciOxideError> + Sync),
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
    /// Prepare as one epoch of `scope`, publishing only complete state.
    /// # Errors
    /// The [`Self::new`] failures plus budget, cancellation and deadline stops.
    pub fn build<'scope, 'capability>(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        material: IsotropicElasticMaterial<R>,
        boundaries: &TaggedBoundaries<R, R, D>,
        body: &(dyn Fn([R; D]) -> Result<[R; D], SciOxideError> + Sync),
        options: ElasticityOptions<R>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "hyperelasticity", 2),
            |context| {
                let space = Space::prepare(
                    mesh, geometry, material, boundaries, body, options, false, context,
                )?;
                let samples = space.space.samples(false, context)?;
                Ok(Self { space, samples })
            },
        )
    }
    accessors!(R);
    fn response(
        &self,
        sample: &Sample<R, D>,
        linear: &[R; 4],
        x: &[R],
        lift: bool,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<material::Response<R>, SciOxideError> {
        context.execution().check()?;
        let h = self.space.variables(sample, linear, x, lift)?;
        let f = core::array::from_fn(|i| {
            core::array::from_fn(|j| h[3 * i + j] + if i == j { R::ONE } else { R::ZERO })
        });
        material::neo_response(
            self.space.material,
            f,
            self.space.mixed().then_some(h[9]),
            self.space.kinematics,
            &self.space.plane_stress,
            context,
        )
    }
    fn kernel(
        &self,
        x: &[R],
        direction: Option<&[R]>,
        output: &mut [R],
        scratch: &mut [R],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.space.check(x, output, scratch, self.space.work())?;
        if let Some(v) = direction {
            self.space.check(v, output, &[], 0)?;
        }
        let (staged, local) = scratch[..self.space.work()].split_at_mut(self.unknowns());
        local.fill(R::ZERO);
        let n = self.space.local();
        self.space
            .space
            .for_each_chunk(local, n, context, &|cell, y, context| {
                for sample in self.samples.cell(cell) {
                    let linear = &self.space.space.reference(sample).linear;
                    let response = self.response(sample, linear, x, true, context)?;
                    let g = if let Some(v) = direction {
                        let mut h = [R::ZERO; 10];
                        for i in 0..n {
                            let row = self.space.global(cell, i);
                            if self.space.prescribed(row).is_none() {
                                let b = self.space.b(sample, linear, i);
                                for k in 0..10 {
                                    h[k] = finite(
                                        h[k] + v[row] * b[k],
                                        scioxide_core::NonFiniteOrigin::Intermediate,
                                    )?;
                                }
                            }
                        }
                        let mut action = [R::ZERO; 10];
                        for (i, a) in action.iter_mut().enumerate() {
                            *a = finite(
                                (0..10).fold(R::ZERO, |s, j| s + response.tangent[i][j] * h[j]),
                                scioxide_core::NonFiniteOrigin::Intermediate,
                            )?;
                        }
                        action
                    } else {
                        response.gradient
                    };
                    for (i, value) in y.iter_mut().enumerate() {
                        let b = self.space.b(sample, linear, i);
                        *value = finite(
                            *value + sample.weight * (0..10).fold(R::ZERO, |s, k| s + b[k] * g[k]),
                            scioxide_core::NonFiniteOrigin::Intermediate,
                        )?;
                    }
                }
                Ok(())
            })?;
        self.space
            .space
            .scatter(self.space.layout, local, staged, context.execution())?;
        for (i, value) in staged.iter_mut().enumerate() {
            if let Some(p) = self.space.prescribed(i) {
                *value = direction.map_or(x[i] - p, |v| v[i]);
            } else if direction.is_none() {
                *value = finite(
                    *value - self.space.load[i],
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        context.execution().check()?;
        output.copy_from_slice(staged);
        Ok(())
    }
    /// Explicitly assemble the consistent tangent as canonical CSR.
    /// Allocates element and CSR buffers. Repeated state updates use
    /// [`Self::linearize`] with reserved storage instead.
    /// # Errors
    /// Returns invalid state, material, arithmetic, size and allocation failures.
    pub fn assemble_tangent_csr(&self, x: &[R]) -> Result<CsrMatrix<R>, SciOxideError> {
        self.assemble_tangent_csr_scoped(x, ExecutionContext::sequential_native())
    }
    /// Explicit tangent assembly as one epoch of `scope`; local plane-stress work is
    /// charged to its ancestors.
    /// # Errors
    /// Propagates shape, material, budget and assembly failures.
    pub fn assemble_tangent_csr_scoped<'scope, 'capability>(
        &self,
        x: &[R],
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<CsrMatrix<R>, SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "hyperelastic-tangent", 2),
            |context| self.tangent_csr(x, context),
        )
    }
    fn tangent_csr(
        &self,
        x: &[R],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<CsrMatrix<R>, SciOxideError> {
        self.space.check(x, x, &[], 0)?;
        let mut workspace = self.linearization_workspace()?;
        self.linearize_in(x, &mut workspace.storage, context)?;
        Ok(workspace.storage.matrix)
    }
    /// Real internal/saddle energy and dead-load work, with no implicit boundary lifting.
    pub fn energy(&self, x: &[R]) -> Result<ElasticityEnergy<R>, SciOxideError> {
        self.energy_scoped(x, ExecutionContext::sequential_native())
    }
    /// Energy as one epoch of `scope`, with nested local solves charged to its ancestors.
    /// # Errors
    /// Propagates shape, material and budget failures.
    pub fn energy_scoped<'scope, 'capability>(
        &self,
        x: &[R],
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<ElasticityEnergy<R>, SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "hyperelastic-energy", 2),
            |context| self.energy_in(x, context),
        )
    }
    fn energy_in(
        &self,
        x: &[R],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<ElasticityEnergy<R>, SciOxideError> {
        self.space.check(x, x, &[], 0)?;
        let mut energy = R::ZERO;
        for sample in &self.samples.volume {
            let linear = &self.space.space.reference(sample).linear;
            energy = finite(
                energy + sample.weight * self.response(sample, linear, x, false, context)?.energy,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
        }
        self.space.energy(energy, x)
    }
    /// Evaluate first Piola and Cauchy stress, including the solved thickness stretch.
    pub fn stress(
        &self,
        cell: usize,
        point: [R; D],
        x: &[R],
    ) -> Result<ElasticityStress<R>, SciOxideError> {
        self.stress_scoped(cell, point, x, ExecutionContext::sequential_native())
    }
    /// Stress evaluation as one epoch of `scope`, with explicit local-solve accounting.
    /// # Errors
    /// Propagates cell, point, shape, material and budget failures.
    pub fn stress_scoped<'scope, 'capability>(
        &self,
        cell: usize,
        point: [R; D],
        x: &[R],
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<ElasticityStress<R>, SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "hyperelastic-stress", 2),
            |context| self.stress_in(cell, point, x, context),
        )
    }
    fn stress_in(
        &self,
        cell: usize,
        point: [R; D],
        x: &[R],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<ElasticityStress<R>, SciOxideError> {
        self.space.check(x, x, &[], 0)?;
        let (sample, reference) = self.space.space.point_sample(cell, point)?;
        let response = self.response(&sample, &reference.linear, x, false, context)?;
        let p: [[R; 3]; 3] =
            core::array::from_fn(|i| core::array::from_fn(|j| response.gradient[3 * i + j]));
        let mut cauchy = [[R::ZERO; 3]; 3];
        for (i, row) in cauchy.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = finite(
                    (0..3).fold(R::ZERO, |s, k| s + p[i][k] * response.deformation[j][k])
                        / response.jacobian,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        let pressure = if self.space.mixed() {
            self.space.variables(&sample, &reference.linear, x, false)?[9]
        } else {
            self.space.material.bulk() * (response.jacobian - R::ONE)
        };
        Ok(ElasticityStress {
            kinematic_tensor: response.deformation,
            cauchy,
            first_piola: p,
            pressure: finite(pressure, scioxide_core::NonFiniteOrigin::Intermediate)?,
        })
    }
}
impl<R: RealScalar, const D: usize> ResidualOperator<R> for HyperelasticityPlan<R, D> {
    fn input_dimension(&self) -> usize {
        self.unknowns()
    }
    fn output_dimension(&self) -> usize {
        self.unknowns()
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.space.work())
    }
    fn residual_into(&self, x: &[R], out: &mut [R], work: &mut [R]) -> Result<(), SciOxideError> {
        crate::attempt::sequential(
            scioxide_core::algorithm_id!("pde", "elasticity", 2),
            |ctx| self.kernel(x, None, out, work, ctx),
        )
    }
    fn residual_into_scoped(
        &self,
        x: &[R],
        out: &mut [R],
        work: &mut [R],
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.kernel(x, None, out, work, ctx)
    }
}
impl<R: RealScalar, const D: usize> JacobianOperator<R> for HyperelasticityPlan<R, D> {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.space.work())
    }
    fn jvp_into(
        &self,
        x: &[R],
        v: &[R],
        out: &mut [R],
        work: &mut [R],
    ) -> Result<(), SciOxideError> {
        crate::attempt::sequential(
            scioxide_core::algorithm_id!("pde", "elasticity", 2),
            |ctx| self.kernel(x, Some(v), out, work, ctx),
        )
    }
    fn jvp_into_scoped(
        &self,
        x: &[R],
        v: &[R],
        out: &mut [R],
        work: &mut [R],
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.kernel(x, Some(v), out, work, ctx)
    }
}
/// The neo-Hookean tangent is symmetric, so adjoint products are Jacobian products.
impl<R: RealScalar, const D: usize> crate::PdeAdjointJacobian<R> for HyperelasticityPlan<R, D> {
    fn vjp_into(
        &self,
        x: &[R],
        seed: &[R],
        out: &mut [R],
        work: &mut [R],
    ) -> Result<(), SciOxideError> {
        self.jvp_into(x, seed, out, work)
    }
    fn vjp_into_scoped(
        &self,
        x: &[R],
        seed: &[R],
        out: &mut [R],
        work: &mut [R],
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.jvp_into_scoped(x, seed, out, work, ctx)
    }
}
