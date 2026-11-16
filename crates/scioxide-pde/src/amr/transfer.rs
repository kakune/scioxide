use super::*;
use scioxide_core::EvaluationRole;
use scioxide_core::ResidualTolerance;
use scioxide_spatial::{EntityRelation, UnstructuredFace};

fn relation(r: &EntityRelation) -> Option<(usize, usize)> {
    match *r {
        EntityRelation::Unchanged {
            source,
            destination,
        }
        | EntityRelation::Refined {
            source,
            destination,
            ..
        }
        | EntityRelation::Coarsened {
            source,
            destination,
            ..
        } => Some((source, destination)),
        _ => None,
    }
}

/// Conservative piecewise-constant cell-average transfer on an exact topology map.
/// Application schedules rows with the configured execution policy.
#[derive(Clone)]
pub struct AmrCellTransfer<R: RealScalar> {
    source: MeshToken,
    destination: MeshToken,
    map: AmrMap<R>,
    storage: usize,
}
impl<R: RealScalar> AmrCellTransfer<R> {
    /// Prepare refinement and coarsening weights from represented physical volumes.
    /// Refinement distributes source mass over the complete destination family;
    /// coarsening sums source masses and divides by destination volume.
    /// # Errors
    /// Rejects endpoint mismatches, missing cell ancestry, unrepresentable volumes,
    /// and resource failures. Supports all spatial adaptive cell families and periodic cells.
    pub fn new<const D: usize>(
        source: &AdaptiveMesh<R, D>,
        destination: &AdaptiveMesh<R, D>,
        topology: &TopologyMap,
        options: AmrOptions<R>,
    ) -> Result<Self> {
        Self::build(
            source,
            destination,
            topology,
            options,
            ExecutionContext::sequential_native(),
        )
    }
    /// Prepare as one epoch of `scope`, with inherited work admission and cancellation.
    /// Only the limits and execution policy of `options` apply.
    /// # Errors
    /// Same errors as `new`, including scope budget exhaustion.
    pub fn build<'scope, 'capability, const D: usize>(
        source: &AdaptiveMesh<R, D>,
        destination: &AdaptiveMesh<R, D>,
        topology: &TopologyMap,
        options: AmrOptions<R>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<Self> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "amr-cell-transfer", 2),
            |context| {
                Self::prepare(
                    source,
                    destination,
                    topology,
                    options.execution,
                    &mut Work::new(options.limits, Some(context)),
                )
            },
        )
    }
    fn prepare<const D: usize>(
        source: &AdaptiveMesh<R, D>,
        destination: &AdaptiveMesh<R, D>,
        topology: &TopologyMap,
        policy: ExecutionPolicy,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Self> {
        topology.validate(source.mesh_token(), destination.mesh_token())?;
        work.dofs(source.cell_count())?;
        work.dofs(destination.cell_count())?;
        work.entries(topology.cells().len())?;
        let mut source_volumes = work.vector(
            source.cell_count(),
            scioxide_core::quantity!("source_volumes"),
        )?;
        let mut dest_volumes = work.vector(
            destination.cell_count(),
            scioxide_core::quantity!("dest_volumes"),
        )?;
        for i in 0..source.cell_count() {
            work.tick(1)?;
            source_volumes.push(source.cell_measure(i)?);
        }
        for i in 0..destination.cell_count() {
            work.tick(1)?;
            dest_volumes.push(destination.cell_measure(i)?);
        }
        let mut refined_volume = work.filled(
            source.cell_count(),
            R::ZERO,
            scioxide_core::quantity!("refined_volume"),
        )?;
        for r in topology.cells() {
            work.tick(1)?;
            if let EntityRelation::Refined {
                source: i,
                destination: j,
                ..
            } = *r
            {
                refined_volume[i] = finite(
                    refined_volume[i] + dest_volumes[j],
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        let mut entries =
            work.vector(topology.cells().len(), scioxide_core::quantity!("entries"))?;
        for r in topology.cells() {
            work.tick(1)?;
            let (i, j, weight) = match *r {
                EntityRelation::Unchanged {
                    source: i,
                    destination: j,
                } => (i, j, R::ONE),
                EntityRelation::Refined {
                    source: i,
                    destination: j,
                    ..
                } => (
                    i,
                    j,
                    finite(
                        source_volumes[i] / refined_volume[i],
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?,
                ),
                EntityRelation::Coarsened {
                    source: i,
                    destination: j,
                    ..
                } => (
                    i,
                    j,
                    finite(
                        source_volumes[i] / dest_volumes[j],
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?,
                ),
                _ => {
                    return Err(invalid(
                        scioxide_core::argument!("topology"),
                        "cell transfer requires covering cell ancestry",
                    ));
                }
            };
            entries.push((j, i, weight));
        }
        let matrix = csr(
            destination.cell_count(),
            source.cell_count(),
            &entries,
            work,
        )?;
        Ok(Self {
            source: source.mesh_token().clone(),
            destination: destination.mesh_token().clone(),
            map: AmrMap::new(matrix, policy),
            storage: work.bytes,
        })
    }
    /// Borrow the conservative linear map for sparse/dense/operator composition.
    pub fn operator(&self) -> &AmrMap<R> {
        &self.map
    }
    /// Exact destination field epoch.
    pub fn destination(&self) -> &MeshToken {
        &self.destination
    }
    /// Conservative cumulative preparation payload.
    pub fn preparation_storage_bound(&self) -> usize {
        self.storage
    }
    /// Transfer a borrowed source field with caller-owned output and staging; scratch
    /// needs [`LinearOperator::apply_workspace_len`] of [`Self::operator`].
    /// # Errors
    /// Rejects a stale source, shape/nonfinite input, arithmetic, cancellation and
    /// executor failures atomically.
    pub fn apply_into<V: NumericScalar<Real = R>, const D: usize>(
        &self,
        input: FieldView<'_, V, Cell, D>,
        output: &mut [V],
        scratch: &mut [V],
        execution: ExecutionContext<'_>,
    ) -> Result<()> {
        self.source.ensure_compatible(input.mesh_token())?;
        self.map.apply(input.values(), output, scratch, execution)
    }
    /// Allocate a destination cell field explicitly.
    /// # Errors
    /// Rejects either foreign endpoint and propagates allocation/evaluation failure.
    pub fn field<V: NumericScalar<Real = R>, const D: usize>(
        &self,
        destination: &AdaptiveMesh<R, D>,
        input: FieldView<'_, V, Cell, D>,
    ) -> Result<Field<V, Cell, D>> {
        self.destination
            .ensure_compatible(destination.mesh_token())?;
        let rows = self.map.matrix.rows();
        let mut output = try_filled(rows, V::ZERO, scioxide_core::quantity!("output"))?;
        let mut scratch = try_filled(rows, V::ZERO, scioxide_core::quantity!("scratch"))?;
        self.apply_into(
            input,
            &mut output,
            &mut scratch,
            ExecutionContext::sequential_native(),
        )?;
        Field::new(destination, output)
    }
}

/// Exterior boundary labels on shared adaptive-face storage.
/// Every exterior whole face has one label; interior faces have none.
pub struct AmrBoundaryTags<const D: usize> {
    field: Field<Option<u32>, UnstructuredFace, D>,
}
impl<const D: usize> AmrBoundaryTags<D> {
    /// Validate and own an exterior face-label list.
    /// # Errors
    /// Rejects duplicates, missing/foreign/interior faces and allocation failure.
    pub fn new<R: RealScalar>(mesh: &AdaptiveMesh<R, D>, entries: &[(usize, u32)]) -> Result<Self> {
        let mut values = try_filled(mesh.face_count(), None, scioxide_core::quantity!("values"))?;
        for &(face, tag) in entries {
            let value = values.get_mut(face).ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("entries"),
                    "boundary face out of range",
                )
            })?;
            if value.replace(tag).is_some() {
                return Err(invalid(
                    scioxide_core::argument!("entries"),
                    "duplicate exterior tag",
                ));
            }
        }
        Self::from_values(mesh, values, scioxide_core::argument!("entries"))
    }
    fn from_values<R: RealScalar>(
        mesh: &AdaptiveMesh<R, D>,
        values: Vec<Option<u32>>,
        argument: scioxide_core::Argument,
    ) -> Result<Self> {
        let mut exterior = try_filled(
            mesh.face_count(),
            false,
            scioxide_core::quantity!("exterior"),
        )?;
        for i in mesh.interfaces().iter().filter(|i| i.right.is_none()) {
            exterior[i.left_face] = true;
        }
        if values.len() != exterior.len()
            || values
                .iter()
                .zip(exterior)
                .any(|(tag, outside)| tag.is_some() != outside)
        {
            return Err(invalid(
                argument,
                "tags must cover exactly the exterior faces",
            ));
        }
        Ok(Self {
            field: Field::new(mesh, values)?,
        })
    }
    /// Borrow canonical shared tag storage.
    pub fn view(&self) -> FieldView<'_, Option<u32>, UnstructuredFace, D> {
        self.field.view()
    }
    /// Transfer labels by topology ancestry; coarsening cannot merge different labels.
    /// # Errors
    /// Rejects stale epochs, conflicting labels, uncovered exterior faces and budgets.
    pub fn transfer<R: RealScalar>(
        &self,
        source: &AdaptiveMesh<R, D>,
        destination: &AdaptiveMesh<R, D>,
        map: &TopologyMap,
    ) -> Result<Self> {
        self.transfer_with_options(source, destination, map, AmrLimits::default())
    }
    /// Transfer tags under explicit ceilings.
    /// # Errors
    /// Rejects foreign topology, incomplete mappings and resource limits.
    pub fn transfer_with_options<R: RealScalar>(
        &self,
        source: &AdaptiveMesh<R, D>,
        destination: &AdaptiveMesh<R, D>,
        map: &TopologyMap,
        limits: AmrLimits,
    ) -> Result<Self> {
        map.validate(source.mesh_token(), destination.mesh_token())?;
        source
            .mesh_token()
            .ensure_compatible(self.field.mesh_token())?;
        let mut work = Work::new(limits, None);
        work.dofs(destination.face_count())?;
        let mut values = work.filled(
            destination.face_count(),
            None,
            scioxide_core::quantity!("values"),
        )?;
        work.charge::<bool>(destination.face_count())?;
        work.tick(add(
            map.faces().len(),
            destination.face_count(),
            scioxide_core::quantity!("work"),
        )?)?;
        for r in map.faces() {
            if let Some((i, j)) = relation(r)
                && let Some(tag) = self.field.values()[i]
            {
                if values[j].is_some_and(|old| old != tag) {
                    return Err(invalid(
                        scioxide_core::argument!("map"),
                        "coarsening merges incompatible boundary tags",
                    ));
                }
                values[j] = Some(tag);
            }
        }
        Self::from_values(destination, values, scioxide_core::argument!("map"))
    }
}

/// Verified algebraic residual of an L2 projection. It is not a continuum error bound.
#[derive(Clone, Copy, Debug)]
pub struct AmrProjectionReport<R: RealScalar> {
    residual_norm: R,
    rhs_norm: R,
}
impl<R: RealScalar> AmrProjectionReport<R> {
    /// Norm of `M_destination * projected - cross_mass * source` in independent coordinates.
    pub const fn residual_norm(&self) -> R {
        self.residual_norm
    }
    /// Norm of the projected right-hand side.
    pub const fn rhs_norm(&self) -> R {
        self.rhs_norm
    }
}

/// Plan-fixed staging of every [`AmrNodalTransfer`] operation; it never grows.
/// Linear-solver storage of projections belongs to the caller's solve callback.
#[derive(Debug, Clone)]
pub struct AmrTransferWorkspace<V> {
    work: Vec<V>,
}

/// A linear solve of the reduced destination mass, `M x = b`, starting from zero.
/// It receives the epoch's context and returns ordinary solve failures as errors.
pub type AmrMassSolve<'f, R, V> =
    dyn FnMut(&AmrMap<R>, &[V], &mut [V], &mut SolveContext<'_, '_>) -> Result<()> + 'f;

/// Consistent nodal transfer over one exact adaptation, including hanging constraints.
/// Projection owns mass/cross-mass mathematics; the caller supplies its linear solver.
/// Its maps schedule rows with the destination space's execution policy.
#[derive(Clone)]
pub struct AmrNodalTransfer<'a, R: RealScalar, const D: usize> {
    source: &'a AmrSpace<R, D>,
    destination: &'a AmrSpace<R, D>,
    mass: AmrMap<R>,
    cross: AmrMap<R>,
    interpolation: Option<AmrMap<R>>,
    storage: usize,
}
impl<'a, R: RealScalar, const D: usize> AmrNodalTransfer<'a, R, D> {
    /// Prepare with default numerical settings and no resource ceiling.
    ///
    /// # Errors
    /// Returns the geometry, coefficient, allocation and configuration errors of [`Self::with_options`].
    pub fn new(
        source: &'a AmrSpace<R, D>,
        destination: &'a AmrSpace<R, D>,
        map: &TopologyMap,
    ) -> Result<Self> {
        Self::with_options(source, destination, map, AmrLimits::default())
    }

    /// Prepare reduced consistent mass and cross-mass, plus interpolation when
    /// the destination only refines the source. Polynomial order must match.
    /// Interpolation uses node incidence and a cell-ancestry index. For fixed
    /// order and bounded constraint support, preparation costs `O(n log n)`
    /// work and `O(n)` storage in the source and destination cell count `n`.
    /// # Errors
    /// Rejects stale maps, mismatched spaces, unusable geometry and resource failures.
    pub fn with_options(
        source: &'a AmrSpace<R, D>,
        destination: &'a AmrSpace<R, D>,
        map: &TopologyMap,
        limits: AmrLimits,
    ) -> Result<Self> {
        Self::build(
            source,
            destination,
            map,
            limits,
            ExecutionContext::sequential_native(),
        )
    }
    /// Prepare as one epoch of `scope`, with inherited node budgets and cancellation.
    /// # Errors
    /// Same as `new`, including inherited resource failures.
    pub fn build<'scope, 'capability>(
        source: &'a AmrSpace<R, D>,
        destination: &'a AmrSpace<R, D>,
        map: &TopologyMap,
        limits: AmrLimits,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<Self> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "amr-nodal-transfer", 2),
            |context| {
                Self::prepare(
                    source,
                    destination,
                    map,
                    &mut Work::new(limits, Some(context)),
                )
            },
        )
    }
    fn prepare(
        source: &'a AmrSpace<R, D>,
        destination: &'a AmrSpace<R, D>,
        map: &TopologyMap,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Self> {
        map.validate(source.mesh.mesh_token(), destination.mesh.mesh_token())?;
        if source.order != destination.order {
            return Err(invalid(
                scioxide_core::argument!("destination"),
                "AMR transfer requires matching polynomial order",
            ));
        }
        work.dofs(source.full_dofs())?;
        work.dofs(destination.full_dofs())?;
        let mass_entries = mass_entries(destination, work)?;
        let cross_entries = cross_entries(source, destination, map, work)?;
        let policy = destination.options.execution;
        let mass = AmrMap::new(
            csr(
                destination.independent_dofs(),
                destination.independent_dofs(),
                &mass_entries,
                work,
            )?,
            policy,
        );
        let cross = AmrMap::new(
            csr(
                destination.independent_dofs(),
                source.full_dofs(),
                &cross_entries,
                work,
            )?,
            policy,
        );
        let interpolation = if map
            .cells()
            .iter()
            .any(|r| matches!(r, EntityRelation::Coarsened { .. }))
        {
            None
        } else {
            let entries = interpolation_entries(source, destination, map, work)?;
            Some(AmrMap::new(
                csr(
                    destination.independent_dofs(),
                    source.full_dofs(),
                    &entries,
                    work,
                )?,
                policy,
            ))
        };
        Ok(Self {
            source,
            destination,
            mass,
            cross,
            interpolation,
            storage: work.bytes,
        })
    }
    /// Reduced destination consistent mass matrix/operator; never row-sum lumped.
    pub fn mass(&self) -> &AmrMap<R> {
        &self.mass
    }
    /// Cross-mass from full source nodes to independent destination loads.
    pub fn cross_mass(&self) -> &AmrMap<R> {
        &self.cross
    }
    /// Refinement interpolation to independent destination nodes, when nested.
    pub fn interpolation(&self) -> Option<&AmrMap<R>> {
        self.interpolation.as_ref()
    }
    /// Conservative cumulative preparation payload.
    pub fn preparation_storage_bound(&self) -> usize {
        self.storage
    }
    fn forward_len(&self) -> Result<usize> {
        let q = scioxide_core::quantity!("transfer_staging");
        add(
            size(self.destination.independent_dofs(), 4, q)?,
            size(self.destination.full_dofs(), 2, q)?,
            q,
        )
    }
    fn pullback_len(&self) -> Result<usize> {
        let q = scioxide_core::quantity!("transfer_staging");
        let n = self.destination.independent_dofs();
        let full = self.source.full_dofs();
        add(
            add(size(3, n, q)?, full, q)?,
            n.max(full).max(self.destination.full_dofs()),
            q,
        )
    }
    /// Staging for interpolation, projection and both pullbacks.
    /// # Errors
    /// Returns checked size overflow and allocation failures.
    pub fn workspace<V: NumericScalar<Real = R>>(&self) -> Result<AmrTransferWorkspace<V>> {
        Ok(AmrTransferWorkspace {
            work: try_filled(
                self.forward_len()?.max(self.pullback_len()?),
                V::ZERO,
                scioxide_core::quantity!("transfer_staging"),
            )?,
        })
    }
    /// Interpolate a constrained source vector to the full destination space, atomically.
    /// # Errors
    /// Rejects coarsening, invalid input constraints/shapes, or arithmetic failure.
    pub fn interpolate_into<V: NumericScalar<Real = R>>(
        &self,
        input: &[V],
        output: &mut [V],
        workspace: &mut AmrTransferWorkspace<V>,
        execution: ExecutionContext<'_>,
    ) -> Result<()> {
        self.source.validate_field(input)?;
        let work = &mut workspace.work[..];
        if output.len() != self.destination.full_dofs() || work.len() < self.forward_len()? {
            return Err(invalid(
                if output.len() != self.destination.full_dofs() {
                    scioxide_core::argument!("output")
                } else {
                    scioxide_core::argument!("work")
                },
                "nodal transfer workspace shape",
            ));
        }
        let map = self.interpolation.as_ref().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("self"),
                "coarsening requires L2 projection",
            )
        })?;
        let (reduced, rest) = work.split_at_mut(self.destination.independent_dofs());
        let (scratch, _) = rest.split_at_mut(
            self.destination
                .independent_dofs()
                .max(self.destination.full_dofs()),
        );
        map.apply(input, reduced, &mut scratch[..map.matrix.rows()], execution)?;
        self.destination.constraints.apply(
            reduced,
            output,
            &mut scratch[..self.destination.full_dofs()],
            execution,
        )
    }
    /// Project through a caller linear solve, as one epoch of `scope`, and verify its
    /// residual before publishing. `solve` receives the mass operator, RHS and zero
    /// initial solution; it owns solver storage and returns ordinary solve failures as
    /// errors. No essential boundary values are imposed during conservative projection.
    /// # Errors
    /// Rejects unconstrained input, callback failure, nonfinite solution, insufficient
    /// workspace, cancellation, or an original-equation residual above `tolerance`.
    pub fn project_into<'scope, 'capability, V: NumericScalar<Real = R>>(
        &self,
        input: &[V],
        tolerance: ResidualTolerance<R>,
        solve: &mut AmrMassSolve<'_, R, V>,
        output: &mut [V],
        workspace: &mut AmrTransferWorkspace<V>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<AmrProjectionReport<R>> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "amr-projection", 2),
            |context| {
                self.project(
                    input,
                    tolerance,
                    solve,
                    output,
                    &mut workspace.work,
                    context,
                )
            },
        )
    }
    fn project<V: NumericScalar<Real = R>>(
        &self,
        input: &[V],
        tolerance: ResidualTolerance<R>,
        solve: &mut AmrMassSolve<'_, R, V>,
        output: &mut [V],
        work: &mut [V],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<AmrProjectionReport<R>> {
        self.source.validate_field(input)?;
        context.execution().check()?;
        if output.len() != self.destination.full_dofs() || work.len() < self.forward_len()? {
            return Err(invalid(
                if output.len() != self.destination.full_dofs() {
                    scioxide_core::argument!("output")
                } else {
                    scioxide_core::argument!("work")
                },
                "projection workspace shape",
            ));
        }
        let n = self.destination.independent_dofs();
        let (rhs, rest) = work.split_at_mut(n);
        let (solution, rest) = rest.split_at_mut(n);
        let (residual, scratch) = rest.split_at_mut(n);
        self.cross
            .apply_into_scoped(
                input,
                rhs,
                &mut scratch[..n],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        solution.fill(V::ZERO);
        solve(&self.mass, rhs, solution, context)?;
        validate(solution, n, scioxide_core::argument!("solve"))?;
        context.execution().check()?;
        self.mass
            .apply_into_scoped(
                solution,
                residual,
                &mut scratch[..n],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        let residual_norm = crate::adjoint::difference(residual, rhs)?;
        let rhs_norm = crate::adjoint::norm(rhs)?;
        if residual_norm > tolerance.threshold(rhs_norm)? {
            return Err(invalid(
                scioxide_core::argument!("solve"),
                "L2 projection solver residual exceeds tolerance",
            ));
        }
        let (staging, scratch) = scratch.split_at_mut(self.destination.full_dofs());
        self.destination
            .constraints
            .apply_into_scoped(
                solution,
                staging,
                &mut scratch[..self.destination.full_dofs()],
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        context.execution().check()?;
        output.copy_from_slice(staging);
        Ok(AmrProjectionReport {
            residual_norm,
            rhs_norm,
        })
    }
}

impl<R: RealScalar> AmrCellTransfer<R> {
    /// Reverse the fixed-geometry conservative map into source-cell cotangents.
    /// Parameter-dependent weights are composed separately through supplied products.
    /// # Errors
    /// Rejects foreign epochs, shapes, inadequate scratch and nonfinite products.
    pub fn pullback_into<V: NumericScalar<Real = R>, const D: usize>(
        &self,
        source: &MeshToken,
        seed: FieldView<'_, V, Cell, D>,
        output: &mut [V],
        scratch: &mut [V],
        execution: ExecutionContext<'_>,
    ) -> Result<()> {
        self.source.ensure_compatible(source)?;
        self.destination.ensure_compatible(seed.mesh_token())?;
        self.map.adjoint(seed.values(), output, scratch, execution)
    }
}
impl<R: RealScalar, const D: usize> AmrNodalTransfer<'_, R, D> {
    fn check_pullback<V: NumericScalar<Real = R>>(
        &self,
        source: &MeshToken,
        destination: &MeshToken,
        seed: &[V],
        output: &[V],
        work: &[V],
    ) -> Result<()> {
        self.source.mesh().mesh_token().ensure_compatible(source)?;
        self.destination
            .mesh()
            .mesh_token()
            .ensure_compatible(destination)?;
        validate(
            seed,
            self.destination.full_dofs(),
            scioxide_core::argument!("seed"),
        )?;
        if output.len() != self.source.full_dofs() || work.len() < self.pullback_len()? {
            return Err(invalid(
                if output.len() != self.source.full_dofs() {
                    scioxide_core::argument!("output")
                } else {
                    scioxide_core::argument!("work")
                },
                "nodal pullback output/workspace shape",
            ));
        }
        Ok(())
    }
    /// Reverse constrained interpolation with geometry held fixed.
    /// Returns a cotangent in full source-node coordinates; callers using independent
    /// coordinates subsequently apply the source constraint adjoint.
    /// # Errors
    /// Rejects coarsening, foreign epochs and invalid products without publishing output.
    pub fn interpolate_pullback_into<'scope, 'capability, V: NumericScalar<Real = R>>(
        &self,
        source: &MeshToken,
        destination: &MeshToken,
        seed: &[V],
        output: &mut [V],
        workspace: &mut AmrTransferWorkspace<V>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<()> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "amr-interpolation-pullback", 2),
            |context| {
                self.interpolate_pullback(
                    source,
                    destination,
                    seed,
                    output,
                    &mut workspace.work,
                    context,
                )
            },
        )
    }
    fn interpolate_pullback<V: NumericScalar<Real = R>>(
        &self,
        source: &MeshToken,
        destination: &MeshToken,
        seed: &[V],
        output: &mut [V],
        work: &mut [V],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<()> {
        self.check_pullback(source, destination, seed, output, work)?;
        let map = self.interpolation.as_ref().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("self"),
                "coarsening requires projection pullback",
            )
        })?;
        let (rhs, rest) = work.split_at_mut(self.destination.independent_dofs());
        let (staged, scratch) = rest.split_at_mut(self.source.full_dofs());
        crate::adjoint::admit(context, EvaluationRole::LinearOperator)?;
        self.destination
            .constraints
            .apply_adjoint_into_scoped(
                seed,
                rhs,
                scratch,
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        crate::adjoint::admit(context, EvaluationRole::LinearOperator)?;
        map.apply_adjoint_into_scoped(
            rhs,
            staged,
            scratch,
            &mut scioxide_core::OperatorContext::new(context),
        )
        .and_then(scioxide_core::Invocation::require)?;
        context.execution().check()?;
        output.copy_from_slice(staged);
        Ok(())
    }
    /// Reverse the consistent projection equation M z = C u with fixed geometry.
    /// Solve M* mu = P_destination* seed and return C* mu. The real symmetric mass
    /// operator equals its adjoint; no forward factor is implicitly reused.
    /// Parameter derivatives contribute mu*(dC u - dM z) plus reconstruction terms
    /// and must be supplied explicitly for differentiated geometry.
    /// # Errors
    /// Epoch, callback, residual and resource failures preserve output.
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps operands, storage and execution controls explicit"
    )]
    pub fn project_pullback_into<'scope, 'capability, V: NumericScalar<Real = R>>(
        &self,
        source: &MeshToken,
        destination: &MeshToken,
        seed: &[V],
        tolerance: ResidualTolerance<R>,
        solve: &mut AmrMassSolve<'_, R, V>,
        output: &mut [V],
        workspace: &mut AmrTransferWorkspace<V>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<AmrProjectionReport<R>> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "amr-projection-pullback", 2),
            |context| {
                self.project_pullback(
                    source,
                    destination,
                    seed,
                    tolerance,
                    solve,
                    output,
                    &mut workspace.work,
                    context,
                )
            },
        )
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps operands, storage and execution controls explicit"
    )]
    fn project_pullback<V: NumericScalar<Real = R>>(
        &self,
        source: &MeshToken,
        destination: &MeshToken,
        seed: &[V],
        tolerance: ResidualTolerance<R>,
        solve: &mut AmrMassSolve<'_, R, V>,
        output: &mut [V],
        work: &mut [V],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<AmrProjectionReport<R>> {
        self.check_pullback(source, destination, seed, output, work)?;
        let n = self.destination.independent_dofs();
        let (rhs, rest) = work.split_at_mut(n);
        let (solution, rest) = rest.split_at_mut(n);
        let (residual, rest) = rest.split_at_mut(n);
        let (staged, scratch) = rest.split_at_mut(self.source.full_dofs());
        crate::adjoint::admit(context, EvaluationRole::LinearOperator)?;
        self.destination
            .constraints
            .apply_adjoint_into_scoped(
                seed,
                rhs,
                scratch,
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        solution.fill(V::ZERO);
        crate::adjoint::admit(context, EvaluationRole::LinearSolve)?;
        solve(&self.mass, rhs, solution, context)?;
        validate(solution, n, scioxide_core::argument!("solve"))?;
        crate::adjoint::admit(context, EvaluationRole::LinearOperator)?;
        self.mass
            .apply_adjoint_into_scoped(
                solution,
                residual,
                scratch,
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        let residual_norm = crate::adjoint::difference(residual, rhs)?;
        let rhs_norm = crate::adjoint::norm(rhs)?;
        if residual_norm > tolerance.threshold(rhs_norm)? {
            return Err(invalid(
                scioxide_core::argument!("solve"),
                "projection transpose residual exceeds tolerance",
            ));
        }
        crate::adjoint::admit(context, EvaluationRole::LinearOperator)?;
        self.cross
            .apply_adjoint_into_scoped(
                solution,
                staged,
                scratch,
                &mut scioxide_core::OperatorContext::new(context),
            )
            .and_then(scioxide_core::Invocation::require)?;
        context.execution().check()?;
        output.copy_from_slice(staged);
        Ok(AmrProjectionReport {
            residual_norm,
            rhs_norm,
        })
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> crate::TaggedBoundaries<R, V, D> {
    /// Bind conditions to the tagged exterior faces of an adaptive mesh epoch.
    ///
    /// Shares the validation of every other boundary family: unused tags,
    /// duplicate or missing face/component conditions and invalid components or
    /// Robin coefficients are rejected.
    /// # Errors
    /// Rejects foreign tags and the conditions above.
    pub fn adaptive(
        mesh: &AdaptiveMesh<R, D>,
        tags: &AmrBoundaryTags<D>,
        components: usize,
        conditions: Vec<crate::BoundaryCondition<R, V, D, u32>>,
    ) -> Result<Self> {
        mesh.mesh_token()
            .ensure_compatible(tags.field.mesh_token())?;
        let mut exterior = Vec::new();
        for (face, tag) in tags.field.values().iter().enumerate() {
            if let Some(tag) = *tag {
                exterior.try_reserve(1).map_err(|_| {
                    SciOxideError::allocation_failure(
                        scioxide_core::quantity!("exterior_faces"),
                        exterior.len() + 1,
                    )
                })?;
                exterior.push((face, tag));
            }
        }
        Self::tagged_faces(
            mesh.mesh_token().clone(),
            exterior,
            components,
            conditions,
            |&tag| tag,
        )
    }
}

type Triplets<R> = Vec<(usize, usize, R)>;

/// Reduced destination consistent mass `Cᵀ M C` triplets.
fn mass_entries<R: RealScalar, const D: usize>(
    destination: &AmrSpace<R, D>,
    work: &mut Work<'_, '_, '_>,
) -> Result<Triplets<R>> {
    let mut mass_entries = Vec::new();
    for (cell, data) in destination.cells.iter().enumerate() {
        for (i, &a) in destination.cell_dofs(cell)?.iter().enumerate() {
            for (j, &b) in destination.cell_dofs(cell)?.iter().enumerate() {
                for (r, x) in row(&destination.constraints.matrix, a) {
                    for (c, y) in row(&destination.constraints.matrix, b) {
                        work.entries(add(
                            mass_entries.len(),
                            1,
                            scioxide_core::quantity!("entries"),
                        )?)?;
                        work.tick(1)?;
                        work.push(
                            &mut mass_entries,
                            (
                                r,
                                c,
                                finite(
                                    x * data.mass[i * destination.local + j] * y,
                                    scioxide_core::NonFiniteOrigin::Intermediate,
                                )?,
                            ),
                            scioxide_core::quantity!("mass_entries"),
                        )?;
                    }
                }
            }
        }
    }
    Ok(mass_entries)
}

/// Cross-mass triplets from full source nodes to independent destination rows,
/// integrated on the finer cell of each ancestry relation.
fn cross_entries<R: RealScalar, const D: usize>(
    source: &AmrSpace<R, D>,
    destination: &AmrSpace<R, D>,
    map: &TopologyMap,
    work: &mut Work<'_, '_, '_>,
) -> Result<Triplets<R>> {
    let mut cross_entries = Vec::new();
    for relation in map.cells() {
        let (a, b) = self::relation(relation).ok_or_else(|| {
            invalid(
                scioxide_core::argument!("map"),
                "projection requires covering cell ancestry",
            )
        })?;
        let (integration, cell) = if matches!(relation, EntityRelation::Coarsened { .. }) {
            (source, a)
        } else {
            (destination, b)
        };
        let mut local = [[R::ZERO; 10]; 10];
        for q in &integration.volume {
            work.tick(size(
                source.local,
                destination.local,
                scioxide_core::quantity!("work"),
            )?)?;
            let point = integration.physical(cell, q.point)?;
            let s = source.reference_sample(
                source.reference(a, point, scioxide_core::argument!("source"))?,
                scioxide_core::argument!("source"),
            )?;
            let d = destination.reference_sample(
                destination.reference(b, point, scioxide_core::argument!("destination"))?,
                scioxide_core::argument!("destination"),
            )?;
            let w = finite(
                q.weight * integration.cells[cell].map.determinant,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
            for i in 0..destination.local {
                for j in 0..source.local {
                    local[i][j] = finite(
                        local[i][j] + w * d.values[i] * s.values[j],
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?;
                }
            }
        }
        for (i, &node) in destination.cell_dofs(b)?.iter().enumerate() {
            for (j, &column) in source.cell_dofs(a)?.iter().enumerate() {
                for (r, x) in row(&destination.constraints.matrix, node) {
                    work.entries(add(
                        cross_entries.len(),
                        1,
                        scioxide_core::quantity!("entries"),
                    )?)?;
                    work.tick(1)?;
                    work.push(
                        &mut cross_entries,
                        (
                            r,
                            column,
                            finite(
                                x * local[i][j],
                                scioxide_core::NonFiniteOrigin::Intermediate,
                            )?,
                        ),
                        scioxide_core::quantity!("cross_entries"),
                    )?;
                }
            }
        }
    }
    Ok(cross_entries)
}

/// Nodal interpolation triplets of every independent destination node from its
/// source ancestor cell.
fn interpolation_entries<R: RealScalar, const D: usize>(
    source: &AmrSpace<R, D>,
    destination: &AmrSpace<R, D>,
    map: &TopologyMap,
    work: &mut Work<'_, '_, '_>,
) -> Result<Triplets<R>> {
    let mut entries = Vec::new();
    let mut ancestors = work.filled(
        destination.cells.len(),
        None,
        scioxide_core::quantity!("ancestors"),
    )?;
    for r in map.cells() {
        work.tick(1)?;
        if let Some((a, b)) = relation(r) {
            ancestors[b].get_or_insert(a);
        }
    }
    for (i, &node) in destination.independent_nodes().iter().enumerate() {
        work.tick(1)?;
        let &(b, _) = destination.incidence[node].first().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("destination"),
                "independent node without incident cell",
            )
        })?;
        work.tick(1)?;
        let a = ancestors[b].ok_or_else(|| {
            invalid(
                scioxide_core::argument!("map"),
                "missing interpolation ancestor",
            )
        })?;
        let s = source.reference_sample(
            source.reference(
                a,
                destination.points[node],
                scioxide_core::argument!("source"),
            )?,
            scioxide_core::argument!("source"),
        )?;
        for (j, &column) in source.cell_dofs(a)?.iter().enumerate() {
            work.tick(1)?;
            work.entries(add(entries.len(), 1, scioxide_core::quantity!("entries"))?)?;
            work.push(
                &mut entries,
                (i, column, s.values[j]),
                scioxide_core::quantity!("entries"),
            )?;
        }
    }
    Ok(entries)
}
