use super::space::Bytes;
use super::*;
use crate::discretization::Elimination;
use scioxide_core::SolveScope;
use scioxide_core::backend::PartitionedSolve;

/// Per-worker coefficient scratch.
struct Worker<V: NumericScalar> {
    diffusion: Vec<V>,
    reaction: Vec<V>,
    factor: Vec<V>,
}
impl<V: NumericScalar> Worker<V> {
    fn evaluate<const D: usize>(
        &mut self,
        point: [V::Real; D],
        coefficients: &dyn SimplexFemCoefficients<V::Real, V, D>,
        components: usize,
        tangent: bool,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        self.diffusion.fill(V::from_real(V::Real::NAN));
        self.reaction.fill(V::from_real(V::Real::NAN));
        admit(context, true)?;
        coefficients.evaluate(point, &mut self.diffusion, &mut self.reaction)?;
        if self
            .diffusion
            .iter()
            .chain(&self.reaction)
            .any(|v| !v.has_finite_components())
        {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Callback,
                "incomplete or nonfinite FEM coefficient output",
            ));
        }
        if tangent {
            crate::validation::hermitian_part(&mut self.diffusion, components * D)
        } else {
            crate::validation::positive_tensor(
                &mut self.diffusion,
                &mut self.factor,
                components * D,
            )
        }
    }
    fn new(components: usize, dimension: usize) -> Result<Self, SciOxideError> {
        let n = size(components, dimension, scioxide_core::quantity!("diffusion"))?;
        let count = size(n, n, scioxide_core::quantity!("diffusion"))?;
        Ok(Self {
            diffusion: try_filled(count, V::ZERO, scioxide_core::quantity!("diffusion"))?,
            reaction: try_filled(
                size(components, components, scioxide_core::quantity!("reaction"))?,
                V::ZERO,
                scioxide_core::quantity!("reaction"),
            )?,
            factor: try_filled(count, V::ZERO, scioxide_core::quantity!("diffusion_factor"))?,
        })
    }
    /// Live bytes of one worker.
    fn bound(components: usize, dimension: usize) -> Result<Bytes, SciOxideError> {
        let n = size(
            components,
            dimension,
            scioxide_core::quantity!("worker_bytes"),
        )?;
        Bytes::new(size_of::<Self>())
            .array::<V>(size(
                size(n, n, scioxide_core::quantity!("worker_bytes"))?,
                2,
                scioxide_core::quantity!("worker_bytes"),
            )?)?
            .array::<V>(add(
                size(
                    components,
                    components,
                    scioxide_core::quantity!("worker_bytes"),
                )?,
                components,
                scioxide_core::quantity!("worker_bytes"),
            )?)
    }
}

/// Cell forms of one coefficient set and boundary binding on a prepared space.
struct Forms<'a, R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    space: &'a SimplexSpace<R, D>,
    components: usize,
    boundaries: &'a TaggedBoundaries<R, V, D>,
    coefficients: &'a dyn SimplexFemCoefficients<R, V, D>,
    constant: Option<&'a Worker<V>>,
    affine: Option<&'a super::affine::AffineForms<R, D>>,
    /// Signed parameter tangents replace positive definiteness by the Hermitian part.
    tangent: bool,
}

/// Identically zero coefficients: the tangent of a coefficient-independent parameter.
struct ZeroCoefficients;

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> SimplexFemCoefficients<R, V, D>
    for ZeroCoefficients
{
    fn evaluate(
        &self,
        _: [R; D],
        diffusion: &mut [V],
        reaction: &mut [V],
    ) -> Result<(), SciOxideError> {
        diffusion.fill(V::ZERO);
        reaction.fill(V::ZERO);
        Ok(())
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> SimplexFemPlan<R, V, D> {
    /// Prepare with default numerical settings and no resource ceiling.
    ///
    /// # Errors
    /// Returns the geometry, coefficient, allocation and configuration errors of [`Self::with_options`].
    pub fn new(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        order: SimplexFemOrder,
        coefficients: &dyn SimplexFemCoefficients<R, V, D>,
        boundaries: &TaggedBoundaries<R, V, D>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(
            mesh,
            geometry,
            order,
            coefficients,
            boundaries,
            SimplexFemOptions::default(),
        )
    }

    /// Prepare fixed coefficients, geometry and boundaries of continuous `order`
    /// elements under sequential native execution.
    /// # Errors
    /// Rejects shape, identity, coefficient, geometry, quadrature and resource
    /// violations. The complete plan is returned only after all cells succeed.
    pub fn with_options(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        order: SimplexFemOrder,
        coefficients: &dyn SimplexFemCoefficients<R, V, D>,
        boundaries: &TaggedBoundaries<R, V, D>,
        options: SimplexFemOptions<R>,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            mesh,
            geometry,
            order,
            coefficients,
            boundaries,
            options,
            ExecutionContext::sequential_native(),
        )
    }

    /// Prepare as one epoch of `scope`. Independent cells receive stable equal
    /// budget shares through core partitioned solves; failing callbacks remain
    /// charged. At most 32 workers are used.
    /// # Errors
    /// Includes all construction failures, insufficient work, cancellation,
    /// deadline, and executor errors. No partial plan is published.
    pub fn build<'scope, 'capability>(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        order: SimplexFemOrder,
        coefficients: &dyn SimplexFemCoefficients<R, V, D>,
        boundaries: &TaggedBoundaries<R, V, D>,
        options: SimplexFemOptions<R>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "simplex-fem", 2),
            |context| {
                Self::prepare(
                    mesh,
                    geometry,
                    order,
                    coefficients,
                    boundaries,
                    options,
                    context,
                )
            },
        )
    }

    fn prepare(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        order: SimplexFemOrder,
        coefficients: &dyn SimplexFemCoefficients<R, V, D>,
        boundaries: &TaggedBoundaries<R, V, D>,
        options: SimplexFemOptions<R>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Self, SciOxideError> {
        context.execution().check()?;
        mesh.mesh_token().ensure_compatible(&boundaries.token)?;
        let components = boundaries.components;
        let layout = Layout {
            order,
            components,
            pressure: false,
        };
        layout.check_limits(&mesh, options.limits, 0)?;
        let storage_bound = Self::storage_bound(&mesh, layout, &options)?;
        if storage_bound > options.limits.max_storage_bytes {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "FEM preparation storage ceiling exceeded",
            ));
        }
        let minimum = if matches!(geometry, SimplexFemGeometry::Quadratic(_)) {
            7
        } else {
            2 * order.degree()
        };
        let space =
            SimplexSpace::prepare(mesh, geometry, order, options, (minimum, minimum), context)?;
        let (matrices, masses, reactive) =
            prepare_elements(&space, layout, coefficients, boundaries, context)?;
        let n = layout.local::<D>();
        let loads = space.loads(boundaries, true)?;
        let mut constrained = try_filled(
            layout.unknowns(&space.mesh)?,
            false,
            scioxide_core::quantity!("constrained"),
        )?;
        for &dof in loads.dirichlet_dofs() {
            constrained[dof] = true;
        }
        let (lift, constrained) = if constrained.iter().any(|&fixed| fixed) {
            let (_, lift, constrained) = space.assemble(
                layout,
                constrained,
                Elimination::Lift,
                |cell, i, j| Ok(matrices[(cell * n + i) * n + j]),
                core::iter::empty(),
            )?;
            (Some(lift).filter(|lift| lift.nnz() > 0), constrained)
        } else {
            (None, constrained)
        };
        let loads = loads.finish(lift);
        // Evaluate the boundary data once at t = 0, rejecting conflicting
        // Dirichlet candidates and invalid data before a plan is published.
        let mut probe = try_filled(constrained.len(), V::ZERO, scioxide_core::quantity!("load"))?;
        loads.load_into(boundaries, &crate::FieldData::zero(), R::ZERO, &mut probe)?;
        drop(probe);
        let constant_modes = !constrained.iter().any(|&fixed| fixed)
            && boundaries.is_natural_everywhere()
            && !reactive;
        let lumped = lumped_mass(&space, &masses, components)?;
        let assembly = AssemblyPattern::new(
            &space,
            layout,
            &constrained,
            Elimination::Symmetric,
            core::iter::empty(),
        )?;
        let mass_assembly = if components > 1 || constrained.iter().any(|&fixed| fixed) {
            let free = try_filled(
                constrained.len(),
                false,
                scioxide_core::quantity!("mass_pattern"),
            )?;
            let l = layout.order.nodes(D);
            Some(alloc::boxed::Box::new(AssemblyPattern::with_coupling(
                &space,
                layout,
                &free,
                Elimination::Symmetric,
                core::iter::empty(),
                |i, j| i / l == j / l,
            )?))
        } else {
            None
        };
        context.execution().check()?;
        Ok(Self {
            entry_bound: add(
                layout.entries(&space.mesh)?,
                constrained.len(),
                scioxide_core::quantity!("entry_bound"),
            )?,
            space,
            components,
            matrices,
            masses,
            assembly,
            mass_assembly,
            boundaries: boundaries.clone(),
            loads,
            constrained,
            lumped,
            constant_modes,
            storage_bound,
        })
    }

    /// Conservative simultaneously-live payload: the space, element forms with
    /// their child ledgers, per-worker coefficient scratch and global vectors.
    fn storage_bound(
        mesh: &SimplexMesh<R, D>,
        layout: Layout,
        options: &SimplexFemOptions<R>,
    ) -> Result<usize, SciOxideError> {
        let (n, l) = (layout.local::<D>(), layout.order.nodes(D));
        let element = Bytes::new(size_of::<Element<'_, R, V>>() + size_of::<PartitionedSolve>())
            .array::<V>(size(n, n, scioxide_core::quantity!("elem_bytes"))?)?
            .array::<R>(size(l, l, scioxide_core::quantity!("elem_bytes"))?)?
            .array::<(V, Option<V>)>(n)?
            .array::<usize>(size(l, 4, scioxide_core::quantity!("elem_bytes"))?)?
            .total();
        let workers = size(
            add(
                crate::validation::worker_bound(options.execution),
                1,
                scioxide_core::quantity!("workers"),
            )?,
            Worker::<V>::bound(layout.components, D)?.total(),
            scioxide_core::quantity!("storage_bound"),
        )?;
        let unknowns = layout.unknowns(mesh)?;
        Ok(
            SimplexSpace::storage_bound(mesh, layout.order, options.limits)?
                .bytes(AssemblyPattern::bound(layout.entries(mesh)?, unknowns)?.total())?
                .bytes(AssemblyPattern::bound(layout.entries(mesh)?, unknowns)?.total())?
                .bytes(workers)?
                .bytes(size(
                    mesh.cell_count(),
                    element,
                    scioxide_core::quantity!("storage_bound"),
                )?)?
                .array::<(V, V, Option<V>, Option<V>)>(unknowns)?
                .array::<[usize; 3]>(add(
                    layout.nodes(mesh)?,
                    1,
                    scioxide_core::quantity!("storage_bound"),
                )?)?
                .total(),
        )
    }

    /// Exact tangent `∂A/∂p` of the free residual rows for a signed derivative
    /// coefficient field and Robin coefficient derivatives (one per condition),
    /// re-assembled on the prepared geometry and quadrature. Element forms are
    /// linear in the sampled coefficients, so the tangent is exact.
    pub(crate) fn tangent_operator(
        &self,
        coefficients: Option<&dyn SimplexFemCoefficients<R, V, D>>,
        robin: &[V],
    ) -> Result<CsrMatrix<V>, SciOxideError> {
        let boundaries = self.boundaries.with_robin_coefficients(robin)?;
        let forms = Forms {
            space: &self.space,
            components: self.components,
            boundaries: &boundaries,
            coefficients: coefficients.unwrap_or(&ZeroCoefficients),
            constant: None,
            affine: None,
            tangent: true,
        };
        let n = self.layout().local::<D>();
        let mut worker = Worker::new(self.components, D)?;
        let l = self.space.order.nodes(D);
        let mut matrix = try_filled(n * n, V::ZERO, scioxide_core::quantity!("element"))?;
        let mut mass = try_filled(l * l, R::ZERO, scioxide_core::quantity!("element"))?;
        let mut element = Element {
            matrix: &mut matrix,
            mass: &mut mass,
            reactive: false,
        };
        let mut current = None;
        crate::attempt::sequential(
            scioxide_core::algorithm_id!("pde", "simplex-fem-tangent", 1),
            |context| {
                let (matrix, _, _) = self.space.assemble(
                    self.layout(),
                    self.constrained.clone(),
                    Elimination::Tangent,
                    |cell, i, j| {
                        if current != Some(cell) {
                            element.matrix.fill(V::ZERO);
                            element.mass.fill(R::ZERO);
                            forms.cell(cell, &mut element, &mut worker, &mut *context)?;
                            current = Some(cell);
                        }
                        Ok(element.matrix[i * n + j])
                    },
                    core::iter::empty(),
                )?;
                Ok(matrix)
            },
        )
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> Forms<'_, R, V, D> {
    /// Prepared operator forms of one cell; boundary data is not evaluated.
    fn cell(
        &self,
        cell: usize,
        element: &mut Element<'_, R, V>,
        worker: &mut Worker<V>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        admit(context, false)?;
        self.space.check_geometry(cell)?;
        self.volume(cell, element, worker, context)?;
        self.robin(cell, element)
    }

    /// Diffusion, reaction and consistent mass at every volume node.
    fn volume(
        &self,
        cell: usize,
        element: &mut Element<'_, R, V>,
        worker: &mut Worker<V>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let c = self.components;
        if let (Some(constant), Some(affine), Some(map)) =
            (self.constant, self.affine, self.space.affine_map(cell))
        {
            element.reactive = constant.reaction.iter().any(|&v| v != V::ZERO);
            if affine.cell(
                map,
                c,
                &constant.diffusion,
                &constant.reaction,
                &mut worker.factor,
                element.matrix,
                element.mass,
            )? {
                return Ok(());
            }
        }
        for reference in self.space.volume_nodes() {
            let sample = self.space.sample(cell, reference)?;
            let coefficients = if let Some(constant) = self.constant {
                constant
            } else {
                worker.evaluate(sample.point, self.coefficients, c, self.tangent, context)?;
                &*worker
            };
            element.reactive |= coefficients.reaction.iter().any(|&value| value != V::ZERO);
            reference::accumulate_forms(
                self.space.order.nodes(D),
                c,
                sample.weight,
                &self.space.reference(&sample).solution.values,
                &sample.gradients,
                &coefficients.diffusion,
                &coefficients.reaction,
                element.matrix,
                element.mass,
            )?;
        }
        Ok(())
    }

    /// Robin mass `α∫φᵢφⱼ` on every exterior facet with a Robin component.
    fn robin(&self, cell: usize, element: &mut Element<'_, R, V>) -> Result<(), SciOxideError> {
        let local = self.space.order.nodes(D);
        let n = local * self.components;
        for (face, opposite) in self.space.exterior_faces(cell)? {
            for component in 0..self.components {
                let BoundaryKind::Robin { alpha } = self.boundaries.kind_at(face, component)?
                else {
                    continue;
                };
                for reference in self.space.facet_nodes(opposite) {
                    let sample = self.space.sample(cell, reference)?;
                    let values = &self.space.reference(&sample).solution.values;
                    for i in 0..local {
                        let row = component * local + i;
                        for j in 0..local {
                            let slot = row * n + component * local + j;
                            element.matrix[slot] = crate::discretization::entry(
                                element.matrix[slot]
                                    + alpha * V::from_real(sample.weight * values[i] * values[j]),
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Diagonal-scaled (HRZ) lumped mass: positive for P1 and P2 and equal to row
/// sums for P1, replicated for each component.
fn lumped_mass<R: RealScalar, const D: usize>(
    space: &SimplexSpace<R, D>,
    masses: &[R],
    components: usize,
) -> Result<Vec<R>, SciOxideError> {
    let local = space.order.nodes(D);
    let mut lumped = try_filled(
        space.nodes,
        R::ZERO,
        scioxide_core::quantity!("lumped_mass"),
    )?;
    for (cell, mass) in masses.chunks_exact(local * local).enumerate() {
        let total = mass.iter().fold(R::ZERO, |s, &m| s + m);
        let diagonal = (0..local).fold(R::ZERO, |s, i| s + mass[i * local + i]);
        for (i, &dof) in space.dofs(cell).iter().enumerate() {
            lumped[dof] = finite(
                lumped[dof] + mass[i * local + i] * (total / diagonal),
                NonFiniteOrigin::Intermediate,
            )?;
        }
    }
    if lumped.iter().any(|&m| !m.is_finite() || m <= R::ZERO) {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "lumped FEM mass is not positive",
        ));
    }
    let mut replicated = try_with_capacity(
        size(
            space.nodes,
            components,
            scioxide_core::quantity!("lumped_mass"),
        )?,
        scioxide_core::quantity!("lumped_mass"),
    )?;
    for _ in 0..components {
        replicated.extend_from_slice(&lumped);
    }
    Ok(replicated)
}

/// Build contiguous element forms with one reference table and bounded worker scratch.
fn prepare_elements<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    space: &SimplexSpace<R, D>,
    layout: Layout,
    coefficients: &dyn SimplexFemCoefficients<R, V, D>,
    boundaries: &TaggedBoundaries<R, V, D>,
    context: &mut SolveContext<'_, '_>,
) -> Result<(Vec<V>, Vec<R>, bool), SciOxideError> {
    let components = layout.components;
    let (n, l) = (layout.local::<D>(), layout.order.nodes(D));
    let mut matrices = try_filled(
        layout.entries(&space.mesh)?,
        V::ZERO,
        scioxide_core::quantity!("matrices"),
    )?;
    let mut masses = try_filled(
        size(
            space.cell_count(),
            l * l,
            scioxide_core::quantity!("masses"),
        )?,
        R::ZERO,
        scioxide_core::quantity!("masses"),
    )?;
    let mut elements = try_with_capacity(space.cell_count(), scioxide_core::quantity!("elements"))?;
    for (matrix, mass) in matrices
        .chunks_exact_mut(n * n)
        .zip(masses.chunks_exact_mut(l * l))
    {
        elements.push(Element {
            matrix,
            mass,
            reactive: false,
        });
    }
    let constant = if coefficients.is_spatially_constant() && space.cell_count() != 0 {
        let mut worker = Worker::new(components, D)?;
        worker.evaluate(
            space.sample(0, 0)?.point,
            coefficients,
            components,
            false,
            context,
        )?;
        Some(worker)
    } else {
        None
    };
    let affine = if constant.is_some() && space.is_affine() {
        Some(super::affine::AffineForms::new(space)?)
    } else {
        None
    };
    let forms = Forms {
        space,
        components,
        boundaries,
        coefficients,
        constant: constant.as_ref(),
        affine: affine.as_ref(),
        tangent: false,
    };
    space.for_each_cell(
        &mut elements,
        || Worker::new(components, D),
        &|cell, element, worker, context| forms.cell(cell, element, worker, context),
        context,
    )?;
    let reactive = elements.iter().any(|element| element.reactive);
    drop(elements);
    Ok((matrices, masses, reactive))
}
