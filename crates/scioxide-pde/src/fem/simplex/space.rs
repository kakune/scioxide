//! Shared continuous simplex function space: DOF numbering, geometry, tabulated
//! reference values, Dirichlet data, element assembly and field packing.
use super::reference::{facet_point, map_coordinates, normal, physical_gradients, reference_node};
use super::*;
use crate::discretization::{Eliminated, Elimination, OperatorBuilder};
use scioxide_core::backend::{PartitionSlot, PartitionedSolve, with_partitioned_solves};
use scioxide_core::{SolveBudgets, quantity};
use scioxide_sparse::{GraphComponents, GraphDirection, GraphPlan};

/// Allowance for allocator headers, rule tables and small execution objects.
pub(crate) const PREPARATION_OVERHEAD_BYTES: usize = 1 << 16;

/// Checked byte total of typed arrays, for preparation storage bounds.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bytes(usize);

impl Bytes {
    /// Start from `bytes`.
    pub(crate) const fn new(bytes: usize) -> Self {
        Self(bytes)
    }
    /// Add `count` values of `T`.
    pub(crate) fn array<T>(self, count: usize) -> Result<Self, SciOxideError> {
        self.bytes(size(count, size_of::<T>(), quantity!("storage_bound"))?)
    }
    /// Add `bytes` bytes.
    pub(crate) fn bytes(self, bytes: usize) -> Result<Self, SciOxideError> {
        add(self.0, bytes, quantity!("storage_bound")).map(Self)
    }
    /// Total bytes.
    pub(crate) const fn total(self) -> usize {
        self.0
    }
}

/// Packed unknowns of `components` order-`order` blocks, optionally followed by
/// one P1 vertex block. Local unknowns follow the same component-major order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub(crate) order: SimplexFemOrder,
    pub(crate) components: usize,
    pub(crate) pressure: bool,
}

impl Layout {
    /// Scalar nodes per component on `mesh`.
    pub(crate) fn nodes<R: RealScalar, const D: usize>(
        self,
        mesh: &SimplexMesh<R, D>,
    ) -> Result<usize, SciOxideError> {
        add(
            mesh.vertex_count(),
            if self.order == SimplexFemOrder::P2 {
                mesh.edge_count()
            } else {
                0
            },
            quantity!("nodes"),
        )
    }
    /// Local unknowns per cell.
    pub(crate) fn local<const D: usize>(self) -> usize {
        self.components * self.order.nodes(D) + if self.pressure { D + 1 } else { 0 }
    }
    /// Packed unknowns on `mesh`.
    pub(crate) fn unknowns<R: RealScalar, const D: usize>(
        self,
        mesh: &SimplexMesh<R, D>,
    ) -> Result<usize, SciOxideError> {
        add(
            size(self.components, self.nodes(mesh)?, quantity!("unknowns"))?,
            if self.pressure {
                mesh.vertex_count()
            } else {
                0
            },
            quantity!("unknowns"),
        )
    }
    /// Dense element entries summed over the cells of `mesh`.
    pub(crate) fn entries<R: RealScalar, const D: usize>(
        self,
        mesh: &SimplexMesh<R, D>,
    ) -> Result<usize, SciOxideError> {
        let local = self.local::<D>();
        size(
            mesh.cell_count(),
            size(local, local, quantity!("entries"))?,
            quantity!("entries"),
        )
    }
    /// Reject layouts beyond the unknown and element-entry ceilings, counting
    /// `extra` additional unknowns.
    pub(crate) fn check_limits<R: RealScalar, const D: usize>(
        self,
        mesh: &SimplexMesh<R, D>,
        limits: SimplexFemLimits,
        extra: usize,
    ) -> Result<(), SciOxideError> {
        let unknowns = add(self.unknowns(mesh)?, extra, quantity!("unknowns"))?;
        if unknowns > limits.max_unknowns || self.entries(mesh)? > limits.max_element_entries {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "FEM unknown or element-entry ceiling exceeded",
            ));
        }
        Ok(())
    }
}

/// Basis values of one reference quadrature node, tabulated once per rule.
#[derive(Clone, Copy)]
pub(crate) struct ReferenceSample<R: RealScalar, const D: usize> {
    pub(crate) solution: BasisSample<R, D>,
    pub(crate) geometry: BasisSample<R, D>,
    /// P1 (barycentric) values, the mixed pressure basis.
    pub(crate) linear: [R; 4],
    pub(crate) weight: R,
    /// Opposite local vertex of a facet node; `None` for a volume node.
    pub(crate) facet: Option<usize>,
}

/// One quadrature node mapped to a cell or one of its exterior facets.
#[derive(Clone, Copy)]
pub(crate) struct Sample<R: RealScalar, const D: usize> {
    pub(crate) cell: usize,
    /// Index of the tabulated [`ReferenceSample`].
    pub(crate) reference: usize,
    pub(crate) point: [R; D],
    /// `|J| w` on cells and the physical surface measure times `w` on facets.
    pub(crate) weight: R,
    /// Physical basis gradients `J⁻ᵀ∇̂φ`.
    pub(crate) gradients: [[R; D]; MAX_NODES],
    /// Unit outward normal on facets; zero on cells.
    pub(crate) normal: [R; D],
}

/// Mapped volume nodes of every cell and the facet nodes of every exterior facet.
#[derive(Clone)]
pub(crate) struct CellSamples<R: RealScalar, const D: usize> {
    pub(crate) volume: Vec<Sample<R, D>>,
    per_cell: usize,
    pub(crate) facets: Vec<Sample<R, D>>,
    facet_offsets: Vec<usize>,
}

impl<R: RealScalar, const D: usize> CellSamples<R, D> {
    /// Volume nodes of `cell`.
    pub(crate) fn cell(&self, cell: usize) -> &[Sample<R, D>] {
        &self.volume[cell * self.per_cell..(cell + 1) * self.per_cell]
    }
    /// Exterior facet nodes of `cell`; empty when facets were not mapped.
    pub(crate) fn cell_facets(&self, cell: usize) -> &[Sample<R, D>] {
        self.facet_offsets
            .get(cell..cell + 2)
            .map_or(&[], |w| &self.facets[w[0]..w[1]])
    }
    /// Retained bytes of the samples of `cells` cells and `boundary_faces`
    /// exterior facets with `nodes` nodes each, for storage bounds.
    pub(crate) fn bound(
        cells: usize,
        boundary_faces: usize,
        nodes: usize,
        facets: bool,
    ) -> Result<Bytes, SciOxideError> {
        let facet_nodes = if facets {
            size(boundary_faces, nodes, quantity!("facet_samples"))?
        } else {
            0
        };
        Bytes::new(0)
            .array::<Sample<R, D>>(size(cells, nodes, quantity!("samples"))?)?
            .array::<Sample<R, D>>(facet_nodes)?
            .array::<usize>(add(cells, 1, quantity!("facet_offsets"))?)
    }
}

/// Continuous P1/P2 nodal space on a straight or quadratic simplex mesh.
#[derive(Clone)]
pub(crate) struct SimplexSpace<R: RealScalar, const D: usize> {
    pub(crate) mesh: SimplexMesh<R, D>,
    edge_points: Option<Vec<[R; D]>>,
    affine: Vec<SimplexFemMappedPoint<R, D>>,
    exterior_offsets: Vec<usize>,
    exterior: Vec<(usize, usize)>,
    pub(crate) order: SimplexFemOrder,
    /// Scalar nodes per component: vertices, then canonical edges for P2.
    pub(crate) nodes: usize,
    cells: Vec<[usize; MAX_NODES]>,
    incidence_offsets: Vec<usize>,
    incidence: Vec<(usize, usize)>,
    basis: ReferenceBasis<R, D>,
    geometry_basis: ReferenceBasis<R, D>,
    linear: ReferenceBasis<R, D>,
    tabulation: Vec<ReferenceSample<R, D>>,
    /// Tabulation ranges: volume `[0, 1)`, facet `o` `[o + 1, o + 2)`.
    ranges: [usize; 6],
    pub(crate) options: SimplexFemOptions<R>,
}

/// Kernel over one partition: first item index and its chunk of values.
type ScopedChunkKernel<'a, V> =
    dyn Fn(usize, &mut [V], &mut SolveContext<'_, '_>) -> Result<(), SciOxideError> + Sync + 'a;

type ChunkKernel<'a, V> = dyn Fn(usize, &mut [V]) -> Result<(), SciOxideError> + Sync + 'a;

/// Kernel over one cell: cell index, its item, the worker scratch and the child context.
type CellKernel<'a, T, W> = dyn Fn(usize, &mut T, &mut W, &mut SolveContext<'_, '_>) -> Result<(), SciOxideError>
    + Sync
    + 'a;

/// Checked lengths and finiteness of an evaluation's input, output and scratch.
pub(crate) fn check_buffers<V: NumericScalar>(
    n: usize,
    x: &[V],
    out: &[V],
    scratch: &[V],
    needed: usize,
) -> Result<(), SciOxideError> {
    if x.len() != n || x.iter().any(|v| !v.has_finite_components()) {
        Err(invalid(
            scioxide_core::argument!("x"),
            "FEM state shape or nonfinite value",
        ))
    } else if out.len() != n {
        Err(invalid(
            scioxide_core::argument!("output"),
            "FEM output shape",
        ))
    } else if scratch.len() < needed {
        Err(invalid(
            scioxide_core::argument!("scratch"),
            "FEM workspace",
        ))
    } else {
        Ok(())
    }
}

impl<R: RealScalar, const D: usize> SimplexSpace<R, D> {
    pub(crate) fn affine_map(&self, cell: usize) -> Option<&SimplexFemMappedPoint<R, D>> {
        self.affine.get(cell)
    }
    pub(crate) fn is_affine(&self) -> bool {
        self.edge_points.is_none()
    }
    pub(crate) fn reference_node(&self, index: usize) -> &ReferenceSample<R, D> {
        &self.tabulation[index]
    }
    /// Evaluate disjoint cell chunks with bounded stack-owned accounting storage.
    /// The executor controls scheduling; numerical output remains in caller scratch.
    pub(crate) fn for_each_chunk<V: NumericScalar>(
        &self,
        values: &mut [V],
        stride: usize,
        context: &mut SolveContext<'_, '_>,
        kernel: &ScopedChunkKernel<'_, V>,
    ) -> Result<(), SciOxideError> {
        let items = values.len() / stride;
        if items == 0 {
            return Ok(());
        }
        let plan = context.execution().plan(
            self.options.execution,
            items,
            scioxide_core::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: core::num::NonZeroUsize::MIN,
                worker_limit: WORKER_LIMIT,
            },
        );
        let mut children: [PartitionedSolve; MAX_WORKERS] =
            core::array::from_fn(|_| PartitionedSolve::default());
        let identity = context.identity();
        with_partitioned_solves(
            context,
            &mut children[..plan.worker_count()],
            SolveBudgets::unbounded(),
            |execution, children| {
                let mut slots: [PartitionSlot<(
                    &mut [V],
                    scioxide_core::backend::PartitionSolvePermit<'_>,
                )>; MAX_WORKERS] = core::array::from_fn(|_| PartitionSlot::empty());
                let mut tail = values;
                for (partition, child) in plan.partitions().zip(children) {
                    let (part, rest) = tail.split_at_mut(partition.range().len() * stride);
                    tail = rest;
                    slots[partition.worker_index()] = PartitionSlot::new((part, child));
                }
                execution.execute_partitioned(
                    plan,
                    &slots[..plan.worker_count()],
                    &|partition, (part, child), _| {
                        child.run(execution, identity, |context| {
                            for (i, chunk) in part.chunks_exact_mut(stride).enumerate() {
                                execution.check()?;
                                kernel(partition.range().start + i, chunk, context)?;
                            }
                            Ok(())
                        })
                    },
                )
            },
        )
    }
    /// Number DOFs, resolve rules of at least the given volume and facet degrees
    /// and tabulate every reference basis on them.
    pub(crate) fn prepare(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        order: SimplexFemOrder,
        mut options: SimplexFemOptions<R>,
        degrees: (usize, usize),
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Self, SciOxideError> {
        context.execution().check()?;
        let edge_points = validate_geometry(&mesh, geometry, &options)?;
        let (volume_rule, facet_rule) = rules::<R, D>(&mut options, degrees)?;
        let basis = ReferenceBasis::new(order)?;
        let geometry_basis = ReferenceBasis::new(if edge_points.is_some() {
            SimplexFemOrder::P2
        } else {
            SimplexFemOrder::P1
        })?;
        let linear = ReferenceBasis::new(SimplexFemOrder::P1)?;
        let (tabulation, ranges) = tabulate(
            [&basis, &geometry_basis, &linear],
            &volume_rule,
            facet_rule.as_ref(),
        )?;
        let cells = number(
            &mesh,
            if edge_points.is_some() {
                SimplexFemOrder::P2
            } else {
                order
            },
        )?;
        let nodes = Layout {
            order,
            components: 1,
            pressure: false,
        }
        .nodes(&mesh)?;
        let (incidence_offsets, incidence) = incidence(&cells, nodes, order.nodes(D))?;
        let (exterior_offsets, exterior) = boundary_incidence(&mesh)?;
        options.volume_rule = Some(volume_rule);
        options.facet_rule = facet_rule;
        context.execution().check()?;
        let mut space = Self {
            mesh,
            edge_points,
            affine: Vec::new(),
            exterior_offsets,
            exterior,
            order,
            nodes,
            cells,
            incidence_offsets,
            incidence,
            basis,
            geometry_basis,
            linear,
            tabulation,
            ranges,
            options,
        };
        if space.edge_points.is_none() {
            let mut affine = try_with_capacity(space.cell_count(), quantity!("affine_geometry"))?;
            let origin = space.geometry_basis.sample([R::ZERO; D])?;
            for cell in 0..space.cell_count() {
                affine.push(space.map(cell, &origin)?);
            }
            space.affine = affine;
        }
        Ok(space)
    }

    /// Conservative live storage of the space and its tabulation, excluding the mesh.
    pub(crate) fn storage_bound(
        mesh: &SimplexMesh<R, D>,
        order: SimplexFemOrder,
        limits: SimplexFemLimits,
    ) -> Result<Bytes, SciOxideError> {
        let nodes = Layout {
            order,
            components: 1,
            pressure: false,
        }
        .nodes(mesh)?;
        Bytes::new(PREPARATION_OVERHEAD_BYTES)
            .array::<SimplexFemMappedPoint<R, D>>(mesh.cell_count())?
            .array::<usize>(add(mesh.cell_count(), 1, quantity!("boundary_incidence"))?)?
            .array::<(usize, usize)>(mesh.boundary_faces().count())?
            .array::<(usize, usize, usize)>(mesh.boundary_faces().count())?
            .array::<ReferenceSample<R, D>>(size(
                limits.max_quadrature_nodes,
                D + 2,
                quantity!("tabulation"),
            )?)?
            .array::<[usize; MAX_NODES]>(mesh.cell_count())?
            .array::<usize>(add(nodes, 1, quantity!("incidence"))?)?
            .array::<(usize, usize)>(size(
                mesh.cell_count(),
                order.nodes(D),
                quantity!("incidence"),
            )?)?
            .array::<usize>(nodes)?
            .array::<[R; D]>(mesh.edge_count())
    }

    /// Global scalar-node indices of `cell`, before component offsets.
    pub(crate) fn dofs(&self, cell: usize) -> &[usize] {
        &self.cells[cell][..self.order.nodes(D)]
    }
    /// Checked [`Self::dofs`].
    pub(crate) fn cell_dofs(&self, cell: usize) -> Result<&[usize], SciOxideError> {
        if cell < self.cells.len() {
            Ok(self.dofs(cell))
        } else {
            Err(SciOxideError::out_of_bounds(
                quantity!("cells"),
                cell,
                self.cells.len(),
            ))
        }
    }
    /// Cell count.
    pub(crate) fn cell_count(&self) -> usize {
        self.cells.len()
    }
    /// Solution basis.
    pub(crate) const fn basis(&self) -> &ReferenceBasis<R, D> {
        &self.basis
    }
    /// Tabulated reference sample of `sample`.
    pub(crate) fn reference(&self, sample: &Sample<R, D>) -> &ReferenceSample<R, D> {
        &self.tabulation[sample.reference]
    }
    /// Tabulation indices of the volume rule.
    pub(crate) fn volume_nodes(&self) -> core::ops::Range<usize> {
        self.ranges[0]..self.ranges[1]
    }
    /// Tabulation indices of the facet rule on the facet opposite `opposite`.
    pub(crate) fn facet_nodes(&self, opposite: usize) -> core::ops::Range<usize> {
        self.ranges[opposite + 1]..self.ranges[opposite + 2]
    }

    /// Physical interpolation node, including curved edge coordinates for P2.
    pub(crate) fn dof_point(&self, node: usize) -> Result<[R; D], SciOxideError> {
        if node >= self.nodes {
            return Err(SciOxideError::out_of_bounds(
                quantity!("nodes"),
                node,
                self.nodes,
            ));
        }
        if node < self.mesh.vertex_count() {
            return Ok(*self.mesh.points()[node].as_array());
        }
        let edge = node - self.mesh.vertex_count();
        if let Some(points) = &self.edge_points {
            return Ok(points[edge]);
        }
        let pair = self.mesh.edge_vertices(edge)?;
        let mut point = [R::ZERO; D];
        for (axis, value) in point.iter_mut().enumerate() {
            *value = crate::spatial::midpoint(
                self.mesh.points()[pair[0]].as_array()[axis],
                self.mesh.points()[pair[1]].as_array()[axis],
            )?;
        }
        Ok(point)
    }

    /// Map a geometry basis sample on `cell`, checking orientation and conditioning.
    pub(crate) fn map(
        &self,
        cell: usize,
        geometry: &BasisSample<R, D>,
    ) -> Result<SimplexFemMappedPoint<R, D>, SciOxideError> {
        if let Some(cached) = self.affine.get(cell) {
            let mut map = *cached;
            for axis in 0..D {
                for j in 0..D {
                    map.point[axis] = finite(
                        map.point[axis] + map.jacobian[axis][j] * geometry.values[j + 1],
                        NonFiniteOrigin::Intermediate,
                    )?;
                }
            }
            return Ok(map);
        }
        let dofs = self.cells.get(cell).ok_or_else(|| {
            SciOxideError::out_of_bounds(quantity!("cells"), cell, self.cells.len())
        })?;
        let vertices = self.mesh.cell_vertices(cell)?;
        let count = if self.edge_points.is_some() {
            SimplexFemOrder::P2.nodes(D)
        } else {
            D + 1
        };
        let mut coordinates = [[R::ZERO; D]; MAX_NODES];
        for (node, coordinate) in coordinates.iter_mut().enumerate().take(count) {
            *coordinate = match &self.edge_points {
                Some(points) if node > D => points[dofs[node] - self.mesh.vertex_count()],
                _ => *self.mesh.points()[vertices[node]].as_array(),
            };
        }
        map_coordinates(
            &coordinates[..count],
            self.options.minimum_reciprocal_condition,
            geometry,
            false,
        )
    }
    /// Map a reference point of `cell`.
    pub(crate) fn map_point(
        &self,
        cell: usize,
        reference: [R; D],
    ) -> Result<SimplexFemMappedPoint<R, D>, SciOxideError> {
        self.map(cell, &self.geometry_basis.sample(reference)?)
    }
    /// Map the tabulated node `reference` of `cell`.
    pub(crate) fn map_node(
        &self,
        cell: usize,
        reference: usize,
    ) -> Result<SimplexFemMappedPoint<R, D>, SciOxideError> {
        self.map(cell, &self.tabulation[reference].geometry)
    }
    /// Check every geometry interpolation node of `cell`, including nodes that
    /// are not quadrature nodes.
    pub(crate) fn check_geometry(&self, cell: usize) -> Result<(), SciOxideError> {
        if cell < self.affine.len() {
            return Ok(());
        }
        let count = if self.edge_points.is_some() {
            SimplexFemOrder::P2.nodes(D)
        } else {
            D + 1
        };
        (0..count).try_for_each(|node| self.map_point(cell, reference_node(node)).map(|_| ()))
    }
    /// Physical point, weight, gradients and normal of tabulated node `reference` on `cell`.
    pub(crate) fn sample(
        &self,
        cell: usize,
        reference: usize,
    ) -> Result<Sample<R, D>, SciOxideError> {
        let tabulated = &self.tabulation[reference];
        let map = self.map(cell, &tabulated.geometry)?;
        self.mapped(cell, reference, tabulated, &map)
    }
    fn mapped(
        &self,
        cell: usize,
        reference: usize,
        tabulated: &ReferenceSample<R, D>,
        map: &SimplexFemMappedPoint<R, D>,
    ) -> Result<Sample<R, D>, SciOxideError> {
        let (normal, measure) = match tabulated.facet {
            Some(opposite) => normal(map, opposite)?,
            None => ([R::ZERO; D], map.determinant),
        };
        Ok(Sample {
            cell,
            reference,
            point: map.point,
            weight: reference::product(measure, tabulated.weight)?,
            gradients: physical_gradients(map, &tabulated.solution, self.order.nodes(D))?,
            normal,
        })
    }
    /// Volume sample of unit rule weight (so its weight is `|J|`) and its reference
    /// values at an arbitrary point; it has no tabulation index.
    pub(crate) fn point_sample(
        &self,
        cell: usize,
        point: [R; D],
    ) -> Result<(Sample<R, D>, ReferenceSample<R, D>), SciOxideError> {
        let tabulated = self.reference_sample(point, R::ONE, None)?;
        let map = self.map(cell, &tabulated.geometry)?;
        let sample = self.mapped(cell, usize::MAX, &tabulated, &map)?;
        Ok((sample, tabulated))
    }
    fn reference_sample(
        &self,
        point: [R; D],
        weight: R,
        facet: Option<usize>,
    ) -> Result<ReferenceSample<R, D>, SciOxideError> {
        reference_sample(
            [&self.basis, &self.geometry_basis, &self.linear],
            point,
            weight,
            facet,
        )
    }

    /// Field values and physical gradients of the leading `values.len()`
    /// component blocks of `input` at one reference point. Only this cell's
    /// input DOFs are read and validated, and outputs change only after every
    /// component evaluated successfully.
    pub(crate) fn evaluate_into<V: NumericScalar<Real = R>>(
        &self,
        cell: usize,
        reference: [R; D],
        input: &[V],
        values: &mut [V],
        gradients: &mut [[V; D]],
    ) -> Result<(), SciOxideError> {
        let (sample, tabulated) = self.point_sample(cell, reference)?;
        let dofs = self.cell_dofs(cell)?;
        let component = |component: usize| -> Result<(V, [V; D]), SciOxideError> {
            let intermediate = NonFiniteOrigin::Intermediate;
            let mut value = V::ZERO;
            let mut gradient = [V::ZERO; D];
            for (i, &dof) in dofs.iter().enumerate() {
                let coefficient = finite(input[component * self.nodes + dof], intermediate)?;
                value = finite(
                    value + coefficient * V::from_real(tabulated.solution.values[i]),
                    intermediate,
                )?;
                for (slot, &derivative) in gradient.iter_mut().zip(&sample.gradients[i]) {
                    *slot = finite(*slot + coefficient * V::from_real(derivative), intermediate)?;
                }
            }
            Ok((value, gradient))
        };
        // Validate every component before publishing; the second pass repeats the
        // same deterministic arithmetic and cannot fail.
        (0..values.len()).try_for_each(|c| component(c).map(|_| ()))?;
        for (c, (value, gradient)) in values.iter_mut().zip(gradients).enumerate() {
            (*value, *gradient) = component(c)?;
        }
        Ok(())
    }

    /// Exterior faces of `cell` with their opposite local vertex.
    pub(crate) fn exterior_faces(
        &self,
        cell: usize,
    ) -> Result<impl Iterator<Item = (usize, usize)> + '_, SciOxideError> {
        if cell >= self.cell_count() {
            return Err(SciOxideError::out_of_bounds(
                quantity!("cell"),
                cell,
                self.cell_count(),
            ));
        }
        Ok(
            self.exterior[self.exterior_offsets[cell]..self.exterior_offsets[cell + 1]]
                .iter()
                .copied(),
        )
    }

    /// Map the volume nodes of every cell and, when `facets`, the facet nodes of
    /// every exterior facet. One iteration is admitted per cell.
    pub(crate) fn samples(
        &self,
        facets: bool,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<CellSamples<R, D>, SciOxideError> {
        let per_cell = self.volume_nodes().len();
        let mut volume = try_with_capacity(
            size(self.cell_count(), per_cell, quantity!("samples"))?,
            quantity!("samples"),
        )?;
        let mut facet_samples = Vec::new();
        let mut facet_offsets = Vec::new();
        if facets {
            facet_offsets = try_with_capacity(
                add(self.cell_count(), 1, quantity!("facet_offsets"))?,
                quantity!("facet_offsets"),
            )?;
            facet_offsets.push(0);
        }
        for cell in 0..self.cell_count() {
            admit(context, false)?;
            for reference in self.volume_nodes() {
                volume.push(self.sample(cell, reference)?);
            }
            if facets {
                for (_, opposite) in self.exterior_faces(cell)? {
                    for reference in self.facet_nodes(opposite) {
                        let sample = self.sample(cell, reference)?;
                        facet_samples.try_reserve(1).map_err(|_| {
                            SciOxideError::allocation_failure(
                                quantity!("facet_samples"),
                                facet_samples.len() + 1,
                            )
                        })?;
                        facet_samples.push(sample);
                    }
                }
                facet_offsets.push(facet_samples.len());
            }
        }
        Ok(CellSamples {
            volume,
            per_cell,
            facets: facet_samples,
            facet_offsets,
        })
    }

    /// Packed index of local unknown `k` of `cell`.
    pub(crate) fn global(&self, layout: Layout, cell: usize, k: usize) -> usize {
        let l = self.order.nodes(D);
        let dofs = &self.cells[cell];
        if k < layout.components * l {
            (k / l) * self.nodes + dofs[k % l]
        } else {
            layout.components * self.nodes + dofs[k - layout.components * l]
        }
    }

    /// Route the element matrix `entry(cell, i, j)` of every cell through one
    /// eliminating builder, then the `extra` global triplets.
    pub(crate) fn assemble<V: NumericScalar>(
        &self,
        layout: Layout,
        constrained: Vec<bool>,
        elimination: Elimination,
        mut entry: impl FnMut(usize, usize, usize) -> Result<V, SciOxideError>,
        extra: impl IntoIterator<Item = (usize, usize, V)>,
    ) -> Result<Eliminated<V>, SciOxideError> {
        let m = layout.local::<D>();
        let capacity = add(
            layout.entries(&self.mesh)?,
            constrained.len(),
            quantity!("entry_bound"),
        )?;
        let mut builder = OperatorBuilder::with_elimination(constrained, capacity, elimination)?;
        for cell in 0..self.cell_count() {
            for i in 0..m {
                let row = self.global(layout, cell, i);
                if builder.constrained()[row] {
                    continue;
                }
                for j in 0..m {
                    builder.push(row, self.global(layout, cell, j), entry(cell, i, j)?)?;
                }
            }
        }
        for (row, column, value) in extra {
            builder.push(row, column, value)?;
        }
        builder.finish()
    }

    /// Element operator application `y = Σₑ Pₑᵀ Aₑ Pₑ x` into `staged` without
    /// scratch. Constrained columns are dropped and constrained rows replaced by
    /// the input (symmetric elimination). Rows are partitioned by the space's
    /// execution policy; every row reduces each incident cell's local row, then
    /// sums the cells in incidence order, for all worker counts.
    pub(crate) fn apply<V: NumericScalar<Real = R>>(
        &self,
        layout: Layout,
        entry: &(dyn Fn(usize, usize, usize) -> V + Sync),
        constrained: Option<&[bool]>,
        x: &[V],
        staged: &mut [V],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let m = layout.local::<D>();
        let free = |column: usize| constrained.is_none_or(|fixed| !fixed[column]);
        self.partitioned(staged, 1, execution, &|first, part| {
            for (offset, value) in part.iter_mut().enumerate() {
                execution.check()?;
                let row = first + offset;
                if !free(row) {
                    *value = x[row];
                    continue;
                }
                let mut sum = V::ZERO;
                for (cell, i) in self.incident(layout, row) {
                    let mut local = V::ZERO;
                    for j in 0..m {
                        let column = self.global(layout, cell, j);
                        if free(column) {
                            local = finite(
                                local + entry(cell, i, j) * x[column],
                                NonFiniteOrigin::Intermediate,
                            )?;
                        }
                    }
                    sum = finite(sum + local, NonFiniteOrigin::Intermediate)?;
                }
                *value = sum;
            }
            Ok(())
        })?;
        execution.check()
    }

    /// Incident cells of packed row `row` with the row's local unknown, in cell order.
    pub(crate) fn incident(
        &self,
        layout: Layout,
        row: usize,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        let l = self.order.nodes(D);
        let velocity = layout.components * self.nodes;
        let (node, base) = if row >= velocity {
            (row - velocity, layout.components * l)
        } else {
            (row % self.nodes, (row / self.nodes) * l)
        };
        self.incidence[self.incidence_offsets[node]..self.incidence_offsets[node + 1]]
            .iter()
            .map(move |&(cell, i)| (cell, base + i))
    }

    /// Sum local vectors (cells × local unknowns) into the packed rows `staged`.
    /// Every row adds incident cells in incidence order for all worker counts.
    pub(crate) fn scatter<V: NumericScalar>(
        &self,
        layout: Layout,
        local: &[V],
        staged: &mut [V],
        execution: ExecutionContext<'_>,
    ) -> Result<(), SciOxideError> {
        let m = layout.local::<D>();
        self.partitioned(staged, 1, execution, &|first, part| {
            for (offset, value) in part.iter_mut().enumerate() {
                execution.check()?;
                *value =
                    self.incident(layout, first + offset)
                        .try_fold(V::ZERO, |sum, (cell, k)| {
                            finite(sum + local[cell * m + k], NonFiniteOrigin::Intermediate)
                        })?;
            }
            Ok(())
        })?;
        execution.check()
    }

    /// Run `kernel(first item, chunk)` over chunks of `stride` values per item,
    /// partitioned by the execution policy.
    pub(crate) fn partitioned<V: NumericScalar>(
        &self,
        values: &mut [V],
        stride: usize,
        execution: ExecutionContext<'_>,
        kernel: &ChunkKernel<'_, V>,
    ) -> Result<(), SciOxideError> {
        let items = values.len() / stride;
        let plan = execution.plan(
            self.options.execution,
            items,
            scioxide_core::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: core::num::NonZeroUsize::MIN,
                worker_limit: WORKER_LIMIT,
            },
        );
        let mut slots: [PartitionSlot<&mut [V]>; MAX_WORKERS] =
            core::array::from_fn(|_| PartitionSlot::empty());
        let mut remaining = values;
        for partition in plan.partitions() {
            let (part, tail) = remaining.split_at_mut(partition.range().len() * stride);
            remaining = tail;
            slots[partition.worker_index()] = PartitionSlot::new(part);
        }
        execution.execute_partitioned(plan, &slots[..plan.worker_count()], &|p, part, _| {
            kernel(p.range().start, part)
        })
    }

    /// Run cell kernels with one child per cell for finite budgets, otherwise
    /// one child per worker. Finite shares stay independent of worker count;
    /// unbounded work needs only bounded stack-owned child storage.
    pub(crate) fn for_each_cell<T: Send, W: Send>(
        &self,
        items: &mut [T],
        mut worker: impl FnMut() -> Result<W, SciOxideError>,
        kernel: &CellKernel<'_, T, W>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if items.is_empty() {
            return Ok(());
        }
        let plan = context.execution().plan(
            self.options.execution,
            items.len(),
            scioxide_core::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: core::num::NonZeroUsize::MIN,
                worker_limit: WORKER_LIMIT,
            },
        );
        let mut workers = try_with_capacity(plan.worker_count(), quantity!("workers"))?;
        for _ in 0..plan.worker_count() {
            workers.push(worker()?);
        }
        let unbounded = scioxide_core::BudgetKind::ALL
            .iter()
            .all(|&kind| context.effective_budget_report(kind).remaining().is_none());
        let mut fixed: [PartitionedSolve; MAX_WORKERS] =
            core::array::from_fn(|_| PartitionedSolve::default());
        let mut children = if unbounded {
            Vec::new()
        } else {
            let mut children = try_with_capacity(items.len(), quantity!("children"))?;
            children.resize_with(items.len(), PartitionedSolve::default);
            children
        };
        let children = if unbounded {
            &mut fixed[..plan.worker_count()]
        } else {
            &mut children[..]
        };
        let identity = context.identity();
        with_partitioned_solves(
            context,
            children,
            SolveBudgets::unbounded(),
            |execution, children| {
                type Slot<'a, T, W> = PartitionSlot<(
                    &'a mut [T],
                    scioxide_core::backend::PartitionSolveBatch<'a>,
                    &'a mut W,
                )>;
                let mut slots: [Slot<'_, T, W>; MAX_WORKERS] =
                    core::array::from_fn(|_| PartitionSlot::empty());
                let mut remaining = &mut *items;
                let mut child_tail = children;
                for (partition, worker) in plan.partitions().zip(&mut workers) {
                    let (part, tail) = remaining.split_at_mut(partition.range().len());
                    remaining = tail;
                    let (child, tail) = child_tail.split_at(if unbounded {
                        1
                    } else {
                        partition.range().len()
                    })?;
                    child_tail = tail;
                    slots[partition.worker_index()] = PartitionSlot::new((part, child, worker));
                }
                execution.execute_partitioned(
                    plan,
                    &slots[..plan.worker_count()],
                    &|partition, (part, children, worker), _| {
                        if unbounded {
                            let child = children.into_iter().next().ok_or_else(|| {
                                invalid(
                                    scioxide_core::argument!("children"),
                                    "missing worker child",
                                )
                            })?;
                            return child.run(execution, identity, |context| {
                                for (i, item) in part.iter_mut().enumerate() {
                                    kernel(partition.range().start + i, item, worker, context)?;
                                }
                                Ok(())
                            });
                        }
                        for (i, (item, child)) in part.iter_mut().zip(children).enumerate() {
                            child.run(execution, identity, |context| {
                                kernel(partition.range().start + i, item, worker, context)
                            })?;
                        }
                        Ok(())
                    },
                )
            },
        )
    }

    /// Source, natural/Robin and Dirichlet sites of `boundaries` on every cell;
    /// source sites are collected only when `sources`.
    pub(crate) fn loads<V: NumericScalar<Real = R>>(
        &self,
        boundaries: &TaggedBoundaries<R, V, D>,
        sources: bool,
    ) -> Result<PreparedLoads<R, V, D>, SciOxideError> {
        let components = boundaries.components;
        let mut loads = PreparedLoads::new(size(components, self.nodes, quantity!("unknowns"))?)?;
        if sources {
            let sites = size(
                size(
                    self.cell_count(),
                    self.volume_nodes().len(),
                    quantity!("source_sites"),
                )?,
                components,
                quantity!("source_sites"),
            )?;
            loads.reserve_sources(
                sites,
                size(sites, self.order.nodes(D), quantity!("source_entries"))?,
            )?;
        }
        for cell in 0..self.cell_count() {
            let dofs = self.dofs(cell);
            if sources {
                for reference in self.volume_nodes() {
                    let map = self.map_node(cell, reference)?;
                    let weight =
                        reference::product(map.determinant, self.tabulation[reference].weight)?;
                    let values = &self.tabulation[reference].solution.values;
                    for component in 0..components {
                        loads.push_source(
                            crate::FieldPoint {
                                point: map.point,
                                component,
                            },
                            dofs.iter().enumerate().map(|(i, &dof)| {
                                (component * self.nodes + dof, weight * values[i])
                            }),
                        )?;
                    }
                }
            }
            for (face, opposite) in self.exterior_faces(cell)? {
                for component in 0..components {
                    let condition = boundaries.slot(face, component)?;
                    if let BoundaryKind::Dirichlet = boundaries.kind_at(face, component)? {
                        self.dirichlet_sites(
                            cell, face, opposite, component, condition, &mut loads,
                        )?;
                        continue;
                    }
                    for reference in self.facet_nodes(opposite) {
                        let sample = self.sample(cell, reference)?;
                        let values = &self.tabulation[reference].solution.values;
                        loads.push_data(
                            crate::BoundaryPoint {
                                point: sample.point,
                                normal: sample.normal,
                                component,
                            },
                            condition,
                            dofs.iter().enumerate().map(|(i, &dof)| {
                                (component * self.nodes + dof, sample.weight * values[i])
                            }),
                        )?;
                    }
                }
            }
        }
        Ok(loads)
    }
    fn dirichlet_sites<V: NumericScalar<Real = R>>(
        &self,
        cell: usize,
        face: usize,
        opposite: usize,
        component: usize,
        condition: usize,
        loads: &mut PreparedLoads<R, V, D>,
    ) -> Result<(), SciOxideError> {
        for (node, &dof) in self.dofs(cell).iter().enumerate() {
            let point = reference_node::<R, D>(node);
            let lambda = if opposite == 0 {
                R::ONE - point.iter().copied().fold(R::ZERO, |s, x| s + x)
            } else {
                point[opposite - 1]
            };
            if lambda != R::ZERO {
                continue;
            }
            let map = self.map_point(cell, point)?;
            let (normal, _) = normal(&map, opposite)?;
            loads.push_dirichlet(crate::discretization::DirichletSite {
                dof: component * self.nodes + dof,
                site: crate::BoundaryPoint {
                    point: map.point,
                    normal,
                    component,
                },
                condition,
                rank: (condition, face),
            })?;
        }
        Ok(())
    }

    /// Add the consistent body load `∫ f·φ` of the first `layout.components`
    /// blocks to `load`, admitting one iteration per cell and one provider call
    /// per quadrature node.
    pub(crate) fn body_load_into<V: NumericScalar<Real = R>>(
        &self,
        layout: Layout,
        body: &(dyn Fn([R; D]) -> Result<[V; D], SciOxideError> + Sync),
        load: &mut [V],
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let l = self.order.nodes(D);
        for cell in 0..self.cell_count() {
            admit(context, false)?;
            for reference in self.volume_nodes() {
                let sample = self.sample(cell, reference)?;
                let values = &self.tabulation[reference].solution.values;
                admit(context, true)?;
                let force = body(sample.point)?;
                if force.iter().any(|x| !x.has_finite_components()) {
                    return Err(SciOxideError::non_finite(
                        NonFiniteOrigin::Callback,
                        "nonfinite body load",
                    ));
                }
                for (a, &component) in force.iter().enumerate().take(layout.components) {
                    for (i, &value) in values.iter().enumerate().take(l) {
                        let row = self.global(layout, cell, a * l + i);
                        load[row] = finite(
                            load[row] + component * V::from_real(sample.weight * value),
                            NonFiniteOrigin::Intermediate,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Static traction-plus-body load and prescribed values of `boundaries` on a
    /// vector layout; pressure rows carry no load and are never prescribed.
    pub(crate) fn static_loads<V: NumericScalar<Real = R>>(
        &self,
        layout: Layout,
        boundaries: &TaggedBoundaries<R, V, D>,
        body: &(dyn Fn([R; D]) -> Result<[V; D], SciOxideError> + Sync),
        context: &mut SolveContext<'_, '_>,
    ) -> Result<(Vec<V>, Vec<Option<V>>), SciOxideError> {
        let unknowns = layout.unknowns(&self.mesh)?;
        let vector = size(layout.components, self.nodes, quantity!("unknowns"))?;
        let loads = self.loads(boundaries, false)?.finish(None);
        let mut load = try_filled(unknowns, V::ZERO, quantity!("load"))?;
        loads.natural_into(
            boundaries,
            &crate::FieldData::zero(),
            R::ZERO,
            &mut load[..vector],
        )?;
        self.body_load_into(layout, body, &mut load, context)?;
        let mut values = try_filled(vector, V::ZERO, quantity!("prescribed"))?;
        loads.dirichlet_into(boundaries, R::ZERO, &mut values)?;
        let mut prescribed = try_filled(unknowns, None, quantity!("prescribed"))?;
        for &dof in loads.dirichlet_dofs() {
            prescribed[dof] = Some(values[dof]);
        }
        Ok((load, prescribed))
    }

    /// Symmetric Dirichlet right-hand side: prescribed values on constrained rows
    /// and `load − A_fc g` on free rows, for the element matrices `entry`.
    pub(crate) fn lifted_rhs<V: NumericScalar<Real = R>>(
        &self,
        layout: Layout,
        prescribed: &[Option<V>],
        load: &[V],
        entry: impl FnMut(usize, usize, usize) -> Result<V, SciOxideError>,
    ) -> Result<Vec<V>, SciOxideError> {
        let mut constrained = try_with_capacity(prescribed.len(), quantity!("constrained"))?;
        constrained.extend(prescribed.iter().map(Option::is_some));
        let (_, lift, _) = self.assemble(
            layout,
            constrained,
            Elimination::Lift,
            entry,
            core::iter::empty(),
        )?;
        let mut rhs = try_with_capacity(load.len(), quantity!("rhs"))?;
        rhs.extend(load.iter().zip(prescribed).map(|(&f, &g)| g.unwrap_or(f)));
        crate::discretization::apply_lift(&lift, &mut rhs)?;
        Ok(rhs)
    }

    /// Connected components of the cell–node graph, labeled in order of their
    /// smallest node, so vertex components are ordered by smallest vertex.
    pub(crate) fn components(&self) -> Result<GraphComponents, SciOxideError> {
        let l = self.order.nodes(D);
        let mut edges = SparseBuilder::<R>::with_capacity(
            self.nodes,
            self.nodes,
            size(self.cell_count(), 2 * l, quantity!("graph_edges"))?,
        )?;
        for cell in 0..self.cell_count() {
            let dofs = self.dofs(cell);
            for &node in &dofs[1..] {
                edges.push(dofs[0], node, R::ONE)?;
                edges.push(node, dofs[0], R::ONE)?;
            }
        }
        let graph = edges.finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)?;
        GraphPlan::from_pattern(graph.structural_pattern(), GraphDirection::Undirected)?
            .weak_components()
    }

    /// Transactionally pack borrowed exact-mesh fields of `layout`.
    pub(crate) fn pack_into<V: NumericScalar<Real = R>>(
        &self,
        layout: Layout,
        fields: SimplexFemFieldView<'_, '_, V, D>,
        pressure: Option<&FieldView<'_, V, Vertex, D>>,
        output: &mut [V],
    ) -> Result<(), SciOxideError> {
        let edges = if self.order == SimplexFemOrder::P2 {
            layout.components
        } else {
            0
        };
        if output.len() != layout.unknowns(&self.mesh)? {
            return Err(invalid(
                scioxide_core::argument!("output"),
                "FEM field layout",
            ));
        }
        if fields.vertices.len() != layout.components
            || fields.edges.len() != edges
            || pressure.is_some() != layout.pressure
        {
            return Err(invalid(
                scioxide_core::argument!("fields"),
                "FEM field layout",
            ));
        }
        let vertex_count = self.mesh.vertex_count();
        let checks = fields
            .vertices
            .iter()
            .chain(pressure)
            .map(|f| (f.mesh_token(), f.values(), vertex_count))
            .chain(
                fields
                    .edges
                    .iter()
                    .map(|f| (f.mesh_token(), f.values(), self.mesh.edge_count())),
            );
        for (token, values, len) in checks {
            self.mesh.mesh_token().ensure_compatible(token)?;
            if values.len() != len || values.iter().any(|v| !v.has_finite_components()) {
                return Err(invalid(
                    scioxide_core::argument!("fields"),
                    "invalid FEM field",
                ));
            }
        }
        for (component, field) in fields.vertices.iter().enumerate() {
            let block = &mut output[component * self.nodes..(component + 1) * self.nodes];
            block[..vertex_count].copy_from_slice(field.values());
            if edges != 0 {
                block[vertex_count..].copy_from_slice(fields.edges[component].values());
            }
        }
        if let Some(pressure) = pressure {
            output[layout.components * self.nodes..].copy_from_slice(pressure.values());
        }
        Ok(())
    }

    /// Owned shared fields of a packed `layout` state.
    pub(crate) fn unpack<V: NumericScalar<Real = R>>(
        &self,
        layout: Layout,
        values: &[V],
    ) -> Result<(SimplexFemField<V, D>, Option<VertexField<V, D>>), SciOxideError> {
        if values.len() != layout.unknowns(&self.mesh)?
            || values.iter().any(|v| !v.has_finite_components())
        {
            return Err(invalid(
                scioxide_core::argument!("values"),
                "invalid FEM scalar field",
            ));
        }
        let p2 = self.order == SimplexFemOrder::P2;
        let vertex_count = self.mesh.vertex_count();
        let copy = |values: &[V]| -> Result<Vec<V>, SciOxideError> {
            let mut out = try_with_capacity(values.len(), quantity!("field_values"))?;
            out.extend_from_slice(values);
            Ok(out)
        };
        let mut vertices = try_with_capacity(layout.components, quantity!("vertices"))?;
        let mut edges =
            try_with_capacity(if p2 { layout.components } else { 0 }, quantity!("edges"))?;
        for block in values[..layout.components * self.nodes].chunks_exact(self.nodes) {
            vertices.push(VertexField::new(&self.mesh, copy(&block[..vertex_count])?)?);
            if p2 {
                edges.push(UnstructuredEdgeField::new(
                    &self.mesh,
                    copy(&block[vertex_count..])?,
                )?);
            }
        }
        let pressure = if layout.pressure {
            Some(VertexField::new(
                &self.mesh,
                copy(&values[layout.components * self.nodes..])?,
            )?)
        } else {
            None
        };
        Ok((SimplexFemField { vertices, edges }, pressure))
    }
}

/// Validate the conditioning cutoff, facet rule and quadratic coordinates.
fn validate_geometry<R: RealScalar, const D: usize>(
    mesh: &SimplexMesh<R, D>,
    geometry: SimplexFemGeometry<'_, R, D>,
    options: &SimplexFemOptions<R>,
) -> Result<Option<Vec<[R; D]>>, SciOxideError> {
    let cutoff = options.minimum_reciprocal_condition;
    if D == 1 && options.facet_rule.is_some() {
        return Err(invalid(
            scioxide_core::argument!("options"),
            "a facet rule for point facets",
        ));
    }
    if !cutoff.is_finite() || cutoff < R::ZERO || cutoff >= R::ONE {
        return Err(invalid(
            scioxide_core::argument!("options"),
            "Jacobian reciprocal condition cutoff must be in [0,1)",
        ));
    }
    let SimplexFemGeometry::Quadratic(field) = geometry else {
        return Ok(None);
    };
    mesh.mesh_token().ensure_compatible(field.mesh_token())?;
    if field.len() != mesh.edge_count() || field.values().iter().flatten().any(|x| !x.is_finite()) {
        return Err(invalid(
            scioxide_core::argument!("geometry"),
            "invalid quadratic edge coordinates",
        ));
    }
    let mut points = try_with_capacity(field.len(), quantity!("edge_points"))?;
    points.extend_from_slice(field.values());
    Ok(Some(points))
}

/// Explicit or positive default volume and facet rules of at least `degrees`.
/// One-dimensional facets are points, so they have no rule.
fn rules<R: RealScalar, const D: usize>(
    options: &mut SimplexFemOptions<R>,
    (volume, facet): (usize, usize),
) -> Result<(SimplexRule<R>, Option<SimplexRule<R>>), SciOxideError> {
    let max = options.limits.max_quadrature_nodes;
    if max == 0 {
        return Err(invalid(
            scioxide_core::argument!("options"),
            "FEM quadrature node ceiling is zero",
        ));
    }
    let positive = |dimension, degree| {
        SimplexRule::positive(
            dimension,
            degree,
            scioxide_integrate::GeometricOptions::default().with_max_nodes(Budget::try_from(max)?),
        )
    };
    let volume_rule = match options.volume_rule.take() {
        Some(rule) => rule,
        None => positive(D, volume)?,
    };
    let facet_rule = match options.facet_rule.take() {
        Some(rule) => Some(rule),
        None if D == 1 => None,
        None => Some(positive(D - 1, facet)?),
    };
    for (rule, dimension, degree) in [
        (Some(&volume_rule), D, volume),
        (facet_rule.as_ref(), D - 1, facet),
    ] {
        if rule.is_some_and(|rule| {
            rule.dimension() != dimension
                || rule.exactness_degree() < degree
                || rule.node_count() > max
        }) {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "inadequate FEM quadrature dimension, degree or node limit",
            ));
        }
    }
    Ok((volume_rule, facet_rule))
}

/// Solution, geometry and P1 samples at one reference point.
fn reference_sample<R: RealScalar, const D: usize>(
    [basis, geometry, linear]: [&ReferenceBasis<R, D>; 3],
    point: [R; D],
    weight: R,
    facet: Option<usize>,
) -> Result<ReferenceSample<R, D>, SciOxideError> {
    let mut barycentric = [R::ZERO; 4];
    barycentric[..=D].copy_from_slice(&linear.sample(point)?.values[..=D]);
    Ok(ReferenceSample {
        solution: basis.sample(point)?,
        geometry: geometry.sample(point)?,
        linear: barycentric,
        weight,
        facet,
    })
}

type Tabulation<R, const D: usize> = (Vec<ReferenceSample<R, D>>, [usize; 6]);

/// Tabulate the volume rule, then the facet rule embedded on each facet.
fn tabulate<R: RealScalar, const D: usize>(
    bases: [&ReferenceBasis<R, D>; 3],
    volume_rule: &SimplexRule<R>,
    facet_rule: Option<&SimplexRule<R>>,
) -> Result<Tabulation<R, D>, SciOxideError> {
    let facet_nodes = facet_rule.map_or(1, SimplexRule::node_count);
    let mut table = try_with_capacity(
        add(
            volume_rule.node_count(),
            size(D + 1, facet_nodes, quantity!("tabulation"))?,
            quantity!("tabulation"),
        )?,
        quantity!("tabulation"),
    )?;
    let mut ranges = [0; 6];
    for (point, weight) in super::reference::rule_nodes(volume_rule) {
        table.push(reference_sample(bases, point, weight, None)?);
    }
    ranges[1] = table.len();
    for opposite in 0..=D {
        if let Some(rule) = facet_rule {
            for (bary, &weight) in rule.barycentric_nodes().chunks_exact(D).zip(rule.weights()) {
                let point = facet_point(opposite, bary);
                table.push(reference_sample(bases, point, weight, Some(opposite))?);
            }
        } else {
            // The facet opposite vertex `opposite` of an interval is the other vertex.
            let point = reference_node(1 - opposite);
            table.push(reference_sample(bases, point, R::ONE, Some(opposite))?);
        }
        ranges[opposite + 2] = table.len();
    }
    Ok((table, ranges))
}

/// Vertices, then local edges in lexicographic local vertex pairs, independently
/// of canonical mesh edge ordering and orientation.
fn number<R: RealScalar, const D: usize>(
    mesh: &SimplexMesh<R, D>,
    order: SimplexFemOrder,
) -> Result<Vec<[usize; MAX_NODES]>, SciOxideError> {
    let mut cells = try_with_capacity(mesh.cell_count(), quantity!("cell_dofs"))?;
    for cell in 0..mesh.cell_count() {
        let vertices = mesh.cell_vertices(cell)?;
        let mut dofs = [0usize; MAX_NODES];
        dofs[..=D].copy_from_slice(vertices);
        let mut k = D + 1;
        if order == SimplexFemOrder::P1 {
            cells.push(dofs);
            continue;
        }
        for i in 0..=D {
            for j in i + 1..=D {
                let pair = [vertices[i].min(vertices[j]), vertices[i].max(vertices[j])];
                let edge = mesh
                    .cell_edges(cell)?
                    .iter()
                    .map(|edge| edge.index())
                    .find(|&edge| mesh.edge_vertices(edge).is_ok_and(|v| *v == pair))
                    .ok_or_else(|| {
                        invalid(
                            scioxide_core::argument!("mesh"),
                            "missing canonical cell edge",
                        )
                    })?;
                dofs[k] = add(mesh.vertex_count(), edge, quantity!("dofs"))?;
                k += 1;
            }
        }
        cells.push(dofs);
    }
    Ok(cells)
}

type Incidence = (Vec<usize>, Vec<(usize, usize)>);

/// Node → (cell, local node) incidence in cell order, as offsets and entries.
fn incidence(
    cells: &[[usize; MAX_NODES]],
    nodes: usize,
    local: usize,
) -> Result<Incidence, SciOxideError> {
    let mut offsets = try_filled(
        add(nodes, 1, quantity!("offsets"))?,
        0usize,
        quantity!("offsets"),
    )?;
    for dofs in cells {
        for &dof in &dofs[..local] {
            offsets[dof + 1] += 1;
        }
    }
    for i in 1..offsets.len() {
        offsets[i] = add(offsets[i], offsets[i - 1], quantity!("offsets"))?;
    }
    let mut entries = try_filled(
        size(cells.len(), local, quantity!("incidence"))?,
        (0usize, 0usize),
        quantity!("incidence"),
    )?;
    let mut next = try_with_capacity(nodes, quantity!("incidence_cursors"))?;
    next.extend_from_slice(&offsets[..nodes]);
    for (cell, dofs) in cells.iter().enumerate() {
        for (i, &dof) in dofs[..local].iter().enumerate() {
            entries[next[dof]] = (cell, i);
            next[dof] += 1;
        }
    }
    Ok((offsets, entries))
}

/// Group canonical exterior faces by their owning cell once.
fn boundary_incidence<R: RealScalar, const D: usize>(
    mesh: &SimplexMesh<R, D>,
) -> Result<Incidence, SciOxideError> {
    let mut grouped = try_with_capacity(
        mesh.boundary_faces().count(),
        quantity!("boundary_incidence"),
    )?;
    let mut offsets = try_filled(
        add(mesh.cell_count(), 1, quantity!("boundary_incidence"))?,
        0usize,
        quantity!("boundary_incidence"),
    )?;
    for face in mesh.boundary_faces() {
        let cell = mesh.face_cells(face)?[0];
        let face_vertices = mesh.face_vertices(face)?;
        let opposite = mesh
            .cell_vertices(cell)?
            .iter()
            .position(|v| !face_vertices.contains(v))
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("mesh"),
                    "invalid exterior incidence",
                )
            })?;
        grouped.push((cell, face, opposite));
        offsets[cell + 1] += 1;
    }
    grouped.sort_unstable();
    for i in 1..offsets.len() {
        offsets[i] += offsets[i - 1];
    }
    let mut faces = try_with_capacity(grouped.len(), quantity!("boundary_incidence"))?;
    faces.extend(
        grouped
            .into_iter()
            .map(|(_, face, opposite)| (face, opposite)),
    );
    Ok((offsets, faces))
}
