use super::*;
use crate::fem::simplex::{
    MAX_NODES,
    reference::{
        BasisSample, ReferenceBasis, facet_normal, map_coordinates, mapped_normal,
        physical_gradients, rule_nodes,
    },
};
use crate::{SimplexFemMappedPoint, SimplexFemOrder};
use scioxide_integrate::{GaussianRulePlan, SimplexRule};
use scioxide_spatial::{UnstructuredEdge, Vertex};

/// Continuous nodal space on straight adaptive cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AmrOrder {
    /// Linear simplex values at vertices (triangles/tetrahedra).
    P1,
    /// Quadratic simplex values at vertices and edge midpoints.
    P2,
    /// Multilinear values at tensor-cell vertices (including segments).
    Q1,
}

/// Owned nodal field using spatial's exact mesh identity and location markers.
#[derive(Debug, Clone)]
pub struct AmrNodalField<V, const D: usize> {
    vertices: Field<V, Vertex, D>,
    edges: Option<Field<V, UnstructuredEdge, D>>,
}
/// Borrowed nodal field; construction of spatial views checks shapes and epochs.
#[derive(Debug, Clone, Copy)]
pub struct AmrNodalFieldView<'a, V, const D: usize> {
    vertices: FieldView<'a, V, Vertex, D>,
    edges: Option<FieldView<'a, V, UnstructuredEdge, D>>,
}
impl<V, const D: usize> AmrNodalField<V, D> {
    /// Borrow values without allocation.
    pub fn view(&self) -> AmrNodalFieldView<'_, V, D> {
        AmrNodalFieldView {
            vertices: self.vertices.view(),
            edges: self.edges.as_ref().map(Field::view),
        }
    }
    /// Vertex values.
    pub const fn vertices(&self) -> &Field<V, Vertex, D> {
        &self.vertices
    }
    /// Edge-midpoint values, present exactly for P2.
    pub const fn edges(&self) -> Option<&Field<V, UnstructuredEdge, D>> {
        self.edges.as_ref()
    }
    /// Separate vertex and edge values.
    pub fn into_parts(self) -> (Field<V, Vertex, D>, Option<Field<V, UnstructuredEdge, D>>) {
        (self.vertices, self.edges)
    }
}
impl<'a, V, const D: usize> AmrNodalFieldView<'a, V, D> {
    /// Vertex values and, exactly for P2, edge-midpoint values.
    pub const fn new(
        vertices: FieldView<'a, V, Vertex, D>,
        edges: Option<FieldView<'a, V, UnstructuredEdge, D>>,
    ) -> Self {
        Self { vertices, edges }
    }
    /// Vertex values.
    pub const fn vertices(&self) -> &FieldView<'a, V, Vertex, D> {
        &self.vertices
    }
    /// Edge-midpoint values, present exactly for P2.
    pub const fn edges(&self) -> Option<&FieldView<'a, V, UnstructuredEdge, D>> {
        self.edges.as_ref()
    }
}

#[derive(Clone)]
pub(super) struct CellData<R: RealScalar, const D: usize> {
    pub(super) dofs: [usize; MAX_NODES],
    pub(super) map: SimplexFemMappedPoint<R, D>,
    pub(super) mass: [R; 100],
    pub(super) stiffness: [R; 100],
}
#[derive(Clone, Copy)]
pub(super) struct Sample<R: RealScalar, const D: usize> {
    pub(super) point: [R; D],
    pub(super) weight: R,
}
#[derive(Clone, Copy)]
pub(super) struct FaceSample<R: RealScalar, const D: usize> {
    pub(super) point: [R; D],
    pub(super) weight: R,
    pub(super) normal: [R; D],
}

/// Immutable nodal layout, positive quadrature, and hanging-node interpolation.
/// Face quadrature points, weights and normals are cached for repeated estimation.
/// Topology is shared with spatial; no conforming replacement mesh is constructed.
#[derive(Clone)]
pub struct AmrSpace<R: RealScalar, const D: usize> {
    pub(super) mesh: AdaptiveMesh<R, D>,
    pub(super) options: AmrOptions<R>,
    pub(super) order: AmrOrder,
    pub(super) local: usize,
    pub(super) points: Vec<[R; D]>,
    pub(super) cells: Vec<CellData<R, D>>,
    basis: Option<ReferenceBasis<R, D>>,
    pub(super) volume: Vec<Sample<R, D>>,
    pub(super) facets: Vec<Sample<R, D>>,
    face_samples: Vec<FaceSample<R, D>>,
    exterior_cells: Vec<Option<usize>>,
    pub(super) constraints: AmrMap<R>,
    masters: Vec<usize>,
    pub(super) storage: usize,
    pub(super) incidence: Vec<Vec<(usize, usize)>>,
}

impl<R: RealScalar, const D: usize> AmrSpace<R, D> {
    /// Prepare with default numerical settings and no resource ceiling.
    ///
    /// # Errors
    /// Returns the geometry, coefficient, allocation and configuration errors of [`Self::with_options`].
    pub fn new(mesh: &AdaptiveMesh<R, D>, order: AmrOrder) -> Result<Self>
    where
        R: scioxide_core::NumericScalar<Real = R>,
    {
        Self::with_options(mesh, order, AmrOptions::default())
    }

    /// Prepare P1/P2 simplex or Q1 tensor constraints and quadrature.
    /// Every cell is the straight-sided image of its vertices: affine maps for
    /// simplices and boxes, with P2 edge nodes at the edge midpoints.
    /// # Errors
    /// Rejects family/order mismatches, ill-conditioned geometry, inconsistent
    /// constraint traces, periodic nodal spaces, and resource/allocation failures.
    pub fn with_options(
        mesh: &AdaptiveMesh<R, D>,
        order: AmrOrder,
        options: AmrOptions<R>,
    ) -> Result<Self>
    where
        R: scioxide_core::NumericScalar<Real = R>,
    {
        Self::build(mesh, order, options, ExecutionContext::sequential_native())
    }
    /// Prepare as one epoch of `scope`, charging node budgets and honoring its
    /// cancellation and deadlines.
    /// # Errors
    /// Same failures as `new`, including inherited budget exhaustion.
    pub fn build<'scope, 'capability>(
        mesh: &AdaptiveMesh<R, D>,
        order: AmrOrder,
        options: AmrOptions<R>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<Self>
    where
        R: scioxide_core::NumericScalar<Real = R>,
    {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "amr-space", 2),
            |context| {
                Self::prepare(
                    mesh,
                    order,
                    options,
                    &mut Work::new(options.limits, Some(context)),
                )
            },
        )
    }
    fn prepare(
        mesh: &AdaptiveMesh<R, D>,
        order: AmrOrder,
        options: AmrOptions<R>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Self>
    where
        R: scioxide_core::NumericScalar<Real = R>,
    {
        validate_space(mesh, order, &options)?;
        let simplex = mesh.cell_kind() == AdaptiveCellKind::Simplex;
        let n = add(
            mesh.vertex_count(),
            if order == AmrOrder::P2 {
                mesh.edge_count()
            } else {
                0
            },
            scioxide_core::quantity!("dofs"),
        )?;
        work.dofs(n)?;
        work.tick(add(
            mesh.cell_count(),
            mesh.face_count(),
            scioxide_core::quantity!("work"),
        )?)?;
        let local = match order {
            AmrOrder::P1 => D + 1,
            AmrOrder::P2 => (D + 1) * (D + 2) / 2,
            AmrOrder::Q1 => 1 << D,
        };
        work.entries(size(
            mesh.cell_count(),
            size(local, local, scioxide_core::quantity!("entries"))?,
            scioxide_core::quantity!("entries"),
        )?)?;
        let basis = if simplex {
            Some(ReferenceBasis::new(if order == AmrOrder::P1 {
                SimplexFemOrder::P1
            } else {
                SimplexFemOrder::P2
            })?)
        } else {
            None
        };
        let (volume, facets) = quadrature::<R, D>(simplex, options.quadrature_degree, work)?;
        let (points, edges) = nodal_points(mesh, order, n, work)?;
        let cells = cell_data(mesh, order, local, &points, &edges, &options, work)?;
        let empty = AmrMap::new(csr(0, 0, &[], work)?, options.execution);
        let mut incidence: Vec<Vec<(usize, usize)>> =
            work.vector(n, scioxide_core::quantity!("incidence"))?;
        for _ in 0..n {
            incidence.push(Vec::new());
        }
        for (cell, data) in cells.iter().enumerate() {
            for (i, &node) in data.dofs[..local].iter().enumerate() {
                work.tick(1)?;
                work.push(
                    &mut incidence[node],
                    (cell, i),
                    scioxide_core::quantity!("incidence"),
                )?;
            }
        }
        let mut result = Self {
            mesh: mesh.clone(),
            options,
            order,
            local,
            points,
            cells,
            basis,
            volume,
            facets,
            face_samples: Vec::new(),
            exterior_cells: Vec::new(),
            constraints: empty,
            masters: Vec::new(),
            storage: 0,
            incidence,
        };
        result.build_constraints(work)?;
        result.prepare_forms(work)?;
        result.prepare_faces(work)?;
        result.storage = work.bytes;
        Ok(result)
    }
    /// Unit-diffusion stiffness and consistent mass of every cell.
    fn prepare_forms(&mut self, work: &mut Work<'_, '_, '_>) -> Result<()> {
        let local = self.local;
        let mut diffusion = [R::ZERO; 9];
        for axis in 0..D {
            diffusion[axis * D + axis] = R::ONE;
        }
        for cell in 0..self.cells.len() {
            let mut mass = [R::ZERO; 100];
            let mut stiffness = [R::ZERO; 100];
            for q in &self.volume {
                work.tick(size(
                    size(local, local, scioxide_core::quantity!("work"))?,
                    size(D, D, scioxide_core::quantity!("work"))?,
                    scioxide_core::quantity!("work"),
                )?)?;
                let sample = self.sample(cell, q.point, scioxide_core::argument!("mesh"))?;
                crate::fem::simplex::reference::accumulate_forms(
                    local,
                    1,
                    finite(
                        q.weight * self.cells[cell].map.determinant,
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?,
                    &sample.values,
                    &sample.gradients,
                    &diffusion[..D * D],
                    &[R::ZERO],
                    &mut stiffness[..local * local],
                    &mut mass[..local * local],
                )?;
            }
            self.cells[cell].mass = mass;
            self.cells[cell].stiffness = stiffness;
        }
        Ok(())
    }
    /// Exact adaptive mesh epoch.
    pub fn mesh(&self) -> &AdaptiveMesh<R, D> {
        &self.mesh
    }
    /// Selected nodal order.
    pub fn order(&self) -> AmrOrder {
        self.order
    }
    /// Numerical preparation choices, including quadrature and geometry conditioning.
    pub fn options(&self) -> AmrOptions<R> {
        self.options
    }
    /// Full vertex-plus-edge nodal count, including hanging nodes.
    pub fn full_dofs(&self) -> usize {
        self.points.len()
    }
    /// Independent nodal count before essential boundary elimination.
    pub fn independent_dofs(&self) -> usize {
        self.masters.len()
    }
    /// Full-node index for every independent degree of freedom, in ascending order.
    pub fn independent_nodes(&self) -> &[usize] {
        &self.masters
    }
    /// Conforming prolongation `P`; its adjoint restricts loads, not point values.
    pub fn constraints(&self) -> &AmrMap<R> {
        &self.constraints
    }
    /// Conservative cumulative preparation payload bound actually admitted.
    pub fn preparation_storage_bound(&self) -> usize {
        self.storage
    }
    /// Physical coordinate of a full node.
    /// # Errors
    /// Rejects an invalid node index.
    pub fn node_point(&self, node: usize) -> Result<[R; D]> {
        self.points
            .get(node)
            .copied()
            .ok_or_else(|| invalid(scioxide_core::argument!("node"), "invalid AMR node"))
    }
    /// Full scalar-node indices of one active cell.
    /// # Errors
    /// Rejects an invalid active-cell index.
    pub fn cell_dofs(&self, cell: usize) -> Result<&[usize]> {
        Ok(&self
            .cells
            .get(cell)
            .ok_or_else(|| invalid(scioxide_core::argument!("cell"), "invalid AMR cell"))?
            .dofs[..self.local])
    }
    /// Pack shared nodal views transactionally. Hanging consistency is checked separately.
    /// # Errors
    /// Rejects foreign epochs, missing/extraneous edges, shapes and nonfinite values.
    pub fn pack_into<V: NumericScalar<Real = R>>(
        &self,
        field: AmrNodalFieldView<'_, V, D>,
        output: &mut [V],
    ) -> Result<()> {
        self.mesh
            .mesh_token()
            .ensure_compatible(field.vertices.mesh_token())?;
        validate(
            field.vertices.values(),
            self.mesh.vertex_count(),
            scioxide_core::argument!("field"),
        )?;
        if output.len() != self.full_dofs() || field.edges.is_some() != (self.order == AmrOrder::P2)
        {
            return Err(invalid(
                if output.len() != self.full_dofs() {
                    scioxide_core::argument!("output")
                } else {
                    scioxide_core::argument!("field")
                },
                "nodal field layout mismatch",
            ));
        }
        if let Some(edges) = &field.edges {
            self.mesh
                .mesh_token()
                .ensure_compatible(edges.mesh_token())?;
            validate(
                edges.values(),
                self.mesh.edge_count(),
                scioxide_core::argument!("field"),
            )?;
        }
        output[..self.mesh.vertex_count()].copy_from_slice(field.vertices.values());
        if let Some(edges) = field.edges {
            output[self.mesh.vertex_count()..].copy_from_slice(edges.values());
        }
        Ok(())
    }
    /// Allocate shared fields from a full nodal vector.
    /// # Errors
    /// Rejects nonfinite values, shapes or allocation failures.
    pub fn field<V: NumericScalar<Real = R>>(&self, values: &[V]) -> Result<AmrNodalField<V, D>> {
        validate(values, self.full_dofs(), scioxide_core::argument!("values"))?;
        let copy = |s: &[V]| -> Result<Vec<V>> {
            let mut v = try_with_capacity(s.len(), scioxide_core::quantity!("field_values"))?;
            v.extend_from_slice(s);
            Ok(v)
        };
        Ok(AmrNodalField {
            vertices: Field::new(&self.mesh, copy(&values[..self.mesh.vertex_count()])?)?,
            edges: if self.order == AmrOrder::P2 {
                Some(Field::new(
                    &self.mesh,
                    copy(&values[self.mesh.vertex_count()..])?,
                )?)
            } else {
                None
            },
        })
    }
    /// Check that a full vector belongs to the hanging-constrained space, using a
    /// 256-epsilon scale-aware row tolerance. This is not an approximation-error test.
    /// # Errors
    /// Rejects shape, nonfinite values or an inconsistent hanging trace.
    pub fn validate_field<V: NumericScalar<Real = R>>(&self, values: &[V]) -> Result<()> {
        validate(values, self.full_dofs(), scioxide_core::argument!("values"))?;
        for (i, &value) in values.iter().enumerate() {
            let mut expected = V::ZERO;
            let mut scale = finite(value.abs(), scioxide_core::NonFiniteOrigin::Intermediate)?;
            for (j, w) in row(&self.constraints.matrix, i) {
                expected = finite(
                    expected + V::from_real(w) * values[self.masters[j]],
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
                scale = finite(
                    scale
                        + w.abs()
                            * finite(
                                values[self.masters[j]].abs(),
                                scioxide_core::NonFiniteOrigin::Intermediate,
                            )?,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
            if finite(
                value - expected,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?
            .abs()
                > R::EPSILON * R::ONE.scalbn(8) * scale.max(R::MIN_POSITIVE)
            {
                return Err(invalid(
                    scioxide_core::argument!("values"),
                    "inconsistent hanging nodal field",
                ));
            }
        }
        Ok(())
    }
    /// Evaluate a full field and its physical gradient at a reference point, without allocation.
    /// # Errors
    /// Rejects shape, invalid cell/reference point or unrepresentable evaluation.
    pub fn evaluate<V: NumericScalar<Real = R>>(
        &self,
        cell: usize,
        reference: [R; D],
        values: &[V],
    ) -> Result<(V, [V; D])> {
        if values.len() != self.full_dofs() {
            return Err(invalid(
                scioxide_core::argument!("values"),
                "incorrect full nodal length",
            ));
        }
        if cell >= self.cells.len() {
            return Err(invalid(
                scioxide_core::argument!("cell"),
                "invalid AMR cell",
            ));
        }
        let sample = self.sample(cell, reference, scioxide_core::argument!("reference"))?;
        let mut value = V::ZERO;
        let mut gradient = [V::ZERO; D];
        for (i, &node) in self.cell_dofs(cell)?.iter().enumerate() {
            value = finite(
                value + V::from_real(sample.values[i]) * values[node],
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
            for (axis, g) in gradient.iter_mut().enumerate() {
                *g = finite(
                    *g + V::from_real(sample.gradients[i][axis]) * values[node],
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        Ok((value, gradient))
    }
    pub(super) fn sample(
        &self,
        cell: usize,
        point: [R; D],
        argument: scioxide_core::Argument,
    ) -> Result<BasisSample<R, D>> {
        let data = self
            .cells
            .get(cell)
            .ok_or_else(|| invalid(argument, "invalid AMR cell"))?;
        let mut sample = self.reference_sample(point, argument)?;
        sample.gradients = physical_gradients(&data.map, &sample, self.local)?;
        Ok(sample)
    }
    pub(super) fn reference_sample(
        &self,
        point: [R; D],
        argument: scioxide_core::Argument,
    ) -> Result<BasisSample<R, D>> {
        let sample = if let Some(basis) = &self.basis {
            basis.sample(point)?
        } else {
            if point
                .iter()
                .any(|x| !x.is_finite() || *x < R::ZERO || *x > R::ONE)
            {
                return Err(invalid(argument, "point outside reference tensor cell"));
            }
            tensor_basis(point)
        };
        Ok(sample)
    }
    /// Constant physical basis Laplacians on straight cells. P1 and Q1 on the
    /// supported affine boxes are harmonic; only P2 needs Hessian contractions.
    pub(super) fn laplacians(&self, cell: usize) -> Result<[R; MAX_NODES]> {
        let data = &self.cells[cell];
        let mut laplacians = [R::ZERO; MAX_NODES];
        if self.order != AmrOrder::P2 {
            return Ok(laplacians);
        }
        if let Some(basis) = &self.basis {
            for (i, h) in basis.hessians().iter().enumerate().take(self.local) {
                for axis in 0..D {
                    for (a, row) in h.iter().enumerate() {
                        for (b, &value) in row.iter().enumerate() {
                            laplacians[i] = finite(
                                laplacians[i]
                                    + value
                                        * data.map.inverse_jacobian[a][axis]
                                        * data.map.inverse_jacobian[b][axis],
                                scioxide_core::NonFiniteOrigin::Intermediate,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(laplacians)
    }
    pub(super) fn physical(&self, cell: usize, reference: [R; D]) -> Result<[R; D]> {
        let m = &self.cells[cell].map;
        let mut point = m.point;
        for (axis, value) in point.iter_mut().enumerate() {
            for (j, r) in reference.iter().enumerate() {
                *value = finite(
                    *value + m.jacobian[axis][j] * *r,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        Ok(point)
    }
    pub(super) fn reference(
        &self,
        cell: usize,
        point: [R; D],
        argument: scioxide_core::Argument,
    ) -> Result<[R; D]> {
        let m = &self.cells[cell].map;
        let mut result = [R::ZERO; D];
        let tolerance = R::EPSILON * R::ONE.scalbn(7) / m.reciprocal_condition;
        for (a, value) in result.iter_mut().enumerate() {
            for (j, x) in point.iter().enumerate() {
                *value = finite(
                    *value
                        + m.inverse_jacobian[a][j]
                            * finite(
                                *x - m.point[j],
                                scioxide_core::NonFiniteOrigin::Intermediate,
                            )?,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
            if *value < -tolerance || *value > R::ONE + tolerance {
                return Err(invalid(argument, "point outside mapped source cell"));
            }
            *value = value.max(R::ZERO).min(R::ONE);
        }
        if self.order != AmrOrder::Q1 {
            let sum = result.iter().copied().fold(R::ZERO, |s, x| s + x);
            if sum > R::ONE + tolerance {
                return Err(invalid(argument, "point outside mapped source simplex"));
            }
            if sum > R::ONE {
                for value in &mut result {
                    *value /= sum;
                }
            }
        }
        Ok(result)
    }
    pub(super) fn on_face(&self, node: usize, face: usize) -> Result<bool> {
        if node < self.mesh.vertex_count() {
            Ok(self.mesh.face_vertices(face)?.any(|v| v == node))
        } else {
            let pair = self.mesh.edge_vertices(node - self.mesh.vertex_count())?;
            Ok(pair.iter().all(|&v| {
                self.mesh
                    .face_vertices(face)
                    .is_ok_and(|mut vertices| vertices.any(|i| i == v))
            }))
        }
    }
    fn build_constraints(&mut self, work: &mut Work<'_, '_, '_>) -> Result<()> {
        let n = self.full_dofs();
        let mut raw: Vec<Option<Vec<(usize, R)>>> =
            work.filled(n, None, scioxide_core::quantity!("raw"))?;
        let mut alternatives: Vec<(usize, Vec<(usize, R)>)> = Vec::new();
        for interface in self.mesh.interfaces() {
            let Some(right_side) = interface.right else {
                continue;
            };
            let right = right_side.cell;
            let left = interface.left_cell;
            let a = self.mesh.level(self.mesh.active_cells()[left])?;
            let b = self.mesh.level(self.mesh.active_cells()[right])?;
            if a == b {
                continue;
            }
            let (coarse, fine, coarse_face, fine_face) = if a < b {
                (left, right, interface.left_face, right_side.face)
            } else {
                (right, left, right_side.face, interface.left_face)
            };
            for &node in self.cell_dofs(fine)? {
                work.tick(self.local)?;
                if !self.on_face(node, fine_face)? || self.cell_dofs(coarse)?.contains(&node) {
                    continue;
                }
                let reference =
                    self.reference(coarse, self.points[node], scioxide_core::argument!("mesh"))?;
                let sample = self.reference_sample(reference, scioxide_core::argument!("mesh"))?;
                let mut trace = work.vector(self.local, scioxide_core::quantity!("trace"))?;
                let mut sum = R::ZERO;
                for (i, &master) in self.cell_dofs(coarse)?.iter().enumerate() {
                    if self.on_face(master, coarse_face)?
                        && sample.values[i].abs() > R::EPSILON * R::ONE.scalbn(6)
                    {
                        trace.push((master, sample.values[i]));
                        sum = finite(
                            sum + sample.values[i],
                            scioxide_core::NonFiniteOrigin::Intermediate,
                        )?;
                    }
                }
                if sum.abs() < R::HALF {
                    return Err(invalid(
                        scioxide_core::argument!("mesh"),
                        "invalid hanging trace partition of unity",
                    ));
                }
                for (_, w) in &mut trace {
                    *w /= sum;
                }
                if raw[node].is_none() {
                    raw[node] = Some(trace);
                } else {
                    work.push(
                        &mut alternatives,
                        (node, trace),
                        scioxide_core::quantity!("alternatives"),
                    )?;
                }
            }
        }
        let mut expanded: Vec<Vec<(usize, R)>> =
            work.vector(n, scioxide_core::quantity!("expanded"))?;
        for _ in 0..n {
            expanded.push(Vec::new());
        }
        self.masters = work.vector(n, scioxide_core::quantity!("masters"))?;
        let mut pending = 0;
        for (node, constraint) in raw.iter().enumerate() {
            if constraint.is_none() {
                expanded[node] = work.vector(1, scioxide_core::quantity!("expanded"))?;
                expanded[node].push((self.masters.len(), R::ONE));
                self.masters.push(node);
            } else {
                pending += 1;
            }
        }
        let mut entries = self.masters.len();
        while pending > 0 {
            let old = pending;
            for node in 0..n {
                work.tick(1)?;
                if !expanded[node].is_empty() {
                    continue;
                }
                let trace = raw[node].as_ref().ok_or_else(|| {
                    invalid(
                        scioxide_core::argument!("mesh"),
                        "missing hanging constraint",
                    )
                })?;
                if trace.iter().any(|(j, _)| expanded[*j].is_empty()) {
                    continue;
                }
                expanded[node] = expand_trace(trace, &expanded, work)?;
                entries = add(
                    entries,
                    expanded[node].len(),
                    scioxide_core::quantity!("entries"),
                )?;
                work.entries(entries)?;
                pending -= 1;
            }
            if old == pending {
                return Err(invalid(
                    scioxide_core::argument!("mesh"),
                    "cyclic hanging-node constraints",
                ));
            }
        }
        for (node, trace) in alternatives {
            let other = expand_trace(&trace, &expanded, work)?;
            let a = &expanded[node];
            let mut difference = R::ZERO;
            for &(j, w) in a {
                let v = other
                    .iter()
                    .find(|&&(k, _)| k == j)
                    .map_or(R::ZERO, |&(_, v)| v);
                difference += (w - v).abs();
            }
            for &(j, w) in &other {
                if !a.iter().any(|&(k, _)| k == j) {
                    difference += w.abs();
                }
            }
            if difference > R::EPSILON * R::ONE.scalbn(10) {
                return Err(invalid(
                    scioxide_core::argument!("mesh"),
                    "inconsistent hanging constraints from incident faces",
                ));
            }
        }
        let mut triplets = work.vector(entries, scioxide_core::quantity!("triplets"))?;
        for (i, trace) in expanded.iter().enumerate() {
            for &(j, w) in trace {
                triplets.push((i, j, w));
            }
        }
        self.constraints = AmrMap::new(
            csr(n, self.masters.len(), &triplets, work)?,
            self.options.execution,
        );
        Ok(())
    }
    /// Cache physical face quadrature and index exterior cells once per space.
    fn prepare_faces(&mut self, work: &mut Work<'_, '_, '_>) -> Result<()> {
        let count = size(
            self.mesh.interfaces().len(),
            self.facets.len(),
            scioxide_core::quantity!("face_samples"),
        )?;
        self.face_samples = work.vector(count, scioxide_core::quantity!("face_samples"))?;
        self.exterior_cells = work.filled(
            self.mesh.face_count(),
            None,
            scioxide_core::quantity!("exterior_cells"),
        )?;
        let argument = scioxide_core::argument!("mesh");
        for interface in self.mesh.interfaces() {
            work.tick(1)?;
            if interface.right.is_none() {
                self.exterior_cells[interface.left_face] = Some(interface.left_cell);
            }
            let cell = interface.left_cell;
            let mut corners = [[R::ZERO; D]; 4];
            let mut count = 0;
            for v in self.mesh.face_vertices(interface.piece_face)? {
                corners[count] = *self.mesh.point(v)?.as_array();
                count += 1;
            }
            if count == 4 {
                corners.swap(2, 3);
            } // canonical cyclic order -> tensor bit order
            let mut midpoint = [R::ZERO; D];
            if self.order == AmrOrder::Q1 {
                let scale = R::ONE / R::try_from_usize_exact(count)?;
                for corner in &corners[..count] {
                    for (value, &x) in midpoint.iter_mut().zip(corner) {
                        *value = finite(
                            *value + x * scale,
                            scioxide_core::NonFiniteOrigin::Intermediate,
                        )?;
                    }
                }
            }
            let reference_normal = if self.order != AmrOrder::Q1 {
                let missing = self.cells[cell].dofs[..D + 1]
                    .iter()
                    .position(|&v| {
                        !self
                            .mesh
                            .face_vertices(interface.left_face)
                            .is_ok_and(|mut it| it.any(|i| i == v))
                    })
                    .ok_or_else(|| invalid(argument, "simplex face lacks opposite vertex"))?;
                facet_normal(missing)
            } else {
                let r = self.reference(cell, midpoint, argument)?;
                let axis = (0..D)
                    .min_by(|&a, &b| {
                        r[a].min(R::ONE - r[a])
                            .partial_cmp(&r[b].min(R::ONE - r[b]))
                            .unwrap_or(core::cmp::Ordering::Equal)
                    })
                    .ok_or_else(|| invalid(argument, "zero dimension"))?;
                let mut normal = [R::ZERO; D];
                normal[axis] = if r[axis] < R::HALF { -R::ONE } else { R::ONE };
                normal
            };
            let (normal, _) = mapped_normal(&self.cells[cell].map, reference_normal)?;
            let factor = if D == 3 && self.order != AmrOrder::Q1 {
                R::ONE.scalbn(1)
            } else {
                R::ONE
            };
            let measure = self.mesh.face_measure(interface.piece_face)?;
            for sample in &self.facets {
                work.tick(size(self.local, D + 1, scioxide_core::quantity!("work"))?)?;
                let mut point = corners[0];
                if self.order != AmrOrder::Q1 {
                    for (j, corner) in corners.iter().enumerate().take(count).skip(1) {
                        for axis in 0..D {
                            point[axis] = finite(
                                point[axis]
                                    + sample.point[j - 1] * (corner[axis] - corners[0][axis]),
                                scioxide_core::NonFiniteOrigin::Intermediate,
                            )?;
                        }
                    }
                } else {
                    point.fill(R::ZERO);
                    for (j, corner) in corners.iter().enumerate().take(count) {
                        let mut w = R::ONE;
                        for axis in 0..D - 1 {
                            w *= if j & (1 << axis) == 0 {
                                R::ONE - sample.point[axis]
                            } else {
                                sample.point[axis]
                            };
                        }
                        for axis in 0..D {
                            point[axis] = finite(
                                point[axis] + corner[axis] * w,
                                scioxide_core::NonFiniteOrigin::Intermediate,
                            )?;
                        }
                    }
                }
                let weight = finite(
                    sample.weight * measure * factor,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
                self.face_samples.push(FaceSample {
                    point,
                    weight,
                    normal,
                });
            }
        }
        Ok(())
    }
    pub(super) fn face_sample(
        &self,
        interface_index: usize,
        q: usize,
        argument: scioxide_core::Argument,
    ) -> Result<FaceSample<R, D>> {
        if interface_index >= self.mesh.interfaces().len() || q >= self.facets.len() {
            return Err(invalid(argument, "invalid AMR face sample"));
        }
        Ok(self.face_samples[interface_index * self.facets.len() + q])
    }
    pub(super) fn exterior_cell(&self, face: usize) -> Result<usize> {
        self.exterior_cells
            .get(face)
            .copied()
            .flatten()
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("face"),
                    "invalid residual exterior face",
                )
            })
    }
}

fn expand_trace<R: RealScalar>(
    trace: &[(usize, R)],
    expanded: &[Vec<(usize, R)>],
    work: &mut Work<'_, '_, '_>,
) -> Result<Vec<(usize, R)>> {
    let count = trace.iter().try_fold(0, |n, (i, _)| {
        add(
            n,
            expanded[*i].len(),
            scioxide_core::quantity!("trace_entries"),
        )
    })?;
    work.entries(count)?;
    work.tick(size(count, count.max(1), scioxide_core::quantity!("work"))?)?;
    let mut result = work.vector(count, scioxide_core::quantity!("expanded_trace"))?;
    for &(i, a) in trace {
        for &(j, b) in &expanded[i] {
            result.push((
                j,
                finite(a * b, scioxide_core::NonFiniteOrigin::Intermediate)?,
            ));
        }
    }
    result.sort_unstable_by_key(|e| e.0);
    let mut kept = 0;
    for i in 0..result.len() {
        if kept > 0 && result[kept - 1].0 == result[i].0 {
            result[kept - 1].1 = finite(
                result[kept - 1].1 + result[i].1,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?;
        } else {
            result[kept] = result[i];
            kept += 1;
        }
    }
    result.truncate(kept);
    result.retain(|&(_, w)| w != R::ZERO);
    Ok(result)
}
fn tensor_basis<R: RealScalar, const D: usize>(point: [R; D]) -> BasisSample<R, D> {
    let mut s = BasisSample {
        values: [R::ZERO; MAX_NODES],
        gradients: [[R::ZERO; D]; MAX_NODES],
    };
    for i in 0..1 << D {
        let factor = |axis: usize| {
            if i & (1 << axis) == 0 {
                R::ONE - point[axis]
            } else {
                point[axis]
            }
        };
        s.values[i] = (0..D).fold(R::ONE, |w, a| w * factor(a));
        for a in 0..D {
            s.gradients[i][a] = (0..D)
                .filter(|&b| b != a)
                .fold(if i & (1 << a) == 0 { -R::ONE } else { R::ONE }, |w, b| {
                    w * factor(b)
                });
        }
    }
    s
}
/// Cell and face reference samples.
type Rules<R, const D: usize> = (Vec<Sample<R, D>>, Vec<Sample<R, D>>);

/// Positive interior simplex cell and face rules of exactness `degree`, the
/// rules of the conforming simplex FEM; interval faces are single points.
fn simplex_rules<R: RealScalar, const D: usize>(
    degree: usize,
    work: &mut Work<'_, '_, '_>,
) -> Result<Rules<R, D>> {
    let options = scioxide_integrate::GeometricOptions::default();
    let samples = |rule: &SimplexRule<R>, work: &mut Work<'_, '_, '_>| {
        work.tick(rule.node_count())?;
        let mut result = work.vector(rule.node_count(), scioxide_core::quantity!("samples"))?;
        result.extend(rule_nodes(rule).map(|(point, weight)| Sample { point, weight }));
        Ok::<_, SciOxideError>(result)
    };
    let volume = samples(&SimplexRule::positive(D, degree, options)?, work)?;
    let facets = if D == 1 {
        let mut point = work.vector(1, scioxide_core::quantity!("samples"))?;
        point.push(Sample {
            point: [R::ZERO; D],
            weight: R::ONE,
        });
        point
    } else {
        samples(&SimplexRule::positive(D - 1, degree, options)?, work)?
    };
    Ok((volume, facets))
}

/// Gauss–Legendre tensor product on the unit cube of `dimension`.
fn tensor_quadrature<R: RealScalar, const D: usize>(
    rule: &GaussianRulePlan<R>,
    dimension: usize,
    work: &mut Work<'_, '_, '_>,
) -> Result<Vec<Sample<R, D>>> {
    let n = rule.nodes().len();
    let count = n.pow(dimension as u32);
    work.tick(count)?;
    let mut result = work.vector(count, scioxide_core::quantity!("samples"))?;
    for mut index in 0..count {
        let mut point = [R::ZERO; D];
        let mut weight = R::ONE;
        for value in point.iter_mut().take(dimension) {
            let j = index % n;
            index /= n;
            *value = (rule.nodes()[j] + R::ONE) * R::HALF;
            weight = weight * rule.weights()[j] * R::HALF;
        }
        result.push(Sample {
            point,
            weight: finite(weight, scioxide_core::NonFiniteOrigin::Intermediate)?,
        });
    }
    Ok(result)
}

/// Reject unsupported dimensions, families, quadrature degrees, geometry
/// cutoffs and periodic identification.
fn validate_space<R: RealScalar, const D: usize>(
    mesh: &AdaptiveMesh<R, D>,
    order: AmrOrder,
    options: &AmrOptions<R>,
) -> Result<()> {
    let simplex = mesh.cell_kind() == AdaptiveCellKind::Simplex;
    let degree = options.quadrature_degree;
    // Consistent mass integrates degree twice the nodal order exactly.
    let (minimum, maximum) = match order {
        AmrOrder::P1 => (2, 7),
        AmrOrder::P2 => (4, 7),
        AmrOrder::Q1 => (2, 31),
    };
    if !(1..=3).contains(&D)
        || (simplex != (order != AmrOrder::Q1))
        || !(minimum..=maximum).contains(&degree)
        || !options.minimum_reciprocal_condition.is_finite()
        || options.minimum_reciprocal_condition <= R::ZERO
        || options.minimum_reciprocal_condition >= R::ONE
    {
        return Err(invalid(
            if !(1..=3).contains(&D) {
                scioxide_core::argument!("mesh")
            } else if simplex != (order != AmrOrder::Q1) {
                scioxide_core::argument!("order")
            } else if !(minimum..=maximum).contains(&degree) {
                scioxide_core::argument!("options.quadrature_degree")
            } else {
                scioxide_core::argument!("options.minimum_reciprocal_condition")
            },
            "unsupported AMR space, quadrature degree or geometry cutoff",
        ));
    }
    if mesh.interfaces().iter().any(|i| i.periodic) {
        return Err(invalid(
            scioxide_core::argument!("mesh"),
            "periodic nodal AMR spaces are unsupported",
        ));
    }
    Ok(())
}

/// Cell and face rules of exactness `degree`, within the quadrature node ceiling.
fn quadrature<R: RealScalar, const D: usize>(
    simplex: bool,
    degree: usize,
    work: &mut Work<'_, '_, '_>,
) -> Result<Rules<R, D>> {
    // Fixed bound for degree <= 2 polynomial objects, rule tables and at most
    // sixteen-node Gaussian preparation (including the eigensolver's arrays).
    let points_per_axis = (degree + 2) / 2;
    work.charge::<u8>(256 * 1024)?;
    work.tick(size(
        points_per_axis.pow(3),
        64,
        scioxide_core::quantity!("work"),
    )?)?;
    let (volume, facets) = if simplex {
        simplex_rules::<R, D>(degree, work)?
    } else {
        let rule = GaussianRulePlan::gauss_legendre(points_per_axis)?;
        (
            tensor_quadrature::<R, D>(&rule, D, work)?,
            tensor_quadrature::<R, D>(&rule, D - 1, work)?,
        )
    };
    super::ceiling(
        scioxide_core::quantity!("amr_quadrature_nodes"),
        volume.len().max(facets.len()),
        work.limits.max_quadrature_nodes,
    )?;
    Ok((volume, facets))
}

type NodalPoints<R, const D: usize> = (Vec<[R; D]>, Vec<([usize; 2], usize)>);

/// Vertex points, then affine edge midpoints and the sorted edge lookup for P2.
fn nodal_points<R: RealScalar, const D: usize>(
    mesh: &AdaptiveMesh<R, D>,
    order: AmrOrder,
    n: usize,
    work: &mut Work<'_, '_, '_>,
) -> Result<NodalPoints<R, D>> {
    let mut points = work.vector(n, scioxide_core::quantity!("points"))?;
    for i in 0..mesh.vertex_count() {
        points.push(*mesh.point(i)?.as_array());
    }
    let mut edges = work.vector(
        if order == AmrOrder::P2 {
            mesh.edge_count()
        } else {
            0
        },
        scioxide_core::quantity!("edges"),
    )?;
    if order == AmrOrder::P2 {
        for i in 0..mesh.edge_count() {
            let mut pair = mesh.edge_vertices(i)?;
            pair.sort_unstable();
            let mut point = [R::ZERO; D];
            for (axis, value) in point.iter_mut().enumerate() {
                *value = crate::spatial::midpoint(points[pair[0]][axis], points[pair[1]][axis])?;
            }
            points.push(point);
            edges.push((pair, i));
        }
        edges.sort_unstable_by_key(|e| e.0);
    }
    Ok((points, edges))
}

/// DOFs and affine reference maps of every cell; forms are filled later.
fn cell_data<R: RealScalar, const D: usize>(
    mesh: &AdaptiveMesh<R, D>,
    order: AmrOrder,
    local: usize,
    points: &[[R; D]],
    edges: &[([usize; 2], usize)],
    options: &AmrOptions<R>,
    work: &mut Work<'_, '_, '_>,
) -> Result<Vec<CellData<R, D>>> {
    let simplex = order != AmrOrder::Q1;
    let linear = ReferenceBasis::<R, D>::new(SimplexFemOrder::P1)?.sample([R::ZERO; D])?;
    let mut cells = work.vector(mesh.cell_count(), scioxide_core::quantity!("cells"))?;
    for cell in 0..mesh.cell_count() {
        work.tick(size(local, local, scioxide_core::quantity!("work"))?)?;
        let mut dofs = [0; MAX_NODES];
        let mut coordinates = [[R::ZERO; D]; 8];
        let mut count = 0;
        for vertex in mesh.cell_vertices(cell)? {
            dofs[count] = vertex;
            coordinates[count] = points[vertex];
            count += 1;
        }
        if order == AmrOrder::P2 {
            let mut k = count;
            for i in 0..count {
                for j in i + 1..count {
                    let mut pair = [dofs[i], dofs[j]];
                    pair.sort_unstable();
                    let index = edges.binary_search_by_key(&pair, |e| e.0).map_err(|_| {
                        invalid(
                            scioxide_core::argument!("mesh"),
                            "missing adaptive cell edge",
                        )
                    })?;
                    dofs[k] = mesh.vertex_count() + edges[index].1;
                    k += 1;
                }
            }
        }
        let shape = if simplex {
            linear
        } else {
            tensor_basis([R::ZERO; D])
        };
        let map = map_coordinates(
            &coordinates[..count],
            options.minimum_reciprocal_condition,
            &shape,
            !simplex,
        )?;
        cells.push(CellData {
            dofs,
            map,
            mass: [R::ZERO; 100],
            stiffness: [R::ZERO; 100],
        });
    }
    Ok(cells)
}
