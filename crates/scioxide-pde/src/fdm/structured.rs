//! Conservative, axis-aligned finite differences on structured meshes in one to three dimensions.

use alloc::vec::Vec;
use core::mem::size_of;

use scioxide_core::backend::{checked_add, checked_mul, try_with_capacity};
use scioxide_core::{
    Dimensions, LinearOperator, NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError,
    Tolerance,
};
use scioxide_sparse::CsrMatrix;
use scioxide_spatial::{
    AxisOrientation, AxisTopology, CellMesh, Extent, FaceTopology, FieldView, Index, MeshToken,
    Vertex, VertexField,
};

use crate::boundary::{
    BoundaryKind, BoundaryPoint, FieldData, FieldPoint, StructuredBoundaries, structured_face_at,
};
use crate::discretization::{
    AssembledOperator, DirichletSite, Elimination, OperatorBuilder, PreparedLoads,
    SpatialDiscretization, SpatialLimits, entry, reserve,
};
use crate::spatial::{midpoint, nonnegative_coefficient, positive_width};
use crate::validation::{finite, invalid};
use scioxide_core::checked_float_product as product;

/// Coefficient provider for a linear system of vertex fields.
///
/// The represented equation is `-div(k grad u) + div(v u) + C u = f`.
/// Diffusion and transport are diagonal in field space; `C` couples fields.
/// Coordinates are real, whereas reaction may be complex. Wave problems supply
/// the squared speed as diffusion.
pub trait StructuredFdmCoefficients<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    /// Nonnegative diffusion along `axis` for `component` at a face center.
    /// # Errors
    /// Callback errors propagate unchanged.
    fn diffusion(&self, axis: usize, component: usize, point: [R; D]) -> Result<R, SciOxideError>;
    /// Signed conservative transport along `axis` for `component`.
    /// # Errors
    /// Callback errors propagate unchanged.
    fn advection(&self, axis: usize, component: usize, point: [R; D]) -> Result<R, SciOxideError> {
        let _ = (axis, component, point);
        Ok(R::ZERO)
    }
    /// Local linear coupling from `column` to `row` at a vertex.
    /// # Errors
    /// Callback errors propagate unchanged.
    fn reaction(&self, row: usize, column: usize, point: [R; D]) -> Result<V, SciOxideError> {
        let _ = (row, column, point);
        Ok(V::ZERO)
    }
}

/// Identically zero coefficients: the tangent of a coefficient-independent parameter.
struct ZeroCoefficients;

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> StructuredFdmCoefficients<R, V, D>
    for ZeroCoefficients
{
    fn diffusion(&self, _: usize, _: usize, _: [R; D]) -> Result<R, SciOxideError> {
        Ok(R::ZERO)
    }
}

/// Prepared vertex-centred conservative finite differences on a structured mesh.
///
/// Serves `D = 1, 2, 3`. Axis zero varies fastest within a field and fields are
/// component-major. Periodic axes identify their seam vertices. Each interior
/// face contributes the conductance `k·area/h` with `k` sampled at the face
/// center; reaction and sources are lumped with dual control volumes, so in one
/// dimension the plan is P1 finite elements with midpoint stiffness and lumped
/// reaction and load. The interior truncation error is second order on uniform
/// or smoothly graded axes and first order on arbitrary nonuniform axes; the
/// boundary relations are second order.
///
/// Boundary conditions apply per exterior face and component. Natural data is
/// the outward conormal flux and Robin data `n·k∇u + αu = g`; a boundary vertex
/// accumulates `area / 2^(D-1)` from every incident exterior face, and advective
/// outflow `v·n` leaves through natural and Robin faces. Dirichlet faces fix their
/// vertices (taking precedence at shared vertices) with symmetric elimination, so
/// pure diffusion assembles a symmetric matrix.
/// Application uses prepared rows and ordinary floating-point rounding, with
/// range-safe replay of exceptional products. Errors leave output unchanged.
///
/// ```
/// use scioxide_pde::{EllipticProblem, FieldData, StructuredFdmPlan};
/// let plan = StructuredFdmPlan::dirichlet_laplacian([0.0], [1.0], [9])?;
/// let u = EllipticProblem::new(&plan, &FieldData::constant(2.0)).solve()?;
/// assert!((u[4] - 0.25_f64).abs() < 1e-12);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone)]
pub struct StructuredFdmPlan<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    mesh: CellMesh<R, D>,
    extent: Extent<D>,
    points: Vec<[R; D]>,
    volumes: Vec<R>,
    boundaries: StructuredBoundaries<R, V, D>,
    operator: AssembledOperator<R, V>,
    loads: PreparedLoads<R, V, D>,
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> StructuredFdmPlan<R, V, D> {
    /// Prepare with default numerical settings and no resource ceiling.
    ///
    /// # Errors
    /// Returns the geometry, coefficient, allocation and configuration errors of [`Self::with_options`].
    pub fn new(
        mesh: CellMesh<R, D>,
        coefficients: &dyn StructuredFdmCoefficients<R, V, D>,
        boundaries: &StructuredBoundaries<R, V, D>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(mesh, coefficients, boundaries, SpatialLimits::default())
    }

    /// Sample coefficients and prepare the bounded structured stencil.
    ///
    /// Components come from `boundaries`.
    ///
    /// # Errors
    /// Rejects foreign boundaries, decreasing/collapsed axes, non-finite
    /// coefficients, negative diffusion, unrepresentable entries (overflow),
    /// resource-limit overflow, and failed allocations. Coefficient callbacks
    /// may be invoked more than once at the same geometric point.
    pub fn with_options(
        mesh: CellMesh<R, D>,
        coefficients: &dyn StructuredFdmCoefficients<R, V, D>,
        boundaries: &StructuredBoundaries<R, V, D>,
        limits: SpatialLimits,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            mesh,
            coefficients,
            boundaries,
            limits,
            Elimination::Symmetric,
        )
    }

    /// Exact tangent `∂A/∂p` of the free residual rows for a signed derivative
    /// coefficient field and Robin coefficient derivatives (one per condition).
    /// Assembly is linear in the sampled coefficients, so re-assembly is exact.
    pub(crate) fn tangent_operator(
        &self,
        coefficients: Option<&dyn StructuredFdmCoefficients<R, V, D>>,
        robin: &[V],
    ) -> Result<CsrMatrix<V>, SciOxideError> {
        let boundaries = self.boundaries.with_robin_coefficients(robin)?;
        let plan = Self::build(
            self.mesh.clone(),
            coefficients.unwrap_or(&ZeroCoefficients),
            &boundaries,
            SpatialLimits::new(usize::MAX, usize::MAX, usize::MAX),
            Elimination::Tangent,
        )?;
        Ok(plan.operator.matrix)
    }

    fn build(
        mesh: CellMesh<R, D>,
        coefficients: &dyn StructuredFdmCoefficients<R, V, D>,
        boundaries: &StructuredBoundaries<R, V, D>,
        limits: SpatialLimits,
        elimination: Elimination,
    ) -> Result<Self, SciOxideError> {
        boundaries
            .mesh_token()
            .ensure_compatible(mesh.mesh_token())?;
        let components = boundaries.components();
        let mut periodic = [false; D];
        let mut shape = [0_usize; D];
        let mut coordinate_count = 0_usize;
        for axis in 0..D {
            let grid_axis = mesh.grid().axis(axis)?;
            if grid_axis.orientation() != AxisOrientation::Increasing {
                return Err(invalid(
                    scioxide_core::argument!("mesh"),
                    "coordinates must increase on every axis",
                ));
            }
            periodic[axis] = mesh.grid().axis_topology(axis)? == AxisTopology::Periodic;
            shape[axis] = grid_axis.len() - usize::from(periodic[axis]);
            coordinate_count = checked_add(
                coordinate_count,
                grid_axis.len(),
                scioxide_core::quantity!("coordinates"),
            )?;
        }
        let extent = Extent::new(shape)?;
        let nodes = extent.element_count()?;
        let unknowns = checked_mul(nodes, components, scioxide_core::quantity!("unknowns"))?;
        let entry_bound = checked_add(
            checked_mul(
                checked_mul(unknowns, D, scioxide_core::quantity!("entry_bound"))?,
                4,
                scioxide_core::quantity!("entry_bound"),
            )?,
            checked_mul(
                unknowns,
                components + 2,
                scioxide_core::quantity!("entry_bound"),
            )?,
            scioxide_core::quantity!("entry_bound"),
        )?;
        let triplet = 2 * size_of::<usize>() + size_of::<V>();
        let mut storage_bound = 0_usize;
        for (count, bytes) in [
            (nodes, size_of::<[R; D]>() + size_of::<R>()),
            (entry_bound, 2 * triplet + size_of::<u32>()),
            (
                unknowns,
                size_of::<R>() + size_of::<bool>() + size_of::<FieldPoint<R, D>>() + triplet,
            ),
            (coordinate_count, size_of::<R>() * 2),
        ] {
            storage_bound = checked_add(
                storage_bound,
                checked_mul(count, bytes, scioxide_core::quantity!("storage_bound"))?,
                scioxide_core::quantity!("storage_bound"),
            )?;
        }
        limits.admit(unknowns, entry_bound, storage_bound)?;

        let mut axis_nodes = try_with_capacity(D, scioxide_core::quantity!("structured_fdm_axes"))?;
        for axis in 0..D {
            let grid_axis = mesh.grid().axis(axis)?;
            let mut values = try_with_capacity(
                grid_axis.len(),
                scioxide_core::quantity!("structured_fdm_axis_coordinates"),
            )?;
            for coordinate in grid_axis.coordinates() {
                values.push(coordinate?);
            }
            for pair in values.windows(2) {
                positive_width(pair[1], pair[0], scioxide_core::argument!("mesh"))?;
            }
            axis_nodes.push(values);
        }
        let widths = dual_widths(&axis_nodes, &shape, &periodic)?;
        let mut points = reserve(nodes)?;
        let mut volumes = reserve(nodes)?;
        for node in 0..nodes {
            let index = extent.delinearize(node)?;
            let point = core::array::from_fn(|axis| axis_nodes[axis][index.as_array()[axis]]);
            let factors: [R; D] = core::array::from_fn(|axis| widths[axis][index.as_array()[axis]]);
            let volume = scioxide_core::checked_float_product_factors(&factors)?;
            if volume <= R::ZERO {
                return Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Underflow,
                    "dual volume is unrepresentable",
                ));
            }
            points.push(point);
            volumes.push(volume);
        }

        let faces = exterior_faces(&mesh, &axis_nodes, &extent, &periodic)?;
        let mut fixed = scioxide_core::backend::try_filled(
            unknowns,
            false,
            scioxide_core::quantity!("fdm_dirichlet_unknowns"),
        )?;
        for face in &faces {
            for component in 0..components {
                if matches!(
                    boundaries.kind_at(face.id, component)?,
                    BoundaryKind::Dirichlet
                ) {
                    for &node in &face.vertices[..face.vertex_count] {
                        fixed[component * nodes + node] = true;
                    }
                }
            }
        }
        let mut builder = OperatorBuilder::with_elimination(
            scioxide_core::backend::try_copy_slice(
                &fixed,
                scioxide_core::quantity!("fdm_dirichlet_unknowns"),
            )?,
            entry_bound,
            elimination,
        )?;
        let mut transport = false;
        for (node, &point) in points.iter().enumerate() {
            let index = *extent.delinearize(node)?.as_array();
            for axis in 0..D {
                if index[axis] + 1 == shape[axis] && !periodic[axis] {
                    continue;
                }
                let mut neighbor = index;
                neighbor[axis] = (index[axis] + 1) % shape[axis];
                let right = extent.linearize(Index::new(neighbor)?)?;
                if right == node {
                    continue;
                }
                let start = index[axis];
                let h = positive_width(
                    axis_nodes[axis][start + 1],
                    axis_nodes[axis][start],
                    scioxide_core::argument!("mesh"),
                )?;
                let mut face = point;
                face[axis] = midpoint(axis_nodes[axis][start], axis_nodes[axis][start + 1])?;
                let mut area = R::ONE;
                for other in 0..D {
                    if other != axis {
                        area = product(area, widths[other][index[other]])?;
                    }
                }
                for component in 0..components {
                    let k = coefficients.diffusion(axis, component, face)?;
                    let k = match elimination {
                        Elimination::Symmetric | Elimination::Lift => nonnegative_coefficient(
                            k,
                            scioxide_core::argument!("coefficients.diffusion"),
                        )?,
                        Elimination::Tangent => crate::spatial::finite_callback(k)?,
                    };
                    let v = crate::spatial::finite_callback(
                        coefficients.advection(axis, component, face)?,
                    )?;
                    transport |= v != R::ZERO;
                    let g = V::from_real(scioxide_core::checked_product_quotient(k, area, h)?);
                    let a = V::from_real(scioxide_core::checked_product_quotient(v, area, R::TWO)?);
                    let (left, right) = (component * nodes + node, component * nodes + right);
                    builder.push(left, left, g + a)?;
                    builder.push(left, right, -g + a)?;
                    builder.push(right, left, -g - a)?;
                    builder.push(right, right, g - a)?;
                }
            }
        }
        let mut reactive = false;
        for (node, &point) in points.iter().enumerate() {
            for row in 0..components {
                for column in 0..components {
                    let value = finite(
                        coefficients.reaction(row, column, point)?,
                        NonFiniteOrigin::Callback,
                    )?;
                    reactive |= value != V::ZERO;
                    builder.push(
                        row * nodes + node,
                        column * nodes + node,
                        entry(value * V::from_real(volumes[node]))?,
                    )?;
                }
            }
        }

        let mut loads = PreparedLoads::new(unknowns)?;
        for face in &faces {
            for component in 0..components {
                let condition = boundaries.slot(face.id, component)?;
                let kind = boundaries.kind_at(face.id, component)?;
                for &node in &face.vertices[..face.vertex_count] {
                    let dof = component * nodes + node;
                    let site = BoundaryPoint {
                        point: points[node],
                        normal: face.normal,
                        component,
                    };
                    if let BoundaryKind::Dirichlet = kind {
                        loads.push_dirichlet(DirichletSite {
                            dof,
                            site,
                            condition,
                            rank: (condition, face.id),
                        })?;
                        continue;
                    }
                    if fixed[dof] {
                        continue;
                    }
                    loads.push_data(site, condition, [(dof, face.share)])?;
                    let v = crate::spatial::finite_callback(coefficients.advection(
                        face.axis,
                        component,
                        points[node],
                    )?)?;
                    transport |= v != R::ZERO;
                    let outflow = V::from_real(entry(face.outward * v * face.share)?);
                    builder.push(dof, dof, outflow)?;
                    if let BoundaryKind::Robin { alpha } = kind {
                        builder.push(dof, dof, entry(alpha * V::from_real(face.share))?)?;
                    }
                }
            }
        }
        for component in 0..components {
            for (node, &point) in points.iter().enumerate() {
                loads.push_source(
                    FieldPoint { point, component },
                    [(component * nodes + node, volumes[node])],
                )?;
            }
        }
        let (matrix, lift, constrained) = builder.finish()?;
        let mut mass = reserve(unknowns)?;
        for _ in 0..components {
            mass.extend_from_slice(&volumes);
        }
        let constant_modes = !transport && !reactive && boundaries.is_natural_everywhere();
        let loads = loads.finish((lift.nnz() > 0).then_some(lift));
        Ok(Self {
            mesh,
            extent,
            points,
            volumes,
            boundaries: boundaries.clone(),
            operator: AssembledOperator {
                bounds: crate::kernels::RowBounds::new(matrix.values(), matrix.row_ptr()),
                compact_columns: crate::kernels::compact_columns(&matrix)?,
                matrix,
                constrained,
                mass,
                components,
                constant_modes,
                entry_bound,
            },
            loads,
        })
    }

    /// Exact mesh whose vertices identify the source and output fields.
    pub const fn mesh(&self) -> &CellMesh<R, D> {
        &self.mesh
    }

    /// Bound boundary conditions, including their data.
    pub const fn boundaries(&self) -> &StructuredBoundaries<R, V, D> {
        &self.boundaries
    }

    /// Conservative upper bound on COO triplets emitted by assembly.
    pub const fn entry_bound(&self) -> usize {
        self.operator.entry_bound
    }

    /// Dual control-volume weights in axis-zero-fast unique-vertex order.
    pub fn volumes(&self) -> &[R] {
        &self.volumes
    }

    /// Coordinate of one unique vertex.
    ///
    /// # Errors
    /// Returns an out-of-bounds error for an invalid unique-vertex position.
    pub fn node_point(&self, node: usize) -> Result<[R; D], SciOxideError> {
        self.points.get(node).copied().ok_or_else(|| {
            SciOxideError::out_of_bounds(
                scioxide_core::quantity!("structured_fdm_vertex"),
                node,
                self.points.len(),
            )
        })
    }

    /// Check each component's integrated load against its known constant mode.
    ///
    /// Each component's load sum must vanish within `tolerance` relative to
    /// its absolute load sum. This does not establish uniqueness or conditioning.
    /// # Errors
    /// Returns a dimension or domain error for an incompatible load, or an input
    /// error when no constant mode is structurally known.
    pub fn check_constant_mode_compatibility(
        &self,
        rhs: &[V],
        tolerance: Tolerance<R>,
    ) -> Result<(), SciOxideError> {
        if rhs.len() != self.operator.unknowns() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(self.operator.unknowns()),
                Dimensions::one(rhs.len()),
            ));
        }
        self.operator
            .check_constant_mode_compatibility(rhs, tolerance)
    }

    fn canonical_full_index(&self, full: [usize; D]) -> Result<usize, SciOxideError> {
        let mut reduced = full;
        for (axis, coordinate) in reduced.iter_mut().enumerate() {
            if *coordinate == self.extent.as_array()[axis] {
                *coordinate = 0;
            }
        }
        self.extent.linearize(Index::new(reduced)?)
    }

    /// Pack full-mesh component fields into unique-vertex unknowns.
    ///
    /// Periodic seam copies must agree within `64·ε` of the field scale.
    /// # Errors
    /// Rejects wrong field identities, shape, seam values, and non-finite data.
    pub fn pack_fields(
        &self,
        fields: &[FieldView<'_, V, Vertex, D>],
    ) -> Result<Vec<V>, SciOxideError> {
        let components = self.operator.components;
        if fields.len() != components {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(components),
                Dimensions::one(fields.len()),
            ));
        }
        let full_extent = self.mesh.vertex_extent();
        let full_count = full_extent.element_count()?;
        let nodes = self.points.len();
        let mut packed = reserve(self.operator.unknowns())?;
        for field in fields {
            self.mesh
                .mesh_token()
                .ensure_compatible(field.mesh_token())?;
            if field.len() != full_count {
                return Err(SciOxideError::dimension_mismatch(
                    Dimensions::one(full_count),
                    Dimensions::one(field.len()),
                ));
            }
            let scale = field
                .values()
                .iter()
                .fold(R::ONE, |largest, value| largest.max(value.abs()));
            let seam_tolerance = R::EPSILON * R::try_from_usize_exact(64)? * scale;
            for full in 0..full_count {
                let index = *full_extent.delinearize(full)?.as_array();
                let canonical = self.canonical_full_index(index)?;
                let canonical_full = full_extent
                    .linearize(Index::new(*self.extent.delinearize(canonical)?.as_array())?)?;
                let value = finite(field.values()[full], NonFiniteOrigin::Input)?;
                if (value - field.values()[canonical_full]).abs() > seam_tolerance {
                    return Err(invalid(
                        scioxide_core::argument!("fields"),
                        "periodic seam field values disagree",
                    ));
                }
            }
            for node in 0..nodes {
                let full = full_extent.linearize(self.extent.delinearize(node)?)?;
                packed.push(field.values()[full]);
            }
        }
        Ok(packed)
    }

    /// Apply to borrowed full-mesh fields and return owned full-mesh fields.
    /// Periodic seam values are restored in the returned fields.
    /// # Errors
    /// Propagates shape, mesh identity, numerical, and allocation errors.
    pub fn apply_fields(
        &self,
        fields: &[FieldView<'_, V, Vertex, D>],
    ) -> Result<Vec<VertexField<V, D>>, SciOxideError> {
        let input = self.pack_fields(fields)?;
        let mut output = crate::discretization::zeros(input.len())?;
        let mut workspace = crate::discretization::zeros(input.len())?;
        self.operator
            .apply(&input, &mut output, &mut workspace, None)?;
        self.field(&output)
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> LinearOperator<V>
    for StructuredFdmPlan<R, V, D>
{
    fn rows(&self) -> usize {
        self.operator.unknowns()
    }
    fn cols(&self) -> usize {
        self.operator.unknowns()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.operator.unknowns())
    }
    fn apply_into(
        &self,
        input: &[V],
        output: &mut [V],
        workspace: &mut [V],
    ) -> Result<(), SciOxideError> {
        self.operator.apply(input, output, workspace, None)
    }
    fn apply_into_scoped(
        &self,
        input: &[V],
        output: &mut [V],
        workspace: &mut [V],
        context: &mut scioxide_core::OperatorContext<'_, '_, '_, V>,
    ) -> Result<scioxide_core::Invocation<()>, SciOxideError> {
        // Deterministic sequential rows with the caller's cancellation and deadline.
        self.operator
            .apply(input, output, workspace, Some(context.execution()))
            .map(scioxide_core::Invocation::Evaluated)
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> SpatialDiscretization<R, V, D>
    for StructuredFdmPlan<R, V, D>
{
    type Field = Vec<VertexField<V, D>>;

    fn mesh_token(&self) -> &MeshToken {
        self.mesh.mesh_token()
    }
    fn components(&self) -> usize {
        self.operator.components
    }
    fn dof_point(&self, unknown: usize) -> Result<FieldPoint<R, D>, SciOxideError> {
        let nodes = self.points.len();
        if unknown >= self.operator.unknowns() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("structured_fdm_unknown"),
                unknown,
                self.operator.unknowns(),
            ));
        }
        Ok(FieldPoint {
            point: self.points[unknown % nodes],
            component: unknown / nodes,
        })
    }
    fn constrained(&self) -> &[bool] {
        &self.operator.constrained
    }
    fn lumped_mass(&self) -> &[R] {
        &self.operator.mass
    }
    fn has_constant_modes(&self) -> bool {
        self.operator.constant_modes
    }
    fn assemble_csr(&self) -> Result<CsrMatrix<V>, SciOxideError> {
        Ok(self.operator.matrix.clone())
    }
    fn load_time_dependence(&self) -> crate::TimeDependence {
        self.boundaries.time_dependence()
    }
    fn load_into(
        &self,
        source: &FieldData<R, V, D>,
        time: R,
        out: &mut [V],
        scratch: &mut [V],
    ) -> Result<(), SciOxideError> {
        self.loads
            .load_transaction(&self.boundaries, source, time, out, scratch)
    }
    fn field(&self, values: &[V]) -> Result<Self::Field, SciOxideError> {
        if values.len() != self.operator.unknowns() {
            return Err(SciOxideError::dimension_mismatch(
                Dimensions::one(self.operator.unknowns()),
                Dimensions::one(values.len()),
            ));
        }
        let nodes = self.points.len();
        let full_extent = self.mesh.vertex_extent();
        let full_count = full_extent.element_count()?;
        let mut results = reserve(self.operator.components)?;
        for component in 0..self.operator.components {
            let mut field = reserve(full_count)?;
            for full in 0..full_count {
                let node = self.canonical_full_index(*full_extent.delinearize(full)?.as_array())?;
                field.push(values[component * nodes + node]);
            }
            results.push(VertexField::new(&self.mesh, field)?);
        }
        Ok(results)
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> crate::adjoint::Sealed
    for StructuredFdmPlan<R, V, D>
{
    fn preparation(&self) -> &alloc::sync::Arc<()> {
        &self.loads.preparation
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>
    crate::DifferentiableDiscretization<R, V, D> for StructuredFdmPlan<R, V, D>
{
    type Coefficients = dyn StructuredFdmCoefficients<R, V, D>;

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

/// Dual widths of the unique vertices on every axis.
fn dual_widths<R: RealScalar, const D: usize>(
    axis_nodes: &[Vec<R>],
    shape: &[usize; D],
    periodic: &[bool; D],
) -> Result<Vec<Vec<R>>, SciOxideError> {
    let mut widths = reserve(D)?;
    for axis in 0..D {
        let values = &axis_nodes[axis];
        let len = shape[axis];
        let width_at = |left: usize| {
            positive_width(
                values[left + 1],
                values[left],
                scioxide_core::argument!("mesh"),
            )
        };
        let mut dual = reserve(len)?;
        for index in 0..len {
            let width = if periodic[axis] && index == 0 {
                midpoint(width_at(0)?, width_at(values.len() - 2)?)?
            } else if index == 0 {
                width_at(0)? * R::HALF
            } else if index + 1 == len && !periodic[axis] {
                width_at(index - 1)? * R::HALF
            } else {
                midpoint(width_at(index - 1)?, width_at(index)?)?
            };
            if !width.is_finite() || width <= R::ZERO {
                return Err(invalid(
                    scioxide_core::argument!("mesh"),
                    "dual width is not finite and positive",
                ));
            }
            dual.push(width);
        }
        widths.push(dual);
    }
    Ok(widths)
}

/// One exterior face with its unique vertices and per-vertex area share.
struct ExteriorFace<R, const D: usize> {
    id: usize,
    axis: usize,
    normal: [R; D],
    outward: R,
    share: R,
    vertices: [usize; 4],
    vertex_count: usize,
}

fn exterior_faces<R: RealScalar, const D: usize>(
    mesh: &CellMesh<R, D>,
    axis_nodes: &[Vec<R>],
    extent: &Extent<D>,
    periodic: &[bool; D],
) -> Result<Vec<ExteriorFace<R, D>>, SciOxideError> {
    let mut faces = Vec::new();
    for id in 0..mesh.topology_sizes().faces() {
        let face = structured_face_at(mesh, id)?;
        let FaceTopology::Exterior { side } = mesh.face_topology(face)? else {
            continue;
        };
        let axis = face.normal_axis();
        let index = *face.index().as_array();
        let outward = match side {
            scioxide_spatial::AxisSide::Start => -R::ONE,
            scioxide_spatial::AxisSide::End => R::ONE,
        };
        let mut normal = [R::ZERO; D];
        normal[axis] = outward;
        let mut area = R::ONE;
        for other in (0..D).filter(|&other| other != axis) {
            area = product(
                area,
                positive_width(
                    axis_nodes[other][index[other] + 1],
                    axis_nodes[other][index[other]],
                    scioxide_core::argument!("mesh"),
                )?,
            )?;
        }
        let share = (1..D).fold(area, |share, _| share * R::HALF);
        let mut vertices = [0; 4];
        let mut vertex_count = 0;
        for corner in 0..1_usize << (D - 1) {
            let mut vertex = index;
            for (bit, other) in (0..D).filter(|&other| other != axis).enumerate() {
                vertex[other] += (corner >> bit) & 1;
            }
            for (axis, coordinate) in vertex.iter_mut().enumerate() {
                if periodic[axis] && *coordinate == extent.as_array()[axis] {
                    *coordinate = 0;
                }
            }
            vertices[vertex_count] = extent.linearize(Index::new(vertex)?)?;
            vertex_count += 1;
        }
        faces.try_reserve(1).map_err(|_| {
            SciOxideError::allocation_failure(
                scioxide_core::quantity!("exterior_faces"),
                faces.len() + 1,
            )
        })?;
        faces.push(ExteriorFace {
            id,
            axis,
            normal,
            outward,
            share,
            vertices,
            vertex_count,
        });
    }
    Ok(faces)
}

#[cfg(test)]
#[path = "structured_tests.rs"]
mod tests;

impl<R: RealScalar + NumericScalar<Real = R>, const D: usize> StructuredFdmPlan<R, R, D> {
    /// Prepare `A = −Δ` on `[lower, upper]` with homogeneous Dirichlet boundaries.
    /// Uniform differences are second order; node `i₀ + n₀ i₁ + n₀ n₁ i₂` has index
    /// `(i₀, i₁, i₂)`. Preparation takes `O(D n)` work and storage for `n` nodes.
    /// # Errors
    /// Rejects dimensions outside 1–3, fewer than two nodes per axis, non-finite or
    /// non-increasing bounds, overflow and failed allocation.
    /// # References
    /// R. J. LeVeque, *Finite Difference Methods for Ordinary and Partial Differential Equations*, 2007, chapter 3.
    /// ```
    /// use scioxide_pde::{EllipticProblem, FieldData, FieldPoint, StructuredFdmPlan};
    /// let fdm = StructuredFdmPlan::dirichlet_laplacian([0.0; 2], [1.0; 2], [101; 2])?;
    /// let source = FieldData::time_invariant(|p: FieldPoint<f64, 2>| {
    ///     let pi = core::f64::consts::PI;
    ///     Ok(2.0 * pi * pi * (pi * p.point[0]).sin() * (pi * p.point[1]).sin())
    /// });
    /// let u = EllipticProblem::new(&fdm, &source).solve()?;
    /// assert!((u[50 + 101 * 50] - 1.0).abs() < 1e-4);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn dirichlet_laplacian(
        lower: [R; D],
        upper: [R; D],
        nodes: [usize; D],
    ) -> Result<Self, SciOxideError> {
        let mut axes = try_with_capacity(D, scioxide_core::quantity!("axes"))?;
        for i in 0..D {
            axes.push(scioxide_spatial::CoordinateAxis::uniform(
                lower[i], upper[i], nodes[i],
            )?);
        }
        let axes = axes
            .try_into()
            .map_err(|_| invalid(scioxide_core::argument!("nodes"), "axis count mismatch"))?;
        let mesh = CellMesh::new(scioxide_spatial::StructuredGrid::new(
            axes,
            [AxisTopology::Bounded; D],
        )?)?;
        let boundaries = StructuredBoundaries::uniform(
            &mesh,
            1,
            BoundaryKind::Dirichlet,
            crate::BoundaryData::zero(),
        )?;
        Self::new(mesh, &crate::ConstantDiffusion::new(R::ONE), &boundaries)
    }
}
