//! Stable mixed incompressible forms on shared P2/P1 simplex spaces.
use super::space::{Bytes, CellSamples, Sample};
use super::*;
use crate::EllipticGauge;
use scioxide_core::{AdjointOperator, BlockOperator, BlockVectorView, SolveScope};
use scioxide_linalg::{BorderedSystem, CompatibilityPolicy};

mod diagnostics;
mod linear;
mod nonlinear;
mod preconditioner;
mod solve;
#[cfg(test)]
mod tests;
mod time;
pub use linear::IncompressibleOperator;
pub use nonlinear::{NavierStokesLinearizationWorkspace, NavierStokesPlan};
pub use preconditioner::{
    IncompressiblePreconditioner, IncompressiblePreconditionerOptions, IncompressibleSaddle,
    IncompressibleVelocityInverse,
};
pub use solve::{IncompressibleSolveOptions, NavierStokesWorkspace, StokesWorkspace};
pub use time::{IncompressibleSystem, IncompressibleTimeWorkspace};

/// Configure P2/P1 preparation and automatic or explicit pressure gauges.
#[derive(Debug, Clone)]
#[must_use]
pub struct IncompressibleOptions<R: RealScalar, V: NumericScalar<Real = R>> {
    fem: SimplexFemOptions<R>,
    gauges: Vec<EllipticGauge<V>>,
    compatibility: Tolerance<R>,
}
impl<R: RealScalar, V: NumericScalar<Real = R>> Default for IncompressibleOptions<R, V> {
    fn default() -> Self {
        Self::new()
    }
}
impl<R: RealScalar, V: NumericScalar<Real = R>> IncompressibleOptions<R, V> {
    /// Select default FEM options and automatic pressure gauges in constant time.
    /// Closed components use zero weighted-mean pressure; traction-anchored
    /// components need no gauge. Compatibility uses square-root epsilon.
    pub fn new() -> Self {
        Self {
            fem: SimplexFemOptions::default(),
            gauges: Vec::new(),
            compatibility: Tolerance::from_sqrt_epsilon(),
        }
    }
    /// Select explicit per-component pressure gauges; an empty list selects automatic gauges.
    pub fn with_gauges(mut self, gauges: Vec<EllipticGauge<V>>) -> Self {
        self.gauges = gauges;
        self
    }
    /// Set quadrature, conditioning, scheduling and preparation limits.
    pub fn with_fem(mut self, fem: SimplexFemOptions<R>) -> Self {
        self.fem = fem;
        self
    }
    /// Independent pressure-nullspace and prescribed-flux compatibility tolerance.
    pub const fn with_compatibility(mut self, tolerance: Tolerance<R>) -> Self {
        self.compatibility = tolerance;
        self
    }
    /// Shared FEM quadrature, conditioning, execution and limits.
    pub const fn fem(&self) -> &SimplexFemOptions<R> {
        &self.fem
    }
    /// Per-component pressure gauges.
    pub fn gauges(&self) -> &[EllipticGauge<V>] {
        &self.gauges
    }
    /// Compatibility tolerance.
    pub const fn compatibility(&self) -> Tolerance<R> {
        self.compatibility
    }
}
/// Owned component-major P2 velocity and compression-positive P1 pressure.
#[derive(Debug, Clone)]
pub struct IncompressibleField<V, const D: usize> {
    velocity: SimplexFemField<V, D>,
    pressure: VertexField<V, D>,
}
impl<V, const D: usize> IncompressibleField<V, D> {
    /// Shared vertex and edge velocity fields.
    pub const fn velocity(&self) -> &SimplexFemField<V, D> {
        &self.velocity
    }
    /// Pressure at shared vertices.
    pub const fn pressure(&self) -> &VertexField<V, D> {
        &self.pressure
    }
    /// Separate velocity and pressure.
    pub fn into_parts(self) -> (SimplexFemField<V, D>, VertexField<V, D>) {
        (self.velocity, self.pressure)
    }
}
/// Borrowed fields on the exact prepared mesh revision.
#[derive(Debug, Clone, Copy)]
pub struct IncompressibleFieldView<'a, 'b, V, const D: usize> {
    velocity: SimplexFemFieldView<'a, 'b, V, D>,
    pressure: FieldView<'b, V, Vertex, D>,
}
impl<'a, 'b, V, const D: usize> IncompressibleFieldView<'a, 'b, V, D> {
    /// Component-major velocity views and compression-positive vertex pressure.
    pub const fn new(
        velocity: SimplexFemFieldView<'a, 'b, V, D>,
        pressure: FieldView<'b, V, Vertex, D>,
    ) -> Self {
        Self { velocity, pressure }
    }
    /// Velocity views.
    pub const fn velocity(&self) -> &SimplexFemFieldView<'a, 'b, V, D> {
        &self.velocity
    }
    /// Pressure view.
    pub const fn pressure(&self) -> &FieldView<'b, V, Vertex, D> {
        &self.pressure
    }
}
/// Independently sampled physical diagnostics, not error certificates.
#[derive(Clone, Copy, Debug)]
pub struct IncompressibleDiagnostics<R: RealScalar> {
    pub(super) divergence_l2: R,
    pub(super) integrated_divergence: R,
    pub(super) boundary_flux: R,
    pub(super) weak_divergence: R,
    pub(super) gauge_defect: R,
    pub(super) reynolds: R,
    pub(super) courant: R,
    pub(super) diffusion_number: R,
}
impl<R: RealScalar> IncompressibleDiagnostics<R> {
    /// L2 norm of the pointwise divergence sampled by volume quadrature.
    pub const fn divergence_l2(&self) -> R {
        self.divergence_l2
    }
    /// Magnitude of integrated divergence.
    pub const fn integrated_divergence(&self) -> R {
        self.integrated_divergence
    }
    /// Magnitude of total outward velocity flux.
    pub const fn boundary_flux(&self) -> R {
        self.boundary_flux
    }
    /// Norm of discrete pressure-test equations, before gauge augmentation.
    pub const fn weak_divergence(&self) -> R {
        self.weak_divergence
    }
    /// Maximum pressure constraint defect.
    pub const fn gauge_defect(&self) -> R {
        self.gauge_defect
    }
    /// Sampled maximum speed times supplied length divided by kinematic viscosity.
    pub const fn reynolds(&self) -> R {
        self.reynolds
    }
    /// Maximum sampled speed times supplied step divided by local cell length.
    pub const fn courant(&self) -> R {
        self.courant
    }
    /// Maximum kinematic viscosity times step divided by squared cell length.
    pub const fn diffusion_number(&self) -> R {
        self.diffusion_number
    }
}

/// Prepared constant-density, constant-viscosity Stokes equations.
///
/// Unknowns are D component-major P2 velocity blocks, P1 vertex pressure, then one explicit
/// gauge multiplier per unanchored connected component. Shared boundary nodes
/// follow [`crate::TaggedBoundaries`]' conflict policy, including discontinuous
/// corner values. [`Self::solve`] supplies library saddle preconditioning.
#[derive(Clone)]
pub struct StokesPlan<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    space: SimplexSpace<R, D>,
    density: R,
    viscosity: R,
    samples: CellSamples<R, D>,
    matrices: Vec<R>,
    masses: Vec<R>,
    pressure_masses: Vec<R>,
    component: Vec<usize>,
    gauge_index: Vec<Option<usize>>,
    gauge_weights: Vec<R>,
    gauge_targets: Vec<V>,
    prescribed: Vec<Option<V>>,
    constrained: Vec<bool>,
    assembly: AssemblyPattern,
    assembly_extra: Vec<V>,
    load: Vec<V>,
    rhs: Vec<V>,
    free: Vec<usize>,
    lengths: Vec<R>,
    storage_bound: usize,
    entry_bound: usize,
    compatibility: Tolerance<R>,
}

/// Packed vectors kept live by bordered nullspace validation, per unknown and
/// per local element unknown: operator stages, bordered right-hand sides and
/// their index maps.
const VALIDATION_COPIES: usize = 16;

/// Reject unsupported dimensions, empty meshes, nonpositive or nonfinite
/// physics and boundaries beyond static velocity/traction data.
fn validate<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    mesh: &SimplexMesh<R, D>,
    density: R,
    viscosity: R,
    boundaries: &TaggedBoundaries<R, V, D>,
) -> Result<(), SciOxideError> {
    let argument = if !density.is_finite() || density <= R::ZERO {
        scioxide_core::argument!("density")
    } else if !viscosity.is_finite() || viscosity <= R::ZERO {
        scioxide_core::argument!("viscosity")
    } else if !crate::fem::simplex::elasticity::static_traction_boundaries(boundaries) {
        scioxide_core::argument!("boundaries")
    } else if !(D == 2 || D == 3) || mesh.is_empty() {
        scioxide_core::argument!("mesh")
    } else {
        return mesh.mesh_token().ensure_compatible(&boundaries.token);
    };
    Err(invalid(
        argument,
        "incompressible flow requires nonempty 2D/3D P2/P1, positive finite density/viscosity and velocity/traction boundaries",
    ))
}

/// Gauge multiplier index of every connected vertex component: traction-anchored
/// components take `Unspecified`, closed components `Orthogonal` or `WeightedMean`.
fn gauge_indices<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    mesh: &SimplexMesh<R, D>,
    component: &[usize],
    count: usize,
    boundaries: &TaggedBoundaries<R, V, D>,
    gauges: &[EllipticGauge<V>],
) -> Result<Vec<Option<usize>>, SciOxideError> {
    if !gauges.is_empty() && gauges.len() != count {
        return Err(invalid(
            scioxide_core::argument!("options.gauges"),
            "one pressure gauge is required per connected component",
        ));
    }
    let mut anchored = try_filled(count, false, scioxide_core::quantity!("anchored"))?;
    let mut supported = try_filled(count, false, scioxide_core::quantity!("supported"))?;
    for face in mesh.boundary_faces() {
        let mut dirichlet = 0;
        for a in 0..D {
            let slot = boundaries.slot(face, a)?;
            dirichlet += usize::from(matches!(
                boundaries.conditions[slot].kind(),
                BoundaryKind::Dirichlet
            ));
        }
        if dirichlet != 0 && dirichlet != D {
            return Err(invalid(
                scioxide_core::argument!("boundaries"),
                "flow boundaries require full-vector velocity or full-vector traction per face",
            ));
        }
        let c = component[mesh.cell_vertices(mesh.face_cells(face)?[0])?[0]];
        supported[c] |= dirichlet == D;
        anchored[c] |= dirichlet == 0;
    }
    if supported.iter().any(|&b| !b) {
        return Err(invalid(
            scioxide_core::argument!("boundaries"),
            "each flow component requires a full velocity boundary face to remove rigid modes",
        ));
    }
    let mut index = try_filled(count, None, scioxide_core::quantity!("gauge_index"))?;
    let mut next = 0;
    for (c, slot) in index.iter_mut().enumerate() {
        let automatic = if anchored[c] {
            EllipticGauge::Unspecified
        } else {
            EllipticGauge::WeightedMean(V::ZERO)
        };
        let gauge = gauges.get(c).unwrap_or(&automatic);
        match (anchored[c], gauge) {
            (true, EllipticGauge::Unspecified) => {}
            (true, _) => {
                return Err(invalid(
                    scioxide_core::argument!("options.gauges"),
                    "traction fixes pressure; an additional gauge is unsupported",
                ));
            }
            (false, EllipticGauge::Unspecified) => {
                return Err(invalid(
                    scioxide_core::argument!("options.gauges"),
                    "closed velocity boundaries require an explicit pressure gauge",
                ));
            }
            (false, gauge) => {
                if let EllipticGauge::WeightedMean(v) = gauge {
                    finite(*v, scioxide_core::NonFiniteOrigin::Intermediate)?;
                }
                *slot = Some(next);
                next += 1;
            }
        }
    }
    Ok(index)
}

/// Minimum mapped altitude `|J| / |facet|` over the volume nodes of `cell`.
fn cell_length<R: RealScalar, const D: usize>(
    space: &SimplexSpace<R, D>,
    cell: usize,
) -> Result<R, SciOxideError> {
    let mut h = R::INFINITY;
    for reference in space.volume_nodes() {
        let map = space.map_node(cell, reference)?;
        for opposite in 0..=D {
            let (_, area) = super::reference::normal(&map, opposite)?;
            h = h.min(map.determinant / area);
        }
    }
    if !h.is_finite() || h <= R::ZERO {
        return Err(invalid(
            scioxide_core::argument!("mesh"),
            "unusable flow cell length",
        ));
    }
    Ok(h)
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> StokesPlan<R, V, D> {
    /// Prepare with default numerical settings and no resource ceiling.
    ///
    /// # Errors
    /// Returns the geometry, coefficient, allocation and configuration errors of [`Self::with_options`].
    pub fn new(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        density: R,
        viscosity: R,
        boundaries: &TaggedBoundaries<R, V, D>,
        body: &(dyn Fn([R; D]) -> Result<[V; D], SciOxideError> + Sync),
    ) -> Result<Self, SciOxideError> {
        Self::with_options(
            mesh,
            geometry,
            density,
            viscosity,
            boundaries,
            body,
            IncompressibleOptions::default(),
        )
    }

    /// Prepare a P2/P1 Stokes problem with frozen boundary and body data.
    /// # Errors
    /// Rejects invalid coefficients, spaces, boundaries, gauges, flux compatibility,
    /// geometry and resource ceilings. No partial plan is published.
    pub fn with_options(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        density: R,
        viscosity: R,
        boundaries: &TaggedBoundaries<R, V, D>,
        body: &(dyn Fn([R; D]) -> Result<[V; D], SciOxideError> + Sync),
        options: IncompressibleOptions<R, V>,
    ) -> Result<Self, SciOxideError> {
        Self::build(
            mesh,
            geometry,
            density,
            viscosity,
            boundaries,
            body,
            options,
            ExecutionContext::sequential_native(),
        )
    }
    /// Prepare as one epoch of `scope`.
    /// # Errors
    /// Includes constructor errors and cancellation/provider/budget failures.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit geometry, physics, loads and scope"
    )]
    pub fn build<'scope, 'capability>(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        density: R,
        viscosity: R,
        boundaries: &TaggedBoundaries<R, V, D>,
        body: &(dyn Fn([R; D]) -> Result<[V; D], SciOxideError> + Sync),
        options: IncompressibleOptions<R, V>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<Self, SciOxideError> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "stokes", 2),
            |context| {
                Self::prepare(
                    mesh, geometry, density, viscosity, boundaries, body, options, context,
                )
            },
        )
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit geometry, physics, loads and context"
    )]
    fn prepare(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        density: R,
        viscosity: R,
        boundaries: &TaggedBoundaries<R, V, D>,
        body: &(dyn Fn([R; D]) -> Result<[V; D], SciOxideError> + Sync),
        options: IncompressibleOptions<R, V>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Self, SciOxideError> {
        context.execution().check()?;
        validate(&mesh, density, viscosity, boundaries)?;
        let layout = Self::layout();
        // Gauge multipliers are bounded by the vertex count.
        layout.check_limits(&mesh, options.fem.limits, mesh.vertex_count())?;
        let storage_bound = Self::storage_bound(&mesh, &options.fem)?;
        if storage_bound > options.fem.limits.max_storage_bytes {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "flow storage ceiling exceeded",
            ));
        }
        // NavierStokesPlan reuses this space: the conservative convection form
        // has volume degree five and boundary degree six.
        let degrees = if matches!(geometry, SimplexFemGeometry::Quadratic(_)) {
            (7, 7)
        } else {
            (5, 6)
        };
        let space = SimplexSpace::prepare(
            mesh,
            geometry,
            SimplexFemOrder::P2,
            options.fem,
            degrees,
            context,
        )?;
        let components = space.components()?;
        let vertices = space.mesh.vertex_count();
        let mut component = try_with_capacity(vertices, scioxide_core::quantity!("component"))?;
        component.extend_from_slice(&components.labels()[..vertices]);
        let gauge_index = gauge_indices(
            &space.mesh,
            &component,
            components.count(),
            boundaries,
            &options.gauges,
        )?;
        let gauges = gauge_index.iter().flatten().count();
        let samples = space.samples(true, context)?;
        let mut lengths =
            try_with_capacity(space.cell_count(), scioxide_core::quantity!("lengths"))?;
        for cell in 0..space.cell_count() {
            lengths.push(cell_length(&space, cell)?);
        }
        let (mut load, prescribed) = space.static_loads(layout, boundaries, body, context)?;
        let physical = load.len();
        load.try_reserve_exact(gauges).map_err(|_| {
            SciOxideError::allocation_failure(scioxide_core::quantity!("load"), physical + gauges)
        })?;
        load.resize(physical + gauges, V::ZERO);
        let mut constrained =
            try_with_capacity(load.len(), scioxide_core::quantity!("constrained"))?;
        constrained.extend(prescribed.iter().map(Option::is_some));
        constrained.resize(load.len(), false);
        let velocity = D * space.nodes;
        let extra = component.iter().enumerate().flat_map(|(v, &c)| {
            gauge_index[c]
                .into_iter()
                .flat_map(move |k| [(velocity + v, physical + k), (physical + k, velocity + v)])
        });
        let assembly = AssemblyPattern::new(
            &space,
            layout,
            &constrained,
            crate::discretization::Elimination::Symmetric,
            extra,
        )?;
        let l = SimplexFemOrder::P2.nodes(D);
        let cells = space.cell_count();
        let mut plan = Self {
            matrices: try_filled(
                layout.entries(&space.mesh)?,
                R::ZERO,
                scioxide_core::quantity!("matrices"),
            )?,
            masses: try_filled(
                size(cells, l * l, scioxide_core::quantity!("masses"))?,
                R::ZERO,
                scioxide_core::quantity!("masses"),
            )?,
            pressure_masses: try_filled(
                size(
                    cells,
                    (D + 1) * (D + 1),
                    scioxide_core::quantity!("pressure_masses"),
                )?,
                R::ZERO,
                scioxide_core::quantity!("pressure_masses"),
            )?,
            gauge_weights: try_filled(
                vertices,
                R::ZERO,
                scioxide_core::quantity!("gauge_weights"),
            )?,
            gauge_targets: try_filled(gauges, V::ZERO, scioxide_core::quantity!("gauge_targets"))?,
            free: Vec::new(),
            rhs: Vec::new(),
            entry_bound: add(
                add(
                    layout.entries(&space.mesh)?,
                    load.len(),
                    scioxide_core::quantity!("entry_bound"),
                )?,
                size(2, vertices, scioxide_core::quantity!("entry_bound"))?,
                scioxide_core::quantity!("entry_bound"),
            )?,
            space,
            density,
            viscosity,
            samples,
            component,
            gauge_index,
            prescribed,
            constrained,
            assembly,
            assembly_extra: Vec::new(),
            load,
            lengths,
            storage_bound,
            compatibility: options.compatibility,
        };
        plan.prepare_matrices(context)?;
        plan.prepare_gauges(&options.gauges)?;
        let mut assembly_extra =
            try_with_capacity(2 * vertices, scioxide_core::quantity!("gauge_values"))?;
        assembly_extra.extend(plan.gauge_entries().map(|(_, _, value)| value));
        plan.assembly_extra = assembly_extra;
        plan.prepare_rhs()?;
        plan.free = try_with_capacity(plan.velocity_unknowns(), scioxide_core::quantity!("free"))?;
        plan.free
            .extend((0..plan.velocity_unknowns()).filter(|&i| plan.prescribed[i].is_none()));
        plan.validate_nullspaces(context)?;
        context.execution().check()?;
        Ok(plan)
    }
    /// Conservative live preparation payload: the space, mapped cell and facet
    /// samples, element blocks, packed vectors and bordered validation copies.
    fn storage_bound(
        mesh: &SimplexMesh<R, D>,
        fem: &SimplexFemOptions<R>,
    ) -> Result<usize, SciOxideError> {
        let layout = Self::layout();
        let unknowns = add(
            layout.unknowns(mesh)?,
            mesh.vertex_count(),
            scioxide_core::quantity!("unknowns"),
        )?;
        let local = size(
            mesh.cell_count(),
            layout.local::<D>(),
            scioxide_core::quantity!("local"),
        )?;
        let quadrature = fem.limits.max_quadrature_nodes;
        let faces = mesh.boundary_faces().count();
        let sites = size(
            size(faces, quadrature, scioxide_core::quantity!("sites"))?,
            D,
            scioxide_core::quantity!("sites"),
        )?;
        let site = Bytes::new(size_of::<crate::BoundaryPoint<R, D>>())
            .array::<(usize, R)>(SimplexFemOrder::P2.nodes(D))?
            .total();
        Ok(
            SimplexSpace::storage_bound(mesh, SimplexFemOrder::P2, fem.limits)?
                .bytes(
                    AssemblyPattern::bound(
                        add(
                            layout.entries(mesh)?,
                            2 * mesh.vertex_count(),
                            scioxide_core::quantity!("assembly_pattern"),
                        )?,
                        unknowns,
                    )?
                    .total(),
                )?
                .bytes(
                    CellSamples::<R, D>::bound(mesh.cell_count(), faces, quadrature, true)?.total(),
                )?
                .bytes(size(sites, site, scioxide_core::quantity!("sites"))?)?
                .array::<(R, R, R, usize, usize, V)>(layout.entries(mesh)?)?
                .array::<[(V, usize); VALIDATION_COPIES]>(add(
                    unknowns,
                    local,
                    scioxide_core::quantity!("buffer_bytes"),
                )?)?
                .total(),
        )
    }
    /// P2 velocity components followed by P1 vertex pressure.
    const fn layout() -> Layout {
        Layout {
            order: SimplexFemOrder::P2,
            components: D,
            pressure: true,
        }
    }
    fn local(&self) -> usize {
        Self::layout().local::<D>()
    }
    fn global(&self, cell: usize, i: usize) -> usize {
        self.space.global(Self::layout(), cell, i)
    }
    fn physical(&self) -> usize {
        self.prescribed.len()
    }
    /// Tabulated P2 values of a sample.
    fn basis(&self, s: &Sample<R, D>) -> &[R; MAX_NODES] {
        &self.space.reference(s).solution.values
    }
    /// Tabulated P1 pressure values of a sample.
    fn linear(&self, s: &Sample<R, D>) -> &[R; 4] {
        &self.space.reference(s).linear
    }
}

#[cfg(test)]
impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> StokesPlan<R, V, D> {
    /// Mapped sample of rule weight `weight` at an arbitrary reference point,
    /// with its reference values.
    fn sample_at(
        &self,
        cell: usize,
        point: [R; D],
        weight: R,
    ) -> Result<(Sample<R, D>, super::space::ReferenceSample<R, D>), SciOxideError> {
        let (mut sample, reference) = self.space.point_sample(cell, point)?;
        sample.weight *= weight;
        Ok((sample, reference))
    }
}

/// Sequential epoch for test evaluations of scoped flow kernels.
#[cfg(test)]
fn ephemeral<T>(
    run: impl FnOnce(&mut SolveContext<'_, '_>) -> Result<T, SciOxideError>,
) -> Result<T, SciOxideError> {
    crate::attempt::sequential(
        scioxide_core::algorithm_id!("pde", "incompressible", 2),
        run,
    )
}
