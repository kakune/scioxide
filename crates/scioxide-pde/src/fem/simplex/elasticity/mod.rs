//! Isotropic small-strain and finite-strain elasticity on shared simplex spaces.
use super::space::{self, Bytes, CellSamples, Sample};
use super::*;
use scioxide_core::{
    AdjointOperator, BlockVectorView, JacobianOperator, ResidualOperator, SolveScope,
};

mod linear;
mod material;
mod modes;
mod nonlinear;
pub use linear::LinearElasticityPlan;
pub use material::IsotropicElasticMaterial;
pub use modes::ElasticityRigidModes;
pub use nonlinear::{HyperelasticityLinearizationWorkspace, HyperelasticityPlan};

/// Three-dimensional constitutive assumption for the geometric domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElasticityKinematics {
    /// Two-dimensional displacement; out-of-plane strain is zero.
    PlaneStrain,
    /// Two-dimensional displacement; out-of-plane normal stress is zero.
    PlaneStress,
    /// Full three-dimensional displacement.
    ThreeDimensional,
}
/// Explicit choice of displacement-only or stable mixed finite elements; it fixes
/// the displacement order, so no separate FEM order can disagree with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElasticityFormulation {
    /// P1 or P2 displacement only. May lock at large bulk modulus.
    Displacement(SimplexFemOrder),
    /// Continuous P2 displacement and P1 tensile pressure on vertices.
    MixedP2P1,
}
impl ElasticityFormulation {
    /// Polynomial order of the displacement field.
    pub const fn displacement_order(self) -> SimplexFemOrder {
        match self {
            Self::Displacement(order) => order,
            Self::MixedP2P1 => SimplexFemOrder::P2,
        }
    }
}
/// Bounded scalar reduction for neo-Hookean plane stress.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct PlaneStressOptions<R: RealScalar> {
    max_iterations: Budget,
    tolerance: R,
}
impl<R: RealScalar> Default for PlaneStressOptions<R> {
    fn default() -> Self {
        Self {
            max_iterations: Budget::ONE.saturating_add(63),
            tolerance: R::EPSILON.scalbn(6),
        }
    }
}
impl<R: RealScalar> PlaneStressOptions<R> {
    /// 64 bracket expansions and Newton iterations, relative tolerance 64 epsilon.
    pub fn new() -> Self {
        Self::default()
    }
    /// Maximum bracket expansions and maximum safeguarded Newton iterations.
    pub const fn with_max_iterations(mut self, iterations: Budget) -> Self {
        self.max_iterations = iterations;
        self
    }
    /// Relative stress and stretch tolerance in `(0, 1)`, validated by plan construction.
    pub const fn with_tolerance(mut self, tolerance: R) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Maximum bracket expansions and Newton iterations.
    pub const fn max_iterations(&self) -> Budget {
        self.max_iterations
    }
    /// Relative stress and stretch tolerance.
    pub const fn tolerance(&self) -> R {
        self.tolerance
    }
}
/// Elasticity kinematics, formulation, shared FEM choices and local solve policy.
#[derive(Debug, Clone)]
#[must_use]
pub struct ElasticityOptions<R: RealScalar> {
    kinematics: ElasticityKinematics,
    formulation: ElasticityFormulation,
    fem: SimplexFemOptions<R>,
    plane_stress: PlaneStressOptions<R>,
}
impl<R: RealScalar> ElasticityOptions<R> {
    /// Construct with an explicit discretization and the default plane-stress policy.
    pub fn new(kinematics: ElasticityKinematics, formulation: ElasticityFormulation) -> Self {
        Self {
            kinematics,
            formulation,
            fem: SimplexFemOptions::default(),
            plane_stress: PlaneStressOptions::default(),
        }
    }
    /// Replace the local nonlinear plane-stress solve policy.
    pub const fn with_plane_stress(mut self, options: PlaneStressOptions<R>) -> Self {
        self.plane_stress = options;
        self
    }
    /// Replace the shared FEM quadrature, conditioning, execution and limits.
    pub fn with_fem(mut self, fem: SimplexFemOptions<R>) -> Self {
        self.fem = fem;
        self
    }
    /// Physical reduction.
    pub const fn kinematics(&self) -> ElasticityKinematics {
        self.kinematics
    }
    /// Displacement-only or mixed formulation.
    pub const fn formulation(&self) -> ElasticityFormulation {
        self.formulation
    }
    /// Shared FEM choices.
    pub const fn fem(&self) -> &SimplexFemOptions<R> {
        &self.fem
    }
    /// Local plane-stress solve policy.
    pub const fn plane_stress(&self) -> PlaneStressOptions<R> {
        self.plane_stress
    }
}
/// Owned displacement fields and optional mixed P1 pressure.
#[derive(Debug, Clone)]
pub struct ElasticityField<V, const D: usize> {
    displacement: SimplexFemField<V, D>,
    pressure: Option<VertexField<V, D>>,
}
impl<V, const D: usize> ElasticityField<V, D> {
    /// Component-major shared vertex/edge displacement fields.
    pub const fn displacement(&self) -> &SimplexFemField<V, D> {
        &self.displacement
    }
    /// Tensile pressure on vertices, present exactly for MixedP2P1.
    pub const fn pressure(&self) -> Option<&VertexField<V, D>> {
        self.pressure.as_ref()
    }
    /// Separate displacement and pressure.
    pub fn into_parts(self) -> (SimplexFemField<V, D>, Option<VertexField<V, D>>) {
        (self.displacement, self.pressure)
    }
}
/// Borrowed shared displacement and optional pressure fields.
#[derive(Debug, Clone, Copy)]
pub struct ElasticityFieldView<'a, 'b, V, const D: usize> {
    displacement: SimplexFemFieldView<'a, 'b, V, D>,
    pressure: Option<FieldView<'b, V, Vertex, D>>,
}
impl<'a, 'b, V, const D: usize> ElasticityFieldView<'a, 'b, V, D> {
    /// Shared displacement views and, for mixed forms, pressure on the same mesh revision.
    pub const fn new(
        displacement: SimplexFemFieldView<'a, 'b, V, D>,
        pressure: Option<FieldView<'b, V, Vertex, D>>,
    ) -> Self {
        Self {
            displacement,
            pressure,
        }
    }
    /// Shared displacement views.
    pub const fn displacement(&self) -> &SimplexFemFieldView<'a, 'b, V, D> {
        &self.displacement
    }
    /// Mixed pressure view.
    pub const fn pressure(&self) -> Option<&FieldView<'b, V, Vertex, D>> {
        self.pressure.as_ref()
    }
}
/// Real energy diagnostics for the supplied state, without a convergence claim.
#[derive(Clone, Copy, Debug)]
pub struct ElasticityEnergy<R: RealScalar> {
    internal: R,
    external_work: R,
    potential: R,
}
impl<R: RealScalar> ElasticityEnergy<R> {
    /// Integrated stored energy for displacement forms, saddle energy for mixed forms.
    pub const fn internal(&self) -> R {
        self.internal
    }
    /// Reference load work; complex linear amplitudes use Re(u^H f).
    pub const fn external_work(&self) -> R {
        self.external_work
    }
    /// Internal minus external work. Not a spatial-error certificate.
    pub const fn potential(&self) -> R {
        self.potential
    }
}
/// Tensor stress and strain/deformation at a requested reference point.
#[derive(Clone, Copy, Debug)]
pub struct ElasticityStress<V: NumericScalar> {
    kinematic_tensor: [[V; 3]; 3],
    cauchy: [[V; 3]; 3],
    first_piola: [[V; 3]; 3],
    pressure: V,
}
impl<V: NumericScalar> ElasticityStress<V> {
    /// Symmetric small strain for Hooke, deformation gradient for neo-Hookean.
    pub const fn kinematic_tensor(&self) -> [[V; 3]; 3] {
        self.kinematic_tensor
    }
    /// Cauchy stress; real linear materials act complex-linearly on amplitudes.
    pub const fn cauchy(&self) -> [[V; 3]; 3] {
        self.cauchy
    }
    /// First Piola stress for finite strain; equals Cauchy stress for Hooke.
    pub const fn first_piola(&self) -> [[V; 3]; 3] {
        self.first_piola
    }
    /// Interpolated mixed pressure, or the constitutive volumetric stress parameter.
    pub const fn pressure(&self) -> V {
        self.pressure
    }
}

#[derive(Clone)]
struct Space<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    space: SimplexSpace<R, D>,
    layout: Layout,
    kinematics: ElasticityKinematics,
    material: IsotropicElasticMaterial<R>,
    plane_stress: PlaneStressOptions<R>,
    prescribed: Vec<Option<V>>,
    constrained: Vec<bool>,
    assembly: AssemblyPattern,
    load: Vec<V>,
    storage_bound: usize,
    entry_bound: usize,
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> Space<R, V, D> {
    #[expect(
        clippy::too_many_arguments,
        reason = "keep preparation inputs explicit"
    )]
    fn prepare(
        mesh: SimplexMesh<R, D>,
        geometry: SimplexFemGeometry<'_, R, D>,
        material: IsotropicElasticMaterial<R>,
        boundaries: &TaggedBoundaries<R, V, D>,
        body: &(dyn Fn([R; D]) -> Result<[V; D], SciOxideError> + Sync),
        options: ElasticityOptions<R>,
        linear: bool,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<Self, SciOxideError> {
        context.execution().check()?;
        let order = options.formulation.displacement_order();
        validate(&geometry, order, boundaries, &options)?;
        mesh.mesh_token().ensure_compatible(&boundaries.token)?;
        let layout = Layout {
            order,
            components: D,
            pressure: options.formulation == ElasticityFormulation::MixedP2P1,
        };
        layout.check_limits(&mesh, options.fem.limits, 0)?;
        let storage_bound = storage_bound::<R, V, D>(&mesh, layout, &options.fem, linear)?;
        if storage_bound > options.fem.limits.max_storage_bytes {
            return Err(invalid(
                scioxide_core::argument!("options"),
                "elasticity preparation storage ceiling exceeded",
            ));
        }
        let minimum = if matches!(geometry, SimplexFemGeometry::Quadratic(_)) {
            7
        } else {
            2 * order.degree()
        };
        let space = SimplexSpace::prepare(
            mesh,
            geometry,
            order,
            options.fem,
            (minimum, minimum),
            context,
        )?;
        for cell in 0..space.cell_count() {
            space.check_geometry(cell)?;
        }
        let (load, prescribed) = space.static_loads(layout, boundaries, body, context)?;
        let mut constrained =
            try_with_capacity(prescribed.len(), scioxide_core::quantity!("constrained"))?;
        constrained.extend(prescribed.iter().map(Option::is_some));
        let assembly = AssemblyPattern::new(
            &space,
            layout,
            &constrained,
            crate::discretization::Elimination::Symmetric,
            core::iter::empty(),
        )?;
        Ok(Self {
            entry_bound: add(
                layout.entries(&space.mesh)?,
                load.len(),
                scioxide_core::quantity!("entry_bound"),
            )?,
            space,
            layout,
            kinematics: options.kinematics,
            material,
            plane_stress: options.plane_stress,
            prescribed,
            constrained,
            assembly,
            load,
            storage_bound,
        })
    }
    fn n(&self) -> usize {
        self.load.len()
    }
    fn local(&self) -> usize {
        self.layout.local::<D>()
    }
    fn mixed(&self) -> bool {
        self.layout.pressure
    }
    fn global(&self, cell: usize, i: usize) -> usize {
        self.space.global(self.layout, cell, i)
    }
    fn prescribed(&self, i: usize) -> Option<V> {
        self.prescribed.get(i).copied().flatten()
    }
    fn work(&self) -> usize {
        self.n() + self.space.cell_count() * self.local()
    }
    fn check(&self, x: &[V], out: &[V], scratch: &[V], needed: usize) -> Result<(), SciOxideError> {
        space::check_buffers(self.n(), x, out, scratch, needed)
    }
    /// Displacement gradient `h[3a + j] = ∂uₐ/∂xⱼ` and mixed pressure `h[9]`.
    fn variables(
        &self,
        sample: &Sample<R, D>,
        linear: &[R; 4],
        x: &[V],
        lift: bool,
    ) -> Result<[V; 10], SciOxideError> {
        let mut h = [V::ZERO; 10];
        let l = self.space.order.nodes(D);
        for a in 0..D {
            for i in 0..l {
                let row = self.global(sample.cell, a * l + i);
                let value = if lift {
                    self.prescribed(row).unwrap_or(x[row])
                } else {
                    x[row]
                };
                for j in 0..D {
                    h[3 * a + j] = finite(
                        h[3 * a + j] + value * V::from_real(sample.gradients[i][j]),
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?;
                }
            }
        }
        if self.mixed() {
            for (i, &p) in linear.iter().enumerate().take(D + 1) {
                h[9] = finite(
                    h[9] + x[self.global(sample.cell, D * l + i)] * V::from_real(p),
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        Ok(h)
    }
    /// Kinematic variables of local unknown `i` with unit coefficient.
    fn b(&self, sample: &Sample<R, D>, linear: &[R; 4], i: usize) -> [R; 10] {
        let mut b = [R::ZERO; 10];
        let l = self.space.order.nodes(D);
        if i < D * l {
            b[3 * (i / l)..3 * (i / l) + D].copy_from_slice(&sample.gradients[i % l]);
        } else {
            b[9] = linear[i - D * l];
        }
        b
    }
    /// Element matrix `Σ_q w_q bᵢᵀ C_q bⱼ` over the volume samples of one cell.
    fn element(
        &self,
        samples: &[Sample<R, D>],
        matrix: &mut [V],
        mut tangent: impl FnMut(&Sample<R, D>) -> Result<[[R; 10]; 10], SciOxideError>,
    ) -> Result<(), SciOxideError> {
        let n = self.local();
        for sample in samples {
            let c = tangent(sample)?;
            let linear = &self.space.reference(sample).linear;
            for i in 0..n {
                let a = self.b(sample, linear, i);
                for j in i..n {
                    let b = self.b(sample, linear, j);
                    let value = finite(
                        matrix[i * n + j]
                            + V::from_real(sample.weight * linear::contract(&c, &a, &b)?),
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?;
                    matrix[i * n + j] = value;
                    matrix[j * n + i] = value;
                }
            }
        }
        Ok(())
    }
    /// Canonical CSR of the element matrices with symmetric Dirichlet elimination.
    fn csr(&self, matrices: &[V]) -> Result<CsrMatrix<V>, SciOxideError> {
        let n = self.local();
        self.assembly
            .matrix(&|cell, i, j| Ok(matrices[(cell * n + i) * n + j]), &[])
    }
    fn fields(&self, x: &[V]) -> Result<ElasticityField<V, D>, SciOxideError> {
        let (displacement, pressure) = self.space.unpack(self.layout, x)?;
        Ok(ElasticityField {
            displacement,
            pressure,
        })
    }
    fn pack(
        &self,
        fields: ElasticityFieldView<'_, '_, V, D>,
        output: &mut [V],
    ) -> Result<(), SciOxideError> {
        self.space.pack_into(
            self.layout,
            fields.displacement,
            fields.pressure.as_ref(),
            output,
        )
    }
    fn energy(&self, internal: R, x: &[V]) -> Result<ElasticityEnergy<R>, SciOxideError> {
        let external_work = finite(
            x.iter()
                .zip(&self.load)
                .fold(V::ZERO, |s, (&u, &f)| s + u.conj() * f),
            scioxide_core::NonFiniteOrigin::Intermediate,
        )?
        .real_component();
        Ok(ElasticityEnergy {
            internal,
            external_work,
            potential: finite(
                internal - external_work,
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?,
        })
    }
}

/// Reject unsupported dimensions, curved low-order geometry, plane-stress
/// tolerances and boundaries beyond static Dirichlet/traction data.
fn validate<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    geometry: &SimplexFemGeometry<'_, R, D>,
    order: SimplexFemOrder,
    boundaries: &TaggedBoundaries<R, V, D>,
    options: &ElasticityOptions<R>,
) -> Result<(), SciOxideError> {
    if !static_traction_boundaries(boundaries) {
        return Err(invalid(
            scioxide_core::argument!("boundaries"),
            "unsupported elasticity boundary",
        ));
    }
    let tolerance = options.plane_stress.tolerance();
    if !(D == 2 && options.kinematics != ElasticityKinematics::ThreeDimensional
        || D == 3 && options.kinematics == ElasticityKinematics::ThreeDimensional)
        || matches!(geometry, SimplexFemGeometry::Quadratic(_)) && order != SimplexFemOrder::P2
        || !tolerance.is_finite()
        || tolerance <= R::ZERO
        || tolerance >= R::ONE
    {
        return Err(invalid(
            scioxide_core::argument!("options"),
            "unsupported elasticity dimension, order or local tolerance",
        ));
    }
    Ok(())
}

/// Conservative live preparation payload: the space, volume samples, traction
/// sites, packed vectors, element scratch and, for linear plans, element
/// matrices and their Dirichlet lift.
fn storage_bound<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>(
    mesh: &SimplexMesh<R, D>,
    layout: Layout,
    fem: &SimplexFemOptions<R>,
    linear: bool,
) -> Result<usize, SciOxideError> {
    let unknowns = layout.unknowns(mesh)?;
    let entries = layout.entries(mesh)?;
    let quadrature = fem.limits.max_quadrature_nodes;
    let sites = size(
        size(
            mesh.boundary_faces().count(),
            quadrature,
            scioxide_core::quantity!("sites"),
        )?,
        D,
        scioxide_core::quantity!("sites"),
    )?;
    let site = Bytes::new(size_of::<crate::BoundaryPoint<R, D>>())
        .array::<(usize, R)>(layout.order.nodes(D))?
        .total();
    let matrices = if linear { entries } else { 0 };
    Ok(SimplexSpace::storage_bound(mesh, layout.order, fem.limits)?
        .bytes(AssemblyPattern::bound(entries, unknowns)?.total())?
        .bytes(CellSamples::<R, D>::bound(mesh.cell_count(), 0, quadrature, false)?.total())?
        .bytes(size(sites, site, scioxide_core::quantity!("sites"))?)?
        .array::<(V, V, V, V, V, Option<V>)>(unknowns)?
        .array::<V>(size(
            mesh.cell_count(),
            layout.local::<D>(),
            scioxide_core::quantity!("scratch"),
        )?)?
        .array::<V>(matrices)?
        .array::<(usize, usize, V)>(matrices)?
        .total())
}

// Public data/field access is identical for both plans; numerical kernels are separate.
macro_rules! accessors {
    ($value:ty) => {
        /// Total scalar unknowns: displacement components followed by vertex pressure.
        pub fn unknowns(&self) -> usize {
            self.space.n()
        }
        /// Number of displacement scalar nodes per component.
        pub fn displacement_nodes(&self) -> usize {
            self.space.space.nodes
        }
        /// Borrow the exact shared mesh.
        pub fn mesh(&self) -> &SimplexMesh<R, D> {
            &self.space.space.mesh
        }
        /// Physical interpolation point of a displacement node.
        pub fn node_point(&self, node: usize) -> Result<[R; D], SciOxideError> {
            self.space.space.dof_point(node)
        }
        /// Upper bound on explicitly requested CSR triplets.
        pub fn entry_bound(&self) -> usize {
            self.space.entry_bound
        }
        /// Conservative peak preparation bytes, excluding supplied inputs and callbacks.
        pub fn preparation_storage_bound(&self) -> usize {
            self.space.storage_bound
        }
        /// Physical reference body and traction load, before elimination.
        pub fn load(&self) -> &[$value] {
            &self.space.load
        }
        /// Prescribed displacement values; pressure is never prescribed.
        pub fn prescribed_values(&self) -> &[Option<$value>] {
            &self.space.prescribed
        }
        /// Convert checked scalar state to shared owned fields.
        pub fn fields(&self, x: &[$value]) -> Result<ElasticityField<$value, D>, SciOxideError> {
            self.space.fields(x)
        }
        /// Transactionally pack borrowed exact-mesh fields into caller storage.
        pub fn pack_fields_into(
            &self,
            fields: ElasticityFieldView<'_, '_, $value, D>,
            output: &mut [$value],
        ) -> Result<(), SciOxideError> {
            self.space.pack(fields, output)
        }
        /// Diagnose support-compatible rigid modes in reference coordinates.
        /// For nonlinear states these are support diagnostics, not certified tangent kernels.
        pub fn rigid_modes(
            &self,
            tolerance: Tolerance<R>,
            max_modes: usize,
        ) -> Result<ElasticityRigidModes<$value>, SciOxideError> {
            modes::build(&self.space, tolerance, max_modes)
        }
    };
}
use accessors;

#[cfg(test)]
mod tests;

/// Elasticity and flow accept time-invariant Dirichlet and traction data on
/// `D` components; Robin data and time dependence are rejected.
pub(crate) fn static_traction_boundaries<
    R: RealScalar,
    V: NumericScalar<Real = R>,
    const D: usize,
>(
    boundaries: &TaggedBoundaries<R, V, D>,
) -> bool {
    boundaries.components == D
        && boundaries.time_dependence() == crate::TimeDependence::TimeInvariant
        && boundaries
            .conditions
            .iter()
            .all(|b| !matches!(b.kind(), BoundaryKind::Robin { .. }))
}
