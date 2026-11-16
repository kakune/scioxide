//! Continuous P1/P2 simplex weak forms and prepared element operators.

use crate::validation::finite;
use crate::validation::invalid;

use alloc::vec::Vec;
use core::mem::size_of;
use scioxide_core::backend::{
    checked_add as add, checked_mul as size, try_filled, try_with_capacity,
};
use scioxide_core::{
    Budget, EvaluationRole, ExecutionContext, ExecutionPolicy, LinearOperator, NonFiniteOrigin,
    NumericScalar, RealScalar, SciOxideError, SolveContext, Tolerance,
};
use scioxide_integrate::SimplexRule;
use scioxide_sparse::{CsrMatrix, DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};
use scioxide_spatial::{
    FieldView, MeshToken, SimplexMesh, UnstructuredEdge, UnstructuredEdgeField, Vertex, VertexField,
};

use crate::boundary::{BoundaryKind, TaggedBoundaries};
use crate::discretization::PreparedLoads;

mod affine;
mod assembly;
mod csr;
use csr::AssemblyPattern;
pub use csr::{SimplexCsrWorkspace, SimplexLinearization};
pub(crate) mod elasticity;
pub(crate) mod incompressible;
mod operator;
pub use operator::SimplexFemMassOperator;
pub(crate) mod reference;
pub(crate) mod space;
use crate::validation::{MAX_WORKERS, WORKER_LIMIT};
use reference::{BasisSample, ReferenceBasis};
use space::{Layout, SimplexSpace};

pub(crate) const MAX_NODES: usize = 10;

fn admit(context: &mut SolveContext<'_, '_>, provider: bool) -> Result<(), SciOxideError> {
    context.execution().check()?;
    let admitted = if provider {
        context
            .record_provider(EvaluationRole::Function)?
            .is_granted()
    } else {
        context.record_iteration()?.is_granted()
    };
    if admitted {
        Ok(())
    } else {
        let kind = if provider {
            scioxide_core::BudgetKind::Evaluation
        } else {
            scioxide_core::BudgetKind::Iteration
        };
        Err(SciOxideError::convergence_failure(
            if provider {
                scioxide_core::ConvergenceReason::EvaluationLimit
            } else {
                scioxide_core::ConvergenceReason::IterationLimit
            },
            scioxide_core::ConvergenceDiagnostics::new()
                .with_counts(context.counts())
                .with_budget(context.effective_budget_report(kind)),
            "FEM solve work budget exhausted",
        ))
    }
}

/// Continuous nodal polynomial degree on triangles and tetrahedra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimplexFemOrder {
    /// One degree of freedom per vertex.
    P1,
    /// Vertex values followed by one midpoint value per canonical edge.
    P2,
}
impl SimplexFemOrder {
    /// Polynomial degree.
    pub const fn degree(self) -> usize {
        match self {
            Self::P1 => 1,
            Self::P2 => 2,
        }
    }
    fn nodes(self, dimension: usize) -> usize {
        match self {
            Self::P1 => dimension + 1,
            Self::P2 => (dimension + 1) * (dimension + 2) / 2,
        }
    }
}

/// Reference-to-physical geometry over the shared straight-sided topology.
/// Curved coordinates are auxiliary geometry, fixed for the lifetime of a plan.
pub enum SimplexFemGeometry<'a, R: RealScalar, const D: usize> {
    /// Affine geometry defined by the mesh vertices.
    Affine,
    /// Physical coordinates at reference edge midpoints, in canonical edge order.
    /// Adjacent cells share these coordinates and consequently identical traces.
    Quadratic(FieldView<'a, [R; D], UnstructuredEdge, D>),
}

/// Hard ceilings on preparation, excluding supplied mesh/rules/callback storage
/// and subsequently requested CSR or owned fields. All products are checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimplexFemLimits {
    max_unknowns: usize,
    max_element_entries: usize,
    max_quadrature_nodes: usize,
    max_storage_bytes: usize,
}

impl Default for SimplexFemLimits {
    fn default() -> Self {
        Self::new(usize::MAX, usize::MAX, 31, usize::MAX)
    }
}

impl SimplexFemLimits {
    /// Ceilings on scalar unknowns (including components), the sum of dense element
    /// entries (including component coupling), the node count of either cubature
    /// rule, and simultaneously live preparation storage in bytes.
    pub const fn new(
        max_unknowns: usize,
        max_element_entries: usize,
        max_quadrature_nodes: usize,
        max_storage_bytes: usize,
    ) -> Self {
        Self {
            max_unknowns,
            max_element_entries,
            max_quadrature_nodes,
            max_storage_bytes,
        }
    }
    /// Maximum global scalar unknown count, including components.
    pub const fn max_unknowns(&self) -> usize {
        self.max_unknowns
    }
    /// Maximum sum of dense element entries, including component coupling.
    pub const fn max_element_entries(&self) -> usize {
        self.max_element_entries
    }
    /// Maximum node count of either the volume or facet cubature rule.
    pub const fn max_quadrature_nodes(&self) -> usize {
        self.max_quadrature_nodes
    }
    /// Conservative upper bound for simultaneously live preparation storage.
    pub const fn max_storage_bytes(&self) -> usize {
        self.max_storage_bytes
    }
}

/// Quadrature, geometry conditioning, execution and storage choices shared by
/// scalar, elasticity and flow plans; immutable rules may be reused across plans.
///
/// The execution policy is the only worker setting: independent cells and output
/// rows use at most 32 workers.
#[derive(Clone)]
#[must_use]
pub struct SimplexFemOptions<R: RealScalar> {
    limits: SimplexFemLimits,
    volume_rule: Option<SimplexRule<R>>,
    facet_rule: Option<SimplexRule<R>>,
    minimum_reciprocal_condition: R,
    execution: ExecutionPolicy,
}

impl<R: RealScalar> core::fmt::Debug for SimplexFemOptions<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SimplexFemOptions")
            .field("limits", &self.limits)
            .field(
                "volume_rule",
                &self.volume_rule.as_ref().map(SimplexRule::node_count),
            )
            .field(
                "facet_rule",
                &self.facet_rule.as_ref().map(SimplexRule::node_count),
            )
            .field(
                "minimum_reciprocal_condition",
                &self.minimum_reciprocal_condition,
            )
            .field("execution", &self.execution)
            .finish()
    }
}

impl<R: RealScalar> Default for SimplexFemOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}
impl<R: RealScalar> SimplexFemOptions<R> {
    /// Positive interior cubature, sequential execution and a `64 * epsilon`
    /// geometry cutoff with no storage ceiling.
    pub fn new() -> Self {
        Self {
            limits: SimplexFemLimits::default(),
            volume_rule: None,
            facet_rule: None,
            minimum_reciprocal_condition: R::EPSILON * R::ONE.scalbn(6),
            execution: ExecutionPolicy::Sequential,
        }
    }
    /// Replace the resource ceilings.
    pub const fn with_limits(mut self, limits: SimplexFemLimits) -> Self {
        self.limits = limits;
        self
    }
    /// Use an explicit volume rule of matching dimension and adequate exactness.
    /// Explicit signed rules waive the sampled-form positivity guarantee.
    pub fn with_volume_rule(mut self, rule: SimplexRule<R>) -> Self {
        self.volume_rule = Some(rule);
        self
    }
    /// Use an explicit facet rule of dimension one less than the volume rule.
    pub fn with_facet_rule(mut self, rule: SimplexRule<R>) -> Self {
        self.facet_rule = Some(rule);
        self
    }
    /// Minimum accepted reciprocal infinity-norm Jacobian condition estimate, in `[0, 1)`.
    pub const fn with_minimum_reciprocal_condition(mut self, cutoff: R) -> Self {
        self.minimum_reciprocal_condition = cutoff;
        self
    }
    /// Scheduling of independent cells and output rows.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.execution = policy;
        self
    }
    /// Resource ceilings.
    pub const fn limits(&self) -> SimplexFemLimits {
        self.limits
    }
    /// Explicit volume rule; `None` selects positive interior cubature.
    pub const fn volume_rule(&self) -> Option<&SimplexRule<R>> {
        self.volume_rule.as_ref()
    }
    /// Explicit facet rule; `None` selects positive interior cubature.
    pub const fn facet_rule(&self) -> Option<&SimplexRule<R>> {
        self.facet_rule.as_ref()
    }
    /// Geometry conditioning cutoff.
    pub const fn minimum_reciprocal_condition(&self) -> R {
        self.minimum_reciprocal_condition
    }
    /// Execution policy of cell and row loops.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.execution
    }
}

/// Borrowed coupled elliptic weak-form provider. Coordinates are real; values
/// may be real or complex. Each call must fill every output entry. Sources are
/// supplied separately as [`crate::FieldData`] when loads are evaluated, and
/// wave problems supply the squared speed as diffusion.
pub trait SimplexFemCoefficients<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>:
    Sync
{
    /// Declare diffusion and reaction independent of position. The default samples
    /// each quadrature point; `true` evaluates and admits the tensors once per plan.
    fn is_spatially_constant(&self) -> bool {
        false
    }

    /// Fill row-major diffusion `[(component, direction)][(component, direction)]`
    /// and reaction `[component][component]` at `point`.
    /// Diffusion must be positive definite and Hermitian to a relative `32*n*epsilon`
    /// entrywise rounding envelope, where `n = components * D`. Accepted tensors
    /// are projected to `(K + Kᴴ)/2` before assembly.
    /// Reaction may be any finite matrix. Sampling does not certify regularity
    /// or positivity between integration nodes.
    /// # Errors
    /// Errors propagate unchanged.
    fn evaluate(
        &self,
        point: [R; D],
        diffusion: &mut [V],
        reaction: &mut [V],
    ) -> Result<(), SciOxideError>;
}

/// Owned component fields. P1 has no edge fields; P2 has one per component.
#[derive(Debug, Clone)]
pub struct SimplexFemField<V, const D: usize> {
    vertices: Vec<VertexField<V, D>>,
    edges: Vec<UnstructuredEdgeField<V, D>>,
}
impl<V, const D: usize> SimplexFemField<V, D> {
    /// Values at shared vertices, in component order.
    pub fn vertices(&self) -> &[VertexField<V, D>] {
        &self.vertices
    }
    /// Values at shared edge midpoints, in component order for P2.
    pub fn edges(&self) -> &[UnstructuredEdgeField<V, D>] {
        &self.edges
    }
    /// Separate vertex and edge fields.
    pub fn into_parts(self) -> (Vec<VertexField<V, D>>, Vec<UnstructuredEdgeField<V, D>>) {
        (self.vertices, self.edges)
    }
}
/// Borrowed component fields with no packing or allocation.
#[derive(Debug, Clone, Copy)]
pub struct SimplexFemFieldView<'a, 'b, V, const D: usize> {
    vertices: &'a [FieldView<'b, V, Vertex, D>],
    edges: &'a [FieldView<'b, V, UnstructuredEdge, D>],
}
impl<'a, 'b, V, const D: usize> SimplexFemFieldView<'a, 'b, V, D> {
    /// Vertex views in component order, and edge views (empty for P1, one per
    /// component for P2).
    pub const fn new(
        vertices: &'a [FieldView<'b, V, Vertex, D>],
        edges: &'a [FieldView<'b, V, UnstructuredEdge, D>],
    ) -> Self {
        Self { vertices, edges }
    }
    /// Vertex views, in component order.
    pub const fn vertices(&self) -> &'a [FieldView<'b, V, Vertex, D>] {
        self.vertices
    }
    /// Edge views; empty for P1, one per component for P2.
    pub const fn edges(&self) -> &'a [FieldView<'b, V, UnstructuredEdge, D>] {
        self.edges
    }
}

/// Checked map at one reference point; ordinary floating-point estimates.
#[derive(Clone, Copy, Debug)]
pub struct SimplexFemMappedPoint<R: RealScalar, const D: usize> {
    pub(crate) point: [R; D],
    pub(crate) jacobian: [[R; D]; D],
    pub(crate) inverse_jacobian: [[R; D]; D],
    pub(crate) determinant: R,
    pub(crate) reciprocal_condition: R,
}
impl<R: RealScalar, const D: usize> SimplexFemMappedPoint<R, D> {
    /// Physical point.
    pub const fn point(&self) -> [R; D] {
        self.point
    }
    /// `dx_physical / dx_reference`, row-major.
    pub const fn jacobian(&self) -> [[R; D]; D] {
        self.jacobian
    }
    /// Inverse Jacobian; physical basis gradients use its transpose.
    pub const fn inverse_jacobian(&self) -> [[R; D]; D] {
        self.inverse_jacobian
    }
    /// Positive volume scaling.
    pub const fn determinant(&self) -> R {
        self.determinant
    }
    /// Reciprocal infinity-norm condition estimate of the Jacobian.
    pub const fn reciprocal_condition(&self) -> R {
        self.reciprocal_condition
    }
}

/// Prepared forms of one cell: component-major diffusion/reaction/Robin matrix
/// and scalar consistent mass.
struct Element<'a, R: RealScalar, V: NumericScalar<Real = R>> {
    matrix: &'a mut [V],
    mass: &'a mut [R],
    reactive: bool,
}

/// Immutable P1/P2 element assembly on shared interval/triangle/tetrahedron topology.
/// Preparation allocates once; borrowed application/evaluation never allocates. Storage and
/// application are linear in cell count for fixed order/components.
///
/// Natural data is the outward conormal flux `n·∑_b K_ab ∇u_b` and Robin data adds
/// `αu`; Dirichlet nodes are eliminated symmetrically. Loads are evaluated at any
/// time from prepared quadrature sites, so sources and boundary data may depend
/// on time while the operator does not.
///
/// ```
/// use scioxide_pde::{BoundaryData, BoundaryKind, ConstantDiffusion, EllipticProblem,
///     FieldData, SimplexFemGeometry, SimplexFemOrder, SimplexFemPlan, TaggedBoundaries};
/// use scioxide_spatial::{Point, TriangleMesh};
/// let points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
///     .map(Point::new).into_iter().collect::<Result<_, _>>()?;
/// let mesh = TriangleMesh::new(points, vec![[0, 1, 2]])?;
/// let boundaries = TaggedBoundaries::<f64, f64, 2>::uniform(
///     &mesh, 1, BoundaryKind::Dirichlet, BoundaryData::zero())?;
/// let fem = SimplexFemPlan::new(mesh, SimplexFemGeometry::Affine, SimplexFemOrder::P1,
///     &ConstantDiffusion::new(1.0), &boundaries)?;
/// let u = EllipticProblem::new(&fem, &FieldData::constant(1.0)).solve()?;
/// assert!(u.iter().all(|x| x.abs() < 1e-14)); // All three vertices are constrained.
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Clone)]
pub struct SimplexFemPlan<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> {
    space: SimplexSpace<R, D>,
    components: usize,
    matrices: Vec<V>,
    masses: Vec<R>,
    assembly: AssemblyPattern,
    mass_assembly: Option<alloc::boxed::Box<AssemblyPattern>>,
    boundaries: crate::boundary::TaggedBoundaries<R, V, D>,
    loads: crate::discretization::PreparedLoads<R, V, D>,
    constrained: Vec<bool>,
    lumped: Vec<R>,
    constant_modes: bool,
    entry_bound: usize,
    storage_bound: usize,
}

#[cfg(test)]
#[path = "../../../examples/support/simplex_fem.rs"]
pub(crate) mod meshes;

#[cfg(test)]
mod prepared_tests;
#[cfg(test)]
mod tests;
