//! One boundary-condition vocabulary for every discretization family.
//!
//! A [`BoundaryCondition`] combines a mesh-family region (a structured
//! [`BoundarySet`] or a `u32` simplex/adaptive face tag), an optional field
//! component, a [`BoundaryKind`] and explicitly time-classified [`BoundaryData`].
//! [`BoundaryConditions`] binds a complete list to one exact mesh revision through
//! a single validation path shared by structured FDM and FVM, simplex FEM and FVM
//! and adaptive FEM.
//!
//! Natural data is always the outward conormal flux `n·K∇u` (for elasticity, the
//! traction component), and Robin data combines it with the value:
//! `n·K∇u + αu = g`. There is no separate derivative coefficient, so a vanishing
//! conormal coefficient cannot be expressed; a fixed value is [`BoundaryKind::Dirichlet`].
//! Periodicity belongs to mesh topology: periodic seam faces accept no condition.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{NonFiniteOrigin, NumericScalar, RealScalar, SciOxideError, Tolerance};
use scioxide_spatial::{
    AxisSide, BoundarySet, CellMesh, FaceIndex, FaceTopology, MeshToken, SimplexBoundaryTags,
    SimplexMesh,
};

use crate::validation::invalid;

/// Declared time dependence of boundary or field data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeDependence {
    /// The data depends only on position (and component). Prepared systems may
    /// reuse successful samples; rebuild them after changing captured data.
    TimeInvariant,
    /// The data may depend on time.
    TimeDependent,
}

/// Physical site at which boundary data is evaluated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryPoint<R, const D: usize> {
    /// Physical coordinates.
    pub point: [R; D],
    /// Outward unit normal of the boundary face carrying the condition.
    pub normal: [R; D],
    /// Zero-based field component.
    pub component: usize,
}

/// Physical site at which source or initial data is evaluated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldPoint<R, const D: usize> {
    /// Physical coordinates.
    pub point: [R; D],
    /// Zero-based field component.
    pub component: usize,
}

type Evaluator<P, R, V> = dyn Fn(P, R) -> Result<V, SciOxideError> + Send + Sync;

/// Explicitly time-classified space-time data, cheap to clone.
///
/// A time-invariant callback cannot observe time, so time dependence is
/// structural rather than inferred from sampled values. Every evaluation
/// rejects non-finite results.
///
/// ```
/// use scioxide_pde::{BoundaryData, BoundaryPoint, TimeDependence};
/// let data = BoundaryData::<f64, f64, 1>::time_dependent(|p: BoundaryPoint<f64, 1>, t| {
///     Ok(p.point[0] + t)
/// });
/// assert_eq!(data.time_dependence(), TimeDependence::TimeDependent);
/// ```
pub struct PdeData<P, R, V> {
    time_dependence: TimeDependence,
    evaluator: Arc<Evaluator<P, R, V>>,
    constant: Option<V>,
}

/// Boundary data `g(point, normal, component, t)`.
pub type BoundaryData<R, V, const D: usize> = PdeData<BoundaryPoint<R, D>, R, V>;
/// Source or initial data `f(point, component, t)`.
pub type FieldData<R, V, const D: usize> = PdeData<FieldPoint<R, D>, R, V>;

impl<P, R, V: Clone> Clone for PdeData<P, R, V> {
    fn clone(&self) -> Self {
        Self {
            time_dependence: self.time_dependence,
            evaluator: Arc::clone(&self.evaluator),
            constant: self.constant.clone(),
        }
    }
}

impl<P: 'static, R: RealScalar, V: NumericScalar<Real = R> + Send + Sync + 'static>
    PdeData<P, R, V>
{
    /// Time-invariant data from `f(site)`.
    #[must_use]
    pub fn time_invariant<F>(function: F) -> Self
    where
        F: Fn(P) -> Result<V, SciOxideError> + Send + Sync + 'static,
    {
        Self {
            time_dependence: TimeDependence::TimeInvariant,
            evaluator: Arc::new(move |site, _| function(site)),
            constant: None,
        }
    }

    /// Time-dependent data from `f(site, t)`.
    #[must_use]
    pub fn time_dependent<F>(function: F) -> Self
    where
        F: Fn(P, R) -> Result<V, SciOxideError> + Send + Sync + 'static,
    {
        Self {
            time_dependence: TimeDependence::TimeDependent,
            evaluator: Arc::new(function),
            constant: None,
        }
    }

    /// Time-invariant constant data.
    #[must_use]
    pub fn constant(value: V) -> Self {
        let mut data = Self::time_invariant(move |_| Ok(value));
        data.constant = Some(value);
        data
    }

    /// Time-invariant zero data.
    #[must_use]
    pub fn zero() -> Self {
        Self::constant(V::ZERO)
    }
}

impl<P, R: RealScalar, V: NumericScalar<Real = R>> PdeData<P, R, V> {
    pub(crate) const fn constant_value(&self) -> Option<V> {
        self.constant
    }
    /// Declared time dependence.
    #[must_use]
    pub const fn time_dependence(&self) -> TimeDependence {
        self.time_dependence
    }

    /// Evaluate at `site` and finite `time`, rejecting non-finite callback values.
    ///
    /// # Errors
    /// Rejects non-finite time with input origin before invoking a callback.
    /// Propagates callback errors unchanged and reports non-finite results with
    /// callback origin.
    pub fn evaluate(&self, site: P, time: R) -> Result<V, SciOxideError> {
        crate::validation::finite(time, NonFiniteOrigin::Input)?;
        crate::validation::finite(
            match self.constant {
                Some(value) => value,
                None => (self.evaluator)(site, time)?,
            },
            NonFiniteOrigin::Callback,
        )
    }
}

impl<P, R, V> fmt::Debug for PdeData<P, R, V> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PdeData")
            .field("time_dependence", &self.time_dependence)
            .finish_non_exhaustive()
    }
}

/// Kind of a boundary condition, sharing one natural-data convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryKind<V> {
    /// Prescribed value `u = g`, imposed strongly by nodal families and through
    /// the face law by cell-centred finite volumes.
    Dirichlet,
    /// Prescribed outward conormal flux `n·K∇u = g`.
    Natural,
    /// `n·K∇u + alpha·u = g`.
    Robin {
        /// Finite boundary reaction coefficient.
        alpha: V,
    },
}

/// One condition on a mesh-family region.
///
/// `Region` is a structured [`BoundarySet`] or a `u32` simplex/adaptive face tag.
/// Without [`Self::for_component`] the condition applies to every component and
/// the data callback receives the component.
///
/// ```
/// use scioxide_pde::{BoundaryCondition, BoundaryData, BoundaryKind};
/// let robin = BoundaryCondition::<f64, f64, 2, u32>::robin(3, 2.0, BoundaryData::constant(1.0))
///     .for_component(1);
/// assert_eq!(robin.kind(), BoundaryKind::Robin { alpha: 2.0 });
/// assert_eq!(robin.component(), Some(1));
/// ```
pub struct BoundaryCondition<R, V, const D: usize, Region> {
    region: Region,
    component: Option<usize>,
    kind: BoundaryKind<V>,
    data: BoundaryData<R, V, D>,
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize, Region>
    BoundaryCondition<R, V, D, Region>
{
    /// Prescribe the value `u = g`.
    #[must_use]
    pub const fn dirichlet(region: Region, data: BoundaryData<R, V, D>) -> Self {
        Self {
            region,
            component: None,
            kind: BoundaryKind::Dirichlet,
            data,
        }
    }

    /// Prescribe the outward conormal flux `n·K∇u = g`.
    #[must_use]
    pub const fn natural(region: Region, data: BoundaryData<R, V, D>) -> Self {
        Self {
            region,
            component: None,
            kind: BoundaryKind::Natural,
            data,
        }
    }

    /// Prescribe `n·K∇u + alpha·u = g`; `alpha` must be finite when bound.
    #[must_use]
    pub const fn robin(region: Region, alpha: V, data: BoundaryData<R, V, D>) -> Self {
        Self {
            region,
            component: None,
            kind: BoundaryKind::Robin { alpha },
            data,
        }
    }

    /// Restrict the condition to one zero-based component.
    #[must_use]
    pub const fn for_component(mut self, component: usize) -> Self {
        self.component = Some(component);
        self
    }

    /// Region of the condition.
    pub const fn region(&self) -> &Region {
        &self.region
    }

    /// Selected component, or `None` for every component.
    pub const fn component(&self) -> Option<usize> {
        self.component
    }

    /// Kind of the condition.
    pub const fn kind(&self) -> BoundaryKind<V>
    where
        V: Copy,
    {
        self.kind
    }

    /// Time-classified data.
    pub const fn data(&self) -> &BoundaryData<R, V, D> {
        &self.data
    }
}

impl<R, V: Clone, const D: usize, Region: Clone> Clone for BoundaryCondition<R, V, D, Region> {
    fn clone(&self) -> Self {
        Self {
            region: self.region.clone(),
            component: self.component,
            kind: self.kind.clone(),
            data: self.data.clone(),
        }
    }
}

impl<R, V: fmt::Debug, const D: usize, Region: fmt::Debug> fmt::Debug
    for BoundaryCondition<R, V, D, Region>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BoundaryCondition")
            .field("region", &self.region)
            .field("component", &self.component)
            .field("kind", &self.kind)
            .field("data", &self.data)
            .finish()
    }
}

/// Resolution of different prescribed values at a shared interpolation node.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DirichletConflictPolicy {
    /// Reject values differing beyond absolute/relative machine epsilon.
    #[default]
    Reject,
    /// The later condition in the original list wins. Within one condition,
    /// the larger canonical face index wins (including normal-dependent data).
    /// This ordering is independent of cell traversal and parallel execution.
    LastConditionWins,
}

/// One prescribed candidate ranked by `(condition, face)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Prescribed<V> {
    pub(crate) value: V,
    pub(crate) rank: (usize, usize),
}

/// Merge a candidate under `policy`; `Reject` checks every pair against machine epsilon.
pub(crate) fn select_prescribed<V: NumericScalar>(
    previous: Option<Prescribed<V>>,
    candidate: Prescribed<V>,
    policy: DirichletConflictPolicy,
) -> Result<Prescribed<V>, SciOxideError> {
    if let Some(previous) = previous {
        let tolerance = Tolerance::<V::Real>::from_machine_epsilon();
        if policy == DirichletConflictPolicy::Reject
            && (candidate.value - previous.value).abs()
                > tolerance.absolute()
                    + tolerance.relative() * candidate.value.abs().max(previous.value.abs())
        {
            return Err(invalid(
                scioxide_core::argument!("boundaries"),
                "conflicting Dirichlet values at shared DOF",
            ));
        }
        if previous.rank >= candidate.rank {
            return Ok(previous);
        }
    }
    Ok(candidate)
}

/// A complete condition list bound to one exact mesh revision.
///
/// Every exterior face and component has exactly one condition. Construction
/// rejects foreign revisions, empty or interior regions, periodic seam faces,
/// duplicate or missing face/component assignments, out-of-range components and
/// non-finite Robin coefficients before any data callback is evaluated.
pub struct BoundaryConditions<R, V, const D: usize, Region> {
    pub(crate) token: MeshToken,
    pub(crate) components: usize,
    pub(crate) conditions: Vec<BoundaryCondition<R, V, D, Region>>,
    /// Sorted exterior face identifiers.
    pub(crate) exterior: Vec<usize>,
    /// Condition index of every exterior face (in `exterior` order) and component.
    pub(crate) slots: Vec<usize>,
    pub(crate) conflict_policy: DirichletConflictPolicy,
}

/// Conditions on structured [`CellMesh`] boundary sets (structured FDM and FVM).
pub type StructuredBoundaries<R, V, const D: usize> = BoundaryConditions<R, V, D, BoundarySet<D>>;
/// Conditions on tagged simplex or adaptive faces (simplex FEM/FVM and AMR).
pub type TaggedBoundaries<R, V, const D: usize> = BoundaryConditions<R, V, D, u32>;

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize, Region>
    BoundaryConditions<R, V, D, Region>
{
    /// The one validation path. `exterior` lists every exterior face identifier
    /// and `faces_of` resolves a region into exterior faces (rejecting others).
    /// Work and storage scale with the exterior faces, never the mesh size.
    fn bind(
        token: MeshToken,
        mut exterior: Vec<usize>,
        components: usize,
        conditions: Vec<BoundaryCondition<R, V, D, Region>>,
        mut faces_of: impl FnMut(&Region) -> Result<Vec<usize>, SciOxideError>,
    ) -> Result<Self, SciOxideError> {
        const CONDITIONS: scioxide_core::Argument = scioxide_core::argument!("conditions");
        if components == 0 {
            return Err(invalid(
                scioxide_core::argument!("components"),
                "boundary conditions require a positive component count",
            ));
        }
        exterior.sort_unstable();
        exterior.dedup();
        let mut slots = try_filled(
            checked_mul(
                exterior.len(),
                components,
                scioxide_core::quantity!("boundary_slots"),
            )?,
            None,
            scioxide_core::quantity!("boundary_slots"),
        )?;
        for (index, condition) in conditions.iter().enumerate() {
            if condition.component.is_some_and(|c| c >= components) {
                return Err(invalid(CONDITIONS, "boundary component is out of range"));
            }
            if let BoundaryKind::Robin { alpha } = condition.kind
                && !alpha.has_finite_components()
            {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Input,
                    "Robin coefficient must be finite",
                ));
            }
            let faces = faces_of(&condition.region)?;
            if faces.is_empty() {
                return Err(invalid(CONDITIONS, "region selects no exterior face"));
            }
            let selected = condition.component.map_or(0..components, |c| c..c + 1);
            for face in faces {
                let position = exterior
                    .binary_search(&face)
                    .map_err(|_| invalid(CONDITIONS, "region selects a non-exterior face"))?;
                for component in selected.clone() {
                    if slots[position * components + component]
                        .replace(index)
                        .is_some()
                    {
                        return Err(invalid(
                            CONDITIONS,
                            "duplicate exterior face/component condition",
                        ));
                    }
                }
            }
        }
        let slots = slots
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid(CONDITIONS, "missing exterior face/component condition"))?;
        Ok(Self {
            token,
            components,
            conditions,
            exterior,
            slots,
            conflict_policy: DirichletConflictPolicy::Reject,
        })
    }

    /// Bind tagged faces: `exterior` pairs every exterior face with its tag.
    pub(crate) fn tagged_faces(
        token: MeshToken,
        exterior: Vec<(usize, u32)>,
        components: usize,
        conditions: Vec<BoundaryCondition<R, V, D, Region>>,
        tag_of: impl Fn(&Region) -> u32,
    ) -> Result<Self, SciOxideError> {
        let mut faces = try_with_capacity(exterior.len(), scioxide_core::quantity!("exterior"))?;
        faces.extend(exterior.iter().map(|&(face, _)| face));
        Self::bind(token, faces, components, conditions, |region| {
            let tag = tag_of(region);
            let mut selected = Vec::new();
            for &(face, _) in exterior.iter().filter(|&&(_, t)| t == tag) {
                selected.try_reserve(1).map_err(|_| {
                    SciOxideError::allocation_failure(
                        scioxide_core::quantity!("boundary_region"),
                        selected.len() + 1,
                    )
                })?;
                selected.push(face);
            }
            Ok(selected)
        })
    }

    /// Exact mesh family and revision.
    pub const fn mesh_token(&self) -> &MeshToken {
        &self.token
    }

    /// Number of field components every exterior face carries.
    pub const fn components(&self) -> usize {
        self.components
    }

    /// Conditions in their original order.
    pub fn conditions(&self) -> &[BoundaryCondition<R, V, D, Region>] {
        &self.conditions
    }

    /// Shared-node Dirichlet precedence.
    pub const fn conflict_policy(&self) -> DirichletConflictPolicy {
        self.conflict_policy
    }

    /// Select shared-node precedence before preparing a discretization.
    /// Every candidate is still evaluated, including traces that lose precedence.
    #[must_use]
    pub const fn with_dirichlet_conflict_policy(mut self, policy: DirichletConflictPolicy) -> Self {
        self.conflict_policy = policy;
        self
    }

    /// `TimeDependent` when any condition's data is time-dependent.
    pub fn time_dependence(&self) -> TimeDependence {
        if self
            .conditions
            .iter()
            .any(|c| c.data.time_dependence() == TimeDependence::TimeDependent)
        {
            TimeDependence::TimeDependent
        } else {
            TimeDependence::TimeInvariant
        }
    }

    /// Index of the condition on an exterior face and component.
    pub(crate) fn slot(&self, face: usize, component: usize) -> Result<usize, SciOxideError> {
        self.exterior
            .binary_search(&face)
            .ok()
            .filter(|_| component < self.components)
            .map(|position| self.slots[position * self.components + component])
            .ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("boundaries"),
                    "face has no boundary condition",
                )
            })
    }

    /// Kind of the condition on an exterior face and component.
    pub(crate) fn kind_at(
        &self,
        face: usize,
        component: usize,
    ) -> Result<BoundaryKind<V>, SciOxideError> {
        Ok(self.conditions[self.slot(face, component)?].kind)
    }

    /// Evaluate the data of condition `index` at a boundary site.
    pub(crate) fn evaluate(
        &self,
        index: usize,
        site: BoundaryPoint<R, D>,
        time: R,
    ) -> Result<V, SciOxideError> {
        self.conditions[index].data.evaluate(site, time)
    }

    /// Whether no condition fixes values or adds a boundary reaction (a Robin
    /// coefficient of exactly zero adds none).
    pub(crate) fn is_natural_everywhere(&self) -> bool {
        self.conditions.iter().all(|c| match c.kind {
            BoundaryKind::Natural => true,
            BoundaryKind::Robin { alpha } => alpha == V::ZERO,
            BoundaryKind::Dirichlet => false,
        })
    }

    /// The same conditions with every Robin coefficient replaced by `robin[i]`
    /// (one value per condition); used to assemble Robin parameter tangents.
    pub(crate) fn with_robin_coefficients(&self, robin: &[V]) -> Result<Self, SciOxideError>
    where
        Region: Clone,
    {
        if robin.len() != self.conditions.len() {
            return Err(SciOxideError::dimension_mismatch(
                scioxide_core::Dimensions::one(self.conditions.len()),
                scioxide_core::Dimensions::one(robin.len()),
            ));
        }
        let mut tangent = self.clone();
        for (condition, &alpha) in tangent.conditions.iter_mut().zip(robin) {
            if let BoundaryKind::Robin { alpha: slot } = &mut condition.kind {
                *slot = alpha;
            }
        }
        Ok(tangent)
    }

    /// Ensure every Robin coefficient is real, as required by real face laws.
    pub(crate) fn require_real_robin(&self) -> Result<(), SciOxideError> {
        if self.conditions.iter().any(|c| {
            matches!(c.kind, BoundaryKind::Robin { alpha } if alpha.imaginary_component() != R::ZERO)
        }) {
            return Err(invalid(
                scioxide_core::argument!("boundaries"),
                "this discretization requires real Robin coefficients",
            ));
        }
        Ok(())
    }
}

impl<R, V: Clone, const D: usize, Region: Clone> Clone for BoundaryConditions<R, V, D, Region> {
    fn clone(&self) -> Self {
        Self {
            token: self.token.clone(),
            components: self.components,
            conditions: self.conditions.clone(),
            exterior: self.exterior.clone(),
            slots: self.slots.clone(),
            conflict_policy: self.conflict_policy,
        }
    }
}

impl<R, V: fmt::Debug, const D: usize, Region: fmt::Debug> fmt::Debug
    for BoundaryConditions<R, V, D, Region>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BoundaryConditions")
            .field("token", &self.token)
            .field("components", &self.components)
            .field("conditions", &self.conditions)
            .field("conflict_policy", &self.conflict_policy)
            .finish_non_exhaustive()
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>
    BoundaryConditions<R, V, D, BoundarySet<D>>
{
    /// Bind one condition to every exterior face and component in `O(f)` storage and work.
    /// # Errors
    /// Rejects zero components, non-finite Robin coefficients and failed allocation.
    pub fn uniform(
        mesh: &CellMesh<R, D>,
        components: usize,
        kind: BoundaryKind<V>,
        data: BoundaryData<R, V, D>,
    ) -> Result<Self, SciOxideError> {
        let mut faces = Vec::new();
        for axis in 0..D {
            for side in [AxisSide::Start, AxisSide::End] {
                for face in mesh.axis_boundary_faces(axis, side)? {
                    if matches!(mesh.face_topology(face)?, FaceTopology::Exterior { .. }) {
                        faces.try_reserve(1).map_err(|_| {
                            SciOxideError::allocation_failure(
                                scioxide_core::quantity!("exterior_faces"),
                                faces.len() + 1,
                            )
                        })?;
                        faces.push(face);
                    }
                }
            }
        }
        let conditions = if faces.is_empty() {
            Vec::new()
        } else {
            alloc::vec![BoundaryCondition {
                region: BoundarySet::new(mesh, faces)?,
                component: None,
                kind,
                data
            }]
        };
        Self::new(mesh, components, conditions)
    }
    /// Bind structured conditions; a region may cover any set of exterior faces.
    ///
    /// ```
    /// use scioxide_pde::{BoundaryCondition, BoundaryData, StructuredBoundaries};
    /// use scioxide_spatial::{AxisSide, AxisTopology, BoundarySet, CoordinateAxis, Grid1d, SegmentMesh};
    /// let mesh = SegmentMesh::new(Grid1d::new(
    ///     [CoordinateAxis::uniform(0.0, 1.0, 5)?], [AxisTopology::Bounded])?)?;
    /// let side = |s| BoundarySet::axis_side(&mesh, 0, s);
    /// let boundaries = StructuredBoundaries::<f64, f64, 1>::new(&mesh, 1, vec![
    ///     BoundaryCondition::dirichlet(side(AxisSide::Start)?, BoundaryData::zero()),
    ///     BoundaryCondition::natural(side(AxisSide::End)?, BoundaryData::constant(1.0)),
    /// ])?;
    /// assert_eq!(boundaries.conditions().len(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    /// Rejects foreign revisions, interior or periodic seam faces, duplicate or
    /// missing face/component conditions, invalid components or Robin coefficients.
    pub fn new(
        mesh: &CellMesh<R, D>,
        components: usize,
        conditions: Vec<BoundaryCondition<R, V, D, BoundarySet<D>>>,
    ) -> Result<Self, SciOxideError> {
        let mut exterior = try_with_capacity(0, scioxide_core::quantity!("exterior_faces"))?;
        for axis in 0..D {
            for side in [AxisSide::Start, AxisSide::End] {
                for face in mesh.axis_boundary_faces(axis, side)? {
                    if matches!(mesh.face_topology(face)?, FaceTopology::Exterior { .. }) {
                        exterior.try_reserve(1).map_err(|_| {
                            SciOxideError::allocation_failure(
                                scioxide_core::quantity!("exterior_faces"),
                                exterior.len() + 1,
                            )
                        })?;
                        exterior.push(structured_face_id(mesh, face)?);
                    }
                }
            }
        }
        Self::bind(
            mesh.mesh_token().clone(),
            exterior,
            components,
            conditions,
            |region| {
                if !region.is_compatible_with(mesh) {
                    return Err(invalid(
                        scioxide_core::argument!("conditions"),
                        "boundary set belongs to another mesh revision",
                    ));
                }
                let mut faces =
                    try_with_capacity(region.len(), scioxide_core::quantity!("boundary_region"))?;
                for &face in region.faces() {
                    if !matches!(mesh.face_topology(face)?, FaceTopology::Exterior { .. }) {
                        return Err(invalid(
                            scioxide_core::argument!("conditions"),
                            "periodic seam faces belong to mesh topology and accept no condition",
                        ));
                    }
                    faces.push(structured_face_id(mesh, face)?);
                }
                Ok(faces)
            },
        )
    }
}

impl<R: RealScalar, V: NumericScalar<Real = R>, const D: usize> BoundaryConditions<R, V, D, u32> {
    /// Bind one condition to every exterior face and component in `O(f)` storage and work.
    /// # Errors
    /// Rejects zero components, non-finite Robin coefficients and failed allocation.
    pub fn uniform(
        mesh: &SimplexMesh<R, D>,
        components: usize,
        kind: BoundaryKind<V>,
        data: BoundaryData<R, V, D>,
    ) -> Result<Self, SciOxideError> {
        let mut exterior = Vec::new();
        for face in mesh.boundary_faces() {
            exterior.try_reserve(1).map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("exterior_faces"),
                    exterior.len() + 1,
                )
            })?;
            exterior.push((face, 0));
        }
        Self::tagged_faces(
            mesh.mesh_token().clone(),
            exterior,
            components,
            alloc::vec![BoundaryCondition {
                region: 0,
                component: None,
                kind,
                data
            }],
            |&tag| tag,
        )
    }
    /// Bind tagged simplex conditions; every exterior face needs a tag.
    ///
    /// ```
    /// use scioxide_pde::{BoundaryCondition, BoundaryData, TaggedBoundaries};
    /// use scioxide_spatial::{Point, SimplexBoundaryTags, TriangleMesh};
    /// let mesh = TriangleMesh::new(vec![Point::new([0., 0.])?,
    ///     Point::new([1., 0.])?, Point::new([0., 1.])?], vec![[0, 1, 2]])?;
    /// let tags = SimplexBoundaryTags::new(&mesh,
    ///     mesh.boundary_faces().map(|face| (face, 0)).collect())?;
    /// let boundaries = TaggedBoundaries::<f64, f64, 2>::new(&mesh, &tags, 1,
    ///     vec![BoundaryCondition::dirichlet(0, BoundaryData::time_invariant(|p| Ok(p.normal[0])))])?;
    /// assert_eq!(boundaries.components(), 1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    /// Rejects foreign tags, untagged exterior faces, unused tags, duplicate or
    /// missing face/component conditions, invalid components or Robin coefficients.
    pub fn new(
        mesh: &SimplexMesh<R, D>,
        tags: &SimplexBoundaryTags<D>,
        components: usize,
        conditions: Vec<BoundaryCondition<R, V, D, u32>>,
    ) -> Result<Self, SciOxideError> {
        mesh.mesh_token().ensure_compatible(tags.mesh_token())?;
        let mut exterior = try_with_capacity(0, scioxide_core::quantity!("exterior_faces"))?;
        for face in mesh.boundary_faces() {
            let tag = tags.tag(mesh, face)?.ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("tags"),
                    "missing exterior face tag",
                )
            })?;
            exterior.try_reserve(1).map_err(|_| {
                SciOxideError::allocation_failure(
                    scioxide_core::quantity!("exterior_faces"),
                    exterior.len() + 1,
                )
            })?;
            exterior.push((face, tag));
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

/// Canonical structured face for an axis-major identifier.
pub(crate) fn structured_face_at<R: RealScalar, const D: usize>(
    mesh: &CellMesh<R, D>,
    mut id: usize,
) -> Result<FaceIndex<D>, SciOxideError> {
    let cells = *mesh.cell_extent().as_array();
    for axis in 0..D {
        let extent = scioxide_spatial::Extent::new(core::array::from_fn(|i| {
            cells[i] + usize::from(i == axis)
        }))?;
        let count = extent.element_count()?;
        if id < count {
            return FaceIndex::new(axis, extent.delinearize(id)?);
        }
        id -= count;
    }
    Err(SciOxideError::out_of_bounds(
        scioxide_core::quantity!("structured_face"),
        id,
        mesh.topology_sizes().faces(),
    ))
}

/// Axis-major identifier of a structured face.
pub(crate) fn structured_face_id<R: RealScalar, const D: usize>(
    mesh: &CellMesh<R, D>,
    face: FaceIndex<D>,
) -> Result<usize, SciOxideError> {
    let cells = *mesh.cell_extent().as_array();
    let mut offset = 0;
    for axis in 0..D {
        let extent = scioxide_spatial::Extent::new(core::array::from_fn(|i| {
            cells[i] + usize::from(i == axis)
        }))?;
        if axis == face.normal_axis() {
            return checked_add(
                offset,
                extent.linearize(face.index())?,
                scioxide_core::quantity!("structured_face_index"),
            );
        }
        offset = checked_add(
            offset,
            extent.element_count()?,
            scioxide_core::quantity!("structured_face_index"),
        )?;
    }
    Err(invalid(
        scioxide_core::argument!("mesh"),
        "invalid structured face axis",
    ))
}

/// Boundary data from a `(point, normal)` callback, for crate tests.
#[cfg(test)]
pub(crate) fn point_normal<
    R: RealScalar,
    V: NumericScalar<Real = R> + Send + Sync + 'static,
    const D: usize,
>(
    f: impl Fn([R; D], [R; D]) -> Result<V, SciOxideError> + Send + Sync + 'static,
) -> BoundaryData<R, V, D> {
    BoundaryData::time_invariant(move |p: BoundaryPoint<R, D>| f(p.point, p.normal))
}
