use crate::validation::shape;

use super::{ConservationLaw, NumericalFlux, RiemannReport, finite, invalid};
use crate::fvm::geometry::{
    Geometry, difference, dot, inverse_normal_matrix, simplex_geometry, structured_face_at,
    structured_face_id, structured_geometry,
};
use alloc::{sync::Arc, vec::Vec};
use core::{mem::size_of, num::NonZeroUsize};
use scioxide_core::backend::PartitionSlot;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{ExecutionContext, ExecutionPolicy, RealScalar, SciOxideError};
use scioxide_core::{PromotingCompensatedSum, Scaled};
use scioxide_spatial::{
    AxisSide, BoundarySet, Cell, CellMesh, FaceTopology, FieldView, MeshToken, SimplexBoundaryTags,
    SimplexMesh,
};

/// Reconstruction of cell averages at face centroids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConservationReconstruction {
    /// First-order piecewise constant states on any supported mesh.
    Constant,
    /// MUSCL minmod slopes on rectilinear structured cells.
    Minmod,
    /// MUSCL monotonized-central slopes on rectilinear structured cells.
    MonotonizedCentral,
    /// Least-squares gradients with face-extrema limiting on simplices.
    BarthJespersen,
}
/// Immutable numerical and scheduling choices.
///
/// The execution policy is the only worker setting; independent cells and faces
/// use at most 32 workers.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct ConservationOptions<R> {
    limits: ConservationLimits,
    reconstruction: ConservationReconstruction,
    cfl: R,
    execution: ExecutionPolicy,
}
impl<R: RealScalar> Default for ConservationOptions<R> {
    fn default() -> Self {
        Self::new()
    }
}
impl<R: RealScalar> ConservationOptions<R> {
    /// Select constant reconstruction, sequential execution, CFL 0.45 and no resource ceiling.
    pub fn new() -> Self {
        Self {
            limits: ConservationLimits::default(),
            reconstruction: ConservationReconstruction::Constant,
            cfl: R::from_f64_ieee(0.45),
            execution: ExecutionPolicy::Sequential,
        }
    }
    /// Set the reconstruction, validated by plan preparation.
    pub const fn with_reconstruction(mut self, reconstruction: ConservationReconstruction) -> Self {
        self.reconstruction = reconstruction;
        self
    }
    /// Set preparation and workspace resource ceilings.
    pub const fn with_limits(mut self, limits: ConservationLimits) -> Self {
        self.limits = limits;
        self
    }
    /// Return preparation and workspace ceilings.
    pub const fn limits(&self) -> ConservationLimits {
        self.limits
    }
    /// Positive Courant fraction, at most one half; validated by plan construction.
    pub const fn with_cfl(mut self, cfl: R) -> Self {
        self.cfl = cfl;
        self
    }
    /// Deterministic independent cell/face scheduling.
    pub const fn with_execution_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.execution = policy;
        self
    }
    /// Reconstruction.
    pub const fn reconstruction(&self) -> ConservationReconstruction {
        self.reconstruction
    }
    /// Courant fraction.
    pub const fn cfl(&self) -> R {
        self.cfl
    }
    /// Execution policy.
    pub const fn execution_policy(&self) -> ExecutionPolicy {
        self.execution
    }
}
/// Explicit ceilings for preparation and one subsequently allocated numerical workspace.
/// Excludes caller mesh/boundaries, callback storage, ODE storage and owned output fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConservationLimits {
    max_unknowns: usize,
    max_faces: usize,
    max_storage_bytes: usize,
}
impl Default for ConservationLimits {
    fn default() -> Self {
        Self::new(usize::MAX, usize::MAX, usize::MAX)
    }
}

impl ConservationLimits {
    /// Ceilings on scalar cell unknowns (including components), mesh faces (including
    /// duplicate periodic seam entities) and preparation plus one workspace in bytes.
    pub const fn new(max_unknowns: usize, max_faces: usize, max_storage_bytes: usize) -> Self {
        Self {
            max_unknowns,
            max_faces,
            max_storage_bytes,
        }
    }
    /// Maximum scalar cell unknowns.
    pub const fn max_unknowns(&self) -> usize {
        self.max_unknowns
    }
    /// Maximum mesh faces.
    pub const fn max_faces(&self) -> usize {
        self.max_faces
    }
    /// Maximum preparation-plus-workspace bytes.
    pub const fn max_storage_bytes(&self) -> usize {
        self.max_storage_bytes
    }
}
/// Exterior states are fed into the same Riemann problem as interior states.
pub enum ConservationBoundary<R, const D: usize, const N: usize> {
    /// Copy the reconstructed interior state into the exterior.
    Transmissive,
    /// Prescribe an exterior state at physical time and face centroid.
    #[expect(
        clippy::type_complexity,
        reason = "preserve the public boundary callback"
    )]
    Exterior(Arc<dyn Fn([R; D], R) -> Result<[R; N], SciOxideError> + Send + Sync>),
    /// Reflect normal momentum; only equations with a reflection contract support this.
    SlipWall,
}
impl<R, const D: usize, const N: usize> Clone for ConservationBoundary<R, D, N> {
    fn clone(&self) -> Self {
        match self {
            Self::Transmissive => Self::Transmissive,
            Self::Exterior(state) => Self::Exterior(Arc::clone(state)),
            Self::SlipWall => Self::SlipWall,
        }
    }
}
impl<R: RealScalar, const D: usize, const N: usize> ConservationBoundary<R, D, N> {
    /// Prescribe the exterior state at a face centroid and time, without per-call allocation.
    pub fn exterior(
        state: impl Fn([R; D], R) -> Result<[R; N], SciOxideError> + Send + Sync + 'static,
    ) -> Self {
        Self::Exterior(Arc::new(state))
    }
    fn exterior_state<E: ConservationLaw<R, D, N>>(
        &self,
        e: &E,
        time: R,
        point: [R; D],
        inside: &[R; N],
        normal: [R; D],
    ) -> Result<[R; N], SciOxideError> {
        let state = match self {
            Self::Transmissive => *inside,
            Self::Exterior(f) => f(point, time)?,
            Self::SlipWall => e.reflect(inside, normal)?,
        };
        if !e.admissible(&state) {
            // Prepared boundary conditions belong to the evaluating entry point's `self`.
            return Err(invalid(
                scioxide_core::argument!("self"),
                "inadmissible exterior boundary state",
            ));
        }
        Ok(state)
    }
}
/// CFL, boundary balance and reconstruction diagnostics for one complete residual evaluation.
#[derive(Debug, Clone, Copy)]
pub struct ConservationReport<R, const N: usize> {
    pub(super) maximum_step: Option<R>,
    pub(super) limiting_cell: Option<usize>,
    pub(super) maximum_speed: R,
    pub(super) boundary_flux: [R; N],
    pub(super) limited_cells: usize,
    pub(super) rank_deficient_cells: usize,
    pub(super) hlle_fallback_faces: usize,
}
impl<R: Copy, const N: usize> ConservationReport<R, N> {
    /// Largest stable explicit step from the CFL bound, absent for a static state.
    pub const fn maximum_step(&self) -> Option<R> {
        self.maximum_step
    }
    /// Cell attaining the step bound.
    pub const fn limiting_cell(&self) -> Option<usize> {
        self.limiting_cell
    }
    /// Largest enclosing signal speed.
    pub const fn maximum_speed(&self) -> R {
        self.maximum_speed
    }
    /// Integrated outward flux through exterior faces, per component.
    pub const fn boundary_flux(&self) -> [R; N] {
        self.boundary_flux
    }
    /// Cells whose reconstructed gradient differs from the unlimited one.
    ///
    /// A MUSCL cell counts when its limited slope differs from the centred
    /// difference on any axis, including a slope set to zero on a one-sided
    /// stencil; every reconstruction counts a cell whose face-extrema factor is
    /// below one or whose gradient is scaled down for admissible face states.
    pub const fn limited_cells(&self) -> usize {
        self.limited_cells
    }
    /// Cells whose least-squares stencil was rank deficient.
    pub const fn rank_deficient_cells(&self) -> usize {
        self.rank_deficient_cells
    }
    /// Faces where HLLC fell back to HLLE.
    pub const fn hlle_fallback_faces(&self) -> usize {
        self.hlle_fallback_faces
    }
}
#[derive(Clone)]
struct Stencil<R, const D: usize> {
    neighbors: Vec<(usize, [R; D])>,
    weights: Vec<[R; D]>,
    rank_deficient: bool,
}
#[derive(Clone, Copy)]
struct Reconstruction<R, const D: usize, const N: usize> {
    gradient: [[R; D]; N],
    limited: bool,
}
/// Caller-owned numerical buffers. Reuse with a plan of the same dimensions.
/// Successful borrowed evaluation allocates no library storage, including first use.
pub struct ConservationWorkspace<R: RealScalar, const D: usize, const N: usize> {
    reconstruction: Vec<Reconstruction<R, D, N>>,
    faces: Vec<RiemannReport<R, N>>,
    boundary_states: Vec<Option<[R; N]>>,
    pub(super) residual: Vec<[R; N]>,
    rates: Vec<PromotingCompensatedSum<R>>,
}
#[derive(Clone)]
pub(super) struct ConservationCore<R: RealScalar, const D: usize, const N: usize> {
    pub(super) token: MeshToken,
    pub(super) geometry: Geometry<R, D>,
    // Physical faces use their own index; duplicate seam faces point to their canonical face.
    aliases: Vec<usize>,
    offsets: Vec<([R; D], [R; D])>,
    stencils: Vec<Stencil<R, D>>,
    boundaries: Vec<ConservationBoundary<R, D, N>>,
    face_boundary: Vec<Option<usize>>,
    pub(super) options: ConservationOptions<R>,
    storage_bound: usize,
}

fn preflight<R: RealScalar, const D: usize, const N: usize>(
    cells: usize,
    faces: usize,
    limits: ConservationLimits,
    options: ConservationOptions<R>,
) -> Result<usize, SciOxideError> {
    if !(1..=3).contains(&D)
        || N == 0
        || !options.cfl.is_finite()
        || options.cfl <= R::ZERO
        || options.cfl > R::HALF
    {
        return Err(invalid(
            scioxide_core::argument!("options"),
            "invalid conservation dimension, CFL or worker bound",
        ));
    }
    let unknowns = checked_mul(cells, N, scioxide_core::quantity!("unknowns"))?;
    // Geometry + adjacency + normalized LS preparation temporaries + retained stencils
    // + reconstruction, residual, flux and per-cell rate buffers. Vec capacity bounds
    // use the maximal structured degree 2D (simplex degree D+1 is no larger).
    let degree = checked_mul(2, D, scioxide_core::quantity!("stencil_degree"))?;
    let cell_bytes = checked_add(
        size_of::<[R; D]>()
            + size_of::<R>()
            + size_of::<PromotingCompensatedSum<R>>()
            + size_of::<Vec<usize>>()
            + size_of::<Stencil<R, D>>()
            + size_of::<Reconstruction<R, D, N>>()
            + size_of::<[R; N]>(),
        checked_mul(
            degree,
            size_of::<usize>() * 2 + size_of::<[R; D]>() * 3,
            scioxide_core::quantity!("cell_bytes"),
        )?,
        scioxide_core::quantity!("cell_bytes"),
    )?;
    let face_bytes = size_of::<crate::fvm::geometry::FaceGeometry<R, D>>()
        + size_of::<usize>()
        + size_of::<Option<usize>>()
        + size_of::<([R; D], [R; D])>()
        + size_of::<Option<[R; N]>>()
        + size_of::<RiemannReport<R, N>>()
        + size_of::<ConservationBoundary<R, D, N>>();
    let bytes = checked_add(
        checked_add(
            checked_mul(cells, cell_bytes, scioxide_core::quantity!("storage_bytes"))?,
            checked_mul(faces, face_bytes, scioxide_core::quantity!("storage_bytes"))?,
            scioxide_core::quantity!("storage_bytes"),
        )?,
        size_of::<ConservationCore<R, D, N>>() + size_of::<ConservationWorkspace<R, D, N>>(),
        scioxide_core::quantity!("storage_bytes"),
    )?;
    if unknowns > limits.max_unknowns
        || faces > limits.max_faces
        || bytes > limits.max_storage_bytes
    {
        return Err(invalid(
            scioxide_core::argument!("options.limits"),
            "conservation preparation or workspace exceeds resource limits",
        ));
    }
    Ok(bytes)
}

impl<R: RealScalar, const D: usize, const N: usize> ConservationCore<R, D, N> {
    #[expect(clippy::too_many_arguments, reason = "prepared inputs remain explicit")]
    fn finish(
        token: MeshToken,
        geometry: Geometry<R, D>,
        aliases: Vec<usize>,
        offsets: Vec<([R; D], [R; D])>,
        boundaries: Vec<ConservationBoundary<R, D, N>>,
        face_boundary: Vec<Option<usize>>,
        options: ConservationOptions<R>,
        storage_bound: usize,
    ) -> Result<Self, SciOxideError> {
        let mut stencils =
            try_with_capacity(geometry.centers.len(), scioxide_core::quantity!("stencils"))?;
        for (cell, faces) in geometry.cell_faces.iter().enumerate() {
            let mut neighbors =
                try_with_capacity(2 * D, scioxide_core::quantity!("stencil_neighbors"))?;
            for &id in faces {
                let f = &geometry.faces[id];
                if aliases[id] != id {
                    continue;
                }
                let neighbor = if f.left == Some(cell) {
                    f.right
                } else {
                    f.left
                };
                if let Some(other) = neighbor {
                    if other == cell {
                        continue;
                    }
                    let d = if f.left == Some(cell) {
                        difference(offsets[id].0, offsets[id].1)
                    } else {
                        difference(offsets[id].1, offsets[id].0)
                    };
                    neighbors.push((other, d));
                }
            }
            let mut weights = Vec::new();
            let mut rank_deficient = false;
            if options.reconstruction == ConservationReconstruction::BarthJespersen {
                let mut rows =
                    try_with_capacity(neighbors.len(), scioxide_core::quantity!("stencil_rows"))?;
                rows.extend(neighbors.iter().map(|(_, d)| *d));
                weights = try_with_capacity(
                    neighbors.len(),
                    scioxide_core::quantity!("stencil_weights"),
                )?;
                match inverse_normal_matrix(&rows) {
                    Ok(inverse) => {
                        for &(_, d) in &neighbors {
                            let length = dot(d, d);
                            weights.push(core::array::from_fn(|i| dot(inverse[i], d) / length));
                        }
                    }
                    Err(_) => rank_deficient = true,
                }
            }
            stencils.push(Stencil {
                neighbors,
                weights,
                rank_deficient,
            });
        }
        Ok(Self {
            token,
            geometry,
            aliases,
            offsets,
            stencils,
            boundaries,
            face_boundary,
            options,
            storage_bound,
        })
    }
    /// Allocate a workspace.
    pub(super) fn workspace(&self) -> Result<ConservationWorkspace<R, D, N>, SciOxideError> {
        Ok(ConservationWorkspace {
            reconstruction: try_filled(
                self.geometry.centers.len(),
                Reconstruction {
                    gradient: [[R::ZERO; D]; N],
                    limited: false,
                },
                scioxide_core::quantity!("workspace"),
            )?,
            faces: try_filled(
                self.geometry.faces.len(),
                RiemannReport {
                    flux: [R::ZERO; N],
                    maximum_speed: R::ZERO,
                    used_hlle_fallback: false,
                },
                scioxide_core::quantity!("workspace"),
            )?,
            boundary_states: try_filled(
                self.geometry.faces.len(),
                None,
                scioxide_core::quantity!("boundary_states"),
            )?,
            residual: try_filled(
                self.geometry.centers.len(),
                [R::ZERO; N],
                scioxide_core::quantity!("workspace"),
            )?,
            rates: try_filled(
                self.geometry.centers.len(),
                PromotingCompensatedSum::ZERO,
                scioxide_core::quantity!("workspace"),
            )?,
        })
    }
    fn reconstruct<E: ConservationLaw<R, D, N>>(
        &self,
        e: &E,
        u: &[[R; N]],
        cell: usize,
        boundary_states: &[Option<[R; N]>],
    ) -> Result<Reconstruction<R, D, N>, SciOxideError> {
        let stencil = &self.stencils[cell];
        let mut gradient = [[R::ZERO; D]; N];
        let mut limited = false;
        match self.options.reconstruction {
            ConservationReconstruction::Constant => {}
            ConservationReconstruction::Minmod | ConservationReconstruction::MonotonizedCentral => {
                for component in 0..N {
                    for axis in 0..D {
                        let mut minus = None;
                        let mut plus = None;
                        for &(other, d) in &stencil.neighbors {
                            if d[axis] != R::ZERO {
                                let derivative = super::arithmetic::difference_quotient(
                                    u[other][component],
                                    u[cell][component],
                                    d[axis],
                                )?;
                                if d[axis] > R::ZERO {
                                    plus = Some((derivative, d[axis]));
                                } else {
                                    minus = Some((derivative, -d[axis]));
                                }
                            }
                        }
                        for &id in &self.geometry.cell_faces[cell] {
                            let Some(boundary) = boundary_states[id] else {
                                continue;
                            };
                            let offset = if self.geometry.faces[id].left == Some(cell) {
                                self.offsets[id].0
                            } else {
                                self.offsets[id].1
                            };
                            let d = offset[axis];
                            if d != R::ZERO {
                                let derivative = super::arithmetic::difference_quotient(
                                    boundary[component],
                                    u[cell][component],
                                    d,
                                )?;
                                if d > R::ZERO && plus.is_none() {
                                    plus = Some((derivative, d));
                                }
                                if d < R::ZERO && minus.is_none() {
                                    minus = Some((derivative, -d));
                                }
                            }
                        }
                        gradient[component][axis] = match (minus, plus) {
                            (Some((a, ha)), Some((b, hb))) => {
                                let centered = finite(
                                    (hb / (ha + hb)) * a + (ha / (ha + hb)) * b,
                                    scioxide_core::NonFiniteOrigin::Intermediate,
                                )?;
                                let slope = if self.options.reconstruction
                                    == ConservationReconstruction::Minmod
                                {
                                    minmod(a, b)
                                } else {
                                    minmod(centered, minmod(a + a, b + b))
                                };
                                limited |= slope != centered;
                                slope
                            }
                            (Some(_), None) | (None, Some(_)) => {
                                limited = true;
                                R::ZERO
                            }
                            _ => R::ZERO,
                        };
                    }
                }
            }
            ConservationReconstruction::BarthJespersen => {
                if !stencil.rank_deficient {
                    for (&(other, _), &weight) in stencil.neighbors.iter().zip(&stencil.weights) {
                        for component in 0..N {
                            let jump = u[other][component] - u[cell][component];
                            for axis in 0..D {
                                gradient[component][axis] = finite(
                                    gradient[component][axis] + weight[axis] * jump,
                                    scioxide_core::NonFiniteOrigin::Intermediate,
                                )?;
                            }
                        }
                    }
                }
            }
        }
        if gradient.iter().flatten().all(|v| *v == R::ZERO) {
            return Ok(Reconstruction { gradient, limited });
        }
        // All reconstructed face values obey cell/neighbor/exterior extrema.
        // Boundary states are sampled from the cell averages, never from an
        // already extrapolated trace that could have introduced a new extremum.
        for component in 0..N {
            let (mut low, mut high) = stencil.neighbors.iter().fold(
                (u[cell][component], u[cell][component]),
                |(lo, hi), &(other, _)| (lo.min(u[other][component]), hi.max(u[other][component])),
            );
            for &id in &self.geometry.cell_faces[cell] {
                if let Some(state) = boundary_states[id] {
                    low = low.min(state[component]);
                    high = high.max(state[component]);
                }
            }
            let mut theta = R::ONE;
            for &id in &self.geometry.cell_faces[cell] {
                let offset = if self.geometry.faces[id].left == Some(cell) {
                    self.offsets[id].0
                } else {
                    self.offsets[id].1
                };
                let jump = finite(
                    dot(gradient[component], offset),
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
                if jump > R::ZERO {
                    theta = theta.min((high - u[cell][component]) / jump);
                } else if jump < R::ZERO {
                    theta = theta.min((low - u[cell][component]) / jump);
                }
            }
            limited |= theta < R::ONE;
            for value in &mut gradient[component] {
                *value *= theta;
            }
        }
        // A common scalar reduces every conserved component; the cell average is unchanged.
        // At most 63 halvings, then the exact constant reconstruction θ = 0, bound all work.
        let mut theta = R::ONE;
        for attempt in 0..=64 {
            let valid = self.geometry.cell_faces[cell].iter().all(|&id| {
                let offset = if self.geometry.faces[id].left == Some(cell) {
                    self.offsets[id].0
                } else {
                    self.offsets[id].1
                };
                super::arithmetic::trace(&u[cell], &gradient, offset, theta)
                    .is_ok_and(|state| e.admissible(&state))
            });
            if valid {
                break;
            }
            limited = true;
            theta = if attempt == 63 {
                R::ZERO
            } else {
                theta * R::HALF
            };
        }
        for row in &mut gradient {
            for value in row {
                *value = finite(*value * theta, scioxide_core::NonFiniteOrigin::Intermediate)?;
            }
        }
        Ok(Reconstruction { gradient, limited })
    }
    fn trace(
        &self,
        u: &[[R; N]],
        reconstruction: &[Reconstruction<R, D, N>],
        cell: usize,
        offset: [R; D],
    ) -> Result<[R; N], SciOxideError> {
        super::arithmetic::trace(&u[cell], &reconstruction[cell].gradient, offset, R::ONE)
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "the numerical operation keeps operands, storage and execution controls explicit"
    )]
    pub(super) fn evaluate<E: ConservationLaw<R, D, N>, F: NumericalFlux<R, E, D, N>>(
        &self,
        e: &E,
        flux: &F,
        time: R,
        u: &[[R; N]],
        work: &mut ConservationWorkspace<R, D, N>,
        execution: ExecutionContext<'_>,
        flux_argument: scioxide_core::Argument,
    ) -> Result<ConservationReport<R, N>, SciOxideError> {
        shape(u.len(), self.geometry.centers.len())?;
        shape(work.reconstruction.len(), u.len())?;
        shape(work.residual.len(), u.len())?;
        shape(work.rates.len(), u.len())?;
        shape(work.faces.len(), self.geometry.faces.len())?;
        shape(work.boundary_states.len(), self.geometry.faces.len())?;
        e.validate()?;
        if !time.is_finite()
            || u.iter()
                .any(|v| v.iter().any(|x| !x.is_finite()) || !e.admissible(v))
        {
            return Err(invalid(
                if time.is_finite() {
                    scioxide_core::argument!("state")
                } else {
                    scioxide_core::argument!("time")
                },
                "invalid time or inadmissible cell average",
            ));
        }
        execution.check()?;
        parallel_fill(&mut work.boundary_states, self.options, execution, |id| {
            let Some(index) = self.face_boundary[id] else {
                return Ok(None);
            };
            let face = &self.geometry.faces[id];
            let (cell, sign) = match (face.left, face.right) {
                (Some(cell), None) => (cell, R::ONE),
                (None, Some(cell)) => (cell, -R::ONE),
                _ => return Ok(None),
            };
            self.boundaries[index]
                .exterior_state(
                    e,
                    time,
                    face.point,
                    &u[cell],
                    core::array::from_fn(|axis| sign * face.normal[axis]),
                )
                .map(Some)
        })?;
        let boundary_states = &work.boundary_states;
        parallel_fill(&mut work.reconstruction, self.options, execution, |cell| {
            self.reconstruct(e, u, cell, boundary_states)
        })?;
        let reconstruction = &work.reconstruction;
        parallel_fill(&mut work.faces, self.options, execution, |id| {
            if self.aliases[id] != id {
                return Ok(RiemannReport {
                    flux: [R::ZERO; N],
                    maximum_speed: R::ZERO,
                    used_hlle_fallback: false,
                });
            }
            let face = &self.geometry.faces[id];
            let left = face
                .left
                .map(|cell| self.trace(u, reconstruction, cell, self.offsets[id].0))
                .transpose()?;
            let right = face
                .right
                .map(|cell| self.trace(u, reconstruction, cell, self.offsets[id].1))
                .transpose()?;
            let boundary = || {
                self.face_boundary[id]
                    .map(|i| &self.boundaries[i])
                    .ok_or_else(|| {
                        invalid(
                            scioxide_core::argument!("self"),
                            "missing exterior boundary",
                        )
                    })
            };
            let (l, r) = match (left, right) {
                (Some(l), Some(r)) => (l, r),
                (Some(l), None) => {
                    let r = match boundary()? {
                        ConservationBoundary::Exterior(_) => {
                            boundary_states[id].ok_or_else(|| {
                                invalid(scioxide_core::argument!("self"), "missing exterior state")
                            })?
                        }
                        other => other.exterior_state(e, time, face.point, &l, face.normal)?,
                    };
                    (l, r)
                }
                (None, Some(r)) => {
                    let n = core::array::from_fn(|i| -face.normal[i]);
                    let l = match boundary()? {
                        ConservationBoundary::Exterior(_) => {
                            boundary_states[id].ok_or_else(|| {
                                invalid(scioxide_core::argument!("self"), "missing exterior state")
                            })?
                        }
                        other => other.exterior_state(e, time, face.point, &r, n)?,
                    };
                    (l, r)
                }
                _ => {
                    return Err(invalid(
                        scioxide_core::argument!("self"),
                        "face has no incident cell",
                    ));
                }
            };
            let mut result = flux.evaluate(e, &l, &r, face.normal)?;
            if !result.maximum_speed.is_finite() || result.maximum_speed < R::ZERO {
                return Err(invalid(flux_argument, "invalid numerical-flux wave bound"));
            }
            for value in &mut result.flux {
                *value = finite(
                    *value * face.area,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
            Ok(result)
        })?;
        let mut report = ConservationReport {
            maximum_step: None,
            limiting_cell: None,
            maximum_speed: R::ZERO,
            boundary_flux: [R::ZERO; N],
            limited_cells: work.reconstruction.iter().filter(|r| r.limited).count(),
            rank_deficient_cells: self.stencils.iter().filter(|s| s.rank_deficient).count(),
            hlle_fallback_faces: 0,
        };
        work.residual.fill([R::ZERO; N]);
        work.rates.fill(PromotingCompensatedSum::ZERO);
        for (id, face) in self.geometry.faces.iter().enumerate() {
            if self.aliases[id] != id {
                work.faces[id] = work.faces[self.aliases[id]];
                continue;
            }
            let f = work.faces[id];
            report.maximum_speed = report.maximum_speed.max(f.maximum_speed);
            report.hlle_fallback_faces += usize::from(f.used_hlle_fallback);
            for (cell, sign) in [(face.left, -R::ONE), (face.right, R::ONE)] {
                if let Some(cell) = cell {
                    for component in 0..N {
                        work.residual[cell][component] = finite(
                            work.residual[cell][component] + sign * f.flux[component],
                            scioxide_core::NonFiniteOrigin::Intermediate,
                        )?;
                    }
                    work.rates[cell].add_product(face.area, f.maximum_speed)?;
                }
            }
            if face.left.is_none() || face.right.is_none() {
                let sign = if face.left.is_some() { R::ONE } else { -R::ONE };
                for component in 0..N {
                    report.boundary_flux[component] = finite(
                        report.boundary_flux[component] + sign * f.flux[component],
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?;
                }
            }
        }
        for (cell, (&volume, &rate)) in self.geometry.volumes.iter().zip(&work.rates).enumerate() {
            for v in &mut work.residual[cell] {
                *v = finite(*v / volume, scioxide_core::NonFiniteOrigin::Intermediate)?;
            }
            if let Some(bound) = cfl_bound(self.options.cfl, volume, rate)?
                && report.maximum_step.is_none_or(|old| bound < old)
            {
                report.maximum_step = Some(bound);
                report.limiting_cell = Some(cell);
            }
        }
        execution.check()?;
        Ok(report)
    }
}
/// Preserve small/large face rates without normalizing ordinary rates into Scaled.
fn cfl_bound<R: RealScalar>(
    cfl: R,
    volume: R,
    rate: PromotingCompensatedSum<R>,
) -> Result<Option<R>, SciOxideError> {
    if let Some(sum) = rate.native()
        && let Ok(rate) = sum.total()
    {
        if rate == R::ZERO {
            return Ok(None);
        }
        let numerator = cfl * volume;
        let bound = numerator / rate;
        if numerator.is_normal() && rate.is_normal() && bound.is_normal() {
            return Ok(Some(bound));
        }
    }
    let rate = rate.total()?;
    if rate.is_zero() {
        return Ok(None);
    }
    let bound = Scaled::from_value(cfl)?
        .mul(Scaled::from_value(volume)?)?
        .div(rate)?;
    let bound = if bound.abs_cmp(Scaled::from_value(R::MAX)?).is_gt() {
        R::MAX
    } else {
        bound.round()?
    };
    if bound <= R::ZERO {
        return Err(invalid(
            scioxide_core::argument!("state"),
            "CFL step is not representable",
        ));
    }
    Ok(Some(bound))
}
fn minmod<R: RealScalar>(a: R, b: R) -> R {
    if a > R::ZERO && b > R::ZERO {
        a.min(b)
    } else if a < R::ZERO && b < R::ZERO {
        a.max(b)
    } else {
        R::ZERO
    }
}
fn parallel_fill<T: Send, R: RealScalar>(
    output: &mut [T],
    options: ConservationOptions<R>,
    execution: ExecutionContext<'_>,
    fill: impl Fn(usize) -> Result<T, SciOxideError> + Sync,
) -> Result<(), SciOxideError> {
    // Executor invariant failures are reported against the plan (`self`), which
    // owns the execution policy and worker bound.
    let plan = execution.plan(
        options.execution,
        output.len(),
        scioxide_core::PlanLimits {
            work_per_item: core::num::NonZeroUsize::MIN,
            auto_threshold: NonZeroUsize::MIN,
            worker_limit: crate::validation::WORKER_LIMIT,
        },
    );
    if plan.worker_count() <= 1 {
        for (i, v) in output.iter_mut().enumerate() {
            *v = fill(i)?;
        }
        return execution.check();
    }
    let mut slots: [PartitionSlot<&mut [T]>; 32] = core::array::from_fn(|_| PartitionSlot::empty());
    let mut tail = output;
    for partition in plan.partitions() {
        let (part, rest) = tail.split_at_mut(partition.range().len());
        tail = rest;
        slots[partition.worker_index()] = PartitionSlot::new(part);
    }
    execution.execute_partitioned(
        plan,
        &slots[..plan.worker_count()],
        &|partition, part, _| {
            execution.check()?;
            for (i, v) in part.iter_mut().enumerate() {
                *v = fill(partition.range().start + i)?;
            }
            execution.check()
        },
    )?;

    Ok(())
}

/// Prepared conservation geometry on segment, quadrilateral or hexahedral meshes.
#[derive(Clone)]
pub struct StructuredConservationPlan<R: RealScalar, const D: usize, const N: usize> {
    mesh: CellMesh<R, D>,
    pub(super) core: ConservationCore<R, D, N>,
}
impl<R: RealScalar, const D: usize, const N: usize> StructuredConservationPlan<R, D, N> {
    /// Prepare with default numerical settings and no resource ceiling.
    ///
    /// # Errors
    /// Returns the geometry, coefficient, allocation and configuration errors of [`Self::with_options`].
    pub fn new(
        mesh: CellMesh<R, D>,
        conditions: Vec<(BoundarySet<D>, ConservationBoundary<R, D, N>)>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(mesh, conditions, ConservationOptions::default())
    }

    /// Prepare canonical face pairing and stencils. Exterior sets must cover every
    /// bounded exterior exactly once; periodic seams need no boundary condition.
    /// # Errors
    /// Rejects bad options, boundary identity/coverage, geometry and resource limits.
    pub fn with_options(
        mesh: CellMesh<R, D>,
        conditions: Vec<(BoundarySet<D>, ConservationBoundary<R, D, N>)>,
        options: ConservationOptions<R>,
    ) -> Result<Self, SciOxideError> {
        let limits = options.limits;
        const CONDITIONS: scioxide_core::Argument = scioxide_core::argument!("conditions");
        if options.reconstruction == ConservationReconstruction::BarthJespersen {
            return Err(invalid(
                scioxide_core::argument!("options.reconstruction"),
                "structured meshes require constant or MUSCL reconstruction",
            ));
        }
        let bytes = preflight::<R, D, N>(
            mesh.topology_sizes().cells(),
            mesh.topology_sizes().faces(),
            limits,
            options,
        )?;
        let mut geometry = structured_geometry(&mesh)?;
        let count = geometry.faces.len();
        let mut aliases = try_with_capacity(count, scioxide_core::quantity!("aliases"))?;
        aliases.extend(0..count);
        let mut face_boundary = try_filled(count, None, scioxide_core::quantity!("face_boundary"))?;
        if conditions.len() > face_boundary.len() {
            return Err(invalid(CONDITIONS, "more boundary conditions than faces"));
        }
        let mut boundaries =
            try_with_capacity(conditions.len(), scioxide_core::quantity!("boundaries"))?;
        for (index, (region, condition)) in conditions.into_iter().enumerate() {
            region.mesh_token().ensure_compatible(mesh.mesh_token())?;
            if region.is_empty() {
                return Err(invalid(CONDITIONS, "empty boundary region"));
            }
            for &face in region.faces() {
                if !matches!(mesh.face_topology(face)?, FaceTopology::Exterior { .. }) {
                    return Err(invalid(
                        CONDITIONS,
                        "condition targets an interior or periodic face",
                    ));
                }
                let id = structured_face_id(&mesh, face)?;
                if face_boundary[id].replace(index).is_some() {
                    return Err(invalid(CONDITIONS, "duplicate boundary coverage"));
                }
            }
            boundaries.push(condition);
        }
        let mut offsets = try_with_capacity(count, scioxide_core::quantity!("offsets"))?;
        for id in 0..count {
            let face = structured_face_at(&mesh, id)?;
            match mesh.face_topology(face)? {
                FaceTopology::Exterior { .. } => {
                    face_boundary[id]
                        .ok_or_else(|| invalid(CONDITIONS, "missing exterior boundary coverage"))?;
                }
                FaceTopology::Periodic { side, paired } => {
                    let other = structured_face_id(&mesh, paired)?;
                    if side == AxisSide::End {
                        aliases[id] = other;
                    }
                    let mut index = *face.index().as_array();
                    let axis = face.normal_axis();
                    index[axis] = mesh.cell_extent().as_array()[axis] - 1;
                    let left = mesh
                        .cell_extent()
                        .linearize(scioxide_spatial::Index::new(index)?)?;
                    index[axis] = 0;
                    let right = mesh
                        .cell_extent()
                        .linearize(scioxide_spatial::Index::new(index)?)?;
                    geometry.faces[id].left = Some(left);
                    geometry.faces[id].right = Some(right);
                }
                _ => {}
            }
            let f = &geometry.faces[id];
            let mut dl = f
                .left
                .map_or([R::ZERO; D], |c| difference(f.point, geometry.centers[c]));
            let dr = f
                .right
                .map_or([R::ZERO; D], |c| difference(f.point, geometry.centers[c]));
            if matches!(
                mesh.face_topology(face)?,
                FaceTopology::Periodic {
                    side: AxisSide::Start,
                    ..
                }
            ) {
                let a = mesh.grid().axis(face.normal_axis())?;
                let width = crate::spatial::positive_width(
                    a.coordinate(a.len() - 1)?,
                    a.coordinate(0)?,
                    scioxide_core::argument!("mesh"),
                )?;
                dl[face.normal_axis()] = finite(
                    dl[face.normal_axis()] + width,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
            offsets.push((dl, dr));
        }
        for faces in &mut geometry.cell_faces {
            faces.clear();
        }
        for (id, f) in geometry.faces.iter().enumerate() {
            if aliases[id] == id {
                if let Some(c) = f.left {
                    geometry.cell_faces[c].push(id);
                }
                if let Some(c) = f.right {
                    geometry.cell_faces[c].push(id);
                }
            }
        }
        let core = ConservationCore::finish(
            mesh.mesh_token().clone(),
            geometry,
            aliases,
            offsets,
            boundaries,
            face_boundary,
            options,
            bytes,
        )?;
        Ok(Self { mesh, core })
    }
}
/// Prepared conservation geometry on triangles or tetrahedra with canonical face tags.
#[derive(Clone)]
pub struct SimplexConservationPlan<R: RealScalar, const D: usize, const N: usize> {
    mesh: SimplexMesh<R, D>,
    pub(super) core: ConservationCore<R, D, N>,
}
impl<R: RealScalar, const D: usize, const N: usize> SimplexConservationPlan<R, D, N> {
    /// Prepare with default numerical settings and no resource ceiling.
    ///
    /// # Errors
    /// Returns the geometry, coefficient, allocation and configuration errors of [`Self::with_options`].
    pub fn new(
        mesh: SimplexMesh<R, D>,
        tags: &SimplexBoundaryTags<D>,
        conditions: Vec<(u32, ConservationBoundary<R, D, N>)>,
    ) -> Result<Self, SciOxideError> {
        Self::with_options(mesh, tags, conditions, ConservationOptions::default())
    }

    /// Prepare bounded simplex geometry and complete exterior Riemann conditions.
    /// # Errors
    /// Rejects incompatible tags, missing/duplicate conditions, invalid options and limits.
    pub fn with_options(
        mesh: SimplexMesh<R, D>,
        tags: &SimplexBoundaryTags<D>,
        conditions: Vec<(u32, ConservationBoundary<R, D, N>)>,
        options: ConservationOptions<R>,
    ) -> Result<Self, SciOxideError> {
        let limits = options.limits;
        const CONDITIONS: scioxide_core::Argument = scioxide_core::argument!("conditions");
        const TAGS: scioxide_core::Argument = scioxide_core::argument!("tags");
        if D != 2 && D != 3 {
            return Err(invalid(
                scioxide_core::argument!("mesh"),
                "simplex conservation requires D=2 or D=3",
            ));
        }
        if !matches!(
            options.reconstruction,
            ConservationReconstruction::Constant | ConservationReconstruction::BarthJespersen
        ) {
            return Err(invalid(
                scioxide_core::argument!("options.reconstruction"),
                "simplex meshes require constant or Barth-Jespersen reconstruction",
            ));
        }
        let bytes = preflight::<R, D, N>(mesh.cell_count(), mesh.face_count(), limits, options)?;
        if !tags.is_compatible_with(&mesh) {
            return Err(invalid(TAGS, "foreign simplex boundary tags"));
        }
        for (i, (tag, _)) in conditions.iter().enumerate() {
            if conditions[..i].iter().any(|(t, _)| t == tag) {
                return Err(invalid(CONDITIONS, "duplicate simplex boundary condition"));
            }
        }
        let mut face_boundary = try_filled(
            mesh.face_count(),
            None,
            scioxide_core::quantity!("face_boundary"),
        )?;
        for face in mesh.boundary_faces() {
            let tag = tags
                .tag(&mesh, face)?
                .ok_or_else(|| invalid(TAGS, "missing exterior tag"))?;
            face_boundary[face] = Some(
                conditions
                    .iter()
                    .position(|(t, _)| *t == tag)
                    .ok_or_else(|| invalid(CONDITIONS, "missing simplex boundary condition"))?,
            );
        }
        if conditions.len() > face_boundary.len() {
            return Err(invalid(CONDITIONS, "more boundary conditions than faces"));
        }
        if (0..conditions.len()).any(|i| !face_boundary.contains(&Some(i))) {
            return Err(invalid(CONDITIONS, "unused simplex boundary condition"));
        }
        let mut boundaries =
            try_with_capacity(conditions.len(), scioxide_core::quantity!("boundaries"))?;
        boundaries.extend(conditions.into_iter().map(|(_, b)| b));
        let geometry = simplex_geometry(&mesh)?;
        let mut aliases =
            try_with_capacity(mesh.face_count(), scioxide_core::quantity!("aliases"))?;
        aliases.extend(0..mesh.face_count());
        let mut offsets =
            try_with_capacity(mesh.face_count(), scioxide_core::quantity!("offsets"))?;
        offsets.extend(geometry.faces.iter().map(|f| {
            (
                f.left
                    .map_or([R::ZERO; D], |c| difference(f.point, geometry.centers[c])),
                f.right
                    .map_or([R::ZERO; D], |c| difference(f.point, geometry.centers[c])),
            )
        }));
        let core = ConservationCore::finish(
            mesh.mesh_token().clone(),
            geometry,
            aliases,
            offsets,
            boundaries,
            face_boundary,
            options,
            bytes,
        )?;
        Ok(Self { mesh, core })
    }
}
macro_rules! common_plan {
    ($plan:ident,$mesh:ident) => {
        impl<R: RealScalar, const D: usize, const N: usize> $plan<R, D, N> {
            /// Source mesh with exact revision identity.
            pub const fn mesh(&self) -> &$mesh<R, D> {
                &self.mesh
            }
            /// Cell centroids in field order.
            pub fn centers(&self) -> &[[R; D]] {
                &self.core.geometry.centers
            }
            /// Positive cell measures in field order.
            pub fn volumes(&self) -> &[R] {
                &self.core.geometry.volumes
            }
            /// Number of scalar ODE components.
            pub fn unknowns(&self) -> usize {
                self.core.geometry.centers.len() * N
            }
            /// Conservative preparation-plus-one-workspace storage bound in bytes.
            pub fn storage_bound(&self) -> usize {
                self.core.storage_bound
            }
            /// Allocate one explicitly owned evaluation workspace.
            /// # Errors
            /// Returns checked allocation failures.
            pub fn workspace(&self) -> Result<ConservationWorkspace<R, D, N>, SciOxideError> {
                self.core.workspace()
            }
            /// Evaluate a mass-normalized residual with caller buffers and injected execution.
            /// Output is unchanged on failure. Complexity is O((cells+faces)*N*D).
            /// # Errors
            /// Propagates shape/identity, callback, admissibility, execution and range failures.
            #[expect(
                clippy::too_many_arguments,
                reason = "preserve the public residual API"
            )]
            pub fn residual_into<E: ConservationLaw<R, D, N>, F: NumericalFlux<R, E, D, N>>(
                &self,
                e: &E,
                f: &F,
                time: R,
                state: FieldView<'_, [R; N], Cell, D>,
                output: &mut [[R; N]],
                work: &mut ConservationWorkspace<R, D, N>,
                execution: ExecutionContext<'_>,
            ) -> Result<ConservationReport<R, N>, SciOxideError> {
                self.core.token.ensure_compatible(state.mesh_token())?;
                shape(output.len(), self.volumes().len())?;
                let report = self.core.evaluate(
                    e,
                    f,
                    time,
                    state.values(),
                    work,
                    execution,
                    scioxide_core::argument!("f"),
                )?;
                output.copy_from_slice(&work.residual);
                Ok(report)
            }
            /// Evaluate integrated oriented face fluxes into a caller slice in canonical order.
            /// Output is unchanged on failure; periodic seam aliases contain the same flux.
            /// # Errors
            /// Propagates the same failures as `residual_into`.
            #[expect(
                clippy::too_many_arguments,
                reason = "preserve the public face-flux API"
            )]
            pub fn face_fluxes_into<E: ConservationLaw<R, D, N>, F: NumericalFlux<R, E, D, N>>(
                &self,
                e: &E,
                f: &F,
                time: R,
                state: FieldView<'_, [R; N], Cell, D>,
                output: &mut [[R; N]],
                work: &mut ConservationWorkspace<R, D, N>,
                execution: ExecutionContext<'_>,
            ) -> Result<ConservationReport<R, N>, SciOxideError> {
                self.core.token.ensure_compatible(state.mesh_token())?;
                shape(output.len(), self.core.geometry.faces.len())?;
                let report = self.core.evaluate(
                    e,
                    f,
                    time,
                    state.values(),
                    work,
                    execution,
                    scioxide_core::argument!("f"),
                )?;
                for (out, face) in output.iter_mut().zip(&work.faces) {
                    *out = face.flux;
                }
                Ok(report)
            }
        }
    };
}
common_plan!(StructuredConservationPlan, CellMesh);
common_plan!(SimplexConservationPlan, SimplexMesh);
