//! Workspace storage and requirements of collocation plans.
use super::*;
use scioxide_core::{AttemptRecord, Capacity};
use scioxide_rootfind::{
    NewtonKrylovResult, NewtonKrylovWorkspace, NewtonKrylovWorkspaceRequirements,
};

/// Describes the workspace storage a [`BvpPlan`] needs for one problem on its initial
/// mesh; returned by [`BvpPlan::workspace_requirements`].
///
/// The requirements include the plan's byte limit. [`BvpWorkspace::check`] and
/// [`BvpWorkspace::try_reserve`] apply that limit to the whole capacity of the
/// workspace, including storage it grew for earlier plans or refined meshes.
#[derive(Debug, Clone)]
pub struct BvpRequirements<T: NumericScalar> {
    pub(super) layout: Layout,
    pub(super) analytic: bool,
    pub(super) nodes: usize,
    pub(super) newton: NewtonKrylovWorkspaceRequirements<T>,
    pub(super) newton_options: NewtonKrylovOptions<T::Real>,
    pub(super) bytes: usize,
    pub(super) limit: usize,
}
impl<T: NumericScalar> BvpRequirements<T> {
    /// Returns the number of mesh nodes these requirements cover.
    pub const fn nodes(&self) -> usize {
        self.nodes
    }
    /// Returns the upper bound on the workspace storage for that mesh, in bytes.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Returns the plan's limit on the workspace storage bound, in bytes.
    pub const fn byte_limit(&self) -> usize {
        self.limit
    }
}

impl<T: NumericScalar> BvpPlan<T> {
    /// Requirements of `problem` on `mesh`, which has the plan's dimensions.
    pub(super) fn requirements<P: BvpSystem<T> + ?Sized>(
        &self,
        problem: &BvpProblem<'_, T, P>,
        mesh: &[T::Real],
    ) -> Result<BvpRequirements<T>, SciOxideError> {
        let nodes = mesh.len();
        let bytes = self.storage_bound(nodes, scioxide_core::argument!("self"))?;
        let collocation = Collocation {
            plan: self,
            problem,
            mesh,
            cache: None,
            direct: None,
        };
        // Only the presence of an injected solver shapes the nonlinear storage.
        let unprepared = collocation::Unprepared;
        let direct = !matches!(self.layout.backend, BvpBackend::Krylov { .. });
        let nonlinear = collocation.nonlinear_problem(
            direct.then_some(&unprepared as &dyn scioxide_rootfind::NonlinearLinearSolver<T>),
        );
        Ok(BvpRequirements {
            layout: self.layout,
            analytic: problem.jacobian.is_some(),
            nodes,
            newton: self.newton.workspace_requirements(&nonlinear)?,
            newton_options: self.newton.options().clone(),
            bytes,
            limit: self.max_bytes,
        })
    }
}

/// Active-mesh numerical storage of one workspace.
pub(super) struct Storage<T: NumericScalar> {
    shape: Option<(Layout, bool)>,
    pub(super) capacity_nodes: usize,
    pub(super) mesh: Vec<T::Real>,
    pub(super) next_mesh: Vec<T::Real>,
    pub(super) point: Vec<T>,
    pub(super) next_point: Vec<T>,
    pub(super) projected: Vec<T>,
    pub(super) assessment: BvpSolution<T>,
    pub(super) assessment_scratch: Vec<T>,
    pub(super) interpolation: Vec<T>,
    pub(super) boundary: Vec<T>,
    pub(super) evaluations: RefCell<collocation::EvaluationCache<T>>,
    pub(super) direct: Option<DirectCorrection<T>>,
    pub(super) nonlinear_scope: scioxide_core::backend::PartitionedSolve,
    pub(super) newton: NewtonKrylovWorkspace<T>,
    pub(super) result: Option<NewtonKrylovResult<T>>,
}
impl<T: NumericScalar> Storage<T> {
    fn new() -> Self {
        Self {
            shape: None,
            capacity_nodes: 0,
            mesh: Vec::new(),
            next_mesh: Vec::new(),
            point: Vec::new(),
            next_point: Vec::new(),
            projected: Vec::new(),
            assessment: BvpSolution::new(),
            assessment_scratch: Vec::new(),
            interpolation: Vec::new(),
            boundary: Vec::new(),
            evaluations: RefCell::new(collocation::EvaluationCache::new()),
            direct: None,
            nonlinear_scope: scioxide_core::backend::PartitionedSolve::default(),
            newton: NewtonKrylovWorkspace::new(),
            result: None,
        }
    }
    /// Returns whether the storage layout matches, and the node capacity after a
    /// reservation.
    fn capacity(&self, requirements: &BvpRequirements<T>) -> (bool, usize) {
        if self.shape == Some((requirements.layout, requirements.analytic)) {
            (true, self.capacity_nodes.max(requirements.nodes))
        } else {
            (false, requirements.nodes)
        }
    }
    /// Grows to meet `requirements` and sizes the buffers for its mesh. Allocates only
    /// when capacity grows.
    pub(super) fn reserve(
        &mut self,
        requirements: &BvpRequirements<T>,
    ) -> Result<(), SciOxideError> {
        let layout = requirements.layout;
        let shape = (layout, requirements.analytic);
        let m = requirements.nodes;
        let (fits, capacity) = self.capacity(requirements);
        storage_bound::<T>(
            layout,
            capacity,
            requirements.limit,
            scioxide_core::argument!("workspace"),
        )?;
        if !fits {
            self.shape = None;
            self.capacity_nodes = 0;
            self.direct = match layout.backend {
                BvpBackend::Krylov { .. } => None,
                _ => Some(DirectCorrection::new(
                    layout,
                    m,
                    capacity,
                    requirements.limit,
                )?),
            };
            self.newton = NewtonKrylovWorkspace::new();
            self.result = None;
        }
        self.capacity_nodes = capacity;
        let dim = layout.size(m)?;
        let additional = m.saturating_sub(self.mesh.len());
        scioxide_core::backend::try_reserve(
            &mut self.mesh,
            additional,
            scioxide_core::quantity!("bvp_storage"),
        )?;
        resize(&mut self.point, dim, T::ZERO)?;
        resize(&mut self.next_point, layout.size(capacity)?, T::ZERO)?;
        resize(&mut self.next_mesh, capacity, T::Real::ZERO)?;
        let assessment = &mut self.assessment;
        assessment.n = layout.n;
        resize(&mut assessment.mesh, m, T::Real::ZERO)?;
        resize(&mut assessment.values, m * layout.n, T::ZERO)?;
        resize(&mut assessment.derivatives, m * layout.n, T::ZERO)?;
        resize(&mut assessment.parameters, layout.k, T::ZERO)?;
        resize(&mut assessment.residuals, m - 1, T::Real::ZERO)?;
        resize(&mut self.assessment_scratch, layout.n, T::ZERO)?;
        resize(&mut self.interpolation, 3 * layout.n, T::ZERO)?;
        resize(&mut self.boundary, layout.n + layout.k, T::ZERO)?;
        resize(&mut self.projected, layout.n, T::ZERO)?;
        if let Some(direct) = &mut self.direct {
            direct.resize(m, capacity, requirements.limit)?;
        }
        let products = requirements.analytic && matches!(layout.backend, BvpBackend::Krylov { .. });
        self.evaluations
            .get_mut()
            .resize(layout.n, layout.k, m, products)?;
        self.newton.try_reserve(&requirements.newton)?;
        match &mut self.result {
            Some(result) => result.try_resize(dim, &requirements.newton_options)?,
            None => {
                self.result = Some(NewtonKrylovResult::new(dim, &requirements.newton_options)?);
            }
        }
        self.shape = Some(shape);
        Ok(())
    }
    /// Copies the assessed solution into `out`. Every buffer is reserved before the
    /// first write, so a failed reservation leaves the contents of `out` unchanged.
    pub(super) fn publish(
        &self,
        n: usize,
        report: BvpReport<T::Real>,
        out: &mut BvpSolution<T>,
    ) -> Result<(), SciOxideError> {
        fn reserve<V>(values: &mut Vec<V>, len: usize) -> Result<(), SciOxideError> {
            let additional = len.saturating_sub(values.len());
            scioxide_core::backend::try_reserve(
                values,
                additional,
                scioxide_core::quantity!("bvp_solution"),
            )
        }
        fn copy<V: Copy>(values: &mut Vec<V>, source: &[V]) {
            values.clear();
            values.extend_from_slice(source);
        }
        let source = &self.assessment;
        reserve(&mut out.mesh, source.mesh.len())?;
        reserve(&mut out.values, source.values.len())?;
        reserve(&mut out.parameters, source.parameters.len())?;
        reserve(&mut out.derivatives, source.derivatives.len())?;
        reserve(&mut out.residuals, source.residuals.len())?;
        copy(&mut out.mesh, &source.mesh);
        copy(&mut out.values, &source.values);
        copy(&mut out.parameters, &source.parameters);
        copy(&mut out.derivatives, &source.derivatives);
        copy(&mut out.residuals, &source.residuals);
        out.n = n;
        out.report = Some(report);
        Ok(())
    }
}

/// Holds the storage of [`BvpPlan`] solves and the record of the last attempt.
///
/// Create one with [`BvpPlan::workspace`], or with [`Self::new`] and
/// [`Self::try_reserve`]. A workspace can be used with several plans and problems. A
/// change of dimensions, backend or Jacobian choice rebuilds the Newton and linear
/// solver storage. Storage grown by mesh refinement stays for later solves.
pub struct BvpWorkspace<T: NumericScalar> {
    pub(super) storage: Storage<T>,
    pub(super) shell: Shell<BvpReport<T::Real>>,
}
impl<T: NumericScalar> Default for BvpWorkspace<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: NumericScalar> core::fmt::Debug for BvpWorkspace<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BvpWorkspace")
            .field("capacity_nodes", &self.storage.capacity_nodes)
            .field("last_attempt", &self.shell.last_attempt())
            .finish_non_exhaustive()
    }
}
impl<T: NumericScalar> BvpWorkspace<T> {
    /// Creates an empty workspace without allocating.
    #[must_use]
    pub fn new() -> Self {
        Self {
            storage: Storage::new(),
            shell: Shell::new(),
        }
    }
    /// Returns whether a solve with `requirements` can run without growing this
    /// workspace.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the storage bound of the current node capacity exceeds the
    ///   byte limit of `requirements`.
    /// - `SizeOverflow` if a size does not fit in `usize`.
    pub fn check(&self, requirements: &BvpRequirements<T>) -> Result<Capacity, SciOxideError> {
        let (fits, capacity) = self.storage.capacity(requirements);
        storage_bound::<T>(
            requirements.layout,
            capacity,
            requirements.limit,
            scioxide_core::argument!("workspace"),
        )?;
        Ok(
            if fits && self.storage.capacity_nodes >= requirements.nodes {
                Capacity::Sufficient
            } else {
                Capacity::NeedsGrowth
            },
        )
    }
    /// Grows this workspace to meet `requirements`.
    ///
    /// After a failed reservation the workspace may keep part of the new storage and
    /// stays usable.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the storage bound exceeds the byte limit of `requirements`.
    /// - `SizeOverflow` if a size does not fit in `usize`, and `AllocationFailure`.
    pub fn try_reserve(&mut self, requirements: &BvpRequirements<T>) -> Result<(), SciOxideError> {
        self.storage.reserve(requirements)
    }
    /// Returns the record of the last solve: its outcome, counts and, unless it failed,
    /// its report.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<BvpReport<T::Real>>> {
        self.shell.last_attempt()
    }
}
