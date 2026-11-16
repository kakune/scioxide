use super::*;
use scioxide_core::backend::{PartitionedSolve, with_partitioned_solves};
use scioxide_core::{EvaluationRole, ScaledNorm, SolveBudgets};

/// Strong residual and conormal traces of one scalar diffusion approximation.
/// Providers must describe the same finite field/equation throughout a call.
/// Coordinates, normals and fluxes are physical; values may be complex.
pub trait AmrResidual<R: RealScalar, V: NumericScalar<Real = R>, const D: usize>: Sync {
    /// `f + div(k grad u_h)` within an active cell; discontinuities belong on faces.
    fn volume(&self, cell: usize, point: [R; D]) -> Result<V>;
    /// `k grad(u_h) dot normal` for the specified incident cell.
    fn flux(&self, cell: usize, point: [R; D], normal: [R; D]) -> Result<V>;
    /// Prescribed conormal flux at an exterior face. Return `None` for Dirichlet
    /// boundaries; for Robin data supply `g - alpha*u_h`.
    fn boundary_flux(&self, face: usize, point: [R; D], normal: [R; D]) -> Result<Option<V>>;
}

/// Epoch-bound squared estimator contributions, in compact active-cell order.
/// The global estimate is the square root of their sum, not a certificate.
pub struct AmrIndicators<R: RealScalar, const D: usize> {
    token: MeshToken,
    volume: Vec<R>,
    interior: Vec<R>,
    boundary: Vec<R>,
    global: R,
}
impl<R: RealScalar, const D: usize> AmrIndicators<R, D> {
    /// Reserve an initially zero result for transactional repeated estimation.
    /// # Errors
    /// Propagates allocation/size failures.
    pub fn new(mesh: &AdaptiveMesh<R, D>) -> Result<Self> {
        Ok(Self {
            token: mesh.mesh_token().clone(),
            volume: try_filled(
                mesh.cell_count(),
                R::ZERO,
                scioxide_core::quantity!("volume"),
            )?,
            interior: try_filled(
                mesh.cell_count(),
                R::ZERO,
                scioxide_core::quantity!("interior"),
            )?,
            boundary: try_filled(
                mesh.cell_count(),
                R::ZERO,
                scioxide_core::quantity!("boundary"),
            )?,
            global: R::ZERO,
        })
    }
    /// Import nonnegative finite squared cell indicators from another PDE estimator.
    /// # Errors
    /// Rejects shape, negative/nonfinite values, or unrepresentable global error.
    pub fn from_squared(mesh: &AdaptiveMesh<R, D>, values: &[R]) -> Result<Self> {
        validate(
            values,
            mesh.cell_count(),
            scioxide_core::argument!("values"),
        )?;
        if values.iter().any(|&v| v < R::ZERO) {
            return Err(invalid(
                scioxide_core::argument!("values"),
                "negative squared indicator",
            ));
        }
        let mut result = Self::new(mesh)?;
        result.volume.copy_from_slice(values);
        result.global = indicator_norm(values.iter().copied(), scioxide_core::argument!("values"))?;
        Ok(result)
    }
    /// Exact mesh revision.
    pub fn mesh_token(&self) -> &MeshToken {
        &self.token
    }
    /// Squared weighted volume-residual contributions.
    pub fn volume(&self) -> &[R] {
        &self.volume
    }
    /// Squared interior contributions; each interface is shared equally.
    pub fn interior(&self) -> &[R] {
        &self.interior
    }
    /// Squared natural-boundary residual contributions.
    pub fn boundary(&self) -> &[R] {
        &self.boundary
    }
    /// Sum of the three squared contributions for each active cell.
    pub fn squared(&self) -> impl Iterator<Item = R> + '_ {
        self.volume
            .iter()
            .zip(&self.interior)
            .zip(&self.boundary)
            .map(|((&a, &b), &c)| a + b + c)
    }
    /// Global estimated energy error.
    pub fn global(&self) -> R {
        self.global
    }
}
fn indicator_norm<R: RealScalar>(
    values: impl Iterator<Item = R>,
    argument: scioxide_core::Argument,
) -> Result<R> {
    let mut norm = ScaledNorm::<R>::new();
    for v in values {
        if v < R::ZERO || !v.is_finite() {
            return Err(invalid(argument, "invalid squared estimator contribution"));
        }
        norm.add(v.sqrt())?;
    }
    norm.finish()
}

/// Explicit estimator staging and per-item child solve accounting.
pub struct AmrEstimatorWorkspace<R: RealScalar> {
    values: Vec<R>,
    children: Vec<PartitionedSolve>,
}

/// Residual estimator with `h_K²/k_K` volume weights and `h_F/max(k)` jump weights.
/// Qualified for shape-regular scalar elliptic diffusion; no coefficient-contrast
/// independent bound, quadrature certificate, or coupled-system claim is made.
/// Cells and faces are scheduled with the space's execution policy.
#[derive(Clone)]
pub struct AmrEstimator<'a, R: RealScalar, const D: usize> {
    space: &'a AmrSpace<R, D>,
    volume_weights: Vec<R>,
    face_weights: Vec<R>,
    limits: AmrLimits,
}
impl<'a, R: RealScalar, const D: usize> AmrEstimator<'a, R, D> {
    /// Prepare with default numerical settings and no resource ceiling.
    ///
    /// # Errors
    /// Returns the geometry, coefficient, allocation and configuration errors of [`Self::with_options`].
    pub fn new(space: &'a AmrSpace<R, D>, diffusion: FieldView<'_, R, Cell, D>) -> Result<Self> {
        Self::with_options(space, diffusion, AmrLimits::default())
    }

    /// Freeze strictly positive cellwise constant scalar diffusion weights.
    /// # Errors
    /// Rejects shape/epoch mismatches, nonpositive diffusion, or resource failures.
    pub fn with_options(
        space: &'a AmrSpace<R, D>,
        diffusion: FieldView<'_, R, Cell, D>,
        limits: AmrLimits,
    ) -> Result<Self> {
        space
            .mesh
            .mesh_token()
            .ensure_compatible(diffusion.mesh_token())?;
        validate(
            diffusion.values(),
            space.mesh.cell_count(),
            scioxide_core::argument!("diffusion"),
        )?;
        if diffusion.values().iter().any(|&k| k <= R::ZERO) {
            return Err(invalid(
                scioxide_core::argument!("diffusion"),
                "diffusion must be strictly positive",
            ));
        }
        let mut work = Work::new(limits, None);
        work.dofs(space.mesh.cell_count())?;
        let mut diameters = work.vector(
            space.mesh.cell_count(),
            scioxide_core::quantity!("diameters"),
        )?;
        let mut volume_weights = work.vector(
            space.mesh.cell_count(),
            scioxide_core::quantity!("volume_weights"),
        )?;
        for (cell, &k) in diffusion.values().iter().enumerate() {
            work.tick(64)?;
            let h = space.mesh.cell_diameter(cell)?;
            diameters.push(h);
            volume_weights.push(finite(
                h * (h / k),
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?);
        }
        let mut face_weights = work.vector(
            space.mesh.interfaces().len(),
            scioxide_core::quantity!("face_weights"),
        )?;
        for interface in space.mesh.interfaces() {
            work.tick(1)?;
            let a = interface.left_cell;
            let b = interface.right.map_or(a, |side| side.cell);
            face_weights.push(finite(
                diameters[a].min(diameters[b]) / diffusion.values()[a].max(diffusion.values()[b]),
                scioxide_core::NonFiniteOrigin::Intermediate,
            )?);
        }
        // Bound each evaluation's cell/face quadrature and contraction work before callbacks.
        let calls = add(
            size(
                space.mesh.cell_count(),
                space.volume.len(),
                scioxide_core::quantity!("provider_calls"),
            )?,
            size(
                size(
                    space.mesh.interfaces().len(),
                    space.facets.len(),
                    scioxide_core::quantity!("provider_calls"),
                )?,
                2,
                scioxide_core::quantity!("provider_calls"),
            )?,
            scioxide_core::quantity!("provider_calls"),
        )?;
        work.tick(size(
            calls,
            size(space.local, D + 1, scioxide_core::quantity!("work"))?,
            scioxide_core::quantity!("work"),
        )?)?;
        Ok(Self {
            space,
            volume_weights,
            face_weights,
            limits,
        })
    }
    /// Allocate reusable staging explicitly; no library allocation occurs in prepared estimation.
    /// # Errors
    /// Rejects resource/size ceilings and allocation failures.
    pub fn workspace(&self) -> Result<AmrEstimatorWorkspace<R>> {
        let mut work = Work::new(self.limits, None);
        let n = self.space.mesh.cell_count();
        let faces = self.space.mesh.interfaces().len();
        let values = work.filled(
            add(
                size(n, 3, scioxide_core::quantity!("workspace.values"))?,
                faces,
                scioxide_core::quantity!("workspace.values"),
            )?,
            R::ZERO,
            scioxide_core::quantity!("workspace.values"),
        )?;
        let mut children = work.vector(
            add(n, faces, scioxide_core::quantity!("workspace.children"))?,
            scioxide_core::quantity!("workspace.children"),
        )?;
        children.resize_with(
            add(n, faces, scioxide_core::quantity!("workspace.children"))?,
            PartitionedSolve::default,
        );
        Ok(AmrEstimatorWorkspace { values, children })
    }
    /// Estimate into a preallocated result, as one epoch of `scope`, with
    /// deterministic per-cell/face budget shares independent of worker count. Each
    /// physical interface is evaluated once. Scratch may change on failure; the
    /// result remains unchanged.
    /// # Errors
    /// Propagates provider, geometry, budget, execution and arithmetic failures.
    pub fn estimate_into<'scope, 'capability, V: NumericScalar<Real = R>>(
        &self,
        provider: &impl AmrResidual<R, V, D>,
        output: &mut AmrIndicators<R, D>,
        workspace: &mut AmrEstimatorWorkspace<R>,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<()> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "amr-estimate", 2),
            |context| self.estimate(provider, output, workspace, context),
        )
    }
    fn estimate<V: NumericScalar<Real = R>>(
        &self,
        provider: &impl AmrResidual<R, V, D>,
        output: &mut AmrIndicators<R, D>,
        workspace: &mut AmrEstimatorWorkspace<R>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<()> {
        self.space
            .mesh
            .mesh_token()
            .ensure_compatible(&output.token)?;
        let n = self.space.mesh.cell_count();
        let nf = self.space.mesh.interfaces().len();
        let items = add(n, nf, scioxide_core::quantity!("items"))?;
        let expected_values = add(
            size(n, 3, scioxide_core::quantity!("workspace.values"))?,
            nf,
            scioxide_core::quantity!("workspace.values"),
        )?;
        if workspace.values.len() != expected_values
            || workspace.children.len() != items
            || output.volume.len() != n
        {
            return Err(invalid(
                if workspace.values.len() != expected_values || workspace.children.len() != items {
                    scioxide_core::argument!("workspace")
                } else {
                    scioxide_core::argument!("output")
                },
                "estimator result/workspace shape",
            ));
        }
        context.execution().check()?;
        if items == 0 {
            output.global = R::ZERO;
            return Ok(());
        }
        let identity = context.identity();
        let plan = context.execution().plan(
            self.space.options.execution,
            items,
            scioxide_core::PlanLimits {
                work_per_item: core::num::NonZeroUsize::MIN,
                auto_threshold: core::num::NonZeroUsize::MIN,
                worker_limit: crate::validation::WORKER_LIMIT,
            },
        );
        // Item values occupy a contiguous prefix until all providers finish.
        with_partitioned_solves(
            context,
            &mut workspace.children,
            SolveBudgets::unbounded(),
            |execution, children| {
                type Slot<'a, R> =
                    PartitionSlot<(&'a mut [R], scioxide_core::backend::PartitionSolveBatch<'a>)>;
                let mut slots: [Slot<'_, R>; crate::validation::MAX_WORKERS] =
                    core::array::from_fn(|_| PartitionSlot::empty());
                let mut values = &mut workspace.values[..items];
                let mut child_tail = children;
                for partition in plan.partitions() {
                    let (part, tail) = values.split_at_mut(partition.range().len());
                    values = tail;
                    let (child, tail) = child_tail.split_at(partition.range().len())?;
                    child_tail = tail;
                    slots[partition.worker_index()] = PartitionSlot::new((part, child));
                }
                execution.execute_partitioned(
                    plan,
                    &slots[..plan.worker_count()],
                    &|partition, (part, children), _| {
                        for (i, (value, child)) in part.iter_mut().zip(children).enumerate() {
                            *value = child.run(execution, identity, |context| {
                                self.item(partition.range().start + i, provider, context)
                            })?;
                        }
                        Ok(())
                    },
                )?;

                Ok(())
            },
        )?;
        workspace.values.copy_within(n..items, 3 * n);
        workspace.values[n..3 * n].fill(R::ZERO);
        for (i, face) in self.space.mesh.interfaces().iter().enumerate() {
            context.execution().check()?;
            let value = workspace.values[3 * n + i];
            if let Some(right_side) = face.right {
                let right = right_side.cell;
                for cell in [face.left_cell, right] {
                    workspace.values[n + cell] = finite(
                        workspace.values[n + cell] + value * R::HALF,
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?;
                }
            } else {
                let slot = 2 * n + face.left_cell;
                workspace.values[slot] = finite(
                    workspace.values[slot] + value,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        let global = indicator_norm(
            (0..n).map(|i| {
                workspace.values[i] + workspace.values[n + i] + workspace.values[2 * n + i]
            }),
            scioxide_core::argument!("provider"),
        )?;
        context.execution().check()?;
        output.volume.copy_from_slice(&workspace.values[..n]);
        output.interior.copy_from_slice(&workspace.values[n..2 * n]);
        output
            .boundary
            .copy_from_slice(&workspace.values[2 * n..3 * n]);
        output.global = global;
        Ok(())
    }
    fn item<V: NumericScalar<Real = R>>(
        &self,
        item: usize,
        provider: &impl AmrResidual<R, V, D>,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<R> {
        let mut norm = ScaledNorm::<R>::new();
        let admit = |context: &mut SolveContext<'_, '_>| -> Result<()> {
            context.execution().check()?;
            context.try_consume(BudgetKind::Node, 1)?.require()?;
            context
                .record_provider(EvaluationRole::Function)?
                .require()?;
            Ok(())
        };
        let n = self.space.mesh.cell_count();
        if item < n {
            for q in &self.space.volume {
                admit(context)?;
                let value = finite(
                    provider.volume(item, self.space.physical(item, q.point)?)?,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
                let w = finite(
                    q.weight * self.space.cells[item].map.determinant * self.volume_weights[item],
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
                norm.add(finite(
                    value.abs() * w.sqrt(),
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?)?;
            }
        } else {
            let i = item - n;
            let face = &self.space.mesh.interfaces()[i];
            for q in 0..self.space.facets.len() {
                let sample = self
                    .space
                    .face_sample(i, q, scioxide_core::argument!("self"))?;
                let mismatch = if let Some(right_side) = face.right {
                    let right = right_side.cell;
                    admit(context)?;
                    let left_flux = finite(
                        provider.flux(face.left_cell, sample.point, sample.normal)?,
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?;
                    admit(context)?;
                    finite(
                        left_flux
                            - finite(
                                provider.flux(right, sample.point, sample.normal)?,
                                scioxide_core::NonFiniteOrigin::Intermediate,
                            )?,
                        scioxide_core::NonFiniteOrigin::Intermediate,
                    )?
                } else {
                    admit(context)?;
                    if let Some(flux) =
                        provider.boundary_flux(face.left_face, sample.point, sample.normal)?
                    {
                        admit(context)?;
                        finite(
                            finite(flux, scioxide_core::NonFiniteOrigin::Intermediate)?
                                - finite(
                                    provider.flux(face.left_cell, sample.point, sample.normal)?,
                                    scioxide_core::NonFiniteOrigin::Intermediate,
                                )?,
                            scioxide_core::NonFiniteOrigin::Intermediate,
                        )?
                    } else {
                        V::ZERO
                    }
                };
                norm.add(finite(
                    mismatch.abs()
                        * finite(
                            sample.weight * self.face_weights[i],
                            scioxide_core::NonFiniteOrigin::Intermediate,
                        )?
                        .sqrt(),
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?)?;
            }
        }
        let value = norm.finish()?;
        finite(value * value, scioxide_core::NonFiniteOrigin::Intermediate)
    }
}

/// Deterministic bulk refinement and optional complete-family coarsening policy.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct AmrMarking<R: RealScalar> {
    refine_fraction: R,
    coarsen_threshold: Option<R>,
}
impl<R: RealScalar> AmrMarking<R> {
    /// Refine the bulk `refine_fraction` of total squared indicator, in `(0, 1]`,
    /// without coarsening; thresholds are validated when marking.
    pub const fn new(refine_fraction: R) -> Self {
        Self {
            refine_fraction,
            coarsen_threshold: None,
        }
    }
    /// Coarsen complete families whose root-sum-square child error is at most `threshold`.
    pub const fn with_coarsen_threshold(mut self, threshold: R) -> Self {
        self.coarsen_threshold = Some(threshold);
        self
    }
    /// Bulk refinement fraction.
    pub const fn refine_fraction(&self) -> R {
        self.refine_fraction
    }
    /// Coarsening threshold, absent when coarsening is disabled.
    pub const fn coarsen_threshold(&self) -> Option<R> {
        self.coarsen_threshold
    }
}
/// Stable hierarchy IDs selected by marking; spatial owns balance and commit.
pub struct AmrMarks {
    token: MeshToken,
    refine: Vec<usize>,
    coarsen: Vec<usize>,
}
impl AmrMarks {
    /// Marked active leaf hierarchy IDs.
    pub fn refine(&self) -> &[usize] {
        &self.refine
    }
    /// Marked parent hierarchy IDs with complete active child families.
    pub fn coarsen(&self) -> &[usize] {
        &self.coarsen
    }
    /// Borrow a spatial adaptation request for this exact source epoch.
    pub fn request(&self) -> scioxide_spatial::AdaptationRequest<'_> {
        scioxide_spatial::AdaptationRequest {
            source: &self.token,
            refine: &self.refine,
            coarsen: &self.coarsen,
        }
    }
}
impl<R: RealScalar> AmrMarking<R> {
    /// Mark the smallest descending prefix reaching the bulk fraction. Equal indicators
    /// use ascending stable hierarchy IDs. Zero total error produces no refinement.
    /// Coarsening skips families containing a refinement mark; spatial validates balance.
    /// Uses `O(n log n)` work and `O(n)` temporary storage for `n` active cells,
    /// including complete-family coarsening checks.
    /// # Errors
    /// Rejects stale indicators, invalid thresholds or exhausted resource budgets.
    pub fn mark<const D: usize>(
        &self,
        mesh: &AdaptiveMesh<R, D>,
        indicators: &AmrIndicators<R, D>,
    ) -> Result<AmrMarks> {
        self.mark_with_options(mesh, indicators, AmrLimits::default())
    }
    /// Mark cells with explicit resource ceilings under sequential execution.
    /// # Errors
    /// Rejects invalid indicators, mesh identity and exceeded ceilings.
    pub fn mark_with_options<const D: usize>(
        &self,
        mesh: &AdaptiveMesh<R, D>,
        indicators: &AmrIndicators<R, D>,
        limits: AmrLimits,
    ) -> Result<AmrMarks> {
        self.mark_scoped(
            mesh,
            indicators,
            limits,
            ExecutionContext::sequential_native(),
        )
    }
    /// Mark cells inside the caller's solve scope.
    /// # Errors
    /// Rejects invalid indicators, identity, limits and stopped execution.
    pub fn mark_scoped<'scope, 'capability, const D: usize>(
        &self,
        mesh: &AdaptiveMesh<R, D>,
        indicators: &AmrIndicators<R, D>,
        limits: AmrLimits,
        scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
    ) -> Result<AmrMarks> {
        crate::attempt::scoped(
            scope,
            scioxide_core::algorithm_id!("pde", "amr-marking", 2),
            |context| self.select(mesh, indicators, limits, context),
        )
    }
    fn select<const D: usize>(
        &self,
        mesh: &AdaptiveMesh<R, D>,
        indicators: &AmrIndicators<R, D>,
        limits: AmrLimits,
        context: &mut SolveContext<'_, '_>,
    ) -> Result<AmrMarks> {
        mesh.mesh_token().ensure_compatible(&indicators.token)?;
        if !self.refine_fraction.is_finite()
            || self.refine_fraction <= R::ZERO
            || self.refine_fraction > R::ONE
            || self
                .coarsen_threshold
                .is_some_and(|x| !x.is_finite() || x < R::ZERO)
        {
            return Err(invalid(
                if !self.refine_fraction.is_finite()
                    || self.refine_fraction <= R::ZERO
                    || self.refine_fraction > R::ONE
                {
                    scioxide_core::argument!("self.refine_fraction")
                } else {
                    scioxide_core::argument!("self.coarsen_threshold")
                },
                "invalid marking thresholds",
            ));
        }
        let mut work = Work::new(limits, Some(context));
        let n = mesh.cell_count();
        work.dofs(n)?;
        let mut values = work.vector(n, scioxide_core::quantity!("values"))?;
        values.extend(indicators.squared());
        let mut order = work.vector(n, scioxide_core::quantity!("order"))?;
        order.extend(0..n);
        work.sort(n)?;
        order.sort_unstable_by(|&a, &b| {
            values[b]
                .partial_cmp(&values[a])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| mesh.active_cells()[a].cmp(&mesh.active_cells()[b]))
        });
        let mut refine = work.vector(n, scioxide_core::quantity!("refine"))?;
        let maximum = values.iter().copied().fold(R::ZERO, |a, b| a.max(b));
        if maximum > R::ZERO {
            let sum = values.iter().try_fold(R::ZERO, |sum, &v| {
                finite(
                    sum + v / maximum,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )
            })?;
            let target = self.refine_fraction * sum;
            let mut selected = R::ZERO;
            for i in order {
                work.tick(1)?;
                if selected >= target {
                    break;
                }
                refine.push(mesh.active_cells()[i]);
                selected = finite(
                    selected + values[i] / maximum,
                    scioxide_core::NonFiniteOrigin::Intermediate,
                )?;
            }
        }
        work.sort(refine.len())?;
        refine.sort_unstable();
        let mut coarsen = work.vector(n, scioxide_core::quantity!("coarsen"))?;
        if let Some(threshold) = self.coarsen_threshold {
            let mut active = work.vector(n, scioxide_core::quantity!("active_index"))?;
            active.extend(
                mesh.active_cells()
                    .iter()
                    .enumerate()
                    .map(|(i, &id)| (id, i)),
            );
            work.sort(n)?;
            active.sort_unstable_by_key(|&(id, _)| id);
            let mut parents = work.vector(n, scioxide_core::quantity!("parents"))?;
            for &id in mesh.active_cells() {
                work.tick(1)?;
                if let Some(parent) = mesh.parent(id)? {
                    parents.push(parent);
                }
            }
            work.sort(parents.len())?;
            parents.sort_unstable();
            parents.dedup();
            for parent in parents {
                let mut norm = ScaledNorm::<R>::new();
                let mut eligible = true;
                for &child in mesh.children(parent)? {
                    work.tick(search_work(refine.len()))?;
                    if refine.binary_search(&child).is_ok() {
                        eligible = false;
                        break;
                    }
                    work.tick(search_work(active.len()))?;
                    let Ok(at) = active.binary_search_by_key(&child, |&(id, _)| id) else {
                        eligible = false;
                        break;
                    };
                    norm.add(values[active[at].1].sqrt())?;
                }
                if eligible && norm.finish()? <= threshold {
                    coarsen.push(parent);
                }
            }
        }
        Ok(AmrMarks {
            token: mesh.mesh_token().clone(),
            refine,
            coarsen,
        })
    }
}
