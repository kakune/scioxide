use super::*;
use alloc::boxed::Box;
use core::cell::RefCell;
use scioxide_core::{
    ConvergenceDiagnostics, ConvergenceReason, Preconditioner, ResidualOperator, ResidualTolerance,
};
use scioxide_linalg::Preconditioning;
use scioxide_linalg::{
    IterativeLinearSolver, IterativeWorkspace, IterativeWorkspaceRequest, LinearSolveTermination,
    Vector, VectorView,
};
use scioxide_ode::OdeSystem;

/// Explicitly prepared storage, saddle solver and preconditioner for constrained
/// velocity evolution.
///
/// [`Self::new`] installs restarted GMRES and the library velocity-mass inverse;
/// `with_*` replaces either for every later binding.
pub struct IncompressibleTimeWorkspace<R: RealScalar> {
    state: Vec<R>,
    residual: Vec<R>,
    scratch: Vec<R>,
    rhs: Vector<R>,
    solution: Vector<R>,
    linear: IterativeWorkspace<R>,
    solver: Box<dyn IterativeLinearSolver<R> + Send>,
    preconditioner: Box<dyn Preconditioner<R> + Send>,
    residual_tolerance: ResidualTolerance<R>,
    warm_acceleration: Vec<R>,
    warm_projection: Vec<R>,
}
impl<R: RealScalar> core::fmt::Debug for IncompressibleTimeWorkspace<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("IncompressibleTimeWorkspace")
            .field("unknowns", &self.state.len())
            .field("solver", &self.solver.algorithm_id())
            .finish_non_exhaustive()
    }
}
impl<R: RealScalar> IncompressibleTimeWorkspace<R>
where
    IncompressiblePreconditioner<R>: Send,
{
    /// Prepare GMRES, the library mass-saddle inverse and all buffers under
    /// `options`; its residual threshold also bounds the independently checked
    /// saddle residual.
    /// # Errors
    /// Returns sparse setup, provider, resource and allocation failures.
    pub fn new<const D: usize>(
        plan: &StokesPlan<R, R, D>,
        options: &IncompressibleSolveOptions<R>,
    ) -> Result<Self, SciOxideError> {
        let n = plan.unknowns();
        Ok(Self {
            state: try_filled(n, R::ZERO, scioxide_core::quantity!("state"))?,
            residual: try_filled(n, R::ZERO, scioxide_core::quantity!("residual"))?,
            scratch: try_filled(
                add(n, plan.work(), scioxide_core::quantity!("scratch"))?,
                R::ZERO,
                scioxide_core::quantity!("scratch"),
            )?,
            rhs: Vector::zeros(n)?,
            solution: Vector::zeros(n)?,
            linear: IterativeWorkspace::new(),
            solver: Box::new(options.solver::<R>()),
            preconditioner: Box::new(IncompressiblePreconditioner::new(
                plan,
                IncompressibleSaddle::VelocityMass,
                *options.preconditioner(),
            )?),
            residual_tolerance: options.residual(),
            warm_acceleration: try_filled(
                n,
                R::ZERO,
                scioxide_core::quantity!("warm_acceleration"),
            )?,
            warm_projection: try_filled(n, R::ZERO, scioxide_core::quantity!("warm_projection"))?,
        })
    }
}
impl<R: RealScalar> IncompressibleTimeWorkspace<R> {
    /// Solve saddles with `solver`, which must accept nonsymmetric indefinite systems.
    #[must_use]
    pub fn with_linear_solver(mut self, solver: Box<dyn IterativeLinearSolver<R> + Send>) -> Self {
        self.solver = solver;
        self
    }
    /// Precondition saddles with `preconditioner` instead of the library inverse.
    #[must_use]
    pub fn with_preconditioner(
        mut self,
        preconditioner: Box<dyn Preconditioner<R> + Send>,
    ) -> Self {
        self.preconditioner = preconditioner;
        self
    }
}
type Forcing<'a, R, const D: usize> =
    &'a (dyn Fn([R; D], R) -> Result<[R; D], SciOxideError> + Sync);
/// Borrowed constrained-velocity ODE on fixed meshes with static velocity data.
///
/// State coordinates are unconstrained P2 velocity entries in packed order.
/// Pressure is recovered by the mass/divergence saddle solve, not integrated.
/// Use explicit ODE methods; no implicit Jacobian capability is advertised.
/// Managed ODE stepping applies viscous and convective restrictions. Direct
/// steppers must apply [`OdeSystem::step_limit`] themselves. Continuity restoration
/// gives `d′ = -d/τ + e`, where `τ` is the capped viscous limit and `e` is the
/// saddle residual. Thus the long-time defect is bounded by `τ * ||e||`, plus
/// integration rounding, rather than accumulating over the number of steps.
pub struct IncompressibleSystem<'a, R: RealScalar, const D: usize> {
    plan: &'a StokesPlan<R, R, D>,
    nonlinear: Option<&'a NavierStokesPlan<R, D>>,
    work: RefCell<&'a mut IncompressibleTimeWorkspace<R>>,
    forcing: Option<Forcing<'a, R, D>>,
    restoration_step: R,
}
impl<'a, R: RealScalar, const D: usize> IncompressibleSystem<'a, R, D> {
    /// Bind unsteady Stokes to the workspace's saddle solver and preconditioner.
    /// `maximum_step` is an additional positive finite step cap; `forcing` adds a
    /// time-dependent body force per volume. Saddle Krylov storage grows here.
    /// # Errors
    /// Rejects invalid step caps, mismatched buffers or preconditioner dimensions,
    /// unusable geometry/scaling and allocation failures.
    pub fn stokes(
        plan: &'a StokesPlan<R, R, D>,
        work: &'a mut IncompressibleTimeWorkspace<R>,
        forcing: Option<Forcing<'a, R, D>>,
        maximum_step: R,
    ) -> Result<Self, SciOxideError> {
        if !maximum_step.is_finite()
            || maximum_step <= R::ZERO
            || work.state.len() != plan.unknowns()
            || work.scratch.len()
                < add(
                    plan.unknowns(),
                    plan.work(),
                    scioxide_core::quantity!("scratch"),
                )?
            || work.preconditioner.dimension() != plan.unknowns()
        {
            return Err(invalid(
                if !maximum_step.is_finite() || maximum_step <= R::ZERO {
                    scioxide_core::argument!("maximum_step")
                } else {
                    scioxide_core::argument!("work")
                },
                "invalid transient flow step cap or workspace",
            ));
        }
        let IncompressibleTimeWorkspace {
            linear,
            solver,
            preconditioner,
            ..
        } = &mut *work;
        let request = IterativeWorkspaceRequest::operator(&plan.mass_system())?
            .with_preconditioner(Preconditioning::Inverse(&**preconditioner))?;
        linear.try_reserve(&solver.workspace_requirements(&request)?)?;
        Ok(Self {
            plan,
            nonlinear: None,
            restoration_step: Self::viscous_limit(plan)?.min(maximum_step),
            work: RefCell::new(work),
            forcing,
        })
    }
    /// Bind conservative nonlinear flow to the same constrained time interface.
    /// # Errors
    /// Returns the [`Self::stokes`] binding failures.
    pub fn navier_stokes(
        plan: &'a NavierStokesPlan<R, D>,
        work: &'a mut IncompressibleTimeWorkspace<R>,
        forcing: Option<Forcing<'a, R, D>>,
        maximum_step: R,
    ) -> Result<Self, SciOxideError> {
        let mut system = Self::stokes(plan.stokes(), work, forcing, maximum_step)?;
        system.nonlinear = Some(plan);
        Ok(system)
    }
    fn viscous_limit(plan: &StokesPlan<R, R, D>) -> Result<R, SciOxideError> {
        // P2 inverse-estimate safety: 1/(4 D p^4), p=2, using mapped minimum altitudes.
        let factor = R::try_from_usize_exact(64 * D)?;
        let mut limit = R::INFINITY;
        for &h in &plan.lengths {
            limit = limit.min(
                scioxide_core::checked_float_product(
                    h,
                    scioxide_core::checked_float_quotient(h, plan.viscosity / plan.density)?,
                )? / factor,
            );
        }
        if !limit.is_finite() || limit <= R::ZERO {
            return Err(invalid(
                scioxide_core::argument!("plan"),
                "unrepresentable viscous step restriction",
            ));
        }
        Ok(limit)
    }
    /// Full packed velocity indices represented by the ODE coordinates.
    pub fn free_velocity_indices(&self) -> &[usize] {
        &self.plan.free
    }
    fn expand(&self, y: &[R], state: &mut [R]) -> Result<(), SciOxideError> {
        if y.len() != self.dimension() || y.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                scioxide_core::argument!("y"),
                "transient velocity shape or nonfinite state",
            ));
        }
        state.fill(R::ZERO);
        for (i, v) in state
            .iter_mut()
            .enumerate()
            .take(self.plan.velocity_unknowns())
        {
            *v = self.plan.prescribed(i).unwrap_or(R::ZERO);
        }
        for (&i, &v) in self.plan.free.iter().zip(y) {
            state[i] = v;
        }
        Ok(())
    }
    fn constraint(&self, residual: &[R], state: &[R]) -> Result<(), SciOxideError> {
        let p = self.plan;
        let defect =
            VectorView::from_slice(&residual[p.velocity_unknowns()..p.physical()]).norm2()?;
        let scale = VectorView::from_slice(&state[..p.velocity_unknowns()]).norm2()?;
        if defect > p.compatibility.absolute() + p.compatibility.relative() * scale {
            return Err(invalid(
                scioxide_core::argument!("y"),
                "velocity violates the discrete divergence constraint; explicitly project initial data",
            ));
        }
        Ok(())
    }
    fn solve(
        &self,
        w: &mut IncompressibleTimeWorkspace<R>,
        ctx: &mut SolveContext<'_, '_>,
        projection: bool,
    ) -> Result<(), SciOxideError> {
        if !ctx
            .record_provider(EvaluationRole::LinearSolve)?
            .is_granted()
        {
            return Err(SciOxideError::convergence_failure(
                ConvergenceReason::EvaluationLimit,
                ConvergenceDiagnostics::new().with_counts(ctx.counts()),
                "flow saddle solve budget exhausted",
            ));
        }
        w.solution.as_mut_slice().copy_from_slice(if projection {
            &w.warm_projection
        } else {
            &w.warm_acceleration
        });
        let solver = &*w.solver;
        // The saddle solve runs in a child epoch of the ODE right-hand-side context.
        let report = solver.solve_into(
            &self.plan.mass_system(),
            w.rhs.view(),
            Preconditioning::Inverse(&*w.preconditioner),
            w.solution.view_mut(),
            &mut w.linear,
            (&mut *ctx).into(),
        )?;
        let termination = report.termination();
        if let LinearSolveTermination::ExecutionStopped(reason) = termination {
            return Err(SciOxideError::execution_stopped(reason));
        }
        solver.require_success(&report)?;
        if !ctx
            .record_provider(EvaluationRole::LinearOperator)?
            .is_granted()
        {
            return Err(SciOxideError::convergence_failure(
                ConvergenceReason::EvaluationLimit,
                ConvergenceDiagnostics::new().with_counts(ctx.counts()),
                "saddle residual check budget exhausted",
            ));
        }
        self.plan
            .mass_system()
            .apply_into_scoped(
                w.solution.as_slice(),
                &mut w.residual,
                &mut w.scratch,
                &mut scioxide_core::OperatorContext::new(ctx),
            )
            .and_then(scioxide_core::Invocation::require)?;
        for (v, &b) in w.residual.iter_mut().zip(w.rhs.as_slice()) {
            *v = finite(*v - b, NonFiniteOrigin::Intermediate)?;
        }
        let norm = VectorView::from_slice(&w.residual).norm2_scoped(
            &scioxide_linalg::LinalgExecutionOptions::new(),
            ctx.execution(),
        )?;
        let scale = w.rhs.view().norm2_scoped(
            &scioxide_linalg::LinalgExecutionOptions::new(),
            ctx.execution(),
        )?;
        if !w
            .residual_tolerance
            .tolerance()
            .is_residual_ok(norm, scale.max(R::ONE))?
        {
            return Err(SciOxideError::convergence_failure(
                ConvergenceReason::NestedSolver,
                ConvergenceDiagnostics::new().with_counts(ctx.counts()),
                "flow saddle true residual exceeds tolerance",
            ));
        }
        Ok(())
    }
    fn evaluate(
        &self,
        time: R,
        y: &[R],
        w: &mut IncompressibleTimeWorkspace<R>,
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if !time.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("time"),
                "nonfinite flow time",
            ));
        }
        self.expand(y, &mut w.state)?;
        if let Some(n) = self.nonlinear {
            n.residual_into_scoped(&w.state, &mut w.residual, &mut w.scratch, ctx)?;
        } else {
            self.plan
                .apply_into_scoped(
                    &w.state,
                    &mut w.residual,
                    &mut w.scratch,
                    &mut scioxide_core::OperatorContext::new(ctx),
                )
                .and_then(scioxide_core::Invocation::require)?;
            for (v, &rhs) in w.residual.iter_mut().zip(self.plan.rhs()) {
                *v = finite(*v - rhs, scioxide_core::NonFiniteOrigin::Intermediate)?;
            }
        }
        w.rhs.as_mut_slice().fill(R::ZERO);
        for (rhs, &defect) in w.rhs.as_mut_slice()
            [self.plan.velocity_unknowns()..self.plan.physical()]
            .iter_mut()
            .zip(&w.residual[self.plan.velocity_unknowns()..self.plan.physical()])
        {
            *rhs = finite(
                -defect / self.restoration_step,
                NonFiniteOrigin::Intermediate,
            )?;
        }
        for &i in &self.plan.free {
            w.rhs.as_mut_slice()[i] = -w.residual[i];
        }
        if let Some(force) = self.forcing {
            let p = self.plan;
            let l = p.space.order.nodes(D);
            for s in &p.samples.volume {
                admit(ctx, true)?;
                let values = force(s.point, time)?;
                for (a, &value) in values.iter().enumerate() {
                    finite(value, scioxide_core::NonFiniteOrigin::Intermediate)?;
                    for i in 0..l {
                        let row = p.global(s.cell, a * l + i);
                        if p.prescribed(row).is_none() {
                            w.rhs.as_mut_slice()[row] = finite(
                                w.rhs.as_slice()[row] + s.weight * p.basis(s)[i] * value,
                                scioxide_core::NonFiniteOrigin::Intermediate,
                            )?;
                        }
                    }
                }
            }
        }
        w.rhs.as_mut_slice()[self.plan.physical()..].copy_from_slice(&self.plan.gauge_targets);
        self.solve(w, ctx, false)
    }
    /// Validate initial discrete incompressibility without changing state.
    /// # Errors
    /// Rejects incompatible data, shape, reentrant workspace use or arithmetic failure.
    pub fn validate_velocity(&self, y: &[R]) -> Result<(), SciOxideError> {
        let mut w = self.work.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("self"),
                "flow time workspace already borrowed",
            )
        })?;
        let w = &mut **w;
        self.expand(y, &mut w.state)?;
        self.plan
            .apply_into(&w.state, &mut w.residual, &mut w.scratch)?;
        for (v, &b) in w.residual.iter_mut().zip(self.plan.rhs()) {
            *v = finite(*v - b, scioxide_core::NonFiniteOrigin::Intermediate)?;
        }
        self.constraint(&w.residual, &w.state)
    }
    /// Explicit mass-orthogonal projection of initial free velocity coordinates.
    /// Caller output changes only after a converged, independently checked solve.
    /// # Errors
    /// Returns shape, provider, compatibility, budget and arithmetic failures.
    pub fn project_velocity_into(
        &self,
        y: &[R],
        out: &mut [R],
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if out.len() != self.dimension() {
            return Err(invalid(
                scioxide_core::argument!("out"),
                "projection output shape",
            ));
        }
        let mut w = self.work.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("self"),
                "flow time workspace already borrowed",
            )
        })?;
        let w = &mut **w;
        self.expand(y, &mut w.state)?;
        self.plan
            .mass_system()
            .apply_into_scoped(
                &w.state,
                w.rhs.as_mut_slice(),
                &mut w.scratch,
                &mut scioxide_core::OperatorContext::new(ctx),
            )
            .and_then(scioxide_core::Invocation::require)?;
        let p = self.plan;
        w.rhs.as_mut_slice()[p.velocity_unknowns()..p.physical()]
            .copy_from_slice(&p.rhs()[p.velocity_unknowns()..p.physical()]);
        w.rhs.as_mut_slice()[p.physical()..].fill(R::ZERO);
        self.solve(w, ctx, true)?;
        w.state.copy_from_slice(w.solution.as_slice());
        w.state[p.velocity_unknowns()..].fill(R::ZERO);
        p.apply_into_scoped(
            &w.state,
            &mut w.residual,
            &mut w.scratch,
            &mut scioxide_core::OperatorContext::new(ctx),
        )
        .and_then(scioxide_core::Invocation::require)?;
        for (v, &b) in w.residual.iter_mut().zip(p.rhs()) {
            *v = finite(*v - b, scioxide_core::NonFiniteOrigin::Intermediate)?;
        }
        self.constraint(&w.residual, &w.state)?;
        ctx.execution().check()?;
        w.warm_projection.copy_from_slice(w.solution.as_slice());
        for (v, &i) in out.iter_mut().zip(&p.free) {
            *v = w.state[i];
        }
        Ok(())
    }
    /// Recover physical velocity and pressure at this exact time/state.
    /// Gauge multipliers are included at the end of the packed output.
    /// # Errors
    /// Failed nested work leaves output unchanged.
    pub fn recover_into(
        &self,
        time: R,
        y: &[R],
        out: &mut [R],
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if out.len() != self.plan.unknowns() {
            return Err(invalid(
                scioxide_core::argument!("out"),
                "flow recovery output shape",
            ));
        }
        let mut w = self.work.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("self"),
                "flow time workspace already borrowed",
            )
        })?;
        let w = &mut **w;
        self.evaluate(time, y, w, ctx)?;
        ctx.execution().check()?;
        w.warm_acceleration.copy_from_slice(w.solution.as_slice());
        let n = self.plan.velocity_unknowns();
        out[..n].copy_from_slice(&w.state[..n]);
        out[n..].copy_from_slice(&w.solution.as_slice()[n..]);
        Ok(())
    }
}
impl<R: RealScalar, const D: usize> OdeSystem<R> for IncompressibleSystem<'_, R, D> {
    fn dimension(&self) -> usize {
        self.plan.free.len()
    }
    fn rhs(
        &self,
        time: R,
        y: &[R],
        out: &mut [R],
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        if out.len() != self.dimension() {
            return Err(invalid(
                scioxide_core::argument!("out"),
                "flow ODE output shape",
            ));
        }
        let mut w = self.work.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("self"),
                "flow time workspace already borrowed",
            )
        })?;
        let w = &mut **w;
        self.evaluate(time, y, w, ctx)?;
        ctx.execution().check()?;
        w.warm_acceleration.copy_from_slice(w.solution.as_slice());
        for (v, &i) in out.iter_mut().zip(&self.plan.free) {
            *v = w.solution.as_slice()[i];
        }
        Ok(())
    }
    fn step_limit(
        &self,
        time: R,
        state: &[R],
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<R>, SciOxideError> {
        ctx.execution().check()?;
        if !time.is_finite() {
            return Err(invalid(
                scioxide_core::argument!("time"),
                "nonfinite flow time",
            ));
        }
        let mut w = self.work.try_borrow_mut().map_err(|_| {
            invalid(
                scioxide_core::argument!("work"),
                "flow time workspace already borrowed",
            )
        })?;
        self.expand(state, &mut w.state)?;
        let mut limit = self.restoration_step;
        if self.nonlinear.is_some() {
            let factor = R::try_from_usize_exact(10 * D)?;
            for sample in &self.plan.samples.volume {
                admit(ctx, false)?;
                let (velocity, _) = self.plan.values(sample, &w.state, false)?;
                let speed = velocity.iter().fold(R::ZERO, |s, &v| s.hypot(v));
                if speed > R::ZERO {
                    limit = limit.min(finite(
                        self.plan.lengths[sample.cell] / speed / factor,
                        NonFiniteOrigin::Intermediate,
                    )?);
                }
            }
        }
        if limit <= R::ZERO {
            return Err(invalid(
                scioxide_core::argument!("state"),
                "unrepresentable convective step restriction",
            ));
        }
        Ok(Some(limit))
    }
}
