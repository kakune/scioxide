//! Shared derivative-free observations, interpolation geometry, and trust regions.

#[path = "quadratic.rs"]
mod quadratic;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelKind {
    Linear,
    Quadratic,
}

use alloc::vec::Vec;
use core::time::Duration;

use scioxide_core::backend::{checked_add, checked_product, try_filled};
use scioxide_core::{
    Budget, EvaluationRole, ExecutionPolicy, FeasibilityTolerance, NumericScalar,
    PartialResultTrust, RankThreshold, RealScalar, SciOxideError, SolveContext,
};
use scioxide_linalg::{LinalgWorkspace, LuDecomposition, Matrix};

use crate::context::SolveSession;
use crate::utils::{commit_optimize_result, validate_finite};
use crate::{ConstrainedProblem, OptimizeResult, OptimizeTermination};

use crate::validation::invalid;
fn breakdown(message: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(scioxide_core::NumericalIssue::Breakdown, message)
}
use crate::storage::resize as fill;

/// Derivative-free controls shared by COBYLA and COBYQA. Constraints keep their
/// `<= 0` and equality conventions; the problem's box is intersected before any
/// callback.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DerivativeFreeOptions<S: RealScalar + NumericScalar<Real = S>> {
    pub(crate) initial_radius: S,
    pub(crate) final_radius: S,
    feasibility: FeasibilityTolerance<S>,
    rank: RankThreshold<S>,
    pub(crate) iterations: Budget,
    evaluations: Budget,
    pub(crate) time: Option<Duration>,
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for DerivativeFreeOptions<S> {
    fn default() -> Self {
        Self {
            initial_radius: <_ as scioxide_core::AlgebraicScalar>::ONE,
            final_radius: S::TEN.powi(-4),
            feasibility: FeasibilityTolerance::default(),
            rank: RankThreshold::default(),
            iterations: Budget::ONE.saturating_add(999),
            evaluations: Budget::ONE.saturating_add(999),
            time: None,
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> DerivativeFreeOptions<S> {
    /// Set the initial and final Euclidean trust radii in coordinate units.
    pub(crate) const fn with_radii(mut self, initial: S, final_radius: S) -> Self {
        self.initial_radius = initial;
        self.final_radius = final_radius;
        self
    }
    /// Reject nonfinite, nonpositive or reversed radii.
    pub(crate) fn validate(&self) -> Result<(), SciOxideError> {
        if !self.initial_radius.is_finite() || !self.final_radius.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "trust radii must be finite",
            ));
        }
        if self.final_radius <= <_ as scioxide_core::AlgebraicScalar>::ZERO
            || self.initial_radius < self.final_radius
        {
            return Err(invalid(
                scioxide_core::argument!("final_radius"),
                "radii must be finite with 0 < final <= initial",
            ));
        }
        Ok(())
    }
    /// Set the feasibility threshold for the normalized constraint rows.
    pub(crate) const fn with_feasibility_tolerance(
        mut self,
        value: FeasibilityTolerance<S>,
    ) -> Self {
        self.feasibility = value;
        self
    }
    /// Set the interpolation rank threshold.
    pub(crate) const fn with_rank_threshold(mut self, value: RankThreshold<S>) -> Self {
        self.rank = value;
        self
    }
    /// Limit outer model updates and geometry repairs.
    pub(crate) const fn with_max_iterations(mut self, value: Budget) -> Self {
        self.iterations = value;
        self
    }
    /// Limit objective evaluations, including interpolation and repair points.
    pub(crate) const fn with_max_evaluations(mut self, value: Budget) -> Self {
        self.evaluations = value;
        self
    }
    /// Set a cooperative deadline when the context supplies a clock.
    pub(crate) const fn with_time_limit(mut self, value: Duration) -> Self {
        self.time = Some(value);
        self
    }
    /// Initial and final trust radii.
    pub(crate) const fn radii(&self) -> (S, S) {
        (self.initial_radius, self.final_radius)
    }
    /// Feasibility threshold.
    pub(crate) const fn feasibility_tolerance(&self) -> FeasibilityTolerance<S> {
        self.feasibility
    }
    /// Interpolation rank threshold.
    pub(crate) const fn rank_threshold(&self) -> RankThreshold<S> {
        self.rank
    }
    /// Outer iteration budget.
    pub(crate) const fn max_iterations(&self) -> Budget {
        self.iterations
    }
    /// Objective evaluation budget.
    pub(crate) const fn max_evaluations(&self) -> Budget {
        self.evaluations
    }
    /// Cooperative deadline, when configured.
    pub(crate) const fn time_limit(&self) -> Option<Duration> {
        self.time
    }
}

/// Storage shape of a derivative-free solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DerivativeFreeShape {
    pub(crate) kind: ModelKind,
    pub(crate) dimension: usize,
    pub(crate) constraints: usize,
    pub(crate) equalities: usize,
    pub(crate) free: usize,
}

/// Model diagnostics at the returned point. Model gradients and errors are
/// estimates, not stationarity measures or error bounds of the original problem.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DerivativeFreeDiagnostics<S: RealScalar> {
    /// Last trust radius in original coordinate units.
    pub radius: S,
    pub resolution: S,
    pub penalty: S,
    /// Largest objective/constraint model mismatch at the last evaluated trial.
    pub model_error: Option<S>,
    /// Predicted LP merit decrease per LP radius, without a KKT claim.
    pub model_stationarity: Option<S>,
    /// Geometry steps plus full interpolation-set rebuilds, including the first.
    pub geometry_repairs: usize,
    /// Whether the final interpolation simplex passed the rank check.
    pub poised: bool,
}

/// Reusable simplex observations, linear models, and dense interpolation factor.
/// All method-owned storage is prepared before the first callback.
pub(crate) struct DerivativeFreeStorage<S: RealScalar + NumericScalar<Real = S>> {
    shape: Option<DerivativeFreeShape>,
    kind: ModelKind,
    quadratic: Option<quadratic::QuadraticWorkspace<S>>,
    observations: usize,
    n: usize,
    m: usize,
    eq: usize,
    free: Vec<usize>,
    bounds: Vec<(Option<S>, Option<S>)>,
    points: Vec<S>,
    values: Vec<S>,
    rows: Vec<S>,
    trial: Vec<S>,
    trial_rows: Vec<S>,
    gradient: Vec<S>,
    step: Vec<S>,
    candidate: Vec<S>,
    geometry_direction: Vec<S>,
    matrix: Matrix<S>,
    geometry: Matrix<S>,
    factor: LuDecomposition<S>,
    factor_work: LinalgWorkspace<'static, S>,
    rhs: Vec<S>,
    solution: Vec<S>,
    diagnostics: DerivativeFreeDiagnostics<S>,
    iterations: usize,
    lp: Option<super::lp::TrstlpWork<S>>,
    lp_shape: (usize, usize),
    lp_a: Vec<S>,
    lp_b: Vec<S>,
    row_scales: Vec<S>,
    objective_scale: S,
    penalty: S,
    resolution: S,
}
impl<S: RealScalar + NumericScalar<Real = S>> DerivativeFreeStorage<S> {
    /// Return the latest method-specific diagnostics.
    pub(crate) const fn diagnostics(&self) -> DerivativeFreeDiagnostics<S> {
        DerivativeFreeDiagnostics {
            resolution: self.resolution,
            penalty: self.penalty,
            ..self.diagnostics
        }
    }
    fn point(&self, k: usize) -> &[S] {
        &self.points[k * self.n..(k + 1) * self.n]
    }
    fn row(&self, k: usize) -> &[S] {
        &self.rows[k * self.m..(k + 1) * self.m]
    }
    fn radius(&self) -> S {
        self.diagnostics.radius
    }
    /// Whether storage was prepared for exactly `shape`.
    pub(crate) fn fits(&self, shape: DerivativeFreeShape) -> bool {
        self.shape == Some(shape)
    }
    /// Allocate storage for `shape` in a fresh value; failure keeps `self`.
    pub(crate) fn reserve(&mut self, shape: DerivativeFreeShape) -> Result<(), SciOxideError> {
        let (n, m) = (shape.dimension, shape.constraints);
        let mut free = Vec::new();
        scioxide_core::backend::try_reserve(
            &mut free,
            n,
            scioxide_core::quantity!("workspace.free"),
        )?;
        let mut next = Self {
            kind: shape.kind,
            n,
            m,
            eq: shape.equalities,
            bounds: try_filled(
                n,
                (None, None),
                scioxide_core::quantity!("workspace.bounds"),
            )?,
            free,
            trial: try_filled(
                n,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("workspace.trial"),
            )?,
            trial_rows: try_filled(
                m,
                <_ as scioxide_core::AlgebraicScalar>::ZERO,
                scioxide_core::quantity!("workspace.trial_rows"),
            )?,
            ..Self::default()
        };
        next.size(shape.free)?;
        next.factor_work.try_reserve(
            &scioxide_linalg::LuDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::LuWorkspaceRequest::factor(
                    next.matrix.rows(),
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential),
                ),
            )?,
        )?;
        next.factor_work.try_reserve(
            &scioxide_linalg::LuDecomposition::<S>::workspace_requirements(
                &scioxide_linalg::LuWorkspaceRequest::solve_matrix(
                    next.matrix.rows(),
                    1,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                ),
            )?,
        )?;
        next.shape = Some(shape);
        *self = next;
        Ok(())
    }
    /// Load the problem's box and reset per-call model state.
    fn load(
        &mut self,
        hard_box: &[(Option<S>, Option<S>)],
        initial_radius: S,
    ) -> Result<(), SciOxideError> {
        self.bounds.copy_from_slice(hard_box);
        self.free.clear();
        for (j, &(lo, hi)) in self.bounds.iter().enumerate() {
            if !matches!((lo, hi), (Some(l), Some(h)) if l == h) {
                self.free.push(j);
            }
        }
        let d = self.free.len();
        if self.shape.is_none_or(|shape| shape.free != d) {
            return Err(invalid(
                scioxide_core::argument!("workspace"),
                "workspace shape differs from the problem's box",
            ));
        }
        self.row_scales
            .fill(<S as scioxide_core::AlgebraicScalar>::ONE);
        self.objective_scale = <S as scioxide_core::AlgebraicScalar>::ZERO;
        self.penalty = <S as scioxide_core::AlgebraicScalar>::ONE;
        self.resolution = initial_radius;
        if let Some(q) = &mut self.quadratic {
            q.reset();
        }
        self.observations = 0;
        self.iterations = 0;
        self.diagnostics.radius = initial_radius;
        self.diagnostics.model_error = None;
        self.diagnostics.model_stationarity = None;
        self.diagnostics.geometry_repairs = 0;
        self.diagnostics.poised = d == 0;
        Ok(())
    }
    /// Size interpolation, model and subproblem buffers for `d` free coordinates.
    fn size(&mut self, d: usize) -> Result<(), SciOxideError> {
        let lp_rows = checked_add(
            checked_add(
                self.m,
                self.eq,
                scioxide_core::quantity!("trust_region_lp_rows"),
            )?,
            checked_product(&[2, d], scioxide_core::quantity!("trust_region_lp_rows"))?,
            scioxide_core::quantity!("trust_region_lp_rows"),
        )?;
        if self.lp.is_none() || self.lp_shape != (d, lp_rows) {
            self.lp = Some(super::lp::TrstlpWork::new(d, lp_rows)?);
            self.lp_shape = (d, lp_rows);
        }
        fill(
            &mut self.lp_a,
            checked_product(
                &[d, lp_rows],
                scioxide_core::quantity!("trust_region_lp_coefficients"),
            )?,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        fill(
            &mut self.lp_b,
            lp_rows,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        fill(
            &mut self.row_scales,
            self.m,
            <S as scioxide_core::AlgebraicScalar>::ONE,
        )?;
        let rows = checked_product(
            &[if self.kind == ModelKind::Linear { 1 } else { 2 }, d],
            scioxide_core::quantity!("interpolation_rows"),
        )?;
        let order = checked_product(
            &[if self.kind == ModelKind::Linear { 1 } else { 0 }, d],
            scioxide_core::quantity!("interpolation_order"),
        )?;
        self.matrix.try_resize_flat(order, order)?;
        self.factor.try_reserve(order)?;
        self.geometry.try_resize_flat(rows, d)?;
        if self.kind == ModelKind::Quadratic
            && d > 0
            && self
                .quadratic
                .as_ref()
                .is_none_or(|q| !q.matches(d, self.m))
        {
            self.quadratic = Some(quadratic::QuadraticWorkspace::new(d, self.m)?);
        }
        let vertices = checked_add(rows, 1, scioxide_core::quantity!("interpolation_vertices"))?;
        fill(
            &mut self.points,
            checked_product(
                &[vertices, self.n],
                scioxide_core::quantity!("workspace.points"),
            )?,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        fill(
            &mut self.values,
            vertices,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        fill(
            &mut self.rows,
            checked_product(
                &[vertices, self.m],
                scioxide_core::quantity!("workspace.rows"),
            )?,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        fill(
            &mut self.gradient,
            checked_product(
                &[
                    checked_add(self.m, 1, scioxide_core::quantity!("model_functions"))?,
                    d,
                ],
                scioxide_core::quantity!("workspace.gradient"),
            )?,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        fill(
            &mut self.rhs,
            order,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        fill(
            &mut self.solution,
            order,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        fill(
            &mut self.step,
            d,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        fill(
            &mut self.candidate,
            d,
            <_ as scioxide_core::AlgebraicScalar>::ZERO,
        )?;
        fill(&mut self.geometry_direction, d, S::ZERO)?;
        Ok(())
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> core::fmt::Debug for DerivativeFreeStorage<S> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("DerivativeFreeStorage")
            .field("shape", &self.shape)
            .field("observations", &self.observations)
            .field("iterations", &self.iterations)
            .finish_non_exhaustive()
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for DerivativeFreeStorage<S> {
    fn default() -> Self {
        Self {
            shape: None,
            kind: ModelKind::Linear,
            quadratic: None,
            observations: 0,
            n: 0,
            m: 0,
            eq: 0,
            free: Vec::new(),
            bounds: Vec::new(),
            points: Vec::new(),
            values: Vec::new(),
            rows: Vec::new(),
            trial: Vec::new(),
            trial_rows: Vec::new(),
            gradient: Vec::new(),
            step: Vec::new(),
            candidate: Vec::new(),
            geometry_direction: Vec::new(),
            matrix: Matrix::empty(),
            geometry: Matrix::empty(),
            factor: LuDecomposition::empty(),
            factor_work: LinalgWorkspace::new(),
            rhs: Vec::new(),
            solution: Vec::new(),
            diagnostics: DerivativeFreeDiagnostics {
                radius: <_ as scioxide_core::AlgebraicScalar>::ONE,
                resolution: <_ as scioxide_core::AlgebraicScalar>::ONE,
                penalty: <S as scioxide_core::AlgebraicScalar>::ONE,
                model_error: None,
                model_stationarity: None,
                geometry_repairs: 0,
                poised: false,
            },
            iterations: 0,
            lp: None,
            lp_shape: (0, 0),
            lp_a: Vec::new(),
            lp_b: Vec::new(),
            row_scales: Vec::new(),
            objective_scale: <S as scioxide_core::AlgebraicScalar>::ZERO,
            penalty: <S as scioxide_core::AlgebraicScalar>::ONE,
            resolution: <S as scioxide_core::AlgebraicScalar>::ONE,
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> crate::attempt::Accepted<S>
    for DerivativeFreeStorage<S>
{
    fn accepted(&self) -> Option<crate::attempt::AcceptedState<'_, S>> {
        (self.observations > 0).then(|| crate::attempt::AcceptedState {
            point: self.point(0),
            value: self.values[0],
            gradient: None,
            iterations: self.iterations,
        })
    }
}

/// COBYLA or COBYQA with an interpolation model and explicit feasibility repair.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DerivativeFree<S: RealScalar + NumericScalar<Real = S>> {
    kind: ModelKind,
    options: DerivativeFreeOptions<S>,
}
impl<S: RealScalar + NumericScalar<Real = S>> DerivativeFree<S> {
    /// Construct a solver from options the caller has validated.
    pub(crate) const fn new(options: DerivativeFreeOptions<S>, kind: ModelKind) -> Self {
        Self { options, kind }
    }
    /// Storage shape for `problem`: nonlinear rows and free coordinates of its box.
    pub(crate) fn shape(&self, problem: &ConstrainedProblem<'_, S>) -> DerivativeFreeShape {
        let constraints = problem.constraints();
        DerivativeFreeShape {
            kind: self.kind,
            dimension: problem.dim(),
            constraints: constraints.nonlinear_functions().count(),
            equalities: constraints.profile().equality_count(),
            free: constraints
                .hard_box()
                .iter()
                .filter(|&&(lo, hi)| !matches!((lo, hi), (Some(l), Some(h)) if l == h))
                .count(),
        }
    }
    /// Identify method, constraint shape, box and options for a solve epoch.
    pub(crate) fn identity(
        &self,
        problem: &ConstrainedProblem<'_, S>,
    ) -> scioxide_core::SolveIdentity {
        crate::context::local_identity(
            match self.kind {
                ModelKind::Linear => scioxide_core::algorithm_id!("optimize", "Cobyla", 3),
                ModelKind::Quadratic => scioxide_core::algorithm_id!("optimize", "Cobyqa", 3),
            },
            problem.dim(),
            &(
                &self.options,
                problem.constraints().profile(),
                problem.constraints().hard_box(),
            ),
        )
    }
    /// Run one epoch in `ctx`. Callback errors leave `out` unchanged; resource
    /// stops return the last fully evaluated center once one exists.
    pub(crate) fn minimize_in_context(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        x0: &[S],
        out: &mut OptimizeResult<S>,
        w: &mut DerivativeFreeStorage<S>,
        ctx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        w.observations = 0;
        if let Some(limit) = self.options.time {
            ctx.limit_time(limit)?;
        }
        w.load(
            problem.constraints().hard_box(),
            self.options.initial_radius,
        )?;
        out.try_reserve_without_gradient(x0.len())?;
        let session = SolveSession::new(ctx, scioxide_core::argument!("scope"));
        let answer = self.kernel(problem, x0, w, &session);
        if let Err(error) = &answer {
            session.record_execution_stop(error);
        }
        match answer {
            Ok(stop) => self.finish(out, w, session.stopped().unwrap_or(stop), &session),
            Err(error) => {
                if let Some(stop) = session.stopped()
                    && w.observations > 0
                {
                    return self.finish(out, w, stop, &session);
                }
                Err(error)
            }
        }
    }
    fn finish(
        &self,
        out: &mut OptimizeResult<S>,
        w: &DerivativeFreeStorage<S>,
        stop: OptimizeTermination,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        commit_optimize_result(
            out,
            w.point(0),
            w.values[0],
            None,
            w.iterations,
            stop,
            match self.kind {
                ModelKind::Linear => "COBYLA linear-model trust-region result",
                ModelKind::Quadratic => "COBYQA quadratic-model trust-region result",
            },
        )?;
        out.diagnostics.primal_feasibility = Some(feasibility(w.row(0), w.eq));
        out.diagnostics.bound_violation = Some(<_ as scioxide_core::AlgebraicScalar>::ZERO);
        out.method = match self.kind {
            ModelKind::Linear => {
                crate::OptimizeDiagnostics::Cobyla(crate::CobylaDiagnostics::from(w.diagnostics()))
            }
            ModelKind::Quadratic => {
                crate::OptimizeDiagnostics::Cobyqa(crate::CobyqaDiagnostics::from(w.diagnostics()))
            }
        };
        crate::utils::synchronize_metadata(out, session, PartialResultTrust::DiagnosticOnly)
    }
    /// Evaluate `x`; `None` rejects a point outside the objective domain. A
    /// `trial` with a non-finite objective or constraint value is rejected too,
    /// while the start and initial interpolation points must be finite.
    fn evaluate(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        x: &[S],
        rows: &mut [S],
        session: &SolveSession<'_, '_, '_>,
        trial: bool,
    ) -> Result<Option<S>, SciOxideError> {
        let f = problem.objective();
        if !session.invoke(EvaluationRole::Domain, || f.is_in_domain(x))? {
            return Ok(None);
        }
        let value = session.invoke(EvaluationRole::Objective, || f.eval(x))?;
        if trial && !value.is_finite() {
            return Ok(None);
        }
        validate_finite(&[value])?;
        for (slot, function) in rows
            .iter_mut()
            .zip(problem.constraints().nonlinear_functions())
        {
            *slot = session.invoke(EvaluationRole::Constraint, || function.eval(x))?;
        }
        if trial && rows.iter().any(|row| !row.is_finite()) {
            return Ok(None);
        }
        validate_finite(rows)?;
        Ok(Some(value))
    }
    fn kernel(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        x0: &[S],
        w: &mut DerivativeFreeStorage<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<OptimizeTermination, SciOxideError> {
        w.points[..w.n].copy_from_slice(x0);
        project(&mut w.points[..w.n], &w.bounds);
        let Some(start) =
            self.evaluate(problem, &w.points[..w.n], &mut w.trial_rows, session, false)?
        else {
            return Err(invalid(
                scioxide_core::argument!("x0"),
                "initial point is outside the objective domain",
            ));
        };
        w.values[0] = start;
        w.rows[..w.m].copy_from_slice(&w.trial_rows);
        w.observations = 1;
        w.resolution = self.options.initial_radius;
        session.accept(PartialResultTrust::DiagnosticOnly)?;
        if w.free.is_empty() {
            return Ok(
                if self
                    .options
                    .feasibility
                    .is_satisfied(feasibility(w.row(0), w.eq), S::ONE)?
                {
                    OptimizeTermination::Converged
                } else {
                    OptimizeTermination::Stagnation
                },
            );
        }
        self.repair(problem, w, session)?;
        let mut stalled = 0;
        loop {
            if w.iterations >= self.options.iterations.get() {
                return Ok(OptimizeTermination::IterationLimit);
            }
            if let Some(stop) = session.iteration()? {
                return Ok(stop);
            }
            w.iterations += 1;
            if !self.model(w, session)? {
                self.repair(problem, w, session)?;
                if !self.model(w, session)? {
                    w.diagnostics.radius = (S::HALF * w.radius()).max(self.options.final_radius);
                    w.resolution = w.resolution.min(w.radius());
                    if w.radius() <= self.options.final_radius {
                        return Ok(OptimizeTermination::Stagnation);
                    }
                    continue;
                }
            }
            let predicted = self.step(w, session)?;
            let dnorm = dot(&w.step, &w.step).sqrt();
            let violation = normalized_violation(w.row(0), w.eq, &w.row_scales);
            let coordinate = w.points[..w.n]
                .iter()
                .fold(S::ONE, |a, &b| a.max(NumericScalar::abs(b)));
            let feasible = self
                .options
                .feasibility
                .is_satisfied(violation, coordinate)?;
            // The model's achievable decrease per radius is a projected merit
            // criticality measure. Account for subtraction roundoff at rho_end.
            let model_gradient = w.gradient[..w.free.len()]
                .iter()
                .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |a, &b| {
                    a.max(NumericScalar::abs(b))
                });
            let critical = (S::TWO
                * S::TWO
                * S::EPSILON
                * (NumericScalar::abs(w.values[0] / w.objective_scale)
                    + w.radius() * model_gradient / w.objective_scale)
                / w.resolution)
                .max(S::SQRT_EPSILON)
                + w.resolution;
            // A linear model step shorter than half the final resolution is
            // Powell's COBYLA stop: its optimum lies within that resolution,
            // even when a nearly active constraint keeps the predicted rate high.
            let resolved = self.kind == ModelKind::Linear && dnorm < S::HALF * w.resolution;
            if w.resolution <= self.options.final_radius
                && w.radius() <= (S::ONE + S::HALF) * w.resolution
                && feasible
                && w.diagnostics.poised
                && (resolved
                    || w.diagnostics
                        .model_stationarity
                        .is_some_and(|v| v <= critical))
            {
                if self.geometry_trial(problem, w, session)? {
                    continue;
                }
                return Ok(OptimizeTermination::Converged);
            }
            let short = dnorm < S::HALF * w.resolution;
            let mut ratio = -S::ONE;
            w.trial.copy_from_slice(&w.points[..w.n]);
            for (&j, &step) in w.free.iter().zip(&w.step) {
                w.trial[j] += step;
            }
            project(&mut w.trial, &w.bounds);
            let representable = w.trial.iter().zip(w.point(0)).any(|(&a, &b)| a != b);
            if (!short || !feasible || self.kind == ModelKind::Quadratic)
                && predicted > <S as scioxide_core::AlgebraicScalar>::ZERO
                && representable
            {
                if let Some(value) = self.observe_trial(problem, w, session)? {
                    let trial_violation = normalized_violation(&w.trial_rows, w.eq, &w.row_scales);
                    let actual = (w.values[0] - value) / w.objective_scale
                        + w.penalty * (violation - trial_violation);
                    if !actual.is_finite() {
                        return Err(breakdown("nonfinite COBYLA merit reduction"));
                    }
                    ratio = actual / predicted;
                    let mut error = NumericScalar::abs(
                        (value - w.values[0] - model_change(w, 0, &w.step)) / w.objective_scale,
                    );
                    for row in 0..w.m {
                        error = error.max(NumericScalar::abs(
                            (w.trial_rows[row] - w.rows[row] - model_change(w, row + 1, &w.step))
                                / w.row_scales[row],
                        ));
                    }
                    w.diagnostics.model_error = Some(error);
                    let accepted = actual > <S as scioxide_core::AlgebraicScalar>::ZERO;
                    let replace = replacement(w, accepted);
                    self.insert_trial(w, value, replace, accepted);
                    if accepted {
                        session.accept(PartialResultTrust::DiagnosticOnly)?;
                        stalled = 0;
                    } else {
                        session.reject()?;
                        stalled += 1;
                    }
                } else {
                    session.reject()?;
                    stalled += 1;
                }
            } else {
                stalled += 1;
            }
            let old_radius = w.radius();
            w.diagnostics.radius = if ratio <= S::ONE / S::TEN {
                (S::HALF * old_radius).min(dnorm.max(S::HALF * w.resolution))
            } else if ratio <= (S::TEN - S::TWO - S::ONE) / S::TEN {
                (S::HALF * old_radius).max(dnorm)
            } else {
                (S::HALF * old_radius).max(S::TWO * dnorm)
            }
            .max(w.resolution);
            if ratio <= S::ONE / S::TEN || short {
                if !self.model(w, session)? {
                    self.repair(problem, w, session)?;
                    continue;
                }
                if self.geometry_trial(problem, w, session)? {
                    continue;
                }
                if w.radius() <= (S::ONE + S::HALF) * w.resolution {
                    if w.resolution <= self.options.final_radius {
                        if stalled > 2 * w.free.len() + 8 {
                            return Ok(OptimizeTermination::Stagnation);
                        }
                    } else {
                        let factor = w.resolution / self.options.final_radius;
                        w.resolution =
                            if factor > (S::TWO + S::TWO + S::ONE) * S::TEN * S::TEN / S::TWO {
                                w.resolution / S::TEN
                            } else if factor > S::TWO.powi(4) {
                                (w.resolution * self.options.final_radius).sqrt()
                            } else {
                                self.options.final_radius
                            };
                        w.diagnostics.radius = w.resolution;
                    }
                }
            }
        }
    }

    fn observe_trial(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        w: &mut DerivativeFreeStorage<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<Option<S>, SciOxideError> {
        if session.counts()?.provider(EvaluationRole::Objective) >= self.options.evaluations.get() {
            session.set_stopped(OptimizeTermination::EvaluationLimit);
            return Err(session.stop_error(OptimizeTermination::EvaluationLimit)?);
        }
        session.proposal()?;
        self.evaluate(problem, &w.trial, &mut w.trial_rows, session, true)
    }

    fn geometry_trial(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        w: &mut DerivativeFreeStorage<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        let d = w.free.len();
        let mut replace = 0;
        let mut worst = S::ONE;
        let local_target = if self.kind == ModelKind::Quadratic {
            let mut model = w.quadratic.take().ok_or_else(|| {
                invalid(
                    scioxide_core::argument!("w"),
                    "unprepared quadratic geometry",
                )
            })?;
            let target = model.local_geometry_target(w);
            w.quadratic = Some(model);
            target?
        } else {
            None
        };
        for k in 1..w.values.len() {
            // Repair the missing local affine direction before optimizing
            // cardinal growth elsewhere. Replacing an already useful local
            // point can otherwise cycle without increasing that local rank.
            if local_target.is_some_and(|target| target != k) {
                continue;
            }
            let distance = w
                .free
                .iter()
                .map(|&j| (w.points[k * w.n + j] - w.points[j]).powi(2))
                .sum::<S>()
                .sqrt();
            let gradient_norm = dot(
                &w.geometry.as_slice()[(k - 1) * d..k * d],
                &w.geometry.as_slice()[(k - 1) * d..k * d],
            )
            .sqrt();
            let gradient_growth = if self.kind == ModelKind::Linear {
                gradient_norm * w.radius() / (S::TWO * S::TWO)
            } else {
                S::ZERO
            };
            let mut score = gradient_growth.max(if self.kind == ModelKind::Linear {
                distance / ((S::TWO + S::ONE / S::TEN) * w.radius())
            } else if local_target == Some(k) {
                S::ONE + distance / w.radius()
            } else {
                S::ZERO
            });
            if self.kind == ModelKind::Quadratic {
                for j in 0..d {
                    w.step.fill(S::ZERO);
                    w.step[j] = w.radius();
                    let plus = NumericScalar::abs(lagrange(w, k - 1, &w.step));
                    w.step[j] = -w.radius();
                    let minus = NumericScalar::abs(lagrange(w, k - 1, &w.step));
                    score = score.max(plus.max(minus) / S::TEN);
                }
            }
            if score > worst {
                worst = score;
                replace = k;
            }
        }
        if replace == 0 {
            return Ok(false);
        }
        let mut best_volume = S::ZERO;
        let mut best_merit = S::INFINITY;
        // Geometry lines include all interpolation directions, the cardinal
        // gradient, and coordinate axes. A quadratic cardinal polynomial's
        // absolute maximum on each bounded line is among its two endpoints and
        // its stationary point; no extra function observations are required.
        for source in 0..w.values.len() + d {
            if source < w.values.len() - 1 {
                for (j, &coordinate) in w.free.iter().enumerate() {
                    w.geometry_direction[j] =
                        w.points[(source + 1) * w.n + coordinate] - w.points[coordinate];
                }
            } else if source == w.values.len() - 1 {
                w.geometry_direction
                    .copy_from_slice(&w.geometry.as_slice()[(replace - 1) * d..replace * d]);
            } else {
                w.geometry_direction.fill(S::ZERO);
                w.geometry_direction[source - w.values.len()] = S::ONE;
            }
            let norm = dot(&w.geometry_direction, &w.geometry_direction).sqrt();
            if norm == S::ZERO {
                continue;
            }
            let mut lower = -w.radius() / norm;
            let mut upper = w.radius() / norm;
            for (j, &coordinate) in w.free.iter().enumerate() {
                let direction = w.geometry_direction[j];
                if direction == S::ZERO {
                    continue;
                }
                for (endpoint, is_lower) in [
                    (w.bounds[coordinate].0, true),
                    (w.bounds[coordinate].1, false),
                ] {
                    if let Some(endpoint) = endpoint {
                        let bound = (endpoint - w.points[coordinate]) / direction;
                        if is_lower == (direction > S::ZERO) {
                            lower = lower.max(bound);
                        } else {
                            upper = upper.min(bound);
                        }
                    }
                }
            }
            let linear = dot(
                &w.geometry.as_slice()[(replace - 1) * d..replace * d],
                &w.geometry_direction,
            );
            let quadratic = lagrange(w, replace - 1, &w.geometry_direction) - linear;
            let stationary = if self.kind == ModelKind::Quadratic && quadratic != S::ZERO {
                -linear / (S::TWO * quadratic)
            } else {
                S::ZERO
            };
            for alpha in [lower, upper, stationary] {
                if alpha < lower || alpha > upper {
                    continue;
                }
                for (step, &direction) in w.step.iter_mut().zip(&w.geometry_direction) {
                    *step = alpha * direction;
                }
                let volume = NumericScalar::abs(lagrange(w, replace - 1, &w.step));
                let merit = model_change(w, 0, &w.step) / w.objective_scale
                    + w.penalty * modeled_feasibility(w, &w.step);
                if !volume.is_finite() || !merit.is_finite() {
                    return Err(breakdown("nonfinite interpolation geometry score"));
                }
                if volume > best_volume || (volume == best_volume && merit < best_merit) {
                    w.candidate.copy_from_slice(&w.step);
                    best_volume = volume;
                    best_merit = merit;
                }
            }
        }
        if best_volume <= S::EPSILON {
            return Ok(false);
        }
        let old_distance = w
            .free
            .iter()
            .map(|&j| (w.points[replace * w.n + j] - w.points[j]).powi(2))
            .sum::<S>()
            .sqrt();
        if best_volume <= S::ONE && old_distance <= (S::TWO + S::ONE / S::TEN) * w.radius() {
            return Ok(false);
        }
        w.step.copy_from_slice(&w.candidate);
        w.trial.copy_from_slice(&w.points[..w.n]);
        for (&j, &step) in w.free.iter().zip(&w.step) {
            w.trial[j] += step;
        }
        let Some(value) = self.observe_trial(problem, w, session)? else {
            return Ok(false);
        };
        let actual = (w.values[0] - value) / w.objective_scale
            + w.penalty
                * (normalized_violation(w.row(0), w.eq, &w.row_scales)
                    - normalized_violation(&w.trial_rows, w.eq, &w.row_scales));
        self.insert_trial(
            w,
            value,
            replace,
            actual > <S as scioxide_core::AlgebraicScalar>::ZERO,
        );
        w.diagnostics.geometry_repairs += 1;
        if actual > <S as scioxide_core::AlgebraicScalar>::ZERO {
            session.accept(PartialResultTrust::DiagnosticOnly)?;
        } else {
            session.reject()?;
        }
        Ok(true)
    }
    fn repair(
        &self,
        problem: &ConstrainedProblem<'_, S>,
        w: &mut DerivativeFreeStorage<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        w.diagnostics.geometry_repairs += 1;
        for k in 0..w.values.len() - 1 {
            if session.counts()?.provider(EvaluationRole::Objective)
                >= self.options.evaluations.get()
            {
                session.set_stopped(OptimizeTermination::EvaluationLimit);
                return Err(session.stop_error(OptimizeTermination::EvaluationLimit)?);
            }
            w.trial.copy_from_slice(&w.points[..w.n]);
            let j = w.free[k % w.free.len()];
            let forward = w.trial[j] + w.radius();
            let backward = w.trial[j] - w.radius();
            w.trial[j] = if w.bounds[j].1.is_none_or(|hi| forward <= hi) {
                forward
            } else if w.bounds[j].0.is_none_or(|lo| backward >= lo) {
                backward
            } else {
                let (lo, hi) = w.bounds[j];
                let upper = hi.unwrap_or(forward);
                let lower = lo.unwrap_or(backward);
                if upper - w.trial[j] >= w.trial[j] - lower {
                    upper
                } else {
                    lower
                }
            };
            if k >= w.free.len() {
                let first = w.points[((k % w.free.len()) + 1) * w.n + j] - w.points[j];
                let opposite = w.points[j] - first;
                let mut second = if w.bounds[j].0.is_none_or(|lo| opposite >= lo)
                    && w.bounds[j].1.is_none_or(|hi| opposite <= hi)
                {
                    -first
                } else {
                    S::TWO * first
                };
                if let Some(lo) = w.bounds[j].0 {
                    second = second.max(lo - w.points[j]);
                }
                if let Some(hi) = w.bounds[j].1 {
                    second = second.min(hi - w.points[j]);
                }
                if second == first {
                    second = S::HALF * first;
                }
                w.trial[j] = w.points[j] + second;
            }
            if !w.trial[j].is_finite() {
                return Err(breakdown("geometry repair produced a nonfinite point"));
            }
            session.proposal()?;
            let Some(value) =
                self.evaluate(problem, &w.trial, &mut w.trial_rows, session, false)?
            else {
                return Err(invalid(
                    scioxide_core::argument!("f"),
                    "geometry repair left the objective domain",
                ));
            };
            w.observations += 1;
            let begin = (k + 1) * w.n;
            w.points[begin..begin + w.n].copy_from_slice(&w.trial);
            w.values[k + 1] = value;
            let begin = (k + 1) * w.m;
            w.rows[begin..begin + w.m].copy_from_slice(&w.trial_rows);
        }
        Ok(())
    }
    fn model(
        &self,
        w: &mut DerivativeFreeStorage<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<bool, SciOxideError> {
        if self.kind == ModelKind::Quadratic {
            let mut model = w.quadratic.take().ok_or_else(|| {
                invalid(scioxide_core::argument!("w"), "unprepared quadratic models")
            })?;
            let result = model.fit(w, self.options.rank, session);
            w.quadratic = Some(model);
            return result;
        }
        let d = w.free.len();
        for k in 0..d {
            for (col, &j) in w.free.iter().enumerate() {
                w.matrix.as_mut_slice()[k * d + col] = w.point(k + 1)[j] - w.point(0)[j];
            }
            let size = dot(
                &w.matrix.as_slice()[k * d..(k + 1) * d],
                &w.matrix.as_slice()[k * d..(k + 1) * d],
            )
            .sqrt();
            if size <= self.options.rank.absolute() * w.radius() {
                w.diagnostics.poised = false;
                return Ok(false);
            }
            w.candidate
                .copy_from_slice(&w.matrix.as_slice()[k * d..(k + 1) * d]);
            for previous in 0..k {
                let basis = &w.geometry.as_slice()[previous * d..(previous + 1) * d];
                let projection = dot(&w.candidate, basis);
                for (value, &axis) in w.candidate.iter_mut().zip(basis) {
                    *value -= projection * axis;
                }
            }
            let independent = dot(&w.candidate, &w.candidate).sqrt();
            if !independent.is_finite() || independent <= (S::ONE / (S::TWO * S::TEN)) * size {
                w.diagnostics.poised = false;
                return Ok(false);
            }
            for (axis, &value) in w.geometry.as_mut_slice()[k * d..(k + 1) * d]
                .iter_mut()
                .zip(&w.candidate)
            {
                *axis = value / independent;
            }
        }
        let factored = session.invoke(EvaluationRole::Factorization, || {
            LuDecomposition::factor_into(
                &w.matrix,
                &scioxide_linalg::LinalgExecutionOptions::new()
                    .with_execution_policy(ExecutionPolicy::Sequential),
                &mut w.factor,
                &mut w.factor_work,
                session.execution()?,
            )
        });
        match factored {
            Ok(()) => {}
            Err(error)
                if matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::Singular { .. }
                ) =>
            {
                w.diagnostics.poised = false;
                return Ok(false);
            }
            Err(error) => return Err(error),
        }
        for row in 0..=w.m {
            for k in 0..d {
                w.rhs[k] = if row == 0 {
                    w.values[k + 1] - w.values[0]
                } else {
                    w.row(k + 1)[row - 1] - w.row(0)[row - 1]
                };
            }
            session.invoke(EvaluationRole::LinearSolve, || {
                w.factor
                    .solve_into(
                        scioxide_linalg::VectorView::from_slice(&w.rhs),
                        &scioxide_linalg::DirectSolveOptions::new(
                            scioxide_linalg::RankPolicy::Exact,
                        ),
                        scioxide_linalg::VectorViewMut::from_slice(&mut w.solution),
                        &mut w.factor_work,
                        session.execution()?,
                    )
                    .and_then(|report| report.termination().require_accepted())
            })?;
            w.gradient[row * d..(row + 1) * d].copy_from_slice(&w.solution);
        }
        for k in 0..d {
            w.rhs.fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
            w.rhs[k] = <S as scioxide_core::AlgebraicScalar>::ONE;
            session.invoke(EvaluationRole::LinearSolve, || {
                w.factor
                    .solve_into(
                        scioxide_linalg::VectorView::from_slice(&w.rhs),
                        &scioxide_linalg::DirectSolveOptions::new(
                            scioxide_linalg::RankPolicy::Exact,
                        ),
                        scioxide_linalg::VectorViewMut::from_slice(&mut w.solution),
                        &mut w.factor_work,
                        session.execution()?,
                    )
                    .map(|_| ())
            })?;
            w.geometry.as_mut_slice()[k * d..(k + 1) * d].copy_from_slice(&w.solution);
        }
        validate_finite(&w.gradient)?;
        w.diagnostics.poised = true;
        Ok(true)
    }
    fn step(
        &self,
        w: &mut DerivativeFreeStorage<S>,
        session: &SolveSession<'_, '_, '_>,
    ) -> Result<S, SciOxideError> {
        let d = w.free.len();
        if w.objective_scale == <S as scioxide_core::AlgebraicScalar>::ZERO {
            let maximum = w.gradient[..d]
                .iter()
                .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |a, &v| {
                    a.max(NumericScalar::abs(v))
                });
            w.objective_scale = if maximum > <S as scioxide_core::AlgebraicScalar>::ZERO {
                maximum
            } else {
                <S as scioxide_core::AlgebraicScalar>::ONE
            };
            for row in 0..w.m {
                let maximum = w.gradient[(row + 1) * d..(row + 2) * d]
                    .iter()
                    .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |a, &v| {
                        a.max(NumericScalar::abs(v))
                    });
                w.row_scales[row] = if maximum > <S as scioxide_core::AlgebraicScalar>::ZERO {
                    maximum
                } else {
                    let residual = NumericScalar::abs(w.rows[row]);
                    if residual > <S as scioxide_core::AlgebraicScalar>::ZERO {
                        residual
                    } else {
                        <S as scioxide_core::AlgebraicScalar>::ONE
                    }
                };
            }
        }
        let mut target = 0;
        for row in 0..w.m {
            for sign in [S::ONE, -S::ONE]
                .into_iter()
                .take(if row < w.eq { 2 } else { 1 })
            {
                for j in 0..d {
                    w.lp_a[target * d + j] =
                        sign * w.gradient[(row + 1) * d + j] / w.row_scales[row];
                }
                w.lp_b[target] = -sign * w.rows[row] / w.row_scales[row];
                target += 1;
            }
        }
        for (axis, &j) in w.free.iter().enumerate() {
            for (sign, endpoint) in [(S::ONE, w.bounds[j].1), (-S::ONE, w.bounds[j].0)] {
                w.lp_a[target * d..(target + 1) * d]
                    .fill(<S as scioxide_core::AlgebraicScalar>::ZERO);
                w.lp_a[target * d + axis] = sign;
                w.lp_b[target] = endpoint.map_or(w.radius(), |b| sign * (b - w.points[j]));
                target += 1;
            }
        }
        // Stage two is invariant under positive objective scaling. Normalize
        // its current gradient so near-stationary LPs retain relative guards.
        let gradient_norm = w.gradient[..d]
            .iter()
            .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |a, &b| {
                a.max(NumericScalar::abs(b))
            });
        let gradient_norm = if gradient_norm > <S as scioxide_core::AlgebraicScalar>::ZERO {
            gradient_norm
        } else {
            S::ONE
        };
        for (out, &g) in w.candidate.iter_mut().zip(&w.gradient[..d]) {
            *out = g / gradient_norm;
        }
        let radius = if self.kind == ModelKind::Quadratic {
            w.resolution
        } else {
            w.radius()
        };
        let lp = w
            .lp
            .as_mut()
            .ok_or_else(|| invalid(scioxide_core::argument!("w"), "unprepared trust-region LP"))?;
        w.step
            .copy_from_slice(lp.solve(&w.lp_a, &w.lp_b, radius, &w.candidate));
        validate_finite(&w.step)?;
        let before = normalized_violation(w.row(0), w.eq, &w.row_scales);
        let after = linearized_violation(w, &w.step);
        let objective = -dot(&w.gradient[..d], &w.step) / w.objective_scale;
        let improvement = before - after;
        if improvement > <S as scioxide_core::AlgebraicScalar>::ZERO
            && objective < <S as scioxide_core::AlgebraicScalar>::ZERO
        {
            w.penalty = w.penalty.max(-S::TWO * objective / improvement);
        }
        let predicted = objective + w.penalty * improvement;
        if !predicted.is_finite() || !w.penalty.is_finite() {
            return Err(breakdown("nonfinite LP merit prediction"));
        }
        w.diagnostics.model_stationarity =
            Some((predicted / radius).max(<S as scioxide_core::AlgebraicScalar>::ZERO));
        if self.kind == ModelKind::Quadratic {
            let mut model = w.quadratic.take().ok_or_else(|| {
                invalid(scioxide_core::argument!("w"), "unprepared quadratic step")
            })?;
            w.candidate.copy_from_slice(&w.step);
            let result = model.step(w, self.options.rank, session);
            let result = result.map(|prediction| {
                // Composite SQP must retain a fixed fraction of the available
                // trust-LP Cauchy decrease, even after a model factorization miss.
                for _ in 0..16 {
                    let cauchy = model.prediction_at(w, &w.candidate);
                    if cauchy > prediction * S::TEN {
                        w.step.copy_from_slice(&w.candidate);
                        return cauchy;
                    }
                    if cauchy > <S as scioxide_core::AlgebraicScalar>::ZERO {
                        break;
                    }
                    for value in &mut w.candidate {
                        *value *= S::HALF;
                    }
                }
                prediction
            });
            w.quadratic = Some(model);
            return result;
        }
        Ok(predicted.max(<S as scioxide_core::AlgebraicScalar>::ZERO))
    }
    fn insert_trial(
        &self,
        w: &mut DerivativeFreeStorage<S>,
        value: S,
        replace: usize,
        accepted: bool,
    ) {
        if accepted || replace > 0 {
            w.observations += 1;
            if let Some(q) = &mut w.quadratic {
                q.record_replacement(replace);
            }
        }
        if accepted {
            if replace > 0 {
                w.points.copy_within(0..w.n, replace * w.n);
                w.rows.copy_within(0..w.m, replace * w.m);
                w.values[replace] = w.values[0];
            }
            w.points[..w.n].copy_from_slice(&w.trial);
            w.rows[..w.m].copy_from_slice(&w.trial_rows);
            w.values[0] = value;
        } else if replace > 0 {
            w.points[replace * w.n..(replace + 1) * w.n].copy_from_slice(&w.trial);
            w.rows[replace * w.m..(replace + 1) * w.m].copy_from_slice(&w.trial_rows);
            w.values[replace] = value;
        }
    }
}

fn dot<S: RealScalar + NumericScalar<Real = S>>(a: &[S], b: &[S]) -> S {
    a.iter().zip(b).fold(
        <_ as scioxide_core::AlgebraicScalar>::ZERO,
        |sum, (&x, &y)| sum + x * y,
    )
}
fn feasibility<S: RealScalar + NumericScalar<Real = S>>(rows: &[S], eq: usize) -> S {
    rows.iter().enumerate().fold(
        <_ as scioxide_core::AlgebraicScalar>::ZERO,
        |worst, (i, &v)| {
            worst.max(if i < eq {
                v.abs()
            } else {
                v.max(<_ as scioxide_core::AlgebraicScalar>::ZERO)
            })
        },
    )
}
fn modeled_feasibility<S: RealScalar + NumericScalar<Real = S>>(
    w: &DerivativeFreeStorage<S>,
    step: &[S],
) -> S {
    (0..w.m).fold(<_ as scioxide_core::AlgebraicScalar>::ZERO, |worst, i| {
        let value = (w.row(0)[i] + model_change(w, i + 1, step)) / w.row_scales[i];
        worst.max(if i < w.eq {
            value.abs()
        } else {
            value.max(<_ as scioxide_core::AlgebraicScalar>::ZERO)
        })
    })
}
fn project<S: RealScalar + NumericScalar<Real = S>>(
    x: &mut [S],
    bounds: &[(Option<S>, Option<S>)],
) {
    for (value, &(lo, hi)) in x.iter_mut().zip(bounds) {
        if let Some(lo) = lo {
            *value = value.max(lo);
        }
        if let Some(hi) = hi {
            *value = value.min(hi);
        }
    }
}
fn normalized_violation<S: RealScalar + NumericScalar<Real = S>>(
    values: &[S],
    equalities: usize,
    scales: &[S],
) -> S {
    values.iter().zip(scales).enumerate().fold(
        <S as scioxide_core::AlgebraicScalar>::ZERO,
        |a, (i, (&v, &scale))| {
            a.max(if i < equalities {
                NumericScalar::abs(v) / scale
            } else {
                v / scale
            })
        },
    )
}

/// Replacement maximizes the simplex volume change, weighted toward old
/// distant vertices. The pole participates only when a new pole is accepted.
fn replacement<S: RealScalar + NumericScalar<Real = S>>(
    w: &DerivativeFreeStorage<S>,
    accepted: bool,
) -> usize {
    let mut replace = 0;
    // A rejected point needs a useful interpolation denominator; a tiny
    // displacement from the pole would introduce a nearly duplicate node.
    let mut score = if accepted { S::ZERO } else { S::ONE / S::TEN };
    let mut sum = <S as scioxide_core::AlgebraicScalar>::ZERO;
    for k in 1..w.values.len() {
        let value = lagrange(w, k - 1, &w.step);
        sum += value;
        let distance = w
            .free
            .iter()
            .map(|&j| (w.points[k * w.n + j] - w.points[j]).powi(2))
            .sum::<S>();
        let weighted =
            NumericScalar::abs(value) * (distance / (w.radius() * w.radius())).max(S::ONE);
        if weighted > score {
            score = weighted;
            replace = k;
        }
    }
    if accepted && NumericScalar::abs(S::ONE - sum) > score {
        replace = 0;
    }
    replace
}

fn model_change<S: RealScalar + NumericScalar<Real = S>>(
    w: &DerivativeFreeStorage<S>,
    function: usize,
    step: &[S],
) -> S {
    let d = w.free.len();
    let linear = dot(&w.gradient[function * d..(function + 1) * d], step);
    w.quadratic
        .as_ref()
        .map_or(linear, |q| linear + q.curvature(function, step) * S::HALF)
}
fn lagrange<S: RealScalar + NumericScalar<Real = S>>(
    w: &DerivativeFreeStorage<S>,
    row: usize,
    step: &[S],
) -> S {
    let d = w.free.len();
    w.quadratic.as_ref().map_or_else(
        || dot(&w.geometry.as_slice()[row * d..(row + 1) * d], step),
        |q| q.lagrange(row, step),
    )
}

fn linearized_violation<S: RealScalar + NumericScalar<Real = S>>(
    w: &DerivativeFreeStorage<S>,
    step: &[S],
) -> S {
    let d = w.free.len();
    (0..w.m).fold(S::ZERO, |worst, i| {
        let value =
            (w.rows[i] + dot(&w.gradient[(i + 1) * d..(i + 2) * d], step)) / w.row_scales[i];
        worst.max(if i < w.eq {
            NumericScalar::abs(value)
        } else {
            value
        })
    })
}
