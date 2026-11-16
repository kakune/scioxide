//! Implicit Runge–Kutta methods for stiff ODE systems.
//!
//! - [`BackwardEuler`]: fixed step, order 1, L-stable.
//! - [`ImplicitTrapezoid`]: fixed step, order 2, A-stable.
//! - [`GaussRk`]: fixed step, two-stage Gauss–Legendre, order 4, symplectic.
//! - [`RadauIia`]: adaptive, three stages, order 5, L-stable, cubic output and events.
//! - [`Sdirk`]: fixed step, three-stage singly diagonally implicit, order 3, L-stable.
//!
//! All methods solve their stages through the stiff kernel. A [`DerivativeStrategy`]
//! selects analytic, automatic or finite-difference Jacobians on dense, sparse and
//! Krylov backends (by default analytic, then automatic, then forward differences),
//! and one rate-based Newton iteration solves each stage system.
//! [`NewtonMode::Simplified`] reuses a lagged Jacobian until movement or slow
//! convergence requires a refresh; [`NewtonMode::Full`] relinearizes and refactors
//! before every correction where offered. Radau always uses simplified Newton with
//! one real and one complex `n×n` system (three complex systems for complex states).
//! Gauss solves a coupled `2n` system; SDIRK solves its stages one after the other
//! with a single factor. Fixed methods with constant analytic or automatic Jacobians
//! and a current native direct factor need one correction per stage.
//! Finite-difference Jacobians, Krylov solves and lagged factors keep the Newton
//! convergence tests. Fixed methods keep the accepted endpoint derivative for the
//! next predictor or linearization. Radau, whose endpoint is its last stage,
//! refreshes no stage derivatives after convergence and evaluates `f(tₙ, yₙ)` once
//! per accepted base. Discarded trials keep the starting derivative, and a new solve
//! epoch clears it.
//!
//! Reuse is limited to one solve epoch and requires the same derivative strategy. A
//! nonconstant Jacobian is refreshed when any component or time moves by more than
//! half of `max(1, |old|, |new|)`, or Newton needs more than two corrections with a
//! contraction rate above 0.3 with a current factor. Slow convergence with a lagged
//! factor first refreshes that factor. Failed chord iterations retry with a fresh
//! factor, then a fresh Jacobian before rejecting. Constant Jacobians skip the
//! movement refresh. Simplified coupled Newton factors tolerate effective-step ratios
//! in `[0.8, 1.2]`; the residual always uses the current equation. Radau factors and
//! its estimator always match the current step. Constant mass matrices with a
//! revision number from [`OdeSystem`] are supported by every method in this module
//! on the dense backend.
//!
//! [`DerivativeStrategy`]: scioxide_differentiate::DerivativeStrategy

use crate::safety::{scalar_stage_residual, stage_residual};

pub(crate) mod backward_euler;
pub(crate) mod gauss_rk;
pub(crate) mod implicit_trapezoid;
pub(crate) mod radau_iia;
pub(crate) mod sdirk;

pub use backward_euler::BackwardEuler;
pub use gauss_rk::GaussRk;
pub use implicit_trapezoid::ImplicitTrapezoid;
pub use radau_iia::RadauIia;
pub use sdirk::Sdirk;

use scioxide_core::{Budget, NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::{Matrix, Vector};

use crate::result::StepResult;
use crate::safety::{checked_stage_time, checked_step_endpoint, validate_finite_state};
use crate::stepper::StepRequest;
use crate::stiff::kernel::{Coupling, MassKey, StageLayout};
use crate::stiff::stage::{StageEquation, StageSolve, solve_stage};
use crate::stiff::{NewtonMode, StageConfig, StageWork};
use crate::traits::{
    OdeSystem, evaluate_rhs_staged, evaluate_rhs_staged_slice, validate_provider_vector,
    validate_stage_buffers,
};
use crate::workspace::ImplicitOdeWorkspace;
use crate::{OdeStageBackend, StageAccuracy};

/// Stage backend, Jacobian source and Newton policy of one method.
#[derive(Clone)]
pub(crate) struct ImplicitConfig<
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
> {
    pub(crate) stage: StageConfig,
    scalar: core::marker::PhantomData<fn() -> (S, T)>,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> ImplicitConfig<S, T> {
    pub(crate) const fn new() -> Self {
        Self {
            stage: StageConfig::new(),
            scalar: core::marker::PhantomData,
        }
    }

    pub(crate) fn backend(&mut self, backend: OdeStageBackend) {
        self.stage.backend = backend;
    }

    pub(crate) fn strategy(&mut self, strategy: DerivativeStrategy) {
        self.stage.strategy = strategy;
    }

    /// Whether `workspace` already serves `stages` simultaneous stage blocks.
    pub(crate) fn fits(
        &self,
        dimension: usize,
        stages: usize,
        provided: bool,
        workspace: &ImplicitOdeWorkspace<S, T>,
    ) -> bool {
        workspace.fits(dimension, self.stage.shape(stages, provided))
    }

    /// Prepare `workspace` for `stages` simultaneous stage blocks.
    pub(crate) fn prepare(
        &self,
        dimension: usize,
        stages: usize,
        provided: bool,
        workspace: &mut ImplicitOdeWorkspace<S, T>,
    ) -> Result<(), SciOxideError> {
        workspace.prepare(dimension, self.stage.shape(stages, provided))
    }

    pub(crate) fn debug(&self, name: &str, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct(name)
            .field("backend", &self.stage.backend)
            .field("derivative_strategy", &self.stage.strategy)
            .field("newton_mode", &self.stage.mode)
            .finish_non_exhaustive()
    }
}

/// One native implicit row; exceptional arithmetic replays original operands.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn mass_row_residual<T: NumericScalar, I>(
    mass: Option<&Matrix<T>>,
    row: usize,
    candidate: &[T],
    base: &[T],
    step: T::Real,
    terms: I,
    forcing: T,
) -> Result<T, SciOxideError>
where
    I: IntoIterator<Item = (T::Real, T)>,
    I::IntoIter: Clone,
{
    let one = <T::Real as scioxide_core::AlgebraicScalar>::ONE;
    match mass {
        Some(mass) => {
            let n = candidate.len();
            stage_residual(
                &mass.as_slice()[row * n..(row + 1) * n],
                one,
                candidate,
                base,
                step,
                terms,
                forcing,
            )
        }
        None => scalar_stage_residual(one, candidate[row], base[row], step, terms, forcing),
    }
}

/// One-stage θ-method description: backward Euler (θ = 1) or the trapezoid rule.
#[derive(Clone, Copy)]
pub(crate) struct EndpointMethod<'a, S: RealScalar> {
    pub(crate) stage: &'a StageConfig,
    pub(crate) implicit_weight: S,
    pub(crate) max_iterations: Budget,
}

/// The endpoint equation `M(x − y) − h((1−θ)f(y) + θf(x)) = 0`.
struct Endpoint<'a, S: RealScalar, T: NumericScalar<Real = S>, P: ?Sized> {
    system: &'a P,
    request: StepRequest<'a, S, T>,
    time: S,
    step: S,
    implicit: S,
    old: &'a [T],
    current: &'a [T],
    rhs: &'a mut Vector<T>,
    candidate: &'a mut Vector<T>,
}

impl<S, T, P> StageEquation<S, T> for Endpoint<'_, S, T, P>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    fn evaluations(&self) -> usize {
        1
    }

    fn residual(
        &mut self,
        mass: Option<&Matrix<T>>,
        iterate: &[T],
        residual: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        validate_stage_buffers(iterate, residual, self.old.len())?;
        evaluate_rhs_staged_slice(
            self.system,
            self.time,
            iterate,
            self.rhs,
            self.candidate,
            self.request,
            cx,
        )?;
        if mass.is_none() && self.implicit == S::ONE {
            return crate::safety::implicit_stage_rows(
                iterate,
                self.old,
                self.step,
                &[S::ONE],
                core::slice::from_ref(&*self.rhs),
                residual,
            );
        }
        let explicit = S::ONE - self.implicit;
        for (row, output) in residual.iter_mut().enumerate() {
            *output = mass_row_residual(
                mass,
                row,
                iterate,
                self.old,
                self.step,
                [
                    (explicit, self.current[row]),
                    (self.implicit, self.rhs.as_slice()[row]),
                ],
                T::ZERO,
            )?;
        }
        Ok(())
    }

    fn point<'b>(&'b self, iterate: &'b [T], _: usize) -> (S, &'b [T], &'b [T]) {
        (self.time, iterate, self.rhs.as_slice())
    }
}

/// One step of a θ-method on any stage backend.
#[allow(clippy::too_many_arguments)]
pub(crate) fn endpoint_step<'a, S, T, System>(
    method: EndpointMethod<'_, S>,
    system: &System,
    t: S,
    y: &mut Vector<T>,
    dt: S,
    request: StepRequest<'_, S, T>,
    workspace: &mut ImplicitOdeWorkspace<S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<StepResult<'a, S, T>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    let request = StepRequest {
        accepted_step: request.accepted_step.filter(|_| !request.self_metered),
        ..request
    };
    validate_finite_state(y, system.dimension(), scioxide_core::NonFiniteOrigin::Input)?;
    let endpoint_time = checked_step_endpoint(t, dt)?;
    workspace.prepare(
        y.len(),
        method.stage.shape(1, request.provider.is_provided()),
    )?;
    let before = workspace.cache_stats();
    let ImplicitOdeWorkspace {
        kernel,
        stage_rhs,
        current_rhs,
        endpoint_rhs,
        cache,
        pending_endpoint,
        trial,
        residual,
        correction,
        rhs_candidate,
        ..
    } = workspace;
    let kernel = kernel.as_mut().ok_or_else(|| {
        crate::safety::invalid(
            scioxide_core::argument!("workspace"),
            "missing stage kernel",
        )
    })?;
    let mut work = StageWork::default();
    let mass = kernel.load_mass(system, request, cx)?;
    *pending_endpoint = false;

    cache.evaluate_into(system, t, y, request, current_rhs, rhs_candidate, cx)?;
    for (trial, (&state, &derivative)) in trial
        .as_mut_slice()
        .iter_mut()
        .zip(y.as_slice().iter().zip(current_rhs.as_slice()))
    {
        *trial = if mass == MassKey::Identity {
            crate::safety::stage_affine(state, dt, [(S::ONE, derivative)])?
        } else {
            state
        };
    }
    let weights = [method.implicit_weight];
    let nodes = [S::ONE];
    let coupling = Coupling {
        time: t,
        step: dt,
        coefficients: &weights,
        nodes: &nodes,
    };
    if method.stage.mode_for(request.provider.is_provided()) == NewtonMode::Simplified {
        kernel.linearize(
            system,
            method.stage.strategy,
            0,
            t,
            y.as_slice(),
            current_rhs.as_slice(),
            false,
            request,
            cx,
        )?;
        kernel.factor_newton(coupling, request, cx)?;
    }
    solve_stage(
        kernel,
        system,
        &mut Endpoint {
            system,
            request,
            time: endpoint_time,
            step: dt,
            implicit: method.implicit_weight,
            old: y.as_slice(),
            current: current_rhs.as_slice(),
            rhs: &mut stage_rhs[0],
            candidate: rhs_candidate,
        },
        trial.as_mut_slice(),
        residual.as_mut_slice(),
        correction.as_mut_slice(),
        StageSolve {
            mode: method.stage.mode_for(request.provider.is_provided()),
            strategy: method.stage.strategy,
            coupling,
            limit: method.max_iterations,
            request,
            accuracy: StageAccuracy::from_request(y.as_slice(), request),
            provider: request.provider,
        },
        &mut work,
        cx,
    )?;
    validate_provider_vector(trial, y.len())?;
    endpoint_rhs
        .as_mut_slice()
        .copy_from_slice(stage_rhs[0].as_slice());
    *pending_endpoint = true;
    y.as_mut_slice().copy_from_slice(trial.as_slice());
    work.step_result(before, kernel.stats())
}

/// Coupled `s`-stage method description.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CoupledMethod<'a, S: RealScalar> {
    pub(crate) stage: &'a StageConfig,
    pub(crate) coefficients: &'a [S],
    pub(crate) weights: &'a [S],
    pub(crate) nodes: &'a [S],
    pub(crate) stiffly_accurate: bool,
    pub(crate) max_iterations: Budget,
}

/// The coupled equations `M(Yᵢ − y) − h Σⱼ aᵢⱼ f(t + cⱼh, Yⱼ) = 0`.
struct Coupled<'a, S: RealScalar, T: NumericScalar<Real = S>, P: ?Sized> {
    system: &'a P,
    request: StepRequest<'a, S, T>,
    method: CoupledMethod<'a, S>,
    time: S,
    step: S,
    old: &'a [T],
    rhs: &'a mut [Vector<T>],
    candidate: &'a mut Vector<T>,
}

impl<S, T, P> StageEquation<S, T> for Coupled<'_, S, T, P>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
{
    fn affine_rhs(&mut self) -> Option<&mut [Vector<T>]> {
        (!self.method.stiffly_accurate).then_some(&mut *self.rhs)
    }

    fn evaluations(&self) -> usize {
        self.method.nodes.len()
    }

    /// A stiffly accurate method takes its last stage as the endpoint and never reads
    /// the stage derivatives after the solve.
    fn reads_derivatives(&self) -> bool {
        !self.method.stiffly_accurate
    }

    fn residual(
        &mut self,
        mass: Option<&Matrix<T>>,
        iterate: &[T],
        residual: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let n = self.old.len();
        let s = self.method.nodes.len();
        validate_stage_buffers(iterate, residual, n * s)?;
        for (stage, (&node, rhs)) in self
            .method
            .nodes
            .iter()
            .zip(self.rhs.iter_mut())
            .enumerate()
        {
            evaluate_rhs_staged_slice(
                self.system,
                checked_stage_time(self.time, self.step, node)?,
                &iterate[stage * n..(stage + 1) * n],
                rhs,
                self.candidate,
                self.request,
                cx,
            )?;
        }
        if n == 0 {
            return Ok(());
        }
        for ((state, output), weights) in iterate
            .chunks_exact(n)
            .zip(residual.chunks_exact_mut(n))
            .zip(self.method.coefficients.chunks_exact(s))
        {
            if mass.is_none() {
                crate::safety::implicit_stage_rows(
                    state, self.old, self.step, weights, self.rhs, output,
                )?;
                continue;
            }
            for (row, output) in output.iter_mut().enumerate() {
                let terms = weights
                    .iter()
                    .zip(self.rhs.iter())
                    .map(|(&weight, rhs)| (weight, rhs.as_slice()[row]));
                *output = mass_row_residual(mass, row, state, self.old, self.step, terms, T::ZERO)?;
            }
        }
        Ok(())
    }

    fn point<'b>(&'b self, iterate: &'b [T], block: usize) -> (S, &'b [T], &'b [T]) {
        let n = self.old.len();
        (
            self.time + self.step * self.method.nodes[block],
            &iterate[block * n..(block + 1) * n],
            self.rhs[block].as_slice(),
        )
    }
}

fn validate_coupled_method<S: RealScalar>(
    method: CoupledMethod<'_, S>,
) -> Result<(), SciOxideError> {
    let stages = method.nodes.len();
    let Some(square) = stages.checked_mul(stages) else {
        return Err(SciOxideError::size_overflow(scioxide_core::quantity!(
            "implicit_rk_coefficient_count"
        )));
    };
    if stages == 0 || method.weights.len() != stages || method.coefficients.len() != square {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("self"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "nodes, weights, and coefficient matrix must have consistent nonzero dimensions",
        ));
    }
    Ok(())
}

/// One step of a coupled implicit RK method on any stage backend.
#[allow(clippy::too_many_arguments)]
pub(crate) fn coupled_step<'a, S, T, System>(
    method: CoupledMethod<'_, S>,
    system: &System,
    t: S,
    y: &mut Vector<T>,
    dt: S,
    request: StepRequest<'_, S, T>,
    workspace: &'a mut ImplicitOdeWorkspace<S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<StepResult<'a, S, T>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    collocation_step(
        method,
        StageLayout::Coupled,
        system,
        t,
        y,
        dt,
        request,
        workspace,
        cx,
    )
}

/// Shared collocation equations and Newton driver; the kernel owns the stage layout.
#[allow(clippy::too_many_arguments)]
pub(crate) fn collocation_step<'a, S, T, System>(
    method: CoupledMethod<'_, S>,
    layout: StageLayout,
    system: &System,
    t: S,
    y: &mut Vector<T>,
    dt: S,
    request: StepRequest<'_, S, T>,
    workspace: &'a mut ImplicitOdeWorkspace<S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<StepResult<'a, S, T>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    let request = StepRequest {
        accepted_step: request.accepted_step.filter(|_| !request.self_metered),
        ..request
    };
    validate_coupled_method(method)?;
    validate_finite_state(y, system.dimension(), scioxide_core::NonFiniteOrigin::Input)?;
    checked_step_endpoint(t, dt)?;
    let stage_count = method.nodes.len();
    workspace.prepare(
        y.len(),
        crate::stiff::kernel::KernelShape {
            layout,
            ..method
                .stage
                .shape(stage_count, request.provider.is_provided())
        },
    )?;
    let before = workspace.cache_stats();
    let n = y.len();
    let ImplicitOdeWorkspace {
        kernel,
        stage_rhs,
        current_rhs,
        endpoint_rhs,
        cache,
        pending_endpoint,
        coupled_state,
        trial,
        work: increment,
        weighted_rhs,
        residual,
        correction,
        rhs_candidate,
        radau,
        ..
    } = workspace;
    let kernel = kernel.as_mut().ok_or_else(|| {
        crate::safety::invalid(
            scioxide_core::argument!("workspace"),
            "missing stage kernel",
        )
    })?;
    let mut work = StageWork::default();
    let mass = kernel.load_mass(system, request, cx)?;
    *pending_endpoint = false;
    cache.evaluate_into(system, t, y, request, current_rhs, rhs_candidate, cx)?;
    if let Some(radau) = radau {
        radau.predict(
            t,
            y.as_slice(),
            dt,
            method.nodes,
            coupled_state.as_mut_slice(),
        )?;
    } else {
        for (stage, &node) in method.nodes.iter().enumerate() {
            for (output, (&state, &derivative)) in coupled_state.as_mut_slice()
                [stage * n..(stage + 1) * n]
                .iter_mut()
                .zip(y.as_slice().iter().zip(current_rhs.as_slice()))
            {
                *output = if mass == MassKey::Identity {
                    crate::safety::stage_affine(state, dt, [(node, derivative)])?
                } else {
                    state
                };
            }
        }
    }
    let coupling = Coupling {
        time: t,
        step: dt,
        coefficients: method.coefficients,
        nodes: method.nodes,
    };
    if method.stage.mode_for(request.provider.is_provided()) == NewtonMode::Simplified {
        kernel.linearize(
            system,
            method.stage.strategy,
            0,
            t,
            y.as_slice(),
            current_rhs.as_slice(),
            false,
            request,
            cx,
        )?;
        kernel.factor_newton(coupling, request, cx)?;
    }
    solve_stage(
        kernel,
        system,
        &mut Coupled {
            system,
            request,
            method,
            time: t,
            step: dt,
            old: y.as_slice(),
            rhs: &mut stage_rhs[..stage_count],
            candidate: rhs_candidate,
        },
        coupled_state.as_mut_slice(),
        residual.as_mut_slice(),
        correction.as_mut_slice(),
        StageSolve {
            mode: method.stage.mode_for(request.provider.is_provided()),
            strategy: method.stage.strategy,
            coupling,
            limit: method.max_iterations,
            request,
            accuracy: StageAccuracy::from_request(y.as_slice(), request),
            provider: request.provider,
        },
        &mut work,
        cx,
    )?;
    validate_provider_vector(coupled_state, n * stage_count)?;

    if method.stiffly_accurate {
        trial
            .as_mut_slice()
            .copy_from_slice(&coupled_state.as_slice()[(stage_count - 1) * n..]);
    } else if mass == MassKey::Identity {
        for (index, output) in trial.as_mut_slice().iter_mut().enumerate() {
            *output = crate::safety::stage_affine(
                y.as_slice()[index],
                dt,
                method
                    .weights
                    .iter()
                    .zip(&stage_rhs[..stage_count])
                    .map(|(&weight, rhs)| (weight, rhs.as_slice()[index])),
            )?;
        }
    } else {
        for (index, output) in weighted_rhs.as_mut_slice().iter_mut().enumerate() {
            *output = crate::safety::stage_affine(
                T::ZERO,
                dt,
                method
                    .weights
                    .iter()
                    .zip(&stage_rhs[..stage_count])
                    .map(|(&weight, rhs)| (weight, rhs.as_slice()[index])),
            )?;
        }
        kernel.solve_mass(
            request,
            weighted_rhs.as_slice(),
            increment.as_mut_slice(),
            cx,
        )?;
        for ((next, &state), &delta) in trial
            .as_mut_slice()
            .iter_mut()
            .zip(y.as_slice())
            .zip(increment.as_slice())
        {
            *next = crate::safety::stage_affine(state, S::ONE, [(S::ONE, delta)])?;
        }
    }
    crate::safety::finite(
        trial.as_slice(),
        scioxide_core::NonFiniteOrigin::Intermediate,
    )?;
    if let Some(radau) = radau {
        radau.finish(
            kernel,
            system,
            t,
            dt,
            y,
            trial,
            coupled_state.as_slice(),
            current_rhs,
            method.nodes,
            request,
            weighted_rhs,
            rhs_candidate,
            cx,
        )?;
    }
    if radau.is_none() {
        evaluate_rhs_staged(
            system,
            checked_step_endpoint(t, dt)?,
            trial,
            endpoint_rhs,
            rhs_candidate,
            request,
            cx,
        )?;

        *pending_endpoint = true;
    }
    let mut result = work.step_result(before, kernel.stats())?;
    if let Some(radau) = radau {
        radau.result(request, &mut result);
    }
    y.as_mut_slice().copy_from_slice(trial.as_slice());
    Ok(result)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stepper::testing::with_context;
    use crate::stepper::{StepRequest, StepTrace};
    use crate::stiff::kernel::{KernelShape, StageKernel};
    use scioxide_core::EvaluationRole;

    struct AutomaticSystem;

    impl OdeSystem<f64> for AutomaticSystem {
        fn dimension(&self) -> usize {
            2
        }

        fn rhs(
            &self,
            _t: f64,
            _y: &[f64],
            _dydt: &mut [f64],
            _cx: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("dydt"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "automatic Jacobian selection must not evaluate the RHS",
            ))
        }

        fn automatic_jacobian(
            &self,
            _t: f64,
            _y: &[f64],
            jacobian: &mut scioxide_linalg::MatrixViewMut<'_, f64>,
            _cx: &mut SolveContext<'_, '_>,
        ) -> Result<Option<()>, SciOxideError> {
            jacobian.set(0, 0, -1.0)?;
            jacobian.set(1, 1, -2.0)?;
            Ok(Some(()))
        }
    }

    #[test]
    fn automatic_jacobian_reports_no_rhs_evaluations() {
        for strategy in [
            DerivativeStrategy::Automatic,
            DerivativeStrategy::AnalyticOrAutomaticOrForward,
        ] {
            let shape = KernelShape {
                layout: StageLayout::Coupled,
                backend: &OdeStageBackend::Dense,
                stages: 1,
                blocks: 1,
                provided: false,
            };
            let mut kernel = StageKernel::<f64, f64>::new(2, shape).unwrap();
            let trace = StepTrace::default();
            let mut product = [0.0; 2];
            let counts = with_context(|cx| {
                kernel.linearize(
                    &AutomaticSystem,
                    strategy,
                    0,
                    0.0,
                    &[1.0, 2.0],
                    &[-1.0, -4.0],
                    false,
                    StepRequest::new(&trace),
                    cx,
                )?;
                let linearized = cx.counts().own();
                kernel.apply_jacobian(
                    &AutomaticSystem,
                    StepRequest::new(&trace),
                    &[1.0, 1.0],
                    &mut product,
                    cx,
                )?;
                Ok(linearized)
            })
            .unwrap();

            assert_eq!(counts.provider(EvaluationRole::RightHandSide), 0);
            assert_eq!(
                counts.provider(EvaluationRole::Jacobian),
                if strategy == DerivativeStrategy::Automatic {
                    1
                } else {
                    2
                }
            );
            assert_eq!(product, [-1.0, -2.0]);
        }
    }
    fn derivative_lifecycle<M>(method: M)
    where
        M: crate::stepper::Stepper<f64, f64, Workspace = ImplicitOdeWorkspace<f64, f64>>,
    {
        let base_calls = core::cell::Cell::new(0);
        // Nonlinear callbacks also reuse accepted derivatives, without affine acceptance.
        let system = crate::OdeFn::new(1, |time: f64, y: &[f64], out: &mut [f64]| {
            if time == 0. && y == [1.] {
                base_calls.set(base_calls.get() + 1);
            }
            out[0] = -y[0] * y[0];
            Ok(())
        })
        .unwrap();
        let mut workspace = ImplicitOdeWorkspace::new();
        method.begin(1, false, &mut workspace).unwrap();
        let trace = StepTrace::default();
        let mut request = StepRequest::new(&trace);
        request.accepted_step = Some(0);
        let mut y = Vector::from_vec(alloc::vec![1.]);
        with_context(|cx| {
            method.step(&system, 0., &mut y, 0.125, request, &mut workspace, cx)?;
            assert_eq!(workspace.current_rhs.as_slice(), &[-1.]);
            assert!(workspace.pending_endpoint);
            assert!((workspace.endpoint_rhs.as_slice()[0] + y.as_slice()[0].powi(2)).abs() < 1e-14);
            method.discard(&mut workspace)?;
            assert!(!workspace.pending_endpoint);
            assert_eq!(workspace.current_rhs.as_slice(), &[-1.]);
            assert!(workspace.cache.contains(Some(0)));
            y.as_mut_slice()[0] = 1.;
            method.step(&system, 0., &mut y, 0.0625, request, &mut workspace, cx)?;
            assert_eq!(base_calls.get(), 1);
            method.accept(&mut workspace)?;
            assert!(workspace.cache.contains(Some(1)));
            assert!((workspace.current_rhs.as_slice()[0] + y.as_slice()[0].powi(2)).abs() < 1e-14);
            method.begin(1, false, &mut workspace)?;
            assert!(!workspace.cache.contains(Some(0)));
            assert!(!workspace.pending_endpoint);
            y.as_mut_slice()[0] = 1.;
            method.step(&system, 0., &mut y, 0.0625, request, &mut workspace, cx)?;
            assert_eq!(base_calls.get(), 2);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn radau_evaluates_each_accepted_base_once_without_refreshing_stages() {
        use crate::stepper::Stepper;
        // (stage calls, calls at the base state, other calls at the base time)
        let calls = core::cell::Cell::new((0, 0, 0));
        let base = core::cell::Cell::new((0., 1.));
        let system = crate::OdeFn::new(1, |time: f64, y: &[f64], out: &mut [f64]| {
            let (stages, bases, others) = calls.get();
            calls.set(if (time, y[0]) == base.get() {
                (stages, bases + 1, others)
            } else if time == base.get().0 {
                (stages, bases, others + 1)
            } else {
                (stages + 1, bases, others)
            });
            out[0] = -y[0] * y[0];
            Ok(())
        })
        .unwrap()
        .with_jacobian(
            |_: f64, y: &[f64], jacobian: &mut scioxide_linalg::MatrixViewMut<'_, f64>| {
                jacobian.set(0, 0, -2. * y[0])
            },
        );
        let method = RadauIia::new();
        let mut workspace = ImplicitOdeWorkspace::new();
        method.begin(1, false, &mut workspace).unwrap();
        let trace = StepTrace::default();
        let mut request = StepRequest::new(&trace);
        request.accepted_step = Some(0);
        let mut y = Vector::from_vec(alloc::vec![1.]);
        with_context(|cx| {
            // A trial, its retry at the same base, and one step from the accepted endpoint.
            for (time, step, accept, evaluated) in [
                (0., 0.125, false, 1),
                (0., 0.0625, true, 0),
                (0.0625, 0.0625, true, 1),
            ] {
                base.set((time, y.as_slice()[0]));
                calls.set((0, 0, 0));
                let iterations = method
                    .step(&system, time, &mut y, step, request, &mut workspace, cx)?
                    .num_nonlinear_iters;
                let (stages, bases, estimator) = calls.get();
                assert_eq!(stages, 3 * iterations, "no post-convergence stage refresh");
                assert_eq!(bases, evaluated, "f(tₙ, yₙ) once per accepted base");
                assert!(estimator <= 1);
                if accept {
                    method.accept(&mut workspace)?;
                    request.accepted_step = request.accepted_step.map(|step| step + 1);
                } else {
                    method.discard(&mut workspace)?;
                    y.as_mut_slice()[0] = 1.;
                }
            }
            Ok(())
        })
        .unwrap();
        assert!((y.as_slice()[0] - 1. / 1.125).abs() < 1e-8);
    }

    #[test]
    fn fixed_implicit_methods_commit_derivatives_only_on_acceptance() {
        derivative_lifecycle(BackwardEuler::new());
        derivative_lifecycle(ImplicitTrapezoid::new());
        derivative_lifecycle(GaussRk::new());
        derivative_lifecycle(Sdirk::new());
    }
}
