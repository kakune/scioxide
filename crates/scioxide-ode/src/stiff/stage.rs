//! The Newton stage driver of implicit Runge–Kutta and BDF methods.
//!
//! Each method family describes its stage equation once through [`StageEquation`]. The
//! driver runs the single [`newton::iterate`](super::newton::iterate) over the kernel's
//! prepared factor (simplified Newton) or relinearizes every block before each correction
//! (full Newton), and hands the whole solve to an injected nonlinear provider when one is
//! configured.
use super::accuracy::StageAccuracy;
use super::kernel::{Coupling, StageKernel};
use super::newton::NewtonStage;
use super::{NewtonMode, StageWork};
use crate::OdeSystem;
use crate::stepper::StepRequest;
use crate::traits::{NonlinearStageSolver, StageLinearSolver};
use scioxide_core::{Budget, NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::{Matrix, Vector};

/// A stage equation `G(x) = 0` of one implicit method family.
pub(crate) trait StageEquation<S, T: NumericScalar> {
    /// Coupled affine stages can transport derivatives with the exact Jacobian.
    fn affine_rhs(&mut self) -> Option<&mut [Vector<T>]> {
        None
    }
    /// Right-hand-side evaluations made by one residual.
    fn evaluations(&self) -> usize;
    /// Whether the method reads stage derivatives after the solve, so that they must be
    /// refreshed at the converged iterate. Otherwise its stage states are only checked.
    fn reads_derivatives(&self) -> bool {
        true
    }
    /// Evaluate `G(iterate)`; `mass` is the dense mass storage, `None` meaning identity.
    /// Right-hand-side evaluations are counted in `cx`.
    fn residual(
        &mut self,
        mass: Option<&Matrix<T>>,
        iterate: &[T],
        residual: &mut [T],
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError>;
    /// Time, state and `f(state)` of linearization `block` after the last residual.
    fn point<'a>(&'a self, iterate: &'a [T], block: usize) -> (S, &'a [T], &'a [T]);
}

/// Selects who solves the stage equations of an implicit Runge–Kutta method.
///
/// Set it with [`OdeProblem::with_stage_solver`](crate::OdeProblem::with_stage_solver).
/// Only [`BackwardEuler`](crate::BackwardEuler),
/// [`ImplicitTrapezoid`](crate::ImplicitTrapezoid), [`GaussRk`](crate::GaussRk),
/// [`Sdirk`](crate::Sdirk) and [`RadauIia`](crate::RadauIia) accept an injected
/// solver. For any other method, the plan rejects it with `InvalidInput`
/// (`UnsupportedConfiguration`). See [Implicit stages](crate#implicit-stages).
#[derive(Clone, Copy)]
pub enum OdeStageSolver<'a, S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> {
    /// Uses the method's own Newton iteration and linear solver. This is the default.
    Native,
    /// Replaces the linear solves inside the simplified Newton iteration with a
    /// [`StageLinearSolver`].
    Linear(&'a dyn StageLinearSolver<S, T>),
    /// Replaces the whole nonlinear stage iteration with a [`NonlinearStageSolver`].
    Nonlinear(&'a dyn NonlinearStageSolver<S, T>),
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> OdeStageSolver<'_, S, T> {
    pub(crate) const fn is_provided(&self) -> bool {
        !matches!(self, Self::Native)
    }
}

/// Policy of one stage solve.
#[derive(Clone, Copy)]
pub(crate) struct StageSolve<
    'a,
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
> {
    pub(crate) mode: NewtonMode,
    pub(crate) strategy: DerivativeStrategy,
    pub(crate) coupling: Coupling<'a, S>,
    /// Method iteration limit; the step request may lower it.
    pub(crate) limit: Budget,
    pub(crate) request: StepRequest<'a, S, T>,
    pub(crate) accuracy: StageAccuracy<'a, S, T>,
    pub(crate) provider: OdeStageSolver<'a, S, T>,
}

struct Driver<'k, 'a, 'c, 'e, S, T, P: ?Sized, E>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    cx: &'k mut SolveContext<'c, 'e>,
    kernel: &'k mut StageKernel<S, T>,
    system: &'k P,
    equation: &'k mut E,
    iterate: &'k mut [T],
    residual: &'k mut [T],
    correction: &'k mut [T],
    solve: StageSolve<'a, S, T>,
}

impl<S, T, P: ?Sized, E> Driver<'_, '_, '_, '_, S, T, P, E>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    /// Express a [`StageAccuracy::correction_norm`] in Newton stopping units. Lagged
    /// linearizations stop at a fraction `min(0.1, max(0.05, 10ε/rtol))` of the
    /// tolerance (0.01 when `rtol = 0`), small enough that a chord error cannot
    /// contaminate the local error estimate.
    fn normalize(&self, norm: S) -> Result<S, SciOxideError> {
        if !self.kernel.requires_rate() {
            return Ok(norm);
        }
        let default = scioxide_core::ErrorTolerance::default();
        let relative = self
            .solve
            .request
            .error_tolerance
            .unwrap_or(&default)
            .relative();
        let fraction = if relative > S::ZERO {
            (S::TEN * S::EPSILON / relative)
                .max(S::try_from_f64_lossy(0.05)?)
                .min(S::try_from_f64_lossy(0.1)?)
        } else {
            S::try_from_f64_lossy(0.01)?
        };
        Ok(norm / S::TEN / fraction)
    }
}

impl<S, T, P, E> NewtonStage<S> for Driver<'_, '_, '_, '_, S, T, P, E>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
    E: StageEquation<S, T>,
{
    type Error = SciOxideError;
    fn residual(&mut self) -> Result<bool, SciOxideError> {
        self.solve.accuracy.admit_iteration(self.cx)?;
        self.equation
            .residual(self.kernel.mass(), self.iterate, self.residual, self.cx)?;

        Ok(self.residual.iter().all(|value| *value == T::ZERO))
    }

    fn correction(&mut self) -> Result<(), SciOxideError> {
        let solve = self.solve;
        if solve.mode == NewtonMode::Full {
            let blocks = self.kernel.blocks();
            for block in 0..blocks {
                let (time, state, base) = self.equation.point(self.iterate, block);
                self.kernel.linearize(
                    self.system,
                    solve.strategy,
                    block,
                    time,
                    state,
                    base,
                    true,
                    solve.request,
                    self.cx,
                )?;
            }
            self.kernel
                .factor(solve.coupling, blocks > 1, solve.request, self.cx)?;
        }
        let linear = match solve.provider {
            Provider::Linear(solver) => Some(solver),
            _ => None,
        };
        self.kernel.solve(
            self.system,
            solve.request,
            solve.accuracy,
            self.iterate,
            self.residual,
            self.correction,
            linear,
            self.cx,
        )
    }

    fn requires_rate(&self) -> bool {
        self.kernel.requires_rate()
    }

    /// A current linearization is certified by the contraction rate carried over from
    /// the last solve; a lagged one, or a missing or slow rate, measures a fresh rate.
    fn certifies_first(&self) -> bool {
        !self.kernel.requires_rate() && super::newton::contracting(self.kernel.contraction())
    }

    fn affine_correction(&mut self) -> Result<Option<bool>, SciOxideError> {
        if !self.system.jacobian_is_constant()
            || !matches!(self.solve.provider, Provider::Native)
            || !self.kernel.affine_direct()
        {
            return Ok(None);
        }
        let transported = if let Some(stages) = self.equation.affine_rhs() {
            let n = self.solve.accuracy.dimension();
            if n > 0 {
                for (rhs, delta) in stages.iter_mut().zip(self.correction.chunks_exact(n)) {
                    self.kernel.apply_jacobian(
                        self.system,
                        self.solve.request,
                        delta,
                        &mut self.residual[..n],
                        self.cx,
                    )?;
                    for (value, &change) in rhs.as_mut_slice().iter_mut().zip(&self.residual[..n]) {
                        *value = crate::safety::state_difference(*value, change)?;
                    }
                }
            }
            true
        } else {
            false
        };
        self.apply()?;
        if transported {
            // Transported derivatives replace callbacks, but corrected trial states
            // must still satisfy the system's admission policy.
            admit_stages(
                self.system,
                &*self.equation,
                self.iterate,
                self.solve.request,
                self.cx,
            )?;
            return Ok(Some(true));
        }
        self.accept()?;
        // Identity-mass residuals are in state units; retain the existing
        // roundoff certification without requiring a second correction norm.
        let accepted = self.kernel.mass().is_some()
            || self
                .solve
                .accuracy
                .residual_satisfied(self.iterate, self.residual)?;
        Ok(Some(accepted))
    }

    fn accept(&mut self) -> Result<bool, SciOxideError> {
        // The final correction changes the stage state. Refresh its derivative
        // before the method retains it for dense output or endpoint reuse.
        if self.equation.reads_derivatives() {
            self.equation
                .residual(self.kernel.mass(), self.iterate, self.residual, self.cx)?;
        } else {
            admit_stages(
                self.system,
                &*self.equation,
                self.iterate,
                self.solve.request,
                self.cx,
            )?;
        }
        Ok(true)
    }

    fn frozen(&self) -> bool {
        self.solve.mode == NewtonMode::Simplified
    }

    fn norm(&mut self) -> Result<S, SciOxideError> {
        // Scale against the corrected candidate, just as endpoint acceptance
        // does. A component may start at zero and acquire a nonzero correction.
        for ((candidate, &value), &delta) in self
            .residual
            .iter_mut()
            .zip(self.iterate.iter())
            .zip(self.correction.iter())
        {
            *candidate = crate::safety::state_difference(value, delta)?;
        }
        let norm = self
            .solve
            .accuracy
            .correction_norm(self.residual, self.correction)?;
        self.normalize(norm)
    }

    /// The rounding level of the corrected candidate, which `norm` left in the
    /// residual buffer.
    fn floor(&mut self) -> Result<S, SciOxideError> {
        let rounding = self.solve.accuracy.rounding_norm(self.residual)?;
        self.normalize(rounding)
    }

    fn apply(&mut self) -> Result<(), SciOxideError> {
        for (value, &delta) in self.iterate.iter_mut().zip(self.correction.iter()) {
            *value = crate::safety::state_difference(*value, delta)?;
        }
        Ok(())
    }
}

/// Require every corrected stage state of a trial step to pass the system's state
/// check, as a right-hand-side evaluation there would.
fn admit_stages<S, T, P, E>(
    system: &P,
    equation: &E,
    iterate: &[T],
    request: crate::stepper::StepRequest<'_, S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
    E: StageEquation<S, T> + ?Sized,
{
    if request.base.is_none() {
        return Ok(());
    }
    for stage in 0..equation.evaluations() {
        let (time, state, _) = equation.point(iterate, stage);
        if !request.trace.provider(system.admits(time, state, cx))? {
            request.trace.inadmissible.set(true);
            return Err(crate::stepper::inadmissible());
        }
    }
    Ok(())
}

/// Solve one stage system into `iterate`, starting from its current value. In simplified
/// mode the caller has linearized and factored the kernel for this step.
///
/// # Errors
/// Newton divergence, stagnation and iteration limits are recoverable convergence
/// failures; budget, provider and linear-solve errors propagate unchanged.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_stage<S, T, P, E>(
    kernel: &mut StageKernel<S, T>,
    system: &P,
    equation: &mut E,
    iterate: &mut [T],
    residual: &mut [T],
    correction: &mut [T],
    solve: StageSolve<'_, S, T>,
    work: &mut StageWork,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
    E: StageEquation<S, T>,
{
    if let Provider::Nonlinear(provider) = solve.provider {
        return nonlinear(
            kernel, system, equation, iterate, residual, correction, provider, solve, work, cx,
        );
    }
    let limit = solve.limit.get().min(solve.accuracy.iteration_limit());
    // Each recovery is admitted through the same scope. Failed iterates remain
    // private and can safely seed the next correction solve.
    for recovery in 0..3 {
        let outcome = super::newton::iterate(
            &mut Driver {
                cx: &mut *cx,
                kernel,
                system,
                equation,
                iterate,
                residual,
                correction,
                solve,
            },
            limit,
        )?;
        work.add_nonlinear_iteration(outcome.report().iterations)?;
        kernel.observe_contraction(outcome.report().rate);
        match outcome {
            super::newton::NewtonOutcome::Converged(report) => {
                kernel.observe_newton(report)?;
                return Ok(());
            }
            _ if recovery < 2
                && (recovery == 0 || !system.jacobian_is_constant())
                && solve.mode == NewtonMode::Simplified
                && matches!(solve.provider, Provider::Native) =>
            {
                if recovery == 0 {
                    kernel.invalidate_factor()?;
                } else {
                    let (time, state, base) = equation.point(iterate, 0);
                    kernel.linearize(
                        system,
                        solve.strategy,
                        0,
                        time,
                        state,
                        base,
                        !system.jacobian_is_constant(),
                        solve.request,
                        cx,
                    )?;
                }
                kernel.factor(solve.coupling, false, solve.request, cx)?;
            }
            _ => return outcome.converged().map(drop),
        }
    }
    unreachable!("bounded Newton recovery returns on its final attempt")
}

/// Hand the stage to an injected nonlinear solver. It receives the effective operator at
/// the step linearization, owns convergence, and is checked only for a finite state; a
/// report without iterations certifies an exact root. The residual is refreshed at the
/// returned state when the method reads its stage derivatives or the report certifies an
/// exact root; otherwise the returned stage states are only checked.
#[allow(clippy::too_many_arguments)]
fn nonlinear<S, T, P, E>(
    kernel: &mut StageKernel<S, T>,
    system: &P,
    equation: &mut E,
    iterate: &mut [T],
    residual: &mut [T],
    correction: &mut [T],
    provider: &dyn NonlinearStageSolver<S, T>,
    solve: StageSolve<'_, S, T>,
    work: &mut StageWork,
    cx: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    P: OdeSystem<S, T> + ?Sized,
    E: StageEquation<S, T>,
{
    let accuracy = solve.accuracy;
    let reads = equation.reads_derivatives();
    let mass = kernel.mass();
    let mut residual_fn = |state: &[T], output: &mut [T], cx: &mut SolveContext<'_, '_>| {
        equation.residual(mass, state, output, cx)
    };
    let report = provider
        .solve_in_place(
            &mut residual_fn,
            &kernel.operator(system, solve.request),
            iterate,
            residual,
            correction,
            accuracy,
            cx,
        )
        .inspect_err(|_| accuracy.mark_provider_failure())?;
    crate::safety::finite(iterate, scioxide_core::NonFiniteOrigin::Callback)
        .inspect_err(|_| accuracy.mark_provider_failure())?;
    let refresh = reads || report.iterations() == 0;
    if refresh {
        residual_fn(iterate, residual, cx)?;
    }

    work.add_nonlinear_iteration(report.iterations())?;
    if !refresh {
        admit_stages(system, &*equation, iterate, solve.request, cx)?;
    }

    if report.iterations() == 0 && residual.iter().any(|value| *value != T::ZERO) {
        return Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::NestedSolver,
            scioxide_core::ConvergenceDiagnostics::new(),
            "nonlinear provider reported success without iterating",
        ));
    }
    Ok(())
}

pub(crate) use OdeStageSolver as Provider;
