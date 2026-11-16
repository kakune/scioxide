//! Three-stage singly diagonally implicit Runge–Kutta method.
//!
//! This implementation uses Alexander's third-order, stiffly accurate,
//! L-stable tableau. Its common diagonal coefficient is the L-stable middle
//! root of `6γ³ - 18γ² + 9γ - 1 = 0`, approximately
//! `0.435866521508459`.

use core::fmt;

use scioxide_core::{Budget, NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::{Matrix, Vector};

use super::{ImplicitConfig, mass_row_residual};
use crate::result::StepResult;
use crate::safety::{checked_stage_time, checked_step_endpoint, validate_finite_state};
use crate::stepper::{DefaultController, MethodInfo, StepRequest, Stepper};
use crate::stiff::kernel::{Coupling, MassKey};
use crate::stiff::stage::{StageEquation, StageSolve, solve_stage};
use crate::stiff::{NewtonMode, StageWork};
use crate::traits::{
    OdeSystem, evaluate_rhs_staged_slice, validate_provider_vector, validate_stage_buffers,
};
use crate::workspace::ImplicitOdeWorkspace;

struct SdirkTableau<S> {
    nodes: [S; 3],
    lower_coefficients: [S; 3],
    gamma: S,
}

/// Integrates with Alexander's three-stage SDIRK method, of order 3.
///
/// The method is singly diagonally implicit, stiffly accurate (the last stage is
/// the step result) and L-stable. Its diagonal coefficient `γ ≈ 0.435866521508459`
/// is the root of `6γ³ − 18γ² + 9γ − 1 = 0` that gives L-stability. Because all
/// three diagonal entries are equal, the stages are solved one after the other
/// with a single factorization of `M − hγJ` per Jacobian.
///
/// It is a fixed-step method: every step has the size set by
/// [`OdeOptions::with_initial_step`](crate::OdeOptions::with_initial_step), or the
/// automatic initial-step estimate when none is set. It has no error estimate,
/// dense output or event support. A step whose Newton iteration fails, or whose
/// iteration matrix is singular, ends the solve with that error; see
/// [`OdeMethod::advance`](crate::OdeMethod::advance). The stage solves are described in
/// [Implicit stages](crate#implicit-stages).
///
/// # References
///
/// - R. Alexander, "Diagonally implicit Runge–Kutta methods for stiff O.D.E.'s",
///   *SIAM J. Numer. Anal.* 14(6), 1006–1021, 1977.
///   <https://doi.org/10.1137/0714068>
/// - E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II*,
///   2nd ed., Springer, 1996, §IV.6.
#[derive(Clone)]
pub struct Sdirk<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S> = S> {
    config: ImplicitConfig<S, T>,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> fmt::Debug
    for Sdirk<S, T>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.config.debug("Sdirk", formatter)
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Sdirk<S, T> {
    /// Creates the method with dense LU stages, the default Jacobian sources and
    /// simplified Newton.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            config: ImplicitConfig::new(),
        }
    }

    /// Selects the linear solver of the stage systems: dense LU (the default),
    /// sparse LU or restarted GMRES.
    ///
    /// See [`OdeStageBackend`](crate::OdeStageBackend).
    ///
    /// Sparse and Krylov stages need the identity mass matrix. With any other mass
    /// matrix the solve fails with `InvalidInput` when the first step loads it.
    pub fn with_backend(mut self, backend: crate::OdeStageBackend) -> Self {
        self.config.backend(backend);
        self
    }

    /// Sets the order in which Jacobian sources are tried.
    ///
    /// The default tries the analytic Jacobian of the system, then automatic
    /// differentiation, then forward differences. See
    /// [Implicit stages](crate#implicit-stages).
    #[must_use]
    pub fn with_derivative_strategy(mut self, strategy: DerivativeStrategy) -> Self {
        self.config.strategy(strategy);
        self
    }

    /// Selects simplified Newton (the default) or full Newton for the stage
    /// systems.
    ///
    /// See [`NewtonMode`](crate::NewtonMode).
    #[must_use]
    pub const fn with_newton_mode(mut self, mode: NewtonMode) -> Self {
        self.config.stage.mode = mode;
        self
    }

    fn tableau() -> Result<SdirkTableau<S>, SciOxideError> {
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        let two = one + one;
        let three = two + one;
        let four = two + two;
        let six = three + three;
        let sixteen = four * four;
        let gamma = S::try_from_f64_lossy(0.435_866_521_508_459)?;
        let a21 = (one - gamma) / two;
        let a31 = (sixteen * gamma - six * gamma * gamma - one) / four;
        let a32 = one - gamma - a31;

        Ok(SdirkTableau {
            nodes: [gamma, a21 + gamma, one],
            lower_coefficients: [a21, a31, a32],
            gamma,
        })
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Default for Sdirk<S, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Stepper<S, T>
    for Sdirk<S, T>
{
    type Workspace = ImplicitOdeWorkspace<S, T>;

    /// A-stable: stable on both half-axes for every step.
    fn stability(&self) -> crate::OdeLinearStability<S> {
        crate::OdeLinearStability::unconditional()
    }

    fn info(&self) -> MethodInfo {
        MethodInfo {
            name: "Sdirk",
            algorithm: scioxide_core::algorithm_id!("ode", "Sdirk", 1),
            order: 3,
            initial_step_order: 3,
            controller: DefaultController::ProportionalIntegral,
            dense_output: false,
            events: false,
            adaptive: false,
            mass_matrices: true,
            injectable: true,
        }
    }

    fn fits(&self, dimension: usize, provided: bool, workspace: &Self::Workspace) -> bool {
        self.config.fits(dimension, 1, provided, workspace)
    }

    fn reserve(
        &self,
        dimension: usize,
        provided: bool,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        self.config.prepare(dimension, 1, provided, workspace)
    }

    fn begin(
        &self,
        dimension: usize,
        provided: bool,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        self.config.prepare(dimension, 1, provided, workspace)?;
        workspace.reset_epoch()
    }
    fn accept(&self, workspace: &mut Self::Workspace) -> Result<(), SciOxideError> {
        workspace.accept_step()
    }
    fn discard(&self, workspace: &mut Self::Workspace) -> Result<(), SciOxideError> {
        workspace.discard_step()
    }

    fn step<'w, System: OdeSystem<S, T> + ?Sized>(
        &self,
        system: &System,
        t: S,
        y: &mut Vector<T>,
        dt: S,
        request: StepRequest<'_, S, T>,
        workspace: &'w mut Self::Workspace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<StepResult<'w, S, T>, SciOxideError> {
        validate_finite_state(y, system.dimension(), scioxide_core::NonFiniteOrigin::Input)?;
        checked_step_endpoint(t, dt)?;
        self.config
            .prepare(y.len(), 1, request.provider.is_provided(), workspace)?;
        sdirk_step(&self.config, system, t, y, dt, request, workspace, cx)
    }
}

/// One SDIRK stage `M(x − y) − F − hγ f(tᵢ, x) = 0` with explicit forcing `F`.
struct Stage<'a, S: RealScalar, T: NumericScalar<Real = S>, P: ?Sized> {
    system: &'a P,
    request: StepRequest<'a, S, T>,
    time: S,
    step: S,
    gamma: S,
    old: &'a [T],
    forcing: &'a [T],
    rhs: &'a mut Vector<T>,
    candidate: &'a mut Vector<T>,
}

impl<S, T, P> StageEquation<S, T> for Stage<'_, S, T, P>
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
        for (row, output) in residual.iter_mut().enumerate() {
            *output = mass_row_residual(
                mass,
                row,
                iterate,
                self.old,
                self.step,
                [(self.gamma, self.rhs.as_slice()[row])],
                self.forcing[row],
            )?;
        }
        Ok(())
    }

    fn point<'b>(&'b self, iterate: &'b [T], _: usize) -> (S, &'b [T], &'b [T]) {
        (self.time, iterate, self.rhs.as_slice())
    }
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn sdirk_step<'a, S, T, System>(
    config: &ImplicitConfig<S, T>,
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
    let SdirkTableau {
        nodes,
        lower_coefficients: [a21, a31, a32],
        gamma,
    } = Sdirk::<S>::tableau()?;
    let before = workspace.cache_stats();
    let ImplicitOdeWorkspace {
        kernel,
        stages,
        stage_rhs,
        current_rhs,
        endpoint_rhs,
        cache,
        pending_endpoint,
        trial,
        work: first,
        weighted_rhs: second,
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
    let coefficients = [gamma];
    let coupling = Coupling {
        time: t,
        step: dt,
        coefficients: &coefficients,
        nodes: &nodes[..1],
    };
    if config.stage.mode_for(request.provider.is_provided()) == NewtonMode::Simplified {
        cache.evaluate_into(system, t, y, request, current_rhs, rhs_candidate, cx)?;
        kernel.linearize(
            system,
            config.stage.strategy,
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
    let forcing = stages[0].as_mut_slice();
    forcing.fill(T::ZERO);
    for (stage, &node) in nodes.iter().enumerate() {
        if stage > 0 {
            for (index, value) in forcing.iter_mut().enumerate() {
                *value = if stage == 1 {
                    crate::safety::stage_affine(T::ZERO, dt, [(a21, first.as_slice()[index])])?
                } else {
                    crate::safety::stage_affine(
                        T::ZERO,
                        dt,
                        [
                            (a31, first.as_slice()[index]),
                            (a32, second.as_slice()[index]),
                        ],
                    )?
                };
            }
        }
        for (candidate, (&state, &forcing)) in trial
            .as_mut_slice()
            .iter_mut()
            .zip(y.as_slice().iter().zip(forcing.iter()))
        {
            *candidate = if mass == MassKey::Identity {
                crate::safety::stage_affine(state, S::ONE, [(S::ONE, forcing)])?
            } else {
                state
            };
        }
        solve_stage(
            kernel,
            system,
            &mut Stage {
                system,
                request,
                time: checked_stage_time(t, dt, node)?,
                step: dt,
                gamma,
                old: y.as_slice(),
                forcing,
                rhs: &mut stage_rhs[0],
                candidate: rhs_candidate,
            },
            trial.as_mut_slice(),
            residual.as_mut_slice(),
            correction.as_mut_slice(),
            StageSolve {
                mode: config.stage.mode_for(request.provider.is_provided()),
                strategy: config.stage.strategy,
                coupling,
                limit: Budget::ONE.saturating_add(19),
                request,
                accuracy: crate::StageAccuracy::from_request(y.as_slice(), request),
                provider: request.provider,
            },
            &mut work,
            cx,
        )?;
        validate_provider_vector(trial, y.len())?;
        match stage {
            0 => first
                .as_mut_slice()
                .copy_from_slice(stage_rhs[0].as_slice()),
            1 => second
                .as_mut_slice()
                .copy_from_slice(stage_rhs[0].as_slice()),
            _ => {}
        }
    }

    // Alexander's tableau is stiffly accurate: the last corrected stage is the endpoint.
    endpoint_rhs
        .as_mut_slice()
        .copy_from_slice(stage_rhs[0].as_slice());
    *pending_endpoint = true;
    crate::safety::finite(
        trial.as_slice(),
        scioxide_core::NonFiniteOrigin::Intermediate,
    )?;
    y.as_mut_slice().copy_from_slice(trial.as_slice());
    work.step_result(before, kernel.stats())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn assert_near<S>(label: &str, actual: S, expected: S, tolerance: S)
    where
        S: RealScalar + NumericScalar<Real = S>,
    {
        let error = scioxide_core::NumericScalar::abs(actual - expected);
        assert!(
            error <= tolerance,
            "{label}: actual={actual:?}, expected={expected:?}, error={error:?}, tolerance={tolerance:?}"
        );
    }

    fn assert_third_order_tableau<S>()
    where
        S: RealScalar + NumericScalar<Real = S>,
    {
        let SdirkTableau {
            nodes: [c1, c2, c3],
            lower_coefficients: [a21, a31, a32],
            gamma,
        } = Sdirk::<S>::tableau().unwrap();
        let [b1, b2, b3] = [a31, a32, gamma];
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        let two = one + one;
        let three = two + one;
        let six = three + three;
        let nine = three * three;
        let eighteen = nine * two;
        let tolerance = S::try_from_f64_lossy(64.0).unwrap() * <S as RealScalar>::EPSILON;

        assert_near("first row sum", c1, gamma, tolerance);
        assert_near("second row sum", c2, a21 + gamma, tolerance);
        assert_near("third row sum", c3, a31 + a32 + gamma, tolerance);
        assert_near("weight normalization", b1 + b2 + b3, one, tolerance);

        let gamma_polynomial =
            six * gamma * gamma * gamma - eighteen * gamma * gamma + nine * gamma - one;
        assert_near(
            "L-stable diagonal root",
            gamma_polynomial,
            <S as scioxide_core::AlgebraicScalar>::ZERO,
            tolerance,
        );

        let weighted_node = b1 * c1 + b2 * c2 + b3 * c3;
        assert_near(
            "second-order condition",
            weighted_node,
            one / two,
            tolerance,
        );
        let weighted_squared_node = b1 * c1 * c1 + b2 * c2 * c2 + b3 * c3 * c3;
        assert_near(
            "third-order node condition",
            weighted_squared_node,
            one / three,
            tolerance,
        );
        let ac1 = gamma * c1;
        let ac2 = a21 * c1 + gamma * c2;
        let ac3 = a31 * c1 + a32 * c2 + gamma * c3;
        let weighted_ac = b1 * ac1 + b2 * ac2 + b3 * ac3;
        assert_near(
            "third-order coupling condition",
            weighted_ac,
            one / six,
            tolerance,
        );
    }

    #[test]
    fn alexander_tableau_satisfies_f64_third_order_conditions() {
        assert_third_order_tableau::<f64>();
    }

    #[test]
    fn alexander_tableau_satisfies_f32_third_order_conditions() {
        assert_third_order_tableau::<f32>();
    }
}
