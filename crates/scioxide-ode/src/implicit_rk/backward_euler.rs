//! Backward Euler method (order 1, L-stable).

use core::fmt;

use scioxide_core::{Budget, NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::Vector;

use super::{EndpointMethod, ImplicitConfig, endpoint_step};
use crate::result::StepResult;
use crate::stepper::{DefaultController, MethodInfo, StepRequest, Stepper};
use crate::stiff::NewtonMode;
use crate::traits::OdeSystem;
use crate::workspace::ImplicitOdeWorkspace;

/// Integrates with the backward (implicit) Euler method, of order 1.
///
/// One step from `(tₙ, yₙ)` with step `h` solves
///
/// ```text
/// M (yₙ₊₁ − yₙ) = h f(tₙ₊₁, yₙ₊₁),   tₙ₊₁ = tₙ + h,
/// ```
///
/// for `yₙ₊₁`, where `M` is the mass matrix of the system (the identity unless
/// [`OdeSystem::mass_matrix_revision`](crate::OdeSystem::mass_matrix_revision)
/// returns a revision). The method is L-stable, so it damps stiff components
/// strongly. It suits very stiff and dissipative problems, and time stepping of
/// discretized PDEs, where first-order accuracy is enough.
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
/// - E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II*,
///   2nd ed., Springer, 1996, §IV.3.
#[derive(Clone)]
pub struct BackwardEuler<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S> = S> {
    config: ImplicitConfig<S, T>,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> fmt::Debug
    for BackwardEuler<S, T>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.config.debug("BackwardEuler", formatter)
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> BackwardEuler<S, T> {
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
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Default
    for BackwardEuler<S, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Stepper<S, T>
    for BackwardEuler<S, T>
{
    type Workspace = ImplicitOdeWorkspace<S, T>;

    /// A-stable: stable on both half-axes for every step.
    fn stability(&self) -> crate::OdeLinearStability<S> {
        crate::OdeLinearStability::unconditional()
    }

    fn info(&self) -> MethodInfo {
        MethodInfo {
            name: "BackwardEuler",
            algorithm: scioxide_core::algorithm_id!("ode", "BackwardEuler", 1),
            order: 1,
            initial_step_order: 1,
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
        endpoint_step(
            EndpointMethod {
                stage: &self.config.stage,
                implicit_weight: S::ONE,
                max_iterations: Budget::ONE.saturating_add(19),
            },
            system,
            t,
            y,
            dt,
            request,
            workspace,
            cx,
        )
    }
}
