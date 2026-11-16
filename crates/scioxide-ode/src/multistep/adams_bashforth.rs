//! Variable-step Adams–Bashforth methods of orders 1 to 5.
use super::{MultistepOrder, adams};
use crate::OdeSystem;
use crate::result::StepResult;
use crate::stepper::{DefaultController, MethodInfo, StepRequest, Stepper};
use crate::workspace::MultistepWorkspace;
use core::marker::PhantomData;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_linalg::Vector;

/// Integrates nonstiff systems with explicit Adams–Bashforth methods of orders 1
/// to 5.
///
/// The order-q method integrates the polynomial that interpolates `f` at the last
/// q accepted points, with coefficients computed for the actual step sizes. With
/// [`MultistepOrder::Fixed`] the step size is constant; the first steps use
/// [`ClassicalRk4`](crate::ClassicalRk4) (target orders 1 to 4) or
/// [`DormandPrince5`](crate::DormandPrince5) (order 5) to fill the history.
/// With [`MultistepOrder::Automatic`] the method starts at order 1 and adapts order
/// and step size. Dense output interpolates the step history, so output times
/// and events are supported. Explicit Adams methods have small stability
/// regions; for stiff problems use [`Bdf`](crate::Bdf).
///
/// # References
///
/// - E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential
///   Equations I*, 2nd ed., Springer, 1993, §III.1 and §III.5.
#[derive(Debug, Clone)]
pub struct AdamsBashforth<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S> = S> {
    policy: MultistepOrder,
    scalar: PhantomData<(S, T)>,
}
impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> AdamsBashforth<S, T> {
    /// Creates the method of fixed order `order` with constant step size.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) unless the order is between 1 and 5.
    pub fn new(order: usize) -> Result<Self, SciOxideError> {
        Self::with_order_policy(MultistepOrder::Fixed(order))
    }
    /// Creates the method with a fixed or automatic order policy.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) unless the order of `policy` is between 1
    ///   and 5.
    pub fn with_order_policy(policy: MultistepOrder) -> Result<Self, SciOxideError> {
        Ok(Self {
            policy: policy.validate(scioxide_core::argument!("policy"))?,
            scalar: PhantomData,
        })
    }
    /// Returns the order policy.
    pub const fn order_policy(&self) -> MultistepOrder {
        self.policy
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Stepper<S, T>
    for AdamsBashforth<S, T>
{
    type Workspace = MultistepWorkspace<S, T>;
    fn info(&self) -> MethodInfo {
        MethodInfo {
            name: "AdamsBashforth",
            algorithm: scioxide_core::algorithm_id!("ode", "AdamsBashforth", 1),
            order: self.policy.maximum(),
            initial_step_order: if self.policy.automatic() {
                1
            } else if self.policy.maximum() <= 4 {
                4
            } else {
                5
            },
            controller: DefaultController::ProportionalIntegral,
            dense_output: true,
            events: true,
            adaptive: self.policy.automatic(),
            mass_matrices: false,
            injectable: false,
        }
    }
    fn fits(&self, dimension: usize, _provided: bool, workspace: &Self::Workspace) -> bool {
        workspace.fits(dimension, self.policy.maximum() + 1)
    }
    fn reserve(
        &self,
        dimension: usize,
        _provided: bool,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        workspace.try_resize(dimension, self.policy.maximum() + 1)
    }
    fn begin(
        &self,
        dimension: usize,
        _provided: bool,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        workspace.try_resize(dimension, self.policy.maximum() + 1)?;
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
        adams::step(
            adams::Method::Bashforth,
            "AdamsBashforth",
            self.policy,
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
