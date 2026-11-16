//! Variable-step Adams predictor–corrector methods of orders 1 to 5.
use super::{MultistepOrder, adams};
use crate::OdeSystem;
use crate::result::StepResult;
use crate::stepper::{DefaultController, MethodInfo, StepRequest, Stepper};
use crate::workspace::MultistepWorkspace;
use core::marker::PhantomData;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_linalg::Vector;

/// Integrates nonstiff systems with Adams predictor–corrector methods of orders 1
/// to 5 in PECE mode.
///
/// Each step predicts with the Adams–Bashforth formula, evaluates `f`, corrects
/// once with the Adams–Moulton formula and evaluates `f` at the corrected point,
/// so a step costs two evaluations of `f`. The corrector is applied once, not
/// iterated to convergence, so the method is explicit and suited to nonstiff
/// problems only. Coefficients are computed for the actual step sizes. With
/// [`MultistepOrder::Fixed`] the step size is constant; the first steps use
/// [`ClassicalRk4`](crate::ClassicalRk4) (target orders 1 to 4) or
/// [`DormandPrince5`](crate::DormandPrince5) (order 5) to fill the history. With
/// [`MultistepOrder::Automatic`] the method starts at order 1 and adapts order and
/// step size. Dense output interpolates the step history, so output times and
/// events are supported.
///
/// # Examples
///
/// ```
/// use scioxide_ode::AdamsMoulton;
///
/// let method = AdamsMoulton::<f64>::new(4)?;
/// assert_eq!(method.order_policy(), scioxide_ode::MultistepOrder::Fixed(4));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential
///   Equations I*, 2nd ed., Springer, 1993, §III.1 and §III.5.
#[derive(Debug, Clone)]
pub struct AdamsMoulton<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S> = S> {
    policy: MultistepOrder,
    scalar: PhantomData<(S, T)>,
}
impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> AdamsMoulton<S, T> {
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
    for AdamsMoulton<S, T>
{
    type Workspace = MultistepWorkspace<S, T>;
    fn info(&self) -> MethodInfo {
        MethodInfo {
            name: "AdamsMoulton",
            algorithm: scioxide_core::algorithm_id!("ode", "AdamsMoulton", 1),
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
            adams::Method::Pece,
            "AdamsMoulton",
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
