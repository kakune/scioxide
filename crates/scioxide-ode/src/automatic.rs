//! Automatic switching between a nonstiff and a stiff method, with hysteresis.
pub(crate) mod indicator;
use indicator::Indicator;

use crate::result::StepResult;
use crate::stepper::{DefaultController, MethodInfo, OdeMethod, StepRequest, Stepper};
use crate::{AdamsMoulton, Bdf, MultistepOrder, OdeSystem};
use scioxide_core::{NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_linalg::Vector;

/// Says why [`AutoStiff`] changed its method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeSwitchReason {
    /// `|h| ρ(J)` exceeded the entry threshold at enough consecutive accepted steps.
    Stiffness,
    /// The nonstiff method rejected enough consecutive steps.
    Rejections,
    /// `|h| ρ(J)` stayed below the exit threshold at enough consecutive accepted
    /// steps, after the minimum number of steps with the stiff method.
    Nonstiffness,
    /// The system has a mass matrix, which only the stiff method supports.
    MassMatrix,
}
/// Describes one method change of [`AutoStiff`].
///
/// The new method starts at [`Self::time`] with an empty history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OdeMethodSwitch<S: RealScalar> {
    pub(crate) time: S,
    pub(crate) from: &'static str,
    pub(crate) to: &'static str,
    pub(crate) reason: OdeSwitchReason,
    pub(crate) indicator: Option<S>,
}

impl<S: RealScalar> OdeMethodSwitch<S> {
    /// Returns the accepted time at which the new method takes its first step.
    pub const fn time(&self) -> S {
        self.time
    }
    /// Returns the name of the method before the switch.
    pub const fn from(&self) -> &'static str {
        self.from
    }
    /// Returns the name of the method after the switch.
    pub const fn to(&self) -> &'static str {
        self.to
    }
    /// Returns why the method changed.
    ///
    /// The reason comes from estimates, so it does not prove that the problem is stiff or nonstiff.
    pub const fn reason(&self) -> OdeSwitchReason {
        self.reason
    }
    /// Returns the estimate of `|h| ρ(J)` at the switch, if one was available.
    pub const fn indicator(&self) -> Option<S> {
        self.indicator
    }
}
/// Configures when [`AutoStiff`] switches between its nonstiff and its stiff
/// method.
///
/// The switch is based on the stiffness indicator `|h| ρ(J)`, where `h` is the
/// step size and `ρ(J)` estimates the spectral radius of the Jacobian
/// `J = ∂f/∂y`. The nonstiff method hands over when the indicator exceeds `enter`
/// at `enter_steps` consecutive accepted steps, or after `rejections`
/// consecutive rejected steps. The stiff method hands back when the indicator is
/// below `leave` at `leave_steps` consecutive accepted steps, and only after at
/// least `residence` accepted steps with the stiff method. For this test `h` is
/// the largest step the stiff method has taken since the switch, so a rejected
/// or sharply reduced step alone does not cause a switch back. The default
/// values are `enter = 1`, `leave = 1/4`, `enter_steps = 2`, `leave_steps = 6`,
/// `residence = 8` and `rejections = 3`.
///
/// `ρ(J)` comes from power iteration with a direction kept across steps, using
/// the geometric mean of the last two power ratios, which also converges when
/// the dominant eigenvalues are a pair `±λ` or a complex-conjugate pair. Each
/// iteration needs one product `J v`. It is free when the stiff method has a
/// current Jacobian; otherwise it comes from
/// [`OdeSystem::jacobian_product`] or from a forward difference of `f`, which
/// costs one evaluation of `f`. These products are taken at most once per
/// accepted step, at a spacing that starts at one step, doubles after each
/// estimate up to twelve steps, and restarts at one when an estimate within a
/// factor two of a switching threshold differs from the previous one by more
/// than a quarter. No Jacobian matrix is formed for the indicator.
#[derive(Debug, Clone, Copy)]
pub struct StiffnessSwitchOptions<S: RealScalar> {
    enter: S,
    leave: S,
    enter_steps: usize,
    leave_steps: usize,
    residence: usize,
    rejections: usize,
}
impl<S: RealScalar> StiffnessSwitchOptions<S> {
    /// Creates thresholds `enter = 1`, `leave = 1/4`, evidence counts `2` and `6`,
    /// minimum residence `8`, and rejection threshold `3`.
    /// [`AutoStiff::new`] validates the complete configuration.
    pub fn new() -> Self {
        Self {
            enter: S::ONE,
            leave: S::ONE / (S::TWO + S::TWO),
            enter_steps: 2,
            leave_steps: 6,
            residence: 8,
            rejections: 3,
        }
    }
    /// Sets the enter/leave stiffness thresholds, checked by `AutoStiff::new`.
    #[must_use]
    pub fn with_thresholds(mut self, enter: S, leave: S) -> Self {
        self.enter = enter;
        self.leave = leave;
        self
    }
    /// Sets the required consecutive evidence counts.
    #[must_use]
    pub fn with_evidence_steps(mut self, enter: usize, leave: usize) -> Self {
        self.enter_steps = enter;
        self.leave_steps = leave;
        self
    }
    /// Sets the minimum residence in accepted steps.
    #[must_use]
    pub fn with_residence(mut self, accepted_steps: usize) -> Self {
        self.residence = accepted_steps;
        self
    }
    /// Sets the consecutive rejection threshold.
    #[must_use]
    pub fn with_rejections(mut self, consecutive: usize) -> Self {
        self.rejections = consecutive;
        self
    }
    fn validate(&self) -> Result<(), SciOxideError> {
        let Self {
            enter,
            leave,
            enter_steps,
            leave_steps,
            residence,
            rejections,
        } = *self;
        for (valid, argument) in [
            (
                enter.is_finite() && enter > S::ZERO,
                scioxide_core::argument!("enter"),
            ),
            (
                leave.is_finite() && leave >= S::ZERO && leave < enter,
                scioxide_core::argument!("leave"),
            ),
            (enter_steps > 0, scioxide_core::argument!("enter_steps")),
            (leave_steps > 0, scioxide_core::argument!("leave_steps")),
            (residence > 0, scioxide_core::argument!("residence")),
            (rejections > 0, scioxide_core::argument!("rejections")),
        ] {
            if !valid {
                return Err(crate::safety::invalid(
                    argument,
                    "invalid stiffness switching hysteresis",
                ));
            }
        }
        Ok(())
    }
}
impl<S: RealScalar> Default for StiffnessSwitchOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}
/// Integrates with a nonstiff and a stiff adaptive method, switching between them
/// when the problem becomes stiff or nonstiff.
///
/// Integration starts with the nonstiff method `N`. [`StiffnessSwitchOptions`]
/// states when it switches to the stiff method `F` and back. Each method keeps its
/// own history, which restarts at a switch, and the report counts the switches
/// in [`OdeSolveReport::method_switches`](crate::OdeSolveReport::method_switches).
/// Output times and events need dense output from both methods; a mass matrix
/// needs a stiff method that supports it, and switches to `F` at the first step.
/// [`AdamsBdf`] pairs variable-order Adams and BDF methods. Other adaptive
/// methods, such as [`DormandPrince5`](crate::DormandPrince5) and
/// [`RosenbrockW`](crate::RosenbrockW), combine through [`Self::new`].
#[allow(private_bounds)]
#[derive(Debug, Clone)]
pub struct AutoStiff<
    S: RealScalar + NumericScalar<Real = S>,
    N: OdeMethod<S, T>,
    F: OdeMethod<S, T>,
    T: NumericScalar<Real = S> = S,
> {
    nonstiff: N,
    stiff: F,
    options: StiffnessSwitchOptions<S>,
    marker: core::marker::PhantomData<T>,
}
/// Switches between variable-order Adams predictor–corrector and BDF methods of
/// orders 1 to 5; see [`AutoStiff`] and [`AdamsBdf::adams_bdf`].
pub type AdamsBdf<S, T = S> = AutoStiff<S, AdamsMoulton<S, T>, Bdf<S, T>, T>;
impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> AdamsBdf<S, T> {
    /// Creates the pair of [`AdamsMoulton`] and [`Bdf`], both with automatic
    /// order up to 5, and the default [`StiffnessSwitchOptions`].
    ///
    /// # Errors
    ///
    /// This function does not fail with the built-in orders; it returns the errors
    /// of [`AdamsMoulton::with_order_policy`] and [`Bdf::with_order_policy`].
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::SciOxideError;
    /// use scioxide_linalg::Vector;
    /// use scioxide_ode::{AdamsBdf, OdeFn, OdeOptions, OdeMethod, OdeProblem, OdeState};
    ///
    /// // y′ = −y, y(0) = 1, integrated to t = 0.5.
    /// let system = OdeFn::new(1, |_: f64, y: &[f64], f: &mut [f64]| {
    ///     f[0] = -y[0];
    ///     Ok(())
    /// })?;
    /// let plan = (AdamsBdf::<f64>::adams_bdf()?, OdeOptions::new());
    /// let initial = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
    /// let solution = plan.0.solve(&OdeProblem::new(&system, 0.5), &plan.1, initial)?;
    /// assert!(solution.report().is_success());
    /// # Ok::<(), SciOxideError>(())
    /// ```
    pub fn adams_bdf() -> Result<Self, SciOxideError> {
        Self::new(
            AdamsMoulton::with_order_policy(MultistepOrder::Automatic(5))?,
            Bdf::with_order_policy(MultistepOrder::Automatic(5))?,
            StiffnessSwitchOptions::default(),
        )
    }
}
#[allow(private_bounds)]
impl<
    S: RealScalar + NumericScalar<Real = S>,
    N: OdeMethod<S, T>,
    F: OdeMethod<S, T>,
    T: NumericScalar<Real = S>,
> AutoStiff<S, N, F, T>
{
    fn diagnostic_scratch(
        &self,
        w: &AutoStiffWorkspace<S, N::Workspace, F::Workspace, T>,
    ) -> usize {
        [
            self.nonstiff.stage_kernel(&w.nonstiff),
            self.stiff.stage_kernel(&w.stiff),
        ]
        .into_iter()
        .flatten()
        .map(|kernel| kernel.diagnostic_scratch())
        .max()
        .unwrap_or(0)
    }
    /// Creates the switching method from the nonstiff method `nonstiff`, the stiff
    /// method `stiff` and the switching options.
    ///
    /// The stiffness indicator needs `O(n)` extra storage for `n` state components.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if either method is not
    ///   adaptive.
    /// - `InvalidInput` if an enter/leave threshold is non-finite or unordered,
    ///   `enter ≤ 0`, `leave < 0`, or an evidence, residence or rejection count is zero.
    pub fn new(
        nonstiff: N,
        stiff: F,
        options: StiffnessSwitchOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        if !nonstiff.info().adaptive || !stiff.info().adaptive {
            return Err(crate::safety::invalid(
                scioxide_core::argument!("stiff"),
                "automatic switching requires two adaptive methods",
            ));
        }
        Ok(Self {
            nonstiff,
            stiff,
            options,
            marker: core::marker::PhantomData,
        })
    }
}
/// Independent method histories and bounded switching observations.
pub(crate) struct AutoStiffWorkspace<
    S: RealScalar + NumericScalar<Real = S>,
    N,
    F,
    T: NumericScalar<Real = S> = S,
> {
    nonstiff: N,
    stiff: F,
    active_stiff: bool,
    indicator: Indicator<S, T>,
    next_rhs_valid: bool,
    stiff_step: S,
    high: usize,
    low: usize,
    accepted: usize,
    rejected: usize,
    base: Vector<T>,
    candidate: Vector<T>,
    rhs: Vector<T>,
    next_rhs: Vector<T>,
    rhs_valid: bool,
    rhs_time: Option<S>,
    pending_time: Option<S>,
    pending_switch: Option<OdeMethodSwitch<S>>,
    observed: Option<usize>,
    pending: bool,
}
impl<S: RealScalar + NumericScalar<Real = S>, N: Default, F: Default, T: NumericScalar<Real = S>>
    Default for AutoStiffWorkspace<S, N, F, T>
{
    fn default() -> Self {
        Self {
            nonstiff: N::default(),
            stiff: F::default(),
            active_stiff: false,
            indicator: Indicator::default(),
            next_rhs_valid: false,
            stiff_step: S::ZERO,
            high: 0,
            low: 0,
            accepted: 0,
            rejected: 0,
            base: Vector::empty(),
            candidate: Vector::empty(),
            rhs: Vector::empty(),
            next_rhs: Vector::empty(),
            rhs_valid: false,
            rhs_time: None,
            pending_time: None,
            pending_switch: None,
            observed: None,
            pending: false,
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, N, F, T: NumericScalar<Real = S>>
    AutoStiffWorkspace<S, N, F, T>
{
    /// Whether the automatic scratch vectors serve `n` components.
    fn sized(&self, n: usize) -> bool {
        [&self.base, &self.candidate, &self.rhs, &self.next_rhs]
            .iter()
            .all(|vector| vector.len() == n)
    }
}
impl<
    S: RealScalar + NumericScalar<Real = S>,
    N: OdeMethod<S, T>,
    F: OdeMethod<S, T>,
    T: NumericScalar<Real = S>,
> Stepper<S, T> for AutoStiff<S, N, F, T>
{
    type Workspace = AutoStiffWorkspace<S, N::Workspace, F::Workspace, T>;
    fn stage_kernel<'w>(
        &self,
        workspace: &'w Self::Workspace,
    ) -> Option<&'w crate::stiff::kernel::StageKernel<S, T>> {
        if workspace.active_stiff {
            self.stiff.stage_kernel(&workspace.stiff)
        } else {
            self.nonstiff.stage_kernel(&workspace.nonstiff)
        }
    }
    /// Stiffness switching controls stability itself, so no explicit segment applies.
    fn stability(&self) -> crate::OdeLinearStability<S> {
        crate::OdeLinearStability::unconditional()
    }

    fn info(&self) -> MethodInfo {
        let (nonstiff, stiff) = (self.nonstiff.info(), self.stiff.info());
        MethodInfo {
            name: "AutoStiff",
            algorithm: scioxide_core::algorithm_id!("ode", "AutoStiff", 1),
            order: nonstiff.order,
            initial_step_order: nonstiff.initial_step_order,
            controller: DefaultController::ProportionalIntegral,
            dense_output: nonstiff.dense_output && stiff.dense_output,
            events: nonstiff.events && stiff.events,
            adaptive: true,
            mass_matrices: stiff.mass_matrices,
            injectable: false,
        }
    }
    fn fits(&self, n: usize, _provided: bool, w: &Self::Workspace) -> bool {
        self.nonstiff.fits(n, false, &w.nonstiff)
            && self.stiff.fits(n, false, &w.stiff)
            && w.sized(n)
            && w.indicator.fits(n, self.diagnostic_scratch(w))
    }
    fn reserve(
        &self,
        n: usize,
        _provided: bool,
        w: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        self.nonstiff.reserve(n, false, &mut w.nonstiff)?;
        self.stiff.reserve(n, false, &mut w.stiff)?;
        for vector in [&mut w.base, &mut w.candidate, &mut w.rhs, &mut w.next_rhs] {
            vector.try_resize(n, T::ZERO)?;
        }
        w.indicator.reserve(n, self.diagnostic_scratch(w))?;
        Ok(())
    }
    fn begin(
        &self,
        n: usize,
        _provided: bool,
        w: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        if !w.sized(n) {
            return Err(crate::safety::invalid(
                scioxide_core::argument!("workspace"),
                "automatic workspace dimension changed",
            ));
        }
        self.nonstiff.begin(n, false, &mut w.nonstiff)?;
        self.stiff.begin(n, false, &mut w.stiff)?;
        w.active_stiff = false;
        w.indicator.reset();
        w.next_rhs_valid = false;
        w.stiff_step = S::ZERO;
        w.rhs_valid = false;
        w.rhs_time = None;
        w.pending_time = None;
        w.pending_switch = None;
        w.observed = None;
        w.high = 0;
        w.low = 0;
        w.accepted = 0;
        w.rejected = 0;
        w.pending = false;
        Ok(())
    }
    fn accept(&self, w: &mut Self::Workspace) -> Result<(), SciOxideError> {
        if !w.pending {
            return Err(crate::safety::invalid(
                scioxide_core::argument!("workspace"),
                "automatic solver has no successful proposal",
            ));
        }
        if w.active_stiff {
            self.stiff.accept(&mut w.stiff)?;
        } else {
            self.nonstiff.accept(&mut w.nonstiff)?;
        }
        w.accepted = scioxide_core::backend::checked_add(
            w.accepted,
            1,
            scioxide_core::quantity!("method_residence"),
        )?;
        w.rejected = 0;
        if w.next_rhs_valid {
            w.rhs.as_mut_slice().copy_from_slice(w.next_rhs.as_slice());
        }
        w.rhs_valid = w.next_rhs_valid;
        w.indicator.accept();
        w.rhs_time = w.pending_time.take();
        w.base
            .as_mut_slice()
            .copy_from_slice(w.candidate.as_slice());
        w.pending = false;
        Ok(())
    }
    fn complete_dense<'w, System: OdeSystem<S, T> + ?Sized>(
        &self,
        sys: &System,
        t: S,
        h: S,
        request: StepRequest<'_, S, T>,
        w: &'w mut Self::Workspace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<&'w crate::dense_output::DenseOutputStorage<S, T>, SciOxideError> {
        if w.active_stiff {
            self.stiff
                .complete_dense(sys, t, h, request, &mut w.stiff, cx)
        } else {
            self.nonstiff
                .complete_dense(sys, t, h, request, &mut w.nonstiff, cx)
        }
    }
    fn discard(&self, w: &mut Self::Workspace) -> Result<(), SciOxideError> {
        if w.active_stiff {
            self.stiff.discard(&mut w.stiff)?;
        } else {
            self.nonstiff.discard(&mut w.nonstiff)?;
        }
        w.rejected = scioxide_core::backend::checked_add(
            w.rejected,
            1,
            scioxide_core::quantity!("method_rejection_evidence"),
        )?;
        w.next_rhs_valid = false;
        w.pending_time = None;
        w.pending = false;
        Ok(())
    }
    fn step<'w, System: OdeSystem<S, T> + ?Sized>(
        &self,
        sys: &System,
        t: S,
        y: &mut Vector<T>,
        h: S,
        request: StepRequest<'_, S, T>,
        w: &'w mut Self::Workspace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<StepResult<'w, S, T>, SciOxideError> {
        crate::safety::checked_step_endpoint(t, h)?;
        crate::safety::validate_finite_state(
            y,
            w.base.len(),
            scioxide_core::NonFiniteOrigin::Input,
        )?;
        let mass = sys.mass_matrix_revision().is_some();
        // Accepted endpoint derivatives are optional: only a difference product
        // needs f(t,y) when the active method or the driver cannot supply one.
        if w.rhs_time != Some(t) || w.base.as_slice() != y.as_slice() {
            w.rhs_valid = false;
            w.indicator.reset();
            w.rhs_time = Some(t);
            w.base.as_mut_slice().copy_from_slice(y.as_slice());
        }
        if w.active_stiff {
            w.stiff_step = w.stiff_step.max(h.abs());
        }
        let scale = if w.active_stiff {
            w.stiff_step
        } else {
            h.abs()
        };
        let rate = if mass {
            None
        } else {
            let kernel = if w.active_stiff {
                self.stiff.stage_kernel(&w.stiff)
            } else {
                self.nonstiff.stage_kernel(&w.nonstiff)
            };
            // The switching band reaches a factor two short of the threshold
            // that would change the method.
            let (stiff, enter, leave) = (w.active_stiff, self.options.enter, self.options.leave);
            let band = move |rate: S| {
                let predicted = scale * rate;
                if stiff {
                    predicted <= S::TWO * leave
                } else {
                    S::TWO * predicted >= enter
                }
            };
            let base_rhs = if w.rhs_valid {
                Some(w.rhs.as_slice())
            } else {
                request.initial_rhs
            };
            w.indicator
                .estimate(sys, t, y.as_slice(), base_rhs, kernel, band, request, cx)?
        };
        let indicator = rate
            .map(|r| scioxide_core::checked_float_product(scale, r))
            .transpose()?;
        let mut reason = None;
        // Advance hysteresis evidence only once for each accepted endpoint.
        // Rejected retries use the independent rejection count.
        if w.observed != Some(w.accepted) {
            w.observed = Some(w.accepted);
            w.high = if indicator.is_some_and(|v| v > self.options.enter) {
                w.high.saturating_add(1)
            } else {
                0
            };
            w.low = if indicator.is_some_and(|v| v < self.options.leave) {
                w.low.saturating_add(1)
            } else {
                0
            };
        }
        if !w.active_stiff
            && (mass || w.rejected >= self.options.rejections || w.high >= self.options.enter_steps)
        {
            reason = Some(if mass {
                OdeSwitchReason::MassMatrix
            } else if w.rejected >= self.options.rejections {
                OdeSwitchReason::Rejections
            } else {
                OdeSwitchReason::Stiffness
            });
        }
        if w.active_stiff
            && !mass
            && w.accepted >= self.options.residence
            && w.low >= self.options.leave_steps
        {
            reason = Some(OdeSwitchReason::Nonstiffness);
        }
        let switch = if let Some(reason) = reason {
            let from = if w.active_stiff {
                self.stiff.info().name
            } else {
                self.nonstiff.info().name
            };
            w.active_stiff = !w.active_stiff;
            w.stiff_step = h.abs();
            if w.active_stiff {
                self.stiff.begin(y.len(), false, &mut w.stiff)?;
            } else {
                self.nonstiff.begin(y.len(), false, &mut w.nonstiff)?;
            }
            w.accepted = 0;
            w.observed = None;
            w.rejected = 0;
            w.high = 0;
            w.low = 0;
            Some(OdeMethodSwitch {
                time: t,
                from,
                to: if w.active_stiff {
                    self.stiff.info().name
                } else {
                    self.nonstiff.info().name
                },
                reason,
                indicator,
            })
        } else {
            None
        };
        if switch.is_some() {
            w.pending_switch = switch;
        }
        w.pending = false;
        w.candidate.as_mut_slice().copy_from_slice(y.as_slice());
        let active_order = if w.active_stiff {
            self.stiff.info().order
        } else {
            self.nonstiff.info().order
        };
        // A difference product may have evaluated the base derivative.
        let request = StepRequest {
            initial_rhs: request.initial_rhs.or(w.indicator.base_derivative(t)),
            ..request
        };
        let mut result = if w.active_stiff {
            self.stiff
                .step(sys, t, &mut w.candidate, h, request, &mut w.stiff, cx)?
        } else {
            self.nonstiff
                .step(sys, t, &mut w.candidate, h, request, &mut w.nonstiff, cx)?
        };
        w.next_rhs_valid = result.endpoint_rhs.is_some();
        if let Some(endpoint) = result.endpoint_rhs {
            w.next_rhs.as_mut_slice().copy_from_slice(endpoint);
        }
        w.pending = true;
        w.pending_time = Some(t + h);
        result.error_order = result.error_order.or(Some(active_order));
        result.method_switch = w.pending_switch.take();
        y.as_mut_slice().copy_from_slice(w.candidate.as_slice());
        Ok(result)
    }
}
