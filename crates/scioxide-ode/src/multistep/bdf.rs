//! Backward differentiation formulas of orders 1 to 5.

use super::polynomial::{Functional, Weights};
use crate::safety::{stage_residual, validate_finite_state};

use core::marker::PhantomData;

use scioxide_core::{Budget, NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::{Matrix, Vector};

use super::{
    CAPACITY, MultistepOrder, choose_order, evaluate_current, finish_step, history_rows, old_nodes,
    prepare_step,
};
use crate::result::StepResult;
use crate::stepper::{DefaultController, MethodInfo, StepRequest, Stepper};
use crate::stiff::kernel::Coupling;
use crate::stiff::stage::{Provider, StageEquation, StageSolve, solve_stage};
use crate::stiff::{NewtonMode, StageConfig, StageWork};
use crate::traits::{OdeSystem, evaluate_rhs_staged_slice, validate_stage_buffers};
use crate::workspace::{ImplicitOdeWorkspace, MultistepWorkspace};

/// Selects the formula of [`Bdf`]: backward or numerical differentiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BdfFormula {
    /// The backward differentiation formula, with coefficients computed for the
    /// actual step sizes.
    Bdf,
    /// The numerical differentiation formula: the BDF minus
    /// `κ γ_q (yₙ₊₁ − yₙ₊₁⁽⁰⁾)`, where `yₙ₊₁⁽⁰⁾` is the predicted value,
    /// `γ_q = Σⱼ₌₁^q 1/j`, and `κ = −0.185, −1/9, −0.0823, −0.0415, 0` for orders 1
    /// to 5 (Shampine and Reichelt, 1997). It applies once the history holds `q`
    /// points.
    Ndf,
}

/// Integrates stiff systems with backward differentiation formulas (BDF) or
/// numerical differentiation formulas (NDF) of orders 1 to 5.
///
/// The order-q formula determines `yₙ₊₁` from
///
/// ```text
/// M Σⱼ₌₀^q αⱼ yₙ₊₁₋ⱼ = h f(tₙ₊₁, yₙ₊₁),
/// ```
///
/// where `h` is the step size, `M` the mass matrix and the coefficients `αⱼ` are
/// computed for the actual step sizes. Each step solves this equation by Newton's
/// method with the iteration matrix `M − (h/α₀) J`, `J = ∂f/∂y`, starting from the
/// value predicted by the step history; see [Implicit stages](crate#implicit-stages).
/// Simplified Newton keeps the LU factors while `h/α₀` stays within `[0.8, 1.2]`
/// times its value at the last factorization. Orders 1 and 2 are A-stable;
/// orders 3 to 5 are stable on the whole negative real axis but not A-stable.
///
/// [`Self::new`] fixes the order and the step size. At orders 3 to 5, the first
/// steps are extrapolated backward Euler steps of order q, so that the history
/// starts with the accuracy of the target order. Orders 1 and 2, and
/// [`MultistepOrder::Automatic`], start at order 1 and raise the order as the
/// history grows; the automatic policy also adapts order and step size to the
/// error tolerance. Dense output interpolates the step history, so output times
/// and events are supported. The dense backend supports a mass matrix; the sparse
/// and Krylov backends need `M = I`.
///
/// # Examples
///
/// ```
/// use scioxide_core::SciOxideError;
/// use scioxide_linalg::Vector;
/// use scioxide_ode::{Bdf, OdeFn, OdeOptions, OdeMethod, OdeProblem, OdeState};
///
/// // y′ = −y, y(0) = 1, integrated to t = 0.5 with order 5 at a constant step.
/// let system = OdeFn::new(1, |_: f64, y: &[f64], f: &mut [f64]| {
///     f[0] = -y[0];
///     Ok(())
/// })?;
/// let plan = (Bdf::<f64>::new(5)?, OdeOptions::new());
/// let initial = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
/// let solution = plan.0.solve(&OdeProblem::new(&system, 0.5), &plan.1, initial)?;
/// assert!(solution.report().is_success());
/// # Ok::<(), SciOxideError>(())
/// ```
///
/// # References
///
/// - E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II*, 2nd
///   ed., Springer, 1996, §IV.9 (extrapolated backward Euler) and chapter V.
/// - R. W. Klopfenstein, "Numerical differentiation formulas for stiff systems of
///   ordinary differential equations", *RCA Review* 32, 447–462, 1971.
/// - L. F. Shampine and M. W. Reichelt, *SIAM J. Sci. Comput.* 18(1), 1–22, 1997.
#[derive(Debug, Clone)]
pub struct Bdf<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S> = S> {
    policy: MultistepOrder,
    formula: BdfFormula,
    max_iterations: Budget,
    stage: StageConfig,
    scalar: PhantomData<(S, T)>,
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Bdf<S, T> {
    /// Creates the BDF of fixed order `order` with constant step size, up to 20
    /// Newton iterations per step and the default stage options.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) unless the order is between 1 and 5.
    pub fn new(order: usize) -> Result<Self, SciOxideError> {
        if order == 0 || order > 5 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("order"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "BDF order must be between 1 and 5",
            ));
        }
        Ok(Self {
            policy: MultistepOrder::Fixed(order),
            formula: BdfFormula::Bdf,
            max_iterations: Budget::ONE.saturating_add(19),
            stage: StageConfig::new(),
            scalar: PhantomData,
        })
    }

    /// Whether `prepare(dimension, workspace)` would leave the storage unchanged.
    fn fits_workspace(&self, dimension: usize, workspace: &MultistepWorkspace<S, T>) -> bool {
        workspace.fits(dimension, self.policy.maximum() + 1)
            && (self.policy.automatic()
                || self.policy.maximum() < 3
                || dimension
                    .checked_mul(startup_rows(self.policy.maximum()))
                    .is_some_and(|values| workspace.bdf_startup.len() == values))
            && workspace
                .implicit
                .fits(dimension, self.stage.shape(1, false))
    }

    fn prepare(
        &self,
        dimension: usize,
        workspace: &mut MultistepWorkspace<S, T>,
    ) -> Result<(), SciOxideError> {
        workspace.try_resize(dimension, self.policy.maximum() + 1)?;
        if !self.policy.automatic() && self.policy.maximum() >= 3 {
            let values = scioxide_core::backend::checked_mul(
                dimension,
                startup_rows(self.policy.maximum()),
                scioxide_core::quantity!("bdf_startup"),
            )?;
            workspace.bdf_startup.try_resize(values, T::ZERO)?;
        }
        workspace
            .implicit
            .prepare(dimension, self.stage.shape(1, false))
    }

    /// Sets the linear solver of the Newton iteration: dense LU, sparse LU or a
    /// matrix-free Krylov method; see [`OdeStageBackend`](crate::OdeStageBackend).
    ///
    /// Only the dense backend supports a mass matrix.
    pub fn with_backend(mut self, backend: crate::OdeStageBackend) -> Self {
        self.stage.backend = backend;
        self
    }

    /// Creates the BDF with a fixed or automatic order policy.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) unless the order of `policy` is between 1
    ///   and 5.
    pub fn with_order_policy(policy: MultistepOrder) -> Result<Self, SciOxideError> {
        let mut solver = Self::new(
            policy
                .validate(scioxide_core::argument!("policy"))?
                .maximum(),
        )?;
        solver.policy = policy;
        Ok(solver)
    }
    /// Sets the formula: [`BdfFormula::Bdf`] (the default) or [`BdfFormula::Ndf`].
    #[must_use]
    pub const fn with_formula(mut self, formula: BdfFormula) -> Self {
        self.formula = formula;
        self
    }
    /// Returns the order policy.
    pub const fn order_policy(&self) -> MultistepOrder {
        self.policy
    }
    /// Returns the formula.
    pub const fn formula(&self) -> BdfFormula {
        self.formula
    }

    /// Sets the largest number of Newton iterations per step; the default is 20.
    ///
    /// A step whose iteration does not converge within the limit is retried with a
    /// smaller step by the automatic order policy.
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iterations: Budget) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets how the Jacobian `J = ∂f/∂y` is obtained: from the system's analytic
    /// Jacobian, by automatic differentiation or by finite differences; see
    /// [Implicit stages](crate#implicit-stages).
    #[must_use]
    pub fn with_derivative_strategy(mut self, strategy: DerivativeStrategy) -> Self {
        self.stage.strategy = strategy;
        self
    }

    /// Sets simplified or full Newton iteration; see [`NewtonMode`].
    #[must_use]
    pub const fn with_newton_mode(mut self, mode: NewtonMode) -> Self {
        self.stage.mode = mode;
        self
    }
}

impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Stepper<S, T>
    for Bdf<S, T>
{
    type Workspace = MultistepWorkspace<S, T>;
    const CHECKED_RESULT: bool = true;

    fn stage_kernel<'w>(
        &self,
        workspace: &'w Self::Workspace,
    ) -> Option<&'w crate::stiff::kernel::StageKernel<S, T>> {
        workspace.implicit.kernel.as_ref()
    }

    /// BDF1–BDF5 are stable on the whole negative real axis (A(α)); only orders
    /// one and two are A-stable, so higher orders declare no imaginary segment.
    fn stability(&self) -> crate::OdeLinearStability<S> {
        crate::OdeLinearStability::Axes {
            real: None,
            imaginary: (self.policy.maximum() > 2).then_some(S::ZERO),
        }
    }

    fn info(&self) -> MethodInfo {
        MethodInfo {
            name: "Bdf",
            algorithm: scioxide_core::algorithm_id!("ode", "Bdf", 1),
            order: self.policy.maximum(),
            initial_step_order: if self.policy.automatic() || self.policy.maximum() <= 2 {
                1
            } else {
                self.policy.maximum()
            },
            controller: DefaultController::ProportionalIntegral,
            dense_output: true,
            events: true,
            adaptive: self.policy.automatic(),
            mass_matrices: true,
            injectable: false,
        }
    }

    fn fits(&self, dimension: usize, _provided: bool, workspace: &Self::Workspace) -> bool {
        self.fits_workspace(dimension, workspace)
    }

    fn reserve(
        &self,
        dimension: usize,
        _provided: bool,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        self.prepare(dimension, workspace)
    }

    fn begin(
        &self,
        dimension: usize,
        _provided: bool,
        workspace: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        self.prepare(dimension, workspace)?;
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
        self.prepare(y.len(), workspace)?;
        bdf_step(
            BdfStepConfiguration {
                stage: &self.stage,
                policy: self.policy,
                formula: self.formula,
                max_iterations: self.max_iterations,
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

#[derive(Debug, Clone, Copy)]
struct BdfStepConfiguration<'a> {
    stage: &'a StageConfig,
    policy: MultistepOrder,
    formula: BdfFormula,
    max_iterations: Budget,
}

/// The BDF/NDF equation `(α₀M x − M·Σ − h f(t, x)) / α₀ = 0` for the history sum `Σ`.
struct BdfStage<'a, S: RealScalar, T: NumericScalar<Real = S>, P: ?Sized> {
    system: &'a P,
    request: StepRequest<'a, S, T>,
    time: S,
    alpha: S,
    step: S,
    history: &'a [T],
    normalized_history: Option<&'a [T]>,
    normalized_step: S,
    rhs: &'a mut Vector<T>,
    candidate: &'a mut Vector<T>,
}

impl<S, T, P> StageEquation<S, T> for BdfStage<'_, S, T, P>
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
        let n = self.history.len();
        validate_stage_buffers(iterate, residual, n)?;
        evaluate_rhs_staged_slice(
            self.system,
            self.time,
            iterate,
            self.rhs,
            self.candidate,
            self.request,
            cx,
        )?;
        let alpha = T::from_real(self.alpha);
        for (row, output) in residual.iter_mut().enumerate() {
            if mass.is_none()
                && let Some(history) = self.normalized_history
            {
                *output = crate::safety::normalized_stage_residual(
                    self.alpha,
                    iterate[row],
                    self.history[row],
                    self.step,
                    self.rhs.as_slice()[row],
                    history[row],
                    self.normalized_step,
                )?;
                continue;
            }
            let value = match mass {
                None => crate::safety::scalar_stage_residual(
                    self.alpha,
                    iterate[row],
                    self.history[row],
                    self.step,
                    [(S::ONE, self.rhs.as_slice()[row])],
                    T::ZERO,
                )?,
                Some(mass) => stage_residual(
                    &mass.as_slice()[row * n..(row + 1) * n],
                    self.alpha,
                    iterate,
                    self.history,
                    self.step,
                    [(S::ONE, self.rhs.as_slice()[row])],
                    T::ZERO,
                )?,
            };
            *output = value / alpha;
        }
        Ok(())
    }

    fn point<'b>(&'b self, iterate: &'b [T], _: usize) -> (S, &'b [T], &'b [T]) {
        (self.time, iterate, self.rhs.as_slice())
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn bdf_step<'a, S, T, System>(
    configuration: BdfStepConfiguration<'_>,
    system: &System,
    t: S,
    y: &mut Vector<T>,
    dt: S,
    request: StepRequest<'_, S, T>,
    workspace: &'a mut MultistepWorkspace<S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<StepResult<'a, S, T>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    let next_time = prepare_step(
        "Bdf",
        system,
        t,
        y,
        dt,
        workspace,
        configuration.policy,
        u8::from(configuration.formula == BdfFormula::Ndf),
    )?;
    if !configuration.policy.automatic()
        && configuration.policy.maximum() >= 3
        && workspace.history_count + 1 < configuration.policy.maximum()
    {
        return startup(configuration, system, t, y, dt, request, workspace, cx);
    }
    let active_order = workspace.active_order;
    let before = workspace.implicit.cache_stats();
    let mut work = StageWork::default();
    evaluate_current(system, t, y, workspace, request, cx)?;

    let nodes = old_nodes(t, dt, workspace, workspace.history_count)?;
    workspace.bdf_coefficients.prepare(
        active_order,
        configuration.formula,
        configuration.policy,
        &nodes,
        workspace.history_count,
    )?;
    let alpha = workspace.bdf_coefficients.alpha;
    let correction = workspace.bdf_coefficients.correction;
    let mut slots = [0; CAPACITY];
    for (offset, slot) in slots.iter_mut().enumerate().take(workspace.history_count) {
        *slot = workspace.history_slot(offset);
    }
    let predictor_order = active_order.min(workspace.history_count);
    let rows = history_rows(
        y.as_slice(),
        None,
        &workspace.history_states,
        &slots,
        predictor_order + 1,
    );
    crate::safety::combine_rows(
        None,
        S::ONE,
        None,
        active_order,
        |j| (-alpha[j + 1], rows[j]),
        workspace.rhs_sum.as_mut_slice(),
        |i, _| {
            crate::safety::stage_affine(
                T::ZERO,
                S::ONE,
                (0..active_order).map(|j| (-alpha[j + 1], rows[j][i])),
            )
        },
    )?;
    workspace.bdf_coefficients.predictor.apply(
        None,
        S::ONE,
        &rows[..=predictor_order],
        workspace.predictor.as_mut_slice(),
    )?;
    if correction != S::ZERO {
        for (sum, &predictor) in workspace
            .rhs_sum
            .as_mut_slice()
            .iter_mut()
            .zip(workspace.predictor.as_slice())
        {
            *sum = crate::safety::stage_affine(*sum, -correction, [(S::ONE, predictor)])?;
        }
    }
    let alpha0 = alpha[0];
    let normalized_step = dt / alpha0;
    let mut normalized = normalized_step.is_normal();
    workspace
        .rhs_candidate
        .as_mut_slice()
        .copy_from_slice(workspace.rhs_sum.as_slice());
    for (value, &original) in workspace
        .rhs_sum
        .as_mut_slice()
        .iter_mut()
        .zip(workspace.rhs_candidate.as_slice())
    {
        *value = original / T::from_real(alpha0);
        normalized &= value.has_finite_components()
            && (value.real_component().is_normal() || original.real_component() == S::ZERO)
            && (T::IS_REAL
                || value.imaginary_component().is_normal()
                || original.imaginary_component() == S::ZERO);
    }
    {
        let MultistepWorkspace {
            implicit,
            predictor,
            rhs_sum,
            rhs_candidate: history,
            current_derivative,
            proposed_state,
            ..
        } = &mut *workspace;
        let ImplicitOdeWorkspace {
            kernel,
            stage_rhs,
            trial,
            residual,
            correction,
            rhs_candidate,
            ..
        } = implicit;
        let kernel = kernel.as_mut().ok_or_else(|| {
            crate::safety::invalid(
                scioxide_core::argument!("workspace"),
                "missing stage kernel",
            )
        })?;
        kernel.load_mass(system, request, cx)?;
        trial.as_mut_slice().copy_from_slice(predictor.as_slice());
        let coefficients = [S::ONE / alpha0];
        let unit = [S::ONE];
        let coupling = Coupling {
            time: t,
            step: dt,
            coefficients: &coefficients,
            nodes: &unit,
        };
        let stage = configuration.stage;
        if stage.mode == NewtonMode::Simplified {
            kernel.linearize(
                system,
                stage.strategy,
                0,
                t,
                y.as_slice(),
                current_derivative.as_slice(),
                false,
                request,
                cx,
            )?;
            kernel.factor_newton(coupling, request, cx)?;
        }
        solve_stage(
            kernel,
            system,
            &mut BdfStage {
                system,
                request,
                time: next_time,
                alpha: alpha0,
                step: dt,
                history: history.as_slice(),
                normalized_history: normalized.then_some(rhs_sum.as_slice()),
                normalized_step,
                rhs: &mut stage_rhs[0],
                candidate: rhs_candidate,
            },
            trial.as_mut_slice(),
            residual.as_mut_slice(),
            correction.as_mut_slice(),
            StageSolve {
                mode: stage.mode,
                strategy: stage.strategy,
                coupling,
                limit: configuration.max_iterations,
                request,
                accuracy: crate::StageAccuracy::from_request(y.as_slice(), request),
                provider: Provider::Native,
            },
            &mut work,
            cx,
        )?;
        crate::safety::finite(
            trial.as_slice(),
            scioxide_core::NonFiniteOrigin::Intermediate,
        )?;
        proposed_state
            .as_mut_slice()
            .copy_from_slice(trial.as_slice());
    }
    // The shared Newton driver refreshes f at the final corrected state.
    // Keep that derivative with the pending proposal; rejection discards it.
    workspace
        .next
        .as_mut_slice()
        .copy_from_slice(workspace.implicit.stage_rhs[0].as_slice());
    workspace.next_derivative_valid = true;
    let mut normalized_error = None;
    if configuration.policy.automatic() {
        for ((scale, old), new) in workspace
            .scales
            .as_mut_slice()
            .iter_mut()
            .zip(y.as_slice())
            .zip(workspace.proposed_state.as_slice())
        {
            *scale = NumericScalar::abs(*old).max(NumericScalar::abs(*new));
        }
        let default = scioxide_core::ErrorTolerance::default();
        let tolerance = request.error_tolerance.unwrap_or(&default);
        let norm = tolerance.prepare_norm(
            workspace.scales.as_slice(),
            workspace.error_magnitudes.as_mut_slice(),
        )?;
        let error_rows = history_rows(
            y.as_slice(),
            Some(workspace.proposed_state.as_slice()),
            &workspace.history_states,
            &slots,
            workspace.history_count.min(active_order + 1) + 2,
        );
        let mut estimates = [S::ZERO; 3];
        for (slot, order) in [
            (0, active_order.saturating_sub(1)),
            (2, active_order + 1),
            (1, active_order),
        ] {
            if (!configuration.policy.automatic() && slot != 1)
                || order == 0
                || order > configuration.policy.maximum()
                || order > workspace.history_count + 1
            {
                continue;
            }
            if workspace.history_count == 0 {
                // Reserve a quarter of the startup error budget for the local
                // defect h/2 M^-1(f1-f0), leaving propagation/Newton margin.
                for ((out, &new), &old) in workspace
                    .rhs_sum
                    .as_mut_slice()
                    .iter_mut()
                    .zip(workspace.next.as_slice())
                    .zip(workspace.current_derivative.as_slice())
                {
                    *out = crate::safety::stage_affine(
                        T::ZERO,
                        dt,
                        [(S::TWO, crate::safety::state_difference(new, old)?)],
                    )?;
                }
                workspace
                    .implicit
                    .kernel
                    .as_mut()
                    .ok_or_else(|| {
                        crate::safety::invalid(
                            scioxide_core::argument!("workspace"),
                            "missing stage kernel",
                        )
                    })?
                    .solve_mass(
                        request,
                        workspace.rhs_sum.as_slice(),
                        workspace.errors.as_mut_slice(),
                        cx,
                    )?;
            } else {
                let estimate = workspace.bdf_coefficients.errors[slot]
                    .as_ref()
                    .ok_or_else(|| {
                        crate::safety::invalid(
                            scioxide_core::argument!("workspace"),
                            "missing BDF error coefficients",
                        )
                    })?;
                match &estimate.weights {
                    Some(weights) => weights.apply(
                        None,
                        estimate.factor,
                        &error_rows,
                        workspace.errors.as_mut_slice(),
                    )?,
                    None => {
                        for (i, error) in workspace.errors.as_mut_slice().iter_mut().enumerate() {
                            *error = crate::safety::state_difference(
                                error_rows[0][i],
                                error_rows[1][i],
                            )?;
                        }
                    }
                }
            }
            estimates[slot] = norm.normalized_norm_by(|i| workspace.errors.as_slice()[i].abs())?;
        }
        choose_order(configuration.policy, estimates, workspace)?;
        normalized_error = Some(estimates[1]);
    }
    finish_step(t, y, request, workspace)?;
    let mut result = work.step_result(before, workspace.implicit.cache_stats())?;
    result.raw_error = configuration
        .policy
        .automatic()
        .then_some(workspace.errors.as_slice());
    result.normalized_error = normalized_error;
    result.endpoint_rhs = Some(workspace.next.as_slice());
    // Allow faster growth at first order; higher orders retain the conservative
    // history-growth bound.
    result.maximum_step_growth = Some(if active_order == 1 {
        S::try_from_f64_lossy(5.)?
    } else {
        S::TWO
    });
    result.error_order = configuration.policy.automatic().then_some(active_order + 1);
    if request.requires_dense_output() {
        result.dense_output = Some(&workspace.dense);
    }
    Ok(result)
}

/// Highest backward-difference order of a startup dense segment of order `q`.
const fn difference_order(q: usize) -> usize {
    q - 2
}

/// Offset, in rows of the state dimension, of the scaled differences `mᵏ∇ᵏy`,
/// `1 ≤ k ≤ min(m, q - 2)`, of subdivision row `m`. Rows before the first
/// hold `q` subdivision endpoints, later reused for dense coefficients, and
/// the running differences `∇⁰..∇^(q-2)` of the current row.
const fn difference_row(q: usize, m: usize) -> usize {
    let order = difference_order(q);
    let mut row = q + order + 1;
    let mut previous = 1;
    while previous < m {
        row += if previous < order { previous } else { order };
        previous += 1;
    }
    row
}

/// Startup scratch rows of a fixed order `q`.
const fn startup_rows(q: usize) -> usize {
    difference_row(q, q + 1)
}

/// Extrapolate fully implicit Euler endpoints in 1/n. Internal subdivisions
/// reuse stage scratch and never enter the multistep history transaction.
#[allow(clippy::too_many_arguments)]
fn startup<'a, S, T, System>(
    configuration: BdfStepConfiguration<'_>,
    system: &System,
    t: S,
    y: &mut Vector<T>,
    dt: S,
    request: StepRequest<'_, S, T>,
    workspace: &'a mut MultistepWorkspace<S, T>,
    cx: &mut SolveContext<'_, '_>,
) -> Result<StepResult<'a, S, T>, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
    System: OdeSystem<S, T> + ?Sized,
{
    let q = configuration.policy.maximum();
    let n = y.len();
    let before = workspace.implicit.cache_stats();
    let mut work = StageWork::default();
    evaluate_current(system, t, y, workspace, request, cx)?;
    // Extrapolation cancels leading discretization errors, so its stage solves
    // must reach working precision even when the outer tolerance is loose.
    let tolerance = scioxide_core::ErrorTolerance::scalar(
        S::try_from_f64_lossy(32.)? * S::EPSILON,
        S::try_from_f64_lossy(32.)? * S::EPSILON,
        scioxide_core::NormSelection::Infinity,
    )?;
    // Internal subdivisions do not share the outer driver's accepted-point key.
    let inner = StepRequest {
        accepted_step: None,
        ..request
    }
    .without_dense_output()
    .with_error_tolerance(&tolerance);
    let dense = request.requires_dense_output();
    let order = difference_order(q);
    let mut nodes = [S::ZERO; CAPACITY];
    let mut completed = 0;
    'rows: for subdivisions in 1..=q {
        let count = S::try_from_usize_exact(subdivisions)?;
        // Preflight the whole row before callbacks: adjacent/subnormal outer
        // steps can admit fewer than q distinct internal increments.
        let mut previous = t;
        for index in 1..=subdivisions {
            let endpoint =
                crate::safety::checked_stage_time(t, dt, S::try_from_usize_exact(index)? / count)?;
            if endpoint == previous {
                break 'rows;
            }
            previous = endpoint;
        }
        nodes[subdivisions - 1] = S::ONE / count;
        workspace
            .corrector
            .as_mut_slice()
            .copy_from_slice(y.as_slice());
        if dense {
            workspace.bdf_startup.as_mut_slice()[q * n..(q + 1) * n].copy_from_slice(y.as_slice());
        }
        let mut time = t;
        for index in 1..=subdivisions {
            let endpoint =
                crate::safety::checked_stage_time(t, dt, S::try_from_usize_exact(index)? / count)?;
            let width = scioxide_core::checked_difference(endpoint, time)?;
            let result = crate::implicit_rk::endpoint_step(
                crate::implicit_rk::EndpointMethod {
                    stage: configuration.stage,
                    implicit_weight: S::ONE,
                    max_iterations: configuration.max_iterations,
                },
                system,
                time,
                &mut workspace.corrector,
                width,
                inner,
                &mut workspace.implicit,
                cx,
            )?;
            work.absorb(&result)?;
            workspace.implicit.accept_step()?;
            time = endpoint;
            if dense {
                // Running backward differences ∇ᵏ of the row, k ≤ min(index, q - 2).
                let running = &mut workspace.bdf_startup.as_mut_slice()[q * n..(q + order + 1) * n];
                let limit = index.min(order);
                for (i, &value) in workspace.corrector.as_slice().iter().enumerate() {
                    let mut carry = value;
                    for k in 0..=limit {
                        let old = running[k * n + i];
                        running[k * n + i] = carry;
                        if k < limit {
                            carry = crate::safety::state_difference(carry, old)?;
                        }
                    }
                }
            }
        }
        workspace.bdf_startup.as_mut_slice()[(subdivisions - 1) * n..subdivisions * n]
            .copy_from_slice(workspace.corrector.as_slice());
        if dense {
            let first = difference_row(q, 1);
            let (head, rows) = workspace.bdf_startup.as_mut_slice().split_at_mut(first * n);
            let start = (difference_row(q, subdivisions) - first) * n;
            let mut scale = S::ONE;
            for k in 1..=subdivisions.min(order) {
                scale *= count;
                let source = &head[(q + k) * n..(q + k + 1) * n];
                let target = &mut rows[start + (k - 1) * n..start + k * n];
                for (out, &difference) in target.iter_mut().zip(source) {
                    *out = crate::safety::stage_affine(T::ZERO, scale, [(S::ONE, difference)])?;
                }
            }
        }
        completed = subdivisions;
    }
    let extrapolation = Weights::new(&nodes[..completed], Functional::Value(S::ZERO))?;
    for i in 0..n {
        workspace.proposed_state.as_mut_slice()[i] =
            extrapolation.affine(T::ZERO, S::ONE, |j| {
                workspace.bdf_startup.as_slice()[j * n + i]
            })?;
    }
    workspace.errors.as_mut_slice().fill(T::ZERO);
    if dense {
        startup_dense(t, y, dt, q, &nodes[..completed], workspace)?;
    }
    finish_step(t, y, request.without_dense_output(), workspace)?;
    let mut result = work.step_result(before, workspace.implicit.cache_stats())?;
    if dense {
        result.dense_output = Some(&workspace.dense);
    }
    Ok(result)
}

/// Dense segment of an extrapolated startup step from `y` to the proposed state.
///
/// With `p` completed rows and `r = max(p, 2) - 2`, `Dₖ` extrapolates the scaled
/// differences `mᵏ∇ᵏy` of rows `m = k..=p` to `hᵏy⁽ᵏ⁾(t + h)` for `1 ≤ k ≤ r`.
/// The polynomial `P` of degree `r + 2` satisfies
///
/// ```text
/// P(0) = y,   P'(0) = h f(t, y),   P⁽ᵏ⁾(1) = Dₖ  (0 ≤ k ≤ r),   D₀ = y₁,
/// ```
///
/// and is stored in powers `P(θ) = y + Σ_d c_d θᵈ`. With `F = P - y - h f θ`
/// and `Rₖ = F⁽ᵏ⁾(1)`, the Hermite basis about `θ = 1`,
///
/// ```text
/// Bₖ(θ) = (θ - 1)ᵏ/k! + aₖ (θ - 1)ʳ⁺¹ + bₖ (θ - 1)ʳ⁺²,   F = Σₖ Rₖ Bₖ,
/// ```
///
/// takes `aₖ`, `bₖ` from `Bₖ(0) = Bₖ'(0) = 0`.
fn startup_dense<S, T>(
    t: S,
    y: &Vector<T>,
    dt: S,
    q: usize,
    nodes: &[S],
    workspace: &mut MultistepWorkspace<S, T>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S>,
{
    let n = y.len();
    let conditions = nodes.len().max(2) - 1;
    let degree = conditions + 1;
    let integer = |value: usize| S::try_from_usize_exact(value);
    let sign = |power: usize| {
        if power.is_multiple_of(2) {
            S::ONE
        } else {
            -S::ONE
        }
    };
    let mut factorials = [S::ONE; CAPACITY];
    for k in 1..CAPACITY {
        factorials[k] = factorials[k - 1] * integer(k)?;
    }
    // Power coefficients of Bₖ, adding c (θ - 1)ʲ = c Σ_d C(j, d) (-1)ʲ⁻ᵈ θᵈ.
    let mut basis = [[S::ZERO; CAPACITY]; CAPACITY];
    for (k, row) in basis.iter_mut().enumerate().take(conditions) {
        let s0 = sign(k) / factorials[k];
        let s1 = if k == 0 {
            S::ZERO
        } else {
            sign(k - 1) / factorials[k - 1]
        };
        let a = -sign(conditions) * (s1 + integer(conditions + 1)? * s0);
        let b = -sign(conditions) * (s1 + integer(conditions)? * s0);
        for (scale, power) in [
            (S::ONE / factorials[k], k),
            (a, conditions),
            (b, conditions + 1),
        ] {
            let mut binomial = S::ONE;
            for (d, coefficient) in row.iter_mut().enumerate().take(power + 1) {
                *coefficient += scale * binomial * sign(power - d);
                binomial = binomial * integer(power - d)? / integer(d + 1)?;
            }
        }
    }
    let mut extrapolations: [Option<Weights<S>>; CAPACITY] = core::array::from_fn(|_| None);
    for (k, slot) in extrapolations
        .iter_mut()
        .enumerate()
        .take(conditions)
        .skip(1)
    {
        *slot = Some(Weights::new(&nodes[k - 1..], Functional::Value(S::ZERO))?);
    }
    let first = difference_row(q, 1);
    let (head, rows) = workspace.bdf_startup.as_mut_slice().split_at_mut(first * n);
    let coefficients = &mut head[..q * n];
    let row = |m: usize, k: usize, i: usize| rows[(difference_row(q, m) - first + k - 1) * n + i];
    let derivative = workspace.current_derivative.as_slice();
    let mut samples = [T::ZERO; CAPACITY];
    for i in 0..n {
        samples[0] = crate::safety::state_difference(
            workspace.proposed_state.as_slice()[i],
            y.as_slice()[i],
        )?;
        for (k, extrapolation) in extrapolations.iter().enumerate().take(conditions).skip(1) {
            let extrapolation = extrapolation.as_ref().ok_or_else(|| {
                crate::safety::invalid(
                    scioxide_core::argument!("workspace"),
                    "missing startup derivative weights",
                )
            })?;
            samples[k] = extrapolation.affine(T::ZERO, S::ONE, |j| row(k + j, k, i))?;
        }
        for d in 1..=degree {
            // R₀ = Δ - h f and R₁ = D₁ - h f; the remaining Rₖ are Dₖ.
            let mut weight = if d == 1 { S::ONE } else { S::ZERO } - basis[0][d];
            if conditions > 1 {
                weight -= basis[1][d];
            }
            let partial = crate::safety::stage_affine(
                T::ZERO,
                S::ONE,
                (0..conditions).map(|k| (basis[k][d], samples[k])),
            )?;
            coefficients[(d - 1) * n + i] =
                crate::safety::stage_affine(partial, dt, [(weight, derivative[i])])?;
        }
    }
    let coefficient = |d: usize, i: usize| coefficients[d * n + i];
    let extension = crate::dense_output::DenseExtension::Multistep { order: degree };
    workspace.dense.prepare_coefficients(
        extension,
        dt,
        y,
        |i, imaginary| {
            let mut values = [S::ZERO; 7];
            for (d, out) in values.iter_mut().enumerate().take(degree) {
                let value = coefficient(d, i);
                *out = if imaginary {
                    value.imaginary_component()
                } else {
                    value.real_component()
                };
            }
            Some(values)
        },
        |i, imaginary| {
            let mut values = [scioxide_core::Scaled::ZERO; 7];
            for (d, out) in values.iter_mut().enumerate().take(degree) {
                let value = coefficient(d, i);
                *out = scioxide_core::Scaled::from_value(if imaginary {
                    value.imaginary_component()
                } else {
                    value.real_component()
                })?;
            }
            Ok(values)
        },
    )?;
    workspace
        .dense
        .finish(extension, t, dt, y, &workspace.proposed_state)?;
    Ok(())
}

#[allow(clippy::needless_range_loop)]
/// Derivative weights of interpolation through the actual accepted nodes.
/// NDF subtracts κ γ (y_new − extrapolated_y), which keeps degree-q consistency on
/// arbitrary distinct nodes. At order five κ is zero.
pub(crate) fn coefficients<S: RealScalar + NumericScalar<Real = S>>(
    order: usize,
    formula: BdfFormula,
    old: &[S; CAPACITY],
    history: usize,
) -> Result<([S; CAPACITY], S), SciOxideError> {
    let mut nodes = [S::ONE; CAPACITY];
    nodes[1..=order].copy_from_slice(&old[..order]);
    let mut alpha = super::polynomial::weights(&nodes[..=order], true)?;
    let kappa = if formula == BdfFormula::Ndf && history >= order {
        S::try_from_f64_lossy([0.0, -0.185, -1.0 / 9.0, -0.0823, -0.0415, 0.0][order])?
    } else {
        S::ZERO
    };
    let correction = kappa * alpha[0];
    alpha[0] -= correction;
    Ok((alpha, correction))
}

/// Predictor and candidate error functionals for one exact normalized geometry.
#[derive(Debug)]
pub(crate) struct Prepared<S: RealScalar> {
    key: Option<(usize, BdfFormula, MultistepOrder, usize)>,
    nodes: [S; CAPACITY],
    alpha: [S; CAPACITY],
    correction: S,
    predictor: Weights<S>,
    errors: [Option<LocalError<S>>; 3],
}
impl<S: RealScalar> Default for Prepared<S> {
    fn default() -> Self {
        Self {
            key: None,
            nodes: [S::ZERO; CAPACITY],
            alpha: [S::ZERO; CAPACITY],
            correction: S::ZERO,
            predictor: Weights::from_native([S::ZERO; CAPACITY], 0, S::ONE),
            errors: core::array::from_fn(|_| None),
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> Prepared<S> {
    pub(crate) fn invalidate(&mut self) {
        self.key = None;
    }

    fn prepare(
        &mut self,
        order: usize,
        formula: BdfFormula,
        policy: MultistepOrder,
        old: &[S; CAPACITY],
        history: usize,
    ) -> Result<(), SciOxideError> {
        let history = history.min(order + 1);
        let key = (order, formula, policy, history);
        let count = (order + 2).min(history + 1);
        if self.key == Some(key) && self.nodes[..count] == old[..count] {
            return Ok(());
        }
        self.key = None;
        let ordinary = super::polynomial::ordinary_geometry(&old[..count]);
        let native = if ordinary {
            self.native::<false>(order, formula, policy, old, history)
        } else {
            self.native::<true>(order, formula, policy, old, history)
        };
        if native {
            self.nodes[..count].copy_from_slice(&old[..count]);
            self.key = Some(key);
            return Ok(());
        }
        let (alpha, correction) = coefficients(order, formula, old, history)?;
        self.alpha = alpha;
        self.correction = correction;
        self.predictor = Weights::new(&old[..=order.min(history)], Functional::Value(S::ONE))?;
        for (slot, candidate) in [order.saturating_sub(1), order, order + 1]
            .into_iter()
            .enumerate()
        {
            self.errors[slot] = if policy.automatic()
                && candidate > 0
                && candidate <= policy.maximum()
                && candidate <= history + 1
                && history > 0
            {
                let (alpha0, correction) = if candidate == order {
                    (alpha[0], correction)
                } else {
                    let (alpha, correction) = coefficients(candidate, formula, old, history)?;
                    (alpha[0], correction)
                };
                Some(LocalError::from_coefficients(
                    candidate, old, history, alpha0, correction,
                )?)
            } else {
                None
            };
        }
        self.nodes[..count].copy_from_slice(&old[..count]);
        self.key = Some(key);
        Ok(())
    }
    /// Append actual nodes once. Barycentric denominators give every leading
    /// divided difference, endpoint derivative and extrapolation functional.
    /// Denominator products follow the same node order as Weights::new.
    /// Non-normal intermediates keep the extended-range preparation below.
    fn native<const CHECK: bool>(
        &mut self,
        order: usize,
        formula: BdfFormula,
        policy: MultistepOrder,
        old: &[S; CAPACITY],
        history: usize,
    ) -> bool {
        let orders = [order.saturating_sub(1), order, order + 1];
        let admitted = |candidate| {
            policy.automatic()
                && candidate > 0
                && candidate <= policy.maximum()
                && candidate <= history + 1
                && history > 0
        };
        for (slot, candidate) in orders.into_iter().enumerate() {
            self.errors[slot] = admitted(candidate).then_some(LocalError {
                weights: None,
                factor: S::ONE,
            });
        }
        let predictor_order = order.min(history);
        let last = orders
            .into_iter()
            .filter(|&candidate| admitted(candidate) && history >= candidate)
            .map(|candidate| candidate + 1)
            .fold((predictor_order + 1).max(order), usize::max);
        if CHECK && !old[..last].iter().all(|&x| x.is_finite() && x < S::ONE) {
            return false;
        }
        let mut nodes = [S::ONE; CAPACITY];
        nodes[1..=last].copy_from_slice(&old[..last]);
        let mut denominators = [S::ONE; CAPACITY];
        let mut gamma = S::ZERO;
        for k in 1..=last {
            let node = nodes[k];
            let mut denominator = S::ONE;
            for j in 0..k {
                let difference = nodes[j] - node;
                denominators[j] *= difference;
                denominator *= -difference;
                if CHECK && (!denominators[j].is_normal() || !denominator.is_normal()) {
                    return false;
                }
            }
            denominators[k] = denominator;
            gamma += S::ONE / (S::ONE - node);
            if CHECK && !gamma.is_normal() {
                return false;
            }
            if k == order
                || orders
                    .into_iter()
                    .any(|candidate| candidate == k && admitted(candidate))
            {
                let kappa = if formula == BdfFormula::Ndf && history >= k {
                    let Ok(value) =
                        S::try_from_f64_lossy([0.0, -0.185, -1.0 / 9.0, -0.0823, -0.0415, 0.0][k])
                    else {
                        return false;
                    };
                    value
                } else {
                    S::ZERO
                };
                let correction = kappa * gamma;
                let alpha0 = gamma - correction;
                if CHECK && !alpha0.is_normal() {
                    return false;
                }
                if k == order {
                    self.alpha = [S::ZERO; CAPACITY];
                    self.alpha[0] = alpha0;
                    self.correction = correction;
                    for (j, alpha) in self.alpha.iter_mut().enumerate().take(k + 1).skip(1) {
                        *alpha = denominators[0] / ((S::ONE - nodes[j]) * denominators[j]);
                        if CHECK && !alpha.is_normal() {
                            return false;
                        }
                    }
                }
                for (slot, candidate) in orders.into_iter().enumerate() {
                    if candidate == k
                        && history >= k
                        && let Some(error) = self.errors[slot].as_mut()
                    {
                        let product = denominators[0];
                        error.factor =
                            (product + correction * product * (S::ONE - old[k])) / alpha0;
                        if CHECK && !error.factor.is_normal() {
                            return false;
                        }
                    }
                }
            }
            for (slot, candidate) in orders.into_iter().enumerate() {
                if candidate + 1 == k
                    && history >= candidate
                    && let Some(error) = self.errors[slot].as_mut()
                {
                    let mut weights = [S::ZERO; CAPACITY];
                    for j in 0..=k {
                        weights[j] = S::ONE / denominators[j];
                        if CHECK && !weights[j].is_normal() {
                            return false;
                        }
                    }
                    error.weights = Some(Weights::from_native(weights, k + 1, S::ZERO));
                }
            }
            if k == predictor_order + 1 {
                let mut weights = [S::ZERO; CAPACITY];
                for j in 0..k {
                    weights[j] = -denominators[0] / denominators[j + 1];
                    if CHECK && !weights[j].is_normal() {
                        return false;
                    }
                }
                self.predictor.replace_native(weights, k, S::ONE);
            }
        }
        true
    }
}

/// Actual-node BDF/NDF local error; an extra history node is required for a
/// divided difference. Cold startup conservatively uses the state increment.
#[derive(Debug)]
pub(crate) struct LocalError<S: RealScalar> {
    weights: Option<Weights<S>>,
    factor: S,
}
impl<S: RealScalar + NumericScalar<Real = S>> LocalError<S> {
    pub(crate) fn new(
        order: usize,
        formula: BdfFormula,
        old: &[S; CAPACITY],
        history: usize,
    ) -> Result<Self, SciOxideError> {
        let (alpha, correction) = coefficients(order, formula, old, history)?;
        Self::from_coefficients(order, old, history, alpha[0], correction)
    }

    fn from_coefficients(
        order: usize,
        old: &[S; CAPACITY],
        history: usize,
        alpha0: S,
        correction: S,
    ) -> Result<Self, SciOxideError> {
        let mut nodes = [S::ONE; CAPACITY];
        nodes[1..=order + 1].copy_from_slice(&old[..=order]);
        let factor = if history < order {
            None
        } else {
            let product = old[..order]
                .iter()
                .fold(S::ONE, |product, x| product * (S::ONE - *x));
            Some((product + correction * product * (S::ONE - old[order])) / alpha0)
        };
        let weights = if factor.is_some() {
            Some(Weights::new(
                &nodes[..=order + 1],
                Functional::Coefficient(order + 1),
            )?)
        } else {
            None
        };
        Ok(Self {
            weights,
            factor: factor.unwrap_or(S::ONE),
        })
    }
    pub(crate) fn component<T: NumericScalar<Real = S>>(
        &self,
        candidate: T,
        current: T,
        previous: impl Fn(usize) -> T,
    ) -> Result<T, SciOxideError> {
        let Some(weights) = &self.weights else {
            return crate::safety::state_difference(candidate, current);
        };
        weights.affine(T::ZERO, self.factor, |j| match j {
            0 => candidate,
            1 => current,
            _ => previous(j - 2),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod endpoint_tests {
    use super::*;
    use crate::stepper::{StepTrace, testing::with_context};

    #[test]
    fn prepared_coefficients_reproduce_polynomials_after_geometry_and_formula_changes() {
        let mut prepared = Prepared::<f64>::default();
        for old in [
            [0., -1., -2., -3., -4., -5., -6., -7.],
            [0., -0.5, -1.75, -2.25, -4., -5.5, -7., -9.],
        ] {
            for formula in [BdfFormula::Bdf, BdfFormula::Ndf] {
                for order in 1..=5 {
                    prepared
                        .prepare(order, formula, MultistepOrder::Automatic(5), &old, 6)
                        .unwrap();
                    let reference = LocalError::new(order, formula, &old, 6).unwrap();
                    let degree = (order + 1) as i32;
                    let expected = reference
                        .component(1., 0., |i| old[i + 1].powi(degree))
                        .unwrap();
                    let actual = prepared.errors[1]
                        .as_ref()
                        .unwrap()
                        .component(1., 0., |i| old[i + 1].powi(degree))
                        .unwrap();
                    assert!((actual - expected).abs() < 1e-9 * expected.abs().max(1.));
                    for degree in 0..=order {
                        let value = prepared
                            .predictor
                            .affine(0., 1., |i| old[i].powi(degree as i32))
                            .unwrap();
                        assert!((value - 1.).abs() < 1e-10, "order={order}, degree={degree}");
                        let derivative = prepared.alpha[0]
                            + prepared.alpha[1..=order]
                                .iter()
                                .enumerate()
                                .map(|(i, &a)| a * old[i].powi(degree as i32))
                                .sum::<f64>();
                        assert!((derivative + prepared.correction - degree as f64).abs() < 1e-10);
                        let error = prepared.errors[1]
                            .as_ref()
                            .unwrap()
                            .component(1., old[0].powi(degree as i32), |i| {
                                old[i + 1].powi(degree as i32)
                            })
                            .unwrap();
                        assert!(
                            error.abs() < 1e-10,
                            "order={order}, degree={degree}, error={error}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn unchanged_exceptional_geometry_reuses_prepared_weights_without_allocation() {
        let mut prepared = Prepared::<f64>::default();
        let mut old = [0.; CAPACITY];
        old[1] = -1e-310;
        prepared
            .prepare(1, BdfFormula::Bdf, MultistepOrder::Fixed(1), &old, 1)
            .unwrap();
        {
            prepared
                .prepare(1, BdfFormula::Bdf, MultistepOrder::Fixed(1), &old, 1)
                .unwrap();
            let value = prepared.predictor.affine(0., 1., |i| old[i]).unwrap();
            assert!((value - 1.).abs() < 1e-14);
        };
        old[1] = -1.;
        prepared
            .prepare(1, BdfFormula::Bdf, MultistepOrder::Fixed(1), &old, 1)
            .unwrap();
        assert!((prepared.predictor.affine(0., 1., |i| [2., 1.][i]).unwrap() - 3.).abs() < 1e-14);
        prepared.invalidate();
        assert!(prepared.key.is_none());
        let coincident = [0.; CAPACITY];
        assert!(
            prepared
                .prepare(2, BdfFormula::Bdf, MultistepOrder::Fixed(2), &coincident, 2)
                .is_err()
        );
        assert!(prepared.key.is_none());
        prepared
            .prepare(1, BdfFormula::Ndf, MultistepOrder::Automatic(5), &old, 0)
            .unwrap();
        assert!(prepared.errors.iter().all(Option::is_none));
    }

    #[test]
    fn endpoint_rhs_is_the_final_corrected_derivative_even_without_dense_output() {
        let system = crate::OdeFn::new(1, |_: f64, y: &[f64], out: &mut [f64]| {
            out[0] = -y[0] * y[0];
            Ok(())
        })
        .unwrap();
        for policy in [MultistepOrder::Fixed(1), MultistepOrder::Automatic(5)] {
            let solver = Bdf::<f64>::with_order_policy(policy).unwrap();
            let mut workspace = MultistepWorkspace::default();
            solver.reserve(1, false, &mut workspace).unwrap();
            solver.begin(1, false, &mut workspace).unwrap();
            let trace = StepTrace::default();
            let mut state = Vector::from_vec(alloc::vec![1.]);
            with_context(|cx| {
                let mut time = 0.;
                for index in 0..3 {
                    let previous = state.as_slice()[0];
                    let result = solver.step(
                        &system,
                        time,
                        &mut state,
                        0.01,
                        StepRequest::new(&trace),
                        &mut workspace,
                        cx,
                    )?;
                    if let Some(error) = result.normalized_error {
                        let expected = scioxide_core::ErrorTolerance::default()
                            .normalized_norm(
                                &[result.raw_error.unwrap()[0].abs()],
                                &[previous.abs().max(state.as_slice()[0].abs())],
                            )
                            .unwrap();
                        assert!((error - expected).abs() < 1e-12 * expected.max(1.));
                    } else {
                        assert!(!policy.automatic());
                    }
                    let derivative = result.endpoint_rhs.unwrap()[0];
                    assert!((derivative + state.as_slice()[0] * state.as_slice()[0]).abs() < 1e-14);
                    assert!(result.dense_output.is_none());
                    if index == 0 {
                        solver.discard(&mut workspace)?;
                        state.as_mut_slice()[0] = 1.;
                    } else {
                        solver.accept(&mut workspace)?;
                        time += 0.01;
                    }
                }
                Ok(())
            })
            .unwrap();
        }
    }

    #[test]
    fn extrapolated_startup_does_not_publish_a_subdivision_derivative() {
        let system = crate::OdeFn::new(1, |_: f64, y: &[f64], out: &mut [f64]| {
            out[0] = -y[0];
            Ok(())
        })
        .unwrap();
        let solver = Bdf::<f64>::new(3).unwrap();
        let mut workspace = MultistepWorkspace::default();
        solver.reserve(1, false, &mut workspace).unwrap();
        solver.begin(1, false, &mut workspace).unwrap();
        let trace = StepTrace::default();
        let mut state = Vector::from_vec(alloc::vec![1.]);
        with_context(|cx| {
            let result = solver.step(
                &system,
                0.,
                &mut state,
                0.01,
                StepRequest::new(&trace),
                &mut workspace,
                cx,
            )?;
            assert!(result.endpoint_rhs.is_none());
            Ok(())
        })
        .unwrap();
    }
}
