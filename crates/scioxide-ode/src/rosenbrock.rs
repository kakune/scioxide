//! Rosenbrock-W methods: linearly implicit stages with an approximate Jacobian.
use crate::OdeStageBackend;
use crate::OdeSystem;
use crate::dense_output::DenseExtension;
use crate::result::StepResult;
use crate::safety::invalid;
use crate::stepper::{DefaultController, MethodInfo, StepRequest, Stepper};
use crate::stiff::StageWork;
use crate::stiff::kernel::{Coupling, KernelShape, StageKernel};
use crate::workspace::ExplicitOdeWorkspace;
use core::marker::PhantomData;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError, SolveContext};
use scioxide_differentiate::{DerivativeCandidate, DerivativeStrategy};
use scioxide_linalg::Vector;

/// Selects the coefficient set of a [`RosenbrockW`] method.
///
/// Both sets are W-methods: they keep their order when the Jacobian is only
/// approximated, while their stability depends on how good the approximation is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RosenbrockMethod {
    /// ROS2 of Verwer et al.: two stages, order 2, `γ = 1 + 1/√2`, with a
    /// first-order embedded error estimate. L-stable with the exact Jacobian.
    Ros2,
    /// ROS34PW2 of Rang and Angermann: four stages, order 3, with a second-order
    /// embedded error estimate. L-stable with the exact Jacobian.
    Ros34Pw2,
}
#[derive(Debug, Clone)]
struct Tableau<S: RealScalar> {
    a: [[S; 4]; 4],
    gamma: [[S; 4]; 4],
    b: [S; 4],
    embedded: [S; 4],
    stages: usize,
}
impl<S: RealScalar> Tableau<S> {
    fn new(method: RosenbrockMethod) -> Result<Self, SciOxideError> {
        // Rang & Angermann (2005), ROS34PW2; Verwer et al. (1999), ROS2.
        // A/Gamma convention independently cross-checked against PETSc rosw.c
        // (3.25): https://petsc.org/release/src/ts/impls/rosw/rosw.c.html
        // These coefficients act on untransformed stage increments.
        let (a, gamma, b, embedded, stages) = match method {
            RosenbrockMethod::Ros2 => {
                let g = 1.0 + core::f64::consts::FRAC_1_SQRT_2;
                (
                    [[0.; 4], [1., 0., 0., 0.], [0.; 4], [0.; 4]],
                    [[g, 0., 0., 0.], [-2. * g, g, 0., 0.], [0.; 4], [0.; 4]],
                    [0.5, 0.5, 0., 0.],
                    [1., 0., 0., 0.],
                    2,
                )
            }
            RosenbrockMethod::Ros34Pw2 => (
                [
                    [0.; 4],
                    [0.871733043016918, 0., 0., 0.],
                    [0.8445706001536942, -0.11299064236484185, 0., 0.],
                    [0., 0., 1., 0.],
                ],
                [
                    [0.435866521508459, 0., 0., 0.],
                    [-0.871733043016918, 0.435866521508459, 0., 0.],
                    [
                        -0.9033805701304408,
                        0.054180672388095326,
                        0.435866521508459,
                        0.,
                    ],
                    [
                        0.24212380706095346,
                        -1.2232505839045147,
                        0.5452602553351021,
                        0.435866521508459,
                    ],
                ],
                [
                    0.24212380706095346,
                    -1.2232505839045147,
                    1.545260255335102,
                    0.435866521508459,
                ],
                [
                    0.3781090314581937,
                    -0.09604229221242318,
                    0.5,
                    0.2179332607542295,
                ],
                4,
            ),
        };
        let mut result = Self {
            a: [[S::ZERO; 4]; 4],
            gamma: [[S::ZERO; 4]; 4],
            b: [S::ZERO; 4],
            embedded: [S::ZERO; 4],
            stages,
        };
        for i in 0..4 {
            for j in 0..4 {
                result.a[i][j] = S::try_from_f64_lossy(a[i][j])?;
                result.gamma[i][j] = S::try_from_f64_lossy(gamma[i][j])?;
            }
            result.b[i] = S::try_from_f64_lossy(b[i])?;
            result.embedded[i] = S::try_from_f64_lossy(embedded[i])?;
        }
        Ok(result)
    }
}

/// Integrates stiff problems with an adaptive Rosenbrock-W method.
///
/// A Rosenbrock method is linearly implicit: each stage solves one linear system
/// with the matrix `I − hγJ`, where `h` is the step, `γ` the diagonal coefficient
/// of the method and `J` an approximation of `∂f/∂y`. There is no Newton
/// iteration. A W-method keeps its order for any `J`, so the Jacobian can be
/// approximate or out of date. The method coefficients are selected with
/// [`RosenbrockMethod`].
///
/// One attempt of an `s`-stage method costs `s − 1` evaluations of `f`, one
/// factorization and `s − 1` Jacobian–vector products. `f(tₙ, yₙ)` and
/// `∂f/∂t(tₙ, yₙ)` are evaluated once per accepted step and reused when a step is
/// rejected. The Jacobian is recomputed after five accepted steps, after a
/// rejection, or when the time or a state component moves by more than half of
/// `max(1, |old|, |new|)`; a Jacobian declared constant by
/// [`OdeSystem::jacobian_is_constant`](crate::OdeSystem::jacobian_is_constant) is
/// never recomputed. The factorization always uses the current `h`.
///
/// The derivative strategy selects the sources of both `J` and `∂f/∂t`. The time
/// derivative comes from
/// [`OdeSystem::time_derivative`](crate::OdeSystem::time_derivative) when it
/// returns a value, and otherwise from forward or central differences in `t` with
/// scale `max(|t|, 1)`; automatic differentiation is not available for it. An
/// autonomous system should return a zero time derivative from that callback.
///
/// Dense output is the cubic Hermite interpolant of the step, and events are
/// supported. Only the identity mass matrix is supported; a system with a mass
/// matrix is rejected with `InvalidInput` (`UnsupportedConfiguration`).
///
/// # Examples
///
/// ```
/// use scioxide_core::{ErrorTolerance, NormSelection};
/// use scioxide_linalg::Vector;
/// use scioxide_ode::{OdeFn, OdeOptions, OdeMethod, OdeProblem, OdeState, RosenbrockMethod, RosenbrockW};
/// // y' = −10 y, y(0) = 1, so y(1) = e⁻¹⁰.
/// let system = OdeFn::new(1, |_: f64, y: &[f64], f: &mut [f64]| {
///     f[0] = -10.0 * y[0];
///     Ok(())
/// })?;
/// let tolerance = ErrorTolerance::scalar(1e-10, 1e-7, NormSelection::RootMeanSquare)?;
/// let options = OdeOptions::new().with_error_tolerance(tolerance);
/// let plan = (RosenbrockW::<f64>::new(RosenbrockMethod::Ros34Pw2)?, options);
/// let initial = OdeState::new(0.0, Vector::from_vec(vec![1.0]));
/// let solution = plan.0.solve(&OdeProblem::new(&system, 1.0), &plan.1, initial)?;
/// assert!(solution.report().is_success());
/// assert!((solution.state().values()[0] - (-10.0_f64).exp()).abs() < 1e-6);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - J. Rang and L. Angermann, "New Rosenbrock W-methods of order 3 for partial
///   differential algebraic equations of index 1", *BIT Numer. Math.* 45(4),
///   761–787, 2005.
/// - J. G. Verwer, E. J. Spee, J. G. Blom and W. Hundsdorfer, "A second-order
///   Rosenbrock method applied to photochemical dispersion problems",
///   *SIAM J. Sci. Comput.* 20(4), 1456–1480, 1999.
/// - E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II*,
///   2nd ed., Springer, 1996, §IV.7.
#[derive(Debug, Clone)]
pub struct RosenbrockW<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S> = S> {
    method: RosenbrockMethod,
    tableau: Tableau<S>,
    backend: OdeStageBackend,
    strategy: DerivativeStrategy,
    marker: PhantomData<T>,
}
impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> RosenbrockW<S, T> {
    /// Creates the method with the coefficients of `method`, dense LU stages and the
    /// default derivative sources.
    ///
    /// # Errors
    ///
    /// Returns the conversion error if a coefficient cannot be represented in `S`.
    pub fn new(method: RosenbrockMethod) -> Result<Self, SciOxideError> {
        Ok(Self {
            method,
            tableau: Tableau::new(method)?,
            backend: OdeStageBackend::Dense,
            strategy: crate::stiff::DEFAULT_STRATEGY,
            marker: PhantomData,
        })
    }
    /// Selects the linear solver of the stage systems: dense LU (the default),
    /// sparse LU or restarted GMRES.
    ///
    /// See [`OdeStageBackend`].
    pub fn with_backend(mut self, backend: OdeStageBackend) -> Self {
        self.backend = backend;
        self
    }
    /// Sets the order in which the sources of `J` and `∂f/∂t` are tried.
    ///
    /// The default tries the analytic derivative, then automatic differentiation
    /// (for `J` only), then forward differences. Since a W-method keeps its order
    /// with an approximate Jacobian, finite differences are always a valid choice.
    #[must_use]
    pub fn with_derivative_strategy(mut self, strategy: DerivativeStrategy) -> Self {
        self.strategy = strategy;
        self
    }
    fn shape(&self) -> KernelShape<'_> {
        KernelShape {
            layout: crate::stiff::kernel::StageLayout::Coupled,
            backend: &self.backend,
            stages: 1,
            blocks: 1,
            provided: false,
        }
    }
}
/// Stage, factorization or Krylov, error and dense-output storage of a Rosenbrock-W method.
pub(crate) struct RosenbrockWorkspace<
    S: RealScalar + NumericScalar<Real = S>,
    T: NumericScalar<Real = S> = S,
> {
    explicit: ExplicitOdeWorkspace<S, T>,
    kernel: Option<StageKernel<S, T>>,
    current_rhs: Vector<T>,
    time_derivative: Vector<T>,
    /// Accepted base step whose `∂f/∂t` `time_derivative` holds; retries reuse it.
    time_derivative_step: Option<usize>,
    /// Source of `∂f/∂t` selected in this solve epoch.
    time_derivative_source: Option<DerivativeCandidate>,
    weighted: Vector<T>,
    product: Vector<T>,
    rhs: Vector<T>,
}
impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Default
    for RosenbrockWorkspace<S, T>
{
    fn default() -> Self {
        Self {
            explicit: ExplicitOdeWorkspace::new(),
            kernel: None,
            current_rhs: Vector::empty(),
            time_derivative: Vector::empty(),
            time_derivative_step: None,
            time_derivative_source: None,
            weighted: Vector::empty(),
            product: Vector::empty(),
            rhs: Vector::empty(),
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> RosenbrockW<S, T> {
    const fn name(&self) -> &'static str {
        match self.method {
            RosenbrockMethod::Ros2 => "RosenbrockW/ROS2",
            RosenbrockMethod::Ros34Pw2 => "RosenbrockW/ROS34PW2",
        }
    }
    const fn order(&self) -> usize {
        match self.method {
            RosenbrockMethod::Ros2 => 2,
            RosenbrockMethod::Ros34Pw2 => 3,
        }
    }
    fn require_fit(&self, n: usize, w: &RosenbrockWorkspace<S, T>) -> Result<(), SciOxideError> {
        if Stepper::<S, T>::fits(self, n, false, w) {
            Ok(())
        } else {
            Err(invalid(
                scioxide_core::argument!("workspace"),
                "prepare matching Rosenbrock workspace",
            ))
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>, T: NumericScalar<Real = S>> Stepper<S, T>
    for RosenbrockW<S, T>
{
    const ADAPTIVE: Option<bool> = Some(true);
    const BORROWS_BASE_STATE: bool = true;
    const CHECKED_RESULT: bool = true;

    type Workspace = RosenbrockWorkspace<S, T>;
    fn stage_kernel<'w>(&self, workspace: &'w Self::Workspace) -> Option<&'w StageKernel<S, T>> {
        workspace.kernel.as_ref()
    }
    /// Both coefficient sets are L-stable with the exact Jacobian; with an
    /// approximate Jacobian, stability depends on the approximation.
    fn stability(&self) -> crate::OdeLinearStability<S> {
        crate::OdeLinearStability::unconditional()
    }

    fn info(&self) -> MethodInfo {
        MethodInfo {
            name: self.name(),
            algorithm: match self.method {
                RosenbrockMethod::Ros2 => {
                    scioxide_core::algorithm_id!("ode", "RosenbrockW-ROS2", 1)
                }
                RosenbrockMethod::Ros34Pw2 => {
                    scioxide_core::algorithm_id!("ode", "RosenbrockW-ROS34PW2", 1)
                }
            },
            order: self.order(),
            initial_step_order: self.order(),
            controller: DefaultController::ProportionalIntegral,
            dense_output: true,
            events: true,
            adaptive: true,
            mass_matrices: false,
            injectable: false,
        }
    }
    fn fits(&self, n: usize, _provided: bool, w: &Self::Workspace) -> bool {
        w.kernel
            .as_ref()
            .is_some_and(|kernel| kernel.matches(n, self.shape()))
            && w.explicit.fits(n, self.tableau.stages, 3)
            && w.explicit.stage_capacity() == self.tableau.stages
            && [
                &w.current_rhs,
                &w.time_derivative,
                &w.weighted,
                &w.product,
                &w.rhs,
            ]
            .iter()
            .all(|vector| vector.len() == n)
    }
    fn reserve(
        &self,
        n: usize,
        _provided: bool,
        w: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        if self.fits(n, false, w) {
            return Ok(());
        }
        let mut explicit = ExplicitOdeWorkspace::new();
        explicit.try_resize(n, self.tableau.stages, 3)?;
        *w = RosenbrockWorkspace {
            explicit,
            kernel: Some(StageKernel::new(n, self.shape())?),
            current_rhs: Vector::zeros(n)?,
            time_derivative: Vector::zeros(n)?,
            time_derivative_step: None,
            time_derivative_source: None,
            weighted: Vector::zeros(n)?,
            product: Vector::zeros(n)?,
            rhs: Vector::zeros(n)?,
        };
        Ok(())
    }
    fn begin(
        &self,
        n: usize,
        _provided: bool,
        w: &mut Self::Workspace,
    ) -> Result<(), SciOxideError> {
        self.require_fit(n, w)?;
        if let Some(kernel) = &mut w.kernel {
            kernel.invalidate()?;
            kernel.initialize_identity_mass();
        }
        w.explicit.reset_epoch();
        w.time_derivative_step = None;
        w.time_derivative_source = None;
        Ok(())
    }
    fn accept(&self, w: &mut Self::Workspace) -> Result<(), SciOxideError> {
        if let Some(kernel) = &mut w.kernel {
            kernel.accept_approximation(5);
        }
        w.time_derivative_step = None;
        if core::mem::take(&mut w.explicit.pending_endpoint) {
            core::mem::swap(&mut w.current_rhs, &mut w.explicit.endpoint_rhs);
            w.explicit.cache.advance();
        } else {
            w.explicit.cache.reset();
        }
        Ok(())
    }
    fn discard(&self, w: &mut Self::Workspace) -> Result<(), SciOxideError> {
        if let Some(kernel) = &mut w.kernel {
            kernel.reject_approximation()?;
        }
        w.explicit.discard_step();
        Ok(())
    }
    /// Evaluate the endpoint derivative of the accepted step, which then serves as
    /// the next base derivative, and build the cubic Hermite interpolant.
    fn complete_dense<'w, System: OdeSystem<S, T> + ?Sized>(
        &self,
        system: &System,
        time: S,
        h: S,
        request: StepRequest<'_, S, T>,
        w: &'w mut Self::Workspace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<&'w crate::dense_output::DenseOutputStorage<S, T>, SciOxideError> {
        let Some((_, state)) = request.base else {
            return Err(invalid(
                scioxide_core::argument!("request"),
                "dense completion requires the accepted base",
            ));
        };
        let e = &mut w.explicit;
        crate::traits::evaluate_rhs_staged(
            system,
            request.endpoint(time, h)?,
            &e.trial,
            &mut e.endpoint_rhs,
            &mut e.rhs_candidate,
            request,
            cx,
        )?;
        e.dense_output
            .prepare_endpoints(h, state, &e.trial, &w.current_rhs, &e.endpoint_rhs)?;
        e.dense_output
            .finish(DenseExtension::CubicHermite, time, h, state, &e.trial)?;
        e.pending_endpoint = true;
        Ok(&e.dense_output)
    }
    fn step<'w, System: OdeSystem<S, T> + ?Sized>(
        &self,
        system: &System,
        time: S,
        output: &mut Vector<T>,
        h: S,
        request: StepRequest<'_, S, T>,
        w: &'w mut Self::Workspace,
        cx: &mut SolveContext<'_, '_>,
    ) -> Result<StepResult<'w, S, T>, SciOxideError> {
        let state = request.base.map_or(&*output, |(_, base)| base);
        request.endpoint(time, h)?;
        let RosenbrockWorkspace {
            explicit: e,
            kernel,
            current_rhs,
            time_derivative,
            time_derivative_step,
            time_derivative_source,
            weighted,
            product,
            rhs,
        } = w;
        let kernel = kernel.as_mut().ok_or_else(|| {
            invalid(
                scioxide_core::argument!("workspace"),
                "prepare matching Rosenbrock workspace",
            )
        })?;
        let before = kernel.stats();
        let work = StageWork::default();
        e.pending_endpoint = false;
        e.cache.evaluate_into(
            system,
            time,
            state,
            request,
            current_rhs,
            &mut e.rhs_candidate,
            cx,
        )?;
        kernel.linearize(
            system,
            self.strategy,
            0,
            time,
            state.as_slice(),
            current_rhs.as_slice(),
            false,
            request,
            cx,
        )?;
        let gamma = [self.tableau.gamma[0][0]];
        let node = [S::ZERO];
        kernel.factor(
            Coupling {
                time,
                step: h,
                coefficients: &gamma,
                nodes: &node,
            },
            false,
            request,
            cx,
        )?;
        // Like the base derivative, ∂f/∂t at the accepted base serves every retry.
        if request.accepted_step.is_none() || *time_derivative_step != request.accepted_step {
            *time_derivative_step = None;
            crate::stiff::jacobian::time_derivative(
                system,
                self.strategy,
                time_derivative_source,
                time,
                state.as_slice(),
                current_rhs.as_slice(),
                time_derivative,
                &mut e.work,
                &mut e.trial,
                &mut e.rhs_candidate,
                request,
                cx,
            )?;
            *time_derivative_step = request.accepted_step;
        }

        let accuracy = crate::StageAccuracy::from_request(state.as_slice(), request);
        let n = state.len();
        for i in 0..self.tableau.stages {
            let node = self.tableau.a[i][..i]
                .iter()
                .copied()
                .fold(S::ZERO, |a, b| a + b);
            let gamma = self.tableau.gamma[i][..=i]
                .iter()
                .copied()
                .fold(S::ZERO, |a, b| a + b);
            let mut coupled = false;
            for k in 0..n {
                e.work.as_mut_slice()[k] = crate::safety::stage_affine(
                    state.as_slice()[k],
                    S::ONE,
                    self.tableau.a[i][..i]
                        .iter()
                        .zip(&e.stages[..i])
                        .map(|(&a, kj)| (a, kj.as_slice()[k])),
                )?;
                let value = crate::safety::stage_affine(
                    T::ZERO,
                    S::ONE,
                    self.tableau.gamma[i][..i]
                        .iter()
                        .zip(&e.stages[..i])
                        .map(|(&a, kj)| (a, kj.as_slice()[k])),
                )?;
                coupled |= value != T::ZERO;
                weighted.as_mut_slice()[k] = value;
            }
            if i == 0 {
                scioxide_linalg::VectorView::from(&*current_rhs).copy_into(
                    scioxide_linalg::CopyPolicy::Preserve,
                    &scioxide_linalg::LinalgExecutionOptions::new(),
                    (e.endpoint_rhs.view_mut()).reborrow(),
                    &mut scioxide_linalg::LinalgWorkspace::new(),
                    cx.execution(),
                )?;
            } else {
                let t = crate::safety::checked_stage_time(time, h, node)?;
                crate::traits::evaluate_rhs_staged(
                    system,
                    t,
                    &e.work,
                    &mut e.endpoint_rhs,
                    &mut e.rhs_candidate,
                    request,
                    cx,
                )?;
            }
            // J·0 = 0 exactly: the first stage, and any stage whose earlier stages
            // combine to zero, needs no Jacobian product.
            if coupled {
                kernel.apply_jacobian(
                    system,
                    request,
                    weighted.as_slice(),
                    product.as_mut_slice(),
                    cx,
                )?;
            } else {
                product.as_mut_slice().fill(T::ZERO);
            }
            for k in 0..n {
                rhs.as_mut_slice()[k] = crate::safety::stage_affine(
                    T::ZERO,
                    h,
                    [
                        (S::ONE, e.endpoint_rhs.as_slice()[k]),
                        (S::ONE, product.as_slice()[k]),
                        (h * gamma, time_derivative.as_slice()[k]),
                    ],
                )?;
            }
            kernel.solve(
                system,
                request,
                accuracy,
                state.as_slice(),
                rhs.as_slice(),
                e.stages[i].as_mut_slice(),
                None,
                cx,
            )?;
        }
        for k in 0..n {
            e.trial.as_mut_slice()[k] = crate::safety::stage_affine(
                state.as_slice()[k],
                S::ONE,
                self.tableau.b[..self.tableau.stages]
                    .iter()
                    .zip(&e.stages)
                    .map(|(&b, kj)| (b, kj.as_slice()[k])),
            )?;
            e.error.as_mut_slice()[k] = crate::safety::stage_affine(
                T::ZERO,
                S::ONE,
                self.tableau.b[..self.tableau.stages]
                    .iter()
                    .zip(&self.tableau.embedded)
                    .zip(&e.stages)
                    .map(|((&b, &bh), kj)| (b - bh, kj.as_slice()[k])),
            )?;
        }
        output.as_mut_slice().copy_from_slice(e.trial.as_slice());
        // Dense output waits for acceptance and a reader; see `complete_dense`.
        let mut result = StepResult::from_checked_parts(Some(e.error.as_slice()), None);
        result.dense_deferred = request.requires_dense_output();
        work.record(&mut result, before, kernel.stats())?;
        Ok(result)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::EvaluationRole;
    #[test]
    fn dense_acceptance_preserves_interval_and_endpoints() {
        for method in [RosenbrockMethod::Ros2, RosenbrockMethod::Ros34Pw2] {
            for h in [0.125, -0.125] {
                let method = RosenbrockW::new(method).unwrap();
                let system = crate::OdeFn::new(1, |_: f64, y: &[f64], out: &mut [f64]| {
                    out[0] = -y[0];
                    Ok(())
                })
                .unwrap();
                let mut workspace = RosenbrockWorkspace::default();
                method.reserve(1, false, &mut workspace).unwrap();
                method.begin(1, false, &mut workspace).unwrap();
                let base = Vector::from_vec(vec![1.0]);
                let mut state = base.clone();
                let trace = crate::stepper::StepTrace::default();
                let request = StepRequest {
                    accepted_step: Some(0),
                    base: Some((0.0, &base)),
                    ..StepRequest::new(&trace).with_dense_output()
                };
                crate::stepper::testing::with_context(|cx| {
                    let result =
                        method.step(&system, 0.0, &mut state, h, request, &mut workspace, cx)?;
                    // The interpolant and its endpoint derivative wait for a reader.
                    assert!(result.dense_output.is_none() && result.dense_deferred);
                    result.validate_for_step(request, 0.0, h, 1)?;
                    let calls = cx.counts().own().provider(EvaluationRole::RightHandSide);
                    let dense =
                        method.complete_dense(&system, 0.0, h, request, &mut workspace, cx)?;
                    assert_eq!(dense.interval(), (0.0, h));
                    let mut output = [0.0];
                    dense.eval_into(0.0, &mut output)?;
                    assert!((output[0] - 1.0).abs() < f64::EPSILON);
                    dense.eval_into(h, &mut output)?;
                    assert!((output[0] - state.as_slice()[0]).abs() < f64::EPSILON);
                    assert_eq!(
                        cx.counts().own().provider(EvaluationRole::RightHandSide),
                        calls + 1
                    );
                    method.accept(&mut workspace)?;
                    // The endpoint derivative becomes the next base derivative.
                    assert!(workspace.explicit.cache.contains(Some(1)));
                    assert!(
                        (workspace.current_rhs.as_slice()[0] + state.as_slice()[0]).abs() < 1e-15
                    );
                    Ok(())
                })
                .unwrap();
            }
        }
    }
}
