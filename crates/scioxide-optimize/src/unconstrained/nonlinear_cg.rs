//! Nonlinear conjugate gradient (Polak–Ribière+).

use scioxide_core::{
    AttemptKind, ExecutionContext, NumericScalar, RealScalar, SciOxideError, SolveScope,
};

use super::settings::{LineSearchSettings, iteration_budgets, line_search_settings};
use crate::attempt::{Entry, Shape};
use crate::result::{OptimizeResult, OptimizeTermination};
use crate::traits::{LineSearch, ObjectiveFunction};
use crate::unconstrained::FirstOrderStorage;
use crate::utils::{
    finish_first_order, get_gradient_into, line_search_termination, norm2, validate_x0,
};

use crate::quasi_newton::{positive, scaled_dot};
use scioxide_core::{Scaled, ScaledAccumulator};

fn conjugate_coefficient<S: RealScalar>(
    old: &[S],
    new: &[S],
    periodic: bool,
) -> Result<(S, Option<S>, Option<crate::CgRestartReason>), SciOxideError> {
    let old_squared = scaled_dot(old, old)?;
    let new_squared = scaled_dot(new, new)?;
    let scale = old_squared.mul(new_squared)?.sqrt()?;
    let orthogonality = if scale.is_zero() {
        None
    } else {
        Some(scaled_dot(old, new)?.abs().div(scale)?.round()?.min(S::ONE))
    };
    if periodic {
        return Ok((
            S::ZERO,
            orthogonality,
            Some(crate::CgRestartReason::Periodic),
        ));
    }
    if !positive(old_squared) {
        return Ok((
            S::ZERO,
            orthogonality,
            Some(crate::CgRestartReason::DescentLoss),
        ));
    }
    let ordinary = (old.len() >= 16)
        .then(|| {
            old.iter()
                .zip(new)
                .try_fold(
                    scioxide_core::CompensatedSum::ZERO,
                    |mut sum, (&old, &new)| {
                        let delta = new - old;
                        let product = new * delta;
                        if !delta.is_finite()
                            || scioxide_core::backend::standard::product_needs_retry(
                                new, delta, product,
                            )
                        {
                            return None;
                        }
                        sum.add(product).ok()?;
                        Some(sum)
                    },
                )
                .and_then(|sum| sum.total().ok())
        })
        .flatten();
    let numerator = if let Some(value) = ordinary {
        Scaled::from_value(value)?
    } else {
        let mut numerator = ScaledAccumulator::ZERO;
        for (&old, &new) in old.iter().zip(new) {
            numerator.add_product(
                Scaled::from_value(new)?,
                Scaled::from_value(new)?.sub(Scaled::from_value(old)?)?,
            )?;
        }
        numerator.total()?
    };
    if numerator.is_sign_negative() {
        return Ok((
            S::ZERO,
            orthogonality,
            Some(crate::CgRestartReason::NegativeBeta),
        ));
    }
    match numerator.div(old_squared)?.round() {
        Ok(beta) => Ok((beta, orthogonality, None)),
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            ) =>
        {
            Ok((
                S::ZERO,
                orthogonality,
                Some(crate::CgRestartReason::Unrepresentable),
            ))
        }
        Err(error) => Err(error),
    }
}

/// Sets the stopping tolerance, limits, derivative source and line search of
/// [`NonlinearCg`].
///
/// The defaults are the default
/// [`GradientTolerance`](scioxide_core::GradientTolerance), 1000 iterations, the
/// [`DerivativeStrategy::AnalyticOrCentral`](scioxide_differentiate::DerivativeStrategy::AnalyticOrCentral), no
/// time limit and
/// [`StrongWolfeLineSearch::for_nonlinear_cg`](crate::StrongWolfeLineSearch::for_nonlinear_cg).
/// The stopping test is described under [Termination](crate#termination).
#[derive(Debug, Clone)]
pub struct NonlinearCgOptions<
    S: RealScalar + NumericScalar<Real = S>,
    L = crate::line_search::StrongWolfeLineSearch<S>,
> {
    settings: LineSearchSettings<S, L>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for NonlinearCgOptions<S> {
    fn default() -> Self {
        Self {
            settings: LineSearchSettings::new(
                crate::line_search::StrongWolfeLineSearch::for_nonlinear_cg(),
            ),
        }
    }
}

line_search_settings!(NonlinearCgOptions);

impl<S: RealScalar + NumericScalar<Real = S>, L> NonlinearCgOptions<S, L> {
    /// Sets the line search.
    #[must_use]
    pub fn with_line_search<L2>(self, line_search: L2) -> NonlinearCgOptions<S, L2> {
        NonlinearCgOptions {
            settings: self.settings.with_line_search(line_search),
        }
    }
}

/// Describes the workspace that a [`NonlinearCg`] solve needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonlinearCgWorkspaceRequirements {
    dimension: usize,
}

impl NonlinearCgWorkspaceRequirements {
    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> Shape<FirstOrderStorage<S>>
    for NonlinearCgWorkspaceRequirements
{
    fn fits(&self, storage: &FirstOrderStorage<S>) -> bool {
        storage.fits(self.dimension, 0, false)
    }
    fn prepare(&self, storage: &mut FirstOrderStorage<S>) -> Result<(), SciOxideError> {
        storage.try_reserve(self.dimension, 0)
    }
}

crate::attempt::local_workspace!(
    /// Holds the storage and the record of the last call for [`NonlinearCg`].
    NonlinearCgWorkspace,
    FirstOrderStorage<S>,
    NonlinearCgWorkspaceRequirements
);

/// Minimizes a smooth function by the nonlinear conjugate gradient method of Polak and Ribière (PR+).
///
/// The search direction is `d = −g + β d₋` with
///
/// ```text
/// β = max(0, gᵀ(g − g₋) / ‖g₋‖²),
/// ```
///
/// where `g` is the current gradient and `g₋`, `d₋` are the previous gradient
/// and direction. The method restarts with `d = −g` every `n` iterations, when
/// `β` would be negative, and when `d` is not a descent direction;
/// [`CgRestartReason`](crate::CgRestartReason) gives the reason. The default
/// line search is [`StrongWolfeLineSearch::for_nonlinear_cg`](crate::StrongWolfeLineSearch::for_nonlinear_cg).
/// An iteration costs O(n) arithmetic and memory plus the line search.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideError};
/// use scioxide_optimize::{NonlinearCg, ObjectiveFunction, OptimizeResult};
///
/// struct Sphere;
///
/// impl ObjectiveFunction<f64> for Sphere {
///     fn dim(&self) -> usize {
///         1
///     }
///
///     fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
///         Ok(x[0] * x[0])
///     }
///
///     fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
///         out[0] = 2.0 * x[0];
///         Ok(true)
///     }
/// }
///
/// # fn main() -> Result<(), SciOxideError> {
/// let solver = NonlinearCg::<f64>::default();
/// let mut workspace = solver.workspace(1)?;
/// let mut output = OptimizeResult::<f64>::empty();
/// solver.minimize_into(
///     &Sphere,
///     &[2.0],
///     &mut output,
///     &mut workspace,
///     ExecutionContext::sequential(),
/// )?;
/// assert!(output.is_success());
/// // x² has its minimum at x = 0.
/// assert!(output.x()[0].abs() < 1e-6);
/// # Ok(())
/// # }
/// ```
///
/// # References
///
/// - J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed.,
///   Springer, 2006, §5.2.
/// - E. Polak and G. Ribière, "Note sur la convergence de méthodes de
///   directions conjuguées", *Rev. Française Informat. Recherche
///   Opérationnelle* 3(16), 35–43, 1969.
#[derive(Debug, Clone)]
pub struct NonlinearCg<
    S: RealScalar + NumericScalar<Real = S>,
    L = crate::line_search::StrongWolfeLineSearch<S>,
> {
    options: NonlinearCgOptions<S, L>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for NonlinearCg<S> {
    fn default() -> Self {
        Self::new(NonlinearCgOptions::default())
    }
}

impl<S, L> NonlinearCg<S, L>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: LineSearch<S>,
{
    /// Creates the optimizer with the given options.
    #[must_use]
    pub const fn new(options: NonlinearCgOptions<S, L>) -> Self {
        Self { options }
    }

    /// Returns the options.
    pub const fn options(&self) -> &NonlinearCgOptions<S, L> {
        &self.options
    }

    /// Returns the workspace requirements for `dimension` variables.
    ///
    /// # Errors
    ///
    /// This method does not fail.
    pub fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<NonlinearCgWorkspaceRequirements, SciOxideError> {
        Ok(NonlinearCgWorkspaceRequirements { dimension })
    }

    /// Allocates a workspace for `dimension` variables.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// reserved.
    pub fn workspace(&self, dimension: usize) -> Result<NonlinearCgWorkspace<S>, SciOxideError> {
        let mut workspace = NonlinearCgWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }

    /// Minimizes `f` from `x0` and writes the result to `output`.
    ///
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// parent [`SolveContext`](scioxide_core::SolveContext). When a limit,
    /// cancellation or deadline stops the solve after it has accepted a point,
    /// the call returns `Ok` with an unsuccessful termination. On error,
    /// `output` is unchanged and [`NonlinearCgWorkspace::last_attempt`] describes the
    /// failed call.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`UnsupportedConfiguration`) if `f` has bounds.
    /// - `DimensionMismatch` if `x0.len()` differs from `f.dim()`.
    /// - `NonFinite` if `x0`, or the objective or gradient at `x0`, is NaN
    ///   or infinite.
    /// - `CapabilityUnavailable` if a time limit is set and `scope` has no clock.
    /// - `ConvergenceFailure` or `ExecutionStopped` if a limit, cancellation
    ///   or deadline stops the solve before it accepts a point.
    /// - `SizeOverflow`, `ResourceLimit` or `AllocationFailure` if the
    ///   workspace cannot grow.
    ///
    /// Errors returned by `f` are returned unchanged.
    pub fn minimize_into<'scope, 'capability, F>(
        &self,
        f: &F,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut NonlinearCgWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError>
    where
        F: ObjectiveFunction<S> + ?Sized,
    {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        crate::validation::unbounded(f, x0)?;
        workspace.try_reserve(&self.workspace_requirements(x0.len())?)?;
        crate::attempt::preflight(&scope, self.options.time_limit())?;
        let entry = Entry::Call {
            identity: self.identity(x0.len()),
            budgets: iteration_budgets(self.options.max_iterations()),
        };
        crate::attempt::minimize(
            &mut workspace.shell,
            &mut workspace.storage,
            &mut workspace.staged,
            output,
            scope,
            entry,
            |context, storage, staged| {
                if let Some(limit) = self.options.time_limit() {
                    context.limit_time(limit)?;
                }
                let session =
                    crate::context::SolveSession::new(context, scioxide_core::argument!("scope"));
                let observed = session.observe(f);
                let outcome = self.minimize_kernel(&observed, x0, staged, storage, &session);
                crate::utils::complete_first_order(outcome, storage, staged, &session)
            },
        )
    }

    fn identity(&self, dimension: usize) -> scioxide_core::SolveIdentity {
        crate::context::local_identity(
            scioxide_core::algorithm_id!("optimize", "NonlinearCg", 1),
            dimension,
            &(core::any::type_name::<L>(), &self.options),
        )
    }

    fn minimize_kernel<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut FirstOrderStorage<S>,
        deadline: &crate::context::SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        validate_x0(f, x0)?;
        workspace.try_reserve(x0.len(), 0)?;
        output.try_reserve(x0.len())?;
        workspace.reset();
        workspace.x.copy_from_slice(x0);
        let derivative_strategy = self.options.derivative_strategy();

        let mut value = f.eval(&workspace.x)?;
        if !value.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Callback,
                "objective returned a non-finite value",
            ));
        }
        workspace.accepted_value = Some(value);
        let finite_difference = crate::derivatives::prepared(&mut workspace.finite_difference)?;
        get_gradient_into(
            f,
            &workspace.x,
            value,
            derivative_strategy,
            &mut workspace.gradient,
            finite_difference,
            deadline.execution()?,
        )?;
        workspace.accepted_gradient = true;
        let magnitude = workspace
            .gradient
            .iter()
            .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |max, &g| {
                max.max(NumericScalar::abs(g))
            });
        let small = <S as RealScalar>::SQRT_EPSILON;
        let direction_scale = if magnitude > <S as scioxide_core::AlgebraicScalar>::ONE / small
            || (magnitude > <S as scioxide_core::AlgebraicScalar>::ZERO && magnitude < small)
        {
            magnitude
        } else {
            <S as scioxide_core::AlgebraicScalar>::ONE
        };
        workspace
            .direction
            .iter_mut()
            .zip(&workspace.gradient)
            .for_each(|(d, &g)| *d = -g / direction_scale);
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;

        for iteration in 0..self.options.max_iterations().get() {
            if let Some(termination) = deadline.iteration()? {
                return finish_first_order(
                    workspace,
                    output,
                    value,
                    true,
                    iteration,
                    termination,
                    "execution limit reached",
                );
            }
            if self.options.gradient_tolerance().is_satisfied(
                norm2(&workspace.gradient)?,
                crate::utils::gradient_scale(value, &workspace.x),
            )? {
                return finish_first_order(
                    workspace,
                    output,
                    value,
                    true,
                    iteration,
                    OptimizeTermination::GradientSmall,
                    "gradient norm below tolerance",
                );
            }
            if !crate::quasi_newton::scaled_dot(&workspace.gradient, &workspace.direction)?
                .is_sign_negative()
            {
                workspace
                    .direction
                    .iter_mut()
                    .zip(&workspace.gradient)
                    .for_each(|(direction, &gradient)| *direction = -gradient / direction_scale);
            }
            let directional_derivative = crate::quasi_newton::descent_slope(
                &workspace.gradient,
                &mut workspace.direction,
                false,
            )?;
            let trial = self.options.line_search().search_into(
                f,
                &workspace.x,
                value,
                &workspace.direction,
                directional_derivative,
                derivative_strategy,
                &mut workspace.trial_x,
                &mut workspace.trial_gradient,
                &mut workspace.s,
                &mut workspace.y,
                finite_difference,
                crate::LineSearchScope::observed(deadline),
                None,
            )?;
            if let Some(termination) = line_search_termination(trial) {
                let termination = deadline.termination()?.unwrap_or(termination);
                return finish_first_order(
                    workspace,
                    output,
                    value,
                    true,
                    iteration,
                    termination,
                    "line search terminated without an accepted trial",
                );
            }
            if let Some(termination) = deadline.termination()? {
                return finish_first_order(
                    workspace,
                    output,
                    value,
                    true,
                    iteration,
                    termination,
                    "execution limit reached during line search",
                );
            }
            if !trial.gradient_available() {
                get_gradient_into(
                    f,
                    &workspace.trial_x,
                    trial.value(),
                    derivative_strategy,
                    &mut workspace.trial_gradient,
                    finite_difference,
                    deadline.execution()?,
                )?;
            } else {
                deadline.cache_hit()?;
            }
            let (mut beta, orthogonality, mut restart) = conjugate_coefficient(
                &workspace.gradient,
                &workspace.trial_gradient,
                (iteration + 1) % x0.len() == 0,
            )?;
            let candidate = workspace
                .s
                .iter_mut()
                .zip(workspace.trial_gradient.iter().zip(&workspace.direction))
                .try_for_each(|(d, (&g, &previous))| -> Result<(), SciOxideError> {
                    let normalized = g / direction_scale;
                    *d = if x0.len() < 16
                        || !normalized.is_finite()
                        || (g != zero
                            && NumericScalar::abs(normalized) < <S as RealScalar>::MIN_POSITIVE)
                    {
                        Scaled::from_value(beta)?
                            .mul(Scaled::from_value(previous)?)?
                            .sub(Scaled::from_value(g)?.div(Scaled::from_value(direction_scale)?)?)?
                            .round()?
                    } else {
                        crate::utils::ordinary_trial_component(-normalized, beta, previous)?
                    };
                    Ok(())
                });
            match candidate {
                Err(error)
                    if matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::NumericalInstability { .. }
                    ) =>
                {
                    restart = Some(crate::CgRestartReason::Unrepresentable)
                }
                Err(error) => return Err(error),
                Ok(()) => {
                    let slope = scaled_dot(&workspace.s, &workspace.trial_gradient)?;
                    if !slope.is_sign_negative()
                        && positive(scaled_dot(
                            &workspace.trial_gradient,
                            &workspace.trial_gradient,
                        )?)
                    {
                        restart = Some(crate::CgRestartReason::DescentLoss);
                    }
                }
            }
            if restart.is_some() {
                beta = zero;
                workspace
                    .s
                    .iter_mut()
                    .zip(&workspace.trial_gradient)
                    .for_each(|(d, &g)| *d = -g / direction_scale);
            }
            deadline.accept(scioxide_core::PartialResultTrust::Admissible)?;
            workspace.direction.copy_from_slice(&workspace.s);
            workspace.globalization.beta = Some(beta);
            workspace.globalization.gradient_orthogonality = orthogonality;
            if let Some(reason) = restart {
                workspace.globalization.restart(reason);
            }
            core::mem::swap(&mut workspace.x, &mut workspace.trial_x);
            core::mem::swap(&mut workspace.gradient, &mut workspace.trial_gradient);
            value = trial.value();
            workspace.accepted_value = Some(value);
            workspace.accepted_gradient = true;
            workspace.accepted_iterations = iteration + 1;
        }

        finish_first_order(
            workspace,
            output,
            value,
            true,
            self.options.max_iterations().get(),
            OptimizeTermination::IterationLimit,
            "maximum iterations reached",
        )
    }
}

impl<S, L> NonlinearCg<S, L>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: LineSearch<S>,
{
    /// Minimizes the objective with this method's settings and owned workspace.
    ///
    /// Returns the complete result, including unsuccessful termination. Call
    /// `require_success()` to require the stopping test to pass.
    ///
    /// # Errors
    ///
    /// Returns the input, callback, allocation and numerical errors of `minimize_into`.
    pub fn minimize<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x0: &[S],
    ) -> Result<OptimizeResult<S>, SciOxideError> {
        let mut workspace = self.workspace(x0.len())?;
        let mut output = OptimizeResult::empty();
        self.minimize_into(
            f,
            x0,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        Ok(output)
    }
}

impl<S, L, F> crate::traits::Optimizer<S, F> for NonlinearCg<S, L>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: LineSearch<S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    fn minimize(&self, f: &F, x0: &[S]) -> Result<OptimizeResult<S>, SciOxideError> {
        self.minimize(f, x0)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    struct Quadratic;

    impl ObjectiveFunction<f64> for Quadratic {
        fn dim(&self) -> usize {
            2
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x.iter().map(|value| value * value).sum())
        }

        fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output
                .iter_mut()
                .zip(x)
                .for_each(|(gradient, value)| *gradient = 2.0 * value);
            Ok(true)
        }
    }

    #[test]
    fn nonlinear_cg_prepared_path_converges() {
        let solver = NonlinearCg::<f64>::default();
        let mut workspace = solver.workspace(2).unwrap();
        let mut output = OptimizeResult::empty();
        output.try_reserve(2).unwrap();
        solver
            .minimize_into(
                &Quadratic,
                &[5.0, 3.0],
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(output.is_success());
        assert!(output.fun < 1e-10);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod globalization_tests {
    use super::*;

    fn coefficients<S: RealScalar>(scales: &[S]) {
        let two = S::TWO;
        for &a in scales {
            let (beta, orthogonality, restart) =
                conjugate_coefficient(&[a, S::ZERO], &[S::ZERO, two * a], false).unwrap();
            assert_eq!(beta, two * two);
            assert_eq!(orthogonality, Some(S::ZERO));
            assert_eq!(restart, None);
            let (beta, _, restart) = conjugate_coefficient(&[a], &[a / two], false).unwrap();
            assert_eq!(beta, S::ZERO);
            assert_eq!(restart, Some(crate::CgRestartReason::NegativeBeta));
            let (beta, _, restart) = conjugate_coefficient(&[a], &[two * a], true).unwrap();
            assert_eq!(beta, S::ZERO);
            assert_eq!(restart, Some(crate::CgRestartReason::Periodic));
        }
    }
    #[test]
    fn extreme_coefficients_f64() {
        coefficients(&[1e-200_f64, 1., 1e200]);
    }
    #[test]
    fn extreme_coefficients_f32() {
        coefficients(&[1e-30_f32, 1., 1e30]);
    }
    #[test]
    fn noisy_pr_plus_direction_is_restarted_and_not_reported_converged() {
        struct Noisy;
        impl ObjectiveFunction<f64> for Noisy {
            fn dim(&self) -> usize {
                2
            }
            fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
                Ok(x[0] * x[0])
            }
            fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
                out.copy_from_slice(if x[0] > 0.5 { &[1., 0.] } else { &[-1., 0.1] });
                Ok(true)
            }
        }
        let solver = NonlinearCg::new(
            NonlinearCgOptions::default()
                .with_line_search(crate::BacktrackingLineSearch::default())
                .with_max_iterations(scioxide_core::Budget::ONE),
        );
        let result = solver.minimize(&Noisy, &[1., 0.]).unwrap();
        assert_eq!(
            first_order_of(&result).last_restart(),
            Some(crate::CgRestartReason::DescentLoss)
        );
        assert!(!result.is_success());
        assert_eq!(first_order_of(&result).beta(), Some(0.));
    }
    #[test]
    fn slopes_are_rescaled_before_rounding() {
        for a in [1e-200_f64, 1e200] {
            let mut direction = [-a, -a];
            let slope = crate::quasi_newton::descent_slope(&[a, a], &mut direction, false).unwrap();
            assert!(slope.is_finite() && slope < 0.);
            assert_eq!(direction, [-0.5, -0.5]);
        }
    }

    #[allow(clippy::panic)]
    fn first_order_of<S: scioxide_core::RealScalar>(
        result: &crate::OptimizeResult<S>,
    ) -> crate::GlobalizationDiagnostics<S> {
        match result.method_diagnostics() {
            crate::OptimizeDiagnostics::FirstOrder(diagnostics) => diagnostics,
            other => panic!("unexpected method diagnostics {other:?}"),
        }
    }
}
