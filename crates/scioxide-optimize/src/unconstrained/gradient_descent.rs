//! Steepest descent with a line search.

use scioxide_core::{
    AttemptKind, ExecutionContext, NumericScalar, RealScalar, SciOxideError, SolveScope,
};

use super::settings::{LineSearchSettings, iteration_budgets, line_search_settings};
use crate::attempt::{Entry, Shape};
use crate::result::{OptimizeResult, OptimizeTermination};
use crate::traits::{LineSearch, ObjectiveFunction};
use crate::unconstrained::FirstOrderStorage;
use crate::utils::{
    finish_first_order, get_gradient_into, line_search_termination, norm2, search_dot as dot,
    validate_x0,
};

// Positive least-squares inverse secant scale: minimize ||s - alpha*y||².
// The estimate is only a trial hint; Armijo remains the acceptance authority.
fn inverse_secant_step<S: RealScalar>(
    x: &[S],
    trial_x: &[S],
    gradient: &[S],
    trial_gradient: &[S],
    displacement: &mut [S],
    difference: &mut [S],
) -> Result<Option<S>, SciOxideError> {
    for ((s, y), ((&x, &next), (&g, &next_g))) in displacement.iter_mut().zip(&mut *difference).zip(
        x.iter()
            .zip(trial_x)
            .zip(gradient.iter().zip(trial_gradient)),
    ) {
        *s = next - x;
        *y = next_g - g;
        if !s.is_finite() || !y.is_finite() {
            return Ok(None);
        }
    }
    let curvature = crate::quasi_newton::scaled_dot(displacement, difference)?;
    let squared = crate::quasi_newton::scaled_dot(difference, difference)?;
    if !crate::quasi_newton::positive(curvature) || squared.is_zero() {
        return Ok(None);
    }
    match curvature.div(squared)?.round() {
        Ok(step) => Ok((step > S::ZERO && step.is_finite()).then_some(step)),
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            ) =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// Sets the stopping tolerance, limits, derivative source and line search of [`GradientDescent`].
///
/// The defaults are the default
/// [`GradientTolerance`](scioxide_core::GradientTolerance), 1000 iterations, the
/// [`DerivativeStrategy::AnalyticOrCentral`](scioxide_differentiate::DerivativeStrategy::AnalyticOrCentral), no
/// time limit and [`BacktrackingLineSearch`](crate::BacktrackingLineSearch).
/// The stopping test is described under [Termination](crate#termination).
#[derive(Debug, Clone)]
pub struct GradientDescentOptions<
    S: RealScalar + NumericScalar<Real = S>,
    L = crate::line_search::BacktrackingLineSearch<S>,
> {
    settings: LineSearchSettings<S, L>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for GradientDescentOptions<S> {
    fn default() -> Self {
        Self {
            settings: LineSearchSettings::new(crate::line_search::BacktrackingLineSearch::default()),
        }
    }
}

line_search_settings!(GradientDescentOptions);

impl<S: RealScalar + NumericScalar<Real = S>, L> GradientDescentOptions<S, L> {
    /// Sets the line search.
    #[must_use]
    pub fn with_line_search<L2>(self, line_search: L2) -> GradientDescentOptions<S, L2> {
        GradientDescentOptions {
            settings: self.settings.with_line_search(line_search),
        }
    }
}

/// Describes the workspace that a [`GradientDescent`] solve needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradientDescentWorkspaceRequirements {
    dimension: usize,
}

impl GradientDescentWorkspaceRequirements {
    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> Shape<FirstOrderStorage<S>>
    for GradientDescentWorkspaceRequirements
{
    fn fits(&self, storage: &FirstOrderStorage<S>) -> bool {
        storage.fits(self.dimension, 0, false)
    }
    fn prepare(&self, storage: &mut FirstOrderStorage<S>) -> Result<(), SciOxideError> {
        storage.try_reserve(self.dimension, 0)
    }
}

crate::attempt::local_workspace!(
    /// Holds the storage and the record of the last call for [`GradientDescent`].
    GradientDescentWorkspace,
    FirstOrderStorage<S>,
    GradientDescentWorkspaceRequirements
);

/// Minimizes a smooth function by steepest descent with a line search.
///
/// Each iteration searches along `−∇f(x)`. After the first iteration the
/// solver passes `sᵀy / yᵀy` to the line search as the initial trial step,
/// where `s` is the last step and `y` the change in the gradient (the step
/// length of Barzilai and Borwein); the line search still decides acceptance.
/// An iteration costs O(n) arithmetic plus the line search. Steepest descent
/// converges at best linearly; [`Lbfgs`](crate::Lbfgs) is usually much faster.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideError};
/// use scioxide_optimize::{GradientDescent, ObjectiveFunction, OptimizeResult};
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
/// let solver = GradientDescent::<f64>::default();
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
///   Springer, 2006, §3.3.
/// - J. Barzilai and J. M. Borwein, "Two-point step size gradient methods",
///   *IMA J. Numer. Anal.* 8(1), 141–148, 1988.
#[derive(Debug, Clone)]
pub struct GradientDescent<
    S: RealScalar + NumericScalar<Real = S>,
    L = crate::line_search::BacktrackingLineSearch<S>,
> {
    options: GradientDescentOptions<S, L>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for GradientDescent<S> {
    fn default() -> Self {
        Self::new(GradientDescentOptions::default())
    }
}

impl<S, L> GradientDescent<S, L>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: LineSearch<S>,
{
    /// Creates the optimizer with the given options.
    #[must_use]
    pub const fn new(options: GradientDescentOptions<S, L>) -> Self {
        Self { options }
    }

    /// Returns the options.
    pub const fn options(&self) -> &GradientDescentOptions<S, L> {
        &self.options
    }

    /// Returns the workspace requirements for `dimension` variables.
    ///
    /// # Errors
    ///
    /// This method does not fail.
    pub const fn workspace_requirements(
        &self,
        dimension: usize,
    ) -> Result<GradientDescentWorkspaceRequirements, SciOxideError> {
        Ok(GradientDescentWorkspaceRequirements { dimension })
    }

    /// Allocates a workspace for `dimension` variables.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// reserved.
    pub fn workspace(
        &self,
        dimension: usize,
    ) -> Result<GradientDescentWorkspace<S>, SciOxideError> {
        let mut workspace = GradientDescentWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }

    /// Minimizes `f` from `x0` and writes the result to `output`.
    ///
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// parent [`SolveContext`](scioxide_core::SolveContext). When a limit,
    /// cancellation or deadline stops the solve after it has accepted a point,
    /// the call returns `Ok` with an unsuccessful termination. On error,
    /// `output` is unchanged and [`GradientDescentWorkspace::last_attempt`] describes the
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
        workspace: &mut GradientDescentWorkspace<S>,
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
            scioxide_core::algorithm_id!("optimize", "GradientDescent", 1),
            dimension,
            &(core::any::type_name::<L>(), &self.options),
        )
    }

    fn minimize_kernel<F>(
        &self,
        f: &F,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut FirstOrderStorage<S>,
        deadline: &crate::context::SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError>
    where
        F: ObjectiveFunction<S> + ?Sized,
    {
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
        let mut has_gradient = false;

        let mut initial_step = None;
        for iteration in 0..self.options.max_iterations().get() {
            if let Some(termination) = deadline.iteration()? {
                return finish_first_order(
                    workspace,
                    output,
                    value,
                    has_gradient,
                    iteration,
                    termination,
                    "execution limit reached",
                );
            }
            if !has_gradient {
                let finite_difference =
                    crate::derivatives::prepared(&mut workspace.finite_difference)?;
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
            workspace
                .direction
                .iter_mut()
                .zip(&workspace.gradient)
                .for_each(|(direction, &gradient)| *direction = -gradient);
            let directional_derivative = dot(&workspace.gradient, &workspace.direction)?;
            let finite_difference = crate::derivatives::prepared(&mut workspace.finite_difference)?;
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
                initial_step,
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
            let conservative = trial.step() * <S as RealScalar>::HALF;
            let fallback = if trial.rejected_trials() > 0
                && conservative > <S as scioxide_core::AlgebraicScalar>::ZERO
            {
                conservative
            } else {
                trial.step()
            };
            initial_step = Some(
                inverse_secant_step(
                    &workspace.x,
                    &workspace.trial_x,
                    &workspace.gradient,
                    &workspace.trial_gradient,
                    &mut workspace.s,
                    &mut workspace.y,
                )?
                .unwrap_or(fallback),
            );
            deadline.accept(scioxide_core::PartialResultTrust::Admissible)?;
            core::mem::swap(&mut workspace.x, &mut workspace.trial_x);
            core::mem::swap(&mut workspace.gradient, &mut workspace.trial_gradient);
            value = trial.value();
            workspace.accepted_value = Some(value);
            workspace.accepted_gradient = true;
            workspace.accepted_iterations = iteration + 1;
            has_gradient = true;
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

impl<S, L> GradientDescent<S, L>
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

impl<S, L, F> crate::traits::Optimizer<S, F> for GradientDescent<S, L>
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
    fn reusable_gradient_descent_converges_and_reuses_storage() {
        let solver = GradientDescent::<f64>::default();
        let mut workspace = solver.workspace(2).unwrap();
        let mut result = OptimizeResult::empty();
        result.try_reserve(2).unwrap();
        solver
            .minimize_into(
                &Quadratic,
                &[5.0, 3.0],
                &mut result,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(result.is_success());
        assert!(result.fun < 1e-10);
        let capacities = (result.x.capacity(), workspace.storage.x.capacity());
        solver
            .minimize_into(
                &Quadratic,
                &[2.0, -4.0],
                &mut result,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(
            capacities,
            (result.x.capacity(), workspace.storage.x.capacity())
        );
        assert_eq!(
            workspace
                .last_attempt()
                .unwrap()
                .counts()
                .own()
                .provider(scioxide_core::EvaluationRole::Objective),
            result.fun_evals
        );
    }
    #[test]
    fn accepted_backtracking_length_is_reused() {
        let objective = crate::ObjectiveFn::new(1, |x: &[f64]| Ok(1.5 * x[0] * x[0]))
            .with_gradient(|x: &[f64], output| {
                output.copy_from_slice(&[3.0 * x[0]]);
                Ok(())
            });
        let result = GradientDescent::<f64>::default()
            .minimize(&objective, &[1.0])
            .unwrap();
        assert!(result.is_success());
        assert!(result.fun() < 1e-12);
        assert_eq!(result.counts().own().rejected(), 1);
        assert_eq!(result.fun_evals(), result.iterations() + 2);
    }
    #[test]
    fn absent_positive_secant_curvature_preserves_descent_and_iteration_stops() {
        for concave in [false, true] {
            let objective = crate::ObjectiveFn::new(1, move |x: &[f64]| {
                Ok(if concave { -x[0] * x[0] } else { x[0] })
            })
            .with_gradient(move |x: &[f64], output| {
                output.copy_from_slice(&[if concave { -2.0 * x[0] } else { 1.0 }]);
                Ok(())
            });
            let solver = GradientDescent::new(
                GradientDescentOptions::default()
                    .with_max_iterations(scioxide_core::Budget::ONE.saturating_add(2)),
            );
            let result = solver.minimize(&objective, &[1.0]).unwrap();
            assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
            assert_eq!(result.iterations(), 3);
            assert_eq!(result.fun_evals(), 4);
            assert!(result.fun() < if concave { -1.0 } else { 1.0 });
        }
    }
}
