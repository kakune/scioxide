//! Newton's method with a line search and Hessian modification.

use scioxide_core::{
    AttemptKind, ExecutionContext, NumericScalar, RealScalar, SciOxideError, SolveScope,
};
use scioxide_linalg::LdlDecomposition;

use super::settings::{LineSearchSettings, iteration_budgets, line_search_settings};
use crate::attempt::{Entry, Shape};
use crate::result::{OptimizeResult, OptimizeTermination};
use crate::traits::{LineSearch, ObjectiveFunction};
use crate::unconstrained::{SecondOrderShape, SecondOrderStorage};
use crate::utils::{
    dot, finish_second_order, get_gradient_into, get_hessian_into, line_search_termination, norm2,
    validate_x0,
};

/// Sets the stopping tolerance, limits, derivative source and line search of [`NewtonOptimizer`].
///
/// The defaults are the default
/// [`GradientTolerance`](scioxide_core::GradientTolerance), 1000 iterations, the
/// [`DerivativeStrategy::AnalyticOrCentral`](scioxide_differentiate::DerivativeStrategy::AnalyticOrCentral), no
/// time limit and [`BacktrackingLineSearch`](crate::BacktrackingLineSearch).
/// The stopping test is described under [Termination](crate#termination).
#[derive(Debug, Clone)]
pub struct NewtonOptions<
    S: RealScalar + NumericScalar<Real = S>,
    L = crate::line_search::BacktrackingLineSearch<S>,
> {
    settings: LineSearchSettings<S, L>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for NewtonOptions<S> {
    fn default() -> Self {
        Self {
            settings: LineSearchSettings::new(crate::line_search::BacktrackingLineSearch::default()),
        }
    }
}

line_search_settings!(NewtonOptions);

impl<S: RealScalar + NumericScalar<Real = S>, L> NewtonOptions<S, L> {
    /// Sets the line search.
    #[must_use]
    pub fn with_line_search<L2>(self, line_search: L2) -> NewtonOptions<S, L2> {
        NewtonOptions {
            settings: self.settings.with_line_search(line_search),
        }
    }
}

fn newton_shape(dimension: usize) -> SecondOrderShape {
    SecondOrderShape {
        dimension,
        dense: true,
        regularize: true,
        cg: false,
        second_order: false,
    }
}

/// Describes the workspace that a [`NewtonOptimizer`] solve needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NewtonOptimizerWorkspaceRequirements {
    dimension: usize,
}

impl NewtonOptimizerWorkspaceRequirements {
    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> Shape<SecondOrderStorage<S>>
    for NewtonOptimizerWorkspaceRequirements
{
    fn fits(&self, storage: &SecondOrderStorage<S>) -> bool {
        storage.fits(newton_shape(self.dimension))
    }
    fn prepare(&self, storage: &mut SecondOrderStorage<S>) -> Result<(), SciOxideError> {
        storage.prepare(newton_shape(self.dimension))
    }
}

crate::attempt::local_workspace!(
    /// Holds the storage and the record of the last call for [`NewtonOptimizer`].
    NewtonOptimizerWorkspace,
    SecondOrderStorage<S>,
    NewtonOptimizerWorkspaceRequirements
);

fn solve_direction<S: RealScalar + NumericScalar<Real = S>>(
    workspace: &mut SecondOrderStorage<S>,
    session: &crate::context::SolveSession<'_, '_, '_>,
) -> Result<(), SciOxideError> {
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let n = workspace.dimension;
    let scale = workspace
        .hessian
        .as_slice()
        .iter()
        .fold(zero, |a, &v| a.max(NumericScalar::abs(v)));
    let scale = if scale == zero { one } else { scale };
    for (rhs, &g) in workspace
        .factor_rhs
        .as_mut_slice()
        .iter_mut()
        .zip(&workspace.gradient)
    {
        *rhs = scioxide_core::checked_float_quotient(-g, scale)?;
    }
    let mut shift = zero;
    for attempt in 0..=16 {
        for (candidate, &value) in workspace
            .candidate_hessian
            .as_mut_slice()
            .iter_mut()
            .zip(workspace.hessian.as_slice())
        {
            *candidate = scioxide_core::checked_float_quotient(value, scale)?;
        }
        workspace.candidate_hessian.project_hermitian_in_place()?;
        for i in 0..n {
            workspace.candidate_hessian.as_mut_slice()[i * n + i] = scioxide_core::checked_sum(&[
                workspace.candidate_hessian.as_slice()[i * n + i],
                shift,
            ])?;
        }
        session.invoke(scioxide_core::EvaluationRole::Factorization, || {
            LdlDecomposition::factor_into(
                workspace.candidate_hessian.view(),
                scioxide_linalg::HermitianInput::Exact,
                &scioxide_linalg::LinalgExecutionOptions::new(),
                &mut workspace.ldl_factor,
                &mut workspace.ldl_workspace,
                session.execution()?,
            )
        })?;
        let inertia = workspace
            .ldl_factor
            .inertia(scioxide_linalg::RankPolicy::Exact)?;
        if attempt == 0 {
            workspace.newton.inertia = Some(inertia);
        }
        workspace.newton.normalized_shift = shift;
        if inertia.negative() == 0 && inertia.zero() == 0 {
            session.invoke(scioxide_core::EvaluationRole::LinearSolve, || {
                workspace
                    .ldl_factor
                    .solve_into(
                        workspace.factor_rhs.view(),
                        &scioxide_linalg::DirectSolveOptions::new(
                            scioxide_linalg::RankPolicy::Exact,
                        ),
                        workspace.factor_solution.view_mut(),
                        &mut workspace.ldl_workspace,
                        session.execution()?,
                    )
                    .and_then(|report| report.termination().require_accepted())
            })?;
            workspace
                .direction
                .copy_from_slice(workspace.factor_solution.as_slice());
            workspace.newton.linear_residual =
                Some(workspace.candidate_hessian.view().verify_solution(
                    workspace.factor_rhs.as_slice(),
                    &workspace.direction,
                    scioxide_core::Tolerance::new(zero, S::SQRT_EPSILON)?,
                )?);
            if dot(&workspace.gradient, &workspace.direction)? < zero {
                return Ok(());
            }
        }
        shift = if attempt == 0 {
            S::SQRT_EPSILON
        } else {
            scioxide_core::checked_float_product(shift, S::try_from_usize_exact(10)?)?
        };
    }
    Err(SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Breakdown,
        "positive-definite modification did not produce a descent direction",
    ))
}

/// Minimizes a twice-differentiable function by Newton's method with a line search.
///
/// Each iteration solves `(H/η + τI) d = −g/η` for the search direction `d`,
/// where `H` is the Hessian, `η` its largest absolute entry and `τ ≥ 0` a
/// diagonal shift. The method factors the matrix by symmetric indefinite
/// (LDLᵀ) factorization, first with `τ = 0`. If the matrix is not positive
/// definite, or `d` is not a descent direction, it retries with `τ = √ε` and
/// then ten times larger, up to 16 shifts. The Hessian comes from
/// [`ObjectiveFunction::hessian_into`](crate::ObjectiveFunction::hessian_into)
/// or from finite differences, as the derivative strategy selects. An
/// iteration costs O(n³) for the factorization; [`NewtonDiagnostics`](crate::NewtonDiagnostics)
/// reports the inertia and the shift.
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideError};
/// use scioxide_optimize::{NewtonOptimizer, ObjectiveFunction, OptimizeResult};
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
/// let solver = NewtonOptimizer::<f64>::default();
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
/// J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
/// 2006, §3.4.
#[derive(Debug, Clone)]
pub struct NewtonOptimizer<
    S: RealScalar + NumericScalar<Real = S>,
    L = crate::line_search::BacktrackingLineSearch<S>,
> {
    options: NewtonOptions<S, L>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for NewtonOptimizer<S> {
    fn default() -> Self {
        Self::new(NewtonOptions::default())
    }
}

impl<S, L> NewtonOptimizer<S, L>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: LineSearch<S>,
{
    /// Creates the optimizer with the given options.
    #[must_use]
    pub const fn new(options: NewtonOptions<S, L>) -> Self {
        Self { options }
    }

    /// Returns the options.
    pub const fn options(&self) -> &NewtonOptions<S, L> {
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
    ) -> Result<NewtonOptimizerWorkspaceRequirements, SciOxideError> {
        Ok(NewtonOptimizerWorkspaceRequirements { dimension })
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
    ) -> Result<NewtonOptimizerWorkspace<S>, SciOxideError> {
        let mut workspace = NewtonOptimizerWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }

    /// Minimizes `f` from `x0` and writes the result to `output`.
    ///
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// parent [`SolveContext`](scioxide_core::SolveContext). When a limit,
    /// cancellation or deadline stops the solve after it has accepted a point,
    /// the call returns `Ok` with an unsuccessful termination. On error,
    /// `output` is unchanged and [`NewtonOptimizerWorkspace::last_attempt`] describes the
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
    /// - `NumericalInstability` (`Breakdown`) if 16 diagonal shifts do not
    ///   give a descent direction.
    /// - `SizeOverflow`, `ResourceLimit` or `AllocationFailure` if the
    ///   workspace cannot grow.
    ///
    /// Errors returned by `f` are returned unchanged.
    pub fn minimize_into<'scope, 'capability, F>(
        &self,
        f: &F,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut NewtonOptimizerWorkspace<S>,
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
                crate::utils::complete_second_order(outcome, storage, staged, &session)
            },
        )
    }

    fn identity(&self, dimension: usize) -> scioxide_core::SolveIdentity {
        crate::context::local_identity(
            scioxide_core::algorithm_id!("optimize", "Newton", 1),
            dimension,
            &(core::any::type_name::<L>(), &self.options),
        )
    }

    fn minimize_kernel<F: ObjectiveFunction<S> + ?Sized>(
        &self,
        f: &F,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut SecondOrderStorage<S>,
        deadline: &crate::context::SolveSession<'_, '_, '_>,
    ) -> Result<(), SciOxideError> {
        validate_x0(f, x0)?;
        workspace.try_reserve(x0.len(), true)?;
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
        {
            let finite_difference =
                crate::derivatives::prepared(&mut workspace.gradient_difference)?;
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

        for iteration in 0..self.options.max_iterations().get() {
            if let Some(termination) = deadline.iteration()? {
                return finish_second_order(
                    workspace,
                    output,
                    value,
                    iteration,
                    termination,
                    "execution limit reached",
                );
            }
            if self.options.gradient_tolerance().is_satisfied(
                norm2(&workspace.gradient)?,
                crate::utils::gradient_scale(value, &workspace.x),
            )? {
                return finish_second_order(
                    workspace,
                    output,
                    value,
                    iteration,
                    OptimizeTermination::GradientSmall,
                    "gradient norm below tolerance",
                );
            }
            {
                let finite_difference =
                    crate::derivatives::prepared(&mut workspace.hessian_difference)?;
                get_hessian_into(
                    f,
                    &workspace.x,
                    value,
                    derivative_strategy,
                    &mut workspace.hessian,
                    finite_difference,
                    deadline.execution()?,
                )?;
            }
            solve_direction(workspace, deadline)?;
            let directional_derivative = dot(&workspace.gradient, &workspace.direction)?;
            let trial = {
                let finite_difference =
                    crate::derivatives::prepared(&mut workspace.gradient_difference)?;
                self.options.line_search().search_into(
                    f,
                    &workspace.x,
                    value,
                    &workspace.direction,
                    directional_derivative,
                    derivative_strategy,
                    &mut workspace.trial_x,
                    &mut workspace.trial_gradient,
                    &mut workspace.rhs,
                    &mut workspace.model_product,
                    finite_difference,
                    crate::LineSearchScope::observed(deadline),
                    None,
                )?
            };
            if let Some(termination) = line_search_termination(trial) {
                let termination = deadline.termination()?.unwrap_or(termination);
                return finish_second_order(
                    workspace,
                    output,
                    value,
                    iteration,
                    termination,
                    "line search terminated without an accepted trial",
                );
            }
            if !trial.gradient_available() {
                let finite_difference =
                    crate::derivatives::prepared(&mut workspace.gradient_difference)?;
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
            deadline.accept(scioxide_core::PartialResultTrust::Admissible)?;
            core::mem::swap(&mut workspace.x, &mut workspace.trial_x);
            core::mem::swap(&mut workspace.gradient, &mut workspace.trial_gradient);
            value = trial.value();
            workspace.accepted_value = Some(value);
            workspace.accepted_gradient = true;
            workspace.accepted_iterations = iteration + 1;
        }
        finish_second_order(
            workspace,
            output,
            value,
            self.options.max_iterations().get(),
            OptimizeTermination::IterationLimit,
            "maximum iterations reached",
        )
    }
}

impl<S, L> NewtonOptimizer<S, L>
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

impl<S, L, F> crate::traits::Optimizer<S, F> for NewtonOptimizer<S, L>
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
    use scioxide_linalg::Matrix;

    struct Quadratic;

    impl ObjectiveFunction<f64> for Quadratic {
        fn dim(&self) -> usize {
            2
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x[0] * x[0] + x[1] * x[1])
        }

        fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
            Ok(true)
        }

        fn hessian_into(
            &self,
            _x: &[f64],
            output: &mut Matrix<f64>,
        ) -> Result<bool, SciOxideError> {
            output.as_mut_slice().copy_from_slice(&[2.0, 0.0, 0.0, 2.0]);
            Ok(true)
        }
    }

    #[test]
    fn small_negative_curvature_uses_a_small_checked_shift() {
        let solver = NewtonOptimizer::<f64>::default();
        let mut prepared = solver.workspace(2).unwrap();
        let workspace = &mut prepared.storage;
        workspace
            .hessian
            .as_mut_slice()
            .copy_from_slice(&[1.0, 2.0, 2.0, 3.9999]);
        workspace.gradient.copy_from_slice(&[1.0, -1.0]);
        crate::context::sequential(solver.identity(2), scioxide_core::Budget::ONE, |context| {
            solve_direction(
                workspace,
                &crate::context::SolveSession::new(context, scioxide_core::argument!("context")),
            )
        })
        .unwrap();
        assert_eq!(workspace.newton.inertia().unwrap().negative(), 1);
        assert!(
            workspace.newton.normalized_shift() > 0.0 && workspace.newton.normalized_shift() < 1e-3
        );
        assert!(dot(&workspace.gradient, &workspace.direction).unwrap() < 0.0);
        assert!(workspace.newton.linear_residual().unwrap().is_finite());
    }

    #[test]
    fn prepared_newton_reuses_factorization_storage() {
        let solver = NewtonOptimizer::<f64>::default();
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
        let counts = workspace.last_attempt().unwrap().counts().own();
        assert_eq!(
            counts.provider(scioxide_core::EvaluationRole::Factorization),
            1
        );
        assert_eq!(
            counts.provider(scioxide_core::EvaluationRole::LinearSolve),
            1
        );
    }
}
