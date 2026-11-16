//! Trust-region Newton method with dogleg or Steihaug-CG subproblems.

use core::time::Duration;

use scioxide_core::Scaled;
use scioxide_core::{
    AttemptKind, Budget, ExecutionContext, ExecutionPolicy, GradientTolerance, NumericScalar,
    RealScalar, SciOxideError, SolveScope, checked_difference, checked_difference_of_products,
    checked_quadratic_roots, checked_sum_of_products,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::{CholeskyDecomposition, CholeskyPivoting};

use super::settings::iteration_budgets;
use crate::attempt::{Entry, Shape};
use crate::result::{OptimizeResult, OptimizeTermination};
use crate::traits::ObjectiveFunction;
use crate::unconstrained::{SecondOrderShape, SecondOrderStorage};
use crate::utils::{
    dot, finish_second_order, get_gradient_into, get_hessian_into, norm2, validate_x0,
};

const MAX_RADIUS_FACTOR: f64 = 1e8;
const ACCEPT_THRESHOLD: f64 = 0.15;

/// Selects the method that solves the trust-region subproblem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrustRegionSubproblem {
    /// Dogleg step from a Cholesky factorization of the dense model Hessian.
    /// A singular or indefinite model gives the Cauchy point.
    #[default]
    Dogleg,
    /// Steihaug's truncated conjugate gradient method. It uses the
    /// objective's Hessian-vector products when
    /// [`ObjectiveFunction::has_hessian_vector_product`](crate::ObjectiveFunction::has_hessian_vector_product)
    /// is `true`, and a dense Hessian otherwise.
    SteihaugCg {
        /// Limit on the Hessian-vector products of each subproblem solve.
        max_inner_iterations: Budget,
        /// Checks the curvature of the dense Hessian before declaring
        /// convergence. At a point with a small gradient and negative
        /// curvature, the solver steps to the trust-region boundary along a
        /// direction of negative curvature instead. The solver then forms the
        /// dense Hessian, analytic or by finite differences as the derivative
        /// strategy selects, and does not use Hessian-vector products.
        second_order_stopping: bool,
    },
}

/// Sets the stopping tolerance, limits, derivative source and subproblem of
/// [`TrustRegionOptimizer`].
///
/// The defaults are the default [`GradientTolerance`], 1000 iterations, the
/// default [`DerivativeStrategy`], no time limit and
/// [`TrustRegionSubproblem::Dogleg`]. The stopping test is described under
/// [Termination](crate#termination).
#[derive(Debug, Clone, Copy)]
pub struct TrustRegionOptions<S: RealScalar> {
    gradient_tolerance: GradientTolerance<S>,
    max_iterations: Budget,
    derivative_strategy: DerivativeStrategy,
    time_limit: Option<Duration>,
    subproblem: TrustRegionSubproblem,
}

impl<S: RealScalar> Default for TrustRegionOptions<S> {
    fn default() -> Self {
        Self {
            gradient_tolerance: GradientTolerance::default(),
            max_iterations: Budget::ONE.saturating_add(999),
            derivative_strategy: DerivativeStrategy::default(),
            time_limit: None,
            subproblem: TrustRegionSubproblem::Dogleg,
        }
    }
}

impl<S: RealScalar> TrustRegionOptions<S> {
    /// Sets the gradient tolerance.
    #[must_use]
    pub const fn with_gradient_tolerance(
        mut self,
        gradient_tolerance: GradientTolerance<S>,
    ) -> Self {
        self.gradient_tolerance = gradient_tolerance;
        self
    }

    /// Sets the iteration limit.
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iterations: Budget) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the order in which derivative sources are tried.
    #[must_use]
    pub const fn with_derivative_strategy(mut self, strategy: DerivativeStrategy) -> Self {
        self.derivative_strategy = strategy;
        self
    }

    /// Sets a wall-clock time limit. The execution context must provide a clock.
    #[must_use]
    pub const fn with_time_limit(mut self, time_limit: Duration) -> Self {
        self.time_limit = Some(time_limit);
        self
    }

    /// Sets the subproblem method.
    #[must_use]
    pub const fn with_subproblem(mut self, subproblem: TrustRegionSubproblem) -> Self {
        self.subproblem = subproblem;
        self
    }

    /// Returns the gradient tolerance.
    pub const fn gradient_tolerance(&self) -> GradientTolerance<S> {
        self.gradient_tolerance
    }

    /// Returns the iteration limit.
    pub const fn max_iterations(&self) -> Budget {
        self.max_iterations
    }

    /// Returns the order in which derivative sources are tried.
    pub const fn derivative_strategy(&self) -> DerivativeStrategy {
        self.derivative_strategy
    }

    /// Returns the time limit, if set.
    pub const fn time_limit(&self) -> Option<Duration> {
        self.time_limit
    }

    /// Returns the subproblem method.
    pub const fn subproblem(&self) -> TrustRegionSubproblem {
        self.subproblem
    }
}

/// Describes the problem a [`TrustRegionOptimizer`] workspace is for.
///
/// With [`TrustRegionSubproblem::SteihaugCg`] without second-order stopping
/// and an objective that provides Hessian-vector products, the workspace
/// stores no dense Hessian. The other configurations store one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrustRegionWorkspaceRequest {
    dimension: usize,
    matrix_free: bool,
}

impl TrustRegionWorkspaceRequest {
    /// Requests storage for `dimension` variables with a dense Hessian.
    pub const fn dense(dimension: usize) -> Self {
        Self {
            dimension,
            matrix_free: false,
        }
    }

    /// Requests storage for `objective`, without a dense Hessian when it provides
    /// Hessian-vector products.
    pub fn of<S, F>(objective: &F) -> Self
    where
        S: RealScalar + NumericScalar<Real = S>,
        F: ObjectiveFunction<S> + ?Sized,
    {
        Self {
            dimension: objective.dim(),
            matrix_free: objective.has_hessian_vector_product(),
        }
    }
}

/// Describes the workspace that a [`TrustRegionOptimizer`] solve needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrustRegionOptimizerWorkspaceRequirements {
    shape: SecondOrderShape,
}

impl TrustRegionOptimizerWorkspaceRequirements {
    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.shape.dimension
    }

    /// Returns `true` if the workspace stores a dense Hessian.
    pub const fn dense(self) -> bool {
        self.shape.dense
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> Shape<SecondOrderStorage<S>>
    for TrustRegionOptimizerWorkspaceRequirements
{
    fn fits(&self, storage: &SecondOrderStorage<S>) -> bool {
        storage.fits(self.shape)
    }
    fn prepare(&self, storage: &mut SecondOrderStorage<S>) -> Result<(), SciOxideError> {
        storage.prepare(self.shape)
    }
}

crate::attempt::local_workspace!(
    /// Holds the storage and the record of the last call for [`TrustRegionOptimizer`].
    TrustRegionOptimizerWorkspace,
    SecondOrderStorage<S>,
    TrustRegionOptimizerWorkspaceRequirements
);

/// Minimizes a twice-differentiable function by a trust-region Newton method.
///
/// At a point `x` with gradient `g` and Hessian `H`, each iteration
/// approximately solves the subproblem
///
/// ```text
/// minimize  m(p) = gᵀp + ½ pᵀB p   subject to  ‖p‖ ≤ Δ,
/// ```
///
/// where `B = (H + Hᵀ)/2` and `Δ` is the trust radius. With the default
/// [`TrustRegionSubproblem::Dogleg`], a positive definite `B` gives the dogleg
/// step between the Cauchy point and the Newton step from a Cholesky
/// factorization; a singular or indefinite `B` gives the Cauchy point.
/// [`TrustRegionSubproblem::SteihaugCg`] solves the subproblem by truncated
/// conjugate gradients and can use Hessian-vector products without a dense
/// Hessian.
///
/// The step is accepted when `f` decreases and the ratio `ρ` of actual to
/// predicted reduction exceeds 0.15. The radius starts at `‖g(x₀)‖` (1 if it
/// is zero), shrinks to `Δ/4` when `ρ < 0.25` and doubles, up to `10⁸`, when
/// `ρ > 0.75` and the step reaches the boundary. A radius below the machine
/// epsilon ends the solve with
/// [`OptimizeTermination::Stagnation`](crate::OptimizeTermination::Stagnation).
/// With the dogleg subproblem an iteration costs O(n³) for the factorization
/// and the workspace holds O(n²) values; a rejected step reuses the
/// factorization.
///
/// # Examples
///
/// ```
/// use scioxide_optimize::{ObjectiveFn, TrustRegionOptimizer};
///
/// // (x − 2)² has its minimum at x = 2; derivatives come from finite differences.
/// let objective = ObjectiveFn::new(1, |x: &[f64]| Ok((x[0] - 2.0).powi(2)));
/// let result = TrustRegionOptimizer::<f64>::default().minimize(&objective, &[0.0])?;
/// assert!(result.is_success());
/// assert!((result.x()[0] - 2.0).abs() < 1e-5);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// - J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed.,
///   Springer, 2006, ch. 4 and §7.1.
/// - T. Steihaug, "The conjugate gradient method and trust regions in large
///   scale optimization", *SIAM J. Numer. Anal.* 20(3), 626–637, 1983.
#[derive(Debug, Clone)]
pub struct TrustRegionOptimizer<S: RealScalar> {
    options: TrustRegionOptions<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for TrustRegionOptimizer<S> {
    fn default() -> Self {
        Self::new(TrustRegionOptions::default())
    }
}

/// Only the structural nonpositive Cholesky pivot permits a Cauchy fallback.
use crate::dense::positive_definite;

fn cauchy_step<S>(
    workspace: &mut SecondOrderStorage<S>,
    radius: S,
    session: &crate::context::SolveSession<'_, '_, '_>,
) -> Result<bool, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let gradient_norm = norm2(&workspace.gradient)?;
    if gradient_norm == zero {
        workspace.rhs.fill(zero);
        return Ok(false);
    }
    workspace
        .rhs
        .iter_mut()
        .zip(&workspace.gradient)
        .for_each(|(unit, &gradient)| *unit = -gradient / gradient_norm);
    crate::dense::matrix_vector_into(
        workspace.hessian.view(),
        &workspace.rhs,
        &mut workspace.model_product,
        &mut workspace.gemv_scratch,
        session,
    )?;
    let curvature = dot(&workspace.rhs, &workspace.model_product)?;
    // Compare in scaled arithmetic before dividing: an unbounded Cauchy length
    // need not be representable when the constrained length is just the radius.
    let boundary = curvature <= zero
        || Scaled::from_value(gradient_norm)?
            .cmp_value(Scaled::from_value(radius)?.mul(Scaled::from_value(curvature)?)?)
            != core::cmp::Ordering::Less;
    let length = if boundary {
        radius
    } else {
        gradient_norm / curvature
    };
    workspace.rhs.iter_mut().for_each(|step| *step *= length);
    Ok(boundary)
}

/// Intersect a normalized interior point and direction with the unit sphere.
pub(super) fn boundary_distance<S: RealScalar + NumericScalar<Real = S>>(
    cauchy: &[S],
    difference: &[S],
) -> Result<S, SciOxideError> {
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let a = dot(difference, difference)?;
    let b = checked_sum_of_products(&[one, one], &[dot(cauchy, difference)?; 2])?;
    let c = checked_difference_of_products(one, dot(cauchy, cauchy)?, one, one)?;
    if a > zero {
        let roots = checked_quadratic_roots(a, b, c)?;
        if let Some(root) = roots
            .iter()
            .find(|root| root.im == zero && root.re >= zero && root.re <= one + one)
        {
            return Ok(root.re);
        }
    }
    Err(SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Breakdown,
        "dogleg segment has no real boundary intersection",
    ))
}

fn dogleg_step<S>(
    workspace: &mut SecondOrderStorage<S>,
    radius: S,
    session: &crate::context::SolveSession<'_, '_, '_>,
    cached_newton: &mut Option<Option<S>>,
) -> Result<bool, SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    workspace
        .factor_rhs
        .as_mut_slice()
        .iter_mut()
        .zip(&workspace.gradient)
        .for_each(|(rhs, &gradient)| *rhs = -gradient);

    let prepared = if let Some(prepared) = *cached_newton {
        prepared
    } else {
        let has_newton = positive_definite(session.invoke(
            scioxide_core::EvaluationRole::Factorization,
            || {
                CholeskyDecomposition::factor_into(
                    &workspace.hessian,
                    scioxide_linalg::HermitianInput::Exact,
                    CholeskyPivoting::None,
                    &scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(ExecutionPolicy::Sequential),
                    &mut workspace.cholesky_factor,
                    &mut workspace.cholesky_workspace,
                    session.execution()?,
                )
            },
        ))?;
        if has_newton {
            session.invoke(scioxide_core::EvaluationRole::LinearSolve, || {
                workspace
                    .cholesky_factor
                    .solve_into(
                        &workspace.factor_rhs,
                        &scioxide_linalg::DirectSolveOptions::new(
                            scioxide_linalg::RankPolicy::Exact,
                        ),
                        workspace.factor_solution.view_mut(),
                        &mut workspace.cholesky_workspace,
                        session.execution()?,
                    )
                    .and_then(|report| report.termination().require_accepted())
            })?;
        }
        let prepared = if has_newton {
            Some(norm2(workspace.factor_solution.as_slice())?)
        } else {
            None
        };
        *cached_newton = Some(prepared);
        prepared
    };
    let newton_norm = prepared.unwrap_or(<S as scioxide_core::AlgebraicScalar>::ZERO);
    let has_newton = prepared.is_some();
    if has_newton && newton_norm <= radius {
        workspace
            .direction
            .copy_from_slice(workspace.factor_solution.as_slice());
        return Ok(newton_norm == radius);
    }
    let boundary = cauchy_step(workspace, radius, session)?;
    if !has_newton || boundary {
        workspace.direction.copy_from_slice(&workspace.rhs);
        return Ok(boundary);
    }
    // Form the segment direction at the Newton scale, then normalize it.
    // The intersection uses a unit direction and a unit sphere, even when the
    // Newton point is vastly farther away than the trust radius.
    workspace
        .direction
        .iter_mut()
        .zip(
            workspace
                .factor_solution
                .as_slice()
                .iter()
                .zip(&mut workspace.rhs),
        )
        .try_for_each(|(difference, (&newton, cauchy))| {
            *difference = checked_difference(newton / newton_norm, *cauchy / newton_norm)?;
            *cauchy /= radius;
            Ok::<(), SciOxideError>(())
        })?;
    let direction_norm = norm2(&workspace.direction)?;
    if direction_norm == <S as scioxide_core::AlgebraicScalar>::ZERO {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "dogleg direction vanished",
        ));
    }
    workspace
        .direction
        .iter_mut()
        .for_each(|direction| *direction /= direction_norm);
    let interpolation = boundary_distance(&workspace.rhs, &workspace.direction)?;
    workspace
        .direction
        .iter_mut()
        .zip(&workspace.rhs)
        .try_for_each(|(difference, &cauchy)| {
            *difference =
                checked_sum_of_products(&[radius, radius * interpolation], &[cauchy, *difference])?;
            Ok::<(), SciOxideError>(())
        })?;
    Ok(true)
}

use crate::globalization::next_radius;

fn accepts<S: RealScalar>(current: S, trial: S, ratio: Scaled<S>) -> Result<bool, SciOxideError> {
    Ok(trial < current
        && ratio.cmp_value(Scaled::from_value(S::try_from_f64_lossy(
            ACCEPT_THRESHOLD,
        )?)?)
            == core::cmp::Ordering::Greater)
}

impl<S> TrustRegionOptimizer<S>
where
    S: RealScalar + NumericScalar<Real = S>,
{
    /// Creates the optimizer with the given options.
    #[must_use]
    pub const fn new(options: TrustRegionOptions<S>) -> Self {
        Self { options }
    }

    /// Returns the options.
    pub const fn options(&self) -> &TrustRegionOptions<S> {
        &self.options
    }

    /// Returns the workspace requirements for `request`.
    ///
    /// # Errors
    ///
    /// This method does not fail.
    pub const fn workspace_requirements(
        &self,
        request: TrustRegionWorkspaceRequest,
    ) -> Result<TrustRegionOptimizerWorkspaceRequirements, SciOxideError> {
        let (cg, second_order) = match self.options.subproblem {
            TrustRegionSubproblem::Dogleg => (false, false),
            TrustRegionSubproblem::SteihaugCg {
                second_order_stopping,
                ..
            } => (true, second_order_stopping),
        };
        Ok(TrustRegionOptimizerWorkspaceRequirements {
            shape: SecondOrderShape {
                dimension: request.dimension,
                dense: !(cg && request.matrix_free) || second_order,
                regularize: false,
                cg,
                second_order,
            },
        })
    }

    /// Allocates a workspace for `request`.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// reserved.
    pub fn workspace(
        &self,
        request: TrustRegionWorkspaceRequest,
    ) -> Result<TrustRegionOptimizerWorkspace<S>, SciOxideError> {
        let mut workspace = TrustRegionOptimizerWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(request)?)?;
        Ok(workspace)
    }

    /// Minimizes `f` from `x0` and writes the result to `output`.
    ///
    /// The workspace grows to fit [`TrustRegionWorkspaceRequest::of`]`(f)`.
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// parent [`SolveContext`](scioxide_core::SolveContext). When a limit,
    /// cancellation or deadline stops the solve after it has accepted a point,
    /// the call returns `Ok` with an unsuccessful termination. On error,
    /// `output` is unchanged and [`TrustRegionOptimizerWorkspace::last_attempt`] describes the
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
    /// - `InvalidInput` if second-order stopping is on and `f` provides only
    ///   Hessian-vector products.
    /// - `SizeOverflow`, `ResourceLimit` or `AllocationFailure` if the
    ///   workspace cannot grow.
    ///
    /// Errors returned by `f` are returned unchanged.
    pub fn minimize_into<'scope, 'capability, F>(
        &self,
        f: &F,
        x0: &[S],
        output: &mut OptimizeResult<S>,
        workspace: &mut TrustRegionOptimizerWorkspace<S>,
        scope: impl Into<SolveScope<'scope, 'capability>>,
    ) -> Result<(), SciOxideError>
    where
        F: ObjectiveFunction<S> + ?Sized,
    {
        workspace.shell.reject(AttemptKind::Call);
        let scope = scope.into();
        crate::validation::unbounded(f, x0)?;
        workspace.try_reserve(&self.workspace_requirements(TrustRegionWorkspaceRequest::of(f))?)?;
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
            match self.options.subproblem {
                TrustRegionSubproblem::Dogleg => {
                    scioxide_core::algorithm_id!("optimize", "TrustRegion", 1)
                }
                TrustRegionSubproblem::SteihaugCg { .. } => {
                    scioxide_core::algorithm_id!("optimize", "TrustNewtonCg", 1)
                }
            },
            dimension,
            &self.options,
        )
    }

    const fn cg_iterations(&self) -> Option<Budget> {
        match self.options.subproblem {
            TrustRegionSubproblem::Dogleg => None,
            TrustRegionSubproblem::SteihaugCg {
                max_inner_iterations,
                ..
            } => Some(max_inner_iterations),
        }
    }

    const fn second_order(&self) -> bool {
        matches!(
            self.options.subproblem,
            TrustRegionSubproblem::SteihaugCg {
                second_order_stopping: true,
                ..
            }
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
        // The curvature check needs the dense Hessian; it then replaces
        // Hessian-vector products.
        let matrix_free = self.cg_iterations().is_some()
            && f.has_hessian_vector_product()
            && !self.second_order();
        workspace.try_reserve_model(x0.len(), false, !matrix_free)?;
        output.try_reserve(x0.len())?;
        workspace.reset();
        workspace.x.copy_from_slice(x0);
        let derivative_strategy = self.options.derivative_strategy();
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;

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
        let initial_norm = norm2(&workspace.gradient)?;
        let mut radius = if self.second_order() {
            initial_norm.max(one)
        } else if initial_norm > zero {
            initial_norm
        } else {
            one
        }
        .min(S::try_from_f64_lossy(MAX_RADIUS_FACTOR)?);

        let mut model_valid = false;
        let mut cached_newton = None;
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
            let gradient_small = self.options.gradient_tolerance().is_satisfied(
                norm2(&workspace.gradient)?,
                crate::utils::gradient_scale(value, &workspace.x),
            )?;
            if gradient_small && !self.second_order() {
                return finish_second_order(
                    workspace,
                    output,
                    value,
                    iteration,
                    OptimizeTermination::GradientSmall,
                    "gradient norm below tolerance",
                );
            }
            if !matrix_free && !model_valid {
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
            if !matrix_free && !model_valid {
                workspace.hessian.project_hermitian_in_place()?;
                model_valid = true;
            }
            deadline.proposal()?;
            let boundary = if gradient_small && self.second_order() {
                if !super::newton_cg::stationary_direction(
                    workspace,
                    radius,
                    deadline,
                    &mut cached_newton,
                )? {
                    return finish_second_order(
                        workspace,
                        output,
                        value,
                        iteration,
                        OptimizeTermination::Converged,
                        "first and second order stationarity checked",
                    );
                }
                true
            } else if let Some(limit) = self.cg_iterations() {
                super::newton_cg::step(f, workspace, radius, limit, matrix_free, deadline)?
            } else {
                dogleg_step(workspace, radius, deadline, &mut cached_newton)?
            };
            if matrix_free {
                f.hessian_vector_product_into(
                    &workspace.x,
                    &workspace.direction,
                    &mut workspace.model_product,
                )?;
            } else {
                crate::dense::matrix_vector_into(
                    workspace.hessian.view(),
                    &workspace.direction,
                    &mut workspace.model_product,
                    &mut workspace.gemv_scratch,
                    deadline,
                )?;
            }
            let predicted = crate::utils::model_prediction(
                &workspace.gradient,
                &workspace.direction,
                &workspace.model_product,
            )?;
            if predicted.is_zero() || predicted.is_sign_negative() {
                radius = next_radius(radius, Scaled::ZERO, boundary)?;
                deadline.reject()?;
                if radius < <S as RealScalar>::EPSILON {
                    return finish_second_order(
                        workspace,
                        output,
                        value,
                        iteration + 1,
                        OptimizeTermination::Stagnation,
                        "nonpositive model reduction and exhausted trust radius",
                    );
                }
                continue;
            }
            // An unrepresentable point or a non-finite value rejects the step
            // and contracts the radius.
            let trial_value = if crate::utils::fill_exact_trial(
                &mut workspace.trial_x,
                &workspace.x,
                one,
                &workspace.direction,
            )? {
                Some(f.eval(&workspace.trial_x)?).filter(|value| value.is_finite())
            } else {
                None
            };
            let Some(trial_value) = trial_value else {
                radius = next_radius(radius, Scaled::ZERO, boundary)?;
                deadline.reject()?;
                if radius < <S as RealScalar>::EPSILON {
                    return finish_second_order(
                        workspace,
                        output,
                        value,
                        iteration + 1,
                        OptimizeTermination::Stagnation,
                        "rejected trials exhausted the trust radius",
                    );
                }
                continue;
            };
            let ratio = Scaled::from_value(value)?
                .sub(Scaled::from_value(trial_value)?)?
                .div(predicted)?;
            radius = next_radius(radius, ratio, boundary)?;
            if accepts(value, trial_value, ratio)? {
                let finite_difference =
                    crate::derivatives::prepared(&mut workspace.gradient_difference)?;
                get_gradient_into(
                    f,
                    &workspace.trial_x,
                    trial_value,
                    derivative_strategy,
                    &mut workspace.trial_gradient,
                    finite_difference,
                    deadline.execution()?,
                )?;
                deadline.accept(scioxide_core::PartialResultTrust::Admissible)?;
                core::mem::swap(&mut workspace.x, &mut workspace.trial_x);
                core::mem::swap(&mut workspace.gradient, &mut workspace.trial_gradient);
                model_valid = false;
                cached_newton = None;
                value = trial_value;
                workspace.accepted_value = Some(value);
                workspace.accepted_gradient = true;
                workspace.accepted_iterations = iteration + 1;
            } else {
                deadline.reject()?;
            }
            if radius < <S as RealScalar>::EPSILON {
                return finish_second_order(
                    workspace,
                    output,
                    value,
                    iteration + 1,
                    OptimizeTermination::Stagnation,
                    "trust-region radius below machine epsilon",
                );
            }
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

impl<S> TrustRegionOptimizer<S>
where
    S: RealScalar + NumericScalar<Real = S>,
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
        let mut workspace = self.workspace(TrustRegionWorkspaceRequest::of(f))?;
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

impl<S, F> crate::traits::Optimizer<S, F> for TrustRegionOptimizer<S>
where
    S: RealScalar + NumericScalar<Real = S>,
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
    fn prepared_trust_region_converges_transactionally() {
        let solver = TrustRegionOptimizer::<f64>::default();
        let mut workspace = solver
            .workspace(TrustRegionWorkspaceRequest::dense(2))
            .unwrap();
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
        assert!(
            workspace
                .last_attempt()
                .unwrap()
                .counts()
                .own()
                .provider(scioxide_core::EvaluationRole::Factorization)
                >= 1
        );
    }
    fn model_step<S: RealScalar + NumericScalar<Real = S>>(
        hessian: &[S],
        gradient: &[S],
        radius: S,
    ) -> (alloc::vec::Vec<S>, bool, usize) {
        let solver = TrustRegionOptimizer::<S>::default();
        let mut prepared = solver
            .workspace(TrustRegionWorkspaceRequest::dense(gradient.len()))
            .unwrap();
        let workspace = &mut prepared.storage;
        workspace.hessian.as_mut_slice().copy_from_slice(hessian);
        workspace.hessian.project_hermitian_in_place().unwrap();
        workspace.gradient.copy_from_slice(gradient);
        let (boundary, counts) = crate::context::sequential(
            solver.identity(gradient.len()),
            scioxide_core::Budget::ONE,
            |context| {
                let session =
                    crate::context::SolveSession::new(context, scioxide_core::argument!("context"));
                let boundary = dogleg_step(workspace, radius, &session, &mut None)?;
                Ok((boundary, session.counts()?))
            },
        )
        .unwrap();
        assert_eq!(
            counts.provider(scioxide_core::EvaluationRole::Factorization),
            1
        );
        let solves = counts.provider(scioxide_core::EvaluationRole::LinearSolve);
        (workspace.direction.clone(), boundary, solves)
    }

    #[test]
    fn rejected_radii_reuse_positive_and_indefinite_factorization_outcomes() {
        for diagonal in [2.0_f64, -2.0] {
            let solver = TrustRegionOptimizer::default();
            let mut prepared = solver
                .workspace(TrustRegionWorkspaceRequest::dense(1))
                .unwrap();
            let work = &mut prepared.storage;
            work.hessian.as_mut_slice()[0] = diagonal;
            work.gradient[0] = 1.0;
            crate::context::sequential(solver.identity(1), scioxide_core::Budget::ONE, |context| {
                let session =
                    crate::context::SolveSession::new(context, scioxide_core::argument!("scope"));
                let mut cache = None;
                for radius in [1.0, 0.25, 0.0625] {
                    dogleg_step(work, radius, &session, &mut cache)?;
                    assert!(norm2(&work.direction)? <= radius);
                }
                let counts = session.counts()?;
                assert_eq!(
                    counts.provider(scioxide_core::EvaluationRole::Factorization),
                    1
                );
                assert_eq!(
                    counts.provider(scioxide_core::EvaluationRole::LinearSolve),
                    usize::from(diagonal > 0.0)
                );
                Ok(())
            })
            .unwrap();
        }
    }

    #[test]
    fn stationary_curvature_checks_reuse_the_same_ldl_model() {
        for diagonal in [2.0_f64, -2.0] {
            let solver = TrustRegionOptimizer::new(TrustRegionOptions::default().with_subproblem(
                TrustRegionSubproblem::SteihaugCg {
                    max_inner_iterations: scioxide_core::Budget::ONE,
                    second_order_stopping: true,
                },
            ));
            let mut prepared = solver
                .workspace(TrustRegionWorkspaceRequest::dense(1))
                .unwrap();
            let work = &mut prepared.storage;
            work.hessian.as_mut_slice()[0] = diagonal;
            crate::context::sequential(solver.identity(1), scioxide_core::Budget::ONE, |context| {
                let session =
                    crate::context::SolveSession::new(context, scioxide_core::argument!("scope"));
                let mut cache = None;
                for radius in [1.0, 0.25, 0.0625] {
                    assert_eq!(
                        super::super::newton_cg::stationary_direction(
                            work, radius, &session, &mut cache
                        )?,
                        diagonal < 0.0
                    );
                }
                let counts = session.counts()?;
                assert_eq!(
                    counts.provider(scioxide_core::EvaluationRole::Factorization),
                    1
                );
                assert_eq!(
                    counts.provider(scioxide_core::EvaluationRole::LinearSolve),
                    usize::from(diagonal < 0.0)
                );
                Ok(())
            })
            .unwrap();
        }
    }

    macro_rules! numerical_contract {
        ($name:ident, $scalar:ty, $tolerance:expr, $small:expr, $large:expr) => {
            #[test]
            fn $name() {
                let tolerance: $scalar = $tolerance;
                // Acceptance has a meaningful threshold and requires actual decrease.
                for rho in [-1.0, 0.0, 1e-8, 0.15] {
                    assert!(
                        !accepts(1.0 as $scalar, 0.9, Scaled::from_value(rho).unwrap()).unwrap()
                    );
                }
                assert!(accepts(1.0 as $scalar, 0.9, Scaled::from_value(0.2).unwrap()).unwrap());
                assert!(!accepts(1.0 as $scalar, 1.0, Scaled::from_value(1.0).unwrap()).unwrap());
                // Interior success leaves the radius alone; expansion cannot shrink it.
                assert_eq!(
                    next_radius(100.0 as $scalar, Scaled::from_value(1.0).unwrap(), false).unwrap(),
                    100.0
                );
                assert_eq!(
                    next_radius(100.0 as $scalar, Scaled::from_value(1.0).unwrap(), true).unwrap(),
                    200.0
                );
                assert_eq!(
                    next_radius(1e8 as $scalar, Scaled::from_value(1.0).unwrap(), true).unwrap(),
                    1e8
                );
                assert_eq!(
                    next_radius(100.0 as $scalar, Scaled::from_value(0.25).unwrap(), true).unwrap(),
                    100.0
                );
                assert_eq!(
                    next_radius(100.0 as $scalar, Scaled::from_value(0.75).unwrap(), true).unwrap(),
                    100.0
                );
                assert_eq!(
                    next_radius(100.0 as $scalar, Scaled::from_value(0.1).unwrap(), false).unwrap(),
                    25.0
                );
                for product in [0.0, 2.0, 4.0] {
                    let predicted =
                        crate::utils::model_prediction(&[1.0 as $scalar], &[1.0], &[product])
                            .unwrap();
                    assert!(predicted.is_sign_negative());
                }
                assert!(
                    crate::utils::model_prediction(&[-1.0 as $scalar], &[1.0], &[2.0])
                        .unwrap()
                        .is_zero()
                );

                // Singular/indefinite models use the constrained line minimizer.
                for diagonal in [0.0, -2.0] {
                    let (step, boundary, solves) =
                        model_step::<$scalar>(&[2.0, 0.0, 0.0, diagonal], &[2.0, 0.0], 10.0);
                    assert!(!boundary);
                    assert_eq!(solves, 0);
                    assert!((step[0] + 1.0).abs() <= tolerance);
                    assert_eq!(step[1], 0.0);
                }
                let (step, boundary, solves) = model_step::<$scalar>(&[-2.0], &[1.0], 3.0);
                assert!(boundary);
                assert_eq!(solves, 0);
                assert!((step[0] + 3.0).abs() <= tolerance);

                // Analytic two-dimensional dogleg geometry, across physical scales.
                for scale in [$small as $scalar, 1.0, $large] {
                    let (step, boundary, solves) =
                        model_step::<$scalar>(&[1.0, 0.0, 0.0, 10.0], &[scale, scale], scale);
                    assert!(boundary);
                    assert_eq!(solves, 1);
                    let normalized = [step[0] / scale, step[1] / scale];
                    assert!((norm2(&normalized).unwrap() - 1.0).abs() <= tolerance * 8.0);
                    let prediction = -normalized[0]
                        - normalized[1]
                        - 0.5
                            * (normalized[0] * normalized[0]
                                + 10.0 * normalized[1] * normalized[1]);
                    assert!(prediction > 0.0);
                    // Exact solution of the line-circle intersection for this model.
                    let c = -2.0 as $scalar / 11.0;
                    let d = [-1.0 - c, -0.1 - c];
                    let a = d[0] * d[0] + d[1] * d[1];
                    let b = 2.0 * c * (d[0] + d[1]);
                    let t = (-b + (b * b - 4.0 * a * (2.0 * c * c - 1.0)).sqrt()) / (2.0 * a);
                    assert!((normalized[0] - (c + t * d[0])).abs() <= tolerance * 8.0);
                    assert!((normalized[1] - (c + t * d[1])).abs() <= tolerance * 8.0);
                }
                // Antisymmetric contributions cannot alter the scalar quadratic.
                let (symmetric, _, _) =
                    model_step::<$scalar>(&[2.0, 0.0, 0.0, 2.0], &[1.0, 1.0], 2.0);
                let (projected, _, _) =
                    model_step::<$scalar>(&[2.0, 10.0, -10.0, 2.0], &[1.0, 1.0], 2.0);
                assert_eq!(symmetric, projected);
            }
        };
    }
    numerical_contract!(numerical_contract_f32, f32, 2e-6, 1e-20, 1e20);
    numerical_contract!(numerical_contract_f64, f64, 2e-14, 1e-160, 1e160);

    #[test]
    fn only_structural_nonpositive_curvature_allows_fallback() {
        assert!(
            !positive_definite(Err(SciOxideError::domain_error(
                scioxide_core::argument!("a"),
                scioxide_core::DomainReason::OutsideDomain,
                "nonpositive pivot",
            )))
            .unwrap()
        );
        assert!(matches!(
            (positive_definite(Err(scioxide_core::SciOxideError::singular(None))))
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::Singular { .. })
        ));
        assert!(matches!(
            positive_definite(Err(SciOxideError::size_overflow(

                scioxide_core::quantity!("evaluations"),
            ))),
            Err(error) if matches!(error.view(), scioxide_core::SciOxideErrorView::SizeOverflow { .. })
        ));
        assert!(
            positive_definite(Err(SciOxideError::domain_error(
                scioxide_core::argument!("provider"),
                scioxide_core::DomainReason::OutsideDomain,
                "provider failure",
            )))
            .is_err()
        );
    }

    #[test]
    fn tiny_positive_gain_rejects_without_trial_gradient_or_commit() {
        struct MisleadingModel(core::cell::Cell<usize>);
        impl ObjectiveFunction<f64> for MisleadingModel {
            fn dim(&self) -> usize {
                1
            }
            fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
                Ok(x[0] * 1e-8)
            }
            fn gradient_into(&self, _: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
                self.0.set(self.0.get() + 1);
                output[0] = 1.0;
                Ok(true)
            }
            fn hessian_into(
                &self,
                _: &[f64],
                output: &mut Matrix<f64>,
            ) -> Result<bool, SciOxideError> {
                output.as_mut_slice()[0] = 1.0;
                Ok(true)
            }
        }
        let objective = MisleadingModel(core::cell::Cell::new(0));
        let solver = TrustRegionOptimizer::new(
            TrustRegionOptions::default().with_max_iterations(scioxide_core::Budget::ONE),
        );
        let result = solver.minimize(&objective, &[1.0]).unwrap();
        assert_eq!(result.x(), &[1.0]);
        assert_eq!(result.fun(), 1e-8);
        assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
        assert_eq!(objective.0.get(), 1);
        assert_eq!(result.fun_evals(), 2);
    }
    macro_rules! nonconvex_contract {
        ($name:ident, $scalar:ty, $tolerance:expr) => {
            #[test]
            fn $name() {
                struct DoubleWell;
                impl ObjectiveFunction<$scalar> for DoubleWell {
                    fn dim(&self) -> usize {
                        1
                    }
                    fn eval(&self, x: &[$scalar]) -> Result<$scalar, SciOxideError> {
                        Ok((x[0] * x[0] - 1.0).powi(2))
                    }
                    fn gradient_into(
                        &self,
                        x: &[$scalar],
                        output: &mut [$scalar],
                    ) -> Result<bool, SciOxideError> {
                        output[0] = 4.0 * x[0] * (x[0] * x[0] - 1.0);
                        Ok(true)
                    }
                    fn hessian_into(
                        &self,
                        x: &[$scalar],
                        output: &mut Matrix<$scalar>,
                    ) -> Result<bool, SciOxideError> {
                        output.as_mut_slice()[0] = 12.0 * x[0] * x[0] - 4.0;
                        Ok(true)
                    }
                }
                let solver = TrustRegionOptimizer::<$scalar>::default();
                let result = solver.minimize(&DoubleWell, &[0.25]).unwrap();
                assert!(result.is_success());
                assert!((result.x()[0] - 1.0).abs() < $tolerance);
                assert!(result.fun() < $tolerance * $tolerance);
                assert!(result.grad().unwrap()[0].abs() < $tolerance * 8.0);
                let mut workspace = solver
                    .workspace(TrustRegionWorkspaceRequest::dense(1))
                    .unwrap();
                let mut prepared = OptimizeResult::empty();
                solver
                    .minimize_into(
                        &DoubleWell,
                        &[0.25],
                        &mut prepared,
                        &mut workspace,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap();
                assert_eq!(result.x(), prepared.x());
                assert_eq!(result.fun_evals(), prepared.fun_evals());
                let counts = workspace.last_attempt().unwrap().counts().own();
                assert!(
                    counts.provider(scioxide_core::EvaluationRole::Factorization)
                        > counts.provider(scioxide_core::EvaluationRole::LinearSolve)
                );
            }
        };
    }
    nonconvex_contract!(nonconvex_f32, f32, 2e-5);
    nonconvex_contract!(nonconvex_f64, f64, 2e-7);

    #[test]
    fn empty_problem_and_trial_derivative_failure_preserve_valid_state() {
        struct Empty;
        impl ObjectiveFunction<f64> for Empty {
            fn dim(&self) -> usize {
                0
            }
            fn eval(&self, _: &[f64]) -> Result<f64, SciOxideError> {
                Ok(3.0)
            }
            fn gradient_into(&self, _: &[f64], _: &mut [f64]) -> Result<bool, SciOxideError> {
                Ok(true)
            }
        }
        let solver = TrustRegionOptimizer::<f64>::default();
        let empty = solver.minimize(&Empty, &[]).unwrap();
        assert!(empty.is_success());
        assert!(empty.x().is_empty());
        assert_eq!(empty.fun(), 3.0);
        struct FailTrial;
        impl ObjectiveFunction<f64> for FailTrial {
            fn dim(&self) -> usize {
                1
            }
            fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
                Ok(x[0] * x[0])
            }
            fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
                output[0] = 2.0 * x[0];
                if x[0] != 1.0 {
                    return Err(scioxide_core::SciOxideError::singular(None));
                }
                Ok(true)
            }
            fn hessian_into(
                &self,
                _: &[f64],
                output: &mut Matrix<f64>,
            ) -> Result<bool, SciOxideError> {
                output.as_mut_slice()[0] = 2.0;
                Ok(true)
            }
        }
        let mut workspace = solver
            .workspace(TrustRegionWorkspaceRequest::dense(1))
            .unwrap();
        let mut output = OptimizeResult::empty();
        assert!(matches!(
            (solver.minimize_into(
                &FailTrial,
                &[1.0],
                &mut output,
                &mut workspace,
                ExecutionContext::sequential()
            ))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::Singular { .. })
        ));
        assert_eq!(workspace.storage.x, &[1.0]);
        assert_eq!(workspace.storage.accepted_value, Some(1.0));
        assert_eq!(workspace.storage.gradient, &[2.0]);
    }
}
