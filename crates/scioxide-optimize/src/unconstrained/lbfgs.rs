//! Limited-memory BFGS (L-BFGS).

use crate::quasi_newton::{damp_pair, positive, scaled_dot};

use scioxide_core::Scaled;
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

fn history_slot(start: usize, offset: usize, capacity: usize) -> usize {
    (start + offset) % capacity
}

/// Applies the limited-memory inverse Hessian to `q`, writing `r`. The initial
/// matrix is `γI` with `γ = sᵀy/yᵀy` of the newest stored pair, or `I` without
/// history (Nocedal–Wright (7.20)).
pub(super) fn two_loop_recursion<S>(
    workspace: &mut FirstOrderStorage<S>,
) -> Result<(), SciOxideError>
where
    S: RealScalar,
{
    let gamma = newest_scale(workspace)?;
    apply_inverse(workspace, 0, gamma)
}

// Initial scaling of the stored model; `1` without usable history.
fn newest_scale<S: RealScalar>(
    workspace: &FirstOrderStorage<S>,
) -> Result<Scaled<S>, SciOxideError> {
    let one = Scaled::from_value(S::ONE)?;
    let len = workspace.history_len;
    if len == 0 {
        return Ok(one);
    }
    let slot = history_slot(workspace.history_start, len - 1, workspace.history_capacity);
    let range = slot * workspace.dimension..(slot + 1) * workspace.dimension;
    let yy = scaled_dot(
        &workspace.history_y[range.clone()],
        &workspace.history_y[range],
    )?;
    if positive(yy) {
        workspace.history_curvature[slot].div(yy)
    } else {
        Ok(one)
    }
}

// Two-loop recursion over the stored pairs from offset `first` on, with the
// initial matrix `gamma I`.
fn apply_inverse<S: RealScalar>(
    workspace: &mut FirstOrderStorage<S>,
    first: usize,
    gamma: Scaled<S>,
) -> Result<(), SciOxideError> {
    let n = workspace.dimension;
    let capacity = workspace.history_capacity;
    let len = workspace.history_len;
    for offset in (first..len).rev() {
        let slot = history_slot(workspace.history_start, offset, capacity);
        let range = slot * n..(slot + 1) * n;
        let alpha = scaled_dot(&workspace.history_s[range.clone()], &workspace.q)?
            .div(workspace.history_curvature[slot])?;
        workspace.history_alpha[slot] = alpha;
        axpy(&mut workspace.q, -alpha, &workspace.history_y[range])?;
    }
    workspace.r.fill(S::ZERO);
    axpy(&mut workspace.r, gamma, &workspace.q)?;
    for offset in first..len {
        let slot = history_slot(workspace.history_start, offset, capacity);
        let range = slot * n..(slot + 1) * n;
        let beta = scaled_dot(&workspace.history_y[range.clone()], &workspace.r)?
            .div(workspace.history_curvature[slot])?;
        let coefficient = workspace.history_alpha[slot].sub(beta)?;
        axpy(&mut workspace.r, coefficient, &workspace.history_s[range])?;
    }
    Ok(())
}

// Native vector kernels have one scalar coefficient conversion per pass.
pub(super) fn axpy<S: RealScalar>(
    out: &mut [S],
    coefficient: Scaled<S>,
    input: &[S],
) -> Result<(), SciOxideError> {
    let native = coefficient
        .round()
        .ok()
        .filter(|a| a.abs() >= S::MIN_POSITIVE || coefficient.is_zero());
    for (out, &input) in out.iter_mut().zip(input) {
        if let Some(a) = native {
            let product = a * input;
            let value = *out + product;
            if !scioxide_core::backend::standard::product_needs_retry(a, input, product)
                && value.is_finite()
            {
                *out = value;
                continue;
            }
        }
        *out = Scaled::from_value(*out)?
            .add(coefficient.mul(Scaled::from_value(input)?)?)?
            .round()?;
    }
    Ok(())
}

// With R = triu(S'Y), D = diag(S'Y), t = R^-1 S'y over the pairs from offset
// `first` on: y'Hy = gamma*y'y + t'(D + gamma*Y'Y)t - 2*gamma*(Y'y)'t.
// This computes only a scalar, without a second dimension-wide vector recursion.
fn compact_curvature<S: RealScalar>(
    workspace: &mut FirstOrderStorage<S>,
    first: usize,
    gamma: S,
) -> Option<(S, S)> {
    let m = workspace.history_capacity;
    let len = workspace.history_len;
    if first < len && !workspace.history_cache_valid {
        return None;
    }
    let yy = scaled_dot(&workspace.y, &workspace.y).ok()?.round().ok()?;
    for offset in (first..len).rev() {
        let i = history_slot(workspace.history_start, offset, m);
        let range = i * workspace.dimension..(i + 1) * workspace.dimension;
        let mut a = scaled_dot(&workspace.history_s[range.clone()], &workspace.y)
            .ok()?
            .round()
            .ok()?;
        workspace.history_b[i] = scaled_dot(&workspace.history_y[range], &workspace.y)
            .ok()?
            .round()
            .ok()?;
        for j in offset + 1..len {
            let j = history_slot(workspace.history_start, j, m);
            a -= workspace.history_sy[i * m + j] * workspace.history_t[j];
        }
        workspace.history_t[i] = a / workspace.history_sy[i * m + i];
    }
    let mut sum = scioxide_core::CompensatedSum::ZERO;
    let mut magnitude = (gamma * yy).abs();
    sum.add(gamma * yy).ok()?;
    for offset in first..len {
        let i = history_slot(workspace.history_start, offset, m);
        let t = workspace.history_t[i];
        let mut product = workspace.history_sy[i * m + i] * t;
        for j in first..len {
            let j = history_slot(workspace.history_start, j, m);
            product += gamma * workspace.history_yy[i * m + j] * workspace.history_t[j];
        }
        let quadratic = t * product;
        let cross = -S::TWO * gamma * workspace.history_b[i] * t;
        magnitude = magnitude + quadratic.abs() + cross.abs();
        sum.add(quadratic).ok()?;
        sum.add(cross).ok()?;
    }
    let value = sum.total().ok()?;
    let terms = len - first;
    let error =
        S::try_from_usize_exact(64usize.checked_mul(terms.max(1).checked_mul(terms.max(1))?)?)
            .ok()?
            * S::EPSILON
            * magnitude;
    (value > error && gamma.is_finite() && magnitude.is_finite()).then_some((value, error))
}

/// Stage the inverse-form correction, forming H*y only for damping or an
/// unresolved compact scalar. Accepted history is unchanged here.
///
/// The test uses the matrix that the pair updates (Nocedal–Wright (7.19),
/// (7.20)): the pairs that stay in memory applied to the initial matrix `γI`,
/// `γ = sᵀy/yᵀy` of the new pair. When `sᵀy ≤ 0` the pair cannot pass and the
/// stored model, which stays positive definite, is the damping target.
pub(super) fn prepare_correction<S: RealScalar>(
    workspace: &mut FirstOrderStorage<S>,
    policy: crate::CurvaturePolicy<S>,
) -> Result<crate::HessianUpdateReport, SciOxideError> {
    let sy = scaled_dot(&workspace.s, &workspace.y)?;
    let yy = scaled_dot(&workspace.y, &workspace.y)?;
    let (first, gamma) = if positive(sy) && positive(yy) {
        let evicted = workspace.history_len == workspace.history_capacity;
        (usize::from(evicted), sy.div(yy)?)
    } else {
        (0, newest_scale(workspace)?)
    };
    if let Ok(scale) = gamma.round()
        && let Some((model, error)) = compact_curvature(workspace, first, scale)
        && let Ok(curvature) = sy.round()
        && curvature > policy.threshold() * (model + error)
    {
        let cutoff = scaled_dot(&workspace.s, &workspace.s)?
            .mul(yy)?
            .sqrt()?
            .mul(Scaled::from_value(S::EPSILON)?)?;
        return Ok(if positive(sy) && !sy.abs_le(cutoff) {
            crate::HessianUpdateReport::Updated
        } else {
            crate::HessianUpdateReport::Skipped(crate::UpdateSkipReason::Curvature)
        });
    }
    let result = (|| {
        workspace.q.copy_from_slice(&workspace.y);
        apply_inverse(workspace, first, gamma)?;
        damp_pair(&workspace.y, &mut workspace.s, &workspace.r, policy)
    })();
    match result {
        Err(error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            ) =>
        {
            Ok(crate::HessianUpdateReport::Skipped(
                crate::UpdateSkipReason::Unrepresentable,
            ))
        }
        result => result,
    }
}

pub(crate) fn push_history<S>(workspace: &mut FirstOrderStorage<S>, curvature: Scaled<S>)
where
    S: RealScalar + NumericScalar<Real = S>,
{
    let capacity = workspace.history_capacity;
    let slot = if workspace.history_len < capacity {
        let slot = history_slot(workspace.history_start, workspace.history_len, capacity);
        workspace.history_len += 1;
        slot
    } else {
        let slot = workspace.history_start;
        workspace.history_start = history_slot(workspace.history_start, 1, capacity);
        slot
    };
    let range = slot * workspace.dimension..(slot + 1) * workspace.dimension;
    workspace.history_s[range.clone()].copy_from_slice(&workspace.s);
    workspace.history_y[range].copy_from_slice(&workspace.y);
    if workspace.history_len == 1 {
        workspace.history_cache_valid = true;
    }
    let m = workspace.history_capacity;
    for offset in 0..workspace.history_len {
        let other = history_slot(workspace.history_start, offset, m);
        let a = other * workspace.dimension..(other + 1) * workspace.dimension;
        let b = slot * workspace.dimension..(slot + 1) * workspace.dimension;
        let sy = scaled_dot(
            &workspace.history_s[a.clone()],
            &workspace.history_y[b.clone()],
        )
        .and_then(Scaled::round);
        let yy = scaled_dot(
            &workspace.history_y[a.clone()],
            &workspace.history_y[b.clone()],
        )
        .and_then(Scaled::round);
        if let (Ok(sy), Ok(yy)) = (sy, yy) {
            workspace.history_sy[other * m + slot] = sy;
            workspace.history_yy[other * m + slot] = yy;
            workspace.history_yy[slot * m + other] = yy;
            workspace.history_cache_valid &= other != slot || (sy > S::ZERO && yy > S::ZERO);
            // Only Hessian-form bounded updates prepare the additional Gram
            // block and the opposite S'Y triangle.
            if !workspace.history_ss.is_empty() {
                let ys = scaled_dot(
                    &workspace.history_s[b.clone()],
                    &workspace.history_y[a.clone()],
                )
                .and_then(Scaled::round);
                let ss = scaled_dot(&workspace.history_s[a], &workspace.history_s[b])
                    .and_then(Scaled::round);
                if let (Ok(ys), Ok(ss)) = (ys, ss) {
                    workspace.history_sy[slot * m + other] = ys;
                    workspace.history_ss[other * m + slot] = ss;
                    workspace.history_ss[slot * m + other] = ss;
                } else {
                    workspace.history_cache_valid = false;
                }
            }
        } else {
            workspace.history_cache_valid = false;
        }
    }
    workspace.history_curvature[slot] = curvature;
}

/// Sets the stopping tolerance, limits, derivative source, line search, memory
/// and curvature policy of [`Lbfgs`].
///
/// The defaults are the default
/// [`GradientTolerance`](scioxide_core::GradientTolerance), 1000 iterations, the
/// [`DerivativeStrategy::AnalyticOrCentral`](scioxide_differentiate::DerivativeStrategy::AnalyticOrCentral), no
/// time limit, [`StrongWolfeLineSearch`](crate::StrongWolfeLineSearch), a
/// memory of 10 pairs and the default [`CurvaturePolicy`](crate::CurvaturePolicy).
/// The stopping test is described under [Termination](crate#termination).
#[derive(Debug, Clone)]
pub struct LbfgsOptions<
    S: RealScalar + NumericScalar<Real = S>,
    L = crate::line_search::StrongWolfeLineSearch<S>,
> {
    settings: LineSearchSettings<S, L>,
    memory: scioxide_core::Budget,
    curvature: crate::CurvaturePolicy<S>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for LbfgsOptions<S> {
    fn default() -> Self {
        Self {
            settings: LineSearchSettings::new(crate::line_search::StrongWolfeLineSearch::default()),
            memory: scioxide_core::Budget::ONE.saturating_add(9),
            curvature: crate::CurvaturePolicy::default(),
        }
    }
}

line_search_settings!(LbfgsOptions);

impl<S: RealScalar + NumericScalar<Real = S>, L> LbfgsOptions<S, L> {
    /// Sets the line search.
    #[must_use]
    pub fn with_line_search<L2>(self, line_search: L2) -> LbfgsOptions<S, L2> {
        LbfgsOptions {
            settings: self.settings.with_line_search(line_search),
            memory: self.memory,
            curvature: self.curvature,
        }
    }

    /// Sets the number `m` of stored correction pairs.
    #[must_use]
    pub fn with_memory(mut self, memory: scioxide_core::Budget) -> Self {
        self.memory = memory;
        self
    }

    /// Returns the number `m` of stored correction pairs.
    pub const fn memory(&self) -> scioxide_core::Budget {
        self.memory
    }

    /// Sets the curvature policy of the updates.
    #[must_use]
    pub fn with_curvature_policy(mut self, policy: crate::CurvaturePolicy<S>) -> Self {
        self.curvature = policy;
        self
    }

    /// Returns the curvature policy of the updates.
    pub const fn curvature_policy(&self) -> crate::CurvaturePolicy<S> {
        self.curvature
    }
}

/// Describes the workspace that a [`Lbfgs`] solve needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LbfgsWorkspaceRequirements {
    dimension: usize,
    history: usize,
}

impl LbfgsWorkspaceRequirements {
    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }
    /// Returns the number of stored correction pairs.
    pub const fn history(self) -> usize {
        self.history
    }
}

impl<S: RealScalar + NumericScalar<Real = S>> Shape<FirstOrderStorage<S>>
    for LbfgsWorkspaceRequirements
{
    fn fits(&self, storage: &FirstOrderStorage<S>) -> bool {
        storage.fits(self.dimension, self.history, false)
    }
    fn prepare(&self, storage: &mut FirstOrderStorage<S>) -> Result<(), SciOxideError> {
        storage.try_reserve(self.dimension, self.history)
    }
}

crate::attempt::local_workspace!(
    /// Holds the storage and the record of the last call for [`Lbfgs`].
    LbfgsWorkspace,
    FirstOrderStorage<S>,
    LbfgsWorkspaceRequirements
);

/// Minimizes a smooth function by the limited-memory BFGS method (L-BFGS).
///
/// The method stores the last `m` pairs `(sᵢ, yᵢ)` of steps and gradient
/// changes, where `m` is [`LbfgsOptions::memory`](crate::LbfgsOptions::memory)
/// (default 10). It applies the inverse-Hessian approximation to the gradient
/// with the two-loop recursion, starting from the scaled identity
/// `(sᵀy / yᵀy) I` of the latest pair. The curvature policy of the options
/// keeps the approximation positive definite. It tests a new pair against the
/// matrix that the pair updates: the pairs that stay in memory applied to the
/// scaled identity of the new pair. The test therefore does not depend on the
/// scaling of earlier pairs. An iteration costs O(nm)
/// arithmetic plus the line search, and the workspace holds O(nm + m²)
/// values. L-BFGS is the usual choice for smooth problems with many variables;
/// for bounds on the variables use [`LbfgsB`](crate::LbfgsB).
///
/// # Examples
///
/// ```
/// use scioxide_core::{ExecutionContext, SciOxideError};
/// use scioxide_optimize::{Lbfgs, ObjectiveFunction, OptimizeResult};
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
/// let solver = Lbfgs::<f64>::default();
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
/// - D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large
///   scale optimization", *Math. Program.* 45, 503–528, 1989.
/// - J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed.,
///   Springer, 2006, §7.2.
#[derive(Debug, Clone)]
pub struct Lbfgs<
    S: RealScalar + NumericScalar<Real = S>,
    L = crate::line_search::StrongWolfeLineSearch<S>,
> {
    options: LbfgsOptions<S, L>,
}

impl<S: RealScalar + NumericScalar<Real = S>> Default for Lbfgs<S> {
    fn default() -> Self {
        Self::new(LbfgsOptions::default())
    }
}

impl<S, L> Lbfgs<S, L>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: LineSearch<S>,
{
    /// Creates the optimizer with the given options.
    #[must_use]
    pub const fn new(options: LbfgsOptions<S, L>) -> Self {
        Self { options }
    }

    /// Returns the options.
    pub const fn options(&self) -> &LbfgsOptions<S, L> {
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
    ) -> Result<LbfgsWorkspaceRequirements, SciOxideError> {
        Ok(LbfgsWorkspaceRequirements {
            dimension,
            history: self.options.memory().get(),
        })
    }

    /// Allocates a workspace for `dimension` variables.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure` if the storage cannot be
    /// reserved.
    pub fn workspace(&self, dimension: usize) -> Result<LbfgsWorkspace<S>, SciOxideError> {
        let mut workspace = LbfgsWorkspace::new();
        workspace.try_reserve(&self.workspace_requirements(dimension)?)?;
        Ok(workspace)
    }

    /// Minimizes `f` from `x0` and writes the result to `output`.
    ///
    /// `scope` is an [execution context](scioxide_core#execution-contexts) or a
    /// parent [`SolveContext`](scioxide_core::SolveContext). When a limit,
    /// cancellation or deadline stops the solve after it has accepted a point,
    /// the call returns `Ok` with an unsuccessful termination. On error,
    /// `output` is unchanged and [`LbfgsWorkspace::last_attempt`] describes the
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
        workspace: &mut LbfgsWorkspace<S>,
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
            scioxide_core::algorithm_id!("optimize", "Lbfgs", 1),
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
        workspace.try_reserve(x0.len(), self.options.memory().get())?;
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
        }

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
            workspace.q.copy_from_slice(&workspace.gradient);
            match two_loop_recursion(workspace) {
                Ok(()) => workspace
                    .direction
                    .iter_mut()
                    .zip(&workspace.r)
                    .for_each(|(d, &r)| *d = -r),
                Err(error)
                    if matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::NumericalInstability { .. }
                    ) =>
                {
                    workspace.history_len = 0;
                    workspace.history_start = 0;
                    workspace
                        .direction
                        .iter_mut()
                        .zip(&workspace.gradient)
                        .for_each(|(d, &g)| *d = -g);
                }
                Err(error) => return Err(error),
            }
            if !crate::quasi_newton::scaled_dot(&workspace.gradient, &workspace.direction)?
                .is_sign_negative()
            {
                workspace.history_len = 0;
                workspace.history_start = 0;
                workspace
                    .direction
                    .iter_mut()
                    .zip(&workspace.gradient)
                    .for_each(|(direction, &gradient)| *direction = -gradient);
            }
            let directional_derivative = crate::quasi_newton::descent_slope(
                &workspace.gradient,
                &mut workspace.direction,
                workspace.history_len == 0,
            )?;
            let trial = {
                let finite_difference =
                    crate::derivatives::prepared(&mut workspace.finite_difference)?;
                self.options.line_search().search_into(
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
                )?
            };
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
                let finite_difference =
                    crate::derivatives::prepared(&mut workspace.finite_difference)?;
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
            workspace
                .s
                .iter_mut()
                .zip(workspace.trial_x.iter().zip(&workspace.x))
                .for_each(|(step, (&trial, &current))| *step = trial - current);
            workspace
                .y
                .iter_mut()
                .zip(workspace.trial_gradient.iter().zip(&workspace.gradient))
                .for_each(|(difference, (&trial, &current))| *difference = trial - current);
            let update = prepare_correction(workspace, self.options.curvature_policy())?;
            let curvature = scaled_dot(&workspace.y, &workspace.s)?;
            deadline.accept(scioxide_core::PartialResultTrust::Admissible)?;
            if !matches!(update, crate::HessianUpdateReport::Skipped(_)) {
                push_history(workspace, curvature);
            }
            workspace.globalization.record_update(update);
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

impl<S, L> Lbfgs<S, L>
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

impl<S, L, F> crate::traits::Optimizer<S, F> for Lbfgs<S, L>
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
mod compact_tests {
    use super::*;
    use crate::{CurvatureAction, CurvaturePolicy, HessianUpdateReport};

    fn compact_history<S: RealScalar>() {
        let cv = |x| S::try_from_f64_lossy(x).unwrap();
        let mut state = FirstOrderStorage::new();
        state.try_reserve(3, 3).unwrap();
        for iteration in 0..9 {
            let v = cv((iteration + 1) as f64 / 10.0);
            state.s.copy_from_slice(&[S::ONE, v, -v]);
            state
                .y
                .copy_from_slice(&[cv(2.0) + v, cv(3.0) * v, -cv(4.0) * v]);
            let curvature = scaled_dot(&state.s, &state.y).unwrap();
            push_history(&mut state, curvature);
            state.y.copy_from_slice(&[cv(0.7), cv(-0.4), cv(1.2)]);
            let gamma = newest_scale(&state).unwrap().round().unwrap();
            let (compact, _) = compact_curvature(&mut state, 0, gamma).unwrap();
            state.q.copy_from_slice(&state.y);
            two_loop_recursion(&mut state).unwrap();
            let recursive = scaled_dot(&state.y, &state.r).unwrap().round().unwrap();
            assert!((compact - recursive).abs() <= cv(512.0) * S::EPSILON * recursive.abs());
            // Without the oldest pair and with another initial scaling.
            let scale = cv(0.3);
            let (compact, _) = compact_curvature(&mut state, 1, scale).unwrap();
            state.q.copy_from_slice(&state.y);
            apply_inverse(&mut state, 1, Scaled::from_value(scale).unwrap()).unwrap();
            let recursive = scaled_dot(&state.y, &state.r).unwrap().round().unwrap();
            assert!((compact - recursive).abs() <= cv(512.0) * S::EPSILON * recursive.abs());
            state.q.copy_from_slice(&state.y);
            two_loop_recursion(&mut state).unwrap();
            // Construct the same inverse BFGS model independently in dense form.
            let newest = history_slot(state.history_start, state.history_len - 1, 3);
            let gamma =
                state.history_sy[newest * 3 + newest] / state.history_yy[newest * 3 + newest];
            let mut h = [[S::ZERO; 3]; 3];
            for (i, row) in h.iter_mut().enumerate() {
                row[i] = gamma;
            }
            for offset in 0..state.history_len {
                let slot = history_slot(state.history_start, offset, 3);
                let s = &state.history_s[slot * 3..slot * 3 + 3];
                let y = &state.history_y[slot * 3..slot * 3 + 3];
                let mut hy = [S::ZERO; 3];
                for (out, row) in hy.iter_mut().zip(&h) {
                    *out = row.iter().zip(y).fold(S::ZERO, |sum, (&a, &b)| sum + a * b);
                }
                let sy = scaled_dot(s, y).unwrap().round().unwrap();
                let yhy = scaled_dot(y, &hy).unwrap().round().unwrap();
                for (i, row) in h.iter_mut().enumerate() {
                    for (j, value) in row.iter_mut().enumerate() {
                        *value = *value + (S::ONE + yhy / sy) * s[i] * s[j] / sy
                            - (s[i] * hy[j] + hy[i] * s[j]) / sy;
                    }
                }
            }
            for (row, &actual) in h.iter().zip(&state.r) {
                let expected = row
                    .iter()
                    .zip(&state.y)
                    .fold(S::ZERO, |sum, (&a, &b)| sum + a * b);
                assert!(
                    (actual - expected).abs()
                        <= cv(1024.0) * S::EPSILON * (S::ONE + expected.abs())
                );
            }
        }
        state.reset();
        let (empty, _) = compact_curvature(&mut state, 0, cv(0.5)).unwrap();
        let yy = scaled_dot(&state.y, &state.y).unwrap().round().unwrap();
        assert!((empty - cv(0.5) * yy).abs() <= cv(4.0) * S::EPSILON * yy);
    }

    #[test]
    fn compact_scalar_and_ring_eviction_match_dense_inverse() {
        compact_history::<f32>();
        compact_history::<f64>();
    }

    fn damping<S: RealScalar>() {
        let cv = |x| S::try_from_f64_lossy(x).unwrap();
        let mut state = FirstOrderStorage::new();
        state.try_reserve(2, 2).unwrap();
        state.s.copy_from_slice(&[S::ONE, S::ZERO]);
        state.y.copy_from_slice(&[S::ONE, S::ONE]);
        push_history(&mut state, Scaled::from_value(S::ONE).unwrap());
        for (action, expected) in [
            (CurvatureAction::Damp, HessianUpdateReport::Damped),
            (
                CurvatureAction::Skip,
                HessianUpdateReport::Skipped(crate::UpdateSkipReason::Curvature),
            ),
        ] {
            state.s.copy_from_slice(&[-S::ONE, cv(0.1)]);
            state.y.copy_from_slice(&[S::ONE, cv(0.2)]);
            let original = state.s.clone();
            state.q.copy_from_slice(&state.y);
            two_loop_recursion(&mut state).unwrap();
            let mut independent = original.clone();
            let policy = CurvaturePolicy::new(action, cv(0.2)).unwrap();
            let report = damp_pair(&state.y, &mut independent, &state.r, policy).unwrap();
            assert_eq!(report, expected);
            assert_eq!(prepare_correction(&mut state, policy).unwrap(), expected);
            for (&a, &b) in state.s.iter().zip(&independent) {
                assert!((a - b).abs() <= cv(32.0) * S::EPSILON * (S::ONE + b.abs()));
            }
            assert_eq!(state.history_len, 1);
        }
        state.history_cache_valid = false;
        state.s.copy_from_slice(&[S::ONE, S::ONE]);
        assert!(compact_curvature(&mut state, 0, S::ONE).is_none());
        assert!(!matches!(
            prepare_correction(&mut state, CurvaturePolicy::default()).unwrap(),
            HessianUpdateReport::Skipped(_)
        ));
    }

    #[test]
    fn lazy_damping_and_invalid_cache_match_full_model_product() {
        damping::<f32>();
        damping::<f64>();
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
    fn bounded_history_is_reused() {
        let solver = Lbfgs::new(
            LbfgsOptions::<f64>::default().with_memory(scioxide_core::Budget::try_from(2).unwrap()),
        );
        let mut workspace = solver.workspace(2).unwrap();
        let capacities = (
            workspace.storage.history_s.capacity(),
            workspace.storage.history_y.capacity(),
        );
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
        assert!(workspace.storage.history_len <= 2);
        assert_eq!(
            capacities,
            (
                workspace.storage.history_s.capacity(),
                workspace.storage.history_y.capacity()
            )
        );
    }

    // Diagonal quadratic ½ Σ λᵢ xᵢ² with λ from 1 to 400.
    struct Ill;

    impl ObjectiveFunction<f64> for Ill {
        fn dim(&self) -> usize {
            20
        }

        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok(x.iter()
                .enumerate()
                .map(|(i, v)| 0.5 * 400f64.powf(i as f64 / 19.0) * v * v)
                .sum())
        }

        fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            for (i, (g, v)) in output.iter_mut().zip(x).enumerate() {
                *g = 400f64.powf(i as f64 / 19.0) * v;
            }
            Ok(true)
        }
    }

    #[test]
    fn skip_policy_accepts_every_pair_of_a_convex_quadratic() {
        for action in [crate::CurvatureAction::Skip, crate::CurvatureAction::Damp] {
            let policy = crate::CurvaturePolicy::new(action, 0.2).unwrap();
            let result = Lbfgs::new(LbfgsOptions::default().with_curvature_policy(policy))
                .minimize(&Ill, &[1.0; 20])
                .unwrap();
            assert_eq!(result.termination(), OptimizeTermination::GradientSmall);
            assert!(result.iterations() < 300);
            assert!(result.x().iter().all(|v| v.abs() < 1e-5));
            assert!(matches!(
                result.method_diagnostics(),
                crate::OptimizeDiagnostics::FirstOrder(d)
                    if d.skipped_updates() == 0 && d.damped_updates() == 0
            ));
        }
    }
}
