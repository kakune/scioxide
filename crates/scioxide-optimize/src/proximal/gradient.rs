//! Proximal gradient and accelerated proximal gradient.

use alloc::vec::Vec;
use core::time::Duration;
use scioxide_core::AlgebraicScalar;

use scioxide_core::backend::{checked_mul, try_filled};
use scioxide_core::{
    AttemptKind, AttemptRecord, Budget, BudgetKind, EvaluationRole, GradientTolerance,
    NonFiniteOrigin, NumericScalar, PartialResultTrust, RealScalar, Scaled, ScaledAccumulator,
    ScaledValue, SciOxideError, SolveBudgets, SolveContext, SolveScope,
};

use super::{finite, positive, shape};
use crate::OptimizeTermination;
use crate::attempt::{Entry, Shape, Shell, Staged, assign, preflight, reserve_len};

/// Defines the smooth part `f` of a composite problem.
///
/// `f` maps real or complex vectors to real values. For complex variables the
/// gradient is taken with respect to the real inner product
/// `⟨u, v⟩ = Re(uᴴv)`, so that `f(x + h) ≈ f(x) + Re(∇f(x)ᴴ h)`. Errors
/// returned by these methods stop the solve and are returned unchanged.
pub trait SmoothGradient<S: NumericScalar> {
    /// Returns the number of variables.
    fn dimension(&self) -> usize;
    /// Evaluates `f(x)`.
    ///
    /// # Errors
    ///
    /// Implementations may return any error; the solver passes it through.
    fn value(&self, x: &[S]) -> Result<S::Real, SciOxideError>;
    /// Writes `∇f(x)` to `output`.
    ///
    /// # Errors
    ///
    /// Implementations may return any error; the solver passes it through.
    fn gradient_into(&self, x: &[S], output: &mut [S]) -> Result<(), SciOxideError>;
}

/// Defines the proximal map of the nonsmooth part `g` of a composite problem.
///
/// For a proper, closed, convex `g` and a step `t > 0`, the proximal map is
///
/// ```text
/// prox_{t g}(v) = argminₓ ( g(x) + ‖x − v‖² / (2t) ).
/// ```
///
/// For the indicator function of a convex set it is the Euclidean projection
/// onto the set; for `g(x) = λ‖x‖₁` it is soft thresholding by `tλ`.
pub trait ProximalMap<S: NumericScalar> {
    /// Writes `prox_{step·g}(point)` to `output`.
    ///
    /// # Errors
    ///
    /// Implementations may return any error; the solver passes it through.
    fn apply_into(&self, point: &[S], step: S::Real, output: &mut [S])
    -> Result<(), SciOxideError>;
}

/// Borrows a composite problem: minimize `f(x) + g(x)`.
///
/// `f` is smooth with a Lipschitz-continuous gradient ([`SmoothGradient`]) and
/// `g` is convex and given by its proximal map ([`ProximalMap`]). Constraints
/// `x ∈ C` are expressed by taking `g` as the indicator function of `C`.
pub struct CompositeProblem<'a, F: ?Sized, P: ?Sized> {
    smooth: &'a F,
    prox: &'a P,
}

impl<F: ?Sized, P: ?Sized> Clone for CompositeProblem<'_, F, P> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<F: ?Sized, P: ?Sized> Copy for CompositeProblem<'_, F, P> {}

impl<'a, F: ?Sized, P: ?Sized> CompositeProblem<'a, F, P> {
    /// Creates a problem from the smooth part and the proximal map of `g`.
    pub const fn new(smooth: &'a F, prox: &'a P) -> Self {
        Self { smooth, prox }
    }
    /// Returns the smooth part `f`.
    pub const fn smooth(&self) -> &'a F {
        self.smooth
    }
    /// Returns the proximal map of `g`.
    pub const fn prox(&self) -> &'a P {
        self.prox
    }
}

/// Configures [`ProximalGradient`] and [`Fista`].
///
/// The step is `1/L`, where `L` is an upper bound on the Lipschitz constant of
/// `∇f`. With backtracking, `L` is multiplied by the growth factor until the
/// trial point `x⁺` satisfies
/// `f(x⁺) ≤ f(y) + Re(∇f(y)ᴴ(x⁺ − y)) + (L/2)‖x⁺ − y‖²`; a non-finite `f(x⁺)`
/// or an unrepresentable proposed coordinate also increases `L`. Majorization,
/// gradient mappings and restart products retain an extended exponent range.
/// The method stops successfully when the gradient mapping
/// `L‖x⁺ − y‖∞` meets the gradient tolerance. A smooth step that rounds back to
/// `y` and leaves the proximal point unchanged reports stagnation when the
/// smooth gradient still exceeds the tolerance. See also
/// [Termination](crate#termination).
#[derive(Debug, Clone, Copy)]
pub struct ProximalOptions<R: RealScalar> {
    lipschitz: R,
    growth: R,
    backtracks: Budget,
    iterations: Budget,
    tolerance: GradientTolerance<R>,
    backtracking: bool,
    restart: bool,
    time_limit: Option<Duration>,
}

impl<R: RealScalar> ProximalOptions<R> {
    /// Creates options with the fixed Lipschitz bound `lipschitz`, at most 1000
    /// iterations, the default gradient tolerance, no backtracking and FISTA
    /// gradient restart enabled. The solver constructor checks that the bound
    /// is positive and finite.
    pub fn new(lipschitz: R) -> Self {
        Self {
            lipschitz,
            growth: R::TWO,
            backtracks: Budget::ONE.saturating_add(31),
            iterations: Budget::ONE.saturating_add(999),
            tolerance: GradientTolerance::default(),
            backtracking: false,
            restart: true,
            time_limit: None,
        }
    }
    /// Enables backtracking: up to `trials` trial steps per iteration, multiplying
    /// `L` by `growth` before each retry. The solver constructor requires a finite
    /// `growth > 1`. If no trial is accepted or the next bound is unrepresentable,
    /// the solve stops with
    /// [`Stagnation`](crate::OptimizeTermination::Stagnation).
    #[must_use]
    pub const fn with_backtracking(mut self, growth: R, trials: Budget) -> Self {
        self.growth = growth;
        self.backtracks = trials;
        self.backtracking = true;
        self
    }
    /// Sets the iteration limit.
    #[must_use]
    pub const fn with_max_iterations(mut self, limit: Budget) -> Self {
        self.iterations = limit;
        self
    }
    /// Sets the tolerance on the gradient mapping `L‖x⁺ − y‖∞`.
    #[must_use]
    pub const fn with_tolerance(mut self, tolerance: GradientTolerance<R>) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Enables or disables the gradient restart of [`Fista`], which resets the
    /// momentum when `Re((y − x⁺)ᴴ(x⁺ − x)) > 0`.
    #[must_use]
    pub const fn with_restart(mut self, enabled: bool) -> Self {
        self.restart = enabled;
        self
    }
    /// Sets a time limit. A solve then needs an execution context with a clock.
    #[must_use]
    pub const fn with_time_limit(mut self, limit: Duration) -> Self {
        self.time_limit = Some(limit);
        self
    }
    /// Returns the initial Lipschitz bound `L`.
    pub const fn lipschitz(&self) -> R {
        self.lipschitz
    }
    /// Returns the backtracking growth factor and trial limit, or `None` if
    /// backtracking is disabled.
    pub const fn backtracking(&self) -> Option<(R, Budget)> {
        if self.backtracking {
            Some((self.growth, self.backtracks))
        } else {
            None
        }
    }
    /// Returns the iteration limit.
    pub const fn max_iterations(&self) -> Budget {
        self.iterations
    }
    /// Returns the gradient-mapping tolerance.
    pub const fn tolerance(&self) -> GradientTolerance<R> {
        self.tolerance
    }
    /// Returns whether [`Fista`] uses gradient restart.
    pub const fn restart(&self) -> bool {
        self.restart
    }
    /// Returns the time limit, if set.
    pub const fn time_limit(&self) -> Option<Duration> {
        self.time_limit
    }
    fn validate(&self) -> Result<(), SciOxideError> {
        positive(scioxide_core::argument!("lipschitz"), self.lipschitz)?;
        if self.backtracking && (!self.growth.is_finite() || self.growth <= R::ONE) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("growth"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "growth must be finite and exceed one",
            ));
        }
        Ok(())
    }
}

/// Describes the workspace size of a [`ProximalGradient`] or [`Fista`] solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProximalWorkspaceRequirements {
    dimension: usize,
    bytes: usize,
}

impl ProximalWorkspaceRequirements {
    fn new<S>(dimension: usize) -> Result<Self, SciOxideError> {
        let bytes = checked_mul(
            checked_mul(dimension, 7, scioxide_core::quantity!("proximal_workspace"))?,
            core::mem::size_of::<S>(),
            scioxide_core::quantity!("proximal_workspace_bytes"),
        )?;
        Ok(Self { dimension, bytes })
    }
    /// Returns the number of variables.
    pub const fn dimension(self) -> usize {
        self.dimension
    }
    /// Returns the workspace size in bytes.
    pub const fn bytes(self) -> usize {
        self.bytes
    }
}

/// Summarizes the last accepted iterate of a proximal-gradient call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProximalProgress<R> {
    iterations: usize,
    mapping_norm: R,
    trust: PartialResultTrust,
}

impl<R: Copy> ProximalProgress<R> {
    /// Returns the number of completed iterations.
    pub const fn iterations(self) -> usize {
        self.iterations
    }
    /// Returns the last gradient mapping `L‖x⁺ − y‖∞`.
    pub const fn mapping_norm(self) -> R {
        self.mapping_norm
    }
    /// Returns how far the accepted iterate has been checked.
    pub const fn trust(self) -> PartialResultTrust {
        self.trust
    }
}

#[derive(Debug)]
struct Storage<S: NumericScalar> {
    x: Vec<S>,
    previous: Vec<S>,
    y: Vec<S>,
    gradient: Vec<S>,
    trial: Vec<S>,
    direction: Vec<S>,
    completed: usize,
    mapping: S::Real,
}

impl<S: NumericScalar> Shape<Storage<S>> for ProximalWorkspaceRequirements {
    fn fits(&self, storage: &Storage<S>) -> bool {
        storage.x.len() == self.dimension
    }
    fn prepare(&self, storage: &mut Storage<S>) -> Result<(), SciOxideError> {
        let allocate = || {
            try_filled(
                self.dimension,
                S::ZERO,
                scioxide_core::quantity!("workspace_vectors"),
            )
        };
        *storage = Storage {
            x: allocate()?,
            previous: allocate()?,
            y: allocate()?,
            gradient: allocate()?,
            trial: allocate()?,
            direction: allocate()?,
            completed: 0,
            mapping: S::Real::ZERO,
        };
        Ok(())
    }
}

/// Holds the storage of a [`ProximalGradient`] or [`Fista`] solve and a record
/// of the last call.
#[derive(Debug)]
pub struct ProximalWorkspace<S: NumericScalar> {
    storage: Storage<S>,
    staged: ProximalResult<S>,
    shell: Shell<ProximalProgress<S::Real>>,
}

impl<S: NumericScalar> Default for ProximalWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: NumericScalar> ProximalWorkspace<S> {
    /// Creates an empty workspace; reserve storage with [`Self::try_reserve`].
    pub fn new() -> Self {
        Self {
            storage: Storage {
                x: Vec::new(),
                previous: Vec::new(),
                y: Vec::new(),
                gradient: Vec::new(),
                trial: Vec::new(),
                direction: Vec::new(),
                completed: 0,
                mapping: S::Real::ZERO,
            },
            staged: ProximalResult::empty(),
            shell: Shell::new(),
        }
    }
    /// Returns whether the workspace is large enough for `requirements`.
    ///
    /// # Errors
    ///
    /// Never fails; this workspace has no byte limit.
    pub fn check(
        &self,
        requirements: &ProximalWorkspaceRequirements,
    ) -> Result<scioxide_core::Capacity, SciOxideError> {
        Ok(
            if requirements.fits(&self.storage)
                && self.staged.x.capacity() >= requirements.dimension
            {
                scioxide_core::Capacity::Sufficient
            } else {
                scioxide_core::Capacity::NeedsGrowth
            },
        )
    }
    /// Reserves storage for `requirements`. On failure the existing storage is
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns `SizeOverflow` or `AllocationFailure`.
    pub fn try_reserve(
        &mut self,
        requirements: &ProximalWorkspaceRequirements,
    ) -> Result<(), SciOxideError> {
        if self.check(requirements)? == scioxide_core::Capacity::Sufficient {
            return Ok(());
        }
        if !requirements.fits(&self.storage) {
            requirements.prepare(&mut self.storage)?;
        }
        self.staged.try_reserve(requirements.dimension)
    }
    /// Returns the record of the most recent call, or `None` before the first.
    pub const fn last_attempt(&self) -> Option<&AttemptRecord<ProximalProgress<S::Real>>> {
        self.shell.last_attempt()
    }
    /// Returns the last accepted iterate of the most recent call, or `None`.
    pub fn last_accepted(&self) -> Option<&[S]> {
        self.shell
            .progress()
            .filter(|progress| progress.iterations > 0)
            .map(|_| self.storage.x.as_slice())
    }
}

/// Holds the result of a [`ProximalGradient`] or [`Fista`] solve.
///
/// Success means that the gradient mapping met its tolerance. The gradient
/// mapping measures first-order stationarity of `f + g`; it is not a bound on
/// the distance to the minimizer.
#[derive(Debug, Clone)]
pub struct ProximalResult<S: NumericScalar> {
    x: Vec<S>,
    mapping_norm: S::Real,
    lipschitz: S::Real,
    iterations: usize,
    restarts: usize,
    termination: OptimizeTermination,
    counts: scioxide_core::SolveCounts,
}

impl<S: NumericScalar> Default for ProximalResult<S> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<S: NumericScalar> ProximalResult<S> {
    /// Returns an empty result, without allocating, to pass to `minimize_into`.
    pub fn empty() -> Self {
        Self {
            x: Vec::new(),
            mapping_norm: S::Real::ZERO,
            lipschitz: S::Real::ZERO,
            iterations: 0,
            restarts: 0,
            termination: OptimizeTermination::NotStarted,
            counts: scioxide_core::SolveCounts::new(),
        }
    }
    /// Reserves storage for a result with `dimension` variables.
    ///
    /// # Errors
    ///
    /// Returns `AllocationFailure` if the storage cannot be reserved.
    pub fn try_reserve(&mut self, dimension: usize) -> Result<(), SciOxideError> {
        reserve_len(&mut self.x, dimension, scioxide_core::quantity!("result.x"))
    }
    /// Returns the last accepted point.
    pub fn x(&self) -> &[S] {
        &self.x
    }
    /// Returns the last gradient mapping `L‖x⁺ − y‖∞`, or `None` before the
    /// first complete iteration.
    pub const fn mapping_norm(&self) -> Option<S::Real> {
        if self.iterations == 0 {
            None
        } else {
            Some(self.mapping_norm)
        }
    }
    /// Returns the Lipschitz bound `L` used in the last iteration.
    pub const fn lipschitz(&self) -> S::Real {
        self.lipschitz
    }
    /// Returns the number of completed iterations.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }
    /// Returns the number of momentum restarts of [`Fista`].
    pub const fn restarts(&self) -> usize {
        self.restarts
    }
    /// Returns why the solver stopped.
    pub const fn termination(&self) -> OptimizeTermination {
        self.termination
    }
    /// Returns the iteration and evaluation counts.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns whether the gradient-mapping test passed.
    pub const fn is_success(&self) -> bool {
        self.termination.is_success()
    }

    /// Requires the method's stopping test to have passed.
    ///
    /// # Errors
    ///
    /// Returns `ExecutionStopped` for cancellation or a deadline, `DomainError`
    /// for verified infeasibility or unboundedness, `InvalidInput` before a solve,
    /// and `ConvergenceFailure` with counts for every other unsuccessful termination.
    pub fn require_success(&self) -> Result<(), SciOxideError> {
        crate::result::termination_error(
            scioxide_core::argument!("self"),
            self.termination(),
            self.counts(),
            None,
        )
        .map_or(Ok(()), Err)
    }
}

impl<S: NumericScalar> Staged for ProximalResult<S> {
    fn reset(&mut self) {
        self.x.clear();
        self.iterations = 0;
        self.restarts = 0;
        self.termination = OptimizeTermination::NotStarted;
        self.counts = scioxide_core::SolveCounts::new();
    }
    fn publish(&mut self, source: &Self) -> Result<(), SciOxideError> {
        self.try_reserve(source.x.len())?;
        assign(&mut self.x, &source.x);
        self.mapping_norm = source.mapping_norm;
        self.lipschitz = source.lipschitz;
        self.iterations = source.iterations;
        self.restarts = source.restarts;
        self.termination = source.termination;
        self.counts = source.counts;
        Ok(())
    }
    fn succeeded(&self) -> bool {
        self.is_success()
    }
}

/// Minimizes a composite function `f(x) + g(x)` by the proximal gradient method.
///
/// Each iteration takes a gradient step on the smooth part `f` and applies the
/// proximal map of `g`:
///
/// ```text
/// xₖ₊₁ = prox_{g/L}(xₖ − ∇f(xₖ)/L),
/// ```
///
/// where `L` bounds the Lipschitz constant of `∇f`, either fixed or found by
/// backtracking ([`ProximalOptions`]). Each iteration costs one value and one
/// gradient of `f` and one proximal map, plus one value of `f` per trial when
/// backtracking. For convex `f` and `g` the objective error decreases as
/// `O(1/k)`; [`Fista`] improves this to `O(1/k²)`. Variables may be real or
/// complex.
///
/// # Errors
///
/// - `InvalidInput` from [`Self::new`] if `L` is not positive and finite, or
///   the backtracking growth factor is not finite and greater than one.
/// - `DimensionMismatch` if `x0` does not match the problem dimension.
/// - `NonFinite` if `x0` is not finite, or if `f`, `∇f` or the proximal map
///   returns NaN or infinity. A non-finite `f` at a backtracking trial only
///   increases `L`.
/// - `CapabilityUnavailable` if a time limit is set and the context has no
///   clock.
/// - Any error returned by a callback.
///
/// # Examples
///
/// ```
/// use scioxide_core::SciOxideError;
/// use scioxide_optimize::{
///     CompositeProblem, ProximalGradient, ProximalMap, ProximalOptions, SmoothGradient,
/// };
///
/// // f(x) = ½(x − 3)² and g is the indicator of [0, 1]; the minimizer is x = 1.
/// struct Quadratic;
/// impl SmoothGradient<f64> for Quadratic {
///     fn dimension(&self) -> usize {
///         1
///     }
///     fn value(&self, x: &[f64]) -> Result<f64, SciOxideError> {
///         Ok(0.5 * (x[0] - 3.0).powi(2))
///     }
///     fn gradient_into(&self, x: &[f64], g: &mut [f64]) -> Result<(), SciOxideError> {
///         g[0] = x[0] - 3.0;
///         Ok(())
///     }
/// }
/// struct UnitInterval;
/// impl ProximalMap<f64> for UnitInterval {
///     fn apply_into(&self, p: &[f64], _: f64, out: &mut [f64]) -> Result<(), SciOxideError> {
///         out[0] = p[0].clamp(0.0, 1.0);
///         Ok(())
///     }
/// }
///
/// let solver = ProximalGradient::new(ProximalOptions::new(1.0))?;
/// let result = solver.minimize(&CompositeProblem::new(&Quadratic, &UnitInterval), &[0.0])?;
/// assert!(result.is_success());
/// assert!((result.x()[0] - 1.0).abs() < 1e-12);
/// # Ok::<(), SciOxideError>(())
/// ```
///
/// # References
///
/// N. Parikh and S. Boyd, "Proximal algorithms", *Foundations and Trends in
/// Optimization* 1(3), 127–239, 2014. <https://doi.org/10.1561/2400000003>
#[derive(Debug, Clone)]
pub struct ProximalGradient<R: RealScalar> {
    options: ProximalOptions<R>,
}
/// Minimizes a composite function `f(x) + g(x)` by FISTA, the accelerated
/// proximal gradient method.
///
/// FISTA applies the proximal gradient step of [`ProximalGradient`] at an
/// extrapolated point `yₖ = xₖ + ((tₖ₋₁ − 1)/tₖ)(xₖ − xₖ₋₁)`, with
/// `tₖ = (1 + √(1 + 4tₖ₋₁²))/2`. For convex `f` and `g` the objective error
/// decreases as `O(1/k²)`. With gradient restart (the default), the momentum is
/// reset whenever `Re((yₖ − xₖ₊₁)ᴴ(xₖ₊₁ − xₖ)) > 0`, which restores fast
/// convergence on strongly convex problems. The cost per iteration, the
/// methods and the errors are those of [`ProximalGradient`].
///
/// # References
///
/// - A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding
///   algorithm for linear inverse problems", *SIAM J. Imaging Sci.* 2(1),
///   183–202, 2009. <https://doi.org/10.1137/080716542>
/// - B. O'Donoghue and E. Candès, "Adaptive restart for accelerated gradient
///   schemes", *Found. Comput. Math.* 15(3), 715–732, 2015.
///   <https://doi.org/10.1007/s10208-013-9150-3>
#[derive(Debug, Clone)]
pub struct Fista<R: RealScalar> {
    options: ProximalOptions<R>,
}

macro_rules! solver {
    ($name:ident, $accelerated:expr) => {
        impl<R: RealScalar> $name<R> {
            /// Creates a solver with checked options.
            ///
            /// # Errors
            ///
            /// Returns `InvalidInput` if the Lipschitz bound is not positive and
            /// finite, or if backtracking is enabled with a growth factor that is
            /// not finite or not greater than one.
            pub fn new(options: ProximalOptions<R>) -> Result<Self, SciOxideError> {
                options.validate()?;
                Ok(Self { options })
            }
            /// Returns the options.
            pub const fn options(&self) -> &ProximalOptions<R> {
                &self.options
            }
            /// Returns the workspace size for `dimension` variables.
            ///
            /// # Errors
            ///
            /// Returns `SizeOverflow` if the size does not fit in `usize`.
            pub fn workspace_requirements<S: NumericScalar<Real = R>>(
                &self,
                dimension: usize,
            ) -> Result<ProximalWorkspaceRequirements, SciOxideError> {
                ProximalWorkspaceRequirements::new::<S>(dimension)
            }
            /// Allocates a workspace for `dimension` variables.
            ///
            /// # Errors
            ///
            /// Returns `SizeOverflow` or `AllocationFailure`.
            pub fn workspace<S: NumericScalar<Real = R>>(
                &self,
                dimension: usize,
            ) -> Result<ProximalWorkspace<S>, SciOxideError> {
                let mut workspace = ProximalWorkspace::new();
                workspace.try_reserve(&self.workspace_requirements::<S>(dimension)?)?;
                Ok(workspace)
            }
            /// Minimizes from `x0` on the calling thread and returns a new result.
            ///
            /// # Errors
            ///
            /// As [`Self::minimize_into`].
            pub fn minimize<S, F, P>(
                &self,
                problem: &CompositeProblem<'_, F, P>,
                x0: &[S],
            ) -> Result<ProximalResult<S>, SciOxideError>
            where
                S: NumericScalar<Real = R>,
                F: SmoothGradient<S> + ?Sized,
                P: ProximalMap<S> + ?Sized,
            {
                let mut workspace = self.workspace(x0.len())?;
                let mut output = ProximalResult::empty();
                self.minimize_into(
                    problem,
                    x0,
                    &mut output,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )?;
                Ok(output)
            }
            /// Minimizes from `x0` and writes the result to `output`.
            ///
            /// `scope` is an execution context or a parent solve context; see
            /// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
            /// An iteration or evaluation limit, cancellation or a deadline ends
            /// the solve with an unsuccessful result. On error, `output` is
            /// unchanged.
            ///
            /// # Errors
            ///
            /// See the type-level `# Errors` section. Also returns
            /// `AllocationFailure` if the workspace must grow and cannot.
            pub fn minimize_into<'scope, 'capability, S, F, P>(
                &self,
                problem: &CompositeProblem<'_, F, P>,
                x0: &[S],
                output: &mut ProximalResult<S>,
                workspace: &mut ProximalWorkspace<S>,
                scope: impl Into<SolveScope<'scope, 'capability>>,
            ) -> Result<(), SciOxideError>
            where
                S: NumericScalar<Real = R>,
                F: SmoothGradient<S> + ?Sized,
                P: ProximalMap<S> + ?Sized,
            {
                workspace.shell.reject(AttemptKind::Call);
                let scope = scope.into();
                shape(problem.smooth.dimension(), x0.len())?;
                finite(x0, NonFiniteOrigin::Input)?;
                workspace.try_reserve(&self.workspace_requirements::<S>(x0.len())?)?;
                preflight(&scope, self.options.time_limit)?;
                let entry = Entry::Call {
                    identity: crate::context::local_identity(
                        scioxide_core::algorithm_id!("optimize", stringify!($name), 1),
                        x0.len(),
                        &self.options,
                    ),
                    budgets: SolveBudgets::unbounded()
                        .with_limit(BudgetKind::Iteration, self.options.iterations),
                };
                crate::attempt::run(
                    &mut workspace.shell,
                    &mut workspace.storage,
                    &mut workspace.staged,
                    output,
                    scope,
                    entry,
                    |context, storage, staged| {
                        run(
                            problem,
                            x0,
                            &self.options,
                            $accelerated,
                            staged,
                            storage,
                            context,
                        )
                    },
                    progress,
                )
            }
        }
    };
}
solver!(ProximalGradient, false);
solver!(Fista, true);

fn progress<S: NumericScalar>(
    storage: &Storage<S>,
    succeeded: bool,
) -> Option<ProximalProgress<S::Real>> {
    Some(ProximalProgress {
        iterations: storage.completed,
        mapping_norm: storage.mapping,
        trust: crate::attempt::trust(succeeded),
    })
}

fn publish<S: NumericScalar>(
    output: &mut ProximalResult<S>,
    w: &mut Storage<S>,
    (mapping, lipschitz, iterations, restarts): (S::Real, S::Real, usize, usize),
    termination: OptimizeTermination,
    context: &SolveContext<'_, '_>,
) {
    w.completed = iterations;
    w.mapping = mapping;
    assign(&mut output.x, &w.x);
    output.mapping_norm = mapping;
    output.lipschitz = lipschitz;
    output.iterations = iterations;
    output.restarts = restarts;
    output.termination = termination;
    output.counts = context.counts();
}

fn run<S, F, P>(
    problem: &CompositeProblem<'_, F, P>,
    x0: &[S],
    options: &ProximalOptions<S::Real>,
    accelerated: bool,
    output: &mut ProximalResult<S>,
    w: &mut Storage<S>,
    context: &mut SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: NumericScalar,
    F: SmoothGradient<S> + ?Sized,
    P: ProximalMap<S> + ?Sized,
{
    if let Some(limit) = options.time_limit {
        context.limit_time(limit)?;
    }
    let (smooth, prox) = (problem.smooth, problem.prox);
    w.completed = 0;
    w.mapping = S::Real::ZERO;
    w.x.copy_from_slice(x0);
    w.y.copy_from_slice(x0);
    let mut bound = options.lipschitz;
    let mut momentum = S::Real::ONE;
    let mut mapping = S::Real::ZERO;
    let mut restarts = 0;
    let mut completed = 0;
    let mut termination = OptimizeTermination::IterationLimit;
    'iterations: for _ in 0..options.iterations.get() {
        if let Some(stop) = super::stop(context)? {
            termination = stop;
            break;
        }
        if !context.record_iteration()?.is_granted() {
            publish(
                output,
                w,
                (mapping, bound, completed, restarts),
                OptimizeTermination::IterationLimit,
                context,
            );
            return Ok(());
        }
        if !context
            .record_provider(EvaluationRole::Objective)?
            .is_granted()
        {
            publish(
                output,
                w,
                (mapping, bound, completed, restarts),
                OptimizeTermination::EvaluationLimit,
                context,
            );
            return Ok(());
        }
        let base = smooth.value(&w.y)?;
        if !base.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Callback,
                "non-finite smooth value",
            ));
        }
        if let Some(stop) = super::stop(context)? {
            termination = stop;
            break 'iterations;
        }
        if !context
            .record_provider(EvaluationRole::Gradient)?
            .is_granted()
        {
            publish(
                output,
                w,
                (mapping, bound, completed, restarts),
                OptimizeTermination::EvaluationLimit,
                context,
            );
            return Ok(());
        }
        smooth.gradient_into(&w.y, &mut w.gradient)?;
        finite(&w.gradient, NonFiniteOrigin::Callback)?;
        if let Some(stop) = super::stop(context)? {
            termination = stop;
            break 'iterations;
        }
        let mut accepted = false;
        for trial_index in 0..if options.backtracking {
            options.backtracks.get()
        } else {
            1
        } {
            if trial_index > 0 {
                let next = bound * options.growth;
                if !next.is_finite() || next <= bound {
                    break;
                }
                bound = next;
            }
            let step = S::Real::ONE / bound;
            if !step.is_finite() && options.backtracking {
                continue;
            }
            positive(scioxide_core::argument!("self"), step)?;
            let mut representable = true;
            for (dest, (&y, &g)) in w.direction.iter_mut().zip(w.y.iter().zip(&w.gradient)) {
                let candidate =
                    scioxide_core::checked_mul_add(-step, g.real_component(), y.real_component())
                        .and_then(|real| {
                            scioxide_core::checked_mul_add(
                                -step,
                                g.imaginary_component(),
                                y.imaginary_component(),
                            )
                            .map(|imaginary| S::from_components(real, imaginary))
                        });
                match candidate {
                    Ok(value) => *dest = value,
                    Err(error)
                        if options.backtracking
                            && matches!(
                                error.view(),
                                scioxide_core::SciOxideErrorView::NumericalInstability {
                                    issue: scioxide_core::NumericalIssue::Overflow,
                                    ..
                                }
                            ) =>
                    {
                        representable = false;
                        break;
                    }
                    Err(error) => return Err(error),
                }
            }
            if !representable {
                continue;
            }
            if !context
                .record_provider(EvaluationRole::Function)?
                .is_granted()
            {
                publish(
                    output,
                    w,
                    (mapping, bound, completed, restarts),
                    OptimizeTermination::EvaluationLimit,
                    context,
                );
                return Ok(());
            }
            prox.apply_into(&w.direction, step, &mut w.trial)?;
            finite(&w.trial, NonFiniteOrigin::Callback)?;
            if let Some(stop) = super::stop(context)? {
                termination = stop;
                break 'iterations;
            }
            if !options.backtracking {
                accepted = true;
                break;
            }
            if !context
                .record_provider(EvaluationRole::Objective)?
                .is_granted()
            {
                publish(
                    output,
                    w,
                    (mapping, bound, completed, restarts),
                    OptimizeTermination::EvaluationLimit,
                    context,
                );
                return Ok(());
            }
            let trial_value = smooth.value(&w.trial)?;
            if let Some(stop) = super::stop(context)? {
                termination = stop;
                break 'iterations;
            }
            // A non-finite trial value fails sufficient decrease: shorten the step.
            if !trial_value.is_finite() {
                continue;
            }
            let mut upper = ScaledAccumulator::from_value(base)?;
            let half_bound = Scaled::from_value(bound)?.mul(Scaled::HALF)?;
            for ((&trial, &y), &g) in w.trial.iter().zip(&w.y).zip(&w.gradient) {
                let delta = ScaledValue::from_value(trial)?.sub(ScaledValue::from_value(y)?)?;
                for (difference, gradient) in [
                    (delta.real(), g.real_component()),
                    (delta.imaginary(), g.imaginary_component()),
                ] {
                    upper.add_product(Scaled::from_value(gradient)?, difference)?;
                    upper.add_scaled(half_bound.mul(difference.square()?)?)?;
                }
            }
            if Scaled::from_value(trial_value)?
                .cmp_value(upper.total()?)
                .is_le()
            {
                accepted = true;
                break;
            }
        }
        if !accepted {
            publish(
                output,
                w,
                (mapping, bound, completed, restarts),
                OptimizeTermination::Stagnation,
                context,
            );
            return Ok(());
        }
        // A nonzero smooth step lost to coordinate rounding cannot establish
        // a fixed point of the exact proximal map. Keep the accepted state.
        if w.direction == w.y && w.trial == w.y {
            let mut resolved = true;
            for gradient in &w.gradient {
                resolved &= options
                    .tolerance
                    .is_satisfied(gradient.abs(), S::Real::ONE)?;
            }
            if !resolved {
                termination = OptimizeTermination::Stagnation;
                break;
            }
        }
        mapping = S::Real::ZERO;
        for (&trial, &y) in w.trial.iter().zip(&w.y) {
            let delta = ScaledValue::from_value(trial)?.sub(ScaledValue::from_value(y)?)?;
            let magnitude = Scaled::from_parts(delta.scaled_magnitude(), delta.scale_exponent())?
                .mul(Scaled::from_value(bound)?)?
                .round()?;
            if magnitude > mapping {
                mapping = magnitude;
            }
        }
        positive(scioxide_core::argument!("self"), bound)?;
        w.previous.copy_from_slice(&w.x);
        w.x.copy_from_slice(&w.trial);
        completed += 1;
        (w.completed, w.mapping) = (completed, mapping);
        context.commit_accepted(PartialResultTrust::Admissible)?;
        if options.tolerance.is_satisfied(mapping, S::Real::ONE)? {
            context.assess_accepted(PartialResultTrust::ResidualChecked)?;
            publish(
                output,
                w,
                (mapping, bound, completed, restarts),
                OptimizeTermination::Converged,
                context,
            );
            return Ok(());
        }
        if accelerated {
            let two = S::Real::TWO;
            let next = (S::Real::ONE + (S::Real::ONE + (two + two) * momentum * momentum).sqrt())
                * S::Real::HALF;
            let factor = (momentum - S::Real::ONE) / next;
            if !next.is_finite() || !factor.is_finite() {
                return Err(SciOxideError::non_finite(
                    NonFiniteOrigin::Intermediate,
                    "unrepresentable FISTA momentum",
                ));
            }
            let mut restart = ScaledAccumulator::ZERO;
            if options.restart {
                for ((&y, &x), &old) in w.y.iter().zip(&w.x).zip(&w.previous) {
                    let left = ScaledValue::from_value(y)?.sub(ScaledValue::from_value(x)?)?;
                    let right = ScaledValue::from_value(x)?.sub(ScaledValue::from_value(old)?)?;
                    restart.add_product(left.real(), right.real())?;
                    restart.add_product(left.imaginary(), right.imaginary())?;
                }
            }
            if options.restart && restart.total()?.cmp_value(Scaled::ZERO).is_gt() {
                w.y.copy_from_slice(&w.x);
                momentum = S::Real::ONE;
                restarts += 1;
            } else {
                for ((y, &x), &old) in w.y.iter_mut().zip(&w.x).zip(&w.previous) {
                    let current = ScaledValue::from_value(x)?;
                    *y = current
                        .add(
                            current
                                .sub(ScaledValue::from_value(old)?)?
                                .mul_real(Scaled::from_value(factor)?)?,
                        )?
                        .try_value()?;
                }
                momentum = next;
            }
        } else {
            w.y.copy_from_slice(&w.x);
        }
    }
    publish(
        output,
        w,
        (mapping, bound, completed, restarts),
        termination,
        context,
    );
    Ok(())
}
