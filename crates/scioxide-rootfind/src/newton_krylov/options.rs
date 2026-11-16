//! Options for Newton–Krylov solves.
use super::*;

/// Selects the forcing term `η`, the relative tolerance of each inner linear solve.
///
/// Each Newton step solves `J d = −F(x)` until `‖F(x) + J d‖ ≤ η ‖F(x)‖`. A
/// small `η` gives a step close to the exact Newton step at a higher GMRES cost.
///
/// The adaptive choice is choice 1 of Eisenstat and Walker. After a step `s`
/// it sets
///
/// ```text
/// η = |‖F(x + s)‖ − ‖F(x) + J s‖| / ‖F(x)‖,
/// ```
///
/// raises it to `η_prev^φ` with `φ = (1 + √5)/2` when that value exceeds 0.1, and
/// clamps it to `[minimum, maximum]`. Norms are taken after the
/// [`NonlinearScaling`] divisors. With forward-difference Jacobian products,
/// every `η` is at least `√ε`.
///
/// The default is `EisenstatWalker { initial: 0.5, minimum: ε, maximum: 0.9 }`.
///
/// # References
///
/// S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an inexact
/// Newton method", *SIAM J. Sci. Comput.* 17(1), 1996.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NewtonKrylovForcing<R: RealScalar> {
    /// Uses the same `η` for every step, strictly between zero and one.
    Constant(R),
    /// Uses Eisenstat–Walker choice 1. All three values lie strictly between zero
    /// and one, with `minimum ≤ initial ≤ maximum`.
    EisenstatWalker {
        /// Forcing term of the first step.
        initial: R,
        /// Smallest forcing term.
        minimum: R,
        /// Largest forcing term.
        maximum: R,
    },
}
impl<R: RealScalar> Default for NewtonKrylovForcing<R> {
    fn default() -> Self {
        Self::EisenstatWalker {
            initial: R::HALF,
            minimum: R::EPSILON,
            maximum: R::ONE - tenth::<R>(),
        }
    }
}
impl<R: RealScalar> NewtonKrylovForcing<R> {
    /// Validate the forcing bounds.
    pub(crate) fn validate(self) -> Result<(), SciOxideError> {
        let fraction = |v: R| v.is_finite() && v > R::ZERO && v < R::ONE;
        match self {
            Self::Constant(v) if fraction(v) => Ok(()),
            Self::EisenstatWalker {
                initial,
                minimum,
                maximum,
            } if fraction(initial)
                && fraction(minimum)
                && fraction(maximum)
                && minimum <= initial
                && initial <= maximum =>
            {
                Ok(())
            }
            _ => Err(invalid(
                scioxide_core::argument!("options.forcing"),
                InvalidInputReason::Inconsistent,
                "invalid forcing bounds",
            )),
        }
    }
    pub(crate) fn initial(self) -> R {
        match self {
            Self::Constant(v) => v,
            Self::EisenstatWalker { initial, .. } => initial,
        }
    }
    /// Largest relative tolerance this policy may request.
    pub(crate) fn maximum(self) -> R {
        match self {
            Self::Constant(v) => v,
            Self::EisenstatWalker { maximum, .. } => maximum,
        }
    }
    /// Next request after a step from residual norm `old_norm` to `new_norm` whose
    /// linear model predicted `model_norm`.
    pub(crate) fn next(self, previous: R, old_norm: R, new_norm: R, model_norm: R) -> R {
        match self {
            Self::Constant(v) => v,
            Self::EisenstatWalker {
                minimum, maximum, ..
            } => {
                let golden = (R::ONE + (R::TWO + R::TWO + R::ONE).sqrt()) * R::HALF;
                let candidate = (new_norm - model_norm).abs() / old_norm;
                let safeguard = previous.powf(golden);
                let candidate = if safeguard > tenth::<R>() {
                    candidate.max(safeguard)
                } else {
                    candidate
                };
                candidate.max(minimum).min(maximum)
            }
        }
    }
}

/// Configures a [`NewtonKrylovSolver`].
///
/// The solve succeeds with [`RootTermination::ResidualTolerance`] when `‖F(x)‖₂`
/// passes the residual tolerance. With [`NonlinearScaling`], the norm is taken
/// after the row divisors. The relative part of the tolerance is measured against
/// the norm at the initial point, or against
/// [`with_residual_scale`](Self::with_residual_scale). A small step whose residual
/// did not decrease ends the solve with [`RootTermination::Stagnation`]; a small
/// step that reduces the residual does not stop the solve.
///
/// | Control | Default |
/// |---|---|
/// | residual tolerance | absolute 0, relative `√ε` |
/// | step tolerance | absolute and relative `√ε` |
/// | Newton iterations | 100 |
/// | GMRES iterations per Newton step, restart length | 200, 30 |
/// | forcing | Eisenstat–Walker, `η` from 0.5 in `[ε, 0.9]` |
/// | Armijo constant, smallest relative step | `10⁻⁴`, `ε^(2/3)` |
/// | nonmonotone weight | 0.85 |
/// | history of inner solves | off |
///
/// [`NewtonKrylovSolver::new`] validates these controls.
#[derive(Debug, Clone, PartialEq)]
pub struct NewtonKrylovOptions<R: RealScalar> {
    pub(crate) residual: ResidualTolerance<R>,
    pub(crate) step: StepTolerance<R>,
    pub(crate) scale: Option<R>,
    pub(crate) iterations: Budget,
    pub(crate) inner_iterations: Budget,
    pub(crate) restart: Budget,
    pub(crate) forcing: NewtonKrylovForcing<R>,
    pub(crate) armijo: R,
    pub(crate) minimum_step: R,
    pub(crate) nonmonotone: R,
    pub(crate) history: bool,
    pub(crate) budgets: SolveBudgets,
    pub(crate) policy: scioxide_core::ExecutionPolicy,
    pub(crate) byte_limit: usize,
}
impl<R: RealScalar> Default for NewtonKrylovOptions<R> {
    fn default() -> Self {
        Self {
            residual: crate::termination::relative_tolerance(),
            step: StepTolerance::from_tolerance(Tolerance::from_sqrt_epsilon()),
            scale: None,
            iterations: Budget::ONE.saturating_add(99),
            inner_iterations: Budget::ONE.saturating_add(199),
            restart: Budget::ONE.saturating_add(29),
            forcing: NewtonKrylovForcing::default(),
            armijo: tenth::<R>().powi(4),
            minimum_step: R::CBRT_EPSILON * R::CBRT_EPSILON,
            nonmonotone: (R::TWO.powi(4) + R::ONE) / (R::TWO.powi(4) + R::TWO + R::TWO),
            history: false,
            budgets: SolveBudgets::unbounded(),
            policy: scioxide_core::ExecutionPolicy::Sequential,
            byte_limit: usize::MAX,
        }
    }
}
impl<R: RealScalar> NewtonKrylovOptions<R> {
    /// Sets the work budgets of each solve. The Newton iteration limit applies as
    /// well, and so do the limits of an enclosing solve
    /// (see [Solve accounting](scioxide_core#solve-accounting)).
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }
    /// Returns the work budgets of each solve.
    pub const fn budgets(&self) -> SolveBudgets {
        self.budgets
    }
    /// Sets the execution policy of the GMRES vector operations (default
    /// sequential).
    pub const fn with_execution_policy(mut self, policy: scioxide_core::ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
    /// Returns the execution policy.
    pub const fn execution_policy(&self) -> scioxide_core::ExecutionPolicy {
        self.policy
    }
    /// Sets the largest size in bytes of the workspace and of a result resized by
    /// [`NewtonKrylovResult::try_resize`] (default unlimited).
    pub const fn with_workspace_byte_limit(mut self, bytes: usize) -> Self {
        self.byte_limit = bytes;
        self
    }
    /// Returns the largest workspace size in bytes.
    pub const fn workspace_byte_limit(&self) -> usize {
        self.byte_limit
    }
    pub(crate) fn validate(&self) -> Result<(), SciOxideError> {
        self.forcing.validate()?;
        let invalid_field = if self.scale.is_some_and(|v| !v.is_finite() || v <= R::ZERO) {
            Some((
                scioxide_core::argument!("options.residual_scale"),
                "residual scale must be positive and finite",
            ))
        } else if !self.armijo.is_finite() || self.armijo <= R::ZERO || self.armijo >= R::ONE {
            Some((
                scioxide_core::argument!("options.armijo"),
                "Armijo decrease must lie strictly between zero and one",
            ))
        } else if !self.minimum_step.is_finite()
            || self.minimum_step <= R::ZERO
            || self.minimum_step > R::ONE
        {
            Some((
                scioxide_core::argument!("options.minimum_step"),
                "minimum step must lie in (0, 1]",
            ))
        } else if !self.nonmonotone.is_finite()
            || self.nonmonotone < R::ZERO
            || self.nonmonotone > R::ONE
        {
            Some((
                scioxide_core::argument!("options.nonmonotone"),
                "nonmonotone weight must lie in [0, 1]",
            ))
        } else {
            None
        };
        if let Some((argument, detail)) = invalid_field {
            return Err(invalid(argument, InvalidInputReason::Inconsistent, detail));
        }
        Ok(())
    }

    /// Sets the residual tolerance, in the units of `F` (after the row divisors, if
    /// any).
    pub fn with_residual_tolerance(mut self, value: ResidualTolerance<R>) -> Self {
        self.residual = value;
        self
    }
    /// Sets the step tolerance used to detect stagnation. Without scaling, a step
    /// is small when `‖Δx‖₂ ≤ atol + rtol · max(1, ‖x‖₂)`; with scaling, the step
    /// is divided by the variable divisors and compared at unit scale.
    pub fn with_step_tolerance(mut self, value: StepTolerance<R>) -> Self {
        self.step = value;
        self
    }
    /// Sets the reference norm of the relative residual tolerance, in place of the
    /// initial residual norm. It must be positive and finite.
    pub fn with_residual_scale(mut self, value: R) -> Self {
        self.scale = Some(value);
        self
    }
    /// Sets the limit on Newton iterations (default 100). GMRES iterations do not
    /// count toward it.
    pub fn with_max_iterations(mut self, value: Budget) -> Self {
        self.iterations = value;
        self
    }
    /// Sets the GMRES iteration limit of each Newton step and the restart length
    /// `m` (defaults 200 and 30). The workspace size grows linearly with `m`.
    pub fn with_krylov(mut self, iterations: Budget, restart: Budget) -> Self {
        self.inner_iterations = iterations;
        self.restart = restart;
        self
    }
    /// Sets the forcing term of the inner linear solves.
    pub fn with_forcing(mut self, value: NewtonKrylovForcing<R>) -> Self {
        self.forcing = value;
        self
    }
    /// Sets the Armijo constant `c ∈ (0, 1)` (default `10⁻⁴`) and the smallest
    /// relative step `minimum ∈ (0, 1]` (default `ε^(2/3)`) of the line search.
    ///
    /// The line search tries the full Newton step `α = 1` first and shortens it by
    /// safeguarded quadratic and cubic interpolation of the merit `½‖F‖²`. It
    /// gives up with [`RootTermination::GlobalizationFailure`] once
    /// `α · maxᵢ |dᵢ| / max(1, |xᵢ|)` falls below `minimum`. A trial whose residual
    /// is not finite, or whose evaluation returns `DomainError` with reason
    /// `OutsideDomain`, halves the step instead.
    ///
    /// # References
    ///
    /// J. E. Dennis and R. B. Schnabel, *Numerical Methods for Unconstrained
    /// Optimization and Nonlinear Equations*, SIAM, 1996, Algorithm A6.3.1.
    pub fn with_backtracking(mut self, c: R, minimum: R) -> Self {
        self.armijo = c;
        self.minimum_step = minimum;
        self
    }
    /// Sets the weight `w ∈ [0, 1]` of the nonmonotone line search (default 0.85).
    ///
    /// A trial step must pass the Armijo test against a reference `C`, a weighted
    /// average of the merits `½‖F‖²` of the accepted points:
    ///
    /// ```text
    /// Q_new = w Q + 1,   C_new = (w Q C + ½‖F(x_new)‖²) / Q_new,
    /// ```
    ///
    /// starting from `Q = 1` and the initial merit. With `w = 0` the test is the
    /// monotone Armijo test. A positive weight lets the iteration accept a full
    /// Newton step that temporarily increases the residual, which helps along
    /// curved valleys.
    ///
    /// # References
    ///
    /// H. Zhang and W. W. Hager, "A nonmonotone line search technique and its
    /// application to unconstrained optimization", *SIAM J. Optim.* 14(4), 2004.
    pub fn with_nonmonotone(mut self, weight: R) -> Self {
        self.nonmonotone = weight;
        self
    }
    /// Records one [`NewtonKrylovObservation`] per inner solve in
    /// [`NewtonKrylovResult::history`] (default off). The result then reserves
    /// space for one observation per Newton iteration.
    pub fn with_history(mut self, enabled: bool) -> Self {
        self.history = enabled;
        self
    }
    pub(crate) fn linear<T: NumericScalar<Real = R>>(
        &self,
        eta: R,
    ) -> Result<GmresSolver<T>, SciOxideError> {
        Ok(
            GmresSolver::new(ResidualTolerance::new(R::ZERO, eta)?, self.inner_iterations)
                .with_restart(self.restart)
                .with_execution_options(
                    scioxide_linalg::LinalgExecutionOptions::new()
                        .with_execution_policy(self.policy),
                ),
        )
    }
}
