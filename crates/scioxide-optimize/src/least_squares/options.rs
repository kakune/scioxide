//! Options for nonlinear least squares: loss, linear solver, globalization and stopping.

use scioxide_core::Scaled;
use scioxide_core::{
    Budget, GradientTolerance, InvalidInputReason, RealScalar, SciOxideError, StepTolerance,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::RankPolicy;

/// Selects the loss function `ρ` applied to each observation.
///
/// The objective is
///
/// ```text
/// f(x) = ½ Σᵢ wᵢ c² ρ(|rᵢ(x) / σᵢ|² / c²),
/// ```
///
/// where `rᵢ` is the i-th residual, `wᵢ` its weight
/// ([`LeastSquaresProblem::residual_weight`](crate::LeastSquaresProblem::residual_weight)),
/// `σᵢ` its scale
/// ([`LeastSquaresProblem::residual_scale`](crate::LeastSquaresProblem::residual_scale))
/// and `c` the loss scale set by [`LeastSquaresOptions::with_loss`]. With the
/// quadratic loss, `f` is half the weighted sum of squared residuals. The
/// complex adapter [`ComplexLeastSquares`](crate::ComplexLeastSquares) treats
/// each real and imaginary pair as one observation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LeastSquaresLoss {
    /// Squared error: `ρ(t) = t`.
    #[default]
    Quadratic,
    /// Huber loss: `ρ(t) = t` for `t ≤ 1` and `ρ(t) = 2√t − 1` for `t > 1`.
    Huber,
    /// Smooth approximation of the Huber loss: `ρ(t) = 2(√(1 + t) − 1)`.
    SoftL1,
    /// Cauchy loss: `ρ(t) = ln(1 + t)`.
    Cauchy,
}

/// Selects how the linearized least-squares subproblem is solved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LeastSquaresLinearSolver {
    /// Uses LSMR when the problem provides only Jacobian products, and otherwise
    /// column-pivoted QR, with an SVD for rank-deficient or wide Jacobians.
    #[default]
    Auto,
    /// Forms the dense Jacobian and solves with a rank-truncated thin SVD.
    DenseSvd,
    /// Uses LSMR with Jacobian and transposed-Jacobian products; never forms the
    /// Jacobian. The problem must provide both products.
    Lsmr,
}

/// Selects how [`LeastSquares`](crate::LeastSquares) safeguards the
/// Gauss–Newton step.
///
/// Both methods use the same linearization, loss and stopping tests. The
/// default is Levenberg–Marquardt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LeastSquaresGlobalization<S: RealScalar> {
    /// Undamped Gauss–Newton steps with backtracking. The step length `α`
    /// starts at 1 and is halved until the actual reduction of the cost is at
    /// least `1e-4` times the reduction predicted by the Gauss–Newton model,
    /// `−rᵀ(αJp) − ½‖αJp‖²` for the step `p`. Converges fastest on
    /// well-conditioned problems with small residuals at the solution.
    GaussNewton {
        /// Maximum backtracking trials per iteration.
        max_backtracking_steps: Budget,
    },
    /// Trust-region Levenberg–Marquardt (Moré, 1978). A trial is accepted when
    /// the ratio of actual to predicted reduction exceeds `1e-4`. The damping
    /// parameter is found by a safeguarded search that targets the trust radius
    /// to within 10% in at most ten iterations, reusing the Jacobian factors.
    LevenbergMarquardt {
        /// Initial guess for the damping parameter; nonnegative and finite. Zero
        /// selects it automatically.
        initial_damping: S,
    },
}

impl<S: RealScalar> LeastSquaresGlobalization<S> {
    /// Returns Gauss–Newton with at most 40 backtracking trials per iteration.
    pub const fn gauss_newton() -> Self {
        Self::GaussNewton {
            max_backtracking_steps: Budget::ONE.saturating_add(39),
        }
    }

    /// Returns Levenberg–Marquardt with automatic initial damping.
    pub const fn levenberg_marquardt() -> Self {
        Self::LevenbergMarquardt {
            initial_damping: S::ZERO,
        }
    }
}

impl<S: RealScalar> Default for LeastSquaresGlobalization<S> {
    fn default() -> Self {
        Self::levenberg_marquardt()
    }
}

/// Selects the convergence tests of nonlinear least squares.
///
/// The default tests the norm of the objective gradient against the gradient
/// tolerance; a small step alone is never success.
/// [`Minpack`](Self::Minpack) replaces that test with Moré's tests on the
/// relative reduction of the objective, the scaled trust radius and the
/// orthogonality of the residual to the Jacobian columns. Success under these
/// tests does not imply a small gradient. They apply only after an accepted
/// step.
///
/// ```
/// use scioxide_core::{RelativeTolerance, StepTolerance, Tolerance};
/// use scioxide_optimize::{LeastSquares, LeastSquaresOptions, LeastSquaresStopping};
///
/// let options = LeastSquaresOptions::<f64>::default()
///     .with_step_tolerance(StepTolerance::new(0.0, 1e-8)?)
///     .with_stopping(LeastSquaresStopping::Minpack {
///         objective: Tolerance::new(0.0, 1e-8)?,
///         orthogonality: RelativeTolerance::new(1e-8)?,
///     });
/// let solver = LeastSquares::new(options)?;
/// # let _ = solver;
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// J. J. Moré, "The Levenberg–Marquardt algorithm: implementation and theory",
/// in *Numerical Analysis*, Lecture Notes in Math. 630, Springer, 1978, 105–116.
#[derive(Debug, Clone, Copy, Default)]
pub enum LeastSquaresStopping<S: RealScalar> {
    /// Tests the norm of the objective gradient against the gradient tolerance.
    #[default]
    OriginalGradient,
    /// Uses Moré's objective-reduction, step and orthogonality tests. Requires
    /// a dense Jacobian; workspaces for operator-only problems reject it.
    Minpack {
        /// Threshold for the actual and predicted reduction relative to the
        /// current objective.
        objective: scioxide_core::Tolerance<S>,
        /// Largest accepted absolute cosine between the residual and a Jacobian
        /// column.
        orthogonality: scioxide_core::RelativeTolerance<S>,
    },
}

/// Configures [`LeastSquares`](crate::LeastSquares): loss, linear solver,
/// globalization, stopping tests and limits.
///
/// The defaults are the quadratic loss, Levenberg–Marquardt, an automatic
/// linear solver and at most 1000 iterations. Setters do not validate;
/// [`LeastSquares::new`](crate::LeastSquares::new) checks the loss scale and
/// the initial damping. The gradient tolerance applies to the objective gradient,
/// in the units of the objective and parameters. The step tolerance applies to the step
/// relative to the norm of the parameters. Under the default stopping tests a
/// small step ends the solve unsuccessfully. See [Termination](crate#termination)
/// for the shared stopping rules.
#[derive(Debug, Clone, Copy)]
pub struct LeastSquaresOptions<S: RealScalar> {
    pub(crate) loss: LeastSquaresLoss,
    pub(crate) loss_scale: S,
    pub(crate) rank: RankPolicy<S>,
    pub(crate) linear_solver: LeastSquaresLinearSolver,
    pub(crate) max_inner_iterations: Budget,
    pub(crate) gradient: GradientTolerance<S>,
    pub(crate) step: StepTolerance<S>,
    pub(crate) max_iterations: Budget,
    pub(crate) derivative_strategy: DerivativeStrategy,
    pub(crate) globalization: LeastSquaresGlobalization<S>,
    pub(crate) stopping: LeastSquaresStopping<S>,
    pub(crate) condition: scioxide_linalg::Estimation,
}
impl<S: RealScalar> Default for LeastSquaresOptions<S> {
    fn default() -> Self {
        Self {
            loss: LeastSquaresLoss::Quadratic,
            loss_scale: S::ONE,
            rank: RankPolicy::DimensionScaledEpsilon,
            linear_solver: LeastSquaresLinearSolver::Auto,
            max_inner_iterations: Budget::ONE.saturating_add(999),
            gradient: GradientTolerance::default(),
            step: StepTolerance::default(),
            max_iterations: Budget::ONE.saturating_add(999),
            derivative_strategy: DerivativeStrategy::default(),
            globalization: LeastSquaresGlobalization::default(),
            stopping: LeastSquaresStopping::OriginalGradient,
            condition: scioxide_linalg::Estimation::Skip,
        }
    }
}
impl<S: RealScalar> LeastSquaresOptions<S> {
    /// Sets the loss and its scale `c`, which must be positive and finite.
    #[must_use]
    pub const fn with_loss(mut self, loss: LeastSquaresLoss, scale: S) -> Self {
        self.loss = loss;
        self.loss_scale = scale;
        self
    }
    /// Sets the rank threshold for dense steps and the reported rank.
    #[must_use]
    pub const fn with_rank_policy(mut self, policy: RankPolicy<S>) -> Self {
        self.rank = policy;
        self
    }
    /// Selects the linear solver.
    #[must_use]
    pub const fn with_linear_solver(mut self, solver: LeastSquaresLinearSolver) -> Self {
        self.linear_solver = solver;
        self
    }
    /// Limits the iterations of each LSMR solve. The limits of the enclosing
    /// solve still apply.
    #[must_use]
    pub const fn with_max_inner_iterations(mut self, iterations: Budget) -> Self {
        self.max_inner_iterations = iterations;
        self
    }
    /// Sets the tolerance for the norm of the objective gradient.
    #[must_use]
    pub const fn with_gradient_tolerance(mut self, tolerance: GradientTolerance<S>) -> Self {
        self.gradient = tolerance;
        self
    }
    /// Sets the tolerance for the step relative to the parameters.
    #[must_use]
    pub const fn with_step_tolerance(mut self, tolerance: StepTolerance<S>) -> Self {
        self.step = tolerance;
        self
    }
    /// Sets the maximum number of iterations.
    #[must_use]
    pub const fn with_max_iterations(mut self, iterations: Budget) -> Self {
        self.max_iterations = iterations;
        self
    }
    /// Sets the order in which Jacobian sources (analytic, automatic, finite
    /// differences) are tried.
    #[must_use]
    pub const fn with_derivative_strategy(mut self, strategy: DerivativeStrategy) -> Self {
        self.derivative_strategy = strategy;
        self
    }
    /// Selects Gauss–Newton or Levenberg–Marquardt.
    #[must_use]
    pub const fn with_globalization(mut self, globalization: LeastSquaresGlobalization<S>) -> Self {
        self.globalization = globalization;
        self
    }
    /// Selects the convergence tests.
    #[must_use]
    pub const fn with_stopping(mut self, stopping: LeastSquaresStopping<S>) -> Self {
        self.stopping = stopping;
        self
    }
    /// Returns the convergence tests.
    pub const fn stopping(&self) -> LeastSquaresStopping<S> {
        self.stopping
    }
    /// Requests a condition estimate of the final dense Jacobian.
    ///
    /// The default skips the estimate, and the final factorization when
    /// convergence does not need it. Requesting the estimate also reports the
    /// final dense rank. LSMR solves always report their own estimate.
    #[must_use]
    pub const fn with_condition_estimation(
        mut self,
        estimation: scioxide_linalg::Estimation,
    ) -> Self {
        self.condition = estimation;
        self
    }
    /// Returns the condition-estimation setting.
    pub const fn condition_estimation(&self) -> scioxide_linalg::Estimation {
        self.condition
    }
    /// Returns the loss.
    pub const fn loss(&self) -> LeastSquaresLoss {
        self.loss
    }
    /// Returns the loss scale `c`.
    pub const fn loss_scale(&self) -> S {
        self.loss_scale
    }
    /// Returns the rank threshold for dense steps.
    pub const fn rank_policy(&self) -> RankPolicy<S> {
        self.rank
    }
    /// Returns the linear solver.
    pub const fn linear_solver(&self) -> LeastSquaresLinearSolver {
        self.linear_solver
    }
    /// Returns the iteration limit of each LSMR solve.
    pub const fn max_inner_iterations(&self) -> Budget {
        self.max_inner_iterations
    }
    /// Returns the gradient tolerance.
    pub const fn gradient_tolerance(&self) -> GradientTolerance<S> {
        self.gradient
    }
    /// Returns the step tolerance.
    pub const fn step_tolerance(&self) -> StepTolerance<S> {
        self.step
    }
    /// Returns the maximum number of iterations.
    pub const fn max_iterations(&self) -> Budget {
        self.max_iterations
    }
    /// Returns the Jacobian-source strategy.
    pub const fn derivative_strategy(&self) -> DerivativeStrategy {
        self.derivative_strategy
    }
    /// Returns the globalization.
    pub const fn globalization(&self) -> LeastSquaresGlobalization<S> {
        self.globalization
    }
}
pub(crate) fn positive<S: RealScalar>(
    value: S,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if !value.is_finite() || value <= S::ZERO {
        return Err(SciOxideError::invalid_input(
            argument,
            InvalidInputReason::OutOfRange,
            "value must be finite and positive",
        ));
    }
    Ok(())
}

/// Loss value and square-root derivative weight, in scaled arithmetic.
pub(crate) fn loss_value<S: RealScalar>(
    loss: LeastSquaresLoss,
    magnitude: Scaled<S>,
    scale: S,
) -> Result<(Scaled<S>, Scaled<S>), SciOxideError> {
    let one = Scaled::from_value(S::ONE)?;
    let half = Scaled::from_value(S::HALF)?;
    let c = Scaled::from_value(scale)?;
    let z = magnitude.div(c)?;
    let square = z.mul(z)?;
    let c2 = c.mul(c)?;
    match loss {
        LeastSquaresLoss::Quadratic => Ok((magnitude.mul(magnitude)?.mul(half)?, one)),
        LeastSquaresLoss::Huber if z.cmp_value(one) != core::cmp::Ordering::Greater => {
            Ok((magnitude.mul(magnitude)?.mul(half)?, one))
        }
        LeastSquaresLoss::Huber => Ok((c.mul(magnitude.sub(c.mul(half)?)?)?, one.div(z)?.sqrt()?)),
        LeastSquaresLoss::SoftL1 => {
            let root = one.add(square)?.sqrt()?;
            // Rationalization avoids cancellation when the residual is tiny.
            Ok((c2.mul(square.div(root.add(one)?)?)?, one.div(root)?.sqrt()?))
        }
        LeastSquaresLoss::Cauchy
            if square.cmp_value(Scaled::from_value(S::EPSILON)?) == core::cmp::Ordering::Less =>
        {
            Ok((magnitude.mul(magnitude)?.mul(half)?, one))
        }
        LeastSquaresLoss::Cauchy => {
            let denominator = one.add(square)?;
            let log = if square.cmp_value(one) != core::cmp::Ordering::Greater {
                square.round()?.ln_1p()
            } else {
                // ln(1+z²) = 2 ln(z) + ln(1+1/z²), without squaring in scalar range.
                let inverse = one.div(z)?.round()?;
                let magnitude_log = log_magnitude(magnitude)? - scale.ln();
                magnitude_log + magnitude_log + (inverse * inverse).ln_1p()
            };
            Ok((
                c2.mul(Scaled::from_value(log)?)?.mul(half)?,
                one.div(denominator)?.sqrt()?,
            ))
        }
    }
}

/// Convert only the normalized significand and exponent, never the full magnitude.
pub(crate) fn log_magnitude<S: RealScalar>(value: Scaled<S>) -> Result<S, SciOxideError> {
    let ([normalized], exponent) = scioxide_core::backend::scaled::factor_common_scale([value])?;
    Ok(normalized.round()?.ln() + S::try_from_i64_exact(exponent)? * S::LN_2)
}
