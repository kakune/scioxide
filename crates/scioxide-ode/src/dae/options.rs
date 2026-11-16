//! Options of the residual DAE integrator: error and residual tolerances,
//! initialization, iteration limits, the initial step and budgets.

use scioxide_core::{
    Budget, BudgetKind, ErrorTolerance, NormSelection, RealScalar, ResidualTolerance,
    SciOxideError, SolveBudgets, Tolerance,
};

use super::DaePartition;
use crate::safety::invalid;
use scioxide_differentiate::DerivativeStrategy;

/// Selects how the integrator treats the initial pair `(y₀, ẏ₀)`.
///
/// A pair is consistent when the differential and algebraic residual blocks of
/// `F(t₀, y₀, ẏ₀)` meet their tolerances. In both modes the integrator also factors the
/// consistency Jacobian once, so a singular one (a system that is not index one at `t₀`)
/// is reported before the first step.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DaeInitialization {
    /// Accepts the supplied pair unchanged if it is consistent (the default); otherwise
    /// the integration fails with `InvalidInput` (`Inconsistent`).
    #[default]
    Check,
    /// Keeps the differential states fixed and solves for the algebraic states and the
    /// differential derivatives with Newton's method; failure to converge is a
    /// `ConvergenceFailure`.
    Project,
}

/// Configures [`DaeBdf`](crate::DaeBdf): numerical controls, the initial step and
/// budgets.
///
/// Setters do not fail. [`DaeBdf::advance`](crate::DaeBdf::advance) checks the whole
/// configuration against the partition in preflight.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct DaeOptions<S: RealScalar> {
    numerics: DaeNumericalOptions<S>,
    initial_step: Option<S>,
    max_steps: Option<Budget>,
    max_evaluations: Option<Budget>,
}

/// Configures the error control, residual tolerances, initialization and Newton
/// iteration of residual DAE integration.
///
/// [`DaeOptions`] and [`DaeAdjointOptions`](crate::DaeAdjointOptions) both use this
/// value; the consuming solver checks it against the partition.
///
/// The integrator controls two kinds of error separately. The local error estimates of
/// the differential and algebraic states are measured against a quarter of
/// [`with_differential_error`](Self::with_differential_error) and
/// [`with_algebraic_error`](Self::with_algebraic_error), and a step is accepted when the
/// larger of the two normalized estimates is at most one. Newton iterates must also bring the
/// differential and algebraic residual blocks below
/// [`with_differential_residual`](Self::with_differential_residual) and
/// [`with_algebraic_residual`](Self::with_algebraic_residual), relative to the fixed
/// reference scales. Component tolerances are indexed within their partition: the `k`-th
/// entry belongs to the `k`-th differential (or algebraic) variable.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct DaeNumericalOptions<S: RealScalar> {
    strategy: DerivativeStrategy,
    differential_error: ErrorTolerance<S>,
    algebraic_error: ErrorTolerance<S>,
    initial_derivative_error: ErrorTolerance<S>,
    differential_residual: Option<ResidualTolerance<S>>,
    algebraic_residual: Option<ResidualTolerance<S>>,
    differential_scale: S,
    algebraic_scale: S,
    initialization: DaeInitialization,
    newton_limit: Budget,
    rejection_limit: Budget,
}

impl<S: RealScalar> DaeOptions<S> {
    /// Returns the default options.
    ///
    /// The defaults are absolute and relative tolerances `√ε` in the maximum norm for
    /// both state partitions and for the initial differential derivatives, and absolute
    /// residual tolerances `√ε` with zero relative parts, where `ε` is the machine
    /// epsilon of `S`; unit residual scales, [`DaeInitialization::Check`], 12 Newton
    /// iterations, 12 consecutive rejections, the derivative sources of
    /// [`DaeNumericalOptions::with_derivative_strategy`], the automatic initial step and
    /// no budgets.
    pub fn new() -> Self {
        Self {
            numerics: DaeNumericalOptions::new(),
            initial_step: None,
            max_steps: None,
            max_evaluations: None,
        }
    }

    /// Sets the numerical controls, keeping the initial step and budgets.
    pub fn with_numerics(mut self, numerics: DaeNumericalOptions<S>) -> Self {
        self.numerics = numerics;
        self
    }
    /// Returns the numerical controls.
    pub const fn numerics(&self) -> &DaeNumericalOptions<S> {
        &self.numerics
    }
    /// Sets the magnitude of the first step; its sign follows the direction of
    /// integration.
    ///
    /// It must be positive and finite.
    ///
    /// Without it, every call starts with
    ///
    /// ```text
    /// h₀ = min(|t_end − t₀| / 1000, 0.5 / ‖ẋ₀‖),
    /// ```
    ///
    /// but at least the distance from `t₀` to the adjacent floating-point number and the
    /// smallest normal magnitude, so that `1/h₀` is finite. Here `t₀` is the start time,
    /// `t_end` the end time and `‖ẋ₀‖` the norm of
    /// [`DaeNumericalOptions::with_differential_error`] applied to the initial differential
    /// derivatives with scales `|x₀|`, where `x₀` are the initial differential states. The
    /// first increment `h₀ ‖ẋ₀‖` is therefore at most half the differential tolerance.
    /// A component whose tolerance threshold is exactly zero, such as a zero state under a
    /// purely relative tolerance, contributes zero to `‖ẋ₀‖`; if its derivative is nonzero,
    /// `h₀` is at most `10⁻⁶`.
    ///
    /// # References
    ///
    /// K. E. Brenan, S. L. Campbell and L. R. Petzold, *Numerical Solution of
    /// Initial-Value Problems in Differential-Algebraic Equations*, SIAM, 1996, §5.4.
    ///
    /// E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential Equations I*,
    /// 2nd ed., Springer, 1993, §II.4.
    pub fn with_initial_step(mut self, step: S) -> Self {
        self.initial_step = Some(step);
        self
    }
    /// Limits the number of attempted steps, including rejected ones.
    pub fn with_max_steps(mut self, steps: Budget) -> Self {
        self.max_steps = Some(steps);
        self
    }
    /// Limits the number of callback evaluations: residuals, Jacobians, preconditioners
    /// and matrix-free products.
    pub fn with_max_evaluations(mut self, evaluations: Budget) -> Self {
        self.max_evaluations = Some(evaluations);
        self
    }

    /// Returns the first step magnitude, if set.
    pub const fn initial_step(&self) -> Option<S> {
        self.initial_step
    }
    /// Returns the limit on attempted steps.
    pub const fn max_steps(&self) -> Option<Budget> {
        self.max_steps
    }
    /// Returns the limit on callback evaluations.
    pub const fn max_evaluations(&self) -> Option<Budget> {
        self.max_evaluations
    }

    /// Budgets of one integration call.
    pub(crate) fn budgets(&self) -> SolveBudgets {
        let budgets = SolveBudgets::unbounded();
        let budgets = match self.max_steps {
            Some(steps) => budgets.with_limit(BudgetKind::Proposal, steps),
            None => budgets,
        };
        match self.max_evaluations {
            Some(evaluations) => budgets.with_limit(BudgetKind::Evaluation, evaluations),
            None => budgets,
        }
    }

    /// Validate everything that depends only on the partition.
    pub(crate) fn validate(&self, partition: &DaePartition) -> Result<(), SciOxideError> {
        let n = partition.dimension();
        let differential = partition.differential();
        for (scale, argument) in [
            (
                self.numerics.differential_scale,
                scioxide_core::argument!("options.differential_scale"),
            ),
            (
                self.numerics.algebraic_scale,
                scioxide_core::argument!("options.algebraic_scale"),
            ),
        ] {
            if !scale.is_finite() || scale < S::ZERO {
                return Err(invalid(
                    argument,
                    "residual scales must be finite and nonnegative",
                ));
            }
        }
        for (tolerance, count, argument) in [
            (
                &self.numerics.differential_error,
                differential,
                scioxide_core::argument!("options.differential_error"),
            ),
            (
                &self.numerics.initial_derivative_error,
                differential,
                scioxide_core::argument!("options.initial_derivative_error"),
            ),
            (
                &self.numerics.algebraic_error,
                n - differential,
                scioxide_core::argument!("options.algebraic_error"),
            ),
        ] {
            if tolerance
                .absolute()
                .as_vector()
                .is_some_and(|values| values.len() != count)
            {
                return Err(invalid(
                    argument,
                    "component tolerance length must match its partition",
                ));
            }
            for i in 0..count {
                tolerance.threshold(i, count, S::ONE)?;
            }
        }
        self.numerics
            .residual_tolerances()?
            .0
            .tolerance()
            .threshold(self.numerics.differential_scale)?;
        self.numerics
            .residual_tolerances()?
            .1
            .tolerance()
            .threshold(self.numerics.algebraic_scale)?;
        if let Some(step) = self.initial_step
            && (!step.is_finite() || step <= S::ZERO)
        {
            return Err(invalid(
                scioxide_core::argument!("options.initial_step"),
                "positive finite initial step required",
            ));
        }
        Ok(())
    }
}

impl<S: RealScalar> DaeNumericalOptions<S> {
    /// Returns the default numerical controls, as listed on [`DaeOptions::new`].
    pub fn new() -> Self {
        let error = || {
            ErrorTolerance::from_tolerance(Tolerance::from_sqrt_epsilon(), NormSelection::Infinity)
        };
        Self {
            strategy: crate::stiff::DEFAULT_STRATEGY,
            differential_error: error(),
            algebraic_error: error(),
            initial_derivative_error: error(),
            differential_residual: None,
            algebraic_residual: None,
            differential_scale: S::ONE,
            algebraic_scale: S::ONE,
            initialization: DaeInitialization::Check,
            newton_limit: Budget::ONE.saturating_add(11),
            rejection_limit: Budget::ONE.saturating_add(11),
        }
    }
    /// Sets the sources of the residual Jacobian blocks and matrix-free products.
    ///
    /// The default tries the analytic callback, then forward differences; automatic
    /// differentiation is not available for DAE residuals and is skipped. A source falls
    /// through only when it is unavailable; a callback error is returned unchanged.
    pub fn with_derivative_strategy(mut self, strategy: DerivativeStrategy) -> Self {
        self.strategy = strategy;
        self
    }
    /// Returns the sources of the residual Jacobian blocks and products.
    pub const fn derivative_strategy(&self) -> DerivativeStrategy {
        self.strategy
    }

    /// Sets the local error tolerance of the differential states.
    pub fn with_differential_error(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.differential_error = tolerance;
        self
    }
    /// Sets the local error tolerance of the algebraic states.
    pub fn with_algebraic_error(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.algebraic_error = tolerance;
        self
    }
    /// Sets the tolerance of differential derivative corrections during initialization,
    /// in units of the derivative.
    ///
    /// Its absolute terms also set the smallest difference increment of the
    /// differential derivatives.
    pub fn with_initial_derivative_error(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.initial_derivative_error = tolerance;
        self
    }
    /// Sets the residual tolerance of the differential equations, in units of the
    /// residual.
    pub fn with_differential_residual(mut self, tolerance: ResidualTolerance<S>) -> Self {
        self.differential_residual = Some(tolerance);
        self
    }
    /// Sets the residual tolerance of the algebraic equations, in units of the
    /// constraints.
    pub fn with_algebraic_residual(mut self, tolerance: ResidualTolerance<S>) -> Self {
        self.algebraic_residual = Some(tolerance);
        self
    }
    /// Sets the reference scale of the differential residual tolerance.
    ///
    /// It must be finite and nonnegative and is never derived from the initial residual.
    pub fn with_differential_scale(mut self, scale: S) -> Self {
        self.differential_scale = scale;
        self
    }
    /// Sets the reference scale of the algebraic residual tolerance.
    ///
    /// It must be finite and nonnegative.
    pub fn with_algebraic_scale(mut self, scale: S) -> Self {
        self.algebraic_scale = scale;
        self
    }
    /// Sets how every integration call treats the initial pair.
    pub fn with_initialization(mut self, initialization: DaeInitialization) -> Self {
        self.initialization = initialization;
        self
    }
    /// Sets the maximum number of Newton iterations per attempt.
    ///
    /// A dense or sparse BDF stage that fails retries up to twice, first with a
    /// factorization at the current step size, then with freshly evaluated Jacobian
    /// blocks. Initialization and matrix-free stages make one attempt.
    pub fn with_newton_limit(mut self, limit: Budget) -> Self {
        self.newton_limit = limit;
        self
    }
    /// Sets the maximum number of consecutive rejected steps.
    pub fn with_rejection_limit(mut self, limit: Budget) -> Self {
        self.rejection_limit = limit;
        self
    }
    /// Returns the local error tolerance of the differential states.
    pub const fn differential_error(&self) -> &ErrorTolerance<S> {
        &self.differential_error
    }
    /// Returns the local error tolerance of the algebraic states.
    pub const fn algebraic_error(&self) -> &ErrorTolerance<S> {
        &self.algebraic_error
    }
    /// Returns the tolerance of differential derivative corrections during
    /// initialization.
    pub const fn initial_derivative_error(&self) -> &ErrorTolerance<S> {
        &self.initial_derivative_error
    }
    /// Returns the differential residual tolerance, or `None` for `(atol, rtol) = (√ε, 0)`.
    pub const fn differential_residual(&self) -> Option<ResidualTolerance<S>> {
        self.differential_residual
    }
    /// Returns the algebraic residual tolerance, or `None` for `(atol, rtol) = (√ε, 0)`.
    pub const fn algebraic_residual(&self) -> Option<ResidualTolerance<S>> {
        self.algebraic_residual
    }
    /// Returns the reference scale of the differential residual tolerance.
    pub const fn differential_scale(&self) -> S {
        self.differential_scale
    }
    /// Returns the reference scale of the algebraic residual tolerance.
    pub const fn algebraic_scale(&self) -> S {
        self.algebraic_scale
    }
    /// Returns how the initial pair is treated.
    pub const fn initialization(&self) -> DaeInitialization {
        self.initialization
    }
    /// Returns the maximum number of Newton iterations per attempt.
    pub const fn newton_limit(&self) -> Budget {
        self.newton_limit
    }
    /// Returns the maximum number of consecutive rejected steps.
    pub const fn rejection_limit(&self) -> Budget {
        self.rejection_limit
    }
}

impl<S: RealScalar> Default for DaeOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> Default for DaeNumericalOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> DaeNumericalOptions<S> {
    pub(crate) fn residual_tolerances(
        &self,
    ) -> Result<(ResidualTolerance<S>, ResidualTolerance<S>), SciOxideError> {
        let resolve = |value| match value {
            Some(value) => Ok(value),
            None => ResidualTolerance::new(S::SQRT_EPSILON, S::ZERO),
        };
        Ok((
            resolve(self.differential_residual)?,
            resolve(self.algebraic_residual)?,
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    #[test]
    fn default_residual_thresholds_preserve_absolute_sqrt_epsilon() {
        let options = DaeNumericalOptions::<f64>::default();
        assert_eq!(options, DaeNumericalOptions::new());
        let (differential, algebraic) = options.residual_tolerances().unwrap();
        for residual in [differential, algebraic] {
            assert_eq!(
                residual.threshold(1.0).unwrap().to_bits(),
                f64::EPSILON.sqrt().to_bits()
            );
            assert_eq!(
                residual.threshold(0.0).unwrap().to_bits(),
                f64::EPSILON.sqrt().to_bits()
            );
        }
        let options = DaeNumericalOptions::<f32>::default();
        let (differential, algebraic) = options.residual_tolerances().unwrap();
        for residual in [differential, algebraic] {
            assert_eq!(
                residual.threshold(1.0).unwrap().to_bits(),
                f32::EPSILON.sqrt().to_bits()
            );
        }
    }
}
