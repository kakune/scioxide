//! Time-scheme configuration of transient PDE problems; stepping belongs to scioxide-ode.

use scioxide_core::{
    Budget, ErrorTolerance, NormSelection, NumericScalar, RealScalar, SciOxideError, SolveBudgets,
    Tolerance,
};
use scioxide_ode::{BuiltinOdeMethod, OdeStageBackend};

use super::output::PdeOutputSchedule;
use crate::mol::CflPolicy;

/// Time integration scheme for a parabolic PDE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ParabolicScheme {
    /// First-order, A-stable and L-stable backward Euler.
    BackwardEuler,
    /// Second-order, A-stable Crank--Nicolson.
    CrankNicolson,
    /// Adaptive explicit Dormand--Prince 5(4) integration.
    ExplicitDopri5,
    /// Adaptive fifth-order, L-stable three-stage Radau IIA.
    #[default]
    RadauIia,
}

impl ParabolicScheme {
    /// Stable human-readable scheme name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::BackwardEuler => "BackwardEuler",
            Self::CrankNicolson => "CrankNicolson",
            Self::ExplicitDopri5 => "DormandPrince5",
            Self::RadauIia => "RadauIia",
        }
    }
}

/// Optional backward-Euler smoothing before Crank--Nicolson stepping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RannacherStartup {
    /// Begin immediately with the selected scheme.
    #[default]
    Disabled,
    /// Replace the first `intervals` Crank--Nicolson steps of size `h`, over
    /// `[0, intervals·h]`, with backward-Euler steps of size `h/2`.
    BackwardEuler {
        /// Number of full intervals receiving startup smoothing.
        intervals: Budget,
    },
}

fn positive_step<S: RealScalar>(
    step: S,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if !step.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "time step must be finite",
        ));
    }
    if step <= S::ZERO {
        return Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::OutOfRange,
            "time step must be positive",
        ));
    }
    Ok(())
}

fn default_tolerance<S: RealScalar>() -> ErrorTolerance<S> {
    ErrorTolerance::from_tolerance(
        Tolerance::from_decimal_places(8, 8),
        NormSelection::RootMeanSquare,
    )
}

/// Configuration of a [`crate::ParabolicProblem`]: scheme, steps, tolerances, output,
/// budgets and stability limiting. The end time belongs to [`crate::ParabolicProblem`].
#[derive(Debug, Clone)]
#[must_use]
pub struct ParabolicOptions<S: RealScalar + NumericScalar<Real = S>> {
    step: Option<S>,
    scheme: ParabolicScheme,
    tolerance: ErrorTolerance<S>,
    cfl: CflPolicy<S>,
    backend: Option<OdeStageBackend>,
    method: Option<BuiltinOdeMethod<S>>,
    output: PdeOutputSchedule<S>,
    step_budget: Budget,
    budgets: SolveBudgets,
    startup: RannacherStartup,
}

impl<S: RealScalar + NumericScalar<Real = S>> ParabolicOptions<S> {
    /// Select adaptive Radau IIA, 8-decimal RMS tolerance, automatic stability
    /// limiting, final-only output and at most 100,000 attempted steps.
    pub fn new() -> Self {
        Self {
            step: None,
            scheme: ParabolicScheme::RadauIia,
            tolerance: default_tolerance(),
            cfl: CflPolicy::Automatic,
            backend: None,
            method: None,
            output: PdeOutputSchedule::FinalOnly,
            step_budget: Budget::ONE.saturating_add(99_999),
            budgets: SolveBudgets::unbounded(),
            startup: RannacherStartup::Disabled,
        }
    }
    /// Select the integration scheme; execution validates its required step.
    pub const fn with_scheme(mut self, scheme: ParabolicScheme) -> Self {
        self.scheme = scheme;
        self
    }
    /// Set the fixed step for Euler/trapezoid, or the initial adaptive step.
    pub const fn with_step(mut self, step: S) -> Self {
        self.step = Some(step);
        self
    }
    /// Physical component-scaled local-error control.
    pub fn with_error_tolerance(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Select automatic, disabled, or caller-supplied explicit stability limiting.
    pub fn with_cfl_policy(mut self, policy: CflPolicy<S>) -> Self {
        self.cfl = policy;
        self
    }

    /// Select the implicit-stage backend of the built-in schemes; the sparse pattern
    /// must match the free unknowns. The default is sparse LU on the Jacobian pattern.
    pub fn with_stage_backend(mut self, backend: OdeStageBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Replace the configured scheme by a built-in ODE method with its own stage
    /// policy; its declared linear stability drives automatic step limits.
    pub fn with_ode_method(mut self, method: impl Into<BuiltinOdeMethod<S>>) -> Self {
        self.method = Some(method.into());
        self
    }

    /// Select final-only, accepted-step or requested-time output.
    pub fn with_output_schedule(mut self, output: PdeOutputSchedule<S>) -> Self {
        self.output = output;
        self
    }

    /// Maximum attempted time steps of one advance, over every phase.
    pub const fn with_step_budget(mut self, step_budget: Budget) -> Self {
        self.step_budget = step_budget;
        self
    }

    /// Local solve budgets of one advance (evaluations, proposals, iterations).
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }

    /// Rannacher startup for Crank--Nicolson; validated at execution preflight.
    pub const fn with_rannacher_startup(mut self, intervals: Budget) -> Self {
        self.startup = RannacherStartup::BackwardEuler { intervals };
        self
    }

    /// Time-step policy.
    pub const fn step(&self) -> Option<S> {
        self.step
    }

    /// Selected scheme.
    pub const fn scheme(&self) -> ParabolicScheme {
        self.scheme
    }

    /// Physical component accuracy for adaptive control and implicit stages.
    pub const fn error_tolerance(&self) -> &ErrorTolerance<S> {
        &self.tolerance
    }

    /// Stability limiting.
    pub const fn cfl_policy(&self) -> CflPolicy<S> {
        self.cfl
    }

    /// Explicit implicit-stage backend, if any.
    pub const fn stage_backend(&self) -> Option<&OdeStageBackend> {
        self.backend.as_ref()
    }

    /// Built-in ODE method replacing the scheme, if any.
    pub const fn ode_method(&self) -> Option<&BuiltinOdeMethod<S>> {
        self.method.as_ref()
    }

    /// Output schedule.
    pub const fn output_schedule(&self) -> &PdeOutputSchedule<S> {
        &self.output
    }

    /// Maximum attempted time steps of one advance.
    pub const fn step_budget(&self) -> Budget {
        self.step_budget
    }

    /// Local solve budgets.
    pub const fn budgets(&self) -> SolveBudgets {
        self.budgets
    }

    /// Configured startup policy.
    pub const fn startup(&self) -> RannacherStartup {
        self.startup
    }

    /// Whole-configuration checks run at execution preflight.
    pub(super) fn validate(&self) -> Result<(), SciOxideError> {
        if let Some(step) = self.step {
            positive_step(step, scioxide_core::argument!("options.step"))?;
        }
        if self.step.is_none()
            && self.method.is_none()
            && matches!(
                self.scheme,
                ParabolicScheme::BackwardEuler | ParabolicScheme::CrankNicolson
            )
        {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("options.step"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "fixed-step schemes require a step",
            ));
        }
        self.cfl.validate(scioxide_core::argument!("options"))?;
        self.output.validate()?;
        if matches!(self.startup, RannacherStartup::BackwardEuler { .. })
            && (self.scheme != ParabolicScheme::CrankNicolson || self.method.is_some())
        {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("options"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "Rannacher startup requires the built-in CrankNicolson scheme",
            ));
        }
        Ok(())
    }
}

/// Configuration of a [`crate::HyperbolicProblem`]: adaptive explicit integration of
/// `M u'' = b(t) − A u`. The end time belongs to [`crate::HyperbolicProblem`].
#[derive(Debug, Clone)]
#[must_use]
pub struct HyperbolicOptions<S: RealScalar + NumericScalar<Real = S>> {
    initial_step: Option<S>,
    tolerance: ErrorTolerance<S>,
    cfl: CflPolicy<S>,
    method: Option<BuiltinOdeMethod<S>>,
    output: PdeOutputSchedule<S>,
    step_budget: Budget,
    budgets: SolveBudgets,
}

impl<S: RealScalar + NumericScalar<Real = S>> HyperbolicOptions<S> {
    /// Select adaptive Dormand–Prince 5(4), an estimated first step, 8-decimal RMS
    /// tolerance, automatic stability limiting and 100,000 attempted steps.
    pub fn new() -> Self {
        Self {
            initial_step: None,
            tolerance: default_tolerance(),
            cfl: CflPolicy::Automatic,
            method: None,
            output: PdeOutputSchedule::FinalOnly,
            step_budget: Budget::ONE.saturating_add(99_999),
            budgets: SolveBudgets::unbounded(),
        }
    }
    /// Set the initial adaptive step, validated at execution.
    pub const fn with_initial_step(mut self, step: S) -> Self {
        self.initial_step = Some(step);
        self
    }
    /// Replace component-scaled local-error control.
    pub fn with_error_tolerance(mut self, tolerance: ErrorTolerance<S>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Select explicit stability limiting.
    pub fn with_cfl_policy(mut self, policy: CflPolicy<S>) -> Self {
        self.cfl = policy;
        self
    }

    /// Replace Dormand--Prince by a built-in ODE method; its declared imaginary-axis
    /// stability drives automatic step limits.
    pub fn with_ode_method(mut self, method: impl Into<BuiltinOdeMethod<S>>) -> Self {
        self.method = Some(method.into());
        self
    }

    /// Select final-only, accepted-step or requested-time output.
    pub fn with_output_schedule(mut self, output: PdeOutputSchedule<S>) -> Self {
        self.output = output;
        self
    }

    /// Maximum attempted time steps of one advance.
    pub const fn with_step_budget(mut self, step_budget: Budget) -> Self {
        self.step_budget = step_budget;
        self
    }

    /// Local solve budgets of one advance.
    pub const fn with_budgets(mut self, budgets: SolveBudgets) -> Self {
        self.budgets = budgets;
        self
    }

    /// Initial adaptive time step.
    pub const fn initial_step(&self) -> Option<S> {
        self.initial_step
    }

    /// Configured component-scaled local-error control.
    pub const fn error_tolerance(&self) -> &ErrorTolerance<S> {
        &self.tolerance
    }

    /// Stability limiting.
    pub const fn cfl_policy(&self) -> CflPolicy<S> {
        self.cfl
    }

    /// Built-in ODE method replacing Dormand--Prince, if any.
    pub const fn ode_method(&self) -> Option<&BuiltinOdeMethod<S>> {
        self.method.as_ref()
    }

    /// Output schedule.
    pub const fn output_schedule(&self) -> &PdeOutputSchedule<S> {
        &self.output
    }

    /// Maximum attempted time steps of one advance.
    pub const fn step_budget(&self) -> Budget {
        self.step_budget
    }

    /// Local solve budgets.
    pub const fn budgets(&self) -> SolveBudgets {
        self.budgets
    }

    /// The equivalent first-order configuration of the shared transient core.
    pub(super) fn validate(&self) -> Result<(), SciOxideError> {
        if let Some(step) = self.initial_step {
            positive_step(step, scioxide_core::argument!("options.initial_step"))?;
        }
        self.parabolic().validate()
    }
    pub(super) fn parabolic(&self) -> ParabolicOptions<S> {
        ParabolicOptions {
            step: self.initial_step,
            scheme: ParabolicScheme::ExplicitDopri5,
            tolerance: self.tolerance.clone(),
            cfl: self.cfl,
            backend: None,
            method: self.method.clone(),
            output: self.output.clone(),
            step_budget: self.step_budget,
            budgets: self.budgets,
            startup: RannacherStartup::Disabled,
        }
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for ParabolicOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar + NumericScalar<Real = S>> Default for HyperbolicOptions<S> {
    fn default() -> Self {
        Self::new()
    }
}
