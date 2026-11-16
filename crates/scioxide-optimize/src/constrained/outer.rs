//! Outer loop of the quadratic penalty and augmented Lagrangian methods.

use super::snapshot::Snapshot;
use crate::constrained::ConstrainedStorage;
use crate::context::SolveSession;
use crate::{
    Bfgs, Constraints, LineSearch, ObjectiveFunction, OptimalityDiagnostics, OptimalityTolerances,
    OptimizeResult, OptimizeTermination,
};
use core::cell::RefCell;

use scioxide_core::{
    Budget, NumericScalar, PartialResultTrust, RealScalar, SciOxideError, SolveState,
};
use scioxide_core::{
    checked_float_product as product, checked_float_quotient as quotient, checked_sum,
};
use scioxide_differentiate::DerivativeStrategy;

type Session<'a, 'b, 'c> = SolveSession<'a, 'b, 'c>;

/// PHR term and its derivative with respect to a scaled constraint value.
fn term<S: RealScalar + NumericScalar<Real = S>>(
    value: S,
    multiplier: S,
    mu: S,
    equality: bool,
) -> Result<(S, S), SciOxideError> {
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let half = one / (one + one);
    let shifted = checked_sum(&[multiplier, product(mu, value)?])?;
    if equality || shifted > zero {
        let coefficient = checked_sum(&[multiplier, product(half, product(mu, value)?)?])?;
        Ok((product(coefficient, value)?, shifted))
    } else {
        // Essential constant: the PHR inequality term is continuous at lambda + mu*c = 0.
        Ok((
            -product(product(half, multiplier)?, quotient(multiplier, mu)?)?,
            zero,
        ))
    }
}

use crate::geometry::violation as feasibility;

fn model_value<S: RealScalar + NumericScalar<Real = S>>(
    snapshot: &Snapshot<S>,
    multipliers: &[S],
    mu: S,
    equalities: usize,
    restoration: bool,
) -> Result<S, SciOxideError> {
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let initial = if restoration {
        zero
    } else {
        snapshot.value.ok_or_else(|| {
            SciOxideError::invalid_input(
                scioxide_core::argument!("f"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "objective is unavailable",
            )
        })?
    };
    snapshot
        .values
        .iter()
        .zip(multipliers)
        .enumerate()
        .try_fold(initial, |sum, (row, (&value, &multiplier))| {
            let (value, _) = term(
                value,
                if restoration { zero } else { multiplier },
                mu,
                row < equalities,
            )?;
            checked_sum(&[sum, value])
        })
}

fn model_gradient<S: RealScalar + NumericScalar<Real = S>>(
    snapshot: &Snapshot<S>,
    multipliers: &[S],
    mu: S,
    equalities: usize,
    restoration: bool,
    output: &mut [S],
) -> Result<(), SciOxideError> {
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    if restoration {
        output.fill(zero);
    } else {
        output.copy_from_slice(&snapshot.gradient);
    }
    for (row, (&value, &multiplier)) in snapshot.values.iter().zip(multipliers).enumerate() {
        let (_, coefficient) = term(
            value,
            if restoration { zero } else { multiplier },
            mu,
            row < equalities,
        )?;
        for (component, &derivative) in output.iter_mut().zip(
            &snapshot.jacobian.as_slice()
                [row * snapshot.point.len()..(row + 1) * snapshot.point.len()],
        ) {
            *component = checked_sum(&[*component, product(coefficient, derivative)?])?;
        }
    }
    Ok(())
}

struct Model<'a, 'b, 'c, 'd, 'e, S: RealScalar + NumericScalar<Real = S>, F: ?Sized> {
    objective: &'a F,
    constraints: Option<&'a Constraints<S>>,
    snapshot: RefCell<&'a mut Snapshot<S>>,
    multipliers: &'a [S],
    mu: S,
    equalities: usize,
    restoration: bool,
    strategy: DerivativeStrategy,
    session: &'b Session<'c, 'd, 'e>,
}

impl<S: RealScalar + NumericScalar<Real = S>, F: ObjectiveFunction<S> + ?Sized> ObjectiveFunction<S>
    for Model<'_, '_, '_, '_, '_, S, F>
{
    fn dim(&self) -> usize {
        self.objective.dim()
    }
    fn is_in_domain(&self, x: &[S]) -> Result<bool, SciOxideError> {
        if self.restoration {
            Ok(true)
        } else {
            self.session.observe(self.objective).is_in_domain(x)
        }
    }
    fn eval(&self, x: &[S]) -> Result<S, SciOxideError> {
        let mut snapshot = self.snapshot.try_borrow_mut().map_err(|_| {
            crate::context::observation_borrow_error(scioxide_core::argument!("workspace"))
        })?;
        snapshot.at(x);
        // A non-finite objective or constraint value rejects this inner trial.
        if !snapshot.trial(
            self.objective,
            self.constraints,
            self.restoration,
            self.session,
        )? {
            return Ok(S::INFINITY);
        }
        model_value(
            &snapshot,
            self.multipliers,
            self.mu,
            self.equalities,
            self.restoration,
        )
    }
    fn gradient_into(&self, x: &[S], output: &mut [S]) -> Result<bool, SciOxideError> {
        let mut snapshot = self.snapshot.try_borrow_mut().map_err(|_| {
            crate::context::observation_borrow_error(scioxide_core::argument!("workspace"))
        })?;
        snapshot.at(x);
        snapshot.derivatives(
            self.objective,
            self.constraints,
            self.strategy,
            self.restoration,
            self.session,
        )?;
        model_gradient(
            &snapshot,
            self.multipliers,
            self.mu,
            self.equalities,
            self.restoration,
            output,
        )?;
        Ok(true)
    }
    fn automatic_gradient_into(&self, x: &[S], output: &mut [S]) -> Result<bool, SciOxideError> {
        self.gradient_into(x, output)
    }
}

pub(super) fn diagnostics<S: RealScalar + NumericScalar<Real = S>>(
    snapshot: &Snapshot<S>,
    multipliers: &[S],
    equalities: usize,
    scratch: &mut [S],
) -> Result<OptimalityDiagnostics<S>, SciOxideError> {
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    scratch.copy_from_slice(&snapshot.gradient);
    let mut complementarity = zero;
    let mut dual = zero;
    for (row, (&value, &lambda)) in snapshot.values.iter().zip(multipliers).enumerate() {
        for (component, &derivative) in scratch.iter_mut().zip(
            &snapshot.jacobian.as_slice()
                [row * snapshot.point.len()..(row + 1) * snapshot.point.len()],
        ) {
            *component = checked_sum(&[*component, product(lambda, derivative)?])?;
        }
        if row >= equalities {
            complementarity = complementarity.max(NumericScalar::abs(product(lambda, value)?));
            dual = dual.max(-lambda);
        }
    }
    Ok(OptimalityDiagnostics {
        primal_feasibility: Some(feasibility(&snapshot.values, equalities)),
        stationarity: Some(crate::utils::norm2(scratch)?),
        dual_feasibility: Some(dual),
        complementarity: Some(complementarity),
        ..OptimalityDiagnostics::unavailable()
    })
}

pub(super) fn bound_violation<S: RealScalar + NumericScalar<Real = S>>(
    constraints: Option<&Constraints<S>>,
    point: &[S],
) -> Option<S> {
    constraints.and_then(|constraints| {
        constraints
            .bound_sets()
            .map(|bounds| crate::geometry::bound_violation(point, bounds.as_slice()))
            .reduce(S::max)
    })
}

fn original_report<S: RealScalar + NumericScalar<Real = S>>(
    snapshot: &Snapshot<S>,
    multipliers: &[S],
    mut report: OptimalityDiagnostics<S>,
    equalities: usize,
) -> Result<OptimalityDiagnostics<S>, SciOxideError> {
    report.primal_feasibility = Some(feasibility(&snapshot.original_values, equalities));
    report.stationarity = report
        .stationarity
        .map(|v| product(v, snapshot.objective_scale))
        .transpose()?;
    report.complementarity = report
        .complementarity
        .map(|v| product(v, snapshot.objective_scale))
        .transpose()?;
    let mut dual = <S as scioxide_core::AlgebraicScalar>::ZERO;
    for (i, &lambda) in multipliers.iter().enumerate().skip(equalities) {
        dual = dual.max(-product(
            quotient(lambda, snapshot.row_scales[i])?,
            snapshot.objective_scale,
        )?);
    }
    report.dual_feasibility = Some(dual);
    Ok(report)
}

/// Recover independent original-problem multipliers without replacing the method's dual state.
pub(super) fn certify_trial<S: RealScalar + NumericScalar<Real = S>>(
    workspace: &mut ConstrainedStorage<S>,
    target: S,
    session: &Session<'_, '_, '_>,
) -> Result<OptimalityDiagnostics<S>, SciOxideError> {
    workspace
        .constraint_values
        .copy_from_slice(&workspace.trial_snapshot.values);
    workspace
        .objective_gradient
        .copy_from_slice(&workspace.trial_snapshot.gradient);
    workspace
        .constraint_jacobian
        .as_mut_slice()
        .copy_from_slice(workspace.trial_snapshot.jacobian.as_slice());
    core::mem::swap(
        &mut workspace.multipliers,
        &mut workspace.certificate_multipliers,
    );
    let outcome = super::solve_certificate_multipliers(workspace, target, session).and_then(|()| {
        diagnostics(
            &workspace.trial_snapshot,
            &workspace.multipliers,
            workspace.equality_count,
            &mut workspace.trial_gradient,
        )
    });
    core::mem::swap(
        &mut workspace.multipliers,
        &mut workspace.certificate_multipliers,
    );
    // The certificate uses scratch arrays; restore the accepted constraint state even on failure.
    workspace
        .constraint_values
        .copy_from_slice(&workspace.accepted_snapshot.values);
    outcome
}

#[derive(Debug, PartialEq, Eq)]
enum Update {
    Retain,
    Grow,
    Restore,
}

fn update<S: RealScalar + NumericScalar<Real = S>>(
    previous: S,
    trial: S,
    target: S,
    stationary: bool,
    stalls: usize,
) -> Result<Update, SciOxideError> {
    let sufficient = trial <= target || trial <= product(previous, S::HALF)?;
    Ok(if sufficient {
        Update::Retain
    } else if !stationary || stalls >= 1 {
        Update::Restore
    } else {
        Update::Grow
    })
}

#[allow(clippy::too_many_arguments)]
fn subproblem<S, L, F>(
    inner: &Bfgs<S, L>,
    f: &F,
    constraints: Option<&Constraints<S>>,
    workspace: &mut ConstrainedStorage<S>,
    session: &Session<'_, '_, '_>,
    restoration: bool,
    threshold: S,
    token: usize,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: LineSearch<S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    workspace
        .trial_snapshot
        .copy_from(&workspace.accepted_snapshot);
    session.proposal()?;
    let mut child_state = SolveState::new();
    let mut stopped = None;
    let outcome = session.with_child(
        &mut child_state,
        inner.solve_identity(scioxide_core::ProblemToken::new(token as u128), f.dim())?,
        inner.child_budgets(),
        |context| {
            let child = SolveSession::new(context, scioxide_core::argument!("context"));
            let model = Model {
                objective: f,
                constraints,
                snapshot: RefCell::new(&mut workspace.trial_snapshot),
                multipliers: &workspace.multipliers,
                mu: if restoration {
                    <S as scioxide_core::AlgebraicScalar>::ONE
                } else {
                    workspace.penalty_parameter
                },
                equalities: workspace.equality_count,
                restoration,
                strategy: inner.derivative_strategy(),
                session: &child,
            };
            let result = inner.minimize_subproblem(
                &model,
                &workspace.x,
                &mut workspace.inner_result,
                &mut workspace.first_order,
                &child,
                threshold,
            );
            stopped = child.stopped();
            result
        },
    );
    session.inherit_child_stop(stopped)?;
    outcome?;
    session.check()?;
    workspace.trial_snapshot.at(workspace.inner_result.x());
    workspace.trial_snapshot.constraints(constraints, session)?;
    workspace.trial_snapshot.derivatives(
        f,
        constraints,
        inner.derivative_strategy(),
        restoration,
        session,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn minimize<S, L, F>(
    inner: &Bfgs<S, L>,
    f: &F,
    x0: &[S],
    constraints: Option<&Constraints<S>>,
    tolerances: OptimalityTolerances<S>,
    iterations: Budget,
    augmented: bool,
    output: &mut OptimizeResult<S>,
    workspace: &mut ConstrainedStorage<S>,
    session: &Session<'_, '_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: LineSearch<S>,
    F: ObjectiveFunction<S> + ?Sized,
{
    crate::utils::validate_x0(f, x0)?;
    if let Some(constraints) = constraints
        && constraints.dimension() != f.dim()
    {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(f.dim()),
            scioxide_core::Dimensions::one(constraints.dimension()),
        ));
    }
    let profile = constraints.map_or(crate::ConstraintProfile::EMPTY, Constraints::profile);
    let eq = profile.equality_count();
    let m = profile.total_count();
    workspace.try_reserve_nested_bfgs(f.dim(), eq, profile.inequality_count())?;
    output.try_reserve_without_gradient(f.dim())?;
    workspace.reset();
    workspace.accepted_snapshot.invalidate();
    workspace.accepted_snapshot.objective_scale = <S as scioxide_core::AlgebraicScalar>::ZERO;
    workspace
        .accepted_snapshot
        .row_scales
        .fill(<S as scioxide_core::AlgebraicScalar>::ONE);
    workspace.x.copy_from_slice(x0);
    workspace.accepted_snapshot.at(x0);
    let initial = workspace.accepted_snapshot.objective(f, session)?;
    workspace.accepted_value = Some(initial);
    workspace.accepted_snapshot.derivatives(
        f,
        constraints,
        inner.derivative_strategy(),
        false,
        session,
    )?;
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    workspace.multipliers.fill(zero);
    workspace.penalty_parameter = one;
    let target = tolerances.feasibility().threshold(one)?;
    let final_stationarity = tolerances.gradient().threshold(one)?;
    let mut previous = feasibility(&workspace.accepted_snapshot.values, eq);
    let mut report = diagnostics(
        &workspace.accepted_snapshot,
        &workspace.multipliers,
        eq,
        &mut workspace.trial_gradient,
    )?;
    report.bound_violation = bound_violation(constraints, &workspace.accepted_snapshot.point);
    workspace.accepted_diagnostics = original_report(
        &workspace.accepted_snapshot,
        &workspace.multipliers,
        report,
        eq,
    )?;
    let mut stalls = 0;
    let mut termination = OptimizeTermination::IterationLimit;
    for outer in 0..iterations.get() {
        if tolerances.satisfied(
            report,
            crate::utils::optimality_scales(
                &workspace.accepted_snapshot.point,
                workspace.accepted_snapshot.value.ok_or_else(|| {
                    SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Breakdown,
                        "missing accepted KKT objective",
                    )
                })?,
                &workspace.accepted_snapshot.gradient,
                workspace.accepted_snapshot.jacobian.view(),
                &workspace.multipliers,
                &[],
                &[],
            )?,
        )? {
            termination = OptimizeTermination::Converged;
            break;
        }
        if let Some(stop) = session.iteration()? {
            termination = stop;
            break;
        }
        // Never relax a caller's threshold; approach the outer gradient target as feasibility improves.
        let threshold = inner.gradient_threshold()?.min(final_stationarity);
        subproblem(
            inner,
            f,
            constraints,
            workspace,
            session,
            false,
            threshold,
            outer,
        )?;
        let trial = feasibility(&workspace.trial_snapshot.values, eq);
        model_gradient(
            &workspace.trial_snapshot,
            &workspace.multipliers,
            workspace.penalty_parameter,
            eq,
            false,
            &mut workspace.trial_gradient,
        )?;
        let stationary = crate::utils::norm2(&workspace.trial_gradient)? <= threshold;
        let action = update(previous, trial, target, stationary, stalls)?;
        let old_merit = model_value(
            &workspace.accepted_snapshot,
            &workspace.multipliers,
            workspace.penalty_parameter,
            eq,
            false,
        )?;
        let new_merit = model_value(
            &workspace.trial_snapshot,
            &workspace.multipliers,
            workspace.penalty_parameter,
            eq,
            false,
        )?;
        let restore = previous > target
            && (action == Update::Restore || (new_merit > old_merit && trial >= previous));
        let mut next_mu = workspace.penalty_parameter;
        if restore {
            session.reject()?;
            subproblem(
                inner,
                f,
                constraints,
                workspace,
                session,
                true,
                final_stationarity.min(inner.gradient_threshold()?),
                outer,
            )?;
            let restored = feasibility(&workspace.trial_snapshot.values, eq);
            if restored > target && restored > product(previous, S::HALF)? {
                termination = OptimizeTermination::Stagnation;
                break;
            }
            // A restored point is a proposal until its original objective and derivatives are valid.
            workspace.trial_snapshot.derivatives(
                f,
                constraints,
                inner.derivative_strategy(),
                false,
                session,
            )?;
            workspace.merit_scratch[..m].copy_from_slice(&workspace.multipliers);
            next_mu = product(next_mu, S::TEN)?;
            stalls = 0;
        } else {
            if new_merit > old_merit && trial >= previous {
                termination = OptimizeTermination::Stagnation;
                break;
            }
            for (row, ((next, &lambda), &value)) in workspace.merit_scratch[..m]
                .iter_mut()
                .zip(&workspace.multipliers)
                .zip(&workspace.trial_snapshot.values)
                .enumerate()
            {
                *next = term(
                    value,
                    if augmented { lambda } else { zero },
                    workspace.penalty_parameter,
                    row < eq,
                )?
                .1;
            }
            if action == Update::Grow {
                next_mu = product(next_mu, S::TEN)?;
                stalls += 1;
            } else {
                stalls = 0;
            }
        }
        let mut candidate_report = diagnostics(
            &workspace.trial_snapshot,
            &workspace.merit_scratch[..m],
            eq,
            &mut workspace.trial_gradient,
        )?;
        let certified = feasibility(&workspace.trial_snapshot.values, eq) <= target;
        if certified {
            candidate_report = certify_trial(workspace, target, session)?;
            let complementarity_target = tolerances.complementarity().threshold(one)?;
            if candidate_report
                .complementarity()
                .is_some_and(|r| r > complementarity_target)
            {
                next_mu = product(next_mu, S::TEN)?;
            }
        }
        candidate_report.bound_violation =
            bound_violation(constraints, &workspace.trial_snapshot.point);
        session.check()?;
        session.accept(PartialResultTrust::DiagnosticOnly)?;
        core::mem::swap(
            &mut workspace.accepted_snapshot,
            &mut workspace.trial_snapshot,
        );
        workspace
            .x
            .copy_from_slice(&workspace.accepted_snapshot.point);
        workspace.accepted_value = workspace.accepted_snapshot.original_value;
        workspace.accepted_iterations = outer + 1;
        workspace
            .constraint_values
            .copy_from_slice(&workspace.accepted_snapshot.values);
        if augmented || restore {
            workspace
                .multipliers
                .copy_from_slice(&workspace.merit_scratch[..m]);
        }
        // Exterior penalty models have no retained dual term; diagnostics use mu*c estimates only.
        workspace.penalty_parameter = next_mu;
        previous = feasibility(&workspace.accepted_snapshot.values, eq);
        report = candidate_report;
        // Report and test with the multipliers that produced `report`.
        let report_multipliers = if certified {
            &workspace.certificate_multipliers[..m]
        } else {
            &workspace.merit_scratch[..m]
        };
        workspace.accepted_diagnostics =
            original_report(&workspace.accepted_snapshot, report_multipliers, report, eq)?;
        if tolerances.satisfied(
            report,
            crate::utils::optimality_scales(
                &workspace.accepted_snapshot.point,
                workspace.accepted_snapshot.value.ok_or_else(|| {
                    SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Breakdown,
                        "missing accepted KKT objective",
                    )
                })?,
                &workspace.accepted_snapshot.gradient,
                workspace.accepted_snapshot.jacobian.view(),
                report_multipliers,
                &[],
                &[],
            )?,
        )? {
            termination = OptimizeTermination::Converged;
            break;
        }
        if !workspace.inner_result.is_success() && !restore && previous <= target {
            termination = workspace.inner_result.termination();
            break;
        }
    }
    let value = crate::utils::accepted_constrained_value(workspace)?;
    super::finish_constrained_without_gradient(
        workspace,
        output,
        value,
        workspace.accepted_iterations,
        termination,
        "constrained progress and original-problem KKT residuals assessed",
    )?;
    output.diagnostics = workspace.accepted_diagnostics;
    Ok(())
}

/// Shape of the accepted state, multipliers and nested BFGS storage.
impl<S: RealScalar + NumericScalar<Real = S>> crate::attempt::Accepted<S>
    for ConstrainedStorage<S>
{
    fn accepted(&self) -> Option<crate::attempt::AcceptedState<'_, S>> {
        Some(crate::attempt::AcceptedState {
            point: &self.x,
            value: self.accepted_value?,
            gradient: None,
            iterations: self.accepted_iterations,
        })
    }
}

/// Run one outer method in an epoch already opened for `problem`.
#[allow(clippy::too_many_arguments)]
pub(super) fn minimize_in_context<S, L>(
    inner: &Bfgs<S, L>,
    problem: &crate::ConstrainedProblem<'_, S>,
    x0: &[S],
    tolerances: OptimalityTolerances<S>,
    iterations: Budget,
    time_limit: Option<core::time::Duration>,
    augmented: bool,
    output: &mut OptimizeResult<S>,
    storage: &mut ConstrainedStorage<S>,
    context: &mut scioxide_core::SolveContext<'_, '_>,
) -> Result<(), SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S>,
    L: LineSearch<S>,
{
    if let Some(limit) = time_limit {
        context.limit_time(limit)?;
    }
    storage.accepted_value = None;
    let session = SolveSession::new(context, scioxide_core::argument!("scope"));
    // An empty collection keeps the unconstrained observation path.
    let constraints = Some(problem.constraints()).filter(|constraints| !constraints.is_empty());
    let outcome = minimize(
        inner,
        problem.objective(),
        x0,
        constraints,
        tolerances,
        iterations,
        augmented,
        output,
        storage,
        &session,
    );
    crate::utils::complete_constrained(outcome, storage, output, &session)
}

/// Declare an exterior-penalty method over the shared outer loop: its options,
/// workspace requirements, workspace and solver. `augmented` keeps PHR
/// multipliers between outer iterations.
macro_rules! outer_method {
    (
        $(#[$solver_meta:meta])*
        $solver:ident,
        $options:ident,
        $workspace:ident,
        $requirements:ident,
        $algorithm:literal,
        $augmented:literal
    ) => {
        #[doc = concat!("Sets the tolerances and outer-iteration limit of [`", stringify!($solver), "`].")]
        #[doc = ""]
        #[doc = "The defaults are the default [`OptimalityTolerances`](crate::OptimalityTolerances) and at most 100 outer iterations. See [Termination](crate#termination) for how the tolerances are scaled."]
        #[derive(Debug, Clone, Copy)]
        pub struct $options<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> {
            tolerances: $crate::OptimalityTolerances<S>,
            max_outer_iterations: scioxide_core::Budget,
            time_limit: Option<core::time::Duration>,
        }

        impl<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> Default for $options<S> {
            fn default() -> Self {
                Self {
                    tolerances: $crate::OptimalityTolerances::default(),
                    max_outer_iterations: scioxide_core::Budget::ONE.saturating_add(99),
                    time_limit: None,
                }
            }
        }

        impl<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> $options<S> {
            /// Sets the convergence tolerances.
            #[must_use]
            pub const fn with_tolerances(mut self, tolerances: $crate::OptimalityTolerances<S>) -> Self {
                self.tolerances = tolerances;
                self
            }
            /// Sets the outer-iteration limit (default 100).
            #[must_use]
            pub const fn with_max_outer_iterations(mut self, budget: scioxide_core::Budget) -> Self {
                self.max_outer_iterations = budget;
                self
            }
            /// Sets a wall-clock limit. The execution context must provide a clock.
            #[must_use]
            pub const fn with_time_limit(mut self, time_limit: core::time::Duration) -> Self {
                self.time_limit = Some(time_limit);
                self
            }
            /// Returns the convergence tolerances.
            pub const fn tolerances(&self) -> $crate::OptimalityTolerances<S> {
                self.tolerances
            }
            /// Returns the outer-iteration limit.
            pub const fn max_outer_iterations(&self) -> scioxide_core::Budget {
                self.max_outer_iterations
            }
            /// Returns the wall-clock limit, if set.
            pub const fn time_limit(&self) -> Option<core::time::Duration> {
                self.time_limit
            }
        }

        #[doc = concat!("Describes the workspace size of a [`", stringify!($solver), "`] solve.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $requirements {
            dimension: usize,
            equalities: usize,
            inequalities: usize,
        }

        impl $requirements {
            /// Returns the number of variables.
            pub const fn dimension(self) -> usize {
                self.dimension
            }
            /// Returns the number of equality rows.
            pub const fn equalities(self) -> usize {
                self.equalities
            }
            /// Returns the number of inequality rows. Each finite bound side
            /// counts as one row.
            pub const fn inequalities(self) -> usize {
                self.inequalities
            }
        }

        impl<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>>
            $crate::attempt::Shape<$crate::constrained::ConstrainedStorage<S>> for $requirements
        {
            fn fits(&self, storage: &$crate::constrained::ConstrainedStorage<S>) -> bool {
                storage.fits(self.dimension, self.equalities, self.inequalities)
            }
            fn prepare(
                &self,
                storage: &mut $crate::constrained::ConstrainedStorage<S>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                storage.try_reserve_nested_bfgs(self.dimension, self.equalities, self.inequalities)
            }
        }

        $crate::attempt::local_workspace!(
            #[doc = concat!("Stores the iterates, multipliers and inner BFGS state of a [`", stringify!($solver), "`] solve.")]
            $workspace,
            $crate::constrained::ConstrainedStorage<S>,
            $requirements
        );

        $(#[$solver_meta])*
        #[derive(Debug, Clone)]
        pub struct $solver<
            S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
            L = $crate::StrongWolfeLineSearch<S>,
        > {
            options: $options<S>,
            inner: $crate::Bfgs<S, L>,
        }

        impl<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> Default for $solver<S> {
            fn default() -> Self {
                Self::new($options::default())
            }
        }

        impl<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> $solver<S> {
            /// Creates a solver with `options` and a default [`Bfgs`](crate::Bfgs)
            /// inner solver.
            #[must_use]
            pub fn new(options: $options<S>) -> Self {
                Self {
                    options,
                    inner: $crate::Bfgs::default(),
                }
            }
        }

        impl<S, L> $solver<S, L>
        where
            S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
            L: $crate::LineSearch<S>,
        {
            /// Sets the [`Bfgs`](crate::Bfgs) solver used for each subproblem.
            ///
            /// Its tolerances apply to the subproblems; the outer stationarity
            /// tolerance caps its gradient tolerance.
            #[must_use]
            pub fn with_inner_optimizer<L2>(self, inner: $crate::Bfgs<S, L2>) -> $solver<S, L2> {
                $solver {
                    options: self.options,
                    inner,
                }
            }

            /// Returns the options.
            pub const fn options(&self) -> &$options<S> {
                &self.options
            }

            /// Returns the workspace size needed for `problem`.
            ///
            /// # Errors
            ///
            /// This method does not fail.
            pub fn workspace_requirements(
                &self,
                problem: &$crate::ConstrainedProblem<'_, S>,
            ) -> Result<$requirements, scioxide_core::SciOxideError> {
                let profile = problem.constraints().profile();
                Ok($requirements {
                    dimension: problem.dim(),
                    equalities: profile.equality_count(),
                    inequalities: profile.inequality_count(),
                })
            }

            /// Allocates a workspace for `problem`.
            ///
            /// # Errors
            ///
            /// - `SizeOverflow` or `AllocationFailure` if the storage cannot be allocated.
            pub fn workspace(
                &self,
                problem: &$crate::ConstrainedProblem<'_, S>,
            ) -> Result<$workspace<S>, scioxide_core::SciOxideError> {
                let mut workspace = $workspace::new();
                workspace.try_reserve(&self.workspace_requirements(problem)?)?;
                Ok(workspace)
            }

            /// Minimizes `problem` from `x0` and writes the result to `output`.
            ///
            /// Each BFGS subproblem runs as a nested solve of `scope`. See
            /// [Reusing plans and workspaces](crate#reusing-plans-and-workspaces).
            ///
            /// # Errors
            ///
            /// - `DimensionMismatch` if `x0` does not have the problem's dimension.
            /// - `NonFinite` if `x0` contains NaN or infinity, or if the objective
            ///   or a constraint is not finite at `x0`.
            /// - `CapabilityUnavailable` if a time limit is set and `scope` has no
            ///   clock.
            /// - Errors from the callbacks, unchanged.
            /// - `ConvergenceFailure` or `ExecutionStopped` if a limit,
            ///   cancellation or deadline stops the solve before the first
            ///   accepted point, as described in [Termination](crate#termination).
            pub fn minimize_into<'scope, 'capability>(
                &self,
                problem: &$crate::ConstrainedProblem<'_, S>,
                x0: &[S],
                output: &mut $crate::OptimizeResult<S>,
                workspace: &mut $workspace<S>,
                scope: impl Into<scioxide_core::SolveScope<'scope, 'capability>>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                workspace.shell.reject(scioxide_core::AttemptKind::Call);
                let scope = scope.into();
                $crate::utils::validate_x0(problem.objective(), x0)?;
                workspace.try_reserve(&self.workspace_requirements(problem)?)?;
                $crate::attempt::preflight(&scope, self.options.time_limit)?;
                let entry = $crate::attempt::Entry::Call {
                    identity: self.identity(problem),
                    budgets: scioxide_core::SolveBudgets::unbounded(),
                };
                $crate::attempt::minimize(
                    &mut workspace.shell,
                    &mut workspace.storage,
                    &mut workspace.staged,
                    output,
                    scope,
                    entry,
                    |context, storage, staged| {
                        $crate::constrained::outer::minimize_in_context(
                            &self.inner,
                            problem,
                            x0,
                            self.options.tolerances,
                            self.options.max_outer_iterations,
                            self.options.time_limit,
                            $augmented,
                            staged,
                            storage,
                            context,
                        )
                    },
                )
            }

            /// Identify method, inner optimizer, constraint shape and options for an epoch.
            pub(crate) fn identity(
                &self,
                problem: &$crate::ConstrainedProblem<'_, S>,
            ) -> scioxide_core::SolveIdentity {
                $crate::context::local_identity(
                    scioxide_core::algorithm_id!("optimize", $algorithm, 2),
                    problem.dim(),
                    &(
                        &self.options,
                        core::any::type_name::<L>(),
                        &self.inner,
                        problem.constraints().profile(),
                    ),
                )
            }
        }

        impl<S, L> $solver<S, L>
        where
            S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
            L: $crate::LineSearch<S>,
        {
            /// Minimizes the objective with this method's settings and owned workspace.
            ///
            /// Returns the complete result, including unsuccessful termination. Call
            /// `require_success()` to require the stopping test to pass.
            ///
            /// # Errors
            ///
            /// Returns the input, callback, allocation and numerical errors of `minimize_into`.
            pub fn minimize(
                &self,
                problem: &$crate::ConstrainedProblem<'_, S>,
                x0: &[S],
            ) -> Result<$crate::OptimizeResult<S>, scioxide_core::SciOxideError> {
                let mut workspace = self.workspace(problem)?;
                let mut output = $crate::OptimizeResult::empty();
                self.minimize_into(
                    problem,
                    x0,
                    &mut output,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )?;
                Ok(output)
            }
        }

        impl<'a, S, L> $crate::Optimizer<S, $crate::ConstrainedProblem<'a, S>> for $solver<S, L>
        where
            S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
            L: $crate::LineSearch<S>,
        {
            fn minimize(
                &self,
                problem: &$crate::ConstrainedProblem<'a, S>,
                x0: &[S],
            ) -> Result<$crate::OptimizeResult<S>, scioxide_core::SciOxideError> {
                self.minimize(problem, x0)
            }
        }
    };
}
pub(super) use outer_method;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inactive_phr_term_is_continuous_and_has_the_correct_derivative() {
        for c in [-1.01_f64, -1.0, -0.99] {
            let (value, derivative) = term(c, 2.0, 2.0, false).unwrap();
            let oracle = ((2.0 + 2.0 * c).max(0.0).powi(2) - 4.0) / 4.0;
            assert!((value - oracle).abs() < 1e-14);
            let h = 1e-6;
            let difference = (term(c + h, 2.0, 2.0, false).unwrap().0
                - term(c - h, 2.0, 2.0, false).unwrap().0)
                / (2.0 * h);
            assert!((derivative - difference).abs() < 1e-6);
        }
    }

    #[test]
    fn progress_controls_penalty_and_restoration() {
        assert_eq!(
            update(2.0_f64, 0.9, 1e-6, false, 0).unwrap(),
            Update::Retain
        );
        assert_eq!(update(2.0_f64, 1.9, 1e-6, true, 0).unwrap(), Update::Grow);
        assert_eq!(
            update(2.0_f64, 1.9, 1e-6, false, 0).unwrap(),
            Update::Restore
        );
        assert_eq!(
            update(2.0_f64, 1.9, 1e-6, true, 1).unwrap(),
            Update::Restore
        );
        assert_eq!(
            update(2.0_f32, 0.0, 1e-4, false, 1).unwrap(),
            Update::Retain
        );
    }
}
