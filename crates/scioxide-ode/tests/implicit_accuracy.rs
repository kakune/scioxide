//! Independent inner-accuracy, resource, and provider tests.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, ErrorTolerance, EvaluationRole, ExecutionContext,
    NormSelection, NumericScalar, RealScalar, SciOxideError, SolveContext,
};
use scioxide_linalg::{Matrix, MatrixViewMut};
use scioxide_ode::{
    BackwardEuler, NewtonMode, OdeFn, OdeMethod, OdeOptions, OdeProblem, OdeSolveReport, OdeState,
    OdeSystem, OdeTermination, OdeWorkspace, StageAccuracy,
};

/// Inner iteration limit of one stage solve used by the standalone accuracy policies.
const INNER_LIMIT: usize = 20;

type Attempt<S, M> = (
    Result<OdeSolveReport<S>, SciOxideError>,
    OdeState<S>,
    OdeWorkspace<S, M>,
);

/// Integrate `problem` from `(0, initial)` with `options` on a prepared workspace.
fn advance<S, M, Sys>(
    method: M,
    options: OdeOptions<S>,
    problem: &OdeProblem<'_, S, Sys>,
    initial: &[S],
) -> Attempt<S, M>
where
    S: RealScalar + NumericScalar<Real = S>,
    M: OdeMethod<S> + Clone,
    Sys: OdeSystem<S> + ?Sized,
{
    let plan = (method, options);
    let mut workspace = plan.0.workspace(problem, &plan.1).unwrap();
    let mut state =
        OdeState::from_slice(<S as scioxide_core::AlgebraicScalar>::ZERO, initial).unwrap();
    let report = plan.0.advance(
        problem,
        &plan.1,
        &mut state,
        &mut (),
        &mut workspace,
        ExecutionContext::sequential(),
    );
    (report, state, workspace)
}

/// One step of size `h` from time zero; the core default tolerance matches a bare
/// stage request.
fn one_step<S, M, Sys>(method: M, system: &Sys, initial: &[S], h: S) -> Attempt<S, M>
where
    S: RealScalar + NumericScalar<Real = S>,
    M: OdeMethod<S> + Clone,
    Sys: OdeSystem<S> + ?Sized,
{
    let options = OdeOptions::new()
        .with_error_tolerance(ErrorTolerance::default())
        .with_initial_step(h);
    advance(method, options, &OdeProblem::new(system, h), initial)
}

fn tightening<S: RealScalar + NumericScalar<Real = S>>() {
    let scales = [
        S::try_from_f64_lossy(1.0e12).unwrap(),
        S::try_from_f64_lossy(1.0e-6).unwrap(),
    ];
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let two = one + one;
    let dt = S::try_from_f64_lossy(0.2).unwrap();
    let system = OdeFn::new(2, |_: S, state: &[S], output: &mut [S]| {
        for ((value, &state), &scale) in output.iter_mut().zip(state).zip(&scales) {
            *value = -state * (state / scale);
        }
        Ok(())
    })
    .unwrap()
    .with_jacobian(|_: S, state: &[S], output: &mut MatrixViewMut<'_, S>| {
        for row in 0..2 {
            for col in 0..2 {
                output.set(
                    row,
                    col,
                    if row == col {
                        -two * (state[row] / scales[row])
                    } else {
                        <S as scioxide_core::AlgebraicScalar>::ZERO
                    },
                )?;
            }
        }
        Ok(())
    });
    // Independent positive root of h*u^2 + u - 1 = 0.
    let exact = two / (one + (one + (two + two) * dt).sqrt());
    for mode in [NewtonMode::Simplified, NewtonMode::Full] {
        let mut errors = Vec::new();
        for tolerance in [1.0e-2, 1.0e-5] {
            let tolerance = S::try_from_f64_lossy(tolerance).unwrap();
            let policy = ErrorTolerance::try_vector(
                &scales.map(|scale| scale * tolerance),
                <S as scioxide_core::AlgebraicScalar>::ZERO,
                NormSelection::Infinity,
            )
            .unwrap();
            let options = OdeOptions::new()
                .with_error_tolerance(policy)
                .with_initial_step(dt);
            let (report, state, _) = advance(
                BackwardEuler::new().with_newton_mode(mode),
                options,
                &OdeProblem::new(&system, dt),
                &scales,
            );
            assert_eq!(report.unwrap().accepted_steps(), 1);
            let error = state
                .values()
                .iter()
                .zip(&scales)
                .map(|(&value, &scale)| scioxide_core::NumericScalar::abs(value / scale - exact))
                .fold(<S as scioxide_core::AlgebraicScalar>::ZERO, |a, b| a.max(b));
            assert!(error <= tolerance);
            errors.push(error);
        }
        // Converged full Newton can reach roundoff at both tolerances.
        let roundoff = S::try_from_f64_lossy(4.0).unwrap() * S::EPSILON;
        assert!(
            errors[1] < errors[0] || errors[1] <= roundoff,
            "{mode:?}: {errors:?}"
        );
    }
}
#[test]
fn tightening_f32() {
    tightening::<f32>();
}
#[test]
fn tightening_f64() {
    tightening::<f64>();
}

struct OperatorDecay<'a> {
    jacobian: &'a dyn scioxide_core::LinearOperator<f64>,
}
impl OdeSystem<f64> for OperatorDecay<'_> {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: f64,
        state: &[f64],
        output: &mut [f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        output[0] = -state[0];
        output[1] = -100.0 * state[1];
        Ok(())
    }
    fn jacobian_operator(
        &self,
        _: f64,
        _: &[f64],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<&dyn scioxide_core::LinearOperator<f64>>, SciOxideError> {
        Ok(Some(self.jacobian))
    }
}
#[test]
fn borrowed_operators_match_dense_backward_euler_without_materialization() {
    let dense = Matrix::from_vec(2, 2, vec![-1.0, 0.0, 0.0, -100.0]).unwrap();
    let view = dense.view();
    let sparse =
        scioxide_sparse::CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![-1.0, -100.0])
            .unwrap();
    let closure = scioxide_core::ClosureOperator::new(2, 2, |input: &[f64], output: &mut [f64]| {
        output[0] = -input[0];
        output[1] = -100.0 * input[1];
        Ok(())
    });
    for operator in [
        &dense.view().try_finite().unwrap() as &dyn scioxide_core::LinearOperator<f64>,
        &view.try_finite().unwrap(),
        &sparse,
        &closure,
    ] {
        let system = OperatorDecay { jacobian: operator };
        let solver = BackwardEuler::new().with_backend(scioxide_ode::OdeStageBackend::Krylov {
            iterations: Budget::try_from(20).unwrap(),
            restart: Budget::try_from(4).unwrap(),
            operator_scratch: operator.apply_workspace_len().unwrap(),
            preconditioner_scratch: 0,
        });
        let (report, state, _) = one_step(solver, &system, &[1.0, 1.0], 0.1);
        assert!(report.unwrap().is_success());
        assert!((state.values()[0] - 1.0 / 1.1).abs() < 1.0e-12);
        assert!((state.values()[1] - 1.0 / 11.0).abs() < 1.0e-12);
    }
}

#[test]
fn inner_iteration_limit_preserves_state() {
    // An inconsistent analytic Jacobian slows full Newton to a linear contraction of
    // about 0.95 per iteration, so the 20-iteration inner limit of backward Euler is
    // exhausted long before the corrections reach the stage tolerance.
    let system = OdeFn::new(1, |_: f64, state: &[f64], output: &mut [f64]| {
        output[0] = -state[0] * state[0];
        Ok(())
    })
    .unwrap()
    .with_jacobian(|_: f64, _: &[f64], output: &mut MatrixViewMut<'_, f64>| {
        output.set(0, 0, -129.0)
    });
    let (report, state, workspace) = one_step(
        BackwardEuler::new().with_newton_mode(NewtonMode::Full),
        &system,
        &[1.0],
        0.2,
    );
    let error = report.unwrap_err();
    assert!(
        matches!(
            (error).view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::IterationLimit,
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(state.time(), 0.0);
    assert_eq!(state.values(), &[1.0]);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    assert_eq!(attempt.progress().unwrap().state(), &[1.0]);
    assert_eq!(attempt.counts().nested().iterations(), INNER_LIMIT);
}

struct FalseSuccess;
impl scioxide_ode::NonlinearStageSolver<f64> for FalseSuccess {
    fn solve_in_place(
        &self,
        _: &mut dyn FnMut(
            &[f64],
            &mut [f64],
            &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError>,
        _operator: &dyn scioxide_core::LinearOperator<f64>,
        _: &mut [f64],
        _: &mut [f64],
        _: &mut [f64],
        _: StageAccuracy<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<scioxide_ode::NonlinearStageReport, SciOxideError> {
        Ok(scioxide_ode::NonlinearStageReport::new(0, 0, 0))
    }
}
impl scioxide_ode::StageLinearSolver<f64> for FalseSuccess {
    fn solve_into(
        &self,
        _: &dyn scioxide_core::LinearOperator<f64>,
        _: &[f64],
        _: &[f64],
        correction: &mut [f64],
        _: StageAccuracy<'_, f64>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        correction.fill(0.0);
        Ok(())
    }
}
#[test]
fn unearned_inner_success_is_rejected_before_commit() {
    let system = OdeFn::new(1, |_: f64, state: &[f64], output: &mut [f64]| {
        output[0] = -state[0];
        Ok(())
    })
    .unwrap();
    for linear in [false, true] {
        let problem = OdeProblem::new(&system, 0.1);
        let problem = if linear {
            problem.with_stage_solver(scioxide_ode::OdeStageSolver::Linear(&FalseSuccess))
        } else {
            problem.with_stage_solver(scioxide_ode::OdeStageSolver::Nonlinear(&FalseSuccess))
        };
        let options = OdeOptions::new()
            .with_error_tolerance(ErrorTolerance::default())
            .with_initial_step(0.1);
        let (result, state, workspace) = advance(BackwardEuler::new(), options, &problem, &[1.0]);
        assert!(matches!(
            (result).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::ConvergenceFailure {
                reason: scioxide_core::ConvergenceReason::NestedSolver,
                ..
            }
        ));
        assert_eq!(state.values(), &[1.0]);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Failed
        );
    }
}

#[test]
fn managed_inner_work_respects_shared_budgets() {
    let system = OdeFn::new(1, |_: f64, state: &[f64], output: &mut [f64]| {
        output[0] = -state[0] * state[0];
        Ok(())
    })
    .unwrap();
    let problem = OdeProblem::new(&system, 1.0);
    let options = OdeOptions::new()
        .with_initial_step(0.1)
        .with_max_steps(Budget::try_from(2).unwrap());
    let (report, state, _) = advance(BackwardEuler::new(), options, &problem, &[1.0]);
    let report = report.unwrap();
    assert_eq!(
        report.termination(),
        Some(OdeTermination::BudgetLimit(BudgetKind::Proposal))
    );
    // Two admitted outer steps of 0.1 finish their own inner Newton solves.
    assert_eq!(report.final_time(), 0.2);
    assert_eq!(state.time(), 0.2);
    assert_eq!(report.counts().iterations(), 2);
    assert!(report.accounting().nested().iterations() > 2);
    for limit in 1..=8 {
        let options = OdeOptions::new()
            .with_initial_step(0.1)
            .with_max_evaluations(Budget::try_from(limit).unwrap());
        let (report, state, _) = advance(BackwardEuler::new(), options, &problem, &[1.0]);
        let report = report.unwrap();
        assert_eq!(
            report.termination(),
            Some(OdeTermination::BudgetLimit(BudgetKind::Evaluation))
        );
        assert!(report.counts().total_providers().unwrap() <= limit);
        assert_eq!(report.final_time(), 0.0);
        assert_eq!(state.values(), &[1.0]);
    }
}

fn extreme_affine<S: RealScalar + NumericScalar<Real = S>>() {
    let large = <S as RealScalar>::MAX * S::try_from_f64_lossy(0.75).unwrap();
    let system = OdeFn::new(1, |_: S, _: &[S], output: &mut [S]| {
        output[0] = large;
        Ok(())
    })
    .unwrap()
    .with_jacobian(|_: S, _: &[S], output: &mut MatrixViewMut<'_, S>| {
        output.set(0, 0, <S as scioxide_core::AlgebraicScalar>::ZERO)
    });
    let (report, state, _) = one_step(
        BackwardEuler::new(),
        &system,
        &[-large],
        S::try_from_f64_lossy(2.0).unwrap(),
    );
    assert!(report.unwrap().is_success());
    assert_eq!(state.values(), &[large]);
}
#[test]
fn stage_cancellation_preserves_f32_range() {
    extreme_affine::<f32>();
}
#[test]
fn stage_cancellation_preserves_f64_range() {
    extreme_affine::<f64>();
}

struct ScaledMass<S> {
    mass: [S; 2],
}
impl<S: RealScalar + NumericScalar<Real = S>> OdeSystem<S> for ScaledMass<S> {
    fn dimension(&self) -> usize {
        2
    }
    fn rhs(
        &self,
        _: S,
        state: &[S],
        output: &mut [S],
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        for ((output, &state), &mass) in output.iter_mut().zip(state).zip(&self.mass) {
            *output = -mass * state;
        }
        Ok(())
    }
    fn jacobian(
        &self,
        _: S,
        _: &[S],
        output: &mut MatrixViewMut<'_, S>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        output.set(0, 0, -self.mass[0])?;
        output.set(1, 1, -self.mass[1])?;
        output.set(0, 1, <S as scioxide_core::AlgebraicScalar>::ZERO)?;
        output.set(1, 0, <S as scioxide_core::AlgebraicScalar>::ZERO)?;
        Ok(Some(()))
    }
    fn mass_matrix_revision(&self) -> Option<u64> {
        Some(1)
    }
    fn mass_matrix(
        &self,
        output: &mut MatrixViewMut<'_, S>,
        _: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        output.set(0, 0, self.mass[0])?;
        output.set(1, 1, self.mass[1])?;
        output.set(0, 1, <S as scioxide_core::AlgebraicScalar>::ZERO)?;
        output.set(1, 0, <S as scioxide_core::AlgebraicScalar>::ZERO)
    }
}
fn mass_scaling<S: RealScalar + NumericScalar<Real = S>, M: OdeMethod<S> + Clone>(
    factory: impl Fn() -> M,
) {
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let h = S::try_from_f64_lossy(0.1).unwrap();
    // A loose tolerance accepts the single step of the adaptive methods at `h`; both
    // mass scalings run the same iteration.
    let loose = S::try_from_f64_lossy(0.1).unwrap();
    let options = OdeOptions::new()
        .with_error_tolerance(
            ErrorTolerance::scalar(loose, loose, NormSelection::RootMeanSquare).unwrap(),
        )
        .with_initial_step(h)
        .with_max_steps(Budget::ONE);
    let mut reference = [one, one + one];
    for (index, mass) in [
        [one, one],
        [
            S::try_from_f64_lossy(1.0e30).unwrap(),
            S::try_from_f64_lossy(1.0e-30).unwrap(),
        ],
    ]
    .into_iter()
    .enumerate()
    {
        let system = ScaledMass { mass };
        let (report, state, _) = advance(
            factory(),
            options.clone(),
            &OdeProblem::new(&system, h),
            &[one, one + one],
        );
        assert_eq!(
            report.unwrap().termination(),
            Some(OdeTermination::ReachedEnd)
        );
        if index == 0 {
            reference.copy_from_slice(state.values());
        }
        for (&actual, expected) in state.values().iter().zip(reference) {
            assert!(
                scioxide_core::NumericScalar::abs((actual - expected) / expected)
                    < S::try_from_f64_lossy(128.0).unwrap() * <S as RealScalar>::EPSILON
            );
        }
    }
}
fn mass_families<S: RealScalar + NumericScalar<Real = S>>() {
    mass_scaling::<S, _>(BackwardEuler::new);
    mass_scaling::<S, _>(scioxide_ode::ImplicitTrapezoid::new);
    mass_scaling::<S, _>(scioxide_ode::GaussRk::new);
    mass_scaling::<S, _>(scioxide_ode::RadauIia::new);
    mass_scaling::<S, _>(scioxide_ode::Sdirk::new);
    mass_scaling::<S, _>(|| scioxide_ode::Bdf::new(1).unwrap());
}
#[test]
fn mass_scaled_stages_f32() {
    mass_families::<f32>();
}
#[test]
fn mass_scaled_stages_f64() {
    mass_families::<f64>();
}

#[test]
fn checked_borrowed_input_failures_have_input_provenance() {
    let calls = std::cell::Cell::new(0);
    let system = OdeFn::new(1, |_: f64, _: &[f64], output: &mut [f64]| {
        calls.set(calls.get() + 1);
        output[0] = 1.0;
        Ok(())
    })
    .unwrap();
    let (result, state, workspace) = one_step(BackwardEuler::new(), &system, &[f64::NAN], 0.1);
    assert!(
        matches!(&result, Err(error) if matches!(error.view(), scioxide_core::SciOxideErrorView::NonFinite { origin: scioxide_core::NonFiniteOrigin::Input, .. }))
    );
    assert_eq!(calls.get(), 0);
    assert!(state.values()[0].is_nan());
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
}

#[test]
fn completing_at_the_exact_provider_budget_commits_without_duplicate_charges() {
    let system = OdeFn::new(1, |_: f64, state: &[f64], output: &mut [f64]| {
        output[0] = -state[0];
        Ok(())
    })
    .unwrap()
    .with_jacobian(|_: f64, _: &[f64], out: &mut MatrixViewMut<'_, f64>| out.set(0, 0, -1.0));
    let problem = OdeProblem::new(&system, 0.1);
    let options = OdeOptions::new().with_initial_step(0.1);
    let (unbounded, expected_state, _) =
        advance(BackwardEuler::new(), options.clone(), &problem, &[1.0]);
    let unbounded = unbounded.unwrap().accounting().total().unwrap();
    let roles = [
        EvaluationRole::RightHandSide,
        EvaluationRole::Jacobian,
        EvaluationRole::Factorization,
        EvaluationRole::LinearSolve,
    ];
    let expected = roles.map(|role| unbounded.provider(role));
    assert!(expected.iter().all(|&count| count > 0), "{expected:?}");
    let total = unbounded.total_providers().unwrap();
    assert_eq!(expected.iter().sum::<usize>(), total);
    let (report, state, _) = advance(
        BackwardEuler::new(),
        options.with_max_evaluations(Budget::try_from(total).unwrap()),
        &problem,
        &[1.0],
    );
    let report = report.unwrap();
    assert_eq!(report.termination(), Some(OdeTermination::ReachedEnd));
    assert_eq!(state.values(), expected_state.values());
    for (role, count) in roles.into_iter().zip(expected) {
        assert_eq!(report.accounting().total().unwrap().provider(role), count);
    }
    assert_eq!(
        report
            .accounting()
            .total()
            .unwrap()
            .total_providers()
            .unwrap(),
        total
    );
}
