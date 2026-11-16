#![allow(clippy::unwrap_used)]
use super::*;
use crate::{ConstraintFn, Constraints, ObjectiveFunction};
use alloc::sync::Arc;
use alloc::vec;
use core::sync::atomic::{AtomicUsize, Ordering};
use scioxide_core::{Bounds, EvaluationRole, ExecutionContext};

struct Quadratic;
impl ObjectiveFunction<f64> for Quadratic {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] * x[0] + x[1] * x[1])
    }
    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output[0] = 2.0 * x[0];
        output[1] = 2.0 * x[1];
        Ok(true)
    }
}

/// An empty constraint collection: the unconstrained problem of dimension `n`.
fn free<S: RealScalar + NumericScalar<Real = S>>(n: usize) -> Constraints<S> {
    Constraints::new(n).unwrap()
}

/// Prepared call in a root epoch with sequential portable dispatch.
fn solve<S: RealScalar + NumericScalar<Real = S>>(
    solver: &Sqp<S>,
    problem: &ConstrainedProblem<'_, S>,
    x0: &[S],
    output: &mut OptimizeResult<S>,
    workspace: &mut SqpWorkspace<S>,
) -> Result<(), SciOxideError> {
    solver.minimize_into(
        problem,
        x0,
        output,
        workspace,
        ExecutionContext::sequential(),
    )
}

/// One epoch rooted in `execution` with caller budgets; returns the written
/// result and the epoch's counts.
fn root<S: RealScalar + NumericScalar<Real = S>>(
    solver: &Sqp<S>,
    problem: &ConstrainedProblem<'_, S>,
    x0: &[S],
    execution: ExecutionContext<'_>,
    mut budgets: SolveBudgets,
) -> (
    Result<OptimizeResult<S>, SciOxideError>,
    scioxide_core::SolveCounts,
) {
    let mut workspace = solver.workspace(problem).unwrap();
    let mut output = OptimizeResult::empty();
    let mut state = SolveState::new();
    let mut context = state
        .begin_solve(execution, solver.identity(problem), &mut budgets)
        .unwrap();
    let result = solver
        .minimize_in_context(
            problem,
            x0,
            &mut output,
            &mut workspace.storage,
            &mut context,
        )
        .map(|()| output);
    let counts = context.counts();
    (result, counts)
}

#[test]
fn prepared_equality_problem_converges_and_counts_kkt_work() {
    let equality = ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1] - 1.0)).with_jacobian(
        |_: &[f64], output| {
            output.copy_from_slice(&[1.0, 1.0]);
            Ok(())
        },
    );
    let constraints = Constraints::new(2).unwrap().equality(equality).unwrap();
    let problem = ConstrainedProblem::new(&Quadratic, &constraints).unwrap();
    let solver = Sqp::new(SqpOptions::default());
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut output = OptimizeResult::empty();
    solve(&solver, &problem, &[2.0, 2.0], &mut output, &mut workspace).unwrap();
    assert!(
        output.is_success(),
        "{}; x={:?}; diagnostics={:?}",
        output.message,
        output.x,
        output.diagnostics
    );
    assert!((output.x[0] - 0.5).abs() < 1e-5);
    assert!((output.x[1] - 0.5).abs() < 1e-5);
    let counts = workspace.last_attempt().unwrap().counts().total().unwrap();
    assert!(counts.provider(EvaluationRole::Factorization) > 0);
    assert!(
        counts.provider(EvaluationRole::Factorization)
            >= counts.provider(EvaluationRole::LinearSolve)
    );
}

#[test]
fn inequality_and_unconstrained_paths_converge() {
    let inequality = ConstraintFn::new(2, |x: &[f64]| Ok(1.0 - x[0] - x[1])).with_jacobian(
        |_: &[f64], output| {
            output.copy_from_slice(&[-1.0, -1.0]);
            Ok(())
        },
    );
    let constraints = Constraints::new(2).unwrap().inequality(inequality).unwrap();
    let problem = ConstrainedProblem::new(&Quadratic, &constraints).unwrap();
    let solver = Sqp::new(SqpOptions::default());
    let output = solver.minimize(&problem, &[2.0, 2.0]).unwrap();
    assert!(output.is_success(), "{}", output.message);
    assert!((output.fun - 0.5).abs() < 1e-4);

    let free = free(2);
    let problem = ConstrainedProblem::new(&Quadratic, &free).unwrap();
    let output = Sqp::<f64>::default()
        .minimize(&problem, &[2.0, 2.0])
        .unwrap();
    assert!(output.is_success());
    assert!(output.fun < 1e-10);
}

#[test]
fn supports_f32_and_rejects_dimension_mismatch() {
    struct QuadraticF32;
    impl ObjectiveFunction<f32> for QuadraticF32 {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f32]) -> Result<f32, SciOxideError> {
            Ok(x[0] * x[0] + x[1] * x[1])
        }
        fn gradient_into(&self, x: &[f32], output: &mut [f32]) -> Result<bool, SciOxideError> {
            output[0] = 2.0 * x[0];
            output[1] = 2.0 * x[1];
            Ok(true)
        }
    }
    let free32 = free(2);
    let problem = ConstrainedProblem::new(&QuadraticF32, &free32).unwrap();
    assert!(
        Sqp::<f32>::default()
            .minimize(&problem, &[2.0, 2.0])
            .unwrap()
            .is_success()
    );
    let free64 = free(2);
    let problem = ConstrainedProblem::new(&Quadratic, &free64).unwrap();
    assert!(Sqp::<f64>::default().minimize(&problem, &[1.0]).is_err());
    let mismatched = free::<f64>(1);
    assert!(ConstrainedProblem::new(&Quadratic, &mismatched).is_err());
}

#[test]
fn inequality_merit_derivative_and_infeasible_small_step_are_non_success() {
    let jacobian = scioxide_linalg::Matrix::from_vec(1, 1, vec![-1.0_f64]).unwrap();
    let mut observation = Observation::new();
    observation.reserve(1, 1).unwrap();
    observation.gradient[0] = 0.0;
    observation.values[0] = 1.0;
    observation.jacobian = jacobian;
    let slope = directional(&observation, &[1.0], 0, 3.0).unwrap();
    assert_eq!(slope, -3.0);

    struct Constant;
    impl ObjectiveFunction<f64> for Constant {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, _: &[f64]) -> Result<f64, SciOxideError> {
            Ok(0.0)
        }
        fn gradient_into(&self, _: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output[0] = 0.0;
            Ok(true)
        }
    }
    let impossible = ConstraintFn::new(1, |x: &[f64]| Ok(x[0] * x[0] + 1.0)).with_jacobian(
        |x: &[f64], output| {
            output.copy_from_slice(&[2.0 * x[0]]);
            Ok(())
        },
    );
    let constraints = Constraints::new(1).unwrap().inequality(impossible).unwrap();
    let problem = ConstrainedProblem::new(&Constant, &constraints).unwrap();
    let solver = Sqp::new(SqpOptions::default());
    let result = solver.minimize(&problem, &[0.0]).unwrap();

    assert_eq!(result.termination(), OptimizeTermination::Stagnation);
    assert!(!result.is_success());
    assert!(result.primal_feasibility().is_some_and(|value| value > 0.0));
}

macro_rules! bounded_case {
    ($name:ident,$s:ty,$tol:expr) => {
        #[test]
        fn $name() {
            struct Objective;
            impl ObjectiveFunction<$s> for Objective {
                fn dim(&self) -> usize {
                    2
                }
                fn eval(&self, x: &[$s]) -> Result<$s, SciOxideError> {
                    assert!((0.0..=2.0).contains(&x[0]) && (-1.0..=1.0).contains(&x[1]));
                    Ok((x[0] + 1.0).powi(2) + (x[1] - 3.0).powi(2))
                }
                fn gradient_into(&self, x: &[$s], o: &mut [$s]) -> Result<bool, SciOxideError> {
                    o[0] = 2.0 * (x[0] + 1.0);
                    o[1] = 2.0 * (x[1] - 3.0);
                    Ok(true)
                }
            }
            let bounds =
                Bounds::new(vec![(Some(0.0), Some(2.0)), (Some(-1.0), Some(1.0))]).unwrap();
            let c = Constraints::new(2).unwrap().bounds(bounds).unwrap();
            let problem = ConstrainedProblem::new(&Objective, &c).unwrap();
            let solver = Sqp::new(SqpOptions::default());
            let mut w = solver.workspace(&problem).unwrap();
            let mut r = OptimizeResult::empty();
            solve(&solver, &problem, &[-10.0, 20.0], &mut r, &mut w).unwrap();
            assert!(r.is_success(), "{r:?}");
            assert!(r.x()[0].abs() < $tol && (r.x()[1] - 1.0).abs() < $tol);
            assert!((w.state().lower_multipliers()[0] - 2.0).abs() < $tol);
            assert!((w.state().upper_multipliers()[1] - 4.0).abs() < $tol);
        }
    };
}
bounded_case!(bounds_f64, f64, 1e-5);
bounded_case!(bounds_f32, f32, 2e-4);

#[test]
fn finite_differences_stay_inside_bounds_and_skip_fixed_coordinates() {
    struct Objective;
    impl ObjectiveFunction<f64> for Objective {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            assert_eq!(x[0], 0.5);
            assert!((0.0..=1.0).contains(&x[1]));
            Ok((x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2))
        }
    }
    let c = Constraints::new(2)
        .unwrap()
        .bounds(Bounds::new(vec![(Some(0.5), Some(0.5)), (Some(0.0), Some(1.0))]).unwrap())
        .unwrap();
    let problem = ConstrainedProblem::new(&Objective, &c).unwrap();
    let r = Sqp::new(SqpOptions::default())
        .minimize(&problem, &[3.0, -4.0])
        .unwrap();
    assert!(r.is_success(), "{r:?}");
    assert_eq!(r.x()[0], 0.5);
    assert!((r.x()[1] - 1.0).abs() < 1e-5);
    assert!(!sqp_of(&r).fixed_derivatives_available());
    assert!(r.grad().is_none());
}

struct SumEquality;
impl crate::ConstraintFunction<f64> for SumEquality {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] + x[1] - 1.0)
    }
    fn jacobian_into(&self, _: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
        out.fill(1.0);
        Ok(true)
    }
}
#[test]
fn prepared_solves_are_allocation_free_and_warm_state_is_explicit() {
    let c = Constraints::new(2).unwrap().equality(SumEquality).unwrap();
    let problem = ConstrainedProblem::new(&Quadratic, &c).unwrap();
    let solver = Sqp::new(SqpOptions::default());
    let mut w = solver.workspace(&problem).unwrap();
    let mut r = OptimizeResult::empty();
    solve(&solver, &problem, &[2.0, 2.0], &mut r, &mut w).unwrap();
    let expected = r.clone();
    let cold_counts = r.counts();
    let state = w.state().clone();
    solve(&solver, &problem, &[2.0, 2.0], &mut r, &mut w).unwrap();
    assert_eq!(r.x(), expected.x());
    assert_eq!(r.counts(), cold_counts);
    assert!(
        w.last_attempt()
            .unwrap()
            .counts()
            .nested()
            .total_providers()
            .unwrap()
            > 0
    );
    w.set_warm_start(&state).unwrap();
    solve(&solver, &problem, expected.x(), &mut r, &mut w).unwrap();
    assert!(r.is_success());
    assert_eq!(r.iterations(), 0);
    let mut malformed = SqpState::new();
    malformed.reserve(1, 1).unwrap();
    w.set_warm_start(&malformed).unwrap();
    assert!(solve(&solver, &problem, &[2.0, 2.0], &mut r, &mut w).is_err());
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::PreflightRejected
    );
}

#[test]
fn every_evaluation_budget_boundary_preserves_an_accepted_point() {
    use scioxide_core::Budget;
    let c = Constraints::new(2).unwrap().equality(SumEquality).unwrap();
    let problem = ConstrainedProblem::new(&Quadratic, &c).unwrap();
    let solver = Sqp::new(SqpOptions::default());
    let complete = solver.minimize(&problem, &[2.0, 2.0]).unwrap();
    for allowance in 1..=complete.counts().own().total_providers().unwrap() + 1 {
        let budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(allowance).unwrap());
        let (result, counts) = root(
            &solver,
            &problem,
            &[2.0, 2.0],
            ExecutionContext::sequential(),
            budgets,
        );
        assert!(counts.own().total_providers().unwrap() <= allowance);
        if let Ok(result) = result {
            assert_eq!(result.counts(), counts);
            assert!((result.fun() - result.x().iter().map(|x| x * x).sum::<f64>()).abs() < 1e-12);
            if result.is_success() {
                assert!((result.x()[0] + result.x()[1] - 1.0).abs() < 1e-5);
            } else {
                assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
            }
        } else {
            assert_eq!(
                allowance, 1,
                "only domain admission can precede the first objective"
            );
        }
    }
}

#[test]
fn failing_trial_derivative_preserves_state_and_caller_result() {
    struct Failing;
    impl ObjectiveFunction<f64> for Failing {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Quadratic.eval(x)
        }
        fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
            if x != [2.0, 2.0] {
                return Err(invalid(
                    scioxide_core::argument!("x"),
                    "injected trial derivative failure",
                ));
            }
            Quadratic.gradient_into(x, out)
        }
    }
    let free = free(2);
    let quadratic = ConstrainedProblem::new(&Quadratic, &free).unwrap();
    let failing = ConstrainedProblem::new(&Failing, &free).unwrap();
    let solver = Sqp::<f64>::default();
    let mut w = solver.workspace(&failing).unwrap();
    let mut out = solver.minimize(&quadratic, &[0.0, 0.0]).unwrap();
    let before = out.clone();
    assert!(solve(&solver, &failing, &[2.0, 2.0], &mut out, &mut w).is_err());
    assert_eq!(out.x(), before.x());
    assert_eq!(out.counts(), before.counts());
    assert_eq!(w.storage.accepted.x, [2.0, 2.0]);
    assert_eq!(w.state().step(), [0.0, 0.0]);
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::Failed
    );
}

#[test]
fn injected_bad_steps_and_negative_curvature_are_rejected() {
    #[derive(Debug)]
    struct Bad;
    impl SqpSubproblemProvider<f64> for Bad {
        fn solve_into(
            &mut self,
            _: SqpSubproblem<'_, f64>,
            state: &mut SqpState<f64>,
            _: &mut SolveContext<'_, '_>,
        ) -> Result<SqpSubproblemReport, SciOxideError> {
            state.step.fill(100.0);
            Ok(SqpSubproblemReport::new(OptimizeTermination::Converged))
        }
    }
    let free = free(2);
    let problem = ConstrainedProblem::new(&Quadratic, &free).unwrap();
    let solver = Sqp::<f64>::default();
    let mut w = solver
        .workspace(&problem)
        .unwrap()
        .with_subproblem_provider(Bad);
    let mut out = OptimizeResult::empty();
    solve(&solver, &problem, &[1.0, 1.0], &mut out, &mut w).unwrap();
    assert_eq!(out.termination(), OptimizeTermination::Stagnation);
    assert!(!out.is_success());
    assert_eq!(out.x(), &[1.0, 1.0]);
    assert_eq!(w.storage.accepted.x, [1.0, 1.0]);
    let h = Matrix::from_vec(1, 1, vec![-1.0]).unwrap();
    let j = Matrix::from_vec(0, 1, vec![]).unwrap();
    let model = SqpSubproblem {
        objective: SqpSubproblemObjective::Quadratic {
            hessian: h.view(),
            gradient: &[1.0],
        },
        jacobian: j.view(),
        values: &[],
        equalities: 0,
        bounds: &[(None, None)],
        tolerances: crate::OptimalityTolerances::default(),
        max_iterations: scioxide_core::Budget::ONE,
    };
    let mut provider = DenseSqpSubproblem::new(1, 0, 0).unwrap();
    let mut state = SqpState::new();
    let id = crate::context::identity(
        scioxide_core::algorithm_id!("optimize", "negative-curvature", 1),
        scioxide_core::ProblemToken::new(0),
        1,
        0,
    )
    .unwrap();
    assert!(
        crate::context::sequential_with_budgets(id, SolveBudgets::unbounded(), |ctx| provider
            .solve_into(model, &mut state, ctx))
        .is_err()
    );
}

fn rejects_provider_success<S: RealScalar + NumericScalar<Real = S> + 'static>(
    solver: &Sqp<S>,
    problem: &ConstrainedProblem<'_, S>,
    x0: &[S],
    before: &OptimizeResult<S>,
    corrupt_call: usize,
) -> SqpDiagnostics {
    #[derive(Debug)]
    struct Corrupt<S: RealScalar + NumericScalar<Real = S>> {
        inner: DenseSqpSubproblem<S>,
        calls: Arc<AtomicUsize>,
        corrupt_call: usize,
        termination: OptimizeTermination,
    }
    impl<S: RealScalar + NumericScalar<Real = S>> SqpSubproblemProvider<S> for Corrupt<S> {
        fn solve_into(
            &mut self,
            model: SqpSubproblem<'_, S>,
            state: &mut SqpState<S>,
            context: &mut SolveContext<'_, '_>,
        ) -> Result<SqpSubproblemReport, SciOxideError> {
            let calls = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
            if calls == self.corrupt_call {
                state.step.fill(S::try_from_usize_exact(100)?);
                return Ok(SqpSubproblemReport::new(self.termination));
            }
            self.inner.solve_into(model, state, context)
        }
    }
    let mut diagnostics = SqpDiagnostics::default();
    for termination in [
        OptimizeTermination::GradientSmall,
        OptimizeTermination::Converged,
    ] {
        let m = problem.constraints().nonlinear_functions().count();
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = Corrupt {
            inner: DenseSqpSubproblem::new(x0.len(), m, m).unwrap(),
            calls: Arc::clone(&calls),
            corrupt_call,
            termination,
        };
        let mut workspace = solver
            .workspace(problem)
            .unwrap()
            .with_subproblem_provider(provider);
        let mut output = before.clone();
        let result = solve(solver, problem, x0, &mut output, &mut workspace);
        let calls = calls.load(Ordering::Relaxed);
        if termination == OptimizeTermination::Converged {
            result.unwrap();
            assert!(calls > corrupt_call);
            assert!(
                output
                    .x()
                    .iter()
                    .zip(x0)
                    .all(|(&a, &b)| (a - b).abs() < S::try_from_usize_exact(100).unwrap())
            );
            diagnostics = workspace.storage.diagnostics;
            continue;
        }
        assert!(
            result.is_err(),
            "invalid {termination:?} provider success was accepted"
        );
        assert_eq!(calls, corrupt_call);
        assert_eq!(alloc::format!("{output:?}"), alloc::format!("{before:?}"));
        assert_eq!(workspace.storage.accepted.x, x0);
        assert!(
            workspace
                .state()
                .step()
                .iter()
                .all(|&v| v == <_ as scioxide_core::AlgebraicScalar>::ZERO)
        );
        assert!(
            workspace
                .state()
                .multipliers()
                .iter()
                .all(|&v| v == <_ as scioxide_core::AlgebraicScalar>::ZERO)
        );
        diagnostics = workspace.storage.diagnostics;
    }
    diagnostics
}

macro_rules! scalar_cases {
    ($module:ident, $s:ty, $tol:expr) => {
        mod $module {
            use super::*;
            #[test]
            fn dependent_opposed_inequalities_keep_nonnegative_duals() {
                struct Shifted;
                impl ObjectiveFunction<$s> for Shifted {
                    fn dim(&self) -> usize {
                        1
                    }
                    fn eval(&self, x: &[$s]) -> Result<$s, SciOxideError> {
                        Ok((x[0] + 1.0) * (x[0] + 1.0))
                    }
                    fn gradient_into(&self, x: &[$s], g: &mut [$s]) -> Result<bool, SciOxideError> {
                        g[0] = 2.0 * (x[0] + 1.0);
                        Ok(true)
                    }
                }
                for coefficients in [[1.0 as $s, -1.0, 1000.0], [-1.0, 1000.0, 1.0]] {
                    let mut constraints = Constraints::new(1).unwrap();
                    for a in coefficients {
                        constraints = constraints
                            .inequality(
                                ConstraintFn::new(1, move |x: &[$s]| Ok(a * x[0])).with_jacobian(
                                    move |_: &[$s], output| {
                                        output.copy_from_slice(&[a]);
                                        Ok(())
                                    },
                                ),
                            )
                            .unwrap();
                    }
                    let problem = ConstrainedProblem::new(&Shifted, &constraints).unwrap();
                    let solver = Sqp::new(SqpOptions::default());
                    let mut w = solver.workspace(&problem).unwrap();
                    let mut result = OptimizeResult::empty();
                    solve(&solver, &problem, &[1.0], &mut result, &mut w).unwrap();
                    assert!(result.is_success(), "{result:?}");
                    assert!(result.x()[0].abs() < $tol);
                    let stationarity = 2.0 * (result.x()[0] + 1.0)
                        + w.state()
                            .multipliers()
                            .iter()
                            .zip(coefficients)
                            .map(|(&l, a)| l * a)
                            .sum::<$s>();
                    assert!(stationarity.abs() < $tol);
                    assert!(w.state().multipliers().iter().all(|&l| l >= 0.0));
                }
            }
            struct ScalarQuadratic;
            impl ObjectiveFunction<$s> for ScalarQuadratic {
                fn dim(&self) -> usize {
                    2
                }
                fn eval(&self, x: &[$s]) -> Result<$s, SciOxideError> {
                    Ok(x[0] * x[0] + x[1] * x[1])
                }
                fn gradient_into(&self, x: &[$s], g: &mut [$s]) -> Result<bool, SciOxideError> {
                    g[0] = 2.0 * x[0];
                    g[1] = 2.0 * x[1];
                    Ok(true)
                }
            }
            #[test]
            fn provider_success_requires_original_kkt_residuals() {
                let free = free(2);
                let problem = ConstrainedProblem::new(&ScalarQuadratic, &free).unwrap();
                let solver = Sqp::default();
                let before = solver.minimize(&problem, &[0.0, 0.0]).unwrap();
                rejects_provider_success(&solver, &problem, &[1.0, 1.0], &before, 1);
            }
            #[test]
            fn final_iteration_checks_accepted_kkt_without_extra_work() {
                let free = free(2);
                let problem = ConstrainedProblem::new(&ScalarQuadratic, &free).unwrap();
                let complete = Sqp::default().minimize(&problem, &[1.0, 1.0]).unwrap();
                let solver =
                    Sqp::new(SqpOptions::default().with_max_iterations(scioxide_core::Budget::ONE));
                for allowance in [None, Some(complete.counts().own().iterations())] {
                    let mut budgets = SolveBudgets::unbounded();
                    if let Some(limit) = allowance {
                        budgets = budgets.with_limit(
                            BudgetKind::Iteration,
                            scioxide_core::Budget::try_from(limit).unwrap(),
                        );
                    }
                    let (result, _) = root(
                        &solver,
                        &problem,
                        &[1.0, 1.0],
                        ExecutionContext::sequential(),
                        budgets,
                    );
                    let result = result.unwrap();
                    assert_eq!(result.termination(), OptimizeTermination::Converged);
                    assert_eq!(result.x(), [0.0, 0.0]);
                    assert_eq!(result.diagnostics().stationarity(), Some(0.0));
                    assert_eq!(result.counts(), complete.counts());
                }
                let (denied, _) = root(
                    &solver,
                    &problem,
                    &[1.0, 1.0],
                    ExecutionContext::sequential(),
                    SolveBudgets::unbounded()
                        .with_limit(BudgetKind::Iteration, scioxide_core::Budget::ONE),
                );
                let denied = denied.unwrap();
                // The quadratic minimizer is exactly zero. The inner QP may
                // finish without consuming the sole outer Newton iteration.
                assert_eq!(denied.termination(), OptimizeTermination::Converged);
                assert_eq!(denied.x(), [0.0, 0.0]);
                assert_eq!(denied.counts().own().iterations(), 1);

                let constraint = ConstraintFn::new(2, |x: &[$s]| Ok(x[0] + x[1] - 1.0))
                    .with_jacobian(|_: &[$s], output| {
                        output.copy_from_slice(&[1.0, 1.0]);
                        Ok(())
                    });
                let constraints = Constraints::new(2).unwrap().equality(constraint).unwrap();
                let problem = ConstrainedProblem::new(&ScalarQuadratic, &constraints).unwrap();
                let unfinished =
                    Sqp::new(SqpOptions::default().with_max_iterations(scioxide_core::Budget::ONE))
                        .minimize(&problem, &[2.0, 2.0])
                        .unwrap();
                assert_eq!(unfinished.termination(), OptimizeTermination::Converged);
                assert!(unfinished.diagnostics().stationarity().unwrap() < $tol);
                assert!(unfinished.x().iter().all(|&x| (x - 0.5).abs() < $tol));
            }
            #[test]
            fn dependent_equalities_more_rows_than_variables_have_honest_duals() {
                let mut c = Constraints::new(2).unwrap();
                for scale in [1.0, 2.0, -3.0] {
                    c = c
                        .equality(
                            ConstraintFn::new(2, move |x: &[$s]| Ok(scale * (x[0] + x[1] - 1.0)))
                                .with_jacobian(move |_: &[$s], output| {
                                    output.copy_from_slice(&[scale, scale]);
                                    Ok(())
                                }),
                        )
                        .unwrap();
                }
                let problem = ConstrainedProblem::new(&ScalarQuadratic, &c).unwrap();
                let solver = Sqp::new(SqpOptions::default());
                let mut w = solver.workspace(&problem).unwrap();
                let mut r = OptimizeResult::empty();
                solve(&solver, &problem, &[2.0, -1.0], &mut r, &mut w).unwrap();
                assert!(r.is_success(), "{r:?}");
                assert!((r.x()[0] - 0.5).abs() < $tol);
                assert!(sqp_of(&r).multipliers_ambiguous());
                let l = w.state().multipliers();
                let weighted = l[0] + 2.0 * l[1] - 3.0 * l[2];
                assert!((2.0 * r.x()[0] + weighted).abs() < $tol);
                assert!((2.0 * r.x()[1] + weighted).abs() < $tol);
            }
            #[test]
            fn maratos_circle_uses_second_order_correction() {
                struct Linear;
                impl ObjectiveFunction<$s> for Linear {
                    fn dim(&self) -> usize {
                        2
                    }
                    fn eval(&self, x: &[$s]) -> Result<$s, SciOxideError> {
                        Ok(x[0])
                    }
                    fn gradient_into(&self, _: &[$s], g: &mut [$s]) -> Result<bool, SciOxideError> {
                        g.copy_from_slice(&[1.0, 0.0]);
                        Ok(true)
                    }
                }
                let c = Constraints::new(2)
                    .unwrap()
                    .equality(
                        ConstraintFn::new(2, |x: &[$s]| Ok(x[0] * x[0] + x[1] * x[1] - 1.0))
                            .with_jacobian(|x: &[$s], output| {
                                output.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
                                Ok(())
                            }),
                    )
                    .unwrap();
                let problem = ConstrainedProblem::new(&Linear, &c).unwrap();
                let solver = Sqp::new(SqpOptions::default());
                let r = solver.minimize(&problem, &[0.0, 1.0]).unwrap();
                assert!(r.is_success(), "{r:?}");
                assert!((r.x()[0] + 1.0).abs() < $tol);
                assert!(r.x()[1].abs() < $tol);
                assert!(sqp_of(&r).accepted_corrections() > 0);
                let d = rejects_provider_success(&solver, &problem, &[0.0, 1.0], &r, 2);
                assert!(d.second_order_corrections() >= 1);
            }
            #[test]
            fn inconsistent_linearizations_enter_restoration() {
                struct Objective;
                impl ObjectiveFunction<$s> for Objective {
                    fn dim(&self) -> usize {
                        1
                    }
                    fn eval(&self, x: &[$s]) -> Result<$s, SciOxideError> {
                        Ok((x[0] - 2.0).powi(2))
                    }
                    fn gradient_into(&self, x: &[$s], g: &mut [$s]) -> Result<bool, SciOxideError> {
                        g[0] = 2.0 * (x[0] - 2.0);
                        Ok(true)
                    }
                }
                let c = Constraints::new(1)
                    .unwrap()
                    .equality(
                        ConstraintFn::new(1, |x: &[$s]| Ok(x[0] * x[0] - 1.0)).with_jacobian(
                            |x: &[$s], output| {
                                output.copy_from_slice(&[2.0 * x[0]]);
                                Ok(())
                            },
                        ),
                    )
                    .unwrap()
                    .equality(
                        ConstraintFn::new(1, |x: &[$s]| Ok(x[0] - 1.0)).with_jacobian(
                            |_: &[$s], output| {
                                output.copy_from_slice(&[1.0]);
                                Ok(())
                            },
                        ),
                    )
                    .unwrap();
                let problem = ConstrainedProblem::new(&Objective, &c).unwrap();
                let solver = Sqp::new(SqpOptions::default());
                let r = solver.minimize(&problem, &[0.5]).unwrap();
                assert!(r.is_success(), "{r:?}");
                assert!((r.x()[0] - 1.0).abs() < $tol);
                assert!(sqp_of(&r).restoration_solves() > 0);
                let d = rejects_provider_success(&solver, &problem, &[0.5], &r, 2);
                assert!(d.restoration_solves() >= 1);
            }
            #[test]
            fn rank_deficient_constrained_least_squares_uses_shared_factors() {
                let design = Matrix::from_vec(3, 2, vec![1.0, 1.0, 2.0, 2.0, 0.0, 0.0]).unwrap();
                let jac = Matrix::from_vec(1, 2, vec![0.0, 1.0]).unwrap();
                let mut provider = DenseSqpSubproblem::new(2, 1, 3).unwrap();
                let mut state = SqpState::new();
                let model = SqpSubproblem {
                    objective: SqpSubproblemObjective::LeastSquares {
                        design: design.view(),
                        target: &[1.0, 2.0, 0.0],
                    },
                    jacobian: jac.view(),
                    values: &[-0.25],
                    equalities: 0,
                    bounds: &[(Some(0.0), None), (Some(0.0), None)],
                    tolerances: crate::OptimalityTolerances::default(),
                    max_iterations: scioxide_core::Budget::try_from(100).unwrap(),
                };
                let id = crate::context::identity(
                    scioxide_core::algorithm_id!("optimize", "test", 1),
                    scioxide_core::ProblemToken::new(0),
                    2,
                    0,
                )
                .unwrap();
                let report =
                    crate::context::sequential_with_budgets(id, SolveBudgets::unbounded(), |ctx| {
                        provider.solve_into(model, &mut state, ctx)
                    })
                    .unwrap();
                assert_eq!(report.termination(), OptimizeTermination::Converged);
                assert!((state.step()[0] + state.step()[1] - (1.0 as $s)).abs() < $tol);
                assert!(state.step()[1] <= 0.25 + $tol);
            }
            #[test]
            fn scaled_bounded_models_take_small_nonstationary_steps() {
                let root = if <$s>::MANTISSA_DIGITS == 24 {
                    256.0 as $s
                } else {
                    32768.0 as $s
                };
                let curvature = root * root;
                let h = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, curvature]).unwrap();
                let design = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, root]).unwrap();
                let padded_h = [1.0, 0.0, 99.0, 0.0, curvature];
                let padded_design = [1.0, 0.0, 99.0, 0.0, root];
                let strided_h =
                    scioxide_linalg::MatrixView::from_slice_with_strides(2, 2, &padded_h, 3, 1)
                        .unwrap();
                let strided_design = scioxide_linalg::MatrixView::from_slice_with_strides(
                    2,
                    2,
                    &padded_design,
                    3,
                    1,
                )
                .unwrap();
                for dependent in [false, true] {
                    let jac = if dependent {
                        Matrix::from_vec(2, 2, vec![2.0, 0.0, 4.0, 0.0]).unwrap()
                    } else {
                        Matrix::from_vec(0, 2, vec![]).unwrap()
                    };
                    let values = vec![0.0; jac.rows()];
                    let mut provider = DenseSqpSubproblem::new(2, jac.rows(), 2).unwrap();
                    let mut state = SqpState::new();
                    for (reuse, least_squares) in [false, true, false, true].into_iter().enumerate()
                    {
                        let target = [1.0, 1.0 / root];
                        let model = SqpSubproblem {
                            objective: if least_squares {
                                SqpSubproblemObjective::LeastSquares {
                                    design: if reuse < 2 {
                                        design.view()
                                    } else {
                                        strided_design
                                    },
                                    target: &target,
                                }
                            } else {
                                SqpSubproblemObjective::Quadratic {
                                    hessian: if reuse < 2 { h.view() } else { strided_h },
                                    gradient: &[-1.0, -1.0],
                                }
                            },
                            jacobian: jac.view(),
                            values: &values,
                            equalities: 0,
                            bounds: &[(None, Some(0.0)), (None, None)],
                            tolerances: crate::OptimalityTolerances::default(),
                            max_iterations: scioxide_core::Budget::try_from(100).unwrap(),
                        };
                        let id = crate::context::identity(
                            scioxide_core::algorithm_id!("optimize", "scaled-SQP", 1),
                            scioxide_core::ProblemToken::new(608),
                            2,
                            0,
                        )
                        .unwrap();
                        let report = crate::context::sequential_with_budgets(
                            id,
                            SolveBudgets::unbounded(),
                            |ctx| provider.solve_into(model, &mut state, ctx),
                        )
                        .unwrap();
                        assert_eq!(report.termination(), OptimizeTermination::Converged);
                        let x = state.step();
                        assert!(x[0].abs() < $tol);
                        assert!(
                            (curvature * x[1] - 1.0).abs() < $tol,
                            "tiny displacement was discarded: {x:?}"
                        );
                        let nonlinear_dual = state
                            .multipliers()
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| (2 * (i + 1)) as $s * v)
                            .sum::<$s>();
                        assert!(
                            (-1.0 + x[0] + state.upper_multipliers()[0] + nonlinear_dual).abs()
                                < $tol
                        );
                        assert!(
                            state
                                .multipliers()
                                .iter()
                                .chain(state.upper_multipliers())
                                .all(|&v| v >= 0.0)
                        );
                        assert_eq!(state.lower_multipliers(), [0.0, 0.0]);
                        assert_eq!(state.upper_multipliers()[1], 0.0);
                        assert!(
                            (x[0] * (state.upper_multipliers()[0] + nonlinear_dual)).abs() < $tol
                        );
                    }
                }
            }
            #[test]
            fn unrepresentable_stationary_point_reports_stagnation() {
                let h = Matrix::from_vec(2, 2, vec![1.0 as $s, 0.0, 0.0, 3.0]).unwrap();
                let jac = Matrix::from_vec(0, 2, vec![]).unwrap();
                let defaults = crate::OptimalityTolerances::<$s>::default();
                let tolerances = crate::OptimalityTolerances::new(
                    defaults.feasibility(),
                    scioxide_core::GradientTolerance::new(<$s>::EPSILON * <$s>::EPSILON, 0.0)
                        .unwrap(),
                    defaults.complementarity(),
                    defaults.rank(),
                );
                let model = SqpSubproblem {
                    objective: SqpSubproblemObjective::Quadratic {
                        hessian: h.view(),
                        gradient: &[-1.0, -1.0],
                    },
                    jacobian: jac.view(),
                    values: &[],
                    equalities: 0,
                    bounds: &[(None, Some(0.0)), (None, None)],
                    tolerances,
                    max_iterations: scioxide_core::Budget::try_from(100).unwrap(),
                };
                let mut provider = DenseSqpSubproblem::new(2, 0, 0).unwrap();
                let mut state = SqpState::new();
                let id = crate::context::identity(
                    scioxide_core::algorithm_id!("optimize", "SQP-rounding", 1),
                    scioxide_core::ProblemToken::new(608),
                    2,
                    0,
                )
                .unwrap();
                let report =
                    crate::context::sequential_with_budgets(id, SolveBudgets::unbounded(), |ctx| {
                        provider.solve_into(model, &mut state, ctx)
                    })
                    .unwrap();
                // The exact optimum (0, 1/3) is not representable; neither adjacent
                // float can satisfy this deliberately sub-roundoff gradient target.
                assert_eq!(report.termination(), OptimizeTermination::Stagnation);
            }
        }
    };
}
scalar_cases!(f64_cases, f64, 1e-5);
scalar_cases!(f32_cases, f32, 2e-3);

#[test]
fn cancellation_and_deadline_never_publish_a_trial() {
    use core::sync::atomic::AtomicU64;
    use scioxide_core::{CancellationToken, Clock, MonotonicInstant};
    struct ClockAt(AtomicU64);
    impl Clock for ClockAt {
        fn now(&self) -> Result<MonotonicInstant, scioxide_core::CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::from_nanos(self.0.load(Ordering::Relaxed)),
            ))
        }
    }
    struct Stop<'a> {
        token: &'a CancellationToken,
        clock: &'a ClockAt,
        time: bool,
    }
    impl ObjectiveFunction<f64> for Stop<'_> {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            if x != [2.0, 2.0] {
                if self.time {
                    self.clock.0.store(100, Ordering::Relaxed);
                } else {
                    self.token.cancel();
                }
            }
            Quadratic.eval(x)
        }
        fn gradient_into(&self, x: &[f64], g: &mut [f64]) -> Result<bool, SciOxideError> {
            Quadratic.gradient_into(x, g)
        }
    }
    let free = free(2);
    for time in [false, true] {
        let token = CancellationToken::new();
        let clock = ClockAt(AtomicU64::new(0));
        let clock_binding = scioxide_core::ClockBinding::new(&clock);
        let solver =
            Sqp::new(SqpOptions::default().with_time_limit(core::time::Duration::from_nanos(50)));
        let objective = Stop {
            token: &token,
            clock: &clock,
            time,
        };
        let problem = ConstrainedProblem::new(&objective, &free).unwrap();
        let mut w = solver.workspace(&problem).unwrap();
        let mut r = OptimizeResult::empty();
        solver
            .minimize_into(
                &problem,
                &[2.0, 2.0],
                &mut r,
                &mut w,
                ExecutionContext::sequential()
                    .with_cancellation(&token)
                    .with_clock(&clock_binding)
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            r.termination(),
            if time {
                OptimizeTermination::TimeLimit
            } else {
                OptimizeTermination::Cancelled
            }
        );
        assert_eq!(r.x(), [2.0, 2.0]);
        assert_eq!(r.fun(), 8.0);
        assert_eq!(r.counts(), w.last_attempt().unwrap().counts());
    }
}

#[test]
fn local_and_ancestor_iteration_limits_are_explicit() {
    use scioxide_core::Budget;
    let c = Constraints::new(2)
        .unwrap()
        .bounds(Bounds::new(vec![(Some(0.5), None); 2]).unwrap())
        .unwrap();
    let problem = ConstrainedProblem::new(&Quadratic, &c).unwrap();
    let solver = Sqp::new(SqpOptions::default().with_max_subproblem_iterations(Budget::ONE));
    let result = solver.minimize(&problem, &[2.0, 2.0]).unwrap();
    assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
    assert_eq!(result.x(), [2.0, 2.0]);
    for kind in [BudgetKind::Iteration, BudgetKind::Proposal] {
        let (r, counts) = root(
            &solver,
            &problem,
            &[2.0, 2.0],
            ExecutionContext::sequential(),
            SolveBudgets::unbounded().with_limit(kind, Budget::ONE),
        );
        let r = r.unwrap();
        assert!(!r.is_success());
        assert_eq!(r.counts(), counts);
        assert_eq!(r.x(), [2.0, 2.0]);
    }
}

#[test]
fn supplied_executor_preserves_sequential_results() {
    use scioxide_core::{ExecutionPlan, Executor, PartitionFn, SequentialExecutor};
    struct Count(AtomicUsize);
    impl Executor for Count {
        fn worker_budget(&self) -> core::num::NonZeroUsize {
            core::num::NonZeroUsize::MIN
        }
        fn execute(
            &self,
            c: ExecutionContext<'_>,
            p: ExecutionPlan,
            op: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            SequentialExecutor.execute(c, p, op)
        }
    }
    let free = free(2);
    let problem = ConstrainedProblem::new(&Quadratic, &free).unwrap();
    let solver = Sqp::<f64>::default();
    let mut w = solver.workspace(&problem).unwrap();
    let mut expected = OptimizeResult::empty();
    solve(&solver, &problem, &[2.0, 2.0], &mut expected, &mut w).unwrap();
    let executor = Count(AtomicUsize::new(0));
    let mut fresh = solver.workspace(&problem).unwrap();
    let mut result = OptimizeResult::empty();
    solver
        .minimize_into(
            &problem,
            &[2.0, 2.0],
            &mut result,
            &mut fresh,
            ExecutionContext::new(&executor),
        )
        .unwrap();
    assert_eq!(result.x(), expected.x());
    assert_eq!(result.counts(), expected.counts());
    assert_eq!(
        executor.0.load(Ordering::Relaxed),
        0,
        "sequential kernels do not submit worker partitions"
    );
}

#[test]
fn intersected_bounds_and_configuration_identity_are_checked() {
    let unit = Constraints::<f64>::new(2)
        .unwrap()
        .bounds(Bounds::new(vec![(Some(0.0), Some(1.0)); 2]).unwrap())
        .unwrap();
    // A disjoint second box is an input error when the collection is built.
    let disjoint = unit.bounds(Bounds::new(vec![(Some(2.0), Some(3.0)); 2]).unwrap());
    assert_eq!(
        disjoint.unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
    let boxed = |u| {
        Constraints::new(2)
            .unwrap()
            .bounds(Bounds::new(vec![(Some(0.0), Some(u)); 2]).unwrap())
            .unwrap()
    };
    let (narrow, wide) = (boxed(1.0_f64), boxed(2.0));
    let solver = Sqp::<f64>::default();
    assert_ne!(
        solver.identity(&ConstrainedProblem::new(&Quadratic, &narrow).unwrap()),
        solver.identity(&ConstrainedProblem::new(&Quadratic, &wide).unwrap())
    );
}

#[test]
fn zero_row_least_squares_is_a_feasible_constant_objective() {
    let design = Matrix::<f64>::from_vec(0, 2, vec![]).unwrap();
    let mut provider = DenseSqpSubproblem::new(2, 0, 0).unwrap();
    let mut state = SqpState::new();
    let model = SqpSubproblem {
        objective: SqpSubproblemObjective::LeastSquares {
            design: design.view(),
            target: &[],
        },
        jacobian: design.view(),
        values: &[],
        equalities: 0,
        bounds: &[(Some(0.0), Some(1.0)), (None, None)],
        tolerances: crate::OptimalityTolerances::default(),
        max_iterations: scioxide_core::Budget::try_from(20).unwrap(),
    };
    let id = crate::context::identity(
        scioxide_core::algorithm_id!("optimize", "constant-LS", 1),
        scioxide_core::ProblemToken::new(0),
        2,
        0,
    )
    .unwrap();
    let report = crate::context::sequential_with_budgets(id, SolveBudgets::unbounded(), |ctx| {
        provider.solve_into(model, &mut state, ctx)
    })
    .unwrap();
    assert_eq!(report.termination(), OptimizeTermination::Converged);
    assert_eq!(state.step(), [0.0, 0.0]);
}

#[test]
fn restoration_state_reshape_invalidates_old_active_ids() {
    let mut state = SqpState::<f64>::new();
    state.reserve(1, 3).unwrap();
    state.active.extend_from_slice(&[2, 4]);
    state.reserve(1, 0).unwrap();
    assert!(state.active_constraints().is_empty());
    let free = free(2);
    let problem = ConstrainedProblem::new(&Quadratic, &free).unwrap();
    let solver = Sqp::<f64>::default();
    let mut w = solver.workspace(&problem).unwrap();
    let mut bad = SqpState::new();
    bad.reserve(2, 0).unwrap();
    bad.active.push(4);
    w.set_warm_start(&bad).unwrap();
    let mut out = OptimizeResult::empty();
    assert!(solve(&solver, &problem, &[0.0, 0.0], &mut out, &mut w).is_err());
    assert_eq!(out.termination(), OptimizeTermination::NotStarted);
}

#[allow(clippy::panic)]
fn sqp_of<S: scioxide_core::RealScalar>(
    result: &crate::OptimizeResult<S>,
) -> crate::SqpDiagnostics {
    match result.method_diagnostics() {
        crate::OptimizeDiagnostics::Sqp(diagnostics, _) => diagnostics,
        other => panic!("expected SQP diagnostics, found {other:?}"),
    }
}
