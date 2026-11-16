//! Public solve-context admission, failure, and accepted-state contracts.

#![allow(clippy::unwrap_used)]

use core::cell::Cell;

use scioxide_core::{
    Budget, BudgetKind, CancellationToken, EvaluationRole, ExecutionContext, InvalidInputReason,
    ProblemToken, SciOxideError, SolveBudgets, SolveState,
};
use scioxide_optimize::{
    Bfgs, GradientDescent, Lbfgs, NewtonOptimizer, NonlinearCg, ObjectiveFunction, OptimizeResult,
    OptimizeTermination, TrustRegionOptimizer,
};

#[test]
fn finite_difference_gradient_preserves_injected_execution_and_admitted_counts() {
    struct CancelDuringStencil<'a> {
        token: &'a CancellationToken,
        calls: Cell<usize>,
    }
    impl ObjectiveFunction<f64> for CancelDuringStencil<'_> {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            self.calls.set(self.calls.get() + 1);
            if self.calls.get() == 2 {
                self.token.cancel();
            }
            Ok(x[0] * x[0] + x[1] * x[1])
        }
    }
    let solver = GradientDescent::new(
        scioxide_optimize::GradientDescentOptions::<f64>::default()
            .with_derivative_strategy(scioxide_differentiate::DerivativeStrategy::Central),
    );
    let token = CancellationToken::new();
    let objective = CancelDuringStencil {
        token: &token,
        calls: Cell::new(0),
    };
    let mut workspace = solver.workspace(2).unwrap();
    let mut result = OptimizeResult::<f64>::empty();
    solver
        .minimize_into(
            &objective,
            &[1.0, 2.0],
            &mut result,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&token),
        )
        .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::Cancelled);
    assert_eq!(result.x(), &[1.0, 2.0]);
    assert_eq!(result.fun_evals(), objective.calls.get());
    assert_eq!(result.counts(), workspace.last_attempt().unwrap().counts());
    assert_eq!(result.fun(), 5.0);
}

struct Quadratic {
    objectives: Cell<usize>,
    gradients: Cell<usize>,
    fail_gradient: bool,
}

impl Quadratic {
    fn new(fail_gradient: bool) -> Self {
        Self {
            objectives: Cell::new(0),
            gradients: Cell::new(0),
            fail_gradient,
        }
    }
}

impl ObjectiveFunction<f64> for Quadratic {
    fn dim(&self) -> usize {
        1
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        self.objectives.set(self.objectives.get() + 1);
        Ok(x[0] * x[0])
    }
    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        self.gradients.set(self.gradients.get() + 1);
        output[0] = 2.0 * x[0];
        if self.fail_gradient {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("x"),
                InvalidInputReason::Malformed,
                "provider failure",
            ));
        }
        Ok(true)
    }
}

macro_rules! local_context_contract {
    ($test:ident, $solver:expr) => {
        local_context_contract!($test, $solver, 1);
    };
    ($test:ident, $solver:expr, $request:expr) => {
        #[test]
        fn $test() {
            let solver = $solver;
            let objective = Quadratic::new(false);
            let mut workspace = solver.workspace($request).unwrap();
            let mut output = OptimizeResult::<f64>::empty();
            let mut parent_state = SolveState::new();
            let mut budgets =
                SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
            let mut parent = parent_state
                .begin_solve(
                    ExecutionContext::sequential(),
                    parent_identity(11, 1),
                    &mut budgets,
                )
                .unwrap();
            solver
                .minimize_into(&objective, &[1.0], &mut output, &mut workspace, &mut parent)
                .unwrap();
            assert_eq!(output.termination(), OptimizeTermination::EvaluationLimit);
            assert_eq!(output.x(), &[1.0]);
            assert_eq!(output.fun(), 1.0);
            assert!(output.grad().is_none());
            assert_eq!(objective.objectives.get(), 1);
            assert_eq!(objective.gradients.get(), 0);
            assert_eq!(
                parent
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Objective),
                1
            );
            assert_eq!(output.counts(), workspace.last_attempt().unwrap().counts());
            drop(parent);

            let failing = Quadratic::new(true);
            let mut budgets = SolveBudgets::unbounded();
            let mut parent = parent_state
                .begin_solve(
                    ExecutionContext::sequential(),
                    parent_identity(12, 1),
                    &mut budgets,
                )
                .unwrap();
            let previous = output.clone();
            let result =
                solver.minimize_into(&failing, &[1.0], &mut output, &mut workspace, &mut parent);
            assert!(matches!(
                (result)
                    .as_ref()
                    .map_err(scioxide_core::SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::Malformed,
                    ..
                })
            ));
            assert_eq!(output.x(), previous.x());
            assert_eq!(output.termination(), previous.termination());
            assert_eq!(
                parent
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Gradient),
                1
            );
            assert_eq!(
                workspace
                    .last_attempt()
                    .unwrap()
                    .counts()
                    .own()
                    .provider(EvaluationRole::Gradient),
                1
            );
            assert_eq!(
                workspace.last_attempt().unwrap().outcome(),
                scioxide_core::AttemptOutcome::Failed
            );
            assert_eq!(failing.gradients.get(), 1);
        }
    };
}

fn parent_identity(token: u128, dimension: usize) -> scioxide_core::SolveIdentity {
    scioxide_core::SolveIdentity::new(
        ProblemToken::new(token),
        scioxide_core::algorithm_id!("test", "BfgsParent", 1),
        dimension as u128,
        0,
    )
}
local_context_contract!(bfgs_admission_and_failure, Bfgs::<f64>::default());
local_context_contract!(lbfgs_admission_and_failure, Lbfgs::<f64>::default());
local_context_contract!(
    gradient_descent_admission_and_failure,
    GradientDescent::<f64>::default()
);
local_context_contract!(
    nonlinear_cg_admission_and_failure,
    NonlinearCg::<f64>::default()
);
local_context_contract!(
    newton_admission_and_failure,
    NewtonOptimizer::<f64>::default()
);
local_context_contract!(
    trust_region_admission_and_failure,
    TrustRegionOptimizer::<f64>::default(),
    scioxide_optimize::TrustRegionWorkspaceRequest::dense(1)
);

#[test]
fn cancellation_during_initial_observation_preserves_only_validated_data() {
    struct CancelAfterValue<'a>(&'a CancellationToken);
    impl ObjectiveFunction<f64> for CancelAfterValue<'_> {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            self.0.cancel();
            Ok(x[0] * x[0])
        }
    }
    let solver = Bfgs::<f64>::default();
    let cancellation = CancellationToken::new();
    let execution = ExecutionContext::sequential().with_cancellation(&cancellation);
    let mut workspace = solver.workspace(1).unwrap();
    let mut result = OptimizeResult::<f64>::empty();
    solver
        .minimize_into(
            &CancelAfterValue(&cancellation),
            &[2.0],
            &mut result,
            &mut workspace,
            execution,
        )
        .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::Cancelled);
    assert_eq!(result.fun(), 4.0);
    assert!(result.grad().is_none());
    assert_eq!(
        result.counts().total().unwrap().total_providers().unwrap(),
        1
    );
}

struct LinearResidual {
    residuals: Cell<usize>,
    jacobians: Cell<usize>,
    fail_jacobian: bool,
}

impl scioxide_optimize::LeastSquaresProblem<f64> for LinearResidual {
    fn param_dim(&self) -> usize {
        1
    }
    fn residual_dim(&self) -> usize {
        1
    }
    fn residual_into(&self, x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
        self.residuals.set(self.residuals.get() + 1);
        output[0] = x[0] - 3.0;
        Ok(())
    }
    fn jacobian_into(
        &self,
        _: &[f64],
        output: &mut scioxide_linalg::Matrix<f64>,
    ) -> Result<bool, SciOxideError> {
        self.jacobians.set(self.jacobians.get() + 1);
        output.as_mut_slice()[0] = 1.0;
        if self.fail_jacobian {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("x"),
                InvalidInputReason::Malformed,
                "provider failure",
            ));
        }
        Ok(true)
    }
}

macro_rules! least_squares_context_contract {
    ($test:ident, $solver:expr) => {
        #[test]
        fn $test() {
            let solver = $solver;
            let mut workspace = solver
                .workspace(scioxide_optimize::LeastSquaresWorkspaceRequest::dense(1, 1))
                .unwrap();
            let mut output = scioxide_optimize::LeastSquaresResult::empty();
            for fail_jacobian in [false, true] {
                let problem = LinearResidual {
                    residuals: Cell::new(0),
                    jacobians: Cell::new(0),
                    fail_jacobian,
                };
                let mut state = SolveState::new();
                let mut budgets = SolveBudgets::unbounded().with_limit(
                    BudgetKind::Evaluation,
                    Budget::try_from(if fail_jacobian { 2 } else { 1 }).unwrap(),
                );
                let mut context = state
                    .begin_solve(
                        ExecutionContext::sequential(),
                        parent_identity(7, 1),
                        &mut budgets,
                    )
                    .unwrap();
                let outcome =
                    solver.solve_into(&problem, &[1.0], &mut output, &mut workspace, &mut context);
                if fail_jacobian {
                    assert!(matches!(
                        (outcome)
                            .as_ref()
                            .map_err(scioxide_core::SciOxideError::view),
                        Err(scioxide_core::SciOxideErrorView::InvalidInput {
                            reason: InvalidInputReason::Malformed,
                            ..
                        })
                    ));
                } else {
                    outcome.unwrap();
                    assert_eq!(output.counts(), workspace.last_attempt().unwrap().counts());
                }
                assert_eq!(output.x(), &[1.0]);
                assert_eq!(output.residual(), &[-2.0]);
                assert_eq!(output.cost().unwrap(), 2.0);
                assert_eq!(output.termination(), OptimizeTermination::EvaluationLimit);
                assert!(output.stationarity().is_none());
                assert_eq!(
                    context
                        .counts()
                        .total()
                        .unwrap()
                        .provider(EvaluationRole::Residual),
                    1
                );
                assert_eq!(
                    context
                        .counts()
                        .total()
                        .unwrap()
                        .provider(EvaluationRole::Jacobian),
                    usize::from(fail_jacobian)
                );
                assert_eq!(problem.residuals.get(), 1);
                assert_eq!(problem.jacobians.get(), usize::from(fail_jacobian));
                assert_eq!(
                    workspace
                        .last_attempt()
                        .unwrap()
                        .counts()
                        .own()
                        .provider(EvaluationRole::Jacobian),
                    usize::from(fail_jacobian)
                );
            }
        }
    };
}

least_squares_context_contract!(
    gauss_newton_admission_and_failure,
    scioxide_optimize::LeastSquares::<f64>::new(
        scioxide_optimize::LeastSquaresOptions::default()
            .with_globalization(scioxide_optimize::LeastSquaresGlobalization::gauss_newton())
    )
    .unwrap()
);
least_squares_context_contract!(
    levenberg_marquardt_admission_and_failure,
    scioxide_optimize::LeastSquares::<f64>::default()
);

macro_rules! nested_context_contract {
    ($test:ident, $solver:expr, $objectives:expr, $gradients:expr, $iterations:expr) => {
        #[test]
        fn $test() {
            let solver = $solver;
            let objective = Quadratic::new(false);
            let constraints = scioxide_optimize::Constraints::new(1).unwrap();
            let problem =
                scioxide_optimize::ConstrainedProblem::new(&objective, &constraints).unwrap();
            let mut workspace = solver.workspace(&problem).unwrap();
            let mut result = OptimizeResult::<f64>::empty();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(2).unwrap());
            let mut parent = state
                .begin_solve(
                    ExecutionContext::sequential(),
                    parent_identity(21, 1),
                    &mut budgets,
                )
                .unwrap();
            solver
                .minimize_into(&problem, &[2.0], &mut result, &mut workspace, &mut parent)
                .unwrap();
            assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
            assert_eq!(result.x(), &[2.0]);
            assert_eq!(result.fun(), 4.0);
            let total = result.counts().total().unwrap();
            assert_eq!(
                result.fun_evals(),
                total.provider(EvaluationRole::Objective)
            );
            assert_eq!(
                result.grad_evals(),
                total.provider(EvaluationRole::Gradient)
            );
            assert_eq!(result.counts(), workspace.last_attempt().unwrap().counts());
            // The child epoch charges every admitted callback to its ancestor.
            let charged = parent.counts().total().unwrap();
            assert_eq!(parent.counts().own().total_providers().unwrap(), 0);
            assert_eq!(
                charged.provider(EvaluationRole::Objective),
                objective.objectives.get()
            );
            assert_eq!(
                charged.provider(EvaluationRole::Gradient),
                objective.gradients.get()
            );
            assert_eq!(
                total.provider(EvaluationRole::Objective),
                objective.objectives.get()
            );
            assert_eq!(
                total.provider(EvaluationRole::Gradient),
                objective.gradients.get()
            );
            assert_eq!(objective.objectives.get(), $objectives);
            assert_eq!(objective.gradients.get(), $gradients);
            assert_eq!(
                result.counts().own().iterations(),
                usize::from($iterations != 0)
            );
            assert_eq!(result.counts().total().unwrap().iterations(), $iterations);
        }
    };
}

nested_context_contract!(
    penalty_nested_budget_is_conserved,
    scioxide_optimize::PenaltyMethod::<f64>::default(),
    1,
    1,
    2
);
nested_context_contract!(
    augmented_nested_budget_is_conserved,
    scioxide_optimize::AugmentedLagrangian::<f64>::default(),
    1,
    1,
    2
);
nested_context_contract!(
    interior_context_budget_is_conserved,
    scioxide_optimize::InteriorPoint::<f64>::default(),
    1,
    0,
    0
);

#[test]
fn nelder_mead_preserves_a_partially_evaluated_initial_simplex() {
    let solver = scioxide_optimize::NelderMead::<f64>::default();
    let mut workspace = solver.workspace(1).unwrap();
    let objective = Quadratic::new(false);
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            parent_identity(50, 1),
            &mut budgets,
        )
        .unwrap();
    let mut result = OptimizeResult::<f64>::empty();
    solver
        .minimize_into(
            &objective,
            &[2.0],
            &mut result,
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
    assert_eq!(result.fun(), 4.0);
    assert_eq!(result.x(), &[2.0]);
    assert_eq!(objective.objectives.get(), 1);
    assert_eq!(result.counts(), workspace.last_attempt().unwrap().counts());
    assert_eq!(result.counts().own().accepted(), 1);
    assert_eq!(result.counts().own().rejected(), 0);
    assert_eq!(workspace.last_accepted().unwrap().point(), &[2.0]);
}

#[test]
fn global_searches_conserve_parent_observation_budgets() {
    let bounds = scioxide_core::Bounds::new(vec![(Some(-5.0), Some(5.0))]).unwrap();
    let differential = scioxide_optimize::DifferentialEvolution::new(
        scioxide_optimize::DifferentialEvolutionOptions::default(),
    )
    .unwrap();
    macro_rules! check {
        ($solver:expr) => {{
            let solver = $solver;
            let objective = Quadratic::new(false);
            let problem = scioxide_optimize::BoundedObjective::new(&objective, &bounds).unwrap();
            let mut state = SolveState::new();
            let mut budgets =
                SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
            let mut context = state
                .begin_solve(
                    ExecutionContext::sequential(),
                    parent_identity(81, 1),
                    &mut budgets,
                )
                .unwrap();
            let mut workspace = solver.workspace(1).unwrap();
            let mut output = OptimizeResult::empty();
            solver
                .minimize_into(&problem, &[2.0], &mut output, &mut workspace, &mut context)
                .unwrap();
            assert_eq!(output.x(), &[2.0]);
            assert_eq!(output.fun(), 4.0);
            assert_eq!(output.termination(), OptimizeTermination::EvaluationLimit);
            assert_eq!(objective.objectives.get(), 1);
            assert_eq!(objective.gradients.get(), 0);
            assert_eq!(
                output.counts().total().unwrap(),
                context.counts().total().unwrap()
            );
        }};
    }
    check!(differential);
}

#[test]
fn differential_evolution_stops_before_an_unaffordable_sample() {
    let bounds = scioxide_core::Bounds::new(vec![(Some(-5.0), Some(5.0))]).unwrap();
    let solver = scioxide_optimize::DifferentialEvolution::new(
        scioxide_optimize::DifferentialEvolutionOptions::default(),
    )
    .unwrap();
    let objective = Quadratic::new(false);
    let problem = scioxide_optimize::BoundedObjective::new(&objective, &bounds).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Sample, Budget::ONE);
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            parent_identity(82, 1),
            &mut budgets,
        )
        .unwrap();
    let mut workspace = solver.workspace(1).unwrap();
    let mut output = OptimizeResult::empty();
    solver
        .minimize_into(&problem, &[2.0], &mut output, &mut workspace, &mut context)
        .unwrap();
    assert_eq!(output.termination(), OptimizeTermination::SampleLimit);
    assert_eq!(objective.objectives.get(), 2);
    assert!(output.fun() <= 4.0);
    assert_eq!(context.budget_report(BudgetKind::Sample).consumed(), 1);
    assert_eq!(output.counts().own().accepted(), 2);
}

#[test]
fn identities_include_line_search_and_inner_solver_configuration() {
    let original = Bfgs::<f64>::default();
    let search = scioxide_optimize::BacktrackingLineSearch::new(
        1.0,
        f64::EPSILON,
        1e-4,
        0.25,
        Budget::try_from(40).unwrap(),
    )
    .unwrap();
    let changed = Bfgs::new(scioxide_optimize::BfgsOptions::default().with_line_search(search));
    let objective = Quadratic::new(false);
    let mut workspace = original.workspace(1).unwrap();
    let mut output = OptimizeResult::<f64>::empty();
    original
        .minimize_into(
            &objective,
            &[2.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let checkpoint = workspace.checkpoint().unwrap();
    let mut changed_workspace = changed.workspace(1).unwrap();
    let mut unchanged = OptimizeResult::<f64>::empty();
    assert!(
        changed
            .resume_into(
                &objective,
                &checkpoint,
                &mut unchanged,
                &mut changed_workspace,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(unchanged.termination(), OptimizeTermination::NotStarted);
}

#[test]
fn restarting_a_prepared_method_starts_new_local_epochs() {
    let solver = Bfgs::<f64>::default();
    let objective = Quadratic::new(false);
    let mut workspace = solver.workspace(1).unwrap();
    let mut output = OptimizeResult::<f64>::empty();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            parent_identity(92, 1),
            &mut budgets,
        )
        .unwrap();
    solver
        .minimize_into(
            &objective,
            &[2.0],
            &mut output,
            &mut workspace,
            &mut context,
        )
        .unwrap();
    let first_epoch = workspace.last_attempt().unwrap().epoch().unwrap();
    let count = objective.objectives.get();
    solver
        .minimize_into(
            &objective,
            &[2.0],
            &mut output,
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert_ne!(
        workspace.last_attempt().unwrap().epoch().unwrap(),
        first_epoch
    );
    assert!(objective.objectives.get() > count);
    assert_eq!(context.counts().own().total_providers().unwrap(), 0);
}

#[test]
fn simplex_counts_admitted_pivots_and_denied_certificate_work() {
    let problem = scioxide_optimize::LpProblem::new(vec![-1.0_f64, -1.0])
        .unwrap()
        .with_constraints(
            scioxide_optimize::LinearConstraints::new(
                scioxide_linalg::Matrix::from_vec(2, 2, vec![1., 0., 0., 1.]).unwrap(),
                vec![1., 1.],
                vec![scioxide_optimize::LinearConstraintSense::LessEqual; 2],
            )
            .unwrap(),
        )
        .unwrap();
    for kind in [BudgetKind::Iteration, BudgetKind::Evaluation] {
        // Iteration allowances are epoch-local: the pivot limit is an
        // option, while evaluations are shared and charge the parent.
        let solver = if kind == BudgetKind::Iteration {
            scioxide_optimize::SimplexSolver::default().with_max_iterations(Budget::ONE)
        } else {
            scioxide_optimize::SimplexSolver::default()
        };
        let mut state = SolveState::new();
        let mut budgets = if kind == BudgetKind::Iteration {
            SolveBudgets::unbounded()
        } else {
            SolveBudgets::unbounded().with_limit(kind, Budget::ONE)
        };
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                parent_identity(101, 1),
                &mut budgets,
            )
            .unwrap();
        let mut workspace = solver.workspace(&problem).unwrap();
        let mut result = scioxide_optimize::ProgrammingResult::empty();
        solver
            .solve_into(&problem, &mut result, &mut workspace, &mut context)
            .unwrap();
        assert_eq!(
            result.counts().total().unwrap(),
            context.counts().total().unwrap()
        );
        assert_eq!(
            result
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Objective),
            0
        );
        assert!(result.x().iter().all(|x| x.is_finite()));
        assert!(result.fun().is_finite());
        assert!(!result.is_success());
        if kind == BudgetKind::Iteration {
            assert_eq!(result.termination(), OptimizeTermination::IterationLimit);
            assert_eq!(result.counts().own().iterations(), 1);
            assert_eq!(result.counts().own().proposals(), 1);
            assert_eq!(result.counts().own().accepted(), 1);
        } else {
            assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
            assert_eq!(result.x(), &[0.0, 0.0]);
            assert_eq!(result.fun(), 0.0);
            assert_eq!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Factorization),
                1
            );
            assert_eq!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::LinearOperator),
                0
            );
            assert_eq!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::LinearSolve),
                0
            );
        }
    }
}

#[test]
fn interior_point_lp_meters_shared_factors_and_both_right_hand_sides() {
    let problem = scioxide_optimize::LpProblem::new(vec![-1.0_f64])
        .unwrap()
        .with_bounds(scioxide_core::Bounds::new(vec![(Some(0.0), Some(1.0))]).unwrap())
        .unwrap();
    let solver = scioxide_optimize::InteriorPointLp::default();
    for evaluations in 1usize..=4 {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(
            BudgetKind::Evaluation,
            Budget::try_from(evaluations).unwrap(),
        );
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                parent_identity(102, 1),
                &mut budgets,
            )
            .unwrap();
        let mut workspace = solver.workspace(&problem).unwrap();
        let mut result = scioxide_optimize::ProgrammingResult::empty();
        solver
            .solve_into(&problem, &mut result, &mut workspace, &mut context)
            .unwrap();
        assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
        assert!(result.fun().is_finite());
        assert_eq!(
            result.counts().total().unwrap(),
            context.counts().total().unwrap()
        );
        assert_eq!(
            result
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Factorization),
            1
        );
        assert_eq!(
            result
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::LinearSolve),
            evaluations - 1
        );
        assert_eq!(
            result.counts().own().accepted(),
            usize::from(evaluations == 4)
        );
        assert_eq!(
            result.counts().own().cache_hits(),
            evaluations.saturating_sub(2)
        );
    }
}

#[test]
fn hsd_recovery_obeys_parent_evaluation_limits() {
    use scioxide_optimize::{
        InteriorPointLp, LinearConstraintSense, LinearConstraints, LpProblem, ProgrammingResult,
    };
    let problem = LpProblem::new(vec![-1.0_f64, 0.0])
        .unwrap()
        .with_constraints(
            LinearConstraints::new(
                scioxide_linalg::Matrix::from_vec(1, 2, vec![1.0, -1.0]).unwrap(),
                vec![1.0],
                vec![LinearConstraintSense::Equal],
            )
            .unwrap(),
        )
        .unwrap();
    let solver = InteriorPointLp::default();
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut result = ProgrammingResult::empty();
    let mut saw_nested_stop = false;
    let mut saw_unbounded = false;
    for limit in 1..=100 {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                parent_identity(1981, 2),
                &mut budgets,
            )
            .unwrap();
        solver
            .solve_into(&problem, &mut result, &mut workspace, &mut context)
            .unwrap();
        assert!(result.fun().is_finite());
        assert_eq!(
            result.counts().total().unwrap(),
            context.counts().total().unwrap()
        );
        if result.termination() == OptimizeTermination::Unbounded {
            assert!(result.verification_report().unwrap().is_accepted());
            saw_unbounded = true;
            break;
        }
        assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
        saw_nested_stop |= result
            .counts()
            .nested()
            .provider(EvaluationRole::Factorization)
            != 0;
    }
    assert!(saw_nested_stop && saw_unbounded);
}

#[test]
fn branch_and_bound_charges_relaxations_to_parent_node_and_work_budgets() {
    let relaxation = scioxide_optimize::LpProblem::new(vec![-1.0_f64])
        .unwrap()
        .with_bounds(scioxide_core::Bounds::new(vec![(Some(0.0), Some(2.5))]).unwrap())
        .unwrap();
    let baseline = scioxide_optimize::SimplexSolver::default()
        .solve(&relaxation)
        .unwrap();
    let problem = scioxide_optimize::MipProblem::new(relaxation, vec![0]).unwrap();
    let solver = scioxide_optimize::BranchAndBound::default()
        .with_presolve(false)
        .with_branching(scioxide_optimize::MipBranching::MostFractional)
        .with_limits(
            scioxide_optimize::MipLimits::default()
                .with_cut_rounds(0)
                .with_repair(false),
        );
    for kind in [BudgetKind::Node, BudgetKind::Evaluation] {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(kind, Budget::ONE);
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                parent_identity(103, 1),
                &mut budgets,
            )
            .unwrap();
        let mut workspace = solver.workspace(&problem).unwrap();
        let mut result = scioxide_optimize::MipResult::empty();
        solver
            .solve_into(&problem, &mut result, &mut workspace, &mut context)
            .unwrap();
        // A native box root needs no provider work. With an evaluation cap,
        // the first provider is the factorization for an appended branch row.
        let admitted_nodes = if kind == BudgetKind::Node { 1 } else { 2 };
        assert_eq!(result.nodes(), admitted_nodes);
        assert_eq!(
            context.budget_report(BudgetKind::Node).consumed(),
            admitted_nodes
        );
        assert_eq!(
            result.counts().total().unwrap(),
            context.counts().total().unwrap()
        );
        assert_eq!(result.iterations(), result.counts().nested().iterations());
        assert_eq!(result.statistics().iterations(), result.iterations());
        assert!(!result.is_success());
        assert_eq!(
            result.trust(),
            scioxide_core::PartialResultTrust::DiagnosticOnly
        );
        if kind == BudgetKind::Node {
            assert_eq!(result.termination(), OptimizeTermination::NodeLimit);
            assert_eq!(result.iterations(), baseline.iterations());
            assert!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::LinearOperator)
                    >= baseline
                        .counts()
                        .total()
                        .unwrap()
                        .provider(EvaluationRole::LinearOperator)
            );
            assert_eq!(result.x(), &[2.5]);
        } else {
            assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
            assert_eq!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Factorization),
                1
            );
            assert_eq!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::LinearOperator),
                0
            );
            assert_eq!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::LinearSolve),
                0
            );
        }
    }
}

#[test]
fn qp_admits_convexity_and_linear_work_and_keeps_valid_projected_points() {
    let problem = scioxide_optimize::QpProblem::new(
        scioxide_linalg::Matrix::from_vec(1, 1, vec![2.0_f64]).unwrap(),
        vec![-4.0],
    )
    .unwrap();
    let solver = scioxide_optimize::QpSolver::default();
    let baseline = solver.solve(&problem).unwrap();
    assert!(baseline.is_success());
    assert_eq!(
        baseline
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Factorization),
        3
    );
    assert_eq!(
        baseline
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::LinearOperator),
        2
    );
    assert_eq!(
        baseline
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::LinearSolve),
        1
    );
    for bounded in [false, true] {
        let problem = if bounded {
            problem
                .clone()
                .with_bounds(scioxide_core::Bounds::new(vec![(Some(0.0), Some(1.0))]).unwrap())
                .unwrap()
        } else {
            problem.clone()
        };
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE);
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                parent_identity(104, 1),
                &mut budgets,
            )
            .unwrap();
        let mut workspace = solver.workspace(&problem).unwrap();
        let mut output = scioxide_optimize::ProgrammingResult::empty();
        let outcome = solver
            .solve_into(&problem, &mut output, &mut workspace, &mut context)
            .map(|()| output);
        assert_eq!(
            context
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Factorization),
            1
        );
        assert_eq!(
            context
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::LinearSolve),
            0
        );
        if bounded {
            let result = outcome.unwrap();
            assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
            assert_eq!(result.x(), &[1.0]);
            assert_eq!(
                result.counts().total().unwrap(),
                context.counts().total().unwrap()
            );
        } else {
            assert!(matches!(
                (outcome)
                    .as_ref()
                    .map_err(scioxide_core::SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::ConvergenceFailure { .. })
            ));
        }
    }
}

#[test]
fn least_squares_gradient_and_step_roles_are_independent() {
    let problem = LinearResidual {
        residuals: core::cell::Cell::new(0),
        jacobians: core::cell::Cell::new(0),
        fail_jacobian: false,
    };
    let options = scioxide_optimize::LeastSquaresOptions::default()
        .with_globalization(scioxide_optimize::LeastSquaresGlobalization::gauss_newton());
    let loose_gradient = scioxide_optimize::LeastSquares::new(
        options
            .with_gradient_tolerance(scioxide_core::GradientTolerance::new(100.0, 0.0).unwrap())
            .with_step_tolerance(scioxide_core::StepTolerance::new(0.0, 0.0).unwrap()),
    )
    .unwrap();
    let loose_step = scioxide_optimize::LeastSquares::new(
        options
            .with_gradient_tolerance(scioxide_core::GradientTolerance::new(0.0, 0.0).unwrap())
            .with_step_tolerance(scioxide_core::StepTolerance::new(100.0, 0.0).unwrap()),
    )
    .unwrap();
    let gradient_result = loose_gradient.solve(&problem, &[0.0]).unwrap();
    assert_eq!(
        gradient_result.termination(),
        OptimizeTermination::GradientSmall
    );
    let step_result = loose_step.solve(&problem, &[0.0]).unwrap();
    assert_eq!(step_result.termination(), OptimizeTermination::StepSmall);
    assert!(!step_result.is_success());
}

#[test]
fn nested_bfgs_proposals_do_not_consume_parent_local_limit() {
    let solver = scioxide_optimize::Bfgs::default();
    let objective = Quadratic::new(false);
    let mut workspace = solver.workspace(1).unwrap();
    let mut result = OptimizeResult::<f64>::empty();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Proposal, Budget::ONE);
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            parent_identity(106, 1),
            &mut budgets,
        )
        .unwrap();
    solver
        .minimize_into(
            &objective,
            &[1.0],
            &mut result,
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert!(result.is_success());
    assert!(result.fun() <= 1e-12);
    assert_eq!(result.counts().own().proposals(), 2);
    assert_eq!(result.counts().own().rejected(), 1);
    assert_eq!(result.counts().own().accepted(), 1);
    assert_eq!(
        result
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Objective),
        3
    );
    assert_eq!(
        result
            .counts()
            .total()
            .unwrap()
            .provider(EvaluationRole::Gradient),
        3
    );
    assert_eq!(result.counts(), workspace.last_attempt().unwrap().counts());
    assert_eq!(context.accepted_state().unwrap().revision(), 0);
}

#[test]
fn final_assessment_does_not_count_as_an_extra_accepted_iterate() {
    let solver = scioxide_optimize::Bfgs::default();
    let objective = Quadratic::new(false);
    let mut workspace = solver.workspace(1).unwrap();
    let mut result = OptimizeResult::<f64>::empty();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            parent_identity(107, 1),
            &mut budgets,
        )
        .unwrap();
    solver
        .minimize_into(
            &objective,
            &[1.0],
            &mut result,
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert!(result.is_success());
    assert_eq!(result.counts().own().proposals(), 2);
    assert_eq!(result.counts().own().rejected(), 1);
    assert_eq!(result.counts().own().accepted(), 1);
    assert_eq!(context.accepted_state().unwrap().revision(), 0);
    assert_eq!(
        workspace.last_accepted().unwrap().progress().trust(),
        result.trust()
    );
}

#[test]
fn stationarity_reports_retain_original_objective_units() {
    macro_rules! check {
        ($scalar:ty, $scales:expr) => {{
            struct ScaledQuadratic($scalar);
            impl ObjectiveFunction<$scalar> for ScaledQuadratic {
                fn dim(&self) -> usize {
                    1
                }
                fn eval(&self, x: &[$scalar]) -> Result<$scalar, SciOxideError> {
                    Ok(self.0 * x[0] * x[0])
                }
                fn gradient_into(
                    &self,
                    x: &[$scalar],
                    output: &mut [$scalar],
                ) -> Result<bool, SciOxideError> {
                    output[0] = 2.0 * self.0 * x[0];
                    Ok(true)
                }
            }
            let solver = scioxide_optimize::Bfgs::new(
                scioxide_optimize::BfgsOptions::default().with_gradient_tolerance(
                    scioxide_core::GradientTolerance::new(0.0, 0.0).unwrap(),
                ),
            );
            for scale in $scales {
                let mut state = SolveState::new();
                let mut budgets = SolveBudgets::unbounded()
                    .with_limit(BudgetKind::Evaluation, Budget::try_from(2).unwrap());
                let mut context = state
                    .begin_solve(
                        ExecutionContext::sequential(),
                        parent_identity(108, 1),
                        &mut budgets,
                    )
                    .unwrap();
                let mut workspace = solver.workspace(1).unwrap();
                let mut result = OptimizeResult::<$scalar>::empty();
                solver
                    .minimize_into(
                        &ScaledQuadratic(scale),
                        &[1.0],
                        &mut result,
                        &mut workspace,
                        &mut context,
                    )
                    .unwrap();
                let report = result.diagnostics().stationarity_report().unwrap().unwrap();
                assert_eq!(report.raw_norm(), 2.0 * scale);
                assert_eq!(report.scaled_norm(), report.raw_norm());
                assert_eq!(
                    report.scaling().rule(),
                    scioxide_core::ResidualScalingRule::Unscaled
                );
                assert_eq!(report.norm(), scioxide_core::NormSelection::L2);
                assert!(
                    result
                        .diagnostics()
                        .primal_feasibility_report()
                        .unwrap()
                        .is_none()
                );
                assert!(result.diagnostics().certified_optimality_gap().is_none());
                assert!(!result.is_success());
            }
        }};
    }
    check!(f32, [2.0_f32.powi(-60), 1.0, 2.0_f32.powi(60)]);
    check!(f64, [2.0_f64.powi(-500), 1.0, 2.0_f64.powi(500)]);
}

#[test]
fn injected_clock_controls_partial_stops_and_provider_failure() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    use core::time::Duration;
    use scioxide_core::{CapabilityError, Clock, MonotonicInstant};

    struct ControlledClock(AtomicUsize);
    impl Clock for ControlledClock {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            match self.0.load(Ordering::Relaxed) {
                0 => Ok(MonotonicInstant::from_duration_since_epoch(Duration::ZERO)),
                1 => Ok(MonotonicInstant::from_duration_since_epoch(
                    Duration::from_secs(1),
                )),
                _ => Err(CapabilityError::new("injected clock failure")),
            }
        }
    }
    struct AdvanceClock<'a> {
        clock: &'a ControlledClock,
        next: usize,
    }
    impl ObjectiveFunction<f64> for AdvanceClock<'_> {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            self.clock.0.store(self.next, Ordering::Relaxed);
            Ok(x[0] * x[0])
        }
    }
    let solver = Bfgs::new(
        scioxide_optimize::BfgsOptions::default().with_time_limit(Duration::from_millis(1)),
    );
    let mut workspace = solver.workspace(1).unwrap();
    for next in [1, 2] {
        let clock = ControlledClock(AtomicUsize::new(0));
        let clock_binding = scioxide_core::ClockBinding::new(&clock);
        let objective = AdvanceClock {
            clock: &clock,
            next,
        };
        let context = ExecutionContext::sequential()
            .with_clock(&clock_binding)
            .unwrap();
        let mut output = OptimizeResult::<f64>::empty();
        let outcome =
            solver.minimize_into(&objective, &[2.0], &mut output, &mut workspace, context);
        if next == 1 {
            outcome.unwrap();
            assert_eq!(output.termination(), OptimizeTermination::TimeLimit);
            assert_eq!(output.x(), &[2.0]);
            assert_eq!(output.fun(), 4.0);
            assert!(output.grad().is_none());
        } else {
            assert!(matches!(
                (outcome)
                    .as_ref()
                    .map_err(scioxide_core::SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::CapabilityFailure { .. })
            ));
            assert_eq!(output.termination(), OptimizeTermination::NotStarted);
        }
        assert_eq!(
            workspace
                .last_attempt()
                .unwrap()
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Objective),
            1
        );
        assert_eq!(
            workspace
                .last_attempt()
                .unwrap()
                .counts()
                .total()
                .unwrap()
                .provider(EvaluationRole::Gradient),
            0
        );
        assert_eq!(
            workspace.last_attempt().unwrap().counts().own().accepted(),
            0
        );
    }
}

#[test]
fn requested_time_limit_without_clock_is_rejected_before_callbacks() {
    let solver = Bfgs::new(
        scioxide_optimize::BfgsOptions::default().with_time_limit(core::time::Duration::ZERO),
    );
    let objective = Quadratic::new(false);
    assert!(matches!(
        (solver.minimize(&objective, &[2.0]))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::CapabilityUnavailable {
            capability: scioxide_core::Capability::Clock,
            ..
        })
    ));
    assert_eq!(objective.objectives.get(), 0);
    assert_eq!(objective.gradients.get(), 0);
}

#[test]
fn dense_operator_work_is_admitted_without_creating_worker_teams() {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionPlan, Executor, PartitionFn};

    struct CallerExecutor;
    impl Executor for CallerExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(4).unwrap()
        }
        fn execute(
            &self,
            _: ExecutionContext<'_>,
            _: ExecutionPlan,
            _: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("plan"),
                InvalidInputReason::Malformed,
                "sequential optimizer kernels must not create worker teams",
            ))
        }
    }
    for limited in [false, true] {
        let solver = Bfgs::<f64>::default();
        let objective = Quadratic::new(false);
        let mut workspace = solver.workspace(1).unwrap();
        let mut result = OptimizeResult::<f64>::empty();
        let mut state = SolveState::new();
        let mut budgets = if limited {
            SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::ONE.saturating_add(1))
        } else {
            SolveBudgets::unbounded()
        };
        let mut context = state
            .begin_solve(
                ExecutionContext::new(&CallerExecutor),
                parent_identity(33, 1),
                &mut budgets,
            )
            .unwrap();
        solver
            .minimize_into(
                &objective,
                &[2.0],
                &mut result,
                &mut workspace,
                &mut context,
            )
            .unwrap();
        if limited {
            assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
            assert_eq!(result.x(), &[2.0]);
            assert_eq!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::LinearOperator),
                0
            );
        } else {
            assert!(result.is_success());
            assert!(result.fun() <= 1e-12);
            assert_eq!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::LinearOperator),
                2
            );
        }
    }
}

#[test]
fn trust_region_forwards_executor_without_creating_worker_teams() {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionPlan, Executor, PartitionFn};

    struct CallerExecutor;
    impl Executor for CallerExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(4).unwrap()
        }
        fn execute(
            &self,
            _: ExecutionContext<'_>,
            _: ExecutionPlan,
            _: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("plan"),
                InvalidInputReason::Malformed,
                "sequential optimizer kernels must not create worker teams",
            ))
        }
    }
    for limited in [false, true] {
        let solver = TrustRegionOptimizer::<f64>::default();
        let objective = Quadratic::new(false);
        let mut state = SolveState::new();
        let mut budgets = if limited {
            SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::ONE.saturating_add(1))
        } else {
            SolveBudgets::unbounded()
        };
        let mut context = state
            .begin_solve(
                ExecutionContext::new(&CallerExecutor),
                parent_identity(33, 1),
                &mut budgets,
            )
            .unwrap();
        let mut workspace = solver
            .workspace(scioxide_optimize::TrustRegionWorkspaceRequest::dense(1))
            .unwrap();
        let mut result = OptimizeResult::<f64>::empty();
        solver
            .minimize_into(
                &objective,
                &[2.0],
                &mut result,
                &mut workspace,
                &mut context,
            )
            .unwrap();
        if limited {
            assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
            assert_eq!(result.x(), &[2.0]);
            assert_eq!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::LinearOperator),
                0
            );
        } else {
            assert!(result.is_success());
            assert!(result.fun() <= 1e-12);
            assert!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::LinearOperator)
                    > 0
            );
            assert!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Factorization)
                    > 0
            );
            assert!(
                result
                    .counts()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::LinearSolve)
                    > 0
            );
        }
    }
}

#[test]
fn custom_line_search_metadata_cannot_replace_observed_work_counts() {
    use scioxide_optimize::{BfgsOptions, LineSearch, LineSearchResult, LineSearchScope};

    #[derive(Debug)]
    struct ReportedCounts;
    impl LineSearch<f64> for ReportedCounts {
        fn search_into<F: ObjectiveFunction<f64> + ?Sized>(
            &self,
            f: &F,
            x: &[f64],
            _: f64,
            direction: &[f64],
            _: f64,
            _: scioxide_differentiate::DerivativeStrategy,
            trial: &mut [f64],
            _: &mut [f64],
            _: &mut [f64],
            _: &mut [f64],
            _: &mut scioxide_optimize::DerivativeWorkspace<f64>,
            control: LineSearchScope<'_>,
            _: Option<f64>,
        ) -> Result<LineSearchResult<f64>, SciOxideError> {
            control.propose()?;
            trial[0] = x[0] + 0.5 * direction[0];
            let value = f.eval(trial)?;
            LineSearchResult::try_accepted(0.5, value, false, None, usize::MAX, usize::MAX)
        }
    }
    let solver = Bfgs::new(BfgsOptions::default().with_line_search(ReportedCounts));
    let result = solver.minimize(&Quadratic::new(false), &[2.0]).unwrap();
    assert!(result.is_success());
    assert_eq!(result.fun_evals(), 2);
    assert_eq!(result.grad_evals(), 2);
    assert_eq!(result.counts().own().proposals(), 1);
}

#[test]
fn advancing_deadlines_publish_the_last_observed_point_during_updates() {
    use core::sync::atomic::{AtomicU64, Ordering};
    use core::time::Duration;
    use scioxide_core::{CapabilityError, Clock, MonotonicInstant};

    struct AdvancingClock(AtomicU64);
    impl Clock for AdvancingClock {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                Duration::from_nanos(self.0.fetch_add(1, Ordering::Relaxed)),
            ))
        }
    }
    struct CoupledQuadratic(Cell<usize>);
    impl ObjectiveFunction<f64> for CoupledQuadratic {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            self.0.set(self.0.get() + 1);
            Ok(0.5 * x[0] * x[0] + 2. * x[0] * x[1] + 2.5 * x[1] * x[1])
        }
        fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
            output[0] = x[0] + 2. * x[1];
            output[1] = 2. * x[0] + 5. * x[1];
            Ok(true)
        }
    }
    let mut stopped_during_update = false;
    for ticks in 1..160 {
        let clock = AdvancingClock(AtomicU64::new(0));
        let clock_binding = scioxide_core::ClockBinding::new(&clock);
        let objective = CoupledQuadratic(Cell::new(0));
        let solver = Bfgs::new(
            scioxide_optimize::BfgsOptions::default().with_time_limit(Duration::from_nanos(ticks)),
        );
        let mut workspace = solver.workspace(2).unwrap();
        let mut result = OptimizeResult::<f64>::empty();
        let outcome = solver.minimize_into(
            &objective,
            &[2., -1.],
            &mut result,
            &mut workspace,
            ExecutionContext::sequential()
                .with_clock(&clock_binding)
                .unwrap(),
        );
        if objective.0.get() > 0 {
            assert!(
                outcome.is_ok(),
                "deadline {ticks} lost its observed partial state: {outcome:?}"
            );
            outcome.unwrap();
            assert!(result.is_success() || result.termination() == OptimizeTermination::TimeLimit);
            let x = result.x();
            let independent = 0.5 * x[0] * x[0] + 2. * x[0] * x[1] + 2.5 * x[1] * x[1];
            assert!((result.fun() - independent).abs() <= 1e-14);
            let counts = result.counts().total().unwrap();
            assert_eq!(counts.provider(EvaluationRole::Factorization), 0);
            stopped_during_update |= result.termination() == OptimizeTermination::TimeLimit
                && result.iterations() == 0
                && counts.provider(EvaluationRole::LinearOperator) >= 2;
        }
    }
    assert!(stopped_during_update);
}

#[test]
fn nelder_mead_injected_clock_controls_partial_stops_and_provider_failure() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    use core::time::Duration;
    use scioxide_core::{CapabilityError, Clock, MonotonicInstant};

    struct ControlledClock(AtomicUsize);
    impl Clock for ControlledClock {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            match self.0.load(Ordering::Relaxed) {
                0 => Ok(MonotonicInstant::from_duration_since_epoch(Duration::ZERO)),
                1 => Ok(MonotonicInstant::from_duration_since_epoch(
                    Duration::from_secs(1),
                )),
                _ => Err(CapabilityError::new("injected clock failure")),
            }
        }
    }
    struct AdvanceClock<'a> {
        clock: &'a ControlledClock,
        next: usize,
    }
    impl ObjectiveFunction<f64> for AdvanceClock<'_> {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            self.clock.0.store(self.next, Ordering::Relaxed);
            Ok(x[0] * x[0])
        }
    }
    let solver = scioxide_optimize::NelderMead::new(
        scioxide_optimize::NelderMeadOptions::default().with_time_limit(Duration::from_millis(1)),
    )
    .unwrap();
    let mut workspace = solver.workspace(1).unwrap();
    for next in [1, 2] {
        let clock = ControlledClock(AtomicUsize::new(0));
        let clock_binding = scioxide_core::ClockBinding::new(&clock);
        let objective = AdvanceClock {
            clock: &clock,
            next,
        };
        let mut output = OptimizeResult::<f64>::empty();
        let outcome = solver.minimize_into(
            &objective,
            &[2.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential()
                .with_clock(&clock_binding)
                .unwrap(),
        );
        if next == 1 {
            outcome.unwrap();
            assert_eq!(output.termination(), OptimizeTermination::TimeLimit);
            assert_eq!(output.x(), &[2.0]);
            assert_eq!(output.fun(), 4.0);
            assert!(output.grad().is_none());
        } else {
            assert!(matches!(
                (outcome)
                    .as_ref()
                    .map_err(scioxide_core::SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::CapabilityFailure { .. })
            ));
            assert_eq!(output.termination(), OptimizeTermination::NotStarted);
        }
        let counts = workspace.last_attempt().unwrap().counts();
        assert_eq!(
            counts.total().unwrap().provider(EvaluationRole::Objective),
            1
        );
        assert_eq!(
            counts.total().unwrap().provider(EvaluationRole::Gradient),
            0
        );
        assert_eq!(counts.own().accepted(), 1);
    }
}

#[test]
fn nelder_mead_cancellation_preserves_validated_observations() {
    struct CancelAfterValue<'a>(&'a CancellationToken);
    impl ObjectiveFunction<f64> for CancelAfterValue<'_> {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            self.0.cancel();
            Ok(x[0] * x[0])
        }
    }
    let solver = scioxide_optimize::NelderMead::<f64>::default();
    let cancellation = CancellationToken::new();
    let mut workspace = solver.workspace(1).unwrap();
    let mut result = OptimizeResult::<f64>::empty();
    solver
        .minimize_into(
            &CancelAfterValue(&cancellation),
            &[2.0],
            &mut result,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::Cancelled);
    assert_eq!(result.fun(), 4.0);
    assert!(result.grad().is_none());
    assert_eq!(
        result.counts().total().unwrap().total_providers().unwrap(),
        1
    );
}

#[test]
fn trust_region_prepares_each_accepted_model_once_across_rejections_and_reuse() {
    use scioxide_linalg::Matrix;
    use std::cell::RefCell;
    struct Valley {
        hessians: RefCell<Vec<[f64; 2]>>,
    }
    impl ObjectiveFunction<f64> for Valley {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2))
        }
        fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
            out[0] = 2.0 * (x[0] - 1.0) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            out[1] = 200.0 * (x[1] - x[0] * x[0]);
            Ok(true)
        }
        fn hessian_into(&self, x: &[f64], out: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
            self.hessians.borrow_mut().push([x[0], x[1]]);
            out.as_mut_slice().copy_from_slice(&[
                2.0 - 400.0 * x[1] + 1200.0 * x[0] * x[0],
                -400.0 * x[0],
                -400.0 * x[0],
                200.0,
            ]);
            Ok(true)
        }
    }
    let solver = TrustRegionOptimizer::<f64>::default();
    let mut workspace = solver
        .workspace(scioxide_optimize::TrustRegionWorkspaceRequest::dense(2))
        .unwrap();
    let mut result = OptimizeResult::empty();
    let objective = Valley {
        hessians: RefCell::new(Vec::new()),
    };
    for start in [[-1.2, 1.0], [-1.2, 1.0]] {
        objective.hessians.borrow_mut().clear();
        solver
            .minimize_into(
                &objective,
                &start,
                &mut result,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(result.is_success());
        let calls = objective.hessians.borrow();
        assert!(calls.windows(2).all(|pair| pair[0] != pair[1]));
        let counts = result.counts().own();
        assert!(counts.rejected() > 0);
        assert_eq!(counts.provider(EvaluationRole::Hessian), calls.len());
        assert_eq!(counts.provider(EvaluationRole::Factorization), calls.len());
        assert_eq!(counts.provider(EvaluationRole::LinearSolve), calls.len());
        assert_eq!(calls.len(), counts.accepted());
        assert!((result.x()[0] - 1.0).abs() < 1e-4);
    }
}
