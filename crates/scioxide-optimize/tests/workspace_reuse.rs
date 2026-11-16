//! Public prepared-solver, work-counter, and allocation contracts for optimization-workspaces work.

#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::{Bounds, SciOxideError, Tolerance};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    AugmentedLagrangian, Bfgs, ConstrainedProblem, ConstraintFn, ConstraintFunction, Constraints,
    GradientDescent, InteriorPoint, Lbfgs, LbfgsWorkspace, LeastSquares, LeastSquaresFn,
    LeastSquaresGlobalization, LeastSquaresProblem, LeastSquaresResult,
    LeastSquaresWorkspaceRequest, MinimizerProgress, NewtonOptimizer, NonlinearCg, ObjectiveFn,
    ObjectiveFunction, OptimizeResult, PenaltyMethod, Sqp, TrustRegionOptimizer,
    TrustRegionWorkspaceRequest,
};

struct ScaledQuadratic {
    dimension: usize,
}

impl ObjectiveFunction<f64> for ScaledQuadratic {
    fn dim(&self) -> usize {
        self.dimension
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x.iter()
            .enumerate()
            .map(|(index, &value)| (index + 1) as f64 * value * value)
            .sum())
    }

    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output
            .iter_mut()
            .zip(x)
            .enumerate()
            .for_each(|(index, (gradient, &value))| {
                *gradient = 2.0 * (index + 1) as f64 * value;
            });
        Ok(true)
    }

    fn hessian_into(&self, _x: &[f64], output: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
        output.as_mut_slice().fill(0.0);
        (0..self.dimension).for_each(|index| {
            output.as_mut_slice()[index * self.dimension + index] = 2.0 * (index + 1) as f64;
        });
        Ok(true)
    }
}

struct DiagonalLeastSquares {
    dimension: usize,
}

struct FailingObjective;

impl ObjectiveFunction<f64> for FailingObjective {
    fn dim(&self) -> usize {
        2
    }

    fn eval(&self, _x: &[f64]) -> Result<f64, SciOxideError> {
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "injected callback failure",
        ))
    }
}

struct FailingLeastSquares;

impl LeastSquaresProblem<f64> for FailingLeastSquares {
    fn param_dim(&self) -> usize {
        2
    }

    fn residual_dim(&self) -> usize {
        2
    }

    fn residual_into(&self, _x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
        output[0] = 123.0;
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("x"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "injected callback failure",
        ))
    }
}

impl LeastSquaresProblem<f64> for DiagonalLeastSquares {
    fn param_dim(&self) -> usize {
        self.dimension
    }

    fn residual_dim(&self) -> usize {
        self.dimension
    }

    fn residual_into(&self, x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
        output
            .iter_mut()
            .zip(x)
            .for_each(|(residual, &value)| *residual = value - 1.0);
        Ok(())
    }

    fn jacobian_into(&self, _x: &[f64], output: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
        output.as_mut_slice().fill(0.0);
        (0..self.dimension).for_each(|index| {
            output.as_mut_slice()[index * self.dimension + index] = 1.0;
        });
        Ok(true)
    }
}

fn assert_prepared_first_order<Solver, Workspace, Prepare, Run>(
    name: &str,
    solver: &Solver,
    dimension: usize,
    prepare: Prepare,
    run: Run,
) where
    Prepare: Fn(&Solver, usize) -> Workspace,
    Run: Fn(
        &Solver,
        &ScaledQuadratic,
        &[f64],
        &mut OptimizeResult<f64>,
        &mut Workspace,
    ) -> Result<(), SciOxideError>,
{
    let objective = ScaledQuadratic { dimension };
    let initial = vec![1.0; dimension];
    let mut workspace = prepare(solver, dimension);
    let mut result = OptimizeResult::empty();
    result.try_reserve(dimension).unwrap();
    // The first prepared call already allocates nothing.
    run(solver, &objective, &initial, &mut result, &mut workspace).unwrap();
    assert!(result.is_success(), "{name}: {}", result.message());
    let call = run(solver, &objective, &initial, &mut result, &mut workspace);
    call.unwrap();
    assert!(result.is_success(), "{name}: {}", result.message());
    assert!(result.fun() <= 1e-10, "{name}: residual {}", result.fun());
}

#[test]
fn first_order_prepared_paths_are_zero_allocation_at_n20_and_n100() {
    for dimension in [20, 100] {
        let gradient_descent = GradientDescent::<f64>::default();
        assert_prepared_first_order(
            "gradient descent prepared",
            &gradient_descent,
            dimension,
            |solver, n| solver.workspace(n).unwrap(),
            |solver, objective, initial, result, workspace| {
                solver.minimize_into(
                    objective,
                    initial,
                    result,
                    workspace,
                    scioxide_core::ExecutionContext::sequential(),
                )
            },
        );

        let nonlinear_cg = NonlinearCg::<f64>::default();
        assert_prepared_first_order(
            "nonlinear CG prepared",
            &nonlinear_cg,
            dimension,
            |solver, n| solver.workspace(n).unwrap(),
            |solver, objective, initial, result, workspace| {
                solver.minimize_into(
                    objective,
                    initial,
                    result,
                    workspace,
                    scioxide_core::ExecutionContext::sequential(),
                )
            },
        );

        let bfgs = Bfgs::<f64>::default();
        assert_prepared_first_order(
            "BFGS prepared",
            &bfgs,
            dimension,
            |solver, n| solver.workspace(n).unwrap(),
            |solver, objective, initial, result, workspace| {
                solver.minimize_into(
                    objective,
                    initial,
                    result,
                    workspace,
                    scioxide_core::ExecutionContext::sequential(),
                )
            },
        );

        let lbfgs = Lbfgs::<f64>::default();
        assert_prepared_first_order(
            "L-BFGS prepared",
            &lbfgs,
            dimension,
            |solver, n| solver.workspace(n).unwrap(),
            |solver, objective, initial, result, workspace| {
                solver.minimize_into(
                    objective,
                    initial,
                    result,
                    workspace,
                    scioxide_core::ExecutionContext::sequential(),
                )
            },
        );
    }
}

#[test]
fn second_order_prepared_paths_are_zero_allocation() {
    let dimension = 20;
    let objective = ScaledQuadratic { dimension };
    let initial = vec![1.0; dimension];

    let newton = NewtonOptimizer::<f64>::default();
    let mut newton_workspace = newton.workspace(dimension).unwrap();
    let mut newton_result = OptimizeResult::empty();
    newton_result.try_reserve(dimension).unwrap();
    newton
        .minimize_into(
            &objective,
            &initial,
            &mut newton_result,
            &mut newton_workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    newton
        .minimize_into(
            &objective,
            &initial,
            &mut newton_result,
            &mut newton_workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(newton_result.is_success());
    assert!(
        newton_workspace
            .last_attempt()
            .unwrap()
            .counts()
            .own()
            .provider(scioxide_core::EvaluationRole::Factorization)
            > 0
    );

    let trust_region = TrustRegionOptimizer::<f64>::default();
    let mut trust_workspace = trust_region
        .workspace(TrustRegionWorkspaceRequest::dense(dimension))
        .unwrap();
    let mut trust_result = OptimizeResult::empty();
    trust_result.try_reserve(dimension).unwrap();
    trust_region
        .minimize_into(
            &objective,
            &initial,
            &mut trust_result,
            &mut trust_workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    trust_region
        .minimize_into(
            &objective,
            &initial,
            &mut trust_result,
            &mut trust_workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(trust_result.is_success());
}

#[test]
fn least_squares_prepared_paths_are_zero_allocation_at_n20_and_n100() {
    for dimension in [20, 100] {
        let problem = DiagonalLeastSquares { dimension };
        let initial = vec![0.0; dimension];
        for (_name, globalization, bound) in [
            (
                "Gauss-Newton prepared",
                LeastSquaresGlobalization::gauss_newton(),
                1e-16,
            ),
            (
                "Levenberg-Marquardt prepared",
                LeastSquaresGlobalization::levenberg_marquardt(),
                1e-12,
            ),
        ] {
            let solver = least_squares(globalization);
            let mut workspace = solver
                .workspace(LeastSquaresWorkspaceRequest::of(&problem).unwrap())
                .unwrap();
            let mut result = LeastSquaresResult::empty();
            result.try_reserve(dimension, dimension).unwrap();
            for _ in 0..2 {
                solver
                    .solve_into(
                        &problem,
                        &initial,
                        &mut result,
                        &mut workspace,
                        scioxide_core::ExecutionContext::sequential(),
                    )
                    .unwrap();
                assert!(result.is_success());
                assert!(result.cost().unwrap() <= bound);
            }
        }
    }
}

fn least_squares(globalization: LeastSquaresGlobalization<f64>) -> LeastSquares<f64> {
    let tolerance = Tolerance::from_decimal_places(8, 8);
    LeastSquares::new(
        scioxide_optimize::LeastSquaresOptions::default()
            .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(tolerance))
            .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(tolerance))
            .with_globalization(globalization),
    )
    .unwrap()
}

struct TwoDimensionalQuadratic;
struct LinearEquality;
struct QuadraticInequality;

impl ConstraintFunction<f64> for LinearEquality {
    fn dim(&self) -> usize {
        2
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] + x[1] - 1.0)
    }

    fn jacobian_into(&self, _x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output.copy_from_slice(&[1.0, 1.0]);
        Ok(true)
    }
}

impl ConstraintFunction<f64> for QuadraticInequality {
    fn dim(&self) -> usize {
        2
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] * x[0] + x[1] - 1.0)
    }
}

impl ObjectiveFunction<f64> for TwoDimensionalQuadratic {
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

fn equality_constraints() -> Constraints<f64> {
    Constraints::new(2)
        .unwrap()
        .equality(LinearEquality)
        .unwrap()
}

/// Objective and gradient work recorded by a prepared workspace.
fn attempt_work(record: &scioxide_core::AttemptRecord<MinimizerProgress<f64>>) -> (usize, usize) {
    let counts = record.counts().total().unwrap();
    (
        counts.provider(scioxide_core::EvaluationRole::Objective),
        counts.provider(scioxide_core::EvaluationRole::Gradient),
    )
}

fn assert_optimize_parity(
    name: &str,
    allocating: &OptimizeResult<f64>,
    prepared: &OptimizeResult<f64>,
    (objective_evaluations, gradient_evaluations): (usize, usize),
) {
    assert_eq!(allocating.x(), prepared.x(), "{name}: solution");
    assert_eq!(allocating.fun(), prepared.fun(), "{name}: objective");
    assert_eq!(allocating.grad(), prepared.grad(), "{name}: gradient");
    assert_eq!(
        allocating.iterations(),
        prepared.iterations(),
        "{name}: iterations"
    );
    assert_eq!(
        allocating.fun_evals(),
        prepared.fun_evals(),
        "{name}: objective evaluations"
    );
    assert_eq!(
        allocating.grad_evals(),
        prepared.grad_evals(),
        "{name}: gradient evaluations"
    );
    assert_eq!(
        allocating.termination(),
        prepared.termination(),
        "{name}: termination"
    );
    assert_eq!(allocating.message(), prepared.message(), "{name}: message");
    assert_eq!(
        objective_evaluations,
        prepared.fun_evals(),
        "{name}: workspace objective counter"
    );
    assert_eq!(
        gradient_evaluations,
        prepared.grad_evals(),
        "{name}: workspace gradient counter"
    );
}

fn assert_least_squares_parity(
    name: &str,
    allocating: &LeastSquaresResult<f64>,
    prepared: &LeastSquaresResult<f64>,
    (residual_evaluations, jacobian_evaluations): (usize, usize),
) {
    assert_eq!(allocating.x(), prepared.x(), "{name}: parameters");
    assert_eq!(
        allocating.cost().unwrap(),
        prepared.cost().unwrap(),
        "{name}: cost"
    );
    assert_eq!(
        allocating.residual(),
        prepared.residual(),
        "{name}: residual"
    );
    assert_eq!(allocating.iterations(), prepared.iterations(), "{name}");
    assert_eq!(allocating.fun_evals(), prepared.fun_evals(), "{name}");
    assert_eq!(allocating.jac_evals(), prepared.jac_evals(), "{name}");
    assert_eq!(allocating.termination(), prepared.termination(), "{name}");
    assert_eq!(allocating.message(), prepared.message(), "{name}");
    assert_eq!(residual_evaluations, prepared.fun_evals());
    assert_eq!(jacobian_evaluations, prepared.jac_evals());
}

/// Failed and preflight-rejected calls leave a published result unchanged.
fn assert_transactional<W>(
    name: &str,
    workspace: &mut W,
    run: impl Fn(
        &dyn ObjectiveFunction<f64>,
        &[f64],
        &mut OptimizeResult<f64>,
        &mut W,
    ) -> Result<(), SciOxideError>,
) {
    let objective = ScaledQuadratic { dimension: 2 };
    let initial = [2.0, -1.0];
    let mut output = OptimizeResult::empty();
    run(&objective, &initial, &mut output, workspace).unwrap();
    let before = output.clone();
    assert!(run(&FailingObjective, &initial, &mut output, workspace).is_err());
    assert_optimize_result_unchanged(name, &before, &output);
    assert!(run(&objective, &[1.0], &mut output, workspace).is_err());
    assert_optimize_result_unchanged(name, &before, &output);
}

fn assert_optimize_result_unchanged(
    name: &str,
    before: &OptimizeResult<f64>,
    after: &OptimizeResult<f64>,
) {
    assert_eq!(before.x(), after.x(), "{name}: solution");
    assert_eq!(before.fun(), after.fun(), "{name}: objective");
    assert_eq!(before.grad(), after.grad(), "{name}: gradient");
    assert_eq!(
        before.iterations(),
        after.iterations(),
        "{name}: iterations"
    );
    assert_eq!(
        before.fun_evals(),
        after.fun_evals(),
        "{name}: function count"
    );
    assert_eq!(
        before.grad_evals(),
        after.grad_evals(),
        "{name}: gradient count"
    );
    assert_eq!(
        before.termination(),
        after.termination(),
        "{name}: termination"
    );
    assert_eq!(before.message(), after.message(), "{name}: message");
}

fn assert_least_squares_result_unchanged(
    before: &LeastSquaresResult<f64>,
    after: &LeastSquaresResult<f64>,
) {
    assert_eq!(before.x(), after.x());
    assert_eq!(before.cost().unwrap(), after.cost().unwrap());
    assert_eq!(before.residual(), after.residual());
    assert_eq!(before.iterations(), after.iterations());
    assert_eq!(before.fun_evals(), after.fun_evals());
    assert_eq!(before.jac_evals(), after.jac_evals());
    assert_eq!(before.termination(), after.termination());
    assert_eq!(before.message(), after.message());
}

#[test]
fn allocating_and_prepared_lifecycles_have_identical_results_and_counters() {
    let objective = ScaledQuadratic { dimension: 2 };
    let initial = [2.0, -1.0];

    macro_rules! check_optimizer {
        ($name:literal, $solver:expr) => {
            check_optimizer!($name, $solver, 2)
        };
        ($name:literal, $solver:expr, $request:expr) => {{
            let solver = $solver;
            let allocating = solver.minimize(&objective, &initial).unwrap();
            let mut workspace = solver.workspace($request).unwrap();
            let mut prepared = OptimizeResult::empty();
            solver
                .minimize_into(
                    &objective,
                    &initial,
                    &mut prepared,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert_optimize_parity(
                $name,
                &allocating,
                &prepared,
                attempt_work(workspace.last_attempt().unwrap()),
            );
        }};
    }

    check_optimizer!("gradient descent", GradientDescent::<f64>::default());
    check_optimizer!("nonlinear CG", NonlinearCg::<f64>::default());
    check_optimizer!("BFGS", Bfgs::<f64>::default());
    check_optimizer!("L-BFGS", Lbfgs::<f64>::default());
    check_optimizer!("Newton", NewtonOptimizer::<f64>::default());
    check_optimizer!(
        "trust region",
        TrustRegionOptimizer::<f64>::default(),
        TrustRegionWorkspaceRequest::dense(2)
    );

    let problem = DiagonalLeastSquares { dimension: 2 };
    for (name, globalization) in [
        ("Gauss-Newton", LeastSquaresGlobalization::gauss_newton()),
        (
            "Levenberg-Marquardt",
            LeastSquaresGlobalization::levenberg_marquardt(),
        ),
    ] {
        let solver = least_squares(globalization);
        let allocating = solver.solve(&problem, &[0.0, 0.0]).unwrap();
        let mut workspace = solver
            .workspace(LeastSquaresWorkspaceRequest::dense(2, 2))
            .unwrap();
        let mut prepared = LeastSquaresResult::empty();
        solver
            .solve_into(
                &problem,
                &[0.0, 0.0],
                &mut prepared,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential_native(),
            )
            .unwrap();
        let counts = workspace.last_attempt().unwrap().counts().total().unwrap();
        assert_least_squares_parity(
            name,
            &allocating,
            &prepared,
            (
                counts.provider(scioxide_core::EvaluationRole::Residual),
                counts.provider(scioxide_core::EvaluationRole::Jacobian),
            ),
        );
    }

    macro_rules! check_constrained {
        ($name:literal, $solver:expr, $constraints:expr, $initial:expr) => {{
            let solver = $solver;
            let constraints = $constraints;
            let problem = ConstrainedProblem::new(&TwoDimensionalQuadratic, &constraints).unwrap();
            let allocating = solver.minimize(&problem, &$initial).unwrap();
            let mut workspace = solver.workspace(&problem).unwrap();
            let mut prepared = OptimizeResult::empty();
            solver
                .minimize_into(
                    &problem,
                    &$initial,
                    &mut prepared,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert_optimize_parity(
                $name,
                &allocating,
                &prepared,
                attempt_work(workspace.last_attempt().unwrap()),
            );
        }};
    }
    check_constrained!(
        "SQP",
        Sqp::<f64>::default(),
        equality_constraints(),
        [2.0, 2.0]
    );
    check_constrained!(
        "penalty",
        PenaltyMethod::<f64>::default(),
        equality_constraints(),
        [2.0, 2.0]
    );
    check_constrained!(
        "augmented Lagrangian",
        AugmentedLagrangian::<f64>::default(),
        equality_constraints(),
        [2.0, 2.0]
    );
    let inequality = ConstraintFn::new(2, |x: &[f64]| Ok(1.0 - x[0] - x[1]));
    check_constrained!(
        "interior point",
        InteriorPoint::<f64>::default(),
        Constraints::new(2).unwrap().inequality(inequality).unwrap(),
        [2.0, 2.0]
    );
}

#[test]
fn prepared_result_commits_are_transactional_across_workspace_families() {
    let objective = ScaledQuadratic { dimension: 2 };
    let initial = [2.0, -1.0];

    let bfgs = Bfgs::<f64>::default();
    let mut bfgs_workspace = bfgs.workspace(2).unwrap();
    let mut bfgs_output = OptimizeResult::empty();
    bfgs.minimize_into(
        &objective,
        &initial,
        &mut bfgs_output,
        &mut bfgs_workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    let before = bfgs_output.clone();
    assert!(
        bfgs.minimize_into(
            &FailingObjective,
            &initial,
            &mut bfgs_output,
            &mut bfgs_workspace,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_optimize_result_unchanged("first order", &before, &bfgs_output);
    assert!(
        bfgs.minimize_into(
            &objective,
            &[1.0],
            &mut bfgs_output,
            &mut bfgs_workspace,
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_optimize_result_unchanged("first order", &before, &bfgs_output);
    let newton = NewtonOptimizer::<f64>::default();
    assert_transactional(
        "second order",
        &mut newton.workspace(2).unwrap(),
        |f, x, output, workspace| {
            newton.minimize_into(
                f,
                x,
                output,
                workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
        },
    );
    let sqp = Sqp::<f64>::default();
    let unconstrained = Constraints::new(2).unwrap();
    let problem = ConstrainedProblem::new(&objective, &unconstrained).unwrap();
    assert_transactional(
        "constrained",
        &mut sqp.workspace(&problem).unwrap(),
        |f, x, output, workspace| {
            sqp.minimize_into(
                &ConstrainedProblem::new(f, &unconstrained)?,
                x,
                output,
                workspace,
                scioxide_core::ExecutionContext::sequential(),
            )
        },
    );

    let solver = least_squares(LeastSquaresGlobalization::gauss_newton());
    let problem = DiagonalLeastSquares { dimension: 2 };
    let mut workspace = solver
        .workspace(LeastSquaresWorkspaceRequest::dense(2, 2))
        .unwrap();
    let mut output = LeastSquaresResult::empty();
    let sequential = scioxide_core::ExecutionContext::sequential;
    solver
        .solve_into(
            &problem,
            &[0.0, 0.0],
            &mut output,
            &mut workspace,
            sequential(),
        )
        .unwrap();
    let before = output.clone();
    assert!(
        solver
            .solve_into(
                &FailingLeastSquares,
                &[0.0, 0.0],
                &mut output,
                &mut workspace,
                sequential(),
            )
            .is_err()
    );
    assert_least_squares_result_unchanged(&before, &output);
    assert!(
        solver
            .solve_into(&problem, &[0.0], &mut output, &mut workspace, sequential())
            .is_err()
    );
    assert_least_squares_result_unchanged(&before, &output);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::PreflightRejected
    );
}

#[test]
fn constraint_values_and_jacobians_are_zero_allocation_after_preparation() {
    let constraints = Constraints::new(2)
        .unwrap()
        .equality(LinearEquality)
        .unwrap()
        .inequality(QuadraticInequality)
        .unwrap()
        .bounds(Bounds::new(vec![(Some(-1.0), Some(1.0)); 2]).unwrap())
        .unwrap();
    let point = [0.5, 0.25];
    let mut equalities = [0.0];
    let mut inequalities = [0.0; 5];
    let mut jacobian = Matrix::zeros(6, 2).unwrap();
    let mut workspace = constraints.workspace().unwrap();

    constraints
        .evaluate_into(&point, &mut equalities, &mut inequalities, &mut workspace)
        .unwrap();
    let _ = constraints
        .jacobian_into(
            &point,
            &equalities,
            &inequalities,
            DerivativeStrategy::default(),
            &mut jacobian,
            &mut workspace,
        )
        .unwrap();

    let stats = (|| {
        constraints.evaluate_into(&point, &mut equalities, &mut inequalities, &mut workspace)?;
        constraints.jacobian_into(
            &point,
            &equalities,
            &inequalities,
            DerivativeStrategy::default(),
            &mut jacobian,
            &mut workspace,
        )
    })()
    .unwrap();

    assert_eq!(equalities, [-0.25]);
    assert_eq!(inequalities, [-0.5, -1.5, -0.5, -1.25, -0.75]);
    assert_eq!(stats.constraint_evaluations(), 2);
    assert_eq!(stats.jacobian_evaluations(), 2);
}

#[test]
fn pure_and_native_objectives_preserve_solver_counters() {
    let pure = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1])).with_gradient(
        |x: &[f64], output| {
            output.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
            Ok(())
        },
    );
    let solver = Bfgs::<f64>::default();
    let pure_result = solver.minimize(&pure, &[2.0, -1.0]).unwrap();
    let native_result = solver
        .minimize(&TwoDimensionalQuadratic, &[2.0, -1.0])
        .unwrap();

    assert_eq!(pure_result.fun_evals(), native_result.fun_evals());
    assert_eq!(pure_result.grad_evals(), native_result.grad_evals());
    assert!((pure_result.fun() - native_result.fun()).abs() < 1e-12);
    pure_result
        .x()
        .iter()
        .zip(native_result.x())
        .for_each(|(&pure, &native)| assert!((pure - native).abs() < 1e-12));
}

#[test]
fn constrained_prepared_paths_are_zero_allocation() {
    let objective = TwoDimensionalQuadratic;
    let initial = [2.0, 2.0];
    let sequential = scioxide_core::ExecutionContext::sequential;
    macro_rules! check {
        ($name:literal, $solver:expr, $constraints:expr) => {{
            let solver = $solver;
            let constraints = $constraints;
            let problem = ConstrainedProblem::new(&objective, &constraints).unwrap();
            let mut workspace = solver.workspace(&problem).unwrap();
            let mut result = OptimizeResult::empty();
            result.try_reserve(2).unwrap();
            solver
                .minimize_into(
                    &problem,
                    &initial,
                    &mut result,
                    &mut workspace,
                    sequential(),
                )
                .unwrap();
            solver
                .minimize_into(
                    &problem,
                    &initial,
                    &mut result,
                    &mut workspace,
                    sequential(),
                )
                .unwrap();
            assert!(result.is_success(), "{result:?}");
        }};
    }
    check!(
        "SQP prepared",
        Sqp::<f64>::default(),
        equality_constraints()
    );
    check!(
        "penalty prepared",
        PenaltyMethod::<f64>::default(),
        equality_constraints()
    );
    check!(
        "augmented-Lagrangian prepared",
        AugmentedLagrangian::<f64>::default(),
        equality_constraints()
    );
    let inequality = ConstraintFn::new(2, |x: &[f64]| Ok(1.0 - x[0] - x[1]));
    check!(
        "interior-point prepared",
        InteriorPoint::<f64>::default(),
        Constraints::new(2).unwrap().inequality(inequality).unwrap()
    );
}

#[test]
fn pure_closure_adapters_and_dynamic_prepared_dispatch_work() {
    let objective = ObjectiveFn::new(2, |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]))
        .with_gradient(|x: &[f64], output| {
            output.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
            Ok(())
        })
        .with_hessian(|_: &[f64], output| {
            output.as_mut_slice().copy_from_slice(&[2.0, 0.0, 0.0, 2.0]);
            Ok(())
        });
    let dynamic: &dyn ObjectiveFunction<f64> = &objective;
    let solver = NewtonOptimizer::<f64>::default();
    let mut workspace = solver.workspace(2).unwrap();
    let mut result = OptimizeResult::empty();
    solver
        .minimize_into(
            dynamic,
            &[2.0, -1.0],
            &mut result,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(result.is_success());

    let problem = LeastSquaresFn::new(2, 2, |x: &[f64], output| {
        output.copy_from_slice(&[x[0] - 1.0, x[1] - 2.0]);
        Ok(())
    })
    .unwrap()
    .with_jacobian(|_: &[f64], output| {
        output.as_mut_slice().copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        Ok(())
    });
    let dynamic: &dyn LeastSquaresProblem<f64> = &problem;
    let solver = least_squares(LeastSquaresGlobalization::gauss_newton());
    let mut workspace = solver
        .workspace(LeastSquaresWorkspaceRequest::of(dynamic).unwrap())
        .unwrap();
    let mut result = LeastSquaresResult::empty();
    solver
        .solve_into(
            dynamic,
            &[0.0, 0.0],
            &mut result,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(result.is_success());
}

#[test]
fn dependent_equalities_and_workspace_growth_have_structured_outcomes() {
    let first = ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1] - 1.0));
    let second = ConstraintFn::new(2, |x: &[f64]| Ok(2.0 * x[0] + 2.0 * x[1] - 2.0));
    let constraints = Constraints::new(2)
        .unwrap()
        .equality(first)
        .unwrap()
        .equality(second)
        .unwrap();
    let problem = ConstrainedProblem::new(&TwoDimensionalQuadratic, &constraints).unwrap();
    let result = Sqp::<f64>::default()
        .minimize(&problem, &[2.0, 2.0])
        .unwrap();
    assert!(result.is_success());
    assert!((result.x()[0] - 0.5).abs() < 1e-8);
    assert!((result.x()[1] - 0.5).abs() < 1e-8);
    let diagnostics = sqp_of(&result);
    assert_eq!(diagnostics.active_rank(), 1);
    assert!(diagnostics.multipliers_ambiguous());
    assert!(result.diagnostics().primal_feasibility().unwrap() < 1e-8);
    assert!(result.diagnostics().stationarity().unwrap() < 1e-8);

    let lbfgs = Lbfgs::new(
        scioxide_optimize::LbfgsOptions::<f64>::default()
            .with_memory(scioxide_core::Budget::try_from(usize::MAX).unwrap()),
    );
    let requirements = lbfgs.workspace_requirements(usize::MAX).unwrap();
    let mut workspace = LbfgsWorkspace::<f64>::new();
    assert!(workspace.try_reserve(&requirements).is_err());
}

#[test]
fn nelder_mead_reuses_simplex_and_trial_storage() {
    let solver = scioxide_optimize::NelderMead::<f64>::default();
    let objective = ScaledQuadratic { dimension: 2 };
    let mut workspace = solver.workspace(2).unwrap();
    let mut output = OptimizeResult::empty();
    output.try_reserve(2).unwrap();
    let sequential = scioxide_core::ExecutionContext::sequential_native;
    solver
        .minimize_into(
            &objective,
            &[1.0, 2.0],
            &mut output,
            &mut workspace,
            sequential(),
        )
        .unwrap();
    let allocating = solver.minimize(&objective, &[1.0, 2.0]).unwrap();
    assert_optimize_parity(
        "Nelder-Mead",
        &allocating,
        &output,
        attempt_work(workspace.last_attempt().unwrap()),
    );
    assert!(output.is_success());
    assert!(output.fun() < 1e-10);
    let saved = output.clone();
    assert!(
        solver
            .minimize_into(
                &FailingObjective,
                &[1.0, 2.0],
                &mut output,
                &mut workspace,
                sequential(),
            )
            .is_err()
    );
    assert_optimize_result_unchanged("Nelder-Mead", &saved, &output);
    assert_eq!(attempt_work(workspace.last_attempt().unwrap()).0, 1);
}

#[test]
fn global_searches_reuse_population_restart_and_nested_storage() {
    let objective = ScaledQuadratic { dimension: 2 };
    let bounds = Bounds::new(vec![(Some(-5.0), Some(5.0)); 2]).unwrap();
    let problem = scioxide_optimize::BoundedObjective::new(&objective, &bounds).unwrap();
    let failing = scioxide_optimize::BoundedObjective::new(&FailingObjective, &bounds).unwrap();
    let differential = scioxide_optimize::DifferentialEvolution::new(
        scioxide_optimize::DifferentialEvolutionOptions::default()
            .with_max_generations(scioxide_core::Budget::try_from(3).unwrap()),
    )
    .unwrap();
    macro_rules! check {
        ($solver:expr, $name:literal) => {{
            let solver = $solver;
            let mut workspace = solver.workspace(2).unwrap();
            let mut output = OptimizeResult::empty();
            output.try_reserve(2).unwrap();
            let sequential = scioxide_core::ExecutionContext::sequential;
            solver
                .minimize_into(
                    &problem,
                    &[1.0, 2.0],
                    &mut output,
                    &mut workspace,
                    sequential(),
                )
                .unwrap();
            solver
                .minimize_into(
                    &problem,
                    &[1.0, 2.0],
                    &mut output,
                    &mut workspace,
                    sequential(),
                )
                .unwrap();
            let allocating = solver.minimize(&problem, &[1.0, 2.0]).unwrap();
            assert_optimize_parity(
                $name,
                &allocating,
                &output,
                attempt_work(workspace.last_attempt().unwrap()),
            );
            assert_eq!(allocating.counts(), output.counts());
            let saved = output.clone();
            assert!(
                solver
                    .minimize_into(
                        &failing,
                        &[1.0, 2.0],
                        &mut output,
                        &mut workspace,
                        sequential()
                    )
                    .is_err()
            );
            assert_optimize_result_unchanged($name, &saved, &output);
            assert_eq!(attempt_work(workspace.last_attempt().unwrap()).0, 1);
        }};
    }
    check!(differential, "differential evolution prepared");
}

#[test]
fn qp_verification_first_convexity_check_is_zero_allocation() {
    use scioxide_core::ExecutionContext;
    use scioxide_optimize::{
        DualCertificate, OptimalityTolerances, ProgrammingCertificate, ProgrammingVerifier,
        QpProblem, QpVerificationProblem, VerificationOptions, VerificationReport,
    };
    let t = Tolerance::new(1e-12, 1e-12).unwrap();
    let verifier = ProgrammingVerifier::new(VerificationOptions::default().with_tolerances(
        OptimalityTolerances::new(
            scioxide_core::FeasibilityTolerance::from_tolerance(t),
            scioxide_core::GradientTolerance::from_tolerance(t),
            scioxide_core::ResidualTolerance::from_tolerance(t),
            scioxide_core::RankThreshold::from_tolerance(t),
        ),
    ));
    for (hessian, linear, point) in [
        (vec![2., 0., 0., 2.], vec![-2., 0.], [1., 0.]),
        (vec![1., 1., 1., 1.], vec![-2., -2.], [1., 1.]),
    ] {
        let problem = QpProblem::new(Matrix::from_vec(2, 2, hessian).unwrap(), linear).unwrap();
        let view = QpVerificationProblem::from_problem(&problem);
        let mut workspace = verifier.workspace(&view).unwrap();
        let mut report = VerificationReport::empty();
        let certificate = ProgrammingCertificate::Optimality {
            point: &point,
            dual: DualCertificate {
                rows: &[],
                lower: &[0., 0.],
                upper: &[0., 0.],
            },
        };
        // The prepared workspace already covers the first convexity check.
        verifier
            .verify_into(
                &view,
                certificate,
                &mut report,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.is_accepted());
    }
}

#[test]
fn derivative_free_models_and_subproblems_reuse_prepared_storage() {
    use scioxide_optimize::{Cobyla, Cobyqa};
    let objective = ObjectiveFn::new(2, |x: &[f64]| {
        Ok((x[0] + x[1] - 1.0).powi(2) + 2.0 * (x[0] - x[1]).powi(2))
    });
    let unconstrained = Constraints::new(2).unwrap();
    let problem = ConstrainedProblem::new(&objective, &unconstrained).unwrap();
    let sequential = scioxide_core::ExecutionContext::sequential;
    macro_rules! check {
        ($solver:expr,$name:literal) => {{
            let solver = $solver;
            let mut workspace = solver.workspace(&problem).unwrap();
            let mut out = OptimizeResult::empty();
            out.try_reserve(2).unwrap();
            solver
                .minimize_into(
                    &problem,
                    &[0.0, 0.0],
                    &mut out,
                    &mut workspace,
                    sequential(),
                )
                .unwrap();
            solver
                .minimize_into(
                    &problem,
                    &[0.0, 0.0],
                    &mut out,
                    &mut workspace,
                    sequential(),
                )
                .unwrap();
            assert!(out.fun() < 1e-6, "{out:?}");
            assert_eq!(out.stationarity(), None);
        }};
    }
    check!(Cobyla::<f64>::default(), "COBYLA prepared");
    check!(Cobyqa::<f64>::default(), "COBYQA prepared");
}

#[test]
fn lbfgsb_prepared_projected_history_executes() {
    use scioxide_optimize::{BoundedObjective, LbfgsB};
    for dimension in [20, 100] {
        let objective = ScaledQuadratic { dimension };
        let bounds = Bounds::new(vec![(Some(-2.0), Some(2.0)); dimension]).unwrap();
        let problem = BoundedObjective::new(&objective, &bounds).unwrap();
        let solver = LbfgsB::<f64>::default();
        let mut workspace = solver.workspace(dimension).unwrap();
        let mut result = OptimizeResult::empty();
        result.try_reserve(dimension).unwrap();
        let start = vec![1.0; dimension];
        let sequential = scioxide_core::ExecutionContext::sequential;
        solver
            .minimize_into(&problem, &start, &mut result, &mut workspace, sequential())
            .unwrap();
        assert!(result.is_success(), "{result:?}");
        assert!(result.iterations() > 2);
        solver
            .minimize_into(&problem, &start, &mut result, &mut workspace, sequential())
            .unwrap();
        assert!(result.is_success());
    }
}

fn sqp_of<S: scioxide_core::RealScalar>(
    result: &scioxide_optimize::OptimizeResult<S>,
) -> scioxide_optimize::SqpDiagnostics {
    match result.method_diagnostics() {
        scioxide_optimize::OptimizeDiagnostics::Sqp(diagnostics, _) => diagnostics,
        other => panic!("unexpected method diagnostics {other:?}"),
    }
}
