//! Integration tests for scioxide-optimize.
//!
//! These tests exercise end-to-end workflows spanning multiple modules.

#![allow(clippy::unwrap_used, clippy::suboptimal_flops)]

use std::cell::Cell;
use std::time::Duration;

use scioxide_core::{Bounds, Budget, SciOxideError, Tolerance};
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    Bfgs, BfgsOptions, BoundedObjective, ConstrainedProblem, ConstraintFunction, Constraints,
    DifferentialEvolution, DifferentialEvolutionOptions, FiniteBounds, GlobalSearchOptions,
    GradientDescentOptions, LbfgsOptions, LeastSquares, LeastSquaresProblem, LineSearch,
    LineSearchFailure, LineSearchResult, LinearConstraintSense, LinearConstraints, LpProblem,
    MipProblem, NelderMead, NelderMeadOptions, NewtonOptions, NonlinearCgOptions,
    ObjectiveFunction, OptimizeTermination, Optimizer, QpProblem, StrongWolfeLineSearch,
};
use scioxide_sparse::{DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};

use scioxide_optimize::{GradientDescent, Lbfgs, NewtonOptimizer, NonlinearCg};

// ---------------------------------------------------------------------------
// Test problems
// ---------------------------------------------------------------------------

/// Rosenbrock function: f(x) = (1-x0)^2 + 100*(x1-x0^2)^2
struct Rosenbrock;

impl ObjectiveFunction<f64> for Rosenbrock {
    fn dim(&self) -> usize {
        2
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        Ok(a * a + 100.0 * b * b)
    }

    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        let dx0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        let dx1 = 200.0 * (x[1] - x[0] * x[0]);
        output.copy_from_slice(&[dx0, dx1]);
        Ok(true)
    }
}

/// Sphere function: `f(x) = sum(x_i^2)`
struct Sphere;

impl ObjectiveFunction<f64> for Sphere {
    fn dim(&self) -> usize {
        3
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x.iter().map(|xi| xi * xi).sum())
    }

    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output
            .iter_mut()
            .zip(x)
            .for_each(|(gradient, &value)| *gradient = 2.0 * value);
        Ok(true)
    }
}

/// Linear inequality constraint: x0 + x1 - 1 <= 0.
struct SumLeqOne;

impl ConstraintFunction<f64> for SumLeqOne {
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

/// Nonlinear least-squares: r(x) = [x0^2 - 1, x1^2 - 4].
struct NonlinearLS;

impl LeastSquaresProblem<f64> for NonlinearLS {
    fn param_dim(&self) -> usize {
        2
    }
    fn residual_dim(&self) -> usize {
        2
    }
    fn residual_into(&self, x: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
        output.copy_from_slice(&[x[0] * x[0] - 1.0, x[1] * x[1] - 4.0]);
        Ok(())
    }
    fn jacobian_into(&self, x: &[f64], output: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
        output
            .as_mut_slice()
            .copy_from_slice(&[2.0 * x[0], 0.0, 0.0, 2.0 * x[1]]);
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// Unconstrained optimization
// ---------------------------------------------------------------------------

#[test]
fn test_bfgs_rosenbrock() {
    let solver = Bfgs::<f64>::default();
    let result = solver.minimize(&Rosenbrock, &[-1.0, 1.0]).unwrap();

    assert!(result.is_success());
    assert!((result.x()[0] - 1.0).abs() < 1e-4);
    assert!((result.x()[1] - 1.0).abs() < 1e-4);
    assert!(result.fun() < 1e-8);
}

struct LineSearchFailureProblem;

impl ObjectiveFunction<f64> for LineSearchFailureProblem {
    fn dim(&self) -> usize {
        1
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] * x[0])
    }

    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output[0] = 2.0 * x[0];
        Ok(true)
    }

    fn hessian_into(&self, _x: &[f64], output: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
        output.as_mut_slice()[0] = 2.0;
        Ok(true)
    }
}

#[derive(Debug)]

struct AlwaysFailingLineSearch;

impl LineSearch<f64> for AlwaysFailingLineSearch {
    fn search_into<F: ObjectiveFunction<f64> + ?Sized>(
        &self,
        _f: &F,
        _x: &[f64],
        _f_x: f64,
        _direction: &[f64],
        _grad_dot_dir: f64,
        _: scioxide_differentiate::DerivativeStrategy,
        _: &mut [f64],
        _: &mut [f64],
        _: &mut [f64],
        _: &mut [f64],
        _: &mut scioxide_optimize::DerivativeWorkspace<f64>,
        _: scioxide_optimize::LineSearchScope<'_>,
        _: Option<f64>,
    ) -> Result<scioxide_optimize::LineSearchResult<f64>, SciOxideError> {
        Err(SciOxideError::convergence_failure(
            scioxide_core::ConvergenceReason::IterationLimit,
            scioxide_core::ConvergenceDiagnostics::new()
                .with_counts((scioxide_core::EvaluationCounts::from_iterations(7)).into()),
            "injected line-search failure",
        ))
    }
}

struct CountingObjective {
    dim: usize,
    evaluations: Cell<usize>,
}

impl CountingObjective {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            evaluations: Cell::new(0),
        }
    }
}

impl ObjectiveFunction<f64> for CountingObjective {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        self.evaluations.set(self.evaluations.get() + 1);
        Ok(x.iter().map(|value| value * value).sum())
    }

    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output
            .iter_mut()
            .zip(x)
            .for_each(|(gradient, &value)| *gradient = 2.0 * value);
        Ok(true)
    }
}

#[test]
fn gradient_solvers_propagate_line_search_failure() {
    let bfgs = Bfgs::new(BfgsOptions::default().with_line_search(AlwaysFailingLineSearch));
    let bfgs_error = bfgs
        .minimize(&LineSearchFailureProblem, &[1.0])
        .unwrap_err();
    assert!(matches!(
        bfgs_error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure { .. }
    ));
    let solvers: Vec<Box<dyn Optimizer<f64>>> = vec![
        Box::new(Lbfgs::new(
            LbfgsOptions::default().with_line_search(AlwaysFailingLineSearch),
        )),
        Box::new(GradientDescent::new(
            GradientDescentOptions::default().with_line_search(AlwaysFailingLineSearch),
        )),
        Box::new(NonlinearCg::new(
            NonlinearCgOptions::default().with_line_search(AlwaysFailingLineSearch),
        )),
        Box::new(NewtonOptimizer::new(
            NewtonOptions::default().with_line_search(AlwaysFailingLineSearch),
        )),
    ];

    for solver in solvers {
        let error = solver
            .minimize(&LineSearchFailureProblem, &[1.0])
            .unwrap_err();

        assert!(
            matches ! (& (error) , error if if let scioxide_core :: SciOxideErrorView :: ConvergenceFailure { diagnostics , detail , .. } = error . view () { diagnostics . counts ().own().iterations() == 7 && detail == "injected line-search failure" } else { false })
        );
    }
}

#[derive(Debug)]

struct UnsuccessfulLineSearch;

impl LineSearch<f64> for UnsuccessfulLineSearch {
    fn search_into<F: ObjectiveFunction<f64> + ?Sized>(
        &self,
        _f: &F,
        _x: &[f64],
        f_x: f64,
        _direction: &[f64],
        _grad_dot_dir: f64,
        _: scioxide_differentiate::DerivativeStrategy,
        _: &mut [f64],
        _: &mut [f64],
        _: &mut [f64],
        _: &mut [f64],
        _: &mut scioxide_optimize::DerivativeWorkspace<f64>,
        _: scioxide_optimize::LineSearchScope<'_>,
        _: Option<f64>,
    ) -> Result<scioxide_optimize::LineSearchResult<f64>, SciOxideError> {
        scioxide_optimize::LineSearchResult::try_unsuccessful(
            LineSearchFailure::StepSizeLimit,
            Some((0.5, f_x / 2.0)),
            2,
            0,
        )
    }
}

#[test]
fn gradient_solvers_do_not_accept_diagnostic_line_search_trials() {
    let bfgs = Bfgs::new(BfgsOptions::default().with_line_search(UnsuccessfulLineSearch));
    let bfgs_result = bfgs.minimize(&LineSearchFailureProblem, &[1.0]).unwrap();
    assert_eq!(
        bfgs_result.termination(),
        OptimizeTermination::LineSearchFailed(LineSearchFailure::StepSizeLimit)
    );
    assert_eq!(bfgs_result.x(), &[1.0]);
    let solvers: Vec<Box<dyn Optimizer<f64>>> = vec![
        Box::new(Lbfgs::new(
            LbfgsOptions::default().with_line_search(UnsuccessfulLineSearch),
        )),
        Box::new(GradientDescent::new(
            GradientDescentOptions::default().with_line_search(UnsuccessfulLineSearch),
        )),
        Box::new(NonlinearCg::new(
            NonlinearCgOptions::default().with_line_search(UnsuccessfulLineSearch),
        )),
        Box::new(NewtonOptimizer::new(
            NewtonOptions::default().with_line_search(UnsuccessfulLineSearch),
        )),
    ];

    for solver in solvers {
        let result = solver.minimize(&LineSearchFailureProblem, &[1.0]).unwrap();
        assert_eq!(
            result.termination(),
            OptimizeTermination::LineSearchFailed(LineSearchFailure::StepSizeLimit)
        );
        assert!(!result.is_success());
        assert_eq!(result.x(), &[1.0]);
        assert_eq!(result.fun(), 1.0);
    }
}

struct EvaluationCountingQuadratic {
    analytic_gradient: bool,
    fun_calls: Cell<usize>,
    grad_calls: Cell<usize>,
}

impl EvaluationCountingQuadratic {
    const fn new(analytic_gradient: bool) -> Self {
        Self {
            analytic_gradient,
            fun_calls: Cell::new(0),
            grad_calls: Cell::new(0),
        }
    }
}

impl ObjectiveFunction<f64> for EvaluationCountingQuadratic {
    fn dim(&self) -> usize {
        1
    }

    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        self.fun_calls.set(self.fun_calls.get() + 1);
        Ok(x[0] * x[0])
    }

    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        self.grad_calls.set(self.grad_calls.get() + 1);
        if !self.analytic_gradient {
            return Ok(false);
        }
        output[0] = 2.0 * x[0];
        Ok(true)
    }

    fn hessian_into(&self, _x: &[f64], output: &mut Matrix<f64>) -> Result<bool, SciOxideError> {
        output.as_mut_slice()[0] = 2.0;
        Ok(true)
    }
}

fn line_search_solvers() -> Vec<(&'static str, Box<dyn Optimizer<f64>>)> {
    vec![
        ("BFGS", Box::new(Bfgs::<f64>::default())),
        ("L-BFGS", Box::new(Lbfgs::<f64>::default())),
        (
            "GradientDescent",
            Box::new(GradientDescent::<f64>::default()),
        ),
        ("NonlinearCg", Box::new(NonlinearCg::<f64>::default())),
        ("Newton", Box::new(NewtonOptimizer::<f64>::default())),
    ]
}

#[test]
fn gradient_solvers_report_exact_analytic_evaluation_counts() {
    for (name, solver) in line_search_solvers() {
        let objective = EvaluationCountingQuadratic::new(true);
        let result = solver.minimize(&objective, &[1.0]).unwrap();
        let expected_fun_evals = if name == "Newton" { 2 } else { 3 };

        assert!(result.is_success(), "{} did not converge", name);
        assert_eq!(result.fun_evals(), expected_fun_evals, "{}", name);
        let expected_grad_evals = if matches!(name, "BFGS" | "L-BFGS" | "NonlinearCg") {
            3
        } else {
            2
        };
        assert_eq!(result.grad_evals(), expected_grad_evals, "{}", name);
        assert_eq!(objective.fun_calls.get(), result.fun_evals(), "{}", name);
        assert_eq!(objective.grad_calls.get(), result.grad_evals(), "{}", name);
    }
}

#[test]
fn gradient_solvers_report_exact_finite_difference_counts() {
    for (name, solver) in line_search_solvers() {
        let objective = EvaluationCountingQuadratic::new(false);
        let result = solver.minimize(&objective, &[1.0]).unwrap();
        let expected_fun_evals = match name {
            "Newton" => 6,
            "GradientDescent" => 7,
            _ => 9,
        };

        assert!(result.is_success(), "{} did not converge", name);
        assert_eq!(result.fun_evals(), expected_fun_evals, "{}", name);
        let expected_grad_evals = if matches!(name, "BFGS" | "L-BFGS" | "NonlinearCg") {
            3
        } else {
            2
        };
        assert_eq!(result.grad_evals(), expected_grad_evals, "{}", name);
        assert_eq!(objective.fun_calls.get(), result.fun_evals(), "{}", name);
        assert_eq!(objective.grad_calls.get(), result.grad_evals(), "{}", name);
    }
}

#[test]
fn bfgs_reuses_strong_wolfe_analytic_and_finite_difference_gradients() {
    for (analytic_gradient, expected_fun_evals) in [(true, 3), (false, 9)] {
        let objective = EvaluationCountingQuadratic::new(analytic_gradient);
        let solver =
            Bfgs::new(BfgsOptions::default().with_line_search(StrongWolfeLineSearch::default()));
        let result = solver.minimize(&objective, &[1.0]).unwrap();

        assert!(result.is_success());
        assert_eq!(result.fun_evals(), expected_fun_evals);
        assert_eq!(result.grad_evals(), 3);
        assert_eq!(objective.fun_calls.get(), result.fun_evals());
        assert_eq!(objective.grad_calls.get(), result.grad_evals());
    }
}

#[test]
fn nested_solvers_preserve_explicit_inner_configuration() {
    let failing_inner =
        || Bfgs::new(BfgsOptions::default().with_line_search(AlwaysFailingLineSearch));

    let penalty = scioxide_optimize::PenaltyMethod::default().with_inner_optimizer(failing_inner());
    let unconstrained = Constraints::new(1).unwrap();
    let problem = ConstrainedProblem::new(&LineSearchFailureProblem, &unconstrained).unwrap();
    let error = penalty.minimize(&problem, &[1.0]).unwrap_err();
    assert!(
        matches ! (& (error) , error if if let scioxide_core :: SciOxideErrorView :: ConvergenceFailure { detail , .. } = error . view () { detail == "injected line-search failure" } else { false })
    );
}

#[test]
fn structural_configuration_errors_precede_objective_evaluation() {
    let objective = CountingObjective::new(2);
    let short_bounds = Bounds::new(vec![(Some(-1.0), Some(1.0))]).unwrap();
    assert!(matches!(
        (Constraints::<f64>::new(2).unwrap().bounds(short_bounds))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
    ));
    assert_eq!(objective.evaluations.get(), 0);

    let objective = CountingObjective::new(2);
    let bounds = Bounds::new(vec![(Some(-1.0), Some(1.0)); 2]).unwrap();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let differential_evolution =
        DifferentialEvolution::new(DifferentialEvolutionOptions::default()).unwrap();
    assert!(matches!(
        (differential_evolution.minimize(&problem, &[2.0, 0.0]))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
    ));
    assert_eq!(objective.evaluations.get(), 0);

    let objective = CountingObjective::new(2);
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let multistart = scioxide_optimize::MultiStart::new(
        GlobalSearchOptions::default().with_starts(Budget::ONE),
        scioxide_optimize::BorrowedStartDesign::new(&[2.0, 0.0], 2).unwrap(),
        scioxide_optimize::PowellLocalFactory::default(),
    )
    .unwrap();
    assert!(matches!(
        (multistart.minimize(&problem))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
    ));
    assert_eq!(objective.evaluations.get(), 0);
}

#[test]
fn nonlinear_solvers_accept_equalities_and_bounds() {
    let equalities = Constraints::new(2).unwrap().equality(SumLeqOne).unwrap();
    let problem = ConstrainedProblem::new(&SimpleQuadratic2D, &equalities).unwrap();
    assert!(
        scioxide_optimize::InteriorPoint::<f64>::default()
            .workspace(&problem)
            .is_ok()
    );

    let bounds = Bounds::new(vec![(Some(-1.0), Some(1.0)); 2]).unwrap();
    let bounds = Constraints::new(2).unwrap().bounds(bounds).unwrap();
    let problem = ConstrainedProblem::new(&SimpleQuadratic2D, &bounds).unwrap();
    assert!(
        scioxide_optimize::Sqp::<f64>::default()
            .workspace(&problem)
            .is_ok()
    );
}

#[test]
fn test_nelder_mead_sphere() {
    let solver = NelderMead::new(
        NelderMeadOptions::default()
            .with_function_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
                Tolerance::from_decimal_places(6, 6),
            ))
            .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(
                Tolerance::from_decimal_places(6, 6),
            ))
            .with_max_iterations(Budget::try_from(1000).unwrap()),
    )
    .unwrap();
    let result = solver.minimize(&Sphere, &[3.0, -2.0, 1.0]).unwrap();

    assert!(result.is_success());
    for xi in result.x() {
        assert!(xi.abs() < 1e-2);
    }
    assert!(result.fun() < 1e-4);
}

// ---------------------------------------------------------------------------
// Constrained optimization
// ---------------------------------------------------------------------------

/// Minimize x0^2 + x1^2 subject to x0 + x1 >= 1.
/// Equivalent to: min x0^2 + x1^2 s.t. -(x0+x1-1) <= 0 => x0+x1-1 >= 0.
/// The constraint x0+x1 <= 1 with min x0^2+x1^2 gives optimum at (0.5, 0.5).
struct SimpleQuadratic2D;

impl ObjectiveFunction<f64> for SimpleQuadratic2D {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] * x[0] + x[1] * x[1])
    }
    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]);
        Ok(true)
    }
}

#[test]
fn test_penalty_constrained_quadratic() {
    let constraints = Constraints::new(2).unwrap().inequality(SumLeqOne).unwrap();
    let problem = ConstrainedProblem::new(&SimpleQuadratic2D, &constraints).unwrap();
    let solver = scioxide_optimize::PenaltyMethod::<f64>::default();

    let result = solver.minimize(&problem, &[2.0, 2.0]).unwrap();

    // An infeasible nonstationary start must move to the feasible minimizer.
    assert!(result.is_success());
    assert!(result.fun() < 1e-8);
    assert!(result.x().iter().all(|x| x.abs() < 1e-4));
    assert!(result.x().iter().sum::<f64>() <= 1.0 + 1e-8);
}

// ---------------------------------------------------------------------------
// Least squares
// ---------------------------------------------------------------------------

#[test]
fn test_gauss_newton_nonlinear_ls() {
    let solver = LeastSquares::<f64>::new(
        scioxide_optimize::LeastSquaresOptions::default()
            .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(
                Tolerance::from_decimal_places(8, 8),
            ))
            .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(
                Tolerance::from_decimal_places(8, 8),
            ))
            .with_globalization(scioxide_optimize::LeastSquaresGlobalization::gauss_newton()),
    )
    .unwrap();
    let result = solver.solve(&NonlinearLS, &[0.5, 1.5]).unwrap();

    assert!(result.is_success());
    assert!((result.x()[0] - 1.0).abs() < 1e-6);
    assert!((result.x()[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_levenberg_marquardt_nonlinear_ls() {
    let solver = LeastSquares::<f64>::new(
        scioxide_optimize::LeastSquaresOptions::default()
            .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(
                Tolerance::from_decimal_places(8, 8),
            ))
            .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(
                Tolerance::from_decimal_places(8, 8),
            ))
            .with_globalization(
                scioxide_optimize::LeastSquaresGlobalization::levenberg_marquardt(),
            ),
    )
    .unwrap();
    let result = solver.solve(&NonlinearLS, &[0.5, 1.5]).unwrap();

    assert!(result.is_success());
    assert!((result.x()[0] - 1.0).abs() < 1e-6);
    assert!((result.x()[1] - 2.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Linear programming
// ---------------------------------------------------------------------------

fn constrained_lp(objective: Vec<f64>, coefficients: Vec<f64>, rhs: Vec<f64>) -> LpProblem<f64> {
    let variables = objective.len();
    let rows = rhs.len();
    let constraints = LinearConstraints::new(
        Matrix::from_vec(rows, variables, coefficients).unwrap(),
        rhs,
        vec![LinearConstraintSense::LessEqual; rows],
    )
    .unwrap();
    LpProblem::new(objective)
        .unwrap()
        .with_constraints(constraints)
        .unwrap()
}

#[test]
fn test_simplex_lp() {
    // Minimize -x0 - 2*x1 subject to x0 + x1 <= 4, x0 <= 3, x1 <= 3
    let lp = constrained_lp(
        vec![-1.0, -2.0],
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        vec![4.0, 3.0, 3.0],
    );

    let solver = scioxide_optimize::SimplexSolver::<f64>::default();
    let result = solver.solve(&lp).unwrap();

    assert!(result.is_success());
    // Optimum at (1, 3) with value -7
    assert!((result.fun() - (-7.0)).abs() < 1e-6);
}

#[test]
fn test_interior_point_lp() {
    let lp = constrained_lp(
        vec![-1.0, -2.0],
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        vec![4.0, 3.0, 3.0],
    );

    let solver = scioxide_optimize::InteriorPointLp::<f64>::default();
    let result = solver.solve(&lp).unwrap();

    assert!(result.is_success());
    assert!((result.fun() - (-7.0)).abs() < 0.1);
}

// ---------------------------------------------------------------------------
// Quadratic programming
// ---------------------------------------------------------------------------

#[test]
fn test_qp_solver() {
    // Minimize 0.5 * x^T [[2,0],[0,2]] x + [-2,-6]^T x
    // Subject to: x0 + x1 <= 2, x0 >= 0, x1 >= 0
    // Unconstrained min at (1,3), constrained optimum at (0,2)
    let constraints = LinearConstraints::new(
        Matrix::from_vec(3, 2, vec![1.0, 1.0, -1.0, 0.0, 0.0, -1.0]).unwrap(),
        vec![2.0, 0.0, 0.0],
        vec![LinearConstraintSense::LessEqual; 3],
    )
    .unwrap();
    let qp = QpProblem::new(
        Matrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 2.0]).unwrap(),
        vec![-2.0, -6.0],
    )
    .unwrap()
    .with_constraints(constraints)
    .unwrap();

    let solver = scioxide_optimize::QpSolver::<f64>::default();
    let result = solver.solve(&qp).unwrap();

    assert!(result.is_success());
    assert!(result.x()[0].abs() < 1e-4);
    assert!((result.x()[1] - 2.0).abs() < 1e-4);
}

// ---------------------------------------------------------------------------
// Mixed-integer programming
// ---------------------------------------------------------------------------

#[test]
fn test_branch_and_bound_mip() {
    // Minimize -x0 - x1, x0 + x1 <= 3.5, x0 <= 2.5, x1 <= 2.5
    // Integer: both variables
    let relaxation = constrained_lp(
        vec![-1.0, -1.0],
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        vec![3.5, 2.5, 2.5],
    );
    let mip = MipProblem::new(relaxation, vec![0, 1]).unwrap();

    let solver = scioxide_optimize::BranchAndBound::<f64>::default();
    let result = solver.solve(&mip).unwrap();

    assert!(result.is_success());
    // Integer-optimal: x0=1, x1=2 or x0=2, x1=1 — both give fun = -3
    assert!((result.fun() - (-3.0)).abs() < 1e-6);
}

#[test]
fn test_sparse_lp_input_is_solved_without_changing_the_public_problem() {
    let mut coefficients = SparseBuilder::new(3, 2);
    coefficients.push(0, 0, 1.0).unwrap();
    coefficients.push(0, 1, 1.0).unwrap();
    coefficients.push(1, 0, 1.0).unwrap();
    coefficients.push(2, 1, 1.0).unwrap();
    let constraints = LinearConstraints::new(
        coefficients
            .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
            .unwrap(),
        vec![4.0, 3.0, 3.0],
        vec![LinearConstraintSense::LessEqual; 3],
    )
    .unwrap();
    let problem = LpProblem::new(vec![-1.0_f64, -2.0])
        .unwrap()
        .with_constraints(constraints)
        .unwrap();

    let result = scioxide_optimize::SimplexSolver::default()
        .solve(&problem)
        .unwrap();

    assert!(result.is_success());
    assert!((result.fun() + 7.0).abs() < 1e-6);
    assert!(
        problem
            .constraints()
            .is_some_and(|value| value.coefficients().is_sparse())
    );
}

#[test]
fn test_explicit_unbounded_variable_and_greater_equal_constraint() {
    let constraints = LinearConstraints::new(
        Matrix::from_vec(1, 1, vec![1.0]).unwrap(),
        vec![-2.0],
        vec![LinearConstraintSense::GreaterEqual],
    )
    .unwrap();
    let problem = LpProblem::new(vec![1.0_f64])
        .unwrap()
        .with_constraints(constraints)
        .unwrap()
        .with_bounds(Bounds::new(vec![(None, None)]).unwrap())
        .unwrap();

    let result = scioxide_optimize::SimplexSolver::default()
        .solve(&problem)
        .unwrap();

    assert!(result.is_success());
    assert!((result.x()[0] + 2.0).abs() < 1e-6);
    assert!((result.fun() + 2.0).abs() < 1e-6);
}

#[test]
fn test_sparse_qp_hessian_is_solved() {
    let mut hessian = SparseBuilder::new(2, 2);
    hessian.push(0, 0, 2.0).unwrap();
    hessian.push(1, 1, 2.0).unwrap();
    let problem = QpProblem::new(
        hessian
            .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
            .unwrap(),
        vec![-2.0_f64, -6.0],
    )
    .unwrap();

    let result = scioxide_optimize::QpSolver::default()
        .solve(&problem)
        .unwrap();

    assert!(result.is_success());
    assert!((result.x()[0] - 1.0).abs() < 1e-10);
    assert!((result.x()[1] - 3.0).abs() < 1e-10);
}

#[test]
fn test_program_structure_errors_are_distinct_from_infeasible_domain_results() {
    let malformed = LinearConstraints::new(
        Matrix::from_vec(1, 1, vec![1.0_f64]).unwrap(),
        Vec::new(),
        vec![LinearConstraintSense::LessEqual],
    );
    assert!(matches!(
        (malformed)
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
    ));

    let constraints = LinearConstraints::new(
        Matrix::from_vec(2, 1, vec![1.0, -1.0]).unwrap(),
        vec![0.0, -1.0],
        vec![LinearConstraintSense::LessEqual; 2],
    )
    .unwrap();
    let valid_but_infeasible = LpProblem::new(vec![1.0])
        .unwrap()
        .with_constraints(constraints)
        .unwrap();
    let result = scioxide_optimize::SimplexSolver::default()
        .solve(&valid_but_infeasible)
        .unwrap();

    assert_eq!(result.termination(), OptimizeTermination::Infeasible);
}

// ---------------------------------------------------------------------------
// Global optimization
// ---------------------------------------------------------------------------

#[test]
fn test_de_sphere() {
    let bounds = Bounds::new(vec![(Some(-5.0), Some(5.0)); 3]).unwrap();
    let solver = DifferentialEvolution::new(DifferentialEvolutionOptions::default()).unwrap();

    let result = solver
        .minimize(
            &BoundedObjective::new(&Sphere, &bounds).unwrap(),
            &[3.0, -2.0, 1.0],
        )
        .unwrap();

    assert!(!result.is_success());
    assert!(matches!(
        result.termination(),
        OptimizeTermination::PopulationStagnation | OptimizeTermination::SearchBudgetCompleted
    ));
    for xi in result.x() {
        assert!(xi.abs() < 0.1);
    }
    assert!(result.fun() < 0.01);
}

// ---------------------------------------------------------------------------
// Re-export accessibility
// ---------------------------------------------------------------------------

#[test]
fn test_reexports_accessible() {
    // Verify that all re-exported types are accessible from the crate root.
    assert!(OptimizeTermination::Converged.is_success());
    assert!(!OptimizeTermination::IterationLimit.is_success());
    assert!(!OptimizeTermination::SearchBudgetCompleted.is_success());
    let _ = Bfgs::<f64>::default();
    let _ = NelderMead::<f64>::default();
    let _ = scioxide_optimize::GradientDescent::<f64>::default();
    let _ = scioxide_optimize::Lbfgs::<f64>::default();
    let _ = scioxide_optimize::NonlinearCg::<f64>::default();
    let _ = scioxide_optimize::NewtonOptimizer::<f64>::default();
    let _ = scioxide_optimize::TrustRegionOptimizer::<f64>::default();
    let _ = scioxide_optimize::PenaltyMethod::<f64>::default();
    let _ = scioxide_optimize::AugmentedLagrangian::<f64>::default();
    let _ = scioxide_optimize::Sqp::<f64>::default();
    let _ = scioxide_optimize::InteriorPoint::<f64>::default();
    let _ = scioxide_optimize::BacktrackingLineSearch::<f64>::default();
    let _ = scioxide_optimize::StrongWolfeLineSearch::<f64>::default();
    let _ = LeastSquares::<f64>::default();
    let _ = scioxide_optimize::SimplexSolver::<f64>::default();
    let _ = scioxide_optimize::InteriorPointLp::<f64>::default();
    let _ = scioxide_optimize::QpSolver::<f64>::default();
    let _ = scioxide_optimize::BranchAndBound::<f64>::default();
    let _ = scioxide_optimize::MultiStart::new(
        GlobalSearchOptions::<f64>::default(),
        scioxide_optimize::UniformStartDesign,
        scioxide_optimize::PowellLocalFactory::default(),
    )
    .unwrap();
    let _ = DifferentialEvolution::new(DifferentialEvolutionOptions::<f64>::default()).unwrap();
    let _ = scioxide_optimize::DeStrategy::Rand1;
}

#[test]
fn public_configuration_builders_round_trip_through_getters() {
    let tolerance = Tolerance::new(1e-7, 1e-8).unwrap();
    let iterations = Budget::try_from(37).unwrap();
    let time_limit = Duration::from_millis(250);
    let strategy = scioxide_differentiate::DerivativeStrategy::Forward;
    let line_search = scioxide_optimize::BacktrackingLineSearch::new(
        0.75,
        1e-12,
        1e-4,
        0.4,
        Budget::try_from(19).unwrap(),
    )
    .unwrap();
    let options = BfgsOptions::default()
        .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(tolerance))
        .with_max_iterations(iterations)
        .with_derivative_strategy(strategy)
        .with_time_limit(time_limit)
        .with_line_search(line_search);
    assert_eq!(options.gradient_tolerance().tolerance(), tolerance);
    assert_eq!(options.max_iterations(), iterations);
    assert_eq!(options.derivative_strategy(), strategy);
    assert_eq!(options.time_limit(), Some(time_limit));
    assert_eq!(options.line_search().initial_step(), 0.75);
    assert_eq!(options.line_search().minimum_step(), 1e-12);
    assert_eq!(options.line_search().sufficient_decrease(), 1e-4);
    assert_eq!(options.line_search().contraction_factor(), 0.4);
    assert_eq!(
        options.line_search().max_iterations(),
        Budget::try_from(19).unwrap()
    );

    let lbfgs = LbfgsOptions::<f64>::default()
        .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(tolerance))
        .with_max_iterations(iterations)
        .with_memory(Budget::try_from(7).unwrap())
        .with_derivative_strategy(strategy)
        .with_time_limit(time_limit);
    assert_eq!(lbfgs.gradient_tolerance().tolerance(), tolerance);
    assert_eq!(lbfgs.max_iterations(), iterations);
    assert_eq!(lbfgs.memory(), Budget::try_from(7).unwrap());
    assert_eq!(lbfgs.derivative_strategy(), strategy);
    assert_eq!(lbfgs.time_limit(), Some(time_limit));

    let trust_region = scioxide_optimize::TrustRegionOptions::<f64>::default()
        .with_gradient_tolerance(scioxide_core::GradientTolerance::from_tolerance(tolerance))
        .with_max_iterations(iterations)
        .with_derivative_strategy(strategy)
        .with_time_limit(time_limit);
    assert_eq!(trust_region.gradient_tolerance().tolerance(), tolerance);
    assert_eq!(trust_region.max_iterations(), iterations);
    assert_eq!(trust_region.derivative_strategy(), strategy);
    assert_eq!(trust_region.time_limit(), Some(time_limit));

    let direct = NelderMeadOptions::<f64>::default()
        .with_function_tolerance(scioxide_core::ResidualTolerance::from_tolerance(tolerance))
        .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(tolerance))
        .with_max_iterations(iterations)
        .with_time_limit(time_limit);
    assert_eq!(direct.function_tolerance().tolerance(), tolerance);
    assert_eq!(direct.max_iterations(), iterations);
    assert_eq!(direct.time_limit(), Some(time_limit));

    let bounds = FiniteBounds::new(vec![(-2.0, 3.0), (-1.0, 4.0)]).unwrap();
    let differential = DifferentialEvolutionOptions::default()
        .with_fitness_tolerance(scioxide_core::ResidualTolerance::from_tolerance(tolerance))
        .with_max_generations(iterations)
        .with_population_size(Budget::try_from(8).unwrap())
        .with_mutation_factor(0.6)
        .with_crossover_rate(0.7)
        .with_strategy(scioxide_optimize::DeStrategy::Best1)
        .with_seed(7)
        .with_time_limit(time_limit);
    assert!(DifferentialEvolution::new(differential.clone()).is_ok());
    assert_eq!(differential.fitness_tolerance().tolerance(), tolerance);
    assert_eq!(differential.max_generations(), iterations);
    assert_eq!(
        differential.population_size(),
        Some(Budget::try_from(8).unwrap())
    );
    assert_eq!(differential.mutation_factor(), 0.6);
    assert_eq!(differential.crossover_rate(), 0.7);
    assert_eq!(
        differential.strategy(),
        scioxide_optimize::DeStrategy::Best1
    );
    assert_eq!(differential.seed(), 7);
    assert_eq!(differential.time_limit(), Some(time_limit));

    let multistart = GlobalSearchOptions::<f64>::default()
        .with_starts(Budget::try_from(5).unwrap())
        .with_seed(11)
        .with_time_limit(time_limit);
    assert_eq!(multistart.starts(), Budget::try_from(5).unwrap());
    assert_eq!(multistart.seed(), 11);
    assert_eq!(multistart.time_limit(), Some(time_limit));
    assert_eq!(bounds.into_entries(), vec![(-2.0, 3.0), (-1.0, 4.0)]);

    let simplex = scioxide_optimize::SimplexSolver::<f64>::default()
        .with_tolerances(scioxide_optimize::OptimalityTolerances::new(
            scioxide_core::FeasibilityTolerance::from_tolerance(tolerance),
            scioxide_core::GradientTolerance::from_tolerance(tolerance),
            scioxide_core::ResidualTolerance::from_tolerance(tolerance),
            scioxide_core::RankThreshold::from_tolerance(tolerance),
        ))
        .with_max_iterations(iterations);
    let interior = scioxide_optimize::InteriorPointLp::<f64>::default()
        .with_tolerances(scioxide_optimize::OptimalityTolerances::new(
            scioxide_core::FeasibilityTolerance::from_tolerance(tolerance),
            scioxide_core::GradientTolerance::from_tolerance(tolerance),
            scioxide_core::ResidualTolerance::from_tolerance(tolerance),
            scioxide_core::RankThreshold::from_tolerance(tolerance),
        ))
        .with_max_iterations(iterations);
    let qp = scioxide_optimize::QpSolver::<f64>::default()
        .with_tolerances(scioxide_optimize::OptimalityTolerances::new(
            scioxide_core::FeasibilityTolerance::from_tolerance(tolerance),
            scioxide_core::GradientTolerance::from_tolerance(tolerance),
            scioxide_core::ResidualTolerance::from_tolerance(tolerance),
            scioxide_core::RankThreshold::from_tolerance(tolerance),
        ))
        .with_max_iterations(iterations);
    let branch = scioxide_optimize::BranchAndBound::<f64>::default()
        .with_integrality_tolerance(
            scioxide_core::IntegralityTolerance::new((tolerance).absolute()).unwrap(),
        )
        .with_relaxation_tolerances(scioxide_optimize::OptimalityTolerances::new(
            scioxide_core::FeasibilityTolerance::from_tolerance(tolerance),
            scioxide_core::GradientTolerance::from_tolerance(tolerance),
            scioxide_core::ResidualTolerance::from_tolerance(tolerance),
            scioxide_core::RankThreshold::from_tolerance(tolerance),
        ))
        .with_max_nodes(iterations)
        .with_max_lp_iterations(iterations);
    assert_eq!(simplex.tolerances().feasibility().tolerance(), tolerance);
    assert_eq!(
        simplex.max_iterations(&LpProblem::new(vec![0.0_f64]).unwrap()),
        iterations
    );
    assert_eq!(interior.tolerances().feasibility().tolerance(), tolerance);
    assert_eq!(interior.max_iterations(), iterations);
    assert_eq!(qp.tolerances().feasibility().tolerance(), tolerance);
    assert_eq!(qp.max_iterations(), iterations);
    assert_eq!(
        branch.integrality_tolerance().absolute(),
        tolerance.absolute()
    );
    assert_eq!(branch.max_nodes(), iterations);
    assert_eq!(branch.max_lp_iterations(), iterations);
}

#[test]
fn result_accessors_and_consuming_parts_preserve_solver_metadata() {
    let result = Bfgs::<f64>::default()
        .minimize(&LineSearchFailureProblem, &[1.0])
        .unwrap();
    let expected = (
        result.x().to_vec(),
        result.fun(),
        result.grad().map(<[f64]>::to_vec),
        result.iterations(),
        result.fun_evals(),
        result.grad_evals(),
        result.termination(),
        result.diagnostics(),
        result.message().to_owned(),
    );
    assert_eq!(result.into_parts(), expected);

    let accepted = LineSearchResult::try_accepted(0.5, 0.25, true, Some(-1.0), 2, 1).unwrap();
    assert_eq!(
        accepted.termination(),
        scioxide_optimize::LineSearchTermination::Accepted
    );
    assert_eq!(accepted.step(), 0.5);
    assert_eq!(accepted.value(), 0.25);
    assert!(accepted.gradient_available());
    assert_eq!(accepted.directional_derivative(), Some(-1.0));
}
