//! Analytic box quadratics have exact minimizers obtained by coordinatewise
//! projection of the known unconstrained minimizer. Coupled active-set and
//! projected-gradient checks provide independent numerical comparisons.

#[cfg(feature = "std")]
use scioxide_core::ThreadExecutor;
use scioxide_core::{
    Bounds, Budget, BudgetKind, EvaluationRole, ExecutionContext, GradientTolerance, ProblemToken,
    SciOxideError, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_differentiate::DerivativeStrategy;
#[cfg(not(feature = "std"))]
#[path = "support/executor.rs"]
mod executor;
#[cfg(not(feature = "std"))]
use executor::PartitionExecutor;
use scioxide_optimize::{
    BoundedObjective, LbfgsB, LbfgsBOptions, ObjectiveFunction, OptimizeResult, OptimizeTermination,
};

struct Obstacle<S> {
    load: Vec<S>,
}
impl<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>> ObjectiveFunction<S>
    for Obstacle<S>
{
    fn dim(&self) -> usize {
        self.load.len()
    }
    fn eval(&self, x: &[S]) -> Result<S, SciOxideError> {
        let energy = (0..=x.len())
            .map(|i| {
                let left = if i == 0 { S::ZERO } else { x[i - 1] };
                let right = if i == x.len() { S::ZERO } else { x[i] };
                let difference = right - left;
                difference * difference / S::TWO
            })
            .fold(S::ZERO, |sum, e| sum + e);
        Ok(energy
            - x.iter()
                .zip(&self.load)
                .fold(S::ZERO, |sum, (&x, &q)| sum + x * q))
    }
    fn gradient_into(&self, x: &[S], out: &mut [S]) -> Result<bool, SciOxideError> {
        for (i, g) in out.iter_mut().enumerate() {
            let left = if i == 0 { S::ZERO } else { x[i - 1] };
            let right = if i + 1 == x.len() { S::ZERO } else { x[i + 1] };
            *g = S::TWO * x[i] - left - right - self.load[i];
        }
        Ok(true)
    }
}

#[allow(clippy::unwrap_used)]
fn obstacle_matches_bvls<S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>>() {
    use scioxide_linalg::Matrix;
    use scioxide_optimize::{BoundedLinearLeastSquares, LinearLeastSquaresProblem};
    let cv = |x| S::try_from_f64_lossy(x).unwrap();
    let n = 12;
    let load: Vec<_> = (1..=n)
        .map(|i| cv(0.6 * (2.0 * core::f64::consts::PI * i as f64 / 13.0).sin() + 0.25))
        .collect();
    let objective = Obstacle { load };
    // Cx = (x_1, x_2-x_1, ..., x_12-x_11, -x_12).
    // Backward cumulative sums give C^T d = q without another solver.
    let coefficients: Vec<_> = (0..=n)
        .flat_map(|row| {
            (0..n).map(move |col| {
                if row == col {
                    S::ONE
                } else if row == col + 1 {
                    -S::ONE
                } else {
                    S::ZERO
                }
            })
        })
        .collect();
    let matrix = Matrix::from_vec(n + 1, n, coefficients).unwrap();
    let mut rhs = vec![S::ZERO; n + 1];
    for i in (0..n).rev() {
        rhs[i] = rhs[i + 1] + objective.load[i];
    }
    for upper in [1.5, 3.0, 4.0] {
        let bounds = Bounds::new(vec![(Some(S::ZERO), Some(cv(upper))); n]).unwrap();
        let linear = LinearLeastSquaresProblem::from_dense(&matrix, &rhs)
            .unwrap()
            .with_bounds(&bounds)
            .unwrap();
        let reference = BoundedLinearLeastSquares::default().solve(&linear).unwrap();
        assert_eq!(
            reference.termination(),
            OptimizeTermination::Converged,
            "{reference:?}"
        );
        let reference_norm = reference
            .x()
            .iter()
            .map(|v| v.try_to_f64_exact().unwrap().abs())
            .fold(0., f64::max);
        // f32 uses the scalar's sqrt-epsilon stopping scale, not an unattainable
        // sub-ulp coordinate target near the upper bound.
        let coordinate_tolerance = if S::MANTISSA_DIGITS <= 24 {
            8. * (f32::EPSILON as f64).sqrt() * (1. + reference_norm)
        } else {
            1e-7
        };
        for memory in [1, 10] {
            let solver = LbfgsB::new(
                LbfgsBOptions::default().with_memory(Budget::try_from(memory).unwrap()),
            );
            let problem = BoundedObjective::new(&objective, &bounds).unwrap();
            let result = solver
                .minimize(&problem, &vec![cv(upper * 0.5); n])
                .unwrap();
            assert_eq!(
                result.termination(),
                OptimizeTermination::GradientSmall,
                "upper {upper}, memory {memory}: {result:?}"
            );
            assert!(
                result.iterations() <= 60,
                "upper {upper}, memory {memory}: {result:?}"
            );
            for (&actual, &expected) in result.x().iter().zip(reference.x()) {
                assert!(
                    (actual.try_to_f64_exact().unwrap() - expected.try_to_f64_exact().unwrap())
                        .abs()
                        <= coordinate_tolerance,
                    "upper {upper}, memory {memory}: {result:?}, reference {reference:?}"
                );
            }
            let x: Vec<_> = result
                .x()
                .iter()
                .map(|v| v.try_to_f64_exact().unwrap())
                .collect();
            let projected = x
                .iter()
                .enumerate()
                .map(|(i, &value)| {
                    assert!((0.0..=upper).contains(&value));
                    let left = if i == 0 { 0. } else { x[i - 1] };
                    let right = if i + 1 == n { 0. } else { x[i + 1] };
                    let gradient =
                        2. * value - left - right - objective.load[i].try_to_f64_exact().unwrap();
                    ((value - gradient).clamp(0., upper) - value).abs()
                })
                .fold(0., f64::max);
            assert!(
                projected <= coordinate_tolerance,
                "projected gradient {projected}: {result:?}"
            );
        }
    }
}

#[test]
fn obstacle_active_set_retains_curvature_f64() {
    obstacle_matches_bvls::<f64>();
}
#[test]
fn obstacle_active_set_retains_curvature_f32() {
    obstacle_matches_bvls::<f32>();
}

struct Quadratic64 {
    target: Vec<f64>,
}
impl ObjectiveFunction<f64> for Quadratic64 {
    fn dim(&self) -> usize {
        self.target.len()
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x.iter()
            .zip(&self.target)
            .map(|(&v, &t)| (v - t) * (v - t))
            .sum())
    }
    fn gradient_into(&self, x: &[f64], output: &mut [f64]) -> Result<bool, SciOxideError> {
        output
            .iter_mut()
            .zip(x.iter().zip(&self.target))
            .for_each(|(out, (&v, &t))| *out = 2.0 * (v - t));
        Ok(true)
    }
}

#[test]
fn mixed_bounds_and_fixed_coordinate() {
    let objective = Quadratic64 {
        target: vec![-2.0, 3.0, 7.0, -1.0],
    };
    let bounds = Bounds::new(vec![
        (Some(0.0), None),
        (None, Some(2.0)),
        (Some(4.0), Some(4.0)),
        (None, None),
    ])
    .unwrap();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let solver = LbfgsB::default();
    let result = solver.minimize(&problem, &[2.0, -1.0, 4.0, 3.0]).unwrap();
    assert!(result.is_success(), "{result:?}");
    let expected = [0.0, 2.0, 4.0, -1.0];
    for (value, expected) in result.x().iter().zip(expected) {
        assert!((value - expected).abs() < 1e-5, "{result:?}");
    }
    assert_eq!(result.bound_violation(), Some(0.0));
    assert!(result.stationarity().unwrap() < 1e-5);
    assert!(result.grad().unwrap()[2] < 0.0); // fixed coordinate keeps the original analytic derivative
}

#[test]
fn narrow_box_with_finite_differences_stays_feasible() {
    struct DomainObjective;
    impl ObjectiveFunction<f64> for DomainObjective {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            assert!(x[0] >= 1.0 && x[0] <= 1.0 + 1e-8 && x[1] == 5.0);
            Ok((x[0] - (1.0 + 5e-9)).powi(2))
        }
    }
    let bounds = Bounds::new(vec![(Some(1.0), Some(1.0 + 1e-8)), (Some(5.0), Some(5.0))]).unwrap();
    let problem = BoundedObjective::new(&DomainObjective, &bounds).unwrap();
    let solver =
        LbfgsB::new(LbfgsBOptions::default().with_derivative_strategy(DerivativeStrategy::Central));
    let result = solver.minimize(&problem, &[1.0, 5.0]).unwrap();
    assert!(result.is_success(), "{result:?}");
    assert_eq!(result.grad().unwrap()[1], 0.0);
}

#[test]
fn all_fixed_coordinates_stop_at_initial_point() {
    let objective = Quadratic64 {
        target: vec![8.0, -9.0],
    };
    let bounds = Bounds::new(vec![(Some(2.0), Some(2.0)), (Some(-3.0), Some(-3.0))]).unwrap();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let result = LbfgsB::default().minimize(&problem, &[2.0, -3.0]).unwrap();
    assert!(result.is_success());
    assert_eq!(result.x(), &[2.0, -3.0]);
    assert_eq!(result.iterations(), 0);
    assert_eq!(result.stationarity(), Some(0.0));
}

#[test]
fn infeasible_start_is_rejected_before_callback() {
    let objective = Quadratic64 { target: vec![0.0] };
    let bounds = Bounds::new(vec![(Some(0.0), Some(1.0))]).unwrap();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    assert!(LbfgsB::default().minimize(&problem, &[2.0]).is_err());
}

#[test]
fn prepared_workspace_reuses_storage_and_reports_partial_limit() {
    let objective = Quadratic64 {
        target: vec![3.0, -2.0],
    };
    let bounds = Bounds::new(vec![(Some(0.0), Some(4.0)), (Some(-3.0), Some(0.0))]).unwrap();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let solver = LbfgsB::new(LbfgsBOptions::default().with_max_iterations(Budget::ONE));
    let mut workspace = solver.workspace(2).unwrap();
    let mut output = OptimizeResult::empty();
    solver
        .minimize_into(
            &problem,
            &[0.0, 0.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output.x().len(), 2);
    assert!(output.bound_violation().unwrap() == 0.0);
    assert_eq!(output.termination(), OptimizeTermination::IterationLimit);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::Stopped
    );
    assert_eq!(workspace.last_accepted().unwrap().point(), output.x());
    solver
        .minimize_into(
            &problem,
            &[0.0, 0.0],
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output.x().len(), 2);
}

#[test]
fn f32_bounded_quadratic() {
    struct Quadratic32;
    impl ObjectiveFunction<f32> for Quadratic32 {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f32]) -> Result<f32, SciOxideError> {
            Ok((x[0] - 2.0).powi(2) + (x[1] + 3.0).powi(2))
        }
        fn gradient_into(&self, x: &[f32], out: &mut [f32]) -> Result<bool, SciOxideError> {
            out[0] = 2.0 * (x[0] - 2.0);
            out[1] = 2.0 * (x[1] + 3.0);
            Ok(true)
        }
    }
    let bounds = Bounds::new(vec![(Some(0.0), Some(1.0)), (Some(-2.0), Some(2.0))]).unwrap();
    let problem = BoundedObjective::new(&Quadratic32, &bounds).unwrap();
    let solver =
        LbfgsB::new(LbfgsBOptions::default().with_gradient_tolerance(GradientTolerance::default()));
    let result = solver.minimize(&problem, &[0.25, 1.0]).unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!((result.x()[0] - 1.0).abs() < 1e-4);
    assert!((result.x()[1] + 2.0).abs() < 1e-4);
}

#[test]
fn coupled_quadratic_changes_active_set() {
    struct Coupled;
    impl ObjectiveFunction<f64> for Coupled {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok((x[0] + x[1] - 1.0).powi(2) + 4.0 * (x[0] - x[1]).powi(2))
        }
        fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
            let a = 2.0 * (x[0] + x[1] - 1.0);
            let b = 8.0 * (x[0] - x[1]);
            out[0] = a + b;
            out[1] = a - b;
            Ok(true)
        }
    }
    let bounds = Bounds::new(vec![(Some(0.0), None), (None, Some(0.2))]).unwrap();
    let problem = BoundedObjective::new(&Coupled, &bounds).unwrap();
    let result = LbfgsB::default().minimize(&problem, &[1.0, -1.0]).unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!((result.x()[0] - 0.32).abs() < 1e-5, "{result:?}");
    assert!((result.x()[1] - 0.2).abs() < 1e-8, "{result:?}");
}

#[test]
fn nonsmooth_objective_never_claims_success_from_step_only() {
    struct Absolute;
    impl ObjectiveFunction<f64> for Absolute {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Ok((x[0] - 0.37).abs())
        }
        fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
            out[0] = if x[0] < 0.37 {
                -1.0
            } else if x[0] > 0.37 {
                1.0
            } else {
                0.0
            };
            Ok(true)
        }
    }
    let bounds = Bounds::new(vec![(Some(0.0), Some(1.0))]).unwrap();
    let problem = BoundedObjective::new(&Absolute, &bounds).unwrap();
    let result = LbfgsB::default().minimize(&problem, &[0.0]).unwrap();
    if result.is_success() {
        assert!(result.stationarity().unwrap() <= 1e-5);
    }
    assert!(result.x()[0] >= 0.0 && result.x()[0] <= 1.0);
}

#[test]
fn context_budget_preserves_last_accepted_feasible_point() {
    let objective = Quadratic64 { target: vec![2.0] };
    let bounds = Bounds::new(vec![(Some(0.0), Some(1.0))]).unwrap();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let solver = LbfgsB::<f64>::default();
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(3).unwrap());
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            SolveIdentity::new(
                ProblemToken::new(657),
                scioxide_core::algorithm_id!("test", "Parent", 1),
                1,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    let mut workspace = solver.workspace(1).unwrap();
    let mut result = OptimizeResult::empty();
    solver
        .minimize_into(&problem, &[0.5], &mut result, &mut workspace, &mut context)
        .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
    assert_eq!(result.x().len(), 1);
    assert!(result.x()[0] >= 0.0 && result.x()[0] <= 1.0);
    assert_eq!(result.counts(), workspace.last_attempt().unwrap().counts());
    assert_eq!(
        context
            .counts()
            .nested()
            .provider(EvaluationRole::Objective),
        result.counts().own().provider(EvaluationRole::Objective)
    );
    assert!(result.counts().own().provider(EvaluationRole::Objective) >= 1);
    assert!(result.bound_violation().is_some());
}

#[test]
fn absent_endpoints_and_injected_executor_match_sequential() {
    let objective = Quadratic64 {
        target: vec![0.5, -0.25],
    };
    let bounds = Bounds::new(vec![(None, None), (None, None)]).unwrap();
    let problem = BoundedObjective::new(&objective, &bounds).unwrap();
    let solver = LbfgsB::<f64>::default();
    let mut workspace = solver.workspace(2).unwrap();
    let mut sequential = OptimizeResult::empty();
    solver
        .minimize_into(
            &problem,
            &[2.0, 1.0],
            &mut sequential,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    #[cfg(feature = "std")]
    let executor = ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    #[cfg(not(feature = "std"))]
    let executor = PartitionExecutor(core::num::NonZeroUsize::new(2).unwrap());
    let mut injected = OptimizeResult::empty();
    solver
        .minimize_into(
            &problem,
            &[2.0, 1.0],
            &mut injected,
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap();
    assert!(sequential.is_success() && injected.is_success());
    assert_eq!(sequential.x(), injected.x());
    assert_eq!(sequential.fun_evals(), injected.fun_evals());
}
