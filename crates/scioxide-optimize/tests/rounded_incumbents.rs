#![allow(clippy::unwrap_used, clippy::suboptimal_flops)]
//! Independent small-domain MIP oracles; no relaxation values serve as expected answers.

use scioxide_core::{
    Bounds, Budget, FeasibilityTolerance, IntegralityTolerance, NumericScalar, RealScalar,
};
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    BranchAndBound, LinearConstraintSense, LinearConstraints, LpProblem, MipProblem,
    OptimalityTolerances, OptimizeTermination,
};

fn scalar<S: RealScalar>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}

fn solver<S: RealScalar>() -> BranchAndBound<S> {
    let default = BranchAndBound::<S>::default();
    let roles = default.relaxation_tolerances();
    default
        .with_presolve(false)
        .with_branching(scioxide_optimize::MipBranching::MostFractional)
        .with_limits(
            scioxide_optimize::MipLimits::default()
                .with_cut_rounds(0)
                .with_repair(false),
        )
        .with_integrality_tolerance(IntegralityTolerance::new(scalar(0.25)).unwrap())
        .with_relaxation_tolerances(OptimalityTolerances::new(
            FeasibilityTolerance::new(scalar(1e-6), scalar(0.0)).unwrap(),
            roles.gradient(),
            roles.complementarity(),
            roles.rank(),
        ))
}

fn bounded<S: RealScalar + NumericScalar<Real = S>>(
    objective: f64,
    lower: f64,
    upper: f64,
) -> MipProblem<S> {
    MipProblem::new(
        LpProblem::new(vec![scalar::<S>(objective)])
            .unwrap()
            .with_bounds(Bounds::new(vec![(Some(scalar(lower)), Some(scalar(upper)))]).unwrap())
            .unwrap(),
        vec![0],
    )
    .unwrap()
}

fn check_result<S: RealScalar + NumericScalar<Real = S>>(
    problem: &MipProblem<S>,
    result: &scioxide_optimize::MipResult<S>,
) {
    for &index in problem.integer_variables() {
        assert_eq!(result.x()[index], RealScalar::round(result.x()[index]));
    }
    let x: Vec<_> = result.x().iter().map(|value| value.into_f64()).collect();
    let lp = problem.relaxation();
    let expected: f64 = lp
        .objective()
        .iter()
        .zip(&x)
        .map(|(c, x)| c.into_f64() * x)
        .sum();
    assert!((result.fun().into_f64() - expected).abs() <= 1e-6);
    if let Some(constraints) = lp.constraints() {
        for row in 0..constraints.len() {
            let lhs: f64 = x
                .iter()
                .enumerate()
                .map(|(col, x)| constraints.coefficients().get(row, col).unwrap().into_f64() * x)
                .sum();
            let rhs = constraints.rhs()[row].into_f64();
            let residual = match constraints.senses()[row] {
                LinearConstraintSense::LessEqual => lhs - rhs,
                LinearConstraintSense::GreaterEqual => rhs - lhs,
                LinearConstraintSense::Equal => (lhs - rhs).abs(),
            };
            assert!(residual <= 1e-6, "constraint residual {residual}");
        }
    }
    for (index, &value) in x.iter().enumerate() {
        let (lower, upper) = lp
            .bounds()
            .map_or((Some(scalar(0.0)), None), |b| b.as_slice()[index]);
        assert!(lower.is_none_or(|lo| value >= lo.into_f64() - 1e-6));
        assert!(upper.is_none_or(|hi| value <= hi.into_f64() + 1e-6));
    }
    assert_eq!(result.integrality_violation(), Some(scalar(0.0)));
}

fn rounding_search<S: RealScalar + NumericScalar<Real = S>>() {
    for (objective, lower, upper, expected) in [
        (-2.0, 0.0, 2.125, 2.0), // feasible rounded root, changed objective
        (2.0, 0.875, 3.0, 1.0),  // rounding up raises minimization objective
        (-2.0, 0.0, 1.875, 1.0), // rounded root violates upper bound
        (2.0, 1.125, 3.0, 2.0),  // rounded root violates lower bound
        (-2.0, -3.0, -1.125, -2.0),
        (2.0, -2.125, -1.0, -2.0),
    ] {
        let problem = bounded::<S>(objective, lower, upper);
        let result = solver().solve(&problem).unwrap();
        assert!(result.is_success(), "{}", result.message());
        check_result(&problem, &result);
        assert_eq!(result.x(), &[scalar::<S>(expected)]);
        assert_eq!(result.absolute_gap(), Some(scalar(0.0)));
        assert!(result.nodes() > 1);
    }
    let problem = bounded::<S>(-2.0, 0.0, 2.125);
    let result = solver()
        .with_max_nodes(Budget::ONE)
        .solve(&problem)
        .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::NodeLimit);
    check_result(&problem, &result);
    assert_eq!(result.fun(), scalar(-4.0));
    assert_eq!(result.absolute_gap(), Some(scalar(0.25)));
    assert_eq!(
        result.trust(),
        scioxide_core::PartialResultTrust::Admissible
    );

    let problem = bounded::<S>(-2.0, 0.0, 1.875);
    let result = solver()
        .with_max_nodes(Budget::ONE)
        .solve(&problem)
        .unwrap();
    assert_eq!(result.termination(), OptimizeTermination::NodeLimit);
    assert_eq!(
        result.trust(),
        scioxide_core::PartialResultTrust::DiagnosticOnly
    );
    assert_eq!(result.certified_optimality_gap(), None);
}

fn enumerated<S: RealScalar + NumericScalar<Real = S>>() {
    // Enumerate the exact integer feasible set independently for both objective directions.
    for rhs in [2.125, 2.875, 3.5] {
        for c in [[-1.0, -2.0], [1.0, -1.0]] {
            let constraints = LinearConstraints::new(
                Matrix::from_vec(1, 2, vec![scalar::<S>(1.0); 2]).unwrap(),
                vec![scalar(rhs)],
                vec![LinearConstraintSense::LessEqual],
            )
            .unwrap();
            let lp = LpProblem::new(c.iter().map(|&value| scalar(value)).collect())
                .unwrap()
                .with_constraints(constraints)
                .unwrap()
                .with_bounds(Bounds::new(vec![(Some(scalar(0.0)), Some(scalar(3.0))); 2]).unwrap())
                .unwrap();
            let problem = MipProblem::new(lp, vec![0, 1]).unwrap();
            let result = solver().solve(&problem).unwrap();
            assert!(result.is_success(), "{}", result.message());
            check_result(&problem, &result);
            let expected = (0..=3)
                .flat_map(|x| (0..=3).map(move |y| (f64::from(x), f64::from(y))))
                .filter(|(x, y)| x + y <= rhs)
                .map(|(x, y)| c[0] * x + c[1] * y)
                .fold(f64::INFINITY, f64::min);
            assert_eq!(result.fun().into_f64(), expected);
        }
    }
}

#[test]
fn rounded_search_f32() {
    rounding_search::<f32>();
    enumerated::<f32>();
}
#[test]
fn rounded_search_f64() {
    rounding_search::<f64>();
    enumerated::<f64>();
}

fn sparse_context<S: RealScalar + NumericScalar<Real = S>>() {
    // min y with x >= 7/8 and y >= 3/8; x alone is integer.
    let constraints = LinearConstraints::new(
        scioxide_sparse::CsrMatrix::from_raw(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![scalar::<S>(1.0); 2],
        )
        .unwrap(),
        vec![scalar(0.875), scalar(0.375)],
        vec![LinearConstraintSense::GreaterEqual; 2],
    )
    .unwrap();
    let problem = MipProblem::new(
        LpProblem::new(vec![scalar(0.0), scalar(1.0)])
            .unwrap()
            .with_constraints(constraints)
            .unwrap(),
        vec![0],
    )
    .unwrap();
    let solver = solver::<S>();
    let mut state = scioxide_core::SolveState::new();
    let mut budgets = scioxide_core::SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            scioxide_core::ExecutionContext::new(&scioxide_core::SequentialExecutor),
            scioxide_core::SolveIdentity::new(
                scioxide_core::ProblemToken::new(42),
                scioxide_core::algorithm_id!("optimize", "rounded-parent", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut result = scioxide_optimize::MipResult::empty();
    solver
        .solve_into(&problem, &mut result, &mut workspace, &mut context)
        .unwrap();
    assert!(result.is_success(), "{}", result.message());
    check_result(&problem, &result);
    assert_eq!(result.x(), &[scalar::<S>(1.0), scalar(0.375)]);
    assert_eq!(result.nodes(), 1);
    // The search ran in a child epoch; its work is the parent's nested work.
    assert_eq!(
        result.counts().total().unwrap(),
        context.counts().total().unwrap()
    );
}

#[test]
fn sparse_context_rounds_root_and_keeps_continuous_coordinates() {
    sparse_context::<f32>();
    sparse_context::<f64>();
}

#[test]
fn rounded_equality_is_rejected_without_claiming_integer_optimality() {
    let constraints = LinearConstraints::new(
        Matrix::from_vec(1, 1, vec![1.0_f64]).unwrap(),
        vec![1.125],
        vec![LinearConstraintSense::Equal],
    )
    .unwrap();
    let problem = MipProblem::new(
        LpProblem::new(vec![1.0])
            .unwrap()
            .with_constraints(constraints)
            .unwrap(),
        vec![0],
    )
    .unwrap();
    let result = solver().solve(&problem).unwrap();
    assert_eq!(result.termination(), OptimizeTermination::Infeasible);
    assert_eq!(
        result.trust(),
        scioxide_core::PartialResultTrust::DiagnosticOnly
    );
    assert_eq!(result.certified_optimality_gap(), None);
}
