//! Native-bound QP regressions and independent original-coordinate certificates.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Bounds, Budget, ExecutionContext};
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    LinearConstraintSense as Sense, LinearConstraints, OptimizeTermination as Stop,
    ProgrammingResult, QpMethod, QpProblem, QpSolver,
};

fn verified(problem: &QpProblem<f64>, method: QpMethod) -> ProgrammingResult<f64> {
    let result = QpSolver::default()
        .with_method(method)
        .solve(problem)
        .unwrap();
    assert!(result.is_success(), "{method:?}: {result:?}");
    assert!(
        result.verification_report().unwrap().is_accepted(),
        "{result:?}"
    );
    result
}

#[test]
fn fixed_bound_and_redundant_equalities_agree_through_both_methods() {
    let problem = QpProblem::new(
        Matrix::from_vec(3, 3, vec![4., 1., 0., 1., 2., 1., 0., 1., 2.]).unwrap(),
        vec![0.; 3],
    )
    .unwrap()
    .with_bounds(
        Bounds::new(vec![
            (Some(1.), Some(1.)),
            (Some(0.), Some(4.)),
            (Some(0.), Some(4.)),
        ])
        .unwrap(),
    )
    .unwrap()
    .with_constraints(
        LinearConstraints::new(
            Matrix::from_vec(3, 3, vec![0., 1., 1., 1., 0., 0., 2., 0., 0.]).unwrap(),
            vec![2., 1., 2.],
            vec![Sense::Equal; 3],
        )
        .unwrap(),
    )
    .unwrap();
    for method in [QpMethod::ActiveSet, QpMethod::PrimalDual] {
        let result = verified(&problem, method);
        for (&x, expected) in result.x().iter().zip([1., 0.5, 1.5]) {
            assert!((x - expected).abs() < 1e-6, "{result:?}");
        }
    }
}

#[test]
fn all_fixed_coordinates_have_valid_signed_bound_certificates() {
    let problem = QpProblem::new(
        Matrix::from_vec(2, 2, vec![2., 1., 1., 4.]).unwrap(),
        vec![3., -7.],
    )
    .unwrap()
    .with_bounds(Bounds::new(vec![(Some(1.), Some(1.)), (Some(-2.), Some(-2.))]).unwrap())
    .unwrap();
    for method in [
        QpMethod::Automatic,
        QpMethod::ActiveSet,
        QpMethod::PrimalDual,
    ] {
        let r = verified(&problem, method);
        assert!(
            (r.x()[0] - 1.).abs() < 1e-10 && (r.x()[1] + 2.).abs() < 1e-10,
            "{r:?}"
        );
    }
}

#[test]
fn beale_degeneracy_terminates_as_a_semidefinite_qp() {
    for diagonal in [0., 1e-3] {
        let mut h = Matrix::zeros(4, 4).unwrap();
        for i in 0..4 {
            h.view_mut().set(i, i, diagonal).unwrap();
        }
        let p = QpProblem::new(h, vec![-10., 57., 9., 24.])
            .unwrap()
            .with_constraints(
                LinearConstraints::new(
                    Matrix::from_vec(
                        3,
                        4,
                        vec![0.5, -5.5, -2.5, 9., 0.5, -1.5, -0.5, 1., 1., 0., 0., 0.],
                    )
                    .unwrap(),
                    vec![0., 0., 1.],
                    vec![Sense::LessEqual; 3],
                )
                .unwrap(),
            )
            .unwrap()
            .with_bounds(Bounds::new(vec![(Some(0.), None); 4]).unwrap())
            .unwrap();
        let r = verified(&p, QpMethod::ActiveSet);
        assert!((r.fun() + 1. - diagonal).abs() < 1e-7, "{r:?}");
        assert!(r.iterations() < 1000);
    }
}

#[test]
fn duplicate_inequalities_and_native_bounds_do_not_create_singular_working_sets() {
    let p = QpProblem::new(
        Matrix::from_vec(2, 2, vec![2., 0., 0., 2.]).unwrap(),
        vec![-2., -6.],
    )
    .unwrap()
    .with_constraints(
        LinearConstraints::new(
            Matrix::from_vec(4, 2, vec![1., 1., 2., 2., -1., 0., 0., -1.]).unwrap(),
            vec![2., 4., 0., 0.],
            vec![Sense::LessEqual; 4],
        )
        .unwrap(),
    )
    .unwrap()
    .with_bounds(Bounds::new(vec![(Some(0.), Some(2.)); 2]).unwrap())
    .unwrap();
    for method in [QpMethod::ActiveSet, QpMethod::PrimalDual] {
        let r = verified(&p, method);
        assert!(
            r.x()[0].abs() < 2e-4 && (r.x()[1] - 2.).abs() < 2e-4,
            "{r:?}"
        );
    }
}

#[test]
fn fixed_bound_conflict_is_infeasible_for_both_methods() {
    let p = QpProblem::new(Matrix::from_vec(1, 1, vec![1.]).unwrap(), vec![0.])
        .unwrap()
        .with_bounds(Bounds::new(vec![(Some(1.), Some(1.))]).unwrap())
        .unwrap()
        .with_constraints(
            LinearConstraints::new(
                Matrix::from_vec(1, 1, vec![1.]).unwrap(),
                vec![2.],
                vec![Sense::Equal],
            )
            .unwrap(),
        )
        .unwrap();
    for method in [QpMethod::ActiveSet, QpMethod::PrimalDual] {
        let r = QpSolver::default().with_method(method).solve(&p).unwrap();
        assert_eq!(r.termination(), Stop::Infeasible, "{r:?}");
        assert!(r.verification_report().unwrap().is_accepted());
    }
}

#[test]
fn scaled_small_curvature_does_not_turn_a_finite_minimum_into_a_ray() {
    for scale in [1e-4, 1., 1e6] {
        let p = QpProblem::new(
            Matrix::from_vec(2, 2, vec![scale, 0., 0., scale * 1e-4]).unwrap(),
            vec![-scale, -2e-4 * scale],
        )
        .unwrap()
        .with_bounds(Bounds::new(vec![(Some(0.), None); 2]).unwrap())
        .unwrap();
        let r = verified(&p, QpMethod::ActiveSet);
        assert!(
            (r.x()[0] - 1.).abs() < 1e-6 && (r.x()[1] - 2.).abs() < 1e-6,
            "{r:?}"
        );
    }
}

#[test]
fn prepared_qp_restarts_changed_problem_and_preserves_output_on_provider_failure() {
    struct Bad;
    impl scioxide_optimize::KktProvider<f64> for Bad {
        fn factor(
            &mut self,
            a: scioxide_linalg::MatrixView<'_, f64>,
            _: ExecutionContext<'_>,
        ) -> Result<scioxide_sparse::LdlInertia, scioxide_core::SciOxideError> {
            scioxide_sparse::LdlInertia::new(a.rows(), 0, 0)
        }
        fn solve_into(
            &mut self,
            _: &[f64],
            out: &mut [f64],
            _: ExecutionContext<'_>,
        ) -> Result<(), scioxide_core::SciOxideError> {
            out.fill(f64::NAN);
            Ok(())
        }
    }
    let p = QpProblem::new(Matrix::from_vec(1, 1, vec![2.]).unwrap(), vec![-2.])
        .unwrap()
        .with_bounds(Bounds::new(vec![(Some(0.), Some(3.))]).unwrap())
        .unwrap();
    let solver = QpSolver::<f64>::default().with_method(QpMethod::ActiveSet);
    let mut w = solver.workspace(&p).unwrap();
    let mut out = ProgrammingResult::empty();
    solver
        .solve_into(&p, &mut out, &mut w, ExecutionContext::sequential())
        .unwrap();
    assert!((out.x()[0] - 1.).abs() < 1e-10);
    let p2 = QpProblem::new(Matrix::from_vec(1, 1, vec![4.]).unwrap(), vec![-8.])
        .unwrap()
        .with_bounds(Bounds::new(vec![(Some(0.), Some(3.))]).unwrap())
        .unwrap();
    solver
        .solve_into(&p2, &mut out, &mut w, ExecutionContext::sequential())
        .unwrap();
    assert!((out.x()[0] - 2.).abs() < 1e-10);
    let mut w = w.with_kkt_provider(Box::new(Bad));
    assert!(
        solver
            .solve_into(&p, &mut out, &mut w, ExecutionContext::sequential())
            .is_err()
    );
    assert!((out.x()[0] - 2.).abs() < 1e-10);
}

#[test]
fn small_budget_is_a_limit_and_f32_fixed_bounds_are_verified() {
    let p = QpProblem::new(
        Matrix::from_vec(2, 2, vec![2.0_f32, 0., 0., 2.]).unwrap(),
        vec![-2., -6.],
    )
    .unwrap()
    .with_bounds(Bounds::new(vec![(Some(1.), Some(1.)), (Some(0.), Some(4.))]).unwrap())
    .unwrap();
    for method in [QpMethod::ActiveSet, QpMethod::PrimalDual] {
        let r = QpSolver::default().with_method(method).solve(&p).unwrap();
        assert!(
            r.is_success() && r.verification_report().unwrap().is_accepted(),
            "{r:?}"
        );
        assert!((r.x()[1] - 3.).abs() < 1e-3);
        let r = QpSolver::default()
            .with_method(method)
            .with_max_iterations(Budget::ONE)
            .solve(&p)
            .unwrap();
        assert_eq!(r.termination(), Stop::IterationLimit, "{r:?}");
    }
}

#[test]
fn distant_rhs_does_not_relax_multiplier_signs_or_bound_blockers() {
    let p = QpProblem::new(Matrix::from_vec(1, 1, vec![2.]).unwrap(), vec![-2.])
        .unwrap()
        .with_bounds(Bounds::new(vec![(Some(0.), Some(0.5))]).unwrap())
        .unwrap()
        .with_constraints(
            LinearConstraints::new(
                Matrix::from_vec(1, 1, vec![1.]).unwrap(),
                vec![1e12],
                vec![Sense::LessEqual],
            )
            .unwrap(),
        )
        .unwrap();
    for method in [QpMethod::ActiveSet, QpMethod::PrimalDual] {
        let r = verified(&p, method);
        assert!((r.x()[0] - 0.5).abs() < 1e-6, "{r:?}");
    }
}

#[test]
fn objective_scaling_preserves_absolute_gradient_and_dual_tolerances() {
    let p = QpProblem::new(
        Matrix::from_vec(2, 2, vec![1e12, 0., 0., 1.]).unwrap(),
        vec![0., -1.],
    )
    .unwrap()
    .with_bounds(Bounds::new(vec![(Some(0.), None); 2]).unwrap())
    .unwrap();
    let r = verified(&p, QpMethod::ActiveSet);
    assert!(
        r.x()[0].abs() < 1e-10 && (r.x()[1] - 1.).abs() < 1e-8,
        "{r:?}"
    );
}

#[test]
fn thin_distinct_bounds_do_not_enter_as_dependent_coordinate_equalities() {
    let p = QpProblem::new(
        Matrix::from_vec(2, 2, vec![2., 0., 0., 2.]).unwrap(),
        vec![-1., 0.],
    )
    .unwrap()
    .with_bounds(Bounds::new(vec![(Some(0.), Some(1e-12)), (None, None)]).unwrap())
    .unwrap();
    let r = verified(&p, QpMethod::ActiveSet);
    assert!((r.x()[0] - 1e-12).abs() < 1e-15, "{r:?}");
}

#[test]
fn evaluation_stops_after_a_feasible_start_publish_the_current_point() {
    use scioxide_core::{BudgetKind, ProblemToken, SolveBudgets, SolveIdentity, SolveState};
    let p = QpProblem::new(
        Matrix::from_vec(2, 2, vec![2., 0., 0., 2.]).unwrap(),
        vec![-2., -6.],
    )
    .unwrap()
    .with_bounds(Bounds::new(vec![(Some(0.), Some(4.)); 2]).unwrap())
    .unwrap();
    for (method, limit) in [(QpMethod::ActiveSet, 2), (QpMethod::PrimalDual, 1)] {
        let solver = QpSolver::default().with_method(method);
        let mut workspace = solver.workspace(&p).unwrap();
        let mut result = ProgrammingResult::empty();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
        let identity = SolveIdentity::new(
            ProblemToken::new(1982),
            scioxide_core::algorithm_id!("test", "QpStop", 1),
            2,
            0,
        );
        let mut scope = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        solver
            .solve_into(&p, &mut result, &mut workspace, &mut scope)
            .unwrap();
        assert_eq!(
            result.termination(),
            Stop::EvaluationLimit,
            "{method:?}: {result:?}"
        );
        assert!(result.x().iter().all(|&x| (0.0..=4.0).contains(&x)));
    }
}

#[test]
fn large_inconsistent_equality_rhs_is_not_hidden_by_augmented_rank_scaling() {
    let p = QpProblem::new(Matrix::from_vec(1, 1, vec![1.]).unwrap(), vec![0.])
        .unwrap()
        .with_constraints(
            LinearConstraints::new(
                Matrix::from_vec(2, 1, vec![1., 1.]).unwrap(),
                vec![1e12, 2e12],
                vec![Sense::Equal; 2],
            )
            .unwrap(),
        )
        .unwrap();
    for method in [QpMethod::ActiveSet, QpMethod::PrimalDual] {
        let r = QpSolver::default().with_method(method).solve(&p).unwrap();
        assert!(!r.is_success(), "{r:?}");
        assert!(
            matches!(
                r.termination(),
                Stop::Infeasible | Stop::SuspectedInfeasible
            ),
            "{r:?}"
        );
    }
}

#[test]
fn rotated_semidefinite_recession_is_checked_with_relative_null_tests() {
    let p = QpProblem::new(
        Matrix::from_vec(2, 2, vec![1., 2., 2., 4.]).unwrap(),
        vec![-2., 1.],
    )
    .unwrap()
    .with_bounds(Bounds::new(vec![(Some(0.), None), (None, None)]).unwrap())
    .unwrap();
    let r = QpSolver::default()
        .with_method(QpMethod::ActiveSet)
        .solve(&p)
        .unwrap();
    assert_eq!(r.termination(), Stop::Unbounded, "{r:?}");
    assert!(r.verification_report().unwrap().is_accepted());
}
