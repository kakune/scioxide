//! Independent programming solutions and certificate workflows.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Bounds, Budget, ExecutionContext, SciOxideError};
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    InteriorPointLp, LinearConstraintSense as Sense, LinearConstraints, LpProblem,
    OptimizeTermination as Stop, ProgrammingResult, QpMethod, QpProblem, QpSolver, SimplexSolver,
};
fn lp(c: Vec<f64>, a: Vec<f64>, b: Vec<f64>, sense: Vec<Sense>) -> LpProblem<f64> {
    let n = c.len();
    let m = b.len();
    LpProblem::new(c)
        .unwrap()
        .with_constraints(
            LinearConstraints::new(Matrix::from_vec(m, n, a).unwrap(), b, sense).unwrap(),
        )
        .unwrap()
}
#[test]
fn revised_simplex_handles_redundancy_and_cycling_fixture() {
    let redundant = lp(
        vec![1., 1.],
        vec![1., 1., 2., 2.],
        vec![1., 2.],
        vec![Sense::Equal; 2],
    );
    let result = SimplexSolver::default().solve(&redundant).unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!((result.fun() - 1.).abs() < 1e-8);
    // Beale's cycling example, minimization signs, Bland pricing must terminate.
    let cycling = lp(
        vec![-10., 57., 9., 24.],
        vec![0.5, -5.5, -2.5, 9., 0.5, -1.5, -0.5, 1., 1., 0., 0., 0.],
        vec![0., 0., 1.],
        vec![Sense::LessEqual; 3],
    );
    let result = SimplexSolver::default().solve(&cycling).unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!((result.fun() + 1.).abs() < 1e-8);
}
#[test]
fn hsd_infeasible_witness_is_independently_accepted() {
    let p = lp(
        vec![1.],
        vec![1., 1.],
        vec![1., 2.],
        vec![Sense::LessEqual, Sense::GreaterEqual],
    );
    let r = InteriorPointLp::default().solve(&p).unwrap();
    assert_eq!(r.termination(), Stop::Infeasible, "{r:?}");
    assert!(r.verification_report().unwrap().is_accepted());
}
#[test]
fn hsd_unbounded_requires_feasible_base_and_ray() {
    let p = lp(vec![-1., 0.], vec![1., -1.], vec![1.], vec![Sense::Equal]);
    let r = InteriorPointLp::default().solve(&p).unwrap();
    assert_eq!(r.termination(), Stop::Unbounded, "{r:?}");
    assert!(r.verification_report().unwrap().is_accepted());
}
#[test]
fn qp_active_and_primal_dual_match_exact_solution() {
    let p = QpProblem::<f64>::new(
        Matrix::from_vec(2, 2, vec![2., 0., 0., 2.]).unwrap(),
        vec![-2., -6.],
    )
    .unwrap()
    .with_constraints(
        LinearConstraints::new(
            Matrix::from_vec(1, 2, vec![1., 1.]).unwrap(),
            vec![2.],
            vec![Sense::LessEqual],
        )
        .unwrap(),
    )
    .unwrap()
    .with_bounds(Bounds::new(vec![(Some(0.), None); 2]).unwrap())
    .unwrap();
    for method in [QpMethod::Automatic, QpMethod::PrimalDual] {
        let r = QpSolver::default().with_method(method).solve(&p).unwrap();
        assert!(r.is_success(), "{method:?}: {r:?}");
        assert!(r.x()[0].abs() < 1e-4);
        assert!((r.x()[1] - 2.).abs() < 1e-4);
    }
}
#[test]
fn qp_semidefinite_and_dependent_constraints() {
    let p = QpProblem::<f64>::new(
        Matrix::from_vec(2, 2, vec![2., 2., 2., 2.]).unwrap(),
        vec![-2., -2.],
    )
    .unwrap()
    .with_constraints(
        LinearConstraints::new(
            Matrix::from_vec(2, 2, vec![1., 1., 2., 2.]).unwrap(),
            vec![1., 2.],
            vec![Sense::Equal; 2],
        )
        .unwrap(),
    )
    .unwrap();
    for method in [QpMethod::Automatic, QpMethod::PrimalDual] {
        let r = QpSolver::default().with_method(method).solve(&p).unwrap();
        assert!(r.is_success(), "{method:?}: {r:?}");
        assert!((r.x().iter().sum::<f64>() - 1.).abs() < 1e-7);
    }
}
#[test]
fn limits_are_not_infeasibility() {
    let p = lp(
        vec![-1., -2.],
        vec![1., 1., 1., 2.],
        vec![3., 5.],
        vec![Sense::Equal; 2],
    );
    for r in [
        SimplexSolver::default()
            .with_max_iterations(Budget::ONE)
            .solve(&p),
        InteriorPointLp::default()
            .with_max_iterations(Budget::ONE)
            .solve(&p),
    ] {
        let r = r.unwrap();
        assert_eq!(r.termination(), Stop::IterationLimit);
    }
}
#[test]
fn invalid_hessian_is_rejected() {
    assert!(matches!(
        (QpProblem::new(
            Matrix::from_vec(2, 2, vec![1., 1., 0., 1.]).unwrap(),
            vec![0., 0.]
        ))
        .as_ref()
        .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
    ));
}

#[test]
fn sparse_basis_updates_and_explicit_fill_limit() {
    use scioxide_sparse::{DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder, SparseLuLimits};
    let mut builder = SparseBuilder::new(3, 2);
    for (i, j, v) in [(0, 0, 1.), (0, 1, 1.), (1, 0, 1.), (2, 1, 1.)] {
        builder.push(i, j, v).unwrap();
    }
    let a = builder
        .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let p = LpProblem::new(vec![-1.0_f64, -2.])
        .unwrap()
        .with_constraints(
            LinearConstraints::new(a, vec![4., 3., 2.], vec![Sense::LessEqual; 3]).unwrap(),
        )
        .unwrap();
    let result = SimplexSolver::default()
        .with_basis_update_limit(Budget::ONE)
        .solve(&p)
        .unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!((result.fun() + 6.).abs() < 1e-8);
    assert!(matches!(
        (SimplexSolver::default()
            .with_sparse_limits(SparseLuLimits::new(1, 1024))
            .solve(&p))
        .as_ref()
        .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::ResourceLimit { .. })
    ));
}

#[test]
fn warm_basis_reuses_a_feasible_solution_and_restarts_changed_structure_cold() {
    let p = lp(
        vec![-1., -2.],
        vec![1., 1., 1., 0., 0., 1.],
        vec![4., 3., 2.],
        vec![Sense::LessEqual; 3],
    );
    let solver = SimplexSolver::default();
    let mut workspace = solver.workspace(&p).unwrap();
    let mut cold = ProgrammingResult::empty();
    solver
        .solve_into(
            &p,
            &mut cold,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let warm = workspace.warm_start().unwrap();
    assert!(!warm.reused_basis());
    let mut repeated = ProgrammingResult::empty();
    solver
        .resume_into(
            &p,
            &warm,
            &mut repeated,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(cold.is_success() && repeated.is_success());
    assert!(workspace.warm_start().unwrap().reused_basis());
    assert_eq!(repeated.iterations(), 0);
    assert_eq!(cold.fun(), repeated.fun());
    // A changed coefficient matrix no longer fails: the resume reoptimizes by
    // restarting cold and still reaches the verified optimum.
    let changed = lp(
        vec![-1., -2.],
        vec![2., 1., 1., 0., 0., 1.],
        vec![4., 3., 2.],
        vec![Sense::LessEqual; 3],
    );
    let mut restarted = ProgrammingResult::empty();
    solver
        .resume_into(
            &changed,
            &warm,
            &mut restarted,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(!workspace.warm_start().unwrap().reused_basis());
    let reference = solver.solve(&changed).unwrap();
    assert!(restarted.is_success(), "{restarted:?}");
    assert_eq!(restarted.x(), reference.x());
    assert_eq!(restarted.fun(), reference.fun());
}

#[test]
fn dense_and_sparse_kkt_providers_agree() {
    use scioxide_optimize::{DenseKktProvider, KktProvider, SparseKktProvider};
    let p = lp(
        vec![-1., -2.],
        vec![1., 1., 1., 0., 0., 1.],
        vec![4., 3., 2.],
        vec![Sense::LessEqual; 3],
    );
    let q = QpProblem::<f64>::new(
        Matrix::from_vec(2, 2, vec![2., 0., 0., 2.]).unwrap(),
        vec![-2., -6.],
    )
    .unwrap()
    .with_constraints(
        LinearConstraints::new(
            Matrix::from_vec(1, 2, vec![1., 1.]).unwrap(),
            vec![2.],
            vec![Sense::Equal],
        )
        .unwrap(),
    )
    .unwrap();
    let providers = || -> [Box<dyn KktProvider<f64> + Send>; 2] {
        [
            Box::new(DenseKktProvider::new(16).unwrap()),
            Box::new(SparseKktProvider::new(
                scioxide_sparse::SparseSymmetricLimits::new(256, 512, 100_000),
            )),
        ]
    };
    for (lp_provider, qp_provider) in providers().into_iter().zip(providers()) {
        let solver = InteriorPointLp::default();
        let mut workspace = solver.workspace(&p).unwrap().with_kkt_provider(lp_provider);
        let mut r = ProgrammingResult::empty();
        solver
            .solve_into(&p, &mut r, &mut workspace, ExecutionContext::sequential())
            .unwrap();
        assert!(r.is_success(), "{r:?}");
        assert!((r.fun() + 6.).abs() < 1e-7);
        let solver = QpSolver::default().with_method(QpMethod::PrimalDual);
        let mut workspace = solver.workspace(&q).unwrap().with_kkt_provider(qp_provider);
        let mut r = ProgrammingResult::empty();
        solver
            .solve_into(&q, &mut r, &mut workspace, ExecutionContext::sequential())
            .unwrap();
        assert!(r.is_success(), "{r:?}");
        assert!(r.x()[0].abs() < 1e-6);
        assert!((r.x()[1] - 2.).abs() < 1e-6);
    }
}
#[test]
fn qp_warm_start_reuses_feasible_primal_and_rejects_shape() {
    let p = QpProblem::<f64>::new(
        Matrix::from_vec(2, 2, vec![2., 0., 0., 2.]).unwrap(),
        vec![-2., -6.],
    )
    .unwrap()
    .with_constraints(
        LinearConstraints::new(
            Matrix::from_vec(1, 2, vec![1., 1.]).unwrap(),
            vec![2.],
            vec![Sense::LessEqual],
        )
        .unwrap(),
    )
    .unwrap();
    let r = QpSolver::default()
        .with_warm_start(scioxide_optimize::QpWarmStart::new(vec![0., 2.]).unwrap())
        .solve(&p)
        .unwrap();
    assert!(r.is_success(), "{r:?}");
    assert!(matches!(
        (QpSolver::default()
            .with_warm_start(scioxide_optimize::QpWarmStart::new(vec![0.]).unwrap())
            .solve(&p))
        .as_ref()
        .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
    ));
}
macro_rules! widths {
    ($name:ident,$s:ty,$exponent:expr) => {
        #[test]
        fn $name() {
            for exponent in [-$exponent, 0, $exponent] {
                let scale = (2.0 as $s).powi(exponent);
                let p = LpProblem::new(vec![-scale])
                    .unwrap()
                    .with_constraints(
                        LinearConstraints::new(
                            Matrix::from_vec(1, 1, vec![scale]).unwrap(),
                            vec![2. * scale],
                            vec![Sense::LessEqual],
                        )
                        .unwrap(),
                    )
                    .unwrap();
                let revised = SimplexSolver::default().solve(&p).unwrap();
                assert!(revised.is_success(), "{revised:?}");
                assert!((revised.x()[0] - 2.).abs() < 2e-4 as $s);
                let r = InteriorPointLp::default().solve(&p).unwrap();
                assert!(r.is_success(), "scale {scale}: {r:?}");
                assert!((r.x()[0] - 2.).abs() < 2e-4 as $s);
                let q = QpProblem::new(
                    Matrix::from_vec(1, 1, vec![2. as $s * scale]).unwrap(),
                    vec![-4. as $s * scale],
                )
                .unwrap()
                .with_bounds(Bounds::new(vec![(Some(0. as $s), Some(1. as $s))]).unwrap())
                .unwrap();
                let r = QpSolver::default()
                    .with_method(QpMethod::PrimalDual)
                    .solve(&q)
                    .unwrap();
                assert!(r.is_success(), "{r:?}");
                assert!((r.x()[0] - 1.).abs() < 2e-4 as $s);
            }
        }
    };
}
widths!(scaled_programming_f32, f32, 30);
widths!(scaled_programming_f64, f64, 300);

#[test]
fn borrowed_operator_preparation_preserves_free_variables_and_entry_limits() {
    let a = Matrix::from_vec(2, 2, vec![1.0_f64, 0., 0., 1.]).unwrap();
    let rhs = [1., 2.];
    let senses = [Sense::LessEqual; 2];
    let a_finite = a.view().try_finite().unwrap();
    let view = scioxide_optimize::LpVerificationProblem::new(&[-1., -2.])
        .unwrap()
        .with_constraints(&a_finite, &rhs, &senses)
        .unwrap();
    let p = LpProblem::from_view(&view, 2).unwrap();
    assert_eq!(p.bounds().unwrap().as_slice(), &[(None, None); 2]);
    assert!(matches!(
        (LpProblem::from_view(&view, 1))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::ResourceLimit {
            required: 2,
            limit: 1,
            ..
        })
    ));
    let result = SimplexSolver::default().solve(&p).unwrap();
    assert!(result.is_success());
    assert!((result.fun() + 5.).abs() < 1e-8);
}
#[test]
fn prepared_hsd_restarts_without_reconstructing_input() {
    let p = lp(
        vec![-1., -2.],
        vec![1., 1., 1., 0., 0., 1.],
        vec![4., 3., 2.],
        vec![Sense::LessEqual; 3],
    );
    let solver = InteriorPointLp::default();
    let mut workspace = solver.workspace(&p).unwrap();
    let mut a = ProgrammingResult::empty();
    let mut b = ProgrammingResult::empty();
    solver
        .solve_into(&p, &mut a, &mut workspace, ExecutionContext::sequential())
        .unwrap();
    solver
        .solve_into(&p, &mut b, &mut workspace, ExecutionContext::sequential())
        .unwrap();
    assert!(a.is_success() && b.is_success());
    assert_eq!(a.counts(), b.counts());
    assert_eq!(a.fun(), b.fun());
    let (tau, kappa) = workspace.embedding_scalars();
    assert!(tau > 0. && kappa > 0.);
}
#[test]
fn basis_provider_failure_does_not_publish_a_warm_basis() {
    struct Bad;
    impl scioxide_optimize::BasisProvider<f64> for Bad {
        fn factor(
            &mut self,
            _: scioxide_optimize::BasisMatrix<'_, f64>,
            _: scioxide_core::ExecutionContext<'_>,
        ) -> Result<(), SciOxideError> {
            Ok(())
        }
        fn solve_into(
            &mut self,
            _: &[f64],
            out: &mut [f64],
            _: bool,
            _: scioxide_core::ExecutionContext<'_>,
        ) -> Result<(), SciOxideError> {
            out.fill(0.);
            Ok(())
        }
    }
    let p = lp(vec![-1.], vec![1.], vec![1.], vec![Sense::LessEqual]);
    let solver = SimplexSolver::default();
    let mut workspace = solver
        .workspace(&p)
        .unwrap()
        .with_basis_provider(Box::new(Bad));
    let mut output = ProgrammingResult::empty();
    assert!(matches!(
        (solver.solve_into(
            &p,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential()
        ))
        .as_ref()
        .map_err(scioxide_core::SciOxideError::view),
        Err(scioxide_core::SciOxideErrorView::NumericalInstability { .. })
    ));
    assert!(workspace.warm_start().unwrap().basis_indices().is_empty());
    assert_eq!(output.termination(), Stop::NotStarted);
}

#[test]
fn semidefinite_active_set_uses_nullspace_recession_steps() {
    let q = QpProblem::<f64>::new(Matrix::from_vec(1, 1, vec![0.]).unwrap(), vec![-1.])
        .unwrap()
        .with_bounds(Bounds::new(vec![(Some(0.), None)]).unwrap())
        .unwrap();
    let unbounded = QpSolver::default()
        .with_method(QpMethod::ActiveSet)
        .solve(&q)
        .unwrap();
    assert_eq!(unbounded.termination(), Stop::Unbounded, "{unbounded:?}");
    let q = QpProblem::<f64>::new(Matrix::from_vec(1, 1, vec![0.]).unwrap(), vec![-1.])
        .unwrap()
        .with_bounds(Bounds::new(vec![(Some(0.), Some(2.))]).unwrap())
        .unwrap();
    let bounded = QpSolver::default()
        .with_method(QpMethod::ActiveSet)
        .solve(&q)
        .unwrap();
    assert!(bounded.is_success(), "{bounded:?}");
    assert_eq!(bounded.x().len(), 1);
    assert!((bounded.x()[0] - 2.0).abs() <= 1e-10);
}

#[test]
fn hsd_unconstrained_shortcut_also_requires_an_accepted_ray() {
    let p = LpProblem::new(vec![-1.0_f64]).unwrap();
    let r = InteriorPointLp::default().solve(&p).unwrap();
    assert_eq!(r.termination(), Stop::Unbounded);
    assert!(r.verification_report().unwrap().is_accepted());
}

#[test]
fn klee_minty_three_dimensional_vertex_is_verified() {
    let p = lp(
        vec![-4., -2., -1.],
        vec![1., 0., 0., 4., 1., 0., 8., 4., 1.],
        vec![5., 25., 125.],
        vec![Sense::LessEqual; 3],
    );
    // Eliminating x3 from the last row gives objective >= -125+4*x1+2*x2.
    for r in [
        SimplexSolver::default()
            .with_basis_update_limit(Budget::ONE)
            .solve(&p),
        InteriorPointLp::default().solve(&p),
    ] {
        let r = r.unwrap();
        assert!(r.is_success(), "{r:?}");
        assert!((r.fun() + 125.).abs() < 1e-5);
        assert!(r.x()[0].abs() < 1e-5 && r.x()[1].abs() < 1e-5 && (r.x()[2] - 125.).abs() < 1e-5);
    }
}

#[test]
fn scaling_never_silently_discards_an_objective_term() {
    let p = LpProblem::new(vec![1e300_f64, -1e-300]).unwrap();
    for result in [
        SimplexSolver::default().solve(&p),
        InteriorPointLp::default().solve(&p),
    ] {
        assert!(matches!(
            (result)
                .as_ref()
                .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Underflow,
                ..
            })
        ));
    }
}

#[test]
fn unconstrained_qp_strategies_share_the_rank_aware_direct_solve() {
    let p = QpProblem::<f64>::new(
        Matrix::from_vec(2, 2, vec![2., 0., 0., 2.]).unwrap(),
        vec![-2., -6.],
    )
    .unwrap();
    for method in [
        QpMethod::Automatic,
        QpMethod::ActiveSet,
        QpMethod::PrimalDual,
    ] {
        let r = QpSolver::default().with_method(method).solve(&p).unwrap();
        assert!(r.is_success());
        assert_eq!(r.x().len(), 2);
        assert!(
            r.x()
                .iter()
                .zip([1.0, 3.0])
                .all(|(x, expected)| (x - expected).abs() <= 1e-10)
        );
        assert!(matches!(
            (QpSolver::default()
                .with_method(method)
                .with_warm_start(scioxide_optimize::QpWarmStart::new(vec![0.]).unwrap())
                .solve(&p))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::DimensionMismatch { .. })
        ));
    }
}
#[test]
fn recession_evidence_uses_the_selected_symmetric_model() {
    let p = QpProblem::<f64>::new_with_symmetry(
        Matrix::from_vec(3, 3, vec![1., 1., 0., -1., 1., 0., 0., 0., 0.]).unwrap(),
        vec![-1., -1., -1.],
        scioxide_optimize::HessianSymmetry::SymmetricPart,
    )
    .unwrap();
    let r = QpSolver::default().solve(&p).unwrap();
    assert_eq!(r.termination(), Stop::Unbounded, "{r:?}");
}

fn bounded_equality(rhs: f64, objective: Vec<f64>) -> LpProblem<f64> {
    lp(objective, vec![1., 1., 1.], vec![rhs], vec![Sense::Equal])
        .with_bounds(
            Bounds::new(vec![
                (Some(0.), Some(1.)),
                (Some(0.), Some(2.)),
                (Some(0.), Some(5.)),
            ])
            .unwrap(),
        )
        .unwrap()
}

fn verified_point(result: &ProgrammingResult<f64>, expected: &[f64]) {
    assert!(result.is_success(), "{result:?}");
    assert!(result.verification_report().unwrap().is_accepted());
    for (&actual, &expected) in result.x().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-7, "{result:?}");
    }
}

#[test]
fn native_bounds_use_only_original_rows_and_refactor_both_lu_actions() {
    let problem = bounded_equality(2.5, vec![-4., -2., -1.]);
    let solver = SimplexSolver::default().with_basis_update_limit(Budget::ONE);
    let mut workspace = solver
        .workspace(&problem)
        .unwrap()
        .with_basis_provider(Box::new(
            scioxide_optimize::DenseBasisProvider::new(1).unwrap(),
        ));
    let mut result = ProgrammingResult::empty();
    solver
        .solve_into(
            &problem,
            &mut result,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    verified_point(&result, &[1., 1.5, 0.]);
}

#[test]
fn bounded_warm_starts_restore_upper_states_and_repair_both_bound_violations() {
    let problem = bounded_equality(2.5, vec![-4., -2., -1.]);
    let solver = SimplexSolver::default();
    let mut workspace = solver.workspace(&problem).unwrap();
    let mut result = ProgrammingResult::empty();
    solver
        .solve_into(
            &problem,
            &mut result,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let warm = workspace.warm_start().unwrap();
    for (rhs, expected) in [(0.5, [0.5, 0., 0.]), (4., [1., 2., 1.])] {
        let changed = bounded_equality(rhs, vec![-4., -2., -1.]);
        solver
            .resume_into(
                &changed,
                &warm,
                &mut result,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        verified_point(&result, &expected);
        assert!(workspace.warm_start().unwrap().reused_basis());
    }
    let changed = bounded_equality(2.5, vec![-1., -2., -4.]);
    solver
        .resume_into(
            &changed,
            &warm,
            &mut result,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    verified_point(&result, &[0., 0., 2.5]);
    assert!(workspace.warm_start().unwrap().reused_basis());
}

#[test]
fn native_box_bounds_include_shifted_upper_only_and_fixed_coordinates() {
    let p = LpProblem::new(vec![-1.0_f64, 1., -2., 3.])
        .unwrap()
        .with_bounds(
            Bounds::new(vec![
                (Some(-1.), Some(2.)),
                (Some(-3.), None),
                (None, Some(4.)),
                (Some(2.), Some(2.)),
            ])
            .unwrap(),
        )
        .unwrap();
    verified_point(
        &SimplexSolver::default().solve(&p).unwrap(),
        &[2., -3., 4., 2.],
    );
}

#[test]
fn default_budget_scales_and_more_than_one_thousand_bound_flips_succeed() {
    let small = LpProblem::new(vec![1.0_f64]).unwrap();
    assert_eq!(SimplexSolver::default().max_iterations(&small).get(), 1000);
    let n = 1003;
    let mut a = vec![0.; n];
    a[0] = 1.;
    let mut c = vec![-1.; n];
    c[0] = 0.;
    let p = lp(c, a, vec![0.], vec![Sense::Equal])
        .with_bounds(Bounds::new(vec![(Some(0.), Some(1.)); n]).unwrap())
        .unwrap();
    let solver = SimplexSolver::default();
    assert_eq!(solver.max_iterations(&p).get(), 10 * (n + 1));
    let result = solver.solve(&p).unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!(result.iterations() > 1000);
    assert!((result.fun() + (n - 1) as f64).abs() < 1e-7);
    assert!(result.verification_report().unwrap().is_accepted());
    let limited = solver.with_max_iterations(Budget::ONE).solve(&p).unwrap();
    assert_eq!(limited.termination(), Stop::IterationLimit);
    assert!(matches!(
        limited.require_success().unwrap_err().view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::IterationLimit,
            ..
        }
    ));
}

#[test]
fn appended_constraint_preserves_native_upper_states() {
    let p = bounded_equality(2.5, vec![-4., -2., -1.]);
    let solver = SimplexSolver::default();
    let mut workspace = solver.workspace(&p).unwrap();
    let mut result = ProgrammingResult::empty();
    solver
        .solve_into(
            &p,
            &mut result,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let warm = workspace.warm_start().unwrap();
    let child = lp(
        vec![-4., -2., -1.],
        vec![1., 1., 1., 1., 0., 0.],
        vec![2.5, 0.25],
        vec![Sense::Equal, Sense::LessEqual],
    )
    .with_bounds(p.bounds().unwrap().clone())
    .unwrap();
    solver
        .resume_into(
            &child,
            &warm,
            &mut result,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    verified_point(&result, &[0.25, 2., 0.25]);
    assert!(workspace.warm_start().unwrap().reused_basis());
}

#[test]
fn native_upper_bound_infeasibility_has_an_independent_witness() {
    let p = lp(vec![0.], vec![1.], vec![2.], vec![Sense::GreaterEqual])
        .with_bounds(Bounds::new(vec![(Some(0.), Some(1.))]).unwrap())
        .unwrap();
    let result = SimplexSolver::default().solve(&p).unwrap();
    assert_eq!(result.termination(), Stop::Infeasible, "{result:?}");
    assert!(result.verification_report().unwrap().is_accepted());
}

#[test]
fn uniform_linear_relations_match_explicit_senses() {
    let a = Matrix::from_rows(&[[1.0_f64, 2.0], [3.0, 4.0]]).unwrap();
    let rhs = vec![5.0, 6.0];
    for (actual, sense) in [
        (
            LinearConstraints::less_equal(a.clone(), rhs.clone()).unwrap(),
            Sense::LessEqual,
        ),
        (
            LinearConstraints::equal(a.clone(), rhs.clone()).unwrap(),
            Sense::Equal,
        ),
        (
            LinearConstraints::greater_equal(a.clone(), rhs.clone()).unwrap(),
            Sense::GreaterEqual,
        ),
    ] {
        let expected = LinearConstraints::new(a.clone(), rhs.clone(), vec![sense; 2]).unwrap();
        assert_eq!(actual.rhs(), expected.rhs());
        assert_eq!(actual.senses(), expected.senses());
    }
    assert!(LinearConstraints::equal(a.clone(), vec![1.0]).is_err());
    assert!(LinearConstraints::less_equal(a.clone(), vec![f64::NAN, 1.0]).is_err());
    assert!(LinearConstraints::greater_equal(a, vec![]).is_err());
}

#[test]
fn problem_solve_rejects_infeasibility_and_unboundedness() {
    let infeasible = lp(
        vec![1.0],
        vec![1.0, 1.0],
        vec![0.0, 1.0],
        vec![Sense::LessEqual, Sense::GreaterEqual],
    );
    let unbounded = LpProblem::new(vec![-1.0_f64]).unwrap();
    for problem in [infeasible, unbounded] {
        let error = problem.solve().unwrap_err();
        assert!(
            matches!(error.view(),scioxide_core::SciOxideErrorView::DomainError {argument,reason: scioxide_core::DomainReason::Undefined,..} if argument==scioxide_core::argument!("self"))
        );
    }
}
