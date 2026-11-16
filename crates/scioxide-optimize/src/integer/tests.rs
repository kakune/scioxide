//! Independent numerical and resource coverage for MILP search.
#![allow(clippy::unwrap_used)]
use super::*;
use crate::{LinearConstraintSense as Sense, LinearConstraints};
use alloc::format;
use alloc::vec;
use scioxide_core::{Bounds, ExecutionContext};
use scioxide_linalg::Matrix;
use scioxide_sparse::CsrMatrix;

fn fixture<S: RealScalar + NumericScalar<Real = S>>(
    cost: Vec<S>,
    rows: Vec<S>,
    rhs: Vec<S>,
    senses: Vec<Sense>,
    bounds: Vec<(Option<S>, Option<S>)>,
    integer: Vec<usize>,
    sparse: bool,
) -> MipProblem<S> {
    let n = cost.len();
    let m = rhs.len();
    let mut p = LpProblem::new(cost)
        .unwrap()
        .with_bounds(Bounds::new(bounds).unwrap())
        .unwrap();
    if m > 0 {
        let c = if sparse {
            let mut ptr = vec![0];
            let mut cols = Vec::new();
            let mut values = Vec::new();
            for row in rows.chunks(n) {
                for (j, &a) in row.iter().enumerate() {
                    if a != <S as scioxide_core::AlgebraicScalar>::ZERO {
                        cols.push(j);
                        values.push(a);
                    }
                }
                ptr.push(values.len());
            }
            LinearConstraints::new(
                CsrMatrix::from_raw(m, n, ptr, cols, values).unwrap(),
                rhs,
                senses,
            )
            .unwrap()
        } else {
            LinearConstraints::new(Matrix::from_vec(m, n, rows).unwrap(), rhs, senses).unwrap()
        };
        p = p.with_constraints(c).unwrap();
    }
    MipProblem::new(p, integer).unwrap()
}
fn plain() -> BranchAndBound<f64> {
    BranchAndBound::default()
        .with_presolve(false)
        .with_branching(MipBranching::MostFractional)
        .with_limits(MipLimits::default().with_cut_rounds(0).with_repair(false))
}
fn solve_state<R: MipRelaxationSolver<f64>>(
    solver: &BranchAndBound<f64>,
    p: &MipProblem<f64>,
    lp: R,
) -> (MipResult<f64>, MipSearchState<f64, R::Workspace>) {
    let mut workspace = BranchAndBoundWorkspace::new(lp);
    let mut r = MipResult::empty();
    solver
        .solve_into(p, &mut r, &mut workspace, ExecutionContext::sequential())
        .unwrap();
    (r, workspace.state)
}
fn knapsack(sparse: bool) -> MipProblem<f64> {
    fixture(
        vec![-3., -2., -4.],
        vec![2., 2., 3.],
        vec![4.5],
        vec![Sense::LessEqual],
        vec![(Some(0.), Some(1.)); 3],
        vec![0, 1, 2],
        sparse,
    )
}

#[test]
fn enumerated_binary_optima_all_policies_and_storage() {
    for sparse in [false, true] {
        for policy in [
            MipBranching::MostFractional,
            MipBranching::Pseudocost,
            MipBranching::Reliability,
        ] {
            for presolve in [false, true] {
                let p = knapsack(sparse);
                let mut exact = 0.0_f64;
                for bits in 0..8 {
                    let x = [
                        (bits & 1) as f64,
                        ((bits >> 1) & 1) as f64,
                        ((bits >> 2) & 1) as f64,
                    ];
                    if 2. * x[0] + 2. * x[1] + 3. * x[2] <= 4.5 {
                        exact = exact.min(-3. * x[0] - 2. * x[1] - 4. * x[2]);
                    }
                }
                let r = BranchAndBound::default()
                    .with_branching(policy)
                    .with_presolve(presolve)
                    .solve(&p)
                    .unwrap();
                assert!(
                    r.is_success(),
                    "{policy:?} {sparse} {presolve}: {} {:?}",
                    r.message(),
                    r.termination()
                );
                assert!((r.fun() - exact).abs() < 1e-9);
                assert!(r.has_incumbent());
                assert!(r.global_lower_bound().unwrap() <= exact + 1e-9);
                assert!(r.absolute_gap().unwrap() < 1e-8);
                assert_eq!(r.certified_optimality_gap(), None);
            }
        }
    }
}
#[test]
fn both_precisions_and_large_absolute_integrality() {
    for upper in [2.0_f32, 2.5, 2.125, 1.875] {
        let p = fixture(
            vec![-2.],
            vec![],
            vec![],
            vec![],
            vec![(Some(0.), Some(upper))],
            vec![0],
            false,
        );
        let r = BranchAndBound::default().solve(&p).unwrap();
        assert!(r.is_success());
        assert_eq!(r.x(), &[upper.floor()]);
    }
    for value in [1_000_000_000.4, -1_000_000_000.4, 0.5, -0.5] {
        assert!(validation::distance(value).unwrap() > 0.1);
    }
    assert_eq!(validation::distance(2_f64.powi(53)).unwrap(), 0.0);
}
#[test]
fn node_limits_keep_frontier_and_verified_bounds() {
    let p = knapsack(false);
    let solver = plain().with_max_nodes(Budget::ONE);
    let (r, s) = solve_state(&solver, &p, crate::SimplexSolver::default());
    assert_eq!(r.termination(), OptimizeTermination::NodeLimit);
    assert_eq!(r.nodes(), 1);
    assert_eq!(
        s.open_nodes().len() + usize::from(s.active_node().is_some()),
        2
    );
    assert!(r.global_lower_bound().unwrap() <= -5.);
    if r.has_incumbent() {
        assert!(r.fun() >= -5.);
    }
}
#[test]
fn presolve_reconstructs_fixed_continuous_and_integer_coordinates() {
    let p = fixture(
        vec![2., -1., 3.],
        vec![1., 1., 0.],
        vec![4.5],
        vec![Sense::LessEqual],
        vec![
            (Some(2.), Some(2.)),
            (Some(0.), Some(4.)),
            (Some(0.5), Some(0.5)),
        ],
        vec![0, 1],
        false,
    );
    let (r, s) = solve_state(
        &BranchAndBound::default(),
        &p,
        crate::SimplexSolver::default(),
    );
    assert!(r.is_success(), "{}", r.message());
    assert_eq!(r.x().len(), 3);
    assert!(
        r.x()
            .iter()
            .zip([2.0, 2.0, 0.5])
            .all(|(x, expected)| (x - expected).abs() <= 1e-10)
    );
    assert!((r.fun() - 3.5).abs() <= 1e-10);
    let map = s.presolve_map().unwrap();
    assert_eq!(map.active_variables(), &[1]);
    assert_eq!(map.objective_offset(), 5.5);
    assert_eq!(map.reconstruct(&[2.]).unwrap(), r.x());
}
#[test]
fn integer_bounds_and_empty_rows_detect_infeasibility() {
    let p = fixture(
        vec![1.],
        vec![],
        vec![],
        vec![],
        vec![(Some(0.25), Some(0.75))],
        vec![0],
        false,
    );
    let r = BranchAndBound::default().solve(&p).unwrap();
    assert_eq!(r.termination(), OptimizeTermination::Infeasible);
    assert!(!r.has_incumbent());
    let p = fixture(
        vec![1.],
        vec![0.],
        vec![-1.],
        vec![Sense::LessEqual],
        vec![(None, None)],
        vec![0],
        false,
    );
    assert_eq!(
        BranchAndBound::default().solve(&p).unwrap().termination(),
        OptimizeTermination::Infeasible
    );
}
#[test]
fn cuts_preserve_every_feasible_binary_assignment() {
    let p = knapsack(false);
    let m = model::Model::prepare(&p, false).unwrap();
    let cuts = cuts::separate(
        &m,
        &[1., 0., 1.],
        [].iter(),
        128,
        8192,
        crate::OptimalityTolerances::programming_default().feasibility(),
    )
    .unwrap();
    assert!(!cuts.is_empty());
    for bits in 0..8 {
        let x = [
            (bits & 1) as f64,
            ((bits >> 1) & 1) as f64,
            ((bits >> 2) & 1) as f64,
        ];
        if m.rows[0].value(&x).unwrap() <= 4.5 {
            for cut in &cuts {
                assert!(cut.value(&x).unwrap() <= cut.rhs);
            }
        }
    }
    let cover = model::Row {
        provenance: None,
        entries: vec![(0, 1.), (2, 1.)],
        rhs: 1.,
        sense: Sense::LessEqual,
    };
    assert!(
        cuts::verify(
            &m,
            &cover,
            &cuts::Proof::Cover {
                row: 0,
                variables: vec![0, 2]
            }
        )
        .unwrap()
    );
    let invalid = model::Row {
        provenance: None,
        entries: vec![(0, 1.)],
        rhs: 0.,
        sense: Sense::LessEqual,
    };
    assert!(
        !cuts::verify(
            &m,
            &invalid,
            &cuts::Proof::Cover {
                row: 0,
                variables: vec![0]
            }
        )
        .unwrap()
    );
}

#[test]
fn resource_stops_preserve_active_node() {
    let p = knapsack(false);
    for limits in [
        plain().limits().with_max_open_nodes(1),
        plain().limits().with_max_depth(0),
        plain().limits().with_max_relaxation_bytes(1),
    ] {
        let solver = plain().with_limits(limits);
        let (r, s) = solve_state(&solver, &p, crate::SimplexSolver::default());
        assert!(!r.is_success());
        assert!(matches!(
            r.termination(),
            OptimizeTermination::MemoryLimit | OptimizeTermination::DepthLimit
        ));
        assert!(s.active_node().is_some());
    }
    let r = plain()
        .with_limits(plain().limits().with_max_search_bytes(1))
        .solve(&p)
        .unwrap();
    assert_eq!(r.termination(), OptimizeTermination::MemoryLimit);
}
#[test]
fn lp_unbounded_does_not_establish_integer_unboundedness() {
    let p = fixture(
        vec![0., -1.],
        vec![1., 0.],
        vec![0.5],
        vec![Sense::Equal],
        vec![(None, None), (None, None)],
        vec![0],
        false,
    );
    let r = plain().solve(&p).unwrap();
    assert_eq!(r.termination(), OptimizeTermination::SuspectedUnbounded);
    assert!(!r.has_incumbent());
}

fn rational_mixed_cases<
    S: RealScalar + NumericScalar<Real = S> + scioxide_core::NumericScalar<Real = S>,
>() {
    use num::rational::Ratio as Q;
    for case in 0..32_i64 {
        let a = 1 + case % 3;
        let b = 1 + case % 2;
        let c = Q::new(3 + case % 14, 2);
        let d = Q::new(1 + case % 5, 2);
        let cx = -(1 + case % 4);
        let cy = if case % 2 == 0 { 1 } else { -1 };
        let mut exact = None;
        for x in 0..=3_i64 {
            let lo = Q::from_integer(0);
            let hi = Q::from_integer(4)
                .min((c - Q::from_integer(a * x)) / Q::from_integer(b))
                .min(d + Q::from_integer(x));
            if lo > hi {
                continue;
            }
            let y = if cy > 0 { lo } else { hi };
            let value = Q::from_integer(cx * x) + Q::from_integer(cy) * y;
            exact = Some(exact.map_or(value, |old: Q<i64>| old.min(value)));
        }
        let v = |x: f64| S::try_from_f64_exact(x).unwrap();
        let p = fixture(
            vec![v(cx as f64), v(cy as f64)],
            vec![v(a as f64), v(b as f64), v(-1.), v(1.)],
            vec![
                v(*c.numer() as f64 / *c.denom() as f64),
                v(*d.numer() as f64 / *d.denom() as f64),
            ],
            vec![Sense::LessEqual; 2],
            vec![(Some(v(0.)), Some(v(3.))), (Some(v(0.)), Some(v(4.)))],
            vec![0],
            case % 3 == 0,
        );
        let r = BranchAndBound::default().solve(&p).unwrap();
        assert!(
            r.is_success(),
            "case {case}: {} {:?}",
            r.message(),
            r.termination()
        );
        let q = exact.unwrap();
        let expected = S::try_from_f64_lossy(*q.numer() as f64 / *q.denom() as f64).unwrap();
        let tol = S::SQRT_EPSILON * v(4.);
        assert!(
            NumericScalar::abs(r.fun() - expected) <= tol,
            "case {case}: {:?} vs {:?}",
            r.fun(),
            expected
        );
        assert!(r.global_lower_bound().unwrap() <= expected + tol);
        assert!(
            validation::candidate(
                &p,
                r.x().to_vec(),
                false,
                crate::OptimalityTolerances::programming_default().feasibility(),
            )
            .unwrap()
            .is_some()
        );
    }
}
#[test]
fn rational_mixed_f32() {
    rational_mixed_cases::<f32>();
}
#[test]
fn rational_mixed_f64() {
    rational_mixed_cases::<f64>();
}

struct ColdSimplex;
impl MipRelaxationSolver<f64> for ColdSimplex {
    type Workspace = ();
    fn workspace(&self, _: &LpProblem<f64>) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn workspace_bytes(&self, _: &LpProblem<f64>) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn solve_into(
        &mut self,
        p: &LpProblem<f64>,
        _: &mut (),
        output: &mut crate::ProgrammingResult<f64>,
        i: Budget,
        _: usize,
        c: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        let session = crate::context::SolveSession::new(c, scioxide_core::argument!("context"));
        *output = crate::SimplexSolver::default()
            .with_max_iterations(i)
            .solve_observed(p, &session)?;
        Ok(())
    }
}
#[test]
fn warm_and_cold_search_agree_and_only_warm_reports_reuse() {
    let p = knapsack(false);
    let solver = plain();
    let (cold, _) = solve_state(&solver, &p, ColdSimplex);
    let (warm, _) = solve_state(&solver, &p, crate::SimplexSolver::default());
    assert!(cold.is_success() && warm.is_success());
    assert!((cold.fun() - warm.fun()).abs() < 1e-10);
    assert!(warm.statistics().basis_reuses() > 0);
    assert_eq!(cold.statistics().basis_reuses(), 0);
    let (again, _) = solve_state(&solver, &p, crate::SimplexSolver::default());
    assert_eq!(warm.x(), again.x());
    assert_eq!(warm.statistics(), again.statistics());
}

#[test]
fn configured_basis_update_capacity_is_admitted_before_preparation() {
    let p = knapsack(false);
    let provider = crate::SimplexSolver::default()
        .with_basis_update_limit(Budget::ONE.saturating_add(10_000_000));
    let (r, state) = solve_state(&plain(), &p, provider);
    assert_eq!(r.termination(), OptimizeTermination::MemoryLimit);
    assert!(!r.has_incumbent());
    assert_eq!(r.statistics().relaxations(), 0);
    assert!(state.active_node().is_some());
}
struct Dishonest(bool);
impl MipRelaxationSolver<f64> for Dishonest {
    type Workspace = ();
    fn workspace(&self, _: &LpProblem<f64>) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn workspace_bytes(&self, _: &LpProblem<f64>) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn solve_into(
        &mut self,
        p: &LpProblem<f64>,
        _: &mut (),
        output: &mut crate::ProgrammingResult<f64>,
        _: Budget,
        _: usize,
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        *output = {
            let evidence = self.0.then(|| {
                crate::OwnedProgrammingCertificate::Infeasibility(crate::OwnedDualCertificate::new(
                    vec![0.; p.constraints().map_or(0, |c| c.len())],
                    vec![0.; p.dimension()],
                    vec![0.; p.dimension()],
                ))
            });
            crate::ProgrammingResult::from_candidate(
                vec![0.; p.dimension()],
                0.,
                if self.0 {
                    OptimizeTermination::Infeasible
                } else {
                    OptimizeTermination::Converged
                },
                0,
                evidence,
            )
        }?;
        Ok(())
    }
}
#[test]
fn injected_status_and_fake_certificate_cannot_prune() {
    let p = knapsack(false);
    for infeasible in [false, true] {
        let (r, s) = solve_state(&plain(), &p, Dishonest(infeasible));
        assert!(!r.is_success());
        assert_eq!(r.global_lower_bound(), None);
        assert!(s.active_node().is_some());
        assert_eq!(
            r.termination(),
            if infeasible {
                OptimizeTermination::SuspectedInfeasible
            } else {
                OptimizeTermination::Stagnation
            }
        );
    }
}
#[test]
fn round_and_fix_repair_supplies_valid_partial_incumbent() {
    let p = fixture(
        vec![0., 1.],
        vec![1., -1.],
        vec![0.5],
        vec![Sense::LessEqual],
        vec![(Some(0.5), Some(2.)), (Some(0.), Some(2.))],
        vec![0],
        false,
    );
    let r = plain()
        .with_max_nodes(Budget::ONE)
        .with_limits(plain().limits().with_repair(true))
        .solve(&p)
        .unwrap();
    assert_eq!(r.termination(), OptimizeTermination::NodeLimit);
    assert!(r.has_incumbent());
    assert_eq!(r.x().len(), 2);
    assert!(
        r.x()
            .iter()
            .zip([1.0, 0.5])
            .all(|(x, expected)| (x - expected).abs() <= 1e-10)
    );
    assert!((r.fun() - 0.5).abs() <= 1e-10);
    assert!(r.global_lower_bound().unwrap() <= 0.5);
    assert_eq!(r.statistics().repairs(), 1);
}
#[test]
fn parent_iteration_evaluation_and_cancellation_stops_are_not_infeasibility() {
    use scioxide_core::{
        AttemptOutcome, BudgetKind, CancellationToken, SciOxideErrorView, SolveBudgets, SolveState,
    };
    let p = knapsack(false);
    let solver = plain();
    // Shared resources (evaluations, nodes) charge the parent; iteration
    // allowances stay local to each epoch, so a parent iteration
    // limit does not stop the child search.
    for kind in [
        BudgetKind::Evaluation,
        BudgetKind::Node,
        BudgetKind::Iteration,
    ] {
        let mut work = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(kind, Budget::ONE);
        let mut ctx = work
            .begin_solve(
                ExecutionContext::sequential(),
                crate::context::local_identity(
                    scioxide_core::algorithm_id!("optimize", "parent", 1),
                    0,
                    &(),
                ),
                &mut budgets,
            )
            .unwrap();
        let mut workspace = solver.workspace(&p).unwrap();
        let mut r = MipResult::empty();
        solver
            .solve_into(&p, &mut r, &mut workspace, &mut ctx)
            .unwrap();
        match kind {
            BudgetKind::Iteration => assert!(r.is_success(), "{}", r.message()),
            _ => {
                assert!(!r.is_success());
                assert!(matches!(
                    r.termination(),
                    OptimizeTermination::EvaluationLimit | OptimizeTermination::NodeLimit
                ));
                assert!(ctx.counts().nested().iterations() >= 1);
            }
        }
    }
    // A cancelled scope is rejected in preflight: no epoch, no output change.
    let token = CancellationToken::new();
    token.cancel();
    let mut workspace = solver.workspace(&p).unwrap();
    let mut r = MipResult::empty();
    let error = solver
        .solve_into(
            &p,
            &mut r,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&token),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ExecutionStopped { .. }
    ));
    assert_eq!(r.termination(), OptimizeTermination::NotStarted);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
}
#[test]
fn cover_boundary_and_integer_rounding_are_conservative() {
    let p = fixture(
        vec![-1., -1.],
        vec![2., 2.],
        vec![4.],
        vec![Sense::LessEqual],
        vec![(Some(0.), Some(1.)); 2],
        vec![0, 1],
        false,
    );
    let m = model::Model::prepare(&p, false).unwrap();
    let cut = model::Row {
        entries: vec![(0, 1.), (1, 1.)],
        rhs: 1.,
        sense: Sense::LessEqual,
        provenance: None,
    };
    assert!(
        !cuts::verify(
            &m,
            &cut,
            &cuts::Proof::Cover {
                row: 0,
                variables: vec![0, 1]
            }
        )
        .unwrap()
    );
    let p = fixture(
        vec![0., 0.],
        vec![-2., 1.],
        vec![-0.5],
        vec![Sense::GreaterEqual],
        vec![(Some(-2.), Some(2.)); 2],
        vec![0, 1],
        false,
    );
    let m = model::Model::prepare(&p, false).unwrap();
    let cut = model::Row {
        entries: vec![(0, 2.), (1, -1.)],
        rhs: 0.,
        sense: Sense::LessEqual,
        provenance: None,
    };
    assert!(cuts::verify(&m, &cut, &cuts::Proof::IntegerRow(0)).unwrap());
    for x in -2..=2 {
        for y in -2..=2 {
            if -2 * x + y >= 0 {
                assert!(cut.value(&[x as f64, y as f64]).unwrap() <= cut.rhs);
            }
        }
    }
}
#[test]
fn binary_cover_is_used_and_budget_zero_disables_cuts() {
    let p = fixture(
        vec![-1., -1.],
        vec![2., 2.],
        vec![3.],
        vec![Sense::LessEqual],
        vec![(Some(0.), Some(1.)); 2],
        vec![0, 1],
        false,
    );
    let r = BranchAndBound::default().solve(&p).unwrap();
    assert!(r.is_success());
    assert_eq!(r.fun(), -1.);
    assert!(r.statistics().cuts() > 0);
    let r = plain().solve(&p).unwrap();
    assert!(r.is_success());
    assert_eq!(r.statistics().cuts(), 0);
}

#[test]
fn numerical_gap_is_retained_when_tolerance_stops_search() {
    let p = fixture(
        vec![-1.],
        vec![],
        vec![],
        vec![],
        vec![(Some(0.), Some(2.25))],
        vec![0],
        false,
    );
    let solver = plain().with_relaxation_tolerances(
        crate::OptimalityTolerances::programming_default()
            .with_gap(scioxide_core::ResidualTolerance::new(0.5, 0.0).unwrap()),
    );
    let r = solver.solve(&p).unwrap();
    assert!(r.is_success());
    assert_eq!(r.x(), &[2.]);
    assert_eq!(r.absolute_gap(), Some(0.25));
    assert_eq!(r.global_lower_bound(), Some(-2.25));
    assert_eq!(r.certified_optimality_gap(), None);
}
#[test]
fn presolve_skips_inexact_substitution_and_reuses_state_as_new_epoch() {
    let p = fixture(
        vec![1., -1.],
        vec![0.1, 1.],
        vec![1.],
        vec![Sense::LessEqual],
        vec![(Some(0.1), Some(0.1)), (Some(0.), Some(2.))],
        vec![1],
        false,
    );
    let m = model::Model::prepare(&p, true).unwrap();
    assert_eq!(m.map.active_variables(), &[0, 1]);
    let solver = BranchAndBound::default();
    let mut workspace = solver.workspace(&p).unwrap();
    let mut r = MipResult::empty();
    solver
        .solve_into(&p, &mut r, &mut workspace, ExecutionContext::sequential())
        .unwrap();
    assert!(r.is_success());
    assert_eq!(r.x(), &[0.1, 0.]);
    assert!(workspace.state().incumbent().is_some());
    let p = knapsack(false);
    solver
        .solve_into(&p, &mut r, &mut workspace, ExecutionContext::sequential())
        .unwrap();
    assert!(r.is_success());
    assert_eq!(workspace.state().incumbent(), Some(r.x()));
}
#[test]
fn partial_frontiers_bound_the_enumerated_optimum_for_every_node_budget() {
    let p = knapsack(false);
    for nodes in 1..12 {
        let (r, s) = solve_state(
            &plain().with_max_nodes(Budget::try_from(nodes).unwrap()),
            &p,
            crate::SimplexSolver::default(),
        );
        if let Some(b) = r.global_lower_bound() {
            assert!(b <= -5. + 1e-9);
        }
        if r.has_incumbent() {
            assert!(r.fun() >= -5. - 1e-9);
            assert_eq!(
                r.absolute_gap().unwrap(),
                (r.fun() - r.global_lower_bound().unwrap()).max(0.)
            );
        }
        assert!(r.statistics().peak_search_bytes() <= plain().limits().max_search_bytes());
        if !r.is_success() {
            assert!(s.active_node().is_some() || !s.open_nodes().is_empty());
        }
    }
}
#[test]
fn original_rounding_validation_preserves_continuous_values_and_rejects_bad_rows() {
    let p = fixture(
        vec![2., 3.],
        vec![1., 0.],
        vec![0.875],
        vec![Sense::LessEqual],
        vec![(Some(0.), Some(2.)), (Some(0.), Some(2.))],
        vec![0],
        false,
    );
    let t = crate::OptimalityTolerances::programming_default().feasibility();
    assert!(
        validation::candidate(&p, vec![0.875, 0.375], true, t)
            .unwrap()
            .is_none()
    );
    let c = validation::candidate(&p, vec![0.125, 0.375], true, t)
        .unwrap()
        .unwrap();
    assert_eq!(c.x, vec![0., 0.375]);
    assert_eq!(c.fun, 1.125);
}
struct LocalProbeLimit;
impl MipRelaxationSolver<f64> for LocalProbeLimit {
    type Workspace = ();
    fn workspace(&self, _: &LpProblem<f64>) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn workspace_bytes(&self, _: &LpProblem<f64>) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn solve_into(
        &mut self,
        p: &LpProblem<f64>,
        w: &mut (),
        output: &mut crate::ProgrammingResult<f64>,
        i: Budget,
        bytes: usize,
        c: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        *output = {
            if i.get() == 1 {
                let _ = c.record_iteration()?;
                crate::ProgrammingResult::from_candidate(
                    vec![0.; p.dimension()],
                    0.,
                    OptimizeTermination::IterationLimit,
                    1,
                    None,
                )
            } else {
                {
                    let mut r = crate::ProgrammingResult::empty();
                    ColdSimplex.solve_into(p, w, &mut r, i, bytes, c)?;
                    Ok::<_, SciOxideError>(r)
                }
            }
        }?;
        Ok(())
    }
}
#[test]
fn local_probe_limit_keeps_search_running_and_is_counted() {
    let p = knapsack(false);
    let solver = plain()
        .with_branching(MipBranching::Reliability)
        .with_limits(plain().limits().with_strong_iterations(1));
    let (r, _) = solve_state(&solver, &p, LocalProbeLimit);
    assert!(r.is_success(), "{}", r.message());
    assert_eq!(r.fun(), -5.);
    assert!(r.statistics().probes() > 0);
    assert!(r.counts().nested().iterations() >= r.statistics().probes());
}

#[test]
fn borrowed_operator_preparation_and_injected_executor_match_owned() {
    use scioxide_core::SequentialExecutor;
    let matrix = Matrix::from_vec(1, 2, vec![2., 2.]).unwrap();
    let costs = [-1., -1.];
    let rhs = [3.];
    let senses = [Sense::LessEqual];
    let bounds = Bounds::new(vec![(Some(0.), Some(1.)); 2]).unwrap();
    let matrix_finite = matrix.view().try_finite().unwrap();
    let view = crate::LpVerificationProblem::new(&costs)
        .unwrap()
        .with_constraints(&matrix_finite, &rhs, &senses)
        .unwrap()
        .with_bounds(&bounds)
        .unwrap();
    assert!(LpProblem::from_view(&view, 1).is_err());
    let p = MipProblem::new(LpProblem::from_view(&view, 2).unwrap(), vec![0, 1]).unwrap();
    let solver = BranchAndBound::default();
    let expected = solver.solve(&p).unwrap();
    let executor = SequentialExecutor;
    let mut workspace = solver.workspace(&p).unwrap();
    let mut actual = MipResult::empty();
    solver
        .solve_into(
            &p,
            &mut actual,
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap();
    assert_eq!(actual.x(), expected.x());
    assert_eq!(actual.fun(), expected.fun());
    assert_eq!(actual.statistics(), expected.statistics());
}

#[test]
fn deadline_before_preparation_is_rejected_in_preflight() {
    struct Clock;
    impl scioxide_core::Clock for Clock {
        fn now(&self) -> Result<scioxide_core::MonotonicInstant, scioxide_core::CapabilityError> {
            Ok(scioxide_core::MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::from_secs(1),
            ))
        }
    }
    let clock = Clock;
    let clock_binding = scioxide_core::ClockBinding::new(&clock);
    let deadline = scioxide_core::Deadline::at(
        &clock_binding,
        scioxide_core::MonotonicInstant::from_duration_since_epoch(core::time::Duration::ZERO),
    );
    let p = knapsack(false);
    let solver = plain();
    let mut workspace = solver.workspace(&p).unwrap();
    let mut r = MipResult::empty();
    let error = solver
        .solve_into(
            &p,
            &mut r,
            &mut workspace,
            ExecutionContext::sequential()
                .with_deadline(deadline)
                .unwrap(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped {
            reason: scioxide_core::ExecutionStopReason::DeadlineReached
        }
    ));
    assert_eq!(r.termination(), OptimizeTermination::NotStarted);
    assert_eq!(r.nodes(), 0);
}
#[test]
fn valid_unbounded_integer_ray_and_all_fixed_presolve() {
    let p = fixture(
        vec![-1.],
        vec![],
        vec![],
        vec![],
        vec![(Some(0.), None)],
        vec![0],
        false,
    );
    let r = BranchAndBound::default().solve(&p).unwrap();
    assert_eq!(r.termination(), OptimizeTermination::Unbounded);
    assert!(r.has_incumbent());
    assert_eq!(r.absolute_gap(), None);
    let p = fixture(
        vec![-2., 3.],
        vec![],
        vec![],
        vec![],
        vec![(Some(0.), Some(2.25)), (Some(-1.), Some(2.))],
        vec![0],
        false,
    );
    let r = BranchAndBound::default().solve(&p).unwrap();
    assert!(r.is_success());
    assert_eq!(r.x(), &[2., -1.]);
    assert_eq!(r.fun(), -7.);
    assert_eq!(r.statistics().relaxations(), 0);
}
struct CancelChild<'a> {
    token: &'a scioxide_core::CancellationToken,
    calls: usize,
}
impl MipRelaxationSolver<f64> for CancelChild<'_> {
    type Workspace = ();
    fn workspace(&self, _: &LpProblem<f64>) -> Result<(), SciOxideError> {
        Ok(())
    }
    fn workspace_bytes(&self, _: &LpProblem<f64>) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn solve_into(
        &mut self,
        p: &LpProblem<f64>,
        w: &mut (),
        output: &mut crate::ProgrammingResult<f64>,
        i: Budget,
        b: usize,
        c: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        *output = {
            self.calls += 1;
            if self.calls == 2 {
                self.token.cancel();
            }
            {
                let mut r = crate::ProgrammingResult::empty();
                ColdSimplex.solve_into(p, w, &mut r, i, b, c)?;
                Ok::<_, SciOxideError>(r)
            }
        }?;
        Ok(())
    }
}
#[test]
fn cancellation_in_child_retains_incumbent_and_active_bound() {
    let p = fixture(
        vec![-2.],
        vec![],
        vec![],
        vec![],
        vec![(Some(0.), Some(2.125))],
        vec![0],
        false,
    );
    let solver = plain();
    let token = scioxide_core::CancellationToken::new();
    let mut workspace = BranchAndBoundWorkspace::new(CancelChild {
        token: &token,
        calls: 0,
    });
    let mut r = MipResult::empty();
    solver
        .solve_into(
            &p,
            &mut r,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&token),
        )
        .unwrap();
    assert_eq!(r.termination(), OptimizeTermination::Cancelled);
    assert!(r.has_incumbent());
    assert_eq!(r.fun(), -4.);
    assert_eq!(r.global_lower_bound(), Some(-4.25));
    assert!(workspace.state().active_node().is_some());
    assert_eq!(workspace.relaxation().calls, 2);
    assert_eq!(workspace.last_accepted().unwrap().point(), r.x());
}

#[test]
fn presolve_does_not_mistake_underflowed_product_residual_for_exactness() {
    fn check<S: RealScalar + NumericScalar<Real = S>>() {
        let one = <S as scioxide_core::AlgebraicScalar>::ONE;
        let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
        let x = one + S::EPSILON;
        let a = S::MIN_POSITIVE * x;
        // The exact product has an unrepresentable nonzero low part, although
        // FMA rounds the residual to zero. It must not authorize substitution.
        assert_eq!(a.mul_add(x, -(a * x)), zero);
        let p = fixture(
            vec![a, zero],
            vec![],
            vec![],
            vec![],
            vec![(Some(x), Some(x)), (Some(zero), Some(zero))],
            vec![1],
            false,
        );
        let m = model::Model::prepare(&p, true).unwrap();
        assert_eq!(m.map.active_variables(), &[0]);
    }
    check::<f32>();
    check::<f64>();
}

fn unbounded_knapsack_cases<
    S: RealScalar + NumericScalar<Real = S> + scioxide_core::NumericScalar<Real = S>,
>() {
    let cases: [(&[i64], &[i64], i64); 4] = [
        (&[3, 7, 11], &[5, 13, 19], 29),
        (&[2, 3, 5, 7], &[3, 5, 8, 11], 17),
        (&[6, 9, 13], &[7, 11, 16], 40),
        (&[5, 4, 3], &[10, 40, 30], 10),
    ];
    let v = |x: i64| S::try_from_f64_exact(x as f64).unwrap();
    for (weights, values, capacity) in cases {
        let n = weights.len();
        // Exact enumeration over the box implied by the capacity row.
        let mut exact = 0_i64;
        let mut x = vec![0_i64; n];
        loop {
            let weight: i64 = x.iter().zip(weights).map(|(x, w)| x * w).sum();
            if weight <= capacity {
                exact = exact.max(x.iter().zip(values).map(|(x, v)| x * v).sum());
            }
            let Some(j) = (0..n).find(|&j| (x[j] + 1) * weights[j] <= capacity) else {
                break;
            };
            x[j] += 1;
            x[..j].fill(0);
        }
        for sparse in [false, true] {
            for presolve in [false, true] {
                for policy in [
                    MipBranching::MostFractional,
                    MipBranching::Pseudocost,
                    MipBranching::Reliability,
                ] {
                    let p = fixture(
                        values.iter().map(|&c| v(-c)).collect(),
                        weights.iter().map(|&w| v(w)).collect(),
                        vec![v(capacity)],
                        vec![Sense::LessEqual],
                        vec![(Some(v(0)), None); n],
                        (0..n).collect(),
                        sparse,
                    );
                    let r = BranchAndBound::default()
                        .with_presolve(presolve)
                        .with_branching(policy)
                        .solve(&p)
                        .unwrap();
                    let label = format!("{weights:?} {sparse} {presolve} {policy:?}");
                    assert!(
                        r.is_success(),
                        "{label}: {} {:?}",
                        r.message(),
                        r.termination()
                    );
                    assert_eq!(r.fun(), v(-exact), "{label}");
                    assert!(r.global_lower_bound().unwrap() <= v(-exact), "{label}");
                    assert!(
                        validation::candidate(
                            &p,
                            r.x().to_vec(),
                            false,
                            crate::OptimalityTolerances::programming_default().feasibility(),
                        )
                        .unwrap()
                        .is_some(),
                        "{label}"
                    );
                }
            }
        }
    }
}
#[test]
fn unbounded_integer_knapsacks_match_enumeration_f32() {
    unbounded_knapsack_cases::<f32>();
}
#[test]
fn unbounded_integer_knapsacks_match_enumeration_f64() {
    unbounded_knapsack_cases::<f64>();
}

fn implied_bound_cases<
    S: RealScalar + NumericScalar<Real = S> + scioxide_core::NumericScalar<Real = S>,
>() {
    use num::BigRational as Q;
    let v = |x: f64| S::try_from_f64_exact(x).unwrap();
    let exact = |x: S| Q::from_float(x.try_to_f64_exact().unwrap()).unwrap();
    // `bound` is the tightest scalar on the outer side of `numerator / a`.
    let outward = |bound: S, numerator: f64, a: f64, upward: bool| {
        let q = Q::from_float(numerator).unwrap() / Q::from_float(a).unwrap();
        let (inner, toward) = if upward {
            (bound.next_after(S::NEG_INFINITY), exact(bound) >= q)
        } else {
            (bound.next_after(S::INFINITY), exact(bound) <= q)
        };
        assert!(toward, "{bound:?} is not outside {q}");
        assert!(
            if upward {
                exact(inner) < q
            } else {
                exact(inner) > q
            },
            "{bound:?} is not tight around {q}"
        );
    };
    let free = (None, None);
    let nonnegative = (Some(v(0.)), None);
    let model = |rows: Vec<f64>, rhs: Vec<f64>, senses, bounds, integer, presolve| {
        let n = rows.len() / rhs.len();
        let p = fixture(
            vec![v(1.); n],
            rows.into_iter().map(v).collect(),
            rhs.into_iter().map(v).collect(),
            senses,
            bounds,
            integer,
            false,
        );
        model::Model::prepare(&p, presolve).unwrap()
    };

    // `3x + 7y <= 29` and its sign-flipped `>=` form imply upper bounds only.
    for (rows, rhs, sense) in [
        (vec![3., 7.], 29., Sense::LessEqual),
        (vec![-3., -7.], -29., Sense::GreaterEqual),
    ] {
        let m = model(
            rows.clone(),
            vec![rhs],
            vec![sense],
            vec![nonnegative; 2],
            vec![0],
            false,
        );
        for (j, &(lower, upper)) in m.bounds.iter().enumerate() {
            assert_eq!(lower, Some(v(0.)));
            outward(upper.unwrap(), rhs, rows[j], true);
        }
    }
    // Equality rows imply both sides; a finite side is never replaced.
    let m = model(
        vec![1., -3.],
        vec![1.],
        vec![Sense::Equal],
        vec![(Some(v(0.)), Some(v(5.))), free],
        vec![0],
        false,
    );
    assert_eq!(m.bounds[0], (Some(v(0.)), Some(v(5.))));
    let (lower, upper) = m.bounds[1];
    outward(lower.unwrap(), -1., 3., false);
    outward(upper.unwrap(), 4., 3., true);
    // `x <= y` needs the second pass, after `3y <= 10` bounds `y`.
    let m = model(
        vec![1., -1., 0., 3.],
        vec![0., 10.],
        vec![Sense::LessEqual; 2],
        vec![nonnegative; 2],
        vec![0],
        false,
    );
    outward(m.bounds[1].1.unwrap(), 10., 3., true);
    outward(
        m.bounds[0].1.unwrap(),
        exact_f64(m.bounds[1].1.unwrap()),
        1.,
        true,
    );
    // Unrepresentable quotients and rows with two unbounded terms add nothing.
    let m = model(
        vec![S::MIN_POSITIVE.into_f64(), 0., 0., 0., 1., 1.],
        vec![S::MAX.into_f64(), 1.],
        vec![Sense::LessEqual; 2],
        vec![nonnegative, free, free],
        vec![0],
        false,
    );
    assert_eq!(m.bounds, [nonnegative, free, free]);
    // Presolve rounds implied integer bounds; without it they stay outward.
    for presolve in [false, true] {
        let m = model(
            vec![3.],
            vec![29.],
            vec![Sense::LessEqual],
            vec![nonnegative],
            vec![0],
            presolve,
        );
        if presolve {
            assert_eq!(m.bounds[0].1, Some(v(9.)));
        } else {
            outward(m.bounds[0].1.unwrap(), 29., 3., true);
        }
        assert!(!m.infeasible);
    }
}
fn exact_f64<S: RealScalar>(x: S) -> f64 {
    x.try_to_f64_exact().unwrap()
}
#[test]
fn implied_bounds_are_outward_and_fill_only_missing_sides_f32() {
    implied_bound_cases::<f32>();
}
#[test]
fn implied_bounds_are_outward_and_fill_only_missing_sides_f64() {
    implied_bound_cases::<f64>();
}
#[test]
fn contradictory_implied_bounds_prove_infeasibility() {
    for presolve in [false, true] {
        let p = fixture(
            vec![1., 1.],
            vec![1., 1.],
            vec![-1.],
            vec![Sense::LessEqual],
            vec![(Some(0.), None); 2],
            vec![0],
            false,
        );
        let r = BranchAndBound::default()
            .with_presolve(presolve)
            .solve(&p)
            .unwrap();
        assert_eq!(
            r.termination(),
            OptimizeTermination::Infeasible,
            "{presolve}"
        );
        assert_eq!(r.nodes(), 0, "{presolve}");
    }
}
