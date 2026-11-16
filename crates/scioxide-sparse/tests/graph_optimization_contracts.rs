//! Public ownership, allocation, budget, scalar, and execution contracts for graph optimization.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::*;
use scioxide_sparse::*;

const OPTIONS: GraphExecutionOptions = GraphExecutionOptions::new();
fn matrix() -> CsrMatrix<f64> {
    let mut b = SparseBuilder::new(3, 3);
    for (u, v, w) in [
        (0, 0, 1.0),
        (0, 1, 0.0),
        (1, 1, 2.0),
        (1, 2, 1.0),
        (2, 0, 3.0),
    ] {
        b.push(u, v, w).unwrap();
    }
    b.finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap()
}
fn flow() -> MaxFlowPlan {
    MaxFlowPlan::new(
        3,
        &[
            CapacityEdge {
                source: 0,
                target: 1,
                capacity: 4,
            },
            CapacityEdge {
                source: 1,
                target: 2,
                capacity: 3,
            },
        ],
    )
    .unwrap()
}
fn certified<S: RealScalar>(
    plan: &AssignmentPlan,
    costs: &[S],
    objective: AssignmentObjective,
    view: AssignmentView<'_, S>,
) {
    let residuals = plan.verify(costs, objective, view).unwrap();
    let tolerance =
        S::EPSILON * S::try_from_f64_exact(128.).unwrap() * view.objective.abs().max(S::ONE);
    assert!(residuals.dual_violation <= tolerance);
    assert!(residuals.complementary_slackness <= tolerance);
    assert!(residuals.objective_gap <= tolerance);
}
#[test]
fn immutable_dense_structure_accepts_changing_costs_and_objectives() {
    let plan = AssignmentPlan::from_dense(2).unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    for (costs, objective, expected) in [
        (
            [4., 1., 2., 5.],
            AssignmentObjective::Minimize,
            [Some(1), Some(0)],
        ),
        (
            [4., 1., 2., 5.],
            AssignmentObjective::Maximize,
            [Some(0), Some(1)],
        ),
        (
            [-3., 1., 2., -4.],
            AssignmentObjective::Minimize,
            [Some(0), Some(1)],
        ),
    ] {
        let AssignmentOutcome::Full(view) = plan
            .solve_scoped(
                &costs,
                objective,
                OPTIONS,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap()
        else {
            panic!("complete pattern");
        };
        assert_eq!(view.left, expected);
        certified(&plan, &costs, objective, view);
    }
    for costs in [&[f64::NAN, 1., 2., 5.][..], &[1., 2., 3.][..]] {
        assert!(
            plan.solve_scoped(
                costs,
                AssignmentObjective::Minimize,
                OPTIONS,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert!(workspace.last_accepted().is_none());
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::PreflightRejected
        );
    }
    assert_eq!(
        workspace
            .check(&plan.workspace_requirements(()).unwrap())
            .unwrap(),
        Capacity::Sufficient
    );
}

#[test]
fn sparse_formats_and_numeric_boundaries() {
    let csr = matrix();
    let coo = csr.to_coo().unwrap();
    let csc = csr.to_csc().unwrap();
    let bsr = BsrMatrix::from_sparse(csr.view(), 1, 1).unwrap();
    let dia = DiaMatrix::from_sparse(csr.view()).unwrap();
    for source in [csr.view(), coo.view(), csc.view(), bsr.view(), dia.view()] {
        let canonical = source.to_csr(ExplicitZeroPolicy::Preserve).unwrap();
        let plan = AssignmentPlan::new(canonical.structural_pattern()).unwrap();
        for objective in [AssignmentObjective::Minimize, AssignmentObjective::Maximize] {
            let AssignmentOutcome::Full(result) =
                plan.solve(canonical.values(), objective).unwrap()
            else {
                panic!("full fixture");
            };
            certified(&plan, canonical.values(), objective, result.as_view());
        }
    }
    let mut complex = SparseBuilder::<Complex<f32>>::new(2, 3);
    complex.push(0, 2, Complex::new(0., 1.)).unwrap();
    let complex = complex
        .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    assert_eq!(
        maximum_bipartite_matching(complex.structural_pattern())
            .unwrap()
            .as_view()
            .cardinality(),
        1
    );
    let dense = AssignmentPlan::from_dense(2).unwrap();
    assert!(matches!(
        dense
            .solve(&[f64::NAN, 1., 2., 3.], AssignmentObjective::Minimize)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::NonFinite
    ));
    let mut b = SparseBuilder::new(2, 2);
    b.push(0, 0, f64::MAX).unwrap();
    b.push(1, 1, f64::MAX).unwrap();
    let m = b
        .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let plan = AssignmentPlan::new(m.structural_pattern()).unwrap();
    // Finite costs whose selected sum overflows are numerical instability.
    let overflow = |error: SciOxideError| {
        matches!(
            error.view(),
            SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        )
    };
    let min = AssignmentObjective::Minimize;
    assert!(overflow(plan.solve(m.values(), min).unwrap_err()));
    let AssignmentOutcome::Full(result) = plan.solve(&[1., 2.], min).unwrap() else {
        panic!("full");
    };
    let view = result.as_view();
    let potentials = [-f64::MAX; 2];
    let candidate = AssignmentView {
        small_potentials: &potentials,
        ..view
    };
    assert!(overflow(
        plan.verify(m.values(), min, candidate).unwrap_err()
    ));
    let tiny = [f64::from_bits(1); 2];
    let AssignmentOutcome::Full(result) = plan.solve(&tiny, AssignmentObjective::Minimize).unwrap()
    else {
        panic!("full");
    };
    assert_eq!(result.as_view().objective, f64::from_bits(2));
    certified(
        &plan,
        &tiny,
        AssignmentObjective::Minimize,
        result.as_view(),
    );
}

#[test]
fn mid_execution_cancellation_reinitializes_all_families() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    struct StopAfter(AtomicUsize);
    impl Cancellation for StopAfter {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_sub(1, Ordering::Relaxed) <= 1
        }
    }
    let matrix = matrix();
    let pattern = matrix.structural_pattern();
    let assignment = AssignmentPlan::new(pattern).unwrap();
    let flow = flow();
    let mut mw = MatchingWorkspace::workspace(pattern).unwrap();
    let mut aw = assignment.workspace(()).unwrap();
    let mut fw = flow.workspace(()).unwrap();
    for algorithm in 0..3 {
        let token = StopAfter(AtomicUsize::new(3));
        let c = ExecutionContext::sequential().with_cancellation(&token);
        let result = match algorithm {
            0 => maximum_bipartite_matching_scoped(pattern, OPTIONS, &mut mw, c).map(|_| ()),
            1 => assignment
                .solve_scoped(
                    matrix.values(),
                    AssignmentObjective::Minimize,
                    OPTIONS,
                    &mut aw,
                    c,
                )
                .map(|_| ()),
            _ => flow
                .maximum_flow_scoped(0, 2, OPTIONS, &mut fw, c)
                .map(|_| ()),
        };
        assert_eq!(
            result.unwrap_err().kind(),
            SciOxideErrorKind::ExecutionStopped
        );
    }
    let c = ExecutionContext::sequential();
    assert!(
        maximum_bipartite_matching_scoped(pattern, OPTIONS, &mut mw, c)
            .unwrap()
            .verify(pattern)
    );
    let AssignmentOutcome::Full(view) = assignment
        .solve_scoped(
            matrix.values(),
            AssignmentObjective::Minimize,
            OPTIONS,
            &mut aw,
            c,
        )
        .unwrap()
    else {
        panic!("full");
    };
    certified(
        &assignment,
        matrix.values(),
        AssignmentObjective::Minimize,
        view,
    );
    assert_eq!(
        flow.maximum_flow_scoped(0, 2, OPTIONS, &mut fw, c)
            .unwrap()
            .value,
        3
    );
}

#[test]
fn long_sparse_augmenting_chain_preserves_dual_feasibility() {
    let n = 128;
    let mut builder = SparseBuilder::<f64>::new(n, n);
    for row in 0..n - 1 {
        builder.push(row, row, 0.).unwrap();
        builder.push(row, row + 1, 1.).unwrap();
    }
    builder.push(n - 1, 0, 0.).unwrap();
    let matrix = builder
        .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let plan = AssignmentPlan::new(matrix.structural_pattern()).unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    for objective in [AssignmentObjective::Minimize, AssignmentObjective::Maximize] {
        let costs: Vec<_> = matrix
            .values()
            .iter()
            .map(|&cost| {
                if objective == AssignmentObjective::Maximize {
                    -cost
                } else {
                    cost
                }
            })
            .collect();
        let AssignmentOutcome::Full(view) = plan
            .solve_scoped(
                &costs,
                objective,
                OPTIONS,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap()
        else {
            panic!("full chain");
        };
        assert!((view.objective.abs() - (n - 1) as f64).abs() < 1e-12);
        assert_eq!(view.left[n - 1], Some(0));
        certified(&plan, &costs, objective, view);
    }
}
// Deterministic xorshift64* entries of a `rows` x `cols` cost pattern: each row
// stores its diagonal (when present) plus distinct random columns, `per_row`
// entries in all, with costs uniform in [1, 1001).
fn random_entries(rows: usize, cols: usize, per_row: usize, seed: u64) -> Vec<(usize, usize, f64)> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut entries = Vec::with_capacity(rows * per_row);
    let mut columns = Vec::with_capacity(per_row);
    for row in 0..rows {
        columns.clear();
        columns.extend((row < cols).then_some(row));
        while columns.len() < per_row {
            let column = (next() % cols as u64) as usize;
            if !columns.contains(&column) {
                columns.push(column);
            }
        }
        for &column in &columns {
            let unit = (next() >> 11) as f64 / (1_u64 << 53) as f64;
            entries.push((row, column, 1. + 1000. * unit));
        }
    }
    entries
}
fn csr(
    rows: usize,
    cols: usize,
    entries: impl Iterator<Item = (usize, usize, f64)>,
) -> CsrMatrix<f64> {
    let mut builder = SparseBuilder::new(rows, cols);
    for (row, column, cost) in entries {
        builder.push(row, column, cost).unwrap();
    }
    builder
        .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap()
}
#[test]
fn rounded_random_costs_keep_duals_within_the_rounding_envelope() {
    // Before row duals were re-derived from matched edges and the reduced-cost
    // tolerance tracked accumulated dual drift, these instances failed with a
    // spurious `NumericalInstability` after many augmentations.
    let n = 1000;
    for seed in [336, 522] {
        let matrix = csr(n, n, random_entries(n, n, 10, seed).into_iter());
        let plan = AssignmentPlan::new(matrix.structural_pattern()).unwrap();
        let negated: Vec<_> = matrix.values().iter().map(|&cost| -cost).collect();
        let mut minimum = plan.workspace(()).unwrap();
        let mut maximum = plan.workspace(()).unwrap();
        let AssignmentOutcome::Full(min) = plan
            .solve_scoped(
                matrix.values(),
                AssignmentObjective::Minimize,
                OPTIONS,
                &mut minimum,
                ExecutionContext::sequential(),
            )
            .unwrap()
        else {
            panic!("the diagonal is a full assignment");
        };
        certified(&plan, matrix.values(), AssignmentObjective::Minimize, min);
        let AssignmentOutcome::Full(max) = plan
            .solve_scoped(
                &negated,
                AssignmentObjective::Maximize,
                OPTIONS,
                &mut maximum,
                ExecutionContext::sequential(),
            )
            .unwrap()
        else {
            panic!("the diagonal is a full assignment");
        };
        certified(&plan, &negated, AssignmentObjective::Maximize, max);
        // Negation is exact, so both objectives run identical arithmetic.
        assert_eq!(max.objective, -min.objective);
        assert_eq!(max.left, min.left);
        // Selected reduced costs stay within one rounding of zero, up to the
        // two roundings of their recomputation.
        let residuals = plan
            .verify(matrix.values(), AssignmentObjective::Minimize, min)
            .unwrap();
        let scale = min
            .small_potentials
            .iter()
            .chain(min.large_potentials)
            .fold(0_f64, |scale, dual| scale.max(dual.abs()));
        assert!(residuals.complementary_slackness <= 4. * f64::EPSILON * scale);
    }
}
#[test]
fn rounded_random_rectangular_costs_agree_in_both_orientations() {
    let (rows, cols) = (1050, 1000);
    let entries = random_entries(rows, cols, 10, 7);
    let tall = csr(rows, cols, entries.iter().copied());
    let wide = csr(cols, rows, entries.iter().map(|&(r, c, w)| (c, r, w)));
    let objectives = [&tall, &wide].map(|matrix| {
        let plan = AssignmentPlan::new(matrix.structural_pattern()).unwrap();
        let AssignmentOutcome::Full(result) = plan
            .solve(matrix.values(), AssignmentObjective::Minimize)
            .unwrap()
        else {
            panic!("every column has a diagonal entry");
        };
        let view = result.as_view();
        certified(&plan, matrix.values(), AssignmentObjective::Minimize, view);
        assert_eq!(view.left.iter().flatten().count(), cols);
        view.objective
    });
    let tolerance = 128. * f64::EPSILON * objectives[0].abs();
    assert!((objectives[0] - objectives[1]).abs() <= tolerance);
}
fn rounded_dense_oracle<S: RealScalar>() {
    fn walk(costs: &[f64], row: usize, used: usize, total: f64, maximize: bool) -> f64 {
        if row == 6 {
            return total;
        }
        (0..6)
            .filter(|&column| used & (1 << column) == 0)
            .map(|column| {
                walk(
                    costs,
                    row + 1,
                    used | (1 << column),
                    total + costs[row * 6 + column],
                    maximize,
                )
            })
            .reduce(|a, b| if maximize { a.max(b) } else { a.min(b) })
            .unwrap()
    }
    let plan = AssignmentPlan::from_dense(6).unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    for seed in 0..20 {
        let costs: Vec<S> = (0..36)
            .map(|i| {
                S::try_from_f64_lossy((((i * 97 + seed * 43) % 113) as f64 - 56.) / 7.).unwrap()
            })
            .collect();
        let oracle: Vec<_> = costs.iter().map(|value| value.into_f64()).collect();
        for objective in [AssignmentObjective::Minimize, AssignmentObjective::Maximize] {
            let AssignmentOutcome::Full(view) = plan
                .solve_scoped(
                    &costs,
                    objective,
                    OPTIONS,
                    &mut workspace,
                    ExecutionContext::sequential(),
                )
                .unwrap()
            else {
                panic!("complete pattern");
            };
            let expected = walk(
                &oracle,
                0,
                0,
                0.,
                objective == AssignmentObjective::Maximize,
            );
            assert!(
                (view.objective.into_f64() - expected).abs()
                    <= 128. * S::EPSILON.into_f64() * expected.abs().max(1.)
            );
            certified(&plan, &costs, objective, view);
        }
    }
}
#[test]
fn rounded_dense_costs_match_independent_permutations() {
    rounded_dense_oracle::<f32>();
    rounded_dense_oracle::<f64>();
}

#[test]
fn matching_workspaces_are_sized_by_the_pattern_and_owned_forms_match_scoped() {
    let square = matrix();
    // Row 0 may take column 0 or 1; row 1 only column 0.
    let wide = CsrMatrix::from_triplets(2, 3, &[0, 0, 1], &[0, 1, 0], &[1.0_f64; 3]).unwrap();
    for pattern in [square.structural_pattern(), wide.structural_pattern()] {
        let requirements = MatchingWorkspace::workspace_requirements(pattern).unwrap();
        let mut workspace = MatchingWorkspace::workspace(pattern).unwrap();
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            Capacity::Sufficient
        );
        assert_eq!(
            MatchingWorkspace::new().check(&requirements).unwrap(),
            Capacity::NeedsGrowth
        );
        let owned = maximum_bipartite_matching_with_options(pattern, OPTIONS).unwrap();
        let default = maximum_bipartite_matching(pattern).unwrap();
        let scoped = maximum_bipartite_matching_scoped(
            pattern,
            OPTIONS,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        for owned in [owned.as_view(), default.as_view()] {
            assert_eq!((owned.left, owned.right), (scoped.left, scoped.right));
            assert_eq!(owned.left_cover, scoped.left_cover);
            assert_eq!(owned.right_cover, scoped.right_cover);
        }
        assert!(scoped.verify(pattern));
    }
    // The owned form with options honors their iteration limit.
    let limited = OPTIONS
        .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE));
    let error =
        maximum_bipartite_matching_with_options(square.structural_pattern(), limited).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::IterationLimit,
            ..
        }
    ));
}

#[test]
fn plan_less_matching_of_a_borrowed_pattern_reproduces_hopcroft_karp() {
    // Row 0 may take column 0 or 1; row 1 only column 0.
    let wide = CsrMatrix::from_triplets(2, 3, &[0, 0, 1], &[0, 1, 0], &[1.0_f64; 3]).unwrap();
    let matching = maximum_bipartite_matching(wide.structural_pattern()).unwrap();
    let view = matching.as_view();
    assert_eq!(view.left, &[Some(1), Some(0)]);
    assert_eq!(view.right, &[Some(1), Some(0), None]);
    assert_eq!(
        (view.left_cover, view.right_cover),
        (&[true, true][..], &[false, false, false][..])
    );
    assert!(view.verify(wide.structural_pattern()));
    // The result owns its storage and outlives the matrix whose pattern it read.
    let matching = {
        let square = matrix();
        maximum_bipartite_matching(square.structural_pattern()).unwrap()
    };
    let view = matching.as_view();
    assert_eq!(view.left, &[Some(1), Some(2), Some(0)]);
    assert_eq!(view.right, &[Some(2), Some(0), Some(1)]);
    assert_eq!(view.cardinality(), 3);
    assert!(view.verify(matrix().structural_pattern()));
    // A matching certifies only the pattern it was computed for.
    assert!(!view.verify(wide.structural_pattern()));
}

#[test]
fn one_shot_maximum_flow_matches_the_plan() {
    let arc = |source, target, capacity| CapacityEdge {
        source,
        target,
        capacity,
    };
    // A self-loop and a zero-capacity edge carry no flow.
    let edges = [
        arc(0, 1, 3),
        arc(0, 2, 2),
        arc(1, 2, 1),
        arc(1, 3, 2),
        arc(2, 3, 3),
        arc(3, 3, 9),
        arc(2, 1, 0),
    ];
    let plan = MaxFlowPlan::new(4, &edges).unwrap();
    assert_eq!((plan.vertex_count(), plan.edges()), (4, &edges[..]));
    let expected = plan.maximum_flow(0, 3).unwrap();
    let actual = maximum_flow(4, &edges, 0, 3).unwrap();
    let (actual, expected) = (actual.as_view(), expected.as_view());
    assert_eq!((actual.source, actual.sink), (0, 3));
    assert_eq!((actual.value, expected.value), (5, 5));
    assert_eq!(actual.edge_flows, expected.edge_flows);
    assert_eq!(actual.source_side, expected.source_side);
    assert_eq!((actual.edge_flows[5], actual.edge_flows[6]), (0, 0));
    assert!(actual.verify(&plan).unwrap());
    let kind = |result: Result<MaximumFlow, SciOxideError>| result.unwrap_err().kind();
    assert_eq!(
        kind(maximum_flow(4, &edges, 1, 1)),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        kind(maximum_flow(4, &edges, 0, 4)),
        SciOxideErrorKind::OutOfBounds
    );
    assert_eq!(
        kind(maximum_flow(3, &edges, 0, 2)),
        SciOxideErrorKind::OutOfBounds
    );
}

#[test]
fn one_shot_minimum_cost_assignment_matches_the_plan() {
    let square = matrix();
    // Costs [[4, 1, 7], [2, 5, ·]]: pairs (0, 1) and (1, 0) cost 1 + 2.
    let wide = CsrMatrix::from_triplets(
        2,
        3,
        &[0, 0, 0, 1, 1],
        &[0, 1, 2, 0, 1],
        &[4.0_f64, 1., 7., 2., 5.],
    )
    .unwrap();
    for (costs, objective) in [(&square, 4.), (&wide, 3.)] {
        let plan = AssignmentPlan::new(costs.structural_pattern()).unwrap();
        let AssignmentOutcome::Full(expected) = plan
            .solve(costs.values(), AssignmentObjective::Minimize)
            .unwrap()
        else {
            panic!("full fixture");
        };
        let actual = minimum_cost_assignment(costs).unwrap();
        let (actual, expected) = (actual.as_view(), expected.as_view());
        assert_eq!((actual.left, actual.right), (expected.left, expected.right));
        let tolerance = 128. * f64::EPSILON * objective;
        assert!((actual.objective - expected.objective).abs() <= tolerance);
        assert!((actual.objective - objective).abs() <= tolerance);
        for (potentials, reference) in [
            (actual.small_potentials, expected.small_potentials),
            (actual.large_potentials, expected.large_potentials),
        ] {
            assert_eq!(potentials.len(), reference.len());
            for (a, b) in potentials.iter().zip(reference) {
                assert!((a - b).abs() <= tolerance);
            }
        }
        certified(&plan, costs.values(), AssignmentObjective::Minimize, actual);
    }
}

#[test]
fn infeasible_patterns_are_a_plan_outcome_and_a_one_shot_domain_error() {
    // Both rows can only take column 0; a row without entries.
    let crowded = CsrMatrix::from_triplets(2, 2, &[0, 1], &[0, 0], &[1.0_f64, 2.]).unwrap();
    let empty_row = CsrMatrix::from_triplets(2, 2, &[0, 0], &[0, 1], &[1.0_f64, 2.]).unwrap();
    for costs in [&crowded, &empty_row] {
        let plan = AssignmentPlan::new(costs.structural_pattern()).unwrap();
        assert!(matches!(
            plan.solve(costs.values(), AssignmentObjective::Minimize)
                .unwrap(),
            AssignmentOutcome::Infeasible
        ));
        let error = minimum_cost_assignment(costs).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::DomainError { argument, .. } if argument == argument!("costs")
        ));
    }
}
