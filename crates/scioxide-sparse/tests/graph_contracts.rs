//! Public graph domains, resource guarantees, and execution parity.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{ExecutionContext, RealScalar, SciOxideError, SciOxideErrorKind};
use scioxide_sparse::*;
fn matrix<S: RealScalar>(n: usize, edges: &[(usize, usize, f64)]) -> CooMatrix<S> {
    let mut b = SparseBuilder::new(n, n);
    for &(u, v, w) in edges {
        b.push(u, v, S::try_from_f64_exact(w).unwrap()).unwrap();
    }
    b.finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
        .unwrap()
}
fn plan<S: RealScalar>(
    n: usize,
    edges: &[(usize, usize, f64)],
    direction: GraphDirection,
) -> GraphPlan<S> {
    GraphPlan::new(matrix(n, edges).view(), direction).unwrap()
}
fn paths<S: RealScalar>(outcome: PathOutcome<GraphPathResult<S>>) -> GraphPathResult<S> {
    let PathOutcome::Paths(p) = outcome else {
        panic!("unexpected cycle")
    };
    p
}
// Equal reachability, and distances within a few roundings of their magnitude.
fn assert_close(actual: &[Option<f64>], expected: &[Option<f64>]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        match (a, e) {
            (Some(a), Some(e)) => assert!((a - e).abs() <= 4. * f64::EPSILON * e.abs().max(1.)),
            _ => assert_eq!(a, e),
        }
    }
}
#[test]
fn empty_domains_duplicates_and_undirected_minimum() {
    let empty = plan::<f64>(0, &[], GraphDirection::Undirected);
    for method in [GraphPathMethod::Johnson, GraphPathMethod::FloydWarshall] {
        assert!(
            paths(empty.shortest_paths(method).unwrap())
                .as_view()
                .distances()
                .is_empty()
        );
    }
    assert!(empty.breadth_first(0).is_err());
    assert_eq!(empty.strong_components().unwrap().count(), 0);
    assert_eq!(empty.minimum_spanning_forest().unwrap().total_weight(), 0.0);
    let g = plan::<f64>(
        3,
        &[(0, 1, 4.0), (1, 0, 2.0), (1, 2, 7.0), (1, 2, -7.0)],
        GraphDirection::Undirected,
    );
    assert_eq!(g.edge_count(), 4);
    for method in [
        GraphPathMethod::Dijkstra { source: 0 },
        GraphPathMethod::BellmanFord { source: 0 },
    ] {
        assert_eq!(
            paths(g.shortest_paths(method).unwrap())
                .as_view()
                .distances(),
            &[Some(0.0), Some(2.0), Some(2.0)]
        );
    }
    assert_eq!(g.minimum_spanning_forest().unwrap().total_weight(), 2.0);
    let rectangular = SparseBuilder::<f32>::new(0, 2)
        .finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
        .unwrap();
    assert!(GraphPlan::new(rectangular.view(), GraphDirection::Directed).is_err());
    let mut bad = matrix::<f64>(1, &[(0, 0, 1.0)]).to_csr().unwrap();
    assert_eq!(
        bad.set_value(0, f64::NAN).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    GraphPlan::new(bad.view(), GraphDirection::Directed).unwrap();
    let g = plan::<f64>(1, &[(0, 0, -1.0)], GraphDirection::Directed);
    for method in [
        GraphPathMethod::BellmanFord { source: 0 },
        GraphPathMethod::Johnson,
        GraphPathMethod::FloydWarshall,
    ] {
        let PathOutcome::NegativeCycle(cycle) = g.shortest_paths(method).unwrap() else {
            panic!("missing negative loop")
        };
        assert_eq!(cycle.vertices(), &[0, 0]);
    }
    assert!(
        g.shortest_paths(GraphPathMethod::Dijkstra { source: 0 })
            .is_err()
    );
    assert!(g.minimum_spanning_forest().is_err());
}
fn formats<S: RealScalar>() {
    let coo = matrix::<S>(4, &[(0, 1, 0.0), (1, 2, 2.0), (2, 3, 3.0)]);
    let csr = coo.to_csr().unwrap();
    let csc = coo.to_csc().unwrap();
    let bsr = BsrMatrix::from_sparse(coo.view(), 1, 1).unwrap();
    let dia = DiaMatrix::from_sparse(coo.view()).unwrap();
    for view in [coo.view(), csr.view(), csc.view(), bsr.view(), dia.view()] {
        let g = GraphPlan::new(view, GraphDirection::Directed).unwrap();
        let p = paths(
            g.shortest_paths(GraphPathMethod::Dijkstra { source: 0 })
                .unwrap(),
        );
        assert_eq!(
            p.as_view().distances()[3],
            Some(S::try_from_f64_exact(5.0).unwrap())
        );
        assert_eq!(p.as_view().path(0, 3).unwrap(), Some(vec![0, 1, 2, 3]));
    }
    // BSR expansion is structural: block-internal zeros become graph edges.
    let expanded = BsrMatrix::from_sparse(coo.view(), 2, 2).unwrap();
    let g = GraphPlan::new(expanded.view(), GraphDirection::Directed).unwrap();
    assert!(g.edge_count() > coo.nnz());
}
#[test]
fn all_sparse_formats_and_complex_structural_views() {
    formats::<f32>();
    formats::<f64>();
    use scioxide_core::Complex;
    let mut b = SparseBuilder::new(2, 2);
    b.push(0, 1, Complex::new(0.0_f32, 0.0)).unwrap();
    let csr = b
        .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let g = GraphPlan::from_pattern(csr.structural_pattern(), GraphDirection::Directed).unwrap();
    assert_eq!(g.breadth_first(0).unwrap().order(), &[0, 1]);
    let mut b = SparseBuilder::new(2, 2);
    b.push(0, 1, Complex::new(0.0_f64, 1.0)).unwrap();
    let csr = b
        .finish_csr(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
        .unwrap();
    assert_eq!(
        GraphPlan::from_pattern(csr.structural_pattern(), GraphDirection::Directed)
            .unwrap()
            .vertex_count(),
        2
    );
}

// Finite weights make every non-finite intermediate an overflow, never `NonFinite`.
fn overflowed<T>(result: Result<T, SciOxideError>) -> bool {
    matches!(
        result.err().as_ref().map(SciOxideError::view),
        Some(scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        })
    )
}
#[test]
fn arithmetic_range_and_subnormal_edges() {
    for (weight, direction) in [
        (f64::MAX, GraphDirection::Directed),
        (-f64::MAX, GraphDirection::Directed),
        (f64::MAX, GraphDirection::Undirected),
    ] {
        let g = plan::<f64>(3, &[(0, 1, weight), (1, 2, weight)], direction);
        for method in [
            GraphPathMethod::Dijkstra { source: 0 },
            GraphPathMethod::BellmanFord { source: 0 },
            GraphPathMethod::Johnson,
            GraphPathMethod::FloydWarshall,
        ] {
            let result = g.shortest_paths(method);
            if weight < 0. && matches!(method, GraphPathMethod::Dijkstra { .. }) {
                let kind = result.err().map(|error| error.kind());
                assert_eq!(kind, Some(scioxide_core::SciOxideErrorKind::InvalidInput));
            } else {
                assert!(overflowed(result), "{method:?} {weight} {direction:?}");
            }
        }
        if direction == GraphDirection::Undirected {
            assert!(overflowed(g.minimum_spanning_forest()));
        }
    }
    let g = plan::<f64>(
        3,
        &[(0, 1, f64::from_bits(1)), (1, 2, f64::from_bits(1))],
        GraphDirection::Directed,
    );
    assert_eq!(
        paths(
            g.shortest_paths(GraphPathMethod::Dijkstra { source: 0 })
                .unwrap()
        )
        .as_view()
        .distances()[2],
        Some(f64::from_bits(2))
    );
    let g = plan::<f32>(
        3,
        &[
            (0, 1, f32::from_bits(1) as f64),
            (1, 2, f32::from_bits(1) as f64),
        ],
        GraphDirection::Directed,
    );
    assert_eq!(
        paths(
            g.shortest_paths(GraphPathMethod::Dijkstra { source: 0 })
                .unwrap()
        )
        .as_view()
        .distances()[2],
        Some(f32::from_bits(2))
    );
}

#[test]
fn long_iterative_traversal_and_mid_execution_cancellation() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    use scioxide_core::Cancellation;
    struct StopAfter(AtomicUsize);
    impl Cancellation for StopAfter {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_sub(1, Ordering::Relaxed) <= 1
        }
    }
    let n = 20_000;
    let mut builder = SparseBuilder::new(n, n);
    for u in 0..n - 1 {
        builder.push(u, u + 1, 1.0_f64).unwrap();
    }
    let matrix = builder
        .finish_coo(DuplicatePolicy::Sum, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let g = GraphPlan::new(matrix.view(), GraphDirection::Directed).unwrap();
    let c = ExecutionContext::sequential();
    assert_eq!(g.depth_first(0).unwrap().order().len(), n);
    assert_eq!(g.strong_components().unwrap().count(), n);
    let method = GraphPathMethod::Dijkstra { source: 0 };
    let options = GraphExecutionOptions::new();
    let mut workspace = g
        .workspace(GraphWorkspaceRequest::new(method, options))
        .unwrap();
    let token = StopAfter(AtomicUsize::new(10));
    let error = g
        .shortest_paths_scoped(method, options, &mut workspace, c.with_cancellation(&token))
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    assert!(matches!(
        g.shortest_paths_scoped(method, options, &mut workspace, c)
            .unwrap(),
        PathOutcome::Paths(_)
    ));
}
#[cfg(feature = "std")]
#[test]
fn f32_executor_paths_and_cycle_witness() {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionPolicy, ThreadExecutor};
    let workers = NonZeroUsize::new(3).unwrap();
    let pool = ThreadExecutor::new(workers);
    let options =
        GraphExecutionOptions::new().with_execution_policy(ExecutionPolicy::Parallel(workers));
    let g = plan::<f32>(
        8,
        &[
            (0, 1, 0.25),
            (1, 2, -0.5),
            (2, 3, 0.125),
            (3, 4, 1.0),
            (4, 5, 0.5),
            (5, 6, 0.25),
        ],
        GraphDirection::Directed,
    );
    let mut w = g
        .workspace(GraphWorkspaceRequest::new(
            GraphPathMethod::Johnson,
            options,
        ))
        .unwrap();
    for method in [GraphPathMethod::Johnson, GraphPathMethod::FloydWarshall] {
        let expected = paths(g.shortest_paths(method).unwrap());
        let PathOutcome::Paths(actual) = g
            .shortest_paths_scoped(method, options, &mut w, ExecutionContext::new(&pool))
            .unwrap()
        else {
            panic!("unexpected cycle")
        };
        assert_eq!(actual.distances(), expected.as_view().distances());
        assert_eq!(actual.path(0, 6).unwrap(), Some(vec![0, 1, 2, 3, 4, 5, 6]));
    }
    let cycle = plan::<f32>(8, &[(0, 1, -0.25), (1, 0, 0.125)], GraphDirection::Directed);
    assert!(matches!(
        cycle
            .shortest_paths_scoped(
                GraphPathMethod::Johnson,
                options,
                &mut w,
                ExecutionContext::new(&pool)
            )
            .unwrap(),
        PathOutcome::NegativeCycle(_)
    ));
}

#[test]
fn one_shot_connected_components_are_weak_components_of_the_directed_pattern() {
    // 0 → 1 ← 2 and 3 → 4 through a stored zero; vertex 5 is isolated. No
    // vertex reaches every other one, so the components are weak, not strong.
    let a = CsrMatrix::from_triplets(6, 6, &[0, 2, 3], &[1, 1, 4], &[1.0_f64, 1., 0.]).unwrap();
    let pattern = a.structural_pattern();
    let components = connected_components(pattern).unwrap();
    let weak = GraphPlan::from_pattern(pattern, GraphDirection::Directed)
        .unwrap()
        .weak_components()
        .unwrap();
    assert_eq!(
        (components.labels(), components.count()),
        (weak.labels(), weak.count())
    );
    assert_eq!(components.labels(), &[0, 0, 0, 1, 1, 2]);
    let empty = CsrMatrix::<f64>::from_triplets(0, 0, &[], &[], &[]).unwrap();
    assert_eq!(
        connected_components(empty.structural_pattern())
            .unwrap()
            .count(),
        0
    );
    let wide = CsrMatrix::from_triplets(2, 3, &[0], &[2], &[1.0_f64]).unwrap();
    assert_eq!(
        connected_components(wide.structural_pattern())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
}

#[test]
fn one_shot_shortest_paths_choose_dijkstra_or_bellman_ford() {
    // Nonnegative weights with a stored zero; 3 → 0 is one-way, so the
    // directed distance to 3 is 3.5, not 0.
    let nonnegative = CsrMatrix::from_triplets(
        4,
        4,
        &[0, 0, 1, 2, 3],
        &[1, 2, 2, 3, 0],
        &[1.0_f64, 5., 2., 0.5, 0.],
    )
    .unwrap();
    // A negative weight without a negative cycle.
    let negative =
        CsrMatrix::from_triplets(4, 4, &[0, 0, 2, 1], &[1, 2, 1, 3], &[4.0_f64, 1., -2., 1.])
            .unwrap();
    for (a, method, expected) in [
        (
            &nonnegative,
            GraphPathMethod::Dijkstra { source: 0 },
            [Some(0.), Some(1.), Some(3.), Some(3.5)],
        ),
        (
            &negative,
            GraphPathMethod::BellmanFord { source: 0 },
            [Some(0.), Some(-1.), Some(1.), Some(0.)],
        ),
    ] {
        let graph = GraphPlan::new(a.view(), GraphDirection::Directed).unwrap();
        let plan = paths(graph.shortest_paths(method).unwrap());
        let one_shot = shortest_paths(a.view(), 0).unwrap();
        let (actual, plan) = (one_shot.as_view(), plan.as_view());
        assert_close(actual.distances(), plan.distances());
        assert_close(actual.distances(), &expected);
        assert_eq!(actual.predecessors(), plan.predecessors());
        assert_eq!(actual.source(), Some(0));
        assert_eq!(actual.path(0, 3).unwrap(), plan.path(0, 3).unwrap());
    }
    // Dijkstra's method alone rejects the negative weight.
    let graph = GraphPlan::new(negative.view(), GraphDirection::Directed).unwrap();
    assert_eq!(
        graph
            .shortest_paths(GraphPathMethod::Dijkstra { source: 0 })
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    let kind = |result: Result<GraphPathResult<f64>, SciOxideError>| result.unwrap_err().kind();
    assert_eq!(
        kind(shortest_paths(nonnegative.view(), 4)),
        SciOxideErrorKind::OutOfBounds
    );
    let wide = CsrMatrix::from_triplets(2, 3, &[0], &[2], &[1.0_f64]).unwrap();
    assert_eq!(
        kind(shortest_paths(wide.view(), 0)),
        SciOxideErrorKind::DimensionMismatch
    );
}

#[test]
fn one_shot_shortest_paths_reject_a_reachable_negative_cycle() {
    // 0 → 1 (−2) and 1 → 0 (1) close a cycle of weight −1. Vertex 2 reaches it;
    // vertex 3 has no outgoing edge.
    let a = CsrMatrix::from_triplets(4, 4, &[0, 1, 2, 0], &[1, 0, 0, 3], &[-2.0_f64, 1., 1., 1.])
        .unwrap();
    let graph = GraphPlan::new(a.view(), GraphDirection::Directed).unwrap();
    for source in [0, 2] {
        let error = shortest_paths(a.view(), source).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::DomainError { argument, .. }
                if argument == scioxide_core::argument!("matrix")
        ));
        assert!(matches!(
            graph
                .shortest_paths(GraphPathMethod::BellmanFord { source })
                .unwrap(),
            PathOutcome::NegativeCycle(_)
        ));
    }
    // A negative cycle that the source cannot reach leaves the distances defined.
    let p = shortest_paths(a.view(), 3).unwrap();
    assert_close(p.as_view().distances(), &[None, None, None, Some(0.)]);
}

#[test]
fn one_shot_laplacian_is_the_unnormalized_out_degree_plan_form() {
    // Weights that are not exact binary fractions, a self-loop (2, 2) that is
    // left out, and a stored zero (3, 1).
    let a = CsrMatrix::from_triplets(
        4,
        4,
        &[0, 0, 1, 2, 2, 3],
        &[1, 2, 2, 0, 2, 1],
        &[0.1_f64, 0.2, 0.7, 0.3, 9., 0.],
    )
    .unwrap();
    let l = laplacian(a.view()).unwrap();
    let plan = GraphPlan::new(a.view(), GraphDirection::Directed)
        .unwrap()
        .laplacian::<f64>(LaplacianDegree::Out, LaplacianNormalization::Unnormalized)
        .unwrap();
    let expected = plan.matrix();
    assert_eq!((l.rows(), l.cols()), (4, 4));
    assert_eq!(l.row_ptr(), expected.row_ptr());
    assert_eq!(l.col_indices(), expected.col_indices());
    for (actual, expected) in l.values().iter().zip(expected.values()) {
        assert!((actual - expected).abs() <= 1e-14);
    }
    assert!((l.get(0, 0).unwrap() - 0.3).abs() <= 1e-14);
    assert!((l.get(2, 2).unwrap() - 0.3).abs() <= 1e-14);
    // L 1 = 0 for out-degrees: every row sums to zero.
    for bounds in l.row_ptr().windows(2) {
        let sum: f64 = l.values()[bounds[0]..bounds[1]].iter().sum();
        assert!(sum.abs() <= 1e-14, "{sum}");
    }
    let negative = CsrMatrix::from_triplets(2, 2, &[0], &[1], &[-1.0_f64]).unwrap();
    assert_eq!(
        laplacian(negative.view()).unwrap_err().kind(),
        SciOxideErrorKind::DomainError
    );
    let wide = CsrMatrix::from_triplets(2, 3, &[0], &[2], &[1.0_f64]).unwrap();
    assert_eq!(
        laplacian(wide.view()).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
}

fn cancelling_forest<S: RealScalar>() {
    let mut builder = SparseBuilder::new(4, 4);
    for (u, weight) in [-S::MAX, -S::MAX, S::MAX].into_iter().enumerate() {
        builder.push(u, u + 1, weight).unwrap();
    }
    let matrix = builder
        .finish_csr(DuplicatePolicy::Reject, ExplicitZeroPolicy::Preserve)
        .unwrap();
    let graph = GraphPlan::new(matrix.view(), GraphDirection::Undirected).unwrap();
    let forest = graph.minimum_spanning_forest().unwrap();
    assert_eq!(forest.edges().len(), 3);
    assert!((forest.total_weight() / S::MAX + S::ONE).abs() <= S::EPSILON);
}
#[test]
fn cancelling_forest_f32() {
    cancelling_forest::<f32>();
}
#[test]
fn cancelling_forest_f64() {
    cancelling_forest::<f64>();
}
