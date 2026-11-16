//! Requirements, atomic publication and executor behavior for the canonical APIs.
#![allow(clippy::unwrap_used)]
#[path = "support/search_cases.rs"]
mod search_cases;
use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicUsize, Ordering};
use scioxide_core::PartitionFn;
use scioxide_core::{
    CancellationToken, Capacity, ExecutionContext, ExecutionPlan, ExecutionPolicy, Executor,
    NdView, SciOxideError, SciOxideErrorKind, SequentialExecutor, StridedLayout,
};
use scioxide_spatial::*;
use search_cases::{QueryCase, run_batch};

fn points<'a>(data: &'a [f64], shape: &'a [usize; 2]) -> PointSet<'a, f64> {
    PointSet::new(NdView::new(data, StridedLayout::new(shape, &[1, 1]).unwrap()).unwrap()).unwrap()
}
fn workers() -> NonZeroUsize {
    NonZeroUsize::new(4).unwrap()
}

struct InlineExecutor(NonZeroUsize);
impl Executor for InlineExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        self.0
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        SequentialExecutor.execute(context, plan, operation)
    }
}

struct CountingExecutor<E> {
    inner: E,
    calls: AtomicUsize,
    partitions: AtomicUsize,
    minimum_workers: AtomicUsize,
    maximum_workers: AtomicUsize,
}
impl<E> CountingExecutor<E> {
    fn new(inner: E) -> Self {
        Self {
            inner,
            calls: AtomicUsize::new(0),
            partitions: AtomicUsize::new(0),
            minimum_workers: AtomicUsize::new(usize::MAX),
            maximum_workers: AtomicUsize::new(0),
        }
    }
    fn reset(&self) {
        self.calls.store(0, Ordering::Relaxed);
        self.partitions.store(0, Ordering::Relaxed);
        self.minimum_workers.store(usize::MAX, Ordering::Relaxed);
        self.maximum_workers.store(0, Ordering::Relaxed);
    }
    fn assert_dispatches(&self, passes: usize, workers: usize) {
        assert_eq!(self.calls.load(Ordering::Relaxed), passes);
        assert_eq!(self.partitions.load(Ordering::Relaxed), passes * workers);
        if passes != 0 {
            assert_eq!(self.minimum_workers.load(Ordering::Relaxed), workers);
            assert_eq!(self.maximum_workers.load(Ordering::Relaxed), workers);
        }
    }
}
impl<E: Executor> Executor for CountingExecutor<E> {
    fn worker_budget(&self) -> NonZeroUsize {
        self.inner.worker_budget()
    }
    fn auto_min_work_per_worker(&self) -> NonZeroUsize {
        self.inner.auto_min_work_per_worker()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.minimum_workers
            .fetch_min(plan.worker_count(), Ordering::Relaxed);
        self.maximum_workers
            .fetch_max(plan.worker_count(), Ordering::Relaxed);
        self.inner.execute(context, plan, &|partition, (), child| {
            self.partitions.fetch_add(1, Ordering::Relaxed);
            operation(partition, (), child)
        })
    }
}

#[test]
fn batch_parallelism_survives_scalar_and_batch_reservations() {
    check_reservation_history(InlineExecutor(workers()));
}

#[cfg(feature = "std")]
#[test]
fn thread_batch_parallelism_survives_scalar_and_batch_reservations() {
    check_reservation_history(scioxide_core::ThreadExecutor::new(workers()));
}

fn check_reservation_history(inner: impl Executor) {
    let data: Vec<_> = (0..4096).map(f64::from).collect();
    let shape = [data.len(), 1];
    let points = points(&data, &shape);
    let tree = KdTree::new(&points).unwrap();
    let options = SearchOptions::new().with_execution_policy(ExecutionPolicy::Parallel(workers()));
    let request =
        SearchWorkspaceRequest::new(8, options).with_batch(data.len(), data.len() * 8, workers());
    let requirements = tree.workspace_requirements(request).unwrap();
    let scalar_requirements = tree
        .workspace_requirements(SearchWorkspaceRequest::new(8, options))
        .unwrap();
    let executor = CountingExecutor::new(inner);
    for (kind, passes) in [
        (QueryCase::Nearest { k: 8 }, 1),
        (QueryCase::Radius { radius: 0.0 }, 2),
    ] {
        let mut workspace = tree.workspace(request).unwrap();
        let mut expected = vec![sentinel(); data.len() * 8];
        let mut expected_counts = vec![99; data.len()];
        executor.reset();
        let expected_total = run_batch(
            &tree,
            &points,
            kind,
            options,
            &mut expected,
            &mut expected_counts,
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap();
        executor.assert_dispatches(passes, 4);
        for reserved_workers in [None, Some(1), Some(2), Some(4)] {
            workspace.try_reserve(&scalar_requirements).unwrap();
            if let Some(reserved_workers) = reserved_workers {
                workspace
                    .try_reserve(
                        &tree
                            .workspace_requirements(
                                SearchWorkspaceRequest::new(8, options).with_batch(
                                    data.len(),
                                    data.len() * 8,
                                    NonZeroUsize::new(reserved_workers).unwrap(),
                                ),
                            )
                            .unwrap(),
                    )
                    .unwrap();
            }
            let mut single = [sentinel(); 8];
            tree.nearest_into(
                &[0.0],
                8,
                options,
                &mut single,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert_eq!(
                workspace.check(&requirements).unwrap(),
                Capacity::Sufficient
            );
            let mut output = vec![sentinel(); expected.len()];
            let mut counts = vec![99; data.len()];
            executor.reset();
            assert_eq!(
                run_batch(
                    &tree,
                    &points,
                    kind,
                    options,
                    &mut output,
                    &mut counts,
                    &mut workspace,
                    ExecutionContext::new(&executor)
                )
                .unwrap(),
                expected_total
            );
            executor.assert_dispatches(passes, 4);
            assert_eq!(output, expected);
            assert_eq!(counts, expected_counts);
        }
    }
}

#[test]
fn insufficient_worker_capacity_fails_before_dispatch_and_can_be_reserved() {
    let data: Vec<_> = (0..300).map(f64::from).collect();
    let shape = [data.len(), 1];
    let points = points(&data, &shape);
    let tree = KdTree::new(&points).unwrap();
    let options = SearchOptions::new().with_execution_policy(ExecutionPolicy::Parallel(workers()));
    let request =
        SearchWorkspaceRequest::new(8, options).with_batch(data.len(), data.len() * 8, workers());
    let requirements = tree.workspace_requirements(request).unwrap();
    let executor = CountingExecutor::new(InlineExecutor(workers()));
    for (kind, passes) in [
        (QueryCase::Nearest { k: 8 }, 1),
        (QueryCase::Radius { radius: 0.0 }, 2),
    ] {
        let mut workspace = tree
            .workspace(SearchWorkspaceRequest::new(8, options).with_batch(
                data.len(),
                data.len() * 8,
                NonZeroUsize::new(2).unwrap(),
            ))
            .unwrap();
        assert_eq!(
            workspace.check(&requirements).unwrap(),
            Capacity::NeedsGrowth
        );
        let mut output = vec![sentinel(); data.len() * 8];
        let mut counts = vec![99; data.len()];
        executor.reset();
        let error = run_batch(
            &tree,
            &points,
            kind,
            options,
            &mut output,
            &mut counts,
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
        executor.assert_dispatches(0, 0);
        assert!(output.iter().all(|n| *n == sentinel()));
        assert!(counts.iter().all(|n| *n == 99));
        workspace.try_reserve(&requirements).unwrap();
        run_batch(
            &tree,
            &points,
            kind,
            options,
            &mut output,
            &mut counts,
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap();
        executor.assert_dispatches(passes, 4);
    }
}

#[test]
fn batch_worker_selection_obeys_policy_context_query_count_and_cap() {
    let data: Vec<_> = (0..300).map(f64::from).collect();
    let shape = [data.len(), 1];
    let points = points(&data, &shape);
    let tree = KdTree::new(&points).unwrap();
    for (policy, budget, queries, expected_workers) in [
        (ExecutionPolicy::Sequential, 4, 300, 1),
        (ExecutionPolicy::Parallel(workers()), 2, 300, 2),
        (ExecutionPolicy::Parallel(workers()), 4, 2, 2),
        (ExecutionPolicy::Auto, 4, 255, 1),
        (ExecutionPolicy::Auto, 4, 256, 4),
        (
            ExecutionPolicy::Parallel(NonZeroUsize::new(64).unwrap()),
            64,
            64,
            32,
        ),
        (ExecutionPolicy::Parallel(workers()), 4, 0, 0),
    ] {
        let query_shape = [queries, 1];
        let queries = self::points(&data[..queries], &query_shape);
        let executor = CountingExecutor::new(InlineExecutor(NonZeroUsize::new(budget).unwrap()));
        let options = SearchOptions::new().with_execution_policy(policy);
        let mut workspace = tree
            .workspace(SearchWorkspaceRequest::new(1, options).with_batch(
                queries.len(),
                queries.len(),
                NonZeroUsize::new(expected_workers).unwrap_or(NonZeroUsize::MIN),
            ))
            .unwrap();
        let mut output = vec![sentinel(); queries.len()];
        let mut counts = vec![99; queries.len()];
        assert_eq!(
            run_batch(
                &tree,
                &queries,
                QueryCase::Nearest { k: 1 },
                options,
                &mut output,
                &mut counts,
                &mut workspace,
                ExecutionContext::new(&executor)
            )
            .unwrap(),
            queries.len()
        );
        executor.assert_dispatches(usize::from(!queries.is_empty()), expected_workers);
        for (index, neighbor) in output.iter().enumerate() {
            assert_eq!(
                *neighbor,
                Neighbor {
                    index,
                    distance: 0.0
                }
            );
        }
        assert!(counts.iter().all(|count| *count == 1));
    }
}

fn sentinel() -> Neighbor<f64> {
    Neighbor {
        index: usize::MAX,
        distance: -1.0,
    }
}

#[test]
fn sparse_radius_batch_storage_tracks_workers_and_actual_output() {
    let data: Vec<_> = (0..16384).map(f64::from).collect();
    let shape = [data.len(), 1];
    let points = points(&data, &shape);
    let tree = KdTree::new(&points).unwrap();
    let options = SearchOptions::new();
    let small = tree
        .workspace_requirements(SearchWorkspaceRequest::new(0, options).with_batch(
            16,
            16,
            workers(),
        ))
        .unwrap();
    let large = tree
        .workspace_requirements(SearchWorkspaceRequest::new(0, options).with_batch(
            data.len(),
            data.len(),
            workers(),
        ))
        .unwrap();
    assert_eq!(
        small.worker_bytes::<f64>().unwrap(),
        large.worker_bytes::<f64>().unwrap()
    );
    assert_eq!(large.workers(), 4);
    assert_eq!(large.neighbors_per_worker(), 0);
    assert_eq!(large.output_len(), data.len());
    assert!(large.bytes::<f64>().unwrap() < 1024 * 1024);
    let mut workspace = SearchWorkspace::new();
    assert_eq!(workspace.check(&large).unwrap(), Capacity::NeedsGrowth);
    workspace.try_reserve(&large).unwrap();
    assert_eq!(workspace.check(&large).unwrap(), Capacity::Sufficient);
    let mut output = vec![sentinel(); data.len() + 1];
    let mut counts = vec![99; data.len() + 1];
    let total = run_batch(
        &tree,
        &points,
        QueryCase::Radius { radius: 0.0 },
        options,
        &mut output,
        &mut counts,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(total, data.len());
    for (index, neighbor) in output[..total].iter().enumerate() {
        assert_eq!(neighbor.index, index);
        assert_eq!(neighbor.distance, 0.0);
        assert_eq!(counts[index], 1);
    }
    assert_eq!(output[total], sentinel());
    assert_eq!(counts[total], 99);
    let mut scalar = tree
        .workspace(SearchWorkspaceRequest::new(1, options))
        .unwrap();
    let mut single = [sentinel()];
    assert_eq!(
        tree.radius_into(
            &[500.0],
            0.0,
            options,
            &mut single,
            &mut scalar,
            ExecutionContext::sequential()
        )
        .unwrap(),
        1
    );
    assert_eq!(single[0].index, 500);
}

#[cfg(feature = "std")]
#[test]
fn real_parallel_batches_and_condensed_distances_match_sequential() {
    let data: Vec<_> = (0..301).map(|i| f64::from((i * 17) % 59)).collect();
    let shape = [data.len(), 1];
    let points = points(&data, &shape);
    let tree = KdTree::with_options(
        &points,
        TreeMetric::Euclidean,
        KdTreeOptions::default().with_leaf_size(2),
    )
    .unwrap();
    let options = SearchOptions::new().with_epsilon(0.1);
    let mut workspace = tree
        .workspace(SearchWorkspaceRequest::new(8, options).with_batch(
            data.len(),
            data.len() * 32,
            workers(),
        ))
        .unwrap();
    let executor = scioxide_core::ThreadExecutor::new(workers());
    for kind in [
        QueryCase::Nearest { k: 8 },
        QueryCase::Radius { radius: 1.0 },
    ] {
        let mut sequential = vec![sentinel(); data.len() * 32];
        let mut parallel = sequential.clone();
        let mut expected_counts = vec![99; data.len()];
        let mut counts = expected_counts.clone();
        let expected = run_batch(
            &tree,
            &points,
            kind,
            options,
            &mut sequential,
            &mut expected_counts,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        let actual = run_batch(
            &tree,
            &points,
            kind,
            options.with_execution_policy(ExecutionPolicy::Parallel(workers())),
            &mut parallel,
            &mut counts,
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap();
        assert_eq!(expected, actual);
        assert_eq!(sequential, parallel);
        assert_eq!(expected_counts, counts);
    }
    let metric = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    let options = DistanceOptions::new();
    let mut workspace =
        DistanceWorkspace::workspace(DistanceWorkspaceRequest::condensed(data.len(), options))
            .unwrap();
    let mut sequential = vec![0.0; condensed_len(data.len()).unwrap()];
    let mut parallel = sequential.clone();
    metric
        .condensed_into(
            &points,
            options,
            &mut sequential,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    metric
        .condensed_into(
            &points,
            options.with_execution_policy(ExecutionPolicy::Parallel(workers())),
            &mut parallel,
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap();
    assert_eq!(sequential, parallel);
}

#[test]
fn late_arithmetic_and_capacity_failures_preserve_both_outputs() {
    let sites = [f64::MAX];
    let site_shape = [1, 1];
    let queries = [f64::MAX, -f64::MAX];
    let query_shape = [2, 1];
    let sites = points(&sites, &site_shape);
    let queries = points(&queries, &query_shape);
    let tree = KdTree::new(&sites).unwrap();
    let options = SearchOptions::new();
    let mut workspace = tree
        .workspace(SearchWorkspaceRequest::new(1, options).with_batch(2, 2, NonZeroUsize::MIN))
        .unwrap();
    for kind in [
        QueryCase::Nearest { k: 1 },
        QueryCase::Radius { radius: 0.0 },
    ] {
        let mut output = [sentinel(); 2];
        let mut counts = [99; 2];
        let result = run_batch(
            &tree,
            &queries,
            kind,
            options,
            &mut output,
            &mut counts,
            &mut workspace,
            ExecutionContext::sequential(),
        );
        match kind {
            QueryCase::Nearest { .. } => {
                assert_eq!(
                    result.unwrap_err().kind(),
                    SciOxideErrorKind::NumericalInstability
                );
                assert_eq!(output, [sentinel(); 2]);
                assert_eq!(counts, [99; 2]);
            }
            QueryCase::Radius { .. } => {
                assert_eq!(result.unwrap(), 1);
                assert_eq!(
                    output,
                    [
                        Neighbor {
                            index: 0,
                            distance: 0.0
                        },
                        sentinel()
                    ]
                );
                assert_eq!(counts, [1, 0]);
            }
        }
    }
    let data = [0.0, 1.0];
    let shape = [2, 1];
    let sites = points(&data, &shape);
    let tree = KdTree::new(&sites).unwrap();
    for (staged, published) in [(3, 4), (4, 3)] {
        let mut workspace = tree
            .workspace(SearchWorkspaceRequest::new(0, options).with_batch(
                2,
                staged,
                NonZeroUsize::MIN,
            ))
            .unwrap();
        let mut output = vec![sentinel(); published];
        let mut counts = [99; 2];
        let error = run_batch(
            &tree,
            &sites,
            QueryCase::Radius { radius: 2.0 },
            options,
            &mut output,
            &mut counts,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert!(matches!(error.kind(), SciOxideErrorKind::DimensionMismatch));
        assert!(output.iter().all(|&n| n == sentinel()));
        assert_eq!(counts, [99; 2]);
    }
}

struct CancellingExecutor<'a> {
    calls: AtomicUsize,
    pass: usize,
    after_all: bool,
    token: &'a CancellationToken,
}
impl Executor for CancellingExecutor<'_> {
    fn worker_budget(&self) -> NonZeroUsize {
        workers()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        let pass = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        SequentialExecutor.execute(context, plan, &|partition, (), child| {
            let worker = partition.worker_index();
            operation(partition, (), child)?;
            if pass == self.pass && !self.after_all && worker == 0 {
                self.token.cancel();
            }
            Ok(())
        })?;
        if pass == self.pass && self.after_all {
            self.token.cancel();
        }
        Ok(())
    }
}
#[test]
fn cancellation_in_each_radius_pass_and_at_publication_is_atomic() {
    let data: Vec<_> = (0..300).map(f64::from).collect();
    let shape = [300, 1];
    let points = points(&data, &shape);
    let tree = KdTree::new(&points).unwrap();
    let options = SearchOptions::new().with_execution_policy(ExecutionPolicy::Parallel(workers()));
    let mut workspace = tree
        .workspace(SearchWorkspaceRequest::new(0, options).with_batch(300, 300, workers()))
        .unwrap();
    for (pass, after_all) in [(1, false), (2, false), (2, true)] {
        let token = CancellationToken::new();
        let executor = CancellingExecutor {
            calls: AtomicUsize::new(0),
            pass,
            after_all,
            token: &token,
        };
        let mut output = vec![sentinel(); 300];
        let mut counts = vec![99; 300];
        assert!(
            run_batch(
                &tree,
                &points,
                QueryCase::Radius { radius: 0.0 },
                options,
                &mut output,
                &mut counts,
                &mut workspace,
                ExecutionContext::new(&executor).with_cancellation(&token)
            )
            .is_err()
        );
        assert!(output.iter().all(|&n| n == sentinel()));
        assert!(counts.iter().all(|&n| n == 99));
        assert_eq!(executor.calls.load(Ordering::Relaxed), pass);
        assert_eq!(
            run_batch(
                &tree,
                &points,
                QueryCase::Radius { radius: 0.0 },
                options,
                &mut output,
                &mut counts,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap(),
            300
        );
    }
}

struct StopAfterPolls(AtomicUsize);
impl scioxide_core::Cancellation for StopAfterPolls {
    fn is_cancelled(&self) -> bool {
        self.0.fetch_add(1, Ordering::Relaxed) >= 10
    }
}

#[test]
fn short_and_skipped_batch_queries_poll_before_publication() {
    let sites = points(&[0.0], &[1, 1]);
    let queries = PointSet::new(
        NdView::new(
            &[0.0],
            StridedLayout::new_broadcast(&[1024, 1], &[0, 1]).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let tree = KdTree::new(&sites).unwrap();
    let options = SearchOptions::new();
    let mut workspace = tree
        .workspace(SearchWorkspaceRequest::new(1, options).with_batch(
            1024,
            1024,
            NonZeroUsize::MIN,
        ))
        .unwrap();
    for k in [0, 1] {
        let cancellation = StopAfterPolls(AtomicUsize::new(0));
        let mut output = vec![sentinel(); 1024];
        let mut counts = vec![99; 1024];
        let error = run_batch(
            &tree,
            &queries,
            QueryCase::Nearest { k },
            options,
            &mut output,
            &mut counts,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
        assert!(output.iter().all(|&n| n == sentinel()));
        assert!(counts.iter().all(|&n| n == 99));
        assert_eq!(
            run_batch(
                &tree,
                &queries,
                QueryCase::Nearest { k },
                options,
                &mut output,
                &mut counts,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap(),
            1024 * k
        );
        assert!(counts.iter().all(|&n| n == k));
    }
}

#[test]
fn requirements_check_overflow_limits_and_retained_old_storage() {
    let data = [0.0, 1.0];
    let shape = [2, 1];
    let points = points(&data, &shape);
    let tree = KdTree::new(&points).unwrap();
    let options = SearchOptions::new();
    for request in [
        SearchWorkspaceRequest::new(0, options).with_batch(usize::MAX, 0, workers()),
        SearchWorkspaceRequest::new(0, options).with_batch(1, usize::MAX, workers()),
    ] {
        assert!(matches!(
            tree.workspace_requirements(request).unwrap_err().kind(),
            SciOxideErrorKind::SizeOverflow
        ));
    }
    let mut workspace = tree
        .workspace(SearchWorkspaceRequest::new(2, options).with_batch(100, 200, workers()))
        .unwrap();
    let small = tree
        .workspace_requirements(SearchWorkspaceRequest::new(0, options))
        .unwrap();
    let limited = tree
        .workspace_requirements(SearchWorkspaceRequest::new(
            0,
            options.with_max_workspace_bytes(small.bytes::<f64>().unwrap()),
        ))
        .unwrap();
    assert!(matches!(
        workspace.check(&limited).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    ));
    assert!(workspace.try_reserve(&limited).is_err());
    assert_eq!(workspace.check(&small).unwrap(), Capacity::Sufficient);
    assert!(
        DistanceWorkspace::<f64>::workspace_requirements(DistanceWorkspaceRequest::pairwise(
            usize::MAX,
            2,
            DistanceOptions::new()
        ))
        .is_err()
    );
    let mut workspace: DistanceWorkspace<f64> =
        DistanceWorkspace::workspace(DistanceWorkspaceRequest::square(10, DistanceOptions::new()))
            .unwrap();
    let small = DistanceWorkspace::<f64>::workspace_requirements(DistanceWorkspaceRequest::square(
        1,
        DistanceOptions::new().with_max_workspace_bytes(8),
    ))
    .unwrap();
    assert!(matches!(
        workspace.check(&small).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    ));
    assert!(workspace.try_reserve(&small).is_err());
}

#[test]
fn borrowed_point_handle_can_end_before_the_tree() {
    let data = [0.0, 1.0];
    let shape = [2, 1];
    let tree = {
        let points = points(&data, &shape);
        KdTree::new(&points).unwrap()
    };
    assert_eq!(tree.nearest(&[0.0], 1).unwrap()[0].index, 0);
}

#[test]
fn distance_outputs_remain_atomic_after_executor_completion() {
    let data: Vec<_> = (0..32).map(f64::from).collect();
    let shape = [32, 1];
    let points = points(&data, &shape);
    let metric = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    let options =
        DistanceOptions::new().with_execution_policy(ExecutionPolicy::Parallel(workers()));
    let mut workspace =
        DistanceWorkspace::workspace(DistanceWorkspaceRequest::square(32, options)).unwrap();
    let upper: Vec<_> = (0..32)
        .flat_map(|i| (i + 1..32).map(move |j| f64::from(j - i)))
        .collect();
    let square: Vec<_> = (0..1024)
        .map(|i| f64::from((i / 32_i32 - i % 32).abs()))
        .collect();
    for operation in 0..4 {
        let token = CancellationToken::new();
        let executor = CancellingExecutor {
            calls: AtomicUsize::new(0),
            pass: 1,
            after_all: true,
            token: &token,
        };
        let context = ExecutionContext::new(&executor).with_cancellation(&token);
        let mut output = [-1.0; 1024];
        let result = match operation {
            0 => metric.condensed_into(&points, options, &mut output, &mut workspace, context),
            1 => metric.pairwise_into(
                &points,
                &points,
                options,
                &mut output,
                &mut workspace,
                context,
            ),
            2 => {
                condensed_to_square_into(32, &upper, options, &mut output, &mut workspace, context)
            }
            _ => square_to_condensed_into(
                scioxide_linalg::MatrixView::from_slice(32, 32, &square).unwrap(),
                0.0,
                options,
                &mut output,
                &mut workspace,
                context,
            ),
        };
        assert!(result.is_err());
        assert_eq!(output, [-1.0; 1024]);
        assert_eq!(executor.calls.load(Ordering::Relaxed), 1);
    }
}

struct DistanceClock(AtomicUsize);
impl scioxide_core::Clock for DistanceClock {
    fn now(&self) -> Result<scioxide_core::MonotonicInstant, scioxide_core::CapabilityError> {
        Ok(scioxide_core::MonotonicInstant::from_duration_since_epoch(
            core::time::Duration::from_secs(self.0.fetch_add(1, Ordering::Relaxed) as u64),
        ))
    }
}

#[test]
fn distance_deadlines_poll_in_blocks_and_preserve_output_on_expiry() {
    let n = 65;
    let data: Vec<_> = (0..n).map(|i| i as f64).collect();
    let shape = [n, 1];
    let points = points(&data, &shape);
    let upper: Vec<_> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (j - i) as f64))
        .collect();
    let square: Vec<_> = (0..n * n).map(|i| (i / n).abs_diff(i % n) as f64).collect();
    let matrix = scioxide_linalg::MatrixView::from_slice(n, n, &square).unwrap();
    let metric = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    let options = DistanceOptions::new();
    let mut workspace =
        DistanceWorkspace::workspace(DistanceWorkspaceRequest::square(n, options)).unwrap();
    for seconds in [8, 10_000] {
        for operation in 0..4 {
            let clock = DistanceClock(AtomicUsize::new(0));
            let binding = scioxide_core::ClockBinding::new(&clock);
            let context = ExecutionContext::sequential()
                .with_deadline(scioxide_core::Deadline::at(
                    &binding,
                    scioxide_core::MonotonicInstant::from_duration_since_epoch(
                        core::time::Duration::from_secs(seconds),
                    ),
                ))
                .unwrap();
            let mut output = vec![-1.0; n * n + 2];
            let result = match operation {
                0 => metric.pairwise_into(
                    &points,
                    &points,
                    options,
                    &mut output,
                    &mut workspace,
                    context,
                ),
                1 => metric.condensed_into(&points, options, &mut output, &mut workspace, context),
                2 => condensed_to_square_into(
                    n,
                    &upper,
                    options,
                    &mut output,
                    &mut workspace,
                    context,
                ),
                _ => square_to_condensed_into(
                    matrix,
                    0.0,
                    options,
                    &mut output,
                    &mut workspace,
                    context,
                ),
            };
            if seconds == 8 {
                assert!(matches!(
                    result.unwrap_err().view(),
                    scioxide_core::SciOxideErrorView::ExecutionStopped {
                        reason: scioxide_core::ExecutionStopReason::DeadlineReached,
                        ..
                    }
                ));
                assert!(output.iter().all(|&x| x == -1.0));
            } else {
                result.unwrap();
                let expected = if operation == 1 || operation == 3 {
                    &upper
                } else {
                    &square
                };
                assert_eq!(&output[..expected.len()], expected);
                assert!(output[expected.len()..].iter().all(|&x| x == -1.0));
                // Includes validation, executor entry/exit and publication.
                // Per-element clock reads would exceed this by orders of magnitude.
                assert!(clock.0.load(Ordering::Relaxed) <= 64);
            }
        }
    }
}

#[test]
fn matrix_views_validate_finiteness_tolerance_and_atomic_conversion() {
    use scioxide_linalg::MatrixView;
    // Padding is not part of the matrix and may be non-finite.
    let data = [0.0, 2.0, f64::NAN, 2.0, 0.0];
    let matrix = MatrixView::from_slice_with_strides(2, 2, &data, 3, 1).unwrap();
    validate_distance_matrix(matrix, 0.0).unwrap();
    let options = DistanceOptions::new();
    let mut workspace =
        DistanceWorkspace::workspace(DistanceWorkspaceRequest::square(2, options)).unwrap();
    let mut output = [-1.0; 2];
    square_to_condensed_into(
        matrix,
        0.0,
        options,
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(output, [2.0, -1.0]);
    for invalid in [
        [0.0, 2.0, f64::NAN, 0.0],
        [0.0, 2.0, f64::INFINITY, 0.0],
        [0.0, -2.0, -2.0, 0.0],
        [0.0, 2.0, 3.0, 0.0],
        [1.0, 2.0, 2.0, 0.0],
    ] {
        let matrix = MatrixView::from_slice(2, 2, &invalid).unwrap();
        let saved = output;
        assert!(
            square_to_condensed_into(
                matrix,
                0.0,
                options,
                &mut output,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(output, saved);
    }
    for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(validate_distance_matrix(matrix, tolerance).is_err());
    }
    let near = [0.125, 2.0, 2.125, 0.0];
    let near = MatrixView::from_slice(2, 2, &near).unwrap();
    validate_distance_matrix(near, 0.125).unwrap();
    assert!(validate_distance_matrix(near, 0.124).is_err());
    let empty = MatrixView::from_slice(0, 0, &[] as &[f64]).unwrap();
    validate_distance_matrix(empty, 0.0).unwrap();
    let nonsquare = MatrixView::from_slice(1, 2, &[0.0, 1.0]).unwrap();
    assert_eq!(
        validate_distance_matrix(nonsquare, 0.0).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let token = CancellationToken::new();
    token.cancel();
    let mut square = [-1.0; 4];
    assert!(
        condensed_to_square_into(
            2,
            &[2.0],
            options,
            &mut square,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(square, [-1.0; 4]);
}

#[test]
fn empty_batches_and_empty_trees_publish_only_implied_counts() {
    let empty_shape = [0, 1];
    let empty = points(&[], &empty_shape);
    let tree = KdTree::new(&empty).unwrap();
    let options = SearchOptions::new().with_max_workspace_bytes(0);
    let request = SearchWorkspaceRequest::new(0, options).with_batch(0, 0, workers());
    assert_eq!(
        tree.workspace_requirements(request)
            .unwrap()
            .bytes::<f64>()
            .unwrap(),
        0
    );
    assert_eq!(
        run_batch(
            &tree,
            &empty,
            QueryCase::Radius { radius: 0.0 },
            options,
            &mut [],
            &mut [],
            &mut SearchWorkspace::new(),
            ExecutionContext::sequential()
        )
        .unwrap(),
        0
    );
    let query_shape = [2, 1];
    let queries = points(&[0.0, 1.0], &query_shape);
    let options = SearchOptions::new();
    let mut workspace = tree
        .workspace(SearchWorkspaceRequest::new(0, options).with_batch(2, 0, workers()))
        .unwrap();
    for kind in [
        QueryCase::Nearest { k: 4 },
        QueryCase::Radius { radius: 1.0 },
    ] {
        let mut output = [sentinel(); 1];
        let mut counts = [99; 3];
        assert_eq!(
            run_batch(
                &tree,
                &queries,
                kind,
                options,
                &mut output,
                &mut counts,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap(),
            0
        );
        assert_eq!(output, [sentinel()]);
        assert_eq!(counts, [0, 0, 99]);
    }
}

#[test]
fn reuse_with_fewer_workers_does_not_require_growth_of_inactive_workers() {
    let shape = [2, 1];
    let points = points(&[0.0, 1.0], &shape);
    let tree = KdTree::new(&points).unwrap();
    let options = SearchOptions::new();
    let mut workspace = tree
        .workspace(SearchWorkspaceRequest::new(1, options).with_batch(2, 2, workers()))
        .unwrap();
    let shape2 = [2, 2];
    let strides2 = [2, 1];
    let points2 = PointSet::new(
        NdView::new(
            &[0.0_f64, 0.0, 1.0, 1.0],
            StridedLayout::new(&shape2, &strides2).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let tree2 = KdTree::new(&points2).unwrap();
    let requirements = tree2
        .workspace_requirements(SearchWorkspaceRequest::new(1, options).with_batch(
            2,
            2,
            NonZeroUsize::MIN,
        ))
        .unwrap();
    workspace.try_reserve(&requirements).unwrap();
    let mut output = [sentinel(); 2];
    let mut counts = [99; 2];
    assert_eq!(
        run_batch(
            &tree2,
            &points2,
            QueryCase::Nearest { k: 1 },
            options,
            &mut output,
            &mut counts,
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap(),
        2
    );
    assert_eq!(counts, [1, 1]);
    assert_eq!(output[0].index, 0);
    assert_eq!(output[1].index, 1);
}

#[test]
fn empty_batch_needs_no_axis_storage_even_for_large_dimensions() {
    let shape = [0, usize::MAX];
    let points = PointSet::<f64>::new(
        NdView::new(&[], StridedLayout::new_broadcast(&shape, &[0, 0]).unwrap()).unwrap(),
    )
    .unwrap();
    let tree = KdTree::new(&points).unwrap();
    let options = SearchOptions::new().with_max_workspace_bytes(0);
    let request = SearchWorkspaceRequest::new(0, options).with_batch(0, 0, workers());
    assert_eq!(
        tree.workspace_requirements(request)
            .unwrap()
            .bytes::<f64>()
            .unwrap(),
        0
    );
    assert_eq!(
        run_batch(
            &tree,
            &points,
            QueryCase::Nearest { k: 1 },
            options,
            &mut [],
            &mut [],
            &mut SearchWorkspace::new(),
            ExecutionContext::sequential()
        )
        .unwrap(),
        0
    );
}

// An executor with expensive dispatch, like `ThreadExecutor`.
struct FlooredExecutor(NonZeroUsize);
impl Executor for FlooredExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        self.0
    }
    fn auto_min_work_per_worker(&self) -> NonZeroUsize {
        NonZeroUsize::new(32_768).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        SequentialExecutor.execute(context, plan, operation)
    }
}

// Auto admits workers by the documented per-item estimates, not one unit per
// query or pair: d * (ceil(log2 n) + leaf + k) per KD query and d per pair.
#[test]
fn auto_planning_uses_documented_work_estimates() {
    let data: Vec<_> = (0..4096).map(|i| f64::from(i) / 4096.0).collect();
    let tree_shape = [300, 1];
    let tree_points = points(&data[..300], &tree_shape);
    let tree = KdTree::new(&tree_points).unwrap();
    // One coordinate, a 9-level descent, 16-point leaves and k = 1: 26 units,
    // so a worker needs ceil(32768 / 26) = 1261 queries.
    for (queries, expected_workers) in [(1260, 1), (2522, 2), (4096, 3)] {
        let query_shape = [queries, 1];
        let queries = points(&data[..queries], &query_shape);
        let executor = CountingExecutor::new(FlooredExecutor(workers()));
        let options = SearchOptions::new().with_execution_policy(ExecutionPolicy::Auto);
        let mut workspace = tree
            .workspace(SearchWorkspaceRequest::new(1, options).with_batch(
                queries.len(),
                queries.len(),
                workers(),
            ))
            .unwrap();
        let mut output = vec![sentinel(); queries.len()];
        let mut counts = vec![0; queries.len()];
        run_batch(
            &tree,
            &queries,
            QueryCase::Nearest { k: 1 },
            options,
            &mut output,
            &mut counts,
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap();
        executor.assert_dispatches(1, expected_workers);
    }
    // Three-dimensional Euclidean pairs: 3 units, so 10923 pairs per worker.
    let cloud: Vec<_> = (0..3 * 128).map(|i| f64::from(i) / 64.0).collect();
    let layout = StridedLayout::new(&[128, 3], &[3, 1]).unwrap();
    let cloud = PointSet::new(NdView::new(&cloud, layout).unwrap()).unwrap();
    let metric = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    let options = DistanceOptions::new().with_execution_policy(ExecutionPolicy::Auto);
    let request = DistanceWorkspaceRequest::pairwise(128, 128, options);
    let mut workspace = DistanceWorkspace::<f64>::workspace(request).unwrap();
    let mut output = vec![0.0; 128 * 128];
    let executor = CountingExecutor::new(FlooredExecutor(workers()));
    metric
        .pairwise_into(
            &cloud,
            &cloud,
            options,
            &mut output,
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap();
    executor.assert_dispatches(1, 1);
    let options = options.with_execution_policy(ExecutionPolicy::Auto);
    let wide: Vec<_> = (0..3 * 256).map(|i| f64::from(i) / 64.0).collect();
    let layout = StridedLayout::new(&[256, 3], &[3, 1]).unwrap();
    let wide = PointSet::new(NdView::new(&wide, layout).unwrap()).unwrap();
    let mut workspace =
        DistanceWorkspace::<f64>::workspace(DistanceWorkspaceRequest::pairwise(256, 128, options))
            .unwrap();
    let mut output = vec![0.0; 256 * 128];
    executor.reset();
    metric
        .pairwise_into(
            &wide,
            &cloud,
            options,
            &mut output,
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap();
    // 32768 pairs / 10923 per worker = 2 (not 4, as one unit per pair implied).
    executor.assert_dispatches(1, 2);
}

// Scalar and batch searches validate in the same order, radius staging names
// its capacity, conversions size their destination before scanning input,
// and distance requirements enforce the byte ceiling up front.
#[test]
fn validation_order_capacity_errors_and_requirement_limits() {
    let data = [0.0, 1.0, 2.0, 3.0];
    let shape = [4, 1];
    let tree_points = points(&data, &shape);
    let tree = KdTree::new(&tree_points).unwrap();
    let bad = SearchOptions::new().with_epsilon(-1.0);
    let wide = [0.0, 0.0];
    let wide_points =
        PointSet::new(NdView::new(&wide, StridedLayout::new(&[1, 2], &[2, 1]).unwrap()).unwrap())
            .unwrap();
    let mut workspace = tree
        .workspace(SearchWorkspaceRequest::new(4, SearchOptions::new()))
        .unwrap();
    let mut output = vec![sentinel(); 4];
    let mut counts = [0; 1];
    let scalar = tree
        .nearest_into(
            &wide,
            1,
            bad,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    let batch = run_batch(
        &tree,
        &wide_points,
        QueryCase::Nearest { k: 1 },
        bad,
        &mut output,
        &mut counts,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert_eq!(scalar.kind(), SciOxideErrorKind::DimensionMismatch);
    assert_eq!(batch.kind(), SciOxideErrorKind::DimensionMismatch);
    // Two reserved candidates but four matches.
    let mut small = tree
        .workspace(SearchWorkspaceRequest::new(2, SearchOptions::new()))
        .unwrap();
    let error = tree
        .radius_into(
            &[1.5],
            10.0,
            SearchOptions::new(),
            &mut output,
            &mut small,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(
        matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource, required: 3, limit: 2 }
        if resource == scioxide_core::quantity!("radius_neighbors"))
    );
    assert!(output.iter().all(|&n| n == sentinel()));
    // An undersized destination is reported before the non-finite input.
    let condensed = [f64::NAN, 1.0, 1.0];
    let options = DistanceOptions::new();
    let mut workspace =
        DistanceWorkspace::<f64>::workspace(DistanceWorkspaceRequest::square(3, options)).unwrap();
    let mut square = [0.0; 4];
    let error = condensed_to_square_into(
        3,
        &condensed,
        options,
        &mut square,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    let limited = DistanceOptions::new().with_max_workspace_bytes(8);
    let error = DistanceWorkspace::<f64>::workspace_requirements(DistanceWorkspaceRequest::square(
        2, limited,
    ))
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
}
