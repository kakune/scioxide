//! Prepared successful numerical paths allocate no storage.
#![allow(clippy::unwrap_used)]
#[path = "support/search_cases.rs"]
mod search_cases;

use scioxide_core::{ExecutionContext, NdView, StridedLayout};
use scioxide_spatial::*;
use search_cases::{QueryCase, run_batch};

#[test]
fn unrepresentable_losers_and_required_range_errors_execute() {
    range_allocations::<f32>();
    range_allocations::<f64>();
}

fn range_allocations<S: scioxide_core::RealScalar>() {
    // Visit the unrepresentable loser before the zero-distance winner.
    let data = [S::MAX, -S::MAX, S::ZERO];
    let shape = [3, 1];
    let query_shape = [1, 1];
    let strides = [1, 1];
    let points =
        PointSet::new(NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    let query = [-S::MAX];
    let queries = PointSet::new(
        NdView::new(&query, StridedLayout::new(&query_shape, &strides).unwrap()).unwrap(),
    )
    .unwrap();
    let tree = KdTree::new(&points).unwrap();
    let options = SearchOptions::new();
    let mut workspace = tree
        .workspace(SearchWorkspaceRequest::new(3, options).with_batch(
            1,
            3,
            core::num::NonZeroUsize::MIN,
        ))
        .unwrap();
    let saved = Neighbor {
        index: usize::MAX,
        distance: -S::ONE,
    };
    let mut output = [saved; 3];
    let mut counts = [99];
    {
        tree.nearest_into(
            &query,
            1,
            options,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(output[0].index, 1);
        tree.radius_into(
            &query,
            S::ZERO,
            options,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        for kind in [
            QueryCase::Nearest { k: 1 },
            QueryCase::Radius { radius: S::ZERO },
        ] {
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
                1
            );
            assert_eq!(output[0].index, 1);
            assert_eq!(counts, [1]);
        }
        output.fill(saved);
        assert!(
            tree.nearest_into(
                &query,
                3,
                options,
                &mut output,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(output, [saved; 3]);
    };
}

#[test]
fn reserved_queries_and_distances_execute() {
    let data: Vec<f64> = (0..512).map(|i| ((i * 17) % 257) as f64).collect();
    let shape = [256, 2];
    let strides = [2, 1];
    let points = PointSet::new(
        NdView::new(
            &data,
            StridedLayout::new_broadcast(&shape, &strides).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let tree = KdTree::new(&points).unwrap();
    let inverse = [2.0, 0.5, 0.5, 1.0];
    let inverse_points = scioxide_linalg::MatrixView::from_slice(2, 2, &inverse).unwrap();
    let metrics = [
        PreparedDistance::new(DistanceMetric::Euclidean).unwrap(),
        PreparedDistance::new(DistanceMetric::CityBlock).unwrap(),
        PreparedDistance::standardized(&[2.0, 3.0]).unwrap(),
        PreparedDistance::mahalanobis(inverse_points).unwrap(),
    ];
    let mut workspace = SearchWorkspace::new();
    workspace
        .try_reserve(
            &tree
                .workspace_requirements(SearchWorkspaceRequest::new(256, SearchOptions::new()))
                .unwrap(),
        )
        .unwrap();
    let mut out = [Neighbor::default(); 256];
    let mut distances = vec![0.0; 65536];
    let mut scratch = distance_workspace(65536);
    {
        tree.nearest_into(
            &[12.0, 13.0],
            8,
            SearchOptions::new(),
            &mut out,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        tree.radius_into(
            &[12.0, 13.0],
            50.0,
            SearchOptions::new().with_epsilon(0.1),
            &mut out,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        for metric in &metrics {
            metric
                .pairwise_into(
                    &points,
                    &points,
                    DistanceOptions::new(),
                    &mut distances,
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
            metric
                .condensed_into(
                    &points,
                    DistanceOptions::new(),
                    &mut distances,
                    &mut scratch,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap();
        }
    };
}

#[test]
fn distance_blocks_and_scaled_replay_execute() {
    let shape = [33, 3];
    let strides = [3, 1];
    let metrics = [
        DistanceMetric::Euclidean,
        DistanceMetric::SquaredEuclidean,
        DistanceMetric::Minkowski(2.0),
        DistanceMetric::CityBlock,
        DistanceMetric::Chebyshev,
    ]
    .map(|m| PreparedDistance::new(m).unwrap());
    let mut output = vec![0.0; 33 * 33];
    let mut workspace = distance_workspace(output.len());
    for scale in [0.1, 1e-200, 1e-160, 1e200] {
        let data: Vec<_> = (0..99).map(|i| (i % 17) as f64 * scale).collect();
        let points = PointSet::new(
            NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
        )
        .unwrap();
        for metric in &metrics {
            let _ = metric.pairwise_into(
                &points,
                &points,
                DistanceOptions::new(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            );
            let _ = metric.condensed_into(
                &points,
                DistanceOptions::new(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            );
        }
    }
}

#[test]
fn every_metric_and_periodic_strided_search_reuses_axis_storage() {
    let data: Vec<f64> = (0..256 * 5)
        .map(|i| ((i * 17) % 257) as f64 / 512.0)
        .collect();
    let shape = [256, 2];
    for strides in [[2, 1], [5, 2], [0, 1], [1, 0]] {
        let points = PointSet::new(
            NdView::new(
                &data,
                StridedLayout::new_broadcast(&shape, &strides).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        for metric in [
            TreeMetric::Euclidean,
            TreeMetric::CityBlock,
            TreeMetric::Chebyshev,
            TreeMetric::Minkowski(1.0),
            TreeMetric::Minkowski(2.0),
            TreeMetric::Minkowski(3.5),
        ] {
            for periodic in [false, true] {
                let lengths = [1.0; 2];
                let tree = KdTree::with_options(
                    &points,
                    metric,
                    KdTreeOptions::default()
                        .with_leaf_size(4)
                        .with_periodic_box(periodic.then_some(&lengths)),
                )
                .unwrap();
                let mut workspace = SearchWorkspace::new();
                workspace
                    .try_reserve(
                        &tree
                            .workspace_requirements(SearchWorkspaceRequest::new(
                                256,
                                SearchOptions::new(),
                            ))
                            .unwrap(),
                    )
                    .unwrap();
                let mut out = [Neighbor::default(); 256];
                {
                    tree.nearest_into(
                        &[0.2, 0.7],
                        8,
                        SearchOptions::new(),
                        &mut out,
                        &mut workspace,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                    tree.radius_into(
                        &[-1.2, 2.7],
                        0.4,
                        SearchOptions::new().with_epsilon(0.1),
                        &mut out,
                        &mut workspace,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                };
            }
        }
    }
}

fn distance_workspace<S: scioxide_core::RealScalar>(len: usize) -> DistanceWorkspace<S> {
    DistanceWorkspace::workspace(DistanceWorkspaceRequest::pairwise(
        1,
        len,
        DistanceOptions::new(),
    ))
    .unwrap()
}

struct InlinePartitions;
impl scioxide_core::Executor for InlinePartitions {
    fn worker_budget(&self) -> core::num::NonZeroUsize {
        core::num::NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        cx: ExecutionContext<'_>,
        plan: scioxide_core::ExecutionPlan,
        operation: &scioxide_core::PartitionFn<'_>,
    ) -> Result<(), scioxide_core::SciOxideError> {
        scioxide_core::Executor::execute(&scioxide_core::SequentialExecutor, cx, plan, operation)
    }
}

#[test]
fn prepared_batches_and_scalar_radius_execute() {
    let data: Vec<_> = (0..300).map(f64::from).collect();
    let shape = [300, 1];
    let strides = [1, 1];
    let points =
        PointSet::new(NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    let tree = KdTree::new(&points).unwrap();
    let workers = core::num::NonZeroUsize::new(4).unwrap();
    let options = SearchOptions::new();
    let mut workspace = tree
        .workspace(SearchWorkspaceRequest::new(8, options).with_batch(300, 2400, workers))
        .unwrap();
    let mut output = vec![Neighbor::default(); 2400];
    let mut counts = [0; 300];
    let mut scalar = tree
        .workspace(SearchWorkspaceRequest::new(1, options))
        .unwrap();
    let mut one = [Neighbor::default()];
    for policy in [
        scioxide_core::ExecutionPolicy::Sequential,
        scioxide_core::ExecutionPolicy::Parallel(workers),
    ] {
        let options = options.with_execution_policy(policy);
        {
            for kind in [
                QueryCase::Nearest { k: 8 },
                QueryCase::Radius { radius: 0.0 },
            ] {
                run_batch(
                    &tree,
                    &points,
                    kind,
                    options,
                    &mut output,
                    &mut counts,
                    &mut workspace,
                    ExecutionContext::new(&InlinePartitions),
                )
                .unwrap();
            }
            tree.radius_into(
                &[100.0],
                0.0,
                options,
                &mut one,
                &mut scalar,
                ExecutionContext::sequential(),
            )
            .unwrap();
        };
    }
    let metric = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    let options = DistanceOptions::new()
        .with_execution_policy(scioxide_core::ExecutionPolicy::Parallel(workers));
    let mut workspace =
        DistanceWorkspace::workspace(DistanceWorkspaceRequest::pairwise(300, 300, options))
            .unwrap();
    let mut output = vec![0.0; 90000];
    {
        metric
            .pairwise_into(
                &points,
                &points,
                options,
                &mut output,
                &mut workspace,
                ExecutionContext::new(&InlinePartitions),
            )
            .unwrap();
        metric
            .condensed_into(
                &points,
                options,
                &mut output,
                &mut workspace,
                ExecutionContext::new(&InlinePartitions),
            )
            .unwrap();
    };
}

#[test]
fn plain_point_sets_copy_layout_metadata_without_allocation() {
    fn check<S: scioxide_core::RealScalar>() {
        let data = [[S::ZERO, S::ONE], [S::ONE, S::ZERO]];
        {
            let array = PointSet::from_points(&data).unwrap();
            let flat = PointSet::from_rows(data.as_flattened(), 2).unwrap();
            let strided = {
                let shape = [2, 2];
                let strides = [2, 1];
                PointSet::new(
                    NdView::new(
                        data.as_flattened(),
                        StridedLayout::new(&shape, &strides).unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap()
            };
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(
                        array.coordinate(i, j).unwrap(),
                        flat.coordinate(i, j).unwrap()
                    );
                    assert_eq!(
                        array.coordinate(i, j).unwrap(),
                        strided.coordinate(i, j).unwrap()
                    );
                }
            }
        };
    }
    check::<f32>();
    check::<f64>();
}
