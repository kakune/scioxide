//! Independent brute-force search, approximation, and execution tests.
#![allow(clippy::unwrap_used)]
#[path = "support/search_cases.rs"]
mod search_cases;
use core::num::NonZeroUsize;
use scioxide_core::{
    CancellationToken, ExecutionContext, ExecutionPlan, ExecutionPolicy, Executor, NdView,
    PartitionFn, RealScalar, SciOxideError, SequentialExecutor, StridedLayout,
};
use scioxide_spatial::*;
use search_cases::{QueryCase, run_batch};
fn points<'a, S: RealScalar>(
    data: &'a [S],
    shape: &'a [usize; 2],
    strides: &'a [usize; 2],
) -> PointSet<'a, S> {
    PointSet::new(NdView::new(data, StridedLayout::new_broadcast(shape, strides).unwrap()).unwrap())
        .unwrap()
}
fn scalar<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
// Correctly rounded minimum image from exact rationals; `L - |a - b|` in
// floating point would cancel near the seam.
fn periodic_gap(a: f64, b: f64, length: f64) -> f64 {
    use num::{BigRational, Signed, ToPrimitive};
    let exact = |x: f64| BigRational::from_float(x).unwrap();
    let direct = (exact(a) - exact(b)).abs();
    let image = exact(length) - direct.clone();
    direct.min(image).to_f64().unwrap()
}
fn brute<S: RealScalar>(data: &[S], query: &[S], p: f64, periodic: bool) -> Vec<Neighbor<S>> {
    let mut result: Vec<_> = data
        .chunks_exact(query.len())
        .enumerate()
        .map(|(index, point)| {
            let diffs: Vec<_> = point
                .iter()
                .zip(query)
                .map(|(a, b)| {
                    if periodic {
                        periodic_gap(a.into_f64(), b.into_f64(), 1.0)
                    } else {
                        (a.into_f64() - b.into_f64()).abs()
                    }
                })
                .collect();
            let d = if p == f64::INFINITY {
                diffs.iter().copied().fold(0.0, f64::max)
            } else {
                diffs.iter().map(|d| d.powf(p)).sum::<f64>().powf(1.0 / p)
            };
            Neighbor {
                index,
                distance: scalar::<S>(d),
            }
        })
        .collect();
    result.sort_by(|a, b| {
        a.distance
            .into_f64()
            .total_cmp(&b.distance.into_f64())
            .then_with(|| a.index.cmp(&b.index))
    });
    result
}
fn reference_queries<S: RealScalar>() {
    for dimension in [1, 2, 3, 16, 48] {
        let shape = [127, dimension];
        let strides = [dimension, 1];
        let data: Vec<S> = (0..shape[0] * dimension)
            .map(|i| scalar((((i * 7919 + 17) % 65521) as f64) / 65521.0))
            .collect();
        let pts = points(&data, &shape, &strides);
        let lengths = vec![S::ONE; dimension];
        for (metric, p) in [
            (TreeMetric::Euclidean, 2.0),
            (TreeMetric::CityBlock, 1.0),
            (TreeMetric::Chebyshev, f64::INFINITY),
            (TreeMetric::Minkowski(3.5), 3.5),
        ] {
            for periodic in [false, true] {
                let tree = KdTree::with_options(
                    &pts,
                    metric,
                    KdTreeOptions::default()
                        .with_leaf_size(4)
                        .with_periodic_box(periodic.then_some(lengths.as_slice())),
                )
                .unwrap();
                for seed in 0..6 {
                    let query: Vec<S> = (0..dimension)
                        .map(|i| scalar((((i * 331 + seed * 109) % 997) as f64) / 997.0))
                        .collect();
                    let reference = brute(&data, &query, p, periodic);
                    for k in [0, 1, 7, 200] {
                        let actual = tree.nearest(&query, k).unwrap();
                        let count = k.min(shape[0]);
                        for (a, b) in actual.iter().zip(&reference[..count]) {
                            assert_eq!(
                                a.index, b.index,
                                "d={dimension}, p={p}, periodic={periodic}"
                            );
                            assert!(
                                (a.distance - b.distance).abs()
                                    <= scalar::<S>(64.0) * S::EPSILON * b.distance.max(S::ONE)
                            );
                        }
                        let approximate = configured_nearest(
                            &tree,
                            &query,
                            k,
                            SearchOptions::new().with_epsilon(0.25),
                        )
                        .unwrap();
                        assert_eq!(approximate.len(), count);
                        for (a, b) in approximate.iter().zip(&reference[..count]) {
                            assert!(a.distance <= b.distance * scalar(1.25001));
                        }
                    }
                    let radius = reference[20].distance;
                    let exact = tree.radius(&query, radius).unwrap();
                    let expected: Vec<_> = reference
                        .iter()
                        .filter(|n| n.distance < radius * (S::ONE - scalar::<S>(64.0) * S::EPSILON))
                        .map(|n| n.index)
                        .collect();
                    for index in expected {
                        assert!(exact.iter().any(|n| n.index == index));
                    }
                    assert!(exact.iter().all(|n| n.distance <= radius));
                    let approximate = configured_radius(
                        &tree,
                        &query,
                        radius,
                        SearchOptions::new().with_epsilon(0.25),
                    )
                    .unwrap();
                    for neighbor in &reference {
                        if neighbor.distance < radius / scalar(1.25001) {
                            assert!(approximate.iter().any(|n| n.index == neighbor.index));
                        }
                    }
                    assert!(
                        approximate
                            .iter()
                            .all(|n| n.distance <= radius * scalar(1.25))
                    );
                }
            }
        }
    }
}
#[test]
fn exhaustive_reference_f32() {
    reference_queries::<f32>();
}
#[test]
fn exhaustive_reference_f64() {
    reference_queries::<f64>();
}
// Seam images near a non-dyadic box length cancel in `L - |p - q|`. Reported
// distances must be the correctly rounded minimum image, and node bounds must
// never prune a point at its own reported distance.
#[test]
fn periodic_seam_images_are_correctly_rounded_and_never_pruned() {
    use num::{BigRational, Signed};
    let exact = |x: f64| BigRational::from_float(x).unwrap();
    let length = 10.0_f64;
    let lengths = [length];
    let mut state = 0x5eed_0426_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / 9007199254740992.
    };
    for _ in 0..256 {
        let (near_end, near_zero) = (length - (1.0 + next()) * 1e-4, next() * 1e-3);
        for (point, query) in [(near_end, near_zero), (near_zero, near_end)] {
            let data = [point, 0.5 * length, 0.25 * length];
            let shape = [3, 1];
            let strides = [1, 1];
            let pts = points(&data, &shape, &strides);
            let direct = (exact(point) - exact(query)).abs();
            let truth = direct.clone().min(exact(length) - direct);
            for leaf in [1, 16] {
                let tree = KdTree::with_options(
                    &pts,
                    TreeMetric::Euclidean,
                    KdTreeOptions::default()
                        .with_leaf_size(leaf)
                        .with_periodic_box(Some(&lengths)),
                )
                .unwrap();
                let nearest = tree.nearest(&[query], 1).unwrap();
                assert_eq!(nearest[0].index, 0);
                let distance = nearest[0].distance;
                let ulp = exact(f64::from_bits(distance.to_bits() + 1) - distance);
                assert!(
                    (exact(distance) - truth.clone()).abs() * BigRational::from_integer(2.into())
                        <= ulp
                );
                let within = tree.radius(&[query], distance).unwrap();
                assert_eq!(within.iter().map(|n| n.index).collect::<Vec<_>>(), [0]);
            }
        }
    }
    // The direct neighbor (8.000000000000001e-4) is nearer than the seam image
    // (8.00000000000611e-4); leaf size changes neither order nor distances.
    let data = [0.0011, 9.9995];
    let shape = [2, 1];
    let strides = [1, 1];
    let pts = points(&data, &shape, &strides);
    let results = [1, 16].map(|leaf| {
        let tree = KdTree::with_options(
            &pts,
            TreeMetric::Euclidean,
            KdTreeOptions::default()
                .with_leaf_size(leaf)
                .with_periodic_box(Some(&lengths)),
        )
        .unwrap();
        tree.nearest(&[0.0003], 2).unwrap()
    });
    assert_eq!(results[0], results[1]);
    assert_eq!(
        results[0].iter().map(|n| n.index).collect::<Vec<_>>(),
        [0, 1]
    );
    assert_eq!(results[0][1].distance, (10.0 - 9.9995) + 0.0003);
}
#[test]
fn ties_degeneracy_periodic_and_empty() {
    let data = [0.0_f64; 80];
    let shape = [40, 2];
    let strides = [2, 1];
    let pts = points(&data, &shape, &strides);
    let tree = KdTree::new(&pts).unwrap();
    assert_eq!(tree.node_count(), 1);
    assert_eq!(
        tree.nearest(&[0.0, 0.0], 4)
            .unwrap()
            .iter()
            .map(|n| n.index)
            .collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    assert_eq!(tree.radius(&[0.0, 0.0], 0.0).unwrap().len(), 40);
    let data = [0.01, 0.99, 0.5];
    let shape = [3, 1];
    let strides = [1, 1];
    let pts = points(&data, &shape, &strides);
    let lengths = [1.0];
    let tree = KdTree::with_options(
        &pts,
        TreeMetric::CityBlock,
        KdTreeOptions::default()
            .with_leaf_size(1)
            .with_periodic_box(Some(&lengths)),
    )
    .unwrap();
    assert_eq!(tree.radius(&[2.0], 0.02).unwrap().len(), 2);
    assert_eq!(tree.nearest(&[-0.01], 1).unwrap()[0].index, 1);
    let empty_shape = [0, 4];
    let empty_strides = [4, 1];
    let empty = points::<f64>(&[], &empty_shape, &empty_strides);
    let tree = KdTree::new(&empty).unwrap();
    assert!(tree.nearest(&[0.0; 4], 1).unwrap().is_empty());
    let zero_shape = [7, 0];
    let zero_strides = [0, 1];
    let zero = points::<f64>(&[], &zero_shape, &zero_strides);
    let tree = KdTree::new(&zero).unwrap();
    assert_eq!(
        tree.nearest(&[], 3)
            .unwrap()
            .iter()
            .map(|n| n.index)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
}
#[test]
fn rollback_and_memory_scaling() {
    for n in [32, 128, 512] {
        let data: Vec<f64> = (0..n * 3).map(|i| i as f64).collect();
        let shape = [n, 3];
        let strides = [3, 1];
        let pts = points(&data, &shape, &strides);
        let tree = KdTree::new(&pts).unwrap();
        assert!(tree.node_count() < 2 * n);
        assert!(tree.storage_bytes() < n * 256);
        let original = Neighbor {
            index: 999,
            distance: -1.0,
        };
        let mut output = [original; 4];
        let mut workspace = SearchWorkspace::new();
        assert!(
            tree.nearest_into(
                &[0.0; 3],
                4,
                SearchOptions::new(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(output, [original; 4]);
        workspace
            .try_reserve(
                &tree
                    .workspace_requirements(SearchWorkspaceRequest::new(n, SearchOptions::new()))
                    .unwrap(),
            )
            .unwrap();
        assert!(
            tree.radius_into(
                &[0.0; 3],
                1e9,
                SearchOptions::new(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(output, [original; 4]);
        let token = CancellationToken::new();
        token.cancel();
        assert!(
            tree.nearest_into(
                &[0.0; 3],
                4,
                SearchOptions::new(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential().with_cancellation(&token)
            )
            .is_err()
        );
        assert_eq!(output, [original; 4]);
        tree.nearest_into(
            &[0.0; 3],
            4,
            SearchOptions::new(),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(output[0].index, 0);
    }
}
struct Threads;
impl Executor for Threads {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        // Move each partition to a worker thread; the reference executor supplies
        // valid scope/report construction and deterministic partition order.
        SequentialExecutor.execute(context, plan, &|partition, (), scope| {
            std::thread::scope(|threads| {
                threads
                    .spawn(|| operation(partition, (), scope))
                    .join()
                    .unwrap()
            })
        })
    }
}
#[test]
fn batch_matches_scalar_and_cancellation_is_atomic() {
    let data: Vec<f64> = (0..600).map(|i| ((i * 7) % 41) as f64).collect();
    let shape = [300, 2];
    let stride = [2, 1];
    let pts = points(&data, &shape, &stride);
    let tree = KdTree::new(&pts).unwrap();
    let mut workspace = tree
        .workspace(
            SearchWorkspaceRequest::new(3, SearchOptions::new()).with_batch(
                300,
                300 * 3,
                NonZeroUsize::new(4).unwrap(),
            ),
        )
        .unwrap();
    let mut counts = vec![0; 300];
    let mut output = vec![Neighbor::default(); 900];
    run_batch(
        &tree,
        &pts,
        QueryCase::Nearest { k: 3 },
        SearchOptions::new().with_execution_policy(ExecutionPolicy::Auto),
        &mut output,
        &mut counts,
        &mut workspace,
        ExecutionContext::new(&Threads),
    )
    .unwrap();
    for (i, row) in data.chunks_exact(2).enumerate() {
        assert_eq!(counts[i], 3);
        assert_eq!(&output[i * 3..i * 3 + 3], tree.nearest(row, 3).unwrap());
    }
    let metric = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    let expected = metric.pairwise(&pts, &pts).unwrap().as_slice().to_vec();
    let mut distances = vec![0.0; 90000];
    let mut scratch = distance_workspace(90000);
    metric
        .pairwise_into(
            &pts,
            &pts,
            DistanceOptions::new().with_execution_policy(ExecutionPolicy::Auto),
            &mut distances,
            &mut scratch,
            ExecutionContext::new(&Threads),
        )
        .unwrap();
    assert_eq!(distances, expected);
    let token = CancellationToken::new();
    token.cancel();
    let saved = output.clone();
    let saved_counts = counts.clone();
    assert!(
        run_batch(
            &tree,
            &pts,
            QueryCase::Nearest { k: 3 },
            SearchOptions::new().with_execution_policy(ExecutionPolicy::Auto),
            &mut output,
            &mut counts,
            &mut workspace,
            ExecutionContext::new(&Threads).with_cancellation(&token)
        )
        .is_err()
    );
    assert_eq!(saved, output);
    assert_eq!(saved_counts, counts);
}

#[test]
fn distance_partitions_cross_rows_and_tiles_atomically() {
    let shape = [33, 3];
    let target_shape = [35, 3];
    let strides = [7, 2];
    let a: Vec<_> = (0..231).map(|i| ((i * 17) % 31) as f64).collect();
    let b: Vec<_> = (0..245).map(|i| ((i * 11) % 23) as f64).collect();
    let left = points(&a, &shape, &strides);
    let right = points(&b, &target_shape, &strides);
    let vi = [2.0, 0.5, 0.0, 0.5, 1.5, 0.25, 0.0, 0.25, 1.0];
    let vi_shape = [3, 3];
    let vi_strides = [3, 1];
    let metrics = [
        PreparedDistance::new(DistanceMetric::Euclidean).unwrap(),
        PreparedDistance::new(DistanceMetric::CityBlock).unwrap(),
        PreparedDistance::standardized(&[2.0, 3.0, 4.0]).unwrap(),
        PreparedDistance::mahalanobis(matrix_view(&vi, &vi_shape, &vi_strides)).unwrap(),
    ];
    for metric in &metrics {
        let expected = metric.pairwise(&left, &right).unwrap().as_slice().to_vec();
        let mut output = vec![42.0; 33 * 35 + 1];
        let mut scratch = distance_workspace(output.len());
        metric
            .pairwise_into(
                &left,
                &right,
                DistanceOptions::new().with_execution_policy(ExecutionPolicy::Parallel(
                    NonZeroUsize::new(4).unwrap(),
                )),
                &mut output,
                &mut scratch,
                ExecutionContext::new(&Threads),
            )
            .unwrap();
        assert_eq!(&output[..expected.len()], &expected);
        assert_eq!(output[expected.len()], 42.0);
        let token = CancellationToken::new();
        token.cancel();
        output.fill(42.0);
        assert!(
            metric
                .pairwise_into(
                    &left,
                    &right,
                    DistanceOptions::new().with_execution_policy(ExecutionPolicy::Auto),
                    &mut output,
                    &mut scratch,
                    ExecutionContext::new(&Threads).with_cancellation(&token)
                )
                .is_err()
        );
        assert!(output.iter().all(|&value| value == 42.0));
    }
    let overflow = [0.0, f64::MAX, -f64::MAX];
    let overflow_shape = [3, 1];
    let overflow_strides = [1, 1];
    let pts = points(&overflow, &overflow_shape, &overflow_strides);
    let mut output = [42.0; 9];
    let mut scratch = distance_workspace(9);
    assert!(
        metrics[0]
            .pairwise_into(
                &pts,
                &pts,
                DistanceOptions::new().with_execution_policy(ExecutionPolicy::Parallel(
                    NonZeroUsize::new(4).unwrap()
                )),
                &mut output,
                &mut scratch,
                ExecutionContext::new(&Threads)
            )
            .is_err()
    );
    assert_eq!(output, [42.0; 9]);
}

#[test]
fn invalid_tree_and_query_domains_are_typed() {
    let data = [0.0_f64, 1.0];
    let shape = [2, 1];
    let stride = [1, 1];
    let pts = points(&data, &shape, &stride);
    assert!(
        matches ! (& (KdTree :: with_options (& pts , TreeMetric :: Minkowski (0.5) , KdTreeOptions :: default ())) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
    );
    assert!(
        KdTree::with_options(
            &pts,
            TreeMetric::Euclidean,
            KdTreeOptions::default()
                .with_leaf_size(0)
                .with_periodic_box(None)
        )
        .is_err()
    );
    assert!(
        KdTree::with_options(
            &pts,
            TreeMetric::Euclidean,
            KdTreeOptions::default()
                .with_leaf_size(1)
                .with_periodic_box(Some(&[1.0]))
        )
        .is_err()
    );
    let tree = KdTree::new(&pts).unwrap();
    assert!(tree.nearest(&[], 1).is_err());
    assert!(tree.nearest(&[f64::NAN], 1).is_err());
    // The shared query preflight reports each public entry point it validates for.
    let invalid = |error: SciOxideError| {
        let scioxide_core::SciOxideErrorView::InvalidInput {
            argument: _view_argument,
            reason: _view_reason,
            detail: _view_detail,
        } = error.view()
        else {
            unreachable!("unexpected error kind")
        };
        _view_argument
    };
    assert_eq!(
        invalid(
            configured_nearest(&tree, &[0.0], 1, SearchOptions::new().with_epsilon(-0.1))
                .unwrap_err()
        ),
        scioxide_core::argument!("options.epsilon")
    );
    assert_eq!(
        invalid(
            configured_radius(&tree, &[0.0], 1.0, SearchOptions::new().with_epsilon(-0.1))
                .unwrap_err()
        ),
        scioxide_core::argument!("options.epsilon")
    );
    assert_eq!(
        invalid(tree.radius(&[0.0], -1.0).unwrap_err()),
        scioxide_core::argument!("radius")
    );
}
struct OmittingExecutor;
impl Executor for OmittingExecutor {
    fn worker_budget(&self) -> NonZeroUsize {
        NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        context: ExecutionContext<'_>,
        plan: ExecutionPlan,
        operation: &PartitionFn<'_>,
    ) -> Result<(), SciOxideError> {
        SequentialExecutor.execute(context, plan, &|partition, (), scope| {
            if partition.worker_index() == 0 {
                Ok(())
            } else {
                operation(partition, (), scope)
            }
        })
    }
}
#[test]
fn incomplete_executor_does_not_publish_and_batch_radius_matches() {
    let data = [0.0_f64, 1.0, 2.0, 3.0];
    let shape = [4, 1];
    let stride = [1, 1];
    let pts = points(&data, &shape, &stride);
    let metric = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    let mut out = [42.0; 16];
    let mut scratch = distance_workspace(16);
    assert!(
        metric
            .pairwise_into(
                &pts,
                &pts,
                DistanceOptions::new().with_execution_policy(ExecutionPolicy::Parallel(
                    NonZeroUsize::new(4).unwrap()
                )),
                &mut out,
                &mut scratch,
                ExecutionContext::new(&OmittingExecutor)
            )
            .is_err()
    );
    assert_eq!(out, [42.0; 16]);
    let tree = KdTree::new(&pts).unwrap();
    let mut workspace = tree
        .workspace(
            SearchWorkspaceRequest::new(4, SearchOptions::new()).with_batch(
                4,
                4 * 4,
                NonZeroUsize::new(4).unwrap(),
            ),
        )
        .unwrap();
    let mut out = [Neighbor::default(); 16];
    let mut counts = [0; 4];
    let total = run_batch(
        &tree,
        &pts,
        QueryCase::Radius { radius: 1.0 },
        SearchOptions::new()
            .with_execution_policy(ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap())),
        &mut out,
        &mut counts,
        &mut workspace,
        ExecutionContext::new(&Threads),
    )
    .unwrap();
    assert_eq!(counts, [2, 3, 3, 2]);
    assert_eq!(total, 10);
    let mut offset = 0;
    for (row, &count) in counts.iter().enumerate() {
        assert_eq!(
            &out[offset..offset + count],
            tree.radius(&[row as f64], 1.0).unwrap()
        );
        offset += count;
    }
}

#[test]
fn empty_operations_validate_static_configuration() {
    let shape = [0, 2];
    let strides = [2, 1];
    let pts = points::<f64>(&[], &shape, &strides);
    let metric = PreparedDistance::standardized(&[1.0]).unwrap();
    assert!(
        metric
            .pairwise_into(
                &pts,
                &pts,
                DistanceOptions::new(),
                &mut [],
                &mut DistanceWorkspace::new(),
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
    );
    assert!(
        metric
            .condensed_into(
                &pts,
                DistanceOptions::new(),
                &mut [],
                &mut DistanceWorkspace::new(),
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
    );
    assert!(
        metric
            .pairwise_into(
                &pts,
                &pts,
                DistanceOptions::new().with_execution_policy(ExecutionPolicy::Sequential),
                &mut [],
                &mut DistanceWorkspace::new(),
                ExecutionContext::sequential()
            )
            .is_err()
    );
    let tree = KdTree::new(&pts).unwrap();
    for (kind, options) in [
        (
            QueryCase::Nearest { k: 0 },
            SearchOptions::new().with_epsilon(-1.0),
        ),
        (QueryCase::Radius { radius: -1.0 }, SearchOptions::new()),
    ] {
        assert!(
            run_batch(
                &tree,
                &pts,
                kind,
                options,
                &mut [],
                &mut [],
                &mut SearchWorkspace::new(),
                ExecutionContext::sequential()
            )
            .is_err()
        );
    }
}

fn tree_metrics() -> [(TreeMetric, f64); 6] {
    [
        (TreeMetric::Euclidean, 2.0),
        (TreeMetric::CityBlock, 1.0),
        (TreeMetric::Chebyshev, f64::INFINITY),
        (TreeMetric::Minkowski(1.0), 1.0),
        (TreeMetric::Minkowski(2.0), 2.0),
        (TreeMetric::Minkowski(3.5), 3.5),
    ]
}

fn assert_neighbors<S: RealScalar>(actual: &[Neighbor<S>], expected: &[Neighbor<S>]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.index, b.index);
        assert!(
            (a.distance - b.distance).abs()
                <= scalar::<S>(128.0) * S::EPSILON * b.distance.max(S::ONE)
        );
    }
}

fn split_ties<S: RealScalar>() {
    for periodic in [false, true] {
        let center = if periodic { 0.5 } else { 0.0 };
        // Repeated cardinal points reside in different subtrees with leaf size 1.
        let data: Vec<S> = (0..48)
            .flat_map(|i| {
                let offset = [[0.25, 0.0], [0.0, -0.25], [-0.25, 0.0], [0.0, 0.25]][(i * 7) % 4];
                offset.map(|x| scalar(center + x))
            })
            .collect();
        let shape = [48, 2];
        let strides = [2, 1];
        let pts = points(&data, &shape, &strides);
        for (metric, _) in tree_metrics() {
            let lengths = [S::ONE; 2];
            let tree = KdTree::with_options(
                &pts,
                metric,
                KdTreeOptions::default()
                    .with_leaf_size(1)
                    .with_periodic_box(periodic.then_some(&lengths)),
            )
            .unwrap();
            for k in [1, 7, 24, 48, 60] {
                let expected: Vec<_> = (0..k.min(48))
                    .map(|index| Neighbor {
                        index,
                        distance: scalar(0.25),
                    })
                    .collect();
                assert_neighbors(&tree.nearest(&[scalar(center); 2], k).unwrap(), &expected);
            }
            let expected: Vec<_> = (0..48)
                .map(|index| Neighbor {
                    index,
                    distance: scalar(0.25),
                })
                .collect();
            assert_neighbors(
                &tree.radius(&[scalar(center); 2], scalar(0.25)).unwrap(),
                &expected,
            );
            assert!(
                tree.radius(&[scalar(center); 2], scalar(0.24))
                    .unwrap()
                    .is_empty()
            );
        }
    }
}

#[test]
fn equal_distance_ties_cross_subtrees_at_every_metric() {
    split_ties::<f32>();
    split_ties::<f64>();
}

#[test]
fn candidate_cutoff_preserves_destination_rounded_ties_and_radius() {
    for unit in [1.0_f32, f32::MIN_POSITIVE, f32::from_bits(1), f32::MAX] {
        let offset = (unit / 8192.0).max(f32::from_bits(1));
        // The negative point is visited first. The positive point has a larger
        // unrounded norm but the same f32 distance, and wins by original index.
        let data = [unit, offset, -unit, 0.0];
        let pts = points(&data, &[2, 2], &[2, 1]);
        let tree = KdTree::with_options(
            &pts,
            TreeMetric::Euclidean,
            KdTreeOptions::default().with_leaf_size(1),
        )
        .unwrap();
        let expected = [
            Neighbor {
                index: 0,
                distance: unit,
            },
            Neighbor {
                index: 1,
                distance: unit,
            },
        ];
        assert_eq!(tree.nearest(&[0.0; 2], 1).unwrap(), expected[..1]);
        assert_eq!(tree.radius(&[0.0; 2], unit).unwrap(), expected);
    }
}

#[test]
fn squared_cutoff_preserves_binary64_root_rounding_ties() {
    // The positive point has a larger square sum, but sqrt rounds both norms
    // to one. It must win by index even if the negative point fills the heap.
    let data = [1.0, f64::EPSILON.sqrt(), -1.0, 0.0];
    let pts = points(&data, &[2, 2], &[2, 1]);
    for leaf_size in [1, 2] {
        let tree = KdTree::with_options(
            &pts,
            TreeMetric::Euclidean,
            KdTreeOptions::default().with_leaf_size(leaf_size),
        )
        .unwrap();
        let expected = [
            Neighbor {
                index: 0,
                distance: 1.0,
            },
            Neighbor {
                index: 1,
                distance: 1.0,
            },
        ];
        assert_eq!(tree.nearest(&[0.0; 2], 1).unwrap(), expected[..1]);
        assert_eq!(tree.radius(&[0.0; 2], 1.0).unwrap(), expected);
        assert!(
            tree.radius(&[0.0; 2], 1_f64.next_down())
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn leaf_norms_match_distance_kernel_bits_across_square_ranges() {
    leaf_norm_bits::<f32>();
    leaf_norm_bits::<f64>();
}

fn leaf_norm_bits<S: RealScalar>() {
    let small = if S::MAX.into_f64() == f64::MAX {
        1e-200
    } else {
        1e-30
    };
    let large = 1.0 / small;
    for unit in [
        S::ZERO,
        S::MIN_SUBNORMAL,
        S::MIN_POSITIVE,
        scalar(small),
        scalar(f64::MIN_POSITIVE.sqrt()),
        S::ONE,
        scalar(large),
        S::MAX / scalar(4.0),
    ] {
        let half = unit / scalar(2.0);
        let data = [
            unit,
            half,
            -unit,
            S::ZERO,
            half,
            half,
            -unit,
            half,
            S::ZERO,
            S::ZERO,
        ];
        let pts = points(&data, &[5, 2], &[2, 1]);
        for (tree_metric, distance_metric) in [
            (TreeMetric::Euclidean, DistanceMetric::Euclidean),
            (TreeMetric::Minkowski(2.0), DistanceMetric::Minkowski(2.0)),
            (TreeMetric::Minkowski(3.5), DistanceMetric::Minkowski(3.5)),
        ] {
            let metric = PreparedDistance::new(distance_metric).unwrap();
            let mut expected: Vec<_> = data
                .chunks_exact(2)
                .enumerate()
                .map(|(index, point)| Neighbor {
                    index,
                    distance: metric.distance(point, &[S::ZERO; 2]).unwrap(),
                })
                .collect();
            expected.sort_by(|a, b| {
                a.distance
                    .into_f64()
                    .total_cmp(&b.distance.into_f64())
                    .then(a.index.cmp(&b.index))
            });
            for leaf_size in [1, 5] {
                let tree = KdTree::with_options(
                    &pts,
                    tree_metric,
                    KdTreeOptions::default().with_leaf_size(leaf_size),
                )
                .unwrap();
                for count in [1, 3, 5] {
                    assert_eq!(
                        tree.nearest(&[S::ZERO; 2], count).unwrap(),
                        expected[..count]
                    );
                }
                for neighbor in &expected {
                    let radius = neighbor.distance;
                    let within: Vec<_> = expected
                        .iter()
                        .copied()
                        .filter(|value| value.distance <= radius)
                        .collect();
                    assert_eq!(tree.radius(&[S::ZERO; 2], radius).unwrap(), within);
                }
            }
        }
    }
}

#[test]
fn distance_range_results_are_independent_of_tree_configuration() {
    range_configurations::<f32>();
    range_configurations::<f64>();
}

// Independent full scan: compute every norm, round representable distances,
// then select. No tree bounds, leaf order, or candidate cutoffs are involved.
fn range_reference<S: RealScalar>(
    data: &[S],
    query: &[S],
    p: f64,
    periodic: bool,
    kind: QueryCase<S>,
) -> Result<Vec<Neighbor<S>>, scioxide_core::SciOxideErrorKind> {
    let mut neighbors = Vec::new();
    let mut range_error = None;
    for (index, point) in data.chunks_exact(query.len()).enumerate() {
        let gaps: Vec<_> = point
            .iter()
            .zip(query)
            .map(|(point, query)| {
                let length = S::MAX.into_f64();
                let query = if periodic {
                    let wrapped = query.into_f64().rem_euclid(length);
                    if wrapped == length { 0.0 } else { wrapped }
                } else {
                    query.into_f64()
                };
                let gap = (point.into_f64() - query).abs();
                if periodic { gap.min(length - gap) } else { gap }
            })
            .collect();
        let distance = if p == f64::INFINITY {
            gaps.iter().copied().fold(0.0, f64::max)
        } else if p == 1.0 {
            gaps.iter().sum()
        } else if p == 2.0 {
            gaps.iter().copied().fold(0.0, f64::hypot)
        } else {
            let scale = gaps.iter().copied().fold(0.0, f64::max);
            if scale == 0.0 || !scale.is_finite() {
                scale
            } else {
                scale
                    * gaps
                        .iter()
                        .map(|gap| (gap / scale).powf(p))
                        .sum::<f64>()
                        .powf(1.0 / p)
            }
        };
        if !distance.is_finite() {
            range_error.get_or_insert(scioxide_core::SciOxideErrorKind::NumericalInstability);
        } else {
            match S::try_from_f64_lossy(distance) {
                Ok(distance) => neighbors.push(Neighbor { index, distance }),
                Err(error) => {
                    range_error.get_or_insert(error.kind());
                }
            }
        }
    }
    neighbors.sort_by(|a, b| {
        a.distance
            .into_f64()
            .total_cmp(&b.distance.into_f64())
            .then(a.index.cmp(&b.index))
    });
    match kind {
        QueryCase::Nearest { k } => {
            let count = k.min(data.len() / query.len());
            if neighbors.len() < count {
                return Err(range_error.unwrap());
            }
            neighbors.truncate(count);
        }
        QueryCase::Radius { radius } => neighbors.retain(|n| n.distance <= radius),
    }
    Ok(neighbors)
}

fn range_configurations<S: RealScalar>() {
    let large = scalar::<S>(if S::MAX.into_f64() == f64::MAX {
        1e308
    } else {
        1e38
    });
    check_range_configurations(&[-large, S::ZERO, large], 1, &[-large, large], false);
    for n in [3, 19] {
        for offset in 0..3 {
            let data: Vec<_> = (0..n)
                .map(|i| [-S::MAX, S::ZERO, S::MAX][(i + offset) % 3])
                .collect();
            check_range_configurations(&data, 1, &[-S::MAX, S::ZERO, S::MAX], false);
        }
    }
    // Periodic coordinate gaps fit, but several metric norms exceed the scalar
    // range. Reverse the points to encounter the losing norm first in a leaf.
    for periodic in [false, true] {
        for reverse in [false, true] {
            let half = S::MAX / scalar(2.0);
            let mut data = vec![S::ZERO; 16];
            data.extend([half; 16]);
            if reverse {
                data.reverse();
            }
            let mut queries = vec![S::ZERO; 16];
            queries.extend([half; 16]);
            check_range_configurations(&data, 16, &queries, periodic);
        }
    }
}

fn check_range_configurations<S: RealScalar>(
    data: &[S],
    dimension: usize,
    queries: &[S],
    periodic: bool,
) {
    let n = data.len() / dimension;
    let query_count = queries.len() / dimension;
    let shape = [n, dimension];
    let query_shape = [query_count, dimension];
    let strides = [dimension, 1];
    let points = points(data, &shape, &strides);
    let query_points = self::points(queries, &query_shape, &strides);
    let lengths = vec![S::MAX; dimension];
    let saved = Neighbor {
        index: usize::MAX,
        distance: -S::ONE,
    };
    for (metric, p) in tree_metrics() {
        for leaf_size in [1, 2, 16, n] {
            let tree = KdTree::with_options(
                &points,
                metric,
                KdTreeOptions::default()
                    .with_leaf_size(leaf_size)
                    .with_periodic_box(periodic.then_some(&lengths)),
            )
            .unwrap();
            let options = SearchOptions::new();
            let mut workspace = tree
                .workspace(SearchWorkspaceRequest::new(n, options).with_batch(
                    query_count,
                    query_count * n,
                    NonZeroUsize::new(4).unwrap(),
                ))
                .unwrap();
            for kind in [
                QueryCase::Nearest { k: 0 },
                QueryCase::Nearest { k: 1 },
                QueryCase::Nearest { k: 2 },
                QueryCase::Nearest { k: n + 1 },
                QueryCase::Radius { radius: S::ZERO },
                QueryCase::Radius { radius: S::MAX },
            ] {
                let expected: Vec<_> = queries
                    .chunks_exact(dimension)
                    .map(|query| range_reference(data, query, p, periodic, kind))
                    .collect();
                for (query, expected) in queries.chunks_exact(dimension).zip(&expected) {
                    let owned = match kind {
                        QueryCase::Nearest { k } => configured_nearest(&tree, query, k, options),
                        QueryCase::Radius { radius } => {
                            configured_radius(&tree, query, radius, options)
                        }
                    };
                    assert_eq!(
                        owned.as_ref().err().map(SciOxideError::kind),
                        expected.as_ref().err().copied(),
                        "metric={metric:?}, leaf={leaf_size}"
                    );
                    if let (Ok(actual), Ok(expected)) = (owned, expected) {
                        assert_neighbors(&actual, expected);
                    }
                    let mut output = vec![saved; n + 1];
                    let result = match kind {
                        QueryCase::Nearest { k } => tree.nearest_into(
                            query,
                            k,
                            options,
                            &mut output,
                            &mut workspace,
                            ExecutionContext::sequential(),
                        ),
                        QueryCase::Radius { radius } => tree.radius_into(
                            query,
                            radius,
                            options,
                            &mut output,
                            &mut workspace,
                            ExecutionContext::sequential(),
                        ),
                    };
                    assert_eq!(
                        result.as_ref().err().map(SciOxideError::kind),
                        expected.as_ref().err().copied()
                    );
                    if let Ok(count) = result {
                        assert_neighbors(&output[..count], expected.as_ref().unwrap());
                        assert!(output[count..].iter().all(|n| *n == saved));
                    } else {
                        assert!(output.iter().all(|n| *n == saved));
                    }
                }
                for policy in [
                    ExecutionPolicy::Sequential,
                    ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap()),
                ] {
                    let mut output = vec![saved; query_count * n + 1];
                    let mut counts = vec![usize::MAX; query_count + 1];
                    let result = run_batch(
                        &tree,
                        &query_points,
                        kind,
                        options.with_execution_policy(policy),
                        &mut output,
                        &mut counts,
                        &mut workspace,
                        ExecutionContext::new(&Threads),
                    );
                    if let Some(error) = expected.iter().find_map(|row| row.as_ref().err()) {
                        assert_eq!(result.unwrap_err().kind(), *error);
                        assert!(output.iter().all(|n| *n == saved));
                        assert!(counts.iter().all(|n| *n == usize::MAX));
                    } else {
                        let total = result.unwrap();
                        let expected: Vec<_> =
                            expected.iter().map(|row| row.as_ref().unwrap()).collect();
                        assert_eq!(
                            counts[..query_count],
                            expected.iter().map(|row| row.len()).collect::<Vec<_>>()
                        );
                        assert_eq!(counts[query_count], usize::MAX);
                        assert_neighbors(
                            &output[..total],
                            &expected.into_iter().flatten().copied().collect::<Vec<_>>(),
                        );
                        assert!(output[total..].iter().all(|n| *n == saved));
                    }
                }
            }
        }
    }
}

fn layout_reference<S: RealScalar>() {
    let shape = [63, 3];
    let data: Vec<S> = (0..441)
        .map(|i| scalar(((i * 7919 + 137) % 65521) as f64 / 65536.0))
        .collect();
    for strides in [[3, 1], [7, 2], [0, 1], [1, 0]] {
        let pts = points(&data, &shape, &strides);
        let logical: Vec<_> = (0..63)
            .flat_map(|row| {
                let data = &data;
                (0..3).map(move |axis| data[row * strides[0] + axis * strides[1]])
            })
            .collect();
        for (metric, p) in tree_metrics() {
            let tree = KdTree::with_options(
                &pts,
                metric,
                KdTreeOptions::default()
                    .with_leaf_size(2)
                    .with_periodic_box(None),
            )
            .unwrap();
            for query in [[0.13, 0.61, 0.79], [-0.25, 1.25, 0.31]] {
                let query = query.map(scalar::<S>);
                let expected = brute(&logical, &query, p, false);
                assert_neighbors(&tree.nearest(&query, 17).unwrap(), &expected[..17]);
                let radius = (expected[16].distance + expected[17].distance) / scalar(2.0);
                let within: Vec<_> = expected
                    .iter()
                    .copied()
                    .filter(|n| n.distance <= radius)
                    .collect();
                assert_neighbors(&tree.radius(&query, radius).unwrap(), &within);
            }
        }
    }
}

#[test]
fn exact_brute_force_for_contiguous_strided_and_broadcast_points() {
    layout_reference::<f32>();
    layout_reference::<f64>();
}

#[test]
fn periodic_unequal_lengths_and_translated_queries_match_brute_force() {
    let lengths = [2.0_f64, 4.0, 8.0];
    let data: Vec<_> = (0..93)
        .map(|i| ((i * 137 + 17) % 1024) as f64 / 1024.0 * lengths[i % 3])
        .collect();
    let shape = [31, 3];
    let strides = [3, 1];
    let pts = points(&data, &shape, &strides);
    for (metric, p) in tree_metrics() {
        let tree = KdTree::with_options(
            &pts,
            metric,
            KdTreeOptions::default()
                .with_leaf_size(1)
                .with_periodic_box(Some(&lengths)),
        )
        .unwrap();
        for query in [
            [-4.25, 12.5, -8.125],
            [8.0, -12.0, 24.0],
            [-f64::from_bits(1), 4.0, 8.0],
        ] {
            let wrapped = core::array::from_fn::<_, 3, _>(|axis| {
                let x = query[axis].rem_euclid(lengths[axis]);
                if x == lengths[axis] { 0.0 } else { x }
            });
            let mut expected: Vec<_> = data
                .chunks_exact(3)
                .enumerate()
                .map(|(index, point)| {
                    let deltas = core::array::from_fn::<_, 3, _>(|axis| {
                        let d = (point[axis] - wrapped[axis]).abs();
                        d.min(lengths[axis] - d)
                    });
                    let distance = if p == f64::INFINITY {
                        deltas.into_iter().fold(0.0, f64::max)
                    } else {
                        deltas
                            .into_iter()
                            .map(|x| x.powf(p))
                            .sum::<f64>()
                            .powf(1.0 / p)
                    };
                    Neighbor { index, distance }
                })
                .collect();
            expected.sort_by(|a, b| {
                a.distance
                    .total_cmp(&b.distance)
                    .then(a.index.cmp(&b.index))
            });
            assert_neighbors(&tree.nearest(&query, 9).unwrap(), &expected[..9]);
            let radius = (expected[8].distance + expected[9].distance) * 0.5;
            expected.retain(|n| n.distance <= radius);
            assert_neighbors(&tree.radius(&query, radius).unwrap(), &expected);
        }
    }
}

#[test]
fn extreme_distances_and_workspace_failure_recovery() {
    for (metric, _) in tree_metrics() {
        for unit in [f64::from_bits(1), 1e-200, 1e200] {
            let data = [unit, 0.0, 0.0, unit * 2.0, unit * 4.0, 0.0];
            let shape = [3, 2];
            let strides = [2, 1];
            let pts = points(&data, &shape, &strides);
            let tree = KdTree::with_options(
                &pts,
                metric,
                KdTreeOptions::default()
                    .with_leaf_size(1)
                    .with_periodic_box(None),
            )
            .unwrap();
            let actual = tree.nearest(&[0.0, 0.0], 3).unwrap();
            assert_eq!(
                actual.iter().map(|n| n.index).collect::<Vec<_>>(),
                [0, 1, 2]
            );
            for (neighbor, factor) in actual.iter().zip([1.0, 2.0, 4.0]) {
                let expected = unit * factor;
                assert!(
                    (neighbor.distance - expected).abs()
                        <= expected * 128.0 * f64::EPSILON + f64::from_bits(1)
                );
            }
            assert_eq!(tree.radius(&[0.0, 0.0], unit * 2.5).unwrap().len(), 2);
        }
        let data = [f64::MAX, 0.0, 0.0, 0.0];
        let shape = [2, 2];
        let strides = [2, 1];
        let pts = points(&data, &shape, &strides);
        let tree = KdTree::with_options(&pts, metric, KdTreeOptions::default()).unwrap();
        let saved = [Neighbor {
            index: 99,
            distance: -1.0,
        }; 2];
        let mut output = saved;
        let mut workspace = SearchWorkspace::new();
        workspace
            .try_reserve(
                &tree
                    .workspace_requirements(SearchWorkspaceRequest::new(0, SearchOptions::new()))
                    .unwrap(),
            )
            .unwrap();
        assert!(
            tree.nearest_into(
                &[0.0; 2],
                2,
                SearchOptions::new(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(output, saved);
        workspace
            .try_reserve(
                &tree
                    .workspace_requirements(SearchWorkspaceRequest::new(2, SearchOptions::new()))
                    .unwrap(),
            )
            .unwrap();
        assert!(
            tree.nearest_into(
                &[-f64::MAX, 0.0],
                2,
                SearchOptions::new(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(output, saved);
        assert_eq!(
            tree.radius_into(
                &[-f64::MAX, 0.0],
                f64::MAX,
                SearchOptions::new(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap(),
            1
        );
        assert_eq!(
            output,
            [
                Neighbor {
                    index: 1,
                    distance: f64::MAX
                },
                saved[1]
            ]
        );
        assert_eq!(
            tree.nearest_into(
                &[0.0; 2],
                1,
                SearchOptions::new(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap(),
            1
        );
        assert_eq!(
            output[0],
            Neighbor {
                index: 1,
                distance: 0.0
            }
        );
    }
}

struct CancellingClock<'a> {
    reads: core::sync::atomic::AtomicUsize,
    token: &'a CancellationToken,
}

#[test]
fn distance_cancellation_after_work_preserves_output_and_reuse() {
    let data: Vec<_> = (0..99 * 3).map(|i| i as f64 / 17.0).collect();
    let shape = [99, 3];
    let strides = [3, 1];
    let pts = points(&data, &shape, &strides);
    let metric = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    let mut workspace = distance_workspace(99 * 99);
    let mut out = vec![-1.0; 99 * 99 + 1];
    for condensed in [false, true] {
        let token = CancellationToken::new();
        let clock = CancellingClock {
            reads: core::sync::atomic::AtomicUsize::new(0),
            token: &token,
        };
        let binding = scioxide_core::ClockBinding::new(&clock);
        let context = ExecutionContext::sequential()
            .with_cancellation(&token)
            .with_deadline(scioxide_core::Deadline::at(
                &binding,
                scioxide_core::MonotonicInstant::from_duration_since_epoch(
                    core::time::Duration::from_secs(1),
                ),
            ))
            .unwrap();
        out.fill(-1.0);
        let run = |out: &mut [f64], workspace: &mut DistanceWorkspace<f64>, cx| {
            if condensed {
                metric.condensed_into(&pts, DistanceOptions::new(), out, workspace, cx)
            } else {
                metric.pairwise_into(&pts, &pts, DistanceOptions::new(), out, workspace, cx)
            }
        };
        assert_eq!(
            run(&mut out, &mut workspace, context).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::ExecutionStopped
        );
        assert!(out.iter().all(|&x| x == -1.0));
        run(&mut out, &mut workspace, ExecutionContext::sequential()).unwrap();
        assert!(out[0] >= 0.0);
        assert_eq!(out[99 * 99], -1.0);
    }
}
impl scioxide_core::Clock for CancellingClock<'_> {
    fn now(&self) -> Result<scioxide_core::MonotonicInstant, scioxide_core::CapabilityError> {
        if self
            .reads
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed)
            >= 5
        {
            self.token.cancel();
        }
        Ok(scioxide_core::MonotonicInstant::from_duration_since_epoch(
            core::time::Duration::ZERO,
        ))
    }
}

#[test]
fn cancellation_during_traversal_preserves_output_and_workspace_reuse() {
    let data: Vec<f64> = (0..2048).map(|i| i as f64).collect();
    let shape = [1024, 2];
    let strides = [2, 1];
    let pts = points(&data, &shape, &strides);
    let tree = KdTree::with_options(
        &pts,
        TreeMetric::Euclidean,
        KdTreeOptions::default()
            .with_leaf_size(1)
            .with_periodic_box(None),
    )
    .unwrap();
    let mut workspace = SearchWorkspace::new();
    workspace
        .try_reserve(
            &tree
                .workspace_requirements(SearchWorkspaceRequest::new(1024, SearchOptions::new()))
                .unwrap(),
        )
        .unwrap();
    let sentinel = Neighbor {
        index: 99,
        distance: -1.0,
    };
    let mut output = vec![sentinel; 1024];
    for radius in [false, true] {
        let token = CancellationToken::new();
        let clock = CancellingClock {
            reads: core::sync::atomic::AtomicUsize::new(0),
            token: &token,
        };
        let binding = scioxide_core::ClockBinding::new(&clock);
        let context = ExecutionContext::sequential()
            .with_cancellation(&token)
            .with_deadline(scioxide_core::Deadline::at(
                &binding,
                scioxide_core::MonotonicInstant::from_duration_since_epoch(
                    core::time::Duration::from_secs(1),
                ),
            ))
            .unwrap();
        output.fill(sentinel);
        let result = if radius {
            tree.radius_into(
                &[0.0; 2],
                10000.0,
                SearchOptions::new(),
                &mut output,
                &mut workspace,
                context,
            )
        } else {
            tree.nearest_into(
                &[0.0; 2],
                1024,
                SearchOptions::new(),
                &mut output,
                &mut workspace,
                context,
            )
        };
        assert!(
            result.is_err_and(|e| e.kind() == scioxide_core::SciOxideErrorKind::ExecutionStopped)
        );
        assert_eq!(output, vec![sentinel; 1024]);
        tree.nearest_into(
            &[0.0; 2],
            1024,
            SearchOptions::new(),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_neighbors(&output, &tree.nearest(&[0.0; 2], 1024).unwrap());
    }
}

#[test]
fn oversized_leaf_observes_cancellation_and_deadline_atomically() {
    struct ExpiringClock(core::sync::atomic::AtomicUsize);
    impl scioxide_core::Clock for ExpiringClock {
        fn now(&self) -> Result<scioxide_core::MonotonicInstant, scioxide_core::CapabilityError> {
            let reads = self.0.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            Ok(scioxide_core::MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::from_secs(u64::from(reads > 0)),
            ))
        }
    }
    let data = [0.0_f64; 2048];
    let pts = points(&data, &[2048, 1], &[1, 1]);
    let tree = KdTree::new(&pts).unwrap();
    assert_eq!(tree.node_count(), 1);
    let options = SearchOptions::new();
    let mut workspace = tree
        .workspace(SearchWorkspaceRequest::new(2048, options))
        .unwrap();
    let sentinel = Neighbor {
        index: 99,
        distance: -1.0,
    };
    for cancellation in [false, true] {
        for radius in [false, true] {
            let token = CancellationToken::new();
            let cancelling = CancellingClock {
                reads: core::sync::atomic::AtomicUsize::new(0),
                token: &token,
            };
            let expiring = ExpiringClock(core::sync::atomic::AtomicUsize::new(0));
            let clock: &dyn scioxide_core::Clock =
                if cancellation { &cancelling } else { &expiring };
            let binding = scioxide_core::ClockBinding::new(clock);
            let context = ExecutionContext::sequential()
                .with_cancellation(&token)
                .with_deadline(scioxide_core::Deadline::at(
                    &binding,
                    scioxide_core::MonotonicInstant::from_duration_since_epoch(
                        core::time::Duration::from_secs(1),
                    ),
                ))
                .unwrap();
            let mut output = vec![sentinel; 2048];
            let result = if radius {
                tree.radius_into(&[0.0], 0.0, options, &mut output, &mut workspace, context)
            } else {
                tree.nearest_into(&[0.0], 1, options, &mut output, &mut workspace, context)
            };
            assert_eq!(
                result.unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::ExecutionStopped
            );
            assert!(output.iter().all(|&n| n == sentinel));
            tree.nearest_into(
                &[0.0],
                1,
                options,
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
            assert_eq!(
                output[0],
                Neighbor {
                    index: 0,
                    distance: 0.0
                }
            );
            assert!(output[1..].iter().all(|&n| n == sentinel));
        }
    }
}

#[test]
fn batch_query_order_and_metrics_agree_with_brute_force() {
    let data: Vec<f64> = (0..93)
        .map(|i| ((i * 137 + 17) % 1024) as f64 / 1024.0)
        .collect();
    let shape = [31, 3];
    let strides = [3, 1];
    let pts = points(&data, &shape, &strides);
    let queries: Vec<_> = (0..7)
        .rev()
        .flat_map(|i| data[i * 3..i * 3 + 3].iter().copied())
        .collect();
    let query_shape = [7, 3];
    let query_pts = points(&queries, &query_shape, &strides);
    for (metric, p) in tree_metrics() {
        for periodic in [false, true] {
            let lengths = [1.0; 3];
            let tree = KdTree::with_options(
                &pts,
                metric,
                KdTreeOptions::default()
                    .with_leaf_size(2)
                    .with_periodic_box(periodic.then_some(&lengths)),
            )
            .unwrap();
            let mut workspace = tree
                .workspace(
                    SearchWorkspaceRequest::new(31, SearchOptions::new()).with_batch(
                        7,
                        7 * 31,
                        NonZeroUsize::new(4).unwrap(),
                    ),
                )
                .unwrap();
            for kind in [
                QueryCase::Nearest { k: 5 },
                QueryCase::Radius { radius: 0.3 },
            ] {
                let mut expected = Vec::new();
                let mut expected_counts = Vec::new();
                for query in queries.chunks_exact(3) {
                    let mut row = brute(&data, query, p, periodic);
                    match kind {
                        QueryCase::Nearest { k, .. } => row.truncate(k),
                        QueryCase::Radius { radius, .. } => row.retain(|n| n.distance <= radius),
                    }
                    expected_counts.push(row.len());
                    expected.extend(row);
                }
                for policy in [
                    ExecutionPolicy::Sequential,
                    ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap()),
                ] {
                    let sentinel = Neighbor {
                        index: 99,
                        distance: -1.0,
                    };
                    let mut output = vec![sentinel; 217];
                    let mut counts = [99; 7];
                    let total = run_batch(
                        &tree,
                        &query_pts,
                        kind,
                        SearchOptions::new().with_execution_policy(policy),
                        &mut output,
                        &mut counts,
                        &mut workspace,
                        ExecutionContext::new(&Threads),
                    )
                    .unwrap();
                    assert_eq!(counts.as_slice(), expected_counts);
                    assert_eq!(total, expected.len());
                    assert_neighbors(&output[..total], &expected);
                    assert!(output[total..].iter().all(|n| *n == sentinel));
                }
            }
        }
    }
}

#[test]
fn batches_reuse_finite_strided_and_broadcast_query_rows() {
    let data: Vec<f32> = (0..93)
        .map(|i| ((i * 137 + 17) % 1024) as f32 / 1024.0)
        .collect();
    let pts = points(&data, &[31, 3], &[3, 1]);
    for strides in [[3, 1], [7, 2], [0, 1], [1, 0]] {
        let queries = points(&data, &[7, 3], &strides);
        for (metric, _) in tree_metrics() {
            for periodic in [false, true] {
                let tree = KdTree::with_options(
                    &pts,
                    metric,
                    KdTreeOptions::default()
                        .with_leaf_size(2)
                        .with_periodic_box(periodic.then_some(&[1.0; 3])),
                )
                .unwrap();
                let mut workspace = tree
                    .workspace(
                        SearchWorkspaceRequest::new(5, SearchOptions::new()).with_batch(
                            7,
                            7 * 31,
                            NonZeroUsize::new(4).unwrap(),
                        ),
                    )
                    .unwrap();
                for kind in [
                    QueryCase::Nearest { k: 5 },
                    QueryCase::Radius { radius: 0.3 },
                ] {
                    let expected: Vec<_> = (0..7)
                        .map(|row| {
                            let query = core::array::from_fn::<_, 3, _>(|axis| {
                                queries.coordinate(row, axis).unwrap()
                            });
                            match kind {
                                QueryCase::Nearest { k } => tree.nearest(&query, k).unwrap(),
                                QueryCase::Radius { radius } => {
                                    tree.radius(&query, radius).unwrap()
                                }
                            }
                        })
                        .collect();
                    for policy in [
                        ExecutionPolicy::Sequential,
                        ExecutionPolicy::Parallel(NonZeroUsize::new(4).unwrap()),
                    ] {
                        let mut output = vec![Neighbor::default(); 7 * 31];
                        let mut counts = [0; 7];
                        let total = run_batch(
                            &tree,
                            &queries,
                            kind,
                            SearchOptions::new().with_execution_policy(policy),
                            &mut output,
                            &mut counts,
                            &mut workspace,
                            ExecutionContext::new(&Threads),
                        )
                        .unwrap();
                        assert_eq!(counts, core::array::from_fn(|i| expected[i].len()));
                        assert_eq!(output[..total], expected.concat());
                    }
                }
            }
        }
    }
}

#[test]
fn unrepresentable_root_bounds_exclude_radius_matches_and_reservations_keep_storage() {
    let data = [f64::MAX, f64::MAX];
    let shape = [2, 1];
    let strides = [1, 1];
    let pts = points(&data, &shape, &strides);
    for (metric, _) in tree_metrics() {
        let tree = KdTree::with_options(&pts, metric, KdTreeOptions::default()).unwrap();
        let mut workspace = SearchWorkspace::new();
        workspace
            .try_reserve(
                &tree
                    .workspace_requirements(SearchWorkspaceRequest::new(2, SearchOptions::new()))
                    .unwrap(),
            )
            .unwrap();
        let limited = tree.workspace_requirements(SearchWorkspaceRequest::new(
            0,
            SearchOptions::new().with_max_workspace_bytes(0),
        ));
        assert!(limited.is_err());
        assert!(
            tree.workspace_requirements(
                SearchWorkspaceRequest::new(2, SearchOptions::new()).with_batch(
                    usize::MAX,
                    0,
                    NonZeroUsize::MIN
                )
            )
            .is_err()
        );
        let sentinel = Neighbor {
            index: 99,
            distance: -1.0,
        };
        let mut output = [sentinel; 2];
        let result = tree.radius_into(
            &[-f64::MAX],
            0.0,
            SearchOptions::new(),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        );
        assert_eq!(result.unwrap(), 0);
        assert_eq!(output, [sentinel; 2]);
        assert_eq!(
            tree.nearest_into(
                &[-f64::MAX],
                1,
                SearchOptions::new(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err()
            .kind(),
            scioxide_core::SciOxideErrorKind::NumericalInstability
        );
        assert_eq!(output, [sentinel; 2]);
        assert_eq!(
            tree.nearest_into(
                &[f64::MAX],
                2,
                SearchOptions::new(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap(),
            2
        );
        assert_eq!(
            output,
            [
                Neighbor {
                    index: 0,
                    distance: 0.0
                },
                Neighbor {
                    index: 1,
                    distance: 0.0
                }
            ]
        );
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

fn matrix_view<'a, S: scioxide_core::RealScalar>(
    data: &'a [S],
    shape: &[usize; 2],
    strides: &[usize; 2],
) -> scioxide_linalg::MatrixView<'a, S> {
    scioxide_linalg::MatrixView::from_slice_with_strides(
        shape[0], shape[1], data, strides[0], strides[1],
    )
    .unwrap()
}

// Coincident leaf points take an exact-zero path, while nonzero differences
// whose squares underflow keep the scaled evaluation; subnormal spreads still
// split; scalar queries do not reserve the batch query row.
#[test]
fn coincident_subnormal_and_scalar_storage_paths() {
    let tiny = 1e-200_f64;
    let data = [0.0, 0.0, 0.0, tiny, 0.0, 0.0];
    let shape = [6, 1];
    let strides = [1, 1];
    let pts = points(&data, &shape, &strides);
    let tree = KdTree::new(&pts).unwrap();
    let near = tree.nearest(&[0.0], 6).unwrap();
    assert_eq!(
        near.iter().map(|n| n.index).collect::<Vec<_>>(),
        [0, 1, 2, 4, 5, 3]
    );
    assert!(near[..5].iter().all(|n| n.distance == 0.0));
    assert_eq!(near[5].distance, tiny);
    let subnormal: Vec<f64> = (0..8).map(f64::from_bits).collect();
    let shape = [8, 1];
    let pts = points(&subnormal, &shape, &strides);
    let tree = KdTree::with_options(
        &pts,
        TreeMetric::Euclidean,
        KdTreeOptions::default().with_leaf_size(1),
    )
    .unwrap();
    assert_eq!(tree.node_count(), 15);
    let scalar = tree
        .workspace_requirements(SearchWorkspaceRequest::new(1, SearchOptions::new()))
        .unwrap();
    let batch = tree
        .workspace_requirements(
            SearchWorkspaceRequest::new(1, SearchOptions::new()).with_batch(
                1,
                1,
                NonZeroUsize::MIN,
            ),
        )
        .unwrap();
    assert_eq!(
        batch.worker_bytes::<f64>().unwrap() - scalar.worker_bytes::<f64>().unwrap(),
        size_of::<f64>()
    );
}

fn configured_nearest<S: RealScalar>(
    tree: &KdTree<'_, S>,
    query: &[S],
    value: usize,
    options: SearchOptions,
) -> Result<Vec<Neighbor<S>>, scioxide_core::SciOxideError> {
    let mut output = vec![Neighbor::default(); tree.len()];
    let mut workspace = tree.workspace(SearchWorkspaceRequest::new(tree.len(), options))?;
    let count = tree.nearest_into(
        query,
        value,
        options,
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    output.truncate(count);
    Ok(output)
}

fn configured_radius<S: RealScalar>(
    tree: &KdTree<'_, S>,
    query: &[S],
    value: S,
    options: SearchOptions,
) -> Result<Vec<Neighbor<S>>, scioxide_core::SciOxideError> {
    let mut output = vec![Neighbor::default(); tree.len()];
    let mut workspace = tree.workspace(SearchWorkspaceRequest::new(tree.len(), options))?;
    let count = tree.radius_into(
        query,
        value,
        options,
        &mut output,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    output.truncate(count);
    Ok(output)
}
