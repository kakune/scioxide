//! Deterministic bounds on actual search work between stop polls.
#![allow(clippy::unwrap_used)]

use super::*;
use alloc::vec;
use core::sync::atomic::{AtomicUsize, Ordering};
use scioxide_core::{Cancellation, CancellationToken, NdView, SciOxideErrorKind, StridedLayout};

// Request cancellation immediately after a successful observation, the worst
// placement relative to the next poll. No wall-clock timing or threads needed.
#[derive(Default)]
struct CancelAfterObservation {
    token: CancellationToken,
    polls: AtomicUsize,
}
impl Cancellation for CancelAfterObservation {
    fn is_cancelled(&self) -> bool {
        let cancelled = self.token.is_cancelled();
        self.polls.fetch_add(1, Ordering::Relaxed);
        self.token.cancel();
        cancelled
    }
}

#[test]
fn cancellation_is_observed_within_64_more_node_visits() {
    let data: Vec<_> = (0..1024).map(f64::from).collect();
    let points = PointSet::new(
        NdView::new(&data, StridedLayout::new(&[1024, 1], &[1, 1]).unwrap()).unwrap(),
    )
    .unwrap();
    let tree = KdTree::with_options(
        &points,
        TreeMetric::Euclidean,
        KdTreeOptions::default().with_leaf_size(1),
    )
    .unwrap();
    for kind in [
        Query::Nearest {
            count: 1024,
            factor: 1.0,
        },
        Query::Radius {
            radius: 1024.0,
            factor: 1.0,
        },
    ] {
        let cancellation = CancelAfterObservation::default();
        let mut poll =
            SearchPoll::new(ExecutionContext::sequential().with_cancellation(&cancellation));
        let mut axes = [Axis::default()];
        let mut candidates = vec![Neighbor::default(); 1024];
        let mut results = SearchState::new(&mut axes, &mut candidates);
        let error = tree
            .search(&[0.0], kind, &mut results, &mut poll)
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
        assert_eq!(cancellation.polls.load(Ordering::Relaxed), 2);
        // The first observation follows 64 visits; at most another 64 occur
        // after it requests cancellation. Count actual visited/pruned nodes.
        assert_eq!(poll.visits, 128);
        assert!(poll.points > 0);
        assert!(results.len < tree.len());
    }
}

#[test]
fn cancellation_is_observed_within_256_more_points_in_oversized_leaf() {
    let data = [0.0; 1024];
    let points = PointSet::new(
        NdView::new(&data, StridedLayout::new(&[1024, 1], &[1, 1]).unwrap()).unwrap(),
    )
    .unwrap();
    let tree = KdTree::new(&points).unwrap();
    assert_eq!(tree.node_count(), 1);
    let cancellation = CancelAfterObservation::default();
    let mut poll = SearchPoll::new(ExecutionContext::sequential().with_cancellation(&cancellation));
    let mut axes = [Axis::default()];
    let mut results = SearchState::count(&mut axes);
    let error = tree
        .search(
            &[0.0],
            Query::Radius {
                radius: 0.0,
                factor: 1.0,
            },
            &mut results,
            &mut poll,
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    assert_eq!(cancellation.polls.load(Ordering::Relaxed), 2);
    assert_eq!(poll.visits, 1);
    assert_eq!(poll.points, 512);
    assert_eq!(results.len, 512);
}

#[test]
fn polling_budget_survives_short_queries() {
    let points =
        PointSet::new(NdView::new(&[0.0], StridedLayout::new(&[1, 1], &[1, 1]).unwrap()).unwrap())
            .unwrap();
    let tree = KdTree::new(&points).unwrap();
    let cancellation = CancelAfterObservation::default();
    let mut poll = SearchPoll::new(ExecutionContext::sequential().with_cancellation(&cancellation));
    let mut axes = [Axis::default()];
    for _ in 0..128 {
        let mut results = SearchState::count(&mut axes);
        tree.search(
            &[0.0],
            Query::Radius {
                radius: 0.0,
                factor: 1.0,
            },
            &mut results,
            &mut poll,
        )
        .unwrap();
        assert_eq!(results.len, 1);
    }
    let mut results = SearchState::count(&mut axes);
    let error = tree
        .search(
            &[0.0],
            Query::Radius {
                radius: 0.0,
                factor: 1.0,
            },
            &mut results,
            &mut poll,
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    assert_eq!(poll.visits, 128);
    assert_eq!(poll.points, 128);
}
