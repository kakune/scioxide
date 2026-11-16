//! Indexed binary min-heap used by graph searches and orderings; it never grows its storage.
use alloc::vec::Vec;
use scioxide_core::RealScalar;

fn swap(heap: &mut [usize], positions: &mut [usize], a: usize, b: usize) {
    heap.swap(a, b);
    positions[heap[a]] = a;
    positions[heap[b]] = b;
}
fn up(
    heap: &mut [usize],
    positions: &mut [usize],
    mut i: usize,
    less: &impl Fn(usize, usize) -> bool,
) -> usize {
    while i > 0 {
        let parent = (i - 1) / 2;
        if !less(heap[i], heap[parent]) {
            break;
        }
        swap(heap, positions, i, parent);
        i = parent;
    }
    i
}
fn down(
    heap: &mut [usize],
    positions: &mut [usize],
    mut i: usize,
    less: &impl Fn(usize, usize) -> bool,
) {
    while i < heap.len() / 2 {
        let left = 2 * i + 1;
        let right = left + 1;
        let child = if right < heap.len() && less(heap[right], heap[left]) {
            right
        } else {
            left
        };
        if !less(heap[child], heap[i]) {
            break;
        }
        swap(heap, positions, i, child);
        i = child;
    }
}
/// Insert a vertex or repair either direction after changing its key.
/// The caller reserves space for every vertex before inserting.
pub(crate) fn update(
    heap: &mut Vec<usize>,
    positions: &mut [usize],
    vertex: usize,
    less: impl Fn(usize, usize) -> bool,
) {
    if positions[vertex] == usize::MAX {
        positions[vertex] = heap.len();
        heap.push(vertex);
    }
    let i = up(heap, positions, positions[vertex], &less);
    down(heap, positions, i, &less);
}
/// Remove a present vertex, leaving the absent-position sentinel.
pub(crate) fn remove(
    heap: &mut Vec<usize>,
    positions: &mut [usize],
    vertex: usize,
    less: impl Fn(usize, usize) -> bool,
) {
    let i = positions[vertex];
    if i == usize::MAX {
        return;
    }
    let Some(last) = heap.pop() else { return };
    positions[vertex] = usize::MAX;
    if i < heap.len() {
        heap[i] = last;
        positions[last] = i;
        let i = up(heap, positions, i, &less);
        down(heap, positions, i, &less);
    }
}
fn less<S: RealScalar>(u: usize, v: usize, distances: &[Option<S>]) -> bool {
    match (distances[u], distances[v]) {
        (Some(a), Some(b)) => a < b || (a == b && u < v),
        _ => false,
    }
}
pub(super) fn decrease<S: RealScalar>(
    heap: &mut Vec<usize>,
    positions: &mut [usize],
    distances: &[Option<S>],
    vertex: usize,
) {
    update(heap, positions, vertex, |u, v| less(u, v, distances));
}
pub(super) fn pop<S: RealScalar>(
    heap: &mut Vec<usize>,
    positions: &mut [usize],
    distances: &[Option<S>],
) -> Option<usize> {
    let vertex = *heap.first()?;
    remove(heap, positions, vertex, |u, v| less(u, v, distances));
    // Searches distinguish finalized vertices from vertices not yet reached.
    positions[vertex] = positions.len();
    Some(vertex)
}
