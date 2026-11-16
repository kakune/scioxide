//! Quotient-graph minimum-degree elimination with supervariable amalgamation.
//!
//! The graph represents eliminated cliques by elements; no fill edges are formed.
//! Degree bounds are the cardinalities of the reachable active supervariables.
//! The mathematical model follows Amestoy, Davis and Duff (1996), §§2--3.
//! This implementation owns separate adjacency lists and joins indistinguishable
//! variables by their canonical incidence lists.
use super::compressed::Graph;
use crate::graph::heap;
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{Permutation, SciOxideError};

const MEMORY: scioxide_core::Quantity = scioxide_core::quantity!("amd_workspace");
const NONE: usize = usize::MAX;

struct Quotient {
    rows: Vec<Vec<usize>>,
    active: Vec<bool>,
    elements: Vec<bool>,
    parent: Vec<usize>,
    weight: Vec<usize>,
    assembly: Vec<usize>,
    degree: Vec<usize>,
    marks: Vec<usize>,
    epoch: usize,
    heap: Vec<usize>,
    positions: Vec<usize>,
    capacity: usize,
    fixed_bytes: usize,
    byte_limit: usize,
}
impl Quotient {
    fn root(&self, mut vertex: usize) -> usize {
        while self.parent[vertex] != vertex {
            vertex = self.parent[vertex];
        }
        vertex
    }
    fn admit(&self, additional: usize) -> Result<(), SciOxideError> {
        let words = checked_add(self.capacity, additional, MEMORY)?;
        // Twice the live row capacity covers the row being replaced and the
        // two temporary frontier lists, each bounded by the graph's incidences.
        let bytes = checked_add(
            self.fixed_bytes,
            checked_mul(words, 2 * core::mem::size_of::<usize>(), MEMORY)?,
            MEMORY,
        )?;
        if bytes > self.byte_limit {
            return Err(SciOxideError::resource_limit(
                MEMORY,
                bytes,
                self.byte_limit,
            ));
        }
        Ok(())
    }
    fn replace(&mut self, vertex: usize, values: &[usize]) -> Result<(), SciOxideError> {
        if values.len() <= self.rows[vertex].capacity() {
            self.rows[vertex].clear();
            self.rows[vertex].extend_from_slice(values);
            return Ok(());
        }
        self.admit(values.len())?;
        let mut row = try_with_capacity(values.len(), MEMORY)?;
        row.extend_from_slice(values);
        self.capacity = self.capacity - self.rows[vertex].capacity() + row.capacity();
        self.rows[vertex] = row;
        Ok(())
    }
    fn clear(&mut self, vertex: usize) {
        self.capacity -= self.rows[vertex].capacity();
        self.rows[vertex] = Vec::new();
    }
    fn mark(&mut self, vertex: usize, output: &mut Vec<usize>) {
        let vertex = self.root(vertex);
        if self.active[vertex] && self.marks[vertex] != self.epoch {
            self.marks[vertex] = self.epoch;
            output.push(vertex);
        }
    }
    fn neighbors(&mut self, vertex: usize, output: &mut Vec<usize>) {
        output.clear();
        advance_epoch(&mut self.epoch, &mut self.marks);
        self.marks[vertex] = self.epoch;
        for index in 0..self.rows[vertex].len() {
            let adjacent = self.root(self.rows[vertex][index]);
            if self.active[adjacent] {
                self.mark(adjacent, output);
            } else if self.elements[adjacent] {
                for offset in 0..self.rows[adjacent].len() {
                    self.mark(self.rows[adjacent][offset], output);
                }
            }
        }
    }
    fn update(&mut self, vertex: usize) {
        heap::update(&mut self.heap, &mut self.positions, vertex, |a, b| {
            (self.degree[a], a) < (self.degree[b], b)
        });
    }
    fn remove(&mut self, vertex: usize) {
        heap::remove(&mut self.heap, &mut self.positions, vertex, |a, b| {
            (self.degree[a], a) < (self.degree[b], b)
        });
    }
}

pub(super) fn order(
    graph: Graph,
    n: usize,
    byte_limit: usize,
) -> Result<Permutation, SciOxideError> {
    let total = graph.len();
    let fixed_bytes = checked_mul(
        checked_add(total, 1, MEMORY)?,
        40 * core::mem::size_of::<usize>(),
        MEMORY,
    )?;
    let mut rows = try_with_capacity(total, MEMORY)?;
    let mut capacity = 0;
    for entries in graph.iter() {
        let mut row = try_with_capacity(entries.len(), MEMORY)?;
        row.extend_from_slice(entries);
        capacity += row.capacity();
        rows.push(row);
    }
    drop(graph);
    let mut q = Quotient {
        rows,
        active: try_filled(total, false, MEMORY)?,
        elements: try_filled(total, false, MEMORY)?,
        parent: try_with_capacity(total, MEMORY)?,
        weight: try_filled(total, 1usize, MEMORY)?,
        assembly: try_filled(total, NONE, MEMORY)?,
        degree: try_filled(total, 0usize, MEMORY)?,
        marks: try_filled(total, 0usize, MEMORY)?,
        epoch: 0,
        heap: try_with_capacity(n, MEMORY)?,
        positions: try_filled(total, NONE, MEMORY)?,
        capacity,
        fixed_bytes,
        byte_limit,
    };
    q.parent.extend(0..total);
    q.active[..n].fill(true);
    q.elements[n..].fill(true);
    q.admit(0)?;
    let mut front = try_with_capacity(n, MEMORY)?;
    let mut scratch = try_with_capacity(total, MEMORY)?;
    let mut dense = try_filled(n, false, MEMORY)?;
    let cutoff = 16usize.max(10 * n.isqrt());
    for (vertex, deferred) in dense.iter_mut().enumerate() {
        let original_degree = if total == n {
            q.rows[vertex].len()
        } else {
            q.rows[vertex]
                .iter()
                .map(|&element| q.rows[element].len().saturating_sub(1))
                .max()
                .unwrap_or(0)
        };
        *deferred = original_degree > cutoff;
        if *deferred {
            q.active[vertex] = false;
        }
    }
    for (vertex, &deferred) in dense.iter().enumerate() {
        if deferred {
            continue;
        }
        q.neighbors(vertex, &mut scratch);
        q.degree[vertex] = scratch.len();
        q.update(vertex);
    }
    let mut pivoted = try_filled(n, false, MEMORY)?;
    while let Some(&pivot) = q.heap.first() {
        pivoted[pivot] = true;
        q.remove(pivot);
        q.neighbors(pivot, &mut front);
        q.active[pivot] = false;
        // Every element incident to the pivot is contained in its new clique.
        for index in 0..q.rows[pivot].len() {
            let element = q.root(q.rows[pivot][index]);
            if q.elements[element] {
                q.elements[element] = false;
                q.assembly[element] = pivot;
                q.clear(element);
            }
        }
        advance_epoch(&mut q.epoch, &mut q.marks);
        for &vertex in &front {
            q.marks[vertex] = q.epoch;
        }
        for &vertex in &front {
            scratch.clear();
            for &entry in &q.rows[vertex] {
                let entry = q.root(entry);
                if (q.active[entry] && q.marks[entry] != q.epoch) || q.elements[entry] {
                    scratch.push(entry);
                }
            }
            scratch.push(pivot);
            scratch.sort_unstable();
            scratch.dedup();
            q.replace(vertex, &scratch)?;
        }
        q.replace(pivot, &front)?;
        q.elements[pivot] = true;
        // A variable with no incidence outside this new element can be
        // eliminated immediately; its assembly parent is the pivot.
        for &vertex in &front {
            if q.rows[vertex].as_slice() == [pivot] {
                q.remove(vertex);
                q.active[vertex] = false;
                q.parent[vertex] = pivot;
                q.assembly[vertex] = pivot;
                q.clear(vertex);
            }
        }
        // Equal incidence lists in this clique identify indistinguishable variables.
        front.sort_unstable_by(|&a, &b| q.rows[a].cmp(&q.rows[b]).then(a.cmp(&b)));
        let mut representative = NONE;
        for &vertex in &front {
            if !q.active[vertex] {
                continue;
            }
            if representative != NONE && q.rows[representative] == q.rows[vertex] {
                q.remove(vertex);
                q.active[vertex] = false;
                q.parent[vertex] = representative;
                q.weight[representative] += q.weight[vertex];
                q.assembly[vertex] = representative;
                q.clear(vertex);
            } else {
                representative = vertex;
            }
        }
        for &vertex in &front {
            if q.active[vertex] {
                q.neighbors(vertex, &mut scratch);
                q.degree[vertex] = scratch.iter().map(|&v| q.weight[v]).sum();
                q.update(vertex);
            }
        }
    }
    postorder(&q.assembly[..n], &dense, &pivoted)
}

/// Visit children before their assembly parent, without recursion on long chains.
fn postorder(
    parents: &[usize],
    dense: &[bool],
    pivoted: &[bool],
) -> Result<Permutation, SciOxideError> {
    let n = parents.len();
    let mut children = try_filled(n, NONE, MEMORY)?;
    let mut siblings = try_filled(n, NONE, MEMORY)?;
    // Earlier eliminated fronts precede absorbed variable leaves. This keeps
    // each amalgamated front contiguous after its contribution subtrees.
    for element in [false, true] {
        for (vertex, &parent) in parents.iter().enumerate().rev() {
            if !dense[vertex] && parent != NONE && pivoted[vertex] == element {
                siblings[vertex] = children[parent];
                children[parent] = vertex;
            }
        }
    }
    let mut stack = try_with_capacity(n, MEMORY)?;
    let mut order = try_with_capacity(n, MEMORY)?;
    for (root, &parent) in parents.iter().enumerate() {
        if dense[root] || parent != NONE {
            continue;
        }
        stack.push((root, children[root]));
        while let Some(&(vertex, child)) = stack.last() {
            if child == NONE {
                stack.pop();
                order.push(vertex);
            } else {
                if let Some(frame) = stack.last_mut() {
                    frame.1 = siblings[child];
                }
                stack.push((child, children[child]));
            }
        }
    }
    order.extend((0..n).filter(|&vertex| dense[vertex]));
    Permutation::new(order)
}

// Zero denotes an untouched mark. Clear the old generation before reusing it.
fn advance_epoch(epoch: &mut usize, marks: &mut [usize]) {
    if *epoch == usize::MAX {
        marks.fill(0);
        *epoch = 0;
    }
    *epoch += 1;
}

#[cfg(test)]
mod tests {
    use super::advance_epoch;

    #[test]
    fn generation_wrap_never_reuses_a_live_visitation_mark() {
        let mut epoch = usize::MAX - 1;
        let mut marks = [epoch, 0];
        advance_epoch(&mut epoch, &mut marks);
        assert!(marks.iter().all(|&mark| mark != epoch));
        marks[0] = epoch;
        advance_epoch(&mut epoch, &mut marks);
        assert_ne!(epoch, 0);
        assert!(marks.iter().all(|&mark| mark != epoch));
        marks[1] = epoch;
        advance_epoch(&mut epoch, &mut marks);
        assert!(marks.iter().all(|&mark| mark != epoch));
    }
}
