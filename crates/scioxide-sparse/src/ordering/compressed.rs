//! Compressed undirected union and rectangular row-incidence graphs.
use alloc::vec::Vec;
use core::ops::Index;
use scioxide_core::backend::{checked_add, checked_mul, try_filled, try_with_capacity};
use scioxide_core::{SciOxideError, StructuralPatternView, Triangle};

pub(super) struct Graph {
    pub ptr: Vec<usize>,
    pub entries: Vec<usize>,
}
impl Graph {
    pub(super) fn len(&self) -> usize {
        self.ptr.len() - 1
    }
    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = &[usize]> {
        self.ptr.windows(2).map(|p| &self.entries[p[0]..p[1]])
    }
}
impl Index<usize> for Graph {
    type Output = [usize];
    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[self.ptr[index]..self.ptr[index + 1]]
    }
}
fn selected(triangle: Option<Triangle>, row: usize, col: usize) -> bool {
    match triangle {
        Some(Triangle::Lower) => row >= col,
        Some(Triangle::Upper) => row <= col,
        None => true,
    }
}
/// Merge sorted rows with sorted transposed rows, excluding diagonals.
pub(super) fn union(
    pattern: StructuralPatternView<'_>,
    triangle: Option<Triangle>,
) -> Result<Graph, SciOxideError> {
    if pattern.rows() != pattern.cols() {
        return Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::two(pattern.rows(), pattern.rows()),
            scioxide_core::Dimensions::two(pattern.rows(), pattern.cols()),
        ));
    }
    let quantity = scioxide_core::quantity!("ordering_adjacency");
    let n = pattern.rows();
    let z = pattern.col_indices().len();
    let mut transposed = try_filled(z, 0usize, quantity)?;
    let transpose_ptr = crate::helpers::counting_transpose(
        n,
        pattern
            .row_ptr()
            .windows(2)
            .enumerate()
            .flat_map(|(row, p)| {
                pattern.col_indices()[p[0]..p[1]]
                    .iter()
                    .copied()
                    .filter(move |&col| row != col && selected(triangle, row, col))
                    .map(move |col| (col, row))
            }),
        |slot, row| transposed[slot] = row,
    )?;
    let mut ptr = try_with_capacity(checked_add(n, 1, quantity)?, quantity)?;
    let mut entries = try_with_capacity(checked_mul(z, 2, quantity)?, quantity)?;
    ptr.push(0);
    for row in 0..n {
        let mut a = pattern.col_indices()[pattern.row_ptr()[row]..pattern.row_ptr()[row + 1]]
            .iter()
            .copied()
            .filter(|&col| row != col && selected(triangle, row, col))
            .peekable();
        let mut b = transposed[transpose_ptr[row]..transpose_ptr[row + 1]]
            .iter()
            .copied()
            .peekable();
        while let Some(value) = match (a.peek(), b.peek()) {
            (Some(&left), Some(&right)) => {
                if left <= right {
                    if left == right {
                        b.next();
                    }
                    a.next()
                } else {
                    b.next()
                }
            }
            (Some(_), None) => a.next(),
            (None, Some(_)) => b.next(),
            (None, None) => None,
        } {
            entries.push(value);
        }
        ptr.push(entries.len());
    }
    Ok(Graph { ptr, entries })
}
/// Columns are variables; each nonsingleton input row is an initial clique element.
/// The column-intersection graph is never expanded into pairwise edges.
pub(super) fn incidence(pattern: StructuralPatternView<'_>) -> Result<Graph, SciOxideError> {
    let quantity = scioxide_core::quantity!("ordering_incidence");
    let n = pattern.cols();
    let total = checked_add(n, pattern.rows(), quantity)?;
    let pointers = checked_add(total, 1, quantity)?;
    let mut entries = try_filled(
        checked_mul(pattern.col_indices().len(), 2, quantity)?,
        0usize,
        quantity,
    )?;
    let mut ptr = crate::helpers::counting_transpose(
        n,
        pattern
            .row_ptr()
            .windows(2)
            .enumerate()
            .filter(|(_, p)| p[1] - p[0] > 1)
            .flat_map(|(row, p)| {
                pattern.col_indices()[p[0]..p[1]]
                    .iter()
                    .map(move |&col| (col, n + row))
            }),
        |slot, element| entries[slot] = element,
    )?;
    let count = ptr[n];
    entries.truncate(count);
    ptr.try_reserve_exact(total - n)
        .map_err(|_| SciOxideError::allocation_failure(quantity, pointers))?;
    for p in pattern.row_ptr().windows(2) {
        if p[1] - p[0] > 1 {
            entries.extend_from_slice(&pattern.col_indices()[p[0]..p[1]]);
        }
        ptr.push(entries.len());
    }
    Ok(Graph { ptr, entries })
}
