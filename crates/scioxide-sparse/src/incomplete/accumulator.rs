//! Touched block storage and ordered elimination reach, reserved before factorization.
use super::*;
use crate::graph::heap;

use scioxide_core::Scaled;
use scioxide_core::backend::try_with_capacity;

#[derive(Debug)]
pub(super) struct Row<S: NumericScalar> {
    pub values: Vec<S>,
    pub levels: Vec<usize>,
    pub columns: Vec<usize>,
    pub rank: Vec<(usize, Scaled<S::Real>)>,
    marked: Vec<bool>,
    pending: Vec<usize>,
    positions: Vec<usize>,
}

impl<S: NumericScalar> Row<S> {
    pub(super) const fn empty() -> Self {
        Self {
            values: Vec::new(),
            levels: Vec::new(),
            columns: Vec::new(),
            rank: Vec::new(),
            marked: Vec::new(),
            pending: Vec::new(),
            positions: Vec::new(),
        }
    }

    pub(super) fn new(p: Shape) -> Result<Self, SciOxideError> {
        let n = p.rows;
        let q = scioxide_core::quantity!("block_row");
        Ok(Self {
            values: try_filled(mul(mul(n, p.block, q)?, p.block, q)?, S::ZERO, q)?,
            levels: try_filled(n, usize::MAX, q)?,
            columns: try_with_capacity(n, q)?,
            rank: try_with_capacity(if p.method.threshold() { n } else { 0 }, q)?,
            marked: try_filled(n, false, q)?,
            pending: try_with_capacity(n, q)?,
            positions: try_filled(n, usize::MAX, q)?,
        })
    }

    pub(super) fn fits(&self, p: Shape) -> bool {
        let n = p.rows;
        let rank = if p.method.threshold() { n } else { 0 };
        self.values.len() == n * p.block * p.block
            && self.levels.len() == n
            && self.marked.len() == n
            && self.positions.len() == n
            && self.columns.capacity() >= n
            && self.pending.capacity() >= n
            && self.rank.capacity() >= rank
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.values.capacity() * size_of::<S>()
            + (self.levels.capacity()
                + self.columns.capacity()
                + self.pending.capacity()
                + self.positions.capacity())
                * size_of::<usize>()
            + self.marked.capacity()
            + self.rank.capacity() * size_of::<(usize, Scaled<S::Real>)>()
    }

    // Also resets a partially eliminated row left by any failed epoch.
    pub(super) fn clear(&mut self) {
        for j in self.columns.drain(..) {
            self.marked[j] = false;
            self.levels[j] = usize::MAX;
            self.positions[j] = usize::MAX;
        }
        self.pending.clear();
        self.rank.clear();
    }

    // Each block is zeroed and scheduled at most once, including structural zeros.
    pub(super) fn touch(&mut self, j: usize, diagonal: usize, b2: usize) {
        if !self.marked[j] {
            self.marked[j] = true;
            self.columns.push(j);
            self.values[j * b2..(j + 1) * b2].fill(S::ZERO);
            if j < diagonal {
                heap::update(&mut self.pending, &mut self.positions, j, |a, b| a < b);
            }
        }
    }

    pub(super) fn next_lower(&mut self) -> Option<usize> {
        let j = *self.pending.first()?;
        heap::remove(&mut self.pending, &mut self.positions, j, |a, b| a < b);
        Some(j)
    }
}

/// Column incidence of completed ICT rows. Nodes use candidate factor positions;
/// only kept lower blocks enter the lists, so truncation cannot leave stale links.
#[derive(Debug)]
pub(super) struct Incidence {
    pub heads: Vec<usize>,
    pub next: Vec<usize>,
    pub rows: Vec<usize>,
}

impl Incidence {
    pub(super) const fn empty() -> Self {
        Self {
            heads: Vec::new(),
            next: Vec::new(),
            rows: Vec::new(),
        }
    }

    pub(super) fn new(p: Shape) -> Result<Self, SciOxideError> {
        let (n, f) = Self::sizes(p);
        let q = scioxide_core::quantity!("factor_incidence");
        Ok(Self {
            heads: try_filled(n, usize::MAX, q)?,
            next: try_filled(f, usize::MAX, q)?,
            rows: try_filled(f, 0, q)?,
        })
    }

    fn sizes(p: Shape) -> (usize, usize) {
        if matches!(p.method, IncompleteMethod::Ict) {
            (p.rows, p.blocks)
        } else {
            (0, 0)
        }
    }

    pub(super) fn fits(&self, p: Shape) -> bool {
        let (n, f) = Self::sizes(p);
        self.heads.len() >= n && self.next.len() >= f && self.rows.len() >= f
    }

    pub(super) fn retained_bytes(&self) -> usize {
        (self.heads.capacity() + self.next.capacity() + self.rows.capacity()) * size_of::<usize>()
    }

    pub(super) fn insert(&mut self, row: usize, col: usize, pos: usize) {
        self.next[pos] = self.heads[col];
        self.rows[pos] = row;
        self.heads[col] = pos;
    }
}
