//! Fixed-capacity distinct indices for structured incidence queries.
use crate::Index;

// Callers prove their maximum incidence count fits N from validated dimensions (1..=3).
pub(crate) struct DistinctIndices<const D: usize, const N: usize> {
    values: [Index<D>; N],
    len: usize,
}

impl<const D: usize, const N: usize> DistinctIndices<D, N> {
    pub(crate) fn new() -> Self {
        Self {
            values: [Index::from_validated([0; D]); N],
            len: 0,
        }
    }

    pub(crate) fn push(&mut self, candidate: Index<D>) {
        if !self.values[..self.len].contains(&candidate) {
            self.values[self.len] = candidate;
            self.len += 1;
        }
    }

    pub(crate) fn into_iter(
        self,
    ) -> impl ExactSizeIterator<Item = Index<D>> + Clone + core::fmt::Debug {
        self.values.into_iter().take(self.len)
    }
}
