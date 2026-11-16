//! Sorted immutable CSR incidence, with storage checked against the caller's limit.
use alloc::vec::Vec;
use scioxide_core::{
    SciOxideError,
    backend::{checked_add, try_filled, try_with_capacity},
};

#[derive(Debug, Default, Clone)]
pub(crate) struct Incidence {
    offsets: Vec<usize>,
    entries: Vec<usize>,
}

impl Incidence {
    pub(crate) fn new(count: usize, pairs: Vec<(usize, usize)>) -> Result<Self, SciOxideError> {
        Self::from_pairs_with(
            pairs,
            |length| {
                let offsets = try_filled(
                    checked_add(count, 1, scioxide_core::quantity!("incidence_offsets"))?,
                    0,
                    scioxide_core::quantity!("incidence_offsets"),
                )?;
                let entries =
                    try_with_capacity(length, scioxide_core::quantity!("incidence_entries"))?;
                Ok((offsets, entries))
            },
            || Ok(()),
        )
    }

    // Sorting determines exact output storage before the caller admits buffers.
    pub(crate) fn from_pairs_with(
        mut pairs: Vec<(usize, usize)>,
        buffers: impl FnOnce(usize) -> Result<(Vec<usize>, Vec<usize>), SciOxideError>,
        mut check: impl FnMut() -> Result<(), SciOxideError>,
    ) -> Result<Self, SciOxideError> {
        pairs.sort_unstable();
        pairs.dedup();
        check()?;
        let (mut offsets, mut entries) = buffers(pairs.len())?;
        for &(row, _) in &pairs {
            offsets[row + 1] += 1;
        }
        for row in 1..offsets.len() {
            offsets[row] += offsets[row - 1];
        }
        entries.extend(pairs.into_iter().map(|(_, value)| value));
        Ok(Self { offsets, entries })
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        (self.offsets.capacity() + self.entries.capacity()) * size_of::<usize>()
    }

    pub(crate) fn at(&self, row: usize) -> &[usize] {
        &self.entries[self.offsets[row]..self.offsets[row + 1]]
    }

    pub(crate) fn neighbors(
        &self,
        index: usize,
        quantity: scioxide_core::Quantity,
    ) -> Result<impl Iterator<Item = usize> + '_, SciOxideError> {
        let count = self.offsets.len().saturating_sub(1);
        if index >= count {
            return Err(SciOxideError::out_of_bounds(quantity, index, count));
        }
        Ok(self.at(index).iter().copied())
    }
}
