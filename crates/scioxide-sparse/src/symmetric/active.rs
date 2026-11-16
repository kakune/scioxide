//! Reserved Hermitian Schur entries with local incidence and coordinate lookup.
use super::*;

const NONE: usize = usize::MAX;
const MEMORY: scioxide_core::Quantity = scioxide_core::quantity!("active_workspace_bytes");

#[derive(Debug, Clone, Copy)]
struct Slot<S> {
    row: usize,
    col: usize,
    value: S,
    row_prev: usize,
    row_next: usize,
    col_prev: usize,
    col_next: usize,
}

/// Original-coordinate lower triangle. Links and index hold stable slot indices;
/// eliminated entries return to a free list, without compaction or reallocation.
#[derive(Debug)]
pub(super) struct Active<S> {
    slots: Vec<Slot<S>>,
    rows: Vec<usize>,
    cols: Vec<usize>,
    index: Vec<usize>,
    free: usize,
    len: usize,
    limit: usize,
}

fn index_len(limit: usize) -> Result<usize, SciOxideError> {
    if limit == 0 {
        return Ok(0);
    }
    mul(2, limit, MEMORY)?
        .checked_next_power_of_two()
        .ok_or_else(|| SciOxideError::size_overflow(MEMORY))
}

pub(super) fn bytes<S>(n: usize, limit: usize) -> Result<usize, SciOxideError> {
    add(
        mul(limit, size_of::<Slot<S>>(), MEMORY)?,
        mul(
            add(mul(2, n, MEMORY)?, index_len(limit)?, MEMORY)?,
            size_of::<usize>(),
            MEMORY,
        )?,
        MEMORY,
    )
}

// SplitMix64's finalizer mixes each coordinate independently: no row*n key,
// platform-sized overflow, ambient seed, or poorly distributed banded keys.
fn mix(mut word: u64) -> u64 {
    word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    word ^ (word >> 31)
}
fn hash(row: usize, col: usize) -> usize {
    mix(mix(row as u64).wrapping_add(0x9e37_79b9_7f4a_7c15) ^ (col as u64)) as usize
}

impl<S: NumericScalar> Active<S> {
    pub(super) fn empty() -> Self {
        Self {
            slots: Vec::new(),
            rows: Vec::new(),
            cols: Vec::new(),
            index: Vec::new(),
            free: NONE,
            len: 0,
            limit: 0,
        }
    }
    pub(super) fn retained_bytes(&self) -> usize {
        self.slots.capacity() * size_of::<Slot<S>>()
            + (self.rows.capacity() + self.cols.capacity() + self.index.capacity())
                * size_of::<usize>()
    }

    pub(super) fn new(n: usize, limit: usize) -> Result<Self, SciOxideError> {
        Ok(Self {
            slots: try_with_capacity(limit, MEMORY)?,
            rows: try_filled(n, NONE, MEMORY)?,
            cols: try_filled(n, NONE, MEMORY)?,
            index: try_filled(index_len(limit)?, NONE, MEMORY)?,
            free: NONE,
            len: 0,
            limit,
        })
    }

    // Reserved entry capacity, which is also the enforced active-entry limit.
    pub(super) fn limit(&self) -> usize {
        self.limit
    }

    pub(super) fn clear(&mut self) {
        self.slots.clear();
        self.rows.fill(NONE);
        self.cols.fill(NONE);
        self.index.fill(NONE);
        self.free = NONE;
        self.len = 0;
    }

    // Finds the coordinate or an empty bucket. At most half the buckets are
    // occupied; deletion repairs clusters, so there are no accumulating tombstones.
    fn locate(&self, row: usize, col: usize) -> usize {
        let mask = self.index.len() - 1;
        let mut bucket = hash(row, col) & mask;
        while self.index[bucket] != NONE {
            let slot = &self.slots[self.index[bucket]];
            if (slot.row, slot.col) == (row, col) {
                break;
            }
            bucket = (bucket + 1) & mask;
        }
        bucket
    }

    pub(super) fn get(&self, row: usize, col: usize) -> S {
        if self.index.is_empty() {
            return S::ZERO;
        }
        let bucket = self.locate(row.max(col), row.min(col));
        let slot = self.index[bucket];
        if slot == NONE {
            S::ZERO
        } else if row >= col {
            self.slots[slot].value
        } else {
            self.slots[slot].value.conj()
        }
    }

    pub(super) fn set(&mut self, row: usize, col: usize, value: S) -> Result<(), SciOxideError> {
        let value = if row >= col { value } else { value.conj() };
        let (row, col) = (row.max(col), row.min(col));
        if self.index.is_empty() {
            return if value == S::ZERO {
                Ok(())
            } else {
                Err(self.full())
            };
        }
        let bucket = self.locate(row, col);
        let existing = self.index[bucket];
        if existing != NONE {
            if value == S::ZERO {
                self.remove(existing, bucket);
            } else {
                self.slots[existing].value = value;
            }
            return Ok(());
        }
        if value == S::ZERO {
            return Ok(());
        }
        if self.len == self.limit {
            return Err(self.full());
        }
        let slot = Slot {
            row,
            col,
            value,
            row_prev: NONE,
            row_next: self.rows[row],
            col_prev: NONE,
            col_next: self.cols[col],
        };
        let id = if self.free == NONE {
            let id = self.slots.len();
            self.slots.push(slot);
            id
        } else {
            let id = self.free;
            self.free = self.slots[id].row_next;
            self.slots[id] = slot;
            id
        };
        if slot.row_next != NONE {
            self.slots[slot.row_next].row_prev = id;
        }
        if slot.col_next != NONE {
            self.slots[slot.col_next].col_prev = id;
        }
        self.rows[row] = id;
        self.cols[col] = id;
        self.index[bucket] = id;
        self.len += 1;
        Ok(())
    }

    fn full(&self) -> SciOxideError {
        SciOxideError::resource_limit(
            scioxide_core::quantity!("active_entries"),
            self.len + 1,
            self.limit,
        )
    }

    /// Visits a complete Hermitian column, returning each diagonal only once.
    pub(super) fn column(&self, col: usize) -> impl Iterator<Item = (usize, S)> + '_ {
        core::iter::successors((self.rows[col] != NONE).then_some(self.rows[col]), |&id| {
            let next = self.slots[id].row_next;
            (next != NONE).then_some(next)
        })
        .map(|id| (self.slots[id].col, self.slots[id].value.conj()))
        .chain(
            core::iter::successors((self.cols[col] != NONE).then_some(self.cols[col]), |&id| {
                let next = self.slots[id].col_next;
                (next != NONE).then_some(next)
            })
            .filter(move |&id| self.slots[id].row != col)
            .map(|id| (self.slots[id].row, self.slots[id].value)),
        )
    }

    /// Unlinks only entries incident on the eliminated vertex.
    pub(super) fn eliminate(&mut self, vertex: usize) {
        while self.rows[vertex] != NONE {
            self.remove_id(self.rows[vertex]);
        }
        while self.cols[vertex] != NONE {
            self.remove_id(self.cols[vertex]);
        }
    }

    fn remove_id(&mut self, id: usize) {
        let slot = &self.slots[id];
        self.remove(id, self.locate(slot.row, slot.col));
    }

    fn remove(&mut self, id: usize, bucket: usize) {
        let slot = self.slots[id];
        if slot.row_prev == NONE {
            self.rows[slot.row] = slot.row_next;
        } else {
            self.slots[slot.row_prev].row_next = slot.row_next;
        }
        if slot.row_next != NONE {
            self.slots[slot.row_next].row_prev = slot.row_prev;
        }
        if slot.col_prev == NONE {
            self.cols[slot.col] = slot.col_next;
        } else {
            self.slots[slot.col_prev].col_next = slot.col_next;
        }
        if slot.col_next != NONE {
            self.slots[slot.col_next].col_prev = slot.col_prev;
        }
        // Move a bucket backward exactly when its probe path crosses the hole.
        // Stable slot indices keep both incidence lists valid across these moves.
        let mask = self.index.len() - 1;
        let mut hole = bucket;
        let mut scan = (hole + 1) & mask;
        while self.index[scan] != NONE {
            let entry = &self.slots[self.index[scan]];
            let home = hash(entry.row, entry.col) & mask;
            if (scan.wrapping_sub(home) & mask) >= (scan.wrapping_sub(hole) & mask) {
                self.index[hole] = self.index[scan];
                hole = scan;
            }
            scan = (scan + 1) & mask;
        }
        self.index[hole] = NONE;
        self.slots[id].row_next = self.free;
        self.free = id;
        self.len -= 1;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use alloc::vec;
    use scioxide_core::Complex;

    #[test]
    fn colliding_wraparound_clusters_survive_deletion_and_slot_reuse() {
        let mut active = Active::<f64>::new(128, 16).unwrap();
        let mask = active.index.len() - 1;
        let keys: Vec<_> = (0..128)
            .flat_map(|row| (0..=row).map(move |col| (row, col)))
            .filter(|&(row, col)| hash(row, col) & mask == mask)
            .take(16)
            .collect();
        assert_eq!(keys.len(), 16);
        for &(row, col) in &keys {
            active.set(row, col, 2.).unwrap();
        }
        assert!(active.set(127, 127, 1.).is_err());
        for &(row, col) in &keys {
            active.set(col, row, 3.).unwrap();
        }
        for (i, &(row, col)) in keys.iter().enumerate() {
            active.set(row, col, 0.).unwrap();
            for &(r, c) in &keys[i + 1..] {
                assert_eq!(active.get(r, c), 3.);
            }
            assert_eq!(active.get(row, col), 0.);
        }
        assert_eq!(active.len, 0);
        for &(row, col) in keys.iter().rev() {
            active.set(row, col, 4.).unwrap();
        }
        assert_eq!(active.slots.len(), 16);
        for vertex in 0..128 {
            active.eliminate(vertex);
            assert_eq!(active.column(vertex).count(), 0);
            for &(row, col) in &keys {
                assert_eq!(
                    active.get(row, col),
                    if row > vertex && col > vertex { 4. } else { 0. }
                );
            }
        }
        active.clear();
        assert!(active.index.iter().all(|&slot| slot == NONE));
        active.set(0, 0, 1.).unwrap();
        assert_eq!(active.column(0).collect::<Vec<_>>(), vec![(0, 1.)]);
    }

    #[test]
    fn complex_incidence_matches_an_independent_dense_store() {
        const N: usize = 24;
        let zero = Complex::new(0., 0.);
        let mut dense = vec![zero; N * N];
        let mut active = Active::new(N, N * (N + 1) / 2).unwrap();
        let mut state = 17u64;
        for step in 0..4000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let row = (state >> 32) as usize % N;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let col = (state >> 32) as usize % N;
            let value = if step % 5 == 0 {
                zero
            } else {
                Complex::new((step % 11 + 1) as f64, if row == col { 0. } else { -0.5 })
            };
            active.set(row, col, value).unwrap();
            dense[row * N + col] = value;
            dense[col * N + row] = value.conj();
            if step % 19 == 0 {
                active.eliminate(row);
                for other in 0..N {
                    dense[row * N + other] = zero;
                    dense[other * N + row] = zero;
                }
            }
            if step % 31 == 0 {
                for col in 0..N {
                    let mut found = vec![zero; N];
                    for (row, value) in active.column(col) {
                        assert_eq!(found[row], zero);
                        found[row] = value;
                    }
                    for (row, value) in found.into_iter().enumerate() {
                        assert_eq!(value, dense[row * N + col]);
                        assert_eq!(active.get(row, col), value);
                    }
                }
                assert_eq!(
                    active.len,
                    (0..N)
                        .map(|i| (0..=i).filter(|&j| dense[i * N + j] != zero).count())
                        .sum()
                );
            }
        }
    }

    #[test]
    fn zero_capacity_and_overflow_are_explicit() {
        let mut active = Active::<f64>::new(2, 0).unwrap();
        active.set(0, 1, 0.).unwrap();
        active.eliminate(0);
        assert_eq!(active.get(0, 1), 0.);
        let error = active.set(0, 0, 1.).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::ResourceLimit { .. }
        ));
        for limit in [usize::MAX, usize::MAX / 2] {
            assert!(matches!(
                bytes::<f64>(2, limit).unwrap_err().view(),
                scioxide_core::SciOxideErrorView::SizeOverflow { .. }
            ));
        }
    }
}
