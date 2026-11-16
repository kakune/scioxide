//! Fallible ordered staging indexes. The arena owns no per-entry allocations.
use super::{Result, Work};
use alloc::vec::Vec;
use core::cmp::Ordering;

#[derive(Clone, Debug)]
struct Entry<K> {
    key: K,
    value: usize,
    children: [Option<usize>; 2],
    height: u8,
}

// Only insertion and lookup are needed. AVL balance bounds both operations
// even for sorted imports; IDs are values, never tree traversal positions.
#[derive(Debug)]
pub(super) struct Index<K> {
    entries: Vec<Entry<K>>,
    root: Option<usize>,
}

impl<K: Ord> Index<K> {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            root: None,
        }
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.root = None;
    }

    pub(super) fn get(&self, key: &K, work: &mut Work<'_, '_, '_>) -> Result<Option<usize>> {
        let mut next = self.root;
        while let Some(id) = next {
            work.tick()?;
            let entry = &self.entries[id];
            next = match key.cmp(&entry.key) {
                Ordering::Equal => return Ok(Some(entry.value)),
                Ordering::Less => entry.children[0],
                Ordering::Greater => entry.children[1],
            };
        }
        Ok(None)
    }

    pub(super) fn insert(
        &mut self,
        key: K,
        value: usize,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<()> {
        self.root = Some(self.insert_at(self.root, key, value, work)?);
        Ok(())
    }

    fn height(&self, id: Option<usize>) -> u8 {
        id.map_or(0, |id| self.entries[id].height)
    }

    fn update(&mut self, id: usize) {
        let [left, right] = self.entries[id].children;
        self.entries[id].height = 1 + self.height(left).max(self.height(right));
    }

    // Promote the child on `side` and return the new subtree root.
    fn rotate(&mut self, id: usize, side: usize) -> usize {
        let Some(child) = self.entries[id].children[side] else {
            return id;
        };
        self.entries[id].children[side] = self.entries[child].children[1 - side];
        self.entries[child].children[1 - side] = Some(id);
        self.update(id);
        self.update(child);
        child
    }

    fn insert_at(
        &mut self,
        root: Option<usize>,
        key: K,
        value: usize,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<usize> {
        work.tick()?;
        let Some(id) = root else {
            let id = self.entries.len();
            work.push(
                &mut self.entries,
                Entry {
                    key,
                    value,
                    children: [None; 2],
                    height: 1,
                },
            )?;
            return Ok(id);
        };
        let side = match key.cmp(&self.entries[id].key) {
            Ordering::Equal => {
                self.entries[id].value = value;
                return Ok(id);
            }
            Ordering::Less => 0,
            Ordering::Greater => 1,
        };
        self.entries[id].children[side] =
            Some(self.insert_at(self.entries[id].children[side], key, value, work)?);
        self.update(id);
        let [left, right] = self.entries[id].children;
        let balance = i16::from(self.height(left)) - i16::from(self.height(right));
        if balance.abs() <= 1 {
            return Ok(id);
        }
        let side = usize::from(balance < 0);
        if let Some(child) = self.entries[id].children[side] {
            let children = self.entries[child].children;
            if self.height(children[side]) < self.height(children[1 - side]) {
                self.entries[id].children[side] = Some(self.rotate(child, 1 - side));
            }
        }
        Ok(self.rotate(id, side))
    }

    // Report all keys >= lower in O(log N + output), used by edge interval joins.
    pub(super) fn from(
        &self,
        lower: &K,
        work: &mut Work<'_, '_, '_>,
        mut visit: impl FnMut(usize, &mut Work<'_, '_, '_>) -> Result<()>,
    ) -> Result<()> {
        self.visit_from(self.root, lower, work, &mut visit)
    }

    fn visit_from(
        &self,
        root: Option<usize>,
        lower: &K,
        work: &mut Work<'_, '_, '_>,
        visit: &mut impl FnMut(usize, &mut Work<'_, '_, '_>) -> Result<()>,
    ) -> Result<()> {
        let Some(id) = root else { return Ok(()) };
        work.tick()?;
        let entry = &self.entries[id];
        if &entry.key >= lower {
            self.visit_from(entry.children[0], lower, work, visit)?;
            visit(entry.value, work)?;
        }
        self.visit_from(entry.children[1], lower, work, visit)
    }
}

impl Work<'_, '_, '_> {
    // In-place heapsort allows errors and stop checks during comparisons,
    // rather than accounting for an uninterruptible sort after it has run.
    pub(super) fn sort<T: Ord>(&mut self, values: &mut [T]) -> Result<()> {
        self.sort_by(values, Ord::cmp)
    }

    pub(super) fn sort_by<T>(
        &mut self,
        values: &mut [T],
        compare: impl Fn(&T, &T) -> Ordering,
    ) -> Result<()> {
        for root in (0..values.len() / 2).rev() {
            self.sift(values, root, &compare)?;
        }
        for end in (1..values.len()).rev() {
            values.swap(0, end);
            self.sift(&mut values[..end], 0, &compare)?;
        }
        Ok(())
    }

    // Restore the max-heap order of `compare` below `root`, charging each
    // comparison. Also serves the balance queue as a reversed (min) heap.
    pub(super) fn sift<T>(
        &mut self,
        values: &mut [T],
        mut root: usize,
        compare: &impl Fn(&T, &T) -> Ordering,
    ) -> Result<()> {
        while root < values.len() / 2 {
            let mut child = root * 2 + 1;
            if child + 1 < values.len() {
                self.tick()?;
                if compare(&values[child], &values[child + 1]).is_lt() {
                    child += 1;
                }
            }
            self.tick()?;
            if !compare(&values[root], &values[child]).is_lt() {
                break;
            }
            values.swap(root, child);
            root = child;
        }
        Ok(())
    }

    pub(super) fn search<T: Ord>(
        &mut self,
        values: &[T],
        key: &T,
    ) -> Result<core::result::Result<usize, usize>> {
        let (mut low, mut high) = (0, values.len());
        while low < high {
            let mid = low + (high - low) / 2;
            self.tick()?;
            match values[mid].cmp(key) {
                Ordering::Less => low = mid + 1,
                Ordering::Equal => return Ok(Ok(mid)),
                Ordering::Greater => high = mid,
            }
        }
        Ok(Err(low))
    }
}
