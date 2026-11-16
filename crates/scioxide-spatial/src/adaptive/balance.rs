//! One-level marks propagate over the source epoch's physical interfaces.
use super::*;

// A fallible min heap preserves the old smallest-stable-ID split order.
#[derive(Default)]
struct Queue(Vec<usize>);
impl Queue {
    fn push(&mut self, id: usize, work: &mut Work<'_, '_, '_>) -> Result<()> {
        work.push(&mut self.0, id)?;
        let mut at = self.0.len() - 1;
        while at > 0 {
            work.tick()?;
            let parent = (at - 1) / 2;
            if self.0[parent] <= self.0[at] {
                break;
            }
            self.0.swap(parent, at);
            at = parent;
        }
        Ok(())
    }
    fn pop(&mut self, work: &mut Work<'_, '_, '_>) -> Result<Option<usize>> {
        let Some(last) = self.0.pop() else {
            return Ok(None);
        };
        if self.0.is_empty() {
            return Ok(Some(last));
        }
        let result = core::mem::replace(&mut self.0[0], last);
        work.sift(&mut self.0, 0, &|a: &usize, b: &usize| b.cmp(a))?;
        Ok(Some(result))
    }
}

fn enqueue(
    pair: [usize; 2],
    ids: &[usize],
    levels: &[u8],
    queued: &mut [bool],
    queue: &mut Queue,
    work: &mut Work<'_, '_, '_>,
) -> Result<()> {
    work.tick()?;
    let [a, b] = pair;
    if ids[a] != ids[b] && levels[a].abs_diff(levels[b]) > 1 {
        let id = ids[if levels[a] < levels[b] { a } else { b }];
        if !queued[id] {
            queue.push(id, work)?;
            queued[id] = true;
        }
    }
    Ok(())
}

impl<S: RealScalar, const D: usize> AdaptiveMesh<S, D> {
    pub(super) fn balance(
        &self,
        storage: &mut Storage<S, D>,
        refine: &[usize],
        coarsen: &[usize],
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Vec<usize>> {
        #[cfg(test)]
        if storage.exhaustive {
            return storage.balance_exhaustive(coarsen, work);
        }
        // The source is balanced and a mark changes a leaf by exactly one
        // level. Every induced split is therefore of a source leaf, at most
        // once. New child faces cannot introduce a neighbor absent from the
        // source graph. Coarsened siblings share one target ID and level.
        let count = self.cell_count();
        let mut ids = Vec::new();
        let mut levels = Vec::new();
        let mut inverse = Vec::new();
        work.reserve(&mut inverse, self.storage.nodes.len())?;
        inverse.resize(self.storage.nodes.len(), usize::MAX);
        for (index, &id) in self.storage.active.iter().enumerate() {
            work.tick()?;
            inverse[id] = index;
            let node = self.storage.nodes[id];
            let merged = match node.parent {
                Some(parent) if work.search(coarsen, &parent)?.is_ok() => Some(parent),
                _ => None,
            };
            let (target, level) = if let Some(parent) = merged {
                (parent, self.storage.nodes[parent].level)
            } else {
                (id, node.level + u8::from(work.search(refine, &id)?.is_ok()))
            };
            work.push(&mut ids, target)?;
            work.push(&mut levels, level)?;
        }
        // CSR adjacency avoids rescanning all interfaces after each split.
        let mut offsets = Vec::new();
        work.reserve(&mut offsets, count + 1)?;
        offsets.resize(count + 1, 0usize);
        for interface in &self.storage.interfaces {
            work.tick()?;
            if let Some(right_side) = interface.right {
                let right = right_side.cell;
                offsets[interface.left_cell + 1] += 1;
                offsets[right + 1] += 1;
            }
        }
        for i in 1..offsets.len() {
            offsets[i] = checked_add(
                offsets[i],
                offsets[i - 1],
                scioxide_core::quantity!("adaptive_adjacency"),
            )?;
        }
        let mut neighbors = Vec::new();
        work.reserve(&mut neighbors, offsets[count])?;
        neighbors.resize(offsets[count], 0usize);
        let mut cursors = work.copy(&offsets[..count])?;
        let mut queued = Vec::new();
        work.reserve(&mut queued, self.storage.nodes.len())?;
        queued.resize(self.storage.nodes.len(), false);
        let mut queue = Queue::default();
        for interface in &self.storage.interfaces {
            work.tick()?;
            if let Some(right_side) = interface.right {
                let right = right_side.cell;
                let left = interface.left_cell;
                neighbors[cursors[left]] = right;
                cursors[left] += 1;
                neighbors[cursors[right]] = left;
                cursors[right] += 1;
                enqueue([left, right], &ids, &levels, &mut queued, &mut queue, work)?;
            }
        }
        let mut induced = Vec::new();
        while let Some(id) = queue.pop(work)? {
            if work.search(coarsen, &id)?.is_ok() {
                return Err(invalid(
                    scioxide_core::argument!("request.coarsen"),
                    "coarsening violates final 2:1 balance",
                ));
            }
            work.push(&mut induced, id)?;
            storage.split(id, work)?;
            let index = inverse[id];
            levels[index] += 1;
            for &neighbor in &neighbors[offsets[index]..offsets[index + 1]] {
                enqueue(
                    [index, neighbor],
                    &ids,
                    &levels,
                    &mut queued,
                    &mut queue,
                    work,
                )?;
            }
        }
        Ok(induced)
    }
}
