//! Flat shared facet, ridge and adjacency storage.
use crate::tessellation::storage::{Storage, retained};
use crate::tessellation::{Result, unstable};
use alloc::vec::Vec;
use scioxide_core::SciOxideError;
use scioxide_core::backend::checked_mul;
// Canonical rows have no orientation; `inside` is the exact sign of the row
// followed by an interior witness. Construction history may keep unsorted rows.
#[derive(Debug, Clone, Copy)]
pub(super) struct Facet<'a> {
    pub(super) vertices: &'a [usize],
    pub(super) inside: i8,
}
#[derive(Debug)]
pub(super) struct Facets {
    width: usize,
    vertices: Vec<usize>,
    inside: Vec<i8>,
}
impl Facets {
    pub(super) fn vertices(&self) -> &[usize] {
        &self.vertices
    }
    pub(super) fn retained_bytes(&self) -> usize {
        retained(&self.vertices) + retained(&self.inside)
    }

    pub(super) fn new(width: usize, capacity: usize, storage: &Storage) -> Result<Self> {
        let quantity = scioxide_core::quantity!("geometry_facets");
        Ok(Self {
            width,
            vertices: storage.capacity(checked_mul(width, capacity, quantity)?, quantity)?,
            inside: storage.capacity(capacity, quantity)?,
        })
    }
    pub(super) fn len(&self) -> usize {
        self.inside.len()
    }
    pub(super) fn is_empty(&self) -> bool {
        self.inside.is_empty()
    }
    pub(super) fn at(&self, index: usize) -> Facet<'_> {
        Facet {
            vertices: &self.vertices[index * self.width..(index + 1) * self.width],
            inside: self.inside[index],
        }
    }
    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = Facet<'_>> {
        self.vertices
            .chunks_exact(self.width)
            .zip(&self.inside)
            .map(|(vertices, &inside)| Facet { vertices, inside })
    }
    pub(super) fn push(&mut self, vertices: &[usize], inside: i8, storage: &Storage) -> Result<()> {
        debug_assert_eq!(vertices.len(), self.width);
        storage.reserve(&mut self.vertices, self.width)?;
        storage.reserve(&mut self.inside, 1)?;
        self.vertices.extend_from_slice(vertices);
        self.inside.push(inside);
        Ok(())
    }
    // The exhaustive canonical reference sorts its own rows.
    #[cfg(test)]
    pub(super) fn sort(&mut self, storage: &Storage) -> Result<()> {
        let before = self.retained_bytes();
        let frame = storage.frame();

        let mut order = storage.capacity(self.len(), scioxide_core::quantity!("facet_order"))?;
        order.extend(0..self.len());
        order.sort_unstable_by(|&a, &b| self.at(a).vertices.cmp(self.at(b).vertices));
        let mut sorted = Self::new(self.width, self.len(), storage)?;
        for i in order {
            let facet = self.at(i);
            sorted.push(facet.vertices, facet.inside, storage)?;
        }
        *self = sorted;
        frame.keep(before, self.retained_bytes());
        Ok(())
    }
    pub(super) fn search(&self, vertices: &[usize]) -> core::result::Result<usize, usize> {
        let mut lo = 0;
        let mut hi = self.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.at(mid).vertices.cmp(vertices) {
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
                core::cmp::Ordering::Equal => return Ok(mid),
            }
        }
        Err(lo)
    }
}

// One key arena and sortable (key offset, facet, opposite vertex) records for
// matching ridges of arbitrary facet sets; no allocation belongs to a ridge.
#[derive(Debug)]
pub(super) struct Ridges {
    width: usize,
    keys: Vec<usize>,
    pub(super) records: Vec<(usize, usize, usize)>,
}
impl Ridges {
    pub(super) fn new(width: usize) -> Self {
        Self {
            width,
            keys: Vec::new(),
            records: Vec::new(),
        }
    }
    pub(super) fn push(
        &mut self,
        vertices: &[usize],
        facet: usize,
        opposite: usize,
        storage: &Storage,
    ) -> Result<()> {
        storage.reserve(&mut self.keys, self.width)?;
        storage.reserve(&mut self.records, 1)?;
        let offset = self.keys.len();
        self.keys.extend(
            vertices
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| (i != opposite).then_some(v)),
        );
        self.keys[offset..].sort_unstable();
        self.records.push((offset, facet, opposite));
        Ok(())
    }
    pub(super) fn sort(&mut self) {
        let keys = &self.keys;
        let width = self.width;
        self.records.sort_unstable_by(|a, b| {
            keys[a.0..a.0 + width]
                .cmp(&keys[b.0..b.0 + width])
                .then_with(|| (a.1, a.2).cmp(&(b.1, b.2)))
        });
    }
    pub(super) fn same(&self, a: usize, b: usize) -> bool {
        let a = self.records[a].0;
        let b = self.records[b].0;
        self.keys[a..a + self.width] == self.keys[b..b + self.width]
    }
}

#[derive(Debug)]
pub(super) struct Topology {
    pub(super) facets: Facets,
    pub(super) neighbors: Vec<Option<usize>>,
    pub(super) boundary_points: Vec<(usize, usize)>,
}
impl Topology {
    pub(super) fn retained_bytes(&self) -> usize {
        retained(&self.neighbors) + retained(&self.boundary_points) + self.facets.retained_bytes()
    }

    pub(super) fn new(facets: Facets, storage: &Storage) -> Result<Self> {
        let before = facets.retained_bytes();
        let frame = storage.frame();

        let width = facets.width;
        let mut neighbors = storage.filled(
            facets.vertices.len(),
            None,
            scioxide_core::quantity!("facet_adjacency"),
        )?;
        let mut ridges = Ridges::new(width - 1);
        for (i, facet) in facets.iter().enumerate() {
            for j in 0..width {
                ridges.push(facet.vertices, i, j, storage)?;
            }
        }
        ridges.sort();
        let mut i = 0;
        while i < ridges.records.len() {
            let mut end = i + 1;
            while end < ridges.records.len() && ridges.same(i, end) {
                end += 1;
            }
            if end - i > 2 {
                return Err(unstable("nonmanifold hull ridge"));
            }
            if end - i == 2 {
                let (_, a, aj) = ridges.records[i];
                let (_, b, bj) = ridges.records[i + 1];
                neighbors[a * width + aj] = Some(b);
                neighbors[b * width + bj] = Some(a);
            }
            i = end;
        }
        let result = Self {
            facets,
            neighbors,
            boundary_points: Vec::new(),
        };
        frame.keep(before, result.retained_bytes());
        Ok(result)
    }
    // Adopt adjacency already known exactly by construction, without the
    // ridge sort of `new`. `neighbors[i * width + j]` lies opposite vertex j.
    pub(super) fn from_parts(facets: Facets, neighbors: Vec<Option<usize>>) -> Self {
        debug_assert_eq!(neighbors.len(), facets.vertices.len());
        Self {
            facets,
            neighbors,
            boundary_points: Vec::new(),
        }
    }
    pub(super) fn vertices(&self, index: usize) -> Result<&[usize]> {
        if index >= self.facets.len() {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("geometry_cell"),
                index,
                self.facets.len(),
            ));
        }
        Ok(self.facets.at(index).vertices)
    }
    pub(super) fn adjacent(&self, index: usize) -> Result<&[Option<usize>]> {
        self.vertices(index)?;
        let width = self.facets.width;
        Ok(&self.neighbors[index * width..(index + 1) * width])
    }
}
