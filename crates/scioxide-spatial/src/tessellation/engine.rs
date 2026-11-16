//! Single-owner conflict hull. Facet history locates homogeneous boundary sites; only
//! the visible patch and its strictly outside lists change during insertion.
use super::topology::{Facets, Topology};
use crate::tessellation::cloud::Cloud;
use crate::tessellation::storage::{Storage, retained};
use crate::tessellation::work::Work;
use crate::tessellation::{Result, unstable};
use alloc::vec::Vec;
use scioxide_core::backend::checked_mul;
use scioxide_core::{Quantity, RealScalar, SciOxideError};

const NONE: usize = usize::MAX;
const STORAGE: Quantity = scioxide_core::quantity!("hull_state");

#[derive(Debug, Clone, Copy)]
struct Node {
    inside: i8,
    replacement: usize,
    slot: usize,
}
#[derive(Debug, Clone, Copy)]
struct Live {
    node: usize,
    head: usize,
}
#[derive(Debug)]
struct Replacement {
    children: core::ops::Range<usize>,
    visited: usize,
}
// Ridges through an inserted point, paired by an open-addressed table on an
// order-independent key hash. Each ridge of a closed cone occurs exactly twice.
#[derive(Debug, Default)]
struct Cone {
    keys: Vec<usize>,
    records: Vec<(usize, usize, u64)>,
    table: Vec<usize>,
    matched: Vec<bool>,
}
impl Cone {
    fn clear(&mut self) {
        self.keys.clear();
        self.records.clear();
    }
    fn push(
        &mut self,
        vertices: &[usize],
        slot: usize,
        opposite: usize,
        storage: &Storage,
    ) -> Result<()> {
        storage.reserve(&mut self.keys, vertices.len() - 1)?;
        storage.reserve(&mut self.records, 1)?;
        let mut hash = 0_u64;
        for (index, &vertex) in vertices.iter().enumerate() {
            if index != opposite {
                self.keys.push(vertex);
                hash = hash.wrapping_add(priority(vertex));
            }
        }
        self.records.push((slot, opposite, hash));
        Ok(())
    }
    fn same(&self, a: usize, b: usize, width: usize) -> bool {
        let a = &self.keys[a * width..(a + 1) * width];
        let b = &self.keys[b * width..(b + 1) * width];
        a.iter().all(|vertex| b.contains(vertex))
    }
    fn pair(
        &mut self,
        width: usize,
        storage: &Storage,
        mut link: impl FnMut((usize, usize), (usize, usize)),
    ) -> Result<()> {
        let count = self.records.len();
        let capacity = checked_mul(count, 2, STORAGE)?.next_power_of_two().max(8);
        self.table.clear();
        self.matched.clear();
        storage.reserve(&mut self.table, capacity)?;
        storage.reserve(&mut self.matched, count)?;
        self.table.resize(capacity, NONE);
        self.matched.resize(count, false);
        let mask = capacity - 1;
        for record in 0..count {
            let hash = self.records[record].2;
            let mut index = (hash as usize) & mask;
            loop {
                let other = self.table[index];
                if other == NONE {
                    self.table[index] = record;
                    break;
                }
                if self.records[other].2 == hash && self.same(other, record, width) {
                    if self.matched[other] {
                        return Err(unstable("nonmanifold insertion cone"));
                    }
                    self.matched[other] = true;
                    self.matched[record] = true;
                    let (a, ai, _) = self.records[other];
                    let (b, bi, _) = self.records[record];
                    link((a, ai), (b, bi));
                    break;
                }
                index = (index + 1) & mask;
            }
        }
        if self.matched.iter().any(|&matched| !matched) {
            return Err(unstable("open insertion cone"));
        }
        Ok(())
    }
}
/// Vertex IDs are sorted sites, which also determine insertion priority, so
/// construction work and tie encounters do not depend on input row order.
#[derive(Debug)]
pub(super) struct Engine {
    lifted: bool,
    history: bool,
    width: usize,
    seed: Vec<usize>,
    nodes: Vec<Node>,
    vertices: Vec<usize>,
    neighbors: Vec<usize>,
    events: Vec<Replacement>,
    slots: Vec<Live>,
    free: Vec<usize>,
    owner: Vec<usize>,
    next: Vec<usize>,
}
// Retain a nonsingular minor of the accepted rows. A candidate extends their
// span iff bordering this minor with one of the unused columns is nonzero.
// Thus each site needs at most d+1 exact predicates, not a search over subsets.
// Use original [x, 1] or homogeneous rows: rounded elimination, differences or
// division would lose exact rank for thin frames and extreme coordinates.
fn initial_simplex<S: RealScalar>(
    cloud: &Cloud<S>,
    lifted: bool,
    work: &mut Work<'_, '_, '_, '_>,
) -> Result<Vec<usize>> {
    let storage = work.storage;
    let frame = storage.frame();

    let d = cloud.dimension;
    let width = d + usize::from(lifted);
    if cloud.len() <= width {
        return Err(SciOxideError::singular(None));
    }
    let mut order = storage.capacity(cloud.len(), STORAGE)?;
    order.extend(0..cloud.len());
    order.sort_unstable_by_key(|&i| (priority(i), i));
    let mut seed = storage.capacity(width + 1, STORAGE)?;
    let mut columns = storage.capacity(d + 1, STORAGE)?;
    for &site in &order {
        extend_affine_basis(cloud, site, &mut seed, &mut columns, work)?;
        if seed.len() == d + 1 {
            break;
        }
    }
    if seed.len() != d + 1 {
        return Err(SciOxideError::singular(None));
    }
    if lifted {
        // An affine basis plus any other unique site has a nonzero lifted
        // determinant or a nonzero cofactor for the existing symbolic policy.
        // The earliest unused site gives the first admissible perturbed basis
        // in priority order. Strict policy inspects published cells instead.
        let extra = order
            .iter()
            .copied()
            .find(|site| !seed.contains(site))
            .ok_or(SciOxideError::singular(None))?;
        seed.push(extra);
        seed.sort_unstable_by_key(|&i| (priority(i), i));
        work.sign(cloud, &seed, true, true)?;
    }
    let result = seed;
    frame.keep(0, retained(&result));
    Ok(result)
}
// Extend a nonsingular augmented-coordinate minor without rounded elimination.
// Callers reserve dimension + 1 entries in both buffers before visiting sites.
pub(super) fn extend_affine_basis<S: RealScalar>(
    cloud: &Cloud<S>,
    site: usize,
    seed: &mut Vec<usize>,
    columns: &mut Vec<usize>,
    work: &mut Work<'_, '_, '_, '_>,
) -> Result<()> {
    let d = cloud.dimension;
    let count = seed.len() + 1;
    // Prefer the affine constant, but do not assume a homogeneous row's
    // last coefficient is nonzero (the interior point need not be zero).
    for column in core::iter::once(d).chain(0..d) {
        if columns.contains(&column) {
            continue;
        }
        for (row, &point) in seed.iter().chain(core::iter::once(&site)).enumerate() {
            for (axis, coordinate) in columns
                .iter()
                .copied()
                .chain(core::iter::once(column))
                .enumerate()
            {
                work.workspace.rows[row * count + axis] =
                    if let Some(homogeneous) = &cloud.homogeneous {
                        homogeneous[point * (d + 1) + coordinate]
                    } else if coordinate == d {
                        1.
                    } else {
                        cloud.point(point)[coordinate].try_to_f64_exact()?
                    };
            }
        }
        if work.determinant(count, count, false, true)? != 0 {
            seed.push(site);
            columns.push(column);
            break;
        }
    }
    Ok(())
}
impl Engine {
    pub(super) fn retained_bytes(&self) -> usize {
        retained(&self.seed)
            + retained(&self.nodes)
            + retained(&self.vertices)
            + retained(&self.neighbors)
            + retained(&self.events)
            + retained(&self.slots)
            + retained(&self.free)
            + retained(&self.owner)
            + retained(&self.next)
    }

    pub(super) fn new<S: RealScalar>(
        cloud: &Cloud<S>,
        lifted: bool,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<Self> {
        let storage = work.storage;
        let frame = storage.frame();

        let width = cloud.dimension + usize::from(lifted);

        let seed = initial_simplex(cloud, lifted, work)?;
        if width + 1 > work.options.limits.max_facets {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("geometry_facets"),
                width + 1,
                work.options.limits.max_facets,
            ));
        }
        let mut nodes = storage.capacity(width + 1, STORAGE)?;
        let mut slots = storage.capacity(width + 1, STORAGE)?;
        let cells = checked_mul(width, width + 1, STORAGE)?;
        let mut vertices = storage.capacity(cells, STORAGE)?;
        let mut neighbors = storage.capacity(cells, STORAGE)?;
        for i in 0..=width {
            let start = vertices.len();
            vertices.extend(
                seed.iter()
                    .enumerate()
                    .filter_map(|(j, &v)| (i != j).then_some(v)),
            );
            let inside =
                work.sign_with_point(cloud, &vertices[start..], seed[i], lifted, lifted)?;
            neighbors.extend((0..=width).filter(|&j| j != i));
            slots.push(Live {
                node: i,
                head: NONE,
            });
            nodes.push(Node {
                inside,
                replacement: NONE,
                slot: i,
            });
        }
        let result = Self {
            lifted,
            history: cloud.homogeneous.is_some(),
            width,
            seed,
            nodes,
            vertices,
            neighbors,
            slots,
            events: Vec::new(),
            free: Vec::new(),
            owner: Vec::new(),
            next: Vec::new(),
        };
        frame.keep(0, result.retained_bytes());
        Ok(result)
    }
    fn vertices(&self, node: usize) -> &[usize] {
        &self.vertices[node * self.width..(node + 1) * self.width]
    }
    fn neighbors(&self, slot: usize) -> &[usize] {
        &self.neighbors[slot * self.width..(slot + 1) * self.width]
    }
    #[inline]
    fn sign<S: RealScalar>(
        &self,
        cloud: &Cloud<S>,
        node: usize,
        point: usize,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<i8> {
        if self.vertices(node).contains(&point) {
            return Ok(0);
        }
        work.sign_with_point(cloud, self.vertices(node), point, self.lifted, self.lifted)
    }
    // `sign` for a point known not to be a facet vertex.
    #[inline]
    fn outside_sign<S: RealScalar>(
        &self,
        cloud: &Cloud<S>,
        node: usize,
        point: usize,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<i8> {
        debug_assert!(!self.vertices(node).contains(&point));
        work.sign_with_point(cloud, self.vertices(node), point, self.lifted, self.lifted)
    }
    fn attach(&mut self, point: usize, slot: usize) {
        self.owner[point] = slot;
        self.next[point] = self.slots[slot].head;
        self.slots[slot].head = point;
    }
    // History DAG search. A replacement cone covers the old outside region;
    // shared replacement lists are visited once per original input point.
    fn locate<S: RealScalar>(
        &mut self,
        cloud: &Cloud<S>,
        point: usize,
        stack: &mut Vec<usize>,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<Option<(usize, bool)>> {
        let storage = work.storage;
        stack.clear();
        stack.extend(0..=self.width);
        let mut boundary = NONE;
        while let Some(node) = stack.pop() {
            let sign = self.sign(cloud, node, point, work)?;
            if sign == self.nodes[node].inside {
                continue;
            }
            let event = self.nodes[node].replacement;
            if event == NONE {
                if sign != 0 {
                    return Ok(Some((self.nodes[node].slot, true)));
                }
                boundary = self.nodes[node].slot;
            } else if self.events[event].visited != point {
                self.events[event].visited = point;
                storage.reserve(stack, self.events[event].children.len())?;
                stack.extend(self.events[event].children.clone());
            }
        }
        Ok((boundary != NONE).then_some((boundary, false)))
    }
    pub(super) fn insert_points<S: RealScalar>(
        &mut self,
        cloud: &Cloud<S>,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<()> {
        let storage = work.storage;
        let before = self.retained_bytes();
        let frame = storage.frame();

        let count = cloud.len();
        let additional = count;
        storage.reserve(&mut self.owner, additional)?;
        storage.reserve(&mut self.next, additional)?;
        self.owner.resize(count, NONE);
        self.next.resize(count, NONE);
        let mut pending = storage.capacity(cloud.len(), STORAGE)?;
        pending.extend((0..count).filter(|p| !self.seed.contains(p)));
        // Fixed keyed SplitMix priorities of canonical (coordinate-sorted) site
        // IDs make insertion deterministic and independent of input row order.
        pending.sort_unstable_by_key(|&p| (priority(p), p));
        let limit = work.options.limits.max_facets;
        let mut stack = storage.capacity(self.slots.len(), STORAGE)?;
        for &point in &pending {
            if let Some((slot, true)) = self.locate(cloud, point, &mut stack, work)? {
                self.attach(point, slot);
            }
        }
        let mut marks = storage.filled(self.slots.len(), 0_u8, STORAGE)?;
        let mut touched = storage.capacity(self.slots.len(), STORAGE)?;
        let mut visible = storage.capacity(self.slots.len(), STORAGE)?;
        let mut horizon = storage.capacity(self.slots.len(), STORAGE)?;
        let mut orphans = storage.capacity(count, STORAGE)?;
        let mut cone = Cone::default();
        for point in pending {
            work.iteration()?;
            if self.owner[point] == NONE {
                continue;
            }
            let additional = self.slots.len() - marks.len();
            storage.reserve(&mut marks, additional)?;
            marks.resize(self.slots.len(), 0);
            stack.clear();
            touched.clear();
            visible.clear();
            horizon.clear();
            orphans.clear();
            stack.push(self.owner[point]);
            while let Some(slot) = stack.pop() {
                if marks[slot] != 0 {
                    continue;
                }
                storage.reserve(&mut touched, 1)?;
                touched.push(slot);
                let node = self.slots[slot].node;
                // An uninserted point is never a vertex of a live facet.
                let sign = self.outside_sign(cloud, node, point, work)?;
                if sign == 0 || sign == self.nodes[node].inside {
                    marks[slot] = 1;
                    continue;
                }
                marks[slot] = 2;
                storage.reserve(&mut visible, 1)?;
                visible.push(slot);
                storage.reserve(&mut stack, self.width)?;
                stack.extend_from_slice(self.neighbors(slot));
            }
            for &slot in &visible {
                for (opposite, &neighbor) in self.neighbors(slot).iter().enumerate() {
                    if marks[neighbor] != 2 {
                        let back = self
                            .neighbors(neighbor)
                            .iter()
                            .position(|&n| n == slot)
                            .ok_or_else(|| unstable("asymmetric hull adjacency"))?;
                        storage.reserve(&mut horizon, 1)?;
                        horizon.push((slot, opposite, neighbor, back));
                    }
                }
            }
            let live_count = self.slots.len() - self.free.len();
            if live_count - visible.len() + horizon.len() > limit {
                return Err(SciOxideError::resource_limit(
                    scioxide_core::quantity!("geometry_facets"),
                    live_count - visible.len() + horizon.len(),
                    limit,
                ));
            }
            // All allocations/failures occur in this call's private candidate.
            let first = self.nodes.len();
            storage.reserve(&mut self.nodes, horizon.len())?;
            storage.reserve(
                &mut self.slots,
                horizon.len().saturating_sub(self.free.len()),
            )?;
            if self.history {
                storage.reserve(&mut self.events, 1)?;
            }
            storage.reserve(&mut self.free, visible.len())?;
            storage.reserve(
                &mut self.vertices,
                checked_mul(horizon.len(), self.width, STORAGE)?,
            )?;
            storage.reserve(
                &mut self.neighbors,
                checked_mul(
                    horizon.len().saturating_sub(self.free.len()),
                    self.width,
                    STORAGE,
                )?,
            )?;
            for &(old, opposite, _, _) in &horizon {
                let source = self.slots[old].node * self.width;
                let start = self.vertices.len();
                for i in 0..self.width {
                    if i != opposite {
                        self.vertices.push(self.vertices[source + i]);
                    }
                }
                self.vertices.push(point);
                debug_assert_eq!(self.vertices.len() - start, self.width);
                let inside = cone_inside(
                    self.nodes[self.slots[old].node].inside,
                    self.width,
                    opposite,
                );
                self.nodes.push(Node {
                    inside,
                    replacement: NONE,
                    slot: NONE,
                });
            }
            for &slot in &visible {
                let mut orphan = self.slots[slot].head;
                while orphan != NONE {
                    let next = self.next[orphan];
                    self.owner[orphan] = NONE;
                    if orphan != point {
                        orphans.push(orphan);
                    }
                    orphan = next;
                }
                if self.history {
                    self.nodes[self.slots[slot].node].replacement = self.events.len();
                }
                self.nodes[self.slots[slot].node].slot = NONE;
                self.slots[slot].head = NONE;
                self.free.push(slot);
            }
            // Reusing slots does not invalidate history nodes or pending owners.
            cone.clear();
            for (offset, &(_, _, neighbor, back)) in horizon.iter().enumerate() {
                let node = first + offset;
                let slot = if let Some(slot) = self.free.pop() {
                    slot
                } else {
                    self.slots.push(Live {
                        node: NONE,
                        head: NONE,
                    });
                    self.slots.len() - 1
                };
                self.nodes[node].slot = slot;
                self.slots[slot].node = node;
                self.slots[slot].head = NONE;
                self.neighbors.resize(self.slots.len() * self.width, NONE);
                self.neighbors[slot * self.width..(slot + 1) * self.width].fill(NONE);
                self.neighbors[slot * self.width + self.width - 1] = neighbor;
                self.neighbors[neighbor * self.width + back] = slot;
                for opposite in 0..self.width - 1 {
                    cone.push(self.vertices(node), slot, opposite, storage)?;
                }
            }
            let width = self.width;
            let neighbors = &mut self.neighbors;
            cone.pair(width - 1, storage, |(a, ai), (b, bi)| {
                neighbors[a * width + ai] = b;
                neighbors[b * width + bi] = a;
            })?;
            if self.history {
                self.events.push(Replacement {
                    children: first..self.nodes.len(),
                    visited: NONE,
                });
            }
            for orphan in orphans.iter().copied() {
                for node in first..self.nodes.len() {
                    let sign = self.outside_sign(cloud, node, orphan, work)?;
                    if sign != 0 && sign != self.nodes[node].inside {
                        self.attach(orphan, self.nodes[node].slot);
                        break;
                    }
                }
            }
            if !self.history {
                // Free slots are consumed before new slots are appended, so
                // every destination is at or before its source node. Forward
                // compaction cannot overwrite a later cone facet. Ordinary
                // hulls never locate through history after initial assignment.
                for node in first..self.nodes.len() {
                    let slot = self.nodes[node].slot;
                    debug_assert!(slot <= node);
                    self.vertices.copy_within(
                        node * self.width..(node + 1) * self.width,
                        slot * self.width,
                    );
                    self.nodes[slot] = self.nodes[node];
                    self.slots[slot].node = slot;
                }
                self.nodes.truncate(self.slots.len());
                self.vertices.truncate(self.slots.len() * self.width);
            }
            for &slot in &touched {
                marks[slot] = 0;
            }
        }
        frame.keep(before, self.retained_bytes());
        Ok(())
    }
    // Seed hull vertices directly from their final incidence; locate only the
    // remaining original homogeneous points. Traverse each exact supporting-face
    // star to retain redundant tight inequalities without an all-facet scan.
    pub(super) fn boundary_incidence<S: RealScalar>(
        &mut self,
        cloud: &Cloud<S>,
        topology: &mut Topology,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<()> {
        let storage = work.storage;
        let before = topology.retained_bytes();
        let frame = storage.frame();

        let mut ids = storage.filled(self.slots.len(), NONE, STORAGE)?;
        let mut starts = storage.filled(cloud.mapping.len(), NONE, STORAGE)?;
        let mut query = storage.capacity(self.width + 1, STORAGE)?;
        for (slot, live) in self.slots.iter().enumerate() {
            if self.nodes[live.node].slot == NONE {
                continue;
            }
            query.clear();
            query.extend_from_slice(self.vertices(live.node));
            for &point in self.vertices(live.node) {
                starts[point] = slot;
            }
            query.sort_unstable();
            ids[slot] = topology
                .facets
                .search(&query)
                .map_err(|_| unstable("missing published facet"))?;
        }
        for event in &mut self.events {
            event.visited = NONE;
        }
        let mut stack = storage.capacity(self.slots.len(), STORAGE)?;
        let mut seen = storage.filled(self.slots.len(), NONE, STORAGE)?;
        for (point, &known) in starts.iter().enumerate() {
            let start = if known != NONE {
                known
            } else {
                let Some((start, outside)) = self.locate(cloud, point, &mut stack, work)? else {
                    continue;
                };
                if outside {
                    return Err(unstable("input outside completed hull"));
                }
                start
            };
            stack.clear();
            stack.push(start);
            while let Some(slot) = stack.pop() {
                if seen[slot] == point {
                    continue;
                }
                seen[slot] = point;
                if self.sign(cloud, self.slots[slot].node, point, work)? != 0 {
                    continue;
                }
                storage.reserve(&mut topology.boundary_points, 1)?;
                topology.boundary_points.push((ids[slot], point));
                storage.reserve(&mut stack, self.width)?;
                stack.extend_from_slice(self.neighbors(slot));
            }
        }
        topology.boundary_points.sort_unstable();
        frame.keep(before, topology.retained_bytes());
        Ok(())
    }
    // Canonical rows in sorted order, with adjacency translated from the live
    // slots instead of re-matching every ridge.
    pub(super) fn topology(&self, storage: &Storage) -> Result<Topology> {
        let frame = storage.frame();

        let live = self.slots.len() - self.free.len();
        let mut facets = Facets::new(self.width, live, storage)?;
        let mut live_slots = storage.capacity(live, STORAGE)?;
        let mut vertices = storage.capacity(self.width, STORAGE)?;
        for (index, slot) in self.slots.iter().enumerate() {
            if self.nodes[slot.node].slot == NONE {
                continue;
            }
            live_slots.push(index);
            vertices.clear();
            vertices.extend_from_slice(self.vertices(slot.node));
            let inversions = vertices
                .iter()
                .enumerate()
                .map(|(i, v)| vertices[i + 1..].iter().filter(|w| *w < v).count())
                .sum::<usize>();
            vertices.sort_unstable();
            let inside = self.nodes[slot.node].inside;
            facets.push(
                &vertices,
                if inversions.is_multiple_of(2) {
                    inside
                } else {
                    -inside
                },
                storage,
            )?;
        }
        let mut order = storage.capacity(live, STORAGE)?;
        order.extend(0..live);
        order.sort_unstable_by(|&a, &b| facets.at(a).vertices.cmp(facets.at(b).vertices));
        let mut rank = storage.filled(self.slots.len(), NONE, STORAGE)?;
        for (position, &facet) in order.iter().enumerate() {
            rank[live_slots[facet]] = position;
        }
        let mut sorted = Facets::new(self.width, live, storage)?;
        let mut neighbors = storage.filled(
            checked_mul(live, self.width, STORAGE)?,
            None,
            scioxide_core::quantity!("facet_adjacency"),
        )?;
        for (position, &facet) in order.iter().enumerate() {
            let row = facets.at(facet);
            sorted.push(row.vertices, row.inside, storage)?;
            let slot = live_slots[facet];
            let original = self.vertices(self.slots[slot].node);
            for (&vertex, &neighbor) in original.iter().zip(self.neighbors(slot)) {
                let opposite = row
                    .vertices
                    .iter()
                    .position(|&v| v == vertex)
                    .ok_or_else(|| unstable("inconsistent hull facet"))?;
                let neighbor = rank
                    .get(neighbor)
                    .copied()
                    .filter(|&neighbor| neighbor != NONE)
                    .ok_or_else(|| unstable("open hull boundary"))?;
                neighbors[position * self.width + opposite] = Some(neighbor);
            }
        }
        let result = Topology::from_parts(sorted, neighbors);
        frame.keep(0, result.retained_bytes());
        Ok(result)
    }
}

// If V is visible, sign(V,p)=-inside(V). Moving its omitted vertex k to
// the end takes width-k swaps, hence sign([V without k,p],V[k]) is
// (-1)^(width-k+1)*inside(V). This also holds for the alternating symbolic
// lifted determinant and for homogeneous rows, without an extra predicate.
fn cone_inside(inside: i8, width: usize, opposite: usize) -> i8 {
    if (width - opposite).is_multiple_of(2) {
        -inside
    } else {
        inside
    }
}
fn priority(index: usize) -> u64 {
    let mut word = (index as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    word ^ (word >> 31)
}

pub(super) fn hull<S: RealScalar>(
    cloud: &Cloud<S>,
    lifted: bool,
    work: &mut Work<'_, '_, '_, '_>,
) -> Result<Topology> {
    let storage = work.storage;
    let frame = storage.frame();

    let mut engine = Engine::new(cloud, lifted, work)?;
    engine.insert_points(cloud, work)?;
    let mut topology = engine.topology(storage)?;
    if cloud.homogeneous.is_some() {
        engine.boundary_incidence(cloud, &mut topology, work)?;
    }
    let result = topology;
    frame.keep(0, result.retained_bytes());
    Ok(result)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::tessellation::tests::with_points;
    use crate::tessellation::tests::{test_requirements, test_workspace};
    use crate::tessellation::{GeometryDegeneracy, GeometryLimits, GeometryOptions};

    #[test]
    fn cone_orientation_matches_exact_row_permutations() {
        let cases: &[(usize, bool, &[f64])] = &[
            (1, false, &[0., 1.]),
            (2, false, &[0., 0., 1., 0., 0., 1.]),
            (3, false, &[0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]),
            (
                4,
                false,
                &[
                    0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
                ],
            ),
            // Exact lifted ties exercise the alternating symbolic determinant.
            (2, true, &[0., 0., 1., 0., 1., 1., 0., 1.]),
            (
                3,
                true,
                &[1., 0., 0., 0., 1., 0., 0., 0., 1., -1., 0., 0., 0., -1., 0.],
            ),
        ];
        for &(d, lifted, values) in cases {
            for weight in [
                None,
                Some(1.),
                Some(-2_f64.powi(900)),
                Some(2_f64.powi(-900)),
            ] {
                let homogeneous = weight.is_some();
                if homogeneous && lifted {
                    continue;
                }
                let storage = Storage::new(usize::MAX, 0).unwrap();
                let o = options().with_limits(options().limits().with_max_dimension(4));
                let mut workspace = test_workspace(test_requirements(d, o).unwrap()).unwrap();
                with_points(values, d, |p| {
                    let mut cloud = Cloud::new(p, o, &storage)?;
                    if let Some(weight) = weight {
                        cloud.homogeneous = Some(
                            cloud
                                .points
                                .chunks_exact(d)
                                .flat_map(|row| row.iter().copied().chain(core::iter::once(weight)))
                                .collect(),
                        );
                    }
                    let mut work = Work::new(
                        &mut workspace,
                        d,
                        o,
                        None,
                        scioxide_core::argument!("points"),
                        &storage,
                    )?;
                    let width = d + usize::from(lifted);
                    let old: Vec<_> = (0..width).collect();
                    let rows = cloud.homogeneous.take();
                    let reference = work.sign_with_point(&cloud, &old, width, lifted, lifted)?;
                    cloud.homogeneous = rows;
                    let outside = work.sign_with_point(&cloud, &old, width, lifted, lifted)?;
                    assert_eq!(
                        outside,
                        if weight.is_some_and(|w| w < 0.) {
                            -reference
                        } else {
                            reference
                        }
                    );
                    assert_ne!(outside, 0);
                    for opposite in 0..width {
                        let cone: Vec<_> = old
                            .iter()
                            .copied()
                            .filter(|&v| v != opposite)
                            .chain(core::iter::once(width))
                            .collect();
                        let witness =
                            work.sign_with_point(&cloud, &cone, opposite, lifted, lifted)?;
                        assert_eq!(
                            cone_inside(-outside, width, opposite),
                            witness,
                            "d={d}, lifted={lifted}, homogeneous={homogeneous}, opposite={opposite}"
                        );
                    }
                    Ok(())
                })
                .unwrap();
            }
        }
    }
    fn options() -> GeometryOptions {
        GeometryOptions::new().with_limits(
            GeometryLimits::new()
                .with_max_dimension(3)
                .with_max_points(1024)
                .with_max_facets(4096)
                .with_max_predicate_terms(100_000_000)
                .with_max_storage_bytes(usize::MAX),
        )
    }
    fn points(n: usize) -> Vec<f64> {
        (0..n)
            .flat_map(|i| {
                let x = (priority(3 * i) >> 11) as f64 / ((1_u64 << 53) as f64) - 0.5;
                let y = (priority(3 * i + 1) >> 11) as f64 / ((1_u64 << 53) as f64) - 0.5;
                let z = (priority(3 * i + 2) >> 11) as f64 / ((1_u64 << 53) as f64) - 0.5;
                let norm = (x * x + y * y + z * z).sqrt();
                [x / norm, y / norm, z / norm]
            })
            .collect()
    }
    #[test]
    fn construction_owners_and_adjacency_remain_valid() {
        let storage = Storage::new(usize::MAX, 0).unwrap();

        let values = points(96);
        let options = options();
        let mut workspace = test_workspace(test_requirements(3, options).unwrap()).unwrap();

        for n in [8, 23, 24, 63, 96] {
            with_points(&values[..3 * n], 3, |p| {
                let cloud = Cloud::new(p, options, &storage)?;
                let mut work = Work::new(
                    &mut workspace,
                    3,
                    options,
                    None,
                    scioxide_core::argument!("points"),
                    &storage,
                )?;
                let mut engine = Engine::new(&cloud, false, &mut work)?;
                engine.insert_points(&cloud, &mut work)?;
                assert!(engine.events.is_empty());
                assert_eq!(engine.nodes.len(), engine.slots.len());
                for (index, slot) in engine.slots.iter().enumerate() {
                    if engine.nodes[slot.node].slot == NONE {
                        assert!(engine.free.contains(&index));
                        continue;
                    }
                    assert_eq!(engine.nodes[slot.node].slot, index);
                    for (omitted, &neighbor) in engine.neighbors(index).iter().enumerate() {
                        assert!(engine.neighbors(neighbor).contains(&index));
                        let vertices = engine.vertices(slot.node);
                        assert!(
                            vertices
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| *i != omitted)
                                .all(|(_, v)| engine
                                    .vertices(engine.slots[neighbor].node)
                                    .contains(v))
                        );
                    }
                    let mut point = slot.head;
                    let mut visited = 0;
                    while point != NONE {
                        assert_eq!(engine.owner[point], index);
                        assert_ne!(engine.owner[point], NONE);
                        point = engine.next[point];
                        visited += 1;
                        assert!(visited <= n);
                    }
                }
                let nodes = engine.nodes.len();
                let first = engine.topology(&storage)?;
                let terms = work.workspace.terms;
                let second = engine.topology(&storage)?;
                assert_eq!(engine.nodes.len(), nodes);
                assert_eq!(work.workspace.terms, terms);
                assert_eq!(first.neighbors, second.neighbors);
                Ok(())
            })
            .unwrap();
        }
    }
    #[test]
    fn conflict_construction_avoids_quadratic_predicate_growth() {
        let terms = |n| {
            with_points(&points(n), 3, |p| {
                Ok(crate::ConvexHull::with_options(p, options())?
                    .provenance()
                    .predicate_terms)
            })
            .unwrap()
        };
        let small = terms(128);
        let large = terms(512);
        // Four times the sites should fit a generous n log n work envelope;
        // the old all-facet scan grows approximately sixteenfold on a sphere.
        assert!(large < 9 * small, "{small} -> {large}");
    }

    const SEED_TERMS: usize = 10_000_000;
    fn seed_work(
        cloud: &Cloud<f64>,
        lifted: bool,
        limit: usize,
        degeneracy: GeometryDegeneracy,
    ) -> (Result<Vec<usize>>, usize) {
        let storage = Storage::new(usize::MAX, 0).unwrap();

        let mut options = options();
        options.limits.max_dimension = cloud.dimension;
        options.limits.max_points = cloud.len();
        options.limits.max_predicate_terms = limit;
        options.degeneracy = degeneracy;
        let mut workspace =
            test_workspace(test_requirements(cloud.dimension, options).unwrap()).unwrap();
        let mut work = Work::new(
            &mut workspace,
            cloud.dimension,
            options,
            None,
            scioxide_core::argument!("points"),
            &storage,
        )
        .unwrap();
        let result = initial_simplex(cloud, lifted, &mut work);
        let terms = work.workspace.terms;
        if let Ok(seed) = &result {
            assert_ne!(work.sign(cloud, seed, lifted, lifted).unwrap(), 0);
        }
        (result, terms)
    }

    fn deficient_cloud(n: usize, d: usize, homogeneous: bool) -> Cloud<f64> {
        let points: Vec<_> = (0..n)
            .flat_map(|i| {
                let coordinates = [i as f64, (i % 31) as f64, (i % 17) as f64];
                let last = coordinates[..d - 1]
                    .iter()
                    .enumerate()
                    .fold(1., |sum, (axis, x)| sum + (2 << axis) as f64 * x);
                coordinates[..d - 1]
                    .iter()
                    .copied()
                    .chain(core::iter::once(last))
                    .collect::<Vec<_>>()
            })
            .collect();
        let homogeneous = homogeneous.then(|| {
            points
                .chunks_exact(d)
                .flat_map(|row| row.iter().copied().chain(core::iter::once(1.)))
                .collect()
        });
        Cloud {
            dimension: d,
            filter_envelope: false,
            points,
            mapping: (0..n).collect(),
            representatives: (0..n).collect(),
            homogeneous,
        }
    }

    #[test]
    fn deficient_seed_predicate_terms_grow_linearly() {
        for d in 2..=4 {
            for (lifted, homogeneous) in [(false, false), (true, false), (false, true)] {
                let terms = |n| {
                    let cloud = deficient_cloud(n, d, homogeneous);
                    let (result, terms) = seed_work(
                        &cloud,
                        lifted,
                        SEED_TERMS,
                        GeometryDegeneracy::Deterministic,
                    );
                    assert!(
                        matches!(result, Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::Singular)
                    );
                    terms
                };
                let small = terms(2_500);
                let large = terms(10_000);
                assert!(small > 0);
                assert!(
                    large <= 5 * small,
                    "d={d}, lifted={lifted}, homogeneous={homogeneous}: {small} -> {large}"
                );
            }
        }
    }

    #[test]
    fn deficient_seed_budget_boundary_and_strict_rank_error() {
        for (lifted, homogeneous) in [(false, false), (true, false), (false, true)] {
            let cloud = deficient_cloud(128, 3, homogeneous);
            for policy in [
                GeometryDegeneracy::Deterministic,
                GeometryDegeneracy::Strict,
            ] {
                let (result, terms) = seed_work(&cloud, lifted, SEED_TERMS, policy);
                assert!(
                    matches!(result, Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::Singular)
                );
                for limit in [0, terms - 1, terms, terms + 1] {
                    let (result, used) = seed_work(&cloud, lifted, limit, policy);
                    if limit < terms {
                        assert!(matches!(
                            result,
                            Err(ref error) if matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource, .. } if resource == scioxide_core::quantity!("geometry_predicate_terms"))
                        ));
                        assert_eq!(used, limit);
                    } else {
                        assert!(
                            matches!(result, Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::Singular)
                        );
                        assert_eq!(used, terms);
                    }
                }
            }
        }
    }

    #[test]
    fn seed_accepts_rank_reached_at_last_priority_and_extreme_exact_frames() {
        for d in 2..=4 {
            let mut cloud = deficient_cloud(128, d, false);
            let last = (0..cloud.len()).max_by_key(|&i| priority(i)).unwrap();
            cloud.points[last * d + d - 1] += 1.;
            for lifted in [false, true] {
                let (result, _) = seed_work(
                    &cloud,
                    lifted,
                    SEED_TERMS,
                    GeometryDegeneracy::Deterministic,
                );
                assert!(result.unwrap().contains(&last));
            }
        }
        for scale in [f64::from_bits(1), 1., f64::MAX] {
            // A subnormal-height frame is full-rank, as is one whose lifted
            // norm overflows f64. The predicate must classify both exactly.
            let points = [
                0., 0., 0., 0., 0., scale, 1., 0., 0., 0., 1., 0., -1., 0., 0.,
            ];
            let cloud = Cloud {
                dimension: 3,
                filter_envelope: false,
                points: points.to_vec(),
                mapping: (0..5).collect(),
                representatives: (0..5).collect(),
                homogeneous: None,
            };
            for lifted in [false, true] {
                assert!(
                    seed_work(
                        &cloud,
                        lifted,
                        SEED_TERMS,
                        GeometryDegeneracy::Deterministic
                    )
                    .0
                    .is_ok()
                );
            }
        }
        let cloud = Cloud {
            dimension: 2,
            filter_envelope: false,
            points: Vec::new(),
            mapping: (0..4).collect(),
            representatives: (0..4).collect(),
            homogeneous: Some(alloc::vec![1., 0., 0., 2., 0., 0., 0., 1., 0., 0., 0., 1.]),
        };
        assert!(
            seed_work(&cloud, false, SEED_TERMS, GeometryDegeneracy::Deterministic)
                .0
                .is_ok()
        );
    }
}
