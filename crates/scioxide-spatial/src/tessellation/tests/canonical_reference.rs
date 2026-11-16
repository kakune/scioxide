//! Exhaustive reference implementation, restricted to small correctness fixtures.
use super::super::topology::{Facets, Topology};
use super::*;
const STORAGE: Quantity = scioxide_core::quantity!("supporting_faces");

fn root(parents: &mut [usize], mut index: usize) -> usize {
    while parents[index] != index {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    index
}
pub(super) fn groups<S: RealScalar>(
    cloud: &Cloud<S>,
    topology: &Topology,
    lifted: bool,
    work: &mut Work<'_, '_, '_, '_>,
) -> Result<Vec<usize>> {
    let storage = work.storage;
    let frame = storage.frame();

    let mut groups = storage.capacity(topology.facets.len(), STORAGE)?;
    groups.extend(0..topology.facets.len());
    for (i, facet) in topology.facets.iter().enumerate() {
        for &neighbor in topology.adjacent(i)?.iter().flatten() {
            if neighbor <= i {
                continue;
            }
            let opposite = topology
                .facets
                .at(neighbor)
                .vertices
                .iter()
                .find(|v| !facet.vertices.contains(v))
                .copied()
                .ok_or_else(|| unstable("duplicate geometry facet"))?;
            if work.sign_with_point(cloud, facet.vertices, opposite, lifted, false)? == 0 {
                let a = root(&mut groups, i);
                let b = root(&mut groups, neighbor);
                groups[a.max(b)] = a.min(b);
            }
        }
    }
    for i in 0..groups.len() {
        groups[i] = root(&mut groups, i);
    }
    let result = groups;
    frame.keep(0, retained(&result));
    Ok(result)
}

pub(super) fn hull<S: RealScalar>(
    cloud: &Cloud<S>,
    topology: Topology,
    work: &mut Work<'_, '_, '_, '_>,
) -> Result<(Topology, Vec<usize>)> {
    let storage = work.storage;
    let before = topology.retained_bytes();
    let frame = storage.frame();

    let groups = groups(cloud, &topology, false, work)?;
    if groups.iter().enumerate().all(|(i, &g)| i == g) {
        frame.keep(before, topology.retained_bytes() + retained(&groups));
        return Ok((topology, groups));
    }

    // Sorted group/site incidence replaces one owned vector per supporting face.
    let mut membership = storage.capacity(topology.facets.len() * cloud.dimension, STORAGE)?;
    for (facet, &group) in topology.facets.iter().zip(&groups) {
        membership.extend(facet.vertices.iter().map(|&v| (group, v)));
    }
    membership.sort_unstable();
    membership.dedup();
    let mut faces = Faces::new(groups.len(), &membership, storage)?;
    let mut incident = storage.capacity(membership.len(), STORAGE)?;
    incident.extend(membership.iter().map(|&(g, v)| (v, g)));
    incident.sort_unstable();
    let mut extreme = storage.filled(cloud.len(), false, STORAGE)?;
    let mut intersection = storage.capacity(cloud.len(), STORAGE)?;
    let mut begin = 0;
    while begin < incident.len() {
        let vertex = incident[begin].0;
        intersection.clear();
        intersection.extend_from_slice(faces.at(incident[begin].1));
        let mut end = begin + 1;
        while end < incident.len() && incident[end].0 == vertex {
            let face = faces.at(incident[end].1);
            intersection.retain(|v| face.binary_search(v).is_ok());
            end += 1;
        }
        extreme[vertex] = intersection.len() == 1;
        begin = end;
    }
    // Compact each row in its existing range; vacant tail entries are ignored.
    for range in &mut faces.ranges {
        let mut end = range.start;
        for i in range.clone() {
            let vertex = faces.vertices[i];
            if extreme[vertex] {
                faces.vertices[end] = vertex;
                end += 1;
            }
        }
        range.end = end;
    }
    let mut witnesses = storage.filled(groups.len(), None, STORAGE)?;
    for (index, facet) in topology.facets.iter().enumerate() {
        for &neighbor in topology.adjacent(index)?.iter().flatten() {
            if groups[index] != groups[neighbor] && witnesses[groups[index]].is_none() {
                witnesses[groups[index]] = topology
                    .facets
                    .at(neighbor)
                    .vertices
                    .iter()
                    .find(|v| !facet.vertices.contains(v))
                    .copied();
            }
        }
    }
    let mut levels = storage.capacity(cloud.dimension, STORAGE)?;
    for _ in 0..cloud.dimension {
        levels.push(PullLevel::new());
    }
    let mut facets = Facets::new(cloud.dimension, topology.facets.len(), storage)?;
    let mut pulling = Pulling {
        width: cloud.dimension,
        supports: &faces,
        prefix: storage.capacity(cloud.dimension, STORAGE)?,
        output: Vec::new(),
    };
    for (group, &witness) in witnesses.iter().enumerate() {
        let face = faces.at(group);
        if face.is_empty() {
            continue;
        }
        work.check()?;
        let witness = witness.ok_or_else(|| unstable("supporting face lacks interior witness"))?;
        pulling.output.clear();
        pulling.pull(face, cloud.dimension - 1, &mut levels, work)?;
        for vertices in pulling.output.chunks_exact_mut(cloud.dimension) {
            if facets.len() == work.options.limits.max_facets {
                return Err(SciOxideError::resource_limit(
                    scioxide_core::quantity!("geometry_facets"),
                    work.options.limits.max_facets.saturating_add(1),
                    work.options.limits.max_facets,
                ));
            }
            vertices.sort_unstable();
            let inside = work.sign_with_point(cloud, vertices, witness, false, false)?;
            if inside == 0 {
                return Err(unstable("degenerate pulling simplex"));
            }
            facets.push(vertices, inside, storage)?;
        }
    }
    facets.sort(storage)?;
    let topology = Topology::new(facets, storage)?;
    if topology.neighbors.iter().any(Option::is_none) {
        return Err(unstable("open canonical hull"));
    }
    let groups = self::groups(cloud, &topology, false, work)?;
    let result = (topology, groups);
    frame.keep(before, result.0.retained_bytes() + retained(&result.1));
    Ok(result)
}

struct Faces {
    vertices: Vec<usize>,
    ranges: Vec<core::ops::Range<usize>>,
}
impl Faces {
    fn new(count: usize, membership: &[(usize, usize)], storage: &Storage) -> Result<Self> {
        let mut vertices = storage.capacity(membership.len(), STORAGE)?;
        let mut ranges = storage.capacity(count, STORAGE)?;
        let mut next = 0;
        for group in 0..count {
            let start = vertices.len();
            while next < membership.len() && membership[next].0 == group {
                vertices.push(membership[next].1);
                next += 1;
            }
            ranges.push(start..vertices.len());
        }
        Ok(Self { vertices, ranges })
    }
    fn at(&self, index: usize) -> &[usize] {
        &self.vertices[self.ranges[index].clone()]
    }
}
// One reusable arena per recursion depth, not per supporting face or cut.
struct PullLevel {
    cuts: Faces,
    order: Vec<usize>,
}
impl PullLevel {
    fn new() -> Self {
        Self {
            cuts: Faces {
                vertices: Vec::new(),
                ranges: Vec::new(),
            },
            order: Vec::new(),
        }
    }
}
struct Pulling<'a> {
    width: usize,
    supports: &'a Faces,
    prefix: Vec<usize>,
    output: Vec<usize>,
}
impl Pulling<'_> {
    fn pull(
        &mut self,
        face: &[usize],
        dimension: usize,
        levels: &mut [PullLevel],
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<()> {
        let storage = work.storage;
        work.check()?;

        if face.len() == dimension + 1 {
            if self.output.len() / self.width == work.options.limits.max_facets {
                return Err(SciOxideError::resource_limit(
                    scioxide_core::quantity!("geometry_facets"),
                    work.options.limits.max_facets.saturating_add(1),
                    work.options.limits.max_facets,
                ));
            }
            storage.reserve(&mut self.output, self.width)?;
            self.output.extend_from_slice(&self.prefix);
            self.output.extend_from_slice(face);
            return Ok(());
        }
        if face.len() <= dimension + 1 || dimension == 0 {
            return Err(unstable("invalid supporting face"));
        }
        let (level, rest) = levels
            .split_first_mut()
            .ok_or_else(|| unstable("pulling depth exceeded"))?;
        level.cuts.vertices.clear();
        level.cuts.ranges.clear();
        level.order.clear();
        for index in 0..self.supports.ranges.len() {
            let support = self.supports.at(index);
            let start = level.cuts.vertices.len();
            storage.reserve(&mut level.cuts.vertices, face.len())?;
            level.cuts.vertices.extend(
                face.iter()
                    .copied()
                    .filter(|v| support.binary_search(v).is_ok()),
            );
            let end = level.cuts.vertices.len();
            if end == start || end - start == face.len() {
                level.cuts.vertices.truncate(start);
            } else {
                storage.reserve(&mut level.cuts.ranges, 1)?;
                level.cuts.ranges.push(start..end);
            }
        }
        storage.reserve(&mut level.order, level.cuts.ranges.len())?;
        level.order.extend(0..level.cuts.ranges.len());
        level
            .order
            .sort_unstable_by(|&a, &b| level.cuts.at(a).cmp(level.cuts.at(b)));
        level
            .order
            .dedup_by(|a, b| level.cuts.at(*a) == level.cuts.at(*b));
        let apex = face[0];
        self.prefix.push(apex);
        for &index in &level.order {
            let cut = level.cuts.at(index);
            if cut.contains(&apex)
                || level.order.iter().any(|&other| {
                    let other = level.cuts.at(other);
                    other.len() > cut.len() && cut.iter().all(|v| other.binary_search(v).is_ok())
                })
            {
                continue;
            }
            self.pull(cut, dimension - 1, rest, work)?;
        }
        self.prefix.pop();
        Ok(())
    }
}
