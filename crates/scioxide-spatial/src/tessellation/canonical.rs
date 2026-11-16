//! Grouping of facets into supporting faces, and pulling subdivisions of faces.
use super::topology::{Facets, Topology};
use crate::tessellation::cloud::Cloud;
use crate::tessellation::storage::retained;
use crate::tessellation::work::Work;
use crate::tessellation::{Result, limit, unstable};
use alloc::vec::Vec;
use core::cmp::Ordering;
use scioxide_core::backend::checked_add;
use scioxide_core::backend::checked_mul;
use scioxide_core::{BudgetKind, Quantity, RealScalar, SciOxideError};
const STORAGE: Quantity = scioxide_core::quantity!("supporting_faces");
const CANONICAL_WORK: Quantity = scioxide_core::quantity!("geometry_canonical_work");

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

    // Index both directions once. Query the smaller of a vertex star and the
    // root face's adjacent supports: a pyramid apex must not trigger a global
    // support scan for each of its small non-simplicial incident faces.
    let mut membership = storage.capacity(
        checked_mul(topology.facets.len(), cloud.dimension, STORAGE)?,
        STORAGE,
    )?;
    for (facet, &group) in topology.facets.iter().zip(&groups) {
        work.canonical(facet.vertices.len())?;
        membership.extend(facet.vertices.iter().map(|&v| (group, v)));
    }
    work.canonical_sort(&mut membership)?;
    work.canonical(membership.len())?;
    membership.dedup();
    let mut faces = Faces::new(groups.len(), &membership, work)?;
    let mut incident = storage.capacity(membership.len(), STORAGE)?;
    for &(g, v) in &membership {
        work.canonical(1)?;
        incident.push((v, g));
    }
    work.canonical_sort(&mut incident)?;
    let incident = Faces::new(cloud.len(), &incident, work)?;
    let mut extreme = storage.filled(cloud.len(), false, STORAGE)?;
    let mut intersection = storage.capacity(cloud.len(), STORAGE)?;
    for (vertex, is_extreme) in extreme.iter_mut().enumerate() {
        work.canonical(1)?;
        let star = incident.at(vertex);
        // A vertex of a full-dimensional d-polytope has at least d supports.
        if star.len() < cloud.dimension {
            continue;
        }
        let mut smallest = star[0];
        for &support in star {
            work.canonical(1)?;
            if faces.at(support).len() < faces.at(smallest).len() {
                smallest = support;
            }
        }
        intersection.clear();
        for &candidate in faces.at(smallest) {
            work.canonical(1)?;
            intersection.push(candidate);
        }
        for &support in star {
            work.canonical(1)?;
            if support == smallest {
                continue;
            }
            let mut end = 0;
            for i in 0..intersection.len() {
                let candidate = intersection[i];
                if work.canonical_contains(faces.at(support), candidate)? {
                    intersection[end] = candidate;
                    end += 1;
                }
            }
            intersection.truncate(end);
            if end == 1 {
                break;
            }
        }
        *is_extreme = intersection.len() == 1;
    }
    // Compact each row in its existing range; vacant tail entries are ignored.
    for range in &mut faces.ranges {
        work.canonical(1)?;
        let mut end = range.start;
        for i in range.clone() {
            work.canonical(1)?;
            let vertex = faces.vertices[i];
            if extreme[vertex] {
                faces.vertices[end] = vertex;
                end += 1;
            }
        }
        range.end = end;
    }
    let mut witnesses = storage.filled(groups.len(), None, STORAGE)?;
    let mut adjacent = Vec::new();
    for (index, facet) in topology.facets.iter().enumerate() {
        for &neighbor in topology.adjacent(index)?.iter().flatten() {
            work.canonical(1)?;
            if groups[index] != groups[neighbor] {
                storage.reserve(&mut adjacent, 1)?;
                adjacent.push((groups[index], groups[neighbor]));
                if witnesses[groups[index]].is_none() {
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
    }
    work.canonical_sort(&mut adjacent)?;
    work.canonical(adjacent.len())?;
    adjacent.dedup();
    let adjacent = Faces::new(groups.len(), &adjacent, work)?;
    let mut levels = storage.capacity(cloud.dimension, STORAGE)?;
    for _ in 0..cloud.dimension {
        levels.push(PullLevel::new());
    }
    let mut facets = Facets::new(cloud.dimension, topology.facets.len(), storage)?;
    let mut pulling = Pulling {
        width: cloud.dimension,
        supports: &faces,
        incident: &incident,
        adjacent: &adjacent,
        root: 0,
        prefix: storage.capacity(cloud.dimension, STORAGE)?,
        output: Vec::new(),
    };
    for (group, &witness) in witnesses.iter().enumerate() {
        let face = faces.at(group);
        if face.is_empty() {
            continue;
        }
        work.canonical(1)?;
        pulling.root = group;
        let witness = witness.ok_or_else(|| unstable("supporting face lacks interior witness"))?;
        pulling.output.clear();
        pulling.pull(cloud, face, cloud.dimension - 1, &mut levels, work)?;
        for vertices in pulling.output.chunks_exact_mut(cloud.dimension) {
            if facets.len() == work.options.limits.max_facets {
                return Err(SciOxideError::resource_limit(
                    scioxide_core::quantity!("geometry_facets"),
                    work.options.limits.max_facets.saturating_add(1),
                    work.options.limits.max_facets,
                ));
            }
            work.canonical_sort(vertices)?;
            let inside = work.sign_with_point(cloud, vertices, witness, false, false)?;
            if inside == 0 {
                return Err(unstable("degenerate pulling simplex"));
            }
            facets.push(vertices, inside, storage)?;
        }
    }
    // Preserve the public facet order while keeping canonical sorting stoppable.
    let mut order = storage.capacity(facets.len(), STORAGE)?;
    work.canonical(facets.len())?;
    order.extend(0..facets.len());
    work.canonical_sort_by(&mut order, |&a, &b, work| {
        work.canonical_compare(facets.at(a).vertices, facets.at(b).vertices)
    })?;
    let mut sorted = Facets::new(cloud.dimension, facets.len(), storage)?;
    for i in order {
        let facet = facets.at(i);
        work.canonical(facet.vertices.len())?;
        sorted.push(facet.vertices, facet.inside, storage)?;
    }
    let facets = sorted;
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
    fn new(
        count: usize,
        membership: &[(usize, usize)],
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<Self> {
        let storage = work.storage;
        let mut vertices = storage.capacity(membership.len(), STORAGE)?;
        let mut ranges = storage.capacity(count, STORAGE)?;
        let mut next = 0;
        for group in 0..count {
            work.canonical(1)?;
            let start = vertices.len();
            while next < membership.len() && membership[next].0 == group {
                work.canonical(1)?;
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
    membership: Vec<(usize, usize)>,
    basis: Vec<usize>,
    columns: Vec<usize>,
}
impl PullLevel {
    fn new() -> Self {
        Self {
            cuts: Faces {
                vertices: Vec::new(),
                ranges: Vec::new(),
            },
            order: Vec::new(),
            membership: Vec::new(),
            basis: Vec::new(),
            columns: Vec::new(),
        }
    }
}
struct Pulling<'a> {
    width: usize,
    supports: &'a Faces,
    incident: &'a Faces,
    adjacent: &'a Faces,
    root: usize,
    prefix: Vec<usize>,
    output: Vec<usize>,
}
impl Pulling<'_> {
    fn pull<S: RealScalar>(
        &mut self,
        cloud: &Cloud<S>,
        face: &[usize],
        dimension: usize,
        levels: &mut [PullLevel],
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<()> {
        let storage = work.storage;
        work.canonical(1)?;
        if face.len() == dimension + 1 {
            if self.output.len() / self.width == work.options.limits.max_facets {
                return Err(SciOxideError::resource_limit(
                    scioxide_core::quantity!("geometry_facets"),
                    work.options.limits.max_facets.saturating_add(1),
                    work.options.limits.max_facets,
                ));
            }
            work.canonical(self.width)?;
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
        level.membership.clear();
        let neighbors = self.adjacent.at(self.root);
        // Accumulate intersections by visiting incidences, never by scanning
        // the current face once for every support. Only neighbors of the root
        // support can define its facets (or facets of its recursive faces).
        for &vertex in face {
            work.canonical(1)?;
            let star = self.incident.at(vertex);
            let (candidates, filter, by_vertex) = if star.len() <= neighbors.len() {
                (star, neighbors, false)
            } else {
                (neighbors, &[][..], true)
            };
            for &support in candidates {
                work.canonical(1)?;
                let present = if by_vertex {
                    work.canonical_contains(self.supports.at(support), vertex)?
                } else {
                    work.canonical_contains(filter, support)?
                };
                if present {
                    storage.reserve(&mut level.membership, 1)?;
                    level.membership.push((support, vertex));
                }
            }
        }
        work.canonical_sort(&mut level.membership)?;
        let mut begin = 0;
        while begin < level.membership.len() {
            let mut end = begin + 1;
            while end < level.membership.len()
                && level.membership[end].0 == level.membership[begin].0
            {
                work.canonical(1)?;
                end += 1;
            }
            work.canonical(1)?;
            // A boundary containing the apex never contributes to pulling.
            if end - begin >= dimension && level.membership[begin].1 != face[0] {
                let start = level.cuts.vertices.len();
                storage.reserve(&mut level.cuts.vertices, end - begin)?;
                for &(_, vertex) in &level.membership[begin..end] {
                    work.canonical(1)?;
                    level.cuts.vertices.push(vertex);
                }
                let cut = &level.cuts.vertices[start..];
                // Proper intersections are faces of dimension <= dimension-1.
                // Extreme vertices exclude three collinear vertices, so in
                // dimensions <= 3 cardinality already proves the needed rank.
                let maximal = if dimension <= 3 {
                    true
                } else {
                    level.basis.clear();
                    level.columns.clear();
                    storage.reserve(&mut level.basis, cloud.dimension + 1)?;
                    storage.reserve(&mut level.columns, cloud.dimension + 1)?;
                    for &vertex in cut {
                        work.canonical(1)?;
                        super::engine::extend_affine_basis(
                            cloud,
                            vertex,
                            &mut level.basis,
                            &mut level.columns,
                            work,
                        )?;
                        if level.basis.len() == dimension {
                            break;
                        }
                    }
                    level.basis.len() == dimension
                };
                if maximal {
                    storage.reserve(&mut level.cuts.ranges, 1)?;
                    level.cuts.ranges.push(start..level.cuts.vertices.len());
                } else {
                    level.cuts.vertices.truncate(start);
                }
            }
            begin = end;
        }
        storage.reserve(&mut level.order, level.cuts.ranges.len())?;
        work.canonical(level.cuts.ranges.len())?;
        level.order.extend(0..level.cuts.ranges.len());
        work.canonical_sort_by(&mut level.order, |&a, &b, work| {
            work.canonical_compare(level.cuts.at(a), level.cuts.at(b))
        })?;
        let mut end = 0;
        for i in 0..level.order.len() {
            let index = level.order[i];
            if end == 0
                || work
                    .canonical_compare(level.cuts.at(level.order[end - 1]), level.cuts.at(index))?
                    != Ordering::Equal
            {
                level.order[end] = index;
                end += 1;
            }
        }
        level.order.truncate(end);
        self.prefix.push(face[0]);
        for &index in &level.order {
            self.pull(cloud, level.cuts.at(index), dimension - 1, rest, work)?;
        }
        self.prefix.pop();
        Ok(())
    }
}

impl Work<'_, '_, '_, '_> {
    // Support and incidence work outside predicates, charged to the local
    // canonical limit and inherited node budgets.
    pub(super) fn charge_canonical(&mut self, count: usize) -> Result<()> {
        self.canonical(count)
    }
    fn canonical(&mut self, count: usize) -> Result<()> {
        self.check()?;
        let required = checked_add(self.canonical_work, count, CANONICAL_WORK)?;
        limit(
            CANONICAL_WORK,
            required,
            self.options.limits.max_canonical_work,
        )?;
        if let Some(context) = &mut self.context
            && !context.try_consume(BudgetKind::Node, count)?.is_granted()
        {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::ResourceLimit(BudgetKind::Node),
                scioxide_core::ConvergenceDiagnostics::new()
                    .with_counts(context.counts())
                    .with_budget(context.effective_budget_report(BudgetKind::Node)),
                "geometry canonical work allowance exhausted",
            ));
        }
        self.canonical_work = required;
        Ok(())
    }
    fn canonical_contains(&mut self, values: &[usize], key: usize) -> Result<bool> {
        let mut range = 0..values.len();
        while range.start < range.end {
            self.canonical(1)?;
            let mid = range.start + (range.end - range.start) / 2;
            match values[mid].cmp(&key) {
                Ordering::Less => range.start = mid + 1,
                Ordering::Greater => range.end = mid,
                Ordering::Equal => return Ok(true),
            }
        }
        Ok(false)
    }
    fn canonical_compare(&mut self, a: &[usize], b: &[usize]) -> Result<Ordering> {
        for (a, b) in a.iter().zip(b) {
            self.canonical(1)?;
            let order = a.cmp(b);
            if order != Ordering::Equal {
                return Ok(order);
            }
        }
        self.canonical(1)?;
        Ok(a.len().cmp(&b.len()))
    }
    fn canonical_sort<T: Ord>(&mut self, values: &mut [T]) -> Result<()> {
        self.canonical_sort_by(values, |a, b, work| {
            work.canonical(1)?;
            Ok(a.cmp(b))
        })
    }
    // Fallible in-place heapsort, as used by adaptive topology. Admission is
    // inside each comparison, so even a single large sort can stop promptly.
    fn canonical_sort_by<T>(
        &mut self,
        values: &mut [T],
        mut compare: impl FnMut(&T, &T, &mut Self) -> Result<Ordering>,
    ) -> Result<()> {
        for root in (0..values.len() / 2).rev() {
            self.canonical_sift(values, root, &mut compare)?;
        }
        for end in (1..values.len()).rev() {
            values.swap(0, end);
            self.canonical_sift(&mut values[..end], 0, &mut compare)?;
        }
        Ok(())
    }
    fn canonical_sift<T>(
        &mut self,
        values: &mut [T],
        mut root: usize,
        compare: &mut impl FnMut(&T, &T, &mut Self) -> Result<Ordering>,
    ) -> Result<()> {
        while root < values.len() / 2 {
            let mut child = root * 2 + 1;
            if child + 1 < values.len()
                && compare(&values[child], &values[child + 1], self)?.is_lt()
            {
                child += 1;
            }
            if !compare(&values[root], &values[child], self)?.is_lt() {
                break;
            }
            values.swap(root, child);
            root = child;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tessellation::tests::{test_requirements, test_workspace};
    use crate::tessellation::{GeometryOptions, storage::Storage};
    use alloc::vec;
    use scioxide_core::ExecutionContext;
    use scioxide_core::{ProblemToken, SolveBudgets, SolveIdentity, SolveState};

    #[test]
    fn pulling_itself_obeys_local_and_inherited_budgets() {
        let values: Vec<_> = (0..8)
            .flat_map(|i| [f64::from(i >> 2), f64::from((i >> 1) & 1), f64::from(i & 1)])
            .collect();
        crate::tessellation::tests::with_points(&values, 3, |points| {
            let storage = Storage::new(usize::MAX, 0)?;
            let o = GeometryOptions::new();
            let cloud = Cloud::new(points, o, &storage)?;
            let mut workspace = test_workspace(test_requirements(3, o)?)?;
            let mut work = Work::new(&mut workspace, 3, o, None,
                scioxide_core::argument!("points"), &storage)?;
            let supports = Faces {
                vertices: vec![0, 1, 2, 3, 0, 1, 4, 5, 0, 2, 4, 6, 1, 3, 5, 7, 2, 3, 6, 7, 4, 5, 6, 7],
                ranges: (0..6).map(|i| 4 * i..4 * i + 4).collect(),
            };
            let mut pairs: Vec<_> = (0..6).flat_map(|g| {
                supports.at(g).iter().map(move |&v| (v, g))
            }).collect();
            pairs.sort_unstable();
            let incident = Faces::new(8, &pairs, &mut work)?;
            let adjacent = Faces::new(1, &[(0, 1), (0, 2), (0, 3), (0, 4)], &mut work)?;
            let pull = |work: &mut Work<'_, '_, '_, '_>| -> Result<Vec<usize>> {
                let mut pulling = Pulling { width: 3, supports: &supports,
                    incident: &incident, adjacent: &adjacent, root: 0,
                    prefix: storage.capacity(3, STORAGE)?, output: Vec::new() };
                let mut levels: Vec<_> = (0..3).map(|_| PullLevel::new()).collect();
                pulling.pull(&cloud, supports.at(0), 2, &mut levels, work)?;
                Ok(pulling.output)
            };
            work.canonical_work = 0;
            assert_eq!(pull(&mut work)?, [0,1,3,0,2,3]);
            let count = work.canonical_work;
            work.canonical_work = 0;
            work.options.limits.max_canonical_work = count - 1;
            assert!(matches!(pull(&mut work), Err(error)
                if matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource, .. }
                    if resource == scioxide_core::quantity!("geometry_canonical_work"))));
            work.options.limits.max_canonical_work = count;
            work.canonical_work = 0;
            assert_eq!(pull(&mut work)?, [0,1,3,0,2,3]);

            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Node,
                scioxide_core::Budget::try_from(count - 1)?);
            let mut cx = state.begin_solve(ExecutionContext::sequential(),
                SolveIdentity::new(ProblemToken::new(0),
                    scioxide_core::algorithm_id!("spatial", "canonical_test", 1), 0, 0),
                &mut budgets)?;
            let accepted = cx.accepted_state();
            let mut work = Work::new(&mut workspace, 3, o, Some(&mut cx),
                scioxide_core::argument!("points"), &storage)?;
            assert!(matches!(pull(&mut work), Err(error)
                if matches!(error.view(), scioxide_core::SciOxideErrorView::ConvergenceFailure {
                    reason: scioxide_core::ConvergenceReason::ResourceLimit(BudgetKind::Node), ..
                })));
            assert_eq!(cx.accepted_state(), accepted);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn canonical_sort_polls_cancellation_between_comparisons() {
        let storage = Storage::new(usize::MAX, 0).unwrap();
        let o = GeometryOptions::new();
        let mut workspace = test_workspace(test_requirements(3, o).unwrap()).unwrap();
        let token = scioxide_core::CancellationToken::new();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut cx = state
            .begin_solve(
                ExecutionContext::sequential().with_cancellation(&token),
                SolveIdentity::new(
                    ProblemToken::new(0),
                    scioxide_core::algorithm_id!("spatial", "canonical_test", 1),
                    0,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        let mut work = Work::new(
            &mut workspace,
            3,
            o,
            Some(&mut cx),
            scioxide_core::argument!("points"),
            &storage,
        )
        .unwrap();
        let mut comparisons = 0;
        let mut values: alloc::vec::Vec<_> = (0..4096).rev().collect();
        let result = work.canonical_sort_by(&mut values, |a, b, work| {
            work.canonical(1)?;
            comparisons += 1;
            token.cancel();
            Ok(a.cmp(b))
        });
        // The poll at the 256th admitted unit stops the sort.
        assert_eq!(comparisons + 1, super::super::work::POLL_UNITS);
        assert!(matches!(result, Err(error)
            if error.kind() == scioxide_core::SciOxideErrorKind::ExecutionStopped));
    }
}
