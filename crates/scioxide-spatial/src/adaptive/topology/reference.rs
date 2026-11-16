//! Small, exhaustive test oracle. No production entry point uses these scans.
#![allow(clippy::unwrap_used)]
use super::*;
use alloc::{format, vec};

fn limits() -> AdaptationLimits {
    AdaptationLimits::new()
        .with_max_depth(4)
        .with_max_cells(10_000)
        .with_max_entities(100_000)
        .with_max_work(usize::MAX)
        .with_max_storage_bytes(256 * 1024 * 1024)
}

fn work() -> Work<'static, 'static, 'static> {
    Work::new(limits(), None, scioxide_core::argument!("mesh")).unwrap()
}

fn assert_mesh<S: RealScalar, const D: usize>(a: &AdaptiveMesh<S, D>, b: &AdaptiveMesh<S, D>) {
    let a = &a.storage;
    let b = &b.storage;
    assert_eq!(a.vertices, b.vertices);
    assert_eq!(a.nodes, b.nodes);
    assert_eq!(a.faces, b.faces);
    assert_eq!(a.active, b.active);
    assert_eq!(a.active_vertices, b.active_vertices);
    assert_eq!(a.active_faces, b.active_faces);
    assert_eq!(a.edges, b.edges);
    assert_eq!(a.interfaces, b.interfaces);
    assert_eq!(a.hanging, b.hanging);
}

// Re-import root geometry with exhaustive interning and interface discovery.
fn reference_root<S: RealScalar, const D: usize>(root: &AdaptiveMesh<S, D>) -> AdaptiveMesh<S, D> {
    let mut storage = Storage::empty(root.storage.simplex);
    storage.exhaustive = true;
    storage.root_extent = root.storage.root_extent;
    storage.periodic = root.storage.periodic;
    storage.reflected = root.storage.reflected;
    let mut work = work();
    for vertex in &root.storage.vertices {
        storage.add_root(vertex.point, &mut work).unwrap();
    }
    for node in &root.storage.nodes {
        storage
            .add_node(Node::root(node.vertices), &mut work)
            .unwrap();
    }
    storage.rebuild_exhaustive(&mut work).unwrap();
    storage.release_indexes();
    let mesh = AdaptiveMesh {
        storage: Arc::new(storage),
        token: root.token.clone(),
        limits: root.limits,
    };
    assert_mesh(root, &mesh);
    mesh
}

fn compare_step<S: RealScalar, const D: usize>(
    a: &AdaptiveMesh<S, D>,
    b: &AdaptiveMesh<S, D>,
    refine: &[usize],
    coarsen: &[usize],
) -> Option<(AdaptiveMesh<S, D>, AdaptiveMesh<S, D>)> {
    let actual = a.adapt_scoped(
        AdaptationRequest {
            source: a.mesh_token(),
            refine,
            coarsen,
        },
        scioxide_core::ExecutionContext::sequential(),
    );
    let expected = b.adapt_scoped(
        AdaptationRequest {
            source: b.mesh_token(),
            refine,
            coarsen,
        },
        scioxide_core::ExecutionContext::sequential(),
    );
    match (actual, expected) {
        (Ok(actual), Ok(expected)) => {
            assert_mesh(&actual.mesh, &expected.mesh);
            assert_eq!(actual.induced_refinements, expected.induced_refinements);
            assert_eq!(actual.map.cells(), expected.map.cells());
            assert_eq!(actual.map.vertices(), expected.map.vertices());
            assert_eq!(actual.map.faces(), expected.map.faces());
            assert_eq!(actual.map.edges(), expected.map.edges());
            actual
                .map
                .validate(a.mesh_token(), actual.mesh.mesh_token())
                .unwrap();
            expected
                .map
                .validate(b.mesh_token(), expected.mesh.mesh_token())
                .unwrap();
            Some((actual.mesh, expected.mesh))
        }
        (Err(actual), Err(expected)) => {
            assert_eq!(
                format!("{:?}", actual.view()),
                format!("{:?}", expected.view())
            );
            assert_mesh(a, b);
            None
        }
        (actual, expected) => {
            assert_eq!(
                format!("{actual:?}"),
                format!("{expected:?}"),
                "reference outcome differs"
            );
            None
        }
    }
}

fn sequences<S: RealScalar, const D: usize>(root: AdaptiveMesh<S, D>) {
    let reference = reference_root(&root);
    let roots = root.active_cells();
    let (a, b) = compare_step(&root, &reference, roots, &[]).unwrap();
    let (a, b) = compare_step(&a, &b, &[], roots).unwrap();
    // Children and entity IDs must be reused after a complete round trip.
    compare_step(&a, &b, roots, &[]).unwrap();
    for seed in [17u64, 103, 509] {
        let (mut a, mut b) = (root.clone(), reference.clone());
        let mut state = seed;
        let mut draw = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..16 {
            let refine: Vec<_> = a
                .active_cells()
                .iter()
                .copied()
                .filter(|&id| a.cell_count() < 32 && a.level(id).unwrap() < 3 && draw() % 5 == 0)
                .take(if D == 3 { 1 } else { 2 })
                .collect();
            let coarsen: Vec<_> = a
                .storage
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(id, node)| {
                    (!node.active
                        && node.child_count > 0
                        && node.children[..usize::from(node.child_count)]
                            .iter()
                            .all(|child| a.storage.nodes[*child].active && !refine.contains(child))
                        && draw() % 3 == 0)
                        .then_some(id)
                })
                .collect();
            if let Some((next_a, next_b)) = compare_step(&a, &b, &refine, &coarsen) {
                a = next_a;
                b = next_b;
            }
        }
    }
}

fn tensor<S: RealScalar, const D: usize>(periodic: bool, sites: usize) -> AdaptiveMesh<S, D> {
    let grid = crate::StructuredGrid::new(
        core::array::from_fn(|_| crate::CoordinateAxis::uniform(S::ONE, S::ZERO, sites).unwrap()),
        [if periodic {
            AxisTopology::Periodic
        } else {
            AxisTopology::Bounded
        }; D],
    )
    .unwrap();
    AdaptiveMesh::from_structured(&CellMesh::new(grid).unwrap(), limits()).unwrap()
}

fn all_families<S: RealScalar>() {
    for periodic in [false, true] {
        sequences(tensor::<S, 1>(periodic, 3));
        sequences(tensor::<S, 2>(periodic, 3));
        sequences(tensor::<S, 3>(periodic, 2));
    }
    let z = S::ZERO;
    let o = S::ONE;
    let triangle = crate::TriangleMesh::new(
        [[z, z], [o, z], [o, o], [z, o]]
            .map(|p| Point::new(p).unwrap())
            .to_vec(),
        vec![[0, 1, 2], [0, 2, 3]],
    )
    .unwrap();
    sequences(AdaptiveMesh::from_simplex(&triangle, limits()).unwrap());
    let tetrahedron = crate::TetrahedronMesh::new(
        [[z, z, z], [o, z, z], [z, o, z], [z, z, o], [z, z, -o]]
            .map(|p| Point::new(p).unwrap())
            .to_vec(),
        vec![[0, 1, 2, 3], [0, 2, 1, 4]],
    )
    .unwrap();
    sequences(AdaptiveMesh::from_simplex(&tetrahedron, limits()).unwrap());
}

#[test]
fn randomized_sequences_match_exhaustive_topology_f32() {
    all_families::<f32>();
}

#[test]
fn randomized_sequences_match_exhaustive_topology_f64() {
    all_families::<f64>();
}

#[test]
fn cascading_closure_keeps_dynamic_smallest_id_order() {
    let root = tensor::<f64, 2>(false, 3);
    let reference = reference_root(&root);
    let (a, b) = compare_step(&root, &reference, &[0], &[]).unwrap();
    let (a, b) = compare_step(&a, &b, a.children(0).unwrap(), &[]).unwrap();
    let grandchildren: Vec<_> = a
        .children(0)
        .unwrap()
        .iter()
        .flat_map(|&child| a.children(child).unwrap().iter().copied())
        .collect();
    let result = a
        .adapt_scoped(
            AdaptationRequest {
                source: a.mesh_token(),
                refine: &grandchildren,
                coarsen: &[],
            },
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(result.induced_refinements.contains(&3));
    assert!(
        result
            .induced_refinements
            .windows(2)
            .any(|pair| pair[0] > pair[1])
    );
    compare_step(&a, &b, &grandchildren, &[]).unwrap();
}

impl<S: RealScalar, const D: usize> Storage<S, D> {
    pub(in crate::adaptive) fn balance_exhaustive(
        &mut self,
        coarsen: &[usize],
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Vec<usize>> {
        let mut induced = Vec::new();
        loop {
            self.rebuild_exhaustive(work)?;
            let mut next = None;
            for interface in &self.interfaces {
                if let Some(right_side) = interface.right {
                    let right = right_side.cell;
                    let a = self.active[interface.left_cell];
                    let b = self.active[right];
                    let la = self.nodes[a].level;
                    let lb = self.nodes[b].level;
                    if la.abs_diff(lb) > 1 {
                        let coarse = if la < lb { a } else { b };
                        next = Some(next.map_or(coarse, |old: usize| old.min(coarse)));
                    }
                }
            }
            let Some(id) = next else { break };
            if coarsen.binary_search(&id).is_ok() {
                return Err(invalid(
                    scioxide_core::argument!("request.coarsen"),
                    "coarsening violates final 2:1 balance",
                ));
            }
            work.push(&mut induced, id)?;
            self.split(id, work)?;
        }
        Ok(induced)
    }
}

impl TopologyMap {
    pub(super) fn build_exhaustive<S: RealScalar, const D: usize>(
        source: &AdaptiveMesh<S, D>,
        destination: &AdaptiveMesh<S, D>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Self> {
        // Relations come from exact identities, not geometry. Cells and faces
        // use stable hierarchy ids and parent chains. Vertices use stable
        // ids. Edges have no hierarchy links, so containment is decided in
        // exact provenance space by `on_edge`. Kind 1 means the destination
        // descends from the source (refined), and -1 the reverse.
        let a = &source.storage;
        let b = &destination.storage;
        let cells = relations(a.active.len(), b.active.len(), work, |i, j| {
            let x = a.active[i];
            let y = b.active[j];
            if x == y {
                Some(0)
            } else if ancestor_node(&b.nodes, x, y) {
                Some(1)
            } else if ancestor_node(&a.nodes, y, x) {
                Some(-1)
            } else {
                None
            }
        })?;
        let vertices = relations(
            a.active_vertices.len(),
            b.active_vertices.len(),
            work,
            |i, j| (a.active_vertices[i] == b.active_vertices[j]).then_some(0),
        )?;
        let mut faces = relations(a.active_faces.len(), b.active_faces.len(), work, |i, j| {
            let x = a.active_faces[i];
            let y = b.active_faces[j];
            if x == y {
                Some(0)
            } else if ancestor_face(&b.faces, x, y) {
                Some(1)
            } else if ancestor_face(&a.faces, y, x) {
                Some(-1)
            } else {
                None
            }
        })?;
        for relation in &mut faces {
            match relation {
                EntityRelation::Refined {
                    source,
                    destination,
                    orientation,
                } => {
                    *orientation = face_sign(
                        &b.faces,
                        a.active_faces[*source],
                        b.active_faces[*destination],
                    )
                }
                EntityRelation::Coarsened {
                    source,
                    destination,
                    orientation,
                } => {
                    *orientation = face_sign(
                        &a.faces,
                        b.active_faces[*destination],
                        a.active_faces[*source],
                    )
                }
                _ => {}
            }
        }
        let mut edges = relations(a.edges.len(), b.edges.len(), work, |i, j| {
            let x = a.edges[i];
            let y = b.edges[j];
            if x == y {
                Some(0)
            } else if y.iter().all(|&v| {
                on_edge(
                    b.vertices[v].key,
                    a.vertices[x[0]].key,
                    a.vertices[x[1]].key,
                )
            }) {
                Some(1)
            } else if x.iter().all(|&v| {
                on_edge(
                    a.vertices[v].key,
                    b.vertices[y[0]].key,
                    b.vertices[y[1]].key,
                )
            }) {
                Some(-1)
            } else {
                None
            }
        })?;
        for relation in &mut edges {
            if let EntityRelation::Refined {
                source,
                destination,
                orientation,
            }
            | EntityRelation::Coarsened {
                source,
                destination,
                orientation,
            } = relation
            {
                *orientation = edge_sign(
                    a.edges[*source].map(|v| a.vertices[v].key),
                    b.edges[*destination].map(|v| b.vertices[v].key),
                );
            }
        }
        Ok(Self {
            source: source.token.clone(),
            destination: destination.token.clone(),
            cells,
            vertices,
            faces,
            edges,
        })
    }
}
// All-pairs classification, charged per pair against the caller's work
// limit. A source with no related destination is Removed. A destination
// that no source touched (tracked in `seen`) is Created. Orientations
// start at +1 and are fixed up by the caller for faces and edges.
fn relations<F: Fn(usize, usize) -> Option<i8>>(
    sources: usize,
    destinations: usize,
    work: &mut Work<'_, '_, '_>,
    relation: F,
) -> Result<Vec<EntityRelation>> {
    let mut result = Vec::new();
    let mut seen = Vec::new();
    work.reserve(&mut seen, destinations)?;
    seen.resize(destinations, false);
    for i in 0..sources {
        let mut found = false;
        for (j, seen) in seen.iter_mut().enumerate() {
            work.tick()?;
            if let Some(kind) = relation(i, j) {
                found = true;
                *seen = true;
                work.push(
                    &mut result,
                    match kind {
                        0 => EntityRelation::Unchanged {
                            source: i,
                            destination: j,
                        },
                        1 => EntityRelation::Refined {
                            source: i,
                            destination: j,
                            orientation: 1,
                        },
                        _ => EntityRelation::Coarsened {
                            source: i,
                            destination: j,
                            orientation: 1,
                        },
                    },
                )?;
            }
        }
        if !found {
            work.push(&mut result, EntityRelation::Removed { source: i })?;
        }
    }
    for (j, seen) in seen.into_iter().enumerate() {
        if !seen {
            work.push(&mut result, EntityRelation::Created { destination: j })?;
        }
    }
    Ok(result)
}
fn ancestor_node(nodes: &[Node], ancestor: usize, mut child: usize) -> bool {
    while let Some(node) = nodes.get(child) {
        let Some(parent) = node.parent else {
            return false;
        };
        if parent == ancestor {
            return true;
        }
        child = parent;
    }
    false
}
fn ancestor_face(faces: &[Face], ancestor: usize, mut child: usize) -> bool {
    while let Some(face) = faces.get(child) {
        let Some(parent) = face.parent else {
            return false;
        };
        if parent == ancestor {
            return true;
        }
        child = parent;
    }
    false
}
impl<S: RealScalar, const D: usize> Storage<S, D> {
    // Different face records denote one physical interface only as periodic
    // seam partners, which agree after seam normalization. Elsewhere,
    // deduplication in `add_node` already made shared faces a single record.
    fn same_face(&self, a: usize, b: usize) -> bool {
        if a == b {
            return true;
        }
        let x = self.faces[a];
        let y = self.faces[b];
        x.len == y.len
            && x.axis == y.axis
            && x.axis.is_some_and(|axis| self.periodic[axis])
            && self.face_key(a).0 == self.face_key(b).0
    }
    fn contains_face(&self, ancestor: usize, mut child: usize) -> bool {
        loop {
            if self.same_face(ancestor, child) {
                return true;
            }
            let Some(parent) = self.faces[child].parent else {
                return false;
            };
            child = parent;
        }
    }
    pub(super) fn rebuild_exhaustive(&mut self, work: &mut Work<'_, '_, '_>) -> Result<()> {
        self.active.clear();
        self.active_vertices.clear();
        self.active_faces.clear();
        self.edges.clear();
        self.interfaces.clear();
        self.hanging.clear();
        for id in 0..self.nodes.len() {
            if self.nodes[id].parent.is_none() {
                self.collect_active(id, work)?;
            }
        }
        for &id in &self.active {
            let node = self.nodes[id];
            for &v in &node.vertices[..self.vertex_arity()] {
                work.push(&mut self.active_vertices, v)?;
            }
            for &f in &node.faces[..self.face_arity()] {
                work.push(&mut self.active_faces, f)?;
            }
            for i in 0..self.vertex_arity() {
                for j in i + 1..self.vertex_arity() {
                    // Every simplex vertex pair is an edge. Tensor corners in
                    // bit order form an edge only when their indices differ
                    // in exactly one bit; other pairs are diagonals.
                    if self.simplex || (i ^ j).is_power_of_two() {
                        let mut edge = [node.vertices[i], node.vertices[j]];
                        edge.sort_unstable();
                        work.push(&mut self.edges, edge)?;
                    }
                }
            }
        }
        self.active_vertices.sort_unstable();
        self.active_vertices.dedup();
        self.active_faces.sort_unstable();
        self.active_faces.dedup();
        self.edges.sort_unstable();
        self.edges.dedup();
        if self.edges.len() > work.limits.max_entities {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("adaptive_edges"),
                self.edges.len(),
                work.limits.max_entities,
            ));
        }
        // Interfaces come from an exhaustive (cell, face) x (cell, face)
        // search, charged per pair. Two faces touch when one is an ancestor
        // of (or seam partner to) the other.
        //
        // Coverage check: a whole face is 2^(20(D-1)) units. A neighbor face
        // k levels finer covers 2^(-k(D-1)) of it, because both tensor and
        // red refinement split a face into 2^(D-1) children per level; depth
        // <= 20 keeps this exact in u64. A coarser or equal neighbor counts
        // as full coverage (k saturates at 0). The pieces must tile the face
        // exactly once, which catches gaps and overlaps in the hierarchy.
        for left in 0..self.active.len() {
            for local in 0..self.face_arity() {
                let a = self.nodes[self.active[left]].faces[local];
                let mut found = false;
                let mut coverage = 0u64;
                let full_coverage = 1u64 << (20 * (D - 1));
                for right in 0..self.active.len() {
                    for other in 0..self.face_arity() {
                        work.tick()?;
                        if left == right && local == other {
                            continue;
                        }
                        let b = self.nodes[self.active[right]].faces[other];
                        if self.contains_face(a, b) || self.contains_face(b, a) {
                            found = true;
                            let difference = self.nodes[self.active[right]]
                                .level
                                .saturating_sub(self.nodes[self.active[left]].level);
                            coverage = coverage
                                .checked_add(1u64 << ((20 - usize::from(difference)) * (D - 1)))
                                .ok_or_else(|| work.invalid("interface coverage overflow"))?;
                            // Coverage is counted from both sides, but each
                            // interface is emitted once, from the smaller pair.
                            if (left, local) >= (right, other) {
                                continue;
                            }
                            // The finer face is the physical piece. Outward
                            // normals of the two cells are opposite, so the
                            // left sign in the piece's orientation is either
                            // the left cell's own sign or the negated right one.
                            let piece = if self.contains_face(a, b) { b } else { a };
                            let lf = self.active_faces.partition_point(|&v| v < a);
                            let rf = self.active_faces.partition_point(|&v| v < b);
                            let periodic = self.face_key(a).1 != self.face_key(b).1;
                            let left_sign = if piece == a {
                                self.nodes[self.active[left]].signs[local]
                            } else {
                                -self.nodes[self.active[right]].signs[other]
                            };
                            work.push(
                                &mut self.interfaces,
                                AdaptiveInterface {
                                    left_cell: left,
                                    right: Some(AdaptiveInterfaceSide {
                                        cell: right,
                                        face: rf,
                                    }),
                                    left_face: lf,
                                    piece_face: self.active_faces.partition_point(|&v| v < piece),
                                    left_sign,
                                    periodic,
                                },
                            )?;
                            // A strictly finer piece hangs on the coarse face.
                            // Its vertices and edges that are not coarse-face
                            // corners (compared seam-normalized) are hanging.
                            // Duplicates from multiple pieces are removed by
                            // the final sort and dedup.
                            if !self.same_face(a, b) {
                                let coarse = if piece == b { a } else { b };
                                let coarse_index =
                                    self.active_faces.partition_point(|&v| v < coarse);
                                let fine = self.faces[piece];
                                let coarse_face = self.faces[coarse];
                                work.push(
                                    &mut self.hanging,
                                    HangingEntity {
                                        kind: HangingEntityKind::Face,
                                        entity: self.active_faces.partition_point(|&v| v < piece),
                                        coarse_face: coarse_index,
                                    },
                                )?;
                                for &v in &fine.vertices[..usize::from(fine.len)] {
                                    if !coarse_face.vertices[..usize::from(coarse_face.len)]
                                        .iter()
                                        .any(|&c| {
                                            self.face_vertex_key(coarse, c)
                                                == self.face_vertex_key(piece, v)
                                        })
                                    {
                                        work.push(
                                            &mut self.hanging,
                                            HangingEntity {
                                                kind: HangingEntityKind::Vertex,
                                                entity: self
                                                    .active_vertices
                                                    .partition_point(|&x| x < v),
                                                coarse_face: coarse_index,
                                            },
                                        )?;
                                    }
                                }
                                for (edge_index, edge) in self.edges.iter().enumerate() {
                                    if edge
                                        .iter()
                                        .all(|v| fine.vertices[..usize::from(fine.len)].contains(v))
                                        && !edge.iter().all(|v| {
                                            coarse_face.vertices[..usize::from(coarse_face.len)]
                                                .iter()
                                                .any(|&c| {
                                                    self.face_vertex_key(coarse, c)
                                                        == self.face_vertex_key(piece, *v)
                                                })
                                        })
                                    {
                                        work.push(
                                            &mut self.hanging,
                                            HangingEntity {
                                                kind: HangingEntityKind::Edge,
                                                entity: edge_index,
                                                coarse_face: coarse_index,
                                            },
                                        )?;
                                    }
                                }
                            }
                        }
                    }
                }
                if found && coverage != full_coverage {
                    return Err(work.invalid(
                        "interior interface pieces must cover each whole face exactly once",
                    ));
                }
                if !found {
                    let face = self.active_faces.partition_point(|&v| v < a);
                    work.push(
                        &mut self.interfaces,
                        AdaptiveInterface {
                            left_cell: left,
                            right: None,
                            left_face: face,
                            piece_face: face,
                            left_sign: self.nodes[self.active[left]].signs[local],
                            periodic: false,
                        },
                    )?;
                }
            }
        }
        self.hanging.sort_unstable();
        self.hanging.dedup();
        Ok(())
    }
}
// Orientation of a descendant face relative to an ancestor: the product of
// the per-level `parent_sign` transports along the ancestry chain.
fn face_sign(faces: &[Face], ancestor: usize, mut child: usize) -> i8 {
    let mut sign = 1;
    while child != ancestor {
        sign *= faces[child].parent_sign;
        let Some(parent) = faces[child].parent else {
            break;
        };
        child = parent;
    }
    sign
}
// Related edges are collinear in provenance space, so comparing the
// direction sign on the first coordinate where `a` varies is enough.
fn edge_sign(a: [Key; 2], b: [Key; 2]) -> i8 {
    let exponent = a.iter().chain(&b).map(|k| k.exponent).max().unwrap_or(0);
    let weight = |key: Key, id: usize| -> i128 {
        key.terms[..usize::from(key.len)]
            .iter()
            .find(|t| t.0 == id)
            .map_or(0, |t| i128::from(t.1) << u32::from(exponent - key.exponent))
    };
    for &(id, _) in a[0].terms[..usize::from(a[0].len)]
        .iter()
        .chain(&a[1].terms[..usize::from(a[1].len)])
    {
        let delta = weight(a[1], id) - weight(a[0], id);
        if delta != 0 {
            return if delta.signum() == (weight(b[1], id) - weight(b[0], id)).signum() {
                1
            } else {
                -1
            };
        }
    }
    1
}
