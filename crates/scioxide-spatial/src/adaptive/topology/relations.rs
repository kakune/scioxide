use super::*;
mod edges;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Match {
    source: usize,
    destination: usize,
    kind: i8,
    orientation: i8,
}

// Identity/ancestry pairs are emitted sparsely, then restored to the public
// source-major, destination-minor order. Unmatched destinations follow sources.
fn finish(
    sources: usize,
    destinations: usize,
    mut matches: Vec<Match>,
    work: &mut Work<'_, '_, '_>,
) -> Result<Vec<EntityRelation>> {
    work.sort(&mut matches)?;
    let mut seen = Vec::new();
    work.reserve(&mut seen, destinations)?;
    seen.resize(destinations, false);
    let mut result = Vec::new();
    let mut pairs = matches.into_iter().peekable();
    for source in 0..sources {
        work.tick()?;
        if pairs.peek().is_none_or(|pair| pair.source != source) {
            work.push(&mut result, EntityRelation::Removed { source })?;
        }
        while pairs.peek().is_some_and(|pair| pair.source == source) {
            let Some(pair) = pairs.next() else { break };
            work.tick()?;
            let Match {
                source,
                destination,
                kind,
                orientation,
            } = pair;
            seen[destination] = true;
            let relation = match kind {
                0 => EntityRelation::Unchanged {
                    source,
                    destination,
                },
                1 => EntityRelation::Refined {
                    source,
                    destination,
                    orientation,
                },
                _ => EntityRelation::Coarsened {
                    source,
                    destination,
                    orientation,
                },
            };
            work.push(&mut result, relation)?;
        }
    }
    for (destination, seen) in seen.into_iter().enumerate() {
        work.tick()?;
        if !seen {
            work.push(&mut result, EntityRelation::Created { destination })?;
        }
    }
    Ok(result)
}

fn hierarchy(
    sources: &[usize],
    destinations: &[usize],
    retained: usize,
    parent: impl Fn(usize) -> Option<(usize, i8)>,
    work: &mut Work<'_, '_, '_>,
) -> Result<Vec<EntityRelation>> {
    let mut source_index = Vec::new();
    let mut destination_index = Vec::new();
    work.reserve(&mut source_index, retained)?;
    work.reserve(&mut destination_index, retained)?;
    source_index.resize(retained, usize::MAX);
    destination_index.resize(retained, usize::MAX);
    for (index, &id) in sources.iter().enumerate() {
        work.tick()?;
        source_index[id] = index;
    }
    for (index, &id) in destinations.iter().enumerate() {
        work.tick()?;
        destination_index[id] = index;
    }
    let mut matches = Vec::new();
    for (destination, &id) in destinations.iter().enumerate() {
        let mut ancestor = id;
        let mut orientation = 1;
        loop {
            work.tick()?;
            let source = source_index[ancestor];
            if source != usize::MAX {
                work.push(
                    &mut matches,
                    Match {
                        source,
                        destination,
                        kind: i8::from(ancestor != id),
                        orientation,
                    },
                )?;
            }
            let Some((next, sign)) = parent(ancestor) else {
                break;
            };
            ancestor = next;
            orientation *= sign;
        }
    }
    for (source, &id) in sources.iter().enumerate() {
        let mut ancestor = id;
        let mut orientation = 1;
        while let Some((next, sign)) = parent(ancestor) {
            work.tick()?;
            ancestor = next;
            orientation *= sign;
            let destination = destination_index[ancestor];
            if destination != usize::MAX {
                work.push(
                    &mut matches,
                    Match {
                        source,
                        destination,
                        kind: -1,
                        orientation,
                    },
                )?;
            }
        }
    }
    finish(sources.len(), destinations.len(), matches, work)
}

impl TopologyMap {
    pub(in crate::adaptive) fn build<S: RealScalar, const D: usize>(
        source: &AdaptiveMesh<S, D>,
        destination: &AdaptiveMesh<S, D>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Self> {
        if Arc::ptr_eq(&source.storage, &destination.storage) {
            return Ok(Self {
                source: source.token.clone(),
                destination: destination.token.clone(),
                cells: unchanged(source.cell_count(), work)?,
                vertices: unchanged(source.vertex_count(), work)?,
                faces: unchanged(source.face_count(), work)?,
                edges: unchanged(source.edge_count(), work)?,
            });
        }
        #[cfg(test)]
        if destination.storage.exhaustive {
            return Self::build_exhaustive(source, destination, work);
        }
        let a = &source.storage;
        let b = &destination.storage;
        // The destination retains the complete source hierarchy, including
        // inactive descendants after coarsening.
        let cells = hierarchy(
            &a.active,
            &b.active,
            b.nodes.len(),
            |id| b.nodes[id].parent.map(|parent| (parent, 1)),
            work,
        )?;
        let faces = hierarchy(
            &a.active_faces,
            &b.active_faces,
            b.faces.len(),
            |id| {
                b.faces[id]
                    .parent
                    .map(|parent| (parent, b.faces[id].parent_sign))
            },
            work,
        )?;
        let mut matches = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < a.active_vertices.len() && j < b.active_vertices.len() {
            work.tick()?;
            match a.active_vertices[i].cmp(&b.active_vertices[j]) {
                core::cmp::Ordering::Less => i += 1,
                core::cmp::Ordering::Greater => j += 1,
                core::cmp::Ordering::Equal => {
                    work.push(
                        &mut matches,
                        Match {
                            source: i,
                            destination: j,
                            kind: 0,
                            orientation: 1,
                        },
                    )?;
                    i += 1;
                    j += 1;
                }
            }
        }
        let vertices = finish(
            a.active_vertices.len(),
            b.active_vertices.len(),
            matches,
            work,
        )?;
        let edges = edges::relations(a, b, work)?;
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

// Shared storage already has identical compact indices; no ancestry search,
// matching scratch, or sorting is needed to describe the identity.
fn unchanged(count: usize, work: &mut Work<'_, '_, '_>) -> Result<Vec<EntityRelation>> {
    let mut result = Vec::new();
    work.reserve(&mut result, count)?;
    for index in 0..count {
        work.tick()?;
        result.push(EntityRelation::Unchanged {
            source: index,
            destination: index,
        });
    }
    Ok(result)
}
