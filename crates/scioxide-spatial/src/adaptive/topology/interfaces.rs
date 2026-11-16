use super::*;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Pair {
    left: usize,
    right: Option<usize>,
    piece: usize,
}

impl<S: RealScalar, const D: usize> Storage<S, D> {
    // Equivalence is strictly the existing whole-face identity plus seam
    // normalization, including its axis. Physical endpoint vertices stay distinct.
    fn physical_faces(&self, work: &mut Work<'_, '_, '_>) -> Result<Vec<usize>> {
        let mut keys = Vec::new();
        let mut classes = Vec::new();
        work.reserve(&mut classes, self.faces.len())?;
        for (id, face) in self.faces.iter().enumerate() {
            work.tick()?;
            classes.push(id);
            let Some(axis) = face.axis.filter(|&axis| self.periodic[axis]) else {
                continue;
            };
            let first_plane = self.face_on_root_plane(id, axis, 0);
            // Away from endpoint planes, exact interning already identifies
            // shared faces. Only seam candidates need the larger key.
            if first_plane || self.face_seam(id).is_some() {
                work.push(&mut keys, ((axis, face.len, self.face_key(id).0), id))?;
            }
        }
        work.sort(&mut keys)?;
        let mut class = 0;
        for i in 0..keys.len() {
            work.tick()?;
            if i == 0 || keys[i - 1].0 != keys[i].0 {
                class = keys[i].1;
            }
            classes[keys[i].1] = class;
        }
        Ok(classes)
    }

    pub(super) fn rebuild_interfaces(&mut self, work: &mut Work<'_, '_, '_>) -> Result<()> {
        let classes = self.physical_faces(work)?;
        let arity = self.face_arity();
        let count = checked_mul(
            self.active.len(),
            arity,
            scioxide_core::quantity!("adaptive_incidences"),
        )?;
        let mut incidences = Vec::new();
        let mut coverage = Vec::new();
        work.reserve(&mut incidences, count)?;
        work.reserve(&mut coverage, count)?;
        coverage.resize(count, 0u64);
        for (cell, &id) in self.active.iter().enumerate() {
            for (local, &face) in self.nodes[id].faces[..arity].iter().enumerate() {
                work.tick()?;
                incidences.push((classes[face], cell * arity + local));
            }
        }
        work.sort(&mut incidences)?;
        let face_at =
            |incidence: usize| self.nodes[self.active[incidence / arity]].faces[incidence % arity];
        let level_at = |incidence: usize| self.nodes[self.active[incidence / arity]].level;
        let mut pairs = Vec::new();
        for incidence in 0..count {
            let face = face_at(incidence);
            let mut ancestor = Some(face);
            while let Some(id) = ancestor {
                work.tick()?;
                let start = work
                    .search(&incidences, &(classes[id], 0))?
                    .unwrap_or_else(|at| at);
                for &(class, other) in &incidences[start..] {
                    work.tick()?;
                    if class != classes[id] {
                        break;
                    }
                    // Equal faces are visited from both sides; ancestor
                    // matches are visited only from their finer side.
                    if incidence == other || (id == face && incidence > other) {
                        continue;
                    }
                    let left = incidence.min(other);
                    let right = incidence.max(other);
                    // On equal (including seam) faces the original search
                    // chooses the right face as its oriented physical piece.
                    let piece = if id == face { face_at(right) } else { face };
                    for (a, b) in [(left, right), (right, left)] {
                        let difference = level_at(b).saturating_sub(level_at(a));
                        let units = 1u64 << ((20 - usize::from(difference)) * (D - 1));
                        coverage[a] = coverage[a]
                            .checked_add(units)
                            .ok_or_else(|| work.invalid("interface coverage overflow"))?;
                    }
                    work.push(
                        &mut pairs,
                        Pair {
                            left,
                            right: Some(right),
                            piece,
                        },
                    )?;
                }
                ancestor = self.faces[id].parent;
            }
        }
        let full = 1u64 << (20 * (D - 1));
        for (incidence, &covered) in coverage.iter().enumerate() {
            work.tick()?;
            if covered == 0 {
                work.push(
                    &mut pairs,
                    Pair {
                        left: incidence,
                        right: None,
                        piece: face_at(incidence),
                    },
                )?;
            } else if covered != full {
                return Err(work
                    .invalid("interior interface pieces must cover each whole face exactly once"));
            }
        }
        work.sort(&mut pairs)?;
        for pair in pairs {
            work.tick()?;
            let left = self.nodes[self.active[pair.left / arity]];
            let a = left.faces[pair.left % arity];
            let lf = work
                .search(&self.active_faces, &a)?
                .map_err(|_| work.invalid("missing active face"))?;
            let piece = work
                .search(&self.active_faces, &pair.piece)?
                .map_err(|_| work.invalid("missing interface piece"))?;
            let (right, left_sign, periodic) = if let Some(right) = pair.right {
                let node = self.nodes[self.active[right / arity]];
                let b = node.faces[right % arity];
                let rf = work
                    .search(&self.active_faces, &b)?
                    .map_err(|_| work.invalid("missing active face"))?;
                let sign = if pair.piece == a {
                    left.signs[pair.left % arity]
                } else {
                    -node.signs[right % arity]
                };
                if classes[a] != classes[b] {
                    let coarse = if pair.piece == b { a } else { b };
                    self.add_hanging(pair.piece, coarse, work)?;
                }
                (
                    Some(AdaptiveInterfaceSide {
                        cell: right / arity,
                        face: rf,
                    }),
                    sign,
                    self.face_seam(a).is_some() != self.face_seam(b).is_some(),
                )
            } else {
                (None, left.signs[pair.left % arity], false)
            };
            work.push(
                &mut self.interfaces,
                AdaptiveInterface {
                    left_cell: pair.left / arity,
                    right,
                    left_face: lf,
                    piece_face: piece,
                    left_sign,
                    periodic,
                },
            )?;
        }
        work.sort(&mut self.hanging)?;
        self.hanging.dedup();
        Ok(())
    }

    fn add_hanging(
        &mut self,
        piece: usize,
        coarse: usize,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<()> {
        let coarse_face = work
            .search(&self.active_faces, &coarse)?
            .map_err(|_| work.invalid("missing coarse face"))?;
        let entity = work
            .search(&self.active_faces, &piece)?
            .map_err(|_| work.invalid("missing fine face"))?;
        work.push(
            &mut self.hanging,
            HangingEntity {
                kind: HangingEntityKind::Face,
                entity,
                coarse_face,
            },
        )?;
        let fine = self.faces[piece];
        let parent = self.faces[coarse];
        let mut corner = [false; 4];
        for (i, &v) in fine.vertices[..usize::from(fine.len)].iter().enumerate() {
            work.tick()?;
            corner[i] = parent.vertices[..usize::from(parent.len)]
                .iter()
                .any(|&c| self.face_vertex_key(coarse, c) == self.face_vertex_key(piece, v));
            if !corner[i] {
                let entity = work
                    .search(&self.active_vertices, &v)?
                    .map_err(|_| work.invalid("missing hanging vertex"))?;
                work.push(
                    &mut self.hanging,
                    HangingEntity {
                        kind: HangingEntityKind::Vertex,
                        entity,
                        coarse_face,
                    },
                )?;
            }
            for j in 0..i {
                if corner[i] && corner[j] {
                    continue;
                }
                let mut edge = [fine.vertices[j], v];
                edge.sort_unstable();
                if let Ok(entity) = work.search(&self.edges, &edge)? {
                    work.push(
                        &mut self.hanging,
                        HangingEntity {
                            kind: HangingEntityKind::Edge,
                            entity,
                            coarse_face,
                        },
                    )?;
                }
            }
        }
        Ok(())
    }
}
