use super::*;
use crate::{PredicateSign, orient2d, orient3d};
use scioxide_core::checked_affine;

// Vertex identity is exact provenance, not rounded coordinates. Each vertex
// is sum_i (w_i / 2^exponent) * root_i. A key is reduced to lowest terms with
// terms sorted by root id, so the same geometric point reached through
// different parents (for example a face center shared by two neighbors)
// gets an identical key, and so an identical vertex. Rounded coordinates
// could differ in the last bit along different paths.
impl Key {
    // Averages 2, 4 or 8 keys. The count is a power of two, so division is
    // an exponent increment. Keys are aligned to the largest exponent,
    // like-root weights are merged, and common factors of two are
    // cancelled. A refined vertex stays inside one root cell, so at most
    // eight roots contribute. With depth <= 20 and at most 3 exponent bits
    // per level, the weights fit in u64.
    fn mean(keys: &[Self], work: &Work<'_, '_, '_>) -> Result<Self> {
        let exponent = keys
            .iter()
            .map(|k| k.exponent)
            .max()
            .ok_or_else(|| work.invalid("empty midpoint"))?;
        let power = keys.len().trailing_zeros() as u8;
        let mut terms = [(0usize, 0u128); 64];
        let mut len = 0;
        for key in keys {
            for &(id, weight) in &key.terms[..usize::from(key.len)] {
                terms[len] = (id, u128::from(weight) << u32::from(exponent - key.exponent));
                len += 1;
            }
        }
        terms[..len].sort_unstable_by_key(|t| t.0);
        let mut combined = [(0usize, 0u128); 8];
        let mut count = 0;
        for &(id, weight) in &terms[..len] {
            if count > 0 && combined[count - 1].0 == id {
                combined[count - 1].1 += weight;
            } else {
                if count == 8 {
                    return Err(work.invalid("vertex exceeds root support"));
                }
                combined[count] = (id, weight);
                count += 1;
            }
        }
        let mut exponent = exponent + power;
        while exponent > 0 && combined[..count].iter().all(|t| t.1 % 2 == 0) {
            for term in &mut combined[..count] {
                term.1 /= 2;
            }
            exponent -= 1;
        }
        let mut result = Self {
            terms: [(0, 0); 8],
            len: count as u8,
            exponent,
        };
        for (out, &(id, weight)) in result.terms.iter_mut().zip(&combined[..count]) {
            *out = (
                id,
                u64::try_from(weight).map_err(|_| work.invalid("dyadic provenance overflow"))?,
            );
        }
        Ok(result)
    }
}
impl<S: RealScalar, const D: usize> Storage<S, D> {
    fn find_vertex(&self, key: &Key, work: &mut Work<'_, '_, '_>) -> Result<Option<usize>> {
        #[cfg(test)]
        if self.exhaustive {
            for (id, vertex) in self.vertices.iter().enumerate() {
                work.tick()?;
                if &vertex.key == key {
                    return Ok(Some(id));
                }
            }
            return Ok(None);
        }
        self.vertex_index.get(key, work)
    }
    fn find_face(
        &self,
        key: &(u8, [usize; 4]),
        work: &mut Work<'_, '_, '_>,
    ) -> Result<Option<usize>> {
        #[cfg(test)]
        if self.exhaustive {
            for (id, face) in self.faces.iter().enumerate() {
                work.tick()?;
                if (face.len, face.vertices) == *key {
                    return Ok(Some(id));
                }
            }
            return Ok(None);
        }
        self.face_index.get(key, work)
    }
    pub(super) fn add_root(
        &mut self,
        point: Point<S, D>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<()> {
        work.tick()?;
        if self.vertices.len() >= work.limits.max_entities {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("adaptive_vertices"),
                self.vertices.len().saturating_add(1),
                work.limits.max_entities,
            ));
        }
        let key = Key::root(self.vertices.len());
        self.vertex_index.insert(key, self.vertices.len(), work)?;
        work.push(&mut self.vertices, Vertex { point, key })
    }
    fn mean_vertex(&mut self, indices: &[usize], work: &mut Work<'_, '_, '_>) -> Result<usize> {
        let mut keys = [Key::root(0); 8];
        for (key, &id) in keys.iter_mut().zip(indices) {
            *key = self.vertices[id].key;
        }
        let key = Key::mean(&keys[..indices.len()], work)?;
        if let Some(id) = self.find_vertex(&key, work)? {
            return Ok(id);
        }
        if self.vertices.len() >= work.limits.max_entities {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("adaptive_vertices"),
                self.vertices.len().saturating_add(1),
                work.limits.max_entities,
            ));
        }
        // Pairwise midpoint reduction uses the core range-safe affine primitive.
        let mut coordinates = [[S::ZERO; D]; 8];
        for (out, &id) in coordinates.iter_mut().zip(indices) {
            *out = *self.vertices[id].point.as_array();
        }
        let half = S::HALF;
        let mut len = indices.len();
        while len > 1 {
            for i in 0..len / 2 {
                let left = coordinates[2 * i];
                let right = coordinates[2 * i + 1];
                for ((output, a), b) in coordinates[i].iter_mut().zip(left).zip(right) {
                    *output = checked_affine(a, b, half)?;
                }
            }
            len /= 2;
        }
        // A new point equal to one of its contributors means the precision
        // is exhausted at this scale. Accepting it would create a
        // zero-length edge.
        let point = Point::new(coordinates[0])?;
        if indices.iter().any(|&id| self.vertices[id].point == point) {
            return Err(work.invalid("refinement coordinate collapsed"));
        }
        let id = self.vertices.len();
        work.push(&mut self.vertices, Vertex { point, key })?;
        self.vertex_index.insert(key, id, work)?;
        Ok(id)
    }
    pub(super) fn add_node(
        &mut self,
        mut node: Node,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<usize> {
        if self.nodes.len() >= work.limits.max_cells {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("adaptive_cells"),
                self.nodes.len().saturating_add(1),
                work.limits.max_cells,
            ));
        }
        for local in 0..self.face_arity() {
            let (vertices, len, sign) = self.local_face(&node, local);
            // A child face lies inside the parent face whose reference
            // hyperplane contains all of its vertices. For simplices, face 0
            // (opposite vertex 0) is sum r = 2 and face k is r[k-1] = 0. For
            // tensor cells, face 2a + s is r[a] = 0 or 2 (half units).
            // Interior child faces match none of them and have no parent.
            let ancestry = if let Some(parent) = node.parent {
                (0..self.face_arity())
                    .find(|&face| {
                        vertices[..len].iter().all(|id| {
                            let Some(local_vertex) = node.vertices[..self.vertex_arity()]
                                .iter()
                                .position(|v| v == id)
                            else {
                                return false;
                            };
                            let r = node.reference[local_vertex];
                            if self.simplex {
                                if face == 0 {
                                    r[..D].iter().map(|&x| usize::from(x)).sum::<usize>() == 2
                                } else {
                                    r[face - 1] == 0
                                }
                            } else {
                                r[face / 2] == if face % 2 == 0 { 0 } else { 2 }
                            }
                        })
                    })
                    .map(|face| {
                        (
                            self.nodes[parent].faces[face],
                            self.nodes[parent].signs[face] * sign,
                        )
                    })
            } else {
                None
            };
            let parent = ancestry.map(|(id, _)| id);
            let parent_sign = ancestry.map_or(1, |(_, sign)| sign);
            // `sorted` is the identity key. `ordered` keeps a quad's cyclic
            // boundary order in a canonical rotation and direction: it
            // starts at the smallest id and moves toward the smaller
            // neighbor. Consumers can then walk the polygon.
            let mut sorted = vertices;
            sorted[..len].sort_unstable();
            let ordered = if len == 4 {
                let first = (0..4).min_by_key(|&i| vertices[i]).unwrap_or(0);
                let forward = vertices[(first + 1) % 4] < vertices[(first + 3) % 4];
                core::array::from_fn(|i| vertices[(first + if forward { i } else { 4 - i }) % 4])
            } else {
                sorted
            };
            let existing = self.find_face(&(len as u8, sorted), work)?;
            // A face seen before, from a neighbor, must report the same
            // ancestry. `parent == Some(id)` is the unsplit case: a child
            // face that coincides with the whole parent face in 1D.
            let id = if let Some(id) = existing {
                if self.faces[id].parent != parent && parent != Some(id) {
                    return Err(work.invalid("inconsistent face ancestry"));
                }
                if parent != Some(id) && self.faces[id].parent_sign != parent_sign {
                    return Err(work.invalid("inconsistent face orientation"));
                }
                id
            } else {
                if self.faces.len() >= work.limits.max_entities {
                    return Err(SciOxideError::resource_limit(
                        scioxide_core::quantity!("adaptive_faces"),
                        self.faces.len().saturating_add(1),
                        work.limits.max_entities,
                    ));
                }
                let id = self.faces.len();
                work.push(
                    &mut self.faces,
                    Face {
                        vertices: sorted,
                        ordered,
                        len: len as u8,
                        parent,
                        parent_sign,
                        axis: if self.simplex { None } else { Some(local / 2) },
                    },
                )?;
                self.face_index.insert((len as u8, sorted), id, work)?;
                id
            };
            node.faces[local] = id;
            node.signs[local] = sign;
        }
        let id = self.nodes.len();
        work.push(&mut self.nodes, node)?;
        Ok(id)
    }
    // Returns a local face's vertices and its induced boundary sign relative
    // to the canonical (sorted or cyclic) face orientation. Simplices use
    // (-1)^local times the sorting parity, as in `simplex.rs`. Tensor faces
    // take their vertices in bit order. For quads, swapping the last two
    // turns bit order (00, 10, 01, 11) into cyclic order. `base`
    // alternates with axis and side, which gives the outward convention of
    // the reference cell; reflected roots reverse it physically.
    fn local_face(&self, node: &Node, local: usize) -> ([usize; 4], usize, i8) {
        let mut vertices = [0; 4];
        if self.simplex {
            let mut n = 0;
            for i in 0..D + 1 {
                if i != local {
                    vertices[n] = node.vertices[i];
                    n += 1;
                }
            }
            let alternation = if local.is_multiple_of(2) { 1 } else { -1 };
            (
                vertices,
                D,
                alternation * crate::simplex::permutation_sign(&vertices[..D]),
            )
        } else {
            let axis = local / 2;
            let side = local % 2;
            let mut n = 0;
            for i in 0..1 << D {
                if (i >> axis) & 1 == side {
                    vertices[n] = node.vertices[i];
                    n += 1;
                }
            }
            let base = if (axis + side).is_multiple_of(2) {
                -1
            } else {
                1
            };
            let reflection = if self.reflected { -1 } else { 1 };
            let orientation = if n == 2 {
                if vertices[0] < vertices[1] { 1 } else { -1 }
            } else if n == 4 {
                vertices.swap(2, 3);
                let minimum = (0..4).min_by_key(|&i| vertices[i]).unwrap_or(0);
                if vertices[(minimum + 1) % 4] < vertices[(minimum + 3) % 4] {
                    1
                } else {
                    -1
                }
            } else {
                1
            };
            (vertices, n, reflection * base * orientation)
        }
    }
    pub(super) fn split(&mut self, id: usize, work: &mut Work<'_, '_, '_>) -> Result<()> {
        work.tick()?;
        let parent = *at(&self.nodes, id)?;
        if parent.level >= work.limits.max_depth {
            return Err(SciOxideError::resource_limit(
                scioxide_core::quantity!("adaptive_depth"),
                usize::from(parent.level) + 1,
                usize::from(work.limits.max_depth),
            ));
        }
        if let Some(context) = work.context.as_deref_mut()
            && !context
                .try_consume(BudgetKind::Subdivision, 1)?
                .is_granted()
        {
            return Err(budget_exhausted(context, BudgetKind::Subdivision));
        }
        // Coarsening keeps children in the hierarchy. Re-refining just
        // reactivates them, so stable ids and vertices are reused rather
        // than duplicated.
        if parent.child_count > 0 {
            for &child in &parent.children[..usize::from(parent.child_count)] {
                self.nodes[child].active = true;
            }
            self.nodes[id].active = false;
            return Ok(());
        }
        let mut children = [[0usize; 8]; 8];
        let mut refs = [[[0u8; 3]; 8]; 8];
        let count = 1 << D;
        // Tensor cells split dyadically into 2^D children. In half units, a
        // child corner sits at child_bit + corner_bit, which is 0, 1 or 2 on
        // each axis. Its contributors are the parent vertices that agree on
        // every axis where the value is not 1 (the midpoint). One
        // contributor is a parent corner; 2, 4 or 8 are an edge midpoint, a
        // face center or the cell center.
        if !self.simplex {
            for child in 0..count {
                for corner in 0..count {
                    let mut r = [0; 3];
                    for (axis, value) in r[..D].iter_mut().enumerate() {
                        *value = (((child >> axis) & 1) + ((corner >> axis) & 1)) as u8;
                    }
                    refs[child][corner] = r;
                    let mut contributors = [0; 8];
                    let mut n = 0;
                    for root in 0..count {
                        if (0..D).all(|a| r[a] == 1 || ((root >> a) & 1) == usize::from(r[a] / 2)) {
                            contributors[n] = parent.vertices[root];
                            n += 1;
                        }
                    }
                    children[child][corner] = if n == 1 {
                        contributors[0]
                    } else {
                        self.mean_vertex(&contributors[..n], work)?
                    };
                }
            }
        } else {
            // Red refinement: split every edge at its midpoint. A triangle
            // gives three corner triangles plus the middle one. A
            // tetrahedron gives four corner tetrahedra plus an inner
            // octahedron, which is cut into four along one of its three
            // diagonals.
            let mut points = [0usize; 10];
            let mut reference = [[0u8; 3]; 10];
            points[..D + 1].copy_from_slice(&parent.vertices[..D + 1]);
            for i in 1..D + 1 {
                reference[i][i - 1] = 2;
            }
            let mut edge = [[0usize; 4]; 4];
            let mut n = D + 1;
            for a in 0..D + 1 {
                for b in a + 1..D + 1 {
                    edge[a][b] = n;
                    edge[b][a] = n;
                    points[n] =
                        self.mean_vertex(&[parent.vertices[a], parent.vertices[b]], work)?;
                    reference[n] =
                        core::array::from_fn(|axis| (reference[a][axis] + reference[b][axis]) / 2);
                    n += 1;
                }
            }
            let templates = if D == 2 {
                [
                    [0, edge[0][1], edge[0][2], 0],
                    [edge[0][1], 1, edge[1][2], 0],
                    [edge[0][2], edge[1][2], 2, 0],
                    [edge[0][1], edge[1][2], edge[0][2], 0],
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    [0; 4],
                ]
            } else {
                // Opposite midpoint pair selected by canonical original vertex order.
                // The octahedron diagonal is interior, so it does not affect
                // conformity. Choosing it from global ids rather than local
                // order keeps the split independent of how the parent's
                // vertices were listed.
                let mut order = [0, 1, 2, 3];
                order.sort_unstable_by_key(|&i| parent.vertices[i]);
                let [a, b, c, d] = order;
                let ab = edge[a][b];
                let cd = edge[c][d];
                let ring = [edge[a][c], edge[a][d], edge[b][d], edge[b][c]];
                let mut t = [[0; 4]; 8];
                for i in 0..4 {
                    t[i][0] = i;
                    let mut k = 1;
                    for (j, &midpoint) in edge[i].iter().enumerate() {
                        if j != i {
                            t[i][k] = midpoint;
                            k += 1;
                        }
                    }
                }
                for i in 0..4 {
                    t[4 + i] = [ab, cd, ring[i], ring[(i + 1) % 4]];
                }
                t
            };
            for i in 0..count {
                for j in 0..D + 1 {
                    children[i][j] = points[templates[i][j]];
                    refs[i][j] = reference[templates[i][j]];
                }
                let p = |j: usize| self.vertices[children[i][j]].point.as_array();
                let sign = if D == 2 {
                    orient2d(
                        &[p(0)[0], p(0)[1]],
                        &[p(1)[0], p(1)[1]],
                        &[p(2)[0], p(2)[1]],
                    )?
                } else {
                    orient3d(
                        &[p(0)[0], p(0)[1], p(0)[2]],
                        &[p(1)[0], p(1)[1], p(1)[2]],
                        &[p(2)[0], p(2)[1], p(2)[2]],
                        &[p(3)[0], p(3)[1], p(3)[2]],
                    )?
                };
                if sign == PredicateSign::Zero {
                    return Err(work.invalid("degenerate refined simplex"));
                }
                // Template orientation is exact in the half-unit reference
                // cell. Normalizing a *rounding-induced* inversion would hide
                // a folded parent/child map, so reject it before normalization.
                let r = refs[i].map(|point| point.map(f64::from));
                let reference_sign = if D == 2 {
                    orient2d(
                        &[r[0][0], r[0][1]],
                        &[r[1][0], r[1][1]],
                        &[r[2][0], r[2][1]],
                    )?
                } else {
                    orient3d(&r[0], &r[1], &r[2], &r[3])?
                };
                if sign != reference_sign {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::IllConditioned,
                        "rounded coordinates reverse the reference child orientation",
                    ));
                }
                // Restore the mesh orientation convention (counterclockwise,
                // or negative orient3d) by swapping two vertices. The
                // reference map is swapped the same way, so it still
                // describes the child.
                if (D == 2 && sign == PredicateSign::Negative)
                    || (D == 3 && sign == PredicateSign::Positive)
                {
                    children[i].swap(0, 1);
                    refs[i].swap(0, 1);
                }
            }
        }
        let mut ids = [0; 8];
        for i in 0..count {
            let mut child = Node::root(children[i]);
            child.parent = Some(id);
            child.level = parent.level + 1;
            child.reference = refs[i];
            ids[i] = self.add_node(child, work)?;
        }
        self.nodes[id].children = ids;
        self.nodes[id].child_count = count as u8;
        self.nodes[id].active = false;
        Ok(())
    }
}
