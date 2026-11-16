//! Centroid seeds and bounded exact-sign visibility walks. Approximate seed
//! coordinates never decide containment or the lowest-index boundary tie rule.
use crate::tessellation::Result;
use crate::tessellation::cloud::Cloud;
use crate::tessellation::delaunay::Delaunay;
use crate::tessellation::delaunay::SimplexLocation;
use crate::tessellation::storage::retained;
use crate::tessellation::topology::Topology;
use crate::tessellation::work::Work;
use alloc::vec::Vec;
use scioxide_core::RealScalar;
use scioxide_core::backend::checked_mul;

#[derive(Debug)]
struct Entry {
    simplex: usize,
    axis: usize,
}

/// An owned median tree: each subtree is a slice whose root is its middle entry.
/// Unlike the public borrowed KD tree, this index needs only one descent, not a
/// nearest-neighbor proof. Every returned simplex is a valid walk seed.
#[derive(Debug)]
pub(super) struct SeedIndex {
    scales: Vec<f64>,
    centers: Vec<f64>,
    entries: Vec<Entry>,
}

impl SeedIndex {
    pub(super) fn retained_bytes(&self) -> usize {
        retained(&self.scales) + retained(&self.centers) + retained(&self.entries)
    }
    pub(super) fn new<S: RealScalar>(
        cloud: &Cloud<S>,
        topology: &Topology,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<Self> {
        let storage = work.storage;
        let d = cloud.dimension;

        let quantity = scioxide_core::quantity!("location_index");
        let mut scales = storage.filled(d, 0_f64, quantity)?;
        for point in cloud.points.chunks_exact(d) {
            for (scale, &x) in scales.iter_mut().zip(point) {
                *scale = scale.max(x.try_to_f64_exact()?.abs());
            }
        }
        // Constant zero axes cannot occur in a full-dimensional cloud, but a
        // unit divisor also keeps the index independent of that invariant.
        for scale in &mut scales {
            if *scale == 0. {
                *scale = 1.;
            }
        }
        let count = topology.facets.len();
        let mut centers = storage.filled(checked_mul(count, d, quantity)?, 0., quantity)?;
        for (center, facet) in centers.chunks_exact_mut(d).zip(topology.facets.iter()) {
            work.check()?;
            for &vertex in facet.vertices {
                for ((value, &x), &scale) in center.iter_mut().zip(cloud.point(vertex)).zip(&scales)
                {
                    // Divide before summing: even opposite finite extremes and
                    // subnormal sites produce bounded, finite heuristic centers.
                    *value += (x.try_to_f64_exact()? / scale) / (d + 1) as f64;
                }
            }
        }
        let mut entries = storage.capacity(count, quantity)?;
        entries.extend((0..count).map(|simplex| Entry { simplex, axis: 0 }));
        Self::partition(&centers, &mut entries, d, work)?;
        Ok(Self {
            scales,
            centers,
            entries,
        })
    }

    fn partition(
        centers: &[f64],
        entries: &mut [Entry],
        d: usize,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<()> {
        if entries.len() <= 1 {
            return Ok(());
        }
        work.check()?;
        let mut widest = 0;
        let mut span = -1.;
        for axis in 0..d {
            let (low, high) =
                entries
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), entry| {
                        let value = centers[entry.simplex * d + axis];
                        (lo.min(value), hi.max(value))
                    });
            if high - low > span {
                widest = axis;
                span = high - low;
            }
        }
        let middle = entries.len() / 2;
        let (left, root, right) = entries.select_nth_unstable_by(middle, |a, b| {
            centers[a.simplex * d + widest]
                .total_cmp(&centers[b.simplex * d + widest])
                .then_with(|| a.simplex.cmp(&b.simplex))
        });
        root.axis = widest;
        Self::partition(centers, left, d, work)?;
        Self::partition(centers, right, d, work)
    }

    pub(super) fn seed<S: RealScalar>(
        &self,
        point: &[S],
        hint: Option<usize>,
        scratch: &mut [f64],
    ) -> Result<usize> {
        let d = self.scales.len();
        let query = &mut scratch[..d];
        for ((value, &x), &scale) in query.iter_mut().zip(point).zip(&self.scales) {
            // Clamp only the heuristic coordinates, before division to avoid
            // overflow when a finite query is far outside tiny geometry.
            *value = x.try_to_f64_exact()?.clamp(-scale, scale) / scale;
        }
        let mut entries = self.entries.as_slice();
        let mut seed = hint.unwrap_or(entries[entries.len() / 2].simplex);
        let mut distance = self.distance(seed, query);
        while !entries.is_empty() {
            let middle = entries.len() / 2;
            let entry = &entries[middle];
            let candidate = self.distance(entry.simplex, query);
            if candidate < distance || (candidate == distance && entry.simplex < seed) {
                seed = entry.simplex;
                distance = candidate;
            }
            entries = if query[entry.axis] < self.centers[entry.simplex * d + entry.axis] {
                &entries[..middle]
            } else {
                &entries[middle + 1..]
            };
        }
        Ok(seed)
    }

    fn distance(&self, simplex: usize, query: &[f64]) -> f64 {
        let d = self.scales.len();
        self.centers[simplex * d..(simplex + 1) * d]
            .iter()
            .zip(query)
            .map(|(center, x)| (center - x) * (center - x))
            .sum()
    }
}

enum Classification {
    Contained { boundary: bool },
    Beyond { face: usize },
}

impl<S: RealScalar> Delaunay<S> {
    // Each opposite face is classified by replacing its vertex with the query.
    // An incoming face crossed with a strict sign is known to be positive here.
    fn classify(
        &self,
        simplex: usize,
        point: &[S],
        first: usize,
        previous: Option<usize>,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<Classification> {
        let d = self.dimension();
        let facet = &self.topology.facets.at(simplex);
        for (row, &vertex) in facet.vertices.iter().enumerate() {
            for (axis, &value) in self.cloud.point(vertex).iter().enumerate() {
                work.workspace.rows[row * d + axis] = value.try_to_f64_exact()?;
            }
        }
        let mut boundary = false;
        work.workspace.face_vertices.clear();
        for offset in 0..=d {
            let face = (first + offset) % (d + 1);
            if previous.is_some() && self.topology.neighbors[simplex * (d + 1) + face] == previous {
                work.workspace.face_vertices.push(facet.vertices[face]);
                continue;
            }
            for (axis, &value) in point.iter().enumerate() {
                work.workspace.rows[face * d + axis] = value.try_to_f64_exact()?;
            }
            let sign = work.raw(d + 1, d, false)?;
            if sign != 0 && sign != facet.inside {
                return Ok(Classification::Beyond { face });
            }
            boundary |= sign == 0;
            if sign != 0 {
                work.workspace.face_vertices.push(facet.vertices[face]);
            }
            for (axis, &value) in self.cloud.point(facet.vertices[face]).iter().enumerate() {
                work.workspace.rows[face * d + axis] = value.try_to_f64_exact()?;
            }
        }
        Ok(Classification::Contained { boundary })
    }

    // The positive exact barycentric numerators identify the minimal face
    // containing the query. A simplex contains that query exactly when it
    // contains every vertex of this face. No coordinate tests are needed.
    fn boundary_simplex(&self, simplex: usize, face: &[usize]) -> usize {
        // A contained point has at least one positive numerator.
        let star = face
            .iter()
            .skip(1)
            .map(|&v| self.vertex_simplices.at(v))
            .fold(self.vertex_simplices.at(face[0]), |best, row| {
                if row.len() < best.len() { row } else { best }
            });
        if face.len() == 1 {
            return star[0];
        }
        star.iter()
            .copied()
            .take_while(|&candidate| candidate < simplex)
            .find(|&candidate| {
                let vertices = self.topology.facets.at(candidate).vertices;
                face.iter().all(|vertex| vertices.contains(vertex))
            })
            .unwrap_or(simplex)
    }

    fn stalled_walk_scan(
        &self,
        point: &[S],
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<Option<SimplexLocation>> {
        for simplex in 0..self.simplex_count() {
            if let Classification::Contained { boundary } =
                self.classify(simplex, point, 0, None, work)?
            {
                return Ok(Some(SimplexLocation { simplex, boundary }));
            }
        }
        Ok(None)
    }

    // The step limit guarantees termination even for a cycling visibility walk
    // on a symbolically triangulated degeneracy. Tests may supply a smaller cap.
    pub(in crate::tessellation) fn walk(
        &self,
        point: &[S],
        mut simplex: usize,
        steps: usize,
        work: &mut Work<'_, '_, '_, '_>,
    ) -> Result<Option<SimplexLocation>> {
        let mut previous = None;
        let mut random = 0x243f_6a88_85a3_08d3_u64;
        for _ in 0..steps {
            // Query-local xorshift: reproducible stochastic face order with no
            // global state, coordinate perturbation, or random dependency.
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let first = (random % (self.dimension() + 1) as u64) as usize;
            match self.classify(simplex, point, first, previous, work)? {
                Classification::Contained { boundary } => {
                    if boundary {
                        simplex = self.boundary_simplex(simplex, &work.workspace.face_vertices);
                    }
                    return Ok(Some(SimplexLocation { simplex, boundary }));
                }
                Classification::Beyond { face } => {
                    let Some(next) =
                        self.topology.neighbors[simplex * (self.dimension() + 1) + face]
                    else {
                        return Ok(None);
                    };
                    previous = Some(simplex);
                    simplex = next;
                }
            }
        }
        self.stalled_walk_scan(point, work)
    }
}
