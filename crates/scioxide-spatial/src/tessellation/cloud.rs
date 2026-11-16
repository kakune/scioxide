//! Sorted, validated site storage.
use crate::PointSet;
use crate::tessellation::storage::{Storage, retained};
use crate::tessellation::{GeometryDegeneracy, GeometryOptions, Result, degenerate};
use alloc::vec::Vec;
use scioxide_core::RealScalar;
use scioxide_core::backend::checked_mul;
#[derive(Debug)]
pub(super) struct Cloud<S: RealScalar> {
    pub(super) dimension: usize,
    pub(super) points: Vec<S>,
    pub(super) mapping: Vec<usize>,
    pub(super) representatives: Vec<usize>,
    pub(super) homogeneous: Option<Vec<f64>>,
    // Every coordinate consumed by a fixed filter lies in its proven envelope.
    pub(super) filter_envelope: bool,
}
impl<S: RealScalar> Cloud<S> {
    pub(super) fn retained_bytes(&self) -> usize {
        retained(&self.points)
            + retained(&self.mapping)
            + retained(&self.representatives)
            + self.homogeneous.as_ref().map_or(0, retained)
    }

    // Canonicalize an input admitted by preparation; derived clouds share that admission.
    pub(super) fn new(
        points: &PointSet<'_, S>,
        options: GeometryOptions,
        storage: &Storage,
    ) -> Result<Self> {
        let frame = storage.frame();

        let d = points.dimension();
        let mut original = storage.capacity(
            checked_mul(
                points.len(),
                d,
                scioxide_core::quantity!("point_coordinates"),
            )?,
            scioxide_core::quantity!("point_coordinates"),
        )?;
        for i in 0..points.len() {
            for axis in 0..d {
                original.push(points.coordinate(i, axis)?);
            }
        }
        // Sites are sorted lexicographically by coordinates (input index on
        // ties) and exact duplicates are merged. This canonical site order,
        // independent of input order, is what symbolic perturbation ranks
        // sites by. It also makes every output reproducible. `mapping`
        // sends each input row to its unique representative.
        // Finite coordinates map monotonically to unsigned keys, with both
        // zeros sharing one key, so comparisons are integer comparisons.
        let mut keys =
            storage.capacity(original.len(), scioxide_core::quantity!("point_ordering"))?;
        keys.extend(original.iter().map(|x| {
            let bits = (x.into_f64() + 0.0).to_bits();
            if bits >> 63 == 1 {
                !bits
            } else {
                bits | 1 << 63
            }
        }));
        let mut order =
            storage.capacity(points.len(), scioxide_core::quantity!("point_ordering"))?;
        order.extend(0..points.len());
        order.sort_unstable_by(|&a, &b| {
            keys[a * d..(a + 1) * d]
                .cmp(&keys[b * d..(b + 1) * d])
                .then(a.cmp(&b))
        });
        let mut values = storage.capacity(
            original.len(),
            scioxide_core::quantity!("unique_coordinates"),
        )?;
        let mut mapping =
            storage.filled(points.len(), 0, scioxide_core::quantity!("input_mapping"))?;
        let mut representatives = storage.capacity(
            points.len(),
            scioxide_core::quantity!("representative_mapping"),
        )?;
        for i in order {
            let point = &original[i * d..(i + 1) * d];
            let duplicate = values.len() >= d && &values[values.len() - d..] == point;
            if duplicate && options.degeneracy == GeometryDegeneracy::Strict {
                return Err(degenerate());
            }
            if !duplicate {
                values.extend_from_slice(point);
                representatives.push(i);
            }
            mapping[i] = representatives.len() - 1;
        }
        let result = Self {
            dimension: d,
            filter_envelope: values
                .iter()
                .all(|x| crate::predicates::fixed_in_range(x.into_f64())),
            points: values,
            mapping,
            representatives,
            homogeneous: None,
        };
        frame.keep(0, result.retained_bytes());
        Ok(result)
    }
    pub(super) fn len(&self) -> usize {
        self.representatives.len()
    }
    pub(super) fn point(&self, i: usize) -> &[S] {
        &self.points[i * self.dimension..(i + 1) * self.dimension]
    }
}
