use super::*;
use scioxide_core::{ScaledNorm, checked_difference};

/// Cell shape of an [`AdaptiveMesh`](crate::AdaptiveMesh). The dimension is
/// that of the mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveCellKind {
    /// Segments, quadrilaterals or hexahedra, with vertices numbered by bits
    /// as in [`CellMesh::cell_vertices`](crate::CellMesh::cell_vertices).
    Tensor,
    /// Positively oriented triangles or tetrahedra.
    Simplex,
}

impl<S: RealScalar, const D: usize> AdaptiveMesh<S, D> {
    /// Returns the cell shape, also for a mesh without cells.
    pub fn cell_kind(&self) -> AdaptiveCellKind {
        if self.storage.simplex {
            AdaptiveCellKind::Simplex
        } else {
            AdaptiveCellKind::Tensor
        }
    }

    /// Returns the length, area or volume of an active cell, computed from
    /// its rounded vertex coordinates.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `cell >= self.cell_count()`.
    /// - `NumericalInstability` if the measure is not representable.
    pub fn cell_measure(&self, cell: usize) -> Result<S> {
        let mut points = [Point::new([S::ZERO; D])?; 8];
        let mut count = 0;
        for vertex in self.cell_vertices(cell)? {
            points[count] = self.point(vertex)?;
            count += 1;
        }
        if self.storage.simplex {
            crate::simplex::measure::cell(&points, &[0, 1, 2, 3][..count])
        } else {
            tensor_measure(&points[..count], D, scioxide_core::argument!("cell"))
        }
    }

    /// Returns the measure of a face: one for an endpoint in one dimension,
    /// a length in two and an area in three.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `face >= self.face_count()`.
    /// - `NumericalInstability` if the measure is not representable.
    pub fn face_measure(&self, face: usize) -> Result<S> {
        let mut points = [Point::new([S::ZERO; D])?; 4];
        let mut count = 0;
        for vertex in self.face_vertices(face)? {
            points[count] = self.point(vertex)?;
            count += 1;
        }
        if self.storage.simplex {
            crate::simplex::measure::face(&points, &[0, 1, 2][..count])
        } else {
            tensor_measure(&points[..count], D - 1, scioxide_core::argument!("face"))
        }
    }

    /// Returns the diameter of an active cell: the largest distance between
    /// two of its vertices.
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if `cell >= self.cell_count()`.
    /// - `NumericalInstability` if the distance is not representable.
    pub fn cell_diameter(&self, cell: usize) -> Result<S> {
        let mut result = S::ZERO;
        for a in self.cell_vertices(cell)? {
            for b in self.cell_vertices(cell)? {
                if b <= a {
                    continue;
                }
                let mut norm = ScaledNorm::<S>::new();
                for (&x, &y) in self
                    .point(a)?
                    .as_array()
                    .iter()
                    .zip(self.point(b)?.as_array())
                {
                    norm.add(checked_difference(x, y)?)?;
                }
                result = result.max(norm.finish()?);
            }
        }
        Ok(result)
    }
}

// Axis-aligned measure of the entity named by `argument`.
fn tensor_measure<S: RealScalar, const D: usize>(
    points: &[Point<S, D>],
    dimension: usize,
    argument: Argument,
) -> Result<S> {
    let mut result = scioxide_core::Scaled::from_value(S::ONE)?;
    let mut axes = 0;
    for axis in 0..D {
        let (low, high) = points
            .iter()
            .fold((S::INFINITY, S::NEG_INFINITY), |(low, high), p| {
                (low.min(p.as_array()[axis]), high.max(p.as_array()[axis]))
            });
        if low != high {
            axes += 1;
            result = result.mul(
                scioxide_core::Scaled::from_value(high)?
                    .sub(scioxide_core::Scaled::from_value(low)?)?,
            )?;
        }
    }
    if axes != dimension {
        return Err(invalid(
            argument,
            "tensor entity has incorrect geometric dimension",
        ));
    }
    result.try_value()
}
