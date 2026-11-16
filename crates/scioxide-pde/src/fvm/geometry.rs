//! Shared oriented finite-volume geometry; numerical face laws live in their owners.
//!
//! Invalid geometry is reported against the caller's `mesh` argument.
use crate::validation::invalid;

use crate::spatial::{midpoint, positive_width};

use alloc::vec::Vec;
use scioxide_core::backend::try_with_capacity;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};

use super::lane::Lane;
use scioxide_spatial::{AxisOrientation, CellMesh, SimplexMesh};

pub(crate) use crate::boundary::{structured_face_at, structured_face_id};

#[derive(Clone)]
pub(crate) struct FaceGeometry<R: RealScalar, const D: usize> {
    pub(crate) point: [R; D],
    pub(crate) normal: [R; D],
    pub(crate) area: R,
    pub(crate) left: Option<usize>,
    pub(crate) right: Option<usize>,
}

#[derive(Clone)]
pub(crate) struct Geometry<R: RealScalar, const D: usize> {
    pub(crate) centers: Vec<[R; D]>,
    pub(crate) volumes: Vec<R>,
    pub(crate) faces: Vec<FaceGeometry<R, D>>,
    pub(crate) cell_faces: Vec<Vec<usize>>,
}

pub(crate) fn dot<L: Lane, const D: usize>(a: [L; D], b: [L; D]) -> L {
    a.into_iter()
        .zip(b)
        .fold(L::ZERO, |sum, (x, y)| sum + x * y)
}
pub(crate) fn difference<R: RealScalar, const D: usize>(a: [R; D], b: [R; D]) -> [R; D] {
    core::array::from_fn(|axis| a[axis] - b[axis])
}
pub(crate) fn structured_geometry<R: RealScalar, const D: usize>(
    mesh: &CellMesh<R, D>,
) -> Result<Geometry<R, D>, SciOxideError> {
    let cells = mesh.cell_extent();
    for axis in 0..D {
        if mesh.grid().axis(axis)?.orientation() != AxisOrientation::Increasing {
            return Err(invalid(
                scioxide_core::argument!("mesh"),
                "structured FVM axes must increase",
            ));
        }
    }
    let mut centers = try_with_capacity(
        cells.element_count()?,
        scioxide_core::quantity!("fvm_cell_centers"),
    )?;
    let mut volumes = try_with_capacity(
        cells.element_count()?,
        scioxide_core::quantity!("fvm_cell_volumes"),
    )?;
    for linear in 0..cells.element_count()? {
        let index = cells.delinearize(linear)?;
        let mut point = [R::ZERO; D];
        let mut volume = R::ONE;
        for (axis, slot) in point.iter_mut().enumerate() {
            let coordinate = index.as_array()[axis];
            let grid_axis = mesh.grid().axis(axis)?;
            let left = grid_axis.coordinate(coordinate)?;
            let right = grid_axis.coordinate(coordinate + 1)?;
            *slot = midpoint(left, right)?;
            volume *= positive_width(right, left, scioxide_core::argument!("mesh"))?;
        }
        if !volume.is_finite() || volume <= R::ZERO {
            return Err(invalid(
                scioxide_core::argument!("mesh"),
                "FVM cell volume is not finite and positive",
            ));
        }
        centers.push(point);
        volumes.push(volume);
    }
    let mut faces = try_with_capacity(
        mesh.topology_sizes().faces(),
        scioxide_core::quantity!("fvm_faces"),
    )?;
    let mut cell_faces = try_with_capacity(
        centers.len(),
        scioxide_core::quantity!("fvm_cell_face_lists"),
    )?;
    for _ in 0..centers.len() {
        cell_faces.push(try_with_capacity(
            2 * D,
            scioxide_core::quantity!("fvm_cell_faces"),
        )?);
    }
    for id in 0..mesh.topology_sizes().faces() {
        let face = structured_face_at(mesh, id)?;
        let axis = face.normal_axis();
        let index = face.index();
        let position = index.as_array();
        let mut point = [R::ZERO; D];
        let mut area = R::ONE;
        for i in 0..D {
            let grid_axis = mesh.grid().axis(i)?;
            let left = grid_axis.coordinate(position[i])?;
            if i == axis {
                point[i] = left;
            } else {
                let right = grid_axis.coordinate(position[i] + 1)?;
                point[i] = midpoint(left, right)?;
                area *= positive_width(right, left, scioxide_core::argument!("mesh"))?;
            }
        }
        if !area.is_finite() || area <= R::ZERO {
            return Err(invalid(
                scioxide_core::argument!("mesh"),
                "FVM face area is not finite and positive",
            ));
        }
        let mut normal = [R::ZERO; D];
        normal[axis] = R::ONE;
        // Geometric endpoint entities remain exterior here, even on a periodic
        // grid. The conservation owner pairs them once after geometry creation.
        let mut neighbor = *position;
        let left = if position[axis] > 0 {
            neighbor[axis] -= 1;
            Some(cells.linearize(scioxide_spatial::Index::new(neighbor)?)?)
        } else {
            None
        };
        let right = if position[axis] < cells.as_array()[axis] {
            Some(cells.linearize(index)?)
        } else {
            None
        };
        if let Some(cell) = left {
            cell_faces[cell].push(id);
        }
        if let Some(cell) = right {
            cell_faces[cell].push(id);
        }
        faces.push(FaceGeometry {
            point,
            normal,
            area,
            left,
            right,
        });
    }
    Ok(Geometry {
        centers,
        volumes,
        faces,
        cell_faces,
    })
}

pub(crate) fn simplex_geometry<R: RealScalar, const D: usize>(
    mesh: &SimplexMesh<R, D>,
) -> Result<Geometry<R, D>, SciOxideError> {
    let mut centers = try_with_capacity(
        mesh.cell_count(),
        scioxide_core::quantity!("fvm_simplex_centroids"),
    )?;
    let mut volumes = try_with_capacity(
        mesh.cell_count(),
        scioxide_core::quantity!("fvm_simplex_volumes"),
    )?;
    let divisor = R::try_from_usize_exact(D + 1)?;
    for (cell, vertices) in mesh.cells().enumerate() {
        let mut point = [R::ZERO; D];
        for &vertex in vertices {
            let coordinates = mesh.points()[vertex].as_array();
            for axis in 0..D {
                point[axis] += coordinates[axis] / divisor;
            }
        }
        if point.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                scioxide_core::argument!("mesh"),
                "simplex centroid is not finite",
            ));
        }
        let volume = mesh.cell_measure(cell)?;
        centers.push(point);
        volumes.push(volume);
    }
    let mut faces = try_with_capacity(
        mesh.face_count(),
        scioxide_core::quantity!("fvm_simplex_faces"),
    )?;
    let mut cell_faces = try_with_capacity(
        centers.len(),
        scioxide_core::quantity!("fvm_simplex_face_lists"),
    )?;
    for _ in 0..centers.len() {
        cell_faces.push(try_with_capacity(
            D + 1,
            scioxide_core::quantity!("fvm_simplex_cell_faces"),
        )?);
    }
    let face_divisor = R::try_from_usize_exact(D)?;
    for face in 0..mesh.face_count() {
        let vertices = mesh.face_vertices(face)?;
        let mut point = [R::ZERO; D];
        for &vertex in vertices {
            let coordinates = mesh.points()[vertex].as_array();
            for axis in 0..D {
                point[axis] += coordinates[axis] / face_divisor;
            }
        }
        let area = mesh.face_measure(face)?;
        let first_vertex = mesh.points()[vertices[0]].as_array();
        let mut normal = [R::ZERO; D];
        if D == 1 {
            // A vertex face: orientation below selects the sign.
            normal[0] = R::ONE;
        } else if D == 2 {
            let second_vertex = mesh.points()[vertices[1]].as_array();
            normal[0] = (second_vertex[1] - first_vertex[1]) / area;
            normal[1] = (first_vertex[0] - second_vertex[0]) / area;
        } else {
            let second_vertex = mesh.points()[vertices[1]].as_array();
            let third_vertex = mesh.points()[vertices[2]].as_array();
            let a = difference(*second_vertex, *first_vertex);
            let b = difference(*third_vertex, *first_vertex);
            let twice_area = area + area;
            normal[0] = (a[1] * b[2] - a[2] * b[1]) / twice_area;
            normal[1] = (a[2] * b[0] - a[0] * b[2]) / twice_area;
            normal[2] = (a[0] * b[1] - a[1] * b[0]) / twice_area;
        }
        let cells = mesh.face_cells(face)?;
        let left = cells[0];
        let orientation = dot(normal, difference(point, centers[left]));
        if !orientation.is_finite() || orientation == R::ZERO {
            return Err(invalid(
                scioxide_core::argument!("mesh"),
                "simplex face normal cannot be oriented",
            ));
        }
        if orientation < R::ZERO {
            for value in &mut normal {
                *value = -*value;
            }
        }
        if point.iter().any(|value| !value.is_finite())
            || normal.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                scioxide_core::argument!("mesh"),
                "simplex face geometry is not finite",
            ));
        }
        cell_faces[left].push(face);
        if let Some(&right) = cells.get(1) {
            cell_faces[right].push(face);
        }
        faces.push(FaceGeometry {
            point,
            normal,
            area,
            left: Some(left),
            right: cells.get(1).copied(),
        });
    }
    Ok(Geometry {
        centers,
        volumes,
        faces,
        cell_faces,
    })
}

#[expect(
    clippy::needless_range_loop,
    reason = "pivoted elimination needs ordered matrix indices"
)]
pub(crate) fn inverse_normal_matrix<L: Lane, const D: usize>(
    rows: &[[L; D]],
) -> Result<[[L; D]; D], SciOxideError> {
    let mut matrix = [[L::ZERO; D]; D];
    for &row in rows {
        let length_sq = dot(row, row);
        if length_sq.primal() == L::Real::INFINITY && row.iter().all(|value| value.all_finite()) {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Overflow,
                "assembled operator entry is unrepresentable",
            ));
        }
        if !length_sq.all_finite() || length_sq.primal() <= L::Real::ZERO {
            return Err(invalid(
                scioxide_core::argument!("mesh"),
                "zero or unrepresentable gradient constraint",
            ));
        }
        for i in 0..D {
            for j in 0..D {
                matrix[i][j] = matrix[i][j] + row[i] * row[j] / length_sq;
            }
        }
    }
    let mut inverse = [[L::ZERO; D]; D];
    for i in 0..D {
        inverse[i][i] = L::ONE;
    }
    let threshold = L::Real::EPSILON * L::Real::try_from_usize_exact(64)?;
    for pivot in 0..D {
        let mut best = pivot;
        for row in pivot + 1..D {
            if matrix[row][pivot].primal().abs() > matrix[best][pivot].primal().abs() {
                best = row;
            }
        }
        if matrix[best][pivot].primal().abs() <= threshold {
            return Err(invalid(
                scioxide_core::argument!("mesh"),
                "cell gradient stencil is rank deficient",
            ));
        }
        matrix.swap(pivot, best);
        inverse.swap(pivot, best);
        let diagonal = matrix[pivot][pivot];
        for col in 0..D {
            matrix[pivot][col] = matrix[pivot][col] / diagonal;
            inverse[pivot][col] = inverse[pivot][col] / diagonal;
        }
        for row in 0..D {
            if row != pivot {
                let factor = matrix[row][pivot];
                for col in 0..D {
                    matrix[row][col] = matrix[row][col] - factor * matrix[pivot][col];
                    inverse[row][col] = inverse[row][col] - factor * inverse[pivot][col];
                }
            }
        }
    }
    Ok(inverse)
}
