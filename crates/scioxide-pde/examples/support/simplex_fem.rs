//! Shared deterministic meshes for simplex FEM coverage and examples.
#![allow(dead_code, unreachable_pub)]
extern crate alloc;
use alloc::vec::Vec;
use scioxide_core::{RealScalar, SciOxideError};
use scioxide_spatial::{Point, SimplexMesh, TetrahedronMesh, TriangleMesh, UnstructuredEdgeField};

pub fn triangle_grid<R: RealScalar>(n: usize) -> Result<TriangleMesh<R>, SciOxideError> {
    let mut points = Vec::new();
    let mut cells = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            points.push(Point::new([
                R::try_from_usize_exact(i)? / R::try_from_usize_exact(n)?,
                R::try_from_usize_exact(j)? / R::try_from_usize_exact(n)?,
            ])?);
        }
    }
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            let b = a + 1;
            let c = a + n + 1;
            let d = c + 1;
            cells.push([a, b, d]);
            cells.push([a, d, c]);
        }
    }
    TriangleMesh::new(points, cells)
}
pub fn tetrahedron_grid<R: RealScalar>(n: usize) -> Result<TetrahedronMesh<R>, SciOxideError> {
    let mut points = Vec::new();
    let mut cells = Vec::new();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                points.push(Point::new([
                    R::try_from_usize_exact(i)? / R::try_from_usize_exact(n)?,
                    R::try_from_usize_exact(j)? / R::try_from_usize_exact(n)?,
                    R::try_from_usize_exact(k)? / R::try_from_usize_exact(n)?,
                ])?);
            }
        }
    }
    let strides = [1, n + 1, (n + 1) * (n + 1)];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let origin = i + j * strides[1] + k * strides[2];
                for p in [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 0, 2],
                    [1, 2, 0],
                    [2, 0, 1],
                    [2, 1, 0],
                ] {
                    let mut cell = [
                        origin,
                        origin + strides[p[0]],
                        origin + strides[p[0]] + strides[p[1]],
                        origin + strides.iter().sum::<usize>(),
                    ];
                    let inversions = (0..3)
                        .map(|a| (a + 1..3).filter(|&b| p[a] > p[b]).count())
                        .sum::<usize>();
                    if inversions % 2 == 1 {
                        cell.swap(1, 2);
                    }
                    cells.push(cell);
                }
            }
        }
    }
    TetrahedronMesh::new(points, cells)
}
pub fn warp<R: RealScalar, const D: usize>(mut x: [R; D]) -> [R; D] {
    let factor = R::ONE + x[0] * R::HALF;
    for value in &mut x[1..] {
        *value *= factor;
    }
    x
}
pub fn curved<R: RealScalar, const D: usize>(
    base: &SimplexMesh<R, D>,
) -> Result<(SimplexMesh<R, D>, UnstructuredEdgeField<[R; D], D>), SciOxideError> {
    let mesh = base.with_points(
        base.points()
            .iter()
            .map(|p| Point::new(warp(*p.as_array())))
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    let mut points = Vec::new();
    for edge in 0..base.edge_count() {
        let [a, b] = *base.edge_vertices(edge)?;
        points.push(warp(core::array::from_fn(|axis| {
            (base.points()[a].as_array()[axis] + base.points()[b].as_array()[axis]) * R::HALF
        })));
    }
    let fields = UnstructuredEdgeField::new(&mesh, points)?;
    Ok((mesh, fields))
}
