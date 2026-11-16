#![allow(dead_code, unreachable_pub, clippy::unwrap_used)]
//! One-dimensional segment meshes built through the public spatial constructors.
use scioxide_core::RealScalar;
use scioxide_spatial::{
    AxisSide, AxisTopology, BoundarySet, CoordinateAxis, SegmentMesh, StructuredGrid,
};
use std::vec::Vec;

/// Bounded uniform mesh of `count` nodes on `[lower, upper]`.
pub fn uniform<S: RealScalar>(lower: S, upper: S, count: usize) -> SegmentMesh<S> {
    SegmentMesh::new(
        StructuredGrid::new(
            [CoordinateAxis::uniform(lower, upper, count).unwrap()],
            [AxisTopology::Bounded],
        )
        .unwrap(),
    )
    .unwrap()
}

/// Bounded mesh with explicit increasing `nodes`.
pub fn explicit<S: RealScalar>(nodes: Vec<S>) -> SegmentMesh<S> {
    SegmentMesh::new(
        StructuredGrid::new(
            [CoordinateAxis::explicit(nodes).unwrap()],
            [AxisTopology::Bounded],
        )
        .unwrap(),
    )
    .unwrap()
}

/// Node coordinates of a segment mesh.
pub fn nodes<S: RealScalar>(mesh: &SegmentMesh<S>) -> Vec<S> {
    mesh.grid()
        .axis(0)
        .unwrap()
        .coordinates()
        .collect::<Result<_, _>>()
        .unwrap()
}

/// One end of a segment mesh as a boundary set.
pub fn endpoint<S: RealScalar>(mesh: &SegmentMesh<S>, side: AxisSide) -> BoundarySet<1> {
    BoundarySet::axis_side(mesh, 0, side).unwrap()
}

/// Periodic copy of a segment mesh with the same nodes.
pub fn periodic<S: RealScalar>(mesh: &SegmentMesh<S>) -> SegmentMesh<S> {
    SegmentMesh::new(
        StructuredGrid::new(
            [CoordinateAxis::explicit(nodes(mesh)).unwrap()],
            [AxisTopology::Periodic],
        )
        .unwrap(),
    )
    .unwrap()
}
