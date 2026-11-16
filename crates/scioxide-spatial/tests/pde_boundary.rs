#![no_std]
#![allow(clippy::unwrap_used)]

extern crate alloc;
extern crate std;

use scioxide_spatial::{
    AxisSide, AxisTopology, BoundarySet, CellField, CellMesh, CoordinateAxis, Grid2d, Index2,
};

#[test]
fn no_std_operator_boundary_consumes_periodic_and_exterior_topology_uniformly() {
    let mesh = CellMesh::new(
        Grid2d::new(
            [
                CoordinateAxis::uniform(0.0_f64, 1.0, 9).unwrap(),
                CoordinateAxis::uniform(-1.0_f64, 1.0, 5).unwrap(),
            ],
            [AxisTopology::Periodic, AxisTopology::Bounded],
        )
        .unwrap(),
    )
    .unwrap();
    let periodic = BoundarySet::axis_side(&mesh, 0, AxisSide::Start).unwrap();
    let exterior = BoundarySet::axis_side(&mesh, 1, AxisSide::End).unwrap();
    assert_eq!(periodic.len(), 4);
    assert_eq!(exterior.len(), 8);

    let coefficients = CellField::<f64, 2>::from_fn(&mesh, |_| 1.0).unwrap();
    let corner = Index2::new([0, 0]).unwrap();
    assert_eq!(mesh.cell_neighbors(corner).unwrap().count(), 3);
    assert!(coefficients.is_compatible_with(&mesh));
}
