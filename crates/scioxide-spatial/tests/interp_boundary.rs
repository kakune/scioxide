#![no_std]
#![allow(clippy::cast_precision_loss, clippy::unwrap_used)]

extern crate alloc;
extern crate std;

use scioxide_spatial::{AxisTopology, CoordinateAxis, Grid1d, Point1, SegmentMesh, VertexField};

#[test]
fn no_std_interpolation_boundary_uses_borrowed_coordinates_and_typed_values() {
    let grid = Grid1d::new(
        [CoordinateAxis::uniform(-1.0_f32, 1.0, 17).unwrap()],
        [AxisTopology::Bounded],
    )
    .unwrap();
    let mesh = SegmentMesh::new(grid).unwrap();
    let values = VertexField::<f64, 1>::from_fn(&mesh, |index| index as f64).unwrap();
    let location = mesh
        .grid()
        .locate(Point1::new([0.25_f32]).unwrap())
        .unwrap();
    assert_eq!(location.cell().as_array(), &[10]);
    assert_eq!(values.view().len(), 17);
    assert!(values.is_compatible_with(&mesh));
}
