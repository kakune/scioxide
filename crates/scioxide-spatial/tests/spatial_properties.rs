#![allow(clippy::unwrap_used)]

use scioxide_spatial::{
    AxisOrientation, AxisTopology, CellMesh, CoordinateAxis, Extent3, Grid1d, Grid3d, Index1,
    Index3, Point1,
};

fn next_state(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

#[test]
fn deterministic_index_round_trip_property_covers_large_integer_components() {
    let extents = [
        [1, 1, 1],
        [17, 13, 11],
        [usize::MAX, 1, 1],
        [1, usize::MAX / 2, 2],
    ];
    let mut state = 0x4752_4144_4941_0401;
    for axes in extents {
        let extent = Extent3::new(axes).unwrap();
        let count = extent.element_count().unwrap();
        for _ in 0..256 {
            let linear = (next_state(&mut state) as usize) % count;
            let index = extent.delinearize(linear).unwrap();
            assert_eq!(extent.linearize(index).unwrap(), linear);
        }
    }
}

fn assert_coordinate_round_trips_f64(start: f64, end: f64) {
    let axis = CoordinateAxis::uniform(start, end, 65).unwrap();
    for site in 0..axis.len() {
        let coordinate = axis.coordinate(site).unwrap();
        let location = axis.locate(coordinate).unwrap();
        assert_eq!(location.cell(), site.min(axis.len() - 2));
    }
}

fn assert_coordinate_round_trips_f32(start: f32, end: f32) {
    let axis = CoordinateAxis::uniform(start, end, 65).unwrap();
    for site in 0..axis.len() {
        let coordinate = axis.coordinate(site).unwrap();
        let location = axis.locate(coordinate).unwrap();
        assert_eq!(location.cell(), site.min(axis.len() - 2));
    }
}

#[test]
fn coordinate_round_trip_property_covers_extreme_f32_and_f64_scales() {
    assert_coordinate_round_trips_f64(-f64::MAX, f64::MAX);
    assert_coordinate_round_trips_f64(f64::MIN_POSITIVE, 1.0);
    assert_coordinate_round_trips_f64(1.0, -1.0);
    assert_coordinate_round_trips_f32(-f32::MAX, f32::MAX);
    assert_coordinate_round_trips_f32(f32::MIN_POSITIVE, 1.0);
    assert_coordinate_round_trips_f32(1.0, -1.0);
}

#[test]
fn orientation_parity_property_covers_every_three_axis_combination() {
    for mask in 0_u8..8 {
        let axes = core::array::from_fn(|axis| {
            if mask & (1 << axis) == 0 {
                CoordinateAxis::uniform(0.0_f64, 1.0, 3).unwrap()
            } else {
                CoordinateAxis::uniform(1.0_f64, 0.0, 3).unwrap()
            }
        });
        let grid = Grid3d::new(axes, [AxisTopology::Bounded; 3]).unwrap();
        let expected = if mask.count_ones() % 2 == 0 {
            AxisOrientation::Increasing
        } else {
            AxisOrientation::Decreasing
        };
        assert_eq!(grid.orientation(), expected);
    }
}

#[test]
fn adjacency_symmetry_property_covers_mixed_periodic_topology() {
    let mesh = CellMesh::new(
        Grid3d::new(
            [
                CoordinateAxis::uniform(0.0_f32, 4.0, 5).unwrap(),
                CoordinateAxis::uniform(0.0_f32, 3.0, 4).unwrap(),
                CoordinateAxis::uniform(0.0_f32, 2.0, 3).unwrap(),
            ],
            [
                AxisTopology::Periodic,
                AxisTopology::Bounded,
                AxisTopology::Periodic,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    for linear in 0..mesh.cell_extent().element_count().unwrap() {
        let cell = mesh.cell_extent().delinearize(linear).unwrap();
        for neighbor in mesh.cell_neighbors(cell).unwrap() {
            assert!(
                mesh.cell_neighbors(neighbor)
                    .unwrap()
                    .any(|item| item == cell)
            );
        }
    }
}

#[test]
fn refinement_property_preserves_family_and_advances_only_real_changes() {
    let grid = Grid1d::new(
        [CoordinateAxis::uniform(0.0_f64, 1.0, 9).unwrap()],
        [AxisTopology::Bounded],
    )
    .unwrap();
    let identity = grid.refined([1]).unwrap();
    assert!(grid.mesh_token().is_compatible_with(identity.mesh_token()));
    for factor in [2, 3, 5] {
        let refined = grid.refined([factor]).unwrap();
        assert!(grid.mesh_token().same_family(refined.mesh_token()));
        assert_eq!(refined.mesh_token().generation().get(), 1);
        assert_eq!(refined.vertex_extent().as_array(), &[8 * factor + 1]);
    }
}

#[test]
fn periodic_wrap_handles_tiny_and_unrepresentable_spans() {
    let next = 1.0_f64.next_up();
    let tiny = Grid1d::new(
        [CoordinateAxis::uniform(1.0, next, 2).unwrap()],
        [AxisTopology::Periodic],
    )
    .unwrap();
    let wrapped = tiny
        .wrap_point(Point1::new([f64::MAX]).unwrap())
        .unwrap()
        .into_array()[0];
    assert!((1.0..=next).contains(&wrapped));

    let wide = Grid1d::new(
        [CoordinateAxis::uniform(-f64::MAX, 1.0, 3).unwrap()],
        [AxisTopology::Periodic],
    )
    .unwrap();
    let wrapped = wide
        .wrap_point(Point1::new([f64::MAX]).unwrap())
        .unwrap()
        .into_array()[0];
    assert!((-f64::MAX..=1.0).contains(&wrapped));
}

#[test]
fn structured_incidence_never_converts_large_indices_to_coordinates() {
    let grid = Grid1d::new(
        [CoordinateAxis::uniform(0.0_f64, 1.0, (1_usize << 32) + 1).unwrap()],
        [AxisTopology::Bounded],
    )
    .unwrap();
    let mesh = CellMesh::new(grid).unwrap();
    let cell = Index1::new([(1_usize << 32) - 1]).unwrap();
    let vertices = mesh.cell_vertices(cell).unwrap().collect::<Vec<_>>();
    assert_eq!(vertices[0].as_array(), &[(1_usize << 32) - 1]);
    assert_eq!(vertices[1].as_array(), &[1_usize << 32]);
}

#[test]
fn invalid_topology_inputs_fail_before_publication() {
    assert!(CoordinateAxis::explicit(vec![0.0_f64, 1.0, 1.0]).is_err());
    assert!(CoordinateAxis::uniform(f64::NAN, 1.0, 2).is_err());
    assert!(
        Extent3::new([usize::MAX, 2, 1])
            .unwrap()
            .element_count()
            .is_err()
    );
    assert!(Index3::new([0, 0, 0]).is_ok());
}

fn assert_iterator_sequence<T: core::fmt::Debug + PartialEq>(
    mut iter: impl ExactSizeIterator<Item = T> + Clone + core::fmt::Debug,
    expected: &[T],
) {
    assert_eq!(iter.clone().collect::<Vec<_>>(), expected);
    for (position, value) in expected.iter().enumerate() {
        let remaining = expected.len() - position;
        assert_eq!(iter.len(), remaining);
        assert_eq!(iter.size_hint(), (remaining, Some(remaining)));
        assert_eq!(iter.next().as_ref(), Some(value));
    }
    assert_eq!(iter.len(), 0);
    assert_eq!(iter.size_hint(), (0, Some(0)));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
}

#[test]
fn structured_iterators_preserve_order_size_cloning_and_exhaustion() {
    use scioxide_spatial::{AxisSide, FaceIndex, Index2, StructuredGrid};
    let mesh = CellMesh::new(
        StructuredGrid::new(
            core::array::from_fn(|_| CoordinateAxis::uniform(0.0_f64, 2.0, 3).unwrap()),
            [AxisTopology::Periodic, AxisTopology::Bounded],
        )
        .unwrap(),
    )
    .unwrap();
    let cell = Index2::new([0, 0]).unwrap();
    assert_iterator_sequence(
        mesh.grid()
            .axis(0)
            .unwrap()
            .coordinates()
            .map(|v| v.unwrap()),
        &[0.0, 1.0, 2.0],
    );
    assert_iterator_sequence(
        mesh.cell_vertices(cell).unwrap().map(|v| *v.as_array()),
        &[[0, 0], [1, 0], [0, 1], [1, 1]],
    );
    assert_iterator_sequence(
        mesh.cell_neighbors(cell).unwrap().map(|v| *v.as_array()),
        &[[1, 0], [0, 1]],
    );
    let vertex = Index2::new([0, 1]).unwrap();
    assert_iterator_sequence(
        mesh.vertex_cells(vertex).unwrap().map(|v| *v.as_array()),
        &[[1, 0], [0, 0], [1, 1], [0, 1]],
    );
    let face = FaceIndex::new(0, cell).unwrap();
    assert_iterator_sequence(
        mesh.face_cells(face).unwrap().map(|v| *v.as_array()),
        &[[1, 0], [0, 0]],
    );
    assert_iterator_sequence(
        mesh.cell_faces(cell)
            .unwrap()
            .map(|f| (f.normal_axis(), *f.index().as_array())),
        &[(0, [0, 0]), (0, [1, 0]), (1, [0, 0]), (1, [0, 1])],
    );
    assert_iterator_sequence(
        mesh.axis_boundary_faces(0, AxisSide::End)
            .unwrap()
            .map(|f| *f.index().as_array()),
        &[[2, 0], [2, 1]],
    );
    // Incidence iterators own their fixed storage and outlive the mesh handle.
    let neighbors = mesh.cell_neighbors(cell).unwrap();
    drop(mesh);
    assert_iterator_sequence(neighbors.map(|v| *v.as_array()), &[[1, 0], [0, 1]]);

    let single = CellMesh::new(
        Grid1d::new(
            [CoordinateAxis::uniform(0.0_f64, 1.0, 2).unwrap()],
            [AxisTopology::Periodic],
        )
        .unwrap(),
    )
    .unwrap();
    let cell = Index1::new([0]).unwrap();
    assert_iterator_sequence(single.cell_neighbors(cell).unwrap(), &[]);
    assert_iterator_sequence(single.vertex_cells(cell).unwrap(), &[cell]);
    assert_iterator_sequence(
        single.face_cells(FaceIndex::new(0, cell).unwrap()).unwrap(),
        &[cell],
    );
    assert!(single.cell_neighbors(Index1::new([1]).unwrap()).is_err());
    assert!(single.vertex_cells(Index1::new([2]).unwrap()).is_err());
}
