#![allow(clippy::unwrap_used)]

use scioxide_spatial::{FieldView, Point, SimplexBoundaryTags, TetrahedronMesh, UnstructuredFace};

#[test]
fn published_simplex_queries_and_views_execute() {
    let mesh = TetrahedronMesh::new(
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            .map(|p| Point::new(p).unwrap())
            .to_vec(),
        vec![[0, 1, 2, 3]],
    )
    .unwrap();
    let tags = SimplexBoundaryTags::new(&mesh, vec![(0, 1)]).unwrap();
    {
        let clone = mesh.clone();
        assert_eq!(mesh.mesh_token(), clone.mesh_token());
        std::hint::black_box(mesh.cell_measure(0).unwrap());
        std::hint::black_box(mesh.cell_neighbors(0).unwrap().count());
        for face in mesh.cell_faces(0).unwrap() {
            std::hint::black_box(mesh.face_edges(face.index()).unwrap());
            std::hint::black_box(mesh.face_measure(face.index()).unwrap());
            std::hint::black_box(tags.tag(&mesh, face.index()).unwrap());
        }
        for edge in mesh.cell_edges(0).unwrap() {
            std::hint::black_box(mesh.edge_cells(edge.index()).unwrap());
            std::hint::black_box(mesh.edge_length(edge.index()).unwrap());
        }
        let _ = std::hint::black_box(
            FieldView::<_, UnstructuredFace, 3>::new(&mesh, &[0.; 4]).unwrap(),
        );
        std::hint::black_box(mesh.vertex_cells(0).unwrap());
        std::hint::black_box(mesh.boundary_faces().count());
    };
}

#[test]
fn prepared_nd_predicates_execute_on_filter_and_exact_paths() {
    use scioxide_core::Budget;
    use scioxide_spatial::{PredicateSign, SimplexPredicateWorkspace};
    for dimension in 4..=8 {
        let mut points = vec![vec![0.; dimension]; dimension + 1];
        for axis in 0..dimension {
            points[axis + 1][axis] = 1.;
        }
        let mut workspace = SimplexPredicateWorkspace::new(
            dimension,
            Budget::try_from((1..=dimension + 1).product::<usize>()).unwrap(),
        )
        .unwrap();
        for degenerate in [false, true] {
            if degenerate {
                points[1].fill(0.);
            }
            let vertices: Vec<&[f64]> = points.iter().map(Vec::as_slice).collect();
            for _ in 0..4 {
                assert_eq!(
                    workspace.orient(&vertices).unwrap(),
                    if degenerate {
                        PredicateSign::Zero
                    } else {
                        PredicateSign::Positive
                    }
                );
            }
        }
    }
}

#[test]
fn opaque_structured_traversals_execute() {
    use scioxide_spatial::{
        AxisSide, AxisTopology, CellMesh, CoordinateAxis, Index, StructuredGrid,
    };
    fn check<const D: usize>() {
        let grid = StructuredGrid::new(
            core::array::from_fn(|_| CoordinateAxis::uniform(0.0_f64, 2.0, 3).unwrap()),
            [AxisTopology::Periodic; D],
        )
        .unwrap();
        let mesh = CellMesh::new(grid).unwrap();
        let cell = Index::new([0; D]).unwrap();
        {
            for coordinate in mesh.grid().axis(0).unwrap().coordinates() {
                std::hint::black_box(coordinate.unwrap());
            }
            std::hint::black_box(mesh.grid().cell_neighbors(cell).unwrap().count());
            std::hint::black_box(mesh.cell_neighbors(cell).unwrap().count());
            for vertex in mesh.cell_vertices(cell).unwrap() {
                std::hint::black_box(mesh.vertex_cells(vertex).unwrap().count());
            }
            for face in mesh.cell_faces(cell).unwrap() {
                std::hint::black_box(mesh.face_cells(face).unwrap().count());
            }
            std::hint::black_box(mesh.axis_boundary_faces(0, AxisSide::End).unwrap().count());
        };
    }
    check::<1>();
    check::<2>();
    check::<3>();
}
