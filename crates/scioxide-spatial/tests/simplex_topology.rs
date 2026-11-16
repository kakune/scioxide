#![allow(clippy::unwrap_used)]

use scioxide_core::{NumericalIssue, RealScalar};
use scioxide_spatial::{
    IntervalMesh, Point, SimplexBoundaryTags, SimplexMesh, TetrahedronMesh, TriangleMesh,
};

fn points<S: RealScalar, const D: usize>(values: &[[f64; D]]) -> Vec<Point<S, D>> {
    values
        .iter()
        .map(|p| {
            Point::new(core::array::from_fn(|i| {
                S::try_from_f64_lossy(p[i]).unwrap()
            }))
            .unwrap()
        })
        .collect()
}
fn square<S: RealScalar>() -> TriangleMesh<S> {
    TriangleMesh::new(
        points(&[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        vec![[0, 1, 2], [0, 2, 3]],
    )
    .unwrap()
}
fn tetra<S: RealScalar>() -> TetrahedronMesh<S> {
    TetrahedronMesh::new(
        points(&[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        vec![[0, 1, 2, 3]],
    )
    .unwrap()
}
fn close<S: RealScalar>(actual: S, expected: f64) {
    let error = (actual.try_to_f64_exact().unwrap() - expected).abs();
    assert!(
        error <= 16. * S::EPSILON.try_to_f64_exact().unwrap() * expected.abs().max(1.),
        "{actual:?} != {expected}"
    );
}
fn incidence<S: RealScalar, const D: usize>(mesh: &SimplexMesh<S, D>) {
    for cell in 0..mesh.cell_count() {
        for face in mesh.cell_faces(cell).unwrap() {
            assert!(mesh.face_cells(face.index()).unwrap().contains(&cell));
        }
        for edge in mesh.cell_edges(cell).unwrap() {
            assert!(mesh.edge_cells(edge.index()).unwrap().contains(&cell));
        }
        for &vertex in mesh.cell_vertices(cell).unwrap() {
            assert!(mesh.vertex_cells(vertex).unwrap().contains(&cell));
        }
        for neighbor in mesh.cell_neighbors(cell).unwrap() {
            assert!(mesh.cell_neighbors(neighbor).unwrap().any(|n| n == cell));
        }
        if D == 3 {
            let mut boundary = vec![0_i32; mesh.edge_count()];
            for face in mesh.cell_faces(cell).unwrap() {
                for edge in mesh.face_edges(face.index()).unwrap() {
                    boundary[edge.index()] += i32::from(face.sign()) * i32::from(edge.sign());
                }
            }
            assert!(boundary.iter().all(|&v| v == 0));
        } else {
            let mut boundary = vec![0_i32; mesh.vertex_count()];
            for face in mesh.cell_faces(cell).unwrap() {
                let vertices = mesh.face_vertices(face.index()).unwrap();
                boundary[vertices[0]] -= i32::from(face.sign());
                boundary[vertices[1]] += i32::from(face.sign());
            }
            assert!(boundary.iter().all(|&v| v == 0));
        }
    }
    for face in 0..mesh.face_count() {
        let cells = mesh.face_cells(face).unwrap();
        if cells.len() == 2 {
            let signs: i32 = cells
                .iter()
                .map(|&cell| {
                    i32::from(
                        mesh.cell_faces(cell)
                            .unwrap()
                            .iter()
                            .find(|f| f.index() == face)
                            .unwrap()
                            .sign(),
                    )
                })
                .sum();
            assert_eq!(signs, 0);
        }
    }
}
fn basic<S: RealScalar>() {
    let mesh = square::<S>();
    assert_eq!(
        (
            mesh.vertex_count(),
            mesh.cell_count(),
            mesh.face_count(),
            mesh.edge_count()
        ),
        (4, 2, 5, 5)
    );
    assert_eq!(mesh.boundary_faces().count(), 4);
    assert_eq!(mesh.cell_neighbors(0).unwrap().collect::<Vec<_>>(), vec![1]);
    assert_eq!(
        mesh.cells().collect::<Vec<_>>(),
        vec![&[0, 1, 2][..], &[0, 2, 3][..]]
    );
    close(mesh.cell_measure(0).unwrap(), 0.5);
    for edge in 0..mesh.edge_count() {
        let vertices = mesh.edge_vertices(edge).unwrap();
        close(
            mesh.edge_length(edge).unwrap(),
            if *vertices == [0, 2] {
                2_f64.sqrt()
            } else {
                1.
            },
        );
        close(
            mesh.face_measure(edge).unwrap(),
            mesh.edge_length(edge).unwrap().try_to_f64_exact().unwrap(),
        );
        assert_eq!(mesh.face_vertices(edge).unwrap(), vertices);
    }
    incidence(&mesh);
    let mesh = tetra::<S>();
    assert_eq!(
        (
            mesh.vertex_count(),
            mesh.cell_count(),
            mesh.face_count(),
            mesh.edge_count()
        ),
        (4, 1, 4, 6)
    );
    close(mesh.cell_measure(0).unwrap(), 1. / 6.);
    let area: f64 = (0..mesh.face_count())
        .map(|f| mesh.face_measure(f).unwrap().try_to_f64_exact().unwrap())
        .sum();
    assert!(
        (area - (1.5 + 3_f64.sqrt() / 2.)).abs() < 16. * S::EPSILON.try_to_f64_exact().unwrap()
    );
    incidence(&mesh);
}
fn intervals<S: RealScalar>() {
    // Cells listed out of coordinate order; vertex 3 sits between 0 and 1.
    let mesh = IntervalMesh::<S>::new(
        points(&[[0.], [1.], [0.25], [0.5]]),
        vec![[3, 1], [0, 2], [2, 3]],
    )
    .unwrap();
    assert_eq!(
        (
            mesh.vertex_count(),
            mesh.cell_count(),
            mesh.face_count(),
            mesh.edge_count()
        ),
        (4, 3, 4, 3)
    );
    let boundary: Vec<_> = mesh
        .boundary_faces()
        .map(|face| mesh.face_vertices(face).unwrap()[0])
        .collect();
    assert_eq!(boundary, vec![0, 1]);
    for (cell, expected) in [0.5, 0.25, 0.25].into_iter().enumerate() {
        close(mesh.cell_measure(cell).unwrap(), expected);
        let edge = mesh.cell_edges(cell).unwrap()[0];
        assert_eq!(edge.sign(), if cell == 0 { -1 } else { 1 });
        close(mesh.edge_length(edge.index()).unwrap(), expected);
    }
    for face in 0..mesh.face_count() {
        close(mesh.face_measure(face).unwrap(), 1.);
        assert!(mesh.face_edges(face).unwrap().is_empty());
        let cells = mesh.face_cells(face).unwrap();
        if cells.len() == 2 {
            // One interval closes and the other opens at an interior vertex.
            let signs: i32 = cells
                .iter()
                .map(|&cell| {
                    i32::from(
                        mesh.cell_faces(cell)
                            .unwrap()
                            .iter()
                            .find(|f| f.index() == face)
                            .unwrap()
                            .sign(),
                    )
                })
                .sum();
            assert_eq!(signs, 0);
        }
    }
    // The face opposite local vertex 0 is the cell end (positive boundary sign).
    let faces = mesh.cell_faces(1).unwrap();
    assert_eq!(mesh.face_vertices(faces[0].index()).unwrap(), &[2]);
    assert_eq!(faces[0].sign(), 1);
    assert_eq!(faces[1].sign(), -1);
    let tags =
        SimplexBoundaryTags::new(&mesh, mesh.boundary_faces().map(|face| (face, 7)).collect())
            .unwrap();
    assert_eq!(tags.len(), 2);
    let interior = (0..mesh.face_count())
        .find(|&face| !mesh.is_boundary_face(face).unwrap())
        .unwrap();
    assert!(SimplexBoundaryTags::new(&mesh, vec![(interior, 1)]).is_err());

    let line =
        |cells: Vec<[usize; 2]>| IntervalMesh::<S>::new(points(&[[0.], [1.], [2.], [3.]]), cells);
    assert!(line(vec![[1, 0], [1, 2], [2, 3]]).is_err(), "inverted");
    assert!(
        line(vec![[0, 1], [1, 1], [2, 3]]).is_err(),
        "repeated vertex"
    );
    assert!(
        line(vec![[0, 1], [0, 1], [1, 2], [2, 3]]).is_err(),
        "duplicate"
    );
    assert!(
        line(vec![[0, 1], [0, 2], [2, 3]]).is_err(),
        "branching start"
    );
    assert!(line(vec![[0, 2], [1, 2], [2, 3]]).is_err(), "branching end");
    assert!(line(vec![[0, 1], [1, 2]]).is_err(), "unused vertex");
    assert!(line(vec![[0, 4], [1, 2], [2, 3]]).is_err(), "invalid index");
    assert!(
        IntervalMesh::<S>::new(points(&[[0.], [0.], [1.]]), vec![[0, 1], [1, 2]]).is_err(),
        "degenerate"
    );
    let renumbered = mesh.with_cells(vec![[3, 1], [0, 2], [2, 3]]).unwrap();
    assert_eq!(renumbered.mesh_token(), mesh.mesh_token());
}

#[test]
fn interval_meshes_are_one_dimensional_simplices() {
    intervals::<f32>();
    intervals::<f64>();
}

#[test]
fn simplex_geometry_and_incidence_both_widths() {
    basic::<f32>();
    basic::<f64>();
}

fn cube<S: RealScalar>() -> TetrahedronMesh<S> {
    let vertices: Vec<_> = (0..8)
        .map(|i| {
            Point::new(core::array::from_fn(|axis| {
                S::try_from_usize_exact((i >> axis) & 1).unwrap()
            }))
            .unwrap()
        })
        .collect();
    // Six equally oriented tetrahedra around the body diagonal (0,7).
    TetrahedronMesh::new(
        vertices,
        vec![
            [0, 1, 3, 7],
            [0, 5, 1, 7],
            [0, 3, 2, 7],
            [0, 2, 6, 7],
            [0, 4, 5, 7],
            [0, 6, 4, 7],
        ],
    )
    .unwrap()
}
#[test]
fn tetrahedral_cube_has_manifold_links_and_unit_volume() {
    for mesh in [
        cube::<f64>(),
        cube::<f64>()
            .with_cells(vec![
                [0, 6, 4, 7],
                [0, 4, 5, 7],
                [0, 2, 6, 7],
                [0, 3, 2, 7],
                [0, 5, 1, 7],
                [0, 1, 3, 7],
            ])
            .unwrap(),
    ] {
        incidence(&mesh);
        assert_eq!(mesh.boundary_faces().count(), 12);
        assert_eq!((mesh.face_count(), mesh.edge_count()), (18, 19));
        close(
            (0..6)
                .map(|cell| mesh.cell_measure(cell).unwrap())
                .sum::<f64>(),
            1.,
        );
    }
    incidence(&cube::<f32>());
}
#[test]
fn empty_and_disconnected_meshes() {
    let empty = TriangleMesh::<f32>::new(vec![], vec![]).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.boundary_faces().count(), 0);
    assert!(
        TetrahedronMesh::<f64>::new(vec![], vec![])
            .unwrap()
            .is_empty()
    );
    let disconnected = TriangleMesh::new(
        points::<f64, 2>(&[[0., 0.], [1., 0.], [0., 1.], [2., 0.], [3., 0.], [2., 1.]]),
        vec![[0, 1, 2], [3, 4, 5]],
    )
    .unwrap();
    assert_eq!(disconnected.boundary_faces().count(), 6);
    incidence(&disconnected);
}
#[test]
fn reject_invalid_cells_and_unused_vertices() {
    let p = points::<f64, 2>(&[[0., 0.], [1., 0.], [0., 1.]]);
    for cells in [
        vec![[0, 2, 1]],
        vec![[0, 1, 1]],
        vec![[0, 1, 2], [1, 2, 0]],
        vec![],
    ] {
        assert!(
            matches ! (& (TriangleMesh :: new (p . clone () , cells)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: InvalidInput { .. }))
        );
    }
    assert!(
        matches ! (& (TriangleMesh :: new (p , vec ! [[0 , 1 , 3]])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: OutOfBounds { .. }))
    );
    assert!(
        TriangleMesh::new(
            points::<f32, 2>(&[[0., 0.], [1., 1.], [2., 2.]]),
            vec![[0, 1, 2]]
        )
        .is_err()
    );
    let p = tetra::<f64>().points().to_vec();
    assert!(TetrahedronMesh::new(p, vec![[0, 2, 1, 3]]).is_err());
    assert!(
        TetrahedronMesh::new(
            points::<f64, 3>(&[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]),
            vec![[0, 1, 2, 3]]
        )
        .is_err()
    );
    assert!(
        matches ! (& (Point :: new ([f64 :: NAN , 0.])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NonFinite { .. }))
    );
}
#[test]
fn reject_nonmanifold_faces_and_pinched_links() {
    // Three triangles on one edge, and two triangles meeting only at a vertex.
    for (p, c) in [
        (
            vec![[0., 0.], [1., 0.], [0., 1.], [0., -1.], [1., 1.]],
            vec![[0, 1, 2], [1, 0, 3], [0, 1, 4]],
        ),
        (
            vec![[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.]],
            vec![[0, 1, 2], [0, 3, 4]],
        ),
    ] {
        assert!(TriangleMesh::new(points::<f64, 2>(&p), c).is_err());
    }
    // Two tetrahedra meeting only at an edge.
    let p = points::<f64, 3>(&[
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., -1., 0.],
        [0., 0., -1.],
    ]);
    assert!(TetrahedronMesh::new(p, vec![[0, 1, 2, 3], [0, 1, 4, 5]]).is_err());
    // Two tetrahedra meeting only at a vertex.
    let p = points::<f64, 3>(&[
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [-1., 0., 0.],
        [0., -1., 0.],
        [0., 0., -1.],
    ]);
    assert!(TetrahedronMesh::new(p, vec![[0, 1, 2, 3], [0, 5, 4, 6]]).is_err());
}
#[test]
fn canonical_entities_do_not_depend_on_cell_order() {
    let a = square::<f64>();
    let b = a.with_cells(vec![[2, 3, 0], [1, 2, 0]]).unwrap();
    for face in 0..a.face_count() {
        assert_eq!(
            a.face_vertices(face).unwrap(),
            b.face_vertices(face).unwrap()
        );
    }
    for edge in 0..a.edge_count() {
        assert_eq!(
            a.edge_vertices(edge).unwrap(),
            b.edge_vertices(edge).unwrap()
        );
    }
}
#[test]
fn revisions_are_exact_including_sibling_branches() {
    let mesh = square::<f64>();
    assert_eq!(mesh.mesh_token(), mesh.clone().mesh_token());
    assert_ne!(mesh.mesh_token(), square::<f64>().mesh_token());
    assert_eq!(
        mesh.mesh_token(),
        mesh.with_points(mesh.points().to_vec())
            .unwrap()
            .mesh_token()
    );
    assert_eq!(
        mesh.mesh_token(),
        mesh.with_cells(vec![[0, 1, 2], [0, 2, 3]])
            .unwrap()
            .mesh_token()
    );
    let a = mesh
        .with_points(points(&[[0., 0.], [2., 0.], [2., 2.], [0., 2.]]))
        .unwrap();
    let b = mesh
        .with_points(points(&[[0., 0.], [3., 0.], [3., 3.], [0., 3.]]))
        .unwrap();
    assert!(a.mesh_token().same_family(b.mesh_token()));
    assert_eq!(a.mesh_token().generation(), b.mesh_token().generation());
    assert_ne!(a.mesh_token(), b.mesh_token());
}
#[test]
fn checked_queries_reject_invalid_ids() {
    let mesh = square::<f64>();
    assert!(mesh.cell_vertices(usize::MAX).is_err());
    assert!(mesh.cell_faces(2).is_err());
    assert!(mesh.cell_edges(2).is_err());
    assert!(mesh.face_cells(5).is_err());
    assert!(mesh.face_edges(5).is_err());
    assert!(mesh.edge_cells(5).is_err());
    assert!(mesh.vertex_cells(4).is_err());
    assert!(mesh.cell_neighbors(2).is_err());
}
#[test]
fn topology_survives_measure_range_limits_and_near_degeneracy() {
    for scale in [f64::from_bits(1), f64::MAX] {
        let mesh = TriangleMesh::new(
            points::<f64, 2>(&[[0., 0.], [scale, 0.], [0., scale]]),
            vec![[0, 1, 2]],
        )
        .unwrap();
        assert!(
            matches ! (& (mesh . cell_measure (0)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NumericalInstability { issue : NumericalIssue :: Underflow | NumericalIssue :: Overflow , .. }))
        );
    }
    let tiny = f64::from_bits(1);
    let mesh = TriangleMesh::new(
        points::<f64, 2>(&[[0., 0.], [2., 0.], [0., tiny]]),
        vec![[0, 1, 2]],
    )
    .unwrap();
    assert_eq!(mesh.cell_measure(0).unwrap(), tiny);
    let large = 2_f64.powi(500);
    let next = f64::from_bits(large.to_bits() + 1);
    let mesh = TriangleMesh::new(
        points::<f64, 2>(&[[0., 0.], [large, large], [large, next]]),
        vec![[0, 1, 2]],
    )
    .unwrap();
    assert_eq!(mesh.cell_measure(0).unwrap(), 2_f64.powi(947));
}

#[test]
fn interior_vertex_links_and_concurrent_queries() {
    let mesh = TetrahedronMesh::new(
        points::<f64, 3>(&[
            [0., 0., 0.],
            [4., 0., 0.],
            [0., 4., 0.],
            [0., 0., 4.],
            [1., 1., 1.],
        ]),
        vec![[4, 1, 2, 3], [0, 4, 2, 3], [0, 1, 4, 3], [0, 1, 2, 4]],
    )
    .unwrap();
    incidence(&mesh);
    assert_eq!(mesh.vertex_cells(4).unwrap(), &[0, 1, 2, 3]);
    assert_eq!(mesh.boundary_faces().count(), 4);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    (0..mesh.cell_count())
                        .map(|cell| mesh.cell_measure(cell).unwrap())
                        .sum::<f64>()
                })
            })
            .collect();
        for worker in workers {
            close(worker.join().unwrap(), 64. / 6.);
        }
    });
    let triangles = TriangleMesh::new(
        points::<f32, 2>(&[[0., 0.], [2., 0.], [2., 2.], [0., 2.], [1., 1.]]),
        vec![[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
    )
    .unwrap();
    incidence(&triangles);
    assert_eq!(triangles.vertex_cells(4).unwrap(), &[0, 1, 2, 3]);
}

fn measure_range_errors<S: RealScalar>() {
    let large = S::MAX;
    let zero = S::ZERO;
    let mesh = TriangleMesh::new(
        [[zero, zero], [large, zero], [zero, large]]
            .map(|p| Point::new(p).unwrap())
            .to_vec(),
        vec![[0, 1, 2]],
    )
    .unwrap();
    assert!(
        matches ! (& (mesh . cell_measure (0)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NumericalInstability { issue : NumericalIssue :: Overflow , .. }))
    );
    assert!(
        matches ! (& (mesh . edge_length (2)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NumericalInstability { issue : NumericalIssue :: Overflow , .. }))
    );
    let tiny = S::MIN_POSITIVE * S::EPSILON;
    let mesh = TriangleMesh::new(
        [[zero, zero], [tiny, zero], [zero, tiny]]
            .map(|p| Point::new(p).unwrap())
            .to_vec(),
        vec![[0, 1, 2]],
    )
    .unwrap();
    assert!(
        matches ! (& (mesh . cell_measure (0)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NumericalInstability { issue : NumericalIssue :: Underflow , .. }))
    );
}
#[test]
fn measure_range_errors_are_numerical_in_both_widths() {
    measure_range_errors::<f32>();
    measure_range_errors::<f64>();
}

// Moving vertices reuses connectivity and rechecks only exact orientation.
#[test]
fn with_points_reuses_topology_and_rechecks_orientation() {
    let points = |shift: f64| {
        vec![
            Point::new([0.0_f64, 0.0]).unwrap(),
            Point::new([1.0 + shift, 0.0]).unwrap(),
            Point::new([0.0, 1.0]).unwrap(),
            Point::new([1.0, 1.0 + shift]).unwrap(),
        ]
    };
    let mesh = TriangleMesh::new(points(0.0), vec![[0, 1, 2], [1, 3, 2]]).unwrap();
    let moved = mesh.with_points(points(0.5)).unwrap();
    assert_eq!(moved.face_count(), mesh.face_count());
    assert_eq!(moved.edge_count(), mesh.edge_count());
    for cell in 0..mesh.cell_count() {
        assert_eq!(
            moved.cell_faces(cell).unwrap(),
            mesh.cell_faces(cell).unwrap()
        );
    }
    assert_ne!(moved.mesh_token(), mesh.mesh_token());
    assert!(moved.cell_measure(0).unwrap() > mesh.cell_measure(0).unwrap());
    let mut inverted = points(0.0);
    inverted[2] = Point::new([0.0, -1.0]).unwrap();
    assert_eq!(
        mesh.with_points(inverted).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
    let mut flat = points(0.0);
    flat[2] = Point::new([2.0, 0.0]).unwrap();
    assert_eq!(
        mesh.with_points(flat).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
}

// Documented contract: validation is combinatorial and orientational. A fan
// wound twice around its center is locally valid and is accepted.
#[test]
fn folded_fans_pass_local_validation() {
    let mut points = vec![Point::new([0.0_f64, 0.0]).unwrap()];
    for k in 0..7 {
        let angle = f64::from(k) * 4.0 * core::f64::consts::PI / 7.0;
        points.push(Point::new([angle.cos(), angle.sin()]).unwrap());
    }
    let cells: Vec<[usize; 3]> = (0..7).map(|k| [0, 1 + k, 1 + (k + 1) % 7]).collect();
    let mesh = TriangleMesh::new(points, cells).unwrap();
    let total: f64 = (0..mesh.cell_count())
        .map(|cell| mesh.cell_measure(cell).unwrap())
        .sum();
    // A regular heptagon inscribed in the unit circle has area about 2.736.
    let heptagon = 3.5 * (2.0 * core::f64::consts::PI / 7.0).sin();
    assert!(total > heptagon);
}
