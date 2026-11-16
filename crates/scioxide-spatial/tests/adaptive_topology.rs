#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex32, Complex64, RealScalar};
use scioxide_spatial::*;
fn limits() -> AdaptationLimits {
    AdaptationLimits::new()
        .with_max_depth(10)
        .with_max_cells(10000)
        .with_max_entities(100000)
        .with_max_work(100_000_000)
        .with_max_storage_bytes(256 * 1024 * 1024)
}
fn tensor<S: RealScalar, const D: usize>(periodic: bool) -> AdaptiveMesh<S, D> {
    let axes = core::array::from_fn(|_| CoordinateAxis::uniform(S::ZERO, S::ONE, 3).unwrap());
    let grid = StructuredGrid::new(
        axes,
        [if periodic {
            AxisTopology::Periodic
        } else {
            AxisTopology::Bounded
        }; D],
    )
    .unwrap();
    AdaptiveMesh::from_structured(&CellMesh::new(grid).unwrap(), limits()).unwrap()
}
fn refine<S: RealScalar, const D: usize>(
    mesh: &AdaptiveMesh<S, D>,
    marks: &[usize],
) -> AdaptationResult<S, D> {
    mesh.adapt_scoped(
        AdaptationRequest {
            source: mesh.mesh_token(),
            refine: marks,
            coarsen: &[],
        },
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap()
}
fn coarsen<S: RealScalar, const D: usize>(
    mesh: &AdaptiveMesh<S, D>,
    marks: &[usize],
) -> AdaptationResult<S, D> {
    mesh.adapt_scoped(
        AdaptationRequest {
            source: mesh.mesh_token(),
            refine: &[],
            coarsen: marks,
        },
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap()
}
fn tensor_roundtrip<S: RealScalar, const D: usize>() {
    let root = tensor::<S, D>(false);
    let count = root.cell_count();
    let result = refine(&root, &[0]);
    assert_eq!(result.mesh.cell_count(), count + (1 << D) - 1);
    assert!(
        result
            .map
            .cells()
            .iter()
            .any(|r| matches!(r, EntityRelation::Refined { .. }))
    );
    let back = coarsen(&result.mesh, &[0]);
    assert_eq!(back.mesh.cell_count(), count);
    assert_eq!(root.interfaces(), back.mesh.interfaces());
    assert_ne!(root.mesh_token(), back.mesh.mesh_token());
    for i in 0..root.vertex_count() {
        assert_eq!(root.point(i).unwrap(), back.mesh.point(i).unwrap());
    }
    let repeated = refine(&back.mesh, &[0]);
    assert_eq!(result.mesh.active_cells(), repeated.mesh.active_cells());
}

fn measures<S: RealScalar, const D: usize>(mesh: AdaptiveMesh<S, D>) {
    let initial = (0..mesh.cell_count()).fold(S::ZERO, |s, i| s + mesh.cell_measure(i).unwrap());
    let fine = refine(&mesh, &[0]);
    let volume = (0..fine.mesh.cell_count()).fold(S::ZERO, |s, i| {
        let v = fine.mesh.cell_measure(i).unwrap();
        assert!(v > S::ZERO);
        assert!(fine.mesh.cell_diameter(i).unwrap() > S::ZERO);
        s + v
    });
    assert!((volume - initial).abs() <= S::EPSILON * S::try_from_f64_lossy(32.).unwrap() * initial);
    for i in fine.mesh.interfaces() {
        assert!(fine.mesh.face_measure(i.piece_face).unwrap() > S::ZERO);
    }
    assert!(mesh.cell_measure(mesh.cell_count()).is_err());
    assert!(mesh.face_measure(mesh.face_count()).is_err());
}

#[test]
fn adaptive_physical_measures_both_widths_all_families() {
    measures(tensor::<f32, 1>(false));
    measures(tensor::<f64, 1>(false));
    measures(tensor::<f32, 2>(false));
    measures(tensor::<f64, 2>(false));
    measures(tensor::<f32, 3>(false));
    measures(tensor::<f64, 3>(false));
    measures(triangles::<f32>());
    measures(triangles::<f64>());
    measures(tetrahedra::<f32>());
    measures(tetrahedra::<f64>());
}
#[test]
fn structured_roundtrips_both_widths_all_dimensions() {
    tensor_roundtrip::<f32, 1>();
    tensor_roundtrip::<f64, 1>();
    tensor_roundtrip::<f32, 2>();
    tensor_roundtrip::<f64, 2>();
    tensor_roundtrip::<f32, 3>();
    tensor_roundtrip::<f64, 3>();
}
fn triangles<S: RealScalar>() -> AdaptiveMesh<S, 2> {
    triangles_with_limits(limits())
}
fn triangles_with_limits<S: RealScalar>(limits: AdaptationLimits) -> AdaptiveMesh<S, 2> {
    let z = S::ZERO;
    let o = S::ONE;
    let mesh = TriangleMesh::new(
        [[z, z], [o, z], [o, o], [z, o]]
            .map(|p| Point::new(p).unwrap())
            .to_vec(),
        vec![[0, 1, 2], [0, 2, 3]],
    )
    .unwrap();
    AdaptiveMesh::from_simplex(&mesh, limits).unwrap()
}
fn tetrahedra<S: RealScalar>() -> AdaptiveMesh<S, 3> {
    let z = S::ZERO;
    let o = S::ONE;
    let mesh = TetrahedronMesh::new(
        [[z, z, z], [o, z, z], [z, o, z], [z, z, o]]
            .map(|p| Point::new(p).unwrap())
            .to_vec(),
        vec![[0, 1, 2, 3]],
    )
    .unwrap();
    AdaptiveMesh::from_simplex(&mesh, limits()).unwrap()
}
fn simplex_roundtrip<S: RealScalar, const D: usize>(root: AdaptiveMesh<S, D>) {
    let result = refine(&root, &[0]);
    assert_eq!(result.mesh.cell_count(), root.cell_count() + (1 << D) - 1);
    let back = coarsen(&result.mesh, &[0]);
    assert_eq!(root.interfaces(), back.mesh.interfaces());
    for &child in result.mesh.children(0).unwrap() {
        assert_eq!(result.mesh.parent(child).unwrap(), Some(0));
        assert_eq!(
            result.mesh.child_reference_vertices(child).unwrap().len(),
            D + 1
        );
    }
}
#[test]
fn simplex_roundtrips_both_widths() {
    simplex_roundtrip(triangles::<f32>());
    simplex_roundtrip(triangles::<f64>());
    simplex_roundtrip(tetrahedra::<f32>());
    simplex_roundtrip(tetrahedra::<f64>());
}
#[test]
fn hanging_interfaces_and_automatic_balance() {
    let root = triangles::<f64>();
    let first = refine(&root, &[0]);
    assert!(
        first
            .mesh
            .hanging_entities()
            .iter()
            .any(|e| e.kind == HangingEntityKind::Vertex)
    );
    // Refine all children so every root interface includes level-two descendants.
    let second = refine(&first.mesh, first.mesh.children(0).unwrap());
    assert!(second.induced_refinements.contains(&1));
    for face in second.mesh.interfaces() {
        if let Some(right_side) = face.right {
            let right = right_side.cell;
            let a = second
                .mesh
                .level(second.mesh.active_cells()[face.left_cell])
                .unwrap();
            let b = second
                .mesh
                .level(second.mesh.active_cells()[right])
                .unwrap();
            assert!(a.abs_diff(b) <= 1);
            assert_eq!(face.left_sign, -face.right_sign().unwrap());
        }
    }
    assert!(
        second
            .mesh
            .adapt_scoped(
                AdaptationRequest {
                    source: second.mesh.mesh_token(),
                    refine: &[],
                    coarsen: &[1]
                },
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
    );
}
#[test]
fn failure_identity_and_mark_order() {
    let root = tensor::<f64, 2>(false);
    let a = refine(&root, &[0, 1]);
    let b = refine(&root, &[1, 0]);
    assert_eq!(a.mesh.interfaces(), b.mesh.interfaces());
    assert_ne!(a.mesh.mesh_token(), b.mesh.mesh_token());
    assert!(
        a.map
            .validate(root.mesh_token(), b.mesh.mesh_token())
            .is_err()
    );
    let grid = CellMesh::new(
        Grid2d::new(
            core::array::from_fn(|_| CoordinateAxis::uniform(0., 1., 3).unwrap()),
            [AxisTopology::Bounded; 2],
        )
        .unwrap(),
    )
    .unwrap();
    for bad in [
        limits().with_max_work(0),
        limits().with_max_storage_bytes(0),
        limits().with_max_depth(0),
        limits().with_max_cells(root.cell_count()),
    ] {
        let result = AdaptiveMesh::from_structured(&grid, bad).and_then(|mesh| {
            mesh.adapt_scoped(
                AdaptationRequest {
                    source: mesh.mesh_token(),
                    refine: &[0],
                    coarsen: &[],
                },
                scioxide_core::ExecutionContext::sequential(),
            )
        });
        assert!(matches!(
            result.unwrap_err().view(),
            scioxide_core::SciOxideErrorView::ResourceLimit { .. }
        ));
    }
    assert!(
        root.adapt_scoped(
            AdaptationRequest {
                source: root.mesh_token(),
                refine: &[0, 0],
                coarsen: &[]
            },
            scioxide_core::ExecutionContext::sequential()
        )
        .is_err()
    );
    let unchanged = refine(&root, &[]);
    assert_eq!(root.mesh_token(), unchanged.mesh.mesh_token());
}
#[test]
fn periodic_interfaces_survive_refinement() {
    let root = tensor::<f64, 2>(true);
    assert!(root.interfaces().iter().all(|f| f.right.is_some()));
    let first = refine(&root, &[0]);
    assert!(first.mesh.interfaces().iter().all(|f| f.right.is_some()));
    assert!(first.mesh.interfaces().iter().any(|f| f.periodic));
    let axes = [CoordinateAxis::uniform(1_f32, -1., 2).unwrap(); 1];
    let one = CellMesh::new(StructuredGrid::new(axes, [AxisTopology::Periodic]).unwrap()).unwrap();
    let one = AdaptiveMesh::from_structured(&one, limits()).unwrap();
    let two = refine(&one, &[0]);
    assert!(two.mesh.interfaces().iter().all(|f| f.right.is_some()));
}
#[test]
fn fields_and_repartition_roundtrip() {
    let root = refine(&triangles::<f64>(), &[0]).mesh;
    let vertices = VertexField::<Complex32, 2>::from_fn(&root, |_| Complex32::new(1., 2.)).unwrap();
    let cells =
        CellField::<Complex64, 2>::from_fn(&root, |i| Complex64::new(i as f64, -1.)).unwrap();
    let faces = Field::<f32, UnstructuredFace, 2>::from_fn(&root, |i| i as f32).unwrap();
    let edges = Field::<f64, UnstructuredEdge, 2>::from_fn(&root, |i| i as f64).unwrap();
    assert_eq!(vertices.len(), root.vertex_count());
    assert_eq!(faces.len(), root.face_count());
    assert_eq!(edges.len(), root.edge_count());
    let view = FieldView::<_, Cell, 2>::new(&root, cells.values()).unwrap();
    assert_eq!(view.values(), cells.values());
    let a = MeshPartition::balanced(&root, 2).unwrap();
    let b = MeshPartition::new(&root, 3, &[2, 0, 1, 2, 0]).unwrap();
    let ab =
        RepartitionMap::with_options(&a, &b, scioxide_spatial::RepartitionOptions::new()).unwrap();
    let ba =
        RepartitionMap::with_options(&b, &a, scioxide_spatial::RepartitionOptions::new()).unwrap();
    let input = a
        .cell_order()
        .iter()
        .map(|&i| cells.values()[i])
        .collect::<Vec<_>>();
    let mut output = input.clone();
    let mut scratch = input.clone();
    ab.gather_into(
        &input,
        &mut output,
        &mut scratch,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    for (value, &global) in output.iter().zip(b.cell_order()) {
        assert_eq!(*value, cells.values()[global]);
    }
    let mut restored = input.clone();
    ba.gather_into(
        &output,
        &mut restored,
        &mut scratch,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(restored, input);
    for part in 0..2 {
        for ghost in a.ghost_cells(part).unwrap() {
            assert_ne!(a.owners()[ghost], part);
        }
    }
    assert!(
        ab.validate(&a, &MeshPartition::new(&root, 3, b.owners()).unwrap())
            .is_err()
    );
}
#[test]
fn empty_mesh_and_empty_partitions() {
    let mesh = TriangleMesh::<f32>::new(vec![], vec![]).unwrap();
    let mesh = AdaptiveMesh::from_simplex(&mesh, limits()).unwrap();
    assert_eq!(refine(&mesh, &[]).mesh.cell_count(), 0);
    let p = MeshPartition::balanced(&mesh, 3).unwrap();
    assert!(p.owned_cells(2).unwrap().is_empty());
    assert!(MeshPartition::balanced(&mesh, 0).is_err());
}

#[test]
fn rounding_collapse_and_extreme_coordinates_are_transactional() {
    fn check<S: RealScalar>() {
        let one = S::ONE;
        let zero = S::ZERO;
        let next = one + S::EPSILON;
        let input = TriangleMesh::new(
            [[one, zero], [next, zero], [one, one]]
                .map(|p| Point::new(p).unwrap())
                .to_vec(),
            vec![[0, 1, 2]],
        )
        .unwrap();
        let mesh = AdaptiveMesh::from_simplex(&input, limits()).unwrap();
        let token = mesh.mesh_token().clone();
        assert!(
            mesh.adapt_scoped(
                AdaptationRequest {
                    source: mesh.mesh_token(),
                    refine: &[0],
                    coarsen: &[]
                },
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(mesh.mesh_token(), &token);
        assert_eq!(mesh.cell_count(), 1);
        let big = S::MAX / S::try_from_usize_exact(4).unwrap();
        let input = TriangleMesh::new(
            [[-big, zero], [big, zero], [zero, big]]
                .map(|p| Point::new(p).unwrap())
                .to_vec(),
            vec![[0, 1, 2]],
        )
        .unwrap();
        let mesh = AdaptiveMesh::from_simplex(&input, limits()).unwrap();
        assert_eq!(refine(&mesh, &[0]).mesh.cell_count(), 4);
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn repeated_hex_refinement_and_single_root_periodicity() {
    let grid = Grid3d::new(
        core::array::from_fn(|_| CoordinateAxis::uniform(0., 1., 2).unwrap()),
        [AxisTopology::Periodic; 3],
    )
    .unwrap();
    let mesh = AdaptiveMesh::from_structured(&CellMesh::new(grid).unwrap(), limits()).unwrap();
    let first = refine(&mesh, &[0]).mesh;
    assert!(first.interfaces().iter().all(|i| i.right.is_some()));
    let child = first.active_cells()[0];
    let second = refine(&first, &[child]).mesh;
    assert!(second.interfaces().iter().all(|i| i.right.is_some()));
    let third = refine(&second, second.children(child).unwrap()).mesh;
    assert!(third.interfaces().iter().all(|i| i.right.is_some()));
    for interface in third.interfaces() {
        let a = third.active_cells()[interface.left_cell];
        let b = third.active_cells()[interface.right.unwrap().cell];
        assert!(third.level(a).unwrap().abs_diff(third.level(b).unwrap()) <= 1);
    }
}

#[test]
fn periodic_root_corners_are_not_hanging_vertices() {
    let root = tensor::<f64, 2>(true);
    let mesh = refine(&root, &[0]).mesh;
    for hanging in mesh.hanging_entities() {
        if hanging.kind == HangingEntityKind::Vertex {
            assert!(mesh.vertex_provenance(hanging.entity).unwrap().1 > 0);
        }
    }
}
#[test]
fn tetrahedron_neighbors_balance_and_keep_oriented_maps() {
    let input = TetrahedronMesh::new(
        [
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., -1.],
        ]
        .map(|p| Point::new(p).unwrap())
        .to_vec(),
        vec![[0, 1, 2, 3], [0, 2, 1, 4]],
    )
    .unwrap();
    let root = AdaptiveMesh::from_simplex(&input, limits()).unwrap();
    let first = refine(&root, &[0]);
    let second = refine(&first.mesh, first.mesh.children(0).unwrap());
    assert!(second.induced_refinements.contains(&1));
    for relation in first.map.edges() {
        if let EntityRelation::Refined {
            source,
            destination,
            orientation,
        } = *relation
        {
            let a = root
                .edge_vertices(source)
                .unwrap()
                .map(|v| root.point(v).unwrap().into_array());
            let b = first
                .mesh
                .edge_vertices(destination)
                .unwrap()
                .map(|v| first.mesh.point(v).unwrap().into_array());
            let dot = (0..3)
                .map(|axis| (a[1][axis] - a[0][axis]) * (b[1][axis] - b[0][axis]))
                .sum::<f64>();
            assert_eq!(orientation, if dot > 0. { 1 } else { -1 });
        }
    }
    for relation in first.map.faces() {
        if let EntityRelation::Refined {
            source,
            destination,
            orientation,
        } = *relation
        {
            let normal = |mesh: &AdaptiveMesh<f64, 3>, face| {
                let p = mesh
                    .face_vertices(face)
                    .unwrap()
                    .map(|v| mesh.point(v).unwrap().into_array())
                    .collect::<Vec<_>>();
                let a = core::array::from_fn::<_, 3, _>(|i| p[1][i] - p[0][i]);
                let b = core::array::from_fn::<_, 3, _>(|i| p[2][i] - p[0][i]);
                [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ]
            };
            let a = normal(&root, source);
            let b = normal(&first.mesh, destination);
            let dot = a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
            assert_eq!(orientation, if dot > 0. { 1 } else { -1 });
        }
    }
}

#[test]
fn result_depth_and_migration_queries_enforce_bounds() {
    let root = triangles_with_limits::<f64>(limits().with_max_depth(1));
    let mesh = refine(&root, &[0]).mesh;
    assert_eq!(mesh.limits(), root.limits());
    assert!(matches!(
        mesh.adapt_scoped(
            AdaptationRequest {
                source: mesh.mesh_token(),
                refine: mesh.children(0).unwrap(),
                coarsen: &[],
            },
            scioxide_core::ExecutionContext::sequential()
        )
        .unwrap_err()
        .view(),
        scioxide_core::SciOxideErrorView::ResourceLimit { .. }
    ));
    assert_eq!(coarsen(&mesh, &[0]).mesh.cell_count(), root.cell_count());
    let a = MeshPartition::balanced(&mesh, 2).unwrap();
    let b = MeshPartition::new(&mesh, 2, &[1, 0, 1, 0, 1]).unwrap();
    let map =
        RepartitionMap::with_options(&a, &b, scioxide_spatial::RepartitionOptions::new()).unwrap();
    assert!(map.migration_cells(2, 0).is_err());
    assert!(map.migration_cells(0, 2).is_err());
    for from in 0..2 {
        for to in 0..2 {
            let expected = (0..mesh.cell_count())
                .filter(|&c| a.owners()[c] == from && b.owners()[c] == to && from != to)
                .collect::<Vec<_>>();
            assert_eq!(
                map.migration_cells(from, to).unwrap().collect::<Vec<_>>(),
                expected
            );
        }
    }
}

#[test]
fn rounded_midpoints_cannot_hide_a_folded_reference_child() {
    fn check<S: RealScalar>() {
        let e = S::EPSILON;
        let one = S::ONE;
        let zero = S::ZERO;
        let root = TriangleMesh::new(
            [[zero, zero], [one, one], [one + e, one + e + e]]
                .map(|p| Point::new(p).unwrap())
                .to_vec(),
            vec![[0, 1, 2]],
        )
        .unwrap();
        let mesh = AdaptiveMesh::from_simplex(&root, limits()).unwrap();
        let token = mesh.mesh_token().clone();
        let result = mesh.adapt_scoped(
            AdaptationRequest {
                source: mesh.mesh_token(),
                refine: &[0],
                coarsen: &[],
            },
            scioxide_core::ExecutionContext::sequential(),
        );
        assert!(
            matches ! (& (result) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: NumericalInstability { .. }))
        );
        assert_eq!(mesh.mesh_token(), &token);
        assert_eq!(mesh.cell_count(), 1);
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn interface_right_sides_couple_incidence_and_derive_signs() {
    fn check<const D: usize>(mesh: &AdaptiveMesh<f64, D>) {
        for interface in mesh.interfaces() {
            assert!(matches!(interface.left_sign, -1 | 1));
            assert!(
                mesh.cell_faces(interface.left_cell)
                    .unwrap()
                    .any(|(face, _)| face == interface.left_face)
            );
            match interface.right {
                Some(right) => {
                    assert!(
                        mesh.cell_faces(right.cell)
                            .unwrap()
                            .any(|(face, _)| face == right.face)
                    );
                    assert_eq!(interface.right_sign(), Some(-interface.left_sign));
                    assert!(
                        interface.piece_face == interface.left_face
                            || interface.piece_face == right.face
                    );
                }
                None => {
                    assert_eq!(interface.right_sign(), None);
                    assert_eq!(interface.piece_face, interface.left_face);
                    assert!(!interface.periodic);
                }
            }
        }
    }
    for root in [
        tensor::<f64, 2>(false),
        tensor::<f64, 2>(true),
        triangles::<f64>(),
    ] {
        check(&root);
        check(&refine(&root, &[0]).mesh);
    }
    for root in [
        tensor::<f64, 3>(false),
        tensor::<f64, 3>(true),
        tetrahedra::<f64>(),
    ] {
        check(&root);
        check(&refine(&root, &[0]).mesh);
    }
}

// Signed face incidence must be physically outward for every axis orientation.
// Normals come from the published cyclic face order: the clockwise segment
// normal in 2D and the cross product of the first two edges in 3D.
fn outward_counts<const D: usize>(mesh: &AdaptiveMesh<f64, D>) -> (usize, usize) {
    let (mut outward, mut inward) = (0, 0);
    for cell in 0..mesh.cell_count() {
        let vertices: Vec<_> = mesh.cell_vertices(cell).unwrap().collect();
        let mut centroid = [0.0; D];
        for &vertex in &vertices {
            for (c, x) in centroid
                .iter_mut()
                .zip(mesh.point(vertex).unwrap().as_array())
            {
                *c += x / vertices.len() as f64;
            }
        }
        for (face, sign) in mesh.cell_faces(cell).unwrap() {
            let points: Vec<[f64; D]> = mesh
                .face_vertices(face)
                .unwrap()
                .map(|v| *mesh.point(v).unwrap().as_array())
                .collect();
            let mut normal = [0.0; D];
            if D == 2 {
                normal[0] = points[1][1] - points[0][1];
                normal[1] = points[0][0] - points[1][0];
            } else {
                let a: Vec<f64> = (0..3).map(|i| points[1][i] - points[0][i]).collect();
                let b: Vec<f64> = (0..3).map(|i| points[2][i] - points[1][i]).collect();
                normal[0] = a[1] * b[2] - a[2] * b[1];
                normal[1] = a[2] * b[0] - a[0] * b[2];
                normal[2] = a[0] * b[1] - a[1] * b[0];
            }
            let away: f64 = (0..D)
                .map(|i| normal[i] * (points[0][i] - centroid[i]))
                .sum();
            if f64::from(sign) * away > 0.0 {
                outward += 1;
            } else {
                inward += 1;
            }
        }
    }
    (outward, inward)
}
fn oriented<const D: usize>(decreasing: [bool; D]) -> AdaptiveMesh<f64, D> {
    let axes = decreasing.map(|reverse| {
        if reverse {
            CoordinateAxis::uniform(1.0, 0.0, 3).unwrap()
        } else {
            CoordinateAxis::uniform(0.0, 1.0, 3).unwrap()
        }
    });
    let grid = StructuredGrid::new(axes, [AxisTopology::Bounded; D]).unwrap();
    AdaptiveMesh::from_structured(&CellMesh::new(grid).unwrap(), limits()).unwrap()
}
#[test]
fn tensor_incidence_is_outward_for_reflected_grids() {
    assert_eq!(outward_counts(&oriented([false; 2])).1, 0);
    for decreasing in [[true, false], [false, true], [true, true]] {
        let mesh = oriented(decreasing);
        for mesh in [refine(&mesh, &[0]).mesh, mesh] {
            let (outward, inward) = outward_counts(&mesh);
            assert_eq!((outward > 0, inward), (true, 0), "{decreasing:?}");
        }
    }
    assert_eq!(outward_counts(&oriented([false; 3])).1, 0);
    for decreasing in [
        [true, false, false],
        [true, true, false],
        [true, true, true],
    ] {
        let mesh = oriented(decreasing);
        for mesh in [refine(&mesh, &[0]).mesh, mesh] {
            let (outward, inward) = outward_counts(&mesh);
            assert_eq!((outward > 0, inward), (true, 0), "{decreasing:?}");
        }
    }
}

#[test]
fn tensor_measure_reports_numerical_underflow() {
    let grid = scioxide_spatial::CellMesh::uniform([0.0_f64; 2], [1e-200; 2], [2; 2]).unwrap();
    let mesh = AdaptiveMesh::from_structured(&grid, limits()).unwrap();
    let error = mesh.cell_measure(0).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Underflow,
            ..
        }
    ));
}

#[test]
fn tensor_area_can_survive_an_overflowing_axis_span() {
    let grid =
        scioxide_spatial::CellMesh::uniform([-f64::MAX, 0.0], [f64::MAX, 1e-308], [2; 2]).unwrap();
    let mesh = AdaptiveMesh::from_structured(&grid, limits()).unwrap();
    let area = mesh.cell_measure(0).unwrap();
    let expected = (f64::MAX * 1e-308) * 2.0;
    assert!((area / expected - 1.0).abs() < 1e-14);
}

#[test]
fn interval_simplex_meshes_are_rejected_and_segments_refine() {
    let points = vec![Point::new([0.0_f64]).unwrap(), Point::new([1.0]).unwrap()];
    let intervals = IntervalMesh::new(points, vec![[0, 1]]).unwrap();
    let error = AdaptiveMesh::from_simplex(&intervals, limits()).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::InvalidInput {
            reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
            ..
        }
    ));
    let segments = scioxide_spatial::CellMesh::uniform([0.0_f64], [1.0], [2]).unwrap();
    let mesh = AdaptiveMesh::from_structured(&segments, limits()).unwrap();
    let refined = mesh
        .adapt(AdaptationRequest {
            source: mesh.mesh_token(),
            refine: &[0],
            coarsen: &[],
        })
        .unwrap();
    assert_eq!(refined.mesh.cell_count(), 2);
    assert!((refined.mesh.cell_measure(0).unwrap() - 0.5).abs() < 1e-15);
}
