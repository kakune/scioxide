//! Exhaustive ownership oracle and deterministic work scaling, never timings.
#![allow(clippy::unwrap_used)]
use super::*;
use alloc::vec;
use scioxide_core::InvalidInputReason;
use scioxide_core::SciOxideErrorView;

fn limits() -> AdaptationLimits {
    AdaptationLimits::new()
        .with_max_depth(12)
        .with_max_cells(200_000)
        .with_max_entities(1_000_000)
        .with_max_work(usize::MAX)
        .with_max_storage_bytes(1024 * 1024 * 1024)
}

fn work() -> Work<'static, 'static, 'static> {
    Work::new(limits(), None, scioxide_core::argument!("owners")).unwrap()
}

#[test]
fn sorted_part_ranges_borrow_storage_with_logarithmic_search_work() {
    let rows: Vec<_> = (0..8192).map(|cell| (2 * (cell / 4) + 1, cell)).collect();
    for part in [0, 1, 2, 2049, 4095, 4096, usize::MAX] {
        let comparisons = core::cell::Cell::new(0);
        let range = sorted_range(&rows, part, |&(owner, _)| {
            comparisons.set(comparisons.get() + 1);
            owner
        });
        let expected: Vec<_> = rows.iter().filter(|&&(p, _)| p == part).copied().collect();
        assert_eq!(range, expected);
        let start = rows
            .iter()
            .position(|&(p, _)| p >= part)
            .unwrap_or(rows.len());
        assert!(core::ptr::eq(range.as_ptr(), rows[start..].as_ptr()));
        assert!(comparisons.get() <= 2 * (rows.len().ilog2() + 2));
    }
    assert!(sorted_range::<(usize, usize), _>(&[], 0, |&(p, _)| p).is_empty());
    let moves = [(0, 2, 1), (1, 0, 3), (1, 0, 4), (1, 2, 5), (2, 0, 7)];
    let range = sorted_range(&moves, (1, 0), |&(a, b, _)| (a, b));
    assert_eq!(range, &moves[1..3]);
    assert!(core::ptr::eq(range.as_ptr(), moves[1..].as_ptr()));
    assert!(sorted_range(&moves, (1, 1), |&(a, b, _)| (a, b)).is_empty());
}

#[test]
fn repartition_sort_consumes_the_work_budget() {
    let mesh = tensor::<f64, 1>([12], false);
    let mut source = MeshPartition::balanced(&mesh, 3).unwrap();
    let destination = MeshPartition::new(
        &mesh,
        4,
        &(0..mesh.cell_count())
            .map(|cell| cell % 4)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    // Preserve construction order: the comparison charge depends on the unsorted input.
    let mut moves: Vec<_> = destination
        .cell_order()
        .iter()
        .filter_map(|&cell| {
            let a = source.owners()[cell];
            let b = destination.owners()[cell];
            (a != b).then_some((a, b, cell))
        })
        .collect();
    let mut measured = work();
    measured.sort(&mut moves).unwrap();
    assert!(measured.operations > 0);
    let required = mesh.cell_count() + measured.operations;
    source.limits = source.limits.with_max_work(required - 1);
    let error = RepartitionMap::new(&source, &destination).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ResourceLimit { resource, required: actual, limit }
            if resource == scioxide_core::quantity!("adaptive_work")
                && actual == required && limit == required - 1
    ));
    source.limits = source.limits.with_max_work(required);
    let map = RepartitionMap::new(&source, &destination).unwrap();
    assert_eq!(map.moves, moves);
    for from in 0..source.partition_count() {
        for to in 0..destination.partition_count() {
            let expected: Vec<_> = (0..mesh.cell_count())
                .filter(|&cell| {
                    from != to && source.owners()[cell] == from && destination.owners()[cell] == to
                })
                .collect();
            assert_eq!(
                map.migration_cells(from, to).unwrap().collect::<Vec<_>>(),
                expected
            );
        }
    }
}

fn tensor<S: RealScalar, const D: usize>(shape: [usize; D], periodic: bool) -> AdaptiveMesh<S, D> {
    let grid = crate::StructuredGrid::new(
        core::array::from_fn(|axis| {
            crate::CoordinateAxis::uniform(S::ONE, S::ZERO, shape[axis] + 1).unwrap()
        }),
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
    ids: &[usize],
) -> AdaptiveMesh<S, D> {
    mesh.adapt_scoped(
        AdaptationRequest {
            source: mesh.mesh_token(),
            refine: ids,
            coarsen: &[],
        },
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap()
    .mesh
}

// Deliberately search every cell and interface for each entity. This oracle
// does not call incident_edges or reuse the production ownership traversal.
fn assert_reference<S: RealScalar, const D: usize>(
    mesh: &AdaptiveMesh<S, D>,
    partition: &MeshPartition,
) {
    let storage = &mesh.storage;
    let owners = partition.owners();
    let interior = || {
        mesh.interfaces().iter().filter_map(|interface| {
            interface.right.map(|right| {
                (
                    interface,
                    owners[interface.left_cell].min(owners[right.cell]),
                )
            })
        })
    };
    for (index, &vertex) in storage.active_vertices.iter().enumerate() {
        let mut expected = storage
            .active
            .iter()
            .zip(owners)
            .filter(|(id, _)| {
                storage.nodes[**id].vertices[..storage.vertex_arity()].contains(&vertex)
            })
            .map(|(_, &owner)| owner)
            .min()
            .unwrap_or(partition.parts);
        for (interface, owner) in interior() {
            let piece_id = storage.active_faces[interface.piece_face];
            let piece = storage.faces[piece_id];
            for side in [interface.left_face, interface.right.unwrap().face] {
                let side_id = storage.active_faces[side];
                let face = storage.faces[side_id];
                if face.vertices[..usize::from(face.len)].contains(&vertex)
                    && piece.vertices[..usize::from(piece.len)].iter().any(|&v| {
                        storage.face_vertex_key(side_id, vertex)
                            == storage.face_vertex_key(piece_id, v)
                    })
                {
                    expected = expected.min(owner);
                }
            }
        }
        assert_eq!(
            partition.vertex_owner(index).unwrap(),
            expected,
            "vertex {index}"
        );
    }
    for (index, &face) in storage.active_faces.iter().enumerate() {
        let mut expected = storage
            .active
            .iter()
            .zip(owners)
            .filter(|(id, _)| storage.nodes[**id].faces[..storage.face_arity()].contains(&face))
            .map(|(_, &owner)| owner)
            .min()
            .unwrap_or(partition.parts);
        for (interface, owner) in interior() {
            if [
                interface.left_face,
                interface.piece_face,
                interface.right.unwrap().face,
            ]
            .contains(&index)
            {
                expected = expected.min(owner);
            }
        }
        assert_eq!(
            partition.face_owner(index).unwrap(),
            expected,
            "face {index}"
        );
    }
    for (index, edge) in storage.edges.iter().enumerate() {
        let mut expected = storage
            .active
            .iter()
            .zip(owners)
            .filter(|(id, _)| {
                edge.iter()
                    .all(|v| storage.nodes[**id].vertices[..storage.vertex_arity()].contains(v))
            })
            .map(|(_, &owner)| owner)
            .min()
            .unwrap_or(partition.parts);
        for (interface, owner) in interior() {
            let piece_id = storage.active_faces[interface.piece_face];
            let piece = storage.faces[piece_id];
            for side in [interface.left_face, interface.right.unwrap().face] {
                let side_id = storage.active_faces[side];
                let face = storage.faces[side_id];
                if !edge
                    .iter()
                    .all(|v| face.vertices[..usize::from(face.len)].contains(v))
                {
                    continue;
                }
                let endpoints = edge.map(|v| storage.face_vertex_key(side_id, v));
                if storage.edges.iter().any(|fine| {
                    fine.iter()
                        .all(|v| piece.vertices[..usize::from(piece.len)].contains(v))
                        && fine.iter().all(|&v| {
                            topology::on_edge(
                                storage.face_vertex_key(piece_id, v),
                                endpoints[0],
                                endpoints[1],
                            )
                        })
                }) {
                    expected = expected.min(owner);
                }
            }
        }
        assert_eq!(
            partition.edge_owner(index).unwrap(),
            expected,
            "edge {index}"
        );
    }
    let mut order = Vec::new();
    for part in 0..partition.parts {
        let owned: Vec<_> = owners
            .iter()
            .enumerate()
            .filter_map(|(cell, &owner)| (owner == part).then_some(cell))
            .collect();
        assert_eq!(partition.owned_cells(part).unwrap(), owned);
        for (local, &cell) in owned.iter().enumerate() {
            assert_eq!(partition.local_cell(cell).unwrap(), (part, local));
        }
        order.extend(owned);
        let mut ghosts = Vec::new();
        let mut interfaces = Vec::new();
        for (index, interface) in mesh.interfaces().iter().enumerate() {
            if let Some(right_side) = interface.right {
                let right = right_side.cell;
                let left = interface.left_cell;
                if owners[left] != owners[right] && [owners[left], owners[right]].contains(&part) {
                    interfaces.push(index);
                    ghosts.push(if owners[left] == part { right } else { left });
                }
            }
        }
        ghosts.sort_unstable();
        ghosts.dedup();
        assert_eq!(
            partition.ghost_cells(part).unwrap().collect::<Vec<_>>(),
            ghosts
        );
        assert_eq!(
            partition.interfaces(part).unwrap().collect::<Vec<_>>(),
            interfaces
        );
    }
    assert_eq!(partition.cell_order(), order);
    assert_eq!(partition.mesh_token(), mesh.mesh_token());
}

fn draw(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn assignments<S: RealScalar, const D: usize>(mesh: &AdaptiveMesh<S, D>, state: &mut u64) {
    for parts in [1, 3, mesh.cell_count() + 2] {
        let owners: Vec<_> = (0..mesh.cell_count())
            .map(|_| draw(state) as usize % parts)
            .collect();
        let partition = MeshPartition::new(mesh, parts, &owners).unwrap();
        assert_eq!(partition.owners(), owners);
        assert_reference(mesh, &partition);
        let balanced = MeshPartition::balanced(mesh, parts).unwrap();
        let expected: Vec<_> = (0..parts)
            .flat_map(|part| {
                core::iter::repeat_n(
                    part,
                    mesh.cell_count() / parts + usize::from(part < mesh.cell_count() % parts),
                )
            })
            .collect();
        assert_eq!(balanced.owners(), expected);
        assert_reference(mesh, &balanced);
    }
}

fn sequences<S: RealScalar, const D: usize>(root: AdaptiveMesh<S, D>) {
    let first_id = root.active_cells()[0];
    let first = refine(&root, &[first_id]);
    if D > 1 && root.cell_count() > 1 {
        assert!(!first.hanging_entities().is_empty());
    }
    if root.interfaces().iter().any(|i| i.periodic) {
        assert!(first.interfaces().iter().any(|i| i.periodic));
    }
    let back = first
        .adapt_scoped(
            AdaptationRequest {
                source: first.mesh_token(),
                refine: &[],
                coarsen: &[first_id],
            },
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap()
        .mesh;
    assert_eq!(back.active_cells(), root.active_cells());
    assignments(&back, &mut 7);
    for seed in [17, 103, 509] {
        let mut state = seed;
        let mut mesh = first.clone();
        assignments(&root, &mut state);
        for _ in 0..12 {
            assignments(&mesh, &mut state);
            let refine: Vec<_> = mesh
                .active_cells()
                .iter()
                .copied()
                .filter(|&id| {
                    mesh.cell_count() < 24
                        && mesh.level(id).unwrap() < 3
                        && draw(&mut state).is_multiple_of(5)
                })
                .take(1)
                .collect();
            let coarsen: Vec<_> = mesh
                .storage
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(id, node)| {
                    (!node.active
                        && node.child_count > 0
                        && node.children[..usize::from(node.child_count)]
                            .iter()
                            .all(|child| {
                                mesh.storage.nodes[*child].active && !refine.contains(child)
                            })
                        && draw(&mut state).is_multiple_of(3))
                    .then_some(id)
                })
                .collect();
            if let Ok(next) = mesh.adapt_scoped(
                AdaptationRequest {
                    source: mesh.mesh_token(),
                    refine: &refine,
                    coarsen: &coarsen,
                },
                scioxide_core::ExecutionContext::sequential(),
            ) {
                mesh = next.mesh;
            }
        }
        assignments(&mesh, &mut state);
    }
}

fn families<S: RealScalar>() {
    for periodic in [false, true] {
        sequences(tensor::<S, 1>([2], periodic));
        sequences(tensor::<S, 2>([2; 2], periodic));
        sequences(tensor::<S, 3>([2; 3], periodic));
    }
    sequences(tensor::<S, 2>([1; 2], true));
    sequences(tensor::<S, 3>([1; 3], true));
    let z = S::ZERO;
    let o = S::ONE;
    let triangles = crate::TriangleMesh::new(
        [[z, z], [o, z], [o, o], [z, o]]
            .map(|p| Point::new(p).unwrap())
            .to_vec(),
        vec![[0, 1, 2], [0, 2, 3]],
    )
    .unwrap();
    sequences(AdaptiveMesh::from_simplex(&triangles, limits()).unwrap());
    let tetrahedra = crate::TetrahedronMesh::new(
        [[z, z, z], [o, z, z], [z, o, z], [z, z, o], [z, z, -o]]
            .map(|p| Point::new(p).unwrap())
            .to_vec(),
        vec![[0, 1, 2, 3], [0, 2, 1, 4]],
    )
    .unwrap();
    sequences(AdaptiveMesh::from_simplex(&tetrahedra, limits()).unwrap());
}

#[test]
fn randomized_ownership_matches_exhaustive_f32() {
    families::<f32>();
}

#[test]
fn randomized_ownership_matches_exhaustive_f64() {
    families::<f64>();
}

fn scaling<const D: usize>(shapes: &[[usize; D]], hanging: bool) {
    let mut previous = None;
    for &shape in shapes {
        let root = tensor::<f64, D>(shape, hanging);
        let marks: Vec<_> = root
            .active_cells()
            .iter()
            .copied()
            .filter(|&id| {
                let mut rest = id;
                let mut parity = 0;
                for size in shape {
                    parity += rest % size;
                    rest /= size;
                }
                !hanging || parity % 2 == 0
            })
            .collect();
        let mesh = refine(&root, &marks);
        if hanging {
            assert!(!mesh.hanging_entities().is_empty());
            assert!(mesh.interfaces().iter().any(|i| i.periodic));
        }
        let balanced = MeshPartition::balanced(&mesh, 8).unwrap();
        let alternating: Vec<_> = (0..mesh.cell_count()).map(|cell| cell % 8).collect();
        let mut counts = [0; 2];
        for (count, owners) in counts.iter_mut().zip([balanced.owners(), &alternating]) {
            let mut work = work();
            MeshPartition::build(&mesh, 8, owners, &mut work).unwrap();
            *count = work.operations;
        }
        if let Some((cells, earlier)) = previous {
            assert_eq!(mesh.cell_count(), 4 * cells);
            for (current, before) in counts.into_iter().zip(earlier) {
                assert!(
                    current <= 5 * before,
                    "4x cells must cost <= 5x work: shape={shape:?}, hanging={hanging}, before={before}, after={current}"
                );
            }
        }
        previous = Some((mesh.cell_count(), counts));
    }
}

#[test]
fn uniform_partition_work_is_near_linear() {
    scaling(&[[16, 16], [32, 32], [64, 64]], false);
    scaling(&[[4, 4, 4], [8, 8, 4], [8, 16, 8]], false);
}

#[test]
fn hanging_periodic_partition_work_is_near_linear() {
    scaling(&[[8, 8], [16, 16], [32, 32]], true);
    scaling(&[[4, 4, 4], [8, 8, 4]], true);
}

fn assert_invalid(error: SciOxideError, expected: Argument, reason: InvalidInputReason) {
    assert!(
        matches!(error.view(), SciOxideErrorView::InvalidInput { argument, reason: actual, .. }
        if argument == expected && actual == reason),
        "{error:?}"
    );
}

#[test]
fn partition_limits_and_invalid_assignments_are_transactional() {
    let mesh = refine(&tensor::<f64, 2>([2; 2], true), &[0]);
    let owners: Vec<_> = (0..mesh.cell_count()).map(|cell| cell % 3).collect();
    let original = MeshPartition::new(&mesh, 3, &owners).unwrap();
    let token = mesh.mesh_token().clone();
    let mut measured = work();
    MeshPartition::build(&mesh, 3, &owners, &mut measured).unwrap();
    // Inject ceilings into this private fixture to isolate partition accounting
    // from the independent mesh-import and adaptation costs.
    let capped = |limits| AdaptiveMesh {
        limits,
        ..mesh.clone()
    };
    let exact = limits().with_max_work(measured.operations);
    let exact_mesh = capped(exact);
    let partition = MeshPartition::new(&exact_mesh, 3, &owners).unwrap();
    assert_eq!(partition.limits(), exact);
    assert_reference(&mesh, &partition);
    for max_work in [0, measured.operations / 2, measured.operations - 1] {
        let error =
            MeshPartition::new(&capped(limits().with_max_work(max_work)), 3, &owners).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ResourceLimit { .. }
        ));
    }
    let balanced = MeshPartition::balanced(&mesh, 3).unwrap();
    let mut measured = work();
    MeshPartition::build(&mesh, 3, balanced.owners(), &mut measured).unwrap();
    let balanced_work = measured.operations + mesh.cell_count();
    assert_reference(
        &mesh,
        &MeshPartition::balanced(&capped(limits().with_max_work(balanced_work)), 3).unwrap(),
    );
    assert!(matches!(
        MeshPartition::balanced(&capped(limits().with_max_work(balanced_work - 1)), 3)
            .unwrap_err()
            .view(),
        SciOxideErrorView::ResourceLimit { .. }
    ));
    for bytes in [0, core::mem::size_of::<usize>() * owners.len()] {
        assert!(matches!(
            MeshPartition::new(&capped(limits().with_max_storage_bytes(bytes)), 3, &owners)
                .unwrap_err()
                .view(),
            SciOxideErrorView::ResourceLimit { .. }
        ));
    }
    for bad in [&owners[..owners.len() - 1], &[3][..]] {
        assert_invalid(
            MeshPartition::new(&mesh, 3, bad).unwrap_err(),
            scioxide_core::argument!("owners"),
            InvalidInputReason::Inconsistent,
        );
    }
    let mut out_of_range = owners.clone();
    out_of_range[0] = 3;
    assert_invalid(
        MeshPartition::new(&mesh, 3, &out_of_range).unwrap_err(),
        scioxide_core::argument!("owners"),
        InvalidInputReason::Inconsistent,
    );
    assert_invalid(
        MeshPartition::new(&mesh, 0, &owners).unwrap_err(),
        scioxide_core::argument!("parts"),
        InvalidInputReason::Inconsistent,
    );
    assert_invalid(
        MeshPartition::balanced(&mesh, 0).unwrap_err(),
        scioxide_core::argument!("parts"),
        InvalidInputReason::Inconsistent,
    );
    assert_eq!(mesh.mesh_token(), &token);
    assert_reference(&mesh, &original);
    let empty = crate::TriangleMesh::<f64>::new(vec![], vec![]).unwrap();
    let empty = AdaptiveMesh::from_simplex(&empty, limits()).unwrap();
    assignments(&empty, &mut 11);
    assert!(original.vertex_owner(mesh.vertex_count()).is_err());
    assert!(original.face_owner(mesh.face_count()).is_err());
    assert!(original.edge_owner(mesh.edge_count()).is_err());
    assert!(original.owned_cells(3).is_err());
    assert!(original.ghost_cells(3).is_err());
    assert!(original.interfaces(3).is_err());
}
