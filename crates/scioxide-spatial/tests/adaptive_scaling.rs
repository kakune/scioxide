//! Deterministic operation counts, never wall-clock performance gates.
#![allow(clippy::unwrap_used)]
use scioxide_spatial::*;

fn limits() -> AdaptationLimits {
    AdaptationLimits::new()
        .with_max_depth(12)
        .with_max_cells(200_000)
        .with_max_entities(1_000_000)
        .with_max_work(usize::MAX)
        .with_max_storage_bytes(1024 * 1024 * 1024)
}

fn mesh<const D: usize>(shape: [usize; D]) -> AdaptiveMesh<f64, D> {
    let grid = StructuredGrid::new(
        core::array::from_fn(|axis| CoordinateAxis::uniform(0., 1., shape[axis] + 1).unwrap()),
        [AxisTopology::Bounded; D],
    )
    .unwrap();
    AdaptiveMesh::from_structured(&CellMesh::new(grid).unwrap(), limits()).unwrap()
}

fn refine<const D: usize>(mesh: &AdaptiveMesh<f64, D>, ids: &[usize]) -> AdaptationResult<f64, D> {
    mesh.adapt_scoped(
        AdaptationRequest {
            source: mesh.mesh_token(),
            refine: ids,
            coarsen: &[],
        },
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap()
}

fn uniform<const D: usize>(shape: [usize; D]) -> usize {
    let mesh = mesh(shape);
    let result = refine(&mesh, mesh.active_cells());
    assert_eq!(result.mesh.cell_count(), mesh.cell_count() * (1 << D));
    assert!(result.induced_refinements.is_empty());
    result.work
}

#[test]
fn empty_requests_cost_only_linear_identity_work_and_less_than_refinement() {
    fn check<const D: usize>(shape: [usize; D]) {
        let mesh = mesh(shape);
        let identity = refine(&mesh, &[]);
        let refined = refine(&mesh, &mesh.active_cells()[..1]);
        let entities =
            mesh.cell_count() + mesh.vertex_count() + mesh.face_count() + mesh.edge_count();
        assert_eq!(identity.work, entities + 1);
        assert!(identity.work < refined.work);
        assert_eq!(identity.mesh.mesh_token(), mesh.mesh_token());
    }
    check([16]);
    check([4, 4]);
    check([2, 2, 2]);
}

fn closure<const D: usize>(shape: [usize; D]) -> usize {
    let root = mesh(shape);
    let marked: Vec<_> = root
        .active_cells()
        .iter()
        .copied()
        .filter(|&id| {
            let mut remainder = id;
            let mut parity = 0;
            for size in shape {
                parity += remainder % size;
                remainder /= size;
            }
            parity % 2 == 0
        })
        .collect();
    let first = refine(&root, &marked).mesh;
    let children: Vec<_> = marked
        .iter()
        .flat_map(|&id| first.children(id).unwrap().iter().copied())
        .collect();
    let second = refine(&first, &children);
    assert_eq!(
        second.induced_refinements.len(),
        root.cell_count() - marked.len()
    );
    assert!(
        second
            .induced_refinements
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    );
    for interface in second.mesh.interfaces() {
        if let Some(right_side) = interface.right {
            let right = right_side.cell;
            let a = second.mesh.active_cells()[interface.left_cell];
            let b = second.mesh.active_cells()[right];
            assert!(
                second
                    .mesh
                    .level(a)
                    .unwrap()
                    .abs_diff(second.mesh.level(b).unwrap())
                    <= 1
            );
        }
    }
    second.work
}

fn scaling<const D: usize>(shapes: &[[usize; D]], run: impl Fn([usize; D]) -> usize) {
    let work: Vec<_> = shapes.iter().copied().map(run).collect();
    for (sizes, counts) in shapes.windows(2).zip(work.windows(2)) {
        assert_eq!(
            sizes[1].iter().product::<usize>(),
            4 * sizes[0].iter().product::<usize>()
        );
        assert!(
            counts[1] <= 5 * counts[0],
            "fourfold growth must cost <= 5x: shapes={sizes:?}, work={counts:?}"
        );
    }
}

#[test]
fn uniform_quad_work_is_near_linear() {
    scaling(&[[16, 16], [32, 32], [64, 64]], uniform);
}

#[test]
fn uniform_hex_work_is_near_linear() {
    scaling(&[[4, 4, 4], [8, 8, 4], [8, 16, 8]], uniform);
}

#[test]
fn many_induced_splits_are_near_linear() {
    scaling(&[[8, 8], [16, 16], [32, 32]], closure);
    scaling(&[[4, 4, 4], [8, 8, 4]], closure);
}
