//! Identity requests account only for their output and share the source epoch.
#![allow(clippy::unwrap_used)]
use super::*;
use scioxide_core::{ExecutionContext, SciOxideErrorView};

fn work(limits: AdaptationLimits) -> Work<'static, 'static, 'static> {
    Work::new(limits, None, scioxide_core::argument!("request")).unwrap()
}

fn assert_identity<S: RealScalar, const D: usize>(
    source: &AdaptiveMesh<S, D>,
    result: &AdaptationResult<S, D>,
) {
    assert!(Arc::ptr_eq(&source.storage, &result.mesh.storage));
    assert_eq!(source.mesh_token(), result.mesh.mesh_token());
    assert_eq!(source.limits(), result.mesh.limits());
    assert!(result.induced_refinements.is_empty());
    result
        .map
        .validate(source.mesh_token(), source.mesh_token())
        .unwrap();
    let mut relations = 0;
    for (rows, count) in [
        (result.map.cells(), source.cell_count()),
        (result.map.vertices(), source.vertex_count()),
        (result.map.faces(), source.face_count()),
        (result.map.edges(), source.edge_count()),
    ] {
        assert_eq!(rows.len(), count);
        for (index, relation) in rows.iter().enumerate() {
            assert_eq!(
                *relation,
                EntityRelation::Unchanged {
                    source: index,
                    destination: index,
                }
            );
        }
        relations += count;
    }
    assert_eq!(result.work, relations + 1);
}

#[test]
fn identity_needs_no_candidate_storage_or_index_work() {
    let grid = crate::Grid2d::new(
        core::array::from_fn(|_| crate::CoordinateAxis::uniform(0., 1., 5).unwrap()),
        [AxisTopology::Periodic; 2],
    )
    .unwrap();
    let root =
        AdaptiveMesh::from_structured(&CellMesh::new(grid).unwrap(), AdaptationLimits::new())
            .unwrap();
    let refined = root
        .adapt_scoped(
            AdaptationRequest {
                source: root.mesh_token(),
                refine: &[0],
                coarsen: &[],
            },
            ExecutionContext::sequential(),
        )
        .unwrap()
        .mesh;
    // Cover both a root and an epoch retaining inactive ancestors and hanging faces.
    for mesh in [root, refined] {
        let request = AdaptationRequest {
            source: mesh.mesh_token(),
            refine: &[],
            coarsen: &[],
        };
        let mut measured = work(mesh.limits());
        let identity = mesh.adapt_impl(request, &mut measured).unwrap();
        assert_identity(&mesh, &identity);
        let capped = AdaptiveMesh {
            limits: mesh
                .limits()
                .with_max_work(identity.work)
                .with_max_storage_bytes(measured.bytes),
            ..mesh.clone()
        };
        let mut copy_work = work(capped.limits().with_max_work(usize::MAX));
        assert!(matches!(
            capped.storage.copy(&mut copy_work).unwrap_err().view(),
            SciOxideErrorView::ResourceLimit { resource, .. }
                if resource == scioxide_core::quantity!("adaptive_bytes")
        ));
        let result = capped
            .adapt_scoped(request, ExecutionContext::sequential())
            .unwrap();
        assert_identity(&capped, &result);
        for limits in [
            capped.limits().with_max_work(identity.work - 1),
            capped.limits().with_max_storage_bytes(measured.bytes - 1),
        ] {
            let restricted = AdaptiveMesh {
                limits,
                ..mesh.clone()
            };
            assert!(matches!(
                restricted
                    .adapt_scoped(request, ExecutionContext::sequential())
                    .unwrap_err()
                    .view(),
                SciOxideErrorView::ResourceLimit { .. }
            ));
        }
    }
}

#[test]
fn empty_mesh_identity_needs_only_publication_work() {
    let mesh = crate::TriangleMesh::<f32>::new(Vec::new(), Vec::new()).unwrap();
    let mesh = AdaptiveMesh::from_simplex(&mesh, AdaptationLimits::new()).unwrap();
    let mesh = AdaptiveMesh {
        limits: mesh.limits().with_max_work(1).with_max_storage_bytes(0),
        ..mesh
    };
    let result = mesh
        .adapt_scoped(
            AdaptationRequest {
                source: mesh.mesh_token(),
                refine: &[],
                coarsen: &[],
            },
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_identity(&mesh, &result);
}
