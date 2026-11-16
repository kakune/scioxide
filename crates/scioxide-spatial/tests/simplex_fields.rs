#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex32, Complex64};
use scioxide_spatial::{
    CellField, FieldView, Point, SimplexBoundaryTags, TriangleMesh, UnstructuredEdgeField,
    UnstructuredFace, UnstructuredFaceField, VertexField,
};

fn square() -> TriangleMesh<f64> {
    TriangleMesh::new(
        [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
            .map(|p| Point::new(p).unwrap())
            .to_vec(),
        vec![[0, 1, 2], [0, 2, 3]],
    )
    .unwrap()
}
#[test]
fn owned_and_borrowed_locations_support_real_and_complex_values() {
    let mesh = square();
    let vertices =
        VertexField::<Complex32, 2>::from_fn(&mesh, |_| Complex32::new(1., -2.)).unwrap();
    let cells = CellField::<Complex64, 2>::new(&mesh, vec![Complex64::new(2., 3.); 2]).unwrap();
    let faces = UnstructuredFaceField::<f32, 2>::from_fn(&mesh, |face| face as f32).unwrap();
    let edges = UnstructuredEdgeField::<f64, 2>::from_fn(&mesh, |edge| edge as f64).unwrap();
    assert_eq!(
        (vertices.len(), cells.len(), faces.len(), edges.len()),
        (4, 2, 5, 5)
    );
    let borrowed = FieldView::<_, UnstructuredFace, 2>::new(&mesh, faces.values()).unwrap();
    assert_eq!(borrowed.to_owned().unwrap().values(), faces.values());
    assert!(
        vertices
            .view()
            .to_owned()
            .unwrap()
            .is_compatible_with(&mesh)
    );
    let mapped = cells.try_map(|value| value.re).unwrap();
    assert_eq!(mapped.values(), &[2., 2.]);
    assert_eq!(
        mapped.try_zip_map(&mapped, |a, b| a + b).unwrap().values(),
        &[4., 4.]
    );
    assert!(
        matches ! (& (CellField :: < f64 , 2 > :: new (& mesh , vec ! [0. ; 4])) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: DimensionMismatch { .. }))
    );
    assert!(FieldView::<f64, UnstructuredFace, 2>::new(&mesh, &[0.; 4]).is_err());
    assert!(
        mapped
            .try_zip_map(&CellField::new(&square(), vec![1., 1.]).unwrap(), |a, b| a
                + b)
            .is_err()
    );
    let changed = mesh.with_cells(vec![[0, 1, 3], [1, 2, 3]]).unwrap();
    assert!(!mapped.is_compatible_with(&changed));
}
#[test]
fn empty_fields_are_real_empty_values() {
    let mesh = TriangleMesh::<f32>::new(vec![], vec![]).unwrap();
    let values = CellField::<Complex64, 2>::from_fn(&mesh, |_| unreachable!()).unwrap();
    assert!(values.is_empty());
    assert!(values.view().is_empty());
    assert!(values.try_map(|v| v.re).unwrap().is_empty());
    assert!(
        UnstructuredFaceField::<f64, 2>::new(&mesh, vec![])
            .unwrap()
            .is_empty()
    );
    assert!(
        UnstructuredEdgeField::<f32, 2>::new(&mesh, vec![])
            .unwrap()
            .is_empty()
    );
}
#[test]
fn boundary_tags_check_exterior_membership_duplicates_and_identity() {
    let mesh = square();
    let boundary: Vec<_> = mesh.boundary_faces().collect();
    let tags = SimplexBoundaryTags::new(
        &mesh,
        vec![(boundary[2], 7), (boundary[0], 7), (boundary[1], 0)],
    )
    .unwrap();
    assert_eq!(tags.len(), 3);
    assert_eq!(tags.tag(&mesh, boundary[0]).unwrap(), Some(7));
    assert_eq!(tags.tag(&mesh, boundary[1]).unwrap(), Some(0));
    assert_eq!(tags.tag(&mesh, boundary[3]).unwrap(), None);
    assert_eq!(tags.tag(&mesh, 1).unwrap(), None); // diagonal (0,2)
    assert!(tags.entries().windows(2).all(|w| w[0].0 < w[1].0));
    assert!(tags.is_compatible_with(&mesh));
    assert!(tags.tag(&square(), boundary[0]).is_err());
    assert!(tags.tag(&mesh, usize::MAX).is_err());
    assert!(SimplexBoundaryTags::new(&mesh, vec![(1, 0)]).is_err());
    assert!(SimplexBoundaryTags::new(&mesh, vec![(boundary[0], 1), (boundary[0], 2)]).is_err());
    assert!(SimplexBoundaryTags::new(&mesh, vec![(mesh.face_count(), 1)]).is_err());
    let changed = mesh.with_cells(vec![[0, 1, 3], [1, 2, 3]]).unwrap();
    assert!(!tags.is_compatible_with(&changed));
}

#[test]
fn unstructured_locations_span_mesh_families_but_keep_exact_identity() {
    use scioxide_spatial::{AdaptationLimits, AdaptiveMesh, UnstructuredEdge};
    let simplex = square();
    let adaptive = AdaptiveMesh::from_simplex(&simplex, AdaptationLimits::new()).unwrap();
    let simplex_faces = UnstructuredFaceField::<u32, 2>::from_fn(&simplex, |i| i as u32).unwrap();
    let adaptive_faces = UnstructuredFaceField::<u32, 2>::from_fn(&adaptive, |i| i as u32).unwrap();
    let simplex_edges = UnstructuredEdgeField::<u32, 2>::from_fn(&simplex, |i| i as u32).unwrap();
    let adaptive_edges = UnstructuredEdgeField::<u32, 2>::from_fn(&adaptive, |i| i as u32).unwrap();
    assert_eq!(simplex_faces.len(), simplex.face_count());
    assert_eq!(adaptive_faces.len(), adaptive.face_count());
    assert_eq!(simplex_edges.len(), simplex.edge_count());
    assert_eq!(adaptive_edges.len(), adaptive.edge_count());
    assert!(
        simplex_faces
            .try_zip_map(&adaptive_faces, |a, b| a + b)
            .is_err()
    );
    assert!(
        simplex_edges
            .try_zip_map(&adaptive_edges, |a, b| a + b)
            .is_err()
    );
    assert!(FieldView::<_, UnstructuredFace, 2>::new(&adaptive, adaptive_faces.values()).is_ok());
    assert!(FieldView::<_, UnstructuredEdge, 2>::new(&adaptive, &[] as &[u32]).is_err());
    assert!(UnstructuredFaceField::<u32, 2>::new(&adaptive, vec![]).is_err());
    assert!(UnstructuredEdgeField::<u32, 2>::new(&simplex, vec![]).is_err());
}
