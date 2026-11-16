#![allow(clippy::unwrap_used)]
use scioxide_core::{
    ExecutionContext, NdView, RealScalar, SciOxideErrorKind, StridedLayout, Tolerance,
};
use scioxide_linalg::Matrix;
use scioxide_spatial::*;

fn n<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn cx() -> ExecutionContext<'static> {
    ExecutionContext::sequential_native()
}
fn check<S: RealScalar>() {
    let data = [[S::ZERO, S::ZERO], [n(3.0), n(4.0)], [n(6.0), n(8.0)]];
    let points = PointSet::from_points(&data).unwrap();
    let rows = PointSet::from_rows(data.as_flattened(), 2).unwrap();
    let view = PointSet::new(
        NdView::new(
            data.as_flattened(),
            StridedLayout::new(&[3, 2], &[2, 1]).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    for i in 0..3 {
        for j in 0..2 {
            assert_eq!(
                points.coordinate(i, j).unwrap(),
                rows.coordinate(i, j).unwrap()
            );
            assert_eq!(
                points.coordinate(i, j).unwrap(),
                view.coordinate(i, j).unwrap()
            );
        }
    }
    let euclidean = PreparedDistance::new(DistanceMetric::Euclidean).unwrap();
    assert_eq!(distance(&data[0], &data[1]).unwrap(), n::<S>(5.0));
    assert_eq!(
        pairwise(&points, &points).unwrap().as_slice(),
        euclidean.pairwise(&points, &points).unwrap().as_slice()
    );
    assert_eq!(
        condensed(&points).unwrap(),
        euclidean.condensed(&points).unwrap()
    );
    let covariance = Matrix::from_vec(2, 2, vec![S::ONE, S::ZERO, S::ZERO, S::ONE]).unwrap();
    for metric in [
        euclidean,
        PreparedDistance::new(DistanceMetric::CityBlock).unwrap(),
        PreparedDistance::mahalanobis(&covariance).unwrap(),
    ] {
        let options = DistanceOptions::new();
        let mut ws =
            DistanceWorkspace::workspace(DistanceWorkspaceRequest::pairwise(3, 3, options))
                .unwrap();
        let mut out = [S::ZERO; 9];
        metric
            .pairwise_into(&points, &points, options, &mut out, &mut ws, cx())
            .unwrap();
        assert_eq!(metric.pairwise(&points, &points).unwrap().as_slice(), out);
        let mut ws =
            DistanceWorkspace::workspace(DistanceWorkspaceRequest::condensed(3, options)).unwrap();
        let mut out = [S::ZERO; 3];
        metric
            .condensed_into(&points, options, &mut out, &mut ws, cx())
            .unwrap();
        assert_eq!(metric.condensed(&points).unwrap(), out);
    }
    for count in [0, 1, 2, 5] {
        let values = vec![S::ONE; condensed_len(count).unwrap()];
        let matrix = condensed_to_square(count, &values).unwrap();
        assert_eq!(square_to_condensed(&matrix, S::ZERO).unwrap(), values);
        let options = DistanceOptions::new();
        let mut ws =
            DistanceWorkspace::workspace(DistanceWorkspaceRequest::square(count, options)).unwrap();
        let mut square = vec![S::ZERO; count * count];
        condensed_to_square_into(count, &values, options, &mut square, &mut ws, cx()).unwrap();
        assert_eq!(matrix.as_slice(), square);
        let mut ws =
            DistanceWorkspace::workspace(DistanceWorkspaceRequest::condensed(count, options))
                .unwrap();
        let mut packed = vec![S::ZERO; values.len()];
        square_to_condensed_into(&matrix, S::ZERO, options, &mut packed, &mut ws, cx()).unwrap();
        assert_eq!(packed, values);
    }
    let empty = PointSet::from_points(&[] as &[[S; 2]]).unwrap();
    assert_eq!(
        (
            pairwise(&empty, &points).unwrap().rows(),
            pairwise(&empty, &points).unwrap().cols()
        ),
        (0, 3)
    );
    assert_eq!(
        (
            pairwise(&points, &empty).unwrap().rows(),
            pairwise(&points, &empty).unwrap().cols()
        ),
        (3, 0)
    );
    assert_eq!(
        square_to_condensed(&Matrix::from_vec(1, 2, vec![S::ZERO; 2]).unwrap(), S::ZERO)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let tree = KdTree::new(&points).unwrap();
    let configured =
        KdTree::with_options(&points, TreeMetric::Euclidean, KdTreeOptions::new()).unwrap();
    assert_eq!(tree.node_count(), configured.node_count());
    assert_eq!(tree.storage_bytes(), configured.storage_bytes());
    let queries: Vec<_> = (0..64).map(|i| [n::<S>(i as f64 / 8.0), S::ONE]).collect();
    let query_points = PointSet::from_points(&queries).unwrap();
    let batch = tree.nearest_batch(&query_points, 2).unwrap();
    for (query, row) in queries.iter().zip(batch.chunks_exact(2)) {
        assert_eq!(row, tree.nearest(query, 2).unwrap());
        assert_eq!(row, configured.nearest(query, 2).unwrap());
        let options = SearchOptions::new();
        let mut ws = tree
            .workspace(SearchWorkspaceRequest::new(3, options))
            .unwrap();
        let mut out = [Neighbor::default(); 3];
        let count = tree
            .nearest_into(query, 2, options, &mut out, &mut ws, cx())
            .unwrap();
        assert_eq!(row, &out[..count]);
        let count = tree
            .radius_into(query, n(5.0), options, &mut out, &mut ws, cx())
            .unwrap();
        assert_eq!(tree.radius(query, n(5.0)).unwrap(), &out[..count]);
    }
    assert!(tree.nearest_batch(&points, 0).unwrap().is_empty());
    assert_eq!(tree.nearest_batch(&points, 9).unwrap().len(), 9);
    assert!(tree.nearest_batch(&empty, 3).unwrap().is_empty());
    assert!(
        KdTree::new(&empty)
            .unwrap()
            .nearest_batch(&points, 3)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        tree.nearest_batch(&PointSet::from_points(&[[S::ZERO]]).unwrap(), 1)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        PointSet::from_rows(&[S::ZERO], 0).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    let error = PointSet::from_rows(&[S::ZERO; 3], 2).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    assert!(
        matches!(error.view(), scioxide_core::SciOxideErrorView::DimensionMismatch { expected, actual } if expected == [2] && actual == [1])
    );
    for value in [S::NAN, S::INFINITY] {
        assert_eq!(
            PointSet::from_rows(&[value], 1).unwrap_err().kind(),
            SciOxideErrorKind::NonFinite
        );
        assert_eq!(
            PointSet::from_points(&[[value]]).unwrap_err().kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    assert_eq!(PointSet::from_points(&[[] as [S; 0]; 3]).unwrap().len(), 3);
    assert_eq!(
        points.coordinate(3, 0).unwrap_err().kind(),
        SciOxideErrorKind::OutOfBounds
    );
    assert_eq!(
        points.coordinate(0, 2).unwrap_err().kind(),
        SciOxideErrorKind::OutOfBounds
    );
    geometry::<S>();
    transforms::<S>();
    grids::<S>();
}
fn geometry<S: RealScalar>() {
    let sites = [[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0.3, 0.4]].map(|p| p.map(n::<S>));
    let points = PointSet::from_points(&sites).unwrap();
    let mesh = Delaunay::new(&points).unwrap();
    let hull = ConvexHull::new(&points).unwrap();
    for other in [
        Delaunay::with_options(&points, GeometryOptions::new()).unwrap(),
        Delaunay::build(&points, GeometryOptions::new(), cx()).unwrap(),
    ] {
        assert_eq!(mesh.points(), other.points());
        assert_eq!(mesh.simplices(), other.simplices());
        assert_eq!(mesh.input_mapping(), other.input_mapping());
        assert_eq!(mesh.representatives(), other.representatives());
        assert_eq!(mesh.dimension(), other.dimension());
        assert_eq!(mesh.simplex_count(), other.simplex_count());
        for i in 0..mesh.simplex_count() {
            assert_eq!(mesh.neighbors(i).unwrap(), other.neighbors(i).unwrap());
        }
        for i in 0..sites.len() {
            assert_eq!(
                mesh.incident_simplices(i).unwrap().collect::<Vec<_>>(),
                other.incident_simplices(i).unwrap().collect::<Vec<_>>()
            );
        }
    }
    for other in [
        ConvexHull::with_options(&points, GeometryOptions::new()).unwrap(),
        ConvexHull::build(&points, GeometryOptions::new(), cx()).unwrap(),
    ] {
        assert_eq!(hull.points(), other.points());
        assert_eq!(hull.facets(), other.facets());
        assert_eq!(hull.input_mapping(), other.input_mapping());
        assert_eq!(hull.representatives(), other.representatives());
        assert_eq!(hull.coplanar_faces(), other.coplanar_faces());
        for i in 0..hull.facet_count() {
            assert_eq!(hull.neighbors(i).unwrap(), other.neighbors(i).unwrap());
        }
    }
    for (i, row) in mesh.simplices().chunks_exact(3).enumerate() {
        assert_eq!(row, mesh.simplex(i).unwrap());
    }
    for (i, row) in hull.facets().chunks_exact(2).enumerate() {
        assert_eq!(row, hull.facet(i).unwrap());
    }
    for (query, inside) in [([0.5, 0.2], true), ([0., 0.], true), ([2., 0.], false)] {
        let query = query.map(n::<S>);
        let found = mesh.find_simplex(&query).unwrap();
        assert_eq!(found.is_some(), inside);
        assert_eq!(
            found,
            mesh.find_simplex_scoped(&query, &mut mesh.workspace(()).unwrap(), cx())
                .unwrap()
        );
        assert_eq!(hull.contains(&query).unwrap(), inside);
        assert_eq!(
            hull.contains(&query).unwrap(),
            hull.contains_scoped(&query, &mut hull.workspace(()).unwrap(), cx())
                .unwrap()
        );
        if let Some(found) = found {
            let mut out = [S::ZERO; 3];
            let mut scratch = out;
            mesh.barycentric_into(found.simplex, &query, &mut out, &mut scratch)
                .unwrap();
            assert_eq!(mesh.barycentric(found.simplex, &query).unwrap(), out);
            assert!(
                (out.iter().fold(S::ZERO, |a, &b| a + b) - S::ONE).abs() < n::<S>(32.) * S::EPSILON
            );
        }
    }
    for query in [&[S::ZERO][..], &[S::NAN, S::ZERO][..]] {
        let kind = if query.len() == 1 {
            SciOxideErrorKind::DimensionMismatch
        } else {
            SciOxideErrorKind::NonFinite
        };
        assert_eq!(mesh.find_simplex(query).unwrap_err().kind(), kind);
        assert_eq!(hull.contains(query).unwrap_err().kind(), kind);
    }
    assert_eq!(
        mesh.barycentric(mesh.simplex_count(), &[S::ZERO; 2])
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::OutOfBounds
    );
    let dual = Voronoi::new(&points).unwrap();
    for other in [
        Voronoi::with_options(&points, GeometryOptions::new()).unwrap(),
        Voronoi::build(&points, GeometryOptions::new(), cx()).unwrap(),
    ] {
        assert_eq!(dual.vertices().unwrap(), other.vertices().unwrap());
        assert_eq!(dual.simplex_vertices(), other.simplex_vertices());
        for i in 0..sites.len() {
            assert_eq!(
                dual.region_is_unbounded(i).unwrap(),
                other.region_is_unbounded(i).unwrap()
            );
            assert_eq!(
                dual.region_vertices(i).unwrap().collect::<Vec<_>>(),
                other.region_vertices(i).unwrap().collect::<Vec<_>>()
            );
        }
    }
    let planes =
        [[-1., 0., 0.], [1., 0., -1.], [0., -1., 0.], [0., 1., -1.]].map(|p| p.map(n::<S>));
    let planes = PointSet::from_points(&planes).unwrap();
    let interior = [n::<S>(0.5); 2];
    let poly = HalfspaceIntersection::new(&planes, &interior).unwrap();
    for other in [
        HalfspaceIntersection::with_options(&planes, &interior, GeometryOptions::new()).unwrap(),
        HalfspaceIntersection::build(&planes, &interior, GeometryOptions::new(), cx()).unwrap(),
    ] {
        assert_eq!(poly.vertices(), other.vertices());
        assert_eq!(poly.kind(), other.kind());
        assert_eq!(poly.dimension(), other.dimension());
        for i in 0..poly.vertices().len() / 2 {
            assert_eq!(
                poly.active_halfspaces(i).unwrap(),
                other.active_halfspaces(i).unwrap()
            );
            assert_eq!(
                poly.neighbors(i).unwrap().collect::<Vec<_>>(),
                other.neighbors(i).unwrap().collect::<Vec<_>>()
            );
        }
    }
    let sites = [
        [1., 0., 0.],
        [-1., 0., 0.],
        [0., 1., 0.],
        [0., -1., 0.],
        [0., 0., 1.],
        [0., 0., -1.],
    ]
    .map(|p| p.map(n::<S>));
    let points = PointSet::from_points(&sites).unwrap();
    let tol = Tolerance::new(n::<S>(32.) * S::EPSILON, S::ZERO).unwrap();
    let sphere = SphericalVoronoi::new(&points, [S::ZERO; 3], S::ONE, tol).unwrap();
    for other in [
        SphericalVoronoi::with_options(&points, [S::ZERO; 3], S::ONE, tol, GeometryOptions::new())
            .unwrap(),
        SphericalVoronoi::build(
            &points,
            [S::ZERO; 3],
            S::ONE,
            tol,
            GeometryOptions::new(),
            cx(),
        )
        .unwrap(),
    ] {
        assert_eq!(sphere.vertices(), other.vertices());
        assert_eq!(sphere.areas(), other.areas());
        assert_eq!(sphere.center(), other.center());
        assert_eq!(sphere.radius(), other.radius());
        assert_eq!(sphere.input_mapping(), other.input_mapping());
        for i in 0..sites.len() {
            assert_eq!(sphere.region(i).unwrap(), other.region(i).unwrap());
            assert_eq!(
                sphere.neighbors(i).unwrap().collect::<Vec<_>>(),
                other.neighbors(i).unwrap().collect::<Vec<_>>()
            );
        }
    }
}
fn transforms<S: RealScalar>() {
    let data = [
        [S::ZERO; 3],
        [S::ONE, S::ZERO, S::ZERO],
        [S::ZERO, S::ONE, S::ZERO],
    ];
    let points = PointSet::from_points(&data).unwrap();
    let fit = estimate_rigid_transform(&points, &points).unwrap();
    for other in [
        estimate_rigid_transform_with_options(
            &points,
            &points,
            None,
            RigidEstimationOptions::new(),
        )
        .unwrap(),
        estimate_rigid_transform_scoped(
            &points,
            &points,
            None,
            RigidEstimationOptions::new(),
            cx(),
        )
        .unwrap(),
    ] {
        assert_eq!(fit.transform, other.transform);
        assert_eq!(fit.rank, other.rank);
    }
    let rotation = Rotation::from_axis_angle([S::ZERO, S::ZERO, S::ONE], n(0.5)).unwrap();
    let transform = RigidTransform::new(rotation, [S::ONE; 3]).unwrap();
    assert_eq!(
        rotation.apply_batch(&points).unwrap(),
        data.map(|p| rotation.apply(p).unwrap())
    );
    assert_eq!(
        transform.apply_batch(&points).unwrap(),
        data.map(|p| transform.apply(p).unwrap())
    );
    let empty = PointSet::from_points(&[] as &[[S; 3]]).unwrap();
    assert!(rotation.apply_batch(&empty).unwrap().is_empty());
    assert!(transform.apply_batch(&empty).unwrap().is_empty());
    let wrong_data = [[S::ZERO; 2]];
    let wrong = PointSet::from_points(&wrong_data).unwrap();
    assert_eq!(
        rotation.apply_batch(&wrong).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        transform.apply_batch(&wrong).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        estimate_rigid_transform(&empty, &empty).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    let times = [S::ZERO, S::ONE, n(2.)];
    let rotations = [Rotation::identity(), rotation, Rotation::identity()];
    let spline = RotationSpline::new(&times, &rotations).unwrap();
    let configured = RotationSpline::with_options(&times, &rotations, RotationSplineOptions::new())
        .unwrap()
        .spline
        .unwrap();
    let slerp = Slerp::new(&times, &rotations).unwrap();
    let queries = [n::<S>(0.5), n(1.5)];
    assert_eq!(
        spline.evaluate_rotations(&queries).unwrap(),
        queries.map(|t| spline.evaluate(t).unwrap().rotation)
    );
    assert_eq!(
        spline.evaluate_rotations(&queries).unwrap(),
        configured.evaluate_rotations(&queries).unwrap()
    );
    assert_eq!(
        slerp.evaluate_rotations(&queries).unwrap(),
        queries.map(|t| slerp.evaluate(t).unwrap())
    );
    assert_eq!(
        spline.evaluate_rotations(&[n(-1.)]).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        slerp.evaluate_rotations(&[n(-1.)]).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
}
fn grids<S: RealScalar>() {
    fn grid<S: RealScalar, const D: usize>() {
        for end in [S::ONE, -S::ONE] {
            let actual = StructuredGrid::uniform([S::ZERO; D], [end; D], [3; D]).unwrap();
            let expected = StructuredGrid::new(
                core::array::from_fn(|_| CoordinateAxis::uniform(S::ZERO, end, 3).unwrap()),
                [AxisTopology::Bounded; D],
            )
            .unwrap();
            assert_eq!(actual.vertex_extent(), expected.vertex_extent());
            assert_eq!(
                actual.point(Index::new([1; D]).unwrap()).unwrap(),
                expected.point(Index::new([1; D]).unwrap()).unwrap()
            );
            assert_eq!(
                CellMesh::uniform([S::ZERO; D], [end; D], [3; D])
                    .unwrap()
                    .topology_sizes(),
                CellMesh::new(expected).unwrap().topology_sizes()
            );
        }
        assert_eq!(
            StructuredGrid::uniform([S::ZERO; D], [S::ONE; D], [1; D])
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::InvalidInput
        );
        assert_eq!(
            StructuredGrid::uniform([S::ZERO; D], [S::ZERO; D], [3; D])
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::InvalidInput
        );
        assert_eq!(
            StructuredGrid::uniform([S::NAN; D], [S::ONE; D], [3; D])
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    grid::<S, 1>();
    grid::<S, 2>();
    grid::<S, 3>();
    let coordinates = [[S::ZERO], [S::ONE]];
    let a = IntervalMesh::from_coordinates(&coordinates, vec![[0, 1]]).unwrap();
    let b = IntervalMesh::new(
        coordinates.map(|p| Point::new(p).unwrap()).to_vec(),
        vec![[0, 1]],
    )
    .unwrap();
    assert_eq!(a.points(), b.points());
    assert_eq!(a.cells().collect::<Vec<_>>(), b.cells().collect::<Vec<_>>());
    let coordinates = [[S::ZERO, S::ZERO], [S::ONE, S::ZERO], [S::ZERO, S::ONE]];
    let a = TriangleMesh::from_coordinates(&coordinates, vec![[0, 1, 2]]).unwrap();
    let b = TriangleMesh::new(
        coordinates.map(|p| Point::new(p).unwrap()).to_vec(),
        vec![[0, 1, 2]],
    )
    .unwrap();
    assert_eq!(a.points(), b.points());
    assert_eq!(a.cells().collect::<Vec<_>>(), b.cells().collect::<Vec<_>>());
    let coordinates = [
        [S::ZERO; 3],
        [S::ONE, S::ZERO, S::ZERO],
        [S::ZERO, S::ONE, S::ZERO],
        [S::ZERO, S::ZERO, S::ONE],
    ];
    let a = TetrahedronMesh::from_coordinates(&coordinates, vec![[0, 1, 2, 3]]).unwrap();
    let b = TetrahedronMesh::new(
        coordinates.map(|p| Point::new(p).unwrap()).to_vec(),
        vec![[0, 1, 2, 3]],
    )
    .unwrap();
    assert_eq!(a.points(), b.points());
    assert_eq!(a.cells().collect::<Vec<_>>(), b.cells().collect::<Vec<_>>());
    assert_eq!(
        IntervalMesh::from_coordinates(&[[S::NAN]], vec![])
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(
        TriangleMesh::from_coordinates(&[[S::ZERO; 2]], vec![[0, 1, 2]])
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::OutOfBounds
    );
}
#[test]
fn f32_forms() {
    check::<f32>();
}
#[test]
fn f64_forms() {
    check::<f64>();
}

#[test]
fn adaptive_owned_and_scoped_forms() {
    fn check<S: RealScalar>() {
        let root = SegmentMesh::uniform([S::ZERO], [S::ONE], [4]).unwrap();
        let mesh = AdaptiveMesh::from_structured(&root, AdaptationLimits::new()).unwrap();
        let request = AdaptationRequest {
            source: mesh.mesh_token(),
            refine: &[0],
            coarsen: &[],
        };
        let owned = mesh.adapt(request).unwrap();
        let scoped = mesh.adapt_scoped(request, cx()).unwrap();
        assert_eq!(owned.mesh.active_cells(), scoped.mesh.active_cells());
        assert_eq!(owned.induced_refinements, scoped.induced_refinements);
        assert_eq!(owned.work, scoped.work);
        let source = MeshPartition::balanced(&mesh, 2).unwrap();
        let destination = MeshPartition::new(&mesh, 2, &[1, 0, 1]).unwrap();
        let plain = RepartitionMap::new(&source, &destination).unwrap();
        let configured =
            RepartitionMap::with_options(&source, &destination, RepartitionOptions::new()).unwrap();
        assert_eq!(plain.permutation(), configured.permutation());
        let mut output = [0; 3];
        plain
            .gather_into(&[10, 20, 30], &mut output, &mut [0; 3], cx())
            .unwrap();
        assert_eq!(
            output,
            plain
                .permutation()
                .iter()
                .map(|&i| [10, 20, 30][i])
                .collect::<Vec<_>>()
                .as_slice()
        );
    }
    check::<f32>();
    check::<f64>();
}
