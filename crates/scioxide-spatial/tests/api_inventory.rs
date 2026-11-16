//! Reviewed root API inventory, compiled as an external consumer.
#![allow(unused_imports)]
use scioxide_spatial::{
    AdaptationLimits, AdaptationRequest, AdaptationResult, AdaptiveCellKind, AdaptiveInterface,
    AdaptiveInterfaceSide, AdaptiveMesh, AxisOrientation, AxisSide, AxisTopology, BoundarySet,
    Cell, CellField, CellKind, CellLocation, CellMesh, ConvexHull, CoordinateAxis, Delaunay,
    DistanceMetric, DistanceOptions, DistanceWorkspace, DistanceWorkspaceRequest,
    DistanceWorkspaceRequirements, Edge, EdgeField, EntityRelation, EulerAngles, EulerFrame,
    EulerSequence, Extent, Extent1, Extent2, Extent3, Face, FaceField, FaceIndex, FaceTopology,
    Field, FieldLocation, FieldView, GeometryConstructionRequirements, GeometryDegeneracy,
    GeometryLimits, GeometryOptions, GeometryProvenance, GeometryWorkspace,
    GeometryWorkspaceRequirements, Grid1d, Grid2d, Grid3d, HalfspaceIntersection, HangingEntity,
    HangingEntityKind, HausdorffResult, HexMesh, Index, Index1, Index2, Index3, IntersectionKind,
    IntervalLocation, IntervalMesh, KdTree, KdTreeOptions, MeshEntities, MeshGeneration, MeshId,
    MeshPartition, MeshToken, Neighbor, OrientedEntity, Point, Point1, Point2, Point3, PointSet,
    PredicateSign, PreparedDistance, QuadMesh, RepartitionMap, RepartitionOptions,
    RigidEstimationOptions, RigidEstimationResult, RigidTransform, Rotation, RotationAxis,
    RotationSpline, RotationSplineOptions, RotationSplineResult, RotationSplineSample,
    RotationSplineTermination, SearchOptions, SearchWorkspace, SearchWorkspaceRequest,
    SearchWorkspaceRequirements, SegmentMesh, SimplexBoundaryTags, SimplexLocation, SimplexMesh,
    SimplexPredicateWorkspace, Slerp, SpatialExecutionContext, SphericalVoronoi, StructuredGrid,
    TetrahedronMesh, TopologyMap, TopologySizes, TransformOptions, TransformWorkspace,
    TransformWorkspaceRequest, TransformWorkspaceRequirements, TreeMetric, TriangleMesh,
    UnstructuredEdge, UnstructuredEdgeField, UnstructuredFace, UnstructuredFaceField, Vertex,
    VertexField, Voronoi, VoronoiRay, VoronoiRidge, condensed, condensed_index, condensed_len,
    condensed_to_square, condensed_to_square_into, directed_hausdorff, distance,
    estimate_rigid_transform, estimate_rigid_transform_scoped,
    estimate_rigid_transform_with_options, incircle, insphere, orient2d, orient3d, pairwise,
    square_to_condensed, square_to_condensed_into, validate_condensed, validate_distance_matrix,
};

#[test]
fn planless_workspace_protocols() -> Result<(), scioxide_core::SciOxideError> {
    use scioxide_core::Capacity;
    let options = DistanceOptions::new().with_max_workspace_bytes(1024);
    assert_eq!(options.max_workspace_bytes(), 1024);
    assert_eq!(
        SearchOptions::new()
            .with_max_workspace_bytes(64)
            .max_workspace_bytes(),
        64
    );
    for request in [
        DistanceWorkspaceRequest::pairwise(2, 3, options),
        DistanceWorkspaceRequest::condensed(3, options),
        DistanceWorkspaceRequest::square(3, options),
    ] {
        let requirements = DistanceWorkspace::<f64>::workspace_requirements(request)?;
        let workspace = DistanceWorkspace::<f64>::workspace(request)?;
        assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    }
    let options = TransformOptions::new().with_max_workspace_bytes(1024);
    assert_eq!(options.max_workspace_bytes(), 1024);
    for request in [
        TransformWorkspaceRequest::apply(3, options),
        TransformWorkspaceRequest::euler(3, options),
        TransformWorkspaceRequest::rotations(3, options),
    ] {
        let requirements = TransformWorkspace::<f64>::workspace_requirements(request)?;
        let workspace = TransformWorkspace::<f64>::workspace(request)?;
        assert_eq!(workspace.check(&requirements)?, Capacity::Sufficient);
    }
    Ok(())
}

#[allow(dead_code)]
fn owned_signatures<'a>(
    points: &'a PointSet<'a, f64>,
    tree: &KdTree<'a, f64>,
    hull: &ConvexHull<f64>,
    delaunay: &Delaunay<f64>,
    mesh: &AdaptiveMesh<f64, 2>,
    source: &MeshPartition,
    target: &MeshPartition,
) -> Result<(), scioxide_core::SciOxideError> {
    use scioxide_linalg::Matrix;
    let _: PointSet<'a, f64> = PointSet::from_points(&[] as &[[f64; 2]])?;
    let _: PointSet<'a, f64> = PointSet::from_rows(&[], 2)?;
    let _: KdTree<'a, f64> = KdTree::new(points)?;
    let _: KdTree<'a, f64> =
        KdTree::with_options(points, TreeMetric::Euclidean, KdTreeOptions::new())?;
    let _: Vec<Neighbor<f64>> = tree.nearest(&[0.; 2], 1)?;
    let _: Vec<Neighbor<f64>> = tree.radius(&[0.; 2], 1.)?;
    let _: Vec<Neighbor<f64>> = tree.nearest_batch(points, 1)?;
    let _: f64 = distance(&[], &[])?;
    let _: Matrix<f64> = pairwise(points, points)?;
    let _: Vec<f64> = condensed(points)?;
    let _: Matrix<f64> = condensed_to_square(0, &[])?;
    let _: Vec<f64> = square_to_condensed(&Matrix::<f64>::from_vec(0, 0, vec![])?, 0.)?;
    let metric = PreparedDistance::new(DistanceMetric::Euclidean)?;
    let _: Matrix<f64> = metric.pairwise(points, points)?;
    let _: Vec<f64> = metric.condensed(points)?;
    let _: ConvexHull<f64> = ConvexHull::new(points)?;
    let _: ConvexHull<f64> = ConvexHull::with_options(points, GeometryOptions::new())?;
    let _: Delaunay<f64> = Delaunay::new(points)?;
    let _: Delaunay<f64> = Delaunay::with_options(points, GeometryOptions::new())?;
    let _: Voronoi<f64> = Voronoi::new(points)?;
    let _: Voronoi<f64> = Voronoi::with_options(points, GeometryOptions::new())?;
    let _: HalfspaceIntersection<f64> = HalfspaceIntersection::new(points, &[])?;
    let _: HalfspaceIntersection<f64> =
        HalfspaceIntersection::with_options(points, &[], GeometryOptions::new())?;
    let tolerance = scioxide_core::Tolerance::new(1e-12, 0.)?;
    let _: SphericalVoronoi<f64> = SphericalVoronoi::new(points, [0.; 3], 1., tolerance)?;
    let _: SphericalVoronoi<f64> =
        SphericalVoronoi::with_options(points, [0.; 3], 1., tolerance, GeometryOptions::new())?;
    let _: bool = hull.contains(&[])?;
    let _: &[usize] = hull.facets();
    let _: &[usize] = delaunay.simplices();
    let _: PointSet<'_, f64> = delaunay.point_set();
    let _: Option<SimplexLocation> = delaunay.find_simplex(&[])?;
    let _: Vec<f64> = delaunay.barycentric(0, &[])?;
    let _: Vec<[f64; 3]> = Rotation::identity().apply_batch(points)?;
    let _: Vec<[f64; 3]> = RigidTransform::identity().apply_batch(points)?;
    let _: RigidEstimationResult<f64> = estimate_rigid_transform(points, points)?;
    let _: RigidEstimationResult<f64> =
        estimate_rigid_transform_with_options(points, points, None, RigidEstimationOptions::new())?;
    let _: RotationSpline<f64> = RotationSpline::new(&[], &[])?;
    let result: RotationSplineResult<f64> =
        RotationSpline::with_options(&[], &[], RotationSplineOptions::new())?;
    if let Some(spline) = result.spline {
        let _: Vec<Rotation<f64>> = spline.evaluate_rotations(&[])?;
    }
    let _: Vec<Rotation<f64>> = Slerp::<f64>::new(&[], &[])?.evaluate_rotations(&[])?;
    let _: Grid2d<f64> = StructuredGrid::uniform([0.; 2], [1.; 2], [2; 2])?;
    let _: QuadMesh<f64> = CellMesh::uniform([0.; 2], [1.; 2], [2; 2])?;
    let _: IntervalMesh<f64> = IntervalMesh::from_coordinates(&[], vec![])?;
    let _: TriangleMesh<f64> = TriangleMesh::from_coordinates(&[], vec![])?;
    let _: TetrahedronMesh<f64> = TetrahedronMesh::from_coordinates(&[], vec![])?;
    let _: AdaptationResult<f64, 2> = mesh.adapt(AdaptationRequest {
        source: mesh.mesh_token(),
        refine: &[],
        coarsen: &[],
    })?;
    let _: RepartitionMap = RepartitionMap::new(source, target)?;
    let _: RepartitionMap =
        RepartitionMap::with_options(source, target, RepartitionOptions::new())?;
    Ok(())
}
