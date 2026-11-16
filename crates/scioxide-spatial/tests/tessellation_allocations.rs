//! Construction allocation growth and transactional failures; no timing gates.
#![allow(clippy::unwrap_used)]

use scioxide_core::{NdView, StridedLayout, Tolerance};
use scioxide_spatial::*;

fn options(n: usize) -> GeometryOptions {
    GeometryOptions::new().with_limits(
        GeometryLimits::new()
            .with_max_dimension(5)
            .with_max_points(n)
            .with_max_facets(16 * n)
            .with_max_predicate_terms(1_000_000_000)
            .with_max_storage_bytes(usize::MAX),
    )
}

fn with_points<T>(values: &[f64], d: usize, f: impl FnOnce(&PointSet<'_, f64>) -> T) -> T {
    let shape = [values.len() / d, d];
    let strides = [d, 1];
    let view = NdView::new(values, StridedLayout::new(&shape, &strides).unwrap()).unwrap();
    f(&PointSet::new(view).unwrap())
}

fn coordinates(n: usize, d: usize) -> Vec<f64> {
    let mut state = 1116_u64;
    (0..n * d)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        })
        .collect()
}

#[test]
fn canonical_point_set_borrows_snapshot_without_allocating() {
    let mesh = with_points(&[1., 0., 0., 1., 0., 0., 1., 0.], 2, |points| {
        Delaunay::with_options(points, options(4)).unwrap()
    });
    {
        let points = mesh.point_set();
        assert_eq!((points.len(), points.dimension()), (3, 2));
        for (site, expected) in mesh.points().chunks_exact(2).enumerate() {
            assert_eq!(points.coordinate(site, 0).unwrap(), expected[0]);
            assert_eq!(points.coordinate(site, 1).unwrap(), expected[1]);
        }
    };
    let tree = KdTree::new(&mesh.point_set()).unwrap();
    let mut workspace = tree
        .workspace(SearchWorkspaceRequest::new(1, SearchOptions::new()))
        .unwrap();
    let mut neighbors = [Neighbor::default(); 1];
    assert_eq!(
        tree.nearest_into(
            &[0., 0.],
            1,
            SearchOptions::new(),
            &mut neighbors,
            &mut workspace,
            SpatialExecutionContext::sequential()
        )
        .unwrap(),
        1
    );
    assert_eq!(neighbors[0].index, 0);
    assert_eq!(neighbors[0].distance, 0.);
}

fn hull_rows(hull: &ConvexHull<f64>) -> Vec<Vec<usize>> {
    (0..hull.facet_count())
        .map(|i| hull.facet(i).unwrap().to_vec())
        .collect()
}

#[test]
fn indexed_incidence_and_boundary_queries_execute() {
    for d in 2..=4 {
        let sites: Vec<_> = (0..1 << d)
            .flat_map(|i| (0..d).map(move |axis| ((i >> axis) & 1) as f64))
            .collect();
        let o = options(sites.len() / d);
        let dual = with_points(&sites, d, |p| Voronoi::with_options(p, o)).unwrap();
        let geometry = dual.delaunay();
        let mut workspace = test_workspace(test_requirements(d, o).unwrap()).unwrap();
        let query = vec![0.5; d];
        let mut planes = Vec::new();
        for axis in 0..d {
            for sign in [-1., 1.] {
                let mut row = vec![0.; d + 1];
                row[axis] = sign;
                row[d] = -1.;
                planes.extend(row);
            }
        }
        let intersection = with_points(&planes, d + 1, |p| {
            HalfspaceIntersection::with_options(p, &vec![0.; d], options(2 * d))
        })
        .unwrap();
        {
            for site in 0..sites.len() / d {
                for cell in geometry.incident_simplices(site).unwrap() {
                    assert!(geometry.simplex(cell).unwrap().contains(&site));
                }
                // All cospherical cells collapse to one dual vertex.
                assert!(dual.region_vertices(site).unwrap().eq([0]));
                assert!(dual.region_is_unbounded(site).unwrap());
                assert!(
                    geometry
                        .find_simplex_scoped(
                            &geometry.points()[site * d..(site + 1) * d],
                            &mut workspace,
                            scioxide_core::ExecutionContext::sequential()
                        )
                        .unwrap()
                        .unwrap()
                        .boundary
                );
            }
            assert!(
                geometry
                    .find_simplex_scoped(
                        &query,
                        &mut workspace,
                        scioxide_core::ExecutionContext::sequential()
                    )
                    .unwrap()
                    .unwrap()
                    .boundary
            );
            for vertex in 0..intersection.vertices().len() / d {
                for other in intersection.neighbors(vertex).unwrap() {
                    assert!(intersection.neighbors(other).unwrap().any(|v| v == vertex));
                }
            }
            assert!(geometry.incident_simplices(usize::MAX).is_err());
            assert!(dual.region_vertices(usize::MAX).is_err());
            assert!(dual.region_is_unbounded(usize::MAX).is_err());
            assert!(intersection.neighbors(usize::MAX).is_err());
        };
    }
    let sites = [
        1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1.,
    ];
    let sphere = with_points(&sites, 3, |p| {
        SphericalVoronoi::with_options(
            p,
            [0.; 3],
            1.,
            Tolerance::new(1e-12, 1e-12).unwrap(),
            options(6),
        )
    })
    .unwrap();
    {
        for site in 0..6 {
            for other in sphere.neighbors(site).unwrap() {
                assert!(sphere.neighbors(other).unwrap().any(|i| i == site));
            }
        }
        assert!(sphere.neighbors(usize::MAX).is_err());
    };
}

#[test]
fn borrowed_dual_views_and_circumcenters_execute() {
    let values = coordinates(32, 2);
    let dual = with_points(&values, 2, |p| Voronoi::with_options(p, options(32))).unwrap();
    // Slices borrow the diagram, so they outlive the temporary iterator view.
    let first_vertices = dual.ridges().next().unwrap().vertices();
    assert!(!first_vertices.is_empty());
    {
        let mut center = [0.; 2];
        for (index, vertex) in dual.simplex_vertices().iter().enumerate() {
            dual.delaunay()
                .circumcenter_into(index, &mut center)
                .unwrap();
            for (a, b) in center.iter().zip(dual.vertex(*vertex).unwrap()) {
                assert!((a - b).abs() <= 1e-12 * (1. + a.abs()));
            }
        }
        let mut ridges = dual.ridges();
        let count = ridges.len();
        ridges.next();
        assert_eq!(ridges.len(), count - 1);
        for ridge in ridges.rev() {
            assert!(ridge.sites()[0] < ridge.sites()[1]);
            assert!(ridge.vertices().windows(2).all(|v| v[0] < v[1]));
        }
        for ray in dual.rays().rev() {
            assert_eq!(ray.sites().len(), 2);
            let norm = ray.direction().unwrap().iter().map(|x| x * x).sum::<f64>();
            assert!((norm - 1.).abs() <= 1e-12);
        }
    };
}

#[test]
fn unrepresentable_circumcenters_preserve_output_without_allocation() {
    // A well-conditioned local frame can have a physical circumcenter outside
    // binary64 range. Its first coordinate is finite and its second overflows.
    let values = [0., 0., 1e308, 0., 5e307, 1e306];
    let geometry = with_points(&values, 2, |p| Delaunay::with_options(p, options(3))).unwrap();
    let dual = with_points(&values, 2, |p| Voronoi::with_options(p, options(3))).unwrap();
    let mut output = [41., 42.];
    {
        assert_eq!(dual.vertex_count(), 1);
        assert!(dual.vertex(0).is_err());
        assert!(dual.vertices().is_err());
        assert!(dual.rays().all(|ray| ray.direction().is_ok()));
        assert!(geometry.circumcenter_into(0, &mut output).is_err());
        assert_eq!(output, [41., 42.]);
        assert!(geometry.circumcenter_into(1, &mut output).is_err());
        assert_eq!(output, [41., 42.]);
        assert!(geometry.circumcenter_into(0, &mut output[..1]).is_err());
        assert_eq!(output, [41., 42.]);
    };
}

#[test]
fn unavailable_affine_queries_execute() {
    let values = [0., 0., 2., 0., 1., 1e-7];
    let dual = with_points(&values, 2, |p| Voronoi::with_options(p, options(3))).unwrap();
    let geometry = dual.delaunay();
    let mut output = [41.; 3];
    let mut stage = output;
    let mut gradients = [42.; 6];
    let mut gradient_stage = gradients;
    let mut center = [43.; 2];
    {
        assert!(
            geometry
                .barycentric_into(0, &[1., 5e-8], &mut output, &mut stage)
                .is_err()
        );
        assert!(
            geometry
                .barycentric_gradients_into(0, &mut gradients, &mut gradient_stage)
                .is_err()
        );
        assert!(geometry.circumcenter_into(0, &mut center).is_err());
        assert!(dual.vertex(0).is_err());
        assert!(dual.vertices().is_err());
        assert!(dual.rays().all(|ray| ray.direction().is_err()));
        assert_eq!(output, [41.; 3]);
        assert_eq!(gradients, [42.; 6]);
        assert_eq!(center, [43.; 2]);
    };
}

#[test]
fn canonical_construction_fixture() {
    let values = [0., 0., 2., 0., 0., 2., 2., 2., 0.5, 0.75];
    with_points(&values, 2, |p| {
        let hull = ConvexHull::with_options(p, options(5)).unwrap();
        let delaunay = Delaunay::with_options(p, options(5)).unwrap();
        let dual = Voronoi::with_options(p, options(5)).unwrap();
        let facets: Vec<_> = (0..hull.facet_count())
            .map(|i| hull.facet(i).unwrap().to_vec())
            .collect();
        let simplices: Vec<_> = (0..delaunay.simplex_count())
            .map(|i| delaunay.simplex(i).unwrap().to_vec())
            .collect();
        let adjacency: Vec<_> = (0..delaunay.simplex_count())
            .map(|i| delaunay.neighbors(i).unwrap().to_vec())
            .collect();
        assert_eq!(facets, [[0, 1], [0, 3], [1, 4], [3, 4]]);
        assert_eq!(simplices, [[0, 1, 2], [0, 2, 3], [1, 2, 4], [2, 3, 4]]);
        assert_eq!(
            adjacency,
            [
                [Some(2), Some(1), None],
                [Some(3), None, Some(0)],
                [Some(3), None, Some(0)],
                [None, Some(2), Some(1)]
            ]
        );
        assert_eq!(dual.simplex_vertices(), [0, 1, 2, 3]);
    });
}

#[test]
fn coplanar_cube_keeps_canonical_pulling_subdivisions() {
    let cube: Vec<_> = (0..8)
        .flat_map(|i| [f64::from(i >> 2), f64::from((i >> 1) & 1), f64::from(i & 1)])
        .collect();
    let hull = with_points(&cube, 3, |p| ConvexHull::with_options(p, options(8))).unwrap();
    // Each square face is pulled from its smallest canonical vertex.
    assert_eq!(
        hull_rows(&hull),
        [
            [0, 1, 3],
            [0, 1, 5],
            [0, 2, 3],
            [0, 2, 6],
            [0, 4, 5],
            [0, 4, 6],
            [1, 3, 7],
            [1, 5, 7],
            [2, 3, 7],
            [2, 6, 7],
            [4, 5, 7],
            [4, 6, 7],
        ]
    );
}

fn test_workspace(
    requirements: GeometryWorkspaceRequirements,
) -> Result<GeometryWorkspace, scioxide_core::SciOxideError> {
    let mut workspace = GeometryWorkspace::new();
    workspace.try_reserve(&requirements)?;
    Ok(workspace)
}
fn test_requirements(
    dimension: usize,
    options: GeometryOptions,
) -> Result<GeometryWorkspaceRequirements, scioxide_core::SciOxideError> {
    let mut values = vec![0.; dimension * (dimension + 1)];
    for i in 0..dimension {
        values[(i + 1) * dimension + i] = 1.;
    }
    with_points(&values, dimension, |p| Delaunay::with_options(p, options))?
        .workspace_requirements(())
}
