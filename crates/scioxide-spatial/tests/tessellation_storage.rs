//! Construction admission and retained query capacity; no timing gates.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Capacity, ExecutionContext, NdView, SciOxideErrorKind, StridedLayout, Tolerance,
};
use scioxide_spatial::{
    ConvexHull, Delaunay, GeometryLimits, GeometryOptions, GeometryWorkspace,
    HalfspaceIntersection, PointSet, SphericalVoronoi, Voronoi,
};

fn points<T>(values: &[f64], dimension: usize, run: impl FnOnce(&PointSet<'_, f64>) -> T) -> T {
    let shape = [values.len() / dimension, dimension];
    let strides = [dimension, 1];
    run(
        &PointSet::new(NdView::new(values, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap(),
    )
}
fn random(count: usize, dimension: usize) -> Vec<f64> {
    let mut state = 1124_u64;
    (0..count * dimension)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        })
        .collect()
}

#[test]
fn discovered_storage_boundary_and_generous_limits() {
    let values = random(128, 3);
    points(&values, 3, |p| {
        let options = GeometryOptions::new();
        let base = ConvexHull::with_options(p, options).unwrap();
        let bytes = base.construction_requirements().bytes();
        let exact = options.with_limits(options.limits().with_max_storage_bytes(bytes));
        assert_eq!(
            ConvexHull::with_options(p, exact).unwrap().facet_count(),
            base.facet_count()
        );
        let short = exact.with_limits(exact.limits().with_max_storage_bytes(bytes - 1));
        assert_eq!(
            ConvexHull::with_options(p, short).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        let large = options.with_limits(
            options
                .limits()
                .with_max_facets(usize::MAX)
                .with_max_points(usize::MAX),
        );
        assert_eq!(
            ConvexHull::with_options(p, large)
                .unwrap()
                .construction_requirements()
                .bytes(),
            bytes
        );
    });
}

#[test]
fn query_requirements_and_retained_capacity_are_separate_from_construction() {
    let values = random(8, 5);
    let hull = points(&values, 5, ConvexHull::new).unwrap();
    let requirements = hull.workspace_requirements(()).unwrap();
    assert!(requirements.bytes() < hull.construction_requirements().bytes());
    let mut workspace = GeometryWorkspace::new();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    workspace.try_reserve(&requirements).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    assert!(
        hull.contains_scoped(&values[..5], &mut workspace, ExecutionContext::sequential())
            .unwrap()
    );
    let planar = [0., 0., 1., 0., 0., 1.];
    let options = GeometryOptions::new()
        .with_limits(GeometryLimits::new().with_max_storage_bytes(requirements.bytes() - 1));
    let small = points(&planar, 2, |p| ConvexHull::with_options(p, options)).unwrap();
    assert_eq!(
        workspace
            .check(&small.workspace_requirements(()).unwrap())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
}

#[test]
fn every_product_enforces_its_discovered_peak() {
    fn boundary(build: impl Fn(GeometryOptions) -> Result<usize, scioxide_core::SciOxideError>) {
        let options = GeometryOptions::new();
        let bytes = build(options).unwrap();
        let exact = options.with_limits(options.limits().with_max_storage_bytes(bytes));
        assert_eq!(build(exact).unwrap(), bytes);
        let short = exact.with_limits(exact.limits().with_max_storage_bytes(bytes - 1));
        assert_eq!(
            build(short).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
    }
    let planar = random(64, 2);
    points(&planar, 2, |p| {
        boundary(|o| ConvexHull::with_options(p, o).map(|r| r.construction_requirements().bytes()));
        boundary(|o| Delaunay::with_options(p, o).map(|r| r.construction_requirements().bytes()));
        boundary(|o| Voronoi::with_options(p, o).map(|r| r.construction_requirements().bytes()));
    });
    let octahedron = [
        1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1.,
    ];
    points(&octahedron, 3, |p| {
        boundary(|o| {
            SphericalVoronoi::with_options(p, [0.; 3], 1., Tolerance::new(1e-12, 1e-12).unwrap(), o)
                .map(|r| r.construction_requirements().bytes())
        })
    });
    let planes = [1., 0., -1., -1., 0., -1., 0., 1., -1., 0., -1., -1.];
    points(&planes, 3, |p| {
        boundary(|o| {
            HalfspaceIntersection::with_options(p, &[0.; 2], o)
                .map(|r| r.construction_requirements().bytes())
        })
    });
}
