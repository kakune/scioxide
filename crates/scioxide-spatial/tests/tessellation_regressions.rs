//! Geometric faces and local approximation failures.
#![allow(clippy::unwrap_used)]

use scioxide_core::{ExecutionContext, NdView, RealScalar, SciOxideErrorKind, StridedLayout};
use scioxide_spatial::{Delaunay, GeometryOptions, PointSet, Voronoi};
use scioxide_spatial::{GeometryDegeneracy, HalfspaceIntersection, SphericalVoronoi};
use std::collections::BTreeSet;

#[test]
fn halfspace_shape_errors_report_dimensions() {
    use scioxide_core::SciOxideErrorView;
    with_points(&[1., -1.], 2, |p| {
        for scoped in [false, true] {
            let error = if scoped {
                HalfspaceIntersection::build(
                    p,
                    &[],
                    GeometryOptions::new(),
                    ExecutionContext::sequential(),
                )
            } else {
                HalfspaceIntersection::new(p, &[])
            }
            .unwrap_err();
            assert!(
                matches!(error.view(), SciOxideErrorView::DimensionMismatch { expected, actual }
                if expected == [1] && actual == [2])
            );
        }
    });
    with_points(&[1., 0., -1.], 3, |p| {
        for scoped in [false, true] {
            let error = if scoped {
                HalfspaceIntersection::build(
                    p,
                    &[0.],
                    GeometryOptions::new(),
                    ExecutionContext::sequential(),
                )
            } else {
                HalfspaceIntersection::new(p, &[0.])
            }
            .unwrap_err();
            assert!(
                matches!(error.view(), SciOxideErrorView::DimensionMismatch { expected, actual }
                if expected == [2] && actual == [3])
            );
        }
    });
}

// Narrowing an approximate construction out of range is a numerical failure,
// and index errors name the queried quantity.
#[test]
fn approximate_overflow_and_index_errors_are_classified() {
    use scioxide_core::{SciOxideErrorView, Tolerance};
    // The vertex x = 1e40 is exact and certified in binary64 but exceeds f32.
    let rows = [1e-30_f32, 0., -1e10, -1., 0., 0., 0., 1., -1., 0., -1., 0.];
    with_points(&rows, 3, |p| {
        let error = HalfspaceIntersection::new(p, &[1.0, 0.5]).unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);
    });
    let octahedron = [
        1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1.,
    ];
    with_points(&octahedron, 3, |p| {
        let sphere =
            SphericalVoronoi::new(p, [0.; 3], 1., Tolerance::new(1e-12, 1e-12).unwrap()).unwrap();
        assert!(matches!(sphere.neighbors(6).err().unwrap().view(),
            SciOxideErrorView::OutOfBounds { collection, .. }
                if collection == scioxide_core::quantity!("sphere_region")));
    });
}

// The dual's provenance covers its own grouping and ridge stages.
#[test]
fn voronoi_provenance_includes_dual_work() {
    let grid: Vec<f64> = (0..64)
        .flat_map(|i| [f64::from(i % 8), f64::from(i / 8)])
        .collect();
    with_points(&grid, 2, |p| {
        let dual = Voronoi::new(p).unwrap();
        let primal = dual.delaunay().provenance();
        let whole = dual.provenance();
        assert_eq!(whole.unique_points, primal.unique_points);
        assert!(whole.predicate_terms > primal.predicate_terms);
        assert!(whole.canonical_work > primal.canonical_work);
        let limited = GeometryOptions::new().with_limits(
            GeometryOptions::new()
                .limits()
                .with_max_canonical_work(primal.canonical_work + 1),
        );
        assert_eq!(
            Voronoi::with_options(p, limited).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
    });
}

#[test]
fn spherical_region_error_describes_geometric_degeneracy() {
    // The extra site is on an octahedron edge and within the admitted radial
    // tolerance. It has no full-dimensional spherical Voronoi region.
    let coordinates = [
        1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1., 0.5, 0.5, 0.,
    ];
    with_points(&coordinates, 3, |p| {
        for policy in [
            GeometryDegeneracy::Deterministic,
            GeometryDegeneracy::Strict,
        ] {
            for scoped in [false, true] {
                let tolerance = scioxide_core::Tolerance::new(0.5, 0.).unwrap();
                let options = GeometryOptions::new().with_degeneracy(policy);
                let error = if scoped {
                    SphericalVoronoi::build(
                        p,
                        [0.; 3],
                        1.,
                        tolerance,
                        options,
                        ExecutionContext::sequential(),
                    )
                } else {
                    SphericalVoronoi::with_options(p, [0.; 3], 1., tolerance, options)
                }
                .unwrap_err();
                assert!(
                    matches!(error.view(), scioxide_core::SciOxideErrorView::InvalidInput { argument, detail, .. }
                    if argument == scioxide_core::argument!("points")
                    && detail == "spherical Voronoi region requires at least three distinct vertices"),
                    "{error}"
                );
            }
        }
    });
}

fn with_points<S: RealScalar, T>(
    values: &[S],
    d: usize,
    run: impl FnOnce(&PointSet<'_, S>) -> T,
) -> T {
    let shape = [values.len() / d, d];
    let strides = [d, 1];
    run(
        &PointSet::new(NdView::new(values, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap(),
    )
}

fn thin_simplex<S: RealScalar>(d: usize, height: f64) {
    let raw: &[f64] = if d == 2 {
        &[0., 0., 2., 0., 1., height]
    } else {
        &[0., 0., 0., 2., 0., 0., 0., 2., 0., 1., 1., height]
    };
    let values: Vec<S> = raw
        .iter()
        .map(|&x| S::try_from_f64_lossy(x).unwrap())
        .collect();
    let dual = with_points(&values, d, Voronoi::new).unwrap();
    let direct = with_points(&values, d, Delaunay::new).unwrap();
    let geometry = dual.delaunay();
    assert_eq!(direct.simplex(0).unwrap(), geometry.simplex(0).unwrap());
    assert_eq!(geometry.simplex_count(), 1);
    let mut workspace = geometry.workspace(()).unwrap();
    assert!(
        geometry
            .find_simplex_scoped(&values[..d], &mut workspace, ExecutionContext::sequential())
            .unwrap()
            .is_some()
    );
    let sentinel = S::try_from_usize_exact(41).unwrap();
    let mut output = vec![sentinel; d + 1];
    let mut scratch = output.clone();
    assert_eq!(
        geometry
            .barycentric_into(0, &values[..d], &mut output, &mut scratch)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::NumericalInstability
    );
    assert!(output.iter().all(|&x| x == sentinel));
    let mut gradients = vec![sentinel; (d + 1) * d];
    let mut stage = gradients.clone();
    assert_eq!(
        geometry
            .barycentric_gradients_into(0, &mut gradients, &mut stage)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::NumericalInstability
    );
    assert!(gradients.iter().all(|&x| x == sentinel));
    let mut center = vec![sentinel; d];
    assert_eq!(
        geometry
            .circumcenter_into(0, &mut center)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::NumericalInstability
    );
    assert!(center.iter().all(|&x| x == sentinel));
    assert_eq!(dual.vertex_count(), 1);
    assert_eq!(
        dual.vertex(0).unwrap_err().kind(),
        SciOxideErrorKind::NumericalInstability
    );
    assert!(dual.vertices().is_err());
    assert_eq!(
        dual.vertex(1).unwrap_err().kind(),
        SciOxideErrorKind::OutOfBounds
    );
    for ray in dual.rays() {
        assert!(ray.direction().is_err());
    }
    for site in 0..d + 1 {
        assert!(dual.region_vertices(site).unwrap().eq([0]));
    }
}

#[test]
fn thin_frames_never_reject_exact_topology() {
    thin_simplex::<f32>(2, 1e-7);
    thin_simplex::<f64>(2, 1e-7);
    thin_simplex::<f32>(3, 1e-9);
    thin_simplex::<f64>(3, 1e-9);
    // Normalization loses this difference, although the exact simplex has rank.
    thin_simplex::<f64>(3, f64::from_bits(1));
}

#[test]
fn approximate_failures_are_local_to_a_simplex_and_vertex() {
    let values = [0., 0., 2., 0., 1., 1e-9, 0., 2., 2., 2.];
    let dual = with_points(&values, 2, Voronoi::new).unwrap();
    let geometry = dual.delaunay();
    let mut failures = 0;
    let mut successes = 0;
    for simplex in 0..geometry.simplex_count() {
        let mut weights = [41.; 3];
        let mut scratch = weights;
        match geometry.barycentric_into(simplex, &[0.5, 0.5], &mut weights, &mut scratch) {
            Ok(()) => {
                successes += 1;
                assert!((weights.iter().sum::<f64>() - 1.).abs() < 1e-12);
            }
            Err(error) => {
                failures += 1;
                assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);
                assert_eq!(weights, [41.; 3]);
            }
        }
    }
    assert!(failures > 0 && successes > 0);
    assert!(dual.vertices().is_err());
    assert!((0..dual.vertex_count()).any(|i| dual.vertex(i).is_ok()));
    assert!((0..dual.vertex_count()).any(|i| dual.vertex(i).is_err()));
}

#[test]
fn jittered_lattices_publish_delaunay_and_voronoi_at_all_reported_amplitudes() {
    for amplitude in [5e-7, 5e-10, 5e-13] {
        let mut state = 1915_u64;
        let mut jitter = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            amplitude * (2. * ((state >> 11) as f64 / (1_u64 << 53) as f64) - 1.)
        };
        let mut values = Vec::new();
        for x in 0..64 {
            for y in 0..64 {
                values.extend([f64::from(x) / 63. + jitter(), f64::from(y) / 63. + jitter()]);
            }
        }
        let dual = with_points(&values, 2, Voronoi::new).unwrap();
        let geometry = dual.delaunay();
        assert!(geometry.simplex_count() > 7900);
        let mut workspace = geometry.workspace(()).unwrap();
        assert!(
            geometry
                .find_simplex_scoped(&[0.5, 0.5], &mut workspace, ExecutionContext::sequential())
                .unwrap()
                .is_some()
        );
        assert!(dual.ridges().len() > geometry.simplex_count());
    }
}

fn ridge_count(values: &[f64], d: usize) -> usize {
    let dual = with_points(values, d, Voronoi::new).unwrap();
    let count = dual.ridges().len();
    for ridge in dual.ridges() {
        assert!(ridge.vertices().windows(2).all(|pair| pair[0] < pair[1]));
        for site in ridge.sites() {
            let vertices: Vec<_> = dual.region_vertices(site).unwrap().collect();
            assert!(ridge.vertices().iter().all(|v| vertices.contains(v)));
        }
    }
    count
}

#[test]
fn geometric_ridges_match_analytic_grid_counts() {
    assert_eq!(ridge_count(&[0., 0., 1., 0., 0., 1., 1., 1.], 2), 4);
    assert_eq!(ridge_count(&[1., 0., 0., 1., -1., 0., 0., -1.], 2), 4);
    assert_eq!(
        ridge_count(
            &[
                1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1.
            ],
            3
        ),
        12
    );
    for d in [2, 3] {
        let cube: Vec<_> = (0..1 << d)
            .flat_map(|i| (0..d).map(move |axis| f64::from((i >> axis) & 1)))
            .collect();
        assert_eq!(ridge_count(&cube, d), d * (1 << (d - 1)));
        let grid: Vec<_> = (0..3 * (1 << (d - 1)))
            .flat_map(|i| {
                (0..d).map(move |axis| {
                    if axis == 0 {
                        (i % 3) as f64
                    } else {
                        (((i / 3) >> (axis - 1)) & 1) as f64
                    }
                })
            })
            .collect();
        assert_eq!(
            ridge_count(&grid, d),
            2 * (1 << (d - 1)) + (d - 1) * 3 * (1 << (d - 2))
        );
    }
    let grid: Vec<_> = (0..8)
        .flat_map(|x| (0..8).flat_map(move |y| [f64::from(x), f64::from(y)]))
        .collect();
    assert_eq!(ridge_count(&grid, 2), 112);
}

#[test]
fn generic_ridges_retain_every_delaunay_edge() {
    let mut state = 1916_u64;
    for d in [2, 3] {
        let values: Vec<_> = (0..12 * d)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 11) as f64 / (1_u64 << 53) as f64
            })
            .collect();
        let dual = with_points(&values, d, Voronoi::new).unwrap();
        let mut expected = BTreeSet::new();
        for cell in 0..dual.delaunay().simplex_count() {
            let sites = dual.delaunay().simplex(cell).unwrap();
            for (i, &a) in sites.iter().enumerate() {
                for &b in &sites[i + 1..] {
                    expected.insert([a, b]);
                }
            }
        }
        assert_eq!(
            dual.ridges()
                .map(|ridge| ridge.sites())
                .collect::<BTreeSet<_>>(),
            expected
        );
    }
}

#[test]
fn default_limits_construct_one_hundred_thousand_random_3d_sites() {
    let n = 100_000;
    let mut state = 0x1923_cafe_5eed_u64;
    let values: Vec<_> = (0..3 * n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / 9007199254740992.
        })
        .collect();
    let options = GeometryOptions::new();
    let geometry = with_points(&values, 3, |p| Delaunay::with_options(p, options)).unwrap();
    assert_eq!(geometry.provenance().unique_points, n);
    assert!(geometry.simplex_count() > n);
    assert!(geometry.simplex_count() <= options.limits().max_facets());
    assert!(geometry.provenance().predicate_terms <= options.limits().max_predicate_terms());
    assert!(geometry.construction_requirements().bytes() <= options.limits().max_storage_bytes());
    for site in 0..n {
        assert!(geometry.incident_simplices(site).unwrap().next().is_some());
    }
}

fn triangle() -> Delaunay<f64> {
    with_points(&[0., 0., 1., 0., 0., 1.], 2, |p| Delaunay::new(p).unwrap())
}

#[test]
fn find_simplex_rejects_wrong_query_dimensions() {
    let mesh = triangle();
    let mut workspace = mesh.workspace(()).unwrap();
    for query in [&[][..], &[0.][..], &[0.; 3][..]] {
        assert_eq!(
            mesh.find_simplex_scoped(query, &mut workspace, ExecutionContext::sequential())
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::DimensionMismatch
        );
    }
}

#[test]
fn hull_contains_rejects_wrong_query_dimensions() {
    with_points(&[0., 0., 1., 0., 0., 1.], 2, |p| {
        let hull = scioxide_spatial::ConvexHull::new(p).unwrap();
        let mut workspace = hull.workspace(()).unwrap();
        for query in [&[][..], &[0.][..], &[0.; 3][..]] {
            assert_eq!(
                hull.contains_scoped(query, &mut workspace, ExecutionContext::sequential())
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::DimensionMismatch
            );
        }
    });
}

#[test]
fn barycentric_rejects_wrong_lengths_without_publishing() {
    let mesh = triangle();
    for (point, output, scratch) in [(1, 3, 3), (3, 3, 3), (2, 2, 3), (2, 4, 3), (2, 3, 2)] {
        let mut result = vec![7.; output];
        let error = mesh
            .barycentric_into(0, &vec![0.; point], &mut result, &mut vec![0.; scratch])
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
        assert_eq!(result, vec![7.; output]);
    }
    let mut result = [0.; 3];
    mesh.barycentric_into(0, &[0.25; 2], &mut result, &mut [0.; 4])
        .unwrap();
    assert!((result.iter().sum::<f64>() - 1.).abs() < 1e-14);
}

#[test]
fn barycentric_gradients_reject_wrong_lengths_without_publishing() {
    let mesh = triangle();
    for (output, scratch) in [(5, 6), (7, 6), (6, 5)] {
        let mut result = vec![7.; output];
        let error = mesh
            .barycentric_gradients_into(0, &mut result, &mut vec![0.; scratch])
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
        assert_eq!(result, vec![7.; output]);
    }
    let mut result = [0.; 6];
    mesh.barycentric_gradients_into(0, &mut result, &mut [0.; 7])
        .unwrap();
    for axis in 0..2 {
        assert!(
            result
                .chunks_exact(2)
                .map(|row| row[axis])
                .sum::<f64>()
                .abs()
                < 1e-14
        );
    }
}

#[test]
fn circumcenter_rejects_wrong_output_lengths() {
    let mesh = triangle();
    for length in [0, 1, 3] {
        let mut result = vec![7.; length];
        assert_eq!(
            mesh.circumcenter_into(0, &mut result).unwrap_err().kind(),
            SciOxideErrorKind::DimensionMismatch
        );
        assert_eq!(result, vec![7.; length]);
    }
}

#[test]
fn spherical_voronoi_rejects_wrong_point_dimensions() {
    let tolerance = scioxide_core::Tolerance::new(1e-12, 1e-12).unwrap();
    for dimension in [1, 2, 4] {
        with_points(&vec![0.; 4 * dimension], dimension, |p| {
            assert_eq!(
                SphericalVoronoi::new(p, [0.; 3], 1., tolerance)
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::DimensionMismatch
            );
            assert_eq!(
                SphericalVoronoi::build(
                    p,
                    [0.; 3],
                    1.,
                    tolerance,
                    GeometryOptions::new(),
                    ExecutionContext::sequential()
                )
                .unwrap_err()
                .kind(),
                SciOxideErrorKind::DimensionMismatch
            );
        });
    }
}

#[test]
fn simplex_orientation_rejects_wrong_vertex_counts_and_lengths() {
    let mut workspace = scioxide_spatial::SimplexPredicateWorkspace::new(
        2,
        scioxide_core::Budget::try_from(6).unwrap(),
    )
    .unwrap();
    let vertices: [&[f64]; 4] = [&[0., 0.], &[1., 0.], &[0., 1.], &[1., 1.]];
    for count in [0, 2, 4] {
        assert_eq!(
            workspace.orient(&vertices[..count]).unwrap_err().kind(),
            SciOxideErrorKind::DimensionMismatch
        );
    }
    for wrong in [&[][..], &[0.][..], &[0.; 3][..]] {
        assert_eq!(
            workspace
                .orient(&[vertices[0], wrong, vertices[2]])
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::DimensionMismatch
        );
    }
    assert_eq!(
        workspace.orient(&vertices[..3]).unwrap(),
        scioxide_spatial::PredicateSign::Positive
    );
}

#[test]
fn spherical_radial_checks_are_translation_invariant() {
    let center = [1e16_f64, 0.0, 0.0];
    let points = [
        [1e16 + 10.0, 0.0, 0.0],
        [1e16 - 10.0, 0.0, 0.0],
        [1e16, 10.0, 0.0],
        [1e16, -10.0, 0.0],
        [1e16, 0.0, 10.0],
        [1e16, 0.0, -10.0],
        [1e16 + 6.0, 8.0, 0.0],
    ];
    let points = scioxide_spatial::PointSet::from_points(&points).unwrap();
    let diagram = SphericalVoronoi::new(
        &points,
        center,
        10.0,
        scioxide_core::Tolerance::new(1e-12, 1e-12).unwrap(),
    )
    .unwrap();
    let total: f64 = diagram.areas().iter().sum();
    assert!((total / (400.0 * core::f64::consts::PI) - 1.0).abs() < 1e-13);
}

#[test]
fn spherical_radial_validation_polls_before_late_invalid_sites() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    struct Stop(AtomicUsize);
    impl scioxide_core::Cancellation for Stop {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed) >= 3
        }
    }
    let mut sites = vec![[1.0_f64, 0.0, 0.0]; 4096];
    sites[4095] = [2.0, 0.0, 0.0];
    let points = scioxide_spatial::PointSet::from_points(&sites).unwrap();
    let stop = Stop(AtomicUsize::new(0));
    let error = SphericalVoronoi::build(
        &points,
        [0.0; 3],
        1.0,
        scioxide_core::Tolerance::from_sqrt_epsilon(),
        scioxide_spatial::GeometryOptions::new(),
        scioxide_core::ExecutionContext::sequential().with_cancellation(&stop),
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ExecutionStopped
    );
}
