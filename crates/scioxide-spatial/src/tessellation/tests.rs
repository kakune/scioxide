use super::work::fixed_rows;
use super::*;
use super::{
    cloud::Cloud,
    storage::{Storage, retained},
    work::Work,
};
use crate::PointSet;
use crate::predicates::fixed_upper;
use alloc::vec;
use scioxide_core::{Quantity, RealScalar};
mod canonical;
mod canonical_reference;
mod incidence;
// Test-local identity for fixtures that call crate-private kernels directly.

fn options() -> GeometryOptions {
    GeometryOptions::new().with_limits(
        GeometryLimits::new()
            .with_max_dimension(8)
            .with_max_points(256)
            .with_max_facets(4096)
            .with_max_predicate_terms(100_000_000)
            .with_max_storage_bytes(1024 * 1024 * 1024),
    )
}

#[test]
fn query_requirement_arithmetic_overflow_is_structured() {
    let options =
        GeometryOptions::new().with_limits(GeometryLimits::new().with_max_dimension(usize::MAX));
    assert_eq!(
        test_requirements(usize::MAX, options).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::SizeOverflow
    );
    assert_eq!(
        test_requirements(usize::MAX / 4, options)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::SizeOverflow
    );
    assert_eq!(
        test_requirements(0, options).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
}

// Independent exhaustive traversal: deliberately does not call the production
// classifier, seed index, walk, or fallback. Exact predicates remain the oracle.
fn exhaustive_location<S: RealScalar>(
    geometry: &Delaunay<S>,
    point: &[S],
) -> (Option<SimplexLocation>, usize) {
    let d = geometry.dimension();
    let mut exact = ExactWorkspace::new(d + 2).unwrap();
    let mut rows = vec![0.; (d + 1) * d];
    let mut terms = 0;
    for simplex in 0..geometry.simplex_count() {
        let vertices = geometry.simplex(simplex).unwrap();
        let mut boundary = false;
        let mut contained = true;
        for replaced in 0..=d {
            for (row, &vertex) in vertices.iter().enumerate() {
                let source = if row == replaced {
                    point
                } else {
                    geometry.cloud.point(vertex)
                };
                for (axis, value) in source.iter().enumerate() {
                    rows[row * d + axis] = value.try_to_f64_exact().unwrap();
                }
            }
            let sign = if d == 2 || d == 3 {
                terms += 1;
                if fixed_rows::<false>(&rows, d, false).is_none() {
                    terms += fixed_upper(d, false);
                }
                fixed_rows::<true>(&rows, d, false).unwrap_or_default()
            } else {
                exact
                    .sign(&rows, d, false, &mut |count| {
                        terms += count;
                        Ok(())
                    })
                    .unwrap()
            };
            boundary |= sign == 0;
            if sign != 0 && sign != geometry.topology.facets.at(simplex).inside {
                contained = false;
                break;
            }
        }
        if contained {
            return (Some(SimplexLocation { simplex, boundary }), terms);
        }
    }
    (None, terms)
}

fn random_coordinates(count: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        })
        .collect()
}

fn check_locations<S: RealScalar>(sites: &[f64], d: usize, queries: &[f64]) {
    let storage = Storage::new(usize::MAX, 0).unwrap();

    let sites: Vec<S> = sites
        .iter()
        .map(|&x| S::try_from_f64_lossy(x).unwrap())
        .collect();
    let queries: Vec<S> = queries
        .iter()
        .map(|&x| S::try_from_f64_lossy(x).unwrap())
        .collect();
    let geometry = with_points(&sites, d, |p| Delaunay::with_options(p, options())).unwrap();
    let requirements = test_requirements(d, options()).unwrap();
    let mut warm = test_workspace(requirements).unwrap();
    for query in queries.chunks_exact(d).chain(queries.chunks_exact(d).rev()) {
        let expected = exhaustive_location(&geometry, query).0;
        let mut cold = test_workspace(requirements).unwrap();
        assert_eq!(
            geometry
                .find_simplex_scoped(
                    query,
                    &mut cold,
                    scioxide_core::ExecutionContext::sequential()
                )
                .unwrap(),
            expected
        );
        assert_eq!(
            geometry
                .find_simplex_scoped(
                    query,
                    &mut warm,
                    scioxide_core::ExecutionContext::sequential()
                )
                .unwrap(),
            expected
        );
        // Every start simplex must agree, including traversal toward an outside
        // point from the opposite side of the convex hull.
        for seed in [0, geometry.simplex_count() - 1] {
            let mut work = Work::new(
                &mut cold,
                d,
                options(),
                None,
                scioxide_core::argument!("points"),
                &storage,
            )
            .unwrap();
            assert_eq!(
                geometry
                    .walk(query, seed, geometry.simplex_count(), &mut work)
                    .unwrap(),
                expected
            );
        }
        if let Some(location) = expected {
            let mut weights = vec![S::ZERO; d + 1];
            let mut scratch = weights.clone();
            geometry
                .barycentric_into(location.simplex, query, &mut weights, &mut scratch)
                .unwrap();
            let tolerance = 256. * S::EPSILON.try_to_f64_exact().unwrap();
            for (axis, &x) in query.iter().enumerate() {
                let actual: f64 = weights
                    .iter()
                    .zip(geometry.simplex(location.simplex).unwrap())
                    .map(|(&weight, &vertex)| {
                        weight.try_to_f64_exact().unwrap()
                            * geometry.cloud.point(vertex)[axis]
                                .try_to_f64_exact()
                                .unwrap()
                    })
                    .sum();
                assert!((actual - x.try_to_f64_exact().unwrap()).abs() <= tolerance);
            }
        }
    }
}

#[test]
fn location_matches_exhaustive_random_and_degenerate_clouds() {
    for d in 1..=4 {
        let sites = random_coordinates(12 * d, 0x1024 + d as u64);
        let queries: Vec<_> = random_coordinates(24 * d, 0x6245 + d as u64)
            .iter()
            .map(|x| 1.5 * x - 0.25)
            .collect();
        check_locations::<f64>(&sites, d, &queries);
        check_locations::<f32>(&sites, d, &queries);

        let mut sites = cube::<f64>(d);
        sites.extend_from_within(..d); // duplicate input site
        let mut queries = sites.clone();
        queries.extend(vec![0.5; d]); // exact shared diagonal/face
        queries.extend(vec![-1.; d]);
        queries.extend(vec![2.; d]);
        queries.extend((0..d).map(|axis| if axis == 0 { 0.5 } else { 0. })); // hull edge
        check_locations::<f64>(&sites, d, &queries);
        check_locations::<f32>(&sites, d, &queries);
    }
    let mut sites = vec![0.; 30];
    for axis in 0..5 {
        sites[(axis + 1) * 5 + axis] = 1.;
    }
    let mut queries = sites.clone();
    queries.extend([0.125; 5]);
    queries.extend([1.; 5]);
    check_locations::<f64>(&sites, 5, &queries);
    check_locations::<f32>(&sites, 5, &queries);
    // Thin but still admitted affine frames and collinear hull sites.
    let sites = [0., 0., 0.5, 0., 1., 0., 0., 1e-5, 1., 1e-5];
    let queries = [0.25, 0., 0.25, 1e-6, 0.75, 5e-6, 0.5, -1e-6];
    check_locations::<f64>(&sites, 2, &queries);
    check_locations::<f32>(&sites, 2, &queries);
}

#[test]
fn location_boundary_ties_and_bounded_fallback_share_the_budget() {
    let storage = Storage::new(usize::MAX, 0).unwrap();

    let sites = cube::<f64>(2);
    let geometry = with_points(&sites, 2, |p| Delaunay::with_options(p, options())).unwrap();
    let mut workspace = test_workspace(test_requirements(2, options()).unwrap()).unwrap();
    for point in [[0.5, 0.5], [0., 0.], [0.5, 0.], [0.2, 0.7], [-0.1, 0.5]] {
        let expected = exhaustive_location(&geometry, &point).0;
        for steps in [0, 1, geometry.simplex_count()] {
            let mut work = Work::new(
                &mut workspace,
                2,
                options(),
                None,
                scioxide_core::argument!("points"),
                &storage,
            )
            .unwrap();
            assert_eq!(
                geometry
                    .walk(&point, geometry.simplex_count() - 1, steps, &mut work)
                    .unwrap(),
                expected
            );
        }
    }
    let expected = Some(SimplexLocation {
        simplex: 0,
        boundary: true,
    });
    assert_eq!(exhaustive_location(&geometry, &[0.5, 0.5]).0, expected);
    let mut limited = options();
    limited.limits.max_predicate_terms = 9; // two filtered signs and one exact zero
    let mut work = Work::new(
        &mut workspace,
        2,
        limited,
        None,
        scioxide_core::argument!("points"),
        &storage,
    )
    .unwrap();
    assert_eq!(
        geometry.walk(&[0.5, 0.5], 1, 2, &mut work).unwrap(),
        expected
    );
    assert_eq!(workspace.predicate_terms(), 9);
    limited.limits.max_predicate_terms = 8;
    let mut work = Work::new(
        &mut workspace,
        2,
        limited,
        None,
        scioxide_core::argument!("points"),
        &storage,
    )
    .unwrap();
    assert!(matches!(
        geometry.walk(&[0.5, 0.5], 1, 0, &mut work),
        Err(ref error) if matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource, .. } if resource == scioxide_core::quantity!("geometry_predicate_terms"))
    ));
    assert_eq!(
        geometry
            .find_simplex_scoped(
                &[0.5, 0.5],
                &mut workspace,
                scioxide_core::ExecutionContext::sequential()
            )
            .unwrap(),
        expected
    );
}

#[test]
fn indexed_location_reduces_predicate_work_for_cold_and_reused_queries() {
    let mut sites = cube::<f64>(2);
    sites.extend(random_coordinates(248, 1024));
    let geometry = with_points(&sites, 2, |p| Delaunay::with_options(p, options())).unwrap();
    let queries = random_coordinates(128, 12345);
    let requirements = test_requirements(2, options()).unwrap();
    let mut workspace = test_workspace(requirements).unwrap();
    let (mut cold_terms, mut warm_terms, mut scan_terms) = (0, 0, 0);
    for point in queries.chunks_exact(2) {
        let (expected, terms) = exhaustive_location(&geometry, point);
        assert!(expected.is_some_and(|location| !location.boundary));
        scan_terms += terms;
        assert_eq!(
            geometry
                .find_simplex_scoped(
                    point,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential()
                )
                .unwrap(),
            expected
        );
        warm_terms += workspace.predicate_terms();
        let mut cold = test_workspace(requirements).unwrap();
        assert_eq!(
            geometry
                .find_simplex_scoped(
                    point,
                    &mut cold,
                    scioxide_core::ExecutionContext::sequential()
                )
                .unwrap(),
            expected
        );
        cold_terms += cold.predicate_terms();
    }
    assert!(
        cold_terms < scan_terms / 4,
        "{cold_terms} versus {scan_terms}"
    );
    assert!(
        warm_terms < scan_terms / 4,
        "{warm_terms} versus {scan_terms}"
    );
}

#[test]
fn location_seeds_preserve_extreme_range_classification() {
    fn check<S: RealScalar>(exponents: &[i32]) {
        let base = [0., 0., 4., 0., 4., 4., 0., 4., 1., 2., 3., 1.];
        for &exponent in exponents {
            let sites: Vec<S> = base
                .iter()
                .map(|&x| S::try_from_f64_lossy(f64::scalbn(x, exponent)).unwrap())
                .collect();
            let geometry =
                with_points(&sites, 2, |p| Delaunay::with_options(p, options())).unwrap();
            let mut workspace = test_workspace(test_requirements(2, options()).unwrap()).unwrap();
            let scale = S::try_from_f64_lossy(1_f64.scalbn(exponent)).unwrap();
            for query in [
                [scale, scale],
                [S::ZERO, S::ZERO],
                [S::MAX, S::MAX],
                [-S::MAX, S::ZERO],
            ] {
                assert_eq!(
                    geometry
                        .find_simplex_scoped(
                            &query,
                            &mut workspace,
                            scioxide_core::ExecutionContext::sequential()
                        )
                        .unwrap(),
                    exhaustive_location(&geometry, &query).0
                );
            }
        }
    }
    check::<f64>(&[-1070, -500, 0, 500, 1000]);
    check::<f32>(&[-145, -60, 0, 60, 120]);
}
fn cube<S: RealScalar>(d: usize) -> Vec<S> {
    (0..1 << d)
        .flat_map(|i| {
            (0..d).map(move |j| S::try_from_f64_lossy(f64::from((i >> j & 1) as u8)).unwrap())
        })
        .collect()
}
fn qualify<S: RealScalar>(d: usize) {
    let input = cube::<S>(d);
    with_points(&input, d, |points| {
        let hull = ConvexHull::with_options(points, options())?;
        let mut workspace = test_workspace(test_requirements(d, options())?)?;
        for p in input.chunks_exact(d) {
            assert!(hull.contains_scoped(
                p,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential()
            )?);
        }
        assert!(hull.contains_scoped(
            &vec![S::try_from_f64_lossy(0.5).unwrap(); d],
            &mut workspace,
            scioxide_core::ExecutionContext::sequential()
        )?);
        assert!(!hull.contains_scoped(
            &vec![S::try_from_f64_lossy(2.).unwrap(); d],
            &mut workspace,
            scioxide_core::ExecutionContext::sequential()
        )?);
        for i in 0..hull.facet_count() {
            for &neighbor in hull.neighbors(i)?.iter().flatten() {
                assert!(hull.neighbors(neighbor)?.contains(&Some(i)));
            }
        }
        let delaunay = Delaunay::with_options(points, options())?;
        assert!(delaunay.simplex_count() > 0);
        for p in input.chunks_exact(d) {
            assert!(
                delaunay
                    .find_simplex_scoped(
                        p,
                        &mut workspace,
                        scioxide_core::ExecutionContext::sequential()
                    )?
                    .is_some()
            );
        }
        let query = vec![S::try_from_f64_lossy(0.37).unwrap(); d];
        let location = delaunay
            .find_simplex_scoped(
                &query,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential(),
            )?
            .unwrap();
        let mut weights = vec![S::ZERO; d + 1];
        let mut scratch = weights.clone();
        delaunay.barycentric_into(location.simplex, &query, &mut weights, &mut scratch)?;
        let eps = S::EPSILON.try_to_f64_exact().unwrap() * 64.;
        assert!(
            (weights
                .iter()
                .map(|x| x.try_to_f64_exact().unwrap())
                .sum::<f64>()
                - 1.)
                .abs()
                <= eps
        );
        for (axis, coordinate) in query.iter().enumerate() {
            let reconstructed = weights
                .iter()
                .zip(delaunay.simplex(location.simplex)?)
                .map(|(w, &v)| {
                    w.try_to_f64_exact().unwrap()
                        * delaunay.points()[v * d + axis].try_to_f64_exact().unwrap()
                })
                .sum::<f64>();
            assert!((reconstructed - coordinate.try_to_f64_exact().unwrap()).abs() <= eps);
        }
        // Independent cube oracle: all cell vertices lie on the sphere centered
        // at (1/2,...,1/2), and every input site has that same squared radius.
        for i in 0..delaunay.simplex_count() {
            let center = delaunay
                .transforms
                .at(i)?
                .circumcenter()
                .collect::<Result<Vec<_>>>()?;
            for x in center {
                assert!((x - 0.5).abs() <= eps);
            }
        }
        if d == 2 {
            delaunay.triangle_mesh()?;
        }
        if d == 3 {
            delaunay.tetrahedron_mesh()?;
        }
        let dual = Voronoi::with_options(points, options())?;
        assert_eq!(dual.vertices()?.len(), d);
        assert!(dual.simplex_vertices().iter().all(|&v| v == 0));
        for i in 0..1 << d {
            assert!(dual.region_is_unbounded(i)?);
        }
        Ok(())
    })
    .unwrap();
}
#[test]
fn cube_f32() {
    for d in 2..=4 {
        qualify::<f32>(d);
    }
}
#[test]
fn cube_f64() {
    for d in 2..=4 {
        qualify::<f64>(d);
    }
}
#[test]
fn dimension_five_simplex() {
    let d = 5;
    let mut points = vec![0.; (d + 1) * d];
    for i in 0..d {
        points[(i + 1) * d + i] = 1.;
    }
    with_points(&points, d, |p| {
        assert_eq!(Delaunay::with_options(p, options())?.simplex_count(), 1);
        assert_eq!(ConvexHull::with_options(p, options())?.facet_count(), 6);
        Ok(())
    })
    .unwrap();
}
#[test]
fn halfspace_cubes() {
    for d in 2..=4 {
        let mut planes = vec![0.; 2 * d * (d + 1)];
        for i in 0..d {
            planes[2 * i * (d + 1) + i] = 1.;
            planes[2 * i * (d + 1) + d] = -1.;
            planes[(2 * i + 1) * (d + 1) + i] = -1.;
        }
        with_points(&planes, d + 1, |p| {
            let result = HalfspaceIntersection::with_options(p, &vec![0.25; d], options())?;
            assert_eq!(result.kind(), IntersectionKind::Bounded);
            assert_eq!(result.vertices().len(), (1 << d) * d);
            for vertex in 0..1 << d {
                assert_eq!(result.neighbors(vertex)?.count(), d);
                for neighbor in result.neighbors(vertex)? {
                    assert!(result.neighbors(neighbor)?.any(|v| v == vertex));
                }
            }
            for x in result.vertices() {
                assert!(*x == 0. || *x == 1.);
            }
            Ok(())
        })
        .unwrap();
    }
}
#[test]
fn sphere_octahedron() {
    let points = [
        1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1.,
    ];
    with_points(&points, 3, |p| {
        let sphere = SphericalVoronoi::with_options(
            p,
            [0.; 3],
            1.,
            scioxide_core::Tolerance::from_machine_epsilon(),
            options(),
        )?;
        assert_eq!(sphere.vertices().len(), 24);
        for site in 0..6 {
            assert_eq!(sphere.neighbors(site)?.count(), 4);
        }
        assert_eq!(sphere.center(), [0.; 3]);
        assert_eq!(sphere.radius(), 1.);
        for &area in sphere.areas() {
            assert!((area - 2. * core::f64::consts::PI / 3.).abs() < 1e-13);
        }
        Ok(())
    })
    .unwrap();
}
#[test]
fn limits_and_degeneracy() {
    with_points(&[0., 0., 1., 0., 0., 1., 0., 0.], 2, |p| {
        let hull = ConvexHull::with_options(p, options())?;
        assert_eq!(hull.input_mapping()[0], hull.input_mapping()[3]);
        let mut strict = options();
        strict.degeneracy = GeometryDegeneracy::Strict;
        assert!(matches!(
            ConvexHull::with_options(p, strict),
            Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::InvalidInput
        ));
        let mut limited = options();
        limited.limits.max_predicate_terms = 0;
        assert!(matches!(
            ConvexHull::with_options(p, limited),
            Err(ref error) if matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource, .. } if resource == scioxide_core::quantity!("geometry_predicate_terms"))
        ));
        Ok(())
    })
    .unwrap();
    with_points(&[0., 0., 1., 1., 2., 2.], 2, |p| {
        assert!(matches!(
            ConvexHull::with_options(p, options()),
            Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::Singular
        ));
        Ok(())
    })
    .unwrap();
}

#[test]
fn permutations_scaling_and_translation() {
    let base = [0., 0., 4., 0., 4., 4., 0., 4., 1., 2., 3., 1.];
    let expected = with_points(&base, 2, |p| Delaunay::with_options(p, options())).unwrap();
    let cells: Vec<_> = (0..expected.simplex_count())
        .map(|i| expected.simplex(i).unwrap().to_vec())
        .collect();
    for exponent in [-1070, -500, 0, 500, 1000] {
        let values: Vec<_> = base.iter().map(|&x| f64::scalbn(x, exponent)).collect();
        let actual = with_points(&values, 2, |p| Delaunay::with_options(p, options())).unwrap();
        assert_eq!(
            cells,
            (0..actual.simplex_count())
                .map(|i| actual.simplex(i).unwrap().to_vec())
                .collect::<Vec<_>>()
        );
    }
    let values: Vec<_> = base
        .chunks_exact(2)
        .rev()
        .flat_map(|p| p.iter().map(|x| x + 1e14))
        .collect();
    let actual = with_points(&values, 2, |p| Delaunay::with_options(p, options())).unwrap();
    assert_eq!(
        cells,
        (0..actual.simplex_count())
            .map(|i| actual.simplex(i).unwrap().to_vec())
            .collect::<Vec<_>>()
    );
}
#[test]
fn nd_predicate_full_range_and_work_exhaustion() {
    for d in 1..=5 {
        let mut workspace = crate::predicates::nd::ExactWorkspace::new(d + 2).unwrap();
        for exponent in [-1074, 0, 1023] {
            let mut rows = vec![0.; d * (d + 1)];
            for i in 0..d {
                rows[(i + 1) * d + i] = 1_f64.scalbn(exponent);
            }
            let sign = workspace.sign(&rows, d, false, &mut |_| Ok(())).unwrap();
            assert_eq!(sign, if d % 2 == 0 { 1 } else { -1 });
        }
        let rows = vec![1.; d * (d + 1)];
        let mut terms = 0;
        assert!(
            workspace
                .sign(&rows, d, false, &mut |count| {
                    terms += count;
                    if terms > 1 {
                        Err(SciOxideError::size_overflow(scioxide_core::quantity!(
                            "test_work_admission"
                        )))
                    } else {
                        Ok(())
                    }
                })
                .is_err()
        );
    }
}
#[test]
fn halfspace_failures_redundancy_and_unboundedness() {
    with_points(&[1., 0., -1., -1., 0., 0.], 3, |p| {
        assert_eq!(
            HalfspaceIntersection::with_options(p, &[0.5, 0.], options())?.kind(),
            IntersectionKind::Unbounded
        );
        Ok(())
    })
    .unwrap();
    with_points(
        &[
            1., 0., -1., -1., 0., 0., 0., 1., -1., 0., -1., 0., 1., 1., -2., 1., 0., -1.,
        ],
        3,
        |p| {
            let result = HalfspaceIntersection::with_options(p, &[0.3, 0.7], options())?;
            assert_eq!(result.vertices().len(), 8);
            assert!((0..4).any(|i| result.active_halfspaces(i).unwrap().len() == 4));
            assert!(HalfspaceIntersection::with_options(p, &[2., 2.], options()).is_err());
            assert!(HalfspaceIntersection::with_options(p, &[0., 0.], options()).is_err());
            Ok(())
        },
    )
    .unwrap();
    let planes = [1_f32, 0., -1., -1., 0., 0., 0., 1., -1., 0., -1., 0.];
    with_points(&planes, 3, |p| {
        assert_eq!(
            HalfspaceIntersection::with_options(p, &[0.5, 0.5], options())?
                .vertices()
                .len(),
            8
        );
        Ok(())
    })
    .unwrap();
}

// Exact original residuals are a deliberately exhaustive TEST oracle. The
// production verifier uses only the defining system and active incidences.
fn check_halfspace_residuals<S: RealScalar>(rows: &[S], result: &HalfspaceIntersection<S>) {
    let d = result.dimension();
    let epsilon = 128. * S::EPSILON.try_to_f64_exact().unwrap() * (d + 1) as f64;
    for (vertex, coordinates) in result.vertices().chunks_exact(d).enumerate() {
        let active = result.active_halfspaces(vertex).unwrap();
        for (plane, row) in rows.chunks_exact(d + 1).enumerate() {
            let mut residual = scioxide_core::RangeAccumulator::<f64>::new();
            let b = row[d].try_to_f64_exact().unwrap();
            residual.add(b).unwrap();
            let mut magnitude = b.abs();
            for (a, x) in row.iter().zip(coordinates) {
                let a = a.try_to_f64_exact().unwrap();
                let x = x.try_to_f64_exact().unwrap();
                residual.add_product(a, x).unwrap();
                magnitude += (a * x).abs();
            }
            let residual = residual.finish().unwrap();
            assert!(
                residual <= epsilon * magnitude,
                "vertex={vertex}, plane={plane}"
            );
            if active.contains(&plane) {
                assert!(residual.abs() <= epsilon * magnitude);
            }
        }
    }
}

fn halfspace_cloud(n: usize) -> Vec<f64> {
    let mut state = 1117_u64;
    let mut rows = Vec::new();
    for _ in 0..n {
        let point = core::array::from_fn::<_, 3, _>(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        });
        let length = point.iter().map(|x| x * x).sum::<f64>().sqrt();
        rows.extend(point.map(|x| x / length));
        rows.push(-1.);
    }
    for axis in 0..3 {
        for sign in [-1., 1.] {
            let mut row = [0.; 4];
            row[axis] = sign;
            row[3] = -2.;
            rows.extend(row);
        }
    }
    rows
}

#[test]
fn halfspace_regular_polygons_near_axes() {
    fn check<S: RealScalar>() {
        for m in 3..=12 {
            for phase in [0_f64, 15., 22.5, 30., 45., 60., 90.] {
                let rows: Vec<S> = (0..m)
                    .flat_map(|k| {
                        let angle =
                            phase.to_radians() + 2. * core::f64::consts::PI * k as f64 / m as f64;
                        [angle.cos(), angle.sin(), -1.].map(|x| S::try_from_f64_lossy(x).unwrap())
                    })
                    .collect();
                with_points(&rows, 3, |p| {
                    let result = HalfspaceIntersection::with_options(p, &[S::ZERO; 2], options());
                    assert!(result.is_ok(), "m={m}, phase={phase}: {result:?}");
                    let result = result.unwrap();
                    assert_eq!(result.kind(), IntersectionKind::Bounded);
                    assert_eq!(result.vertices().len(), 2 * m);
                    check_halfspace_residuals(&rows, &result);
                    Ok(())
                })
                .unwrap();
            }
        }
    }
    check::<f64>();
    check::<f32>();
}

#[test]
fn halfspace_exact_zero_and_epsilon_scale_coordinates() {
    fn check<S: RealScalar>() {
        for shift in [0., S::EPSILON.try_to_f64_exact().unwrap()] {
            // The top vertex is (shift, 1); all coefficients are exact dyadics.
            let rows: Vec<S> = [
                1.,
                1.,
                -1. - shift,
                -1.,
                1.,
                -1. + shift,
                -1.,
                -1.,
                -1. + shift,
                1.,
                -1.,
                -1. - shift,
            ]
            .map(|x| S::try_from_f64_exact(x).unwrap())
            .into();
            with_points(&rows, 3, |p| {
                let result = HalfspaceIntersection::with_options(p, &[S::ZERO; 2], options())?;
                assert_eq!(result.vertices().len(), 8);
                let top = result
                    .vertices()
                    .chunks_exact(2)
                    .max_by(|a, b| a[1].partial_cmp(&b[1]).unwrap())
                    .unwrap();
                assert!(
                    (top[1].try_to_f64_exact()? - 1.).abs()
                        <= 4. * S::EPSILON.try_to_f64_exact()?
                );
                assert!(
                    (top[0].try_to_f64_exact()? - shift).abs()
                        <= 4. * S::EPSILON.try_to_f64_exact()? * shift
                );
                check_halfspace_residuals(&rows, &result);
                Ok(())
            })
            .unwrap();
        }
    }
    check::<f64>();
    check::<f32>();
}

#[test]
fn halfspace_nearly_tight_nonactive_planes_and_duplicate_incidence() {
    fn check<S: RealScalar>(near: S) {
        let mut rows: Vec<S> = [3., 0., -1., -1., 0., 0., 0., 7., -2., 0., -1., 0., 3., 7.]
            .into_iter()
            .map(|x| S::try_from_f64_exact(x).unwrap())
            .collect();
        rows.push(-near);
        // Original duplicate indices, including rescaled duplicates, are retained.
        rows.extend([3., 0., -1., 6., 0., -2.].map(|x| S::try_from_f64_exact(x).unwrap()));
        with_points(&rows, 3, |p| {
            let result = HalfspaceIntersection::with_options(
                p,
                &[S::try_from_f64_exact(0.125)?, S::try_from_f64_exact(0.125)?],
                options(),
            )?;
            assert_eq!(result.vertices().len(), 8);
            check_halfspace_residuals(&rows, &result);
            for i in 0..4 {
                let active = result.active_halfspaces(i)?;
                assert!(!active.contains(&4));
                assert_eq!(active.contains(&0), active.contains(&5));
                assert_eq!(active.contains(&0), active.contains(&6));
            }
            assert!(result.active_halfspaces(usize::MAX).is_err());
            Ok(())
        })
        .unwrap();
    }
    check(3_f64.next_up());
    check(3_f32.next_up());
}

#[test]
fn halfspace_certificate_work_grows_subquadratically() {
    let mut previous = None;
    for n in [64, 256, 1024] {
        let rows = halfspace_cloud(n);
        let mut o = options();
        o.limits.max_points = n + 6;
        o.limits.max_facets = 8 * n;
        o.limits.max_storage_bytes = usize::MAX;
        let (result, residual_checks) = measure_residual_checks(|| {
            with_points(&rows, 4, |p| {
                HalfspaceIntersection::with_options(p, &[0.; 3], o)
            })
            .unwrap()
        });
        let vertices = result.vertices().len() / 3;
        let incidence: usize = (0..vertices)
            .map(|v| result.active_halfspaces(v).unwrap().len())
            .sum();
        // Each vertex has at most d² inverse-defect dot products and three
        // passes of d original + d preconditioned residuals, then active checks.
        assert!(residual_checks <= (3 * 3 + 6 * 3) * vertices + incidence);
        let terms = result.provenance().predicate_terms;
        if let Some((previous_terms, previous_checks)) = previous {
            assert!(terms < 8 * previous_terms, "n={n}: {terms} predicate terms");
            assert!(
                residual_checks < 6 * previous_checks,
                "n={n}: {} residual checks",
                residual_checks
            );
        }
        previous = Some((terms, residual_checks));
        if n <= 256 {
            check_halfspace_residuals(&rows, &result);
        }
    }
}

#[test]
fn halfspace_ill_conditioned_vertex_is_rejected() {
    let rows = [
        1.,
        3.,
        -1.,
        3.,
        9. + 3. * 2_f64.powi(-49),
        -(3. + 2_f64.powi(-50)),
        -1.,
        0.,
        -1.,
        0.,
        -1.,
        -1.,
    ];
    with_points(&rows, 3, |p| {
        assert!((HalfspaceIntersection::with_options(p, &[0.; 2], options())).is_err());
        Ok(())
    })
    .unwrap();
}

#[test]
fn halfspace_extreme_row_scales_keep_original_constraints() {
    let base = [3., 0., -1., -1., 0., 0., 0., 7., -2., 0., -1., 0.];
    let rows: Vec<f64> = base
        .chunks_exact(3)
        .zip([
            2_f64.powi(900),
            2_f64.powi(-900),
            2_f64.powi(-900),
            2_f64.powi(900),
        ])
        .flat_map(|(row, scale)| row.iter().map(move |x| x * scale))
        .collect();
    with_points(&rows, 3, |p| {
        let result = HalfspaceIntersection::with_options(p, &[0.125; 2], options())?;
        assert_eq!(result.vertices().len(), 8);
        check_halfspace_residuals(&rows, &result);
        Ok(())
    })
    .unwrap();
}

#[test]
fn halfspace_subnormal_coordinates_need_exact_zero_radius_proofs() {
    let tiny = f64::from_bits(1);
    let rows = [1., 0., -tiny, -1., 0., -tiny, 0., 3., -1., 0., -1., -1.];
    with_points(&rows, 3, |p| {
        let result = HalfspaceIntersection::with_options(p, &[0.; 2], options())?;
        assert_eq!(result.vertices().len(), 8);
        assert!(
            result
                .vertices()
                .chunks_exact(2)
                .all(|v| v[0].abs() == tiny)
        );
        check_halfspace_residuals(&rows, &result);
        Ok(())
    })
    .unwrap();
}
#[test]
fn spherical_degeneracies_and_both_widths() {
    let input = [
        1_f32, 0., 0., -1., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1., 1., 0., 0.,
    ];
    with_points(&input, 3, |p| {
        let sphere = SphericalVoronoi::with_options(
            p,
            [0.; 3],
            1.,
            scioxide_core::Tolerance::from_machine_epsilon(),
            options(),
        )?;
        assert_eq!(sphere.input_mapping()[0], sphere.input_mapping()[6]);
        assert!((sphere.areas().iter().sum::<f32>() - 4. * core::f32::consts::PI).abs() < 1e-5);
        let mut strict = options();
        strict.degeneracy = GeometryDegeneracy::Strict;
        assert!(matches!(
            SphericalVoronoi::with_options(
                p,
                [0.; 3],
                1.,
                scioxide_core::Tolerance::from_machine_epsilon(),
                strict
            ),
            Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::InvalidInput
        ));
        assert!(
            SphericalVoronoi::with_options(
                p,
                [0.; 3],
                2.,
                scioxide_core::Tolerance::from_machine_epsilon(),
                options()
            )
            .is_err()
        );
        Ok(())
    })
    .unwrap();
    with_points(
        &[1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1., 0.],
        3,
        |p| {
            assert!(matches!(
                SphericalVoronoi::with_options(
                    p,
                    [0.; 3],
                    1.,
                    scioxide_core::Tolerance::from_machine_epsilon(),
                    options()
                ),
                Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::Singular
            ));
            Ok(())
        },
    )
    .unwrap();
}

#[test]
fn spherical_scale_and_unrepresentable_areas() {
    let axes = [
        1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 1., 0., 0., -1.,
    ];
    for radius in [1e-100, 2., 1e100] {
        let values: Vec<_> = axes.iter().map(|x| x * radius).collect();
        with_points(&values, 3, |p| {
            let sphere = SphericalVoronoi::with_options(
                p,
                [0.; 3],
                radius,
                scioxide_core::Tolerance::from_machine_epsilon(),
                options(),
            )?;
            for &area in sphere.areas() {
                assert!(((area / radius) / radius - 2. * core::f64::consts::PI / 3.).abs() < 1e-13);
            }
            Ok(())
        })
        .unwrap();
    }
    for radius in [1e-200, 1e200] {
        let values: Vec<_> = axes.iter().map(|x| x * radius).collect();
        with_points(&values, 3, |p| {
            assert!(
                (SphericalVoronoi::with_options(
                    p,
                    [0.; 3],
                    radius,
                    scioxide_core::Tolerance::from_machine_epsilon(),
                    options()
                ))
                .is_err()
            );
            Ok(())
        })
        .unwrap();
    }
}

pub(super) fn test_workspace(
    requirements: GeometryWorkspaceRequirements,
) -> Result<GeometryWorkspace> {
    GeometryWorkspace::prepared(requirements)
}
pub(super) fn test_requirements(
    dimension: usize,
    options: GeometryOptions,
) -> Result<GeometryWorkspaceRequirements> {
    GeometryWorkspaceRequirements::checked(
        dimension,
        options,
        scioxide_core::argument!("dimension"),
    )
}

#[test]
fn filtered_and_exact_stages_have_precise_local_work_boundaries() {
    let storage = Storage::new(usize::MAX, 0).unwrap();
    for d in 2..=8 {
        for lifted in [false, true] {
            let order = d + 1 + usize::from(lifted);
            let upper = (1..=order).product::<usize>() * if lifted { d } else { 1 };
            let mut workspace = test_workspace(test_requirements(d, options()).unwrap()).unwrap();
            // Unit-axis sphere with an interior last row, or the standard simplex.
            let mut rows = vec![0.; order * d];
            for axis in 0..d {
                rows[axis * d + axis] = 1.;
            }
            if lifted {
                rows[d * d] = -1.;
            }
            for degenerate in [false, true] {
                if degenerate {
                    rows.fill(0.);
                }
                let expected = if degenerate { 1 + upper } else { 1 };
                for maximum in [0, expected - 1, expected] {
                    let limited =
                        options().with_limits(options().limits().with_max_predicate_terms(maximum));
                    let mut work = Work::new(
                        &mut workspace,
                        d,
                        limited,
                        None,
                        scioxide_core::argument!("points"),
                        &storage,
                    )
                    .unwrap();
                    work.workspace.rows[..rows.len()].copy_from_slice(&rows);
                    let result = work.raw(order, d, lifted);
                    if maximum < expected {
                        assert!(matches!(result, Err(error) if matches!(error.view(),
                            scioxide_core::SciOxideErrorView::ResourceLimit { resource, required, limit }
                            if resource == scioxide_core::quantity!("geometry_predicate_terms")
                                && required == maximum + 1 && limit == maximum)));
                    } else {
                        assert_eq!(result.unwrap() == 0, degenerate);
                    }
                    assert_eq!(work.workspace.predicate_terms(), maximum);
                }
            }
        }
    }
}

#[cfg(test)]
pub(in crate::tessellation) fn with_points<S: RealScalar, T>(
    values: &[S],
    dimension: usize,
    operation: impl FnOnce(&PointSet<'_, S>) -> Result<T>,
) -> Result<T> {
    let shape = [values.len() / dimension, dimension];
    let strides = [dimension, 1];
    let layout = scioxide_core::StridedLayout::new(&shape, &strides)?;
    let view = scioxide_core::NdView::new(values, layout)?;
    operation(&PointSet::new(view)?)
}

std::thread_local! {
    static RESIDUAL_CHECKS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}
pub(in crate::tessellation) fn record_residual_check() {
    RESIDUAL_CHECKS.with(|count| count.set(count.get() + 1));
}
fn measure_residual_checks<T>(run: impl FnOnce() -> T) -> (T, usize) {
    RESIDUAL_CHECKS.with(|count| count.set(0));
    let result = run();
    (result, RESIDUAL_CHECKS.with(core::cell::Cell::get))
}
