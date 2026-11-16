//! Public lifecycle, resource and query contracts for tessellation.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Budget, BudgetKind, CancellationToken, ExecutionContext, NdView, ProblemToken, SolveBudgets,
    SolveIdentity, SolveState, StridedLayout,
};
use scioxide_spatial::*;

fn options() -> GeometryOptions {
    GeometryOptions::new().with_limits(
        GeometryLimits::new()
            .with_max_dimension(5)
            .with_max_points(128)
            .with_max_facets(1024)
            .with_max_predicate_terms(10_000_000)
            .with_max_storage_bytes(128 * 1024 * 1024),
    )
}
fn with_points<T>(values: &[f64], dimension: usize, f: impl FnOnce(&PointSet<'_, f64>) -> T) -> T {
    let shape = [values.len() / dimension, dimension];
    let strides = [dimension, 1];
    f(
        &PointSet::new(NdView::new(values, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap(),
    )
}
const SQUARE: [f64; 8] = [0., 0., 1., 0., 1., 1., 0., 1.];
#[test]
fn prepared_queries_execute_and_commit_atomically() {
    with_points(&SQUARE, 2, |points| {
        let hull = ConvexHull::with_options(points, options()).unwrap();
        let delaunay = Delaunay::with_options(points, options()).unwrap();
        let mut workspace = test_workspace(test_requirements(2, options()).unwrap()).unwrap();
        let mut weights = [0.; 3];
        let mut stage = [0.; 3];
        {
            assert!(
                hull.contains_scoped(
                    &[0.3, 0.7],
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential()
                )
                .unwrap()
            );
            let cell = delaunay
                .find_simplex_scoped(
                    &[0.3, 0.7],
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential(),
                )
                .unwrap()
                .unwrap();
            delaunay
                .barycentric_into(cell.simplex, &[0.3, 0.7], &mut weights, &mut stage)
                .unwrap();
        };
        let before = weights;
        assert!(
            delaunay
                .barycentric_into(0, &[f64::NAN, 0.], &mut weights, &mut stage)
                .is_err()
        );
        assert_eq!(weights, before);
        assert!(
            delaunay
                .barycentric_into(usize::MAX, &[0., 0.], &mut weights, &mut stage)
                .is_err()
        );
        assert_eq!(weights, before);
        assert!(
            hull.contains_scoped(
                &[0.],
                &mut workspace,
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
        );
        assert!(
            hull.contains_scoped(
                &[f64::INFINITY, 0.],
                &mut workspace,
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
        );
    });
}

#[test]
fn location_hints_are_allocation_free_across_geometries_and_failures() {
    let before = with_points(&SQUARE, 2, |p| Delaunay::with_options(p, options())).unwrap();
    let extended: Vec<_> = SQUARE.into_iter().chain([0.5, 0.5]).collect();
    let after = with_points(&extended, 2, |p| Delaunay::with_options(p, options())).unwrap();
    let single = with_points(&SQUARE[..6], 2, |p| Delaunay::with_options(p, options())).unwrap();
    let queries = [[0.75, 0.1], [0.25, 0.8], [0.5, 0.5], [0., 0.], [1.5, 1.5]];
    let geometries = [&after, &before, &single, &after];
    let requirements = test_requirements(2, options()).unwrap();
    let expected: Vec<_> = geometries
        .iter()
        .flat_map(|geometry| {
            queries.iter().map(|query| {
                let mut cold = test_workspace(requirements).unwrap();
                geometry
                    .find_simplex_scoped(
                        query,
                        &mut cold,
                        scioxide_core::ExecutionContext::sequential(),
                    )
                    .unwrap()
            })
        })
        .collect();
    let mut workspace = test_workspace(requirements).unwrap();
    let mut too_small = test_workspace(test_requirements(1, options()).unwrap()).unwrap();
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    for (geometry, expected) in geometries.iter().zip(expected.chunks_exact(queries.len())) {
        for (query, &expected) in queries.iter().zip(expected) {
            assert_eq!(
                geometry
                    .find_simplex_scoped(
                        query,
                        &mut workspace,
                        scioxide_core::ExecutionContext::sequential()
                    )
                    .unwrap(),
                expected
            );
            assert!(
                geometry
                    .find_simplex_scoped(
                        query,
                        &mut workspace,
                        ExecutionContext::sequential().with_cancellation(&cancelled)
                    )
                    .is_err()
            );
            assert!(
                geometry
                    .find_simplex_scoped(
                        query,
                        &mut too_small,
                        scioxide_core::ExecutionContext::sequential()
                    )
                    .is_err()
            );
            for invalid in [&[0.][..], &[f64::NAN, 0.][..], &[0., f64::INFINITY][..]] {
                assert!(
                    geometry
                        .find_simplex_scoped(
                            invalid,
                            &mut workspace,
                            scioxide_core::ExecutionContext::sequential()
                        )
                        .is_err()
                );
            }
            assert_eq!(
                geometry
                    .find_simplex_scoped(
                        query,
                        &mut workspace,
                        scioxide_core::ExecutionContext::sequential()
                    )
                    .unwrap(),
                expected
            );
            if let Some(location) = expected {
                let mut weights = [0.; 3];
                let mut scratch = [0.; 3];
                geometry
                    .barycentric_into(location.simplex, query, &mut weights, &mut scratch)
                    .unwrap();
                assert!((weights.iter().sum::<f64>() - 1.).abs() < 1e-14);
            }
        }
    }
}

#[test]
fn all_explicit_limits_fail_before_publishing_topology() {
    with_points(&SQUARE, 2, |points| {
        for (resource, change) in [
            (scioxide_core::quantity!("geometry_points"), 0),
            (scioxide_core::quantity!("geometry_facets"), 1),
            (scioxide_core::quantity!("geometry_predicate_terms"), 2),
            (scioxide_core::quantity!("geometry_storage_bytes"), 3),
            (scioxide_core::quantity!("geometry_dimension"), 4),
        ] {
            let mut o = options();
            let limits = o.limits();
            o = o.with_limits(match change {
                0 => limits.with_max_points(2),
                1 => limits.with_max_facets(2),
                2 => limits.with_max_predicate_terms(0),
                3 => limits.with_max_storage_bytes(1),
                _ => limits.with_max_dimension(1),
            });
            assert!(
                matches!(ConvexHull::with_options(points,o),Err(error) if matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource: r, .. } if r == resource))
            );
        }
        let o = options().with_limits(options().limits().with_max_points(usize::MAX));
        let ordinary = ConvexHull::with_options(points, options()).unwrap();
        let generous = ConvexHull::with_options(points, o).unwrap();
        assert_eq!(
            ordinary.construction_requirements(),
            generous.construction_requirements()
        );
        let mut workspace = GeometryWorkspace::new();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let geometry = Delaunay::build(points, options(), &mut context).unwrap();
        assert!(
            geometry
                .find_simplex_scoped(&[0.25, 0.25], &mut workspace, &mut context)
                .is_err()
        );
    });
}
fn identity() -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(568),
        scioxide_core::algorithm_id!("spatial", "tessellation", 1),
        0,
        0,
    )
}
#[test]
fn context_counts_stops_and_workspace_reuse() {
    with_points(&SQUARE, 2, |points| {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Node, Budget::ONE);
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        assert!(matches!(
            Delaunay::build(points, options(), &mut context),
            Err(ref error) if matches!(error.view(), scioxide_core::SciOxideErrorView::ConvergenceFailure { reason: scioxide_core::ConvergenceReason::ResourceLimit(BudgetKind::Node), diagnostics, .. } if diagnostics.budget().is_some_and(|report| report.consumed() == 1))
        ));
        assert_eq!(context.budget_report(BudgetKind::Node).consumed(), 1);
        let token = CancellationToken::new();
        token.cancel();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential().with_cancellation(&token),
                identity(),
                &mut budgets,
            )
            .unwrap();
        assert!(
            matches ! (& (Delaunay::build(points, options (), & mut context)) , Err (__scioxide_error) if matches ! (__scioxide_error . view () , scioxide_core :: SciOxideErrorView :: ExecutionStopped { .. }))
        );
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let result = Delaunay::build(points, options(), &mut context).unwrap();
        assert_eq!(
            context.budget_report(BudgetKind::Node).consumed(),
            result.provenance().predicate_terms
        );
    });
}

#[test]
fn halfspace_certificate_context_counts_limits_and_reuse() {
    // Non-dyadic vertices with zero components exercise exact equality proofs.
    let rows = [3., 0., -1., -1., 0., 0., 0., 7., -2., 0., -1., 0.];
    with_points(&rows, 3, |points| {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let result =
            HalfspaceIntersection::build(points, &[0.125; 2], options(), &mut context).unwrap();
        let terms = result.provenance().predicate_terms;
        assert_eq!(context.budget_report(BudgetKind::Node).consumed(), terms);
        let ordinary = HalfspaceIntersection::with_options(points, &[0.125; 2], options()).unwrap();
        assert_eq!(ordinary.provenance().predicate_terms, terms);
        assert_eq!(ordinary.vertices(), result.vertices());
        let mut limited = options();
        limited = limited.with_limits(limited.limits().with_max_predicate_terms(terms - 1));
        assert!(matches!(
            HalfspaceIntersection::build(points, &[0.125; 2], limited, &mut context),
            Err(ref error) if matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource, .. } if resource == scioxide_core::quantity!("geometry_predicate_terms"))
        ));
        let token = CancellationToken::new();
        token.cancel();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential().with_cancellation(&token),
                identity(),
                &mut budgets,
            )
            .unwrap();
        let error =
            HalfspaceIntersection::build(points, &[0.125; 2], options(), &mut context).unwrap_err();
        assert!(
            matches!(error, e if e.kind() == scioxide_core::SciOxideErrorKind::ExecutionStopped)
        );
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        let again =
            HalfspaceIntersection::build(points, &[0.125; 2], options(), &mut context).unwrap();
        assert_eq!(again.vertices(), result.vertices());
        assert_eq!(again.provenance().predicate_terms, terms);
    });
}
#[test]
fn borrowed_strides_preserve_delaunay_topology() {
    let data = [
        0., 99., 0., 99., 1., 99., 0., 99., 1., 99., 1., 99., 0., 99., 1., 99.,
    ];
    let shape = [4, 2];
    let strides = [4, 2];
    let points =
        PointSet::new(NdView::new(&data, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    let result = Delaunay::with_options(&points, options()).unwrap();
    let packed = with_points(&SQUARE, 2, |points| {
        Delaunay::with_options(points, options())
    })
    .unwrap();
    same_delaunay(&result, &packed);
}

#[test]
fn barycentric_coordinates_preserve_scalar_range() {
    check_barycentric_range::<f32>();
    check_barycentric_range::<f64>();
}

fn check_barycentric_range<S: scioxide_core::RealScalar>() {
    let zero = S::ZERO;
    let one = S::ONE;
    let largest = S::MAX;
    let half = largest * S::HALF;
    let values = [-one, zero, zero, one, one, zero];
    let shape = [3, 2];
    let strides = [2, 1];
    let points =
        PointSet::new(NdView::new(&values, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    let result = Delaunay::with_options(&points, options()).unwrap();
    for (query, weights) in [
        ([zero, largest], [-half, largest, -half]),
        ([-half, half], [one, half, -half]),
    ] {
        let mut output = [zero; 3];
        let mut scratch = [zero; 3];
        result
            .barycentric_into(0, &query, &mut output, &mut scratch)
            .unwrap();
        for (actual, expected) in output.into_iter().zip(weights) {
            assert!((actual / expected - one).abs() <= S::EPSILON);
        }
    }
}

fn same_hull(a: &ConvexHull<f64>, b: &ConvexHull<f64>) {
    assert_eq!(a.points(), b.points());
    assert_eq!(a.input_mapping(), b.input_mapping());
    assert_eq!(a.representatives(), b.representatives());
    assert_eq!(a.coplanar_faces(), b.coplanar_faces());
    assert_eq!(a.facet_count(), b.facet_count());
    for i in 0..a.facet_count() {
        assert_eq!(a.facet(i).unwrap(), b.facet(i).unwrap());
        assert_eq!(a.neighbors(i).unwrap(), b.neighbors(i).unwrap());
    }
}
fn same_delaunay(a: &Delaunay<f64>, b: &Delaunay<f64>) {
    assert_eq!(a.points(), b.points());
    assert_eq!(a.input_mapping(), b.input_mapping());
    assert_eq!(a.simplex_count(), b.simplex_count());
    for i in 0..a.simplex_count() {
        assert_eq!(a.simplex(i).unwrap(), b.simplex(i).unwrap());
        assert_eq!(a.neighbors(i).unwrap(), b.neighbors(i).unwrap());
    }
}
fn check_construction_forms(values: &[f64], dimension: usize) {
    let hull = with_points(values, dimension, |p| {
        ConvexHull::with_options(p, options())
    })
    .unwrap();
    let delaunay =
        with_points(values, dimension, |p| Delaunay::with_options(p, options())).unwrap();
    let dual = with_points(values, dimension, |p| Voronoi::with_options(p, options())).unwrap();
    let scoped_hull = with_points(values, dimension, |p| {
        ConvexHull::build(p, options(), ExecutionContext::sequential())
    })
    .unwrap();
    let scoped_delaunay = with_points(values, dimension, |p| {
        Delaunay::build(p, options(), ExecutionContext::sequential())
    })
    .unwrap();
    let actual = with_points(values, dimension, |p| {
        Voronoi::build(p, options(), ExecutionContext::sequential())
    })
    .unwrap();
    same_hull(&hull, &scoped_hull);
    same_delaunay(&delaunay, &scoped_delaunay);
    same_delaunay(dual.delaunay(), actual.delaunay());
    assert_eq!(dual.simplex_vertices(), actual.simplex_vertices());
    assert_eq!(dual.ridges().len(), actual.ridges().len());
    for (a, b) in dual.ridges().zip(actual.ridges()) {
        assert_eq!(a.sites(), b.sites());
        assert_eq!(a.vertices(), b.vertices());
        assert_eq!(a.is_unbounded(), b.is_unbounded());
    }
    assert!(
        dual.vertices()
            .unwrap()
            .iter()
            .zip(actual.vertices().unwrap())
            .all(|(a, b)| (a - b).abs() <= 1e-12 * (1. + a.abs()))
    );
}

#[test]
fn canonical_topology_agrees_between_construction_forms() {
    check_construction_forms(&[0., 2., -1., 1., 2.], 1);
    let hypercube: Vec<f64> = (0..16)
        .rev()
        .flat_map(|i| (0..4).map(move |axis| f64::from((i >> axis) & 1)))
        .collect();
    check_construction_forms(&hypercube, 4);
    check_construction_forms(&[1., 0., 0., 1., -1., 0., 0., -1., 1., 0.], 2);
    // Cospherical vertices with coplanar supporting faces in reverse order.
    let cube: Vec<f64> = (0..8)
        .rev()
        .flat_map(|i| (0..3).map(move |axis| f64::from((i >> axis) & 1)))
        .collect();
    check_construction_forms(&cube, 3);
    let mut boundary = cube;
    boundary.extend([0.5, 0., 0., 0.5, 0.5, 0., 0., 0., 0., 0.5, 0.5, 0.5]);
    check_construction_forms(&boundary, 3);
    // Include non-extreme points on the boundary and in the interior.
    check_construction_forms(&[0., 0., 1., 0., 0., 1., 2., 0., 2., 2., 0., 2., 1., 1.], 2);
}

#[test]
fn strict_duplicates_and_construction_facet_limits() {
    let strict = options().with_degeneracy(GeometryDegeneracy::Strict);
    let duplicate: Vec<_> = SQUARE
        .into_iter()
        .chain(SQUARE[..2].iter().copied())
        .collect();
    with_points(&duplicate, 2, |p| {
        assert_eq!(
            ConvexHull::with_options(p, strict).unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
    });
    let limited = options().with_limits(options().limits().with_max_facets(3));
    let old = with_points(&SQUARE[..6], 2, |p| ConvexHull::with_options(p, limited)).unwrap();
    with_points(&SQUARE, 2, |p| {
        assert!(
            matches!(ConvexHull::with_options(p, limited), Err(ref error)
            if matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource, .. }
                if resource == scioxide_core::quantity!("geometry_facets")))
        );
    });
    assert_eq!(old.facet_count(), 3);
}

#[test]
fn storage_boundary_and_iteration_budget_are_explicit() {
    let build = |maximum| {
        let o = options().with_limits(options().limits().with_max_storage_bytes(maximum));
        with_points(&SQUARE, 2, |p| ConvexHull::with_options(p, o))
    };
    let bytes = build(options().limits().max_storage_bytes())
        .unwrap()
        .construction_requirements()
        .bytes();
    assert_eq!(build(bytes).unwrap().facet_count(), 4);
    assert!(
        matches!(build(bytes - 1), Err(error) if matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource, .. } if resource == scioxide_core::quantity!("geometry_storage_bytes")))
    );
    with_points(&SQUARE, 2, |points| {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        assert!(context.record_iteration().unwrap().is_granted());
        let accepted = context.accepted_state();
        ConvexHull::build(points, options(), &mut context).unwrap();
        assert_eq!(context.budget_report(BudgetKind::Iteration).consumed(), 1);
        assert_eq!(context.counts().own().iterations(), 1);
        assert!(context.counts().nested().iterations() > 0);
        assert_eq!(context.accepted_state(), accepted);
    });
}
#[test]
fn strict_policy_rejects_cocircular_and_cospherical_ties() {
    let mut o = options();
    o = o.with_degeneracy(GeometryDegeneracy::Strict);
    with_points(&SQUARE, 2, |p| {
        assert!(matches!(
            Delaunay::with_options(p, o),
            Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::InvalidInput
        ))
    });
    let cube: Vec<f64> = (0..8)
        .flat_map(|i| (0..3).map(move |axis| f64::from((i >> axis) & 1)))
        .collect();
    with_points(&cube, 3, |p| {
        assert!(matches!(
            Delaunay::with_options(p, o),
            Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::InvalidInput
        ))
    });
}

// Strict rejects only published cells with a shared empty sphere. Upper lifted
// facets and non-empty cocircular quadruples leave a unique triangulation, and
// neither acceptance nor provenance may depend on row order.
#[test]
fn strict_policy_accepts_unique_triangulations_in_every_row_order() {
    let strict = options().with_degeneracy(GeometryDegeneracy::Strict);
    let fixtures: [&[[f64; 2]]; 3] = [
        &[[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0.3, 0.4]],
        &[[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [0.1, 0.05]],
        &[[0., 0.], [2., 0.], [2., 2.], [0., 2.], [1., 1.], [0.5, 1.7]],
    ];
    for sites in fixtures {
        let mut reference = None;
        for permutation in permutations(sites.len()) {
            let values: Vec<f64> = permutation.iter().flat_map(|&i| sites[i]).collect();
            let summary = with_points(&values, 2, |p| {
                let strict = Delaunay::with_options(p, strict).unwrap();
                let deterministic = Delaunay::with_options(p, options()).unwrap();
                assert_eq!(strict.points(), deterministic.points());
                let cells: Vec<Vec<usize>> = (0..strict.simplex_count())
                    .map(|i| strict.simplex(i).unwrap().to_vec())
                    .collect();
                let provenance = deterministic.provenance();
                (
                    cells,
                    strict.points().to_vec(),
                    provenance.predicate_terms,
                    provenance.resolved_ties,
                )
            });
            assert_eq!(reference.get_or_insert_with(|| summary.clone()), &summary);
        }
    }
    // Cocircular hull corners with an empty circle remain non-unique.
    for permutation in permutations(4) {
        let values: Vec<f64> = permutation
            .iter()
            .flat_map(|&i| [SQUARE[2 * i], SQUARE[2 * i + 1]])
            .collect();
        with_points(&values, 2, |p| {
            assert_eq!(
                Delaunay::with_options(p, strict).unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::InvalidInput
            );
            assert_eq!(
                Voronoi::with_options(p, strict).unwrap_err().kind(),
                scioxide_core::SciOxideErrorKind::InvalidInput
            );
        });
    }
}
fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut output = Vec::new();
    for shorter in permutations(n - 1) {
        for position in 0..=shorter.len() {
            let mut next = shorter.clone();
            next.insert(position, n - 1);
            output.push(next);
        }
    }
    output
}
// Coplanar hull faces inside one cospherical group give one ray each, with
// every equidistant site of the face.
#[test]
fn cube_voronoi_rays_follow_geometric_hull_faces() {
    let cube: Vec<f64> = (0..8)
        .flat_map(|i| (0..3).map(move |axis| f64::from((i >> axis) & 1)))
        .collect();
    with_points(&cube, 3, |p| {
        let dual = Voronoi::with_options(p, options()).unwrap();
        assert_eq!(dual.vertex_count(), 1);
        assert_eq!(dual.rays().len(), 6);
        let mut directions = Vec::new();
        for ray in dual.rays() {
            assert_eq!(ray.vertex(), 0);
            assert_eq!(ray.sites().len(), 4);
            assert!(ray.sites().windows(2).all(|pair| pair[0] < pair[1]));
            let direction = ray.direction().unwrap();
            let axis = direction.iter().position(|x| x.abs() == 1.).unwrap();
            for site in ray.sites() {
                let coordinate = dual.delaunay().points()[3 * site + axis];
                assert_eq!(coordinate, if direction[axis] > 0. { 1. } else { 0. });
            }
            directions.push((axis, direction[axis] > 0.));
        }
        directions.sort_unstable();
        directions.dedup();
        assert_eq!(directions.len(), 6);
    });
}
fn subspace(n: usize, dimension: usize, rank: usize) -> Vec<f64> {
    (0..n)
        .flat_map(|i| {
            let coordinates = [i as f64, (i % 31) as f64, (i % 17) as f64];
            (0..dimension).map(move |axis| {
                if axis < rank {
                    coordinates[axis]
                } else {
                    coordinates[..rank]
                        .iter()
                        .enumerate()
                        .fold(axis as f64, |sum, (j, x)| sum + (2 << j) as f64 * x)
                }
            })
        })
        .collect()
}

#[test]
fn ten_thousand_lower_dimensional_sites_report_rank_or_unboundedness() {
    let mut o = options();
    o = o.with_limits(o.limits().with_max_points(10_000));
    o = o.with_limits(o.limits().with_max_facets(16));
    o = o.with_limits(o.limits().with_max_storage_bytes(256 * 1024 * 1024));
    for d in 2..=4 {
        for rank in 1..d {
            let values = subspace(10_000, d, rank);
            with_points(&values, d, |p| {
                assert!(matches!(
                    ConvexHull::with_options(p, o),
                    Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::Singular
                ));
                assert!(matches!(
                    Delaunay::with_options(p, o),
                    Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::Singular
                ));
                assert!(matches!(
                    Voronoi::with_options(p, o),
                    Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::Singular
                ));
            });
            // Remove the affine translation so the normals span only `rank`
            // axes. The origin strictly satisfies every offset -1 inequality.
            let planes: Vec<_> = values
                .chunks_exact(d)
                .flat_map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(axis, x)| x - if axis < rank { 0. } else { axis as f64 })
                        .chain(core::iter::once(-1.))
                })
                .collect();
            with_points(&planes, d + 1, |p| {
                assert_eq!(
                    HalfspaceIntersection::with_options(p, &vec![0.; d], o)
                        .unwrap()
                        .kind(),
                    IntersectionKind::Unbounded
                );
            });
        }
    }
    let circle: Vec<_> = (0..10_000)
        .flat_map(|i| {
            let angle = f64::from(i) * core::f64::consts::TAU / 10_000.;
            [angle.cos(), angle.sin(), 0.]
        })
        .collect();
    with_points(&circle, 3, |p| {
        assert!(matches!(
            SphericalVoronoi::with_options(
                p,
                [0.; 3],
                1.,
                scioxide_core::Tolerance::new(1e-12, 1e-12).unwrap(),
                o
            ),
            Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::Singular
        ));
    });
}

#[test]
fn spanning_last_point_recovers_one_shot_construction() {
    for d in 2..=4 {
        let mut values = subspace(24, d, 1);
        for axis in 1..d {
            with_points(&values, d, |p| {
                assert_eq!(
                    ConvexHull::with_options(p, options()).unwrap_err().kind(),
                    scioxide_core::SciOxideErrorKind::Singular
                );
                assert_eq!(
                    Delaunay::with_options(p, options()).unwrap_err().kind(),
                    scioxide_core::SciOxideErrorKind::Singular
                );
                assert_eq!(
                    Voronoi::with_options(p, options()).unwrap_err().kind(),
                    scioxide_core::SciOxideErrorKind::Singular
                );
            });
            let mut point = values[..d].to_vec();
            point[axis] += 1.;
            values.extend(point);
        }
        with_points(&values, d, |p| {
            assert!(
                ConvexHull::with_options(p, options())
                    .unwrap()
                    .facet_count()
                    > 0
            );
            assert!(
                Delaunay::with_options(p, options())
                    .unwrap()
                    .simplex_count()
                    > 0
            );
            assert!(Voronoi::with_options(p, options()).unwrap().vertex_count() > 0);
        });
    }
}

#[test]
fn deficient_construction_accounts_context_work_and_predicate_limits() {
    let values = subspace(128, 3, 2);
    with_points(&values, 3, |p| {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        assert!(matches!(
            Delaunay::build(p, options(), &mut context),
            Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::Singular
        ));
        let terms = context.budget_report(BudgetKind::Node).consumed();
        assert_eq!(context.budget_report(BudgetKind::Node).consumed(), terms);
        for limit in [terms - 1, terms] {
            let mut o = options();
            o = o.with_degeneracy(GeometryDegeneracy::Strict);
            o = o.with_limits(o.limits().with_max_predicate_terms(limit));
            let result = Delaunay::with_options(p, o);
            if limit < terms {
                assert!(matches!(
                    result,
                    Err(ref error) if matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource, .. } if resource == scioxide_core::quantity!("geometry_predicate_terms"))
                ));
            } else {
                assert!(
                    matches!(result, Err(ref error) if error.kind() == scioxide_core::SciOxideErrorKind::Singular)
                );
            }
        }
        let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Node, Budget::ONE);
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        assert!(matches!(
            Delaunay::build(p, options(), &mut context),
            Err(ref error) if matches!(error.view(), scioxide_core::SciOxideErrorView::ConvergenceFailure { reason: scioxide_core::ConvergenceReason::ResourceLimit(BudgetKind::Node), .. })
        ));
        assert_eq!(context.budget_report(BudgetKind::Node).consumed(), 1);
        let token = CancellationToken::new();
        token.cancel();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential().with_cancellation(&token),
                identity(),
                &mut budgets,
            )
            .unwrap();
        assert!(
            matches!(Delaunay::build(p, options(), &mut context), Err(e) if matches!(e.view(), scioxide_core::SciOxideErrorView::ExecutionStopped { .. }))
        );
        assert_eq!(context.budget_report(BudgetKind::Node).consumed(), 0);
    });
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

#[test]
fn canonical_limits_counts_and_publication_are_transactional() {
    let values: Vec<_> = (0..6)
        .flat_map(|i| {
            (0..6).flat_map(move |j| [f64::from(i), f64::from(j), f64::from(i * i + j * j)])
        })
        .collect();
    with_points(&values, 3, |points| {
        let old = ConvexHull::with_options(points, options()).unwrap();
        let count = old.provenance().canonical_work;
        assert!(count > 1);
        let limited =
            |work| options().with_limits(options().limits().with_max_canonical_work(work));
        for maximum in [0, 1, count - 1] {
            assert!(
                matches!(ConvexHull::with_options(points, limited(maximum)), Err(error)
                if matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit {
                    resource, required, limit,
                } if resource == scioxide_core::quantity!("geometry_canonical_work")
                    && limit == maximum && required > limit))
            );
        }
        same_hull(
            &old,
            &ConvexHull::with_options(points, limited(count)).unwrap(),
        );

        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
            .unwrap();
        assert!(context.record_iteration().unwrap().is_granted());
        let accepted = context.accepted_state();
        assert!(
            matches!(ConvexHull::build(points, limited(1), &mut context), Err(error)
            if matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource, .. }
                if resource == scioxide_core::quantity!("geometry_canonical_work")))
        );
        assert_eq!(context.accepted_state(), accepted);
        assert_eq!(context.counts().own().iterations(), 1);
        let geometry = ConvexHull::build(points, options(), &mut context).unwrap();
        same_hull(&old, &geometry);
        assert_eq!(context.accepted_state(), accepted);
        let total = geometry.provenance().predicate_terms + count;
        drop(context);

        for maximum in [total - 1, total] {
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Node, Budget::try_from(maximum).unwrap());
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), identity(), &mut budgets)
                .unwrap();
            let accepted = context.accepted_state();
            let result = ConvexHull::build(points, options(), &mut context);
            if maximum < total {
                assert!(matches!(result, Err(error)
                    if matches!(error.view(), scioxide_core::SciOxideErrorView::ConvergenceFailure {
                        reason: scioxide_core::ConvergenceReason::ResourceLimit(BudgetKind::Node), diagnostics, ..
                    } if diagnostics.budget().is_some())));
            } else {
                same_hull(&old, &result.unwrap());
                assert_eq!(context.budget_report(BudgetKind::Node).consumed(), total);
            }
            assert_eq!(context.accepted_state(), accepted);
        }
        // Failed candidates do not alter an earlier published snapshot.
        same_hull(&old, &ConvexHull::with_options(points, options()).unwrap());
    });
}

#[test]
fn spherical_canonical_pulling_inherits_the_work_limit() {
    let values: Vec<_> = (0..8)
        .flat_map(|i| (0..3).map(move |axis| if (i >> axis) & 1 == 0 { -1. } else { 1. }))
        .collect();
    with_points(&values, 3, |points| {
        let tolerance = scioxide_core::Tolerance::new(1e-12, 1e-12).unwrap();
        let radius = 3_f64.sqrt();
        let geometry =
            SphericalVoronoi::with_options(points, [0.; 3], radius, tolerance, options()).unwrap();
        assert_eq!(geometry.vertices().len() / 3, 6);
        for site in 0..8 {
            assert_eq!(geometry.region(site).unwrap().len(), 3);
            assert_eq!(geometry.neighbors(site).unwrap().count(), 3);
        }
        let count = geometry.provenance().canonical_work;
        assert!(count > 0);
        let limited = options().with_limits(options().limits().with_max_canonical_work(count - 1));
        assert!(
            matches!(SphericalVoronoi::with_options(points, [0.; 3], radius, tolerance, limited), Err(error)
            if matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource, .. }
                if resource == scioxide_core::quantity!("geometry_canonical_work")))
        );
    });
}
