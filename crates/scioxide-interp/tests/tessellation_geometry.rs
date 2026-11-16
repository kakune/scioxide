//! Scale, closed-hull and local Sibson regression batteries.
#![allow(clippy::unwrap_used)]
mod support;

use crate::support::from_f64;
use scioxide_core::{NdView, RealScalar, SciOxideErrorView, StridedLayout};
use scioxide_interp::{
    BilinearInterpolator, CloughTocherInterpolator, CloughTocherOptions, Differentiable,
    ExtrapolationPolicy as P, Interpolant, NaturalNeighborInterpolator,
    NearestNeighborInterpolator, ScatteredValues, SimplexLinearInterpolator,
};
use scioxide_spatial::{Delaunay, GeometryLimits, GeometryOptions, PointSet};

fn mesh<S: RealScalar>(points: &[S], dimension: usize) -> Delaunay<S> {
    let shape = [points.len() / dimension, dimension];
    let strides = [dimension, 1];
    let view = NdView::new(points, StridedLayout::new(&shape, &strides).unwrap()).unwrap();
    Delaunay::with_options(
        &PointSet::new(view).unwrap(),
        GeometryOptions::new().with_limits(
            GeometryLimits::new()
                .with_max_dimension(dimension)
                .with_max_points(shape[0])
                .with_max_facets(32 * shape[0])
                .with_max_predicate_terms(100_000_000),
        ),
    )
    .unwrap()
}
fn square(n: usize) -> Vec<f64> {
    let mut p = vec![0., 0., 1., 0., 0., 1., 1., 1.];
    for i in 1..=n - 4 {
        p.extend([
            (i as f64 * 0.7548776662466927).fract(),
            (i as f64 * 0.5698402909980532).fract(),
        ]);
    }
    p
}
fn battery<S: RealScalar>() {
    for n in [4, 50, 200, 1024] {
        let mut points = square(n);
        if n == 50 {
            for p in points[8..].chunks_exact_mut(2) {
                p[0] = 0.4 + p[0] * 0.2;
                p[1] = 0.4 + p[1] * 0.2;
            }
        }
        let coordinates: Vec<S> = points.iter().copied().map(from_f64).collect();
        let mesh = mesh(&coordinates, 2);
        let values = ScatteredValues::from_input_values(
            &mesh,
            coordinates
                .chunks_exact(2)
                .map(|p| [S::ONE, p[0], p[1], p[0] + S::TWO * p[1]])
                .collect(),
        )
        .unwrap();
        let model = NaturalNeighborInterpolator::new(&mesh, &values).unwrap();
        let mut workspace = model.workspace().unwrap();
        let mut queries: Vec<_> = (0..=16)
            .flat_map(|i| (0..=16).map(move |j| [i as f64 / 16., j as f64 / 16.]))
            .collect();
        for t in [0.125, 0.5, 0.875] {
            for delta in [1e-12, 1e-9, 1e-6] {
                queries.extend([[delta, t], [1. - delta, t], [t, delta], [t, 1. - delta]]);
            }
        }
        // Adjacent floating-point queries must approach a site's Kronecker
        // limit continuously; they must not be mistaken for exact site hits.
        for p in coordinates.chunks_exact(2).skip(4).take(40) {
            queries.push([
                p[0].next_after(S::ONE).try_to_f64_exact().unwrap(),
                p[1].try_to_f64_exact().unwrap(),
            ]);
        }
        queries.extend([[0.546, 0.998], [0.997, 0.354]]);
        for q in queries {
            let q = q.map(from_f64::<S>);
            let mut actual = [S::ZERO; 4];
            let result = model.evaluate_into(q, P::Reject, &mut actual, &mut workspace);
            assert!(result.is_ok(), "n={n}, query={q:?}, error={result:?}");
            let expected = [S::ONE, q[0], q[1], q[0] + S::TWO * q[1]];
            for (a, e) in actual.into_iter().zip(expected) {
                let error = (a - e).abs().try_to_f64_exact().unwrap();
                assert!(
                    error <= 64. * S::EPSILON.try_to_f64_exact().unwrap(),
                    "n={n}, q={q:?}: {a:?} != {e:?}"
                );
            }
        }
    }
}
#[test]
fn closed_hull_battery_f64() {
    battery::<f64>();
}
#[test]
fn closed_hull_battery_f32() {
    battery::<f32>();
}

#[test]
fn planar_projection_is_scale_and_translation_invariant() {
    for scale in [1e-12_f64, 1e-9, 1., 1e6, 1e12] {
        for offset in [0., 1e7, 1e9] {
            if offset + scale == offset {
                continue;
            }
            let actual_scale = (offset + scale) - offset;
            let points: Vec<_> = square(50)
                .into_iter()
                .map(|x| offset + actual_scale * x)
                .collect();
            // Tiny extents at large offsets cannot retain the interior sites.
            if actual_scale < offset.abs() * 128. * f64::EPSILON {
                continue;
            }
            let mesh = mesh(&points, 2);
            let values = ScatteredValues::from_input_values(
                &mesh,
                points
                    .chunks_exact(2)
                    .map(|p| ((p[0] - offset) + 2. * (p[1] - offset)) / actual_scale)
                    .collect(),
            )
            .unwrap();
            let linear = SimplexLinearInterpolator::<_, 2>::new(&mesh, &values).unwrap();
            let natural = NaturalNeighborInterpolator::new(&mesh, &values).unwrap();
            let nearest = NearestNeighborInterpolator::<_, 2>::new(&mesh, &values).unwrap();
            let clough = CloughTocherInterpolator::with_options(
                &mesh,
                &values,
                CloughTocherOptions::new().with_gradient_tolerance(
                    scioxide_core::Tolerance::new(32. * f64::EPSILON, 32. * f64::EPSILON).unwrap(),
                ),
            )
            .unwrap();
            let query = [offset + 1.3 * actual_scale, offset + 0.4 * actual_scale];
            let projected = [offset + actual_scale, query[1]];
            let tolerance = 1024. * f64::EPSILON * (1. + offset.abs() / actual_scale);
            let expected = ((projected[0] - offset) + 2. * (projected[1] - offset)) / actual_scale;
            macro_rules! check {
                ($model:expr, $affine:expr) => {{
                    let model = &$model;
                    let mut workspace = model.workspace().unwrap();
                    let report = model.report(query, P::Clamp, &mut workspace).unwrap();
                    for (a, e) in report.effective().into_iter().zip(projected) {
                        assert!((a - e).abs() / actual_scale <= tolerance);
                    }
                    let value = model.evaluate(query, P::Clamp).unwrap();
                    if $affine {
                        assert!(
                            (value - expected).abs() <= tolerance,
                            "scale={scale}, offset={offset}: {value} != {expected}"
                        );
                    } else {
                        assert_eq!(value, model.evaluate(projected, P::Reject).unwrap());
                    }
                }};
            }
            check!(linear, true);
            check!(natural, true);
            check!(clough, true);
            check!(nearest, false);
            let limit = (query[0] - projected[0]).abs();
            let mut workspace = linear.workspace().unwrap();
            let _ = linear
                .report(
                    query,
                    P::BoundedContinue {
                        max_distance: limit,
                    },
                    &mut workspace,
                )
                .unwrap();
        }
    }
}

#[test]
fn continuation_validation_and_exact_limits_match_grids() {
    let coordinates = [0., 0., 1., 0., 0., 1., 1., 1.];
    let mesh = mesh(&coordinates, 2);
    let values = ScatteredValues::from_input_values(&mesh, vec![0., 1., 2., 3.]).unwrap();
    let linear = SimplexLinearInterpolator::<_, 2>::new(&mesh, &values).unwrap();
    let grid = BilinearInterpolator::new(vec![0., 1.], vec![0., 1.], vec![0., 1., 2., 3.]).unwrap();
    for limit in [-1., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let p = P::BoundedContinue {
            max_distance: limit,
        };
        let a = linear.evaluate([0.5, 0.5], p).unwrap_err();
        let b = grid.evaluate([0.5, 0.5], p).unwrap_err();
        assert!(matches!(a.view(), SciOxideErrorView::InvalidInput { .. }));
        assert_eq!(a, b);
    }
    for query in [[1.5, 0.5], [-0.5, 0.5]] {
        let p = P::BoundedContinue { max_distance: 0.5 };
        assert!(
            (linear.evaluate(query, p).unwrap() - grid.evaluate(query, p).unwrap()).abs() < 1e-14
        );
    }
}

fn sibson_references<S: RealScalar>() {
    let sites: Vec<Vec<f64>> = include_str!("data/sibson_sites.csv")
        .lines()
        .filter(|r| !r.starts_with('#'))
        .map(|r| r.split(',').map(|x| x.parse().unwrap()).collect())
        .collect();
    let coordinates: Vec<S> = sites
        .iter()
        .flat_map(|p| [from_f64(p[0]), from_f64(p[1])])
        .collect();
    let mesh = mesh(&coordinates, 2);
    // Kronecker components expose every weight through the public value protocol.
    let values = ScatteredValues::from_input_values(
        &mesh,
        (0..6)
            .map(|i| {
                let mut v = vec![S::ZERO; 8];
                v[i] = S::ONE;
                let p = &coordinates[2 * i..2 * i + 2];
                v[6] = p[0] * p[0] + p[1] * p[1];
                v[7] = from_f64(sites[i][2]);
                v
            })
            .collect(),
    )
    .unwrap();
    let model = NaturalNeighborInterpolator::new(&mesh, &values).unwrap();
    for row in include_str!("data/sibson.csv")
        .lines()
        .filter(|r| !r.starts_with('#'))
    {
        let data: Vec<f64> = row.split(',').map(|x| x.parse().unwrap()).collect();
        let actual = model
            .evaluate([from_f64(data[0]), from_f64(data[1])], P::Reject)
            .unwrap();
        for (a, e) in actual.iter().zip(&data[2..]) {
            assert!(
                (a.try_to_f64_exact().unwrap() - e).abs()
                    <= 128. * S::EPSILON.try_to_f64_exact().unwrap() * e.abs().max(1.),
                "{row}: {a:?} != {e}"
            );
        }
    }
}
#[test]
fn exact_sibson_references_f64() {
    sibson_references::<f64>();
}
#[test]
fn exact_sibson_references_f32() {
    sibson_references::<f32>();
}

fn spatial_projection<S: RealScalar>() {
    for scale in [1e-12, 1e-9, 1., 1e6, 1e12] {
        for offset in [0_f64, 1e7, 1e9] {
            let lower = from_f64::<S>(offset).try_to_f64_exact().unwrap();
            let upper = from_f64::<S>(offset + scale).try_to_f64_exact().unwrap();
            let extent = upper - lower;
            if extent <= 0. || S::EPSILON.try_to_f64_exact().unwrap() * offset.abs() / extent > 1e-3
            {
                continue;
            }
            let coordinates: Vec<S> = (0..8)
                .flat_map(|i| {
                    (0..3)
                        .map(move |axis| from_f64(if i & (1 << axis) == 0 { lower } else { upper }))
                })
                .collect();
            let mesh = mesh(&coordinates, 3);
            let values = ScatteredValues::from_input_values(
                &mesh,
                coordinates
                    .chunks_exact(3)
                    .map(|p| {
                        from_f64::<S>(
                            (p[0].try_to_f64_exact().unwrap() - lower
                                + 2. * (p[1].try_to_f64_exact().unwrap() - lower)
                                + 3. * (p[2].try_to_f64_exact().unwrap() - lower))
                                / extent,
                        )
                    })
                    .collect(),
            )
            .unwrap();
            let model = SimplexLinearInterpolator::<_, 3>::new(&mesh, &values).unwrap();
            let nearest = NearestNeighborInterpolator::<_, 3>::new(&mesh, &values).unwrap();
            let mut workspace = model.workspace().unwrap();
            let tolerance =
                256. * S::EPSILON.try_to_f64_exact().unwrap() * (1. + lower.abs() / extent);
            for normalized in [
                [1.3, 0.4, 0.6],
                [1.2, 1.3, 0.6],
                [1.2, 1.3, 1.4],
                [-0.3, 0.2, 0.4],
            ] {
                let query = normalized.map(|x| from_f64::<S>(lower + extent * x));
                let expected =
                    query.map(|x| from_f64::<S>(x.try_to_f64_exact().unwrap().clamp(lower, upper)));
                let report = model.report(query, P::Clamp, &mut workspace).unwrap();
                for (a, e) in report.effective().into_iter().zip(expected) {
                    assert!(
                        (a.try_to_f64_exact().unwrap() - e.try_to_f64_exact().unwrap()).abs()
                            / extent
                            <= tolerance
                    );
                }
                let value = model
                    .evaluate(query, P::Clamp)
                    .unwrap()
                    .try_to_f64_exact()
                    .unwrap();
                let target = expected
                    .into_iter()
                    .enumerate()
                    .map(|(axis, x)| {
                        (axis + 1) as f64 * (x.try_to_f64_exact().unwrap() - lower) / extent
                    })
                    .sum::<f64>();
                assert!(
                    (value - target).abs() <= tolerance,
                    "scale={scale} offset={offset} {value}!={target}"
                );
                let limit = query
                    .into_iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(S::ZERO, |a, b| a.max(b));
                // Round the test allowance upward; a rounded-down subtraction
                // is intentionally rejected by the exact bound comparison.
                let limit = limit.next_after(S::INFINITY);
                let _ = model
                    .report(
                        query,
                        P::BoundedContinue {
                            max_distance: limit,
                        },
                        &mut workspace,
                    )
                    .unwrap();
                let mut nearest_workspace = nearest.workspace().unwrap();
                assert_eq!(
                    nearest
                        .report(query, P::Clamp, &mut nearest_workspace)
                        .unwrap()
                        .effective(),
                    report.effective()
                );
            }
        }
    }
}
#[test]
fn spatial_projection_f32() {
    spatial_projection::<f32>();
}
#[test]
fn spatial_projection_f64() {
    spatial_projection::<f64>();
}

#[test]
fn oblique_faces_edges_and_vertices() {
    for scale in [1e-12, 1., 1e12] {
        for offset in [0_f64, 1e9] {
            if scale < offset * 1024. * f64::EPSILON {
                continue;
            }
            let coordinates: Vec<_> = [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]
                .into_iter()
                .map(|x| offset + scale * x)
                .collect();
            let mesh = mesh(&coordinates, 3);
            let values = ScatteredValues::from_input_values(&mesh, vec![0., 1., 2., 3.]).unwrap();
            let model = SimplexLinearInterpolator::<_, 3>::new(&mesh, &values).unwrap();
            let mut workspace = model.workspace().unwrap();
            for (q, p) in [
                ([0.6, 0.6, 0.6], [1. / 3.; 3]),
                ([-0.1, 0.6, 0.8], [0., 0.4, 0.6]),
                ([-1., -1., 2.], [0., 0., 1.]),
            ] {
                let report = model
                    .report(q.map(|x| offset + scale * x), P::Clamp, &mut workspace)
                    .unwrap();
                for (a, e) in report.effective().into_iter().zip(p) {
                    assert!(
                        ((a - offset) / scale - e).abs()
                            <= 128. * f64::EPSILON * (1. + offset / scale)
                    );
                }
            }
        }
    }
}

#[test]
fn higher_dimensional_projection_uses_reserved_qr() {
    let mut coordinates = vec![0.; 20];
    for i in 0..4 {
        coordinates[(i + 1) * 4 + i] = 1.;
    }
    let mesh = mesh(&coordinates, 4);
    let values = ScatteredValues::from_input_values(&mesh, vec![0., 1., 2., 3., 4.]).unwrap();
    let model = SimplexLinearInterpolator::<_, 4>::new(&mesh, &values).unwrap();
    let mut workspace = model.workspace().unwrap();
    let mut output = 0.;
    model
        .evaluate_into([0.4; 4], P::Clamp, &mut output, &mut workspace)
        .unwrap();
    assert!((output - 2.5_f64).abs() < 1e-14);
    for q in [[0.4; 4], [-0.1, 0.6, 0.6, 0.6], [0.4; 4]] {
        model
            .evaluate_into(q, P::Clamp, &mut output, &mut workspace)
            .unwrap();
    }
}

#[test]
fn a_distance_rounded_down_to_the_limit_is_rejected() {
    // 1 - (-epsilon/4) rounds to 1, but its exact distance exceeds 1.
    let boundary = -f64::EPSILON / 4.;
    let coordinates = [-1., 0., boundary, 0., -1., 1., boundary, 1.];
    let mesh = mesh(&coordinates, 2);
    let values = ScatteredValues::from_input_values(&mesh, vec![0., 1., 2., 3.]).unwrap();
    let model = SimplexLinearInterpolator::<_, 2>::new(&mesh, &values).unwrap();
    let grid =
        BilinearInterpolator::new(vec![-1., boundary], vec![0., 1.], vec![0., 1., 2., 3.]).unwrap();
    let policy = P::BoundedContinue { max_distance: 1. };
    assert_eq!(1. - boundary, 1.);
    assert!(model.evaluate([1., 0.5], policy).is_err());
    assert!(grid.evaluate([1., 0.5], policy).is_err());
}

#[test]
fn near_hull_values_stay_finite_at_large_value_scales() {
    let points = square(50);
    let mesh = mesh(&points, 2);
    let values = ScatteredValues::from_input_values(
        &mesh,
        points
            .chunks_exact(2)
            .map(|p| (0.25 + 0.25 * p[0] + 0.25 * p[1]) * f64::MAX)
            .collect(),
    )
    .unwrap();
    let model = NaturalNeighborInterpolator::new(&mesh, &values).unwrap();
    for q in [[1e-12, 0.4], [0.4, 1. - 1e-12], [0.5, 0.5]] {
        let actual = model.evaluate(q, P::Reject).unwrap() / f64::MAX;
        let expected = 0.25 + 0.25 * q[0] + 0.25 * q[1];
        assert!((actual - expected).abs() < 64. * f64::EPSILON);
    }
}

fn oblique_natural_hull<S: RealScalar>() {
    let transform = |p: [f64; 2]| [0.75 * p[0] + 0.25 * p[1], p[1] - 0.25 * p[0]];
    let coordinates: Vec<S> = square(50)
        .chunks_exact(2)
        .flat_map(|p| transform([p[0], p[1]]).map(from_f64::<S>))
        .collect();
    let mesh = mesh(&coordinates, 2);
    let values = ScatteredValues::from_input_values(
        &mesh,
        coordinates
            .chunks_exact(2)
            .map(|p| [S::ONE, p[0], p[1]])
            .collect(),
    )
    .unwrap();
    let model = NaturalNeighborInterpolator::new(&mesh, &values).unwrap();
    let delta = 1e-12_f64.max(8. * S::EPSILON.try_to_f64_exact().unwrap());
    for t in [0.125, 0.5, 0.875] {
        for q in [[delta, t], [1. - delta, t], [t, delta], [t, 1. - delta]] {
            let q = transform(q).map(from_f64::<S>);
            let actual = model.evaluate(q, P::Reject).unwrap();
            for (a, e) in actual.into_iter().zip([S::ONE, q[0], q[1]]) {
                assert!(
                    (a - e).abs() <= from_f64::<S>(64.) * S::EPSILON,
                    "q={q:?}, {a:?} != {e:?}"
                );
            }
        }
    }
}
#[test]
fn oblique_natural_hull_f32() {
    oblique_natural_hull::<f32>();
}
#[test]
fn oblique_natural_hull_f64() {
    oblique_natural_hull::<f64>();
}

/// Rotation makes collinear and coplanar hull sites inexact, so Delaunay keeps
/// sliver simplices whose affine frames are unavailable. Clamp values and
/// gradients use their hull faces instead.
fn rotated_slivers<S: RealScalar>() {
    let tolerance = S::EPSILON.sqrt().try_to_f64_exact().unwrap();
    let (c, s) = (0.6_f64.cos(), 0.6_f64.sin());
    let rotate = |x: f64, y: f64| [c * x - s * y, s * x + c * y];
    let coordinates: Vec<S> = (0..25)
        .flat_map(|i| rotate(f64::from(i % 5) * 0.1, f64::from(i / 5) * 0.1))
        .map(from_f64)
        .collect();
    let mesh = mesh(&coordinates, 2);
    let affine = |p: [f64; 2]| 1. + p[0] - 2. * p[1];
    let values = ScatteredValues::from_input_values(
        &mesh,
        coordinates
            .chunks_exact(2)
            .map(|p| from_f64::<S>(affine([p[0], p[1]].map(|x| x.try_to_f64_exact().unwrap()))))
            .collect(),
    )
    .unwrap();
    let linear = SimplexLinearInterpolator::new(&mesh, &values).unwrap();
    let cubic =
        CloughTocherInterpolator::with_options(&mesh, &values, CloughTocherOptions::new()).unwrap();
    // Below the bottom row the projection lies on the rotated x axis.
    let tangent = [c, s];
    let along = tangent[0] - 2. * tangent[1];
    let mut linear_ws = linear.workspace().unwrap();
    let mut cubic_ws = cubic.workspace().unwrap();
    let (mut value, mut gradient) = (S::ZERO, [S::ZERO; 2]);
    for step in 1..20 {
        let t = 0.02 * f64::from(step);
        let expected = affine(rotate(t, 0.));
        let close = |actual: S, expected: f64| {
            let actual = actual.try_to_f64_exact().unwrap();
            assert!(
                (actual - expected).abs() <= tolerance,
                "{actual} != {expected}"
            );
        };
        let q = rotate(t, -0.05).map(from_f64::<S>);
        {
            linear
                .evaluate_into(q, P::Clamp, &mut value, &mut linear_ws)
                .unwrap();
            close(value, expected);
            linear
                .gradient_into(q, P::Clamp, &mut gradient, &mut linear_ws)
                .unwrap();
            close(gradient[0], along * tangent[0]);
            close(gradient[1], along * tangent[1]);
            cubic
                .evaluate_into(q, P::Clamp, &mut value, &mut cubic_ws)
                .unwrap();
            close(value, expected);
            cubic
                .gradient_into(q, P::Clamp, &mut gradient, &mut cubic_ws)
                .unwrap();
            close(gradient[0], along * tangent[0]);
            close(gradient[1], along * tangent[1]);
        };
        // On the rotated boundary row rounding either lands outside the hull
        // or inside a sliver, which is evaluated on its closest face.
        match linear.evaluate(rotate(t, 0.).map(from_f64::<S>), P::Reject) {
            Ok(value) => close(value, expected),
            Err(error) => assert!(matches!(
                error.view(),
                SciOxideErrorView::DomainError { .. }
            )),
        }
    }
}
#[test]
fn rotated_slivers_f32() {
    rotated_slivers::<f32>();
}
#[test]
fn rotated_slivers_f64() {
    rotated_slivers::<f64>();
}

#[test]
fn split_box_faces_have_clamp_gradients() {
    // Non-dyadic extents round the two triangles of each face differently;
    // queries projecting onto the shared diagonal must still find a region.
    let (c, s) = (0.9_f64.cos(), 0.9_f64.sin());
    for (dims, rotated) in [([0.85, 0.47, 0.34], false), ([0.74, 1.06, 0.81], true)] {
        let place = |p: [f64; 3]| {
            if rotated {
                [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]
            } else {
                p
            }
        };
        let coordinates: Vec<f64> = (0..8)
            .flat_map(|i| {
                place(core::array::from_fn(|a| {
                    if i & (1 << a) == 0 { 0. } else { dims[a] }
                }))
            })
            .collect();
        let mesh = mesh(&coordinates, 3);
        let slope = [0.3, -1.2, 0.7];
        let values = ScatteredValues::from_input_values(
            &mesh,
            coordinates
                .chunks_exact(3)
                .map(|p| 2. + slope[0] * p[0] + slope[1] * p[1] + slope[2] * p[2])
                .collect(),
        )
        .unwrap();
        let model = SimplexLinearInterpolator::<_, 3>::new(&mesh, &values).unwrap();
        // Outside the face z = dims[2] (the rotation is about z), the
        // tangential gradient drops the z component.
        let q = place([0.5 * dims[0], 0.5 * dims[1], 1.7 * dims[2]]);
        let g = model.gradient(q, P::Clamp).unwrap();
        for (a, e) in g.into_iter().zip([slope[0], slope[1], 0.]) {
            assert!((a - e).abs() < 1e-9, "{g:?}");
        }
        let top = place([0.5 * dims[0], 0.5 * dims[1], dims[2]]);
        let value = model.evaluate(q, P::Clamp).unwrap();
        let expected = 2. + slope[0] * top[0] + slope[1] * top[1] + slope[2] * top[2];
        assert!((value - expected).abs() < 1e-12);
    }
}

#[test]
fn thin_but_resolved_slivers_keep_their_contracts() {
    // A triangle thin enough to fail the frame screen, but far thicker than
    // rounding: its affine piece f = y / 1e-9 must not be replaced by a face.
    let coordinates = [0., 0., 1., 0., 0.5, 1e-9];
    let mesh = mesh(&coordinates, 2);
    let values = ScatteredValues::from_input_values(&mesh, vec![0., 0., 1.]).unwrap();
    let model = SimplexLinearInterpolator::new(&mesh, &values).unwrap();
    let accept = |result: Result<f64, scioxide_core::SciOxideError>, expected: f64| match result {
        Ok(value) => assert!((value - expected).abs() < 1e-6, "{value} != {expected}"),
        Err(error) => assert!(
            matches!(error.view(), SciOxideErrorView::NumericalInstability { .. }),
            "{error:?}"
        ),
    };
    accept(model.evaluate([0.5, 5e-10], P::Reject), 0.5);
    accept(
        model.evaluate([0.5, -0.01], P::BoundedContinue { max_distance: 1. }),
        -1e7,
    );
    accept(model.evaluate([0.5, -0.01], P::MethodDefined), -1e7);
    // Clamp projects onto the long edge exactly, where the value is zero.
    accept(model.evaluate([0.5, -0.01], P::Clamp), 0.);
}
