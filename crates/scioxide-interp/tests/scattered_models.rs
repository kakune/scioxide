//! Tessellation and global RBF families: affine values, workspaces and allocation.
#![allow(clippy::unwrap_used)]
mod support;

use crate::support::from_f64;

use scioxide_core::{
    Complex, ExecutionContext, ExecutionPolicy, NdView, NdViewMut, RankThreshold, RealScalar,
    SciOxideErrorKind, SciOxideErrorView, StridedLayout,
};
use scioxide_interp::{
    BatchEvaluator, BatchRequest, CloughTocherInterpolator, CloughTocherOptions, Differentiable,
    ExtrapolationPolicy, Interpolant, NaturalNeighborInterpolator, NearestNeighborInterpolator,
    RbfInterpolator, RbfKernel, RbfOptions, ScatteredOptions, ScatteredValues,
    SimplexLinearInterpolator, interpolate_scattered,
};
use scioxide_spatial::{Delaunay, PointSet};

const REJECT: ExtrapolationPolicy<f64> = ExtrapolationPolicy::Reject;

fn mesh<S: RealScalar>(coordinates: &[S]) -> Delaunay<S> {
    let shape = [coordinates.len() / 2, 2];
    let strides = [2, 1];
    let points = PointSet::new(
        NdView::new(coordinates, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
    )
    .unwrap();
    Delaunay::new(&points).unwrap()
}

/// Site coordinates of a five-site square with an interior point.
fn coordinates<S: RealScalar>() -> Vec<S> {
    [0., 0., 1., 0., 0., 1., 1., 1., 0.4, 0.35]
        .into_iter()
        .map(from_f64)
        .collect()
}

/// Smooth three-component samples at the input sites.
fn samples<S: RealScalar>() -> Vec<[S; 3]> {
    coordinates::<S>()
        .chunks_exact(2)
        .map(|p| {
            let (x, y) = (p[0], p[1]);
            [
                from_f64::<S>(1.) + x * x - y,
                (x * from_f64::<S>(3.)).sin() + y,
                from_f64::<S>(2.) * x * y,
            ]
        })
        .collect()
}

fn indexed_nearest_matches_scan<S: RealScalar>() {
    // Independent R2 queries exercise different tree partitions and site order.
    let mut coordinates = coordinates::<S>();
    for i in 1..=128 {
        coordinates.extend([
            from_f64::<S>((i as f64 * 0.7548776662466927).fract()),
            from_f64::<S>((i as f64 * 0.5698402909980532).fract()),
        ]);
    }
    let geometry = mesh(&coordinates);
    let labels: Vec<S> = (0..geometry.points().len() / 2)
        .map(|i| S::try_from_usize_exact(i).unwrap())
        .collect();
    let values = ScatteredValues::from_site_values(&geometry, labels).unwrap();
    let model = NearestNeighborInterpolator::<_, 2>::new(&geometry, &values).unwrap();
    let clone = model.clone();
    let mut workspace = model.workspace().unwrap();
    let mut queries: Vec<_> = (1..=256)
        .map(|i| {
            [
                from_f64::<S>((i as f64 * 0.6180339887498949).fract()),
                from_f64::<S>((i as f64 * 0.414_213_562_373_095).fract()),
            ]
        })
        .collect();
    queries.extend(coordinates.chunks_exact(2).map(|p| [p[0], p[1]]));
    queries.extend([
        [from_f64(-0.5), from_f64(0.5)],
        [from_f64(1.5), from_f64(0.5)],
    ]);
    for policy in [
        ExtrapolationPolicy::Clamp,
        ExtrapolationPolicy::MethodDefined,
    ] {
        for &query in &queries {
            let mut actual = S::ZERO;
            let report = clone
                .evaluate_into_with_report(query, policy, &mut actual, &mut workspace)
                .unwrap();
            let effective = report.effective();
            let (site, _) = geometry
                .points()
                .chunks_exact(2)
                .enumerate()
                .map(|(site, p)| {
                    let distance = p
                        .iter()
                        .zip(effective)
                        .fold(S::ZERO, |norm, (&x, y)| norm.hypot(x - y));
                    (site, distance)
                })
                .reduce(|best, candidate| {
                    if candidate.1 < best.1 {
                        candidate
                    } else {
                        best
                    }
                })
                .unwrap();
            assert_eq!(actual, S::try_from_usize_exact(site).unwrap());
            assert_eq!(report, model.report(query, policy, &mut workspace).unwrap());
        }
    }
    // A grid puts exact distance ties in different leaves of a nontrivial tree.
    let grid: Vec<S> = (-5..=5)
        .flat_map(|x| {
            (-5..=5).flat_map(move |y| {
                [
                    from_f64::<S>(2.0 * x as f64 + 1.0),
                    from_f64::<S>(2.0 * y as f64 + 1.0),
                ]
            })
        })
        .collect();
    let geometry = mesh(&grid);
    let labels = (0..grid.len() / 2)
        .map(|i| S::try_from_usize_exact(i).unwrap())
        .collect();
    let values = ScatteredValues::from_site_values(&geometry, labels).unwrap();
    let model = NearestNeighborInterpolator::<_, 2>::new(&geometry, &values).unwrap();
    let mut workspace = model.workspace().unwrap();
    let mut actual = S::ZERO;
    model
        .evaluate_into(
            [S::ZERO; 2],
            ExtrapolationPolicy::Reject,
            &mut actual,
            &mut workspace,
        )
        .unwrap();
    let expected = geometry
        .points()
        .chunks_exact(2)
        .position(|p| p == [from_f64::<S>(-1.), from_f64::<S>(-1.)])
        .unwrap();
    assert_eq!(actual, S::try_from_usize_exact(expected).unwrap());
}

#[test]
fn indexed_nearest_matches_exhaustive_f32() {
    indexed_nearest_matches_scan::<f32>();
}

#[test]
fn indexed_nearest_matches_exhaustive_f64() {
    indexed_nearest_matches_scan::<f64>();
}

/// Every family with array, complex and per-component scalar values agrees bit for bit.
fn tessellation_values_match_components<S: RealScalar>() {
    let coordinates = coordinates::<S>();
    let mesh = mesh(&coordinates);
    let arrays = samples::<S>();
    let complex: Vec<_> = arrays.iter().map(|v| Complex::new(v[0], v[1])).collect();
    let options = ScatteredOptions::new();
    let array_values = ScatteredValues::from_input_values(&mesh, arrays.clone()).unwrap();
    let complex_values = ScatteredValues::from_input_values(&mesh, complex).unwrap();
    let columns: Vec<_> = (0..3)
        .map(|c| {
            ScatteredValues::from_input_values(&mesh, arrays.iter().map(|v| v[c]).collect())
                .unwrap()
        })
        .collect();
    let queries = [[0.25, 0.5], [0.7, 0.2], [0.4, 0.35], [0.5, 0.0]].map(|q| q.map(from_f64::<S>));
    let policy = ExtrapolationPolicy::Reject;

    let nearest =
        NearestNeighborInterpolator::<_, 2, _>::with_options(&mesh, &array_values, options)
            .unwrap();
    let linear =
        SimplexLinearInterpolator::<_, 2, _>::with_options(&mesh, &array_values, options).unwrap();
    let natural = NaturalNeighborInterpolator::with_options(&mesh, &array_values, options).unwrap();
    let clough =
        CloughTocherInterpolator::with_options(&mesh, &array_values, CloughTocherOptions::new())
            .unwrap();
    let linear_complex =
        SimplexLinearInterpolator::<_, 2, _>::with_options(&mesh, &complex_values, options)
            .unwrap();
    let clough_complex =
        CloughTocherInterpolator::with_options(&mesh, &complex_values, CloughTocherOptions::new())
            .unwrap();
    for (c, column) in columns.iter().enumerate() {
        let nearest_c =
            NearestNeighborInterpolator::<_, 2>::with_options(&mesh, column, options).unwrap();
        let linear_c =
            SimplexLinearInterpolator::<_, 2>::with_options(&mesh, column, options).unwrap();
        let natural_c = NaturalNeighborInterpolator::with_options(&mesh, column, options).unwrap();
        let clough_c =
            CloughTocherInterpolator::with_options(&mesh, column, CloughTocherOptions::new())
                .unwrap();
        for query in queries {
            assert_eq!(
                nearest.evaluate(query, policy).unwrap()[c],
                nearest_c.evaluate(query, policy).unwrap()
            );
            assert_eq!(
                linear.evaluate(query, policy).unwrap()[c],
                linear_c.evaluate(query, policy).unwrap()
            );
            assert_eq!(
                natural.evaluate(query, policy).unwrap()[c],
                natural_c.evaluate(query, policy).unwrap()
            );
            assert_eq!(
                clough.evaluate(query, policy).unwrap()[c],
                clough_c.evaluate(query, policy).unwrap()
            );
            let [dx, dy] = linear.gradient(query, policy).unwrap();
            let [cx, cy] = linear_c.gradient(query, policy).unwrap();
            assert_eq!((dx[c], dy[c]), (cx, cy));
            let [dx, dy] = clough.gradient(query, policy).unwrap();
            let [cx, cy] = clough_c.gradient(query, policy).unwrap();
            assert_eq!((dx[c], dy[c]), (cx, cy));
            if c < 2 {
                let value = linear_complex.evaluate(query, policy).unwrap();
                assert_eq!(
                    [value.re, value.im][c],
                    linear_c.evaluate(query, policy).unwrap()
                );
                let value = clough_complex.evaluate(query, policy).unwrap();
                assert_eq!(
                    [value.re, value.im][c],
                    clough_c.evaluate(query, policy).unwrap()
                );
            }
        }
    }
}

#[test]
fn tessellation_values_match_components_in_both_widths() {
    tessellation_values_match_components::<f64>();
    tessellation_values_match_components::<f32>();
}

fn rbf_values_match_components<S: RealScalar + scioxide_core::NumericScalar<Real = S>>() {
    let points: Vec<[S; 2]> = coordinates::<S>()
        .chunks_exact(2)
        .map(|p| [p[0], p[1]])
        .collect();
    let arrays = samples::<S>();
    let kernel = RbfKernel::Multiquadric(from_f64(1.));
    let options = RbfOptions::new().with_rank_threshold(
        RankThreshold::new(from_f64(0.), from_f64::<S>(64.) * S::EPSILON).unwrap(),
    );
    let model =
        RbfInterpolator::with_options(points.clone(), arrays.clone(), kernel, options).unwrap();
    let complex = RbfInterpolator::with_options(
        points.clone(),
        arrays.iter().map(|v| Complex::new(v[1], v[2])).collect(),
        kernel,
        options,
    )
    .unwrap();
    for c in 0..3 {
        let column = RbfInterpolator::with_options(
            points.clone(),
            arrays.iter().map(|v| v[c]).collect(),
            kernel,
            options,
        )
        .unwrap();
        for query in [[0.25, 0.5], [0.9, 0.1]].map(|q| q.map(from_f64::<S>)) {
            let expected = column.evaluate(query, ExtrapolationPolicy::Reject).unwrap();
            assert_eq!(
                model.evaluate(query, ExtrapolationPolicy::Reject).unwrap()[c],
                expected
            );
            if c > 0 {
                let value = complex
                    .evaluate(query, ExtrapolationPolicy::Reject)
                    .unwrap();
                assert_eq!([value.re, value.im][c - 1], expected);
            }
        }
    }
}

#[test]
fn rbf_values_match_components_in_both_widths() {
    rbf_values_match_components::<f64>();
    rbf_values_match_components::<f32>();
}

#[test]
fn queries_execute_after_reservation() {
    let coordinates = coordinates::<f64>();
    let mesh = mesh(&coordinates);
    let values = ScatteredValues::from_input_values(&mesh, samples::<f64>()).unwrap();
    let options = ScatteredOptions::new();
    let nearest =
        NearestNeighborInterpolator::<_, 2, _>::with_options(&mesh, &values, options).unwrap();
    let linear =
        SimplexLinearInterpolator::<_, 2, _>::with_options(&mesh, &values, options).unwrap();
    let natural = NaturalNeighborInterpolator::with_options(&mesh, &values, options).unwrap();
    let clough =
        CloughTocherInterpolator::with_options(&mesh, &values, CloughTocherOptions::new()).unwrap();
    let points: Vec<[f64; 2]> = coordinates.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
    let rbf_options =
        RbfOptions::new().with_rank_threshold(RankThreshold::new(0.0, 1e-12).unwrap());
    let gaussian = RbfKernel::Gaussian(1.0);
    let rbf =
        RbfInterpolator::with_options(points, samples::<f64>(), gaussian, rbf_options).unwrap();
    let mut workspaces = [
        nearest.workspace().unwrap(),
        linear.workspace().unwrap(),
        natural.workspace().unwrap(),
        clough.workspace().unwrap(),
    ];
    let mut rbf_workspace = rbf.workspace().unwrap();
    let mut value = [0.0; 3];
    let mut gradient = [0.0; 6];
    let queries = [[0.25, 0.5], [0.7, 0.2], [1.5, 0.5], [0.4, 0.35]];
    let run = |workspaces: &mut [scioxide_interp::ScatteredWorkspace<f64>; 4],
               rbf_workspace: &mut scioxide_interp::QueryWorkspace<f64>,
               value: &mut [f64; 3],
               gradient: &mut [f64; 6]| {
        for query in queries {
            let policy = ExtrapolationPolicy::Clamp;
            let [a, b, c, d] = workspaces;
            nearest.evaluate_into(query, policy, value, a).unwrap();
            linear.evaluate_into(query, policy, value, b).unwrap();
            linear.gradient_into(query, policy, gradient, b).unwrap();
            natural.evaluate_into(query, policy, value, c).unwrap();
            clough.evaluate_into(query, policy, value, d).unwrap();
            clough.gradient_into(query, policy, gradient, d).unwrap();
            let _ = clough.report(query, policy, d).unwrap();
            let _ = nearest
                .evaluate_into_with_report(query, policy, value, a)
                .unwrap();
            let _ = linear
                .evaluate_into_with_report(query, policy, value, b)
                .unwrap();
            let _ = natural
                .evaluate_into_with_report(query, policy, value, c)
                .unwrap();
            let _ = clough
                .evaluate_into_with_report(query, policy, value, d)
                .unwrap();
            rbf.evaluate_into(query, ExtrapolationPolicy::Clamp, value, rbf_workspace)
                .unwrap();
        }
    };
    // The first pass binds each geometry workspace to its snapshot.
    run(
        &mut workspaces,
        &mut rbf_workspace,
        &mut value,
        &mut gradient,
    );
    {
        run(
            &mut workspaces,
            &mut rbf_workspace,
            &mut value,
            &mut gradient,
        );
    };
}

#[test]
fn undersized_workspaces_and_outputs_are_dimension_mismatches() {
    let small = mesh(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    let coordinates = coordinates::<f64>();
    let large = mesh(&coordinates);
    let small_values = ScatteredValues::from_site_values(&small, vec![1.0, 2.0, 3.0]).unwrap();
    let large_values = ScatteredValues::from_input_values(&large, samples::<f64>()).unwrap();
    let options = ScatteredOptions::new();
    let tiny =
        SimplexLinearInterpolator::<_, 2>::with_options(&small, &small_values, options).unwrap();
    let model =
        SimplexLinearInterpolator::<_, 2, _>::with_options(&large, &large_values, options).unwrap();
    let mut undersized = tiny.workspace().unwrap();
    assert_eq!(
        model.workspace_capacity(&undersized),
        scioxide_core::Capacity::NeedsGrowth
    );
    let mut out = [7.0; 3];
    assert!(matches!(
        model.evaluate_into([0.25, 0.25], REJECT, &mut out, &mut undersized),
        Err(error) if matches!(error.view(), SciOxideErrorView::DimensionMismatch { .. })
    ));
    assert!(matches!(
        model.report([0.25, 0.25], REJECT, &mut undersized),
        Err(error) if matches!(error.view(), SciOxideErrorView::DimensionMismatch { .. })
    ));
    assert_eq!(out, [7.0; 3]);
    // A larger workspace serves the smaller model.
    let mut large_workspace = model.workspace().unwrap();
    let mut scalar = 0.0;
    tiny.evaluate_into([0.25, 0.25], REJECT, &mut scalar, &mut large_workspace)
        .unwrap();
    assert!((scalar - 1.75).abs() < 1e-12);
    // Wrong output shapes leave the output unchanged.
    let mut short = [7.0; 2];
    let mut workspace = model.workspace().unwrap();
    assert!(matches!(
        model.gradient_into([0.25, 0.25], REJECT, &mut short[..], &mut workspace),
        Err(error) if matches!(error.view(), SciOxideErrorView::DimensionMismatch { .. })
    ));
    assert_eq!(short, [7.0; 2]);

    // RBF query workspaces hold small values inline; wider values need their own.
    let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let gaussian = RbfKernel::Gaussian(1.0);
    let narrow = RbfInterpolator::new(points.clone(), vec![1.0, 2.0, 3.0], gaussian).unwrap();
    let wide =
        RbfInterpolator::new(points, vec![[1.0; 20], [2.0; 20], [3.0; 20]], gaussian).unwrap();
    let mut narrow_workspace = narrow.workspace().unwrap();
    let mut wide_out = [7.0; 20];
    assert!(matches!(
        wide.evaluate_into([0.25, 0.25], REJECT, &mut wide_out, &mut narrow_workspace),
        Err(error) if matches!(error.view(), SciOxideErrorView::DimensionMismatch { .. })
    ));
    assert_eq!(wide_out, [7.0; 20]);
    let mut wide_workspace = wide.workspace().unwrap();
    wide.evaluate_into([0.0, 0.0], REJECT, &mut wide_out, &mut wide_workspace)
        .unwrap();
    assert!(wide_out.iter().all(|v| (v - 1.0).abs() < 1e-10));
}

#[test]
fn tessellation_batches_report_hull_distances_and_are_transactional() {
    let coordinates = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0];
    let mesh = mesh(&coordinates);
    let values = ScatteredValues::from_input_values(&mesh, vec![1.0, 4.0, 3.0]).unwrap();
    let linear = SimplexLinearInterpolator::<_, 2>::new(&mesh, &values).unwrap();
    let batch = BatchEvaluator::new(&linear, ExecutionPolicy::Sequential);
    let mut workspace = batch
        .workspace(BatchRequest::values(2).with_reports())
        .unwrap();
    let points = [0.25_f64, 0.25, 1.0, 1.0];
    let queries = NdView::new(&points, StridedLayout::new(&[2, 2], &[2, 1]).unwrap()).unwrap();
    let mut output = [0.0_f64; 2];
    let mut reports = [ExtrapolationPolicy::Clamp
        .resolve(linear.domains(), [0.0, 0.0])
        .unwrap(); 2];
    let mut out =
        NdViewMut::new(&mut output, StridedLayout::new(&[2, 1], &[1, 1]).unwrap()).unwrap();
    batch
        .evaluate_into(
            &queries,
            ExtrapolationPolicy::Clamp,
            &mut out,
            Some(&mut reports),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!((output[0] - 2.25).abs() < 1e-12);
    // (1, 1) lies inside the bounding box but outside the hull: it is clamped
    // to the hull point (0.5, 0.5).
    assert!((output[1] - 3.5).abs() < 1e-12);
    assert!((reports[1].effective()[0] - 0.5).abs() < 1e-12);
    assert!((reports[1].outside_distance()[1] - 0.5).abs() < 1e-12);
    let before = output;
    let mut out =
        NdViewMut::new(&mut output, StridedLayout::new(&[2, 1], &[1, 1]).unwrap()).unwrap();
    assert!(
        batch
            .evaluate_into(
                &queries,
                REJECT,
                &mut out,
                None,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .is_err()
    );
    assert_eq!(output, before);
}

fn empty_hull_batches<S: RealScalar>() {
    let coordinates = [0., 0., 1., 0., 0., 1.].map(from_f64::<S>);
    let mesh = mesh(&coordinates);
    let values = ScatteredValues::from_input_values(&mesh, vec![Vec::<S>::new(); 3]).unwrap();
    let model = SimplexLinearInterpolator::<_, 2, _>::new(&mesh, &values).unwrap();
    let batch = BatchEvaluator::new(&model, ExecutionPolicy::Sequential);
    let mut workspace = batch
        .workspace(BatchRequest::values(2).with_reports())
        .unwrap();
    let points = [0.25, 0.25, 0.8, 0.8].map(from_f64::<S>);
    let queries = NdView::new(&points, StridedLayout::new(&[2, 2], &[2, 1]).unwrap()).unwrap();
    let mut storage = [];
    let mut out =
        NdViewMut::new(&mut storage, StridedLayout::new(&[2, 0], &[0, 1]).unwrap()).unwrap();
    let sentinel = ExtrapolationPolicy::Clamp
        .resolve(model.domains(), [from_f64(99.); 2])
        .unwrap();
    let mut reports = [sentinel; 2];
    let context = ExecutionContext::sequential_native();
    {
        for reported in [false, true] {
            let result = batch.evaluate_into(
                &queries,
                ExtrapolationPolicy::Reject,
                &mut out,
                if reported { Some(&mut reports) } else { None },
                &mut workspace,
                context,
            );
            assert!(result.is_err());
            assert_eq!(reports, [sentinel; 2]);
        }
        batch
            .evaluate_into(
                &queries,
                ExtrapolationPolicy::Clamp,
                &mut out,
                Some(&mut reports),
                &mut workspace,
                context,
            )
            .unwrap();
    };
    let mut query_ws = model.workspace().unwrap();
    let expected = model
        .report(
            [from_f64(0.8); 2],
            ExtrapolationPolicy::Clamp,
            &mut query_ws,
        )
        .unwrap();
    assert_eq!(reports[1], expected);
    for coordinate in reports[1].effective() {
        support::assert_close(
            coordinate,
            0.5_f64,
            0.0,
            16.0 * S::EPSILON.try_to_f64_exact().unwrap(),
        );
    }
    for distance in reports[1].outside_distance() {
        support::assert_close(
            distance,
            0.3_f64,
            0.0,
            16.0 * S::EPSILON.try_to_f64_exact().unwrap(),
        );
    }
}

#[test]
fn empty_scattered_values_use_hull_reports_in_both_widths() {
    empty_hull_batches::<f32>();
    empty_hull_batches::<f64>();
}

fn one_call_scattered_family<S: RealScalar>() {
    let bits = |values: &[S]| {
        values
            .iter()
            .map(|v| v.try_to_f64_exact().unwrap().to_bits())
            .collect::<Vec<_>>()
    };
    // Planar sites with three-component values.
    let points: Vec<[S; 2]> = coordinates::<S>()
        .chunks_exact(2)
        .map(|p| [p[0], p[1]])
        .collect();
    let queries: Vec<[S; 2]> = [
        [0.1, 0.2],
        [0.5, 0.5],
        [0.33, 0.61],
        [0.9, 0.05],
        [0.4, 0.35],
    ]
    .map(|[x, y]| [from_f64(x), from_f64(y)])
    .to_vec();
    let out = interpolate_scattered(&points, samples::<S>(), &queries).unwrap();
    let mesh = mesh(&coordinates::<S>());
    let values = ScatteredValues::from_input_values(&mesh, samples::<S>()).unwrap();
    let model = SimplexLinearInterpolator::<S, 2, [S; 3]>::new(&mesh, &values).unwrap();
    assert_eq!(out.len(), queries.len());
    for (value, &query) in out.iter().zip(&queries) {
        let expected = model.evaluate(query, ExtrapolationPolicy::Reject).unwrap();
        assert_eq!(bits(value), bits(&expected));
    }
    // Sites of a cube in three dimensions reproduce an affine function.
    let cube: Vec<[S; 3]> = (0..8)
        .map(|i| [i & 1, (i >> 1) & 1, (i >> 2) & 1].map(|bit| from_f64(f64::from(bit))))
        .chain([[0.3, 0.4, 0.5].map(from_f64)])
        .collect();
    let affine = |[x, y, z]: [S; 3]| S::ONE + x - from_f64::<S>(2.) * y + from_f64::<S>(0.5) * z;
    let heights: Vec<S> = cube.iter().copied().map(affine).collect();
    let probes = [[0.2, 0.7, 0.1], [0.9, 0.1, 0.6]].map(|q| q.map(from_f64::<S>));
    let out = interpolate_scattered(&cube, heights, &probes).unwrap();
    for (value, &probe) in out.iter().zip(&probes) {
        let error = (*value - affine(probe)).abs().try_to_f64_exact().unwrap();
        assert!(error <= 64. * S::EPSILON.try_to_f64_exact().unwrap());
    }
    assert!(
        interpolate_scattered(&points, samples::<S>(), &[])
            .unwrap()
            .is_empty()
    );
}

#[test]
fn one_call_scattered_interpolation_equals_the_explicit_path_in_both_widths() {
    one_call_scattered_family::<f32>();
    one_call_scattered_family::<f64>();
}

#[test]
fn one_call_scattered_interpolation_reports_hull_duplicate_and_rank_errors() {
    let points = [[0.0_f64, 0.], [1., 0.], [0., 1.], [1., 1.]];
    let values = vec![1., 2., 3., 4.];
    let outside = interpolate_scattered(&points, values.clone(), &[[0.5, 0.5], [1.5, 0.5]]);
    assert_eq!(outside.unwrap_err().kind(), SciOxideErrorKind::DomainError);
    // A repeated site must repeat its value.
    let repeated = [[0.0_f64, 0.], [1., 0.], [0., 1.], [1., 0.]];
    let same = interpolate_scattered(&repeated, vec![1., 2., 3., 2.], &[[0.25, 0.25]]).unwrap();
    assert!((same[0] - 1.75).abs() < 1e-14);
    let conflicting = interpolate_scattered(&repeated, vec![1., 2., 3., 5.], &[[0.25, 0.25]]);
    assert_eq!(
        conflicting.unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    // Collinear sites, and fewer than three sites, do not span the plane.
    let collinear = [[0.0_f64, 0.], [1., 1.], [2., 2.]];
    let error = interpolate_scattered(&collinear, vec![1., 2., 3.], &[[0.5, 0.5]]).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::Singular);
    let error = interpolate_scattered(&points[..2], vec![1., 2.], &[[0.5, 0.]]).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::Singular);
    let mismatch = interpolate_scattered(&points, vec![1., 2., 3.], &[[0.5, 0.5]]);
    assert_eq!(
        mismatch.unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let nan = interpolate_scattered(&points, values, &[[f64::NAN, 0.5]]);
    assert_eq!(nan.unwrap_err().kind(), SciOxideErrorKind::NonFinite);
}
