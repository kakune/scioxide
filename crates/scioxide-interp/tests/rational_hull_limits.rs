//! Rational knot limits, projected derivatives and range-safe Sibson edges.
#![allow(clippy::unwrap_used)]
mod support;

use scioxide_core::{
    DomainReason, NdView, RealScalar, SciOxideError, SciOxideErrorView, StridedLayout,
};
use scioxide_interp::{
    BSplineBasis, CloughTocherInterpolator, CloughTocherOptions, Differentiable,
    ExtrapolationPolicy as P, Interpolant, NaturalNeighborInterpolator, Nurbs, ScatteredValues,
    SimplexLinearInterpolator, TwiceDifferentiable,
};
use scioxide_spatial::{Delaunay, GeometryLimits, GeometryOptions, PointSet};
use support::from_f64 as s;

fn close<S: RealScalar>(actual: S, expected: f64) {
    let actual = actual.try_to_f64_exact().unwrap();
    assert!(
        (actual - expected).abs()
            <= 1024. * S::EPSILON.try_to_f64_exact().unwrap() * expected.abs().max(1.),
        "{actual} != {expected}"
    );
}
fn undefined<T>(result: Result<T, SciOxideError>) {
    assert!(matches!(
        result.err().unwrap().view(),
        SciOxideErrorView::DomainError {
            reason: DomainReason::Undefined,
            ..
        }
    ));
}
fn basis<S: RealScalar>(knots: &[f64], degree: usize) -> BSplineBasis<S> {
    BSplineBasis::new(knots.iter().copied().map(s::<S>).collect(), degree).unwrap()
}
fn nurbs_limits<S: RealScalar>() {
    let line = basis::<S>(&[0., 0., 1., 1., 2., 2.], 1);
    for controls in [[1., 1., 1., 1.], [0., 1., 1., 2.]] {
        let model = Nurbs::new(
            [line.clone()],
            controls.map(s::<S>).to_vec(),
            [1., 1., 2., 2.].map(s::<S>).to_vec(),
        )
        .unwrap();
        close(
            model.gradient([s(1.)], P::Reject).unwrap()[0],
            if controls[0] == 1. { 0. } else { 1. },
        );
        close(model.hessian([s(1.)], P::Reject).unwrap()[0][0], 0.);
    }
    // R = 1/(1+x), independently represented on [0,1] and [1,3].
    // The right homogeneous piece is twice the left representation.
    let rational_basis = basis::<S>(&[0., 0., 1., 1., 3., 3.], 1);
    let controls = [1., 0.5, 0.5, 0.25].map(s::<S>).to_vec();
    let weights = [1., 2., 4., 8.].map(s::<S>).to_vec();
    let model = Nurbs::new([rational_basis.clone()], controls.clone(), weights.clone()).unwrap();
    close(model.evaluate([s(1.)], P::Reject).unwrap(), 0.5);
    close(model.gradient([s(1.)], P::Reject).unwrap()[0], -0.25);
    close(model.hessian([s(1.)], P::Reject).unwrap()[0][0], 0.25);
    for (x, d, dd) in [
        (0., -1., 2.),
        (3., -0.0625, 0.03125),
        (-1., 0., 0.),
        (4., 0., 0.),
    ] {
        close(model.gradient([s(x)], P::Clamp).unwrap()[0], d);
        close(model.hessian([s(x)], P::Clamp).unwrap()[0][0], dd);
    }
    let tensor = Nurbs::new(
        [rational_basis.clone(), rational_basis],
        controls
            .iter()
            .flat_map(|a| controls.iter().map(move |b| *a * *b))
            .collect(),
        weights
            .iter()
            .flat_map(|a| weights.iter().map(move |b| *a * *b))
            .collect(),
    )
    .unwrap();
    let gradient = tensor.gradient([s(1.); 2], P::Reject).unwrap();
    let hessian = tensor.hessian([s(1.); 2], P::Reject).unwrap();
    for x in gradient {
        close(x, -0.125);
    }
    close(hessian[0][0], 0.125);
    close(hessian[1][1], 0.125);
    close(hessian[0][1], 0.0625);
    close(hessian[1][0], 0.0625);
    for controls in [
        [0., 1., 2., 3.],
        [0., 1., 1., 3.],
        [0., 1., 1. + S::EPSILON.try_to_f64_exact().unwrap(), 2.],
    ] {
        let bad = Nurbs::new(
            [line.clone()],
            controls.map(s::<S>).to_vec(),
            vec![S::ONE; 4],
        )
        .unwrap();
        let mut workspace = bad.workspace().unwrap();
        let mut out = s::<S>(99.);
        undefined(bad.gradient_into([s(1.)], P::Reject, &mut out, &mut workspace));
        assert_eq!(out, s(99.));
        undefined(bad.hessian_into([s(1.)], P::Reject, &mut out, &mut workspace));
        close(bad.evaluate([s(0.)], P::Reject).unwrap(), 0.);
    }
    let c1 = Nurbs::new(
        [basis::<S>(&[0., 0., 0., 1., 1., 1., 2., 2., 2.], 2)],
        [0., 0.5, 1., 1., 1.5, 3.].map(s::<S>).to_vec(),
        vec![S::ONE; 6],
    )
    .unwrap();
    close(c1.gradient([s(1.)], P::Reject).unwrap()[0], 1.);
    undefined(c1.hessian([s(1.)], P::Reject));
    // Only the left/left corner jumps: independent single-axis checks miss it.
    let mut corners = vec![S::ONE; 16];
    corners[5] = s(2.);
    let corner = Nurbs::new([line.clone(), line], corners, vec![S::ONE; 16]).unwrap();
    undefined(corner.gradient([s(1.); 2], P::Reject));
    undefined(corner.hessian([s(1.); 2], P::Reject));
    let mut workspace = tensor.workspace().unwrap();
    let mut gradient = [S::ZERO; 2];
    let mut hessian = [[S::ZERO; 2]; 2];
    {
        tensor
            .gradient_into([s(1.); 2], P::Reject, &mut gradient, &mut workspace)
            .unwrap();
        tensor
            .hessian_into([s(1.); 2], P::Reject, &mut hessian, &mut workspace)
            .unwrap();
    };
}
fn mesh<S: RealScalar>(points: &[S], dimension: usize) -> Delaunay<S> {
    let shape = [points.len() / dimension, dimension];
    let strides = [dimension, 1];
    let points =
        PointSet::new(NdView::new(points, StridedLayout::new(&shape, &strides).unwrap()).unwrap())
            .unwrap();
    Delaunay::with_options(
        &points,
        GeometryOptions::new().with_limits(GeometryLimits::new().with_max_dimension(dimension)),
    )
    .unwrap()
}
fn hull_gradients<S: RealScalar>() {
    let coordinates = [0., 0., 1., 0., 0., 1.].map(s::<S>);
    let mesh = mesh(&coordinates, 2);
    let values = ScatteredValues::from_input_values(
        &mesh,
        vec![[s(0.), s(0.)], [s(1.), s(0.)], [s(0.), s(1.)]],
    )
    .unwrap();
    let linear = SimplexLinearInterpolator::<_, 2, _>::new(&mesh, &values).unwrap();
    let cubic = CloughTocherInterpolator::with_options(
        &mesh,
        &values,
        CloughTocherOptions::new().with_gradient_tolerance(
            scioxide_core::Tolerance::new(S::EPSILON * s(32.), S::EPSILON * s(32.)).unwrap(),
        ),
    )
    .unwrap();
    fn check<S: RealScalar, I: Differentiable<S, 2, Value = [S; 2]>>(model: &I) {
        let mut workspace = model.workspace().unwrap();
        let mut out = [[S::ZERO; 2]; 2];
        for (q, expected) in [
            ([0.8, 0.8], [[0.5, -0.5], [-0.5, 0.5]]),
            ([1.1, 0.5], [[0.5, -0.5], [-0.5, 0.5]]),
            ([2., -0.5], [[0., 0.], [0., 0.]]),
            ([0.2, 0.2], [[1., 0.], [0., 1.]]),
            ([0.5, 0.5], [[1., 0.], [0., 1.]]),
        ] {
            let q = q.map(s::<S>);
            model
                .gradient_into(q, P::Clamp, &mut out, &mut workspace)
                .unwrap();
            for (a, e) in out
                .into_iter()
                .flatten()
                .zip(expected.into_iter().flatten())
            {
                close(a, e);
            }
            if q != [s(0.5); 2] {
                let delta = s::<S>(if S::EPSILON.try_to_f64_exact().unwrap() > 1e-10 {
                    0.001
                } else {
                    1e-5
                });
                for axis in 0..2 {
                    let mut a = q;
                    let mut b = q;
                    a[axis] += delta;
                    b[axis] -= delta;
                    let va = model.evaluate(a, P::Clamp).unwrap();
                    let vb = model.evaluate(b, P::Clamp).unwrap();
                    for component in 0..2 {
                        let fd = (va[component] - vb[component]) / (S::TWO * delta);
                        assert!((fd - out[axis][component]).abs() < s(0.002));
                    }
                }
            }
        }
        out = [[s(99.); 2]; 2];
        undefined(model.gradient_into([s(1.5), s(0.5)], P::Clamp, &mut out, &mut workspace));
        assert_eq!(out, [[s(99.); 2]; 2]);
        for q in [[0.8, 0.8], [2., -0.5], [0.2, 0.2]] {
            model
                .gradient_into(q.map(s::<S>), P::Clamp, &mut out, &mut workspace)
                .unwrap();
        }
    }
    check(&linear);
    check(&cubic);
    let tangential_constant =
        ScatteredValues::from_input_values(&mesh, vec![S::ZERO, S::ONE, S::ONE]).unwrap();
    let tangent = SimplexLinearInterpolator::<_, 2>::new(&mesh, &tangential_constant).unwrap();
    for x in tangent.gradient([s(1.5), s(0.5)], P::Clamp).unwrap() {
        close(x, 0.);
    }
    let constant = ScatteredValues::from_input_values(&mesh, vec![s::<S>(2.); 3]).unwrap();
    let l = SimplexLinearInterpolator::<_, 2>::new(&mesh, &constant).unwrap();
    let c = CloughTocherInterpolator::with_options(
        &mesh,
        &constant,
        CloughTocherOptions::new().with_gradient_tolerance(
            scioxide_core::Tolerance::new(S::EPSILON * s(32.), S::EPSILON * s(32.)).unwrap(),
        ),
    )
    .unwrap();
    for g in [
        l.gradient([s(1.5), s(0.5)], P::Clamp).unwrap(),
        c.gradient([s(1.5), s(0.5)], P::Clamp).unwrap(),
    ] {
        for x in g {
            close(x, 0.);
        }
    }
}
fn higher_hull<S: RealScalar, const D: usize>() {
    let mut coordinates = vec![S::ZERO; (D + 1) * D];
    for axis in 0..D {
        coordinates[(axis + 1) * D + axis] = S::ONE;
    }
    let mesh = mesh(&coordinates, D);
    let values = ScatteredValues::from_input_values(
        &mesh,
        (0..=D)
            .map(|i| S::try_from_usize_exact(i).unwrap())
            .collect(),
    )
    .unwrap();
    let model = SimplexLinearInterpolator::<_, D>::new(&mesh, &values).unwrap();
    let mut workspace = model.workspace().unwrap();
    let mut gradient = [S::ZERO; D];
    let q = [s(0.6); D];
    model
        .gradient_into(q, P::Clamp, &mut gradient, &mut workspace)
        .unwrap();
    for (i, x) in gradient.into_iter().enumerate() {
        close(x, (i + 1) as f64 - (D + 1) as f64 / 2.);
    }
    model
        .gradient_into(q, P::Clamp, &mut gradient, &mut workspace)
        .unwrap();
}
fn sibson_extremes<S: RealScalar>(scales: &[f64]) {
    for &scale in scales {
        let a = s::<S>(scale);
        let coordinates = [-a, -a, a, -a, -a, a, a, a];
        let mesh = mesh(&coordinates, 2);
        let values = ScatteredValues::from_input_values(
            &mesh,
            vec![
                [s(0.), s(0.), S::ONE],
                [S::ONE, s(0.), S::ONE],
                [s(0.), S::ONE, S::ONE],
                [S::ONE, S::ONE, S::ONE],
            ],
        )
        .unwrap();
        let model = NaturalNeighborInterpolator::new(&mesh, &values).unwrap();
        let mut workspace = model.workspace().unwrap();
        let mut out = [S::ZERO; 3];
        for (q, expected) in [
            ([0., -1.], [0.5, 0., 1.]),
            ([0., 1.], [0.5, 1., 1.]),
            ([-1., 0.], [0., 0.5, 1.]),
            ([1., 0.], [1., 0.5, 1.]),
            ([-1., -1.], [0., 0., 1.]),
            ([1., 1.], [1., 1., 1.]),
            ([-1., 1.], [0., 1., 1.]),
            ([1., -1.], [1., 0., 1.]),
            ([0., -1.5], [0.5, 0., 1.]),
            ([0., 1.5], [0.5, 1., 1.]),
            ([-1.5, 0.], [0., 0.5, 1.]),
            ([1.5, 0.], [1., 0.5, 1.]),
            ([0., -0.99], [0.5, 0.005, 1.]),
            ([0., 0.99], [0.5, 0.995, 1.]),
            ([-0.99, 0.], [0.005, 0.5, 1.]),
            ([0.99, 0.], [0.995, 0.5, 1.]),
        ] {
            model
                .evaluate_into(q.map(|v| s::<S>(v) * a), P::Clamp, &mut out, &mut workspace)
                .unwrap();
            for (x, e) in out.into_iter().zip(expected) {
                close(x, e);
            }
        }
        model
            .evaluate_into([S::ZERO, -a], P::Reject, &mut out, &mut workspace)
            .unwrap();
    }
}
#[test]
fn rational_limits_f32() {
    nurbs_limits::<f32>();
}
#[test]
fn rational_limits_f64() {
    nurbs_limits::<f64>();
}
#[test]
fn projected_gradients_f32() {
    hull_gradients::<f32>();
    higher_hull::<f32, 3>();
    higher_hull::<f32, 4>();
}
#[test]
fn projected_gradients_f64() {
    hull_gradients::<f64>();
    higher_hull::<f64, 3>();
    higher_hull::<f64, 4>();
}
#[test]
fn sibson_ranges_f32() {
    sibson_extremes::<f32>(&[1e-30, 1., 1e30, 2e38]);
}
#[test]
fn sibson_ranges_f64() {
    sibson_extremes::<f64>(&[1e-200, 1., 1e200, 1e308]);
}

fn flat_seams<S: RealScalar>() {
    // A site in the middle of a straight hull edge is not a polytope vertex.
    let coordinates = [0., 0., 0.5, 0., 1., 0., 0., 1., 1., 1.].map(s::<S>);
    let mesh = mesh(&coordinates, 2);
    let values =
        ScatteredValues::from_input_values(&mesh, vec![s::<S>(0.), s(0.5), s(1.), s(0.), s(1.)])
            .unwrap();
    let model = SimplexLinearInterpolator::<_, 2>::new(&mesh, &values).unwrap();
    let g = model.gradient([s(0.5), s(-0.5)], P::Clamp).unwrap();
    close(g[0], 1.);
    close(g[1], 0.);
    // The same seam can carry a real kink in the boundary trace.
    let values =
        ScatteredValues::from_input_values(&mesh, vec![s::<S>(0.), s(1.), s(0.), s(0.), s(0.)])
            .unwrap();
    let kink = SimplexLinearInterpolator::<_, 2>::new(&mesh, &values).unwrap();
    undefined(kink.gradient([s(0.5), s(-0.5)], P::Clamp));
    let coordinates: Vec<S> = (0..8)
        .flat_map(|i| {
            (0..3).map(move |axis| {
                if i & (1 << axis) == 0 {
                    S::ZERO
                } else {
                    S::ONE
                }
            })
        })
        .collect();
    let mesh = self::mesh(&coordinates, 3);
    let values = ScatteredValues::from_input_values(
        &mesh,
        coordinates
            .chunks_exact(3)
            .map(|p| p[0] + s::<S>(2.) * p[1] + s::<S>(3.) * p[2])
            .collect(),
    )
    .unwrap();
    let model = SimplexLinearInterpolator::<_, 3>::new(&mesh, &values).unwrap();
    let g = model.gradient([s(1.5), s(0.5), s(0.5)], P::Clamp).unwrap();
    for (a, e) in g.into_iter().zip([0., 2., 3.]) {
        close(a, e);
    }
}
fn rounded_limits<S: RealScalar>() {
    // Unequal per-piece weights round each quotient differently, yet every
    // piece represents f(x) = slope * x exactly.
    let kink = S::EPSILON.sqrt();
    for (slope, weights) in [(0.1, [1., 1., 5., 5.]), (0.7, [1., 1., 10., 10.])] {
        let controls = [0., slope, slope, 2. * slope].map(s::<S>);
        let line = basis::<S>(&[0., 0., 1., 1., 2., 2.], 1);
        let weights = weights.map(s::<S>);
        let model = Nurbs::new([line.clone()], controls.to_vec(), weights.to_vec()).unwrap();
        close(model.gradient([s(1.)], P::Reject).unwrap()[0], slope);
        close(model.hessian([s(1.)], P::Reject).unwrap()[0][0], 0.);
        let tensor = Nurbs::new(
            [line.clone(), line.clone()],
            controls
                .iter()
                .flat_map(|a| controls.iter().map(move |b| *a + *b))
                .collect(),
            weights
                .iter()
                .flat_map(|a| weights.iter().map(move |b| *a * *b))
                .collect(),
        )
        .unwrap();
        for x in tensor.gradient([s(1.); 2], P::Reject).unwrap() {
            close(x, slope);
        }
        for x in tensor
            .hessian([s(1.); 2], P::Reject)
            .unwrap()
            .into_iter()
            .flatten()
        {
            close(x, 0.);
        }
        // A relative slope change far below the data's precision is still a kink.
        let mut bent = controls;
        bent[3] += kink * s(slope);
        let bent = Nurbs::new([line], bent.to_vec(), weights.to_vec()).unwrap();
        undefined(bent.gradient([s(1.)], P::Reject));
    }
    // Two collinear hull edges meet at a non-dyadic split site; f = x + 2y.
    let coordinates = [0., 0., 0.3, 0., 1., 0., 0.5, 0.8].map(s::<S>);
    let mesh = mesh(&coordinates, 2);
    let affine: Vec<S> = coordinates
        .chunks_exact(2)
        .map(|p| p[0] + s::<S>(2.) * p[1])
        .collect();
    let values = ScatteredValues::from_input_values(&mesh, affine.clone()).unwrap();
    let linear = SimplexLinearInterpolator::new(&mesh, &values).unwrap();
    let cubic =
        CloughTocherInterpolator::with_options(&mesh, &values, CloughTocherOptions::new()).unwrap();
    for q in [[0.3, -1.], [0.3, -1e-3]] {
        let q = q.map(s::<S>);
        for g in [
            linear.gradient(q, P::Clamp).unwrap(),
            cubic.gradient(q, P::Clamp).unwrap(),
        ] {
            // Clough–Tocher reproduces affine data up to gradient estimation.
            assert!((g[0] - S::ONE).abs() <= kink, "{g:?}");
            close(g[1], 0.);
        }
    }
    let mut bent = affine;
    bent[1] += kink;
    let values = ScatteredValues::from_input_values(&mesh, bent).unwrap();
    let linear = SimplexLinearInterpolator::new(&mesh, &values).unwrap();
    undefined(linear.gradient([s(0.3), s(-1.)], P::Clamp));
}
#[test]
fn rounded_limits_f32() {
    rounded_limits::<f32>();
}
#[test]
fn rounded_limits_f64() {
    rounded_limits::<f64>();
}
#[test]
fn flat_face_seams_f32() {
    flat_seams::<f32>();
}
#[test]
fn flat_face_seams_f64() {
    flat_seams::<f64>();
}

#[test]
fn projected_nonlinear_cubic_matches_value_differences() {
    let coordinates = [0.0_f64, 0., 1., 0., 0., 1., 1., 1., 0.3, 0.4];
    let mesh = mesh(&coordinates, 2);
    let values = ScatteredValues::from_input_values(
        &mesh,
        coordinates
            .chunks_exact(2)
            .map(|p| p[0] * p[0] + p[0] * p[1] + 3. * p[1] * p[1])
            .collect(),
    )
    .unwrap();
    let model =
        CloughTocherInterpolator::with_options(&mesh, &values, CloughTocherOptions::new()).unwrap();
    for q in [[1.3, 0.4], [-0.2, 0.7], [0.7, 1.3], [-0.5, -0.4]] {
        let g = model.gradient(q, P::Clamp).unwrap();
        for axis in 0..2 {
            let mut a = q;
            let mut b = q;
            a[axis] += 1e-5;
            b[axis] -= 1e-5;
            let fd = (model.evaluate(a, P::Clamp).unwrap() - model.evaluate(b, P::Clamp).unwrap())
                / 2e-5;
            assert!((fd - g[axis]).abs() < 1e-7);
        }
    }
}

#[test]
fn projected_batches_commit_values_and_reports_together() {
    use scioxide_core::{ExecutionContext, ExecutionPolicy, NdViewMut};
    use scioxide_interp::{BatchEvaluator, BatchRequest};
    let coordinates = [0., 0., 1., 0., 0., 1.];
    let mesh = mesh(&coordinates, 2);
    let values = ScatteredValues::from_input_values(&mesh, vec![0., 1., 0.]).unwrap();
    let model = SimplexLinearInterpolator::<_, 2>::new(&mesh, &values).unwrap();
    let plan = BatchEvaluator::new(&model, ExecutionPolicy::Sequential);
    let mut workspace = plan
        .workspace(BatchRequest::gradients(2).with_reports())
        .unwrap();
    let mut output = [99.; 4];
    let sentinel = model
        .report([0.2, 0.2], P::Reject, &mut model.workspace().unwrap())
        .unwrap();
    let mut reports = [sentinel; 2];
    for points in [[0.8, 0.8, 2., -0.5], [0.8, 0.8, 1.5, 0.5]] {
        let before = output;
        let before_reports = reports;
        let queries = NdView::new(&points, StridedLayout::new(&[2, 2], &[2, 1]).unwrap()).unwrap();
        let mut out = NdViewMut::new(
            &mut output,
            StridedLayout::new(&[2, 2, 1], &[2, 1, 1]).unwrap(),
        )
        .unwrap();
        let result = plan.gradient_into(
            &queries,
            P::Clamp,
            &mut out,
            Some(&mut reports),
            &mut workspace,
            ExecutionContext::sequential_native(),
        );
        if points[2] == 1.5 {
            undefined(result);
            assert_eq!(output, before);
            assert_eq!(reports, before_reports);
        } else {
            result.unwrap();
            for (a, e) in output.into_iter().zip([0.5, -0.5, 0., 0.]) {
                close(a, e);
            }
            assert_eq!(reports[1].effective(), [1., 0.]);
        }
    }
}
