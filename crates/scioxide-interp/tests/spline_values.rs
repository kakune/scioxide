//! Affine values, transactional preflight, workspace contracts and allocation
//! behavior of the B-spline and piecewise cubic families.
#![allow(clippy::unwrap_used)]

mod support;

use crate::support::from_f64;

use scioxide_core::{
    AffineValue, Complex, ExecutionContext, ExecutionPolicy, NdView, NdViewMut, RealScalar,
    SciOxideErrorKind, StridedLayout,
};
use scioxide_interp::{
    AkimaInterpolator, AkimaMethod, BSpline, BSplineBasis, BatchEvaluator, BatchRequest,
    CubicSpline, Differentiable, ExtrapolationPolicy, Integrable, Interpolant, KnotSide, Nurbs,
    PchipInterpolator, PenalizedSplinePlan, SplineBoundary, SplinePenalty,
    TensorSplineInterpolationPlan, TwiceDifferentiable,
};

fn scalar_query_allocations<S: RealScalar>() {
    let policy = ExtrapolationPolicy::Reject;
    for degree in 0..=5 {
        for controls in [1024, 16384] {
            let spans = controls - degree;
            let knots = core::iter::repeat_n(S::ZERO, degree + 1)
                .chain((1..spans).map(|k| from_f64(k as f64)))
                .chain(core::iter::repeat_n(from_f64(spans as f64), degree + 1))
                .collect();
            let basis = BSplineBasis::new(knots, degree).unwrap();
            let spline = BSpline::new(
                [basis],
                (0..controls)
                    .map(|k| from_f64::<S>((k % 7) as f64))
                    .collect(),
            )
            .unwrap();
            let mut workspace = spline.workspace().unwrap();
            let mut value = S::ZERO;
            {
                for point in [S::ZERO, from_f64(3.25), from_f64(spans as f64)] {
                    spline
                        .evaluate_into([point], policy, &mut value, &mut workspace)
                        .unwrap();
                    if degree <= 3 {
                        assert_eq!(spline.evaluate([point], policy).unwrap(), value);
                    }
                    spline
                        .gradient_into([point], policy, &mut value, &mut workspace)
                        .unwrap();
                    if degree <= 3 {
                        assert_eq!(spline.gradient([point], policy).unwrap()[0], value);
                    }
                    spline
                        .derivative_into(
                            [point],
                            [2],
                            Some([KnotSide::Right]),
                            ExtrapolationPolicy::MethodDefined,
                            &mut value,
                            &mut workspace,
                        )
                        .unwrap();
                    if degree <= 3 {
                        assert_eq!(
                            spline
                                .derivative(
                                    [point],
                                    [2],
                                    Some([KnotSide::Right]),
                                    ExtrapolationPolicy::MethodDefined
                                )
                                .unwrap(),
                            value
                        );
                    }
                }
                let before = value;
                assert!(
                    spline
                        .evaluate_into([S::NAN], policy, &mut value, &mut workspace)
                        .is_err()
                );
                assert_eq!(value, before);
                assert!(
                    spline
                        .gradient_into([S::NAN], policy, &mut value, &mut workspace)
                        .is_err()
                );
                assert_eq!(value, before);
                spline
                    .evaluate_into([from_f64(3.25)], policy, &mut value, &mut workspace)
                    .unwrap();
            };
        }
    }
}

#[test]
fn scalar_owned_and_prepared_queries_execute_f32() {
    scalar_query_allocations::<f32>();
}

#[test]
fn scalar_owned_and_prepared_queries_execute_f64() {
    scalar_query_allocations::<f64>();
}

/// Real components of each value.
fn components<S: RealScalar, V: AffineValue<Scale = S>>(values: &[V]) -> Vec<Vec<S>> {
    values
        .iter()
        .map(|v| {
            (0..v.dimension().unwrap())
                .map(|c| v.component(c).unwrap())
                .collect()
        })
        .collect()
}

/// Real components of one value.
fn flat<S: RealScalar, V: AffineValue<Scale = S> + ?Sized>(value: &V) -> Vec<S> {
    (0..value.dimension().unwrap())
        .map(|c| value.component(c).unwrap())
        .collect()
}

/// Column `c` of node-major values.
fn column<S: RealScalar>(rows: &[Vec<S>], c: usize) -> Vec<S> {
    rows.iter().map(|row| row[c]).collect()
}

/// Every protocol operation of `model` equals the scalar models of its components bit for bit.
fn assert_componentwise<S, M, C>(model: &M, scalars: &[C], points: &[[S; 1]])
where
    S: RealScalar,
    M: Differentiable<S, 1> + Integrable<S, 1>,
    C: Differentiable<S, 1, Value = S> + Integrable<S, 1>,
{
    let policy = ExtrapolationPolicy::MethodDefined;
    let flat = |value: &M::Value| components(core::slice::from_ref(value)).remove(0);
    for &point in points {
        let value = flat(&model.evaluate(point, policy).unwrap());
        let gradient = flat(&model.gradient(point, policy).unwrap()[0]);
        let integral = flat(
            &model
                .integral([from_f64(-0.75)], [point[0]], policy)
                .unwrap(),
        );
        for (c, scalar) in scalars.iter().enumerate() {
            assert_eq!(value[c], scalar.evaluate(point, policy).unwrap());
            assert_eq!(gradient[c], scalar.gradient(point, policy).unwrap()[0]);
            assert_eq!(
                integral[c],
                scalar
                    .integral([from_f64(-0.75)], [point[0]], policy)
                    .unwrap()
            );
        }
    }
}

fn points<S: RealScalar>() -> Vec<[S; 1]> {
    [-0.5, 0.0, 0.3, 1.0, 1.7, 2.5, 3.0, 3.4]
        .into_iter()
        .map(|x| [from_f64(x)])
        .collect()
}

fn piecewise_families<S: RealScalar, V: AffineValue<Scale = S> + Clone + Sync>(values: Vec<V>) {
    let x: Vec<S> = [0.0, 0.5, 1.0, 2.0, 3.0]
        .into_iter()
        .map(from_f64)
        .collect();
    let values = values[..x.len()].to_vec();
    let rows = components(&values);
    let dimension = rows[0].len();
    let pchip = PchipInterpolator::new(x.clone(), values.clone()).unwrap();
    let scalars: Vec<_> = (0..dimension)
        .map(|c| PchipInterpolator::new(x.clone(), column(&rows, c)).unwrap())
        .collect();
    assert_componentwise(&pchip, &scalars, &points());
    for method in [AkimaMethod::Akima, AkimaMethod::Makima] {
        let akima = AkimaInterpolator::with_method(x.clone(), values.clone(), method).unwrap();
        let scalars: Vec<_> = (0..dimension)
            .map(|c| AkimaInterpolator::with_method(x.clone(), column(&rows, c), method).unwrap())
            .collect();
        assert_componentwise(&akima, &scalars, &points());
    }
    let cubic = CubicSpline::new(x.clone(), values).unwrap();
    let scalars: Vec<_> = (0..dimension)
        .map(|c| CubicSpline::new(x.clone(), column(&rows, c)).unwrap())
        .collect();
    assert_componentwise(&cubic, &scalars, &points());
    // Prepared antiderivatives are per component too.
    let prepared = cubic.prepare_antiderivative().unwrap();
    let mut scratch = vec![S::ZERO; dimension];
    let mut out = vec![S::ZERO; dimension];
    prepared
        .integral_into(
            from_f64(0.25),
            from_f64(2.75),
            ExtrapolationPolicy::Reject,
            out.as_mut_slice(),
            &mut scratch,
        )
        .unwrap();
    for (c, scalar) in scalars.iter().enumerate() {
        let expected = scalar
            .prepare_antiderivative()
            .unwrap()
            .integral(from_f64(0.25), from_f64(2.75), ExtrapolationPolicy::Reject)
            .unwrap();
        assert_eq!(out[c], expected);
    }
}

fn bspline_families<S: RealScalar, V: AffineValue<Scale = S> + Clone + Sync>(values: Vec<V>) {
    let knots: Vec<S> = [0.0, 0.0, 0.0, 0.0, 1.25, 1.25, 2.25, 3.0, 3.0, 3.0, 3.0]
        .into_iter()
        .map(from_f64)
        .collect();
    let basis = BSplineBasis::new(knots, 3).unwrap();
    let rows = components(&values);
    let dimension = rows[0].len();
    let spline = BSpline::new([basis.clone()], values.clone()).unwrap();
    let scalars: Vec<_> = (0..dimension)
        .map(|c| BSpline::new([basis.clone()], column(&rows, c)).unwrap())
        .collect();
    let policy = ExtrapolationPolicy::MethodDefined;
    for point in points::<S>() {
        let value = components(&[spline.evaluate(point, policy).unwrap()]).remove(0);
        let gradient = flat(&spline.gradient(point, policy).unwrap()[0]);
        let hessian = flat(&spline.hessian(point, policy).unwrap()[0][0]);
        let left = components(&[spline
            .derivative(point, [1], Some([KnotSide::Left]), policy)
            .unwrap()])
        .remove(0);
        let integral =
            components(&[spline.integral([from_f64(-0.75)], point, policy).unwrap()]).remove(0);
        for (c, scalar) in scalars.iter().enumerate() {
            assert_eq!(value[c], scalar.evaluate(point, policy).unwrap());
            assert_eq!(gradient[c], scalar.gradient(point, policy).unwrap()[0]);
            assert_eq!(hessian[c], scalar.hessian(point, policy).unwrap()[0][0]);
            assert_eq!(
                left[c],
                scalar
                    .derivative(point, [1], Some([KnotSide::Left]), policy)
                    .unwrap()
            );
            assert_eq!(
                integral[c],
                scalar.integral([from_f64(-0.75)], point, policy).unwrap()
            );
        }
    }
    // Refinement, derivative splines and prepared antiderivatives.
    let refined = spline.insert_knot(from_f64(1.5)).unwrap();
    let slope = spline.derivative_spline().unwrap();
    let prepared = spline.prepare_antiderivative().unwrap();
    for (c, scalar) in scalars.iter().enumerate() {
        let refined_scalar = scalar.insert_knot(from_f64(1.5)).unwrap();
        let slope_scalar = scalar.derivative_spline().unwrap();
        for k in 0..refined_scalar.bases()[0].control_count() {
            assert_eq!(
                refined.coefficients()[k * dimension + c],
                refined_scalar.coefficients()[k]
            );
        }
        for k in 0..slope_scalar.bases()[0].control_count() {
            assert_eq!(
                slope.coefficients()[k * dimension + c],
                slope_scalar.coefficients()[k]
            );
        }
        let all = components(&[prepared
            .integral(from_f64(0.5), from_f64(2.5), ExtrapolationPolicy::Reject)
            .unwrap()])
        .remove(0);
        let one = scalar
            .prepare_antiderivative()
            .unwrap()
            .integral(from_f64(0.5), from_f64(2.5), ExtrapolationPolicy::Reject)
            .unwrap();
        assert_eq!(all[c], one);
    }
    // Tensor surfaces: each component is the scalar surface.
    let linear = BSplineBasis::new(
        [0.0, 0.0, 1.0, 2.0, 2.0]
            .into_iter()
            .map(from_f64)
            .collect(),
        1,
    )
    .unwrap();
    assert!(BSpline::new([linear.clone(), basis.clone()], values.clone()).is_err());
    let grid: Vec<V> = (0..21).map(|i| values[i % values.len()].clone()).collect();
    let surface = BSpline::new([linear.clone(), basis.clone()], grid.clone()).unwrap();
    let grid_rows = components(&grid);
    for c in 0..dimension {
        let scalar = BSpline::new([linear.clone(), basis.clone()], column(&grid_rows, c)).unwrap();
        for point in [
            [from_f64(0.3), from_f64(1.2)],
            [from_f64(1.0), from_f64(2.9)],
            [from_f64(2.0), from_f64(0.0)],
        ] {
            let value = components(&[surface.evaluate(point, policy).unwrap()]).remove(0);
            assert_eq!(value[c], scalar.evaluate(point, policy).unwrap());
            let gradient = surface.gradient(point, policy).unwrap();
            let expected = scalar.gradient(point, policy).unwrap();
            for axis in 0..2 {
                assert_eq!(flat(&gradient[axis])[c], expected[axis]);
            }
            let hessian = surface.hessian(point, policy).unwrap();
            let expected = scalar.hessian(point, policy).unwrap();
            for row in 0..2 {
                for col in 0..2 {
                    assert_eq!(flat(&hessian[row][col])[c], expected[row][col]);
                }
            }
        }
        let integral = surface
            .integral(
                [from_f64(0.0), from_f64(0.5)],
                [from_f64(1.5), from_f64(3.0)],
                policy,
            )
            .unwrap();
        assert_eq!(
            components(&[integral])[0][c],
            scalar
                .integral(
                    [from_f64(0.0), from_f64(0.5)],
                    [from_f64(1.5), from_f64(3.0)],
                    policy
                )
                .unwrap()
        );
    }
}

fn fits<S: RealScalar, V: AffineValue<Scale = S> + Clone + Sync>(values: Vec<V>) {
    let basis = BSplineBasis::new(
        [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]
            .into_iter()
            .map(from_f64)
            .collect(),
        2,
    )
    .unwrap();
    let samples: Vec<S> = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]
        .into_iter()
        .map(from_f64)
        .collect();
    let data: Vec<V> = (0..samples.len())
        .map(|i| values[i % values.len()].clone())
        .collect();
    let rows = components(&data);
    let plan = PenalizedSplinePlan::new(
        basis.clone(),
        samples,
        None,
        SplinePenalty::new(from_f64(0.1), 2).unwrap(),
    )
    .unwrap();
    let fit = plan.fit(&data).unwrap();
    for c in 0..rows[0].len() {
        let scalar = plan.fit(&column(&rows, c)).unwrap();
        let n = basis.control_count();
        let dimension = rows[0].len();
        for k in 0..n {
            assert_eq!(
                fit.spline().coefficients()[k * dimension + c],
                scalar.spline().coefficients()[k]
            );
        }
        assert_eq!(fit.diagnostics()[c], scalar.diagnostics()[0]);
    }
    let nodes: Vec<S> = [0.0, 0.4, 1.3, 2.0].into_iter().map(from_f64).collect();
    let interpolation = TensorSplineInterpolationPlan::new([basis.clone()], [nodes]).unwrap();
    let grid: Vec<V> = (0..4).map(|i| values[i % values.len()].clone()).collect();
    let grid_rows = components(&grid);
    let spline = interpolation.fit(&grid).unwrap();
    for c in 0..grid_rows[0].len() {
        let scalar = interpolation.fit(&column(&grid_rows, c)).unwrap();
        for k in 0..basis.control_count() {
            assert_eq!(
                spline.coefficients()[k * grid_rows[0].len() + c],
                scalar.coefficients()[k]
            );
        }
    }
}

fn nurbs_family<S: RealScalar, V: AffineValue<Scale = S> + Clone + Sync>(values: Vec<V>) {
    let basis = BSplineBasis::new(
        [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]
            .into_iter()
            .map(from_f64)
            .collect(),
        2,
    )
    .unwrap();
    let controls: Vec<V> = (0..4).map(|i| values[i % values.len()].clone()).collect();
    let weights: Vec<S> = [1.0, 0.5, 2.0, 0.75].into_iter().map(from_f64).collect();
    let rows = components(&controls);
    let model = Nurbs::new([basis.clone()], controls, weights.clone()).unwrap();
    let policy = ExtrapolationPolicy::Reject;
    for c in 0..rows[0].len() {
        let scalar = Nurbs::new([basis.clone()], column(&rows, c), weights.clone()).unwrap();
        for x in [0.0, 0.4, 1.3, 1.6, 2.0] {
            let point = [from_f64(x)];
            assert_eq!(
                components(&[model.evaluate(point, policy).unwrap()])[0][c],
                scalar.evaluate(point, policy).unwrap()
            );
            assert_eq!(
                flat(&model.gradient(point, policy).unwrap()[0])[c],
                scalar.gradient(point, policy).unwrap()[0]
            );
            assert_eq!(
                flat(&model.hessian(point, policy).unwrap()[0][0])[c],
                scalar.hessian(point, policy).unwrap()[0][0]
            );
        }
    }
}

fn all_families<S: RealScalar>() {
    let real: Vec<S> = [1.0, -2.0, 0.5, 3.0, -1.25, 2.0, 0.0]
        .into_iter()
        .map(from_f64)
        .collect();
    let complex: Vec<Complex<S>> = real
        .iter()
        .zip(real.iter().rev())
        .map(|(&a, &b)| Complex::new(a, b))
        .collect();
    let vectors: Vec<Vec<S>> = real
        .iter()
        .map(|&a| vec![a, a * a, -a, from_f64(0.5)])
        .collect();
    let arrays: Vec<[S; 3]> = real
        .iter()
        .map(|&a| [a, from_f64::<S>(1.0) - a, a + a])
        .collect();
    piecewise_families(complex.clone());
    piecewise_families(vectors.clone());
    piecewise_families(arrays.clone());
    bspline_families(complex.clone());
    bspline_families(vectors.clone());
    bspline_families(arrays.clone());
    fits(complex.clone());
    fits(vectors);
    nurbs_family(complex);
    nurbs_family(arrays);
}

#[test]
fn affine_values_equal_scalar_components_f32() {
    all_families::<f32>();
}

#[test]
fn affine_values_equal_scalar_components_f64() {
    all_families::<f64>();
}

/// Each listed failure leaves `out` unchanged.
fn assert_preflight<I: Differentiable<f64, 1> + Integrable<f64, 1>>(model: &I) {
    let mut workspace = model.workspace().unwrap();
    let width = model.value_dimension();
    let mut out = vec![7.0; width];
    let mut wrong = vec![7.0; width + 1];
    let reject = ExtrapolationPolicy::Reject;
    assert!(
        model
            .evaluate_into([0.5], reject, wrong.as_mut_slice(), &mut workspace)
            .is_err()
    );
    assert!(
        model
            .evaluate_into([f64::NAN], reject, out.as_mut_slice(), &mut workspace)
            .is_err()
    );
    assert!(
        model
            .evaluate_into([99.0], reject, out.as_mut_slice(), &mut workspace)
            .is_err()
    );
    assert!(
        model
            .gradient_into([99.0], reject, out.as_mut_slice(), &mut workspace)
            .is_err()
    );
    assert!(
        model
            .integral_into([1.0], [0.0], reject, out.as_mut_slice(), &mut workspace)
            .is_err()
    );
    assert!(
        model
            .integral_into([0.0], [99.0], reject, out.as_mut_slice(), &mut workspace)
            .is_err()
    );
    assert_eq!(out, vec![7.0; width]);
    assert_eq!(wrong, vec![7.0; width + 1]);
}

#[test]
fn preflight_failures_leave_output_unchanged() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![[1.0, 2.0], [0.0, 1.0], [3.0, -1.0]];
    assert_preflight(&PchipInterpolator::new(x.clone(), y.clone()).unwrap());
    assert_preflight(&AkimaInterpolator::new(x.clone(), y.clone()).unwrap());
    assert_preflight(
        &CubicSpline::with_boundary(x.clone(), y.clone(), SplineBoundary::Natural).unwrap(),
    );
    assert_preflight(&BSpline::linear(x, y).unwrap());
    // An undefined ordinary derivative is rejected before any write.
    let jump = BSpline::new(
        [BSplineBasis::new(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], 1).unwrap()],
        vec![[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
    )
    .unwrap();
    let mut workspace = jump.workspace().unwrap();
    let mut out = [7.0; 2];
    let error = jump
        .gradient_into([1.0], ExtrapolationPolicy::Reject, &mut out, &mut workspace)
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DomainError);
    assert_eq!(out, [7.0; 2]);
    let mut hessian = [7.0; 2];
    assert!(
        jump.hessian_into(
            [1.0],
            ExtrapolationPolicy::Reject,
            &mut hessian,
            &mut workspace
        )
        .is_err()
    );
    assert_eq!(hessian, [7.0; 2]);
}

#[test]
fn undersized_workspaces_are_dimension_mismatches() {
    let wide: Vec<Vec<f64>> = (0..3).map(|i| vec![f64::from(i); 20]).collect();
    let x = vec![0.0, 1.0, 2.0];
    let small = PchipInterpolator::new(x.clone(), vec![0.0, 1.0, 2.0]).unwrap();
    let mut workspace = small.workspace().unwrap();
    let mut out = vec![7.0; 20];
    let pchip = PchipInterpolator::new(x.clone(), wide.clone()).unwrap();
    let cubic =
        CubicSpline::with_boundary(x.clone(), wide.clone(), SplineBoundary::Natural).unwrap();
    let linear = BSpline::linear(x, wide.clone()).unwrap();
    let nurbs = Nurbs::new(
        [BSplineBasis::new(vec![0.0, 0.0, 1.0, 2.0, 2.0], 1).unwrap()],
        wide,
        vec![1.0, 2.0, 1.0],
    )
    .unwrap();
    let reject = ExtrapolationPolicy::Reject;
    let kind = SciOxideErrorKind::DimensionMismatch;
    assert_eq!(
        pchip
            .evaluate_into([0.5], reject, &mut out, &mut workspace)
            .unwrap_err()
            .kind(),
        kind
    );
    assert_eq!(
        cubic
            .integral_into([0.0], [1.0], reject, &mut out, &mut workspace)
            .unwrap_err()
            .kind(),
        kind
    );
    assert_eq!(
        linear
            .gradient_into([0.5], reject, &mut out, &mut workspace)
            .unwrap_err()
            .kind(),
        kind
    );
    assert_eq!(
        nurbs
            .hessian_into([0.5], reject, &mut out, &mut workspace)
            .unwrap_err()
            .kind(),
        kind
    );
    assert_eq!(out, vec![7.0; 20]);
    // The basis workspace is sized by the degree.
    let cubic_basis = BSplineBasis::new(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], 3).unwrap();
    let constant = BSplineBasis::new(vec![0.0, 1.0], 0).unwrap();
    let mut basis_workspace = constant.workspace().unwrap();
    let mut row = [7.0; 4];
    let big = BSplineBasis::new([0.0; 6].into_iter().chain([1.0; 6]).collect(), 5).unwrap();
    let mut big_row = [7.0; 6];
    assert_eq!(
        big.integrals_into(0.0, 1.0, reject, &mut big_row, &mut basis_workspace)
            .unwrap_err()
            .kind(),
        kind
    );
    assert_eq!(big_row, [7.0; 6]);
    let mut cubic_workspace = cubic_basis.workspace().unwrap();
    cubic_basis
        .evaluate_into(0.5, 0, None, reject, &mut row, &mut cubic_workspace)
        .unwrap();
    assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-15);
}

#[test]
fn reserved_queries_and_fits_execute() {
    let x = vec![0.0, 0.5, 1.0, 2.0, 3.0];
    let y: Vec<[f64; 2]> = x.iter().map(|&v| [v * v, 1.0 - v]).collect();
    let pchip = PchipInterpolator::new(x.clone(), y.clone()).unwrap();
    let cubic = CubicSpline::with_boundary(x, y.clone(), SplineBoundary::Natural).unwrap();
    let basis =
        BSplineBasis::new(vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0], 3).unwrap();
    let spline =
        BSpline::new([basis.clone()], y.into_iter().chain([[0.5, 0.5]]).collect()).unwrap();
    let surface = BSpline::new(
        [
            basis.clone(),
            BSplineBasis::new(vec![0.0, 0.0, 1.0, 1.0], 1).unwrap(),
        ],
        vec![1.0; 12],
    )
    .unwrap();
    let nurbs = Nurbs::new(
        [basis.clone()],
        vec![1.0, 2.0, 0.5, 3.0, -1.0, 0.0],
        vec![1.0; 6],
    )
    .unwrap();
    let prepared = cubic.prepare_antiderivative().unwrap();
    let mut pchip_ws = pchip.workspace().unwrap();
    let mut cubic_ws = cubic.workspace().unwrap();
    let mut spline_ws = spline.workspace().unwrap();
    let mut surface_ws = surface.workspace().unwrap();
    let mut nurbs_ws = nurbs.workspace().unwrap();
    let mut basis_ws = basis.workspace().unwrap();
    let mut pair = [0.0; 2];
    let mut quad = [0.0; 4];
    let mut scalar = 0.0;
    let mut row = [0.0; 6];
    let policy = ExtrapolationPolicy::MethodDefined;
    for i in 0..16 {
        let t = 0.2 * f64::from(i) + 0.1;
        pchip
            .evaluate_into([t], policy, &mut pair, &mut pchip_ws)
            .unwrap();
        pchip
            .gradient_into([t], policy, &mut pair, &mut pchip_ws)
            .unwrap();
        cubic
            .integral_into([0.0], [t], policy, &mut pair, &mut cubic_ws)
            .unwrap();
        prepared
            .integral_into(0.0, t, policy, &mut pair, &mut quad)
            .unwrap();
        spline
            .evaluate_into([t], policy, &mut pair, &mut spline_ws)
            .unwrap();
        spline
            .hessian_into([t], policy, &mut pair, &mut spline_ws)
            .unwrap();
        spline
            .derivative_into(
                [t],
                [2],
                Some([KnotSide::Left]),
                policy,
                &mut pair,
                &mut spline_ws,
            )
            .unwrap();
        spline
            .integral_into([0.0], [t], policy, &mut pair, &mut spline_ws)
            .unwrap();
        surface
            .hessian_into([t, 0.5], policy, &mut quad, &mut surface_ws)
            .unwrap();
        surface
            .integral_into([0.0, 0.0], [t, 1.0], policy, &mut scalar, &mut surface_ws)
            .unwrap();
        nurbs
            .hessian_into(
                [1.5],
                ExtrapolationPolicy::Reject,
                &mut scalar,
                &mut nurbs_ws,
            )
            .unwrap();
        basis
            .evaluate_into(
                1.5,
                1,
                None,
                ExtrapolationPolicy::Reject,
                &mut row,
                &mut basis_ws,
            )
            .unwrap();
        basis
            .integrals_into(
                0.0,
                2.5,
                ExtrapolationPolicy::Reject,
                &mut row,
                &mut basis_ws,
            )
            .unwrap();
    }

    let samples = vec![0.0, 0.4, 1.0, 1.7, 2.2, 3.0];
    let values: Vec<[f64; 2]> = samples.iter().map(|&v| [v, v * v]).collect();
    let plan = PenalizedSplinePlan::new(
        basis.clone(),
        samples,
        None,
        SplinePenalty::new(0.1, 2).unwrap(),
    )
    .unwrap();
    let mut fit_ws = plan.workspace(2).unwrap();
    let mut coefficients = [0.0; 12];
    let _ = plan
        .fit_into(
            &values,
            &mut coefficients,
            &mut fit_ws,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    let nodes = vec![0.0, 0.3, 1.2, 1.8, 2.6, 3.0];
    let interpolation = TensorSplineInterpolationPlan::new([basis], [nodes]).unwrap();
    let mut interpolation_ws = interpolation.workspace(2).unwrap();
    for _ in 0..4 {
        let report = plan
            .fit_into(
                &values,
                &mut coefficients,
                &mut fit_ws,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(report.normal_residual() < 1e-14);
        interpolation
            .fit_into(
                &values,
                &mut coefficients,
                &mut interpolation_ws,
                ExecutionContext::sequential(),
            )
            .unwrap();
    }
    assert_eq!(fit_ws.diagnostics().len(), 2);

    // The shared batch shell over a spline.
    let batch = BatchEvaluator::new(&spline, ExecutionPolicy::Sequential);
    let queries = [0.25, 1.5, 2.75];
    let view = NdView::new(&queries, StridedLayout::new(&[3, 1], &[1, 1]).unwrap()).unwrap();
    let mut storage = [0.0; 6];
    let mut batch_ws = batch.workspace(BatchRequest::gradients(3)).unwrap();
    {
        let mut out = NdViewMut::new(
            &mut storage,
            StridedLayout::new(&[3, 1, 2], &[2, 2, 1]).unwrap(),
        )
        .unwrap();
        batch
            .gradient_into(
                &view,
                policy,
                &mut out,
                None,
                &mut batch_ws,
                ExecutionContext::sequential(),
            )
            .unwrap();
    };
    for (i, &t) in queries.iter().enumerate() {
        let [expected] = spline.gradient([t], policy).unwrap();
        assert_eq!(storage[2 * i..2 * i + 2], expected);
    }
}

#[test]
fn linear_constructor_matches_piecewise_linear_interpolation() {
    let x = vec![-1.0_f32, 0.5, 2.0, 2.5];
    let y = vec![2.0, -1.0, 0.5, 4.0];
    let line = BSpline::linear(x.clone(), y.clone()).unwrap();
    let reject = ExtrapolationPolicy::Reject;
    for (&node, &value) in x.iter().zip(&y) {
        assert_eq!(line.evaluate([node], reject).unwrap(), value);
    }
    for w in x.windows(2).zip(y.windows(2)) {
        let (xs, ys) = w;
        let t = 0.25 * xs[0] + 0.75 * xs[1];
        let expected = ys[0] + (ys[1] - ys[0]) * 0.75;
        assert!((line.evaluate([t], reject).unwrap() - expected).abs() < 4e-6);
        assert!(
            (line.gradient([t], reject).unwrap()[0] - (ys[1] - ys[0]) / (xs[1] - xs[0])).abs()
                < 4e-6
        );
    }
    // Continuation extends the outer segments; the kink at a node has no
    // ordinary derivative but both one-sided slopes exist.
    let continued = line
        .evaluate([3.0], ExtrapolationPolicy::MethodDefined)
        .unwrap();
    assert!((continued - 7.5).abs() < 4e-6);
    assert!(line.derivative([0.5], [1], None, reject).is_err());
    assert!(
        (line
            .derivative([0.5], [1], Some([KnotSide::Left]), reject)
            .unwrap()
            + 2.0)
            .abs()
            < 4e-6
    );
    assert!(BSpline::linear(vec![0.0_f32, 1.0, 1.0], vec![0.0, 1.0, 2.0]).is_err());
}
