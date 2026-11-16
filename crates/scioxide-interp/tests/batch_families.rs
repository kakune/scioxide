//! Independent affine and manufactured-kernel tests in both scalar widths.
#![allow(clippy::unwrap_used)]
mod support;

use support::assert_close;

use scioxide_core::{
    Complex, ExecutionContext, ExecutionPolicy, Interval, NdView, NdViewMut, RankThreshold,
    StridedLayout,
};
use scioxide_interp::*;
macro_rules! cases {
    ($module:ident,$scalar:ty,$tol:expr) => {
        mod $module {
            use super::*;
            type S = $scalar;
            const TOL: f64 = ($tol as f64) * 0.5;
            fn report<const D: usize>() -> ExtrapolationReport<S, D> {
                scioxide_interp::ExtrapolationPolicy::MethodDefined
                    .resolve(
                        [99.; D].map(|_| scioxide_core::Interval::new(0.0, 1.0).unwrap()),
                        [99.; D],
                    )
                    .unwrap()
            }

            /// Native complex values through the batch shell and point forms.
            fn spline_values<I: Interpolant<S, 1, Value = Complex<S>>>(model: I) {
                let plan = BatchEvaluator::new(&model, ExecutionPolicy::Sequential);
                let query = [0.25, 0.75, 1.5];
                let q = NdView::new(&query, StridedLayout::new(&[3, 1], &[1, 1]).unwrap()).unwrap();
                let mut values = [99.; 6];
                let mut out =
                    NdViewMut::new(&mut values, StridedLayout::new(&[3, 2], &[2, 1]).unwrap())
                        .unwrap();
                let mut w = plan.workspace(BatchRequest::values(3)).unwrap();
                plan.evaluate_into(
                    &q,
                    ExtrapolationPolicy::Reject,
                    &mut out,
                    None,
                    &mut w,
                    ExecutionContext::sequential_native(),
                )
                .unwrap();
                let many = model
                    .evaluate_many(&query.map(|x| [x]), ExtrapolationPolicy::Reject)
                    .unwrap();
                for (i, x) in query.into_iter().enumerate() {
                    let point = model.evaluate([x], ExtrapolationPolicy::Reject).unwrap();
                    assert_eq!([values[2 * i], values[2 * i + 1]], [point.re, point.im]);
                    assert_eq!([many[i].re, many[i].im], [point.re, point.im]);
                    assert_close(values[2 * i], 1. + 2. * x, TOL, TOL);
                    assert_close(values[2 * i + 1], 3. + 4. * x, TOL, TOL);
                }
                let outside = model.evaluate_many(&[[0.5], [9.0]], ExtrapolationPolicy::Reject);
                assert_eq!(
                    outside.unwrap_err().kind(),
                    model
                        .evaluate([9.0], ExtrapolationPolicy::Reject)
                        .unwrap_err()
                        .kind()
                );
                let value = model.evaluate([0.5], ExtrapolationPolicy::Reject).unwrap();
                assert_close(value.re, 2., TOL, ($tol as f64) * 0.5);
                assert_close(value.im, 5., TOL, ($tol as f64) * 0.5);
            }
            #[test]
            fn piecewise_and_spline_complex_values() {
                let nodes = vec![0., 1., 2.];
                let y = vec![
                    Complex::new(1., 3.),
                    Complex::new(3., 7.),
                    Complex::new(5., 11.),
                ];
                spline_values(PchipInterpolator::new(nodes.clone(), y.clone()).unwrap());
                for method in [AkimaMethod::Akima, AkimaMethod::Makima] {
                    spline_values(
                        AkimaInterpolator::with_method(nodes.clone(), y.clone(), method).unwrap(),
                    );
                }
                spline_values(
                    CubicSpline::with_boundary(nodes.clone(), y.clone(), SplineBoundary::Natural)
                        .unwrap(),
                );
                spline_values(BSpline::linear(nodes, y).unwrap());
            }
            /// Native complex values through the batch shell, against analytic lines.
            fn native<I: Interpolant<S, 1, Value = Complex<S>>>(model: &I) {
                let plan = BatchEvaluator::new(model, ExecutionPolicy::Sequential);
                let mut w = plan
                    .workspace(BatchRequest::values(3).with_reports())
                    .unwrap();
                let query = [0.25, 0.75, 1.5];
                let q = NdView::new(&query, StridedLayout::new(&[3, 1], &[1, 1]).unwrap()).unwrap();
                let mut values = [99.; 6];
                let mut reports = [report(); 3];
                let mut out =
                    NdViewMut::new(&mut values, StridedLayout::new(&[3, 2], &[2, 1]).unwrap())
                        .unwrap();
                plan.evaluate_into(
                    &q,
                    ExtrapolationPolicy::Reject,
                    &mut out,
                    Some(&mut reports),
                    &mut w,
                    ExecutionContext::sequential(),
                )
                .unwrap();
                for (i, x) in query.into_iter().enumerate() {
                    assert_close(values[2 * i], 1. + 2. * x, TOL, TOL);
                    assert_close(values[2 * i + 1], 3. + 4. * x, TOL, TOL);
                }
                let value = model.evaluate([0.5], ExtrapolationPolicy::Reject).unwrap();
                assert_close(value.re, 2., TOL, ($tol as f64) * 0.5);
                assert_close(value.im, 5., TOL, ($tol as f64) * 0.5);
            }
            #[test]
            fn native_complex_polynomial_families() {
                let nodes = vec![0., 1., 2.];
                let y = vec![
                    Complex::new(1., 3.),
                    Complex::new(3., 7.),
                    Complex::new(5., 11.),
                ];
                native(&BarycentricInterpolator::new(nodes.clone(), y.clone()).unwrap());
                native(&NewtonInterpolator::new(nodes.clone(), y.clone()).unwrap());
                native(
                    &HermiteInterpolator::new(
                        nodes.clone(),
                        y.clone(),
                        vec![Complex::new(2., 4.); 3],
                    )
                    .unwrap(),
                );
                native(&NevilleInterpolator::new(nodes.clone(), y.clone()).unwrap());
                native(&FloaterHormannInterpolator::new(nodes, y, 1).unwrap());
            }
            #[test]
            fn native_complex_rbf_and_manufactured_combination() {
                let points = [[0., 0.], [1., 0.], [0., 1.]];
                let kernel = RbfKernel::Gaussian(1.);
                let f = |x: S, y: S| -> Complex<S> {
                    points
                        .iter()
                        .zip([
                            Complex::new(1., 2.),
                            Complex::new(-0.5, 1.),
                            Complex::new(2., -1.),
                        ])
                        .fold(Complex::new(0., 0.), |sum, (&[a, b], weight)| {
                            let phi = kernel.evaluate((x - a).hypot(y - b)).unwrap();
                            sum + weight * Complex::new(phi, 0.)
                        })
                };
                let values = points.map(|[x, y]| f(x, y));
                let options = RbfOptions::new()
                    .with_rank_threshold(RankThreshold::new(0., $tol * 0.01).unwrap());
                let model = RbfInterpolator::with_options(
                    points.to_vec(),
                    values.to_vec(),
                    kernel,
                    options,
                )
                .unwrap();
                let mut w = model.workspace().unwrap();
                let mut out = Complex::new(99., 99.);
                model
                    .evaluate_into([0.25, 0.5], ExtrapolationPolicy::Reject, &mut out, &mut w)
                    .unwrap();
                let expected = f(0.25, 0.5);
                assert_close(out.re, expected.re, TOL, TOL);
                assert_close(out.im, expected.im, TOL, TOL);
                let owned = model
                    .evaluate([0.25, 0.5], ExtrapolationPolicy::Reject)
                    .unwrap();
                assert_eq!(out, owned);
            }
            #[test]
            fn bicubic_vector_gradients_and_hessians() {
                let x = vec![-1., 0., 0.5, 2.];
                let y = vec![-2., 0., 1.];
                let values: Vec<[S; 2]> = y
                    .iter()
                    .flat_map(|&b| {
                        x.iter().map(move |&a| {
                            [a * a + b * b + a * b, 3. * a * a - 2. * b * b + 4. * a * b]
                        })
                    })
                    .collect();
                let model = BicubicInterpolator::new(x, y, values).unwrap();
                let mut w = model.workspace().unwrap();
                let mut gradient = [[99.; 2]; 2];
                let mut hessian = [[[99.; 2]; 2]; 2];
                model
                    .gradient_into(
                        [0.25, 0.5],
                        ExtrapolationPolicy::Reject,
                        &mut gradient,
                        &mut w,
                    )
                    .unwrap();
                model
                    .hessian_into(
                        [0.25, 0.5],
                        ExtrapolationPolicy::Reject,
                        &mut hessian,
                        &mut w,
                    )
                    .unwrap();
                assert_eq!(
                    model
                        .gradient([0.25, 0.5], ExtrapolationPolicy::Reject)
                        .unwrap(),
                    gradient
                );
                for (a, b) in gradient.into_iter().flatten().zip([1., 3.5, 1.25, -1.]) {
                    assert_close(a, b, TOL, ($tol as f64) * 0.5);
                }
                for (a, b) in hessian
                    .into_iter()
                    .flatten()
                    .flatten()
                    .zip([2., 6., 1., 4., 1., 4., 2., -4.])
                {
                    assert_close(a, b, TOL, ($tol as f64) * 0.5);
                }
            }
            #[test]
            fn native_complex_rational_queries() {
                let nodes: [S; 3] = [0., 1., 2.];
                let values = nodes.map(|x| Complex::new(1. + 2. * x, 3. - x));
                let plan =
                    FloaterHormannInterpolator::new(nodes.to_vec(), values.to_vec(), 1).unwrap();
                let mut w = plan.workspace().unwrap();
                let mut out = Complex::new(0., 0.);
                plan.evaluate_into([0.5], ExtrapolationPolicy::Reject, &mut out, &mut w)
                    .unwrap();
                assert_close(out.re, 2., TOL, ($tol as f64) * 0.5);
                assert_close(out.im, 2.5, TOL, ($tol as f64) * 0.5);
                let rational = BarycentricRational::new(
                    nodes.map(|x| Complex::new(x, 0.)).to_vec(),
                    values.to_vec(),
                    vec![Complex::new(1., 0.); 3],
                    RationalDomain::Real(Interval::new(0., 2.).unwrap()),
                )
                .unwrap();
                // A complex plan has both coordinate forms; name the one-coordinate form.
                let rational_plan =
                    BatchEvaluator::<S, _, 1>::new(&rational, ExecutionPolicy::Sequential);
                let mut rw = rational_plan.workspace(BatchRequest::values(1)).unwrap();
                let mut storage = [0.; 2];
                let mut view =
                    NdViewMut::new(&mut storage, StridedLayout::new(&[1, 2], &[2, 1]).unwrap())
                        .unwrap();
                rational_plan
                    .evaluate_into(
                        &NdView::new(&[1.], StridedLayout::new(&[1, 1], &[1, 1]).unwrap()).unwrap(),
                        ExtrapolationPolicy::Reject,
                        &mut view,
                        None,
                        &mut rw,
                        ExecutionContext::sequential(),
                    )
                    .unwrap();
                assert_eq!(storage, [3., 2.]);
            }
        }
    };
}
cases!(single, f32, 3e-5);
cases!(double, f64, 3e-12);
