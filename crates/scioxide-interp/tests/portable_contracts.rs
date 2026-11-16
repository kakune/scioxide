//! Independent analytic oracles shared by both portable scalar widths.
#![allow(clippy::unwrap_used)]
mod support;

use support::{assert_close, assert_error};

use scioxide_interp::{
    BarycentricInterpolator, BicubicInterpolator, BilinearInterpolator, BulirschStoerInterpolator,
    CubicSpline, Differentiable, ExtrapolationPolicy, HermiteInterpolator, Integrable, Interpolant,
    NevilleInterpolator, NewtonInterpolator, PchipInterpolator, PreparedPolynomialAntiderivative,
    SplineBoundary,
};

macro_rules! portable_cases {
    ($module:ident, $scalar:ty, $target:expr) => {
        mod $module {
            use super::*;
            type S = $scalar;
            const TOL: f64 = ($target as f64) * 0.5;
            const POLICY: ExtrapolationPolicy<S> = ExtrapolationPolicy::MethodDefined;

            fn fused(interp: &BarycentricInterpolator<S>, query: S) -> (S, S) {
                let mut workspace = interp.workspace().unwrap();
                let (mut value, mut derivative) = (0.0, 0.0);
                interp
                    .evaluate_with_derivative_into(
                        [query],
                        POLICY,
                        &mut value,
                        &mut derivative,
                        &mut workspace,
                    )
                    .unwrap();
                (value, derivative)
            }
            #[test]
            fn singleton_and_subnormal_barycentric() {
                let tiny = S::from_bits(1);
                let singleton = BarycentricInterpolator::new(vec![0.0], vec![tiny]).unwrap();
                for query in [-S::MAX, -tiny, 0.0, tiny, S::MAX] {
                    assert_eq!(fused(&singleton, query), (tiny, 0.0));
                }
                assert_error(
                    &singleton.evaluate([S::INFINITY], POLICY),
                    scioxide_core::SciOxideErrorKind::NonFinite,
                );
                let nodes = [0.0, tiny * 4.0, tiny * 8.0];
                let linear = BarycentricInterpolator::new(nodes.to_vec(), nodes.to_vec()).unwrap();
                for query in [0.0, tiny, tiny * 2.0, tiny * 4.0, tiny * 7.0, tiny * 8.0] {
                    let (value, derivative) = fused(&linear, query);
                    assert_eq!(value, query);
                    assert_close(derivative, 1.0, TOL, TOL);
                }
            }
            #[test]
            fn polynomial_value_derivative_and_integral_oracles() {
                let x: [S; 4] = [-1.0, 0.0, 1.0, 2.0];
                let f = |x: S| x * x * x - 2.0 * x + 1.0;
                let df = |x: S| 3.0 * x * x - 2.0;
                let y = x.map(f);
                let bary = BarycentricInterpolator::new(x.to_vec(), y.to_vec()).unwrap();
                let newton = NewtonInterpolator::new(x.to_vec(), y.to_vec()).unwrap();
                let hermite =
                    HermiteInterpolator::new(x.to_vec(), y.to_vec(), x.map(df).to_vec()).unwrap();
                let spline = CubicSpline::with_boundary(
                    x.to_vec(),
                    y.to_vec(),
                    SplineBoundary::Clamped(df(-1.0), df(2.0)),
                )
                .unwrap();
                fn global<I: Differentiable<S, 1, Value = S>>(
                    interp: &I,
                    prepared: &PreparedPolynomialAntiderivative<S>,
                    f: impl Fn(S) -> S,
                    df: impl Fn(S) -> S,
                ) {
                    for t in [-0.75, -0.25, 0.25, 0.75, 1.5] {
                        assert_close(interp.evaluate([t], POLICY).unwrap(), f(t), TOL, TOL);
                        assert_close(interp.gradient([t], POLICY).unwrap()[0], df(t), TOL, TOL);
                    }
                    assert_close(
                        prepared.integral(-1.0, 2.0, POLICY).unwrap(),
                        3.75,
                        TOL,
                        TOL,
                    );
                }
                for t in [-0.75, -0.25, 0.25, 0.75, 1.5] {
                    assert_close(spline.evaluate([t], POLICY).unwrap(), f(t), TOL, TOL);
                    assert_close(spline.gradient([t], POLICY).unwrap()[0], df(t), TOL, TOL);
                }
                assert_close(
                    spline.integral([-1.0], [2.0], POLICY).unwrap(),
                    3.75,
                    TOL,
                    TOL,
                );
                global(&bary, &bary.prepare_antiderivative().unwrap(), f, df);
                global(&newton, &newton.prepare_antiderivative().unwrap(), f, df);
                global(&hermite, &hermite.prepare_antiderivative().unwrap(), f, df);
            }
            #[test]
            fn shape_and_rational_oracles() {
                let x: [S; 4] = [0.0, 1.0, 2.0, 3.0];
                let pchip =
                    PchipInterpolator::new(x.to_vec(), x.map(|x| 2.0 * x + 1.0).to_vec()).unwrap();
                let rational =
                    BulirschStoerInterpolator::new(x.to_vec(), x.map(|x| 1.0 / (1.0 + x)).to_vec())
                        .unwrap();
                for t in [0.25, 0.75, 1.25, 2.5] {
                    assert_close(
                        pchip.evaluate([t], POLICY).unwrap(),
                        2.0 * t + 1.0,
                        TOL,
                        TOL,
                    );
                    assert_close(pchip.gradient([t], POLICY).unwrap()[0], 2.0, TOL, TOL);
                    assert_close(
                        rational.evaluate([t], POLICY).unwrap(),
                        1.0 / (1.0 + t),
                        TOL,
                        TOL,
                    );
                }
                let bspline =
                    scioxide_interp::BSpline::linear(vec![0.0, 3.0], vec![1.0, 7.0]).unwrap();
                assert_close(bspline.evaluate([0.75], POLICY).unwrap(), 2.5, TOL, TOL);
                assert_close(bspline.gradient([0.75], POLICY).unwrap()[0], 2.0, TOL, TOL);
                let neville =
                    NevilleInterpolator::new(x.to_vec(), x.map(|x| x * x).to_vec()).unwrap();
                let (mut value, mut estimate) = (0.0, 0.0);
                neville
                    .evaluate_with_error_into(
                        [0.75],
                        POLICY,
                        &mut value,
                        &mut estimate,
                        &mut neville.workspace().unwrap(),
                    )
                    .unwrap();
                assert_close(value, 0.5625, TOL, TOL);
                assert!(estimate.is_finite());
                let z: Vec<S> = x
                    .iter()
                    .flat_map(|&a| x.map(|b| b + 2.0 * a + a * b))
                    .collect();
                let bilinear =
                    BilinearInterpolator::new(x.to_vec(), x.to_vec(), z.clone()).unwrap();
                let bicubic = BicubicInterpolator::new(x.to_vec(), x.to_vec(), z).unwrap();
                assert_close(
                    bilinear.evaluate([0.25, 1.5], POLICY).unwrap(),
                    3.625,
                    TOL,
                    TOL,
                );
                assert_close(
                    bicubic.evaluate([0.25, 1.5], POLICY).unwrap(),
                    3.625,
                    TOL,
                    TOL,
                );
            }
        }
    };
}
portable_cases!(binary32, f32, 3e-5);
portable_cases!(binary64, f64, 3e-12);
