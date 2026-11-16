//! Analytic tests for the common query contract; no comparator is an oracle.
#![allow(clippy::unwrap_used)]
mod support;

use support::{assert_close, assert_error};

use scioxide_core::{Complex, Interval};
use scioxide_interp::{
    BSpline, BSplineBasis, BarycentricInterpolator, BicubicInterpolator, BilinearInterpolator,
    BulirschStoerInterpolator, CubicSpline, ExtrapolationPolicy, HermiteInterpolator,
    NevilleInterpolator, NewtonInterpolator, PchipInterpolator, PreparedPolynomialAntiderivative,
    SplineBoundary,
};
use scioxide_interp::{Differentiable, Integrable, Interpolant, TwiceDifferentiable};
macro_rules! cases {
    ($module:ident, $scalar:ty, $tol:expr) => {
        mod $module {
            use super::*;
            type S = $scalar;
            const TOL: f64 = ($tol as f64) * 0.5;
            const CLAMP: ExtrapolationPolicy<S> = ExtrapolationPolicy::Clamp;
            const REJECT: ExtrapolationPolicy<S> = ExtrapolationPolicy::Reject;

            #[test]
            fn classification_and_limits() {
                let domain = Interval::new(0.0 as S, 2.0).unwrap();
                let report = CLAMP.resolve([domain, domain], [-1.0, 3.0]).unwrap();
                assert_eq!(report.original(), [-1.0, 3.0]);
                assert_eq!(report.effective(), [0.0, 2.0]);
                assert_eq!(report.outside_distance(), [1.0, 1.0]);
                let bounded = ExtrapolationPolicy::BoundedContinue { max_distance: 1.0 };
                assert_eq!(
                    bounded.resolve([domain], [-1.0]).unwrap().effective(),
                    [-1.0]
                );
                assert_error(
                    &bounded.resolve([domain], [-1.0 - S::EPSILON]),
                    scioxide_core::SciOxideErrorKind::DomainError,
                );
                for limit in [-1.0, S::NAN, S::INFINITY] {
                    assert_error(
                        &(ExtrapolationPolicy::BoundedContinue {
                            max_distance: limit,
                        })
                        .resolve([domain], [1.0]),
                        scioxide_core::SciOxideErrorKind::InvalidInput,
                    );
                }
                for policy in [REJECT, CLAMP, bounded, ExtrapolationPolicy::MethodDefined] {
                    assert_error(
                        &policy.resolve([domain], [S::NAN]),
                        scioxide_core::SciOxideErrorKind::NonFinite,
                    );
                    assert_eq!(
                        policy.resolve([domain], [2.0]).unwrap().outside_distance(),
                        [0.0]
                    );
                }
                let negative_tiny = Interval::new(-S::MIN_POSITIVE, -S::MIN_POSITIVE).unwrap();
                assert_error(
                    &bounded.resolve([negative_tiny], [1.0]),
                    scioxide_core::SciOxideErrorKind::DomainError,
                );
                let singleton = Interval::new(-S::MAX, -S::MAX).unwrap();
                let far = CLAMP.resolve([singleton], [S::MAX]).unwrap();
                assert_eq!(far.effective(), [-S::MAX]);
                assert_eq!(far.outside_distance(), [S::INFINITY]);
                assert_error(
                    &bounded.resolve([singleton], [S::MAX]),
                    scioxide_core::SciOxideErrorKind::DomainError,
                );
            }
            /// Constant-extension calculus of `f(x) = 1 + 2x` sampled on `[0, 2]`.
            fn constant_extension<I: Differentiable<S, 1, Value = S> + Integrable<S, 1>>(f: &I) {
                let bounded = ExtrapolationPolicy::BoundedContinue { max_distance: 1.0 };
                assert_close(f.evaluate([-1.0], CLAMP).unwrap(), 1.0, TOL, TOL);
                assert_close(f.evaluate([3.0], CLAMP).unwrap(), 5.0, TOL, TOL);
                assert_close(f.evaluate([3.0], bounded).unwrap(), 7.0, TOL, TOL);
                assert_error(
                    &f.evaluate([3.0], REJECT),
                    scioxide_core::SciOxideErrorKind::DomainError,
                );
                assert_eq!(f.gradient([-1.0], CLAMP).unwrap(), [0.0]);
                assert_eq!(f.gradient([3.0], CLAMP).unwrap(), [0.0]);
                assert_close(f.gradient([0.0], CLAMP).unwrap()[0], 2.0, TOL, TOL);
                assert_close(f.gradient([2.0], CLAMP).unwrap()[0], 2.0, TOL, TOL);
                assert_close(f.integral([-1.0], [3.0], CLAMP).unwrap(), 12.0, TOL, TOL);
                assert_close(f.integral([-3.0], [-1.0], CLAMP).unwrap(), 2.0, TOL, TOL);
                assert_close(f.integral([3.0], [5.0], CLAMP).unwrap(), 10.0, TOL, TOL);
                assert_close(f.integral([3.0], [3.0], CLAMP).unwrap(), 0.0, TOL, TOL);
                assert_close(f.integral([-1.0], [3.0], bounded).unwrap(), 12.0, TOL, TOL);
                assert_error(
                    &f.integral([-2.0], [3.0], bounded),
                    scioxide_core::SciOxideErrorKind::DomainError,
                );
                assert_error(
                    &f.integral([1.0], [0.0], CLAMP),
                    scioxide_core::SciOxideErrorKind::InvalidInput,
                );
            }
            #[test]
            fn scalar_families_share_constant_extension_calculus() {
                let nodes = vec![0.0 as S, 1.0, 2.0];
                let values = vec![1.0 as S, 3.0, 5.0]; // f(x)=1+2x
                let barycentric =
                    BarycentricInterpolator::new(nodes.clone(), values.clone()).unwrap();
                let newton = NewtonInterpolator::new(nodes.clone(), values.clone()).unwrap();
                let hermite =
                    HermiteInterpolator::new(nodes.clone(), values.clone(), vec![2.0; 3]).unwrap();
                let pchip = PchipInterpolator::new(nodes.clone(), values.clone()).unwrap();
                let cubic = CubicSpline::with_boundary(
                    nodes.clone(),
                    values.clone(),
                    SplineBoundary::Natural,
                )
                .unwrap();
                let linear = BSpline::linear(nodes, values).unwrap();
                constant_extension(&pchip);
                constant_extension(&cubic);
                constant_extension(&linear);
                // Global polynomials: protocol queries and prepared integrals.
                fn global<I: Differentiable<S, 1, Value = S>>(
                    f: &I,
                    prepared: &PreparedPolynomialAntiderivative<S>,
                ) {
                    let bounded = ExtrapolationPolicy::BoundedContinue { max_distance: 1.0 };
                    assert_close(f.evaluate([-1.0], CLAMP).unwrap(), 1.0, TOL, TOL);
                    assert_close(f.evaluate([3.0], CLAMP).unwrap(), 5.0, TOL, TOL);
                    assert_close(f.evaluate([3.0], bounded).unwrap(), 7.0, TOL, TOL);
                    assert_error(
                        &f.evaluate([3.0], REJECT),
                        scioxide_core::SciOxideErrorKind::DomainError,
                    );
                    assert_eq!(f.gradient([-1.0], CLAMP).unwrap(), [0.0]);
                    assert_eq!(f.gradient([3.0], CLAMP).unwrap(), [0.0]);
                    assert_close(f.gradient([0.0], CLAMP).unwrap()[0], 2.0, TOL, TOL);
                    assert_close(f.gradient([2.0], CLAMP).unwrap()[0], 2.0, TOL, TOL);
                    assert_close(prepared.integral(-1.0, 3.0, CLAMP).unwrap(), 12.0, TOL, TOL);
                    assert_close(prepared.integral(-3.0, -1.0, CLAMP).unwrap(), 2.0, TOL, TOL);
                    assert_close(prepared.integral(3.0, 5.0, CLAMP).unwrap(), 10.0, TOL, TOL);
                    assert_close(prepared.integral(3.0, 3.0, CLAMP).unwrap(), 0.0, TOL, TOL);
                    assert_close(
                        prepared.integral(-1.0, 3.0, bounded).unwrap(),
                        12.0,
                        TOL,
                        TOL,
                    );
                    assert_error(
                        &prepared.integral(-2.0, 3.0, bounded),
                        scioxide_core::SciOxideErrorKind::DomainError,
                    );
                    assert_error(
                        &prepared.integral(1.0, 0.0, CLAMP),
                        scioxide_core::SciOxideErrorKind::InvalidInput,
                    );
                }
                global(&barycentric, &barycentric.prepare_antiderivative().unwrap());
                global(&newton, &newton.prepare_antiderivative().unwrap());
                global(&hermite, &hermite.prepare_antiderivative().unwrap());
                let singleton = BarycentricInterpolator::new(vec![1.0 as S], vec![4.0]).unwrap();
                assert_close(
                    singleton
                        .prepare_antiderivative()
                        .unwrap()
                        .integral(-1.0, 3.0, CLAMP)
                        .unwrap(),
                    16.0,
                    TOL,
                    TOL,
                );
                assert_eq!(singleton.gradient([-1.0], CLAMP).unwrap(), [0.0]);
            }
            #[test]
            fn quadratic_integral_tails_and_affine_outputs() {
                let nodes = vec![0.0 as S, 1.0, 2.0];
                let polynomial = NewtonInterpolator::new(
                    nodes,
                    vec![
                        Complex::new(0.0, 1.0),
                        Complex::new(1.0, 3.0),
                        Complex::new(4.0, 5.0),
                    ],
                )
                .unwrap();
                let prepared = polynomial.prepare_antiderivative().unwrap();
                let owned = prepared.integral(-1.0, 3.0, CLAMP).unwrap();
                assert_close(owned.re, 20.0 / 3.0, TOL, TOL);
                assert_close(owned.im, 12.0, TOL, ($tol as f64) * 0.5);
                let mut output = [99.0 as S; 2];
                let mut scratch = [0.0; 2];
                prepared
                    .integral_into(-1.0, 3.0, CLAMP, &mut output, &mut scratch)
                    .unwrap();
                assert_close(output[0], owned.re, TOL, TOL);
                assert_close(output[1], owned.im, TOL, TOL);
                let mut workspace = polynomial.workspace().unwrap();
                polynomial
                    .gradient_into([-1.0], CLAMP, &mut output, &mut workspace)
                    .unwrap();
                assert_eq!(output, [0.0, 0.0]);
                output = [99.0; 2];
                assert!(
                    polynomial
                        .evaluate_into([3.0], REJECT, &mut output, &mut workspace)
                        .is_err()
                );
                assert_eq!(output, [99.0; 2]);
                polynomial
                    .evaluate_into([3.0], CLAMP, &mut output, &mut workspace)
                    .unwrap();
                assert_eq!(output, [4.0, 5.0]);
                let vector = NewtonInterpolator::new(
                    vec![0.0 as S, 1.0],
                    vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                )
                .unwrap();
                assert_eq!(vector.gradient([2.0], CLAMP).unwrap()[0], vec![0.0, 0.0]);
                assert_eq!(
                    vector
                        .prepare_antiderivative()
                        .unwrap()
                        .integral(-1.0, 2.0, CLAMP)
                        .unwrap(),
                    vec![6.0, 9.0]
                );
            }
            #[test]
            fn independent_method_workspaces_reuse_after_policy_and_numerical_failures() {
                let rational =
                    BulirschStoerInterpolator::new(vec![0.0 as S, 1.0], vec![1.0, -1.0]).unwrap();
                let mut rational_scratch = rational.workspace().unwrap();
                let mut rational_value = 0.0;
                assert!(
                    rational
                        .evaluate_into([0.5], REJECT, &mut rational_value, &mut rational_scratch)
                        .is_err()
                );
                rational
                    .evaluate_into([-1.0], CLAMP, &mut rational_value, &mut rational_scratch)
                    .unwrap();
                assert_close(rational_value, 1.0, TOL, TOL);
                assert_close(rational.evaluate([-1.0], CLAMP).unwrap(), 1.0, TOL, TOL);
                let basis = BSplineBasis::new(vec![0.0 as S, 0.0, 1.0, 1.0], 1).unwrap();
                let spline = BSpline::new([basis], vec![1.0, 3.0]).unwrap();
                let mut spline_scratch = spline.workspace().unwrap();
                let mut slope = 9.0;
                spline
                    .derivative_into([-1.0], [1], None, CLAMP, &mut slope, &mut spline_scratch)
                    .unwrap();
                assert_eq!(slope, 0.0);
                let mut value = 9.0;
                spline
                    .evaluate_into([2.0], CLAMP, &mut value, &mut spline_scratch)
                    .unwrap();
                assert_close(value, 3.0, TOL, ($tol as f64) * 0.5);
                assert_close(spline.evaluate([2.0], CLAMP).unwrap(), 3.0, TOL, TOL);
                let polynomial =
                    NevilleInterpolator::new(vec![0.0 as S, 1.0], vec![1.0, 3.0]).unwrap();
                let mut neville = polynomial.workspace().unwrap();
                let mut run = |policy| {
                    let (mut value, mut estimate) = (0.0, 0.0);
                    polynomial
                        .evaluate_with_error_into(
                            [2.0],
                            policy,
                            &mut value,
                            &mut estimate,
                            &mut neville,
                        )
                        .map(|()| (value, estimate))
                };
                assert!(run(REJECT).is_err());
                assert_close(run(CLAMP).unwrap().0, 3.0, TOL, TOL);
                assert_close(
                    run(ExtrapolationPolicy::MethodDefined).unwrap().0,
                    5.0,
                    TOL,
                    TOL,
                );
            }
            #[test]
            fn scattered_uses_bounding_rectangle_and_checked_continuation() {
                use scioxide_core::RankThreshold;
                use scioxide_interp::{RbfInterpolator, RbfKernel, RbfOptions};
                let options = RbfOptions::new()
                    .with_rank_threshold(RankThreshold::new(0.0, S::EPSILON * 32.0).unwrap());
                let rbf = RbfInterpolator::with_options(
                    vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                    vec![1.0 as S, 2.0, 3.0],
                    RbfKernel::Gaussian(1.0 as S),
                    options,
                )
                .unwrap();
                assert_close(rbf.evaluate([-1.0, -1.0], CLAMP).unwrap(), 1.0, TOL, TOL);
                assert_error(
                    &rbf.evaluate([-1.0, 0.0], REJECT),
                    scioxide_core::SciOxideErrorKind::DomainError,
                );
                let bounded = ExtrapolationPolicy::BoundedContinue { max_distance: 0.5 };
                assert_close(
                    rbf.evaluate([-0.5, 0.25], bounded).unwrap(),
                    rbf.evaluate([-0.5, 0.25], ExtrapolationPolicy::MethodDefined)
                        .unwrap(),
                    TOL,
                    TOL,
                );
                assert_error(
                    &rbf.evaluate([-1.0, 0.0], bounded),
                    scioxide_core::SciOxideErrorKind::DomainError,
                );
                // The missing upper-right triangle remains inside the documented rectangle.
                let mut workspace = rbf.workspace().unwrap();
                assert_eq!(
                    rbf.report([1.0, 1.0], REJECT, &mut workspace)
                        .unwrap()
                        .outside_distance(),
                    [0.0; 2]
                );
            }
            #[test]
            fn bicubic_clamped_axes_and_mixed_derivatives() {
                let nodes = vec![0.0 as S, 1.0, 2.0];
                let values = nodes
                    .iter()
                    .flat_map(|&y| nodes.iter().map(move |&x| x * x + y * y + x * y))
                    .collect();
                let bicubic = BicubicInterpolator::new(nodes.clone(), nodes, values).unwrap();
                let hessian = &bicubic;
                assert_close(hessian.evaluate([-1.0, 1.0], CLAMP).unwrap(), 1.0, TOL, TOL);
                let gradient = hessian.gradient([-1.0, 1.0], CLAMP).unwrap();
                assert_eq!(gradient[0], 0.0);
                assert_close(gradient[1], 2.0, TOL, ($tol as f64) * 0.5);
                let h = hessian.hessian([-1.0, 1.0], CLAMP).unwrap();
                assert_eq!(h[0], [0.0, 0.0]);
                assert_eq!(h[1][0], 0.0);
                assert_close(h[1][1], 2.0, TOL, ($tol as f64) * 0.5);
                let transposed = hessian.hessian([1.0, -1.0], CLAMP).unwrap();
                assert_close(transposed[0][0], 2.0, TOL, TOL);
                assert_eq!(transposed[1], [0.0, 0.0]);
                assert_eq!(hessian.hessian([-1.0, 3.0], CLAMP).unwrap(), [[0.0; 2]; 2]);
                assert_close(
                    hessian.hessian([0.0, 1.0], CLAMP).unwrap()[0][1],
                    1.0,
                    TOL,
                    TOL,
                );
                let bilinear = BilinearInterpolator::new(
                    vec![0.0 as S, 1.0],
                    vec![0.0, 1.0],
                    vec![0.0, 1.0, 2.0, 3.0],
                )
                .unwrap();
                assert_close(
                    bilinear.evaluate([-1.0, 2.0], CLAMP).unwrap(),
                    2.0,
                    TOL,
                    TOL,
                );
            }
        }
    };
}
cases!(binary32, f32, 8e-5);
cases!(binary64, f64, 2e-12);
