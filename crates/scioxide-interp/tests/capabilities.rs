#![allow(clippy::unwrap_used)]

mod support;

use support::assert_error;

use scioxide_interp::{
    BSpline, BSplineBasis, BarycentricInterpolator, BulirschStoerInterpolator, Differentiable,
    ExtrapolationPolicy, Interpolant,
};

#[test]
fn capabilities_dispatch_independently() {
    let rational =
        BulirschStoerInterpolator::new(vec![0.0_f64, 1.0, 2.0], vec![1.0, 0.5, 0.2]).unwrap();
    assert!(
        (rational
            .evaluate([1.0], ExtrapolationPolicy::Reject)
            .unwrap()
            - 0.5)
            .abs()
            < 1e-12
    );

    let bspline = quadratic();
    let policy = ExtrapolationPolicy::Reject;
    assert!((bspline.gradient([0.5], policy).unwrap()[0] - 1.0).abs() < 1e-12);
    assert!((bspline.evaluate([0.5], policy).unwrap() - 0.5).abs() < 1e-12);

    let barycentric =
        BarycentricInterpolator::new(vec![0.0_f64, 1.0, 2.0], vec![0.0, 1.0, 4.0]).unwrap();
    // Global polynomials integrate through their prepared antiderivative.
    let prepared = barycentric.prepare_antiderivative().unwrap();
    assert!(
        (prepared
            .integral(0.0, 2.0, ExtrapolationPolicy::Reject)
            .unwrap()
            - 8.0 / 3.0)
            .abs()
            < 1e-12
    );
    assert!(
        (barycentric
            .evaluate([1.5], ExtrapolationPolicy::Reject)
            .unwrap()
            - 2.25)
            .abs()
            < 1e-12
    );
}

#[test]
fn supported_capabilities_keep_structured_failures() {
    let interpolator =
        BarycentricInterpolator::new(vec![0.0_f64, 1.0, 2.0], vec![0.0, 1.0, 4.0]).unwrap();

    assert_error(
        &interpolator.gradient([3.0], ExtrapolationPolicy::Reject),
        scioxide_core::SciOxideErrorKind::DomainError,
    );
    assert_error(
        &interpolator.prepare_antiderivative().unwrap().integral(
            1.0,
            0.0,
            ExtrapolationPolicy::Reject,
        ),
        scioxide_core::SciOxideErrorKind::InvalidInput,
    );
}

fn quadratic() -> BSpline<f64> {
    BSpline::new(
        [BSplineBasis::new(vec![0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0], 2).unwrap()],
        vec![0.0_f64, 0.5, 1.0],
    )
    .unwrap()
}

#[test]
fn derivative_spline_matches_point_derivative() {
    let bspline = quadratic();
    let derivative_spline = bspline.derivative_spline().unwrap();
    let policy = ExtrapolationPolicy::Reject;

    assert_eq!(derivative_spline.bases()[0].degree(), 1);
    assert!(
        (derivative_spline.evaluate([0.5], policy).unwrap()
            - bspline.gradient([0.5], policy).unwrap()[0])
            .abs()
            < 1e-12
    );
}
