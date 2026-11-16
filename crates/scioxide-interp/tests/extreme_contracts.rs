//! Cross-family extreme-coordinate and non-finite provenance contracts.

#![allow(clippy::unwrap_used)]

mod support;

use support::assert_error;

use scioxide_core::{NonFiniteOrigin, SciOxideError};
use scioxide_interp::Interpolant;
use scioxide_interp::{
    BSpline, BSplineBasis, BarycentricInterpolator, BicubicInterpolator, BilinearInterpolator,
    BulirschStoerInterpolator, CubicSpline, ExtrapolationPolicy, HermiteInterpolator,
    NevilleInterpolator, NewtonInterpolator, PchipInterpolator, RbfInterpolator, RbfKernel,
    RbfOptions, SplineBoundary,
};

fn assert_finite(value: Result<f64, SciOxideError>) {
    assert!(value.unwrap().is_finite());
}

#[test]
fn finite_extreme_coordinates_work_across_continue_paths() {
    let nodes = vec![-f64::MAX, f64::MAX];
    let values = vec![2.0, 2.0];

    let barycentric = BarycentricInterpolator::new(nodes.clone(), values.clone()).unwrap();
    assert_finite(barycentric.evaluate([0.0], ExtrapolationPolicy::MethodDefined));

    let newton = NewtonInterpolator::new(nodes.clone(), values.clone()).unwrap();
    assert_finite(newton.evaluate([0.0], ExtrapolationPolicy::MethodDefined));

    let hermite = HermiteInterpolator::new(nodes.clone(), values.clone(), vec![0.0, 0.0]).unwrap();
    assert_finite(hermite.evaluate([0.0], ExtrapolationPolicy::MethodDefined));

    let neville = NevilleInterpolator::new(nodes.clone(), values.clone()).unwrap();
    let (mut value, mut error) = (0.0, 0.0);
    neville
        .evaluate_with_error_into(
            [0.0],
            ExtrapolationPolicy::MethodDefined,
            &mut value,
            &mut error,
            &mut neville.workspace().unwrap(),
        )
        .unwrap();
    assert!(value.is_finite() && error.is_finite());

    let bulirsch = BulirschStoerInterpolator::new(nodes.clone(), values.clone()).unwrap();
    assert_finite(bulirsch.evaluate([0.0], ExtrapolationPolicy::MethodDefined));

    let pchip = PchipInterpolator::new(nodes.clone(), values.clone()).unwrap();
    assert_finite(pchip.evaluate([0.0], ExtrapolationPolicy::MethodDefined));

    let bspline = BSpline::new(
        [BSplineBasis::new(vec![-f64::MAX, -f64::MAX, f64::MAX, f64::MAX], 1).unwrap()],
        values,
    )
    .unwrap();
    let mut spline_workspace = bspline.workspace().unwrap();
    let mut spline_value = 0.0;
    bspline
        .evaluate_into(
            [0.0],
            ExtrapolationPolicy::MethodDefined,
            &mut spline_value,
            &mut spline_workspace,
        )
        .unwrap();
    assert!(spline_value.is_finite());

    let cubic = CubicSpline::with_boundary(
        vec![-f64::MAX, 0.0, f64::MAX],
        vec![2.0, 2.0, 2.0],
        SplineBoundary::Natural,
    )
    .unwrap();
    assert_finite(cubic.evaluate([f64::MAX / 2.0], ExtrapolationPolicy::MethodDefined));

    let bilinear = BilinearInterpolator::new(nodes.clone(), nodes.clone(), vec![2.0; 4]).unwrap();
    assert_finite(bilinear.evaluate([0.0, 0.0], ExtrapolationPolicy::MethodDefined));

    let bicubic = BicubicInterpolator::new(nodes.clone(), nodes, vec![2.0; 4]).unwrap();
    assert_finite(bicubic.evaluate([0.0, 0.0], ExtrapolationPolicy::MethodDefined));

    let base = 1.0e200;
    let gap = 1.0e190;
    let points = vec![[base, base], [base + gap, base], [base, base + gap]];
    let rbf = RbfInterpolator::with_options(
        points.clone(),
        vec![1.0, 2.0, 3.0],
        RbfKernel::Gaussian(1.0e-190),
        RbfOptions::new()
            .with_rank_threshold(scioxide_core::RankThreshold::new(0.0, 1.0e-12).unwrap()),
    )
    .unwrap();
    assert_finite(rbf.evaluate(points[1], ExtrapolationPolicy::MethodDefined));
}

#[test]
fn stored_non_finite_query_reports_input_origin() {
    let interpolator = BulirschStoerInterpolator::new(vec![0.0, 1.0], vec![1.0, 0.5]).unwrap();

    assert!(matches!(
        assert_error(
            &interpolator.evaluate([f64::NAN], ExtrapolationPolicy::MethodDefined),
            scioxide_core::SciOxideErrorKind::NonFinite
        )
        .view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
}
#[test]
fn pchip_local_widths_do_not_underflow_against_remote_nodes() {
    macro_rules! run {
        ($s:ty) => {{
            use scioxide_interp::{ExtrapolationPolicy, Interpolant, PchipInterpolator};
            let tiny = <$s>::from_bits(4);
            let nodes = [0.0, tiny, tiny * 2.0, <$s>::MAX];
            let model = PchipInterpolator::new(nodes.to_vec(), nodes.to_vec()).unwrap();
            for slope in model.slopes() {
                assert!((*slope - 1.0).abs() <= 8.0 * <$s>::EPSILON);
            }
            for point in [tiny, tiny * 1.5, 1.0] {
                let value = model
                    .evaluate([point], ExtrapolationPolicy::Reject)
                    .unwrap();
                assert!(
                    (value - point).abs() <= 8.0 * <$s>::EPSILON * point.abs() + <$s>::from_bits(1),
                    "point={point:?}, value={value:?}"
                );
            }
        }};
    }
    run!(f32);
    run!(f64);
}
