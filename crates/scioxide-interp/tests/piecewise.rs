//! Independent rational slope fixtures and public piecewise contracts.
#![allow(clippy::unwrap_used)]
mod support;

use support::assert_error;

use scioxide_core::RealScalar;
use scioxide_interp::{
    AkimaInterpolator, AkimaMethod, Differentiable, ExtrapolationPolicy, Integrable, Interpolant,
    PchipInterpolator,
};
const P: ExtrapolationPolicy<f64> = ExtrapolationPolicy::Reject;
fn slopes<S: RealScalar>() {
    let cv = |v: f64| S::try_from_f64_lossy(v).unwrap();
    let x = [0., 1., 3., 4., 7.].map(cv).to_vec();
    let y = [0., 2., 3., 3., 8.].map(cv).to_vec();
    // Exact rational evaluation of the published weights; no comparator library.
    for (method, expected) in [
        (
            AkimaMethod::Akima,
            [11. / 4., 7. / 8., 5. / 19., 5. / 13., 5. / 2.],
        ),
        (
            AkimaMethod::Makima,
            [169. / 68., 29. / 40., 5. / 21., 15. / 59., 13. / 6.],
        ),
    ] {
        let plan = AkimaInterpolator::with_method(x.clone(), y.clone(), method).unwrap();
        for ((&node, &value), (&slope, expected)) in
            x.iter().zip(&y).zip(plan.slopes().iter().zip(expected))
        {
            assert_eq!(
                plan.evaluate([node], ExtrapolationPolicy::Reject).unwrap(),
                value
            );
            assert!((slope - cv(expected)).abs() < cv(32.) * S::EPSILON);
            assert!(
                (plan.gradient([node], ExtrapolationPolicy::Reject).unwrap()[0] - slope).abs()
                    < cv(32.) * S::EPSILON
            );
        }
    }
}
#[test]
fn rational_slopes_both_precisions() {
    slopes::<f64>();
    slopes::<f32>();
}
#[test]
fn lines_plateaus_and_small_scales() {
    for scale in [1e-200, 1., 1e200] {
        for method in [AkimaMethod::Akima, AkimaMethod::Makima] {
            for x in [vec![0., 3.], vec![0., 1., 3., 4., 7.]] {
                let y = x.iter().map(|v| scale * v).collect();
                let plan = AkimaInterpolator::with_method(x, y, method).unwrap();
                assert!((plan.evaluate([2.], P).unwrap() / scale - 2.).abs() < 1e-14);
                assert!((plan.gradient([2.], P).unwrap()[0] / scale - 1.).abs() < 1e-14);
                assert!((plan.integral([0.], [3.], P).unwrap() / scale - 4.5).abs() < 1e-14);
            }
        }
        let plan = PchipInterpolator::new(
            vec![0., 1., 3., 4.],
            vec![0., scale, 3. * scale, 4. * scale],
        )
        .unwrap();
        assert!(plan.slopes().iter().all(|v| (v / scale - 1.).abs() < 1e-14));
    }
    let x = vec![0., 1., 2., 3., 4., 5.];
    let y = vec![0., 0., 0., 1., 1., 1.];
    let plan = AkimaInterpolator::with_method(x, y, AkimaMethod::Makima).unwrap();
    for i in 0..=500 {
        let v = plan.evaluate([f64::from(i) / 100.], P).unwrap();
        assert!((-1e-14..=1. + 1e-14).contains(&v));
    }
}
#[test]
fn extrapolation_and_integrals() {
    let plan = AkimaInterpolator::new(vec![0., 1.], vec![1.0_f64, 3.]).unwrap();
    assert_error(
        &plan.evaluate([-1.], P),
        scioxide_core::SciOxideErrorKind::DomainError,
    );
    assert_eq!(
        plan.evaluate([-1.], ExtrapolationPolicy::Clamp).unwrap(),
        1.
    );
    assert_eq!(
        plan.gradient([-1.], ExtrapolationPolicy::Clamp).unwrap()[0],
        0.
    );
    assert_eq!(
        plan.integral([-1.], [2.], ExtrapolationPolicy::Clamp)
            .unwrap(),
        6.
    );
    assert!(
        (plan
            .integral([-1.], [2.], ExtrapolationPolicy::MethodDefined)
            .unwrap()
            - 6.)
            .abs()
            < 1e-13
    );
    assert_eq!(
        plan.evaluate(
            [-1.],
            ExtrapolationPolicy::BoundedContinue { max_distance: 1. }
        )
        .unwrap(),
        -1.
    );
    assert!(
        plan.evaluate(
            [-1.1],
            ExtrapolationPolicy::BoundedContinue { max_distance: 1. }
        )
        .is_err()
    );
    assert!(plan.integral([1.], [0.], P).is_err());
    assert!(
        plan.gradient([f64::NAN], ExtrapolationPolicy::Clamp)
            .is_err()
    );
}
#[test]
fn pchip_shape_and_endpoint_formula() {
    // h=(1,2), delta=(2,1/2): endpoint=(4*2-1/2)/3=5/2.
    let plan = PchipInterpolator::new(vec![0., 1., 3.], vec![0., 2., 3.]).unwrap();
    assert!((plan.slopes()[0] - 2.5_f64).abs() < 1e-14);
    assert!((plan.slopes()[1] - 6.0_f64 / 7.).abs() < 1e-14);
    for sign in [-1., 1.] {
        let x = vec![0., 0.125, 1., 3., 3.25, 8.];
        let y = [0., 0., 0.1, 0.2, 10., 10.].map(|v| v * sign).to_vec();
        let plan = PchipInterpolator::new(x.clone(), y.clone()).unwrap();
        for (xs, ys) in x.windows(2).zip(y.windows(2)) {
            for k in 0..=100 {
                let t = xs[0] + (xs[1] - xs[0]) * f64::from(k) / 100.;
                let v = plan.evaluate([t], P).unwrap();
                assert!(v >= ys[0].min(ys[1]) - 1e-13 && v <= ys[0].max(ys[1]) + 1e-13);
                assert!(sign * plan.gradient([t], P).unwrap()[0] >= -1e-12);
            }
        }
    }
}
#[test]
fn invalid_samples() {
    for (x, y) in [
        (vec![], vec![]),
        (vec![0.], vec![0.]),
        (vec![0., 0.], vec![0., 1.]),
        (vec![0., 1.], vec![0.]),
        (vec![0., 1.], vec![0., f64::NAN]),
        (vec![0., f64::MIN_POSITIVE], vec![-f64::MAX, f64::MAX]),
    ] {
        assert!(AkimaInterpolator::with_method(x, y, AkimaMethod::Makima).is_err());
    }
}

#[test]
fn smooth_data_accuracy_and_derivative_order() {
    fn error(n: u32, method: Option<AkimaMethod>) -> (f64, f64) {
        let x = (0..=n)
            .map(|i| f64::from(i) / f64::from(n))
            .collect::<Vec<_>>();
        let y: Vec<f64> = x.iter().map(|v| v * v * v + v).collect();
        fn measure<I: Differentiable<f64, 1, Value = f64>>(plan: &I, x: &[f64]) -> (f64, f64) {
            x.windows(2).fold((0., 0.), |(ev, ed), w| {
                let t = w[0] + 0.375 * (w[1] - w[0]);
                (
                    ev.max((plan.evaluate([t], P).unwrap() - (t * t * t + t)).abs()),
                    ed.max((plan.gradient([t], P).unwrap()[0] - (3. * t * t + 1.)).abs()),
                )
            })
        }
        match method {
            Some(method) => measure(
                &AkimaInterpolator::with_method(x.clone(), y, method).unwrap(),
                &x,
            ),
            None => measure(&PchipInterpolator::new(x.clone(), y).unwrap(), &x),
        }
    }
    for method in [None, Some(AkimaMethod::Akima), Some(AkimaMethod::Makima)] {
        let coarse = error(16, method);
        let fine = error(32, method);
        assert!(
            fine.0 < coarse.0 / 5.,
            "value order {method:?}: {coarse:?} -> {fine:?}"
        );
        assert!(
            fine.1 < coarse.1 / 3.,
            "derivative order {method:?}: {coarse:?} -> {fine:?}"
        );
    }
}
#[test]
fn akima_zero_weights_average_distinct_adjacent_secants() {
    let p = AkimaInterpolator::new(vec![0., 1., 2., 3., 4.], vec![0., 1., 2., 5., 8.]).unwrap();
    assert_eq!(p.slopes()[2], 2.);
}

#[test]
fn remote_large_secants_do_not_erase_small_adjacent_slopes() {
    for method in [AkimaMethod::Akima, AkimaMethod::Makima] {
        let plan = AkimaInterpolator::with_method(
            vec![0., 1e-200, 1e200, 2e200, 3e200],
            vec![0., 1., 2., 3., 1e300],
            method,
        )
        .unwrap();
        assert!((plan.slopes()[2] / 1e-200_f64 - 1.).abs() < 1e-14);
    }
}
