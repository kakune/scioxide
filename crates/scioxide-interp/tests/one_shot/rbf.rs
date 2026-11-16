//! `RbfInterpolator::new` against `with_options`, and the validation of `build`.

use scioxide_core::{
    ExecutionContext, NonFiniteOrigin, RankThreshold, SciOxideError, SciOxideErrorKind,
    SciOxideErrorView, Tolerance,
};
use scioxide_interp::{
    ExtrapolationPolicy, Interpolant, RbfInterpolator, RbfKernel, RbfOptions, RbfRegularization,
};

/// The argument named by an `InvalidInput` error.
fn blamed(error: &SciOxideError) -> &'static str {
    match error.view() {
        SciOxideErrorView::InvalidInput { argument, .. } => argument.as_str(),
        _ => panic!("expected InvalidInput, got {error:?}"),
    }
}

macro_rules! widths {
    ($module:ident, $s:ty) => {
        mod $module {
            use super::*;
            type S = $s;

            fn points() -> Vec<[S; 2]> {
                vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.25]]
            }

            fn values() -> Vec<S> {
                vec![1.0, 2.0, 3.0, 4.0, 2.0]
            }

            #[test]
            fn new_equals_with_options_with_default_options() {
                let defaults = RbfOptions::<S>::default();
                assert_eq!(defaults, RbfOptions::new());
                assert_eq!(defaults.regularization(), RbfRegularization::None);
                assert_eq!(
                    defaults.rank_threshold(),
                    RankThreshold::from_tolerance(Tolerance::from_machine_epsilon())
                );
                assert_eq!(defaults.minimum_rcond(), 0.0);
                assert_eq!(defaults.maximum_bytes(), usize::MAX);
                for kernel in [
                    RbfKernel::Gaussian(1.0),
                    RbfKernel::Multiquadric(1.0),
                    RbfKernel::InverseMultiquadric(1.0),
                    RbfKernel::ThinPlateSpline,
                ] {
                    let one_shot = RbfInterpolator::new(points(), values(), kernel).unwrap();
                    let configured =
                        RbfInterpolator::with_options(points(), values(), kernel, defaults)
                            .unwrap();
                    let scoped = RbfInterpolator::build(
                        points(),
                        values(),
                        kernel,
                        defaults,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap();
                    assert_eq!(one_shot.conditioning(), configured.conditioning());
                    assert_eq!(one_shot.conditioning(), scoped.conditioning());
                    for query in [[0.1, 0.2], [0.5, 0.5], [0.9, 0.3], [0.25, 0.75], [1.0, 1.0]] {
                        let value = one_shot
                            .evaluate(query, ExtrapolationPolicy::Reject)
                            .unwrap();
                        assert_eq!(
                            value.to_bits(),
                            configured
                                .evaluate(query, ExtrapolationPolicy::Reject)
                                .unwrap()
                                .to_bits()
                        );
                    }
                }
            }

            #[test]
            fn build_validates_kernel_then_options_before_the_data() {
                let options = RbfOptions::<S>::new();
                let build = |kernel: RbfKernel<S>, options: RbfOptions<S>, points: Vec<[S; 2]>| {
                    RbfInterpolator::build(
                        points,
                        values(),
                        kernel,
                        options,
                        ExecutionContext::sequential_native(),
                    )
                    .unwrap_err()
                };
                let gaussian = RbfKernel::Gaussian(1.0);
                // Each check runs before the data are examined: no points at all.
                for shape in [0.0, -1.0] {
                    let error = build(RbfKernel::Gaussian(shape), options, Vec::new());
                    assert_eq!(blamed(&error), "kernel");
                }
                assert!(matches!(
                    build(RbfKernel::Multiquadric(S::NAN), options, Vec::new()).view(),
                    SciOxideErrorView::NonFinite {
                        origin: NonFiniteOrigin::Input,
                        ..
                    }
                ));
                for lambda in [0.0, -1.0] {
                    let bad = options.with_regularization(RbfRegularization::Diagonal(lambda));
                    assert_eq!(
                        blamed(&build(gaussian, bad, Vec::new())),
                        "options.regularization"
                    );
                    // The kernel is checked first.
                    let both = build(RbfKernel::Gaussian(0.0), bad, Vec::new());
                    assert_eq!(blamed(&both), "kernel");
                }
                let infinite =
                    options.with_regularization(RbfRegularization::Diagonal(S::INFINITY));
                assert_eq!(
                    build(gaussian, infinite, Vec::new()).kind(),
                    SciOxideErrorKind::NonFinite
                );
                for rcond in [S::NAN, -0.25, 1.5, S::INFINITY] {
                    let bad = options.with_minimum_rcond(rcond);
                    assert_eq!(
                        blamed(&build(gaussian, bad, Vec::new())),
                        "options.minimum_rcond"
                    );
                    let both = options
                        .with_regularization(RbfRegularization::Diagonal(0.0))
                        .with_minimum_rcond(rcond);
                    assert_eq!(
                        blamed(&build(gaussian, both, Vec::new())),
                        "options.regularization"
                    );
                }
                // Valid options reach the data checks.
                assert_eq!(blamed(&build(gaussian, options, Vec::new())), "points");
                let boundary = options.with_minimum_rcond(1.0);
                assert_eq!(
                    build(gaussian, boundary, points()).kind(),
                    SciOxideErrorKind::Singular
                );
            }
        }
    };
}

widths!(single, f32);
widths!(double, f64);

/// The fractional part of `x ≥ 0`.
fn fractional(x: f64) -> f64 {
    x - x.floor()
}

/// 30 points of an additive recurrence in the unit square and its four corners.
fn sites() -> Vec<[f64; 2]> {
    (1..=30)
        .map(|k| {
            let k = f64::from(k);
            [
                fractional(0.5 + 0.754_877_666_2 * k),
                fractional(0.5 + 0.569_840_291_0 * k),
            ]
        })
        .chain([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        .collect()
}

#[test]
fn default_rank_threshold_accepts_a_usable_multiquadric_system() {
    let points = sites();
    let values: Vec<f64> = points
        .iter()
        .map(|&[x, y]| (3.0 * x).sin() * (2.0 * y).cos())
        .collect();
    let kernel = RbfKernel::Multiquadric(1.0);
    let rbf = RbfInterpolator::new(points.clone(), values.clone(), kernel).unwrap();
    for (&site, &expected) in points.iter().zip(&values) {
        let value = rbf.evaluate(site, ExtrapolationPolicy::Reject).unwrap();
        assert!(
            (value - expected).abs() <= 1e-8,
            "{site:?}: {value} vs {expected}"
        );
    }
    let square_root = RbfOptions::new().with_rank_threshold(RankThreshold::default());
    let rejected = RbfInterpolator::with_options(points, values, kernel, square_root);
    assert_eq!(rejected.unwrap_err().kind(), SciOxideErrorKind::Singular);
}
