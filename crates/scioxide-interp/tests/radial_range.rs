//! Independent radial formulas at extreme scales and public model consumers.
#![allow(clippy::unwrap_used)]

use scioxide_core::{NonFiniteOrigin, RankThreshold, SciOxideErrorView, Tolerance};
use scioxide_interp::{
    Covariance, CovarianceKernel, ExtrapolationPolicy, GaussianProcessModel, Interpolant,
    KrigingMean, LocalModelOptions, LocalModelPlan, LocalRbfModel, LocalRbfOptions, LocalSamples,
    NodeSeparation, RbfInterpolator, RbfKernel, RbfOptions,
};

#[test]
fn range_recovery_in_public_kernels_allocates_nothing() {
    let covariance = Covariance::new(CovarianceKernel::Matern52, 1e300_f64, [1.0]).unwrap();
    {
        assert_eq!(RbfKernel::Gaussian(1.0).evaluate(1e200).unwrap(), 0.0);
        assert!(
            RbfKernel::InverseMultiquadric(1e200)
                .evaluate(1e110)
                .unwrap()
                > 0.0
        );
        assert!(covariance.covariance([0.0], [630.0]).unwrap() > 0.0);
    };
}

macro_rules! reference_cases {
    ($name:ident, $s:ty, $bits:literal) => {
        #[test]
        fn $name() {
            for row in include_str!("data/radial.csv")
                .lines()
                .filter(|r| !r.starts_with('#'))
            {
                let fields: Vec<_> = row.split(',').collect();
                if fields[0] != $bits {
                    continue;
                }
                let [shape, variance, scale, a, b, expected] =
                    core::array::from_fn(|i| fields[i + 2].parse::<$s>().unwrap());
                let actual = match fields[1] {
                    "gaussian" => RbfKernel::Gaussian(shape).evaluate(a),
                    "multiquadric" => RbfKernel::Multiquadric(shape).evaluate(a),
                    "inverse" => RbfKernel::InverseMultiquadric(shape).evaluate(a),
                    "thin_plate" => RbfKernel::ThinPlateSpline.evaluate(a),
                    family => {
                        let kernel = match family {
                            "squared_exponential" => CovarianceKernel::SquaredExponential,
                            "matern12" => CovarianceKernel::Matern12,
                            "matern32" => CovarianceKernel::Matern32,
                            "matern52" => CovarianceKernel::Matern52,
                            _ => unreachable!(),
                        };
                        let covariance = Covariance::new(kernel, variance, [scale]).unwrap();
                        let forward = covariance.covariance([a], [b]);
                        assert_eq!(forward, covariance.covariance([b], [a]));
                        assert_eq!(covariance.covariance([a], [a]).unwrap(), variance);
                        if let Ok(value) = forward {
                            assert!(value >= 0.0 && value <= variance, "{row}: {value:e}");
                        }
                        forward
                    }
                }
                .unwrap();
                assert!(actual.is_finite(), "{row}");
                if expected == 0.0 {
                    assert_eq!(actual, expected, "{row}");
                } else {
                    assert_ne!(actual, 0.0, "representable tail lost: {row}");
                    // Exponent sensitivity includes scalar-rounded Matérn constants.
                    // For subnormals use integer ULP distance; a large absolute
                    // tolerance must never allow zero to replace a nonzero tail.
                    let relative = 32.0
                        * <$s>::EPSILON as f64
                        * (1.0 + (a as f64 - b as f64).abs().min(1500.0));
                    let error = (actual as f64 / expected as f64 - 1.0).abs();
                    let ulps = actual.to_bits().abs_diff(expected.to_bits());
                    assert!(
                        error <= relative || ulps <= 4,
                        "{row}: {actual:e}, {ulps} ULP"
                    );
                }
            }
        }
    };
}
reference_cases!(high_precision_binary32, f32, "32");
reference_cases!(high_precision_binary64, f64, "64");

fn input_error<T>(result: Result<T, scioxide_core::SciOxideError>) {
    assert!(matches!(
        result.err().unwrap().view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
}

#[test]
fn invalid_inputs_and_unrepresentable_growing_kernels_remain_errors() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        input_error(RbfKernel::Gaussian(1.0).evaluate(bad));
        input_error(RbfKernel::InverseMultiquadric(bad).evaluate(1.0));
        input_error(Covariance::new(CovarianceKernel::Matern52, bad, [1.0]));
        input_error(Covariance::new(CovarianceKernel::Matern52, 1.0, [bad]));
        for family in [
            CovarianceKernel::SquaredExponential,
            CovarianceKernel::Matern12,
            CovarianceKernel::Matern32,
            CovarianceKernel::Matern52,
        ] {
            let covariance = Covariance::new(family, 1.0, [1.0]).unwrap();
            input_error(covariance.covariance([bad], [0.0]));
            input_error(covariance.covariance([0.0], [bad]));
        }
    }
    assert!(RbfKernel::Gaussian(0.0).evaluate(1.0).is_err());
    assert!(RbfKernel::Gaussian(1.0).evaluate(-1.0).is_err());
    assert!(
        RbfKernel::Multiquadric(f64::MAX)
            .evaluate(f64::MAX)
            .is_err()
    );
    assert!(RbfKernel::ThinPlateSpline.evaluate(f64::MAX).is_err());
    assert!(
        RbfKernel::Multiquadric(f32::MAX)
            .evaluate(f32::MAX)
            .is_err()
    );
    assert!(RbfKernel::ThinPlateSpline.evaluate(f32::MAX).is_err());
}

#[test]
fn anisotropic_covariance_recovers_before_distance_projection() {
    let covariance = Covariance::new(
        CovarianceKernel::SquaredExponential,
        1.0_f64,
        [1e308, 1e-300],
    )
    .unwrap();
    let value = covariance
        .covariance([1e308, 1e-300], [-1e308, -1e-300])
        .unwrap();
    assert!((value / (-4.0_f64).exp() - 1.0).abs() < 16.0 * f64::EPSILON);
}

macro_rules! consumers {
    ($name:ident, $s:ty, $huge:expr, $tiny:expr) => {
        #[test]
        fn $name() {
            let huge: $s = $huge;
            let tiny: $s = $tiny;
            let points = [[-huge, 0.0], [huge, 0.0]];
            let values = [1.0, 2.0];
            let rank = RankThreshold::new(0.0, 32.0 * <$s>::EPSILON).unwrap();
            for separation in [
                NodeSeparation::Exact,
                NodeSeparation::Tolerant(Tolerance::new(0.0, <$s>::EPSILON).unwrap()),
            ] {
                let global = RbfInterpolator::with_options(
                    points.to_vec(),
                    values.to_vec(),
                    RbfKernel::Gaussian(tiny),
                    RbfOptions::new()
                        .with_node_separation(separation)
                        .with_rank_threshold(rank),
                )
                .unwrap();
                for (p, expected) in points.into_iter().zip(values) {
                    let value = global.evaluate(p, ExtrapolationPolicy::Reject).unwrap();
                    assert!((value - expected).abs() < 256.0 * <$s>::EPSILON);
                }
            }
            // Distances from the selection center fit; pairwise sample
            // distances exceed MAX. Local kernel assembly must recover them.
            let data = LocalSamples::new(&points, &values).unwrap();
            let set = data.points().unwrap();
            let plan = LocalModelPlan::<$s, 2>::new(&set, data.values().unwrap()).unwrap();
            let numerical = LocalModelOptions::new().with_rank_threshold(rank);
            let local = LocalRbfModel::new(
                &plan,
                [0.0, 0.0],
                LocalRbfOptions::new(2, RbfKernel::Gaussian(tiny))
                    .unwrap()
                    .with_numerical(numerical),
            )
            .unwrap();
            let midpoint = local
                .evaluate([0.0, 0.0], ExtrapolationPolicy::Reject)
                .unwrap();
            let z: $s = tiny * huge;
            let expected = 3.0 * (-(z * z)).exp() / (1.0 + (-4.0 * z * z).exp());
            assert!((midpoint[0] - expected).abs() < 256.0 * <$s>::EPSILON);

            for family in [
                CovarianceKernel::SquaredExponential,
                CovarianceKernel::Matern12,
                CovarianceKernel::Matern32,
                CovarianceKernel::Matern52,
            ] {
                let covariance = Covariance::new(family, 1.0, [huge, 1.0]).unwrap();
                let model = GaussianProcessModel::with_options(
                    &set,
                    data.values().unwrap(),
                    covariance,
                    &[0.0; 2],
                    KrigingMean::Constant(&[0.0]),
                    numerical,
                )
                .unwrap();
                let prediction = model.prediction(0.0).unwrap();
                let mut workspace = prediction.workspace().unwrap();
                for (p, expected) in points.into_iter().zip(values) {
                    let mut out = [0.0; 3];
                    prediction
                        .evaluate_into(p, ExtrapolationPolicy::Reject, &mut out, &mut workspace)
                        .unwrap();
                    assert!((out[0] - expected).abs() < 256.0 * <$s>::EPSILON);
                    assert!(out[1].abs() < 256.0 * <$s>::EPSILON);
                }
            }
        }
    };
}
consumers!(models_binary32, f32, 3e38, 1e-38);
consumers!(models_binary64, f64, 1e308, 1e-308);

#[test]
fn inverse_multiquadric_models_preserve_subnormal_predictions() {
    let points = [[0.0_f64], [1e110]];
    let values = [0.0, 1.0];
    let kernel = RbfKernel::InverseMultiquadric(1e200);
    let rank = RankThreshold::new(0.0, 1e-12).unwrap();
    let global = RbfInterpolator::with_options(
        points.map(|[x]| [x, 0.0]).to_vec(),
        values.to_vec(),
        kernel,
        RbfOptions::new().with_rank_threshold(rank),
    )
    .unwrap();
    let data = LocalSamples::new(&points, &values).unwrap();
    let set = data.points().unwrap();
    let plan = LocalModelPlan::<f64, 1>::new(&set, data.values().unwrap()).unwrap();
    let local = LocalRbfModel::new(&plan, [0.0], LocalRbfOptions::new(2, kernel).unwrap()).unwrap();
    let a = global
        .evaluate([2e110, 0.0], ExtrapolationPolicy::MethodDefined)
        .unwrap();
    let b = local
        .evaluate([2e110], ExtrapolationPolicy::MethodDefined)
        .unwrap()[0];
    for actual in [a, b] {
        assert!(actual > 0.0 && actual.is_subnormal());
        assert!(actual.to_bits().abs_diff(1e-310_f64.to_bits()) <= 4);
    }
}
