//! Focused constructor and query error contracts at both scalar widths.
#![allow(clippy::unwrap_used)]

mod support;

use scioxide_core::{
    InvalidInputReason, NdView, RealScalar, SciOxideError, SciOxideErrorKind as K,
    SciOxideErrorView, StridedLayout,
};
use scioxide_interp::*;
use scioxide_spatial::{Delaunay, PointSet};
use support::{assert_error, from_f64};

fn curve_inputs<S: RealScalar>() {
    for family in ["cubic", "pchip", "newton", "hermite", "neville", "bulirsch"] {
        let construct = |nodes: Vec<S>, values: Vec<S>| -> Result<(), SciOxideError> {
            let slopes = vec![S::ONE; nodes.len()];
            match family {
                "cubic" => {
                    CubicSpline::with_boundary(nodes, values, SplineBoundary::Natural).map(drop)
                }
                "pchip" => PchipInterpolator::new(nodes, values).map(drop),
                "newton" => NewtonInterpolator::new(nodes, values).map(drop),
                "hermite" => HermiteInterpolator::new(nodes, values, slopes).map(drop),
                "neville" => NevilleInterpolator::new(nodes, values).map(drop),
                _ => BulirschStoerInterpolator::new(nodes, values).map(drop),
            }
        };
        assert_error(&construct(vec![], vec![]), K::InvalidInput);
        for nodes in [vec![S::ONE, S::ZERO], vec![S::ZERO, S::ZERO]] {
            let result = construct(nodes, vec![S::ZERO, S::ONE]);
            assert!(matches!(
                assert_error(&result, K::InvalidInput).view(),
                SciOxideErrorView::InvalidInput {
                    reason: InvalidInputReason::Inconsistent,
                    ..
                }
            ));
        }
        assert_error(
            &construct(vec![S::ZERO, S::ONE], vec![S::ZERO]),
            K::DimensionMismatch,
        );
        for bad in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
            assert_error(
                &construct(vec![S::ZERO, bad], vec![S::ZERO, S::ONE]),
                K::NonFinite,
            );
            assert_error(
                &construct(vec![S::ZERO, S::ONE], vec![S::ZERO, bad]),
                K::NonFinite,
            );
        }
    }
    assert_error(
        &HermiteInterpolator::new(vec![S::ZERO, S::ONE], vec![S::ZERO, S::ONE], vec![S::ONE]),
        K::DimensionMismatch,
    );
    assert_error(
        &HermiteInterpolator::new(
            vec![S::ZERO, S::ONE],
            vec![S::ZERO, S::ONE],
            vec![S::ONE, S::NAN],
        ),
        K::NonFinite,
    );
    for degree in [2, 3, usize::MAX] {
        assert_error(
            &FloaterHormannInterpolator::new(vec![S::ZERO, S::ONE], vec![S::ZERO, S::ONE], degree),
            K::InvalidInput,
        );
    }
    for degree in 0..=3 {
        assert_error(&cardinal_bspline::<S>(degree, S::NAN), K::NonFinite);
    }
}

fn singleton_curves<S: RealScalar>() {
    let nodes = vec![S::ZERO];
    let values = vec![from_f64::<S>(3.)];
    let newton = NewtonInterpolator::new(nodes.clone(), values.clone()).unwrap();
    let bulirsch = BulirschStoerInterpolator::new(nodes.clone(), values.clone()).unwrap();
    let neville = NevilleInterpolator::new(nodes.clone(), values.clone()).unwrap();
    for query in [S::ZERO, S::ONE, -S::ONE] {
        assert_eq!(
            newton
                .evaluate([query], ExtrapolationPolicy::MethodDefined)
                .unwrap(),
            values[0]
        );
        assert_eq!(
            bulirsch
                .evaluate([query], ExtrapolationPolicy::MethodDefined)
                .unwrap(),
            values[0]
        );
        let (mut value, mut error) = (S::ZERO, S::ONE);
        neville
            .evaluate_with_error_into(
                [query],
                ExtrapolationPolicy::MethodDefined,
                &mut value,
                &mut error,
                &mut neville.workspace().unwrap(),
            )
            .unwrap();
        assert_eq!((value, error), (values[0], S::ZERO));
    }
    assert_error(
        &CubicSpline::with_boundary(nodes, values, SplineBoundary::Natural),
        K::InvalidInput,
    );
}

fn grid_inputs<S: RealScalar>() {
    let x = vec![S::ZERO, S::ONE, from_f64(2.)];
    for axis in [
        vec![S::ZERO, S::ONE, S::NAN],
        vec![S::ZERO, from_f64(2.), S::ONE],
        vec![S::ZERO, S::ONE, S::ONE],
    ] {
        let kind = if axis[2].is_nan() {
            K::NonFinite
        } else {
            K::InvalidInput
        };
        assert_error(
            &BicubicInterpolator::new(axis.clone(), x.clone(), vec![S::ONE; 9]),
            kind,
        );
        assert_error(
            &BicubicInterpolator::new(x.clone(), axis, vec![S::ONE; 9]),
            kind,
        );
    }
    assert_error(
        &BicubicInterpolator::new(x.clone(), x.clone(), vec![S::ONE; 8]),
        K::DimensionMismatch,
    );
    let mut values = vec![S::ONE; 9];
    values[4] = S::NAN;
    assert_error(
        &BicubicInterpolator::new(x.clone(), x, values),
        K::NonFinite,
    );
}

fn spline_inputs<S: RealScalar>() {
    for (knots, degree) in [
        (vec![], 1),
        (vec![S::ZERO, S::ONE], 1),
        (vec![S::ZERO, S::ZERO, S::ONE, S::ZERO], 1),
        (vec![S::ZERO; 4], 1),
        (vec![S::ZERO, S::ZERO, S::ZERO, S::ONE, S::ONE], 1),
    ] {
        assert_error(&BSplineBasis::new(knots, degree), K::InvalidInput);
    }
    assert_error(
        &BSplineBasis::new(vec![S::ZERO, S::ZERO, S::ONE, S::NAN], 1),
        K::NonFinite,
    );
    assert_error(&BSplineBasis::<S>::new(vec![], usize::MAX), K::SizeOverflow);
    let basis = BSplineBasis::new(vec![S::ZERO, S::ZERO, S::ONE, S::ONE], 1).unwrap();
    assert_error(
        &BSpline::new([basis.clone(), basis.clone()], vec![S::ZERO; 3]),
        K::DimensionMismatch,
    );
    assert_error(&BSpline::<S, S, 0>::new([], vec![S::ZERO]), K::InvalidInput);
    assert_error(
        &BSpline::new([basis.clone()], vec![S::ZERO, S::NAN]),
        K::NonFinite,
    );
    for bad in [-S::ONE, S::ZERO] {
        let result = Nurbs::new([basis.clone()], vec![S::ZERO, S::ONE], vec![S::ONE, bad]);
        assert!(matches!(
            assert_error(&result, K::InvalidInput).view(),
            SciOxideErrorView::InvalidInput {
                reason: InvalidInputReason::OutOfRange,
                ..
            }
        ));
    }
    for bad in [S::NAN, S::INFINITY] {
        assert_error(
            &Nurbs::new([basis.clone()], vec![S::ZERO, S::ONE], vec![S::ONE, bad]),
            K::NonFinite,
        );
    }
    assert_error(
        &Nurbs::new([basis.clone()], vec![S::ZERO, S::ONE], vec![S::ONE]),
        K::DimensionMismatch,
    );
    assert_error(
        &TensorSplineInterpolationPlan::new([basis.clone()], [vec![S::ZERO]]),
        K::DimensionMismatch,
    );
    assert_error(
        &TensorSplineInterpolationPlan::new([basis.clone()], [vec![S::ZERO, S::ZERO]]),
        K::InvalidInput,
    );
    assert_error(
        &TensorSplineInterpolationPlan::new([basis.clone()], [vec![-S::ONE, S::ONE]]),
        K::DomainError,
    );
    assert_error(
        &TensorSplineInterpolationPlan::new([basis.clone()], [vec![S::ZERO, S::NAN]]),
        K::NonFinite,
    );
    assert_error(
        &TensorSplineInterpolationPlan::<S, 0>::new([], []),
        K::InvalidInput,
    );
    let split = BSplineBasis::new(
        vec![
            S::ZERO,
            S::ZERO,
            from_f64(0.5),
            from_f64(0.5),
            S::ONE,
            S::ONE,
        ],
        1,
    )
    .unwrap();
    assert_error(
        &TensorSplineInterpolationPlan::new(
            [split],
            [vec![
                S::ZERO,
                from_f64(0.125),
                from_f64(0.25),
                from_f64(0.375),
            ]],
        ),
        K::Singular,
    );
    assert!(TensorSplineInterpolationPlan::new([basis], [vec![S::ONE, S::ZERO]]).is_ok());
}

fn local_inputs<S: RealScalar>() {
    for (variance, kind) in [
        (S::ZERO, K::InvalidInput),
        (-S::ONE, K::InvalidInput),
        (S::NAN, K::NonFinite),
        (S::INFINITY, K::NonFinite),
    ] {
        assert_error(
            &Covariance::new(CovarianceKernel::Matern32, variance, [S::ONE]),
            kind,
        );
        assert_error(
            &Covariance::new(CovarianceKernel::SquaredExponential, S::ONE, [variance]),
            kind,
        );
    }
    assert_error(
        &Covariance::<S, 0>::new(CovarianceKernel::Matern12, S::ONE, []),
        K::InvalidInput,
    );
    assert_error(
        &LocalSamples::<S, 2>::new(&[], &[] as &[S]),
        K::InvalidInput,
    );
    assert_error(
        &LocalSamples::<S, 0>::new(&[[]], &[S::ONE]),
        K::InvalidInput,
    );
    assert_error(
        &LocalSamples::new(&[[S::ZERO]], &[] as &[S]),
        K::DimensionMismatch,
    );
    assert_error(&LocalSamples::new(&[[S::NAN]], &[S::ONE]), K::NonFinite);
    assert_error(&LocalSamples::new(&[[S::ZERO]], &[S::NAN]), K::NonFinite);
    assert_error(
        &LocalSamples::new(
            &[[S::ZERO], [S::ONE]],
            &[vec![S::ONE], vec![S::ONE, S::ONE]],
        ),
        K::DimensionMismatch,
    );
}

fn scattered_inputs<S: RealScalar>() {
    let coordinates = [
        S::ZERO,
        S::ZERO,
        S::ONE,
        S::ZERO,
        S::ZERO,
        S::ONE,
        S::ONE,
        S::ONE,
    ];
    let shape = [4, 2];
    let strides = [2, 1];
    let points = PointSet::new(
        NdView::new(&coordinates, StridedLayout::new(&shape, &strides).unwrap()).unwrap(),
    )
    .unwrap();
    let mesh = Delaunay::new(&points).unwrap();
    for bad in [S::NAN, S::INFINITY] {
        assert_error(
            &ScatteredValues::from_input_values(&mesh, vec![S::ZERO, S::ONE, bad, S::ONE]),
            K::NonFinite,
        );
    }
    let samples =
        ScatteredValues::from_input_values(&mesh, vec![S::ZERO, S::ONE, S::ONE, from_f64(2.)])
            .unwrap();
    assert_error(
        &CloughTocherInterpolator::with_options(
            &mesh,
            &samples,
            CloughTocherOptions::new().with_maximum_iterations(0),
        ),
        K::InvalidInput,
    );
    assert_error(
        &CloughTocherInterpolator::with_options(
            &mesh,
            &samples,
            CloughTocherOptions::new().with_maximum_preparation_bytes(0),
        ),
        K::ResourceLimit,
    );
    let nearest = NearestNeighborInterpolator::<_, 2>::new(&mesh, &samples).unwrap();
    let linear = SimplexLinearInterpolator::<_, 2>::new(&mesh, &samples).unwrap();
    let natural = NaturalNeighborInterpolator::new(&mesh, &samples).unwrap();
    let clough =
        CloughTocherInterpolator::with_options(&mesh, &samples, CloughTocherOptions::new())
            .unwrap();
    for query in [[S::NAN, S::ZERO], [S::ZERO, S::NAN], [S::INFINITY, S::ZERO]] {
        let mut out = from_f64::<S>(17.);
        assert_error(
            &nearest.evaluate_into(
                query,
                ExtrapolationPolicy::Reject,
                &mut out,
                &mut nearest.workspace().unwrap(),
            ),
            K::NonFinite,
        );
        assert_eq!(out, from_f64(17.));
        assert_error(
            &linear.evaluate_into(
                query,
                ExtrapolationPolicy::Reject,
                &mut out,
                &mut linear.workspace().unwrap(),
            ),
            K::NonFinite,
        );
        assert_eq!(out, from_f64(17.));
        assert_error(
            &natural.evaluate_into(
                query,
                ExtrapolationPolicy::Reject,
                &mut out,
                &mut natural.workspace().unwrap(),
            ),
            K::NonFinite,
        );
        assert_eq!(out, from_f64(17.));
        assert_error(
            &clough.evaluate_into(
                query,
                ExtrapolationPolicy::Reject,
                &mut out,
                &mut clough.workspace().unwrap(),
            ),
            K::NonFinite,
        );
        assert_eq!(out, from_f64(17.));
    }
}

macro_rules! width_tests {
    ($module:ident,$scalar:ty) => {
        mod $module {
            use super::*;
            #[test]
            fn curve_errors() {
                curve_inputs::<$scalar>();
            }
            #[test]
            fn singleton_behavior() {
                singleton_curves::<$scalar>();
            }
            #[test]
            fn grid_errors() {
                grid_inputs::<$scalar>();
            }
            #[test]
            fn spline_errors() {
                spline_inputs::<$scalar>();
            }
            #[test]
            fn local_errors() {
                local_inputs::<$scalar>();
            }
            #[test]
            fn scattered_errors() {
                scattered_inputs::<$scalar>();
            }
        }
    };
}
width_tests!(f32, f32);
width_tests!(f64, f64);
