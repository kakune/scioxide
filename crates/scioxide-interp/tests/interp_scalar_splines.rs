//! Spline coverage with supported ordered real coordinates.

#![allow(clippy::unwrap_used)]

use scioxide_interp::{CubicSpline, ExtrapolationPolicy, Interpolant, SplineBoundary};
fn values(values: &[f64]) -> Vec<f64> {
    values.to_vec()
}
fn assert_nodes(spline: &CubicSpline<f64>, x: &[f64], y: &[f64]) {
    x.iter().zip(y).for_each(|(&node, &expected)| {
        let actual = spline
            .evaluate([node], ExtrapolationPolicy::Reject)
            .unwrap();
        assert!((actual - expected).abs() <= 1.0e-12);
    });
}

#[test]
fn real_coordinate_supports_all_nonperiodic_spline_boundaries() {
    let x = [0.0, 1.0, 2.0, 3.0];
    let y = [0.0, 1.0, 4.0, 9.0];
    for boundary in [
        SplineBoundary::Natural,
        SplineBoundary::Clamped(0.0, 6.0),
        SplineBoundary::NotAKnot,
    ] {
        let spline = CubicSpline::with_boundary(values(&x), values(&y), boundary).unwrap();
        assert_nodes(&spline, &x, &y);
    }
}

#[test]
fn real_coordinate_supports_the_shared_cyclic_solver() {
    let x = [0.0, 1.0, 2.0, 3.0, 4.0];
    let y = [0.0, 1.0, 0.0, -1.0, 0.0];
    let spline =
        CubicSpline::with_boundary(values(&x), values(&y), SplineBoundary::Periodic).unwrap();
    assert_nodes(&spline, &x, &y);
}

#[test]
fn ten_thousand_knot_splines_accept_decaying_auxiliary_solutions() {
    let n = 10_000;
    let x: Vec<_> = (0..n)
        .map(|i| core::f64::consts::TAU * i as f64 / (n - 1) as f64)
        .collect();
    let mut periodic: Vec<_> = x.iter().map(|x| x.sin()).collect();
    periodic[n - 1] = periodic[0];
    let mut localized = vec![0.0; n];
    localized[n / 2] = 1.0;
    for (y, boundary) in [
        (periodic, SplineBoundary::Periodic),
        (localized, SplineBoundary::Natural),
    ] {
        let spline = CubicSpline::with_boundary(x.clone(), y.clone(), boundary).unwrap();
        assert_nodes(&spline, &x, &y);
        for i in [0, n / 2, n - 2] {
            assert!(
                spline
                    .evaluate([(x[i] + x[i + 1]) / 2.0], ExtrapolationPolicy::Reject)
                    .unwrap()
                    .is_finite()
            );
        }
    }
}
