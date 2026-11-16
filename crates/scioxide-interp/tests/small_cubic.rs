//! Independent small-node spline references and periodic seam contracts.
#![allow(clippy::unwrap_used)]
mod support;

use crate::support::from_f64;
use scioxide_core::RealScalar;
use scioxide_interp::{
    CubicSpline, Differentiable, ExtrapolationPolicy as P, Integrable, Interpolant, SplineBoundary,
};
fn references<S: RealScalar>() {
    for row in include_str!("data/small_cubic.csv")
        .lines()
        .filter(|r| !r.starts_with('#'))
    {
        let cells: Vec<_> = row.split(',').collect();
        let a: Vec<f64> = cells[1..].iter().map(|x| x.parse().unwrap()).collect();
        let n = a[0] as usize;
        let boundary = if cells[0] == "periodic" {
            SplineBoundary::Periodic
        } else {
            SplineBoundary::NotAKnot
        };
        let spline = CubicSpline::with_boundary(
            a[1..1 + n].iter().copied().map(from_f64::<S>).collect(),
            a[4..4 + n].iter().copied().map(from_f64::<S>).collect(),
            boundary,
        )
        .unwrap();
        let actual = [
            spline.evaluate([from_f64(a[7])], P::Reject).unwrap(),
            spline.gradient([from_f64(a[7])], P::Reject).unwrap()[0],
            spline
                .integral([from_f64(a[8])], [from_f64(a[9])], P::Reject)
                .unwrap(),
        ];
        for (actual, expected) in actual.into_iter().zip(&a[10..]) {
            let tolerance = 128. * S::EPSILON.try_to_f64_exact().unwrap() * expected.abs().max(1.);
            assert!(
                (actual.try_to_f64_exact().unwrap() - expected).abs() <= tolerance,
                "{row}: {actual:?} != {expected}"
            );
        }
        if cells[0] == "periodic" {
            let left = spline.gradient([from_f64(a[1])], P::Reject).unwrap()[0];
            let right = spline.gradient([from_f64(a[n])], P::Reject).unwrap()[0];
            assert!(
                (left - right).abs() <= from_f64::<S>(128.) * S::EPSILON * left.abs().max(S::ONE)
            );
        }
    }
}
#[test]
fn scipy_small_cubic_f64() {
    references::<f64>();
}
#[test]
fn scipy_small_cubic_f32() {
    references::<f32>();
}

fn normalization<S: RealScalar>() {
    for n in [2, 3, 5] {
        let x: Vec<S> = (0..n).map(|i| from_f64(i as f64)).collect();
        let mut values = vec![[S::ONE, S::TWO]; n];
        values[n - 1][0] = S::ONE + S::SQRT_EPSILON / from_f64(2.);
        let model = CubicSpline::with_boundary(x.clone(), values.clone(), SplineBoundary::Periodic)
            .unwrap();
        assert_eq!(model.evaluate([x[n - 1]], P::Reject).unwrap(), values[0]);
        for t in [0., 0.25, 0.5, 0.75, 1.] {
            let result = model
                .evaluate([from_f64((n - 1) as f64 * t)], P::Reject)
                .unwrap();
            assert_eq!(result, [S::ONE, S::TWO]);
        }
        values[n - 1][1] = S::TWO + from_f64::<S>(4.) * S::SQRT_EPSILON;
        assert!(CubicSpline::with_boundary(x, values, SplineBoundary::Periodic).is_err());
    }
    for boundary in [
        SplineBoundary::Natural,
        SplineBoundary::NotAKnot,
        SplineBoundary::Periodic,
        SplineBoundary::Clamped(S::ZERO, S::ZERO),
    ] {
        assert!(CubicSpline::with_boundary(vec![S::ZERO], vec![S::ONE], boundary).is_err());
    }
}
#[test]
fn normalized_periodic_ends_f64() {
    normalization::<f64>();
}
#[test]
fn normalized_periodic_ends_f32() {
    normalization::<f32>();
}

#[test]
fn periodic_sine_error_follows_fourth_order_bound() {
    // For uniform cubic periodic interpolation, ||f-s||∞ <= 5 h⁴ ||f''''||∞/384.
    // Here f=sin and ||f''''||∞=1. Samples include midpieces, not only nodes.
    for intervals in [8, 16, 32] {
        let h = core::f64::consts::TAU / intervals as f64;
        let x: Vec<_> = (0..=intervals).map(|i| i as f64 * h).collect();
        let mut y: Vec<_> = x.iter().map(|x| x.sin()).collect();
        y[intervals] = y[0];
        let model = CubicSpline::with_boundary(x, y, SplineBoundary::Periodic).unwrap();
        let bound = 5. * h.powi(4) / 384. + 128. * f64::EPSILON;
        for i in 0..=16 * intervals {
            let q = core::f64::consts::TAU * i as f64 / (16 * intervals) as f64;
            assert!((model.evaluate([q], P::Reject).unwrap() - q.sin()).abs() <= bound);
        }
    }
}
