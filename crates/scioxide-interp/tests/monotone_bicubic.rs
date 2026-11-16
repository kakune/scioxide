//! Shape and interface tests independent of the construction constraints.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, RealScalar};
use scioxide_interp::{
    BicubicInterpolator, Differentiable, ExtrapolationPolicy, Interpolant, TwiceDifferentiable,
};
fn qualify<S: RealScalar>() {
    let cv = |v: f64| S::try_from_f64_lossy(v).unwrap();
    let x = [0., 0.125, 1., 3.].map(cv).to_vec();
    let y = [-1., 0., 0.25, 4.].map(cv).to_vec();
    let z = [
        0., 0., 1., 2., 0., 0.5, 2., 3., 0., 0.5, 2., 5., 0.1, 1., 4., 8.,
    ]
    .map(cv)
    .to_vec();
    let plan = BicubicInterpolator::monotone(x.clone(), y.clone(), z.clone()).unwrap();
    let p = ExtrapolationPolicy::Reject;
    let tol = cv(256.) * S::EPSILON;
    for j in 0..y.len() - 1 {
        for i in 0..x.len() - 1 {
            for a in 0..=10 {
                for b in 0..=10 {
                    let qx = x[i] + (x[i + 1] - x[i]) * cv(f64::from(a) / 10.);
                    let qy = y[j] + (y[j + 1] - y[j]) * cv(f64::from(b) / 10.);
                    let value = plan.evaluate([qx, qy], p).unwrap();
                    let [dx, dy] = plan.gradient([qx, qy], p).unwrap();
                    assert!(value >= z[j * 4 + i] - tol && value <= z[(j + 1) * 4 + i + 1] + tol);
                    assert!(dx >= -tol && dy >= -tol);
                }
            }
        }
    }
    let transposed = (0..4)
        .flat_map(|i| {
            let z = &z;
            (0..4).map(move |j| z[j * 4 + i])
        })
        .collect();
    let transpose = BicubicInterpolator::monotone(y.clone(), x.clone(), transposed).unwrap();
    let q = (cv(0.6), cv(0.1));
    assert!(
        (plan.evaluate([q.0, q.1], p).unwrap() - transpose.evaluate([q.1, q.0], p).unwrap()).abs()
            < tol
    );
    let a = plan.gradient([q.0, q.1], p).unwrap();
    let b = transpose.gradient([q.1, q.0], p).unwrap();
    assert!((a[0] - b[1]).abs() < tol && (a[1] - b[0]).abs() < tol);
    // First derivatives agree on both sides of each internal mesh line.
    let eps = cv(1e-6);
    for &node in &x[1..3] {
        let left = plan.gradient([node - eps, q.1], p).unwrap();
        let right = plan.gradient([node + eps, q.1], p).unwrap();
        assert!((left[0] - right[0]).abs() < cv(2e-4) && (left[1] - right[1]).abs() < cv(2e-4));
    }
    for &node in &y[1..3] {
        let left = plan.gradient([q.0, node - eps], p).unwrap();
        let right = plan.gradient([q.0, node + eps], p).unwrap();
        assert!((left[0] - right[0]).abs() < cv(2e-4) && (left[1] - right[1]).abs() < cv(2e-4));
    }
    let h = plan.hessian([q.0, q.1], p).unwrap();
    assert_eq!(h[0][1], h[1][0]);
}
#[test]
fn shape_both_precisions() {
    qualify::<f64>();
    qualify::<f32>();
}
#[test]
fn affine_planes_and_constant_axes() {
    for a in [-2.0_f64, 0., 2.] {
        for b in [-3., 0., 3.] {
            let x = vec![0., 0.25, 1., 4.];
            let y = vec![-1., 0., 2.];
            let z = y
                .iter()
                .flat_map(|v| x.iter().map(move |u| a * u + b * v + 1.))
                .collect();
            let plan = BicubicInterpolator::monotone(x, y, z).unwrap();
            let p = ExtrapolationPolicy::Reject;
            assert!(
                (plan.evaluate([0.5, 0.3], p).unwrap() - (0.5 * a + 0.3 * b + 1.)).abs() < 1e-12
            );
            let d = plan.gradient([0.5, 0.3], p).unwrap();
            assert!((d[0] - a).abs() < 1e-12 && (d[1] - b).abs() < 1e-12);
            let h = plan.hessian([0.5, 0.3], p).unwrap();
            assert!(h.iter().flatten().all(|v| v.abs() < 1e-11));
        }
    }
}
#[test]
fn two_by_two_complex_components_and_invalid_grids() {
    let z = vec![
        Complex::new(0., 0.),
        Complex::new(1., -1.),
        Complex::new(2., 3.),
        Complex::new(3., 2.),
    ];
    let plan = BicubicInterpolator::monotone(vec![0., 1.], vec![0., 1.], z).unwrap();
    let v = plan
        .evaluate([0.25, 0.5], ExtrapolationPolicy::Reject)
        .unwrap();
    assert!((v.re - 1.25_f64).abs() < 1e-12 && (v.im - 1.25_f64).abs() < 1e-12);
    assert!(
        BicubicInterpolator::monotone(vec![0., 1., 2.], vec![0., 1.], vec![0., 2., 1., 3., 4., 5.])
            .is_err()
    );
    assert!(
        BicubicInterpolator::monotone(vec![0., 1.], vec![0., 1.], vec![0., 1., 1., f64::NAN])
            .is_err()
    );
}
