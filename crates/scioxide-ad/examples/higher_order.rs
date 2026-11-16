//! Jacobians, Hessian-vector products, Hessians and directional Taylor series.
use scioxide_ad::{Real, hessian, hvp, jacobian, jet};
use scioxide_core::SciOxideError;

fn energy<R: Real>(x: &[R]) -> R {
    x.windows(2)
        .map(|w| (w[1] - w[0] * w[0]).powi(2) * 10.0)
        .sum::<R>()
        + x[0].exp()
}

fn main() -> Result<(), SciOxideError> {
    let point = [0.5, -0.25, 1.0];
    let j = jacobian(
        |x, y| {
            y.push(x[0] * x[1]);
            y.push(x[2].sin());
            Ok(())
        },
        &point,
    )?;
    assert_eq!(j.as_slice(), [-0.25, 0.5, 0.0, 0.0, 0.0, 1.0_f64.cos()]);
    let h = hessian(|x| Ok(energy(x)), &point)?;
    let v = [1.0, 2.0, -1.0];
    let hv = hvp(|x| Ok(energy(x)), &point, &v)?;
    for i in 0..3 {
        let row: f64 = (0..3).map(|k| h.as_slice()[i * 3 + k] * v[k]).sum();
        assert!((row - hv.as_slice()[i]).abs() < 1e-12);
    }
    // The second Taylor coefficient along v is ½·vᵀHv.
    let c: [f64; 3] = jet(|x| Ok(energy(x)), &point, &v)?;
    let quadratic: f64 = (0..3).map(|i| v[i] * hv.as_slice()[i]).sum();
    assert!((c[2] - quadratic / 2.0).abs() < 1e-12);
    println!("H·v = {:?}", hv.as_slice());
    Ok(())
}
