//! Forward mode: one tangent, several tangents and Taylor coefficients.
use scioxide_ad::{Dual, Jet, MultiDual, Real};
use scioxide_core::SciOxideError;

/// `f(x, y) = x·exp(y) + sin(x·y)`, written once.
fn f<R: Real>(x: R, y: R) -> R {
    x * y.exp() + (x * y).sin()
}

fn main() -> Result<(), SciOxideError> {
    let (x, y) = (0.7, -0.3);
    // One directional derivative: ∂f/∂x.
    let d = f(Dual::variable(x), Dual::new(y, 0.0));
    let fx = y.exp() + y * (x * y).cos();
    assert!((d.derivative() - fx).abs() < 1e-15);
    // The gradient in one pass.
    let [mx, my] = MultiDual::<f64, 2>::variables([x, y]);
    let g = f(mx, my);
    let fy = x * y.exp() + x * (x * y).cos();
    assert!((g.gradient()[0] - fx).abs() < 1e-15 && (g.gradient()[1] - fy).abs() < 1e-15);
    // Taylor coefficients cₖ = (1/k!)·dᵏ/dtᵏ f(x + t, y).
    let t = f(Jet::<f64, 4>::variable(x), Jet::new([y, 0.0, 0.0, 0.0]));
    let fxx = -y * y * (x * y).sin();
    assert!((t.coefficients()[2] - fxx / 2.0).abs() < 1e-15);
    println!(
        "f = {}, ∇f = {:?}, jet = {:?}",
        d.value(),
        g.gradient(),
        t.coefficients()
    );
    Ok(())
}
