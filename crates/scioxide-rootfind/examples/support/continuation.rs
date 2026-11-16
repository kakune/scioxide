//! Analytically specified continuation fixtures used by examples.
use scioxide_core::JacobianOperator;
use scioxide_rootfind::ResidualFn;

/// `F(x, λ) = x² − λ`, whose solution curve folds at `(0, 0)`.
pub fn fold() -> impl JacobianOperator<f64> {
    ResidualFn::with_parameter(1, |y: &[f64], out: &mut [f64]| {
        out[0] = y[0] * y[0] - y[1];
        Ok(())
    })
    .with_jvp(|y: &[f64], v: &[f64], out: &mut [f64]| {
        out[0] = 2.0 * y[0] * v[0] - v[1];
        Ok(())
    })
}

/// `F(x) = x² − c`, with roots `±√c`.
pub fn square(constant: f64) -> impl JacobianOperator<f64> {
    ResidualFn::new(1, move |x: &[f64], out: &mut [f64]| {
        out[0] = x[0] * x[0] - constant;
        Ok(())
    })
    .with_jvp(|x: &[f64], v: &[f64], out: &mut [f64]| {
        out[0] = 2.0 * x[0] * v[0];
        Ok(())
    })
}
