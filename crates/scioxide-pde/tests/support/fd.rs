//! Central finite differences with a tolerance derived from the step.
#![allow(dead_code, unreachable_pub)]

/// Central difference of `f` at `x` with step `h = ∛ε · max(1, |x|)`, and a
/// tolerance: the step-halving truncation estimate `|D_h − D_2h|` (scaled for
/// safety) plus rounding `(64 ε + relative) · max|f| / h`, where `relative` is the
/// accuracy of the evaluated solves.
pub fn central(f: impl Fn(f64) -> f64, x: f64, relative: f64) -> (f64, f64) {
    let h = f64::EPSILON.cbrt() * x.abs().max(1.0);
    let samples = [f(x + h), f(x - h), f(x + 2.0 * h), f(x - 2.0 * h)];
    let d_h = (samples[0] - samples[1]) / (2.0 * h);
    let d_2h = (samples[2] - samples[3]) / (4.0 * h);
    let scale = samples.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let tolerance = 2.0 * (d_h - d_2h).abs() + (64.0 * f64::EPSILON + relative) * scale / h;
    (d_h, tolerance)
}

/// [`central`] for every component of a vector-valued `f`, sharing its four samples.
pub fn central_each(f: impl Fn(f64) -> Vec<f64>, x: f64, relative: f64) -> Vec<(f64, f64)> {
    let h = f64::EPSILON.cbrt() * x.abs().max(1.0);
    let samples = [f(x + h), f(x - h), f(x + 2.0 * h), f(x - 2.0 * h)];
    (0..samples[0].len())
        .map(|i| {
            let d_h = (samples[0][i] - samples[1][i]) / (2.0 * h);
            let d_2h = (samples[2][i] - samples[3][i]) / (4.0 * h);
            let scale = samples.iter().fold(0.0_f64, |m, v| m.max(v[i].abs()));
            let tolerance = 2.0 * (d_h - d_2h).abs() + (64.0 * f64::EPSILON + relative) * scale / h;
            (d_h, tolerance)
        })
        .collect()
}
