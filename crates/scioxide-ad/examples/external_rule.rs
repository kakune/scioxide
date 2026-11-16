//! A matrix-free rule with its own pullback and pushforward on a tape.
use std::sync::Arc;

use scioxide_ad::{ExternalRule, Real, Tape, TapeOptions, TapeValue, Var};
use scioxide_core::{ExecutionContext, SciOxideError};

/// `y = A·x` for the fixed tridiagonal `A = tridiag(−1, 2, −1)`, never stored.
struct Laplacian;

fn row<V: TapeValue>(x: &[V], i: usize) -> V {
    let mut value = x[i] + x[i];
    if i > 0 {
        value -= x[i - 1];
    }
    if i + 1 < x.len() {
        value -= x[i + 1];
    }
    value
}

fn apply<V: TapeValue>(x: &[V], y: &mut [V]) {
    for (i, target) in y.iter_mut().enumerate() {
        *target = row(x, i);
    }
}

impl<V: TapeValue> ExternalRule<V> for Laplacian {
    fn evaluate(&self, x: &[V], y: &mut [V]) -> Result<(), SciOxideError> {
        apply(x, y);
        Ok(())
    }
    fn pullback(&self, _: &[V], _: &[V], w: &[V], g: &mut [V]) -> Result<(), SciOxideError> {
        // A is symmetric: Aᵀ·w = A·w.
        for (i, target) in g.iter_mut().enumerate() {
            *target += row(w, i);
        }
        Ok(())
    }
    fn pushforward(&self, _: &[V], _: &[V], t: &[V], out: &mut [V]) -> Result<(), SciOxideError> {
        apply(t, out);
        Ok(())
    }
}

fn main() -> Result<(), SciOxideError> {
    let rule: Arc<dyn ExternalRule<f64>> = Arc::new(Laplacian);
    let cx = ExecutionContext::sequential();
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    // E(x) = ½·xᵀAx + Σ sin xᵢ has gradient A·x + cos x.
    let point = [0.3, -0.2, 0.5, 1.0];
    let mut gradient = [0.0; 4];
    tape.gradient_into(
        |x| {
            let mut ax = [Var::constant(0.0); 4];
            Var::external(&rule, x, &mut ax)?;
            Ok(x.iter()
                .zip(&ax)
                .map(|(&a, &b)| a * b * 0.5 + a.sin())
                .sum())
        },
        &point,
        &mut gradient,
        cx,
    )?;
    let mut expected = [0.0; 4];
    apply(&point, &mut expected);
    for (g, (e, x)) in gradient.iter().zip(expected.iter().zip(&point)) {
        assert!((g - (e + x.cos())).abs() < 1e-15);
    }
    println!("∇E = {gradient:?}");
    Ok(())
}
