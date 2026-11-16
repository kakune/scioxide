//! Reuse one tape for gradients and linearizations at several points.
use scioxide_ad::{Real, Tape, TapeOptions};
use scioxide_core::{ExecutionContext, SciOxideError};

fn residual<R: Real>(x: &[R], y: &mut Vec<R>) {
    y.push(x[0] * x[1] - 1.0);
    y.push(x[0].sin() + x[1] * x[1]);
}

fn main() -> Result<(), SciOxideError> {
    let cx = ExecutionContext::sequential();
    let mut tape = Tape::<f64>::new(TapeOptions::new());
    let mut gradient = [0.0; 2];
    for point in [[1.0, 2.0], [0.5, -1.0], [2.0, 0.25]] {
        let value = tape.gradient_into(|x| Ok(x[0] * x[1].exp()), &point, &mut gradient, cx)?;
        assert!((value - point[0] * point[1].exp()).abs() < 1e-15);
        let mut linear = tape.linearize(
            |x, y| {
                residual(x, y);
                Ok(())
            },
            &point,
            cx,
        )?;
        let mut j = [0.0; 4];
        linear.jacobian_into(&mut j, cx)?;
        assert_eq!(j, [point[1], point[0], point[0].cos(), 2.0 * point[1]]);
    }
    // After the first recording the tape has the capacity it needs.
    assert!(!tape.report().recorded_twice());
    println!(
        "{} words, {} bytes reserved",
        tape.report().words(),
        tape.reserved_bytes()
    );
    Ok(())
}
