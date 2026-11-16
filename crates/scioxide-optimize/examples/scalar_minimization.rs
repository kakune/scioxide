//! Bounded, automatically bracketed, and injected scalar minimization.
use scioxide_core::{CancellationToken, ExecutionContext, Interval, SciOxideError};
use scioxide_optimize::{ScalarMinimizer, ScalarMinimumStart};
fn main() -> Result<(), SciOxideError> {
    let objective = |x: f64| Ok((x - 0.25).powi(2));
    let solver = ScalarMinimizer::default();
    let start = ScalarMinimumStart::Bounded(Interval::new(-1.0, 2.0)?);
    let result = solver.minimize(&objective, start)?;
    println!(
        "bounded: x={}, f={}, status={}",
        result.x(),
        result.fun(),
        result.termination()
    );
    let bracket = solver.bracket_minimum(&objective, -1.0, 0.0)?;
    if let Some(bracket) = bracket.bracket() {
        let result = solver.minimize(&objective, ScalarMinimumStart::Bracket(bracket))?;
        println!("bracketed: x={}, calls={}", result.x(), result.fun_evals());
    }
    let cancellation = CancellationToken::new();
    let scope = ExecutionContext::sequential().with_cancellation(&cancellation);
    let result = solver.minimize_scoped(&objective, start, scope)?;
    assert!(result.is_success());
    Ok(())
}
