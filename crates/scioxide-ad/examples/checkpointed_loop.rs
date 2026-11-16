//! Reverse a long time-stepping loop with binomial checkpoints.
use std::sync::Arc;

use scioxide_ad::{CheckpointSchedule, LoopBody, Real, checkpointed_loop, grad};
use scioxide_core::SciOxideError;

/// Explicit Euler for the logistic equation `ẏ = r·y·(1 − y)`.
struct Logistic {
    dt: f64,
}

impl LoopBody for Logistic {
    fn step<R: Real>(
        &self,
        _: usize,
        state: &mut [R],
        parameters: &[R],
    ) -> Result<(), SciOxideError> {
        let y = state[0];
        state[0] = y + parameters[0] * y * (-y + 1.0) * self.dt;
        Ok(())
    }
}

fn main() -> Result<(), SciOxideError> {
    let steps = 1000;
    let body = Arc::new(Logistic { dt: 0.01 });
    // Four snapshots instead of a thousand stored steps.
    let schedule = CheckpointSchedule::binomial(steps, 4)?;
    println!("forward steps when reversing: {}", schedule.forward_steps());
    let checkpointed = grad(
        |x| {
            let mut y = [x[0]];
            checkpointed_loop(&body, steps, &mut y, &x[1..], &schedule)?;
            Ok(y[0])
        },
        &[0.1, 0.8],
    )?;
    let stored = grad(
        |x| {
            let mut y = [x[0]];
            for index in 0..steps {
                body.step(index, &mut y, &x[1..])?;
            }
            Ok(y[0])
        },
        &[0.1, 0.8],
    )?;
    for (a, b) in checkpointed.as_slice().iter().zip(stored.as_slice()) {
        assert!((a - b).abs() < 1e-12 * b.abs());
    }
    println!("∂y/∂(y₀, r) = {:?}", checkpointed.as_slice());
    Ok(())
}
