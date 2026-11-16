//! Reuse an implicit plan and workspace for a stiff scalar problem.
//!
//! Run with `cargo run -p scioxide-ode --example stiff_systems`.

use std::error::Error;

use scioxide_core::{
    ErrorTolerance, EvaluationRole, ExecutionContext, NormSelection, SciOxideError, SolveContext,
    Tolerance,
};
use scioxide_linalg::{MatrixViewMut, Vector};
use scioxide_ode::{BackwardEuler, OdeMethod, OdeOptions, OdeProblem, OdeState, OdeSystem};

struct StiffDecay;

impl OdeSystem<f64> for StiffDecay {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(
        &self,
        _time: f64,
        state: &[f64],
        derivative: &mut [f64],
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        derivative[0] = -1_000.0 * state[0];
        Ok(())
    }

    fn jacobian(
        &self,
        _time: f64,
        _state: &[f64],
        jacobian: &mut MatrixViewMut<'_, f64>,
        _cx: &mut SolveContext<'_, '_>,
    ) -> Result<Option<()>, SciOxideError> {
        let jacobian_data = jacobian
            .as_contiguous_slice_mut()
            .ok_or_else(|| SciOxideError::external("row-major output required"))?;
        jacobian_data[0] = -1_000.0;
        Ok(Some(()))
    }

    fn jacobian_is_constant(&self) -> bool {
        true
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let tolerance =
        ErrorTolerance::from_tolerance(Tolerance::new(1e-10, 1e-8)?, NormSelection::RootMeanSquare);
    let options = OdeOptions::new()
        .with_error_tolerance(tolerance)
        .with_initial_step(1e-4);
    let plan = (BackwardEuler::new(), options);
    let problem = OdeProblem::new(&StiffDecay, 0.02);
    let mut workspace = plan.0.workspace(&problem, &plan.1)?;

    for initial in [1.0, 2.0] {
        let mut state = OdeState::new(0.0, Vector::from_vec(vec![initial]));
        let report = plan.0.advance(
            &problem,
            &plan.1,
            &mut state,
            &mut (),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        println!(
            "initial={initial}, final={:.6e}, factorizations={}, cache hits={}",
            state.values()[0],
            report.counts().provider(EvaluationRole::Factorization),
            report.counts().cache_hits()
        );
    }
    Ok(())
}
