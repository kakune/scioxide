//! Real-coordinate complex derivatives with a reserved workspace.
use scioxide_core::{
    Budget, BudgetKind, Complex, ErrorTolerance, ExecutionContext, NormSelection, ProblemToken,
    SciOxideError, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_differentiate::*;

fn main() -> Result<(), SciOxideError> {
    let method = AdaptiveDifference::default().with_tolerance(ErrorTolerance::scalar(
        1e-9_f64,
        1e-9,
        NormSelection::Infinity,
    )?);
    let request = DerivativeRequest::new(&method, DerivativeShape::gradient(1));
    let mut workspace = AdaptiveDifferenceWorkspace::<Complex<f64>>::workspace(request)?;
    println!(
        "reserved adaptive payload: {} bytes",
        AdaptiveDifferenceWorkspace::<Complex<f64>>::workspace_requirements(request)?.bytes()
    );
    let mut values = [Complex::new(0.0, 0.0)];
    let mut errors = [0.0];
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(64)?);
    let identity = SolveIdentity::new(
        ProblemToken::new(1),
        scioxide_core::algorithm_id!("example", "adaptive", 1),
        0,
        0,
    );
    let mut context = state.begin_solve(ExecutionContext::sequential(), identity, &mut budgets)?;
    let report = method.gradient_into(
        |x: &[f64], y: &mut [Complex<f64>]| {
            y[0] = Complex::new(x[0].sin(), x[0] * x[0]);
            Ok(())
        },
        &[1.0],
        None,
        &mut values,
        &mut errors,
        &mut workspace,
        &mut context,
    )?;
    println!(
        "termination: {:?}; evaluations: {}",
        report.termination(),
        report.evaluations()
    );
    if report.output_available() {
        println!(
            "derivative: {:?}; estimated absolute error: {}",
            values[0], errors[0]
        );
    }
    Ok(())
}
