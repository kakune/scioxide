//! Explicit simplex, fixed scales and reusable Nelder–Mead storage.
use scioxide_core::{Budget, ExecutionContext, SciOxideError};
use scioxide_linalg::Matrix;
use scioxide_optimize::{NelderMead, NelderMeadOptions, ObjectiveFn, OptimizeResult};

fn main() -> Result<(), SciOxideError> {
    let objective = ObjectiveFn::new(2, |x: &[f64]| {
        Ok((x[0] - 8.0).powi(2) + 4.0 * (x[1] - 8.0).powi(2))
    });
    let simplex = Matrix::from_vec(3, 2, vec![10.0, 7.0, 10.5, 7.0, 10.0, 7.25])?;
    let options = NelderMeadOptions::default()
        .with_initial_simplex(simplex)
        .with_reference_scales(1.0, 1.0)
        .with_max_evaluations(Budget::try_from(2_000)?);
    let solver = NelderMead::new(options)?;
    let mut workspace = solver.workspace(2)?;
    let mut output = OptimizeResult::empty();
    output.try_reserve(2)?;
    solver.minimize_into(
        &objective,
        &[10.0, 7.0],
        &mut output,
        &mut workspace,
        ExecutionContext::sequential(),
    )?;
    println!(
        "{:?}: x={:?}, calls={}, geometry={:?}",
        output.termination(),
        output.x(),
        output.fun_evals(),
        output.method_diagnostics()
    );
    Ok(())
}
