//! Reuse Anderson history storage for independent fixed-point solves.
use scioxide_core::SciOxideError;
use scioxide_rootfind::{
    FixedPointMethod, FixedPointOptions, FixedPointResult, FixedPointSolver, ResidualFn,
};
fn main() -> Result<(), SciOxideError> {
    let solver = FixedPointSolver::new(
        FixedPointMethod::Anderson,
        scioxide_rootfind::FixedPointInput::Map,
        FixedPointOptions::<f64>::default(),
    )?;
    let map = ResidualFn::new(2, |x: &[f64], out: &mut [f64]| {
        out.iter_mut().zip(x).for_each(|(out, x)| *out = x.cos());
        Ok(())
    });
    let mut workspace = solver.workspace(&map)?;
    let mut result = FixedPointResult::new(2)?;
    for initial in [[1.0, 1.0], [0.0, 0.0]] {
        solver.solve_into(
            &map,
            &initial,
            &mut result,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        println!(
            "{:?}: x={:?}, residual={:?}",
            result.termination(),
            result.point(),
            result.residual()
        );
    }
    Ok(())
}
