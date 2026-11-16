//! Sparse constructs the coloring; scioxide-differentiate differentiates and returns canonical CSR values.
use scioxide_core::{SciOxideError, StructuralPatternView};
use scioxide_differentiate::{AdaptiveDifference, ColoredJacobianPlan, sparse_jacobian};
use scioxide_sparse::derivative_coloring;
fn main() -> Result<(), SciOxideError> {
    let pattern = StructuralPatternView::new(3, 3, &[0, 1, 3, 4], &[0, 0, 1, 2])?;
    let schedule = derivative_coloring(pattern)?;
    let function = |x: &[f64], y: &mut [f64]| {
        y[0] = x[0] * x[0];
        y[1] = x[0] + x[1].powi(3);
        y[2] = x[2].sin();
        Ok(())
    };
    let x = [1.0, 2.0, 3.0];
    let values = sparse_jacobian(function, &x, &schedule)?;
    println!("CSR values: {values:?}");
    let plan = ColoredJacobianPlan::new(&schedule, AdaptiveDifference::<f64>::default())?;
    let result = plan.evaluate(function, &x, None)?;
    println!(
        "{} colors, {} callback attempts, {:?}",
        schedule.color_count(),
        result.report().evaluations(),
        result.report().termination()
    );
    println!("estimated errors: {:?}", result.errors());
    Ok(())
}
