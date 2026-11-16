//! Range-safe resultant and prepared discriminant.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_poly::{PolynomialApproximationPolicy, SylvesterDiscriminant, approximate_resultant};
fn main() -> Result<(), SciOxideError> {
    let policy = PolynomialApproximationPolicy::default();
    let f = [1f64, 0., 1.];
    let g = [-1f64, 0., 1.];
    let result = approximate_resultant(&f, &g)?;
    println!(
        "resultant = {}, reciprocal matrix condition = {}",
        result.value().try_value()?,
        result.report().reciprocal_condition()
    );
    let plan = SylvesterDiscriminant::new(2, policy)?;
    let mut workspace = plan.workspace()?;
    let discriminant =
        plan.compute_scoped(&f, &mut workspace, ExecutionContext::sequential_native())?;
    println!("discriminant = {}", discriminant.value().try_value()?);
    Ok(())
}
