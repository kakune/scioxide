//! Reuse products and composition without query allocations.
use scioxide_core::SciOxideError;
use scioxide_poly::PolynomialOperationRequest;
use scioxide_poly::{PolynomialOperationPlan, polynomial_product_condition};
fn main() -> Result<(), SciOxideError> {
    let plan = PolynomialOperationPlan::<f64>::new(&[
        PolynomialOperationRequest::Multiply {
            left_len: 3,
            right_len: 2,
        },
        PolynomialOperationRequest::TaylorShift { len: 3 },
    ])?;
    let mut work = plan.workspace(plan.requests())?;
    let (a, b) = ([1., 2., 3.], [-1., 1.]);
    let mut product = [0.; 4];
    let algorithm = plan.multiply_into(
        &a,
        &b,
        &mut product,
        &mut work,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    let report = polynomial_product_condition(&a, &b, &product)?;
    println!(
        "{algorithm:?}: {product:?}; sensitivity {}",
        report.condition_estimate()
    );
    let mut shift = [0.; 3];
    plan.taylor_shift_into(
        &a,
        2.,
        &mut shift,
        &mut work,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    println!("shift {shift:?}");
    Ok(())
}
