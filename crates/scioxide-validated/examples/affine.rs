//! Correlation, complex elementary evaluation and explicit reduction.
use scioxide_core::{Complex, SciOxideError};
use scioxide_validated::{AffineBinary, AffineContext, AffineForm, AffineOptions, AffineUnary};
fn main() -> Result<(), SciOxideError> {
    let context = AffineContext::new(
        2,
        AffineOptions::new()
            .with_max_symbols(8)
            .with_byte_limit(64 * 1024),
    )?;
    let mut workspace = context.workspace::<f64>()?;
    let x = AffineForm::from_coefficients(&context, 1.0, &[0.125])?;
    let cancellation = x.binary(&x, AffineBinary::Sub, &mut workspace)?;
    assert_eq!(
        cancellation.report().real().set().bounds(),
        Some((Some(-0.0), Some(0.0)))
    );
    let exponential = x.unary(AffineUnary::Exp, &mut workspace)?;
    let reduced = exponential.reduce(1, &mut workspace)?;
    assert!(
        reduced
            .report()
            .real()
            .contains_interval(&exponential.report().real())
    );
    let mut complex_workspace = context.workspace::<Complex<f64>>()?;
    let z = AffineForm::from_coefficients(
        &context,
        Complex::new(0.5, 0.25),
        &[Complex::new(0.015625, 0.0078125)],
    )?;
    let result = z.unary(AffineUnary::Exp, &mut complex_workspace)?;
    println!(
        "real exp: {:?}; reduced: {:?}; complex exp: {:?}",
        exponential.report(),
        reduced.report(),
        result.report()
    );
    Ok(())
}
