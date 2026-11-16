//! Reuse a checked repeated-pole factorization across numerators.
use scioxide_core::SciOxideError;
use scioxide_poly::{ApproxPartialFractionPlan, ApproxPolynomial};

fn main() -> Result<(), SciOxideError> {
    let linear = ApproxPolynomial::monomial(vec![-1.0_f64, 1.0])?;
    let quadratic = ApproxPolynomial::monomial(vec![1.0, 0.0, 1.0])?;
    let denominator = linear.try_mul(&linear)?.try_mul(&quadratic)?;
    let plan = ApproxPartialFractionPlan::new(&denominator, &[(&linear, 2), (&quadratic, 1)])?;
    // 1/(x-1) + 2/(x-1)^2 + (3+4x)/(x²+1).
    let result = plan.decompose(&[4.0, -1.0, -4.0, 5.0])?;
    println!("polynomial part: {:?}", result.polynomial());
    for term in plan.terms() {
        println!(
            "factor {} power {}: {:?}",
            term.factor_index(),
            term.power(),
            &result.residues()[term.coefficients()]
        );
    }
    println!("diagnostics: {:?}", result.report());
    Ok(())
}
