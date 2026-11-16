//! Prepared Chebyshev and approximate rational-function example.

use scioxide_core::SciOxideError;
use scioxide_poly::{
    ApproxPolynomial, ApproxRationalFunction, OrthogonalBasisSpec, OrthogonalFamily,
};

fn main() -> Result<(), SciOxideError> {
    let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT)?;
    let chebyshev = ApproxPolynomial::orthogonal(vec![2.5, 2.0, 1.5], spec)?;
    let evaluation = chebyshev.evaluate_with_derivative(0.25)?;
    let conversion = chebyshev.to_monomial()?;
    println!(
        "Chebyshev value {}, conversion condition {}",
        evaluation.value(),
        conversion.condition().condition_estimate()
    );

    let numerator = ApproxPolynomial::monomial(vec![1.0, 1.0])?;
    let denominator = ApproxPolynomial::monomial(vec![2.0, 1.0])?;
    let rational = ApproxRationalFunction::new(numerator, denominator)?;
    let parts = rational.evaluate_parts(3.0)?;
    println!(
        "r(3) = {}, denominator backward error <= {}",
        parts.value(),
        parts.denominator().condition().backward_error()
    );
    Ok(())
}
