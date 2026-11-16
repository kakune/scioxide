//! Explicit normalization, affine coordinates, and basis-native calculus.
use scioxide_core::SciOxideError;
use scioxide_poly::PolynomialBasis;
use scioxide_poly::{ApproxPolynomial, BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily};

fn main() -> Result<(), SciOxideError> {
    let spec = OrthogonalBasisSpec::new(
        OrthogonalFamily::Jacobi {
            alpha: 0.25,
            beta: 0.75,
        },
        BasisNormalization::Orthonormal,
        2.0,
        3.0,
    )?;
    let p = ApproxPolynomial::orthogonal(vec![1.0_f64, 0.5, -0.25], spec)?;
    let derivative = p.derivative()?;
    let primitive = p.antiderivative(2.0, 0.0)?;
    let target = OrthogonalBasisSpec::new(
        OrthogonalFamily::Gegenbauer { lambda: 1.5 },
        BasisNormalization::Monic,
        0.0,
        1.0,
    )?;
    let converted = p.to_orthogonal(target)?;
    println!(
        "p(2) = {:?}, derivative = {:?}",
        p.evaluate(2.0)?,
        derivative.evaluate(2.0)?
    );
    println!(
        "primitive(2) = {:?}, integral[1,3] = {:?}",
        primitive.evaluate(2.0)?,
        p.integral(1.0, 3.0)?
    );
    println!(
        "converted p(2) = {:?}",
        converted.polynomial().evaluate(2.0)?
    );
    let borrowed = p
        .basis()
        .plan()
        .evaluate_with_derivative(p.basis().coefficients(), 2.0)?;
    println!(
        "borrowed condition estimate = {:?}",
        borrowed.condition().condition_estimate()
    );
    Ok(())
}
