//! Evaluate all nine classical families through the same immutable enum.
//! Run with `cargo run -p scioxide-poly --example orthogonal_families`.

use scioxide_core::SciOxideError;
use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily};

fn main() -> Result<(), SciOxideError> {
    let families = [
        OrthogonalFamily::ChebyshevT,
        OrthogonalFamily::ChebyshevU,
        OrthogonalFamily::Legendre,
        OrthogonalFamily::HermiteH,
        OrthogonalFamily::HermiteHe,
        OrthogonalFamily::Laguerre,
        OrthogonalFamily::GeneralizedLaguerre { alpha: 0.5 },
        OrthogonalFamily::Jacobi {
            alpha: 0.5,
            beta: 1.5,
        },
        OrthogonalFamily::Gegenbauer { lambda: 1.5 },
    ];
    for family in families {
        let value = family.evaluate(2, 0.5_f64)?;
        let derivative = family.evaluate_derivative(2, 0.5)?;
        let weight = family.weight(0.5)?;
        let zeros = family.zeros(5)?;
        println!(
            "{family:?}: P₂(0.5)={value}, P₂′(0.5)={derivative}, w(0.5)={weight}, zeros={zeros:?}"
        );

        // The same family selects an expansion's normalization and coordinate map.
        let spec = OrthogonalBasisSpec::classical(family)?;
        let polynomial = ApproxPolynomial::orthogonal(vec![0.0, 0.0, 1.0], spec)?;
        let evaluation = polynomial.evaluate_with_derivative(0.5)?;
        assert!((evaluation.value() - value).abs() <= 1e-12 * (1.0 + value.abs()));
        assert!((evaluation.derivative() - derivative).abs() <= 1e-12 * (1.0 + derivative.abs()));
    }
    Ok(())
}
