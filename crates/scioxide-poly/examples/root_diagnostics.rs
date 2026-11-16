//! Ordinary estimates from borrowed coefficients and solver-supplied candidates.
use scioxide_core::{Complex, SciOxideError};
use scioxide_poly::{ApproxPolynomial, polynomial_root_diagnostics};
fn main() -> Result<(), SciOxideError> {
    let coefficients = [2.0_f64, -3.0, 1.0];
    let candidates = [Complex::new(1.0 + 1e-7, 0.0), Complex::new(2.0 - 1e-7, 0.0)];
    let diagnostics = polynomial_root_diagnostics(&coefficients, &candidates)?;
    println!(
        "Ordinary estimates (not certificates): {:?}",
        diagnostics.status()
    );
    println!("Root radii: {:?}", diagnostics.radii());
    for cluster in diagnostics.clusters() {
        println!(
            "{} candidates: {:?}, gap {:?}",
            cluster.candidate_count(),
            cluster.disk(),
            cluster.separation()
        );
    }
    let newton = ApproxPolynomial::newton(vec![0.0_f64, 1.0], vec![2.0])?;
    println!(
        "Native Newton coefficient model: {:?}",
        newton.root_condition(2.0)?
    );
    let conversion = newton.to_monomial()?;
    println!(
        "Explicit conversion diagnostics: {:?}",
        conversion.condition()
    );
    println!(
        "Monomial coefficient model: {:?}",
        conversion.polynomial().root_condition(2.0)?
    );
    Ok(())
}
