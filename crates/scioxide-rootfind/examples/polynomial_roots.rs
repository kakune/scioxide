//! Verified polynomial roots, native bases, and explicit partial results.
use scioxide_core::{Budget, Complex, SciOxideError};
use scioxide_poly::{ApproxPolynomial, polynomial_root_diagnostics};
use scioxide_rootfind::{PolynomialRootMethod, PolynomialRootOptions};

fn main() -> Result<(), SciOxideError> {
    let options = PolynomialRootOptions::default();
    let coefficients = [
        Complex::new(-2.0_f64, 2.0),
        Complex::new(-1.0, -1.0),
        Complex::new(-1.0, 0.0),
        Complex::new(1.0, 0.0),
    ];
    let roots = scioxide_rootfind::polynomial_roots_with_options(
        &coefficients,
        &options.clone().with_method(PolynomialRootMethod::Aberth),
    )?;
    println!("complex polynomial: {:?}", roots.report().termination());
    for (root, diagnostic) in roots.iter().zip(roots.diagnostics()) {
        if diagnostic.is_available() {
            println!(
                "{root:?}: error={:?}, checked={}",
                diagnostic.backward_error(),
                diagnostic.is_validated()
            );
        }
    }
    if roots.report().termination().is_success() {
        let diagnostics = polynomial_root_diagnostics(&coefficients, roots.roots())?;
        println!(
            "cluster uncertainty (not a multiplicity certificate): {:?}",
            diagnostics.clusters()
        );
    }
    let chebyshev = ApproxPolynomial::orthogonal(
        vec![0.0_f64, 0.0, 0.0, 1.0],
        scioxide_poly::OrthogonalBasisSpec::classical(scioxide_poly::OrthogonalFamily::ChebyshevT)?,
    )?;
    let native = scioxide_rootfind::polynomial_roots_with_options(&chebyshev, &options)?;
    println!(
        "T3: {:?}; model={:?}; status={:?}",
        native.roots(),
        native.report().backward_error_model(),
        native.report().termination()
    );

    let limited = options
        .with_max_iterations(Budget::ONE)
        .with_restart_budget(Budget::ONE)
        .with_polish(false);
    let partial = scioxide_rootfind::polynomial_roots_with_options(
        &coefficients,
        &limited.with_method(PolynomialRootMethod::Aberth),
    )?;
    println!(
        "limited solve: {:?}; inspect each diagnostic before using a candidate",
        partial.report().termination()
    );
    Ok(())
}
