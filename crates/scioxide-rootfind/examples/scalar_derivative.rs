//! Open and safeguarded scalar solvers that use derivatives or secants.
//!
//! Run: `cargo run --example scalar_derivative -p scioxide-rootfind`
use scioxide_core::{Interval, SciOxideError};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_rootfind::{
    Halley, Newton, SafeguardedInitialGuess, SafeguardedNewton, ScalarFn, ScalarRootOptions,
    ScalarRootSolver, Secant,
};

fn main() -> Result<(), SciOxideError> {
    // The cube root of 2, with analytic first and second derivatives.
    let cube = ScalarFn::new(|x: f64| Ok(x * x * x - 2.0))
        .with_derivative(|x| Ok(3.0 * x * x))
        .with_second_derivative(|x| Ok(6.0 * x));
    let options = ScalarRootOptions::default();
    let guarded = SafeguardedInitialGuess::new(Interval::new(0.0, 2.0)?, 0.0)?;
    let results = [
        (
            "Newton",
            Newton::new(DerivativeStrategy::Analytic).solve(&cube, 1.0, &options)?,
        ),
        (
            "Newton (central differences)",
            Newton::new(DerivativeStrategy::Central).solve(&cube, 1.0, &options)?,
        ),
        ("Halley", Halley.solve(&cube, 1.0, &options)?),
        // A zero derivative at the start bisects instead of failing.
        (
            "Safeguarded Newton",
            SafeguardedNewton::new(DerivativeStrategy::Analytic).solve(&cube, guarded, &options)?,
        ),
        ("Secant", Secant.solve(&cube, (1.0, 2.0), &options)?),
    ];
    for (name, result) in results {
        assert!(result.is_success());
        assert!((result.point() - 2.0_f64.cbrt()).abs() < 1e-12);
        println!(
            "{name:>28}: x = {:.15}, {} iterations",
            result.point(),
            result.counts().iterations()
        );
    }
    Ok(())
}
