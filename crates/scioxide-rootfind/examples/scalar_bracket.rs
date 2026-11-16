//! Bracketing scalar solvers: find a sign change, then refine its certificate.
//!
//! Run: `cargo run --example scalar_bracket -p scioxide-rootfind`
use scioxide_core::{EvaluationRole, SciOxideError};
use scioxide_rootfind::{
    BracketVerification, ScalarFn, ScalarRootInput, ScalarRootMethod, ScalarRootOptions,
    ScalarRootSolver, find_bracket, find_root,
};

fn main() -> Result<(), SciOxideError> {
    // Kepler's equation E − e sin E = M for e = 0.5 and M = 1.
    let kepler = |e: f64| -> Result<f64, SciOxideError> { Ok(e - 0.5 * e.sin() - 1.0) };
    let bracket = find_bracket(kepler, 0.0)?;
    let anomaly = find_root(kepler, bracket.lower(), bracket.upper())?;
    println!("one-shot: E = {anomaly:.15}");

    // The configured methods report their bracket and work counts.
    let kepler = ScalarFn::new(kepler);
    assert_eq!(bracket.verify(&kepler)?, BracketVerification::Verified);
    let options = ScalarRootOptions::default();
    for method in [
        ScalarRootMethod::Bisection,
        ScalarRootMethod::Illinois,
        ScalarRootMethod::Brent,
        ScalarRootMethod::Toms748,
    ] {
        let input = ScalarRootInput::Bracket(bracket.interval());
        let result = method.solve(&kepler, input, &options)?;
        assert!(result.is_success());
        // Every bracketing result certifies a sign change around its root.
        assert!(
            result
                .bracket()
                .is_some_and(|c| c.lower() <= result.point() && result.point() <= c.upper())
        );
        println!(
            "{:>9}: E = {:.15}, {} evaluations",
            ScalarRootSolver::<f64>::name(&method),
            result.point(),
            result.counts().provider(EvaluationRole::Function)
        );
    }
    Ok(())
}
