//! Parameter equations and exact confidence limits (std and no_std numerical APIs).
use scioxide_core::{Interval, SciOxideError};
use scioxide_random::InversionOptions;
use scioxide_stats::*;
fn main() -> Result<(), SciOxideError> {
    let request = ParameterInversionRequest {
        model: ProbabilityParameter::GammaRate {
            x: 2.0_f64,
            shape: 1.0,
        },
        bracket: Interval::new(0.1, 10.0)?,
        target: ParameterProbability::from_log(-6.0, scioxide_stats::TailSide::Upper)?,
        options: InversionOptions::default(),
        tail_options: Default::default(),
    };
    let report = request.solve_scoped(scioxide_core::ExecutionContext::sequential_native())?;
    println!(
        "parameter outcome: {:?}; root: {:?}; calls: {}",
        report.solution(),
        report.root().map(|r| r.termination()),
        report.evaluations()
    );
    let limits = binomial_confidence_limits_scoped::<f64>(
        3,
        10,
        ConfidenceProbability::default(),
        TestAlternative::TwoSided,
        &InversionOptions::default(),
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    println!("95% binomial proportion set: {:?}", limits.set());
    Ok(())
}
