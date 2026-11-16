//! Differentiate an integral using an AD Jacobian on a tape and a moving upper bound.
#[path = "support/sensitivity_ad.rs"]
mod sensitivity_ad;
use scioxide_core::{ErrorTolerance, ExecutionContext, Interval, NormSelection};
use scioxide_integrate::{
    GaussKronrodRule, JvpQuery, QuadratureSensitivityPlan, SensitivityMode, SensitivityOptions,
};
fn main() -> Result<(), scioxide_core::SciOxideError> {
    let integrand = sensitivity_ad::Exponential;
    let mut provider_workspace = integrand.workspace::<f64>()?;
    let tolerance = ErrorTolerance::scalar(1e-10, 1e-10, NormSelection::Infinity)?;
    let options = SensitivityOptions::default()
        .with_mode(SensitivityMode::Jvp)
        .with_primal_tolerance(tolerance.clone())
        .with_derivative_tolerance(tolerance);
    let plan = QuadratureSensitivityPlan::fixed(
        &integrand,
        Interval::new(0.0, 1.0)?,
        GaussKronrodRule::G10K21,
        options,
    )?;
    let mut workspace = plan.workspace()?;
    let (mut value, mut derivative) = (0.0, 0.0);
    // At p=0.7: a=0, b=1, Da=0, Db=0.3; the parameter direction is one.
    let report = plan.jvp_into(
        &integrand,
        JvpQuery::new(&[0.7], &[1.0]).with_endpoint_tangent(Some([0.0, 0.3])),
        (&mut value, &mut derivative),
        (
            &mut workspace,
            core::slice::from_mut(&mut provider_workspace),
        ),
        ExecutionContext::sequential_native(),
    )?;
    println!(
        "integral={value}, derivative={derivative}, estimated derivative error={:?}, termination={:?}",
        workspace
            .derivative_errors()
            .and_then(|errors| errors.first())
            .and_then(|error| error.total()),
        report.termination()
    );
    Ok(())
}
