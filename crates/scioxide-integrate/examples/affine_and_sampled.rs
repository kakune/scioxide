//! Complex callback integration and nonuniform sampled cumulative integration.
use scioxide_core::{
    Budget, Complex, ErrorTolerance, ExecutionContext, Interval, NormSelection, SciOxideError,
};
use scioxide_integrate::{
    AdaptiveIntegrator, AdaptiveRule, SampledSimpsonPlan, SubdivisionOptions,
};
fn main() -> Result<(), SciOxideError> {
    let options = SubdivisionOptions::default()
        .with_tolerance(ErrorTolerance::scalar(
            1e-10,
            1e-10,
            NormSelection::Infinity,
        )?)
        .with_max_subdivisions(Budget::try_from(64)?);
    let solver = AdaptiveIntegrator::new(AdaptiveRule::GaussKronrod, options)?;
    // A complex value has two real components, each with its own error evidence.
    let mut workspace = solver.workspace(2)?;
    let mut integral = Complex::new(0.0, 0.0);
    let report = solver.integrate_into(
        |x: f64| Ok(Complex::new(x.cos(), x.sin())),
        Interval::new(0.0, core::f64::consts::PI)?,
        &mut integral,
        &mut workspace,
        ExecutionContext::sequential_native(),
    )?;
    println!(
        "Integral: {integral:?}; {:?}; {} calls",
        report.termination(),
        report.evaluations()
    );
    if let Some(errors) = workspace.component_errors() {
        let totals = errors.iter().map(|error| error.total()).collect::<Vec<_>>();
        println!("Per-component error estimates: {totals:?}");
    }
    let coordinates = [0.0_f64, 0.2, 0.6, 1.0];
    let samples = coordinates.map(|x| Complex::new(x * x, x));
    let plan = SampledSimpsonPlan::new(&coordinates)?;
    let mut workspace = plan.cumulative_workspace(2)?;
    let mut cumulative = [Complex::new(0.0, 0.0); 4];
    plan.cumulative_into(&samples, &mut cumulative, &mut workspace)?;
    println!("Cumulative sampled integrals: {cumulative:?}");
    Ok(())
}
