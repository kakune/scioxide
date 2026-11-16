//! One-sided nonuniform weights and complex-step derivatives on real nodes.
use scioxide_core::{Complex64, SciOxideError};
use scioxide_differentiate::{
    ComplexStep, ComplexStepPlan, DerivativeOperation, FiniteDifferenceStencil,
    NodeDifferenceOptions, RelativeComplexStep, RelativeDifference,
};
fn main() -> Result<(), SciOxideError> {
    let nodes = [0.0_f64, 0.1, 0.25, 0.6, 1.0];
    let stencil = FiniteDifferenceStencil::new(0.0, &nodes, 1)?;
    let samples = nodes.map(|x| x * x * x + 2.0 * x);
    println!("boundary weights: {:?}", stencil.weights()?);
    println!("boundary derivative: {}", stencil.apply(&samples)?);
    println!(
        "normalized weight amplification: {}",
        stencil.weight_amplification()
    );
    let plan = ComplexStepPlan::new(&[0.5], &[&nodes], ComplexStep::default().with_step(1e-100))?;
    let result = plan.evaluate(
        |z: &[Complex64], y: &mut [Complex64]| {
            y[0] = z[0] * z[0] * z[0];
            Ok(())
        },
        None,
    )?;
    println!(
        "composed Hessian: {:?}; report: {:?}",
        result.values(),
        result.report()
    );
    let offsets = [-0.25, 0.0, 0.25];
    let real = RelativeDifference::new(
        DerivativeOperation::First,
        &[&offsets],
        1,
        NodeDifferenceOptions::default(),
    )?;
    let complex = RelativeComplexStep::new(&[&offsets], ComplexStep::default().with_step(1e-100))?;
    let mut real_work = real.workspace::<f64>(())?;
    let mut complex_work = complex.workspace(())?;
    let mut derivative = [0.0];
    let mut second = [0.0];
    for point in [0.5, 1.0, 2.0] {
        let report = real.evaluate_into(
            |x: &[f64], y: &mut [_]| {
                y[0] = x[0] * x[0];
                Ok(())
            },
            &[point],
            None,
            &mut derivative,
            &mut real_work,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        let complex_report = complex.evaluate_into(
            |x: &[Complex64], y: &mut [Complex64]| {
                y[0] = x[0] * x[0];
                Ok(())
            },
            &[point],
            None,
            &mut second,
            &mut complex_work,
            scioxide_core::ExecutionContext::sequential(),
        )?;
        println!(
            "point {point}: derivative {}, second {}; completed {} / {}",
            derivative[0],
            second[0],
            report.output_available(),
            complex_report.output_available()
        );
    }
    let hessian = RelativeDifference::new(
        DerivativeOperation::Hessian,
        &[&offsets, &offsets],
        1,
        NodeDifferenceOptions::default(),
    )?;
    let mut hessian_work = hessian.workspace::<f64>(())?;
    let mut matrix = [0.0; 4];
    let report = hessian.evaluate_into(
        |x: &[f64], y: &mut [_]| {
            y[0] = x[0] * x[0] + x[0] * x[1] + x[1] * x[1];
            Ok(())
        },
        &[1.0, 2.0],
        None,
        &mut matrix,
        &mut hessian_work,
        scioxide_core::ExecutionContext::sequential(),
    )?;
    // Three nodes on each axis form nine distinct points. Shared base and
    // axis observations are reused inside the reserved caller workspace.
    println!(
        "Hessian {matrix:?}; {} callbacks; {} reserved bytes",
        report.evaluations(),
        hessian
            .workspace_requirements::<f64>(())
            .map(|required| required.bytes())?
    );
    Ok(())
}
