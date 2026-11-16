//! Runnable three-layer example from the workspace README.
use scioxide_core::*;
use scioxide_differentiate::*;
use scioxide_sparse::derivative_coloring;

fn main() -> Result<(), SciOxideError> {
    let x = [1.0_f64, 2.0];
    let f = |p: &[f64]| Ok::<_, SciOxideError>(p[0] * p[0] + p[1] * p[1]);
    let sample = |p: &[f64], y: &mut [f64]| {
        y[0] = f(p)?;
        Ok(())
    };
    // One-shot: central differences with the default step.
    let simple = gradient(f, &x)?;
    assert!((simple[0] - 2.0).abs() < 1e-8);
    // Method value: the settings of every call.
    let method = FiniteDifference::default()
        .with_step(DifferenceStep::absolute_coordinates(&[0.125, 0.25])?);
    let configured = method.gradient(sample, &x, None, None)?.into_values()?;
    // Workspace: repeated calls without allocation.
    let request = DerivativeRequest::new(&method, DerivativeShape::gradient(x.len()));
    let mut workspace = FiniteDifferenceWorkspace::workspace(request)?;
    let mut buffer = [0.0; 2];
    let context = ExecutionContext::sequential();
    method
        .gradient_into(sample, &x, None, None, &mut buffer, &mut workspace, context)?
        .require_success()?;
    assert_eq!(configured, buffer);
    let adaptive = AdaptiveDifference::default().gradient(sample, &x, None)?;
    assert!(adaptive.report().is_success());
    let offsets = [-0.5, -0.25, 0.0, 0.25, 0.5];
    let axes = [offsets.as_slice(); 2];
    let a = offsets.map(|o| x[0] + o);
    let b = offsets.map(|o| x[1] + o);
    let options = NodeDifferenceOptions::default();
    let operation = DerivativeOperation::Gradient;
    let physical = PhysicalDifferencePlan::new(operation, &x, &[&a, &b], 1, options)?;
    assert!(physical.evaluate(sample, None)?.report().output_available());
    let relative = RelativeDifference::new(DerivativeOperation::Hessian, &axes, 1, options)?;
    assert!(
        relative
            .evaluate(sample, &x, None)?
            .report()
            .output_available()
    );
    let extension = |p: &[Complex64], y: &mut [Complex64]| {
        y[0] = p[0] * p[0] + p[1] * p[1];
        Ok(())
    };
    let complex = ComplexStep::default().gradient(extension, &x)?;
    assert_eq!(complex.into_values()?, [2.0, 4.0]);
    let pattern = StructuralPatternView::new(2, 2, &[0, 1, 2], &[0, 1])?;
    let schedule = derivative_coloring(pattern)?;
    schedule.validate_pattern(pattern)?;
    let diagonal = |p: &[f64], y: &mut [f64]| {
        y[0] = p[0] * p[0];
        y[1] = p[1] * p[1];
        Ok(())
    };
    assert_eq!(sparse_jacobian(diagonal, &x, &schedule)?, [2.0, 4.0]);
    Ok(())
}
