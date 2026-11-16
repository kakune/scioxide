//! Variable finite domains, explicit maps and localized-feature partitions.
use scioxide_core::{Complex, ExecutionContext, Interval, SciOxideError};
use scioxide_integrate::{CubaturePlan, IteratedDomainPlan, IteratedOptions, SubdivisionOptions};
fn main() -> Result<(), SciOxideError> {
    let options = SubdivisionOptions::<f64>::default();
    let tetrahedron = IteratedDomainPlan::new(3, IteratedOptions::new(options.clone()))?;
    let result = tetrahedron.integrate(
        |_: &[f64]| Ok(1.0),
        |x: &[f64]| Interval::new(0.0, if x.is_empty() { 1.0 } else { x[x.len() - 1] }),
    )?;
    assert!(result.1.is_success());
    println!("Variable tetrahedron: {:?}", result);
    let triangle = IteratedDomainPlan::new(2, IteratedOptions::new(options.clone()))?;
    let mut ws = triangle.workspace(2)?;
    let mut output = Complex::new(0.0, 0.0);
    let report = triangle.integrate_into(
        |x: &[f64]| Ok(Complex::new(x[0], x[1])),
        |x: &[f64]| Interval::new(0.0, if x.is_empty() { 1.0 } else { x[0] }),
        &|x: &[f64], y: &mut [f64]| {
            y[0] = 2.0 * x[0];
            y[1] = 3.0 * x[1];
            Ok(6.0)
        },
        &mut output,
        &mut ws,
        ExecutionContext::sequential_native(),
    )?;
    assert!(report.is_success());
    println!("Mapped triangle: {output:?}");
    let rectangle = CubaturePlan::new(&[Interval::new(0.0, 1.0)?; 2], options)?
        .with_breakpoints(0, &[0.12335, 0.12345, 0.12355])?;
    let mut ws = rectangle.workspace(1)?;
    let mut value = 0.0;
    let report = rectangle.integrate_into(
        |x: &[f64]| {
            let t = (x[0] - 0.12345) / 0.0001;
            Ok(if t.abs() < 1.0 {
                (1.0 - t * t).powi(2)
            } else {
                0.0
            })
        },
        &mut value,
        &mut ws,
        ExecutionContext::sequential_native(),
    )?;
    assert!(report.is_success());
    println!("Hinted compact peak: {value}");
    Ok(())
}
