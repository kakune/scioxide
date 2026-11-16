use scioxide_core::{Bounds, SciOxideError};
use scioxide_optimize::{BoundedObjective, LbfgsB, ObjectiveFunction};

struct Quadratic;
impl ObjectiveFunction<f64> for Quadratic {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok((x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2))
    }
    fn gradient_into(&self, x: &[f64], gradient: &mut [f64]) -> Result<bool, SciOxideError> {
        gradient[0] = 2.0 * (x[0] - 2.0);
        gradient[1] = 2.0 * (x[1] + 1.0);
        Ok(true)
    }
}

fn main() -> Result<(), SciOxideError> {
    let bounds = Bounds::new(vec![(Some(0.0), Some(1.0)), (None, Some(0.0))])?;
    let problem = BoundedObjective::new(&Quadratic, &bounds)?;
    let result = LbfgsB::default().minimize(&problem, &[0.25, 0.0])?;
    assert!(result.is_success());
    assert!((result.x()[0] - 1.0).abs() < 1e-7);
    assert!((result.x()[1] + 1.0).abs() < 1e-7);
    println!(
        "x={:?}, projected stationarity={:?}",
        result.x(),
        result.stationarity()
    );
    Ok(())
}
