//! A bounded simplicial search with explicit geometry and local work limits.
use scioxide_core::{Bounds, Budget};
use scioxide_optimize::{
    BorrowedStartDesign, BoundedObjective, ObjectiveFn, PowellLocalFactory, Shgo, ShgoOptions,
};
use scioxide_spatial::GeometryOptions;

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let bounds = Bounds::new(vec![(Some(-2.0_f64), Some(2.0)); 2])?;
    let options = ShgoOptions::new(Budget::try_from(5)?, GeometryOptions::new());
    let samples = [-1.0, -1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0];
    let search = Shgo::new(
        options,
        BorrowedStartDesign::new(&samples, 2)?,
        PowellLocalFactory::default(),
    );
    let objective = ObjectiveFn::new(2, |x: &[f64]| {
        Ok((x[0] * x[0] - 1.0).powi(2) + (x[1] * x[1] - 1.0).powi(2))
    });
    let result = search.minimize(&BoundedObjective::new(&objective, &bounds)?)?;
    if let Some((point, value)) = result.best_sample() {
        println!("best sampled point: {point:?}, value: {value}");
    }
    if let Some(local) = result.best_refined() {
        println!(
            "best successful local point: {:?}, value: {}",
            local.x(),
            local.fun()
        );
    }
    println!(
        "search status: {} (sampled landscape, no global certificate)",
        result.termination()
    );
    Ok(())
}
