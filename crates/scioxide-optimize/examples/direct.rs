//! Compare standard DIRECT and its locally biased variant on a bounded double well.
use scioxide_core::{Bounds, Budget, ExecutionContext};
use scioxide_optimize::{
    BoundedObjective, Direct, DirectOptions, DirectResult, DirectVariant, ObjectiveFn,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bounds = Bounds::new(vec![(Some(-2.0_f64), Some(2.0)); 2])?;
    let options = DirectOptions::new(
        Budget::try_from(300)?,
        Budget::try_from(601)?,
        Budget::try_from(10_000_000)?,
    );
    let objective = ObjectiveFn::new(2, |x: &[f64]| {
        let a = x[0] * x[0] - 1.0;
        let b = x[1] * x[1] - 1.0;
        Ok(a * a + b * b)
    });
    let problem = BoundedObjective::new(&objective, &bounds)?;
    for variant in [DirectVariant::Original, DirectVariant::LocallyBiased] {
        let solver = Direct::new(options.clone().with_variant(variant))?;
        let mut work = solver.workspace(2)?;
        let mut result = DirectResult::empty();
        solver.minimize_into(
            &problem,
            &mut result,
            &mut work,
            ExecutionContext::sequential(),
        )?;
        let (point, value) = result
            .best_sample()
            .ok_or("DIRECT returned no sampled point")?;
        println!(
            "{variant:?}: x={point:?}, f={value}, evaluations={}, stop={}",
            result.evaluations(),
            result.termination()
        );
    }
    Ok(())
}
