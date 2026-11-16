//! Canonical heavy-tail constructions and direct logarithmic tails.
use scioxide_core::SciOxideError;
use scioxide_random::{
    Affine, Burr12, Distribution, GeneralizedPareto, InverseSurvival, LevyStable, LogSurvival,
    Pareto, SeedableRng, SplitMix64,
};

fn main() -> Result<(), SciOxideError> {
    let burr = Burr12::new(2.0_f64, 3.0)?;
    let lomax = Affine::new(Pareto::new(3.0_f64)?, -1.0, 1.0)?;
    let bounded = GeneralizedPareto::new(-0.5_f64)?;
    let landau = LevyStable::new(1.0_f64, 1.0)?;
    let mut rng = SplitMix64::from_seed(537);
    println!("Burr upper log tail: {}", burr.log_sf(100.0)?);
    println!("Lomax upper quantile: {}", lomax.isf(1e-6)?);
    println!("bounded Pareto endpoint: {}", bounded.isf(0.0)?);
    println!("Landau sample: {}", landau.sample(&mut rng)?);
    Ok(())
}
