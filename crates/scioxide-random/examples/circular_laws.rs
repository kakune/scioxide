//! Angular and sphere-surface sampling with an injected generator.
use scioxide_random::{Distribution, SeedableRng, SplitMix64, VonMises, VonMisesFisher};
fn main() -> Result<(), scioxide_core::SciOxideError> {
    let mut rng = SplitMix64::from_seed(541);
    let angle = VonMises::new(0.0_f64, 10.0)?.sample(&mut rng)?;
    let direction = VonMisesFisher::new(&[1.0_f64, 0.0, 0.0], 10.0)?.sample(&mut rng)?;
    println!("angle={angle}; direction={direction:?}");
    Ok(())
}
