//! Bootstrap intervals and permutation inference over borrowed samples.
use scioxide_core::SciOxideError;
use scioxide_random::{SeedableRng, SplitMix64};
use scioxide_stats::{bootstrap, jackknife, mean, permutation_test};

fn main() -> Result<(), SciOxideError> {
    let data = [1.0_f64, 2.0, 3.0, 9.0];
    let mut rng = SplitMix64::from_seed(550);
    let interval = bootstrap(mean, &data, &mut rng)?;
    let estimate = jackknife(mean, &data)?;
    let test = permutation_test(
        |x, y| Ok(mean(x)? - mean(y)?),
        &[1.0, 2.0],
        &[3.0, 4.0],
        &mut rng,
    )?;
    println!(
        "bootstrap={interval:?}; jackknife={estimate:?}; p={}",
        test.p_value()
    );
    Ok(())
}
