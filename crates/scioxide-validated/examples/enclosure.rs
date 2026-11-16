//! A partial-domain calculation retains its enclosure and weakens its decoration.
use scioxide_core::SciOxideError;
use scioxide_validated::DecoratedInterval;

fn main() -> Result<(), SciOxideError> {
    let numerator = DecoratedInterval::new(1.0_f64, 2.0)?;
    let denominator = DecoratedInterval::new(0.0, 4.0)?;
    let result = numerator.div(&denominator)?;
    println!(
        "quotient: {:?}; decoration: {:?}",
        result.set(),
        result.decoration()
    );
    Ok(())
}
