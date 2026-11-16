//! Scalar radiation/transport functions with reusable work budgets.
use scioxide_core::{Budget, SciOxideError};
use scioxide_special::{
    ClausenOptions, SynchrotronOptions, TransportOptions, clausen_with_options,
    synchrotron_1_scaled, synchrotron_1_with_options, transport_5_with_options,
};

pub(crate) fn main() -> Result<(), SciOxideError> {
    let budget = Budget::try_from(128)?;
    let radiation = synchrotron_1_with_options(2., SynchrotronOptions::new(budget))?;
    let thermal = transport_5_with_options(4., TransportOptions::new(budget))?;
    let angle = clausen_with_options(core::f64::consts::FRAC_PI_2, ClausenOptions::new(budget))?;
    for (name, result) in [
        ("synchrotron_1", radiation),
        ("transport_5", thermal),
        ("clausen", angle),
    ] {
        println!(
            "{name}: value={}, method={}, work={}, termination={:?}",
            result.try_value()?,
            result.algorithm().method(),
            result.counts().iterations(),
            result.termination()
        );
    }
    println!(
        "scaled synchrotron tail at 1000: {}",
        synchrotron_1_scaled(1000.)?
    );
    Ok(())
}
