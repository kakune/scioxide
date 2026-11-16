//! Scaled tunnelling and a sequence of Coulomb partial waves.
use scioxide_special::{coulomb_wave_sequence, coulomb_wave_with_options, hydrogenic_radial};
pub(crate) fn main() -> Result<(), scioxide_core::SciOxideError> {
    let tunnel = coulomb_wave_with_options(
        0.0,
        300.0,
        1.0_f64,
        scioxide_special::CoulombOptions::default(),
    )?;
    assert!(tunnel.is_success());
    println!(
        "F scale: {:?}; G scale: {:?}",
        tunnel.f().scaled(),
        tunnel.g().scaled()
    );
    // [F, G, F′, G′] at the orders l = 0, …, 7 from one recurrence.
    let waves = coulomb_wave_sequence(0.0, 1.0, 2.0_f64, 8)?;
    let [f, g, _, _] = waves[7];
    println!("F_7(1, 2) = {f}; G_7(1, 2) = {g}");
    println!(
        "hydrogenic ground state at r=1: {}",
        hydrogenic_radial(1, 0, 1.0, 1.0_f64)?
    );
    Ok(())
}
