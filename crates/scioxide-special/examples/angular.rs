//! Integer/half-integer couplings and normalized sphere coordinates.
use scioxide_core::{ExecutionContext, SciOxideError};
use scioxide_special::{
    AngularMomentum, AngularOptions, AngularPlan, AngularState, clebsch_gordan, spherical_harmonic,
};
pub(crate) fn main() -> Result<(), SciOxideError> {
    // ⟨½ ½; ½ −½ | 0 0⟩ from the doubled quantum numbers 2j = [1, 1, 0], 2m = [1, −1, 0].
    let singlet = clebsch_gordan::<f64>([1, 1, 0], [1, -1, 0])?;
    println!("singlet coupling: {singlet}");
    // A plan and one reserved workspace evaluate many coefficients without allocating.
    let plan = AngularPlan::new(AngularMomentum::new(20)?, AngularOptions::default())?;
    let mut workspace = plan.workspace()?;
    let (up, down) = (AngularState::new(1, 1)?, AngularState::new(1, -1)?);
    for twice_total in [0, 2] {
        let total = AngularState::new(twice_total, 0)?;
        let coefficient = plan.clebsch_gordan_scoped::<f64>(
            up,
            down,
            total,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )?;
        println!(
            "⟨½ ½; ½ −½ | {} 0⟩ = {}",
            twice_total / 2,
            coefficient.try_value()?
        );
    }
    println!("Y_2^1: {:?}", spherical_harmonic(2, 1, 0.75_f64, 1.25)?);
    Ok(())
}
