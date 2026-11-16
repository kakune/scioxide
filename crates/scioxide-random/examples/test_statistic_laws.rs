//! Direct extreme tails and reusable finite-KS scratch.
use scioxide_core::SciOxideError;
use scioxide_random::{
    FisherF, InverseSurvival, KolmogorovSmirnovTwoSided, LogSurvival, NoncentralStudentT,
    StudentizedRange, Survival,
};
fn main() -> Result<(), SciOxideError> {
    println!(
        "F critical value: {}",
        FisherF::with_options(
            5.0_f64,
            11.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )?
        .isf(0.05)?
    );
    println!(
        "Noncentral t log tail: {}",
        NoncentralStudentT::with_options(
            5.0_f64,
            2.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )?
        .log_sf(20.0)?
    );
    println!(
        "Three-group range survival: {}",
        StudentizedRange::with_options(
            3.0_f64,
            10.0,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )?
        .sf(3.0)?
    );
    let law = KolmogorovSmirnovTwoSided::<f64>::with_options(
        20,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )?;
    let mut workspace = law.workspace(0.15)?;
    println!(
        "KS tail: {:?}",
        law.evaluate_tail(0.15, true, &mut workspace)?
    );
    Ok(())
}
