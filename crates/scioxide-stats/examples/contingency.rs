//! Independence tests and false-discovery adjustment.
use scioxide_stats::{
    ContingencyView, FalseDiscoveryMethod, PValueAdjustment, adjust_pvalues, chi2_contingency_test,
    fisher_exact_test,
};

fn main() -> Result<(), scioxide_core::SciOxideError> {
    let table = ContingencyView::new(&[10, 20, 20, 10], 2, 2)?;
    let approximate = chi2_contingency_test(table)?;
    let conditional = fisher_exact_test(table)?;
    let adjusted = adjust_pvalues(
        &[conditional.p_value(), approximate.p_value()],
        PValueAdjustment::FalseDiscovery(FalseDiscoveryMethod::BenjaminiHochberg),
    )?;
    println!(
        "Fisher p={}; adjusted p={adjusted:?}",
        conditional.p_value()
    );
    Ok(())
}
