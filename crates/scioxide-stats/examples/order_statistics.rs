//! Explicit definitions and caller-owned borrowed-axis results.
use scioxide_core::{ExecutionContext, SciOxideError, StridedLayout};
use scioxide_stats::*;
fn main() -> Result<(), SciOxideError> {
    let values = [1.0_f64, 3.0, 2.0, 4.0, 6.0, 5.0];
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[2, 3], &[3, 1])?,
        Axis::Index(1),
    )?;
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let mut workspace = StatisticsWorkspace::default();
    workspace.try_reserve(&StatisticsWorkspace::workspace_requirements(
        view, None, options, context,
    )?)?;
    let mut output = [StatisticValue::Location(0.0); 2];
    statistics_into(
        view,
        Statistic::quantile(
            QuantileProbability::new(0.5)?,
            QuantileRule::new(QuantileMethod::MedianUnbiased),
        ),
        options,
        &mut output,
        &mut workspace,
        context,
    )?;
    println!("row medians: {output:?}");
    println!(
        "exact average ranks: {:?}",
        rank_slice(&values, RankMethod::Average)?
    );
    Ok(())
}
