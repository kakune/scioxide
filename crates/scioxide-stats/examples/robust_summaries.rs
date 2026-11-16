//! Robust summaries, explicit rolling windows and information units.
use scioxide_core::{ExecutionContext, SciOxideError, StridedLayout};
use scioxide_stats::*;
fn main() -> Result<(), SciOxideError> {
    let values = [1.0_f64, 2.0, 3.0, 4.0, 100.0];
    let mut workspace = StatisticsWorkspace::default();
    workspace.try_reserve(&StatisticsWorkspace::scalar_requirements(
        values.len(),
        StatisticsOptions::new(),
    )?)?;
    let mad = workspace
        .evaluate(
            &values,
            Unweighted,
            Statistic::median_absolute_deviation(QuantileMethod::Linear, 1.0),
        )
        .and_then(StatisticValue::scalar)?;
    let trimmed = workspace
        .evaluate(
            &values,
            Unweighted,
            Statistic::trimmed_mean(0.2, TrimSide::Both),
        )
        .and_then(StatisticValue::scalar)?;
    println!("MAD={mad}, trimmed mean={trimmed}");
    let shape = [values.len()];
    let strides = [1];
    let view = BatchView::new(
        &values,
        StridedLayout::new(&shape, &strides)?,
        Axis::Index(0),
    )?;
    let window = RollingWindow::new(1, 1).with_boundary(RollingBoundary::Truncate);
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let mut windows = StatisticsWorkspace::default();
    windows.try_reserve(
        &(StatisticsWorkspace::workspace_requirements(view, Some(window), options, context)?),
    )?;
    let mut output = [StatisticValue::Location(0.0); 5];
    rolling_statistics_into(
        view,
        Statistic::quantile(
            QuantileProbability::new(0.5)?,
            QuantileRule::new(QuantileMethod::Linear),
        ),
        window,
        options,
        &mut output,
        &mut windows,
        context,
    )?;
    println!("rolling medians: {output:?}");
    println!(
        "fair coin information: {:?}",
        entropy_with_options(&[1.0, 1.0], None, EntropyOptions::new().with_base(2.0))?
    );
    Ok(())
}
