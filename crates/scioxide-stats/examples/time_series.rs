//! Signal generation, finite-grid injected fitting, diagnostics and forecasts.
//! The reference grid illustrates injection; it is not a continuous optimizer.
use scioxide_core::*;
use scioxide_signal::*;
use scioxide_stats::*;
#[path = "../tests/support/fit_solver.rs"]
mod reference;
fn main() -> Result<(), SciOxideError> {
    let execution = ExecutionContext::sequential();
    let transfer = TransferFunction::new(IirDomain::Digital, &[1., 0.25], &[1., -0.5])?;
    let filter = IirFilter::from_transfer(&transfer, SignalExecutionOptions::new())?;
    // Supplied deterministic fixture innovations; no ambient random state.
    let noise: Vec<f64> = (0..96)
        .map(|i| (((i * 17 + 3) % 31) as f64 - 15.) / 8.)
        .collect();
    let mut data = vec![0.; noise.len()];
    let mut state = IirState::new(&filter, 1)?;
    let mut work = filter.workspace(IirRequest::Filter(SignalLayout::Contiguous(noise.len())))?;
    filter.filter_into(&noise, &mut data, &mut state, &mut work, execution)?;
    let samples = &data[32..];
    let summary = time_series_summary_with_options(
        samples,
        None,
        TimeSeriesStatistic::Acf,
        8,
        TimeSeriesOptions::new(),
        StatisticsOptions::new().with_max_bytes(65536),
    )?;
    println!(
        "Lag-one ACF: {} ({} pairs)",
        summary[1].value, summary[1].pairs
    );
    let problem = PreparedArimaFit::new(
        samples,
        None,
        ArimaOrder { p: 1, d: 0, q: 1 },
        ArimaTrend::None,
        StatisticsOptions::new().with_max_bytes(1_000_000),
    )?;
    let bounds = Bounds::new(vec![
        (Some(-0.75), Some(0.75)),
        (Some(-0.75), Some(0.75)),
        (Some(0.5), Some(2.5)),
    ])?;

    let fit = problem.fit(
        &[0.5, 0.25, 1.],
        &bounds,
        &mut reference::GridSolver { intervals: 4 },
    )?;
    println!(
        "Fit status: {:?}; NLL: {:?}",
        fit.report().termination(),
        fit.report().objective()
    );
    if let Some(model) = fit.into_model() {
        let mut workspace = model.workspace(8)?;
        let mut predictions = vec![
            ArimaForecast {
                mean: 0.,
                variance: 0.
            };
            8
        ];
        model.forecast_into(&mut predictions, &mut workspace, execution)?;
        println!("First forecast: {:?}", predictions[0]);
    }
    let options = SpectralEstimateOptions::new();
    let spectrum = SpectralEstimatePlan::build(&vec![1.; samples.len()], options, execution)?;
    let mut work =
        PeriodogramWorkspace::new(&spectrum, StatisticsOptions::new().with_max_bytes(65536))?;
    let mut bins = vec![PeriodogramBin::default(); spectrum.bins()];
    let report = periodogram_diagnostics_into(&spectrum, samples, &mut bins, &mut work, execution)?;
    println!("Centered periodogram power: {}", report.integrated_power);
    Ok(())
}
