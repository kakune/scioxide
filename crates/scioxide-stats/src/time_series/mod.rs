//! Autocorrelation, periodogram diagnostics and Gaussian ARIMA models.
//!
//! Masked or skipped observations stay gaps on the original time grid.
use crate::error_helpers::{count as number, finite, invalid};
mod spectra;
mod summaries;
use scioxide_core::backend::try_filled;
use scioxide_core::{NumericScalar, RealScalar, SciOxideError};
pub use spectra::{
    PeriodogramBin, PeriodogramDiagnostics, PeriodogramWorkspace, periodogram_diagnostics_into,
};
pub use summaries::{
    LagNormalization, TimeSeriesLag, TimeSeriesOptions, TimeSeriesRequirements,
    TimeSeriesStatistic, TimeSeriesWorkspace, time_series_summaries_into, time_series_summary,
    time_series_summary_into, time_series_summary_with_options,
};
// Private helpers report.
fn add(a: usize, b: usize) -> Result<usize, SciOxideError> {
    scioxide_core::backend::checked_add(a, b, scioxide_core::quantity!("time_series_extent"))
}
fn mul(a: usize, b: usize) -> Result<usize, SciOxideError> {
    scioxide_core::backend::checked_mul(a, b, scioxide_core::quantity!("time_series_extent"))
}

mod arima;
pub use arima::{
    ArimaFitReport, ArimaForecast, ArimaInnovation, ArimaLikelihood, ArimaModel, ArimaOrder,
    ArimaTrend, ArimaWorkspace, PreparedArimaFit,
};
