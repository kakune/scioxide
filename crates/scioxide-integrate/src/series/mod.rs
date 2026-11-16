//! Summation of infinite series of scalar, complex or vector terms.
//!
//! A [`SeriesState`] stores the last `order + 1` terms and partial sums and
//! exact running sums, so its size does not depend on `max_terms`. Each step
//! computes one new estimate per component. The returned estimate is the newest
//! one unless its rounding floor already exceeds the error of the best earlier
//! estimate (see [`SeriesState`]). Levin and Wynn skip terms that cannot change
//! the rounded partial sum. Their error estimates are heuristic: they compare
//! successive estimates, correct for slow geometric contraction and never fall
//! below the rounding error of the transform inputs.

use crate::kernel::checked_callback_number;
mod double;
mod euler;
pub(crate) mod kernel;
mod options;
mod plan;
mod problem;
mod report;
mod runner;
mod state;
#[cfg(test)]
mod tests;
mod workspace;

pub use options::{SeriesMethod, SeriesOptions};
pub use plan::{SeriesSummation, sum_series};
pub use problem::{EulerMaclaurinProvider, SeriesProblem, SeriesTerms};
pub use report::{SeriesError, SeriesReport, SeriesResult, SeriesTermination};
pub use state::SeriesState;
pub use workspace::{SeriesRequirements, SeriesWorkspace};

use scioxide_core::{RealScalar, SciOxideError};

/// Largest prefetched batch of independent terms.
const TERM_BATCH: usize = 32;

/// Checks an error bound returned by an Euler–Maclaurin provider.
fn bound<R: RealScalar>(value: R) -> Result<R, SciOxideError> {
    if !value.is_finite() || value < R::ZERO {
        return Err(crate::kernel::invalid(
            scioxide_core::argument!("provider"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "series error bound must be finite and nonnegative",
        ));
    }
    Ok(value)
}
