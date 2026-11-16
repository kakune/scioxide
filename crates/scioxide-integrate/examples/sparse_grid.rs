//! Bounded sparse-grid integration with a caller-owned evaluation cache.

use scioxide_core::{Budget, ErrorTolerance, Interval, NormSelection, SciOxideError, Tolerance};
use scioxide_integrate::{SparseGridOptions, SparseGridPlan};

fn main() -> Result<(), SciOxideError> {
    let tolerance = ErrorTolerance::from_tolerance(
        Tolerance::<f64>::from_decimal_places(8, 8),
        NormSelection::Infinity,
    );
    let options = SparseGridOptions::default()
        .with_tolerance(tolerance)
        .with_max_level(3)
        .with_max_indices(Budget::try_from(32)?)
        .with_max_evaluations(Budget::try_from(128)?);
    let plan = SparseGridPlan::new(2, options)?;
    let mut workspace = plan.workspace(1)?;
    let bounds = [Interval::new(0.0, 1.0)?, Interval::new(0.0, 1.0)?];
    let function = |x: &[f64]| Ok(x[0] * x[0] + x[1] * x[1]);
    let result = (|| {
        let mut value = bounds
            .first()
            .map(|b| b.lower())
            .unwrap_or_else(|| <_ as scioxide_core::AlgebraicScalar>::ZERO);
        let report = plan.integrate_into(
            &function,
            &bounds,
            &mut value,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native(),
        )?;
        Ok::<_, scioxide_core::SciOxideError>((value, report))
    })()?;
    println!(
        "estimate={:?}, frontier_error={:?}, status={:?}, evaluations={}, directions={:?}",
        result.0,
        result.1.error_estimate(),
        result.1.termination(),
        result.1.evaluations(),
        workspace.directional_indicators(),
    );
    Ok::<(), SciOxideError>(())
}
