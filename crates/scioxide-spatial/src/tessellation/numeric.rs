//! Approximate coordinate conversion and normalization.
use crate::tessellation::{Result, unstable};
use scioxide_core::{NonFiniteOrigin, RealScalar, SciOxideError};
// Narrow an approximate construction. Values outside the scalar's finite
// range are a numerical failure of that construction, not a caller conversion.
pub(super) fn convert<S: RealScalar>(value: f64) -> Result<S> {
    S::try_from_f64_lossy(value)
        .ok()
        .filter(|result| result.is_finite())
        .ok_or_else(|| unstable("unrepresentable approximate coordinate"))
}
// Scale by the largest magnitude before squaring, so the length can neither
// overflow nor underflow.
pub(super) fn normalize(values: &mut [f64]) -> Result<()> {
    let scale = values.iter().fold(0_f64, |s, x| s.max(x.abs()));
    if scale == 0. || !scale.is_finite() {
        return Err(unstable("degenerate normal"));
    }
    let length = values
        .iter()
        .map(|x| (x / scale) * (x / scale))
        .sum::<f64>()
        .sqrt();
    for x in values {
        *x = (*x / scale) / length;
    }
    Ok(())
}
// Validates query dimensions and finite coordinates.
pub(super) fn validate_query<S: RealScalar>(point: &[S], dimension: usize) -> Result<()> {
    if point.len() != dimension {
        return Err(SciOxideError::dimension_mismatch(
            [dimension],
            [point.len()],
        ));
    }
    if point.iter().any(|x| !x.is_finite()) {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "finite coordinates required",
        ));
    }
    Ok(())
}
