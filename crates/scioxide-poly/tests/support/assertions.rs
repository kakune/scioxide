//! Shared absolute-error assertions for classical family identities.
pub(crate) fn assert_near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() < tolerance,
        "expected {actual} ≈ {expected} (diff={})",
        (actual - expected).abs()
    );
}
