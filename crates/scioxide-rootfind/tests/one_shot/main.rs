//! One-shot forms: equality with their configured forms, reference values,
//! errors and edge cases.
#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]

mod fixed_point;
mod polynomial;
mod scalar;
mod special_zeros;
mod system;
