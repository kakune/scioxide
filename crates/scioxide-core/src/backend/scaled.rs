//! Common-scale factoring and tolerance tests for values with an extended exponent range.
//!
//! The value types themselves are [`crate::Scaled`], [`crate::ScaledValue`] and
//! [`crate::ScaledAccumulator`].
pub use crate::scaled::{
    factor_common_scale, scaled_magnitude_le_tolerance, scaled_tolerance_threshold,
};
