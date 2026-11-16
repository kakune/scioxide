//! Stable identities used by the shared error contract.
//!
//! Errors carry no identity of the function that reported them, so the crate
//! exports no `Operation` type and no `operation!` macro:
//!
//! ```compile_fail
//! use scioxide_core::Operation;
//! ```
//! ```compile_fail
//! let _ = scioxide_core::operation! { "a::b" };
//! ```
use core::fmt;

/// Closed source/destination representation for checked conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConversionType {
    /// IEEE binary32.
    F32,
    /// IEEE binary64.
    F64,
    /// Complex binary32.
    Complex32,
    /// Complex binary64.
    Complex64,
    /// Signed sixty-four-bit integer.
    I64,
    /// Signed pointer-sized integer.
    Isize,
    /// Unsigned one-hundred-twenty-eight-bit integer.
    U128,
    /// Unsigned pointer-sized integer.
    Usize,
}
impl From<crate::backend::ScalarKind> for ConversionType {
    fn from(kind: crate::backend::ScalarKind) -> Self {
        match kind {
            crate::backend::ScalarKind::F32 => Self::F32,
            crate::backend::ScalarKind::F64 => Self::F64,
            crate::backend::ScalarKind::Complex32 => Self::Complex32,
            crate::backend::ScalarKind::Complex64 => Self::Complex64,
        }
    }
}
impl fmt::Display for ConversionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Complex32 => "complex32",
            Self::Complex64 => "complex64",
            Self::I64 => "i64",
            Self::Isize => "isize",
            Self::U128 => "u128",
            Self::Usize => "usize",
        })
    }
}
