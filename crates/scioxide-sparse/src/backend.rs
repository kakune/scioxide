//! Lower-level kernels shared with other SciOxide crates.
//!
//! Most users do not need this module. It holds the arithmetic on 2×2
//! Hermitian pivot blocks that the sparse and dense `LDLᴴ` factorizations use.
//! It follows the same versioning rules as the rest of the public API.

/// Determinants, eigenvalues and solves of 2×2 Hermitian blocks
/// `[a conj(b); b c]`, with rescaling to avoid overflow and underflow.
pub mod hermitian;
