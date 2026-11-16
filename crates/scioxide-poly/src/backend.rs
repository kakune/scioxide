//! Lower-level building blocks used by other SciOxide crates.
//!
//! Most users do not need this module. It holds the kernels behind the
//! polynomial types: divided differences, Newton and barycentric evaluation,
//! basis conversion and the Weierstrass correction. Its `_into` functions take
//! no [`ExecutionContext`](scioxide_core::ExecutionContext), and some may leave
//! their output changed on error; each function states what it guarantees.

/// Divided differences, barycentric weights, Newton evaluation and basis
/// conversion kernels.
pub mod polynomial_basis;

/// The Weierstrass correction used by simultaneous root iteration and root
/// inclusion disks.
pub mod polynomial_iteration;

/// Second-form barycentric evaluation of polynomial and rational interpolants,
/// and Chebyshev points with their barycentric weights.
pub mod barycentric;
