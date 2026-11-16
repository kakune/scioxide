//! Types shared by the Hermitian and general eigenproblems.

use alloc::vec::Vec;
use scioxide_core::backend::try_reserve;
use scioxide_core::{Quantity, SciOxideError};

/// Order of the returned eigenvalues and their eigenvectors.
///
/// Real eigenvalues are sorted by value. Complex eigenvalues are sorted by real
/// part, and eigenvalues with equal real parts by decreasing imaginary part. For
/// a real matrix, each complex conjugate pair is therefore listed with the
/// positive imaginary part first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[must_use]
pub enum EigenvalueOrder {
    /// Increasing value or real part.
    ///
    /// For a complex matrix, rounding can give the two members of a nearly
    /// conjugate pair slightly different real parts, so they need not be
    /// adjacent.
    #[default]
    Ascending,
    /// Decreasing value or real part.
    Descending,
}

pub(super) fn reserve<T>(
    values: &mut Vec<T>,
    required: usize,
    quantity: Quantity,
) -> Result<(), SciOxideError> {
    if values.capacity() < required {
        try_reserve(values, required.saturating_sub(values.len()), quantity)?;
    }
    Ok(())
}

/// Iteration counts of a successful eigenvalue computation.
///
/// The counts of the computation itself and of its nested solves do not
/// overlap.
#[derive(Debug, Clone, Copy)]
pub struct EigenReport {
    pub(super) counts: scioxide_core::SolveCounts,
}
impl EigenReport {
    /// Returns the counts of this computation and of its nested solves.
    pub const fn counts(&self) -> scioxide_core::SolveCounts {
        self.counts
    }
    /// Returns the iterations of this computation; nested solves are in `counts().nested()`.
    pub const fn iterations(&self) -> usize {
        self.counts.own().iterations()
    }
}
