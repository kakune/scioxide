//! Plain transform storage accounting and byte-limit checks.
use scioxide_core::{Quantity, SciOxideError};

pub(super) fn admit_bytes(
    actual: usize,
    maximum: usize,
    quantity: Quantity,
) -> Result<(), SciOxideError> {
    if actual <= maximum {
        Ok(())
    } else {
        Err(SciOxideError::resource_limit(quantity, actual, maximum))
    }
}
/// An allocated Vec's byte capacity is representable by construction.
pub(crate) fn vector_bytes<T>(values: &alloc::vec::Vec<T>) -> usize {
    values.capacity() * core::mem::size_of::<T>()
}
