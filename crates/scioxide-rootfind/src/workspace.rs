//! Checked byte counts and limits of the root-finding workspaces.
use alloc::vec::Vec;
use scioxide_core::{
    SciOxideError,
    backend::{checked_add, checked_mul},
};
pub(crate) fn sum(
    parts: impl IntoIterator<Item = Result<usize, SciOxideError>>,
) -> Result<usize, SciOxideError> {
    parts.into_iter().try_fold(0, |total, part| {
        checked_add(total, part?, scioxide_core::quantity!("workspace_bytes"))
    })
}
pub(crate) fn buffer<T>(values: &Vec<T>) -> Result<usize, SciOxideError> {
    checked_mul(
        values.capacity(),
        core::mem::size_of::<T>(),
        scioxide_core::quantity!("workspace_bytes"),
    )
}
pub(crate) fn limit(bytes: usize, maximum: usize) -> Result<(), SciOxideError> {
    if bytes > maximum {
        Err(SciOxideError::resource_limit(
            scioxide_core::quantity!("workspace_bytes"),
            bytes,
            maximum,
        ))
    } else {
        Ok(())
    }
}
