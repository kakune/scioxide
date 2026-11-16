//! Shared fallible storage growth.
use alloc::vec::Vec;
use scioxide_core::SciOxideError;
pub(crate) fn resize<T: Clone>(v: &mut Vec<T>, n: usize, value: T) -> Result<(), SciOxideError> {
    resize_with_quantity(v, n, value, scioxide_core::quantity!("optimizer_storage"))
}

pub(crate) fn resize_with_quantity<S: Clone>(
    values: &mut Vec<S>,
    len: usize,
    value: S,
    context: scioxide_core::Quantity,
) -> Result<(), SciOxideError> {
    if len > values.len() {
        scioxide_core::backend::try_reserve(values, len - values.len(), context)?;
    }
    values.resize(len, value);
    Ok(())
}
