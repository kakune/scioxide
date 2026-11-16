//! Workspace capacity checks. Each plan family defines its buffers and lengths;
//! this module computes their sizes in bytes and checks them against the limit.
//! Reservation can fail after some buffers have grown, and those buffers are
//! kept. Sizes count heap arrays, including nested ones, but not stack objects,
//! allocator overhead or temporary allocations made during preparation.

use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_reserve};
use scioxide_core::{Capacity, Dimensions, Quantity, SciOxideError};

pub(super) const BYTES: Quantity = scioxide_core::quantity!("workspace_bytes");

/// Counts the larger of each buffer's capacity and required length, summing
/// equal-sized slots before converting them to bytes.
#[inline]
pub(super) fn buffer_bytes<T>(
    buffers: impl IntoIterator<Item = (usize, usize)>,
    quantity: Quantity,
) -> Result<usize, SciOxideError> {
    let slots = buffers
        .into_iter()
        .try_fold(0, |sum, (capacity, required)| {
            checked_add(sum, capacity.max(required), quantity)
        })?;
    checked_mul(slots, core::mem::size_of::<T>(), quantity)
}

#[inline]
pub(super) fn sum_bytes(
    parts: impl IntoIterator<Item = usize>,
    quantity: Quantity,
) -> Result<usize, SciOxideError> {
    parts
        .into_iter()
        .try_fold(0, |sum, part| checked_add(sum, part, quantity))
}

#[inline]
pub(super) fn try_sum_bytes(
    parts: impl IntoIterator<Item = Result<usize, SciOxideError>>,
    quantity: Quantity,
) -> Result<usize, SciOxideError> {
    parts
        .into_iter()
        .try_fold(0, |sum, part| checked_add(sum, part?, quantity))
}

#[inline]
pub(super) fn memory_limit(bytes: usize, limit: usize) -> Result<(), SciOxideError> {
    if bytes > limit {
        Err(SciOxideError::resource_limit(BYTES, bytes, limit))
    } else {
        Ok(())
    }
}

/// Test the limit before asking the family whether every individual buffer fits.
#[inline]
pub(super) fn check(
    retained: usize,
    limit: usize,
    sufficient: impl FnOnce() -> Result<bool, SciOxideError>,
) -> Result<Capacity, SciOxideError> {
    memory_limit(retained, limit)?;
    Ok(if sufficient()? {
        Capacity::Sufficient
    } else {
        Capacity::NeedsGrowth
    })
}

/// A sufficient workspace is configured by its family without growth. Compute
/// the mismatch diagnostic only on rejection.
#[inline]
pub(super) fn prepare(
    capacity: Capacity,
    required: usize,
    retained: impl FnOnce() -> Result<usize, SciOxideError>,
) -> Result<(), SciOxideError> {
    match capacity {
        Capacity::Sufficient => Ok(()),
        Capacity::NeedsGrowth => Err(SciOxideError::dimension_mismatch(
            Dimensions::one(required),
            Dimensions::one(retained()?),
        )),
    }
}

/// Returns the original growth error. Buffers grown before the failure are kept
/// and counted by later checks and reservations.
#[inline]
pub(super) fn reserve<B>(
    buffers: &mut B,
    projected: usize,
    limit: usize,
    grow: impl FnOnce(&mut B) -> Result<(), SciOxideError>,
    retained: impl FnOnce(&B) -> Result<usize, SciOxideError>,
) -> Result<(), SciOxideError> {
    memory_limit(projected, limit)?;
    grow(buffers)?;
    memory_limit(retained(buffers)?, limit)
}

#[inline]
pub(super) fn reserve_buffer<T: Clone>(
    buffer: &mut Vec<T>,
    len: usize,
    value: T,
    quantity: Quantity,
) -> Result<(), SciOxideError> {
    try_reserve(buffer, len.saturating_sub(buffer.len()), quantity)?;
    buffer.resize(len, value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scioxide_core::{SciOxideErrorKind, SciOxideErrorView};

    #[test]
    fn payloads_charge_each_buffer_and_detect_both_overflows() {
        assert_eq!(buffer_bytes::<f64>([(8, 1), (1, 8)], BYTES).unwrap(), 128);
        assert_eq!(buffer_bytes::<f32>([(0, 0)], BYTES).unwrap(), 0);
        for error in [
            buffer_bytes::<f64>([(usize::MAX, 0)], BYTES).unwrap_err(),
            buffer_bytes::<u8>([(usize::MAX, 0), (0, 1)], BYTES).unwrap_err(),
            sum_bytes([usize::MAX, 1], BYTES).unwrap_err(),
        ] {
            assert!(
                matches!(error.view(), SciOxideErrorView::SizeOverflow {  context }
                if context == BYTES)
            );
        }
        let quantity = scioxide_core::quantity!("hessian_samples");
        let error =
            buffer_bytes::<scioxide_core::Complex64>([(0, usize::MAX)], quantity).unwrap_err();
        assert!(
            matches!(error.view(), SciOxideErrorView::SizeOverflow { context, .. } if context == quantity)
        );
    }

    #[test]
    fn limits_precede_growth_and_charge_actual_capacity_after_growth() {
        let mut retained = 8;
        let error = reserve(
            &mut retained,
            16,
            15,
            |_| unreachable!(),
            |_| unreachable!(),
        )
        .unwrap_err();
        assert!(
            matches!(error.view(), SciOxideErrorView::ResourceLimit {  resource, required: 16, limit: 15 }
            if resource == BYTES)
        );
        assert_eq!(retained, 8);
        // Model allocator capacity above the projection without relying on a
        // particular allocator's rounding or introducing another allocator.
        let error = reserve(
            &mut retained,
            16,
            16,
            |bytes| {
                *bytes = 24;
                Ok(())
            },
            |bytes| Ok(*bytes),
        )
        .unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::ResourceLimit {
                required: 24,
                limit: 16,
                ..
            }
        ));
        assert_eq!(retained, 24);
        assert_eq!(
            check(retained, 23, || unreachable!()).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        reserve(&mut retained, 24, 24, |_| Ok(()), |bytes| Ok(*bytes)).unwrap();
        assert_eq!(
            check(retained, 24, || Ok(true)).unwrap(),
            Capacity::Sufficient
        );
    }

    #[test]
    fn preparation_rejects_a_missing_buffer_even_with_spare_bytes() {
        let capacity = check(128, 128, || Ok(false)).unwrap();
        let error = prepare(capacity, 64, || Ok(128)).unwrap_err();
        assert!(
            matches!(error.view(), SciOxideErrorView::DimensionMismatch {  expected, actual }
            if expected == [64] && actual == [128])
        );
        prepare(Capacity::Sufficient, 64, || unreachable!()).unwrap();
    }

    #[test]
    fn growth_failure_preserves_its_diagnostic_and_retained_scratch() {
        let mut retained = 8;
        let quantity = scioxide_core::quantity!("fixed_workspace");
        let error = reserve(
            &mut retained,
            16,
            32,
            |bytes| {
                *bytes = 24;
                Err(SciOxideError::allocation_failure(quantity, 2))
            },
            |_| unreachable!(),
        )
        .unwrap_err();
        assert!(
            matches!(error.view(), SciOxideErrorView::AllocationFailure {  context, requested: 2 }
            if context == quantity)
        );
        assert_eq!(retained, 24);
        assert_eq!(
            check(retained, 32, || Ok(false)).unwrap(),
            Capacity::NeedsGrowth
        );
    }
}
