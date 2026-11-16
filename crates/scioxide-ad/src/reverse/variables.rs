//! Recycled vectors of recording variables.
//!
//! The variables handed to a recorded function are a `Vec<Var<'r, V>>`, whose
//! lifetime changes with every recording, so a tape cannot store the vector itself.
//! It stores an empty `Vec<RawVar<V>>` of the same element size and alignment and
//! converts the allocation between the two types, so warm recordings allocate
//! nothing. This is the only `unsafe` code of the crate.

use alloc::vec::Vec;

use super::{TapeValue, Var};

/// A `Var` without its lifetime: same size and alignment, never read.
#[repr(C)]
pub(crate) struct RawVar<V> {
    _value: V,
    _slot: u32,
    _recorder: usize,
}

/// An empty vector whose capacity is kept between recordings.
pub(crate) struct Variables<V> {
    storage: Vec<RawVar<V>>,
}

impl<V> Default for Variables<V> {
    fn default() -> Self {
        Self {
            storage: Vec::new(),
        }
    }
}

impl<V: TapeValue> Variables<V> {
    /// Returns the retained allocation as an empty vector of variables.
    #[allow(
        unsafe_code,
        reason = "reuses an empty allocation for an element type of equal layout"
    )]
    pub(crate) fn take<'r>(&mut self) -> Vec<Var<'r, V>> {
        const {
            assert!(size_of::<RawVar<V>>() == size_of::<Var<'static, V>>());
            assert!(align_of::<RawVar<V>>() == align_of::<Var<'static, V>>());
        }
        let mut storage = core::mem::take(&mut self.storage);
        storage.clear();
        let (pointer, _, capacity) = storage.into_raw_parts();
        // SAFETY: the allocation comes from a `Vec<RawVar<V>>` with `capacity`
        // elements; `Var<'r, V>` has the same size and alignment (asserted above), so
        // the allocation layout is unchanged. The length is zero, so no element is
        // reinterpreted.
        unsafe { Vec::from_raw_parts(pointer.cast::<Var<'r, V>>(), 0, capacity) }
    }

    /// Keeps the allocation of `variables` for the next recording.
    #[allow(
        unsafe_code,
        reason = "reuses an empty allocation for an element type of equal layout"
    )]
    pub(crate) fn keep(&mut self, mut variables: Vec<Var<'_, V>>) {
        variables.clear();
        let (pointer, _, capacity) = variables.into_raw_parts();
        // SAFETY: as in `take`, in the other direction: same layout, zero length.
        self.storage = unsafe { Vec::from_raw_parts(pointer.cast::<RawVar<V>>(), 0, capacity) };
    }

    /// Reserves room for `count` variables.
    pub(crate) fn try_reserve(&mut self, count: usize) -> Result<(), scioxide_core::SciOxideError> {
        let additional = count.saturating_sub(self.storage.len());
        self.storage.try_reserve_exact(additional).map_err(|_| {
            scioxide_core::SciOxideError::allocation_failure(
                scioxide_core::quantity!("tape_variables"),
                count,
            )
        })
    }

    /// The retained capacity in variables.
    pub(crate) fn capacity(&self) -> usize {
        self.storage.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_survives_a_round_trip() {
        let mut variables = Variables::<f64>::default();
        variables.try_reserve(8).unwrap();
        let capacity = variables.capacity();
        let mut taken = variables.take();
        assert!(taken.capacity() >= 8);
        taken.push(Var::passive_value(1.0));
        variables.keep(taken);
        assert_eq!(variables.capacity(), capacity);
        let taken = variables.take();
        assert!(taken.is_empty());
        variables.keep(taken);
    }

    #[test]
    fn growth_while_taken_is_kept() {
        let mut variables = Variables::<crate::Lanes<f32, 4>>::default();
        let mut taken = variables.take();
        for _ in 0..100 {
            taken.push(Var::passive_value(crate::Lanes::splat(1.0)));
        }
        variables.keep(taken);
        assert!(variables.capacity() >= 100);
    }
}
