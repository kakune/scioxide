//! Real-component views of scalar slices.
//!
//! Packed complex products work on the real components (the 1m method), so they
//! read panels and output rows as consecutive real numbers: one per real scalar,
//! `re, im` per complex scalar.

use core::mem::{align_of, size_of};

use crate::NumericScalar;

/// Returns the number of real components per scalar: 1 for real and 2 for complex types.
///
/// ```
/// assert_eq!(scioxide_core::backend::component_count::<f64>(), 1);
/// assert_eq!(scioxide_core::backend::component_count::<scioxide_core::Complex<f32>>(), 2);
/// ```
pub const fn component_count<S: NumericScalar>() -> usize {
    if S::IS_REAL { 1 } else { 2 }
}

/// Layout proof shared by both views, evaluated at compile time per scalar.
const fn layout<S: NumericScalar>() -> usize {
    let count = component_count::<S>();
    assert!(
        size_of::<S>() == count * size_of::<S::Real>() && align_of::<S>() == align_of::<S::Real>()
    );
    count
}

/// Views a scalar slice as a slice of its real components, without copying.
///
/// Complex elements contribute `re, im` in that order. An empty slice gives an
/// empty view.
///
/// ```
/// use scioxide_core::{Complex, backend::components};
/// assert_eq!(components(&[Complex::new(1.0_f64, -2.0)]), &[1.0, -2.0]);
/// ```
#[allow(
    unsafe_code,
    reason = "sealed scalar layouts have exactly the initialized component bytes"
)]
pub fn components<S: NumericScalar>(values: &[S]) -> &[S::Real] {
    let count = const { layout::<S>() };
    // SAFETY: `NumericScalar` is sealed to f32, f64 (`S == S::Real`) and
    // `Complex<f32>`/`Complex<f64>`, whose `#[repr(C)]` layout is exactly
    // `re, im: S::Real`. The compile-time check proves size and alignment, so the
    // view covers the same initialized bytes, every component bit pattern is
    // valid, and the shared borrow keeps its lifetime.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<S::Real>(), values.len() * count) }
}

/// Views a mutable scalar slice as a mutable slice of its real components.
///
/// Any bits written to a component form a valid scalar.
///
/// ```
/// use scioxide_core::{Complex, backend::components_mut};
/// let mut values = [Complex::new(1.0_f32, 2.0)];
/// components_mut(&mut values)[1] = -3.0;
/// assert_eq!(values[0].im, -3.0);
/// ```
#[allow(
    unsafe_code,
    reason = "exclusive scalar borrows transfer to exactly the component bytes"
)]
pub fn components_mut<S: NumericScalar>(values: &mut [S]) -> &mut [S::Real] {
    let count = const { layout::<S>() };
    // SAFETY: as in `components`, the sealed representations are exactly
    // `count` consecutive `S::Real` with equal alignment, so the view covers the
    // same bytes; every real bit pattern is a valid component, and the
    // exclusive borrow is moved into the view without aliasing.
    unsafe {
        core::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<S::Real>(), values.len() * count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Complex;

    #[test]
    fn component_views_preserve_addresses_lengths_and_order() {
        let mut real = [1.0_f64, -2.0];
        assert_eq!(components(&real).as_ptr(), real.as_ptr());
        components_mut(&mut real)[1] = 3.0;
        assert_eq!(real, [1.0, 3.0]);
        let mut complex = [Complex::new(1.0_f32, 2.0), Complex::new(-3.0, 4.0)];
        assert_eq!(components(&complex), [1.0, 2.0, -3.0, 4.0]);
        components_mut(&mut complex)[3] = -0.5;
        assert_eq!(complex[1], Complex::new(-3.0, -0.5));
        let mut wide = [Complex::new(-0.0_f64, f64::from_bits(1))];
        assert_eq!(components(&wide)[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(components(&wide)[1].to_bits(), 1);
        components_mut(&mut wide)[1] = f64::NAN;
        assert!(wide[0].im.is_nan());
        let mut narrow = [1.0_f32];
        components_mut(&mut narrow)[0] = -0.0;
        assert_eq!(components(&narrow)[0].to_bits(), (-0.0_f32).to_bits());
        assert!(components::<Complex<f64>>(&[]).is_empty());
        assert_eq!(
            (component_count::<f32>(), component_count::<Complex<f64>>()),
            (1, 2)
        );
    }
}
