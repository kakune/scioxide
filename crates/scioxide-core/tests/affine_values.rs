//! Analytic and transactional coverage of the core affine value space.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    AffineValue, AffineValueMut, Complex, SciOxideError, affine_combination_into, write_value,
};

#[test]
fn scalar_projection_preserves_bits_and_checks_actual_shape() {
    let nan = f64::from_bits(0x7ff8_0000_0000_0042);
    for value in [-0.0_f64, f64::MIN_POSITIVE, f64::INFINITY, nan] {
        assert_eq!(value.try_to_scalar().unwrap().to_bits(), value.to_bits());
        assert_eq!([value].try_to_scalar().unwrap().to_bits(), value.to_bits());
    }
    assert_eq!(
        (-0.0_f32).try_to_scalar().unwrap().to_bits(),
        (-0.0_f32).to_bits()
    );
    for values in [vec![], vec![1.0_f64, 2.0]] {
        assert!(matches!(
            values.try_to_scalar().unwrap_err().view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
    }
    struct Misleading;
    impl AffineValue for Misleading {
        type Scale = f64;
        type Owned = Vec<f64>;
        const FIXED_DIMENSION: Option<usize> = Some(1);
        fn dimension(&self) -> Result<usize, SciOxideError> {
            Ok(2)
        }
        fn component(&self, index: usize) -> Result<f64, SciOxideError> {
            [1.0_f64, 2.0].component(index)
        }
        fn try_to_owned(&self) -> Result<Self::Owned, SciOxideError> {
            scioxide_core::backend::try_filled(2, 1.0, scioxide_core::quantity!("test_value"))
        }
    }
    assert!(matches!(
        Misleading.try_to_scalar().unwrap_err().view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
}

#[test]
fn both_precisions_and_complex_values_use_real_weights() {
    macro_rules! check {
        ($s:ty) => {{
            let left = Complex::new(2.0 as $s, -4.0 as $s);
            let right = Complex::new(6.0 as $s, 8.0 as $s);
            let mut output = Complex::new(99.0 as $s, 99.0 as $s);
            affine_combination_into(
                &[(0.75 as $s, &left), (0.25 as $s, &right)],
                &mut output,
                &mut [0.0 as $s; 2],
            )
            .unwrap();
            assert_eq!(output, Complex::new(3.0 as $s, -1.0 as $s));
            let mut real = 99.0 as $s;
            affine_combination_into(&[(0.25 as $s, &(4.0 as $s))], &mut real, &mut [0.0 as $s])
                .unwrap();
            assert_eq!(real, 1.0 as $s);
        }};
    }
    check!(f32);
    check!(f64);
}

#[test]
fn checked_shapes_and_finiteness_leave_borrowed_outputs_unchanged() {
    let mut output = [8.0_f64, 9.0];
    let mut scratch = [0.0; 2];
    let bad = [1.0];
    assert!(matches!(
        (affine_combination_into(&[(1.0, &bad)], &mut output, &mut scratch))
            .as_ref()
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(output, [8.0, 9.0]);
    for input in [[1.0, f64::NAN], [1.0, f64::INFINITY], [f64::MAX, f64::MAX]] {
        assert!(affine_combination_into(&[(2.0, &input)], &mut output, &mut scratch).is_err());
        assert_eq!(output, [8.0, 9.0]);
    }
}

#[test]
fn exact_product_accumulation_survives_overflow_and_cancellation() {
    let a = [f64::MAX, 1.0];
    let b = [-f64::MAX, 2.0];
    let mut output = [0.0; 2];
    affine_combination_into(&[(2.0, &a), (2.0, &b)], &mut output, &mut [0.0; 2]).unwrap();
    assert_eq!(output, [0.0, 6.0]);
    let tiny = f64::from_bits(1);
    affine_combination_into(
        &[(0.5, &[tiny]), (0.5, &[tiny])],
        &mut output[..1],
        &mut [0.0],
    )
    .unwrap();
    assert_eq!(output[0], tiny);
}

#[test]
fn fixed_dynamic_and_borrowed_complex_vectors_share_the_contract() {
    let values = [Complex::new(1.0_f64, -2.0), Complex::new(3.0, -4.0)];
    let mut dynamic = values.to_vec();
    write_value(&[5.0, 6.0, 7.0, 8.0], &mut dynamic[..]).unwrap();
    assert_eq!(dynamic[1], Complex::new(7.0, 8.0));
    assert_eq!(values.dimension().unwrap(), dynamic.dimension().unwrap());
    assert_eq!(values.component(3).unwrap(), -4.0);
    assert!(values.component(4).is_err());
    let mut empty: [f32; 0] = [];
    affine_combination_into(&[] as &[(f32, &[f32])], &mut empty, &mut []).unwrap();
}

// An application-owned value need not implement ordering, field arithmetic,
// square roots, or Copy. Its owned representation is also application-owned.
struct Position([f64; 2]);
impl AffineValue for Position {
    type Scale = f64;
    type Owned = Self;
    fn dimension(&self) -> Result<usize, SciOxideError> {
        Ok(2)
    }
    fn component(&self, index: usize) -> Result<f64, SciOxideError> {
        self.0.component(index)
    }
    fn try_to_owned(&self) -> Result<Self, SciOxideError> {
        Ok(Self(self.0))
    }
}
impl AffineValueMut for Position {
    fn for_each_component_mut(&mut self, visit: impl FnMut(&mut f64)) {
        self.0.for_each_component_mut(visit);
    }
}
#[test]
fn downstream_values_do_not_claim_coordinate_capabilities() {
    let a = Position([-3.0, 7.0]);
    let b = Position([1.0, -1.0]);
    let mut output = Position([0.0; 2]);
    affine_combination_into(&[(0.5, &a), (0.5, &b)], &mut output, &mut [0.0; 2]).unwrap();
    assert_eq!(output.0, [-1.0, 3.0]);
}

struct Huge;
impl AffineValue for Huge {
    type Scale = f64;
    type Owned = Vec<f64>;
    fn dimension(&self) -> Result<usize, SciOxideError> {
        Ok(usize::MAX)
    }
    fn component(&self, index: usize) -> Result<f64, SciOxideError> {
        if index == usize::MAX {
            return Err(SciOxideError::out_of_bounds(
                scioxide_core::quantity!("virtual_vector"),
                index,
                usize::MAX,
            ));
        }
        Ok(0.0)
    }
    fn try_to_owned(&self) -> Result<Self::Owned, SciOxideError> {
        scioxide_core::backend::try_filled(
            usize::MAX,
            0.0,
            scioxide_core::quantity!("virtual_vector"),
        )
    }
}
#[test]
fn composite_value_dimensions_are_checked_before_traversal() {
    let values = vec![Huge, Huge];
    assert!(matches!(
        (values.dimension()).as_ref().unwrap_err().view(),
        scioxide_core::SciOxideErrorView::SizeOverflow { .. }
    ));
}

#[test]
fn fixed_arrays_propagate_dimensions_and_preserve_nested_component_order() {
    assert_eq!(<[f64; 3] as AffineValue>::FIXED_DIMENSION, Some(3));
    assert_eq!(<[[f32; 3]; 2] as AffineValue>::FIXED_DIMENSION, Some(6));
    assert_eq!(<[[f64; 0]; 8] as AffineValue>::FIXED_DIMENSION, Some(0));
    let values = vec![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    assert_eq!(values.dimension().unwrap(), 6);
    for index in 0..6 {
        assert_eq!(values.component(index).unwrap(), (index + 1) as f64);
    }
    assert!(values.component(6).is_err());
    let mut out = [[0.0; 3]; 2];
    affine_combination_into(&[(0.5, &values)], &mut out, &mut [0.0; 6]).unwrap();
    assert_eq!(out, [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]);
    assert_eq!(<[f64; 0]>::default().dimension().unwrap(), 0);
}

#[test]
fn fixed_width_component_access_does_not_rescan_outer_elements() {
    use core::cell::Cell;
    #[derive(Clone, Copy)]
    struct Counted<'a>(&'a Cell<usize>, [f64; 3]);
    impl AffineValue for Counted<'_> {
        type Scale = f64;
        type Owned = Self;
        const FIXED_DIMENSION: Option<usize> = <[f64; 3]>::FIXED_DIMENSION;
        fn dimension(&self) -> Result<usize, SciOxideError> {
            self.0.set(self.0.get() + 1);
            self.1.dimension()
        }
        fn component(&self, index: usize) -> Result<f64, SciOxideError> {
            self.1.component(index)
        }
        fn try_to_owned(&self) -> Result<Self, SciOxideError> {
            Ok(*self)
        }
    }
    impl AffineValueMut for Counted<'_> {
        fn for_each_component_mut(&mut self, visit: impl FnMut(&mut f64)) {
            self.1.for_each_component_mut(visit);
        }
    }
    let calls = Cell::new(0);
    let values = vec![Counted(&calls, [1.0, 2.0, 3.0]); 128];
    scioxide_core::validate_value(&values).unwrap();
    let mut output = vec![0.0; 384];
    affine_combination_into(&[(2.0, &values)], &mut output, &mut [0.0; 384]).unwrap();
    assert_eq!(calls.get(), 0);
    assert!(output.chunks_exact(3).all(|v| v == [2.0, 4.0, 6.0]));
}
