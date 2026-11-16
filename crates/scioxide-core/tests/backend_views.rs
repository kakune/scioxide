//! External consumers exercise typed representation and allocation contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::backend::{
    ComplexSlice, ComplexSliceMut, RealBackend, Scalar, ScalarKind, ScalarValue,
};
use scioxide_core::{Complex32, Complex64, ScalarConversionReason, SciOxideErrorView};

macro_rules! storage {
    ($name:ident, $ty:ty, $value:expr, $next:expr) => {
        #[test]
        fn $name() {
            {
                let mut values: [$ty; 2] = [$value, $value];
                let address = values.as_ptr();
                let original: &[$ty] = <$ty>::slice(&values).try_into().unwrap();
                assert_eq!(original.as_ptr(), address);
                assert_eq!(original.len(), 2);
                let exclusive: &mut [$ty] = <$ty>::slice_mut(&mut values).try_into().unwrap();
                assert_eq!(exclusive.as_ptr(), address);
                exclusive[1] = $next;
                assert_eq!(values[1], $next);
                let empty: &[$ty] = <$ty>::slice(&[]).try_into().unwrap();
                assert!(empty.is_empty());
                let empty: &mut [$ty] = <$ty>::slice_mut(&mut []).try_into().unwrap();
                assert!(empty.is_empty());
                assert_eq!(<$ty>::from_value(Scalar::value($value)).unwrap(), $value);
            };
        }
    };
}
storage!(real32_storage, f32, -0.0_f32, 3.0_f32);
storage!(real64_storage, f64, -0.0_f64, 3.0_f64);
storage!(
    complex32_storage,
    Complex32,
    Complex32::new(1.0, -0.0),
    Complex32::new(2.0, 3.0)
);
storage!(
    complex64_storage,
    Complex64,
    Complex64::new(1.0, -0.0),
    Complex64::new(2.0, 3.0)
);

#[test]
fn unsupported_conversions_are_typed_and_execute() {
    {
        for value in [
            ScalarValue::F32(1.0),
            ScalarValue::F64(1.0),
            ScalarValue::Complex32(Complex32::new(1.0, 0.0)),
            ScalarValue::Complex64(Complex64::new(1.0, 0.0)),
        ] {
            for (kind, result) in [
                (ScalarKind::F32, f32::from_value(value).map(|_| ())),
                (ScalarKind::F64, f64::from_value(value).map(|_| ())),
                (
                    ScalarKind::Complex32,
                    Complex32::from_value(value).map(|_| ()),
                ),
                (
                    ScalarKind::Complex64,
                    Complex64::from_value(value).map(|_| ()),
                ),
            ] {
                if kind == value.kind() {
                    assert!(result.is_ok());
                } else {
                    assert!(
                        matches!(result.unwrap_err().view(), SciOxideErrorView::ScalarConversion { reason: ScalarConversionReason::Unsupported, from_type, to_type } if from_type == value.kind().into() && to_type == kind.into())
                    );
                }
            }
        }
        assert!(<&[f64]>::try_from(f32::slice(&[])).is_err());
        assert!(<&mut [Complex32]>::try_from(f64::slice_mut(&mut [])).is_err());
    };
}

#[test]
fn real_component_views_and_sampling_preserve_exact_bits() {
    {
        let mut single = [Complex32::new(-0.0, f32::from_bits(1))];
        let address = single.as_ptr();
        let ComplexSliceMut::F32(view) = f32::complex_slice_mut(&mut single) else {
            unreachable!()
        };
        assert_eq!(view.as_ptr(), address);
        view[0].im = 2.0;
        let ComplexSlice::F32(view) = f32::complex_slice(&single) else {
            unreachable!()
        };
        assert_eq!(view[0].re.to_bits(), (-0.0_f32).to_bits());
        assert_eq!(view[0].im, 2.0);
        let mut double = [Complex64::new(-0.0, f64::from_bits(1))];
        let address = double.as_ptr();
        let ComplexSliceMut::F64(view) = f64::complex_slice_mut(&mut double) else {
            unreachable!()
        };
        assert_eq!(view.as_ptr(), address);
        view[0].im = 2.0;
        let ComplexSlice::F64(view) = f64::complex_slice(&double) else {
            unreachable!()
        };
        assert_eq!(view[0].re.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(view[0].im, 2.0);
        for word in [
            0,
            1,
            255,
            256,
            (1_u64 << 11) - 1,
            1 << 11,
            (1 << 40) - 1,
            1 << 40,
            u64::MAX,
        ] {
            assert_eq!(
                f32::from_u64_half_open01(word).to_bits(),
                (((word >> 40) as f32) / 16_777_216.0).to_bits()
            );
            assert_eq!(
                f64::from_u64_half_open01(word).to_bits(),
                (((word >> 11) as f64) / 9_007_199_254_740_992.0).to_bits()
            );
        }
        for word in [0, 1, 255, 256, u32::MAX] {
            assert_eq!(
                f32::from_u32_half_open01(word).to_bits(),
                (((word >> 8) as f32) / 16_777_216.0).to_bits()
            );
            assert_eq!(
                f64::from_u32_half_open01(word).to_bits(),
                (f64::from(word) / 4_294_967_296.0).to_bits()
            );
        }
        for bits in [
            0_u64,
            1,
            1 << 63,
            f64::INFINITY.to_bits(),
            0x7ff8_0000_0000_0042,
        ] {
            assert_eq!(
                f64::from_value(f64::from_bits(bits).value())
                    .unwrap()
                    .to_bits(),
                bits
            );
        }
    };
}
