//! External consumers of the documented implementation boundary.
#![allow(clippy::unwrap_used)]

use scioxide_core::backend::PartitionSlot;

#[test]
fn precise_phase_reduction_preserves_axes_and_rejects_nonfinite_input() {
    use scioxide_core::backend::reduce_radians;
    use scioxide_core::{RealScalar, SciOxideErrorKind};
    {
        for x in [0.0_f64, -0.0, 0.25, -0.25] {
            let (quadrant, high, low) = reduce_radians(x).unwrap();
            assert_eq!(quadrant, 0);
            assert_eq!(high.to_bits(), x.to_bits());
            assert_eq!(low, 0.0);
        }
        for x in [
            core::f64::consts::FRAC_PI_4.next_after(f64::INFINITY),
            -4.493409457909064,
            1e100,
            f64::MAX,
        ] {
            let (quadrant, high, low) = reduce_radians(x).unwrap();
            assert!(high.abs() <= core::f64::consts::FRAC_PI_4);
            let (s, c) = RealScalar::sin_cos(high + low);
            let reduced = match quadrant & 3 {
                0 => (s, c),
                1 => (c, -s),
                2 => (-s, -c),
                _ => (-c, s),
            };
            let expected = RealScalar::sin_cos(x);
            assert!((reduced.0 - expected.0).abs() <= 8.0 * f64::EPSILON);
            assert!((reduced.1 - expected.1).abs() <= 8.0 * f64::EPSILON);
        }
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                reduce_radians(x).unwrap_err().kind(),
                SciOxideErrorKind::NonFinite
            );
        }
    };
}

#[test]
fn partition_claims_transfer_borrowed_storage_once_without_allocation() {
    let mut data = [1.0_f64, 2.0];
    {
        let slot = PartitionSlot::new(&mut data[..]);
        let borrow = slot.take().unwrap();
        borrow[1] = 3.0;
        assert!(slot.take().is_none());
        assert!(slot.was_claimed());
    };
    assert_eq!(data, [1.0, 3.0]);
}

#[test]
fn integer_rounding_preserves_boundaries_and_reports_actual_range_failure() {
    use scioxide_core::backend::scalar_from_u128_rounded;
    use scioxide_core::{ConversionType, RealScalar, ScalarConversionReason, SciOxideErrorView};

    fn check<S: RealScalar>() {
        let boundary = 1_u128 << S::MANTISSA_DIGITS;
        for (input, expected) in [
            (0, 0),
            (boundary - 1, boundary - 1),
            (boundary, boundary),
            (boundary + 1, boundary),
            (boundary + 2, boundary + 2),
            (boundary + 3, boundary + 4),
        ] {
            let expected = S::try_from_f64_exact(expected as f64).unwrap();
            assert_eq!(
                scalar_from_u128_rounded::<S>(input)
                    .unwrap()
                    .into_f64()
                    .to_bits(),
                expected.into_f64().to_bits(),
            );
        }
    }
    {
        check::<f32>();
        check::<f64>();
        assert_eq!(
            scalar_from_u128_rounded::<f64>(u128::MAX)
                .unwrap()
                .to_bits(),
            (u128::MAX as f64).to_bits(),
        );
        assert!(matches!(
            scalar_from_u128_rounded::<f32>(u128::MAX)
                .unwrap_err()
                .view(),
            SciOxideErrorView::ScalarConversion {
                from_type: ConversionType::U128,
                to_type: ConversionType::F32,
                reason: ScalarConversionReason::OutOfRange,
            }
        ));
    };
}
