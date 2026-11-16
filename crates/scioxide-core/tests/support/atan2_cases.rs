// Shared by library and public integration tests. The oracle consists of exact
// destination-format bits for geometric axis/infinity limits, not host atan2.
use super::{Complex, RealScalar};

macro_rules! boundary_cases {
    ($name:ident, $ty:ty, $bits:ty, $sign:expr, $pi:expr, $half_pi:expr, $quarter_pi:expr, $three_quarter_pi:expr) => {
        #[test]
        fn $name() {
            let check = |y: $ty, x: $ty, expected: $bits| {
                let y = core::hint::black_box(y);
                let x = core::hint::black_box(x);
                assert_eq!(
                    RealScalar::atan2(y, x).to_bits(),
                    expected,
                    "atan2({y:?}, {x:?})"
                );
                assert_eq!(Complex::new(x, y).arg().to_bits(), expected);
            };
            // Either signed zero ordinate selects the x-axis using the sign of x.
            for (x, positive) in [
                (0.0 as $ty, 0),
                (-0.0, $pi),
                (<$ty>::from_bits(1), 0),
                (-<$ty>::from_bits(1), $pi),
                (<$ty>::MIN_POSITIVE, 0),
                (-<$ty>::MIN_POSITIVE, $pi),
                (1.0, 0),
                (-1.0, $pi),
                (<$ty>::MAX, 0),
                (-<$ty>::MAX, $pi),
                (<$ty>::INFINITY, 0),
                (<$ty>::NEG_INFINITY, $pi),
            ] {
                check(0.0, x, positive);
                check(-0.0, x, positive | $sign);
            }
            for magnitude in [
                <$ty>::from_bits(1),
                <$ty>::MIN_POSITIVE,
                1.0,
                <$ty>::MAX,
            ] {
                for x in [0.0, -0.0] {
                    check(magnitude, x, $half_pi);
                    check(-magnitude, x, $half_pi | $sign);
                }
                check(magnitude, <$ty>::INFINITY, 0);
                check(-magnitude, <$ty>::INFINITY, $sign);
                check(magnitude, <$ty>::NEG_INFINITY, $pi);
                check(-magnitude, <$ty>::NEG_INFINITY, $pi | $sign);
                for x in [magnitude, -magnitude] {
                    check(<$ty>::INFINITY, x, $half_pi);
                    check(<$ty>::NEG_INFINITY, x, $half_pi | $sign);
                }
            }
            for x in [0.0, -0.0] {
                check(<$ty>::INFINITY, x, $half_pi);
                check(<$ty>::NEG_INFINITY, x, $half_pi | $sign);
            }
            check(<$ty>::INFINITY, <$ty>::INFINITY, $quarter_pi);
            check(<$ty>::NEG_INFINITY, <$ty>::INFINITY, $quarter_pi | $sign);
            check(<$ty>::INFINITY, <$ty>::NEG_INFINITY, $three_quarter_pi);
            check(<$ty>::NEG_INFINITY, <$ty>::NEG_INFINITY, $three_quarter_pi | $sign);
            // NaN payload/sign are intentionally not specified by the contract.
            for value in [0.0, -0.0, 1.0, -1.0, <$ty>::INFINITY, <$ty>::NEG_INFINITY, <$ty>::NAN] {
                for nan in [<$ty>::NAN, -<$ty>::NAN] {
                    assert!(RealScalar::atan2(nan, value).is_nan());
                    assert!(RealScalar::atan2(value, nan).is_nan());
                }
            }
        }
    };
}

// Nearest-even IEEE encodings of pi, pi/2, pi/4 and 3pi/4.
boundary_cases!(binary32_atan2_boundaries, f32, u32, 0x8000_0000,
    0x4049_0fdb, 0x3fc9_0fdb, 0x3f49_0fdb, 0x4016_cbe4);
boundary_cases!(binary64_atan2_boundaries, f64, u64, 0x8000_0000_0000_0000,
    0x4009_21fb_5444_2d18, 0x3ff9_21fb_5444_2d18,
    0x3fe9_21fb_5444_2d18, 0x4002_d97c_7f33_21d2);
