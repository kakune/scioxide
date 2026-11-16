//! Regressions for scalar fast paths, independent of timing measurements.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Complex, RealScalar, sincpi};

#[test]
fn sincpi_retains_one_destination_ulp_on_independent_fixtures() {
    for data in [
        include_str!("data/stable_transforms.txt"),
        include_str!("data/stable_boundaries.txt"),
        include_str!("data/sincpi.txt"),
    ] {
        for line in data.lines().filter(|line| !line.starts_with('#')) {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.get(1) != Some(&"sincpi") {
                continue;
            }
            let bits = |index| u64::from_str_radix(fields[index], 16).unwrap();
            if fields[0] == "32" {
                let actual = sincpi(f32::from_bits(bits(2) as u32)).unwrap();
                let expected = f32::from_bits(bits(4) as u32);
                assert_eq!(
                    actual.is_sign_negative(),
                    expected.is_sign_negative(),
                    "{line}"
                );
                assert!(
                    actual.to_bits().abs_diff(expected.to_bits()) <= 1,
                    "{line}: {actual:?}"
                );
            } else {
                let actual = sincpi(f64::from_bits(bits(2))).unwrap();
                let expected = f64::from_bits(bits(4));
                assert_eq!(
                    actual.is_sign_negative(),
                    expected.is_sign_negative(),
                    "{line}"
                );
                assert!(
                    actual.to_bits().abs_diff(expected.to_bits()) <= 1,
                    "{line}: {actual:?}"
                );
            }
        }
    }
}

macro_rules! width_regressions {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            for exponent in [i32::MIN, i32::MIN + 1, -16_777_217, 16_777_217, i32::MAX] {
                let sign = if exponent & 1 == 0 { 1.0 } else { -1.0 };
                assert_eq!(RealScalar::powi(-1.0 as $ty, exponent), sign);
                let zero = RealScalar::powi(-0.0 as $ty, exponent);
                assert_eq!(zero.is_sign_negative(), exponent & 1 != 0);
                assert_eq!(zero.is_infinite(), exponent < 0);
            }
            assert_eq!(RealScalar::powi(<$ty>::NAN, 0), 1.0);
            assert_eq!(RealScalar::powi(-1.0 as $ty, 0), 1.0);
            for (re, im) in [
                (<$ty>::MAX, <$ty>::MAX),
                (-<$ty>::MAX, <$ty>::MAX),
                (<$ty>::from_bits(1), <$ty>::from_bits(1)),
                (-<$ty>::from_bits(1), <$ty>::from_bits(1)),
                (<$ty>::MIN_POSITIVE, <$ty>::MAX),
                (<$ty>::MAX, <$ty>::MIN_POSITIVE),
            ] {
                for im in [im, -im] {
                    let root = Complex::new(re, im).sqrt();
                    assert!(root.is_finite());
                    assert!(!root.re.is_sign_negative());
                    assert_eq!(root.im.is_sign_negative(), im.is_sign_negative());
                    // Normalize before squaring to avoid an overflowing test
                    // intermediate at the very boundary being exercised.
                    let scale = re.abs().max(im.abs());
                    let scale_root = scale.sqrt();
                    let normalized = root / scale_root;
                    let square = normalized * normalized;
                    let tolerance = 16.0 * <$ty>::EPSILON;
                    assert!((square.re - re / scale).abs() <= tolerance);
                    assert!((square.im - im / scale).abs() <= tolerance);
                }
            }
        }
    };
}
width_regressions!(binary32_routes_and_complex_sqrt, f32);
width_regressions!(binary64_routes_and_complex_sqrt, f64);
