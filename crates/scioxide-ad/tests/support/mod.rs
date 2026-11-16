//! Readers and ulp comparisons for the reference files in `tests/data/`.
#![allow(dead_code, clippy::unwrap_used, clippy::panic)]

use scioxide_core::{NumericScalar, RealScalar};

/// Rows of a reference file, split into fields; comment lines are skipped.
pub fn rows(text: &str) -> impl Iterator<Item = (&str, Vec<&str>)> {
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| (line, line.split_whitespace().collect()))
}

/// Decodes the hex bits of a `width`-bit IEEE value into `S`.
pub fn decode<S: RealScalar>(text: &str, width: usize) -> S {
    let bits = u64::from_str_radix(text, 16).unwrap();
    let value = if width == 32 {
        f64::from(f32::from_bits(bits as u32))
    } else {
        f64::from_bits(bits)
    };
    S::from_f64_ieee(value)
}

/// IEEE bits of `x` as an unsigned integer of the width of `S`.
pub fn bits<S: RealScalar>(x: S) -> u64 {
    if S::MANTISSA_DIGITS == 24 {
        u64::from((x.into_f64() as f32).to_bits())
    } else {
        x.into_f64().to_bits()
    }
}

/// Distance in units in the last place, on the ordered integer line (so `-0` and
/// `+0` are adjacent and a sign change counts every value in between).
pub fn ulps<S: RealScalar>(a: S, b: S) -> u64 {
    let ordered = |x: S| -> i128 {
        let sign = if S::MANTISSA_DIGITS == 24 {
            1u64 << 31
        } else {
            1u64 << 63
        };
        let raw = bits(x);
        if raw & sign != 0 {
            -i128::from(raw & !sign)
        } else {
            i128::from(raw)
        }
    };
    ordered(a).abs_diff(ordered(b)) as u64
}

/// Asserts `actual` within `allowance` ulps of `expected`, exactly for non-finite
/// expectations and the last four subnormals.
pub fn close<S: RealScalar>(actual: S, expected: S, allowance: u64, context: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{context}: {actual:?}, expected NaN");
        return;
    }
    if !expected.is_finite() {
        assert_eq!(actual, expected, "{context}");
        return;
    }
    if actual == S::ZERO && expected == S::ZERO {
        return;
    }
    assert!(actual.is_finite(), "{context}: {actual:?} != {expected:?}");
    let allowance = if bits(NumericScalar::abs(expected)) <= 4 {
        0
    } else {
        allowance
    };
    let distance = ulps(actual, expected);
    assert!(
        distance <= allowance,
        "{context}: {actual:?} != {expected:?}, {distance} ulp"
    );
}

/// Whether every value is zero or has magnitude in `[√MIN_POSITIVE, √MAX]`, the
/// range in which plain IEEE evaluation of the derivative formulas is asserted.
pub fn moderate<S: RealScalar>(values: &[S]) -> bool {
    let low = S::MIN_POSITIVE.sqrt();
    let high = S::MAX.sqrt();
    values.iter().all(|&v| {
        let m = NumericScalar::abs(v);
        v == S::ZERO || (m >= low && m <= high)
    })
}
