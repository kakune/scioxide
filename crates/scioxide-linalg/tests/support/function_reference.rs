//! Paired-precision references and conditioning-based entrywise norm bounds.
#![allow(clippy::unwrap_used)]

use scioxide_core::{NumericScalar, RealScalar};

pub fn records<S: NumericScalar>(data: &'static str) -> impl Iterator<Item = Vec<&'static str>> {
    data.lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .filter(|fields| {
            fields[0].parse::<usize>().unwrap() == core::mem::size_of::<S::Real>() * 8
                && (fields[1] == "0") == S::IS_REAL
        })
}

pub fn number<R: RealScalar>(value: &str) -> R {
    R::try_from_f64_lossy(value.parse().unwrap()).unwrap()
}

pub fn values<S: NumericScalar>(field: &str) -> Vec<S> {
    let components: Vec<_> = field.split(',').map(number::<S::Real>).collect();
    assert_eq!(components.len() % 2, 0);
    components
        .chunks_exact(2)
        .map(|pair| S::from_components(pair[0], pair[1]))
        .collect()
}

pub fn expected<S: NumericScalar>(decimal: &str, bits: &str) -> Vec<S> {
    let values = values::<S>(decimal);
    let ieee: Vec<_> = bits
        .split(',')
        .map(|value| {
            let raw = u64::from_str_radix(value, 16).unwrap();
            if core::mem::size_of::<S::Real>() == 4 {
                f64::from(f32::from_bits(raw as u32))
            } else {
                f64::from_bits(raw)
            }
        })
        .collect();
    for (value, pair) in values.iter().zip(ieee.chunks_exact(2)) {
        assert_eq!(value.real_component().into_f64(), pair[0]);
        assert_eq!(value.imaginary_component().into_f64(), pair[1]);
    }
    assert_eq!(ieee.len(), 2 * values.len());
    values
}

/// First-order forward error: gamma_(64n) times the independent sensitivity
/// mass. The 64n allowance covers normwise reduction/product roundoff rather
/// than fitting individual cases. It is a stability test, not a certificate.
pub fn assert_close<S: NumericScalar>(
    actual: &[S],
    expected: &[S],
    n: usize,
    sensitivity: &str,
    label: &str,
) {
    assert_eq!(actual.len(), expected.len());
    let nu = 64.0 * n as f64 * S::Real::EPSILON.into_f64() / 2.0;
    let allowance = nu / (1.0 - nu) * sensitivity.parse::<f64>().unwrap()
        + 2.0 * actual.len() as f64 * S::Real::MIN_SUBNORMAL.into_f64();
    let error: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (a - b).abs().into_f64())
        .sum();
    assert!(
        error <= allowance,
        "{label}: error {error:e}, bound {allowance:e}"
    );
}
