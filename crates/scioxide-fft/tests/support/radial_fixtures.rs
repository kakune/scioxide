//! Common exact binary inputs and row decoding for the radial oracles.
use scioxide_core::{Complex, RealScalar};

/// Execute each distinct radial case once, then check its component references.
/// Families supply execution and comparison rules so their envelopes stay explicit.
pub fn check_width<S: RealScalar>(
    width: &str,
    family: &str,
    mut execute: impl FnMut(&[&str], &[Complex<S>], &mut [Complex<S>]),
    check: impl Fn(S, f64, f64, usize),
) {
    let mut last = String::new();
    let mut result = Vec::new();
    for fields in rows(width, family) {
        let key = [&fields[..8], &fields[11..13], &fields[15..]]
            .concat()
            .join(" ");
        let n: usize = fields[2].parse().unwrap();
        if key != last {
            let input = input::<S>(n, fields[12].parse().unwrap());
            result = vec![Complex::new(S::ZERO, S::ZERO); n];
            execute(&fields, &input, &mut result);
            last = key;
        }
        let j: usize = fields[8].parse().unwrap();
        for (actual, expected, mass) in [
            (result[j].re, fields[9], fields[13]),
            (result[j].im, fields[10], fields[14]),
        ] {
            check(actual, expected.parse().unwrap(), mass.parse().unwrap(), n);
        }
    }
}

pub fn rows(width: &str, family: &str) -> impl Iterator<Item = Vec<&'static str>> {
    include_str!("../fixtures/hankel.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .filter(move |fields| fields[0] == width && fields[1] == family)
}
pub fn input<S: RealScalar>(n: usize, exponent: i32) -> Vec<Complex<S>> {
    let scalar = |value| S::try_from_f64_lossy(value).unwrap();
    (0..n)
        .map(|j| {
            Complex::new(
                scalar(j as f64 * j as f64 / 10.0 + if j % 2 == 0 { 0.25 } else { -0.25 })
                    .scalbn(exponent),
                scalar(j as f64 * 0.125).scalbn(exponent),
            )
        })
        .collect()
}
