//! Independent paired-precision zero tables, including high orders and indices.
#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]
#![expect(
    clippy::panic,
    reason = "Tests report unexpected fixture cases or solver outcomes."
)]
use scioxide_core::{Complex, RealScalar};
use scioxide_rootfind::KelvinZeroFunction as K;
use scioxide_rootfind::*;
fn family<S: RealScalar>(name: &str, order: S, prime: bool) -> SpecialZeroFamily<S> {
    use SpecialZeroFamily as F;
    match (name, prime) {
        ("j", false) => F::BesselJ(order),
        ("j", true) => F::BesselJPrime(order),
        ("y", false) => F::BesselY(order),
        ("y", true) => F::BesselYPrime(order),
        ("ai", false) => F::AiryAi,
        ("ai", true) => F::AiryAiPrime,
        ("bi", false) => F::AiryBi,
        ("bi", true) => F::AiryBiPrime,
        ("ber", false) => F::Kelvin(K::Ber),
        ("ber", true) => F::Kelvin(K::BerPrime),
        ("bei", false) => F::Kelvin(K::Bei),
        ("bei", true) => F::Kelvin(K::BeiPrime),
        ("ker", false) => F::Kelvin(K::Ker),
        ("ker", true) => F::Kelvin(K::KerPrime),
        ("kei", false) => F::Kelvin(K::Kei),
        ("kei", true) => F::Kelvin(K::KeiPrime),
        ("c", false) => F::FresnelC,
        ("c", true) => F::FresnelCPrime,
        ("s", false) => F::FresnelS,
        ("s", true) => F::FresnelSPrime,
        ("erf", false) => F::Erf,
        _ => panic!("unknown fixture family"),
    }
}
fn check<S: RealScalar>(width: &str, decode: impl Fn(&str) -> S) {
    let mut cases = std::collections::BTreeSet::new();
    for row in include_str!("data/special_zeros.tsv")
        .lines()
        .chain(include_str!("data/special_zero_starts.tsv").lines())
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = row.split_whitespace().collect();
        if fields[0] != width {
            continue;
        }
        assert!(
            cases.insert(fields[..6].to_vec()),
            "duplicate oracle: {row}"
        );
        let f = family(fields[1], decode(fields[2]), fields[3] == "1");
        let branch = if fields[5] == "real" {
            SpecialZeroBranch::Real
        } else {
            SpecialZeroBranch::UPPER
        };
        let r = special_zeros_with_options(
            f,
            branch,
            fields[4].parse().unwrap(),
            1,
            &SpecialZeroOptions::default(),
        )
        .unwrap_or_else(|e| panic!("{row}: {e}"));
        assert_eq!(
            r.report().termination(),
            RootTermination::Complete,
            "{row}: {:?}",
            r.report()
        );
        let actual = r.roots()[0].point();
        let expected = Complex::new(decode(fields[6]), decode(fields[7]));
        let error = (actual.re - expected.re).hypot(actual.im - expected.im);
        let bound = S::try_from_f64_lossy(64.0).unwrap()
            * S::EPSILON
            * expected.re.hypot(expected.im).max(S::ONE);
        assert!(
            error <= bound,
            "{row}: {actual:?} vs {expected:?}; error {error:?} > {bound:?}"
        );
        assert!(r.roots()[0].estimated_error().is_finite());
        assert!(r.roots()[0].inverse_slope().is_some());
    }
}
#[test]
fn f64_independent_zero_tables() {
    check("64", |s| {
        f64::from_bits(u64::from_str_radix(s, 16).unwrap())
    });
}
#[test]
fn f32_independent_zero_tables() {
    check("32", |s| {
        f32::from_bits(u32::from_str_radix(s, 16).unwrap())
    });
}
