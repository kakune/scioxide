//! Independent fast-path boundaries, conservative errors and execution contracts.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{Budget, CancellationToken, ExecutionContext, RealScalar, SciOxideError};
use scioxide_special::*;

fn evaluate<S: RealScalar>(
    family: &str,
    order: S,
    x: S,
    scaled: bool,
    budget: Budget,
) -> Result<SpecialFunctionResult<S>, SciOxideError> {
    let b = BesselOptions::new(budget);
    let a = AiryOptions::new(budget);
    let k = KelvinOptions::new(budget);
    match (family, scaled) {
        ("j", _) => bessel_j_with_options(order, x, b),
        ("y", _) => bessel_y_with_options(order, x, b),
        ("ai", false) => airy_ai_with_options(x, a),
        ("ai", true) => airy_ai_scaled_with_options(x, a),
        ("bi", false) => airy_bi_with_options(x, a),
        ("bi", true) => airy_bi_scaled_with_options(x, a),
        ("ber", false) => ber_with_options(x, k),
        ("ber", true) => ber_scaled_with_options(x, k),
        ("bei", false) => bei_with_options(x, k),
        ("bei", true) => bei_scaled_with_options(x, k),
        ("ker", false) => ker_with_options(x, k),
        ("ker", true) => ker_scaled_with_options(x, k),
        ("kei", false) => kei_with_options(x, k),
        ("kei", true) => kei_scaled_with_options(x, k),
        _ => panic!("unknown fixture family"),
    }
}

fn plain<S: RealScalar>(family: &str, order: S, x: S, scaled: bool) -> Result<S, SciOxideError> {
    match (family, scaled) {
        ("j", _) => bessel_j(order, x),
        ("y", _) => bessel_y(order, x),
        ("ai", false) => airy_ai(x),
        ("ai", true) => airy_ai_scaled(x),
        ("bi", false) => airy_bi(x),
        ("bi", true) => airy_bi_scaled(x),
        ("ber", false) => ber(x),
        ("ber", true) => ber_scaled(x),
        ("bei", false) => bei(x),
        ("bei", true) => bei_scaled(x),
        ("ker", false) => ker(x),
        ("ker", true) => ker_scaled(x),
        ("kei", false) => kei(x),
        ("kei", true) => kei_scaled(x),
        _ => panic!("unknown fixture family"),
    }
}

fn references<S: RealScalar>(width: &str) {
    let decode = |bits: &str| {
        let bits = u64::from_str_radix(bits, 16).unwrap();
        if width == "32" {
            f64::from(f32::from_bits(bits as u32))
        } else {
            f64::from_bits(bits)
        }
    };
    let epsilon = S::EPSILON.into_f64();
    let minimum = if width == "32" {
        f64::from(f32::from_bits(1))
    } else {
        f64::from_bits(1)
    };
    for row in include_str!("data/real_axis.tsv")
        .lines()
        .filter(|r| !r.starts_with('#'))
    {
        let f: Vec<_> = row.split_whitespace().collect();
        if f[0] != width {
            continue;
        }
        let order = S::try_from_f64_exact(f[2].parse().unwrap()).unwrap();
        let x = S::try_from_f64_lossy(decode(f[3])).unwrap();
        let scaled = f[4] == "1";
        let r = evaluate(f[1], order, x, scaled, Budget::try_from(65536).unwrap()).unwrap();
        assert!(r.termination().is_success(), "{row}: {r:?}");
        assert_eq!(
            r.try_value().unwrap(),
            plain(f[1], order, x, scaled).unwrap(),
            "{row}"
        );
        let units = if f[1].len() == 3 { 512.0 } else { 64.0 };
        for (result, expected, scale) in [
            (r.value(), decode(f[5]), f[7]),
            (r.derivative(), decode(f[6]), f[8]),
        ] {
            let Some(result) = result else {
                assert_eq!(expected, 0.0, "{row}");
                continue;
            };
            let actual = result.value().into_f64();
            let bound = result.absolute_error().into_f64();
            let error = (actual - expected).abs();
            let rounding = (epsilon * expected.abs()).max(minimum);
            assert!(bound.is_finite() && bound >= 0.0, "{row}: {bound}");
            assert!(
                error <= bound + rounding,
                "{row}: error {error:e} > bound {bound:e}"
            );
            let limit = units * (epsilon * scale.parse::<f64>().unwrap()).max(minimum);
            assert!(
                error <= limit,
                "{row}: error {error:e} > accuracy {limit:e}"
            );
        }
    }
}

#[test]
fn independent_binary64_boundaries() {
    references::<f64>("64");
}
#[test]
fn independent_binary32_boundaries() {
    references::<f32>("32");
}

#[test]
fn fixed_and_iterative_paths_admit_their_work_with_reused_storage() {
    for family in ["j", "y", "ai", "bi", "ber", "bei", "ker", "kei"] {
        for x in [2.8_f64, 8.0, 20.0, 30.0, 180.0] {
            let x = if matches!(family, "ai" | "bi") { -x } else { x };
            let result = evaluate(family, 0.0, x, false, Budget::try_from(65536).unwrap()).unwrap();
            let used = result.counts().iterations();
            assert!(used > 1);
            assert!(evaluate(family, 0.0, x, false, Budget::try_from(used - 1).unwrap()).is_err());
            assert_eq!(
                result,
                evaluate(family, 0.0, x, false, Budget::try_from(used).unwrap()).unwrap()
            );
        }
    }
}

#[test]
fn fast_paths_respect_precancelled_scopes() {
    let token = CancellationToken::new();
    token.cancel();
    let context = ExecutionContext::sequential().with_cancellation(&token);
    assert!(bessel_j_scoped(0.0_f64, 2.8, BesselOptions::default(), context).is_err());
    assert!(airy_ai_scoped(-2.338_f64, AiryOptions::default(), context).is_err());
    assert!(ber_scoped(30.0_f64, KelvinOptions::default(), context).is_err());
}
