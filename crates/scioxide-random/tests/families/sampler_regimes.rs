//! Parameter branches missed by the one-parameter fixture sweeps.
//! Seed 1335, 32768 draws/case; power and alpha are in shared statistics support.
#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
use scioxide_core::RealScalar;
use scioxide_random::*;

use super::statistics;

fn regimes<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let mut rng = SplitMix64::from_seed(1335);
    macro_rules! check {
        ($law:expr) => {{
            let law = $law;
            statistics::continuous(stringify!($law), &law, || law.sample(&mut rng).unwrap());
        }};
    }
    for shape in [0.1, 0.5, 1.0, 3.0] {
        check!(Gamma::new(s(shape), s(2.0)).unwrap());
    }
    for (a, b) in [(0.3, 0.3), (0.3, 5.0), (5.0, 0.3), (2.0, 5.0), (30.0, 50.0)] {
        check!(Beta::new(s(a), s(b)).unwrap());
    }
    for degrees in [0.5, 1.0, 5.0, 30.0] {
        check!(ChiSquared::new(s(degrees)).unwrap());
        check!(StudentT::new(s(degrees)).unwrap());
    }
    for shape in [-0.5, 0.0, 0.5] {
        check!(GeneralizedPareto::new(s(shape)).unwrap());
        check!(GeneralizedExtremeValue::new(s(shape)).unwrap());
    }
    // Fused affine normal and the exponential-Ziggurat laws.
    for (mean, scale) in [(2.0, 3.0), (-1e3, 1e-2)] {
        check!(Normal::new(s(mean), s(scale)).unwrap());
    }
    for rate in [0.25, 1.5, 40.0] {
        check!(Exponential::new(s(rate)).unwrap());
    }
    for shape in [0.3, 1.5, 8.0] {
        check!(WeibullMin::new(s(shape)).unwrap());
    }
    for shape in [0.5, 3.0] {
        check!(Pareto::new(s(shape)).unwrap());
    }
    check!(Logistic::new(s(-2.0), s(0.5)).unwrap());
    check!(GumbelRight::<S>::new());
    // Central and Poisson-mixture noncentral chi-square branches.
    for (degrees, noncentrality) in [(3.0, 0.0), (0.5, 3.0)] {
        check!(
            NoncentralChiSquared::with_options(
                s(degrees),
                s(noncentrality),
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
            )
            .unwrap()
        );
    }
    for (a, b) in [(0.5, 0.5), (5.0, 10.0), (30.0, 50.0)] {
        check!(
            FisherF::with_options(
                s(a),
                s(b),
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
            )
            .unwrap()
        );
    }
}

#[test]
fn continuous_regimes_f32() {
    regimes::<f32>();
}
#[test]
fn continuous_regimes_f64() {
    regimes::<f64>();
}

/// The base strip's tail beyond the Ziggurat cutoff `R ≈ 7.697` has mass
/// `exp(-R) ≈ 4.5e-4`; its frequency stays within five binomial deviations.
#[test]
fn exponential_tail_frequency_matches_the_cutoff_mass() {
    fn check<S: RealScalar>() {
        let law = Exponential::new(S::ONE).unwrap();
        let mut rng = SplitMix64::from_seed(1346);
        let draws = 400_000_usize;
        let cutoff = S::try_from_f64_lossy(7.697_117_470_131_05).unwrap();
        let beyond = (0..draws)
            .filter(|_| law.sample(&mut rng).unwrap() > cutoff)
            .count() as f64;
        let mass = (-7.697_117_470_131_05_f64).exp();
        let expected = mass * draws as f64;
        let deviation = (expected * (1.0 - mass)).sqrt();
        assert!(
            (beyond - expected).abs() <= 5.0 * deviation,
            "{beyond} tail draws, expected {expected}"
        );
    }
    check::<f32>();
    check::<f64>();
}
