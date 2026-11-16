//! Direct samplers retain the shared Pearson/DKW distributional contract.
#![allow(clippy::unwrap_used)]
use scioxide_core::RealScalar;
use scioxide_random::*;

use crate::statistics;

fn goodness<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let mut rng = SplitMix64::from_seed(1343);
    for weights in [[0.2, 0.3, 0.5], [0.0, 1.0, 0.0], [3.0, 0.0, 2.0]] {
        let law = Mixture::new(
            [-6.0, 0.0, 5.0]
                .map(|mu| Normal::new(s(mu), S::ONE).unwrap())
                .to_vec(),
            &weights.map(s),
        )
        .unwrap();
        statistics::continuous("direct mixture", &law, || law.sample(&mut rng).unwrap());
    }
    for bins in [16, 256, 4096] {
        let edges: Vec<_> = (0..=bins)
            .map(|i| S::try_from_usize_exact(i).unwrap())
            .collect();
        let masses: Vec<_> = (0..bins)
            .map(|i| s(if i % 5 == 0 { 0.0 } else { (i % 7 + 1) as f64 }))
            .collect();
        let law = Histogram::from_masses(&edges, &masses).unwrap();
        statistics::continuous("prepared histogram", &law, || law.sample(&mut rng).unwrap());
    }
    for power in [0.2, 1.0, 2.0, 20.0] {
        let law = PowerNormal::new(s(power)).unwrap();
        statistics::continuous("direct power normal", &law, || {
            law.sample(&mut rng).unwrap()
        });
    }
    for (low, high) in [(-1.0, 2.0), (8.0, 9.0), (-9.0, -8.0)] {
        let law = Truncated::new(Normal::new(S::ZERO, S::ONE).unwrap(), s(low), s(high)).unwrap();
        statistics::continuous("direct truncation", &law, || law.sample(&mut rng).unwrap());
    }
    for exponent in [0.0, 0.5, 1.0 - 1e-5, 1.0, 1.0 + 1e-5, 1.1, 4.0] {
        let law = Zipfian::new(s(exponent), 1000).unwrap();
        // Independent finite-PMF oracle, prepared once rather than repeatedly
        // recomputing the law's O(n) normalizer for every test bin.
        let a = s(exponent).into_f64();
        let mut probabilities: Vec<_> = (1..=1000).map(|k| (k as f64).powf(-a)).collect();
        let total: f64 = probabilities.iter().sum();
        probabilities.iter_mut().for_each(|p| *p /= total);
        let mut cumulative = vec![0.0];
        for &p in &probabilities {
            cumulative.push((cumulative.last().unwrap() + p).min(1.0));
        }
        for k in [1, 10, 1000] {
            assert!(
                (law.pmf(k).unwrap().into_f64() - probabilities[k - 1]).abs()
                    < statistics::reference_error::<S>()
            );
            assert!(
                (law.cdf(k).unwrap().into_f64() - cumulative[k]).abs()
                    < statistics::reference_error::<S>()
            );
        }
        statistics::discrete(
            "direct Zipfian",
            1,
            1000,
            |k| probabilities[k as usize - 1],
            |k| cumulative[k as usize],
            statistics::reference_error::<S>(),
            || law.sample(&mut rng).unwrap() as i64,
        );
    }
}

#[test]
fn goodness_f32() {
    goodness::<f32>();
}
#[test]
fn goodness_f64() {
    goodness::<f64>();
}

fn histogram_boundaries<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let law = Histogram::from_masses(
        &[s(-1.0), s(0.0), s(1.0), s(2.0), s(3.0), s(4.0)],
        &[S::ZERO, S::ONE, S::ZERO, S::ONE, S::ZERO],
    )
    .unwrap();
    assert_eq!(law.quantile(S::ZERO).unwrap(), S::ZERO);
    assert_eq!(law.quantile(S::ONE).unwrap(), s(3.0));
    assert_eq!(law.quantile(S::HALF).unwrap(), S::ONE);
    assert_eq!(law.isf(S::HALF).unwrap(), s(2.0));
    assert_eq!(law.log_pdf(S::ONE).unwrap(), S::NEG_INFINITY);
    assert!(law.contains(S::ONE).unwrap());
    assert!(law.contains(s(2.0)).unwrap());
    assert!(!law.contains(s(1.5)).unwrap());
    for p in [-0.1, 1.1] {
        assert!(law.quantile(s(p)).is_err());
        assert!(law.isf(s(p)).is_err());
    }
    assert!(law.quantile(S::NAN).is_err());
    assert!(law.isf(S::NAN).is_err());
    let tiny = s(if S::EPSILON.into_f64() > 1e-10 {
        1e-30
    } else {
        1e-300
    });
    let law =
        Histogram::from_masses(&[S::ZERO, S::ONE, s(2.0), s(3.0)], &[tiny, S::ONE, tiny]).unwrap();
    let tol = 32.0 * S::EPSILON.into_f64() * tiny.ln().into_f64().abs();
    for x in [
        law.quantile(tiny * S::HALF).unwrap(),
        s(3.0) - law.isf(tiny * S::HALF).unwrap(),
    ] {
        assert!((x.into_f64() - 0.5).abs() < tol);
    }
}
#[test]
fn histogram_boundaries_both_widths() {
    histogram_boundaries::<f32>();
    histogram_boundaries::<f64>();
}

#[test]
fn inverse_extremes() {
    for power in [0.01, 0.2, 1.0, 2.0, 100.0] {
        let law = PowerNormal::new(power).unwrap();
        assert_eq!(law.quantile(0.0).unwrap(), f64::NEG_INFINITY);
        assert_eq!(law.quantile(1.0).unwrap(), f64::INFINITY);
        for p in [1e-300, 1e-20, 0.1, 0.5, 0.9] {
            for (x, log) in [
                (law.quantile(p).unwrap(), false),
                (law.isf(p).unwrap(), true),
            ] {
                let got = if log {
                    law.log_sf(x).unwrap()
                } else {
                    law.log_cdf(x).unwrap()
                };
                assert!(
                    (got - p.ln()).abs() <= 2e-8 * (1.0 + p.ln().abs()),
                    "power={power}, p={p}, x={x}, log={got}"
                );
            }
        }
    }
    for (low, high) in [(38.3, 39.0), (40.0, 41.0), (-41.0, -40.0)] {
        let law = Truncated::new(Normal::new(0.0_f64, 1.0).unwrap(), low, high).unwrap();
        let x = law.quantile(0.3).unwrap();
        assert!((law.cdf(x).unwrap() - 0.3).abs() < 1e-8);
    }
    let law = Truncated::new(Normal::new(0.0_f32, 1.0).unwrap(), 16.0, 17.0).unwrap();
    let x = law.quantile(0.3).unwrap();
    assert!((law.cdf(x).unwrap() - 0.3).abs() < 0.002);
}

/// Count dispatches separately from evaluations, so a CDF-inverting draw fails
/// these tests even when its output distribution happens to be correct.
struct Probe<'a> {
    law: Normal<f64>,
    evaluations: &'a core::cell::Cell<usize>,
    inverses: &'a core::cell::Cell<usize>,
    draws: &'a core::cell::Cell<usize>,
}
macro_rules! forward {
    ($trait:ident, $method:ident, $counter:ident) => {
        impl $trait<f64> for Probe<'_> {
            fn $method(&self, x: f64) -> Result<f64, scioxide_core::SciOxideError> {
                self.$counter.set(self.$counter.get() + 1);
                self.law.$method(x)
            }
        }
    };
}
forward!(Density, pdf, evaluations);
forward!(LogDensity, log_pdf, evaluations);
forward!(Cdf, cdf, evaluations);
forward!(LogCdf, log_cdf, evaluations);
forward!(Survival, sf, evaluations);
forward!(LogSurvival, log_sf, evaluations);
forward!(Quantile, quantile, inverses);
forward!(InverseSurvival, isf, inverses);
impl Support<f64> for Probe<'_> {
    fn support(
        &self,
    ) -> Result<scioxide_core::ExtendedInterval<f64>, scioxide_core::SciOxideError> {
        self.law.support()
    }
}
impl Distribution<f64> for Probe<'_> {
    fn sample<R: Rng<f64>>(&self, rng: &mut R) -> Result<f64, scioxide_core::SciOxideError> {
        self.draws.set(self.draws.get() + 1);
        self.law.sample(rng)
    }
}
#[test]
fn component_and_base_inverse_dispatch() {
    use core::cell::Cell;
    let evaluations = Cell::new(0);
    let inverses = Cell::new(0);
    let draws = Cell::new(0);
    let probe = || Probe {
        law: Normal::new(0.0, 1.0).unwrap(),
        evaluations: &evaluations,
        inverses: &inverses,
        draws: &draws,
    };
    let mix = Mixture::new(vec![probe(), probe()], &[0.0, 1.0]).unwrap();
    let mut rng = SplitMix64::from_seed(1343);
    for _ in 0..64 {
        mix.sample(&mut rng).unwrap();
    }
    assert_eq!(draws.get(), 64);
    assert_eq!(evaluations.get(), 0);
    assert_eq!(inverses.get(), 0);
    for (low, high) in [(-1.0, 2.0), (8.0, 9.0), (-9.0, -8.0)] {
        let law = Truncated::new(probe(), low, high).unwrap();
        evaluations.set(0);
        inverses.set(0);
        law.quantile(0.3).unwrap();
        law.isf(0.3).unwrap();
        assert_eq!(inverses.get(), 2);
        assert!(evaluations.get() <= 2);
    }
}

#[test]
fn shared_mixture_weights_and_log_range() {
    let normal = Normal::new(0.0_f64, 1.0).unwrap();
    for weights in [
        [0.0, 0.0],
        [-1.0, 1.0],
        [f64::NAN, 1.0],
        [f64::INFINITY, 1.0],
    ] {
        let alias = WeightedSampler::new(&weights).unwrap_err();
        let mixture = Mixture::new(vec![normal.clone(), normal.clone()], &weights).unwrap_err();
        assert_eq!(alias.kind(), mixture.kind());
    }
    for weights in [[f64::MAX, f64::MAX], [f64::from_bits(1), f64::MAX]] {
        let law = Mixture::new(vec![normal.clone(), normal.clone()], &weights).unwrap();
        assert!((law.log_pdf(0.0).unwrap() - normal.log_pdf(0.0).unwrap()).abs() < 1e-12);
    }
    // The tiny component remains visible to evaluations even though its
    // normalized linear weight is below the scalar's representable range.
    let law = Mixture::new(
        vec![
            Uniform::new(0.0, 1.0).unwrap(),
            Uniform::new(2.0, 3.0).unwrap(),
        ],
        &[f64::from_bits(1), f64::MAX],
    )
    .unwrap();
    assert!(law.log_pdf(0.5).unwrap().is_finite());
    assert_eq!(law.pdf(0.5).unwrap(), 0.0);
}

#[test]
fn truncation_preserves_generalized_inverse_sides_at_gaps() {
    let law = Histogram::from_masses(&[0.0_f64, 1.0, 2.0, 3.0], &[1.0, 0.0, 3.0]).unwrap();
    let truncated = Truncated::new(&law, 0.0, 3.0).unwrap();
    assert!((truncated.quantile(law.cdf(1.0).unwrap()).unwrap() - 1.0).abs() < 1e-12);
    assert!((truncated.isf(law.sf(2.0).unwrap()).unwrap() - 2.0).abs() < 1e-12);
    let law = Histogram::from_masses(&[0.0_f64, 1.0, 2.0, 3.0], &[3.0, 0.0, 1.0]).unwrap();
    let truncated = Truncated::new(&law, 0.0, 3.0).unwrap();
    assert!((truncated.quantile(law.cdf(1.0).unwrap()).unwrap() - 1.0).abs() < 1e-12);
    assert!((truncated.isf(law.sf(2.0).unwrap()).unwrap() - 2.0).abs() < 1e-12);
}
