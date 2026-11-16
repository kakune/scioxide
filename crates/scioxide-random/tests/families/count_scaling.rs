#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
use super::statistics;
use scioxide_core::{Budget, RealScalar, SciOxideError};
use scioxide_random::*;

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual == expected || (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{actual} vs {expected}"
    );
}

#[test]
fn independent_large_count_references_and_default_quantiles() {
    for line in include_str!("../data/count_scaling.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let row: Vec<_> = line.split_whitespace().collect();
        let n = row[1].parse().unwrap();
        if row[0] == "q:zipfian" {
            let law = Zipfian::new(row[2].parse::<f64>().unwrap(), n).unwrap();
            let p = row[3].parse().unwrap();
            let expected = row[4].parse::<usize>().unwrap();
            assert_eq!(law.quantile(p).unwrap(), expected);
            assert!(law.cdf(expected).unwrap() >= p);
            assert!(expected == 1 || law.cdf(expected - 1).unwrap() < p);
            continue;
        }
        let k = row[3].parse().unwrap();
        let actual = match row[0] {
            "zipfian" => {
                let law = Zipfian::new(row[2].parse::<f64>().unwrap(), n).unwrap();
                [
                    law.log_pmf(k).unwrap(),
                    law.log_cdf(k).unwrap(),
                    law.log_sf(k).unwrap(),
                ]
            }
            "hypergeometric" => {
                let half = row[2].parse().unwrap();
                let law = Hypergeometric::<f64>::new(n, half, half).unwrap();
                [
                    law.log_pmf(k).unwrap(),
                    law.log_cdf(k).unwrap(),
                    law.log_sf(k).unwrap(),
                ]
            }
            _ => unreachable!(),
        };
        for (value, expected) in actual.into_iter().zip(&row[4..]) {
            close(value, expected.parse().unwrap(), 8e-12);
        }
    }
    if usize::BITS > 49 {
        let law = Binomial::new(1_000_000_000_000_000_u64 as usize, 1e-13_f64).unwrap();
        for k in [80, 100] {
            close(law.cdf(k).unwrap(), law.log_cdf(k).unwrap().exp(), 2e-13);
        }
    }
}

#[derive(Default)]
struct Widths {
    narrow: usize,
    wide: usize,
    fail: bool,
}
impl BitGenerator for Widths {
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        unreachable!("native word requested")
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        unreachable!("native word requested")
    }
    fn next_uniform_u32(&mut self) -> Result<u32, SciOxideError> {
        self.narrow += 1;
        if self.fail {
            Err(SciOxideError::not_implemented("injected failure"))
        } else {
            Ok(u32::MAX)
        }
    }
    fn next_uniform_u64(&mut self) -> Result<u64, SciOxideError> {
        self.wide += 1;
        if self.fail {
            Err(SciOxideError::not_implemented("injected failure"))
        } else {
            Ok(u64::MAX)
        }
    }
    fn fill_bytes(&mut self, _: &mut [u8]) -> Result<(), SciOxideError> {
        unreachable!("bytes requested")
    }
}

#[test]
fn uniform_count_laws_forward_provider_word_widths() {
    for n in [17_u64, 1 << 32, (1 << 32) + 1] {
        let options = LawOptions::<f64>::default().with_sample_word_budget(Budget::ONE);
        for fail in [false, true] {
            let mut rng = Widths {
                fail,
                ..Default::default()
            };
            let result = DiscreteUniform::with_options(0, n as i64, options)
                .unwrap()
                .sample(&mut rng);
            assert_eq!(result.is_err(), fail);
            assert_eq!(
                (rng.narrow, rng.wide),
                if n <= 1 << 32 { (1, 0) } else { (0, 1) }
            );
            if usize::BITS > 32 {
                let mut rng = Widths {
                    fail,
                    ..Default::default()
                };
                let result = Zipfian::with_options(0.0, n as usize, options)
                    .unwrap()
                    .sample(&mut rng);
                assert_eq!(result.is_err(), fail);
                assert_eq!(
                    (rng.narrow, rng.wide),
                    if n <= 1 << 32 { (1, 0) } else { (0, 1) }
                );
            }
        }
    }
}

#[test]
fn hrua_word_exhaustion_stops_before_the_second_provider_request() {
    let options = LawOptions::<f64>::default().with_sample_word_budget(Budget::ONE);
    let law = Hypergeometric::with_options(200_000, 100_000, 100_000, options).unwrap();
    let mut rng = Widths::default();
    assert_eq!(
        law.sample(&mut rng).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!((rng.narrow, rng.wide), (0, 1));
}

#[test]
fn large_f32_queries_and_inverse_thresholds_match_the_widened_parameters() {
    let a = 1.1_f32;
    let law = Zipfian::new(a, 100_000).unwrap();
    let reference = Zipfian::new(f64::from(a), 100_000).unwrap();
    for k in [1, 40, 1000, 99_999] {
        close(
            f64::from(law.pmf(k).unwrap()),
            reference.pmf(k).unwrap(),
            2e-7,
        );
        close(
            f64::from(law.log_sf(k).unwrap()),
            reference.log_sf(k).unwrap(),
            2e-7,
        );
        close(
            f64::from(law.cdf(k).unwrap()),
            reference.cdf(k).unwrap(),
            2e-7,
        );
    }
    for p in [0.01_f32, 0.5, 0.99] {
        assert_eq!(
            law.quantile(p).unwrap(),
            reference.quantile(f64::from(p)).unwrap()
        );
        assert_eq!(law.isf(p).unwrap(), reference.isf(f64::from(p)).unwrap());
    }
    let law = Hypergeometric::<f32>::new(200_000, 100_000, 100_000).unwrap();
    let reference = Hypergeometric::<f64>::new(200_000, 100_000, 100_000).unwrap();
    for k in [49_000, 49_900, 50_000, 50_100, 51_000] {
        close(
            f64::from(law.log_pmf(k).unwrap()),
            reference.log_pmf(k).unwrap(),
            2e-7,
        );
        close(
            f64::from(law.log_cdf(k).unwrap()),
            reference.log_cdf(k).unwrap(),
            2e-7,
        );
        close(
            f64::from(law.sf(k).unwrap()),
            reference.sf(k).unwrap(),
            2e-7,
        );
    }
}

#[test]
fn large_hypergeometric_sampling_has_calibrated_fit_and_exact_totals() {
    fn width<S: RealScalar>() {
        let law = Hypergeometric::<S>::new(200_000, 100_000, 100_000).unwrap();
        let reference = Hypergeometric::<f64>::new(200_000, 100_000, 100_000).unwrap();
        let mut rng = SplitMix64::from_seed(1580);
        statistics::discrete(
            "large hypergeometric",
            49_000,
            51_000,
            |k| reference.pmf(k as usize).unwrap(),
            |k| reference.cdf(k as usize).unwrap(),
            1e-10,
            || law.sample(&mut rng).unwrap() as i64,
        );
        let multi = MultivariateHypergeometric::<S>::new(&[100_000, 100_000], 100_000).unwrap();
        let mut rng = SplitMix64::from_seed(1581);
        statistics::discrete(
            "large multivariate hypergeometric",
            49_000,
            51_000,
            |k| reference.pmf(k as usize).unwrap(),
            |k| reference.cdf(k as usize).unwrap(),
            1e-10,
            || {
                let counts = multi.sample(&mut rng).unwrap();
                assert_eq!(counts.iter().sum::<usize>(), 100_000);
                assert!(counts.iter().all(|&count| count <= 100_000));
                counts[0] as i64
            },
        );
    }
    width::<f32>();
    width::<f64>();
}
