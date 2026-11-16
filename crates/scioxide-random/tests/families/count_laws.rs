#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
use scioxide_core::RealScalar;
use scioxide_random::{
    Binomial, BitGenerator, Cdf, CountDistribution, Mass, OutputPolicy, Poisson, SamplingPolicy,
    SamplingWorkspace, SeedableRng, Skellam, SplitMix64,
};

struct Words {
    word: u64,
    draws: usize,
}
impl BitGenerator for Words {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.draws += 1;
        Ok(self.word)
    }
    fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        Ok(self.next_u64().unwrap() as u32)
    }
    fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
        output.fill(0);
        Ok(())
    }
}

fn domains<S: RealScalar>() {
    let mut rng = Words { word: 0, draws: 0 };
    for p in [0.0, 0.25, 1.0] {
        let law = Binomial::new(0, S::try_from_f64_lossy(p).unwrap()).unwrap();
        assert_eq!(law.pmf(0).unwrap(), S::ONE);
        assert_eq!(law.pmf(1).unwrap(), S::ZERO);
        assert_eq!(law.cdf(usize::MAX).unwrap(), S::ONE);
        assert_eq!(law.prepare_sampler().unwrap().sample(&mut rng).unwrap(), 0);
    }
    let zero = Poisson::new(S::ZERO).unwrap();
    assert_eq!(zero.pmf(0).unwrap(), S::ONE);
    assert_eq!(zero.pmf(usize::MAX).unwrap(), S::ZERO);
    assert_eq!(zero.cdf(usize::MAX).unwrap(), S::ONE);
    assert_eq!(zero.prepare_sampler().unwrap().sample(&mut rng).unwrap(), 0);
    for p in [S::ZERO, S::ONE] {
        let law = Binomial::new(usize::MAX, p).unwrap();
        assert_eq!(
            law.prepare_sampler().unwrap().sample(&mut rng).unwrap(),
            if p == S::ZERO { 0 } else { usize::MAX }
        );
    }
    assert_eq!(rng.draws, 0);
    for p in [-0.01, 1.01] {
        let error = Binomial::new(0, S::try_from_f64_lossy(p).unwrap()).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }
    let error = Poisson::new(-S::ONE).unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
}

#[test]
fn checked_domains_both_widths() {
    domains::<f32>();
    domains::<f64>();
}

#[test]
fn sampler_limits_are_not_law_limits() {
    let large = Poisson::new(f64::MAX).unwrap();
    assert_eq!(large.pmf(0).unwrap(), 0.0);
    let error = large.prepare_sampler().unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::NotImplemented
    );
    assert!(Poisson::new(f32::MAX).is_ok());
    if usize::BITS > 52 {
        let max = ((1_u64 << 52) - 1) as usize;
        assert!(Binomial::new(max, 0.5).unwrap().prepare_sampler().is_ok());
        let error = Binomial::new(max + 1, 0.5)
            .unwrap()
            .prepare_sampler()
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::NotImplemented
        );
        let lambda = (1_u64 << 51) as f64;
        assert!(Poisson::new(lambda).unwrap().prepare_sampler().is_ok());
        let error = Poisson::new(lambda.next_up())
            .unwrap()
            .prepare_sampler()
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::NotImplemented
        );
    }
    // The old f32 precision cap no longer rejects this sampler.
    assert!(
        Poisson::new(16_777_216_f32)
            .unwrap()
            .prepare_sampler()
            .is_ok()
    );
}

/// The three forms of `law` draw what its prepared sampler draws from the same seed.
fn law_draws_through_its_sampler<L: CountDistribution, P: CountDistribution<Count = L::Count>>(
    law: &L,
    sampler: &P,
) where
    L::Count: PartialEq + core::fmt::Debug,
{
    let reference = |seed| {
        sampler
            .sample_n(&mut SplitMix64::from_seed(seed), 64)
            .unwrap()
    };
    let mut rng = SplitMix64::from_seed(2469);
    let singles: Vec<_> = (0..64).map(|_| law.sample(&mut rng).unwrap()).collect();
    assert_eq!(singles, reference(2469));
    assert_eq!(
        law.sample_n(&mut SplitMix64::from_seed(2470), 64).unwrap(),
        reference(2470)
    );
    let mut output = vec![L::Count::default(); 64];
    law.sample_n_into(
        &mut SplitMix64::from_seed(2471),
        SamplingPolicy::rollback(OutputPolicy::Atomic),
        &mut output,
        &mut SamplingWorkspace::new(),
    )
    .unwrap();
    assert_eq!(output, reference(2471));
}

#[test]
fn prepared_laws_draw_through_their_samplers() {
    for (n, p) in [(16, 0.5), (1_000, 0.001), (1_000, 0.4), (1_000, 0.9)] {
        let law = Binomial::new(n, p).unwrap();
        law_draws_through_its_sampler(&law, &law.prepare_sampler().unwrap());
        let law = Binomial::new(n, p as f32).unwrap();
        law_draws_through_its_sampler(&law, &law.prepare_sampler().unwrap());
    }
    for rate in [0.0, 5.0, 30.0] {
        let law = Poisson::new(rate).unwrap();
        law_draws_through_its_sampler(&law, &law.prepare_sampler().unwrap());
    }
    for (mu1, mu2) in [(3.0, 1.0), (30.0, 20.0), (0.0, 4.0)] {
        let law = Skellam::new(mu1, mu2).unwrap();
        law_draws_through_its_sampler(&law, &law.prepare_sampler().unwrap());
    }
}

#[test]
fn prepared_law_preparation_errors_precede_drawing() {
    let mut rng = Words { word: 0, draws: 0 };
    let mut output = [0_usize; 4];
    let policy = SamplingPolicy::advance_on_error(OutputPolicy::Incremental);
    let not_implemented = |error: scioxide_core::SciOxideError| {
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::NotImplemented
        );
    };
    if usize::BITS > 52 {
        let law = Binomial::new(1_usize << 52, 0.5).unwrap();
        not_implemented(law.sample(&mut rng).unwrap_err());
        not_implemented(law.sample_n(&mut rng, 4).unwrap_err());
        let mut workspace = SamplingWorkspace::new();
        not_implemented(
            law.sample_n_into(&mut rng, policy, &mut output, &mut workspace)
                .unwrap_err(),
        );
    }
    let law = Poisson::new(f64::MAX).unwrap();
    not_implemented(law.sample(&mut rng).unwrap_err());
    not_implemented(law.sample_n(&mut rng, 4).unwrap_err());
    let law = Skellam::new(1.0, f64::MAX).unwrap();
    not_implemented(law.sample(&mut rng).unwrap_err());
    assert_eq!(rng.draws, 0);
}

#[test]
fn nonzero_subnormal_probabilities_are_not_rounded_into_zero_laws() {
    for p in [
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        f64::from(f32::from_bits(1)),
    ] {
        for n in [1, 16, 1000] {
            let mut rng = Words { word: 0, draws: 0 };
            let sample = Binomial::new(n, p)
                .unwrap()
                .prepare_sampler()
                .unwrap()
                .sample(&mut rng)
                .unwrap();
            assert_eq!(sample, if n <= 16 { n } else { 1 });
            assert!(rng.draws <= if n <= 16 { n * 17 } else { 18 });
        }
        let mut rng = Words { word: 0, draws: 0 };
        assert_eq!(
            Poisson::new(p)
                .unwrap()
                .prepare_sampler()
                .unwrap()
                .sample(&mut rng)
                .unwrap(),
            1
        );
        assert!(rng.draws <= 18);
    }
    let mut rng = Words { word: 0, draws: 0 };
    Binomial::new(1, f64::from_bits(1))
        .unwrap()
        .prepare_sampler()
        .unwrap()
        .sample(&mut rng)
        .unwrap();
    assert_eq!(rng.draws, 17);
}

#[test]
fn widths_share_raw_stream_for_identical_law_parameters() {
    for p in [f32::from_bits(1), 0.25, 0.5, 1.0 - f32::EPSILON] {
        for n in [0, 16, 17, 1000, 1_000_000] {
            let a = Binomial::new(n, p).unwrap().prepare_sampler().unwrap();
            let b = Binomial::new(n, f64::from(p))
                .unwrap()
                .prepare_sampler()
                .unwrap();
            let mut x = SplitMix64::from_seed(527);
            let mut y = x.clone();
            for _ in 0..100 {
                assert_eq!(a.sample(&mut x).unwrap(), b.sample(&mut y).unwrap());
            }
            assert_eq!(x.position(), y.position());
        }
    }
    for rate in [f32::from_bits(1), 0.0, 11.999, 12.0, 16_777_216.0] {
        let a = Poisson::new(rate).unwrap().prepare_sampler().unwrap();
        let b = Poisson::new(f64::from(rate))
            .unwrap()
            .prepare_sampler()
            .unwrap();
        let mut x = SplitMix64::from_seed(527);
        let mut y = x.clone();
        for _ in 0..100 {
            assert_eq!(a.sample(&mut x).unwrap(), b.sample(&mut y).unwrap());
        }
        assert_eq!(x.position(), y.position());
    }
}

#[test]
fn independent_decimal_probability_fixtures() {
    for line in include_str!("../data/count_laws.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let parts: Vec<_> = line.split_whitespace().collect();
        let n: u64 = parts[1].parse().unwrap();
        let k: u64 = parts[3].parse().unwrap();
        if n > usize::MAX as u64 || k > usize::MAX as u64 {
            continue;
        }
        let p: f64 = parts[2].parse().unwrap();
        let expected: f64 = parts[4].parse().unwrap();
        let (mass, cdf) = if parts[0] == "b" {
            let law = Binomial::new(n as usize, p).unwrap();
            (
                law.pmf(k as usize).unwrap(),
                if parts[5] == "-" {
                    None
                } else {
                    Some(law.cdf(k as usize).unwrap())
                },
            )
        } else {
            let law = Poisson::new(p).unwrap();
            (
                law.pmf(k as usize).unwrap(),
                if parts[5] == "-" {
                    None
                } else {
                    Some(law.cdf(k as usize).unwrap())
                },
            )
        };
        assert!(
            (mass - expected).abs() <= 5e-12 * expected,
            "{line}: mass={mass}"
        );
        if f64::from(p as f32) == p {
            let actual = if parts[0] == "b" {
                Binomial::new(n as usize, p as f32)
                    .unwrap()
                    .pmf(k as usize)
                    .unwrap()
            } else {
                Poisson::new(p as f32).unwrap().pmf(k as usize).unwrap()
            };
            assert!(
                (f64::from(actual) - expected).abs()
                    <= 2e-6 * expected + f64::from(f32::from_bits(1)),
                "f32: {line}"
            );
        }
        if let Some(cdf) = cdf {
            let expected: f64 = parts[5].parse().unwrap();
            assert!(
                (cdf - expected).abs() <= 5e-12 * expected,
                "{line}: cdf={cdf}"
            );
        }
    }
}

#[test]
fn large_counts_have_calibrated_center_variance_and_residues() {
    if usize::BITS <= 52 {
        return;
    }
    let n = ((1_u64 << 52) - 1) as usize;
    let binomial = Binomial::new(n, 0.5).unwrap().prepare_sampler().unwrap();
    let rate = (1_u64 << 51) as f64;
    let poisson = Poisson::new(rate).unwrap().prepare_sampler().unwrap();
    for family in 0..2 {
        let mut rng = SplitMix64::from_seed(0x527);
        let mean = if family == 0 { n as f64 * 0.5 } else { rate };
        let variance = if family == 0 { n as f64 * 0.25 } else { rate };
        let mut residues = [0_u32; 8];
        let (sum, squares) = (0..100_000).fold((0.0, 0.0), |(sum, squares), _| {
            let k = if family == 0 {
                binomial.sample(&mut rng).unwrap()
            } else {
                poisson.sample(&mut rng).unwrap()
            };
            if family == 0 {
                assert!(k <= n);
            }
            residues[k % 8] += 1;
            let z = (k as f64 - mean) / variance.sqrt();
            (sum + z, squares + z * z)
        });
        assert!((sum / 100_000.0).abs() < 0.026);
        assert!((squares / 100_000.0 - 1.0).abs() < 0.036);
        for count in residues {
            assert!((f64::from(count) - 12_500.0).abs() < 850.0);
        }
    }
}
