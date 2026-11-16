#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::statistics;
use scioxide_core::{Budget, RealScalar, SciOxideError, Tolerance};
use scioxide_random::*;

fn check<S: RealScalar, X: Copy + core::fmt::Debug + PartialEq, L>(
    law: L,
    k: X,
    row: &[&str],
    parse: impl Fn(&str) -> X,
) where
    L: Mass<S, X>
        + LogMass<S, X>
        + Cdf<S, X>
        + Survival<S, X>
        + LogCdf<S, X>
        + LogSurvival<S, X>
        + Quantile<S, X>
        + InverseSurvival<S, X>,
{
    if row[0].starts_with("q:") {
        let p = S::try_from_f64_lossy(row[4].parse().unwrap()).unwrap();
        let expected = parse(row[5]);
        assert_eq!(
            law.quantile(p).unwrap_or_else(|e| panic!("{row:?}: {e:?}")),
            expected,
            "{row:?}"
        );
        assert_eq!(
            law.isf(S::ONE - p)
                .unwrap_or_else(|e| panic!("{row:?}: {e:?}")),
            expected,
            "{row:?}"
        );
        return;
    }
    let logs = [law.log_pmf(k), law.log_cdf(k), law.log_sf(k)];
    let probs = [law.pmf(k), law.cdf(k), law.sf(k)];
    let tolerance = if S::EPSILON.into_f64() > 1e-10 {
        2e-6
    } else {
        5e-12
    };
    for (i, (log, prob)) in logs.into_iter().zip(probs).enumerate() {
        // Skellam tails use the explicitly requested scalar-precision floor.
        // Its mass is evaluated independently by the scaled Bessel kernel.
        let tolerance = if row[0] == "skellam" && i > 0 {
            tolerance.max(64.0 * S::EPSILON.into_f64())
        } else {
            tolerance
        };
        let expected: f64 = row[i + 5].parse().unwrap();
        let actual = log
            .unwrap_or_else(|e| panic!("{row:?}, log index {i}: {e:?}"))
            .into_f64();
        if expected == f64::NEG_INFINITY {
            assert_eq!(actual, expected, "{row:?}");
        } else {
            assert!(
                (actual - expected).abs()
                    <= tolerance + 2.0 * S::EPSILON.into_f64() * expected.abs(),
                "{row:?}: log {i}, got {actual}, expected {expected}"
            );
        }
        let actual = prob.unwrap().into_f64();
        let expected = expected.exp();
        let subnormal = if S::EPSILON.into_f64() > 1e-10 {
            f64::from(f32::from_bits(1))
        } else {
            f64::from_bits(1)
        };
        assert!(
            (actual - expected).abs() <= tolerance * expected + subnormal,
            "{row:?}: probability {i}, got {actual}, expected {expected}"
        );
    }
}
fn fixtures<S: RealScalar>() {
    // These references test accuracy near scalar precision, rather than the
    // shared square-root-epsilon default used for ordinary law queries.
    let reference_options = LawOptions::default().with_tolerance(
        Tolerance::new(S::ZERO, S::try_from_f64_lossy(64.0).unwrap() * S::EPSILON).unwrap(),
    );
    for row in include_str!("../data/discrete_count.tsv")
        .lines()
        .filter(|s| !s.starts_with('#') && !s.is_empty())
    {
        let r: Vec<_> = row.split('\t').collect();
        let p = |i: usize| S::try_from_f64_lossy(r[i].parse().unwrap()).unwrap();
        let k = if r[0].starts_with("q:") {
            0
        } else {
            r[4].parse::<i64>().unwrap()
        };
        let name = r[0].strip_prefix("q:").unwrap_or(r[0]);
        macro_rules! test {
            ($law:expr) => {
                check::<S, usize, _>($law.unwrap(), usize::try_from(k).unwrap(), &r, |s| {
                    s.parse().unwrap()
                })
            };
        }
        match name {
            "bernoulli" | "binom" => test!(Binomial::new(r[1].parse().unwrap(), p(2))),
            "poisson" => test!(Poisson::new(p(1))),
            "betabinom" => test!(BetaBinomial::new(r[1].parse().unwrap(), p(2), p(3))),
            "betanbinom" => test!(BetaNegativeBinomial::new(p(1), p(2), p(3))),
            "nbinom" => test!(NegativeBinomial::new(p(1), p(2))),
            "geom" => test!(Geometric::new(p(1))),
            "logser" => test!(LogSeries::new(p(1))),
            "yulesimon" => test!(YuleSimon::new(p(1))),
            "zipf" => test!(Zipf::new(p(1))),
            "zipfian" => test!(Zipfian::new(p(1), r[2].parse().unwrap())),
            "poisson_binom" => test!(PoissonBinomial::new(
                &[0.125, 0.25, 0.5, 0.75, 0.875].map(|p| S::try_from_f64_lossy(p).unwrap())
            )),
            "skellam" => check::<S, i64, _>(
                Skellam::with_options(p(1), p(2), reference_options).unwrap(),
                k,
                &r,
                |s| s.parse().unwrap(),
            ),
            _ => panic!("unknown fixture {name}"),
        }
    }
}
#[test]
fn independent_f64_fixtures() {
    fixtures::<f64>();
}
#[test]
fn independent_f32_fixtures() {
    fixtures::<f32>();
}

struct Words {
    word: u64,
    draws: usize,
    fail: bool,
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
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        if self.fail {
            return Err(SciOxideError::not_implemented("injected failure"));
        }
        self.draws += 1;
        Ok(self.word)
    }
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        Ok((self.next_u64()? >> 32) as u32)
    }
    fn fill_bytes(&mut self, output: &mut [u8]) -> Result<(), SciOxideError> {
        output.fill(0);
        Ok(())
    }
}
#[test]
fn domains_support_moments_and_explicit_limits() {
    for p in [0.0, -1.0, 1.1, f64::INFINITY, f64::NAN] {
        assert!(Geometric::new(p).is_err());
    }
    assert!(LogSeries::new(1.0_f64).is_err());
    assert!(NegativeBinomial::new(0.0_f64, 0.5).is_err());
    assert!(BetaBinomial::new(5, -1.0_f64, 2.0).is_err());
    assert!(BetaNegativeBinomial::new(-1.0_f64, 2.0, 3.0).is_err());
    assert!(YuleSimon::new(0.0_f64).is_err());
    assert!(Zipf::new(1.0_f64).is_err());
    assert!(Zipfian::new(1.0_f64, 0).is_err());
    assert!(PoissonBinomial::new(&[0.5, f64::NAN]).is_err());
    assert!(Skellam::new(-1.0_f64, 1.0).is_err());
    let error = YuleSimon::new(1.0_f64).unwrap().mean().unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    let error = YuleSimon::new(2.0_f64).unwrap().variance().unwrap_err();
    assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    assert!(Zipf::new(2.0_f64).unwrap().mean().is_err());
    assert!(Zipf::new(3.0_f64).unwrap().variance().is_err());
    assert!(
        BetaNegativeBinomial::new(1.0_f64, 1.0, 2.0)
            .unwrap()
            .mean()
            .is_err()
    );
    assert!(
        BetaNegativeBinomial::new(1.0_f64, 2.0, 2.0)
            .unwrap()
            .variance()
            .is_err()
    );
    assert_eq!(
        BetaNegativeBinomial::new(0.0_f64, 0.1, 0.1)
            .unwrap()
            .variance()
            .unwrap(),
        0.0
    );
    assert_eq!(
        PoissonBinomial::new(&[0.0, 1.0, 0.25, 1.0])
            .unwrap()
            .count_support(),
        (2, Some(3))
    );
    assert_eq!(
        Skellam::new(0.0_f64, 3.0).unwrap().signed_count_support(),
        (None, Some(0))
    );
    assert!(Geometric::new(0.5_f64).unwrap().quantile(1.0).is_err());
    assert!(Skellam::new(1.0_f64, 1.0).unwrap().isf(0.0).is_err());
    let tiny = LawOptions::default()
        .with_tolerance(Tolerance::from_sqrt_epsilon())
        .with_work_budget(Budget::ONE)
        .with_sample_word_budget(Budget::ONE);
    let error = PoissonBinomial::with_options(&[0.5, 0.25], tiny).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    let error = BetaBinomial::with_options(100, 2.0_f64, 3.0, tiny)
        .unwrap()
        .cdf(50)
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    let mut words = Words {
        word: 0,
        draws: 0,
        fail: false,
    };
    let law = NegativeBinomial::with_options(3.0_f64, 0.5, tiny).unwrap();
    assert!(law.sample(&mut words).is_err());
    assert_eq!(words.draws, 1);
    words.fail = true;
    let error = Geometric::new(0.5_f64)
        .unwrap()
        .sample(&mut words)
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::NotImplemented
    );
    assert_eq!(
        Geometric::new(1.0_f64).unwrap().sample(&mut words).unwrap(),
        1
    );
    assert_eq!(
        BetaBinomial::new(0, 1.0_f64, 1.0)
            .unwrap()
            .sample(&mut words)
            .unwrap(),
        0
    );
    assert_eq!(
        PoissonBinomial::<f64>::new(&[])
            .unwrap()
            .sample(&mut words)
            .unwrap(),
        0
    );
    assert_eq!(
        Skellam::new(0.0_f64, 0.0)
            .unwrap()
            .prepare_sampler()
            .unwrap()
            .sample(&mut words)
            .unwrap(),
        0
    );
}
#[test]
fn signed_poisson_limits_and_reflection() {
    for k in -20..20 {
        let left = Skellam::new(2.0_f64, 3.0).unwrap();
        let right = Skellam::new(3.0_f64, 2.0).unwrap();
        assert!((left.cdf(k).unwrap() - right.sf(-k - 1).unwrap()).abs() < 5e-12);
        assert!((left.pmf(k).unwrap() - right.pmf(-k).unwrap()).abs() < 5e-12);
        let negative = Skellam::new(0.0_f64, 3.0).unwrap();
        if k < 0 {
            assert_eq!(
                negative.cdf(k).unwrap(),
                Poisson::new(3.0_f64)
                    .unwrap()
                    .sf((-k - 1) as usize)
                    .unwrap()
            );
        } else {
            assert_eq!(negative.cdf(k).unwrap(), 1.0);
        }
    }
}
// Seed 0x532_2026; each case uses shared 32768-draw Pearson/DKW bounds.
fn statistical<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let mut rng = SplitMix64::from_seed(0x532_2026);
    macro_rules! test {
        (prepared $law:expr, $count:ty, $lo:expr, $hi:expr) => {{
            let law = $law.unwrap();
            test!(@run law, law.prepare_sampler().unwrap(), $count, $lo, $hi);
        }};
        ($law:expr) => {
            test!($law, usize, 0, 128);
        };
        ($law:expr, $count:ty, $lo:expr, $hi:expr) => {{
            let law = $law.unwrap();
            test!(@run law, &law, $count, $lo, $hi);
        }};
        (@run $law:ident, $sampler:expr, $count:ty, $lo:expr, $hi:expr) => {{
            let law = &$law;
            let sampler = $sampler;
            statistics::discrete(
                stringify!($law),
                $lo,
                $hi,
                |k| {
                    if k < $lo {
                        0.0
                    } else {
                        law.pmf(k as $count).unwrap().into_f64()
                    }
                },
                |k| {
                    if k < $lo {
                        0.0
                    } else {
                        law.cdf(k as $count).unwrap().into_f64()
                    }
                },
                statistics::reference_error::<S>(),
                || sampler.sample(&mut rng).unwrap() as i64,
            );
        }};
    }
    // Point masses, direct trials, BINV/BTPE, and both reflected paths.
    for (n, p) in [
        (0, 0.3),
        (20, 0.0),
        (20, 1.0),
        (1, 0.25),
        (16, 0.25),
        (17, 0.25),
        (100, 0.099),
        (100, 0.1),
        (100, 0.3),
        (100, 0.95),
        (100, 0.7),
    ] {
        test!(prepared Binomial::new(n, s(p)), usize, 0, 128);
    }
    for rate in [0.0, 3.0, 11.999, 12.0, 30.0, 1000.0] {
        test!(prepared Poisson::new(s(rate)), usize, 0, 1200);
    }
    // Beta-binomial uses Bernoulli mixtures for small shapes/large concentration
    // with n<=128, and a beta/binomial composition otherwise.
    for (n, a, b) in [
        (20, 0.3, 0.3),
        (20, 2.0, 3.0),
        (20, 2048.0, 4096.0),
        (129, 0.3, 2.0),
    ] {
        test!(BetaBinomial::new(n, s(a), s(b)), usize, 0, 129);
    }
    for p in [0.25, 0.9, 1.0] {
        test!(NegativeBinomial::new(s(2.5), s(p)));
        test!(Geometric::new(s(p)), usize, 1, 128);
    }
    test!(BetaNegativeBinomial::new(s(2.0), s(3.0), s(2.0)));
    test!(BetaNegativeBinomial::new(s(0.0), s(3.0), s(2.0)));
    for p in [0.05, 0.9] {
        test!(LogSeries::new(s(p)), usize, 1, 128);
    }
    test!(PoissonBinomial::new(&[s(0.25), s(0.5), s(0.75)]));
    test!(PoissonBinomial::new(&[s(0.0), s(1.0)]));
    for (a, b) in [(0.0, 0.0), (3.0, 0.0), (0.0, 2.0), (3.0, 2.0), (30.0, 20.0)] {
        test!(
            prepared Skellam::with_options(
                s(a),
                s(b),
                LawOptions::default()
                    .with_tolerance(Tolerance::new(S::ZERO, s(64.0) * S::EPSILON).unwrap(),)
            ),
            i64,
            -128,
            128
        );
    }
    test!(YuleSimon::new(s(4.0)), usize, 1, 128);
    test!(Zipf::new(s(2.0)), usize, 1, 128);
    for a in [0.0, 0.5, 2.0] {
        test!(Zipfian::new(s(a), 100), usize, 1, 100);
    }
}
#[test]
fn seeded_sampling_f64() {
    statistical::<f64>();
}
#[test]
fn seeded_sampling_f32() {
    statistical::<f32>();
}

#[test]
fn moments_and_point_mass_limits_are_stable() {
    let close = |x: f64, y: f64| assert!((x - y).abs() <= 5e-12 * y.abs(), "{x} versus {y}");
    let bb = BetaBinomial::new(20, 2.0_f64, 3.0).unwrap();
    close(bb.mean().unwrap(), 8.0);
    close(bb.variance().unwrap(), 20.0);
    let bnb = BetaNegativeBinomial::new(2.0_f64, 3.0, 2.0).unwrap();
    close(bnb.mean().unwrap(), 2.0);
    close(bnb.variance().unwrap(), 16.0);
    let nb = NegativeBinomial::new(2.5_f64, 0.25).unwrap();
    close(nb.mean().unwrap(), 7.5);
    close(nb.variance().unwrap(), 30.0);
    let y = YuleSimon::new(4.0_f64).unwrap();
    close(y.mean().unwrap(), 4.0 / 3.0);
    close(y.variance().unwrap(), 8.0 / 9.0);
    let z = Zipf::new(4.0_f64).unwrap();
    close(z.mean().unwrap(), 1.110_626_535_326_148_1);
    close(z.variance().unwrap(), 0.286_326_453_664_502_8);
    close(
        Zipf::new(200.0_f64).unwrap().variance().unwrap(),
        6.223_015_277_861_142e-61,
    );
    close(
        LogSeries::new(1e-30_f64).unwrap().variance().unwrap(),
        5e-31,
    );
    close(
        Zipfian::new(0.0_f64, 100).unwrap().variance().unwrap(),
        833.25,
    );
    assert_eq!(Zipfian::new(200.0_f64, 1).unwrap().variance().unwrap(), 0.0);
    let pb = PoissonBinomial::new(&[0.125, 0.25, 0.5, 0.75, 0.875]).unwrap();
    close(pb.mean().unwrap(), 2.5);
    close(pb.variance().unwrap(), 0.84375);
}
#[test]
fn finite_normalization_and_extreme_log_tails() {
    let bb = BetaBinomial::new(100, 0.125_f64, 0.25).unwrap();
    let z = Zipfian::new(0.5_f64, 100).unwrap();
    for sum in [
        (0..=100).map(|k| bb.pmf(k).unwrap()).sum::<f64>(),
        (1..=100).map(|k| z.pmf(k).unwrap()).sum::<f64>(),
    ] {
        assert!((sum - 1.0).abs() < 5e-12);
    }
    let geom = Geometric::new(0.25_f64).unwrap();
    assert_eq!(geom.sf(3000).unwrap(), 0.0);
    assert!(geom.log_sf(3000).unwrap().is_finite());
    let zipf = Zipf::new(200.0_f64).unwrap();
    assert_eq!(zipf.sf(100).unwrap(), 0.0);
    assert!(zipf.log_sf(100).unwrap().is_finite());
    let subnormal = PoissonBinomial::new(&[f64::from_bits(1)]).unwrap();
    assert_eq!(subnormal.pmf(1).unwrap(), f64::from_bits(1));
    assert_eq!(
        Geometric::new(1.0_f64).unwrap().log_pmf(2).unwrap(),
        f64::NEG_INFINITY
    );
    assert_eq!(
        Skellam::new(0.0_f64, 0.0).unwrap().log_pmf(1).unwrap(),
        f64::NEG_INFINITY
    );
}

#[test]
fn large_beta_shapes_do_not_overflow_the_probability_ratio() {
    let law = BetaBinomial::new(20, f64::MAX, f64::MAX).unwrap();
    assert_eq!(law.mean().unwrap(), 10.0);
    assert_eq!(law.variance().unwrap(), 5.0);
    let law = BetaBinomial::new(20, f32::MAX, f32::MAX).unwrap();
    assert_eq!(law.mean().unwrap(), 10.0);
    assert_eq!(law.variance().unwrap(), 5.0);
}

#[test]
fn equal_parameters_share_prepared_raw_streams_across_widths() {
    macro_rules! same {
        (prepared $a:expr,$b:expr) => {{
            let a = $a.unwrap();
            let b = $b.unwrap();
            same!(@run a.prepare_sampler().unwrap(), b.prepare_sampler().unwrap());
        }};
        ($a:expr,$b:expr) => {{
            let a = $a.unwrap();
            let b = $b.unwrap();
            same!(@run a, b);
        }};
        (@run $a:expr, $b:expr) => {{
            let a = $a;
            let b = $b;
            let mut left = SplitMix64::from_seed(532);
            let mut right = SplitMix64::from_seed(532);
            for _ in 0..100 {
                assert_eq!(a.sample(&mut left).unwrap(), b.sample(&mut right).unwrap());
            }
        }};
    }
    same!(
        BetaBinomial::new(20, 2.0_f32, 3.0),
        BetaBinomial::new(20, 2.0_f64, 3.0)
    );
    same!(
        BetaBinomial::new(100, 0.125_f32, 0.25),
        BetaBinomial::new(100, 0.125_f64, 0.25)
    );
    same!(
        BetaNegativeBinomial::new(2.0_f32, 3.0, 2.0),
        BetaNegativeBinomial::new(2.0_f64, 3.0, 2.0)
    );
    same!(Geometric::new(0.25_f32), Geometric::new(0.25_f64));
    same!(LogSeries::new(0.5_f32), LogSeries::new(0.5_f64));
    same!(
        NegativeBinomial::new(2.5_f32, 0.25),
        NegativeBinomial::new(2.5_f64, 0.25)
    );
    same!(
        PoissonBinomial::new(&[0.25_f32, 0.5, 0.75]),
        PoissonBinomial::new(&[0.25_f64, 0.5, 0.75])
    );
    same!(
        prepared Skellam::new(3.0_f32, 2.0),
        Skellam::new(3.0_f64, 2.0)
    );
    same!(YuleSimon::new(4.0_f32), YuleSimon::new(4.0_f64));
    same!(Zipf::new(4.0_f32), Zipf::new(4.0_f64));
    same!(Zipfian::new(0.5_f32, 100), Zipfian::new(0.5_f64, 100));
}

#[test]
fn integer_shape_increments_do_not_round_silently() {
    if let Ok(limit) = usize::try_from(1_u64 << 53) {
        let error = BetaBinomial::new(limit, 2.0_f64, 3.0)
            .unwrap()
            .pmf(0)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::NotImplemented
        );
        let error = NegativeBinomial::new(2.0_f64, 0.5)
            .unwrap()
            .pmf(limit)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::NotImplemented
        );
        let error = BetaNegativeBinomial::new(2.0_f64, 3.0, 2.0)
            .unwrap()
            .pmf(limit)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            scioxide_core::SciOxideErrorKind::NotImplemented
        );
    }
}

#[test]
fn power_law_lower_tails_remain_positive_near_normalization_poles() {
    // Independent 100-digit positive defining sums through count 1000.
    for (rho, z, y, b) in [
        (
            f64::EPSILON,
            1.662_108_419_908_734e-15,
            1.662_108_419_908_734e-15,
            1.440_507_239_167_966_8e-15,
        ),
        (
            f64::from(f32::EPSILON),
            8.923_372_637_735_532e-7,
            8.923_372_534_253_01e-7,
            7.733_661_316_115_237e-7,
        ),
    ] {
        for (actual, expected) in [
            (Zipf::new(1.0 + rho).unwrap().cdf(1000).unwrap(), z),
            (YuleSimon::new(rho).unwrap().cdf(1000).unwrap(), y),
            (
                BetaNegativeBinomial::new(1.0, rho, 2.0)
                    .unwrap()
                    .cdf(1000)
                    .unwrap(),
                b,
            ),
        ] {
            assert!(
                (actual / expected - 1.0).abs() < 5e-12,
                "{actual} versus {expected}"
            );
        }
    }
}

#[test]
fn small_shape_beta_binomial_sampling_avoids_gamma_underflow() {
    let law = BetaBinomial::new(100, 0.125_f64, 0.25).unwrap();
    let sampler = &law;
    let mut rng = SplitMix64::from_seed(532);
    let mut hits = 0;
    for _ in 0..20_000 {
        let k = sampler.sample(&mut rng).unwrap();
        assert!(k <= 100);
        hits += usize::from(k == 0);
    }
    assert!((hits as f64 / 20_000.0 - law.pmf(0).unwrap()).abs() < 0.025);
    let tiny = BetaBinomial::new(64, f64::MIN_POSITIVE, f64::MIN_POSITIVE).unwrap();
    let sampler = &tiny;
    let mut zeros = 0;
    for _ in 0..20_000 {
        let k = sampler.sample(&mut rng).unwrap();
        assert!(k == 0 || k == 64);
        zeros += usize::from(k == 0);
    }
    assert!((zeros as f64 / 20_000.0 - 0.5).abs() < 0.025);
}

#[test]
fn beta_binomial_large_shapes_and_subnormal_moments_keep_count_increments() {
    let law = BetaBinomial::new(20, f64::MAX, f64::MAX).unwrap();
    let binomial = Binomial::new(20, 0.5_f64).unwrap();
    for k in [0, 10, 20] {
        assert!((law.pmf(k).unwrap() / binomial.pmf(k).unwrap() - 1.0).abs() < 5e-12);
    }
    let mut rng = SplitMix64::from_seed(532);
    let extreme = BetaBinomial::new(20, f64::MAX, 2.0).unwrap();
    for _ in 0..100 {
        assert!(law.sample(&mut rng).unwrap() <= 20);
        assert!(extreme.sample(&mut rng).unwrap() <= 20);
    }
    let tiny = BetaBinomial::new(20, f64::from_bits(1), 2.0).unwrap();
    assert_eq!(tiny.mean().unwrap(), f64::from_bits(10));
    assert_eq!(tiny.variance().unwrap(), f64::from_bits(73));
    assert!(
        BetaBinomial::new(1000, f64::MAX, f64::MAX)
            .unwrap()
            .pmf(0)
            .is_err()
    );
}
