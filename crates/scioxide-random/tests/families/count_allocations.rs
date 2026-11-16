//! Preparation, analytics, and successful draws need no heap storage.
#![allow(clippy::unwrap_used)]

use scioxide_core::RealScalar;
use scioxide_random::{Binomial, CountDistribution, Mass, Poisson, SeedableRng, SplitMix64};
fn check<S: RealScalar>() {
    let mut rng = SplitMix64::from_seed(527);
    for (n, p) in [(0, 0.25), (16, 0.5), (1000, 0.001), (1_000_000, 0.5)] {
        let p = S::try_from_f64_lossy(p).unwrap();
        let law = Binomial::new(n, p).unwrap();
        let sampler = law.prepare_sampler().unwrap();
        sampler.sample(&mut rng).unwrap();
        law.pmf(n / 2).unwrap();
    }
    for rate in [0.0, 0.001, 12.0, 1e9] {
        let rate = S::try_from_f64_lossy(rate).unwrap();
        let law = Poisson::new(rate).unwrap();
        let sampler = law.prepare_sampler().unwrap();
        sampler.sample(&mut rng).unwrap();
        law.pmf(0).unwrap();
    }
}
#[test]
fn count_operations_execute_f32_f64() {
    check::<f32>();
    check::<f64>();
}

#[test]
fn prepared_compound_count_queries_and_draws_execute() {
    use scioxide_random::{
        BetaBinomial, BetaNegativeBinomial, Geometric, LogMass, LogSeries, LogSurvival,
        NegativeBinomial, PoissonBinomial, Quantile, Skellam, YuleSimon, Zipf, Zipfian,
    };
    fn width<S: RealScalar>() {
        let s = |x| S::try_from_f64_lossy(x).unwrap();
        let mut rng = SplitMix64::from_seed(532);
        macro_rules! check {
            (prepared $law:expr,$k:expr) => {{
                let law = $law.unwrap();
                let sampler = law.prepare_sampler().unwrap();
                check!(@run law, sampler, $k);
            }};
            ($law:expr,$k:expr) => {{
                let law = $law.unwrap();
                check!(@run law, &law, $k);
            }};
            (@run $law:ident, $sampler:expr, $k:expr) => {{
                let law = &$law;
                let sampler = $sampler;
                law.log_pmf($k).unwrap();
                law.log_sf($k).unwrap();
                law.quantile(s(0.75)).unwrap();
                sampler.sample(&mut rng).unwrap();
            }};
        }
        check!(BetaBinomial::new(20, s(2.0), s(3.0)), 8);
        check!(BetaBinomial::new(100, s(0.125), s(0.25)), 50);
        check!(BetaNegativeBinomial::new(s(2.0), s(3.0), s(2.0)), 3);
        check!(Geometric::new(s(0.25)), 4);
        check!(LogSeries::new(s(0.5)), 5);
        check!(NegativeBinomial::new(s(2.5), s(0.25)), 5);
        check!(PoissonBinomial::new(&[s(0.25), s(0.5), s(0.75)]), 1);
        check!(prepared Skellam::new(s(3.0), s(2.0)), 0);
        check!(YuleSimon::new(s(4.0)), 10);
        check!(Zipf::new(s(4.0)), 10);
        check!(Zipfian::new(s(0.5), 100), 50);
    }
    width::<f32>();
    width::<f64>();
}

#[test]
fn large_central_urn_and_zipfian_preparation_and_queries_execute() {
    use scioxide_random::{Cdf, Hypergeometric, Quantile, Zipfian};
    let mut rng = SplitMix64::from_seed(1580);
    let urn = Hypergeometric::<f64>::new(200_000, 100_000, 100_000).unwrap();
    let zipf = Zipfian::new(1.1_f64, 100_000).unwrap();
    {
        urn.pmf(50_000).unwrap();
        urn.cdf(50_000).unwrap();
        urn.quantile(0.5).unwrap();
        urn.sample(&mut rng).unwrap();
    };
    {
        zipf.pmf(100).unwrap();
        zipf.cdf(100).unwrap();
        zipf.quantile(0.5).unwrap();
        zipf.sample(&mut rng).unwrap();
    };
}
