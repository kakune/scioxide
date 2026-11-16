//! Deterministic composition checks, including every built-in engine and luxury level.
#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use scioxide_random::*;

const DRAWS: usize = 20_000;

fn cdf_case<T: PartialOrd + Copy>(
    engine: &str,
    seed: u64,
    case: &str,
    cuts: &[(T, f64)],
    draw: impl FnMut() -> T,
) {
    statistics::cdf_fit(&format!("{engine}/{seed}/{case}"), cuts, DRAWS, 3e-6, draw);
}

fn cuts<T: Copy, D: Quantile<f64, T> + Cdf<f64, T>>(law: &D) -> Vec<(T, f64)> {
    (1..20)
        .map(|i| {
            let cut = law.quantile(f64::from(i) / 20.0).unwrap();
            (cut, law.cdf(cut).unwrap())
        })
        .collect()
}

fn frequencies(engine: &str, seed: u64, case: &str, counts: &[usize]) {
    let probabilities = vec![1.0 / counts.len() as f64; counts.len()];
    statistics::assert_fit(
        &format!("{engine}/{seed}/{case}"),
        counts,
        &probabilities,
        0.0,
    );
}

fn matrix<B: BitGenerator + Clone + Rng<f32> + Rng<f64>>(name: &str, seed: u64, engine: B) {
    let normal = Normal::new(0.0_f64, 1.0).unwrap();
    let normal_cuts = cuts(&normal);
    let mut rng = engine.clone();
    cdf_case(name, seed, "Normal f64", &normal_cuts, || {
        let x = normal.sample(&mut rng).unwrap();
        assert!(x.is_finite());
        x
    });
    let normal = Normal::new(0.0_f32, 1.0).unwrap();
    let mut rng = engine.clone();
    cdf_case(name, seed, "Normal f32", &normal_cuts, || {
        let x = normal.sample(&mut rng).unwrap();
        assert!(x.is_finite());
        f64::from(x)
    });
    let gamma = Gamma::new(3.0_f64, 1.0).unwrap();
    let mut rng = engine.clone();
    cdf_case(name, seed, "Gamma", &cuts(&gamma), || {
        let x = gamma.sample(&mut rng).unwrap();
        assert!(x.is_finite() && x >= 0.0);
        x
    });
    for (case, probability) in [("Binomial BINV", 0.05), ("Binomial BTPE", 0.3)] {
        let law = Binomial::new(100, probability).unwrap();
        let sampler = law.prepare_sampler().unwrap();
        let mut rng = engine.clone();
        cdf_case(name, seed, case, &cuts(&law), || {
            sampler.sample(&mut rng).unwrap()
        });
    }
    for (case, rate) in [("Poisson inversion", 5.0), ("Poisson PTRS", 30.0)] {
        let law = Poisson::new(rate).unwrap();
        let sampler = law.prepare_sampler().unwrap();
        let mut rng = engine.clone();
        cdf_case(name, seed, case, &cuts(&law), || {
            sampler.sample(&mut rng).unwrap()
        });
    }
    let law = Hypergeometric::<f64>::new(1000, 300, 100).unwrap();
    let mut rng = engine.clone();
    cdf_case(name, seed, "Hypergeometric", &cuts(&law), || {
        law.sample(&mut rng).unwrap()
    });

    let mut rng = engine.clone();
    let mut counts = [0; 1000];
    for _ in 0..DRAWS {
        counts[uniform_index(1000, &mut rng).unwrap()] += 1;
    }
    frequencies(name, seed, "uniform_index", &counts);

    let mut rng = engine;
    let mut counts = [0; 120];
    for _ in 0..DRAWS {
        let mut permutation = [0, 1, 2, 3, 4];
        shuffle(&mut permutation, &mut rng).unwrap();
        counts[statistics::permutation_rank(&permutation)] += 1;
    }
    frequencies(name, seed, "shuffle", &counts);
}

macro_rules! engine {
    ($test:ident, $type:ty) => {
        #[test]
        fn $test() {
            for seed in [42, 12345] {
                matrix(stringify!($type), seed, <$type>::from_seed(seed));
            }
        }
    };
}
engine!(splitmix, SplitMix64);
engine!(lcg, Lcg64);
engine!(pcg, Pcg64);
engine!(xoshiro128, Xoshiro128StarStar);
engine!(xoshiro256, Xoshiro256StarStar);
engine!(mt19937, Mt19937);
engine!(taus2, Taus2);
engine!(taus113, Taus113);
engine!(gfsr4, Gfsr4);
engine!(mrg, Mrg);
engine!(cmrg, Cmrg);
engine!(mrg32k3a, Mrg32k3a);
engine!(philox, Philox4x64);
engine!(threefry, Threefry4x64);

#[test]
fn ranlux() {
    for seed in [42, 12345] {
        for level in 0..=4 {
            matrix(
                &format!("Ranlux/{level}"),
                seed,
                Ranlux::with_luxury(seed, RanluxLuxury::try_from(level).unwrap()),
            );
        }
    }
}
#[test]
fn ranlxs() {
    for seed in [42, 12345] {
        for level in 0..=2 {
            matrix(
                &format!("Ranlxs/{level}"),
                seed,
                Ranlxs::with_luxury(seed, RanlxsLuxury::try_from(level).unwrap()),
            );
        }
    }
}
#[test]
fn ranlxd() {
    for seed in [42, 12345] {
        for level in 1..=2 {
            matrix(
                &format!("Ranlxd/{level}"),
                seed,
                Ranlxd::with_luxury(seed, RanlxdLuxury::try_from(level).unwrap()),
            );
        }
    }
}

#[path = "../support/statistics.rs"]
mod statistics;
