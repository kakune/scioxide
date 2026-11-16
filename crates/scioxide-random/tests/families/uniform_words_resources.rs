//! Resource contracts for uniform words.
#![allow(clippy::unwrap_used)]
use scioxide_random::*;

#[test]
fn restricted_engines_sample_with_reused_storage() {
    fn check<B: BitGenerator + SeedableRng + Rng<f32> + Rng<f64>>() {
        let mut rng = B::from_seed(12345);
        let normal32 = Normal::new(0.0_f32, 1.0).unwrap();
        let normal64 = Normal::new(0.0_f64, 1.0).unwrap();
        let binomial = Binomial::new(100, 0.3_f64)
            .unwrap()
            .prepare_sampler()
            .unwrap();
        let poisson = Poisson::new(30.0_f64).unwrap().prepare_sampler().unwrap();
        for _ in 0..128 {
            normal32.sample(&mut rng).unwrap();
            normal64.sample(&mut rng).unwrap();
            binomial.sample(&mut rng).unwrap();
            poisson.sample(&mut rng).unwrap();
        }
    }
    check::<Ranlux>();
    check::<Ranlxs>();
    check::<Ranlxd>();
    check::<Mrg>();
    check::<Cmrg>();
    check::<Mrg32k3a>();
}
