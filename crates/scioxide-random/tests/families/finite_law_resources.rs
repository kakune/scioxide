//! Allocation, provider and exact resource boundaries for finite laws.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Budget, SciOxideError, Tolerance};
use scioxide_random::*;

#[test]
fn finite_tables_admit_each_buffer_before_allocation() {
    let tiny = LawOptions::<f64>::default().with_max_bytes(7);
    for result in [
        PoissonBinomial::with_options(&[0.0], tiny).map(|_| ()),
        FisherNoncentralHypergeometric::with_options(10, 10, 3, 2.0, tiny).map(|_| ()),
        WalleniusNoncentralHypergeometric::with_options(10, 10, 3, 2.0, tiny).map(|_| ()),
    ] {
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::ResourceLimit
        );
    }
    let limit = LawOptions::<f64>::default().with_max_bytes(24);
    assert!(PoissonBinomial::with_options(&[0.2, 0.3], limit).is_ok());
    assert!(PoissonBinomial::with_options(&[0.2, 0.3], limit.with_max_bytes(23)).is_err());
    assert!(FisherNoncentralHypergeometric::with_options(10, 5, 2, 2.0, limit).is_ok());
    assert!(WalleniusNoncentralHypergeometric::with_options(10, 5, 2, 2.0, limit).is_ok());
    assert!(
        FisherNoncentralHypergeometric::with_options(10, 5, 2, 2.0, limit.with_max_bytes(23))
            .is_err()
    );
    assert!(
        WalleniusNoncentralHypergeometric::with_options(10, 5, 2, 2.0, limit.with_max_bytes(23))
            .is_err()
    );
}
struct Zero {
    calls: usize,
}
impl BitGenerator for Zero {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        self.calls += 1;
        Ok(0)
    }
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        Ok(self.next_u64()? as u32)
    }
    fn fill_bytes(&mut self, out: &mut [u8]) -> Result<(), SciOxideError> {
        out.fill(0);
        Ok(())
    }
}
#[test]
fn prepared_operations_and_raw_word_limits() {
    macro_rules! check {
        ($law:expr,$k:expr) => {{
            let law = $law.unwrap();
            let mut rng = SplitMix64::from_seed(533);
            {
                let sampler = &law;
                for _ in 0..16 {
                    law.log_pmf($k).unwrap();
                    law.log_cdf($k).unwrap();
                    law.log_sf($k).unwrap();
                    law.mean().unwrap();
                    law.variance().unwrap();
                    law.quantile(0.25).unwrap();
                    law.isf(0.25).unwrap();
                    sampler.sample(&mut rng).unwrap();
                }
            };
        }};
    }
    check!(Boltzmann::new(0.5, 16), 3);
    check!(DiscreteLaplace::new(0.5), -2);
    check!(DiscreteUniform::<f64>::new(-10, 11), -2);
    check!(Hypergeometric::<f64>::new(50, 20, 12), 5);
    check!(FisherNoncentralHypergeometric::new(50, 20, 12, 2.0), 5);
    check!(WalleniusNoncentralHypergeometric::new(50, 20, 12, 2.0), 5);
    check!(Geometric::from_rate(0.5), 4);
    check!(BetaBinomial::new(20, 12.0, 19.0), 5);
    let options = LawOptions::default()
        .with_tolerance(Tolerance::from_sqrt_epsilon())
        .with_sample_word_budget(Budget::ONE);
    let u = DiscreteUniform::<f64>::with_options(0, 3, options).unwrap();
    let mut zero = Zero { calls: 0 };
    let error = u.sample(&mut zero).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(zero.calls, 1);
    let point = Hypergeometric::<f64>::new(usize::MAX, usize::MAX, 3).unwrap();
    assert_eq!(point.sample(&mut zero).unwrap(), 3);
    assert_eq!(zero.calls, 1);
    let point = DiscreteUniform::<f64>::new(i64::MIN, i64::MIN + 1).unwrap();
    assert_eq!(point.sample(&mut zero).unwrap(), i64::MIN);
    assert_eq!(zero.calls, 1);
}
#[test]
fn matching_widths_and_color_reflection() {
    let a = WalleniusNoncentralHypergeometric::new(50, 20, 12, 2.0_f32).unwrap();
    let b = WalleniusNoncentralHypergeometric::new(50, 20, 12, 2.0_f64).unwrap();
    let reflected = WalleniusNoncentralHypergeometric::new(50, 30, 12, 0.5_f64).unwrap();
    for k in 0..=12 {
        assert!((b.pmf(k).unwrap() - reflected.pmf(12 - k).unwrap()).abs() < 1e-14);
    }
    let mut r1 = SplitMix64::from_seed(533);
    let mut r2 = r1.clone();
    for _ in 0..1024 {
        assert_eq!(a.sample(&mut r1).unwrap(), b.sample(&mut r2).unwrap());
    }
}
#[test]
fn provider_error_is_not_retried() {
    struct Fail(usize);
    impl BitGenerator for Fail {
        #[inline]
        fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.next_u32()
        }
        #[inline]
        fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.next_u64()
        }
        fn next_u64(&mut self) -> Result<u64, SciOxideError> {
            self.0 += 1;
            Err(SciOxideError::not_implemented("injected provider failure"))
        }
        fn next_u32(&mut self) -> Result<u32, SciOxideError> {
            self.next_u64().map(|v| v as u32)
        }
        fn fill_bytes(&mut self, _: &mut [u8]) -> Result<(), SciOxideError> {
            self.next_u64().map(|_| ())
        }
    }
    let law = Hypergeometric::<f64>::new(50, 20, 12).unwrap();
    let mut rng = Fail(0);
    let error = law.sample(&mut rng).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::NotImplemented
    );
    assert_eq!(rng.0, 1);
}

#[test]
fn maximum_native_count_point_masses_preserve_exact_endpoints() {
    macro_rules! check {
        ($law:expr) => {{
            let law = $law.unwrap();
            let mut rng = Zero { calls: 0 };
            assert_eq!(law.count_support(), (usize::MAX, Some(usize::MAX)));
            assert_eq!(law.quantile(0.5).unwrap(), usize::MAX);
            assert_eq!(law.isf(0.5).unwrap(), usize::MAX);
            assert_eq!(law.pmf(usize::MAX).unwrap(), 1.0);
            assert_eq!(law.sample(&mut rng).unwrap(), usize::MAX);
            assert_eq!(rng.calls, 0);
        }};
    }
    check!(Hypergeometric::<f64>::new(
        usize::MAX,
        usize::MAX,
        usize::MAX
    ));
    check!(FisherNoncentralHypergeometric::new(
        usize::MAX,
        usize::MAX,
        usize::MAX,
        2.0_f64
    ));
    check!(WalleniusNoncentralHypergeometric::new(
        usize::MAX,
        usize::MAX,
        usize::MAX,
        2.0_f64
    ));
}
