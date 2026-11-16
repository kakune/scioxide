#![allow(clippy::unwrap_used)]
use core::f64::consts::{PI, TAU};
use scioxide_core::{Budget, RealScalar, SciOxideError};
use scioxide_random::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{a} != {b}");
}
fn check<S: RealScalar>() {
    let s = |v| S::try_from_f64_lossy(v).unwrap();
    let tol = if S::MANTISSA_DIGITS == 24 {
        3e-5
    } else {
        2e-10
    };
    for k in [0.0, 0.001, 1.0, 10.0, 1000.0] {
        let circle = VonMises::new(s(0.0), s(k)).unwrap();
        let line = VonMisesLine::new(s(0.0), s(k)).unwrap();
        close(circle.cdf(s(0.0)).unwrap().into_f64(), 0.5, tol);
        close(
            circle.pdf(s(0.2)).unwrap().into_f64(),
            circle.pdf(s(0.2 + TAU)).unwrap().into_f64(),
            tol,
        );
        close(
            circle.winding_cdf(s(0.2 + TAU)).unwrap().into_f64(),
            circle.cdf(s(0.2)).unwrap().into_f64() + 1.0,
            tol,
        );
        assert_eq!(line.pdf(s(4.0)).unwrap(), S::ZERO);
        for p in [0.001, 0.1, 0.5, 0.9, 0.999] {
            let x = circle.quantile(s(p)).unwrap();
            close(circle.cdf(x).unwrap().into_f64(), p, tol);
            let x = circle.isf(s(p)).unwrap();
            close(circle.sf(x).unwrap().into_f64(), p, tol);
        }
        let fisher = VonMisesFisher::new(&[s(1.0), s(0.0)], s(k)).unwrap();
        let x = [s(0.2_f64.cos()), s(0.2_f64.sin())];
        close(
            fisher.log_pdf(&x).unwrap().into_f64(),
            circle.log_pdf(s(0.2)).unwrap().into_f64(),
            tol,
        );
        close(
            fisher.mean_resultant_length().unwrap().into_f64(),
            circle.mean_resultant_length().unwrap().into_f64(),
            tol,
        );
    }
    for rho in [0.0, 0.1, 0.9, 0.999] {
        let law = WrappedCauchy::new(s(0.0), s(rho)).unwrap();
        for p in [0.001, 0.1, 0.5, 0.9, 0.999] {
            let x = law.quantile(s(p)).unwrap();
            close(law.cdf(x).unwrap().into_f64(), p, tol * 4.0);
            let x = law.isf(s(p)).unwrap();
            close(law.sf(x).unwrap().into_f64(), p, tol * 4.0);
        }
        close(law.mean().unwrap().into_f64(), PI, tol);
    }
    let uniform = VonMisesLine::new(s(0.0), s(0.0)).unwrap();
    close(uniform.variance().unwrap().into_f64(), PI * PI / 3.0, tol);
    close(uniform.entropy().unwrap().into_f64(), TAU.ln(), tol);
    close(
        WrappedCauchy::new(s(0.0), s(0.0))
            .unwrap()
            .variance()
            .unwrap()
            .into_f64(),
        PI * PI / 3.0,
        tol,
    );
    for d in [1, 2, 3, 5, 20] {
        let law = UniformDirection::<S>::new(d).unwrap();
        let mut rng = SplitMix64::from_seed(541);
        for _ in 0..50 {
            assert!(
                law.contains_support(&law.sample(&mut rng).unwrap())
                    .unwrap()
            );
        }
    }
    for d in [2, 3, 5, 20] {
        for k in [0.0, 1e-6, 1.0, 1000.0] {
            let mut mu = vec![S::ZERO; d];
            mu[0] = S::ONE;
            let law = VonMisesFisher::new(&mu, s(k)).unwrap();
            let mut rng = SplitMix64::from_seed(541);
            for _ in 0..50 {
                let x = law.sample(&mut rng).unwrap();
                assert!(law.contains_support(&x).unwrap(), "d={d}, k={k}");
            }
            assert!(law.entropy().unwrap().is_finite());
            assert!(law.variance().unwrap().iter().all(|v| *v >= S::ZERO));
        }
    }
}
#[test]
fn both_scalar_widths() {
    check::<f64>();
    check::<f32>();
}

#[test]
fn domains_endpoints_and_budgets() {
    assert!(VonMises::new(0.0, -1.0).is_err());
    assert!(VonMises::new(f64::NAN, 1.0).is_err());
    assert!(VonMises::new(f64::MAX, 1.0).is_err());
    assert!(WrappedCauchy::new(0.0, 1.0).is_err());
    assert!(UniformDirection::<f64>::new(0).is_err());
    assert!(VonMisesFisher::new(&[1.0], 1.0).is_err());
    assert!(VonMisesFisher::new(&[0.0, 0.0], 1.0).is_err());
    assert!(VonMisesFisher::new(&[1.0, f64::NAN], 1.0).is_err());
    let law = VonMises::new(0.0, 1.0).unwrap();
    assert_eq!(law.winding_cdf(-PI).unwrap(), 0.0);
    assert_eq!(law.winding_cdf(PI).unwrap(), 1.0);
    close(law.winding_cdf(PI.next_down()).unwrap(), 1.0, 2e-14);
    close(law.winding_cdf(PI.next_up()).unwrap(), 1.0, 2e-14);
    for p in [-1.0, 2.0, f64::NAN] {
        assert!(law.quantile(p).is_err());
    }
    assert_eq!(law.cdf(law.quantile(0.0).unwrap()).unwrap(), 0.0);
    assert_eq!(law.cdf(law.quantile(1.0).unwrap()).unwrap(), 1.0);
    let uniform = VonMises::new(0.0, 0.0).unwrap();
    assert!(uniform.mean_direction().is_err());
    let options = LawOptions::default().with_work_budget(Budget::ONE);
    assert!(VonMises::with_options(0.0, 1.0, options).is_err());
    let law = WrappedCauchy::with_options(0.0, 0.5, options).unwrap();
    assert!(law.variance().is_err());
    let options = LawOptions::default().with_sample_word_budget(Budget::ONE);
    let law = VonMisesFisher::with_options(&[1.0, 0.0, 0.0], 1.0, options).unwrap();
    assert!(law.sample(&mut SplitMix64::from_seed(1)).is_err());
    let options = LawOptions::default().with_max_bytes(1);
    assert_eq!(
        UniformDirection::<f64>::with_options(3, options)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    let law = VonMisesFisher::new(&[1.0, 0.0, 0.0], 1.0).unwrap();
    assert_eq!(law.pdf(&[0.0, 0.0, 0.0]).unwrap(), 0.0);
    assert!(law.pdf(&[1.0, 0.0]).is_err());
}

#[test]
fn seeded_sampling_evidence() {
    seeded::<f32>();
    seeded::<f64>();
}
fn seeded<S: RealScalar>() {
    // Hoeffding bounds: each bounded [-1,1] mean has false-positive probability
    // <= 2 exp(-N*t^2/2). N=32768,t=.055 and <=1000 assertions give <2e-10.
    let n = statistics::DRAWS;
    let threshold = 0.055;
    let mut rng = SplitMix64::from_seed(541);
    let scalar = |x| S::try_from_f64_lossy(x).unwrap();
    for k in [0.0, 0.001, 1.0, 10.0, 1000.0] {
        let law = VonMises::new(S::ZERO, scalar(k)).unwrap();
        let mut cosine = 0.0;
        let mut sine = 0.0;
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            let x = law.sample(&mut rng).unwrap();
            cosine += x.into_f64().cos();
            sine += x.into_f64().sin();
            values.push(x);
        }
        assert!(
            (cosine / n as f64 - law.mean_resultant_length().unwrap().into_f64()).abs() < threshold
        );
        assert!(sine.abs() / (n as f64) < threshold);
        let mut values = values.into_iter();
        statistics::continuous("angular", &law, || values.next().unwrap());
    }
    for d in [2, 3, 5, 20] {
        for k in [0.0, 0.001, 1.0, 10.0, 1000.0] {
            let mut mu = vec![S::ZERO; d];
            mu[0] = S::ONE;
            let law = VonMisesFisher::new(&mu, scalar(k)).unwrap();
            let mut sums = vec![0.0; d];
            let mut squares = vec![0.0; d];
            for _ in 0..n {
                let x = law.sample(&mut rng).unwrap();
                for ((sum, square), x) in sums.iter_mut().zip(&mut squares).zip(x) {
                    let x = x.into_f64();
                    *sum += x;
                    *square += x * x;
                }
            }
            let means = law.mean().unwrap();
            let variances = law.variance().unwrap();
            for ((sum, square), (mean, variance)) in
                sums.iter().zip(&squares).zip(means.iter().zip(&variances))
            {
                let mean = mean.into_f64();
                let variance = variance.into_f64();
                assert!(
                    (sum / n as f64 - mean).abs() < threshold,
                    "d={d},k={k}, observed={},mean={mean}",
                    sum / n as f64
                );
                assert!((square / n as f64 - mean * mean - variance).abs() < threshold);
            }
        }
    }
    for rho in [0.0, 0.9] {
        let law = WrappedCauchy::new(S::ZERO, scalar(rho)).unwrap();
        let cut = law.quantile(scalar(0.25)).unwrap();
        let mut cosine = 0.0;
        let mut count = 0;
        for _ in 0..n {
            let x = law.sample(&mut rng).unwrap();
            cosine += x.into_f64().cos();
            count += usize::from(x <= cut);
        }
        assert!((cosine / n as f64 - rho).abs() < threshold);
        assert!((count as f64 / n as f64 - 0.25).abs() < threshold * 0.5);
    }
}

struct BadFloat(f64);
impl BitGenerator for BadFloat {
    #[inline]
    fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
        self.next_u32()
    }
    #[inline]
    fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
        self.next_u64()
    }
    fn next_u32(&mut self) -> Result<u32, SciOxideError> {
        Ok(0)
    }
    fn next_u64(&mut self) -> Result<u64, SciOxideError> {
        Ok(0)
    }
    fn fill_bytes(&mut self, out: &mut [u8]) -> Result<(), SciOxideError> {
        out.fill(0);
        Ok(())
    }
}
impl Rng<f64> for BadFloat {
    fn next_float(&mut self) -> Result<f64, SciOxideError> {
        Ok(self.0)
    }
}
#[test]
fn provider_validation_and_atomic_batches() {
    let law = VonMises::new(0.0, 1.0).unwrap();
    for x in [f64::NAN, -0.1, 1.0] {
        assert!(law.sample(&mut BadFloat(x)).is_err());
    }
    let options = LawOptions::default().with_sample_word_budget(Budget::ONE);
    let law = VonMisesFisher::with_options(&[1.0, 0.0, 0.0], 1.0, options).unwrap();
    let mut rng = SplitMix64::from_seed(541);
    let mut before = rng.clone();
    let mut out = [7.0; 6];
    let mut workspace = SamplingWorkspace::new();
    assert!(
        law.sample_n_into(
            &mut rng,
            SamplingPolicy::rollback(OutputPolicy::Atomic),
            &mut out,
            &mut workspace
        )
        .is_err()
    );
    assert_eq!(out, [7.0; 6]);
    assert_eq!(rng.next_u64().unwrap(), before.next_u64().unwrap());
    assert!(
        law.sample_n_into(
            &mut rng,
            SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
            &mut [],
            &mut workspace
        )
        .is_ok()
    );
    assert!(
        law.sample_n_into(
            &mut rng,
            SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
            &mut [0.0; 2],
            &mut workspace
        )
        .is_err()
    );
}

fn fixture_evidence<S: RealScalar>() {
    let scalar = |x: f64| S::try_from_f64_lossy(x).unwrap();
    let tolerance = if S::MANTISSA_DIGITS == 24 { 3e-5 } else { 5e-9 };
    for line in include_str!("../data/circular.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let fields = line.split('\t').collect::<Vec<_>>();
        let value = |i: usize| fields[i].parse::<f64>().unwrap();
        let d = fields[1].parse::<usize>().unwrap();
        let k = scalar(value(2));
        let x = scalar(value(3));
        if fields[0] == "wrapcauchy" && k == S::ONE {
            continue;
        }
        let (pdf, sf, resultant, entropy) = match fields[0] {
            "vonmises" => {
                let law = VonMises::new(scalar(0.0), k).unwrap();
                let logsf = law.log_sf(x).unwrap();
                if value(5) > -25.0 {
                    let q = law.isf(scalar(value(5).exp())).unwrap();
                    close(q.into_f64(), x.into_f64(), tolerance * 2.0);
                }
                (
                    law.log_pdf(x).unwrap(),
                    logsf,
                    law.mean_resultant_length().unwrap(),
                    law.entropy().unwrap(),
                )
            }
            "wrapcauchy" => {
                let law = WrappedCauchy::new(scalar(0.0), k).unwrap();
                // For rho within 1e-15 of one, f32 rounds to the excluded atom.
                (
                    law.log_pdf(x).unwrap(),
                    law.log_sf(x).unwrap(),
                    law.mean_resultant_length().unwrap(),
                    law.entropy().unwrap(),
                )
            }
            "uniform_direction" => {
                let law = UniformDirection::<S>::new(d).unwrap();
                let mut point = vec![S::ZERO; d];
                point[0] = S::ONE;
                (
                    law.log_pdf(&point).unwrap(),
                    S::ZERO,
                    S::ZERO,
                    law.entropy().unwrap(),
                )
            }
            "vonmises_fisher" => {
                let mut mu = vec![S::ZERO; d];
                mu[0] = S::ONE;
                let mut point = mu.clone();
                point[0] = scalar(value(3).cos());
                point[1] = scalar(value(3).sin());
                let law = VonMisesFisher::new(&mu, k).unwrap();
                (
                    law.log_pdf(&point).unwrap(),
                    S::ZERO,
                    law.mean_resultant_length().unwrap(),
                    law.entropy().unwrap(),
                )
            }
            _ => unreachable!(),
        };
        for (observed, expected) in [
            (pdf, value(4)),
            (sf, value(5)),
            (resultant, value(6)),
            (entropy, value(7)),
        ] {
            close(observed.into_f64(), expected, tolerance);
        }
    }
}
#[test]
fn independent_f64_fixtures() {
    fixture_evidence::<f64>();
}
#[test]
fn independent_f32_fixtures() {
    fixture_evidence::<f32>();
}

#[test]
fn independent_fixture_ieee_projections() {
    let decimals = include_str!("../data/circular.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect::<Vec<_>>();
    let ieee = include_str!("../data/circular_ieee.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect::<Vec<_>>();
    assert_eq!(decimals.len(), ieee.len());
    for (decimal, raw) in decimals.iter().zip(ieee) {
        let values = decimal.split('\t').collect::<Vec<_>>();
        let words = raw.split('\t').collect::<Vec<_>>();
        assert_eq!(&values[..2], &words[..2]);
        assert_eq!(words.len(), 14);
        for (i, value) in values[2..].iter().enumerate() {
            let value = value.parse::<f64>().unwrap();
            assert_eq!(
                value.to_bits(),
                u64::from_str_radix(words[2 + i], 16).unwrap()
            );
            assert_eq!(
                (value as f32).to_bits(),
                u32::from_str_radix(words[8 + i], 16).unwrap()
            );
        }
    }
}

struct Reverse;
impl scioxide_core::Executor for Reverse {
    fn worker_budget(&self) -> core::num::NonZeroUsize {
        core::num::NonZeroUsize::new(4).unwrap()
    }
    fn execute(
        &self,
        context: scioxide_core::ExecutionContext<'_>,
        plan: scioxide_core::ExecutionPlan,
        operation: &scioxide_core::PartitionFn<'_>,
    ) -> Result<(), scioxide_core::SciOxideError> {
        for index in (0..plan.worker_count()).rev() {
            scioxide_core::Executor::execute(
                &scioxide_core::SequentialExecutor,
                context,
                plan,
                &|part, (), scope| {
                    if part.worker_index() == index {
                        operation(part, (), scope)
                    } else {
                        Ok(())
                    }
                },
            )?;
        }
        scioxide_core::Executor::execute(
            &scioxide_core::SequentialExecutor,
            context,
            plan,
            &|_, (), _| Ok(()),
        )
    }
}
#[test]
fn executor_replay_and_rotation() {
    use scioxide_core::{ExecutionContext, ExecutionPolicy};
    let workers = core::num::NonZeroUsize::new(4).unwrap();
    let root = Pcg64::from_seed(541);
    let parallel = LawOptions::default().with_execution_policy(ExecutionPolicy::Parallel(workers));
    assert_eq!(
        LawOptions::<f64>::default().execution_policy(),
        ExecutionPolicy::Sequential
    );
    let law = VonMisesFisher::new(&[0.6_f64, 0.8, 0.0], 10.0).unwrap();
    let scheduled = VonMisesFisher::with_options(&[0.6_f64, 0.8, 0.0], 10.0, parallel).unwrap();
    let mut a = [0.0; 300];
    let mut b = [0.0; 300];
    let mut workspace = SamplingWorkspace::new();
    law.sample_indexed_into(
        &root,
        &mut a,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    scheduled
        .sample_indexed_into(
            &root,
            &mut b,
            &mut workspace,
            ExecutionContext::new(&Reverse),
        )
        .unwrap();
    assert_eq!(a, b);
    for row in a.chunks_exact(3) {
        assert!(law.contains_support(row).unwrap());
    }
    let axial = VonMisesFisher::new(&[1.0, 0.0, 0.0], 10.0).unwrap();
    for theta in [0.0_f64, 0.3, 1.0, PI] {
        let (s, c) = theta.sin_cos();
        let rotated = [0.6 * c - 0.8 * s, 0.8 * c + 0.6 * s, 0.0];
        close(
            law.log_pdf(&rotated).unwrap(),
            axial.log_pdf(&[c, s, 0.0]).unwrap(),
            2e-12,
        );
    }
    for mu in [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        let law = VonMisesFisher::new(&mu, 1000.0).unwrap();
        let mut rng = SplitMix64::from_seed(541);
        let x = law.sample(&mut rng).unwrap();
        assert!(law.contains_support(&x).unwrap());
        assert!(x.iter().zip(mu).map(|(a, b)| a * b).sum::<f64>() > 0.99);
    }
    let law = VonMises::new(0.0, 10.0).unwrap();
    let scheduled = VonMises::with_options(0.0, 10.0, parallel).unwrap();
    let mut a = [0.0; 100];
    let mut b = [0.0; 100];
    law.sample_indexed_into(
        &root,
        &mut a,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    scheduled
        .sample_indexed_into(
            &root,
            &mut b,
            &mut workspace,
            ExecutionContext::new(&Reverse),
        )
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn extreme_concentration_and_direct_log_tails() {
    let law = VonMises::new(0.0_f64, 1.0).unwrap();
    // Independent Boost normalization fixture; native cos uses full-range
    // reduction and is independent of the production half-angle identity.
    let log_normalizer = 0.3_f64.cos() - (-1.1184549357909181);
    for angle in [1e20_f64, 1e100, 1e300] {
        close(
            law.log_pdf(angle).unwrap(),
            angle.cos() - log_normalizer,
            5e-9,
        );
    }
    let tiny = VonMises::new(0.0_f64, 1e-300).unwrap();
    close(tiny.mean_resultant_length().unwrap() / 1e-300, 0.5, 2e-12);
    let tiny = VonMises::new(0.0_f32, 1e-20).unwrap();
    close(
        f64::from(tiny.mean_resultant_length().unwrap() / 1e-20),
        0.5,
        2e-6,
    );
    let concentrated = VonMises::new(0.0_f64, 1e300).unwrap();
    // The floating TAU lies just below the exact period. Full-range phase
    // reduction resolves that offset even when it exceeds the peak width.
    assert_eq!(concentrated.winding_cdf(TAU).unwrap(), 1.0);
    let x = concentrated.quantile(0.1).unwrap();
    close(concentrated.cdf(x).unwrap(), 0.1, 2e-10);
    close(concentrated.variance().unwrap() * 1e300, 1.0, 2e-10);
    let concentrated = VonMises::new(0.0_f32, 3e38).unwrap();
    assert!(concentrated.log_pdf(core::f32::consts::PI).is_err());
    for k in [1e12_f64, 1e30, 1e300] {
        let law = VonMises::new(0.0, k).unwrap();
        assert!(law.log_pdf(0.0).unwrap().is_finite());
        assert!(law.entropy().unwrap().is_finite());
        assert!(law.circular_variance().unwrap() > 0.0);
        let mut rng = SplitMix64::from_seed(541);
        let x = law.sample(&mut rng).unwrap();
        assert!(x.abs() < 10.0 / k.sqrt());
        for d in [3, 5] {
            let mut mu = vec![0.0; d];
            mu[0] = 1.0;
            let law = VonMisesFisher::new(&mu, k).unwrap();
            assert!(law.log_pdf(&mu).unwrap().is_finite());
            assert!(law.entropy().unwrap().is_finite());
            let x = law.sample(&mut rng).unwrap();
            assert!(law.contains_support(&x).unwrap());
            assert!(x[1..].iter().any(|x| *x != 0.0));
        }
    }
    let law = VonMises::new(0.0, 1000.0).unwrap();
    assert!(law.log_sf(3.0).unwrap() < -1900.0);
    assert_eq!(law.sf(3.0).unwrap(), 0.0);
    let x = law.isf(1e-100).unwrap();
    close(law.log_sf(x).unwrap(), -100.0 * 10.0_f64.ln(), 2e-10);
}

fn normalization<S: RealScalar>() {
    let scalar = |x| S::try_from_f64_lossy(x).unwrap();
    let tolerance = if S::MANTISSA_DIGITS == 24 {
        3e-5
    } else {
        2e-10
    };
    for k in [0.0, 1.0, 10.0, 1000.0] {
        let law = VonMisesLine::new(S::ZERO, scalar(k)).unwrap();
        let n = 8192;
        let total = (0..n)
            .map(|i| {
                law.pdf(scalar(-PI + TAU * (i as f64 + 0.5) / n as f64))
                    .unwrap()
                    .into_f64()
            })
            .sum::<f64>()
            * TAU
            / n as f64;
        close(total, 1.0, tolerance);
    }
    let law = WrappedCauchy::new(S::ZERO, scalar(0.9)).unwrap();
    let n = 8192;
    let total = (0..n)
        .map(|i| {
            law.pdf(scalar(TAU * (i as f64 + 0.5) / n as f64))
                .unwrap()
                .into_f64()
        })
        .sum::<f64>()
        * TAU
        / n as f64;
    close(total, 1.0, tolerance);
    let law = VonMisesFisher::new(&[S::ONE, S::ZERO, S::ZERO], scalar(10.0)).unwrap();
    let n = 32768;
    let total = (0..n)
        .map(|i| {
            let z = -1.0 + 2.0 * (i as f64 + 0.5) / n as f64;
            law.pdf(&[scalar(z), scalar((1.0 - z * z).sqrt()), S::ZERO])
                .unwrap()
                .into_f64()
        })
        .sum::<f64>()
        * 2.0
        * TAU
        / n as f64;
    close(total, 1.0, tolerance.max(1e-7));
}
#[test]
fn density_normalization_both_widths() {
    normalization::<f32>();
    normalization::<f64>();
}

#[test]
fn near_one_wrapped_parameter_and_buffer_admission() {
    // A large f32 dimension must not turn a zero vector into an admitted
    // direction when the dimension-scaled rounding allowance exceeds one.
    let zeros = vec![0.0_f32; 2_000_000];
    let options = LawOptions::default().with_work_budget(Budget::try_from(zeros.len()).unwrap());
    assert!(VonMisesFisher::with_options(&zeros, 0.0, options).is_err());
    let law = WrappedCauchy::new(0.0_f32, 1.0_f32.next_down()).unwrap();
    assert!(law.log_pdf(0.0).unwrap().is_finite());
    assert!(law.log_sf(core::f32::consts::PI).unwrap().is_finite());
    assert_eq!(law.quantile(0.5).unwrap(), core::f32::consts::PI);
    let rho = 1.0_f64.next_down();
    let wrapped = WrappedCauchy::new(0.0, rho).unwrap();
    assert_eq!(wrapped.quantile(0.5).unwrap(), PI);
    assert_eq!(wrapped.isf(0.5).unwrap(), PI);
    assert_eq!(wrapped.log_pdf(TAU).unwrap(), wrapped.log_pdf(0.0).unwrap());
    let wrapped = WrappedCauchy::new(0.0_f64, 0.9).unwrap();
    close(
        wrapped.log_sf(1e-100).unwrap() / 1e-100,
        -1.9 / (TAU * 0.1),
        2e-12,
    );
    let options = LawOptions::default().with_max_bytes(8);
    let law = VonMises::with_options(0.0_f64, 1.0, options).unwrap();
    let mut rng = SplitMix64::from_seed(541);
    assert_eq!(
        law.sample_n(&mut rng, 2).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    let mut out = [7.0; 2];
    let mut workspace = SamplingWorkspace::new();
    assert_eq!(
        law.sample_n_into(
            &mut rng,
            SamplingPolicy::advance_on_error(OutputPolicy::Atomic),
            &mut out,
            &mut workspace
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(out, [7.0; 2]);
    // Caller-borrowed output is not charged: incremental scalars need no staging.
    law.sample_n_into(
        &mut rng,
        SamplingPolicy::advance_on_error(OutputPolicy::Incremental),
        &mut out,
        &mut workspace,
    )
    .unwrap();
    // Indexed batches stage the whole output under the same limit.
    assert_eq!(
        law.sample_indexed_into(
            &Pcg64::from_seed(1),
            &mut out,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
}

#[path = "../support/statistics.rs"]
mod statistics;

/// Independent analytic axial vMF CDF in dimension five. With
/// v=kappa*(1-w), integrate v*(1-v/(2*kappa))*exp(-v) exactly. The
/// 1e-4 allowance covers binary32 coordinate rounding at kappa=1000;
/// the smallest tested kappa=.1 keeps cancellation well within that allowance.
fn vmf5_integral(kappa: f64, lower: f64) -> f64 {
    let upper = 2.0 * kappa;
    if lower >= upper {
        return 0.0;
    }
    let tail = |v: f64| (-v).exp() * (v + 1.0 - (v * v + 2.0 * v + 2.0) / upper);
    tail(lower) - tail(upper)
}

fn directional_goodness<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    let mut rng = SplitMix64::from_seed(1335);
    let law = UniformDirection::<S>::new(1).unwrap();
    let mut counts = [0; 2];
    for _ in 0..statistics::DRAWS {
        let x = law.sample(&mut rng).unwrap()[0];
        assert!(x == S::ONE || x == -S::ONE);
        counts[usize::from(x > S::ZERO)] += 1;
    }
    statistics::assert_fit("sphere/d=1", &counts, &[0.5, 0.5], 0.0);
    for d in [2, 3, 5, 20] {
        let law = UniformDirection::<S>::new(d).unwrap();
        let values: Vec<_> = (0..statistics::DRAWS)
            .map(|_| law.sample(&mut rng).unwrap())
            .collect();
        let reference = Beta::new((d - 1) as f64 / 2.0, (d - 1) as f64 / 2.0).unwrap();
        for coordinate in [0, 1] {
            let mut values = values.iter();
            statistics::continuous(&format!("sphere/{d}/{coordinate}"), &reference, || {
                (values.next().unwrap()[coordinate].into_f64() + 1.0) / 2.0
            });
        }
    }
    // Zero concentration, angular d=2, the two d=3 inversion formulas,
    // and Wood rejection in d=5, including strong concentration.
    for kappa in [0.0, 0.1, 1.0, 10.0, 1000.0] {
        let law = VonMisesFisher::new(&[S::ONE, S::ZERO], s(kappa)).unwrap();
        let reference = VonMisesLine::new(0.0, kappa).unwrap();
        statistics::continuous("vMF/d=2", &reference, || {
            let x = law.sample(&mut rng).unwrap();
            x[1].into_f64().atan2(x[0].into_f64())
        });
        for d in [3, 5] {
            let mut mu = vec![S::ZERO; d];
            mu[0] = S::ONE;
            let law = VonMisesFisher::new(&mu, s(kappa)).unwrap();
            let cuts: Vec<_> = if kappa == 0.0 {
                let reference = Beta::new((d - 1) as f64 / 2.0, (d - 1) as f64 / 2.0).unwrap();
                (1..32)
                    .map(|i| {
                        let p = f64::from(i) / 32.0;
                        (2.0 * reference.quantile(p).unwrap() - 1.0, p)
                    })
                    .collect()
            } else if d == 3 {
                (1..32)
                    .map(|i| {
                        let p = f64::from(i) / 32.0;
                        (
                            1.0 + ((-2.0 * kappa).exp() + p * (-(-2.0 * kappa).exp_m1())).ln()
                                / kappa,
                            p,
                        )
                    })
                    .collect()
            } else {
                let total = vmf5_integral(kappa, 0.0);
                let gamma = Gamma::new(2.0, 1.0).unwrap();
                (1..32)
                    .rev()
                    .filter_map(|i| {
                        let v = if kappa < 1.0 {
                            2.0 * kappa * f64::from(i) / 32.0
                        } else {
                            gamma.quantile(f64::from(i) / 32.0).unwrap()
                        };
                        (v < 2.0 * kappa)
                            .then(|| (1.0 - v / kappa, vmf5_integral(kappa, v) / total))
                    })
                    .collect()
            };
            statistics::cdf_fit(
                &format!("vMF/{d}/{kappa}"),
                &cuts,
                statistics::DRAWS,
                1e-4,
                || law.sample(&mut rng).unwrap()[0].into_f64(),
            );
        }
    }
}
#[test]
fn directional_goodness_f32() {
    directional_goodness::<f32>();
}
#[test]
fn directional_goodness_f64() {
    directional_goodness::<f64>();
}
