#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use scioxide_core::{Budget, RealScalar, SciOxideError, Tolerance};
use scioxide_random::*;
trait Law<S: RealScalar>: ContinuousLaw<S> + Quantile<S> + InverseSurvival<S> {}
impl<S: RealScalar, D: ContinuousLaw<S> + Quantile<S> + InverseSurvival<S>> Law<S> for D {}
fn law<S: RealScalar>(name: &str, a: S, b: S, p: S) -> Box<dyn Law<S>> {
    match name {
        "crystalball" => Box::new(CrystalBall::new(a, b).unwrap()),
        "skewcauchy" => Box::new(SkewCauchy::new(a).unwrap()),
        "jf_skew_t" => Box::new(JonesFaddySkewT::new(a, b).unwrap()),
        "skewnorm" => Box::new(SkewNormal::new(a).unwrap()),
        "geninvgauss" => Box::new(GeneralizedInverseGaussian::new(a, b).unwrap()),
        "genhyperbolic" => Box::new(
            GeneralizedHyperbolic::with_options(
                p,
                a,
                b,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        "norminvgauss" => Box::new(
            NormalInverseGaussian::with_options(
                a,
                b,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap(),
        ),
        _ => unreachable!(),
    }
}
fn fixtures<S: RealScalar>() {
    let cast = |x: f64| S::try_from_f64_lossy(x).unwrap();
    let tol = if S::EPSILON.into_f64() > 1e-10 {
        3e-4
    } else {
        2e-7
    };
    for line in include_str!("../data/asymmetric.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let cols: Vec<_> = line.split_whitespace().collect();
        let nums: Vec<f64> = cols[1..].iter().map(|s| s.parse().unwrap()).collect();
        let d = law(cols[0], cast(nums[0]), cast(nums[1]), cast(nums[2]));
        let x = cast(nums[3]);
        for (label, got, want) in [
            ("pdf", d.log_pdf(x), nums[4]),
            ("cdf", d.log_cdf(x), nums[5]),
            ("sf", d.log_sf(x), nums[6]),
        ] {
            let got = got
                .unwrap_or_else(|e| panic!("{line}: {label}: {e:?}"))
                .into_f64();
            assert!(
                (got - want).abs() <= tol * (1. + want.abs()),
                "{line}: {label}: {got} != {want}"
            );
        }
        for (upper, lp) in [(false, nums[5]), (true, nums[6])] {
            if lp < -0.69 && lp > -70. {
                let p = cast(lp.exp());
                let got = if upper { d.isf(p) } else { d.quantile(p) }
                    .unwrap_or_else(|e| panic!("inverse {line}: {e:?}"));
                assert!(
                    (got.into_f64() - nums[3]).abs() <= tol * 10. * (1. + nums[3].abs()),
                    "inverse {line}: {got:?}"
                );
            }
        }
    }
}
#[test]
fn independent_f64() {
    fixtures::<f64>();
}
#[test]
fn independent_f32() {
    fixtures::<f32>();
}
#[test]
fn domains_and_moments() {
    assert!(CrystalBall::new(0., 4.).is_err());
    assert!(CrystalBall::new(2., 1.).is_err());
    assert!(SkewCauchy::new(1.).is_err());
    assert!(SkewNormal::new(f64::NAN).is_err());
    assert!(JonesFaddySkewT::new(0., 2.).is_err());
    assert!(GeneralizedInverseGaussian::new(1., 0.).is_err());
    assert!(
        GeneralizedHyperbolic::with_options(
            0.,
            1.,
            1.,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
    assert!(
        NormalInverseGaussian::with_options(
            1.,
            1.,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_err()
    );
    assert!(CrystalBall::new(1., 2.).unwrap().mean().is_err());
    assert!(CrystalBall::new(1., 3.).unwrap().variance().is_err());
    assert!(JonesFaddySkewT::new(0.5, 2.).unwrap().mean().is_err());
    assert!(JonesFaddySkewT::new(1., 2.).unwrap().variance().is_err());
    assert!(
        GeneralizedHyperbolic::with_options(
            -1.,
            1.,
            1.,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap()
        .mean()
        .is_err()
    );
    assert!(
        GeneralizedHyperbolic::with_options(
            -2.,
            1.,
            1.,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .unwrap()
        .variance()
        .is_err()
    );
    let d = GeneralizedHyperbolic::with_options(
        -3.0_f64,
        1.,
        1.,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    assert!((d.mean().unwrap() - 0.25).abs() < 1e-10);
    assert!((d.variance().unwrap() - 0.3125).abs() < 1e-10);
    for name in [
        "crystalball",
        "skewcauchy",
        "jf_skew_t",
        "skewnorm",
        "geninvgauss",
        "genhyperbolic",
        "norminvgauss",
    ] {
        let (a, b, p) = match name {
            "skewcauchy" => (0.4, 0., 0.),
            "genhyperbolic" | "norminvgauss" => (2., 0.5, 1.),
            _ => (2., 4., 1.),
        };
        let d = law(name, a, b, p);
        assert!(d.log_pdf(f64::NAN).is_err());
        assert!(d.cdf(f64::INFINITY).is_err());
        assert!(d.quantile(-0.1).is_err());
        assert_eq!(d.isf(0.).unwrap(), f64::INFINITY);
        assert_eq!(
            d.quantile(0.).unwrap(),
            if name == "geninvgauss" {
                0.
            } else {
                f64::NEG_INFINITY
            }
        );
    }
}
#[test]
fn exact_reductions() {
    for x in [-5.0_f64, -1., 0., 1., 5.] {
        let n = Normal::new(0., 1.).unwrap();
        let s = SkewNormal::new(0.).unwrap();
        assert!((n.log_sf(x).unwrap() - s.log_sf(x).unwrap()).abs() < 1e-12);
        let t = StudentT::new(4.).unwrap();
        let j = JonesFaddySkewT::new(2., 2.).unwrap();
        assert!((t.pdf(x).unwrap() - j.pdf(x).unwrap()).abs() < 1e-12);
        let c = Cauchy::new(0., 1.).unwrap();
        let s = SkewCauchy::new(0.).unwrap();
        assert!((c.pdf(x).unwrap() - s.pdf(x).unwrap()).abs() < 1e-12);
        let n = NormalInverseGaussian::with_options(
            2.,
            0.5,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        let g = GeneralizedHyperbolic::with_options(
            -0.5,
            2.,
            0.5,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap();
        assert_eq!(n.log_pdf(x).unwrap(), g.log_pdf(x).unwrap());
    }
}
fn goodness<S: RealScalar, D: Distribution<S> + Quantile<S> + Cdf<S>>(d: D) {
    let mut rng = Xoshiro256StarStar::from_seed(539);
    statistics::continuous(core::any::type_name::<D>(), &d, || {
        d.sample(&mut rng).unwrap()
    });
}
#[test]
fn seeded_quantile_goodness() {
    goodness::<f64, _>(CrystalBall::new(2., 4.).unwrap());
    goodness::<f64, _>(SkewCauchy::new(0.8).unwrap());
    goodness::<f64, _>(JonesFaddySkewT::new(2., 5.).unwrap());
    goodness::<f64, _>(SkewNormal::new(4.).unwrap());
    goodness::<f64, _>(GeneralizedInverseGaussian::new(-2., 0.1).unwrap());
    goodness::<f64, _>(
        GeneralizedHyperbolic::with_options(
            1.,
            2.,
            0.5,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    goodness::<f64, _>(
        NormalInverseGaussian::with_options(
            2.,
            0.5,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    goodness::<f64, _>(GeneralizedInverseGaussian::new(2., 0.1).unwrap());
    goodness::<f64, _>(
        GeneralizedHyperbolic::with_options(
            -3.,
            1.,
            -1.,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
}
#[test]
fn budgets() {
    let options = LawOptions::default()
        .with_tolerance(Tolerance::from_sqrt_epsilon())
        .with_work_budget(Budget::ONE);
    let error = GeneralizedInverseGaussian::with_options(1., 2., options).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(
        GeneralizedHyperbolic::with_options(1., 2., 0.5, options)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
}
#[test]
fn extreme_logarithmic_tails() {
    let s = SkewNormal::new(4.0_f64).unwrap();
    let l = s.log_sf(40.).unwrap();
    assert!((l - (-803.9152948331939)).abs() < 2e-7);
    let j = JonesFaddySkewT::new(2., 5.).unwrap();
    assert!(j.log_sf(1e200).unwrap().is_finite());
    let s = SkewNormal::new(-1e6_f64).unwrap();
    assert!((s.log_sf(0.).unwrap() + (core::f64::consts::PI * 1e6).ln()).abs() < 1e-7);
    goodness::<f64, _>(GeneralizedInverseGaussian::new(0., 1e-6).unwrap());
    goodness::<f64, _>(
        GeneralizedHyperbolic::with_options(
            -3.,
            1.,
            1.,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
}
fn moments<S: RealScalar, D: Mean<S> + Variance<S>>(d: D, expected: &[f64]) {
    let (m, m2) = (expected[0], expected[1]);
    let variance = m2 - m * m;
    let tol = if S::MANTISSA_DIGITS == 24 { 3e-4 } else { 2e-8 };
    assert!((d.mean().unwrap().into_f64() - m).abs() < tol * (1. + m.abs()));
    assert!((d.variance().unwrap().into_f64() - variance).abs() < tol * (1. + variance));
}
fn moment_width<S: RealScalar>() {
    let s = |x| S::try_from_f64_lossy(x).unwrap();
    for line in include_str!("../data/asymmetric_moments.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let cols: Vec<_> = line.split_whitespace().collect();
        let n: Vec<f64> = cols[1..].iter().map(|v| v.parse().unwrap()).collect();
        let (a, b, p) = (s(n[0]), s(n[1]), s(n[2]));
        let expected = &n[3..];
        match cols[0] {
            "crystalball" => moments(CrystalBall::new(a, b).unwrap(), expected),
            "jf_skew_t" => moments(JonesFaddySkewT::new(a, b).unwrap(), expected),
            "skewnorm" => moments(SkewNormal::new(a).unwrap(), expected),
            "geninvgauss" => moments(GeneralizedInverseGaussian::new(a, b).unwrap(), expected),
            "genhyperbolic" => moments(
                GeneralizedHyperbolic::with_options(
                    p,
                    a,
                    b,
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap(),
                expected,
            ),
            "norminvgauss" => moments(
                NormalInverseGaussian::with_options(
                    a,
                    b,
                    scioxide_random::LawOptions::default()
                        .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
                )
                .unwrap(),
                expected,
            ),
            _ => unreachable!(),
        }
    }
}
#[test]
fn independent_moments() {
    moment_width::<f32>();
    moment_width::<f64>();
}
#[test]
fn independent_mesh_normalization_and_affine() {
    for name in [
        "crystalball",
        "skewcauchy",
        "jf_skew_t",
        "skewnorm",
        "geninvgauss",
        "genhyperbolic",
        "norminvgauss",
    ] {
        let (a, b, p) = match name {
            "skewcauchy" => (0.4, 0., 0.),
            "genhyperbolic" | "norminvgauss" => (2., 0.5, 1.),
            _ => (2., 4., 1.),
        };
        let d = law(name, a, b, p);
        // Independent fixed midpoint mesh, not the production adaptive tail rule.
        let h = 24. / 8192.;
        let mass = (0..8192)
            .map(|i| {
                let t = -12. + (f64::from(i) + 0.5) * h;
                let (x, j) = if name == "geninvgauss" {
                    (t.exp(), t.exp())
                } else {
                    (t.sinh(), t.cosh())
                };
                d.pdf(x).unwrap() * j * h
            })
            .sum::<f64>();
        assert!((mass - 1.).abs() < 4e-4, "{name} mass={mass}");
        let transformed = Affine::new(&*d, 2., -3.).unwrap();
        assert!((transformed.log_sf(-1.).unwrap() - d.log_cdf(1.).unwrap()).abs() < 1e-10);
    }
}

#[test]
fn seeded_quantile_goodness_f32() {
    goodness::<f32, _>(CrystalBall::new(2., 4.).unwrap());
    goodness::<f32, _>(SkewCauchy::new(0.8).unwrap());
    goodness::<f32, _>(JonesFaddySkewT::new(2., 5.).unwrap());
    goodness::<f32, _>(SkewNormal::new(4.).unwrap());
    goodness::<f32, _>(GeneralizedInverseGaussian::new(-2., 0.1).unwrap());
    goodness::<f32, _>(
        GeneralizedHyperbolic::with_options(
            1.,
            2.,
            0.5,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    goodness::<f32, _>(
        NormalInverseGaussian::with_options(
            2.,
            0.5,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    goodness::<f32, _>(GeneralizedInverseGaussian::new(0., 1e-6).unwrap());
    goodness::<f32, _>(GeneralizedInverseGaussian::new(2., 0.1).unwrap());
    goodness::<f32, _>(
        GeneralizedHyperbolic::with_options(
            -3.,
            1.,
            -1.,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
}
#[test]
fn fixed_proposal_guard_and_atomic_failure() {
    struct Rejecting {
        calls: usize,
    }
    impl BitGenerator for Rejecting {
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
        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), SciOxideError> {
            dest.fill(0);
            Ok(())
        }
    }
    impl Rng<f64> for Rejecting {
        fn next_float(&mut self) -> Result<f64, SciOxideError> {
            self.calls += 1;
            Ok(0.999)
        }
    }
    let d = GeneralizedInverseGaussian::new(1., 2.).unwrap();
    let mut rng = Rejecting { calls: 0 };
    let error = d.sample(&mut rng).unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );
    // Every proposal draws three open uniforms; the guard stops after 1,024.
    assert_eq!(rng.calls, 3 * 1024);
    let mut out = [7.; 2];
    let batch = SamplingPolicy::advance_on_error(OutputPolicy::Atomic);
    let mut work = batch
        .workspace(scioxide_random::SamplingRequest::scalars(2))
        .unwrap();
    assert!(
        d.sample_n_into(&mut rng, batch, &mut out, &mut work)
            .is_err()
    );
    assert_eq!(out, [7.; 2]);
}

#[test]
fn finite_extreme_ratios_keep_log_probabilities() {
    fn check<S: RealScalar>(large: S) {
        let a = S::ONE - S::EPSILON;
        let s = SkewCauchy::new(a).unwrap();
        let scale = (S::ONE - a).into_f64();
        let x = large.into_f64();
        let expected_pdf = -core::f64::consts::PI.ln() + 2. * (scale.ln() - x.ln());
        let expected_tail = 2. * scale.ln() - x.ln() - core::f64::consts::PI.ln();
        let tol = if S::MANTISSA_DIGITS == 24 {
            1e-4
        } else {
            1e-10
        };
        assert!((s.log_pdf(-large).unwrap().into_f64() - expected_pdf).abs() < tol);
        assert!((s.log_cdf(-large).unwrap().into_f64() - expected_tail).abs() < tol);
        let c = CrystalBall::new(
            S::try_from_f64_lossy(10.).unwrap(),
            S::try_from_f64_lossy(4.).unwrap(),
        )
        .unwrap();
        assert!(c.log_pdf(-large).unwrap().is_finite());
        assert!(c.log_cdf(-large).unwrap().is_finite());
    }
    check(3e38_f32);
    check(1e308_f64);
}

#[test]
fn unresolved_variance_is_not_a_degenerate_law() {
    let law = GeneralizedInverseGaussian::new(1.0_f32, 1e8).unwrap();
    let error = law.variance().unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::NumericalInstability
    );
    let law = GeneralizedHyperbolic::with_options(
        1.0_f32,
        1e8,
        0.,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    assert!(law.variance().unwrap() > 0.);
    let expected = (1e8_f64.ln() - core::f64::consts::TAU.ln()) * 0.5;
    assert!((f64::from(law.log_pdf(0.).unwrap()) - expected).abs() < 3e-5);
}

#[test]
fn skew_cauchy_near_unit_half_mass_keeps_the_small_tail() {
    fn check<S: RealScalar>() {
        let a = S::ONE - S::EPSILON;
        let x = S::EPSILON * S::EPSILON;
        let law = SkewCauchy::new(a).unwrap();
        let reflected = SkewCauchy::new(-a).unwrap();
        let expected = (S::EPSILON * S::HALF).ln();
        let actual = law.log_cdf(x).unwrap();
        assert!((actual - expected).abs() < S::try_from_f64_lossy(100.).unwrap() * S::EPSILON);
        assert_eq!(actual, reflected.log_sf(-x).unwrap());
    }
    check::<f32>();
    check::<f64>();
}

#[path = "../support/statistics.rs"]
mod statistics;
