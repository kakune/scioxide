#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use scioxide_core::{Complex, RealScalar};
use scioxide_special::ErrorFunctionOptions;
use scioxide_special::*;
fn oracle<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    for line in include_str!("data/probability.tsv")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let cols: Vec<_> = line.split_whitespace().collect();
        if cols[0] != if S::MANTISSA_DIGITS == 24 { "32" } else { "64" } {
            continue;
        }
        let v: Vec<S> = cols[2..]
            .iter()
            .map(|s| S::try_from_f64_lossy(s.parse().unwrap()).unwrap())
            .collect();
        let (x, y, z) = (v[0], v[1], v[2]);
        let result = match cols[1] {
            "normal_inv_log_cdf" => normal_inv_log_cdf(x).map(|r| Complex::new(r, S::ZERO)),
            "gamma_p_inv_log" => gamma_p_inv_log(x, y).map(|r| Complex::new(r, S::ZERO)),
            "gamma_q_inv_log" => gamma_q_inv_log(x, y).map(|r| Complex::new(r, S::ZERO)),
            "beta_inc_inv_log" => beta_inc_inv_log(x, y, z).map(|r| Complex::new(r, S::ZERO)),
            "voigt_profile" => voigt_profile(x, y, z).map(|r| Complex::new(r, S::ZERO)),
            "erfcx" => erfcx(x).map(|r| Complex::new(r, S::ZERO)),
            "erfi" => erfi(x).map(|r| Complex::new(r, S::ZERO)),
            "erfi_scaled" => dawson(x).map(|r| Complex::new(r, S::ZERO)),
            "log_erfc" => log_erfc(x).map(|r| Complex::new(r, S::ZERO)),
            "normal_log_cdf" => normal_log_cdf(x).map(|r| Complex::new(r, S::ZERO)),
            "faddeeva" => faddeeva(Complex::new(x, y)),
            "owens_t" => owens_t(x, y).map(|r| Complex::new(r, S::ZERO)),
            "gamma_log_p" => gamma_log_p(x, y).map(|r| Complex::new(r, S::ZERO)),
            "gamma_log_q" => gamma_log_q(x, y).map(|r| Complex::new(r, S::ZERO)),
            "beta_inc_log" => beta_inc_log(x, y, z).map(|r| Complex::new(r, S::ZERO)),
            "beta_inc_log_q" => beta_inc_log_q(x, y, z).map(|r| Complex::new(r, S::ZERO)),
            _ => unreachable!(),
        };
        let r = result.unwrap_or_else(|e| panic!("{line}: {e:?}"));
        let tail = cols[1].starts_with("gamma") || cols[1].starts_with("beta");
        let (rel, abs) = if S::MANTISSA_DIGITS == 24 {
            if tail { (5e-4, 2e-6) } else { (2e-5, 2e-6) }
        } else if tail {
            (2e-10, 2e-13)
        } else {
            (2e-12, 2e-14)
        };
        for (a, b) in [(r.re, v[3]), (r.im, v[4])] {
            let a = a.try_to_f64_exact().unwrap();
            let b = b.try_to_f64_exact().unwrap()
                * if cols[1] == "erfi_scaled" {
                    core::f64::consts::PI.sqrt() * 0.5
                } else {
                    1.0
                };
            let tiny_floor = if S::MANTISSA_DIGITS == 24 {
                16. * f32::from_bits(1) as f64
            } else {
                8. * f64::from_bits(1)
            };
            let budget = if b != 0. && b.abs() < 1e-20 {
                tiny_floor.max(rel * b.abs())
            } else {
                abs.max(rel * b.abs())
            };
            assert!(
                (a - b).abs() <= budget,
                "{line}: actual {a:e} expected {b:e}"
            );
        }
    }
}
#[test]
fn oracle_f64() {
    oracle::<f64>();
}
#[test]
fn oracle_f32() {
    oracle::<f32>();
}
#[test]
fn inverse_tails() {
    for lp in [-0.001_f64, -1., -10., -100., -1000., -10000.] {
        let x = normal_inv_log_cdf(lp).unwrap();
        assert!(
            (normal_log_cdf(x).unwrap() - lp).abs() < 1e-10 * lp.abs().max(1.),
            "normal {lp} {x}"
        );
        let x = gamma_q_inv_log(2.5, lp).unwrap();
        assert!(
            (gamma_log_q(2.5, x).unwrap() - lp).abs() < 1e-10 * lp.abs().max(1.),
            "gamma {lp} {x}"
        );
    }
    for lp in [-1.0_f64, -10., -100., -1000.] {
        let x = gamma_p_inv_log(2.5, lp).unwrap();
        assert!(
            (gamma_log_p(2.5, x).unwrap() - lp).abs() < 1e-10 * lp.abs().max(1.),
            "gamma lower {lp} {x}"
        );
        let x = beta_inc_inv_log(2.5, 3.5, lp).unwrap();
        assert!(
            (beta_inc_log(2.5, 3.5, x).unwrap() - lp).abs() < 1e-10 * lp.abs().max(1.),
            "beta {lp} {x}"
        );
    }
}

#[test]
fn contracts_and_scaled_ranges() {
    use scioxide_core::Budget;
    let tiny = ProbabilityOptions::default().with_iteration_budget(Budget::ONE);
    // The fixed-cost erfcx kernel completes within any allowance.
    let r = erfcx_with_options(
        20.,
        scioxide_special::ErrorFunctionOptions::new((tiny).iteration_budget()),
    )
    .unwrap();
    assert_eq!(r.termination(), SpecialFunctionTermination::Converged);
    assert_eq!(r.counts().iterations(), 1);
    for report in [
        normal_inv_log_cdf_with_options(-1000., tiny).unwrap(),
        owens_t_with_options(2., 0.5, tiny).unwrap(),
    ] {
        assert_eq!(
            report.termination(),
            SpecialFunctionTermination::IterationLimit
        );
        assert!(report.counts().iterations() <= 1);
        assert!(report.try_value().is_err());
    }
    let r = faddeeva_with_options(Complex::new(1., 2.), tiny).unwrap();
    assert_eq!(r.termination(), SpecialFunctionTermination::IterationLimit);
    assert!(r.counts().iterations() <= 1);
    for r in [
        erfcx_with_options(-30., ErrorFunctionOptions::default()).unwrap(),
        erfi_with_options(30., ErrorFunctionOptions::default()).unwrap(),
    ] {
        assert_eq!(r.scalar_representation(), ScalarRepresentation::Overflow);
        assert!(r.logarithmic().is_some());
        assert!(r.scaled().is_some());
        assert!(r.try_value().is_err());
    }
    let r = faddeeva_with_options(Complex::new(1., -30.), ProbabilityOptions::default()).unwrap();
    assert_eq!(r.scalar_representation(), ScalarRepresentation::Overflow);
    assert!(r.logarithmic().is_some());
    assert!(r.scaled().is_some());
    assert!(r.try_value().is_err());
    {
        let outcome = &(erfcx(f64::NAN));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
    {
        let outcome = &(faddeeva(Complex::new(f64::INFINITY, 0.)));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::NonFinite { .. }
        ));
    }
    {
        let outcome = &(voigt_profile(0., -1., 1.));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(voigt_profile(0., 0., 0.));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(normal_quantile(-0.1_f64));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(gamma_q_inv(0., 0.1_f64));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    {
        let outcome = &(beta_inc_q_inv(1., -1., 0.1_f64));
        let err = outcome.as_ref().expect_err("expected error");
        assert!(matches!(
            err.view(),
            scioxide_core::SciOxideErrorView::DomainError { .. }
        ));
    }
    assert_eq!(normal_quantile(0.0_f64).unwrap(), f64::NEG_INFINITY);
    assert_eq!(normal_isf(0.0_f64).unwrap(), f64::INFINITY);
    assert_eq!(gamma_log_p(2., 0.0_f64).unwrap(), f64::NEG_INFINITY);
    assert_eq!(beta_inc_log_q(2., 3., 1.0_f64).unwrap(), f64::NEG_INFINITY);
    let budget = ProbabilityOptions::new(Budget::try_from(1024).unwrap());
    assert_eq!(
        erfcx_with_options(
            20.,
            scioxide_special::ErrorFunctionOptions::new((budget).iteration_budget())
        )
        .unwrap()
        .termination(),
        SpecialFunctionTermination::Converged
    );
}

#[test]
fn identities_and_quantile_space() {
    for x in [-10.0_f64, -2., -0.1, 0., 0.1, 2., 10.] {
        let p = normal_cdf(x).unwrap();
        let q = normal_sf(x).unwrap();
        assert!((p + q - 1.).abs() < 2e-15);
        let recovered = if x < 0. {
            normal_quantile(p)
        } else {
            normal_isf(q)
        }
        .unwrap();
        assert!((recovered - x).abs() < 2e-12, "{x} {recovered}");
        assert!(
            (voigt_profile(x, 1., 0.).unwrap()
                - (-x * x / 2.).exp() / (2. * std::f64::consts::PI).sqrt())
            .abs()
                < 2e-13
        );
        assert!(
            (voigt_profile(x, 0., 1.).unwrap() - 1. / (std::f64::consts::PI * (1. + x * x))).abs()
                < 2e-13
        );
        let z = Complex::new(x, 0.7);
        let r = faddeeva_with_options(z, ProbabilityOptions::default()).unwrap();
        let v = r.try_value().unwrap();
        let conjugate = faddeeva(Complex::new(-x, 0.7)).unwrap();
        assert!((v.re - conjugate.re).abs() < 1e-13);
        assert!((v.im + conjugate.im).abs() < 1e-13);
        let h = 1e-5;
        let fd = (faddeeva(Complex::new(x + h, 0.7)).unwrap()
            - faddeeva(Complex::new(x - h, 0.7)).unwrap())
            / (2. * h);
        let d = *r.derivative().unwrap().value();
        assert!((fd - d).norm() < 1e-8);
        let t = owens_t(x, 0.8).unwrap();
        assert_eq!(t, owens_t(-x, 0.8).unwrap());
        assert_eq!(-t, owens_t(x, -0.8).unwrap());
    }
    for x in [0.0001_f64, 0.01, 0.2, 0.5, 0.8, 0.99] {
        let p = beta_inc(2.5, 3.5, x).unwrap();
        let q = beta_inc_q(2.5, 3.5, x).unwrap();
        assert!((p + q - 1.).abs() < 2e-13);
        let inv = if p < q {
            beta_inc_inv(2.5, 3.5, p)
        } else {
            beta_inc_q_inv(2.5, 3.5, q)
        }
        .unwrap();
        assert!((inv - x).abs() < 2e-11, "{x} {inv}");
    }
    for x in [0.0001_f32, 0.2, 0.5, 0.8, 0.99] {
        let q = beta_inc_q(2.5, 3.5, x).unwrap();
        let inv = beta_inc_q_inv(2.5, 3.5, q).unwrap();
        assert!((inv - x).abs() < 2e-4, "{x} {inv}");
    }
    assert!(erfcx(f64::MAX).unwrap() > 0.);
    assert!(normal_inv_log_cdf(-10000.0_f32).unwrap().is_finite());
    assert!(gamma_q_inv_log(2.5, -1000.0_f32).unwrap().is_finite());
}

#[test]
fn success_and_partial_paths_execute() {
    {
        for x in [0.1_f64, 2., 20.] {
            normal_cdf(x).unwrap();
            normal_sf(x).unwrap();
            normal_log_cdf(x).unwrap();
            normal_log_sf(x).unwrap();
            normal_quantile((-x).exp()).unwrap();
            normal_isf((-x).exp()).unwrap();
            let _ = normal_sf_with_options(x, ProbabilityOptions::default()).unwrap();
            let _ =
                normal_quantile_with_options((-x).exp(), ProbabilityOptions::default()).unwrap();
            erfcx(x).unwrap();
            dawson(x).unwrap();
            log_erfc(x).unwrap();
            normal_inv_log_cdf(-x).unwrap();
            owens_t(x, 0.8).unwrap();
            faddeeva(Complex::new(x, 0.7)).unwrap();
            voigt_profile(x, 1., 0.7).unwrap();
            gamma_log_q(2.5, x).unwrap();
            gamma_q_inv_log(2.5, -x).unwrap();
            beta_inc_log(2.5, 3.5, 0.1).unwrap();
            beta_inc_inv_log(2.5, 3.5, -x).unwrap();
        }
        let o = ProbabilityOptions::default().with_iteration_budget(scioxide_core::Budget::ONE);
        let _ = faddeeva_with_options(Complex::new(1., 2.), o).unwrap();
        let _ = normal_inv_log_cdf_with_options(-100., o).unwrap();
    };
}

#[test]
fn tiny_log_complements_and_subnormal_tails() {
    for x in [8.0_f64, 20., 37.] {
        let l = normal_log_cdf(x).unwrap();
        let q = normal_sf(x).unwrap();
        assert!(l < 0.);
        assert!((l / q + 1.).abs() < 2e-12);
    }
    for x in [710.0_f64, 740., 750.] {
        let l = gamma_log_q(1., x).unwrap();
        assert!((l + x).abs() < 1e-10);
    }
    let p = f64::from_bits(1);
    let x = normal_quantile(p).unwrap();
    assert!((normal_log_cdf(x).unwrap() - p.ln()).abs() < 2e-12);
    let x = gamma_q_inv(1., p).unwrap();
    assert!((x + p.ln()).abs() < 2e-11);
}

#[test]
fn derivative_report_retains_cancellation_uncertainty() {
    let r =
        faddeeva_with_options(Complex::new(1e8_f64, 0.), ProbabilityOptions::default()).unwrap();
    let d = r.derivative().unwrap();
    let asymptotic = Complex::new(0., -1. / (std::f64::consts::PI.sqrt() * 1e16));
    assert!((*d.value() - asymptotic).norm() <= d.absolute_error());
    assert!(d.absolute_error() > 0.);
}
