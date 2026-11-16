//! Independent mpmath 1.4.1 references retained in the oracle heavy_tail.py.
//! Decimal inputs, 50-digit defining formulas and S1 Fourier inversion;
//! SciPy names supply parameter conventions only. See the oracle README.

#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{Budget, RealScalar, Tolerance};
use scioxide_random::{
    Affine, Burr12, Cdf, Distribution, GeneralizedExtremeValue, GeneralizedGamma,
    GeneralizedPareto, InverseSurvival, InverseWeibull, LawOptions, LevyStable, LogCdf, LogDensity,
    LogSurvival, Mean, Mielke, Moyal, Pareto, Quantile, RelativisticBreitWigner, SeedableRng,
    SplitMix64, Support, Variance,
};

#[path = "../support/laws.rs"]
mod laws;
use laws::CheckedLaw;

fn fixture_law<S: RealScalar>(name: &str, a: S, b: S) -> Box<dyn CheckedLaw<S>> {
    let o = S::ONE;
    let z = S::ZERO;
    match name {
        "burr" => Box::new(Mielke::new(a * b, a).unwrap()),
        "burr12" => Box::new(Burr12::new(a, b).unwrap()),
        "fisk" => Box::new(Burr12::new(a, o).unwrap()),
        "genpareto" => Box::new(GeneralizedPareto::new(a).unwrap()),
        "genextreme" => Box::new(GeneralizedExtremeValue::new(a).unwrap()),
        "invweibull" => Box::new(InverseWeibull::new(a).unwrap()),
        "landau" => Box::new(LevyStable::new(o, o).unwrap()),
        "levy" => {
            Box::new(Affine::new(GeneralizedGamma::new(S::HALF, -o).unwrap(), z, S::HALF).unwrap())
        }
        "levy_l" => {
            Box::new(Affine::new(GeneralizedGamma::new(S::HALF, -o).unwrap(), z, -S::HALF).unwrap())
        }
        "levy_stable" => Box::new(LevyStable::new(a, b).unwrap()),
        "lomax" => Box::new(Affine::new(Pareto::new(a).unwrap(), -o, o).unwrap()),
        "moyal" => Box::new(Moyal::new()),
        "pareto" => Box::new(Pareto::new(a).unwrap()),
        "rel_breitwigner" => Box::new(RelativisticBreitWigner::new(a).unwrap()),
        _ => unreachable!(),
    }
}

fn seeded_bins<S: RealScalar>() {
    for (index, line) in include_str!("../data/heavy_tail.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .step_by(2)
        .enumerate()
    {
        let cols: Vec<_> = line.split_whitespace().collect();
        let a = S::try_from_f64_lossy(cols[1].parse().unwrap()).unwrap();
        let b = S::try_from_f64_lossy(cols[2].parse().unwrap()).unwrap();
        let d = fixture_law(cols[0], a, b);
        let mut rng = SplitMix64::from_seed(537 + index as u64);
        statistics::continuous(line, &*d, || d.draw(&mut rng).unwrap());
    }
}
#[test]
fn seeded_goodness_f64() {
    seeded_bins::<f64>();
}
#[test]
fn seeded_goodness_f32() {
    seeded_bins::<f32>();
}

fn close(got: f64, want: f64, tol: f64) {
    assert!(
        (got - want).abs() <= tol * (1. + want.abs()),
        "{got} != {want}"
    );
}

#[test]
fn analytic_formulas_and_equivalences() {
    let b = Burr12::new(2., 3.).unwrap();
    close(b.cdf(2.).unwrap(), 1. - 5f64.powi(-3), 1e-13);
    close(b.log_sf(2.).unwrap(), -3. * 5f64.ln(), 1e-13);
    close(b.log_pdf(2.).unwrap(), (6. * 2. / 5f64.powi(4)).ln(), 1e-13);
    let p = b.cdf(2.).unwrap();
    close(b.quantile(p).unwrap(), 2., 1e-12);
    close(b.isf(1. - p).unwrap(), 2., 1e-12);
    let burr = Mielke::new(6., 2.).unwrap();
    close(burr.cdf(2.).unwrap(), (4. / 5f64).powi(3), 1e-13);
    close(
        burr.isf(1e-15).unwrap(),
        burr.quantile(1. - 1e-15).unwrap(),
        5e-4,
    );
    assert!(burr.isf(1e-20).unwrap().is_finite());
    let gp = GeneralizedPareto::new(-0.5).unwrap();
    close(gp.cdf(1.).unwrap(), 0.75, 1e-13);
    close(gp.quantile(0.75).unwrap(), 1., 1e-13);
    assert_eq!(gp.support().unwrap().bounds(), Some((Some(0.), Some(2.))));
    close(
        GeneralizedPareto::new(0.).unwrap().log_sf(3.).unwrap(),
        -3.,
        1e-13,
    );
    let gev = GeneralizedExtremeValue::new(0.5).unwrap();
    close(gev.log_cdf(0.).unwrap(), -1., 1e-13);
    close(gev.quantile((-1f64).exp()).unwrap(), 0., 1e-13);
    assert_eq!(gev.support().unwrap().bounds(), Some((None, Some(2.))));
    let inv = InverseWeibull::new(2.).unwrap();
    close(inv.log_cdf(2.).unwrap(), -0.25, 1e-13);
    close(inv.quantile((-0.25f64).exp()).unwrap(), 2., 1e-13);
    let pareto = Pareto::new(3.).unwrap();
    close(pareto.log_sf(2.).unwrap(), -3. * 2f64.ln(), 1e-13);
    close(pareto.mean().unwrap(), 1.5, 1e-13);
    close(pareto.variance().unwrap(), 0.75, 1e-13);
    let lomax = Affine::new(pareto, -1., 1.).unwrap();
    close(lomax.cdf(1.).unwrap(), 1. - 2f64.powi(-3), 1e-13);
    let levy = Affine::new(GeneralizedGamma::new(0.5, -1.).unwrap(), 0., 0.5).unwrap();
    close(levy.log_cdf(1.).unwrap(), scioxide_special_erfc_log(), 1e-9);
    let moyal = Moyal::<f64>::new();
    close(
        moyal.mean().unwrap(),
        0.5772156649015329 + core::f64::consts::LN_2,
        1e-13,
    );
    close(
        moyal.variance().unwrap(),
        core::f64::consts::PI.powi(2) / 2.,
        1e-13,
    );
    close(moyal.quantile(moyal.cdf(1.).unwrap()).unwrap(), 1., 1e-8);
    let far = moyal.isf(1e-300).unwrap();
    close(moyal.log_sf(far).unwrap(), 1e-300_f64.ln(), 1e-13);
    close(
        moyal.log_sf(1000.).unwrap(),
        -500. + (2. / core::f64::consts::PI).ln() / 2.,
        1e-13,
    );
    let huge = 1e308_f64;
    let gp = GeneralizedPareto::new(4.0_f64).unwrap();
    close(
        gp.log_sf(huge).unwrap(),
        -(4f64.ln() + huge.ln()) / 4.,
        1e-12,
    );
    let gev = GeneralizedExtremeValue::new(-4.0_f64).unwrap();
    close(
        gev.log_sf(huge).unwrap(),
        (4f64.ln() + huge.ln()) / -4.,
        1e-12,
    );
    let inv = InverseWeibull::new(2.0_f64).unwrap();
    close(inv.log_sf(huge).unwrap(), -2. * huge.ln(), 1e-12);
    let tiny_gp = GeneralizedPareto::new(1e-300_f64).unwrap();
    close(tiny_gp.log_sf(1.).unwrap(), -1., 1e-12);
    close(
        tiny_gp.quantile(0.5).unwrap(),
        core::f64::consts::LN_2,
        1e-12,
    );
    let tiny_gev = GeneralizedExtremeValue::new(1e-300_f64).unwrap();
    close(tiny_gev.log_cdf(1.).unwrap(), -(-1f64).exp(), 1e-12);
}
fn scioxide_special_erfc_log() -> f64 {
    0.31731050786291415_f64.ln()
}

#[test]
fn stable_and_breit_wigner_properties() {
    // Independent 40-digit characteristic-function Fourier integrals.
    for (alpha, beta, x, pdf, cdf) in [
        (0.7, 0.5, 1., 0.29754517666712974, 0.43503104759894256),
        (1., 1., 0., 0.26224012637535166, 0.3652387015123748),
        (1.5, 0.5, 1., 0.14151357067986658, 0.7967806891350713),
        (1.5, 0.5, -1., 0.26804649655446153, 0.3219871538583492),
    ] {
        let d = LevyStable::<f64>::new(alpha, beta).unwrap();
        close(d.log_pdf(x).unwrap().exp(), pdf, 3e-6);
        close(d.log_cdf(x).unwrap().exp(), cdf, 3e-6);
        close(d.log_sf(x).unwrap().exp(), 1. - cdf, 3e-6);
    }
    for (alpha, beta) in [
        (0.5, 1.),
        (0.5, -1.),
        (1., 0.),
        (2., 0.),
        (1., 1.),
        (1.5, 0.5),
    ] {
        let d = LevyStable::<f64>::new(alpha, beta).unwrap();
        for x in [-1., 0., 1., 3.] {
            let lc = d
                .log_cdf(x)
                .unwrap_or_else(|e| panic!("lower alpha={alpha} beta={beta} x={x}: {e:?}"));
            let ls = d
                .log_sf(x)
                .unwrap_or_else(|e| panic!("upper alpha={alpha} beta={beta} x={x}: {e:?}"));
            close(lc.exp() + ls.exp(), 1., 2e-6);
            let pdf = d
                .log_pdf(x)
                .unwrap_or_else(|e| panic!("density alpha={alpha} beta={beta} x={x}: {e:?}"))
                .exp();
            assert!(
                pdf >= 0. && pdf.is_finite(),
                "alpha={alpha} beta={beta} x={x} pdf={pdf}"
            );
        }
        if alpha == 1.5 {
            close(d.mean().unwrap(), 0., 1e-13);
            assert!(d.variance().is_err());
        }
    }
    let stable = LevyStable::<f64>::new(1.5, 0.5).unwrap();
    let far = stable.log_sf(1e6).unwrap();
    assert!(far.is_finite() && far < -15., "stable far tail {far}");
    let landau = LevyStable::<f64>::new(1., 1.).unwrap();
    let far = landau.log_sf(1000.).unwrap();
    close(far.exp(), 2. / (core::f64::consts::PI * 1000.), 0.02);
    let w = RelativisticBreitWigner::<f64>::new(1.).unwrap();
    close(
        w.log_pdf(1.).unwrap().exp(),
        4. / (core::f64::consts::PI * (1. + 2f64.sqrt()).sqrt()),
        1e-12,
    );
    for x in [0.01, 0.5, 1., 2., 10., 100.] {
        close(
            w.log_cdf(x).unwrap().exp() + w.log_sf(x).unwrap().exp(),
            1.,
            2e-6,
        );
    }
    close(w.mean().unwrap(), 0.965, 0.01);
    assert!(w.variance().unwrap() > 0.);
}

#[test]
fn both_widths_sampling_and_budget() {
    let mut rng = SplitMix64::from_seed(537);
    let d = Burr12::<f32>::new(1.5, 2.).unwrap();
    for _ in 0..100 {
        let x = d.sample(&mut rng).unwrap();
        assert!(x >= 0. && x.is_finite());
    }
    assert!(GeneralizedPareto::<f32>::new(f32::NAN).is_err());
    assert!(InverseWeibull::<f32>::new(0.).is_err());
    assert!(Pareto::<f32>::new(-1.).is_err());
    assert!(LevyStable::<f32>::new(0., 0.).is_err());
    assert!(GeneralizedPareto::<f32>::new(-f32::from_bits(1)).is_err());
    assert!(GeneralizedExtremeValue::<f32>::new(f32::MIN_POSITIVE).is_ok());
    assert!(GeneralizedExtremeValue::<f32>::new(f32::from_bits(1)).is_err());
    let tiny = LawOptions::default()
        .with_tolerance(Tolerance::<f64>::from_sqrt_epsilon())
        .with_work_budget(Budget::ONE);
    assert!(
        LevyStable::with_options(1.5, 0.3, tiny)
            .unwrap()
            .log_pdf(1.)
            .is_err()
    );
    let tiny_bw = LawOptions::default()
        .with_tolerance(Tolerance::<f64>::from_sqrt_epsilon())
        .with_work_budget(Budget::ONE);
    assert!(
        RelativisticBreitWigner::with_options(1., tiny_bw)
            .unwrap()
            .log_sf(1.)
            .is_err()
    );
    let narrow = RelativisticBreitWigner::<f64>::new(1e-20).unwrap();
    let x = 2e-10;
    close(
        narrow.log_cdf(x).unwrap().exp() + narrow.log_sf(x).unwrap().exp(),
        1.,
        1e-5,
    );
    assert!(narrow.log_sf(x).unwrap() < 0.);
}

use crate::statistics;
