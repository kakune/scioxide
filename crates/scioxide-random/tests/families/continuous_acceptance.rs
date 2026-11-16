//! Cross-law trait, inverse, and sampling checks for the completed inventory.
#![allow(clippy::unwrap_used)]
use scioxide_core::RealScalar;
use scioxide_random::{
    ContinuousLaw, Distribution, InverseSurvival, Quantile, SeedableRng, SplitMix64,
};

fn check<
    S: RealScalar,
    D: ContinuousLaw<S> + Quantile<S> + InverseSurvival<S> + Distribution<S>,
>(
    law: D,
    x: S,
    tol: f64,
    name: &str,
) {
    let lower = law.log_cdf(x).unwrap();
    let upper = law.log_sf(x).unwrap();
    assert!(
        (lower.exp().into_f64() + upper.exp().into_f64() - 1.).abs() < tol * 10.,
        "{name} tail complement at {x:?}: lower={lower:?} upper={upper:?}"
    );
    for p in [0.1, 0.5, 0.9] {
        let p = S::try_from_f64_lossy(p).unwrap();
        let q = law.quantile(p).unwrap();
        assert!(q.is_finite(), "interior quantile is nonfinite");
        let observed = law.cdf(q).unwrap().into_f64();
        assert!(
            (observed - p.into_f64()).abs() < tol,
            "{name} quantile roundtrip p={p:?} observed={observed} q={q:?}"
        );
        let u = law.isf(p).unwrap();
        assert!(u.is_finite(), "interior inverse survival is nonfinite");
        let sf = law.log_sf(u).unwrap().exp().into_f64();
        assert!(
            (sf - p.into_f64()).abs() < tol,
            "{name} survival roundtrip p={p:?} observed={sf} q={u:?}"
        );
    }
    let mut rng = SplitMix64::from_seed(534536);
    for _ in 0..16 {
        let sample = law.sample(&mut rng).unwrap();
        assert!(sample.is_finite());
        assert!(law.contains(sample).unwrap());
    }
}

#[test]
fn all_new_laws_f64() {
    macro_rules! law {
        ($law:expr,$x:expr) => {
            check($law, $x, 1e-6, stringify!($law))
        };
    }
    law!(scioxide_random::Cosine::<f64>::new().unwrap(), 0.2);
    law!(scioxide_random::DoubleGamma::<f64>::new(2.).unwrap(), 0.3);
    law!(
        scioxide_random::DoubleWeibull::<f64>::new(1.5).unwrap(),
        0.3
    );
    law!(
        scioxide_random::GeneralizedLogistic::<f64>::new(2.).unwrap(),
        0.3
    );
    law!(
        scioxide_random::GeneralizedNormal::<f64>::new(2.).unwrap(),
        0.3
    );
    law!(scioxide_random::GumbelRight::<f64>::new(), 0.3);
    law!(scioxide_random::HyperbolicSecant::<f64>::new(), 0.3);
    law!(
        scioxide_random::AsymmetricLaplace::<f64>::new(2.).unwrap(),
        0.3
    );
    law!(scioxide_random::Alpha::<f64>::new(2.).unwrap(), 1.);
    law!(scioxide_random::Anglit::<f64>::new(), 0.2);
    law!(scioxide_random::Argus::<f64>::new(2.).unwrap(), 0.5);
    law!(scioxide_random::BetaPrime::<f64>::new(2., 3.).unwrap(), 1.);
    law!(scioxide_random::Bradford::<f64>::new(2.).unwrap(), 0.5);
    law!(
        scioxide_random::GaussHypergeometric::<f64>::new(2., 3., 0.5, 0.3).unwrap(),
        0.5
    );
    law!(scioxide_random::IrwinHall::<f64>::new(5).unwrap(), 2.5);
    law!(
        scioxide_random::JohnsonSB::<f64>::new(0.5, 2.).unwrap(),
        0.5
    );
    law!(scioxide_random::Kappa4::<f64>::new(0.5, 0.3).unwrap(), 0.3);
    law!(scioxide_random::Mielke::<f64>::new(2., 3.).unwrap(), 1.);
    law!(
        scioxide_random::Trapezoid::<f64>::new(0.2, 0.8).unwrap(),
        0.5
    );
    law!(scioxide_random::TukeyLambda::<f64>::new(0.5).unwrap(), 0.3);
    law!(
        scioxide_random::ExponentialPower::<f64>::new(2.).unwrap(),
        1.
    );
    law!(scioxide_random::FatigueLife::<f64>::new(0.5).unwrap(), 1.);
    law!(
        scioxide_random::GeneralizedGamma::<f64>::new(2., 1.5).unwrap(),
        1.
    );
    law!(scioxide_random::Gompertz::<f64>::new(2.).unwrap(), 1.);
    law!(
        scioxide_random::InverseGaussian::<f64>::new(1.).unwrap(),
        1.
    );
    law!(
        scioxide_random::ReciprocalInverseGaussian::<f64>::new(1.).unwrap(),
        1.
    );
    law!(scioxide_random::Pearson3::<f64>::new(1.).unwrap(), 0.);
    law!(scioxide_random::WeibullMin::<f64>::new(2.).unwrap(), 1.);
}

#[test]
fn all_new_laws_f32() {
    macro_rules! law {
        ($law:expr,$x:expr) => {
            check($law, $x, 5e-3, stringify!($law))
        };
    }
    law!(scioxide_random::Cosine::<f32>::new().unwrap(), 0.2);
    law!(scioxide_random::DoubleGamma::<f32>::new(2.).unwrap(), 0.3);
    law!(
        scioxide_random::DoubleWeibull::<f32>::new(1.5).unwrap(),
        0.3
    );
    law!(
        scioxide_random::GeneralizedLogistic::<f32>::new(2.).unwrap(),
        0.3
    );
    law!(
        scioxide_random::GeneralizedNormal::<f32>::new(2.).unwrap(),
        0.3
    );
    law!(scioxide_random::GumbelRight::<f32>::new(), 0.3);
    law!(scioxide_random::HyperbolicSecant::<f32>::new(), 0.3);
    law!(
        scioxide_random::AsymmetricLaplace::<f32>::new(2.).unwrap(),
        0.3
    );
    law!(scioxide_random::Alpha::<f32>::new(2.).unwrap(), 1.);
    law!(scioxide_random::Anglit::<f32>::new(), 0.2);
    law!(scioxide_random::Argus::<f32>::new(2.).unwrap(), 0.5);
    law!(scioxide_random::BetaPrime::<f32>::new(2., 3.).unwrap(), 1.);
    law!(scioxide_random::Bradford::<f32>::new(2.).unwrap(), 0.5);
    law!(
        scioxide_random::GaussHypergeometric::<f32>::new(2., 3., 0.5, 0.3).unwrap(),
        0.5
    );
    law!(scioxide_random::IrwinHall::<f32>::new(5).unwrap(), 2.5);
    law!(
        scioxide_random::JohnsonSB::<f32>::new(0.5, 2.).unwrap(),
        0.5
    );
    law!(scioxide_random::Kappa4::<f32>::new(0.5, 0.3).unwrap(), 0.3);
    law!(scioxide_random::Mielke::<f32>::new(2., 3.).unwrap(), 1.);
    law!(
        scioxide_random::Trapezoid::<f32>::new(0.2, 0.8).unwrap(),
        0.5
    );
    law!(scioxide_random::TukeyLambda::<f32>::new(0.5).unwrap(), 0.3);
    law!(
        scioxide_random::ExponentialPower::<f32>::new(2.).unwrap(),
        1.
    );
    law!(scioxide_random::FatigueLife::<f32>::new(0.5).unwrap(), 1.);
    law!(
        scioxide_random::GeneralizedGamma::<f32>::new(2., 1.5).unwrap(),
        1.
    );
    law!(scioxide_random::Gompertz::<f32>::new(2.).unwrap(), 1.);
    law!(
        scioxide_random::InverseGaussian::<f32>::new(1.).unwrap(),
        1.
    );
    law!(
        scioxide_random::ReciprocalInverseGaussian::<f32>::new(1.).unwrap(),
        1.
    );
    law!(scioxide_random::Pearson3::<f32>::new(1.).unwrap(), 0.);
    law!(scioxide_random::WeibullMin::<f32>::new(2.).unwrap(), 1.);
}

#[test]
fn extreme_shape_regimes_f64() {
    macro_rules! law {
        ($law:expr,$x:expr) => {
            check($law, $x, 2e-4, stringify!($law))
        };
    }
    law!(scioxide_random::DoubleGamma::<f64>::new(0.2).unwrap(), 0.3);
    law!(
        scioxide_random::DoubleWeibull::<f64>::new(0.2).unwrap(),
        0.3
    );
    law!(
        scioxide_random::GeneralizedLogistic::<f64>::new(10.).unwrap(),
        1.
    );
    law!(
        scioxide_random::GeneralizedNormal::<f64>::new(0.5).unwrap(),
        1.
    );
    law!(
        scioxide_random::AsymmetricLaplace::<f64>::new(0.1).unwrap(),
        1.
    );
    law!(scioxide_random::Alpha::<f64>::new(0.1).unwrap(), 1.);
    law!(scioxide_random::Argus::<f64>::new(0.2).unwrap(), 0.5);
    law!(
        scioxide_random::BetaPrime::<f64>::new(0.3, 0.4).unwrap(),
        1.
    );
    law!(scioxide_random::Bradford::<f64>::new(0.001).unwrap(), 0.5);
    law!(
        scioxide_random::GaussHypergeometric::<f64>::new(0.5, 0.5, 2., -0.9).unwrap(),
        0.5
    );
    law!(scioxide_random::IrwinHall::<f64>::new(20).unwrap(), 10.);
    law!(
        scioxide_random::JohnsonSB::<f64>::new(5., 0.2).unwrap(),
        0.5
    );
    law!(scioxide_random::Kappa4::<f64>::new(-0.5, 0.2).unwrap(), 0.3);
    law!(scioxide_random::Mielke::<f64>::new(0.5, 0.8).unwrap(), 1.);
    law!(scioxide_random::Trapezoid::<f64>::new(0., 1.).unwrap(), 0.5);
    law!(scioxide_random::TukeyLambda::<f64>::new(-0.5).unwrap(), 1.);
    law!(
        scioxide_random::ExponentialPower::<f64>::new(0.4).unwrap(),
        1.
    );
    law!(scioxide_random::FatigueLife::<f64>::new(3.).unwrap(), 1.);
    law!(
        scioxide_random::GeneralizedGamma::<f64>::new(0.5, -1.).unwrap(),
        1.
    );
    law!(scioxide_random::Gompertz::<f64>::new(0.1).unwrap(), 1.);
    law!(
        scioxide_random::InverseGaussian::<f64>::new(0.1).unwrap(),
        0.1
    );
    law!(
        scioxide_random::ReciprocalInverseGaussian::<f64>::new(10.).unwrap(),
        1.
    );
    law!(scioxide_random::Pearson3::<f64>::new(-2.).unwrap(), 0.);
    law!(scioxide_random::WeibullMin::<f64>::new(0.3).unwrap(), 1.);
}

#[test]
fn domains_limits_and_budgets() {
    use scioxide_random::{Density, LogSurvival, Mean, Variance};
    assert!(scioxide_random::DoubleGamma::<f64>::new(0.).is_err());
    assert!(scioxide_random::DoubleWeibull::<f64>::new(-1.).is_err());
    assert!(scioxide_random::GeneralizedNormal::<f64>::new(f64::NAN).is_err());
    assert!(scioxide_random::BetaPrime::<f64>::new(1., 0.).is_err());
    assert!(scioxide_random::Trapezoid::<f64>::new(0.8, 0.2).is_err());
    assert!(scioxide_random::JohnsonSB::<f64>::new(0., 0.).is_err());
    let tiny_argus = scioxide_random::Argus::<f64>::new(1e-200).unwrap();
    assert!((tiny_argus.pdf(0.5).unwrap() - 3. * 0.5 * (1. - 0.25_f64).sqrt()).abs() < 1e-14);
    assert!((tiny_argus.mean().unwrap() - 3. * core::f64::consts::PI / 16.).abs() < 1e-14);
    assert!(scioxide_random::Argus::<f64>::new(f64::MAX).is_err());
    assert!(scioxide_random::GeneralizedGamma::<f64>::new(1., 0.).is_err());
    assert!(scioxide_random::GaussHypergeometric::<f64>::new(1., 1., 0., -1.).is_err());
    assert!(scioxide_random::IrwinHall::<f64>::new(0).is_err());
    assert_eq!(
        scioxide_random::IrwinHall::<f64>::with_options(
            20,
            scioxide_random::LawOptions::default().with_max_bytes(10)
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::ResourceLimit
    );
    assert!(
        scioxide_random::BetaPrime::<f64>::new(2., 1.)
            .unwrap()
            .mean()
            .is_err()
    );
    assert!(
        scioxide_random::Mielke::<f64>::new(1., 2.)
            .unwrap()
            .variance()
            .is_err()
    );
    assert!(
        scioxide_random::GeneralizedGamma::<f64>::new(1., -1.)
            .unwrap()
            .mean()
            .is_err()
    );
    assert!(
        scioxide_random::TukeyLambda::<f64>::new(-0.5)
            .unwrap()
            .variance()
            .is_err()
    );
    assert!(
        scioxide_random::Kappa4::<f64>::new(0.5, -1.)
            .unwrap()
            .mean()
            .is_err()
    );
    assert!(
        scioxide_random::Kappa4::<f64>::new(-1., 0.5)
            .unwrap()
            .variance()
            .is_err()
    );
    let moment_options = scioxide_random::LawOptions::default()
        .with_tolerance(scioxide_core::Tolerance::from_sqrt_epsilon())
        .with_work_budget(scioxide_core::Budget::ONE);
    assert!(
        scioxide_random::JohnsonSB::<f64>::with_options(0., 1., moment_options)
            .unwrap()
            .mean()
            .is_err()
    );
    for moment in [
        scioxide_random::ExponentialPower::<f64>::new(0.4)
            .unwrap()
            .mean(),
        scioxide_random::Gompertz::<f64>::new(0.1).unwrap().mean(),
        scioxide_random::JohnsonSB::<f64>::new(5., 0.2)
            .unwrap()
            .mean(),
        scioxide_random::Kappa4::<f64>::new(-0.5, 0.2)
            .unwrap()
            .mean(),
    ] {
        assert!(moment.unwrap().is_finite());
    }
    let k = scioxide_random::Kappa4::<f64>::new(1., 1.).unwrap();
    assert!((k.pdf(0.).unwrap() - 1.).abs() < 1e-14);
    assert!((k.pdf(1.).unwrap() - 1.).abs() < 1e-14);
    let gl = scioxide_random::GeneralizedLogistic::<f64>::new(2.).unwrap();
    assert!((gl.log_sf(1000.).unwrap() - (2_f64.ln() - 1000.)).abs() < 1e-12);
    let g = scioxide_random::GumbelRight::<f64>::new();
    assert!((g.log_sf(1000.).unwrap() + 1000.).abs() < 1e-12);
    let h = scioxide_random::HyperbolicSecant::<f64>::new();
    let expected = (2_f64 / core::f64::consts::PI).ln() - core::f64::consts::PI * 1000. / 2.;
    assert!((h.log_sf(1000.).unwrap() - expected).abs() < 1e-11);
    let k0 = scioxide_random::Kappa4::<f64>::new(0., 0.).unwrap();
    assert!((k0.log_sf(1000.).unwrap() + 1000.).abs() < 1e-12);
}

#[test]
fn representable_extreme_log_tails() {
    use scioxide_random::LogSurvival;
    let finite_tail = |result: Result<f64, scioxide_core::SciOxideError>| {
        let value = result.unwrap();
        assert!(value.is_finite() && value < 0., "tail={value}");
    };
    finite_tail(
        scioxide_random::BetaPrime::<f64>::new(2., 3.)
            .unwrap()
            .log_sf(1e100),
    );
    finite_tail(
        scioxide_random::Bradford::<f64>::new(2.)
            .unwrap()
            .log_sf(1. - 1e-12),
    );
    finite_tail(
        scioxide_random::GaussHypergeometric::<f64>::new(2., 3., 0.5, 0.3)
            .unwrap()
            .log_sf(1. - 1e-8),
    );
    finite_tail(
        scioxide_random::InverseGaussian::<f64>::new(1.)
            .unwrap()
            .log_sf(100.),
    );
    finite_tail(
        scioxide_random::ReciprocalInverseGaussian::<f64>::new(1.)
            .unwrap()
            .log_sf(1e4),
    );
    finite_tail(
        scioxide_random::TukeyLambda::<f64>::new(-1.)
            .unwrap()
            .log_sf(1e100),
    );
    finite_tail(
        scioxide_random::GeneralizedNormal::<f64>::new(2.)
            .unwrap()
            .log_sf(20.),
    );
}
