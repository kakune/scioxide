//! Uniform mathematical-domain and moment policies across law families.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    DomainReason, RealScalar, SciOxideError, SciOxideErrorKind, SciOxideErrorView,
};
use scioxide_random::*;

fn undefined<T: core::fmt::Debug>(name: &str, result: Result<T, SciOxideError>) {
    let error = result.unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DomainError, "{name}");
    assert!(
        matches!(error.view(), SciOxideErrorView::DomainError {
        reason: DomainReason::Undefined, argument, ..
    } if argument == scioxide_core::argument!("self")),
        "{name}: {error}"
    );
}
fn moments<S: RealScalar, D: Mean<S> + Variance<S>>(name: &str, law: D) {
    undefined(name, law.mean());
    undefined(name, law.variance());
}
fn scalar_moments<S: RealScalar>() {
    let c = |x| S::try_from_f64_lossy(x).unwrap();
    moments("StudentT", StudentT::new(c(1.)).unwrap());
    moments("Pareto", Pareto::new(c(1.)).unwrap());
    moments("GeneralizedPareto", GeneralizedPareto::new(c(1.)).unwrap());
    moments("BetaPrime", BetaPrime::new(c(2.), c(1.)).unwrap());
    moments("Mielke", Mielke::new(c(2.), c(1.)).unwrap());
    moments("Burr12", Burr12::new(c(1.), c(1.)).unwrap());
    moments("InverseWeibull", InverseWeibull::new(c(1.)).unwrap());
    moments(
        "GeneralizedExtremeValue",
        GeneralizedExtremeValue::new(c(-1.)).unwrap(),
    );
    moments("Kappa4", Kappa4::new(c(0.), c(-1.)).unwrap());
    moments("TukeyLambda", TukeyLambda::new(c(-1.)).unwrap());
    moments(
        "GeneralizedGamma",
        GeneralizedGamma::new(c(1.), c(-1.)).unwrap(),
    );
    moments("CrystalBall", CrystalBall::new(c(2.), c(2.)).unwrap());
    moments(
        "JonesFaddySkewT",
        JonesFaddySkewT::new(c(0.5), c(2.)).unwrap(),
    );
    moments(
        "GeneralizedHyperbolic",
        GeneralizedHyperbolic::with_options(
            c(-0.5),
            c(1.),
            c(1.),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    moments("LevyStable", LevyStable::new(c(1.), c(0.)).unwrap());
    moments(
        "DoubleParetoLogNormal",
        DoubleParetoLogNormal::new(c(0.), c(1.), c(1.), c(1.)).unwrap(),
    );
    moments(
        "BetaNegativeBinomial",
        BetaNegativeBinomial::new(c(2.), c(1.), c(2.)).unwrap(),
    );
    moments("YuleSimon", YuleSimon::new(c(1.)).unwrap());
    moments("Zipf", Zipf::new(c(2.)).unwrap());
    moments(
        "FisherF",
        FisherF::with_options(
            c(3.),
            c(2.),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    moments(
        "NoncentralF",
        NoncentralF::with_options(
            c(3.),
            c(2.),
            c(1.),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    moments(
        "NoncentralStudentT",
        NoncentralStudentT::with_options(
            c(1.),
            c(1.),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    moments(
        "StudentizedRange",
        StudentizedRange::with_options(
            c(3.),
            c(1.),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap(),
    );
    // Existing first moments and divergent second moments must remain distinct.
    let t = StudentT::new(c(2.)).unwrap();
    assert_eq!(t.mean().unwrap(), S::ZERO);
    undefined("StudentT variance boundary", t.variance());
    assert!(StudentT::new(c(3.)).unwrap().variance().unwrap() > S::ZERO);
    assert_eq!(
        Binomial::new(10, S::ZERO).unwrap().variance().unwrap(),
        S::ZERO
    );
}
#[test]
fn all_scalar_moment_families() {
    scalar_moments::<f32>();
    scalar_moments::<f64>();
}
#[test]
fn all_multivariate_moment_families() {
    let identity = scioxide_linalg::Matrix::<f64>::identity(1).unwrap();
    let t = MultivariateStudentT::with_options(
        &[0.],
        &identity,
        1.,
        Default::default(),
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    undefined("MultivariateStudentT mean", t.mean());
    undefined("MultivariateStudentT variance", t.variance());
    undefined("MultivariateStudentT covariance", t.covariance());
    let nig = NormalInverseGamma::with_options(
        0.,
        1.,
        1.,
        1.,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    undefined("NormalInverseGamma mean", nig.mean());
    undefined("NormalInverseGamma variance", nig.variance());
    let wishart = InverseWishart::with_options(
        2.,
        &identity,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    undefined("InverseWishart mean", wishart.mean());
    undefined("InverseWishart variance", wishart.variance());
    let t = MatrixT::with_options(
        &identity,
        &identity,
        &identity,
        1.,
        scioxide_random::LawOptions::default()
            .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
    )
    .unwrap();
    undefined("MatrixT mean", t.mean());
    undefined("MatrixT variance", t.variance());
}
#[test]
fn positive_parameters_have_one_classification() {
    for value in [0., -1., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let expected = if value.is_finite() {
            SciOxideErrorKind::DomainError
        } else {
            SciOxideErrorKind::NonFinite
        };
        let errors = [
            Normal::new(0., value).unwrap_err(),
            Gamma::new(value, 1.).unwrap_err(),
            Beta::new(value, 1.).unwrap_err(),
            Cauchy::new(0., value).unwrap_err(),
            LogNormal::new(0., value).unwrap_err(),
            Exponential::new(value).unwrap_err(),
            StudentT::new(value).unwrap_err(),
            ChiSquared::new(value).unwrap_err(),
            Pareto::new(value).unwrap_err(),
            RelativisticBreitWigner::new(value).unwrap_err(),
            FisherF::with_options(
                value,
                3.,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap_err(),
            NormalInverseGamma::with_options(
                0.,
                value,
                3.,
                1.,
                scioxide_random::LawOptions::default()
                    .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
            )
            .unwrap_err(),
        ];
        for error in errors {
            assert_eq!(error.kind(), expected, "{value}: {error}");
            if value.is_finite() {
                assert!(matches!(
                    error.view(),
                    SciOxideErrorView::DomainError {
                        reason: DomainReason::OutsideDomain,
                        ..
                    }
                ));
            }
        }
    }
}

#[test]
fn checked_log_sums_preserve_singular_density_limits() {
    let law = Mixture::new(
        vec![Beta::new(0.5, 2.).unwrap(), Beta::new(0.5, 3.).unwrap()],
        &[1., 1.],
    )
    .unwrap();
    assert_eq!(law.log_pdf(0.).unwrap(), f64::INFINITY);
    assert_eq!(law.pdf(0.).unwrap(), f64::INFINITY);
    assert_eq!(law.log_cdf(0.).unwrap(), f64::NEG_INFINITY);
    assert_eq!(law.log_sf(0.).unwrap(), 0.);
    let endpoint = Beta::new(0.5_f64, 1.0).unwrap();
    assert_eq!(
        OrderStatistic::new(&endpoint, 1, 2)
            .unwrap()
            .pdf(0.0)
            .unwrap(),
        f64::INFINITY
    );
    assert!(
        OrderStatistic::new(&endpoint, 2, 2)
            .unwrap()
            .log_pdf(0.0)
            .is_err()
    );
    let base = Beta::new(0.5_f64, 2.0).unwrap();
    assert_eq!(
        Affine::new(&base, 1.0, 2.0).unwrap().pdf(1.0).unwrap(),
        f64::INFINITY
    );
    assert_eq!(
        Truncated::new(&base, 0.0, 0.5).unwrap().pdf(0.0).unwrap(),
        f64::INFINITY
    );
    assert_eq!(
        Transformed::abs(&base).unwrap().pdf(0.0).unwrap(),
        f64::INFINITY
    );
    assert_eq!(
        Transformed::exp(&base).unwrap().pdf(1.0).unwrap(),
        f64::INFINITY
    );
}

#[test]
fn uniform_circular_direction_is_undefined_but_variance_exists() {
    let von_mises = VonMises::new(0., 0.).unwrap();
    let cauchy = WrappedCauchy::new(0., 0.).unwrap();
    undefined("VonMises mean direction", von_mises.mean_direction());
    undefined("WrappedCauchy mean direction", cauchy.mean_direction());
    assert_eq!(von_mises.circular_variance().unwrap(), 1.);
    assert_eq!(cauchy.circular_variance().unwrap(), 1.);
}

#[test]
fn matrix_law_degrees_share_domain_and_nonfinite_policies() {
    let identity = scioxide_linalg::Matrix::<f64>::identity(2).unwrap();
    for value in [0., -1., f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
        let error = MultivariateStudentT::with_options(
            &[0., 0.],
            &identity,
            value,
            Default::default(),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap_err();
        let expected = if value.is_finite() {
            SciOxideErrorKind::DomainError
        } else {
            SciOxideErrorKind::NonFinite
        };
        assert_eq!(error.kind(), expected, "{value}: {error}");
    }
    assert!(
        MultivariateStudentT::gaussian_limit_with_options(
            &[0., 0.],
            &identity,
            Default::default(),
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215))
        )
        .is_ok()
    );
    for error in [
        Wishart::with_options(
            1.,
            &identity,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap_err(),
        InverseWishart::with_options(
            1.,
            &identity,
            scioxide_random::LawOptions::default()
                .with_work_budget(scioxide_core::Budget::ONE.saturating_add(16_777_215)),
        )
        .unwrap_err(),
    ] {
        assert!(matches!(
            error.view(),
            SciOxideErrorView::DomainError {
                reason: DomainReason::OutsideDomain,
                ..
            }
        ));
    }
}

#[test]
fn composed_law_parameters_use_mathematical_and_weight_policies() {
    let normal = || Normal::new(0.0, 1.0).unwrap();
    let errors = [
        Histogram::from_masses(&[0., 1.], &[-1.]).unwrap_err(),
        Histogram::from_densities(&[0., 1.], &[0.]).unwrap_err(),
        Affine::new(normal(), 0., 0.).unwrap_err(),
        Truncated::new(normal(), 1., -1.).unwrap_err(),
        OrderStatistic::new(normal(), 0, 5).unwrap_err(),
    ];
    for error in errors {
        assert!(matches!(
            error.view(),
            SciOxideErrorView::DomainError {
                reason: DomainReason::OutsideDomain,
                ..
            }
        ));
    }
    // Mixtures share the alias sampler's structural weight validation.
    for error in [
        Histogram::from_masses(&[0., 1.], &[1., 2.]).unwrap_err(),
        Mixture::new(vec![normal()], &[-1.]).unwrap_err(),
        Mixture::new(vec![normal()], &[0.]).unwrap_err(),
    ] {
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    }
}
