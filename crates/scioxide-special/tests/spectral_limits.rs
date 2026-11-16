//! Truncation, work and focal-domain failures have distinct recovery paths.
#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::{
    Budget, Complex, ConvergenceReason, RealScalar, SciOxideError, SciOxideErrorView,
};
use scioxide_special::{
    MathieuParity, MathieuPlan, RadialKind, SpectralNormalization, SpectralOptions, SpectralTerms,
    SpheroidalKind, SpheroidalPlan,
};

fn options() -> SpectralOptions<f64> {
    SpectralOptions::default()
        .with_iteration_budget(Budget::try_from(2_usize.saturating_pow(40)).unwrap())
}

fn mathieu(
    order: u32,
    q: f64,
    options: SpectralOptions<f64>,
) -> Result<MathieuPlan<f64>, SciOxideError> {
    MathieuPlan::new(
        order,
        MathieuParity::Even,
        q,
        SpectralNormalization::Standard,
        options,
    )
}

fn truncation(error: SciOxideError) {
    let SciOxideErrorView::ConvergenceFailure {
        reason,
        diagnostics,
        detail,
    } = error.view()
    else {
        panic!("unexpected error: {error:?}");
    };
    assert_eq!(reason, ConvergenceReason::TruncationLimit);
    assert!(diagnostics.counts().total().unwrap().iterations() > 0);
    assert!(diagnostics.residual().unwrap().raw_norm().is_finite());
    assert!(detail.contains("SpectralTerms::maximum=256"), "{detail}");
    for field in [
        "terms=256",
        "refinements=",
        "eigen_residual=",
        "coefficient_change=",
        "truncation_bound=",
        "work=",
        "preparation_bytes=",
    ] {
        assert!(detail.contains(field), "{detail}");
    }
    assert!(!detail.contains("budget was exhausted"));
}

#[test]
fn truncation_cap_retains_diagnostics_and_raising_it_recovers() {
    truncation(mathieu(0, 1e7, options()).unwrap_err());
    let plan = mathieu(
        0,
        1e7,
        options().with_terms(SpectralTerms::new(16, 1024).unwrap()),
    )
    .unwrap();
    let expected = -19_993_675.694_689_546;
    assert!((plan.characteristic_value().unwrap() - expected).abs() < 1e-8);
    assert!(plan.diagnostics().terms() > 256);
    for kind in [SpheroidalKind::Prolate, SpheroidalKind::Oblate] {
        truncation(
            SpheroidalPlan::new(kind, 0, 0, 1e4, SpectralNormalization::Standard, options())
                .unwrap_err(),
        );
    }
}

#[test]
fn impossible_refinements_are_rejected_before_allocating() {
    for (order, terms) in [
        (0, SpectralTerms::new(16, 16).unwrap()),
        (0, SpectralTerms::new(2, 8).unwrap()),
        (500, SpectralTerms::default()),
        (1000, SpectralTerms::default()),
    ] {
        let error = mathieu(order, 1.0, options().with_terms(terms)).unwrap_err();
        assert!(matches!(
            error.view(),
            SciOxideErrorView::InvalidInput { .. }
        ));
        if order == 0 {
            assert!(mathieu(order, 0.0, options().with_terms(terms)).is_ok());
        }
    }
    let terms = SpectralTerms::new(16, 16).unwrap();
    let error = SpheroidalPlan::new(
        SpheroidalKind::Oblate,
        0,
        0,
        1.0,
        SpectralNormalization::Standard,
        options().with_terms(terms),
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::InvalidInput { .. }
    ));
}

#[test]
fn large_parameters_have_actionable_failures() {
    for q in [1.0, 1e4, 1e7, 1e8] {
        for order in [0, 64, 494] {
            match mathieu(order, q, SpectralOptions::default()) {
                Ok(plan) => assert!(plan.characteristic_value().unwrap().is_finite()),
                Err(error) => truncation(error),
            }
        }
    }
    let plan = mathieu(
        1000,
        1.0,
        options().with_terms(SpectralTerms::new(16, 1024).unwrap()),
    )
    .unwrap();
    assert!((plan.characteristic_value().unwrap() - 1e6).abs() < 1e-5);
    let error = mathieu(0, 1e7, options().with_iteration_budget(Budget::ONE)).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ConvergenceFailure {
            reason: ConvergenceReason::IterationLimit,
            ..
        }
    ));
}

fn focal_poles<S: RealScalar>()
where
    Complex<S>: scioxide_core::NumericScalar<Real = S>,
{
    let plan = SpheroidalPlan::new(
        SpheroidalKind::Oblate,
        0,
        0,
        S::ONE,
        SpectralNormalization::Standard,
        SpectralOptions::default(),
    )
    .unwrap();
    for sign in [S::ONE, -S::ONE] {
        for zero in [S::ZERO, -S::ZERO] {
            for kind in [RadialKind::Second, RadialKind::Third, RadialKind::Fourth] {
                let error = plan.radial(Complex::new(zero, sign), kind).unwrap_err();
                assert!(matches!(
                    error.view(),
                    SciOxideErrorView::DomainError {
                        reason: scioxide_core::DomainReason::Pole,
                        ..
                    }
                ));
            }
        }
    }
}

#[test]
fn oblate_higher_kinds_reject_both_focal_points() {
    focal_poles::<f32>();
    focal_poles::<f64>();
}
