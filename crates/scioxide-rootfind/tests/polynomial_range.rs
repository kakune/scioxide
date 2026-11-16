//! Nonuniform root scales retain every nonzero coefficient and original-input diagnostics.
#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]

use scioxide_core::{Complex, NumericScalar, RankThreshold, RealScalar};
use scioxide_rootfind::{
    PolynomialRootAlgorithm, PolynomialRootMethod, PolynomialRootOptions,
    polynomial_roots_with_options,
};

#[test]
fn rank_classification_never_trims_the_polynomial() {
    rank_classification::<f32>();
    rank_classification::<f64>();
}
fn rank_classification<S: RealScalar + NumericScalar<Real = S>>() {
    let c = [
        S::ONE,
        -(S::TWO + S::ONE),
        S::TWO,
        S::try_from_f64_lossy(1e-30).unwrap(),
    ];
    for (threshold, degree, rank) in [(S::ZERO, 3, 4), (S::EPSILON, 2, 3), (S::TWO * S::TWO, 0, 0)]
    {
        let roots = polynomial_roots_with_options(
            &c,
            &PolynomialRootOptions::default()
                .with_rank_threshold(RankThreshold::new(threshold, S::ZERO).unwrap()),
        )
        .unwrap();
        assert_eq!(roots.len(), 3);
        assert_eq!(roots.report().effective_degree(), degree);
        assert_eq!(roots.report().effective_rank(), rank);
        assert!(roots.report().termination().is_success(), "{roots:?}");
    }
}
#[test]
fn exact_zero_factors_and_complex_analytic_roots_are_reported() {
    for c in [
        &[0.0_f64, 0.0, 0.0, 1.0][..],
        &[0.0, 0.0, 1.0, -1e200, 1.0][..],
    ] {
        let roots = polynomial_roots_with_options(
            c,
            &PolynomialRootOptions::default().with_method(PolynomialRootMethod::Companion),
        )
        .unwrap();
        assert!(roots.report().termination().is_success(), "{roots:?}");
        assert_eq!(
            roots.report().effective_method(),
            Some(PolynomialRootAlgorithm::Analytic)
        );
        assert_eq!(
            roots.iter().filter(|r| r.re == 0.0 && r.im == 0.0).count(),
            if c.len() == 4 { 3 } else { 2 }
        );
    }
    let c = [1.0_f64, -1e200, 1.0].map(|a| Complex::new(a * 0.5, a * 0.5));
    let roots = polynomial_roots_with_options(&c, &PolynomialRootOptions::default()).unwrap();
    assert!(roots.report().termination().is_success(), "{roots:?}");
    assert!((roots[0].re / 1e-200 - 1.0).abs() < 32.0 * f64::EPSILON);
    assert!((roots[1].re / 1e200 - 1.0).abs() < 32.0 * f64::EPSILON);
}

#[test]
fn unrepresentable_root_ranges_publish_unsuccessful_reports() {
    use scioxide_rootfind::RootTermination;
    for c in [
        &[1.0_f64, -1.0, f64::from_bits(1)][..],
        &[1.0, f64::MAX, 0.0, f64::from_bits(1)][..],
    ] {
        let roots = polynomial_roots_with_options(c, &PolynomialRootOptions::default()).unwrap();
        assert!(
            matches!(
                roots.report().termination(),
                RootTermination::NumericalFailure(_)
            ),
            "{roots:?}"
        );
        assert!(roots.iter().all(|r| r.re.is_finite() && r.im.is_finite()));
        assert!(roots.diagnostics().iter().all(|d| !d.is_validated()));
    }
}

#[cfg(feature = "std")]
#[test]
fn stopped_before_execution_reports_no_effective_algorithm() {
    let options = PolynomialRootOptions::<f64>::default();
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    let mut w = PolynomialRootWorkspace::workspace(2, &options).unwrap();
    let mut out = [Complex::new(99.0, 99.0); 2];
    let report = polynomial_roots_into(
        &[-1.0, 0.0, 1.0],
        &options,
        &mut out,
        &mut w,
        ExecutionContext::sequential().with_cancellation(&token),
    )
    .unwrap();
    assert_eq!(report.effective_method(), None);
    assert!(!report.termination().is_success());
    assert_eq!(report.counts().total_providers().unwrap(), 0);
}

#[cfg(feature = "std")]
use scioxide_core::ExecutionContext;
#[cfg(feature = "std")]
use scioxide_rootfind::{PolynomialRootWorkspace, polynomial_roots_into};
