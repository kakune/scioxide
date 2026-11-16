#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]

use scioxide_core::Complex;
use scioxide_core::{Budget, NumericScalar, SciOxideError, Tolerance};
use scioxide_poly::PolynomialBasis;
use scioxide_poly::{ApproxPolynomial, MonomialBasis, checked_basis_evaluation};
use scioxide_rootfind::{
    PolynomialRootAlgorithm, PolynomialRootMethod, PolynomialRootOptions, PolynomialRootWorkspace,
    RootTermination,
};

fn roots_of_unity_f64(degree: usize) -> Vec<f64> {
    let mut coefficients = vec![0.0; degree + 1];
    coefficients[0] = -1.0;
    coefficients[degree] = 1.0;
    coefficients
}

fn approx_from_roots<S: NumericScalar>(
    roots: &[S],
) -> Result<ApproxPolynomial<S, MonomialBasis<S>>, SciOxideError> {
    ApproxPolynomial::from_roots(roots)
}

fn max_residual_f64(coefficients: &[f64], roots: &[Complex<f64>]) -> f64 {
    roots
        .iter()
        .map(|root| {
            checked_basis_evaluation(
                coefficients,
                *root,
                scioxide_poly::PolynomialRootBasisRef::Monomial,
            )
            .unwrap()
            .value()
            .unwrap()
            .norm()
        })
        .fold(0.0, f64::max)
}

fn max_normalized_residual_f32(coefficients: &[f32], roots: &[Complex<f32>]) -> f32 {
    roots
        .iter()
        .map(|root| {
            let magnitude = root.norm();
            let bound = coefficients
                .iter()
                .rev()
                .fold(0.0, |accumulator, coefficient| {
                    accumulator * magnitude + coefficient.abs()
                });
            checked_basis_evaluation(
                coefficients,
                *root,
                scioxide_poly::PolynomialRootBasisRef::Monomial,
            )
            .unwrap()
            .value()
            .unwrap()
            .norm()
                / bound
        })
        .fold(0.0, f32::max)
}

#[test]
fn every_explicit_method_solves_separated_real_roots() {
    let polynomial = approx_from_roots(&[-8.0_f64, -2.0, -0.5, 0.25, 1.5, 4.0]).unwrap();
    for method in [
        PolynomialRootMethod::Companion,
        PolynomialRootMethod::Aberth,
        PolynomialRootMethod::DurandKerner,
        PolynomialRootMethod::Laguerre,
    ] {
        let roots = scioxide_rootfind::polynomial_roots_with_options(
            polynomial.basis().coefficients(),
            &PolynomialRootOptions::default().with_method(method),
        )
        .unwrap();
        assert_eq!(roots.len(), polynomial.degree());
        assert!(
            max_residual_f64(polynomial.basis().coefficients(), &roots) <= 1.0e-7,
            "method {method:?}: {roots:?}"
        );
    }
}

#[test]
fn roots_of_unity_and_clustered_roots_are_residual_checked() {
    let unity = roots_of_unity_f64(50);
    let roots = scioxide_rootfind::polynomial_roots_with_options(
        &unity,
        &PolynomialRootOptions::default().with_method(PolynomialRootMethod::Aberth),
    )
    .unwrap();
    assert!(max_residual_f64(&unity, &roots) <= 1.0e-10);

    let clustered = approx_from_roots(&[0.98_f64, 0.99, 1.0, 1.01, 1.02]).unwrap();
    let roots = scioxide_rootfind::polynomial_roots_with_options(
        clustered.basis().coefficients(),
        &PolynomialRootOptions::default().with_method(PolynomialRootMethod::Aberth),
    )
    .unwrap();
    assert!(max_residual_f64(clustered.basis().coefficients(), &roots) <= 1.0e-10);
}

#[test]
fn scaled_repeated_conjugate_and_zero_roots_succeed() {
    let widely_scaled = approx_from_roots(&[-64.0_f64, -0.25, 0.5, 32.0]).unwrap();
    let roots = scioxide_rootfind::polynomial_roots_with_options(
        widely_scaled.basis().coefficients(),
        &PolynomialRootOptions::default(),
    )
    .unwrap();
    assert!(max_residual_f64(widely_scaled.basis().coefficients(), &roots) <= 1.0e-6);

    let repeated = [-1.0_f64, 3.0, -3.0, 1.0];
    let roots = scioxide_rootfind::polynomial_roots_with_options(
        &repeated,
        &PolynomialRootOptions::default().with_method(PolynomialRootMethod::Aberth),
    )
    .unwrap();
    assert!(max_residual_f64(&repeated, &roots) <= 1.0e-10);

    let conjugates = [4.0_f64, 0.0, 5.0, 0.0, 1.0];
    let roots = scioxide_rootfind::polynomial_roots_with_options(
        &conjugates,
        &PolynomialRootOptions::default().with_method(PolynomialRootMethod::Laguerre),
    )
    .unwrap();
    assert!(max_residual_f64(&conjugates, &roots) <= 1.0e-10);
    assert!(roots.windows(2).any(|pair| pair[0] == pair[1].conj()));

    let zero_roots = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 1.0];
    let roots = scioxide_rootfind::polynomial_roots_with_options(
        &zero_roots,
        &PolynomialRootOptions::default(),
    )
    .unwrap();
    assert_eq!(
        roots
            .iter()
            .filter(|root| **root == Complex::new(0.0, 0.0))
            .count(),
        3
    );
    assert!(max_residual_f64(&zero_roots, &roots) <= 1.0e-10);
}

#[test]
fn uniform_power_of_two_coefficient_scaling_preserves_roots() {
    let coefficients = [-6.0_f64, 11.0, -6.0, 1.0];
    let options = PolynomialRootOptions::default();
    let reference = scioxide_rootfind::polynomial_roots_with_options(
        &coefficients,
        &options.clone().with_method(PolynomialRootMethod::Companion),
    )
    .unwrap();

    for exponent in [-500, 500] {
        let factor = 2.0_f64.powi(exponent);
        let scaled = coefficients.map(|coefficient| coefficient * factor);
        let roots = scioxide_rootfind::polynomial_roots_with_options(
            &scaled,
            &options.clone().with_method(PolynomialRootMethod::Companion),
        )
        .unwrap();
        reference
            .iter()
            .zip(roots.iter())
            .for_each(|(expected, actual)| {
                assert!((*actual - *expected).norm() <= 2.0e-12);
            });
        assert!(
            roots.report().backward_error().unwrap()
                <= options.backward_error_tolerance().tolerance().relative()
        );
        assert_eq!(roots.report().coefficient_scale(), 11.0 * factor);
    }
}

#[test]
fn f32_and_f64_cover_precision_sensitive_fixtures() {
    let f32_fixtures = [
        vec![-1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0],
        approx_from_roots(&[0.92_f32, 0.96, 1.0, 1.04, 1.08])
            .unwrap()
            .basis()
            .coefficients()
            .to_vec(),
        vec![-1.0_f32, 3.0, -3.0, 1.0],
        vec![4.0_f32, 0.0, 5.0, 0.0, 1.0],
        approx_from_roots(&[-64.0_f32, -0.25, 0.5, 32.0])
            .unwrap()
            .basis()
            .coefficients()
            .to_vec(),
        vec![0.0_f32, 0.0, 0.0, 1.0, 0.0, 1.0],
        approx_from_roots(&(1_u16..=6).map(f32::from).collect::<Vec<_>>())
            .unwrap()
            .basis()
            .coefficients()
            .to_vec(),
    ];
    for coefficients in f32_fixtures {
        let roots = scioxide_rootfind::polynomial_roots_with_options(
            &coefficients,
            &PolynomialRootOptions::default().with_method(PolynomialRootMethod::Aberth),
        )
        .unwrap();
        assert!(
            max_normalized_residual_f32(&coefficients, &roots) <= 2.0e-4,
            "coefficients={coefficients:?}, roots={roots:?}"
        );
    }

    let expected: Vec<f64> = (1..=10).map(f64::from).collect();
    let wilkinson = approx_from_roots(&expected).unwrap();
    let roots = scioxide_rootfind::polynomial_roots_with_options(
        wilkinson.basis().coefficients(),
        &PolynomialRootOptions::default().with_method(PolynomialRootMethod::Aberth),
    )
    .unwrap();
    assert!(max_residual_f64(wilkinson.basis().coefficients(), &roots) <= 1.0e-3);
}

#[test]
fn preflight_errors_are_typed_and_transactional() {
    let options = PolynomialRootOptions::default();
    for coefficients in [vec![], vec![1.0], vec![1.0, 0.0]] {
        assert_eq!(
            scioxide_rootfind::polynomial_roots_with_options(&coefficients, &options)
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::InvalidInput
        );
    }
    assert_eq!(
        scioxide_rootfind::polynomial_roots_with_options(&[f64::NAN, 1.0], &options)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );

    let coefficients = [-1.0_f64, 0.0, 0.0, 1.0];
    let sentinel = Complex::new(42.0, -7.0);
    let mut output = vec![sentinel; 3];
    let mut workspace = PolynomialRootWorkspace::new();
    let forced_failure = PolynomialRootOptions::default()
        .with_backward_error_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
            Tolerance::new(0.0, 0.0).unwrap(),
        ))
        .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(
            Tolerance::new(0.0, 0.0).unwrap(),
        ))
        .with_max_iterations(Budget::ONE)
        .with_restart_budget(Budget::ONE)
        .with_polish(false);
    PolynomialRootWorkspace::workspace_requirements(
        coefficients.len() - 1,
        &PolynomialRootOptions::<f64>::default().with_method(PolynomialRootMethod::Aberth),
    )
    .and_then(|requirements| workspace.try_reserve(&requirements))
    .unwrap();
    let result = scioxide_rootfind::polynomial_roots_into(
        &coefficients,
        &forced_failure.with_method(PolynomialRootMethod::Aberth),
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    );
    let report = result.unwrap();
    assert_eq!(
        report.termination(),
        scioxide_rootfind::RootTermination::IterationLimit
    );
    assert_ne!(output, vec![sentinel; 3]);
    for (root, diagnostic) in output.iter().zip(workspace.diagnostics()) {
        assert!(diagnostic.is_available());
        let independent = checked_basis_evaluation(
            &coefficients,
            *root,
            scioxide_poly::PolynomialRootBasisRef::Monomial,
        )
        .unwrap()
        .normalized_residual();
        assert_eq!(diagnostic.backward_error(), Some(independent));
    }

    let mut short_output = vec![sentinel; 2];
    assert_eq!(
        scioxide_rootfind::polynomial_roots_into(
            &coefficients,
            &options,
            &mut short_output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential_native()
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(short_output, vec![sentinel; 2]);
}

#[test]
fn workspace_growth_determinism_and_stats_are_observable() {
    let coefficients = roots_of_unity_f64(20);
    let options = PolynomialRootOptions::default();
    let mut workspace = PolynomialRootWorkspace::new();
    PolynomialRootWorkspace::workspace_requirements(
        3,
        &PolynomialRootOptions::<f64>::default().with_method(PolynomialRootMethod::Aberth),
    )
    .and_then(|requirements| workspace.try_reserve(&requirements))
    .unwrap();
    PolynomialRootWorkspace::workspace_requirements(
        20,
        &PolynomialRootOptions::<f64>::default().with_method(PolynomialRootMethod::Aberth),
    )
    .and_then(|requirements| workspace.try_reserve(&requirements))
    .unwrap();
    assert_eq!(
        PolynomialRootWorkspace::workspace_requirements(
            usize::MAX,
            &PolynomialRootOptions::<f64>::default().with_method(PolynomialRootMethod::Aberth)
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::SizeOverflow
    );

    let mut first = vec![Complex::new(0.0, 0.0); 20];
    let first_stats = scioxide_rootfind::polynomial_roots_into(
        &coefficients,
        &options.clone().with_method(PolynomialRootMethod::Aberth),
        &mut first,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    let mut second = vec![Complex::new(0.0, 0.0); 20];
    let second_stats = scioxide_rootfind::polynomial_roots_into(
        &coefficients,
        &options.clone().with_method(PolynomialRootMethod::Aberth),
        &mut second,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(first_stats, second_stats);
    assert_eq!(first_stats.requested_method(), PolynomialRootMethod::Aberth);
    assert_eq!(
        first_stats.effective_method(),
        Some(PolynomialRootAlgorithm::Aberth)
    );
    assert!(first_stats.iterations() > 0);
    assert!(first_stats.active_root_updates() >= 20);
    assert!(first_stats.pair_interactions() >= 20 * 19);
    assert!(first_stats.backward_error().unwrap() <= f64::sqrt(f64::EPSILON));
    assert_eq!(first_stats.effective_degree(), 20);
    assert_eq!(first_stats.effective_rank(), 2);
    assert_eq!(first_stats.termination(), RootTermination::Complete);
    assert_eq!(
        first_stats.policy(),
        options.backward_error_tolerance().tolerance()
    );
    assert!(first_stats.coefficient_scale().is_finite());
    assert!(first_stats.max_root_condition().value().is_some());
}

#[test]
fn auto_reports_degree_based_method_selection() {
    let cubic = [-1.0_f64, 0.0, 0.0, 1.0];
    let mut roots = vec![Complex::new(0.0, 0.0); 3];
    let mut workspace = PolynomialRootWorkspace::new();
    PolynomialRootWorkspace::workspace_requirements(
        cubic.len() - 1,
        &PolynomialRootOptions::<f64>::default(),
    )
    .and_then(|requirements| workspace.try_reserve(&requirements))
    .unwrap();
    let stats = scioxide_rootfind::polynomial_roots_into(
        &cubic,
        &PolynomialRootOptions::default(),
        &mut roots,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(
        stats.effective_method(),
        Some(PolynomialRootAlgorithm::Companion)
    );

    let degree_50 = roots_of_unity_f64(50);
    let mut roots = vec![Complex::new(0.0, 0.0); 50];
    PolynomialRootWorkspace::workspace_requirements(
        degree_50.len() - 1,
        &PolynomialRootOptions::<f64>::default(),
    )
    .and_then(|requirements| workspace.try_reserve(&requirements))
    .unwrap();
    let stats = scioxide_rootfind::polynomial_roots_into(
        &degree_50,
        &PolynomialRootOptions::default(),
        &mut roots,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(
        stats.effective_method(),
        Some(PolynomialRootAlgorithm::Companion)
    );
    assert!(!stats.fallback_used());
    assert!(max_residual_f64(&degree_50, &roots) <= 1.0e-9);

    let degree_65 = roots_of_unity_f64(65);
    let mut roots = vec![Complex::new(0.0, 0.0); 65];
    PolynomialRootWorkspace::workspace_requirements(
        degree_65.len() - 1,
        &PolynomialRootOptions::<f64>::default(),
    )
    .and_then(|requirements| workspace.try_reserve(&requirements))
    .unwrap();
    let stats = scioxide_rootfind::polynomial_roots_into(
        &degree_65,
        &PolynomialRootOptions::default(),
        &mut roots,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(
        stats.effective_method(),
        Some(PolynomialRootAlgorithm::Aberth)
    );
    assert!(max_residual_f64(&degree_65, &roots) <= 1.0e-9);
}

#[test]
fn auto_fallback_is_bounded_and_observable() {
    let coefficients = roots_of_unity_f64(3);
    let options = PolynomialRootOptions::default()
        .with_backward_error_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
            Tolerance::new(1.0e-2, 1.0e-2).unwrap(),
        ))
        .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(
            Tolerance::new(1.0e-2, 1.0e-2).unwrap(),
        ))
        .with_max_iterations(Budget::ONE)
        .with_restart_budget(Budget::try_from(4).unwrap())
        .with_polish(false);
    let mut roots = vec![Complex::new(0.0, 0.0); 3];
    let mut workspace = PolynomialRootWorkspace::new();

    PolynomialRootWorkspace::workspace_requirements(
        coefficients.len() - 1,
        &PolynomialRootOptions::<f64>::default(),
    )
    .and_then(|requirements| workspace.try_reserve(&requirements))
    .unwrap();
    let stats = scioxide_rootfind::polynomial_roots_into(
        &coefficients,
        &options,
        &mut roots,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();

    assert_eq!(stats.requested_method(), PolynomialRootMethod::Auto);
    assert_eq!(
        stats.effective_method(),
        Some(PolynomialRootAlgorithm::Aberth)
    );
    assert!(stats.fallback_used());
    assert!(stats.backward_error().unwrap() <= 2.0e-2);
}

#[test]
fn shared_condition_model_and_explicit_cluster_followup() {
    let coefficients = [2.0_f64, -3.0, 1.0];
    let roots = scioxide_rootfind::polynomial_roots_with_options(
        &coefficients,
        &PolynomialRootOptions::default(),
    )
    .unwrap();
    assert_eq!(roots.report().max_root_condition().value(), Some(12.0));
    let diagnostics =
        scioxide_poly::polynomial_root_diagnostics(&coefficients, roots.roots()).unwrap();
    assert_eq!(diagnostics.clusters().len(), 2);
    assert_eq!(
        diagnostics.status(),
        scioxide_poly::RootDiagnosticsStatus::Estimated
    );
    let repeated = scioxide_rootfind::polynomial_roots_with_options(
        &[1.0_f64, -2.0, 1.0],
        &PolynomialRootOptions::default(),
    )
    .unwrap();
    assert_eq!(
        repeated.report().max_root_condition(),
        scioxide_poly::RootEstimate::Singular
    );
}
