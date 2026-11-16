use scioxide_core::{Complex, SciOxideErrorView};
use scioxide_poly::CheckedBasisEvaluation;
use scioxide_rootfind::{
    PolynomialRootMethod as Method, PolynomialRootOptions, PolynomialRootWorkspace,
    polynomial_roots_with_options,
};

#[test]
fn auto_storage_changes_at_64_and_remains_linear_at_large_degrees() {
    let auto = PolynomialRootOptions::<f64>::default();
    let aberth = PolynomialRootOptions::<f64>::default().with_method(Method::Aberth);
    for degree in [63, 64, 65, 10_000] {
        let r = PolynomialRootWorkspace::workspace_requirements(degree, &auto).unwrap();
        assert_eq!(r.evaluation_records(), degree);
        let bytes = r.required_bytes().unwrap();
        if degree <= 64 {
            assert_eq!(r.companion_dimension(), degree);
            assert!(
                bytes
                    > PolynomialRootWorkspace::workspace_requirements(degree, &aberth)
                        .unwrap()
                        .required_bytes()
                        .unwrap()
            );
        } else {
            assert_eq!(r.companion_dimension(), 0);
            assert_eq!(
                bytes,
                PolynomialRootWorkspace::workspace_requirements(degree, &aberth)
                    .unwrap()
                    .required_bytes()
                    .unwrap()
            );
            let expected = r.real_elements() * size_of::<f64>()
                + r.complex_elements() * size_of::<Complex<f64>>()
                + r.index_elements() * size_of::<usize>()
                + r.diagnostic_records()
                    * size_of::<scioxide_rootfind::PolynomialRootDiagnostic<f64>>()
                + degree * size_of::<Option<CheckedBasisEvaluation<f64>>>();
            assert_eq!(bytes, expected);
        }
        let bounded = PolynomialRootOptions::<f64>::default().with_workspace_byte_limit(bytes);
        let workspace = PolynomialRootWorkspace::workspace(degree, &bounded).unwrap();
        assert_eq!(
            workspace
                .check(&PolynomialRootWorkspace::workspace_requirements(degree, &bounded).unwrap())
                .unwrap(),
            scioxide_core::Capacity::Sufficient
        );
        let too_small =
            PolynomialRootOptions::<f64>::default().with_workspace_byte_limit(bytes - 1);
        assert!(matches!(
            PolynomialRootWorkspace::workspace_requirements(degree, &too_small)
                .unwrap_err()
                .view(),
            SciOxideErrorView::ResourceLimit { .. }
        ));
    }
    let mut retained = PolynomialRootWorkspace::workspace(64, &auto).unwrap();
    let linear_bytes = PolynomialRootWorkspace::workspace_requirements(65, &auto)
        .unwrap()
        .required_bytes()
        .unwrap();
    let bounded = PolynomialRootOptions::<f64>::default().with_workspace_byte_limit(linear_bytes);
    let requirements = PolynomialRootWorkspace::workspace_requirements(65, &bounded).unwrap();
    assert!(matches!(retained.try_reserve(&requirements)
        .unwrap_err().view(), SciOxideErrorView::ResourceLimit { required, limit, .. }
            if required > limit && limit == linear_bytes));
}

#[test]
fn auto_uses_reserved_declared_degree_selection_after_zero_deflation() {
    let mut coefficients = vec![0.0_f64; 66];
    coefficients[62] = -1.0;
    coefficients[65] = 1.0;
    let roots =
        polynomial_roots_with_options(&coefficients, &PolynomialRootOptions::default()).unwrap();
    assert!(roots.report().termination().is_success());
    assert_eq!(
        roots.report().effective_method(),
        Some(scioxide_rootfind::PolynomialRootAlgorithm::Aberth)
    );
    assert_eq!(
        roots
            .iter()
            .filter(|r| **r == Complex::new(0.0, 0.0))
            .count(),
        62
    );
    coefficients[62] = 0.0;
    coefficients[63] = -1.0;
    let roots =
        polynomial_roots_with_options(&coefficients, &PolynomialRootOptions::default()).unwrap();
    assert!(roots.report().termination().is_success());
    assert_eq!(
        roots.report().effective_method(),
        Some(scioxide_rootfind::PolynomialRootAlgorithm::Analytic)
    );
}
