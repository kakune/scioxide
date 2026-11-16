//! Independent analytic and public workflow tests for ordinary root diagnostics.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, NumericScalar, RealScalar};
use scioxide_poly::*;

fn real_cases<S: RealScalar + NumericScalar<Real = S>>() {
    let cast = |v: f64| S::try_from_f64_lossy(v).unwrap();
    let coeff = [cast(2.0), cast(-3.0), cast(1.0)];
    let roots = [
        Complex::new(cast(1.0), cast(0.0)),
        Complex::new(cast(2.0), cast(0.0)),
    ];
    let report = polynomial_root_diagnostics(&coeff, &roots).unwrap();
    assert_eq!(report.status(), RootDiagnosticsStatus::Estimated);
    assert_eq!(report.clusters().len(), 2);
    assert_eq!(report.radii().unwrap().outer().value(), Some(cast(4.0)));
    assert_eq!(
        report.radii().unwrap().inner_nonzero().value(),
        Some(cast(0.4))
    );
    for (index, (root, expected)) in report.roots().iter().zip([6.0, 12.0]).enumerate() {
        assert_eq!(root.condition().absolute().value(), Some(cast(expected)));
        assert_eq!(root.candidate_separation().value(), Some(cast(1.0)));
        let disk = root.disk().unwrap();
        assert_eq!(disk.center(), roots[index]);
        // n=2: n(n+1)*8 eps * bound/|a_n prod(z_i-z_j)|.
        let radius = cast(48.0 * expected) * S::EPSILON;
        assert!((disk.radius().value() - radius).abs() <= cast(8.) * S::EPSILON * radius);
    }
    for scale in [cast(0.000_976_562_5), cast(1024.0), cast(-2.0)] {
        let scaled = coeff.map(|value| value * scale);
        let other = polynomial_root_diagnostics(&scaled, &roots).unwrap();
        assert_eq!(other, report);
    }
    let reversed = polynomial_root_diagnostics(&coeff, &[roots[1], roots[0]]).unwrap();
    assert_eq!(reversed.clusters(), report.clusters());
    assert_eq!(reversed.roots()[0].input_index(), 1);
    let repeated =
        polynomial_root_diagnostics(&[cast(1.0), cast(-2.0), cast(1.0)], &[roots[0], roots[0]])
            .unwrap();
    assert_eq!(
        repeated.status(),
        RootDiagnosticsStatus::Inconclusive(RootEstimateFailure::CoincidentCandidates)
    );
    assert_eq!(repeated.clusters()[0].candidate_count(), 2);
    assert_eq!(
        repeated.roots()[0].condition().absolute(),
        RootEstimate::Singular
    );
    assert!(repeated.roots()[0].disk().is_none());
    // Signed zeros compare equal and therefore coincide; distinct nearby
    // candidates still receive disks (see the repeated-root cluster test).
    let zero = S::ZERO;
    let signed = polynomial_root_diagnostics(
        &[cast(0.0), cast(0.0), cast(1.0)],
        &[Complex::new(zero, zero), Complex::new(-zero, -zero)],
    )
    .unwrap();
    assert_eq!(
        signed.status(),
        RootDiagnosticsStatus::Inconclusive(RootEstimateFailure::CoincidentCandidates)
    );
    let policy = PolynomialApproximationPolicy::default();
    let zero = ApproxPolynomial::monomial_with_options(vec![cast(0.0), cast(1.0)], policy)
        .unwrap()
        .root_condition(cast(0.0))
        .unwrap();
    assert_eq!(zero.relative(), RootEstimate::Unavailable);
    assert_eq!(zero.absolute().value(), Some(cast(0.0)));
}
#[test]
fn both_real_precisions() {
    real_cases::<f32>();
    real_cases::<f64>();
}

#[test]
fn linear_diagnostics_and_unrepresentable_global_fallback_are_explicit() {
    let result = polynomial_root_diagnostics(&[-6.0_f64, 2.], &[Complex::new(3., 0.)]).unwrap();
    assert_eq!(result.status(), RootDiagnosticsStatus::Estimated);
    assert_eq!(result.radii().unwrap().degree(), 1);
    assert_eq!(result.radii().unwrap().outer().value(), Some(4.));
    let diagnostic = &result.roots()[0];
    assert_eq!(diagnostic.condition().absolute().value(), Some(6.));
    assert_eq!(diagnostic.condition().relative().value(), Some(2.));
    assert_eq!(diagnostic.candidate_separation(), RootEstimate::Unavailable);
    let disk = diagnostic.disk().unwrap();
    assert_eq!(disk.center(), Complex::new(3., 0.));
    // n=1, bound=12, denominator=2: rounding floor = 16 eps * 6.
    assert!((disk.radius().value() - 96. * f64::EPSILON).abs() <= f64::EPSILON * f64::EPSILON);
    let fallback =
        polynomial_root_diagnostics(&[f64::MAX, f64::from_bits(1)], &[Complex::new(0., 0.)])
            .unwrap();
    assert_eq!(
        fallback.status(),
        RootDiagnosticsStatus::Inconclusive(RootEstimateFailure::ScalarRange)
    );
    assert_eq!(
        fallback.radii().unwrap().outer(),
        RootEstimate::Inconclusive(RootEstimateFailure::ScalarRange)
    );
    assert_eq!(fallback.clusters().len(), 1);
    assert_eq!(fallback.clusters()[0].candidate_count(), 1);
    assert!(fallback.clusters()[0].disk().is_none());
    assert!(fallback.roots()[0].disk().is_none());
}

fn complex_cases<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let c = |re: f64, im: f64| {
        Complex::new(
            S::try_from_f64_lossy(re).unwrap(),
            S::try_from_f64_lossy(im).unwrap(),
        )
    };
    // (z-i)(z-2) = z² -(2+i)z +2i.
    let coeff = [c(0.0, 2.0), c(-2.0, -1.0), c(1.0, 0.0)];
    let roots = [c(0.0, 1.0), c(2.0, 0.0)];
    let result = polynomial_root_diagnostics(&coeff, &roots).unwrap();
    let rotated = polynomial_root_diagnostics(&coeff.map(|v| v * c(0.0, 2.0)), &roots).unwrap();
    assert_eq!(result, rotated);
    assert_eq!(result.clusters().len(), 2);
    for (root, actual) in result.roots().iter().zip(roots) {
        let disk = root.disk().unwrap();
        assert!((disk.center() - actual).norm() <= disk.radius().value());
    }
}
#[test]
fn both_complex_precisions() {
    complex_cases::<f32>();
    complex_cases::<f64>();
}

#[test]
fn range_and_domain_are_explicit() {
    assert!(matches!(
        (polynomial_root_radii::<f64>(&[])).unwrap_err().view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    assert!(matches!(
        (polynomial_root_radii(&[0.0_f64, -0.0]))
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
    assert!(matches!(
        (polynomial_root_radii(&[1.0, f64::NAN]))
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    let constant = polynomial_root_diagnostics(&[3.0_f64, 0.0], &[]).unwrap();
    assert!(constant.clusters().is_empty());
    assert_eq!(constant.radii().unwrap().outer(), RootEstimate::Unavailable);
    let power = polynomial_root_radii(&[0.0_f64, 0.0, 2.0, 0.0]).unwrap();
    assert_eq!(power.zero_roots(), 2);
    assert_eq!(power.outer().value(), Some(0.0));
    assert_eq!(power.inner_nonzero(), RootEstimate::Unavailable);
    let extreme = polynomial_root_radii(&[f64::MAX, f64::from_bits(1)]).unwrap();
    assert!(matches!(
        extreme.outer(),
        RootEstimate::Inconclusive(RootEstimateFailure::ScalarRange)
    ));
    assert!(matches!(
        polynomial_root_radii(&[f64::from_bits(1), f64::MAX])
            .unwrap()
            .inner_nonzero(),
        RootEstimate::Inconclusive(RootEstimateFailure::ScalarRange)
    ));
    let tiny = f64::from_bits(1);
    let condition = checked_basis_evaluation(
        &[tiny, -tiny],
        Complex::new(1.0, 0.0),
        scioxide_poly::PolynomialRootBasisRef::Monomial,
    )
    .unwrap()
    .root_condition()
    .unwrap();
    assert_eq!(condition.absolute().value(), Some(2.0));
    let huge = checked_basis_evaluation(
        &[f64::MAX, -f64::MAX],
        Complex::new(1.0, 0.0),
        scioxide_poly::PolynomialRootBasisRef::Monomial,
    )
    .unwrap()
    .root_condition()
    .unwrap();
    assert_eq!(huge.absolute().value(), Some(2.0));
    assert!(matches!(
        (polynomial_root_diagnostics(&[1.0_f64, 1.0], &[]))
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert!(matches!(
        (polynomial_root_diagnostics(&[1.0_f64, 1.0], &[Complex::new(f64::INFINITY, 0.0)]))
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
}

#[test]
fn repeated_root_neighborhood_forms_one_cluster() {
    let roots = [
        Complex::new(-2.0_f64, 0.0),
        Complex::new(1.0 - 1e-7, 0.0),
        Complex::new(1.0 + 1e-7, 0.0),
    ];
    let result = polynomial_root_diagnostics(&[2.0_f64, -3.0, 0.0, 1.0], &roots).unwrap();
    assert_eq!(result.status(), RootDiagnosticsStatus::Estimated);
    assert_eq!(result.clusters().len(), 2);
    assert_eq!(result.clusters()[1].candidate_count(), 2);
    let disk = result.clusters()[1].disk().unwrap();
    assert!((disk.center().re - 1.0).abs() <= disk.radius().value());
    assert!(result.clusters()[1].separation().value().unwrap() > 2.9);
}

#[test]
fn native_basis_conditioning_tracks_coefficient_model() {
    let policy = PolynomialApproximationPolicy::default();
    let monomial =
        ApproxPolynomial::monomial_with_options(vec![2.0_f64, -3.0, 1.0], policy).unwrap();
    assert_eq!(
        monomial.root_condition(2.0).unwrap().absolute().value(),
        Some(12.0)
    );
    // Newton coefficients [0,1] at node 2 represent z-2 with a fixed zero constant:
    // coefficientwise perturbations leave the root exactly at its node.
    let newton =
        ApproxPolynomial::newton_with_options(vec![0.0_f64, 1.0], vec![2.0], policy).unwrap();
    assert_eq!(
        newton.root_condition(2.0).unwrap().absolute().value(),
        Some(0.0)
    );
    let converted = newton.to_monomial().unwrap();
    assert_eq!(
        converted
            .polynomial()
            .root_condition(2.0)
            .unwrap()
            .absolute()
            .value(),
        Some(4.0)
    );
    let bernstein =
        ApproxPolynomial::bernstein_with_options(vec![1.0_f64, -1.0], 0.0, 1.0, policy).unwrap();
    assert_eq!(
        bernstein.root_condition(2.0).unwrap().absolute().value(),
        Some(1.5)
    );
    let complex = ApproxPolynomial::bernstein_with_options(
        vec![Complex::new(1.0_f64, 0.0), Complex::new(-1.0, 0.0)],
        0.0,
        1.0,
        policy,
    )
    .unwrap();
    let sensitivity = complex
        .root_condition(Complex::new(0.5, 1.0))
        .unwrap()
        .absolute()
        .value()
        .unwrap();
    assert!((sensitivity - 1.25_f64.sqrt()).abs() < 1e-14);
    // Independent first-order coefficient perturbation for the simple root at 2.
    let delta = 1e-7;
    let changed_constant = 2.0 + delta;
    let perturbed_root = (3.0 + (9.0_f64 - 4.0 * changed_constant).sqrt()) / 2.0;
    assert!(((perturbed_root - 2.0) / delta + 1.0).abs() < 1e-6);
}

#[test]
fn prepared_results_are_transactional_and_reusable() {
    use scioxide_core::{CancellationToken, ExecutionContext};
    let coeff = [2.0_f64, -3.0, 1.0];
    let roots = [Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
    let expected = polynomial_root_diagnostics(&coeff, &roots).unwrap();
    let mut output = expected.clone();
    let mut workspace = RootDiagnosticsWorkspace::new();
    let cx = ExecutionContext::sequential();
    let requirements = RootDiagnosticsWorkspace::<f64>::workspace_requirements(2).unwrap();
    assert_eq!(requirements.degree(), 2);
    assert!(requirements.required_bytes().unwrap() > 0);
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        scioxide_core::Capacity::NeedsGrowth
    );
    let error = polynomial_root_diagnostics_into(&coeff, &roots, &mut output, &mut workspace, cx)
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(output, expected);
    // A reservation for a larger degree also serves this one.
    workspace
        .try_reserve(&RootDiagnosticsWorkspace::workspace_requirements(5).unwrap())
        .unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        scioxide_core::Capacity::Sufficient
    );
    // Undersized output capacity is a dimension mismatch and leaves output unchanged.
    let mut unreserved = RootDiagnostics::new();
    let error =
        polynomial_root_diagnostics_into(&coeff, &roots, &mut unreserved, &mut workspace, cx)
            .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(unreserved, RootDiagnostics::new());
    for _ in 0..2 {
        polynomial_root_diagnostics_into(&coeff, &roots, &mut output, &mut workspace, cx).unwrap();
        assert_eq!(output, expected);
    }
    let token = CancellationToken::new();
    token.cancel();
    let error = polynomial_root_diagnostics_into(
        &coeff,
        &roots,
        &mut output,
        &mut workspace,
        cx.with_cancellation(&token),
    )
    .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
    ));
    assert_eq!(output, expected);
    assert!(matches!(
        RootDiagnosticsWorkspace::<f64>::workspace_requirements(usize::MAX)
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::SizeOverflow { .. }
    ));
}

#[test]
fn hundred_digit_analytic_oracles_are_enclosed() {
    use std::collections::BTreeMap;
    let mut cases = BTreeMap::<String, (Vec<Complex<f64>>, Vec<Complex<f64>>)>::new();
    for line in include_str!("data/root_diagnostics.csv").lines().skip(1) {
        let fields: Vec<_> = line.split(';').collect();
        let coefficients = fields[1]
            .split('|')
            .map(|pair| {
                let (re, im) = pair.split_once(',').unwrap();
                Complex::new(re.parse::<f64>().unwrap(), im.parse::<f64>().unwrap())
            })
            .collect();
        cases
            .entry(fields[0].to_owned())
            .or_insert_with(|| (coefficients, Vec::new()))
            .1
            .push(Complex::new(
                fields[2].parse().unwrap(),
                fields[3].parse().unwrap(),
            ));
    }
    for (name, (coefficients, oracle)) in cases {
        let candidates: Vec<_> = oracle
            .iter()
            .map(|root| *root + Complex::new(1e-7, -2e-7))
            .collect();
        let result = polynomial_root_diagnostics(&coefficients, &candidates).unwrap();
        assert_eq!(result.status(), RootDiagnosticsStatus::Estimated, "{name}");
        for root in &oracle {
            // Account conservatively for rounding each 100-digit oracle to f64.
            let oracle_radius = 2.0 * f64::EPSILON * root.norm();
            assert!(
                result.clusters().iter().any(|cluster| {
                    let disk = cluster.disk().unwrap();
                    (disk.center() - *root).norm() + oracle_radius <= disk.radius().value()
                }),
                "{name}: {root:?}"
            );
            assert!(
                root.norm() + oracle_radius <= result.radii().unwrap().outer().value().unwrap()
            );
            assert!(
                root.norm() - oracle_radius
                    >= result.radii().unwrap().inner_nonzero().value().unwrap()
            );
        }
        let coefficients32: Vec<_> = coefficients
            .iter()
            .map(|z| Complex::new(z.re as f32, z.im as f32))
            .collect();
        let candidates32: Vec<_> = oracle
            .iter()
            .map(|z| Complex::new(z.re as f32 + 1e-5, z.im as f32 - 2e-5))
            .collect();
        let result32 = polynomial_root_diagnostics(&coefficients32, &candidates32).unwrap();
        assert_eq!(
            result32.status(),
            RootDiagnosticsStatus::Estimated,
            "{name}"
        );
        for root in &oracle {
            assert!(
                result32.clusters().iter().any(|cluster| {
                    let disk = cluster.disk().unwrap();
                    let center =
                        Complex::new(f64::from(disk.center().re), f64::from(disk.center().im));
                    (center - *root).norm() + 2.0 * f64::EPSILON * root.norm()
                        <= f64::from(disk.radius().value())
                }),
                "f32 {name}: {root:?}"
            );
        }
    }
}

#[cfg(feature = "std")]
#[test]
fn injected_executor_has_the_same_canonical_result() {
    use scioxide_core::{ExecutionContext, ThreadExecutor};
    let coefficients = [2.0_f64, -3.0, 1.0];
    let roots = [Complex::new(2.0, 0.0), Complex::new(1.0, 0.0)];
    let expected = polynomial_root_diagnostics(&coefficients, &roots).unwrap();
    let requirements = RootDiagnosticsWorkspace::workspace_requirements(2).unwrap();
    let mut output = RootDiagnostics::new();
    let mut workspace = RootDiagnosticsWorkspace::workspace(2).unwrap();
    output.try_reserve(&requirements).unwrap();
    let executor = ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    polynomial_root_diagnostics_into(
        &coefficients,
        &roots,
        &mut output,
        &mut workspace,
        ExecutionContext::new(&executor),
    )
    .unwrap();
    assert_eq!(output, expected);
}

#[test]
fn prepared_families_expose_native_sensitivity_without_conversion() {
    let policy = PolynomialApproximationPolicy::default();
    let monomial =
        ApproxPolynomial::monomial_with_options(vec![2.0_f64, 3.0, 1.0], policy).unwrap();
    let chebyshev = monomial
        .to_orthogonal(
            scioxide_poly::OrthogonalBasisSpec::new(
                scioxide_poly::OrthogonalFamily::ChebyshevT,
                scioxide_poly::BasisNormalization::Classical,
                0.0,
                1.0,
            )
            .unwrap(),
        )
        .unwrap();
    let legendre = monomial
        .to_orthogonal(
            scioxide_poly::OrthogonalBasisSpec::new(
                scioxide_poly::OrthogonalFamily::Legendre,
                scioxide_poly::BasisNormalization::Classical,
                0.0,
                1.0,
            )
            .unwrap(),
        )
        .unwrap();
    let barycentric = monomial.to_barycentric(vec![-1.0, 0.0, 1.0]).unwrap();
    for condition in [
        chebyshev.polynomial().root_condition(0.25).unwrap(),
        legendre.polynomial().root_condition(0.25).unwrap(),
        barycentric.polynomial().root_condition(0.25).unwrap(),
    ] {
        assert!(condition.absolute().value().unwrap() > 0.0);
        assert!(condition.coefficient().value().unwrap() >= 1.0);
    }
}

#[test]
fn canonical_pair_products_are_permutation_invariant() {
    let coefficients = [-2.0_f64, 0.0, 0.0, 1.0];
    let roots = [
        Complex::new(1.26, 0.0001),
        Complex::new(-0.63, 1.091),
        Complex::new(-0.63, -1.091),
    ];
    let original = polynomial_root_diagnostics(&coefficients, &roots).unwrap();
    let permuted =
        polynomial_root_diagnostics(&coefficients, &[roots[2], roots[0], roots[1]]).unwrap();
    assert_eq!(original.clusters(), permuted.clusters());
    for (left, right) in original.roots().iter().zip(permuted.roots()) {
        assert_eq!(left.disk(), right.disk());
        assert_eq!(left.condition(), right.condition());
        assert_eq!(left.candidate_separation(), right.candidate_separation());
    }
}
