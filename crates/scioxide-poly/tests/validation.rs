//! Shared input contracts across owned, borrowed and prepared entry points.
#![allow(clippy::unwrap_used)]
#[path = "support/exact_field.rs"]
mod exact_field;
use exact_field::Q;
use scioxide_core::{
    Complex, ExecutionContext, Interval, InvalidInputReason, SciOxideError, SciOxideErrorKind,
    SciOxideErrorView,
};
use scioxide_poly::*;

fn reason(error: &SciOxideError) -> InvalidInputReason {
    match error.view() {
        SciOxideErrorView::InvalidInput { reason, .. } => reason,
        other => unreachable!("unexpected error: {other:?}"),
    }
}

#[test]
fn coefficient_admission_agrees_across_public_routes() {
    let policy = PolynomialApproximationPolicy::default();
    let elimination = SylvesterElimination::<f64>::new(1, 1, 0, policy).unwrap();
    let discriminant = SylvesterDiscriminant::<f64>::new(1, policy).unwrap();
    let subresultant = SylvesterElimination::<f64>::new(2, 2, 1, policy).unwrap();
    let spec = OrthogonalBasisSpec::new(
        OrthogonalFamily::Legendre,
        BasisNormalization::Classical,
        0.0_f64,
        1.0,
    )
    .unwrap();
    let orthogonal = spec.prepare(1, Default::default()).unwrap();
    for input in [vec![], vec![f64::NAN, 1.0], vec![1.0, f64::INFINITY]] {
        let errors = [
            ApproxPolynomial::monomial_with_options(input.clone(), policy).unwrap_err(),
            ApproxPolynomial::orthogonal_with_plan(input.clone(), orthogonal.clone(), policy)
                .unwrap_err(),
            ApproxPolynomial::bernstein_with_options(input.clone(), 0.0, 1.0, policy).unwrap_err(),
            ApproxPolynomial::newton_with_options(
                input.clone(),
                vec![0.0; input.len().saturating_sub(1)],
                policy,
            )
            .unwrap_err(),
            orthogonal
                .evaluate_with_derivative(&input, 0.0)
                .unwrap_err(),
            polynomial_root_matrix_into(
                &input,
                PolynomialRootBasisRef::Monomial,
                &mut scioxide_linalg::Matrix::zeros(
                    input.len().saturating_sub(1),
                    input.len().saturating_sub(1),
                )
                .unwrap(),
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap_err(),
            approximate_resultant(&input, &[1.0, 1.0]).unwrap_err(),
            approximate_resultant(&[1.0, 1.0], &input).unwrap_err(),
            approximate_discriminant(&input).unwrap_err(),
            approximate_subresultant(&input, &[1.0, 0.0, 1.0], 1).unwrap_err(),
            elimination
                .coefficient_scoped(
                    &input,
                    &[1.0, 1.0],
                    0,
                    &mut elimination.workspace().unwrap(),
                    ExecutionContext::sequential_native(),
                )
                .unwrap_err(),
            discriminant
                .compute_scoped(
                    &input,
                    &mut discriminant.workspace().unwrap(),
                    ExecutionContext::sequential_native(),
                )
                .unwrap_err(),
            subresultant
                .coefficient_scoped(
                    &[1.0, 0.0, 1.0],
                    &input,
                    0,
                    &mut subresultant.workspace().unwrap(),
                    ExecutionContext::sequential_native(),
                )
                .unwrap_err(),
            polynomial_root_radii(&input).unwrap_err(),
            polynomial_root_diagnostics(&input, &[]).unwrap_err(),
            checked_basis_evaluation(
                &input,
                Complex::new(0.0, 0.0),
                PolynomialRootBasisRef::Monomial,
            )
            .unwrap_err(),
        ];
        for error in &errors {
            assert_eq!(error.kind(), errors[0].kind());
            if input.is_empty() {
                assert_eq!(reason(error), InvalidInputReason::Empty);
            }
        }
    }
    assert_eq!(
        reason(&ExactPolynomial::<Q>::new(vec![]).unwrap_err()),
        InvalidInputReason::Empty
    );
    let complex = vec![Complex::new(1.0_f64, f64::NAN), Complex::new(1.0, 0.0)];
    assert_eq!(
        ApproxPolynomial::monomial_with_options(complex.clone(), policy)
            .unwrap_err()
            .kind(),
        approximate_resultant(&complex, &[Complex::new(1.0, 0.0)])
            .unwrap_err()
            .kind()
    );
}

#[test]
fn canonical_zero_tails_and_root_only_restrictions() {
    let policy = PolynomialApproximationPolicy::default();
    let p = ApproxPolynomial::monomial_with_options(vec![1.0_f64, 1.0, -0.0], policy).unwrap();
    assert_eq!(p.basis().coefficients(), &[1.0, 1.0]);
    let zero = ApproxPolynomial::monomial_with_options(vec![-0.0_f64, 0.0], policy).unwrap();
    assert_eq!(zero.basis().coefficients()[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        polynomial_root_radii(&[1.0_f64, 1.0, 0.0])
            .unwrap()
            .degree(),
        1
    );
    assert!(
        (approximate_resultant(&[1.0_f64, 1.0, 0.0], &[2.0, 1.0, 0.0])
            .unwrap()
            .value()
            .round()
            .unwrap()
            - 1.0)
            .abs()
            < 1e-14
    );
    let plan = OrthogonalBasisSpec::new(
        OrthogonalFamily::Legendre,
        BasisNormalization::Classical,
        0.0_f64,
        1.0,
    )
    .unwrap()
    .prepare(2, Default::default())
    .unwrap();
    for basis in [
        PolynomialRootBasisRef::Monomial,
        PolynomialRootBasisRef::Orthogonal(&plan),
    ] {
        let error = polynomial_root_matrix_into(
            &[1.0_f64, 1.0, 0.0],
            basis,
            &mut scioxide_linalg::Matrix::zeros(2, 2).unwrap(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert_eq!(reason(&error), InvalidInputReason::OutOfRange);
    }
    assert_eq!(
        reason(&polynomial_root_radii(&[0.0_f64]).unwrap_err()),
        InvalidInputReason::OutOfRange
    );
    let newton =
        ApproxPolynomial::newton_with_options(vec![1.0_f64, 0.0], vec![0.0], policy).unwrap();
    assert_eq!(newton.basis().coefficients(), &[1.0, 0.0]);
}

#[test]
fn exact_and_approximate_factor_structure_errors_agree() {
    let approximate =
        |c: &[i128]| ApproxPolynomial::monomial(c.iter().map(|&x| x as f64).collect()).unwrap();
    let exact = |c: &[i128]| ExactPolynomial::new(c.iter().map(|&x| Q(x, 1)).collect()).unwrap();
    for (denominator, factor, multiplicity) in [
        (&[0][..], &[0, 1][..], 1),
        (&[1, 2][..], &[0, 1][..], 1),
        (&[0, 1][..], &[0, 2][..], 1),
        (&[0, 1][..], &[1][..], 1),
        (&[0, 1][..], &[0, 1][..], 0),
        (&[0, 1][..], &[0, 1][..], 2),
        (&[0, 1][..], &[0, 1][..], usize::MAX),
    ] {
        let a = ApproxPartialFractionPlan::new(
            &approximate(denominator),
            &[(&approximate(factor), multiplicity)],
        )
        .unwrap_err();
        let b =
            ExactPartialFractionPlan::new(&exact(denominator), &[(&exact(factor), multiplicity)])
                .unwrap_err();
        assert_eq!(a.kind(), b.kind());
        assert_eq!(reason(&a), reason(&b));
    }
    let a = ApproxPartialFractionPlan::new(
        &approximate(&[0, 0, 1]),
        &[(&approximate(&[0, 0, 1]), usize::MAX)],
    )
    .unwrap_err();
    let b = ExactPartialFractionPlan::new(&exact(&[0, 0, 1]), &[(&exact(&[0, 0, 1]), usize::MAX)])
        .unwrap_err();
    assert_eq!(a.kind(), SciOxideErrorKind::SizeOverflow);
    assert_eq!(a.kind(), b.kind());
}

#[test]
fn tensor_owned_and_borrowed_admission_agrees() {
    let unit = Interval::new(0.0_f64, 1.0).unwrap();
    for (domains, components, controls) in [
        (vec![], 1, vec![1.0, 2.0]),
        (vec![unit], 0, vec![]),
        (vec![Interval::new(1.0, 1.0).unwrap()], 1, vec![1.0, 2.0]),
        (vec![unit], 1, vec![1.0]),
        (vec![unit], 1, vec![f64::NAN, 2.0]),
    ] {
        let a =
            TensorBernstein::new(&[1], domains.clone(), components, controls.clone()).unwrap_err();
        let b = TensorBernsteinView::new(&[2], &domains, components, &controls, Default::default())
            .unwrap_err();
        assert_eq!(a.kind(), b.kind());
        if matches!(a.view(), SciOxideErrorView::InvalidInput { .. }) {
            assert_eq!(reason(&a), reason(&b));
        }
    }
    let tensor = TensorBernstein::new(&[1], vec![unit], 1, vec![1.0, 3.0]).unwrap();
    let scalar = ApproxPolynomial::bernstein(vec![1.0_f64, 3.0], 0.0, 1.0).unwrap();
    assert_eq!(
        tensor.control_bounds().unwrap(),
        scalar.control_bounds().unwrap()
    );
    let owned = TensorBernstein::<f64>::new(&[], vec![], usize::MAX, vec![]).unwrap_err();
    let borrowed =
        TensorBernsteinView::<f64>::new(&[], &[], usize::MAX, &[], Default::default()).unwrap_err();
    assert_eq!(owned.kind(), SciOxideErrorKind::SizeOverflow);
    assert_eq!(owned.kind(), borrowed.kind());
    assert_eq!(
        reason(
            &TensorBernsteinView::<f64>::new(&[0], &[unit], 1, &[], Default::default(),)
                .unwrap_err()
        ),
        InvalidInputReason::OutOfRange
    );
}

#[test]
fn borrowed_sparse_plans_share_order_and_zero_term_rules() {
    for (exponents, coefficients) in [([1, 0], [1, 2]), ([0, 0], [1, 2]), ([0, 1], [0, 2])] {
        let approximate = coefficients.map(|v| v as f64);
        let exact = coefficients.map(|v| Q(v, 1));
        let a = SparseEvaluationPlan::new(
            1,
            MonomialOrder::Lexicographic,
            &exponents,
            &approximate,
            Default::default(),
        )
        .unwrap_err();
        let b =
            ExactSparseView::new(1, MonomialOrder::Lexicographic, &exponents, &exact).unwrap_err();
        assert_eq!(a.kind(), b.kind());
        assert_eq!(reason(&a), InvalidInputReason::OutOfRange);
        assert_eq!(reason(&a), reason(&b));
    }
}

#[test]
fn fitting_errors_name_the_input_constraint() {
    assert_eq!(
        reason(&PolynomialFitPlan::<f64>::monomial(&[], 1, None, Default::default()).unwrap_err()),
        InvalidInputReason::Empty
    );
    for weights in [[-1.0_f64, 1.0], [0.0, 0.0]] {
        assert_eq!(
            reason(
                &PolynomialFitPlan::monomial(&[0.0, 1.0], 1, Some(&weights), Default::default())
                    .unwrap_err()
            ),
            InvalidInputReason::OutOfRange
        );
    }
}

/// Every operation over two policy-carrying operands reports a mismatch as the
/// same `InvalidInput(Inconsistent)`, whichever entry point receives it.
#[test]
fn policy_mismatch_has_one_classification_across_entry_points() {
    let default = PolynomialApproximationPolicy::<f64>::default();
    let other = PolynomialApproximationPolicy::new(
        scioxide_core::Tolerance::new(0.25, 0.0).unwrap(),
        scioxide_linalg::RankPolicy::DimensionScaledEpsilon,
    );
    assert_ne!(default, other);
    let monomial = |coefficients: Vec<f64>, policy| {
        ApproxPolynomial::monomial_with_options(coefficients, policy).unwrap()
    };
    let (p, q) = (
        monomial(vec![1.0, 2.0, 1.0], default),
        monomial(vec![1.0, 1.0], other),
    );
    let linear = monomial(vec![1.0, 1.0], default);
    let denominator = monomial(vec![1.0, 2.0, 1.0], default);
    let foreign = monomial(vec![1.0, 1.0], other);
    let errors = [
        p.try_add(&q).unwrap_err(),
        p.try_sub(&q).unwrap_err(),
        p.try_mul(&q).unwrap_err(),
        p.div_rem(&q).unwrap_err(),
        p.compose(&q).unwrap_err(),
        ApproxRationalFunction::new(p.clone(), q.clone()).unwrap_err(),
        approximate_gcd(&p, &q).unwrap_err(),
        ApproxPartialFractionPlan::new(&denominator, &[(&linear, 1), (&foreign, 1)]).unwrap_err(),
    ];
    for error in errors {
        assert_eq!(
            reason(&error),
            InvalidInputReason::Inconsistent,
            "{error:?}"
        );
    }
}

/// Polynomial arithmetic whose finite operands leave the finite range is
/// `NumericalInstability(Overflow)` at every entry point.
#[test]
fn arithmetic_overflow_has_one_classification_across_entry_points() {
    let policy = PolynomialApproximationPolicy::<f64>::default();
    let large = ApproxPolynomial::monomial_with_options(vec![f64::MAX, f64::MAX], policy).unwrap();
    let negative =
        ApproxPolynomial::monomial_with_options(vec![-f64::MAX, -f64::MAX], policy).unwrap();
    let sparse = |coefficient: f64| {
        ApproxSparsePolynomial::from_terms(
            1,
            MonomialOrder::Lexicographic,
            vec![(vec![1_usize], coefficient), (vec![0], coefficient)],
        )
        .unwrap()
    };
    let (positive, opposite) = (sparse(f64::MAX), sparse(-f64::MAX));
    let errors = [
        large.try_add(&large).unwrap_err(),
        large.try_sub(&negative).unwrap_err(),
        large.try_mul(&large).unwrap_err(),
        positive.try_add(&positive).unwrap_err(),
        positive.try_sub(&opposite).unwrap_err(),
        positive.try_mul(&positive).unwrap_err(),
    ];
    for error in errors {
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::Overflow,
                    ..
                }
            ),
            "{error:?}"
        );
    }
}

#[test]
fn empty_nonfinite_underflow_and_shape_errors_have_one_classification() {
    let cx = ExecutionContext::sequential();
    // Empty operands are `Empty` before any prepared-shape lookup.
    let plan = PolynomialOperationPlan::<f64>::new(&[
        PolynomialOperationRequest::Multiply {
            left_len: 2,
            right_len: 2,
        },
        PolynomialOperationRequest::Compose {
            outer_len: 2,
            inner_len: 2,
        },
        PolynomialOperationRequest::TaylorShift { len: 2 },
    ])
    .unwrap();
    let mut workspace = plan.workspace(plan.requests()).unwrap();
    let mut out = [0.0; 3];
    let (empty, pair) = ([0.0_f64; 0], [1.0_f64, 1.0]);
    let mut exact_workspace = ExactPolynomialWorkspace::<Q>::workspace(4).unwrap();
    let mut exact_out = [Q(0, 1); 3];
    let exact_pair = [Q(1, 1), Q(1, 1)];
    for error in [
        plan.multiply_into(&empty, &pair, &mut out, &mut workspace, cx)
            .unwrap_err(),
        plan.multiply_into(&pair, &empty, &mut out, &mut workspace, cx)
            .unwrap_err(),
        plan.compose_into(&empty, &pair, &mut out, &mut workspace, cx)
            .unwrap_err(),
        plan.compose_into(&pair, &empty, &mut out, &mut workspace, cx)
            .unwrap_err(),
        plan.taylor_shift_into(&empty, 1.0, &mut out, &mut workspace, cx)
            .unwrap_err(),
        exact_multiply_into(&[], &exact_pair, &mut exact_out, &mut exact_workspace, cx)
            .unwrap_err(),
        exact_multiply_into(&exact_pair, &[], &mut exact_out, &mut exact_workspace, cx)
            .unwrap_err(),
    ] {
        assert_eq!(reason(&error), InvalidInputReason::Empty, "{error:?}");
    }
    // Output shape disagreement is `DimensionMismatch`.
    for error in [
        plan.multiply_into(&pair, &pair, &mut out[..2], &mut workspace, cx)
            .unwrap_err(),
        plan.compose_into(&pair, &pair, &mut out[..1], &mut workspace, cx)
            .unwrap_err(),
        exact_multiply_into(
            &exact_pair,
            &exact_pair,
            &mut exact_out[..2],
            &mut exact_workspace,
            cx,
        )
        .unwrap_err(),
        exact_compose_into(
            &exact_pair,
            &exact_pair,
            &mut exact_out,
            &mut exact_workspace,
            cx,
        )
        .unwrap_err(),
        exact_taylor_shift_into(
            &exact_pair,
            Q(1, 1),
            &mut exact_out,
            &mut exact_workspace,
            cx,
        )
        .unwrap_err(),
    ] {
        assert_eq!(
            error.kind(),
            SciOxideErrorKind::DimensionMismatch,
            "{error:?}"
        );
    }

    // Nonfinite interval endpoints and subdivision points are `NonFinite`
    // input; finite unordered endpoints remain `OutOfRange`.
    let policy = PolynomialApproximationPolicy::<f64>::default();
    let monomial = ApproxPolynomial::monomial_with_options(vec![1.0_f64, 2.0], policy).unwrap();
    let bernstein =
        ApproxPolynomial::bernstein_with_options(vec![1.0_f64, 2.0], 0.0, 1.0, policy).unwrap();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for error in [
            ApproxPolynomial::bernstein_with_options(vec![1.0_f64], bad, 1.0, policy).unwrap_err(),
            ApproxPolynomial::bernstein_with_options(vec![1.0_f64], 0.0, bad, policy).unwrap_err(),
            monomial.to_bernstein(bad, 1.0).unwrap_err(),
            monomial.to_bernstein(0.0, bad).unwrap_err(),
            bernstein.subdivide(bad).unwrap_err(),
        ] {
            assert_eq!(error.kind(), SciOxideErrorKind::NonFinite, "{error:?}");
        }
    }
    for error in [
        ApproxPolynomial::bernstein_with_options(vec![1.0_f64], 1.0, 0.0, policy).unwrap_err(),
        monomial.to_bernstein(1.0, 1.0).unwrap_err(),
        bernstein.subdivide(1.0).unwrap_err(),
    ] {
        assert_eq!(reason(&error), InvalidInputReason::OutOfRange, "{error:?}");
    }

    // An orthonormal constant that underflows destroys the normalization.
    let underflow = OrthogonalBasisSpec::new(
        OrthogonalFamily::GeneralizedLaguerre { alpha: 320.0 },
        BasisNormalization::Orthonormal,
        0.0_f64,
        1.0,
    )
    .unwrap()
    .prepare(1, Default::default())
    .unwrap_err();
    assert!(matches!(
        underflow.view(),
        SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Underflow,
            ..
        }
    ));

    // Tensor point and output shapes are `DimensionMismatch`.
    let tensor = TensorBernstein::new(
        &[1],
        vec![Interval::new(0.0_f64, 1.0).unwrap()],
        1,
        vec![1.0_f64, 3.0],
    )
    .unwrap();
    let view = tensor.as_view();
    let mut scratch = vec![0.0_f64; view.workspace_requirements(1).unwrap().scratch_len()];
    for (point, len) in [(&[0.5, 0.5][..], 1), (&[0.5][..], 2)] {
        let mut output = vec![0.0; len];
        let error = view
            .evaluate_into(point, &mut output, &mut scratch, cx)
            .unwrap_err();
        assert_eq!(
            error.kind(),
            SciOxideErrorKind::DimensionMismatch,
            "{error:?}"
        );
    }
}

/// Operation reports carry no policy-derived thresholds, so a policy whose
/// tolerance threshold would overflow at the result scale still reports.
#[test]
fn operation_reports_form_no_policy_thresholds() {
    let large = [1e300_f64];
    for report in [
        polynomial_product_condition(&large, &[1.0], &large).unwrap(),
        polynomial_composition_condition(&large, &[1.0], &large).unwrap(),
    ] {
        assert_eq!(report.coefficient_scale(), 1e300);
        assert_eq!(report.condition_estimate(), 1.0);
    }
}

/// Prepared conversion operators that contradict the request are
/// `Inconsistent`, like coefficients beyond a prepared degree.
#[test]
fn contradicting_conversion_operators_are_inconsistent() {
    let spec = |family| {
        OrthogonalBasisSpec::new(family, BasisNormalization::Classical, 0.0_f64, 1.0).unwrap()
    };
    let chebyshev = spec(OrthogonalFamily::ChebyshevT);
    let (cheb2, cheb1) = (
        chebyshev.prepare(2, Default::default()).unwrap(),
        chebyshev.prepare(1, Default::default()).unwrap(),
    );
    let legendre = spec(OrthogonalFamily::Legendre)
        .prepare(2, Default::default())
        .unwrap();
    let expansion = ApproxPolynomial::orthogonal_with_plan(
        vec![1.0_f64, 2.0, 1.0],
        cheb2.clone(),
        Default::default(),
    )
    .unwrap();
    for error in [
        expansion
            .to_monomial_with_operator(&cheb2.conversion_operator(&cheb2).unwrap())
            .unwrap_err(),
        expansion
            .to_orthogonal_with_operator(&legendre.conversion_operator(&legendre).unwrap())
            .unwrap_err(),
        expansion
            .to_orthogonal_with_operator(&cheb2.monomial_operator().unwrap())
            .unwrap_err(),
        cheb2.conversion_operator(&cheb1).unwrap_err(),
    ] {
        assert_eq!(
            reason(&error),
            InvalidInputReason::Inconsistent,
            "{error:?}"
        );
    }
}
