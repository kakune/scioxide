//! Every one-shot form equals its configured form with default settings, to the
//! last bit, and the configured forms keep their settings.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Argument, AttemptOutcome, Complex, ExecutionContext, Interval, NumericScalar, NumericalIssue,
    ProblemToken, RankThreshold, RealScalar, SciOxideErrorKind, SciOxideErrorView, SolveBudgets,
    SolveIdentity, SolveScope, SolveState, Tolerance, algorithm_id,
};
use scioxide_linalg::{HermitianEigenvaluesResult, RankPolicy};
use scioxide_poly::*;

fn scalar<C: NumericScalar>(re: f64, im: f64) -> C {
    let r = |x| C::Real::try_from_f64_lossy(x).unwrap();
    C::from_components(r(re), if C::IS_REAL { r(0.) } else { r(im) })
}

fn constructors<C: NumericScalar>() {
    let c = |re, im| scalar::<C>(re, im);
    let r = |x| C::Real::try_from_f64_lossy(x).unwrap();
    let policy = PolynomialApproximationPolicy::default;
    let coefficients = vec![c(1., 0.5), c(-2., 0.), c(3., -1.)];
    let nodes = vec![c(0., 0.), c(1., 0.), c(2.5, 0.)];
    assert_eq!(
        ApproxPolynomial::monomial(coefficients.clone()).unwrap(),
        ApproxPolynomial::monomial_with_options(coefficients.clone(), policy()).unwrap()
    );
    assert_eq!(
        ApproxPolynomial::from_roots(&nodes).unwrap(),
        ApproxPolynomial::from_roots_with_options(&nodes, policy()).unwrap()
    );
    assert_eq!(
        ApproxPolynomial::newton(coefficients.clone(), nodes[..2].to_vec()).unwrap(),
        ApproxPolynomial::newton_with_options(coefficients.clone(), nodes[..2].to_vec(), policy())
            .unwrap()
    );
    assert_eq!(
        ApproxPolynomial::interpolate_barycentric(nodes.clone(), coefficients.clone()).unwrap(),
        ApproxPolynomial::interpolate_barycentric_with_options(
            nodes,
            coefficients.clone(),
            policy()
        )
        .unwrap()
    );
    assert_eq!(
        ApproxPolynomial::bernstein(coefficients.clone(), r(-1.), r(2.)).unwrap(),
        ApproxPolynomial::bernstein_with_options(coefficients.clone(), r(-1.), r(2.), policy())
            .unwrap()
    );
    let spec =
        OrthogonalBasisSpec::on_interval(OrthogonalFamily::ChebyshevU, r(-1.), r(2.)).unwrap();
    assert_eq!(
        ApproxPolynomial::orthogonal(coefficients.clone(), spec).unwrap(),
        ApproxPolynomial::orthogonal_with_plan(
            coefficients.clone(),
            spec.prepare(coefficients.len() - 1, Default::default())
                .unwrap(),
            policy(),
        )
        .unwrap()
    );
    assert_eq!(
        ApproxPolynomial::<C, OrthogonalBasis<C>>::orthogonal(Vec::new(), spec)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
}

fn real_constructors<R: RealScalar + NumericScalar<Real = R>>() {
    let r = |x| R::try_from_f64_lossy(x).unwrap();
    let policy = PolynomialApproximationPolicy::default;
    let (nodes, values) = (vec![r(0.), r(1.), r(3.)], vec![r(1.), r(-1.), r(4.)]);
    assert_eq!(
        ApproxPolynomial::interpolate_newton(nodes.clone(), values.clone()).unwrap(),
        ApproxPolynomial::interpolate_newton_with_options(nodes, values, policy()).unwrap()
    );
    let pairs = [Complex::new(r(0.5), r(2.))];
    assert_eq!(
        ApproxPolynomial::from_conjugate_pairs(&[r(1.)], &pairs).unwrap(),
        ApproxPolynomial::from_conjugate_pairs_with_options(&[r(1.)], &pairs, policy()).unwrap()
    );
}

#[test]
fn plain_constructors_equal_their_configured_forms_with_the_default_policy() {
    constructors::<f32>();
    constructors::<f64>();
    constructors::<Complex<f32>>();
    constructors::<Complex<f64>>();
    real_constructors::<f32>();
    real_constructors::<f64>();
}

#[test]
fn policy_setters_replace_one_field_and_configured_forms_keep_the_policy() {
    let default = PolynomialApproximationPolicy::<f64>::default();
    let tolerance = Tolerance::new(1e-10, 1e-6).unwrap();
    let rank = RankPolicy::Threshold(RankThreshold::new(1e-3, 0.0).unwrap());
    let tightened = default.with_tolerance(tolerance);
    assert_eq!(tightened.tolerance(), tolerance);
    assert_eq!(tightened.rank_policy(), default.rank_policy());
    let ranked = default.with_rank_policy(rank);
    assert_eq!(ranked.rank_policy(), rank);
    assert_eq!(ranked.tolerance(), default.tolerance());
    assert_eq!(
        default.with_tolerance(tolerance).with_rank_policy(rank),
        PolynomialApproximationPolicy::new(tolerance, rank)
    );

    let plain = ApproxPolynomial::monomial(vec![1.0, 2.0]).unwrap();
    let configured = ApproxPolynomial::monomial_with_options(vec![1.0, 2.0], ranked).unwrap();
    assert_eq!(plain.policy(), default);
    assert_eq!(configured.policy(), ranked);
    assert_eq!(configured.derivative().unwrap().policy(), ranked);
    for error in [
        plain.try_add(&configured).unwrap_err(),
        plain.try_mul(&configured).unwrap_err(),
        plain.div_rem(&configured).unwrap_err(),
    ] {
        assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
    }

    // A relative tolerance of MAX overflows its threshold at coefficient scale 2.
    let overflowing = default.with_tolerance(Tolerance::new(0.0, f64::MAX).unwrap());
    let error = ApproxPolynomial::monomial_with_options(vec![1.0, 2.0], overflowing).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::Overflow,
            ..
        }
    ));
}

#[test]
fn spec_constructors_map_intervals_and_reject_unbounded_families() {
    let family = OrthogonalFamily::<f64>::ChebyshevT;
    let spec = OrthogonalBasisSpec::on_interval(family, 1.0, 5.0).unwrap();
    assert_eq!(
        spec,
        OrthogonalBasisSpec::new(family, BasisNormalization::Classical, 3.0, 2.0).unwrap()
    );
    let wide = OrthogonalBasisSpec::on_interval(family, -f64::MAX, f64::MAX).unwrap();
    assert_eq!((wide.shift(), wide.scale()), (0.0, f64::MAX));
    let tiny = f64::from_bits(1);
    let two = OrthogonalBasisSpec::on_interval(family, 0.0, 2.0 * tiny).unwrap();
    assert_eq!((two.shift(), two.scale()), (tiny, tiny));
    assert_eq!(
        OrthogonalBasisSpec::classical(OrthogonalFamily::<f64>::HermiteHe).unwrap(),
        OrthogonalBasisSpec::new(
            OrthogonalFamily::HermiteHe,
            BasisNormalization::Classical,
            0.0,
            1.0
        )
        .unwrap()
    );
    let orthonormal = spec.with_normalization(BasisNormalization::Orthonormal);
    assert_eq!(orthonormal.normalization(), BasisNormalization::Orthonormal);
    assert_eq!((orthonormal.shift(), orthonormal.scale()), (3.0, 2.0));

    let invalid = |lower, upper, family| -> Option<(Argument, _)> {
        let error = OrthogonalBasisSpec::on_interval(family, lower, upper).unwrap_err();
        match error.view() {
            SciOxideErrorView::InvalidInput {
                argument, reason, ..
            } => Some((argument, reason)),
            _ => {
                assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
                None
            }
        }
    };
    use scioxide_core::InvalidInputReason::{Inconsistent, OutOfRange};
    let upper = |reason| Some((scioxide_core::argument!("upper"), reason));
    assert_eq!(invalid(1.0, 1.0, family), upper(OutOfRange));
    assert_eq!(invalid(2.0, 1.0, family), upper(OutOfRange));
    assert_eq!(invalid(0.0, tiny, family), upper(OutOfRange));
    assert_eq!(invalid(f64::NAN, 1.0, family), None);
    assert_eq!(invalid(0.0, f64::INFINITY, family), None);
    for unbounded in [
        OrthogonalFamily::Laguerre,
        OrthogonalFamily::GeneralizedLaguerre { alpha: 0.5 },
        OrthogonalFamily::HermiteH,
        OrthogonalFamily::HermiteHe,
    ] {
        assert_eq!(
            invalid(0.0, 1.0, unbounded),
            Some((scioxide_core::argument!("family"), Inconsistent))
        );
    }
    let error = OrthogonalBasisSpec::on_interval(
        OrthogonalFamily::Jacobi {
            alpha: -2.0,
            beta: 0.0,
        },
        0.0,
        1.0,
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DomainError);
}

fn fits<C: NumericScalar>() {
    let c = |re, im| scalar::<C>(re, im);
    let r = |x| C::Real::try_from_f64_lossy(x).unwrap();
    let points: Vec<C> = [-1.0, -0.5, 0.0, 0.25, 1.0]
        .iter()
        .map(|&x| c(x, 0.))
        .collect();
    let values: Vec<C> = [1.0, -0.5, 2.0, 0.0, 3.0]
        .iter()
        .map(|&y| c(y, 0.5 * y))
        .collect();
    let policy = PolynomialApproximationPolicy::default();
    // Degree 2 is overdetermined; degree 6 has more coefficients than points.
    for degree in [2, 6] {
        let plan = PolynomialFitPlan::monomial(&points, degree, None, policy).unwrap();
        assert_eq!(
            ApproxPolynomial::fit_monomial(&points, &values, degree).unwrap(),
            *plan.fit(&values).unwrap().polynomial()
        );
        let spec =
            OrthogonalBasisSpec::on_interval(OrthogonalFamily::Legendre, r(-1.), r(1.)).unwrap();
        let basis = spec.prepare(degree, Default::default()).unwrap();
        let plan = PolynomialFitPlan::orthogonal(&points, basis, None, policy).unwrap();
        assert_eq!(
            ApproxPolynomial::fit_orthogonal(&points, &values, degree, spec).unwrap(),
            *plan.fit(&values).unwrap().polynomial()
        );
    }
    // Repeated points make the design rank deficient.
    let repeated = vec![c(0.5, 0.); 4];
    let plan = PolynomialFitPlan::monomial(&repeated, 2, None, policy).unwrap();
    let fit = plan.fit(&values[..4]).unwrap();
    assert_eq!(fit.report().rank().rank(), 1);
    assert_eq!(
        ApproxPolynomial::fit_monomial(&repeated, &values[..4], 2).unwrap(),
        fit.into_polynomial()
    );
    assert_eq!(
        ApproxPolynomial::fit_monomial(&points, &values[..2], 1)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        ApproxPolynomial::<C, MonomialBasis<C>>::fit_monomial(&[], &[], 1)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
}

#[test]
fn fit_one_shots_equal_the_plan_fits_including_rank_deficient_designs() {
    fits::<f32>();
    fits::<f64>();
    fits::<Complex<f32>>();
    fits::<Complex<f64>>();
}

#[test]
fn vandermonde_and_tensor_one_shots_equal_their_builders() {
    let points = [-1.5_f64, 0.0, 0.5, 2.0];
    assert_eq!(
        vandermonde(&points, 3).unwrap(),
        Vandermonde::<f64>::monomial(3)
            .unwrap()
            .matrix(&points)
            .unwrap()
    );
    let complex = [Complex::new(0.5_f64, 1.0)];
    assert_eq!(
        vandermonde(&complex, 2).unwrap(),
        Vandermonde::<f64>::monomial(2)
            .unwrap()
            .matrix(&complex)
            .unwrap()
    );
    assert_eq!(
        vandermonde(&[f64::NAN], 1).unwrap_err().kind(),
        SciOxideErrorKind::NonFinite
    );

    let domains = vec![
        Interval::new(0.0_f64, 1.0).unwrap(),
        Interval::new(-1.0, 1.0).unwrap(),
    ];
    let controls = vec![0.0, 1.0, 2.0, 1.0, 3.0, -1.0, 0.5, 2.0, 1.5, -0.5, 4.0, 0.0];
    let tensor = TensorBernstein::new(&[1, 2], domains.clone(), 2, controls.clone()).unwrap();
    let configured = TensorBernstein::with_options(
        &[1, 2],
        domains,
        2,
        controls,
        PolynomialExecutionOptions::new(),
    )
    .unwrap();
    assert_eq!(tensor, configured);
    let view = tensor.as_view();
    let mut scratch = vec![0.0; view.workspace_requirements(1).unwrap().scratch_len()];
    for point in [[0.25, 0.5], [1.5, -2.0]] {
        let mut expected = [0.0; 2];
        view.evaluate_into(
            &point,
            &mut expected,
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(tensor.evaluate(&point).unwrap(), expected);
    }
    assert_eq!(
        tensor.evaluate(&[0.5]).unwrap_err().kind(),
        SciOxideErrorKind::DimensionMismatch
    );
}

#[test]
fn sparse_one_shots_equal_their_plan_and_configured_forms() {
    let order = MonomialOrder::GradedLexicographic;
    let terms = [([0, 0], 1.0_f64), ([1, 0], 2.0), ([2, 3], -0.5)];
    let p = ApproxSparsePolynomial::from_terms(2, order, terms).unwrap();
    let plan = p
        .evaluation_plan(PolynomialExecutionOptions::new())
        .unwrap();
    let mut scratch = vec![0.0; plan.workspace_requirements(1).unwrap().scratch_len()];
    for point in [[0.5, 2.0], [-3.0, 1e-3]] {
        assert_eq!(
            p.evaluate(&point).unwrap(),
            plan.evaluate(&point, &mut scratch).unwrap()
        );
    }
    let replacements = [p.clone(), p.clone()];
    assert_eq!(
        p.substitute(&replacements, 2).unwrap(),
        p.substitute_with_options(&replacements, 2, SparseExpansionLimits::default())
            .unwrap()
    );
    assert_eq!(SparseExpansionLimits::default().max_terms(), usize::MAX);
    assert_eq!(
        p.substitute_with_options(&replacements, 2, SparseExpansionLimits::new(1))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
}

fn elimination<C: NumericScalar>() {
    let c = |re, im| scalar::<C>(re, im);
    let policy = PolynomialApproximationPolicy::default();
    let left = [c(1., 0.5), c(-2., 0.), c(0.5, 0.), c(1., 0.)];
    let right = [c(-1., 0.), c(0., 1.), c(2., 0.)];
    let resultant = SylvesterElimination::new(3, 2, 0, policy).unwrap();
    let one_shot = approximate_resultant(&left, &right).unwrap();
    assert_eq!(resultant.coefficient(&left, &right, 0).unwrap(), one_shot);
    let mut workspace = resultant.workspace().unwrap();
    let scoped = resultant
        .coefficient_scoped(
            &left,
            &right,
            0,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(scoped, one_shot);

    let subresultant = SylvesterElimination::new(3, 2, 1, policy).unwrap();
    let coefficients = approximate_subresultant(&left, &right, 1).unwrap();
    for (k, expected) in coefficients.iter().enumerate() {
        assert_eq!(
            subresultant.coefficient(&left, &right, k).unwrap(),
            *expected
        );
    }

    let discriminant = SylvesterDiscriminant::new(3, policy).unwrap();
    let one_shot = approximate_discriminant(&left).unwrap();
    assert_eq!(discriminant.compute(&left).unwrap(), one_shot);
    let mut workspace = discriminant.workspace().unwrap();
    let scoped = discriminant
        .compute_scoped(&left, &mut workspace, ExecutionContext::sequential_native())
        .unwrap();
    assert_eq!(scoped, one_shot);
}

#[test]
fn elimination_one_shots_equal_the_method_values_with_the_default_policy() {
    elimination::<f32>();
    elimination::<f64>();
    elimination::<Complex<f32>>();
    elimination::<Complex<f64>>();
}

#[test]
fn scoped_elimination_runs_as_one_child_epoch_and_records_the_attempt() {
    let policy = PolynomialApproximationPolicy::default();
    let (left, right) = ([1.0_f64, 2.0, 3.0, 1.0], [-2.0, 1.0, 1.0]);
    let resultant = SylvesterElimination::new(3, 2, 0, policy).unwrap();
    let discriminant = SylvesterDiscriminant::new(3, policy).unwrap();
    let mut elimination_workspace = resultant.workspace().unwrap();
    let mut discriminant_workspace = discriminant.workspace().unwrap();
    assert!(elimination_workspace.last_attempt().is_none());
    let expected = (
        resultant.coefficient(&left, &right, 0).unwrap(),
        discriminant.compute(&left).unwrap(),
    );
    let identity = SolveIdentity::new(
        ProblemToken::new(1),
        algorithm_id!("scioxide-poly-tests", "enclosing", 1),
        0,
        0,
    );
    let mut state = SolveState::new();
    let nested = SolveScope::from(ExecutionContext::sequential_native())
        .run(&mut state, identity, SolveBudgets::unbounded(), |cx| {
            let value = resultant.coefficient_scoped(
                &left,
                &right,
                0,
                &mut elimination_workspace,
                &mut *cx,
            )?;
            let first =
                discriminant.compute_scoped(&left, &mut discriminant_workspace, &mut *cx)?;
            let second =
                discriminant.compute_scoped(&left, &mut discriminant_workspace, &mut *cx)?;
            assert_eq!(first, second);
            Ok((value, first))
        })
        .unwrap();
    assert_eq!(nested, expected);
    let attempt = elimination_workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
    assert_eq!(attempt.epoch().map(|epoch| epoch.get()), Some(1));
    // Each discriminant, including its resultant, is one epoch of its own state.
    let attempt = discriminant_workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
    assert_eq!(attempt.epoch().map(|epoch| epoch.get()), Some(2));

    // Input rejected before the epoch is a preflight rejection; a failure inside
    // the epoch is a failed attempt.
    let cx = ExecutionContext::sequential();
    assert!(
        resultant
            .coefficient_scoped(&[], &right, 0, &mut elimination_workspace, cx)
            .is_err()
    );
    let attempt = elimination_workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
    assert!(attempt.epoch().is_none());
    assert!(
        resultant
            .coefficient_scoped(&left[..2], &right, 0, &mut elimination_workspace, cx)
            .is_err()
    );
    let attempt = elimination_workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Failed);
    assert_eq!(attempt.epoch().map(|epoch| epoch.get()), Some(2));
}

#[test]
fn family_zeros_use_the_default_tridiagonal_eigensolver() {
    for family in [
        OrthogonalFamily::<f64>::Legendre,
        OrthogonalFamily::Jacobi {
            alpha: 0.5,
            beta: -0.25,
        },
        OrthogonalFamily::Gegenbauer { lambda: 1.5 },
        OrthogonalFamily::Laguerre,
        OrthogonalFamily::GeneralizedLaguerre { alpha: 2.0 },
        OrthogonalFamily::HermiteH,
        OrthogonalFamily::HermiteHe,
    ] {
        for n in [1, 2, 7, 16] {
            let recurrence = family.jacobi_recurrence(n).unwrap();
            let expected = HermitianEigenvaluesResult::tridiagonal(
                recurrence.diagonal(),
                recurrence.off_diagonal(),
            )
            .unwrap()
            .into_eigenvalues();
            assert_eq!(family.zeros(n).unwrap(), expected, "{family:?} {n}");
        }
    }
}
