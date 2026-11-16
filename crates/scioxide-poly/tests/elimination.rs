//! Numerical elimination against analytic and exact determinant identities.
#![allow(clippy::unwrap_used)]
use scioxide_core::AlgebraicScalar;
use scioxide_core::{
    Complex, ExecutionContext, InvalidInputReason, NumericScalar, RealScalar, SciOxideErrorView,
};
use scioxide_poly::{
    PolynomialApproximationPolicy, SylvesterDiscriminant, SylvesterElimination,
    approximate_discriminant, approximate_resultant, approximate_subresultant,
};
fn domains<C: NumericScalar>() {
    let c = |x| C::from_real(C::Real::try_from_f64_lossy(x).unwrap());
    let p = PolynomialApproximationPolicy::default();
    let tol = <C::Real as RealScalar>::EPSILON * C::Real::try_from_f64_lossy(128.).unwrap();
    let a = [c(1.), c(0.), c(1.)];
    let b = [c(-1.), c(0.), c(1.)];
    let r = approximate_resultant(&a, &b).unwrap();
    assert!((r.value().try_value().unwrap() - c(4.)).abs() < tol);
    assert_eq!(r.report().rank().unwrap().rank(), 4);
    let sub = approximate_subresultant(&a, &b, 1).unwrap();
    assert!((sub[0].value().try_value().unwrap() - c(-2.)).abs() < tol);
    assert_eq!(sub[1].value().try_value().unwrap(), c(0.));
    let error = approximate_subresultant(&a, &b, 0).unwrap_err();
    assert!(matches!(error.view(), SciOxideErrorView::InvalidInput {
        argument, reason: InvalidInputReason::OutOfRange, ..
    } if argument == scioxide_core::argument!("index")));
    let selected = SylvesterElimination::new(2, 2, 1, p).unwrap();
    let mut workspace = selected.workspace().unwrap();
    for (k, expected) in sub.iter().enumerate() {
        assert_eq!(
            selected
                .coefficient_scoped(
                    &a,
                    &b,
                    k,
                    &mut workspace,
                    ExecutionContext::sequential_native()
                )
                .unwrap(),
            *expected
        );
    }
    for (coeffs, expected) in [
        (vec![c(3.)], 1.),
        (vec![c(3.), c(2.)], 1.),
        (vec![c(1.), c(2.), c(3.)], -8.),
        (vec![c(1.), c(-2.), c(1.)], 0.),
        (vec![c(-1.), c(0.), c(0.), c(1.)], -27.),
    ] {
        let plan = SylvesterDiscriminant::new(coeffs.len() - 1, p).unwrap();
        let mut workspace = plan.workspace().unwrap();
        let prepared = plan
            .compute_scoped(
                &coeffs,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
        assert!((prepared.value().try_value().unwrap() - c(expected)).abs() < tol * c(10.).abs());
        assert!(
            (approximate_discriminant(&coeffs)
                .unwrap()
                .value()
                .try_value()
                .unwrap()
                - c(expected))
            .abs()
                < tol * c(10.).abs()
        );
    }
    let w = SylvesterElimination::new(2, 2, 0, p).unwrap();
    let mut workspace = w.workspace().unwrap();
    let context = ExecutionContext::sequential_native();
    assert_eq!(
        w.coefficient_scoped(&a, &b, 0, &mut workspace, context)
            .unwrap(),
        r
    );
    assert_eq!(
        w.coefficient_scoped(&a, &a, 0, &mut workspace, context)
            .unwrap()
            .value()
            .try_value()
            .unwrap(),
        c(0.)
    );
    assert_eq!(
        w.coefficient_scoped(&a, &b, 0, &mut workspace, context)
            .unwrap(),
        r
    );
    assert!(
        w.coefficient_scoped(&a[..2], &b, 0, &mut workspace, context)
            .is_err()
    );
    assert!(
        w.coefficient_scoped(&a, &b, 1, &mut workspace, context)
            .is_err()
    );
    assert_eq!(
        approximate_resultant(&[c(3.)], &b)
            .unwrap()
            .value()
            .try_value()
            .unwrap(),
        c(9.)
    );
    assert_eq!(
        approximate_resultant(&[c(3.)], &[c(2.)])
            .unwrap()
            .value()
            .try_value()
            .unwrap(),
        c(1.)
    );
    assert_eq!(
        approximate_resultant(&[c(0.)], &b)
            .unwrap()
            .value()
            .try_value()
            .unwrap(),
        c(0.)
    );
    assert!(approximate_resultant::<C>(&[], &b).is_err());
    assert!(approximate_discriminant(&[c(0.)]).is_err());
}
#[test]
fn four_scalar_domains() {
    domains::<f32>();
    domains::<f64>();
    domains::<Complex<f32>>();
    domains::<Complex<f64>>();
}
#[test]
fn complex_phase_and_scaling() {
    let a = [Complex::new(0., -1.), Complex::new(1., 0.)];
    let b = [Complex::new(-2., 0.), Complex::new(1., 0.)];
    let r = approximate_resultant(&a, &b)
        .unwrap()
        .value()
        .try_value()
        .unwrap();
    assert!((r - Complex::new(-2., 1.)).norm() < 1e-13);
    let a = [-2f64.powi(800), 2f64.powi(800)];
    let b = [-2f64.powi(801), 2f64.powi(800)];
    let r = approximate_resultant(&a, &b).unwrap().value();
    assert!(r.try_value().is_err());
    assert_eq!(r.real().exponent(), 1600);
    assert_eq!(r.real().significand(), -1.);
    let tiny = 2f64.powi(-800);
    let r = approximate_resultant(&[-tiny, tiny], &[-2. * tiny, tiny])
        .unwrap()
        .value();
    assert!(r.try_value().is_err());
    assert_eq!(r.real().exponent(), -1600);
}
#[test]
fn degree_and_near_common_root_are_not_silently_truncated() {
    let a = [-1., 1., 0.];
    let b = [-1. - 1e-12, 1.];
    let r = approximate_resultant(&a, &b).unwrap();
    assert_eq!(r.report().degrees(), [1, 1]);
    assert!((r.value().try_value().unwrap() + 1e-12).abs() < 1e-15);
    assert!(r.report().reciprocal_condition() < 1e-10);
    assert!(approximate_resultant(&[f64::NAN], &b).is_err());
}

#[test]
fn cancellation_and_recovery() {
    use scioxide_core::{CancellationToken, ExecutionContext};
    use scioxide_poly::SylvesterDiscriminant;
    let policy = PolynomialApproximationPolicy::default();
    let plan = SylvesterElimination::new(2, 2, 0, policy).unwrap();
    let (a, b) = ([1., 0., 1.], [-1., 0., 1.]);
    let expected = approximate_resultant(&a, &b).unwrap();
    let mut ws = plan.workspace().unwrap();
    let cx = ExecutionContext::sequential();
    for _ in 0..2 {
        let result = plan.coefficient_scoped(&a, &b, 0, &mut ws, cx).unwrap();
        assert_eq!(result, expected);
    }
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        plan.coefficient_scoped(&a, &b, 0, &mut ws, cx.with_cancellation(&token))
            .is_err()
    );
    assert_eq!(
        plan.coefficient_scoped(&a, &b, 0, &mut ws, cx).unwrap(),
        expected
    );
    let d = SylvesterDiscriminant::new(2, policy).unwrap();
    let mut dw = d.workspace().unwrap();
    let result = d.compute_scoped(&a, &mut dw, cx).unwrap();
    assert!((result.value().try_value().unwrap() + 4f64).abs() < 1e-13);
    assert!(
        d.compute_scoped(&a, &mut dw, cx.with_cancellation(&token))
            .is_err()
    );
    assert_eq!(d.compute_scoped(&a, &mut dw, cx).unwrap(), result);
    assert!(SylvesterElimination::<f64>::new(usize::MAX, 1, 0, policy).is_err());
    assert!(SylvesterElimination::<f64>::new(2, 2, 2, policy).is_err());
}

#[path = "support/exact_field.rs"]
mod exact_field;
#[test]
fn independent_exact_fields_qualify_numerical_resultants() {
    fn check<C: NumericScalar>() {
        use exact_field::Q;
        use scioxide_poly::{ExactPolynomial, exact_subresultants};
        let convert = |x: i128| C::from_real(C::Real::try_from_f64_lossy(x as f64).unwrap());
        for seed in 1..=12i128 {
            let a = [seed % 3 - 1, 2 - seed % 5, seed % 2 + 1, 1];
            let b = [seed % 5 - 2, seed % 3 - 1, 2];
            let exact =
                |v: &[i128]| ExactPolynomial::new(v.iter().map(|&x| Q(x, 1)).collect()).unwrap();
            let q = *exact_subresultants(&exact(&a), &exact(&b))
                .unwrap()
                .resultant();
            let expected = convert(q.0) / convert(q.1);
            let a: Vec<_> = a.into_iter().map(convert).collect();
            let b: Vec<_> = b.into_iter().map(convert).collect();
            let actual = approximate_resultant(&a, &b)
                .unwrap()
                .value()
                .try_value()
                .unwrap();
            let error = (actual - expected).abs();
            let tolerance = <C::Real as RealScalar>::EPSILON
                * C::Real::try_from_f64_lossy(512.).unwrap()
                * expected.abs().max(C::Real::ONE);
            assert!(error <= tolerance, "seed {seed}");
        }
    }
    check::<f32>();
    check::<f64>();
    check::<Complex<f32>>();
    check::<Complex<f64>>();
}
#[test]
fn nonreal_coefficients_in_both_precisions() {
    fn check<R: RealScalar>()
    where
        Complex<R>: NumericScalar<Real = R>,
    {
        let r = |x| R::try_from_f64_lossy(x).unwrap();
        let a = [Complex::new(r(1.), r(2.)), Complex::new(r(2.), r(-1.))];
        let b = [Complex::new(r(3.), r(-1.)), Complex::new(r(-1.), r(1.))];
        let expected = a[1] * b[0] - a[0] * b[1];
        let value = approximate_resultant(&a, &b)
            .unwrap()
            .value()
            .try_value()
            .unwrap();
        assert!((value - expected).norm() <= <R as RealScalar>::EPSILON * r(128.));
        let polynomial = [a[0], a[1], b[0]];
        let expected = a[1] * a[1] - Complex::new(r(4.), r(0.)) * a[0] * b[0];
        let value = approximate_discriminant(&polynomial)
            .unwrap()
            .value()
            .try_value()
            .unwrap();
        assert!(
            (value - expected).norm() <= <R as RealScalar>::EPSILON * r(512.) * expected.norm()
        );
    }
    check::<f32>();
    check::<f64>();
}
#[test]
fn discriminant_scaling_and_unrepresentable_equilibration() {
    let scale = 2f64.powi(800);
    let result = approximate_discriminant(&[scale, 0., scale])
        .unwrap()
        .value();
    assert_eq!(result.real().exponent(), 1602);
    assert_eq!(result.real().significand(), -1.);
    let tiny = f64::from_bits(1);
    assert!(approximate_resultant(&[tiny, f64::MAX], &[1., 1.]).is_err());
}
#[test]
fn unequal_degree_subresultant_minors() {
    // Independent three-by-three cofactor expansion gives 5 + 2x.
    // The matrices are [[1,3,1],[1,1,0],[0,1,-2]] and
    // [[1,3,2],[1,1,-2],[0,1,1]], respectively.
    let (a, b) = ([1f64, 2., 3., 1.], [-2f64, 1., 1.]);
    for result in [
        approximate_subresultant(&a, &b, 1).unwrap(),
        approximate_subresultant(&b, &a, 1).unwrap(),
    ] {
        for (coefficient, expected) in result.iter().zip([5., 2.]) {
            assert!((coefficient.value().try_value().unwrap() - expected).abs() < 1e-12);
        }
    }
}
