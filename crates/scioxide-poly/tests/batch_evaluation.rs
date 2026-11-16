//! Point and batch evaluation agree to the last bit in every basis, publish
//! atomically and allocate nothing in their `_into` forms.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    CancellationToken, Complex, ExecutionContext, NumericScalar, RealScalar, SciOxideErrorKind,
};
use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily, PolynomialBasis};

fn scalar<C: NumericScalar>(re: f64, im: f64) -> C {
    let r = |x| C::Real::try_from_f64_lossy(x).unwrap();
    C::from_components(r(re), if C::IS_REAL { r(0.) } else { r(im) })
}

/// Points inside and outside every interval used below.
fn points<C: NumericScalar>() -> Vec<C> {
    [
        (-1.5, 0.25),
        (-0.5, 0.0),
        (0.0, -0.5),
        (0.375, 0.125),
        (1.25, 0.0),
        (3.0, 1.0),
    ]
    .into_iter()
    .map(|(re, im)| scalar(re, im))
    .collect()
}

fn check<C: NumericScalar, B: PolynomialBasis<C>>(p: &ApproxPolynomial<C, B>, scratch_free: bool) {
    let points = points::<C>();
    let m = points.len();
    let cx = ExecutionContext::sequential();
    let mut scratch = vec![C::ZERO; p.workspace_requirements(m).unwrap().scratch_len()];
    let (mut values, mut derivatives) = (vec![C::ZERO; m], vec![C::ZERO; m]);
    p.evaluate_batch_with_derivative_into(&points, &mut values, &mut derivatives, &mut scratch, cx)
        .unwrap();
    let batch = p.evaluate_batch(&points).unwrap();
    for (k, &point) in points.iter().enumerate() {
        let full = p.evaluate_with_derivative(point).unwrap();
        let value = p.evaluate(point).unwrap();
        assert_eq!(value, full.value());
        assert_eq!(batch[k], value);
        assert_eq!(values[k], value);
        assert_eq!(derivatives[k], full.derivative());
    }
    let mut only = vec![C::ZERO; m];
    p.evaluate_batch_into(&points, &mut only, &mut scratch, cx)
        .unwrap();
    assert_eq!(only, batch);

    // An empty batch writes nothing and needs only the basis scratch.
    p.evaluate_batch_into(&[], &mut [], &mut scratch, cx)
        .unwrap();
    assert!(p.evaluate_batch(&[]).unwrap().is_empty());

    // Exactly the required scratch suffices; one entry short is rejected.
    let basis = p.workspace_requirements(0).unwrap().scratch_len();
    let required = m + basis;
    let mut exact = vec![C::ZERO; required];
    p.evaluate_batch_into(&points, &mut only, &mut exact, cx)
        .unwrap();
    let sentinel = vec![scalar::<C>(7.0, 7.0); m];
    let mut output = sentinel.clone();
    let error = p
        .evaluate_batch_into(&points, &mut output, &mut exact[..required - 1], cx)
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    let mut slopes = sentinel.clone();
    let both = 2 * m + basis;
    let error = p
        .evaluate_batch_with_derivative_into(
            &points,
            &mut output,
            &mut slopes,
            &mut scratch[..both - 1],
            cx,
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    let error = p
        .evaluate_batch_into(&points, &mut output[..m - 1], &mut scratch, cx)
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);
    let error = p
        .evaluate_batch_with_derivative_into(
            &points,
            &mut output,
            &mut slopes[..1],
            &mut scratch,
            cx,
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::DimensionMismatch);

    // A NaN point and a cancelled context leave both outputs unchanged.
    let mut poisoned = points.clone();
    poisoned[m / 2] = C::from_real(C::Real::NAN);
    let error = p
        .evaluate_batch_with_derivative_into(&poisoned, &mut output, &mut slopes, &mut scratch, cx)
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
    let token = CancellationToken::new();
    token.cancel();
    let cancelled = cx.with_cancellation(&token);
    let error = p
        .evaluate_batch_with_derivative_into(
            &points,
            &mut output,
            &mut slopes,
            &mut scratch,
            cancelled,
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ExecutionStopped);
    assert!(
        p.evaluate_batch_into(&points, &mut output, &mut scratch, cancelled)
            .is_err()
    );
    assert_eq!(output, sentinel);
    assert_eq!(slopes, sentinel);

    // The `_into` forms allocate nothing; the owned point forms allocate only
    // the Bernstein scratch.
    for _ in 0..2 {
        {
            p.evaluate_batch_into(&points, &mut values, &mut scratch, cx)
                .unwrap();
        };
        {
            p.evaluate_batch_with_derivative_into(
                &points,
                &mut values,
                &mut derivatives,
                &mut scratch,
                cx,
            )
            .unwrap();
        };
        if scratch_free {
            {
                std::hint::black_box(p.evaluate(points[0]).unwrap());
                let _ = std::hint::black_box(p.evaluate_with_derivative(points[1]).unwrap());
                std::hint::black_box(p.root_condition(points[2]).unwrap());
            };
        }
    }
}

fn all_bases<C: NumericScalar>() {
    let c = |re, im| scalar::<C>(re, im);
    let coefficients = vec![c(1., 0.5), c(-2., 0.), c(0.5, -0.25), c(3., 1.)];
    let nodes = vec![c(-1., 0.), c(0.5, 0.), c(2., 0.)];
    check(
        &ApproxPolynomial::monomial(coefficients.clone()).unwrap(),
        true,
    );
    check(
        &ApproxPolynomial::newton(coefficients.clone(), nodes.clone()).unwrap(),
        true,
    );
    let mut interpolation = nodes;
    interpolation.push(c(3., 0.));
    check(
        &ApproxPolynomial::interpolate_barycentric(interpolation, coefficients.clone()).unwrap(),
        true,
    );
    let r = |x| C::Real::try_from_f64_lossy(x).unwrap();
    check(
        &ApproxPolynomial::bernstein(coefficients.clone(), r(-1.), r(2.)).unwrap(),
        false,
    );
    let spec = OrthogonalBasisSpec::on_interval(OrthogonalFamily::Legendre, r(-1.), r(2.)).unwrap();
    check(
        &ApproxPolynomial::orthogonal(coefficients, spec).unwrap(),
        true,
    );
}

#[test]
fn every_basis_agrees_with_its_point_forms_in_four_scalar_domains() {
    all_bases::<f32>();
    all_bases::<f64>();
    all_bases::<Complex<f32>>();
    all_bases::<Complex<f64>>();
}

#[test]
fn point_evaluation_rejects_non_finite_points_and_reports_overflow() {
    let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0]).unwrap();
    assert_eq!(
        p.evaluate(f64::NAN).unwrap_err().kind(),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(
        p.evaluate_with_derivative(f64::INFINITY)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::NonFinite
    );
    let large = ApproxPolynomial::monomial(vec![0.0, f64::MAX]).unwrap();
    assert_eq!(
        large.evaluate(4.0).unwrap_err().kind(),
        SciOxideErrorKind::NumericalInstability
    );
    let mut output = [3.0; 2];
    let mut scratch = [0.0; 4];
    let error = large
        .evaluate_batch_into(
            &[1.0, 4.0],
            &mut output,
            &mut scratch,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::NumericalInstability);
    assert_eq!(output, [3.0; 2]);
}

#[test]
fn batch_requirements_count_values_derivatives_and_controls() {
    let bernstein = ApproxPolynomial::bernstein(vec![1.0_f64; 5], 0.0, 1.0).unwrap();
    assert_eq!(
        bernstein.workspace_requirements(0).unwrap().scratch_len(),
        5
    );
    assert_eq!(
        bernstein.workspace_requirements(3).unwrap().scratch_len(),
        11
    );
    let monomial = ApproxPolynomial::monomial(vec![1.0_f64; 5]).unwrap();
    assert_eq!(monomial.workspace_requirements(3).unwrap().scratch_len(), 6);
    assert_eq!(
        monomial
            .workspace_requirements(usize::MAX)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::SizeOverflow
    );
}
