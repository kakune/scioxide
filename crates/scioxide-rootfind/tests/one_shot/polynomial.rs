//! `polynomial_roots`.
use scioxide_core::{
    Argument, Complex, InvalidInputReason, NonFiniteOrigin, NumericScalar, RealScalar,
    SciOxideError, SciOxideErrorView, argument,
};
use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily};
use scioxide_rootfind::{
    PolynomialRootInput, PolynomialRootMethod, PolynomialRootOptions, RootTermination,
    polynomial_roots, polynomial_roots_with_options,
};

fn invalid(error: &SciOxideError) -> Option<(Argument, InvalidInputReason)> {
    match error.view() {
        SciOxideErrorView::InvalidInput {
            argument, reason, ..
        } => Some((argument, reason)),
        _ => None,
    }
}

fn c<S: RealScalar>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}

/// Asserts that both root sets have the same values and signs of zero.
fn assert_same_roots<S: RealScalar>(actual: &[Complex<S>], expected: &[Complex<S>]) {
    assert_eq!(actual, expected);
    for (a, e) in actual.iter().zip(expected) {
        assert_eq!(a.re.is_sign_negative(), e.re.is_sign_negative());
        assert_eq!(a.im.is_sign_negative(), e.im.is_sign_negative());
    }
}

fn matches_default_options<'a, S, C>(input: impl Into<PolynomialRootInput<'a, C>> + Copy)
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
    C: NumericScalar<Real = S> + 'a,
{
    let configured =
        polynomial_roots_with_options(input, &PolynomialRootOptions::default()).unwrap();
    assert!(configured.report().termination().is_success());
    let one_shot = polynomial_roots(input).unwrap();
    assert_same_roots(&one_shot, configured.roots());
}

fn monomial_inputs_match_default_options<S>()
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    // x³ − 6x² + 11x − 6, x² + 1, x⁴ + 5x² + 4 and x⁷⁰ − 1, which `Auto`
    // solves by closed forms, companion eigenvalues and Aberth–Ehrlich iteration.
    matches_default_options(&[-6.0, 11.0, -6.0, 1.0].map(c::<S>));
    matches_default_options(&[1.0, 0.0, 1.0].map(c::<S>));
    matches_default_options(&[4.0, 0.0, 5.0, 0.0, 1.0].map(c::<S>));
    let mut unity = vec![S::ZERO; 71];
    unity[0] = -S::ONE;
    unity[70] = S::ONE;
    matches_default_options(&unity);
    let complex = [(-2.0, 2.0), (-1.0, -1.0), (-1.0, 0.0), (1.0, 0.0)]
        .map(|(re, im)| Complex::new(c::<S>(re), c::<S>(im)));
    matches_default_options(&complex);
}

fn chebyshev_input_matches_default_options<S>()
where
    S: RealScalar + NumericScalar<Real = S>,
    Complex<S>: NumericScalar<Real = S>,
{
    // T₃ + T₁/2 on [−1, 1].
    let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT).unwrap();
    let p = ApproxPolynomial::orthogonal(vec![S::ZERO, c(0.5), S::ZERO, S::ONE], spec).unwrap();
    matches_default_options(&p);
}

#[test]
fn polynomial_roots_equals_the_configured_form_with_default_options() {
    monomial_inputs_match_default_options::<f64>();
    monomial_inputs_match_default_options::<f32>();
    chebyshev_input_matches_default_options::<f64>();
    chebyshev_input_matches_default_options::<f32>();
}

#[test]
fn polynomial_roots_reaches_reference_values() {
    let roots = polynomial_roots(&[-6.0_f64, 11.0, -6.0, 1.0]).unwrap();
    assert_eq!(roots.len(), 3);
    for (root, expected) in roots.iter().zip([1.0, 2.0, 3.0]) {
        assert!((root.re - expected).abs() <= 1e-13, "{roots:?}");
        assert!(root.im.abs() <= 1e-13, "{roots:?}");
    }
    let roots = polynomial_roots(&[-6.0_f32, 11.0, -6.0, 1.0]).unwrap();
    for (root, expected) in roots.iter().zip([1.0, 2.0, 3.0]) {
        assert!((root.re - expected).abs() <= 1e-5, "{roots:?}");
        assert!(root.im.abs() <= 1e-5, "{roots:?}");
    }
    // x² + 1 has the conjugate pair ∓i, sorted by imaginary part.
    let roots = polynomial_roots(&[1.0_f64, 0.0, 1.0]).unwrap();
    assert!((roots[0] - Complex::new(0.0, -1.0)).norm() <= 1e-15);
    assert!((roots[1] - Complex::new(0.0, 1.0)).norm() <= 1e-15);
}

#[test]
fn polynomial_roots_reports_its_own_arguments() {
    for coefficients in [&[1.0_f64][..], &[1.0, 0.0][..], &[][..]] {
        let error = polynomial_roots(coefficients).unwrap_err();
        assert_eq!(
            invalid(&error),
            Some((argument!("input"), InvalidInputReason::OutOfRange))
        );
    }
    let error = polynomial_roots(&[f64::NAN, 1.0]).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
}

#[test]
fn unsupported_methods_name_the_options_method() {
    let spec = OrthogonalBasisSpec::classical(OrthogonalFamily::ChebyshevT).unwrap();
    let p = ApproxPolynomial::orthogonal(vec![0.0_f64, 0.0, 0.0, 1.0], spec).unwrap();
    for method in [
        PolynomialRootMethod::Laguerre,
        PolynomialRootMethod::DurandKerner,
    ] {
        let options = PolynomialRootOptions::default().with_method(method);
        let error = polynomial_roots_with_options(&p, &options).unwrap_err();
        assert_eq!(
            invalid(&error),
            Some((argument!("options.method"), InvalidInputReason::OutOfRange))
        );
    }
}

#[test]
fn an_unsuccessful_termination_is_an_error() {
    // The roots 1 and about 2¹⁰⁷⁴ of 1 − x + 2⁻¹⁰⁷⁴x² leave the floating-point
    // range, so the configured form ends with a numerical failure.
    let coefficients = [1.0_f64, -1.0, f64::from_bits(1)];
    let configured =
        polynomial_roots_with_options(&coefficients, &PolynomialRootOptions::default()).unwrap();
    let termination = configured.report().termination();
    let error = polynomial_roots(&coefficients).unwrap_err();
    assert!(
        matches!(
            (termination, error.view()),
            (
                RootTermination::NumericalFailure(issue),
                SciOxideErrorView::NumericalInstability { issue: reported, .. },
            ) if reported == issue
        ),
        "{termination:?}: {error:?}"
    );
}
