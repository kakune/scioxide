//! Monomial antiderivatives and integrals against closed forms and the
//! Bernstein and orthogonal calculus of the same polynomial.
#![allow(clippy::unwrap_used)]
use scioxide_core::{AlgebraicScalar, Complex, NumericScalar, RealScalar, SciOxideErrorKind};
use scioxide_poly::{ApproxPolynomial, OrthogonalBasisSpec, OrthogonalFamily, PolynomialBasis};

fn scalar<C: NumericScalar>(re: f64, im: f64) -> C {
    let r = |x| C::Real::try_from_f64_lossy(x).unwrap();
    C::from_components(r(re), if C::IS_REAL { r(0.) } else { r(im) })
}

fn relative<C: NumericScalar>(actual: C, expected: C) -> C::Real {
    (actual - expected).abs() / expected.abs().max(C::Real::ONE)
}

#[test]
fn closed_form_integrals_and_antiderivatives() {
    // ∫₀¹ (1 + 2x + 3x²) dx = 3, and ∫₋₁² gives 2 − (−1) + 4 − 1 + 8 + 1 = 15.
    let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 3.0]).unwrap();
    assert_eq!(p.integral(0.0, 1.0).unwrap(), 3.0);
    assert!((p.integral(-1.0, 2.0).unwrap() - 15.0).abs() < 1e-14);
    assert!((p.integral(2.0, -1.0).unwrap() + 15.0).abs() < 1e-14);
    let primitive = p.antiderivative(0.0, 0.0).unwrap();
    assert_eq!(primitive.basis().coefficients(), &[0.0, 1.0, 1.0, 1.0]);
    assert_eq!(primitive.policy(), p.policy());
    // F(2) = 5 anchors x + x² + x³ + c with c = 5 − 14 = −9.
    let anchored = p.antiderivative(2.0, 5.0).unwrap();
    assert_eq!(anchored.basis().coefficients(), &[-9.0, 1.0, 1.0, 1.0]);
}

fn identities<C: NumericScalar>(tolerance: f64) {
    let c = |re, im| scalar::<C>(re, im);
    let tol = C::Real::try_from_f64_lossy(tolerance).unwrap();
    let p =
        ApproxPolynomial::monomial(vec![c(0.5, 1.), c(-3., 0.5), c(1.25, 0.), c(2., -1.)]).unwrap();
    for (anchor, value) in [
        (c(0., 0.), c(1., -1.)),
        (c(-1.5, 0.5), c(2., 0.)),
        (c(3., 0.), c(0., 0.)),
    ] {
        let primitive = p.antiderivative(anchor, value).unwrap();
        assert!(relative(primitive.evaluate(anchor).unwrap(), value) <= tol);
        // Differentiating recovers p within two roundings per coefficient.
        let derivative = primitive.derivative().unwrap();
        let bound = C::Real::TWO * C::Real::EPSILON;
        for (&actual, &expected) in derivative
            .basis()
            .coefficients()
            .iter()
            .zip(p.basis().coefficients())
        {
            assert!(
                (actual - expected).abs() <= bound * expected.abs(),
                "{actual:?} {expected:?}"
            );
        }
    }
    let (lower, upper) = (c(-0.75, 0.25), c(1.5, -0.5));
    let integral = p.integral(lower, upper).unwrap();
    let primitive = p.antiderivative(c(0., 0.), c(0., 0.)).unwrap();
    let expected = primitive.evaluate(upper).unwrap() - primitive.evaluate(lower).unwrap();
    assert!(relative(integral, expected) <= tol);
    assert_eq!(p.integral(lower, lower).unwrap(), C::ZERO);
    // The zero polynomial integrates to the anchored constant.
    let zero = ApproxPolynomial::monomial(vec![C::ZERO]).unwrap();
    let constant = zero.antiderivative(c(2., 0.), c(-4., 1.)).unwrap();
    assert_eq!(constant.basis().coefficients(), &[c(-4., 1.)]);
    assert_eq!(zero.integral(lower, upper).unwrap(), C::ZERO);
}

#[test]
fn anchored_antiderivatives_invert_differentiation_in_four_scalar_domains() {
    identities::<f32>(1e-5);
    identities::<f64>(1e-12);
    identities::<Complex<f32>>(1e-5);
    identities::<Complex<f64>>(1e-12);
}

#[test]
fn integrals_agree_with_bernstein_and_orthogonal_calculus() {
    let p = ApproxPolynomial::monomial(vec![0.25_f64, -1.0, 2.0, 0.5, -0.75]).unwrap();
    for (lower, upper) in [(0.0, 1.0), (-1.0, 2.0), (0.5, -0.25), (-3.0, 3.0)] {
        let monomial = p.integral(lower, upper).unwrap();
        let bernstein = p
            .to_bernstein(-1.0, 2.0)
            .unwrap()
            .into_polynomial()
            .integral(lower, upper)
            .unwrap();
        let spec = OrthogonalBasisSpec::on_interval(OrthogonalFamily::Legendre, -1.0, 2.0).unwrap();
        let orthogonal = p
            .to_orthogonal(spec)
            .unwrap()
            .into_polynomial()
            .integral(lower, upper)
            .unwrap();
        for reference in [bernstein, orthogonal] {
            assert!(
                relative(monomial, reference) <= 1e-12,
                "{monomial} {reference}"
            );
        }
    }
}

#[test]
fn non_finite_limits_and_overflowing_constants_are_errors() {
    let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0]).unwrap();
    for (anchor, value) in [(f64::NAN, 0.0), (0.0, f64::INFINITY)] {
        assert_eq!(
            p.antiderivative(anchor, value).unwrap_err().kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    for (lower, upper) in [(f64::NEG_INFINITY, 0.0), (0.0, f64::NAN)] {
        assert_eq!(
            p.integral(lower, upper).unwrap_err().kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    // F(x) = MAX·x² / 2 + c with F(2) = 0 needs c = −2·MAX.
    let large = ApproxPolynomial::monomial(vec![0.0, f64::MAX]).unwrap();
    assert_eq!(
        large.antiderivative(2.0, 0.0).unwrap_err().kind(),
        SciOxideErrorKind::NumericalInstability
    );
    assert_eq!(
        large.integral(0.0, 2.0).unwrap_err().kind(),
        SciOxideErrorKind::NumericalInstability
    );
}
