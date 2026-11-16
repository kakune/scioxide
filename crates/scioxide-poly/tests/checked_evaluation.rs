//! The single checked point-evaluation route: values, residuals, rounding
//! allowance, conditioning and one error classification for every basis.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Complex, InvalidInputReason, NonFiniteOrigin, NumericalIssue, RealScalar, SciOxideError,
    SciOxideErrorView,
};
use scioxide_poly::{
    BasisNormalization, OrthogonalBasisPlan, OrthogonalBasisSpec, OrthogonalFamily,
    PolynomialBackwardErrorModel, PolynomialRootBasisRef as Basis, checked_basis_evaluation,
};

fn plan<S: RealScalar>(family: OrthogonalFamily<S>, degree: usize) -> OrthogonalBasisPlan<S> {
    OrthogonalBasisSpec::new(family, BasisNormalization::Classical, S::ZERO, S::ONE)
        .unwrap()
        .prepare(degree, Default::default())
        .unwrap()
}

fn real<S: RealScalar>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}

fn widths<S: RealScalar>() {
    // c = -1 + x/2 + x^3 in each basis on the identity coordinate map:
    // monomial x^3 + x/2 - 1, Chebyshev 4x^3 - 5x/2 - 1, Legendre 5x^3/2 - x - 1,
    // each written as `cubic x^3 - linear x - 1`.
    let c = [-S::ONE, S::HALF, S::ZERO, S::ONE];
    let chebyshev = plan::<S>(OrthogonalFamily::ChebyshevT, 3);
    let legendre = plan::<S>(OrthogonalFamily::Legendre, 3);
    let cases = [
        (
            Basis::Monomial,
            [1.0, -0.5],
            PolynomialBackwardErrorModel::Componentwise,
        ),
        (
            Basis::Orthogonal(&chebyshev),
            [4.0, 2.5],
            PolynomialBackwardErrorModel::Normwise,
        ),
        (
            Basis::Orthogonal(&legendre),
            [2.5, 1.0],
            PolynomialBackwardErrorModel::Normwise,
        ),
    ];
    let tolerance = real::<S>(32.0) * S::EPSILON;
    for (basis, [cubic, linear], model) in cases {
        let (cubic, linear) = (Complex::new(real::<S>(cubic), S::ZERO), real::<S>(linear));
        for z in [
            Complex::new(S::HALF, S::ZERO),
            Complex::new(S::HALF, S::HALF),
        ] {
            let e = checked_basis_evaluation(&c, z, basis).unwrap();
            assert_eq!(e.backward_error_model(), model);
            let linear = Complex::new(linear, S::ZERO);
            let three = Complex::new(real::<S>(3.0), S::ZERO);
            let six = Complex::new(real::<S>(6.0), S::ZERO);
            for (actual, expected) in [
                (
                    e.value().unwrap(),
                    cubic * z * z * z - linear * z - Complex::new(S::ONE, S::ZERO),
                ),
                (e.derivative().unwrap(), three * cubic * z * z - linear),
                (e.second_derivative().unwrap(), six * cubic * z),
            ] {
                assert!((actual - expected).norm() <= tolerance * (S::ONE + expected.norm()));
            }
            for (scaled, raw) in [
                (e.scaled_value().unwrap(), e.value().unwrap()),
                (e.scaled_derivative().unwrap(), e.derivative().unwrap()),
                (
                    e.scaled_second_derivative().unwrap(),
                    e.second_derivative().unwrap(),
                ),
            ] {
                assert_eq!(scaled.round().unwrap(), raw);
            }
            assert!(e.normalized_residual() > S::ZERO);
            assert!(e.residual_with_roundoff() > e.normalized_residual());
            assert!(!e.roundoff_limited());
            let condition = e.root_condition().unwrap().absolute().value().unwrap();
            let residual = e.value().unwrap().norm() / e.normalized_residual();
            let derivative = e.derivative().unwrap().norm();
            assert!((condition - residual / derivative).abs() <= tolerance * condition);
        }
    }
    // The monomial envelope is componentwise: sum |a_i| |z|^i.
    let z = Complex::new(S::HALF, S::ZERO);
    let e = checked_basis_evaluation(&c, z, Basis::Monomial).unwrap();
    let envelope = S::ONE + real::<S>(0.25) + real::<S>(0.125);
    let expected = e.value().unwrap().norm() / envelope;
    assert!((e.normalized_residual() - expected).abs() <= tolerance * expected);
    let root = checked_basis_evaluation(
        &[-S::ONE, S::ZERO, S::ONE],
        Complex::new(S::ONE, S::ZERO),
        Basis::Monomial,
    )
    .unwrap();
    assert!(root.roundoff_limited());
    assert_eq!(root.normalized_residual(), S::ZERO);
}

#[test]
fn native_basis_values_derivatives_and_residuals_in_both_widths() {
    widths::<f32>();
    widths::<f64>();
}

#[test]
fn extended_range_envelopes_and_gradual_underflow_remain_meaningful() {
    // The coefficient envelope exceeds f64 although cancellation leaves every
    // requested output representable. Its ratio to the derivative is also finite.
    let c = [f64::MAX, -f64::MAX];
    let e = checked_basis_evaluation(&c, Complex::new(1.0, 0.0), Basis::Monomial).unwrap();
    assert_eq!(e.value().unwrap(), Complex::new(0.0, 0.0));
    let condition = e.root_condition().unwrap().absolute().value().unwrap();
    assert!((condition - 2.0).abs() < 8.0 * f64::EPSILON);
    assert!(e.roundoff_limited());
    let tiny = f64::from_bits(1);
    let e =
        checked_basis_evaluation(&[tiny, tiny], Complex::new(0.0, 0.0), Basis::Monomial).unwrap();
    assert_eq!(e.value().unwrap().re, tiny);
    assert_eq!(e.derivative().unwrap().re, tiny);
    assert_eq!(e.normalized_residual(), 1.0);
    assert!(e.residual_with_roundoff() >= 1.0);
    let e = checked_basis_evaluation(&[0.0_f64], Complex::new(0.0, 0.0), Basis::Monomial).unwrap();
    assert!(e.roundoff_limited());
    assert_eq!(e.root_condition().unwrap().absolute().value(), None);
}

#[test]
fn corrections_and_diagnostics_do_not_project_unused_outputs() {
    let c = [f64::MAX * 0.75; 2];
    let point = Complex::new(1.0, 0.0);
    let e = checked_basis_evaluation(&c, point, Basis::Monomial).unwrap();
    assert!(e.value().is_err());
    assert!((e.newton_correction().unwrap().unwrap().re - 2.0).abs() <= 8.0 * f64::EPSILON);
    assert_eq!(e.normalized_residual(), 1.0);
    assert_eq!(e.root_condition().unwrap().absolute().value(), Some(2.0));
    let mut c = [0.0_f64; 9];
    c[0] = -1.0;
    c[8] = 1.0;
    let e = checked_basis_evaluation(&c, Complex::new(1e100, 0.0), Basis::Monomial).unwrap();
    assert!(e.value().is_err());
    assert!(e.derivative().is_err());
    assert!(
        (e.newton_correction().unwrap().unwrap().re / 1e100 - 0.125).abs() <= 8.0 * f64::EPSILON
    );
    let constant = checked_basis_evaluation(&[2.0_f64], point, Basis::Monomial).unwrap();
    assert_eq!(constant.newton_correction().unwrap(), None);
}

#[test]
fn native_and_extended_states_agree_at_the_range_boundary() {
    // The same polynomial evaluated natively and, after scaling every
    // coefficient by 2^1000, in extended range has the same scale-free residual.
    let c = [-1.0_f64, 0.25, 0.0, 1.0];
    let scale = 2.0_f64.powi(1000);
    let wide = c.map(|value| value * scale);
    for basis in [
        Basis::Monomial,
        Basis::Orthogonal(&plan::<f64>(OrthogonalFamily::ChebyshevT, 3)),
    ] {
        let z = Complex::new(0.75, 0.25);
        let native = checked_basis_evaluation(&c, z, basis).unwrap();
        let extended = checked_basis_evaluation(&wide, z, basis).unwrap();
        let tolerance = 32.0 * f64::EPSILON * native.normalized_residual();
        assert!((native.normalized_residual() - extended.normalized_residual()).abs() <= tolerance);
        assert_eq!(
            native.root_condition().unwrap().absolute().value(),
            extended.root_condition().unwrap().absolute().value()
        );
    }
}

fn assert_empty(error: SciOxideError) {
    assert!(matches!(
        error.view(),
        SciOxideErrorView::InvalidInput {
            argument,
            reason: InvalidInputReason::Empty,
            ..
        } if argument == scioxide_core::argument!("coefficients")
    ));
}

fn assert_nonfinite_input(error: SciOxideError) {
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Input,
            ..
        }
    ));
}

#[test]
fn invalid_inputs_share_one_classification_for_every_basis() {
    let chebyshev = plan::<f64>(OrthogonalFamily::ChebyshevT, 3);
    let origin = Complex::new(0.0, 0.0);
    for basis in [Basis::Monomial, Basis::Orthogonal(&chebyshev)] {
        assert_empty(checked_basis_evaluation::<f64>(&[], origin, basis).unwrap_err());
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_nonfinite_input(
                checked_basis_evaluation(&[1.0, bad], origin, basis).unwrap_err(),
            );
            assert_nonfinite_input(
                checked_basis_evaluation(&[Complex::new(1.0, bad)], origin, basis).unwrap_err(),
            );
            for point in [Complex::new(bad, 0.0), Complex::new(0.0, bad)] {
                assert_nonfinite_input(
                    checked_basis_evaluation(&[1.0, 2.0], point, basis).unwrap_err(),
                );
            }
        }
    }
}

fn assert_ill_conditioned(error: SciOxideError) {
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NumericalInstability {
            issue: NumericalIssue::IllConditioned,
            ..
        }
    ));
}

#[test]
fn long_binary32_inputs_have_no_rounding_allowance_in_any_basis() {
    // gamma_(8n) for monomials needs 8 n 2^-23 < 1, i.e. n < 2^20.
    let point = Complex::new(0.5_f32, 0.0);
    let mut c = vec![0.0_f32; (1 << 20) - 1];
    c[0] = 1.0;
    assert!(checked_basis_evaluation(&c, point, Basis::Monomial).is_ok());
    for len in [1 << 20, (1 << 20) + 1] {
        c.resize(len, 0.0);
        assert_ill_conditioned(checked_basis_evaluation(&c, point, Basis::Monomial).unwrap_err());
    }
    // gamma_(32n) for orthogonal recurrences needs n < 2^18.
    let degree = 1 << 18;
    let legendre = plan::<f32>(OrthogonalFamily::Legendre, degree);
    c.truncate(degree - 1);
    assert!(checked_basis_evaluation(&c, point, Basis::Orthogonal(&legendre)).is_ok());
    c.resize(degree + 1, 0.0);
    assert_ill_conditioned(
        checked_basis_evaluation(&c, point, Basis::Orthogonal(&legendre)).unwrap_err(),
    );
}

#[test]
fn orthogonal_coefficients_beyond_the_prepared_degree_contradict_the_plan() {
    let chebyshev = plan::<f64>(OrthogonalFamily::ChebyshevT, 2);
    assert!(matches!(
        checked_basis_evaluation(
            &[1.0; 4],
            Complex::new(0.0, 0.0),
            Basis::Orthogonal(&chebyshev)
        )
        .unwrap_err()
        .view(),
        SciOxideErrorView::InvalidInput {
            reason: scioxide_core::InvalidInputReason::Inconsistent,
            ..
        }
    ));
}
