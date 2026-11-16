//! Recurrence linearizations against tridiagonal zeros and analytic polynomials.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideErrorKind};
use scioxide_linalg::{GeneralEigenvaluesResult, Matrix};
use scioxide_poly::*;

fn families<R: RealScalar>() -> [OrthogonalFamily<R>; 9] {
    let r = |x| R::try_from_f64_lossy(x).unwrap();
    [
        OrthogonalFamily::ChebyshevT,
        OrthogonalFamily::ChebyshevU,
        OrthogonalFamily::Legendre,
        OrthogonalFamily::Jacobi {
            alpha: r(0.25),
            beta: r(0.75),
        },
        OrthogonalFamily::Gegenbauer { lambda: r(1.25) },
        OrthogonalFamily::Laguerre,
        OrthogonalFamily::GeneralizedLaguerre { alpha: r(0.5) },
        OrthogonalFamily::HermiteH,
        OrthogonalFamily::HermiteHe,
    ]
}

fn all_linearizations<R: RealScalar + NumericScalar<Real = R>>() {
    let r = |x| R::try_from_f64_lossy(x).unwrap();
    let accuracy = r(512.) * R::EPSILON;
    for family in families::<R>() {
        for normalization in [
            BasisNormalization::Classical,
            BasisNormalization::Monic,
            BasisNormalization::Orthonormal,
        ] {
            for (shift, scale) in [(r(0.), r(1.)), (r(-2.), r(3.))] {
                for degree in [1, 2, 5, 8] {
                    let spec =
                        OrthogonalBasisSpec::new(family, normalization, shift, scale).unwrap();
                    let mut coefficients = vec![r(0.); degree + 1];
                    coefficients[degree] = r(1.);
                    let basis_plan = spec
                        .prepare(coefficients.len() - 1, Default::default())
                        .unwrap();
                    let p = ApproxPolynomial::orthogonal_with_plan(
                        coefficients,
                        basis_plan,
                        Default::default(),
                    )
                    .unwrap();
                    let matrix = p.root_matrix().unwrap();
                    let eigen = GeneralEigenvaluesResult::compute(&matrix).unwrap();
                    let mut actual = eigen.eigenvalues().to_vec();
                    actual.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
                    let zeros = family.zeros(degree).unwrap();
                    for (root, zero) in actual.iter().zip(zeros) {
                        let expected = shift + scale * zero;
                        assert!(
                            (root.re - expected).abs() <= accuracy * (r(1.) + expected.abs()),
                            "{family:?}/{normalization:?}/{degree}: {root:?} vs {expected:?}"
                        );
                        assert!(root.im.abs() <= accuracy * (r(1.) + expected.abs()));
                        let evaluation = checked_basis_evaluation(
                            p.basis().coefficients(),
                            *root,
                            p.basis().root_basis(),
                        )
                        .unwrap();
                        assert_eq!(
                            evaluation.backward_error_model(),
                            PolynomialBackwardErrorModel::Normwise
                        );
                        // A matrix eigenproblem controls root displacement; monic affine
                        // bases can have a much smaller coefficient perturbation scale.
                        let displacement =
                            (evaluation.value().unwrap() / evaluation.derivative().unwrap()).norm();
                        assert!(
                            displacement <= accuracy * (r(1.) + expected.abs()),
                            "{family:?}/{normalization:?}/{degree}: {displacement:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn every_family_normalization_and_affine_map_f64() {
    all_linearizations::<f64>();
}
#[test]
fn every_family_normalization_and_affine_map_f32() {
    all_linearizations::<f32>();
}

#[test]
fn quadratic_roots_have_closed_form_references() {
    let expected = [
        [-0.5_f64.sqrt(), 0.5_f64.sqrt()],
        [-0.5, 0.5],
        [-(1.0_f64 / 3.).sqrt(), (1.0_f64 / 3.).sqrt()],
        [0.1 - 99.0_f64.sqrt() / 20., 0.1 + 99.0_f64.sqrt() / 20.],
        [-(1.0_f64 / 4.5).sqrt(), (1.0_f64 / 4.5).sqrt()],
        [2. - 2.0_f64.sqrt(), 2. + 2.0_f64.sqrt()],
        [2.5 - 2.5_f64.sqrt(), 2.5 + 2.5_f64.sqrt()],
        [-0.5_f64.sqrt(), 0.5_f64.sqrt()],
        [-1., 1.],
    ];
    for (family, roots) in families::<f64>().into_iter().zip(expected) {
        let spec = OrthogonalBasisSpec::new(family, BasisNormalization::Classical, 0., 1.).unwrap();
        let p = ApproxPolynomial::orthogonal_with_plan(
            vec![0., 0., 1.],
            spec.prepare(2, Default::default()).unwrap(),
            Default::default(),
        )
        .unwrap();
        for root in roots {
            let result = checked_basis_evaluation(
                p.basis().coefficients(),
                Complex::new(root, 0.),
                p.basis().root_basis(),
            )
            .unwrap();
            assert!(result.normalized_residual() < 8e-15, "{family:?}: {root}");
        }
    }
}

#[test]
fn mixed_complex_series_has_independent_values_and_physical_derivatives() {
    // Monic mapped Legendre: Q0=1, Q1=x-2, Q2=(x-2)²-3.
    let spec = OrthogonalBasisSpec::new(
        OrthogonalFamily::Legendre,
        BasisNormalization::Monic,
        2.,
        3.,
    )
    .unwrap();
    let c = [
        Complex::new(1., 2.),
        Complex::new(-3., 1.),
        Complex::new(2., -0.5),
    ];
    let p = ApproxPolynomial::orthogonal_with_plan(
        c.to_vec(),
        spec.prepare(c.len() - 1, Default::default()).unwrap(),
        Default::default(),
    )
    .unwrap();
    for z in [Complex::new(0.25, 0.5), Complex::new(4., -1.)] {
        let t = z - Complex::new(2., 0.);
        let evaluation = checked_basis_evaluation(&c, z, p.basis().root_basis()).unwrap();
        let expected = c[0] + c[1] * t + c[2] * (t * t - Complex::new(3., 0.));
        assert!((evaluation.value().unwrap() - expected).norm() < 2e-14);
        assert!(
            (evaluation.derivative().unwrap() - (c[1] + c[2] * t * Complex::new(2., 0.))).norm()
                < 2e-14
        );
        assert!(
            (evaluation.second_derivative().unwrap() - c[2] * Complex::new(2., 0.)).norm() < 2e-14
        );
        let bound = c.iter().map(|v| v.norm()).fold(0.0_f64, f64::max)
            * (1. + t.norm() + (t * t - Complex::new(3., 0.)).norm());
        assert!((evaluation.normalized_residual() - expected.norm() / bound).abs() < 2e-15);
    }
    let matrix = p.root_matrix().unwrap();
    let roots = GeneralEigenvaluesResult::<f64>::compute(&matrix).unwrap();
    for root in roots.eigenvalues() {
        assert!(
            checked_basis_evaluation(&c, *root, p.basis().root_basis())
                .unwrap()
                .normalized_residual()
                < 1e-14
        );
    }
}

#[test]
fn root_matrix_preflight_preserves_scratch_and_scaled_ratios() {
    let plan = OrthogonalBasisSpec::new(
        OrthogonalFamily::Jacobi {
            alpha: 0.25_f64,
            beta: 0.75,
        },
        BasisNormalization::Orthonormal,
        4.,
        2.,
    )
    .unwrap()
    .prepare(3, Default::default())
    .unwrap();
    let basis = PolynomialRootBasisRef::Orthogonal(&plan);
    let mut matrix = Matrix::from_vec(2, 2, vec![99.; 4]).unwrap();
    for (coefficients, kind) in [
        (&[1.][..], SciOxideErrorKind::InvalidInput),
        (&[1., 2.][..], SciOxideErrorKind::DimensionMismatch),
        (&[1., 2., 0.][..], SciOxideErrorKind::InvalidInput),
        (&[1., f64::NAN, 1.][..], SciOxideErrorKind::NonFinite),
    ] {
        assert_eq!(
            polynomial_root_matrix_into(
                coefficients,
                basis,
                &mut matrix,
                scioxide_core::ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
            kind
        );
        assert_eq!(matrix.as_slice(), &[99.; 4]);
    }
    let coefficients = [1., -2., 3.];
    polynomial_root_matrix_into(
        &coefficients,
        basis,
        &mut matrix,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
    let expected = matrix.clone();
    for exponent in [-500, 500] {
        polynomial_root_matrix_into(
            &coefficients.map(|x| x * 2.0_f64.powi(exponent)),
            basis,
            &mut matrix,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert_eq!(matrix, expected);
    }
    assert!(matches!(
        checked_basis_evaluation(&[0.; 5], Complex::new(0., 0.), basis)
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput {
            reason: scioxide_core::InvalidInputReason::Inconsistent,
            ..
        }
    ));
}

#[test]
fn root_matrix_entry_overflow_leaves_the_matrix_unchanged() {
    let chebyshev = OrthogonalBasisSpec::new(
        OrthogonalFamily::ChebyshevT,
        BasisNormalization::Classical,
        0.0_f64,
        1.,
    )
    .unwrap()
    .prepare(2, Default::default())
    .unwrap();
    // Each quotient c_i / c_n ~ 1e600 overflows only in the last column, after
    // every structural entry of the companion or comrade matrix is known.
    let coefficients = [1e300, 1e300, 1e-300];
    for basis in [
        PolynomialRootBasisRef::Monomial,
        PolynomialRootBasisRef::Orthogonal(&chebyshev),
    ] {
        let mut matrix = Matrix::from_vec(2, 2, vec![99.; 4]).unwrap();
        let error = polynomial_root_matrix_into(
            &coefficients,
            basis,
            &mut matrix,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert!(
            matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::Overflow,
                    ..
                }
            ),
            "{error:?}"
        );
        assert_eq!(matrix.as_slice(), &[99.; 4]);
        // The same matrix is filled completely by a representable expansion.
        polynomial_root_matrix_into(
            &[2., -3., 1.],
            basis,
            &mut matrix,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        assert!(matrix.as_slice().iter().all(|value| *value != 99.));
    }
}

#[test]
fn native_sensitivity_keeps_unrequested_fields_in_extended_range() {
    let spec = OrthogonalBasisSpec::new(
        OrthogonalFamily::ChebyshevT,
        BasisNormalization::Classical,
        0.0_f64,
        1.,
    )
    .unwrap();
    let p = ApproxPolynomial::orthogonal_with_plan(
        vec![0., 0., f64::MAX],
        spec.prepare(2, Default::default()).unwrap(),
        Default::default(),
    )
    .unwrap();
    let condition = p.root_condition(0.).unwrap();
    assert_eq!(condition.coefficient().value(), Some(2.));
    assert_eq!(condition.absolute(), RootEstimate::Singular);
    let evaluation = checked_basis_evaluation(
        p.basis().coefficients(),
        Complex::new(0., 0.),
        p.basis().root_basis(),
    )
    .unwrap();
    assert_eq!(evaluation.root_condition().unwrap(), condition);
    assert_eq!(evaluation.value().unwrap().re, -f64::MAX);
    assert_eq!(
        evaluation.second_derivative().unwrap_err().kind(),
        SciOxideErrorKind::NumericalInstability
    );
}

#[test]
fn linear_matrix_entries_retry_unsafe_intermediate_ratios() {
    for (scale, coefficients, expected) in [
        (1e-300_f64, [1e300, 1e-200], -1e200),
        (1e300, [1e-300, 1e300], -1e-300),
    ] {
        let spec = OrthogonalBasisSpec::new(
            OrthogonalFamily::Legendre,
            BasisNormalization::Classical,
            0.,
            scale,
        )
        .unwrap();
        let p = ApproxPolynomial::orthogonal_with_plan(
            coefficients.to_vec(),
            spec.prepare(coefficients.len() - 1, Default::default())
                .unwrap(),
            Default::default(),
        )
        .unwrap();
        let entry = p.root_matrix().unwrap().as_slice()[0];
        assert!((entry / expected - 1.).abs() <= 4. * f64::EPSILON);
    }
}
