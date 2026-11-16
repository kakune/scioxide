//! Prepared recurrence roots retain normalization, mapping and native diagnostics.
#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideErrorKind};
use scioxide_poly::PolynomialBasis;
use scioxide_poly::{
    ApproxPolynomial, BasisNormalization, OrthogonalBasisSpec, OrthogonalFamily,
    PolynomialBackwardErrorModel, PolynomialRootBasis, checked_basis_evaluation,
};
use scioxide_rootfind::{
    PolynomialRootMethod, PolynomialRootOptions, PolynomialRootWorkspace, polynomial_roots_into,
    polynomial_roots_with_options,
};

fn exercise<R: RealScalar + NumericScalar<Real = R>>()
where
    Complex<R>: NumericScalar<Real = R>,
{
    let r = |x| R::try_from_f64_lossy(x).unwrap();
    let families = [
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
    ];
    for family in families {
        let zeros = family.zeros(3).unwrap();
        for normalization in [
            BasisNormalization::Classical,
            BasisNormalization::Monic,
            BasisNormalization::Orthonormal,
        ] {
            let spec = OrthogonalBasisSpec::new(family, normalization, r(2.), r(1.5)).unwrap();
            let p = ApproxPolynomial::orthogonal_with_plan(
                vec![r(0.), r(0.), r(0.), r(1.)],
                spec.prepare(3, Default::default()).unwrap(),
                Default::default(),
            )
            .unwrap();
            for method in [
                PolynomialRootMethod::Auto,
                PolynomialRootMethod::Companion,
                PolynomialRootMethod::Aberth,
            ] {
                let options = PolynomialRootOptions::default().with_method(method);
                let result = polynomial_roots_with_options(&p, &options).unwrap();
                assert!(
                    result.report().termination().is_success(),
                    "{family:?}/{normalization:?}/{method:?}: {:?}",
                    result.report()
                );
                assert_eq!(
                    result.report().backward_error_model(),
                    PolynomialBackwardErrorModel::Normwise
                );
                for (&zero, (root, diagnostic)) in zeros
                    .iter()
                    .zip(result.roots().iter().zip(result.diagnostics()))
                {
                    let expected = r(2.) + r(1.5) * zero;
                    let tolerance = r(512.) * R::EPSILON * (r(1.) + expected.abs());
                    assert!(
                        (root.re - expected).abs() <= tolerance,
                        "{family:?}/{method:?}: {root:?} vs {expected:?}"
                    );
                    assert!(root.im.abs() <= tolerance);
                    let checked = checked_basis_evaluation(
                        p.basis().coefficients(),
                        *root,
                        p.basis().root_basis(),
                    )
                    .unwrap();
                    assert_eq!(
                        diagnostic.backward_error(),
                        Some(checked.normalized_residual())
                    );
                    assert_eq!(
                        diagnostic.condition(),
                        checked.root_condition().unwrap().absolute()
                    );
                }
            }
        }
    }
}

#[test]
fn all_prepared_families_f64() {
    exercise::<f64>();
}
#[test]
fn all_prepared_families_f32() {
    exercise::<f32>();
}

#[test]
fn unsupported_methods_preserve_output_and_prepared_coefficients() {
    let spec = OrthogonalBasisSpec::new(
        OrthogonalFamily::Jacobi {
            alpha: 0.25_f64,
            beta: 0.75,
        },
        BasisNormalization::Orthonormal,
        2.,
        3.,
    )
    .unwrap();
    let p = ApproxPolynomial::orthogonal_with_plan(
        vec![0., 0., 0., 1.],
        spec.prepare(3, Default::default()).unwrap(),
        Default::default(),
    )
    .unwrap();
    let snapshot = p.clone();
    for method in [
        PolynomialRootMethod::Laguerre,
        PolynomialRootMethod::DurandKerner,
    ] {
        let options = PolynomialRootOptions::default().with_method(method);
        let mut workspace = PolynomialRootWorkspace::workspace(3, &options).unwrap();
        let mut output = [Complex::new(99., 7.); 3];
        assert_eq!(
            polynomial_roots_into(
                &p,
                &options,
                &mut output,
                &mut workspace,
                scioxide_core::ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
            SciOxideErrorKind::InvalidInput
        );
        assert_eq!(output, [Complex::new(99., 7.); 3]);
        assert_eq!(p, snapshot);
    }
}
