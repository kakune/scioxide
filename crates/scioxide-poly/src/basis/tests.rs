use super::*;
use scioxide_core::Complex;

fn policy() -> PolynomialApproximationPolicy<f64> {
    PolynomialApproximationPolicy::new(
        scioxide_core::Tolerance::new(1.0e-14, 1.0e-14).unwrap(),
        RankPolicy::DimensionScaledEpsilon,
    )
}

#[test]
fn every_prepared_basis_evaluates_value_and_derivative() {
    let expected_value = 17.0;
    let expected_derivative = 14.0;
    let monomial = ApproxPolynomial::monomial_with_options(vec![1.0, 2.0, 3.0], policy()).unwrap();
    let newton =
        ApproxPolynomial::newton_with_options(vec![1.0, 5.0, 3.0], vec![0.0, 1.0], policy())
            .unwrap();
    let barycentric = ApproxPolynomial::interpolate_barycentric_with_options(
        vec![0.0, 1.0, 2.0],
        vec![1.0, 6.0, 17.0],
        policy(),
    )
    .unwrap();
    let chebyshev = ApproxPolynomial::orthogonal_with_plan(
        vec![2.5, 2.0, 1.5],
        crate::OrthogonalBasisSpec::new(
            crate::OrthogonalFamily::ChebyshevT,
            crate::BasisNormalization::Classical,
            0.0,
            1.0,
        )
        .unwrap()
        .prepare(2, Default::default())
        .unwrap(),
        policy(),
    )
    .unwrap();
    let legendre = ApproxPolynomial::orthogonal_with_plan(
        vec![2.0, 2.0, 2.0],
        crate::OrthogonalBasisSpec::new(
            crate::OrthogonalFamily::Legendre,
            crate::BasisNormalization::Classical,
            0.0,
            1.0,
        )
        .unwrap()
        .prepare(2, Default::default())
        .unwrap(),
        policy(),
    )
    .unwrap();
    let bernstein =
        ApproxPolynomial::bernstein_with_options(vec![1.0, 2.0, 6.0], 0.0, 1.0, policy()).unwrap();

    assert_eq!(
        monomial.classify_coefficients().unwrap().effective_degree(),
        2
    );
    assert_eq!(
        newton.classify_coefficients().unwrap().effective_degree(),
        2
    );
    assert_eq!(
        barycentric
            .classify_coefficients()
            .unwrap()
            .effective_degree(),
        2
    );
    assert_eq!(
        chebyshev
            .classify_coefficients()
            .unwrap()
            .effective_degree(),
        2
    );
    assert_eq!(
        legendre.classify_coefficients().unwrap().effective_degree(),
        2
    );
    assert_eq!(
        bernstein
            .classify_coefficients()
            .unwrap()
            .effective_degree(),
        2
    );
    for (index, evaluation) in [
        monomial.evaluate_with_derivative(2.0).unwrap(),
        newton.evaluate_with_derivative(2.0).unwrap(),
        barycentric.evaluate_with_derivative(2.0).unwrap(),
        chebyshev.evaluate_with_derivative(2.0).unwrap(),
        legendre.evaluate_with_derivative(2.0).unwrap(),
        bernstein.evaluate_with_derivative(2.0).unwrap(),
    ]
    .into_iter()
    .enumerate()
    {
        assert!(
            f64::abs(evaluation.value() - expected_value) <= 1.0e-12,
            "basis {index} value {}",
            evaluation.value()
        );
        assert!(
            f64::abs(evaluation.derivative() - expected_derivative) <= 1.0e-12,
            "basis {index} derivative {}",
            evaluation.derivative()
        );
    }
}

#[test]
fn interpolation_prepares_newton_and_barycentric_bases_directly() {
    let nodes = vec![-1.0_f64, 0.0, 2.0, 4.0];
    let values: Vec<_> = nodes
        .iter()
        .map(|node| node * node + 2.0 * node - 3.0)
        .collect();
    let newton =
        ApproxPolynomial::interpolate_newton_with_options(nodes.clone(), values.clone(), policy())
            .unwrap();
    let barycentric =
        ApproxPolynomial::interpolate_barycentric_with_options(nodes, values, policy()).unwrap();

    for point in [-0.75, 0.5, 3.25] {
        let expected = point * point + 2.0 * point - 3.0;
        let expected_derivative = 2.0 * point + 2.0;
        for evaluation in [
            newton.evaluate_with_derivative(point).unwrap(),
            barycentric.evaluate_with_derivative(point).unwrap(),
        ] {
            assert!((evaluation.value() - expected).abs() <= 2.0e-13);
            assert!((evaluation.derivative() - expected_derivative).abs() <= 2.0e-13);
        }
    }
}

#[test]
fn interpolation_preparation_supports_f32() {
    let polynomial = ApproxPolynomial::interpolate_barycentric(
        vec![-1.0_f32, 0.0, 1.0],
        vec![1.0_f32, 0.0, 1.0],
    )
    .unwrap();
    let evaluation = polynomial.evaluate_with_derivative(0.25).unwrap();
    assert!((evaluation.value() - 0.0625).abs() <= 16.0 * f32::EPSILON);
    assert!((evaluation.derivative() - 0.5).abs() <= 16.0 * f32::EPSILON);
}

#[test]
fn interpolation_validates_nodes_once_without_imposing_separation() {
    use scioxide_core::{NonFiniteOrigin, SciOxideErrorKind, SciOxideErrorView};
    for nodes in [
        vec![-0.0_f64, 0.0],
        vec![2.0, 1.0, 2.0],
        vec![1.0, 1.0, 2.0],
    ] {
        let values = vec![0.0; nodes.len()];
        for error in [
            ApproxPolynomial::interpolate_newton_with_options(
                nodes.clone(),
                values.clone(),
                policy(),
            )
            .unwrap_err(),
            ApproxPolynomial::interpolate_barycentric_with_options(nodes, values, policy())
                .unwrap_err(),
        ] {
            assert_eq!(error.kind(), SciOxideErrorKind::InvalidInput);
        }
    }
    for nodes in [vec![1.0, 1.0 + f64::EPSILON], vec![1.0 + f64::EPSILON, 1.0]] {
        let newton = ApproxPolynomial::interpolate_newton_with_options(
            nodes.clone(),
            nodes.clone(),
            policy(),
        )
        .unwrap();
        let barycentric =
            ApproxPolynomial::interpolate_barycentric_with_options(nodes.clone(), nodes, policy())
                .unwrap();
        for result in [
            newton.evaluate_with_derivative(0.5).unwrap(),
            barycentric.evaluate_with_derivative(0.5).unwrap(),
        ] {
            assert!((result.value() - 0.5).abs() <= 8.0 * f64::EPSILON);
            assert!((result.derivative() - 1.0).abs() <= 8.0 * f64::EPSILON);
        }
    }
    for (nodes, values) in [
        (vec![f64::NAN], vec![0.0]),
        (vec![0.0], vec![f64::INFINITY]),
    ] {
        for error in [
            ApproxPolynomial::interpolate_newton_with_options(
                nodes.clone(),
                values.clone(),
                policy(),
            )
            .unwrap_err(),
            ApproxPolynomial::interpolate_barycentric_with_options(nodes, values, policy())
                .unwrap_err(),
        ] {
            assert!(matches!(
                error.view(),
                SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
    }
    let constant = ApproxPolynomial::monomial_with_options(vec![2.0_f64], policy()).unwrap();
    assert_eq!(
        constant.to_newton(vec![f64::NAN]).unwrap_err().kind(),
        SciOxideErrorKind::NonFinite
    );
    assert_eq!(
        constant.to_barycentric(vec![f64::NAN]).unwrap_err().kind(),
        SciOxideErrorKind::NonFinite
    );
}

#[test]
fn divided_differences_keep_unrepresentable_intermediate_orders() {
    let wide_values = ApproxPolynomial::interpolate_newton_with_options(
        vec![-1.0_f64, 1.0],
        vec![-f64::MAX, f64::MAX],
        policy(),
    )
    .unwrap();
    assert!((wide_values.basis().coefficients()[1] / f64::MAX - 1.0).abs() <= f64::EPSILON);

    // The first difference on [0, MIN_POSITIVE] overflows, but the final
    // coefficient is 1/MIN_POSITIVE after division by the full node span.
    let polynomial = ApproxPolynomial::interpolate_newton_with_options(
        vec![-f64::MAX, 0.0, f64::MIN_POSITIVE],
        vec![0.0, 0.0, f64::MAX],
        policy(),
    )
    .unwrap();
    assert_eq!(polynomial.basis().coefficients()[..2], [0.0, 0.0]);
    assert!(
        (polynomial.basis().coefficients()[2] * f64::MIN_POSITIVE - 1.0).abs()
            <= 8.0 * f64::EPSILON
    );

    // Promotion after the last first-order entry has already been committed
    // must resume at the failed entry, without applying that update twice.
    let polynomial = ApproxPolynomial::interpolate_newton_with_options(
        vec![-f64::MAX, 0.0, f64::MIN_POSITIVE, 1.0],
        vec![0.0, 0.0, f64::MAX, 0.0],
        policy(),
    )
    .unwrap();
    let coefficients = polynomial.basis().coefficients();
    assert!((coefficients[2] * f64::MIN_POSITIVE - 1.0).abs() <= 8.0 * f64::EPSILON);
    assert!((coefficients[3] * f64::MIN_POSITIVE + 1.0).abs() <= 8.0 * f64::EPSILON);
}

#[test]
fn native_rank_thresholds_preserve_scaled_classification_boundaries() {
    fn check<R: RealScalar>() {
        for scale in [
            R::ZERO,
            R::MIN_POSITIVE * R::EPSILON,
            R::MIN_POSITIVE,
            R::EPSILON,
            R::ONE,
            R::MAX,
        ] {
            for dimension in [0, 1, 3, 16, 257] {
                let expected = Scaled::from_value(scale)
                    .unwrap()
                    .mul(Scaled::from_value(R::try_from_usize_exact(dimension).unwrap()).unwrap())
                    .unwrap()
                    .mul(Scaled::from_value(R::EPSILON).unwrap())
                    .unwrap()
                    .round()
                    .unwrap();
                let actual =
                    rank_threshold(RankPolicy::DimensionScaledEpsilon, scale, dimension).unwrap();
                assert!((actual - expected).abs() <= R::EPSILON * expected.abs());
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

fn newton_envelope<C: NumericScalar>() {
    let real = |x| C::Real::try_from_f64_lossy(x).unwrap();
    let scalar = |x| {
        C::from_components(
            real(x),
            if C::IS_REAL {
                C::Real::ZERO
            } else {
                real(x / 4.0)
            },
        )
    };
    for count in 1..=19 {
        let coefficients: Vec<_> = (0..count)
            .map(|index| scalar(1.0 / (index + 1) as f64))
            .collect();
        let nodes: Vec<_> = (0..count - 1)
            .map(|index| scalar(index as f64 / 16.0))
            .collect();
        let polynomial = ApproxPolynomial::newton(coefficients.clone(), nodes.clone()).unwrap();
        let x = scalar(0.375);
        // Independent forward products of the Newton terms and differentiated
        // products also provide the absolute envelope used by the report.
        let mut product = C::ONE;
        let mut slope = C::ZERO;
        let mut value = C::ZERO;
        let mut derivative = C::ZERO;
        let mut envelope = C::Real::ZERO;
        for (index, &coefficient) in coefficients.iter().enumerate() {
            value = value + coefficient * product;
            derivative = derivative + coefficient * slope;
            envelope += coefficient.abs() * product.abs();
            if let Some(&node) = nodes.get(index) {
                slope = slope * (x - node) + product;
                product = product * (x - node);
            }
        }
        let evaluation = polynomial.evaluate_with_derivative(x).unwrap();
        let tolerance = real(256.0) * C::Real::EPSILON;
        assert!((evaluation.value() - value).abs() <= tolerance * value.abs().max(C::Real::ONE));
        assert!(
            (evaluation.derivative() - derivative).abs()
                <= tolerance * derivative.abs().max(C::Real::ONE)
        );
        let expected_condition = (envelope / value.abs()).max(C::Real::ONE);
        assert!(
            (evaluation.condition().condition_estimate() - expected_condition).abs()
                <= tolerance * expected_condition
        );
    }
}

#[test]
fn fused_newton_reports_match_independent_terms_at_block_boundaries() {
    newton_envelope::<f32>();
    newton_envelope::<f64>();
    newton_envelope::<Complex<f32>>();
    newton_envelope::<Complex<f64>>();
}

#[test]
fn newton_extended_evaluation_preserves_wide_gaps_and_tiny_products() {
    let wide =
        ApproxPolynomial::newton_with_options(vec![0.0, 0.25], vec![-f64::MAX], policy()).unwrap();
    let evaluation = wide.evaluate_with_derivative(f64::MAX).unwrap();
    assert!((evaluation.value() / f64::MAX - 0.5).abs() <= f64::EPSILON);
    assert_eq!(evaluation.derivative(), 0.25);
    assert_eq!(evaluation.condition().condition_estimate(), 1.0);

    let tiny = f64::from_bits(1);
    let polynomial = ApproxPolynomial::newton_with_options(
        vec![0.0, 0.0, f64::MIN_POSITIVE],
        vec![-f64::MAX, 0.0],
        policy(),
    )
    .unwrap();
    let evaluation = polynomial.evaluate_with_derivative(tiny).unwrap();
    assert_eq!(evaluation.value() / tiny, 4.0);
    assert!((evaluation.derivative() - 4.0).abs() <= 4.0 * f64::EPSILON);
    assert!(wide.evaluate_with_derivative(f64::INFINITY).is_err());
    let overflow =
        ApproxPolynomial::newton_with_options(vec![0.0, f64::MAX], vec![0.0], policy()).unwrap();
    assert!(overflow.evaluate_with_derivative(2.0).is_err());
}

#[test]
fn newton_block_range_boundaries_preserve_all_terms() {
    // Exact dyadic powers independently exercise twelve unchecked steps at
    // both ends of the admitted coefficient interval before extended restart.
    for (leading, node, point, value, derivative) in [
        (
            2.0_f64.powi(-128),
            2.0_f64.powi(-16) + 2.0_f64.powi(-68),
            2.0_f64.powi(-16),
            2.0_f64.powi(-944),
            -12.0 * 2.0_f64.powi(-876),
        ),
        (
            2.0_f64.powi(768),
            -65536.0,
            65536.0,
            2.0_f64.powi(972),
            12.0 * 2.0_f64.powi(955),
        ),
    ] {
        let mut coefficients = vec![0.0; 13];
        coefficients[12] = leading;
        let polynomial =
            ApproxPolynomial::newton_with_options(coefficients, vec![node; 12], policy()).unwrap();
        let evaluation = polynomial.evaluate_with_derivative(point).unwrap();
        assert!((evaluation.value() / value - 1.0).abs() <= 8.0 * f64::EPSILON);
        assert!((evaluation.derivative() / derivative - 1.0).abs() <= 8.0 * f64::EPSILON);
        assert_eq!(evaluation.condition().condition_estimate(), 1.0);
    }
}

#[test]
fn explicit_basis_conversions_preserve_evaluation() {
    let representations = [
        ApproxPolynomial::newton_with_options(vec![1.0, 5.0, 3.0], vec![0.0, 1.0], policy())
            .unwrap()
            .to_monomial()
            .unwrap(),
        ApproxPolynomial::orthogonal_with_plan(
            vec![2.5, 2.0, 1.5],
            crate::OrthogonalBasisSpec::new(
                crate::OrthogonalFamily::ChebyshevT,
                crate::BasisNormalization::Classical,
                0.0,
                1.0,
            )
            .unwrap()
            .prepare(2, Default::default())
            .unwrap(),
            policy(),
        )
        .unwrap()
        .to_monomial()
        .unwrap(),
        ApproxPolynomial::orthogonal_with_plan(
            vec![2.0, 2.0, 2.0],
            crate::OrthogonalBasisSpec::new(
                crate::OrthogonalFamily::Legendre,
                crate::BasisNormalization::Classical,
                0.0,
                1.0,
            )
            .unwrap()
            .prepare(2, Default::default())
            .unwrap(),
            policy(),
        )
        .unwrap()
        .to_monomial()
        .unwrap(),
        ApproxPolynomial::bernstein_with_options(vec![1.0, 2.0, 6.0], 0.0, 1.0, policy())
            .unwrap()
            .to_monomial()
            .unwrap(),
    ];
    representations
        .iter()
        .enumerate()
        .for_each(|(index, conversion)| {
            let evaluation = conversion
                .polynomial()
                .evaluate_with_derivative(0.25)
                .unwrap();
            let expected = 1.0 + 2.0 * 0.25 + 3.0 * 0.25 * 0.25;
            assert!(
                f64::abs(evaluation.value() - expected) <= 1.0e-12,
                "basis {index} value {} expected {expected}",
                evaluation.value()
            );
            assert!(conversion.condition().condition_estimate().is_finite());
        });
}

fn assert_conversion_round_trip<B: PolynomialBasis<f64>>(
    source: &ApproxPolynomial<f64, MonomialBasis<f64>>,
    conversion: PolynomialConversion<f64, B>,
) {
    assert!(conversion.condition().condition_estimate().is_finite());
    let round_trip = conversion.polynomial().to_monomial().unwrap();
    for point in [-0.75_f64, 0.25, 1.5] {
        let expected = source.evaluate(point).unwrap();
        let native = conversion.polynomial().evaluate(point).unwrap();
        let restored = round_trip.polynomial().evaluate(point).unwrap();
        assert!((native - expected).abs() <= 2.0e-12);
        assert!((restored - expected).abs() <= 2.0e-12);
    }
}

#[test]
fn monomial_conversions_round_trip_through_every_prepared_basis() {
    let source =
        ApproxPolynomial::monomial_with_options(vec![1.0_f64, 2.0, 3.0], policy()).unwrap();
    assert_conversion_round_trip(&source, source.to_newton(vec![0.0_f64, 1.0, 2.0]).unwrap());
    assert_conversion_round_trip(
        &source,
        source.to_barycentric(vec![0.0_f64, 1.0, 2.0]).unwrap(),
    );
    assert_conversion_round_trip(
        &source,
        source
            .to_orthogonal(
                crate::OrthogonalBasisSpec::new(
                    crate::OrthogonalFamily::ChebyshevT,
                    crate::BasisNormalization::Classical,
                    0.0,
                    1.0,
                )
                .unwrap(),
            )
            .unwrap(),
    );
    assert_conversion_round_trip(
        &source,
        source
            .to_orthogonal(
                crate::OrthogonalBasisSpec::new(
                    crate::OrthogonalFamily::Legendre,
                    crate::BasisNormalization::Classical,
                    0.0,
                    1.0,
                )
                .unwrap(),
            )
            .unwrap(),
    );
    assert_conversion_round_trip(&source, source.to_bernstein(0.0, 1.0).unwrap());
}

#[test]
fn complex_coefficients_are_first_class() {
    let polynomial = ApproxPolynomial::monomial_with_options(
        vec![Complex::new(1.0, -1.0), Complex::new(2.0, 0.5)],
        policy(),
    )
    .unwrap();
    let point = Complex::new(0.25, -0.5);
    let evaluation = polynomial.evaluate_with_derivative(point).unwrap();
    assert_eq!(evaluation.derivative(), Complex::new(2.0, 0.5));
    assert!(
        (evaluation.value() - (Complex::new(1.0, -1.0) + Complex::new(2.0, 0.5) * point)).abs()
            < 1.0e-14
    );
}

#[test]
fn balanced_root_construction_is_permutation_invariant() {
    let roots = [-8.0_f64, -0.5, 0.25, 2.0, 16.0, 32.0];
    let ordered = ApproxPolynomial::from_roots_with_options(&roots, policy()).unwrap();
    for offset in 0..roots.len() {
        let permutation = roots
            .iter()
            .cycle()
            .skip(offset)
            .take(roots.len())
            .copied()
            .collect::<Vec<_>>();
        let rotated = ApproxPolynomial::from_roots_with_options(&permutation, policy()).unwrap();
        assert_eq!(
            ordered.basis().coefficients(),
            rotated.basis().coefficients()
        );

        let reversed = permutation.into_iter().rev().collect::<Vec<_>>();
        let reversed = ApproxPolynomial::from_roots_with_options(&reversed, policy()).unwrap();
        assert_eq!(
            ordered.basis().coefficients(),
            reversed.basis().coefficients()
        );
    }
    for root in roots {
        let evaluation = ordered.evaluate_with_derivative(root).unwrap();
        assert!(evaluation.condition().backward_error().is_finite());
        assert!(evaluation.value().abs() <= 1.0e-7);
    }
}

#[test]
fn conjugate_pair_batching_produces_real_coefficients() {
    let polynomial = ApproxPolynomial::from_conjugate_pairs_with_options(
        &[3.0_f64],
        &[Complex::new(2.0, 5.0)],
        policy(),
    )
    .unwrap();
    assert_eq!(polynomial.basis().coefficients(), &[-87.0, 41.0, -7.0, 1.0]);
    let complex = ApproxPolynomial::from_roots_with_options(
        &[
            Complex::new(3.0, 0.0),
            Complex::new(2.0, 5.0),
            Complex::new(2.0, -5.0),
        ],
        policy(),
    )
    .unwrap();
    complex
        .basis()
        .coefficients()
        .iter()
        .for_each(|coefficient| assert!(f64::abs(coefficient.im) <= 1.0e-14));
}

#[test]
#[expect(
    clippy::approx_constant,
    reason = "reference roots are the independently computed literals, not the constants"
)]
fn companion_and_colleague_matrices_have_independently_validated_roots() {
    let monomial =
        ApproxPolynomial::monomial_with_options(vec![-2.0_f64, 0.0, 1.0], policy()).unwrap();
    let scaled = ApproxPolynomial::monomial_with_options(
        vec![-2.0_f64 * 2.0_f64.powi(500), 0.0, 2.0_f64.powi(500)],
        policy(),
    )
    .unwrap();
    let companion = monomial.root_matrix().unwrap();
    let scaled_companion = scaled.root_matrix().unwrap();
    assert_eq!(companion, scaled_companion);

    let chebyshev = ApproxPolynomial::orthogonal_with_plan(
        vec![0.0_f64, 0.0, 1.0],
        crate::OrthogonalBasisSpec::new(
            crate::OrthogonalFamily::ChebyshevT,
            crate::BasisNormalization::Classical,
            0.0,
            1.0,
        )
        .unwrap()
        .prepare(2, Default::default())
        .unwrap(),
        policy(),
    )
    .unwrap();
    let legendre = ApproxPolynomial::orthogonal_with_plan(
        vec![0.0_f64, 0.0, 1.0],
        crate::OrthogonalBasisSpec::new(
            crate::OrthogonalFamily::Legendre,
            crate::BasisNormalization::Classical,
            0.0,
            1.0,
        )
        .unwrap()
        .prepare(2, Default::default())
        .unwrap(),
        policy(),
    )
    .unwrap();
    let fixtures = [
        (
            companion,
            1.414_213_562_373_095_1_f64,
            PolynomialRootBasisRef::Monomial,
        ),
        (
            chebyshev.root_matrix().unwrap(),
            0.707_106_781_186_547_6_f64,
            PolynomialRootBasisRef::Orthogonal(chebyshev.basis().plan()),
        ),
        (
            legendre.root_matrix().unwrap(),
            0.577_350_269_189_625_7_f64,
            PolynomialRootBasisRef::Orthogonal(legendre.basis().plan()),
        ),
    ];
    fixtures
        .into_iter()
        .for_each(|(linearization, expected, basis)| {
            let roots =
                scioxide_linalg::GeneralEigenvaluesResult::<f64>::compute(&linearization).unwrap();
            assert_eq!(roots.eigenvalues().len(), 2);
            roots.eigenvalues().iter().for_each(|root| {
                assert!(f64::abs(root.im) <= 1.0e-14);
                assert!(f64::abs(f64::abs(root.re) - expected) <= 1.0e-12);
                let coefficients = match basis {
                    PolynomialRootBasisRef::Monomial => monomial.basis().coefficients(),
                    PolynomialRootBasisRef::Orthogonal(plan)
                        if plan.spec().family() == crate::OrthogonalFamily::ChebyshevT =>
                    {
                        chebyshev.basis().coefficients()
                    }
                    _ => legendre.basis().coefficients(),
                };
                let residual = crate::checked_basis_evaluation(coefficients, *root, basis).unwrap();
                assert!(residual.value().unwrap().abs() <= 1.0e-12);
            });
        });
}

#[test]
fn rounding_model_checks_its_dimension_boundary_without_large_storage() {
    let limit = 1usize << 23;
    assert!(rounding_error::<f32>(limit - 1).unwrap().is_finite());
    // Exact powers of two, an inexact f32 count above 2^24 and a saturated
    // operation count all share one classification.
    for size in [limit, limit * 2, (1usize << 24) + 1, usize::MAX] {
        assert!(matches!(
            rounding_error::<f32>(size).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::IllConditioned,
                ..
            }
        ));
    }
}
