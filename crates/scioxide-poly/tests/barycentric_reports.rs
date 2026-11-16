//! Barycentric polynomial invariants and operation-specific diagnostic models.
#![allow(clippy::unwrap_used)]

#[path = "support/tolerance.rs"]
mod tolerance;
use tolerance::assert_close;

use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideErrorKind, Tolerance};
use scioxide_linalg::RankPolicy;
use scioxide_poly::PolynomialBasis;
use scioxide_poly::{
    ApproxPolynomial, PolynomialApproximationPolicy, PolynomialConditionReport,
    polynomial_composition_condition, polynomial_product_condition,
};

// Direct Lagrange products and their differentiated products are independent of
// barycentric weights, Newton differences and monomial conversion.
fn lagrange<C: NumericScalar>(nodes: &[C], values: &[C], x: C) -> (C, C) {
    nodes
        .iter()
        .enumerate()
        .fold((C::ZERO, C::ZERO), |(value, derivative), (i, &node)| {
            let basis = nodes
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .fold(C::ONE, |product, (_, &other)| {
                    product * (x - other) / (node - other)
                });
            let slope = nodes.iter().enumerate().filter(|(j, _)| *j != i).fold(
                C::ZERO,
                |sum, (j, &other)| {
                    sum + nodes
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| *k != i && *k != j)
                        .fold(C::ONE / (node - other), |product, (_, &third)| {
                            product * (x - third) / (node - third)
                        })
                },
            );
            (value + values[i] * basis, derivative + values[i] * slope)
        })
}

fn interpolation<R: RealScalar>() {
    let cast = |x| R::try_from_f64_lossy(x).unwrap();
    for nodes in [
        vec![cast(-1.0), cast(-0.5), R::ZERO, cast(0.5), R::ONE],
        (0..5)
            .map(|i| (cast(i as f64) * R::PI / cast(4.0)).cos())
            .collect(),
        vec![R::ZERO, cast(0.05), cast(0.1), cast(0.5), R::ONE],
    ] {
        let values: Vec<_> = nodes.iter().map(|&x| R::ONE + x + x * x).collect();
        let polynomial =
            ApproxPolynomial::interpolate_barycentric(nodes.clone(), values.clone()).unwrap();
        let monomial = polynomial.to_monomial().unwrap().into_polynomial();
        let newton = monomial
            .to_newton(nodes[..=monomial.degree()].to_vec())
            .unwrap()
            .into_polynomial();
        for x in [cast(-0.25), cast(0.25), cast(0.75)]
            .into_iter()
            .chain(nodes)
        {
            let (value, derivative) = lagrange(polynomial.basis().nodes(), &values, x);
            for evaluation in [
                polynomial.evaluate_with_derivative(x).unwrap(),
                monomial.evaluate_with_derivative(x).unwrap(),
                newton.evaluate_with_derivative(x).unwrap(),
            ] {
                assert_close(evaluation.value(), value, 512);
                assert_close(evaluation.derivative(), derivative, 512);
            }
        }
    }
}

#[test]
fn interpolation_and_conversions_agree_with_independent_lagrange() {
    interpolation::<f32>();
    interpolation::<f64>();
    let source = ApproxPolynomial::monomial(vec![0.0_f64, 0.0, 1.0]).unwrap();
    let converted = source.to_barycentric(vec![0.0, 1.0, 2.0]).unwrap();
    assert_close(converted.polynomial().evaluate(0.5).unwrap(), 0.25, 16);
    assert_eq!(converted.polynomial().basis().weights(), &[0.5, -1.0, 0.5]);
}

#[test]
fn complex_nodes_use_derived_weights_and_preserve_the_polynomial() {
    let nodes = vec![
        Complex::new(-1.0_f64, 0.5),
        Complex::new(0.0, -0.5),
        Complex::new(1.0, 1.0),
    ];
    let source = ApproxPolynomial::monomial(vec![
        Complex::new(1.0, 0.5),
        Complex::new(-2.0, 1.0),
        Complex::new(0.5, -1.0),
    ])
    .unwrap();
    let converted = source.to_barycentric(nodes.clone()).unwrap();
    let polynomial = converted.polynomial();
    let monomial = polynomial.to_monomial().unwrap().into_polynomial();
    for x in nodes.into_iter().chain([Complex::new(0.25, 0.75)]) {
        let (value, derivative) = lagrange(
            polynomial.basis().nodes(),
            polynomial.basis().coefficients(),
            x,
        );
        for p in [
            polynomial.evaluate_with_derivative(x).unwrap(),
            monomial.evaluate_with_derivative(x).unwrap(),
        ] {
            assert_close(p.value(), value, 128);
            assert_close(p.derivative(), derivative, 128);
        }
    }
}

fn near_node<R: RealScalar>(tiny: R) {
    let polynomial = ApproxPolynomial::interpolate_barycentric(
        vec![R::ZERO, R::ONE, R::TWO],
        vec![R::ZERO, R::TWO, R::TWO + R::TWO],
    )
    .unwrap();
    for x in [R::ZERO, -R::ZERO, tiny, -tiny, R::ONE, R::TWO] {
        let evaluation = polynomial.evaluate_with_derivative(x).unwrap();
        if x != R::ZERO {
            assert_close(evaluation.value() / x, R::TWO, 16);
        } else {
            assert_eq!(evaluation.value(), R::ZERO);
        }
        assert_close(evaluation.derivative(), R::TWO, 16);
    }
    let constant =
        ApproxPolynomial::interpolate_barycentric(vec![R::ZERO, R::ONE], vec![R::ONE, R::ONE])
            .unwrap();
    let result = constant.evaluate_with_derivative(tiny).unwrap();
    assert_eq!(result.value(), R::ONE);
    assert_eq!(result.derivative(), R::ZERO);
}

#[test]
fn exact_and_subnormal_node_distances_are_range_safe_without_allocation() {
    near_node(f64::from_bits(1));
    near_node(1e-310_f64);
    near_node(f32::from_bits(1));
    let tiny = f64::from_bits(1);
    let complex = ApproxPolynomial::interpolate_barycentric(
        vec![Complex::new(0.0_f64, 0.0), Complex::new(1.0, 0.0)],
        vec![Complex::new(0.0, 0.0), Complex::new(2.0, 0.0)],
    )
    .unwrap();
    let evaluation = complex
        .evaluate_with_derivative(Complex::new(0.0, tiny))
        .unwrap();
    assert_close(evaluation.value().im / tiny, 2.0, 16);
    assert_close(evaluation.derivative(), Complex::new(2.0, 0.0), 16);
    let p = ApproxPolynomial::interpolate_barycentric(
        vec![-f64::MAX, 0.0, f64::MAX],
        vec![-1.0, 0.0, 1.0],
    )
    .unwrap();
    let e = p.evaluate_with_derivative(f64::MAX / 2.0).unwrap();
    assert_close(e.value(), 0.5, 16);
    assert_close(e.derivative() * f64::MAX, 1.0, 16);
}

#[test]
fn invalid_samples_and_unrepresentable_weights_are_structured_errors() {
    assert_eq!(
        ApproxPolynomial::interpolate_barycentric(vec![0.0_f64], vec![])
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    for (nodes, values) in [(vec![], vec![]), (vec![-0.0, 0.0], vec![1.0, 2.0])] {
        assert_eq!(
            ApproxPolynomial::interpolate_barycentric(nodes, values)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::InvalidInput
        );
    }
    for (nodes, values) in [
        (vec![f64::NAN], vec![1.0]),
        (vec![0.0], vec![f64::INFINITY]),
    ] {
        assert_eq!(
            ApproxPolynomial::interpolate_barycentric(nodes, values)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::NonFinite
        );
    }
    assert_eq!(
        ApproxPolynomial::interpolate_barycentric(
            vec![0.0, f64::from_bits(1), f64::MAX],
            vec![0.0; 3]
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::NumericalInstability
    );
    let singleton = ApproxPolynomial::interpolate_barycentric(vec![0.0_f64], vec![3.0]).unwrap();
    assert_eq!(
        singleton
            .evaluate_with_derivative(1.0)
            .unwrap()
            .derivative(),
        0.0
    );
}

fn gamma(n: usize) -> f64 {
    let x = n as f64 * f64::EPSILON;
    x / (1.0 - x)
}
fn rounding_report(report: PolynomialConditionReport<f64>, n: usize) {
    assert_close(report.backward_error() / gamma(n), 1.0, 8);
    assert_close(
        report.forward_error_estimate() / gamma(n),
        report.condition_estimate(),
        8,
    );
}

#[test]
fn each_operation_separates_backward_error_from_forward_amplification() {
    let source = ApproxPolynomial::monomial(vec![-1.0_f64, 0.0, 1.0]).unwrap();
    let root = source.evaluate_with_derivative(1.0).unwrap().condition();
    assert_close(root.backward_error() / gamma(3), 1.0, 8);
    assert_eq!(root.condition_estimate(), f64::INFINITY);
    assert_eq!(root.forward_error_estimate(), f64::INFINITY);
    rounding_report(source.evaluate_with_derivative(0.5).unwrap().condition(), 3);
    for conversion in [
        source.to_monomial().unwrap().condition(),
        source
            .to_barycentric(vec![0.0, 1.0, 2.0])
            .unwrap()
            .condition(),
        source.to_bernstein(0.0, 1.0).unwrap().condition(),
        source.to_newton(vec![0.0, 1.0, 2.0]).unwrap().condition(),
    ] {
        rounding_report(conversion, 3);
    }
    rounding_report(
        polynomial_product_condition(&[1.0, 1.0], &[1.0, -1.0], &[1.0, 0.0, -1.0]).unwrap(),
        3,
    );
    rounding_report(
        polynomial_composition_condition(&[1.0, -1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0, 1.0]).unwrap(),
        3,
    );
    let divisor = ApproxPolynomial::monomial(vec![-1.0, 1.0]).unwrap();
    let division = source.div_rem(&divisor).unwrap();
    assert_eq!(division.condition().backward_error(), 0.0);
    assert_close(
        division.condition().forward_error_estimate() / gamma(3),
        division.condition().condition_estimate(),
        8,
    );
    let zero = ApproxPolynomial::monomial(vec![0.0_f64])
        .unwrap()
        .evaluate_with_derivative(0.0)
        .unwrap()
        .condition();
    rounding_report(zero, 1);
    assert_eq!(zero.condition_estimate(), 1.0);
}

#[test]
fn overflowing_and_singular_diagnostics_remain_reports() {
    let tiny = f64::from_bits(1);
    let report = polynomial_product_condition(&[1.0_f64], &[1.0], &[tiny]).unwrap();
    assert_eq!(report.condition_estimate(), f64::INFINITY);
    assert_eq!(report.forward_error_estimate(), f64::INFINITY);
    assert_close(report.backward_error() / gamma(1), 1.0, 8);
    let singular = polynomial_composition_condition(&[1.0_f64, -1.0], &[1.0], &[0.0]).unwrap();
    assert_eq!(singular.condition_estimate(), f64::INFINITY);
    assert_eq!(singular.forward_error_estimate(), f64::INFINITY);
    let source = ApproxPolynomial::monomial(vec![0.0_f64, 0.0, tiny]).unwrap();
    let conversion = source.to_bernstein(0.0, 1e160).unwrap();
    assert_eq!(conversion.condition().condition_estimate(), f64::INFINITY);
    assert_eq!(
        conversion.condition().forward_error_estimate(),
        f64::INFINITY
    );
    assert_close(conversion.condition().backward_error() / gamma(3), 1.0, 8);
    let value = conversion.polynomial().evaluate(0.5e160).unwrap();
    assert_close(value / (tiny * 1e160 * 1e160), 0.25, 32);
}

#[test]
fn effective_degree_uses_degree_bearing_coordinates_without_pruning() {
    for (values, expected) in [
        (vec![1.0_f64, 0.0, 0.0], 2),
        (vec![1.0; 3], 0),
        (vec![0.0; 3], 0),
        (vec![0.0, 1.0, 2.0], 1),
    ] {
        let polynomial =
            ApproxPolynomial::interpolate_barycentric(vec![0.0, 1.0, 2.0], values).unwrap();
        assert_eq!(polynomial.degree(), 2);
        assert_eq!(
            polynomial
                .classify_coefficients()
                .unwrap()
                .effective_degree(),
            expected
        );
        assert_eq!(
            polynomial
                .to_monomial()
                .unwrap()
                .polynomial()
                .classify_coefficients()
                .unwrap()
                .effective_degree(),
            expected
        );
    }
    for (controls, expected) in [
        (vec![1.0_f64; 3], 0),
        (vec![0.0; 3], 0),
        (vec![0.0, 0.5, 1.0], 1),
        (vec![1.0, 0.0, 0.0], 2),
        (vec![3.0], 0),
    ] {
        let polynomial = ApproxPolynomial::bernstein(controls.clone(), -2.0, 3.0).unwrap();
        assert_eq!(polynomial.basis().coefficients(), controls);
        assert_eq!(
            polynomial
                .classify_coefficients()
                .unwrap()
                .effective_degree(),
            expected
        );
    }
    let policy =
        PolynomialApproximationPolicy::new(Tolerance::new(0.25, 0.0).unwrap(), RankPolicy::Exact);
    for (tail, expected) in [(0.25, 1), (0.5, 2)] {
        let polynomial =
            ApproxPolynomial::bernstein_with_options(vec![1.0, 1.5, 2.0 + tail], 0.0, 1.0, policy)
                .unwrap();
        assert_eq!(
            polynomial
                .classify_coefficients()
                .unwrap()
                .effective_degree(),
            expected
        );
    }
    for (tail, expected) in [(0.25, 1), (0.5, 2)] {
        let polynomial = ApproxPolynomial::interpolate_barycentric_with_options(
            vec![0.0_f64, 1.0, 2.0],
            vec![1.0, 2.0, 3.0 + 2.0 * tail],
            policy,
        )
        .unwrap();
        assert_eq!(
            polynomial
                .classify_coefficients()
                .unwrap()
                .effective_degree(),
            expected
        );
    }
    let tiny = f64::from_bits(1);
    let clustered = ApproxPolynomial::interpolate_barycentric(
        vec![0.0, tiny, 2.0 * tiny],
        vec![0.0, tiny, 2.0 * tiny],
    )
    .unwrap();
    let evaluation = clustered.evaluate_with_derivative(0.0).unwrap();
    assert_close(evaluation.derivative(), 1.0, 16);
    assert_eq!(
        clustered
            .classify_coefficients()
            .unwrap()
            .effective_degree(),
        1
    );
    // Classification must not require unrepresentable physical power coefficients.
    let narrow = ApproxPolynomial::bernstein(vec![0.0, 0.0, 1.0], 0.0, f64::from_bits(1)).unwrap();
    assert_eq!(narrow.degree(), 2);
    let complex =
        ApproxPolynomial::bernstein(vec![Complex::new(1.0_f64, 2.0); 3], 0.0, 1.0).unwrap();
    assert_eq!(
        complex.classify_coefficients().unwrap().effective_degree(),
        0
    );
}

#[test]
fn explicit_classification_retains_extended_newton_coefficients_and_immutable_samples() {
    // The quadratic Newton coefficient is 5e349, while node values and the
    // derivative at zero are representable. Classification must not project it.
    let nodes = vec![0.0_f64, 1e-200, 2e-200];
    let values = vec![0.0, 0.0, 1e-50];
    let polynomial =
        ApproxPolynomial::interpolate_barycentric(nodes.clone(), values.clone()).unwrap();
    let original = polynomial.clone();
    let first = polynomial.classify_coefficients().unwrap();
    assert_eq!(first.effective_degree(), 2);
    assert_eq!(first.effective_rank(), 1);
    assert_eq!(first, polynomial.classify_coefficients().unwrap());
    assert_eq!(polynomial, original);
    assert_eq!(polynomial.basis().nodes(), nodes);
    assert_eq!(polynomial.basis().coefficients(), values);
    let evaluation = polynomial.evaluate_with_derivative(0.0).unwrap();
    assert_eq!(evaluation.value(), 0.0);
    assert_close(evaluation.derivative() / -5e149, 1.0, 64);
}

/// The second form's rounding error grows with the Lebesgue function Λ(z).
/// Evaluation reports add |p(z)| Λ(z) to the sample envelope, so the forward
/// estimate covers clustered nodes while well-separated nodes stay modest.
#[test]
fn lebesgue_term_bounds_the_barycentric_forward_error() {
    let delta = 1e-12_f64;
    let clustered = ApproxPolynomial::interpolate_barycentric(
        vec![-1.0, 0.0, delta, 1.0],
        vec![0.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let evaluation = clustered.evaluate_with_derivative(0.5).unwrap();
    // p is the last Lagrange basis polynomial x(x+1)(x-δ)/(2(1-δ)).
    let exact = 0.375 * (0.5 - delta) / (1.0 - delta);
    let error = (evaluation.value() - exact).abs() / exact;
    let report = evaluation.condition();
    assert!(error > 1e3 * f64::EPSILON, "clustered nodes lose accuracy");
    assert!(
        error <= report.forward_error_estimate(),
        "{error} > {}",
        report.forward_error_estimate()
    );
    assert!(report.condition_estimate() > 1e11);

    // Chebyshev extrema: Λ grows only logarithmically, and Λ = 1 at a node.
    let n = 17;
    let nodes: Vec<f64> = (0..n)
        .map(|j| -(core::f64::consts::PI * j as f64 / (n - 1) as f64).cos())
        .collect();
    let values: Vec<f64> = nodes.iter().map(|&x| x.exp()).collect();
    let smooth = ApproxPolynomial::interpolate_barycentric(nodes.clone(), values).unwrap();
    for point in [-0.99, -0.3, 0.123, 0.77] {
        let report = smooth.evaluate_with_derivative(point).unwrap().condition();
        assert!(report.condition_estimate() < 8.0, "{point}");
    }
    let at_node = smooth
        .evaluate_with_derivative(nodes[5])
        .unwrap()
        .condition();
    assert_close(at_node.condition_estimate(), 2.0, 4);
}

/// The Lebesgue term never makes a representable evaluation fail: neither the
/// condition nor root diagnostics form `Σ|ℓ_j f_j| + |p| Λ` natively.
#[test]
fn near_overflow_samples_keep_finite_reports_and_root_diagnostics() {
    let big = 0.6 * f64::MAX;
    let nodes = vec![0.0_f64, 1.0, 2.0];
    let constant = ApproxPolynomial::interpolate_barycentric(nodes.clone(), vec![big; 3]).unwrap();
    // Λ = 1 at a node; Λ(0.5) = 0.375 + 0.75 + 0.125 for these nodes.
    for (point, condition) in [(1.0, 2.0), (0.5, 2.5)] {
        let evaluation = constant.evaluate_with_derivative(point).unwrap();
        assert_close(evaluation.value(), big, 8);
        assert_close(evaluation.condition().condition_estimate(), condition, 8);
        let root = constant.root_condition(point).unwrap();
        assert_close(root.coefficient().value().unwrap(), condition, 8);
    }
    let ramp = ApproxPolynomial::interpolate_barycentric(
        nodes.clone(),
        vec![0.1 * f64::MAX, 0.5 * f64::MAX, 0.9 * f64::MAX],
    )
    .unwrap();
    for point in [1.5, 2.0] {
        let evaluation = ramp.evaluate_with_derivative(point).unwrap();
        assert!(evaluation.value().is_finite());
        assert!(evaluation.condition().condition_estimate().is_finite());
        ramp.root_condition(point).unwrap();
    }
    let complex = ApproxPolynomial::interpolate_barycentric(
        nodes.into_iter().map(|x| Complex::new(x, 0.0)).collect(),
        vec![Complex::new(big, big / 2.0); 3],
    )
    .unwrap();
    let evaluation = complex
        .evaluate_with_derivative(Complex::new(1.0, 0.0))
        .unwrap();
    assert_eq!(evaluation.value(), Complex::new(big, big / 2.0));
    assert_close(evaluation.condition().condition_estimate(), 2.0, 8);
}
