//! Independent exact-dyadic coefficient references and root-order regressions.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    Complex, NonFiniteOrigin, NumericScalar, SciOxideErrorKind, SciOxideErrorView,
};
use scioxide_poly::PolynomialBasis;
use scioxide_poly::{ApproxPolynomial, PolynomialApproximationPolicy};

fn sequential<C: NumericScalar>(roots: &[C]) -> Vec<C> {
    let mut coefficients = vec![C::ONE];
    for &root in roots {
        let mut next = vec![C::ZERO; coefficients.len() + 1];
        for (k, &coefficient) in coefficients.iter().enumerate() {
            next[k] = next[k] - root * coefficient;
            next[k + 1] = next[k + 1] + coefficient;
        }
        coefficients = next;
    }
    coefficients
}

fn error(actual: &[Complex<f64>], expected: &[Complex<f64>]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn check_case(family: &str, roots: &[Complex<f64>], expected: &[Complex<f64>]) {
    let policy = PolynomialApproximationPolicy::default();
    let result = ApproxPolynomial::from_roots_with_options(roots, policy).unwrap();
    let baseline_error = error(&sequential(roots), expected);
    let scale = expected.iter().map(|c| c.abs()).fold(0.0, f64::max);
    let tolerance = 16.0 * baseline_error.max(f64::EPSILON * scale) + f64::from_bits(1);
    let actual_error = error(result.basis().coefficients(), expected);
    assert!(
        actual_error <= tolerance,
        "{family}/{}: tree {actual_error:e}, sequential {baseline_error:e}, tolerance {tolerance:e}",
        roots.len()
    );

    let mut permuted = roots.to_vec();
    permuted.reverse();
    permuted.rotate_left(roots.len() / 3);
    assert_eq!(
        result.basis().coefficients(),
        ApproxPolynomial::from_roots_with_options(&permuted, policy)
            .unwrap()
            .basis()
            .coefficients()
    );

    if matches!(family, "mixed" | "clustered") {
        let real: Vec<_> = roots.iter().map(|root| root.re).collect();
        let result = ApproxPolynomial::from_roots_with_options(&real, policy).unwrap();
        let actual: Vec<_> = result
            .basis()
            .coefficients()
            .iter()
            .map(|&value| Complex::new(value, 0.0))
            .collect();
        assert!(
            error(&actual, expected) <= tolerance,
            "real {family}/{}",
            roots.len()
        );
    }

    if roots.len() <= 64 {
        let roots32: Vec<_> = roots
            .iter()
            .map(|root| Complex::new(root.re as f32, root.im as f32))
            .collect();
        let result32 = ApproxPolynomial::from_roots(&roots32).unwrap();
        let widen = |coefficients: &[Complex<f32>]| {
            coefficients
                .iter()
                .map(|value| Complex::new(f64::from(value.re), f64::from(value.im)))
                .collect::<Vec<_>>()
        };
        let baseline32 = error(&widen(&sequential(&roots32)), expected);
        let tolerance32 = 16.0 * baseline32.max(f64::from(f32::EPSILON) * scale);
        assert!(
            error(&widen(result32.basis().coefficients()), expected) <= tolerance32,
            "f32 {family}/{}",
            roots.len()
        );
        if family == "conjugate" {
            let real: Vec<_> = roots32[..roots.len() / 2]
                .iter()
                .map(|root| root.re)
                .collect();
            let pairs: Vec<_> = roots32[roots.len() / 2..]
                .iter()
                .step_by(2)
                .copied()
                .collect();
            let result = ApproxPolynomial::from_conjugate_pairs(&real, &pairs).unwrap();
            let actual: Vec<_> = result
                .basis()
                .coefficients()
                .iter()
                .map(|&value| Complex::new(f64::from(value), 0.0))
                .collect();
            assert!(
                error(&actual, expected) <= tolerance32,
                "f32 conjugate/{}",
                roots.len()
            );
        }
    }

    if family == "conjugate" {
        let mut real: Vec<_> = roots[..roots.len() / 2]
            .iter()
            .map(|root| root.re)
            .collect();
        let mut pairs: Vec<_> = roots[roots.len() / 2..]
            .iter()
            .step_by(2)
            .copied()
            .collect();
        let result =
            ApproxPolynomial::from_conjugate_pairs_with_options(&real, &pairs, policy).unwrap();
        let actual: Vec<_> = result
            .basis()
            .coefficients()
            .iter()
            .map(|&value| Complex::new(value, 0.0))
            .collect();
        let actual_error = error(&actual, expected);
        assert!(
            actual_error <= tolerance,
            "conjugate/{}: {actual_error:e} > {tolerance:e}",
            roots.len()
        );
        real.reverse();
        pairs.rotate_left(1);
        pairs.reverse();
        assert_eq!(
            result.basis().coefficients(),
            ApproxPolynomial::from_conjugate_pairs_with_options(&real, &pairs, policy)
                .unwrap()
                .basis()
                .coefficients()
        );
    }
}

#[test]
fn root_products_match_exact_dyadic_references() {
    let mut rows = include_str!("data/root_products.csv")
        .lines()
        .skip(1)
        .peekable();
    while let Some(first) = rows.peek() {
        let header: Vec<_> = first.split(',').collect();
        let family = header[0];
        let degree = header[1].parse::<usize>().unwrap();
        let mut roots = Vec::with_capacity(degree);
        let mut expected = Vec::with_capacity(degree + 1);
        for index in 0..=degree {
            let fields: Vec<_> = rows.next().unwrap().split(',').collect();
            assert_eq!(fields[0], family);
            assert_eq!(fields[1].parse::<usize>().unwrap(), degree);
            assert_eq!(fields[2].parse::<usize>().unwrap(), index);
            if index < degree {
                roots.push(Complex::new(
                    fields[3].parse().unwrap(),
                    fields[4].parse().unwrap(),
                ));
            }
            expected.push(Complex::new(
                fields[5].parse().unwrap(),
                fields[6].parse().unwrap(),
            ));
        }
        check_case(family, &roots, &expected);
    }
}

#[test]
fn root_product_boundaries_and_invalid_inputs() {
    let policy = PolynomialApproximationPolicy::default();
    assert_eq!(
        ApproxPolynomial::<f64, _>::from_roots_with_options(&[], policy)
            .unwrap()
            .basis()
            .coefficients(),
        &[1.0]
    );
    assert_eq!(
        ApproxPolynomial::from_conjugate_pairs_with_options(&[], &[], policy)
            .unwrap()
            .basis()
            .coefficients(),
        &[1.0]
    );
    assert_eq!(
        ApproxPolynomial::from_roots_with_options(&[2.0, 2.0, 2.0], policy)
            .unwrap()
            .basis()
            .coefficients(),
        &[-8.0, 12.0, -6.0, 1.0]
    );
    assert_eq!(
        ApproxPolynomial::from_conjugate_pairs_with_options(
            &[1.0],
            &[Complex::new(0.0, 1.0), Complex::new(2.0, 1.0)],
            policy
        )
        .unwrap()
        .basis()
        .coefficients(),
        &[-5.0, 9.0, -10.0, 10.0, -5.0, 1.0]
    );
    // Finite roots can still have an unrepresentable final coefficient vector.
    assert_eq!(
        ApproxPolynomial::from_conjugate_pairs_with_options(
            &[0.0],
            &[Complex::new(f64::MAX, 1.0)],
            policy
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::NumericalInstability
    );
    for root in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            ApproxPolynomial::from_roots_with_options(&[root], policy)
                .unwrap_err()
                .view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
        assert!(matches!(
            ApproxPolynomial::from_conjugate_pairs_with_options(&[root], &[], policy)
                .unwrap_err()
                .view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
    }
    for root in [Complex::new(1.0, 0.0), Complex::new(1.0, -1.0)] {
        assert_eq!(
            ApproxPolynomial::from_conjugate_pairs_with_options(&[], &[root], policy)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::InvalidInput
        );
    }
    for root in [
        Complex::new(f64::NAN, 1.0),
        Complex::new(1.0, f64::INFINITY),
    ] {
        assert!(matches!(
            ApproxPolynomial::from_roots_with_options(&[root], policy)
                .unwrap_err()
                .view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
        assert!(matches!(
            ApproxPolynomial::from_conjugate_pairs_with_options(&[], &[root], policy)
                .unwrap_err()
                .view(),
            SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Input,
                ..
            }
        ));
    }
}

#[test]
fn conjugate_quadratics_retain_small_terms_until_final_projection() {
    let polynomial =
        ApproxPolynomial::from_conjugate_pairs(&[1.0e200_f64], &[Complex::new(1.0e-200, 1.0e-200)])
            .unwrap();
    for (&actual, expected) in
        polynomial
            .basis()
            .coefficients()
            .iter()
            .zip([-2.0e-200_f64, 2.0, -1.0e200, 1.0])
    {
        assert!((actual - expected).abs() <= 16.0 * f64::EPSILON * expected.abs());
    }
}

#[test]
fn odd_real32_products_preserve_permutations_and_coefficients() {
    let roots = [-0.75_f32, 0.5, -0.25, 0.125, 0.375];
    let policy = PolynomialApproximationPolicy::default();
    let polynomial = ApproxPolynomial::from_roots_with_options(&roots, policy).unwrap();
    let reference = sequential(&roots.map(|root| Complex::new(f64::from(root), 0.0)));
    for (&actual, expected) in polynomial.basis().coefficients().iter().zip(reference) {
        assert!((f64::from(actual) - expected.re).abs() <= 16.0 * f64::from(f32::EPSILON));
    }
    let mut reversed = roots;
    reversed.reverse();
    assert_eq!(
        polynomial.basis().coefficients(),
        ApproxPolynomial::from_roots_with_options(&reversed, policy)
            .unwrap()
            .basis()
            .coefficients()
    );
}
