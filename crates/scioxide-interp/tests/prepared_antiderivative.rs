//! Public prepared-antiderivative correctness, policy, and allocation regressions.

#![allow(clippy::unwrap_used)]

mod support;

use support::{assert_close, assert_error};

use scioxide_core::SciOxideError;
use scioxide_interp::{
    BarycentricInterpolator, ExtrapolationPolicy, HermiteInterpolator, Interpolant,
    NewtonInterpolator,
};

fn evaluate_power(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |value, &coefficient| value.mul_add(x, coefficient))
}

fn evaluate_power_derivative(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .enumerate()
        .skip(1)
        .rev()
        .fold(0.0, |value, (degree, &coefficient)| {
            value.mul_add(x, coefficient * degree as f64)
        })
}

fn exact_power_integral(coefficients: &[f64], a: f64, b: f64) -> f64 {
    coefficients
        .iter()
        .enumerate()
        .map(|(degree, &coefficient)| {
            let integrated_degree = i32::try_from(degree + 1).unwrap();
            coefficient * (b.powi(integrated_degree) - a.powi(integrated_degree))
                / f64::from(integrated_degree)
        })
        .sum()
}

fn chebyshev_lobatto_nodes(count: usize, lower: f64, upper: f64) -> Vec<f64> {
    (0..count)
        .map(|index| {
            let angle = std::f64::consts::PI * index as f64 / (count - 1) as f64;
            lower + (upper - lower) * (1.0 - angle.cos()) * 0.5
        })
        .collect()
}

#[test]
fn all_global_interpolators_prepare_the_same_high_degree_integral() {
    let coefficients = [
        0.75, -0.5, 0.25, -0.125, 0.0625, -0.03125, 0.015625, -0.0078125,
    ];
    let nodes = chebyshev_lobatto_nodes(coefficients.len(), -1.0, 1.0);
    let values = nodes
        .iter()
        .map(|&node| evaluate_power(&coefficients, node))
        .collect::<Vec<_>>();
    let derivatives = nodes
        .iter()
        .map(|&node| evaluate_power_derivative(&coefficients, node))
        .collect::<Vec<_>>();
    let barycentric = BarycentricInterpolator::new(nodes.clone(), values.clone()).unwrap();
    let newton = NewtonInterpolator::new(nodes.clone(), values.clone()).unwrap();
    let hermite = HermiteInterpolator::new(nodes, values, derivatives).unwrap();
    let a = -0.8;
    let b = 0.65;
    let expected = exact_power_integral(&coefficients, a, b);

    let barycentric_prepared = barycentric.prepare_antiderivative().unwrap();
    let newton_prepared = newton.prepare_antiderivative().unwrap();
    let hermite_prepared = hermite.prepare_antiderivative().unwrap();

    assert_eq!(barycentric_prepared.domain(), barycentric.domains()[0]);
    assert_eq!(barycentric_prepared.degree(), coefficients.len() - 1);
    assert_eq!(newton_prepared.degree(), coefficients.len() - 1);
    assert_eq!(hermite_prepared.degree(), 2 * coefficients.len() - 1);

    for result in [
        barycentric_prepared.integral(a, b, ExtrapolationPolicy::Reject),
        newton_prepared.integral(a, b, ExtrapolationPolicy::Reject),
        hermite_prepared.integral(a, b, ExtrapolationPolicy::Reject),
    ] {
        assert_close(result.unwrap(), expected, 1.0e-10, 1.0e-10);
    }
}

#[test]
fn structured_chebyshev_families_prepare_correct_integrals() {
    type StructuredConstructor =
        fn(Vec<f64>, f64, f64) -> Result<BarycentricInterpolator<f64>, SciOxideError>;

    for constructor in [
        BarycentricInterpolator::from_chebyshev_roots as StructuredConstructor,
        BarycentricInterpolator::from_chebyshev_lobatto as StructuredConstructor,
    ] {
        let seed = constructor(vec![0.0; 17], -2.0, 3.0).unwrap();
        let values = seed
            .nodes()
            .iter()
            .map(|&node| node.mul_add(node, 2.0 * node) + 1.0)
            .collect::<Vec<_>>();
        let interpolator = constructor(values, -2.0, 3.0).unwrap();
        let [domain] = interpolator.domains();
        let expected = exact_power_integral(&[1.0, 2.0, 1.0], domain.lower(), domain.upper());
        let actual = interpolator
            .prepare_antiderivative()
            .unwrap()
            .integral(domain.lower(), domain.upper(), ExtrapolationPolicy::Reject)
            .unwrap();
        assert_close(actual, expected, 1.0e-11, 1.0e-11);
    }
}

#[test]
fn normalized_coordinates_handle_large_offsets_and_narrow_domains() {
    let base = 1.0e12_f64;
    let spacing = 1.0e-3_f64;
    let nodes = (0..4)
        .map(|index| base + index as f64 * spacing)
        .collect::<Vec<_>>();
    let normalized = nodes
        .iter()
        .map(|&node| (node - base) / spacing)
        .collect::<Vec<_>>();
    let values = normalized
        .iter()
        .map(|&u| 3.0f64.mul_add(u * u, 2.0 * u) + 1.0)
        .collect::<Vec<_>>();
    let derivatives = normalized
        .iter()
        .map(|&u| (6.0 * u + 2.0) / spacing)
        .collect::<Vec<_>>();
    let expected = spacing * exact_power_integral(&[1.0, 2.0, 3.0], normalized[0], normalized[3]);
    let barycentric = BarycentricInterpolator::new(nodes.clone(), values.clone()).unwrap();
    let newton = NewtonInterpolator::new(nodes.clone(), values.clone()).unwrap();
    let hermite = HermiteInterpolator::new(nodes.clone(), values, derivatives).unwrap();

    for result in [
        barycentric.prepare_antiderivative().unwrap().integral(
            nodes[0],
            nodes[3],
            ExtrapolationPolicy::Reject,
        ),
        newton.prepare_antiderivative().unwrap().integral(
            nodes[0],
            nodes[3],
            ExtrapolationPolicy::Reject,
        ),
        hermite.prepare_antiderivative().unwrap().integral(
            nodes[0],
            nodes[3],
            ExtrapolationPolicy::Reject,
        ),
    ] {
        assert_close(result.unwrap(), expected, 2.5e-9, 2.5e-9);
    }
}

#[test]
fn prepared_queries_preserve_validation_order_and_extrapolation_policy() {
    let interpolator = BarycentricInterpolator::new(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
    let prepared = interpolator.prepare_antiderivative().unwrap();

    assert_error(
        &prepared.integral(f64::NAN, -1.0, ExtrapolationPolicy::Reject),
        scioxide_core::SciOxideErrorKind::NonFinite,
    );
    assert_error(
        &prepared.integral(2.0, f64::NAN, ExtrapolationPolicy::Reject),
        scioxide_core::SciOxideErrorKind::NonFinite,
    );
    assert_error(
        &prepared.integral(2.0, -1.0, ExtrapolationPolicy::Reject),
        scioxide_core::SciOxideErrorKind::InvalidInput,
    );
    assert_error(
        &prepared.integral(-1.0, -1.0, ExtrapolationPolicy::Reject),
        scioxide_core::SciOxideErrorKind::DomainError,
    );
    assert_eq!(
        prepared
            .integral(-1.0, -1.0, ExtrapolationPolicy::MethodDefined)
            .unwrap(),
        0.0
    );
    assert_close(
        prepared
            .integral(-1.0, 2.0, ExtrapolationPolicy::MethodDefined)
            .unwrap(),
        1.5,
        5e-15,
        5e-15,
    );
}

#[test]
fn singleton_and_f32_prepared_integrals_are_supported() {
    let singleton = HermiteInterpolator::new(vec![4.0_f32], vec![3.0_f32], vec![2.0_f32])
        .unwrap()
        .prepare_antiderivative()
        .unwrap();
    assert_eq!(singleton.degree(), 1);
    assert_eq!(
        singleton
            .integral(4.0, 4.0, ExtrapolationPolicy::Reject)
            .unwrap(),
        0.0
    );
    assert!(
        (singleton
            .integral(4.0, 4.5, ExtrapolationPolicy::MethodDefined)
            .unwrap()
            - 1.75)
            .abs()
            < 2.0e-6
    );

    let nodes = [10_000.0_f32, 10_000.25, 10_000.5];
    let values = [1.0_f32, 1.5, 2.0];
    let prepared = NewtonInterpolator::new(nodes.to_vec(), values.to_vec())
        .unwrap()
        .prepare_antiderivative()
        .unwrap();
    assert!(
        (prepared
            .integral(nodes[0], nodes[2], ExtrapolationPolicy::Reject)
            .unwrap()
            - 0.75)
            .abs()
            < 2.0e-5
    );
}

#[test]
fn repeated_prepared_queries_execute() {
    let count = 129usize;
    let interpolator =
        BarycentricInterpolator::from_chebyshev_lobatto(vec![2.0; count], -1.0, 1.0).unwrap();
    let prepared = interpolator.prepare_antiderivative().unwrap();
    assert_eq!(prepared.degree(), count - 1);

    let result = (0..64)
        .try_fold(0.0, |total, index| {
            let a = -1.0 + index as f64 / 64.0;
            let b = a + 0.5;
            prepared
                .integral(a, b, ExtrapolationPolicy::Reject)
                .map(|value| total + value)
        })
        .unwrap();
    assert_close(result, 64.0, 5e-14, 5e-14);
}
