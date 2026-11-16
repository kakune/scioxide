//! Public polynomial ownership and callback-boundary regressions.
#![allow(clippy::unwrap_used)]

use scioxide_core::{SciOxideError, SciOxideErrorKind};
use scioxide_poly::backend::polynomial_basis::{
    barycentric_weights, confluent_divided_differences_into, linear_difference_ratio,
    linear_multiply_difference, newton_value_and_derivative_from_iter,
    real_divided_differences_into,
};

#[test]
fn invalid_shapes_leave_in_place_tables_unchanged() {
    let mut values = [1.0_f64, 2.0, 3.0];
    for (nodes, dimension) in [(&[][..], 1), (&[0.0, 1.0][..], 1), (&[0.0, 1.0][..], 2)] {
        let error = real_divided_differences_into(nodes, dimension, &mut values).unwrap_err();
        assert!(matches!(
            error.kind(),
            SciOxideErrorKind::InvalidInput | SciOxideErrorKind::DimensionMismatch
        ));
        assert_eq!(values, [1.0, 2.0, 3.0]);
    }
    let mut empty: [f64; 0] = [];
    real_divided_differences_into(&[0.0, 1.0], 0, &mut empty).unwrap();
}

#[test]
fn iterator_errors_propagate_without_partial_results() {
    let error = SciOxideError::external("callback rejected");
    let mut calls = 0;
    let result = newton_value_and_derivative_from_iter(
        [Err(error)].into_iter(),
        1.0_f64,
        0.0,
        0.0,
        |_, _, _| {
            calls += 1;
            Ok(0.0)
        },
        |a, b| Ok(a + b),
    );
    assert_eq!(calls, 0);
    assert_eq!(
        result.unwrap_err(),
        SciOxideError::external("callback rejected")
    );
    let constant = newton_value_and_derivative_from_iter(
        core::iter::empty(),
        3.0_f64,
        0.0,
        0.0,
        |_, _, _| {
            calls += 1;
            Ok(0.0)
        },
        |a, b| Ok(a + b),
    )
    .unwrap();
    assert_eq!(constant, (3.0, 0.0));
    assert_eq!(calls, 0);
}

/// Range promotion of every public divided-difference form: subnormal node
/// gaps, overflowing node and value differences, and one extreme component
/// beside an ordinary one, which keeps its single-component table bit for bit.
#[test]
fn divided_difference_forms_promote_range_loss() {
    use scioxide_poly::{ApproxPolynomial, PolynomialApproximationPolicy, PolynomialBasis};
    let max = f64::MAX;
    let gap = f64::MIN_POSITIVE / 16.0;
    let subnormal = [0.0, gap, 2.0 * gap];
    let wide = [-max, 0.0, max];

    // Scalar construction (barycentric and conversion share this table).
    let policy = PolynomialApproximationPolicy::default();
    let scalar = |nodes: &[f64], values: Vec<f64>| {
        ApproxPolynomial::interpolate_newton_with_options(nodes.to_vec(), values, policy)
            .unwrap()
            .basis()
            .coefficients()
            .to_vec()
    };
    assert_eq!(
        scalar(&subnormal, vec![0.0, 3.0 * gap, 6.0 * gap]),
        [0.0, 3.0]
    );
    assert_eq!(scalar(&wide, vec![-max, 0.0, max]), [-max, 1.0]);

    // In place, single component.
    let mut values = [0.0, 3.0 * gap, 6.0 * gap];
    real_divided_differences_into(&subnormal, 1, &mut values).unwrap();
    assert_eq!(values, [0.0, 3.0, 0.0]);
    let mut values = [-max, 0.0, max];
    real_divided_differences_into(&wide, 1, &mut values).unwrap();
    assert_eq!(values, [-max, 1.0, 0.0]);

    // In place, an ordinary component (x² + 1) beside an extreme one.
    let nodes = [0.0, 1.0, 2.0, 3.0];
    let ordinary = [1.0, 2.0, 5.0, 10.0];
    let mut alone = ordinary;
    real_divided_differences_into(&nodes, 1, &mut alone).unwrap();
    let quarter = max / 4.0;
    let extreme = [quarter, -quarter, quarter, -quarter];
    let mut interleaved: Vec<f64> = ordinary
        .iter()
        .zip(&extreme)
        .flat_map(|(&a, &b)| [a, b])
        .collect();
    real_divided_differences_into(&nodes, 2, &mut interleaved).unwrap();
    let component =
        |c: usize| -> Vec<f64> { interleaved.iter().skip(c).step_by(2).copied().collect() };
    assert_eq!(component(0), alone);
    assert_eq!(component(1), [quarter, -max / 2.0, max / 2.0, -max / 3.0]);

    // Confluent Hermite data: 3x at subnormal gaps and x at the range limits.
    let mut coefficients = [0.0; 4];
    confluent_divided_differences_into(
        &subnormal[..2],
        &[0.0, 3.0 * gap],
        &[3.0; 2],
        1,
        &mut coefficients,
    )
    .unwrap();
    assert_eq!(coefficients, [0.0, 3.0, 0.0, 0.0]);
    confluent_divided_differences_into(&[-max, max], &[-max, max], &[1.0; 2], 1, &mut coefficients)
        .unwrap();
    assert_eq!(coefficients, [-max, 1.0, 0.0, 0.0]);

    // Confluent, x² beside an extreme linear component (MAX/8) x.
    let nodes = [0.0, 1.0, 2.0];
    let (values, slopes) = ([0.0, 1.0, 4.0], [0.0, 2.0, 4.0]);
    let mut alone = [0.0; 6];
    confluent_divided_differences_into(&nodes, &values, &slopes, 1, &mut alone).unwrap();
    let eighth = max / 8.0;
    let pair = |ordinary: &[f64; 3], scale: f64| -> Vec<f64> {
        ordinary.iter().flat_map(|&value| [value, scale]).collect()
    };
    let mut interleaved = [0.0; 12];
    confluent_divided_differences_into(
        &nodes,
        &[0.0, 0.0, 1.0, eighth, 4.0, 2.0 * eighth],
        &pair(&slopes, eighth),
        2,
        &mut interleaved,
    )
    .unwrap();
    let component =
        |c: usize| -> Vec<f64> { interleaved.iter().skip(c).step_by(2).copied().collect() };
    assert_eq!(component(0), alone);
    assert_eq!(component(1), [0.0, eighth, 0.0, 0.0, 0.0, 0.0]);
}

/// Sorted nodes with wide gaps satisfy the whole-table growth chain, but
/// samples near the overflow threshold still make the first-order numerators
/// overflow. Every such table must take the checked path and stay finite.
macro_rules! wide_gap_extreme_samples {
    ($name:ident, $scalar:ty) => {
        #[test]
        fn $name() {
            let max = <$scalar>::MAX;
            // (−MAX − MAX)/32 and the second order (MAX/8)/64 are exact.
            let mut values = [max, -max];
            real_divided_differences_into(&[0.0, 32.0], 1, &mut values).unwrap();
            assert_eq!(values, [max, -max / 16.0]);
            let mut values = [max, -max, max];
            real_divided_differences_into(&[0.0, 32.0, 64.0], 1, &mut values).unwrap();
            assert_eq!(values, [max, -max / 16.0, max / 512.0]);

            // One extreme component beside two ordinary ones (x and 2x + 1).
            let mut interleaved = [0.0, 1.0, max, 32.0, 65.0, -max];
            real_divided_differences_into(&[0.0, 32.0], 3, &mut interleaved).unwrap();
            assert_eq!(interleaved, [0.0, 1.0, max, 1.0, 2.0, -max / 16.0]);

            // The same samples through interpolation construction.
            let polynomial = scioxide_poly::ApproxPolynomial::interpolate_newton(
                vec![0.0 as $scalar, 32.0],
                vec![max, -max],
            )
            .unwrap();
            assert_eq!(
                scioxide_poly::PolynomialBasis::coefficients(polynomial.basis()),
                [max, -max / 16.0]
            );
        }
    };
}
wide_gap_extreme_samples!(wide_gap_extreme_samples_f32, f32);
wide_gap_extreme_samples!(wide_gap_extreme_samples_f64, f64);

macro_rules! weights {
    ($name:ident, $scalar:ty) => {
        #[test]
        fn $name() {
            assert_eq!(barycentric_weights(&[2.0 as $scalar]).unwrap(), [1.0]);
            assert_eq!(
                barycentric_weights::<$scalar>(&[]).unwrap_err().kind(),
                SciOxideErrorKind::InvalidInput
            );
            assert_eq!(
                barycentric_weights(&[-0.0 as $scalar, 0.0])
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::InvalidInput
            );
            for value in [<$scalar>::NAN, <$scalar>::INFINITY] {
                assert_eq!(
                    barycentric_weights(&[0.0, value]).unwrap_err().kind(),
                    SciOxideErrorKind::NonFinite
                );
                // A zero coefficient must not conceal malformed operands.
                for error in [
                    linear_multiply_difference(0.0, value, 1.0).unwrap_err(),
                    linear_difference_ratio(value, 0.0, 1.0, 0.0).unwrap_err(),
                ] {
                    assert_eq!(
                        match error.view() {
                            scioxide_core::SciOxideErrorView::NonFinite { origin, .. } =>
                                Some(origin),
                            _ => None,
                        }
                        .unwrap(),
                        scioxide_core::NonFiniteOrigin::Input
                    );
                }
            }
            // Exact symmetry independently gives the ratio [1, -2, 1].
            for gap in [<$scalar>::from_bits(1), <$scalar>::MAX] {
                let actual = barycentric_weights(&[-gap, 0.0, gap]).unwrap();
                for (value, expected) in actual.iter().zip([0.5, -1.0, 0.5]) {
                    assert!((*value - expected).abs() <= 4.0 * <$scalar>::EPSILON);
                }
            }
            assert_eq!(
                linear_difference_ratio(
                    <$scalar>::MAX,
                    -<$scalar>::MAX,
                    <$scalar>::MAX,
                    -<$scalar>::MAX
                )
                .unwrap(),
                1.0
            );
            assert_eq!(
                linear_multiply_difference(0.5, <$scalar>::MAX, -<$scalar>::MAX).unwrap(),
                <$scalar>::MAX
            );
            assert!(linear_difference_ratio(1.0 as $scalar, 0.0, 0.0, -0.0).is_err());
        }
    };
}
weights!(weights_f32, f32);
weights!(weights_f64, f64);

#[test]
fn ratio_fallback_preserves_subnormals_and_error_origins() {
    use scioxide_core::{NonFiniteOrigin, SciOxideErrorView};
    for gap in [f64::MIN_POSITIVE, f64::from_bits(1)] {
        let ratio = linear_difference_ratio(gap, 0.0, 2.0, 0.0).unwrap();
        assert_eq!(ratio, gap / 2.0);
    }
    let error = linear_difference_ratio(f64::MAX, -f64::MAX, f64::MIN_POSITIVE, 0.0).unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::NonFinite {
            origin: NonFiniteOrigin::Intermediate,
            ..
        }
    ));
}
