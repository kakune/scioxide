//! Analytic and exact-moment tests for sampled quadrature.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Budget, RealScalar};
use scioxide_integrate::*;

#[test]
fn repeated_coordinates_empty_inputs_and_failures() {
    let plan = SampledTrapezoidPlan::new(&[0.0, 1.0, 2.0, 3.0]).unwrap();
    let mut workspace = plan.cumulative_workspace(1).unwrap();
    let mut out = [99.0; 4];
    plan.cumulative_into(&[0.0, 1.0, 2.0, 3.0], &mut out, &mut workspace)
        .unwrap();
    assert_eq!(out, [0.0, 0.5, 2.0, 4.5]);
    assert!(SampledSimpsonPlan::new(&[0.0, 1.0, 1.0]).is_err());
    assert!(SampledSimpsonPlan::new(&[0.0, 1.0, 0.0]).is_err());
    assert!(SampledRombergPlan::new(4, 1.0_f64).is_err());
    let before = out;
    assert!(matches!(
        (plan.cumulative_into(&[0.0, 1.0, f64::NAN, 0.0], &mut out, &mut workspace))
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::NonFinite { .. }
    ));
    assert_eq!(out, before);
    let mut zero = 99.0;
    SampledTrapezoidPlan::<f64>::new(&[])
        .unwrap()
        .integrate_into::<f64, _>(&[], &mut zero, &mut workspace)
        .unwrap();
    assert_eq!(zero, 0.0);
    let mut empty_vector = [99.0; 2];
    SampledTrapezoidPlan::<f64>::new(&[])
        .unwrap()
        .integrate_into::<f64, _>(&[], &mut empty_vector, &mut plan.workspace(2).unwrap())
        .unwrap();
    assert_eq!(empty_vector, [0.0; 2]);
    SampledSimpsonPlan::new(&[1.0])
        .unwrap()
        .integrate_into(&[12.0], &mut zero, &mut workspace)
        .unwrap();
    assert_eq!(zero, 0.0);
}

#[test]
fn romberg_dyadic_prefixes_and_native_f32() {
    let plan = SampledRombergPlan::new(17, 0.125_f64).unwrap();
    let samples: Vec<_> = (0..17).map(|i| (i as f64 * 0.125).powi(4)).collect();
    let mut output = [0.0; 5];
    let mut workspace = plan.cumulative_workspace(1).unwrap();
    plan.cumulative_into(&samples, &mut output, &mut workspace)
        .unwrap();
    assert_eq!(plan.cumulative_indices(), &[1, 2, 4, 8, 16]);
    for (&index, &integral) in plan.cumulative_indices()[2..].iter().zip(&output[2..]) {
        assert!((integral - (index as f64 * 0.125).powi(5) / 5.0).abs() < 3e-14);
    }
    let mut total = 0.0;
    plan.integrate_into(&samples, &mut total, &mut workspace)
        .unwrap();
    assert_eq!(total, output[4]);
    let mut output = [0.0_f32; 2];
    let samples = [
        [0.0_f32, 1.0],
        [1.0, 2.0],
        [4.0, 3.0],
        [9.0, 4.0],
        [16.0, 5.0],
    ];
    SampledRombergPlan::new(5, 1.0)
        .unwrap()
        .integrate_into(
            &samples,
            &mut output,
            &mut SampledRombergPlan::new(5, 1.0_f32)
                .unwrap()
                .workspace(2)
                .unwrap(),
        )
        .unwrap();
    assert!((output[0] - 64.0 / 3.0).abs() < 3e-6);
    assert!((output[1] - 12.0).abs() < 3e-6);
}

#[test]
fn stencils_match_known_rational_weights_and_error_coefficients() {
    for (nodes, endpoints, weights, error) in [
        (1, NewtonCotesEndpoints::Open, vec![2.0], 1.0 / 3.0),
        (2, NewtonCotesEndpoints::Closed, vec![0.5, 0.5], -1.0 / 12.0),
        (
            3,
            NewtonCotesEndpoints::Closed,
            vec![1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0],
            -1.0 / 90.0,
        ),
        (2, NewtonCotesEndpoints::Open, vec![1.5, 1.5], 0.75),
    ] {
        let plan = NewtonCotes::<f64>::new(nodes, endpoints, Budget::ONE).unwrap();
        for (&actual, expected) in plan.weights().iter().zip(weights) {
            assert!((actual - expected).abs() < 2e-15);
        }
        assert!((plan.error_coefficient() - error).abs() < 2e-15);
    }
    for endpoints in [NewtonCotesEndpoints::Open, NewtonCotesEndpoints::Closed] {
        for nodes in 2..=11 {
            let plan = NewtonCotes::<f64>::new(nodes, endpoints, Budget::ONE).unwrap();
            let start = usize::from(endpoints == NewtonCotesEndpoints::Open);
            let width = if start == 0 { nodes - 1 } else { nodes + 1 } as f64;
            for degree in 0..=plan.exactness_degree() {
                // Normalize the polynomial to [0,1], independently of the preparation arithmetic.
                let samples: Vec<_> = (0..nodes)
                    .map(|j| ((j + start) as f64 / width).powi(degree as i32))
                    .collect();
                let mut out = 0.0;
                plan.integrate_samples_into(&samples, 1.0 / width, &mut out, &mut [0.0])
                    .unwrap();
                assert!(
                    (out - 1.0 / (degree + 1) as f64).abs() < 3e-14,
                    "nodes {nodes}, degree {degree}, {endpoints:?}, {out}"
                );
            }
            let native = NewtonCotes::<f32>::new(nodes, endpoints, Budget::ONE).unwrap();
            assert!((native.weights().iter().sum::<f32>() - width as f32).abs() < 5e-5);
        }
    }
}

#[test]
fn composite_samples_match_the_mapped_rule_layout() {
    // Three closed Simpson panels share endpoints: 7 samples with spacing (b - a) / 6.
    let simpson = NewtonCotes::<f64>::new(
        3,
        NewtonCotesEndpoints::Closed,
        Budget::try_from(3).unwrap(),
    )
    .unwrap();
    let interval = scioxide_core::Interval::new(-1.0, 2.0).unwrap();
    let mut metadata =
        vec![scioxide_integrate::FixedRuleNode::new(0.0, 0.0, None); simpson.node_count()];
    simpson.map_nodes_into(interval, &mut metadata).unwrap();
    let samples: Vec<[f64; 2]> = metadata
        .iter()
        .map(|node| {
            let x = node.point();
            [x * x * x, 1.0]
        })
        .collect();
    let mut output = [0.0; 2];
    simpson
        .integrate_samples_into(&samples, 0.5, &mut output, &mut [0.0; 2])
        .unwrap();
    assert!((output[0] - 3.75).abs() < 1e-14);
    assert!((output[1] - 3.0).abs() < 1e-14);
    // Open midpoint panels skip their endpoints: samples sit at odd grid points.
    let midpoint =
        NewtonCotes::<f64>::new(1, NewtonCotesEndpoints::Open, Budget::try_from(4).unwrap())
            .unwrap();
    let mut out = 0.0;
    midpoint
        .integrate_samples_into(&[1.0, 3.0, 5.0, 7.0], 0.125, &mut out, &mut [0.0])
        .unwrap();
    assert!((out - 4.0).abs() < 1e-15);
    let before = output;
    assert!(
        simpson
            .integrate_samples_into(&samples[..6], 0.5, &mut output, &mut [0.0; 2])
            .is_err()
    );
    assert!(
        simpson
            .integrate_samples_into(&samples, f64::NAN, &mut output, &mut [0.0; 2])
            .is_err()
    );
    assert_eq!(output, before);
}

#[test]
fn panel_rules_share_the_signed_monotone_coordinate_contract() {
    fn check<S: RealScalar>() {
        let one = S::ONE;
        let two = S::TWO;
        for coordinates in [
            [S::ZERO, one, one],
            [S::ZERO, one, S::ZERO],
            [two, one, one],
            [two, one, two],
        ] {
            for error in [
                SampledTrapezoidPlan::new(&coordinates).unwrap_err(),
                SampledSimpsonPlan::new(&coordinates).unwrap_err(),
            ] {
                assert!(matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::InvalidInput { .. }
                ));
            }
        }
        for bad in [S::NAN, S::INFINITY, -S::INFINITY] {
            for coordinates in [[bad, one, two], [S::ZERO, bad, two], [S::ZERO, one, bad]] {
                for error in [
                    SampledTrapezoidPlan::new(&coordinates).unwrap_err(),
                    SampledSimpsonPlan::new(&coordinates).unwrap_err(),
                ] {
                    assert!(matches!(
                        error.view(),
                        scioxide_core::SciOxideErrorView::NonFinite {
                            origin: scioxide_core::NonFiniteOrigin::Input,
                            ..
                        }
                    ));
                }
            }
        }
        let tolerance = S::try_from_usize_exact(32).unwrap() * S::EPSILON;
        for coordinates in [
            &[][..],
            &[one][..],
            &[S::ZERO, two][..],
            &[two, S::ZERO][..],
        ] {
            let values = vec![one; coordinates.len()];
            let expected = coordinates.last().copied().unwrap_or(S::ZERO)
                - coordinates.first().copied().unwrap_or(S::ZERO);
            let mut workspace = SampledTrapezoidPlan::new(coordinates)
                .unwrap()
                .workspace(1)
                .unwrap();
            let mut trapezoid = one;
            let mut simpson = one;
            SampledTrapezoidPlan::new(coordinates)
                .unwrap()
                .integrate_into(&values, &mut trapezoid, &mut workspace)
                .unwrap();
            SampledSimpsonPlan::new(coordinates)
                .unwrap()
                .integrate_into(&values, &mut simpson, &mut workspace)
                .unwrap();
            assert!((trapezoid - expected).abs() <= tolerance);
            assert!((simpson - expected).abs() <= tolerance);
        }
        for coordinates in [[S::ZERO, one, two], [two, one, S::ZERO]] {
            let mut workspace = SampledTrapezoidPlan::new(&coordinates)
                .unwrap()
                .cumulative_workspace(1)
                .unwrap();
            let mut trapezoid = [S::ZERO; 3];
            let mut simpson = [S::ZERO; 3];
            SampledTrapezoidPlan::new(&coordinates)
                .unwrap()
                .cumulative_into(&coordinates, &mut trapezoid, &mut workspace)
                .unwrap();
            SampledSimpsonPlan::new(&coordinates)
                .unwrap()
                .cumulative_into(&coordinates, &mut simpson, &mut workspace)
                .unwrap();
            for ((actual, other), x) in trapezoid.into_iter().zip(simpson).zip(coordinates) {
                let expected = (x * x - coordinates[0] * coordinates[0]) * S::HALF;
                assert!((actual - expected).abs() <= tolerance);
                assert!((other - expected).abs() <= tolerance);
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn simpson_checks_representability_without_a_conditioning_cutoff() {
    fn check<S: RealScalar>() {
        let tiny = S::EPSILON.sqrt();
        let coordinates = [S::ZERO, tiny, S::ONE];
        let plan = SampledSimpsonPlan::new(&coordinates).unwrap();
        let mut workspace = plan.cumulative_workspace(1).unwrap();
        let mut prefixes = [S::ZERO; 3];
        plan.cumulative_into(&coordinates.map(|x| x * x), &mut prefixes, &mut workspace)
            .unwrap();
        let third = S::ONE / S::try_from_usize_exact(3).unwrap();
        let tolerance = S::try_from_usize_exact(32).unwrap() * S::EPSILON / tiny;
        assert!((prefixes[1] - tiny * tiny * tiny * third).abs() <= tolerance);
        assert!((prefixes[2] - third).abs() <= tolerance);
        let mut total = S::ZERO;
        plan.integrate_into(&coordinates.map(|x| x * x), &mut total, &mut workspace)
            .unwrap();
        assert!((total - prefixes[2]).abs() <= tolerance);
        // A tiny spacing beside a huge one makes the mirrored weight overflow.
        for coordinates in [
            [S::ZERO, S::MIN_POSITIVE, S::MAX],
            [S::ZERO, S::ONE, S::MAX * S::HALF],
        ] {
            assert!(matches!(
                SampledSimpsonPlan::new(&coordinates).unwrap_err().view(),
                scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            ));
        }
    }
    check::<f32>();
    check::<f64>();
}

fn simpson_total<S: RealScalar>(x: &[S], y: &[S]) -> Result<S, scioxide_core::SciOxideError> {
    let mut total = S::ZERO;
    SampledSimpsonPlan::new(x)?.integrate_into(
        y,
        &mut total,
        &mut SampledSimpsonPlan::new(x)?.workspace(1)?,
    )?;
    Ok(total)
}

fn simpson_prefixes<S: RealScalar>(x: &[S], y: &[S]) -> Vec<S> {
    let mut prefixes = vec![S::ZERO; x.len()];
    SampledSimpsonPlan::new(x)
        .unwrap()
        .cumulative_into(
            y,
            &mut prefixes,
            &mut SampledSimpsonPlan::new(x)
                .unwrap()
                .cumulative_workspace(1)
                .unwrap(),
        )
        .unwrap();
    prefixes
}

#[test]
fn simpson_mirrored_half_stencils_have_identical_weights() {
    fn check<S: RealScalar>() {
        let value = |v: f64| S::try_from_f64_lossy(v).unwrap();
        let large = S::MAX.sqrt();
        for spacing in [1.0, 0.1, 0.125, 3.7, -0.3] {
            let h = value(spacing);
            let x = [S::ZERO, h, h + h];
            // Equal effective weights of the outer samples, bit for bit.
            let left = simpson_total(&x, &[S::ONE, S::ZERO, S::ZERO]).unwrap();
            let right = simpson_total(&x, &[S::ZERO, S::ZERO, S::ONE]).unwrap();
            assert_eq!(left.into_f64().to_bits(), right.into_f64().to_bits());
            for magnitude in [S::ONE, large] {
                let y = [magnitude, S::ZERO, -magnitude];
                assert_eq!(simpson_total(&x, &y).unwrap(), S::ZERO);
                assert_eq!(simpson_prefixes(&x, &y)[2], S::ZERO);
            }
        }
        for spacing in [1.0, 0.125, -0.5] {
            // Antisymmetric samples on a longer, exactly uniform grid cancel exactly.
            let h = value(spacing);
            let x: Vec<S> = (0..9).map(|i| h * value(f64::from(i))).collect();
            let y: Vec<S> = (0..9).map(|i| large * value(f64::from(i - 4))).collect();
            assert_eq!(simpson_total(&x, &y).unwrap(), S::ZERO);
            assert_eq!(simpson_prefixes(&x, &y)[8], S::ZERO);
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn simpson_cancellation_matches_exact_references() {
    fn check<S: RealScalar>(large: f64, tiny: f64) {
        let value = |v: f64| S::try_from_f64_lossy(v).unwrap();
        let close = |actual: S, expected: S| {
            (actual - expected).abs() <= value(2.0) * S::EPSILON * expected.abs()
        };
        let x = [S::ZERO, S::ONE, value(2.0)];
        // Exact integral 4/3 * tiny: the large samples have equal effective weights.
        let total = simpson_total(&x, &[value(large), value(tiny), value(-large)]).unwrap();
        assert!(close(total, value(4.0 / 3.0) * value(tiny)), "{total:?}");
        let x = [S::ZERO, value(2.0), value(4.0)];
        assert_eq!(
            simpson_total(&x, &[S::MAX, S::ZERO, -S::MAX]).unwrap(),
            S::ZERO
        );
        let x: Vec<S> = (0..5).map(|i| value(f64::from(i))).collect();
        let y = [1e16, 1.0, -1e16, 1.0, 1.0].map(value);
        assert!(close(simpson_prefixes(&x, &y)[2], value(4.0 / 3.0)));
    }
    check::<f32>(1e30, 1e-30);
    check::<f64>(1e300, 1e-300);
}

#[test]
fn simpson_weights_of_overflowing_spans_stay_finite() {
    fn check<S: RealScalar>(half_span: f64) {
        let value = |v: f64| S::try_from_f64_lossy(v).unwrap();
        let a = value(half_span);
        let x = [-a, S::ZERO, a];
        assert!(!(a + a).is_finite());
        // Effective weights a/3, 4a/3, a/3: the integral is -2a/3.
        let total = simpson_total(&x, &[S::ONE, -S::ONE, S::ONE]).unwrap();
        let expected = -(a / value(3.0)) * value(2.0);
        assert!((total - expected).abs() <= value(4.0) * S::EPSILON * expected.abs());
        let prefixes = simpson_prefixes(&x, &[S::ONE, -S::ONE, S::ONE]);
        assert_eq!(prefixes[2], total);
        // The integral of one over the span is genuinely unrepresentable.
        assert!(matches!(
            simpson_total(&x, &[S::ONE; 3]).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::NumericalInstability { .. }
        ));
    }
    check::<f32>(3e38);
    check::<f64>(1e308);
}

#[test]
fn romberg_coefficients_reproduce_dyadic_polynomial_moments() {
    fn check<S: RealScalar>() {
        let tolerance = S::try_from_usize_exact(128).unwrap() * S::EPSILON;
        for sign in [S::ONE, -S::ONE] {
            let spacing = sign / S::try_from_usize_exact(16).unwrap();
            let plan = SampledRombergPlan::new(17, spacing).unwrap();
            let mut workspace = plan.cumulative_workspace(1).unwrap();
            for degree in 0..=6 {
                let values: Vec<S> = (0..17)
                    .map(|i| (spacing * S::try_from_usize_exact(i).unwrap()).powi(degree))
                    .collect();
                let mut prefixes = [S::ZERO; 5];
                plan.cumulative_into(&values, &mut prefixes, &mut workspace)
                    .unwrap();
                for (level, (&index, actual)) in
                    plan.cumulative_indices().iter().zip(prefixes).enumerate()
                {
                    if degree as usize > 2 * level + 1 {
                        continue;
                    }
                    let endpoint = spacing * S::try_from_usize_exact(index).unwrap();
                    let expected = endpoint.powi(degree + 1)
                        / S::try_from_usize_exact((degree + 1) as usize).unwrap();
                    assert!((actual - expected).abs() <= tolerance);
                }
                let mut total = S::ZERO;
                plan.integrate_into(&values, &mut total, &mut workspace)
                    .unwrap();
                assert!((total - prefixes[4]).abs() <= tolerance);
            }
        }
    }
    check::<f32>();
    check::<f64>();
}
