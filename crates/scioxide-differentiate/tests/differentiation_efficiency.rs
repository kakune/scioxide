//! Callback-cardinality and exact-arithmetic regressions for differentiation kernels.
#![allow(clippy::unwrap_used, clippy::float_cmp)]

use scioxide_core::{
    Budget, BudgetKind, Complex64, ErrorTolerance, ExecutionContext, NormSelection, NumericScalar,
    ProblemToken, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_differentiate::*;
use std::collections::BTreeSet;

fn point_bits(point: &[f64]) -> Vec<u64> {
    point.iter().map(|value| value.to_bits()).collect()
}

fn quadratic(point: &[f64]) -> f64 {
    point.iter().map(|value| value * value).sum::<f64>()
        + point.windows(2).map(|pair| pair[0] * pair[1]).sum::<f64>()
}

fn expected_hessian(dimension: usize) -> Vec<f64> {
    let mut expected = vec![0.0; dimension * dimension];
    for row in 0..dimension {
        expected[row * dimension + row] = 2.0;
        if row + 1 < dimension {
            expected[row * dimension + row + 1] = 1.0;
            expected[(row + 1) * dimension + row] = 1.0;
        }
    }
    expected
}

fn hessian_workspace(dimension: usize) -> FiniteDifferenceWorkspace<f64> {
    let method = FiniteDifference::default();
    let request = DerivativeRequest::new(&method, DerivativeShape::hessian(dimension));
    FiniteDifferenceWorkspace::workspace(request).unwrap()
}

fn fixed_count(dimension: usize, scheme: DifferenceScheme) -> usize {
    let point = vec![1.0; dimension];
    let mut workspace = hessian_workspace(dimension);
    let mut output = vec![f64::NAN; dimension * dimension];
    let mut points = BTreeSet::new();
    let report = FiniteDifference::default()
        .with_scheme(scheme)
        .hessian_into(
            |sample: &[f64], y: &mut [f64]| {
                assert!(
                    points.insert(point_bits(sample)),
                    "duplicate fixed Hessian point"
                );
                y[0] = quadratic(sample);
                Ok(())
            },
            &point,
            None,
            None,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert!(report.output_available());
    assert_eq!(report.evaluations(), points.len());
    for (actual, expected) in output.iter().zip(expected_hessian(dimension)) {
        let tolerance = if scheme == DifferenceScheme::Forward {
            2.0e-3
        } else {
            2.0e-6
        };
        assert!((*actual - expected).abs() < tolerance);
    }
    report.evaluations()
}

#[test]
fn fixed_hessians_evaluate_each_bit_identical_point_once() {
    assert_eq!(fixed_count(2, DifferenceScheme::Central), 9);
    assert_eq!(fixed_count(6, DifferenceScheme::Central), 73);
    assert_eq!(fixed_count(2, DifferenceScheme::Forward), 6);
    assert_eq!(fixed_count(6, DifferenceScheme::Forward), 28);
}

#[test]
fn fixed_hessian_resource_limit_tracks_the_reduced_callback_count() {
    let point = [1.0, 2.0];
    let method = FiniteDifference::default().with_scheme(DifferenceScheme::Central);
    for (limit, available) in [(8, false), (9, true)] {
        let mut workspace = hessian_workspace(2);
        let mut output = [99.0; 4];
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                SolveIdentity::new(
                    ProblemToken::new(837),
                    scioxide_core::algorithm_id!("test", "hessian-cache", 1),
                    0,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        let report = method
            .hessian_into(
                |sample: &[f64], y: &mut [f64]| {
                    y[0] = quadratic(sample);
                    Ok(())
                },
                &point,
                None,
                None,
                &mut output,
                &mut workspace,
                &mut context,
            )
            .unwrap();
        assert_eq!(report.output_available(), available);
        assert_eq!(report.evaluations(), limit);
        if available {
            assert_ne!(output, [99.0; 4]);
        } else {
            assert_eq!(output, [99.0; 4]);
            assert_eq!(
                report.termination(),
                DifferentiationTermination::EvaluationLimit
            );
        }
    }
}

fn high_order_count(relative: bool) -> usize {
    let point = [1.0, 2.0];
    let offsets = [-0.5, -0.25, 0.0, 0.25, 0.5];
    let physical_x = offsets.map(|offset| point[0] + offset);
    let physical_y = offsets.map(|offset| point[1] + offset);
    let mut points = BTreeSet::new();
    let (report, values) = if relative {
        let method = RelativeDifference::new(
            DerivativeOperation::Hessian,
            &[&offsets, &offsets],
            1,
            scioxide_differentiate::NodeDifferenceOptions::default(),
        )
        .unwrap();
        let result = method
            .evaluate(
                |sample: &[f64], output: &mut [_]| {
                    assert!(
                        points.insert(point_bits(sample)),
                        "duplicate relative Hessian point"
                    );
                    output[0] = quadratic(sample);
                    Ok(())
                },
                &point,
                None,
            )
            .unwrap();
        let (report, values, errors) = result.into_parts();
        assert!(errors.is_none());
        if let Some(values) = &values {
            assert_eq!(report.completed_entries(), values.len());
        }
        (report, values.unwrap())
    } else {
        let plan = PhysicalDifferencePlan::new(
            DerivativeOperation::Hessian,
            &point,
            &[&physical_x, &physical_y],
            1,
            scioxide_differentiate::NodeDifferenceOptions::default(),
        )
        .unwrap();
        let result = plan
            .evaluate(
                |sample: &[f64], output: &mut [_]| {
                    assert!(
                        points.insert(point_bits(sample)),
                        "duplicate physical Hessian point"
                    );
                    output[0] = quadratic(sample);
                    Ok(())
                },
                None,
            )
            .unwrap();
        let (report, values, errors) = result.into_parts();
        assert!(errors.is_none());
        if let Some(values) = &values {
            assert_eq!(report.completed_entries(), values.len());
        }
        (report, values.unwrap())
    };
    // Frozen bits from the uncached tensor path on the same physical nodes.
    let actual_bits = [
        values[0].to_bits(),
        values[1].to_bits(),
        values[2].to_bits(),
        values[3].to_bits(),
    ];
    assert_eq!(
        actual_bits,
        [2.0000000000000004, 1.0, 1.0, 2.0000000000000004].map(f64::to_bits)
    );
    assert_eq!(report.evaluations(), points.len());
    report.evaluations()
}

#[test]
fn high_order_hessians_reuse_base_and_axis_nodes() {
    // 1 base + 2 * 4 axis points + 4 * 4 mixed corners.
    assert_eq!(high_order_count(false), 25);
    assert_eq!(high_order_count(true), 25);
}

#[test]
fn adaptive_hessians_never_repeat_a_sample_point() {
    for method in [
        AdaptiveDifferenceMethod::Richardson,
        AdaptiveDifferenceMethod::Ridders,
    ] {
        let options = AdaptiveDifference::default()
            .with_method(method)
            .with_tolerance(
                ErrorTolerance::scalar(1.0e-8, 1.0e-8, NormSelection::Infinity).unwrap(),
            );
        let mut points = BTreeSet::new();
        let result = options
            .hessian(
                |sample: &[f64], output: &mut [f64]| {
                    assert!(
                        points.insert(point_bits(sample)),
                        "duplicate adaptive Hessian point for {method:?}"
                    );
                    output[0] = quadratic(sample);
                    Ok(())
                },
                &[1.0, 2.0],
                None,
            )
            .unwrap();
        assert!(result.report().output_available());
        assert_eq!(result.report().evaluations(), points.len());
        for (actual, expected) in result.values().unwrap().iter().zip([2.0, 1.0, 1.0, 2.0]) {
            assert!((*actual - expected).abs() < 1.0e-8);
        }
    }
}

#[test]
fn composed_complex_hessian_reuses_each_imaginary_axis_base() {
    let point = [1.0, 2.0];
    let offsets = [-0.5, -0.25, 0.0, 0.25, 0.5];
    let method = RelativeComplexStep::new(
        &[&offsets, &offsets],
        ComplexStep::default().with_step(1.0e-100),
    )
    .unwrap();
    let mut points = BTreeSet::new();
    let result = method
        .evaluate(
            |sample: &[Complex64], output: &mut [Complex64]| {
                let key: Vec<_> = sample
                    .iter()
                    .flat_map(|value| {
                        [
                            value.real_component().to_bits(),
                            value.imaginary_component().to_bits(),
                        ]
                    })
                    .collect();
                assert!(points.insert(key), "duplicate complex Hessian point");
                output[0] = sample[0] * sample[0] + sample[0] * sample[1] + sample[1] * sample[1];
                Ok(())
            },
            &point,
            None,
        )
        .unwrap();
    // One real base plus two imaginary directions times one shared imaginary
    // base and four non-base real nodes on each of two axes.
    assert_eq!(result.report().evaluations(), 19);
    assert_eq!(result.report().evaluations(), points.len());
    // Frozen bits from the uncached ordered-composition path.
    let values = result.values().unwrap();
    let actual_bits = [
        values[0].to_bits(),
        values[1].to_bits(),
        values[2].to_bits(),
        values[3].to_bits(),
    ];
    assert_eq!(
        actual_bits,
        [2.0, 1.0000000000000022, 1.0000000000000022, 2.0].map(f64::to_bits)
    );
}

#[test]
fn exact_stencils_survive_overflowing_naive_differences_and_subnormal_results() {
    let central = FiniteDifferenceStencil::new(0.0, &[-1.0, 0.0, 1.0], 1).unwrap();
    assert_eq!(
        central.apply(&[-f64::MAX, 0.0, f64::MAX]).unwrap(),
        f64::MAX
    );

    let tiny = f64::from_bits(1);
    let subnormal = FiniteDifferenceStencil::new(0.0, &[-1.0, 1.0], 1).unwrap();
    assert_eq!(subnormal.apply(&[-tiny, tiny]).unwrap(), tiny);
}

#[test]
fn adaptive_derivatives_match_independent_mpfr_references() {
    for line in include_str!("data/differentiation.txt").lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let mut fields = line.split_ascii_whitespace();
        let width = fields.next().unwrap();
        let input_field = fields.next().unwrap();
        let first_field = fields.next().unwrap();
        let second_field = fields.next().unwrap();
        for method in [
            AdaptiveDifferenceMethod::Richardson,
            AdaptiveDifferenceMethod::Ridders,
        ] {
            for (operation, expected_field) in [
                (DerivativeOperation::First, first_field),
                (DerivativeOperation::Second, second_field),
            ] {
                if width == "64" {
                    let x = f64::from_bits(u64::from_str_radix(input_field, 16).unwrap());
                    let expected = f64::from_bits(u64::from_str_radix(expected_field, 16).unwrap());
                    let options = AdaptiveDifference::default()
                        .with_method(method)
                        .with_tolerance(
                            ErrorTolerance::scalar(1.0e-9, 1.0e-9, NormSelection::Infinity)
                                .unwrap(),
                        );
                    let function = |t: f64| Ok(t.exp() * t.sin());
                    let result = match operation {
                        DerivativeOperation::First => options.derivative(function, x),
                        _ => options.second_derivative(function, x),
                    }
                    .unwrap();
                    assert!(
                        result.report().output_available(),
                        "{method:?} {operation:?}"
                    );
                    let actual = result.values().unwrap()[0];
                    let error = (actual - expected).abs();
                    assert!(error <= 2.0e-8 * expected.abs().max(1.0));
                    assert!(result.errors().unwrap()[0].is_finite());
                } else {
                    let x = f32::from_bits(u32::from_str_radix(input_field, 16).unwrap());
                    let expected = f32::from_bits(u32::from_str_radix(expected_field, 16).unwrap());
                    let options = AdaptiveDifference::default()
                        .with_method(method)
                        .with_tolerance(
                            ErrorTolerance::scalar(2.0e-3, 2.0e-3, NormSelection::Infinity)
                                .unwrap(),
                        );
                    let function = |t: f32| Ok(t.exp() * t.sin());
                    let result = match operation {
                        DerivativeOperation::First => options.derivative(function, x),
                        _ => options.second_derivative(function, x),
                    }
                    .unwrap();
                    assert!(
                        result.report().output_available(),
                        "{method:?} {operation:?}"
                    );
                    let actual = result.values().unwrap()[0];
                    let error = (actual - expected).abs();
                    assert!(error <= 8.0e-3 * expected.abs().max(1.0));
                    assert!(result.errors().unwrap()[0].is_finite());
                }
            }
        }
    }
}

#[test]
fn fixed_reports_count_completed_entries_without_publishing_a_partial_gradient() {
    use scioxide_core::{
        Budget, BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity, SolveState,
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("test", "partial-gradient", 1),
        2,
        0,
    );
    let mut state = SolveState::new();
    let mut limits =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(3).unwrap());
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut limits)
        .unwrap();
    let mut work = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
        &FiniteDifference::default(),
        DerivativeShape::gradient(2),
    ))
    .unwrap();
    let mut output = [77.0_f64; 2];
    let mut calls = 0;
    let report = FiniteDifference::default()
        .with_scheme(DifferenceScheme::Central)
        .gradient_into(
            |x: &[f64], y: &mut [f64]| {
                calls += 1;
                y[0] = x[0] * x[0] + x[1] * x[1];
                Ok(())
            },
            &[1.0, 2.0],
            None,
            None,
            &mut output,
            &mut work,
            &mut context,
        )
        .unwrap();
    assert_eq!(
        report.termination(),
        DifferentiationTermination::EvaluationLimit
    );
    assert_eq!(report.evaluations(), calls);
    assert_eq!(calls, 3);
    assert_eq!(report.completed_entries(), 1);
    assert!(!report.output_available());
    assert!(!report.is_success());
    assert_eq!(output, [77.0; 2]);
    let mut limits = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut limits)
        .unwrap();
    let report = FiniteDifference::default()
        .gradient_into(
            |x: &[f64], y: &mut [f64]| {
                y[0] = x[0] * x[0] + x[1] * x[1];
                Ok(())
            },
            &[1.0, 2.0],
            None,
            None,
            &mut output,
            &mut work,
            &mut context,
        )
        .unwrap();
    assert_eq!(report.termination(), DifferentiationTermination::Completed);
    assert_eq!(report.completed_entries(), output.len());
    assert!(report.output_available());
    assert!(report.is_success());
    for (actual, expected) in output.into_iter().zip([2.0, 4.0]) {
        assert!((actual - expected).abs() < 1e-8);
    }
}
