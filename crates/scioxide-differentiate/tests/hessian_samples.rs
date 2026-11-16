//! Evaluation-local callback reuse and transactional/resource contracts.
#![allow(clippy::unwrap_used, clippy::float_cmp)]

#[cfg(feature = "std")]
use scioxide_core::ThreadExecutor;
use scioxide_core::{
    Budget, BudgetKind, Complex64, ExecutionContext, Executor, ProblemToken, RealScalar,
    SequentialExecutor, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_differentiate::*;

fn scalar(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>() + x[0] * x[1]
}

#[test]
fn fixed_hessian_counts_and_resource_stops_in_both_precisions() {
    fn check<S: RealScalar>() {
        let point = [S::ONE; 2];
        let mut w = FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
            &FiniteDifference::default(),
            DerivativeShape::hessian(2),
        ))
        .unwrap();
        // Exact dyadic geometry isolates resource/publication behavior from
        // the separate rounded-node numerical tests.
        let method =
            FiniteDifference::default().with_step(DifferenceStep::absolute(S::HALF).unwrap());
        let identity = SolveIdentity::new(
            ProblemToken::new(1),
            scioxide_core::algorithm_id!("Hessian", "test", 1),
            0,
            0,
        );
        for allowance in [8, 9] {
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(allowance).unwrap());
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
                .unwrap();
            let mut output = [S::ONE; 4];
            let mut calls = 0;
            let report = method
                .hessian_into(
                    |x: &[S], y: &mut [S]| {
                        calls += 1;
                        y[0] = x[0] * x[0] + x[1] * x[1];
                        Ok(())
                    },
                    &point,
                    None,
                    None,
                    &mut output,
                    &mut w,
                    &mut context,
                )
                .unwrap();
            assert_eq!(calls, allowance);
            assert_eq!(report.evaluations(), calls);
            if allowance == 8 {
                assert!(!report.output_available());
                assert_eq!(output, [S::ONE; 4]);
            } else {
                assert!(report.output_available());
                assert_eq!(output, [S::TWO, S::ZERO, S::ZERO, S::TWO]);
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn high_order_unique_points_and_previous_contraction_bits() {
    for nodes in [
        &[-0.5, 0.0, 0.5][..],
        &[-0.5, 0.25, 0.5][..],
        &[-0.5, -0.25, 0.0, 0.25, 0.5][..],
    ] {
        let p = PhysicalDifferencePlan::new(
            DerivativeOperation::Hessian,
            &[0.0, 0.0],
            &[nodes, nodes],
            1,
            scioxide_differentiate::NodeDifferenceOptions::default(),
        )
        .unwrap();
        let mut w = p.workspace::<f64>(()).unwrap();
        let mut actual = [0.0; 4];
        let mut points = Vec::new();
        let r = p
            .evaluate_into(
                |x: &[f64], y: &mut [_]| {
                    points.push(x.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    y[0] = scalar(x);
                    Ok(())
                },
                None,
                &mut actual,
                &mut w,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        let k = nodes.len();
        let b = usize::from(nodes.contains(&0.0));
        assert_eq!(points.len(), b + 2 * (k - b) + (k - b) * (k - b));
        assert_eq!(r.evaluations(), points.len());
        points.sort();
        points.dedup();
        assert_eq!(points.len(), r.evaluations());
        let first = FiniteDifferenceStencil::new(0.0, nodes, 1).unwrap();
        let second = FiniteDifferenceStencil::new(0.0, nodes, 2).unwrap();
        let diag = second
            .apply(&nodes.iter().map(|&x| scalar(&[x, 0.0])).collect::<Vec<_>>())
            .unwrap();
        let contracted: Vec<_> = nodes
            .iter()
            .map(|&a| {
                first
                    .apply(&nodes.iter().map(|&b| scalar(&[a, b])).collect::<Vec<_>>())
                    .unwrap()
            })
            .collect();
        let mixed = first.apply(&contracted).unwrap();
        let old = [diag, mixed, mixed, diag];
        assert_eq!(actual.map(f64::to_bits), old.map(f64::to_bits));
        for (&a, &b) in actual.iter().zip(&[2.0, 1.0, 1.0, 2.0]) {
            assert!((a - b).abs() <= 4.0 * f64::EPSILON);
        }
        p.evaluate_into(
            |x: &[f64], y: &mut [f64]| {
                y[0] = scalar(x);
                Ok(())
            },
            None,
            &mut actual,
            &mut w,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    }
}

#[test]
fn adaptive_hessian_never_repeats_physical_samples() {
    for method in [
        AdaptiveDifferenceMethod::Richardson,
        AdaptiveDifferenceMethod::Ridders,
    ] {
        let p = AdaptiveDifference::default()
            .with_method(method)
            .with_execution(scioxide_core::ExecutionPolicy::Auto);
        let mut w = AdaptiveDifferenceWorkspace::<f64>::workspace(DerivativeRequest::new(
            &p,
            DerivativeShape::hessian(2),
        ))
        .unwrap();
        let mut actual = [0.0; 4];
        let mut errors = [0.0; 4];
        let mut points = Vec::new();
        let r = p
            .hessian_into(
                |x: &[f64], y: &mut [f64]| {
                    points.push(x.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    y[0] = scalar(x);
                    Ok(())
                },
                &[1.0, 2.0],
                None,
                &mut actual,
                &mut errors,
                &mut w,
                scioxide_core::ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(r.output_available());
        assert_eq!(r.evaluations(), points.len());
        points.sort();
        points.dedup();
        assert_eq!(r.evaluations(), points.len());
        for (&a, &b) in actual.iter().zip(&[2.0, 1.0, 1.0, 2.0]) {
            assert!((a - b).abs() < 1e-8);
        }
        p.hessian_into(
            |x: &[f64], y: &mut [f64]| {
                y[0] = scalar(x);
                Ok(())
            },
            &[2.0, 3.0],
            None,
            &mut actual,
            &mut errors,
            &mut w,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        let previous = actual;
        assert!(
            p.hessian_into(
                |_: &[f64], _: &mut [f64]| Ok(()),
                &[1.0, 2.0],
                None,
                &mut actual,
                &mut errors,
                &mut w,
                scioxide_core::ExecutionContext::sequential()
            )
            .is_err()
        );
        assert_eq!(actual, previous);
        assert!(
            p.hessian_into(
                |x: &[f64], y: &mut [f64]| {
                    y[0] = scalar(x);
                    Ok(())
                },
                &[1.0, 2.0],
                None,
                &mut actual,
                &mut errors,
                &mut w,
                scioxide_core::ExecutionContext::sequential()
            )
            .unwrap()
            .output_available()
        );
    }
}

#[test]
fn independent_decimal_exponential_derivatives_and_estimates() {
    // Math::BigFloat 100-digit exp/sin/cos at x=1, from
    // exp_sin_reference.pl.
    for method in [
        AdaptiveDifferenceMethod::Richardson,
        AdaptiveDifferenceMethod::Ridders,
    ] {
        let p = AdaptiveDifference::default()
            .with_method(method)
            .with_execution(scioxide_core::ExecutionPolicy::Auto);
        let function = |x: f64| Ok(x.exp() * x.sin());
        for (order, exact) in [(1, 3.756_049_227_094_727_5), (2, 2.937_387_879_831_770_3)] {
            let r = if order == 1 {
                p.derivative(function, 1.0)
            } else {
                p.second_derivative(function, 1.0)
            }
            .unwrap();
            assert!(r.report().is_success());
            let actual = r.values().unwrap()[0];
            let estimate = r.errors().unwrap()[0];
            assert!(
                (actual - exact).abs() <= estimate,
                "{method:?}/order {order}: {actual} vs {exact}; estimate {estimate}"
            );
        }
    }
}

#[test]
fn fixed_stencils_retain_range_before_physical_scaling() {
    fn check<S: RealScalar>() {
        let four = S::try_from_f64_exact(4.0).unwrap();
        let max = S::MAX;
        let method = FiniteDifference::default().with_step(DifferenceStep::absolute(four).unwrap());
        let result = method
            .hessian(
                |x: &[S], y: &mut [S]| {
                    let t = x[0] / four;
                    y[0] = max * (S::ONE - (S::TWO) * t * t);
                    Ok(())
                },
                &[S::ZERO],
                None,
                None,
            )
            .unwrap();
        // Analytic f'' = -4 MAX / 4^2 = -MAX/4. All callback values
        // are finite even though the unscaled weighted sum is -4 MAX.
        assert_eq!(result.values().unwrap()[0], -max / four);
        let tiny = S::ZERO.next_after(S::ONE);
        let method =
            FiniteDifference::default().with_step(DifferenceStep::absolute(S::HALF).unwrap());
        let result = method
            .gradient(
                |x: &[S], y: &mut [S]| {
                    y[0] = (x[0] + S::HALF) * tiny;
                    Ok(())
                },
                &[S::ZERO],
                None,
                None,
            )
            .unwrap();
        // Rounded callback samples are [0,0,tiny]. The exact normalized
        // sum tiny/2 must survive until division by the physical step 1/2.
        assert_eq!(result.values().unwrap()[0], tiny);

        let scioxide_core::FloatDecomposition::Finite {
            significand: bits,
            exponent,
            ..
        } = tiny.decompose()
        else {
            unreachable!("caller established a finite scalar before decomposition");
        };
        let small_exponent = i32::from(exponent) + (63 - bits.leading_zeros()) as i32;
        let scioxide_core::FloatDecomposition::Finite {
            significand: bits,
            exponent,
            ..
        } = max.decompose()
        else {
            unreachable!("caller established a finite scalar before decomposition");
        };
        let large_exponent = i32::from(exponent) + (63 - bits.leading_zeros()) as i32;
        let large = S::ONE.scalbn(large_exponent);
        let method = FiniteDifference::default()
            .with_step(DifferenceStep::absolute_coordinates(&[tiny, large]).unwrap());
        let result = method
            .hessian(
                |x: &[S], y: &mut [S]| {
                    y[0] = (x[0] / tiny) * (x[1] / large);
                    Ok(())
                },
                &[S::ZERO; 2],
                None,
                None,
            )
            .unwrap();
        let exact = S::ONE.scalbn(-small_exponent - large_exponent);
        assert_eq!(result.values().unwrap(), &[S::ZERO, exact, exact, S::ZERO]);
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn composed_complex_hessian_samples_each_base_gradient_once() {
    let point = [0.0, 0.0];
    let nodes = [-0.5, 0.0, 0.5];
    let plan = ComplexStepPlan::new(
        &point,
        &[&nodes, &nodes],
        ComplexStep::default().with_step(1e-30),
    )
    .unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let mut output = [0.0; 4];
    let mut points = Vec::new();
    let report = plan
        .evaluate_into(
            |x: &[Complex64], y: &mut [Complex64]| {
                points.push(
                    x.iter()
                        .map(|z| (z.re.to_bits(), z.im.to_bits()))
                        .collect::<Vec<_>>(),
                );
                y[0] = x[0] * x[0] + x[0] * x[1] + x[1] * x[1];
                Ok(())
            },
            None,
            &mut output,
            &mut workspace,
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.evaluations(), 11); // old: 1 + 2*6 = 13
    points.sort();
    points.dedup();
    assert_eq!(points.len(), 11);
    assert_eq!(output, [2.0, 1.0, 1.0, 2.0]);
    plan.evaluate_into(
        |x: &[Complex64], y: &mut [Complex64]| {
            y[0] = x[0] * x[0] + x[0] * x[1] + x[1] * x[1];
            Ok(())
        },
        None,
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential(),
    )
    .unwrap();
}

#[test]
fn signed_zero_full_points_are_distinct_in_both_precisions() {
    fn check<S: RealScalar>() {
        for n in [2, 6] {
            for zeros in [0, 1, n] {
                let mut point = vec![S::ZERO; n];
                for value in point.iter_mut().take(zeros) {
                    *value = -S::ZERO;
                }
                for scheme in [DifferenceScheme::Forward, DifferenceScheme::Central] {
                    let method = FiniteDifference::default().with_scheme(scheme).with_step(
                        DifferenceStep::absolute(S::try_from_f64_exact(0.25).unwrap()).unwrap(),
                    );
                    let mut workspace =
                        FiniteDifferenceWorkspace::workspace(DerivativeRequest::new(
                            &FiniteDifference::default(),
                            DerivativeShape::hessian(n),
                        ))
                        .unwrap();
                    let mut output = vec![S::ZERO; n * n];
                    let mut points = Vec::new();
                    let report = method
                        .hessian_into(
                            |x: &[S], y: &mut [S]| {
                                points.push(
                                    x.iter().map(|v| v.into_f64().to_bits()).collect::<Vec<_>>(),
                                );
                                y[0] = x.iter().fold(S::ZERO, |sum, &v| sum + v * v);
                                Ok(())
                            },
                            &point,
                            None,
                            None,
                            &mut output,
                            &mut workspace,
                            scioxide_core::ExecutionContext::sequential(),
                        )
                        .unwrap();
                    let expected = match scheme {
                        DifferenceScheme::Central => 1 + 2 * n * n + 2 * zeros * (n - 1),
                        DifferenceScheme::Forward => 1 + 2 * n + n * (n - 1) / 2 + zeros * (n - 1),
                    };
                    assert_eq!(report.evaluations(), expected);
                    points.sort();
                    points.dedup();
                    assert_eq!(points.len(), expected);
                }
            }
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn adaptive_executor_reuses_samples_with_unchanged_contraction_and_reports() {
    use std::sync::Mutex;
    let method =
        AdaptiveDifference::<f64>::default().with_execution(scioxide_core::ExecutionPolicy::Auto);
    let reference = method
        .hessian(
            |x: &[f64], y: &mut [f64]| {
                y[0] = scalar(x);
                Ok(())
            },
            &[1.0, 2.0],
            None,
        )
        .unwrap();
    let mut workspace = AdaptiveDifferenceWorkspace::workspace(DerivativeRequest::new(
        &method,
        DerivativeShape::hessian(2),
    ))
    .unwrap();
    #[cfg(not(feature = "std"))]
    let executors: Vec<Box<dyn Executor>> = vec![Box::new(SequentialExecutor)];
    #[cfg(feature = "std")]
    let executors: Vec<Box<dyn Executor>> = vec![
        Box::new(SequentialExecutor),
        Box::new(ThreadExecutor::new(
            core::num::NonZeroUsize::new(2).unwrap(),
        )),
    ];
    for executor in executors {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let identity = SolveIdentity::new(
            ProblemToken::new(1),
            scioxide_core::algorithm_id!("Hessian", "executor", 1),
            0,
            0,
        );
        let mut context = state
            .begin_solve(
                ExecutionContext::new(executor.as_ref()),
                identity,
                &mut budgets,
            )
            .unwrap();
        let points = Mutex::new(Vec::new());
        let mut output = [0.0; 4];
        let mut errors = [0.0; 4];
        let report = method
            .hessian_into(
                scioxide_differentiate::Parallel(|x: &[f64], y: &mut [f64]| {
                    points
                        .lock()
                        .unwrap()
                        .push(x.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    y[0] = scalar(x);
                    Ok(())
                }),
                &[1.0, 2.0],
                None,
                &mut output,
                &mut errors,
                &mut workspace,
                &mut context,
            )
            .unwrap();
        let mut points = points.into_inner().unwrap();
        let count = points.len();
        points.sort();
        points.dedup();
        assert_eq!(count, points.len());
        assert_eq!(report.evaluations(), count);
        assert_eq!(report.evaluations(), reference.report().evaluations());
        assert_eq!(report.iterations(), reference.report().iterations());
        assert_eq!(
            output.map(f64::to_bits),
            reference
                .values()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
                .as_slice()
        );
        assert_eq!(
            errors.map(f64::to_bits),
            reference
                .errors()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
                .as_slice()
        );
    }
}
