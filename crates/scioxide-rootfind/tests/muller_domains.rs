use core::cell::Cell;
use scioxide_core::{
    Budget, BudgetKind, Complex, EvaluationRole, ExecutionContext, ProblemToken, ResidualTolerance,
    SciOxideError, SolveBudgets, SolveState, StepTolerance, Tolerance,
};
use scioxide_rootfind::{
    ComplexMuller, ComplexRootSolver, ComplexScalarFn, ComplexScalarFunction, Muller,
    RootTermination, ScalarFn, ScalarRootOptions, ScalarRootSolver,
};

macro_rules! numerical_cases {
    ($name:ident, $s:ty, $target:expr, $large:expr, $small:expr) => {
        mod $name {
            use super::*;
            fn options() -> ScalarRootOptions<$s> {
                ScalarRootOptions::default()
                    .with_residual_tolerance(ResidualTolerance::from_tolerance(
                        Tolerance::new($target, 0.0).unwrap(),
                    ))
                    .with_step_tolerance(StepTolerance::from_tolerance(
                        Tolerance::new(0.0, 0.0).unwrap(),
                    ))
            }
            #[test]
            fn factored_real_and_complex_roots_have_independent_residuals() {
                for (first, second) in [
                    (Complex::new(1.0, 0.0), Complex::new(-2.0, 0.0)),
                    (Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)),
                    (Complex::new(0.0, 1.0), Complex::new(0.0, -1.0)),
                    (Complex::new(1.0, $small), Complex::new(1.0, -$small)),
                    (Complex::new(1.0, 2.0), Complex::new(-2.0, 0.5)),
                ] {
                    let f = ComplexScalarFn::new(|z: Complex<$s>| Ok((z - first) * (z - second)));
                    let initial = [
                        Complex::new(-0.5, 0.0),
                        Complex::new(0.5, 0.0),
                        Complex::new(2.0, 0.0),
                    ];
                    let result = ComplexMuller
                        .solve(&f, (initial[0], initial[1], initial[2]), &options())
                        .unwrap();
                    assert!(result.is_success(), "{result:?}");
                    // Independent factor distances: not the solver's discriminant or its cached f(x).
                    let z = result.point();
                    let residual = (z - first).norm() * (z - second).norm();
                    let scale = initial
                        .iter()
                        .map(|z| (*z - first).norm() * (*z - second).norm())
                        .fold(0.0 as $s, <$s>::max);
                    assert!(residual / scale <= $target, "{result:?}: {residual}");
                    assert!(f.eval(z).unwrap().norm() / scale <= $target);
                    if (first - second).norm() > 0.1 {
                        assert!((z - first).norm().min((z - second).norm()) <= 32.0 * $target);
                    }
                }
            }
            #[test]
            fn real_roots_can_require_complex_intermediates() {
                // x^3 - 2x + 2: negative first discriminant, then a real root.
                let crossed = Cell::new(false);
                let f = ComplexScalarFn::new(|z: Complex<$s>| {
                    crossed.set(crossed.get() || z.im != 0.0);
                    Ok(z * z * z - z * 2.0 + Complex::new(2.0, 0.0))
                });
                let result = ComplexMuller
                    .solve(
                        &f,
                        (
                            Complex::new(-4.0, 0.0),
                            Complex::new(-3.0, 0.0),
                            Complex::new(-2.0, 0.0),
                        ),
                        &options(),
                    )
                    .unwrap();
                assert!(result.is_success(), "{result:?}");
                let z = result.point();
                assert!(crossed.get());
                assert!(z.im.abs() <= 128.0 * $target);
                assert!(f.eval(z).unwrap().norm() / 54.0 <= $target);
            }
            #[test]
            fn extended_range_spacings_and_divided_differences() {
                for scale in [$large, 1.0 / $large] {
                    let f = ScalarFn::new(|x: $s| Ok(x / scale - 0.25));
                    let result = Muller.solve(&f, (-scale, 0.0, scale), &options()).unwrap();
                    assert!(result.is_success(), "{result:?}");
                    assert!((result.point() / scale - 0.25).abs() <= $target);
                }
                let f = ScalarFn::new(|x: $s| Ok(x / <$s>::MAX - 0.25));
                let result = Muller
                    .solve(&f, (-<$s>::MAX, 0.0, <$s>::MAX), &options())
                    .unwrap();
                assert!(result.is_success(), "{result:?}");
                assert!((result.point() / <$s>::MAX - 0.25).abs() <= $target);
            }
            #[test]
            fn cancellation_and_degenerate_interpolation() {
                let f = ScalarFn::new(|x: $s| Ok((x - 1.0) * (x + 100.0)));
                let result = Muller.solve(&f, (0.0, 0.25, 0.5), &options()).unwrap();
                assert!(result.is_success(), "{result:?}");
                assert!(
                    ((result.point() - 1.0) * (result.point() + 100.0)).abs() / 100.0 <= $target
                );
                for initial in [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 2.0)] {
                    let result = Muller
                        .solve(&ScalarFn::new(|_: $s| Ok(1.0)), initial, &options())
                        .unwrap();
                    assert_eq!(result.termination(), RootTermination::Stagnation);
                    assert!(!result.is_success());
                }
            }
        }
    };
}
numerical_cases!(single, f32, 1e-5, 1e35, 1e-3);
numerical_cases!(double, f64, 1e-12, 1e300, 1e-7);

#[test]
fn real_domain_stop_is_typed_and_never_calls_a_complex_point() {
    let calls = Cell::new(0);
    let f = ScalarFn::new(|x: f64| {
        calls.set(calls.get() + 1);
        Ok(x * x + 1.0)
    });
    let result = Muller
        .solve(&f, (-1.0, 0.0, 1.0), &ScalarRootOptions::default())
        .unwrap();
    assert_eq!(result.termination(), RootTermination::ComplexStepRequired);
    assert!(!result.is_success());
    assert_eq!(calls.get(), 3);
    assert_eq!(result.residual(), result.point() * result.point() + 1.0);
}

#[test]
fn complex_budget_failure_and_epoch_contracts() {
    let calls = Cell::new(0);
    let f = ComplexScalarFn::new(|z: Complex<f64>| {
        calls.set(calls.get() + 1);
        Ok(z * z + Complex::new(1.0, 0.0))
    });
    let init = (
        Complex::new(-1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
    );
    for count in [1, 2, 3] {
        let options = ScalarRootOptions::default().with_budgets(
            SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(count).unwrap()),
        );
        // Every call starts a fresh epoch with the full allowance.
        for _ in 0..2 {
            calls.set(0);
            let result = ComplexMuller.solve(&f, init, &options).unwrap();
            assert_eq!(
                result.termination(),
                RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
            );
            assert_eq!(result.counts().provider(EvaluationRole::Function), count);
            assert_eq!(calls.get(), count);
        }
    }
    let failing = ComplexScalarFn::new(|_: Complex<f64>| Ok(Complex::new(f64::NAN, 0.0)));
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("rootfind-test", "parent", 1),
        1,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    assert_eq!(
        ComplexMuller
            .solve_scoped(&failing, init, &ScalarRootOptions::default(), &mut context)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    assert_eq!(
        context.counts().nested().provider(EvaluationRole::Function),
        1
    );
    calls.set(0);
    assert_eq!(
        ComplexMuller
            .solve(
                &f,
                (Complex::new(f64::INFINITY, 0.0), init.1, init.2),
                &ScalarRootOptions::default()
            )
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    assert_eq!(calls.get(), 0);
}

#[test]
fn complex_cancellation_preserves_the_last_evaluated_state() {
    let cancellation = scioxide_core::CancellationToken::new();
    let f = ComplexScalarFn::new(|z: Complex<f64>| {
        if z.re == 1.0 {
            cancellation.cancel();
        }
        Ok(z * z + Complex::new(1.0, 0.0))
    });
    let result = ComplexMuller
        .solve_scoped(
            &f,
            (
                Complex::new(-1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
            ),
            &ScalarRootOptions::default(),
            ExecutionContext::sequential().with_cancellation(&cancellation),
        )
        .unwrap();
    assert_eq!(result.termination(), RootTermination::Cancelled);
    assert_eq!(result.counts().provider(EvaluationRole::Function), 3);
    assert_eq!(
        result.residual(),
        result.point() * result.point() + Complex::new(1.0, 0.0)
    );
}

#[test]
fn erased_complex_callback_and_injected_executor_match_convenience() {
    let f = ComplexScalarFn::new(|z: Complex<f64>| Ok(z * z + Complex::new(1.0, 0.0)));
    let erased: &dyn ComplexScalarFunction<f64> = &f;
    let options = ScalarRootOptions::default();
    let init = (
        Complex::new(-1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
    );
    let expected = ComplexMuller.solve(erased, init, &options).unwrap();
    let executor = scioxide_core::SequentialExecutor;
    let actual = ComplexMuller
        .solve_scoped(erased, init, &options, ExecutionContext::new(&executor))
        .unwrap();
    assert!(actual.is_success());
    assert_eq!(actual.point(), expected.point());
    assert_eq!(actual.counts(), expected.counts());
    assert!(erased.eval(actual.point()).unwrap().norm() <= 1e-12);
}

#[test]
fn complex_resource_stops_and_callback_errors_do_not_claim_success() {
    let calls = Cell::new(0);
    let f = ComplexScalarFn::new(|z: Complex<f64>| {
        calls.set(calls.get() + 1);
        if calls.get() == 4 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("test_callback"),
                scioxide_core::InvalidInputReason::Malformed,
                "injected error",
            ));
        }
        Ok(z * z * z + Complex::new(1.0, 0.0))
    });
    let options = ScalarRootOptions::default();
    let init = (
        Complex::new(0.0, 0.0),
        Complex::new(0.5, 0.0),
        Complex::new(1.0, 0.0),
    );
    assert_eq!(
        ComplexMuller.solve(&f, init, &options).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
    assert_eq!(calls.get(), 4);
    let f = ComplexScalarFn::new(|z: Complex<f64>| Ok(z * z * z + Complex::new(1.0, 0.0)));
    let options = options.with_max_iterations(Budget::ONE);
    let result = ComplexMuller.solve(&f, init, &options).unwrap();
    assert_eq!(result.termination(), RootTermination::IterationLimit);
    assert!(!result.is_success());
    assert_eq!(result.counts().provider(EvaluationRole::Function), 4);
}

#[test]
fn complex_output_keeps_a_tiny_imaginary_component() {
    let root = Complex::new(1.0, 1e-300);
    let f = ComplexScalarFn::new(|z: Complex<f64>| Ok(z - root));
    let options = ScalarRootOptions::default();
    let result = ComplexMuller
        .solve(
            &f,
            (
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
            ),
            &options,
        )
        .unwrap();
    assert!(result.is_success());
    assert_eq!(result.point().re, root.re);
    assert!((result.point().im / root.im - 1.0).abs() <= 8.0 * f64::EPSILON);
    assert!(f.eval(result.point()).unwrap().norm() <= 1e-12);
}
