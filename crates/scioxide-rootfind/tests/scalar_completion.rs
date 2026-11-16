use scioxide_core::{
    Budget, BudgetKind, EvaluationRole, ExecutionContext, Interval, ProblemToken,
    ResidualTolerance, SciOxideError, SolveBudgets, SolveState, StepTolerance, Tolerance,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_rootfind::{
    Brent, Brenth, Chandrupatla, Halley, Newton, RootTermination, SafeguardedHalley,
    SafeguardedInitialGuess, SafeguardedNewton, ScalarFn, ScalarRootInput, ScalarRootMethod,
    ScalarRootOptions, ScalarRootSolver, Toms748,
};

/// The interpolating bracket methods, selected at run time.
const INTERPOLATING: [ScalarRootMethod; 4] = [
    ScalarRootMethod::Brent,
    ScalarRootMethod::Brenth,
    ScalarRootMethod::Chandrupatla,
    ScalarRootMethod::Toms748,
];

fn evaluations(result: &scioxide_rootfind::RootResult<impl scioxide_core::RealScalar>) -> usize {
    result.counts().provider(EvaluationRole::Function)
}

macro_rules! real_cases {
    ($module:ident, $s:ty, $tol:expr) => {
        mod $module {
            use super::*;
            use scioxide_core::RealScalar;
            #[test]
            fn real_methods_check_residuals_and_brackets() {
                let f = ScalarFn::new(|x: $s| Ok(x * x - 2.0))
                    .with_derivative(|x| Ok(2.0 * x))
                    .with_second_derivative(|_| Ok(2.0));
                let options = ScalarRootOptions::default()
                    .with_residual_tolerance(ResidualTolerance::from_tolerance(
                        Tolerance::new($tol, 0.0).unwrap(),
                    ))
                    .with_step_tolerance(StepTolerance::from_tolerance(
                        Tolerance::new(0.0, 0.0).unwrap(),
                    ));
                let interval = Interval::new(0.0, 2.0).unwrap();
                let init = SafeguardedInitialGuess::new(interval, 0.0).unwrap();
                let results = [
                    Brent.solve(&f, interval, &options),
                    Brenth.solve(&f, interval, &options),
                    Chandrupatla.solve(&f, interval, &options),
                    Toms748.solve(&f, interval, &options),
                    SafeguardedNewton::new(DerivativeStrategy::Analytic).solve(&f, init, &options),
                    SafeguardedHalley.solve(&f, init, &options),
                    Halley.solve(&f, 1.0, &options),
                    Newton::new(DerivativeStrategy::Analytic).solve(&f, 1.0, &options),
                ];
                for result in results {
                    let r = result.unwrap();
                    assert!(r.is_success(), "{r:?}");
                    assert!((r.point() * r.point() - 2.0).abs() <= $tol);
                    assert!((r.point() - (2.0 as $s).sqrt()).abs() <= $tol);
                    if let Some(b) = r.bracket() {
                        assert!(b.lower() <= r.point() && r.point() <= b.upper());
                    }
                }
            }
            #[test]
            fn adjacent_floats_and_multiple_roots() {
                let a = 1.0 as $s;
                let b = <$s>::from_bits(a.to_bits() + 1);
                let f = ScalarFn::new(move |x: $s| Ok(if x == a { -1.0 } else { 1.0 }));
                let options = ScalarRootOptions::default()
                    .with_residual_tolerance(ResidualTolerance::from_tolerance(
                        Tolerance::new(0.0, 0.0).unwrap(),
                    ))
                    .with_step_tolerance(StepTolerance::from_tolerance(
                        Tolerance::new(0.0, 0.0).unwrap(),
                    ));
                for method in INTERPOLATING {
                    let r = method
                        .solve(
                            &f,
                            ScalarRootInput::Bracket(Interval::new(a, b).unwrap()),
                            &options,
                        )
                        .unwrap();
                    assert_eq!(r.termination(), RootTermination::Stagnation);
                    assert_eq!(evaluations(&r), 2);
                }
                let multiple = ScalarFn::new(|x: $s| Ok((x - 1.0).powi(3)))
                    .with_derivative(|x: $s| Ok(3.0 * (x - 1.0).powi(2)))
                    .with_second_derivative(|x| Ok(6.0 * (x - 1.0)));
                let r = Halley
                    .solve(&multiple, 1.5, &ScalarRootOptions::default())
                    .unwrap();
                assert!(r.is_success());
                assert!((r.point() - 1.0).powi(3).abs() <= 2.0 * <$s>::SQRT_EPSILON);
            }
        }
    };
}
real_cases!(single, f32, 5e-7);
real_cases!(double, f64, 5e-14);

#[test]
fn tolerance_roles_use_explicit_function_units() {
    let f = ScalarFn::new(|_: f64| Ok(1e-10)).with_derivative(|_| Ok(0.0));
    let loose = ScalarRootOptions::default().with_residual_tolerance(
        ResidualTolerance::from_tolerance(Tolerance::new(1e-9, 0.0).unwrap()),
    );
    assert!(
        Newton::new(DerivativeStrategy::Analytic)
            .solve(&f, 0.0, &loose)
            .unwrap()
            .is_success()
    );
    let strict = loose.with_residual_tolerance(ResidualTolerance::from_tolerance(
        Tolerance::new(0.0, 1e-12).unwrap(),
    ));
    assert_eq!(
        Newton::new(DerivativeStrategy::Analytic)
            .solve(&f, 0.0, &strict)
            .unwrap()
            .termination(),
        RootTermination::Stagnation
    );
    assert!(
        Newton::new(DerivativeStrategy::Analytic)
            .solve(
                &f,
                0.0,
                &ScalarRootOptions::default().with_residual_scale(0.0)
            )
            .is_err()
    );
    assert!(
        Newton::new(DerivativeStrategy::Analytic)
            .solve(
                &f,
                0.0,
                &ScalarRootOptions::default().with_residual_scale(f64::NAN)
            )
            .is_err()
    );
}
#[test]
fn every_toms_substep_respects_evaluation_and_proposal_budgets() {
    for limit in 1..12 {
        let calls = core::cell::Cell::new(0);
        let f = ScalarFn::new(|x: f64| {
            calls.set(calls.get() + 1);
            Ok(x.cos() - x)
        });
        let options =
            ScalarRootOptions::default().with_budgets(SolveBudgets::unbounded().with_limit(
                BudgetKind::Evaluation,
                Budget::ONE.saturating_add(limit - 1),
            ));
        let r = Toms748
            .solve(&f, Interval::new(0.0, 1.0).unwrap(), &options)
            .unwrap();
        assert_eq!(evaluations(&r), calls.get());
        assert!(calls.get() <= limit);
        assert_eq!(r.residual(), r.point().cos() - r.point());
        if !r.is_success() {
            assert_eq!(
                r.termination(),
                RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
            );
        }
    }
    let f = ScalarFn::new(|x: f64| Ok(x.cos() - x));
    let options = ScalarRootOptions::default()
        .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Proposal, Budget::ONE));
    let r = Toms748
        .solve(&f, Interval::new(0.0, 1.0).unwrap(), &options)
        .unwrap();
    assert_eq!(
        r.termination(),
        RootTermination::WorkLimit(scioxide_core::BudgetKind::Proposal)
    );
    assert_eq!(evaluations(&r), 3);
}
#[test]
fn derivative_failures_are_counted_and_never_swallowed() {
    let calls = core::cell::Cell::new(0);
    let f = ScalarFn::new(|x: f64| Ok(x * x - 2.0))
        .with_derivative(|x| Ok(2.0 * x))
        .with_second_derivative(|_| {
            calls.set(calls.get() + 1);
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("test_derivative"),
                scioxide_core::InvalidInputReason::Malformed,
                "provider failure",
            ))
        });
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
        Halley
            .solve_scoped(&f, 1.0, &ScalarRootOptions::default(), &mut context)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
    assert_eq!(calls.get(), 1);
    assert_eq!(
        context.counts().nested().provider(EvaluationRole::Hessian),
        1
    );
    let unit = Interval::new(0.0, 1.0).unwrap();
    assert_eq!(
        SafeguardedInitialGuess::new(unit, 2.0).unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
    // Non-finite seeds are `NonFinite`, as for every other scalar seed.
    for guess in [f64::NAN, f64::INFINITY] {
        assert_eq!(
            SafeguardedInitialGuess::new(unit, guess)
                .unwrap_err()
                .kind(),
            scioxide_core::SciOxideErrorKind::NonFinite
        );
    }
}

macro_rules! complex_cases {
    ($module:ident,$s:ty,$tol:expr) => {
        mod $module {
            use super::*;
            use scioxide_core::Complex;
            use scioxide_rootfind::{
                ComplexHalley, ComplexNewton, ComplexRootSolver, ComplexScalarFn,
                ComplexScalarFunction, ComplexSecant,
            };
            #[test]
            fn analytic_conjugate_and_repeated_roots() {
                let options = ScalarRootOptions::default()
                    .with_residual_tolerance(ResidualTolerance::from_tolerance(
                        Tolerance::new($tol, 0.0).unwrap(),
                    ))
                    .with_step_tolerance(StepTolerance::from_tolerance(
                        Tolerance::new(0.0, 0.0).unwrap(),
                    ));
                for (a, b) in [
                    (Complex::new(0.0, 1.0), Complex::new(0.0, -1.0)),
                    (Complex::new(1.0, 0.25), Complex::new(1.0, 0.25)),
                ] {
                    let f = ComplexScalarFn::new(|z: Complex<$s>| Ok((z - a) * (z - b)))
                        .with_derivative(|z| Ok(z + z - a - b))
                        .with_second_derivative(|_| Ok(Complex::new(2.0, 0.0)));
                    let start = Complex::new(0.5, 0.5);
                    for result in [
                        ComplexNewton.solve(&f, start, &options),
                        ComplexHalley.solve(&f, start, &options),
                        ComplexSecant.solve(&f, (start, Complex::new(0.75, 0.75)), &options),
                    ] {
                        let r = result.unwrap();
                        assert!(r.is_success(), "{r:?}");
                        let independent = (r.point() - a) * (r.point() - b);
                        assert!(independent.norm() <= $tol, "{r:?}");
                        assert_eq!(r.residual(), f.eval(r.point()).unwrap());
                        let error = (r.point() - a).norm().min((r.point() - b).norm());
                        let limit = if a == b {
                            ($tol as $s).sqrt() * 2.0
                        } else {
                            $tol * 4.0
                        };
                        assert!(error <= limit, "{r:?}");
                    }
                }
            }
            #[test]
            fn extreme_linear_corrections_preserve_finite_roots() {
                let target = Complex::new(0.25 as $s, 0.125 as $s);
                for scale in [<$s>::MIN_POSITIVE, <$s>::MAX / 4.0] {
                    let f = ComplexScalarFn::new(|z: Complex<$s>| Ok((z - target) * scale))
                        .with_derivative(|_| Ok(Complex::new(scale, 0.0)))
                        .with_second_derivative(|_| Ok(Complex::new(0.0, 0.0)));
                    let options = ScalarRootOptions::default()
                        .with_residual_tolerance(ResidualTolerance::from_tolerance(
                            Tolerance::new(0.0, $tol).unwrap(),
                        ))
                        .with_residual_scale(scale);
                    for r in [
                        ComplexNewton.solve(&f, Complex::new(0.0, 0.0), &options),
                        ComplexHalley.solve(&f, Complex::new(0.0, 0.0), &options),
                        ComplexSecant.solve(
                            &f,
                            (Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)),
                            &options,
                        ),
                    ] {
                        let r = r.unwrap();
                        assert!(r.is_success(), "{r:?}");
                        assert!((r.point() - target).norm() <= $tol);
                    }
                }
            }
        }
    };
}
complex_cases!(complex_single, f32, 2e-6);
complex_cases!(complex_double, f64, 2e-13);

#[test]
fn complex_scoped_solves_use_executors_and_derivative_budgets() {
    use scioxide_core::Complex;
    use scioxide_rootfind::{
        ComplexHalley, ComplexRootSolver, ComplexScalarFn, ComplexScalarFunction,
    };
    let function = ComplexScalarFn::new(|z: Complex<f64>| Ok(z * z + Complex::new(1.0, 0.0)))
        .with_derivative(|z| Ok(z + z))
        .with_second_derivative(|_| Ok(Complex::new(2.0, 0.0)));
    let f: &dyn ComplexScalarFunction<f64> = &function;
    let options = ScalarRootOptions::default();
    let initial = Complex::new(0.5, 0.5);
    let reference = ComplexHalley.solve(f, initial, &options).unwrap();
    let executor = scioxide_core::SequentialExecutor;
    for _ in 0..2 {
        let r = ComplexHalley
            .solve_scoped(f, initial, &options, ExecutionContext::new(&executor))
            .unwrap();
        assert_eq!(r.point(), reference.point());
        assert_eq!(r.counts(), reference.counts());
    }
    for limit in 1..8 {
        let options =
            ScalarRootOptions::default().with_budgets(SolveBudgets::unbounded().with_limit(
                BudgetKind::Evaluation,
                Budget::ONE.saturating_add(limit - 1),
            ));
        let r = ComplexHalley.solve(f, initial, &options).unwrap();
        let counts = r.counts();
        let total = counts.provider(EvaluationRole::Function)
            + counts.provider(EvaluationRole::Jacobian)
            + counts.provider(EvaluationRole::Hessian);
        assert!(total <= limit);
        assert_eq!(r.residual(), f.eval(r.point()).unwrap());
        assert!(!r.is_success());
        assert_eq!(
            r.termination(),
            RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
        );
    }
}

#[test]
fn dispatch_validates_geometry_and_delegates_exactly() {
    let calls = core::cell::Cell::new(0);
    let f = ScalarFn::new(|x: f64| {
        calls.set(calls.get() + 1);
        Ok(x * x - 2.0)
    });
    let options = ScalarRootOptions::default();
    assert_eq!(
        ScalarRootMethod::Halley
            .solve(
                &f,
                ScalarRootInput::Bracket(Interval::new(0.0, 2.0).unwrap()),
                &options
            )
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
    assert_eq!(calls.get(), 0);
    let expected = Toms748
        .solve(&f, Interval::new(0.0, 2.0).unwrap(), &options)
        .unwrap();
    let actual = ScalarRootMethod::Toms748
        .solve(
            &f,
            ScalarRootInput::Bracket(Interval::new(0.0, 2.0).unwrap()),
            &options,
        )
        .unwrap();
    assert_eq!(actual.point(), expected.point());
    assert_eq!(actual.counts(), expected.counts());
    assert_eq!(actual.termination(), expected.termination());
    assert_eq!(actual.algorithm(), expected.algorithm());
    assert_eq!(
        ScalarRootSolver::<f64>::name(&ScalarRootMethod::Toms748),
        ScalarRootSolver::<f64>::name(&Toms748)
    );
}
#[test]
fn bracket_geometry_survives_flat_scaled_and_extreme_problems() {
    let options = ScalarRootOptions::default()
        .with_residual_tolerance(ResidualTolerance::from_tolerance(
            Tolerance::new(0.0, 0.0).unwrap(),
        ))
        .with_step_tolerance(StepTolerance::from_tolerance(
            Tolerance::new(1e-12, 1e-12).unwrap(),
        ))
        .with_max_iterations(Budget::ONE.saturating_add(511));
    for method in INTERPOLATING {
        for degree in [1, 3, 9, 19] {
            let f = ScalarFn::new(move |x: f64| Ok(x.powi(degree) - 1.0));
            let r = method
                .solve(
                    &f,
                    ScalarRootInput::Bracket(Interval::new(0.0, 2.0).unwrap()),
                    &options,
                )
                .unwrap();
            assert!(r.is_success(), "{r:?}");
            assert!((r.point() - 1.0).abs() <= 2e-12, "{r:?}");
        }
        let f = ScalarFn::new(|x: f64| Ok(x / f64::MAX - 0.25));
        let r = method
            .solve(
                &f,
                ScalarRootInput::Bracket(Interval::new(-f64::MAX, f64::MAX).unwrap()),
                &options,
            )
            .unwrap();
        assert!(r.is_success(), "{r:?}");
        assert!((r.point() / f64::MAX - 0.25).abs() < 1e-12);
    }
}

#[test]
fn finite_difference_probes_stay_inside_the_safeguarding_bracket() {
    for strategy in [DerivativeStrategy::Forward, DerivativeStrategy::Central] {
        let f = ScalarFn::new(|x: f64| {
            assert!((0.0..=2.0).contains(&x), "probe {x} left the bracket");
            Ok(x * x - 2.0)
        });
        let initial = SafeguardedInitialGuess::new(Interval::new(0.0, 2.0).unwrap(), 2.0).unwrap();
        let r = SafeguardedNewton::new(strategy)
            .solve(&f, initial, &ScalarRootOptions::default())
            .unwrap();
        assert!(r.is_success(), "{r:?}");
        assert!((r.point() - 2.0_f64.sqrt()).abs() < 1e-10);
        assert!(r.counts().provider(EvaluationRole::Jacobian) > 0);
    }
}

#[test]
fn safeguarded_derivatives_recover_fast_local_convergence_after_bisection() {
    let f = ScalarFn::new(|x: f64| Ok(x * x - 2.0))
        .with_derivative(|x| Ok(2.0 * x))
        .with_second_derivative(|_| Ok(2.0));
    let initial = SafeguardedInitialGuess::new(Interval::new(0.0, 2.0).unwrap(), 0.0).unwrap();
    let options = ScalarRootOptions::default()
        .with_residual_tolerance(ResidualTolerance::from_tolerance(
            Tolerance::new(5e-14, 0.0).unwrap(),
        ))
        .with_step_tolerance(StepTolerance::from_tolerance(
            Tolerance::new(0.0, 0.0).unwrap(),
        ));
    for result in [
        SafeguardedNewton::new(DerivativeStrategy::Analytic).solve(&f, initial, &options),
        SafeguardedHalley.solve(&f, initial, &options),
    ] {
        let r = result.unwrap();
        assert!(r.is_success(), "{r:?}");
        assert!((r.point() * r.point() - 2.0).abs() <= 5e-14);
        assert!(
            evaluations(&r) <= 10,
            "lost local derivative acceleration: {r:?}"
        );
    }
}

#[test]
fn typed_derivative_selection_preserves_newton_fallback_and_accounting() {
    use core::cell::Cell;
    struct Providers {
        analytic: bool,
        automatic: bool,
        fail: bool,
        evaluations: Cell<usize>,
        analytic_calls: Cell<usize>,
        automatic_calls: Cell<usize>,
    }
    impl scioxide_rootfind::ScalarFunction<f64> for Providers {
        fn eval(&self, x: f64) -> Result<f64, SciOxideError> {
            self.evaluations.set(self.evaluations.get() + 1);
            Ok(x * x - 2.0)
        }
        fn derivative(&self, x: f64) -> Result<Option<f64>, SciOxideError> {
            self.analytic_calls.set(self.analytic_calls.get() + 1);
            if self.fail {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("derivative"),
                    scioxide_core::InvalidInputReason::Malformed,
                    "analytic provider failed",
                ));
            }
            Ok(self.analytic.then_some(2.0 * x))
        }
        fn automatic_derivative(&self, x: f64) -> Result<Option<f64>, SciOxideError> {
            self.automatic_calls.set(self.automatic_calls.get() + 1);
            Ok(self.automatic.then_some(2.0 * x))
        }
    }
    for (analytic, automatic, fail) in [
        (true, true, false),
        (false, true, false),
        (false, false, false),
        (false, true, true),
    ] {
        let function = Providers {
            analytic,
            automatic,
            fail,
            evaluations: Cell::new(0),
            analytic_calls: Cell::new(0),
            automatic_calls: Cell::new(0),
        };
        let result = Newton::new(DerivativeStrategy::AnalyticOrAutomaticOrForward).solve(
            &function,
            1.0,
            &ScalarRootOptions::default(),
        );
        if fail {
            let error = result.unwrap_err();
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::InvalidInput {
                    reason: scioxide_core::InvalidInputReason::Malformed,
                    ..
                }
            ));
            assert_eq!(function.analytic_calls.get(), 1);
            assert_eq!(function.automatic_calls.get(), 0);
        } else {
            let result = result.unwrap();
            assert!(result.is_success());
            assert!((result.point() * result.point() - 2.0).abs() < 1e-8);
            assert_eq!(evaluations(&result), function.evaluations.get());
            assert!(function.analytic_calls.get() > 0);
            if analytic {
                assert_eq!(function.automatic_calls.get(), 0);
            } else {
                assert_eq!(
                    function.analytic_calls.get(),
                    function.automatic_calls.get()
                );
            }
            assert_eq!(
                result.counts().provider(EvaluationRole::Jacobian),
                function.analytic_calls.get()
                    + function.automatic_calls.get()
                    + if !analytic && !automatic {
                        function.analytic_calls.get()
                    } else {
                        0
                    }
            );
        }
    }
}

#[test]
fn every_scalar_family_rejects_invalid_scale_before_callbacks() {
    use core::cell::Cell;
    use scioxide_core::{Complex, Interval, SciOxideErrorView};
    use scioxide_rootfind::{
        ComplexRootSolver, ScalarRootInput as Input, ScalarRootMethod as Method,
    };
    let calls = Cell::new(0);
    let real = ScalarFn::new(|_: f64| {
        calls.set(calls.get() + 1);
        Ok(0.0)
    });
    let complex = scioxide_rootfind::ComplexScalarFn::new(|_: Complex<f64>| {
        calls.set(calls.get() + 1);
        Ok(Complex::new(0.0, 0.0))
    });
    let interval = Interval::new(0.0, 2.0).unwrap();
    let guarded = scioxide_rootfind::SafeguardedInitialGuess::new(interval, 1.0).unwrap();
    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let options = ScalarRootOptions::default().with_residual_scale(scale);
        for (method, input) in [
            (Method::Bisection, Input::Bracket(interval)),
            (Method::Brent, Input::Bracket(interval)),
            (Method::Brenth, Input::Bracket(interval)),
            (Method::Chandrupatla, Input::Bracket(interval)),
            (Method::Toms748, Input::Bracket(interval)),
            (Method::Illinois, Input::Bracket(interval)),
            (Method::RegulaFalsi, Input::Bracket(interval)),
            (Method::Ridder, Input::Bracket(interval)),
            (
                Method::Newton(DerivativeStrategy::Analytic),
                Input::Guess(1.0),
            ),
            (Method::Halley, Input::Guess(1.0)),
            (
                Method::SafeguardedNewton(DerivativeStrategy::Analytic),
                Input::Safeguarded(guarded),
            ),
            (Method::SafeguardedHalley, Input::Safeguarded(guarded)),
            (Method::Secant, Input::Pair(0.0, 2.0)),
            (Method::Muller, Input::Triple(0.0, 1.0, 2.0)),
        ] {
            let error = method.solve(&real, input, &options).unwrap_err();
            assert!(
                matches!(error.view(), SciOxideErrorView::InvalidInput { argument, .. } if argument == scioxide_core::argument!("scale"))
            );
        }
        let a = Complex::new(0.0, 0.0);
        let b = Complex::new(1.0, 0.0);
        let c = Complex::new(2.0, 0.0);
        for error in [
            scioxide_rootfind::ComplexNewton
                .solve(&complex, a, &options)
                .unwrap_err(),
            scioxide_rootfind::ComplexHalley
                .solve(&complex, a, &options)
                .unwrap_err(),
            scioxide_rootfind::ComplexSecant
                .solve(&complex, (a, b), &options)
                .unwrap_err(),
            scioxide_rootfind::ComplexMuller
                .solve(&complex, (a, b, c), &options)
                .unwrap_err(),
        ] {
            assert!(
                matches!(error.view(), SciOxideErrorView::InvalidInput { argument, .. } if argument == scioxide_core::argument!("scale"))
            );
        }
        assert_eq!(calls.get(), 0);
    }
}

#[test]
fn complex_and_real_callbacks_reject_non_finite_outputs_alike() {
    use scioxide_core::{Complex, SciOxideErrorKind};
    use scioxide_rootfind::{
        ComplexHalley, ComplexNewton, ComplexRootSolver, ComplexScalarFn, ComplexScalarFunction,
        ScalarFunction,
    };
    let view = |error: SciOxideError| {
        assert_eq!(error.kind(), SciOxideErrorKind::NonFinite);
        format!("{:?}", error.view())
    };
    let options = ScalarRootOptions::default();
    let start = Complex::new(0.5, 0.5);
    let newton = Newton::new(DerivativeStrategy::Analytic);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        // Non-finite first derivatives.
        let real = ScalarFn::new(|x: f64| Ok(x * x - 2.0))
            .with_derivative(move |_| Ok(bad))
            .with_second_derivative(|_| Ok(2.0));
        let complex = ComplexScalarFn::new(|z: Complex<f64>| Ok(z * z + Complex::new(1.0, 0.0)))
            .with_derivative(move |_| Ok(Complex::new(1.0, bad)))
            .with_second_derivative(|_| Ok(Complex::new(2.0, 0.0)));
        let expected = view(real.derivative(1.0).unwrap_err());
        assert!(expected.contains("Callback"), "{expected}");
        assert_eq!(view(complex.derivative(start).unwrap_err()), expected);
        assert_eq!(
            view(newton.solve(&real, 1.0, &options).unwrap_err()),
            expected
        );
        assert_eq!(
            view(Halley.solve(&real, 1.0, &options).unwrap_err()),
            expected
        );
        assert_eq!(
            view(ComplexNewton.solve(&complex, start, &options).unwrap_err()),
            expected
        );
        assert_eq!(
            view(ComplexHalley.solve(&complex, start, &options).unwrap_err()),
            expected
        );
        // Non-finite second derivatives.
        let real = ScalarFn::new(|x: f64| Ok(x * x - 2.0))
            .with_derivative(|x| Ok(2.0 * x))
            .with_second_derivative(move |_| Ok(bad));
        let complex = ComplexScalarFn::new(|z: Complex<f64>| Ok(z * z + Complex::new(1.0, 0.0)))
            .with_derivative(|z| Ok(z + z))
            .with_second_derivative(move |_| Ok(Complex::new(bad, 0.0)));
        assert_eq!(view(real.second_derivative(1.0).unwrap_err()), expected);
        assert_eq!(
            view(complex.second_derivative(start).unwrap_err()),
            expected
        );
        assert_eq!(
            view(Halley.solve(&real, 1.0, &options).unwrap_err()),
            expected
        );
        assert_eq!(
            view(ComplexHalley.solve(&complex, start, &options).unwrap_err()),
            expected
        );
        // Non-finite values.
        let real = ScalarFn::new(move |_: f64| Ok(bad));
        let complex = ComplexScalarFn::new(move |_: Complex<f64>| Ok(Complex::new(0.0, bad)));
        assert_eq!(view(real.eval(1.0).unwrap_err()), expected);
        assert_eq!(view(complex.eval(start).unwrap_err()), expected);
    }

    // Implementations without adapter checks are rejected identically by solvers.
    struct Real(f64, f64);
    impl ScalarFunction<f64> for Real {
        fn eval(&self, x: f64) -> Result<f64, SciOxideError> {
            Ok(if x == 1.0 { x * x - 2.0 } else { self.0 })
        }
        fn derivative(&self, _: f64) -> Result<Option<f64>, SciOxideError> {
            Ok(Some(self.1))
        }
        fn second_derivative(&self, _: f64) -> Result<Option<f64>, SciOxideError> {
            Ok(Some(self.1))
        }
    }
    struct Raw(Complex<f64>, Complex<f64>);
    impl ComplexScalarFunction<f64> for Raw {
        fn eval(&self, z: Complex<f64>) -> Result<Complex<f64>, SciOxideError> {
            Ok(if z == Complex::new(0.5, 0.5) {
                z * z
            } else {
                self.0
            })
        }
        fn derivative(&self, _: Complex<f64>) -> Result<Option<Complex<f64>>, SciOxideError> {
            Ok(Some(self.1))
        }
        fn second_derivative(
            &self,
            _: Complex<f64>,
        ) -> Result<Option<Complex<f64>>, SciOxideError> {
            Ok(Some(self.1))
        }
    }
    let expected = view(ScalarFn::new(|_: f64| Ok(f64::NAN)).eval(0.0).unwrap_err());
    let nan = Complex::new(f64::NAN, 0.0);
    let finite = Complex::new(1.0, 0.0);
    for (real, complex) in [
        (Real(0.0, f64::NAN), Raw(finite, nan)),
        (Real(f64::NAN, 1.0), Raw(nan, finite)),
    ] {
        assert_eq!(
            view(newton.solve(&real, 1.0, &options).unwrap_err()),
            expected
        );
        assert_eq!(
            view(Halley.solve(&real, 1.0, &options).unwrap_err()),
            expected
        );
        assert_eq!(
            view(ComplexNewton.solve(&complex, start, &options).unwrap_err()),
            expected
        );
        assert_eq!(
            view(ComplexHalley.solve(&complex, start, &options).unwrap_err()),
            expected
        );
    }
}
