use scioxide_core::{Interval, Tolerance};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_rootfind::{
    BacktrackingLineSearch, Brent, DenseJacobian, NewtonSystem, PolynomialRootMethod,
    PolynomialRootOptions, ResidualFn, RootTermination, ScalarFn, ScalarFunction,
    ScalarRootOptions, ScalarRootSolver, SystemRootSolver,
};

#[test]
fn crate_root_supports_concrete_and_erased_scalar_dispatch() {
    use scioxide_rootfind::{ScalarRootInput, ScalarRootMethod};
    let function = ScalarFn::new(|x: f64| Ok(x * x - 2.0));
    let erased_function: &dyn ScalarFunction<f64> = &function;
    let options = ScalarRootOptions::default()
        .with_residual_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
            Tolerance::new(1.0e-12, 1.0e-12).unwrap(),
        ))
        .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(
            Tolerance::new(1.0e-12, 1.0e-12).unwrap(),
        ));
    let bracket = Interval::new(1.0, 2.0).unwrap();

    let result = Brent.solve(erased_function, bracket, &options).unwrap();
    let selected = ScalarRootMethod::Brent
        .solve(&function, ScalarRootInput::Bracket(bracket), &options)
        .unwrap();

    assert!(result.is_success());
    assert!(result.termination().is_success());
    assert!((result.point() - core::f64::consts::SQRT_2).abs() < 1.0e-10);
    assert!(result.residual().abs() < 1.0e-10);
    assert_eq!(selected.point(), result.point());
    assert_eq!(selected.counts(), result.counts());
}

#[test]
fn crate_root_supports_in_place_system_callbacks_and_erased_solvers() {
    let function = ResidualFn::new(1, |point: &[f64], output: &mut [f64]| {
        output[0] = point[0] - 2.0;
        Ok(())
    })
    .with_jacobian(|_point: &[f64], output: &mut [f64]| {
        output[0] = 1.0;
        Ok(())
    });
    let erased_function: &dyn DenseJacobian<f64> = &function;
    let solver = &NewtonSystem::new(
        (scioxide_rootfind::NewtonSystemOptions::default()
            .with_line_search(BacktrackingLineSearch::<f64>::default()))
        .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap();

    let result = solver.solve(erased_function, &[0.0]).unwrap();

    assert_eq!(result.termination(), RootTermination::ResidualTolerance);
    assert_eq!(result.point(), &[2.0]);
    assert_eq!(result.residual(), &[0.0]);
    assert!(
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::Residual)
            > 0
    );
}

#[test]
fn crate_root_selects_explicit_polynomial_methods() {
    let options = PolynomialRootOptions::default().with_polish(true);
    let roots = scioxide_rootfind::polynomial_roots_with_options(
        &[2.0_f64, -3.0, 1.0],
        &options.with_method(PolynomialRootMethod::Companion),
    )
    .unwrap();
    let mut real_parts: Vec<_> = roots.into_iter().map(|root| root.re).collect();
    real_parts.sort_by(f64::total_cmp);

    assert_eq!(real_parts.len(), 2);
    assert!((real_parts[0] - 1.0).abs() < 1.0e-10);
    assert!((real_parts[1] - 2.0).abs() < 1.0e-10);
}

#[test]
fn special_zero_root_exports() {
    use scioxide_rootfind::{
        RootTermination, SpecialZero, SpecialZeroBranch, SpecialZeroFamily, SpecialZeroOptions,
        SpecialZeroReport, SpecialZeroWorkspace, SpecialZeros,
    };
    let _ = core::mem::size_of::<(
        SpecialZero<f32>,
        SpecialZeroBranch,
        SpecialZeroFamily<f32>,
        SpecialZeroOptions<f32>,
        SpecialZeroReport<f32>,
        RootTermination,
        SpecialZeroWorkspace<f32>,
        SpecialZeros<f32>,
    )>();
}

#[test]
fn errors_report_arguments_and_preserve_classification() {
    use scioxide_core::{Budget, BudgetKind, SolveBudgets, argument};
    use scioxide_rootfind::{
        Bisection, BracketSearchOptions, ContinuationOptions, FixedPointOptions, Halley,
        HistoryRestart, NewtonKrylovOptions, ScalarRootInput, ScalarRootMethod, SpecialZeroBranch,
        SpecialZeroFamily, SpecialZeroOptions, find_bracket_with_options,
        special_zeros_with_options,
    };

    let function = ScalarFn::new(|x: f64| Ok(x * x - 2.0));
    let options = ScalarRootOptions::default();

    // Shared bracket validation identifies the invalid initial bracket.
    let error = Bisection
        .solve(&function, Interval::new(2.0, 3.0).unwrap(), &options)
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::InvalidInput { argument, .. }
            if argument == argument!("init")
    ));

    // Runtime selection rejects input geometry that does not fit the method.
    let error = ScalarRootMethod::Newton(DerivativeStrategy::Analytic)
        .solve(
            &function,
            ScalarRootInput::Bracket(Interval::new(0.0, 2.0).unwrap()),
            &options,
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::InvalidInput { argument, .. }
            if argument == argument!("init")
    ));

    // A missing analytic capability is attributed to the callback argument.
    let first_only = ScalarFn::new(|x: f64| Ok(x * x - 2.0)).with_derivative(|x: f64| Ok(2.0 * x));
    let error = Halley.solve(&first_only, 1.0, &options).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::InvalidInput { argument, .. }
            if argument == argument!("f")
    ));

    // Budget stops raised inside the shared session retain their classification.
    let bracket_options = BracketSearchOptions::default()
        .with_budgets(SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::ONE));
    let error = find_bracket_with_options(
        &function,
        scioxide_rootfind::BracketSearchInput::Guess(1.0),
        scioxide_rootfind::BracketSearchDirection::Automatic,
        &bracket_options,
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::ConvergenceFailure
    );

    // Shared validation rejects invalid expansion factors.
    for error in [
        scioxide_rootfind::find_bracket_scoped(
            &function,
            scioxide_rootfind::BracketSearchInput::Guess(1.0),
            scioxide_rootfind::BracketSearchDirection::Automatic,
            &BracketSearchOptions::default()
                .with_max_expansions(Budget::ONE)
                .with_expansion_factor(0.5_f64),
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap_err(),
        find_bracket_with_options(
            &function,
            scioxide_rootfind::BracketSearchInput::Guess(1.0),
            scioxide_rootfind::BracketSearchDirection::Automatic,
            &BracketSearchOptions::<f64>::default().with_expansion_factor(0.5),
        )
        .unwrap_err(),
    ] {
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput { argument, .. }
                if argument == argument!("expansion_factor")
        ));
    }

    // Arguments name the offending parameter rather than a configuration placeholder.
    for (error, expected_argument) in [
        (
            NewtonSystem::new(scioxide_rootfind::NewtonSystemOptions::default())
                .unwrap()
                .solve(&ResidualFn::new(0, |_: &[f64], _: &mut [f64]| Ok(())), &[])
                .map(|_| ())
                .unwrap_err(),
            argument!("function"),
        ),
        (
            scioxide_rootfind::polynomial_roots_with_options(
                &[1.0_f64],
                &PolynomialRootOptions::default(),
            )
            .map(|_| ())
            .unwrap_err(),
            argument!("input"),
        ),
        (
            scioxide_rootfind::NewtonKrylovSolver::<f64>::new(
                NewtonKrylovOptions::<f64>::default().with_backtracking(0.5, 2.0),
            )
            .map(|_| ())
            .unwrap_err(),
            argument!("options.minimum_step"),
        ),
        (
            scioxide_rootfind::FixedPointSolver::<f64>::new(
                scioxide_rootfind::FixedPointMethod::Anderson,
                scioxide_rootfind::FixedPointInput::Map,
                FixedPointOptions::<f64>::default().with_history(0, HistoryRestart::Oldest),
            )
            .map(|_| ())
            .unwrap_err(),
            argument!("options.history"),
        ),
        (
            ContinuationOptions::<f64>::new(0.1, 1.0e-6, 0.15, Budget::ONE)
                .map(|_| ())
                .unwrap_err(),
            argument!("max_points"),
        ),
        (
            special_zeros_with_options(
                SpecialZeroFamily::BesselJ(0.0_f64),
                SpecialZeroBranch::Real,
                0,
                1,
                &SpecialZeroOptions::default(),
            )
            .map(|_| ())
            .unwrap_err(),
            argument!("start"),
        ),
    ] {
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::InvalidInput { argument, .. }
                if argument == expected_argument
        ));
    }
}
