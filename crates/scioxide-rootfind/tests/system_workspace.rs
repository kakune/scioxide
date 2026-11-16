#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]

/// A residual that only exposes its shape and evaluation; its Jacobian comes
/// from finite differences.
use scioxide_core::{Budget, ResidualOperator, SciOxideError, Tolerance};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_rootfind::{
    BacktrackingLineSearch, Broyden, BroydenOptions, BroydenUpdate, DenseJacobian, NewtonSystem,
    NewtonSystemOptions, ResidualFn, RootLineSearch, RootTermination, SystemRootResult,
    SystemRootSolver, SystemWorkspace, TrustRegionSystem, TrustRegionSystemOptions,
};
use std::cell::Cell;
use std::rc::Rc;

fn residual_only(
    dimension: usize,
    residual: impl Fn(&[f64], &mut [f64]) -> Result<(), SciOxideError>,
) -> impl DenseJacobian<f64> {
    ResidualFn::new(dimension, residual)
}

fn diagonal_f64(dimension: usize) -> impl DenseJacobian<f64> {
    ResidualFn::new(dimension, move |point: &[f64], output: &mut [f64]| {
        output
            .iter_mut()
            .zip(point)
            .enumerate()
            .for_each(|(index, (slot, &value))| {
                *slot = value * value - (index + 1) as f64;
            });
        Ok(())
    })
    .with_jacobian(move |point: &[f64], output: &mut [f64]| {
        output.fill(0.0);
        point.iter().enumerate().for_each(|(index, &value)| {
            output[index * dimension + index] = 2.0 * value;
        });
        Ok(())
    })
}

fn prepared_state<F: DenseJacobian<f64>>(
    function: &F,
) -> (Vec<f64>, SystemRootResult<f64>, SystemWorkspace<f64>) {
    let dimension = function.input_dimension();
    let initial = vec![1.0; dimension];
    let mut result = SystemRootResult::empty();
    result.try_reserve(dimension).unwrap();
    let mut workspace = SystemWorkspace::new();
    NewtonSystem::new(
        scioxide_rootfind::NewtonSystemOptions::<f64>::default()
            .with_line_search(BacktrackingLineSearch::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .workspace_requirements(function)
    .and_then(|r| workspace.try_reserve(&r))
    .unwrap();
    (initial, result, workspace)
}

#[test]
fn analytic_automatic_and_finite_difference_counters_are_exact() {
    let analytic = diagonal_f64(2);
    let (initial, mut result, mut workspace) = prepared_state(&analytic);
    NewtonSystem::new(
        (scioxide_rootfind::NewtonSystemOptions::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .solve_into(
        &analytic,
        &initial,
        &mut result,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::Residual),
        result.iterations() + 1
    );
    assert_eq!(
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::Jacobian),
        result.iterations()
    );
    assert_eq!(
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::Factorization),
        result.iterations()
    );
    assert_eq!(
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::LinearSolve),
        result.iterations()
    );

    let automatic = ResidualFn::new(1, |point: &[f64], output: &mut [f64]| {
        output[0] = point[0] * point[0] - 2.0;
        Ok(())
    })
    .with_automatic_jacobian(|point: &[f64], output: &mut [f64]| {
        output[0] = 2.0 * point[0];
        Ok(())
    });
    let (initial, mut result, mut workspace) = prepared_state(&automatic);
    NewtonSystem::new(
        scioxide_rootfind::NewtonSystemOptions::<f64>::default()
            .with_line_search(BacktrackingLineSearch::default())
            .with_derivative_strategy(DerivativeStrategy::Automatic),
    )
    .unwrap()
    .workspace_requirements(&automatic)
    .and_then(|r| workspace.try_reserve(&r))
    .unwrap();
    NewtonSystem::new(
        (scioxide_rootfind::NewtonSystemOptions::default())
            .with_derivative_strategy(DerivativeStrategy::Automatic),
    )
    .unwrap()
    .solve_into(
        &automatic,
        &initial,
        &mut result,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert!(
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::Jacobian)
            > 0
    );
    assert_eq!(
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::Factorization),
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::Jacobian)
    );
    assert_eq!(
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::LinearSolve),
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::Jacobian)
    );

    let finite_difference = ResidualFn::new(1, |point: &[f64], output: &mut [f64]| {
        output[0] = point[0] * point[0] - 2.0;
        Ok(())
    });
    let (initial, mut result, mut workspace) = prepared_state(&finite_difference);
    NewtonSystem::new(
        scioxide_rootfind::NewtonSystemOptions::<f64>::default()
            .with_line_search(BacktrackingLineSearch::default())
            .with_derivative_strategy(DerivativeStrategy::Central),
    )
    .unwrap()
    .workspace_requirements(&finite_difference)
    .and_then(|r| workspace.try_reserve(&r))
    .unwrap();
    NewtonSystem::new(
        (scioxide_rootfind::NewtonSystemOptions::default())
            .with_derivative_strategy(DerivativeStrategy::Central),
    )
    .unwrap()
    .solve_into(
        &finite_difference,
        &initial,
        &mut result,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::Residual),
        result.iterations()
            + 1
            + 2 * result
                .counts()
                .provider(scioxide_core::EvaluationRole::Jacobian)
    );
}

#[test]
fn residual_adapter_checks_shapes_and_reports_available_jacobians() {
    let function = ResidualFn::new(2, |point: &[f64], output: &mut [f64]| {
        output.copy_from_slice(point);
        Ok(())
    });
    let mut output = [7.0; 3];
    assert_eq!(
        function
            .residual_into(&[0.0], &mut output[..2], &mut [])
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        function
            .residual_into(&[0.0; 2], &mut output, &mut [])
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, [7.0; 3]);
    let mut jacobian = [7.0; 4];
    assert!(
        !function
            .jacobian_into(&[0.0; 2], &mut jacobian, &mut [])
            .unwrap()
    );
    assert!(
        !function
            .automatic_jacobian_into(&[0.0; 2], &mut jacobian, &mut [])
            .unwrap()
    );
    let analytic = function.with_jacobian(|_: &[f64], output: &mut [f64]| {
        output.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        Ok(())
    });
    assert_eq!(
        analytic
            .jacobian_into(&[0.0; 2], &mut jacobian[..3], &mut [])
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert!(
        analytic
            .jacobian_into(&[0.0; 2], &mut jacobian, &mut [])
            .unwrap()
    );
    assert_eq!(jacobian, [1.0, 0.0, 0.0, 1.0]);
    assert!(
        !analytic
            .automatic_jacobian_into(&[0.0; 2], &mut jacobian, &mut [])
            .unwrap()
    );
}

/// The Jacobian sources of a dense solve: unavailable sources fall back in
/// strategy order, and no available source is an unsupported configuration.
#[test]
fn dense_jacobian_sources_follow_the_strategy_order() {
    let residual = |point: &[f64], output: &mut [f64]| {
        output[0] = point[0] * point[0] - 2.0;
        Ok(())
    };
    let analytic = Cell::new(0);
    let automatic = Cell::new(0);
    let both = ResidualFn::new(1, residual)
        .with_jacobian(|point: &[f64], output: &mut [f64]| {
            analytic.set(analytic.get() + 1);
            output[0] = 2.0 * point[0];
            Ok(())
        })
        .with_automatic_jacobian(|point: &[f64], output: &mut [f64]| {
            automatic.set(automatic.get() + 1);
            output[0] = 2.0 * point[0];
            Ok(())
        });
    let only_automatic = ResidualFn::new(1, residual).with_automatic_jacobian(
        |point: &[f64], output: &mut [f64]| {
            automatic.set(automatic.get() + 1);
            output[0] = 2.0 * point[0];
            Ok(())
        },
    );

    let solve = |function: &dyn DenseJacobian<f64>, strategy| {
        NewtonSystem::new(
            (scioxide_rootfind::NewtonSystemOptions::default()).with_derivative_strategy(strategy),
        )
        .unwrap()
        .solve(function, &[1.0])
    };
    for strategy in [
        DerivativeStrategy::Analytic,
        DerivativeStrategy::AnalyticOrAutomaticOrForward,
    ] {
        analytic.set(0);
        automatic.set(0);
        assert!(solve(&both, strategy).unwrap().is_success());
        assert!(analytic.get() > 0);
        assert_eq!(automatic.get(), 0);
    }
    for strategy in [
        DerivativeStrategy::Automatic,
        DerivativeStrategy::AnalyticOrAutomaticOrForward,
    ] {
        automatic.set(0);
        let result = solve(&only_automatic, strategy).unwrap();
        assert!(result.is_success());
        assert_eq!(
            automatic.get(),
            result
                .counts()
                .provider(scioxide_core::EvaluationRole::Factorization)
        );
    }
    for strategy in [DerivativeStrategy::Forward, DerivativeStrategy::Central] {
        let result = solve(&residual_only(1, residual), strategy).unwrap();
        assert!(result.is_success());
        assert!((result.point()[0] - 2.0_f64.sqrt()).abs() < 1e-8);
    }
    for strategy in [DerivativeStrategy::Analytic, DerivativeStrategy::Automatic] {
        assert!(matches!(
            solve(&residual_only(1, residual), strategy)
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::InvalidInput {
                reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                ..
            }
        ));
    }
}

#[test]
fn callback_errors_and_nonfinite_values_leave_result_unchanged() {
    let calls = Cell::new(0_usize);
    let failing = ResidualFn::new(1, |point: &[f64], output: &mut [f64]| {
        let call = calls.get();
        calls.set(call + 1);
        if call > 0 {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("mid_iteration_callback_sentinel"),
                scioxide_core::InvalidInputReason::Malformed,
                "intentional failure",
            ));
        }
        output[0] = point[0] - 2.0;
        Ok(())
    })
    .with_jacobian(|_point: &[f64], output: &mut [f64]| {
        output[0] = 1.0;
        Ok(())
    });
    let mut result = SystemRootResult::empty();
    result.try_reserve(1).unwrap();
    let before = result.clone();
    let mut workspace = SystemWorkspace::new();
    NewtonSystem::new(
        scioxide_rootfind::NewtonSystemOptions::<f64>::default()
            .with_line_search(BacktrackingLineSearch::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .workspace_requirements(&failing)
    .and_then(|r| workspace.try_reserve(&r))
    .unwrap();
    let solve = NewtonSystem::new(
        (scioxide_rootfind::NewtonSystemOptions::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .solve_into(
        &failing,
        &[0.0],
        &mut result,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    );
    assert_eq!(
        solve.unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
    assert_eq!(result.point(), before.point());
    assert_eq!(result.residual(), before.residual());
    assert_eq!(result.iterations(), before.iterations());
    assert_eq!(result.residual_norm(), before.residual_norm());
    assert_eq!(result.termination(), before.termination());

    let nonfinite = ResidualFn::new(1, |_point: &[f64], output: &mut [f64]| {
        output[0] = f64::NAN;
        Ok(())
    });
    assert_eq!(
        NewtonSystem::new(
            (scioxide_rootfind::NewtonSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Forward)
        )
        .unwrap()
        .solve(&nonfinite, &[0.0])
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
}

struct ExhaustingLineSearch {
    calls: Rc<Cell<usize>>,
    cancellation: Option<Rc<scioxide_core::CancellationToken>>,
}

impl RootLineSearch<f64> for ExhaustingLineSearch {
    fn search_into(
        &self,
        function: &dyn ResidualOperator<f64>,
        point: &[f64],
        direction: &[f64],
        residual: &[f64],
        slope: scioxide_core::Scaled<f64>,
        trial_point: &mut [f64],
        trial_residual: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<Option<f64>, SciOxideError> {
        let attempt = self.calls.replace(self.calls.get() + 1);
        if attempt == 0 {
            return BacktrackingLineSearch::default().search_into(
                function,
                point,
                direction,
                residual,
                slope,
                trial_point,
                trial_residual,
                scratch,
            );
        }
        if let Some(cancellation) = &self.cancellation {
            cancellation.cancel();
        }
        Ok(None)
    }
}

#[test]
fn exhausted_broyden_retry_publishes_the_last_accepted_point_once() {
    let calls = Rc::new(Cell::new(0));
    let function = diagonal_f64(1);
    let solver = Broyden::new(
        scioxide_rootfind::BroydenUpdate::First,
        (scioxide_rootfind::BroydenOptions::default().with_line_search(ExhaustingLineSearch {
            calls: Rc::clone(&calls),
            cancellation: None,
        }))
        .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap();
    let mut result = SystemRootResult::empty();
    result.try_reserve(1).unwrap();
    let mut workspace = solver.workspace(&function).unwrap();
    let identity = scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(1809),
        scioxide_core::algorithm_id!("test", "dense-parent", 1),
        0,
        0,
    );
    let mut state = scioxide_core::SolveState::new();
    let mut budgets = scioxide_core::SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            scioxide_core::ExecutionContext::sequential(),
            identity,
            &mut budgets,
        )
        .unwrap();
    solver
        .solve_into(&function, &[2.0], &mut result, &mut workspace, &mut context)
        .unwrap();
    assert_eq!(result.termination(), RootTermination::GlobalizationFailure);
    assert!((result.point()[0] - 1.25).abs() < f64::EPSILON);
    // The stale secant step, the refreshed Newton step and its perturbed fallback
    // each exhaust once; nothing is retried after the fresh model's fallback.
    assert_eq!(calls.get(), 4);
    assert_eq!(
        context
            .counts()
            .total()
            .unwrap()
            .provider(scioxide_core::EvaluationRole::Jacobian),
        2
    );
    assert_eq!(
        workspace.last_accepted().unwrap().as_slice(),
        result.point()
    );
}

#[test]
fn broyden_refresh_respects_admission_and_cancellation() {
    for cancel in [false, true] {
        let calls = Rc::new(Cell::new(0));
        let cancellation = Rc::new(scioxide_core::CancellationToken::new());
        let function = diagonal_f64(1);
        let solver = Broyden::new(
            scioxide_rootfind::BroydenUpdate::First,
            (scioxide_rootfind::BroydenOptions::default().with_line_search(ExhaustingLineSearch {
                calls: Rc::clone(&calls),
                cancellation: cancel.then(|| Rc::clone(&cancellation)),
            }))
            .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap();

        let mut workspace = solver.workspace(&function).unwrap();
        let mut result = SystemRootResult::empty();
        result.try_reserve(1).unwrap();
        let identity = scioxide_core::SolveIdentity::new(
            scioxide_core::ProblemToken::new(1809),
            scioxide_core::algorithm_id!("test", "dense-parent", 1),
            0,
            0,
        );
        let mut state = scioxide_core::SolveState::new();
        // Eleven admitted providers precede the attempted Jacobian restart: the
        // initial residual, Jacobian and factorization; per step `QᵀF`, the
        // triangular solve and the model slope; the accepted trial residual and the
        // secant correction product.
        let mut budgets = if cancel {
            scioxide_core::SolveBudgets::unbounded()
        } else {
            scioxide_core::SolveBudgets::unbounded().with_limit(
                scioxide_core::BudgetKind::Evaluation,
                Budget::try_from(11).unwrap(),
            )
        };
        let mut context = state
            .begin_solve(
                scioxide_core::ExecutionContext::sequential()
                    .with_cancellation(cancellation.as_ref()),
                identity,
                &mut budgets,
            )
            .unwrap();
        solver
            .solve_into(&function, &[2.0], &mut result, &mut workspace, &mut context)
            .unwrap();
        assert_eq!(
            result.termination(),
            if cancel {
                RootTermination::Cancelled
            } else {
                RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
            }
        );
        assert!((result.point()[0] - 1.25).abs() <= f64::EPSILON);
        assert_eq!(
            result
                .counts()
                .provider(scioxide_core::EvaluationRole::Jacobian),
            1
        );
        assert_eq!(
            result.accounting().total().unwrap(),
            context.counts().total().unwrap()
        );
        assert_eq!(calls.get(), 2);
    }
}

#[test]
fn callback_step_size_failure_is_propagated_without_a_restart() {
    let jacobian_calls = Cell::new(0);
    let function = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
        let x = x[0];
        if x > 1.0 {
            return Err(SciOxideError::convergence_failure(
                scioxide_core::ConvergenceReason::StepSizeLimit,
                scioxide_core::ConvergenceDiagnostics::new(),
                "callback sentinel",
            ));
        }
        out[0] = x * x * x - 2.0 * x + 2.0;
        Ok(())
    })
    .with_jacobian(|x: &[f64], out: &mut [f64]| {
        jacobian_calls.set(jacobian_calls.get() + 1);
        out[0] = 3.0 * x[0] * x[0] - 2.0;
        Ok(())
    });

    let error = Broyden::new(
        scioxide_rootfind::BroydenUpdate::First,
        (scioxide_rootfind::BroydenOptions::default()
            .with_line_search(BacktrackingLineSearch::default()))
        .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .solve(&function, &[0.0])
    .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::ConvergenceFailure {
            reason: scioxide_core::ConvergenceReason::StepSizeLimit,
            detail: "callback sentinel",
            ..
        }
    ));
    assert_eq!(jacobian_calls.get(), 1);
}

struct MutatingFailedLineSearch;

impl RootLineSearch<f64> for MutatingFailedLineSearch {
    fn search_into(
        &self,
        _function: &dyn ResidualOperator<f64>,
        _point: &[f64],
        _direction: &[f64],
        _residual_norm: &[f64],
        _merit_slope: scioxide_core::Scaled<f64>,
        trial_point: &mut [f64],
        trial_residual: &mut [f64],
        _scratch: &mut [f64],
    ) -> Result<Option<f64>, SciOxideError> {
        trial_point.fill(91.0);
        trial_residual.fill(92.0);
        Err(SciOxideError::invalid_input(
            scioxide_core::argument!("line_search_sentinel"),
            scioxide_core::InvalidInputReason::Malformed,
            "intentional failure after mutating candidate storage",
        ))
    }
}

#[test]
fn line_search_failure_does_not_publish_candidate_state() {
    let function = diagonal_f64(2);
    let (initial, mut result, mut workspace) = prepared_state(&function);
    let before = result.clone();

    NewtonSystem::new(
        scioxide_rootfind::NewtonSystemOptions::<f64>::default()
            .with_line_search(BacktrackingLineSearch::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .workspace_requirements(&function)
    .and_then(|r| workspace.try_reserve(&r))
    .unwrap();
    let solve = NewtonSystem::new(
        (scioxide_rootfind::NewtonSystemOptions::default()
            .with_line_search(MutatingFailedLineSearch))
        .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .solve_into(
        &function,
        &initial,
        &mut result,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    );
    assert_eq!(
        solve.unwrap_err().kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
    assert_eq!(result.point(), before.point());
    assert_eq!(result.residual(), before.residual());
    assert_eq!(result.iterations(), before.iterations());
    assert_eq!(result.termination(), before.termination());
}

#[test]
fn iteration_limit_is_a_useful_non_success_result() {
    let function = diagonal_f64(2);
    let initial = vec![1.0; function.input_dimension()];

    let result = NewtonSystem::new(
        (scioxide_rootfind::NewtonSystemOptions::default()
            .with_residual_tolerance(scioxide_core::ResidualTolerance::from_tolerance(
                Tolerance::new(1.0e-30, 1.0e-30).unwrap(),
            ))
            .with_step_tolerance(scioxide_core::StepTolerance::from_tolerance(
                Tolerance::new(1.0e-30, 1.0e-30).unwrap(),
            ))
            .with_max_iterations(Budget::ONE))
        .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .solve(&function, &initial)
    .unwrap();
    assert_eq!(result.termination(), RootTermination::IterationLimit);
    assert!(!result.is_success());
    assert!(result.point().iter().all(|value| value.is_finite()));
    assert!(result.residual().iter().all(|value| value.is_finite()));
    assert_eq!(
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::Residual),
        2
    );
    assert_eq!(
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::Jacobian),
        1
    );
    assert_eq!(
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::Factorization),
        1
    );
    assert_eq!(
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::LinearSolve),
        1
    );
}

#[test]
fn workspace_growth_and_overflow_are_observable() {
    let mut workspace = SystemWorkspace::<f64>::new();
    NewtonSystem::new(
        scioxide_rootfind::NewtonSystemOptions::<f64>::default()
            .with_line_search(BacktrackingLineSearch::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .workspace_requirements(&diagonal_f64(2))
    .and_then(|r| workspace.try_reserve(&r))
    .unwrap();
    assert_eq!(workspace.dimension(), 2);
    NewtonSystem::new(
        scioxide_rootfind::NewtonSystemOptions::<f64>::default()
            .with_line_search(BacktrackingLineSearch::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .workspace_requirements(&diagonal_f64(5))
    .and_then(|r| workspace.try_reserve(&r))
    .unwrap();
    assert_eq!(workspace.dimension(), 5);
    assert_eq!(
        NewtonSystem::new(
            scioxide_rootfind::NewtonSystemOptions::<f64>::default()
                .with_line_search(BacktrackingLineSearch::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic)
        )
        .unwrap()
        .workspace_requirements(&diagonal_f64(usize::MAX))
        .and_then(|r| workspace.try_reserve(&r))
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::SizeOverflow
    );
    assert_eq!(workspace.dimension(), 5);
}

/// `F(x) = x − 1` whose residual and Jacobian calls demand exact scratch.
struct Scratched;
impl ResidualOperator<f64> for Scratched {
    fn input_dimension(&self) -> usize {
        2
    }
    fn output_dimension(&self) -> usize {
        2
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(3)
    }
    fn residual_into(
        &self,
        point: &[f64],
        output: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SciOxideError> {
        assert_eq!(scratch.len(), 3);
        output
            .iter_mut()
            .zip(point)
            .for_each(|(output, x)| *output = x - 1.0);
        Ok(())
    }
}
impl DenseJacobian<f64> for Scratched {
    fn jacobian_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(5)
    }
    fn jacobian_into(
        &self,
        _point: &[f64],
        output: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<bool, SciOxideError> {
        assert_eq!(scratch.len(), 5);
        output.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        Ok(true)
    }
}

#[test]
fn provider_scratch_is_reserved_and_passed_exactly() {
    let plain = NewtonSystem::new(
        scioxide_rootfind::NewtonSystemOptions::<f64>::default()
            .with_line_search(BacktrackingLineSearch::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .workspace_requirements(&residual_only(2, |_: &[f64], _: &mut [f64]| Ok(())))
    .unwrap();
    let scratched = NewtonSystem::new(
        scioxide_rootfind::NewtonSystemOptions::<f64>::default()
            .with_line_search(BacktrackingLineSearch::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .workspace_requirements(&Scratched)
    .unwrap();
    assert_eq!(
        scratched.required_bytes(),
        plain.required_bytes() + 5 * core::mem::size_of::<f64>()
    );
    for strategy in [DerivativeStrategy::Analytic, DerivativeStrategy::Central] {
        let solver = [&NewtonSystem::new(
            (scioxide_rootfind::NewtonSystemOptions::default()
                .with_line_search(BacktrackingLineSearch::default()))
            .with_derivative_strategy(strategy),
        )
        .unwrap()];
        for solver in solver {
            let mut workspace = solver.workspace(&Scratched).unwrap();
            let mut result = SystemRootResult::empty();
            result.try_reserve(2).unwrap();
            solver
                .solve_into(
                    &Scratched,
                    &[3.0, -2.0],
                    &mut result,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert!(result.is_success(), "{result:?}");
            assert!(result.point().iter().all(|x| (x - 1.0).abs() < 1e-10));
        }
        for result in [
            Broyden::new(
                scioxide_rootfind::BroydenUpdate::First,
                (scioxide_rootfind::BroydenOptions::default()
                    .with_line_search(BacktrackingLineSearch::default()))
                .with_derivative_strategy(strategy),
            )
            .unwrap()
            .solve(&Scratched, &[3.0, -2.0]),
            TrustRegionSystem::new(
                (scioxide_rootfind::TrustRegionSystemOptions::default())
                    .with_derivative_strategy(strategy),
            )
            .unwrap()
            .solve(&Scratched, &[3.0, -2.0]),
        ] {
            let result = result.unwrap();
            assert!(result.is_success(), "{result:?}");
            assert!(result.point().iter().all(|x| (x - 1.0).abs() < 1e-10));
        }
    }
}

fn solve_one_dimensional<S>()
where
    S: scioxide_core::RealScalar + scioxide_core::NumericScalar<Real = S>,
{
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    let two = one + one;
    let function = ResidualFn::new(1, move |point: &[S], output: &mut [S]| {
        output[0] = point[0] - two;
        Ok(())
    })
    .with_jacobian(move |_point: &[S], output: &mut [S]| {
        output[0] = one;
        Ok(())
    });
    let result = NewtonSystem::new(
        (scioxide_rootfind::NewtonSystemOptions::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .solve(&function, &[one])
    .unwrap();
    assert!(result.is_success());
    assert_eq!(result.point(), &[two]);
}

#[test]
fn prepared_contract_supports_f32_and_f64() {
    solve_one_dimensional::<f32>();
    solve_one_dimensional::<f64>();
}

#[test]
fn compatible_prepared_broyden_still_charges_retained_larger_storage() {
    use scioxide_core::SciOxideErrorKind;
    use scioxide_rootfind::{BroydenOptions, BroydenUpdate};
    let options =
        BroydenOptions::<f64>::default().with_derivative_strategy(DerivativeStrategy::Analytic);
    let solver = Broyden::new(BroydenUpdate::First, options.clone()).unwrap();
    let large = diagonal_f64(16);
    let small = diagonal_f64(4);
    let mut workspace = solver.workspace(&large).unwrap();
    let requirements = solver.workspace_requirements(&small).unwrap();
    workspace.try_reserve(&requirements).unwrap();
    let limited = Broyden::new(
        BroydenUpdate::First,
        options.with_workspace_byte_limit(requirements.required_bytes()),
    )
    .unwrap();
    let request = limited.workspace_requirements(&small).unwrap();
    assert_eq!(
        workspace.check(&request).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
}

/// `Fᵢ(x) = xᵢ² − (i + 1)`, whose positive root is `xᵢ = √(i + 1)`.
fn squares(point: &[f64], output: &mut [f64]) -> Result<(), SciOxideError> {
    for (index, (slot, &value)) in output.iter_mut().zip(point).enumerate() {
        *slot = value * value - (index + 1) as f64;
    }
    Ok(())
}

/// Solves `function` from `initial` with Newton, Broyden's first update and the
/// trust region, under the default strategy (`None`) or `strategy`.
fn dense_solves<F: DenseJacobian<f64>>(
    function: &F,
    initial: &[f64],
    strategy: Option<DerivativeStrategy>,
) -> [Result<SystemRootResult<f64>, SciOxideError>; 3] {
    let mut newton = NewtonSystemOptions::default();
    let mut broyden = BroydenOptions::default();
    let mut trust_region = TrustRegionSystemOptions::default();
    if let Some(strategy) = strategy {
        newton = newton.with_derivative_strategy(strategy);
        broyden = broyden.with_derivative_strategy(strategy);
        trust_region = trust_region.with_derivative_strategy(strategy);
    }
    [
        NewtonSystem::new(newton).unwrap().solve(function, initial),
        Broyden::new(BroydenUpdate::First, broyden)
            .unwrap()
            .solve(function, initial),
        TrustRegionSystem::new(trust_region)
            .unwrap()
            .solve(function, initial),
    ]
}

#[test]
fn default_strategy_solves_residual_only_problems_by_forward_differences() {
    let function = residual_only(2, squares);
    let defaults = dense_solves(&function, &[1.5, 1.5], None);
    let forward = dense_solves(&function, &[1.5, 1.5], Some(DerivativeStrategy::Forward));
    for (default, forward) in defaults.into_iter().zip(forward) {
        let (default, forward) = (default.unwrap(), forward.unwrap());
        assert!(default.is_success(), "{default:?}");
        assert!((default.point()[0] - 1.0).abs() < 1e-7);
        assert!((default.point()[1] - 2.0_f64.sqrt()).abs() < 1e-7);
        assert_eq!(default.point(), forward.point());
        assert_eq!(default.iterations(), forward.iterations());
        assert_eq!(
            default
                .counts()
                .provider(scioxide_core::EvaluationRole::Residual),
            forward
                .counts()
                .provider(scioxide_core::EvaluationRole::Residual)
        );
    }
    for result in dense_solves(&function, &[1.5, 1.5], Some(DerivativeStrategy::Analytic)) {
        assert!(matches!(
            result.unwrap_err().view(),
            scioxide_core::SciOxideErrorView::InvalidInput {
                reason: scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                ..
            }
        ));
    }
}

#[test]
fn default_strategy_keeps_analytic_jacobian_iterates_and_counts() {
    let function = diagonal_f64(3);
    let defaults = dense_solves(&function, &[1.0; 3], None);
    let analytic = dense_solves(&function, &[1.0; 3], Some(DerivativeStrategy::Analytic));
    for (default, analytic) in defaults.into_iter().zip(analytic) {
        let (default, analytic) = (default.unwrap(), analytic.unwrap());
        assert!(default.is_success(), "{default:?}");
        assert_eq!(default.point(), analytic.point());
        assert_eq!(default.termination(), analytic.termination());
        assert_eq!(default.accounting(), analytic.accounting());
    }
}

/// Rejects starting points of the wrong length and with non-finite components
/// in `solve` and `solve_into` before the first callback, leaving `output`
/// unchanged and recording a preflight rejection.
fn assert_initial_rejected<M, F>(solver: &M, function: &F, calls: &Cell<usize>)
where
    M: SystemRootSolver<f64, F, Workspace = SystemWorkspace<f64>>,
    F: DenseJacobian<f64>,
{
    let cx = scioxide_core::ExecutionContext::sequential_native;
    let mut workspace = solver.workspace(function).unwrap();
    let mut output = SystemRootResult::empty();
    output.try_reserve(2).unwrap();
    solver
        .solve_into(function, &[1.5, 1.5], &mut output, &mut workspace, cx())
        .unwrap();
    let (point, residual) = (output.point().to_vec(), output.residual().to_vec());
    let (termination, counts) = (output.termination(), output.accounting());
    calls.set(0);
    for (initial, kind) in [
        (
            &[1.5][..],
            scioxide_core::SciOxideErrorKind::DimensionMismatch,
        ),
        (
            &[1.5; 3][..],
            scioxide_core::SciOxideErrorKind::DimensionMismatch,
        ),
        (
            &[f64::NAN, 1.5][..],
            scioxide_core::SciOxideErrorKind::NonFinite,
        ),
        (
            &[1.5, f64::INFINITY][..],
            scioxide_core::SciOxideErrorKind::NonFinite,
        ),
    ] {
        for error in [
            solver.solve(function, initial).unwrap_err(),
            solver
                .solve_into(function, initial, &mut output, &mut workspace, cx())
                .unwrap_err(),
        ] {
            assert_eq!(error.kind(), kind, "{initial:?}");
            if kind == scioxide_core::SciOxideErrorKind::NonFinite {
                assert!(matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NonFinite {
                        origin: scioxide_core::NonFiniteOrigin::Input,
                        ..
                    }
                ));
            }
        }
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            scioxide_core::AttemptOutcome::PreflightRejected
        );
        assert_eq!(output.point(), point);
        assert_eq!(output.residual(), residual);
        assert_eq!(output.termination(), termination);
        assert_eq!(output.accounting(), counts);
    }
    assert_eq!(calls.get(), 0);
}

#[test]
fn invalid_initial_points_are_rejected_before_the_first_callback() {
    let calls = Cell::new(0);
    let function = residual_only(2, |point: &[f64], output: &mut [f64]| {
        calls.set(calls.get() + 1);
        squares(point, output)
    });
    assert_initial_rejected(
        &NewtonSystem::new(NewtonSystemOptions::default()).unwrap(),
        &function,
        &calls,
    );
    assert_initial_rejected(
        &Broyden::new(BroydenUpdate::Second, BroydenOptions::default()).unwrap(),
        &function,
        &calls,
    );
    assert_initial_rejected(
        &TrustRegionSystem::new(TrustRegionSystemOptions::default()).unwrap(),
        &function,
        &calls,
    );
}

#[test]
fn allocating_and_prepared_paths_have_identical_outcomes() {
    let function = diagonal_f64(4);
    let initial = vec![1.0; function.input_dimension()];

    macro_rules! check_solver {
        ($solver:expr) => {{
            let solver = $solver;

            let allocating = solver.solve(&function, &initial).unwrap();
            let (_, mut prepared, _) = prepared_state(&function);
            let mut workspace = solver.workspace(&function).unwrap();
            solver
                .solve_into(
                    &function,
                    &initial,
                    &mut prepared,
                    &mut workspace,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
                .unwrap();
            assert_eq!(prepared.termination(), allocating.termination());
            assert_eq!(prepared.iterations(), allocating.iterations());
            assert_eq!(prepared.counts(), allocating.counts());
            assert_eq!(prepared.point(), allocating.point());
            assert_eq!(prepared.residual(), allocating.residual());
            assert_eq!(
                prepared.residual_norm().unwrap(),
                allocating.residual_norm().unwrap()
            );
            if solver.name() == "Broyden" {
                // One Jacobian factored once; every secant step is one triangular solve.
                assert_eq!(
                    prepared
                        .counts()
                        .provider(scioxide_core::EvaluationRole::Jacobian),
                    1
                );
                assert_eq!(
                    prepared
                        .counts()
                        .provider(scioxide_core::EvaluationRole::Factorization),
                    1
                );
                assert_eq!(
                    prepared
                        .counts()
                        .provider(scioxide_core::EvaluationRole::LinearSolve),
                    prepared.iterations()
                );
            }
        }};
    }
    check_solver!(
        &NewtonSystem::new(
            (scioxide_rootfind::NewtonSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic)
        )
        .unwrap()
    );
    check_solver!(
        &Broyden::new(
            scioxide_rootfind::BroydenUpdate::First,
            (scioxide_rootfind::BroydenOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic)
        )
        .unwrap()
    );
    check_solver!(
        &TrustRegionSystem::new(
            (scioxide_rootfind::TrustRegionSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic)
        )
        .unwrap()
    );
}

#[test]
fn scaled_equivalent_systems_have_equivalent_certified_outcomes() {
    for scale in [1.0e-100_f64, 1.0, 1.0e100] {
        let function = ResidualFn::new(2, move |point: &[f64], output: &mut [f64]| {
            output[0] = scale * (point[0] - 2.0);
            output[1] = scale * (point[1] + 3.0);
            Ok(())
        })
        .with_jacobian(move |_point: &[f64], output: &mut [f64]| {
            output.copy_from_slice(&[scale, 0.0, 0.0, scale]);
            Ok(())
        });
        macro_rules! check_solver {
            ($solver:expr) => {{
                let solver = $solver;

                let result = solver.solve(&function, &[0.0, 0.0]).unwrap();
                assert_eq!(result.termination(), RootTermination::ResidualTolerance);
                assert!((result.point()[0] - 2.0).abs() < 1.0e-8);
                assert!((result.point()[1] + 3.0).abs() < 1.0e-8);
            }};
        }
        check_solver!(
            &NewtonSystem::new(
                (scioxide_rootfind::NewtonSystemOptions::default())
                    .with_derivative_strategy(DerivativeStrategy::Analytic)
            )
            .unwrap()
        );
        check_solver!(
            &Broyden::new(
                scioxide_rootfind::BroydenUpdate::First,
                (scioxide_rootfind::BroydenOptions::default())
                    .with_derivative_strategy(DerivativeStrategy::Analytic)
            )
            .unwrap()
        );
        check_solver!(
            &TrustRegionSystem::new(
                (scioxide_rootfind::TrustRegionSystemOptions::default())
                    .with_derivative_strategy(DerivativeStrategy::Analytic)
            )
            .unwrap()
        );
    }
}

#[test]
fn collapsed_system_candidate_is_unsuccessful_without_redundant_callback() {
    let calls = Rc::new(Cell::new(0_usize));
    let callback_calls = Rc::clone(&calls);
    let function = ResidualFn::new(1, move |_point: &[f64], output: &mut [f64]| {
        callback_calls.set(callback_calls.get() + 1);
        output[0] = 1.0;
        Ok(())
    })
    .with_jacobian(|_point: &[f64], output: &mut [f64]| {
        output[0] = f64::MAX;
        Ok(())
    });
    let result = NewtonSystem::new(
        (scioxide_rootfind::NewtonSystemOptions::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .solve(&function, &[1.0])
    .unwrap();
    assert_eq!(result.termination(), RootTermination::Stagnation);
    assert!(!result.is_success());
    assert_eq!(calls.get(), 1);
    assert_eq!(
        result
            .counts()
            .provider(scioxide_core::EvaluationRole::Residual),
        1
    );

    macro_rules! check_solver {
        ($solver:expr) => {{
            let solver = $solver;

            calls.set(0);
            match solver.solve(&function, &[1.0]) {
                Ok(result) => {
                    assert!(!result.is_success(), "{} reported a root", solver.name());
                    assert_eq!(result.termination(), RootTermination::Stagnation);
                }
                Err(error) => assert!(matches!(
                    error.view(),
                    scioxide_core::SciOxideErrorView::NumericalInstability { .. }
                )),
            }
            assert_eq!(calls.get(), 1);
        }};
    }
    check_solver!(
        &Broyden::new(
            scioxide_rootfind::BroydenUpdate::First,
            (scioxide_rootfind::BroydenOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic)
        )
        .unwrap()
    );
    check_solver!(
        &TrustRegionSystem::new(
            (scioxide_rootfind::TrustRegionSystemOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic)
        )
        .unwrap()
    );
}
