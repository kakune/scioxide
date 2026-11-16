//! Independent simplex geometry, transformed objectives, and callback accounting.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use scioxide_core::{
    Budget, ExecutionContext, NumericScalar, RealScalar, ResidualTolerance, SciOxideError,
    StepTolerance,
};
use scioxide_linalg::Matrix;
use scioxide_optimize::{
    NelderMead, NelderMeadGeometry, NelderMeadOptions, ObjectiveFn, ObjectiveFunction,
    OptimizeResult, OptimizeTermination,
};
use std::cell::RefCell;

fn numerical_cases<S: RealScalar + NumericScalar<Real = S>>() {
    let number = |x| S::try_from_f64_lossy(x).unwrap();
    let zero = <S as scioxide_core::AlgebraicScalar>::ZERO;
    let one = <S as scioxide_core::AlgebraicScalar>::ONE;
    // Powers of two preserve the transformed samples independently of the solver.
    for (translation, coordinate_scale, objective_scale, offset) in [
        (0.0, 1.0, 1.0, 0.0),
        (8.0, 4.0, 16.0, 4.0),
        (-8.0, 0.25, 0.0625, -4.0),
    ] {
        let t = number(translation);
        let xs = number(coordinate_scale);
        let fs = number(objective_scale);
        let offset = number(offset);
        let start = [t + xs * number(2.0), t - xs];
        let simplex = Matrix::from_vec(
            3,
            2,
            vec![
                start[0],
                start[1],
                start[0] + xs * number(0.25),
                start[1],
                start[0],
                start[1] + xs * number(0.125),
            ],
        )
        .unwrap();
        let tolerance = number(0.002);
        let options = NelderMeadOptions::default()
            .with_reference_scales(xs, fs)
            .with_step_tolerance(StepTolerance::new(zero, tolerance).unwrap())
            .with_function_tolerance(ResidualTolerance::new(zero, tolerance * tolerance).unwrap())
            .with_initial_simplex(simplex.clone());
        let solver = NelderMead::new(options).unwrap();
        let objective = ObjectiveFn::new(2, |x: &[S]| {
            let a = (x[0] - t) / xs;
            let b = (x[1] - t) / xs;
            Ok(offset + fs * (a * a + number(4.0) * b * b))
        });
        let mut work = solver.workspace(2).unwrap();
        let mut output = OptimizeResult::empty();
        solver
            .minimize_into(
                &objective,
                &start,
                &mut output,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert!(output.is_success(), "{:?}", output.termination());
        assert!(((output.x()[0] - t) / xs).abs() < number(0.01));
        assert!(((output.x()[1] - t) / xs).abs() < number(0.01));
        let diagnostics = nelder_mead_of(&output);
        assert_eq!(diagnostics.geometry(), NelderMeadGeometry::FullRank);
        let mut diameter = zero;
        for a in work.simplex().chunks_exact(2) {
            for b in work.simplex().chunks_exact(2) {
                diameter = diameter.max((a[0] - b[0]).abs()).max((a[1] - b[1]).abs());
            }
        }
        let (lo, hi) = work
            .values()
            .iter()
            .copied()
            .fold((work.values()[0], work.values()[0]), |(lo, hi), x| {
                (lo.min(x), hi.max(x))
            });
        assert_eq!(diagnostics.diameter(), Some(diameter));
        assert_eq!(diagnostics.objective_spread(), Some(hi - lo));
        assert!(diameter <= diagnostics.step_threshold().unwrap());
        assert!(hi - lo <= diagnostics.function_threshold().unwrap());
        assert_eq!(diagnostics.iterations(), output.iterations());
    }
    // Constant values alone cannot terminate a wide simplex.
    let solver =
        NelderMead::new(NelderMeadOptions::default().with_max_iterations(Budget::ONE)).unwrap();
    let constant = ObjectiveFn::new(2, |_: &[S]| Ok(one));
    assert_eq!(
        solver
            .minimize(&constant, &[one, one])
            .unwrap()
            .termination(),
        OptimizeTermination::IterationLimit
    );
    let collapsed = Matrix::from_vec(3, 2, vec![zero; 6]).unwrap();
    for repairs in [0, 2] {
        let options = NelderMeadOptions::default()
            .with_max_repairs(repairs)
            .with_initial_simplex(collapsed.clone());
        let result = NelderMead::new(options)
            .unwrap()
            .minimize(&constant, &[zero, zero])
            .unwrap();
        assert_eq!(result.is_success(), repairs > 0);
        assert_eq!(nelder_mead_of(&result).repairs(), usize::from(repairs > 0));
        if repairs == 0 {
            assert_eq!(result.termination(), OptimizeTermination::Stagnation);
            assert_eq!(nelder_mead_of(&result).rank().unwrap().rank(), 0);
        }
    }
    // Classical 1-D shrink stays nonzero; zero-dimensional problems observe once.
    let line = ObjectiveFn::new(1, |x: &[S]| Ok((x[0] - one) * (x[0] - one)));
    let result = NelderMead::default().minimize(&line, &[zero]).unwrap();
    assert!(result.is_success());
    assert!((result.x()[0] - one).abs() < number(0.002));
    struct Empty;
    impl<S: RealScalar + NumericScalar<Real = S>> ObjectiveFunction<S> for Empty {
        fn dim(&self) -> usize {
            0
        }
        fn eval(&self, _: &[S]) -> Result<S, SciOxideError> {
            Ok(<S as scioxide_core::AlgebraicScalar>::ONE)
        }
    }
    let empty = Empty;
    let result = NelderMead::<S>::default().minimize(&empty, &[]).unwrap();
    assert!(result.is_success());
    assert_eq!(result.fun_evals(), 1);
}

#[test]
fn real_precisions_and_independent_geometry() {
    numerical_cases::<f32>();
    numerical_cases::<f64>();
}

#[test]
fn every_callback_boundary_preserves_the_best_observation() {
    // These tables force expansion, outside/inside contraction, and shrink.
    for observations in [
        vec![0.0, 1.0, 2.0, -1.0, -2.0],
        vec![0.0, 1.0, 2.0, 1.5, 1.25],
        vec![0.0, 1.0, 2.0, 3.0, 1.5],
        vec![0.0, 1.0, 2.0, 3.0, 3.0, 0.5, -0.5],
    ] {
        for limit in 1..=observations.len() {
            let seen = RefCell::new(Vec::new());
            let objective = ObjectiveFn::new(2, |x: &[f64]| {
                let value = observations[seen.borrow().len()];
                seen.borrow_mut().push((x.to_vec(), value));
                Ok(value)
            });
            let solver = NelderMead::new(
                NelderMeadOptions::default()
                    .with_max_iterations(Budget::ONE)
                    .with_max_evaluations(Budget::try_from(limit).unwrap()),
            )
            .unwrap();
            let result = solver.minimize(&objective, &[0.0, 0.0]).unwrap();
            assert_eq!(seen.borrow().len(), limit);
            assert_eq!(result.fun_evals(), limit);
            let best = seen
                .borrow()
                .iter()
                .cloned()
                .reduce(|a, b| if b.1 < a.1 { b } else { a })
                .unwrap();
            assert_eq!(result.x(), best.0);
            assert_eq!(result.fun(), best.1);
            assert_eq!(
                result.termination(),
                if limit < observations.len() {
                    OptimizeTermination::EvaluationLimit
                } else {
                    OptimizeTermination::IterationLimit
                }
            );
        }
    }
}

#[test]
fn repair_budget_and_provider_failure_are_transactional() {
    let simplex = Matrix::from_vec(3, 2, vec![0.0; 6]).unwrap();
    for limit in 1..=5 {
        let solver = NelderMead::new(
            NelderMeadOptions::default()
                .with_initial_simplex(simplex.clone())
                .with_max_evaluations(Budget::try_from(limit).unwrap()),
        )
        .unwrap();
        let calls = RefCell::new(0);
        let f = ObjectiveFn::new(2, |_: &[f64]| {
            *calls.borrow_mut() += 1;
            Ok(1.0)
        });
        let result = solver.minimize(&f, &[0.0, 0.0]).unwrap();
        assert_eq!(result.termination(), OptimizeTermination::EvaluationLimit);
        assert_eq!(*calls.borrow(), limit);
        assert_eq!(result.fun_evals(), limit);
    }
    let solver = NelderMead::<f64>::default();
    let mut work = solver.workspace(2).unwrap();
    let mut output = OptimizeResult::empty();
    let good = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|x| x * x).sum()));
    solver
        .minimize_into(
            &good,
            &[1.0, 1.0],
            &mut output,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let saved = format!("{output:?}");
    // A non-finite start fails transactionally; later non-finite vertices are
    // rejected trials ranked above every finite vertex.
    for failed_at in 2..=7 {
        let calls = RefCell::new(0);
        let rejected = ObjectiveFn::new(2, |x: &[f64]| {
            *calls.borrow_mut() += 1;
            Ok(if *calls.borrow() == failed_at {
                f64::NAN
            } else {
                x.iter().map(|x| x * x).sum()
            })
        });
        let result = solver.minimize(&rejected, &[1.0, 1.0]).unwrap();
        assert!(result.is_success(), "NaN at call {failed_at}: {result:?}");
        assert!(result.fun().is_finite() && result.fun() < 1e-8);
    }
    {
        let failed_at = 1;
        let calls = RefCell::new(0);
        let bad = ObjectiveFn::new(2, |_: &[f64]| {
            *calls.borrow_mut() += 1;
            if *calls.borrow() == failed_at {
                Ok(f64::NAN)
            } else {
                Ok(1.0)
            }
        });
        assert!(matches!(
            (solver.minimize_into(
                &bad,
                &[1.0, 1.0],
                &mut output,
                &mut work,
                ExecutionContext::sequential()
            ))
            .as_ref()
            .map_err(scioxide_core::SciOxideError::view),
            Err(scioxide_core::SciOxideErrorView::NonFinite { .. })
        ));
        assert_eq!(format!("{output:?}"), saved);
        let attempt = work.last_attempt().unwrap();
        assert_eq!(
            attempt
                .counts()
                .own()
                .provider(scioxide_core::EvaluationRole::Objective),
            failed_at
        );
        assert_eq!(attempt.outcome(), scioxide_core::AttemptOutcome::Failed);
    }
}

#[test]
fn invalid_configuration_is_rejected_before_callbacks() {
    for (coordinate, objective) in [(0.0, 1.0), (1.0, f64::INFINITY), (-1.0, 1.0)] {
        let error = NelderMead::new(
            NelderMeadOptions::<f64>::default().with_reference_scales(coordinate, objective),
        )
        .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
    }
    let wrong = Matrix::from_vec(2, 2, vec![0.0; 4]).unwrap();
    assert!(NelderMead::new(NelderMeadOptions::default().with_initial_simplex(wrong)).is_err());
    let nonfinite = Matrix::from_vec(2, 1, vec![0.0, f64::NAN]).unwrap();
    assert_eq!(
        NelderMead::new(NelderMeadOptions::default().with_initial_simplex(nonfinite))
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    let simplex = Matrix::from_vec(2, 1, vec![1.0, 2.0]).unwrap();
    let solver =
        NelderMead::new(NelderMeadOptions::default().with_initial_simplex(simplex)).unwrap();
    let f = ObjectiveFn::new(1, |_: &[f64]| -> Result<f64, SciOxideError> {
        panic!("must not call objective")
    });
    assert!(solver.minimize(&f, &[0.0]).is_err());
    let mut work = solver.workspace(1).unwrap();
    let mut output = OptimizeResult::empty();
    assert!(
        solver
            .minimize_into(
                &f,
                &[0.0],
                &mut output,
                &mut work,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::PreflightRejected
    );
    assert!(work.last_accepted().is_none());
    assert_eq!(output.termination(), OptimizeTermination::NotStarted);
}

#[test]
fn noise_and_unrepresentable_initial_geometry_do_not_claim_optimality() {
    let calls = RefCell::new(0);
    let noisy = ObjectiveFn::new(2, |_: &[f64]| {
        *calls.borrow_mut() += 1;
        Ok(if *calls.borrow() % 2 == 0 { 1.0 } else { -1.0 })
    });
    let solver = NelderMead::new(
        NelderMeadOptions::default().with_max_evaluations(Budget::try_from(40).unwrap()),
    )
    .unwrap();
    let result = solver.minimize(&noisy, &[0.0, 0.0]).unwrap();
    assert!(!result.is_success());
    assert!(*calls.borrow() <= 40);
    let constant = ObjectiveFn::new(2, |_: &[f64]| Ok(0.0));
    let result = solver.minimize(&constant, &[f64::MAX; 2]).unwrap();
    assert_eq!(result.termination(), OptimizeTermination::Stagnation);
    assert_eq!(result.fun_evals(), 1);
}

#[test]
fn strided_simplex_is_copied_and_dependent_edges_are_repaired() {
    use scioxide_linalg::MatrixView;
    let backing = [0.0, 99.0, 0.0, 99.0, 1.0, 99.0, 1.0, 99.0, 2.0, 99.0, 2.0];
    let view = MatrixView::from_slice_with_strides(3, 2, &backing, 4, 2).unwrap();
    let options = NelderMeadOptions::default().with_initial_simplex(view.try_to_matrix().unwrap());
    assert_eq!(options.initial_simplex().unwrap().get(2, 1).unwrap(), 2.0);
    let constant = ObjectiveFn::new(2, |_: &[f64]| Ok(1.0));
    let result = NelderMead::new(options)
        .unwrap()
        .minimize(&constant, &[0.0, 0.0])
        .unwrap();
    assert!(result.is_success());
    assert_eq!(nelder_mead_of(&result).repairs(), 1);
}

#[test]
fn representational_collapse_exhausts_only_the_configured_repairs() {
    let constant = ObjectiveFn::new(1, |_: &[f64]| Ok(0.0));
    for repairs in [0, 1, 2] {
        let solver = NelderMead::new(
            NelderMeadOptions::default()
                .with_max_repairs(repairs)
                .with_step_tolerance(StepTolerance::new(0.0, 0.0).unwrap()),
        )
        .unwrap();
        let result = solver.minimize(&constant, &[1.0]).unwrap();
        assert_eq!(result.termination(), OptimizeTermination::Stagnation);
        assert_eq!(nelder_mead_of(&result).repairs(), repairs);
        assert_eq!(
            nelder_mead_of(&result).geometry(),
            NelderMeadGeometry::Degenerate
        );
    }
}

#[test]
fn opposite_extremes_have_valid_extended_range_stopping_and_rank() {
    macro_rules! check {
        ($s:ty) => {{
            let max = <$s>::MAX;
            let simplex = Matrix::from_vec(2, 1, vec![max, -max]).unwrap();
            let options = NelderMeadOptions::default()
                .with_reference_scales(max, max)
                .with_step_tolerance(StepTolerance::new(0.0, 2.0).unwrap())
                .with_function_tolerance(ResidualTolerance::new(0.0, 2.0).unwrap())
                .with_initial_simplex(simplex.clone());
            let linear = ObjectiveFn::new(1, |x: &[$s]| Ok(x[0]));
            let result = NelderMead::new(options)
                .unwrap()
                .minimize(&linear, &[max])
                .unwrap();
            assert!(result.is_success());
            assert_eq!(result.fun_evals(), 2);
            let diagnostics = nelder_mead_of(&result);
            assert_eq!(diagnostics.rank().unwrap().rank(), 1);
            assert_eq!(diagnostics.diameter(), None);
            assert_eq!(diagnostics.objective_spread(), None);
            assert_eq!(diagnostics.step_threshold(), None);
        }};
    }
    check!(f32);
    check!(f64);
}

#[test]
fn injected_executor_and_parent_limits_preserve_sequential_observations() {
    use core::num::NonZeroUsize;
    use scioxide_core::{
        BudgetKind, ExecutionPlan, Executor, PartitionFn, ProblemToken, SolveBudgets, SolveState,
    };
    struct CallerExecutor;
    impl Executor for CallerExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(2).unwrap()
        }
        fn execute(
            &self,
            _: ExecutionContext<'_>,
            _: ExecutionPlan,
            _: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            panic!("sequential Nelder-Mead must not request workers")
        }
    }
    for (parent, local) in [(1, 5), (5, 1), (3, 5), (5, 3)] {
        let solver = NelderMead::new(
            NelderMeadOptions::default().with_max_evaluations(Budget::try_from(local).unwrap()),
        )
        .unwrap();
        let f = ObjectiveFn::new(2, |x: &[f64]| Ok(x.iter().map(|x| x * x).sum()));
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(parent).unwrap());
        let mut context = state
            .begin_solve(
                ExecutionContext::new(&CallerExecutor),
                scioxide_core::SolveIdentity::new(
                    ProblemToken::new(600),
                    scioxide_core::algorithm_id!("test", "Parent", 1),
                    2,
                    0,
                ),
                &mut budgets,
            )
            .unwrap();
        let mut work = solver.workspace(2).unwrap();
        let mut output = OptimizeResult::empty();
        solver
            .minimize_into(&f, &[1.0, 2.0], &mut output, &mut work, &mut context)
            .unwrap();
        assert_eq!(output.termination(), OptimizeTermination::EvaluationLimit);
        assert_eq!(output.fun_evals(), parent.min(local));
        assert_eq!(output.counts(), work.last_attempt().unwrap().counts());
        assert_eq!(
            context
                .counts()
                .nested()
                .provider(scioxide_core::EvaluationRole::Objective),
            parent.min(local)
        );
    }
}

fn nelder_mead_of<S: scioxide_core::RealScalar>(
    result: &scioxide_optimize::OptimizeResult<S>,
) -> scioxide_optimize::NelderMeadDiagnostics<S> {
    match result.method_diagnostics() {
        scioxide_optimize::OptimizeDiagnostics::NelderMead(diagnostics) => diagnostics,
        other => panic!("unexpected method diagnostics {other:?}"),
    }
}

#[test]
fn higher_dimensional_incremental_geometry_uses_only_prepared_storage() {
    let solver = NelderMead::<f64>::default();
    let objective = ObjectiveFn::new(4, |x: &[f64]| Ok(x.iter().map(|v| v * v).sum()));
    let mut workspace = solver.workspace(4).unwrap();
    let mut output = OptimizeResult::empty();
    output.try_reserve(4).unwrap();
    {
        solver
            .minimize_into(
                &objective,
                &[1.0; 4],
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
    };
    assert!(output.is_success());
    assert!(output.fun() < 1e-10);
}
