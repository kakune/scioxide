#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]

use scioxide_core::ResidualOperator;
use scioxide_core::{Budget, Complex, ResidualTolerance, StepTolerance};
use scioxide_rootfind::{
    FixedPointInput, FixedPointMethod as Method, FixedPointOptions, FixedPointResult,
    FixedPointSolver, HistoryRestart, ResidualFn, RootTermination,
};

const METHODS: [Method; 7] = [
    Method::Picard,
    Method::Steffensen,
    Method::Anderson,
    Method::BroydenFirst,
    Method::BroydenSecond,
    Method::DiagonalBroyden,
    Method::ExcitingMixing,
];

macro_rules! real_cases {
    ($name:ident, $r:ty, $tol:expr) => {
        #[test]
        fn $name() {
            let options = FixedPointOptions::<$r>::default()
                .with_residual_tolerance(ResidualTolerance::new($tol, 0.0).unwrap())
                .with_step_tolerance(StepTolerance::new(0.0, 0.0).unwrap());
            for method in METHODS {
                let map = ResidualFn::new(2, |x: &[$r], out: &mut [$r]| {
                    out[0] = 0.5 * x[0] + 1.0;
                    out[1] = 0.25 * x[1] - 0.75;
                    Ok(())
                });
                let result = FixedPointSolver::new(
                    method,
                    scioxide_rootfind::FixedPointInput::Map,
                    (options).clone(),
                )
                .and_then(|solver| solver.solve(&map, &[0.0, 0.0]))
                .unwrap();
                assert!(result.is_success(), "{method:?}: {result:?}");
                // Analytic fixed point, independent of the solver's reported residual.
                assert!((result.point()[0] - 2.0).abs() <= 4.0 * $tol);
                assert!((result.point()[1] + 1.0).abs() <= 4.0 * $tol);
                let mut output = [0.0; 2];
                map.residual_into(result.point(), &mut output, &mut [])
                    .unwrap();
                let independent = output
                    .iter()
                    .zip(result.point())
                    .map(|(g, x)| (g - x) * (g - x))
                    .sum::<$r>()
                    .sqrt();
                assert!(independent <= $tol, "{method:?}: {independent}");
            }
        }
    };
}
real_cases!(real_f32, f32, 2e-5);
real_cases!(real_f64, f64, 2e-11);

macro_rules! complex_cases {
    ($name:ident, $r:ty, $tol:expr) => {
        #[test]
        fn $name() {
            let options = FixedPointOptions::<$r>::default()
                .with_residual_tolerance(ResidualTolerance::new($tol, 0.0).unwrap())
                .with_step_tolerance(StepTolerance::new(0.0, 0.0).unwrap());
            let root = Complex::new(2.0 as $r, -1.0);
            let coefficient = Complex::new(0.25 as $r, 0.125);
            for method in METHODS.into_iter().filter(|m| *m != Method::ExcitingMixing) {
                let map = ResidualFn::new(1, |x: &[Complex<$r>], out: &mut [Complex<$r>]| {
                    out[0] = root + coefficient * (x[0] - root);
                    Ok(())
                });
                let result = FixedPointSolver::new(
                    method,
                    scioxide_rootfind::FixedPointInput::Map,
                    (options).clone(),
                )
                .and_then(|solver| solver.solve(&map, &[Complex::new(0.0, 0.0)]))
                .unwrap();
                assert!(result.is_success(), "{method:?}: {result:?}");
                assert!((result.point()[0] - root).norm() <= 4.0 * $tol);
            }
            assert!(
                FixedPointSolver::new(
                    Method::ExcitingMixing,
                    scioxide_rootfind::FixedPointInput::Map,
                    (options).clone()
                )
                .and_then(|solver| solver.workspace(&ResidualFn::new(
                    1,
                    |_: &[Complex<$r>], _: &mut [Complex<$r>]| Ok(())
                )))
                .is_err()
            );
        }
    };
}
complex_cases!(complex_f32, f32, 2e-5);
complex_cases!(complex_f64, f64, 2e-11);

fn strict() -> FixedPointOptions<f64> {
    FixedPointOptions::default()
        .with_residual_tolerance(ResidualTolerance::new(1e-11, 0.0).unwrap())
        .with_step_tolerance(StepTolerance::new(0.0, 0.0).unwrap())
}
#[test]
fn noncontractive_map_and_no_solution() {
    let map = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
        out[0] = 2.0 * x[0] - 3.0;
        Ok(())
    });
    for method in [
        Method::Steffensen,
        Method::Anderson,
        Method::BroydenFirst,
        Method::BroydenSecond,
        Method::DiagonalBroyden,
    ] {
        let result = FixedPointSolver::new(
            method,
            scioxide_rootfind::FixedPointInput::Map,
            (strict()).clone(),
        )
        .and_then(|solver| solver.solve(&map, &[0.0]))
        .unwrap();
        assert!(result.is_success(), "{method:?}: {result:?}");
        assert!((result.point()[0] - 3.0).abs() < 1e-10);
    }
    for method in METHODS {
        let result = FixedPointSolver::new(
            method,
            scioxide_rootfind::FixedPointInput::Map,
            (strict().with_max_iterations(Budget::try_from(8).unwrap())).clone(),
        )
        .and_then(|solver| {
            solver.solve(
                &ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
                    out[0] = x[0] + 1.0;
                    Ok(())
                }),
                &[0.0],
            )
        })
        .unwrap();
        assert!(!result.is_success(), "{method:?}");
        assert_eq!(result.residual(), &[1.0]);
    }
}

#[test]
fn dependent_history_regularization_and_restarts() {
    let map = ResidualFn::new(4, |x: &[f64], out: &mut [f64]| {
        for (out, &x) in out.iter_mut().zip(x) {
            *out = x.cos();
        }
        Ok(())
    });
    let solver = FixedPointSolver::new(
        Method::Anderson,
        scioxide_rootfind::FixedPointInput::Map,
        strict(),
    )
    .unwrap();
    let initial = [1.0; 4];
    let result = solver.solve(&map, &initial).unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!(result.diagnostics().skipped_updates() > 0);
    assert!(result.point().iter().all(|x| (x - x.cos()).abs() < 1e-11));
    let options = strict()
        .with_regularization(1e-6)
        .with_history(2, HistoryRestart::Restart);
    let solver = FixedPointSolver::new(Method::Anderson, FixedPointInput::Map, options).unwrap();
    let result = solver.solve(&map, &initial).unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!(result.diagnostics().regularized_solves() > 0);
    assert!(result.diagnostics().restarts() > 0);
}
#[test]
fn residual_report_pairs_the_scaled_stopping_norm_with_its_reference() {
    // F(x₀) = (1e3, 0) with divisors 1e3: raw norm 1e3, scaled norm 1. One
    // damped step cannot converge, so the report shows the stopping ratio.
    let map = ResidualFn::new(2, |x: &[f64], out: &mut [f64]| {
        out.copy_from_slice(&[x[0] - 1e3, x[1]]);
        Ok(())
    });
    let options = FixedPointOptions::default()
        .with_component_scales(vec![1e3, 1e3])
        .with_max_iterations(Budget::ONE);
    let result = FixedPointSolver::new(Method::Picard, FixedPointInput::Residual, options)
        .and_then(|solver| solver.solve(&map, &[0.0, 0.0]))
        .unwrap();
    assert!(!result.is_success(), "{result:?}");
    let report = result.residual_report().unwrap();
    let raw = result.residual().iter().fold(0.0f64, |a, &v| a.hypot(v));
    let scaled = raw / 1e3;
    assert_eq!(result.residual_norm(), Some(raw));
    assert!((result.scaled_residual_norm().unwrap() - scaled).abs() <= 4.0 * f64::EPSILON * scaled);
    assert_eq!(report.raw_norm(), result.scaled_residual_norm().unwrap());
    // The initial scaled norm is one, so the normalized residual equals the
    // scaled norm, not the raw norm in function units.
    assert!((report.scaled_norm() - scaled).abs() <= 4.0 * f64::EPSILON * scaled);
}
#[test]
fn component_scaling_and_large_initial_residual_do_not_hide_failure() {
    let options = strict().with_component_scales(vec![1e100, 1e-100]);
    let map = ResidualFn::new(2, |x: &[f64], out: &mut [f64]| {
        out[0] = 0.5 * (2e100 - x[0]);
        out[1] = 0.25 * (2e-100 - x[1]);
        Ok(())
    });
    let result = FixedPointSolver::new(Method::Anderson, FixedPointInput::Residual, options)
        .and_then(|solver| solver.solve(&map, &[0.0; 2]))
        .unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!((result.point()[0] / 1e100 - 2.0).abs() < 1e-10);
    assert!((result.point()[1] / 1e-100 - 2.0).abs() < 1e-10);
    let result = FixedPointSolver::new(
        Method::Picard,
        scioxide_rootfind::FixedPointInput::Map,
        strict().with_max_iterations(Budget::try_from(2).unwrap()),
    )
    .and_then(|solver| {
        solver.solve(
            &ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
                out[0] = x[0] * 0.01 + 100.0;
                Ok(())
            }),
            &[1e100],
        )
    })
    .unwrap();
    assert!(!result.is_success());
    assert!(result.residual()[0].abs() > 1e90);
}
#[test]
fn transaction_preflight_callback_and_zero_dimension() {
    use std::cell::Cell;
    let calls = Cell::new(0);
    let callback = ResidualFn::new(2, |x: &[f64], out: &mut [f64]| {
        calls.set(calls.get() + 1);
        out.copy_from_slice(x);
        Ok(())
    });
    let options = strict();
    let solver =
        FixedPointSolver::new(Method::Anderson, FixedPointInput::Map, (options).clone()).unwrap();
    let mut w = solver.workspace(&callback).unwrap();
    let mut result = FixedPointResult::new(2).unwrap();
    assert!(
        solver
            .solve_into(
                &callback,
                &[0.0],
                &mut result,
                &mut w,
                scioxide_core::ExecutionContext::sequential_native()
            )
            .is_err()
    );
    assert_eq!(calls.get(), 0);
    solver
        .solve_into(
            &callback,
            &[1.0, 2.0],
            &mut result,
            &mut w,
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
    let before = format!("{result:?}");
    assert!(
        solver
            .solve_into(
                &ResidualFn::new(2, |_: &[f64], _: &mut [f64]| {
                    Err(scioxide_core::SciOxideError::singular(None))
                }),
                &[0.0; 2],
                &mut result,
                &mut w,
                scioxide_core::ExecutionContext::sequential_native()
            )
            .is_err()
    );
    assert_eq!(format!("{result:?}"), before);
    assert!(
        solver
            .solve_into(
                &ResidualFn::new(2, |_: &[f64], out: &mut [f64]| {
                    out.fill(f64::NAN);
                    Ok(())
                }),
                &[0.0; 2],
                &mut result,
                &mut w,
                scioxide_core::ExecutionContext::sequential_native()
            )
            .is_err()
    );
    assert_eq!(format!("{result:?}"), before);
    for method in METHODS {
        let empty = FixedPointSolver::new(
            method,
            scioxide_rootfind::FixedPointInput::Map,
            (options).clone(),
        )
        .and_then(|solver| {
            solver.solve(&ResidualFn::new(0, |_: &[f64], _: &mut [f64]| Ok(())), &[])
        })
        .unwrap();
        assert!(empty.is_success());
        assert!(empty.point().is_empty());
    }
}
#[test]
fn evaluation_budget_returns_last_checked_point() {
    use scioxide_core::{BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveState};
    let options = strict();
    let solver = FixedPointSolver::new(
        Method::Anderson,
        scioxide_rootfind::FixedPointInput::Map,
        options,
    )
    .unwrap();
    let map = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
        out[0] = 0.5 * x[0] + 1.0;
        Ok(())
    });
    let mut w = solver.workspace(&map).unwrap();
    let mut result = FixedPointResult::new(1).unwrap();
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(1).unwrap());
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            scioxide_core::SolveIdentity::new(
                ProblemToken::new(9),
                scioxide_core::algorithm_id!("test", "dense-parent", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    solver
        .solve_into(&map, &[0.0], &mut result, &mut w, &mut context)
        .unwrap();
    assert_eq!(
        result.termination(),
        RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
    );
    assert_eq!(result.point(), &[0.0]);
    assert_eq!(result.residual(), &[1.0]);
}

#[test]
fn dense_sparse_operator_and_borrowed_forms_agree() {
    use scioxide_core::LinearOperator;
    use scioxide_linalg::{Matrix, Vector};
    use scioxide_sparse::CsrMatrix;
    use std::cell::RefCell;
    let dense = Matrix::from_vec(2, 2, vec![0.25, 0.125, 0.0, 0.5]).unwrap();
    let sparse =
        CsrMatrix::from_raw(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![0.25, 0.125, 0.5]).unwrap();
    let initial = Vector::from_vec(vec![0.0, 0.0]);
    for operator in [
        &dense.view().try_finite().unwrap() as &dyn LinearOperator<f64>,
        &sparse,
    ] {
        let scratch = RefCell::new(vec![0.0; operator.apply_workspace_len().unwrap()]);
        let callback = ResidualFn::new(2, |x: &[f64], out: &mut [f64]| {
            operator.apply_into(x, out, &mut scratch.borrow_mut())?;
            out[0] += 1.375;
            out[1] += 0.5;
            Ok(())
        });
        let result = FixedPointSolver::new(
            Method::Anderson,
            scioxide_rootfind::FixedPointInput::Map,
            (strict()).clone(),
        )
        .and_then(|solver| solver.solve(&callback, initial.view().as_contiguous_slice().unwrap()))
        .unwrap();
        assert!(result.is_success());
        assert!((result.point()[0] - 2.0).abs() < 1e-10);
        assert!((result.point()[1] - 1.0).abs() < 1e-10);
    }
}

#[test]
fn coupled_complex_history_uses_hermitian_products() {
    let root = [Complex::new(1.0, -2.0), Complex::new(2.0, 1.0)];
    let map = ResidualFn::new(2, |x: &[Complex<f64>], out: &mut [Complex<f64>]| {
        let a = x[0] - root[0];
        let b = x[1] - root[1];
        out[0] = root[0] + Complex::new(0.2, 0.1) * a + Complex::new(0.1, -0.1) * b;
        out[1] = root[1] + Complex::new(0.0, 0.1) * a + Complex::new(0.3, -0.1) * b;
        Ok(())
    });
    for method in [
        Method::Anderson,
        Method::BroydenFirst,
        Method::BroydenSecond,
    ] {
        let result = FixedPointSolver::new(
            method,
            scioxide_rootfind::FixedPointInput::Map,
            (strict()).clone(),
        )
        .and_then(|solver| solver.solve(&map, &[Complex::new(0.0, 0.0); 2]))
        .unwrap();
        assert!(result.is_success(), "{method:?}: {result:?}");
        for (x, r) in result.point().iter().zip(root) {
            assert!((*x - r).norm() < 1e-10);
        }
    }
}

#[test]
fn cancellation_and_new_epoch_preserve_output_and_reset_history() {
    use scioxide_core::{
        CancellationToken, ExecutionContext, ProblemToken, SolveBudgets, SolveState,
    };
    let options = strict();
    let solver = FixedPointSolver::new(
        Method::Anderson,
        scioxide_rootfind::FixedPointInput::Map,
        options,
    )
    .unwrap();
    let map = ResidualFn::new(2, |x: &[f64], out: &mut [f64]| {
        out.iter_mut().zip(x).for_each(|(out, x)| *out = x.cos());
        Ok(())
    });
    let mut w = solver.workspace(&map).unwrap();
    let mut result = FixedPointResult::new(2).unwrap();
    solver
        .solve_into(
            &map,
            &[1.0; 2],
            &mut result,
            &mut w,
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
    let previous = format!("{result:?}");
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancellation),
            scioxide_core::SolveIdentity::new(
                ProblemToken::new(1),
                scioxide_core::algorithm_id!("test", "dense-parent", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    assert!(
        solver
            .solve_into(&map, &[0.0; 2], &mut result, &mut w, &mut context)
            .is_err()
    );
    assert_eq!(format!("{result:?}"), previous);
    solver
        .solve_into(
            &map,
            &[1.0; 2],
            &mut result,
            &mut w,
            scioxide_core::ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(format!("{result:?}"), previous);
}

#[cfg(feature = "std")]
#[test]
fn injected_executor_matches_sequential() {
    use scioxide_core::{ExecutionContext, ProblemToken, SolveBudgets, SolveState, ThreadExecutor};
    let pool = ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    let options = strict();
    let solver = FixedPointSolver::new(
        Method::Anderson,
        scioxide_rootfind::FixedPointInput::Map,
        options,
    )
    .unwrap();
    let map = ResidualFn::new(4, |x: &[f64], out: &mut [f64]| {
        out.iter_mut().zip(x).for_each(|(out, x)| *out = x.cos());
        Ok(())
    });
    let expected = solver.solve(&map, &[1.0; 4]).unwrap();
    let mut w = solver.workspace(&map).unwrap();
    let mut result = FixedPointResult::new(4).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::new(&pool),
            scioxide_core::SolveIdentity::new(
                ProblemToken::new(7),
                scioxide_core::algorithm_id!("test", "dense-parent", 1),
                0,
                0,
            ),
            &mut budgets,
        )
        .unwrap();
    solver
        .solve_into(&map, &[1.0; 4], &mut result, &mut w, &mut context)
        .unwrap();
    assert_eq!(result.point(), expected.point());
    assert_eq!(result.residual(), expected.residual());
    assert_eq!(result.counts(), expected.counts());
}

#[test]
fn initialized_broyden_first_and_second_share_the_secant_kernel() {
    use scioxide_differentiate::DerivativeStrategy;
    use scioxide_rootfind::{Broyden, BroydenUpdate, SystemRootSolver};
    let f = ResidualFn::new(2, |x: &[f64], out: &mut [f64]| {
        out[0] = x[0].powi(2) - 2.0;
        out[1] = x[1].powi(2) - 3.0;
        Ok(())
    })
    .with_jacobian(|x: &[f64], out: &mut [f64]| {
        out.fill(0.0);
        out[0] = 2.0 * x[0];
        out[3] = 2.0 * x[1];
        Ok(())
    });
    for update in [BroydenUpdate::First, BroydenUpdate::Second] {
        let result = Broyden::new(
            update,
            (scioxide_rootfind::BroydenOptions::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap()
        .solve(&f, &[1.0, 1.0])
        .unwrap();
        assert!(result.is_success());
        assert!((result.point()[0].powi(2) - 2.0).abs() < 1e-7);
        assert_eq!(
            result
                .counts()
                .provider(scioxide_core::EvaluationRole::Jacobian),
            1
        );
    }
}

#[test]
fn invalid_options_and_stagnation_are_not_success() {
    assert!(
        FixedPointSolver::<f64>::new(
            Method::Anderson,
            FixedPointInput::Map,
            strict().with_residual_scale(0.0)
        )
        .is_err()
    );
    assert!(
        FixedPointSolver::<f64>::new(
            Method::Anderson,
            FixedPointInput::Map,
            strict().with_damping(f64::INFINITY)
        )
        .is_err()
    );
    assert!(
        FixedPointSolver::<f64>::new(
            Method::Anderson,
            FixedPointInput::Map,
            strict().with_regularization(-1.0)
        )
        .is_err()
    );
    assert!(
        FixedPointSolver::<f64>::new(
            Method::Anderson,
            FixedPointInput::Map,
            strict().with_history(0, HistoryRestart::Oldest)
        )
        .is_err()
    );
    assert!(
        FixedPointSolver::<f64>::new(
            Method::Anderson,
            FixedPointInput::Map,
            strict().with_component_scales(vec![f64::NAN])
        )
        .is_err()
    );
    let options = strict().with_damping(1e-20);
    let result = FixedPointSolver::new(
        Method::Picard,
        scioxide_rootfind::FixedPointInput::Map,
        options,
    )
    .and_then(|solver| {
        solver.solve(
            &ResidualFn::new(1, |_: &[f64], out: &mut [f64]| {
                out[0] = 2.0;
                Ok(())
            }),
            &[1.0],
        )
    })
    .unwrap();
    assert_eq!(result.termination(), RootTermination::Stagnation);
    assert_eq!(result.residual(), &[1.0]);
}
