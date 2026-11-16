//! Manufactured roots and independently evaluated merit conditions.
#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]
use scioxide_core::{
    NumericScalar, RealScalar, ResidualOperator, ResidualTolerance, Scaled, SciOxideError,
    StepTolerance,
};
use scioxide_differentiate::DerivativeStrategy;
use scioxide_linalg::Vector;
use scioxide_rootfind::{
    BacktrackingLineSearch, Broyden, NewtonSystem, ResidualFn, RootLineSearch, RootTermination,
    SystemRootSolver, TrustRegionSystem,
};

/// Run `search` from `x` along `direction` with fresh trial buffers.
fn line_search<S: RealScalar + NumericScalar<Real = S>>(
    search: &impl RootLineSearch<S>,
    function: &dyn ResidualOperator<S>,
    x: &[S],
    direction: &[S],
    residual: &[S],
    slope: Scaled<S>,
) -> Result<Option<S>, SciOxideError> {
    let mut point = vec![S::ZERO; x.len()];
    let mut trial = vec![S::ZERO; x.len()];
    search.search_into(
        function,
        x,
        direction,
        residual,
        slope,
        &mut point,
        &mut trial,
        &mut [],
    )
}

macro_rules! precision_cases {
    ($module:ident, $scalar:ty, $start:expr, $tolerance:expr, $huge:expr, $tiny:expr) => {
        mod $module {
            use super::*;
            #[test]
            fn large_initial_residual_never_certifies_an_inaccurate_root() {
                let function = ResidualFn::new(1, |x: &[$scalar], f: &mut [$scalar]| {
                    f[0] = x[0] * x[0] - 2.0;
                    Ok(())
                })
                .with_jacobian(|x: &[$scalar], j: &mut [$scalar]| {
                    j[0] = 2.0 * x[0];
                    Ok(())
                });

                macro_rules! check {
                    ($solver:expr) => {
                        let result = $solver.solve(&function, &[$start]).unwrap();
                        assert!(result.is_success(), "{result:?}");
                        let x = result.point()[0];
                        assert!((x * x - 2.0).abs() <= $tolerance, "{result:?}");
                        assert_eq!(result.residual_norm().unwrap(), result.residual()[0].abs());
                    };
                }
                check!(
                    NewtonSystem::new(
                        (scioxide_rootfind::NewtonSystemOptions::default()
                            .with_residual_tolerance(
                                ResidualTolerance::new($tolerance, 0.0).unwrap()
                            )
                            .with_step_tolerance(StepTolerance::new(0.0, 0.0).unwrap()))
                        .with_derivative_strategy(DerivativeStrategy::Analytic)
                    )
                    .unwrap()
                );
                check!(
                    Broyden::new(
                        scioxide_rootfind::BroydenUpdate::First,
                        (scioxide_rootfind::BroydenOptions::default()
                            .with_residual_tolerance(
                                ResidualTolerance::new($tolerance, 0.0).unwrap()
                            )
                            .with_step_tolerance(StepTolerance::new(0.0, 0.0).unwrap()))
                        .with_derivative_strategy(DerivativeStrategy::Analytic)
                    )
                    .unwrap()
                );
                check!(
                    TrustRegionSystem::new(
                        (scioxide_rootfind::TrustRegionSystemOptions::default()
                            .with_residual_tolerance(
                                ResidualTolerance::new($tolerance, 0.0).unwrap()
                            )
                            .with_step_tolerance(StepTolerance::new(0.0, 0.0).unwrap()))
                        .with_derivative_strategy(DerivativeStrategy::Analytic)
                    )
                    .unwrap()
                );
            }

            #[test]
            fn deficient_consistent_system_has_a_root_and_stationary_nonroot_does_not() {
                for inconsistent in [false, true] {
                    let function = ResidualFn::new(2, move |x: &[$scalar], f: &mut [$scalar]| {
                        let residual = x[0] + x[1] - 3.0;
                        f.copy_from_slice(&[
                            residual,
                            if inconsistent { 1.0 } else { 2.0 * residual },
                        ]);
                        Ok(())
                    })
                    .with_jacobian(move |_: &[$scalar], j: &mut [$scalar]| {
                        j.copy_from_slice(if inconsistent {
                            &[1.0, 1.0, 0.0, 0.0]
                        } else {
                            &[1.0, 1.0, 2.0, 2.0]
                        });
                        Ok(())
                    });

                    let result = TrustRegionSystem::new(
                        (scioxide_rootfind::TrustRegionSystemOptions::default()
                            .with_residual_tolerance(
                                ResidualTolerance::new($tolerance, 0.0).unwrap(),
                            ))
                        .with_derivative_strategy(DerivativeStrategy::Analytic),
                    )
                    .unwrap()
                    .solve(&function, &[0.0, 0.0])
                    .unwrap();
                    if inconsistent {
                        assert_eq!(result.termination(), RootTermination::Stagnation);
                        assert!(result.residual_norm().unwrap() >= 1.0);
                    } else {
                        assert!(result.is_success(), "{result:?}");
                        assert!((result.point().iter().sum::<$scalar>() - 3.0).abs() <= $tolerance);
                    }
                }
            }

            #[test]
            fn armijo_uses_actual_slope_across_exponent_range() {
                for scale in [$tiny, 1.0, $huge] {
                    let function = ResidualFn::new(1, move |x: &[$scalar], f: &mut [$scalar]| {
                        f[0] = scale * x[0];
                        Ok(())
                    });
                    let x = Vector::from_vec(vec![1.0]);
                    let direction = Vector::from_vec(vec![-0.01]);
                    let residual = Vector::from_vec(vec![scale]);
                    let slope = Scaled::from_value(scale)
                        .unwrap()
                        .mul(Scaled::from_value(-0.01 * scale).unwrap())
                        .unwrap();
                    let search = BacktrackingLineSearch::new(0.9, 1e-12).unwrap();
                    let alpha = line_search(
                        &search,
                        &function,
                        x.as_slice(),
                        direction.as_slice(),
                        residual.as_slice(),
                        slope,
                    )
                    .unwrap()
                    .unwrap();
                    assert_eq!(alpha, 1.0);
                    // Independently cancel scale² from the quadratic merit inequality.
                    let trial: $scalar = 1.0 - 0.01 * alpha;
                    assert!(0.5 * trial * trial <= 0.5 + 0.9 * alpha * (-0.01));
                    assert!(
                        line_search(
                            &search,
                            &function,
                            x.as_slice(),
                            direction.as_slice(),
                            residual.as_slice(),
                            -slope
                        )
                        .is_err()
                    );
                }
            }

            #[test]
            fn trust_gradient_and_curvature_survive_unrepresentable_products() {
                for scale in [$tiny, $huge] {
                    let function = ResidualFn::new(1, move |x: &[$scalar], f: &mut [$scalar]| {
                        f[0] = scale * (x[0] - 2.0);
                        Ok(())
                    })
                    .with_jacobian(move |_: &[$scalar], j: &mut [$scalar]| {
                        j[0] = scale;
                        Ok(())
                    });

                    let result = TrustRegionSystem::new(
                        (scioxide_rootfind::TrustRegionSystemOptions::default()
                            .with_residual_scale(scale)
                            .with_residual_tolerance(
                                ResidualTolerance::new(0.0, $tolerance).unwrap(),
                            ))
                        .with_derivative_strategy(DerivativeStrategy::Analytic),
                    )
                    .unwrap()
                    .solve(&function, &[0.0])
                    .unwrap();
                    assert!(result.is_success(), "{result:?}");
                    assert!((result.point()[0] - 2.0).abs() <= $tolerance);
                }
            }
        }
    };
}
precision_cases!(single, f32, 1e4, 1e-5, 1e25, 1e-25);
precision_cases!(double, f64, 1e8, 1e-10, 1e200, 1e-200);

#[test]
fn broyden_line_search_uses_one_initial_jacobian() {
    let function = ResidualFn::new(1, |x: &[f64], f: &mut [f64]| {
        f[0] = x[0] * x[0] - 2.0;
        Ok(())
    })
    .with_jacobian(|x: &[f64], j: &mut [f64]| {
        j[0] = 2.0 * x[0];
        Ok(())
    });

    let solver = Broyden::new(
        scioxide_rootfind::BroydenUpdate::First,
        (scioxide_rootfind::BroydenOptions::default()
            .with_line_search(BacktrackingLineSearch::default())
            .with_step_tolerance(StepTolerance::new(1e-14, 1e-14).unwrap()))
        .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap();
    let result = solver.solve(&function, &[1.0]).unwrap();
    assert!(result.is_success());
    assert!(result.iterations() > 1);
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
}

#[test]
fn stationary_nonroot_is_a_globalization_failure_for_every_line_search_method() {
    // F = (x² + 1, y) at the origin: J = diag(0, 1) and JᵀF = 0, so every model
    // step has zero merit slope and no line search can make progress.
    let function = ResidualFn::new(2, |x: &[f64], f: &mut [f64]| {
        f.copy_from_slice(&[x[0] * x[0] + 1.0, x[1]]);
        Ok(())
    })
    .with_jacobian(|x: &[f64], j: &mut [f64]| {
        j.copy_from_slice(&[2.0 * x[0], 0.0, 0.0, 1.0]);
        Ok(())
    });
    let newton = NewtonSystem::new(
        scioxide_rootfind::NewtonSystemOptions::default()
            .with_line_search(BacktrackingLineSearch::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap()
    .solve(&function, &[0.0, 0.0])
    .unwrap();
    let broyden = |update| {
        Broyden::new(
            update,
            scioxide_rootfind::BroydenOptions::default()
                .with_line_search(BacktrackingLineSearch::default())
                .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap()
        .solve(&function, &[0.0, 0.0])
        .unwrap()
    };
    for result in [
        newton,
        broyden(scioxide_rootfind::BroydenUpdate::First),
        broyden(scioxide_rootfind::BroydenUpdate::Second),
    ] {
        assert_eq!(
            result.termination(),
            RootTermination::GlobalizationFailure,
            "{result:?}"
        );
        assert_eq!(result.point(), [0.0, 0.0]);
        assert_eq!(result.residual_norm(), Some(1.0));
    }
}

fn tridiagonal_counts<S: scioxide_core::RealScalar>() {
    for dimension in [4, 16] {
        for update in [
            scioxide_rootfind::BroydenUpdate::First,
            scioxide_rootfind::BroydenUpdate::Second,
        ] {
            for strategy in [DerivativeStrategy::Analytic, DerivativeStrategy::Forward] {
                let residual_calls = core::cell::Cell::new(0);
                let jacobian_calls = core::cell::Cell::new(0);
                let function = ResidualFn::new(dimension, |x: &[S], out: &mut [S]| {
                    for (i, value) in out.iter_mut().enumerate() {
                        let left = if i > 0 { x[i - 1] } else { S::ZERO };
                        let right = if i + 1 < dimension { x[i + 1] } else { S::ZERO };
                        let xi = x[i];
                        *value =
                            (S::TWO + S::ONE - S::TWO * xi) * xi - left - S::TWO * right + S::ONE;
                    }
                    residual_calls.set(residual_calls.get() + 1);
                    Ok(())
                })
                .with_jacobian(|x: &[S], out: &mut [S]| {
                    out.fill(S::ZERO);
                    for (i, row) in out.chunks_exact_mut(dimension).enumerate() {
                        row[i] = S::TWO + S::ONE - (S::TWO + S::TWO) * x[i];
                        if i > 0 {
                            row[i - 1] = -S::ONE;
                        }
                        if i + 1 < dimension {
                            row[i + 1] = -S::TWO;
                        }
                    }
                    jacobian_calls.set(jacobian_calls.get() + 1);
                    Ok(())
                });
                let initial = vec![-S::ONE; dimension];

                let result = Broyden::new(
                    update,
                    (scioxide_rootfind::BroydenOptions::default()
                        .with_line_search(BacktrackingLineSearch::default()))
                    .with_derivative_strategy(strategy),
                )
                .unwrap()
                .solve(&function, &initial)
                .unwrap();
                assert!(result.is_success(), "{update:?} {strategy:?}: {result:?}");
                assert!(result.iterations() > 1);
                assert_eq!(
                    result
                        .counts()
                        .provider(scioxide_core::EvaluationRole::Jacobian),
                    1
                );
                let factors = result
                    .counts()
                    .provider(scioxide_core::EvaluationRole::Factorization);
                if update == scioxide_rootfind::BroydenUpdate::First {
                    // Rejected inverse caches refactor the retained model,
                    // without repeating any derivative or residual probes.
                    assert!((1..=result.iterations()).contains(&factors));
                } else {
                    assert_eq!(factors, 1);
                }
                assert_eq!(
                    jacobian_calls.get(),
                    usize::from(strategy == DerivativeStrategy::Analytic)
                );
                assert_eq!(
                    result
                        .counts()
                        .provider(scioxide_core::EvaluationRole::Residual),
                    residual_calls.get()
                );
                let probes = if strategy == DerivativeStrategy::Forward {
                    dimension
                } else {
                    0
                };
                assert!(
                    residual_calls.get() <= 1 + probes + 2 * result.iterations(),
                    "{result:?}"
                );
            }
        }
    }
}

#[test]
fn tridiagonal_line_search_cost_has_no_repeated_differencing() {
    tridiagonal_counts::<f32>();
    tridiagonal_counts::<f64>();
}

fn non_descent_restart<S: scioxide_core::RealScalar>() {
    let function = ResidualFn::new(1, |x: &[S], f: &mut [S]| {
        let x = x[0];
        f[0] = x * x * x - S::TWO * x + S::TWO;
        Ok(())
    })
    .with_jacobian(|x: &[S], j: &mut [S]| {
        j[0] = (S::TWO + S::ONE) * x[0] * x[0] - S::TWO;
        Ok(())
    });
    let start = [S::ZERO];

    for update in [
        scioxide_rootfind::BroydenUpdate::First,
        scioxide_rootfind::BroydenUpdate::Second,
    ] {
        let result = Broyden::new(
            update,
            (scioxide_rootfind::BroydenOptions::default()
                .with_max_iterations(scioxide_core::Budget::ONE.saturating_add(1))
                .with_line_search(BacktrackingLineSearch::default()))
            .with_derivative_strategy(DerivativeStrategy::Analytic),
        )
        .unwrap()
        .solve(&function, &start)
        .unwrap();
        // The first point is 1, where the secant model B = -1 gives direction +1
        // with true merit slope +1, so the search exhausts. The refreshed derivative
        // gives direction -1: the full step to 0 has merit 2, and the quadratic
        // backtrack 1/(2(2 - 1/2 + 1)) = 1/5 is accepted.
        assert_eq!(result.termination(), RootTermination::IterationLimit);
        let fifth = S::ONE / (S::TWO + S::TWO + S::ONE);
        assert!((result.point()[0] - (S::ONE - fifth)).abs() <= S::EPSILON);
        assert!(result.residual_norm().unwrap() < S::ONE);
        assert_eq!(
            result
                .counts()
                .provider(scioxide_core::EvaluationRole::Jacobian),
            2
        );
        assert_eq!(
            result
                .counts()
                .provider(scioxide_core::EvaluationRole::Factorization),
            2
        );
        // One model solve per direction: the first step, the stale secant step and
        // the refreshed step.
        assert_eq!(
            result
                .counts()
                .provider(scioxide_core::EvaluationRole::LinearSolve),
            3
        );
    }
}

#[test]
fn non_descent_secant_direction_restarts_once_and_counts_refresh() {
    non_descent_restart::<f32>();
    non_descent_restart::<f64>();
}

#[test]
fn non_descent_is_rejected_before_any_trial_callback() {
    let calls = std::cell::Cell::new(0);
    let function = ResidualFn::new(1, |_: &[f64], _: &mut [f64]| {
        calls.set(calls.get() + 1);
        Ok(())
    });
    let one = Vector::from_vec(vec![1.0]);
    let zero = Vector::from_vec(vec![0.0]);
    for slope in [Scaled::from_value(1.0).unwrap(), Scaled::ZERO] {
        let result = line_search(
            &BacktrackingLineSearch::default(),
            &function,
            zero.as_slice(),
            one.as_slice(),
            one.as_slice(),
            slope,
        );
        assert_eq!(
            result.unwrap_err().kind(),
            scioxide_core::SciOxideErrorKind::NumericalInstability
        );
        assert_eq!(calls.get(), 0);
    }
}

#[test]
fn exhausted_model_descent_is_an_absent_step() {
    let function = ResidualFn::new(1, |x: &[f64], out: &mut [f64]| {
        out[0] = x[0] * x[0] + 1.0;
        Ok(())
    });
    let point = Vector::from_vec(vec![0.0]);
    let one = Vector::from_vec(vec![1.0]);
    let alpha = line_search(
        &BacktrackingLineSearch::default(),
        &function,
        point.as_slice(),
        one.as_slice(),
        one.as_slice(),
        Scaled::from_value(-1.0).unwrap(),
    )
    .unwrap();
    assert_eq!(alpha, None);
}

#[test]
fn deficient_model_obeys_budgets_and_injected_execution() {
    use scioxide_core::{
        Budget, BudgetKind, EvaluationRole, ExecutionContext, ProblemToken, SolveBudgets,
        SolveState,
    };
    use scioxide_rootfind::SystemRootResult;
    let function = ResidualFn::new(2, |x: &[f64], f: &mut [f64]| {
        let residual = x[0] - 1.0;
        f.copy_from_slice(&[residual, 2.0 * residual]);
        Ok(())
    })
    .with_jacobian(|_: &[f64], j: &mut [f64]| {
        j.copy_from_slice(&[1.0, 0.0, 2.0, 0.0]);
        Ok(())
    });

    let solver = TrustRegionSystem::new(
        (scioxide_rootfind::TrustRegionSystemOptions::default())
            .with_derivative_strategy(DerivativeStrategy::Analytic),
    )
    .unwrap();
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(0),
        scioxide_core::algorithm_id!("test", "dense-parent", 1),
        0,
        0,
    );
    let mut workspace = solver.workspace(&function).unwrap();
    let mut result = SystemRootResult::empty();
    result.try_reserve(2).unwrap();
    let mut state = SolveState::new();
    let mut budgets =
        SolveBudgets::unbounded().with_limit(BudgetKind::Evaluation, Budget::try_from(4).unwrap());
    {
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        solver
            .solve_into(
                &function,
                &[0.0, 0.0],
                &mut result,
                &mut workspace,
                &mut context,
            )
            .unwrap();
        assert_eq!(
            result.termination(),
            RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation)
        );
        assert_eq!(result.point(), &[0.0, 0.0]);
        assert_eq!(
            result.accounting().total().unwrap(),
            context.counts().total().unwrap()
        );
        assert_eq!(result.counts().provider(EvaluationRole::Factorization), 1);
        assert_eq!(result.counts().provider(EvaluationRole::Residual), 1);
    }
    // The consistent deficient model's Gauss–Newton step is its least-squares
    // solution: one Jacobian, one factorization and one triangular solve.
    let reference = solver.solve(&function, &[0.0, 0.0]).unwrap();
    assert!(reference.is_success());
    assert_eq!(reference.iterations(), 1);
    assert_eq!(
        reference
            .counts()
            .provider(scioxide_core::EvaluationRole::Jacobian),
        1
    );
    assert_eq!(
        reference
            .counts()
            .provider(scioxide_core::EvaluationRole::Factorization),
        1
    );
    assert_eq!(
        reference
            .counts()
            .provider(scioxide_core::EvaluationRole::LinearSolve),
        1
    );
    #[cfg(feature = "std")]
    {
        let executor = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(ExecutionContext::new(&executor), identity, &mut budgets)
            .unwrap();
        solver
            .solve_into(
                &function,
                &[0.0, 0.0],
                &mut result,
                &mut workspace,
                &mut context,
            )
            .unwrap();
        assert_eq!(result.point(), reference.point());
        assert_eq!(result.residual(), reference.residual());
        assert_eq!(result.counts(), reference.counts());
    }
}
