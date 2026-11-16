#![allow(clippy::unwrap_used, clippy::expect_used)]
use super::*;
use crate::{ConstraintFn, ObjectiveFn, ObjectiveFunction};
use alloc::vec;
use scioxide_core::{Bounds, ExecutionContext, SolveState};

/// An empty constraint collection: the unconstrained problem of dimension `n`.
fn free<S: RealScalar + NumericScalar<Real = S>>(n: usize) -> Constraints<S> {
    Constraints::new(n).unwrap()
}

/// Prepared call in a root epoch with sequential portable dispatch.
fn solve<S: RealScalar + NumericScalar<Real = S>>(
    solver: &InteriorPoint<S>,
    problem: &ConstrainedProblem<'_, S>,
    x0: &[S],
    output: &mut OptimizeResult<S>,
    workspace: &mut InteriorPointWorkspace<S>,
) -> Result<(), SciOxideError> {
    solver.minimize_into(
        problem,
        x0,
        output,
        workspace,
        ExecutionContext::sequential(),
    )
}

/// One epoch rooted in `execution` with caller budgets; returns the written
/// result and the epoch's counts.
fn root(
    solver: &InteriorPoint<f64>,
    problem: &ConstrainedProblem<'_, f64>,
    x0: &[f64],
    execution: ExecutionContext<'_>,
    mut budgets: SolveBudgets,
) -> (
    Result<OptimizeResult<f64>, SciOxideError>,
    scioxide_core::SolveCounts,
) {
    let mut workspace = solver.workspace(problem).unwrap();
    let mut output = OptimizeResult::empty();
    let mut state = SolveState::new();
    let mut context = state
        .begin_solve(execution, solver.identity(problem), &mut budgets)
        .unwrap();
    let result = solver
        .minimize_in_context(
            problem,
            x0,
            &mut output,
            &mut workspace.storage,
            &mut context,
        )
        .map(|()| output);
    let counts = context.counts();
    (result, counts)
}

macro_rules! numerical_cases {
    ($module:ident,$s:ty,$eps:expr) => {
        mod $module {
            use super::*;
            fn check(out: &OptimizeResult<$s>) {
                assert!(
                    out.is_success(),
                    "{:?} x={:?} residuals={:?}",
                    out.termination(),
                    out.x(),
                    out.diagnostics()
                );
                let d = out.diagnostics();
                assert!(d.primal_feasibility().unwrap() < $eps);
                assert!(d.stationarity().unwrap() < $eps);
                assert!(
                    d.complementarity().unwrap() < $eps * out.fun().abs().max(1.),
                    "{out:?}"
                );
            }
            #[test]
            fn mixed_infeasible_start() {
                let f =
                    ObjectiveFn::new(2, |x: &[$s]| Ok((x[0] - 2.).powi(2) + (x[1] - 1.).powi(2)))
                        .with_gradient(|x: &[$s], output| {
                            output.copy_from_slice(&[2. * (x[0] - 2.), 2. * (x[1] - 1.)]);
                            Ok(())
                        });
                let c = Constraints::new(2)
                    .unwrap()
                    .equality(
                        ConstraintFn::new(2, |x: &[$s]| Ok(x[0] + x[1] - 1.)).with_jacobian(
                            |_: &[$s], output| {
                                output.copy_from_slice(&[1., 1.]);
                                Ok(())
                            },
                        ),
                    )
                    .unwrap()
                    .inequality(
                        ConstraintFn::new(2, |x: &[$s]| Ok(0.25 - x[1])).with_jacobian(
                            |_: &[$s], output| {
                                output.copy_from_slice(&[0., -1.]);
                                Ok(())
                            },
                        ),
                    )
                    .unwrap();
                let problem = ConstrainedProblem::new(&f, &c).unwrap();
                let solver = InteriorPoint::new(InteriorPointOptions::default());
                let mut w = solver.workspace(&problem).unwrap();
                let mut out = OptimizeResult::empty();
                solve(&solver, &problem, &[-3., 4.], &mut out, &mut w).unwrap();
                check(&out);
                assert!((out.x()[0] - 0.75).abs() < $eps);
                assert!((out.x()[1] - 0.25).abs() < $eps);
                let l = w.state().multipliers();
                assert!((2. * (out.x()[0] - 2.) + l[0]).abs() < $eps);
                assert!((2. * (out.x()[1] - 1.) + l[0] - l[1]).abs() < $eps);
                assert!(l[1] >= 0.);
                assert!((l[1] * (0.25 - out.x()[1])).abs() < $eps * out.fun().abs().max(1.));
            }
            #[test]
            fn dependent_equalities() {
                let f = ObjectiveFn::new(2, |x: &[$s]| Ok(x[0] * x[0] + x[1] * x[1]))
                    .with_gradient(|x: &[$s], output| {
                        output.copy_from_slice(&[2. * x[0], 2. * x[1]]);
                        Ok(())
                    });
                let mut c = Constraints::new(2).unwrap();
                for scale in [1., 2., -3.] {
                    c = c
                        .equality(
                            ConstraintFn::new(2, move |x: &[$s]| Ok(scale * (x[0] + x[1] - 1.)))
                                .with_jacobian(move |_: &[$s], output| {
                                    output.copy_from_slice(&[scale, scale]);
                                    Ok(())
                                }),
                        )
                        .unwrap();
                }
                let problem = ConstrainedProblem::new(&f, &c).unwrap();
                let solver = InteriorPoint::new(InteriorPointOptions::default());
                let mut w = solver.workspace(&problem).unwrap();
                let mut out = OptimizeResult::empty();
                solve(&solver, &problem, &[2., -1.], &mut out, &mut w).unwrap();
                check(&out);
                assert!((out.x()[0] - 0.5).abs() < $eps);
                assert!(interior_of(&out).regularization_attempts() > 0);
            }
            #[test]
            fn direct_and_fixed_bounds() {
                let f = ObjectiveFn::new(3, |x: &[$s]| {
                    Ok((x[0] + 1.).powi(2) + (x[1] - 3.).powi(2) + x[2] * x[2])
                })
                .with_gradient(|x: &[$s], output| {
                    output.copy_from_slice(&[2. * (x[0] + 1.), 2. * (x[1] - 3.), 2. * x[2]]);
                    Ok(())
                });
                let c = Constraints::new(3)
                    .unwrap()
                    .bounds(
                        Bounds::new(vec![
                            (Some(0.), Some(2.)),
                            (Some(-1.), Some(1.)),
                            (Some(0.5), Some(0.5)),
                        ])
                        .unwrap(),
                    )
                    .unwrap();
                let problem = ConstrainedProblem::new(&f, &c).unwrap();
                let solver = InteriorPoint::new(InteriorPointOptions::default());
                let mut w = solver.workspace(&problem).unwrap();
                let mut out = OptimizeResult::empty();
                solve(&solver, &problem, &[-10., 10., 7.], &mut out, &mut w).unwrap();
                check(&out);
                assert!(out.x()[0].abs() < $eps);
                assert!((out.x()[1] - 1.).abs() < $eps);
                assert_eq!(out.x()[2], 0.5);
                assert!((w.state().lower_multipliers()[2] - 1.).abs() < $eps);
            }
            #[test]
            fn curved_inequality_and_unconstrained() {
                let f = ObjectiveFn::new(2, |x: &[$s]| Ok((x[0] - 2.).powi(2) + x[1] * x[1]))
                    .with_gradient(|x: &[$s], output| {
                        output.copy_from_slice(&[2. * (x[0] - 2.), 2. * x[1]]);
                        Ok(())
                    });
                let c = Constraints::new(2)
                    .unwrap()
                    .inequality(
                        ConstraintFn::new(2, |x: &[$s]| Ok(x[0] * x[0] + x[1] * x[1] - 1.))
                            .with_jacobian(|x: &[$s], output| {
                                output.copy_from_slice(&[2. * x[0], 2. * x[1]]);
                                Ok(())
                            }),
                    )
                    .unwrap();
                let problem = ConstrainedProblem::new(&f, &c).unwrap();
                let solver = InteriorPoint::new(InteriorPointOptions::default());
                let out = solver.minimize(&problem, &[3., 1.]).unwrap();
                check(&out);
                assert!((out.x()[0] - 1.).abs() < $eps);
                assert!(out.x()[1].abs() < $eps);
                let free = free(2);
                let problem = ConstrainedProblem::new(&f, &free).unwrap();
                let out = InteriorPoint::<$s>::default()
                    .minimize(&problem, &[0., 1.])
                    .unwrap();
                check(&out);
                assert!((out.x()[0] - 2.).abs() < $eps);
            }
            #[test]
            fn inconsistent_linearization_restores_and_flat_violation_fails() {
                let f = ObjectiveFn::new(1, |x: &[$s]| Ok((x[0] - 2.).powi(2))).with_gradient(
                    |x: &[$s], output| {
                        output.copy_from_slice(&[2. * (x[0] - 2.)]);
                        Ok(())
                    },
                );
                let c = Constraints::new(1)
                    .unwrap()
                    .equality(
                        ConstraintFn::new(1, |x: &[$s]| Ok(x[0] * x[0] - 1.)).with_jacobian(
                            |x: &[$s], output| {
                                output.copy_from_slice(&[2. * x[0]]);
                                Ok(())
                            },
                        ),
                    )
                    .unwrap()
                    .equality(
                        ConstraintFn::new(1, |x: &[$s]| Ok(x[0] - 1.)).with_jacobian(
                            |_: &[$s], output| {
                                output.copy_from_slice(&[1.]);
                                Ok(())
                            },
                        ),
                    )
                    .unwrap();
                let problem = ConstrainedProblem::new(&f, &c).unwrap();
                let solver = InteriorPoint::new(InteriorPointOptions::default());
                let out = solver.minimize(&problem, &[0.]).unwrap();
                check(&out);
                assert!(interior_of(&out).restoration_solves() > 0);
                let c = Constraints::new(1)
                    .unwrap()
                    .equality(ConstraintFn::new(1, |_: &[$s]| Ok(1.)).with_jacobian(
                        |_: &[$s], output| {
                            output.copy_from_slice(&[0.]);
                            Ok(())
                        },
                    ))
                    .unwrap();
                let problem = ConstrainedProblem::new(&f, &c).unwrap();
                let out = solver.minimize(&problem, &[0.]).unwrap();
                assert!(!out.is_success());
                assert_eq!(out.termination(), OptimizeTermination::Stagnation);
                assert_eq!(out.x(), &[0.]);
            }
        }
    };
}
numerical_cases!(f32_cases, f32, 5e-3);
numerical_cases!(f64_cases, f64, 1e-5);

struct Quadratic;
impl ObjectiveFunction<f64> for Quadratic {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] * x[0] + x[1] * x[1])
    }
    fn gradient_into(&self, x: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
        out[0] = 2. * x[0];
        out[1] = 2. * x[1];
        Ok(true)
    }
}
struct Sum;
impl crate::ConstraintFunction<f64> for Sum {
    fn dim(&self) -> usize {
        2
    }
    fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
        Ok(x[0] + x[1] - 1.)
    }
    fn jacobian_into(&self, _: &[f64], out: &mut [f64]) -> Result<bool, SciOxideError> {
        out.fill(1.);
        Ok(true)
    }
}
#[test]
fn reused_storage_has_no_internal_allocations_and_resets_state() {
    let c = Constraints::new(2).unwrap().equality(Sum).unwrap();
    let problem = ConstrainedProblem::new(&Quadratic, &c).unwrap();
    let solver = InteriorPoint::new(InteriorPointOptions::default());
    let mut w = solver.workspace(&problem).unwrap();
    let mut out = OptimizeResult::empty();
    solve(&solver, &problem, &[2., 2.], &mut out, &mut w).unwrap();
    assert!(out.is_success());
    let expected = out.clone();
    let lambda = w.state().multipliers().to_vec();
    solve(&solver, &problem, &[2., 2.], &mut out, &mut w).unwrap();
    assert_eq!(out.x(), expected.x());
    assert_eq!(out.counts(), expected.counts());
    assert_eq!(w.state().multipliers(), lambda);
}
#[test]
fn evaluation_budget_sweep_keeps_only_valid_observations() {
    use scioxide_core::BudgetKind;
    let c = Constraints::new(2).unwrap().equality(Sum).unwrap();
    let problem = ConstrainedProblem::new(&Quadratic, &c).unwrap();
    let solver = InteriorPoint::new(InteriorPointOptions::default());
    let complete = solver.minimize(&problem, &[2., 2.]).unwrap();
    for allowance in 1..=complete.counts().own().total_providers().unwrap() + 1 {
        let budgets = SolveBudgets::unbounded()
            .with_limit(BudgetKind::Evaluation, Budget::try_from(allowance).unwrap());
        let (result, counts) = root(
            &solver,
            &problem,
            &[2., 2.],
            ExecutionContext::sequential(),
            budgets,
        );
        assert!(counts.own().total_providers().unwrap() <= allowance);
        match result {
            Ok(r) => {
                assert_eq!(r.counts(), counts);
                assert_eq!(r.fun(), r.x()[0] * r.x()[0] + r.x()[1] * r.x()[1]);
                if r.is_success() {
                    assert!((r.x()[0] + r.x()[1] - 1.).abs() < 1e-5);
                } else {
                    assert_eq!(r.termination(), OptimizeTermination::EvaluationLimit);
                }
            }
            Err(_) => assert_eq!(allowance, 1),
        }
    }
}
#[test]
fn invalid_provider_and_trial_derivative_preserve_result() {
    use scioxide_sparse::LdlInertia;
    struct Bad {
        factor: DenseKktProvider<f64>,
    }
    impl KktProvider<f64> for Bad {
        fn factor(
            &mut self,
            a: scioxide_linalg::MatrixView<'_, f64>,
            e: ExecutionContext<'_>,
        ) -> Result<LdlInertia, SciOxideError> {
            self.factor.factor(a, e)
        }
        fn solve_into(
            &mut self,
            _: &[f64],
            out: &mut [f64],
            _: ExecutionContext<'_>,
        ) -> Result<(), SciOxideError> {
            out.fill(123.);
            Ok(())
        }
    }
    let free = free(2);
    let problem = ConstrainedProblem::new(&Quadratic, &free).unwrap();
    let solver = InteriorPoint::<f64>::default();
    let mut w = solver.workspace(&problem).unwrap().with_kkt_provider(Bad {
        factor: DenseKktProvider::new(2).unwrap(),
    });
    let mut out = solver.minimize(&problem, &[3., 3.]).unwrap();
    let original = out.clone();
    assert!(solve(&solver, &problem, &[2., 2.], &mut out, &mut w).is_err());
    assert_eq!(out.x(), original.x());
    assert_eq!(out.counts(), original.counts());
    assert_eq!(w.storage.accepted.x, &[2., 2.]);
    struct Fails;
    impl ObjectiveFunction<f64> for Fails {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            Quadratic.eval(x)
        }
        fn gradient_into(&self, x: &[f64], g: &mut [f64]) -> Result<bool, SciOxideError> {
            if x != [2., 2.] {
                return Err(invalid(
                    scioxide_core::argument!("x"),
                    "injected trial derivative failure",
                ));
            }
            Quadratic.gradient_into(x, g)
        }
    }
    let failing = ConstrainedProblem::new(&Fails, &free).unwrap();
    let mut w = solver.workspace(&failing).unwrap();
    assert!(solve(&solver, &failing, &[2., 2.], &mut out, &mut w).is_err());
    assert_eq!(out.x(), original.x());
    assert_eq!(w.storage.accepted.x, &[2., 2.]);
}
#[test]
fn numerical_derivatives_respect_fixed_and_finite_bounds() {
    struct Bounded;
    impl ObjectiveFunction<f64> for Bounded {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            assert_eq!(x[0], 0.5);
            assert!((0.0..=1.0).contains(&x[1]));
            Ok((x[1] - 0.75).powi(2))
        }
    }
    let c = Constraints::new(2)
        .unwrap()
        .bounds(Bounds::new(vec![(Some(0.5), Some(0.5)), (Some(0.), Some(1.))]).unwrap())
        .unwrap();
    let problem = ConstrainedProblem::new(&Bounded, &c).unwrap();
    let solver = InteriorPoint::new(InteriorPointOptions::default());
    let r = solver.minimize(&problem, &[-3., 8.]).unwrap();
    assert!(r.is_success(), "{r:?}");
    assert!((r.x()[1] - 0.75).abs() < 1e-5);
    assert!(!interior_of(&r).fixed_derivatives_available());
    assert!(r.grad().is_none());
}
#[test]
fn narrow_bounds_and_shape_errors_are_explicit() {
    let c = Constraints::new(2)
        .unwrap()
        .bounds(Bounds::new(vec![(Some(1.), Some(1.0f64.next_up())), (None, None)]).unwrap())
        .unwrap();
    let problem = ConstrainedProblem::new(&Quadratic, &c).unwrap();
    let solver = InteriorPoint::new(InteriorPointOptions::default());
    assert!(solver.minimize(&problem, &[1., 0.]).is_err());
    let one = free::<f64>(1);
    assert_eq!(
        ConstrainedProblem::new(&Quadratic, &one)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    let two = free(2);
    let problem = ConstrainedProblem::new(&Quadratic, &two).unwrap();
    let solver = InteriorPoint::<f64>::default();
    let mut w = solver.workspace(&problem).unwrap();
    let mut out = OptimizeResult::empty();
    assert_eq!(
        solve(&solver, &problem, &[1.], &mut out, &mut w)
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        scioxide_core::AttemptOutcome::PreflightRejected
    );
    assert_eq!(out.termination(), OptimizeTermination::NotStarted);
}
macro_rules! curvature_cases {
    ($module:ident,$s:ty,$eps:expr) => {
        mod $module {
            use super::*;
            struct Quartic;
            impl ObjectiveFunction<$s> for Quartic {
                fn dim(&self) -> usize {
                    1
                }
                fn eval(&self, x: &[$s]) -> Result<$s, SciOxideError> {
                    Ok(x[0].powi(4) - 2. * x[0] * x[0])
                }
                fn gradient_into(&self, x: &[$s], g: &mut [$s]) -> Result<bool, SciOxideError> {
                    g[0] = 4. * x[0].powi(3) - 4. * x[0];
                    Ok(true)
                }
            }
            impl LagrangianHessian<$s> for Quartic {
                fn hessian_into(
                    &self,
                    x: &[$s],
                    _: &[$s],
                    mut h: MatrixViewMut<'_, $s>,
                ) -> Result<(), SciOxideError> {
                    h.set(0, 0, 12. * x[0] * x[0] - 4.)
                }
            }
            #[test]
            fn exact_negative_curvature_requires_inertia_correction() {
                let free = free(1);
                let problem = ConstrainedProblem::new(&Quartic, &free)
                    .unwrap()
                    .with_lagrangian_hessian(&Quartic);
                let solver = InteriorPoint::<$s>::default();
                let mut w = solver.workspace(&problem).unwrap();
                let mut out = OptimizeResult::empty();
                solve(&solver, &problem, &[0.2], &mut out, &mut w).unwrap();
                assert!(out.is_success(), "{out:?}");
                assert!((out.x()[0] - 1.).abs() < $eps);
                assert!(interior_of(&out).regularization_attempts() > 0);
                assert!(out.counts().own().provider(EvaluationRole::Hessian) > 0);
            }
            #[test]
            fn nonlinear_equality_correction_reaches_circle_minimum() {
                let f =
                    ObjectiveFn::new(2, |x: &[$s]| Ok(x[0])).with_gradient(|_: &[$s], output| {
                        output.copy_from_slice(&[1., 0.]);
                        Ok(())
                    });
                let c = Constraints::new(2)
                    .unwrap()
                    .equality(
                        ConstraintFn::new(2, |x: &[$s]| Ok(x[0] * x[0] + x[1] * x[1] - 1.))
                            .with_jacobian(|x: &[$s], output| {
                                output.copy_from_slice(&[2. * x[0], 2. * x[1]]);
                                Ok(())
                            }),
                    )
                    .unwrap();
                let problem = ConstrainedProblem::new(&f, &c).unwrap();
                let solver = InteriorPoint::new(InteriorPointOptions::default());
                let r = solver.minimize(&problem, &[0., 1.]).unwrap();
                assert!(r.is_success(), "{r:?}");
                assert!((r.x()[0] + 1.).abs() < $eps);
                assert!(r.x()[1].abs() < $eps);
                assert!(interior_of(&r).accepted_corrections() > 0);
            }
            #[test]
            fn exact_nonlinear_constraint_curvature_uses_accepted_multipliers() {
                struct Curvature;
                impl LagrangianHessian<$s> for Curvature {
                    fn hessian_into(
                        &self,
                        _: &[$s],
                        lambda: &[$s],
                        mut h: MatrixViewMut<'_, $s>,
                    ) -> Result<(), SciOxideError> {
                        assert_eq!(lambda.len(), 1);
                        for i in 0..2 {
                            for j in 0..2 {
                                h.set(i, j, if i == j { 2. + 2. * lambda[0] } else { 0. })?;
                            }
                        }
                        Ok(())
                    }
                }
                let f = ObjectiveFn::new(2, |x: &[$s]| Ok((x[0] - 2.).powi(2) + x[1] * x[1]))
                    .with_gradient(|x: &[$s], output| {
                        output.copy_from_slice(&[2. * (x[0] - 2.), 2. * x[1]]);
                        Ok(())
                    });
                let c = Constraints::new(2)
                    .unwrap()
                    .inequality(
                        ConstraintFn::new(2, |x: &[$s]| Ok(x[0] * x[0] + x[1] * x[1] - 1.))
                            .with_jacobian(|x: &[$s], output| {
                                output.copy_from_slice(&[2. * x[0], 2. * x[1]]);
                                Ok(())
                            }),
                    )
                    .unwrap();
                let problem = ConstrainedProblem::new(&f, &c)
                    .unwrap()
                    .with_lagrangian_hessian(&Curvature);
                let solver = InteriorPoint::new(InteriorPointOptions::default());
                let mut w = solver.workspace(&problem).unwrap();
                let mut r = OptimizeResult::empty();
                solve(&solver, &problem, &[3., 1.], &mut r, &mut w).unwrap();
                assert!(r.is_success(), "{r:?}");
                assert!((r.x()[0] - 1.).abs() < $eps);
                assert!(r.x()[1].abs() < $eps);
                assert!((w.state().multipliers()[0] - 1.).abs() < $eps);
            }
            #[test]
            fn sparse_provider_matches_dense() {
                let c = Constraints::new(1)
                    .unwrap()
                    .bounds(Bounds::new(vec![(Some(0.), Some(2.))]).unwrap())
                    .unwrap();
                let problem = ConstrainedProblem::new(&Quartic, &c)
                    .unwrap()
                    .with_lagrangian_hessian(&Quartic);
                let solver = InteriorPoint::new(InteriorPointOptions::default());
                let mut w = solver.workspace(&problem).unwrap().with_kkt_provider(
                    crate::SparseKktProvider::new(scioxide_sparse::SparseSymmetricLimits::new(
                        100, 100, 1_000_000,
                    )),
                );
                let mut out = OptimizeResult::empty();
                solve(&solver, &problem, &[0.8], &mut out, &mut w).unwrap();
                assert!(out.is_success(), "{out:?}");
                assert!((out.x()[0] - 1.).abs() < $eps);
            }
        }
    };
}
curvature_cases!(curvature_f32, f32, 2e-3);
curvature_cases!(curvature_f64, f64, 1e-5);

#[test]
fn cancellation_deadline_and_executor_use_injected_capabilities() {
    use core::sync::atomic::{AtomicU64, Ordering};
    use scioxide_core::{CancellationToken, Clock, MonotonicInstant, SequentialExecutor};
    struct ClockAt(AtomicU64);
    impl Clock for ClockAt {
        fn now(&self) -> Result<MonotonicInstant, scioxide_core::CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                core::time::Duration::from_nanos(self.0.load(Ordering::Relaxed)),
            ))
        }
    }
    struct Stop<'a> {
        token: &'a CancellationToken,
        clock: &'a ClockAt,
        time: bool,
    }
    impl ObjectiveFunction<f64> for Stop<'_> {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            if x != [2., 2.] {
                if self.time {
                    self.clock.0.store(100, Ordering::Relaxed);
                } else {
                    self.token.cancel();
                }
            }
            Quadratic.eval(x)
        }
        fn gradient_into(&self, x: &[f64], g: &mut [f64]) -> Result<bool, SciOxideError> {
            Quadratic.gradient_into(x, g)
        }
    }
    let free = free(2);
    for time in [false, true] {
        let token = CancellationToken::new();
        let clock = ClockAt(AtomicU64::new(0));
        let clock_binding = scioxide_core::ClockBinding::new(&clock);
        let solver = InteriorPoint::new(
            InteriorPointOptions::default().with_time_limit(core::time::Duration::from_nanos(50)),
        );
        let objective = Stop {
            token: &token,
            clock: &clock,
            time,
        };
        let problem = ConstrainedProblem::new(&objective, &free).unwrap();
        let mut w = solver.workspace(&problem).unwrap();
        let mut r = OptimizeResult::empty();
        solver
            .minimize_into(
                &problem,
                &[2., 2.],
                &mut r,
                &mut w,
                ExecutionContext::new(&SequentialExecutor)
                    .with_cancellation(&token)
                    .with_clock(&clock_binding)
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            r.termination(),
            if time {
                OptimizeTermination::TimeLimit
            } else {
                OptimizeTermination::Cancelled
            }
        );
        assert_eq!(r.x(), &[2., 2.]);
    }
    let problem = ConstrainedProblem::new(&Quadratic, &free).unwrap();
    let solver = InteriorPoint::<f64>::default();
    let mut w = solver.workspace(&problem).unwrap();
    let mut expected = OptimizeResult::empty();
    solve(&solver, &problem, &[2., 2.], &mut expected, &mut w).unwrap();
    let mut w = solver.workspace(&problem).unwrap();
    let mut r = OptimizeResult::empty();
    solver
        .minimize_into(
            &problem,
            &[2., 2.],
            &mut r,
            &mut w,
            ExecutionContext::new(&SequentialExecutor),
        )
        .unwrap();
    assert_eq!(r.x(), expected.x());
    assert_eq!(r.counts(), expected.counts());
}
#[test]
fn malformed_hessians_are_not_replaced_with_bfgs() {
    struct Bad(bool);
    impl LagrangianHessian<f64> for Bad {
        fn hessian_into(
            &self,
            _: &[f64],
            _: &[f64],
            mut h: MatrixViewMut<'_, f64>,
        ) -> Result<(), SciOxideError> {
            h.set(0, 0, if self.0 { f64::NAN } else { 1. })?;
            h.set(1, 1, 1.)?;
            h.set(0, 1, 2.)?;
            h.set(1, 0, 0.)
        }
    }
    let free = free(2);
    let problem = ConstrainedProblem::new(&Quadratic, &free).unwrap();
    let solver = InteriorPoint::<f64>::default();
    let mut w = solver.workspace(&problem).unwrap();
    let mut r = solver.minimize(&problem, &[2., 2.]).unwrap();
    let expected = r.clone();
    for nan in [false, true] {
        let hessian = Bad(nan);
        let curved = problem.with_lagrangian_hessian(&hessian);
        assert!(solve(&solver, &curved, &[2., 2.], &mut r, &mut w).is_err());
        assert_eq!(r.x(), expected.x());
        assert_eq!(r.counts(), expected.counts());
    }
}

#[test]
fn final_allowed_iteration_is_checked_and_nested_limits_are_conserved() {
    use scioxide_core::BudgetKind;
    let free = free(2);
    let problem = ConstrainedProblem::new(&Quadratic, &free).unwrap();
    let solver =
        InteriorPoint::new(InteriorPointOptions::default().with_max_iterations(Budget::ONE));
    let result = solver.minimize(&problem, &[2., 2.]).unwrap();
    assert!(result.is_success());
    assert_eq!(result.iterations(), 1);
    let c = Constraints::new(2).unwrap().equality(Sum).unwrap();
    let problem = ConstrainedProblem::new(&Quadratic, &c).unwrap();
    let solver = InteriorPoint::new(InteriorPointOptions::default());
    for (kind, expected) in [
        (BudgetKind::Iteration, OptimizeTermination::IterationLimit),
        (BudgetKind::Proposal, OptimizeTermination::ProposalLimit),
    ] {
        let (result, counts) = root(
            &solver,
            &problem,
            &[2., 2.],
            ExecutionContext::sequential(),
            SolveBudgets::unbounded().with_limit(kind, Budget::ONE),
        );
        let result = result.unwrap();
        assert_eq!(result.termination(), expected);
        // min x²+y² subject to x+y=1 has the exact solution (1/2,1/2).
        // One outer iteration now admits the complete inner linear solve.
        assert!(result.x().iter().all(|x| (*x - 0.5).abs() < 1e-12));
        assert_eq!(result.counts(), counts);
        if kind == BudgetKind::Iteration {
            assert_eq!(counts.own().iterations(), 1);
        }
    }
}
#[test]
fn contradictory_constraints_and_all_fixed_coordinates_are_honest() {
    let c = Constraints::new(2)
        .unwrap()
        .equality(Sum)
        .unwrap()
        .equality(
            ConstraintFn::new(2, |x: &[f64]| Ok(x[0] + x[1] + 1.)).with_jacobian(
                |_: &[f64], output| {
                    output.copy_from_slice(&[1., 1.]);
                    Ok(())
                },
            ),
        )
        .unwrap();
    let problem = ConstrainedProblem::new(&Quadratic, &c).unwrap();
    let solver = InteriorPoint::new(InteriorPointOptions::default());
    let r = solver.minimize(&problem, &[0., 0.]).unwrap();
    assert_eq!(r.termination(), OptimizeTermination::Stagnation);
    assert!(r.primal_feasibility().unwrap() >= 1.);
    let c = Constraints::new(2)
        .unwrap()
        .bounds(Bounds::new(vec![(Some(1.), Some(1.)), (Some(-2.), Some(-2.))]).unwrap())
        .unwrap();
    let problem = ConstrainedProblem::new(&Quadratic, &c).unwrap();
    let mut w = solver.workspace(&problem).unwrap();
    let mut r = OptimizeResult::empty();
    solve(&solver, &problem, &[0., 0.], &mut r, &mut w).unwrap();
    assert!(r.is_success());
    assert_eq!(r.x(), &[1., -2.]);
    assert_eq!(w.state().lower_multipliers(), &[2., 0.]);
    assert_eq!(w.state().upper_multipliers(), &[0., 4.]);
}

#[test]
fn provider_roundoff_cannot_move_a_fixed_coordinate() {
    use scioxide_sparse::LdlInertia;
    struct Inexact(DenseKktProvider<f64>);
    impl KktProvider<f64> for Inexact {
        fn factor(
            &mut self,
            a: scioxide_linalg::MatrixView<'_, f64>,
            e: ExecutionContext<'_>,
        ) -> Result<LdlInertia, SciOxideError> {
            self.0.factor(a, e)
        }
        fn solve_into(
            &mut self,
            b: &[f64],
            x: &mut [f64],
            e: ExecutionContext<'_>,
        ) -> Result<(), SciOxideError> {
            self.0.solve_into(b, x, e)?;
            x[0] = 1e-12;
            Ok(())
        }
    }
    struct Bounded;
    impl ObjectiveFunction<f64> for Bounded {
        fn dim(&self) -> usize {
            2
        }
        fn eval(&self, x: &[f64]) -> Result<f64, SciOxideError> {
            assert_eq!(x[0], 0.5);
            Quadratic.eval(x)
        }
        fn gradient_into(&self, x: &[f64], g: &mut [f64]) -> Result<bool, SciOxideError> {
            Quadratic.gradient_into(x, g)
        }
    }
    let c = Constraints::new(2)
        .unwrap()
        .bounds(Bounds::new(vec![(Some(0.5), Some(0.5)), (None, None)]).unwrap())
        .unwrap();
    let problem = ConstrainedProblem::new(&Bounded, &c).unwrap();
    let solver = InteriorPoint::new(InteriorPointOptions::default());
    let mut w = solver
        .workspace(&problem)
        .unwrap()
        .with_kkt_provider(Inexact(DenseKktProvider::new(2).unwrap()));
    let mut r = OptimizeResult::empty();
    solve(&solver, &problem, &[0.5, 2.], &mut r, &mut w).unwrap();
    assert!(r.is_success());
    assert_eq!(r.x(), &[0.5, 0.]);
}

#[test]
fn incomplete_hessian_writes_cannot_reuse_old_curvature() {
    struct Incomplete;
    impl LagrangianHessian<f64> for Incomplete {
        fn hessian_into(
            &self,
            _: &[f64],
            _: &[f64],
            mut h: MatrixViewMut<'_, f64>,
        ) -> Result<(), SciOxideError> {
            h.set(0, 0, 2.)?;
            h.set(1, 1, 2.)
        }
    }
    let free = free(2);
    let problem = ConstrainedProblem::new(&Quadratic, &free).unwrap();
    let solver = InteriorPoint::<f64>::default();
    let mut w = solver.workspace(&problem).unwrap();
    let mut r = solver.minimize(&problem, &[2., 2.]).unwrap();
    let original = r.clone();
    let curved = problem.with_lagrangian_hessian(&Incomplete);
    assert!(solve(&solver, &curved, &[2., 2.], &mut r, &mut w).is_err());
    assert_eq!(r.x(), original.x());
    assert_eq!(r.counts(), original.counts());
}

#[allow(clippy::panic)]
fn interior_of<S: scioxide_core::RealScalar>(
    result: &crate::OptimizeResult<S>,
) -> crate::InteriorPointDiagnostics<S> {
    match result.method_diagnostics() {
        crate::OptimizeDiagnostics::InteriorPoint(diagnostics) => diagnostics,
        other => panic!("expected interior-point diagnostics, found {other:?}"),
    }
}
