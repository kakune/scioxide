//! Independent hand-derived original-coordinate witnesses.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Bounds, ExecutionContext, NumericScalar, RealScalar, SolveScope, Tolerance};
use scioxide_linalg::Matrix;
use scioxide_optimize::*;

fn exact_tolerances() -> OptimalityTolerances<f64> {
    let t = Tolerance::new(1e-12, 1e-12).unwrap();
    OptimalityTolerances::new(
        scioxide_core::FeasibilityTolerance::from_tolerance(t),
        scioxide_core::GradientTolerance::from_tolerance(t),
        scioxide_core::ResidualTolerance::from_tolerance(t),
        scioxide_core::RankThreshold::from_tolerance(t),
    )
}
fn verifier<S: RealScalar>(t: OptimalityTolerances<S>) -> ProgrammingVerifier<S> {
    ProgrammingVerifier::new(VerificationOptions::default().with_tolerances(t))
}
/// Owned verification with explicit thresholds.
fn check<'c, S, P>(
    problem: &P,
    claim: impl Into<ProgrammingClaim<'c, S>>,
    t: OptimalityTolerances<S>,
) -> Result<VerificationReport<S>, scioxide_core::SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S> + 'c,
    P: VerificationProblem<S> + ?Sized,
{
    verifier(t).verify(problem, claim)
}
/// Prepared verification in a fresh epoch of `scope`.
fn check_into<'c, 'scope, 'capability, S, P>(
    problem: &P,
    claim: impl Into<ProgrammingClaim<'c, S>>,
    t: OptimalityTolerances<S>,
    workspace: &mut VerificationWorkspace<S>,
    scope: impl Into<SolveScope<'scope, 'capability>>,
) -> Result<VerificationReport<S>, scioxide_core::SciOxideError>
where
    S: RealScalar + NumericScalar<Real = S> + 'c,
    P: VerificationProblem<S> + ?Sized,
{
    let mut report = VerificationReport::empty();
    verifier(t).verify_into(problem, claim, &mut report, workspace, scope)?;
    Ok(report)
}
#[test]
fn lp_optimum_and_corrupted_evidence() {
    // min -x-2y, x+y<=3, x,y>=0: x=0,y=3, row dual=2, lower dual=(1,0).
    let a = Matrix::from_vec(1, 2, vec![1., 1.]).unwrap();
    let b = Bounds::new(vec![(Some(0.), None), (Some(0.), None)]).unwrap();
    let a_finite = a.view().try_finite().unwrap();
    let p = LpVerificationProblem::new(&[-1., -2.])
        .unwrap()
        .with_constraints(&a_finite, &[3.], &[LinearConstraintSense::LessEqual])
        .unwrap()
        .with_bounds(&b)
        .unwrap();
    let certificate = ProgrammingCertificate::Optimality {
        point: &[0., 3.],
        dual: DualCertificate {
            rows: &[2.],
            lower: &[1., 0.],
            upper: &[0., 0.],
        },
    };
    let report = check(&p, certificate, exact_tolerances()).unwrap();
    assert!(report.is_accepted());
    assert!((report.primal_objective().unwrap() + 6.0).abs() <= 1e-12);
    assert!(report.objective_gap().unwrap().abs() <= 1e-12);
    assert!((report.dual_bound().unwrap() + 6.0).abs() <= 1e-12);
    for (point, rows, lower) in [
        ([0., 3.1], [2.], [1., 0.]),
        ([0., 3.], [1.], [1., 0.]),
        ([0., 3.], [2.], [-1., 0.]),
        ([1., 2.], [2.], [1., 0.]),
    ] {
        let r = check(
            &p,
            ProgrammingCertificate::Optimality {
                point: &point,
                dual: DualCertificate {
                    rows: &rows,
                    lower: &lower,
                    upper: &[0., 0.],
                },
            },
            exact_tolerances(),
        )
        .unwrap();
        assert!(!r.is_accepted(), "{r:?}");
    }
}
#[test]
fn normalized_farkas_including_bounds_and_corruption() {
    // x<=-1, x>=0 is impossible: y=alpha=1 gives contradiction 1.
    let a = Matrix::from_vec(1, 1, vec![1.]).unwrap();
    let b = Bounds::new(vec![(Some(0.), None)]).unwrap();
    let a_finite = a.view().try_finite().unwrap();
    let p = LpVerificationProblem::new(&[7.])
        .unwrap()
        .with_constraints(&a_finite, &[-1.], &[LinearConstraintSense::LessEqual])
        .unwrap()
        .with_bounds(&b)
        .unwrap();
    for scale in [1., 1e-200, 1e200] {
        let d = [scale];
        let report = check(
            &p,
            ProgrammingCertificate::Infeasibility(DualCertificate {
                rows: &d,
                lower: &d,
                upper: &[0.],
            }),
            exact_tolerances(),
        )
        .unwrap();
        assert!(report.is_accepted());
        assert!(report.witness_margin().unwrap() >= 1. && report.witness_margin().unwrap() < 2.);
    }
    let report = check(
        &p,
        ProgrammingCertificate::Infeasibility(DualCertificate {
            rows: &[1.],
            lower: &[0.],
            upper: &[0.],
        }),
        exact_tolerances(),
    )
    .unwrap();
    assert!(!report.is_accepted());
}
#[test]
fn recession_requires_base_point_direction_and_strict_descent() {
    let b = Bounds::new(vec![(Some(0.), None)]).unwrap();
    let p = LpVerificationProblem::new(&[-2.])
        .unwrap()
        .with_bounds(&b)
        .unwrap();
    for (point, direction, accepted) in [
        (0., 1., true),
        (0., 1e-200, true),
        (-1., 1., false),
        (0., -1., false),
        (0., 0., false),
    ] {
        let r = check(
            &p,
            ProgrammingCertificate::Unboundedness {
                point: &[point],
                direction: &[direction],
            },
            exact_tolerances(),
        )
        .unwrap();
        assert_eq!(r.is_accepted(), accepted);
    }
}
#[test]
fn qp_original_objective_convexity_and_operator_limit() {
    let q = Matrix::from_vec(2, 2, vec![2., 0., 0., 0.]).unwrap();
    let p = QpProblem::new(q.clone(), vec![-2., 0.]).unwrap();
    let cert = ProgrammingCertificate::Optimality {
        point: &[1., 5.],
        dual: DualCertificate {
            rows: &[],
            lower: &[0., 0.],
            upper: &[0., 0.],
        },
    };
    let r = check(
        &QpVerificationProblem::from_problem(&p),
        cert,
        exact_tolerances(),
    )
    .unwrap();
    assert!(r.is_accepted());
    assert!((r.primal_objective().unwrap() + 1.0).abs() <= 1e-12);
    let q_finite = q.view().try_finite().unwrap();
    let op = QpVerificationProblem::new(&q_finite, LpVerificationProblem::new(&[-2., 0.]).unwrap())
        .unwrap();
    let r = check(&op, cert, exact_tolerances()).unwrap();
    assert_eq!(r.status(), VerificationStatus::KktOnly);
    assert_eq!(r.dual_bound(), None);
    let indefinite = QpProblem::new(Matrix::from_vec(1, 1, vec![-1.]).unwrap(), vec![0.]).unwrap();
    let r = check(
        &QpVerificationProblem::from_problem(&indefinite),
        ProgrammingCertificate::Optimality {
            point: &[0.],
            dual: DualCertificate {
                rows: &[],
                lower: &[0.],
                upper: &[0.],
            },
        },
        exact_tolerances(),
    )
    .unwrap();
    assert!(!r.is_accepted());
}
#[test]
fn qp_null_direction_and_positive_curvature() {
    for curvature in [0., 1e-16, 1.] {
        let p =
            QpProblem::new(Matrix::from_vec(1, 1, vec![curvature]).unwrap(), vec![-1.]).unwrap();
        let r = check(
            &QpVerificationProblem::from_problem(&p),
            ProgrammingCertificate::Unboundedness {
                point: &[0.],
                direction: &[1.],
            },
            exact_tolerances(),
        )
        .unwrap();
        assert_eq!(r.is_accepted(), curvature == 0.);
    }
}
#[test]
fn f32_witness_and_workspace_shape_errors() {
    let p = LpProblem::new(vec![1f32]).unwrap();
    let view = LpVerificationProblem::from_problem(&p);
    let cert = ProgrammingCertificate::Optimality {
        point: &[0f32],
        dual: DualCertificate {
            rows: &[],
            lower: &[1f32],
            upper: &[0f32],
        },
    };
    let r = check(&view, cert, OptimalityTolerances::default()).unwrap();
    assert!(r.is_accepted());
    let other = LpVerificationProblem::new(&[1f32, 1.]).unwrap();
    let mut w = ProgrammingVerifier::default().workspace(&other).unwrap();
    assert!(
        check_into(
            &view,
            cert,
            OptimalityTolerances::default(),
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert!(
        check(
            &view,
            ProgrammingCertificate::Unboundedness {
                point: &[f32::NAN],
                direction: &[1.]
            },
            OptimalityTolerances::default()
        )
        .is_err()
    );
}

#[test]
fn solver_evidence_reverifies() {
    let constraints = LinearConstraints::new(
        Matrix::from_vec(2, 2, vec![1., 1., -1., -1.]).unwrap(),
        vec![1., -3.],
        vec![LinearConstraintSense::LessEqual; 2],
    )
    .unwrap();
    let p = LpProblem::new(vec![1f64, 1.])
        .unwrap()
        .with_constraints(constraints)
        .unwrap();
    let r = SimplexSolver::default().solve(&p).unwrap();
    assert!(
        check(
            &LpVerificationProblem::from_problem(&p),
            &r,
            exact_tolerances()
        )
        .unwrap()
        .is_accepted(),
        "{r:?}"
    );
}

macro_rules! backend_evidence {
    ($name:ident,$s:ty) => {
        #[test]
        fn $name() {
            use scioxide_core::{AdjointOperator, LinearOperator};
            struct Row($s);
            impl LinearOperator<$s> for Row {
                fn rows(&self) -> usize {
                    1
                }
                fn cols(&self) -> usize {
                    2
                }
                fn apply_workspace_len(&self) -> Result<usize, scioxide_core::SciOxideError> {
                    Ok(0)
                }
                fn apply_into(
                    &self,
                    x: &[$s],
                    y: &mut [$s],
                    _: &mut [$s],
                ) -> Result<(), scioxide_core::SciOxideError> {
                    y[0] = self.0 * (x[0] + x[1]);
                    Ok(())
                }
            }
            impl AdjointOperator<$s> for Row {
                fn apply_adjoint_workspace_len(
                    &self,
                ) -> Result<usize, scioxide_core::SciOxideError> {
                    Ok(0)
                }
                fn apply_adjoint_into(
                    &self,
                    x: &[$s],
                    y: &mut [$s],
                    _: &mut [$s],
                ) -> Result<(), scioxide_core::SciOxideError> {
                    y.fill(self.0 * x[0]);
                    Ok(())
                }
            }
            for scale in [(1.0 as $s) / (1048576.0 as $s), 1.0 as $s, 1048576.0 as $s] {
                let dense = Matrix::from_vec(1, 2, vec![scale, scale]).unwrap();
                let sparse = scioxide_sparse::CsrMatrix::from_raw(
                    1,
                    2,
                    vec![0, 2],
                    vec![0, 1],
                    vec![scale, scale],
                )
                .unwrap();
                let view = dense.view();
                let operator = Row(scale);
                let bounds = Bounds::new(vec![(Some(0. as $s), None); 2]).unwrap();
                let rhs = [3. as $s * scale];
                let y = [2. as $s / scale];
                for a in [
                    &dense.view().try_finite().unwrap() as &dyn AdjointOperator<$s>,
                    &sparse,
                    &view.try_finite().unwrap(),
                    &operator,
                ] {
                    let p = LpVerificationProblem::new(&[-1. as $s, -2. as $s])
                        .unwrap()
                        .with_constraints(a, &rhs, &[LinearConstraintSense::LessEqual])
                        .unwrap()
                        .with_bounds(&bounds)
                        .unwrap();
                    let mut workspace = ProgrammingVerifier::default().workspace(&p).unwrap();
                    for _ in 0..2 {
                        let r = check_into(
                            &p,
                            ProgrammingCertificate::Optimality {
                                point: &[0. as $s, 3. as $s],
                                dual: DualCertificate {
                                    rows: &y,
                                    lower: &[1. as $s, 0. as $s],
                                    upper: &[0. as $s, 0. as $s],
                                },
                            },
                            OptimalityTolerances::default(),
                            &mut workspace,
                            ExecutionContext::sequential(),
                        )
                        .unwrap();
                        assert!(r.is_accepted(), "{r:?}");
                        assert_eq!(r.primal_objective(), Some(-6. as $s));
                        assert_eq!(r.dual_bound(), Some(-6. as $s));
                    }
                }
            }
        }
    };
}
backend_evidence!(original_scale_backend_parity_f32, f32);
backend_evidence!(original_scale_backend_parity_f64, f64);

#[test]
fn large_objective_cannot_hide_primal_violation() {
    let bounds = Bounds::new(vec![(Some(0.), Some(1.))]).unwrap();
    let p = LpVerificationProblem::new(&[-1e20])
        .unwrap()
        .with_bounds(&bounds)
        .unwrap();
    let r = check(
        &p,
        ProgrammingCertificate::Optimality {
            point: &[1.0001],
            dual: DualCertificate {
                rows: &[],
                lower: &[0.],
                upper: &[1e20],
            },
        },
        exact_tolerances(),
    )
    .unwrap();
    assert!(!r.is_accepted());
    assert!(r.bound_violation() > 9e-5);
}

#[test]
fn mixed_senses_fixed_and_absent_bounds() {
    let a = Matrix::from_vec(2, 2, vec![-1., -1., 1., 0.]).unwrap();
    let b = Bounds::new(vec![(Some(0.), Some(0.)), (None, None)]).unwrap();
    let a_finite = a.view().try_finite().unwrap();
    let p = LpVerificationProblem::new(&[-1., -2.])
        .unwrap()
        .with_constraints(
            &a_finite,
            &[-3., 0.],
            &[
                LinearConstraintSense::GreaterEqual,
                LinearConstraintSense::Equal,
            ],
        )
        .unwrap()
        .with_bounds(&b)
        .unwrap();
    let r = check(
        &p,
        ProgrammingCertificate::Optimality {
            point: &[0., 3.],
            dual: DualCertificate {
                rows: &[-2., -1.],
                lower: &[0., 0.],
                upper: &[0., 0.],
            },
        },
        exact_tolerances(),
    )
    .unwrap();
    assert!(r.is_accepted());
    let r = check(
        &p,
        ProgrammingCertificate::Optimality {
            point: &[0., 3.],
            dual: DualCertificate {
                rows: &[-2., -1.],
                lower: &[0., 1e-16],
                upper: &[0., 0.],
            },
        },
        exact_tolerances(),
    )
    .unwrap();
    assert!(!r.is_accepted());
}

#[test]
fn verifier_budget_admits_each_operator_once_and_rejects_bad_shapes_first() {
    use scioxide_core::{
        Budget, BudgetKind, EvaluationRole, ExecutionContext, ProblemToken, SolveBudgets,
        SolveIdentity, SolveState,
    };
    let a = Matrix::from_vec(1, 1, vec![1.]).unwrap();
    let a_finite = a.view().try_finite().unwrap();
    let p = LpVerificationProblem::new(&[-1.])
        .unwrap()
        .with_constraints(&a_finite, &[1.], &[LinearConstraintSense::LessEqual])
        .unwrap();
    let mut w = ProgrammingVerifier::default().workspace(&p).unwrap();
    let identity = SolveIdentity::new(
        ProblemToken::new(602),
        scioxide_core::algorithm_id!("optimize", "verification", 1),
        1,
        0,
    );
    for limit in [1, 2] {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(
            BudgetKind::Evaluation,
            Budget::ONE.saturating_add(limit - 1),
        );
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let malformed = ProgrammingCertificate::Optimality {
            point: &[],
            dual: DualCertificate {
                rows: &[1.],
                lower: &[0.],
                upper: &[0.],
            },
        };
        assert!(check_into(&p, malformed, exact_tolerances(), &mut w, &mut context).is_err());
        assert_eq!(
            context
                .counts()
                .nested()
                .provider(EvaluationRole::LinearOperator),
            0
        );
        let certificate = ProgrammingCertificate::Optimality {
            point: &[1.],
            dual: DualCertificate {
                rows: &[1.],
                lower: &[0.],
                upper: &[0.],
            },
        };
        let r = check_into(&p, certificate, exact_tolerances(), &mut w, &mut context).unwrap();
        // Operator applications run in the verifier's child epoch and charge the parent.
        assert_eq!(
            context
                .counts()
                .nested()
                .provider(EvaluationRole::LinearOperator),
            limit
        );
        assert_eq!(r.is_accepted(), limit == 2);
        if limit == 1 {
            assert_eq!(r.status(), VerificationStatus::Incomplete);
            assert_eq!(r.termination(), Some(OptimizeTermination::EvaluationLimit));
        }
    }
}

#[test]
fn executor_context_and_cancellation_preserve_verifier_contract() {
    use core::num::NonZeroUsize;
    use scioxide_core::{
        CancellationToken, ExecutionContext, ExecutionPlan, Executor, PartitionFn, ProblemToken,
        SciOxideError, SolveBudgets, SolveIdentity, SolveState,
    };
    struct CallerExecutor;
    impl Executor for CallerExecutor {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(4).unwrap()
        }
        fn execute(
            &self,
            _: ExecutionContext<'_>,
            _: ExecutionPlan,
            _: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("plan"),
                scioxide_core::InvalidInputReason::UnsupportedConfiguration,
                "verification reductions are sequential",
            ))
        }
    }
    let problem = QpProblem::new(Matrix::from_vec(1, 1, vec![2.]).unwrap(), vec![-2.]).unwrap();
    let view = QpVerificationProblem::from_problem(&problem);
    let mut workspace = ProgrammingVerifier::default().workspace(&view).unwrap();
    let certificate = ProgrammingCertificate::Optimality {
        point: &[1.],
        dual: DualCertificate {
            rows: &[],
            lower: &[0.],
            upper: &[0.],
        },
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(602),
        scioxide_core::algorithm_id!("optimize", "verification", 1),
        1,
        0,
    );
    for cancelled in [false, true] {
        let cancellation = CancellationToken::new();
        if cancelled {
            cancellation.cancel();
        }
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                ExecutionContext::new(&CallerExecutor).with_cancellation(&cancellation),
                identity,
                &mut budgets,
            )
            .unwrap();
        let report = check_into(
            &view,
            certificate,
            exact_tolerances(),
            &mut workspace,
            &mut context,
        );
        if cancelled {
            // A cancelled scope is rejected in preflight before any operator work.
            assert!(matches!(
                report.unwrap_err().view(),
                scioxide_core::SciOxideErrorView::ExecutionStopped { .. }
            ));
            assert_eq!(
                workspace.last_attempt().unwrap().outcome(),
                scioxide_core::AttemptOutcome::PreflightRejected
            );
            assert_eq!(
                context.counts().nested(),
                scioxide_core::EvaluationCounts::new()
            );
        } else {
            assert!(report.unwrap().is_accepted());
        }
    }
}

#[test]
fn provider_failure_and_nonfinite_products_never_accept() {
    use scioxide_core::{AdjointOperator, LinearOperator, SciOxideError};
    struct Bad(bool);
    impl LinearOperator<f64> for Bad {
        fn rows(&self) -> usize {
            1
        }
        fn cols(&self) -> usize {
            1
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_into(
            &self,
            _: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            if self.0 {
                return Err(SciOxideError::invalid_input(
                    scioxide_core::argument!("input"),
                    scioxide_core::InvalidInputReason::Malformed,
                    "provider failure",
                ));
            }
            out[0] = f64::NAN;
            Ok(())
        }
    }
    impl AdjointOperator<f64> for Bad {
        fn apply_adjoint_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_adjoint_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            w: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.apply_into(x, y, w)
        }
    }
    for fail in [false, true] {
        let operator = Bad(fail);
        let p = LpVerificationProblem::new(&[-1.])
            .unwrap()
            .with_constraints(&operator, &[1.], &[LinearConstraintSense::LessEqual])
            .unwrap();
        let result = check(
            &p,
            ProgrammingCertificate::Optimality {
                point: &[1.],
                dual: DualCertificate {
                    rows: &[1.],
                    lower: &[0.],
                    upper: &[0.],
                },
            },
            exact_tolerances(),
        );
        if fail {
            assert!(matches!(
                (result)
                    .as_ref()
                    .map_err(scioxide_core::SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::InvalidInput { .. })
            ));
        } else {
            assert!(matches!(
                (result)
                    .as_ref()
                    .map_err(scioxide_core::SciOxideError::view),
                Err(scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Callback,
                    ..
                })
            ));
        }
    }
}

#[test]
fn pivot_rank_and_reduced_cost_roles_do_not_replace_certificate_checks() {
    let p = LpProblem::new(vec![-1f64])
        .unwrap()
        .with_constraints(
            LinearConstraints::new(
                Matrix::from_vec(1, 1, vec![1.]).unwrap(),
                vec![1.],
                vec![LinearConstraintSense::LessEqual],
            )
            .unwrap(),
        )
        .unwrap();
    let tight = Tolerance::new(1e-12, 1e-12).unwrap();
    let loose = Tolerance::new(10., 0.).unwrap();
    let base = OptimalityTolerances::new(
        scioxide_core::FeasibilityTolerance::from_tolerance(tight),
        scioxide_core::GradientTolerance::from_tolerance(tight),
        scioxide_core::ResidualTolerance::from_tolerance(tight),
        scioxide_core::RankThreshold::from_tolerance(loose),
    )
    .with_pivot(scioxide_core::ResidualTolerance::from_tolerance(tight));
    let result = SimplexSolver::default()
        .with_tolerances(base)
        .solve(&p)
        .unwrap();
    assert!(result.is_success());
    assert_eq!(result.x(), &[1.]);
    let result = SimplexSolver::default()
        .with_tolerances(
            base.with_reduced_cost(scioxide_core::GradientTolerance::from_tolerance(loose)),
        )
        .solve(&p)
        .unwrap();
    assert!(!result.is_success());
    assert_eq!(result.termination(), OptimizeTermination::Stagnation);
}

macro_rules! quadratic_witness_width {
    ($name:ident,$s:ty) => {
        #[test]
        fn $name() {
            let dense =
                Matrix::from_vec(2, 2, vec![2. as $s, 0. as $s, 0. as $s, 0. as $s]).unwrap();
            let view = dense.view();
            let problem = QpVerificationProblem::from_dense(
                &view,
                LpVerificationProblem::new(&[-2. as $s, 0. as $s]).unwrap(),
            )
            .unwrap();
            let r = check(
                &problem,
                ProgrammingCertificate::Optimality {
                    point: &[1. as $s, 0. as $s],
                    dual: DualCertificate {
                        rows: &[],
                        lower: &[0. as $s, 0. as $s],
                        upper: &[0. as $s, 0. as $s],
                    },
                },
                OptimalityTolerances::default(),
            )
            .unwrap();
            assert!(r.is_accepted());
            assert_eq!(r.dual_bound(), Some(-1. as $s));
            let sparse =
                scioxide_sparse::CsrMatrix::from_raw(2, 2, vec![0, 1, 1], vec![0], vec![2. as $s])
                    .unwrap();
            let problem = QpVerificationProblem::from_sparse(
                &sparse,
                LpVerificationProblem::new(&[-2. as $s, 0. as $s]).unwrap(),
            )
            .unwrap();
            let r = check(
                &problem,
                ProgrammingCertificate::Optimality {
                    point: &[1. as $s, 0. as $s],
                    dual: DualCertificate {
                        rows: &[],
                        lower: &[0. as $s, 0. as $s],
                        upper: &[0. as $s, 0. as $s],
                    },
                },
                OptimalityTolerances::default(),
            )
            .unwrap();
            assert!(r.is_accepted());
            assert_eq!(r.dual_bound(), Some(-1. as $s));

            let bounds = Bounds::new(vec![(Some(1. as $s), None), (None, None)]).unwrap();
            let a = Matrix::from_vec(1, 2, vec![1. as $s, 0. as $s]).unwrap();
            let a_finite = a.view().try_finite().unwrap();
            let linear = LpVerificationProblem::new(&[0. as $s, 0. as $s])
                .unwrap()
                .with_constraints(&a_finite, &[0. as $s], &[LinearConstraintSense::LessEqual])
                .unwrap()
                .with_bounds(&bounds)
                .unwrap();
            let dense_finite = dense.view().try_finite().unwrap();
            let problem = QpVerificationProblem::new(&dense_finite, linear).unwrap();
            for (y, pass) in [(1. as $s, true), (-1. as $s, false)] {
                let r = check(
                    &problem,
                    ProgrammingCertificate::Infeasibility(DualCertificate {
                        rows: &[y],
                        lower: &[y, 0. as $s],
                        upper: &[0. as $s, 0. as $s],
                    }),
                    OptimalityTolerances::default(),
                )
                .unwrap();
                assert_eq!(r.is_accepted(), pass);
            }
        }
    };
}
quadratic_witness_width!(quadratic_evidence_f32, f32);
quadratic_witness_width!(quadratic_evidence_f64, f64);

#[test]
fn domain_problems_verify_like_borrowed_views() {
    use scioxide_optimize::{LpProblem, QpProblem, QpSolver, SimplexSolver};
    let lp = LpProblem::new(vec![1.0_f64, 2.0])
        .unwrap()
        .with_bounds(Bounds::new(vec![(Some(0.0), None); 2]).unwrap())
        .unwrap();
    let result = SimplexSolver::default().solve(&lp).unwrap();
    let verifier = ProgrammingVerifier::default();
    let direct = verifier.verify(&lp, &result).unwrap();
    let borrowed = verifier
        .verify(&LpVerificationProblem::from_problem(&lp), &result)
        .unwrap();
    assert_eq!(direct.status(), borrowed.status());
    assert_eq!(direct.primal_objective(), borrowed.primal_objective());
    assert_eq!(direct.objective_gap(), borrowed.objective_gap());
    let qp = QpProblem::new(
        Matrix::from_rows(&[[2.0_f64, 0.0], [0.0, 2.0]]).unwrap(),
        vec![-2.0, -4.0],
    )
    .unwrap();
    let result = QpSolver::default().solve(&qp).unwrap();
    let direct = verifier.verify(&qp, &result).unwrap();
    let borrowed = verifier
        .verify(&QpVerificationProblem::from_problem(&qp), &result)
        .unwrap();
    assert_eq!(direct.status(), borrowed.status());
    assert_eq!(direct.primal_objective(), borrowed.primal_objective());
    assert_eq!(direct.objective_gap(), borrowed.objective_gap());
}
