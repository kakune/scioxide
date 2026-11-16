use super::*;
use core::cell::Cell;
use scioxide_core::{Budget, BudgetKind, ExecutionContext, ProblemToken, SolveBudgets, SolveState};

const TEST_ARGUMENT: scioxide_core::Argument = scioxide_core::argument!("fixture");

#[test]
fn layout_and_preconditioner_preflight_and_transactions() {
    assert!(PdeFieldLayout::new(&[]).is_err());
    assert!(PdeFieldLayout::new(&[0, 0]).is_err());
    assert!(PdeFieldLayout::new(&[usize::MAX, 1]).is_err());
    let fields = PdeFieldLayout::new(&[1, 0, 2]).unwrap();
    assert_eq!(fields.range(1).unwrap(), 1..1);
    assert!(fields.field(&[1., 2.], 0).is_err());
    assert!(fields.range(3).is_err());
    let mut x = [1., 2., 3.];
    fields.field_mut(&mut x, 2).unwrap().fill(4.);
    assert_eq!(x, [1., 4., 4.]);
    struct Block {
        dimension: Cell<usize>,
        calls: Cell<usize>,
        fail: Cell<bool>,
    }
    impl Preconditioner<f64> for Block {
        fn dimension(&self) -> usize {
            self.dimension.get()
        }
        fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn precondition_into(
            &self,
            x: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.calls.set(self.calls.get() + 1);
            out.copy_from_slice(x);
            if self.fail.get() {
                Err(invalid(TEST_ARGUMENT, "injected block failure"))
            } else {
                Ok(())
            }
        }
    }
    let first = Block {
        dimension: Cell::new(1),
        calls: Cell::new(0),
        fail: Cell::new(false),
    };
    let second = Block {
        dimension: Cell::new(2),
        calls: Cell::new(0),
        fail: Cell::new(false),
    };
    let empty = Inverse(Vec::<f64>::new());
    let blocks: [&dyn Preconditioner<f64>; 3] = [&first, &empty, &second];
    let pre = PdeFieldPreconditioner::new(&fields, &blocks).unwrap();
    let mut out = [9.; 3];
    let mut scratch = [0.; 3];
    second.dimension.set(3);
    assert!(pre.precondition_into(&x, &mut out, &mut scratch).is_err());
    assert_eq!(first.calls.get(), 0);
    assert_eq!(out, [9.; 3]);
    second.dimension.set(2);
    second.fail.set(true);
    assert!(pre.precondition_into(&x, &mut out, &mut scratch).is_err());
    assert_eq!(out, [9.; 3]);
    second.fail.set(false);
    pre.precondition_into(&x, &mut out, &mut scratch).unwrap();
    assert_eq!(out, x);
}

#[test]
fn invalid_bindings_missing_derivatives_and_callback_failure_preserve_outputs() {
    struct Incomplete(Cell<usize>);
    impl OdeSystem<f64> for Incomplete {
        fn dimension(&self) -> usize {
            2
        }
        fn rhs(
            &self,
            _: f64,
            _: &[f64],
            out: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            self.0.set(self.0.get() + 1);
            out[0] = 0.;
            Ok(())
        }
    }
    let provider = Incomplete(Cell::new(0));
    let fields = PdeFieldLayout::new(&[1, 1]).unwrap();
    // A provider of another dimension is rejected when bound, before any callback.
    let narrow_fields = PdeFieldLayout::new(&[1]).unwrap();
    let narrow_mode = PdeJacobianMode::Dense;
    assert!(
        NonlinearPdeProblem::new(&provider, 0., &narrow_fields, &narrow_mode)
            .bind(
                &mut NonlinearPdeProblem::new(&provider, 0.0, &narrow_fields, &narrow_mode)
                    .workspace()
                    .unwrap()
            )
            .is_err()
    );
    let plan_fields = fields.clone();
    let plan_mode = PdeJacobianMode::Dense;
    let mut w = NonlinearPdeProblem::new(&provider, 0.0, &plan_fields, &plan_mode)
        .workspace()
        .unwrap();
    assert!(
        NonlinearPdeProblem::new(&provider, f64::NAN, &plan_fields, &plan_mode)
            .bind(&mut w)
            .is_err()
    );
    // Assembled modes supply their own adjoint; a matrix-free one is rejected.
    let adjoint = |_: f64, _: &[f64], _: &[f64], _: &mut [f64], _: &mut [f64]| Ok(());
    assert!(
        NonlinearPdeProblem::new(&provider, 0., &plan_fields, &plan_mode)
            .with_adjoint(&adjoint, 0)
            .bind(&mut w)
            .is_err()
    );
    let residual = NonlinearPdeProblem::new(&provider, 0., &plan_fields, &plan_mode)
        .bind(&mut w)
        .unwrap();
    let mut out = [7.; 2];
    let mut scratch = [0.; 4];
    assert!(
        residual
            .residual_into(&[1.], &mut out, &mut scratch)
            .is_err()
    );
    assert!(
        residual
            .residual_into(&[1., 2.], &mut out, &mut [])
            .is_err()
    );
    assert_eq!(provider.0.get(), 0);
    assert!(
        residual
            .residual_into(&[1., 2.], &mut out, &mut scratch)
            .is_err()
    );
    assert_eq!(out, [7.; 2]);
    assert!(
        residual
            .jvp_into(&[1., 2.], &[1., 1.], &mut out, &mut scratch)
            .is_err()
    );
    assert_eq!(out, [7.; 2]);
    let held = residual.workspace.borrow_mut();
    assert!(
        residual
            .jvp_into(&[1., 2.], &[1., 1.], &mut out, &mut scratch)
            .is_err()
    );
    drop(held);
    let operator_fields = fields;
    let operator_mode = PdeJacobianMode::Operator {
        operator_scratch: 0,
    };
    assert!(
        NonlinearPdeProblem::new(&provider, 0., &operator_fields, &operator_mode)
            .bind(
                &mut NonlinearPdeProblem::new(&provider, 0.0, &plan_fields, &plan_mode)
                    .workspace()
                    .unwrap()
            )
            .is_err()
    );
    let mut w = NonlinearPdeProblem::new(&provider, 0.0, &operator_fields, &operator_mode)
        .workspace()
        .unwrap();
    let r = NonlinearPdeProblem::new(&provider, 0., &operator_fields, &operator_mode)
        .bind(&mut w)
        .unwrap();
    // Without a supplied adjoint the tangent's adjoint product is unavailable and
    // never falls back; its output is unchanged.
    let tangent = PdeTangent::new(&r, &[1., 2.]).unwrap();
    assert!(
        tangent
            .apply_adjoint_into(&[1., 1.], &mut out, &mut scratch)
            .is_err()
    );
    assert_eq!(out, [7.; 2]);
    assert_eq!(provider.0.get(), 1);
    assert!(
        r.jvp_into(&[1., 2.], &[1., 1.], &mut out, &mut scratch)
            .is_err()
    );
    assert_eq!(out, [7.; 2]);
}

#[test]
fn inner_failure_and_nested_budget_stops_retain_accepted_state() {
    let (a, csr, mass, target, laplace, pattern) = fixture::<f64>(4);
    let mut provider = ReactionDiffusion::new(&a, &mass, &target, &laplace).unwrap();
    provider.assembled = Some(&csr);
    let fields = PdeFieldLayout::new(&[4, 4]).unwrap();
    for mode in [
        PdeJacobianMode::Dense,
        PdeJacobianMode::Sparse(pattern),
        PdeJacobianMode::Operator {
            operator_scratch: 0,
        },
    ] {
        let plan_fields = fields.clone();
        let plan_mode = mode;
        let mut work = NonlinearPdeProblem::new(&provider, 0.0, &plan_fields, &plan_mode)
            .workspace()
            .unwrap();
        let residual = NonlinearPdeProblem::new(&provider, 0., &plan_fields, &plan_mode)
            .bind(&mut work)
            .unwrap();
        let problem = NewtonKrylovProblem::with_jacobian(&residual);
        let initial: Vec<_> = target.iter().map(|&x| x * 0.4).collect();
        let fail = options()
            .with_krylov(Budget::ONE, Budget::ONE)
            .with_forcing(scioxide_rootfind::NewtonKrylovForcing::Constant(1e-12));
        let report = NewtonKrylovSolver::new((fail).clone())
            .and_then(|root_plan| root_plan.solve(&problem, &initial))
            .unwrap();
        assert!(matches!(
            report.termination(),
            scioxide_rootfind::RootTermination::LinearFailure(_)
        ));
        assert_eq!(report.point(), initial);
        let opts = options();
        let mut w = NewtonKrylovSolver::new((opts).clone())
            .and_then(|root_plan| root_plan.workspace(&problem))
            .unwrap();
        let mut result = NewtonKrylovResult::new(8, &opts).unwrap();
        let identity = scioxide_core::SolveIdentity::new(
            ProblemToken::new(634),
            scioxide_core::algorithm_id!("test", "nonlinear-parent", 1),
            0,
            0,
        );
        for limit in [4, 5] {
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
                .unwrap();
            NewtonKrylovSolver::new((opts).clone())
                .and_then(|root_plan| {
                    root_plan.solve_into(&problem, &initial, &mut result, &mut w, &mut context)
                })
                .unwrap();
            assert!(!result.is_success());
            assert_eq!(
                result.termination(),
                scioxide_rootfind::RootTermination::WorkLimit(BudgetKind::Evaluation)
            );
            assert_eq!(result.point(), initial);
        }
        NewtonKrylovSolver::new((opts).clone())
            .and_then(|root_plan| {
                root_plan.solve_into(
                    &problem,
                    &initial,
                    &mut result,
                    &mut w,
                    scioxide_core::ExecutionContext::sequential_native(),
                )
            })
            .unwrap();
        assert!(result.is_success());
    }
}

fn strided(execution: ExecutionContext<'_>) -> (Vec<f64>, scioxide_core::EvaluationCounts) {
    let (a, _, mass, target, laplace, _) = fixture::<f64>(3);
    let provider = ReactionDiffusion::new(&a, &mass, &target, &laplace).unwrap();
    let fields = PdeFieldLayout::new(&[3, 3]).unwrap();
    let plan_fields = fields;
    let plan_mode = PdeJacobianMode::Operator {
        operator_scratch: 0,
    };
    let mut pw = NonlinearPdeProblem::new(&provider, 0.0, &plan_fields, &plan_mode)
        .workspace()
        .unwrap();
    let residual = NonlinearPdeProblem::new(&provider, 0., &plan_fields, &plan_mode)
        .bind(&mut pw)
        .unwrap();
    let problem = NewtonKrylovProblem::with_jacobian(&residual);
    let opts = options();
    let mut w = NewtonKrylovSolver::new((opts).clone())
        .and_then(|root_plan| root_plan.workspace(&problem))
        .unwrap();
    let mut result = NewtonKrylovResult::new(6, &opts).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(634),
        scioxide_core::algorithm_id!("test", "nonlinear-parent", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(execution, identity, &mut budgets)
        .unwrap();
    let input = vec![0.3; 12];
    let mut output = vec![99.; 12];
    let layout = scioxide_core::StridedLayout::new(&[6], &[2]).unwrap();
    NewtonKrylovSolver::new(opts)
        .unwrap()
        .solve_into(
            &problem,
            scioxide_core::NdView::new(&input, layout).unwrap(),
            &mut result,
            &mut w,
            &mut context,
        )
        .unwrap();
    assert!(result.is_success());
    for (i, &value) in result.point().iter().enumerate() {
        output[2 * i] = value;
    }
    for i in 0..6 {
        assert!((output[2 * i] - target[i]).abs() < 1e-9);
        assert_eq!(output[2 * i + 1], 99.);
    }
    (output, result.counts())
}
#[test]
fn borrowed_strided_states() {
    let _ = strided(ExecutionContext::sequential());
}
#[cfg(feature = "std")]
#[test]
fn injected_executor_replays_the_same_solve() {
    let pool = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    assert_eq!(
        strided(ExecutionContext::sequential()),
        strided(ExecutionContext::new(&pool))
    );
}

#[test]
fn continuation_reuses_pde_residuals_and_correction_engine() {
    use scioxide_rootfind::{
        Continuation, ContinuationOptions, ContinuationPath, ContinuationProblem, LinearHomotopy,
    };
    let (a, _, mass, target, laplace, _) = fixture::<f64>(3);
    let target2: Vec<_> = target.iter().map(|x| x * 1.4).collect();
    let laplace2: Vec<_> = laplace.iter().map(|x| x * 1.4).collect();
    let first = ReactionDiffusion::new(&a, &mass, &target, &laplace).unwrap();
    let last = ReactionDiffusion::new(&a, &mass, &target2, &laplace2).unwrap();
    // One immutable plan serves both endpoint problems.
    let plan_fields = PdeFieldLayout::new(&[3, 3]).unwrap();
    let plan_mode = PdeJacobianMode::Operator {
        operator_scratch: 0,
    };
    let mut w0 = NonlinearPdeProblem::new(&first, 0.0, &plan_fields, &plan_mode)
        .workspace()
        .unwrap();
    let mut w1 = NonlinearPdeProblem::new(&first, 0.0, &plan_fields, &plan_mode)
        .workspace()
        .unwrap();
    let r0 = NonlinearPdeProblem::new(&first, 0., &plan_fields, &plan_mode)
        .bind(&mut w0)
        .unwrap();
    let r1 = NonlinearPdeProblem::new(&last, 0., &plan_fields, &plan_mode)
        .bind(&mut w1)
        .unwrap();
    let homotopy = LinearHomotopy::new(&r0, &r1).unwrap();
    let problem = ContinuationProblem::new(&homotopy).unwrap();
    let opts = ContinuationOptions::new(0.2, 1e-6, 0.25, Budget::try_from(100).unwrap())
        .unwrap()
        .with_residual_tolerance(ResidualTolerance::new(1e-10, 0.).unwrap());
    let mut seed = target.clone();
    seed.push(0.);
    let continuation =
        Continuation::new(ContinuationPath::Parameters(vec![0.5, 1.]), opts).unwrap();
    let path = continuation.state(&problem, &seed).unwrap();
    let path = continuation.follow(&problem, path).unwrap();
    assert!(path.is_success(), "{:?}", path.termination());
    for (&x, &q) in path.point().iter().zip(&target2) {
        assert!((x - q).abs() < 1e-8);
    }
}

#[test]
fn cancellation_during_derivative_keeps_the_initial_accepted_point() {
    use scioxide_core::CancellationToken;
    struct Cancel<'a>(&'a CancellationToken);
    impl OdeSystem<f64> for Cancel<'_> {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(
            &self,
            t: f64,
            x: &[f64],
            out: &mut [f64],
            cx: &mut SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            Scalar.rhs(t, x, out, cx)
        }
        fn jacobian_product(
            &self,
            _: f64,
            x: &[f64],
            v: &[f64],
            out: &mut [f64],
            _: &mut [f64],
            _: &mut SolveContext<'_, '_>,
        ) -> Result<Option<()>, SciOxideError> {
            out[0] = 2. * x[0] * v[0];
            self.0.cancel();
            Ok(Some(()))
        }
    }
    let cancel = CancellationToken::new();
    let provider = Cancel(&cancel);
    let fields = PdeFieldLayout::new(&[1]).unwrap();
    let plan_fields = fields;
    let plan_mode = PdeJacobianMode::Operator {
        operator_scratch: 0,
    };
    let mut work = NonlinearPdeProblem::new(&provider, 0.0, &plan_fields, &plan_mode)
        .workspace()
        .unwrap();
    let residual = NonlinearPdeProblem::new(&provider, 0., &plan_fields, &plan_mode)
        .bind(&mut work)
        .unwrap();
    let problem = NewtonKrylovProblem::with_jacobian(&residual);
    let opts = options();
    let mut w = NewtonKrylovSolver::new((opts).clone())
        .and_then(|root_plan| root_plan.workspace(&problem))
        .unwrap();
    let mut result = NewtonKrylovResult::new(1, &opts).unwrap();
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(634),
        scioxide_core::algorithm_id!("test", "nonlinear-parent", 1),
        0,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancel),
            identity,
            &mut budgets,
        )
        .unwrap();
    NewtonKrylovSolver::new(opts)
        .and_then(|root_plan| {
            root_plan.solve_into(&problem, &[1.], &mut result, &mut w, &mut context)
        })
        .unwrap();
    assert_eq!(
        result.termination(),
        scioxide_rootfind::RootTermination::Cancelled
    );
    assert_eq!(result.point(), [1.]);
}
