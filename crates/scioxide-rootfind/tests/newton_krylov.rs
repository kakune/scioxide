//! Independent manufactured nonlinear equations and prepared resource contracts.
#![expect(
    clippy::unwrap_used,
    reason = "Tests require successful setup and inspect expected results."
)]
use core::cell::Cell;

use scioxide_core::{
    Budget, BudgetKind, EvaluationRole, ExecutionContext, JacobianOperator, NumericScalar,
    ProblemToken, RealScalar, ResidualOperator, ResidualTolerance, SciOxideError, SolveBudgets,
    SolveState, StepTolerance,
};
use scioxide_rootfind::{
    NewtonKrylovForcing, NewtonKrylovOptions, NewtonKrylovProblem, NewtonKrylovResult,
    NewtonKrylovSolver, RootTermination,
};

struct System<T> {
    n: usize,
    root: T,
    residual_calls: Cell<usize>,
    jvp_calls: Cell<usize>,
    fail: Cell<usize>,
}
impl<T: NumericScalar> System<T> {
    fn new(n: usize, root: T) -> Self {
        Self {
            n,
            root,
            residual_calls: Cell::new(0),
            jvp_calls: Cell::new(0),
            fail: Cell::new(usize::MAX),
        }
    }
}
fn r<R: RealScalar>(value: f64) -> R {
    R::try_from_f64_lossy(value).unwrap()
}
impl<T: NumericScalar> ResidualOperator<T> for System<T> {
    fn input_dimension(&self) -> usize {
        self.n
    }
    fn output_dimension(&self) -> usize {
        self.n
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn residual_into(&self, x: &[T], y: &mut [T], _: &mut [T]) -> Result<(), SciOxideError> {
        self.residual_calls.set(self.residual_calls.get() + 1);
        if self.residual_calls.get() == self.fail.get() {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("residual"),
                scioxide_core::InvalidInputReason::Malformed,
                "failure",
            ));
        }
        let c = T::from_real(r(0.05));
        for (i, out) in y.iter_mut().enumerate() {
            *out = x[i] * x[i] - self.root * self.root;
            if i > 0 {
                *out = *out + c * (x[i - 1] - self.root);
            }
            if i + 1 < self.n {
                *out = *out + c * (x[i + 1] - self.root);
            }
        }
        Ok(())
    }
}
impl<T: NumericScalar> JacobianOperator<T> for System<T> {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn jvp_into(&self, x: &[T], v: &[T], y: &mut [T], _: &mut [T]) -> Result<(), SciOxideError> {
        self.jvp_calls.set(self.jvp_calls.get() + 1);
        let two = T::ONE + T::ONE;
        let c = T::from_real(r(0.05));
        for (i, out) in y.iter_mut().enumerate() {
            *out = two * x[i] * v[i];
            if i > 0 {
                *out = *out + c * v[i - 1];
            }
            if i + 1 < self.n {
                *out = *out + c * v[i + 1];
            }
        }
        Ok(())
    }
}
fn options<R: RealScalar>() -> NewtonKrylovOptions<R> {
    let tolerance = R::EPSILON * r(100.0);
    NewtonKrylovOptions::default()
        .with_residual_tolerance(ResidualTolerance::new(tolerance, R::ZERO).unwrap())
        .with_step_tolerance(StepTolerance::new(R::EPSILON, R::EPSILON).unwrap())
        .with_history(true)
}
fn verify<T: NumericScalar>(system: &System<T>, result: &NewtonKrylovResult<T>) {
    assert!(result.is_success(), "{:?}", result);
    let mut f = vec![T::ZERO; system.n];
    system
        .residual_into(result.point(), &mut f, &mut [])
        .unwrap();
    let tol = T::Real::EPSILON * r(150.0);
    for ((&x, &reported), &independent) in result.point().iter().zip(result.residual()).zip(&f) {
        assert!((x - system.root).abs() < tol);
        assert!((reported - independent).abs() < tol);
    }
    assert!(result.history().iter().all(|h| {
        h.linear_residual()
            .is_some_and(|v| v <= h.forcing() * h.residual_norm() * r(1.0001))
    }));
}

#[test]
fn exact_evaluation_boundaries_preserve_accepted_nonlinear_state() {
    let system = System::new(3, 1.0f64);
    let opts = options();
    for analytic in [true, false] {
        let problem = if analytic {
            NewtonKrylovProblem::with_jacobian(&system)
        } else {
            NewtonKrylovProblem::numerical(&system)
        };
        let complete = NewtonKrylovSolver::new((opts).clone())
            .and_then(|root_plan| root_plan.solve(&problem, &[0.4; 3]))
            .unwrap();
        let total = complete
            .accounting()
            .total()
            .unwrap()
            .total_providers()
            .unwrap();
        for limit in 1..=total {
            system.residual_calls.set(0);
            system.jvp_calls.set(0);
            let mut w = NewtonKrylovSolver::new((opts).clone())
                .and_then(|root_plan| root_plan.workspace(&problem))
                .unwrap();
            let mut output = NewtonKrylovResult::new(3, &opts).unwrap();
            let mut state = SolveState::new();
            let mut budgets = SolveBudgets::unbounded()
                .with_limit(BudgetKind::Evaluation, Budget::try_from(limit).unwrap());
            let identity = scioxide_core::SolveIdentity::new(
                ProblemToken::new(5),
                scioxide_core::algorithm_id!("test", "nonlinear-parent", 1),
                0,
                0,
            );
            let mut context = state
                .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
                .unwrap();
            NewtonKrylovSolver::new((opts).clone())
                .and_then(|root_plan| {
                    root_plan.solve_into(&problem, &[0.4; 3], &mut output, &mut w, &mut context)
                })
                .unwrap();
            assert!(
                output
                    .accounting()
                    .total()
                    .unwrap()
                    .total_providers()
                    .unwrap()
                    <= limit
            );
            assert_eq!(
                system.residual_calls.get(),
                output
                    .accounting()
                    .total()
                    .unwrap()
                    .provider(EvaluationRole::Residual)
                    + output
                        .accounting()
                        .total()
                        .unwrap()
                        .provider(EvaluationRole::Function)
            );
            if analytic {
                assert_eq!(
                    system.jvp_calls.get(),
                    output
                        .accounting()
                        .total()
                        .unwrap()
                        .provider(EvaluationRole::LinearOperator)
                );
            }
            let mut independent = [0.; 3];
            system
                .residual_into(output.point(), &mut independent, &mut [])
                .unwrap();
            assert_eq!(output.residual(), independent);
            if limit < total {
                assert_eq!(
                    output.termination(),
                    RootTermination::WorkLimit(scioxide_core::BudgetKind::Evaluation),
                    "limit {limit}/{total}"
                );
            } else {
                assert!(output.is_success());
            }
        }
    }
}
#[test]
fn callback_failure_is_transactional_and_workspace_is_reusable() {
    let system = System::new(3, 1.0f64);
    let problem = NewtonKrylovProblem::with_jacobian(&system);
    let opts = options();
    let mut w = NewtonKrylovSolver::new((opts).clone())
        .and_then(|root_plan| root_plan.workspace(&problem))
        .unwrap();
    let mut result = NewtonKrylovSolver::new((opts).clone())
        .and_then(|root_plan| root_plan.solve(&problem, &[0.4; 3]))
        .unwrap();
    let before = result.clone();
    system.residual_calls.set(0);
    system.fail.set(2);
    assert_eq!(
        NewtonKrylovSolver::new((opts).clone())
            .and_then(|root_plan| root_plan.solve_into(
                &problem,
                &[0.2; 3],
                &mut result,
                &mut w,
                scioxide_core::ExecutionContext::sequential_native()
            ))
            .unwrap_err()
            .kind(),
        scioxide_core::SciOxideErrorKind::InvalidInput
    );
    assert_eq!(result.point(), before.point());
    assert_eq!(result.accounting(), before.accounting());
    system.fail.set(usize::MAX);
    NewtonKrylovSolver::new(opts)
        .and_then(|root_plan| {
            root_plan.solve_into(
                &problem,
                &[0.2; 3],
                &mut result,
                &mut w,
                scioxide_core::ExecutionContext::sequential_native(),
            )
        })
        .unwrap();
    verify(&system, &result);
}
#[test]
fn fixed_forcing_and_backtracking_are_observable() {
    let system = System::new(3, 1.0f64);
    let problem = NewtonKrylovProblem::with_jacobian(&system);
    let opts = options().with_forcing(NewtonKrylovForcing::Constant(0.01));
    let result = NewtonKrylovSolver::new(opts)
        .and_then(|root_plan| root_plan.solve(&problem, &[0.05; 3]))
        .unwrap();
    verify(&system, &result);
    assert!(result.accounting().total().unwrap().rejected() > 0);
    assert!(result.history().iter().any(|h| h.step_length() < 1.0));
    assert!(result.history().iter().all(|h| h.forcing() == 0.01));
}

#[test]
fn linear_failure_stops_but_small_improving_steps_continue() {
    let system = System::new(9, 1.0f64);
    let problem = NewtonKrylovProblem::with_jacobian(&system);
    let opts = options()
        .with_krylov(Budget::ONE, Budget::ONE)
        .with_forcing(NewtonKrylovForcing::Constant(1e-12));
    let initial = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let result = NewtonKrylovSolver::new(opts)
        .and_then(|root_plan| root_plan.solve(&problem, &initial))
        .unwrap();
    assert!(matches!(
        result.termination(),
        RootTermination::LinearFailure(_)
    ));
    assert_eq!(result.point(), initial);
    assert!(!result.is_success());
    let opts = options().with_step_tolerance(StepTolerance::new(100., 0.).unwrap());
    let result = NewtonKrylovSolver::new(opts)
        .and_then(|root_plan| root_plan.solve(&problem, &initial))
        .unwrap();
    verify(&system, &result);
}
#[test]
fn forward_difference_products_cost_one_residual_each() {
    for scale in [1e-100, 1., 1e100] {
        let system = System::new(4, scale);
        let problem = NewtonKrylovProblem::numerical(&system);
        let opts = options().with_residual_tolerance(ResidualTolerance::new(0., 1e-14).unwrap());
        let result = NewtonKrylovSolver::new((opts).clone())
            .and_then(|root_plan| root_plan.solve(&problem, &[0.4 * scale; 4]))
            .unwrap();
        assert!(result.is_success(), "{result:?}");
        for x in result.point() {
            assert!((x / scale - 1.).abs() < 1e-12);
        }
        let total = result.accounting().total().unwrap();
        assert_eq!(system.jvp_calls.get(), 0);
        assert_eq!(total.provider(EvaluationRole::Function), 0);
        assert_eq!(
            system.residual_calls.get(),
            total.provider(EvaluationRole::Residual)
        );
        // Initial residual, one per nonlinear trial, at most one probe per
        // admitted operator product (zero directions need no probe).
        assert!(
            total.provider(EvaluationRole::Residual)
                <= 1 + total.proposals() + total.provider(EvaluationRole::LinearOperator)
        );
        assert!(
            total.provider(EvaluationRole::Residual) > 1 + total.proposals() + result.iterations()
        );
    }
}
#[test]
fn nonmonotone_weight_is_validated_and_bounds_converge() {
    for weight in [-0.1, 1.5, f64::NAN] {
        assert!(NewtonKrylovSolver::<f64>::new(options::<f64>().with_nonmonotone(weight)).is_err());
    }
    let system = System::new(3, 1.0f64);
    for weight in [0.0, 1.0] {
        let opts = options().with_nonmonotone(weight);
        let result = NewtonKrylovSolver::new((opts).clone())
            .and_then(|root_plan| {
                root_plan.solve(&NewtonKrylovProblem::with_jacobian(&system), &[0.05; 3])
            })
            .unwrap();
        verify(&system, &result);
    }
}
#[test]
fn invalid_preparation_is_explicit() {
    let system = System::new(3, 1.0f64);
    let opts = options();
    let problem = NewtonKrylovProblem::numerical(&system);
    let empty = System::new(0, 1.0f64);
    assert!(
        NewtonKrylovSolver::new((opts).clone())
            .and_then(|root_plan| root_plan.workspace(&NewtonKrylovProblem::numerical(&empty)))
            .is_err()
    );
    for forcing in [0., 1., f64::NAN, -0.1] {
        assert!(
            NewtonKrylovSolver::<f64>::new(
                options::<f64>().with_forcing(NewtonKrylovForcing::Constant(forcing))
            )
            .is_err()
        );
    }
    let mut w = NewtonKrylovSolver::new((opts).clone())
        .and_then(|root_plan| root_plan.workspace(&problem))
        .unwrap();
    let mut result = NewtonKrylovResult::new(3, &opts).unwrap();
    system.residual_calls.set(0);
    assert!(
        NewtonKrylovSolver::new((opts).clone())
            .and_then(|root_plan| root_plan.solve_into(
                &problem,
                &[0.3; 2],
                &mut result,
                &mut w,
                scioxide_core::ExecutionContext::sequential_native()
            ))
            .is_err()
    );
    assert!(
        NewtonKrylovSolver::new(opts)
            .and_then(|root_plan| root_plan.solve_into(
                &problem,
                &[f64::NAN; 3],
                &mut result,
                &mut w,
                scioxide_core::ExecutionContext::sequential_native()
            ))
            .is_err()
    );
    assert_eq!(system.residual_calls.get(), 0);
    NewtonKrylovSolver::new(options().with_history(false))
        .unwrap()
        .solve_into(
            &problem,
            &[0.3; 3],
            &mut result,
            &mut w,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert!(result.is_success());
    assert!(result.history().is_empty());
}
#[test]
fn variable_preconditioning_is_injected_and_counted() {
    struct Variable(Cell<usize>);
    impl scioxide_core::Preconditioner<f64> for Variable {
        fn dimension(&self) -> usize {
            3
        }
        fn precondition_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn precondition_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.0.set(self.0.get() + 1);
            let scale = if self.0.get().is_multiple_of(2) {
                0.5
            } else {
                0.75
            };
            y.iter_mut().zip(x).for_each(|(y, x)| *y = scale * x);
            Ok(())
        }
    }
    let system = System::new(3, 1.0f64);
    let preconditioner = Variable(Cell::new(0));
    let problem = NewtonKrylovProblem::with_jacobian(&system).with_preconditioner(&preconditioner);
    let result = NewtonKrylovSolver::new(options())
        .and_then(|root_plan| root_plan.solve(&problem, &[0.3; 3]))
        .unwrap();
    verify(&system, &result);
    assert!(preconditioner.0.get() > 0);
    assert_eq!(
        preconditioner.0.get(),
        result
            .accounting()
            .total()
            .unwrap()
            .provider(EvaluationRole::Preconditioner)
    );
}
#[test]
fn nested_residual_work_uses_one_context_including_difference_probes() {
    struct Nested(System<f64>, Cell<usize>);
    impl ResidualOperator<f64> for Nested {
        fn input_dimension(&self) -> usize {
            3
        }
        fn output_dimension(&self) -> usize {
            3
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            _: &[f64],
            _: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            Err(SciOxideError::invalid_input(
                scioxide_core::argument!("context"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "context required",
            ))
        }
        fn residual_into_scoped(
            &self,
            x: &[f64],
            y: &mut [f64],
            scratch: &mut [f64],
            context: &mut scioxide_core::SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            assert!(
                context
                    .record_provider(EvaluationRole::Objective)?
                    .is_granted()
            );
            self.1.set(self.1.get() + 1);
            self.0.residual_into(x, y, scratch)
        }
    }
    let provider = Nested(System::new(3, 1.), Cell::new(0));
    let problem = NewtonKrylovProblem::numerical(&provider);
    let result = NewtonKrylovSolver::new(options())
        .and_then(|root_plan| root_plan.solve(&problem, &[0.4; 3]))
        .unwrap();
    verify(&provider.0, &result);
    assert_eq!(
        result
            .accounting()
            .total()
            .unwrap()
            .provider(EvaluationRole::Objective),
        provider.1.get()
    );
    let total = result.accounting().total().unwrap();
    assert_eq!(provider.1.get(), total.provider(EvaluationRole::Residual));
    assert!(
        total.provider(EvaluationRole::Residual)
            <= 1 + total.proposals() + total.provider(EvaluationRole::LinearOperator)
    );
}
#[test]
fn shared_iteration_and_proposal_limits_keep_the_checked_point() {
    let system = System::new(3, 1.0f64);
    let opts = options();
    for numerical in [false, true] {
        let problem = if numerical {
            NewtonKrylovProblem::numerical(&system)
        } else {
            NewtonKrylovProblem::with_jacobian(&system)
        };
        for kind in [BudgetKind::Iteration, BudgetKind::Proposal] {
            for limit in 1..12 {
                let mut w = NewtonKrylovSolver::new((opts).clone())
                    .and_then(|root_plan| root_plan.workspace(&problem))
                    .unwrap();
                let mut output = NewtonKrylovResult::new(3, &opts).unwrap();
                let mut state = SolveState::new();
                let mut budgets =
                    SolveBudgets::unbounded().with_limit(kind, Budget::try_from(limit).unwrap());
                let identity = scioxide_core::SolveIdentity::new(
                    ProblemToken::new(5),
                    scioxide_core::algorithm_id!("test", "nonlinear-parent", 1),
                    0,
                    0,
                );
                let mut context = state
                    .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
                    .unwrap();
                NewtonKrylovSolver::new((opts).clone())
                    .and_then(|root_plan| {
                        root_plan.solve_into(
                            &problem,
                            &[0.05; 3],
                            &mut output,
                            &mut w,
                            &mut context,
                        )
                    })
                    .unwrap();
                assert!(context.budget_report(kind).consumed() <= limit);
                let mut f = [0.; 3];
                system
                    .residual_into(output.point(), &mut f, &mut [])
                    .unwrap();
                assert_eq!(output.residual(), f);
                assert!(
                    output.is_success()
                        || matches!(
                            output.termination(),
                            RootTermination::IterationLimit
                                | RootTermination::WorkLimit(scioxide_core::BudgetKind::Proposal)
                        )
                );
            }
        }
    }
}

#[test]
fn sparse_dense_and_borrowed_operators_match_dense_newton() {
    use scioxide_core::LinearOperator;
    use scioxide_linalg::Matrix;
    use scioxide_sparse::ExplicitZeroPolicy;
    struct Assembled<'a> {
        system: &'a System<f64>,
        coupling: &'a dyn LinearOperator<f64>,
    }
    impl ResidualOperator<f64> for Assembled<'_> {
        fn input_dimension(&self) -> usize {
            3
        }
        fn output_dimension(&self) -> usize {
            3
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            s: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.system.residual_into(x, y, s)
        }
    }
    impl JacobianOperator<f64> for Assembled<'_> {
        fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
            self.coupling.apply_workspace_len()
        }
        fn jvp_into(
            &self,
            x: &[f64],
            v: &[f64],
            y: &mut [f64],
            s: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.coupling.apply_into(v, y, s)?;
            y.iter_mut()
                .zip(x)
                .zip(v)
                .for_each(|((y, x), v)| *y += 2. * x * v);
            Ok(())
        }
    }
    let matrix = Matrix::from_vec(3, 3, vec![0., 0.05, 0., 0.05, 0., 0.05, 0., 0.05, 0.]).unwrap();
    let csr = matrix.to_sparse_csr(ExplicitZeroPolicy::Drop).unwrap();
    let csc = csr.to_csc().unwrap();
    let view = matrix.view();
    let sparse_view = csr.view();
    let system = System::new(3, 1.0f64);
    let opts = options();
    let mut solutions = Vec::new();
    for coupling in [
        &matrix.view().try_finite().unwrap() as &dyn LinearOperator<f64>,
        &view.try_finite().unwrap(),
        &csr,
        &csc,
        &sparse_view,
    ] {
        let provider = Assembled {
            system: &system,
            coupling,
        };
        let problem = NewtonKrylovProblem::with_jacobian(&provider);
        let result = NewtonKrylovSolver::new((opts).clone())
            .and_then(|root_plan| root_plan.solve(&problem, &[0.3; 3]))
            .unwrap();
        verify(&system, &result);
        solutions.push(result.point().to_vec());
    }
    for x in &solutions {
        assert_eq!(x, &solutions[0]);
    }
    use scioxide_rootfind::{BacktrackingLineSearch, NewtonSystem, ResidualFn, SystemRootSolver};
    let dense = ResidualFn::new(3, |x: &[f64], y: &mut [f64]| {
        system.residual_into(x, y, &mut [])
    })
    .with_jacobian(|x: &[f64], j: &mut [f64]| {
        j.copy_from_slice(matrix.as_slice());
        for (i, &x) in x.iter().enumerate() {
            j[i * 3 + i] = 2. * x;
        }
        Ok(())
    });
    let dense_result = NewtonSystem::new(
        (scioxide_rootfind::NewtonSystemOptions::default()
            .with_residual_tolerance(ResidualTolerance::new(1e-13, 0.0).unwrap())
            .with_step_tolerance(StepTolerance::new(1e-15, 0.0).unwrap())
            .with_line_search(BacktrackingLineSearch::default()))
        .with_derivative_strategy(scioxide_differentiate::DerivativeStrategy::Analytic),
    )
    .unwrap()
    .solve(&dense, &[0.3; 3])
    .unwrap();
    assert!(dense_result.is_success());
    for (&a, &b) in dense_result.point().iter().zip(&solutions[0]) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn strided_input_and_fresh_child_epochs() {
    use scioxide_core::{NdView, StridedLayout};
    let system = System::new(3, 1.0_f64);
    let problem = NewtonKrylovProblem::with_jacobian(&system);
    let opts = options();
    let plan = NewtonKrylovSolver::new(opts.clone()).unwrap();
    let mut w = plan.workspace(&problem).unwrap();
    let mut result = NewtonKrylovResult::new(3, &opts).unwrap();
    let layout = StridedLayout::new(&[3], &[2]).unwrap();
    let data = [0.3, 99.0, 0.3, 99.0, 0.3];
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(7),
        scioxide_core::algorithm_id!("test", "parent", 1),
        0,
        0,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    for epoch in 1..=2 {
        plan.solve_into(
            &problem,
            NdView::new(&data, layout).unwrap(),
            &mut result,
            &mut w,
            &mut context,
        )
        .unwrap();
        verify(&system, &result);
        assert_eq!(w.last_attempt().unwrap().epoch().unwrap().get(), epoch);
        assert_eq!(context.counts().own().total_providers().unwrap(), 0);
    }
    assert_eq!(data[1], 99.0);
    assert_eq!(data[3], 99.0);
}

#[cfg(feature = "std")]
#[test]
fn injected_executor_matches_sequential_and_cancellation_keeps_initial_state() {
    use scioxide_core::{CancellationToken, ThreadExecutor};
    let system = System::new(9, 1.0f64);
    let problem = NewtonKrylovProblem::numerical(&system);
    let opts = options();
    let sequential = NewtonKrylovSolver::new((opts).clone())
        .and_then(|root_plan| root_plan.solve(&problem, &[0.3; 9]))
        .unwrap();
    let pool = ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    let mut w = NewtonKrylovSolver::new((opts).clone())
        .and_then(|root_plan| root_plan.workspace(&problem))
        .unwrap();
    let mut result = NewtonKrylovResult::new(9, &opts).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(3),
        scioxide_core::algorithm_id!("test", "nonlinear-parent", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::new(&pool), identity, &mut budgets)
        .unwrap();
    NewtonKrylovSolver::new((opts).clone())
        .and_then(|root_plan| {
            root_plan.solve_into(&problem, &[0.3; 9], &mut result, &mut w, &mut context)
        })
        .unwrap();
    verify(&system, &result);
    assert_eq!(result.point(), sequential.point());
    assert_eq!(result.accounting(), sequential.accounting());
    struct Cancelling<'a>(&'a CancellationToken, System<f64>);
    impl ResidualOperator<f64> for Cancelling<'_> {
        fn input_dimension(&self) -> usize {
            3
        }
        fn output_dimension(&self) -> usize {
            3
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            s: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.1.residual_into(x, y, s)?;
            self.0.cancel();
            Ok(())
        }
    }
    let cancel = CancellationToken::new();
    let provider = Cancelling(&cancel, System::new(3, 1.));
    let problem = NewtonKrylovProblem::numerical(&provider);
    let mut w = NewtonKrylovSolver::new((opts).clone())
        .and_then(|root_plan| root_plan.workspace(&problem))
        .unwrap();
    let mut result = NewtonKrylovResult::new(3, &opts).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let identity = scioxide_core::SolveIdentity::new(
        ProblemToken::new(3),
        scioxide_core::algorithm_id!("test", "nonlinear-parent", 1),
        0,
        0,
    );
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential().with_cancellation(&cancel),
            identity,
            &mut budgets,
        )
        .unwrap();
    NewtonKrylovSolver::new(opts)
        .and_then(|root_plan| {
            root_plan.solve_into(&problem, &[0.3; 3], &mut result, &mut w, &mut context)
        })
        .unwrap();
    assert_eq!(result.termination(), RootTermination::Cancelled);
    assert_eq!(result.point(), [0.3; 3]);
}

#[test]
fn exponent_scaled_systems_preserve_residual_stopping() {
    struct Scaled(System<f64>, f64);
    impl ResidualOperator<f64> for Scaled {
        fn input_dimension(&self) -> usize {
            3
        }
        fn output_dimension(&self) -> usize {
            3
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            s: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.0.residual_into(x, y, s)?;
            y.iter_mut().for_each(|v| *v *= self.1);
            Ok(())
        }
    }
    impl JacobianOperator<f64> for Scaled {
        fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn jvp_into(
            &self,
            x: &[f64],
            v: &[f64],
            y: &mut [f64],
            s: &mut [f64],
        ) -> Result<(), SciOxideError> {
            self.0.jvp_into(x, v, y, s)?;
            y.iter_mut().for_each(|v| *v *= self.1);
            Ok(())
        }
    }
    for scale in [1e-250, 1., 1e250] {
        let system = Scaled(System::new(3, 1.), scale);
        let problem = NewtonKrylovProblem::with_jacobian(&system);
        let opts = options()
            .with_residual_scale(scale)
            .with_residual_tolerance(ResidualTolerance::new(0., 1e-12).unwrap());
        let result = NewtonKrylovSolver::new((opts).clone())
            .and_then(|root_plan| root_plan.solve(&problem, &[0.2; 3]))
            .unwrap();
        assert!(result.is_success(), "{result:?}");
        for x in result.point() {
            assert!((x - 1.).abs() < 1e-12);
        }
        assert!(result.residual_report().unwrap().scaled_norm() < 1e-12);
    }
}

#[test]
fn adaptive_forcing_matches_independent_choice_one_recurrence() {
    let system = System::new(9, 1.0f64);
    let result = NewtonKrylovSolver::new(options())
        .and_then(|root_plan| {
            root_plan.solve(&NewtonKrylovProblem::with_jacobian(&system), &[0.3; 9])
        })
        .unwrap();
    assert!(result.is_success());
    assert_eq!(result.history()[0].forcing(), 0.5);
    let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut checked = 0;
    for pair in result.history().windows(2) {
        let (previous, current) = (pair[0], pair[1]);
        // A full step's linear-model residual is the verified inner residual.
        if previous.step_length() != 1.0 {
            continue;
        }
        let model = previous.linear_residual().unwrap();
        let mut expected = (current.residual_norm() - model).abs() / previous.residual_norm();
        let safeguard = previous.forcing().powf(golden);
        if safeguard > 0.1 {
            expected = expected.max(safeguard);
        }
        expected = expected.clamp(f64::EPSILON, 0.9);
        assert!((current.forcing() - expected).abs() <= 1e-14 * expected.max(1e-2));
        checked += 1;
    }
    assert!(checked >= 2, "{:?}", result.history());
    assert!(result.history().last().unwrap().forcing() < 0.1);
}

#[test]
fn singular_jacobian_returns_typed_inner_breakdown() {
    struct Constant;
    impl ResidualOperator<f64> for Constant {
        fn input_dimension(&self) -> usize {
            1
        }
        fn output_dimension(&self) -> usize {
            1
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            _: &[f64],
            y: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            y[0] = 1.;
            Ok(())
        }
    }
    impl JacobianOperator<f64> for Constant {
        fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn jvp_into(
            &self,
            _: &[f64],
            _: &[f64],
            y: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            y[0] = 0.;
            Ok(())
        }
    }
    let result = NewtonKrylovSolver::new(options())
        .and_then(|root_plan| {
            root_plan.solve(&NewtonKrylovProblem::with_jacobian(&Constant), &[7.])
        })
        .unwrap();
    assert!(matches!(
        result.termination(),
        RootTermination::LinearFailure(scioxide_linalg::LinearSolveTermination::Breakdown(_))
    ));
    assert_eq!(result.point(), [7.]);
    assert_eq!(result.residual(), [1.]);
}

#[test]
fn deadline_before_and_after_initial_residual_has_explicit_trust() {
    use core::{
        sync::atomic::{AtomicBool, Ordering},
        time::Duration,
    };
    use scioxide_core::{CapabilityError, Clock, Deadline, ExecutionStopReason, MonotonicInstant};
    struct Clocked {
        expired: AtomicBool,
    }
    impl Clock for Clocked {
        fn now(&self) -> Result<MonotonicInstant, CapabilityError> {
            Ok(MonotonicInstant::from_duration_since_epoch(
                Duration::from_secs(u64::from(self.expired.load(Ordering::Relaxed))),
            ))
        }
    }
    impl ResidualOperator<f64> for Clocked {
        fn input_dimension(&self) -> usize {
            1
        }
        fn output_dimension(&self) -> usize {
            1
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            x: &[f64],
            y: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            y[0] = x[0] * x[0] - 1.;
            self.expired.store(true, Ordering::Relaxed);
            Ok(())
        }
    }
    for initially_expired in [false, true] {
        let provider = Clocked {
            expired: AtomicBool::new(initially_expired),
        };
        let provider_binding = scioxide_core::ClockBinding::new(&provider);
        let problem = NewtonKrylovProblem::numerical(&provider);
        let opts = options();
        let mut w = NewtonKrylovSolver::new((opts).clone())
            .and_then(|root_plan| root_plan.workspace(&problem))
            .unwrap();
        let mut result = NewtonKrylovResult::new(1, &opts).unwrap();
        let deadline = Deadline::at(
            &provider_binding,
            MonotonicInstant::from_duration_since_epoch(Duration::from_secs(1)),
        );
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let identity = scioxide_core::SolveIdentity::new(
            ProblemToken::new(5),
            scioxide_core::algorithm_id!("test", "nonlinear-parent", 1),
            0,
            0,
        );
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential()
                    .with_deadline(deadline)
                    .unwrap(),
                identity,
                &mut budgets,
            )
            .unwrap();
        let outcome = NewtonKrylovSolver::new((opts).clone()).and_then(|root_plan| {
            root_plan.solve_into(&problem, &[0.3], &mut result, &mut w, &mut context)
        });
        if initially_expired {
            assert!(matches!(
                outcome.unwrap_err().view(),
                scioxide_core::SciOxideErrorView::ExecutionStopped {
                    reason: ExecutionStopReason::DeadlineReached,
                    ..
                }
            ));
            assert!(result.residual_report().is_none());
        } else {
            outcome.unwrap();
            assert_eq!(result.point(), [0.3]);
            assert_eq!(result.termination(), RootTermination::TimeLimit);
        }
    }
}

/// `System` with declared scratch lengths; a callback given any other length fails.
struct Scratched<'a> {
    inner: &'a System<f64>,
    residual: usize,
    jvp: usize,
}
fn exact_scratch(scratch: &[f64], declared: usize) -> Result<(), SciOxideError> {
    if scratch.len() == declared {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(
            [declared],
            [scratch.len()],
        ))
    }
}
impl ResidualOperator<f64> for Scratched<'_> {
    fn input_dimension(&self) -> usize {
        self.inner.n
    }
    fn output_dimension(&self) -> usize {
        self.inner.n
    }
    fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.residual)
    }
    fn residual_into(&self, x: &[f64], y: &mut [f64], s: &mut [f64]) -> Result<(), SciOxideError> {
        exact_scratch(s, self.residual)?;
        self.inner.residual_into(x, y, &mut [])
    }
}
impl JacobianOperator<f64> for Scratched<'_> {
    fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(self.jvp)
    }
    fn jvp_into(
        &self,
        x: &[f64],
        v: &[f64],
        y: &mut [f64],
        s: &mut [f64],
    ) -> Result<(), SciOxideError> {
        exact_scratch(s, self.jvp)?;
        self.inner.jvp_into(x, v, y, &mut [])
    }
}

/// Dense direct solve of the operator's columns; its applications use exact scratch.
struct Columns;
impl scioxide_rootfind::NonlinearLinearSolver<f64> for Columns {
    fn solve_into(
        &self,
        a: &scioxide_rootfind::NonlinearLinearization<'_, f64>,
        b: &scioxide_linalg::Vector<f64>,
        x: &mut scioxide_linalg::Vector<f64>,
        _: ResidualTolerance<f64>,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<scioxide_linalg::IterativeSolveReport<f64>, SciOxideError> {
        let scope = scioxide_core::SolveScope::from(context);
        let a = a.operator();
        let n = a.cols();
        let identity = scioxide_core::SolveIdentity::new(
            ProblemToken::new(0),
            scioxide_core::algorithm_id!("test", "columns", 1),
            n as u128,
            0,
        );
        scope.run(
            &mut SolveState::new(),
            identity,
            SolveBudgets::unbounded(),
            |c| {
                let mut m = vec![vec![0.0; n + 1]; n];
                let mut scratch = vec![0.0; a.apply_workspace_len()?];
                let mut column = vec![0.0; n];
                for j in 0..n {
                    let mut unit = vec![0.0; n];
                    unit[j] = 1.0;
                    a.apply_into_scoped(
                        &unit,
                        &mut column,
                        &mut scratch,
                        &mut scioxide_core::OperatorContext::new(c),
                    )
                    .and_then(scioxide_core::Invocation::require)?;
                    for (row, &value) in m.iter_mut().zip(&column) {
                        row[j] = value;
                    }
                }
                for (row, &value) in m.iter_mut().zip(b.as_slice()) {
                    row[n] = value;
                }
                // Diagonally dominant: elimination without pivoting.
                for k in 0..n {
                    let (pivots, rows) = m.split_at_mut(k + 1);
                    let pivot = &pivots[k];
                    for row in rows {
                        let factor = row[k] / pivot[k];
                        for (target, &value) in row[k..].iter_mut().zip(&pivot[k..]) {
                            *target -= factor * value;
                        }
                    }
                }
                for k in (0..n).rev() {
                    let tail: f64 = (k + 1..n).map(|j| m[k][j] * x.as_slice()[j]).sum();
                    x.as_mut_slice()[k] = (m[k][n] - tail) / m[k][k];
                }
                Ok(scioxide_linalg::IterativeSolveReport::new(
                    scioxide_linalg::LinearSolveTermination::Converged,
                    0,
                    None,
                    scioxide_core::PartialResultTrust::DiagnosticOnly,
                ))
            },
        )
    }
}

#[test]
fn reused_workspaces_pass_each_problem_its_exact_scratch() {
    let system = System::new(3, 1.0f64);
    let large = Scratched {
        inner: &system,
        residual: 4,
        jvp: 3,
    };
    let small = Scratched {
        inner: &system,
        residual: 1,
        jvp: 0,
    };
    let plan = NewtonKrylovSolver::new(options()).unwrap();
    // Forward-difference products pass the residual scratch to the callback.
    fn numerical<'a>(p: &'a Scratched<'_>) -> NewtonKrylovProblem<'a, f64> {
        NewtonKrylovProblem::numerical(p)
    }
    // An injected solver owns its storage, so only the product scratch differs.
    fn analytic<'a>(p: &'a Scratched<'_>) -> NewtonKrylovProblem<'a, f64> {
        NewtonKrylovProblem::with_jacobian(p).with_linear_solver(&Columns)
    }
    for problem in [numerical, analytic] {
        let mut workspace = plan.workspace(&problem(&large)).unwrap();
        let mut result = plan.solve(&problem(&large), &[0.4; 3]).unwrap();
        for scratched in [&large, &small, &large] {
            plan.solve_into(
                &problem(scratched),
                &[0.4; 3],
                &mut result,
                &mut workspace,
                ExecutionContext::sequential_native(),
            )
            .unwrap();
            assert!(result.is_success(), "{result:?}");
        }
    }
}

#[test]
fn inexact_steps_are_taken_on_the_last_outer_iteration() {
    // One GMRES iteration per linear solve meets the first two forcing requests
    // and then misses them while staying within the largest forcing.
    let f = scioxide_rootfind::ResidualFn::new(2, |x: &[f64], y: &mut [f64]| {
        y[0] = x[0] - 1.0;
        y[1] = 100.0 * x[1] - 3.0;
        Ok(())
    });
    let mut previous = f64::INFINITY;
    for iterations in 1..=6 {
        let options = NewtonKrylovOptions::default()
            .with_krylov(Budget::ONE, Budget::ONE)
            .with_max_iterations(Budget::try_from(iterations).unwrap())
            .with_history(true);
        let result = NewtonKrylovSolver::new(options)
            .unwrap()
            .solve(&NewtonKrylovProblem::numerical(&f), &[0.0, 0.0])
            .unwrap();
        assert_eq!(result.termination(), RootTermination::IterationLimit);
        assert_eq!(result.iterations(), iterations);
        let last = result.history().last().unwrap();
        assert!(last.step_length() > 0.0, "{iterations}: {last:?}");
        let norm = result.residual_norm().unwrap();
        assert!(norm < previous, "{iterations}: {norm} after {previous}");
        previous = norm;
    }
}

struct ComponentUnits {
    magnitude: f64,
}
impl scioxide_rootfind::NonlinearScaling<f64> for ComponentUnits {
    fn scales_into(
        &self,
        _: &[f64],
        residual: &mut [f64],
        variables: &mut [f64],
        _: &mut [f64],
        _: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<(), SciOxideError> {
        residual.copy_from_slice(&[self.magnitude, 1. / self.magnitude]);
        variables.fill(1.);
        Ok(())
    }
}
#[test]
fn borrowed_scaling_balances_equations_and_checks_divisors() {
    let magnitude = 1e9;
    let function = scioxide_rootfind::ResidualFn::new(2, |x: &[f64], out: &mut [f64]| {
        out[0] = magnitude * (x[0] * x[0] - 2.);
        out[1] = (x[1] * x[1] - 3.) / magnitude;
        Ok(())
    });
    let scaling = ComponentUnits { magnitude };
    let problem = NewtonKrylovProblem::numerical(&function).with_scaling(&scaling);
    let solver = NewtonKrylovSolver::new(
        NewtonKrylovOptions::default()
            .with_residual_tolerance(ResidualTolerance::new(1e-10, 1e-10).unwrap()),
    )
    .unwrap();
    let result = solver.solve(&problem, &[1., 1.]).unwrap();
    assert!(result.is_success(), "{result:?}");
    assert!((result.point()[0] - 2_f64.sqrt()).abs() < 1e-9);
    assert!((result.point()[1] - 3_f64.sqrt()).abs() < 1e-9);
    for magnitude in [0., -1., f64::NAN, f64::INFINITY] {
        let scaling = ComponentUnits { magnitude };
        let problem = problem.with_scaling(&scaling);
        assert!(solver.solve(&problem, &[1., 1.]).is_err());
    }
}

#[test]
fn constant_component_units_preserve_nonmonotone_globalization() {
    struct Rosenbrock;
    impl ResidualOperator<f64> for Rosenbrock {
        fn input_dimension(&self) -> usize {
            2
        }
        fn output_dimension(&self) -> usize {
            2
        }
        fn residual_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn residual_into(
            &self,
            x: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out.copy_from_slice(&[10. * (x[1] - x[0] * x[0]), 1. - x[0]]);
            Ok(())
        }
    }
    impl JacobianOperator<f64> for Rosenbrock {
        fn jvp_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn jvp_into(
            &self,
            x: &[f64],
            v: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out.copy_from_slice(&[-20. * x[0] * v[0] + 10. * v[1], -v[0]]);
            Ok(())
        }
    }
    struct Units;
    impl scioxide_rootfind::NonlinearScaling<f64> for Units {
        fn scales_into(
            &self,
            _: &[f64],
            rows: &mut [f64],
            variables: &mut [f64],
            _: &mut [f64],
            _: &mut scioxide_core::SolveContext<'_, '_>,
        ) -> Result<(), SciOxideError> {
            rows.fill(1.);
            variables.fill(1.);
            Ok(())
        }
    }
    let problem = NewtonKrylovProblem::with_jacobian(&Rosenbrock).with_linear_solver(&Columns);
    let opts = options()
        .with_nonmonotone(0.9)
        .with_max_iterations(Budget::try_from(500).unwrap())
        .with_step_tolerance(StepTolerance::new(1e-100, 1e-100).unwrap());
    let solver = NewtonKrylovSolver::new(opts.clone()).unwrap();
    let initial = [-1.2, 1.];
    let reference = solver.solve(&problem, &initial).unwrap();
    let scaled = solver
        .solve(&problem.with_scaling(&Units), &initial)
        .unwrap();
    let monotone = NewtonKrylovSolver::new(opts.with_nonmonotone(0.))
        .unwrap()
        .solve(&problem, &initial)
        .unwrap();
    assert!(reference.is_success() && scaled.is_success() && monotone.is_success());
    let steps = |result: &NewtonKrylovResult<f64>| {
        result
            .history()
            .iter()
            .map(|o| o.step_length())
            .collect::<Vec<_>>()
    };
    assert_ne!(steps(&reference), steps(&monotone));
    assert_eq!(steps(&reference), steps(&scaled));
    assert_eq!(reference.point(), scaled.point());
}
