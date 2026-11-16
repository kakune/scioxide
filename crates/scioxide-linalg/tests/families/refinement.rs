#![allow(clippy::unwrap_used, clippy::expect_used)]

use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, Capacity, Complex, EvaluationRole, ExecutionContext,
    LinearOperator, NumericScalar, ProblemToken, RealScalar, ResidualTolerance, SciOxideError,
    SciOxideErrorKind, SolveBudgets, SolveContext, SolveIdentity, SolveState,
};
use scioxide_linalg::*;
#[path = "../refinement/stops.rs"]
mod stops;

fn options<R: RealScalar>() -> RefinementOptions<R> {
    RefinementOptions::new(
        ResidualTolerance::new(R::ZERO, R::EPSILON * R::try_from_usize_exact(16).unwrap()).unwrap(),
    )
    .with_criterion(RefinementCriterion::Componentwise)
    .with_execution_options(
        LinalgExecutionOptions::new().with_iteration_budget(Budget::ONE.saturating_add(11)),
    )
}
fn budgeted<R: RealScalar>(options: RefinementOptions<R>, budget: Budget) -> RefinementOptions<R> {
    options.with_execution_options(options.execution_options().with_iteration_budget(budget))
}
fn limited<R: RealScalar>(options: RefinementOptions<R>, bytes: usize) -> RefinementOptions<R> {
    options.with_execution_options(options.execution_options().with_byte_limit(bytes))
}
fn context<'s, 'e>(
    state: &'s mut SolveState,
    budgets: &'s mut SolveBudgets,
    execution: ExecutionContext<'e>,
) -> SolveContext<'s, 'e> {
    state
        .begin_solve(
            execution,
            SolveIdentity::new(
                ProblemToken::new(473),
                scioxide_core::algorithm_id!("test", "refinement", 1),
                0,
                0,
            ),
            budgets,
        )
        .unwrap()
}
fn scalar<S: NumericScalar>(re: f64, im: f64) -> S {
    S::from_components(
        S::Real::try_from_f64_exact(re).unwrap(),
        S::Real::try_from_f64_exact(im).unwrap(),
    )
}

struct Diagonal<S> {
    values: Vec<S>,
    zero: bool,
}
impl<S: NumericScalar> LinearOperator<S> for Diagonal<S> {
    fn rows(&self) -> usize {
        self.values.len()
    }
    fn cols(&self) -> usize {
        self.values.len()
    }
    fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
        Ok(0)
    }
    fn apply_into(&self, x: &[S], y: &mut [S], _: &mut [S]) -> Result<(), SciOxideError> {
        for ((out, x), d) in y.iter_mut().zip(x).zip(&self.values) {
            *out = if self.zero { S::ZERO } else { *x / *d };
        }
        Ok(())
    }
}
#[test]
fn stagnation_and_fallback_are_residual_driven() {
    let a = Matrix::<f64>::identity(2).unwrap();
    let low = Diagonal {
        values: vec![1f32; 2],
        zero: true,
    };
    let high = Diagonal {
        values: vec![1.; 2],
        zero: false,
    };
    let rhs = Vector::from_vec(vec![1., 2.]);
    let mut x = Vector::zeros(2).unwrap();
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &low, None, None).unwrap();
    let plan = IterativeRefinement::new(options()).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    let report = plan
        .refine_into(
            &problem,
            rhs.view(),
            x.view_mut(),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap()[0];
    assert_eq!(report.termination(), RefinementTermination::Stagnation);
    assert_eq!(x.as_slice(), [0., 0.]);
    let mut opts = options();
    opts = opts.with_fallback(RefinementFallback::OnFailure);
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &low, None, Some(&high)).unwrap();
    let plan = IterativeRefinement::new(opts).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    let report = plan
        .refine_into(
            &problem,
            rhs.view(),
            x.view_mut(),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap()[0];
    assert!(report.is_success());
    assert_eq!(
        report.fallback_reason(),
        Some(RefinementFallbackReason::Stagnation)
    );
    assert_eq!(report.iterations(), 2);
    assert_eq!(x.as_slice(), rhs.as_slice());
}

#[test]
fn normalization_handles_binary_boundaries_and_complex_components() {
    let a = Matrix::<Complex<f64>>::identity(2).unwrap();
    let inverse = Diagonal {
        values: vec![Complex::new(1f32, 0.); 2],
        zero: false,
    };
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let plan = IterativeRefinement::new(options()).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    for scale in [
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        2f64.powi(-167),
        1.,
        2f64.powi(133),
        2f64.powi(1022),
    ] {
        let b = [Complex::new(scale, scale), Complex::new(scale, 0.)];
        let mut x = [Complex::new(0., 0.); 2];
        let report = plan
            .refine_into(
                &problem,
                VectorView::from_slice(&b),
                VectorViewMut::from_slice(&mut x),
                &mut w,
                ExecutionContext::sequential(),
            )
            .unwrap()[0];
        assert!(report.is_success(), "scale={scale:e}: {report:?}");
        assert_eq!(x, b);
    }
    // The imaginary component may underflow independently of the real component.
    let b = [Complex::new(1., 1e-60), Complex::new(1., 0.)];
    let mut x = [Complex::new(0., 0.); 2];
    let report = plan
        .refine_into(
            &problem,
            VectorView::from_slice(&b),
            VectorViewMut::from_slice(&mut x),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap()[0];
    assert!(report.is_success(), "{report:?}");
    assert_eq!(report.fallback_reason(), None);
}

#[test]
fn extreme_residual_products_and_denominators_do_not_overflow() {
    // Exact binary fixture: each row has cancelling products outside f64 range.
    let big = 2f64.powi(1022);
    let a = Matrix::from_vec(2, 2, vec![big, big, big, -big]).unwrap();
    let inverse = Diagonal {
        values: vec![1.; 2],
        zero: true,
    };
    let mut opts = options();
    opts = budgeted(opts, Budget::ONE);
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let plan = IterativeRefinement::new(opts).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    let rhs = Vector::from_vec(vec![0., big]);
    let mut x = Vector::from_vec(vec![4., -4.]);
    // The second residual itself is unrepresentable: this is a structured error.
    let before = x.clone();
    assert!(
        plan.refine_into(
            &problem,
            rhs.view(),
            x.view_mut(),
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(x, before);
    let a = Matrix::from_vec(2, 2, vec![big, big, big, big]).unwrap();
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let plan = IterativeRefinement::new(opts).unwrap();
    let rhs = Vector::from_vec(vec![1., 1.]);
    let report = plan
        .refine_into(
            &problem,
            rhs.view(),
            x.view_mut(),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap()[0];
    assert!(report.is_success());
    assert!(report.final_error().unwrap().normwise() > 0.);
    assert!(report.final_error().unwrap().normwise() < 1e-307);
}
#[test]
fn equilibration_preserves_sparse_pattern_and_scales_before_narrowing() {
    let a = Matrix::from_vec(2, 2, vec![2f64.powi(500), 0., 0., 2f64.powi(-500)]).unwrap();
    let scaling = EquilibrationMetadata::prepare(RefinementSystem::Dense(a.view())).unwrap();
    let mut low = Matrix::<f32>::zeros(2, 2).unwrap();
    scaling
        .scale_matrix_into(a.view(), &mut low.view_mut())
        .unwrap();
    assert_eq!(low.as_slice(), [1., 0., 0., 1.]);
    let inverse = LuDecomposition::factor(&low).unwrap();
    let inverse = inverse.inverse_operator();
    let problem = RefinementProblem::new(
        RefinementSystem::Dense(a.view()),
        &inverse,
        Some(&scaling),
        None,
    )
    .unwrap();
    let plan = IterativeRefinement::new(options()).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    let rhs = Vector::from_vec(vec![2f64.powi(500), 2f64.powi(-500)]);
    let mut x = Vector::zeros(2).unwrap();
    let report = plan
        .refine_into(
            &problem,
            rhs.view(),
            x.view_mut(),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap()[0];
    assert!(report.is_success());
    assert_eq!(x.as_slice(), [1., 1.]);
    let csr = scioxide_sparse::CsrMatrix::from_raw(
        2,
        2,
        vec![0, 2, 3],
        vec![0, 1, 1],
        vec![2f64.powi(500), 0., 2f64.powi(-500)],
    )
    .unwrap();
    let mut values = [7f32; 3];
    scaling.scale_sparse_values_into(&csr, &mut values).unwrap();
    assert_eq!(values, [1., 0., 1.]);
    let sparse_scaling = EquilibrationMetadata::prepare(RefinementSystem::Sparse(&csr)).unwrap();
    assert_eq!(scaling, sparse_scaling);
    let problem = RefinementProblem::new(
        RefinementSystem::Sparse(&csr),
        &inverse,
        Some(&scaling),
        None,
    )
    .unwrap();
    let plan = IterativeRefinement::new(options()).unwrap();
    x.as_mut_slice().fill(0.);
    assert!(
        plan.refine_into(
            &problem,
            rhs.view(),
            x.view_mut(),
            &mut w,
            ExecutionContext::sequential()
        )
        .unwrap()[0]
            .is_success()
    );
}
fn sparse_family<L: NumericScalar, H: NumericScalar>() {
    use scioxide_sparse::{CsrMatrix, SparseLu, SparseLuLimits, SparseLuSymbolic};
    let low = CsrMatrix::from_raw(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            scalar::<L>(4., 0.),
            scalar(1., 0.125),
            scalar(2., -0.125),
            scalar(3., 0.),
        ],
    )
    .unwrap();
    let high = CsrMatrix::from_raw(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![
            scalar::<H>(4., 0.),
            scalar(1., 0.125),
            scalar(2., -0.125),
            scalar(3., 0.),
        ],
    )
    .unwrap();
    let symbolic = SparseLuSymbolic::analyze(
        low.structural_pattern(),
        scioxide_sparse::SparseOrdering::default(),
        SparseLuLimits::new(8, 100_000),
    )
    .unwrap();
    let mut factor = SparseLu::factor(symbolic.clone(), low.view()).unwrap();
    let mut fw = (symbolic).workspace(()).unwrap();
    factor
        .refactor(
            low.view(),
            scioxide_sparse::SparseSolveOptions::new(),
            &mut fw,
            ExecutionContext::sequential(),
        )
        .unwrap();
    let rhs = Vector::from_vec(vec![scalar::<H>(5., 0.125), scalar(5., -0.125)]);
    let mut x = Vector::zeros(2).unwrap();
    let mut opts = options();
    let problem =
        RefinementProblem::new(RefinementSystem::Sparse(&high), &factor, None, None).unwrap();
    let plan = IterativeRefinement::new(opts).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    assert!(
        plan.refine_into(
            &problem,
            rhs.view(),
            x.view_mut(),
            &mut w,
            ExecutionContext::sequential()
        )
        .unwrap()[0]
            .is_success()
    );
    opts = opts.with_criterion(RefinementCriterion::Normwise);
    let problem = RefinementProblem::new(
        RefinementSystem::Operator {
            operator: &high,
            norm_bound: H::Real::try_from_usize_exact(8).unwrap(),
        },
        &factor,
        None,
        None,
    )
    .unwrap();
    let plan = IterativeRefinement::new(opts).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    x.as_mut_slice().fill(H::ZERO);
    let report = plan
        .refine_into(
            &problem,
            rhs.view(),
            x.view_mut(),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap()[0];
    assert!(report.is_success());
    assert!(report.final_error().unwrap().uses_operator_bound());
    assert!(report.final_error().unwrap().componentwise().is_none());
    assert!(
        IterativeRefinement::<f64, f64>::new(options())
            .unwrap()
            .workspace_requirements(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
            .is_err()
    );
}
#[test]
fn sparse_factor_and_operator_domains() {
    sparse_family::<f32, f32>();
    sparse_family::<f64, f64>();
    sparse_family::<f32, f64>();
    sparse_family::<Complex<f32>, Complex<f32>>();
    sparse_family::<Complex<f64>, Complex<f64>>();
    sparse_family::<Complex<f32>, Complex<f64>>();
}
#[test]
fn shared_budget_and_workspace_limits() {
    let a = Matrix::<f64>::identity(2).unwrap();
    let inverse = Diagonal {
        values: vec![1.; 2],
        zero: false,
    };
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let plan = IterativeRefinement::<f64, f64>::new(options()).unwrap();
    let bytes = plan
        .workspace_requirements(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap()
        .required_bytes();
    assert_eq!(
        plan.workspace_requirements(
            &scioxide_linalg::RefinementWorkspaceRequest::new(&problem)
                .with_right_hand_sides(usize::MAX)
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::SizeOverflow
    );
    let capped = |limit| IterativeRefinement::new(limited(options(), limit)).unwrap();
    let tight = capped(bytes - 1);
    assert_eq!(
        tight
            .workspace_requirements(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(
        tight
            .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let plan = capped(bytes);
    let requirements = plan
        .workspace_requirements(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    assert_eq!(requirements.byte_limit(), Some(bytes));
    let mut w = RefinementWorkspace::new();
    assert_eq!(w.check(&requirements).unwrap(), Capacity::NeedsGrowth);
    w.try_reserve(&requirements).unwrap();
    assert_eq!(w.check(&requirements).unwrap(), Capacity::Sufficient);
    // Storage retained for two columns exceeds the one-column limit.
    let mut wide = capped(usize::MAX)
        .workspace(
            &scioxide_linalg::RefinementWorkspaceRequest::new(&problem).with_right_hand_sides(2),
        )
        .unwrap();
    assert_eq!(
        wide.check(&requirements).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let mut x = Vector::from_vec(vec![3., 4.]);
    assert_eq!(
        plan.refine_into(
            &problem,
            Vector::from_vec(vec![1., 1.]).view(),
            x.view_mut(),
            &mut wide,
            ExecutionContext::sequential(),
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(x.as_slice(), [3., 4.]);
    assert_eq!(
        wide.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    let rhs = Vector::from_vec(vec![1., 1.]);
    // Evaluations are shared with the parent; its iterations are scope-local.
    for (kind, limit, termination, x_after) in [
        (
            BudgetKind::Evaluation,
            0,
            RefinementTermination::EvaluationLimit,
            [0., 0.],
        ),
        (
            BudgetKind::Evaluation,
            2,
            RefinementTermination::EvaluationLimit,
            [0., 0.],
        ),
        (
            BudgetKind::Iteration,
            0,
            RefinementTermination::Converged,
            [1., 1.],
        ),
    ] {
        let mut x = Vector::zeros(2).unwrap();
        let mut state = SolveState::new();
        let mut budgets =
            SolveBudgets::unbounded().with_limit(kind, Budget::try_from(limit.max(1)).unwrap());
        let mut ctx = context(&mut state, &mut budgets, ExecutionContext::sequential());
        if limit == 0 {
            assert!(ctx.try_consume(kind, 1).unwrap().is_granted());
        }
        let report = plan
            .refine_into(&problem, rhs.view(), x.view_mut(), &mut w, &mut ctx)
            .unwrap()[0];
        assert_eq!(report.termination(), termination);
        assert_eq!(x.as_slice(), x_after);
        let attempt = w.last_attempt().unwrap();
        assert_eq!(
            attempt.outcome(),
            if report.is_success() {
                AttemptOutcome::Succeeded
            } else {
                AttemptOutcome::Stopped
            }
        );
        assert_eq!(
            ctx.counts().nested().provider(EvaluationRole::Residual),
            attempt.counts().own().provider(EvaluationRole::Residual)
        );
    }
}

#[test]
fn empty_system_and_unsupported_precision_pairs() {
    let a = Matrix::<f64>::zeros(0, 0).unwrap();
    let low = Diagonal {
        values: Vec::<f32>::new(),
        zero: false,
    };
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &low, None, None).unwrap();
    let plan = IterativeRefinement::new(options()).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    assert!(
        plan.refine_into(
            &problem,
            VectorView::from_slice(&[]),
            VectorViewMut::from_slice(&mut []),
            &mut w,
            ExecutionContext::sequential()
        )
        .unwrap()[0]
            .is_success()
    );
    assert!(IterativeRefinement::<f64, f32>::new(options()).is_err());
    assert!(IterativeRefinement::<Complex<f32>, f64>::new(options()).is_err());
}
#[test]
fn ill_conditioned_system_does_not_confuse_backward_and_forward_error() {
    let delta = 2f64.powi(-45);
    let a = Matrix::from_vec(2, 2, vec![1., 1., 1., 1. + delta]).unwrap();
    let inverse = Diagonal {
        values: vec![1.; 2],
        zero: true,
    };
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let plan = IterativeRefinement::new(options()).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    let rhs = Vector::from_vec(vec![2., 2. + delta]);
    let mut x = Vector::from_vec(vec![1.01, 0.99]);
    let report = plan
        .refine_into(
            &problem,
            rhs.view(),
            x.view_mut(),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap()[0];
    assert!(report.is_success());
    assert!((x.as_slice()[0] - 1.).abs() > 0.009);
}

#[test]
fn cancellation_and_executor_failure_preserve_output() {
    struct Cancel;
    impl scioxide_core::Cancellation for Cancel {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    struct Omitted;
    impl scioxide_core::Executor for Omitted {
        fn worker_budget(&self) -> core::num::NonZeroUsize {
            core::num::NonZeroUsize::new(4).unwrap()
        }
        fn execute(
            &self,
            _: ExecutionContext<'_>,
            _: scioxide_core::ExecutionPlan,
            _: &scioxide_core::PartitionFn<'_>,
        ) -> Result<(), scioxide_core::SciOxideError> {
            Ok(())
        }
    }
    let a = Matrix::<f64>::identity(4).unwrap();
    let inverse = Diagonal {
        values: vec![1.; 4],
        zero: false,
    };
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let plan = IterativeRefinement::new(options()).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    let rhs = Vector::from_vec(vec![1.; 4]);
    let mut x = Vector::zeros(4).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = context(
        &mut state,
        &mut budgets,
        ExecutionContext::sequential().with_cancellation(&Cancel),
    );
    let report = plan
        .refine_into(&problem, rhs.view(), x.view_mut(), &mut w, &mut ctx)
        .unwrap()[0];
    assert!(matches!(
        report.termination(),
        RefinementTermination::ExecutionStopped(_)
    ));
    assert!(report.final_error().is_none());
    let mut ctx = context(&mut state, &mut budgets, ExecutionContext::new(&Omitted));
    assert!(
        plan.refine_into(&problem, rhs.view(), x.view_mut(), &mut w, &mut ctx)
            .is_err()
    );
    assert_eq!(x.as_slice(), [0.; 4]);
}
#[cfg(feature = "std")]
#[test]
fn executor_matches_sequential_residual_and_solution() {
    let a = Matrix::<f64>::identity(8).unwrap();
    let inverse = Diagonal {
        values: vec![1.; 8],
        zero: false,
    };
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let plan = IterativeRefinement::new(options()).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    let rhs = Vector::from_vec(vec![1.; 8]);
    let mut x = Vector::zeros(8).unwrap();
    let pool = scioxide_core::ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = context(&mut state, &mut budgets, ExecutionContext::new(&pool));
    let report = plan
        .refine_into(&problem, rhs.view(), x.view_mut(), &mut w, &mut ctx)
        .unwrap()[0];
    assert!(report.is_success());
    assert_eq!(x.as_slice(), rhs.as_slice());
    assert_eq!(report.final_error().unwrap().normwise(), 0.);
}
#[test]
fn failing_later_block_column_rolls_back_every_column() {
    struct Fails(core::sync::atomic::AtomicUsize);
    impl LinearOperator<f64> for Fails {
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
            x: &[f64],
            out: &mut [f64],
            _: &mut [f64],
        ) -> Result<(), SciOxideError> {
            out[0] = x[0];
            if self.0.fetch_add(1, core::sync::atomic::Ordering::Relaxed) == 1 {
                return Err(SciOxideError::non_finite(
                    scioxide_core::NonFiniteOrigin::Callback,
                    "injected failure",
                ));
            }
            Ok(())
        }
    }
    let a = Matrix::<f64>::identity(1).unwrap();
    let inverse = Fails(core::sync::atomic::AtomicUsize::new(0));
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let plan = IterativeRefinement::new(options()).unwrap();
    let mut w = plan
        .workspace(
            &scioxide_linalg::RefinementWorkspaceRequest::new(&problem).with_right_hand_sides(2),
        )
        .unwrap();
    let rhs = Matrix::from_vec(1, 2, vec![1., 2.]).unwrap();
    let mut x = Matrix::zeros(1, 2).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut ctx = context(&mut state, &mut budgets, ExecutionContext::sequential());
    assert!(
        plan.refine_into(&problem, rhs.view(), x.view_mut(), &mut w, &mut ctx)
            .is_err()
    );
    assert_eq!(x.as_slice(), [0., 0.]);
    assert_eq!(ctx.accepted_state().unwrap().revision(), 0);
}
#[test]
fn singular_primary_inverse_uses_explicit_fallback() {
    struct Singular;
    impl LinearOperator<f32> for Singular {
        fn rows(&self) -> usize {
            1
        }
        fn cols(&self) -> usize {
            1
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(0)
        }
        fn apply_into(&self, _: &[f32], _: &mut [f32], _: &mut [f32]) -> Result<(), SciOxideError> {
            Err(scioxide_core::SciOxideError::singular(None))
        }
    }
    let a = Matrix::<f64>::identity(1).unwrap();
    let high = Diagonal {
        values: vec![1.],
        zero: false,
    };
    let mut opts = options();
    opts = opts.with_fallback(RefinementFallback::OnFailure);
    let problem = RefinementProblem::new(
        RefinementSystem::Dense(a.view()),
        &Singular,
        None,
        Some(&high),
    )
    .unwrap();
    for (budget, success) in [(Budget::ONE, false), (Budget::ONE.saturating_add(1), true)] {
        let plan = IterativeRefinement::new(budgeted(opts, budget)).unwrap();
        let mut w = plan
            .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
            .unwrap();
        let rhs = Vector::from_vec(vec![1.]);
        let mut x = Vector::zeros(1).unwrap();
        let report = plan
            .refine_into(
                &problem,
                rhs.view(),
                x.view_mut(),
                &mut w,
                ExecutionContext::sequential(),
            )
            .unwrap()[0];
        assert_eq!(report.is_success(), success);
        assert_eq!(report.iterations(), budget.get());
        assert_eq!(
            report.fallback_reason(),
            Some(RefinementFallbackReason::NumericalFailure)
        );
        if success {
            assert_eq!(report.fallback_iterations(), 1);
            assert_eq!(x.as_slice(), [1.]);
        } else {
            assert_eq!(report.termination(), RefinementTermination::IterationLimit);
            assert_eq!(report.fallback_iterations(), 0);
            assert_eq!(x.as_slice(), [0.]);
        }
    }
}

#[test]
fn unrepresentable_tiny_residual_does_not_pass_exact_zero_tolerance() {
    let tiny = f64::from_bits(1);
    let a = Matrix::from_vec(1, 1, vec![tiny]).unwrap();
    let inverse = Diagonal {
        values: vec![1.],
        zero: true,
    };
    let mut opts = options();
    opts = opts.with_tolerance(ResidualTolerance::new(0., 0.).unwrap());
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let plan = IterativeRefinement::new(opts).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    let rhs = Vector::from_vec(vec![0.]);
    let mut x = Vector::from_vec(vec![0.25]);
    let report = plan
        .refine_into(
            &problem,
            rhs.view(),
            x.view_mut(),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap()[0];
    assert_eq!(report.termination(), RefinementTermination::Stagnation);
    assert_eq!(report.final_error().unwrap().normwise(), 1.);
}

#[test]
fn changed_provider_workspace_is_rejected_before_application() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    struct Changing {
        scratch: AtomicUsize,
        calls: AtomicUsize,
    }
    impl LinearOperator<f64> for Changing {
        fn rows(&self) -> usize {
            1
        }
        fn cols(&self) -> usize {
            1
        }
        fn apply_workspace_len(&self) -> Result<usize, SciOxideError> {
            Ok(self.scratch.load(Ordering::Relaxed))
        }
        fn apply_into(&self, x: &[f64], y: &mut [f64], _: &mut [f64]) -> Result<(), SciOxideError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            y.copy_from_slice(x);
            Ok(())
        }
    }
    let op = Changing {
        scratch: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
    };
    let inverse = Diagonal {
        values: vec![1.],
        zero: false,
    };
    let mut opts = options();
    opts = opts.with_criterion(RefinementCriterion::Normwise);
    let problem = RefinementProblem::new(
        RefinementSystem::Operator {
            operator: &op,
            norm_bound: 1.,
        },
        &inverse,
        None,
        None,
    )
    .unwrap();
    let plan = IterativeRefinement::new(opts).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    op.scratch.store(8, Ordering::Relaxed);
    let mut x = [0.];
    assert!(
        plan.refine_into(
            &problem,
            VectorView::from_slice(&[1.]),
            VectorViewMut::from_slice(&mut x),
            &mut w,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(x, [0.]);
    assert_eq!(op.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn residual_retains_error_hidden_by_rounded_products() {
    let u = 2f64.powi(-30);
    let a = Matrix::from_vec(1, 1, vec![1. + u]).unwrap();
    let inverse = Diagonal {
        values: vec![1.],
        zero: true,
    };
    let mut opts = options();
    opts = opts.with_tolerance(ResidualTolerance::new(0., 0.).unwrap());
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let plan = IterativeRefinement::new(opts).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    let rhs = Vector::from_vec(vec![1.]);
    let mut x = Vector::from_vec(vec![1. - u]);
    // Native multiplication rounds (1+u)(1-u) to one, concealing u².
    assert_eq!((1. + u) * (1. - u), 1.);
    let report = plan
        .refine_into(
            &problem,
            rhs.view(),
            x.view_mut(),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap()[0];
    assert_eq!(report.termination(), RefinementTermination::Stagnation);
    assert!((report.final_error().unwrap().normwise() - 2f64.powi(-61)).abs() < 1e-33);
}

#[test]
fn shape_errors_and_failed_scaling_do_not_publish() {
    let inverse = Diagonal {
        values: vec![1.],
        zero: false,
    };
    let nonsquare = Matrix::<f64>::zeros(1, 2).unwrap();
    assert!(matches!(
        (RefinementProblem::new(
            RefinementSystem::Dense(nonsquare.view()),
            &inverse,
            None,
            None
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    let a = Matrix::<f64>::identity(1).unwrap();
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let plan = IterativeRefinement::new(options()).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    let mut out = [7.];
    assert!(matches!(
        (plan.refine_into(
            &problem,
            VectorView::from_slice(&[]),
            VectorViewMut::from_slice(&mut out),
            &mut w,
            ExecutionContext::sequential()
        ))
        .as_ref()
        .err()
        .unwrap()
        .view(),
        scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
    ));
    assert_eq!(out, [7.]);
    let scaling = EquilibrationMetadata::from_exponents(vec![1000], vec![0]).unwrap();
    let source = Matrix::from_vec(1, 1, vec![1e200]).unwrap();
    let mut output = Matrix::from_vec(1, 1, vec![7f32]).unwrap();
    assert!(
        scaling
            .scale_matrix_into(source.view(), &mut output.view_mut())
            .is_err()
    );
    assert_eq!(output.as_slice(), [7.]);
}

#[test]
fn nonfinite_inverse_output_and_missing_fallback_preserve_output() {
    let a = Matrix::<f64>::identity(1).unwrap();
    let inverse = Diagonal {
        values: vec![0f32],
        zero: false,
    };
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let plan = IterativeRefinement::new(options()).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    let mut x = [0.];
    let error = plan
        .refine_into(
            &problem,
            VectorView::from_slice(&[1.]),
            VectorViewMut::from_slice(&mut x),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Callback,
            ..
        }
    ));
    assert_eq!(x, [0.]);
    assert_eq!(w.last_attempt().unwrap().outcome(), AttemptOutcome::Failed);
    let inverse = Diagonal {
        values: vec![1f32],
        zero: false,
    };
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let plan =
        IterativeRefinement::new(options().with_fallback(RefinementFallback::OnFailure)).unwrap();
    let mut w = RefinementWorkspace::new();
    let mut x = [7.];
    assert_eq!(
        plan.refine_into(
            &problem,
            VectorView::from_slice(&[1.]),
            VectorViewMut::from_slice(&mut x),
            &mut w,
            ExecutionContext::sequential(),
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(x, [7.]);
    assert_eq!(
        w.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
}

#[test]
fn widened_corrections_flush_underflow_and_route_overflow_to_conversion() {
    let a = Matrix::<f64>::identity(1).unwrap();
    let run = |divisor: f32, rhs: f64, fallback: Option<&Diagonal<f64>>| {
        let inverse = Diagonal {
            values: vec![divisor],
            zero: false,
        };
        let problem = RefinementProblem::new(
            RefinementSystem::Dense(a.view()),
            &inverse,
            None,
            fallback.map(|f| f as &(dyn LinearOperator<f64> + Sync)),
        )
        .unwrap();
        let policy = if fallback.is_some() {
            RefinementFallback::OnFailure
        } else {
            RefinementFallback::Disabled
        };
        let plan = IterativeRefinement::new(options().with_fallback(policy)).unwrap();
        let mut w = plan
            .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
            .unwrap();
        let mut x = [0.];
        let report = plan
            .refine_into(
                &problem,
                VectorView::from_slice(&[rhs]),
                VectorViewMut::from_slice(&mut x),
                &mut w,
                ExecutionContext::sequential(),
            )
            .unwrap()[0];
        (report, x, w.last_attempt().unwrap().outcome())
    };
    // A 2^-1080 correction is far below the subnormal range: it rounds to zero
    // instead of failing, and the unchanged iterate stagnates.
    let rhs = f64::from_bits(1 << 14);
    assert_eq!(rhs, 2f64.powi(-1000) * 2f64.powi(-60));
    let (report, x, outcome) = run(2f32.powi(20), rhs, None);
    assert_eq!(report.termination(), RefinementTermination::Stagnation);
    assert_eq!(report.fallback_reason(), None);
    assert_eq!(x, [0.]);
    assert_eq!(outcome, AttemptOutcome::Stopped);
    // Widening 2 * MAX overflows: without fallback the correction fails as a report.
    let (report, x, outcome) = run(0.5, f64::MAX, None);
    assert_eq!(
        report.termination(),
        RefinementTermination::CorrectionFailed
    );
    assert_eq!(report.fallback_reason(), None);
    assert_eq!(x, [0.]);
    assert_eq!(outcome, AttemptOutcome::Stopped);
    // With fallback, the overflow switches once to the residual-precision inverse.
    let high = Diagonal {
        values: vec![1.],
        zero: false,
    };
    let (report, x, outcome) = run(0.5, f64::MAX, Some(&high));
    assert_eq!(report.termination(), RefinementTermination::Converged);
    assert_eq!(
        report.fallback_reason(),
        Some(RefinementFallbackReason::Conversion)
    );
    assert_eq!(report.fallback_iterations(), 1);
    assert_eq!(x, [f64::MAX]);
    assert_eq!(outcome, AttemptOutcome::Succeeded);
}

#[test]
fn default_and_composed_iteration_budgets_bound_corrections() {
    let a = Matrix::<f64>::identity(2).unwrap();
    // Each correction halves the error, so convergence needs more than ten steps.
    let inverse = Diagonal {
        values: vec![2.; 2],
        zero: false,
    };
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let tolerance = ResidualTolerance::new(0., 1e-14).unwrap();
    let refine = |options: RefinementOptions<f64>| {
        IterativeRefinement::<f64, f64>::new(options)
            .unwrap()
            .refine(&problem, &[1., 1.][..], &[0., 0.][..])
            .unwrap()
            .1
    };
    let default = refine(RefinementOptions::new(tolerance));
    assert_eq!(default.termination(), RefinementTermination::IterationLimit);
    assert_eq!(default.iterations(), 10);
    let three = refine(budgeted(
        RefinementOptions::new(tolerance),
        Budget::try_from(3).unwrap(),
    ));
    assert_eq!(three.termination(), RefinementTermination::IterationLimit);
    assert_eq!(three.iterations(), 3);
    let enough = refine(budgeted(
        RefinementOptions::new(tolerance),
        Budget::try_from(80).unwrap(),
    ));
    assert!(enough.is_success());
    assert_eq!(
        RefinementOptions::<f64>::new(tolerance).execution_options(),
        LinalgExecutionOptions::new()
    );
}

#[test]
fn owned_refinement_rejects_time_limits_without_a_clock() {
    let a = Matrix::<f64>::identity(1).unwrap();
    let inverse = Diagonal {
        values: vec![1.],
        zero: false,
    };
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let plan = IterativeRefinement::<f64, f64>::new(
        RefinementOptions::new(ResidualTolerance::new(0., 1e-14).unwrap()).with_execution_options(
            LinalgExecutionOptions::new().with_time_limit(core::time::Duration::from_secs(1)),
        ),
    )
    .unwrap();
    let error = plan.refine(&problem, &[1.][..], &[0.][..]).unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::CapabilityUnavailable);
}

#[test]
fn provider_and_equilibration_shape_mismatches_are_dimension_errors() {
    let shape = |result: Result<(), SciOxideError>| {
        assert_eq!(
            result.unwrap_err().kind(),
            SciOxideErrorKind::DimensionMismatch
        );
    };
    let a = Matrix::<f64>::identity(2).unwrap();
    let system = RefinementSystem::Dense(a.view());
    let inverse = |n| Diagonal {
        values: vec![1f32; n],
        zero: false,
    };
    let (two, three) = (inverse(2), inverse(3));
    let high = Diagonal {
        values: vec![1f64; 3],
        zero: false,
    };
    shape(RefinementProblem::new(system, &three, None, None).map(|_| ()));
    shape(RefinementProblem::new(system, &two, None, Some(&high)).map(|_| ()));
    let scaling = EquilibrationMetadata::from_exponents(vec![0; 3], vec![0; 3]).unwrap();
    shape(RefinementProblem::new(system, &two, Some(&scaling), None).map(|_| ()));
    shape(EquilibrationMetadata::from_exponents(vec![0; 2], vec![0; 3]).map(|_| ()));
    let scaling = EquilibrationMetadata::from_exponents(vec![0; 2], vec![0; 2]).unwrap();
    let mut short = [0.0; 1];
    shape(scaling.scale_rhs_into(
        VectorView::from_slice(&[1.0, 2.0]),
        &mut VectorViewMut::from_slice(&mut short),
    ));
    let mut wide = Matrix::<f32>::zeros(2, 3).unwrap();
    shape(scaling.scale_matrix_into(a.view(), &mut wide.view_mut()));
}

#[test]
fn explicit_residual_rows_follow_the_execution_policy() {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionPlan, ExecutionPolicy, Executor, PartitionFn};
    // Runs partitions inline while recording the widest plan it receives.
    struct Recording(core::sync::atomic::AtomicUsize);
    impl Executor for Recording {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(4).unwrap()
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: ExecutionPlan,
            operation: &PartitionFn<'_>,
        ) -> Result<(), SciOxideError> {
            self.0
                .fetch_max(plan.worker_count(), core::sync::atomic::Ordering::Relaxed);
            scioxide_core::SequentialExecutor.execute(context, plan, operation)
        }
    }
    let executor = Recording(core::sync::atomic::AtomicUsize::new(0));
    let n = 8;
    let a = Matrix::<f64>::identity(n).unwrap();
    let inverse = Diagonal {
        values: vec![1f32; n],
        zero: false,
    };
    let problem =
        RefinementProblem::new(RefinementSystem::Dense(a.view()), &inverse, None, None).unwrap();
    let sequential = options().with_execution_options(
        options::<f64>()
            .execution_options()
            .with_execution_policy(ExecutionPolicy::Sequential),
    );
    let plan = IterativeRefinement::new(sequential).unwrap();
    let mut w = plan
        .workspace(&scioxide_linalg::RefinementWorkspaceRequest::new(&problem))
        .unwrap();
    let rhs = vec![1.0; n];
    let mut x = vec![0.0; n];
    let report = plan
        .refine_into(
            &problem,
            VectorView::from_slice(&rhs),
            VectorViewMut::from_slice(&mut x),
            &mut w,
            ExecutionContext::new(&executor),
        )
        .unwrap()[0];
    assert!(report.is_success());
    assert_eq!(x, rhs);
    assert_eq!(executor.0.load(core::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn equilibration_exponents_name_the_out_of_range_side() {
    use scioxide_core::{InvalidInputReason, SciOxideErrorView};
    for (rows, columns, side) in [
        (vec![4097], vec![0], scioxide_core::argument!("rows")),
        (vec![0], vec![-4097], scioxide_core::argument!("columns")),
    ] {
        let error = EquilibrationMetadata::from_exponents(rows, columns).unwrap_err();
        assert!(
            matches!(
                error.view(),
                SciOxideErrorView::InvalidInput {
                    argument,
                    reason: InvalidInputReason::OutOfRange,
                    ..
                } if argument == side
            ),
            "{error:?}"
        );
    }
    assert!(EquilibrationMetadata::from_exponents(vec![4096], vec![-4096]).is_ok());
}
