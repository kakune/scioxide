//! Sparse power and function correctness, resources, and failure contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, Capacity, Complex, ExecutionContext, NumericScalar,
    ProblemToken, RealScalar, SciOxideErrorKind, SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_linalg::{
    AnalyticMatrixFunction, Estimation, LinalgWorkspace, Matrix, MatrixFunction,
    MatrixFunctionOptions, MatrixFunctionPlan, SparseFunctionOptions, SparseFunctionPlan,
    SparsePowerOptions, SparsePowerPlan,
};
use scioxide_sparse::{CsrMatrix, DiaMatrix, SparseProductPlanOptions};

fn powers(entries: usize) -> SparsePowerOptions {
    SparsePowerOptions::new()
        .with_max_entries(entries)
        .with_execution_options(
            scioxide_linalg::LinalgExecutionOptions::new().with_byte_limit(2_000_000),
        )
}

fn functions<R: RealScalar>(entries: usize) -> SparseFunctionOptions<R> {
    SparseFunctionOptions::new()
        .with_max_entries(entries)
        .with_execution_options(
            scioxide_linalg::LinalgExecutionOptions::new().with_byte_limit(2_000_000),
        )
}

fn diagonal() -> CsrMatrix<f64> {
    CsrMatrix::from_raw(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 2.0, 4.0]).unwrap()
}

#[test]
fn diagonal_powers_and_functions() {
    let a = diagonal();
    let cube = SparsePowerPlan::new(a.view(), 3, &powers(3))
        .unwrap()
        .evaluate(a.view())
        .unwrap();
    assert_eq!(cube.report().structural_fill(), 3);
    assert_eq!(cube.value().values(), &[1.0, 8.0, 64.0]);
    let identity = SparsePowerPlan::new(a.view(), 0, &powers(3))
        .unwrap()
        .evaluate(a.view())
        .unwrap();
    assert_eq!(identity.value().values(), &[1.0, 1.0, 1.0]);
    let inverse = SparsePowerPlan::new(a.view(), -1, &powers(3))
        .unwrap()
        .evaluate(a.view())
        .unwrap();
    for (value, expected) in inverse.value().values().iter().zip([1.0, 0.5, 0.25]) {
        assert!((value - expected).abs() < 1e-13);
    }
    let exp = SparseFunctionPlan::new(a.view(), MatrixFunction::Exp, &functions(3))
        .unwrap()
        .evaluate(a.view())
        .unwrap();
    assert_eq!(exp.report().structural_fill(), 3);
    assert_eq!(exp.report().nonzero_fill(), 3);
    assert!(exp.report().estimated_error().is_none());
    for (value, expected) in exp.value().values().iter().zip([1.0_f64, 2.0, 4.0]) {
        assert!((value - expected.exp()).abs() < 1e-10);
    }
    let estimated = SparseFunctionPlan::new(
        a.view(),
        MatrixFunction::Exp,
        &functions(3).with_error_estimate(Estimation::Estimate),
    )
    .unwrap()
    .evaluate(a.view())
    .unwrap();
    assert!(estimated.report().estimated_error().unwrap().is_finite());
    for (value, expected) in estimated.value().values().iter().zip(exp.value().values()) {
        assert_eq!(value, expected);
    }
}

#[test]
fn sparse_block_exp_and_fill_budgets() {
    let a = CsrMatrix::from_raw(3, 3, vec![0, 1, 2, 2], vec![1, 2], vec![1.0_f64, 2.0]).unwrap();
    assert_eq!(
        SparseFunctionPlan::new(a.view(), MatrixFunction::Exp, &functions(8))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    let plan = SparseFunctionPlan::new(a.view(), MatrixFunction::Exp, &functions(9)).unwrap();
    let result = plan.evaluate(a.view()).unwrap();
    assert!((result.value().get(0, 2).unwrap() - 1.0).abs() < 1e-10);
    assert!((result.value().get(0, 1).unwrap() - 1.0).abs() < 1e-10);
    assert!((result.value().get(1, 2).unwrap() - 2.0).abs() < 1e-10);
    // The full block keeps exact zeros below the diagonal as structure.
    assert_eq!(result.value().nnz(), 9);
    assert_eq!(result.report().nonzero_fill(), 6);
    assert_eq!(
        SparsePowerPlan::new(a.view(), 2, &powers(1))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
}

#[test]
fn pattern_changes_and_singular_inverse_are_errors() {
    let a = diagonal();
    let changed = CsrMatrix::from_raw(3, 3, vec![0, 1, 2, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
    let plan = SparsePowerPlan::new(a.view(), 2, &powers(3)).unwrap();
    assert_eq!(
        plan.evaluate(changed.view()).unwrap_err().kind(),
        SciOxideErrorKind::InvalidInput
    );
    let singular = CsrMatrix::from_raw(2, 2, vec![0, 1, 1], vec![0], vec![1.0]).unwrap();
    let negative = SparsePowerPlan::new(singular.view(), -1, &powers(4)).unwrap();
    let mut output = negative.create_output().unwrap();
    output
        .copy_values_from(&vec![7.0; output.values().len()])
        .unwrap();
    let before = output.clone();
    assert!(
        negative
            .evaluate_into(
                singular.view(),
                &mut output,
                &mut negative.workspace(()).unwrap(),
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(output.values(), before.values());
    let rectangular = CsrMatrix::from_raw(1, 2, vec![0, 1], vec![1], vec![1.0]).unwrap();
    assert_eq!(
        SparsePowerPlan::new(rectangular.view(), 2, &powers(4))
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
}

struct Identity;
impl AnalyticMatrixFunction<f64> for Identity {
    fn radius(&self, _: Complex<f64>) -> f64 {
        10.0
    }
    fn coefficient(
        &self,
        center: Complex<f64>,
        order: usize,
    ) -> Result<Complex<f64>, scioxide_core::SciOxideError> {
        Ok(match order {
            0 => center,
            1 => Complex::new(1.0, 0.0),
            _ => Complex::new(0.0, 0.0),
        })
    }
    fn tail_bound(
        &self,
        _: Complex<f64>,
        _: usize,
        _: f64,
    ) -> Result<f64, scioxide_core::SciOxideError> {
        Ok(0.0)
    }
}

#[test]
fn custom_analytic_function_and_complex_power() {
    let a = diagonal();
    let result =
        SparseFunctionPlan::new(a.view(), MatrixFunction::Analytic(&Identity), &functions(3))
            .unwrap()
            .evaluate(a.view())
            .unwrap();
    assert_eq!(result.value().values(), &[1.0, 2.0, 4.0]);
    assert_eq!(
        SparseFunctionPlan::new(
            a.view(),
            MatrixFunction::Analytic(&Identity),
            &functions(3).with_error_estimate(Estimation::Estimate)
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::InvalidInput
    );
    let complex = CsrMatrix::from_raw(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![Complex::new(0.0_f32, 1.0), Complex::new(2.0, 0.0)],
    )
    .unwrap();
    let square = SparsePowerPlan::new(complex.view(), 2, &powers(2))
        .unwrap()
        .evaluate(complex.view())
        .unwrap();
    assert!((square.value().values()[0].re + 1.0).abs() < 1e-5);
    assert!(square.value().values()[0].im.abs() < 1e-5);
}

fn dense_reference<S: NumericScalar>()
where
    Complex<S::Real>: NumericScalar<Real = S::Real>,
{
    let s = |value| S::from_real(S::Real::try_from_f64_lossy(value).unwrap());
    let a = CsrMatrix::from_raw(
        2,
        2,
        vec![0, 2, 3],
        vec![0, 1, 1],
        vec![s(1.0), s(0.125), s(2.0)],
    )
    .unwrap();
    let sparse = SparseFunctionPlan::new(a.view(), MatrixFunction::Exp, &functions(4))
        .unwrap()
        .evaluate(a.view())
        .unwrap();
    let dense = Matrix::from_csr(&a).unwrap();
    let mut reference = Matrix::<Complex<S::Real>>::zeros(2, 2).unwrap();
    MatrixFunctionPlan::new(
        &dense,
        MatrixFunction::Exp,
        &MatrixFunctionOptions::default(),
    )
    .unwrap()
    .evaluate_into(
        reference.view_mut(),
        &mut LinalgWorkspace::new(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    let allowance = if core::mem::size_of::<S::Real>() == 4 {
        1e-4
    } else {
        1e-11
    };
    for row in 0..2 {
        for col in 0..2 {
            let actual = sparse.value().get(row, col).unwrap();
            let expected = reference.view().get(row, col).unwrap();
            assert!(
                (actual.real_component().into_f64() - expected.re.into_f64()).abs() < allowance
            );
            assert!(
                (actual.imaginary_component().into_f64() - expected.im.into_f64()).abs()
                    < allowance
            );
        }
    }
    let cube = SparsePowerPlan::new(a.view(), 3, &powers(4))
        .unwrap()
        .evaluate(a.view())
        .unwrap();
    assert!(
        (cube.value().get(0, 1).unwrap().real_component().into_f64() - 0.875).abs() < allowance
    );
}

#[test]
fn dense_reference_all_scalar_domains() {
    dense_reference::<f32>();
    dense_reference::<f64>();
    dense_reference::<Complex<f32>>();
    dense_reference::<Complex<f64>>();
}

fn nonreal_complex_domain<R: RealScalar>()
where
    Complex<R>: NumericScalar<Real = R>,
{
    let a = CsrMatrix::from_raw(
        1,
        1,
        vec![0, 1],
        vec![0],
        vec![Complex::new(R::ZERO, R::ONE)],
    )
    .unwrap();
    let output = SparseFunctionPlan::new(a.view(), MatrixFunction::Exp, &functions(1))
        .unwrap()
        .evaluate(a.view())
        .unwrap();
    let value = output.value().get(0, 0).unwrap();
    let allowance = if core::mem::size_of::<R>() == 4 {
        1e-5
    } else {
        1e-12
    };
    assert!((value.re.into_f64() - 1.0_f64.cos()).abs() < allowance);
    assert!((value.im.into_f64() - 1.0_f64.sin()).abs() < allowance);
    let square = SparsePowerPlan::new(a.view(), 2, &powers(1))
        .unwrap()
        .evaluate(a.view())
        .unwrap();
    assert!((square.value().get(0, 0).unwrap().re.into_f64() + 1.0).abs() < allowance);
}

#[test]
fn nonreal_complex_functions_in_both_precisions() {
    nonreal_complex_domain::<f32>();
    nonreal_complex_domain::<f64>();
}

#[test]
fn dia_padding_is_not_staged_by_conversion() {
    let order = 400;
    let mut ptr = vec![0usize; order + 1];
    ptr[1] = 2;
    for item in &mut ptr[2..order - 1] {
        *item = 2;
    }
    ptr[order - 1] = 3;
    ptr[order] = 4;
    let matrix = CsrMatrix::from_raw(
        order,
        order,
        ptr,
        vec![order - 2, order - 1, 0, 0],
        vec![1.0_f64; 4],
    )
    .unwrap();
    let dia = DiaMatrix::from_sparse(matrix.view()).unwrap();
    assert!(dia.view().values().len() > order);
    let bytes = |view| {
        SparseFunctionPlan::new(view, MatrixFunction::Exp, &functions(order + 12))
            .unwrap()
            .workspace_requirements(())
            .unwrap()
            .required_bytes()
    };
    let (compact, padded) = (bytes(matrix.view()), bytes(dia.view()));
    let limited = |bytes| {
        functions::<f64>(order + 12).with_execution_options(
            (functions::<f64>(order + 12))
                .execution_options()
                .with_byte_limit(bytes),
        )
    };
    assert!(SparseFunctionPlan::new(matrix.view(), MatrixFunction::Exp, &limited(compact)).is_ok());
    // Conversion emits canonical CSR directly without staging padded slots, so
    // it fits within the plan's own workspace.
    let report = SparseFunctionPlan::new(dia.view(), MatrixFunction::Exp, &limited(padded))
        .unwrap()
        .evaluate(dia.view())
        .unwrap()
        .report();
    assert_eq!(report.admitted_workspace_bytes(), padded);
}

#[test]
fn empty_identity_and_real_to_complex_branch() {
    let empty = CsrMatrix::<f64>::from_raw(0, 0, vec![0], vec![], vec![]).unwrap();
    let power = SparsePowerPlan::new(empty.view(), 0, &powers(0))
        .unwrap()
        .evaluate(empty.view())
        .unwrap();
    assert_eq!(power.value().nnz(), 0);
    let function = SparseFunctionPlan::new(empty.view(), MatrixFunction::Exp, &functions(0))
        .unwrap()
        .evaluate(empty.view())
        .unwrap();
    assert_eq!(function.value().nnz(), 0);
    let negative = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![-1.0_f64]).unwrap();
    let plan =
        SparseFunctionPlan::new(negative.view(), MatrixFunction::Sqrt, &functions(1)).unwrap();
    assert_eq!(
        plan.evaluate(negative.view()).unwrap_err().kind(),
        SciOxideErrorKind::DomainError
    );
    let mut root = plan.create_output::<Complex<f64>>().unwrap();
    plan.evaluate_into(
        negative.view(),
        &mut root,
        &mut plan.workspace(()).unwrap(),
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert!(root.get(0, 0).unwrap().re.abs() < 1e-12);
    assert!((root.get(0, 0).unwrap().im - 1.0).abs() < 1e-12);
    // The spectrum is only known per evaluation, so the cut is rejected then.
    let log = SparseFunctionPlan::new(negative.view(), MatrixFunction::Log, &functions(1)).unwrap();
    assert_eq!(
        log.evaluate(negative.view()).unwrap_err().kind(),
        SciOxideErrorKind::DomainError
    );
}

#[test]
fn diagonal_preparation_scales_linearly_and_byte_budget_rejects() {
    let n = 512;
    let mut row_ptr = Vec::with_capacity(n + 1);
    row_ptr.extend(0..=n);
    let mut columns = Vec::with_capacity(n);
    columns.extend(0..n);
    let matrix = CsrMatrix::from_raw(n, n, row_ptr, columns, vec![1.0_f64; n]).unwrap();
    let plan = SparseFunctionPlan::new(
        matrix.view(),
        MatrixFunction::Exp,
        &SparseFunctionOptions::new()
            .with_max_entries(n)
            .with_execution_options(
                scioxide_linalg::LinalgExecutionOptions::new().with_byte_limit(1_000_000),
            ),
    )
    .unwrap();
    assert_eq!(plan.structural_fill(), n);
    assert_eq!(
        SparseFunctionPlan::new(
            matrix.view(),
            MatrixFunction::Exp,
            &SparseFunctionOptions::new()
                .with_max_entries(n)
                .with_execution_options(
                    scioxide_linalg::LinalgExecutionOptions::new().with_byte_limit(1000)
                ),
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::ResourceLimit
    );
}

#[test]
fn bounded_power_accepts_dense_overlap_paths() {
    let a = CsrMatrix::from_raw(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![1.0_f64; 9],
    )
    .unwrap();
    let result = SparsePowerPlan::new(a.view(), 2, &powers(9))
        .unwrap()
        .evaluate(a.view())
        .unwrap();
    assert_eq!(result.value().nnz(), 9);
    assert!(result.value().values().iter().all(|&x| x == 3.0));
}

#[test]
fn executor_and_cancellation_forms() {
    use core::num::NonZeroUsize;
    use scioxide_core::{CancellationToken, ExecutionPolicy};

    let a = diagonal();
    let workers = NonZeroUsize::new(2).unwrap();
    let plan = SparsePowerPlan::new(
        a.view(),
        3,
        &powers(3).with_execution_options(
            (powers(3))
                .execution_options()
                .with_execution_policy(ExecutionPolicy::Parallel(workers)),
        ),
    )
    .unwrap();
    #[cfg(feature = "std")]
    let executor = scioxide_core::ThreadExecutor::new(workers);
    #[cfg(not(feature = "std"))]
    let executor = {
        // Exercise admitted partitions through the portable injected interface.
        struct PartitionExecutor(NonZeroUsize);
        impl scioxide_core::Executor for PartitionExecutor {
            fn worker_budget(&self) -> NonZeroUsize {
                self.0
            }
            fn execute(
                &self,
                context: ExecutionContext<'_>,
                plan: scioxide_core::ExecutionPlan,
                operation: &scioxide_core::PartitionFn<'_>,
            ) -> Result<(), scioxide_core::SciOxideError> {
                scioxide_core::SequentialExecutor.execute(context, plan, operation)
            }
        }
        PartitionExecutor(workers)
    };
    let mut workspace = plan.workspace(()).unwrap();
    let mut output = plan.create_output().unwrap();
    plan.evaluate_into(
        a.view(),
        &mut output,
        &mut workspace,
        ExecutionContext::new(&executor),
    )
    .unwrap();
    assert_eq!(output.values(), &[1.0, 8.0, 64.0]);

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    output
        .copy_values_from(&vec![0.0; output.values().len()])
        .unwrap();
    assert!(
        plan.evaluate_into(
            a.view(),
            &mut output,
            &mut workspace,
            ExecutionContext::new(&executor).with_cancellation(&cancelled),
        )
        .is_err()
    );
    assert_eq!(output.values(), &[0.0; 3]);
    let functions = SparseFunctionPlan::new(a.view(), MatrixFunction::Exp, &functions(3)).unwrap();
    let mut workspace = functions.workspace(()).unwrap();
    let mut output = functions.create_output::<Complex<f64>>().unwrap();
    assert!(
        functions
            .evaluate_into(
                a.view(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential().with_cancellation(&cancelled),
            )
            .is_err()
    );
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    assert!(output.values().iter().all(|z| *z == Complex::new(0.0, 0.0)));
}

#[test]
fn product_workers_follow_the_execution_policy() {
    use core::num::NonZeroUsize;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use scioxide_core::ExecutionPolicy;

    /// Records the widest admitted partition plan and runs it inline.
    struct Recording(AtomicUsize);
    impl scioxide_core::Executor for Recording {
        fn worker_budget(&self) -> NonZeroUsize {
            NonZeroUsize::new(4).unwrap()
        }
        fn execute(
            &self,
            context: ExecutionContext<'_>,
            plan: scioxide_core::ExecutionPlan,
            operation: &scioxide_core::PartitionFn<'_>,
        ) -> Result<(), scioxide_core::SciOxideError> {
            self.0.fetch_max(plan.worker_count(), Ordering::Relaxed);
            scioxide_core::SequentialExecutor.execute(context, plan, operation)
        }
    }
    let a = diagonal();
    let parallel = |n| ExecutionPolicy::Parallel(NonZeroUsize::new(n).unwrap());
    for (policy, widest) in [
        (ExecutionPolicy::Sequential, 1),
        (parallel(2), 2),
        (parallel(65), 3),
    ] {
        let options = powers(3)
            .with_execution_options(powers(3).execution_options().with_execution_policy(policy));
        let plan = SparsePowerPlan::new(a.view(), 3, &options).unwrap();
        let mut workspace = plan.workspace(()).unwrap();
        let mut output = plan.create_output().unwrap();
        let executor = Recording(AtomicUsize::new(1));
        plan.evaluate_into(
            a.view(),
            &mut output,
            &mut workspace,
            ExecutionContext::new(&executor),
        )
        .unwrap();
        assert_eq!(output.values(), &[1.0, 8.0, 64.0], "{policy:?}");
        assert_eq!(executor.0.load(Ordering::Relaxed), widest, "{policy:?}");
    }
}

#[test]
fn iteration_budget_is_local_and_parents_see_nested_counts() {
    let a = diagonal();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let identity = SolveIdentity::new(
        ProblemToken::new(690),
        scioxide_core::algorithm_id!("linalg", "sparse-power", 1),
        3,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    // A parent's own iteration allowance is not consumed by the child epoch.
    let power = SparsePowerPlan::new(a.view(), 3, &powers(3)).unwrap();
    let mut workspace = power.workspace(()).unwrap();
    let mut output = power.create_output().unwrap();
    let report = power
        .evaluate_into(a.view(), &mut output, &mut workspace, &mut context)
        .unwrap();
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), report.work());
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
    assert_eq!(attempt.counts().own().iterations(), report.work());
    // The plan's own iteration budget bounds the products.
    let limited = SparsePowerPlan::new(
        a.view(),
        3,
        &powers(3).with_execution_options(
            (powers(3))
                .execution_options()
                .with_iteration_budget(Budget::ONE),
        ),
    )
    .unwrap();
    let mut workspace = limited.workspace(()).unwrap();
    let mut output = limited.create_output().unwrap();
    assert_eq!(
        limited
            .evaluate_into(a.view(), &mut output, &mut workspace, &mut context)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
    assert_eq!(output.values(), &[0.0; 3]);
    assert_eq!(
        workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
}

#[test]
fn function_work_limit_covers_all_blocks() {
    let singleton = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![1.0_f64]).unwrap();
    let single_work = SparseFunctionPlan::new(singleton.view(), MatrixFunction::Exp, &functions(1))
        .unwrap()
        .evaluate(singleton.view())
        .unwrap()
        .report()
        .work();
    assert!(single_work > 0);
    let double = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]).unwrap();
    let options = functions(2);
    let options = options.with_execution_options(
        options
            .execution_options()
            .with_iteration_budget(Budget::try_from(single_work).unwrap()),
    );

    assert_eq!(
        SparseFunctionPlan::new(double.view(), MatrixFunction::Exp, &options)
            .unwrap()
            .evaluate(double.view())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ConvergenceFailure
    );
}

#[test]
fn sparse_inverse_respects_its_local_budget_on_initial_and_reused_factors() {
    let a = diagonal();
    let limited = SparsePowerPlan::new(
        a.view(),
        -1,
        &powers(9).with_execution_options(
            scioxide_linalg::LinalgExecutionOptions::new()
                .with_iteration_budget(Budget::try_from(5).unwrap()),
        ),
    )
    .unwrap();
    let mut workspace = limited.workspace(()).unwrap();
    let mut output = limited.create_output().unwrap();
    for _ in 0..2 {
        assert_eq!(
            limited
                .evaluate_into(
                    a.view(),
                    &mut output,
                    &mut workspace,
                    ExecutionContext::sequential()
                )
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::ConvergenceFailure
        );
        assert!(output.values().iter().all(|&value| value == 0.));
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::Stopped
        );
    }
}

#[test]
fn reservation_admits_retained_dense_capacity_before_growth() {
    // A function plan on one connected 6 x 6 block retains dense arenas.
    let n = 6;
    let block = CsrMatrix::from_raw(
        n,
        n,
        (0..=n).map(|row| row * n).collect(),
        (0..n * n).map(|entry| entry % n).collect(),
        (0..n * n)
            .map(|entry| if entry % (n + 1) == 0 { 1.0_f64 } else { 0.1 })
            .collect(),
    )
    .unwrap();
    let function =
        SparseFunctionPlan::new(block.view(), MatrixFunction::Exp, &functions(n * n)).unwrap();
    let function_requirements = function.workspace_requirements(()).unwrap();
    let mut workspace = function.workspace(()).unwrap();
    // A power plan whose own bytes meet its limit, which the retained dense
    // capacity would exceed once its sparse buffers replace the old ones.
    let m = 4096;
    let diagonal =
        CsrMatrix::from_raw(m, m, (0..=m).collect(), (0..m).collect(), vec![2.0_f64; m]).unwrap();
    // Query the typed product workspace before imposing the exact byte limit;
    // the helper's small-fixture cap need not fit all worker-local column maps.
    let unbounded =
        powers(m).with_execution_options(powers(m).execution_options().with_byte_limit(usize::MAX));
    let limit = SparsePowerPlan::new(diagonal.view(), 2, &unbounded)
        .unwrap()
        .workspace_requirements(())
        .unwrap()
        .required_bytes();
    assert!(limit > function_requirements.required_bytes());
    let power = SparsePowerPlan::new(
        diagonal.view(),
        2,
        &powers(m).with_execution_options((powers(m)).execution_options().with_byte_limit(limit)),
    )
    .unwrap();
    let power_requirements = power.workspace_requirements(()).unwrap();
    assert_eq!(power_requirements.required_bytes(), limit);
    // `check` reports the storage the workspace would hold after growth.
    assert_eq!(
        workspace.check(&power_requirements).unwrap_err().kind(),
        SciOxideErrorKind::ResourceLimit
    );
    assert_eq!(
        workspace
            .try_reserve(&power_requirements)
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::ResourceLimit
    );
    // No buffer was replaced.
    assert_eq!(
        workspace.check(&function_requirements).unwrap(),
        Capacity::Sufficient
    );
}

#[test]
fn reservation_for_another_layout_ignores_the_buffers_it_replaces() {
    let diagonal = |n: usize| {
        CsrMatrix::from_raw(n, n, (0..=n).collect(), (0..n).collect(), vec![2.0_f64; n]).unwrap()
    };
    let large = diagonal(4096);
    let unbounded = powers(4096)
        .with_execution_options(powers(4096).execution_options().with_byte_limit(usize::MAX));
    let large_plan = SparsePowerPlan::new(large.view(), 3, &unbounded).unwrap();
    let mut workspace = large_plan.workspace(()).unwrap();
    // A small plan whose own storage fits a limit below the retained bytes.
    let small = diagonal(2);
    let limit = 1 << 16;
    let options =
        powers(2).with_execution_options(powers(2).execution_options().with_byte_limit(limit));
    let small_plan = SparsePowerPlan::new(small.view(), 3, &options).unwrap();
    let requirements = small_plan.workspace_requirements(()).unwrap();
    assert!(
        large_plan
            .workspace_requirements(())
            .unwrap()
            .required_bytes()
            > limit
    );
    assert!(requirements.required_bytes() <= limit);
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::NeedsGrowth
    );
    workspace.try_reserve(&requirements).unwrap();
    assert_eq!(
        workspace.check(&requirements).unwrap(),
        Capacity::Sufficient
    );
    let mut output = small_plan.create_output().unwrap();
    small_plan
        .evaluate_into(
            small.view(),
            &mut output,
            &mut workspace,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(output.values(), &[8.0, 8.0]);
}

#[test]
fn output_pattern_time_limit_and_unchanged_output() {
    let a = diagonal();
    let plan = SparsePowerPlan::new(a.view(), 2, &powers(3)).unwrap();
    let mut workspace = plan.workspace(()).unwrap();
    let mut wrong =
        CsrMatrix::from_raw(3, 3, vec![0, 1, 1, 2], vec![0, 2], vec![5.0, 6.0]).unwrap();
    assert_eq!(
        plan.evaluate_into(
            a.view(),
            &mut wrong,
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(wrong.values(), &[5.0, 6.0]);
    let mut small = CsrMatrix::from_raw(1, 1, vec![0, 1], vec![0], vec![5.0]).unwrap();
    assert_eq!(
        plan.evaluate_into(
            a.view(),
            &mut small,
            &mut workspace,
            ExecutionContext::sequential()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::DimensionMismatch
    );
    let mut nonfinite = diagonal();
    assert_eq!(
        nonfinite.set_value(0, f64::NAN).unwrap_err().kind(),
        SciOxideErrorKind::NonFinite
    );
    let mut output = plan.create_output().unwrap();
    output.copy_values_from(&[3.0; 3]).unwrap();
    let timed = SparsePowerPlan::new(
        a.view(),
        2,
        &powers(3).with_execution_options(
            (powers(3))
                .execution_options()
                .with_time_limit(core::time::Duration::from_secs(1)),
        ),
    )
    .unwrap();
    assert_eq!(
        timed.evaluate(a.view()).unwrap_err().kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    assert_eq!(
        timed
            .evaluate_into(
                a.view(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential()
            )
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
    assert_eq!(output.values(), &[3.0; 3]);
    let function = SparseFunctionPlan::new(
        a.view(),
        MatrixFunction::Exp,
        &functions(3).with_execution_options(
            (functions::<f64>(3))
                .execution_options()
                .with_time_limit(core::time::Duration::from_secs(1)),
        ),
    )
    .unwrap();
    assert_eq!(
        function.evaluate(a.view()).unwrap_err().kind(),
        SciOxideErrorKind::CapabilityUnavailable
    );
}

fn arrowhead(n: usize) -> CsrMatrix<f64> {
    let mut pointers = vec![0, n];
    let mut columns: Vec<_> = (0..n).collect();
    for row in 1..n {
        columns.extend([0, row]);
        pointers.push(columns.len());
    }
    CsrMatrix::from_raw(n, n, pointers, columns, vec![1.0; 3 * n - 2]).unwrap()
}

#[test]
fn arrowhead_fill_rejection_reports_exact_requirement() {
    let n = 100_000_usize;
    let a = arrowhead(n);
    let function = SparseFunctionPlan::new(
        a.view(),
        MatrixFunction::Exp,
        &SparseFunctionOptions::new().with_max_entries(4 * n),
    )
    .unwrap_err();
    let inverse = SparsePowerPlan::new(
        a.view(),
        -1,
        &SparsePowerOptions::new().with_max_entries(4 * n),
    )
    .unwrap_err();
    for error in [function, inverse] {
        if let Some(square) = n.checked_mul(n) {
            assert!(
                matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { required, limit, .. } if required == square && limit == 4*n)
            );
        } else {
            assert_eq!(error.kind(), SciOxideErrorKind::SizeOverflow);
        }
    }
    let a = arrowhead(8);
    let direct = scioxide_sparse::SparseProductPlan::with_options(
        &a,
        &a,
        SparseProductPlanOptions::new().with_max_output_entries(32),
    )
    .unwrap_err();
    let power = SparsePowerPlan::new(a.view(), 2, &powers(32)).unwrap_err();
    for error in [&direct, &power] {
        assert!(
            matches!(error.view(), scioxide_core::SciOxideErrorView::ResourceLimit { resource, required: 64, limit: 32 } if resource == scioxide_core::quantity!("sparse_entries"))
        );
    }
    assert!(matches!(
        power.view(),
        scioxide_core::SciOxideErrorView::ResourceLimit {
            required: 64,
            limit: 32,
            ..
        }
    ));
}

#[test]
fn powers_use_only_the_required_products() {
    let a = CsrMatrix::from_raw(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64; 2]).unwrap();
    for exponent in [
        0_i64,
        1,
        2,
        3,
        4,
        7,
        8,
        15,
        31,
        64,
        i64::MAX,
        -1,
        -2,
        -3,
        -4,
        -15,
        i64::MIN,
    ] {
        let magnitude = exponent.unsigned_abs();
        let products = if magnitude == 0 {
            0
        } else {
            (63 - magnitude.leading_zeros() + magnitude.count_ones() - 1) as usize
        };
        let plan = SparsePowerPlan::new(a.view(), exponent, &powers(2)).unwrap();
        let result = plan.evaluate(a.view()).unwrap();
        assert_eq!(
            result.report().work(),
            products + if exponent < 0 { 4 } else { 0 },
            "{exponent}"
        );
        assert_eq!(result.value().values(), &[1.0, 1.0]);
    }
    // A^1 does not require an identity pattern or its fill allowance.
    let zero = CsrMatrix::<f64>::from_raw(2, 2, vec![0; 3], vec![], vec![]).unwrap();
    let plan = SparsePowerPlan::new(zero.view(), 1, &powers(0)).unwrap();
    assert_eq!(plan.evaluate(zero.view()).unwrap().report().work(), 0);
    assert_eq!(plan.structural_fill(), 0);
    assert!(matches!(
        SparsePowerPlan::new(zero.view(), 0, &powers(1))
            .unwrap_err()
            .view(),
        scioxide_core::SciOxideErrorView::ResourceLimit {
            required: 2,
            limit: 1,
            ..
        }
    ));
}

#[test]
fn nonfinite_exponents_are_rejected_at_every_public_boundary() {
    for n in [0, 1] {
        let dense = Matrix::<f64>::zeros(n, n).unwrap();
        let sparse = CsrMatrix::<f64>::from_raw(n, n, vec![0; n + 1], vec![], vec![]).unwrap();
        for exponent in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let function = MatrixFunction::Power(exponent);
            assert_eq!(
                MatrixFunctionPlan::new(&dense, function, &MatrixFunctionOptions::new())
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::DomainError
            );
            assert_eq!(
                SparseFunctionPlan::new(sparse.view(), function, &functions(0))
                    .unwrap_err()
                    .kind(),
                SciOxideErrorKind::DomainError
            );
        }
    }
    let a = Matrix::from_vec(1, 1, vec![1.0_f64]).unwrap();
    let operator = a.view().try_finite().unwrap();
    let plan = scioxide_linalg::FunctionAction::default()
        .with_restart(Budget::ONE)
        .with_projection(Budget::ONE);
    let mut workspace = plan.workspace(&operator, 1).unwrap();
    for exponent in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        // A zero RHS must not bypass function validation.
        let mut output = [7.0];
        let error = plan
            .apply_into(
                &operator,
                MatrixFunction::Power(exponent),
                scioxide_core::BlockVectorView::new(&[0.0], 1, 1).unwrap(),
                scioxide_core::BlockVectorViewMut::new(&mut output, 1, 1).unwrap(),
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::DomainError);
        assert_eq!(output, [7.0]);
    }
}

#[test]
fn invalid_sparse_inputs_are_preflight_rejections() {
    let a = diagonal();
    let mut nan = diagonal();
    assert_eq!(
        nan.set_value(1, f64::NAN).unwrap_err().kind(),
        SciOxideErrorKind::NonFinite
    );
    let changed = CsrMatrix::from_raw(3, 3, vec![0, 1, 2, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
    let power = SparsePowerPlan::new(a.view(), 2, &powers(3)).unwrap();
    let function = SparseFunctionPlan::new(a.view(), MatrixFunction::Exp, &functions(9)).unwrap();
    {
        let (input, kind) = (&changed, SciOxideErrorKind::InvalidInput);
        let mut workspace = power.workspace(()).unwrap();
        let mut output = power.create_output().unwrap();
        let error = power
            .evaluate_into(
                input.view(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), kind);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::PreflightRejected
        );
        let mut workspace = function.workspace(()).unwrap();
        let mut output = function.create_output::<f64>().unwrap();
        let error = function
            .evaluate_into(
                input.view(),
                &mut output,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap_err();
        assert_eq!(error.kind(), kind);
        assert_eq!(
            workspace.last_attempt().unwrap().outcome(),
            AttemptOutcome::PreflightRejected
        );
    }
}

#[test]
fn sparse_inverse_forwards_the_selected_accuracy_to_lu() {
    use scioxide_sparse::{
        SparseLu, SparseLuLimits, SparseLuSymbolic, SparseOperation, SparseOrdering,
        SparseSolveOptions,
    };
    let a = CsrMatrix::from_raw(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![4., 0.1, 0.2, 0.3, 3., 0.4, 0.5, 0.6, 2.],
    )
    .unwrap();
    let mut previous = None;
    for accuracy in [
        scioxide_core::Accuracy::Standard,
        scioxide_core::Accuracy::Exact,
    ] {
        let options = powers(9).with_execution_options(
            scioxide_linalg::LinalgExecutionOptions::new().with_accuracy(accuracy),
        );
        let inverse = SparsePowerPlan::new(a.view(), -1, &options)
            .unwrap()
            .evaluate(a.view())
            .unwrap();
        let symbolic = SparseLuSymbolic::analyze(
            a.structural_pattern(),
            SparseOrdering::FillReducing,
            SparseLuLimits::new(9, usize::MAX),
        )
        .unwrap();
        let solve_options = SparseSolveOptions::new().with_accuracy(accuracy);
        let factor = SparseLu::factor_scoped(
            symbolic,
            a.view(),
            solve_options,
            ExecutionContext::sequential(),
        )
        .unwrap();
        for col in 0..3 {
            let mut rhs = [0.; 3];
            rhs[col] = 1.;
            let mut expected = [0.; 3];
            factor
                .solve_into(
                    &rhs,
                    SparseOperation::Normal,
                    solve_options,
                    &mut expected,
                    &mut factor.workspace(1).unwrap(),
                    ExecutionContext::sequential(),
                )
                .unwrap();
            for (row, column, value) in inverse.value().view().iter().filter(|&(_, c, _)| c == col)
            {
                assert_eq!(column, col);
                assert_eq!(value, expected[row]);
            }
        }
        let values = inverse.value().values().to_vec();
        if let Some(previous) = previous {
            assert_ne!(values, previous);
        }
        previous = Some(values);
    }
}
