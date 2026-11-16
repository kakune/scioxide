#![allow(clippy::unwrap_used)]

use scioxide_core::{
    AttemptOutcome, Budget, BudgetKind, ExecutionContext, ProblemToken, SciOxideErrorKind,
    SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_linalg::{
    GeneralEigenDecomposition, GeneralEigenOptions, GeneralEigenvaluesBuffer, LinalgWorkspace,
};
use scioxide_linalg::{
    HermitianEigenDecomposition, HermitianEigenOptions, HermitianEigenWorkspaceRequest,
    HermitianEigenvaluesBuffer, Matrix,
};

#[test]
fn singleton_tridiagonal_failures_preserve_output() {
    use scioxide_linalg::EigenIndexRange;

    let options =
        HermitianEigenOptions::default().with_index_range(EigenIndexRange::new(0, 1).unwrap());
    let mut output =
        HermitianEigenDecomposition::tridiagonal_with_options(&[0.5_f64], &[], &options).unwrap();
    let original = output.clone();
    let mut workspace = LinalgWorkspace::new();
    let timed = options.with_execution_options(
        options
            .execution_options()
            .with_time_limit(core::time::Duration::from_secs(1)),
    );
    let limited = options.with_execution_options(options.execution_options().with_byte_limit(1));
    let invalid_range = options.with_index_range(EigenIndexRange::new(0, 2).unwrap());
    for (diagonal, off, options, kind) in [
        (0.0, &[][..], invalid_range, SciOxideErrorKind::InvalidInput),
        (
            0.0,
            &[0.0][..],
            options,
            SciOxideErrorKind::DimensionMismatch,
        ),
        (f64::NAN, &[][..], options, SciOxideErrorKind::NonFinite),
        (
            f64::INFINITY,
            &[][..],
            options,
            SciOxideErrorKind::NonFinite,
        ),
        (
            f64::NEG_INFINITY,
            &[][..],
            options,
            SciOxideErrorKind::NonFinite,
        ),
        (0.0, &[][..], limited, SciOxideErrorKind::ResourceLimit),
        (
            0.0,
            &[][..],
            timed,
            SciOxideErrorKind::CapabilityUnavailable,
        ),
    ] {
        let error = HermitianEigenDecomposition::tridiagonal_into(
            &[diagonal],
            off,
            &options,
            &mut output,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), kind);
        assert_eq!(output.eigenvalues(), original.eigenvalues());
        assert_eq!(output.eigenvectors(), original.eigenvectors());
        assert_eq!(
            output.report().unwrap().iterations(),
            original.report().unwrap().iterations()
        );
    }
}

#[test]
fn rectangular_selected_tridiagonal_repeated_eigenvalues_are_orthonormal() {
    let options = HermitianEigenOptions::default()
        .with_index_range(scioxide_linalg::EigenIndexRange::new(1, 4).unwrap());
    let result =
        HermitianEigenDecomposition::tridiagonal_with_options(&[2.0_f64; 5], &[0.0; 4], &options)
            .unwrap();
    for (column, &value) in result.eigenvalues().iter().enumerate() {
        assert!((value - 2.0).abs() < 1e-14);
        for previous in 0..=column {
            let dot = result
                .eigenvectors()
                .as_slice()
                .chunks_exact(3)
                .map(|row| row[column] * row[previous])
                .sum::<f64>();
            assert!((dot - if previous == column { 1.0 } else { 0.0 }).abs() < 1e-14);
        }
    }
}

#[test]
fn selected_two_sample_eigenvectors_refine_the_initial_direction() {
    let options = HermitianEigenOptions::default()
        .with_index_range(scioxide_linalg::EigenIndexRange::new(1, 2).unwrap());
    let result =
        HermitianEigenDecomposition::tridiagonal_with_options(&[0.02_f64; 2], &[0.125], &options)
            .unwrap();
    for &value in result.eigenvectors().as_slice() {
        assert!((value - core::f64::consts::FRAC_1_SQRT_2).abs() <= 2.0 * f64::EPSILON);
    }
}

fn general_matrix() -> Matrix<f64> {
    Matrix::from_vec(
        4,
        4,
        vec![
            4.0_f64, -2.0, 1.0, 3.0, 1.0, 5.0, -3.0, 2.0, 2.0, 1.0, -1.0, 4.0, -2.0, 3.0, 2.0, 1.0,
        ],
    )
    .unwrap()
}

#[test]
fn general_qr_steps_are_nested_in_a_solve_context() {
    let a = general_matrix();
    let options = GeneralEigenOptions::<f64>::new();
    let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
    let mut state = SolveState::new();
    let identity = SolveIdentity::new(
        ProblemToken::new(944),
        scioxide_core::algorithm_id!("linalg", "general_eigen", 1),
        4,
        0,
    );
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let mut workspace = LinalgWorkspace::new();
    let mut values = GeneralEigenvaluesBuffer::empty();
    GeneralEigenvaluesBuffer::compute_into(&a, &options, &mut values, &mut workspace, &mut context)
        .unwrap();
    let nested = context.counts().nested().iterations();
    assert_eq!(context.counts().own().iterations(), 0);
    assert!(nested > 1);
    let attempt = workspace.last_attempt().unwrap();
    assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
    assert_eq!(attempt.counts().own().iterations(), nested);

    let mut vectors = GeneralEigenDecomposition::empty();
    let mut vector_workspace = LinalgWorkspace::new();
    assert!(
        GeneralEigenDecomposition::decompose_into(
            &a,
            &options.with_execution_options(
                (options)
                    .execution_options()
                    .with_iteration_budget(Budget::ONE)
            ),
            &mut vectors,
            &mut vector_workspace,
            &mut context,
        )
        .is_err()
    );
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), nested + 1);
    assert_eq!(
        vector_workspace.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    assert!(vectors.eigenvalues().is_empty());
}

/// A diagonally dominant symmetric matrix and its tridiagonal part.
type Fixture = (Matrix<f64>, Vec<f64>, Vec<f64>);

fn hermitian_fixture(n: usize) -> Fixture {
    let entry = |row: usize, col: usize| {
        if row == col {
            3.0 + row as f64
        } else if row.abs_diff(col) == 1 {
            1.0
        } else {
            0.25 / (1 + row + col) as f64
        }
    };
    let matrix = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|index| entry(index / n, index % n))
            .collect(),
    )
    .unwrap();
    let diagonal = (0..n).map(|i| entry(i, i)).collect();
    let off = (1..n).map(|i| entry(i, i - 1)).collect();
    (matrix, diagonal, off)
}

#[derive(Clone, Copy, Debug)]
enum HermitianJob {
    Values,
    Vectors,
    Tridiagonal,
}

fn job_requirements(
    job: HermitianJob,
    n: usize,
    options: &HermitianEigenOptions<f64>,
) -> Result<scioxide_linalg::LinalgRequirements, scioxide_core::SciOxideError> {
    match job {
        HermitianJob::Values => HermitianEigenvaluesBuffer::<f64>::workspace_requirements::<f64>(
            &HermitianEigenWorkspaceRequest::new(n, options),
        ),
        HermitianJob::Vectors => HermitianEigenDecomposition::<f64>::workspace_requirements(
            &HermitianEigenWorkspaceRequest::new(n, options),
        ),
        HermitianJob::Tridiagonal => HermitianEigenDecomposition::<f64>::workspace_requirements(
            &scioxide_linalg::HermitianEigenWorkspaceRequest::tridiagonal(n, options),
        ),
    }
}

struct Outputs {
    values: HermitianEigenvaluesBuffer<f64>,
    vectors: HermitianEigenDecomposition<f64>,
}

impl Outputs {
    fn new() -> Self {
        Self {
            values: HermitianEigenvaluesBuffer::empty(),
            vectors: HermitianEigenDecomposition::empty(),
        }
    }
}

fn run_job<'s, 'c>(
    job: HermitianJob,
    (matrix, diagonal, off): &Fixture,
    options: &HermitianEigenOptions<f64>,
    outputs: &mut Outputs,
    workspace: &mut LinalgWorkspace<'_, f64>,
    context: impl Into<scioxide_linalg::LinalgExecutionContext<'s, 'c>>,
) -> Result<scioxide_linalg::EigenReport, scioxide_core::SciOxideError> {
    match job {
        HermitianJob::Values => HermitianEigenvaluesBuffer::compute_into(
            matrix,
            scioxide_linalg::HermitianInput::Exact,
            options,
            &mut outputs.values,
            workspace,
            context,
        ),
        HermitianJob::Vectors => HermitianEigenDecomposition::decompose_into(
            matrix,
            scioxide_linalg::HermitianInput::Exact,
            options,
            &mut outputs.vectors,
            workspace,
            context,
        ),
        HermitianJob::Tridiagonal => HermitianEigenDecomposition::tridiagonal_into(
            diagonal,
            off,
            options,
            &mut outputs.vectors,
            workspace,
            context,
        ),
    }
}

const JOBS: [HermitianJob; 3] = [
    HermitianJob::Values,
    HermitianJob::Vectors,
    HermitianJob::Tridiagonal,
];

// Run a job after reserving its public requirements.

/// The documented normwise class `c n epsilon` with the test constant `c = 100`.
fn bound(n: usize) -> f64 {
    100.0 * n as f64 * f64::EPSILON
}

/// Entry `(row, col)` of a row-major matrix.
fn at(matrix: &Matrix<f64>, row: usize, col: usize) -> f64 {
    matrix.as_slice()[row * matrix.cols() + col]
}

/// Largest `|A V - V diag(values)|` and `|V^T V - I|` entries.
fn eigenpair_errors(
    a: impl Fn(usize, usize) -> f64,
    values: &[f64],
    vectors: &Matrix<f64>,
) -> (f64, f64) {
    let (n, count) = (vectors.rows(), vectors.cols());
    let (mut residual, mut orthogonality) = (0.0_f64, 0.0_f64);
    for (col, &value) in values.iter().enumerate().take(count) {
        for row in 0..n {
            let product = (0..n).fold(0.0, |sum, k| sum + a(row, k) * at(vectors, k, col));
            residual = residual.max((product - at(vectors, row, col) * value).abs());
        }
        for other in 0..count {
            let gram = (0..n).fold(0.0, |sum, k| {
                sum + at(vectors, k, col) * at(vectors, k, other)
            });
            let identity = if col == other { 1.0 } else { 0.0 };
            orthogonality = orthogonality.max((gram - identity).abs());
        }
    }
    (residual, orthogonality)
}

#[test]
fn hermitian_requirements_report_limits_overflow_storage_and_kind() {
    use scioxide_core::SciOxideErrorKind;
    let options = HermitianEigenOptions::<f64>::default();
    let limited = options.with_execution_options((options).execution_options().with_byte_limit(1));
    let fixture = hermitian_fixture(6);
    for job in JOBS {
        assert_eq!(
            job_requirements(job, 6, &limited).unwrap_err().kind(),
            SciOxideErrorKind::ResourceLimit
        );
        assert_eq!(
            job_requirements(job, usize::MAX, &options)
                .unwrap_err()
                .kind(),
            SciOxideErrorKind::SizeOverflow
        );
        let mut outputs = Outputs::new();
        run_job(
            job,
            &fixture,
            &options,
            &mut outputs,
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap();
        let before = (
            outputs.values.clone().into_eigenvalues(),
            outputs.vectors.clone().into_parts(),
        );
        let unchanged = |outputs: &Outputs| {
            assert_eq!(outputs.values.eigenvalues(), before.0.as_slice());
            assert_eq!(outputs.vectors.eigenvalues(), before.1.0.as_slice());
            assert_eq!(outputs.vectors.eigenvectors(), &before.1.1);
        };
        let error = run_job(
            job,
            &fixture,
            &limited,
            &mut outputs,
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit);
        unchanged(&outputs);
        let mut short = [0.0_f64; 3];
        let error = run_job(
            job,
            &fixture,
            &options,
            &mut outputs,
            &mut LinalgWorkspace::from_scalar_slice(&mut short),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err();
        // Reservation errors name the public entry point.
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::DimensionMismatch { .. }
        ));
        unchanged(&outputs);
        let error = run_job(
            job,
            &fixture,
            &options.with_execution_options(
                (options)
                    .execution_options()
                    .with_iteration_budget(scioxide_core::Budget::ONE),
            ),
            &mut outputs,
            &mut LinalgWorkspace::new(),
            scioxide_core::ExecutionContext::sequential(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
        unchanged(&outputs);
    }
    let timed = options.with_execution_options(
        (options)
            .execution_options()
            .with_time_limit(core::time::Duration::from_secs(1)),
    );
    let (matrix, diagonal, off) = hermitian_fixture(4);
    for error in [
        scioxide_linalg::HermitianEigenvaluesResult::compute_with_options(
            &matrix,
            scioxide_linalg::HermitianInput::Exact,
            &timed,
        )
        .unwrap_err(),
        HermitianEigenDecomposition::decompose_with_options(
            &matrix,
            scioxide_linalg::HermitianInput::Exact,
            &timed,
        )
        .unwrap_err(),
        HermitianEigenDecomposition::tridiagonal_with_options(&diagonal, &off, &timed).unwrap_err(),
    ] {
        assert_eq!(error.kind(), SciOxideErrorKind::CapabilityUnavailable);
    }
}

#[test]
fn hermitian_solve_context_callers_see_nested_counts_and_attempts() {
    use scioxide_core::{
        AttemptOutcome, Budget, BudgetKind, ExecutionContext, ProblemToken, SolveBudgets,
        SolveIdentity, SolveState,
    };
    let identity = SolveIdentity::new(
        ProblemToken::new(944),
        scioxide_core::algorithm_id!("test", "hermitian_nested", 1),
        0,
        0,
    );
    let fixture = hermitian_fixture(8);
    for job in JOBS {
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::ONE);
        let mut context = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let options = HermitianEigenOptions::default();
        let mut outputs = Outputs::new();
        let mut workspace = LinalgWorkspace::new();
        run_job(
            job,
            &fixture,
            &options,
            &mut outputs,
            &mut workspace,
            &mut context,
        )
        .unwrap();
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::Succeeded);
        let used = attempt.counts().own().iterations();
        assert!(used > 1, "{job:?}");
        assert_eq!(context.counts().own().iterations(), 0);
        assert_eq!(context.counts().nested().iterations(), used);
        let error = run_job(
            job,
            &fixture,
            &options.with_index_range(scioxide_linalg::EigenIndexRange::new(0, 9).unwrap()),
            &mut outputs,
            &mut workspace,
            &mut context,
        )
        .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::InvalidInput);
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::PreflightRejected);
        assert!(attempt.epoch().is_none());
        assert_eq!(context.counts().nested().iterations(), used);
    }
}

#[test]
fn blocked_values_failure_keeps_published_output() {
    use core::sync::atomic::{AtomicUsize, Ordering};
    struct CancelAfter(AtomicUsize);
    impl scioxide_core::Cancellation for CancelAfter {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed) >= 20
        }
    }
    let (matrix, _, _) = hermitian_fixture(97);
    let options = HermitianEigenOptions::default();
    let mut workspace = LinalgWorkspace::new();
    let mut values = HermitianEigenvaluesBuffer::empty();
    HermitianEigenvaluesBuffer::compute_into(
        &matrix,
        scioxide_linalg::HermitianInput::Exact,
        &options,
        &mut values,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let before = values.eigenvalues().to_vec();
    let cancellation = CancelAfter(AtomicUsize::new(0));
    assert!(
        HermitianEigenvaluesBuffer::compute_into(
            &matrix,
            scioxide_linalg::HermitianInput::Exact,
            &options,
            &mut values,
            &mut workspace,
            ExecutionContext::sequential().with_cancellation(&cancellation)
        )
        .is_err()
    );
    assert!(cancellation.0.load(Ordering::Relaxed) > 20);
    assert_eq!(values.eigenvalues(), before);
    let error = HermitianEigenvaluesBuffer::compute_into(
        &matrix,
        scioxide_linalg::HermitianInput::Exact,
        &options.with_execution_options(
            (options)
                .execution_options()
                .with_iteration_budget(Budget::ONE),
        ),
        &mut values,
        &mut workspace,
        ExecutionContext::sequential(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), SciOxideErrorKind::ConvergenceFailure);
    assert_eq!(values.eigenvalues(), before);
}

#[test]
fn default_divide_conquer_budget_covers_order_512() {
    let n = 512;
    let scale = 1e-6_f64;
    let diagonal = vec![2.0 * scale; n];
    let off = vec![scale; n - 1];
    let result = HermitianEigenDecomposition::tridiagonal(&diagonal, &off).unwrap();
    for (index, &value) in result.eigenvalues().iter().enumerate() {
        let reference =
            2.0 - 2.0 * (core::f64::consts::PI * (index + 1) as f64 / (n + 1) as f64).cos();
        assert!((value / scale - reference).abs() <= 64.0 * n as f64 * f64::EPSILON);
    }
    let (residual, orthogonality) = eigenpair_errors(
        |r, c| match r.abs_diff(c) {
            0 => diagonal[r],
            1 => scale,
            _ => 0.0,
        },
        result.eigenvalues(),
        result.eigenvectors(),
    );
    assert!(residual <= bound(n) * scale);
    assert!(orthogonality <= bound(n));
}
