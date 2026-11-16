//! Outer spectral budgets and independent projected dense iteration accounting.
use super::*;
use scioxide_core::{AttemptOutcome, AttemptRecord, ResidualTolerance};
use scioxide_linalg::*;

fn execution(iterations: usize) -> LinalgExecutionOptions {
    LinalgExecutionOptions::new().with_iteration_budget(Budget::try_from(iterations).unwrap())
}

fn counts(attempt: Option<&AttemptRecord<()>>, iterations: usize, outcome: AttemptOutcome) {
    let attempt = attempt.unwrap();
    assert_eq!(attempt.outcome(), outcome);
    assert_eq!(attempt.counts().own().iterations(), iterations);
    assert!(attempt.counts().nested().iterations() > 0);
}

fn close<S: NumericScalar<Real = f64>>(a: &[S], b: &[S]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!((a - b).abs() <= 1e-13 * (1. + a.abs().max(b.abs())));
    }
}

fn same_vectors<S: NumericScalar<Real = f64>>(a: MatrixView<'_, S>, b: MatrixView<'_, S>) {
    assert_eq!((a.rows(), a.cols()), (b.rows(), b.cols()));
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            close(&[a.get(i, j).unwrap()], &[b.get(i, j).unwrap()]);
        }
    }
}

#[test]
fn explicit_hundred_spectrum_cycles_match_default_and_reuse_storage() {
    let n = 60;
    let a = Clustered::new(n);
    let initial = vec![1.; n];
    for structure in [OperatorStructure::Hermitian, OperatorStructure::General] {
        let options = PartialSpectrumSolver::new(2)
            .with_structure(structure)
            .with_basis(Budget::try_from(12).unwrap());
        let default = options;
        let explicit = options.with_execution_options(execution(100));
        let mut work = default.workspace(&a).unwrap();
        let mut expected = PartialSpectrumBuffer::new(n, 2).unwrap();
        let mut actual = PartialSpectrumBuffer::new(n, 2).unwrap();
        a.calls.set(0);
        let baseline = default
            .solve_into(
                &a,
                initial.as_slice(),
                &mut expected,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(baseline.termination(), SpectralTermination::Converged);
        assert!(baseline.cycles() > 1);
        assert_eq!(baseline.applications(), a.calls.get());
        counts(
            work.last_attempt(),
            baseline.cycles(),
            AttemptOutcome::Succeeded,
        );
        a.calls.set(0);
        let report = explicit
            .solve_into(
                &a,
                initial.as_slice(),
                &mut actual,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), baseline.termination());
        assert_eq!(report.cycles(), baseline.cycles());
        assert_eq!(report.applications(), baseline.applications());
        assert_eq!(report.applications(), a.calls.get());
        counts(
            work.last_attempt(),
            report.cycles(),
            AttemptOutcome::Succeeded,
        );
        close(actual.eigenvalues(), expected.eigenvalues());
        close(actual.residuals(), expected.residuals());
        same_vectors(actual.eigenvectors(), expected.eigenvectors());
    }
}

#[test]
fn explicit_hundred_svd_cycles_match_default_and_reuse_storage() {
    let n = 60;
    let a = Clustered::new(n);
    let initial = vec![1.; n];
    let options = PartialSvdSolver::new(2).with_basis(Budget::try_from(12).unwrap());
    let default = options;
    let explicit = options.with_execution_options(execution(100));
    let mut work = default.workspace(&a).unwrap();
    let mut expected = PartialSvdBuffer::new(n, n, 2).unwrap();
    let mut actual = PartialSvdBuffer::new(n, n, 2).unwrap();
    let baseline = default
        .solve_into(
            &a,
            &initial,
            &mut expected,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(baseline.termination(), SpectralTermination::Converged);
    assert!(baseline.cycles() > 1);
    assert_eq!(baseline.applications(), a.calls.get());
    counts(
        work.last_attempt(),
        baseline.cycles(),
        AttemptOutcome::Succeeded,
    );
    a.calls.set(0);
    let report = explicit
        .solve_into(
            &a,
            &initial,
            &mut actual,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), baseline.termination());
    assert_eq!(report.cycles(), baseline.cycles());
    assert_eq!(report.applications(), baseline.applications());
    assert_eq!(report.applications(), a.calls.get());
    counts(
        work.last_attempt(),
        report.cycles(),
        AttemptOutcome::Succeeded,
    );
    close(actual.singular_values(), expected.singular_values());
    for (a, b) in actual.residuals().iter().zip(expected.residuals()) {
        close(a, b);
    }
    same_vectors(actual.left_vectors(), expected.left_vectors());
    same_vectors(actual.right_vectors(), expected.right_vectors());
}

#[test]
fn small_spectral_budgets_allow_complete_outer_cycles() {
    let n = 60;
    let a = Clustered::new(n);
    let initial = vec![1.; n];
    let tolerance = ResidualTolerance::new(0., 1e-30).unwrap();
    for limit in [1, 2, 4] {
        for structure in [OperatorStructure::Hermitian, OperatorStructure::General] {
            let plan = PartialSpectrumSolver::new(2)
                .with_basis(Budget::try_from(8).unwrap())
                .with_structure(structure)
                .with_tolerance(tolerance)
                .with_execution_options(execution(limit));
            let mut work = plan.workspace(&a).unwrap();
            let mut out = PartialSpectrumBuffer::new(n, 2).unwrap();
            let report = plan
                .solve_into(
                    &a,
                    initial.as_slice(),
                    &mut out,
                    &mut work,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(report.termination(), SpectralTermination::IterationLimit);
            assert_eq!(report.cycles(), limit);
            assert_eq!(out.eigenvalues().len(), 2);
            counts(work.last_attempt(), limit, AttemptOutcome::Stopped);
        }
        let plan = PartialSvdSolver::new(2)
            .with_basis(Budget::try_from(8).unwrap())
            .with_tolerance(tolerance)
            .with_execution_options(execution(limit));
        let mut work = plan.workspace(&a).unwrap();
        let mut out = PartialSvdBuffer::new(n, n, 2).unwrap();
        let report = plan
            .solve_into(
                &a,
                &initial,
                &mut out,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), SpectralTermination::IterationLimit);
        assert_eq!(report.cycles(), limit);
        assert_eq!(out.singular_values().len(), 2);
        counts(work.last_attempt(), limit, AttemptOutcome::Stopped);
    }
}

#[test]
fn small_lobpcg_budgets_allow_complete_outer_iterations() {
    let n = 60;
    let a = Clustered::new(n);
    let mass = Matrix::<f64>::identity(n).unwrap();
    let mass = mass.view().try_finite().unwrap();
    let initial: Vec<_> = (0..2)
        .flat_map(|j| (0..n).map(move |i| if j == 0 { 1. } else { (i % 7) as f64 - 3. }))
        .collect();
    for generalized in [false, true] {
        let op = if generalized {
            LobpcgOperator::new(&a).with_mass(&mass)
        } else {
            LobpcgOperator::new(&a)
        };
        for limit in [1, 2, 4] {
            let plan = LobpcgSolver::new(2)
                .with_tolerance(ResidualTolerance::new(0., 1e-30).unwrap())
                .with_execution_options(execution(limit));
            let mut work = plan.workspace(op, 0).unwrap();
            let mut out = LobpcgBuffer::new(n, 2).unwrap();
            let report = plan
                .solve_into(
                    op,
                    BlockVectorView::new(&initial, n, 2).unwrap(),
                    BlockVectorView::new(&[], n, 0).unwrap(),
                    &mut out,
                    &mut work,
                    ExecutionContext::sequential(),
                )
                .unwrap();
            assert_eq!(report.termination(), SpectralTermination::IterationLimit);
            assert_eq!(report.iterations(), limit);
            assert_eq!(out.eigenvalues().len(), 2);
            counts(work.last_attempt(), limit, AttemptOutcome::Stopped);
        }
    }
}

#[test]
fn one_outer_iteration_can_converge_after_multiple_dense_sweeps() {
    let a = Matrix::from_vec(3, 3, vec![1., 0., 0., 0., 2., 0., 0., 0., 4.]).unwrap();
    let a = a.view().try_finite().unwrap();
    for structure in [OperatorStructure::Hermitian, OperatorStructure::General] {
        let plan = PartialSpectrumSolver::new(2)
            .with_structure(structure)
            .with_execution_options(execution(1));
        let mut work = plan.workspace(&a).unwrap();
        let mut out = PartialSpectrumBuffer::new(3, 2).unwrap();
        let report = plan
            .solve_into(
                &a,
                [1.; 3].as_slice(),
                &mut out,
                &mut work,
                ExecutionContext::sequential(),
            )
            .unwrap();
        assert_eq!(report.termination(), SpectralTermination::Converged);
        assert_eq!(report.cycles(), 1);
        counts(work.last_attempt(), 1, AttemptOutcome::Succeeded);
        close(
            out.eigenvalues(),
            &[Complex::new(4., 0.), Complex::new(2., 0.)],
        );
    }
    let plan = PartialSvdSolver::new(2).with_execution_options(execution(1));
    let mut work = plan.workspace(&a).unwrap();
    let mut out = PartialSvdBuffer::new(3, 3, 2).unwrap();
    let report = plan
        .solve_into(
            &a,
            &[1.; 3],
            &mut out,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), SpectralTermination::Converged);
    assert_eq!(report.cycles(), 1);
    counts(work.last_attempt(), 1, AttemptOutcome::Succeeded);
    close(out.singular_values(), &[4., 2.]);

    let plan = LobpcgSolver::new(3).with_execution_options(execution(1));
    let mut work = plan.workspace(&a, 0).unwrap();
    let mut out = LobpcgBuffer::new(3, 3).unwrap();
    let report = plan
        .solve_into(
            &a,
            BlockVectorView::new(&[1., 1., 1., 1., -1., 0., 1., 0., -1.], 3, 3).unwrap(),
            BlockVectorView::new(&[], 3, 0).unwrap(),
            &mut out,
            &mut work,
            ExecutionContext::sequential(),
        )
        .unwrap();
    assert_eq!(report.termination(), SpectralTermination::Converged);
    assert_eq!(report.iterations(), 1);
    counts(work.last_attempt(), 1, AttemptOutcome::Succeeded);
    close(out.eigenvalues(), &[1., 2., 4.]);
}

#[test]
fn explicit_budgets_bound_action_steps_not_projected_dense_stages() {
    // Function actions: twelve Arnoldi steps fit a budget of twelve even though
    // the projected Schur and function stages need many more sweeps.
    let n = 12;
    let a = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .map(|i| {
                if i % (n + 1) == 0 {
                    0.1 * (i / n) as f64
                } else {
                    0.0
                }
            })
            .collect(),
    )
    .unwrap();
    let op = a.view().try_finite().unwrap();
    let input = vec![1.0; n];
    let twelve = Budget::try_from(n).unwrap();
    let plan = FunctionAction::default()
        .with_restart(twelve)
        .with_projection(twelve)
        .with_execution_options(execution(n));
    let (output, report) = plan
        .apply::<f64, f64>(
            &op,
            MatrixFunction::Exp,
            BlockVectorView::new(&input, n, 1).unwrap(),
        )
        .unwrap();
    assert_eq!(report.termination(), SpectralTermination::Converged);
    for (i, value) in output.iter().enumerate() {
        assert!((value - (0.1 * i as f64).exp()).abs() <= 1e-12);
    }
    // Randomized low-rank: one iteration per sketch column leaves the small
    // SVD and pivoted QR their own allowance.
    let (m, n, width) = (40, 30, 10);
    let b = Matrix::from_vec(
        m,
        n,
        (0..m * n)
            .map(|i| 1.0 / (i / n + i % n + 1) as f64)
            .collect(),
    )
    .unwrap();
    let op = b.view().try_finite().unwrap();
    let sketch = Matrix::from_vec(
        n,
        width,
        (0..n * width)
            .map(|i| ((i * 7 % 13) as f64) - 6.0)
            .collect(),
    )
    .unwrap();
    let probes = Matrix::from_vec(n, 1, (0..n).map(|i| (i % 3) as f64 - 1.0).collect()).unwrap();
    let plan = RandomizedLowRank::new(5).with_execution_options(execution(width));
    let (_, report) = plan.svd(&op, sketch.view(), probes.view()).unwrap();
    assert_eq!(report.rank(), 5);
    let (_, report) = plan
        .interpolative(&op, sketch.view(), probes.view())
        .unwrap();
    assert_eq!(report.rank(), 5);
}

#[test]
fn reported_cycles_describe_the_published_snapshot() {
    // Every cycle count reported under an application cap reproduces the
    // published (possibly partially verified) values when the run is repeated
    // with exactly that many cycles.
    let n = 60;
    let a = Clustered::new(n);
    let initial = vec![1.; n];
    let tolerance = ResidualTolerance::new(0., 1e-30).unwrap();
    let spectrum = |cycles: usize, applications: Option<usize>| {
        let mut options = PartialSpectrumSolver::new(2)
            .with_basis(Budget::try_from(8).unwrap())
            .with_tolerance(tolerance)
            .with_execution_options(execution(cycles));
        if let Some(applications) = applications {
            options = options.with_applications(Budget::try_from(applications).unwrap());
        }
        let plan = options;
        let (out, report) = plan.solve(&a, initial.as_slice()).unwrap();
        (out.eigenvalues().to_vec(), report)
    };
    let svd = |cycles: usize, applications: Option<usize>| {
        let mut options = PartialSvdSolver::new(2)
            .with_basis(Budget::try_from(8).unwrap())
            .with_tolerance(tolerance)
            .with_execution_options(execution(cycles));
        if let Some(applications) = applications {
            options = options.with_applications(Budget::try_from(applications).unwrap());
        }
        let plan = options;
        let (out, report) = plan.solve(&a, &initial).unwrap();
        (out.singular_values().to_vec(), report)
    };
    let mut checked = 0;
    for applications in 1..60 {
        let (values, report) = spectrum(6, Some(applications));
        if report.termination() == SpectralTermination::EvaluationLimit && report.cycles() > 0 {
            let full = spectrum(report.cycles(), None).0;
            assert_eq!(full[..values.len()], values[..], "{applications}");
            checked += 1;
        }
        let (values, report) = svd(6, Some(applications));
        if report.termination() == SpectralTermination::EvaluationLimit && report.cycles() > 0 {
            let full = svd(report.cycles(), None).0;
            assert_eq!(full[..values.len()], values[..], "{applications}");
            checked += 1;
        }
    }
    assert!(checked > 0);
}

#[test]
fn application_caps_stop_before_charging_an_iteration() {
    let n = 6;
    let a = Matrix::from_vec(
        n,
        n,
        (0..n * n).map(|i| ((i * 7 % 11) as f64) - 5.0).collect(),
    )
    .unwrap();
    let op = a.view().try_finite().unwrap();
    for cap in [1, 2, 3] {
        let plan = OneNormEstimator::new()
            .with_applications(Budget::try_from(cap).unwrap())
            .with_execution_options(execution(cap));
        let mut workspace = plan.workspace(&op).unwrap();
        let report = plan
            .estimate_scoped(&op, &mut workspace, ExecutionContext::sequential())
            .unwrap();
        // Equal caps stop on the application limit with one unit per application.
        assert_eq!(report.termination(), SpectralTermination::EvaluationLimit);
        assert_eq!(report.applications(), cap);
        let attempt = workspace.last_attempt().unwrap();
        assert_eq!(attempt.outcome(), AttemptOutcome::Stopped);
        assert_eq!(attempt.counts().own().iterations(), cap);
    }
}
