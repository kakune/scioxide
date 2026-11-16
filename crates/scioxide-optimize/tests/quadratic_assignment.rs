#![allow(clippy::unwrap_used, clippy::too_many_arguments)]
#[cfg(feature = "std")]
use scioxide_core::ThreadExecutor;
use scioxide_core::{
    AttemptOutcome, Budget, CancellationToken, ExecutionContext, ExecutionPolicy, PermutationView,
    ProblemToken, SciOxideErrorView, SolveBudgets, SolveIdentity, SolveState,
};
#[cfg(not(feature = "std"))]
#[path = "support/executor.rs"]
mod executor;
#[cfg(not(feature = "std"))]
use executor::PartitionExecutor;
use scioxide_linalg::MatrixView;
use scioxide_optimize::{
    OptimizeTermination, QapDirection, QapMethod, QapOptions, QapPlan, QapProblemView, QapResult,
};

fn parent_identity(token: u128) -> SolveIdentity {
    SolveIdentity::new(
        ProblemToken::new(token),
        scioxide_core::algorithm_id!("optimize", "qap-parent", 1),
        0,
        0,
    )
}
use scioxide_sparse::{DuplicatePolicy, ExplicitZeroPolicy, SparseBuilder};

fn plan(a: &[f64], b: &[f64], n: usize, fixed: &[(usize, usize)]) -> QapPlan<f64> {
    QapProblemView::new(
        MatrixView::from_slice(n, n, a).unwrap(),
        MatrixView::from_slice(n, n, b).unwrap(),
        fixed,
    )
    .unwrap()
    .prepare()
    .unwrap()
}
fn brute(a: &[f64], b: &[f64], n: usize, fixed: &[(usize, usize)], maximize: bool) -> f64 {
    fn walk(
        a: &[f64],
        b: &[f64],
        n: usize,
        fixed: &[(usize, usize)],
        maximize: bool,
        p: &mut [usize],
        depth: usize,
        best: &mut f64,
    ) {
        if depth == n {
            if fixed.iter().any(|&(i, j)| p[i] != j) {
                return;
            }
            let mut v = 0.0;
            for i in 0..n {
                for j in 0..n {
                    v += a[i * n + j] * b[p[i] * n + p[j]];
                }
            }
            if (maximize && v > *best) || (!maximize && v < *best) {
                *best = v;
            }
            return;
        }
        for value in 0..n {
            if !p[..depth].contains(&value) {
                p[depth] = value;
                walk(a, b, n, fixed, maximize, p, depth + 1, best);
            }
        }
    }
    let mut best = if maximize {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    walk(a, b, n, fixed, maximize, &mut vec![0; n], 0, &mut best);
    best
}
#[test]
fn asymmetric_fixed_and_objective_reconstruction() {
    let a = [0., 3., 1., 8., 0., 2., 4., 6., 0.];
    let b = [0., 5., 7., 2., 0., 9., 3., 1., 0.];
    let p = plan(&a, &b, 3, &[(1, 2)]);
    for direction in [QapDirection::Minimize, QapDirection::Maximize] {
        for method in [QapMethod::Faq, QapMethod::TwoOpt, QapMethod::FaqThenTwoOpt] {
            let options = QapOptions::default()
                .with_method(method)
                .with_max_workspace_bytes(1_000_000)
                .with_restarts(4)
                .with_seed(11);
            let result = p.solve_with_options(direction, None, &options).unwrap();
            let perm = result.permutation().unwrap();
            assert_eq!(perm[1], 2);
            let mut scratch = [0_u8; 1];
            let value = p
                .objective(PermutationView::new(perm, &mut scratch).unwrap())
                .unwrap();
            assert_eq!(value, result.objective().unwrap());
            let optimum = brute(&a, &b, 3, &[(1, 2)], direction == QapDirection::Maximize);
            if direction == QapDirection::Minimize {
                assert!(value >= optimum - 1e-10);
            } else {
                assert!(value <= optimum + 1e-10);
            }
            let again = p.solve_with_options(direction, None, &options).unwrap();
            assert_eq!(perm, again.permutation().unwrap());
        }
    }
}
#[test]
fn empty_fully_fixed_and_bad_starts() {
    let empty = plan(&[], &[], 0, &[]);
    let options = QapOptions::default()
        .with_method(QapMethod::FaqThenTwoOpt)
        .with_max_workspace_bytes(100_000);
    assert_eq!(
        empty
            .solve_with_options(QapDirection::Minimize, None, &options)
            .unwrap()
            .objective(),
        Some(0.)
    );
    let p = plan(&[1., 2., 3., 4.], &[5., 6., 7., 8.], 2, &[(0, 1), (1, 0)]);
    assert_eq!(
        p.solve_with_options(QapDirection::Minimize, None, &options)
            .unwrap()
            .permutation()
            .unwrap(),
        &[1, 0]
    );
    assert!(
        p.solve_with_options(
            QapDirection::Minimize,
            {
                let mut scratch = [0_u8; 1];
                Some(PermutationView::new(&[0, 1], &mut scratch).unwrap())
            },
            &options
        )
        .is_err()
    );
    assert!(
        QapProblemView::new(
            MatrixView::from_slice(2, 2, &[0., 1., 2., 3.]).unwrap(),
            MatrixView::from_slice(2, 2, &[0., 1., 2., 3.]).unwrap(),
            &[(0, 0), (0, 1)]
        )
        .is_err()
    );
}

#[test]
fn sparse_and_f32_match_dense_objective() {
    let a = [0.0_f32, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0];
    let b = [0.0_f32, 0.0, 7.0, 5.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let dense = QapProblemView::new(
        MatrixView::from_slice(3, 3, &a).unwrap(),
        MatrixView::from_slice(3, 3, &b).unwrap(),
        &[],
    )
    .unwrap()
    .prepare()
    .unwrap();
    let mut ab = SparseBuilder::new(3, 3);
    let mut bb = SparseBuilder::new(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            if a[i * 3 + j] != 0.0 {
                ab.push(i, j, a[i * 3 + j]).unwrap();
            }
            if b[i * 3 + j] != 0.0 {
                bb.push(i, j, b[i * 3 + j]).unwrap();
            }
        }
    }
    let asparse = ab
        .finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Drop)
        .unwrap();
    let bsparse = bb
        .finish_coo(DuplicatePolicy::Reject, ExplicitZeroPolicy::Drop)
        .unwrap();
    let sparse =
        QapPlan::from_sparse_with_limit(asparse.view(), bsparse.view(), &[], 1_000_000).unwrap();
    let options = QapOptions::default()
        .with_method(QapMethod::FaqThenTwoOpt)
        .with_max_workspace_bytes(1_000_000)
        .with_seed(71)
        .with_restarts(3);
    let d = dense
        .solve_with_options(QapDirection::Maximize, None, &options)
        .unwrap();
    let s = sparse
        .solve_with_options(QapDirection::Maximize, None, &options)
        .unwrap();
    assert!((d.objective().unwrap() - s.objective().unwrap()).abs() <= 1e-10);
    assert_eq!(d.permutation().unwrap(), s.permutation().unwrap());
    assert!(QapPlan::from_sparse_with_limit(asparse.view(), bsparse.view(), &[], 1).is_err());
    assert!(
        QapProblemView::new(
            MatrixView::from_slice(3, 3, &a).unwrap(),
            MatrixView::from_slice(3, 3, &b).unwrap(),
            &[]
        )
        .unwrap()
        .prepare_with_limit(1)
        .is_err()
    );
}

#[test]
fn prepared_context_budget_and_transaction() {
    let a = [0., 3., 1., 8., 0., 2., 4., 6., 0.];
    let b = [0., 5., 7., 2., 0., 9., 3., 1., 0.];
    let p = plan(&a, &b, 3, &[]);
    let options = QapOptions::default()
        .with_method(QapMethod::FaqThenTwoOpt)
        .with_max_workspace_bytes(1_000_000);
    let mut output = QapResult::empty(3).unwrap();
    let mut work = p.workspace(&options).unwrap();
    // A cancelled scope is rejected in preflight and leaves the output unchanged.
    let token = CancellationToken::new();
    token.cancel();
    let error = p
        .solve_into(
            QapDirection::Minimize,
            None,
            &options,
            &mut output,
            &mut work,
            ExecutionContext::sequential().with_cancellation(&token),
        )
        .unwrap_err();
    assert!(matches!(
        error.view(),
        SciOxideErrorView::ExecutionStopped { .. }
    ));
    assert!(output.objective().is_none());
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::PreflightRejected
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            parent_identity(675),
            &mut budgets,
        )
        .unwrap();
    p.solve_into(
        QapDirection::Minimize,
        None,
        &options,
        &mut output,
        &mut work,
        &mut context,
    )
    .unwrap();
    assert_eq!(output.termination(), OptimizeTermination::Converged);
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::Succeeded
    );
    let previous = output.objective();
    let mut scratch = [0_u8; 1];
    let bad = PermutationView::new(&[1, 0], &mut scratch).unwrap();
    assert!(
        p.solve_into(
            QapDirection::Minimize,
            Some(bad),
            &options,
            &mut output,
            &mut work,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output.objective(), previous);
}

#[test]
fn executor_restarts_replay_sequential_result() {
    let a = [0., 3., 1., 8., 0., 2., 4., 6., 0.];
    let b = [0., 5., 7., 2., 0., 9., 3., 1., 0.];
    let p = plan(&a, &b, 3, &[(1, 2)]);
    let options = QapOptions::default()
        .with_method(QapMethod::FaqThenTwoOpt)
        .with_max_workspace_bytes(1_000_000)
        .with_seed(37)
        .with_restarts(4);
    let reference = p
        .solve_with_options(QapDirection::Minimize, None, &options)
        .unwrap();
    let options = options.with_execution(ExecutionPolicy::Parallel(
        core::num::NonZeroUsize::new(2).unwrap(),
    ));
    let mut work = p.workspace(&options).unwrap();
    let mut output = QapResult::empty(3).unwrap();
    #[cfg(feature = "std")]
    let pool = ThreadExecutor::new(core::num::NonZeroUsize::new(2).unwrap());
    #[cfg(not(feature = "std"))]
    let pool = PartitionExecutor(core::num::NonZeroUsize::new(2).unwrap());
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::new(&pool),
            parent_identity(675),
            &mut budgets,
        )
        .unwrap();
    p.solve_into(
        QapDirection::Minimize,
        None,
        &options,
        &mut output,
        &mut work,
        &mut context,
    )
    .unwrap();
    assert_eq!(output.objective(), reference.objective());
    assert_eq!(
        output.permutation().unwrap(),
        reference.permutation().unwrap()
    );
    assert_eq!(output.fw_iterations(), reference.fw_iterations());
    assert_eq!(output.swap_checks(), reference.swap_checks());
    assert!(context.counts().nested().iterations() >= reference.fw_iterations());
}

#[test]
fn refinement_never_worsens_and_is_swap_stationary() {
    let a = [
        0., 5., 8., 6., 5., 0., 5., 1., 8., 5., 0., 2., 6., 1., 2., 0.,
    ];
    let b = [
        0., 1., 8., 4., 1., 0., 5., 2., 8., 5., 0., 5., 4., 2., 5., 0.,
    ];
    let p = plan(&a, &b, 4, &[]);
    let faq = QapOptions::default()
        .with_method(QapMethod::Faq)
        .with_max_workspace_bytes(1_000_000);
    let refined = QapOptions::default()
        .with_method(QapMethod::FaqThenTwoOpt)
        .with_max_workspace_bytes(1_000_000);
    let baseline = p
        .solve_with_options(QapDirection::Minimize, None, &faq)
        .unwrap();
    let final_result = p
        .solve_with_options(QapDirection::Minimize, None, &refined)
        .unwrap();
    assert!(final_result.objective().unwrap() <= baseline.objective().unwrap());
    let current = final_result.objective().unwrap();
    let mut permutation = final_result.permutation().unwrap().to_vec();
    let mut scratch = [0_u8; 1];
    for i in 0..4 {
        for j in i + 1..4 {
            permutation.swap(i, j);
            let candidate = p
                .objective(PermutationView::new(&permutation, &mut scratch).unwrap())
                .unwrap();
            assert!(candidate >= current);
            permutation.swap(i, j);
        }
    }
    assert!(current >= brute(&a, &b, 4, &[], false));
}

#[test]
fn swap_check_limit_reports_iteration_limit_with_a_checked_permutation() {
    let a = [0., 3., 1., 8., 0., 2., 4., 6., 0.];
    let b = [0., 5., 7., 2., 0., 9., 3., 1., 0.];
    let p = plan(&a, &b, 3, &[]);
    let options = QapOptions::default()
        .with_method(QapMethod::TwoOpt)
        .with_max_workspace_bytes(1_000_000)
        .with_max_swap_checks(Budget::ONE)
        .with_restarts(2);
    let mut output = QapResult::empty(3).unwrap();
    let mut work = p.workspace(&options).unwrap();
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(
            ExecutionContext::sequential(),
            parent_identity(676),
            &mut budgets,
        )
        .unwrap();
    p.solve_into(
        QapDirection::Minimize,
        None,
        &options,
        &mut output,
        &mut work,
        &mut context,
    )
    .unwrap();
    assert_eq!(output.termination(), OptimizeTermination::IterationLimit);
    assert_eq!(
        work.last_attempt().unwrap().outcome(),
        AttemptOutcome::Stopped
    );
    let permutation = output.permutation().unwrap();
    let mut scratch = [0_u8; 1];
    assert_eq!(
        output.objective(),
        Some(
            p.objective(PermutationView::new(permutation, &mut scratch).unwrap())
                .unwrap()
        )
    );
    // One admitted swap check per restart, charged as nested parent work.
    assert_eq!(context.counts().nested().proposals(), 3);
    assert_eq!(output.swap_checks(), 3);
}

#[test]
fn payload_caps_are_resources_in_preparation_and_execution() {
    let matrix = [1.0_f64, 0.0, 0.0, 1.0];
    let view = QapProblemView::new(
        MatrixView::from_slice(2, 2, &matrix).unwrap(),
        MatrixView::from_slice(2, 2, &matrix).unwrap(),
        &[],
    )
    .unwrap();
    assert!(matches!(
        view.prepare_with_limit(1).err().unwrap().view(),
        scioxide_core::SciOxideErrorView::ResourceLimit { limit: 1, .. }
    ));
    let p = plan(&matrix, &matrix, 2, &[]);
    let options = QapOptions::default()
        .with_method(QapMethod::Faq)
        .with_max_workspace_bytes(1);
    assert!(matches!(
        p.workspace(&options).err().unwrap().view(),
        scioxide_core::SciOxideErrorView::ResourceLimit { limit: 1, .. }
    ));
    let options = QapOptions::default()
        .with_method(QapMethod::Faq)
        .with_max_workspace_bytes(0);
    assert!(matches!(
        p.solve_with_options(QapDirection::Minimize, None, &options)
            .err()
            .unwrap()
            .view(),
        scioxide_core::SciOxideErrorView::InvalidInput { .. }
    ));
}

fn allocation_free_faq<S: scioxide_core::RealScalar>() {
    for n in [4, 40] {
        let a: Vec<S> = (0..n * n)
            .map(|i| S::try_from_f64_lossy(((i * 7 + 3) % 19) as f64 / 19.0).unwrap())
            .collect();
        let b: Vec<S> = (0..n * n)
            .map(|i| S::try_from_f64_lossy(((i * 3 + 2) % 13) as f64 / 13.0).unwrap())
            .collect();
        for fixed in [&[][..], &[(1, n - 1)][..]] {
            let p = QapProblemView::new(
                MatrixView::from_slice(n, n, &a).unwrap(),
                MatrixView::from_slice(n, n, &b).unwrap(),
                fixed,
            )
            .unwrap()
            .prepare()
            .unwrap();
            for method in [QapMethod::Faq, QapMethod::FaqThenTwoOpt] {
                let options = QapOptions::default()
                    .with_method(method)
                    .with_max_workspace_bytes(p.workspace_bytes().unwrap())
                    .with_restarts(2);
                let mut work = p.workspace(&options).unwrap();
                let mut output = QapResult::empty(n).unwrap();
                for _ in 0..2 {
                    {
                        p.solve_into(
                            QapDirection::Minimize,
                            None,
                            &options,
                            &mut output,
                            &mut work,
                            ExecutionContext::sequential_native(),
                        )
                        .unwrap();
                    };
                    assert!(output.objective().unwrap().is_finite());
                    assert!(output.relaxation_objective().unwrap().is_finite());
                    assert!(output.fw_gap().unwrap().is_finite());
                }
            }
        }
    }
}

#[test]
fn first_and_repeated_faq_calls_execute_both_widths() {
    allocation_free_faq::<f32>();
    allocation_free_faq::<f64>();
}

#[test]
fn one_free_node_has_zero_gap_and_an_exact_relaxed_objective() {
    for p in [
        plan(&[2.], &[3.], 1, &[]),
        plan(&[1., 2., 3., 4.], &[5., 6., 7., 8.], 2, &[(1, 0)]),
    ] {
        for direction in [QapDirection::Minimize, QapDirection::Maximize] {
            let options = QapOptions::default()
                .with_method(QapMethod::Faq)
                .with_max_workspace_bytes(usize::MAX);
            let result = p.solve_with_options(direction, None, &options).unwrap();
            assert_eq!(result.termination(), OptimizeTermination::Converged);
            assert_eq!(result.fw_iterations(), 1);
            assert_eq!(result.fw_gap(), Some(0.));
            assert_eq!(result.relaxation_objective(), result.objective());
        }
    }
}

#[test]
fn kernel_and_seed_overflow_preserve_output_and_workspace_is_reusable() {
    let good = plan(&[1., 2., 3., 4.], &[4., 3., 2., 1.], 2, &[]);
    let bad = plan(&[f64::MAX; 4], &[2.; 4], 2, &[]);
    let options = QapOptions::default()
        .with_method(QapMethod::Faq)
        .with_max_workspace_bytes(usize::MAX);
    let mut work = good.workspace(&options).unwrap();
    let mut output = QapResult::empty(2).unwrap();
    good.solve_into(
        QapDirection::Minimize,
        None,
        &options,
        &mut output,
        &mut work,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let expected = output.objective();
    let permutation = output.permutation().unwrap().to_vec();
    let error = bad
        .solve_into(
            QapDirection::Minimize,
            None,
            &options,
            &mut output,
            &mut work,
            ExecutionContext::sequential_native(),
        )
        .unwrap_err();
    assert!(
        matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(output.objective(), expected);
    assert_eq!(output.permutation().unwrap(), permutation);
    good.solve_into(
        QapDirection::Minimize,
        None,
        &options,
        &mut output,
        &mut work,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    assert_eq!(output.objective(), expected);

    let seeded = plan(&[f64::MAX; 4], &[2.; 4], 2, &[(0, 1)]);
    let mut work = seeded.workspace(&options).unwrap();
    assert!(
        seeded
            .solve_into(
                QapDirection::Minimize,
                None,
                &options,
                &mut output,
                &mut work,
                ExecutionContext::sequential()
            )
            .is_err()
    );
    assert_eq!(output.objective(), expected);
    assert_eq!(output.permutation().unwrap(), permutation);
}

#[test]
fn block_storage_is_admitted_at_the_reported_payload_boundary() {
    let a = vec![1.0; 40 * 40];
    let view = QapProblemView::new(
        MatrixView::from_slice(40, 40, &a).unwrap(),
        MatrixView::from_slice(40, 40, &a).unwrap(),
        &[(7, 13)],
    )
    .unwrap();
    let p = view.prepare().unwrap();
    let bytes = p.workspace_bytes().unwrap();
    assert!(view.prepare_with_limit(bytes).is_ok());
    assert!(view.prepare_with_limit(bytes - 1).is_err());
    let options = QapOptions::default()
        .with_method(QapMethod::Faq)
        .with_max_workspace_bytes(bytes);
    let mut work = p.workspace(&options).unwrap();
    let mut output = QapResult::empty(40).unwrap();
    p.solve_into(
        QapDirection::Minimize,
        None,
        &options,
        &mut output,
        &mut work,
        ExecutionContext::sequential(),
    )
    .unwrap();
    let too_small = QapOptions::default()
        .with_method(QapMethod::Faq)
        .with_max_workspace_bytes(bytes - 1);
    assert!(p.workspace(&too_small).is_err());
    let expected = output.objective();
    assert!(
        p.solve_into(
            QapDirection::Minimize,
            None,
            &too_small,
            &mut output,
            &mut work,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert_eq!(output.objective(), expected);
}

#[test]
fn defaults_match_explicit_heuristic_settings() {
    let a = [0.0_f64, 3.0, 1.0, 8.0, 0.0, 2.0, 4.0, 6.0, 0.0];
    let b = [0.0, 5.0, 7.0, 2.0, 0.0, 9.0, 3.0, 1.0, 0.0];
    let p = plan(&a, &b, 3, &[]);
    let options = QapOptions::default()
        .with_method(QapMethod::FaqThenTwoOpt)
        .with_max_workspace_bytes(usize::MAX)
        .with_max_fw_iterations(Budget::try_from(30).unwrap())
        .with_max_swap_checks(Budget::try_from(100_000).unwrap())
        .with_step_tolerance(scioxide_core::StepTolerance::new(0.03, 0.0).unwrap())
        .with_restarts(0)
        .with_seed(42)
        .with_execution(ExecutionPolicy::Sequential);
    for direction in [QapDirection::Minimize, QapDirection::Maximize] {
        let default = p.solve(direction, None).unwrap();
        let explicit = p.solve_with_options(direction, None, &options).unwrap();
        assert_eq!(default.permutation(), explicit.permutation());
        assert_eq!(
            default.objective().unwrap().to_bits(),
            explicit.objective().unwrap().to_bits()
        );
        assert_eq!(default.termination(), explicit.termination());
        assert_eq!(default.fw_iterations(), explicit.fw_iterations());
        assert_eq!(default.swap_checks(), explicit.swap_checks());
    }
}
