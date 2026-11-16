//! Prepared allocation, budget and executor contracts.
#![allow(clippy::unwrap_used)]

use scioxide_core::*;
use scioxide_linalg::{Matrix, MatrixView};
use scioxide_stats::*;

#[test]
fn robust_slope_storage_budget_accounts_for_extended_components() {
    for method in [RobustSlopeMethod::TheilSen, RobustSlopeMethod::Siegel] {
        let n = 5;
        let pairs = if method == RobustSlopeMethod::TheilSen {
            n * (n - 1) / 2
        } else {
            n
        };
        let native = if method == RobustSlopeMethod::Siegel {
            n
        } else {
            0
        };
        let bytes = (pairs + 2 * n) * core::mem::size_of::<Scaled<f64>>()
            + n * core::mem::size_of::<f32>()
            + native * core::mem::size_of::<f64>();
        assert!(
            RobustSlopeWorkspace::<f32>::new(
                n,
                method,
                StatisticsOptions::new().with_max_bytes(bytes)
            )
            .is_ok()
        );
        assert!(
            RobustSlopeWorkspace::<f32>::new(
                n,
                method,
                StatisticsOptions::new().with_max_bytes(bytes - 1)
            )
            .is_err()
        );
    }
}

#[test]
fn sparse_view_and_complex_f32_operator_domains() {
    let values = [
        Complex::new(1_f32, 1.),
        Complex::new(2., -1.),
        Complex::new(3., 0.),
    ];
    let beta = Complex::new(2., -3.);
    let y = values.map(|v| v * beta);
    let dense = Matrix::from_vec(3, 1, values.to_vec()).unwrap();
    let sparse = dense
        .to_sparse_csr(scioxide_sparse::ExplicitZeroPolicy::Preserve)
        .unwrap();
    let view = dense.view();
    for operator in [
        &dense.view().try_finite().unwrap() as &dyn AdjointOperator<Complex<f32>>,
        &sparse,
        &view.try_finite().unwrap(),
    ] {
        let options = OperatorRegressionOptions::default()
            .with_iterations(Budget::try_from(30).unwrap())
            .with_method(OperatorRegressionMethod::Lsmr)
            .with_residual(Default::default())
            .with_normal(Default::default());
        let model = PreparedOperatorRegression::new(
            operator,
            Weights::<_, Analytic>::new(&[1., 2., 0.]),
            options.with_max_bytes(usize::MAX),
        )
        .unwrap();
        let mut work = model.workspace().unwrap();
        let mut out = model.output().unwrap();
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let mut context = state
            .begin_solve(
                ExecutionContext::sequential(),
                test_identity().unwrap(),
                &mut budgets,
            )
            .unwrap();
        let report = model
            .fit_into(&y, &mut out, &mut work, &mut context)
            .unwrap();
        assert!(report.termination.is_success());
        assert!((out.coefficients().view().get(0).unwrap() - beta).abs() < 2e-5);
    }
}

#[test]
fn cancellation_preserves_prepared_outputs() {
    let a = [1., 0., 1., 1., 1., 2.];
    let y = [1., 2., 4.];
    let model = PreparedRegression::new(
        MatrixView::from_slice(3, 2, &a).unwrap(),
        Unweighted,
        Default::default(),
    )
    .unwrap();
    let mut output = model.fit(&y).unwrap();
    let before = output.coefficients().as_slice().to_vec();
    let mut workspace = model.workspace().unwrap();
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        model
            .fit_into(
                &y,
                &mut output,
                &mut workspace,
                ExecutionContext::sequential().with_cancellation(&token)
            )
            .is_err()
    );
    assert_eq!(output.coefficients().as_slice(), before);
}

#[test]
fn validation_and_partial_huber() {
    let a = [1., 0., 1., 1., 1., 2., 1., 3.];
    let y = [1., 2.2, 2.8, 4.1];
    let view = MatrixView::from_slice(4, 2, &a).unwrap();
    assert!(
        PreparedRegression::new(
            view,
            Unweighted,
            LinearRegressionOptions::default().with_max_bytes(1)
        )
        .is_err()
    );
    assert!(
        PreparedRegression::new(
            view,
            Weights::<_, Analytic>::new(&[0.; 4]),
            Default::default(),
        )
        .is_err()
    );
    assert!(
        PreparedRegression::new(
            view,
            Weights::<_, Frequency>::new(&[1., 1., 1., 0.5]),
            Default::default(),
        )
        .is_err()
    );
    assert!(
        PreparedHuber::new(
            view,
            HuberOptions::default()
                .with_iterations(Budget::ONE)
                .with_max_bytes(0)
        )
        .unwrap()
        .workspace()
        .is_err()
    );
    let options = HuberOptions::default().with_iterations(Budget::try_from(1).unwrap());
    let fit = scioxide_stats::PreparedHuber::new(view, options.with_max_bytes(usize::MAX))
        .and_then(|plan| plan.fit(&y))
        .unwrap();
    assert_eq!(fit.termination(), StatsTermination::BudgetExhausted);
    assert_eq!(fit.iterations(), 1);
    let groups = PreparedGroups::new(&[&y, &[2., 3., 4., 5.]], StatisticsOptions::new()).unwrap();
    let tiny = PostHocOptions::default().with_evaluations(Budget::try_from(1).unwrap());
    assert!(
        groups
            .pairwise_with_options(PostHocMethod::TukeyKramer, tiny)
            .is_err()
    );
}

#[test]
fn regression_batch_fits_run_under_the_caller_context() {
    struct Checks {
        count: core::sync::atomic::AtomicUsize,
        limit: usize,
    }
    impl Cancellation for Checks {
        fn is_cancelled(&self) -> bool {
            self.count
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed)
                >= self.limit
        }
    }
    let checks = |limit| Checks {
        count: core::sync::atomic::AtomicUsize::new(0),
        limit,
    };
    let design = [1., 0., 1., 1., 1., 2., 1., 3.];
    let y = [1., 2.2, 2.8, 4.1];
    let model = PreparedRegression::new(
        MatrixView::from_slice(4, 2, &design).unwrap(),
        Unweighted,
        Default::default(),
    )
    .unwrap();
    let fit = checks(usize::MAX);
    model
        .fit_into(
            &y,
            &mut model.output().unwrap(),
            &mut model.workspace().unwrap(),
            ExecutionContext::sequential().with_cancellation(&fit),
        )
        .unwrap();
    let run = |cancellation: &Checks, output: &mut [LinearRegressionFit<f64>]| {
        regression_batch_into(
            &model,
            &[&y[..]],
            StatisticsOptions::new(),
            output,
            &mut [model.output().unwrap()],
            &mut [model.workspace().unwrap()],
            ExecutionContext::sequential().with_cancellation(cancellation),
        )
    };
    let mut output = [model.output().unwrap()];
    let batch = checks(usize::MAX);
    run(&batch, &mut output).unwrap();
    let (fit, batch) = (fit.count.into_inner(), batch.count.into_inner());
    // Every check of the inner fit polls the caller's cancellation.
    assert!(batch > fit, "batch {batch} checks, fit {fit} checks");
    let reference = output[0].coefficients().as_slice().to_vec();
    // Cancellation observed inside the last fit check aborts the batch atomically.
    let mut stopped = [model.output().unwrap()];
    assert!(run(&checks(batch - 2), &mut stopped).is_err());
    assert_eq!(stopped[0].coefficients().as_slice(), &[0.0, 0.0]);
    assert_eq!(output[0].coefficients().as_slice(), &reference[..]);
}

#[cfg(feature = "std")]
#[test]
fn batch_executor_reproducibility_and_transaction() {
    use core::num::NonZeroUsize;
    let workers = NonZeroUsize::new(2).unwrap();
    let pool = ThreadExecutor::new(workers);
    let execution = ExecutionContext::new(&pool);
    let a = [1., 0., 1., 1., 1., 2., 1., 3.];
    let y = [1., 2.2, 2.8, 4.1];
    let model = PreparedRegression::new(
        MatrixView::from_slice(4, 2, &a).unwrap(),
        Unweighted,
        Default::default(),
    )
    .unwrap();
    let mut output = (0..4).map(|_| model.output().unwrap()).collect::<Vec<_>>();
    let mut stage = (0..4).map(|_| model.output().unwrap()).collect::<Vec<_>>();
    let mut work = (0..2)
        .map(|_| model.workspace().unwrap())
        .collect::<Vec<_>>();
    let options =
        StatisticsOptions::new().with_execution_policy(ExecutionPolicy::Parallel(workers));
    regression_batch_into(
        &model,
        &[&y[..]; 4],
        options,
        &mut output,
        &mut stage,
        &mut work,
        execution,
    )
    .unwrap();
    let reference = model.fit(&y).unwrap();
    for fit in &output {
        assert_eq!(
            fit.coefficients().as_slice(),
            reference.coefficients().as_slice()
        );
    }
    let bad = [f64::NAN, 2., 3., 4.];
    assert!(
        regression_batch_into(
            &model,
            &[&y, &bad, &y, &y],
            options,
            &mut output,
            &mut stage,
            &mut work,
            execution
        )
        .is_err()
    );
    for fit in &output {
        assert_eq!(
            fit.coefficients().as_slice(),
            reference.coefficients().as_slice()
        );
    }
}

fn test_identity() -> Result<scioxide_core::SolveIdentity, scioxide_core::SciOxideError> {
    Ok(scioxide_core::SolveIdentity::new(
        scioxide_core::ProblemToken::new(1),
        scioxide_core::algorithm_id!("stats-test", "parent", 1),
        0,
        0,
    ))
}
