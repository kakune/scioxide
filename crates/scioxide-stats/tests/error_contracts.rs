//! Budget errors retain required sizes independently of invalid inputs and overflow.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{SciOxideError, SciOxideErrorKind, SciOxideErrorView};
use scioxide_linalg::MatrixView;
use scioxide_stats::*;

fn resource(error: SciOxideError, expected_limit: usize) -> usize {
    assert_eq!(error.kind(), SciOxideErrorKind::ResourceLimit, "{error}");
    let SciOxideErrorView::ResourceLimit {
        required, limit, ..
    } = error.view()
    else {
        panic!("missing resource details");
    };
    assert_eq!(limit, expected_limit);
    assert!(required > limit);
    required
}
fn error<T>(result: Result<T, SciOxideError>) -> SciOxideError {
    result.err().unwrap()
}

#[test]
fn regression_family_budgets_live_in_options_and_plans_retain_them() {
    let samples = [1.0_f64, 2.0, 3.0, 4.0];
    // The Huber options budget is authoritative; nothing can widen it per call.
    let huber =
        HuberOptions::default().with_iterations(scioxide_core::Budget::try_from(20).unwrap());
    assert!(huber_location_with_options(&samples, huber).is_ok());
    assert_eq!(
        resource(
            error(huber_location_with_options(
                &samples,
                huber.with_max_bytes(1)
            )),
            1
        ),
        4 * size_of::<f64>()
    );

    // The preparation envelope bounds every later reservation of the plan.
    let design = [1., 0., 1., 1., 1., 2., 1., 3.];
    let view = MatrixView::from_slice(4, 2, &design).unwrap();
    let envelope = (4 * 2 + 2 * 2 + 4 + 2) * 32 * size_of::<f64>();
    let options = LinearRegressionOptions::default().with_max_bytes(envelope);
    assert_eq!(
        resource(
            error(PreparedRegression::new(
                view,
                Unweighted,
                options.with_max_bytes(envelope - 1)
            )),
            envelope - 1
        ),
        envelope
    );
    let plan = PreparedRegression::new(view, Unweighted, options).unwrap();
    assert_eq!(plan.options().max_bytes(), envelope);
    assert!(plan.workspace().is_ok() && plan.output().is_ok());
    let copy = plan.clone();
    let fit = plan.fit(&samples).unwrap();
    let copy = copy.fit(&samples).unwrap();
    assert_eq!(
        fit.coefficients().as_slice(),
        copy.coefficients().as_slice()
    );
    assert!((fit.coefficients().as_slice()[1] - 1.0).abs() < 1e-12);

    // Post-hoc families, group summaries and rank tests take their budget from options.
    let (a, b, c) = ([1.0_f64, 2., 3.], [2., 3., 4.], [4., 5., 7.]);
    let groups = PreparedGroups::new(&[&a, &b, &c], StatisticsOptions::new()).unwrap();
    let posthoc = PostHocOptions::default()
        .with_evaluations(scioxide_core::Budget::try_from(8_000_000).unwrap());
    let family = 3 * size_of::<PostHocComparison<f64>>();
    let tukey = PostHocMethod::TukeyKramer;
    assert!(
        groups
            .pairwise_with_options(tukey, posthoc.with_max_bytes(family))
            .is_ok()
    );
    assert_eq!(
        resource(
            error(groups.pairwise_with_options(tukey, posthoc.with_max_bytes(family - 1))),
            family - 1
        ),
        family
    );
    let controls = 2 * size_of::<PostHocComparison<f64>>();
    assert_eq!(
        resource(
            error(groups.compare_control_with_options(
                0,
                TestAlternative::TwoSided,
                posthoc.with_max_bytes(controls - 1)
            )),
            controls - 1
        ),
        controls
    );
    let limited = |bytes| StatisticsOptions::new().with_max_bytes(bytes);
    let pooled = 6 * 16 * size_of::<f64>();
    assert!(kruskal_wallis_test_with_options(&[&a, &b], limited(pooled)).is_ok());
    assert_eq!(
        resource(
            error(kruskal_wallis_test_with_options(
                &[&a, &b],
                limited(pooled - 1)
            )),
            pooled - 1
        ),
        pooled
    );
    let blocks = [1.0_f64, 2., 3., 1., 3., 2.];
    let blocks = MatrixView::from_slice(2, 3, &blocks).unwrap();
    let scratch = 3 * (2 * size_of::<f64>() + size_of::<usize>());
    assert!(friedman_test_with_options(blocks, limited(scratch)).is_ok());
    assert_eq!(
        resource(
            error(friedman_test_with_options(blocks, limited(scratch - 1))),
            scratch - 1
        ),
        scratch
    );
}

#[test]
fn family_budget_failures_share_the_resource_contract() {
    let observations = [
        SurvivalObservation::new(None, 1.0_f64, SurvivalOutcome::Event).unwrap(),
        SurvivalObservation::new(None, 2., SurvivalOutcome::Censored).unwrap(),
    ];
    resource(
        error(kaplan_meier_with_options(
            &observations,
            Default::default(),
            SurvivalOptions::new().with_max_bytes(1),
        )),
        1,
    );
    let options = ClassicalOptions::default()
        .with_max_workspace_bytes(1)
        .with_calibration(TestCalibration::Exact);
    resource(
        error(mann_whitney_test_with_options(
            &[1., 2., 3.],
            &[4., 5., 6.],
            options,
        )),
        1,
    );
    resource(
        error(ks_one_sample_test_with_options(
            &[0.1, 0.3, 0.8],
            Ok,
            options,
        )),
        1,
    );
    let data = [1., 1., 1., 2., 1., 3., 1., 4.];
    let view = MatrixView::from_slice(4, 2, &data).unwrap();
    resource(
        error(PreparedPca::with_options(
            view,
            MultivariateOptions::new().with_max_bytes(1),
        )),
        1,
    );
    resource(
        error(PreparedCca::with_options(
            view,
            view,
            MultivariateOptions::new().with_max_bytes(1),
        )),
        1,
    );
    resource(
        error(PreparedRegression::new(
            view,
            Unweighted,
            LinearRegressionOptions::default().with_max_bytes(1),
        )),
        1,
    );
    let plan = ResamplingPlan::<f64>::new(
        &[4],
        ResamplingMethod::Bootstrap {
            design: BootstrapDesign::Independent,
            method: BootstrapMethod::Percentile,
            confidence: Default::default(),
        },
        ResamplingOptions::default()
            .with_replicates(99)
            .with_batch_size(4),
    )
    .unwrap();
    let bytes = plan.workspace_bytes().unwrap();
    let method = plan.method();
    let short = ResamplingPlan::<f64>::new(
        &[4],
        method,
        (ResamplingOptions::default()
            .with_replicates(99)
            .with_batch_size(4))
        .with_max_workspace_bytes(bytes - 1),
    )
    .unwrap();
    assert_eq!(resource(error(short.workspace()), bytes - 1), bytes);
    assert!(
        ResamplingPlan::<f64>::new(
            &[4],
            method,
            (ResamplingOptions::default()
                .with_replicates(99)
                .with_batch_size(4))
            .with_max_workspace_bytes(bytes)
        )
        .unwrap()
        .workspace()
        .is_ok()
    );
}

#[test]
fn ordering_and_fit_budget_boundaries_and_overflow() {
    let x = [0.1_f64, 0.3, 0.8];
    // Cramér–von Mises only needs the sorted sample, with no tail workspace.
    let bytes = core::mem::size_of_val(&x);
    assert!(
        cramer_von_mises_statistic_with_options(
            &x,
            Ok,
            scioxide_stats::ClassicalOptions::default().with_max_workspace_bytes(bytes)
        )
        .is_ok()
    );
    assert_eq!(
        resource(
            error(cramer_von_mises_statistic_with_options(
                &x,
                Ok,
                scioxide_stats::ClassicalOptions::default().with_max_workspace_bytes(bytes - 1)
            )),
            bytes - 1
        ),
        bytes
    );
    assert!(
        FitWorkspace::<f64>::new(
            2,
            scioxide_stats::StatisticsOptions::new().with_max_bytes(16)
        )
        .is_ok()
    );
    assert_eq!(
        resource(
            error(FitWorkspace::<f64>::new(
                2,
                scioxide_stats::StatisticsOptions::new().with_max_bytes(15)
            )),
            15
        ),
        16
    );
    assert_eq!(
        error(FitWorkspace::<f64>::new(
            0,
            scioxide_stats::StatisticsOptions::new().with_max_bytes(0)
        ))
        .kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        error(FitWorkspace::<f64>::new(
            usize::MAX,
            scioxide_stats::StatisticsOptions::new().with_max_bytes(0)
        ))
        .kind(),
        SciOxideErrorKind::SizeOverflow
    );
}

#[test]
fn invalid_observations_retain_input_origin() {
    assert!(matches!(
        error(t_test_one_sample_with_options(
            &[1., f64::NAN, 3.],
            0.,
            Default::default()
        ))
        .view(),
        SciOxideErrorView::NonFinite {
            origin: scioxide_core::NonFiniteOrigin::Input,
            ..
        }
    ));
}
