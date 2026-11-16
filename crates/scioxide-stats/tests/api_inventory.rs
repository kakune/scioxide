//! The authoritative scioxide-stats root census.
//!
//! Every crate-root export is imported here as an external consumer, and a test
//! fails when this list and the `pub use` items of `src/lib.rs` differ. Exported
//! plans are asserted `Send + Sync + Clone`, and the canonical requirements,
//! configuration and execution forms are pinned by signature.
#![allow(clippy::type_complexity)]
use std::collections::BTreeSet;

macro_rules! inventory {
    ($($name:ident),* $(,)?) => {
        #[allow(unused_imports)]
        use scioxide_stats::{$($name),*};
        const INVENTORY: &[&str] = &[$(stringify!($name)),*];
    };
}

inventory! {
    Histogram, adjust_pvalues, anderson_darling_statistic, anderson_k_sample_test, ansari_bradley_test, barnard_exact_test, bartlett_test, binomial_confidence_limits, binomial_confidence_limits_with_options, binomial_test, boschloo_exact_test, brunner_munzel_test, bws_test, cdf, chatterjee_test, chi2_contingency_test, classical_permutation_test_into, cox_fit, cox_fit_with_options, cramer_von_mises_statistic, cramer_von_mises_test, cramer_von_mises_two_sample_test, differential_entropy, distribution_distance, entropy, epps_singleton_two_sample_test, fisher_exact_test, fit_scoped, fit_with_options, fligner_test, frequency_covariance, friedman_test, glm_fit, histogram, huber_location, invert_confidence_limits, invert_confidence_limits_with_options, invert_integer_parameter, invert_integer_parameter_with_options, invert_parameter, invert_parameter_with_options, isotonic_regression, jarque_bera_test, kaplan_meier, kendall_test, kruskal_wallis_test, ks_one_sample_statistic, ks_one_sample_test, ks_two_sample_permutation_test_into, ks_two_sample_test, kurtosis_test, ledoit_wolf_covariance, levene_test, linear_odr, logrank_test, mann_whitney_test, maximum, mean, median, median_test, minimum, mood_scale_test, nelson_aalen, noncentrality_confidence_limits, noncentrality_confidence_limits_with_options, normal_variance_confidence_limits, normal_variance_confidence_limits_with_options, normality_test, ordinary_least_squares, page_trend_test, pearson_test, point_biserial_test, poisson_means_test, poisson_rate_confidence_limits, poisson_rate_confidence_limits_with_options, posterior_summary, power_transform_normmax, power_transform_normmax_with_options, ppcc_max, ppcc_max_with_options, probplot, quantile, quantile_test, rank_sum_test, reduce_covariance, reduce_moment, reduce_moments, reduce_trimmed_moments, reduce_weighted_covariance, reduce_weighted_moments, relative_entropy, robust_slope_with_options, rolling_statistics, rolling_statistics_with_options, sample_covariance, sample_standard_deviation, sample_variance, shapiro_wilk_statistic, shapiro_wilk_test, sigma_clip_into, simulated_power, simulated_power_into, skew_test, sobol_bootstrap, sobol_indices_monte_carlo, sobol_indices_qmc, somers_contingency_test, somers_test, spearman_test, statistics, statistics_with_options, t_test_from_summaries, t_test_independent, t_test_one_sample, t_test_paired, time_series_summary, wasserstein_distance_nd, wasserstein_distance_nd_scoped, wilcoxon_test,
    AffineOdrModel, Analytic, AndersonKVariant, ArimaFitReport, ArimaForecast, ArimaInnovation,
    ArimaLikelihood, ArimaModel, ArimaOrder, ArimaTrend, ArimaWorkspace, AssociationMeasure, Axis,
    BatchView, BatchWeights, BetaBinomialPrior, BiasCorrection, BinEstimate, BinLocation, BinPlan,
    BinnedDiagnostics, BinnedRequirements, BinnedResult, BinnedStatistic, BinnedWorkspace,
    BootstrapDesign, BootstrapInterval, BootstrapMethod, BrunnerMunzelCalibration, BwsPlan,
    CcaWorkspace, CdfMethod, CdfRequirements, CdfWorkspace, ChiSquareResult, CircularOptions,
    CircularSummary, ClassicalBatchTest, ClassicalBatchWorkspace, ClassicalEffect, ClassicalMethod,
    ClassicalOptions, ClassicalPermutationResult, ClassicalTestResult, CombinedPValue,
    ConfidenceProbability, ConjugateWeights, ContingencyDesign, ContingencyView,
    ContinuityCorrection, CorrelationValidity, CovarianceSnapshot, CovarianceValidity,
    CoxBaselinePoint, CoxFit, CoxInferenceOptions, CoxReport, CoxTies, CredibleInterval,
    CredibleSummary, CressieReadParameter, DifferentialEntropyMethod, DifferentialEntropyOptions,
    DirectionNormalization, DirectionalRequirements, DirectionalSummary, DirectionalWorkspace,
    DistanceRequirements, DistanceWorkspace, DistributionBatchTest, DistributionBatchWorkspace,
    DistributionDistance, DistributionFitModel, DistributionMethod, DistributionTestResult,
    EmpiricalDistribution, EmpiricalWeights, EntropyOptions, ExactContingencyResult,
    ExactFitDerivatives, ExpectileResult, ExpectileWeights, FalseDiscoveryMethod, FamilywiseMethod,
    FitDerivativeCapabilities, FitDerivatives, FitIdentifiability, FitInformation, FitMethod,
    FitModel, FitObjective, FitObservation, FitReport, FitRequest, FitResult, FitSolver,
    FitSolverCapabilities, FitTruncation, FitWorkspace, Frequency, GammaPoissonPrior, GaussianKde,
    GlmAccepted, GlmBatchInputs, GlmBatchOutput, GlmBatchWorkspace, GlmCoefficientInference,
    GlmCovariance, GlmFamily, GlmFit, GlmFitView, GlmInputs, GlmLink, GlmObjective,
    GlmOperatorWorkspace, GlmOptions, GlmOutput, GlmReport, GlmResult, GlmSolver, GlmStorageOptions,
    GlmWorkspace, GoodnessOfFitModel, GoodnessOfFitResult, GoodnessOfFitStatistic, GroupTest,
    GroupTestMethod, HuberAccepted, HuberFit, HuberFitView, HuberOptions, HuberOutput, HuberReport,
    HuberScatterAccepted, HuberScatterFit, HuberScatterOptions, HuberScatterOutput,
    HuberScatterReport, HuberScatterWorkspace, HuberWorkspace, IndexStatistic, IndexedReplicates,
    InformationValue, IntegerParameterInversionRequest, IntegerParameterReport,
    IntegerParameterSolution, IntegerProbabilityParameter, InterpolatedQuantiles,
    IsotonicBatchOutput, IsotonicBlock, IsotonicDiagnostics, IsotonicDirection, IsotonicLaneResult,
    IsotonicRequirements, IsotonicResult, IsotonicWeights, IsotonicWorkspace, JackknifeEstimate,
    KaplanMeierPoint, KdeBandwidth, KdeCovariance, KdeOptions, KdeRequirements, KdeScale,
    KdeWorkspace, KendallVariant, KendallWeightCombination, KsPermutationResult, KurtosisConvention,
    LMomentNormalization, LagNormalization, LawFitModel, LinearRegression, LinearRegressionFit,
    LinearRegressionOptions, LocationModel, LogrankResult, Mass, MassNormalization,
    MeanConfidenceInterval, MeanIntervalAssumptions, MeanIntervalMethod, MeanIntervalPlan, MeanPowerDesign,
    MeanPowerModel, MeanPowerReference, MedianTestOutcome, MedianTies, MergePolicy, MgcPlan,
    MgcStatistic, MissingPolicy, ModeResult, MomentOrder, MomentWeights, MomentsSnapshot,
    MultivariateOptions, MultivariateRequirements, NelsonAalenPoint, NonFinitePolicy,
    NormalCredibleSummary, NormalInverseGammaPrior, NormalPosteriorMarginals,
    NormalPosteriorRequest, NullParameters, NullSampler, OdrFit, OdrIdentifiability,
    OdrIdentifiabilityDiagnostic, OdrInputs, OdrModel, OdrOptions, OdrOutput, OdrPredictors,
    OdrReport, OdrWorkspace, OnlineCovariance, OnlineMoments, OnlineWeightedCovariance,
    OnlineWeightedMoments, OperatorRegressionMethod, OperatorRegressionOptions,
    OperatorRegressionReport, OperatorRegressionResult, OperatorRegressionWorkspace, OrderResult,
    PValueAdjustment, PValueAdjustmentWorkspace, PValueCombination, ParameterConfidenceLimits,
    ParameterInversionReport, ParameterInversionRequest, ParameterMonotonicity,
    ParameterProbability, ParameterSolution, PcaScaling, PcaWorkspace, PercentileKind,
    PeriodogramBin, PeriodogramDiagnostics, PeriodogramWorkspace, PermutationKind, PermutationMode,
    PostHocComparison, PostHocMethod, PostHocOptions, PostHocResult, PosteriorOptions,
    PosteriorSummary, PowerEstimate, PowerFitMethod, PowerTransform,
    PreparedArimaFit, PreparedCca, PreparedCox, PreparedFit, PreparedGlm, PreparedGroups,
    PreparedHuber, PreparedHuberScatter, PreparedOdr, PreparedOperatorGlm,
    PreparedOperatorRegression, PreparedPca, PreparedPowerFit, PreparedPpcc, PreparedRegression,
    ProbabilityParameter, ProbabilityPlot, ProbabilityPlotFit, QuantileConvention, QuantileMethod,
    QuantileProbability, QuantileResult, QuantileRule, Rank, RankMethod, ReductionRequirements,
    ReductionWorkspace, RegressionContrast, RegressionCovariance, RegressionFactorization,
    RegressionWeights, RegressionWorkspace, Reliability, ReplicateExecution, ReplicateOptions, ResampledIndices,
    ResamplingInference, ResamplingMethod, ResamplingOptions, ResamplingOutput, ResamplingPlan,
    ResamplingProblem, ResamplingResult, ResamplingTest, ResamplingWorkspace, RobustIntercept,
    RobustSlope, RobustSlopeMethod, RobustSlopeWorkspace, RollingBoundary, RollingWindow,
    SampleSizeResult, SampleStatistic, SchoenfeldResidual, SelectionSamples, SequentialReplicates,
    SigmaClipOptions, SigmaClipResult, SignedRankZeros, SobolBootstrap, SobolEvaluations,
    SobolIndices, SobolMonteCarloResult, StandardizationRequirements, StandardizationWorkspace,
    Statistic, StatisticCapability, StatisticValue, StatisticsGeometry, StatisticsOptions,
    StatisticsRequirements, StatisticsWorkspace, StatsTermination, SurvivalEstimator,
    SurvivalObservation, SurvivalOptions, SurvivalOutcome, SurvivalRequirements, SurvivalWorkspace,
    TTestSummary, TTestVariance, TailMeanConvention, TailMeanResult, TailSide, TailWeights,
    TestAlternative, TestCalibration, TestSamplePolicy, TimeSeriesCentering, TimeSeriesLag,
    TimeSeriesOptions, TimeSeriesRequirements, TimeSeriesStatistic, TimeSeriesWorkspace,
    TransportCandidate, TransportOptions, TransportProblem, TransportResult, TransportSolver,
    TransportWorkspace, TrimInclusion, TrimLimits, TrimSide, UnconditionalOptions, Unweighted,
    VarianceCenter, VariancePooling, WeightModel, WeightSemantics, WeightedCovarianceSnapshot,
    WeightedMomentsSnapshot, Weights, WorkspaceIndexStatistic, WorkspaceSampleStatistic,
    adjust_pvalues_with_options, adjust_pvalues_into, anderson_darling_statistic_with_options, anderson_k_sample_test_with_options,
    ansari_bradley_test_with_options, barnard_exact_test_with_options, bartlett_test_with_options, binned_statistics, binned_statistics_into,
    binomial_confidence_limits_scoped, binomial_test_with_options, bootstrap, boschloo_exact_test_with_options, brunner_munzel_test_with_options,
    bws_test_with_options, cdf_with_options, cdf_into, chatterjee_test_with_options, power_divergence_contingency_test, classical_permutation_test,
    classical_tests_into, combine_pvalues, contingency_association, cox_fit_scoped,
    cramer_von_mises_statistic_with_options, cramer_von_mises_test_with_options, cramer_von_mises_two_sample_test_with_options,
    credible_interval, differential_entropy_with_options, directional_statistics_into, distribution_distance_with_options,
    distribution_tests_into, entropy_with_options, epps_singleton_two_sample_test_with_options, fisher_exact_test_with_options, fit,
    fit_information, fit_into, fit_objectives_into, fligner_test_with_options, frequency_covariance_with_options,
    friedman_test_with_options, glm_batch_into, goodness_of_fit_test_into, group_tests_into, huber_location_with_options,
    invert_confidence_limits_scoped, invert_integer_parameter_scoped, invert_integer_parameters_into,
    invert_parameter_scoped, invert_parameters_into, isotonic_regression_with_options, isotonic_regression_axes_into,
    isotonic_regression_into, jackknife, jarque_bera_test_with_options, kaplan_meier_with_options, kaplan_meier_into,
    kde_into, kendall_test_with_options, kruskal_wallis_test_with_options, ks_one_sample_statistic_with_options, ks_one_sample_test_with_options,
    ks_two_sample_permutation_test, ks_two_sample_test_with_options, kurtosis_test_with_options, ledoit_wolf_covariance_with_options,
    levene_test_with_options, linear_odr_with_options, linear_regression, logrank_test_with_options, mann_whitney_test_with_options,
    mean_confidence_interval, median_test_with_options, monte_carlo_test, mood_scale_test_with_options, nelson_aalen_with_options,
    nelson_aalen_into, noncentrality_confidence_limits_scoped, normal_credible_summary,
    normal_posterior_marginals, normal_posteriors_into, normal_variance_confidence_limits_scoped,
    normality_test_with_options, odds_ratio, page_trend_test_with_options, pearson_test_with_options, periodogram_diagnostics_into,
    permutation_test, point_biserial_test_with_options, poisson_means_test_with_options, poisson_rate_confidence_limits_scoped,
    posterior_summaries_into, posterior_summary_with_options, power, power_divergence_test, power_transform_into,
    power_transform_llf, power_transform_normmax_scoped, ppcc_max_scoped, probplot_with_options, quantile_in_place,
    quantile_test_with_options, rank_axes_into, rank_slice, rank_slice_into, rank_sum_test_with_options, reduce_covariance_with_options,
    reduce_covariance_into, reduce_moment_with_options, reduce_moment_into, reduce_moments_with_options, reduce_moments_into,
    reduce_trimmed_moments_with_options, reduce_trimmed_moments_into, reduce_weighted_covariance_with_options,
    reduce_weighted_covariance_into, reduce_weighted_moments_with_options, reduce_weighted_moments_into,
    regression_batch_into, relative_risk, robust_slope, rolling_statistics_into, sample_covariance_with_options,
    sample_size, shapiro_wilk_statistic_with_options, shapiro_wilk_test_with_options, shrink_covariance, skew_test_with_options,
    sobol_bootstrap_scoped, sobol_indices, sobol_indices_monte_carlo_scoped, sobol_indices_qmc_scoped,
    somers_contingency_test_with_options, somers_test_with_options, spearman_test_with_options, standardize, standardize_axes_into,
    statistic, statistics_into, t_test_from_summaries_with_options, t_test_independent_with_options, t_test_one_sample_with_options,
    t_test_paired_with_options, tail_mean_in_place, time_series_summaries_into, time_series_summary_with_options,
    time_series_summary_into, weighted_kendall_tau, wilcoxon_test_with_options,
}

/// Asserts the plan auto traits for each listed instantiation and
/// records the plan names for the completeness test.
macro_rules! plans {
    ($($name:ident $(<$($argument:tt),*>)?),* $(,)?) => {
        const _: fn() = || {
            fn plan<T: Send + Sync + Clone>() {}
            $(plan::<$name $(<$($argument),*>)?>();)*
        };
        const PLANS: &[&str] = &[$(stringify!($name)),*];
    };
}

plans! {
    BinPlan<f64>, BwsPlan, MeanIntervalPlan, MgcPlan, ResamplingPlan<f64>,
}

// Prepared regression, GLM, group and Huber plans are `Send + Sync + Clone`;
// operator plans are whenever the borrowed operator type is `Sync`.
const _: fn() = || {
    fn plan<T: Send + Sync + Clone>() {}
    type Operator = dyn scioxide_core::AdjointOperator<f64> + Sync;
    plan::<PreparedRegression<f64>>();
    plan::<PreparedRegression<scioxide_core::Complex<f64>>>();
    plan::<PreparedOperatorRegression<'static, f64, Operator>>();
    plan::<PreparedGlm<f64>>();
    plan::<PreparedOperatorGlm<'static, f64, Operator>>();
    plan::<PreparedGroups<f64>>();
    plan::<PreparedHuber<'static, f64>>();
};

// Regression-family budgets live in options and plans retain them; batch forms
// take inputs, options, output, workspace and then the context.
const _: fn() = || {
    use scioxide_core::{ExecutionContext, ExecutionPolicy, SciOxideError};
    type Cx = ExecutionContext<'static>;
    type Fit = LinearRegressionFit<f64>;
    type Posterior = Option<NormalInverseGammaPrior<f64>>;
    let _: fn(LinearRegressionOptions<f64>, usize) -> LinearRegressionOptions<f64> =
        LinearRegressionOptions::with_max_bytes;
    let _: fn(PostHocOptions, usize) -> PostHocOptions = PostHocOptions::with_max_bytes;
    let _: fn(&PreparedRegression<f64>) -> Result<RegressionWorkspace<f64>, SciOxideError> =
        PreparedRegression::workspace;
    let _: fn(&PreparedRegression<f64>) -> Result<Fit, SciOxideError> = PreparedRegression::output;
    let _: fn(&[f64], HuberOptions<f64>) -> Result<HuberFit<f64>, SciOxideError> =
        huber_location_with_options::<f64>;
    let _: fn(&[&[f64]], StatisticsOptions) -> Result<GroupTest<f64>, SciOxideError> =
        kruskal_wallis_test_with_options::<f64>;
    let _: fn(
        &PreparedRegression<f64>,
        &[&[f64]],
        StatisticsOptions,
        &mut [Fit],
        &mut [Fit],
        &mut [RegressionWorkspace<f64>],
        Cx,
    ) -> Result<(), SciOxideError> = regression_batch_into::<f64>;
    let _: fn(
        &[PreparedGroups<f64>],
        GroupTestMethod,
        StatisticsOptions,
        &mut [GroupTest<f64>],
        &mut [GroupTest<f64>],
        Cx,
    ) -> Result<(), SciOxideError> = group_tests_into::<f64>;
    let _: fn(
        &[NormalPosteriorRequest<'static, f64>],
        ExecutionPolicy,
        &mut [Posterior],
        &mut [Posterior],
        Cx,
    ) -> Result<(), SciOxideError> = normal_posteriors_into::<f64>;
};

/// Names re-exported by the `pub use` statements of a crate root. Each
/// statement re-exports either one path or one braced list from a module.
fn root_exports(source: &str) -> BTreeSet<&str> {
    source
        .split("\npub use ")
        .skip(1)
        .filter_map(|statement| statement.split_once(';'))
        .flat_map(|(path, _)| {
            path.rsplit_once("::")
                .map_or(path, |(_, names)| names)
                .trim_matches(|c: char| c == '{' || c == '}' || c.is_whitespace())
                .split(',')
        })
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .collect()
}

#[test]
fn inventory_lists_every_root_export_once() {
    let listed: BTreeSet<_> = INVENTORY.iter().copied().collect();
    assert_eq!(listed.len(), INVENTORY.len(), "duplicate inventory entry");
    let root = root_exports(include_str!("../src/lib.rs"));
    let unlisted: Vec<_> = root.difference(&listed).collect();
    let stale: Vec<_> = listed.difference(&root).collect();
    assert!(
        unlisted.is_empty() && stale.is_empty(),
        "update tests/api_inventory.rs\nunlisted: {unlisted:?}\nstale: {stale:?}"
    );
}

#[test]
fn every_exported_plan_is_send_sync_clone() {
    let exported: BTreeSet<_> = INVENTORY
        .iter()
        .copied()
        .filter(|name| name.ends_with("Plan"))
        .collect();
    let asserted: BTreeSet<_> = PLANS.iter().copied().collect();
    assert_eq!(
        exported, asserted,
        "update plans! in tests/api_inventory.rs"
    );
}

#[test]
fn root_exports_read_single_and_braced_statements() {
    let source = "mod a;\npub use a::One;\npub use b::{\n    Two, three,\n};\nuse c::Hidden;\npub(crate) use d::Internal;";
    assert_eq!(
        root_exports(source),
        BTreeSet::from(["One", "Two", "three"])
    );
}

// One requirements protocol and one execution form per operation: inputs, then
// options, output, workspace and context. Options are configured through
// `with_*` setters and read through accessors.
const _: fn() = || {
    use scioxide_core::{Capacity, ExecutionContext, SciOxideError};
    type Cx = ExecutionContext<'static>;
    type View = BatchView<'static, f64>;
    type Ws = StatisticsWorkspace<f64>;

    // Reusable statistics workspaces.
    let _: fn(
        View,
        Option<RollingWindow>,
        StatisticsOptions,
        Cx,
    ) -> Result<StatisticsRequirements<f64>, SciOxideError> =
        StatisticsWorkspace::workspace_requirements::<View>;
    let _: fn(&mut Ws, &StatisticsRequirements<f64>) -> Result<(), SciOxideError> =
        StatisticsWorkspace::try_reserve;
    let _: fn(&Ws, &StatisticsRequirements<f64>) -> Result<Capacity, SciOxideError> =
        StatisticsWorkspace::check;
    let _: fn(&StatisticsRequirements<f64>) -> Result<Ws, SciOxideError> =
        StatisticsWorkspace::workspace;

    // Scalar/axis summaries, rolling windows, ranking and standardization.
    let _: fn(
        View,
        Statistic<f64>,
        StatisticsOptions,
        &mut [StatisticValue<f64>],
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = statistics_into::<f64, Unweighted>;
    let _: fn(
        View,
        Statistic<f64>,
        RollingWindow,
        StatisticsOptions,
        &mut [StatisticValue<f64>],
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = rolling_statistics_into::<f64, Unweighted>;
    let _: fn(&[f64], RankMethod, &mut [Option<Rank>], &mut Ws, Cx) -> Result<(), SciOxideError> =
        rank_slice_into::<f64>;
    let _: fn(
        View,
        RankMethod,
        StatisticsOptions,
        &mut [Option<Rank>],
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = rank_axes_into::<f64>;
    let _: fn(
        View,
        usize,
        StatisticsOptions,
        &mut [f64],
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = standardize_axes_into::<f64>;
    let _: fn(
        View,
        StatisticsOptions,
        &mut [MomentsSnapshot<f64>],
        &mut ReductionWorkspace<f64>,
        Cx,
    ) -> Result<(), SciOxideError> = reduce_moments_into::<f64>;
    let _: fn(
        &GaussianKde<f64>,
        View,
        KdeScale,
        StatisticsOptions,
        &mut [f64],
        &mut KdeWorkspace<f64>,
        Cx,
    ) -> Result<(), SciOxideError> = kde_into::<f64>;

    // Inference options keep private fields behind setters and accessors.
    let _: fn(ClassicalOptions, TestAlternative) -> ClassicalOptions =
        ClassicalOptions::with_alternative;
    let _: fn(&ClassicalOptions) -> usize = ClassicalOptions::max_work;
    let _: fn(ResamplingOptions<f64>, usize) -> ResamplingOptions<f64> =
        ResamplingOptions::with_max_rng_calls;
    let _: fn() -> TransportOptions<f64> = TransportOptions::new;
    let _: fn(TimeSeriesOptions, LagNormalization) -> TimeSeriesOptions =
        TimeSeriesOptions::with_normalization;
    let _: fn(PosteriorOptions, usize) -> PosteriorOptions = PosteriorOptions::with_max_lag;
    let _: fn(&PostHocOptions) -> ConfidenceProbability = PostHocOptions::confidence;
    let _: fn(UnconditionalOptions, ClassicalOptions) -> UnconditionalOptions =
        UnconditionalOptions::with_classical;
    let _: fn(
        LinearRegressionOptions<f64>,
        RegressionFactorization,
    ) -> LinearRegressionOptions<f64> = LinearRegressionOptions::with_factorization;
};

// Prepared multivariate, survival and time-series plans hold only shared
// borrows and owned data, so they are `Send + Sync + Clone`.
const _: fn() = || {
    fn plan<T: Send + Sync + Clone>() {}
    plan::<PreparedOdr<'static, f64>>();
    plan::<PreparedCox<'static, f64>>();
    plan::<PreparedArimaFit<'static, f64>>();
    plan::<PreparedPca<f64>>();
    plan::<PreparedCca<f64>>();
};

// Time-series, survival and multivariate budgets live in options; executions
// order inputs, options, output, workspace and context.
const _: fn() = || {
    use scioxide_core::{Capacity, ExecutionContext, SciOxideError};
    use scioxide_linalg::{Matrix, MatrixView};
    type Cx = ExecutionContext<'static>;
    type View = BatchView<'static, f64>;
    type Ws = TimeSeriesWorkspace<f64>;
    type Req = TimeSeriesRequirements<f64>;
    type Counts = Weights<&'static [u64], Frequency>;

    let _: fn(View, usize, StatisticsOptions, Cx) -> Result<Req, SciOxideError> =
        TimeSeriesWorkspace::workspace_requirements;
    let _: fn(usize, usize, StatisticsOptions) -> Result<Req, SciOxideError> =
        TimeSeriesWorkspace::scalar_requirements;
    let _: fn(&mut Ws, &Req) -> Result<(), SciOxideError> = TimeSeriesWorkspace::try_reserve;
    let _: fn(&Ws, &Req) -> Result<Capacity, SciOxideError> = TimeSeriesWorkspace::check;
    let _: fn(&Req) -> Result<Ws, SciOxideError> = TimeSeriesWorkspace::workspace;
    let _: fn(
        View,
        TimeSeriesStatistic,
        usize,
        TimeSeriesOptions,
        StatisticsOptions,
        &mut [TimeSeriesLag<f64>],
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = time_series_summaries_into::<f64>;
    let _: fn(
        &[f64],
        Option<&[bool]>,
        TimeSeriesStatistic,
        usize,
        TimeSeriesOptions,
        StatisticsOptions,
        &mut [TimeSeriesLag<f64>],
        &mut Ws,
        Cx,
    ) -> Result<(), SciOxideError> = time_series_summary_into::<f64>;
    let _: fn(
        &[f64],
        Option<&[bool]>,
        TimeSeriesStatistic,
        usize,
        TimeSeriesOptions,
        StatisticsOptions,
    ) -> Result<Vec<TimeSeriesLag<f64>>, SciOxideError> = time_series_summary_with_options::<f64>;
    let _: fn(
        View,
        usize,
        ArimaOrder,
        ArimaTrend,
        StatisticsOptions,
    ) -> Result<PreparedArimaFit<'static, f64>, SciOxideError> = PreparedArimaFit::from_view;
    let _: fn(&PreparedArimaFit<'static, f64>) -> Result<ArimaWorkspace<f64>, SciOxideError> =
        PreparedArimaFit::workspace;
    let _: fn(&ArimaModel<f64>, usize) -> Result<ArimaWorkspace<f64>, SciOxideError> =
        ArimaModel::workspace;
    let _: fn(
        &scioxide_signal::SpectralEstimatePlan<f64>,
        StatisticsOptions,
    ) -> Result<PeriodogramWorkspace<f64>, SciOxideError> = PeriodogramWorkspace::new;
    let _: fn(
        &scioxide_signal::SpectralEstimatePlan<f64>,
        &[f64],
        &mut [PeriodogramBin<f64>],
        &mut PeriodogramWorkspace<f64>,
        Cx,
    ) -> Result<PeriodogramDiagnostics<f64>, SciOxideError> = periodogram_diagnostics_into::<f64>;

    let _: fn(
        &[SurvivalObservation<f64>],
        &[SurvivalObservation<f64>],
        SurvivalOptions,
    ) -> Result<LogrankResult<f64>, SciOxideError> = logrank_test_with_options::<f64>;
    let _: fn(
        &'static [SurvivalObservation<f64>],
        MatrixView<'static, f64>,
        CoxTies,
        SurvivalOptions,
    ) -> Result<PreparedCox<'static, f64>, SciOxideError> = PreparedCox::new;

    type Covariance = Result<(Vec<f64>, Matrix<f64>), SciOxideError>;
    let _: fn(MatrixView<'static, f64>, MultivariateOptions<f64>) -> Covariance =
        sample_covariance_with_options::<f64>;
    let _: fn(
        MatrixView<'static, f64>,
        MultivariateOptions<f64>,
    ) -> Result<(f64, Matrix<f64>), SciOxideError> = ledoit_wolf_covariance_with_options::<f64>;
    let _: fn(MatrixView<'static, f64>, Counts, MultivariateOptions<f64>) -> Covariance =
        frequency_covariance_with_options::<f64, Counts>;
};

const _: fn() = || {
    fn options<T: Default>() {}
    options::<StatisticsOptions>();
    options::<ClassicalOptions>();
    options::<UnconditionalOptions>();
    options::<SurvivalOptions>();
    options::<CoxInferenceOptions<f64>>();
    options::<MultivariateOptions<f64>>();
    options::<LinearRegressionOptions<f64>>();
    options::<GlmStorageOptions>();
    options::<OdrOptions>();
    options::<KdeOptions<'static, f64>>();
    options::<EntropyOptions<f64>>();
    options::<DifferentialEntropyOptions<f64>>();
    options::<CircularOptions<f64>>();
    options::<PosteriorOptions>();
    options::<HuberOptions<f64>>();
    options::<HuberScatterOptions<f64>>();
    options::<GlmOptions<f64>>();
    options::<PostHocOptions>();
    options::<OperatorRegressionOptions<f64>>();
    options::<TransportOptions<f64>>();
    options::<ResamplingOptions<f64>>();
    options::<TimeSeriesOptions>();
    options::<SigmaClipOptions<f64>>();
};
