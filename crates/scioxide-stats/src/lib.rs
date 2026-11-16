//! Descriptive statistics, empirical distributions, hypothesis tests and statistical models.
//!
//! The crate summarizes samples (moments, quantiles, ranks, robust and circular
//! statistics), estimates distributions (empirical CDFs, kernel densities,
//! histograms, isotonic fits), and tests hypotheses with classical, rank,
//! distribution, contingency-table, resampling and Monte Carlo methods. It fits
//! linear, generalized linear, robust, errors-in-variables, survival and ARIMA
//! models, and covers PCA and CCA, conjugate Bayesian updates, MCMC diagnostics,
//! power analysis and Sobol sensitivity indices. Functions accept `f32` and
//! `f64` data as slices or strided arrays, optionally with a validity mask and
//! typed weights; location and dispersion summaries also accept complex data.
//!
//! # Quick start
//!
//! ```
//! use scioxide_stats::{mean, sample_variance, t_test_one_sample};
//! let x = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
//! assert!((mean(&x)? - 5.0).abs() < 1e-12);
//! assert!((sample_variance(&x)? - 32.0 / 7.0).abs() < 1e-12);
//! let test = t_test_one_sample(&x, 4.0)?;
//! assert!((test.statistic() - 1.75_f64.sqrt()).abs() < 1e-12);
//! assert!((test.p_value() - 0.2275).abs() < 1e-3);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! | Task | Entry points | Method |
//! |---|---|---|
//! | Mean, variance, quantile and other summaries of a slice | [`mean`], [`sample_variance`], [`sample_standard_deviation`], [`median`], [`quantile`], [`statistic`] | |
//! | Summaries along one axis of a strided array | [`statistics`], [`statistics_into`], [`StatisticsWorkspace`] | independent lanes, optionally in parallel |
//! | Moving-window summaries | [`rolling_statistics`], [`rolling_statistics_into`], [`RollingWindow`] | |
//! | Ranks, z-scores, sigma clipping | [`rank_slice`], [`rank_axes_into`], [`standardize`], [`standardize_axes_into`] | |
//! | Streaming moments and covariance | [`OnlineMoments`], [`OnlineWeightedMoments`], [`OnlineCovariance`], [`OnlineWeightedCovariance`] | one-pass updates; accumulators merge |
//! | Moments and covariance of a batch | [`reduce_moments`], [`reduce_trimmed_moments`], [`reduce_covariance`], [`reduce_weighted_covariance`] | |
//! | Confidence interval for a mean | [`mean_confidence_interval`], [`MeanIntervalPlan`] | Student t or normal critical value |
//! | Quantiles, empirical CDF, tail means | [`EmpiricalDistribution`], [`quantile_in_place`], [`tail_mean_in_place`], [`cdf`] | the nine Hyndman–Fan sample quantiles; sort once, or select in place |
//! | Kernel density estimation | [`GaussianKde::new`], [`kde_into`] | Gaussian kernel with Scott, Silverman or explicit bandwidth |
//! | Histograms and binned statistics | [`histogram`], [`BinPlan`], [`binned_statistics`] | |
//! | Monotone regression | [`isotonic_regression`], [`isotonic_regression_axes_into`] | pool-adjacent-violators algorithm |
//! | Directional data | [`directional_statistics_into`], [`Statistic::circular`] | mean resultant vector |
//! | Entropy, divergence, distance between distributions | [`entropy`], [`differential_entropy`], [`distribution_distance`] | Shannon entropy, Kullback–Leibler divergence; spacing estimators; Wasserstein-1 and energy distance |
//! | Discrete optimal transport | [`TransportProblem`] | a [`TransportSolver`] you provide |
//! | Location, scale and correlation tests | [`t_test_one_sample`], [`t_test_independent`], [`t_test_paired`], [`mann_whitney_test`], [`wilcoxon_test`], [`pearson_test`], [`spearman_test`], [`kendall_test`], [`classical_tests_into`] | Student and Welch t-tests; rank tests, exact or normal approximation |
//! | Goodness of fit and normality | [`ks_one_sample_test`], [`ks_two_sample_test`], [`cramer_von_mises_test`], [`anderson_k_sample_test`], [`shapiro_wilk_test`], [`normality_test`], [`goodness_of_fit_test_into`], [`distribution_tests_into`] | Kolmogorov–Smirnov, Cramér–von Mises, Anderson–Darling, Shapiro–Wilk; Monte Carlo calibration for fitted families |
//! | Equality of variances | [`levene_test`], [`bartlett_test`], [`fligner_test`], [`ansari_bradley_test`], [`mood_scale_test`] | |
//! | Contingency tables | [`chi2_contingency_test`], [`power_divergence_contingency_test`], [`power_divergence_test`], [`fisher_exact_test`], [`barnard_exact_test`], [`boschloo_exact_test`], [`contingency_association`] | Pearson χ² and Cressie–Read power divergence; exact conditional and unconditional tests |
//! | Multiple testing | [`adjust_pvalues`], [`combine_pvalues`] | Bonferroni, Holm, Benjamini–Hochberg, Benjamini–Yekutieli; Fisher, Pearson, Tippett, Stouffer and Mudholkar–George combination |
//! | Bootstrap, jackknife, permutation and Monte Carlo tests | [`ResamplingPlan`], [`bootstrap`], [`jackknife`], [`permutation_test`], [`monte_carlo_test`] | BCa intervals by default; percentile and basic through the plan |
//! | Linear least squares regression | [`ordinary_least_squares`], [`linear_regression`], [`PreparedRegression`], [`regression_batch_into`] | pivoted QR, SVD for rank-deficient designs |
//! | Regression with a matrix-free design | [`PreparedOperatorRegression`], [`PreparedOperatorGlm`] | LSQR or LSMR |
//! | Generalized linear models | [`glm_fit`], [`PreparedGlm`], [`glm_batch_into`] | iteratively reweighted least squares |
//! | Robust regression and location | [`PreparedHuber`], [`huber_location`], [`robust_slope`] | Huber M-estimation; Theil–Sen and Siegel repeated-median slopes |
//! | Comparing groups | [`PreparedGroups`], [`group_tests_into`], [`kruskal_wallis_test`], [`friedman_test`] | one-way ANOVA, Welch, Alexander–Govern, Kruskal–Wallis, Friedman; Tukey–Kramer, Games–Howell and Dunnett comparisons |
//! | Fitting a distribution | [`fit`], [`fit_into`], [`FitRequest`], [`probplot`], [`ppcc_max`] | maximum likelihood, moments or quantile matching, with a [`FitSolver`] you provide |
//! | Box–Cox and Yeo–Johnson transforms | [`PowerTransform`], [`power_transform_into`], [`power_transform_normmax`] | |
//! | Covariance, PCA, CCA | [`sample_covariance`], [`ledoit_wolf_covariance`], [`shrink_covariance`], [`PreparedPca::new`], [`PreparedCca`] | SVD of the centered data; Ledoit–Wolf shrinkage |
//! | Errors in variables | [`linear_odr`], [`PreparedOdr`] | orthogonal distance regression; SVD for linear models |
//! | Robust scatter | [`PreparedHuberScatter`] | Huber M-estimator of location and scatter |
//! | Survival analysis | [`kaplan_meier`], [`nelson_aalen`], [`logrank_test`], [`cox_fit`], [`PreparedCox`] | Kaplan–Meier with Greenwood variance, Nelson–Aalen, log-rank test, Cox proportional hazards with Efron or Breslow ties |
//! | Time series | [`time_series_summary`], [`time_series_summaries_into`], [`periodogram_diagnostics_into`] | autocovariance, autocorrelation, Levinson–Durbin partial autocorrelation, periodogram |
//! | ARIMA models | [`PreparedArimaFit`] | Gaussian likelihood |
//! | MCMC convergence diagnostics | [`posterior_summary`], [`posterior_summaries_into`] | rank-normalized split R̂ and effective sample size |
//! | Conjugate Bayesian updates | [`BetaBinomialPrior`], [`GammaPoissonPrior`], [`NormalInverseGammaPrior`], [`credible_interval`] | closed-form posteriors |
//! | Confidence limits by inverting a test | [`invert_parameter`], [`binomial_confidence_limits`], [`poisson_rate_confidence_limits`], [`noncentrality_confidence_limits`] | bisection on a monotone tail probability |
//! | Power and sample size | [`power`], [`sample_size`], [`simulated_power`] | Student t or normal model, or simulation |
//! | Global sensitivity analysis | [`sobol_indices`], [`sobol_indices_monte_carlo`], [`sobol_indices_qmc`], [`sobol_bootstrap`] | Saltelli first-order and Jansen total-order estimators |
//!
//! Most families also have a batch form (`*_into` or `*_batch_into`) that
//! evaluates many independent problems from one strided array.
//!
//! # Data, weights and missing values
//!
//! Slice functions take `&[S]`. Batch functions take a [`BatchView`], which
//! borrows values with a [`StridedLayout`](scioxide_core::StridedLayout) and
//! selects the sample axis with [`Axis`]. The other axes enumerate independent
//! batches, so a 2 × 3 array with `Axis::Index(1)` holds two samples of three
//! observations. A view can also carry a validity mask, where `false` marks a
//! missing observation, and weights. Masks and weights have one entry per
//! observation, in batch-major, sample-minor order, whatever the strides.
//!
//! [`StatisticsOptions`] decides what happens to missing and non-finite
//! observations. With the default [`MissingPolicy::Reject`] and
//! [`NonFinitePolicy::Reject`], the first one is an error. With `Skip`, it is
//! left out of the estimate.
//!
//! Weights carry their statistical meaning in their type, through
//! [`Weights<D, M>`](Weights). The model `M` decides how weights enter
//! variances, degrees of freedom and quantiles, and functions accept only the
//! models for which they are defined:
//!
//! | Model | Meaning |
//! |---|---|
//! | [`Unweighted`] | each observation counts once |
//! | [`Frequency`] | exact integer replication counts (`u64` for empirical distributions) |
//! | [`Mass`] | nonnegative relative masses, without a sampling-variance interpretation |
//! | [`Reliability`] | relative reliabilities, with Kish's effective sample size |
//! | [`Analytic`] | inverse variances with a common unknown noise scale |
//!
//! ```
//! use scioxide_stats::{
//!     EmpiricalDistribution, Frequency, QuantileMethod, QuantileProbability, QuantileRule, Weights,
//! };
//!
//! // The sample 1, 2, 2, 10, given as values with counts
//! let counts = [1u64, 2, 1];
//! let distribution =
//!     EmpiricalDistribution::new(&[1.0_f64, 2.0, 10.0], Weights::<_, Frequency>::new(&counts))?;
//! let median = distribution.quantile(
//!     QuantileProbability::new(0.5)?,
//!     QuantileRule::new(QuantileMethod::Linear),
//! )?;
//! assert_eq!(median.value(), 2.0);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Results and termination
//!
//! Results are read through accessors. A value that is undefined for the data,
//! such as the sample variance of one observation in a streaming snapshot,
//! is `None`. Hypothesis tests always expose a numerical p-value.
//!
//! Iterative and replicate computations (generalized linear models, Huber
//! fits, distribution fitting, Cox regression, resampling) report one
//! [`StatsTermination`].
//! [`StatsTermination::is_success`] is true for `Completed`, `Converged` and
//! `ExactFit`. A work limit, cancellation or deadline can end such a
//! computation with `BudgetExhausted` or `Stopped`, and `RankDeficient`,
//! `Separated`, `NoProgress` and `NotConverged` report a fit that stopped
//! without meeting its criterion. Check the status before you use the estimate.
//! Parameter inversion reports the `RootTermination` of its bisection through
//! [`ParameterInversionReport::root`].
//!
//! Distribution fitting, Cox regression, nonlinear orthogonal distance
//! regression and optimal transport do not contain an optimizer. You pass one through [`FitSolver`] or [`TransportSolver`], and
//! generalized linear models accept one in place of their built-in iteration
//! ([`GlmSolver`]).
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors). In this crate:
//!
//! - `InvalidInput`: too few observations for the statistic, a probability or
//!   confidence level out of range, negative or non-integer weights where
//!   the model forbids them, an observation masked as missing under
//!   [`MissingPolicy::Reject`], or options that do not apply to the method.
//! - `NonFinite`: a NaN or infinite observation under
//!   [`NonFinitePolicy::Reject`], a non-finite weight or parameter, or a
//!   non-finite value returned by a callback.
//! - `DimensionMismatch`: a mask, weight, output or workspace whose length does
//!   not match the data.
//! - `NumericalInstability`: a degenerate sample, such as zero variance where a
//!   test divides by the standard error, or a result outside the range of the
//!   output type.
//! - `ConvergenceFailure`: an exact test, permutation enumeration or iteration
//!   needs more work than its configured work limit (for example
//!   [`ClassicalOptions::max_work`]).
//! - `ResourceLimit`, `SizeOverflow`, `AllocationFailure`: a workspace exceeds
//!   [`StatisticsOptions::max_bytes`] or another configured byte limit, or
//!   cannot be allocated.
//!
//! Each function lists its own conditions. Errors returned by your callbacks
//! and samplers propagate unchanged.
//!
//! # One-shot forms
//!
//! One-shots take plain problem data, allocate their storage, and use sequential
//! native capabilities. Configured forms use the same numerical kernels.
//!
//! | Choice | Fixed value |
//! |---|---|
//! | Weights | Unweighted, except functions whose subject is a weight model |
//! | Sample dispersion | Divisor `n − 1` |
//! | Quantiles | Hyndman–Fan type 7 (linear interpolation) |
//! | Mean interval | Two-sided 95% Student t, `n − 1` degrees of freedom |
//! | Test alternative | Two-sided; greater for Chatterjee and Page |
//! | Test calibration | Auto |
//! | Continuity correction | Half a unit; Yates for 2 × 2 Pearson tables |
//! | Independent t test | Welch unequal variances |
//! | Signed-rank zeros | Discard |
//! | Brunner–Munzel reference | Student t |
//! | Variance-test center | Median |
//! | Anderson k-sample ties | Midrank |
//! | Median-test ties | Below |
//! | Isotonic direction | Increasing |
//! | Robust intercept | Joint |
//! | KDE bandwidth | Scott, full covariance |
//! | Missing and non-finite values | Reject |
//! | Bootstrap | Independent BCa, 9999 replicates, 95% confidence |
//! | Permutations | Independent randomized assignments, 9999 replicates |
//!
//! Generators remain explicit arguments and are never seeded internally by a
//! random one-shot. Jackknife needs no generator. Options have private fields,
//! infallible setters and a `Default`; preparation or execution validates them.
//!
//! A one-shot cannot return an incomplete estimate. Successful terminations
//! (Completed, Converged and ExactFit) return their answer; other terminations
//! map to errors as follows:
//!
//! | Termination | Error |
//! |---|---|
//! | BudgetExhausted | IterationLimit for GLM and Huber; NestedSolver for injected fits; Sample resource exhaustion for resampling |
//! | Stopped | ExecutionStopped |
//! | NotConverged | NestedSolver convergence failure |
//! | NoProgress | Stagnation convergence failure |
//! | Separated | Divergence convergence failure |
//! | RankDeficient | Singular |
//!
//! Iteration and proposal limits are per call: injected solvers use their own
//! configured limits, resampling uses its replicate count, and clipping uses
//! its pass limit. Child evaluations and generator draws charge every ancestor's
//! shared budget; the parent's own counts and accepted state stay unchanged.
//!
//! # Reusing plans and workspaces
//!
//! The four levels are a one-shot `f(inputs)`, a configured owned call
//! `f_with_options(inputs, options)`, a reusable plan or `Prepared*` value,
//! and `f_into(inputs, options, output, workspace, context)`. Plan methods
//! provide owned `fit` or `evaluate` calls and scoped or caller-output forms.
//! Scoped calls accept an execution context for a root solve or a mutable
//! solve context for a child epoch. Atomic kernels may run in
//! parallel. Workspaces report their size with `workspace_requirements` and are
//! created with `workspace`, grown with `try_reserve` and checked with `check`,
//! as described in [Plans and workspaces](scioxide_core#plans-and-workspaces).
//! With a sufficient workspace, the numerical work does not allocate
//! ([Allocation](scioxide_core#allocation)), and on error the output is
//! unchanged ([Output on error](scioxide_core#output-on-error)).
//!
//! ```
//! use scioxide_core::{ExecutionContext, StridedLayout};
//! use scioxide_stats::{
//!     Axis, BatchView, Statistic, StatisticValue, StatisticsOptions, StatisticsWorkspace,
//!     statistics_into,
//! };
//!
//! // Two rows of three observations; the mean of each row
//! let values = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let view = BatchView::new(&values, StridedLayout::new(&[2, 3], &[3, 1])?, Axis::Index(1))?;
//! let options = StatisticsOptions::new();
//! let context = ExecutionContext::sequential_native();
//! let request = StatisticsWorkspace::workspace_requirements(view, None, options, context)?;
//! let mut workspace = StatisticsWorkspace::workspace(&request)?;
//! let mut output = [StatisticValue::Location(0.0); 2];
//! statistics_into(view, Statistic::mean(), options, &mut output, &mut workspace, context)?;
//! assert_eq!([output[0].location(), output[1].location()], [Some(2.0), Some(5.0)]);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! For a given number of workers, parallel results do not depend on thread
//! scheduling. [`MergePolicy`] chooses whether moments are accumulated in input
//! order or in fixed partitions.
//!
//! # Feature flags
//!
//! - `std` (default): enables `std` in the SciOxide crates this crate depends on,
//!   which provides the thread executor and clock of `scioxide-core`.
//!
//! Without default features the crate is `no_std` and needs only `alloc`. The
//! API is the same.
//!
//! # References
//!
//! - R. J. Hyndman and Y. Fan, "Sample quantiles in statistical packages",
//!   *The American Statistician* 50(4), 1996, 361–365.
//! - P. McCullagh and J. A. Nelder, *Generalized Linear Models*, 2nd ed.,
//!   Chapman & Hall, 1989.
//! - B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*,
//!   Chapman & Hall, 1993.
//! - J. P. Klein and M. L. Moeschberger, *Survival Analysis*, 2nd ed.,
//!   Springer, 2003.
//! - G. E. P. Box, G. M. Jenkins, G. C. Reinsel and G. M. Ljung, *Time Series
//!   Analysis: Forecasting and Control*, 5th ed., Wiley, 2015.
//! - A. Vehtari, A. Gelman, D. Simpson, B. Carpenter and P.-C. Bürkner,
//!   "Rank-normalization, folding, and localization: an improved R̂ for
//!   assessing convergence of MCMC", *Bayesian Analysis* 16(2), 2021, 667–718.
//! - A. Saltelli et al., "Variance based sensitivity analysis of model output",
//!   *Computer Physics Communications* 181(2), 2010, 259–270.
//!
//! References for individual methods are on each item's page.
//!
//! # Immutable results
//!
//! Result types keep their fields private so that an estimate cannot be
//! changed independently of its diagnostics. Each example below assigns to a
//! private field and fails to compile.
//!
//! ```compile_fail,E0616
//! fn corrupt(mut result: scioxide_stats::FitResult<f64>) { result.parameters = None; }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut result: scioxide_stats::IsotonicResult<f64>) { result.blocks.clear(); }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut result: scioxide_stats::ClassicalTestResult<f64>) { result.interval = None; }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut result: scioxide_stats::ClassicalPermutationResult<f64>) { result.test = None; }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut result: scioxide_stats::DistributionTestResult<f64>) {
//!     result.monte_carlo_interval = None;
//! }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut result: scioxide_stats::KsPermutationResult<f64>) { result.test = None; }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut result: scioxide_stats::GoodnessOfFitResult<f64>) { result.test = None; }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut result: scioxide_stats::ResamplingResult<f64>) { result.inference = None; }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut test: scioxide_stats::ResamplingTest) { test.tail_uncertainty = None; }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut output: scioxide_stats::ResamplingOutput<f64>) { output.distribution.clear(); }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut result: scioxide_stats::ExactContingencyResult) { result.p_value_lower = 0.0; }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut report: scioxide_stats::FitReport<f64>) { report.objective = None; }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut information: scioxide_stats::FitInformation<f64>) {
//!     information.covariance = None;
//! }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut plot: scioxide_stats::ProbabilityPlot<f64>) { plot.ordered.clear(); }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut indices: scioxide_stats::SobolIndices) { indices.randomized_standard_error = None; }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut bootstrap: scioxide_stats::SobolBootstrap) { bootstrap.replicates = 0; }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut result: scioxide_stats::SobolMonteCarloResult) { result.hybrids.clear(); }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut estimate: scioxide_stats::PowerEstimate) { estimate.interval = None; }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut limits: scioxide_stats::ParameterConfidenceLimits<f64>) {
//!     limits.lower_report = None;
//! }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut report: scioxide_stats::ParameterInversionReport<f64>) { report.root = None; }
//! ```
//! ```compile_fail,E0616
//! fn corrupt(mut result: scioxide_stats::SigmaClipResult<f64>) { result.lower = None; }
//! ```

#![deny(missing_docs)]
#![deny(unreachable_pub)]
#![allow(clippy::module_name_repetitions)]
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod classical;
pub use classical::{
    BrunnerMunzelCalibration, BwsPlan, ClassicalBatchTest, ClassicalBatchWorkspace,
    ClassicalEffect, ClassicalMethod, ClassicalOptions, ClassicalPermutationResult,
    ClassicalTestResult, KendallVariant, KendallWeightCombination, MgcPlan, MgcStatistic,
    SignedRankZeros, TTestSummary, TTestVariance, TestCalibration, TestSamplePolicy, binomial_test,
    binomial_test_with_options, brunner_munzel_test, brunner_munzel_test_with_options, bws_test,
    bws_test_with_options, chatterjee_test, chatterjee_test_with_options,
    classical_permutation_test, classical_permutation_test_into, classical_tests_into,
    kendall_test, kendall_test_with_options, mann_whitney_test, mann_whitney_test_with_options,
    page_trend_test, page_trend_test_with_options, pearson_test, pearson_test_with_options,
    point_biserial_test, point_biserial_test_with_options, poisson_means_test,
    poisson_means_test_with_options, quantile_test, quantile_test_with_options, rank_sum_test,
    rank_sum_test_with_options, somers_contingency_test, somers_contingency_test_with_options,
    somers_test, somers_test_with_options, spearman_test, spearman_test_with_options,
    t_test_from_summaries, t_test_from_summaries_with_options, t_test_independent,
    t_test_independent_with_options, t_test_one_sample, t_test_one_sample_with_options,
    t_test_paired, t_test_paired_with_options, weighted_kendall_tau, wilcoxon_test,
    wilcoxon_test_with_options,
};
mod contingency;
pub use contingency::{
    AssociationMeasure, ChiSquareResult, CombinedPValue, ContingencyDesign, ContingencyView,
    CressieReadParameter, ExactContingencyResult, FalseDiscoveryMethod, FamilywiseMethod,
    PValueAdjustment, PValueAdjustmentWorkspace, PValueCombination, UnconditionalOptions,
    adjust_pvalues, adjust_pvalues_into, adjust_pvalues_with_options, barnard_exact_test,
    barnard_exact_test_with_options, boschloo_exact_test, boschloo_exact_test_with_options,
    chi2_contingency_test, combine_pvalues, contingency_association, fisher_exact_test,
    fisher_exact_test_with_options, odds_ratio, power_divergence_contingency_test,
    power_divergence_test, relative_risk,
};
mod distribution_tests;
pub use distribution_tests::{
    AndersonKVariant, DistributionBatchTest, DistributionBatchWorkspace, DistributionMethod,
    DistributionTestResult, GoodnessOfFitModel, GoodnessOfFitResult, GoodnessOfFitStatistic,
    KsPermutationResult, MedianTestOutcome, MedianTies, NullParameters, VarianceCenter,
    anderson_darling_statistic, anderson_darling_statistic_with_options, anderson_k_sample_test,
    anderson_k_sample_test_with_options, ansari_bradley_test, ansari_bradley_test_with_options,
    bartlett_test, bartlett_test_with_options, cramer_von_mises_statistic,
    cramer_von_mises_statistic_with_options, cramer_von_mises_test,
    cramer_von_mises_test_with_options, cramer_von_mises_two_sample_test,
    cramer_von_mises_two_sample_test_with_options, distribution_tests_into,
    epps_singleton_two_sample_test, epps_singleton_two_sample_test_with_options, fligner_test,
    fligner_test_with_options, goodness_of_fit_test_into, jarque_bera_test,
    jarque_bera_test_with_options, ks_one_sample_statistic, ks_one_sample_statistic_with_options,
    ks_one_sample_test, ks_one_sample_test_with_options, ks_two_sample_permutation_test,
    ks_two_sample_permutation_test_into, ks_two_sample_test, ks_two_sample_test_with_options,
    kurtosis_test, kurtosis_test_with_options, levene_test, levene_test_with_options, median_test,
    median_test_with_options, mood_scale_test, mood_scale_test_with_options, normality_test,
    normality_test_with_options, shapiro_wilk_statistic, shapiro_wilk_statistic_with_options,
    shapiro_wilk_test, shapiro_wilk_test_with_options, skew_test, skew_test_with_options,
};

#[cfg(test)]
extern crate std;

mod inference;
pub use inference::{
    ContinuityCorrection, KdeScale, LMomentNormalization, StatsTermination, SurvivalOutcome,
    TimeSeriesCentering, TrimInclusion, VariancePooling,
};

mod arithmetic;
mod error_helpers;
mod resampling;
pub use resampling::{
    BootstrapDesign, BootstrapInterval, BootstrapMethod, IndexStatistic, IndexedReplicates,
    JackknifeEstimate, NullSampler, PermutationKind, PermutationMode, ReplicateExecution,
    ReplicateOptions, ResampledIndices, ResamplingInference, ResamplingMethod, ResamplingOptions,
    ResamplingOutput, ResamplingPlan, ResamplingProblem, ResamplingResult, ResamplingTest,
    ResamplingWorkspace, SampleStatistic, SequentialReplicates, TestAlternative,
    WorkspaceIndexStatistic, WorkspaceSampleStatistic, bootstrap, jackknife, monte_carlo_test,
    permutation_test,
};
mod power;
pub use power::{
    MeanPowerDesign, MeanPowerModel, MeanPowerReference, PowerEstimate, SampleSizeResult, power,
    sample_size, simulated_power, simulated_power_into,
};
mod sensitivity;
pub use sensitivity::{
    SobolBootstrap, SobolEvaluations, SobolIndices, SobolMonteCarloResult, sobol_bootstrap,
    sobol_bootstrap_scoped, sobol_indices, sobol_indices_monte_carlo,
    sobol_indices_monte_carlo_scoped, sobol_indices_qmc, sobol_indices_qmc_scoped,
};
mod posterior;
pub use posterior::{
    PosteriorOptions, PosteriorSummary, posterior_summaries_into, posterior_summary,
    posterior_summary_with_options,
};
mod parameter_inversion;
pub use parameter_inversion::{
    IntegerParameterInversionRequest, IntegerParameterReport, IntegerParameterSolution,
    IntegerProbabilityParameter, ParameterConfidenceLimits, ParameterInversionReport,
    ParameterInversionRequest, ParameterMonotonicity, ParameterProbability, ParameterSolution,
    ProbabilityParameter, binomial_confidence_limits, binomial_confidence_limits_scoped,
    binomial_confidence_limits_with_options, invert_confidence_limits,
    invert_confidence_limits_scoped, invert_confidence_limits_with_options,
    invert_integer_parameter, invert_integer_parameter_scoped,
    invert_integer_parameter_with_options, invert_integer_parameters_into, invert_parameter,
    invert_parameter_scoped, invert_parameter_with_options, invert_parameters_into,
    noncentrality_confidence_limits, noncentrality_confidence_limits_scoped,
    noncentrality_confidence_limits_with_options, normal_variance_confidence_limits,
    normal_variance_confidence_limits_scoped, normal_variance_confidence_limits_with_options,
    poisson_rate_confidence_limits, poisson_rate_confidence_limits_scoped,
    poisson_rate_confidence_limits_with_options,
};
mod conjugate;
pub use conjugate::{
    BetaBinomialPrior, CredibleInterval, CredibleSummary, GammaPoissonPrior, NormalCredibleSummary,
    NormalInverseGammaPrior, NormalPosteriorMarginals, NormalPosteriorRequest, credible_interval,
    normal_credible_summary, normal_posterior_marginals, normal_posteriors_into,
};
mod group_summary;
mod regression;
pub use regression::{
    GlmAccepted, GlmBatchInputs, GlmBatchOutput, GlmBatchWorkspace, GlmCoefficientInference,
    GlmCovariance, GlmFamily, GlmFit, GlmFitView, GlmInputs, GlmLink, GlmObjective,
    GlmOperatorWorkspace, GlmOptions, GlmOutput, GlmReport, GlmResult, GlmSolver,
    GlmStorageOptions, GlmWorkspace, GroupTest, GroupTestMethod, HuberAccepted, HuberFit,
    HuberFitView, HuberOptions, HuberOutput, HuberReport, HuberWorkspace, LinearRegression,
    LinearRegressionFit, LinearRegressionOptions, OperatorRegressionMethod,
    OperatorRegressionOptions, OperatorRegressionReport, OperatorRegressionResult,
    OperatorRegressionWorkspace, PostHocComparison, PostHocMethod, PostHocOptions, PostHocResult,
    PreparedGlm, PreparedGroups, PreparedHuber, PreparedOperatorGlm, PreparedOperatorRegression,
    PreparedRegression, RegressionContrast, RegressionCovariance, RegressionFactorization,
    RegressionWorkspace, RobustIntercept, RobustSlope, RobustSlopeMethod, RobustSlopeWorkspace,
    friedman_test, friedman_test_with_options, glm_batch_into, group_tests_into, huber_location,
    huber_location_with_options, kruskal_wallis_test, kruskal_wallis_test_with_options,
    linear_regression, regression_batch_into, robust_slope, robust_slope_with_options,
};
mod batch;
mod central;
mod confidence;
mod covariance;
mod descriptive;
mod empirical;
mod isotonic;
mod isotonic_axis;
mod moments;
mod order;
mod order_axis;
mod ranking;
mod reduction;
mod weighted;
mod weighted_covariance;

pub use central::{BiasCorrection, KurtosisConvention, MomentOrder};

pub use confidence::{
    ConfidenceProbability, MeanConfidenceInterval, MeanIntervalAssumptions, MeanIntervalMethod,
    MeanIntervalPlan,
};

pub use covariance::{
    CorrelationValidity, CovarianceSnapshot, CovarianceValidity, OnlineCovariance,
};

pub use empirical::{
    QuantileConvention, QuantileMethod, QuantileProbability, QuantileResult, TailMeanConvention,
    TailMeanResult, TailSide,
};
pub(crate) use empirical::{
    RawEmpiricalDistribution, RawWeightedEmpiricalDistribution, frequency_quantile_in_place,
    weighted_inverse_ecdf_quantile_in_place,
};
pub use moments::{MomentsSnapshot, OnlineMoments};

pub use reduction::{MergePolicy, MissingPolicy, NonFinitePolicy, TrimLimits, WeightSemantics};

pub use weighted::{OnlineWeightedMoments, WeightedMomentsSnapshot};

pub use weighted_covariance::{OnlineWeightedCovariance, WeightedCovarianceSnapshot};

pub use order::{ExpectileResult, ModeResult, OrderResult, PercentileKind, Rank, RankMethod};
pub(crate) use order::{OrderedStatistic, RawOrderWorkspace, rankdata};

pub(crate) use order_axis::{KernelWeights, OrderData, OrderPolicy};

pub(crate) use isotonic::RawIsotonicWorkspace;
pub use isotonic::{IsotonicBlock, IsotonicDiagnostics, IsotonicDirection, IsotonicResult};

pub(crate) use isotonic_axis::IsotonicWorkspaceRequirements;
pub use isotonic_axis::{IsotonicBatchOutput, IsotonicLaneResult};
pub(crate) use isotonic_axis::{IsotonicPolicy, RawIsotonicBatchWorkspace};

mod binned;
mod estimation;
pub(crate) use binned::BinnedWorkspaceRequirements;
pub(crate) use binned::RawBinnedWorkspace;
pub use binned::{
    BinEstimate, BinLocation, BinPlan, BinnedDiagnostics, BinnedResult, BinnedStatistic,
};
pub(crate) use estimation::ObservationData;
mod cdf;
pub(crate) use cdf::RawCdfWorkspace;
pub(crate) use cdf::{CdfWorkspaceRequirements, RawCdfMethod};
mod kde;
pub(crate) use kde::KdeWorkspaceRequirements;
pub use kde::{KdeBandwidth, KdeCovariance, KdeOptions};
pub(crate) use kde::{RawGaussianKde, RawKdeWorkspace};

mod robust;
pub(crate) use robust::RobustStatistic;
pub use robust::TrimSide;
mod descriptive_workspace;
pub(crate) use descriptive_workspace::RawDescriptiveWorkspace;
mod standardize;
pub use standardize::{SigmaClipOptions, SigmaClipResult, StandardizationWorkspace, standardize};

mod circular;
pub(crate) use circular::circular_statistics;
pub use circular::{
    CircularOptions, CircularSummary, DirectionNormalization, DirectionalSummary,
    DirectionalWorkspace,
};
mod information;
pub use information::{
    DifferentialEntropyMethod, DifferentialEntropyOptions, EntropyOptions, InformationValue,
    MassNormalization, differential_entropy, differential_entropy_with_options, entropy,
    entropy_with_options, relative_entropy,
};

mod distances;
pub use distances::{
    DistanceWorkspace, DistributionDistance, distribution_distance,
    distribution_distance_with_options,
};

mod transport;
pub use transport::{
    TransportCandidate, TransportOptions, TransportProblem, TransportResult, TransportSolver,
    TransportWorkspace,
};

mod descriptive_axes;
mod rolling;
mod rolling_median;
pub use descriptive_axes::{RollingBoundary, RollingWindow};

mod numeric_statistics;
pub(crate) use numeric_statistics::{ArithmeticStatistic, ArithmeticValue, numeric_statistic};

mod fitting;
pub use fitting::{
    DistributionFitModel, ExactFitDerivatives, FitDerivativeCapabilities, FitDerivatives,
    FitIdentifiability, FitInformation, FitMethod, FitModel, FitObjective, FitObservation,
    FitReport, FitRequest, FitResult, FitSolver, FitSolverCapabilities, FitTruncation,
    FitWorkspace, LawFitModel, PowerFitMethod, PowerTransform, PreparedFit, PreparedPowerFit,
    PreparedPpcc, ProbabilityPlot, ProbabilityPlotFit, fit, fit_information, fit_into,
    fit_objectives_into, fit_scoped, fit_with_options, power_transform_into, power_transform_llf,
    power_transform_normmax, power_transform_normmax_scoped, power_transform_normmax_with_options,
    ppcc_max, ppcc_max_scoped, ppcc_max_with_options, probplot, probplot_with_options,
};

mod survival;
pub use survival::{
    CoxBaselinePoint, CoxFit, CoxInferenceOptions, CoxReport, CoxTies, KaplanMeierPoint,
    LogrankResult, NelsonAalenPoint, PreparedCox, SchoenfeldResidual, SurvivalEstimator,
    SurvivalObservation, SurvivalOptions, SurvivalRequirements, SurvivalWorkspace, cox_fit,
    cox_fit_scoped, cox_fit_with_options, kaplan_meier, kaplan_meier_into,
    kaplan_meier_with_options, logrank_test, logrank_test_with_options, nelson_aalen,
    nelson_aalen_into, nelson_aalen_with_options,
};

mod multivariate;
mod odr;
mod scatter;
mod time_series;
pub use multivariate::{
    CcaWorkspace, MultivariateOptions, MultivariateRequirements, PcaScaling, PcaWorkspace,
    PreparedCca, PreparedPca, ledoit_wolf_covariance, ledoit_wolf_covariance_with_options,
    sample_covariance, sample_covariance_with_options, shrink_covariance,
};

pub use odr::{
    AffineOdrModel, OdrFit, OdrIdentifiability, OdrIdentifiabilityDiagnostic, OdrInputs, OdrModel,
    OdrOptions, OdrOutput, OdrPredictors, OdrReport, OdrWorkspace, PreparedOdr, linear_odr,
    linear_odr_with_options,
};

pub use scatter::{
    HuberScatterAccepted, HuberScatterFit, HuberScatterOptions, HuberScatterOutput,
    HuberScatterReport, HuberScatterWorkspace, PreparedHuberScatter,
};

pub use time_series::{
    ArimaFitReport, ArimaForecast, ArimaInnovation, ArimaLikelihood, ArimaModel, ArimaOrder,
    ArimaTrend, ArimaWorkspace, LagNormalization, PeriodogramBin, PeriodogramDiagnostics,
    PeriodogramWorkspace, PreparedArimaFit, TimeSeriesLag, TimeSeriesOptions,
    TimeSeriesRequirements, TimeSeriesStatistic, TimeSeriesWorkspace, periodogram_diagnostics_into,
    time_series_summaries_into, time_series_summary, time_series_summary_into,
    time_series_summary_with_options,
};

mod execution;
mod options;
mod view;
mod weights;
pub use weights::{
    Analytic, BatchWeights, ConjugateWeights, EmpiricalWeights, Frequency, IsotonicWeights, Mass,
    RegressionWeights, Reliability, Unweighted, WeightModel, Weights,
};

pub use view::{Axis, BatchView};

pub use options::StatisticsOptions;

mod summary;
pub use summary::{
    maximum, mean, mean_confidence_interval, median, minimum, quantile, sample_standard_deviation,
    sample_variance,
};
mod statistic;
pub use statistic::{
    ExpectileWeights, InterpolatedQuantiles, LocationModel, QuantileRule, Statistic,
    StatisticCapability, StatisticValue, statistic,
};

pub use statistics_workspace::{rolling_statistics, rolling_statistics_with_options};
pub use statistics_workspace::{statistics, statistics_with_options};
mod statistics_workspace;
pub use statistics_workspace::{
    StatisticsRequirements, StatisticsWorkspace, rank_axes_into, rank_slice_into,
    rolling_statistics_into, standardize_axes_into, statistics_into,
};

pub(crate) use isotonic::RawIsotonicOptions;

mod estimator_api;
pub use estimator_api::{
    BinnedRequirements, BinnedWorkspace, CdfMethod, CdfRequirements, CdfWorkspace, GaussianKde,
    KdeRequirements, KdeWorkspace, binned_statistics, binned_statistics_into, cdf, cdf_into,
    cdf_with_options, kde_into,
};

pub use estimator_api::{
    IsotonicRequirements, IsotonicWorkspace, isotonic_regression, isotonic_regression_axes_into,
    isotonic_regression_into, isotonic_regression_with_options,
};

mod quantile_api;
pub use quantile_api::{
    EmpiricalDistribution, SelectionSamples, TailWeights, quantile_in_place, tail_mean_in_place,
};

mod reduction_api;
pub use reduction_api::{
    MomentWeights, ReductionRequirements, ReductionWorkspace, reduce_covariance,
    reduce_covariance_into, reduce_covariance_with_options, reduce_moment, reduce_moment_into,
    reduce_moment_with_options, reduce_moments, reduce_moments_into, reduce_moments_with_options,
    reduce_trimmed_moments, reduce_trimmed_moments_into, reduce_trimmed_moments_with_options,
    reduce_weighted_covariance, reduce_weighted_covariance_into,
    reduce_weighted_covariance_with_options, reduce_weighted_moments, reduce_weighted_moments_into,
    reduce_weighted_moments_with_options,
};

pub use circular::DirectionalRequirements;
pub use estimator_api::{
    directional_statistics_into, frequency_covariance, frequency_covariance_with_options,
};

pub use statistics_workspace::StatisticsGeometry;

pub use distances::DistanceRequirements;
pub use standardize::StandardizationRequirements;

pub use statistics_workspace::rank_slice;

pub use statistics_workspace::sigma_clip_into;

pub use binned::{Histogram, histogram};
pub use regression::{glm_fit, ordinary_least_squares};

pub use transport::{wasserstein_distance_nd, wasserstein_distance_nd_scoped};
