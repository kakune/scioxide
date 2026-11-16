# scioxide-stats

Statistics for Rust.
The crate summarizes samples with moments, quantiles, ranks, robust and circular statistics, and estimates distributions with empirical CDFs, kernel densities, histograms and isotonic fits.
It tests hypotheses with classical, rank, distribution, contingency-table, resampling and Monte Carlo methods, and fits linear, generalized linear, robust, errors-in-variables, survival and ARIMA models.
Data are `f32` or `f64` slices or strided arrays with optional missing-value masks and typed weights.

## Main entry points

- Summaries: `mean`, `sample_variance`, `median`, `quantile`, `statistic`, `statistics`, `rolling_statistics`, `rank_slice`, `standardize`, `OnlineMoments`, `OnlineCovariance`, `reduce_moments`, `mean_confidence_interval`.
- Data and weights: `BatchView`, `Axis`, `Weights`, `Frequency`, `Reliability`, `Analytic`, `Mass`, `StatisticsOptions`.
- Distributions: `EmpiricalDistribution`, `quantile_in_place`, `cdf`, `GaussianKde::new`, `histogram`, `BinPlan`, `isotonic_regression`, `entropy`, `distribution_distance`, `TransportProblem`.
- Tests: `t_test_one_sample`, `t_test_independent`, `mann_whitney_test`, `wilcoxon_test`, `ks_two_sample_test`, `shapiro_wilk_test`, `levene_test`, `chi2_contingency_test`, `fisher_exact_test`, `adjust_pvalues`, `combine_pvalues`.
- Resampling: `ResamplingPlan`, `bootstrap`, `jackknife`, `permutation_test`, `monte_carlo_test`.
- Models: `ordinary_least_squares`, `glm_fit`, `linear_regression`, `PreparedRegression`, `PreparedGlm`, `PreparedHuber`, `robust_slope`, `PreparedGroups`, `fit`, `PowerTransform`, `PreparedPca::new`, `PreparedCca::new`, `linear_odr`, `PreparedArimaFit`.
- Survival: `kaplan_meier`, `nelson_aalen`, `logrank_test`, `cox_fit`.
- Bayesian and design: `BetaBinomialPrior`, `NormalInverseGammaPrior`, `posterior_summary`, `invert_parameter`, `power`, `sample_size`, `simulated_power`, `sobol_indices`.

Configured calls use `*_with_options`; reusable models expose owned and scoped execution, and `*_into` forms write caller storage.
Random one-shots take an explicit generator.
Bootstrap uses BCa by default; other intervals use a plan.

## Feature flags

- `std` (default): enables `std` in the SciOxide dependencies.
  Without it the crate is `no_std` and needs `alloc`; the API is the same.

See the [API documentation](https://docs.rs/scioxide-stats) for a quick start, a guide to choosing a method, and the rules for weights, termination and errors.

## License

Original code, examples and independently computed reference data are licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
Reference-tool licenses do not themselves license computed numbers; copied inputs retain their applicable upstream conditions.
