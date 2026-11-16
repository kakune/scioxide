//! law contracts test suite.

#[path = "families/continuous_acceptance.rs"]
mod continuous_acceptance;
#[path = "families/continuous_reference.rs"]
mod continuous_reference;
#[path = "families/distribution_cdf_kernels.rs"]
mod distribution_cdf_kernels;
#[path = "families/law_contract.rs"]
mod law_contract;
#[path = "families/law_contracts_workflows.rs"]
mod law_contracts_workflows;
#[path = "families/law_options.rs"]
mod law_options;
#[path = "families/law_policies.rs"]
mod law_policies;
#[path = "families/quantile_accuracy.rs"]
mod quantile_accuracy;

#[path = "families/quantile_acceleration.rs"]
mod quantile_acceleration;

#[path = "families/direct_sampling.rs"]
mod direct_sampling;

#[path = "families/scalar_evaluation.rs"]
mod scalar_evaluation;

#[path = "support/statistics.rs"]
mod statistics;

#[path = "families/range_safe.rs"]
mod range_safe;

#[path = "families/moment_cancellation.rs"]
mod moment_cancellation;

#[path = "families/error_taxonomy.rs"]
mod error_taxonomy;
