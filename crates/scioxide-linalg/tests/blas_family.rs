//! blas family: numerical, resource and transaction contracts.

mod support;

#[path = "families/blas_level2_coverage.rs"]
mod blas_level2_coverage;
#[path = "families/blas_level3_coverage.rs"]
mod blas_level3_coverage;
#[path = "families/blas_provider.rs"]
mod blas_provider;
#[path = "families/blas_scalar_domains.rs"]
mod blas_scalar_domains;
