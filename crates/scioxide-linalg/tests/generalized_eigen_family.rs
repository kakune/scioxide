//! generalized eigen family: numerical, resource and transaction contracts.

mod support;

#[path = "families/generalized_eigen.rs"]
mod generalized_eigen;
#[path = "families/generalized_hermitian_reference.rs"]
mod reference;

#[path = "families/generalized_eigen_workspace_contracts.rs"]
mod generalized_eigen_workspace_contracts;
