//! vectors family: numerical, resource and transaction contracts.

mod support;

#[path = "families/affine_values.rs"]
mod affine_values;
#[path = "families/copy_policy.rs"]
mod copy_policy;
#[path = "families/reduction_allocations.rs"]
mod reduction_allocations;
#[path = "families/vector_substitution.rs"]
mod vector_substitution;

#[path = "families/vectors_workspace_contracts.rs"]
mod vectors_workspace_contracts;
