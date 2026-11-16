//! householder family: numerical, resource and transaction contracts.

mod support;

#[path = "families/householder.rs"]
mod householder;
#[path = "families/householder_pivoting.rs"]
mod householder_pivoting;

#[path = "families/householder_allocations.rs"]
mod householder_allocations;

#[path = "families/householder_workspace_contracts.rs"]
mod householder_workspace_contracts;
