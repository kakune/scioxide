//! svd family: numerical, resource and transaction contracts.

mod support;

#[path = "families/svd_typed.rs"]
mod svd_typed;

#[path = "families/svd_allocations.rs"]
mod svd_allocations;

#[path = "families/svd_workspace_contracts.rs"]
mod svd_workspace_contracts;

#[path = "families/svd_action_workspaces.rs"]
mod svd_action_workspaces;
