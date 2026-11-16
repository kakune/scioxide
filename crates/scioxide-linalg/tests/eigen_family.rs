//! eigen family: numerical, resource and transaction contracts.

mod support;

#[path = "families/typed_eigen.rs"]
mod typed_eigen;

#[path = "families/eigen_workspaces.rs"]
mod eigen_workspaces;

#[path = "families/eigen_contracts.rs"]
mod eigen_contracts;

#[path = "families/eigen_action_workspaces.rs"]
mod eigen_action_workspaces;

#[path = "families/first_components.rs"]
mod first_components;
