//! execution family: numerical, resource and transaction contracts.

mod support;

#[path = "families/accuracy_tiers.rs"]
mod accuracy_tiers;
#[path = "families/context_contracts.rs"]
mod context_contracts;
#[path = "families/edge_contracts.rs"]
mod edge_contracts;
#[path = "families/execution_options.rs"]
mod execution_options;
#[path = "families/native_dispatch.rs"]
mod native_dispatch;
#[path = "families/options_composition.rs"]
mod options_composition;
#[path = "families/plain_forms.rs"]
mod plain_forms;
#[path = "families/reports.rs"]
mod reports;

#[path = "families/operator_dispatch.rs"]
mod operator_dispatch;
