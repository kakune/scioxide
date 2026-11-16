//! schur family: numerical, resource and transaction contracts.

mod support;

#[path = "families/schur_reference.rs"]
mod reference;
#[path = "families/schur_invariants.rs"]
mod schur_invariants;
#[path = "families/schur_reordering.rs"]
mod schur_reordering;

#[path = "families/schur_subspace_contracts.rs"]
mod schur_subspace_contracts;
