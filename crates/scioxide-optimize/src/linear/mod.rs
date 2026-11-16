//! Linear programming solvers.
//!
//! This module provides two LP solvers:
//!
//! - [`SimplexSolver`]: bounded two-phase revised simplex.
//! - [`InteriorPointLp`]: homogeneous self-dual interior-point method.
//!
//! Both solve an [`LpProblem`](crate::problem::LpProblem) and return a
//! [`ProgrammingResult`](crate::ProgrammingResult).

mod basis;
mod certificate;
mod hsd;
mod interior_point_lp;
mod mip;
mod simplex;
mod standard_form;

pub use interior_point_lp::{InteriorPointLp, InteriorPointLpWorkspace};
pub use mip::MipSimplexWorkspace;
pub use simplex::{SimplexSolver, SimplexWarmStart, SimplexWorkspace};

pub use basis::{BasisMatrix, BasisProvider, DenseBasisProvider, SparseBasisProvider};
