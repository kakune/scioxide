//! Quadratic programming solvers.
//!
//! This module solves convex QPs of the form:
//!
//! Minimize `0.5 x^T Q x + c^T x` subject to `A_ub x <= b_ub`,
//! `A_eq x = b_eq`, and optional variable bounds.
//!
//! # Solver
//!
//! - [`QpSolver`]: active-set, projected-gradient or primal-dual method for dense QP.

mod active_factor;
mod active_set;
mod common;
mod model;
mod projected;
mod qp_solver;
mod unconstrained;

pub use qp_solver::{QpSolver, QpWorkspace};

mod primal_dual;
/// Selects the method of [`QpSolver`] for problems with constraints or bounds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QpMethod {
    /// Projected gradient with Barzilai–Borwein steps for bound-only problems,
    /// and the active-set method otherwise.
    #[default]
    Automatic,
    /// The active-set method, also for bound-only problems.
    ActiveSet,
    /// Primal-dual predictor-corrector interior-point steps on the augmented
    /// KKT system.
    PrimalDual,
}

/// Holds an initial primal point for [`QpSolver::with_warm_start`].
///
/// Feasibility is checked against each problem when it is solved.
#[derive(Debug, Clone)]
pub struct QpWarmStart<S: scioxide_core::RealScalar> {
    pub(crate) point: alloc::vec::Vec<S>,
}
impl<S: scioxide_core::RealScalar> QpWarmStart<S> {
    /// Creates a warm start from `point`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `point` is empty or contains NaN or infinity.
    pub fn new(point: alloc::vec::Vec<S>) -> Result<Self, scioxide_core::SciOxideError> {
        if point.is_empty() || point.iter().any(|v| !v.is_finite()) {
            return Err(scioxide_core::SciOxideError::invalid_input(
                scioxide_core::argument!("point"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "point must be finite and nonempty",
            ));
        }
        Ok(Self { point })
    }
    /// Returns the initial primal point.
    pub fn point(&self) -> &[S] {
        &self.point
    }
}
