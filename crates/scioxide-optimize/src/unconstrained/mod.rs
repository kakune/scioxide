//! Local minimizers for unconstrained and box-constrained problems.
//!
//! All solvers implement [`crate::Optimizer`] and minimize a
//! [`crate::ObjectiveFunction`].
//!
//! | Solver | Method |
//! |---|---|
//! | [`GradientDescent`] | Steepest descent with a line search |
//! | [`NonlinearCg`] | Polak–Ribière+ nonlinear conjugate gradient |
//! | [`Bfgs`] | BFGS with a dense inverse-Hessian approximation |
//! | [`Lbfgs`] | Limited-memory BFGS |
//! | [`LbfgsB`] | Projected limited-memory BFGS for bound constraints |
//! | [`NewtonOptimizer`] | Newton's method with Hessian modification |
//! | [`TrustRegionOptimizer`] | Dogleg or Steihaug-CG trust-region method |
//! | [`Powell`] | Powell's conjugate-direction method, with optional bounds |
//! | [`NelderMead`] | Nelder–Mead simplex method |

mod bfgs;
mod first_order;
mod gradient_descent;
mod lbfgs;
mod lbfgsb;
mod nelder_mead;
mod newton;
mod nonlinear_cg;
mod powell;
mod second_order;
mod settings;
pub(crate) use settings::iteration_budgets;
mod trust_region;
pub use powell::{
    Powell, PowellDiagnostics, PowellOptions, PowellWorkspace, PowellWorkspaceRequirements,
};

pub use bfgs::{Bfgs, BfgsCheckpoint, BfgsOptions, BfgsWorkspace, BfgsWorkspaceRequirements};
pub use gradient_descent::{
    GradientDescent, GradientDescentOptions, GradientDescentWorkspace,
    GradientDescentWorkspaceRequirements,
};
pub use lbfgs::{Lbfgs, LbfgsOptions, LbfgsWorkspace, LbfgsWorkspaceRequirements};
pub use lbfgsb::{LbfgsB, LbfgsBOptions, LbfgsBWorkspace, LbfgsBWorkspaceRequirements};
pub use nelder_mead::{
    NelderMead, NelderMeadDiagnostics, NelderMeadGeometry, NelderMeadOptions, NelderMeadWorkspace,
    NelderMeadWorkspaceRequirements,
};
pub use newton::{
    NewtonOptimizer, NewtonOptimizerWorkspace, NewtonOptimizerWorkspaceRequirements, NewtonOptions,
};
pub use nonlinear_cg::{
    NonlinearCg, NonlinearCgOptions, NonlinearCgWorkspace, NonlinearCgWorkspaceRequirements,
};
pub use trust_region::{
    TrustRegionOptimizer, TrustRegionOptimizerWorkspace, TrustRegionOptimizerWorkspaceRequirements,
    TrustRegionOptions, TrustRegionSubproblem, TrustRegionWorkspaceRequest,
};

mod newton_cg;
pub use newton_cg::{NewtonCgTermination, NewtonDiagnostics};

pub(crate) use first_order::FirstOrderStorage;
pub(crate) use second_order::{SecondOrderShape, SecondOrderStorage};
