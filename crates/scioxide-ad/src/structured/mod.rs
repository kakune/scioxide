//! Structured operations: products, solves, factorizations, normalizations,
//! implicit functions and checkpointed loops.

mod factor;
mod implicit;
mod linear;
mod loops;
mod normalize;
mod solve;

pub use factor::{cholesky_into, least_squares_into, qr_into};
pub use implicit::{Residual, implicit_solution_into};
pub use linear::{dot, gemm_into, gemv_into};
pub use loops::{LoopBody, checkpointed_loop};
pub use normalize::{log_softmax_into, logsumexp, softmax_into};
pub use solve::solve_into;
pub(crate) use solve::{Factor, subtract_product};
