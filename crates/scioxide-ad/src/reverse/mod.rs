//! Reverse mode: variables recorded on a caller-owned tape.

mod call;
mod external;
mod linearization;
mod loops;
mod operator;
mod pattern;
mod recorder;
mod second;
mod session;
mod solve;
mod sweep;
mod tape;
#[cfg(test)]
mod tests;
mod transforms;
mod value;
mod var;
mod variables;
mod word;

pub use external::ExternalRule;
pub use linearization::Linearization;
pub use operator::{HessianOperator, JacobianOperator};
pub use tape::{Tape, TapeOptions, TapeReport, TapeRequirements};
pub use transforms::{
    grad, hessian, hvp, jacobian, sparse_hessian, sparse_jacobian, value_and_grad, vjp,
};
pub use value::TapeValue;
pub use var::Var;
