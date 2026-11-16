//! Sylvester, Lyapunov and algebraic Riccati equations.
//!
//! Sylvester and Lyapunov plans use the Bartels–Stewart method on Schur forms.
//! Riccati plans compute a stable deflating subspace of a matrix pencil with
//! the QZ algorithm.
pub(crate) mod blocked;
mod linear;
mod riccati;

pub use linear::{
    LyapunovKind, LyapunovPlan, MatrixEquationOptions, MatrixEquationReport, MatrixEquationResult,
    SylvesterPlan, SylvesterSign,
};
pub use riccati::{
    RiccatiBalancing, RiccatiKind, RiccatiOptions, RiccatiPlan, RiccatiProblem, RiccatiReport,
    RiccatiResult,
};
