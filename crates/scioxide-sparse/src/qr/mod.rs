//! Sparse Householder QR in a fixed column order, and minimum-norm solves.
use crate::helpers::{admit, invalid};
use alloc::vec::Vec;
use core::mem::size_of;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::standard::finite_candidate;
use scioxide_core::backend::{
    checked_add as add, checked_mul as mul, try_filled, try_with_capacity,
};
use scioxide_core::{
    Argument, BlockVectorView, BlockVectorViewMut, ExecutionContext, NormSelection, NumericScalar,
    Permutation, RankThreshold, RealScalar, ResidualReport, ScaledNorm, SciOxideError, SolveScope,
    StructuralPattern, StructuralPatternView,
};

use crate::{SparseOperation, SparseSolveOptions, SparseView};

mod numeric;
mod solve;
mod symbolic;
#[cfg(test)]
mod tests;

use numeric::{Data, Kernel, apply_q, apply_q_operation, factor_kernel};
pub use numeric::{SparseQr, SparseQrWorkspace};
pub use solve::{SparseQrMinimumNormPlan, SparseQrSolveReport};
pub use symbolic::{SparseQrLimits, SparseQrRequirements, SparseQrSymbolic};
