//! Structured direct linear-system representations and factorizations.

pub(crate) mod tridiagonal;

pub use tridiagonal::{
    CyclicTridiagonalPlan, TridiagonalLuDecomposition, TridiagonalMatrix, TridiagonalMatrixView,
};

pub(crate) mod direct;
pub(crate) use direct::{admit, identity};
mod storage;
mod triangular;
pub use direct::{StructuredOperation, StructuredWorkspaceRequest};
pub use storage::{
    BandedMatrix, BandedMatrixView, HermitianBandMatrix, HermitianBandMatrixView, PackedKind,
    PackedMatrix, PackedMatrixView, PackedMatrixViewMut,
};
pub use triangular::TriangularPlan;
mod band;
pub use band::{BandedCholeskyDecomposition, BandedLuDecomposition};

mod constructors;
pub use constructors::{
    CompanionKind, ConvolutionMode, DftDirection, DftNormalization, HelmertKind, MatrixForm,
    PascalKind, StructuredMatrix, StructuredMatrixView,
};

mod displacement;
pub use displacement::{StructuredSolveOptions, StructuredSolvePlan, StructuredSolveReport};

mod products;
pub use products::{
    BlockDiagonalOperator, CompositeSolveOptions, CompositeSolveReport, KroneckerOperator,
    khatri_rao, khatri_rao_into, khatri_rao_with_options,
};

mod utilities;
pub use utilities::{
    MatrixNorm, OperatorWorkspaceRequest, apply_operator_matrix_into, materialize_operator_into,
    operator_norm, operator_norm_scoped, operator_norm_with_options,
};
