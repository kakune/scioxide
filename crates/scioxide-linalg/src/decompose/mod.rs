//! Dense matrix factorizations.
//!
//! - [`LuDecomposition`]: LU with partial pivoting.
//! - [`CholeskyDecomposition`]: Cholesky of Hermitian positive definite or
//!   semidefinite matrices.
//! - [`LdlDecomposition`]: `L D Lᴴ` of Hermitian indefinite matrices with rook
//!   pivoting, and inertia.
//! - [`HouseholderDecomposition`]: QR and LQ with optional pivoting, with
//!   [`HouseholderMinimumNormPlan`] and [`QrUpdateDecomposition`].
//! - [`RqDecomposition`]: RQ factorization.
//! - [`SvdDecomposition`], [`ThinSvdDecomposition`], [`SvdValues`]: SVD via
//!   Golub–Kahan bidiagonalization.
//! - [`PolarDecomposition`], [`CsdDecomposition`], [`GsvdDecomposition`]: polar,
//!   cosine–sine and generalized singular value decompositions.

pub(crate) mod hermitian;
pub use hermitian::{HermitianInput, HermitianProjectionPolicy, HermitianProjectionReport};
mod cholesky;
mod cs_kernel;
mod gsvd;
pub use gsvd::{
    GeneralizedSingularValue, GsvdDecomposition, GsvdOptions, GsvdReport, GsvdWorkspaceRequest,
};
mod householder;
mod minimum_norm;
pub use minimum_norm::HouseholderMinimumNormPlan;
pub(crate) mod lu;
mod qr;
mod svd;
pub(crate) mod util;

pub use cholesky::{
    CholeskyDecomposition, CholeskyDefinitenessReport, CholeskyInverseOperator, CholeskyPivoting,
    CholeskyWorkspaceRequest,
};
pub use householder::{
    HouseholderDecomposition, HouseholderOrientation, HouseholderPivoting,
    HouseholderWorkspaceRequest,
};
pub use lu::{LuDecomposition, LuInverseOperator, LuWorkspaceRequest, SignLogDeterminant};
pub(crate) use lu::{LuKernelFactor, factor_kernel, factor_kernel_lengths};
pub(crate) use svd::solve::{SpectralView, filtered as filtered_svd_coefficient};
pub use svd::{SvdDecomposition, SvdValues, SvdWorkspaceRequest, ThinSvdDecomposition};
pub(crate) use svd::{
    full_candidate, spectrum_condition_estimate, spectrum_rank, thin_candidate, values_candidate,
};

mod ldl;
pub use ldl::{LdlDecomposition, LdlInverseOperator, LdlWorkspaceRequest};

mod polar;
pub use polar::{PolarDecomposition, PolarOptions, PolarReport, PolarSide, PolarWorkspaceRequest};

mod csd;
pub use csd::{CsdDecomposition, CsdReport, CsdWorkspaceRequest};

mod rq;
pub use rq::{RqDecomposition, RqWorkspaceRequest};
mod qr_update;
pub use qr_update::{
    QrModification, QrRankDecision, QrRankSource, QrUpdateDecomposition, QrUpdateOptions,
    QrUpdateReport, QrUpdateWorkspaceRequest, TriangularRankWorkspace,
};
