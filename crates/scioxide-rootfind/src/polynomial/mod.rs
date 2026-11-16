//! Roots of polynomials in the monomial basis or an orthogonal basis.
mod api;
mod options;
mod pipeline;
pub use api::{
    PolynomialRootInput, polynomial_roots, polynomial_roots_into, polynomial_roots_with_options,
};
pub use options::PolynomialRootOptions;
pub use pipeline::{
    PolynomialRootAlgorithm, PolynomialRootDiagnostic, PolynomialRootMethod, PolynomialRootReport,
    PolynomialRootWorkspace, PolynomialRootWorkspaceRequirements, PolynomialRoots,
};
