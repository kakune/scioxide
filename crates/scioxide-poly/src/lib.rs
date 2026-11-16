//! Polynomials: exact algebra, floating-point polynomials in several bases,
//! least-squares fitting, elimination and partial fractions.
//!
//! The crate has two coefficient domains. [`ExactPolynomial`] and
//! [`ExactRationalFunction`] compute without rounding over a ring or field you
//! supply through [`ExactCoefficient`] and [`ExactField`], such as the rationals
//! or a prime field; floating-point types do not implement these traits.
//! [`ApproxPolynomial`] holds real or complex floating-point coordinates in an
//! explicit basis: monomial, Newton, barycentric (nodal values), Bernstein on an
//! interval, or a classical orthogonal family. Every floating-point result comes
//! with a backward-error and conditioning estimate. The crate also builds
//! Vandermonde matrices, fits noisy samples, computes approximate GCDs,
//! resultants and discriminants, decomposes rational functions into partial
//! fractions, and evaluates tensor-product Bernstein patches and sparse
//! multivariate polynomials.
//!
//! # Quick start
//!
//! ```
//! use scioxide_poly::ApproxPolynomial;
//!
//! // p(x) = 1 + 2x + 3x², so p(0.5) = 2.75 and p′(0.5) = 2 + 6·0.5 = 5.
//! let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 3.0])?;
//! assert!((p.evaluate(0.5)? - 2.75).abs() < 1e-14);
//! let result = p.evaluate_with_derivative(0.5)?;
//! assert!((result.derivative() - 5.0).abs() < 1e-14);
//! assert!(result.condition().forward_error_estimate() < 1e-14);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! | Problem | Entry points | Method |
//! |---|---|---|
//! | Arithmetic, division and derivatives of a polynomial given by coefficients | [`ApproxPolynomial::monomial`], [`ApproxPolynomial::from_roots`] | Horner's rule; long division with a measured backward error |
//! | Values at many points | [`ApproxPolynomial::evaluate_batch`], [`ApproxPolynomial::evaluate_batch_into`] | the point kernel of the basis, one point after another |
//! | Antiderivative and definite integral | [`ApproxPolynomial::integral`], [`ApproxPolynomial::antiderivative`] (monomial, Bernstein, orthogonal) | termwise integration in the stored basis |
//! | Interpolate values at distinct nodes | [`ApproxPolynomial::interpolate_barycentric`], [`ApproxPolynomial::interpolate_newton`] | second (true) barycentric formula; Newton divided differences |
//! | Curve or surface on an interval or box | [`ApproxPolynomial::bernstein`], [`TensorBernstein`] | Bernstein–Bézier form with de Casteljau evaluation and subdivision |
//! | Expansion in Chebyshev, Legendre, Jacobi, Gegenbauer, Laguerre or Hermite polynomials | [`OrthogonalFamily`], [`OrthogonalBasisSpec`], [`ApproxPolynomial::orthogonal`] | three-term recurrence and Clenshaw summation |
//! | Change of basis | `to_monomial`, `to_newton`, `to_bernstein`, `to_barycentric` on [`ApproxPolynomial`]; [`OrthogonalTransform`] | explicit conversion with a [`PolynomialConversion`] report |
//! | Least-squares fit of noisy or overdetermined samples | [`ApproxPolynomial::fit_monomial`], [`ApproxPolynomial::fit_orthogonal`]; weights, report and reuse: [`PolynomialFitPlan`] | thin SVD of the weighted, column-scaled design matrix |
//! | Design matrix of a basis at sample points | [`vandermonde`], [`Vandermonde`] | one recurrence pass over all columns |
//! | Repeated products, compositions, Taylor shifts and divisions | [`PolynomialOperationPlan`]; exact: [`exact_multiply_into`], [`exact_compose_into`], [`exact_taylor_shift_into`] | direct, Karatsuba or FFT convolution, chosen by operand length ([`PolynomialProductAlgorithm`]) |
//! | Evaluation and interpolation at many points, exactly | [`ExactPolynomialProductTree`] | subproduct tree |
//! | Common factor of two inexact polynomials | [`approximate_gcd`], [`ApproxRationalFunction::simplify`] | SVD of the equilibrated Sylvester matrix |
//! | Resultant, subresultants, discriminant | [`approximate_resultant`], [`approximate_subresultant`], [`approximate_discriminant`], [`SylvesterElimination`], [`SylvesterDiscriminant`]; exact: [`exact_subresultants`], [`exact_discriminant`] | determinants of equilibrated Sylvester minors; exact subresultant sequence |
//! | Partial fractions with known, pairwise coprime denominator factors | [`ApproxPartialFractionPlan`], [`ExactPartialFractionPlan`] | equilibrated linear system for the term numerators; exact: extended Euclidean algorithm |
//! | Eigenvalue problem whose eigenvalues are the roots | [`ApproxPolynomial::root_matrix`], [`polynomial_root_matrix_into`] | companion matrix (monomial), comrade matrix (orthogonal) |
//! | Error disks and conditioning of computed roots | [`polynomial_root_diagnostics`], [`polynomial_root_radii`] | Weierstrass corrections and root condition numbers |
//! | Sparse multivariate polynomials | [`ApproxSparsePolynomial`], [`ExactSparsePolynomial`] | terms sorted in a [`MonomialOrder`] |
//!
//! To compute the roots themselves, use `polynomial_roots` in
//! `scioxide-rootfind`.
//!
//! # Coefficients and bases
//!
//! Coefficients are in ascending order: `[c₀, c₁, …, cₙ]` is
//! `p(x) = c₀ + c₁x + … + cₙxⁿ`, and the same holds for the coordinates of every
//! other basis. [`PolynomialBasis::coefficients`] on [`ApproxPolynomial::basis`]
//! returns the stored coordinates; for a barycentric polynomial these are the
//! values at the nodes. Constructors take ownership of the coordinates and nodes
//! they store and borrow data they only read. A Bernstein basis is defined on an
//! interval `[a, b]`, and an orthogonal basis maps its physical variable affinely
//! to the family's reference variable; [`OrthogonalBasisSpec::on_interval`]
//! places a Chebyshev, Legendre, Jacobi or Gegenbauer basis on `[a, b]`.
//! Conversion between bases is always explicit and returns a report.
//!
//! The policy [`PolynomialApproximationPolicy`] holds the tolerance and the
//! numerical-rank rule used by rank decisions such as the degree of an
//! approximate GCD. Every polynomial carries one: the plain constructors, such as
//! [`ApproxPolynomial::monomial`], use the default policy, and the
//! `_with_options` forms, such as [`ApproxPolynomial::monomial_with_options`],
//! take another. Construction does not drop small leading coefficients: only
//! exact trailing zeros are removed from monomial coefficients. Call
//! [`ApproxPolynomial::classify_coefficients`] to find the numerical degree.
//!
//! ```
//! use scioxide_poly::*;
//! // Least-squares line through (1, 3), (2, 5), (3, 7) in the Chebyshev basis on [1, 3].
//! // The data lie on y = 2x + 1, so the fit is exact.
//! let spec = OrthogonalBasisSpec::on_interval(OrthogonalFamily::ChebyshevT, 1.0_f64, 3.0)?;
//! let p = ApproxPolynomial::fit_orthogonal(&[1.0_f64, 2.0, 3.0], &[3.0, 5.0, 7.0], 1, spec)?;
//! assert!((p.evaluate(2.5)? - 6.0).abs() < 1e-12);
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Diagnostics
//!
//! Floating-point results carry a [`PolynomialConditionReport`]. Its backward
//! error is a model estimate: evaluation, conversion, product and composition use
//! `γₙ = nε / (1 − nε)`, where `n` is the number of coordinates and `ε` the machine
//! epsilon. Division instead measures the residual of `a = q·b + r`. The
//! forward-error estimate is the backward error times the condition estimate. For
//! evaluation in the monomial basis the condition estimate is
//!
//! ```text
//! κ(x) = (Σᵢ |cᵢ| |x|ⁱ) / |p(x)|,
//! ```
//!
//! and other bases use the analogous sum over their basis functions. A relative
//! estimate is `+∞` when the reference value is zero. These are estimates, not
//! bounds. Root disks from [`polynomial_root_diagnostics`] are not certified
//! enclosures either; for verified enclosures use `scioxide-validated`.
//!
//! [`ApproxRationalFunction::evaluate`] treats a point as a pole when the
//! denominator is zero or its forward-error estimate is at least 1.
//! [`PolynomialFitPlan`] minimizes `Σᵢ wᵢ |p(xᵢ) − yᵢ|²`, so a weight multiplies
//! a squared residual.
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors). In this crate:
//!
//! - `InvalidInput`: an empty coefficient or node list, repeated interpolation
//!   nodes, a zero divisor, a degree or index out of range, negative or all-zero
//!   fitting weights, or partial-fraction factors that are not pairwise coprime
//!   or do not multiply to the denominator.
//! - `NonFinite`: a NaN or infinite coefficient, node, sample or evaluation
//!   point.
//! - `DomainError`: an orthogonal family parameter outside its range (for
//!   example Jacobi `α ≤ −1`), or a pole of a rational function.
//! - `DimensionMismatch`: an output, scratch slice or workspace shorter than its
//!   requirement, or sample, value and weight lengths that differ.
//! - `NumericalInstability`: a coefficient, value or estimate overflows, or a
//!   factorization breaks down.
//! - `SizeOverflow`, `ResourceLimit`, `AllocationFailure`: a degree, term count
//!   or workspace too large to represent, over a configured limit, or impossible
//!   to allocate.
//!
//! Each item lists its own conditions. Forms that write into an output you
//! provide follow [Output on error](scioxide_core#output-on-error).
//!
//! # Reusing plans and workspaces
//!
//! The plans compute data that their calls reuse: [`PolynomialOperationPlan`]
//! (operation layouts and FFT plans), [`PolynomialFitPlan`] (the SVD of the
//! design matrix), [`ApproxPartialFractionPlan`] (the factored coefficient
//! system), [`ExactPartialFractionPlan`] (modular inverses),
//! [`OrthogonalBasisPlan`] (recurrence coefficients) and
//! [`SparseEvaluationPlan`] (the power layout). Their `_into` forms take a
//! workspace and an [`ExecutionContext`](scioxide_core::ExecutionContext) as the
//! last argument. A growable workspace is sized with `workspace_requirements`,
//! `try_reserve` and `check`; one reserved for the largest request also serves
//! smaller ones, as does an [`ExactPolynomialWorkspace`] for the plan-less exact
//! operations. The method values [`SylvesterElimination`] and
//! [`SylvesterDiscriminant`] hold validated degrees and a policy; their
//! fixed-size workspaces come from `workspace()`, and their `_scoped` forms run
//! inside the caller's solve. The batch forms of [`ApproxPolynomial`] take a
//! scratch slice of [`ApproxPolynomial::workspace_requirements`] entries for the
//! number of points. See
//! [Plans and workspaces](scioxide_core#plans-and-workspaces) and
//! [Allocation](scioxide_core#allocation).
//!
//! # Feature flags
//!
//! - `std` (default): enables `std` in the SciOxide dependencies. Without it the
//!   crate is `no_std` and needs `alloc`; the API is the same.
//!
//! The [`backend`] module holds lower-level building blocks used by other
//! SciOxide crates. Most users do not need it.
//!
//! # References
//!
//! - N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
//!   SIAM, 2002, chapter 5 (polynomial evaluation and its error analysis).
//! - J.-P. Berrut and L. N. Trefethen, "Barycentric Lagrange interpolation",
//!   *SIAM Review* 46(3), 2004.
//! - G. Farin, *Curves and Surfaces for CAGD*, 5th ed., Morgan Kaufmann, 2002.
//! - W. Gautschi, *Orthogonal Polynomials: Computation and Approximation*,
//!   Oxford University Press, 2004.
//! - NIST Digital Library of Mathematical Functions, chapter 18.
//!   <https://dlmf.nist.gov/18>
//! - J. von zur Gathen and J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
//!   Cambridge University Press, 2013.
//! - R. M. Corless, P. M. Gianni, B. M. Trager and S. M. Watt, "The singular
//!   value decomposition for polynomial systems", *Proc. ISSAC '95*, 1995.
//! - S. Barnett, "A companion matrix analogue for orthogonal polynomials",
//!   *Linear Algebra Appl.* 12, 1975.

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![doc(test(attr(deny(unused_imports, unused_variables))))]
#![deny(unreachable_pub)]

#[cfg_attr(test, macro_use)]
extern crate alloc;

#[cfg(all(test, not(feature = "std")))]
extern crate std;

#[cfg(test)]
extern crate self as scioxide_poly;
#[cfg(test)]
#[path = "../tests/support/assertions.rs"]
mod test_assertions;
#[cfg(test)]
#[path = "../tests/support/counted_ring.rs"]
mod test_counted_ring;
#[cfg(test)]
#[path = "../tests/support/exact_field.rs"]
mod test_field;
#[cfg(test)]
#[path = "../tests/support/prime_field.rs"]
mod test_prime_field;

mod admission;
mod approx_rational;
mod approximate;
pub mod backend;
mod basis;
mod bernstein;
mod composition;
mod division;
mod elimination;
mod elimination_exact;
mod evaluation;
mod exact;
mod exact_partial_fractions;
mod execution;
mod fast;
mod fast_condition;
mod fast_exact;
mod fitting;
mod horner;
mod multivariate;
mod orthogonal;
mod partial_fractions;
mod root_diagnostics;
mod scratch;
mod util;
mod vandermonde;

pub use approx_rational::{
    ApproxRationalFunction, ApproximateRationalEvaluation, ApproximateRationalSimplification,
};
pub use approximate::{ApproximateGcd, ApproximateGcdReport, approximate_gcd};
pub use basis::{
    ApproxPolynomial, BarycentricBasis, BernsteinBasis, BernsteinBounds, MonomialBasis,
    NewtonBasis, PolynomialApproximationPolicy, PolynomialBasis, PolynomialCoefficientReport,
    PolynomialConditionReport, PolynomialConversion, PolynomialEvaluation, PolynomialRootBasis,
    PolynomialRootBasisRef, polynomial_root_matrix_into,
};
pub use bernstein::{TensorBernstein, TensorBernsteinView};
pub use division::PolynomialDivision;
pub use elimination::{
    ApproximateElimination, DiscriminantWorkspace, EliminationReport, EliminationWorkspace,
    SylvesterDiscriminant, SylvesterElimination, approximate_discriminant, approximate_resultant,
    approximate_subresultant,
};
pub use elimination_exact::{ExactSubresultants, exact_discriminant, exact_subresultants};
pub use evaluation::{
    CheckedBasisEvaluation, PolynomialBackwardErrorModel, checked_basis_evaluation,
};
pub use exact::{ExactCoefficient, ExactField, ExactPolynomial, ExactRationalFunction};
pub use exact_partial_fractions::{
    ExactPartialFractionPlan, ExactPartialFractionTerm, ExactPartialFractions,
};
pub use execution::PolynomialExecutionOptions;
pub use fast::{
    PolynomialOperationPlan, PolynomialOperationRequest, PolynomialProductAlgorithm,
    PolynomialWorkspace, PolynomialWorkspaceRequirements,
};
pub use fast_condition::{polynomial_composition_condition, polynomial_product_condition};
pub use fast_exact::{
    ExactPolynomialProductTree, ExactPolynomialWorkspace, ExactPolynomialWorkspaceRequirements,
    exact_compose_into, exact_multiply_into, exact_taylor_shift_into,
};
pub use fitting::{
    PolynomialFit, PolynomialFitPlan, PolynomialFitReport, PolynomialFitRequirements,
    PolynomialFitWorkspace,
};
pub use multivariate::{
    ApproxSparsePolynomial, ExactSparsePolynomial, ExactSparseView, Monomial, MonomialOrder,
    SparseEvaluationPlan, SparseExpansionLimits, SparseTermIter, SparseTerms,
};
pub use orthogonal::{
    BasisNormalization, JacobiRecurrence, OrthogonalAntiderivative, OrthogonalBasis,
    OrthogonalBasisPlan, OrthogonalBasisSpec, OrthogonalDerivative, OrthogonalFamily,
    OrthogonalTransform,
};
pub use partial_fractions::{
    ApproxPartialFractionPlan, ApproxPartialFractions, PartialFractionFactor,
    PartialFractionReport, PartialFractionRequirements, PartialFractionTerm,
    PartialFractionWorkspace,
};
pub use root_diagnostics::{
    RootCluster, RootConditionReport, RootDiagnostic, RootDiagnostics, RootDiagnosticsRequirements,
    RootDiagnosticsStatus, RootDiagnosticsWorkspace, RootDiskEstimate, RootEstimate,
    RootEstimateFailure, RootRadiusReport, polynomial_root_diagnostics,
    polynomial_root_diagnostics_into, polynomial_root_radii,
};
pub use scratch::PolynomialScratchRequirements;
pub use vandermonde::{Vandermonde, vandermonde};
