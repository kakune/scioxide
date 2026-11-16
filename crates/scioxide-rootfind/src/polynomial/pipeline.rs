//! Polynomial root solve: candidate generation, polishing and backward-error checks.

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::ops::{Deref, Neg};

use scioxide_core::Complex;
use scioxide_core::Scaled;
use scioxide_core::backend::{checked_add, checked_mul, try_reserve};
use scioxide_core::checked_quadratic_roots;
use scioxide_core::{
    Argument, Budget, NumericScalar, NumericalIssue, Quantity, RealScalar, SciOxideError,
};
use scioxide_linalg::{GeneralEigenOptions, GeneralEigenvaluesBuffer, LinalgWorkspace, Matrix};
use scioxide_poly::{CheckedBasisEvaluation, PolynomialRootBasisRef, checked_basis_evaluation};

use super::PolynomialRootOptions;
use crate::result::RootTermination;

mod certification;
mod companion;
mod driver;
mod laguerre;
mod report;
mod seeding;
mod simultaneous;
mod workspace;
use crate::invalid;
use certification::*;
use companion::*;
use driver::{PolynomialFailure, Prepared, admit, admitted, invoke, iteration};
pub(super) use driver::{solve_into, validate_coefficients, validate_method};
use laguerre::*;
pub use report::{
    PolynomialRootAlgorithm, PolynomialRootDiagnostic, PolynomialRootMethod, PolynomialRootReport,
    PolynomialRootWorkspaceRequirements, PolynomialRoots,
};
use seeding::*;
use simultaneous::*;
pub use workspace::PolynomialRootWorkspace;
