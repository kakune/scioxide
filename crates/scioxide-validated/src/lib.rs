//! Verified numerics: interval and affine arithmetic, and certified linear
//! systems, roots and integrals.
//!
//! Every value this crate returns is an enclosure: a set proved to contain the
//! exact mathematical result, with all rounding errors accounted for. The crate
//! provides interval arithmetic with decorations and enclosures of the
//! elementary functions, real and complex affine arithmetic, and verifiers that
//! prove the solution of a square linear system, the existence, uniqueness or
//! absence of roots of scalar functions, nonlinear systems and polynomials, and
//! the value of a definite integral. Intervals have `f32` or `f64` endpoints.
//! Affine forms, linear and nonlinear systems, polynomials and integrands may
//! also be complex.
//!
//! # Quick start
//!
//! ```
//! use scioxide_validated::{roots, DecoratedInterval as D};
//!
//! let (one, two) = (D::point(1.0_f64)?, D::point(2.0)?);
//! let found = roots(|x| x.square()?.sub(&one), |x| x.mul(&two), -2.0, 3.0)?;
//! assert_eq!(found.len(), 2);
//! assert!(found[0].contains(-1.0) && found[1].contains(1.0));
//! # Ok::<(), scioxide_core::SciOxideError>(())
//! ```
//!
//! # Choosing a method
//!
//! | Problem | One call | Configured entry point | Method |
//! |---|---|---|---|
//! | Range of an arithmetic expression | — | [`DecoratedInterval`] | interval arithmetic with outward rounding |
//! | Range of `exp`, `ln`, `sin`, `atan2`, `powf` and other elementary functions | — | [`DecoratedInterval`], [`ElementaryEnclosure`] | argument reduction and polynomial approximation with proved error bounds |
//! | Range of an expression whose variables repeat, real or complex | — | [`AffineContext`], [`AffineForm`] | affine arithmetic |
//! | Square linear system `Ax = b` with point or interval data | [`solve`] | [`LinearVerifier`] with [`DenseIntervalOperator`], [`SparseIntervalOperator`] or [`LuIntervalInverse`] | contraction with an approximate-inverse preconditioner |
//! | Root of a scalar function in an interval | [`roots`] | [`ScalarRootVerifier`] | interval Newton method |
//! | Root of a nonlinear system in a box | — | [`SystemRootVerifier`] | Krawczyk operator |
//! | Roots of a polynomial, counted in disks | [`polynomial_roots`] | [`PolynomialRootPlan`] | Rouché inclusion test and Pellet's test |
//! | Definite integral over a finite or infinite interval | [`integrate`] | [`QuadratureVerifier`] | adaptive bisection with interval rectangle, trapezoid or Gauss–Legendre bounds |
//!
//! # Proposals and verification
//!
//! A verifier checks a proposed answer against the original problem. The
//! proposal can come from anywhere, for example an approximate solution from
//! `scioxide-linalg` or root estimates from `scioxide-rootfind`, and is not trusted:
//! [`LinearCertificate`], [`SystemRootCertificate`], [`QuadratureCertificate`] and the
//! other certificate types only store it. Every `verify` method recomputes all
//! bounds it needs from the problem, so a certificate is valid only together
//! with the report of a verification that succeeded. The generating methods
//! [`LinearVerifier::solve`], [`ScalarRootVerifier::isolate`] and
//! [`PolynomialRootPlan::isolate`] prove each result they return.
//! [`QuadratureVerifier::generate`] only builds a proposal; pass it to
//! [`QuadratureVerifier::verify`] to certify it.
//!
//! Each report has one status with three kinds of outcome:
//!
//! - success: `Verified` or `Proved`;
//! - the check shows that the proposal is wrong: `Rejected` or `Refuted`;
//! - neither: `Inconclusive`, with a [`ProofInconclusive`] reason. The reason
//!   is a bound that is not finite, an exhausted evaluation or iteration
//!   budget, cancellation or a deadline, or a method-specific test that fails.
//!
//! Configured forms return these outcomes in `Ok`. The one-shot functions
//! [`solve`], [`roots`], [`polynomial_roots`] and [`integrate`] return `Ok` only
//! for verified or complete results and turn every other outcome into an error.
//! A failed proof does not establish the
//! opposite claim: an inconclusive root proof says nothing about whether a root
//! exists.
//!
//! # Interval extensions you provide
//!
//! The verifiers evaluate your functions through interval extensions:
//! [`IntervalScalarFunction`], [`IntervalLinearOperator`],
//! [`IntervalResidualOperator`], [`IntervalJacobianOperator`] and
//! [`QuadratureOracle`]. Closures implement [`IntervalScalarFunction`] and
//! [`QuadratureOracle`] without a context parameter: `Fn(DecoratedInterval<R>)`
//! and `Fn(QuadratureRequest<R>, &mut [IntervalBox<R>])`, respectively. Implement
//! the unsuffixed trait method and override its `_scoped` method for nested solves.
//! An extension must return, for every input set, a set
//! that contains every value the function takes on it, including the rounding
//! errors of its own evaluation. The arithmetic of [`DecoratedInterval`] and
//! [`ElementaryEnclosure`] produces such enclosures. A proof is only as sound
//! as these extensions: a verifier cannot detect an extension that returns too
//! narrow a set, and point evaluations or error estimates are not enclosures.
//!
//! Complex values are enclosed componentwise by an [`IntervalBox`], a rectangle
//! of a real and an imaginary interval. For real scalars the imaginary
//! interval must be exactly zero.
//!
//! # Floating-point environment
//!
//! Endpoints are rounded outward with error-free transformations, so the crate
//! never changes the floating-point control registers. It requires
//! round-to-nearest-even, gradual underflow (no flush-to-zero) and masked
//! floating-point exceptions on an x86-64, x86 with SSE2, or AArch64 target.
//! Each numerical operation reads the control registers before it computes and
//! again after each of your callbacks. Any other environment or target, and
//! running under Miri, gives `InvalidInput` with
//! [`InvalidInputReason::UnsupportedConfiguration`].
//!
//! # Errors
//!
//! Errors follow the [shared rules](scioxide_core#errors). In this crate:
//!
//! - `InvalidInput`: an unsupported floating-point environment; a malformed
//!   proposal, such as a negative radius or a center outside its region; a
//!   quadrature partition with gaps or overlaps; affine values from different
//!   contexts; a decoration stronger than its set permits; or an interval
//!   extension whose shape changes or that returns an empty or malformed
//!   enclosure.
//! - `NonFinite`: a NaN or infinite center, radius or coefficient.
//! - `DimensionMismatch`: a proposal, output or right-hand side whose length
//!   does not match the problem.
//! - `ResourceLimit`: storage beyond [`ValidatedOptions::byte_limit`] or
//!   [`AffineOptions::byte_limit`].
//! - `SizeOverflow` and `AllocationFailure`: storage that cannot be sized or
//!   reserved.
//! - `NumericalInstability`: a required enclosure is unbounded (`Overflow`),
//!   nonsingularity is not proved (`IllConditioned`), or an internal invariant fails (`Breakdown`).
//! - `ConvergenceFailure`: a one-shot proof reaches a budget, capacity or resolution
//!   limit, or its tests remain inconclusive (`Stagnation`).
//! - `ExecutionStopped`: cancellation or a deadline stops a one-shot proof.
//! - `DomainError`: a one-shot callback enclosure is empty or not known continuous.
//! - `Singular`: a zero pivot in the one-shot linear factorization, not a singularity proof.
//!
//! Errors from your callbacks propagate unchanged. Points outside the domain of
//! an interval function are not errors: they are excluded from the result and
//! reported by [`EnclosureStatus`] and [`Decoration`].
//!
//! # Accuracy
//!
//! Enclosures are always rigorous. Their width is as follows.
//!
//! | Operations | Width |
//! |---|---|
//! | [`DecoratedInterval::add`], `sub`, `mul`, `div`, `reciprocal`, `square`, `sqrt`, `abs`, `neg` | tightest: the exact range with each endpoint rounded outward to the adjacent `f32` or `f64` |
//! | elementary functions of [`DecoratedInterval`] at a point argument | at most two floating-point steps wide, checked against correctly rounded reference values for all 21 unary functions in `f32` and `f64` |
//! | [`LinearVerifier`] solves | for point data, the error bound exceeds the true error by at most the factor `(1 + q)/(1 − q)`, plus rounding, where `q < 1` is the proved bound on `‖I − RA‖` for the matrix `A` and the preconditioner `R` |
//!
//! The width of other verified results depends on the problem and on the
//! extensions you provide.
//!
//! # Reusing plans and workspaces
//!
//! Verifiers bind a problem to its validated options without preparing reusable
//! numerical data. [`PolynomialRootPlan`] scales and stores coefficient bounds
//! once, in O(d) work and storage for degree `d`.
//! The owned form, such as [`LinearVerifier::solve`], allocates its workspace
//! and output and runs with `ExecutionContext::sequential_native()`.
//! `_scoped` means that the call runs inside the caller's solve, whose scope is
//! the last parameter; `_into` writes an output you provide. See
//! [Plans and workspaces](scioxide_core#plans-and-workspaces) and
//! [Solve accounting](scioxide_core#solve-accounting).
//!
//! [`LinearWorkspace`], [`RootWorkspace`] and [`QuadratureWorkspace`]
//! are sized by `workspace_requirements`; root and quadrature methods take a
//! [`ProofRequest`]. A workspace that is too small grows before any callback
//! runs, and with a workspace that is large enough, later calls do not allocate.
//! `last_attempt` reports the outcome and work counts of the last call.
//!
//! Batch forms, such as [`ScalarRootVerifier::verify_batch_into`] and
//! [`LinearVerifier::solve_block_into`], run independent proofs in parallel
//! under the execution policy of [`ValidatedOptions`], with at most 64
//! partitions. The budgets are divided between the proofs by index, so the
//! reports do not depend on the policy or the number of workers. Initialize
//! report slots with [`LinearVerificationReport::not_run`],
//! [`RootVerificationReport::not_run`] or
//! [`PolynomialVerificationReport::not_run`].
//!
//! # Feature flags
//!
//! - `std` (default): enables the `std` features of `scioxide-core`,
//!   `scioxide-linalg` and `scioxide-sparse`. Without it the crate is `no_std` and
//!   needs `alloc`; the API is the same.
//!
//! # References
//!
//! - R. E. Moore, R. B. Kearfott and M. J. Cloud, *Introduction to Interval
//!   Analysis*, SIAM, 2009.
//! - IEEE Std 1788-2015, *IEEE Standard for Interval Arithmetic*, 2015
//!   (decorations).
//! - A. Neumaier, *Interval Methods for Systems of Equations*, Cambridge
//!   University Press, 1990.
//! - S. M. Rump, "Verification methods: Rigorous results using floating-point
//!   arithmetic", *Acta Numerica* 19, 2010, pp. 287–449.
//! - L. H. de Figueiredo and J. Stolfi, "Affine arithmetic: concepts and
//!   applications", *Numerical Algorithms* 37, 2004, pp. 147–158.
//! - R. Krawczyk, "Newton-Algorithmen zur Bestimmung von Nullstellen mit
//!   Fehlerschranken", *Computing* 4, 1969, pp. 187–201.
//! - P. Henrici, *Applied and Computational Complex Analysis*, Vol. 1, Wiley,
//!   1974.
//! - M. H. Payne and R. N. Hanek, "Radian reduction for trigonometric
//!   functions", *ACM SIGNUM Newsletter* 18(1), 1983.

#![deny(unsafe_code)]
#![deny(unreachable_pub)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod one_shot;
mod outcome;
pub use one_shot::{integrate, polynomial_roots, roots, solve};
pub use outcome::{ProofInconclusive, RefinementLimit};
mod options;
mod protocol;
mod shell;
pub use options::{ProofRequest, ValidatedOptions};
pub use protocol::{
    IntervalBox, IntervalJacobianOperator, IntervalLinearOperator, IntervalResidualOperator,
    IntervalScalarFunction,
};
mod affine;
mod elementary;
mod environment;
mod interval_math;
mod linear;
mod quadrature;
mod rectangle;
mod roots;
mod rounding;
pub use linear::{
    DenseIntervalOperator, LinearBounds, LinearCertificate, LinearCertificateView,
    LinearCertificateViewMut, LinearInconclusive, LinearProofFailure, LinearRejection,
    LinearRequirements, LinearVerificationReport, LinearVerificationStatus, LinearVerifier,
    LinearWorkspace, LuIntervalInverse, SparseIntervalOperator, interval_enclosure,
    point_enclosure,
};
pub use quadrature::{
    QuadratureAssumption, QuadratureCertificate, QuadratureCertificateView,
    QuadratureGenerationReport, QuadratureInconclusive, QuadratureObligation, QuadratureOracle,
    QuadraturePanel, QuadraturePiece, QuadratureProofFailure, QuadratureQuery, QuadratureRejection,
    QuadratureRequest, QuadratureRequirements, QuadratureTermination, QuadratureVerificationStatus,
    QuadratureVerifier, QuadratureWorkspace,
};
pub use roots::{
    PolynomialIsolationReport, PolynomialRootCertificate, PolynomialRootCover, PolynomialRootPlan,
    PolynomialVerificationReport, RootClaim, RootDisk, RootInconclusive, RootIsolation,
    RootIsolationNode, RootIsolationReport, RootIsolationTermination, RootIsolationView,
    RootProofFailure, RootRequirements, RootVerificationReport, RootVerificationStatus,
    RootWorkspace, ScalarRootCertificate, ScalarRootVerifier, SystemRootCertificate,
    SystemRootCertificateView, SystemRootCertificateViewMut, SystemRootVerifier,
};

pub use affine::{
    AffineBinary, AffineContext, AffineFallback, AffineForm, AffineOptions, AffineProjection,
    AffineReport, AffineState, AffineSymbol, AffineTerm, AffineUnary, AffineView, AffineViewMut,
    AffineWorkspace,
};
pub use elementary::{ElementaryEnclosure, EnclosureStatus};

use rounding::{Operation, endpoint};
use scioxide_core::{ExtendedInterval, Interval, InvalidInputReason, RealScalar, SciOxideError};

/// States what is known about a function on the input of an interval operation.
///
/// Decorations are ordered from weakest to strongest, `Trv < Def < Dac < Com`,
/// and are named as in IEEE Std 1788-2015. An operation returns the weaker of
/// its operands' decorations and the decoration it can assert itself. A
/// decoration says nothing about how wide the enclosure is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Decoration {
    /// Nothing is asserted beyond the enclosure. Empty results and functions
    /// defined on only part of the input have this decoration.
    Trv,
    /// The function is defined on the whole input; continuity is not asserted.
    Def,
    /// The function is defined and continuous on the whole input.
    Dac,
    /// The function is defined and continuous on the whole input, and the input
    /// and the result are nonempty and bounded.
    Com,
}

/// Closed real interval with a [`Decoration`], for interval arithmetic with
/// outward rounding.
///
/// The set is an [`ExtendedInterval`]: empty, or `[a, b]` with
/// `−∞ ≤ a ≤ b ≤ +∞`, where an infinite end means that side is unbounded. It
/// contains only real numbers. Arithmetic returns a `DecoratedInterval` that
/// encloses `{x ∘ y : x ∈ X, y ∈ Y}`. Elementary functions return an
/// [`ElementaryEnclosure`], whose [`ElementaryEnclosure::hull`] is again a
/// `DecoratedInterval`. Every operation takes constant time and does not
/// allocate.
///
/// Interval arithmetic treats each occurrence of a variable independently, so
/// `x − x` for `x = [1, 2]` gives `[−1, 1]`. Use [`Self::square`] instead of
/// `x · x`, or [`AffineForm`] when variables repeat.
///
/// ```
/// use scioxide_core::Interval;
/// use scioxide_validated::{DecoratedInterval, Decoration, EnclosureStatus};
///
/// let x = DecoratedInterval::new(1.0_f64, 2.0)?;
/// let sum = x.add(&DecoratedInterval::point(2.0)?)?;
/// let bounds = Interval::try_from(sum)?;
/// assert!(bounds.contains(3.0) && bounds.contains(4.0));
/// assert_eq!(sum.decoration(), Decoration::Com);
/// // sin is increasing up to π/2 ≈ 1.571 and decreasing after it.
/// let sine = x.sin()?;
/// assert_eq!(sine.status(), EnclosureStatus::Defined);
/// assert!(sine.contains(1.0) && sine.contains(1.0_f64.sin()));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct DecoratedInterval<S: RealScalar> {
    set: ExtendedInterval<S>,
    decoration: Decoration,
}

fn strongest<S: RealScalar>(set: &ExtendedInterval<S>) -> Decoration {
    if set.is_empty() {
        Decoration::Trv
    } else if set.is_finite() {
        Decoration::Com
    } else {
        Decoration::Dac
    }
}

impl<S: RealScalar> From<ExtendedInterval<S>> for DecoratedInterval<S> {
    fn from(set: ExtendedInterval<S>) -> Self {
        Self {
            decoration: strongest(&set),
            set,
        }
    }
}

impl<S: RealScalar> From<Interval<S>> for DecoratedInterval<S> {
    fn from(set: Interval<S>) -> Self {
        Self::from(ExtendedInterval::from(set))
    }
}

impl<S: RealScalar> TryFrom<DecoratedInterval<S>> for Interval<S> {
    type Error = SciOxideError;

    /// Extracts finite bounds and drops the decoration without allocating.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` with argument `interval` for an empty or unbounded set.
    fn try_from(value: DecoratedInterval<S>) -> Result<Self, Self::Error> {
        Self::try_from(value.set())
    }
}

impl<S: RealScalar> DecoratedInterval<S> {
    /// Creates the finite interval `[lower, upper]` without allocating.
    ///
    /// The decoration is `Com`; signed zero endpoints are preserved.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if either end is NaN or infinite.
    /// - `InvalidInput` if `lower > upper`.
    pub fn new(lower: S, upper: S) -> Result<Self, SciOxideError> {
        Interval::new(lower, upper).map(Self::from)
    }

    /// Creates the singleton `[value, value]`, decorated `Com`, without allocating.
    ///
    /// # Errors
    ///
    /// Returns `NonFinite` if `value` is NaN or infinite.
    pub fn point(value: S) -> Result<Self, SciOxideError> {
        Self::new(value, value)
    }

    /// Returns the empty set, decorated `Trv`.
    pub fn empty() -> Self {
        Self::from(ExtendedInterval::empty())
    }

    /// Returns the whole real line, decorated `Dac`.
    pub fn entire() -> Self {
        Self::from(ExtendedInterval::entire())
    }

    /// Pairs a set with a decoration, for example to weaken it to `Trv`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the decoration is stronger than the set allows:
    ///   `Com` for an unbounded set, or anything above `Trv` for the empty set.
    pub fn with_decoration(
        set: ExtendedInterval<S>,
        decoration: Decoration,
    ) -> Result<Self, SciOxideError> {
        if decoration > strongest(&set) {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("decoration"),
                InvalidInputReason::Inconsistent,
                "decoration is stronger than the interval set permits",
            ));
        }
        Ok(Self { set, decoration })
    }

    /// Returns the interval set, with the signs of zero ends kept.
    pub fn set(&self) -> ExtendedInterval<S> {
        self.set
    }

    /// Returns the decoration.
    pub fn decoration(&self) -> Decoration {
        self.decoration
    }

    /// Returns whether the finite value `value` lies in the set.
    pub fn contains(&self, value: S) -> bool {
        self.set.contains(value)
    }

    /// Returns whether `other` is a subset of this set; decorations are ignored
    /// and the empty set is a subset of every set.
    pub fn contains_interval(&self, other: &Self) -> bool {
        self.set.contains_interval(&other.set)
    }

    /// Returns the smallest interval that contains both sets, decorated `Trv`.
    pub fn hull(&self, other: &Self) -> Self {
        Self {
            set: self.set.hull(&other.set),
            decoration: Decoration::Trv,
        }
    }

    /// Returns the intersection of both sets, decorated `Trv`.
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            set: self.set.intersection(&other.set),
            decoration: Decoration::Trv,
        }
    }

    /// Returns the intersection with `domain`, decorated `Trv`, for example to
    /// restrict an argument to the domain of a function.
    pub fn restrict(&self, domain: &ExtendedInterval<S>) -> Self {
        Self {
            set: self.set.intersection(domain),
            decoration: Decoration::Trv,
        }
    }

    fn bounds(&self) -> Option<(S, S)> {
        self.set
            .bounds()
            .map(|(lo, hi)| (lo.unwrap_or(-S::INFINITY), hi.unwrap_or(S::INFINITY)))
    }

    #[inline(always)]
    fn result(lo: S, hi: S, decoration: Decoration) -> Result<Self, SciOxideError> {
        let zero = S::ZERO;
        let lo = if lo == zero { -zero } else { lo };
        let hi = if hi == zero { zero } else { hi };
        let set = ExtendedInterval::new(
            if lo == -S::INFINITY { None } else { Some(lo) },
            if hi == S::INFINITY { None } else { Some(hi) },
        )?;
        Ok(Self {
            decoration: decoration.min(strongest(&set)),
            set,
        })
    }

    /// Returns the tightest interval that contains `{x + y : x ∈ self, y ∈ other}`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the [floating-point environment](crate#floating-point-environment)
    ///   is not supported. This is the only error of the interval operations.
    pub fn add(&self, other: &Self) -> Result<Self, SciOxideError> {
        environment::check()?;
        self.add_admitted(other)
    }

    /// Arithmetic kernel; the public caller checks the floating-point environment.
    pub(crate) fn add_admitted(&self, other: &Self) -> Result<Self, SciOxideError> {
        let Some(((a, b), (c, d))) = self.bounds().zip(other.bounds()) else {
            return Ok(Self::empty());
        };
        Self::result(
            endpoint(a, c, Operation::Add, false),
            endpoint(b, d, Operation::Add, true),
            self.decoration.min(other.decoration),
        )
    }

    /// Returns `{−x : x ∈ self}`, which is exact.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the floating-point environment is not supported.
    pub fn neg(&self) -> Result<Self, SciOxideError> {
        environment::check()?;
        self.neg_admitted()
    }

    /// Arithmetic kernel; the public caller checks the floating-point environment.
    pub(crate) fn neg_admitted(&self) -> Result<Self, SciOxideError> {
        let Some((a, b)) = self.bounds() else {
            return Ok(Self::empty());
        };
        Self::result(-b, -a, self.decoration)
    }

    /// Returns the tightest interval that contains `{x − y : x ∈ self, y ∈ other}`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the floating-point environment is not supported.
    pub fn sub(&self, other: &Self) -> Result<Self, SciOxideError> {
        environment::check()?;
        self.sub_admitted(other)
    }

    /// Arithmetic kernel; the public caller checks the floating-point environment.
    pub(crate) fn sub_admitted(&self, other: &Self) -> Result<Self, SciOxideError> {
        self.add_admitted(&other.neg_admitted()?)
    }

    /// Directed bounds on one fixed-sign denominator branch, including zero limits.
    #[inline(always)]
    fn quotient_bounds(a: S, b: S, c: S, d: S, positive: bool) -> (S, S) {
        let zero = S::ZERO;
        let (l, h) = if positive { (a, b) } else { (b, a) };
        let ld = if l >= zero { d } else { c };
        let hd = if h <= zero { d } else { c };
        rounding::quotient_bounds(l, ld, h, hd)
    }

    /// Returns the tightest interval that contains `{x · y : x ∈ self, y ∈ other}`.
    ///
    /// Zero times an unbounded interval is zero, since the sets contain only
    /// real numbers.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the floating-point environment is not supported.
    pub fn mul(&self, other: &Self) -> Result<Self, SciOxideError> {
        environment::check()?;
        self.mul_admitted(other)
    }

    /// Arithmetic kernel; the public caller checks the floating-point environment.
    pub(crate) fn mul_admitted(&self, other: &Self) -> Result<Self, SciOxideError> {
        let Some(((a, b), (c, d))) = self.bounds().zip(other.bounds()) else {
            return Ok(Self::empty());
        };
        let ((l, r), (h, s)) = rounding::product_corners(a, b, c, d);
        Self::result(
            endpoint(l, r, Operation::Multiply, false),
            endpoint(h, s, Operation::Multiply, true),
            self.decoration.min(other.decoration),
        )
    }

    /// Returns the tightest interval that contains
    /// `{x / y : x ∈ self, y ∈ other, y ≠ 0}`.
    ///
    /// Division by zero is excluded from the result, not reported as an error.
    /// If `other` contains zero the decoration is `Trv`, and if it is `[0, 0]`
    /// the result is empty. When the quotient set has two parts, for example
    /// `[1, 2] / [−1, 1]`, the result is their hull.
    ///
    /// ```
    /// use scioxide_core::ExtendedInterval;
    /// use scioxide_validated::{DecoratedInterval, Decoration};
    ///
    /// let numerator = DecoratedInterval::new(1.0_f64, 2.0)?;
    /// let denominator = DecoratedInterval::new(0.0, 4.0)?;
    /// // [1, 2] / (0, 4] = [1/4, +∞)
    /// let quotient = numerator.div(&denominator)?;
    /// assert_eq!(quotient.set(), ExtendedInterval::new(Some(0.25), None)?);
    /// assert_eq!(quotient.decoration(), Decoration::Trv);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the floating-point environment is not supported.
    pub fn div(&self, other: &Self) -> Result<Self, SciOxideError> {
        environment::check()?;
        self.div_admitted(other)
    }

    /// Arithmetic kernel; the public caller checks the floating-point environment.
    pub(crate) fn div_admitted(&self, other: &Self) -> Result<Self, SciOxideError> {
        let Some(((a, b), (c, d))) = self.bounds().zip(other.bounds()) else {
            return Ok(Self::empty());
        };
        let zero = S::ZERO;
        let decoration = if c <= zero && d >= zero {
            Decoration::Trv
        } else {
            self.decoration.min(other.decoration)
        };
        // Signed zero marks the one-sided denominator limit. Evaluate division
        // directly: multiplying a rounded reciprocal would lose tightness.
        let (lo, hi) = if c < zero {
            if d < zero {
                Self::quotient_bounds(a, b, c, d, false)
            } else if d > zero {
                // Any nonzero numerator has both infinite one-sided limits.
                return if a == zero && b == zero {
                    Self::result(zero, zero, decoration)
                } else {
                    Self::result(-S::INFINITY, S::INFINITY, decoration)
                };
            } else {
                Self::quotient_bounds(a, b, c, -zero, false)
            }
        } else if d > zero {
            Self::quotient_bounds(a, b, if c > zero { c } else { zero }, d, true)
        } else {
            return Ok(Self::empty());
        };
        Self::result(lo, hi, decoration)
    }

    /// Returns the tightest interval that contains `{1/x : x ∈ self, x ≠ 0}`,
    /// with zero handled as in [`Self::div`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the floating-point environment is not supported.
    pub fn reciprocal(&self) -> Result<Self, SciOxideError> {
        environment::check()?;
        self.reciprocal_admitted()
    }

    /// Arithmetic kernel; the public caller checks the floating-point environment.
    pub(crate) fn reciprocal_admitted(&self) -> Result<Self, SciOxideError> {
        Self::point(S::ONE)?.div_admitted(self)
    }

    /// Returns `{|x| : x ∈ self}`, which is exact.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the floating-point environment is not supported.
    pub fn abs(&self) -> Result<Self, SciOxideError> {
        environment::check()?;
        self.abs_admitted()
    }

    /// Arithmetic kernel; the public caller checks the floating-point environment.
    pub(crate) fn abs_admitted(&self) -> Result<Self, SciOxideError> {
        let Some((a, b)) = self.bounds() else {
            return Ok(Self::empty());
        };
        let lo = if a <= S::ZERO && b >= S::ZERO {
            S::ZERO
        } else {
            a.abs().min(b.abs())
        };
        Self::result(lo, a.abs().max(b.abs()), self.decoration)
    }

    /// Returns the tightest interval that contains `{x² : x ∈ self}`.
    ///
    /// Unlike `self.mul(self)`, which treats the two factors independently, the
    /// result never contains negative numbers: `[−1, 2]` squares to `[0, 4]`,
    /// not `[−2, 4]`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the floating-point environment is not supported.
    pub fn square(&self) -> Result<Self, SciOxideError> {
        environment::check()?;
        self.square_admitted()
    }

    /// Arithmetic kernel; the public caller checks the floating-point environment.
    pub(crate) fn square_admitted(&self) -> Result<Self, SciOxideError> {
        let Some((a, b)) = self.bounds() else {
            return Ok(Self::empty());
        };
        let lo = if a <= S::ZERO && b >= S::ZERO {
            S::ZERO
        } else {
            a.abs().min(b.abs())
        };
        let hi = a.abs().max(b.abs());
        let (lo, hi) = rounding::square_bounds(lo, hi);
        Self::result(lo, hi, self.decoration)
    }

    /// Returns the tightest interval that contains `{√x : x ∈ self, x ≥ 0}`.
    ///
    /// Negative numbers are excluded: the decoration is `Trv` if `self`
    /// contains any, and the result is empty if all of `self` is negative.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the floating-point environment is not supported.
    pub fn sqrt(&self) -> Result<Self, SciOxideError> {
        environment::check()?;
        self.sqrt_admitted()
    }

    /// Arithmetic kernel; the public caller checks the floating-point environment.
    pub(crate) fn sqrt_admitted(&self) -> Result<Self, SciOxideError> {
        let Some((a, b)) = self.bounds() else {
            return Ok(Self::empty());
        };
        if b < S::ZERO {
            return Ok(Self::empty());
        }
        let decoration = if a < S::ZERO {
            Decoration::Trv
        } else {
            self.decoration
        };
        Self::result(
            endpoint(
                if a > S::ZERO { a } else { S::ZERO },
                S::ZERO,
                Operation::Sqrt,
                false,
            ),
            endpoint(b, S::ZERO, Operation::Sqrt, true),
            decoration,
        )
    }
}
