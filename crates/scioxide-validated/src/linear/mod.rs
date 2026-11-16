//! Verified solution of square linear systems.
use scioxide_core::AlgebraicScalar;
mod adapters;
pub(crate) mod proof;
#[cfg(test)]
mod tests;
mod workspace;

use crate::interval_math::*;
use crate::shell::{Outcome, invalid};
use crate::{DecoratedInterval as D, rectangle::Box2};
use crate::{IntervalBox, IntervalLinearOperator, ValidatedOptions};
use alloc::vec::Vec;
use scioxide_core::backend::{checked_add, checked_mul, try_filled};
use scioxide_core::{Argument, Complex, NumericScalar, Quantity, RealScalar, SciOxideError};

pub use adapters::{DenseIntervalOperator, LuIntervalInverse, SparseIntervalOperator};
pub use proof::LinearVerifier;
pub use workspace::{LinearRequirements, LinearWorkspace};

type Result<T> = core::result::Result<T, SciOxideError>;

/// Returns the rectangle with corners `lower` and `upper`: the real interval
/// `[Re lower, Re upper]` and the imaginary interval `[Im lower, Im upper]`.
///
/// For real `S` the imaginary interval is zero.
///
/// ```
/// use scioxide_validated::interval_enclosure;
///
/// let x = interval_enclosure(0.5_f64, 1.5)?;
/// assert!(x.re.contains(1.0) && x.im.contains(0.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # Errors
///
/// - `NonFinite` if a component is NaN or infinite.
/// - `InvalidInput` if a component of `lower` exceeds that of `upper`.
pub fn interval_enclosure<S: NumericScalar>(lower: S, upper: S) -> Result<IntervalBox<S::Real>> {
    Ok(Complex::new(
        scioxide_core::ExtendedInterval::new(
            Some(lower.real_component()),
            Some(upper.real_component()),
        )?,
        scioxide_core::ExtendedInterval::new(
            Some(lower.imaginary_component()),
            Some(upper.imaginary_component()),
        )?,
    ))
}
/// Returns the rectangle that contains only `value`.
///
/// # Errors
///
/// - `NonFinite` if a component of `value` is NaN or infinite.
pub fn point_enclosure<S: NumericScalar>(value: S) -> Result<IntervalBox<S::Real>> {
    interval_enclosure(value, value)
}

/// Reason why [`LinearVerifier::verify`] rejects a proposed error bound.
///
/// A rejection does not imply that the system is singular or that the
/// proposed box misses the solution; a wider box may verify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearRejection {
    /// The proposed box is not mapped into itself by the error equation and
    /// does not contain the contraction bound; see [`LinearVerifier`].
    EnclosureTooNarrow,
}
/// Reason specific to linear systems why a proof is inconclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearProofFailure {
    /// The bound `q` on `‖I − RA‖` is not below 1, so the matrix is not proved
    /// nonsingular. A better preconditioner `R` may succeed.
    RegularityNotProved,
    /// Applying the preconditioner to the right-hand side gives no finite
    /// starting approximation.
    SeedFailure,
}
/// Reason why a linear-system proof is inconclusive.
pub type LinearInconclusive = crate::ProofInconclusive<LinearProofFailure>;

/// Outcome of a linear-system proof.
///
/// Only `Verified` certifies a result; a small residual alone does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearVerificationStatus {
    /// The report slot has not been filled; no proof has run.
    NotRun,
    /// The matrix is proved nonsingular and, except for
    /// [`LinearVerifier::bounds`], the error bound is proved.
    Verified,
    /// A proposed error bound fails the check. Only
    /// [`LinearVerifier::verify`] returns this; error bounds that the plan
    /// computes itself are never rejected.
    Rejected(LinearRejection),
    /// The proof could not be completed.
    Inconclusive(LinearInconclusive),
}
/// Proved upper bounds on matrix norms, from one pass over the matrix.
///
/// Norms are infinity norms on the real and imaginary parts: for a complex
/// matrix the row sum is `Σⱼ (|Re aᵢⱼ| + |Im aᵢⱼ|)`. `A` is the system matrix
/// and `R` the preconditioner. The optional bounds are computed only by
/// [`LinearVerifier::bounds`] and [`LinearVerifier::bounds_scoped`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearBounds<R: RealScalar> {
    matrix: R,
    preconditioner: Option<R>,
    defect: R,
    inverse: Option<R>,
    condition: Option<R>,
}
impl<R: RealScalar> LinearBounds<R> {
    /// Returns an upper bound on `‖A‖`.
    pub const fn matrix(&self) -> R {
        self.matrix
    }
    /// Returns an upper bound on `‖R‖`, if computed.
    pub const fn preconditioner(&self) -> Option<R> {
        self.preconditioner
    }
    /// Returns `q`, an upper bound on `‖I − RA‖`. The matrix is proved
    /// nonsingular when `q < 1`.
    pub const fn defect(&self) -> R {
        self.defect
    }
    /// Returns an upper bound on `‖A⁻¹‖`, if computed, `q < 1` and the bound is
    /// finite.
    pub const fn inverse(&self) -> Option<R> {
        self.inverse
    }
    /// Returns an upper bound on the condition number `‖A‖ ‖A⁻¹‖`, if computed,
    /// `q < 1` and the bound is finite.
    pub const fn condition(&self) -> Option<R> {
        self.condition
    }
}
/// Status and diagnostics of a linear-system proof.
///
/// The norm bounds need a complete pass over the matrix, so a stopped proof
/// may have none.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct LinearVerificationReport<R: RealScalar> {
    status: LinearVerificationStatus,
    bounds: Option<LinearBounds<R>>,
    residual: Option<R>,
    correction: Option<R>,
}
impl<R: RealScalar> LinearVerificationReport<R> {
    /// Returns an empty report with status `NotRun`, for the report slots of
    /// [`LinearVerifier::solve_block_into`].
    pub const fn not_run() -> Self {
        Self::without_diagnostics(LinearVerificationStatus::NotRun)
    }
    const fn without_diagnostics(status: LinearVerificationStatus) -> Self {
        Self {
            status,
            bounds: None,
            residual: None,
            correction: None,
        }
    }
    /// Returns the status.
    pub const fn status(&self) -> LinearVerificationStatus {
        self.status
    }
    /// Returns the norm bounds of this proof.
    pub const fn bounds(&self) -> Option<LinearBounds<R>> {
        self.bounds
    }
    /// Returns an upper bound on `‖b − Ax̃‖` at the returned center `x̃`.
    ///
    /// A solve updates the bound of its starting point by the change of center
    /// times `‖A‖`, which can exceed the residual of `x̃` itself.
    pub const fn residual(&self) -> Option<R> {
        self.residual
    }
    /// Returns an upper bound on `‖R(b − Ax̃)‖`.
    pub const fn correction(&self) -> Option<R> {
        self.correction
    }
    /// Returns whether the status is `Verified`.
    pub fn is_verified(&self) -> bool {
        self.status == LinearVerificationStatus::Verified
    }
}
impl<R: RealScalar> Outcome for LinearVerificationReport<R> {
    fn not_run() -> Self {
        Self::not_run()
    }
    fn stopped(reason: scioxide_core::ExecutionStopReason) -> Self {
        Self::without_diagnostics(LinearVerificationStatus::Inconclusive(
            LinearInconclusive::Stopped(reason),
        ))
    }
    fn succeeded(&self) -> bool {
        self.is_verified()
    }
}
/// Borrowed proposed solution: a center `x̃` and error radii, one per
/// component.
///
/// The proposal claims that the solution `x*` satisfies
/// `|Re(x*ᵢ − x̃ᵢ)| ≤ Re rᵢ` and `|Im(x*ᵢ − x̃ᵢ)| ≤ Im rᵢ`. Radii are
/// nonnegative, and their imaginary parts are zero for real systems. The
/// proposal is not trusted; see [`LinearVerifier::verify`].
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_validated::LinearCertificateView;
///
/// let (center, radii) = ([0.5_f64], [Complex::new(1e-12, 0.0)]);
/// let proposal = LinearCertificateView {
///     center: &center,
///     radii: &radii,
/// };
/// let mut boxes = [scioxide_validated::point_enclosure(0.0)?];
/// proposal.enclosure_into(&mut boxes)?;
/// assert!(boxes[0].re.contains(0.5));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LinearCertificateView<'a, S: NumericScalar> {
    /// Approximate solution `x̃`.
    pub center: &'a [S],
    /// Error radii of the real and imaginary parts.
    pub radii: &'a [Complex<S::Real>],
}
/// Mutable center and radii of a [`LinearCertificate`], the output of
/// [`LinearVerifier::solve_into`] and [`LinearVerifier::propose_into`].
///
/// ```
/// use scioxide_validated::LinearCertificate;
///
/// let mut certificate = LinearCertificate::<f64>::new(2)?;
/// let out = certificate.view_mut();
/// assert_eq!((out.center.len(), out.radii.len()), (2, 2));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub struct LinearCertificateViewMut<'a, S: NumericScalar> {
    /// Approximate solution `x̃`.
    pub center: &'a mut [S],
    /// Error radii of the real and imaginary parts.
    pub radii: &'a mut [Complex<S::Real>],
}
/// Owned center and error radii of a linear system's solution.
///
/// The storage proves nothing by itself: keep the report of the solve or
/// verification that filled it.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_validated::LinearCertificate;
///
/// let mut certificate = LinearCertificate::<f64>::new(1)?;
/// let out = certificate.view_mut();
/// out.center[0] = 2.0;
/// out.radii[0] = Complex::new(0.25, 0.0);
/// let boxes = certificate.enclosure()?;
/// assert!(boxes[0].re.contains(1.75) && boxes[0].re.contains(2.25));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LinearCertificate<S: NumericScalar> {
    center: Vec<S>,
    radii: Vec<Complex<S::Real>>,
}
impl<S: NumericScalar> LinearCertificate<S> {
    /// Allocates a proposal of `order` components, with zero center and radii.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if the storage cannot be reserved.
    pub fn new(order: usize) -> Result<Self> {
        Ok(Self {
            center: try_filled(order, S::ZERO, scioxide_core::quantity!("center"))?,
            radii: try_filled(
                order,
                Complex::new(S::Real::ZERO, S::Real::ZERO),
                scioxide_core::quantity!("radii"),
            )?,
        })
    }
    /// Returns the number of components.
    pub fn order(&self) -> usize {
        self.center.len()
    }
    /// Borrows the center and radii.
    pub fn view(&self) -> LinearCertificateView<'_, S> {
        LinearCertificateView {
            center: &self.center,
            radii: &self.radii,
        }
    }
    /// Borrows the center and radii for writing; a changed proposal must be
    /// verified again.
    pub fn view_mut(&mut self) -> LinearCertificateViewMut<'_, S> {
        LinearCertificateViewMut {
            center: &mut self.center,
            radii: &mut self.radii,
        }
    }
    /// Returns the rectangles `x̃ᵢ ± rᵢ`, rounded outward. This does not verify
    /// the proposal.
    ///
    /// # Errors
    ///
    /// - The errors of [`LinearCertificateView::enclosure_into`].
    /// - `AllocationFailure` if the output cannot be reserved.
    pub fn enclosure(&self) -> Result<Vec<IntervalBox<S::Real>>> {
        let mut output = try_filled(
            self.center.len(),
            zero()?,
            scioxide_core::quantity!("enclosure"),
        )?;
        self.view().enclosure_into(&mut output)?;
        Ok(output)
    }
}
impl<S: NumericScalar> LinearCertificateView<'_, S> {
    /// Validate this proposal of order `n` as the `argument`.
    /// # Errors
    /// Returns `DimensionMismatch` for a wrong order, `NonFinite` for NaN or infinite
    /// centers or radii, and invalid input for negative or (for real scalars)
    /// imaginary radii.
    fn validate(self, n: usize, argument: Argument) -> Result<()> {
        crate::shell::exact(n, self.center.len())?;
        crate::shell::exact(n, self.radii.len())?;
        if self.center.iter().any(|x| !x.has_finite_components())
            || self
                .radii
                .iter()
                .any(|r| !r.re.is_finite() || !r.im.is_finite())
        {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Input,
                "certificate center and radii must be finite",
            ));
        }
        if self.radii.iter().any(|r| {
            r.re < S::Real::ZERO || r.im < S::Real::ZERO || (S::IS_REAL && r.im != S::Real::ZERO)
        }) {
            return Err(invalid(argument, "invalid certificate radii"));
        }
        Ok(())
    }
    /// Writes the rectangles `x̃ᵢ ± rᵢ`, rounded outward, to `output`. This does
    /// not verify the proposal.
    ///
    /// # Errors
    ///
    /// - `DimensionMismatch` if `output`, the center and the radii differ in
    ///   length.
    /// - `NonFinite` if a center or radius is NaN or infinite.
    /// - `InvalidInput` if a radius is negative, a real system has a nonzero
    ///   imaginary radius, or the
    ///   [floating-point environment](crate#floating-point-environment) is not
    ///   supported.
    pub fn enclosure_into(self, output: &mut [IntervalBox<S::Real>]) -> Result<()> {
        self.validate(output.len(), scioxide_core::argument!("self"))?;
        crate::environment::check()?;
        // Preflight every fallible calculation before publishing any lane.
        for (&x, r) in self.center.iter().zip(self.radii) {
            Box2::point(x)?.add(Box2::symmetric(r.re, r.im)?)?;
        }
        for ((slot, &x), r) in output.iter_mut().zip(self.center).zip(self.radii) {
            *slot = pack(Box2::point(x)?.add(Box2::symmetric(r.re, r.im)?)?);
        }
        Ok(())
    }
}
