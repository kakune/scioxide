//! Condition numbers, inclusion-disk estimates and radius bounds for polynomial roots.
//!
//! All results are floating-point estimates. None is an outward-rounded
//! certificate.

use alloc::vec::Vec;
use core::marker::PhantomData;
use scioxide_core::AlgebraicScalar;
use scioxide_core::backend::{checked_add, try_filled, try_reserve};
use scioxide_core::{
    Capacity, Complex, ExecutionContext, InvalidInputReason, NumericScalar, Quantity, RealScalar,
    ReliableEstimate, SciOxideError,
};
use scioxide_core::{Scaled, ScaledValue};

use crate::backend::polynomial_iteration::WeierstrassCorrection;
use crate::horner::{ScaledHorner, complex_horner};
use crate::util::{bytes_of, minimum_len};
use crate::{ApproxPolynomial, PolynomialBasis};

/// Names the reason a root estimate could not be computed.
///
/// ```
/// use scioxide_poly::*;
///
/// let reason = RootEstimateFailure::ScalarRange;
/// assert!(matches!(
///     RootEstimate::<f64>::Inconclusive(reason),
///     RootEstimate::Inconclusive(RootEstimateFailure::ScalarRange)
/// ));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootEstimateFailure {
    /// The value overflows the scalar type, or a nonzero value underflows to
    /// zero.
    ScalarRange,
    /// Two candidates compare equal (`+0` and `−0` count as equal), so the
    /// Weierstrass correction is undefined. Candidates that are close but not
    /// equal do not cause this failure (see
    /// [`weierstrass_correction`](crate::backend::polynomial_iteration::weierstrass_correction)).
    CoincidentCandidates,
}

/// Holds a floating-point estimate, or the reason none is available.
///
/// A finite value is a heuristic estimate, not a verified bound. The other
/// variants replace sentinel values such as infinity or NaN.
///
/// ```
/// use scioxide_poly::*;
/// let estimate = RootEstimate::<f64>::Unavailable;
/// assert_eq!(estimate.value(), None);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RootEstimate<S: RealScalar> {
    /// A finite heuristic estimate.
    Finite(ReliableEstimate<S>),
    /// The denominator of the estimate is zero, for example `p′(z) = 0` at a
    /// multiple root.
    Singular,
    /// The quantity is not defined, for example the relative condition number
    /// at `z = 0`.
    Unavailable,
    /// The estimate could not be computed in the scalar type, or the candidates
    /// do not allow it.
    Inconclusive(RootEstimateFailure),
}
impl<S: RealScalar> RootEstimate<S> {
    /// Returns the finite estimate, or `None` for the other variants.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let estimate = RootEstimate::<f64>::Unavailable;
    /// assert_eq!(estimate.value(), None);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn value(self) -> Option<S> {
        match self {
            Self::Finite(value) => Some(value.value()),
            _ => None,
        }
    }

    /// Returns the larger of two estimates.
    ///
    /// A non-finite variant wins over a finite one, in the order `Singular`,
    /// `Inconclusive`, `Unavailable`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let estimate = RootEstimate::<f64>::Unavailable;
    /// assert_eq!(
    ///     estimate.maximum(RootEstimate::Singular),
    ///     RootEstimate::Singular
    /// );
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn maximum(self, other: Self) -> Self {
        match (self, other) {
            (Self::Singular, _) | (_, Self::Singular) => Self::Singular,
            (Self::Inconclusive(reason), _) | (_, Self::Inconclusive(reason)) => {
                Self::Inconclusive(reason)
            }
            (Self::Unavailable, _) | (_, Self::Unavailable) => Self::Unavailable,
            (Self::Finite(left), Self::Finite(right)) => {
                if left.value() >= right.value() {
                    self
                } else {
                    other
                }
            }
        }
    }
}
fn finite<S: RealScalar>(value: S) -> Result<RootEstimate<S>, SciOxideError> {
    Ok(RootEstimate::Finite(ReliableEstimate::new(
        value,
        scioxide_core::EstimateReliability::Heuristic,
    )?))
}

fn estimate<S: RealScalar>(value: Scaled<S>) -> Result<RootEstimate<S>, SciOxideError> {
    // Detect ordinary range failure before constructing an allocating error payload.
    if !value.abs_le(Scaled::from_value(S::MAX)?) {
        return Ok(RootEstimate::Inconclusive(RootEstimateFailure::ScalarRange));
    }
    match value.round() {
        // A nonzero estimate that rounds to zero has underflowed; reporting 0 would
        // wrongly claim a perfectly conditioned (or exact) quantity.
        Ok(rounded) if rounded.is_zero() && !value.is_zero() => {
            Ok(RootEstimate::Inconclusive(RootEstimateFailure::ScalarRange))
        }
        Ok(rounded) => finite(rounded),
        Err(ref error)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            ) =>
        {
            Ok(RootEstimate::Inconclusive(RootEstimateFailure::ScalarRange))
        }
        Err(error) => Err(error),
    }
}
fn ratio<S: RealScalar>(
    numerator: Scaled<S>,
    denominator: Scaled<S>,
) -> Result<RootEstimate<S>, SciOxideError> {
    if denominator.is_zero() {
        return Ok(RootEstimate::Singular);
    }
    estimate(numerator.div(denominator)?)
}

/// Holds first-order condition numbers of a polynomial and of its root at a
/// point `z`.
///
/// Let `B(z)` be the perturbation envelope of the coefficients `aₖ` of `p` in its
/// basis `φₖ`:
///
/// ```text
/// monomial basis:    B(z) = Σₖ |aₖ| |z|ᵏ
/// orthogonal basis:  B(z) = maxₖ |aₖ| · Σₖ |φₖ(z)|
/// ```
///
/// The monomial envelope allows each coefficient to change relative to its own
/// magnitude. The orthogonal envelope allows each coefficient to change relative
/// to the largest one. The report holds `B/|p(z)|`, `B/|p′(z)|` and
/// `B/(|z| |p′(z)|)`.
///
/// # References
///
/// N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
/// SIAM, 2002, §5.1.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_poly::*;
/// let coefficients = [-1.0_f64, 0.0, 1.0];
/// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
/// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
/// let report = result.roots()[0].condition();
/// assert!(report.absolute().value().is_some());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootConditionReport<S: RealScalar> {
    coefficient: RootEstimate<S>,
    absolute: RootEstimate<S>,
    relative: RootEstimate<S>,
}
impl<S: RealScalar> RootConditionReport<S> {
    /// Returns the condition number `B(z)/|p(z)|` of the value `p(z)`.
    ///
    /// Its reciprocal is the relative backward error of `z` as a root. It is
    /// `Singular` when `p(z) = 0`.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// let report = result.roots()[0].condition();
    /// assert!(matches!(report.coefficient(), RootEstimate::Singular));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn coefficient(self) -> RootEstimate<S> {
        self.coefficient
    }
    /// Returns the absolute condition number `B(z)/|p′(z)|` of a simple root at `z`.
    ///
    /// It is `Singular` when `p′(z) = 0`.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// let report = result.roots()[0].condition();
    /// // Σ|aₖ||z|ᵏ = 2 and |p′(−1)| = 2 for p = x² − 1 at z = −1.
    /// let absolute = report.absolute().value();
    /// assert!(absolute.is_some_and(|c| (c - 1.0).abs() < 1e-13));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn absolute(self) -> RootEstimate<S> {
        self.absolute
    }
    /// Returns the relative condition number `B(z)/(|z| |p′(z)|)` of a simple
    /// root at `z`.
    ///
    /// It is `Unavailable` at `z = 0` and `Singular` when `p′(z) = 0`.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// let report = result.roots()[0].condition();
    /// // Σ|aₖ||z|ᵏ = 2, |z| = 1 and |p′(−1)| = 2 for p = x² − 1 at z = −1.
    /// let relative = report.relative().value();
    /// assert!(relative.is_some_and(|c| (c - 1.0).abs() < 1e-13));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn relative(self) -> RootEstimate<S> {
        self.relative
    }
}
/// Forms the condition report from a basis-specific envelope, value and
/// derivative in extended exponent range.
pub(crate) fn condition_from_scaled<S: RealScalar>(
    bound: Scaled<S>,
    value: Scaled<S>,
    derivative: Scaled<S>,
    point: Scaled<S>,
) -> Result<RootConditionReport<S>, SciOxideError> {
    // With the chosen coefficient-perturbation envelope `bound`,
    // `bound / |p(z)|` is the reciprocal backward
    // error of `z` as a root, and `bound / |p'(z)|` the first-order root
    // displacement per unit relative coefficient perturbation (absolute, or
    // relative to `|z|`). A vanishing denominator is reported as `Singular`
    // instead of dividing.
    Ok(RootConditionReport {
        coefficient: ratio(bound, value)?,
        absolute: ratio(bound, derivative)?,
        relative: if point.is_zero() {
            RootEstimate::Unavailable
        } else {
            ratio(bound, derivative.mul(point)?)?
        },
    })
}

impl<C: NumericScalar, B: PolynomialBasis<C>> ApproxPolynomial<C, B> {
    /// Computes the condition numbers of `p(z)` and of a root at `z` in the
    /// polynomial's own basis.
    ///
    /// The envelope `B(z)` is described on [`RootConditionReport`]. Monomial and
    /// orthogonal bases use the formulas given there; the other bases use their
    /// componentwise evaluation envelope. The polynomial is not converted to
    /// monomial form. The cost is one evaluation of `p` and `p′`.
    ///
    /// A polynomial with real coefficients accepts only real points. For a
    /// nonreal `z`, construct the polynomial with complex coefficients. The
    /// Bernstein basis allocates one vector of `degree + 1` entries; the other
    /// bases allocate nothing.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `point` is NaN or infinite.
    /// - Errors of the basis evaluation, such as overflow.
    /// - `AllocationFailure` if the Bernstein scratch cannot be allocated.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let p = ApproxPolynomial::monomial(vec![1.0_f64, 2.0, 1.0])?;
    /// let report = p.root_condition(0.5)?;
    /// assert!(report.absolute().value().is_some());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn root_condition(&self, point: C) -> Result<RootConditionReport<C::Real>, SciOxideError> {
        validate_complex(as_complex(point))?;
        let mut scratch = try_filled(
            self.basis().scratch_len(),
            C::ZERO,
            scioxide_core::quantity!("evaluation_scratch"),
        )?;
        self.basis().root_condition_prepared(point, &mut scratch)
    }
}

fn as_complex<C: NumericScalar>(value: C) -> Complex<C::Real> {
    Complex::new(value.real_component(), value.imaginary_component())
}
fn magnitude<C: NumericScalar>(value: C) -> Result<Scaled<C::Real>, SciOxideError> {
    ScaledValue::from_value(as_complex(value))?.magnitude()
}
fn validate_complex<S: RealScalar>(value: Complex<S>) -> Result<(), SciOxideError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "coefficients and candidates must be finite",
        ))
    }
}
/// Validates coefficients, strips exact leading zeros and rejects the zero polynomial.
fn coefficients<C: NumericScalar>(values: &[C]) -> Result<&[C], SciOxideError> {
    let values = crate::util::canonical_coefficients(values, scioxide_core::argument!("values"))?;
    if values == [C::ZERO] {
        return Err(SciOxideError::invalid_input(
            scioxide_core::argument!("values"),
            InvalidInputReason::OutOfRange,
            "the zero polynomial has no finite root set",
        ));
    }
    Ok(values)
}

/// Holds Cauchy bounds on the moduli of the roots of a monomial-basis polynomial.
///
/// See [`polynomial_root_radii`] for the definitions.
///
/// ```
/// use scioxide_poly::*;
/// let report = polynomial_root_radii(&[-1.0_f64, 0.0, 1.0])?;
/// assert_eq!(report.degree(), 2);
/// assert_eq!(report.zero_roots(), 0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootRadiusReport<S: RealScalar> {
    outer: RootEstimate<S>,
    inner_nonzero: RootEstimate<S>,
    zero_roots: usize,
    degree: usize,
}
impl<S: RealScalar> RootRadiusReport<S> {
    /// Returns the upper bound `1 + maxₖ<ₙ |aₖ|/|aₙ|` on all root moduli.
    ///
    /// It is `Unavailable` for a constant polynomial.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let report = polynomial_root_radii(&[-1.0_f64, 0.0, 1.0])?;
    /// assert!(report.outer().value().is_some_and(|radius| radius >= 1.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn outer(self) -> RootEstimate<S> {
        self.outer
    }
    /// Returns the lower bound on the moduli of the nonzero roots.
    ///
    /// It is `Unavailable` when the polynomial has no nonzero roots.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let report = polynomial_root_radii(&[-1.0_f64, 0.0, 1.0])?;
    /// let inner = report.inner_nonzero().value();
    /// assert!(inner.is_some_and(|radius| radius <= 1.0 && radius > 0.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn inner_nonzero(self) -> RootEstimate<S> {
        self.inner_nonzero
    }
    /// Returns the multiplicity of the root at zero: the number of exactly zero
    /// low-order coefficients.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let report = polynomial_root_radii(&[-1.0_f64, 0.0, 1.0])?;
    /// assert_eq!(report.zero_roots(), 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn zero_roots(self) -> usize {
        self.zero_roots
    }
    /// Returns the degree after removing exactly zero leading coefficients.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let report = polynomial_root_radii(&[-1.0_f64, 0.0, 1.0])?;
    /// assert_eq!(report.degree(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn degree(self) -> usize {
        self.degree
    }
}

/// Computes Cauchy bounds on the root moduli of `p(z) = Σₖ aₖ zᵏ`.
///
/// `values` holds the real or complex coefficients `a₀, …, aₙ` in ascending
/// order. Exactly zero leading coefficients are removed; small nonzero ones are
/// kept. The `m` exactly zero low-order coefficients are roots at zero. For the
/// remaining coefficients `aₘ, …, aₙ`, every root satisfies
///
/// ```text
/// 1 / (1 + maxₖ>ₘ |aₖ|/|aₘ|)  ≤  |z|  <  1 + maxₖ<ₙ |aₖ|/|aₙ|,   z ≠ 0.
/// ```
///
/// The lower bound applies the Cauchy bound to the reversed polynomial. The
/// bounds are computed in floating point and are not rigorous. The cost is O(n)
/// time with no allocation.
///
/// # Errors
///
/// - `InvalidInput` if `values` is empty or every coefficient is zero.
/// - `NonFinite` if a coefficient is NaN or infinite.
///
/// # References
///
/// M. Marden, *Geometry of Polynomials*, 2nd ed., AMS, 1966, §27.
///
/// ```
/// use scioxide_poly::*;
/// let report = polynomial_root_radii(&[-1.0_f64, 0.0, 1.0])?;
/// assert_eq!(report.degree(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn polynomial_root_radii<C: NumericScalar>(
    values: &[C],
) -> Result<RootRadiusReport<C::Real>, SciOxideError> {
    let values = coefficients(values)?;
    let degree = values.len() - 1;
    if degree == 0 {
        return Ok(RootRadiusReport {
            outer: RootEstimate::Unavailable,
            inner_nonzero: RootEstimate::Unavailable,
            zero_roots: 0,
            degree,
        });
    }
    // Exact zero low-order coefficients are roots at the origin; removing them
    // first keeps the inner bound meaningful for the nonzero roots.
    let zero_roots = values.iter().take_while(|value| **value == C::ZERO).count();
    let reduced = &values[zero_roots..];
    if reduced.len() == 1 {
        return Ok(RootRadiusReport {
            outer: finite(C::Real::ZERO)?,
            inner_nonzero: RootEstimate::Unavailable,
            zero_roots,
            degree,
        });
    }
    let cauchy = |leading: C, rest: &[C]| -> Result<Scaled<C::Real>, SciOxideError> {
        let leading = magnitude(leading)?;
        let largest = rest.iter().try_fold(Scaled::ZERO, |largest, value| {
            let value = magnitude(*value)?;
            Ok::<_, SciOxideError>(if largest.abs_le(value) {
                value
            } else {
                largest
            })
        })?;
        Scaled::from_value(C::Real::ONE)?.add(largest.div(leading)?)
    };
    // Cauchy's bound: every root satisfies `|z| < 1 + max_{i<n} |a_i| / |a_n|`.
    // Applied to the reversed polynomial (roots `1/z`) it gives a lower bound on
    // the nonzero root moduli.
    let outer = cauchy(reduced[reduced.len() - 1], &reduced[..reduced.len() - 1])?;
    let reversed = cauchy(reduced[0], &reduced[1..])?;
    Ok(RootRadiusReport {
        outer: estimate(outer)?,
        inner_nonzero: estimate(Scaled::from_value(C::Real::ONE)?.div(reversed)?)?,
        zero_roots,
        degree,
    })
}

/// Computes the monomial condition report at a complex point in O(n) time.
///
/// Rejects zero or empty polynomials and non-finite coefficients or points.
pub(crate) fn polynomial_root_condition<C: NumericScalar>(
    values: &[C],
    point: Complex<C::Real>,
) -> Result<RootConditionReport<C::Real>, SciOxideError> {
    let values = coefficients(values)?;
    validate_complex(point)?;
    let evaluation = complex_horner::<_, 1, true>(values, point)?.into_scaled()?;
    condition_from_scaled(
        evaluation.bound,
        evaluation.value.magnitude()?,
        evaluation.derivative.magnitude()?,
        ScaledValue::from_value(point)?.magnitude()?,
    )
}

/// Holds an estimated inclusion disk in the complex plane.
///
/// The disk is a floating-point estimate, not a verified enclosure.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_poly::*;
/// let coefficients = [-1.0_f64, 0.0, 1.0];
/// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
/// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
/// let disk = result.roots()[0]
///     .disk()
///     .ok_or_else(|| scioxide_core::SciOxideError::singular(None))?;
/// assert!((disk.center().re + 1.0).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootDiskEstimate<S: RealScalar> {
    center: Complex<S>,
    radius: ReliableEstimate<S>,
}
impl<S: RealScalar> RootDiskEstimate<S> {
    /// Returns the center of the disk.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// let disk = result.roots()[0]
    ///     .disk()
    ///     .ok_or_else(|| scioxide_core::SciOxideError::singular(None))?;
    /// assert_eq!(disk.center(), Complex::new(-1.0, 0.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn center(self) -> Complex<S> {
        self.center
    }
    /// Returns the radius of the disk.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// let disk = result.roots()[0]
    ///     .disk()
    ///     .ok_or_else(|| scioxide_core::SciOxideError::singular(None))?;
    /// assert!(disk.radius().value() >= 0.0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn radius(self) -> ReliableEstimate<S> {
        self.radius
    }
}
/// Holds the diagnostics of one candidate root.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_poly::*;
/// let coefficients = [-1.0_f64, 0.0, 1.0];
/// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
/// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
/// let root = result.roots()[0];
/// assert!((root.candidate().re + 1.0).abs() < 1e-13);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootDiagnostic<S: RealScalar> {
    index: usize,
    candidate: Complex<S>,
    condition: RootConditionReport<S>,
    disk: Option<RootDiskEstimate<S>>,
    separation: RootEstimate<S>,
    cluster: usize,
}
impl<S: RealScalar> RootDiagnostic<S> {
    /// Returns the index of this candidate in the input slice.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// let root = result.roots()[0];
    /// assert_eq!(root.input_index(), 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn input_index(self) -> usize {
        self.index
    }
    /// Returns the candidate root as given.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// let root = result.roots()[0];
    /// assert_eq!(root.candidate(), Complex::new(-1.0, 0.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn candidate(self) -> Complex<S> {
        self.candidate
    }
    /// Returns the monomial-basis condition numbers at this candidate.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// let root = result.roots()[0];
    /// assert!(root.condition().absolute().value().is_some());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn condition(self) -> RootConditionReport<S> {
        self.condition
    }
    /// Returns the inclusion disk around this candidate.
    ///
    /// It is `None` when the diagnostics status is
    /// [`Inconclusive`](RootDiagnosticsStatus::Inconclusive).
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// let root = result.roots()[0];
    /// assert!(root.disk().is_some());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn disk(self) -> Option<RootDiskEstimate<S>> {
        self.disk
    }
    /// Returns the distance to the nearest other candidate.
    ///
    /// This is the separation of the candidates, not of the exact roots. It is
    /// `Unavailable` for a single candidate.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// let root = result.roots()[0];
    /// assert_eq!(root.candidate_separation().value(), Some(2.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn candidate_separation(self) -> RootEstimate<S> {
        self.separation
    }
    /// Returns the index of this candidate's cluster in
    /// [`RootDiagnostics::clusters`].
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// let root = result.roots()[0];
    /// assert!(root.cluster() < result.clusters().len());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn cluster(self) -> usize {
        self.cluster
    }
}
/// Holds a cluster: a connected component of overlapping inclusion disks.
///
/// In exact arithmetic, a cluster of `k` disks that does not meet the others
/// contains exactly `k` roots. With the floating-point disks of this crate the
/// count is an estimate.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_poly::*;
/// let coefficients = [-1.0_f64, 0.0, 1.0];
/// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
/// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
/// let cluster = result.clusters()[0];
/// assert_eq!(cluster.candidate_count(), 1);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootCluster<S: RealScalar> {
    disk: Option<RootDiskEstimate<S>>,
    count: usize,
    separation: RootEstimate<S>,
}
impl<S: RealScalar> RootCluster<S> {
    /// Returns a disk that covers every disk of the cluster.
    ///
    /// The center is that of the cluster's first candidate; the disk is not the
    /// smallest enclosing one. It is `None` if the radius overflows.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// let cluster = result.clusters()[0];
    /// assert!(cluster.disk().is_some());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn disk(self) -> Option<RootDiskEstimate<S>> {
        self.disk
    }
    /// Returns the number of candidates in the cluster.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// let cluster = result.clusters()[0];
    /// assert_eq!(cluster.candidate_count(), 1);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn candidate_count(self) -> usize {
        self.count
    }
    /// Returns the smallest gap between this cluster's disk and the disk of any
    /// other cluster.
    ///
    /// The gap is zero when the covering disks overlap. It is `Unavailable` when
    /// there is only one cluster.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// let cluster = result.clusters()[0];
    /// let separation = cluster.separation().value();
    /// assert!(separation.is_some_and(|distance| distance > 0.0));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn separation(self) -> RootEstimate<S> {
        self.separation
    }
}
/// Reports whether every candidate received an inclusion disk.
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_poly::*;
/// let coefficients = [-1.0_f64, 0.0, 1.0];
/// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
/// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
/// assert_eq!(result.roots().len(), 2);
/// assert_eq!(result.status(), RootDiagnosticsStatus::Estimated);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootDiagnosticsStatus {
    /// Every candidate has a Weierstrass inclusion disk, and the clusters are
    /// formed from them.
    Estimated,
    /// A disk could not be computed. The result then holds one cluster with all
    /// candidates and the Cauchy disk `|z| ≤ outer` centered at zero.
    Inconclusive(RootEstimateFailure),
}
/// Holds the result of [`polynomial_root_diagnostics`].
///
/// The value can be reused as the output of
/// [`polynomial_root_diagnostics_into`].
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_poly::*;
/// let coefficients = [-1.0_f64, 0.0, 1.0];
/// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
/// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
/// assert_eq!(result.roots().len(), 2);
/// assert_eq!(result.status(), RootDiagnosticsStatus::Estimated);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RootDiagnostics<S: RealScalar> {
    radii: Option<RootRadiusReport<S>>,
    roots: Vec<RootDiagnostic<S>>,
    clusters: Vec<RootCluster<S>>,
    status: RootDiagnosticsStatus,
}
impl<S: RealScalar> Default for RootDiagnostics<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> RootDiagnostics<S> {
    /// Creates an empty result without allocating.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let req = RootDiagnosticsWorkspace::<f64>::workspace_requirements(2)?;
    /// let mut result = RootDiagnostics::<f64>::new();
    /// result.try_reserve(&req)?;
    /// assert!(result.roots().is_empty());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn new() -> Self {
        Self {
            radii: None,
            roots: Vec::new(),
            clusters: Vec::new(),
            status: RootDiagnosticsStatus::Estimated,
        }
    }
    /// Reserves capacity for the given requirements and keeps the current result.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if memory cannot be reserved. Capacity added before
    ///   the failure is kept.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let req = RootDiagnosticsWorkspace::<f64>::workspace_requirements(2)?;
    /// let mut result = RootDiagnostics::<f64>::new();
    /// result.try_reserve(&req)?;
    /// assert!(result.roots().is_empty());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_reserve(
        &mut self,
        requirements: &RootDiagnosticsRequirements<S>,
    ) -> Result<(), SciOxideError> {
        reserve(
            &mut self.roots,
            requirements.degree,
            scioxide_core::quantity!("diagnostic_roots"),
        )?;
        reserve(
            &mut self.clusters,
            requirements.degree,
            scioxide_core::quantity!("diagnostic_clusters"),
        )
    }
    fn fits(&self, degree: usize) -> Result<(), SciOxideError> {
        minimum_len(degree, self.roots.capacity())?;
        minimum_len(degree, self.clusters.capacity())
    }
    /// Returns the Cauchy radius bounds, or `None` before the first successful
    /// call.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// assert_eq!(result.radii().map(|radii| radii.degree()), Some(2));
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn radii(&self) -> Option<RootRadiusReport<S>> {
        self.radii
    }
    /// Returns the per-candidate diagnostics, sorted by real part and then by
    /// imaginary part.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// assert_eq!(result.roots().len(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn roots(&self) -> &[RootDiagnostic<S>] {
        &self.roots
    }
    /// Returns the clusters, ordered by their first candidate in the order of
    /// [`Self::roots`].
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// assert_eq!(result.clusters().len(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn clusters(&self) -> &[RootCluster<S>] {
        &self.clusters
    }
    /// Returns whether every candidate received an inclusion disk.
    ///
    /// ```
    /// use scioxide_core::Complex;
    /// use scioxide_poly::*;
    /// let coefficients = [-1.0_f64, 0.0, 1.0];
    /// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
    /// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
    /// assert_eq!(result.status(), RootDiagnosticsStatus::Estimated);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn status(&self) -> RootDiagnosticsStatus {
        self.status
    }
}
/// Describes the storage needed to analyze a polynomial of a given degree.
///
/// The same value sizes both [`RootDiagnosticsWorkspace`] and the
/// [`RootDiagnostics`] output. Each holds one root record and one cluster record
/// per candidate. The workspace also holds one index per candidate. No memory
/// limit applies.
///
/// ```
/// use scioxide_poly::*;
/// let req = RootDiagnosticsWorkspace::<f64>::workspace_requirements(2)?;
/// assert_eq!(req.degree(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootDiagnosticsRequirements<S: RealScalar> {
    degree: usize,
    scalar: PhantomData<fn() -> S>,
}
impl<S: RealScalar> RootDiagnosticsRequirements<S> {
    /// Returns the number of candidates, which equals the degree.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let req = RootDiagnosticsWorkspace::<f64>::workspace_requirements(2)?;
    /// assert_eq!(req.degree(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn degree(self) -> usize {
        self.degree
    }
    /// Returns the workspace size in bytes, excluding `Vec` headers.
    ///
    /// The output needs the same number of bytes minus the indices.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the byte count overflows `usize`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let req = RootDiagnosticsWorkspace::<f64>::workspace_requirements(2)?;
    /// assert!(req.required_bytes()? > 0);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn required_bytes(self) -> Result<usize, SciOxideError> {
        let quantity = scioxide_core::quantity!("diagnostic_bytes");
        checked_add(
            checked_add(
                bytes_of::<RootDiagnostic<S>>(self.degree)?,
                bytes_of::<RootCluster<S>>(self.degree)?,
                quantity,
            )?,
            bytes_of::<usize>(self.degree)?,
            quantity,
        )
    }
}
/// Holds reusable scratch storage for [`polynomial_root_diagnostics_into`].
///
/// The workspace grows on request and follows
/// [Plans and workspaces](scioxide_core#plans-and-workspaces).
///
/// ```
/// use scioxide_core::{Capacity, Complex, ExecutionContext};
/// use scioxide_poly::{
///     RootDiagnostics, RootDiagnosticsWorkspace, polynomial_root_diagnostics_into,
/// };
/// let requirements = RootDiagnosticsWorkspace::<f64>::workspace_requirements(2)?;
/// let mut workspace = RootDiagnosticsWorkspace::new();
/// let mut output = RootDiagnostics::new();
/// assert_eq!(workspace.check(&requirements)?, Capacity::NeedsGrowth);
/// workspace.try_reserve(&requirements)?;
/// output.try_reserve(&requirements)?;
/// let roots = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
/// let cx = ExecutionContext::sequential();
/// let p = [-1.0_f64, 0.0, 1.0];
/// polynomial_root_diagnostics_into(&p, &roots, &mut output, &mut workspace, cx)?;
/// assert_eq!(output.roots().len(), 2);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct RootDiagnosticsWorkspace<S: RealScalar> {
    staged: RootDiagnostics<S>,
    parents: Vec<usize>,
}
impl<S: RealScalar> Default for RootDiagnosticsWorkspace<S> {
    fn default() -> Self {
        Self::new()
    }
}
impl<S: RealScalar> RootDiagnosticsWorkspace<S> {
    /// Creates an empty workspace without allocating.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let req = RootDiagnosticsWorkspace::<f64>::workspace_requirements(2)?;
    /// let mut workspace = RootDiagnosticsWorkspace::<f64>::new();
    /// workspace.try_reserve(&req)?;
    /// assert_eq!(workspace.check(&req)?, scioxide_core::Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub const fn new() -> Self {
        Self {
            staged: RootDiagnostics::new(),
            parents: Vec::new(),
        }
    }
    /// Returns the storage needed for `degree` candidates.
    ///
    /// Root diagnostics have no plan, so this function is on the workspace type.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size in bytes exceeds `isize::MAX`.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let req = RootDiagnosticsWorkspace::<f64>::workspace_requirements(2)?;
    /// let mut workspace = RootDiagnosticsWorkspace::<f64>::new();
    /// workspace.try_reserve(&req)?;
    /// assert_eq!(workspace.check(&req)?, scioxide_core::Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace_requirements(
        degree: usize,
    ) -> Result<RootDiagnosticsRequirements<S>, SciOxideError> {
        let requirements = RootDiagnosticsRequirements {
            degree,
            scalar: PhantomData,
        };
        if requirements.required_bytes()? > isize::MAX as usize {
            return Err(SciOxideError::size_overflow(scioxide_core::quantity!(
                "diagnostic_bytes"
            )));
        }
        Ok(requirements)
    }
    /// Creates a workspace large enough for `degree` candidates.
    ///
    /// # Errors
    ///
    /// - `SizeOverflow` if the size in bytes exceeds `isize::MAX`.
    /// - `AllocationFailure` if memory cannot be reserved.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let req = RootDiagnosticsWorkspace::<f64>::workspace_requirements(2)?;
    /// let mut workspace = RootDiagnosticsWorkspace::<f64>::new();
    /// workspace.try_reserve(&req)?;
    /// assert_eq!(workspace.check(&req)?, scioxide_core::Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn workspace(degree: usize) -> Result<Self, SciOxideError> {
        let mut workspace = Self::new();
        workspace.try_reserve(&Self::workspace_requirements(degree)?)?;
        Ok(workspace)
    }
    /// Reports whether the current capacity meets `requirements`.
    ///
    /// # Errors
    ///
    /// Never returns an error. The `Result` keeps the signature uniform with
    /// other workspaces.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let req = RootDiagnosticsWorkspace::<f64>::workspace_requirements(2)?;
    /// let mut workspace = RootDiagnosticsWorkspace::<f64>::new();
    /// workspace.try_reserve(&req)?;
    /// assert_eq!(workspace.check(&req)?, scioxide_core::Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn check(
        &self,
        requirements: &RootDiagnosticsRequirements<S>,
    ) -> Result<Capacity, SciOxideError> {
        Ok(if self.fits(requirements.degree).is_ok() {
            Capacity::Sufficient
        } else {
            Capacity::NeedsGrowth
        })
    }
    /// Grows the workspace to meet `requirements`.
    ///
    /// Existing capacity that is large enough is reused. The output needs its
    /// own [`RootDiagnostics::try_reserve`] call.
    ///
    /// # Errors
    ///
    /// - `AllocationFailure` if memory cannot be reserved. Capacity added before
    ///   the failure is kept.
    ///
    /// ```
    /// use scioxide_poly::*;
    /// let req = RootDiagnosticsWorkspace::<f64>::workspace_requirements(2)?;
    /// let mut workspace = RootDiagnosticsWorkspace::<f64>::new();
    /// workspace.try_reserve(&req)?;
    /// assert_eq!(workspace.check(&req)?, scioxide_core::Capacity::Sufficient);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn try_reserve(
        &mut self,
        requirements: &RootDiagnosticsRequirements<S>,
    ) -> Result<(), SciOxideError> {
        self.staged.try_reserve(requirements)?;
        reserve(
            &mut self.parents,
            requirements.degree,
            scioxide_core::quantity!("union_find_indices"),
        )
    }
    fn fits(&self, degree: usize) -> Result<(), SciOxideError> {
        self.staged.fits(degree)?;
        minimum_len(degree, self.parents.capacity())
    }
}
/// Reserves capacity for `capacity` elements in total.
fn reserve<T>(
    values: &mut Vec<T>,
    capacity: usize,
    quantity: Quantity,
) -> Result<(), SciOxideError> {
    try_reserve(values, capacity.saturating_sub(values.len()), quantity)
}

/// Computes condition numbers, inclusion disks and clusters for a full set of
/// candidate roots of a monomial-basis polynomial.
///
/// `values` holds the coefficients `a₀, …, aₙ` of `p(z) = Σₖ aₖ zᵏ` in ascending
/// order; exactly zero leading coefficients are removed. `roots` holds one
/// candidate `zᵢ` per root, for example the output of a root finder. For each
/// candidate the function computes the [`RootConditionReport`] and the
/// Weierstrass correction
///
/// ```text
/// wᵢ = p(zᵢ) / (aₙ Πⱼ≠ᵢ (zᵢ − zⱼ)).
/// ```
///
/// In exact arithmetic the disks with centers `zᵢ − wᵢ` and radii
/// `(n − 1)|wᵢ|` contain all roots, and a connected union of `k` disks that is
/// disjoint from the others contains exactly `k` roots. The radius here adds a
/// rounding term `n · 8(n + 1)ε B(zᵢ) / (|aₙ| |Πⱼ≠ᵢ (zᵢ − zⱼ)|)`, where `ε` is
/// the machine epsilon and `B` the envelope of [`RootConditionReport`]. Because
/// the disks are computed in floating point, they are estimates, not
/// enclosures. Overlapping disks are merged into [`RootCluster`]s.
///
/// If two candidates are equal or a disk overflows, the status is
/// [`RootDiagnosticsStatus::Inconclusive`] and all candidates form one cluster
/// with the Cauchy disk of [`polynomial_root_radii`]. The result does not depend
/// on the order of the candidates. The cost is O(n²) time and O(n) memory. Use
/// [`polynomial_root_diagnostics_into`] to reuse storage.
///
/// # Errors
///
/// - `InvalidInput` if `values` is empty or every coefficient is zero.
/// - `NonFinite` if a coefficient or a candidate is NaN or infinite.
/// - `DimensionMismatch` if the number of candidates differs from the degree.
/// - `AllocationFailure` if the result cannot be allocated.
///
/// # Examples
///
/// ```
/// use scioxide_core::Complex;
/// use scioxide_poly::*;
///
/// // x² − 1 has the roots ±1.
/// let coefficients = [-1.0_f64, 0.0, 1.0];
/// let candidates = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
/// let result = polynomial_root_diagnostics(&coefficients, &candidates)?;
/// assert_eq!(result.roots().len(), 2);
/// assert_eq!(result.status(), RootDiagnosticsStatus::Estimated);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn polynomial_root_diagnostics<C: NumericScalar>(
    values: &[C],
    roots: &[Complex<C::Real>],
) -> Result<RootDiagnostics<C::Real>, SciOxideError> {
    let requirements = RootDiagnosticsWorkspace::workspace_requirements(roots.len())?;
    let mut output = RootDiagnostics::new();
    let mut workspace = RootDiagnosticsWorkspace::new();
    output.try_reserve(&requirements)?;
    workspace.try_reserve(&requirements)?;
    polynomial_root_diagnostics_into(
        values,
        roots,
        &mut output,
        &mut workspace,
        scioxide_core::ExecutionContext::sequential_native(),
    )?;
    Ok(output)
}
/// Computes [`polynomial_root_diagnostics`] into an output you provide.
///
/// Reserve both `output` and `workspace` with the requirements from
/// [`RootDiagnosticsWorkspace::workspace_requirements`] for the degree. The call
/// then does not allocate. It checks `context` for cancellation and deadlines
/// between candidates. On error, `output` is unchanged
/// ([Output on error](scioxide_core#output-on-error)).
///
/// # Errors
///
/// - The errors of [`polynomial_root_diagnostics`] other than
///   `AllocationFailure`.
/// - `DimensionMismatch` if `output` or `workspace` has too little capacity.
/// - `ExecutionStopped` if `context` is cancelled or its deadline passes.
///
/// ```
/// use scioxide_core::{Complex, ExecutionContext};
/// use scioxide_poly::*;
/// let req = RootDiagnosticsWorkspace::<f64>::workspace_requirements(2)?;
/// let mut workspace = RootDiagnosticsWorkspace::<f64>::workspace(2)?;
/// let mut result = RootDiagnostics::new();
/// result.try_reserve(&req)?;
/// let roots = [Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)];
/// let cx = ExecutionContext::sequential();
/// let p = [-1.0_f64, 0.0, 1.0];
/// polynomial_root_diagnostics_into(&p, &roots, &mut result, &mut workspace, cx)?;
/// assert_eq!(result.roots().len(), 2);
/// assert_eq!(result.status(), RootDiagnosticsStatus::Estimated);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn polynomial_root_diagnostics_into<C: NumericScalar>(
    values: &[C],
    roots: &[Complex<C::Real>],
    output: &mut RootDiagnostics<C::Real>,
    workspace: &mut RootDiagnosticsWorkspace<C::Real>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    context.check()?;
    analyze(values, roots, output, workspace, context)
}

fn distance<S: RealScalar>(
    left: Complex<S>,
    right: Complex<S>,
) -> Result<Scaled<S>, SciOxideError> {
    ScaledValue::from_value(left)?
        .sub(ScaledValue::from_value(right)?)?
        .magnitude()
}
fn local_disk<C: NumericScalar>(
    values: &[C],
    roots: &[Complex<C::Real>],
    index: usize,
    evaluation: &ScaledHorner<C::Real>,
    order: &[usize],
    context: ExecutionContext<'_>,
) -> Result<Result<RootDiskEstimate<C::Real>, RootEstimateFailure>, SciOxideError> {
    let point = roots[index];
    // Weierstrass (Durand-Kerner) correction
    // `w_i = p(z_i) / (a_n prod_{j != i} (z_i - z_j))`. By the Braess-Hadeler /
    // Carstensen inclusion theorem, in exact arithmetic the disks centered at
    // `z_i - w_i` with radius `(n - 1) |w_i|` cover all roots, and a connected
    // component of `k` disks isolated from the others contains exactly `k`. The
    // floor below adds a rounding model, not a proof. The product runs in the
    // shared sorted `order` so the result does not depend on how the caller
    // ordered the candidates. The shared kernel owns the coincidence rule:
    // only exactly equal candidates leave the correction undefined.
    context.check()?;
    let leading = ScaledValue::from_value(as_complex(values[values.len() - 1]))?;
    let others = order
        .iter()
        .filter(|&&other| other != index)
        .map(|&other| roots[other]);
    let (correction, product) = match crate::backend::polynomial_iteration::weierstrass_correction(
        point,
        others,
        evaluation.value.div(leading)?,
    )? {
        WeierstrassCorrection::Defined {
            correction,
            product,
        } => (correction, product),
        WeierstrassCorrection::Coincident => {
            return Ok(Err(RootEstimateFailure::CoincidentCandidates));
        }
    };
    let denominator = leading.magnitude()?.mul(product.magnitude()?)?;
    let center = ScaledValue::from_value(point)?.sub(correction)?;
    let n = C::Real::try_from_usize_exact(roots.len())?;
    // A rounding floor prevents an accidentally zero residual from implying an exact disk.
    // This is an error model, not directed interval arithmetic.
    let floor = evaluation
        .bound
        // The floor models the rounding error of evaluating `p(z_i)` by Horner,
        // roughly `(n + 1) eps * sum |a_k| |z|^k` with a safety factor of 8, carried
        // through the same division as `w_i`.
        .mul(Scaled::from_value(C::Real::EPSILON)?)?
        .mul(Scaled::from_value(
            (n + C::Real::ONE) * C::Real::try_from_usize_exact(8)?,
        )?)?
        .div(denominator)?;
    let radius = correction
        .magnitude()?
        .mul(Scaled::from_value(n - C::Real::ONE)?)?
        .add(floor.mul(Scaled::from_value(n)?)?)?;
    let maximum = Scaled::from_value(C::Real::MAX)?;
    if !center.real().abs_le(maximum) || !center.imaginary().abs_le(maximum) {
        return Ok(Err(RootEstimateFailure::ScalarRange));
    }
    match (center.round(), estimate(radius)?) {
        (Ok(center), RootEstimate::Finite(radius)) => Ok(Ok(RootDiskEstimate { center, radius })),
        (Err(error), _)
            if matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability { .. }
            ) =>
        {
            Ok(Err(RootEstimateFailure::ScalarRange))
        }
        (Ok(_), _) => Ok(Err(RootEstimateFailure::ScalarRange)),
        (Err(error), _) => Err(error),
    }
}
fn parent(
    parents: &mut [usize],
    mut index: usize,
    context: ExecutionContext<'_>,
) -> Result<usize, SciOxideError> {
    // Union-find lookup with path halving and cancellation polling.
    while parents[index] != index {
        context.check()?;
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    Ok(index)
}
fn analyze<C: NumericScalar>(
    values: &[C],
    roots: &[Complex<C::Real>],
    output: &mut RootDiagnostics<C::Real>,
    workspace: &mut RootDiagnosticsWorkspace<C::Real>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let values = coefficients(values)?;
    let degree = values.len() - 1;
    if roots.len() != degree {
        return Err(SciOxideError::dimension_mismatch([degree], [roots.len()]));
    }
    for root in roots {
        validate_complex(*root)?;
    }
    // Execution must not allocate: every record is pushed into storage reserved
    // up front, so insufficient capacity is rejected before any work is done.
    output.fits(degree)?;
    workspace.fits(degree)?;
    let stage = &mut workspace.staged;
    context.check()?;
    let radii = polynomial_root_radii(values)?;
    stage.roots.clear();
    stage.clusters.clear();
    workspace.parents.clear();
    stage.radii = Some(radii);
    stage.status = RootDiagnosticsStatus::Estimated;
    workspace.parents.extend(0..degree);
    // `parents` first doubles as a canonical processing order (lexicographic by
    // candidate, ties by input index) so every diagnostic is independent of the
    // caller's candidate order; `group_disks` later reuses it as union-find
    // storage.
    workspace.parents.sort_unstable_by(|left, right| {
        crate::util::compare_complex(&roots[*left], &roots[*right]).then_with(|| left.cmp(right))
    });
    for &index in &workspace.parents {
        let candidate = roots[index];
        context.check()?;
        let evaluation = complex_horner::<_, 1, true>(values, candidate)?.into_scaled()?;
        let condition = condition_from_scaled(
            evaluation.bound,
            evaluation.value.magnitude()?,
            evaluation.derivative.magnitude()?,
            ScaledValue::from_value(candidate)?.magnitude()?,
        )?;
        let mut nearest: Option<Scaled<C::Real>> = None;
        for (other_index, other) in roots.iter().enumerate() {
            if other_index == index {
                continue;
            }
            context.check()?;
            let gap = distance(candidate, *other)?;
            nearest = Some(nearest.map_or(gap, |old| if gap.abs_le(old) { gap } else { old }));
        }
        let disk = match local_disk(
            values,
            roots,
            index,
            &evaluation,
            &workspace.parents,
            context,
        )? {
            Ok(disk) => Some(disk),
            Err(reason) => {
                stage.status = RootDiagnosticsStatus::Inconclusive(reason);
                None
            }
        };
        stage.roots.push(RootDiagnostic {
            index,
            candidate,
            condition,
            disk,
            separation: nearest.map_or(Ok(RootEstimate::Unavailable), estimate)?,
            cluster: 0,
        });
    }
    if stage.status == RootDiagnosticsStatus::Estimated {
        group_disks(stage, &mut workspace.parents, context)?;
    }
    // Partial local disks cannot support an inclusion claim, so any failure
    // falls back to the single global Cauchy disk for all candidates.
    if matches!(stage.status, RootDiagnosticsStatus::Inconclusive(_)) {
        global_fallback(stage, radii);
    }
    context.check()?;
    // Results are staged and published by swapping, so `output` is untouched on
    // error and its previous buffers become the next run's staging storage.
    core::mem::swap(output, stage);
    Ok(())
}

fn group_disks<S: RealScalar>(
    stage: &mut RootDiagnostics<S>,
    parents: &mut Vec<usize>,
    context: ExecutionContext<'_>,
) -> Result<(), SciOxideError> {
    let degree = stage.roots.len();
    parents.clear();
    parents.extend(0..degree);
    for i in 0..degree {
        for j in 0..i {
            context.check()?;
            if let (Some(left), Some(right)) = (stage.roots[i].disk, stage.roots[j].disk) {
                let sum = Scaled::from_value(left.radius.value())?
                    .add(Scaled::from_value(right.radius.value())?)?;
                if distance(left.center, right.center)?.abs_le(sum) {
                    let a = parent(parents, i, context)?;
                    let b = parent(parents, j, context)?;
                    // Always link to the smaller root so each component's representative is its
                    // smallest record index; the next loop relies on this to assign cluster ids.
                    parents[a.max(b)] = a.min(b);
                }
            }
        }
    }
    for i in 0..degree {
        context.check()?;
        let representative = parent(parents, i, context)?;
        // Because the representative is the component's smallest index, it is visited
        // first and has already been given its cluster id when later members arrive.
        let cluster = if representative == i {
            let index = stage.clusters.len();
            stage.clusters.push(RootCluster {
                disk: stage.roots[i].disk,
                count: 0,
                separation: RootEstimate::Unavailable,
            });
            index
        } else {
            stage.roots[representative].cluster
        };
        stage.roots[i].cluster = cluster;
        let group = &mut stage.clusters[cluster];
        group.count += 1;
        // Grow the representative's disk (fixed center) until it contains each member
        // disk. Not the minimal enclosing disk, but a valid enclosure of the union.
        if let (Some(enclosing), Some(disk)) = (group.disk, stage.roots[i].disk) {
            let radius = distance(enclosing.center, disk.center)?
                .add(Scaled::from_value(disk.radius.value())?)?;
            group.disk = match estimate(radius)? {
                RootEstimate::Finite(radius) => Some(RootDiskEstimate {
                    center: enclosing.center,
                    radius: if radius.value() > enclosing.radius.value() {
                        radius
                    } else {
                        enclosing.radius
                    },
                }),
                _ => {
                    stage.status =
                        RootDiagnosticsStatus::Inconclusive(RootEstimateFailure::ScalarRange);
                    None
                }
            };
        }
    }
    for i in 0..stage.clusters.len() {
        let mut gap: Option<Scaled<S>> = None;
        let mut missing = false;
        for j in 0..stage.clusters.len() {
            if i == j {
                continue;
            }
            context.check()?;
            if let (Some(left), Some(right)) = (stage.clusters[i].disk, stage.clusters[j].disk) {
                let sum = Scaled::from_value(left.radius.value())?
                    .add(Scaled::from_value(right.radius.value())?)?;
                let distance = distance(left.center, right.center)?;
                // Enlarged enclosures of distinct clusters may overlap; report zero
                // separation rather than a negative gap.
                let value = if distance.abs_le(sum) {
                    Scaled::ZERO
                } else {
                    distance.sub(sum)?
                };
                gap = Some(gap.map_or(value, |old| if value.abs_le(old) { value } else { old }));
            } else {
                missing = true;
            }
        }
        stage.clusters[i].separation = if missing {
            RootEstimate::Inconclusive(RootEstimateFailure::ScalarRange)
        } else {
            gap.map_or(Ok(RootEstimate::Unavailable), |gap| estimate(gap))?
        };
    }
    Ok(())
}

fn global_fallback<S: RealScalar>(stage: &mut RootDiagnostics<S>, radii: RootRadiusReport<S>) {
    for root in &mut stage.roots {
        root.disk = None;
        root.cluster = 0;
    }
    stage.clusters.clear();
    if !stage.roots.is_empty() {
        let disk = match radii.outer {
            RootEstimate::Finite(radius) => Some(RootDiskEstimate {
                center: Complex::new(S::ZERO, S::ZERO),
                radius,
            }),
            _ => None,
        };
        stage.clusters.push(RootCluster {
            disk,
            count: stage.roots.len(),
            separation: RootEstimate::Unavailable,
        });
    }
}
