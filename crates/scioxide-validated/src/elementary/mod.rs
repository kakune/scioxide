//! Enclosures of the ranges of the elementary functions on intervals.
mod approx;
mod binary;
mod constants;
mod kernel;
mod phase;
use crate::{DecoratedInterval as D, Decoration, rectangle::narrow};
use kernel::{B, bounds, range};
use scioxide_core::{RealScalar, SciOxideError};

/// Recognize exactly representable integer exponents supported by both evaluators.
pub(crate) fn integer_exponent(value: f64) -> Option<i32> {
    (value.is_finite()
        && value >= f64::from(i32::MIN)
        && value <= f64::from(i32::MAX)
        && value == RealScalar::trunc(value))
    .then_some(value as i32)
}

/// Says where the arguments of an elementary function lie relative to its
/// real domain.
///
/// The status does not depend on the width of the enclosure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnclosureStatus {
    /// An argument is the empty set, so there is nothing to evaluate.
    EmptyInput,
    /// Every argument lies in the domain.
    Defined,
    /// Some, but not all, arguments lie in the domain; the result encloses the
    /// values at those that do.
    PartiallyDefined,
    /// No argument lies in the domain; the result is empty.
    Undefined,
    /// The enclosure is valid, but the fixed precision of the argument
    /// reduction cannot decide where the arguments lie, for example whether an
    /// interval contains a pole of `tan`.
    Inconclusive,
}

/// Enclosure of the range of an elementary function, as one or two intervals.
///
/// The methods `exp`, `ln`, `sin`, `tan`, `atan2`, `powf` and the other
/// elementary functions of [`DecoratedInterval`](crate::DecoratedInterval)
/// return this type. The result is the union of at most two sorted intervals
/// that do not overlap, the pieces. It contains the value of the function at
/// every argument in its domain, and [`Self::status`] tells whether some
/// arguments lie outside the domain. A range that splits into more than two
/// parts is enclosed with gaps filled, which [`Self::gaps_filled`] reports.
///
/// Each endpoint comes from a fixed-degree polynomial approximation after
/// argument reduction. The approximation and rounding errors are bounded and
/// added before the final outward rounding. For a point argument the enclosure
/// is at most two floating-point steps wide; see the
/// [crate accuracy table](crate#accuracy). Every method takes constant time and
/// does not allocate.
///
/// ```
/// use scioxide_validated::{DecoratedInterval, EnclosureStatus};
///
/// let x = DecoratedInterval::new(1.0_f64, 2.0)?;
/// // tan has a pole at π/2 ≈ 1.571: two pieces, neither containing zero.
/// let tangent = x.tan()?;
/// assert_eq!(tangent.status(), EnclosureStatus::PartiallyDefined);
/// assert_eq!(tangent.pieces().len(), 2);
/// assert!(tangent.contains(2.0_f64.tan()) && !tangent.contains(0.0));
/// assert!(tangent.hull().contains(0.0));
/// // ln is defined only on the positive part of [−1, 1].
/// let partial = DecoratedInterval::new(-1.0_f64, 1.0)?.ln()?;
/// assert_eq!(partial.status(), EnclosureStatus::PartiallyDefined);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ElementaryEnclosure<S: RealScalar> {
    pieces: [D<S>; 2],
    len: usize,
    status: EnclosureStatus,
    gaps_filled: bool,
}
impl<S: RealScalar> ElementaryEnclosure<S> {
    /// Returns the pieces in increasing order: none for an empty result, and
    /// at most two.
    pub fn pieces(&self) -> &[D<S>] {
        &self.pieces[..self.len]
    }
    /// Returns where the arguments lie relative to the domain of the function.
    pub fn status(&self) -> EnclosureStatus {
        self.status
    }
    /// Returns whether the pieces may cover gaps in the exact range, because
    /// the range has more parts than the two pieces can represent. `false`
    /// does not mean that the pieces are tight.
    pub fn gaps_filled(&self) -> bool {
        self.gaps_filled
    }
    /// Returns the smallest interval that contains every piece. The hull of two
    /// pieces is decorated `Trv`.
    pub fn hull(&self) -> D<S> {
        match self.len {
            0 => D::empty(),
            1 => self.pieces[0],
            _ => self.pieces[0].hull(&self.pieces[1]),
        }
    }
    /// Returns whether `value` lies in one of the pieces; a value in the gap
    /// between two pieces is not contained.
    pub fn contains(&self, value: S) -> bool {
        self.pieces().iter().any(|piece| piece.contains(value))
    }
    fn empty(status: EnclosureStatus) -> Self {
        Self {
            pieces: [D::empty(); 2],
            len: 0,
            status,
            gaps_filled: false,
        }
    }
    fn new(
        mut ranges: [(f64, f64); 2],
        mut len: usize,
        status: EnclosureStatus,
        decoration: Decoration,
        gaps_filled: bool,
    ) -> Result<Self, SciOxideError> {
        if len == 0 {
            return Ok(Self::empty(status));
        }
        if len == 2 && ranges[0].0 > ranges[1].0 {
            ranges.swap(0, 1);
        }
        if len == 2 && ranges[0].1 >= ranges[1].0 {
            ranges[0].1 = ranges[0].1.max(ranges[1].1);
            len = 1;
        }
        let decoration = if matches!(status, EnclosureStatus::Defined) {
            decoration
        } else {
            Decoration::Trv
        };
        let mut result = Self {
            pieces: [D::empty(); 2],
            len,
            status,
            gaps_filled,
        };
        for (piece, &(lo, hi)) in result.pieces[..len].iter_mut().zip(&ranges[..len]) {
            *piece = D::result(narrow::<S>(lo, false)?, narrow::<S>(hi, true)?, decoration)?;
        }
        // Destination rounding can close a gap that exists in binary64.
        if len == 2 {
            let left = result.pieces[0].bounds();
            let right = result.pieces[1].bounds();
            if let Some(((a, b), (c, d))) = left.zip(right)
                && b >= c
            {
                result.pieces[0] = D::result(a, b.max(d), decoration)?;
                result.len = 1;
            }
        }
        Ok(result)
    }
    /// Construct one outward-rounded piece with the given status and decoration.
    fn one(
        value: B,
        status: EnclosureStatus,
        decoration: Decoration,
    ) -> Result<Self, SciOxideError> {
        Self::new([bounds(value)?, (0.0, 0.0)], 1, status, decoration, false)
    }
}
#[derive(Clone, Copy)]
enum Domain {
    Entire,
    Positive,
    AboveMinusOne,
    Unit,
    HyperbolicUnit,
    AtLeastOne,
}
impl Domain {
    fn clip(self, a: f64, b: f64) -> Option<(f64, f64, bool)> {
        let (lo, hi, open) = match self {
            Self::Entire => (f64::NEG_INFINITY, f64::INFINITY, false),
            Self::Positive => (0.0, f64::INFINITY, true),
            Self::AboveMinusOne => (-1.0, f64::INFINITY, true),
            Self::Unit => (-1.0, 1.0, false),
            Self::HyperbolicUnit => (-1.0, 1.0, true),
            Self::AtLeastOne => (1.0, f64::INFINITY, false),
        };
        if b < lo || a > hi || (open && (b == lo || a == hi)) {
            return None;
        }
        // Infinite interval endpoints are limits, never excluded members.
        let clipped = a < lo
            || b > hi
            || (open && ((a == lo && lo.is_finite()) || (b == hi && hi.is_finite())));
        Some((if a > lo { a } else { lo }, b.min(hi), clipped))
    }
}
macro_rules! monotone_methods {
    ($(($name:ident,$admitted:ident,$kernel:ident,$domain:ident,$decreasing:expr,$doc:literal)),* $(,)?) => {$ (
        #[doc=$doc]
        ///
        /// Arguments outside the domain are excluded from the result and reported by
        /// [`ElementaryEnclosure::status`].
        ///
        /// # Errors
        ///
        /// - `InvalidInput` if the [floating-point environment](crate#floating-point-environment)
        ///   is not supported.
        /// - `NumericalInstability` if an internal enclosure step breaks down.
        pub fn $name(&self) -> Result<ElementaryEnclosure<S>,SciOxideError> {
            crate::environment::check()?;
            self.$admitted()
        }
        pub(crate) fn $admitted(&self) -> Result<ElementaryEnclosure<S>,SciOxideError> {
            self.monotone(
                kernel::$kernel,
                Domain::$domain,
                $decreasing,

            )
        }
    )*};
}
impl<S: RealScalar> D<S> {
    fn wide_bounds(&self) -> Option<(f64, f64)> {
        self.bounds().map(|(a, b)| (a.into_f64(), b.into_f64()))
    }
    /// Monotone kernel range.
    #[inline]
    fn monotone(
        &self,
        f: fn(f64) -> Result<B, SciOxideError>,
        domain: Domain,
        decreasing: bool,
    ) -> Result<ElementaryEnclosure<S>, SciOxideError> {
        let Some((a, b)) = self.wide_bounds() else {
            return Ok(ElementaryEnclosure::empty(EnclosureStatus::EmptyInput));
        };
        let Some((a, b, clipped)) = domain.clip(a, b) else {
            return Ok(ElementaryEnclosure::empty(EnclosureStatus::Undefined));
        };
        let left = bounds(f(a)?)?;
        let right = if a == b { left } else { bounds(f(b)?)? };
        let (lo, hi) = if decreasing {
            (right.0, left.1)
        } else {
            (left.0, right.1)
        };
        ElementaryEnclosure::one(
            range(lo, hi)?,
            if clipped {
                EnclosureStatus::PartiallyDefined
            } else {
                EnclosureStatus::Defined
            },
            self.decoration(),
        )
    }
    monotone_methods!(
        (
            exp,
            exp_admitted,
            exp,
            Entire,
            false,
            "Encloses `{eˣ : x ∈ self}`."
        ),
        (
            exp2,
            exp2_admitted,
            exp2,
            Entire,
            false,
            "Encloses `{2ˣ : x ∈ self}`."
        ),
        (
            exp10,
            exp10_admitted,
            exp10,
            Entire,
            false,
            "Encloses `{10ˣ : x ∈ self}`."
        ),
        (
            exp_m1,
            exp_m1_admitted,
            expm1,
            Entire,
            false,
            "Encloses `{eˣ − 1 : x ∈ self}`, without cancellation for `x` near zero."
        ),
        (
            ln,
            ln_admitted,
            ln,
            Positive,
            false,
            "Encloses `{ln x : x ∈ self, x > 0}`."
        ),
        (
            log2,
            log2_admitted,
            log2,
            Positive,
            false,
            "Encloses `{log₂ x : x ∈ self, x > 0}`."
        ),
        (
            log10,
            log10_admitted,
            log10,
            Positive,
            false,
            "Encloses `{log₁₀ x : x ∈ self, x > 0}`."
        ),
        (
            ln_1p,
            ln_1p_admitted,
            log1p,
            AboveMinusOne,
            false,
            "Encloses `{ln(1 + x) : x ∈ self, x > −1}`, without cancellation for `x` near zero."
        ),
        (
            asin,
            asin_admitted,
            asin,
            Unit,
            false,
            "Encloses `{arcsin x : x ∈ self, −1 ≤ x ≤ 1}`, with values in `[−π/2, π/2]`."
        ),
        (
            acos,
            acos_admitted,
            acos,
            Unit,
            true,
            "Encloses `{arccos x : x ∈ self, −1 ≤ x ≤ 1}`, with values in `[0, π]`."
        ),
        (
            atan,
            atan_admitted,
            atan,
            Entire,
            false,
            "Encloses `{arctan x : x ∈ self}`, with values in `(−π/2, π/2)`."
        ),
        (
            sinh,
            sinh_admitted,
            sinh,
            Entire,
            false,
            "Encloses `{sinh x : x ∈ self}`."
        ),
        (
            tanh,
            tanh_admitted,
            tanh,
            Entire,
            false,
            "Encloses `{tanh x : x ∈ self}`."
        ),
        (
            asinh,
            asinh_admitted,
            asinh,
            Entire,
            false,
            "Encloses `{arsinh x : x ∈ self}`."
        ),
        (
            acosh,
            acosh_admitted,
            acosh,
            AtLeastOne,
            false,
            "Encloses `{arcosh x : x ∈ self, x ≥ 1}`, with values in `[0, +∞)`."
        ),
        (
            atanh,
            atanh_admitted,
            atanh,
            HyperbolicUnit,
            false,
            "Encloses `{artanh x : x ∈ self, −1 < x < 1}`."
        ),
        (
            cbrt,
            cbrt_admitted,
            cbrt,
            Entire,
            false,
            "Encloses `{∛x : x ∈ self}`, the real cube root, which is negative for `x < 0`."
        ),
    );
    /// Encloses `{cosh x : x ∈ self}`, including the minimum `cosh 0 = 1` when
    /// `self` contains zero.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the [floating-point environment](crate#floating-point-environment)
    ///   is not supported.
    /// - `NumericalInstability` if an internal enclosure step breaks down.
    pub fn cosh(&self) -> Result<ElementaryEnclosure<S>, SciOxideError> {
        crate::environment::check()?;
        self.cosh_admitted()
    }

    /// Enclosure kernel; the public caller checks the floating-point environment.
    pub(crate) fn cosh_admitted(&self) -> Result<ElementaryEnclosure<S>, SciOxideError> {
        self.abs_admitted()?
            .monotone(kernel::cosh, Domain::Entire, false)
    }
    /// Encloses `{sin x : x ∈ self}`, including the extrema `±1` at the points
    /// `π/2 + kπ` that `self` contains.
    ///
    /// The argument is reduced modulo `π/2` by Payne–Hanek reduction.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the [floating-point environment](crate#floating-point-environment)
    ///   is not supported.
    /// - `NumericalInstability` if an internal enclosure step breaks down.
    pub fn sin(&self) -> Result<ElementaryEnclosure<S>, SciOxideError> {
        crate::environment::check()?;
        self.sin_admitted()
    }

    /// Enclosure kernel; the public caller checks the floating-point environment.
    pub(crate) fn sin_admitted(&self) -> Result<ElementaryEnclosure<S>, SciOxideError> {
        self.trig(false)
    }
    /// Encloses `{cos x : x ∈ self}`, including the extrema `±1` at the points
    /// `kπ` that `self` contains. The argument is reduced as for [`Self::sin`].
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the [floating-point environment](crate#floating-point-environment)
    ///   is not supported.
    /// - `NumericalInstability` if an internal enclosure step breaks down.
    pub fn cos(&self) -> Result<ElementaryEnclosure<S>, SciOxideError> {
        crate::environment::check()?;
        self.cos_admitted()
    }

    /// Enclosure kernel; the public caller checks the floating-point environment.
    pub(crate) fn cos_admitted(&self) -> Result<ElementaryEnclosure<S>, SciOxideError> {
        self.trig(true)
    }
    /// Sine or cosine range.
    fn trig(&self, cosine: bool) -> Result<ElementaryEnclosure<S>, SciOxideError> {
        let Some((a, b)) = self.wide_bounds() else {
            return Ok(ElementaryEnclosure::empty(EnclosureStatus::EmptyInput));
        };
        if !a.is_finite() || !b.is_finite() || b - a >= 6.0 {
            return ElementaryEnclosure::one(
                range(-1.0, 1.0)?,
                EnclosureStatus::Defined,
                self.decoration(),
            );
        }
        let left = bounds(phase::value(a, cosine)?)?;
        let right = if a == b {
            left
        } else {
            bounds(phase::value(b, cosine)?)?
        };
        let mut lo = left.0.min(right.0);
        let mut hi = left.1.max(right.1);
        // Inside the central half-turn the sine is monotone and cosine
        // has only its exact maximum at zero. This also handles point inputs.
        if a >= -core::f64::consts::FRAC_PI_2 && b <= core::f64::consts::FRAC_PI_2 {
            if cosine && a <= 0.0 && b >= 0.0 {
                hi = 1.0;
            }
            return ElementaryEnclosure::one(
                range(lo, hi)?,
                EnclosureStatus::Defined,
                self.decoration(),
            );
        }
        let crossings = phase::crossings(a, b)?;
        let maximum = if cosine { 0 } else { 1 };
        if crossings[maximum].0 {
            hi = 1.0;
        }
        if crossings[maximum + 2].0 {
            lo = -1.0;
        }
        ElementaryEnclosure::one(range(lo, hi)?, EnclosureStatus::Defined, self.decoration())
    }
    /// Encloses `{tan x : x ∈ self, cos x ≠ 0}`.
    ///
    /// If `self` contains one pole `π/2 + kπ`, the result has two pieces, one
    /// for each side of the pole, and the status is `PartiallyDefined`. If it
    /// contains more than one pole, the result is the whole real line. If the
    /// argument reduction cannot decide whether `self` contains a pole, the
    /// result is the whole real line with status `Inconclusive`.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if the [floating-point environment](crate#floating-point-environment)
    ///   is not supported.
    /// - `NumericalInstability` if an internal enclosure step breaks down.
    pub fn tan(&self) -> Result<ElementaryEnclosure<S>, SciOxideError> {
        crate::environment::check()?;
        self.tan_admitted()
    }

    /// Enclosure kernel; the public caller checks the floating-point environment.
    pub(crate) fn tan_admitted(&self) -> Result<ElementaryEnclosure<S>, SciOxideError> {
        let Some((a, b)) = self.wide_bounds() else {
            return Ok(ElementaryEnclosure::empty(EnclosureStatus::EmptyInput));
        };
        if !a.is_finite() || !b.is_finite() || b - a >= 4.0 {
            return ElementaryEnclosure::one(
                range(f64::NEG_INFINITY, f64::INFINITY)?,
                EnclosureStatus::PartiallyDefined,
                self.decoration(),
            );
        }
        let crossings = phase::crossings(a, b)?;
        if crossings[1].0 && crossings[3].0 {
            // Two poles include a complete intervening tangent branch. The
            // endpoint branches alone can leave a spurious gap in that image.
            return ElementaryEnclosure::one(
                range(f64::NEG_INFINITY, f64::INFINITY)?,
                if crossings[1].1 && crossings[3].1 {
                    EnclosureStatus::PartiallyDefined
                } else {
                    EnclosureStatus::Inconclusive
                },
                self.decoration(),
            );
        }
        let possible = crossings[1].0 || crossings[3].0;
        let certain = crossings[1].1 || crossings[3].1;
        let eval = phase::tangent;
        let left = bounds(eval(a)?)?;
        let right = bounds(eval(b)?)?;
        if possible && !certain || !left.0.is_finite() || !right.1.is_finite() {
            return ElementaryEnclosure::one(
                range(f64::NEG_INFINITY, f64::INFINITY)?,
                EnclosureStatus::Inconclusive,
                self.decoration(),
            );
        }
        if possible {
            ElementaryEnclosure::new(
                [(f64::NEG_INFINITY, right.1), (left.0, f64::INFINITY)],
                2,
                EnclosureStatus::PartiallyDefined,
                self.decoration(),
                false,
            )
        } else {
            ElementaryEnclosure::one(
                range(left.0, right.1)?,
                EnclosureStatus::Defined,
                self.decoration(),
            )
        }
    }
}
