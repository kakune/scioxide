//! Closed real intervals, bounded or unbounded.
//!
//! [`Interval`] is a closed interval `[lower, upper]` with finite ends.
//! [`ExtendedInterval`] also represents the empty set and half-lines, and
//! provides hull, intersection and containment.

use crate::arithmetic::{checked_difference, checked_midpoint};
use crate::error::{InvalidInputReason, NonFiniteOrigin, SciOxideError};
use crate::scalar::RealScalar;

/// A closed interval `[lower, upper]` with finite ends.
///
/// Both ends are included, finite, and satisfy `lower ≤ upper`. A single point
/// (`lower == upper`) is a valid interval. Equality compares both ends exactly.
/// Tuples `(lower, upper)` and arrays `[lower, upper]` convert with [`TryFrom`],
/// which checks these conditions, and convert back with [`From`].
///
/// # Examples
///
/// ```
/// use scioxide_core::Interval;
///
/// let iv = Interval::<f64>::new(0.0, 1.0).unwrap();
/// assert!(iv.contains(0.5));
/// assert_eq!(iv.width().unwrap(), 1.0);
/// assert_eq!((iv.lower(), iv.upper()), (0.0, 1.0));
///
/// let widened = iv.with_upper(2.0).unwrap();
/// assert_eq!(widened, Interval::try_from([0.0, 2.0]).unwrap());
/// assert_eq!(<(f64, f64)>::from(widened), (0.0, 2.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Interval<S: RealScalar> {
    lower: S,
    upper: S,
}

impl<S: RealScalar> Interval<S> {
    /// Creates the closed interval `[lower, upper]`; both ends are included.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if either end is NaN or infinite.
    /// - `InvalidInput` (`Inconsistent`) if `lower > upper`.
    pub fn new(lower: S, upper: S) -> Result<Self, SciOxideError> {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "interval bounds must be finite",
            ));
        }
        if lower > upper {
            return Err(SciOxideError::invalid_input(
                crate::argument!("lower"),
                InvalidInputReason::Inconsistent,
                "interval lower bound must not exceed upper bound",
            ));
        }
        Ok(Self { lower, upper })
    }

    /// Returns the lower end.
    #[must_use]
    pub fn lower(&self) -> S {
        self.lower
    }

    /// Returns the upper end.
    #[must_use]
    pub fn upper(&self) -> S {
        self.upper
    }

    /// Returns a copy with a different lower end.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `lower` is NaN or infinite.
    /// - `InvalidInput` if `lower` exceeds the current upper end.
    pub fn with_lower(self, lower: S) -> Result<Self, SciOxideError> {
        Self::new(lower, self.upper)
    }

    /// Returns a copy with a different upper end.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `upper` is NaN or infinite.
    /// - `InvalidInput` if `upper` is less than the current lower end.
    pub fn with_upper(self, upper: S) -> Result<Self, SciOxideError> {
        Self::new(self.lower, upper)
    }

    /// Returns whether `lower ≤ x ≤ upper`; NaN and infinities return `false`.
    #[must_use]
    pub fn contains(&self, x: S) -> bool {
        x.is_finite() && x >= self.lower && x <= self.upper
    }

    /// Returns the width `upper − lower`, correctly rounded.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` ([`Overflow`](crate::NumericalIssue::Overflow))
    ///   if the width exceeds the largest finite value, for example for
    ///   `[−f64::MAX, f64::MAX]`.
    pub fn width(&self) -> Result<S, SciOxideError> {
        checked_difference(self.upper, self.lower)
    }

    /// Returns the midpoint `(lower + upper) / 2`, correctly rounded.
    ///
    /// The midpoint is computed without intermediate overflow, so
    /// `[−f64::MAX, f64::MAX]` has midpoint `0`. See [`crate::checked_midpoint`].
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` if the rounded midpoint is not finite.
    pub fn midpoint(&self) -> Result<S, SciOxideError>
    where
        S: RealScalar,
    {
        checked_midpoint(self.lower, self.upper)
    }

    /// Returns the point of the interval nearest to `x`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `x` is NaN or infinite.
    pub fn clamp(&self, x: S) -> Result<S, SciOxideError> {
        if !x.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "clamp query must be finite",
            ));
        }
        if x < self.lower {
            Ok(self.lower)
        } else if x > self.upper {
            Ok(self.upper)
        } else {
            Ok(x)
        }
    }
}

impl<S: RealScalar> TryFrom<(S, S)> for Interval<S> {
    type Error = SciOxideError;

    fn try_from((lower, upper): (S, S)) -> Result<Self, Self::Error> {
        Self::new(lower, upper)
    }
}

impl<S: RealScalar> TryFrom<[S; 2]> for Interval<S> {
    type Error = SciOxideError;

    fn try_from([lower, upper]: [S; 2]) -> Result<Self, Self::Error> {
        Self::new(lower, upper)
    }
}

impl<S: RealScalar> From<Interval<S>> for (S, S) {
    fn from(interval: Interval<S>) -> Self {
        (interval.lower, interval.upper)
    }
}

impl<S: RealScalar> From<Interval<S>> for [S; 2] {
    fn from(interval: Interval<S>) -> Self {
        [interval.lower, interval.upper]
    }
}

/// A closed interval of the real line that may be empty or unbounded.
///
/// The set is one of: empty, `[a, b]`, `(−∞, b]`, `[a, ∞)` or the whole line.
/// Finite ends are included. Infinity is never a member and is never stored;
/// an unbounded side is represented by `None`.
///
/// Construction and conversion to [`Interval`] keep the ends bit for bit.
/// Hull and intersection set a zero end to `−0` for a lower end and `+0` for an
/// upper end (the IEEE 1788 convention), so the result does not depend on
/// operand order. Equality compares ends exactly, with `−0 == +0`. The
/// operations do no arithmetic, so no rounding occurs.
///
/// ```
/// use scioxide_core::{ExtendedInterval, Interval};
/// let left = ExtendedInterval::new(None, Some(2.0_f64))?;
/// let right = ExtendedInterval::new(Some(1.0), None)?;
/// assert_eq!(Interval::try_from(left.intersection(&right))?, Interval::new(1.0, 2.0)?);
/// assert!(left.hull(&right).is_entire());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct ExtendedInterval<S: RealScalar> {
    set: IntervalSet<S>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum IntervalSet<S: RealScalar> {
    Empty,
    Finite(Interval<S>),
    LowerUnbounded(S),
    UpperUnbounded(S),
    Entire,
}

impl<S: RealScalar> ExtendedInterval<S> {
    /// Returns the empty set.
    pub fn empty() -> Self {
        Self {
            set: IntervalSet::Empty,
        }
    }

    /// Returns the whole real line.
    pub fn entire() -> Self {
        Self {
            set: IntervalSet::Entire,
        }
    }

    /// Creates a nonempty interval; `None` for an end means that side is unbounded.
    ///
    /// The signs of zero ends are kept.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if a supplied end is NaN or infinite. Use `None` for an
    ///   unbounded side.
    /// - `InvalidInput` (`Inconsistent`) if both ends are given and `lower > upper`.
    pub fn new(lower: Option<S>, upper: Option<S>) -> Result<Self, SciOxideError> {
        if lower.is_some_and(|x| !x.is_finite()) || upper.is_some_and(|x| !x.is_finite()) {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "present interval bounds must be finite",
            ));
        }
        let set = match (lower, upper) {
            (Some(lower), Some(upper)) => IntervalSet::Finite(Interval::new(lower, upper)?),
            (None, Some(upper)) => IntervalSet::LowerUnbounded(upper),
            (Some(lower), None) => IntervalSet::UpperUnbounded(lower),
            (None, None) => IntervalSet::Entire,
        };
        Ok(Self { set })
    }

    /// Returns whether this is the empty set.
    pub fn is_empty(&self) -> bool {
        matches!(self.set, IntervalSet::Empty)
    }

    /// Returns whether this is the whole real line.
    pub fn is_entire(&self) -> bool {
        matches!(self.set, IntervalSet::Entire)
    }

    /// Returns whether this is a nonempty bounded interval, which converts to
    /// [`Interval`]; `false` for the empty set.
    pub fn is_finite(&self) -> bool {
        matches!(self.set, IntervalSet::Finite(_))
    }

    /// Returns `(lower, upper)`, or `None` for the empty set.
    ///
    /// An inner `None` means that side is unbounded.
    pub fn bounds(&self) -> Option<(Option<S>, Option<S>)> {
        match self.set {
            IntervalSet::Empty => None,
            IntervalSet::Finite(interval) => Some((Some(interval.lower), Some(interval.upper))),
            IntervalSet::LowerUnbounded(upper) => Some((None, Some(upper))),
            IntervalSet::UpperUnbounded(lower) => Some((Some(lower), None)),
            IntervalSet::Entire => Some((None, None)),
        }
    }

    /// Returns whether `x` belongs to the set; NaN and infinities return `false`.
    pub fn contains(&self, x: S) -> bool {
        x.is_finite()
            && self.bounds().is_some_and(|(lower, upper)| {
                lower.is_none_or(|lower| lower <= x) && upper.is_none_or(|upper| x <= upper)
            })
    }

    /// Returns whether `other` is a subset of this set; the empty set is a subset
    /// of every set.
    pub fn contains_interval(&self, other: &Self) -> bool {
        match (self.bounds(), other.bounds()) {
            (_, None) => true,
            (None, Some(_)) => false,
            (Some((lower, upper)), Some((other_lower, other_upper))) => {
                lower.is_none_or(|lower| other_lower.is_some_and(|x| lower <= x))
                    && upper.is_none_or(|upper| other_upper.is_some_and(|x| x <= upper))
            }
        }
    }

    /// Returns the smallest interval that contains both sets (their convex hull,
    /// not their union).
    ///
    /// Zero ends become `−0` (lower) or `+0` (upper), also when one operand is
    /// empty.
    pub fn hull(&self, other: &Self) -> Self {
        if let (IntervalSet::Finite(a), IntervalSet::Finite(b)) = (self.set, other.set) {
            return a.hull(&b).into();
        }
        let bounds = match (self.bounds(), other.bounds()) {
            (None, bounds) | (bounds, None) => bounds,
            (Some((a, b)), Some((c, d))) => Some((
                a.zip(c).map(|(a, c)| a.min(c)),
                b.zip(d).map(|(b, d)| b.max(d)),
            )),
        };
        Self::computed(bounds)
    }

    /// Returns the intersection of both sets; disjoint sets give the empty set.
    ///
    /// Zero ends become `−0` (lower) or `+0` (upper).
    pub fn intersection(&self, other: &Self) -> Self {
        let bounds = self.bounds().zip(other.bounds()).map(|((a, b), (c, d))| {
            (
                match (a, c) {
                    (Some(a), Some(c)) => Some(a.max(c)),
                    _ => a.or(c),
                },
                match (b, d) {
                    (Some(b), Some(d)) => Some(b.min(d)),
                    _ => b.or(d),
                },
            )
        });
        Self::computed(bounds)
    }

    // Only called with finite endpoints selected from validated sets.
    fn computed(bounds: Option<(Option<S>, Option<S>)>) -> Self {
        let Some((lower, upper)) = bounds else {
            return Self::empty();
        };
        let lower = lower.map(lower_zero);
        let upper = upper.map(upper_zero);
        let set = match (lower, upper) {
            // Crossed endpoints can come only from an intersection of disjoint sets.
            (Some(lower), Some(upper)) if lower > upper => IntervalSet::Empty,
            (Some(lower), Some(upper)) => IntervalSet::Finite(Interval { lower, upper }),
            (None, Some(upper)) => IntervalSet::LowerUnbounded(upper),
            (Some(lower), None) => IntervalSet::UpperUnbounded(lower),
            (None, None) => IntervalSet::Entire,
        };
        Self { set }
    }
}

// `min` and `max` may return either signed zero, so computed endpoints are
// canonicalized to -0 for lower bounds and +0 for upper bounds (the
// IEEE 1788 convention). Hull and intersection then give deterministic bit
// patterns whatever the operand order and zero signs.
fn lower_zero<S: RealScalar>(x: S) -> S {
    if x.is_zero() { S::NEG_ZERO } else { x }
}
fn upper_zero<S: RealScalar>(x: S) -> S {
    if x.is_zero() { S::ZERO } else { x }
}

impl<S: RealScalar> From<Interval<S>> for ExtendedInterval<S> {
    fn from(interval: Interval<S>) -> Self {
        Self {
            set: IntervalSet::Finite(interval),
        }
    }
}

impl<S: RealScalar> TryFrom<ExtendedInterval<S>> for Interval<S> {
    type Error = SciOxideError;

    /// Converts a nonempty bounded set to an [`Interval`], keeping the ends bit for bit.
    ///
    /// # Errors
    ///
    /// - `InvalidInput` (`OutOfRange`) for the empty set or an unbounded set.
    fn try_from(interval: ExtendedInterval<S>) -> Result<Self, Self::Error> {
        match interval.set {
            IntervalSet::Finite(interval) => Ok(interval),
            _ => Err(SciOxideError::invalid_input(
                crate::argument!("interval"),
                InvalidInputReason::OutOfRange,
                "finite conversion requires a nonempty bounded interval",
            )),
        }
    }
}

impl<S: RealScalar> Interval<S> {
    /// Returns the smallest interval that contains both intervals.
    ///
    /// Zero ends become `−0` (lower) or `+0` (upper).
    pub fn hull(&self, other: &Self) -> Self {
        Self {
            lower: lower_zero(self.lower.min(other.lower)),
            upper: upper_zero(self.upper.max(other.upper)),
        }
    }

    /// Returns the intersection, which is empty for disjoint intervals.
    ///
    /// Zero ends become `−0` (lower) or `+0` (upper).
    pub fn intersection(&self, other: &Self) -> ExtendedInterval<S> {
        ExtendedInterval::from(*self).intersection(&(*other).into())
    }

    /// Returns whether `other` is a subset of this interval.
    pub fn contains_interval(&self, other: &Self) -> bool {
        ExtendedInterval::from(*self).contains_interval(&(*other).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Construction ----

    #[test]
    fn test_new_valid_f64() {
        let iv = Interval::<f64>::new(0.0, 1.0);
        assert!(iv.is_ok());
    }

    #[test]
    fn test_new_valid_f32() {
        let iv = Interval::<f32>::new(-1.0, 1.0);
        assert!(iv.is_ok());
    }

    #[test]
    fn test_new_equal_bounds() {
        let iv = Interval::<f64>::new(5.0, 5.0);
        assert!(iv.is_ok());
    }

    #[test]
    fn test_new_inverted_bounds() {
        let result = Interval::<f64>::new(2.0, 1.0);
        assert!(matches!(
            (result).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_new_nan_lower() {
        let result = Interval::<f64>::new(f64::NAN, 1.0);
        assert!(matches!(
            (result).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::NonFinite { .. }
        ));
    }

    #[test]
    fn test_new_nan_upper() {
        let result = Interval::<f64>::new(0.0, f64::NAN);
        assert!(matches!(
            (result).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::NonFinite { .. }
        ));
    }

    #[test]
    fn test_new_infinite_lower() {
        let result = Interval::<f64>::new(f64::NEG_INFINITY, 1.0);
        assert!(matches!(
            (result).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::NonFinite { .. }
        ));
    }

    #[test]
    fn test_new_infinite_upper() {
        let result = Interval::<f64>::new(0.0, f64::INFINITY);
        assert!(matches!(
            (result).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::NonFinite { .. }
        ));
    }

    // ---- Value-object API ----

    #[test]
    fn test_accessors() {
        let iv = Interval::<f64>::new(-2.0, 3.0).unwrap();
        assert_eq!(iv.lower(), -2.0);
        assert_eq!(iv.upper(), 3.0);
    }

    #[test]
    fn test_with_lower_validates_replacement() {
        let iv = Interval::<f64>::new(0.0, 2.0).unwrap();
        let updated = iv.with_lower(1.0).unwrap();

        assert_eq!(updated, Interval::new(1.0, 2.0).unwrap());
        assert_eq!(iv, Interval::new(0.0, 2.0).unwrap());
        assert!(matches!(
            (iv.with_lower(3.0)).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::InvalidInput { .. }
        ));
        assert!(matches!(
            (iv.with_lower(f64::NAN)).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::NonFinite { .. }
        ));
    }

    #[test]
    fn test_with_upper_validates_replacement() {
        let iv = Interval::<f64>::new(0.0, 2.0).unwrap();
        let updated = iv.with_upper(3.0).unwrap();

        assert_eq!(updated, Interval::new(0.0, 3.0).unwrap());
        assert_eq!(iv, Interval::new(0.0, 2.0).unwrap());
        assert!(matches!(
            (iv.with_upper(-1.0)).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::InvalidInput { .. }
        ));
        assert!(matches!(
            (iv.with_upper(f64::INFINITY)).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::NonFinite { .. }
        ));
    }

    #[test]
    fn test_interval_is_copy_clone_and_uses_value_equality() {
        fn assert_copy_clone<T: Copy + Clone>() {}

        assert_copy_clone::<Interval<f64>>();
        let iv = Interval::<f64>::new(-1.0, 1.0).unwrap();
        let copied = iv;

        assert_eq!(copied, iv);
        assert_ne!(iv, Interval::new(-1.0, 2.0).unwrap());
    }

    #[test]
    fn test_checked_tuple_conversion() {
        let iv = Interval::<f64>::try_from((-1.0, 2.0)).unwrap();
        let bounds: (f64, f64) = iv.into();

        assert_eq!(bounds, (-1.0, 2.0));
        assert!(matches!(
            (Interval::<f64>::try_from((2.0, 1.0)))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_checked_array_conversion() {
        let iv = Interval::<f32>::try_from([-1.0, 2.0]).unwrap();
        let bounds: [f32; 2] = iv.into();

        assert_eq!(bounds, [-1.0, 2.0]);
        assert!(matches!(
            (Interval::<f32>::try_from([f32::NEG_INFINITY, 2.0]))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::NonFinite { .. }
        ));
    }

    // ---- contains ----

    #[test]
    fn test_contains_inside() {
        let iv = Interval::<f64>::new(0.0, 1.0).unwrap();
        assert!(iv.contains(0.5));
    }

    #[test]
    fn test_contains_boundary() {
        let iv = Interval::<f64>::new(0.0, 1.0).unwrap();
        assert!(iv.contains(0.0));
        assert!(iv.contains(1.0));
    }

    #[test]
    fn test_contains_outside() {
        let iv = Interval::<f64>::new(0.0, 1.0).unwrap();
        assert!(!iv.contains(-0.1));
        assert!(!iv.contains(1.1));
        assert!(!iv.contains(f64::NAN));
        assert!(!iv.contains(f64::INFINITY));
        assert!(!iv.contains(f64::NEG_INFINITY));
    }

    // ---- width ----

    #[test]
    fn test_width() {
        let iv = Interval::<f64>::new(2.0, 5.0).unwrap();
        assert!((iv.width().unwrap() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_width_degenerate() {
        let iv = Interval::<f64>::new(3.0, 3.0).unwrap();
        assert!((iv.width().unwrap() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn width_reports_finite_range_overflow_for_both_precisions() {
        for result in [
            Interval::<f64>::new(-f64::MAX, f64::MAX)
                .unwrap()
                .width()
                .map(|_| ()),
            Interval::<f32>::new(-f32::MAX, f32::MAX)
                .unwrap()
                .width()
                .map(|_| ()),
        ] {
            assert!(matches!(
                (result).as_ref().unwrap_err().view(),
                crate::SciOxideErrorView::NumericalInstability {
                    issue: crate::NumericalIssue::Overflow,
                    ..
                }
            ));
        }
    }

    // ---- midpoint ----

    #[test]
    fn test_midpoint() {
        let iv = Interval::<f64>::new(2.0, 6.0).unwrap();
        assert!((iv.midpoint().unwrap() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_midpoint_large_bounds_stays_finite() {
        let iv = Interval::<f64>::new(1e308, 1.2e308).unwrap();
        let mid = iv.midpoint().unwrap();
        assert!(mid.is_finite());
        assert!(mid >= iv.lower() && mid <= iv.upper());
    }

    #[test]
    fn midpoint_handles_opposite_maximum_adjacent_and_subnormal_endpoints() {
        let extreme = Interval::<f64>::new(-f64::MAX, f64::MAX).unwrap();
        assert_eq!(extreme.midpoint().unwrap(), 0.0);
        let extreme_f32 = Interval::<f32>::new(-f32::MAX, f32::MAX).unwrap();
        assert_eq!(extreme_f32.midpoint().unwrap(), 0.0);

        let adjacent = Interval::<f64>::new(1.0, f64::from_bits(1.0_f64.to_bits() + 1)).unwrap();
        let midpoint = adjacent.midpoint().unwrap();
        assert!(adjacent.contains(midpoint));

        let minimum = f64::from_bits(1);
        let subnormal = Interval::<f64>::new(-minimum, minimum).unwrap();
        assert_eq!(subnormal.width().unwrap(), minimum + minimum);
        assert_eq!(subnormal.midpoint().unwrap(), 0.0);
    }

    // ---- clamp ----

    #[test]
    fn test_clamp_inside() {
        let iv = Interval::<f64>::new(0.0, 10.0).unwrap();
        assert!((iv.clamp(5.0).unwrap() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clamp_below() {
        let iv = Interval::<f64>::new(0.0, 10.0).unwrap();
        assert!((iv.clamp(-5.0).unwrap() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clamp_above() {
        let iv = Interval::<f64>::new(0.0, 10.0).unwrap();
        assert!((iv.clamp(15.0).unwrap() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clamp_boundary() {
        let iv = Interval::<f64>::new(0.0, 10.0).unwrap();
        assert!((iv.clamp(0.0).unwrap() - 0.0).abs() < f64::EPSILON);
        assert!((iv.clamp(10.0).unwrap() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn clamp_rejects_non_finite_queries_and_preserves_signed_zero() {
        let iv = Interval::<f64>::new(-0.0, 0.0).unwrap();
        for query in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                (iv.clamp(query)).as_ref().unwrap_err().view(),
                crate::SciOxideErrorView::NonFinite {
                    origin: NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
        assert!(iv.clamp(-0.0).unwrap().is_sign_negative());
        assert!(iv.clamp(0.0).unwrap().is_sign_positive());
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    macro_rules! boundaries {
        ($name:ident, $s:ty) => {
            #[test]
            fn $name() {
                type E = ExtendedInterval<$s>;
                let empty = E::empty();
                let entire = E::entire();
                assert_eq!(empty.bounds(), None);
                assert_eq!(entire.bounds(), Some((None, None)));
                assert_eq!(E::new(None, None).unwrap(), entire);
                assert!(empty.is_empty());
                assert!(!empty.is_finite());
                assert!(!empty.is_entire());
                assert!(entire.is_entire());
                assert!(!entire.is_finite());
                for invalid in [<$s>::NAN, <$s>::INFINITY, <$s>::NEG_INFINITY] {
                    for pair in [
                        (Some(invalid), None),
                        (None, Some(invalid)),
                        (Some(invalid), Some(0.0)),
                        (Some(0.0), Some(invalid)),
                    ] {
                        assert!(matches!(
                            (E::new(pair.0, pair.1)).as_ref().unwrap_err().view(),
                            crate::SciOxideErrorView::NonFinite {
                                origin: NonFiniteOrigin::Input,
                                ..
                            }
                        ));
                    }
                    assert!(!entire.contains(invalid));
                    assert!(!empty.contains(invalid));
                }
                assert!(matches!(
                    (E::new(Some(1.0), Some(-1.0))).as_ref().unwrap_err().view(),
                    crate::SciOxideErrorView::InvalidInput {
                        reason: InvalidInputReason::Inconsistent,
                        ..
                    }
                ));
                for set in [
                    empty,
                    entire,
                    E::new(None, Some(0.0)).unwrap(),
                    E::new(Some(0.0), None).unwrap(),
                ] {
                    assert!(!set.is_finite());
                    assert!(matches!(
                        (Interval::try_from(set)).as_ref().unwrap_err().view(),
                        crate::SciOxideErrorView::InvalidInput { .. }
                    ));
                }
                for lower in [-0.0, 0.0] {
                    for upper in [-0.0, 0.0] {
                        let original = Interval::<$s>::new(lower, upper).unwrap();
                        let set = E::new(Some(lower), Some(upper)).unwrap();
                        assert!(set.is_finite());
                        assert!(set.contains(-0.0) && set.contains(0.0));
                        assert_eq!(set, original.into());
                        let (actual_lower, actual_upper) = set.bounds().unwrap();
                        let (actual_lower, actual_upper) =
                            (actual_lower.unwrap(), actual_upper.unwrap());
                        assert_eq!(actual_lower.to_bits(), lower.to_bits());
                        assert_eq!(actual_upper.to_bits(), upper.to_bits());
                        let converted = Interval::try_from(E::from(original)).unwrap();
                        assert_eq!(converted.lower().to_bits(), lower.to_bits());
                        assert_eq!(converted.upper().to_bits(), upper.to_bits());
                        for computed in [
                            set.hull(&empty),
                            set.intersection(&entire),
                            set.hull(&set),
                            set.intersection(&set),
                            original.hull(&original).into(),
                            original.intersection(&original),
                        ] {
                            let interval = Interval::try_from(computed).unwrap();
                            assert!(interval.lower().is_sign_negative());
                            assert!(interval.upper().is_sign_positive());
                        }
                    }
                }
                let left = E::new(None, Some(-0.0)).unwrap();
                let right = E::new(Some(0.0), None).unwrap();
                assert!(left.contains(0.0) && right.contains(-0.0));
                assert_eq!(
                    left.intersection(&right),
                    Interval::new(-0.0, 0.0).unwrap().into()
                );
                assert!(left.hull(&right).is_entire());
                assert!(left.bounds().unwrap().1.unwrap().is_sign_negative());
                assert!(right.bounds().unwrap().0.unwrap().is_sign_positive());
                assert!(
                    left.hull(&empty)
                        .bounds()
                        .unwrap()
                        .1
                        .unwrap()
                        .is_sign_positive()
                );
                assert!(
                    right
                        .intersection(&entire)
                        .bounds()
                        .unwrap()
                        .0
                        .unwrap()
                        .is_sign_negative()
                );
                let tiny = <$s>::from_bits(1);
                for (lower, upper) in [(-<$s>::MAX, <$s>::MAX), (-tiny, tiny), (tiny, tiny)] {
                    let interval = Interval::new(lower, upper).unwrap();
                    let set = E::from(interval);
                    assert_eq!(Interval::try_from(set).unwrap(), interval);
                    assert!(set.contains(lower) && set.contains(upper));
                    assert_eq!(set.intersection(&entire), set);
                    assert_eq!(set.hull(&empty), set);
                }
                fn value_traits<T: Copy + Clone + Send + Sync>() {}
                value_traits::<E>();
            }
        };
    }
    boundaries!(boundaries_f32, f32);
    boundaries!(boundaries_f64, f64);
}
