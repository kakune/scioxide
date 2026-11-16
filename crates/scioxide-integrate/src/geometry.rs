//! Stable interval geometry shared by deterministic and stochastic methods.
//!

use scioxide_core::{Interval, NumericalIssue, RealScalar, SciOxideError};

use scioxide_core::Scaled;

/// Half-open arithmetic proved when an interval is prepared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HalfOpenMapping {
    Unit,
    OriginScale,
    Native,
    Checked,
}

/// A non-degenerate interval whose span is stored with an extended exponent range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct IntervalGeometry<S: RealScalar> {
    lower: S,
    upper: S,
    span: Option<Scaled<S>>,
    native_span: S,
    native_half_open: bool,
    half_open: HalfOpenMapping,
}

impl<S: RealScalar> IntervalGeometry<S> {
    /// Map a half-open unit coordinate without ever returning the upper bound.
    #[inline(always)]
    pub(crate) fn half_open_point(self, unit: S) -> Result<S, SciOxideError> {
        if !unit.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Callback,
                "random provider returned a non-finite coordinate",
            ));
        }
        if unit < S::ZERO || unit >= S::ONE {
            return Err(SciOxideError::invalid_input(
                scioxide_core::argument!("random_provider"),
                scioxide_core::InvalidInputReason::OutOfRange,
                "random unit coordinate must lie in [0, 1)",
            ));
        }
        self.half_open_point_admitted(unit)
    }

    /// Map an already checked finite coordinate in `[0, 1)`.
    #[inline(always)]
    pub(crate) fn half_open_point_admitted(self, unit: S) -> Result<S, SciOxideError> {
        let clamp = |point| {
            if point >= self.upper {
                self.upper.next_after(<S as RealScalar>::NEG_INFINITY)
            } else {
                point
            }
        };
        Ok(match self.half_open {
            HalfOpenMapping::Unit => {
                if unit == S::ZERO {
                    self.lower
                } else {
                    unit
                }
            }
            HalfOpenMapping::OriginScale => {
                if unit == S::ZERO {
                    self.lower
                } else {
                    // With zero origin, this is one correctly rounded product even
                    // when the offset underflows. No subsequent addition loses it.
                    clamp(unit * self.native_span)
                }
            }
            HalfOpenMapping::Native => clamp(self.lower + unit * self.native_span),
            HalfOpenMapping::Checked => return self.checked_half_open_point(unit),
        })
    }

    /// Guarded native or scaled mapping for intervals without a cheaper proof.
    #[cold]
    #[inline(never)]
    fn checked_half_open_point(self, unit: S) -> Result<S, SciOxideError> {
        if unit == S::ZERO {
            return Ok(self.lower);
        }
        if self.native_half_open && unit != S::HALF {
            let offset = unit * self.native_span;
            if offset >= S::MIN_POSITIVE {
                // Preparation proved lower+span == upper and finite. IEEE
                // monotonicity then bounds every nonnegative rounded offset;
                // one predecessor clamp restores the half-open endpoint.
                let point = self.lower + offset;
                return Ok(if point >= self.upper {
                    self.upper.next_after(<S as RealScalar>::NEG_INFINITY)
                } else {
                    point
                });
            }
        }
        // `lower + unit * span` can round up to `upper` even though `unit < 1`; stepping to
        // the predecessor restores the half-open contract that stochastic samplers rely on
        // (e.g. so adjacent strata never share a boundary sample).
        let mapped = self.point_at_fraction(unit)?;
        let point = if mapped >= self.upper() {
            self.upper()
                .next_after(<_ as scioxide_core::RealScalar>::NEG_INFINITY)
        } else {
            mapped
        };
        if point < self.lower() || point >= self.upper() || !point.is_finite() {
            return Err(SciOxideError::numerical_instability(
                NumericalIssue::Breakdown,
                "interval has no valid half-open representable coordinate",
            ));
        }
        Ok(point)
    }

    /// Whether checked unit coordinates map to themselves after clearing zero's sign.
    pub(crate) fn is_positive_unit(self) -> bool {
        self.half_open == HalfOpenMapping::Unit && !self.lower.is_sign_negative()
    }
}

impl<S: RealScalar> IntervalGeometry<S> {
    #[inline(always)]
    pub(crate) fn native(interval: Interval<S>) -> Option<Self> {
        let native_span = interval.upper() - interval.lower();
        if native_span.is_finite() && native_span >= S::MIN_POSITIVE / S::EPSILON {
            let lower = interval.lower();
            let native_half_open = lower + native_span == interval.upper();
            Some(Self {
                lower: interval.lower(),
                upper: interval.upper(),
                span: None,
                native_span,
                native_half_open,
                half_open: HalfOpenMapping::Checked,
            })
        } else {
            None
        }
    }
    #[inline(always)]
    pub(crate) fn prepare(interval: Interval<S>) -> Result<Option<Self>, SciOxideError> {
        if interval.lower() == interval.upper() {
            return Ok(None);
        }
        if let Some(geometry) = Self::native(interval) {
            return Ok(Some(geometry));
        }
        Self::extended(interval).map(Some)
    }

    /// Check half-open mapping for Monte Carlo sampling once; deterministic
    /// preparation skips this check.
    #[inline]
    pub(crate) fn prepare_half_open(interval: Interval<S>) -> Result<Option<Self>, SciOxideError> {
        let Some(mut geometry) = Self::prepare(interval)? else {
            return Ok(None);
        };
        let lower = geometry.lower;
        geometry.half_open = if lower == S::ZERO {
            if geometry.upper == S::ONE {
                HalfOpenMapping::Unit
            } else {
                HalfOpenMapping::OriginScale
            }
        } else if geometry.native_half_open
            && lower.abs() >= (S::TWO * S::TWO) * (S::MIN_POSITIVE / S::EPSILON)
            && interval.midpoint().is_ok_and(|middle| {
                let mapped = lower + S::HALF * geometry.native_span;
                mapped == middle && mapped.is_sign_negative() == middle.is_sign_negative()
            })
        {
            // Native offsets are covered by monotonicity. Subnormal offsets
            // are below a quarter ulp at lower, so both native and extended
            // mapping round to lower. Midpoint agreement preserves core's
            // special half convention.
            HalfOpenMapping::Native
        } else {
            HalfOpenMapping::Checked
        };
        Ok(Some(geometry))
    }
    #[cold]
    #[inline(never)]
    fn extended(interval: Interval<S>) -> Result<Self, SciOxideError> {
        Ok(Self {
            lower: interval.lower(),
            upper: interval.upper(),
            span: Some(
                Scaled::from_value(interval.upper())?.sub(Scaled::from_value(interval.lower())?)?,
            ),
            native_span: interval.upper() - interval.lower(),
            native_half_open: false,
            half_open: HalfOpenMapping::Checked,
        })
    }

    pub(crate) fn lower(self) -> S {
        self.lower
    }

    pub(crate) fn upper(self) -> S {
        self.upper
    }

    pub(crate) fn span(self) -> Result<Scaled<S>, SciOxideError> {
        self.span
            .map_or_else(|| Scaled::from_value(self.native_span), Ok)
    }

    /// Normal finite span, absent when subnormal arithmetic needs extended precision.
    pub(crate) fn native_span(self) -> Option<S> {
        self.span.is_none().then_some(self.native_span)
    }

    pub(crate) fn half_span(self) -> Result<Scaled<S>, SciOxideError> {
        self.span()?.shifted(-1)
    }

    /// Map a fraction in `[0, 1]`, preserving the exact interval endpoints.
    #[inline(always)]
    pub(crate) fn point_at_fraction(self, fraction: S) -> Result<S, SciOxideError> {
        if !fraction.is_finite() || fraction < S::ZERO || fraction > S::ONE {
            return Err(SciOxideError::numerical_instability(
                NumericalIssue::Breakdown,
                "interval fraction must be finite and lie in [0, 1]",
            ));
        }
        // Endpoints and the midpoint are returned exactly rather than through the rounded
        // affine formula: rules with endpoint nodes must hit the user's bounds bit-for-bit,
        // and bisection must produce children that share exactly the same midpoint.
        if fraction == S::ZERO {
            return Ok(self.lower());
        }
        if fraction == S::ONE {
            return Ok(self.upper());
        }
        if fraction == S::HALF {
            return scioxide_core::checked_midpoint(self.lower, self.upper);
        }
        if self.span.is_none() {
            let offset = fraction * self.native_span;
            let point = self.lower + offset;
            if offset.abs() >= S::MIN_POSITIVE
                && point.is_finite()
                && point >= self.lower
                && point <= self.upper
            {
                return Ok(point);
            }
        }
        self.extended_point(fraction)
    }

    #[cold]
    #[inline(never)]
    fn extended_point(self, fraction: S) -> Result<S, SciOxideError> {
        Scaled::from_value(self.lower())?
            .add(self.span()?.mul(Scaled::from_value(fraction)?)?)?
            .round()
    }

    /// Map a reference coordinate in `[-1, 1]` without forming the scalar span.
    #[inline]
    pub(crate) fn canonical_point(self, canonical: S) -> Result<S, SciOxideError> {
        if canonical == -S::ONE {
            return Ok(self.lower());
        }
        if canonical == S::ONE {
            return Ok(self.upper());
        }
        // `x = lower + (t + 1)/2 * span`; routing through the unit fraction avoids forming
        // the midpoint and half-span separately, which would overflow for extreme bounds.
        if self.span.is_none() && canonical.is_finite() && canonical.abs() <= S::ONE {
            return self.point_at_fraction((canonical + S::ONE) * S::HALF);
        }
        self.extended_canonical_point(canonical)
    }

    #[cold]
    #[inline(never)]
    fn extended_canonical_point(self, canonical: S) -> Result<S, SciOxideError> {
        let fraction = Scaled::from_value(canonical)?
            .add(Scaled::from_value(S::ONE)?)?
            .shifted(-1)?
            .round()?;
        self.point_at_fraction(fraction)
    }

    pub(crate) fn scaled_half_weight(self, weight: S) -> Result<Scaled<S>, SciOxideError> {
        self.half_span()?.mul(Scaled::from_value(weight)?)
    }
}

/// Lebesgue measure of a box, kept in extended range so extreme spans cannot overflow.
pub(crate) fn checked_volume<S: RealScalar>(
    mut geometries: impl Iterator<Item = Result<IntervalGeometry<S>, SciOxideError>> + Clone,
) -> Result<Scaled<S>, SciOxideError> {
    let mut native = Some(S::ONE);
    for geometry in geometries.clone() {
        let geometry = geometry?;
        native = native.and_then(|volume| {
            let product = volume * geometry.native_span()?;
            (product.is_finite() && product >= S::MIN_POSITIVE).then_some(product)
        });
        if native.is_none() {
            break;
        }
    }
    if let Some(volume) = native {
        return Scaled::from_value(volume);
    }
    geometries.try_fold(Scaled::from_value(S::ONE)?, |volume, geometry| {
        volume.mul(geometry?.span()?)
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn opposite_sign_maximum_bounds_have_stable_geometry() {
        let interval = Interval::new(-f64::MAX, f64::MAX).unwrap();
        let geometry = IntervalGeometry::prepare(interval).unwrap().unwrap();
        assert_eq!(geometry.canonical_point(-1.0).unwrap(), -f64::MAX);
        assert_eq!(geometry.canonical_point(0.0).unwrap(), 0.0);
        assert_eq!(geometry.canonical_point(1.0).unwrap(), f64::MAX);
    }

    #[test]
    fn native_midpoints_match_core_at_rounding_boundaries() {
        for (lower, upper) in [
            (1.0, 1.0_f64.next_up()),
            (1.0_f64.next_up(), 2.0),
            (-2.0, -1.0_f64.next_up()),
            (-1.0, 1.0_f64.next_up()),
            (f64::from_bits(1), 1.0),
            (0.0, f64::MIN_POSITIVE),
        ] {
            let interval = Interval::new(lower, upper).unwrap();
            let geometry = IntervalGeometry::prepare_half_open(interval)
                .unwrap()
                .unwrap();
            assert_eq!(
                geometry.point_at_fraction(0.5).unwrap(),
                interval.midpoint().unwrap()
            );
            let expected = interval.midpoint().unwrap().min(upper.next_down());
            assert_eq!(
                geometry.half_open_point(0.5).unwrap().to_bits(),
                expected.to_bits()
            );
        }
    }

    #[test]
    fn half_open_mapping_keeps_subnormal_offsets_and_spacing_boundaries() {
        fn check<S: RealScalar>() {
            let tiny = S::ZERO.next_after(S::ONE);
            let last = S::ONE.next_after(S::ZERO);
            let large = S::TWO / S::EPSILON;
            let three_quarters = S::ONE - S::HALF * S::HALF;
            for (lower, upper, expected) in [
                (S::ZERO, tiny, [S::ZERO; 4]),
                (-S::ZERO, tiny, [-S::ZERO, S::ZERO, S::ZERO, S::ZERO]),
                (
                    S::ZERO,
                    S::TWO,
                    [S::ZERO, tiny + tiny, S::ONE, S::TWO - S::EPSILON],
                ),
                (S::HALF, S::ONE, [S::HALF, S::HALF, three_quarters, last]),
                (
                    -S::ONE,
                    S::ONE,
                    [-S::ONE, -S::ONE, S::ZERO, S::ONE - S::EPSILON],
                ),
                (
                    large,
                    large + S::TWO * S::TWO,
                    [large, large, large + S::TWO, large + S::TWO],
                ),
                (tiny, S::ONE, [tiny, tiny + tiny, S::HALF, last]),
            ] {
                let geometry =
                    IntervalGeometry::prepare_half_open(Interval::new(lower, upper).unwrap())
                        .unwrap()
                        .unwrap();
                for (unit, expected) in [S::ZERO, tiny, S::HALF, last].into_iter().zip(expected) {
                    let actual = geometry.half_open_point(unit).unwrap();
                    assert_eq!(actual.into_f64().to_bits(), expected.into_f64().to_bits());
                }
            }
            let threshold = (S::TWO * S::TWO) * (S::MIN_POSITIVE / S::EPSILON);
            for lower in [threshold, threshold.next_after(S::ONE), -threshold] {
                let geometry =
                    IntervalGeometry::prepare_half_open(Interval::new(lower, S::ONE).unwrap())
                        .unwrap()
                        .unwrap();
                assert_eq!(
                    geometry
                        .half_open_point(S::MIN_POSITIVE.next_after(S::ZERO))
                        .unwrap(),
                    lower
                );
            }
        }
        check::<f32>();
        check::<f64>();
    }
}
