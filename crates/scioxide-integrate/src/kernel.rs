//! Private integration primitives shared by the crate's numerical methods.
//!

use alloc::vec::Vec;

use scioxide_core::{Interval, NonFiniteOrigin, RealScalar, SciOxideError};

use crate::geometry::IntervalGeometry;
use scioxide_core::Scaled;

const CALLBACK_DETAIL: &str = "integrand produced a non-finite value";
const INTERMEDIATE_DETAIL: &str = "finite inputs produced a non-finite intermediate value";
/// Conservative multiplier for nested extrapolant changes outside smooth asymptotics.
pub(crate) const EXTRAPOLATION_ERROR_GUARD: usize = 4;

/// Eight epsilon cover the two affine roundings, the rounded reference gap,
/// and both independently rounded neighbors (or a neighbor and an endpoint).
/// A failed sufficient proof requests the ordinary per-point validator.
#[inline]
pub(crate) fn separated<S: RealScalar>(spacing: S, magnitude: S) -> bool {
    magnitude.is_finite()
        && spacing >= S::MIN_POSITIVE
        && spacing > (S::TWO * S::TWO * S::TWO) * S::EPSILON * magnitude
}

/// Validate a value returned by an integrand callback.
#[inline(always)]
pub(crate) fn checked_callback_value<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(non_finite_callback())
    }
}

#[cold]
#[inline(never)]
fn non_finite_callback() -> SciOxideError {
    SciOxideError::non_finite(NonFiniteOrigin::Callback, CALLBACK_DETAIL)
}

/// Validate real or complex series values with the shared callback error contract.
pub(crate) fn checked_callback_number<T: scioxide_core::NumericScalar>(
    value: T,
) -> Result<T, SciOxideError> {
    if value.has_finite_components() {
        Ok(value)
    } else {
        Err(non_finite_callback())
    }
}

/// Validate a row of callback components without changing their error origin.
pub(crate) fn checked_callback_values<S: RealScalar>(values: &[S]) -> Result<(), SciOxideError> {
    values
        .iter()
        .try_for_each(|&value| checked_callback_value(value).map(drop))
}

/// Validate an intermediate scalar produced inside a numerical kernel.
pub(crate) fn checked_intermediate<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            INTERMEDIATE_DETAIL,
        ))
    }
}

/// Construct a structured input failure without hiding its reason.
pub(crate) fn invalid(
    argument: scioxide_core::Argument,
    reason: scioxide_core::InvalidInputReason,
    detail: &'static str,
) -> SciOxideError {
    SciOxideError::invalid_input(argument, reason, detail)
}
/// Validate a one-dimensional shape, always expected before actual.
#[inline(always)]
pub(crate) fn shape(expected: usize, actual: usize) -> Result<(), SciOxideError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(expected),
            scioxide_core::Dimensions::one(actual),
        ))
    }
}
/// Convert a discrete count without silently rounding it.
pub(crate) fn number<S: RealScalar>(value: usize) -> Result<S, SciOxideError> {
    S::try_from_usize_exact(value)
}

/// Project a nonnegative diagnostic, saturating only numerical overflow.
/// Other arithmetic failures are returned unchanged.
pub(crate) fn diagnostic<S: RealScalar>(
    result: Result<S, SciOxideError>,
) -> Result<S, SciOxideError> {
    result.or_else(|error| match error.view() {
        scioxide_core::SciOxideErrorView::NumericalInstability {
            issue: scioxide_core::NumericalIssue::Overflow,
            ..
        } => Ok(S::MAX),
        _ => Err(error),
    })
}

/// Rounding-error floor `50 ε |scale|` of an accumulated rule sum, kept positive for
/// subnormal scales. A zero scale gives a zero floor; that does not mean the integral is exact.
#[inline(always)]
pub(crate) fn roundoff_floor<S: RealScalar>(scale: S) -> Result<S, SciOxideError> {
    let scale = checked_intermediate(scale)?.abs();
    let floor = (S::try_from_f64_lossy(50.0)? * S::EPSILON) * scale;
    if scale == S::ZERO || floor >= S::MIN_POSITIVE {
        return Ok(floor);
    }
    scaled_roundoff_floor(Scaled::from_value(scale)?)
}

/// Same arithmetic floor when the absolute integral itself exceeds the scalar range.
pub(crate) fn scaled_roundoff_floor<S: RealScalar>(scale: Scaled<S>) -> Result<S, SciOxideError> {
    if scale == Scaled::ZERO {
        return Ok(S::ZERO);
    }
    // `50 * eps * |scale|` is QUADPACK's `50 * epmach * resabs` term: the smallest error an
    // accumulated rule sum can honestly claim. The product runs in scaled arithmetic so that
    // a subnormal `scale` is not flushed to zero before it is rounded back.
    let floor = scale
        .abs()
        .mul(Scaled::from_value(<S as RealScalar>::EPSILON)?)?
        .mul(Scaled::from_value(S::try_from_f64_lossy(50.0)?)?)?
        .round()?;
    // Keep the floor strictly positive for any non-zero scale so tolerance comparisons stay
    // meaningful even when `50 * eps * |scale|` underflows; `min_positive * eps` is below
    // every floor derived from a normal scale and therefore only acts in that regime.
    Ok(floor.max(S::MIN_POSITIVE * <S as RealScalar>::EPSILON))
}

/// Golden-ratio conjugate `(√5 − 1)/2`, the off-grid witness fraction of an
/// interval. Its binary expansion has no long runs of equal digits, so no
/// dyadic grid or nested bisection samples close to it.
pub(crate) const WITNESS_FRACTION: f64 = 0.618_033_988_749_894_9;

/// Nodes of a witness window: unit-spaced grid samples around the witness.
pub(crate) const WITNESS_WINDOW: usize = 8;

/// Degree-seven and degree-five Lagrange interpolants at one offset of a
/// unit-spaced eight-node window, prepared once per grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WitnessInterpolant<S: RealScalar> {
    seven: [S; WITNESS_WINDOW],
    five: [S; WITNESS_WINDOW - 2],
}

impl<S: RealScalar> WitnessInterpolant<S> {
    /// Interpolants at `offset` from the first window node; the degree-five
    /// interpolant uses the six inner nodes.
    #[inline]
    pub(crate) fn at(offset: S) -> Self {
        Self {
            seven: lagrange(offset),
            five: lagrange(offset - S::ONE),
        }
    }

    /// Disagreement of an off-grid witness sample with its grid window.
    ///
    /// Zero when `|f(w) - p7(w)|` is within four times the larger of the
    /// interpolants' own change `|p7(w) - p5(w)|` and the shared arithmetic
    /// floor of the interpolation sums: resolved integrands contribute nothing.
    /// Otherwise four times `|f(w) - p7(w)|`. Grid samples that agree with each
    /// other but not with the witness reveal structure the grid cannot see, such
    /// as oscillation in step with a dyadic grid. This remains a heuristic: an
    /// integrand can still hide structure from every finite set of samples.
    #[inline(always)]
    pub(crate) fn excess(
        &self,
        window: &[S; WITNESS_WINDOW],
        witness: S,
    ) -> Result<S, SciOxideError> {
        let parts = self.disagreement(window, witness, S::ONE);
        if parts.iter().all(|part| part.is_finite()) {
            return Self::judge(parts, S::ONE);
        }
        self.scaled_excess(window, witness)
    }

    /// Values near the scalar maximum retry with an exact power-of-two scale;
    /// even edge-cell Lebesgue sums of these short windows stay far below
    /// that headroom.
    #[cold]
    #[inline(never)]
    fn scaled_excess(&self, window: &[S; WITNESS_WINDOW], witness: S) -> Result<S, SciOxideError> {
        let scale = S::ONE / S::from_f64_ieee(256.0);
        Self::judge(self.disagreement(window, witness, scale), scale)
    }

    #[inline(always)]
    fn judge([deviation, change, magnitude]: [S; 3], scale: S) -> Result<S, SciOxideError> {
        let deviation = checked_intermediate(deviation)?;
        let guard = S::from_f64_ieee(EXTRAPOLATION_ERROR_GUARD as f64);
        if deviation <= guard * checked_intermediate(change)?.max(roundoff_floor(magnitude)?) {
            return Ok(S::ZERO);
        }
        let excess = guard * deviation / scale;
        Ok(if excess.is_finite() { excess } else { S::MAX })
    }

    /// `|f(w) - p7(w)|`, `|p7(w) - p5(w)|` and the absolute mass of the
    /// interpolation sums, all times `scale`. The weights sum to one, so
    /// anchoring at a window sample keeps constants exact.
    #[inline(always)]
    fn disagreement(&self, window: &[S; WITNESS_WINDOW], witness: S, scale: S) -> [S; 3] {
        let anchor = window[WITNESS_WINDOW / 2 - 1] * scale;
        let offsets = window.map(|value| value * scale - anchor);
        let seven: [S; WITNESS_WINDOW] = core::array::from_fn(|i| self.seven[i] * offsets[i]);
        let five: [S; WITNESS_WINDOW - 2] = core::array::from_fn(|i| self.five[i] * offsets[i + 1]);
        let interpolated = lane_sum(&seven);
        [
            (witness * scale - anchor - interpolated).abs(),
            (interpolated - lane_sum(&five)).abs(),
            (anchor.abs() + (witness * scale).abs()) + lane_sum(&seven.map(S::abs)),
        ]
    }
}

/// Sum in four interleaved lanes, which shortens the dependency chain.
#[inline(always)]
fn lane_sum<S: RealScalar>(terms: &[S]) -> S {
    let mut lanes = [S::ZERO; 4];
    for (i, &term) in terms.iter().enumerate() {
        lanes[i % 4] += term;
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

/// Integral-level error of a witness excess over `geometry`: missing structure
/// seen at the witness may extend across the whole grid. Overflow saturates.
#[inline]
pub(crate) fn witness_error<S: RealScalar>(
    excess: S,
    geometry: IntervalGeometry<S>,
) -> Result<S, SciOxideError> {
    if excess == S::ZERO {
        return Ok(S::ZERO);
    }
    if let Some(span) = geometry.native_span() {
        let error = span * excess;
        if error.is_finite() && error >= S::MIN_POSITIVE {
            return Ok(error);
        }
    }
    diagnostic(
        geometry
            .span()
            .and_then(|span| span.mul(Scaled::from_value(excess)?))
            .and_then(Scaled::round),
    )
}

/// Lagrange basis of the nodes `0, 1, ..., N - 1` at `t`. Numerators multiply
/// `t - k` in increasing `k`, sharing their common prefix; the denominators
/// `prod (i - k)` are exact integers for these short windows.
#[inline]
fn lagrange<S: RealScalar, const N: usize>(t: S) -> [S; N] {
    let node = |k: usize| S::from_f64_ieee(k as f64);
    let differences: [S; N] = core::array::from_fn(|k| t - node(k));
    let mut prefix = S::ONE;
    core::array::from_fn(|i| {
        let numerator = differences[i + 1..]
            .iter()
            .fold(prefix, |product, &difference| product * difference);
        let denominator = (0..N)
            .filter(|&k| k != i)
            .fold(S::ONE, |product, k| product * (node(i) - node(k)));
        prefix *= differences[i];
        numerator / denominator
    })
}

/// Stable finite interval coordinate and weight map.
#[derive(Clone, Copy)]
pub(crate) struct AffineMap1D<S: RealScalar> {
    geometry: IntervalGeometry<S>,
}

impl<S: RealScalar> AffineMap1D<S> {
    /// Map for an interval whose span is a normal finite number, or `None`.
    #[inline(always)]
    pub(crate) fn native(interval: Interval<S>) -> Option<Self> {
        IntervalGeometry::native(interval).map(|geometry| Self { geometry })
    }

    #[inline]
    pub(crate) fn from_interval(interval: Interval<S>) -> Result<Option<Self>, SciOxideError> {
        Ok(IntervalGeometry::prepare(interval)?.map(|geometry| Self { geometry }))
    }

    #[inline]
    pub(crate) fn point(self, canonical: S) -> Result<S, SciOxideError> {
        self.geometry.canonical_point(canonical)
    }

    pub(crate) fn weight(self, canonical: S) -> Result<S, SciOxideError> {
        if let Some(span) = self.geometry.native_span() {
            let weight = (span * S::HALF) * canonical;
            if canonical == S::ZERO || weight.is_finite() && weight.abs() >= S::MIN_POSITIVE {
                return Ok(weight);
            }
        }
        self.scaled_weight(canonical)?.round()
    }

    pub(crate) fn native_half_span(self) -> Option<S> {
        self.geometry.native_span().map(|span| span * S::HALF)
    }

    /// Half-span as a product that rescales instead of overflowing.
    pub(crate) fn half_weight(self) -> Result<scioxide_core::PromotingProduct<S>, SciOxideError> {
        match self.native_half_span() {
            Some(half) => scioxide_core::PromotingProduct::from_value(half),
            None => Ok(scioxide_core::PromotingProduct::from_scaled(
                self.scaled_weight(S::ONE)?,
            )),
        }
    }

    /// Frame whose midpoint products are exact and whose interior affine maps
    /// and nonzero weights stay normal and finite throughout the complete rule.
    pub(crate) fn native_frame(self, min_node: S, min_weight: S, max_weight: S) -> Option<(S, S)> {
        let geometry = self.geometry;
        let half = self.native_half_span()?;
        let (middle, half) = Self::endpoint_frame(geometry.lower(), geometry.upper(), half)?;
        (half * min_node >= S::MIN_POSITIVE
            && half * min_weight >= S::MIN_POSITIVE
            && (half * max_weight).is_finite())
        .then_some((middle, half))
    }

    /// The caller has checked that the half-span is finite and positive. Halving the
    /// endpoints must be exact and leave ample room for interior affine sums.
    #[inline(always)]
    fn endpoint_frame(lower: S, upper: S, half: S) -> Option<(S, S)> {
        let exact_half = |x: S| x == S::ZERO || x.abs() >= S::TWO * S::MIN_POSITIVE;
        if exact_half(lower)
            && exact_half(upper)
            && lower.abs().max(upper.abs()) <= S::MAX * (S::HALF * S::HALF)
        {
            Some((lower * S::HALF + upper * S::HALF, half))
        } else {
            None
        }
    }

    pub(crate) fn scaled_weight(self, canonical: S) -> Result<Scaled<S>, SciOxideError> {
        self.geometry.scaled_half_weight(canonical)
    }
}

/// Owned, validated fixed-rule data prepared before application.
#[derive(Debug)]
pub(crate) struct PreparedRule<S: RealScalar> {
    nodes: Vec<S>,
    weights: Vec<S>,
    min_node: S,
    min_weight: S,
    max_weight: S,
    min_native_half: S,
    max_native_half: S,
    separation: S,
    interval_gap: Option<S>,
    lower: S,
    upper: S,
    ordered: bool,
    endpoints: bool,
    negative_zero: bool,
    coefficients: Vec<[S; 2]>,
}

impl<S: RealScalar> PreparedRule<S> {
    pub(crate) fn nodes(&self) -> &[S] {
        &self.nodes
    }
    pub(crate) fn weights(&self) -> &[S] {
        &self.weights
    }

    /// Validate owned rule data; `argument` names the
    /// parameter from which the rule was constructed.
    pub(crate) fn new(
        nodes: Vec<S>,
        weights: Vec<S>,
        argument: scioxide_core::Argument,
    ) -> Result<Self, SciOxideError> {
        if nodes.is_empty() || nodes.len() != weights.len() {
            return Err(SciOxideError::invalid_input(
                argument,
                scioxide_core::InvalidInputReason::Inconsistent,
                "quadrature nodes and weights must have the same non-zero length",
            ));
        }
        if nodes.iter().any(|node| !node.is_finite()) {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Output,
                "prepared rule contains a non-finite node",
            ));
        }
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= S::ZERO)
        {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "prepared rule contains a non-finite or non-positive weight",
            ));
        }
        let min_node = nodes
            .iter()
            .filter(|&&x| x != S::ZERO)
            .fold(S::ONE, |m, &x| m.min(x.abs()));
        let min_weight = weights.iter().fold(S::MAX, |m, &x| m.min(x));
        let max_weight = weights.iter().fold(S::ZERO, |m, &x| m.max(x));
        // These support-independent bounds are deliberately sufficient. The
        // factor-two lower and factor-eight upper margins cover division
        // rounding and imply the ordinary per-coefficient product admission.
        // A miss keeps the original checked-frame mapping and rounding order.
        let min_native_half =
            (S::TWO * S::MIN_POSITIVE / min_node.min(min_weight)).max(S::MIN_POSITIVE / S::EPSILON);
        let max_native_half = (S::MAX * (S::HALF * S::HALF * S::HALF)) / max_weight.max(S::ONE);
        let mut validator = MappedPoints::new(PointDomain::RealLine, false);
        let ordered = nodes.iter().all(|&node| validator.push(node) == Ok(true));
        let separation = nodes
            .windows(2)
            .fold(S::ONE, |gap, pair| gap.min((pair[1] - pair[0]).abs()));
        let endpoint_clearance = nodes
            .iter()
            .filter(|&&x| x.abs() != S::ONE)
            .fold(S::ONE, |gap, &x| gap.min(S::ONE - x.abs()));
        let negative_zero = nodes
            .iter()
            .any(|&node| node == S::ZERO && node.is_sign_negative());
        let endpoints = nodes.iter().any(|&node| node.abs() == S::ONE);
        let lower = nodes.iter().copied().fold(S::MAX, S::min);
        let upper = nodes.iter().copied().fold(-S::MAX, S::max);
        let interval_gap = (ordered
            && lower >= -S::ONE
            && upper <= S::ONE
            && (nodes.len() != 1 || nodes[0].abs() < S::ONE))
            .then_some(separation.min(endpoint_clearance));
        let mut coefficients = scioxide_core::backend::try_with_capacity(
            nodes.len(),
            scioxide_core::quantity!("canonical_coefficients"),
        )?;
        coefficients.extend(
            nodes
                .iter()
                .zip(&weights)
                .map(|(&node, &weight)| [node, weight]),
        );
        Ok(Self {
            nodes,
            weights,
            min_node,
            min_weight,
            max_weight,
            min_native_half,
            max_native_half,
            separation,
            interval_gap,
            lower,
            upper,
            ordered,
            endpoints,
            negative_zero,
            coefficients,
        })
    }

    pub(crate) const fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// Check a complete rule once, keeping range branches outside its node loop.
    #[inline(always)]
    pub(crate) fn map_into(
        &self,
        map: AffineMap1D<S>,
        scale_weights: bool,
        destination: &mut [crate::FixedRuleNode<S>],
    ) -> Result<bool, SciOxideError> {
        shape(self.len(), destination.len())?;
        let geometry = map.geometry;
        let min_weight = if scale_weights {
            self.min_weight
        } else {
            S::ONE
        };
        let max_weight = if scale_weights {
            self.max_weight
        } else {
            S::ONE
        };
        if let Some((middle, half)) = map.native_frame(self.min_node, min_weight, max_weight) {
            let write = |slot: &mut crate::FixedRuleNode<S>, point, weight| {
                *slot = crate::FixedRuleNode::new(
                    point,
                    if scale_weights { half * weight } else { weight },
                    None,
                );
            };
            if self.endpoints {
                for (slot, (&node, &weight)) in destination
                    .iter_mut()
                    .zip(self.nodes.iter().zip(&self.weights))
                {
                    let point = if node == -S::ONE {
                        geometry.lower()
                    } else if node == S::ONE {
                        geometry.upper()
                    } else {
                        middle + half * node
                    };
                    write(slot, point, weight);
                }
            } else {
                for (slot, (&node, &weight)) in destination
                    .iter_mut()
                    .zip(self.nodes.iter().zip(&self.weights))
                {
                    write(slot, middle + half * node, weight);
                }
            }
            return Ok(self
                .interval_gap
                .is_some_and(|gap| separated(half * gap, middle.abs() + half)));
        }
        self.map_checked_into(map, scale_weights, destination)?;
        Ok(false)
    }

    #[cold]
    #[inline(never)]
    fn map_checked_into(
        &self,
        map: AffineMap1D<S>,
        scale_weights: bool,
        destination: &mut [crate::FixedRuleNode<S>],
    ) -> Result<(), SciOxideError> {
        for (slot, (&node, &weight)) in destination
            .iter_mut()
            .zip(self.nodes.iter().zip(&self.weights))
        {
            *slot = crate::FixedRuleNode::new(
                map.point(node)?,
                if scale_weights {
                    map.weight(weight)?
                } else {
                    weight
                },
                None,
            );
        }
        Ok(())
    }

    /// Affine map for this call, checked once for the whole rule.
    #[inline(always)]
    pub(crate) fn native_interval(
        &self,
        interval: Interval<S>,
        scale_weights: bool,
    ) -> Option<NativeStencil<'_, S>> {
        let gap = self.interval_gap?;
        let half = (interval.upper() - interval.lower()) * S::HALF;
        if !(half >= self.min_native_half && half <= self.max_native_half) {
            return None;
        }
        let (middle, half) = AffineMap1D::endpoint_frame(interval.lower(), interval.upper(), half)?;
        if !separated(half * gap, middle.abs() + half) {
            return None;
        }
        Some(NativeStencil {
            rule: self,
            scale: if scale_weights { half } else { S::ONE },
            map: if self.endpoints {
                NativePointMap::Closed {
                    middle,
                    half,
                    lower: interval.lower(),
                    upper: interval.upper(),
                }
            } else {
                NativePointMap::Open { middle, half }
            },
        })
    }

    #[inline(always)]
    pub(crate) fn native_translation(&self, lower: S) -> Option<NativeStencil<'_, S>> {
        self.admits_translation(lower).then_some(NativeStencil {
            rule: self,
            scale: S::ONE,
            map: NativePointMap::Open {
                middle: lower,
                half: S::ONE,
            },
        })
    }

    #[inline(always)]
    pub(crate) fn native_identity(&self) -> Option<NativeStencil<'_, S>> {
        (self.ordered && !self.negative_zero).then_some(NativeStencil {
            rule: self,
            scale: S::ONE,
            map: NativePointMap::Open {
                middle: S::ZERO,
                half: S::ONE,
            },
        })
    }

    /// Translation of a positive ascending rule; the same gap proof guards
    /// distinct neighbors and the open half-line endpoint before any callback.
    pub(crate) fn admits_translation(&self, lower: S) -> bool {
        self.ordered
            && self.nodes[0] == self.lower
            && self.lower > S::ZERO
            && separated(
                self.separation.min(self.lower),
                lower.abs() + self.upper.abs(),
            )
    }
}

/// Checked affine map of a rule, valid for one call. Its reference nodes do not
/// change, and coordinates are generated without fallible work during sampling.
#[derive(Clone, Copy)]
pub(crate) struct NativeStencil<'a, S: RealScalar> {
    rule: &'a PreparedRule<S>,
    scale: S,
    map: NativePointMap<S>,
}
#[derive(Clone, Copy)]
enum NativePointMap<S: RealScalar> {
    Open {
        middle: S,
        half: S,
    },
    Closed {
        middle: S,
        half: S,
        lower: S,
        upper: S,
    },
}
impl<S: RealScalar> NativePointMap<S> {
    #[inline(always)]
    fn point(self, node: S) -> S {
        match self {
            Self::Open { middle, half } => middle + half * node,
            Self::Closed {
                middle,
                half,
                lower,
                upper,
            } => {
                if node == -S::ONE {
                    lower
                } else if node == S::ONE {
                    upper
                } else {
                    middle + half * node
                }
            }
        }
    }
}
impl<S: RealScalar> NativeStencil<'_, S> {
    pub(crate) fn map_into(self, destination: &mut [crate::FixedRuleNode<S>]) {
        for (slot, (&node, &weight)) in destination
            .iter_mut()
            .zip(self.rule.nodes.iter().zip(&self.rule.weights))
        {
            *slot = crate::FixedRuleNode::new(self.map.point(node), self.scale * weight, None);
        }
    }
    #[inline(always)]
    pub(crate) fn sample<F, V>(
        self,
        function: &F,
        rows: &mut [S],
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: scioxide_core::AffineValue<Scale = S>,
    {
        match self.map {
            NativePointMap::Open { middle, half } => {
                self.sample_with(function, rows, context, |node| middle + half * node)
            }
            NativePointMap::Closed {
                middle,
                half,
                lower,
                upper,
            } => self.sample_with(function, rows, context, |node| {
                if node == -S::ONE {
                    lower
                } else if node == S::ONE {
                    upper
                } else {
                    middle + half * node
                }
            }),
        }
    }
    #[inline(always)]
    fn sample_with<F, V>(
        self,
        function: &F,
        rows: &mut [S],
        context: &mut scioxide_core::SolveContext<'_, '_>,
        point: impl Fn(S) -> S,
    ) -> Result<S, SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: scioxide_core::AffineValue<Scale = S>,
    {
        if self.scale * self.rule.max_weight < S::ONE || self.scale * self.rule.min_weight > S::ONE
        {
            self.sample_weighted::<F, V, true>(function, rows, context, point)
        } else {
            self.sample_weighted::<F, V, false>(function, rows, context, point)
        }
    }

    #[inline(always)]
    fn sample_weighted<F, V, const POSITIVE_NONUNIT: bool>(
        self,
        function: &F,
        rows: &mut [S],
        context: &mut scioxide_core::SolveContext<'_, '_>,
        point: impl Fn(S) -> S,
    ) -> Result<S, SciOxideError>
    where
        F: Fn(S) -> Result<V, SciOxideError> + Sync + ?Sized,
        V: scioxide_core::AffineValue<Scale = S>,
    {
        let coefficients = &self.rule.coefficients;
        crate::evaluation::sample_scalar_with::<S, _, 1, POSITIVE_NONUNIT, _>(
            coefficients,
            rows,
            context,
            {
                #[inline(always)]
                |coefficient: &[S; 2]| {
                    crate::evaluation::scalar_value(&function(point(coefficient[0]))?)
                }
            },
            |coefficient| self.scale * coefficient[1],
            |[value]| Ok(value + S::ZERO),
            |[sum]| sum.round(),
        )
    }
}

/// Domain and endpoint policy for one ordered mapped stencil.
#[derive(Clone, Copy)]
pub(crate) enum PointDomain<S: RealScalar> {
    Interval { interval: Interval<S>, closed: bool },
    HalfLine { lower: S },
    RealLine,
}

/// One finite/domain/distinctness/monotonicity validator. Callers choose whether
/// either monotone direction or only ascending order is part of their contract.
pub(crate) struct MappedPoints<S: RealScalar> {
    domain: PointDomain<S>,
    previous: Option<S>,
    direction: Option<core::cmp::Ordering>,
}
impl<S: RealScalar> MappedPoints<S> {
    pub(crate) fn new(domain: PointDomain<S>, ascending: bool) -> Self {
        Self {
            domain,
            previous: None,
            direction: ascending.then_some(core::cmp::Ordering::Greater),
        }
    }

    pub(crate) fn push(&mut self, point: S) -> Result<bool, SciOxideError> {
        if !point.is_finite() {
            return Err(SciOxideError::non_finite(
                scioxide_core::NonFiniteOrigin::Output,
                "mapped rule exposed a non-finite callback coordinate",
            ));
        }
        match self.domain {
            PointDomain::Interval { interval, closed } => {
                if point < interval.lower() || point > interval.upper() {
                    return Err(SciOxideError::numerical_instability(
                        scioxide_core::NumericalIssue::Breakdown,
                        "mapped rule exposed an out-of-domain callback coordinate",
                    ));
                }
                if !closed && (point == interval.lower() || point == interval.upper()) {
                    return Ok(false);
                }
            }
            PointDomain::HalfLine { lower } if point <= lower => return Ok(false),
            _ => {}
        }
        if let Some(previous) = self.previous {
            let direction = point
                .partial_cmp(&previous)
                .unwrap_or(core::cmp::Ordering::Equal);
            if direction == core::cmp::Ordering::Equal
                || self.direction.is_some_and(|d| d != direction)
            {
                return Ok(false);
            }
            self.direction = Some(direction);
        }
        self.previous = Some(point);
        Ok(true)
    }
}

/// Check ordered mapped points, allowing endpoints only for closed rules.
pub(crate) fn mapped_geometry<S: RealScalar>(
    points: impl IntoIterator<Item = S>,
    interval: Interval<S>,
    closed: bool,
) -> bool {
    let mut validator = MappedPoints::new(PointDomain::Interval { interval, closed }, false);
    points
        .into_iter()
        .all(|point| validator.push(point) == Ok(true))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn bulk_admission_implies_valid_metadata_at_spacing_boundaries() {
        fn check<S: RealScalar>() {
            let mut admitted = 0;
            let mut fallback = 0;
            for canonical in [
                alloc::vec![-S::HALF, S::ZERO, S::HALF],
                alloc::vec![S::ONE, S::HALF, S::ZERO, -S::HALF, -S::ONE],
                alloc::vec![S::ZERO],
            ] {
                let weights = alloc::vec![S::ONE; canonical.len()];
                let rule = PreparedRule::new(canonical, weights, scioxide_core::argument!("rule"))
                    .unwrap();
                let mut nodes =
                    alloc::vec![crate::FixedRuleNode::new(S::ZERO, S::ZERO, None); rule.len()];
                for scale in [
                    S::MIN_POSITIVE,
                    S::EPSILON,
                    S::ONE,
                    S::MAX * S::HALF * S::HALF,
                ] {
                    for lower in [-scale, S::ZERO, scale] {
                        for width in [
                            scale * S::EPSILON * (S::TWO * S::TWO),
                            scale * S::EPSILON * S::try_from_usize_exact(64).unwrap(),
                            scale,
                        ] {
                            let upper = lower + width;
                            if !upper.is_finite() || upper <= lower {
                                continue;
                            }
                            let interval = Interval::new(lower, upper).unwrap();
                            let map = AffineMap1D::from_interval(interval).unwrap().unwrap();
                            let mapped = rule.map_into(map, true, &mut nodes).unwrap();
                            if let Some(stencil) = rule.native_interval(interval, true) {
                                assert!(mapped);
                                let mut direct = alloc::vec![
                                    crate::FixedRuleNode::new(S::ZERO, S::ZERO, None);
                                    rule.len()
                                ];
                                stencil.map_into(&mut direct);
                                for (mapped, direct) in nodes.iter().zip(direct) {
                                    assert_eq!(
                                        mapped.point().into_f64().to_bits(),
                                        direct.point().into_f64().to_bits()
                                    );
                                    assert_eq!(
                                        mapped.primary_weight().into_f64().to_bits(),
                                        direct.primary_weight().into_f64().to_bits()
                                    );
                                }
                            }
                            if mapped {
                                admitted += 1;
                                assert!(mapped_geometry(
                                    nodes.iter().map(|node| node.point()),
                                    interval,
                                    rule.len() != 1
                                ));
                                assert!(nodes.iter().all(|node| node.primary_weight().is_finite()
                                    && node.primary_weight() > S::ZERO));
                            } else {
                                fallback += 1;
                            }
                        }
                    }
                }
            }
            assert!(admitted > 0 && fallback > 0);
            let unordered = PreparedRule::new(
                alloc::vec![S::ZERO, S::HALF, S::ZERO],
                alloc::vec![S::ONE; 3],
                scioxide_core::argument!("rule"),
            )
            .unwrap();
            let map = AffineMap1D::from_interval(Interval::new(S::ZERO, S::ONE).unwrap())
                .unwrap()
                .unwrap();
            assert!(
                !unordered
                    .map_into(
                        map,
                        true,
                        &mut [crate::FixedRuleNode::new(S::ZERO, S::ZERO, None); 3]
                    )
                    .unwrap()
            );
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn native_half_bounds_preserve_legacy_admission_and_mapping_bits() {
        fn check<S: RealScalar>() {
            let tiny = S::MIN_POSITIVE.sqrt();
            let large = S::MAX.sqrt();
            for weights in [
                [tiny, tiny * S::TWO, S::HALF],
                [large, large * S::HALF, S::TWO],
                [tiny, S::HALF, large],
            ] {
                let rule = PreparedRule::new(
                    alloc::vec![-S::HALF, S::ZERO, S::HALF],
                    weights.into(),
                    scioxide_core::argument!("rule"),
                )
                .unwrap();
                for scale_weights in [false, true] {
                    let mut admitted = 0;
                    let mut rejected = 0;
                    for bound in [rule.min_native_half, rule.max_native_half] {
                        for half in [
                            bound.next_after(S::NEG_INFINITY),
                            bound,
                            bound.next_after(S::INFINITY),
                        ] {
                            let interval = Interval::new(-half, half).unwrap();
                            let native = rule.native_interval(interval, scale_weights);
                            assert_eq!(
                                native.is_some(),
                                half >= rule.min_native_half && half <= rule.max_native_half
                            );
                            let Some(stencil) = native else {
                                rejected += 1;
                                continue;
                            };
                            admitted += 1;
                            let map = AffineMap1D::from_interval(interval).unwrap().unwrap();
                            assert!(
                                map.native_frame(
                                    rule.min_node,
                                    if scale_weights {
                                        rule.min_weight
                                    } else {
                                        S::ONE
                                    },
                                    if scale_weights {
                                        rule.max_weight
                                    } else {
                                        S::ONE
                                    },
                                )
                                .is_some()
                            );
                            let empty = crate::FixedRuleNode::new(S::ZERO, S::ZERO, None);
                            let mut legacy = [empty; 3];
                            assert!(rule.map_into(map, scale_weights, &mut legacy).unwrap());
                            let mut direct = [empty; 3];
                            stencil.map_into(&mut direct);
                            for (legacy, direct) in legacy.into_iter().zip(direct) {
                                assert_eq!(
                                    direct.point().into_f64().to_bits(),
                                    legacy.point().into_f64().to_bits()
                                );
                                assert_eq!(
                                    direct.primary_weight().into_f64().to_bits(),
                                    legacy.primary_weight().into_f64().to_bits()
                                );
                            }
                        }
                    }
                    assert!(admitted > 0 && rejected > 0);
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn translated_admission_protects_the_open_endpoint_and_neighbors() {
        fn check<S: RealScalar>() {
            let rule = PreparedRule::new(
                alloc::vec![S::HALF, S::ONE, S::TWO],
                alloc::vec![S::ONE; 3],
                scioxide_core::argument!("rule"),
            )
            .unwrap();
            assert!(rule.admits_translation(S::ZERO));
            assert!(!rule.admits_translation(S::ONE / S::EPSILON));
            for lower in [-S::MAX, -S::ONE, S::ZERO, S::ONE, S::ONE / S::EPSILON] {
                if rule.admits_translation(lower) {
                    let mut validator = MappedPoints::new(PointDomain::HalfLine { lower }, true);
                    assert!(
                        rule.nodes()
                            .iter()
                            .all(|&node| validator.push(lower + node) == Ok(true))
                    );
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn mapped_domains_share_finite_endpoint_and_direction_checks() {
        fn check<S: RealScalar>() {
            let interval = Interval::new(S::ZERO, S::ONE).unwrap();
            assert!(mapped_geometry([S::ONE, S::HALF, S::ZERO], interval, true));
            assert!(!mapped_geometry([S::ZERO], interval, false));
            assert!(!mapped_geometry([S::HALF, S::HALF], interval, true));
            assert!(!mapped_geometry([S::ZERO, S::ONE, S::HALF], interval, true));
            for invalid in [-S::ONE, S::TWO, S::NAN, S::INFINITY] {
                assert!(!mapped_geometry([invalid], interval, true));
            }
            let mut half = MappedPoints::new(PointDomain::HalfLine { lower: S::ZERO }, true);
            assert!(!half.push(S::ZERO).unwrap());
            assert!(half.push(S::ONE).unwrap());
            assert!(!half.push(S::HALF).unwrap());
            let mut line = MappedPoints::new(PointDomain::RealLine, true);
            assert!(line.push(-S::MAX).unwrap());
            assert!(line.push(S::MAX).unwrap());
            assert!(line.push(S::NAN).is_err());
        }
        check::<f32>();
        check::<f64>();
    }
    #[test]
    fn compensation_and_degenerate_mapping() {
        let sum = [1e16_f64, 1.0, -1e16]
            .into_iter()
            .try_fold(
                scioxide_core::PromotingCompensatedSum::ZERO,
                |mut sum, x| {
                    sum.add_value(x)?;
                    Ok::<_, SciOxideError>(sum)
                },
            )
            .unwrap();
        assert_eq!(sum.round().unwrap(), 1.0);
        let sum = [1e8_f32, 1.0, -1e8]
            .into_iter()
            .try_fold(
                scioxide_core::PromotingCompensatedSum::ZERO,
                |mut sum, x| {
                    sum.add_value(x)?;
                    Ok::<_, SciOxideError>(sum)
                },
            )
            .unwrap();
        assert_eq!(sum.round().unwrap(), 1.0);
        assert!(
            AffineMap1D::from_interval(Interval::new(2.0_f64, 2.0).unwrap())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn callback_admission_preserves_finite_bits_at_scalar_boundaries() {
        fn check<S: RealScalar>() {
            for value in [
                -S::MAX,
                -S::MIN_POSITIVE,
                -S::MIN_SUBNORMAL,
                S::NEG_ZERO,
                S::ZERO,
                S::MIN_SUBNORMAL,
                S::MIN_POSITIVE,
                S::MAX,
            ] {
                assert_eq!(
                    checked_callback_value(value).unwrap().into_f64().to_bits(),
                    value.into_f64().to_bits()
                );
            }
            for value in [S::NAN, S::INFINITY, -S::INFINITY] {
                assert_eq!(
                    checked_callback_value(value).unwrap_err().kind(),
                    scioxide_core::SciOxideErrorKind::NonFinite
                );
            }
        }
        check::<f32>();
        check::<f64>();
    }
}
