//! Double-exponential nodes and weights prepared in `S`, and the per-segment
//! checks of mapped nodes.

use super::DoubleExponentialDomain;
use super::kernel::MAX_TAIL_SHELLS;
use crate::geometry::IntervalGeometry;
use alloc::vec::Vec;
use scioxide_core::{Interval, RealScalar, Scaled, SciOxideError};

/// A symmetric node pair on the reference domain: endpoint fractions for finite
/// domains, endpoint distances for half-lines, and a positive coordinate for
/// the real line.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Node<S> {
    near: S,
    far: S,
    near_weight: S,
    far_weight: S,
}

/// The finest dyadic grid, shared by every nested level and finite segment.
/// The table ends where the transform saturates; it holds no non-finite entries.
#[derive(Debug, PartialEq)]
pub(super) struct BasisTable<S: RealScalar> {
    grid: usize,
    values: Vec<Node<S>>,
}

impl<S: RealScalar> BasisTable<S> {
    pub(super) fn new(
        levels: usize,
        domain: DoubleExponentialDomain<'_, S>,
    ) -> Result<Self, SciOxideError> {
        let grid = 1usize << (levels - 1);
        let scale = S::try_from_usize_exact(grid)?;
        let mut values = scioxide_core::backend::try_with_capacity(
            MAX_TAIL_SHELLS * grid + 1,
            scioxide_core::quantity!("de_nodes"),
        )?;
        let pi_half = S::PI * S::HALF;
        for index in 0..=MAX_TAIL_SHELLS * grid {
            let t = S::try_from_usize_exact(index)? / scale;
            let u = pi_half * t.sinh();
            let base = pi_half * t.cosh();
            let node = match domain {
                DoubleExponentialDomain::Finite { .. } => {
                    let e = (-S::TWO * u).exp();
                    if e < S::MIN_POSITIVE {
                        break;
                    }
                    let denominator = S::ONE + e;
                    Node {
                        near: e / denominator,
                        far: S::ZERO,
                        near_weight: S::TWO * S::TWO * base * e / (denominator * denominator),
                        far_weight: S::ZERO,
                    }
                }
                DoubleExponentialDomain::HalfLine { .. } => {
                    let far = u.exp();
                    let near = (-u).exp();
                    Node {
                        near,
                        far,
                        near_weight: base * near,
                        far_weight: base * far,
                    }
                }
                DoubleExponentialDomain::RealLine => {
                    let exponential = u.exp();
                    let reciprocal = (-u).exp();
                    Node {
                        near: (exponential - reciprocal) * S::HALF,
                        far: S::ZERO,
                        near_weight: base * ((exponential + reciprocal) * S::HALF),
                        far_weight: S::ZERO,
                    }
                }
            };
            if !node.near.is_finite()
                || !node.far.is_finite()
                || !node.near_weight.is_finite()
                || !node.far_weight.is_finite()
                || node.near_weight <= S::ZERO
                || (matches!(domain, DoubleExponentialDomain::HalfLine { .. })
                    && (node.near <= S::ZERO || node.far_weight <= S::ZERO))
            {
                break;
            }
            values.push(node);
        }
        Ok(Self { grid, values })
    }

    #[inline(always)]
    fn pair(&self, map: DeMap<S>, index: usize) -> PairPreparation<S> {
        if index == 0 {
            return map.center().map_or(PairPreparation::Saturated, |center| {
                PairPreparation::Complete(PreparedPair {
                    negative: Some(center),
                    positive: Some(center),
                })
            });
        }
        self.values
            .get(index)
            .map_or(PairPreparation::Saturated, |&node| map.pair(node))
    }

    pub(super) fn stride(&self, level: usize) -> usize {
        self.grid >> level
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum DeMap<S: RealScalar> {
    Finite {
        geometry: IntervalGeometry<S>,
        center: S,
        half_width: S,
        pi_half: S,
    },
    SemiInfinite {
        lower: S,
        pi_half: S,
    },
    Infinite {
        pi_half: S,
    },
}

#[derive(Debug, Clone, Copy)]
pub(super) struct WeightedPoint<S: RealScalar> {
    pub(super) point: S,
    pub(super) weight: S,
    pub(super) endpoint: Option<(S, S)>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PreparedPair<S: RealScalar> {
    pub(super) negative: Option<WeightedPoint<S>>,
    pub(super) positive: Option<WeightedPoint<S>>,
}

/// Advances using the stored endpoint distances, even where coordinates coincide.
pub(super) fn pair_advances<S: RealScalar>(inner: PreparedPair<S>, outer: PreparedPair<S>) -> bool {
    let negative = match (inner.negative, outer.negative) {
        (Some(a), Some(b)) => match (a.endpoint, b.endpoint) {
            (Some((_, a)), Some((_, b))) => b < a,
            _ => b.point < a.point,
        },
        (None, Some(_)) => false,
        _ => true,
    };
    let positive = match (inner.positive, outer.positive) {
        (Some(a), Some(b)) => match (a.endpoint, b.endpoint) {
            (Some((ea, da)), Some((eb, db))) if ea == eb => {
                if ea >= a.point {
                    db < da
                } else {
                    da < db
                }
            }
            _ => a.point < b.point,
        },
        (None, Some(_)) => false,
        _ => true,
    };
    negative && positive
}

pub(super) enum PairPreparation<S: RealScalar> {
    Complete(PreparedPair<S>),
    Saturated,
}

/// Geometry checks of every segment, including neighbor separation, prepared
/// once. Mapped coordinates and weights are not stored between calls.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PairStatus {
    Complete,
    Empty,
    Saturated,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct Admission {
    pub(super) evaluations: usize,
    pub(super) partial: bool,
    pub(super) failed: bool,
}
impl Admission {
    fn add(&mut self, status: PairStatus) {
        self.evaluations += 2 * usize::from(status == PairStatus::Complete);
        self.partial |= status != PairStatus::Complete;
        self.failed |= status == PairStatus::Saturated;
    }
}

#[derive(Debug, PartialEq)]
struct LevelAdmission {
    pairs: Vec<PairStatus>,
    odd: [Admission; MAX_TAIL_SHELLS],
    full: [Admission; MAX_TAIL_SHELLS],
}

#[derive(Debug, PartialEq)]
pub(super) struct PreparedMap<S: RealScalar> {
    map: DeMap<S>,
    levels: Vec<LevelAdmission>,
}
impl<S: RealScalar> PreparedMap<S> {
    pub(super) fn new(
        map: DeMap<S>,
        basis: &BasisTable<S>,
        levels: usize,
    ) -> Result<Self, SciOxideError> {
        let mut admitted = scioxide_core::backend::try_with_capacity(
            levels,
            scioxide_core::quantity!("de_levels"),
        )?;
        for level in 0..levels {
            let count = 1usize << level;
            let stride = basis.stride(level);
            let mut pairs = scioxide_core::backend::try_with_capacity(
                MAX_TAIL_SHELLS * count + 1,
                scioxide_core::quantity!("de_admission"),
            )?;
            pairs.push(PairStatus::Complete);
            let mut odd = [Admission::default(); MAX_TAIL_SHELLS];
            let mut full = odd;
            for index in 1..=MAX_TAIL_SHELLS * count {
                let status = match basis.pair(map, index * stride) {
                    PairPreparation::Saturated
                        if level > 0 && matches!(map, DeMap::Finite { .. }) =>
                    {
                        PairStatus::Empty
                    }
                    PairPreparation::Saturated => PairStatus::Saturated,
                    PairPreparation::Complete(pair) => {
                        let inner = basis.pair(map, (index - 1) * stride);
                        let outer = basis.pair(map, (index + 1) * stride);
                        if matches!(inner, PairPreparation::Complete(other) if !pair_advances(other, pair))
                            || (level > 0
                                && matches!(outer, PairPreparation::Complete(other) if !pair_advances(pair, other)))
                        {
                            PairStatus::Saturated
                        } else {
                            PairStatus::Complete
                        }
                    }
                };
                pairs.push(status);
                let shell = (index - 1) / count;
                full[shell].add(status);
                if index % 2 == 1 {
                    odd[shell].add(status);
                }
            }
            admitted.push(LevelAdmission { pairs, odd, full });
        }
        Ok(Self {
            map,
            levels: admitted,
        })
    }
    pub(super) fn center(&self) -> Result<WeightedPoint<S>, SciOxideError> {
        self.map.center()
    }
    pub(super) fn finite(&self) -> bool {
        matches!(self.map, DeMap::Finite { .. })
    }
    pub(super) fn evaluations(&self, level: usize, index: usize) -> Option<usize> {
        match self.levels[level].pairs[index] {
            PairStatus::Complete => Some(2),
            PairStatus::Empty => Some(0),
            PairStatus::Saturated => None,
        }
    }
    pub(super) fn admission(&self, level: usize, shell: usize, odd: bool) -> Admission {
        if odd {
            self.levels[level].odd[shell]
        } else {
            self.levels[level].full[shell]
        }
    }
    #[inline(always)]
    pub(super) fn pair(
        &self,
        level: usize,
        index: usize,
        basis: &BasisTable<S>,
    ) -> PairPreparation<S> {
        match self.levels[level].pairs[index] {
            PairStatus::Complete => self
                .map
                .mapped_pair(basis.values[index * basis.stride(level)]),
            PairStatus::Empty => PairPreparation::Complete(PreparedPair {
                negative: None,
                positive: None,
            }),
            PairStatus::Saturated => PairPreparation::Saturated,
        }
    }
}

/// Numerical geometry failures are reported when a call runs; storage failures
/// fail plan preparation.
#[derive(Debug, PartialEq)]
pub(super) enum Geometry<S: RealScalar> {
    Ready(Vec<PreparedMap<S>>),
    Unavailable,
    Failed(SciOxideError),
}
impl<S: RealScalar> Geometry<S> {
    pub(super) fn new(
        domain: DoubleExponentialDomain<'_, S>,
        basis: &BasisTable<S>,
        levels: usize,
    ) -> Result<Self, SciOxideError> {
        let segments = match domain {
            DoubleExponentialDomain::Finite {
                singular_points, ..
            } => singular_points.len() + 1,
            _ => 1,
        };
        let mut maps = scioxide_core::backend::try_with_capacity(
            segments,
            scioxide_core::quantity!("de_maps"),
        )?;
        let mut add =
            |map: Result<Option<DeMap<S>>, SciOxideError>| -> Result<Option<Self>, SciOxideError> {
                match map {
                    Ok(Some(map)) => {
                        maps.push(PreparedMap::new(map, basis, levels)?);
                        Ok(None)
                    }
                    Ok(None) => Ok(Some(Self::Unavailable)),
                    Err(error) => Ok(Some(Self::Failed(error))),
                }
            };
        match domain {
            DoubleExponentialDomain::Finite {
                interval,
                singular_points,
            } => {
                if interval.lower() == interval.upper() {
                    return Ok(Self::Unavailable);
                }
                let mut lower = interval.lower();
                for upper in singular_points
                    .iter()
                    .copied()
                    .chain(core::iter::once(interval.upper()))
                {
                    if let Some(failure) = add(DeMap::finite(Interval::new(lower, upper)?))? {
                        return Ok(failure);
                    }
                    lower = upper;
                }
            }
            DoubleExponentialDomain::HalfLine { lower } => {
                let map = DeMap::semi_infinite(lower);
                let prepared = map
                    .center()
                    .map(|center| (center.point > lower).then_some(map));
                if let Some(failure) = add(prepared)? {
                    return Ok(failure);
                }
            }
            DoubleExponentialDomain::RealLine => {
                add(Ok(Some(DeMap::infinite())))?;
            }
        }
        Ok(Self::Ready(maps))
    }
}

impl<S: RealScalar> DeMap<S> {
    pub(super) fn finite(interval: Interval<S>) -> Result<Option<Self>, SciOxideError> {
        let geometry = IntervalGeometry::prepare(interval)?.ok_or_else(|| {
            SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "finite transform requires a non-degenerate interval",
            )
        })?;
        // A rounded half-width is needed for the scalar weight formula; an interval so narrow
        // that it underflows, or whose midpoint rounds onto an endpoint, has no interior DE
        // nodes and is reported as a geometry failure by the caller.
        let half_width = geometry.half_span()?.round()?;
        let center = geometry.point_at_fraction(S::HALF)?;
        if half_width == S::ZERO || center <= interval.lower() || center >= interval.upper() {
            return Ok(None);
        }
        Ok(Some(Self::Finite {
            geometry,
            center,
            half_width,
            pi_half: S::PI * S::HALF,
        }))
    }

    pub(super) fn semi_infinite(lower: S) -> Self {
        Self::SemiInfinite {
            lower,
            pi_half: S::PI * S::HALF,
        }
    }

    pub(super) fn infinite() -> Self {
        Self::Infinite {
            pi_half: S::PI * S::HALF,
        }
    }

    pub(super) fn center(self) -> Result<WeightedPoint<S>, SciOxideError> {
        // At t = 0 every transform has sinh(t) = 0 and cosh(t) = 1, so the Jacobians reduce
        // to `(b-a)/2 * pi/2` (tanh-sinh), `pi/2 * exp(0)` (exp-sinh, point a + 1) and
        // `pi/2 * cosh(0)` (sinh-sinh, point 0).
        let (point, weight) = match self {
            Self::Finite {
                center,
                half_width,
                pi_half,
                ..
            } => (center, half_width * pi_half),
            Self::SemiInfinite { lower, pi_half } => (
                Scaled::from_value(lower)?
                    .add(Scaled::from_value(S::ONE)?)?
                    .round()?,
                pi_half,
            ),
            Self::Infinite { pi_half } => (S::ZERO, pi_half),
        };
        if !point.is_finite() || !weight.is_finite() || weight <= S::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "the transformed center is not finite and usable",
            ));
        }
        let endpoint = match self {
            Self::Finite {
                geometry,
                half_width,
                ..
            } => Some((geometry.lower(), half_width)),
            Self::SemiInfinite { lower, .. } => Some((lower, S::ONE)),
            Self::Infinite { .. } => None,
        };
        Ok(WeightedPoint {
            point,
            weight,
            endpoint,
        })
    }

    #[inline(always)]
    fn mapped_pair(self, node: Node<S>) -> PairPreparation<S> {
        let (near, far, near_weight, far_weight, endpoints) = match self {
            Self::Finite {
                geometry,
                half_width,
                ..
            } => {
                let offset = match geometry.native_span() {
                    Some(span) => span * node.near,
                    None => match geometry
                        .span()
                        .and_then(|s| s.mul(Scaled::from_value(node.near)?))
                        .and_then(Scaled::round)
                    {
                        Ok(offset) => offset,
                        Err(_) => return PairPreparation::Saturated,
                    },
                };
                let weight = half_width * node.near_weight;
                (
                    geometry.lower() + offset,
                    geometry.upper() - offset,
                    weight,
                    weight,
                    Some(((geometry.lower(), offset), (geometry.upper(), offset))),
                )
            }
            Self::SemiInfinite { lower, .. } => (
                lower + node.near,
                lower + node.far,
                node.near_weight,
                node.far_weight,
                Some(((lower, node.near), (lower, node.far))),
            ),
            Self::Infinite { .. } => (
                -node.near,
                node.near,
                node.near_weight,
                node.near_weight,
                None,
            ),
        };
        PairPreparation::Complete(PreparedPair {
            negative: Some(WeightedPoint {
                point: near,
                weight: near_weight,
                endpoint: endpoints.map(|p| p.0),
            }),
            positive: Some(WeightedPoint {
                point: far,
                weight: far_weight,
                endpoint: endpoints.map(|p| p.1),
            }),
        })
    }
    fn pair(self, node: Node<S>) -> PairPreparation<S> {
        let PairPreparation::Complete(pair) = self.mapped_pair(node) else {
            return PairPreparation::Saturated;
        };
        let [Some(near), Some(far)] = [pair.negative, pair.positive] else {
            return PairPreparation::Saturated;
        };
        if !near.point.is_finite()
            || !far.point.is_finite()
            || !near.weight.is_finite()
            || !far.weight.is_finite()
            || near.weight <= S::ZERO
            || far.weight <= S::ZERO
            || near
                .endpoint
                .is_some_and(|(_, distance)| distance <= S::ZERO)
            || far
                .endpoint
                .is_some_and(|(_, distance)| distance <= S::ZERO)
            || (!matches!(self, Self::Finite { .. }) && far.point <= near.point)
        {
            return PairPreparation::Saturated;
        }
        PairPreparation::Complete(pair)
    }

    #[cfg(test)]
    fn direct_pair(self, t: S) -> PairPreparation<S> {
        if t == S::ZERO {
            return match self.center() {
                Ok(center) => PairPreparation::Complete(PreparedPair {
                    negative: Some(center),
                    positive: Some(center),
                }),
                Err(_) => PairPreparation::Saturated,
            };
        }

        let two = S::TWO;
        let (sinh_t, cosh_t) = (t.sinh(), t.cosh());
        if !sinh_t.is_finite() || !cosh_t.is_finite() {
            return PairPreparation::Saturated;
        }

        match self {
            Self::Finite {
                geometry,
                half_width,
                pi_half,
                ..
            } => {
                // x = c + (b-a)/2 tanh(u), u = pi/2 sinh t. Near the endpoints tanh(u) -> 1 and
                // `1 - tanh(u)` would cancel catastrophically, so the distance to the endpoint
                // is formed directly from e = exp(-2u):
                //     1 - tanh(u) = 2e / (1 + e),   dx/dt = (b-a)/2 * pi/2 cosh t * 4e/(1+e)^2.
                // Retain the offset and place coordinates directly from each endpoint.
                // The callback uses the offset if the coordinate rounds onto that endpoint.
                // e == 0 means the offset underflowed: the transform has saturated.
                let u = pi_half * sinh_t;
                let e = (-two * u).exp();
                if !u.is_finite() || !e.is_finite() || e == S::ZERO {
                    return PairPreparation::Saturated;
                }
                let denominator = S::ONE + e;
                // Avoid subnormal exponentials: their relative precision no longer
                // supports trustworthy endpoint offsets or Jacobians. The omitted
                // shell remains unresolved unless earlier shells estimate its mass.
                if e < S::MIN_POSITIVE {
                    return PairPreparation::Saturated;
                }
                let fraction = e / denominator;
                let offset = match geometry.native_span() {
                    Some(span) => span * fraction,
                    None => match geometry
                        .span()
                        .and_then(|span| span.mul(Scaled::from_value(fraction)?))
                        .and_then(Scaled::round)
                    {
                        Ok(offset) => offset,
                        Err(_) => return PairPreparation::Saturated,
                    },
                };
                let weight = half_width * S::TAU * cosh_t * e / (denominator * denominator);
                if !weight.is_finite()
                    || weight <= S::ZERO
                    || !offset.is_finite()
                    || offset <= S::ZERO
                {
                    return PairPreparation::Saturated;
                }
                let negative = Some(WeightedPoint {
                    point: geometry.lower() + offset,
                    weight,
                    endpoint: Some((geometry.lower(), offset)),
                });
                let positive = Some(WeightedPoint {
                    point: geometry.upper() - offset,
                    weight,
                    endpoint: Some((geometry.upper(), offset)),
                });
                PairPreparation::Complete(PreparedPair { negative, positive })
            }
            Self::SemiInfinite { lower, pi_half } => {
                // x = a + exp(pi/2 sinh t): the -t sample approaches `a` (near) and the +t
                // sample runs to infinity (far), with dx/dt = pi/2 cosh t * exp(+-u).
                let u = pi_half * sinh_t;
                let far = u.exp();
                let near = (-u).exp();
                let base_weight = pi_half * cosh_t;
                let near_point = lower + near;
                let far_point = lower + far;
                let near_weight = base_weight * near;
                let far_weight = base_weight * far;
                if !u.is_finite()
                    || !far.is_finite()
                    || !near.is_finite()
                    || near == S::ZERO
                    || !near_point.is_finite()
                    || !far_point.is_finite()
                    || far_point <= near_point
                    || !near_weight.is_finite()
                    || !far_weight.is_finite()
                    || near_weight <= S::ZERO
                    || far_weight <= S::ZERO
                {
                    return PairPreparation::Saturated;
                }
                PairPreparation::Complete(PreparedPair {
                    negative: Some(WeightedPoint {
                        point: near_point,
                        weight: near_weight,
                        endpoint: Some((lower, near)),
                    }),
                    positive: Some(WeightedPoint {
                        point: far_point,
                        weight: far_weight,
                        endpoint: Some((lower, far)),
                    }),
                })
            }
            Self::Infinite { pi_half } => {
                // x = sinh(u), dx/dt = pi/2 cosh t cosh u; sinh/cosh are assembled from one
                // pair of exponentials so both sides are exact mirror images.
                let u = pi_half * sinh_t;
                let exponential = u.exp();
                let reciprocal = (-u).exp();
                if !u.is_finite() || !exponential.is_finite() || !reciprocal.is_finite() {
                    return PairPreparation::Saturated;
                }
                let x = (exponential - reciprocal) * S::HALF;
                let cosh_u = (exponential + reciprocal) * S::HALF;
                let weight = pi_half * cosh_t * cosh_u;
                if !x.is_finite() || x <= S::ZERO || !weight.is_finite() || weight <= S::ZERO {
                    return PairPreparation::Saturated;
                }
                PairPreparation::Complete(PreparedPair {
                    negative: Some(WeightedPoint {
                        point: -x,
                        weight,
                        endpoint: None,
                    }),
                    positive: Some(WeightedPoint {
                        point: x,
                        weight,
                        endpoint: None,
                    }),
                })
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn check<S: RealScalar>(levels: usize) {
        let domains = [
            DoubleExponentialDomain::Finite {
                interval: Interval::new(-S::ONE, S::ONE).unwrap(),
                singular_points: &[],
            },
            DoubleExponentialDomain::HalfLine { lower: -S::ONE },
            DoubleExponentialDomain::RealLine,
        ];
        for domain in domains {
            let basis = BasisTable::new(levels, domain).unwrap();
            let map = match domain {
                DoubleExponentialDomain::Finite { interval, .. } => {
                    DeMap::finite(interval).unwrap().unwrap()
                }
                DoubleExponentialDomain::HalfLine { lower } => DeMap::semi_infinite(lower),
                DoubleExponentialDomain::RealLine => DeMap::infinite(),
            };
            assert!(basis.values.len() <= MAX_TAIL_SHELLS * basis.grid + 1);
            for index in (0..=MAX_TAIL_SHELLS * basis.grid).step_by((basis.grid / 32).max(1)) {
                let t = S::try_from_usize_exact(index).unwrap()
                    / S::try_from_usize_exact(basis.grid).unwrap();
                match (basis.pair(map, index), map.direct_pair(t)) {
                    (PairPreparation::Complete(a), PairPreparation::Complete(b)) => {
                        for (a, b) in [a.negative.zip(b.negative), a.positive.zip(b.positive)]
                            .into_iter()
                            .flatten()
                        {
                            assert!(
                                (a.point - b.point).abs() <= S::EPSILON * a.point.abs().max(S::ONE)
                            );
                            assert!(
                                (a.weight - b.weight).abs()
                                    <= S::from_f64_ieee(8.0) * S::EPSILON * a.weight.abs()
                            );
                            assert_eq!(a.endpoint, b.endpoint);
                        }
                    }
                    (PairPreparation::Saturated, PairPreparation::Saturated) => {}
                    _ => panic!("prepared/direct saturation disagreement"),
                }
            }
        }
    }

    #[test]
    fn canonical_rules_match_direct_formulas_at_every_supported_precision_and_level_bound() {
        for levels in [3, 9, 15] {
            check::<f32>(levels);
            check::<f64>(levels);
        }
    }
}
